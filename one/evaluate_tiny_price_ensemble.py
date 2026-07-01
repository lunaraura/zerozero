import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
TARGET_SPECS = [
    value.strip()
    for value in os.getenv(
        "PRICE_TINY_ENSEMBLE_TARGET_SPECS",
        "direction_30s,move_before_adverse_30s,instability_30s,return_bps_30s",
    ).split(",")
    if value.strip()
]
MOVE_CONFIDENCE_THRESHOLD = float(os.getenv("PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE", "0.70"))
DIRECTION_CONFIDENCE_THRESHOLD = float(os.getenv("PRICE_TINY_ENSEMBLE_DIRECTION_CONFIDENCE", "0.70"))
INSTABILITY_THRESHOLDS = [
    float(value.strip())
    for value in os.getenv("PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLDS", "0.60,0.70").split(",")
    if value.strip()
]
REGRESSION_ABS_THRESHOLDS_BPS = [
    float(value.strip())
    for value in os.getenv("PRICE_TINY_ENSEMBLE_REGRESSION_ABS_BPS", "1,2,5").split(",")
    if value.strip()
]
FEATURE_SCHEMA_HASH_FILTER = os.getenv("PRICE_TINY_ENSEMBLE_FEATURE_SCHEMA_HASH", "").strip()

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
PREDICTION_DIR = Path(
    os.getenv("PRICE_TINY_FORWARD_TEST_PREDICTION_ARCHIVE_DIR", VENUE_DIR / "tiny_price_forward_test_predictions")
)
if not PREDICTION_DIR.is_absolute():
    PREDICTION_DIR = PROJECT_ROOT / PREDICTION_DIR
OUTPUT_PATH = Path(
    os.getenv("PRICE_TINY_ENSEMBLE_EVALUATION_PATH", VENUE_DIR / f"{SYMBOL}_tiny_price_ensemble_evaluation.csv")
)
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH
DISAGREEMENT_OUTPUT_PATH = Path(
    os.getenv("PRICE_TINY_ENSEMBLE_DISAGREEMENT_PATH", VENUE_DIR / f"{SYMBOL}_tiny_price_ensemble_disagreements.csv")
)
if not DISAGREEMENT_OUTPUT_PATH.is_absolute():
    DISAGREEMENT_OUTPUT_PATH = PROJECT_ROOT / DISAGREEMENT_OUTPUT_PATH


def read_csv(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()


def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def target_key(target_spec):
    text = str(target_spec).strip().lower()
    if text.startswith("move_before_adverse"):
        return "move"
    if text.startswith("instability"):
        return "instability"
    if text.startswith("return_bps") or text.startswith("next_mid_delta_bps") or text.startswith("next_mid_log_return"):
        return "return"
    if text.startswith("direction"):
        return "direction"
    return text.replace("-", "_")


def candidate_files():
    if not PREDICTION_DIR.exists():
        return []
    return sorted(PREDICTION_DIR.glob(f"{SYMBOL}_tiny_price_forward_test_predictions__*.csv"))


def file_target_spec(path, frame):
    if "target_spec" in frame.columns and len(frame["target_spec"].dropna()):
        return str(frame["target_spec"].dropna().iloc[0]).strip()
    name = path.name.lower()
    for target_spec in TARGET_SPECS:
        if target_spec.lower() in name:
            return target_spec
    return ""


def select_latest_prediction_files():
    selected = {}
    details = []
    for path in candidate_files():
        frame = read_csv(path)
        if len(frame) == 0:
            continue
        target_spec = file_target_spec(path, frame)
        if target_spec not in TARGET_SPECS:
            continue
        if "symbol" in frame.columns and len(frame["symbol"].dropna()):
            if str(frame["symbol"].dropna().iloc[0]).upper() != SYMBOL:
                continue
        if "primary_venue" in frame.columns and len(frame["primary_venue"].dropna()):
            if str(frame["primary_venue"].dropna().iloc[0]).lower() != (PRIMARY_VENUE or "legacy"):
                continue
        if FEATURE_SCHEMA_HASH_FILTER and "feature_schema_hash" in frame.columns:
            hashes = set(str(value) for value in frame["feature_schema_hash"].dropna().unique())
            if FEATURE_SCHEMA_HASH_FILTER not in hashes:
                continue
        mtime = path.stat().st_mtime
        if target_spec not in selected or mtime > selected[target_spec]["mtime"]:
            selected[target_spec] = {"path": path, "mtime": mtime, "rows": len(frame)}
        details.append({"target_spec": target_spec, "path": str(path), "rows": len(frame), "mtime": mtime})
    return selected, details


def numeric(frame, column, default=np.nan):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def prepare_prediction_frame(path, target_spec):
    raw = read_csv(path)
    if len(raw) == 0:
        return pd.DataFrame()
    raw = raw.copy()
    key = target_key(target_spec)
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).copy()
    raw["timestamp"] = raw["timestamp"].astype("int64")
    raw["symbol"] = raw.get("symbol", SYMBOL)
    raw["primary_venue"] = raw.get("primary_venue", PRIMARY_VENUE or "legacy")
    raw["horizon_seconds"] = numeric(raw, "horizon_seconds", 30).fillna(30).astype("int64")
    if "feature_schema_hash" not in raw.columns:
        raw["feature_schema_hash"] = ""
    if FEATURE_SCHEMA_HASH_FILTER:
        raw = raw[raw["feature_schema_hash"].astype(str) == FEATURE_SCHEMA_HASH_FILTER].copy()
    if len(raw) == 0:
        return pd.DataFrame()

    reduced = raw[["timestamp", "symbol", "primary_venue", "horizon_seconds", "feature_schema_hash"]].copy()
    reduced[f"{key}_target_spec"] = target_spec
    reduced[f"{key}_prediction_path"] = str(path)
    reduced[f"{key}_model_id"] = raw.get("model_id", "")
    reduced[f"{key}_selected_model_name"] = raw.get("selected_model_name", raw.get("model_type", ""))
    reduced[f"{key}_predicted_direction"] = numeric(raw, "predicted_direction", 0).fillna(0).astype(int)
    reduced[f"{key}_confidence"] = numeric(raw, "confidence", 0.0).fillna(0.0)
    reduced[f"{key}_prob_down"] = numeric(raw, "prob_down", np.nan)
    reduced[f"{key}_prob_flat"] = numeric(raw, "prob_flat", np.nan)
    reduced[f"{key}_prob_up"] = numeric(raw, "prob_up", np.nan)
    predicted_return = numeric(raw, "predicted_return_bps", np.nan)
    if predicted_return.isna().all():
        predicted_return = numeric(raw, "predicted_next_mid_delta_bps", np.nan)
    reduced[f"{key}_predicted_return_bps"] = predicted_return
    actual_return = numeric(raw, "actual_realized_return_bps", np.nan)
    if actual_return.isna().all():
        actual_return = numeric(raw, "actual_next_mid_delta_bps", np.nan)
    reduced[f"{key}_actual_return_bps"] = actual_return
    reduced[f"{key}_actual_direction"] = numeric(raw, "actual_direction", np.nan)
    reduced[f"{key}_actual_max_favorable_excursion_bps"] = numeric(raw, "actual_max_favorable_excursion_bps", np.nan)
    reduced[f"{key}_actual_max_adverse_excursion_bps"] = numeric(raw, "actual_max_adverse_excursion_bps", np.nan)
    return reduced.drop_duplicates(["timestamp", "symbol", "primary_venue", "horizon_seconds", "feature_schema_hash"], keep="last")


def merge_prediction_frames(frames):
    if not frames:
        return pd.DataFrame()
    join_keys = ["timestamp", "symbol", "primary_venue", "horizon_seconds", "feature_schema_hash"]
    return functools.reduce(lambda left, right: left.merge(right, on=join_keys, how="inner"), frames)


def first_available(frame, columns, default=np.nan):
    output = pd.Series(default, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            output = output.where(output.notna(), values)
    return output


def add_actual_columns(frame):
    frame = frame.copy()
    frame["actual_return_bps"] = first_available(
        frame,
        [
            "direction_actual_return_bps",
            "move_actual_return_bps",
            "return_actual_return_bps",
            "instability_actual_return_bps",
        ],
    )
    frame["actual_direction_price"] = np.where(frame["actual_return_bps"] > 0, 1, np.where(frame["actual_return_bps"] < 0, -1, 0))
    frame["actual_max_favorable_excursion_bps"] = first_available(
        frame,
        [
            "direction_actual_max_favorable_excursion_bps",
            "move_actual_max_favorable_excursion_bps",
            "return_actual_max_favorable_excursion_bps",
            "instability_actual_max_favorable_excursion_bps",
        ],
    )
    frame["actual_max_adverse_excursion_bps"] = first_available(
        frame,
        [
            "direction_actual_max_adverse_excursion_bps",
            "move_actual_max_adverse_excursion_bps",
            "return_actual_max_adverse_excursion_bps",
            "instability_actual_max_adverse_excursion_bps",
        ],
    )
    return frame


def instability_probability(frame):
    if "instability_prob_up" in frame.columns and frame["instability_prob_up"].notna().any():
        return pd.to_numeric(frame["instability_prob_up"], errors="coerce").fillna(0.0)
    direction = pd.to_numeric(frame.get("instability_predicted_direction", 0), errors="coerce").fillna(0)
    confidence = pd.to_numeric(frame.get("instability_confidence", 0), errors="coerce").fillna(0.0)
    return np.where(direction > 0, confidence, 1.0 - confidence)


def metric_row(name, frame, mask, signal_direction, description="", comparison_group="rules"):
    total = len(frame)
    mask = np.asarray(mask, dtype=bool)
    signal_direction = np.asarray(signal_direction, dtype=np.int64)
    active = mask & (signal_direction != 0) & np.isfinite(frame["actual_return_bps"].to_numpy(dtype=np.float64))
    rows_kept = int(active.sum())
    actual_return = frame["actual_return_bps"].to_numpy(dtype=np.float64)
    actual_direction = frame["actual_direction_price"].to_numpy(dtype=np.int64)
    if rows_kept == 0:
        return {
            "comparison_group": comparison_group,
            "rule": name,
            "description": description,
            "rows_total": total,
            "rows_kept": 0,
            "rows_suppressed": int(total),
            "sign_accuracy": np.nan,
            "directional_win_rate": np.nan,
            "avg_strategy_return_bps": np.nan,
            "avg_realized_return_when_predicted_up_bps": np.nan,
            "avg_realized_return_when_predicted_down_bps": np.nan,
            "avg_max_favorable_excursion_bps": np.nan,
            "avg_max_adverse_excursion_bps": np.nan,
        }
    strategy_return = actual_return[active] * signal_direction[active]
    up_active = active & (signal_direction > 0)
    down_active = active & (signal_direction < 0)
    favorable = pd.Series(np.nan, index=frame.index, dtype="float64")
    adverse = pd.Series(np.nan, index=frame.index, dtype="float64")
    if "actual_max_favorable_excursion_bps" in frame.columns and "actual_max_adverse_excursion_bps" in frame.columns:
        mfe = pd.to_numeric(frame["actual_max_favorable_excursion_bps"], errors="coerce")
        mae = pd.to_numeric(frame["actual_max_adverse_excursion_bps"], errors="coerce")
        favorable = pd.Series(np.where(signal_direction > 0, mfe, np.abs(mae)), index=frame.index)
        adverse = pd.Series(np.where(signal_direction > 0, mae, -mfe), index=frame.index)
    return {
        "comparison_group": comparison_group,
        "rule": name,
        "description": description,
        "rows_total": total,
        "rows_kept": rows_kept,
        "rows_suppressed": int(total - rows_kept),
        "sign_accuracy": float((signal_direction[active] == actual_direction[active]).mean()),
        "directional_win_rate": float((strategy_return > 0).mean()),
        "avg_strategy_return_bps": float(np.mean(strategy_return)),
        "avg_realized_return_when_predicted_up_bps": float(actual_return[up_active].mean()) if up_active.any() else np.nan,
        "avg_realized_return_when_predicted_down_bps": float(actual_return[down_active].mean()) if down_active.any() else np.nan,
        "avg_max_favorable_excursion_bps": float(favorable[active].mean()) if favorable[active].notna().any() else np.nan,
        "avg_max_adverse_excursion_bps": float(adverse[active].mean()) if adverse[active].notna().any() else np.nan,
    }


def build_rule_metrics(frame):
    rows = []
    direction_dir = numeric(frame, "direction_predicted_direction", 0).fillna(0).astype(int).to_numpy()
    direction_conf = numeric(frame, "direction_confidence", 0.0).fillna(0.0).to_numpy()
    move_dir = numeric(frame, "move_predicted_direction", 0).fillna(0).astype(int).to_numpy()
    move_conf = numeric(frame, "move_confidence", 0.0).fillna(0.0).to_numpy()
    regression_return = numeric(frame, "return_predicted_return_bps", np.nan).to_numpy(dtype=np.float64)
    regression_dir = np.where(regression_return > 0, 1, np.where(regression_return < 0, -1, 0))
    instability_prob = np.asarray(instability_probability(frame), dtype=np.float64)

    direction_high = direction_conf >= DIRECTION_CONFIDENCE_THRESHOLD
    move_high = move_conf >= MOVE_CONFIDENCE_THRESHOLD
    directions_agree = (direction_dir == move_dir) & (direction_dir != 0)

    rows.append(metric_row("single_direction_any_signal", frame, direction_dir != 0, direction_dir, "direction_30s non-zero prediction", "single_model"))
    rows.append(metric_row("single_direction_high_confidence", frame, direction_high, direction_dir, f"direction confidence >= {DIRECTION_CONFIDENCE_THRESHOLD}", "single_model"))
    rows.append(metric_row("single_move_before_adverse_any_signal", frame, move_dir != 0, move_dir, "move_before_adverse non-zero prediction", "single_model"))
    rows.append(metric_row("single_move_before_adverse_high_confidence", frame, move_high, move_dir, f"move_before_adverse confidence >= {MOVE_CONFIDENCE_THRESHOLD}", "single_model"))
    for threshold in INSTABILITY_THRESHOLDS:
        low_instability = instability_prob < threshold
        rows.append(metric_row(f"single_instability_low_lt_{threshold:.2f}_direction_side", frame, low_instability, direction_dir, f"instability probability < {threshold:.2f}, using direction_30s side", "single_model"))
        rows.append(metric_row(f"single_instability_low_lt_{threshold:.2f}_move_side", frame, low_instability, move_dir, f"instability probability < {threshold:.2f}, using move_before_adverse side", "single_model"))

    rows.append(metric_row("move_before_adverse_confidence_rule", frame, move_high, move_dir, f"move_before_adverse confidence >= {MOVE_CONFIDENCE_THRESHOLD}"))
    rows.append(metric_row("direction_agrees_with_move_before_adverse", frame, directions_agree, move_dir, "direction_30s agrees with move_before_adverse direction"))
    rows.append(metric_row("move_high_and_direction_agrees", frame, move_high & directions_agree, move_dir, "move confidence high AND direction agrees"))

    for threshold in INSTABILITY_THRESHOLDS:
        low_instability = instability_prob < threshold
        rows.append(metric_row(f"move_high_direction_agrees_instability_lt_{threshold:.2f}", frame, move_high & directions_agree & low_instability, move_dir, f"move high, direction agrees, instability probability < {threshold:.2f}"))
        rows.append(metric_row(f"move_high_instability_lt_{threshold:.2f}", frame, move_high & low_instability, move_dir, f"move high, instability probability < {threshold:.2f}"))

    for threshold in REGRESSION_ABS_THRESHOLDS_BPS:
        regression_active = np.isfinite(regression_return) & (np.abs(regression_return) >= threshold)
        rows.append(metric_row(f"single_regression_abs_return_ge_{threshold:.1f}bps", frame, regression_active, regression_dir, f"abs regression predicted_return_bps >= {threshold:.1f}", "single_model"))
        regression_agrees = regression_active & (regression_dir == move_dir) & (move_dir != 0)
        rows.append(metric_row(f"move_high_direction_agrees_regression_agrees_abs_ge_{threshold:.1f}bps", frame, move_high & directions_agree & regression_agrees, move_dir, f"move high, direction agrees, regression sign agrees and abs return >= {threshold:.1f}bps"))
        for instability_threshold in INSTABILITY_THRESHOLDS:
            low_instability = instability_prob < instability_threshold
            rows.append(metric_row(f"full_filter_instability_lt_{instability_threshold:.2f}_reg_abs_ge_{threshold:.1f}bps", frame, move_high & directions_agree & regression_agrees & low_instability, move_dir, f"move high + direction agrees + instability < {instability_threshold:.2f} + regression abs >= {threshold:.1f}bps"))
    return rows


def disagreement_rows(frame):
    rows = []
    direction_dir = numeric(frame, "direction_predicted_direction", 0).fillna(0).astype(int).to_numpy()
    direction_conf = numeric(frame, "direction_confidence", 0.0).fillna(0.0).to_numpy()
    move_dir = numeric(frame, "move_predicted_direction", 0).fillna(0).astype(int).to_numpy()
    move_conf = numeric(frame, "move_confidence", 0.0).fillna(0.0).to_numpy()
    instability_prob = np.asarray(instability_probability(frame), dtype=np.float64)
    cases = [
        ("move_high_conf_direction_disagrees", (move_conf >= MOVE_CONFIDENCE_THRESHOLD) & (move_dir != 0) & (direction_dir != 0) & (move_dir != direction_dir), move_dir),
        ("direction_high_conf_move_low_conf", (direction_conf >= DIRECTION_CONFIDENCE_THRESHOLD) & (move_conf < MOVE_CONFIDENCE_THRESHOLD), direction_dir),
        ("instability_high", instability_prob >= max(INSTABILITY_THRESHOLDS or [0.70]), move_dir),
        ("instability_low", instability_prob < min(INSTABILITY_THRESHOLDS or [0.60]), move_dir),
    ]
    for name, mask, signal in cases:
        rows.append(metric_row(name, frame, mask, signal, name.replace("_", " "), "disagreement"))
    return rows


def main():
    selected, details = select_latest_prediction_files()
    print("Tiny-price multi-target ensemble evaluator")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Prediction archive dir: {PREDICTION_DIR}")
    print(f"Target specs requested: {TARGET_SPECS}")
    if FEATURE_SCHEMA_HASH_FILTER:
        print(f"Feature schema hash filter: {FEATURE_SCHEMA_HASH_FILTER}")
    print("Selected prediction files")
    for target_spec in TARGET_SPECS:
        item = selected.get(target_spec)
        if not item:
            print(f"- {target_spec}: missing")
        else:
            print(f"- {target_spec}: {item['path']} rows={item['rows']}")

    frames = []
    loaded_targets = []
    for target_spec in TARGET_SPECS:
        item = selected.get(target_spec)
        if not item:
            continue
        prepared = prepare_prediction_frame(item["path"], target_spec)
        if len(prepared) == 0:
            print(f"- {target_spec}: no compatible rows after metadata filtering")
            continue
        frames.append(prepared)
        loaded_targets.append(target_spec)

    if len(frames) < 2:
        print("Blocked: need at least two compatible target prediction files.")
        print("Run tiny-price-train once per target spec so archived prediction files exist.")
        print("Paper-only. No trades/orders/private API.")
        return

    joined = merge_prediction_frames(frames)
    joined = add_actual_columns(joined)
    joined = joined.dropna(subset=["actual_return_bps"]).reset_index(drop=True)
    print(f"Loaded targets: {loaded_targets}")
    print(f"Shared joined rows: {len(joined)}")
    if len(joined) == 0:
        print("Blocked: no shared timestamps across compatible prediction files.")
        print("Check feature_schema_hash, horizon_seconds, symbol, and primary_venue.")
        print("Paper-only. No trades/orders/private API.")
        return

    metric_frame = pd.DataFrame(build_rule_metrics(joined))
    disagreement_frame = pd.DataFrame(disagreement_rows(joined))
    atomic_write_csv(metric_frame, OUTPUT_PATH)
    atomic_write_csv(disagreement_frame, DISAGREEMENT_OUTPUT_PATH)

    print("Rule evaluation")
    for _, row in metric_frame.iterrows():
        print(
            f"- [{row['comparison_group']}] {row['rule']}: rows={int(row['rows_kept'])} "
            f"suppressed={int(row['rows_suppressed'])} "
            f"sign_acc={row['sign_accuracy']:.2%} "
            f"win={row['directional_win_rate']:.2%} "
            f"avg_strategy={row['avg_strategy_return_bps']:.4f}bps "
            f"up_return={row['avg_realized_return_when_predicted_up_bps']:.4f}bps "
            f"down_return={row['avg_realized_return_when_predicted_down_bps']:.4f}bps "
            f"mfe={row['avg_max_favorable_excursion_bps']:.4f} "
            f"mae={row['avg_max_adverse_excursion_bps']:.4f}"
        )
    print("Disagreement table")
    for _, row in disagreement_frame.iterrows():
        print(
            f"- {row['rule']}: rows={int(row['rows_kept'])} "
            f"sign_acc={row['sign_accuracy']:.2%} "
            f"win={row['directional_win_rate']:.2%} "
            f"avg_strategy={row['avg_strategy_return_bps']:.4f}bps"
        )
    print(f"Evaluation CSV: {OUTPUT_PATH}")
    print(f"Disagreement CSV: {DISAGREEMENT_OUTPUT_PATH}")
    print("Paper-only. No trades/orders/private API.")


if __name__ == "__main__":
    main()
