import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from evaluate_tiny_price_walk_forward import (
    prepare_frame,
    softmax,
    class_to_direction,
    target_slug,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR

MOVE_TARGET = os.getenv("PRICE_TINY_WALK_FORWARD_MOVE_TARGET", "move_before_adverse_30s").strip().lower()
INSTABILITY_TARGET = os.getenv("PRICE_TINY_WALK_FORWARD_INSTABILITY_TARGET", "instability_30s").strip().lower()

ESTIMATED_FEE_BPS = float(os.getenv("PRICE_TINY_ESTIMATED_FEE_BPS", "0"))
ESTIMATED_SLIPPAGE_BPS = float(os.getenv("PRICE_TINY_ESTIMATED_SLIPPAGE_BPS", "0"))
CHARGE_HALF_SPREAD = os.getenv("PRICE_TINY_CHARGE_HALF_SPREAD", "true").strip().lower() in {"1", "true", "yes", "y"}

RULES = [
    {"name": "move>=0.70_and_instability<0.70", "move_threshold": 0.70, "instability_threshold": 0.70},
    {"name": "move>=0.65_and_instability<0.70", "move_threshold": 0.65, "instability_threshold": 0.70},
    {"name": "move>=0.60_and_instability<0.70", "move_threshold": 0.60, "instability_threshold": 0.70},
    {"name": "move>=0.60_and_instability<0.75", "move_threshold": 0.60, "instability_threshold": 0.75},
    {"name": "move>=0.55_and_instability<0.70", "move_threshold": 0.55, "instability_threshold": 0.70},
]

SIDE_REGIME_DIAGNOSTICS_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_walk_forward_ensemble_side_regime_diagnostics.csv"
CONTEXT_COLUMNS = [
    "feature_spread_percent",
    "feature_rolling_volatility_30s",
    "feature_rolling_volatility_60s",
    "feature_rolling_volatility_120s",
    "feature_recent_high_low_range_30s",
    "feature_recent_high_low_range_60s",
    "feature_recent_high_low_range_120s",
    "feature_imbalance10",
    "feature_imbalance25",
    "feature_is_asia_session",
    "feature_is_london_session",
    "feature_is_us_cash_session",
]


def read_csv(path, nrows=None):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, nrows=nrows)
    except EmptyDataError:
        return pd.DataFrame()


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def output_path_for(target_spec, summary=False):
    prefix = f"{SYMBOL}_tiny_price_walk_forward"
    if summary:
        prefix += "_summary"
    return VENUE_DIR / f"{prefix}_{target_slug(target_spec)}.csv"


def require_file(path, help_text):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing required file: {path}\n{help_text}")


def first_value(frame, column, default=""):
    if column not in frame.columns or len(frame) == 0:
        return default
    values = frame[column].dropna()
    if len(values) == 0:
        return default
    return values.iloc[0]


def load_target_walk_forward(target_spec):
    rows_path = output_path_for(target_spec)
    summary_path = output_path_for(target_spec, summary=True)
    help_text = (
        "Run the base walk-forward evaluator first, for example:\n"
        '$env:PRICE_TINY_WALK_FORWARD_TARGET_SPECS="move_before_adverse_30s,instability_30s"\n'
        "npm run tiny-price-walk-forward-evaluate"
    )
    require_file(rows_path, help_text)
    require_file(summary_path, help_text)
    walk_rows = read_csv(rows_path)
    summary = read_csv(summary_path)
    if len(walk_rows) == 0 or len(summary) == 0:
        raise RuntimeError(f"Walk-forward output is empty for {target_spec}")
    training_path = Path(str(first_value(summary, "training_rows_path", "")))
    if not training_path.is_absolute():
        training_path = PROJECT_ROOT / training_path
    require_file(training_path, f"Training rows referenced by {summary_path} are missing.")
    raw = read_csv(training_path)
    frame, feature_columns, selected_col, realized_col, method, horizon = prepare_frame(raw, target_spec)
    return {
        "target_spec": target_spec,
        "walk_rows": walk_rows,
        "summary": summary,
        "training_path": training_path,
        "frame": frame,
        "feature_columns": feature_columns,
        "selected_col": selected_col,
        "realized_col": realized_col,
        "method": method,
        "horizon": horizon,
    }


def load_artifact(path):
    artifact_path = Path(str(path))
    if not artifact_path.is_absolute():
        artifact_path = PROJECT_ROOT / artifact_path
    require_file(artifact_path, "Saved walk-forward model artifact is missing.")
    with artifact_path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    return artifact_path, artifact


def predict_logistic(artifact, frame):
    feature_columns = artifact["feature_columns"]
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Artifact feature columns missing from test frame: {missing[:10]}")
    model = artifact["model"]
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = np.asarray(model["bias"], dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    std = np.where(std < 1e-9, 1.0, std)
    x_raw = frame[feature_columns].to_numpy(dtype=np.float64)
    x = (x_raw - mean) / std
    probs = softmax(x @ weights + bias)
    pred_class = np.argmax(probs, axis=1)
    return probs, class_to_direction(pred_class), probs.max(axis=1)


def test_slice(frame, window_row):
    start_ts = int(pd.to_numeric(pd.Series([window_row["test_start_timestamp"]]), errors="coerce").iloc[0])
    end_ts = int(pd.to_numeric(pd.Series([window_row["test_end_timestamp"]]), errors="coerce").iloc[0])
    mask = (frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts)
    return frame.loc[mask].copy()


def target_column(frame, name):
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    return np.full(len(frame), np.nan, dtype=np.float64)


def replay_target_predictions(target_data, role):
    output = []
    for _, window_row in target_data["walk_rows"].iterrows():
        artifact_path, artifact = load_artifact(window_row["walk_forward_artifact_path"])
        frame = test_slice(target_data["frame"], window_row)
        if len(frame) == 0:
            continue
        probs, pred_direction, confidence = predict_logistic(artifact, frame)
        horizon = int(target_data["horizon"])
        realized_return = target_column(frame, f"target_return_bps_{horizon}s")
        mfe = target_column(frame, f"target_max_favorable_excursion_bps_{horizon}s")
        mae = target_column(frame, f"target_max_adverse_excursion_bps_{horizon}s")
        spread = target_column(frame, "feature_spread_percent")
        for row_index, (_, source_row) in enumerate(frame.iterrows()):
            row = {
                "window_index": int(window_row["window_index"]),
                "timestamp": int(source_row["timestamp"]),
                "time": source_row.get("time", ""),
                f"{role}_artifact_path": str(artifact_path),
                f"{role}_model_name": artifact.get("model_name", ""),
                f"{role}_feature_schema_hash": artifact.get("feature_schema_hash", ""),
                "actual_return_bps": float(realized_return[row_index]),
                "target_max_favorable_excursion_bps": float(mfe[row_index]),
                "target_max_adverse_excursion_bps": float(mae[row_index]),
                "feature_spread_percent": float(spread[row_index]) if np.isfinite(spread[row_index]) else np.nan,
            }
            for column in CONTEXT_COLUMNS:
                if column in source_row.index and column not in row:
                    value = pd.to_numeric(pd.Series([source_row.get(column, np.nan)]), errors="coerce").iloc[0]
                    row[column] = float(value) if np.isfinite(value) else np.nan
            if role == "move":
                row["move_direction"] = int(pred_direction[row_index])
                row["move_confidence"] = float(confidence[row_index])
                row["move_prob_down"] = float(probs[row_index, 0])
                row["move_prob_neutral"] = float(probs[row_index, 1])
                row["move_prob_up"] = float(probs[row_index, 2])
            else:
                # Instability is trained as 0/1, and the existing 3-class helper maps
                # event=1 to the "up" class. Use that event probability directly.
                row["instability_probability"] = float(probs[row_index, 2])
                row["instability_confidence"] = float(confidence[row_index])
                row["instability_prob_no_event"] = float(probs[row_index, 1])
            output.append(row)
    return pd.DataFrame(output)


def estimated_cost_bps(rows):
    base_cost = ESTIMATED_FEE_BPS + ESTIMATED_SLIPPAGE_BPS
    if not CHARGE_HALF_SPREAD or "feature_spread_percent" not in rows.columns:
        return np.full(len(rows), base_cost, dtype=np.float64)
    spread_ratio = pd.to_numeric(rows["feature_spread_percent"], errors="coerce").to_numpy(dtype=np.float64)
    # Recorder/training feature stores spread as a ratio, so bps = ratio * 10000.
    spread_bps = np.where(np.isfinite(spread_ratio) & (spread_ratio >= 0), spread_ratio * 10000.0, 0.0)
    return spread_bps + base_cost


def estimated_spread_cost_bps(rows):
    if not CHARGE_HALF_SPREAD or "feature_spread_percent" not in rows.columns:
        return np.zeros(len(rows), dtype=np.float64)
    spread_ratio = pd.to_numeric(rows["feature_spread_percent"], errors="coerce").to_numpy(dtype=np.float64)
    return np.where(np.isfinite(spread_ratio) & (spread_ratio >= 0), spread_ratio * 10000.0, 0.0)


def side_specific_mfe_mae(rows, direction):
    long_mfe = pd.to_numeric(rows["target_max_favorable_excursion_bps"], errors="coerce").to_numpy(dtype=np.float64)
    long_mae = pd.to_numeric(rows["target_max_adverse_excursion_bps"], errors="coerce").to_numpy(dtype=np.float64)
    direction = np.asarray(direction, dtype=np.int64)
    strategy_mfe = np.where(direction > 0, long_mfe, -long_mae)
    strategy_mae = np.where(direction > 0, long_mae, -long_mfe)
    return strategy_mfe, strategy_mae


def average(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.mean()) if values.notna().any() else np.nan


def bucket_fixed(values, bins, labels, missing_label="missing"):
    values = pd.to_numeric(values, errors="coerce")
    output = pd.Series(missing_label, index=values.index, dtype="object")
    finite = values.notna()
    if finite.any():
        output.loc[finite] = pd.cut(values.loc[finite], bins=bins, labels=labels, include_lowest=True).astype("object")
    return output.fillna(missing_label).astype(str)


def bucket_quantile(values, label_prefix):
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    output = pd.Series("missing", index=values.index, dtype="object")
    finite = values.notna()
    unique_count = values.loc[finite].nunique()
    if unique_count >= 4:
        labels = [f"{label_prefix}_p00_p25", f"{label_prefix}_p25_p50", f"{label_prefix}_p50_p75", f"{label_prefix}_p75_p100"]
        try:
            output.loc[finite] = pd.qcut(values.loc[finite], q=4, labels=labels, duplicates="drop").astype("object")
        except ValueError:
            output.loc[finite] = f"{label_prefix}_single"
    elif finite.any():
        output.loc[finite] = f"{label_prefix}_single"
    return output.fillna("missing").astype(str)


def first_available_numeric(frame, columns):
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def session_bucket(frame):
    if {"feature_is_asia_session", "feature_is_london_session", "feature_is_us_cash_session"}.intersection(frame.columns):
        asia = first_available_numeric(frame, ["feature_is_asia_session"]).fillna(0) > 0
        london = first_available_numeric(frame, ["feature_is_london_session"]).fillna(0) > 0
        us_cash = first_available_numeric(frame, ["feature_is_us_cash_session"]).fillna(0) > 0
        output = pd.Series("other_session", index=frame.index, dtype="object")
        output.loc[asia] = "asia_session"
        output.loc[london] = "london_session"
        output.loc[us_cash] = "us_cash_session"
        return output
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
    hours = pd.to_datetime(timestamps, unit="ms", utc=True, errors="coerce").dt.hour
    output = pd.Series("unknown", index=frame.index, dtype="object")
    output.loc[(hours >= 0) & (hours < 7)] = "asia_utc_00_06"
    output.loc[(hours >= 7) & (hours < 13)] = "london_utc_07_12"
    output.loc[(hours >= 13) & (hours < 21)] = "us_utc_13_20"
    output.loc[(hours >= 21) & (hours < 24)] = "late_us_utc_21_23"
    return output


def add_bucket_columns(frame):
    frame = frame.copy()
    frame["bucket_spread_percentile"] = bucket_quantile(first_available_numeric(frame, ["feature_spread_percent"]), "spread")
    frame["bucket_volatility_range"] = bucket_quantile(
        first_available_numeric(
            frame,
            [
                "feature_rolling_volatility_60s",
                "feature_rolling_volatility_30s",
                "feature_rolling_volatility_120s",
                "feature_recent_high_low_range_60s",
                "feature_recent_high_low_range_30s",
                "feature_recent_high_low_range_120s",
            ],
        ),
        "vol_range",
    )
    frame["bucket_instability_probability"] = bucket_fixed(
        pd.to_numeric(frame["instability_probability"], errors="coerce"),
        [-np.inf, 0.50, 0.60, 0.70, 0.80, np.inf],
        ["instability_<0.50", "instability_0.50_0.60", "instability_0.60_0.70", "instability_0.70_0.80", "instability_>=0.80"],
    )
    frame["bucket_move_confidence"] = bucket_fixed(
        pd.to_numeric(frame["move_confidence"], errors="coerce"),
        [-np.inf, 0.55, 0.60, 0.65, 0.70, np.inf],
        ["move_conf_<0.55", "move_conf_0.55_0.60", "move_conf_0.60_0.65", "move_conf_0.65_0.70", "move_conf_>=0.70"],
    )
    frame["bucket_hour_session"] = session_bucket(frame)
    frame["bucket_bid_ask_imbalance"] = bucket_fixed(
        first_available_numeric(frame, ["feature_imbalance10", "feature_imbalance25"]),
        [-np.inf, -0.20, -0.05, 0.05, 0.20, np.inf],
        ["ask_heavy", "mild_ask_heavy", "balanced", "mild_bid_heavy", "bid_heavy"],
    )
    return frame


def side_metric_row(frame, side_label, meta, bucket_type="all", bucket_value="all"):
    direction_value = 1 if side_label == "long" else -1
    if len(frame):
        rows = frame[pd.to_numeric(frame["diagnostic_direction"], errors="coerce").fillna(0).astype(int) == direction_value].copy()
    else:
        rows = frame
    if len(rows) == 0:
        return {
            **meta,
            "bucket_type": bucket_type,
            "bucket_value": bucket_value,
            "side": side_label,
            "rows": 0,
            "sign_accuracy": np.nan,
            "gross_avg_return_bps": np.nan,
            "net_avg_return_bps": np.nan,
            "average_mfe_bps": np.nan,
            "average_mae_bps": np.nan,
            "average_spread_cost_bps": np.nan,
            "net_positive_rate": np.nan,
            "paper_only": True,
            "no_promotion": True,
        }
    actual_return = pd.to_numeric(rows["actual_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
    gross = pd.to_numeric(rows["diagnostic_gross_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
    net = pd.to_numeric(rows["diagnostic_net_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
    sign_accuracy = float((np.sign(actual_return) == np.sign(direction_value)).mean()) if len(rows) else np.nan
    return {
        **meta,
        "bucket_type": bucket_type,
        "bucket_value": bucket_value,
        "side": side_label,
        "rows": int(len(rows)),
        "sign_accuracy": sign_accuracy,
        "gross_avg_return_bps": average(gross),
        "net_avg_return_bps": average(net),
        "average_mfe_bps": average(rows["diagnostic_mfe_bps"]),
        "average_mae_bps": average(rows["diagnostic_mae_bps"]),
        "average_spread_cost_bps": average(rows["diagnostic_spread_cost_bps"]),
        "net_positive_rate": float(np.nanmean(net > 0)) if np.isfinite(net).any() else np.nan,
        "paper_only": True,
        "no_promotion": True,
    }


def add_diagnostic_strategy_columns(frame, direction):
    frame = frame.copy()
    direction = np.asarray(direction, dtype=np.int64)
    actual_return = pd.to_numeric(frame["actual_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
    total_cost = estimated_cost_bps(frame)
    spread_cost = estimated_spread_cost_bps(frame)
    mfe, mae = side_specific_mfe_mae(frame, direction)
    frame["diagnostic_direction"] = direction
    frame["diagnostic_gross_return_bps"] = actual_return * direction
    frame["diagnostic_net_return_bps"] = frame["diagnostic_gross_return_bps"] - total_cost
    frame["diagnostic_spread_cost_bps"] = spread_cost
    frame["diagnostic_mfe_bps"] = mfe
    frame["diagnostic_mae_bps"] = mae
    return frame


def append_side_bucket_rows(rows, frame, meta):
    if len(frame) == 0:
        for side in ["long", "short"]:
            rows.append(side_metric_row(frame, side, meta))
        return
    frame = add_bucket_columns(frame)
    bucket_columns = [
        ("all", None),
        ("spread_percentile_bucket", "bucket_spread_percentile"),
        ("volatility_range_bucket", "bucket_volatility_range"),
        ("instability_probability_bucket", "bucket_instability_probability"),
        ("move_confidence_bucket", "bucket_move_confidence"),
        ("hour_session_bucket", "bucket_hour_session"),
        ("bid_ask_imbalance_bucket", "bucket_bid_ask_imbalance"),
    ]
    for bucket_type, column in bucket_columns:
        if column is None:
            for side in ["long", "short"]:
                rows.append(side_metric_row(frame, side, meta, "all", "all"))
            continue
        if column not in frame.columns:
            continue
        for bucket_value, bucket_frame in frame.groupby(column, dropna=False, sort=True):
            for side in ["long", "short"]:
                rows.append(side_metric_row(bucket_frame, side, meta, bucket_type, str(bucket_value)))


def evaluate_rule(window_rows, rule):
    if len(window_rows) == 0:
        return {
            "rows_kept": 0,
            "long_count": 0,
            "short_count": 0,
            "sign_accuracy": np.nan,
            "gross_avg_strategy_return_bps": np.nan,
            "estimated_net_avg_strategy_return_bps": np.nan,
            "average_mfe_bps": np.nan,
            "average_mae_bps": np.nan,
            "average_spread_cost_bps": np.nan,
            "net_positive_rate": np.nan,
        }
    direction = pd.to_numeric(window_rows["move_direction"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    move_confidence = pd.to_numeric(window_rows["move_confidence"], errors="coerce").to_numpy(dtype=np.float64)
    instability = pd.to_numeric(window_rows["instability_probability"], errors="coerce").to_numpy(dtype=np.float64)
    keep = (
        (direction != 0)
        & np.isfinite(move_confidence)
        & np.isfinite(instability)
        & (move_confidence >= rule["move_threshold"])
        & (instability < rule["instability_threshold"])
    )
    kept = window_rows.loc[keep].copy()
    kept_direction = direction[keep]
    if len(kept) == 0:
        return {
            "rows_kept": 0,
            "long_count": 0,
            "short_count": 0,
            "sign_accuracy": np.nan,
            "gross_avg_strategy_return_bps": np.nan,
            "estimated_net_avg_strategy_return_bps": np.nan,
            "average_mfe_bps": np.nan,
            "average_mae_bps": np.nan,
            "average_spread_cost_bps": np.nan,
            "net_positive_rate": np.nan,
        }
    actual_return = pd.to_numeric(kept["actual_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
    gross_return = actual_return * kept_direction
    net_return = gross_return - estimated_cost_bps(kept)
    spread_cost = estimated_spread_cost_bps(kept)
    finite = np.isfinite(actual_return)
    sign_accuracy = float((np.sign(actual_return[finite]) == np.sign(kept_direction[finite])).mean()) if finite.any() else np.nan
    strategy_mfe, strategy_mae = side_specific_mfe_mae(kept, kept_direction)
    return {
        "rows_kept": int(len(kept)),
        "long_count": int((kept_direction > 0).sum()),
        "short_count": int((kept_direction < 0).sum()),
        "sign_accuracy": sign_accuracy,
        "gross_avg_strategy_return_bps": average(gross_return),
        "estimated_net_avg_strategy_return_bps": average(net_return),
        "average_mfe_bps": average(strategy_mfe),
        "average_mae_bps": average(strategy_mae),
        "average_spread_cost_bps": average(spread_cost),
        "net_positive_rate": float(np.nanmean(net_return > 0)) if np.isfinite(net_return).any() else np.nan,
    }


def build_summary(detail_rows):
    if not detail_rows:
        return []
    frame = pd.DataFrame(detail_rows)
    summary_rows = []
    for rule_name, group in frame.groupby("rule_name", sort=False):
        rows_kept = pd.to_numeric(group["rows_kept"], errors="coerce").fillna(0)
        gross = pd.to_numeric(group["gross_avg_strategy_return_bps"], errors="coerce")
        net = pd.to_numeric(group["estimated_net_avg_strategy_return_bps"], errors="coerce")
        active_group = group.loc[rows_kept > 0].copy()
        positive_windows = int((gross > 0).sum()) if gross.notna().any() else 0
        negative_windows = int((gross < 0).sum()) if gross.notna().any() else 0
        summary_rows.append(
            {
                "symbol": SYMBOL,
                "primary_venue": PRIMARY_VENUE,
                "rule_name": rule_name,
                "move_threshold": float(first_value(group, "move_threshold", np.nan)),
                "instability_threshold": float(first_value(group, "instability_threshold", np.nan)),
                "windows": int(len(group)),
                "active_windows": int((rows_kept > 0).sum()),
                "total_rows_kept": int(rows_kept.sum()),
                "total_long_count": int(pd.to_numeric(group["long_count"], errors="coerce").fillna(0).sum()),
                "total_short_count": int(pd.to_numeric(group["short_count"], errors="coerce").fillna(0).sum()),
                "mean_sign_accuracy": average(active_group["sign_accuracy"]) if len(active_group) else np.nan,
                "mean_gross_avg_strategy_return_bps": average(active_group["gross_avg_strategy_return_bps"]) if len(active_group) else np.nan,
                "mean_estimated_net_avg_strategy_return_bps": average(active_group["estimated_net_avg_strategy_return_bps"]) if len(active_group) else np.nan,
                "worst_window_avg_return_bps": float(gross.min()) if gross.notna().any() else np.nan,
                "worst_window_estimated_net_avg_return_bps": float(net.min()) if net.notna().any() else np.nan,
                "positive_windows": positive_windows,
                "negative_windows": negative_windows,
                "average_mfe_bps": average(active_group["average_mfe_bps"]) if len(active_group) else np.nan,
                "average_mae_bps": average(active_group["average_mae_bps"]) if len(active_group) else np.nan,
                "estimated_fee_bps": ESTIMATED_FEE_BPS,
                "estimated_slippage_bps": ESTIMATED_SLIPPAGE_BPS,
                "charge_half_spread": CHARGE_HALF_SPREAD,
                "paper_only": True,
                "no_promotion": True,
            }
        )
    summary_by_rule = {row["rule_name"]: row for row in summary_rows}
    for row in detail_rows:
        summary = summary_by_rule[row["rule_name"]]
        row["worst_window_avg_return_bps"] = summary["worst_window_avg_return_bps"]
        row["worst_window_estimated_net_avg_return_bps"] = summary["worst_window_estimated_net_avg_return_bps"]
        row["positive_windows"] = summary["positive_windows"]
        row["negative_windows"] = summary["negative_windows"]
    return summary_rows


def main():
    print("Tiny-price walk-forward ensemble evaluator")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Move target: {MOVE_TARGET}")
    print(f"Instability target: {INSTABILITY_TARGET}")
    print(
        "Cost assumptions: "
        f"fee={ESTIMATED_FEE_BPS:.4f}bps "
        f"slippage={ESTIMATED_SLIPPAGE_BPS:.4f}bps "
        f"charge_half_spread={CHARGE_HALF_SPREAD}"
    )

    move_data = load_target_walk_forward(MOVE_TARGET)
    instability_data = load_target_walk_forward(INSTABILITY_TARGET)
    if int(move_data["horizon"]) != int(instability_data["horizon"]):
        raise RuntimeError(f"Target horizons do not match: {move_data['horizon']} vs {instability_data['horizon']}")

    move_predictions = replay_target_predictions(move_data, "move")
    instability_predictions = replay_target_predictions(instability_data, "instability")
    if len(move_predictions) == 0 or len(instability_predictions) == 0:
        raise RuntimeError("No replayed walk-forward predictions were produced.")

    merged = move_predictions.merge(
        instability_predictions[
            [
                "window_index",
                "timestamp",
                "instability_probability",
                "instability_confidence",
                "instability_prob_no_event",
                "instability_artifact_path",
                "instability_model_name",
                "instability_feature_schema_hash",
            ]
        ],
        on=["window_index", "timestamp"],
        how="inner",
    )
    if len(merged) == 0:
        raise RuntimeError("Move and instability walk-forward outputs have no shared test timestamps.")

    detail_rows = []
    diagnostic_rows = []
    for window_index, window_rows in merged.groupby("window_index", sort=True):
        for rule in RULES:
            metrics = evaluate_rule(window_rows, rule)
            direction = pd.to_numeric(window_rows["move_direction"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
            move_confidence = pd.to_numeric(window_rows["move_confidence"], errors="coerce").to_numpy(dtype=np.float64)
            instability = pd.to_numeric(window_rows["instability_probability"], errors="coerce").to_numpy(dtype=np.float64)
            keep = (
                (direction != 0)
                & np.isfinite(move_confidence)
                & np.isfinite(instability)
                & (move_confidence >= rule["move_threshold"])
                & (instability < rule["instability_threshold"])
            )
            kept_rows = window_rows.loc[keep].copy()
            kept_rows = add_diagnostic_strategy_columns(kept_rows, direction[keep]) if len(kept_rows) else kept_rows
            append_side_bucket_rows(
                diagnostic_rows,
                kept_rows,
                {
                    "symbol": SYMBOL,
                    "primary_venue": PRIMARY_VENUE,
                    "diagnostic_source": "walk_forward_rule",
                    "window_index": int(window_index),
                    "test_start_timestamp": int(window_rows["timestamp"].min()),
                    "test_end_timestamp": int(window_rows["timestamp"].max()),
                    "rule_name": rule["name"],
                    "move_threshold": rule["move_threshold"],
                    "instability_threshold": rule["instability_threshold"],
                    "move_target": MOVE_TARGET,
                    "instability_target": INSTABILITY_TARGET,
                },
            )
            detail_rows.append(
                {
                    "symbol": SYMBOL,
                    "primary_venue": PRIMARY_VENUE,
                    "window_index": int(window_index),
                    "window_rows": int(len(window_rows)),
                    "test_start_timestamp": int(window_rows["timestamp"].min()),
                    "test_end_timestamp": int(window_rows["timestamp"].max()),
                    "rule_name": rule["name"],
                    "move_threshold": rule["move_threshold"],
                    "instability_threshold": rule["instability_threshold"],
                    **metrics,
                    "move_target": MOVE_TARGET,
                    "instability_target": INSTABILITY_TARGET,
                    "move_artifact_path": str(first_value(window_rows, "move_artifact_path", "")),
                    "instability_artifact_path": str(first_value(window_rows, "instability_artifact_path", "")),
                    "paper_only": True,
                    "no_promotion": True,
                }
            )

    summary_rows = build_summary(detail_rows)
    detail_path = VENUE_DIR / f"{SYMBOL}_tiny_price_walk_forward_ensemble.csv"
    summary_path = VENUE_DIR / f"{SYMBOL}_tiny_price_walk_forward_ensemble_summary.csv"
    write_csv(detail_rows, detail_path)
    write_csv(summary_rows, summary_path)
    write_csv(diagnostic_rows, SIDE_REGIME_DIAGNOSTICS_PATH)

    print(f"Shared replay rows: {len(merged)}")
    print(f"Windows evaluated: {len(set(merged['window_index']))}")
    print(f"Rules evaluated: {len(RULES)}")
    print(f"Detail output: {detail_path}")
    print(f"Summary output: {summary_path}")
    print(f"Side/regime diagnostics output: {SIDE_REGIME_DIAGNOSTICS_PATH}")
    print("")
    print("Summary by rule")
    for row in summary_rows:
        print(
            f"- {row['rule_name']}: kept={row['total_rows_kept']} "
            f"mean_gross={row['mean_gross_avg_strategy_return_bps']:.4f}bps "
            f"mean_net={row['mean_estimated_net_avg_strategy_return_bps']:.4f}bps "
            f"positive_windows={row['positive_windows']} negative_windows={row['negative_windows']}"
        )
    print("Research only. No challenger registration, promotion, live prediction writes, orders, or private API use.")


if __name__ == "__main__":
    main()
