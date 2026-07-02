import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_tiny_price_model import predict_model  # noqa: E402


SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
SOURCE_VENUE = os.getenv("SNAPSHOT_SOURCE_VENUE", os.getenv("PRIMARY_VENUE", "kraken")).strip().lower()
TRAIN_VENUE = os.getenv("SNAPSHOT_TRAIN_VENUE", f"{SOURCE_VENUE}_train_snapshot").strip().lower()
HOLDOUT_VENUE = os.getenv("SNAPSHOT_HOLDOUT_VENUE", f"{SOURCE_VENUE}_holdout_snapshot").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

FEATURE_GROUPS = os.getenv("PRICE_TINY_FEATURE_GROUPS", "base_tiny_price_v1").strip()
EXPECTED_SCHEMA_HASH = os.getenv("PRICE_TINY_EXPECTED_FEATURE_SCHEMA_HASH", "543c07fec8e33baf").strip()
MOVE_TARGET_SPEC = os.getenv("PRICE_TINY_MOVE_TARGET_SPEC", "move_before_adverse_30s_net_aware").strip()
INSTABILITY_TARGET_SPEC = os.getenv("PRICE_TINY_INSTABILITY_TARGET_SPEC", "instability_30s").strip()
MOVE_THRESHOLD = float(os.getenv("PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD", "0.70"))
INSTABILITY_THRESHOLD = float(os.getenv("PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD", "0.70"))
ALLOWED_SIDES = os.getenv("PRICE_TINY_ENSEMBLE_ALLOWED_SIDES", "long").strip().lower()
MODEL_SPECS = os.getenv("PRICE_TINY_MODEL_SPECS", "ridge_logistic").strip()
TRAIN_ALLOWED_MODEL_TYPES = os.getenv("PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES", "ridge_logistic").strip()
MAX_TRAIN_ROWS = os.getenv("PRICE_TINY_MAX_TRAIN_ROWS", "0").strip()

if ALLOWED_SIDES not in {"long", "short", "both"}:
    raise SystemExit("PRICE_TINY_ENSEMBLE_ALLOWED_SIDES must be one of: long, short, both")


def npm_command():
    return "npm.cmd" if os.name == "nt" else "npm"


def iso_time(timestamp_ms):
    if timestamp_ms is None or not np.isfinite(float(timestamp_ms)):
        return ""
    return datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def run_command(command, env_overrides, title):
    env = os.environ.copy()
    env.update({key: str(value) for key, value in env_overrides.items() if value is not None})
    env["PROMOTE_BEST"] = "false"
    env["PRICE_TINY_AUTO_REGISTER_CHALLENGERS"] = "false"
    print(f"\n=== {title} ===")
    print(" ".join(command))
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout)
    if completed.returncode != 0:
        raise SystemExit(f"{title} failed with exit code {completed.returncode}")
    return completed.stdout


def run_npm(script, env_overrides, title):
    return run_command([npm_command(), "run", script], env_overrides, title)


def load_latest_training_metadata(venue):
    path = OUTPUT_DIR / venue / f"{SYMBOL}_tiny_price_training_rows_latest.json"
    if not path.exists():
        raise SystemExit(f"Missing tiny-price training metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(venue, target_spec, title):
    env = {
        "SYMBOL": SYMBOL,
        "PRIMARY_VENUE": venue,
        "PRICE_TINY_FEATURE_GROUPS": FEATURE_GROUPS,
        "PRICE_TINY_TARGET_SPEC": target_spec,
        "PRICE_TINY_MODEL_SPECS": MODEL_SPECS,
        "PROMOTE_BEST": "false",
        "PRICE_TINY_AUTO_REGISTER_CHALLENGERS": "false",
    }
    run_npm("tiny-price-build", env, title)
    metadata = load_latest_training_metadata(venue)
    rows_path = Path(metadata.get("training_rows_path", ""))
    if not rows_path.is_absolute():
        rows_path = PROJECT_ROOT / rows_path
    if not rows_path.exists():
        raise SystemExit(f"Built training rows path is missing: {rows_path}")
    return metadata, rows_path


def train_model(venue, target_spec, rows_path, title):
    safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", target_spec)
    prediction_dir = OUTPUT_DIR / venue / "snapshot_holdout_forward_test_predictions"
    env = {
        "SYMBOL": SYMBOL,
        "PRIMARY_VENUE": venue,
        "PRICE_TINY_TRAINING_ROWS_PATH": str(rows_path),
        "PRICE_TINY_FEATURE_GROUPS": FEATURE_GROUPS,
        "PRICE_TINY_TARGET_SPEC": target_spec,
        "PRICE_TINY_MODEL_SPECS": MODEL_SPECS,
        "PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES": TRAIN_ALLOWED_MODEL_TYPES,
        "PRICE_TINY_ALLOWED_SHADOW_MODEL_TYPES": TRAIN_ALLOWED_MODEL_TYPES,
        "PRICE_TINY_MAX_TRAIN_ROWS": MAX_TRAIN_ROWS,
        "PRICE_TINY_FORWARD_TEST_PREDICTIONS_PATH": str(
            prediction_dir / f"{SYMBOL}_{safe_target}_forward_test_predictions.csv"
        ),
        "PRICE_TINY_FORWARD_TEST_PREDICTION_ARCHIVE_DIR": str(prediction_dir),
        "PROMOTE_BEST": "false",
        "PRICE_TINY_AUTO_REGISTER_CHALLENGERS": "false",
        "TRAIN_PRICE_TINY_MODEL": "false",
    }
    output = run_npm("tiny-price-train", env, title)
    matches = re.findall(r"candidate_model_path=(.+)", output)
    if not matches:
        raise SystemExit(f"Could not find candidate_model_path in {title} output.")
    model_path = Path(matches[-1].strip())
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        raise SystemExit(f"Candidate model file was not written: {model_path}")
    return model_path


def load_artifact(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_features(frame, artifact):
    feature_columns = artifact.get("feature_columns", [])
    missing = [column for column in feature_columns if column not in frame.columns]
    schema_mismatch = bool(EXPECTED_SCHEMA_HASH and artifact.get("feature_schema_hash") != EXPECTED_SCHEMA_HASH)
    if missing:
        return None, missing, schema_mismatch, np.zeros(len(frame), dtype=bool)
    raw = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(raw.to_numpy(dtype=np.float64)).all(axis=1)
    feature_mean = np.asarray(artifact.get("feature_mean", [0.0] * len(feature_columns)), dtype=np.float64)
    feature_std = np.asarray(artifact.get("feature_std", [1.0] * len(feature_columns)), dtype=np.float64)
    feature_std = np.where(np.abs(feature_std) < 1e-12, 1.0, feature_std)
    x = (raw.to_numpy(dtype=np.float64) - feature_mean) / feature_std
    return x, missing, schema_mismatch, finite_mask


def score_artifact(frame, artifact, model_path, prefix):
    x, missing, schema_mismatch, finite_mask = normalized_features(frame, artifact)
    result = {
        f"{prefix}_model_path": str(model_path),
        f"{prefix}_model_id": artifact.get("model_id", ""),
        f"{prefix}_schema_hash": artifact.get("feature_schema_hash", ""),
        f"{prefix}_feature_groups": ",".join(artifact.get("feature_spec", {}).get("enabled_feature_groups", [])),
        f"{prefix}_missing_features": ",".join(missing),
        f"{prefix}_schema_mismatch": schema_mismatch,
    }
    if x is None:
        return frame.assign(**result), np.zeros(len(frame), dtype=bool), result
    selected_model_name = artifact.get("selected_model_name") or artifact.get("model_type")
    model = artifact.get("models", {}).get(selected_model_name) or artifact.get("models", {}).get(artifact.get("model_type", ""))
    if model is None:
        raise SystemExit(f"Artifact does not contain selected model '{selected_model_name}': {model_path}")
    pred_delta, pred_log, direction, confidence, probs = predict_model(
        selected_model_name,
        model,
        x,
        float(artifact.get("delta_target_mean", 0.0)),
        float(artifact.get("delta_target_std", 1.0) or 1.0),
    )
    scored = frame.copy()
    scored[f"{prefix}_direction"] = direction
    scored[f"{prefix}_confidence"] = confidence
    scored[f"{prefix}_predicted_return_bps"] = pred_delta
    scored[f"{prefix}_predicted_log_return"] = pred_log
    if probs is not None and np.ndim(probs) == 2 and probs.shape[1] >= 3:
        scored[f"{prefix}_prob_down"] = probs[:, 0]
        scored[f"{prefix}_prob_neutral"] = probs[:, 1]
        scored[f"{prefix}_prob_up"] = probs[:, 2]
    else:
        scored[f"{prefix}_prob_down"] = np.nan
        scored[f"{prefix}_prob_neutral"] = np.nan
        scored[f"{prefix}_prob_up"] = np.nan
    for key, value in result.items():
        scored[key] = value
    return scored, finite_mask, result


def first_existing_column(frame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def summarize_predictions(scored, finite_mask, move_diag, instability_diag):
    realized_return_column = first_existing_column(
        scored,
        [
            "target_net_aware_realized_return_bps_30s",
            "target_return_bps_30s",
            "target_next_mid_delta_bps_30s",
        ],
    )
    mfe_column = first_existing_column(
        scored,
        ["target_net_aware_max_favorable_excursion_bps_30s", "target_max_favorable_excursion_bps_30s"],
    )
    mae_column = first_existing_column(
        scored,
        ["target_net_aware_max_adverse_excursion_bps_30s", "target_max_adverse_excursion_bps_30s"],
    )
    cost_column = first_existing_column(
        scored,
        ["target_net_aware_estimated_spread_cost_bps_30s", "target_net_aware_estimated_spread_cost_bps"],
    )
    if not realized_return_column:
        raise SystemExit("Holdout rows do not contain a realized 30s return target column.")

    realized = pd.to_numeric(scored[realized_return_column], errors="coerce")
    cost = pd.to_numeric(scored[cost_column], errors="coerce").fillna(0.0) if cost_column else pd.Series(0.0, index=scored.index)
    move_direction = pd.to_numeric(scored["move_direction"], errors="coerce").fillna(0).astype(int)
    move_confidence = pd.to_numeric(scored["move_confidence"], errors="coerce")
    instability_probability = pd.to_numeric(scored["instability_prob_up"], errors="coerce")

    valid = finite_mask & realized.notna().to_numpy() & move_confidence.notna().to_numpy() & instability_probability.notna().to_numpy()
    required_schema_mismatch = bool(move_diag.get("move_schema_mismatch")) or bool(instability_diag.get("instability_schema_mismatch"))
    if required_schema_mismatch:
        valid = np.zeros(len(scored), dtype=bool)

    threshold_passed = (move_confidence >= MOVE_THRESHOLD) & (instability_probability < INSTABILITY_THRESHOLD)
    side_passed = pd.Series(True, index=scored.index)
    if ALLOWED_SIDES == "long":
        side_passed = move_direction > 0
    elif ALLOWED_SIDES == "short":
        side_passed = move_direction < 0
    else:
        side_passed = move_direction != 0
    signal_mask = valid & threshold_passed.to_numpy() & side_passed.to_numpy()
    paper_side = np.where(signal_mask & (move_direction.to_numpy() > 0), "long", "")
    paper_side = np.where(signal_mask & (move_direction.to_numpy() < 0), "short", paper_side)
    gross = np.where(move_direction.to_numpy() > 0, realized.to_numpy(), -realized.to_numpy())
    net = gross - cost.to_numpy()
    scored["holdout_valid_for_scoring"] = valid
    scored["threshold_passed"] = threshold_passed
    scored["side_passed"] = side_passed
    scored["paper_signal"] = paper_side
    scored["gross_strategy_return_bps"] = np.where(signal_mask, gross, np.nan)
    scored["net_strategy_return_bps"] = np.where(signal_mask, net, np.nan)
    scored["estimated_cost_bps"] = np.where(signal_mask, cost.to_numpy(), np.nan)

    signals = scored[signal_mask].copy()
    summary = {
        "symbol": SYMBOL,
        "source_venue": SOURCE_VENUE,
        "train_venue": TRAIN_VENUE,
        "holdout_venue": HOLDOUT_VENUE,
        "feature_groups": FEATURE_GROUPS,
        "expected_feature_schema_hash": EXPECTED_SCHEMA_HASH,
        "move_target_spec": MOVE_TARGET_SPEC,
        "instability_target_spec": INSTABILITY_TARGET_SPEC,
        "move_threshold": MOVE_THRESHOLD,
        "instability_threshold": INSTABILITY_THRESHOLD,
        "allowed_sides": ALLOWED_SIDES,
        "rows": int(len(scored)),
        "evaluated_rows": int(valid.sum()),
        "signals": int(len(signals)),
        "long_signals": int((signals["paper_signal"] == "long").sum()) if len(signals) else 0,
        "short_signals": int((signals["paper_signal"] == "short").sum()) if len(signals) else 0,
        "sign_acc": float((signals["gross_strategy_return_bps"] > 0).mean()) if len(signals) else np.nan,
        "gross_bps": float(signals["gross_strategy_return_bps"].mean()) if len(signals) else np.nan,
        "net_bps": float(signals["net_strategy_return_bps"].mean()) if len(signals) else np.nan,
        "net_positive_rate": float((signals["net_strategy_return_bps"] > 0).mean()) if len(signals) else np.nan,
        "average_mfe_bps": float(pd.to_numeric(signals[mfe_column], errors="coerce").mean()) if len(signals) and mfe_column else np.nan,
        "average_mae_bps": float(pd.to_numeric(signals[mae_column], errors="coerce").mean()) if len(signals) and mae_column else np.nan,
        "stale_rows": 0,
        "missing_future_rows": int((~realized.notna()).sum()),
        "missing_or_nonfinite_feature_rows": int((~finite_mask).sum()),
        "required_schema_mismatch_rows": int(len(scored) if required_schema_mismatch else 0),
        "move_schema_mismatch": bool(move_diag.get("move_schema_mismatch")),
        "instability_schema_mismatch": bool(instability_diag.get("instability_schema_mismatch")),
        "move_missing_feature_count": len([v for v in move_diag.get("move_missing_features", "").split(",") if v]),
        "instability_missing_feature_count": len([v for v in instability_diag.get("instability_missing_features", "").split(",") if v]),
        "realized_return_column": realized_return_column,
        "mfe_column": mfe_column or "",
        "mae_column": mae_column or "",
        "cost_column": cost_column or "",
        "holdout_start_timestamp": int(scored["timestamp"].min()) if len(scored) and "timestamp" in scored.columns else np.nan,
        "holdout_end_timestamp": int(scored["timestamp"].max()) if len(scored) and "timestamp" in scored.columns else np.nan,
    }
    return scored, summary


def main():
    print("Tiny-price chronological snapshot holdout evaluator")
    print(f"SYMBOL: {SYMBOL}")
    print(f"SOURCE_VENUE: {SOURCE_VENUE}")
    print(f"TRAIN_VENUE: {TRAIN_VENUE}")
    print(f"HOLDOUT_VENUE: {HOLDOUT_VENUE}")
    print(f"Feature groups: {FEATURE_GROUPS}")
    print(f"Expected schema hash: {EXPECTED_SCHEMA_HASH}")
    print("Paper-only. No private API. No orders. No promotion.")

    run_command(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "split_realtime_snapshots.py")],
        {
            "SYMBOL": SYMBOL,
            "SNAPSHOT_SOURCE_VENUE": SOURCE_VENUE,
            "SNAPSHOT_TRAIN_VENUE": TRAIN_VENUE,
            "SNAPSHOT_HOLDOUT_VENUE": HOLDOUT_VENUE,
        },
        "split chronological snapshots",
    )

    move_train_metadata, move_train_rows = build_rows(TRAIN_VENUE, MOVE_TARGET_SPEC, "build train move rows")
    move_model_path = train_model(TRAIN_VENUE, MOVE_TARGET_SPEC, move_train_rows, "train move model on train snapshot")
    instability_train_metadata, instability_train_rows = build_rows(
        TRAIN_VENUE, INSTABILITY_TARGET_SPEC, "build train instability rows"
    )
    instability_model_path = train_model(
        TRAIN_VENUE,
        INSTABILITY_TARGET_SPEC,
        instability_train_rows,
        "train instability model on train snapshot",
    )

    holdout_metadata, holdout_rows_path = build_rows(HOLDOUT_VENUE, MOVE_TARGET_SPEC, "build holdout scoring rows")
    holdout = read_csv(holdout_rows_path)
    if len(holdout) == 0:
        raise SystemExit(f"Holdout rows are empty: {holdout_rows_path}")

    move_artifact = load_artifact(move_model_path)
    instability_artifact = load_artifact(instability_model_path)
    scored, move_finite, move_diag = score_artifact(holdout, move_artifact, move_model_path, "move")
    scored, instability_finite, instability_diag = score_artifact(scored, instability_artifact, instability_model_path, "instability")
    combined_finite = move_finite & instability_finite
    scored, summary = summarize_predictions(scored, combined_finite, move_diag, instability_diag)

    output_dir = OUTPUT_DIR / HOLDOUT_VENUE
    predictions_path = output_dir / f"{SYMBOL}_tiny_price_snapshot_holdout_predictions.csv"
    summary_path = output_dir / f"{SYMBOL}_tiny_price_snapshot_holdout_summary.csv"
    config_path = output_dir / f"{SYMBOL}_tiny_price_snapshot_holdout_config.json"
    atomic_write_csv(scored, predictions_path)
    atomic_write_csv(pd.DataFrame([summary]), summary_path)
    atomic_write_json(
        {
            "summary": summary,
            "move_model_path": str(move_model_path),
            "instability_model_path": str(instability_model_path),
            "move_train_rows_path": str(move_train_rows),
            "instability_train_rows_path": str(instability_train_rows),
            "holdout_rows_path": str(holdout_rows_path),
            "move_train_metadata": move_train_metadata,
            "instability_train_metadata": instability_train_metadata,
            "holdout_metadata": holdout_metadata,
            "paper_only": True,
        },
        config_path,
    )

    print("\nHoldout evaluation complete.")
    print(f"Move model path: {move_model_path}")
    print(f"Instability model path: {instability_model_path}")
    print(f"Holdout rows path: {holdout_rows_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")
    print(f"Config: {config_path}")
    print(f"Holdout period: {iso_time(summary['holdout_start_timestamp'])} -> {iso_time(summary['holdout_end_timestamp'])}")
    print(f"rows={summary['rows']} evaluated={summary['evaluated_rows']} signals={summary['signals']}")
    print(f"long={summary['long_signals']} short={summary['short_signals']}")
    if np.isfinite(summary["sign_acc"]):
        print(f"sign_acc={summary['sign_acc']:.2%}")
        print(f"gross_bps={summary['gross_bps']:.4f}")
        print(f"net_bps={summary['net_bps']:.4f}")
        print(f"net_positive_rate={summary['net_positive_rate']:.2%}")
        print(f"average_mfe_bps={summary['average_mfe_bps']:.4f}")
        print(f"average_mae_bps={summary['average_mae_bps']:.4f}")
    else:
        print("No holdout paper signals passed the configured gate.")
    print(f"stale_rows={summary['stale_rows']}")
    print(f"missing_future_rows={summary['missing_future_rows']}")
    print(f"required_schema_mismatch_rows={summary['required_schema_mismatch_rows']}")
    print("Paper-only. No private API. No orders. No promotion.")


if __name__ == "__main__":
    main()
