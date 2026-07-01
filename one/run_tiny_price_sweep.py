import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
TRAINING_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_training_rows.csv"
RESULTS_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_sweep_results.csv"
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "tiny_price" / (PRIMARY_VENUE or "legacy")

FEATURE_SETS = [
    value.strip()
    for value in os.getenv(
        "PRICE_TINY_SWEEP_FEATURE_SETS",
        "tiny_price_v1,tiny_price_v2,tiny_price_v3,tiny_price_v4",
    ).split(",")
    if value.strip()
]
HORIZONS = [
    int(value.strip())
    for value in os.getenv("PRICE_TINY_SWEEP_HORIZONS", "1,3,5,10").split(",")
    if value.strip()
]
LOOKBACK_PROFILES = [
    value.strip().lower()
    for value in os.getenv("PRICE_TINY_SWEEP_LOOKBACK_PROFILES", "short").split(",")
    if value.strip()
]
if not LOOKBACK_PROFILES:
    LOOKBACK_PROFILES = ["short"]
MODEL_TYPES = [
    value.strip()
    for value in os.getenv(
        "PRICE_TINY_SWEEP_MODEL_TYPES",
        "zero_return_baseline,previous_return_baseline,ridge_regression,logistic_regression,mlp_hidden_4,mlp_hidden_8,mlp_hidden_16,mlp_hidden_20",
    ).split(",")
    if value.strip()
]
HIDDEN_SIZES = sorted(
    {
        int(model_type.rsplit("_", 1)[1])
        for model_type in MODEL_TYPES
        if model_type.startswith("mlp_hidden_") and model_type.rsplit("_", 1)[1].isdigit()
    }
)
if not HIDDEN_SIZES:
    HIDDEN_SIZES = [4, 8, 16, 20]

REBUILD_ROWS_MODE = os.getenv("PRICE_TINY_SWEEP_REBUILD_ROWS", "auto").strip().lower()
RUN_EVALUATE = os.getenv("PRICE_TINY_SWEEP_RUN_EVALUATE", "true").strip().lower() in {"1", "true", "yes", "on"}
STOP_ON_ERROR = os.getenv("PRICE_TINY_SWEEP_STOP_ON_ERROR", "true").strip().lower() in {"1", "true", "yes", "on"}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, nrows=5)
    except EmptyDataError:
        return pd.DataFrame()


def run_python(script_name, env):
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)]
    print(f"Running: {script_name}")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


def training_rows_match(feature_set, horizon, lookback_profile):
    frame = read_csv(TRAINING_PATH)
    if len(frame) == 0:
        return False
    if "feature_set_name" not in frame.columns or "horizon_seconds" not in frame.columns or "lookback_profile" not in frame.columns:
        return False
    row = frame.iloc[0]
    try:
        return (
            str(row["feature_set_name"]).strip().lower() == feature_set
            and int(row["horizon_seconds"]) == int(horizon)
            and str(row["lookback_profile"]).strip().lower() == lookback_profile
        )
    except Exception:
        return False


def should_rebuild_rows(feature_set, horizon, lookback_profile):
    if REBUILD_ROWS_MODE in {"1", "true", "yes", "on", "always"}:
        return True
    if REBUILD_ROWS_MODE in {"0", "false", "no", "off", "never"}:
        return False
    return not training_rows_match(feature_set, horizon, lookback_profile)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_candidate(feature_set, horizon, lookback_profile, started_at):
    candidates = sorted(CANDIDATE_ROOT.glob("*/model.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        if path.stat().st_mtime + 2 < started_at:
            continue
        try:
            artifact = load_json(path)
        except Exception:
            continue
        if str(artifact.get("feature_set_name", "")).strip().lower() != feature_set:
            continue
        if int(artifact.get("horizon_seconds", -1)) != int(horizon):
            continue
        if str(artifact.get("lookback_profile", "short")).strip().lower() != lookback_profile:
            continue
        return path, artifact
    for path in candidates:
        try:
            artifact = load_json(path)
        except Exception:
            continue
        if (
            str(artifact.get("feature_set_name", "")).strip().lower() == feature_set
            and int(artifact.get("horizon_seconds", -1)) == int(horizon)
            and str(artifact.get("lookback_profile", "short")).strip().lower() == lookback_profile
        ):
            return path, artifact
    raise FileNotFoundError(f"No tiny price candidate found for feature_set={feature_set}, horizon={horizon}, lookback_profile={lookback_profile}")


def flatten_calibration_buckets(buckets):
    if buckets is None:
        return "[]"
    return json.dumps(buckets, separators=(",", ":"))


def result_row(feature_set, horizon, lookback_profile, model_type, report, artifact_path, artifact):
    hidden_nodes = report.get("hidden_nodes")
    best_any = report.get("best_threshold_any_rows", {})
    best_min_100 = report.get("best_threshold_min_100_rows", {})
    best_min_300 = report.get("best_threshold_min_300_rows", {})
    return {
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE or "legacy",
        "horizon_seconds": horizon,
        "lookback_profile": lookback_profile,
        "feature_set": feature_set,
        "model_type": model_type,
        "hidden_nodes": "" if hidden_nodes is None else hidden_nodes,
        "train_rows": artifact.get("training_rows", 0),
        "validation_rows": artifact.get("validation_rows", 0),
        "forward_test_rows": report.get("forward_test_rows", artifact.get("test_rows", 0)),
        "mae_bps": report.get("mae_bps"),
        "rmse_bps": report.get("rmse_bps"),
        "mae_lift_vs_zero_return_baseline": report.get("mae_lift_vs_zero_return_baseline"),
        "directional_accuracy": report.get("directional_accuracy"),
        "sign_accuracy_excluding_flat": report.get("sign_accuracy_excluding_flat"),
        "average_realized_return_when_predicted_up_bps": report.get("average_realized_bps_when_pred_up"),
        "average_realized_return_when_predicted_down_bps": report.get("average_realized_bps_when_pred_down"),
        "forward_test_avg_return_bps": report.get("forward_test_avg_return_bps"),
        "prediction_count_above_confidence_threshold": report.get("prediction_count_above_confidence_threshold"),
        "calibration_buckets": flatten_calibration_buckets(report.get("calibration_buckets")),
        "confidence_threshold_directional_report": flatten_calibration_buckets(report.get("confidence_threshold_directional_report")),
        "best_threshold_any_rows": flatten_calibration_buckets(best_any),
        "best_threshold_min_100_rows": flatten_calibration_buckets(best_min_100),
        "best_threshold_min_300_rows": flatten_calibration_buckets(best_min_300),
        "best_threshold_any_rows_threshold": best_any.get("threshold"),
        "best_threshold_any_rows_kept": best_any.get("rows_kept"),
        "best_threshold_any_rows_avg_return_bps": best_any.get("avg_realized_return_bps"),
        "best_threshold_min_100_threshold": best_min_100.get("threshold"),
        "best_threshold_min_100_rows_kept": best_min_100.get("rows_kept"),
        "best_threshold_min_100_avg_return_bps": best_min_100.get("avg_realized_return_bps"),
        "best_threshold_min_300_threshold": best_min_300.get("threshold"),
        "best_threshold_min_300_rows_kept": best_min_300.get("rows_kept"),
        "best_threshold_min_300_avg_return_bps": best_min_300.get("avg_realized_return_bps"),
        "price_candidate_useful": bool(report.get("price_candidate_useful", False)),
        "direction_candidate_useful": bool(report.get("direction_candidate_useful", False)),
        "candidate_useful": bool(report.get("candidate_useful", False)),
        "calibration_inverted": bool(report.get("calibration_inverted", False)),
        "warnings": " | ".join(report.get("warnings", [])),
        "pred_delta_std_bps": report.get("pred_delta_std_bps"),
        "prediction_distribution_collapsed": bool(report.get("prediction_distribution_collapsed", False)),
        "zero_return_baseline_mae_bps": report.get("zero_return_baseline_mae_bps"),
        "zero_return_baseline_rmse_bps": report.get("zero_return_baseline_rmse_bps"),
        "majority_direction_baseline_accuracy": report.get("majority_direction_baseline_accuracy"),
        "majority_direction_baseline_sign_accuracy_excluding_flat": report.get("majority_direction_baseline_sign_accuracy_excluding_flat"),
        "selection_objective": artifact.get("selection_objective", ""),
        "selected_model_name": artifact.get("selected_model_name", ""),
        "selected_candidate_path": str(artifact_path),
    }


def write_results(rows):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "primary_venue",
        "horizon_seconds",
        "lookback_profile",
        "feature_set",
        "model_type",
        "hidden_nodes",
        "train_rows",
        "validation_rows",
        "forward_test_rows",
        "mae_bps",
        "rmse_bps",
        "mae_lift_vs_zero_return_baseline",
        "directional_accuracy",
        "sign_accuracy_excluding_flat",
        "average_realized_return_when_predicted_up_bps",
        "average_realized_return_when_predicted_down_bps",
        "forward_test_avg_return_bps",
        "prediction_count_above_confidence_threshold",
        "calibration_buckets",
        "confidence_threshold_directional_report",
        "best_threshold_any_rows",
        "best_threshold_min_100_rows",
        "best_threshold_min_300_rows",
        "best_threshold_any_rows_threshold",
        "best_threshold_any_rows_kept",
        "best_threshold_any_rows_avg_return_bps",
        "best_threshold_min_100_threshold",
        "best_threshold_min_100_rows_kept",
        "best_threshold_min_100_avg_return_bps",
        "best_threshold_min_300_threshold",
        "best_threshold_min_300_rows_kept",
        "best_threshold_min_300_avg_return_bps",
        "price_candidate_useful",
        "direction_candidate_useful",
        "candidate_useful",
        "calibration_inverted",
        "warnings",
        "pred_delta_std_bps",
        "prediction_distribution_collapsed",
        "zero_return_baseline_mae_bps",
        "zero_return_baseline_rmse_bps",
        "majority_direction_baseline_accuracy",
        "majority_direction_baseline_sign_accuracy_excluding_flat",
        "selection_objective",
        "selected_model_name",
        "selected_candidate_path",
    ]
    tmp_path = RESULTS_PATH.with_suffix(RESULTS_PATH.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(RESULTS_PATH)


def number(value, default=float("-inf")):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def complexity(row):
    model_type = row.get("model_type", "")
    order = {
        "zero_return_baseline": 0,
        "previous_return_baseline": 1,
        "ridge_regression": 2,
        "logistic_regression": 2,
    }
    if model_type in order:
        return order[model_type]
    if model_type.startswith("mlp_hidden_"):
        try:
            return 10 + int(model_type.rsplit("_", 1)[1])
        except Exception:
            return 999
    return 999


def print_table(title, rows, key, reverse=True, limit=8):
    print("")
    print(title)
    if not rows:
        print("- no rows")
        return
    sorted_rows = sorted(rows, key=lambda row: number(row.get(key)), reverse=reverse)[:limit]
    for row in sorted_rows:
        print(
            f"- {row['feature_set']} h={row['horizon_seconds']}s lookback={row.get('lookback_profile', 'short')} {row['model_type']}: "
            f"{key}={number(row.get(key), 0.0):.6f}, "
            f"mae={number(row.get('mae_bps'), 0.0):.4f}bps, "
            f"dir_acc={number(row.get('directional_accuracy'), 0.0):.2%}, "
            f"useful={row.get('candidate_useful')}"
        )


def print_summary(rows):
    print_table("Top results by MAE lift vs zero-return baseline", rows, "mae_lift_vs_zero_return_baseline")
    print_table("Top results by directional accuracy", rows, "directional_accuracy")
    print_table("Top results by forward-test average return", rows, "forward_test_avg_return_bps")
    print("")
    print("Simplest price-useful model")
    price_useful = [row for row in rows if row.get("price_candidate_useful")]
    if not price_useful:
        print("- none found in this sweep")
    else:
        best = sorted(
            price_useful,
            key=lambda row: (
                complexity(row),
                -number(row.get("mae_lift_vs_zero_return_baseline"), 0.0),
                -number(row.get("directional_accuracy"), 0.0),
            ),
        )[0]
        print(
            f"- {best['feature_set']} h={best['horizon_seconds']}s lookback={best.get('lookback_profile', 'short')} {best['model_type']} "
            f"mae_lift={number(best.get('mae_lift_vs_zero_return_baseline'), 0.0):.4f}bps "
            f"dir_acc={number(best.get('directional_accuracy'), 0.0):.2%}"
        )
    print("")
    print("Simplest direction-useful model")
    direction_useful = [row for row in rows if row.get("direction_candidate_useful")]
    if not direction_useful:
        print("- none found in this sweep")
        return
    best = sorted(
        direction_useful,
        key=lambda row: (
            complexity(row),
            -number(row.get("forward_test_avg_return_bps"), 0.0),
            -number(row.get("directional_accuracy"), 0.0),
        ),
    )[0]
    print(
        f"- {best['feature_set']} h={best['horizon_seconds']}s lookback={best.get('lookback_profile', 'short')} {best['model_type']} "
        f"avg_return={number(best.get('forward_test_avg_return_bps'), 0.0):.4f}bps "
        f"dir_acc={number(best.get('directional_accuracy'), 0.0):.2%} "
        f"mae_lift={number(best.get('mae_lift_vs_zero_return_baseline'), 0.0):.4f}bps"
    )


def main():
    print("Tiny price sweep runner")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Feature sets: {FEATURE_SETS}")
    print(f"Horizons: {HORIZONS}")
    print(f"Lookback profiles: {LOOKBACK_PROFILES}")
    print(f"Model types: {MODEL_TYPES}")
    print("Chronological splits are handled by train_tiny_price_model.py. No promotion. Paper-only.")
    rows = []
    for feature_set in FEATURE_SETS:
        for horizon in HORIZONS:
            for lookback_profile in LOOKBACK_PROFILES:
                print("")
                print(f"=== Sweep run: feature_set={feature_set}, horizon={horizon}s, lookback_profile={lookback_profile} ===")
                child_env = os.environ.copy()
                child_env["SYMBOL"] = SYMBOL
                child_env["PRIMARY_VENUE"] = PRIMARY_VENUE
                child_env["PRICE_TINY_FEATURE_SET"] = feature_set
                child_env["PRICE_TINY_HORIZON_SECONDS"] = str(horizon)
                child_env["PRICE_TINY_LOOKBACK_PROFILE"] = lookback_profile
                child_env["PRICE_TINY_HIDDEN_SIZES"] = ",".join(str(value) for value in HIDDEN_SIZES)
                child_env["PROMOTE_BEST"] = "false"
                if "PRICE_TINY_SWEEP_EPOCHS" in os.environ:
                    child_env["PRICE_TINY_EPOCHS"] = os.environ["PRICE_TINY_SWEEP_EPOCHS"]
                started_at = time.time()
                try:
                    if should_rebuild_rows(feature_set, horizon, lookback_profile):
                        run_python("build_tiny_price_training_rows.py", child_env)
                    else:
                        print("Training rows already match this feature set/horizon/lookback profile; skipping rebuild.")
                    run_python("train_tiny_price_model.py", child_env)
                    if RUN_EVALUATE:
                        run_python("evaluate_tiny_price_predictions.py", child_env)
                    artifact_path, artifact = latest_candidate(feature_set, horizon, lookback_profile, started_at)
                    reports = artifact.get("forward_test_reports", {})
                    if not reports:
                        raise RuntimeError(f"Candidate is missing forward_test_reports: {artifact_path}")
                    for model_type in MODEL_TYPES:
                        report = reports.get(model_type)
                        if report is None:
                            print(f"Warning: report missing for {model_type}; skipping.")
                            continue
                        rows.append(result_row(feature_set, horizon, lookback_profile, model_type, report, artifact_path, artifact))
                    write_results(rows)
                    print(f"Saved partial sweep results: {RESULTS_PATH}")
                except Exception as exc:
                    print(f"ERROR: sweep run failed for {feature_set} h={horizon}s lookback_profile={lookback_profile}: {exc}")
                    if STOP_ON_ERROR:
                        raise
    write_results(rows)
    print("")
    print(f"Sweep results written: {RESULTS_PATH}")
    print(f"Result rows: {len(rows)}")
    print_summary(rows)
    print("Paper-only. No trades/orders/private API. No promotion.")


if __name__ == "__main__":
    main()
