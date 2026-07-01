import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from live_3m_model_utils import (
    ACTUAL_LABEL_COLUMNS,
    LIVE_PREDICTION_COLUMNS,
    atomic_write_csv,
    bootstrap_prediction_from_latest_row,
    build_input_window,
    choose_feature_columns,
    coerce_feature_ready,
    coerce_numeric_columns,
    copy_file_if_exists,
    load_csv_or_empty,
    load_model,
    parse_bool,
    predict_with_model,
    prediction_values_to_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
LOOP_SECONDS = int(os.getenv("LOOP_SECONDS", "60"))
RUN_ONCE = parse_bool(os.getenv("RUN_ONCE", "false"))
LOOKBACK = int(os.getenv("LOOKBACK", "30"))
MAX_NEW_PREDICTIONS_PER_RUN = int(os.getenv("MAX_NEW_PREDICTIONS_PER_RUN", "500"))
RUN_LABELER_IN_LOOP = parse_bool(os.getenv("RUN_LABELER_IN_LOOP", "true"))
RUN_TRAINER_IN_LOOP = parse_bool(os.getenv("RUN_TRAINER_IN_LOOP", "true"))
TRAIN_EVERY_MINUTES = int(os.getenv("TRAIN_EVERY_MINUTES", "60"))

REALTIME_DIR = PROJECT_ROOT / "data" / "realtime"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "live_snapshot"
REALTIME_SOURCE_DIR = REALTIME_DIR / PRIMARY_VENUE if PRIMARY_VENUE else REALTIME_DIR
SNAPSHOT_SOURCE_DIR = SNAPSHOT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else SNAPSHOT_DIR
LIVE_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "live_predictions" / f"{SYMBOL}_live_3m_predictions.csv"
)
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "active" / f"{SYMBOL}_live_3m_model.json"
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL


def newest_candidate_model_path():
    if not CANDIDATE_ROOT.exists():
        return None
    candidates = sorted(CANDIDATE_ROOT.glob("*/model.json"))
    return candidates[-1] if candidates else None


def load_best_available_model():
    active = load_model(ACTIVE_MODEL_PATH)
    if active is not None:
        return active, ACTIVE_MODEL_PATH

    candidate_path = newest_candidate_model_path()
    if candidate_path is not None:
        candidate = load_model(candidate_path)
        if candidate is not None:
            return candidate, candidate_path

    return None, None


def snapshot_realtime_files():
    copied_raw = copy_file_if_exists(
        REALTIME_SOURCE_DIR / f"{SYMBOL}_1m_flow.csv",
        SNAPSHOT_SOURCE_DIR / f"{SYMBOL}_1m_flow.csv",
    )
    copy_file_if_exists(
        REALTIME_SOURCE_DIR / f"{SYMBOL}_10s_flow.csv",
        SNAPSHOT_SOURCE_DIR / f"{SYMBOL}_10s_flow.csv",
    )
    if not copied_raw:
        raise FileNotFoundError(
            f"Missing realtime input: {REALTIME_SOURCE_DIR / f'{SYMBOL}_1m_flow.csv'}"
        )


def rebuild_snapshot_features():
    env = os.environ.copy()
    env["SYMBOL"] = SYMBOL
    env["OUTPUT_DIR"] = str(SNAPSHOT_DIR)
    if PRIMARY_VENUE:
        env["PRIMARY_VENUE"] = PRIMARY_VENUE
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_1m_flow_features.py")],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def run_support_script(script_name):
    env = os.environ.copy()
    env["SYMBOL"] = SYMBOL
    if PRIMARY_VENUE:
        env["PRIMARY_VENUE"] = PRIMARY_VENUE
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"\n--- {script_name} ---")
    print(result.stdout.strip() if result.stdout else "(no output)")
    if result.returncode != 0:
        print(f"{script_name} exited with code {result.returncode}")
    return result.returncode == 0


def load_snapshot_features():
    path = SNAPSHOT_SOURCE_DIR / f"{SYMBOL}_1m_flow_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot feature file was not created: {path}")
    frame = pd.read_csv(path)
    frame = coerce_numeric_columns(frame)
    frame = coerce_feature_ready(frame)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return frame


def append_new_predictions(feature_frame, model_artifact, model_path):
    existing = load_csv_or_empty(LIVE_PREDICTIONS_PATH, LIVE_PREDICTION_COLUMNS)
    if "timestamp" in existing.columns and len(existing) > 0:
        existing["timestamp"] = pd.to_numeric(existing["timestamp"], errors="coerce")
    predicted_timestamps = set(
        int(value)
        for value in existing.get("timestamp", pd.Series(dtype=float)).dropna().tolist()
    )

    if model_artifact is not None:
        lookback = int(model_artifact.get("lookback", LOOKBACK))
        feature_columns = model_artifact["feature_columns"]
        target_columns = model_artifact["regression_target_columns"]
    else:
        lookback = LOOKBACK
        # Bootstrap mode does not use a learned feature vector yet. It only
        # needs a contiguous ready lookback window and the latest row's current
        # pressure/imbalance fields. Requiring every optional numeric feature
        # to be finite here can block BTC/ETH when sparse snapshot fields exist.
        feature_columns = []
        target_columns = []

    new_rows = []
    latest_prediction_before_run = (
        max(predicted_timestamps) if predicted_timestamps else None
    )
    diagnostics = {
        "candidate_rows_after_latest_prediction_timestamp": 0,
        "skipped_lookback_window_invalid": 0,
        "skipped_feature_ready_false": 0,
        "skipped_required_features_missing": 0,
        "skipped_prediction_already_exists": 0,
        "skipped_prediction_failed": 0,
    }
    blocking_reasons = {}
    missing_model_feature_columns = [
        column for column in feature_columns if column not in feature_frame.columns
    ]

    def add_blocking_reason(reason):
        blocking_reasons[reason] = blocking_reasons.get(reason, 0) + 1

    for end_index in feature_frame.index.tolist():
        timestamp = int(feature_frame.loc[end_index, "timestamp"])
        if timestamp in predicted_timestamps:
            diagnostics["skipped_prediction_already_exists"] += 1
            continue
        if latest_prediction_before_run is not None and timestamp <= latest_prediction_before_run:
            diagnostics["skipped_prediction_already_exists"] += 1
            continue

        diagnostics["candidate_rows_after_latest_prediction_timestamp"] += 1

        if not bool(feature_frame.loc[end_index, "feature_ready"]):
            diagnostics["skipped_feature_ready_false"] += 1
            add_blocking_reason("current row feature_ready=false")
            continue

        if missing_model_feature_columns:
            diagnostics["skipped_required_features_missing"] += 1
            add_blocking_reason(
                "model feature columns missing: "
                + ", ".join(missing_model_feature_columns[:10])
            )
            continue

        if model_artifact is None:
            ready_history = feature_frame.iloc[: end_index + 1]
            ready_history = ready_history[ready_history["feature_ready"]]
            if len(ready_history) < lookback:
                diagnostics["skipped_lookback_window_invalid"] += 1
                add_blocking_reason("not enough prior feature_ready rows for bootstrap LOOKBACK")
                continue
            window = np.empty((1, 0), dtype=np.float64)
            window_start = int(ready_history["timestamp"].iloc[-lookback])
            window_end = timestamp
        else:
            window, window_start, window_end = build_input_window(
                feature_frame,
                end_index,
                feature_columns,
                lookback,
            )
            if window is None:
                start_index = end_index - lookback + 1
                if start_index < 0:
                    add_blocking_reason("not enough rows for LOOKBACK")
                else:
                    candidate_window = feature_frame.iloc[start_index : end_index + 1]
                    if len(candidate_window) != lookback:
                        add_blocking_reason("lookback slice length mismatch")
                    elif "feature_ready" in candidate_window.columns and not bool(
                        candidate_window["feature_ready"].all()
                    ):
                        add_blocking_reason("lookback window crosses feature_ready=false rows")
                    elif not np.all(np.diff(candidate_window["timestamp"].to_numpy(dtype=np.int64)) == 60_000):
                        add_blocking_reason("lookback window crosses missing 1m interval")
                    elif feature_columns:
                        values = candidate_window[feature_columns].replace(
                            [np.inf, -np.inf],
                            np.nan,
                        )
                        if values.isna().any().any():
                            add_blocking_reason("required model feature values are missing/non-finite")
                        else:
                            add_blocking_reason("lookback window invalid")
                    else:
                        add_blocking_reason("lookback window invalid")
                diagnostics["skipped_lookback_window_invalid"] += 1
                continue

        try:
            if model_artifact is None:
                prediction_values = bootstrap_prediction_from_latest_row(
                    feature_frame.loc[end_index]
                )
            else:
                probabilities, regression = predict_with_model(model_artifact, window)
                prediction_values = prediction_values_to_row(
                    probabilities,
                    regression,
                    target_columns,
                    model_artifact,
                )
        except Exception as error:
            diagnostics["skipped_prediction_failed"] += 1
            add_blocking_reason(f"bootstrap/model prediction failed: {error}")
            continue

        row = {
            "timestamp": timestamp,
            "time": feature_frame.loc[end_index, "time"],
            "input_window_start": window_start,
            "input_window_end": window_end,
            **prediction_values,
            "label_ready": False,
        }
        new_rows.append(row)

    if MAX_NEW_PREDICTIONS_PER_RUN > 0:
        new_rows = new_rows[-MAX_NEW_PREDICTIONS_PER_RUN:]

    if new_rows:
        updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        updated = updated.sort_values("timestamp").drop_duplicates(
            "timestamp",
            keep="last",
        )
        output_columns = []
        for column in [*LIVE_PREDICTION_COLUMNS, *ACTUAL_LABEL_COLUMNS, *updated.columns]:
            if column not in output_columns:
                output_columns.append(column)
        for column in output_columns:
            if column not in updated.columns:
                updated[column] = np.nan
        atomic_write_csv(updated[output_columns], LIVE_PREDICTIONS_PATH)

    latest_raw_timestamp = int(feature_frame["timestamp"].max()) if len(feature_frame) else None
    latest_feature_timestamp = latest_raw_timestamp
    ready_rows = feature_frame[feature_frame["feature_ready"]].copy()
    latest_feature_ready_timestamp = (
        int(ready_rows["timestamp"].max()) if len(ready_rows) else None
    )
    latest_prediction_timestamp = None
    if LIVE_PREDICTIONS_PATH.exists():
        predictions = pd.read_csv(LIVE_PREDICTIONS_PATH)
        if len(predictions) > 0:
            latest_prediction_timestamp = int(pd.to_numeric(predictions["timestamp"], errors="coerce").max())

    print("Live 3m prediction loop tick")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Realtime input 1m path: {REALTIME_SOURCE_DIR / f'{SYMBOL}_1m_flow.csv'}")
    print(f"Snapshot feature path: {SNAPSHOT_SOURCE_DIR / f'{SYMBOL}_1m_flow_features.csv'}")
    print(f"Latest raw 1m timestamp: {latest_raw_timestamp}")
    print(f"Latest feature timestamp: {latest_feature_timestamp}")
    print(f"Latest feature_ready timestamp: {latest_feature_ready_timestamp}")
    print(f"Latest prediction timestamp: {latest_prediction_timestamp}")
    print(f"LOOKBACK: {lookback}")
    print(f"Model source: {model_path if model_path else 'bootstrap heuristic; no active/candidate model yet'}")
    print(f"Eligible feature_ready rows count: {int(feature_frame['feature_ready'].sum())}")
    print(
        "Candidate rows after latest prediction timestamp: "
        f"{diagnostics['candidate_rows_after_latest_prediction_timestamp']}"
    )
    print(
        "Rows skipped because lookback window invalid: "
        f"{diagnostics['skipped_lookback_window_invalid']}"
    )
    print(
        "Rows skipped because feature_ready=false: "
        f"{diagnostics['skipped_feature_ready_false']}"
    )
    print(
        "Rows skipped because required features missing: "
        f"{diagnostics['skipped_required_features_missing']}"
    )
    print(
        "Rows skipped because prediction already exists: "
        f"{diagnostics['skipped_prediction_already_exists']}"
    )
    print(
        "Rows skipped because bootstrap/model prediction failed: "
        f"{diagnostics['skipped_prediction_failed']}"
    )
    print(f"Final newly written predictions: {len(new_rows)}")
    if not new_rows:
        if blocking_reasons:
            print("Exact blocking reasons:")
            for reason, count in sorted(blocking_reasons.items()):
                print(f"- {reason}: {count}")
        elif len(feature_frame) == 0:
            print("Exact blocking reason: snapshot feature file has zero rows.")
        elif diagnostics["candidate_rows_after_latest_prediction_timestamp"] == 0:
            print("Exact blocking reason: no rows are newer than the latest prediction timestamp.")
        else:
            print("Exact blocking reason: no candidate rows survived the prediction filters.")
    print(f"Prediction output: {LIVE_PREDICTIONS_PATH}")
    print("No trades were placed.")


def run_once(last_training_time):
    snapshot_realtime_files()
    rebuild_snapshot_features()
    feature_frame = load_snapshot_features()
    model_artifact, model_path = load_best_available_model()
    append_new_predictions(feature_frame, model_artifact, model_path)

    if RUN_LABELER_IN_LOOP:
        run_support_script("label_live_3m_predictions.py")

    training_ran = False
    now = time.time()
    if RUN_TRAINER_IN_LOOP and (
        last_training_time is None
        or now - last_training_time >= TRAIN_EVERY_MINUTES * 60
    ):
        training_ran = True
        run_support_script("train_live_3m_from_labeled_predictions.py")
        last_training_time = now

    print("\nLoop orchestration diagnostics")
    print(f"Labeler enabled: {RUN_LABELER_IN_LOOP}")
    print(f"Trainer enabled: {RUN_TRAINER_IN_LOOP}")
    print(f"TRAIN_EVERY_MINUTES: {TRAIN_EVERY_MINUTES}")
    print(f"Training was run this tick: {training_ran}")
    return last_training_time


def main():
    last_training_time = None
    while True:
        try:
            last_training_time = run_once(last_training_time)
        except Exception as error:
            print(f"Live loop tick failed: {error}")
        if RUN_ONCE:
            break
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()
