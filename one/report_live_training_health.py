import os
from pathlib import Path

import numpy as np
import pandas as pd

from live_3m_model_utils import (
    ACTUAL_LABEL_COLUMNS,
    build_input_window,
    choose_feature_columns,
    coerce_feature_ready,
    coerce_numeric_columns,
    parse_bool,
)
from hierarchical_context import print_context_availability_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
LOOKBACK = int(os.getenv("LOOKBACK", "30"))

REALTIME_DIR = PROJECT_ROOT / "data" / "realtime"
VENUE_REALTIME_DIR = REALTIME_DIR / PRIMARY_VENUE if PRIMARY_VENUE else REALTIME_DIR
RAW_1M_PATH = VENUE_REALTIME_DIR / f"{SYMBOL}_1m_flow.csv"
FEATURES_PATH = VENUE_REALTIME_DIR / f"{SYMBOL}_1m_flow_features.csv"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "live_predictions" / f"{SYMBOL}_live_3m_predictions.csv"
)
LABELED_PATH = (
    PROJECT_ROOT / "data" / "live_training" / f"{SYMBOL}_labeled_3m_training_rows.csv"
)

CORE_TARGET_COLUMNS = [
    "actual_future_return_3",
    "actual_future_return_class",
    "actual_path_event_class",
    "actual_path_class",
]
STAGE_1_TARGET_COLUMNS = [
    "actual_future_return_3",
]
STAGE_2_TARGET_COLUMNS = [
    "actual_future_return_3",
    "actual_future_range_percent_3",
    "actual_market_pressure_3m",
    "actual_breakout_pressure_3m",
]
STAGE_3_TARGET_COLUMNS = ACTUAL_LABEL_COLUMNS
CLASS_COLUMNS = [
    "actual_future_return_class",
    "actual_path_event_class",
    "actual_path_class",
]


def load_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for column in frame.columns:
        if column not in {"time", "label_ready", "core_label_ready", "full_regression_label_ready", "feature_ready"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("timestamp" if "timestamp" in frame.columns else frame.columns[0]).reset_index(drop=True)


def complete_mask(frame, columns):
    missing = [column for column in columns if column not in frame.columns]
    if missing or len(frame) == 0:
        return pd.Series(False, index=frame.index)
    values = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.notna().all(axis=1)


def print_count(label, count):
    print(f"- {label}: {int(count)}")


def print_class_distribution(frame, column):
    print(f"\n{column} distribution")
    if column not in frame.columns or len(frame) == 0:
        print("- unavailable")
        return
    values = pd.to_numeric(frame[column], errors="coerce").dropna().astype(int)
    total = len(values) or 1
    for class_id in [0, 1, 2]:
        count = int((values == class_id).sum())
        print(f"- class {class_id}: {count} ({count / total:.2%})")


def print_missing_targets(frame):
    print("\nMissing target reasons")
    if len(frame) == 0:
        print("- no live prediction rows")
        return

    target_sets = {
        "stage 1 minimal": STAGE_1_TARGET_COLUMNS + ["actual_future_return_class"],
        "stage 2 direction+pressure": STAGE_2_TARGET_COLUMNS + ["actual_future_return_class"],
        "stage 3 full": STAGE_3_TARGET_COLUMNS,
    }
    for label, columns in target_sets.items():
        mask = complete_mask(frame, columns)
        print(f"- complete {label} rows: {int(mask.sum())}/{len(frame)}")
        missing_counts = {}
        for column in columns:
            if column not in frame.columns:
                missing_counts[column] = len(frame)
            else:
                missing_counts[column] = int(pd.to_numeric(frame[column], errors="coerce").isna().sum())
        missing_counts = {key: value for key, value in missing_counts.items() if value > 0}
        if missing_counts:
            top = sorted(missing_counts.items(), key=lambda item: item[1], reverse=True)[:8]
            print("  top missing columns:")
            for column, count in top:
                print(f"  - {column}: {count}")

    if "label_ready" in frame.columns:
        corrupt_ready = frame[frame["label_ready"].apply(parse_bool) & ~complete_mask(frame, STAGE_3_TARGET_COLUMNS)]
        print(f"- label_ready=true but full targets incomplete: {len(corrupt_ready)}")


def feature_window_health(feature_frame):
    if len(feature_frame) == 0:
        return {
            "feature_columns": 0,
            "valid_windows": 0,
            "invalid_windows": 0,
            "invalid_not_ready": 0,
            "invalid_non_contiguous": 0,
            "invalid_missing_values": 0,
        }
    feature_frame = coerce_numeric_columns(feature_frame)
    feature_frame = coerce_feature_ready(feature_frame)
    columns = choose_feature_columns(feature_frame)
    valid_windows = 0
    invalid_not_ready = 0
    invalid_non_contiguous = 0
    invalid_missing_values = 0
    invalid_windows = 0
    for end_index in range(len(feature_frame)):
        start_index = end_index - LOOKBACK + 1
        if start_index < 0:
            continue
        window = feature_frame.iloc[start_index : end_index + 1]
        built, _, _ = build_input_window(feature_frame, end_index, columns, LOOKBACK)
        if built is not None:
            valid_windows += 1
            continue
        invalid_windows += 1
        if "feature_ready" in window.columns and not bool(window["feature_ready"].all()):
            invalid_not_ready += 1
        elif not np.all(np.diff(window["timestamp"].to_numpy(dtype=np.int64)) == 60_000):
            invalid_non_contiguous += 1
        elif window[columns].replace([np.inf, -np.inf], np.nan).isna().any().any():
            invalid_missing_values += 1
    return {
        "feature_columns": len(columns),
        "valid_windows": valid_windows,
        "invalid_windows": invalid_windows,
        "invalid_not_ready": invalid_not_ready,
        "invalid_non_contiguous": invalid_non_contiguous,
        "invalid_missing_values": invalid_missing_values,
    }


def main():
    raw = load_csv(RAW_1M_PATH)
    features = load_csv(FEATURES_PATH)
    predictions = load_csv(PREDICTIONS_PATH)
    labeled = load_csv(LABELED_PATH)

    if len(features):
        features = coerce_feature_ready(features)
    if len(predictions):
        for column in ["label_ready", "core_label_ready", "full_regression_label_ready"]:
            if column in predictions.columns:
                predictions[column] = predictions[column].apply(parse_bool)

    print("Live training health report")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"LOOKBACK: {LOOKBACK}")
    print(f"Raw 1m path: {RAW_1M_PATH}")
    print(f"Feature path: {FEATURES_PATH}")
    print(f"Prediction path: {PREDICTIONS_PATH}")
    print(f"Labeled training path: {LABELED_PATH}")

    print("\nRow counts")
    print_count("raw 1m rows", len(raw))
    print_count("feature rows", len(features))
    print_count("feature_ready rows", features["feature_ready"].sum() if "feature_ready" in features.columns else 0)
    print_count("live prediction rows", len(predictions))
    print_count(
        "core_label_ready rows",
        predictions["core_label_ready"].sum() if "core_label_ready" in predictions.columns else complete_mask(predictions, CORE_TARGET_COLUMNS).sum(),
    )
    print_count(
        "full_regression_label_ready rows",
        predictions["full_regression_label_ready"].sum() if "full_regression_label_ready" in predictions.columns else complete_mask(predictions, STAGE_3_TARGET_COLUMNS).sum(),
    )
    print_count("labeled training rows", len(labeled))

    if len(raw):
        print(f"- latest raw timestamp: {int(raw['timestamp'].max())}")
    if len(features):
        print(f"- latest feature timestamp: {int(features['timestamp'].max())}")
        if "feature_ready" in features.columns and features["feature_ready"].any():
            print(f"- latest feature_ready timestamp: {int(features[features['feature_ready']]['timestamp'].max())}")
    if len(predictions):
        print(f"- latest prediction timestamp: {int(predictions['timestamp'].max())}")

    print_context_availability_summary(features, "Feature context availability summary")
    print_context_availability_summary(predictions, "Prediction context availability summary")
    print_context_availability_summary(labeled, "Labeled training context availability summary")

    window_health = feature_window_health(features)
    print("\nFeature/lookback health")
    for label, count in window_health.items():
        print_count(label, count)
    not_ready_columns = [column for column in features.columns if column.startswith("not_ready_")]
    if not_ready_columns:
        print("Feature readiness reasons:")
        for column in not_ready_columns:
            print(f"- {column}: {int(features[column].sum())}")

    print_missing_targets(predictions)
    for column in CLASS_COLUMNS:
        print_class_distribution(predictions, column)

    if len(predictions) and {"actual_future_return_class", "actual_path_event_class"}.issubset(predictions.columns):
        comparable = predictions.dropna(subset=["actual_future_return_class", "actual_path_event_class"])
        if len(comparable):
            agreement = (
                comparable["actual_future_return_class"].astype(int)
                == comparable["actual_path_event_class"].astype(int)
            ).mean()
            print(f"\nfuture_return_class vs path_event_class agreement: {agreement:.2%} over {len(comparable)} rows")

    print("\nTraining caution")
    full_rows = int(complete_mask(predictions, STAGE_3_TARGET_COLUMNS).sum())
    stage1_rows = int(complete_mask(predictions, STAGE_1_TARGET_COLUMNS + ["actual_future_return_class"]).sum())
    print(f"- stage 1 usable rows: {stage1_rows}")
    print(f"- stage 3 full usable rows: {full_rows}")
    if full_rows < 500:
        print("- full-label count is still small; treat serious training metrics as noisy.")
    print("No trades were placed.")


if __name__ == "__main__":
    main()
