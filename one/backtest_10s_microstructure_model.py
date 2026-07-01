import os
from pathlib import Path

import numpy as np
import pandas as pd

from microstructure_model_utils import (
    EVENT_TARGET_COLUMNS,
    REGRESSION_TARGET_COLUMNS,
    add_missing_optional_context_columns,
    fill_optional_context_feature_values,
    feature_schema_hash,
    get_micro_feature_columns,
    load_model,
    micro_schema_diagnostics,
    percent,
    print_micro_schema_diagnostics,
    precision_recall,
    predictions_frame,
    predict_with_artifact,
    required_micro_feature_columns,
    validate_regression_scalers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.80"))
MODEL_PATH_ENV = os.getenv("MICROSTRUCTURE_MODEL_PATH", "").strip()
PROBABILITY_THRESHOLD = float(os.getenv("PROBABILITY_THRESHOLD", "0.50"))
MICRO_EVENT_PROB_TEMPERATURE = float(os.getenv("MICRO_EVENT_PROB_TEMPERATURE", "1.0"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
TRAINING_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_10s_microstructure_training_rows.csv"
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "active" / SYMBOL / "microstructure_10s" / "model.json"
MODEL_PATH = Path(MODEL_PATH_ENV) if MODEL_PATH_ENV else ACTIVE_MODEL_PATH
if MODEL_PATH and not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH


def load_training_rows():
    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            f"Missing training rows: {TRAINING_PATH}. "
            "Run scripts/build_10s_microstructure_training_rows.py first."
        )
    frame = pd.read_csv(TRAINING_PATH)
    for column in frame.columns:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required = ["timestamp", *REGRESSION_TARGET_COLUMNS, *EVENT_TARGET_COLUMNS]
    frame = frame.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)
    return frame


def validation_split(frame):
    split_index = int(len(frame) * TRAIN_SPLIT)
    split_index = max(1, min(split_index, len(frame) - 1))
    return frame.iloc[split_index:].copy()


def print_event_distributions(frame):
    print("\nValidation event distributions")
    total = len(frame) or 1
    for column in EVENT_TARGET_COLUMNS:
        positives = int(frame[column].sum())
        print(f"- {column}: {positives}/{len(frame)} ({percent(positives / total)})")


def print_precision_recall(validation, predictions):
    print("\nValidation precision/recall at probability >= 0.50")
    for event in ["upside_scare_event_10s", "downside_scare_event_10s"]:
        precision, recall, tp, fp, fn = precision_recall(
            validation[event],
            predictions[f"prob_{event}"],
            threshold=PROBABILITY_THRESHOLD,
        )
        print(
            f"- {event}: precision={percent(precision)}, recall={percent(recall)}, "
            f"tp={tp}, fp={fp}, fn={fn}"
        )


def top_slice_metrics(validation, predictions, fraction):
    probability = predictions[
        ["prob_upside_scare_event_10s", "prob_downside_scare_event_10s"]
    ].max(axis=1)
    count = max(1, int(len(validation) * fraction))
    subset = validation.assign(_probability=probability).sort_values(
        "_probability",
        ascending=False,
    ).head(count)
    predicted_up = predictions.loc[subset.index, "prob_upside_scare_event_10s"]
    predicted_down = predictions.loc[subset.index, "prob_downside_scare_event_10s"]
    predicted_side = np.where(predicted_up >= predicted_down, "UPSIDE", "DOWNSIDE")
    continuation = []
    reversal = []
    for side, (_, row) in zip(predicted_side, subset.iterrows()):
        if side == "UPSIDE":
            continuation.append(row.get("continuation_30s", 0))
            reversal.append(row.get("reversal_after_upside_scare_30s", 0))
        else:
            continuation.append(row.get("continuation_30s", 0))
            reversal.append(row.get("reversal_after_downside_scare_30s", 0))

    return {
        "count": int(len(subset)),
        "avg_probability": float(subset["_probability"].mean()),
        "avg_max_runup": float(subset["max_runup_10s"].mean()),
        "avg_max_drawdown": float(subset["max_drawdown_10s"].mean()),
        "upside_scare_rate": float(subset["upside_scare_event_10s"].mean()),
        "downside_scare_rate": float(subset["downside_scare_event_10s"].mean()),
        "continuation_30s_rate": float(np.mean(continuation)) if continuation else 0.0,
        "reversal_30s_rate": float(np.mean(reversal)) if reversal else 0.0,
    }


def print_top_slices(validation, predictions):
    print("\nTop predicted scare-probability slices")
    print(
        "slice | rows | avg prob | avg max_runup | avg max_drawdown | "
        "up scare | down scare | continuation 30s | reversal 30s"
    )
    for fraction in [0.005, 0.01, 0.02, 0.05]:
        metrics = top_slice_metrics(validation, predictions, fraction)
        print(
            f"top {fraction:.1%} | {metrics['count']} | "
            f"{percent(metrics['avg_probability'])} | "
            f"{percent(metrics['avg_max_runup'])} | "
            f"{percent(metrics['avg_max_drawdown'])} | "
            f"{percent(metrics['upside_scare_rate'])} | "
            f"{percent(metrics['downside_scare_rate'])} | "
            f"{percent(metrics['continuation_30s_rate'])} | "
            f"{percent(metrics['reversal_30s_rate'])}"
        )


def print_probability_buckets(validation, predictions):
    probability = predictions[
        ["prob_upside_scare_event_10s", "prob_downside_scare_event_10s"]
    ].max(axis=1)
    buckets = pd.cut(
        probability,
        bins=[0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0],
        include_lowest=True,
    )
    print("\nAverage actual max_runup/max_drawdown by predicted scare-probability bucket")
    for bucket in buckets.cat.categories:
        mask = buckets == bucket
        subset = validation[mask]
        if len(subset) == 0:
            print(f"- {bucket}: rows=0")
            continue
        print(
            f"- {bucket}: rows={len(subset)}, "
            f"avg max_runup={percent(subset['max_runup_10s'].mean())}, "
            f"avg max_drawdown={percent(subset['max_drawdown_10s'].mean())}"
        )


def print_aftermath_rates(validation, predictions):
    probability = predictions[
        ["prob_upside_scare_event_10s", "prob_downside_scare_event_10s"]
    ].max(axis=1)
    selected = validation[probability >= PROBABILITY_THRESHOLD]
    print("\nContinuation vs reversal after predicted scare events")
    print(f"Predicted scare threshold: {PROBABILITY_THRESHOLD:.2f}")
    print(f"Selected rows: {len(selected)}")
    if len(selected) == 0:
        return
    print(f"- continuation_30s: {percent(selected['continuation_30s'].mean())}")
    print(f"- continuation_60s: {percent(selected['continuation_60s'].mean())}")
    print(
        "- reversal_after_upside_scare_30s: "
        f"{percent(selected['reversal_after_upside_scare_30s'].mean())}"
    )
    print(
        "- reversal_after_downside_scare_30s: "
        f"{percent(selected['reversal_after_downside_scare_30s'].mean())}"
    )
    print(
        "- reversal_after_upside_scare_60s: "
        f"{percent(selected['reversal_after_upside_scare_60s'].mean())}"
    )
    print(
        "- reversal_after_downside_scare_60s: "
        f"{percent(selected['reversal_after_downside_scare_60s'].mean())}"
    )


def main():
    artifact = load_model(MODEL_PATH)
    if artifact is None:
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}. Set MICROSTRUCTURE_MODEL_PATH or train/promote a model first."
        )

    regression_columns = list(artifact.get("regression_target_columns", REGRESSION_TARGET_COLUMNS))
    event_columns = list(artifact.get("event_target_columns", EVENT_TARGET_COLUMNS))
    validate_regression_scalers(artifact, regression_columns, allow_legacy=False)
    frame = load_training_rows()
    model_columns = list(artifact.get("feature_columns", []))
    frame = add_missing_optional_context_columns(frame, model_columns)
    frame = fill_optional_context_feature_values(frame, model_columns)
    current_columns = get_micro_feature_columns(frame)
    model_symbol = str(artifact.get("model_symbol", artifact.get("symbol", ""))).upper()
    if model_symbol != SYMBOL:
        raise ValueError(f"Model symbol {model_symbol} does not match requested SYMBOL {SYMBOL}.")
    current_schema_hash = feature_schema_hash(model_columns)
    model_schema_hash = artifact.get("feature_schema_hash", "")
    if model_schema_hash != feature_schema_hash(model_columns):
        raise ValueError(
            "Feature schema mismatch: "
            f"model={model_schema_hash or 'missing'}, saved_columns={feature_schema_hash(model_columns)}. "
            "Rebuild micro training rows and retrain this symbol/venue model."
        )
    model_feature_count = int(artifact.get("feature_count", len(artifact.get("feature_columns", []))))
    if model_feature_count != len(model_columns):
        raise ValueError(
            f"Feature count mismatch: model={model_feature_count}, saved_columns={len(model_columns)}."
        )
    missing_required = [
        column for column in required_micro_feature_columns(model_columns)
        if column not in current_columns
    ]
    if missing_required:
        print("Feature schema mismatch diagnostics")
        print_micro_schema_diagnostics(micro_schema_diagnostics(model_columns, current_columns))
        raise ValueError(f"Missing required real microstructure feature columns: {missing_required[:20]}")
    validation = validation_split(frame)
    regression, event_probabilities = predict_with_artifact(
        artifact,
        validation,
        event_temperature=MICRO_EVENT_PROB_TEMPERATURE,
    )
    predictions = predictions_frame(
        validation.reset_index(drop=True),
        regression,
        event_probabilities,
        regression_columns,
        event_columns,
    )
    validation = validation.reset_index(drop=True)

    print("10s microstructure paper backtest")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Training rows: {TRAINING_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model symbol: {model_symbol}")
    print(f"Model id: {artifact.get('model_id', 'missing')}")
    print(f"Model trained_until_timestamp: {artifact.get('trained_until_timestamp', 'missing')}")
    print(f"Model feature count: {model_feature_count}")
    print(f"Feature schema hash: {model_schema_hash or 'missing'}")
    print(f"MICRO_EVENT_PROB_TEMPERATURE: {MICRO_EVENT_PROB_TEMPERATURE}")
    print(f"Current available feature count: {len(current_columns)}")
    print(f"Total rows: {len(frame)}")
    print(f"Chronological validation rows: {len(validation)}")
    if len(validation):
        print(f"Validation timestamp range: {int(validation['timestamp'].min())} -> {int(validation['timestamp'].max())}")
    print("No trades are placed.")

    print_event_distributions(validation)
    print_precision_recall(validation, predictions)
    print_top_slices(validation, predictions)
    print_probability_buckets(validation, predictions)
    print_aftermath_rates(validation, predictions)
    print("\nThis is paper-only scenario/event scoring, not a trading system.")


if __name__ == "__main__":
    main()
