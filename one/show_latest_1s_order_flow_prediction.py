import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_1s_order_flow_training_rows import CLASS_NAMES, build_feature_row
from microstructure_model_utils import (
    feature_schema_hash,
    infer_snapshot_step_seconds,
    load_snapshot_rows,
    percent,
)
from train_1s_order_flow_model import (
    BURST_TARGET_COLUMNS,
    load_model,
    predict_with_artifact,
    threshold_decode_class_probabilities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
VENUE_TAG = PRIMARY_VENUE or "legacy"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
FLOW_1S_MODEL_SELECTION = os.getenv("FLOW_1S_MODEL_SELECTION", "latest_candidate").strip().lower()
FLOW_1S_MODEL_PATH = os.getenv("FLOW_1S_MODEL_PATH", "").strip()
FLOW_1S_SHOW_MAX_STALENESS_SECONDS = int(os.getenv("FLOW_1S_SHOW_MAX_STALENESS_SECONDS", "120"))
VALID_MODEL_SELECTIONS = {"latest_candidate", "active_only"}
DEFAULT_REGRESSION_TARGET_COLUMNS = [
    "future_log_market_buy_volume_1s",
    "future_log_market_sell_volume_1s",
    "future_market_pressure_1s",
    "future_log_trade_count_1s",
]

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
SNAPSHOT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_10s_flow.csv"
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "active" / SYMBOL / "order_flow_1s" / VENUE_TAG / "model.json"
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "order_flow_1s" / VENUE_TAG


def newest_candidate_model_path():
    if not CANDIDATE_ROOT.exists():
        return None
    candidates = sorted(CANDIDATE_ROOT.glob("*/model.json"))
    return candidates[-1] if candidates else None


def path_is_allowed_for_symbol_and_venue(path):
    path = Path(path).resolve()
    if path == ACTIVE_MODEL_PATH.resolve():
        return True
    try:
        path.relative_to(CANDIDATE_ROOT.resolve())
        return path.name == "model.json"
    except ValueError:
        return False


def choose_model_path():
    if FLOW_1S_MODEL_PATH:
        path = Path(FLOW_1S_MODEL_PATH)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        return path if path_is_allowed_for_symbol_and_venue(path) else None

    if FLOW_1S_MODEL_SELECTION not in VALID_MODEL_SELECTIONS:
        print(
            "Invalid FLOW_1S_MODEL_SELECTION="
            f"{FLOW_1S_MODEL_SELECTION!r}; expected latest_candidate or active_only."
        )
        return None

    if FLOW_1S_MODEL_SELECTION == "active_only":
        return ACTIVE_MODEL_PATH if ACTIVE_MODEL_PATH.exists() else None

    candidate = newest_candidate_model_path()
    if candidate is not None:
        return candidate
    return ACTIVE_MODEL_PATH if ACTIVE_MODEL_PATH.exists() else None


def model_symbol(artifact):
    return str(artifact.get("model_symbol", artifact.get("symbol", ""))).strip().upper()


def model_venue(artifact):
    return str(artifact.get("primary_venue", "legacy")).strip().lower() or "legacy"


def is_optional_context_feature(column):
    return isinstance(column, str) and column.startswith("feature_context_")


def optional_context_default(column):
    if str(column).endswith("_context_age_ms"):
        return -1.0
    return 0.0


def current_feature_columns(frame):
    columns = []
    for column in frame.columns:
        if not isinstance(column, str):
            continue
        if not column.startswith("feature_") or column == "feature_ready":
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any() or is_optional_context_feature(column):
            columns.append(column)
    return sorted(set(columns))


def add_optional_defaults(frame, model_columns):
    frame = frame.copy()
    for column in model_columns:
        if column not in frame.columns and is_optional_context_feature(column):
            frame[column] = optional_context_default(column)
        elif column in frame.columns and is_optional_context_feature(column):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(
                optional_context_default(column)
            )
    return frame


def build_latest_feature_frame(snapshots):
    snapshot_step_seconds = infer_snapshot_step_seconds(snapshots)
    skip_reasons = {}
    for index in range(len(snapshots) - 1, -1, -1):
        features, reason = build_feature_row(snapshots, index, snapshot_step_seconds)
        if features is not None:
            return pd.DataFrame([features]), snapshot_step_seconds, skip_reasons
        skip_reasons[reason or "unknown"] = skip_reasons.get(reason or "unknown", 0) + 1
    return pd.DataFrame(), snapshot_step_seconds, skip_reasons


def print_schema_diagnostics(model_columns, current_columns):
    model_set = set(model_columns)
    current_set = set(current_columns)
    required_missing = sorted(
        column
        for column in model_columns
        if column not in current_set and not is_optional_context_feature(column)
    )
    optional_missing = sorted(
        column
        for column in model_columns
        if column not in current_set and is_optional_context_feature(column)
    )
    extra_current = sorted(current_set - model_set)
    print("Schema validation")
    print(f"- required missing columns: {len(required_missing)}")
    if required_missing:
        for column in required_missing[:50]:
            print(f"  - {column}")
        if len(required_missing) > 50:
            print(f"  - ... {len(required_missing) - 50} more")
    print(f"- optional missing columns filled: {len(optional_missing)}")
    if optional_missing:
        for column in optional_missing[:50]:
            print(f"  - {column}")
        if len(optional_missing) > 50:
            print(f"  - ... {len(optional_missing) - 50} more")
    print(f"- extra current columns ignored: {len(extra_current)}")
    if extra_current:
        for column in extra_current[:50]:
            print(f"  - {column}")
        if len(extra_current) > 50:
            print(f"  - ... {len(extra_current) - 50} more")
    print(f"- artifact feature_schema_hash: {feature_schema_hash(model_columns)}")
    print(
        "- current required-feature hash: "
        f"{feature_schema_hash(sorted(column for column in current_columns if column in model_set))}"
    )
    return required_missing, optional_missing, extra_current


def format_volume(value):
    if value is None or not np.isfinite(float(value)):
        return "unavailable"
    return f"{float(value):.8g}"


def format_probability(value):
    return percent(float(value)) if value is not None and np.isfinite(float(value)) else "unavailable"


def decoded_nonnegative_regression_value(column, value):
    value = float(value)
    clipped = False
    if column.startswith("future_log_"):
        raw_decoded = float(np.expm1(value))
        decoded = max(0.0, raw_decoded)
        clipped = raw_decoded < 0.0
        converted_name = column.replace("future_log_", "future_")
        return converted_name, decoded, clipped
    if column in {
        "future_market_buy_volume_1s",
        "future_market_sell_volume_1s",
        "future_trade_count_1s",
    }:
        decoded = max(0.0, value)
        clipped = value < 0.0
        return column, decoded, clipped
    return column, value, False


def regression_sanity(model_regression_columns, regression_row):
    warnings = []
    decoded = {}
    for index, column in enumerate(model_regression_columns):
        value = float(regression_row[index])
        if "pressure" in column:
            if not np.isfinite(value):
                warnings.append(f"{column} is non-finite")
            decoded[column] = value
            continue
        decoded_name, decoded_value, clipped = decoded_nonnegative_regression_value(column, value)
        if not np.isfinite(decoded_value):
            warnings.append(f"{decoded_name} is non-finite")
        if decoded_name in {
            "future_market_buy_volume_1s",
            "future_market_sell_volume_1s",
            "future_trade_count_1s",
        } and decoded_value < 0.0:
            warnings.append(f"{decoded_name} decoded below zero")
        if clipped:
            warnings.append(f"{decoded_name} clipped to nonnegative display value")
        decoded[decoded_name] = decoded_value
    return {
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "decoded": decoded,
    }


def artifact_target_horizon_seconds(artifact):
    return int(artifact.get("target_horizon_seconds", 1) or 1)


def main():
    model_path = choose_model_path()
    if model_path is None:
        print(f"No 1s order-flow model found for {SYMBOL} on {VENUE_TAG}")
        print(f"SYMBOL: {SYMBOL}")
        print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
        print(f"Snapshot input path: {SNAPSHOT_PATH}")
        print("No trades were placed. No orders were sent.")
        return

    artifact = load_model(model_path)
    if artifact is None:
        print(f"Could not load 1s order-flow model: {model_path}")
        print("No prediction shown. No trades were placed. No orders were sent.")
        return

    loaded_symbol = model_symbol(artifact)
    loaded_venue = model_venue(artifact)
    if loaded_symbol != SYMBOL:
        print(
            "Model symbol mismatch. "
            f"Requested SYMBOL={SYMBOL}, model_symbol={loaded_symbol or 'missing'}."
        )
        print(f"Rejected model path: {model_path}")
        print("No prediction shown. No trades were placed. No orders were sent.")
        return
    if loaded_venue != VENUE_TAG:
        print(
            "Model venue mismatch. "
            f"Requested PRIMARY_VENUE={VENUE_TAG}, model primary_venue={loaded_venue or 'missing'}."
        )
        print(f"Rejected model path: {model_path}")
        print("No prediction shown. No trades were placed. No orders were sent.")
        return

    try:
        snapshots = load_snapshot_rows(SNAPSHOT_PATH)
    except FileNotFoundError:
        print(f"Missing snapshot input file: {SNAPSHOT_PATH}")
        print(f"SYMBOL: {SYMBOL}")
        print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
        print("No prediction shown. No trades were placed. No orders were sent.")
        return
    except ValueError as error:
        print(f"Invalid snapshot input file: {error}")
        print(f"Snapshot input path: {SNAPSHOT_PATH}")
        print("No prediction shown. No trades were placed. No orders were sent.")
        return

    feature_frame, snapshot_step_seconds, skip_reasons = build_latest_feature_frame(snapshots)
    if len(feature_frame) == 0:
        print("No feature-ready 1s order-flow snapshot rows are available yet.")
        print(f"SYMBOL: {SYMBOL}")
        print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
        print(f"Snapshot input path: {SNAPSHOT_PATH}")
        print("Latest feature skip reasons:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"- {reason}: {count}")
        print("No prediction shown. No trades were placed. No orders were sent.")
        return

    model_feature_columns = list(artifact.get("feature_columns", []))
    saved_columns_hash = feature_schema_hash(model_feature_columns)
    model_schema_hash = str(artifact.get("feature_schema_hash", ""))
    current_columns = current_feature_columns(feature_frame)
    feature_frame = add_optional_defaults(feature_frame, model_feature_columns)
    required_missing, _, _ = print_schema_diagnostics(model_feature_columns, current_columns)

    if model_schema_hash != saved_columns_hash:
        print("Feature schema mismatch.")
        print(f"Model feature_schema_hash: {model_schema_hash or 'missing'}")
        print(f"Saved feature_columns hash: {saved_columns_hash}")
        print("Retrain or rebuild this 1s order-flow model for the current feature schema.")
        print("No prediction shown. No trades were placed. No orders were sent.")
        return
    if required_missing:
        print("Missing required real 1s order-flow feature columns.")
        print("No prediction shown. No trades were placed. No orders were sent.")
        return

    model_feature_count = int(artifact.get("feature_count", len(model_feature_columns)))
    if model_feature_count != len(model_feature_columns):
        print(
            "Feature count mismatch. "
            f"model={model_feature_count} saved_columns={len(model_feature_columns)}."
        )
        print("No prediction shown. No trades were placed. No orders were sent.")
        return

    model_input = feature_frame[model_feature_columns].replace([np.inf, -np.inf], np.nan)
    if model_input.isna().any().any():
        bad_columns = model_input.columns[model_input.isna().any()].tolist()
        print("Feature rows contain non-finite values for model columns.")
        for column in bad_columns[:50]:
            print(f"- {column}")
        if len(bad_columns) > 50:
            print(f"- ... {len(bad_columns) - 50} more")
        print("No prediction shown. No trades were placed. No orders were sent.")
        return

    class_probabilities, burst_probabilities, regression = predict_with_artifact(
        artifact,
        feature_frame,
    )
    threshold_decode = artifact.get("threshold_decode", {})
    directional_min_prob = float(threshold_decode.get("directional_min_prob", 0.45))
    directional_neutral_margin = float(threshold_decode.get("directional_neutral_margin", 0.05))
    raw_predicted_class_id = int(np.argmax(class_probabilities[0]))
    raw_predicted_class = CLASS_NAMES.get(raw_predicted_class_id, str(raw_predicted_class_id))
    thresholded_class_id = int(
        threshold_decode_class_probabilities(
            class_probabilities,
            min_prob=directional_min_prob,
            neutral_margin=directional_neutral_margin,
        )[0]
    )
    thresholded_class = CLASS_NAMES.get(thresholded_class_id, str(thresholded_class_id))
    confidence = float(class_probabilities[0, raw_predicted_class_id])
    latest_feature_timestamp = int(feature_frame["timestamp"].iloc[0])
    latest_snapshot_timestamp = int(pd.to_numeric(snapshots["timestamp"], errors="coerce").max())
    staleness_seconds = max(0.0, (time.time() * 1000.0 - latest_feature_timestamp) / 1000.0)
    is_stale = staleness_seconds > FLOW_1S_SHOW_MAX_STALENESS_SECONDS

    print("Latest 1s order-flow paper prediction")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Snapshot input path: {SNAPSHOT_PATH}")
    print(f"Latest snapshot timestamp: {latest_snapshot_timestamp}")
    print(f"Latest feature timestamp: {latest_feature_timestamp}")
    print(f"Latest feature time: {feature_frame['time'].iloc[0]}")
    print(f"Latest feature staleness seconds: {staleness_seconds:.1f}")
    if is_stale:
        print(
            "WARNING: latest feature is stale beyond "
            f"FLOW_1S_SHOW_MAX_STALENESS_SECONDS={FLOW_1S_SHOW_MAX_STALENESS_SECONDS}. "
            "Current-live interpretation suppressed."
        )
    print(f"Inferred snapshot step seconds: {snapshot_step_seconds:.3g}")
    print(f"Model symbol from metadata: {loaded_symbol}")
    print(f"Model primary_venue from metadata: {loaded_venue}")
    print(f"Model path: {model_path}")
    print(f"Model id: {artifact.get('model_id', 'missing')}")
    print(f"Model trained_until_timestamp: {artifact.get('trained_until_timestamp', 'missing')}")
    print(f"Model class target: {artifact.get('class_target_column', 'next_1s_flow_class')}")
    print(f"Model target horizon seconds: {artifact_target_horizon_seconds(artifact)}")
    print(f"Model feature count: {model_feature_count}")
    print(f"feature_schema_hash: {model_schema_hash}")
    print(
        "Threshold decoder: "
        f"min_prob={directional_min_prob:.2%}, "
        f"neutral_margin={directional_neutral_margin:.2%}"
    )
    print("No trades were placed. No orders were sent.")

    model_regression_columns = artifact.get("regression_target_columns", DEFAULT_REGRESSION_TARGET_COLUMNS)
    sanity = regression_sanity(model_regression_columns, regression[0])
    print("\nRegression sanity")
    print(f"- status: {sanity['status']}")
    if sanity["warnings"]:
        print("- WARNING: decoded volume/count output clipped for display; model may be extrapolating.")
        for warning in sanity["warnings"][:10]:
            print(f"  - {warning}")

    horizon_seconds = artifact_target_horizon_seconds(artifact)
    print(f"\nNext-{horizon_seconds}-second flow class probabilities")
    print(f"- sell_dominant: {format_probability(class_probabilities[0, 0])}")
    print(f"- neutral: {format_probability(class_probabilities[0, 1])}")
    print(f"- buy_dominant: {format_probability(class_probabilities[0, 2])}")
    print(f"- raw_argmax_class: {raw_predicted_class}")
    print(f"- thresholded_class: {thresholded_class}")
    print(f"- raw_argmax_confidence: {format_probability(confidence)}")

    print("\nNext-second aggressive burst probabilities")
    for index, column in enumerate(BURST_TARGET_COLUMNS):
        print(f"- {column}: {format_probability(burst_probabilities[0, index])}")

    print(f"\nNext-{horizon_seconds}-second flow regression forecasts")
    for index, column in enumerate(model_regression_columns):
        value = float(regression[0, index])
        if "pressure" in column:
            print(f"- {column}: {format_probability(value)}")
        elif column.startswith("future_log_"):
            converted_name = column.replace("future_log_", "future_")
            converted_value = sanity["decoded"].get(converted_name, np.nan)
            print(f"- {column}: {value:.6g}")
            print(f"- {converted_name} converted from log: {format_volume(converted_value)}")
        else:
            _, decoded_value, _ = decoded_nonnegative_regression_value(column, value)
            print(f"- {column}: {format_volume(decoded_value)}")

    if is_stale:
        print("\nInterpretation suppressed because the latest feature is stale.")
    else:
        print("\nPaper-only interpretation")
        if thresholded_class == "buy_dominant":
            print("- local 1s flow pressure: buy-dominant")
        elif thresholded_class == "sell_dominant":
            print("- local 1s flow pressure: sell-dominant")
        else:
            print("- local 1s flow pressure: neutral/mixed")
        print("- This is a paper-only order-flow context signal, not a buy/sell command.")


if __name__ == "__main__":
    main()
