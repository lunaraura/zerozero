import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
REGIME_TIMEFRAME = os.getenv("REGIME_TIMEFRAME", "30m").strip()

OLD_PREDICTIONS_RECENT_PATH = (
    PROJECT_ROOT / "data" / f"{SYMBOL}_model_predictions_recent.csv"
)
OLD_PREDICTIONS_DEFAULT_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_model_predictions.csv"
OLD_PREDICTIONS_PATH_OVERRIDE = os.getenv("OLD_PREDICTIONS_PATH", "").strip()
FLOW_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "realtime" / f"{SYMBOL}_1m_flow_predictions.csv"
)
FLOW_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "realtime" / f"{SYMBOL}_1m_flow_features.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "ensemble" / f"{SYMBOL}_ensemble_training_rows.csv"

# The old candle model normally updates every 5 minutes. A realtime row may use
# the latest old-model prediction only if that prediction was already known.
MAX_OLD_PREDICTION_AGE_MS = int(
    os.getenv("MAX_OLD_PREDICTION_AGE_MS", str(5 * 60 * 1000))
)
MAX_FLOW_FEATURE_AGE_MS = int(os.getenv("MAX_FLOW_FEATURE_AGE_MS", str(60 * 1000)))

CURRENT_FEATURE_COLUMNS = [
    "spread_percent",
    "volume_zscore_20",
    "trade_count_zscore_20",
    "market_pressure",
    "order_book_imbalance_10bps",
    "order_book_imbalance_25bps",
    "breakout_pressure_index",
    "absorption_index",
]


def normalize_timestamp_series(values):
    timestamps = pd.to_numeric(values, errors="coerce")
    # Some CSVs use seconds instead of milliseconds. Internally we align in ms.
    return np.where(timestamps < 10_000_000_000, timestamps * 1000, timestamps)


def read_csv(path, description):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")

    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{description} is missing a timestamp column: {path}")

    frame["timestamp"] = normalize_timestamp_series(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).copy()
    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    return frame.reset_index(drop=True)


def choose_old_predictions_path():
    if OLD_PREDICTIONS_PATH_OVERRIDE:
        return Path(OLD_PREDICTIONS_PATH_OVERRIDE)
    if OLD_PREDICTIONS_RECENT_PATH.exists():
        return OLD_PREDICTIONS_RECENT_PATH
    return OLD_PREDICTIONS_DEFAULT_PATH


def numeric_or_zero(frame, source_column, output_column=None):
    output_column = output_column or source_column
    if source_column in frame.columns:
        return pd.to_numeric(frame[source_column], errors="coerce")
    return pd.Series(np.zeros(len(frame)), index=frame.index, name=output_column)


def get_old_probability_columns(old_predictions):
    if {"prob_up", "prob_down"}.issubset(old_predictions.columns):
        return "prob_up", "prob_down"

    if {"old_prob_up", "old_prob_down"}.issubset(old_predictions.columns):
        return "old_prob_up", "old_prob_down"

    # Older binary experiments sometimes saved one upward probability column.
    if "probability" in old_predictions.columns:
        old_predictions["derived_prob_down"] = 1.0 - pd.to_numeric(
            old_predictions["probability"], errors="coerce"
        )
        return "probability", "derived_prob_down"

    raise ValueError(
        "Old model predictions need prob_up/prob_down, old_prob_up/old_prob_down, "
        "or a binary probability column."
    )


def prepare_old_predictions(path):
    old_predictions = read_csv(path, "old model prediction CSV")
    up_column, down_column = get_old_probability_columns(old_predictions)

    prepared = pd.DataFrame(
        {
            "old_prediction_timestamp": old_predictions["timestamp"],
            "old_prob_up": pd.to_numeric(old_predictions[up_column], errors="coerce"),
            "old_prob_down": pd.to_numeric(
                old_predictions[down_column], errors="coerce"
            ),
        }
    )
    return prepared.dropna(subset=["old_prob_up", "old_prob_down"])


def prepare_flow_predictions(path):
    flow_predictions = read_csv(path, "realtime flow prediction CSV")
    required = ["prob_short", "prob_neutral", "prob_long", "actual_class"]
    missing = [column for column in required if column not in flow_predictions.columns]
    if missing:
        raise ValueError(f"Flow predictions are missing required columns: {missing}")

    if "actual_return_3" not in flow_predictions.columns:
        raise ValueError(
            "Flow predictions need actual_return_3 so the ensemble can backtest "
            "the same 3-minute outcome."
        )

    prepared = pd.DataFrame(
        {
            "timestamp": flow_predictions["timestamp"],
            "time": flow_predictions["time"]
            if "time" in flow_predictions.columns
            else pd.to_datetime(flow_predictions["timestamp"], unit="ms", utc=True)
            .astype(str),
            "flow_prediction_timestamp": flow_predictions["timestamp"],
            "flow_prob_short": pd.to_numeric(
                flow_predictions["prob_short"], errors="coerce"
            ),
            "flow_prob_neutral": pd.to_numeric(
                flow_predictions["prob_neutral"], errors="coerce"
            ),
            "flow_prob_long": pd.to_numeric(
                flow_predictions["prob_long"], errors="coerce"
            ),
            "actual_class": pd.to_numeric(
                flow_predictions["actual_class"], errors="coerce"
            ),
            "future_return_3": pd.to_numeric(
                flow_predictions["actual_return_3"], errors="coerce"
            ),
        }
    )

    for horizon in [1, 2, 3]:
        return_column = f"pred_return_{horizon}"
        if return_column not in flow_predictions.columns:
            raise ValueError(f"Flow predictions are missing required {return_column}")
        prepared[f"predicted_return_{horizon}"] = pd.to_numeric(
            flow_predictions[return_column], errors="coerce"
        )

        optional_mappings = {
            f"pred_market_pressure_{horizon}": f"predicted_market_pressure_{horizon}",
            f"pred_imbalance_10bps_{horizon}": f"predicted_imbalance_{horizon}",
            f"pred_breakout_pressure_index_{horizon}": (
                f"predicted_breakout_pressure_{horizon}"
            ),
        }
        for source_column, output_column in optional_mappings.items():
            if source_column in flow_predictions.columns:
                prepared[output_column] = pd.to_numeric(
                    flow_predictions[source_column], errors="coerce"
                )

    required_output = [
        "flow_prob_short",
        "flow_prob_neutral",
        "flow_prob_long",
        "actual_class",
        "future_return_3",
    ]
    return prepared.dropna(subset=required_output)


def prepare_flow_features(path):
    flow_features = read_csv(path, "realtime flow feature CSV")
    prepared = pd.DataFrame(
        {
            "feature_timestamp": flow_features["timestamp"],
        }
    )

    selected_columns = list(CURRENT_FEATURE_COLUMNS)
    for column in flow_features.columns:
        lower = column.lower()
        if column.startswith("liquidity_") or lower.endswith("_flag"):
            selected_columns.append(column)

    for column in sorted(set(selected_columns)):
        prepared[column] = numeric_or_zero(flow_features, column)

    return prepared


def find_regime_path():
    preferred = PROJECT_ROOT / "data" / f"{SYMBOL}_{REGIME_TIMEFRAME}_regime_features.csv"
    if preferred.exists():
        return preferred

    for timeframe in ["30m", "15m"]:
        candidate = PROJECT_ROOT / "data" / f"{SYMBOL}_{timeframe}_regime_features.csv"
        if candidate.exists():
            return candidate

    return None


def prepare_regime_features():
    regime_path = find_regime_path()
    if regime_path is None:
        return None, None

    regime = pd.read_csv(regime_path)
    timestamp_column = "close_timestamp" if "close_timestamp" in regime.columns else "timestamp"
    if timestamp_column not in regime.columns:
        raise ValueError(f"Regime CSV is missing timestamp/close_timestamp: {regime_path}")

    regime[timestamp_column] = normalize_timestamp_series(regime[timestamp_column])
    regime = regime.dropna(subset=[timestamp_column]).copy()
    regime[timestamp_column] = regime[timestamp_column].astype(np.int64)
    regime = regime.sort_values(timestamp_column).drop_duplicates(timestamp_column)

    prepared = pd.DataFrame({"regime_close_timestamp": regime[timestamp_column]})
    excluded = {
        timestamp_column,
        "timestamp",
        "close_timestamp",
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    for column in regime.columns:
        if column in excluded:
            continue
        if column == "regime":
            encoded = pd.get_dummies(regime[column].astype(str), prefix="regime")
            prepared = pd.concat([prepared, encoded.astype(float)], axis=1)
        elif pd.api.types.is_numeric_dtype(regime[column]):
            prepared[f"regime_{column}"] = pd.to_numeric(regime[column], errors="coerce")

    return prepared, regime_path


def merge_latest_known(base, right, left_time_column, right_time_column):
    # This is the anti-leakage join: each prediction row may only see rows
    # whose timestamp is at or before the prediction timestamp.
    return pd.merge_asof(
        base.sort_values(left_time_column),
        right.sort_values(right_time_column),
        left_on=left_time_column,
        right_on=right_time_column,
        direction="backward",
    )


def add_agreement_features(frame):
    old_long_edge = frame["old_prob_up"] - frame["old_prob_down"]
    flow_long_edge = frame["flow_prob_long"] - frame["flow_prob_short"]

    frame["old_flow_long_agreement"] = (
        (old_long_edge > 0) & (flow_long_edge > 0)
    ).astype(int)
    frame["old_flow_short_agreement"] = (
        (old_long_edge < 0) & (flow_long_edge < 0)
    ).astype(int)
    frame["directional_conflict"] = (
        np.sign(old_long_edge) * np.sign(flow_long_edge) < 0
    ).astype(int)
    frame["max_directional_confidence"] = frame[
        ["old_prob_up", "old_prob_down", "flow_prob_long", "flow_prob_short"]
    ].max(axis=1)
    frame["confidence_spread"] = (
        old_long_edge.abs() + flow_long_edge.abs()
    ) / 2.0
    return frame


def print_time_range(label, frame, timestamp_column="timestamp"):
    if len(frame) == 0:
        print(f"{label}: 0 rows")
        return
    first = pd.to_datetime(frame[timestamp_column].iloc[0], unit="ms", utc=True)
    last = pd.to_datetime(frame[timestamp_column].iloc[-1], unit="ms", utc=True)
    print(f"{label}: {len(frame)} rows | {first} -> {last}")


def main():
    print("Building stacked ensemble training rows")
    print(f"SYMBOL: {SYMBOL}")
    old_predictions_path = choose_old_predictions_path()
    print(f"Old prediction path: {old_predictions_path}")
    print(f"Recent old prediction path: {OLD_PREDICTIONS_RECENT_PATH}")
    print(f"Fallback old prediction path: {OLD_PREDICTIONS_DEFAULT_PATH}")
    print(f"Flow prediction path: {FLOW_PREDICTIONS_PATH}")
    print(f"Flow feature path: {FLOW_FEATURES_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Max old prediction age: {MAX_OLD_PREDICTION_AGE_MS} ms")
    print(f"Max flow feature age: {MAX_FLOW_FEATURE_AGE_MS} ms")

    old_predictions = prepare_old_predictions(old_predictions_path)
    flow_predictions = prepare_flow_predictions(FLOW_PREDICTIONS_PATH)
    flow_features = prepare_flow_features(FLOW_FEATURES_PATH)
    regime_features, regime_path = prepare_regime_features()

    print_time_range("Old model predictions", old_predictions, "old_prediction_timestamp")
    print_time_range("Flow predictions", flow_predictions)
    print_time_range("Flow features", flow_features, "feature_timestamp")
    print(f"Regime path: {regime_path if regime_path else 'not found; skipped'}")

    merged = merge_latest_known(
        flow_predictions,
        old_predictions,
        "timestamp",
        "old_prediction_timestamp",
    )
    merged["old_prediction_age_ms"] = (
        merged["timestamp"] - merged["old_prediction_timestamp"]
    )
    before_old_filter = len(merged)
    old_prediction_missing = merged["old_prediction_timestamp"].isna()
    old_prediction_stale = (
        merged["old_prediction_timestamp"].notna()
        & (
            (merged["old_prediction_age_ms"] < 0)
            | (merged["old_prediction_age_ms"] > MAX_OLD_PREDICTION_AGE_MS)
        )
    )
    matched_old_predictions = int(
        (
            merged["old_prediction_timestamp"].notna()
            & (merged["old_prediction_age_ms"] >= 0)
            & (merged["old_prediction_age_ms"] <= MAX_OLD_PREDICTION_AGE_MS)
        ).sum()
    )
    merged = merged[
        merged["old_prob_up"].notna()
        & merged["old_prob_down"].notna()
        & (merged["old_prediction_age_ms"] >= 0)
        & (merged["old_prediction_age_ms"] <= MAX_OLD_PREDICTION_AGE_MS)
    ].copy()
    after_old_filter = len(merged)

    merged = merge_latest_known(merged, flow_features, "timestamp", "feature_timestamp")
    merged["flow_feature_age_ms"] = merged["timestamp"] - merged["feature_timestamp"]
    before_feature_filter = len(merged)
    merged = merged[
        merged["feature_timestamp"].notna()
        & (merged["flow_feature_age_ms"] >= 0)
        & (merged["flow_feature_age_ms"] <= MAX_FLOW_FEATURE_AGE_MS)
    ].copy()
    after_feature_filter = len(merged)

    if regime_features is not None:
        merged = merge_latest_known(
            merged,
            regime_features,
            "timestamp",
            "regime_close_timestamp",
        )
        merged["regime_age_ms"] = merged["timestamp"] - merged["regime_close_timestamp"]

    merged = add_agreement_features(merged)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    numeric_columns = [
        column
        for column in merged.columns
        if column != "time" and pd.api.types.is_numeric_dtype(merged[column])
    ]
    merged[numeric_columns] = merged[numeric_columns].replace([np.inf, -np.inf], np.nan)

    required = [
        "old_prob_up",
        "old_prob_down",
        "flow_prob_short",
        "flow_prob_neutral",
        "flow_prob_long",
        "actual_class",
        "future_return_3",
    ]
    before_required_filter = len(merged)
    merged = merged.dropna(subset=required).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print("\nMerge diagnostics")
    print(f"Rows after flow base load: {len(flow_predictions)}")
    print(f"Flow rows matched old predictions: {matched_old_predictions}")
    print(f"Flow rows dropped because old prediction was missing: {int(old_prediction_missing.sum())}")
    print(f"Flow rows dropped because old prediction was stale: {int(old_prediction_stale.sum())}")
    print(f"Rows after old prediction age filter: {after_old_filter}")
    print(f"Rows after realtime feature age filter: {after_feature_filter}")
    print(f"Dropped by missing/stale old prediction: {before_old_filter - after_old_filter}")
    print(
        "Dropped by missing/stale realtime feature row: "
        f"{before_feature_filter - after_feature_filter}"
    )
    print(f"Dropped by missing required target/model fields: {before_required_filter - len(merged)}")
    print(f"Final ensemble rows: {len(merged)}")
    if len(merged) > 0:
        print_time_range("Final ensemble row range", merged)
        print("Class distribution:")
        class_counts = merged["actual_class"].astype(int).value_counts().sort_index()
        for class_id, count in class_counts.items():
            print(f"- class {class_id}: {count} ({count / len(merged):.2%})")
    print(f"\nSaved ensemble training rows to: {OUTPUT_PATH}")
    print("No trades were placed.")


if __name__ == "__main__":
    main()
