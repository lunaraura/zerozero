import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
os.environ.setdefault("SYMBOL", SYMBOL)

import train_python_model as old_model


HISTORICAL_5M_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_5m_imported.csv"
LEGACY_HISTORICAL_5M_PATH = PROJECT_ROOT / "data" / "btc_5m_imported.csv"
REALTIME_DIR = PROJECT_ROOT / "data" / "realtime"
VENUE_REALTIME_DIR = REALTIME_DIR / PRIMARY_VENUE if PRIMARY_VENUE else REALTIME_DIR
REALTIME_1M_PATH = VENUE_REALTIME_DIR / f"{SYMBOL}_1m_flow.csv"
FLOW_PREDICTIONS_PATH = (
    VENUE_REALTIME_DIR / f"{SYMBOL}_1m_flow_predictions.csv"
)
MODEL_PATH = PROJECT_ROOT / "models" / "python_old_5m" / SYMBOL / "model.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_model_predictions_recent.csv"

MAX_OLD_PREDICTION_AGE_MS = int(
    os.getenv("MAX_OLD_PREDICTION_AGE_MS", str(5 * 60 * 1000))
)

OPTIONAL_OLD_MODEL_COLUMNS = [
    "quoteVolume",
    "numberOfTrades",
    "takerBuyBaseVolume",
    "takerBuyQuoteVolume",
]


def normalize_timestamp_series(values):
    timestamps = pd.to_numeric(values, errors="coerce")
    return np.where(timestamps < 10_000_000_000, timestamps * 1000, timestamps)


def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing saved Python old 5m model: {MODEL_PATH}. "
            "Run scripts/train_python_model.py once after this update so it saves "
            "the model weights and feature normalization."
        )

    with MODEL_PATH.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)

    # The feature builders in train_python_model.py read module-level settings.
    # Set them from the artifact so recent inference uses the same shape that
    # was trained, instead of whatever happens to be in the current environment.
    old_model.LOOKBACK = int(artifact["lookback"])
    old_model.HORIZON = int(artifact["horizon"])
    old_model.MIN_TARGET_MOVE = float(artifact["min_target_move"])
    old_model.FIRST_VALID_FEATURE_INDEX = int(
        artifact.get("first_valid_feature_index", old_model.FIRST_VALID_FEATURE_INDEX)
    )
    old_model.USE_REGIME_FEATURES = bool(artifact.get("use_regime_features", False))
    old_model.REGIME_TIMEFRAME = artifact.get("regime_timeframe", old_model.REGIME_TIMEFRAME)
    old_model.REGIME_PATH = (
        PROJECT_ROOT
        / "data"
        / f"{SYMBOL}_{old_model.REGIME_TIMEFRAME}_regime_features.csv"
    )
    old_model.LEGACY_REGIME_PATH = (
        PROJECT_ROOT
        / "data"
        / f"btc_{old_model.REGIME_TIMEFRAME}_regime_features.csv"
    )

    model = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in artifact["model"].items()
    }
    feature_mean = np.asarray(artifact["feature_mean"], dtype=np.float32).reshape(1, -1)
    feature_std = np.asarray(artifact["feature_std"], dtype=np.float32).reshape(1, -1)
    feature_std[feature_std < old_model.EPSILON] = 1.0

    return artifact, model, feature_mean, feature_std


def load_historical_5m():
    path = HISTORICAL_5M_PATH
    if not path.exists() and SYMBOL == "BTCUSDT" and LEGACY_HISTORICAL_5M_PATH.exists():
        path = LEGACY_HISTORICAL_5M_PATH

    if not path.exists():
        print(f"Historical 5m file not found for {SYMBOL}; using realtime 1m aggregates only.")
        columns = ["time", "timestamp", "open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=columns + OPTIONAL_OLD_MODEL_COLUMNS), None

    candles = pd.read_csv(path)
    required = ["time", "timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in candles.columns]
    if missing:
        raise ValueError(f"Historical 5m CSV is missing required columns: {missing}")

    for column in OPTIONAL_OLD_MODEL_COLUMNS:
        if column not in candles.columns:
            candles[column] = np.nan

    candles["timestamp"] = normalize_timestamp_series(candles["timestamp"])
    candles = candles.dropna(subset=["timestamp"]).copy()
    candles["timestamp"] = candles["timestamp"].astype(np.int64)
    return candles, path


def load_realtime_1m():
    if not REALTIME_1M_PATH.exists():
        raise FileNotFoundError(f"Missing realtime 1m flow file: {REALTIME_1M_PATH}")

    frame = pd.read_csv(REALTIME_1M_PATH)
    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Realtime 1m flow CSV is missing required columns: {missing}")

    for column in frame.columns:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["timestamp"] = normalize_timestamp_series(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    return frame.reset_index(drop=True)


def aggregate_realtime_to_completed_5m(realtime_1m):
    frame = realtime_1m.copy()
    frame["bucket_timestamp"] = (frame["timestamp"] // (5 * 60 * 1000)) * (
        5 * 60 * 1000
    )
    rows = []

    for bucket_timestamp, group in frame.groupby("bucket_timestamp"):
        group = group.sort_values("timestamp")
        expected_timestamps = bucket_timestamp + np.arange(5) * 60 * 1000
        actual_timestamps = group["timestamp"].to_numpy(dtype=np.int64)

        # Only completed, contiguous 5-minute groups are safe. The newest partial
        # bucket is intentionally ignored so the 5m model sees completed candles.
        if len(group) < 5 or not np.array_equal(actual_timestamps[-5:], expected_timestamps):
            continue

        group = group.tail(5)
        volume = group["volume"].sum()
        taker_buy_base = group["taker_buy_volume"].sum()
        taker_buy_quote = (
            group["taker_buy_volume"] * group["close"]
        ).sum()
        rows.append(
            {
                "timestamp": int(bucket_timestamp),
                "time": pd.to_datetime(bucket_timestamp, unit="ms", utc=True).isoformat(),
                "open": float(group["open"].iloc[0]),
                "high": float(group["high"].max()),
                "low": float(group["low"].min()),
                "close": float(group["close"].iloc[-1]),
                "volume": float(volume),
                "quoteVolume": float(group["quote_volume"].sum()),
                "numberOfTrades": float(group["trade_count"].sum()),
                "takerBuyBaseVolume": float(taker_buy_base),
                "takerBuyQuoteVolume": float(taker_buy_quote),
            }
        )

    return pd.DataFrame(rows)


def combine_candles(historical_5m, realtime_5m):
    combined = pd.concat([historical_5m, realtime_5m], ignore_index=True)
    required = ["time", "timestamp", "open", "high", "low", "close", "volume"]
    for column in required + OPTIONAL_OLD_MODEL_COLUMNS:
        if column not in combined.columns:
            combined[column] = np.nan

    for column in ["timestamp", "open", "high", "low", "close", "volume"] + OPTIONAL_OLD_MODEL_COLUMNS:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    combined = combined.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    combined["timestamp"] = combined["timestamp"].astype(np.int64)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    combined["time"] = combined["time"].fillna(
        pd.to_datetime(combined["timestamp"], unit="ms", utc=True).astype(str)
    )
    return combined.reset_index(drop=True)


def load_flow_prediction_range():
    if not FLOW_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing flow prediction file: {FLOW_PREDICTIONS_PATH}. "
            "Run scripts/train_1m_flow_model.py first."
        )

    flow_predictions = pd.read_csv(FLOW_PREDICTIONS_PATH)
    if "timestamp" not in flow_predictions.columns:
        raise ValueError("Flow prediction CSV is missing timestamp.")

    timestamps = normalize_timestamp_series(flow_predictions["timestamp"])
    timestamps = pd.Series(timestamps).dropna().astype(np.int64)
    if timestamps.empty:
        raise RuntimeError("Flow prediction CSV has no usable timestamps.")

    return int(timestamps.min()), int(timestamps.max()), len(timestamps)


def build_recent_prediction_rows(candles, artifact, model, feature_mean, feature_std):
    flow_start, flow_end, flow_count = load_flow_prediction_range()
    arrays = old_model.create_arrays(candles)
    indicators = old_model.build_indicators(candles)
    regime_context = old_model.load_regime_features() if old_model.USE_REGIME_FEATURES else None

    first_valid = old_model.LOOKBACK + old_model.FIRST_VALID_FEATURE_INDEX
    # prediction_index == len(candles) is allowed for the newest known candle:
    # the input window ends at len(candles)-1 and future target is unknown.
    candidate_indexes = range(first_valid, len(candles) + 1)
    rows = []
    skipped_bad_features = 0
    skipped_outside_flow_range = 0

    for prediction_index in candidate_indexes:
        prediction_timestamp = int(candles.iloc[prediction_index - 1]["timestamp"])
        if prediction_timestamp < flow_start - MAX_OLD_PREDICTION_AGE_MS:
            skipped_outside_flow_range += 1
            continue
        if prediction_timestamp > flow_end:
            skipped_outside_flow_range += 1
            continue

        feature_window = old_model.build_feature_window(
            arrays,
            indicators,
            prediction_index,
        )
        if feature_window is None:
            skipped_bad_features += 1
            continue

        flattened_window = feature_window.reshape(-1)
        if old_model.USE_REGIME_FEATURES:
            regime_features = old_model.regime_features_for_prediction(
                candles,
                prediction_index,
                regime_context,
            )
            flattened_window = np.concatenate([flattened_window, regime_features])

        if flattened_window.shape[0] != feature_mean.shape[1]:
            raise RuntimeError(
                "Recent feature shape does not match the saved model. "
                f"Recent size: {flattened_window.shape[0]}, "
                f"saved size: {feature_mean.shape[1]}. "
                "Re-train scripts/train_python_model.py with the same feature settings."
            )

        x = ((flattened_window.reshape(1, -1) - feature_mean) / feature_std).astype(
            np.float32
        )
        _, _, probabilities = old_model.forward(model, x)

        actual_class = np.nan
        future_return = np.nan
        if prediction_index + old_model.HORIZON - 1 < len(candles):
            actual_class, future_return = old_model.three_class_target(
                arrays["close"],
                prediction_index,
            )

        rows.append(
            {
                "timestamp": prediction_timestamp,
                "time": candles.iloc[prediction_index - 1]["time"],
                "prob_down": probabilities[0, old_model.CLASS_DOWN],
                "prob_neutral": probabilities[0, old_model.CLASS_NEUTRAL],
                "prob_up": probabilities[0, old_model.CLASS_UP],
                "actual_class": actual_class,
                "future_return": future_return,
            }
        )

    diagnostics = {
        "flow_start": flow_start,
        "flow_end": flow_end,
        "flow_count": flow_count,
        "skipped_bad_features": skipped_bad_features,
        "skipped_outside_flow_range": skipped_outside_flow_range,
        "regime_rows_without_context": (
            regime_context["rows_without_context"] if regime_context else 0
        ),
    }
    return pd.DataFrame(rows), diagnostics


def main():
    artifact, model, feature_mean, feature_std = load_model_artifact()
    historical_5m, historical_path = load_historical_5m()
    realtime_1m = load_realtime_1m()
    realtime_5m = aggregate_realtime_to_completed_5m(realtime_1m)
    candles = combine_candles(historical_5m, realtime_5m)

    if len(candles) < old_model.LOOKBACK + old_model.FIRST_VALID_FEATURE_INDEX:
        raise RuntimeError(
            f"Only {len(candles)} combined 5m candles are available. "
            f"Need at least {old_model.LOOKBACK + old_model.FIRST_VALID_FEATURE_INDEX} "
            "for the old 5m feature stack."
        )

    output, diagnostics = build_recent_prediction_rows(
        candles,
        artifact,
        model,
        feature_mean,
        feature_std,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("Recent old 5m model prediction bridge")
    print(f"SYMBOL: {SYMBOL}")
    print(f"Historical 5m path: {historical_path if historical_path else 'not found'}")
    print(f"Realtime 1m path: {REALTIME_1M_PATH}")
    print(f"Flow prediction path: {FLOW_PREDICTIONS_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Historical 5m rows: {len(historical_5m)}")
    print(f"Realtime 1m rows: {len(realtime_1m)}")
    print(f"Completed realtime 5m candles: {len(realtime_5m)}")
    print(f"Combined 5m candles: {len(candles)}")
    print(f"Lookback: {old_model.LOOKBACK}")
    print(f"Horizon: {old_model.HORIZON}")
    print(f"Feature count per candle: {len(old_model.FEATURE_NAMES)}")
    print(f"Flattened saved input size: {feature_mean.shape[1]}")
    print(
        "Flow prediction timestamp range: "
        f"{pd.to_datetime(diagnostics['flow_start'], unit='ms', utc=True)} -> "
        f"{pd.to_datetime(diagnostics['flow_end'], unit='ms', utc=True)} "
        f"({diagnostics['flow_count']} rows)"
    )
    print(f"Recent old-model prediction rows written: {len(output)}")
    print(f"Skipped outside flow overlap range: {diagnostics['skipped_outside_flow_range']}")
    print(f"Skipped because features were not ready: {diagnostics['skipped_bad_features']}")
    print(
        "Rows without regime context: "
        f"{diagnostics['regime_rows_without_context']}"
    )
    if len(output) > 0:
        print(
            "Recent prediction range: "
            f"{output.iloc[0]['time']} -> {output.iloc[-1]['time']}"
        )
    print("No trades were placed.")


if __name__ == "__main__":
    main()
