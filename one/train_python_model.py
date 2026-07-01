import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "BTCUSDT").strip().upper()
DATA_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_5m_imported.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_model_predictions.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "python_old_5m" / SYMBOL / "model.json"
LEGACY_DATA_PATH = PROJECT_ROOT / "data" / "btc_5m_imported.csv"

LOOKBACK = int(os.getenv("LOOKBACK", "60"))
HORIZON = int(os.getenv("HORIZON", "5"))
MIN_TARGET_MOVE = float(os.getenv("MIN_TARGET_MOVE", "0.0062"))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.8"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "5000"))
VALIDATION_MAX_WINDOWS = int(os.getenv("VALIDATION_MAX_WINDOWS", "5000"))
PYTHON_EPOCHS = int(os.getenv("PYTHON_EPOCHS", "25"))
PYTHON_BATCH_SIZE = int(os.getenv("PYTHON_BATCH_SIZE", "128"))
PYTHON_LEARNING_RATE = float(os.getenv("PYTHON_LEARNING_RATE", "0.001"))
PYTHON_HIDDEN_UNITS = int(os.getenv("PYTHON_HIDDEN_UNITS", "64"))
PYTHON_EARLY_STOPPING_PATIENCE = int(
    os.getenv("PYTHON_EARLY_STOPPING_PATIENCE", "5")
)
USE_REGIME_FEATURES = os.getenv("USE_REGIME_FEATURES", "false").strip().lower() in [
    "true",
    "1",
    "yes",
    "y",
]
REGIME_TIMEFRAME = os.getenv("REGIME_TIMEFRAME", "30m")
REGIME_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_{REGIME_TIMEFRAME}_regime_features.csv"
LEGACY_REGIME_PATH = PROJECT_ROOT / "data" / f"btc_{REGIME_TIMEFRAME}_regime_features.csv"

EPSILON = 1e-8
FIRST_VALID_FEATURE_INDEX = 49
CLASS_DOWN = 0
CLASS_NEUTRAL = 1
CLASS_UP = 2
CLASS_NAMES = {
    CLASS_DOWN: "strong down",
    CLASS_NEUTRAL: "neutral / no trade",
    CLASS_UP: "strong up",
}

FEATURE_NAMES = [
    "return_1",
    "return_3",
    "return_5",
    "return_12",
    "RSI_14",
    "EMA20_distance",
    "EMA50_distance",
    "MACD_line",
    "MACD_signal",
    "MACD_histogram",
    "volume_change",
    "relative_volume_20",
    "volume_zscore_20",
    "candle_range_percent",
    "ATR_14",
    "rolling_volatility_20",
    "close_position_within_candle_range",
    "distance_from_rolling_20_high",
    "distance_from_rolling_20_low",
    "takerBuyRatio",
    "takerSellRatio",
    "tradeCountChange",
]

REGIME_NUMERIC_FEATURE_NAMES = [
    "return_1",
    "return_4",
    "return_12",
    "ema20_distance",
    "ema50_distance",
    "ema20_slope_4",
    "ema50_slope_4",
    "rsi14",
    "atr14_percent",
    "rolling_volatility_20",
    "trend_score",
    "chop_score",
]

REGIME_ONE_HOT_FEATURE_NAMES = [
    "regime_bullish",
    "regime_bearish",
    "regime_chop",
    "regime_high_volatility_chop",
]

REGIME_FEATURE_NAMES = REGIME_NUMERIC_FEATURE_NAMES + REGIME_ONE_HOT_FEATURE_NAMES
REGIME_TO_ONE_HOT_INDEX = {
    "bullish": 0,
    "bearish": 1,
    "chop": 2,
    "high_volatility_chop": 3,
}


def safe_ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return 0.0
    return numerator / max(abs(denominator), EPSILON)


def clip(value, minimum=-5.0, maximum=5.0):
    if not np.isfinite(value):
        return 0.0
    return float(max(minimum, min(maximum, value)))


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(logits, -40, 40))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def load_candles():
    data_path = DATA_PATH
    if not data_path.exists() and SYMBOL == "BTCUSDT" and LEGACY_DATA_PATH.exists():
        data_path = LEGACY_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run npm run download-binance-history first."
        )

    candles = pd.read_csv(data_path)

    required_columns = ["time", "timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required_columns if column not in candles.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Keep the feature shape fixed at 22 columns. If optional Binance fields are
    # missing, fill them with NaN and use neutral feature defaults later.
    for column in [
        "quoteVolume",
        "numberOfTrades",
        "takerBuyBaseVolume",
        "takerBuyQuoteVolume",
    ]:
        if column not in candles.columns:
            candles[column] = np.nan

    candles = candles.sort_values("timestamp").drop_duplicates("timestamp")
    candles = candles.reset_index(drop=True)
    return candles, data_path


def load_regime_features():
    if not USE_REGIME_FEATURES:
        return None

    regime_path = REGIME_PATH
    if not regime_path.exists() and SYMBOL == "BTCUSDT" and LEGACY_REGIME_PATH.exists():
        regime_path = LEGACY_REGIME_PATH

    if not regime_path.exists():
        raise FileNotFoundError(
            f"{REGIME_PATH} not found. Create regime features first or set "
            "USE_REGIME_FEATURES=false."
        )

    regimes = pd.read_csv(regime_path)
    required_columns = ["close_timestamp", "regime"] + REGIME_NUMERIC_FEATURE_NAMES
    missing = [column for column in required_columns if column not in regimes.columns]
    if missing:
        raise ValueError(f"Regime CSV is missing required columns: {missing}")

    regimes["close_timestamp"] = pd.to_numeric(
        regimes["close_timestamp"],
        errors="coerce",
    )

    for column in REGIME_NUMERIC_FEATURE_NAMES:
        regimes[column] = pd.to_numeric(regimes[column], errors="coerce").fillna(0.0)

    regimes["regime"] = regimes["regime"].fillna("unknown").astype(str)
    regimes = regimes.dropna(subset=["close_timestamp"])
    regimes = regimes.sort_values("close_timestamp").drop_duplicates(
        "close_timestamp",
        keep="last",
    )
    regimes = regimes.reset_index(drop=True)

    return {
        "close_timestamps": regimes["close_timestamp"].to_numpy(dtype=np.float64),
        "numeric_features": regimes[REGIME_NUMERIC_FEATURE_NAMES].to_numpy(
            dtype=np.float32
        ),
        "regimes": regimes["regime"].to_numpy(),
        "rows_without_context": 0,
        "path": regime_path,
    }


def regime_row_to_features(regime_context, regime_row_index):
    if regime_context is None or regime_row_index < 0:
        return np.zeros(len(REGIME_FEATURE_NAMES), dtype=np.float32)

    numeric_features = regime_context["numeric_features"][regime_row_index]
    regime_name = str(regime_context["regimes"][regime_row_index]).strip().lower()
    one_hot = np.zeros(len(REGIME_ONE_HOT_FEATURE_NAMES), dtype=np.float32)
    one_hot_index = REGIME_TO_ONE_HOT_INDEX.get(regime_name)

    if one_hot_index is not None:
        one_hot[one_hot_index] = 1.0

    return np.concatenate([numeric_features, one_hot]).astype(np.float32)


def regime_features_for_prediction(candles, prediction_index, regime_context):
    if regime_context is None:
        return None

    # Anti-leakage alignment:
    # The prediction at index i only knows candles through i - 1, so its
    # timestamp T is the close time of that latest input candle. We attach only
    # the latest completed regime row with close_timestamp <= T.
    prediction_timestamp = float(candles.iloc[prediction_index - 1]["timestamp"])
    regime_row_index = (
        np.searchsorted(
            regime_context["close_timestamps"],
            prediction_timestamp,
            side="right",
        )
        - 1
    )

    if regime_row_index < 0:
        regime_context["rows_without_context"] += 1

    return regime_row_to_features(regime_context, regime_row_index)


def ema(values, period):
    result = np.full(len(values), np.nan, dtype=np.float64)
    multiplier = 2.0 / (period + 1.0)
    previous_ema = np.nan

    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue

        if not np.isfinite(previous_ema):
            if index < period - 1:
                continue

            seed_values = values[index - period + 1 : index + 1]
            if not np.all(np.isfinite(seed_values)):
                continue

            previous_ema = float(np.mean(seed_values))
        else:
            previous_ema = value * multiplier + previous_ema * (1.0 - multiplier)

        result[index] = previous_ema

    return result


def rsi_14(close):
    result = np.full(len(close), np.nan, dtype=np.float64)

    for index in range(14, len(close)):
        changes = np.diff(close[index - 14 : index + 1])
        gains = np.where(changes > 0, changes, 0.0).sum()
        losses = np.where(changes < 0, -changes, 0.0).sum()
        average_gain = gains / 14.0
        average_loss = losses / 14.0

        if average_loss < EPSILON:
            result[index] = 100.0
        else:
            relative_strength = average_gain / average_loss
            result[index] = 100.0 - 100.0 / (1.0 + relative_strength)

    return result


def atr_14(high, low, close):
    true_ranges = np.zeros(len(close), dtype=np.float64)
    result = np.full(len(close), np.nan, dtype=np.float64)

    for index in range(len(close)):
        previous_close = close[index - 1] if index > 0 else close[index]
        true_ranges[index] = max(
            high[index] - low[index],
            abs(high[index] - previous_close),
            abs(low[index] - previous_close),
        )

        if index >= 13:
            result[index] = np.mean(true_ranges[index - 13 : index + 1])

    return result


def rolling_return_volatility_20(close):
    returns = np.full(len(close), np.nan, dtype=np.float64)
    result = np.full(len(close), np.nan, dtype=np.float64)

    for index in range(1, len(close)):
        returns[index] = safe_ratio(close[index] - close[index - 1], close[index - 1])

    for index in range(20, len(close)):
        result[index] = np.std(returns[index - 19 : index + 1])

    return result


def build_indicators(candles):
    close = candles["close"].to_numpy(dtype=np.float64)
    high = candles["high"].to_numpy(dtype=np.float64)
    low = candles["low"].to_numpy(dtype=np.float64)

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26

    return {
        "ema20": ema(close, 20),
        "ema50": ema(close, 50),
        "macd_line": macd_line,
        "macd_signal": ema(macd_line, 9),
        "rsi14": rsi_14(close),
        "atr14": atr_14(high, low, close),
        "rolling_volatility20": rolling_return_volatility_20(close),
    }


def has_enough_feature_history(index, indicators):
    if index < FIRST_VALID_FEATURE_INDEX:
        return False

    return all(
        np.isfinite(indicators[name][index])
        for name in [
            "ema20",
            "ema50",
            "macd_line",
            "macd_signal",
            "rsi14",
            "atr14",
            "rolling_volatility20",
        ]
    )


def features_for_candle(candles, arrays, indicators, index):
    if not has_enough_feature_history(index, indicators):
        return None

    close = arrays["close"]
    high = arrays["high"]
    low = arrays["low"]
    volume = arrays["volume"]
    taker_buy_base_volume = arrays["takerBuyBaseVolume"]
    number_of_trades = arrays["numberOfTrades"]

    candle_range = high[index] - low[index]
    rolling_volume = volume[index - 19 : index + 1]
    rolling_high = np.max(high[index - 19 : index + 1])
    rolling_low = np.min(low[index - 19 : index + 1])
    volume_average_20 = np.mean(rolling_volume)
    volume_std_20 = np.std(rolling_volume)
    taker_buy = taker_buy_base_volume[index]

    taker_buy_ratio = (
        taker_buy / max(volume[index], EPSILON) if np.isfinite(taker_buy) else 0.5
    )
    trade_count_change = (
        safe_ratio(
            number_of_trades[index] - number_of_trades[index - 1],
            number_of_trades[index - 1],
        )
        if np.isfinite(number_of_trades[index])
        and np.isfinite(number_of_trades[index - 1])
        else 0.0
    )

    return [
        clip(safe_ratio(close[index] - close[index - 1], close[index - 1])),
        clip(safe_ratio(close[index] - close[index - 3], close[index - 3])),
        clip(safe_ratio(close[index] - close[index - 5], close[index - 5])),
        clip(safe_ratio(close[index] - close[index - 12], close[index - 12])),
        clip((indicators["rsi14"][index] - 50.0) / 50.0, -1.0, 1.0),
        clip(safe_ratio(close[index] - indicators["ema20"][index], close[index])),
        clip(safe_ratio(close[index] - indicators["ema50"][index], close[index])),
        clip(safe_ratio(indicators["macd_line"][index], close[index])),
        clip(safe_ratio(indicators["macd_signal"][index], close[index])),
        clip(
            safe_ratio(
                indicators["macd_line"][index] - indicators["macd_signal"][index],
                close[index],
            )
        ),
        clip(safe_ratio(volume[index] - volume[index - 1], volume[index - 1])),
        clip(safe_ratio(volume[index], volume_average_20), 0.0, 10.0),
        clip(safe_ratio(volume[index] - volume_average_20, volume_std_20)),
        clip(safe_ratio(candle_range, close[index]), 0.0, 1.0),
        clip(safe_ratio(indicators["atr14"][index], close[index]), 0.0, 1.0),
        clip(indicators["rolling_volatility20"][index], 0.0, 1.0),
        clip((close[index] - low[index]) / max(candle_range, EPSILON), 0.0, 1.0),
        clip(safe_ratio(close[index] - rolling_high, close[index])),
        clip(safe_ratio(close[index] - rolling_low, close[index])),
        clip(taker_buy_ratio, 0.0, 1.0),
        clip(1.0 - taker_buy_ratio, 0.0, 1.0),
        clip(trade_count_change),
    ]


def three_class_target(close, prediction_index):
    current_close = close[prediction_index - 1]
    future_close = close[prediction_index + HORIZON - 1]
    future_return = safe_ratio(future_close - current_close, current_close)

    if future_return > MIN_TARGET_MOVE:
        return CLASS_UP, future_return

    if future_return < -MIN_TARGET_MOVE:
        return CLASS_DOWN, future_return

    return CLASS_NEUTRAL, future_return


def build_feature_window(arrays, indicators, prediction_index):
    window_start = prediction_index - LOOKBACK
    window_end = prediction_index - 1

    if window_start < FIRST_VALID_FEATURE_INDEX:
        return None

    rows = []
    for index in range(window_start, window_end + 1):
        features = features_for_candle(None, arrays, indicators, index)
        if features is None:
            return None
        rows.append(features)

    return np.asarray(rows, dtype=np.float32)


def create_arrays(candles):
    return {
        "open": candles["open"].to_numpy(dtype=np.float64),
        "high": candles["high"].to_numpy(dtype=np.float64),
        "low": candles["low"].to_numpy(dtype=np.float64),
        "close": candles["close"].to_numpy(dtype=np.float64),
        "volume": candles["volume"].to_numpy(dtype=np.float64),
        "numberOfTrades": candles["numberOfTrades"].to_numpy(dtype=np.float64),
        "takerBuyBaseVolume": candles["takerBuyBaseVolume"].to_numpy(dtype=np.float64),
    }


def build_examples(
    candles,
    prediction_indexes,
    arrays=None,
    indicators=None,
    regime_context=None,
):
    arrays = arrays or create_arrays(candles)
    indicators = indicators or build_indicators(candles)
    inputs = []
    classes = []
    returns = []
    used_indexes = []

    for prediction_index in prediction_indexes:
        actual_class, future_return = three_class_target(
            arrays["close"],
            prediction_index,
        )
        feature_window = build_feature_window(arrays, indicators, prediction_index)
        if feature_window is None:
            continue

        flattened_window = feature_window.reshape(-1)

        if regime_context is not None:
            regime_features = regime_features_for_prediction(
                candles,
                prediction_index,
                regime_context,
            )
            flattened_window = np.concatenate(
                [flattened_window, regime_features],
            )

        inputs.append(flattened_window)
        classes.append(actual_class)
        returns.append(future_return)
        used_indexes.append(prediction_index)

    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(classes, dtype=np.int64),
        np.asarray(returns, dtype=np.float32),
        np.asarray(used_indexes, dtype=np.int64),
    )


def create_prediction_indexes(candles):
    first_valid = LOOKBACK + FIRST_VALID_FEATURE_INDEX
    last_valid = len(candles) - HORIZON
    return np.arange(first_valid, last_valid + 1, dtype=np.int64)


def take_most_recent(values, max_count):
    if max_count == 0:
        return values
    return values[-min(max_count, len(values)) :]


def class_distribution(classes):
    total = len(classes)
    counts = {
        class_id: int(np.sum(classes == class_id))
        for class_id in [CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP]
    }
    percentages = {
        class_id: counts[class_id] / total if total else 0.0
        for class_id in counts
    }
    return counts, percentages


def print_class_distribution(name, classes):
    counts, percentages = class_distribution(classes)
    print(f"{name} class distribution:")
    for class_id in [CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP]:
        print(
            f"- class {class_id} ({CLASS_NAMES[class_id]}): "
            f"{counts[class_id]} ({percentages[class_id]:.2%})"
        )


def balanced_sample_indexes(indexes, classes, max_samples, rng):
    if max_samples == 0 or max_samples >= len(indexes):
        return indexes

    per_class = max(1, max_samples // 3)
    selected = []

    for class_id in [CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP]:
        class_indexes = indexes[classes == class_id]
        if len(class_indexes) == 0:
            continue

        sample_count = min(per_class, len(class_indexes))
        selected.extend(rng.choice(class_indexes, size=sample_count, replace=False))

    remaining_slots = max_samples - len(selected)
    if remaining_slots > 0:
        selected_set = set(int(value) for value in selected)
        remaining = np.asarray(
            [index for index in indexes if int(index) not in selected_set],
            dtype=np.int64,
        )
        if len(remaining) > 0:
            selected.extend(
                rng.choice(
                    remaining,
                    size=min(remaining_slots, len(remaining)),
                    replace=False,
                )
            )

    return np.asarray(rng.permutation(selected), dtype=np.int64)


def initialize_model(input_size, hidden_units, rng):
    scale_1 = np.sqrt(2.0 / input_size)
    scale_2 = np.sqrt(2.0 / hidden_units)
    return {
        "w1": rng.normal(0.0, scale_1, size=(input_size, hidden_units)).astype(np.float32),
        "b1": np.zeros(hidden_units, dtype=np.float32),
        "w2": rng.normal(0.0, scale_2, size=(hidden_units, 3)).astype(np.float32),
        "b2": np.zeros(3, dtype=np.float32),
    }


def forward(model, x):
    hidden_pre = x @ model["w1"] + model["b1"]
    hidden = np.maximum(hidden_pre, 0.0)
    logits = hidden @ model["w2"] + model["b2"]
    probabilities = softmax(logits)
    return hidden_pre, hidden, probabilities


def save_trained_model(model, feature_mean, feature_std, best_epoch, validation_loss):
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": SYMBOL,
        "model_type": "numpy_dense_3_class_old_5m",
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "min_target_move": MIN_TARGET_MOVE,
        "feature_names": FEATURE_NAMES,
        "first_valid_feature_index": FIRST_VALID_FEATURE_INDEX,
        "use_regime_features": USE_REGIME_FEATURES,
        "regime_timeframe": REGIME_TIMEFRAME,
        "regime_feature_names": REGIME_FEATURE_NAMES if USE_REGIME_FEATURES else [],
        "feature_mean": feature_mean.reshape(-1).tolist(),
        "feature_std": feature_std.reshape(-1).tolist(),
        "model": {name: value.tolist() for name, value in model.items()},
        "best_epoch": int(best_epoch),
        "validation_loss": float(validation_loss),
        "class_names": CLASS_NAMES,
    }

    with MODEL_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        import json

        json.dump(payload, handle, indent=2)


def sparse_cross_entropy(probabilities, classes):
    probabilities = np.clip(probabilities, EPSILON, 1.0)
    return float(-np.mean(np.log(probabilities[np.arange(len(classes)), classes])))


def train_dense_model(x_train, y_train, x_validation, y_validation):
    rng = np.random.default_rng(RANDOM_SEED)
    model = initialize_model(x_train.shape[1], PYTHON_HIDDEN_UNITS, rng)
    adam = {
        name: {
            "m": np.zeros_like(value, dtype=np.float32),
            "v": np.zeros_like(value, dtype=np.float32),
        }
        for name, value in model.items()
    }
    beta_1 = 0.9
    beta_2 = 0.999
    step = 0
    best_validation_loss = float("inf")
    best_model = copy.deepcopy(model)
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, PYTHON_EPOCHS + 1):
        order = rng.permutation(len(x_train))

        for start in range(0, len(order), PYTHON_BATCH_SIZE):
            batch_indexes = order[start : start + PYTHON_BATCH_SIZE]
            x_batch = x_train[batch_indexes]
            y_batch = y_train[batch_indexes]
            batch_size = len(x_batch)
            hidden_pre, hidden, probabilities = forward(model, x_batch)

            d_logits = probabilities.copy()
            d_logits[np.arange(batch_size), y_batch] -= 1.0
            d_logits /= batch_size
            gradients = {
                "w2": hidden.T @ d_logits,
                "b2": np.sum(d_logits, axis=0),
            }
            d_hidden = d_logits @ model["w2"].T
            d_hidden[hidden_pre <= 0] = 0.0
            gradients["w1"] = x_batch.T @ d_hidden
            gradients["b1"] = np.sum(d_hidden, axis=0)

            step += 1
            for name in model:
                adam[name]["m"] = beta_1 * adam[name]["m"] + (1.0 - beta_1) * gradients[name]
                adam[name]["v"] = beta_2 * adam[name]["v"] + (1.0 - beta_2) * (gradients[name] ** 2)
                m_hat = adam[name]["m"] / (1.0 - beta_1**step)
                v_hat = adam[name]["v"] / (1.0 - beta_2**step)
                model[name] -= PYTHON_LEARNING_RATE * m_hat / (np.sqrt(v_hat) + EPSILON)

        _, _, train_probabilities = forward(model, x_train)
        _, _, validation_probabilities = forward(model, x_validation)
        train_loss = sparse_cross_entropy(train_probabilities, y_train)
        validation_loss = sparse_cross_entropy(validation_probabilities, y_validation)
        validation_accuracy = float(
            np.mean(np.argmax(validation_probabilities, axis=1) == y_validation)
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"validation loss {validation_loss:.4f} | "
            f"validation accuracy {validation_accuracy:.2%}"
        )

        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PYTHON_EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping after epoch {epoch}. "
                f"Best validation epoch: {best_epoch}."
            )
            break

    return best_model, best_epoch, best_validation_loss


def main():
    candles, data_path = load_candles()
    all_indexes = create_prediction_indexes(candles)
    split_candle_index = int(len(candles) * TRAIN_SPLIT)
    training_indexes = all_indexes[all_indexes + HORIZON - 1 < split_candle_index]
    validation_indexes = all_indexes[all_indexes >= split_candle_index]
    arrays = create_arrays(candles)
    indicators = build_indicators(candles)
    regime_context = load_regime_features()
    close = arrays["close"]

    print("Python 3-class training alternative")
    print(f"Symbol: {SYMBOL}")
    print(f"Input data path: {data_path}")
    print(f"Prediction output path: {PREDICTIONS_PATH}")
    print(f"Rows loaded: {len(candles)}")
    print(f"First timestamp: {candles.iloc[0]['time']}")
    print(f"Last timestamp: {candles.iloc[-1]['time']}")
    print(f"Lookback: {LOOKBACK}")
    print(f"Horizon: {HORIZON} candles ({HORIZON * 5} minutes)")
    print(f"Feature count: {len(FEATURE_NAMES)}")
    print(f"Regime features enabled: {USE_REGIME_FEATURES}")
    print(f"Regime timeframe: {REGIME_TIMEFRAME}")
    print(
        f"Regime feature count: "
        f"{len(REGIME_FEATURE_NAMES) if USE_REGIME_FEATURES else 0}"
    )
    if USE_REGIME_FEATURES:
        print(f"Regime file: {regime_context['path'] if regime_context else REGIME_PATH}")
        print(
            f"Regime rows loaded: "
            f"{len(regime_context['close_timestamps']) if regime_context else 0}"
        )
    print(f"Minimum target move: {MIN_TARGET_MOVE:.2%}")
    print(f"Total possible windows: {len(all_indexes)}")

    all_classes = np.asarray(
        [three_class_target(close, index)[0] for index in all_indexes],
        dtype=np.int64,
    )
    print_class_distribution("All eligible windows", all_classes)

    training_classes = np.asarray(
        [three_class_target(close, index)[0] for index in training_indexes],
        dtype=np.int64,
    )
    validation_classes = np.asarray(
        [three_class_target(close, index)[0] for index in validation_indexes],
        dtype=np.int64,
    )
    validation_indexes_used = take_most_recent(
        validation_indexes,
        VALIDATION_MAX_WINDOWS,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    sampled_training_indexes = balanced_sample_indexes(
        training_indexes,
        training_classes,
        MAX_TRAIN_SAMPLES,
        rng,
    )

    print(f"Training windows available: {len(training_indexes)}")
    print(f"Sampled training windows: {len(sampled_training_indexes)}")
    print(f"Validation windows available: {len(validation_indexes)}")
    print(f"Validation windows used: {len(validation_indexes_used)}")

    x_train, y_train, _, _ = build_examples(
        candles,
        sampled_training_indexes,
        arrays,
        indicators,
        regime_context,
    )
    x_validation, y_validation, validation_returns, validation_used_indexes = build_examples(
        candles,
        validation_indexes_used,
        arrays,
        indicators,
        regime_context,
    )

    if len(x_train) == 0 or len(x_validation) == 0:
        raise RuntimeError("Not enough windows after feature generation.")

    rows_without_regime_context = (
        regime_context["rows_without_context"] if regime_context else 0
    )
    print(f"Rows without available regime context: {rows_without_regime_context}")
    print(f"Final flattened input size: {x_train.shape[1]}")

    print_class_distribution("Training sample", y_train)
    print_class_distribution("Validation", y_validation)

    feature_mean = x_train.mean(axis=0, keepdims=True)
    feature_std = x_train.std(axis=0, keepdims=True)
    feature_std[feature_std < EPSILON] = 1.0
    x_train_scaled = (x_train - feature_mean) / feature_std
    x_validation_scaled = (x_validation - feature_mean) / feature_std

    best_model, best_epoch, best_validation_loss = train_dense_model(
        x_train_scaled,
        y_train,
        x_validation_scaled,
        y_validation,
    )
    _, _, validation_probabilities = forward(best_model, x_validation_scaled)

    output = pd.DataFrame(
        {
            "time": candles.iloc[validation_used_indexes - 1]["time"].to_numpy(),
            "timestamp": candles.iloc[validation_used_indexes - 1][
                "timestamp"
            ].to_numpy(),
            "prob_down": validation_probabilities[:, CLASS_DOWN],
            "prob_neutral": validation_probabilities[:, CLASS_NEUTRAL],
            "prob_up": validation_probabilities[:, CLASS_UP],
            "actual_class": y_validation.astype(int),
            "future_return": validation_returns,
        }
    )
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS_PATH, index=False)
    save_trained_model(
        best_model,
        feature_mean,
        feature_std,
        best_epoch,
        best_validation_loss,
    )
    print(
        f"Saved best-epoch validation predictions to: {PREDICTIONS_PATH} "
        f"(epoch {best_epoch}, validation loss {best_validation_loss:.4f})"
    )
    print(f"Saved Python old 5m model to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
