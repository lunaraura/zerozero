import os
import copy
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = [
    value.strip().upper()
    for value in os.getenv("SYMBOLS", os.getenv("SYMBOL", "SOLUSDT")).split(",")
    if value.strip()
]
TARGET_SYMBOL = os.getenv("TARGET_SYMBOL", "SOLUSDT").strip().upper()
FINE_TUNE_SYMBOL = os.getenv("FINE_TUNE_SYMBOL", "").strip().upper()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

PREDICTIONS_PATH = OUTPUT_DIR / f"{TARGET_SYMBOL}_1m_flow_predictions.csv"

LOOKBACK = int(os.getenv("LOOKBACK", "30"))
FORECAST_HORIZONS = [
    int(value.strip())
    for value in os.getenv("FORECAST_HORIZONS", "1,2,3").split(",")
    if value.strip()
]
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.80"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
FINE_TUNE_EPOCHS = int(os.getenv("FINE_TUNE_EPOCHS", str(max(1, EPOCHS // 4))))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "0.003"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.0015"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))

EPSILON = 1e-8
CLASS_SHORT_WIN = 0
CLASS_NEUTRAL = 1
CLASS_LONG_WIN = 2
CLASS_NAMES = {
    CLASS_SHORT_WIN: "short_win",
    CLASS_NEUTRAL: "neutral",
    CLASS_LONG_WIN: "long_win",
}

TARGET_BASE_NAMES = [
    "future_return",
    "future_range_percent",
    "future_volume_zscore_20",
    "future_market_pressure",
    "future_order_book_imbalance_10bps",
    "future_imbalance_change_10bps",
    "future_spread_percent",
    "future_breakout_pressure_index",
    "future_absorption_index",
]

FUTURE_SOURCE_COLUMNS = {
    "future_range_percent": "range_percent",
    "future_volume_zscore_20": "volume_zscore_20",
    "future_market_pressure": "market_pressure",
    "future_order_book_imbalance_10bps": "order_book_imbalance_10bps",
    "future_imbalance_change_10bps": "order_book_imbalance_10bps_change",
    "future_spread_percent": "spread_percent",
    "future_breakout_pressure_index": "breakout_pressure_index",
    "future_absorption_index": "absorption_index",
}

EXCLUDED_INPUT_COLUMNS = {
    "time",
    "timestamp",
    "feature_ready",
    # Raw price-level columns do not transfer well across symbols. The model
    # can still use returns, ranges, spread percentages, and other normalized
    # distance/pressure features derived from those raw prices.
    "open",
    "high",
    "low",
    "close",
    "best_bid",
    "best_ask",
    "mid_price",
    "actual_class",
    "pred_class",
    "prob_short",
    "prob_neutral",
    "prob_long",
}

EXCLUDED_INPUT_PREFIXES = (
    "future_",
    "actual_",
    "pred_",
    "prob_",
)


def safe_ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return 0.0
    if abs(denominator) < EPSILON:
        return 0.0
    return float(numerator / denominator)


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(logits, -40, 40))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def get_feature_path(symbol):
    return OUTPUT_DIR / f"{symbol}_1m_flow_features.csv"


def load_feature_rows(symbol):
    input_path = get_feature_path(symbol)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_path}. Run scripts/build_1m_flow_features.py first."
        )

    frame = pd.read_csv(input_path)
    required = [
        "timestamp",
        "time",
        "close",
        "high",
        "low",
        "feature_ready",
        "range_percent",
        "volume_zscore_20",
        "market_pressure",
        "order_book_imbalance_10bps",
        "order_book_imbalance_10bps_change",
        "bid_depth_change_ratio_10bps",
        "ask_depth_change_ratio_10bps",
        "spread_percent",
        "breakout_pressure_index",
        "absorption_index",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature CSV is missing required columns: {missing}")

    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    frame = frame.reset_index(drop=True)

    for column in frame.columns:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame["feature_ready"].dtype != bool:
        frame["feature_ready"] = frame["feature_ready"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )

    return frame, input_path


def choose_input_feature_columns(frame):
    feature_columns = []

    for column in frame.columns:
        if column in EXCLUDED_INPUT_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in EXCLUDED_INPUT_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            feature_columns.append(column)

    return feature_columns


def create_regression_target(frame, current_index, horizon):
    future_index = current_index + horizon
    current_close = frame.loc[current_index, "close"]
    future_close = frame.loc[future_index, "close"]
    target_values = [
        safe_ratio(future_close - current_close, current_close),
    ]

    for target_name in TARGET_BASE_NAMES[1:]:
        source_column = FUTURE_SOURCE_COLUMNS[target_name]
        target_values.append(float(frame.loc[future_index, source_column]))

    return target_values


def create_trade_path_class(frame, current_index):
    entry = frame.loc[current_index, "close"]
    long_take_profit = entry * (1.0 + TAKE_PROFIT)
    long_stop_loss = entry * (1.0 - STOP_LOSS)
    short_take_profit = entry * (1.0 - TAKE_PROFIT)
    short_stop_loss = entry * (1.0 + STOP_LOSS)

    for step in range(1, 4):
        future_index = current_index + step
        if future_index >= len(frame):
            break

        high = frame.loc[future_index, "high"]
        low = frame.loc[future_index, "low"]
        long_tp_hit = high >= long_take_profit
        long_sl_hit = low <= long_stop_loss
        short_tp_hit = low <= short_take_profit
        short_sl_hit = high >= short_stop_loss

        # With 1m OHLC rows we cannot know intraminute order when both sides
        # trigger in the same candle, so ambiguous paths stay neutral.
        if long_tp_hit and not long_sl_hit and not short_tp_hit:
            return CLASS_LONG_WIN
        if short_tp_hit and not short_sl_hit and not long_tp_hit:
            return CLASS_SHORT_WIN
        if long_sl_hit or short_sl_hit:
            return CLASS_NEUTRAL

    return CLASS_NEUTRAL


def create_target_names():
    names = []
    for horizon in FORECAST_HORIZONS:
        for base_name in TARGET_BASE_NAMES:
            names.append(f"{base_name}_{horizon}")
    return names


def build_examples(frame, feature_columns, symbol_index, symbol_count, symbol):
    max_horizon = max(FORECAST_HORIZONS)
    inputs = []
    regression_targets = []
    classes = []
    times = []
    timestamps = []
    symbols = []
    symbol_one_hot = np.zeros(symbol_count, dtype=np.float64)
    symbol_one_hot[symbol_index] = 1.0

    for current_index in range(LOOKBACK - 1, len(frame) - max_horizon):
        lookback_start = current_index - LOOKBACK + 1
        window = frame.iloc[lookback_start : current_index + 1]

        if not bool(window["feature_ready"].all()):
            continue

        input_window = window[feature_columns]
        if input_window.isna().any().any():
            continue
        numeric_window = input_window.to_numpy(dtype=np.float64)
        one_hot_window = np.tile(symbol_one_hot, (LOOKBACK, 1))
        model_window = np.hstack([numeric_window, one_hot_window])

        target_values = []
        valid_target = True
        for horizon in FORECAST_HORIZONS:
            horizon_target = create_regression_target(frame, current_index, horizon)
            if not np.all(np.isfinite(horizon_target)):
                valid_target = False
                break
            target_values.extend(horizon_target)

        if not valid_target:
            continue

        inputs.append(model_window.reshape(-1))
        regression_targets.append(target_values)
        classes.append(create_trade_path_class(frame, current_index))
        times.append(frame.loc[current_index, "time"])
        timestamps.append(frame.loc[current_index, "timestamp"])
        symbols.append(symbol)

    return {
        "inputs": np.asarray(inputs, dtype=np.float64),
        "regression_targets": np.asarray(regression_targets, dtype=np.float64),
        "classes": np.asarray(classes, dtype=np.int64),
        "times": np.asarray(times),
        "timestamps": np.asarray(timestamps),
        "symbols": np.asarray(symbols),
    }


def split_time_ordered(examples):
    split_index = int(len(examples["inputs"]) * TRAIN_SPLIT)
    train = {}
    validation = {}

    for key, values in examples.items():
        train[key] = values[:split_index]
        validation[key] = values[split_index:]

    return train, validation


def empty_examples():
    return {
        "inputs": np.empty((0, 0), dtype=np.float64),
        "regression_targets": np.empty((0, 0), dtype=np.float64),
        "classes": np.empty((0,), dtype=np.int64),
        "times": np.asarray([]),
        "timestamps": np.asarray([]),
        "symbols": np.asarray([]),
    }


def combine_example_sets(example_sets):
    non_empty_sets = [items for items in example_sets if len(items["inputs"]) > 0]

    if not non_empty_sets:
        return empty_examples()

    return {
        "inputs": np.vstack([items["inputs"] for items in non_empty_sets]),
        "regression_targets": np.vstack(
            [items["regression_targets"] for items in non_empty_sets]
        ),
        "classes": np.concatenate([items["classes"] for items in non_empty_sets]),
        "times": np.concatenate([items["times"] for items in non_empty_sets]),
        "timestamps": np.concatenate([items["timestamps"] for items in non_empty_sets]),
        "symbols": np.concatenate([items["symbols"] for items in non_empty_sets]),
    }


def intersect_feature_columns(frames_by_symbol):
    feature_columns_by_symbol = {
        symbol: choose_input_feature_columns(frame)
        for symbol, frame in frames_by_symbol.items()
    }
    common_columns = set.intersection(
        *[set(columns) for columns in feature_columns_by_symbol.values()]
    )
    first_symbol = next(iter(frames_by_symbol.keys()))

    return [
        column
        for column in feature_columns_by_symbol[first_symbol]
        if column in common_columns
    ]


def standardize(train_values, values):
    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True)
    std[std < EPSILON] = 1.0
    return (values - mean) / std, mean, std


def apply_standardization(values, mean, std):
    return (values - mean) / std


def create_numeric_input_mask(feature_count, symbol_count):
    row_width = feature_count + symbol_count
    mask = np.zeros(LOOKBACK * row_width, dtype=bool)

    for row_index in range(LOOKBACK):
        start = row_index * row_width
        mask[start : start + feature_count] = True

    return mask


def standardize_inputs(train_inputs, values, numeric_mask):
    mean = np.zeros((1, train_inputs.shape[1]), dtype=np.float64)
    std = np.ones((1, train_inputs.shape[1]), dtype=np.float64)
    mean[:, numeric_mask] = train_inputs[:, numeric_mask].mean(axis=0, keepdims=True)
    std[:, numeric_mask] = train_inputs[:, numeric_mask].std(axis=0, keepdims=True)
    std[:, numeric_mask] = np.where(std[:, numeric_mask] < EPSILON, 1.0, std[:, numeric_mask])

    return (values - mean) / std, mean, std


def one_hot(classes):
    result = np.zeros((len(classes), 3), dtype=np.float64)
    result[np.arange(len(classes)), classes] = 1.0
    return result


def initialize_model(input_size, regression_output_size, rng):
    hidden_1 = 128
    hidden_2 = 64
    return {
        "w1": rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_1)),
        "b1": np.zeros(hidden_1),
        "w2": rng.normal(0, np.sqrt(2 / hidden_1), (hidden_1, hidden_2)),
        "b2": np.zeros(hidden_2),
        "w_reg": rng.normal(0, np.sqrt(2 / hidden_2), (hidden_2, regression_output_size)),
        "b_reg": np.zeros(regression_output_size),
        "w_cls": rng.normal(0, np.sqrt(2 / hidden_2), (hidden_2, 3)),
        "b_cls": np.zeros(3),
    }


def forward(model, x):
    z1 = x @ model["w1"] + model["b1"]
    h1 = np.maximum(z1, 0)
    z2 = h1 @ model["w2"] + model["b2"]
    h2 = np.maximum(z2, 0)
    regression = h2 @ model["w_reg"] + model["b_reg"]
    class_logits = h2 @ model["w_cls"] + model["b_cls"]
    probabilities = softmax(class_logits)
    cache = {
        "x": x,
        "z1": z1,
        "h1": h1,
        "z2": z2,
        "h2": h2,
        "regression": regression,
        "probabilities": probabilities,
    }
    return regression, probabilities, cache


def train_model(
    x_train,
    y_reg_train,
    y_cls_train,
    x_val,
    y_reg_val,
    y_cls_val,
    initial_model=None,
    epochs=EPOCHS,
    label="training",
):
    rng = np.random.default_rng(RANDOM_SEED)
    model = (
        copy.deepcopy(initial_model)
        if initial_model is not None
        else initialize_model(x_train.shape[1], y_reg_train.shape[1], rng)
    )
    adam = {
        name: {
            "m": np.zeros_like(value),
            "v": np.zeros_like(value),
        }
        for name, value in model.items()
    }
    beta_1 = 0.9
    beta_2 = 0.999
    step = 0
    best_model = copy.deepcopy(model)
    best_val_loss = float("inf")
    regression_weight = 1.0
    class_weight = 1.0

    for epoch in range(1, epochs + 1):
        epoch_losses = []

        for start in range(0, len(x_train), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(x_train))
            x_batch = x_train[start:end]
            y_reg_batch = y_reg_train[start:end]
            y_cls_batch = y_cls_train[start:end]
            batch_size = len(x_batch)

            # Sequential prequential-style feed: first predict this small
            # time-ordered batch, then train on that same batch. No random
            # shuffling is used before or during training.
            reg_pred, probabilities, cache = forward(model, x_batch)
            y_cls_one_hot = one_hot(y_cls_batch)
            reg_error = reg_pred - y_reg_batch
            reg_loss = np.mean(reg_error**2)
            cls_loss = -np.mean(
                np.log(np.clip(probabilities[np.arange(batch_size), y_cls_batch], EPSILON, 1.0))
            )
            loss = regression_weight * reg_loss + class_weight * cls_loss
            epoch_losses.append(loss)

            d_reg = (2.0 / batch_size) * reg_error / y_reg_batch.shape[1]
            d_logits = (probabilities - y_cls_one_hot) / batch_size

            gradients = {
                "w_reg": cache["h2"].T @ d_reg,
                "b_reg": d_reg.sum(axis=0),
                "w_cls": cache["h2"].T @ d_logits,
                "b_cls": d_logits.sum(axis=0),
            }

            d_h2 = d_reg @ model["w_reg"].T + d_logits @ model["w_cls"].T
            d_z2 = d_h2.copy()
            d_z2[cache["z2"] <= 0] = 0
            gradients["w2"] = cache["h1"].T @ d_z2
            gradients["b2"] = d_z2.sum(axis=0)

            d_h1 = d_z2 @ model["w2"].T
            d_z1 = d_h1.copy()
            d_z1[cache["z1"] <= 0] = 0
            gradients["w1"] = cache["x"].T @ d_z1
            gradients["b1"] = d_z1.sum(axis=0)

            step += 1
            for name in model:
                adam[name]["m"] = beta_1 * adam[name]["m"] + (1 - beta_1) * gradients[name]
                adam[name]["v"] = beta_2 * adam[name]["v"] + (1 - beta_2) * (gradients[name] ** 2)
                m_hat = adam[name]["m"] / (1 - beta_1**step)
                v_hat = adam[name]["v"] / (1 - beta_2**step)
                model[name] -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + EPSILON)

        val_reg_pred, val_prob, _ = forward(model, x_val)
        val_reg_loss = np.mean((val_reg_pred - y_reg_val) ** 2)
        val_cls_loss = -np.mean(
            np.log(np.clip(val_prob[np.arange(len(y_cls_val)), y_cls_val], EPSILON, 1.0))
        )
        val_loss = val_reg_loss + val_cls_loss
        val_accuracy = np.mean(np.argmax(val_prob, axis=1) == y_cls_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print(
            f"{label} epoch {epoch:02d} | "
            f"train loss {np.mean(epoch_losses):.4f} | "
            f"val reg mse {val_reg_loss:.4f} | "
            f"val class loss {val_cls_loss:.4f} | "
            f"val accuracy {val_accuracy:.2%}"
        )

    return best_model


def class_distribution(classes):
    return {
        class_id: int(np.sum(classes == class_id))
        for class_id in [CLASS_SHORT_WIN, CLASS_NEUTRAL, CLASS_LONG_WIN]
    }


def print_class_distribution(name, classes):
    counts = class_distribution(classes)
    total = len(classes) or 1
    print(f"{name} class distribution:")
    for class_id in [CLASS_SHORT_WIN, CLASS_NEUTRAL, CLASS_LONG_WIN]:
        print(
            f"- class {class_id} {CLASS_NAMES[class_id]}: "
            f"{counts[class_id]} ({counts[class_id] / total:.2%})"
        )


def save_validation_predictions(validation, y_reg_actual, y_reg_pred, probabilities, target_names):
    pred_class = np.argmax(probabilities, axis=1)
    output = pd.DataFrame(
        {
            "time": validation["times"],
            "timestamp": validation["timestamps"],
            "actual_class": validation["classes"],
            "pred_class": pred_class,
            "prob_short": probabilities[:, CLASS_SHORT_WIN],
            "prob_neutral": probabilities[:, CLASS_NEUTRAL],
            "prob_long": probabilities[:, CLASS_LONG_WIN],
        }
    )

    target_index = {name: index for index, name in enumerate(target_names)}

    for horizon in FORECAST_HORIZONS:
        name = f"future_return_{horizon}"
        output[f"actual_return_{horizon}"] = y_reg_actual[:, target_index[name]]
        output[f"pred_return_{horizon}"] = y_reg_pred[:, target_index[name]]

    selected_pairs = [
        ("future_market_pressure_1", "actual_market_pressure_1", "pred_market_pressure_1"),
        (
            "future_order_book_imbalance_10bps_1",
            "actual_imbalance_10bps_1",
            "pred_imbalance_10bps_1",
        ),
        (
            "future_breakout_pressure_index_1",
            "actual_breakout_pressure_index_1",
            "pred_breakout_pressure_index_1",
        ),
        (
            "future_absorption_index_1",
            "actual_absorption_index_1",
            "pred_absorption_index_1",
        ),
    ]

    for target_name, actual_column, predicted_column in selected_pairs:
        if target_name in target_index:
            output[actual_column] = y_reg_actual[:, target_index[target_name]]
            output[predicted_column] = y_reg_pred[:, target_index[target_name]]

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS_PATH, index=False)


def print_regression_mae(y_actual, y_pred, target_names):
    mae = np.mean(np.abs(y_pred - y_actual), axis=0)
    print("\nValidation regression MAE per target:")
    for name, value in zip(target_names, mae):
        print(f"- {name}: {value:.10g}")


def print_probability_diagnostics(classes, probabilities):
    print("\nAverage predicted probability for winning vs losing classes:")
    short_rows = classes == CLASS_SHORT_WIN
    neutral_rows = classes == CLASS_NEUTRAL
    long_rows = classes == CLASS_LONG_WIN

    def avg(mask, column):
        if not np.any(mask):
            return 0.0
        return float(np.mean(probabilities[mask, column]))

    print(f"- actual short_win rows avg prob_short: {avg(short_rows, CLASS_SHORT_WIN):.2%}")
    print(f"- actual neutral rows avg prob_neutral: {avg(neutral_rows, CLASS_NEUTRAL):.2%}")
    print(f"- actual long_win rows avg prob_long: {avg(long_rows, CLASS_LONG_WIN):.2%}")
    non_winning_rows = classes == CLASS_NEUTRAL
    print(
        "- neutral rows avg directional probability: "
        f"{float(np.mean(np.maximum(probabilities[non_winning_rows, CLASS_SHORT_WIN], probabilities[non_winning_rows, CLASS_LONG_WIN]))) if np.any(non_winning_rows) else 0.0:.2%}"
    )


def main():
    if len(FORECAST_HORIZONS) == 0:
        raise ValueError("FORECAST_HORIZONS must include at least one horizon.")

    symbols = []
    for symbol in [*SYMBOLS, TARGET_SYMBOL, FINE_TUNE_SYMBOL]:
        if symbol and symbol not in symbols:
            symbols.append(symbol)

    frames_by_symbol = {}
    paths_by_symbol = {}
    for symbol in symbols:
        frame, input_path = load_feature_rows(symbol)
        frames_by_symbol[symbol] = frame
        paths_by_symbol[symbol] = input_path

    feature_columns = intersect_feature_columns(frames_by_symbol)
    if not feature_columns:
        raise RuntimeError("No common numeric feature columns found across symbols.")

    examples_by_symbol = {}
    train_by_symbol = {}
    validation_by_symbol = {}
    for symbol_index, symbol in enumerate(symbols):
        examples = build_examples(
            frames_by_symbol[symbol],
            feature_columns,
            symbol_index,
            len(symbols),
            symbol,
        )
        train_split, validation_split = split_time_ordered(examples)
        examples_by_symbol[symbol] = examples
        train_by_symbol[symbol] = train_split
        validation_by_symbol[symbol] = validation_split

    train = combine_example_sets([train_by_symbol[symbol] for symbol in symbols])
    validation = combine_example_sets(
        [validation_by_symbol[symbol] for symbol in symbols]
    )
    target_validation = validation_by_symbol[TARGET_SYMBOL]

    if len(train["inputs"]) == 0 or len(validation["inputs"]) == 0:
        summary = ", ".join(
            f"{symbol}: {len(examples_by_symbol[symbol]['inputs'])} usable "
            f"({len(train_by_symbol[symbol]['inputs'])} train / "
            f"{len(validation_by_symbol[symbol]['inputs'])} validation)"
            for symbol in symbols
        )
        raise RuntimeError(
            "Combined train/validation split produced an empty segment. "
            f"Per-symbol examples: {summary}. "
            "Record more continuous 1m rows or temporarily lower LOOKBACK for a smoke test."
        )
    if len(target_validation["inputs"]) == 0:
        raise RuntimeError(
            f"TARGET_SYMBOL {TARGET_SYMBOL} has no validation examples to save."
        )

    numeric_mask = create_numeric_input_mask(len(feature_columns), len(symbols))
    x_train_scaled, x_mean, x_std = standardize_inputs(
        train["inputs"],
        train["inputs"],
        numeric_mask,
    )
    x_validation_scaled = apply_standardization(validation["inputs"], x_mean, x_std)
    x_target_validation_scaled = apply_standardization(
        target_validation["inputs"],
        x_mean,
        x_std,
    )
    y_reg_train_scaled, y_mean, y_std = standardize(
        train["regression_targets"],
        train["regression_targets"],
    )
    y_reg_validation_scaled = apply_standardization(
        validation["regression_targets"],
        y_mean,
        y_std,
    )
    y_reg_target_validation_scaled = apply_standardization(
        target_validation["regression_targets"],
        y_mean,
        y_std,
    )
    target_names = create_target_names()

    print("1m flow multi-output forecasting model")
    print(f"SYMBOLS: {symbols}")
    print(f"TARGET_SYMBOL: {TARGET_SYMBOL}")
    print(f"FINE_TUNE_SYMBOL: {FINE_TUNE_SYMBOL or 'none'}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Prediction output path: {PREDICTIONS_PATH}")
    print(f"LOOKBACK: {LOOKBACK}")
    print(f"Forecast horizons: {FORECAST_HORIZONS}")
    print(f"Numeric feature count per row: {len(feature_columns)}")
    print(f"Symbol one-hot count per row: {len(symbols)}")
    print(f"Total feature count per row: {len(feature_columns) + len(symbols)}")
    print(f"Flattened input size: {train['inputs'].shape[1]}")
    print(f"Regression output count: {len(target_names)}")
    print(f"Sequential predict-then-train feed batch size: {BATCH_SIZE}")
    print("Final regression targets:")
    for target_name in target_names:
        print(f"- {target_name}")
    print("Raw price-level inputs excluded: open, high, low, close, best_bid, best_ask, mid_price")
    print(f"Combined train examples: {len(train['inputs'])}")
    print(f"Combined validation examples: {len(validation['inputs'])}")
    print(f"Target validation examples: {len(target_validation['inputs'])}")
    print("\nPer-symbol dataset diagnostics:")
    for symbol in symbols:
        symbol_examples = examples_by_symbol[symbol]
        symbol_train = train_by_symbol[symbol]
        symbol_validation = validation_by_symbol[symbol]
        print(f"- {symbol}")
        print(f"  input path: {paths_by_symbol[symbol]}")
        print(f"  row count: {len(frames_by_symbol[symbol])}")
        print(f"  usable examples: {len(symbol_examples['inputs'])}")
        print(f"  train examples: {len(symbol_train['inputs'])}")
        print(f"  validation examples: {len(symbol_validation['inputs'])}")
        if len(symbol_train["times"]) > 0:
            print(
                f"  train range: {symbol_train['times'][0]} -> "
                f"{symbol_train['times'][-1]}"
            )
        if len(symbol_validation["times"]) > 0:
            print(
                f"  validation range: {symbol_validation['times'][0]} -> "
                f"{symbol_validation['times'][-1]}"
            )
    print(f"TAKE_PROFIT: {TAKE_PROFIT:.4%}")
    print(f"STOP_LOSS: {STOP_LOSS:.4%}")
    print("No trades are placed. Live data is not read directly.")
    print_class_distribution("Combined train", train["classes"])
    print_class_distribution("Combined validation", validation["classes"])
    print_class_distribution(
        f"{TARGET_SYMBOL} validation",
        target_validation["classes"],
    )

    model = train_model(
        x_train_scaled,
        y_reg_train_scaled,
        train["classes"],
        x_validation_scaled,
        y_reg_validation_scaled,
        validation["classes"],
        label="pretrain",
    )

    if FINE_TUNE_SYMBOL:
        fine_tune_train = train_by_symbol[FINE_TUNE_SYMBOL]
        fine_tune_validation = validation_by_symbol[FINE_TUNE_SYMBOL]
        if len(fine_tune_train["inputs"]) == 0 or len(fine_tune_validation["inputs"]) == 0:
            print(
                f"Skipping fine-tune: {FINE_TUNE_SYMBOL} lacks train or validation examples."
            )
        else:
            print(
                f"\nFine-tuning on {FINE_TUNE_SYMBOL} for {FINE_TUNE_EPOCHS} epochs."
            )
            fine_x_train = apply_standardization(
                fine_tune_train["inputs"],
                x_mean,
                x_std,
            )
            fine_y_train = apply_standardization(
                fine_tune_train["regression_targets"],
                y_mean,
                y_std,
            )
            fine_x_validation = apply_standardization(
                fine_tune_validation["inputs"],
                x_mean,
                x_std,
            )
            fine_y_validation = apply_standardization(
                fine_tune_validation["regression_targets"],
                y_mean,
                y_std,
            )
            model = train_model(
                fine_x_train,
                fine_y_train,
                fine_tune_train["classes"],
                fine_x_validation,
                fine_y_validation,
                fine_tune_validation["classes"],
                initial_model=model,
                epochs=FINE_TUNE_EPOCHS,
                label=f"fine-tune {FINE_TUNE_SYMBOL}",
            )

    y_reg_pred_scaled, probabilities, _ = forward(model, x_target_validation_scaled)
    y_reg_pred = y_reg_pred_scaled * y_std + y_mean
    predicted_class = np.argmax(probabilities, axis=1)
    validation_accuracy = np.mean(predicted_class == target_validation["classes"])

    print_regression_mae(
        target_validation["regression_targets"],
        y_reg_pred,
        target_names,
    )
    print(f"\n{TARGET_SYMBOL} validation class accuracy: {validation_accuracy:.2%}")
    print_probability_diagnostics(target_validation["classes"], probabilities)
    save_validation_predictions(
        target_validation,
        target_validation["regression_targets"],
        y_reg_pred,
        probabilities,
        target_names,
    )
    print(f"\nSaved validation predictions to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
