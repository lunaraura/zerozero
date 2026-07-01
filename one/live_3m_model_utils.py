import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


EPSILON = 1e-8
CLASS_SHORT = 0
CLASS_NEUTRAL = 1
CLASS_LONG = 2
CLASS_NAMES = {
    CLASS_SHORT: "short_win",
    CLASS_NEUTRAL: "neutral",
    CLASS_LONG: "long_win",
}

RAW_PRICE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "best_bid",
    "best_ask",
    "mid_price",
}

NON_FEATURE_COLUMNS = {
    "time",
    "timestamp",
    "feature_ready",
}

REGRESSION_TARGET_COLUMNS = [
    "actual_future_return_3",
    "actual_future_range_percent_3",
    "actual_future_volume_3m",
    "actual_future_trade_count_3m",
    "actual_market_buy_volume_3m",
    "actual_market_sell_volume_3m",
    "actual_volume_delta_3m",
    "actual_market_pressure_3m",
    "actual_order_book_imbalance_10bps_3m",
    "actual_bid_depth_change_10bps_3m",
    "actual_ask_depth_change_10bps_3m",
    "actual_spread_percent_3m",
    "actual_breakout_pressure_3m",
    "actual_absorption_3m",
]

PREDICTION_COLUMN_FOR_TARGET = {
    "actual_future_return_3": "pred_future_return_3",
    "actual_future_range_percent_3": "pred_future_range_percent_3",
    "actual_future_volume_3m": "pred_future_volume_3m",
    "actual_future_trade_count_3m": "pred_future_trade_count_3m",
    "actual_market_buy_volume_3m": "pred_future_market_buy_volume_3m",
    "actual_market_sell_volume_3m": "pred_future_market_sell_volume_3m",
    "actual_volume_delta_3m": "pred_future_volume_delta_3m",
    "actual_market_pressure_3m": "pred_future_market_pressure_3m",
    "actual_order_book_imbalance_10bps_3m": (
        "pred_future_order_book_imbalance_10bps_3m"
    ),
    "actual_bid_depth_change_10bps_3m": (
        "pred_future_bid_depth_change_10bps_3m"
    ),
    "actual_ask_depth_change_10bps_3m": (
        "pred_future_ask_depth_change_10bps_3m"
    ),
    "actual_spread_percent_3m": "pred_future_spread_percent_3m",
    "actual_breakout_pressure_3m": "pred_future_breakout_pressure_3m",
    "actual_absorption_3m": "pred_future_absorption_3m",
}

LIVE_PREDICTION_COLUMNS = [
    "timestamp",
    "time",
    "input_window_start",
    "input_window_end",
    "model_id",
    "model_trained_until_timestamp",
    "feature_schema_hash",
    "training_row_count",
    "validation_score",
    "prob_short",
    "prob_neutral",
    "prob_long",
    "pred_future_return_3",
    "pred_future_range_percent_3",
    "pred_future_volume_3m",
    "pred_future_trade_count_3m",
    "pred_future_market_buy_volume_3m",
    "pred_future_market_sell_volume_3m",
    "pred_future_volume_delta_3m",
    "pred_future_market_pressure_3m",
    "pred_future_order_book_imbalance_10bps_3m",
    "pred_future_bid_depth_change_10bps_3m",
    "pred_future_ask_depth_change_10bps_3m",
    "pred_future_spread_percent_3m",
    "pred_future_breakout_pressure_3m",
    "pred_future_absorption_3m",
    "label_ready",
]

ACTUAL_LABEL_COLUMNS = [
    "actual_future_return_3",
    "actual_future_range_percent_3",
    "actual_future_volume_3m",
    "actual_future_trade_count_3m",
    "actual_market_buy_volume_3m",
    "actual_market_sell_volume_3m",
    "actual_volume_delta_3m",
    "actual_market_pressure_3m",
    "actual_order_book_imbalance_10bps_3m",
    "actual_bid_depth_change_10bps_3m",
    "actual_ask_depth_change_10bps_3m",
    "actual_spread_percent_3m",
    "actual_breakout_pressure_3m",
    "actual_absorption_3m",
    "actual_path_class",
]


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(logits, -40, 40))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def safe_ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return 0.0
    if abs(denominator) < EPSILON:
        return 0.0
    return float(numerator / denominator)


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def copy_file_if_exists(source, destination):
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def normalize_timestamps(frame):
    if "timestamp" not in frame.columns:
        return frame
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["timestamp"] = np.where(timestamps < 10_000_000_000, timestamps * 1000, timestamps)
    return frame


def load_csv_or_empty(path, columns):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    return normalize_timestamps(frame)


def coerce_numeric_columns(frame, skip=("time",)):
    for column in frame.columns:
        if column not in skip:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def coerce_feature_ready(frame):
    if "feature_ready" not in frame.columns:
        frame["feature_ready"] = False
        return frame
    if frame["feature_ready"].dtype != bool:
        frame["feature_ready"] = frame["feature_ready"].astype(str).str.lower().isin(
            ["true", "1", "yes", "y"]
        )
    return frame


def choose_feature_columns(frame):
    columns = []
    for column in frame.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if column in RAW_PRICE_COLUMNS:
            continue
        if column.startswith("not_ready_"):
            continue
        if column.startswith("actual_") or column.startswith("pred_"):
            continue
        if column.startswith("prob_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def has_contiguous_timestamps(timestamps, expected_step_ms=60_000):
    values = np.asarray(timestamps, dtype=np.int64)
    if len(values) <= 1:
        return True
    return bool(np.all(np.diff(values) == expected_step_ms))


def build_input_window(feature_frame, end_index, feature_columns, lookback):
    start_index = end_index - lookback + 1
    if start_index < 0:
        return None, None, None

    window = feature_frame.iloc[start_index : end_index + 1].copy()
    if len(window) != lookback:
        return None, None, None
    if not bool(window["feature_ready"].all()):
        return None, None, None
    if not has_contiguous_timestamps(window["timestamp"].to_numpy()):
        return None, None, None

    missing = [column for column in feature_columns if column not in window.columns]
    if missing:
        return None, None, None

    values = window[feature_columns].replace([np.inf, -np.inf], np.nan)
    if values.isna().any().any():
        return None, None, None

    return (
        values.to_numpy(dtype=np.float64).reshape(1, -1),
        int(window["timestamp"].iloc[0]),
        int(window["timestamp"].iloc[-1]),
    )


def initialize_model(input_size, hidden_units, output_size, rng):
    return {
        "w1": rng.normal(0, math.sqrt(2 / max(1, input_size)), (input_size, hidden_units)),
        "b1": np.zeros(hidden_units),
        "w_class": rng.normal(0, math.sqrt(2 / max(1, hidden_units)), (hidden_units, 3)),
        "b_class": np.zeros(3),
        "w_reg": rng.normal(0, math.sqrt(2 / max(1, hidden_units)), (hidden_units, output_size)),
        "b_reg": np.zeros(output_size),
    }


def forward(model, x):
    hidden_pre = x @ model["w1"] + model["b1"]
    hidden = np.maximum(hidden_pre, 0.0)
    class_logits = hidden @ model["w_class"] + model["b_class"]
    probabilities = softmax(class_logits)
    regression_scaled = hidden @ model["w_reg"] + model["b_reg"]
    return hidden_pre, hidden, probabilities, regression_scaled


def load_model(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    artifact["model"] = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in artifact["model"].items()
    }
    artifact["feature_mean"] = np.asarray(artifact["feature_mean"], dtype=np.float64)
    artifact["feature_std"] = np.asarray(artifact["feature_std"], dtype=np.float64)
    artifact["target_mean"] = np.asarray(artifact["target_mean"], dtype=np.float64)
    artifact["target_std"] = np.asarray(artifact["target_std"], dtype=np.float64)
    artifact["feature_std"][artifact["feature_std"] < EPSILON] = 1.0
    artifact["target_std"][artifact["target_std"] < EPSILON] = 1.0
    return artifact


def save_model(path, artifact):
    serializable = dict(artifact)
    serializable["model"] = {
        name: np.asarray(value).tolist()
        for name, value in artifact["model"].items()
    }
    serializable["feature_mean"] = np.asarray(artifact["feature_mean"]).tolist()
    serializable["feature_std"] = np.asarray(artifact["feature_std"]).tolist()
    serializable["target_mean"] = np.asarray(artifact["target_mean"]).tolist()
    serializable["target_std"] = np.asarray(artifact["target_std"]).tolist()
    atomic_write_json(serializable, path)


def predict_with_model(artifact, flattened_window):
    x = (flattened_window - artifact["feature_mean"]) / artifact["feature_std"]
    _, _, probabilities, regression_scaled = forward(artifact["model"], x)
    regression = regression_scaled * artifact["target_std"] + artifact["target_mean"]
    return probabilities[0], regression[0]


def prediction_metadata_to_row(artifact):
    if artifact is None:
        return {
            "model_id": "bootstrap",
            "model_trained_until_timestamp": "",
            "feature_schema_hash": "bootstrap",
            "training_row_count": 0,
            "validation_score": 0.0,
        }

    return {
        "model_id": artifact.get("model_id", "unknown"),
        "model_trained_until_timestamp": artifact.get(
            "model_trained_until_timestamp",
            "",
        ),
        "feature_schema_hash": artifact.get("feature_schema_hash", ""),
        "training_row_count": artifact.get(
            "training_row_count",
            artifact.get("training_rows", ""),
        ),
        "validation_score": artifact.get("validation_score", ""),
    }


def prediction_values_to_row(probabilities, regression, target_columns, artifact=None):
    row = {
        **prediction_metadata_to_row(artifact),
        "prob_short": float(probabilities[CLASS_SHORT]),
        "prob_neutral": float(probabilities[CLASS_NEUTRAL]),
        "prob_long": float(probabilities[CLASS_LONG]),
    }
    regression_by_target = {
        target: float(regression[index])
        for index, target in enumerate(target_columns)
    }
    for target, prediction_column in PREDICTION_COLUMN_FOR_TARGET.items():
        row[prediction_column] = regression_by_target.get(target, 0.0)
    return row


def bootstrap_prediction_from_latest_row(row):
    pressure = float(row.get("market_pressure", 0.0) or 0.0)
    breakout = float(row.get("breakout_pressure_index", 0.0) or 0.0)
    imbalance = float(row.get("order_book_imbalance_10bps", 0.0) or 0.0)
    signal = np.clip((pressure + breakout + imbalance) / 3.0, -1.0, 1.0)
    directional = min(0.22, abs(signal) * 0.18)
    prob_neutral = 0.54 - directional * 0.5
    if signal >= 0:
        prob_long = 0.23 + directional
        prob_short = 1.0 - prob_neutral - prob_long
    else:
        prob_short = 0.23 + directional
        prob_long = 1.0 - prob_neutral - prob_short
    probabilities = np.asarray([prob_short, prob_neutral, prob_long], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()

    prediction_row = {
        **prediction_metadata_to_row(None),
        "prob_short": float(probabilities[CLASS_SHORT]),
        "prob_neutral": float(probabilities[CLASS_NEUTRAL]),
        "prob_long": float(probabilities[CLASS_LONG]),
        "pred_future_return_3": float(row.get("return_3", 0.0) or 0.0),
        "pred_future_range_percent_3": float(row.get("range_percent", 0.0) or 0.0),
        "pred_future_volume_3m": 0.0,
        "pred_future_trade_count_3m": 0.0,
        "pred_future_market_buy_volume_3m": 0.0,
        "pred_future_market_sell_volume_3m": 0.0,
        "pred_future_volume_delta_3m": 0.0,
        "pred_future_market_pressure_3m": pressure,
        "pred_future_order_book_imbalance_10bps_3m": imbalance,
        "pred_future_bid_depth_change_10bps_3m": float(
            row.get("bid_depth_change_ratio_10bps", 0.0) or 0.0
        ),
        "pred_future_ask_depth_change_10bps_3m": float(
            row.get("ask_depth_change_ratio_10bps", 0.0) or 0.0
        ),
        "pred_future_spread_percent_3m": float(row.get("spread_percent", 0.0) or 0.0),
        "pred_future_breakout_pressure_3m": breakout,
        "pred_future_absorption_3m": float(row.get("absorption_index", 0.0) or 0.0),
    }
    return prediction_row


def compounded_return_and_drawdown(returns):
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
    return equity - 1.0, max_drawdown
