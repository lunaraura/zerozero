import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from live_3m_model_utils import atomic_write_csv, parse_bool, safe_ratio


EPSILON = 1e-8

OFFER_SPECS = [
    ("LONG", 3, 0.0020, 0.0010),
    ("SHORT", 3, 0.0020, 0.0010),
    ("LONG", 3, 0.0030, 0.0015),
    ("SHORT", 3, 0.0030, 0.0015),
    ("LONG", 10, 0.0040, 0.0020),
    ("SHORT", 10, 0.0040, 0.0020),
    ("LONG", 10, 0.0060, 0.0030),
    ("SHORT", 10, 0.0060, 0.0030),
]

OLD_CONTEXT_COLUMNS = [
    "old_prob_down",
    "old_prob_neutral",
    "old_prob_up",
    "old_directional_confidence",
    "old_prediction_age_ms",
    "old_context_available",
]

OFFER_INPUT_COLUMNS = [
    "offer_side_long",
    "offer_side_short",
    "offer_horizon_minutes",
    "offer_take_profit",
    "offer_stop_loss",
]

TARGET_COLUMNS = [
    "accept_target",
    "target_allocation_fraction",
    "target_allocation_bucket",
    "allocation_score",
    "opportunity_score",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "favorable_velocity",
    "final_return",
]

MODEL_TARGET_COLUMNS = [
    "target_allocation_fraction",
    "opportunity_score",
    "net_return",
    "max_adverse_excursion",
]

OUTCOME_COLUMNS = {
    "tp_hit",
    "sl_hit",
    "hit_result",
    "realized_return",
    "net_return",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "final_return",
    "final_return_at_horizon",
    "time_to_max_favorable",
    "time_to_max_adverse",
    "favorable_velocity",
    "adverse_velocity",
    "risk_reward_ratio",
    "opportunity_score",
    "time_to_tp",
    "time_to_sl",
    "time_to_low",
    "time_to_high",
    "lowest_return",
    "highest_return",
    "downside_velocity",
    "upside_velocity",
    "accept_target",
    "target_allocation_fraction",
    "target_allocation_bucket",
    "allocation_score",
    "quality_score",
}

NON_FEATURE_COLUMNS = {
    "timestamp",
    "time",
    "symbol",
    "offer_side",
    "entry_price",
    "tp_price",
    "sl_price",
    *OUTCOME_COLUMNS,
}

RAW_PRICE_FEATURE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "best_bid",
    "best_ask",
    "mid_price",
}


def normalize_timestamps(frame):
    if "timestamp" not in frame.columns:
        return frame
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["timestamp"] = np.where(timestamps < 10_000_000_000, timestamps * 1000, timestamps)
    return frame


def coerce_numeric(frame, skip=("time", "symbol", "offer_side", "hit_result")):
    for column in frame.columns:
        if column not in skip:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def read_csv_sorted(path, required_columns, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{description} is missing required columns: {missing}")
    frame = normalize_timestamps(frame)
    frame = coerce_numeric(frame)
    frame = frame.dropna(subset=["timestamp"]).copy()
    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def choose_snapshot_feature_columns(feature_frame):
    columns = []
    for column in feature_frame.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if column in RAW_PRICE_FEATURE_COLUMNS:
            continue
        if column.startswith("not_ready_"):
            continue
        if column.startswith("actual_") or column.startswith("pred_") or column.startswith("prob_"):
            continue
        if column == "feature_ready":
            continue
        if pd.api.types.is_numeric_dtype(feature_frame[column]):
            columns.append(column)
    return columns


def prepare_feature_rows(feature_frame):
    feature_frame = feature_frame.copy()
    if "feature_ready" not in feature_frame.columns:
        feature_frame["feature_ready"] = False
    if feature_frame["feature_ready"].dtype != bool:
        feature_frame["feature_ready"] = feature_frame["feature_ready"].astype(str).str.lower().isin(
            ["true", "1", "yes", "y"]
        )
    return feature_frame


def old_prediction_path(project_root, symbol, use_recent=True):
    recent = Path(project_root) / "data" / f"{symbol}_model_predictions_recent.csv"
    default = Path(project_root) / "data" / f"{symbol}_model_predictions.csv"
    if use_recent and recent.exists():
        return recent
    if default.exists():
        return default
    return None


def load_old_context(project_root, symbol, use_old_model_context, max_age_ms):
    if not use_old_model_context:
        return None, None
    path = old_prediction_path(project_root, symbol)
    if path is None:
        return None, None
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        return None, path
    frame = normalize_timestamps(frame)
    frame = coerce_numeric(frame)
    frame = frame.dropna(subset=["timestamp"]).copy()
    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    required = ["prob_down", "prob_neutral", "prob_up"]
    if any(column not in frame.columns for column in required):
        return None, path
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    frame = frame.rename(
        columns={
            "timestamp": "old_prediction_timestamp",
            "prob_down": "old_prob_down",
            "prob_neutral": "old_prob_neutral",
            "prob_up": "old_prob_up",
        }
    )
    frame["old_directional_confidence"] = (
        frame[["old_prob_down", "old_prob_up"]].max(axis=1) - frame["old_prob_neutral"]
    )
    return frame[
        [
            "old_prediction_timestamp",
            "old_prob_down",
            "old_prob_neutral",
            "old_prob_up",
            "old_directional_confidence",
        ]
    ], path


def attach_old_context(rows, old_context, max_age_ms):
    rows = rows.sort_values("timestamp").copy()
    if old_context is None or len(old_context) == 0:
        rows["old_prob_down"] = 1.0 / 3.0
        rows["old_prob_neutral"] = 1.0 / 3.0
        rows["old_prob_up"] = 1.0 / 3.0
        rows["old_directional_confidence"] = 0.0
        rows["old_prediction_age_ms"] = np.nan
        rows["old_context_available"] = 0
        return rows

    merged = pd.merge_asof(
        rows,
        old_context.sort_values("old_prediction_timestamp"),
        left_on="timestamp",
        right_on="old_prediction_timestamp",
        direction="backward",
    )
    merged["old_prediction_age_ms"] = merged["timestamp"] - merged["old_prediction_timestamp"]
    available = (
        merged["old_prediction_timestamp"].notna()
        & (merged["old_prediction_age_ms"] >= 0)
        & (merged["old_prediction_age_ms"] <= max_age_ms)
    )
    merged["old_context_available"] = available.astype(int)
    for column in ["old_prob_down", "old_prob_neutral", "old_prob_up"]:
        merged.loc[~available, column] = 1.0 / 3.0
    merged.loc[~available, "old_directional_confidence"] = 0.0
    merged.loc[~available, "old_prediction_age_ms"] = np.nan
    return merged.drop(columns=["old_prediction_timestamp"], errors="ignore")


def offer_prices(entry_price, side, take_profit, stop_loss):
    if side == "LONG":
        return entry_price * (1.0 + take_profit), entry_price * (1.0 - stop_loss)
    return entry_price * (1.0 - take_profit), entry_price * (1.0 + stop_loss)


def return_for_price(entry_price, side, price):
    raw_return = safe_ratio(price - entry_price, entry_price)
    return raw_return if side == "LONG" else -raw_return


def allocation_fraction_to_bucket(fraction):
    mapping = {
        0.00: 0,
        0.01: 1,
        0.025: 2,
        0.05: 3,
        0.10: 4,
        0.20: 5,
    }
    return mapping.get(float(fraction), 0)


def allocation_score_to_fraction(score):
    if score <= 0.0:
        return 0.0
    if score > 0.0075:
        return 0.20
    if score > 0.0040:
        return 0.10
    if score > 0.0020:
        return 0.05
    if score > 0.0010:
        return 0.025
    return 0.01


def allocation_targets(
    opportunity_score,
    max_favorable_excursion,
    max_adverse_excursion,
    favorable_velocity,
    favorable_weight,
    adverse_weight,
    velocity_weight,
):
    allocation_score = (
        opportunity_score
        + favorable_weight * max_favorable_excursion
        - adverse_weight * abs(max_adverse_excursion)
        + velocity_weight * favorable_velocity
    )
    target_fraction = allocation_score_to_fraction(allocation_score)
    target_bucket = allocation_fraction_to_bucket(target_fraction)
    return allocation_score, target_fraction, target_bucket


def replay_offer(
    entry_price,
    side,
    take_profit,
    stop_loss,
    future_rows,
    round_trip_cost,
    adverse_penalty,
    time_penalty,
    allocation_favorable_weight=0.50,
    allocation_adverse_weight=1.00,
    allocation_velocity_weight=0.25,
):
    horizon_minutes = len(future_rows)
    tp_price, sl_price = offer_prices(entry_price, side, take_profit, stop_loss)
    tp_hit = False
    sl_hit = False
    hit_result = "expired"
    time_to_tp = np.nan
    time_to_sl = np.nan
    returns_seen = []

    for step, (_, candle) in enumerate(future_rows.iterrows(), start=1):
        high_return = return_for_price(entry_price, side, float(candle["high"]))
        low_return = return_for_price(entry_price, side, float(candle["low"]))
        if side == "LONG":
            favorable = high_return
            adverse = low_return
            step_tp_hit = float(candle["high"]) >= tp_price
            step_sl_hit = float(candle["low"]) <= sl_price
        else:
            favorable = low_return
            adverse = high_return
            step_tp_hit = float(candle["low"]) <= tp_price
            step_sl_hit = float(candle["high"]) >= sl_price

        returns_seen.extend([favorable, adverse])

        if step_tp_hit and step_sl_hit:
            tp_hit = True
            sl_hit = True
            time_to_tp = step
            time_to_sl = step
            hit_result = "ambiguous"
            break
        if step_tp_hit:
            tp_hit = True
            time_to_tp = step
            hit_result = "tp_first"
            break
        if step_sl_hit:
            sl_hit = True
            time_to_sl = step
            hit_result = "sl_first"
            break

    final_close = float(future_rows["close"].iloc[-1])

    if side == "LONG":
        future_highs = future_rows["high"].to_numpy(dtype=np.float64)
        future_lows = future_rows["low"].to_numpy(dtype=np.float64)
        final_return = safe_ratio(final_close, entry_price) - 1.0
        max_future_high = float(np.max(future_highs))
        min_future_low = float(np.min(future_lows))
        max_favorable_excursion = safe_ratio(max_future_high, entry_price) - 1.0
        max_adverse_excursion = safe_ratio(min_future_low, entry_price) - 1.0
        time_to_max_favorable = int(np.argmax(future_highs) + 1)
        time_to_max_adverse = int(np.argmin(future_lows) + 1)
    else:
        future_highs = future_rows["high"].to_numpy(dtype=np.float64)
        future_lows = future_rows["low"].to_numpy(dtype=np.float64)
        final_return = safe_ratio(entry_price, final_close) - 1.0
        min_future_low = float(np.min(future_lows))
        max_future_high = float(np.max(future_highs))
        max_favorable_excursion = safe_ratio(entry_price, min_future_low) - 1.0
        max_adverse_excursion = safe_ratio(entry_price, max_future_high) - 1.0
        time_to_max_favorable = int(np.argmin(future_lows) + 1)
        time_to_max_adverse = int(np.argmax(future_highs) + 1)

    final_return_at_horizon = final_return
    highest_return = max(returns_seen) if returns_seen else final_return
    lowest_return = min(returns_seen) if returns_seen else final_return
    favorable_velocity = safe_ratio(
        max_favorable_excursion,
        max(1, time_to_max_favorable),
    )
    adverse_velocity = safe_ratio(
        max_adverse_excursion,
        max(1, time_to_max_adverse),
    )
    risk_reward_ratio = safe_ratio(
        max_favorable_excursion,
        max(abs(max_adverse_excursion), EPSILON),
    )
    opportunity_score = (
        max_favorable_excursion
        - adverse_penalty * abs(max_adverse_excursion)
        - round_trip_cost
        - time_penalty * time_to_max_favorable
    )

    if hit_result == "tp_first":
        realized_return = take_profit
    elif hit_result in {"sl_first", "ambiguous"}:
        # Conservative handling: if both are inside the same candle, assume the
        # adverse stop was hit first for paper labeling.
        realized_return = -stop_loss
    else:
        realized_return = final_return_at_horizon

    net_return = realized_return - round_trip_cost
    accept_target = int(opportunity_score > 0)
    allocation_score, target_allocation_fraction, target_allocation_bucket = allocation_targets(
        opportunity_score,
        max_favorable_excursion,
        max_adverse_excursion,
        favorable_velocity,
        allocation_favorable_weight,
        allocation_adverse_weight,
        allocation_velocity_weight,
    )
    # Backward-compatible alias for older diagnostics. The primary target is
    # now opportunity_score, not TP/SL-specific quality_score.
    quality_score = opportunity_score
    time_to_high = int(np.argmax(future_rows["high"].to_numpy(dtype=np.float64)) + 1)
    time_to_low = int(np.argmin(future_rows["low"].to_numpy(dtype=np.float64)) + 1)
    downside_velocity = safe_ratio(abs(min(0.0, lowest_return)), time_to_low)
    upside_velocity = safe_ratio(max(0.0, highest_return), time_to_high)

    return {
        "tp_price": tp_price,
        "sl_price": sl_price,
        "tp_hit": int(tp_hit),
        "sl_hit": int(sl_hit),
        "hit_result": hit_result,
        "realized_return": realized_return,
        "net_return": net_return,
        "max_favorable_excursion": max_favorable_excursion,
        "max_adverse_excursion": max_adverse_excursion,
        "final_return": final_return,
        "final_return_at_horizon": final_return_at_horizon,
        "time_to_max_favorable": time_to_max_favorable,
        "time_to_max_adverse": time_to_max_adverse,
        "favorable_velocity": favorable_velocity,
        "adverse_velocity": adverse_velocity,
        "risk_reward_ratio": risk_reward_ratio,
        "opportunity_score": opportunity_score,
        "allocation_score": allocation_score,
        "target_allocation_fraction": target_allocation_fraction,
        "target_allocation_bucket": target_allocation_bucket,
        "time_to_tp": time_to_tp,
        "time_to_sl": time_to_sl,
        "time_to_low": time_to_low,
        "time_to_high": time_to_high,
        "lowest_return": lowest_return,
        "highest_return": highest_return,
        "downside_velocity": downside_velocity,
        "upside_velocity": upside_velocity,
        "accept_target": accept_target,
        "quality_score": quality_score,
    }


def add_offer_input_columns(frame):
    frame = frame.copy()
    frame["offer_side_long"] = (frame["offer_side"] == "LONG").astype(float)
    frame["offer_side_short"] = (frame["offer_side"] == "SHORT").astype(float)
    return frame


def choose_model_feature_columns(frame):
    frame = add_offer_input_columns(frame)
    columns = []
    for column in frame.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if column in OUTCOME_COLUMNS:
            continue
        if column in {"tp_hit", "sl_hit", "hit_result"}:
            continue
        if column in {"tp_price", "sl_price", "entry_price"}:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    for column in OFFER_INPUT_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(logits, -40, 40))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def initialize_model(input_size, hidden_units, regression_outputs, rng):
    return {
        "w1": rng.normal(0, math.sqrt(2 / max(1, input_size)), (input_size, hidden_units)),
        "b1": np.zeros(hidden_units),
        "w_class": rng.normal(0, math.sqrt(2 / max(1, hidden_units)), (hidden_units, 2)),
        "b_class": np.zeros(2),
        "w_bucket": rng.normal(0, math.sqrt(2 / max(1, hidden_units)), (hidden_units, 6)),
        "b_bucket": np.zeros(6),
        "w_reg": rng.normal(0, math.sqrt(2 / max(1, hidden_units)), (hidden_units, regression_outputs)),
        "b_reg": np.zeros(regression_outputs),
    }


def forward(model, x):
    hidden_pre = x @ model["w1"] + model["b1"]
    hidden = np.maximum(hidden_pre, 0.0)
    accept_probabilities = softmax(hidden @ model["w_class"] + model["b_class"])
    bucket_probabilities = softmax(hidden @ model["w_bucket"] + model["b_bucket"])
    regression_scaled = hidden @ model["w_reg"] + model["b_reg"]
    return hidden_pre, hidden, accept_probabilities, bucket_probabilities, regression_scaled


def save_model(path, artifact):
    path = Path(path)
    payload = dict(artifact)
    payload["model"] = {name: np.asarray(value).tolist() for name, value in artifact["model"].items()}
    for key in ["feature_mean", "feature_std", "target_mean", "target_std"]:
        payload[key] = np.asarray(artifact[key]).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def load_model(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    artifact["model"] = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in artifact["model"].items()
    }
    for key in ["feature_mean", "feature_std", "target_mean", "target_std"]:
        artifact[key] = np.asarray(artifact[key], dtype=np.float64)
    artifact["feature_std"][artifact["feature_std"] < EPSILON] = 1.0
    artifact["target_std"][artifact["target_std"] < EPSILON] = 1.0
    return artifact


def predict_artifact(artifact, frame):
    working = add_offer_input_columns(frame.copy())
    for column in artifact["feature_columns"]:
        if column not in working.columns:
            working[column] = 0.0
    x = working[artifact["feature_columns"]].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    x = (x - artifact["feature_mean"]) / artifact["feature_std"]
    _, _, probabilities, _, regression_scaled = forward(artifact["model"], x)
    regression = regression_scaled * artifact["target_std"] + artifact["target_mean"]
    return probabilities[:, 1], regression


def predict_artifact_full(artifact, frame):
    working = add_offer_input_columns(frame.copy())
    for column in artifact["feature_columns"]:
        if column not in working.columns:
            working[column] = 0.0
    x = working[artifact["feature_columns"]].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    x = (x - artifact["feature_mean"]) / artifact["feature_std"]
    _, _, accept_probabilities, bucket_probabilities, regression_scaled = forward(artifact["model"], x)
    regression = regression_scaled * artifact["target_std"] + artifact["target_mean"]
    return accept_probabilities[:, 1], bucket_probabilities, regression


def compounded_return_and_drawdown(returns):
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
    return equity - 1.0, max_drawdown
