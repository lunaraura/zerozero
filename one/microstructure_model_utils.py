import datetime as dt
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


EPSILON = 1e-9
WINDOW_SECONDS = [3, 5, 10, 30, 60]

REGRESSION_TARGET_COLUMNS = [
    "future_return_10s",
    "max_runup_10s",
    "max_drawdown_10s",
    "upside_velocity_10s",
    "downside_velocity_10s",
    "future_spread_expansion_10s",
    "future_bid_log_depth_change_10s",
    "future_ask_log_depth_change_10s",
]

DIAGNOSTIC_TARGET_COLUMNS = [
    "future_bid_depth_change_10s",
    "future_ask_depth_change_10s",
]

DEFAULT_MICRO_MAX_ABS_RETURN_TARGET = 0.02
DEFAULT_MICRO_MAX_ABS_SPREAD_TARGET = 0.01
DEFAULT_MICRO_MAX_ABS_LOG_DEPTH_TARGET = 5.0

EVENT_TARGET_COLUMNS = [
    "upside_scare_event_10s",
    "downside_scare_event_10s",
    "aggressive_buy_burst_10s",
    "aggressive_sell_burst_10s",
    "bid_liquidity_drop_10s",
    "ask_liquidity_drop_10s",
    "spread_expansion_event_10s",
    "direction_flip_10s",
    "continuation_30s",
    "continuation_60s",
    "reversal_after_upside_scare_30s",
    "reversal_after_downside_scare_30s",
    "reversal_after_upside_scare_60s",
    "reversal_after_downside_scare_60s",
]

AFTERMATH_COLUMNS = [
    "future_return_30s",
    "future_return_60s",
]

BASE_COLUMNS = [
    "timestamp",
    "time",
    "feature_ready",
]

MODEL_FEATURE_PREFIX = "feature_"


def percent(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def safe_ratio(numerator, denominator, default=0.0):
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return default
    if abs(denominator) < EPSILON:
        return default
    return float(numerator / denominator)


def sigmoid(values):
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def regression_bound_for_target(
    target,
    max_abs_return=DEFAULT_MICRO_MAX_ABS_RETURN_TARGET,
    max_abs_spread=DEFAULT_MICRO_MAX_ABS_SPREAD_TARGET,
    max_abs_log_depth=DEFAULT_MICRO_MAX_ABS_LOG_DEPTH_TARGET,
):
    if target == "future_spread_expansion_10s":
        return float(max_abs_spread)
    if "log_depth_change" in str(target):
        return float(max_abs_log_depth)
    if target in {
        "future_return_10s",
        "max_runup_10s",
        "max_drawdown_10s",
        "upside_velocity_10s",
        "downside_velocity_10s",
    }:
        return float(max_abs_return)
    return None


def regression_clip_bounds(
    regression_columns,
    max_abs_return=DEFAULT_MICRO_MAX_ABS_RETURN_TARGET,
    max_abs_spread=DEFAULT_MICRO_MAX_ABS_SPREAD_TARGET,
    max_abs_log_depth=DEFAULT_MICRO_MAX_ABS_LOG_DEPTH_TARGET,
):
    return {
        column: regression_bound_for_target(
            column,
            max_abs_return=max_abs_return,
            max_abs_spread=max_abs_spread,
            max_abs_log_depth=max_abs_log_depth,
        )
        for column in regression_columns
    }


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


def load_snapshot_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing snapshot file: {path}")

    frame = pd.read_csv(path, low_memory=False)
    required = [
        "timestamp",
        "time",
        "mid_price",
        "best_bid",
        "best_ask",
        "spread_percent",
        "market_buy_volume_10s",
        "market_sell_volume_10s",
        "total_trade_volume_10s",
        "trade_count_10s",
        "market_pressure_10s",
        "bid_depth_10bps",
        "ask_depth_10bps",
        "order_book_imbalance_10bps",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Snapshot CSV is missing required columns: {missing}")

    optional_defaults = {
        "bid_depth_25bps": 0.0,
        "ask_depth_25bps": 0.0,
        "order_book_imbalance_25bps": 0.0,
        "bid_depth_change_10bps": 0.0,
        "ask_depth_change_10bps": 0.0,
        "imbalance_change_10bps": 0.0,
        "large_bid_wall_distance": 0.0,
        "large_ask_wall_distance": 0.0,
        "large_bid_wall_size": 0.0,
        "large_ask_wall_size": 0.0,
    }
    for column, default in optional_defaults.items():
        if column not in frame.columns:
            frame[column] = default

    for column in frame.columns:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    frame = frame.reset_index(drop=True)
    frame.attrs["timestamp_values"] = frame["timestamp"].to_numpy(dtype=np.int64)
    return frame


def infer_snapshot_step_seconds(frame):
    if len(frame) < 2:
        return 1.0
    diffs = frame["timestamp"].diff().dropna()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs) / 1000.0)


def timestamp_values(frame):
    values = frame.attrs.get("timestamp_values")
    if values is None or len(values) != len(frame):
        values = frame["timestamp"].to_numpy(dtype=np.int64)
        frame.attrs["timestamp_values"] = values
    return values


def is_valid_current_row(row):
    checks = [
        row.get("mid_price", np.nan),
        row.get("best_bid", np.nan),
        row.get("best_ask", np.nan),
        row.get("bid_depth_10bps", np.nan),
        row.get("ask_depth_10bps", np.nan),
    ]
    return all(np.isfinite(value) and value > 0 for value in checks)


def row_window(frame, end_index, seconds):
    timestamps = timestamp_values(frame)
    end_timestamp = int(timestamps[end_index])
    start_timestamp = end_timestamp - seconds * 1000
    start_index = int(np.searchsorted(timestamps, start_timestamp, side="right"))
    return frame.iloc[start_index : end_index + 1].copy()


def previous_row_window(frame, end_index, seconds):
    timestamps = timestamp_values(frame)
    end_timestamp = int(timestamps[end_index])
    start_timestamp = end_timestamp - seconds * 1000
    previous_start = start_timestamp - seconds * 1000
    previous_start_index = int(np.searchsorted(timestamps, previous_start, side="right"))
    previous_end_index = int(np.searchsorted(timestamps, start_timestamp, side="right"))
    return frame.iloc[previous_start_index:previous_end_index].copy()


def future_window(frame, start_index, seconds):
    timestamps = timestamp_values(frame)
    start_timestamp = int(timestamps[start_index])
    end_timestamp = start_timestamp + seconds * 1000
    end_index = int(np.searchsorted(timestamps, end_timestamp, side="right"))
    return frame.iloc[start_index + 1 : end_index].copy()


def has_enough_window(window, seconds, snapshot_step_seconds):
    if len(window) == 0:
        return False
    if snapshot_step_seconds <= 0:
        return False
    expected = max(1, int(math.floor(seconds / snapshot_step_seconds)))
    return len(window) >= max(1, min(expected, seconds))


def slope_per_second(values, timestamps):
    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(timestamps)
    values = values[valid]
    timestamps = timestamps[valid]
    if len(values) < 2:
        return 0.0
    x = (timestamps - timestamps[0]) / 1000.0
    if np.allclose(x, x[0]):
        return 0.0
    slope, _ = np.polyfit(x, values, 1)
    if not np.isfinite(slope):
        return 0.0
    return float(slope)


def volume_sum(frame, column):
    if len(frame) == 0 or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def mean_value(frame, column, default=0.0):
    if len(frame) == 0 or column not in frame.columns:
        return default
    value = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
    return default if pd.isna(value) else float(value)


def first_value(frame, column, default=0.0):
    if len(frame) == 0 or column not in frame.columns:
        return default
    return float(pd.to_numeric(frame[column], errors="coerce").iloc[0])


def last_value(frame, column, default=0.0):
    if len(frame) == 0 or column not in frame.columns:
        return default
    return float(pd.to_numeric(frame[column], errors="coerce").iloc[-1])


def build_feature_row(frame, end_index, snapshot_step_seconds):
    row = frame.loc[end_index]
    timestamp = int(row["timestamp"])
    if not is_valid_current_row(row):
        return None, "invalid current book row"

    history_60s = row_window(frame, end_index, 60)
    if not has_enough_window(history_60s, 60, snapshot_step_seconds):
        return None, "not enough 60s history"

    current_mid = float(row["mid_price"])
    feature_row = {
        "timestamp": timestamp,
        "time": row.get("time", ""),
        "feature_ready": True,
        "snapshot_step_seconds": snapshot_step_seconds,
    }

    past_60_volume = max(volume_sum(history_60s, "total_trade_volume_10s"), EPSILON)
    past_60_trades = max(volume_sum(history_60s, "trade_count_10s"), EPSILON)
    past_60_spread = max(mean_value(history_60s, "spread_percent"), EPSILON)

    previous_velocity = 0.0
    previous_pressure = 0.0
    for seconds in WINDOW_SECONDS:
        current_window = row_window(frame, end_index, seconds)
        previous_window = previous_row_window(frame, end_index, seconds)
        if len(current_window) == 0:
            return None, f"empty {seconds}s feature window"

        start_mid = first_value(current_window, "mid_price", current_mid)
        end_mid = last_value(current_window, "mid_price", current_mid)
        window_return = safe_ratio(end_mid - start_mid, start_mid)
        velocity = safe_ratio(window_return, max(seconds, 1))
        acceleration = velocity - previous_velocity
        previous_velocity = velocity

        buy_volume = volume_sum(current_window, "market_buy_volume_10s")
        sell_volume = volume_sum(current_window, "market_sell_volume_10s")
        total_volume = volume_sum(current_window, "total_trade_volume_10s")
        trade_count = volume_sum(current_window, "trade_count_10s")
        pressure = safe_ratio(buy_volume - sell_volume, total_volume)
        pressure_acceleration = pressure - previous_pressure
        previous_pressure = pressure

        prev_volume = max(volume_sum(previous_window, "total_trade_volume_10s"), EPSILON)
        prev_trades = max(volume_sum(previous_window, "trade_count_10s"), EPSILON)
        prev_spread = max(mean_value(previous_window, "spread_percent", past_60_spread), EPSILON)
        bid_depth_start = first_value(current_window, "bid_depth_10bps", 0.0)
        ask_depth_start = first_value(current_window, "ask_depth_10bps", 0.0)
        bid_depth_end = last_value(current_window, "bid_depth_10bps", 0.0)
        ask_depth_end = last_value(current_window, "ask_depth_10bps", 0.0)
        imbalance_start = first_value(current_window, "order_book_imbalance_10bps", 0.0)
        imbalance_end = last_value(current_window, "order_book_imbalance_10bps", 0.0)
        spread_now = last_value(current_window, "spread_percent", 0.0)
        spread_mean = mean_value(current_window, "spread_percent", 0.0)
        spread_expansion = safe_ratio(spread_now - prev_spread, prev_spread)
        bid_depth_change = safe_ratio(bid_depth_end - bid_depth_start, bid_depth_start)
        ask_depth_change = safe_ratio(ask_depth_end - ask_depth_start, ask_depth_start)
        imbalance_change = imbalance_end - imbalance_start
        liquidity_thinning = (
            max(0.0, -bid_depth_change)
            + max(0.0, -ask_depth_change)
            + max(0.0, spread_expansion)
        ) / 3.0
        slippage_risk = (
            abs(pressure)
            + abs(imbalance_end)
            + max(0.0, spread_expansion)
            + liquidity_thinning
            + safe_ratio(spread_now, max(past_60_spread, EPSILON))
        ) / 5.0

        max_mid = float(pd.to_numeric(current_window["mid_price"], errors="coerce").max())
        min_mid = float(pd.to_numeric(current_window["mid_price"], errors="coerce").min())
        failed_bounce = 1.0 if max_mid > start_mid and end_mid < start_mid else 0.0
        failed_breakdown = 1.0 if min_mid < start_mid and end_mid > start_mid else 0.0

        prefix = f"feature_{seconds}s_"
        feature_row[prefix + "return"] = window_return
        feature_row[prefix + "price_velocity"] = velocity
        feature_row[prefix + "price_acceleration"] = acceleration
        feature_row[prefix + "volume_burst"] = safe_ratio(total_volume, past_60_volume * seconds / 60.0)
        feature_row[prefix + "trade_count_burst"] = safe_ratio(trade_count, past_60_trades * seconds / 60.0)
        feature_row[prefix + "market_buy_volume"] = buy_volume
        feature_row[prefix + "market_sell_volume"] = sell_volume
        feature_row[prefix + "market_pressure"] = pressure
        feature_row[prefix + "pressure_acceleration"] = pressure_acceleration
        feature_row[prefix + "spread_level"] = spread_mean
        feature_row[prefix + "spread_expansion"] = spread_expansion
        feature_row[prefix + "bid_depth_change"] = bid_depth_change
        feature_row[prefix + "ask_depth_change"] = ask_depth_change
        feature_row[prefix + "order_book_imbalance"] = mean_value(current_window, "order_book_imbalance_10bps")
        feature_row[prefix + "imbalance_change"] = imbalance_change
        feature_row[prefix + "liquidity_thinning_score"] = liquidity_thinning
        feature_row[prefix + "slippage_risk_score"] = slippage_risk
        feature_row[prefix + "failed_bounce_flag"] = failed_bounce
        feature_row[prefix + "failed_breakdown_flag"] = failed_breakdown

    feature_row["feature_current_mid_price"] = current_mid
    feature_row["feature_current_spread_percent"] = float(row["spread_percent"])
    feature_row["feature_current_bid_depth_10bps"] = float(row["bid_depth_10bps"])
    feature_row["feature_current_ask_depth_10bps"] = float(row["ask_depth_10bps"])
    feature_row["feature_current_order_book_imbalance_10bps"] = float(row["order_book_imbalance_10bps"])
    feature_row["feature_current_market_pressure_10s"] = float(row["market_pressure_10s"])
    return feature_row, None


def calculate_targets(frame, index, threshold):
    entry = frame.loc[index]
    entry_mid = float(entry["mid_price"])
    if not np.isfinite(entry_mid) or entry_mid <= 0:
        return None, "invalid entry mid price"

    future_10s = future_window(frame, index, 10)
    future_30s = future_window(frame, index, 30)
    future_60s = future_window(frame, index, 60)
    if len(future_10s) == 0:
        return None, "missing 10s future window"
    if len(future_30s) == 0 or len(future_60s) == 0:
        return None, "missing aftermath future window"

    future_mid_10 = pd.to_numeric(future_10s["mid_price"], errors="coerce")
    future_mid_30 = pd.to_numeric(future_30s["mid_price"], errors="coerce")
    future_mid_60 = pd.to_numeric(future_60s["mid_price"], errors="coerce")
    if future_mid_10.isna().any() or future_mid_30.isna().any() or future_mid_60.isna().any():
        return None, "missing future mid price"

    final_return_10 = safe_ratio(float(future_mid_10.iloc[-1]) - entry_mid, entry_mid)
    future_return_30 = safe_ratio(float(future_mid_30.iloc[-1]) - entry_mid, entry_mid)
    future_return_60 = safe_ratio(float(future_mid_60.iloc[-1]) - entry_mid, entry_mid)

    max_mid_10 = float(future_mid_10.max())
    min_mid_10 = float(future_mid_10.min())
    max_runup = safe_ratio(max_mid_10 - entry_mid, entry_mid)
    max_drawdown = safe_ratio(min_mid_10 - entry_mid, entry_mid)
    max_timestamp = int(future_10s.loc[future_mid_10.idxmax(), "timestamp"])
    min_timestamp = int(future_10s.loc[future_mid_10.idxmin(), "timestamp"])
    entry_timestamp = int(entry["timestamp"])
    time_to_max = max(1.0, (max_timestamp - entry_timestamp) / 1000.0)
    time_to_min = max(1.0, (min_timestamp - entry_timestamp) / 1000.0)
    upside_velocity = safe_ratio(max_runup, time_to_max)
    downside_velocity = safe_ratio(max_drawdown, time_to_min)

    current_spread = float(entry["spread_percent"])
    future_spread_max = float(pd.to_numeric(future_10s["spread_percent"], errors="coerce").max())
    future_spread_expansion = future_spread_max - current_spread
    current_bid_depth = float(entry["bid_depth_10bps"])
    current_ask_depth = float(entry["ask_depth_10bps"])
    future_bid_depth_change = safe_ratio(
        float(pd.to_numeric(future_10s["bid_depth_10bps"], errors="coerce").iloc[-1]) - current_bid_depth,
        current_bid_depth,
    )
    future_ask_depth_change = safe_ratio(
        float(pd.to_numeric(future_10s["ask_depth_10bps"], errors="coerce").iloc[-1]) - current_ask_depth,
        current_ask_depth,
    )
    future_bid_depth = float(pd.to_numeric(future_10s["bid_depth_10bps"], errors="coerce").iloc[-1])
    future_ask_depth = float(pd.to_numeric(future_10s["ask_depth_10bps"], errors="coerce").iloc[-1])
    future_bid_log_depth_change = float(
        np.clip(
            np.log((future_bid_depth + EPSILON) / (current_bid_depth + EPSILON)),
            -5.0,
            5.0,
        )
    )
    future_ask_log_depth_change = float(
        np.clip(
            np.log((future_ask_depth + EPSILON) / (current_ask_depth + EPSILON)),
            -5.0,
            5.0,
        )
    )

    past_60s = row_window(frame, index, 60)
    past_buy_baseline = max(volume_sum(past_60s, "market_buy_volume_10s") / 6.0, EPSILON)
    past_sell_baseline = max(volume_sum(past_60s, "market_sell_volume_10s") / 6.0, EPSILON)
    future_buy = volume_sum(future_10s, "market_buy_volume_10s")
    future_sell = volume_sum(future_10s, "market_sell_volume_10s")
    future_total_volume = max(volume_sum(future_10s, "total_trade_volume_10s"), EPSILON)
    future_pressure = safe_ratio(future_buy - future_sell, future_total_volume)
    past_10s_return = safe_ratio(
        float(entry["mid_price"]) - first_value(row_window(frame, index, 10), "mid_price", entry_mid),
        first_value(row_window(frame, index, 10), "mid_price", entry_mid),
    )

    upside_scare = max_runup >= threshold
    downside_scare = max_drawdown <= -threshold
    continuation_30s = (
        (upside_scare and future_return_30 > 0)
        or (downside_scare and future_return_30 < 0)
    )
    continuation_60s = (
        (upside_scare and future_return_60 > 0)
        or (downside_scare and future_return_60 < 0)
    )

    targets = {
        "future_return_10s": final_return_10,
        "max_runup_10s": max_runup,
        "max_drawdown_10s": max_drawdown,
        "upside_velocity_10s": upside_velocity,
        "downside_velocity_10s": downside_velocity,
        "future_spread_expansion_10s": future_spread_expansion,
        "future_bid_log_depth_change_10s": future_bid_log_depth_change,
        "future_ask_log_depth_change_10s": future_ask_log_depth_change,
        # Kept as diagnostics only. The default regression target set uses
        # clipped log-depth changes so one extreme book-depth jump does not
        # dominate the shared representation.
        "future_bid_depth_change_10s": future_bid_depth_change,
        "future_ask_depth_change_10s": future_ask_depth_change,
        "upside_scare_event_10s": int(upside_scare),
        "downside_scare_event_10s": int(downside_scare),
        "aggressive_buy_burst_10s": int(future_buy > 3.0 * past_buy_baseline and future_pressure > 0.35),
        "aggressive_sell_burst_10s": int(future_sell > 3.0 * past_sell_baseline and future_pressure < -0.35),
        "bid_liquidity_drop_10s": int(future_bid_depth_change <= -0.25),
        "ask_liquidity_drop_10s": int(future_ask_depth_change <= -0.25),
        "spread_expansion_event_10s": int(
            future_spread_expansion > max(current_spread, EPSILON) or future_spread_expansion > 0.001
        ),
        "direction_flip_10s": int(
            abs(past_10s_return) >= threshold / 3.0
            and abs(final_return_10) >= threshold / 3.0
            and np.sign(past_10s_return) != np.sign(final_return_10)
        ),
        "future_return_30s": future_return_30,
        "future_return_60s": future_return_60,
        "continuation_30s": int(continuation_30s),
        "continuation_60s": int(continuation_60s),
        "reversal_after_upside_scare_30s": int(upside_scare and future_return_30 <= -threshold / 2.0),
        "reversal_after_downside_scare_30s": int(downside_scare and future_return_30 >= threshold / 2.0),
        "reversal_after_upside_scare_60s": int(upside_scare and future_return_60 <= -threshold / 2.0),
        "reversal_after_downside_scare_60s": int(downside_scare and future_return_60 >= threshold / 2.0),
    }
    if not all(np.isfinite(float(value)) for value in targets.values()):
        return None, "non-finite target values"
    return targets, None


def build_training_rows(frame, threshold):
    snapshot_step_seconds = infer_snapshot_step_seconds(frame)
    rows = []
    skipped_reasons = {}
    for index in range(len(frame)):
        features, feature_reason = build_feature_row(frame, index, snapshot_step_seconds)
        if features is None:
            skipped_reasons[feature_reason] = skipped_reasons.get(feature_reason, 0) + 1
            continue
        targets, target_reason = calculate_targets(frame, index, threshold)
        if targets is None:
            skipped_reasons[target_reason] = skipped_reasons.get(target_reason, 0) + 1
            continue
        rows.append({**features, **targets})
    return pd.DataFrame(rows), skipped_reasons, snapshot_step_seconds


def build_latest_feature_frame(frame):
    snapshot_step_seconds = infer_snapshot_step_seconds(frame)
    rows = []
    for index in range(len(frame)):
        features, _ = build_feature_row(frame, index, snapshot_step_seconds)
        if features is not None:
            rows.append(features)
    return pd.DataFrame(rows), snapshot_step_seconds


def build_latest_feature_only(frame):
    snapshot_step_seconds = infer_snapshot_step_seconds(frame)
    for index in range(len(frame) - 1, -1, -1):
        features, _ = build_feature_row(frame, index, snapshot_step_seconds)
        if features is not None:
            return pd.DataFrame([features]), snapshot_step_seconds
    return pd.DataFrame(), snapshot_step_seconds


def is_optional_context_feature(column):
    return column.startswith("feature_context_")


def optional_context_default(column):
    if column.endswith("_context_age_ms"):
        return -1.0
    return 0.0


def get_micro_feature_columns(row_or_frame=None):
    """Return the canonical, deterministic microstructure model feature list.

    Training rows and one-row live prediction frames can contain different
    optional context columns depending on whether 3m/regime/HTF context is
    available. The model must still use one stable order, so every caller uses
    this resolver and the model artifact stores the exact resulting list.
    """
    if row_or_frame is None:
        return []

    if isinstance(row_or_frame, pd.Series):
        columns = list(row_or_frame.index)
        numeric_columns = []
        for column in columns:
            if not isinstance(column, str):
                continue
            if not column.startswith(MODEL_FEATURE_PREFIX):
                continue
            if column == "feature_ready":
                continue
            value = pd.to_numeric(pd.Series([row_or_frame[column]]), errors="coerce").iloc[0]
            if pd.notna(value) or is_optional_context_feature(column):
                numeric_columns.append(column)
        return sorted(set(numeric_columns))

    frame = row_or_frame
    columns = []
    for column in frame.columns:
        if not isinstance(column, str):
            continue
        if not column.startswith(MODEL_FEATURE_PREFIX):
            continue
        if column == "feature_ready":
            continue
        if is_optional_context_feature(column):
            columns.append(column)
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            columns.append(column)
    return sorted(set(columns))


def feature_columns(frame):
    return get_micro_feature_columns(frame)


def feature_schema_hash(columns):
    payload = json.dumps(list(columns), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def add_missing_optional_context_columns(feature_frame, columns):
    feature_frame = feature_frame.copy()
    for column in columns:
        if column not in feature_frame.columns and is_optional_context_feature(column):
            feature_frame[column] = optional_context_default(column)
        elif column == "feature_ready" and column not in feature_frame.columns:
            # Older candidate artifacts accidentally included this readiness
            # flag as a model input. New canonical schemas exclude it, but
            # supplying 1.0 keeps old paper models loadable without treating it
            # as a real market feature.
            feature_frame[column] = 1.0
    return feature_frame


def fill_optional_context_feature_values(feature_frame, columns=None):
    feature_frame = feature_frame.copy()
    selected_columns = columns if columns is not None else get_micro_feature_columns(feature_frame)
    for column in selected_columns:
        if column in feature_frame.columns and is_optional_context_feature(column):
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce").fillna(
                optional_context_default(column)
            )
        elif column == "feature_ready" and column in feature_frame.columns:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce").fillna(1.0)
    return feature_frame


def required_micro_feature_columns(columns):
    return [
        column
        for column in columns
        if not is_optional_context_feature(column)
        and column != "feature_ready"
    ]


def micro_schema_diagnostics(model_columns, current_columns):
    model_columns = list(model_columns)
    current_columns = list(current_columns)
    model_set = set(model_columns)
    current_set = set(current_columns)
    optional_model_columns = [
        column for column in model_columns if is_optional_context_feature(column)
    ]
    required_model_columns = required_micro_feature_columns(model_columns)
    differing = []
    for index in range(max(len(model_columns), len(current_columns))):
        model_column = model_columns[index] if index < len(model_columns) else "<missing>"
        current_column = current_columns[index] if index < len(current_columns) else "<missing>"
        if model_column != current_column:
            differing.append(
                {
                    "index": index,
                    "model": model_column,
                    "current": current_column,
                }
            )
        if len(differing) >= 20:
            break
    return {
        "model_count": len(model_columns),
        "current_count": len(current_columns),
        "required_missing_columns": sorted(
            column for column in required_model_columns if column not in current_set
        ),
        "optional_missing_columns_filled": sorted(
            column for column in optional_model_columns if column not in current_set
        ),
        "missing_from_current": sorted(model_set - current_set),
        "extra_in_current": sorted(current_set - model_set),
        "first_20_order_differences": differing,
        "artifact_feature_schema_hash": feature_schema_hash(model_columns),
        "current_required_feature_hash": feature_schema_hash(
            sorted(column for column in current_columns if column in set(required_model_columns))
        ),
    }


def print_micro_schema_diagnostics(diagnostics):
    print(f"Model feature count: {diagnostics['model_count']}")
    print(f"Current feature count: {diagnostics['current_count']}")
    print(f"Artifact feature_schema_hash: {diagnostics['artifact_feature_schema_hash']}")
    print(f"Current required-feature hash: {diagnostics['current_required_feature_hash']}")
    print("Required missing columns:")
    required_missing = diagnostics["required_missing_columns"]
    print(
        "- none"
        if not required_missing
        else "\n".join(f"- {column}" for column in required_missing[:50])
    )
    if len(required_missing) > 50:
        print(f"- ... {len(required_missing) - 50} more")
    print("Optional missing columns filled:")
    optional_missing = diagnostics["optional_missing_columns_filled"]
    print(
        "- none"
        if not optional_missing
        else "\n".join(f"- {column}" for column in optional_missing[:50])
    )
    if len(optional_missing) > 50:
        print(f"- ... {len(optional_missing) - 50} more")
    print("Extra current columns ignored:")
    extra = diagnostics["extra_in_current"]
    print("- none" if not extra else "\n".join(f"- {column}" for column in extra[:50]))
    if len(extra) > 50:
        print(f"- ... {len(extra) - 50} more")
    print("First ordered column differences:")
    differing = diagnostics["first_20_order_differences"]
    if not differing:
        print("- none")
    for item in differing:
        print(f"- index {item['index']}: model={item['model']} current={item['current']}")


def initialize_model(input_size, hidden_units, regression_outputs, event_outputs, rng):
    return {
        "w1": rng.normal(0, math.sqrt(2.0 / max(1, input_size)), (input_size, hidden_units)),
        "b1": np.zeros(hidden_units),
        "w_reg": rng.normal(0, math.sqrt(2.0 / max(1, hidden_units)), (hidden_units, regression_outputs)),
        "b_reg": np.zeros(regression_outputs),
        "w_event": rng.normal(0, math.sqrt(2.0 / max(1, hidden_units)), (hidden_units, event_outputs)),
        "b_event": np.zeros(event_outputs),
    }


def forward(model, x):
    hidden_pre = x @ model["w1"] + model["b1"]
    hidden = np.maximum(hidden_pre, 0.0)
    regression = hidden @ model["w_reg"] + model["b_reg"]
    event_logits = hidden @ model["w_event"] + model["b_event"]
    event_probabilities = sigmoid(event_logits)
    return hidden_pre, hidden, regression, event_probabilities


def forward_with_logits(model, x):
    hidden_pre = x @ model["w1"] + model["b1"]
    hidden = np.maximum(hidden_pre, 0.0)
    regression = hidden @ model["w_reg"] + model["b_reg"]
    event_logits = hidden @ model["w_event"] + model["b_event"]
    event_probabilities = sigmoid(event_logits)
    return hidden_pre, hidden, regression, event_logits, event_probabilities


def standardize(train_values, validation_values):
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std < EPSILON] = 1.0
    return (train_values - mean) / std, (validation_values - mean) / std, mean, std


def save_model(path, artifact):
    serializable = dict(artifact)
    serializable["model"] = {
        name: np.asarray(value).tolist()
        for name, value in artifact["model"].items()
    }
    for key in ["feature_mean", "feature_std", "target_mean", "target_std"]:
        if key in artifact:
            serializable[key] = np.asarray(artifact[key]).tolist()
    atomic_write_json(serializable, path)


def regression_scalers_from_arrays(regression_columns, mean, std, clip_bounds):
    scalers = {}
    for index, column in enumerate(regression_columns):
        clip_bound = clip_bounds.get(column)
        scalers[column] = {
            "mean": float(mean[index]),
            "std": float(std[index]),
            "clip_min": float(-clip_bound) if clip_bound is not None else None,
            "clip_max": float(clip_bound) if clip_bound is not None else None,
        }
    return scalers


def validate_regression_scalers(artifact, regression_columns=None, allow_legacy=False):
    regression_columns = list(regression_columns or artifact.get("regression_target_columns", REGRESSION_TARGET_COLUMNS))
    scalers = artifact.get("regression_target_scalers")
    if not isinstance(scalers, dict):
        if allow_legacy:
            return False, "legacy artifact missing regression_target_scalers"
        raise ValueError(
            "Model artifact is missing regression_target_scalers. "
            "Rebuild micro rows and retrain the 10s microstructure model."
        )
    missing = [column for column in regression_columns if column not in scalers]
    if missing:
        if allow_legacy:
            return False, f"legacy artifact missing scalers for {missing[:10]}"
        raise ValueError(f"Model artifact is missing regression scalers for: {missing[:10]}")
    for column in regression_columns:
        scaler = scalers[column]
        mean = float(scaler.get("mean", np.nan))
        std = float(scaler.get("std", np.nan))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= EPSILON:
            raise ValueError(f"Invalid regression scaler for {column}: mean={mean}, std={std}")
    return True, "ok"


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
        if key not in artifact:
            artifact[key] = np.asarray([], dtype=np.float64)
            continue
        artifact[key] = np.asarray(artifact[key], dtype=np.float64)
        artifact[key][np.abs(artifact[key]) < EPSILON] = np.where(
            key.endswith("std"),
            1.0,
            artifact[key][np.abs(artifact[key]) < EPSILON],
        )
    artifact["feature_std"][artifact["feature_std"] < EPSILON] = 1.0
    if len(artifact["target_std"]):
        artifact["target_std"][artifact["target_std"] < EPSILON] = 1.0
    return artifact


def predict_with_artifact(artifact, feature_frame, event_temperature=None):
    columns = artifact["feature_columns"]
    feature_frame = add_missing_optional_context_columns(feature_frame, columns)
    missing = [column for column in columns if column not in feature_frame.columns]
    if missing:
        raise ValueError(f"Feature rows are missing model columns: {missing[:10]}")
    feature_frame = fill_optional_context_feature_values(feature_frame, columns)
    x = feature_frame[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    if np.isnan(x).any():
        raise ValueError("Feature rows contain NaN values for model columns.")
    x = (x - artifact["feature_mean"]) / artifact["feature_std"]
    _, _, regression_scaled, event_logits, event_probabilities = forward_with_logits(artifact["model"], x)
    temperature = (
        float(event_temperature)
        if event_temperature is not None
        else float(artifact.get("event_probability_temperature", 1.0))
    )
    if not np.isfinite(temperature) or temperature <= 0:
        temperature = 1.0
    event_probabilities = sigmoid(event_logits / temperature)

    regression_columns = list(artifact.get("regression_target_columns", REGRESSION_TARGET_COLUMNS))
    event_only = bool(artifact.get("event_only", False))
    if event_only:
        regression_columns = []
    if regression_columns:
        validate_regression_scalers(artifact, regression_columns, allow_legacy=False)
        scalers = artifact["regression_target_scalers"]
        regression = np.zeros((len(feature_frame), len(regression_columns)), dtype=np.float64)
        for index, column in enumerate(regression_columns):
            scaler = scalers[column]
            regression[:, index] = (
                regression_scaled[:, index] * float(scaler["std"])
                + float(scaler["mean"])
            )
    else:
        regression = np.zeros((len(feature_frame), 0), dtype=np.float64)
    return regression, event_probabilities


def regression_sanity_report(
    regression_values,
    regression_columns,
    artifact=None,
    extreme_multiplier=3.0,
):
    values = np.asarray(regression_values, dtype=np.float64).reshape(-1)
    regression_columns = list(regression_columns)
    scalers = artifact.get("regression_target_scalers", {}) if artifact else {}
    clipped = {}
    warnings = []
    failed = False
    for index, column in enumerate(regression_columns):
        value = float(values[index])
        scaler = scalers.get(column, {}) if isinstance(scalers, dict) else {}
        clip_min = scaler.get("clip_min")
        clip_max = scaler.get("clip_max")
        bound = regression_bound_for_target(column)
        if clip_min is None or clip_max is None:
            if bound is None:
                clipped[column] = value
                continue
            clip_min, clip_max = -bound, bound
        clip_min = float(clip_min)
        clip_max = float(clip_max)
        clipped_value = float(np.clip(value, clip_min, clip_max))
        clipped[column] = clipped_value
        if not np.isfinite(value):
            warnings.append(f"{column} is non-finite")
            failed = True
            continue
        if clipped_value != value:
            warnings.append(
                f"{column} clipped from {value:.6g} to {clipped_value:.6g}"
            )
            max_abs = max(abs(clip_min), abs(clip_max), EPSILON)
            if abs(value) > max_abs * extreme_multiplier:
                failed = True
    return {
        "ok": not warnings,
        "warning": bool(warnings),
        "failed": failed,
        "clipped_values": clipped,
        "warnings": warnings,
    }


def event_saturation_report(event_probabilities, event_columns=None, warning_threshold=0.30, extreme_threshold=0.50):
    probabilities = np.asarray(event_probabilities, dtype=np.float64).reshape(-1)
    event_columns = list(event_columns or EVENT_TARGET_COLUMNS)
    low_or_high = (probabilities <= 0.01) | (probabilities >= 0.99)
    fraction_saturated = float(low_or_high.mean()) if len(probabilities) else 0.0
    saturated_events = [
        event_columns[index]
        for index, saturated in enumerate(low_or_high)
        if saturated and index < len(event_columns)
    ]
    return {
        "ok": fraction_saturated <= warning_threshold,
        "warning": fraction_saturated > warning_threshold,
        "failed": fraction_saturated > extreme_threshold,
        "fraction_saturated": fraction_saturated,
        "saturated_events": saturated_events,
    }


def predictions_frame(
    base_frame,
    regression,
    event_probabilities,
    regression_columns=None,
    event_columns=None,
):
    regression_columns = (
        list(REGRESSION_TARGET_COLUMNS)
        if regression_columns is None
        else list(regression_columns)
    )
    event_columns = (
        list(EVENT_TARGET_COLUMNS)
        if event_columns is None
        else list(event_columns)
    )
    output = base_frame[["timestamp", "time"]].copy()
    if regression is not None:
        regression = np.asarray(regression, dtype=np.float64)
        if regression.ndim == 1:
            regression = regression.reshape(len(output), -1)
    if regression_columns and regression is not None and regression.shape[1] > 0:
        usable_columns = regression_columns[: regression.shape[1]]
        for index, column in enumerate(usable_columns):
            output[f"pred_{column}"] = regression[:, index]
    for index, column in enumerate(event_columns):
        output[f"prob_{column}"] = event_probabilities[:, index]
    return output


def precision_recall(actual, probability, threshold=0.5):
    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(probability >= threshold, dtype=np.int64)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return precision, recall, tp, fp, fn


def current_utc_tag():
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def copy_if_promoted(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
