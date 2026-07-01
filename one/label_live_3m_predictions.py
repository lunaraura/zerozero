import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from live_3m_model_utils import (
    ACTUAL_LABEL_COLUMNS,
    LIVE_PREDICTION_COLUMNS,
    atomic_write_csv,
    coerce_feature_ready,
    coerce_numeric_columns,
    parse_bool,
    safe_ratio,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "0.003"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.0015"))
RETURN_CLASS_THRESHOLD = float(os.getenv("RETURN_CLASS_THRESHOLD", "0.0005"))
FUTURE_RETURN_CLASS_MODE = os.getenv("FUTURE_RETURN_CLASS_MODE", "fixed").strip().lower()
if FUTURE_RETURN_CLASS_MODE not in {"fixed", "volatility"}:
    raise ValueError("FUTURE_RETURN_CLASS_MODE must be fixed or volatility")
VOL_LOOKBACK_MINUTES = int(os.getenv("VOL_LOOKBACK_MINUTES", "240"))
MIN_VOL_LOOKBACK_ROWS = int(os.getenv("MIN_VOL_LOOKBACK_ROWS", "60"))
VOL_MULTIPLIER = float(os.getenv("VOL_MULTIPLIER", "0.50"))
MIN_CLASS_MOVE = float(os.getenv("MIN_CLASS_MOVE", "0.0002"))

PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "live_predictions" / f"{SYMBOL}_live_3m_predictions.csv"
)
REALTIME_DIR = PROJECT_ROOT / "data" / "realtime"
VENUE_REALTIME_DIR = REALTIME_DIR / PRIMARY_VENUE if PRIMARY_VENUE else REALTIME_DIR
RAW_1M_PATH = VENUE_REALTIME_DIR / f"{SYMBOL}_1m_flow.csv"
FEATURES_PATH = VENUE_REALTIME_DIR / f"{SYMBOL}_1m_flow_features.csv"
LABELED_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "live_training" / f"{SYMBOL}_labeled_3m_training_rows.csv"
)
LABELED_BACKUP_PATH = LABELED_OUTPUT_PATH.with_suffix(
    LABELED_OUTPUT_PATH.suffix + ".bak"
)

REQUIRED_ACTUAL_TARGET_COLUMNS = ACTUAL_LABEL_COLUMNS + [
    "actual_future_return_class",
    "actual_path_event_class",
]
CORE_ACTUAL_TARGET_COLUMNS = [
    "actual_future_return_3",
    "actual_future_return_class",
    "actual_path_event_class",
    "actual_future_range_percent_3",
    "actual_future_volume_3m",
    "actual_future_trade_count_3m",
    "actual_market_buy_volume_3m",
    "actual_market_sell_volume_3m",
    "actual_volume_delta_3m",
    "actual_market_pressure_3m",
    "actual_path_class",
]
DERIVED_ACTUAL_TARGET_COLUMNS = [
    column
    for column in ACTUAL_LABEL_COLUMNS
    if column not in CORE_ACTUAL_TARGET_COLUMNS
]
READINESS_COLUMNS = [
    "core_label_ready",
    "full_regression_label_ready",
]
REGIME_LABEL_COLUMNS = [
    "actual_regime_class_15m",
    "actual_regime_class_30m",
    "actual_future_return_15m",
    "actual_future_return_30m",
    "actual_max_drawdown_15m",
    "actual_max_runup_15m",
    "actual_trend_slope_15m",
    "actual_trend_slope_30m",
]
EXTRA_CLASS_LABEL_COLUMNS = [
    "actual_future_return_class",
    "actual_path_event_class",
]
FUTURE_RETURN_CLASS_DEBUG_COLUMNS = [
    "future_return_class_threshold",
    "future_return_class_vol_3m",
    "future_return_class_mode",
    "future_return_class_vol_multiplier",
]
STRING_COLUMNS = [
    "future_return_class_mode",
]
FLOAT_COLUMNS = [
    "future_return_class_threshold",
    "future_return_class_vol_3m",
    "future_return_class_vol_multiplier",
]
ALL_LABEL_COLUMNS = (
    ACTUAL_LABEL_COLUMNS
    + EXTRA_CLASS_LABEL_COLUMNS
    + REGIME_LABEL_COLUMNS
    + FUTURE_RETURN_CLASS_DEBUG_COLUMNS
)


def normalize_debug_column_dtypes(frame):
    for column in STRING_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].astype("object")

    for column in FLOAT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def load_predictions():
    if not PREDICTIONS_PATH.exists():
        return normalize_debug_column_dtypes(pd.DataFrame(
            columns=LIVE_PREDICTION_COLUMNS + ALL_LABEL_COLUMNS + READINESS_COLUMNS
        ))
    frame = pd.read_csv(PREDICTIONS_PATH)
    for column in LIVE_PREDICTION_COLUMNS + ALL_LABEL_COLUMNS + READINESS_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = normalize_debug_column_dtypes(frame)
    frame["label_ready"] = frame["label_ready"].apply(parse_bool)
    for column in READINESS_COLUMNS:
        frame[column] = frame[column].apply(parse_bool)
    non_numeric_columns = {"time", "label_ready", "future_return_class_mode", *READINESS_COLUMNS}
    for column in frame.columns:
        if column not in non_numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame.reset_index(drop=True)


def load_raw_rows():
    if not RAW_1M_PATH.exists():
        raise FileNotFoundError(f"Missing raw realtime file: {RAW_1M_PATH}")
    frame = pd.read_csv(RAW_1M_PATH)
    frame = coerce_numeric_columns(frame)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame.reset_index(drop=True)


def load_feature_rows(raw_rows):
    if FEATURES_PATH.exists():
        frame = pd.read_csv(FEATURES_PATH)
        frame = coerce_numeric_columns(frame)
        frame = coerce_feature_ready(frame)
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        return frame.reset_index(drop=True)

    return pd.DataFrame(columns=["timestamp"])


def complete_actual_target_mask(frame, columns=None):
    columns = columns or REQUIRED_ACTUAL_TARGET_COLUMNS
    missing_columns = [
        column for column in columns if column not in frame.columns
    ]
    if missing_columns:
        return pd.Series(False, index=frame.index)

    numeric_targets = frame[columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    complete = numeric_targets.notna().all(axis=1)
    complete = complete & numeric_targets.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    for class_column in [
        "actual_path_class",
        "actual_future_return_class",
        "actual_path_event_class",
    ]:
        if class_column in numeric_targets.columns:
            complete = complete & numeric_targets[class_column].isin([0, 1, 2])
    return complete


def has_raw_future_window(prediction_timestamp, raw_by_timestamp):
    required_timestamps = [
        prediction_timestamp,
        prediction_timestamp + 60_000,
        prediction_timestamp + 120_000,
        prediction_timestamp + 180_000,
    ]
    return all(timestamp in raw_by_timestamp.index for timestamp in required_timestamps)


def has_raw_future_window_for_horizon(prediction_timestamp, raw_by_timestamp, horizon_minutes):
    required_timestamps = [
        prediction_timestamp + 60_000 * minute
        for minute in range(0, horizon_minutes + 1)
    ]
    return all(timestamp in raw_by_timestamp.index for timestamp in required_timestamps)


def has_future_feature_window(prediction_timestamp, features_by_timestamp):
    expected_future_timestamps = [
        prediction_timestamp + 60_000,
        prediction_timestamp + 120_000,
        prediction_timestamp + 180_000,
    ]
    return all(timestamp in features_by_timestamp.index for timestamp in expected_future_timestamps)


def backup_existing_training_csv():
    if not LABELED_OUTPUT_PATH.exists():
        return False

    LABELED_BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LABELED_OUTPUT_PATH, LABELED_BACKUP_PATH)
    return True


def path_class(entry_close, future_rows):
    long_take_profit = entry_close * (1.0 + TAKE_PROFIT)
    long_stop_loss = entry_close * (1.0 - STOP_LOSS)
    short_take_profit = entry_close * (1.0 - TAKE_PROFIT)
    short_stop_loss = entry_close * (1.0 + STOP_LOSS)

    for _, row in future_rows.iterrows():
        high = row["high"]
        low = row["low"]
        long_tp_hit = high >= long_take_profit
        long_sl_hit = low <= long_stop_loss
        short_tp_hit = low <= short_take_profit
        short_sl_hit = high >= short_stop_loss

        if (long_tp_hit and short_tp_hit) or (
            (long_tp_hit or short_tp_hit) and (long_sl_hit or short_sl_hit)
        ):
            return 1
        if long_tp_hit and not long_sl_hit:
            return 2
        if short_tp_hit and not short_sl_hit:
            return 0
        if long_sl_hit or short_sl_hit:
            return 1

    return 1


def fixed_future_return_class(future_return, threshold):
    # Fixed mode preserves the original strict > / < behavior.
    if future_return > threshold:
        return 2
    if future_return < -threshold:
        return 0
    return 1


def volatility_scaled_threshold(prediction_timestamp, raw_rows):
    lookback_start = prediction_timestamp - VOL_LOOKBACK_MINUTES * 60_000
    past = raw_rows[
        (raw_rows["timestamp"] <= prediction_timestamp)
        & (raw_rows["timestamp"] >= lookback_start)
    ].copy()
    if "close" not in past.columns:
        return None, None, "missing close column for volatility threshold"

    past = past.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    closes = pd.to_numeric(past["close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    returns = closes.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < MIN_VOL_LOOKBACK_ROWS:
        return (
            None,
            None,
            "insufficient volatility lookback rows "
            f"({len(returns)} < {MIN_VOL_LOOKBACK_ROWS})",
        )

    vol_1m = float(returns.std(ddof=0))
    if not np.isfinite(vol_1m):
        return None, None, "non-finite volatility threshold"
    vol_3m = vol_1m * np.sqrt(3.0)
    threshold = max(MIN_CLASS_MOVE, VOL_MULTIPLIER * vol_3m)
    return float(threshold), float(vol_3m), None


def future_return_class_metadata(prediction_timestamp, future_return, raw_rows):
    if FUTURE_RETURN_CLASS_MODE == "volatility":
        threshold, vol_3m, reason = volatility_scaled_threshold(
            prediction_timestamp,
            raw_rows,
        )
        if threshold is None:
            return None, reason
        if future_return >= threshold:
            class_id = 2
        elif future_return <= -threshold:
            class_id = 0
        else:
            class_id = 1
    else:
        threshold = RETURN_CLASS_THRESHOLD
        vol_3m = np.nan
        class_id = fixed_future_return_class(future_return, threshold)

    return {
        "actual_future_return_class": int(class_id),
        "future_return_class_threshold": float(threshold),
        "future_return_class_vol_3m": vol_3m,
        "future_return_class_mode": str(FUTURE_RETURN_CLASS_MODE),
        "future_return_class_vol_multiplier": float(VOL_MULTIPLIER),
    }, None


def linear_regression_slope(values):
    numeric_values = np.asarray(values, dtype=np.float64)
    if len(numeric_values) < 2:
        return 0.0
    x = np.arange(len(numeric_values), dtype=np.float64)
    slope, _ = np.polyfit(x, numeric_values, 1)
    if not np.isfinite(slope):
        return 0.0
    return float(slope)


def classify_regime(
    future_return,
    projected_slope,
    max_drawdown,
    max_runup,
    close_position,
    higher_high,
    higher_low,
    lower_high,
    lower_low,
    threshold,
):
    structure_score = 0.0
    if higher_high and higher_low:
        structure_score += threshold * 0.75
    elif higher_high or higher_low:
        structure_score += threshold * 0.25
    if lower_high and lower_low:
        structure_score -= threshold * 0.75
    elif lower_high or lower_low:
        structure_score -= threshold * 0.25

    range_location_score = (close_position - 0.5) * threshold
    excursion_balance = (max_runup - abs(max_drawdown)) * 0.20
    score = (
        future_return * 0.45
        + projected_slope * 0.25
        + excursion_balance
        + range_location_score
        + structure_score
    )

    if score > threshold and future_return > -threshold * 0.50:
        return 2
    if score < -threshold and future_return < threshold * 0.50:
        return 0
    return 1


def calculate_single_regime_label(prediction_timestamp, raw_by_timestamp, horizon_minutes):
    expected_future_timestamps = [
        prediction_timestamp + 60_000 * minute
        for minute in range(1, horizon_minutes + 1)
    ]
    required_timestamps = [prediction_timestamp] + expected_future_timestamps
    if any(timestamp not in raw_by_timestamp.index for timestamp in required_timestamps):
        return None

    entry = raw_by_timestamp.loc[prediction_timestamp]
    future = raw_by_timestamp.loc[expected_future_timestamps]
    required_columns = ["close", "high", "low"]
    raw_values = pd.concat([entry[required_columns].to_frame().T, future[required_columns]])
    raw_values = raw_values.apply(pd.to_numeric, errors="coerce")
    if raw_values.replace([np.inf, -np.inf], np.nan).isna().any().any():
        return None

    entry_close = float(entry["close"])
    final_close = float(future["close"].iloc[-1])
    if not np.isfinite(entry_close) or abs(entry_close) < 1e-12:
        return None

    future_return = safe_ratio(final_close - entry_close, entry_close)
    max_drawdown = safe_ratio(float(future["low"].min()) - entry_close, entry_close)
    max_runup = safe_ratio(float(future["high"].max()) - entry_close, entry_close)

    close_path = pd.concat(
        [
            pd.Series([entry_close]),
            pd.to_numeric(future["close"], errors="coerce"),
        ],
        ignore_index=True,
    )
    normalized_close_path = close_path / entry_close - 1.0
    slope_per_minute = linear_regression_slope(normalized_close_path)
    projected_slope = slope_per_minute * horizon_minutes

    rolling_high = float(future["high"].max())
    rolling_low = float(future["low"].min())
    close_position = safe_ratio(final_close - rolling_low, rolling_high - rolling_low)
    close_position = clip(close_position, 0.0, 1.0)

    midpoint = max(1, horizon_minutes // 2)
    first_half = future.iloc[:midpoint]
    second_half = future.iloc[midpoint:]
    if len(second_half) == 0:
        second_half = future.iloc[-1:]
    higher_high = float(second_half["high"].max()) > float(first_half["high"].max())
    higher_low = float(second_half["low"].min()) > float(first_half["low"].min())
    lower_high = float(second_half["high"].max()) < float(first_half["high"].max())
    lower_low = float(second_half["low"].min()) < float(first_half["low"].min())

    threshold = 0.0010 if horizon_minutes <= 15 else 0.0015
    regime_class = classify_regime(
        future_return,
        projected_slope,
        max_drawdown,
        max_runup,
        close_position,
        higher_high,
        higher_low,
        lower_high,
        lower_low,
        threshold,
    )

    return {
        "regime_class": regime_class,
        "future_return": future_return,
        "max_drawdown": max_drawdown,
        "max_runup": max_runup,
        "trend_slope": slope_per_minute,
    }


def calculate_regime_labels(prediction_timestamp, raw_by_timestamp):
    labels = {}
    regime_15m = calculate_single_regime_label(
        prediction_timestamp,
        raw_by_timestamp,
        15,
    )
    if regime_15m is not None:
        labels.update(
            {
                "actual_regime_class_15m": regime_15m["regime_class"],
                "actual_future_return_15m": regime_15m["future_return"],
                "actual_max_drawdown_15m": regime_15m["max_drawdown"],
                "actual_max_runup_15m": regime_15m["max_runup"],
                "actual_trend_slope_15m": regime_15m["trend_slope"],
            }
        )

    regime_30m = calculate_single_regime_label(
        prediction_timestamp,
        raw_by_timestamp,
        30,
    )
    if regime_30m is not None:
        labels.update(
            {
                "actual_regime_class_30m": regime_30m["regime_class"],
                "actual_future_return_30m": regime_30m["future_return"],
                "actual_trend_slope_30m": regime_30m["trend_slope"],
            }
        )

    return labels


def safe_numeric_value(value, default=0.0):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric_value):
        return default
    return numeric_value


def numeric_label_series(labels):
    numeric_labels = {
        key: value
        for key, value in labels.items()
        if key not in STRING_COLUMNS
    }
    return pd.Series(numeric_labels).apply(pd.to_numeric, errors="coerce")


def clip(value, lower, upper):
    return float(min(max(value, lower), upper))


def build_raw_derived_features(raw_until_future_end):
    """
    Build only the derived columns needed by the live label targets.

    This deliberately uses rows only up to the label horizon being computed.
    That keeps the label calculation independent from any later rows that may
    already exist in the live CSV.
    """
    frame = raw_until_future_end.copy().sort_values("timestamp").reset_index(drop=True)
    for column in frame.columns:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    optional_defaults = {
        "snapshot_market_pressure_10s_last": 0.0,
        "snapshot_imbalance_10bps_last": 0.0,
        "snapshot_imbalance_25bps_last": 0.0,
        "order_book_imbalance_25bps": 0.0,
        "bid_depth_25bps": 0.0,
        "ask_depth_25bps": 0.0,
    }
    for column, default in optional_defaults.items():
        if column not in frame.columns:
            frame[column] = default

    candle_range = frame["high"] - frame["low"]
    close_position = pd.Series(0.5, index=frame.index, dtype="float64")
    valid_range = candle_range.abs() > 1e-12
    close_position.loc[valid_range] = (
        (frame.loc[valid_range, "close"] - frame.loc[valid_range, "low"])
        / candle_range.loc[valid_range]
    )
    close_position = close_position.clip(0.0, 1.0)
    directional_close_position = close_position * 2.0 - 1.0

    market_pressure = pd.Series(0.0, index=frame.index, dtype="float64")
    valid_volume = frame["volume"].abs() > 1e-12
    market_pressure.loc[valid_volume] = (
        (frame.loc[valid_volume, "taker_buy_volume"] - frame.loc[valid_volume, "taker_sell_volume"])
        / frame.loc[valid_volume, "volume"]
    )
    market_pressure = market_pressure.clip(-1.0, 1.0)
    market_pressure_change = market_pressure.diff().fillna(0.0).clip(-2.0, 2.0)
    imbalance_10_change = (
        frame["order_book_imbalance_10bps"].diff().fillna(0.0).clip(-2.0, 2.0)
    )

    depth_total_10bps = frame["bid_depth_10bps"] + frame["ask_depth_10bps"]
    bid_share_10bps = pd.Series(0.5, index=frame.index, dtype="float64")
    ask_share_10bps = pd.Series(0.5, index=frame.index, dtype="float64")
    valid_depth_10bps = depth_total_10bps.abs() > 1e-12
    bid_share_10bps.loc[valid_depth_10bps] = (
        frame.loc[valid_depth_10bps, "bid_depth_10bps"] / depth_total_10bps.loc[valid_depth_10bps]
    )
    ask_share_10bps.loc[valid_depth_10bps] = (
        frame.loc[valid_depth_10bps, "ask_depth_10bps"] / depth_total_10bps.loc[valid_depth_10bps]
    )

    depth_total_25bps = frame["bid_depth_25bps"] + frame["ask_depth_25bps"]
    bid_share_25bps = pd.Series(0.5, index=frame.index, dtype="float64")
    ask_share_25bps = pd.Series(0.5, index=frame.index, dtype="float64")
    valid_depth_25bps = depth_total_25bps.abs() > 1e-12
    bid_share_25bps.loc[valid_depth_25bps] = (
        frame.loc[valid_depth_25bps, "bid_depth_25bps"] / depth_total_25bps.loc[valid_depth_25bps]
    )
    ask_share_25bps.loc[valid_depth_25bps] = (
        frame.loc[valid_depth_25bps, "ask_depth_25bps"] / depth_total_25bps.loc[valid_depth_25bps]
    )

    range_percent = pd.Series(0.0, index=frame.index, dtype="float64")
    valid_close = frame["close"].abs() > 1e-12
    range_percent.loc[valid_close] = candle_range.loc[valid_close] / frame.loc[valid_close, "close"]
    range_rank = range_percent.rank(pct=True).fillna(0.5)

    snapshot_pressure = frame["snapshot_market_pressure_10s_last"].fillna(0.0)
    snapshot_imbalance_10 = frame["snapshot_imbalance_10bps_last"].fillna(0.0)
    snapshot_imbalance_25 = frame["snapshot_imbalance_25bps_last"].fillna(0.0)

    flow_pressure_index = (
        market_pressure
        + frame["order_book_imbalance_10bps"]
        + frame["order_book_imbalance_25bps"]
        + snapshot_pressure
    ) / 4.0
    liquidity_support_index = (bid_share_10bps + bid_share_25bps) / 2.0
    liquidity_resistance_index = (ask_share_10bps + ask_share_25bps) / 2.0

    frame["breakout_pressure_index"] = (
        flow_pressure_index
        + directional_close_position
        + market_pressure_change
        + imbalance_10_change
    ) / 4.0
    frame["absorption_index"] = (
        liquidity_resistance_index * market_pressure.clip(0.0, 1.0)
        + liquidity_support_index * (-market_pressure).clip(0.0, 1.0)
        + (1.0 - range_rank)
        + (snapshot_imbalance_10.abs() + snapshot_imbalance_25.abs()) / 2.0
    ) / 4.0

    return frame


def calculate_derived_labels_from_raw(
    prediction_timestamp,
    raw_rows,
    raw_by_timestamp,
    entry,
    future,
):
    future_end_timestamp = prediction_timestamp + 180_000
    raw_until_future_end = raw_rows[raw_rows["timestamp"] <= future_end_timestamp].copy()
    feature_context = build_raw_derived_features(raw_until_future_end)
    feature_context = feature_context.set_index("timestamp", drop=False)
    expected_future_timestamps = [
        prediction_timestamp + 60_000,
        prediction_timestamp + 120_000,
        prediction_timestamp + 180_000,
    ]

    if not all(timestamp in feature_context.index for timestamp in expected_future_timestamps):
        return None, "missing derived future features computed from raw rows"

    future_features = feature_context.loc[expected_future_timestamps]
    required_feature_columns = [
        "order_book_imbalance_10bps",
        "breakout_pressure_index",
        "absorption_index",
    ]
    feature_values = future_features[required_feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if feature_values.replace([np.inf, -np.inf], np.nan).isna().any().any():
        return None, "missing or non-finite derived future feature values"

    def feature_mean(column):
        return float(pd.to_numeric(future_features[column], errors="coerce").mean())

    def raw_last_change(column):
        start = safe_numeric_value(entry[column], np.nan)
        end = safe_numeric_value(future[column].iloc[-1], np.nan)
        if not np.isfinite(start) or not np.isfinite(end):
            return np.nan
        return safe_ratio(end - start, start)

    labels = {
        "actual_order_book_imbalance_10bps_3m": feature_mean(
            "order_book_imbalance_10bps"
        ),
        "actual_bid_depth_change_10bps_3m": raw_last_change("bid_depth_10bps"),
        "actual_ask_depth_change_10bps_3m": raw_last_change("ask_depth_10bps"),
        "actual_spread_percent_3m": float(future["spread_percent"].mean())
        if "spread_percent" in future.columns
        else 0.0,
        "actual_breakout_pressure_3m": feature_mean("breakout_pressure_index"),
        "actual_absorption_3m": feature_mean("absorption_index"),
    }
    numeric_labels = pd.Series(labels).apply(pd.to_numeric, errors="coerce")
    if numeric_labels.replace([np.inf, -np.inf], np.nan).isna().any():
        missing = numeric_labels[numeric_labels.isna()].index.tolist()
        return None, f"missing derived actual target values: {', '.join(missing)}"

    return labels, None


def calculate_labels(prediction_timestamp, raw_rows, raw_by_timestamp, features_by_timestamp):
    expected_future_timestamps = [
        prediction_timestamp + 60_000,
        prediction_timestamp + 120_000,
        prediction_timestamp + 180_000,
    ]
    required_timestamps = [prediction_timestamp] + expected_future_timestamps

    if any(timestamp not in raw_by_timestamp.index for timestamp in required_timestamps):
        return None, False, False, "incomplete future window"

    entry = raw_by_timestamp.loc[prediction_timestamp]
    future = raw_by_timestamp.loc[expected_future_timestamps]
    required_raw_columns = [
        "close",
        "high",
        "low",
        "volume",
        "trade_count",
        "taker_buy_volume",
        "taker_sell_volume",
        "spread_percent",
        "bid_depth_10bps",
        "ask_depth_10bps",
    ]
    missing_raw_columns = [
        column for column in required_raw_columns if column not in raw_by_timestamp.columns
    ]
    if missing_raw_columns:
        return None, False, False, f"missing raw columns: {', '.join(missing_raw_columns)}"

    raw_values = pd.concat([entry[required_raw_columns].to_frame().T, future[required_raw_columns]])
    raw_values = raw_values.apply(pd.to_numeric, errors="coerce")
    if raw_values.replace([np.inf, -np.inf], np.nan).isna().any().any():
        return None, False, False, "missing or non-finite raw values"

    entry_close = float(entry["close"])
    future_close = float(future["close"].iloc[-1])
    high = float(future["high"].max())
    low = float(future["low"].min())
    volume = float(future["volume"].sum())
    trade_count = float(future["trade_count"].sum())
    market_buy_volume = float(future["taker_buy_volume"].sum())
    market_sell_volume = float(future["taker_sell_volume"].sum())
    volume_delta = market_buy_volume - market_sell_volume
    future_return = safe_ratio(future_close - entry_close, entry_close)
    event_class = path_class(entry_close, future)
    class_metadata, class_reason = future_return_class_metadata(
        prediction_timestamp,
        future_return,
        raw_rows,
    )
    if class_metadata is None:
        return None, False, False, class_reason or "missing future return class threshold"

    labels = {
        "actual_future_return_3": future_return,
        **class_metadata,
        "actual_path_event_class": event_class,
        "actual_future_range_percent_3": safe_ratio(high - low, entry_close),
        "actual_future_volume_3m": volume,
        "actual_future_trade_count_3m": trade_count,
        "actual_market_buy_volume_3m": market_buy_volume,
        "actual_market_sell_volume_3m": market_sell_volume,
        "actual_volume_delta_3m": volume_delta,
        "actual_market_pressure_3m": safe_ratio(volume_delta, volume),
        # Keep actual_path_class for backward compatibility. New training can
        # choose either actual_future_return_class or actual_path_event_class
        # explicitly instead of forcing one ambiguous label to do both jobs.
        "actual_path_class": event_class,
    }
    numeric_core_labels = numeric_label_series(labels)
    if numeric_core_labels.replace([np.inf, -np.inf], np.nan).isna().any():
        missing = numeric_core_labels[numeric_core_labels.isna()].index.tolist()
        return labels, True, False, f"missing core actual target values: {', '.join(missing)}"

    derived_labels = None
    derived_reason = None
    derived_source = "raw-computed"

    if all(timestamp in features_by_timestamp.index for timestamp in expected_future_timestamps):
        future_features = features_by_timestamp.loc[expected_future_timestamps]
        required_feature_columns = [
            "order_book_imbalance_10bps",
            "breakout_pressure_index",
            "absorption_index",
        ]
        missing_feature_columns = [
            column for column in required_feature_columns if column not in future_features.columns
        ]
        if not missing_feature_columns:
            feature_values = future_features[required_feature_columns].apply(
                pd.to_numeric,
                errors="coerce",
            )
            if not feature_values.replace([np.inf, -np.inf], np.nan).isna().any().any():
                def feature_mean(column):
                    return float(pd.to_numeric(future_features[column], errors="coerce").mean())

                derived_labels = {
                    "actual_order_book_imbalance_10bps_3m": feature_mean(
                        "order_book_imbalance_10bps"
                    ),
                    "actual_bid_depth_change_10bps_3m": safe_ratio(
                        float(future["bid_depth_10bps"].iloc[-1]) - float(entry["bid_depth_10bps"]),
                        float(entry["bid_depth_10bps"]),
                    ),
                    "actual_ask_depth_change_10bps_3m": safe_ratio(
                        float(future["ask_depth_10bps"].iloc[-1]) - float(entry["ask_depth_10bps"]),
                        float(entry["ask_depth_10bps"]),
                    ),
                    "actual_spread_percent_3m": float(future["spread_percent"].mean())
                    if "spread_percent" in future.columns
                    else 0.0,
                    "actual_breakout_pressure_3m": feature_mean("breakout_pressure_index"),
                    "actual_absorption_3m": feature_mean("absorption_index"),
                }
                derived_source = "feature-file"
            else:
                derived_reason = "missing or non-finite future feature values"
        else:
            derived_reason = f"missing feature columns: {', '.join(missing_feature_columns)}"

    if derived_labels is None:
        derived_labels, derived_reason = calculate_derived_labels_from_raw(
            prediction_timestamp,
            raw_rows,
            raw_by_timestamp,
            entry,
            future,
        )
        derived_source = "raw-computed"

    if derived_labels is None:
        return labels, True, False, derived_reason or "missing derived future features"

    labels.update(derived_labels)
    numeric_labels = numeric_label_series(labels)
    if numeric_labels.replace([np.inf, -np.inf], np.nan).isna().any():
        missing = numeric_labels[numeric_labels.isna()].index.tolist()
        return labels, True, False, f"missing actual target values: {', '.join(missing)}"

    return labels, True, True, derived_source


def print_class_distribution(classes):
    counts = pd.Series(classes).dropna().astype(int).value_counts().sort_index()
    total = int(counts.sum()) or 1
    for class_id in [0, 1, 2]:
        count = int(counts.get(class_id, 0))
        print(f"- class {class_id}: {count} ({count / total:.2%})")


def print_regime_distribution(classes):
    names = {
        0: "bearish regime",
        1: "neutral/chop regime",
        2: "bullish regime",
    }
    counts = pd.Series(classes).dropna().astype(int).value_counts().sort_index()
    total = int(counts.sum()) or 1
    for class_id in [0, 1, 2]:
        count = int(counts.get(class_id, 0))
        print(f"- class {class_id} {names[class_id]}: {count} ({count / total:.2%})")


def print_future_return_class_threshold_summary(frame):
    print("Future return class threshold summary:")
    print(f"- mode: {FUTURE_RETURN_CLASS_MODE}")
    print(f"- VOL_LOOKBACK_MINUTES: {VOL_LOOKBACK_MINUTES}")
    print(f"- MIN_VOL_LOOKBACK_ROWS: {MIN_VOL_LOOKBACK_ROWS}")
    print(f"- VOL_MULTIPLIER: {VOL_MULTIPLIER}")
    print(f"- MIN_CLASS_MOVE: {MIN_CLASS_MOVE:.6g}")
    print(f"- fixed RETURN_CLASS_THRESHOLD: {RETURN_CLASS_THRESHOLD:.6g}")

    if "future_return_class_threshold" not in frame.columns:
        print("- no threshold column available")
        return

    thresholds = pd.to_numeric(
        frame["future_return_class_threshold"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(thresholds) == 0:
        print("- no computed thresholds yet")
        return

    print(f"- min threshold: {thresholds.min():.6g}")
    print(f"- p10 threshold: {thresholds.quantile(0.10):.6g}")
    print(f"- mean threshold: {thresholds.mean():.6g}")
    print(f"- median threshold: {thresholds.median():.6g}")
    print(f"- p90 threshold: {thresholds.quantile(0.90):.6g}")
    print(f"- max threshold: {thresholds.max():.6g}")
    print("- class counts:")
    print_class_distribution(
        frame["actual_future_return_class"]
        if "actual_future_return_class" in frame.columns
        else []
    )


def main():
    predictions = load_predictions()
    if len(predictions) == 0:
        print("No live predictions found to label.")
        print("No trades were placed.")
        return

    raw_rows = load_raw_rows()
    feature_rows = load_feature_rows(raw_rows)
    raw_by_timestamp = raw_rows.set_index("timestamp", drop=False)
    features_by_timestamp = feature_rows.set_index("timestamp", drop=False)

    backed_up = backup_existing_training_csv()
    repaired_corrupted_prediction_rows = int(
        (predictions["label_ready"] & ~complete_actual_target_mask(predictions)).sum()
    )
    if repaired_corrupted_prediction_rows:
        predictions.loc[~complete_actual_target_mask(predictions), "label_ready"] = False

    newly_labeled = 0
    newly_core_labeled = 0
    rows_fully_labeled_from_raw_derived_targets = 0
    newly_regime_labeled_15m = 0
    newly_regime_labeled_30m = 0
    skipped_incomplete_future_window = 0
    skipped_missing_actual_target_values = 0
    blocked_only_by_missing_derived_future_features = 0
    rows_without_future_return_class_threshold = 0
    rows_reclassified_by_future_return_mode = 0
    skip_reasons = {}
    for index, row in predictions.iterrows():
        if bool(row["label_ready"]):
            continue
        timestamp = int(row["timestamp"])
        feature_window_complete = has_future_feature_window(timestamp, features_by_timestamp)
        labels, core_complete, full_complete, reason = calculate_labels(
            timestamp,
            raw_rows,
            raw_by_timestamp,
            features_by_timestamp,
        )
        if labels is None:
            reason = reason or "unknown missing actual target values"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            if reason == "incomplete future window":
                skipped_incomplete_future_window += 1
            else:
                skipped_missing_actual_target_values += 1
            continue
        for column, value in labels.items():
            predictions.loc[index, column] = value

        core_ready = complete_actual_target_mask(
            predictions.loc[[index]],
            CORE_ACTUAL_TARGET_COLUMNS,
        ).iloc[0]
        full_ready = complete_actual_target_mask(predictions.loc[[index]]).iloc[0]
        predictions.loc[index, "core_label_ready"] = bool(core_complete and core_ready)
        predictions.loc[index, "full_regression_label_ready"] = bool(
            full_complete and full_ready
        )

        if core_complete and core_ready:
            newly_core_labeled += 1

        if full_ready:
            predictions.loc[index, "label_ready"] = True
            predictions.loc[index, "full_regression_label_ready"] = True
            newly_labeled += 1
            if reason == "raw-computed" or not feature_window_complete:
                rows_fully_labeled_from_raw_derived_targets += 1
        else:
            predictions.loc[index, "label_ready"] = False
            if core_ready:
                blocked_only_by_missing_derived_future_features += 1
                reason = reason or "missing derived future features"
            else:
                skipped_missing_actual_target_values += 1
                reason = reason or "computed labels did not pass completeness check"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    if "actual_future_return_3" in predictions.columns:
        known_return = pd.to_numeric(
            predictions["actual_future_return_3"],
            errors="coerce",
        ).notna()
        for index, row in predictions[known_return].iterrows():
            class_metadata, class_reason = future_return_class_metadata(
                int(row["timestamp"]),
                float(row["actual_future_return_3"]),
                raw_rows,
            )
            if class_metadata is None:
                rows_without_future_return_class_threshold += 1
                predictions.loc[index, "actual_future_return_class"] = np.nan
                predictions.loc[index, "future_return_class_threshold"] = np.nan
                predictions.loc[index, "future_return_class_vol_3m"] = np.nan
                predictions.loc[index, "future_return_class_mode"] = str(FUTURE_RETURN_CLASS_MODE)
                predictions.loc[index, "future_return_class_vol_multiplier"] = VOL_MULTIPLIER
                reason = class_reason or "missing future return class threshold"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            for column, value in class_metadata.items():
                predictions.loc[index, column] = value
            rows_reclassified_by_future_return_mode += 1

    if "actual_path_class" in predictions.columns:
        missing_path_event = predictions["actual_path_event_class"].isna()
        known_path = pd.to_numeric(predictions["actual_path_class"], errors="coerce").notna()
        predictions.loc[missing_path_event & known_path, "actual_path_event_class"] = predictions.loc[
            missing_path_event & known_path,
            "actual_path_class",
        ]

    for index, row in predictions.iterrows():
        timestamp = int(row["timestamp"])
        already_had_15m = pd.notna(row.get("actual_regime_class_15m", np.nan))
        already_had_30m = pd.notna(row.get("actual_regime_class_30m", np.nan))
        regime_labels = calculate_regime_labels(timestamp, raw_by_timestamp)
        if not regime_labels:
            continue
        for column, value in regime_labels.items():
            predictions.loc[index, column] = value
        if "actual_regime_class_15m" in regime_labels and not already_had_15m:
            newly_regime_labeled_15m += 1
        if "actual_regime_class_30m" in regime_labels and not already_had_30m:
            newly_regime_labeled_30m += 1

    predictions = predictions.sort_values("timestamp").drop_duplicates(
        "timestamp",
        keep="last",
    )
    predictions = normalize_debug_column_dtypes(predictions)
    raw_window_complete_mask = predictions["timestamp"].apply(
        lambda value: has_raw_future_window(int(value), raw_by_timestamp)
        if pd.notna(value)
        else False
    )
    feature_window_complete_mask = predictions["timestamp"].apply(
        lambda value: has_future_feature_window(int(value), features_by_timestamp)
        if pd.notna(value)
        else False
    )
    regime_15m_raw_window_complete_mask = predictions["timestamp"].apply(
        lambda value: has_raw_future_window_for_horizon(int(value), raw_by_timestamp, 15)
        if pd.notna(value)
        else False
    )
    regime_30m_raw_window_complete_mask = predictions["timestamp"].apply(
        lambda value: has_raw_future_window_for_horizon(int(value), raw_by_timestamp, 30)
        if pd.notna(value)
        else False
    )
    core_mask = complete_actual_target_mask(predictions, CORE_ACTUAL_TARGET_COLUMNS)
    complete_mask = complete_actual_target_mask(predictions)
    predictions["core_label_ready"] = core_mask
    predictions["full_regression_label_ready"] = complete_mask
    predictions["label_ready"] = complete_mask
    atomic_write_csv(predictions, PREDICTIONS_PATH)

    labeled = predictions[predictions["label_ready"].apply(parse_bool) & complete_mask].copy()
    labeled = normalize_debug_column_dtypes(labeled)
    LABELED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(labeled, LABELED_OUTPUT_PATH)

    latest_raw_timestamp = int(raw_rows["timestamp"].max()) if len(raw_rows) else None
    latest_prediction_timestamp = (
        int(predictions["timestamp"].max()) if len(predictions) else None
    )
    print("Live 3m prediction labeler")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Raw 1m path: {RAW_1M_PATH}")
    print(f"Feature path: {FEATURES_PATH}")
    print(f"Latest raw 1m timestamp: {latest_raw_timestamp}")
    print(f"Latest prediction timestamp: {latest_prediction_timestamp}")
    print(f"FUTURE_RETURN_CLASS_MODE: {FUTURE_RETURN_CLASS_MODE}")
    print(f"VOL_LOOKBACK_MINUTES: {VOL_LOOKBACK_MINUTES}")
    print(f"MIN_VOL_LOOKBACK_ROWS: {MIN_VOL_LOOKBACK_ROWS}")
    print(f"VOL_MULTIPLIER: {VOL_MULTIPLIER}")
    print(f"MIN_CLASS_MOVE: {MIN_CLASS_MOVE:.6g}")
    print(f"RETURN_CLASS_THRESHOLD: {RETURN_CLASS_THRESHOLD:.6g}")
    print(f"Total prediction rows: {len(predictions)}")
    print(f"Rows with label_ready=true: {int(predictions['label_ready'].apply(parse_bool).sum())}")
    print(f"Rows with core_label_ready=true: {int(core_mask.sum())}")
    print(f"Rows with complete actual targets: {int(complete_mask.sum())}")
    print(f"Rows with complete raw future windows: {int(raw_window_complete_mask.sum())}")
    print(f"Rows with missing raw future windows: {int((~raw_window_complete_mask).sum())}")
    print(
        "Rows with complete 15m regime future windows: "
        f"{int(regime_15m_raw_window_complete_mask.sum())}"
    )
    print(
        "Rows with complete 30m regime future windows: "
        f"{int(regime_30m_raw_window_complete_mask.sum())}"
    )
    print(f"Rows with complete future feature windows: {int(feature_window_complete_mask.sum())}")
    print(
        "Rows blocked only by missing derived future features: "
        f"{blocked_only_by_missing_derived_future_features}"
    )
    print(
        "Rows labeled from raw-only targets: "
        f"{rows_fully_labeled_from_raw_derived_targets}"
    )
    print(f"Rows fully labeled for trainer: {int(complete_mask.sum())}")
    print(f"Newly core-labeled rows: {newly_core_labeled}")
    print(f"Newly labeled rows: {newly_labeled}")
    print(f"Newly labeled 15m regime rows: {newly_regime_labeled_15m}")
    print(f"Newly labeled 30m regime rows: {newly_regime_labeled_30m}")
    print(
        "Rows reclassified by future return class mode: "
        f"{rows_reclassified_by_future_return_mode}"
    )
    print(
        "Rows without available future return class threshold: "
        f"{rows_without_future_return_class_threshold}"
    )
    print(f"Rows repaired from corrupted label_ready state: {repaired_corrupted_prediction_rows}")
    print(f"Rows skipped because incomplete future window: {skipped_incomplete_future_window}")
    print(f"Rows skipped because missing actual target values: {skipped_missing_actual_target_values}")
    if skip_reasons:
        print("Label skip reasons:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"- {reason}: {count}")
    print(f"Existing training CSV backup written: {LABELED_BACKUP_PATH if backed_up else 'not needed'}")
    print(f"Total labeled rows: {len(labeled)}")
    raw_classes = (
        labeled["actual_path_class"].dropna().astype(str).unique().tolist()
        if len(labeled)
        else []
    )
    print(f"Raw unique actual_path_class values: {raw_classes}")
    print("3m future_return_class distribution:")
    print_class_distribution(predictions["actual_future_return_class"] if "actual_future_return_class" in predictions.columns else [])
    print_future_return_class_threshold_summary(predictions)
    print("3m path_event_class distribution:")
    print_class_distribution(predictions["actual_path_event_class"] if "actual_path_event_class" in predictions.columns else [])
    print("3m legacy actual_path_class distribution:")
    print_class_distribution(labeled["actual_path_class"] if len(labeled) else [])
    print("15m regime class distribution:")
    print_regime_distribution(predictions["actual_regime_class_15m"])
    print("30m regime class distribution:")
    print_regime_distribution(predictions["actual_regime_class_30m"])
    print(f"Labeled training snapshot: {LABELED_OUTPUT_PATH}")
    print("No trades were placed.")


if __name__ == "__main__":
    main()
