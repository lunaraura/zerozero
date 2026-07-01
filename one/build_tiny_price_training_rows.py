import os
import json
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from tiny_price_feature_utils import (
    feature_schema_hash,
    select_model_feature_columns,
    select_target_columns,
    slugify,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
FEATURE_SET = os.getenv("PRICE_TINY_FEATURE_SET", "tiny_price_v1").strip().lower()
PRICE_TINY_FEATURE_GROUPS = [
    value.strip()
    for value in os.getenv("PRICE_TINY_FEATURE_GROUPS", "base_tiny_price_v1").split(",")
    if value.strip()
]
SUPPORTED_FEATURE_GROUPS = {
    "base_tiny_price_v1",
    "cross_venue_features",
    "pressure_change_features",
    "spread_volatility_features",
    "depth_acceleration_features",
    "snapshot_freshness_features",
    "regime_context_features",
    "calendar_session_features",
}
UNKNOWN_FEATURE_GROUPS = sorted(set(PRICE_TINY_FEATURE_GROUPS) - SUPPORTED_FEATURE_GROUPS)
if UNKNOWN_FEATURE_GROUPS:
    raise ValueError(
        "Unknown PRICE_TINY_FEATURE_GROUPS: "
        f"{UNKNOWN_FEATURE_GROUPS}. Supported groups: {sorted(SUPPORTED_FEATURE_GROUPS)}"
    )
TARGET_SPEC_NAME = os.getenv("PRICE_TINY_TARGET_SPEC", "").strip().lower()
HORIZON_SECONDS = int(os.getenv("PRICE_TINY_HORIZON_SECONDS", "1"))
LOOKBACK_PROFILE = os.getenv(
    "PRICE_TINY_LOOKBACK_PROFILE",
    "long" if (os.getenv("PRICE_TINY_FEATURE_GROUPS") or os.getenv("PRICE_TINY_TARGET_SPEC")) else "short",
).strip().lower()
PRICE_TINY_FLAT_BPS = float(os.getenv("PRICE_TINY_FLAT_BPS", "0.10"))
PRICE_TINY_TARGET_MOVE_BPS = float(os.getenv("PRICE_TINY_TARGET_MOVE_BPS", str(PRICE_TINY_FLAT_BPS)))
MAX_FUTURE_GAP_MS = int(os.getenv("PRICE_TINY_MAX_FUTURE_GAP_MS", str(max(1500, HORIZON_SECONDS * 1500))))
MAX_CONTEXT_AGE_MS = int(os.getenv("PRICE_TINY_MAX_CONTEXT_AGE_MS", "5000"))
PRICE_TINY_REQUIRE_CROSSVENUE_CONTEXT = os.getenv("PRICE_TINY_REQUIRE_CROSSVENUE_CONTEXT", "false").strip().lower() in {"1", "true", "yes"}
PRICE_TINY_CROSSVENUE_MAX_AGE_MS = int(os.getenv("PRICE_TINY_CROSSVENUE_MAX_AGE_MS", "3000"))
PRICE_TINY_CROSSVENUE_JOIN_POLICY = os.getenv("PRICE_TINY_CROSSVENUE_JOIN_POLICY", "backward").strip().lower()
if PRICE_TINY_CROSSVENUE_JOIN_POLICY not in {"backward", "nearest"}:
    print(f"Unknown PRICE_TINY_CROSSVENUE_JOIN_POLICY={PRICE_TINY_CROSSVENUE_JOIN_POLICY}; using backward.")
    PRICE_TINY_CROSSVENUE_JOIN_POLICY = "backward"
CROSSVENUE_MISSING_POLICY = "zero_with_context_available_mask"
REGIME_CONTEXT_WINDOWS = [60, 300, 900]
BTC_CONTEXT_SYMBOL = os.getenv("PRICE_TINY_BTC_CONTEXT_SYMBOL", "BTCUSDT").strip().upper()
BTC_CONTEXT_MAX_AGE_MS = int(os.getenv("PRICE_TINY_BTC_CONTEXT_MAX_AGE_MS", "5000"))
BTC_CONTEXT_JOIN_POLICY = os.getenv("PRICE_TINY_BTC_CONTEXT_JOIN_POLICY", "backward").strip().lower()
if BTC_CONTEXT_JOIN_POLICY not in {"backward", "nearest"}:
    print(f"Unknown PRICE_TINY_BTC_CONTEXT_JOIN_POLICY={BTC_CONTEXT_JOIN_POLICY}; using backward.")
    BTC_CONTEXT_JOIN_POLICY = "backward"
MISSING_FEATURE_POLICY = os.getenv("PRICE_TINY_MISSING_FEATURE_POLICY", "fill_zero").strip().lower()
MACRO_CALENDAR_PATH = Path(os.getenv("PRICE_TINY_MACRO_CALENDAR_PATH", PROJECT_ROOT / "data" / "calendar" / "macro_events.csv"))
MACRO_EVENT_WINDOW_MINUTES = float(os.getenv("PRICE_TINY_MACRO_EVENT_WINDOW_MINUTES", "30"))
MACRO_EVENT_DEFAULT_MINUTES = float(os.getenv("PRICE_TINY_MACRO_EVENT_DEFAULT_MINUTES", "1000000"))
PRICE_TINY_INSTABILITY_RANGE_MULT = float(os.getenv("PRICE_TINY_INSTABILITY_RANGE_MULT", "2.0"))
PRICE_TINY_INSTABILITY_RETURN_BPS = float(os.getenv("PRICE_TINY_INSTABILITY_RETURN_BPS", "8"))
PRICE_TINY_INSTABILITY_ADVERSE_BPS = float(os.getenv("PRICE_TINY_INSTABILITY_ADVERSE_BPS", "5"))
PRICE_TINY_INSTABILITY_SPREAD_MULT = float(os.getenv("PRICE_TINY_INSTABILITY_SPREAD_MULT", "2.0"))
PRICE_TINY_BUILD_MAX_ROWS = int(os.getenv("PRICE_TINY_BUILD_MAX_ROWS", "0") or "0")
PRICE_TINY_TARGET_MIN_NET_BPS = float(os.getenv("PRICE_TINY_TARGET_MIN_NET_BPS", "1.0"))
PRICE_TINY_TARGET_SPREAD_MULTIPLIER = float(os.getenv("PRICE_TINY_TARGET_SPREAD_MULTIPLIER", "1.0"))
PRICE_TINY_TARGET_FEE_BPS = float(os.getenv("PRICE_TINY_TARGET_FEE_BPS", "0.0"))
PRICE_TINY_TARGET_SLIPPAGE_BPS = float(os.getenv("PRICE_TINY_TARGET_SLIPPAGE_BPS", "0.0"))
PRICE_TINY_TARGET_HORIZONS_SECONDS = [
    int(value.strip())
    for value in os.getenv("PRICE_TINY_TARGET_HORIZONS_SECONDS", "30,45,60").split(",")
    if value.strip()
]

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
if not MACRO_CALENDAR_PATH.is_absolute():
    MACRO_CALENDAR_PATH = PROJECT_ROOT / MACRO_CALENDAR_PATH

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
INPUT_PATH = VENUE_DIR / f"{SYMBOL}_10s_flow.csv"
FLOW_1S_PATH = VENUE_DIR / f"{SYMBOL}_1s_order_flow_predictions.csv"
MICRO_10S_PATH = VENUE_DIR / f"{SYMBOL}_10s_microstructure_predictions.csv"
CROSS_VENUE_PATH = OUTPUT_DIR / f"{SYMBOL}_cross_venue_features.csv"
BTC_CONTEXT_PATH = VENUE_DIR / f"{BTC_CONTEXT_SYMBOL}_10s_flow.csv"
LATEST_OUTPUT_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_training_rows.csv"
LATEST_METADATA_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_training_rows_latest.json"
BTC_CONTEXT_CACHE = None


def parse_target_spec(name, fallback_horizon):
    if not name:
        name = f"direction_{fallback_horizon}s"
    horizon = fallback_horizon
    for part in reversed(name.split("_")):
        if part.endswith("s") and part[:-1].isdigit():
            horizon = int(part[:-1])
            break
    if name.startswith("move_before_adverse") and "net_aware" in name:
        method = "move_before_adverse_net_aware"
    elif name.startswith("move_before_adverse"):
        method = "move_before_adverse"
    elif name.startswith("first_touch"):
        method = "first_touch"
    elif name.startswith("next_mid_delta_bps"):
        method = "next_mid_delta_bps"
    elif name.startswith("next_mid_log_return"):
        method = "next_mid_log_return"
    elif name.startswith("return"):
        method = "return_bps"
    elif name.startswith("direction"):
        method = "direction"
    elif name.startswith("instability"):
        method = "instability"
    elif name.startswith("chop") or "no_trade" in name:
        method = "chop_no_trade"
    else:
        method = "direction"
    return {
        "name": name,
        "horizon_seconds": horizon,
        "label_construction_method": method,
        "no_lookahead_validation": "features use rows with timestamp <= prediction timestamp; targets use only future rows after prediction timestamp",
    }


TARGET_SPEC = parse_target_spec(TARGET_SPEC_NAME, HORIZON_SECONDS)
HORIZON_SECONDS = int(TARGET_SPEC["horizon_seconds"])
MAX_FUTURE_GAP_MS = int(os.getenv("PRICE_TINY_MAX_FUTURE_GAP_MS", str(max(1500, HORIZON_SECONDS * 1500))))


LOOKBACK_PROFILES = {
    # "short" intentionally preserves the original tiny-price feature windows.
    "short": {
        "rolling_mean_seconds": 60,
        "return_seconds": [1, 3, 5, 10],
        "volatility_seconds": [10, 30],
        "range_seconds": [10, 30],
    },
    "medium": {
        "rolling_mean_seconds": 120,
        "return_seconds": [1, 3, 5, 10, 30],
        "volatility_seconds": [10, 30, 60],
        "range_seconds": [10, 30, 60],
    },
    "long": {
        "rolling_mean_seconds": 300,
        "return_seconds": [1, 3, 5, 10, 30, 60],
        "volatility_seconds": [30, 60, 120],
        "range_seconds": [30, 60, 120],
    },
}


def get_lookback_profile(profile=None):
    name = (profile or LOOKBACK_PROFILE or "short").strip().lower()
    if name not in LOOKBACK_PROFILES:
        print(f"Unknown PRICE_TINY_LOOKBACK_PROFILE={name}; using short.")
        name = "short"
    config = dict(LOOKBACK_PROFILES[name])
    config["name"] = name
    return config


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        # Simulated flow files include string hidden diagnostics such as
        # hidden_active_event_type. low_memory=False avoids chunk-level mixed
        # dtype inference warnings without changing model feature selection.
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


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


def experiment_training_rows_path(schema_hash):
    feature_slug = slugify("+".join(PRICE_TINY_FEATURE_GROUPS), "base_tiny_price_v1")
    target_slug = slugify(TARGET_SPEC["name"], f"direction_{HORIZON_SECONDS}s")
    return VENUE_DIR / (
        f"{SYMBOL}_tiny_price_training_rows__"
        f"{feature_slug}__{target_slug}__{HORIZON_SECONDS}s__{schema_hash}.csv"
    )


def numeric(frame, column, default=np.nan):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def normalize_input(frame):
    frame = frame.copy()
    protected_string_columns = {
        "time",
        "simulation_run_id",
        "source_scenario",
        "source_seed",
    }
    for column in frame.columns:
        if column not in protected_string_columns and not str(column).startswith("hidden_"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in protected_string_columns:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    for column in [column for column in frame.columns if str(column).startswith("hidden_")]:
        if frame[column].dtype == "object":
            frame[column] = frame[column].fillna("").astype(str)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    required = ["timestamp", "mid_price", "best_bid", "best_ask", "bid_depth_10bps", "ask_depth_10bps"]
    for column in required:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame.dropna(subset=required)
    frame = frame[
        (frame["mid_price"] > 0)
        & (frame["best_bid"] > 0)
        & (frame["best_ask"] > 0)
        & (frame["bid_depth_10bps"] > 0)
        & (frame["ask_depth_10bps"] > 0)
    ].copy()
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return frame


def past_index(timestamps, index, seconds, lower_bound=0):
    target = timestamps[index] - seconds * 1000
    return max(int(lower_bound), int(np.searchsorted(timestamps, target, side="left")))


def future_index(timestamps, index, seconds, upper_bound=None):
    upper_bound = len(timestamps) if upper_bound is None else int(upper_bound)
    target = timestamps[index] + seconds * 1000
    found = int(np.searchsorted(timestamps, target, side="left"))
    if found >= upper_bound:
        return None
    if timestamps[found] - target > MAX_FUTURE_GAP_MS:
        return None
    return found


def run_bounds(frame):
    if "simulation_run_id" not in frame.columns or len(frame) == 0:
        return np.zeros(len(frame), dtype=np.int64), np.full(len(frame), len(frame), dtype=np.int64), False
    labels = frame["simulation_run_id"].fillna("").astype(str).to_numpy()
    if not any(label for label in labels):
        return np.zeros(len(frame), dtype=np.int64), np.full(len(frame), len(frame), dtype=np.int64), False
    starts = np.zeros(len(frame), dtype=np.int64)
    ends = np.full(len(frame), len(frame), dtype=np.int64)
    start = 0
    for position in range(1, len(frame) + 1):
        if position == len(frame) or labels[position] != labels[start]:
            starts[start:position] = start
            ends[start:position] = position
            start = position
    return starts, ends, True


def is_return_like_crossvenue_column(column):
    name = str(column).lower()
    return "return" in name or name.endswith("_diff_bps") or "imbalance_diff" in name


def add_asof_context(
    rows,
    context_path,
    prefix,
    columns,
    max_age_ms=None,
    join_policy="backward",
    unavailable_age_ms=-1.0,
    add_age_seconds=False,
    min_context_column=None,
    min_context_value=None,
    clean_return_sentinels=False,
):
    max_age_ms = MAX_CONTEXT_AGE_MS if max_age_ms is None else int(max_age_ms)
    join_policy = join_policy if join_policy in {"backward", "nearest"} else "backward"
    context = context_path.copy() if isinstance(context_path, pd.DataFrame) else read_csv(context_path)
    output_columns = [
        f"feature_{prefix}_context_available",
        f"feature_{prefix}_context_age_ms",
        *[f"feature_{prefix}_{c}" for c in columns],
    ]
    if add_age_seconds:
        output_columns.append(f"feature_{prefix}_context_age_seconds")
    for column in output_columns:
        if column not in rows.columns:
            rows[column] = 0.0
    if len(context) == 0 or "timestamp" not in context.columns:
        rows[f"feature_{prefix}_context_age_ms"] = unavailable_age_ms
        if add_age_seconds:
            rows[f"feature_{prefix}_context_age_seconds"] = unavailable_age_ms / 1000.0 if unavailable_age_ms > 0 else 0.0
        return rows
    context = context.copy()
    context["timestamp"] = pd.to_numeric(context["timestamp"], errors="coerce")
    context = context.dropna(subset=["timestamp"]).sort_values("timestamp")
    if len(context) == 0:
        rows[f"feature_{prefix}_context_age_ms"] = unavailable_age_ms
        if add_age_seconds:
            rows[f"feature_{prefix}_context_age_seconds"] = unavailable_age_ms / 1000.0 if unavailable_age_ms > 0 else 0.0
        return rows
    left = rows[["timestamp"]].copy()
    left["_row_order"] = np.arange(len(left))
    context_with_key = context.rename(columns={"timestamp": "_context_timestamp"})
    keep_context_columns = [c for c in columns if c in context_with_key.columns]
    if min_context_column and min_context_column in context_with_key.columns and min_context_column not in keep_context_columns:
        keep_context_columns.append(min_context_column)
    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        context_with_key[["_context_timestamp", *keep_context_columns]].sort_values("_context_timestamp"),
        left_on="timestamp",
        right_on="_context_timestamp",
        direction=join_policy,
        tolerance=max_age_ms,
    )
    merged = merged.sort_values("_row_order").reset_index(drop=True)
    matched_timestamp = pd.to_numeric(merged["_context_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    row_timestamp = rows["timestamp"].to_numpy(dtype=np.float64)
    raw_age = row_timestamp - matched_timestamp
    ages = np.abs(raw_age) if join_policy == "nearest" else raw_age
    available = np.isfinite(ages) & (ages >= 0) & (ages <= max_age_ms)
    if min_context_column and min_context_column in merged.columns:
        context_min_values = pd.to_numeric(merged[min_context_column], errors="coerce").to_numpy(dtype=np.float64)
        available &= np.isfinite(context_min_values) & (context_min_values >= float(min_context_value))
    rows[f"feature_{prefix}_context_available"] = available.astype(float)
    rows[f"feature_{prefix}_context_age_ms"] = np.where(available, ages, unavailable_age_ms)
    if add_age_seconds:
        rows[f"feature_{prefix}_context_age_seconds"] = np.where(available, ages / 1000.0, unavailable_age_ms / 1000.0 if unavailable_age_ms > 0 else 0.0)
    for column in columns:
        output_column = f"feature_{prefix}_{column}"
        if column not in merged.columns:
            rows[output_column] = 0.0
            continue
        values = pd.to_numeric(merged[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if clean_return_sentinels and is_return_like_crossvenue_column(column):
            # A missing return encoded as -1.0 looks like a real -100% move to a
            # model. Crypto cross-venue return features should not be anywhere
            # near -100% at 1s/3s/10s horizons, so treat that as missing.
            values = values.mask(values <= -0.999, 0.0)
        rows[output_column] = values.where(available, 0.0).fillna(0.0)
    return rows


def finite_float(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def path_returns_bps(mid, index, future_index_value, current_mid):
    if future_index_value <= index:
        return np.asarray([], dtype=np.float64)
    future_mid = mid[index + 1 : future_index_value + 1]
    future_mid = future_mid[np.isfinite(future_mid) & (future_mid > 0)]
    if len(future_mid) == 0 or current_mid <= 0:
        return np.asarray([], dtype=np.float64)
    return (future_mid / current_mid - 1.0) * 10000.0


def first_touch_direction(path_bps, threshold_bps):
    if len(path_bps) == 0:
        return 0
    up_hits = np.where(path_bps >= threshold_bps)[0]
    down_hits = np.where(path_bps <= -threshold_bps)[0]
    first_up = int(up_hits[0]) if len(up_hits) else None
    first_down = int(down_hits[0]) if len(down_hits) else None
    if first_up is None and first_down is None:
        return 0
    if first_down is None or (first_up is not None and first_up < first_down):
        return 1
    if first_up is None or first_down < first_up:
        return -1
    return 0


def estimate_round_trip_spread_cost_bps(source_row):
    spread_ratio = finite_float(source_row.get("spread_percent", 0.0))
    if spread_ratio < 0:
        spread_ratio = 0.0
    # Recorder stores spread_percent as a ratio; one round trip is roughly
    # one full spread when half-spread is charged on entry and exit.
    return float(spread_ratio * 10000.0 * PRICE_TINY_TARGET_SPREAD_MULTIPLIER)


def net_aware_threshold_bps(source_row):
    spread_cost = estimate_round_trip_spread_cost_bps(source_row)
    threshold = (
        spread_cost
        + PRICE_TINY_TARGET_FEE_BPS
        + PRICE_TINY_TARGET_SLIPPAGE_BPS
        + PRICE_TINY_TARGET_MIN_NET_BPS
    )
    return float(max(0.0, threshold)), float(spread_cost)


def add_net_aware_targets(row, frame, timestamps, mid, index, group_end, horizons):
    source_row = frame.iloc[index]
    current_mid = finite_float(source_row.get("mid_price", np.nan), np.nan)
    if not np.isfinite(current_mid) or current_mid <= 0:
        return
    threshold_bps, spread_cost_bps = net_aware_threshold_bps(source_row)
    row["target_net_aware_min_net_bps"] = PRICE_TINY_TARGET_MIN_NET_BPS
    row["target_net_aware_spread_multiplier"] = PRICE_TINY_TARGET_SPREAD_MULTIPLIER
    row["target_net_aware_fee_bps"] = PRICE_TINY_TARGET_FEE_BPS
    row["target_net_aware_slippage_bps"] = PRICE_TINY_TARGET_SLIPPAGE_BPS
    row["target_net_aware_estimated_spread_cost_bps"] = spread_cost_bps
    row["target_net_aware_threshold_bps"] = threshold_bps
    for horizon in sorted(set(int(value) for value in horizons if int(value) > 0)):
        f_index = future_index(timestamps, index, horizon, group_end)
        if f_index is None:
            continue
        future_mid = mid[f_index]
        if not np.isfinite(future_mid) or future_mid <= 0:
            continue
        path_bps = path_returns_bps(mid, index, f_index, current_mid)
        if len(path_bps) == 0:
            continue
        max_runup = float(np.max(path_bps))
        max_drawdown = float(np.min(path_bps))
        label = first_touch_direction(path_bps, threshold_bps)
        row[f"target_move_before_adverse_net_aware_{horizon}s"] = label
        row[f"target_net_aware_threshold_bps_{horizon}s"] = threshold_bps
        row[f"target_net_aware_estimated_spread_cost_bps_{horizon}s"] = spread_cost_bps
        row[f"target_net_aware_max_favorable_excursion_bps_{horizon}s"] = max(0.0, max_runup)
        row[f"target_net_aware_max_adverse_excursion_bps_{horizon}s"] = min(0.0, max_drawdown)
        row[f"target_net_aware_realized_return_bps_{horizon}s"] = float((future_mid / current_mid - 1.0) * 10000.0)


def add_pressure_change_features(row, frame, timestamps, index, group_start=0):
    source = frame.iloc[index]
    buy = finite_float(source.get("market_buy_volume_10s", 0.0))
    sell = finite_float(source.get("market_sell_volume_10s", 0.0))
    total = buy + sell
    pressure = finite_float(source.get("market_pressure_10s", (buy - sell) / max(total, 1e-12)))
    row["feature_market_buy_volume_log1p"] = float(np.log1p(max(0.0, buy)))
    row["feature_market_sell_volume_log1p"] = float(np.log1p(max(0.0, sell)))
    row["feature_trade_count_log1p"] = float(np.log1p(max(0.0, finite_float(source.get("trade_count_10s", 0.0)))))
    row["feature_buy_sell_imbalance"] = float((buy - sell) / max(total, 1e-12))
    row["feature_aggressive_flow_pressure"] = pressure
    for seconds in [1, 3, 5, 10, 30]:
        p_index = past_index(timestamps, index, seconds, group_start)
        past_source = frame.iloc[p_index]
        past_buy = finite_float(past_source.get("market_buy_volume_10s", 0.0))
        past_sell = finite_float(past_source.get("market_sell_volume_10s", 0.0))
        past_total = past_buy + past_sell
        past_pressure = finite_float(past_source.get("market_pressure_10s", (past_buy - past_sell) / max(past_total, 1e-12)))
        row[f"feature_market_pressure_change_{seconds}s"] = pressure - past_pressure
        row[f"feature_market_buy_volume_change_log1p_{seconds}s"] = np.log1p(max(0.0, buy)) - np.log1p(max(0.0, past_buy))
        row[f"feature_market_sell_volume_change_log1p_{seconds}s"] = np.log1p(max(0.0, sell)) - np.log1p(max(0.0, past_sell))


def add_spread_volatility_features(row, frame, timestamps, index, group_start=0):
    spread = finite_float(frame.iloc[index].get("spread_percent", 0.0))
    for seconds in [5, 10, 30, 60]:
        p_index = past_index(timestamps, index, seconds, group_start)
        if "spread_percent" in frame.columns:
            values = pd.to_numeric(frame.iloc[p_index : index + 1]["spread_percent"], errors="coerce").to_numpy(dtype=np.float64)
        else:
            values = np.asarray([spread], dtype=np.float64)
        values = values[np.isfinite(values)]
        row[f"feature_spread_change_{seconds}s"] = spread - (float(values[0]) if len(values) else spread)
        row[f"feature_spread_mean_{seconds}s"] = float(np.mean(values)) if len(values) else spread
        row[f"feature_spread_volatility_{seconds}s"] = float(np.std(values)) if len(values) else 0.0


def add_depth_acceleration_features(row, frame, timestamps, index, group_start=0):
    source = frame.iloc[index]
    for column in ["bid_depth_10bps", "ask_depth_10bps", "bid_depth_25bps", "ask_depth_25bps", "order_book_imbalance_10bps", "order_book_imbalance_25bps"]:
        current = finite_float(source.get(column, 0.0))
        for seconds in [1, 3, 5, 10]:
            p_index = past_index(timestamps, index, seconds, group_start)
            past = finite_float(frame.iloc[p_index].get(column, current))
            change = current - past
            row[f"feature_{column}_change_{seconds}s"] = change
            row[f"feature_{column}_relative_change_{seconds}s"] = change / max(abs(past), 1e-9)
        p1 = past_index(timestamps, index, 1, group_start)
        p3 = past_index(timestamps, index, 3, group_start)
        v1 = current - finite_float(frame.iloc[p1].get(column, current))
        v3 = finite_float(frame.iloc[p1].get(column, current)) - finite_float(frame.iloc[p3].get(column, current))
        row[f"feature_{column}_acceleration_3s"] = v1 - v3


def add_snapshot_freshness_features(row, timestamps, index, group_start=0):
    previous_timestamp = int(timestamps[index - 1]) if index > group_start else int(timestamps[index])
    gap_ms = int(timestamps[index]) - previous_timestamp
    row["feature_snapshot_gap_ms"] = float(gap_ms)
    row["feature_snapshot_gap_seconds"] = float(gap_ms / 1000.0)
    row["feature_snapshot_gap_is_large"] = 1.0 if gap_ms > 1500 else 0.0


def rolling_mid_window(mid, timestamps, index, seconds, group_start=0):
    p_index = past_index(timestamps, index, seconds, group_start)
    values = mid[p_index : index + 1]
    return values[np.isfinite(values) & (values > 0)]


def log_return_series(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 2:
        return np.asarray([0.0], dtype=np.float64)
    return np.diff(np.log(values))


def trend_slope_per_second(values, seconds):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 2:
        return 0.0
    y = np.log(values)
    x = np.linspace(-float(seconds), 0.0, len(y))
    x = x - x.mean()
    denominator = float(np.sum(x * x))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(x * (y - y.mean())) / denominator)


def add_regime_context_features(row, frame, timestamps, index, prefix="regime", group_start=0):
    mid = frame["mid_price"].to_numpy(dtype=np.float64)
    current_mid = float(mid[index])
    for seconds in REGIME_CONTEXT_WINDOWS:
        values = rolling_mid_window(mid, timestamps, index, seconds, group_start)
        returns = log_return_series(values)
        first_mid = float(values[0]) if len(values) else current_mid
        last_mid = float(values[-1]) if len(values) else current_mid
        window_return = float(np.log(last_mid / first_mid)) if first_mid > 0 and last_mid > 0 else 0.0
        up_fraction = float((returns > 0).mean()) if len(returns) else 0.0
        down_fraction = float((returns < 0).mean()) if len(returns) else 0.0
        signed_persistence = float(np.sign(returns).mean()) if len(returns) else 0.0
        row[f"feature_{prefix}_return_{seconds}s"] = window_return
        row[f"feature_{prefix}_trend_slope_{seconds}s"] = window_return / max(float(seconds), 1.0)
        row[f"feature_{prefix}_volatility_{seconds}s"] = float(np.std(returns)) if len(returns) else 0.0
        row[f"feature_{prefix}_range_{seconds}s"] = float(np.nanmax(values) / max(np.nanmin(values), 1e-12) - 1.0) if len(values) else 0.0
        row[f"feature_{prefix}_directional_persistence_{seconds}s"] = signed_persistence
        row[f"feature_{prefix}_directional_consistency_{seconds}s"] = abs(signed_persistence)
        row[f"feature_{prefix}_up_fraction_{seconds}s"] = up_fraction
        row[f"feature_{prefix}_down_fraction_{seconds}s"] = down_fraction


def compute_regime_context_feature_frame(frame):
    if (
        "simulation_run_id" in frame.columns
        and len(frame)
        and frame["simulation_run_id"].fillna("").astype(str).ne("").any()
    ):
        pieces = []
        for _, group in frame.groupby(frame["simulation_run_id"].fillna("").astype(str), sort=False):
            piece = compute_regime_context_feature_frame(group.drop(columns=["simulation_run_id"], errors="ignore"))
            if len(piece):
                pieces.append(piece)
        if not pieces:
            return pd.DataFrame()
        return pd.concat(pieces, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if len(frame) == 0:
        return pd.DataFrame()
    source = frame[["timestamp", "mid_price"]].copy()
    source["timestamp"] = pd.to_numeric(source["timestamp"], errors="coerce")
    source["mid_price"] = pd.to_numeric(source["mid_price"], errors="coerce")
    source = source.dropna(subset=["timestamp", "mid_price"])
    source = source[source["mid_price"] > 0].sort_values("timestamp").drop_duplicates("timestamp")
    if len(source) == 0:
        return pd.DataFrame()
    timestamps = source["timestamp"].to_numpy(dtype=np.int64)
    mid = source["mid_price"].to_numpy(dtype=np.float64)
    log_mid = np.log(mid)
    time_index = pd.to_datetime(timestamps, unit="ms", utc=True)
    mid_series = pd.Series(mid, index=time_index)
    returns = pd.Series(log_mid, index=time_index).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sign_returns = np.sign(returns)
    positive_returns = (returns > 0).astype(float)
    negative_returns = (returns < 0).astype(float)
    output = pd.DataFrame({"timestamp": timestamps})
    for seconds in REGIME_CONTEXT_WINDOWS:
        past_indices = np.searchsorted(timestamps, timestamps - seconds * 1000, side="left")
        window_return = log_mid - log_mid[past_indices]
        window = f"{seconds}s"
        rolling_max = mid_series.rolling(window, min_periods=1).max().to_numpy(dtype=np.float64)
        rolling_min = mid_series.rolling(window, min_periods=1).min().to_numpy(dtype=np.float64)
        volatility = returns.rolling(window, min_periods=2).std().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
        signed_persistence = sign_returns.rolling(window, min_periods=1).mean().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
        up_fraction = positive_returns.rolling(window, min_periods=1).mean().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
        down_fraction = negative_returns.rolling(window, min_periods=1).mean().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
        output[f"feature_regime_return_{seconds}s"] = window_return
        output[f"feature_regime_trend_slope_{seconds}s"] = window_return / max(float(seconds), 1.0)
        output[f"feature_regime_volatility_{seconds}s"] = volatility
        output[f"feature_regime_range_{seconds}s"] = rolling_max / np.maximum(rolling_min, 1e-12) - 1.0
        output[f"feature_regime_directional_persistence_{seconds}s"] = signed_persistence
        output[f"feature_regime_directional_consistency_{seconds}s"] = np.abs(signed_persistence)
        output[f"feature_regime_up_fraction_{seconds}s"] = up_fraction
        output[f"feature_regime_down_fraction_{seconds}s"] = down_fraction
    return output.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_precomputed_regime_context_features(output, frame):
    regime = compute_regime_context_feature_frame(frame)
    if len(output) == 0:
        return output
    if len(regime) == 0:
        for seconds in REGIME_CONTEXT_WINDOWS:
            for suffix in [
                "return",
                "trend_slope",
                "volatility",
                "range",
                "directional_persistence",
                "directional_consistency",
                "up_fraction",
                "down_fraction",
            ]:
                output[f"feature_regime_{suffix}_{seconds}s"] = 0.0
        return output
    merged = output.merge(regime, on="timestamp", how="left")
    regime_columns = [column for column in merged.columns if column.startswith("feature_regime_")]
    merged[regime_columns] = merged[regime_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return merged


def build_btc_regime_context_frame():
    global BTC_CONTEXT_CACHE
    if BTC_CONTEXT_CACHE is not None:
        return BTC_CONTEXT_CACHE.copy()
    if SYMBOL == BTC_CONTEXT_SYMBOL:
        BTC_CONTEXT_CACHE = pd.DataFrame()
        return BTC_CONTEXT_CACHE.copy()
    raw = read_csv(BTC_CONTEXT_PATH)
    frame = normalize_input(raw) if len(raw) else pd.DataFrame()
    if len(frame) == 0:
        BTC_CONTEXT_CACHE = pd.DataFrame()
        return BTC_CONTEXT_CACHE.copy()
    regime = compute_regime_context_feature_frame(frame)
    rename = {}
    for seconds in REGIME_CONTEXT_WINDOWS:
        rename.update(
            {
                f"feature_regime_return_{seconds}s": f"btc_return_{seconds}s",
                f"feature_regime_trend_slope_{seconds}s": f"btc_trend_slope_{seconds}s",
                f"feature_regime_volatility_{seconds}s": f"btc_volatility_{seconds}s",
                f"feature_regime_range_{seconds}s": f"btc_range_{seconds}s",
                f"feature_regime_directional_persistence_{seconds}s": f"btc_directional_persistence_{seconds}s",
            }
        )
    BTC_CONTEXT_CACHE = regime[["timestamp", *[column for column in rename if column in regime.columns]]].rename(columns=rename)
    return BTC_CONTEXT_CACHE.copy()


def add_btc_lead_lag_context(output):
    btc_columns = []
    for seconds in REGIME_CONTEXT_WINDOWS:
        btc_columns.extend(
            [
                f"btc_return_{seconds}s",
                f"btc_trend_slope_{seconds}s",
                f"btc_volatility_{seconds}s",
                f"btc_range_{seconds}s",
                f"btc_directional_persistence_{seconds}s",
            ]
        )
    output = add_asof_context(
        output,
        build_btc_regime_context_frame(),
        "btcleadlag",
        btc_columns,
        max_age_ms=BTC_CONTEXT_MAX_AGE_MS,
        join_policy=BTC_CONTEXT_JOIN_POLICY,
        unavailable_age_ms=0.0,
        add_age_seconds=True,
    )
    available = pd.to_numeric(output["feature_btcleadlag_context_available"], errors="coerce").fillna(0.0) > 0.5
    for seconds in REGIME_CONTEXT_WINDOWS:
        symbol_return = pd.to_numeric(output.get(f"feature_regime_return_{seconds}s", 0.0), errors="coerce").fillna(0.0)
        btc_return = pd.to_numeric(output.get(f"feature_btcleadlag_btc_return_{seconds}s", 0.0), errors="coerce").fillna(0.0)
        symbol_slope = pd.to_numeric(output.get(f"feature_regime_trend_slope_{seconds}s", 0.0), errors="coerce").fillna(0.0)
        btc_slope = pd.to_numeric(output.get(f"feature_btcleadlag_btc_trend_slope_{seconds}s", 0.0), errors="coerce").fillna(0.0)
        output[f"feature_btcleadlag_symbol_minus_btc_return_{seconds}s"] = (symbol_return - btc_return).where(available, 0.0)
        output[f"feature_btcleadlag_symbol_minus_btc_trend_slope_{seconds}s"] = (symbol_slope - btc_slope).where(available, 0.0)
        output[f"feature_btcleadlag_same_direction_{seconds}s"] = (
            (np.sign(symbol_return) == np.sign(btc_return)) & (symbol_return != 0.0) & (btc_return != 0.0) & available
        ).astype(float)
    return output


def previous_weekday_date(day):
    day = day - dt.timedelta(days=1)
    while day.weekday() >= 5:
        day = day - dt.timedelta(days=1)
    return day


def next_weekday_date(day):
    day = day + dt.timedelta(days=1)
    while day.weekday() >= 5:
        day = day + dt.timedelta(days=1)
    return day


def session_reference_minutes(local_timestamp, hour, minute, mode):
    day = local_timestamp.date()
    weekday = local_timestamp.weekday() < 5
    event_today = local_timestamp.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if mode == "since":
        if weekday and local_timestamp >= event_today:
            reference = event_today
        else:
            prev_day = previous_weekday_date(day)
            reference = pd.Timestamp(
                dt.datetime.combine(prev_day, dt.time(hour, minute)),
                tz=local_timestamp.tz,
            )
        return max(0.0, (local_timestamp - reference).total_seconds() / 60.0)
    if weekday and local_timestamp <= event_today:
        reference = event_today
    else:
        next_day = next_weekday_date(day)
        reference = pd.Timestamp(
            dt.datetime.combine(next_day, dt.time(hour, minute)),
            tz=local_timestamp.tz,
        )
    return max(0.0, (reference - local_timestamp).total_seconds() / 60.0)


def macro_importance_to_float(value):
    if pd.isna(value):
        return 0.0
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric_value):
        return float(numeric_value)
    text = str(value).strip().lower()
    mapping = {
        "low": 0.25,
        "medium": 0.50,
        "med": 0.50,
        "high": 1.00,
        "critical": 1.00,
    }
    return mapping.get(text, 0.0)


def load_macro_calendar():
    if not MACRO_CALENDAR_PATH.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(MACRO_CALENDAR_PATH, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()
    if "event_time_utc" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    parsed = pd.to_datetime(frame["event_time_utc"], errors="coerce", utc=True)
    frame["event_timestamp"] = (parsed.astype("int64") // 1_000_000).where(parsed.notna(), np.nan)
    if "event_importance" in frame.columns:
        frame["event_importance_numeric"] = frame["event_importance"].map(macro_importance_to_float)
    else:
        frame["event_importance_numeric"] = 0.0
    frame = frame.dropna(subset=["event_timestamp"]).sort_values("event_timestamp").drop_duplicates("event_timestamp")
    return frame.reset_index(drop=True)


def add_macro_calendar_features(output):
    timestamps = pd.to_numeric(output["timestamp"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    output["feature_minutes_to_next_macro_event"] = MACRO_EVENT_DEFAULT_MINUTES
    output["feature_minutes_since_last_macro_event"] = MACRO_EVENT_DEFAULT_MINUTES
    output["feature_is_macro_event_window"] = 0.0
    output["feature_macro_event_importance"] = 0.0
    events = load_macro_calendar()
    output.attrs["macro_calendar_loaded"] = bool(len(events))
    output.attrs["macro_event_count"] = int(len(events))
    if len(events) == 0:
        output.attrs["macro_event_window_pct"] = 0.0
        return output
    event_timestamps = events["event_timestamp"].to_numpy(dtype=np.int64)
    event_importance = events["event_importance_numeric"].to_numpy(dtype=np.float64)
    next_indices = np.searchsorted(event_timestamps, timestamps, side="left")
    prev_indices = next_indices - 1
    has_next = next_indices < len(event_timestamps)
    has_prev = prev_indices >= 0
    minutes_to = np.full(len(output), MACRO_EVENT_DEFAULT_MINUTES, dtype=np.float64)
    minutes_since = np.full(len(output), MACRO_EVENT_DEFAULT_MINUTES, dtype=np.float64)
    next_importance = np.zeros(len(output), dtype=np.float64)
    prev_importance = np.zeros(len(output), dtype=np.float64)
    minutes_to[has_next] = (event_timestamps[next_indices[has_next]] - timestamps[has_next]) / 60000.0
    minutes_since[has_prev] = (timestamps[has_prev] - event_timestamps[prev_indices[has_prev]]) / 60000.0
    next_importance[has_next] = event_importance[next_indices[has_next]]
    prev_importance[has_prev] = event_importance[prev_indices[has_prev]]
    in_window = np.minimum(minutes_to, minutes_since) <= MACRO_EVENT_WINDOW_MINUTES
    output["feature_minutes_to_next_macro_event"] = np.maximum(0.0, minutes_to)
    output["feature_minutes_since_last_macro_event"] = np.maximum(0.0, minutes_since)
    output["feature_is_macro_event_window"] = in_window.astype(float)
    output["feature_macro_event_importance"] = np.where(
        in_window,
        np.maximum(next_importance, prev_importance),
        next_importance,
    )
    output.attrs["macro_event_window_pct"] = float(in_window.mean()) if len(in_window) else 0.0
    return output


def add_calendar_session_features(output):
    if len(output) == 0:
        return output
    timestamps = pd.to_numeric(output["timestamp"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    utc_times = pd.to_datetime(timestamps, unit="ms", utc=True)
    minute_utc = (
        utc_times.hour.to_numpy(dtype=np.float64) * 60.0
        + utc_times.minute.to_numpy(dtype=np.float64)
        + utc_times.second.to_numpy(dtype=np.float64) / 60.0
    )
    day_of_week = utc_times.dayofweek.to_numpy(dtype=np.float64)
    output["feature_time_sin_utc"] = np.sin(2.0 * np.pi * minute_utc / 1440.0)
    output["feature_time_cos_utc"] = np.cos(2.0 * np.pi * minute_utc / 1440.0)
    output["feature_day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    output["feature_day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    output["feature_is_weekend"] = (day_of_week >= 5).astype(float)
    output["feature_is_london_session"] = ((minute_utc >= 8 * 60) & (minute_utc < 16 * 60 + 30)).astype(float)
    output["feature_is_asia_session"] = ((minute_utc >= 0) & (minute_utc < 8 * 60)).astype(float)

    ny_times = utc_times.tz_convert("America/New_York")
    ny_minute = (
        ny_times.hour.to_numpy(dtype=np.float64) * 60.0
        + ny_times.minute.to_numpy(dtype=np.float64)
        + ny_times.second.to_numpy(dtype=np.float64) / 60.0
    )
    ny_weekday = ny_times.dayofweek.to_numpy(dtype=np.int64) < 5
    us_open = 9 * 60 + 30
    us_close = 16 * 60
    output["feature_is_us_cash_session"] = (ny_weekday & (ny_minute >= us_open) & (ny_minute < us_close)).astype(float)
    output["feature_is_us_open_window"] = (ny_weekday & (ny_minute >= us_open - 30) & (ny_minute <= us_open + 30)).astype(float)
    output["feature_is_us_close_window"] = (ny_weekday & (ny_minute >= us_close - 30) & (ny_minute <= us_close + 30)).astype(float)
    output["feature_is_us_lunch_window"] = (ny_weekday & (ny_minute >= 12 * 60) & (ny_minute < 13 * 60)).astype(float)
    output["feature_minutes_since_us_equity_open"] = [
        session_reference_minutes(ts, 9, 30, "since") for ts in ny_times
    ]
    output["feature_minutes_until_us_equity_open"] = [
        session_reference_minutes(ts, 9, 30, "until") for ts in ny_times
    ]
    output["feature_minutes_since_us_equity_close"] = [
        session_reference_minutes(ts, 16, 0, "since") for ts in ny_times
    ]
    output["feature_minutes_until_us_equity_close"] = [
        session_reference_minutes(ts, 16, 0, "until") for ts in ny_times
    ]
    output = add_macro_calendar_features(output)
    output.attrs["calendar_session_features_enabled"] = True
    return output


def apply_feature_groups(output):
    if "calendar_session_features" in PRICE_TINY_FEATURE_GROUPS and len(output):
        output = add_calendar_session_features(output)
    if "cross_venue_features" in PRICE_TINY_FEATURE_GROUPS and len(output):
        output = add_asof_context(
            output,
            CROSS_VENUE_PATH,
            "crossvenue",
            [
                "venue_count",
                "venue_mid_diff_bps",
                "venue_spread_diff_bps",
                "log_bid_depth_ratio_10bps",
                "log_ask_depth_ratio_10bps",
                "venue_imbalance_diff_10bps",
                "leading_venue_return_1s",
                "leading_venue_return_3s",
                "leading_venue_return_10s",
                "cross_venue_pressure_agreement",
                "cross_venue_pressure_divergence",
            ],
            max_age_ms=PRICE_TINY_CROSSVENUE_MAX_AGE_MS,
            join_policy=PRICE_TINY_CROSSVENUE_JOIN_POLICY,
            unavailable_age_ms=0.0,
            add_age_seconds=True,
            min_context_column="venue_count",
            min_context_value=2,
            clean_return_sentinels=True,
        )
        if PRICE_TINY_REQUIRE_CROSSVENUE_CONTEXT:
            before = len(output)
            available = pd.to_numeric(output["feature_crossvenue_context_available"], errors="coerce").fillna(0.0) > 0.5
            output = output[available].copy().reset_index(drop=True)
            output.attrs["crossvenue_strict_dropped_rows"] = int(before - len(output))
    if "regime_context_features" in PRICE_TINY_FEATURE_GROUPS and len(output):
        output = add_btc_lead_lag_context(output)
    return output


def crossvenue_diagnostics(rows):
    diagnostics = {
        "crossvenue_requested": "cross_venue_features" in PRICE_TINY_FEATURE_GROUPS,
        "crossvenue_available_rows": 0,
        "crossvenue_available_pct": 0.0,
        "crossvenue_missing_rows": 0,
        "crossvenue_max_age_ms": 0.0,
        "crossvenue_median_age_ms": 0.0,
        "crossvenue_rows_by_venue_count": {},
        "crossvenue_column_summary": [],
        "crossvenue_join_policy": PRICE_TINY_CROSSVENUE_JOIN_POLICY,
        "crossvenue_max_join_age_ms": PRICE_TINY_CROSSVENUE_MAX_AGE_MS,
        "crossvenue_missing_policy": CROSSVENUE_MISSING_POLICY,
        "crossvenue_strict_required": PRICE_TINY_REQUIRE_CROSSVENUE_CONTEXT,
        "crossvenue_strict_dropped_rows": int(rows.attrs.get("crossvenue_strict_dropped_rows", 0)) if hasattr(rows, "attrs") else 0,
    }
    if len(rows) == 0 or "feature_crossvenue_context_available" not in rows.columns:
        return diagnostics
    available = pd.to_numeric(rows["feature_crossvenue_context_available"], errors="coerce").fillna(0.0) > 0.5
    diagnostics["crossvenue_available_rows"] = int(available.sum())
    diagnostics["crossvenue_missing_rows"] = int((~available).sum())
    diagnostics["crossvenue_available_pct"] = float(available.mean()) if len(available) else 0.0
    if "feature_crossvenue_context_age_ms" in rows.columns and available.any():
        ages = pd.to_numeric(rows.loc[available, "feature_crossvenue_context_age_ms"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(ages):
            diagnostics["crossvenue_max_age_ms"] = float(ages.max())
            diagnostics["crossvenue_median_age_ms"] = float(ages.median())
    if "feature_crossvenue_venue_count" in rows.columns:
        counts = pd.to_numeric(rows["feature_crossvenue_venue_count"], errors="coerce").fillna(0).astype(int).value_counts().sort_index()
        diagnostics["crossvenue_rows_by_venue_count"] = {str(int(key)): int(value) for key, value in counts.items()}
    cross_columns = [column for column in rows.columns if column.startswith("feature_crossvenue_")]
    for column in cross_columns:
        values = pd.to_numeric(rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        diagnostics["crossvenue_column_summary"].append(
            {
                "column": column,
                "nonzero": int((values.fillna(0.0) != 0.0).sum()),
                "nunique": int(values.nunique(dropna=True)),
                "min": float(values.min()) if len(values.dropna()) else np.nan,
                "max": float(values.max()) if len(values.dropna()) else np.nan,
            }
        )
    return diagnostics


def regime_context_diagnostics(rows):
    diagnostics = {
        "regime_context_requested": "regime_context_features" in PRICE_TINY_FEATURE_GROUPS,
        "btc_context_symbol": BTC_CONTEXT_SYMBOL,
        "btc_context_path": str(BTC_CONTEXT_PATH),
        "btc_context_available_rows": 0,
        "btc_context_available_pct": 0.0,
        "btc_context_missing_rows": 0,
        "btc_context_max_age_ms": 0.0,
        "btc_context_median_age_ms": 0.0,
        "btc_context_join_policy": BTC_CONTEXT_JOIN_POLICY,
        "btc_context_max_join_age_ms": BTC_CONTEXT_MAX_AGE_MS,
        "regime_context_column_summary": [],
    }
    if len(rows) == 0 or not diagnostics["regime_context_requested"]:
        return diagnostics
    if "feature_btcleadlag_context_available" in rows.columns:
        available = pd.to_numeric(rows["feature_btcleadlag_context_available"], errors="coerce").fillna(0.0) > 0.5
        diagnostics["btc_context_available_rows"] = int(available.sum())
        diagnostics["btc_context_missing_rows"] = int((~available).sum())
        diagnostics["btc_context_available_pct"] = float(available.mean()) if len(available) else 0.0
        if "feature_btcleadlag_context_age_ms" in rows.columns and available.any():
            ages = pd.to_numeric(rows.loc[available, "feature_btcleadlag_context_age_ms"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(ages):
                diagnostics["btc_context_max_age_ms"] = float(ages.max())
                diagnostics["btc_context_median_age_ms"] = float(ages.median())
    regime_columns = [
        column
        for column in rows.columns
        if column.startswith("feature_regime_") or column.startswith("feature_btcleadlag_")
    ]
    for column in regime_columns:
        values = pd.to_numeric(rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        diagnostics["regime_context_column_summary"].append(
            {
                "column": column,
                "nonzero": int((values.fillna(0.0) != 0.0).sum()),
                "nunique": int(values.nunique(dropna=True)),
                "min": float(values.min()) if len(values.dropna()) else np.nan,
                "max": float(values.max()) if len(values.dropna()) else np.nan,
            }
        )
    return diagnostics


def calendar_session_diagnostics(rows):
    requested = "calendar_session_features" in PRICE_TINY_FEATURE_GROUPS
    in_window_pct = 0.0
    if len(rows) and "feature_is_macro_event_window" in rows.columns:
        in_window_pct = float(pd.to_numeric(rows["feature_is_macro_event_window"], errors="coerce").fillna(0.0).mean())
    return {
        "calendar_session_requested": requested,
        "macro_calendar_path": str(MACRO_CALENDAR_PATH),
        "macro_calendar_loaded": bool(rows.attrs.get("macro_calendar_loaded", False)) if hasattr(rows, "attrs") else False,
        "macro_event_count": int(rows.attrs.get("macro_event_count", 0)) if hasattr(rows, "attrs") else 0,
        "macro_event_window_pct": in_window_pct,
    }


def net_aware_target_diagnostics(rows, raw_count):
    diagnostics = []
    horizons = sorted(set([*PRICE_TINY_TARGET_HORIZONS_SECONDS, HORIZON_SECONDS]))
    for horizon in horizons:
        target_column = f"target_move_before_adverse_net_aware_{horizon}s"
        labels = pd.to_numeric(rows[target_column], errors="coerce") if len(rows) and target_column in rows.columns else pd.Series(dtype="float64")
        ready = labels.notna() if len(labels) else pd.Series(dtype=bool)
        ready_rows = rows.loc[ready].copy() if len(rows) and len(ready) else pd.DataFrame()
        ready_labels = labels.loc[ready] if len(labels) else pd.Series(dtype="float64")
        class_balance = (
            {
                str(int(key)): int(value)
                for key, value in ready_labels.astype(int).value_counts().sort_index().to_dict().items()
            }
            if len(ready_labels)
            else {}
        )
        long_count = int((ready_labels > 0).sum()) if len(ready_labels) else 0
        short_count = int((ready_labels < 0).sum()) if len(ready_labels) else 0

        def avg_col(*columns):
            for column in columns:
                if column in ready_rows.columns:
                    values = pd.to_numeric(ready_rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if len(values):
                        return float(values.mean())
            return np.nan

        target_spec = "move_before_adverse_30s_net_aware" if int(horizon) == 30 else f"move_before_adverse_30s_net_aware_{int(horizon)}s"
        diagnostics.append(
            {
                "target_spec": target_spec,
                "horizon_seconds": int(horizon),
                "raw_rows": int(raw_count),
                "target_ready_rows": int(ready.sum()) if len(ready) else 0,
                "class_balance": class_balance,
                "long_short_balance": f"{long_count}:{short_count}",
                "average_mfe_bps": avg_col(
                    f"target_net_aware_max_favorable_excursion_bps_{horizon}s",
                    f"target_max_favorable_excursion_bps_{horizon}s",
                ),
                "average_mae_bps": avg_col(
                    f"target_net_aware_max_adverse_excursion_bps_{horizon}s",
                    f"target_max_adverse_excursion_bps_{horizon}s",
                ),
                "average_realized_return_bps": avg_col(
                    f"target_net_aware_realized_return_bps_{horizon}s",
                    f"target_return_bps_{horizon}s",
                ),
                "average_estimated_spread_cost_bps": avg_col(
                    f"target_net_aware_estimated_spread_cost_bps_{horizon}s",
                    "target_net_aware_estimated_spread_cost_bps",
                ),
                "net_favorable_rate": float((ready_labels != 0).mean()) if len(ready_labels) else np.nan,
            }
        )
    return diagnostics


def instability_target(frame, timestamps, mid, index, future_index_value, current_mid, path_bps, group_start):
    path_bps = np.asarray(path_bps, dtype=np.float64)
    if len(path_bps) == 0:
        return {
            "label": 0,
            "future_range_bps": 0.0,
            "recent_volatility_bps": 0.0,
            "reason_count": 0,
            "spread_widening": 0,
            "liquidity_drop": 0,
            "fast_reversal": 0,
        }
    recent_start = past_index(timestamps, index, max(60, HORIZON_SECONDS * 4), group_start)
    recent_mid = mid[recent_start : index + 1]
    recent_returns_bps = np.diff(np.log(recent_mid)) * 10000.0 if len(recent_mid) > 1 else np.asarray([0.0])
    recent_returns_bps = recent_returns_bps[np.isfinite(recent_returns_bps)]
    recent_volatility_bps = float(np.std(recent_returns_bps)) if len(recent_returns_bps) else 0.0
    horizon_steps = max(1, future_index_value - index)
    future_range_bps = float(np.max(path_bps) - np.min(path_bps))
    final_return_bps = float(path_bps[-1])
    max_runup_bps = float(np.max(path_bps))
    max_drawdown_bps = float(np.min(path_bps))
    range_threshold = max(
        PRICE_TINY_INSTABILITY_RETURN_BPS,
        PRICE_TINY_INSTABILITY_RANGE_MULT * recent_volatility_bps * math.sqrt(horizon_steps),
    )
    range_flag = future_range_bps >= range_threshold
    return_flag = abs(final_return_bps) >= PRICE_TINY_INSTABILITY_RETURN_BPS
    adverse_flag = max(abs(max_runup_bps), abs(max_drawdown_bps)) >= PRICE_TINY_INSTABILITY_ADVERSE_BPS
    up_hit = np.any(path_bps >= PRICE_TINY_INSTABILITY_ADVERSE_BPS)
    down_hit = np.any(path_bps <= -PRICE_TINY_INSTABILITY_ADVERSE_BPS)
    fast_reversal = bool(up_hit and down_hit)

    spread_widening = False
    if "spread_percent" in frame.columns:
        recent_spread = pd.to_numeric(frame.iloc[recent_start : index + 1]["spread_percent"], errors="coerce")
        future_spread = pd.to_numeric(frame.iloc[index + 1 : future_index_value + 1]["spread_percent"], errors="coerce")
        recent_baseline = float(recent_spread.replace([np.inf, -np.inf], np.nan).dropna().median()) if len(recent_spread.dropna()) else 0.0
        future_max = float(future_spread.replace([np.inf, -np.inf], np.nan).dropna().max()) if len(future_spread.dropna()) else 0.0
        spread_widening = recent_baseline > 0 and future_max >= PRICE_TINY_INSTABILITY_SPREAD_MULT * recent_baseline

    liquidity_drop = False
    if {"bid_depth_10bps", "ask_depth_10bps"}.issubset(frame.columns):
        recent_depth = (
            pd.to_numeric(frame.iloc[recent_start : index + 1]["bid_depth_10bps"], errors="coerce")
            + pd.to_numeric(frame.iloc[recent_start : index + 1]["ask_depth_10bps"], errors="coerce")
        )
        future_depth = (
            pd.to_numeric(frame.iloc[index + 1 : future_index_value + 1]["bid_depth_10bps"], errors="coerce")
            + pd.to_numeric(frame.iloc[index + 1 : future_index_value + 1]["ask_depth_10bps"], errors="coerce")
        )
        recent_depth_baseline = float(recent_depth.replace([np.inf, -np.inf], np.nan).dropna().median()) if len(recent_depth.dropna()) else 0.0
        future_depth_min = float(future_depth.replace([np.inf, -np.inf], np.nan).dropna().min()) if len(future_depth.dropna()) else recent_depth_baseline
        liquidity_drop = recent_depth_baseline > 0 and future_depth_min <= recent_depth_baseline / max(PRICE_TINY_INSTABILITY_SPREAD_MULT, 1e-9)

    flags = [range_flag, return_flag, adverse_flag, fast_reversal, spread_widening, liquidity_drop]
    reason_count = int(sum(bool(flag) for flag in flags))
    return {
        "label": 1 if reason_count > 0 else 0,
        "future_range_bps": future_range_bps,
        "recent_volatility_bps": recent_volatility_bps,
        "reason_count": reason_count,
        "spread_widening": int(bool(spread_widening)),
        "liquidity_drop": int(bool(liquidity_drop)),
        "fast_reversal": int(bool(fast_reversal)),
    }


def build_rows(frame):
    profile = get_lookback_profile()
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
    mid = frame["mid_price"].to_numpy(dtype=np.float64)
    group_starts, group_ends, has_simulation_runs = run_bounds(frame)
    rows = []
    skip_future = 0
    for index in range(len(frame)):
        group_start = int(group_starts[index])
        group_end = int(group_ends[index])
        f_index = future_index(timestamps, index, HORIZON_SECONDS, group_end)
        if f_index is None:
            skip_future += 1
            continue
        current_mid = mid[index]
        future_mid = mid[f_index]
        if not np.isfinite(current_mid) or not np.isfinite(future_mid) or current_mid <= 0 or future_mid <= 0:
            continue
        row = {
            "timestamp": int(timestamps[index]),
            "time": frame.iloc[index].get("time", ""),
            "symbol": SYMBOL,
            "primary_venue": PRIMARY_VENUE or "legacy",
            "feature_set_name": FEATURE_SET,
            "feature_groups": ",".join(PRICE_TINY_FEATURE_GROUPS),
            "feature_spec_name": "+".join(PRICE_TINY_FEATURE_GROUPS),
            "missing_feature_policy": MISSING_FEATURE_POLICY,
            "target_spec_name": TARGET_SPEC["name"],
            "target_label_method": TARGET_SPEC["label_construction_method"],
            "horizon_seconds": HORIZON_SECONDS,
            "target_horizon_seconds": HORIZON_SECONDS,
            "lookback_profile": profile["name"],
            "current_mid_price": current_mid,
            "future_mid_price": future_mid,
        }
        for debug_column in ["simulation_run_id", "source_scenario", "source_seed"]:
            if debug_column in frame.columns:
                row[debug_column] = str(frame.iloc[index].get(debug_column, ""))
        future_return = float(np.log(future_mid / current_mid))
        delta_bps = float((future_mid / current_mid - 1.0) * 10000.0)
        path_bps = path_returns_bps(mid, index, f_index, current_mid)
        max_runup = float(np.max(path_bps)) if len(path_bps) else delta_bps
        max_drawdown = float(np.min(path_bps)) if len(path_bps) else delta_bps
        first_touch = first_touch_direction(path_bps, PRICE_TINY_TARGET_MOVE_BPS)
        add_net_aware_targets(
            row,
            frame,
            timestamps,
            mid,
            index,
            group_end,
            sorted(set([*PRICE_TINY_TARGET_HORIZONS_SECONDS, HORIZON_SECONDS])),
        )
        row["target_next_mid_log_return_1s"] = future_return
        row["target_next_mid_delta_bps_1s"] = delta_bps
        row["target_next_mid_direction_1s"] = 1 if delta_bps > PRICE_TINY_FLAT_BPS else (-1 if delta_bps < -PRICE_TINY_FLAT_BPS else 0)
        row[f"target_next_mid_log_return_{HORIZON_SECONDS}s"] = future_return
        row[f"target_next_mid_delta_bps_{HORIZON_SECONDS}s"] = delta_bps
        if TARGET_SPEC["label_construction_method"] == "move_before_adverse_net_aware":
            target_direction = int(row.get(f"target_move_before_adverse_net_aware_{HORIZON_SECONDS}s", 0) or 0)
        elif TARGET_SPEC["label_construction_method"] in {"first_touch", "move_before_adverse"}:
            target_direction = first_touch
        elif TARGET_SPEC["label_construction_method"] == "chop_no_trade":
            target_direction = 0 if abs(delta_bps) <= PRICE_TINY_TARGET_MOVE_BPS else (1 if delta_bps > 0 else -1)
        else:
            target_direction = 1 if delta_bps > PRICE_TINY_FLAT_BPS else (-1 if delta_bps < -PRICE_TINY_FLAT_BPS else 0)
        row[f"target_next_mid_direction_{HORIZON_SECONDS}s"] = target_direction
        row[f"target_return_bps_{HORIZON_SECONDS}s"] = delta_bps
        row[f"target_direction_{HORIZON_SECONDS}s"] = target_direction
        row[f"target_first_touch_direction_{HORIZON_SECONDS}s"] = first_touch
        row[f"target_max_favorable_excursion_bps_{HORIZON_SECONDS}s"] = max(0.0, max_runup)
        row[f"target_max_adverse_excursion_bps_{HORIZON_SECONDS}s"] = min(0.0, max_drawdown)
        row[f"target_move_before_adverse_{HORIZON_SECONDS}s"] = first_touch
        row[f"target_move_before_adverse_net_aware_{HORIZON_SECONDS}s"] = int(
            row.get(f"target_move_before_adverse_net_aware_{HORIZON_SECONDS}s", target_direction)
        )
        row[f"target_chop_no_trade_{HORIZON_SECONDS}s"] = 1 if abs(delta_bps) <= PRICE_TINY_TARGET_MOVE_BPS else 0
        if TARGET_SPEC["label_construction_method"] == "instability":
            instability = instability_target(
                frame,
                timestamps,
                mid,
                index,
                f_index,
                current_mid,
                path_bps,
                group_start,
            )
            row[f"target_instability_{HORIZON_SECONDS}s"] = instability["label"]
            row[f"target_future_range_bps_{HORIZON_SECONDS}s"] = instability["future_range_bps"]
            row[f"target_instability_recent_volatility_bps_{HORIZON_SECONDS}s"] = instability["recent_volatility_bps"]
            row[f"target_instability_reason_count_{HORIZON_SECONDS}s"] = instability["reason_count"]
            row[f"target_instability_spread_widening_{HORIZON_SECONDS}s"] = instability["spread_widening"]
            row[f"target_instability_liquidity_drop_{HORIZON_SECONDS}s"] = instability["liquidity_drop"]
            row[f"target_instability_fast_reversal_{HORIZON_SECONDS}s"] = instability["fast_reversal"]

        rolling_mean_seconds = int(profile["rolling_mean_seconds"])
        rolling_start = past_index(timestamps, index, rolling_mean_seconds, group_start)
        rolling_mid = mid[rolling_start : index + 1]
        rolling_mean = float(np.nanmean(rolling_mid)) if len(rolling_mid) else current_mid
        row[f"feature_mid_vs_rolling_mean_{rolling_mean_seconds}s"] = current_mid / max(rolling_mean, 1e-12) - 1.0
        row["feature_spread_percent"] = float(frame.iloc[index].get("spread_percent", 0.0))
        row["feature_bid_distance_to_mid"] = (current_mid - float(frame.iloc[index].get("best_bid", current_mid))) / current_mid
        row["feature_ask_distance_to_mid"] = (float(frame.iloc[index].get("best_ask", current_mid)) - current_mid) / current_mid
        for column in ["bid_depth_10bps", "ask_depth_10bps", "bid_depth_25bps", "ask_depth_25bps"]:
            row[f"feature_{column}_log1p"] = float(np.log1p(max(0.0, float(frame.iloc[index].get(column, 0.0)))))
        row["feature_imbalance10"] = float(frame.iloc[index].get("order_book_imbalance_10bps", 0.0))
        row["feature_imbalance25"] = float(frame.iloc[index].get("order_book_imbalance_25bps", 0.0))
        for seconds in profile["return_seconds"]:
            p_index = past_index(timestamps, index, seconds, group_start)
            past_mid = mid[p_index]
            row[f"feature_mid_return_{seconds}s"] = float(np.log(current_mid / past_mid)) if past_mid > 0 else 0.0
        for seconds in profile["volatility_seconds"]:
            p_index = past_index(timestamps, index, seconds, group_start)
            window_mid = mid[p_index : index + 1]
            returns = np.diff(np.log(window_mid)) if len(window_mid) > 1 else np.asarray([0.0])
            row[f"feature_rolling_volatility_{seconds}s"] = float(np.nanstd(returns))
        for seconds in profile["range_seconds"]:
            p_index = past_index(timestamps, index, seconds, group_start)
            window_mid = mid[p_index : index + 1]
            row[f"feature_recent_high_low_range_{seconds}s"] = float((np.nanmax(window_mid) / max(np.nanmin(window_mid), 1e-12)) - 1.0)

        if FEATURE_SET in {"tiny_price_v2", "tiny_price_v3", "tiny_price_v4"} or "pressure_change_features" in PRICE_TINY_FEATURE_GROUPS:
            add_pressure_change_features(row, frame, timestamps, index, group_start)
        if "spread_volatility_features" in PRICE_TINY_FEATURE_GROUPS:
            add_spread_volatility_features(row, frame, timestamps, index, group_start)
        if "depth_acceleration_features" in PRICE_TINY_FEATURE_GROUPS:
            add_depth_acceleration_features(row, frame, timestamps, index, group_start)
        if "snapshot_freshness_features" in PRICE_TINY_FEATURE_GROUPS:
            add_snapshot_freshness_features(row, timestamps, index, group_start)
        rows.append(row)
    output = pd.DataFrame(rows)
    if "regime_context_features" in PRICE_TINY_FEATURE_GROUPS and len(output):
        output = add_precomputed_regime_context_features(output, frame)
    output = apply_feature_groups(output)
    if FEATURE_SET in {"tiny_price_v3", "tiny_price_v4"} and len(output):
        output = add_asof_context(
            output,
            FLOW_1S_PATH,
            "flow1s",
            [
                "prob_sell_dominant_1s",
                "prob_neutral_1s",
                "prob_buy_dominant_1s",
                "pred_market_pressure_1s",
                "buy_burst_prob_1s",
                "sell_burst_prob_1s",
            ],
        )
    if FEATURE_SET == "tiny_price_v4" and len(output):
        output = add_asof_context(
            output,
            MICRO_10S_PATH,
            "micro10s",
            [
                "prob_upside_scare_event_10s",
                "prob_downside_scare_event_10s",
                "prob_spread_expansion_event_10s",
                "prob_bid_liquidity_drop_10s",
                "prob_ask_liquidity_drop_10s",
            ],
        )
    output.attrs["skip_future"] = skip_future
    output.attrs["simulation_run_boundaries_enforced"] = bool(has_simulation_runs)
    return output


def main():
    raw = read_csv(INPUT_PATH)
    frame = normalize_input(raw) if len(raw) else pd.DataFrame()
    if PRICE_TINY_BUILD_MAX_ROWS > 0 and len(frame) > PRICE_TINY_BUILD_MAX_ROWS:
        frame = frame.tail(PRICE_TINY_BUILD_MAX_ROWS).copy().reset_index(drop=True)
    rows = build_rows(frame) if len(frame) else pd.DataFrame()
    simulation_run_boundaries_enforced = bool(
        rows.attrs.get("simulation_run_boundaries_enforced", False)
    ) if hasattr(rows, "attrs") else False
    cross_diag = crossvenue_diagnostics(rows)
    regime_diag = regime_context_diagnostics(rows)
    calendar_diag = calendar_session_diagnostics(rows)
    net_aware_diag = net_aware_target_diagnostics(rows, len(raw))
    experiment_output_path = LATEST_OUTPUT_PATH
    if len(rows):
        feature_columns = select_model_feature_columns(rows)
        rows[feature_columns] = rows[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        schema_hash = feature_schema_hash(sorted(feature_columns))
        target_columns = select_target_columns(rows)
        experiment_output_path = experiment_training_rows_path(schema_hash)
        rows["model_feature_columns"] = ",".join(feature_columns)
        rows["feature_schema_hash"] = schema_hash
        rows["target_columns"] = ",".join(sorted(target_columns))
        rows["training_rows_path"] = str(experiment_output_path)
        rows["training_rows_latest_path"] = str(LATEST_OUTPUT_PATH)
        rows["crossvenue_available_pct"] = cross_diag["crossvenue_available_pct"]
        rows["crossvenue_max_age_ms"] = cross_diag["crossvenue_max_age_ms"]
        rows["crossvenue_median_age_ms"] = cross_diag["crossvenue_median_age_ms"]
        rows["crossvenue_join_policy"] = cross_diag["crossvenue_join_policy"]
        rows["crossvenue_missing_policy"] = cross_diag["crossvenue_missing_policy"]
        rows["crossvenue_max_join_age_ms"] = cross_diag["crossvenue_max_join_age_ms"]
        rows["crossvenue_strict_required"] = cross_diag["crossvenue_strict_required"]
        rows["regime_context_btc_available_pct"] = regime_diag["btc_context_available_pct"]
        rows["regime_context_btc_max_age_ms"] = regime_diag["btc_context_max_age_ms"]
        rows["regime_context_btc_join_policy"] = regime_diag["btc_context_join_policy"]
        rows["regime_context_btc_symbol"] = regime_diag["btc_context_symbol"]
        rows["calendar_session_requested"] = calendar_diag["calendar_session_requested"]
        rows["macro_calendar_loaded"] = calendar_diag["macro_calendar_loaded"]
        rows["macro_event_count"] = calendar_diag["macro_event_count"]
        rows["macro_event_window_pct"] = calendar_diag["macro_event_window_pct"]
    atomic_write_csv(rows, experiment_output_path)
    atomic_write_csv(rows, LATEST_OUTPUT_PATH)
    atomic_write_json(
        {
            "symbol": SYMBOL,
            "primary_venue": PRIMARY_VENUE or "legacy",
            "latest_training_rows_path": str(LATEST_OUTPUT_PATH),
            "training_rows_path": str(experiment_output_path),
            "feature_groups": PRICE_TINY_FEATURE_GROUPS,
            "feature_spec": "+".join(PRICE_TINY_FEATURE_GROUPS),
            "target_spec": TARGET_SPEC["name"],
            "target_label_method": TARGET_SPEC["label_construction_method"],
            "horizon_seconds": HORIZON_SECONDS,
            "feature_schema_hash": rows["feature_schema_hash"].iloc[0] if len(rows) and "feature_schema_hash" in rows.columns else "",
            "row_count": int(len(rows)),
            "simulation_run_boundaries_enforced": simulation_run_boundaries_enforced,
            "crossvenue_available_pct": cross_diag["crossvenue_available_pct"],
            "crossvenue_max_age_ms": cross_diag["crossvenue_max_age_ms"],
            "crossvenue_join_policy": cross_diag["crossvenue_join_policy"],
            "crossvenue_missing_policy": cross_diag["crossvenue_missing_policy"],
            "regime_context_btc_available_pct": regime_diag["btc_context_available_pct"],
            "regime_context_btc_max_age_ms": regime_diag["btc_context_max_age_ms"],
            "regime_context_btc_join_policy": regime_diag["btc_context_join_policy"],
            "regime_context_btc_symbol": regime_diag["btc_context_symbol"],
            "calendar_session_requested": calendar_diag["calendar_session_requested"],
            "macro_calendar_path": calendar_diag["macro_calendar_path"],
            "macro_calendar_loaded": calendar_diag["macro_calendar_loaded"],
            "macro_event_count": calendar_diag["macro_event_count"],
            "macro_event_window_pct": calendar_diag["macro_event_window_pct"],
            "net_aware_target_diagnostics": net_aware_diag,
            "net_aware_target_min_net_bps": PRICE_TINY_TARGET_MIN_NET_BPS,
            "net_aware_target_spread_multiplier": PRICE_TINY_TARGET_SPREAD_MULTIPLIER,
            "net_aware_target_fee_bps": PRICE_TINY_TARGET_FEE_BPS,
            "net_aware_target_slippage_bps": PRICE_TINY_TARGET_SLIPPAGE_BPS,
            "net_aware_target_horizons_seconds": PRICE_TINY_TARGET_HORIZONS_SECONDS,
            "paper_only": True,
        },
        LATEST_METADATA_PATH,
    )
    print("Tiny price training row builder")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Input: {INPUT_PATH}")
    print(f"Experiment output: {experiment_output_path}")
    print(f"Latest pointer output: {LATEST_OUTPUT_PATH}")
    print(f"Latest pointer metadata: {LATEST_METADATA_PATH}")
    print(f"Feature set: {FEATURE_SET}")
    print(f"Feature groups: {','.join(PRICE_TINY_FEATURE_GROUPS)}")
    print(f"Target spec: {TARGET_SPEC['name']}")
    print(f"Target label method: {TARGET_SPEC['label_construction_method']}")
    print(f"Horizon seconds: {HORIZON_SECONDS}")
    print("Net-aware target config:")
    print(f"- PRICE_TINY_TARGET_MIN_NET_BPS: {PRICE_TINY_TARGET_MIN_NET_BPS}")
    print(f"- PRICE_TINY_TARGET_SPREAD_MULTIPLIER: {PRICE_TINY_TARGET_SPREAD_MULTIPLIER}")
    print(f"- PRICE_TINY_TARGET_FEE_BPS: {PRICE_TINY_TARGET_FEE_BPS}")
    print(f"- PRICE_TINY_TARGET_SLIPPAGE_BPS: {PRICE_TINY_TARGET_SLIPPAGE_BPS}")
    print(f"- PRICE_TINY_TARGET_HORIZONS_SECONDS: {PRICE_TINY_TARGET_HORIZONS_SECONDS}")
    print(f"Lookback profile: {get_lookback_profile()['name']}")
    print(f"Raw rows: {len(raw)}")
    print(f"Valid snapshot rows: {len(frame)}")
    print(f"Build max rows cap: {PRICE_TINY_BUILD_MAX_ROWS if PRICE_TINY_BUILD_MAX_ROWS > 0 else 'none'}")
    print(f"Training rows: {len(rows)}")
    print(f"Skipped rows without future target: {rows.attrs.get('skip_future', 0) if hasattr(rows, 'attrs') else 0}")
    print(f"Simulation run boundaries enforced: {simulation_run_boundaries_enforced}")
    selected_feature_columns = select_model_feature_columns(rows) if len(rows) else []
    print(f"Feature columns: {selected_feature_columns}")
    print(f"Feature count: {len(selected_feature_columns)}")
    print(f"Feature schema hash: {rows['feature_schema_hash'].iloc[0] if len(rows) and 'feature_schema_hash' in rows.columns else ''}")
    if cross_diag["crossvenue_requested"]:
        print("Cross-venue diagnostics")
        print(f"- crossvenue_available_rows: {cross_diag['crossvenue_available_rows']}")
        print(f"- crossvenue_available_pct: {cross_diag['crossvenue_available_pct']:.4%}")
        print(f"- crossvenue_missing_rows: {cross_diag['crossvenue_missing_rows']}")
        print(f"- crossvenue_max_age_ms: {cross_diag['crossvenue_max_age_ms']:.2f}")
        print(f"- crossvenue_median_age_ms: {cross_diag['crossvenue_median_age_ms']:.2f}")
        print(f"- crossvenue_rows_by_venue_count: {cross_diag['crossvenue_rows_by_venue_count']}")
        print(f"- crossvenue_join_policy: {cross_diag['crossvenue_join_policy']}")
        print(f"- crossvenue_max_join_age_ms: {cross_diag['crossvenue_max_join_age_ms']}")
        print(f"- crossvenue_missing_policy: {cross_diag['crossvenue_missing_policy']}")
        print(f"- crossvenue_strict_required: {cross_diag['crossvenue_strict_required']}")
        print(f"- crossvenue_strict_dropped_rows: {cross_diag['crossvenue_strict_dropped_rows']}")
        print("- per-crossvenue-column nonzero/nunique summary:")
        for item in cross_diag["crossvenue_column_summary"]:
            print(
                f"  {item['column']}: nonzero={item['nonzero']} "
                f"nunique={item['nunique']} min={item['min']:.6g} max={item['max']:.6g}"
            )
    if regime_diag["regime_context_requested"]:
        print("Regime context diagnostics")
        print(f"- btc_context_symbol: {regime_diag['btc_context_symbol']}")
        print(f"- btc_context_path: {regime_diag['btc_context_path']}")
        print(f"- btc_context_available_rows: {regime_diag['btc_context_available_rows']}")
        print(f"- btc_context_available_pct: {regime_diag['btc_context_available_pct']:.4%}")
        print(f"- btc_context_missing_rows: {regime_diag['btc_context_missing_rows']}")
        print(f"- btc_context_max_age_ms: {regime_diag['btc_context_max_age_ms']:.2f}")
        print(f"- btc_context_median_age_ms: {regime_diag['btc_context_median_age_ms']:.2f}")
        print(f"- btc_context_join_policy: {regime_diag['btc_context_join_policy']}")
        print(f"- btc_context_max_join_age_ms: {regime_diag['btc_context_max_join_age_ms']}")
        print("- per-regime-context-column nonzero/nunique summary:")
        for item in regime_diag["regime_context_column_summary"]:
            print(
                f"  {item['column']}: nonzero={item['nonzero']} "
                f"nunique={item['nunique']} min={item['min']:.6g} max={item['max']:.6g}"
            )
    if calendar_diag["calendar_session_requested"]:
        print("Calendar/session diagnostics")
        print("- calendar_session_features_enabled: True")
        print(f"- macro_calendar_path: {calendar_diag['macro_calendar_path']}")
        print(f"- macro_calendar_loaded: {calendar_diag['macro_calendar_loaded']}")
        print(f"- macro_event_count: {calendar_diag['macro_event_count']}")
        print(f"- macro_event_window_pct: {calendar_diag['macro_event_window_pct']:.4%}")
    print("Net-aware move-before-adverse target diagnostics")
    for item in net_aware_diag:
        print(f"- target_spec: {item['target_spec']}")
        print(f"  raw_rows: {item['raw_rows']}")
        print(f"  target_ready_rows: {item['target_ready_rows']}")
        print(f"  class_balance: {item['class_balance']}")
        print(f"  long_short_balance: {item['long_short_balance']}")
        print(f"  average_mfe_bps: {item['average_mfe_bps']:.4f}" if np.isfinite(item["average_mfe_bps"]) else "  average_mfe_bps: n/a")
        print(f"  average_mae_bps: {item['average_mae_bps']:.4f}" if np.isfinite(item["average_mae_bps"]) else "  average_mae_bps: n/a")
        print(f"  average_realized_return_bps: {item['average_realized_return_bps']:.4f}" if np.isfinite(item["average_realized_return_bps"]) else "  average_realized_return_bps: n/a")
        print(f"  average_estimated_spread_cost_bps: {item['average_estimated_spread_cost_bps']:.4f}" if np.isfinite(item["average_estimated_spread_cost_bps"]) else "  average_estimated_spread_cost_bps: n/a")
        print(f"  net_favorable_rate: {item['net_favorable_rate']:.2%}" if np.isfinite(item["net_favorable_rate"]) else "  net_favorable_rate: n/a")
    if len(rows):
        counts = rows["target_next_mid_direction_1s"].value_counts().to_dict()
        print(f"Direction distribution: {counts}")
        instability_column = f"target_instability_{HORIZON_SECONDS}s"
        if instability_column in rows.columns:
            instability_counts = rows[instability_column].value_counts().sort_index().to_dict()
            unstable_pct = float(pd.to_numeric(rows[instability_column], errors="coerce").fillna(0.0).mean())
            print(f"Instability target distribution ({instability_column}): {instability_counts}")
            print(f"Percent unstable: {unstable_pct:.2%}")
            print("Instability target is a paper-only risk/gating label, not a direction model.")
    print("Paper-only. No trades/orders/private API.")


if __name__ == "__main__":
    main()
