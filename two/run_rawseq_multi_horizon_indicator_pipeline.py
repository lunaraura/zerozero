#!/usr/bin/env python3
"""Paper-only multi-horizon return prediction with causal indicator features.

This script builds a canonical technical-indicator feature bank from current
and historical OHLCV-like data, predicts market-relative cumulative future
returns over multiple output horizons, and compares simple baselines with a
multi-output ridge model plus an optional small MLP.

Safety: no private API, no orders, no promotion, and no champion mutation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


warnings.simplefilter("ignore", pd.errors.PerformanceWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "realtime" / "kraken" / "SOLUSDT_10s_flow.csv"

SOURCE_PATH = Path(os.getenv("RAWSEQ_MH_SOURCE_PATH", str(DEFAULT_SOURCE))).expanduser()
if not SOURCE_PATH.is_absolute():
    SOURCE_PATH = PROJECT_ROOT / SOURCE_PATH

OUTPUT_ROOT = Path(
    os.getenv(
        "RAWSEQ_MH_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_multi_horizon_indicator_returns"),
    )
).expanduser()
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper() or "SOLUSDT"
VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower() or "kraken"
INSTRUMENT = os.getenv("RAWSEQ_MH_INSTRUMENT", "inventory_spot_long_flat").strip().lower()
HORIZON_BUCKETS = [
    int(float(item.strip()))
    for item in os.getenv("RAWSEQ_MH_HORIZON_BUCKETS", "1,3,6,12,24,48").split(",")
    if item.strip()
]
FEATURE_WINDOWS = [
    int(float(item.strip()))
    for item in os.getenv("RAWSEQ_MH_FEATURE_WINDOWS", "3,5,10,20,30,60,120,240").split(",")
    if item.strip()
]
RIDGE_ALPHAS = [
    float(item.strip())
    for item in os.getenv("RAWSEQ_MH_RIDGE_ALPHAS", "0.01,0.1,1,10,100").split(",")
    if item.strip()
]
ELASTIC_NET_ALPHAS = [
    float(item.strip())
    for item in os.getenv("RAWSEQ_MH_ELASTIC_NET_ALPHAS", "0.001,0.01,0.1").split(",")
    if item.strip()
]
ELASTIC_NET_L1_RATIO = float(os.getenv("RAWSEQ_MH_ELASTIC_NET_L1_RATIO", "0.25"))
ELASTIC_NET_ITERATIONS = int(float(os.getenv("RAWSEQ_MH_ELASTIC_NET_ITERATIONS", "120")))
ELASTIC_NET_LR = float(os.getenv("RAWSEQ_MH_ELASTIC_NET_LR", "0.02"))
THRESHOLDS_BPS = [
    float(item.strip())
    for item in os.getenv("RAWSEQ_MH_THRESHOLDS_BPS", "0,0.1,0.25,0.5,1,2").split(",")
    if item.strip()
]
COST_BPS = [
    float(item.strip())
    for item in os.getenv("RAWSEQ_MH_COSTS_BPS", "0.1,1,5").split(",")
    if item.strip()
]
DECISION_COST_BPS = float(os.getenv("RAWSEQ_MH_DECISION_COST_BPS", str(COST_BPS[0] if COST_BPS else 0.1)))
VALIDATION_FRACTION = float(os.getenv("RAWSEQ_MH_VALIDATION_FRACTION", "0.20"))
HOLDOUT_FRACTION = float(os.getenv("RAWSEQ_MH_HOLDOUT_FRACTION", "0.20"))
PURGE_ROWS_ENV = os.getenv("RAWSEQ_MH_PURGE_ROWS", "").strip()
EMBARGO_ROWS_ENV = os.getenv("RAWSEQ_MH_EMBARGO_ROWS", "").strip()
MAX_ROWS_ENV = os.getenv("RAWSEQ_MH_MAX_ROWS", "").strip()
MIN_POSITION_TRADES = int(float(os.getenv("RAWSEQ_MH_MIN_POSITION_TRADES", "30")))
ENABLE_MLP = os.getenv("RAWSEQ_MH_ENABLE_MLP", "false").strip().lower() in {"1", "true", "yes", "y"}
ENABLE_SEQUENCE_MLP = os.getenv("RAWSEQ_MH_ENABLE_SEQUENCE_MLP", "false").strip().lower() in {"1", "true", "yes", "y"}
ENABLE_TORCH_SEQUENCE_MODELS = os.getenv("RAWSEQ_MH_ENABLE_TORCH_SEQUENCE_MODELS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
ENABLE_TREE_BASELINE = os.getenv("RAWSEQ_MH_ENABLE_TREE_BASELINE", "true").strip().lower() in {"1", "true", "yes", "y"}
ENABLE_BOOSTED_TREE_BASELINE = os.getenv("RAWSEQ_MH_ENABLE_BOOSTED_TREE_BASELINE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
INCLUDE_FLOW_FEATURES = os.getenv("RAWSEQ_MH_INCLUDE_FLOW_FEATURES", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
FEATURE_GROUP_SPECS = [
    item.strip()
    for item in os.getenv(
        "RAWSEQ_MH_ABLATION_GROUPS",
        "raw,raw_trend,raw_momentum,raw_volatility,raw_volume,raw_cross_market,all",
    ).split(",")
    if item.strip()
]
MLP_HIDDEN = int(float(os.getenv("RAWSEQ_MH_MLP_HIDDEN", "32")))
MLP_EPOCHS = int(float(os.getenv("RAWSEQ_MH_MLP_EPOCHS", "20")))
MLP_LR = float(os.getenv("RAWSEQ_MH_MLP_LR", "0.01"))
MLP_SEED = int(float(os.getenv("RAWSEQ_MH_MLP_SEED", "900")))
SEQUENCE_LEN = int(float(os.getenv("RAWSEQ_MH_SEQUENCE_LEN", "60")))
SEQUENCE_LENS = [
    int(float(item.strip()))
    for item in os.getenv("RAWSEQ_MH_SEQUENCE_LENS", "60,120,240").split(",")
    if item.strip()
]
if not SEQUENCE_LENS:
    SEQUENCE_LENS = [SEQUENCE_LEN]
SEQUENCE_MLP_HIDDEN = int(float(os.getenv("RAWSEQ_MH_SEQUENCE_MLP_HIDDEN", str(MLP_HIDDEN))))
SEQUENCE_MLP_EPOCHS = int(float(os.getenv("RAWSEQ_MH_SEQUENCE_MLP_EPOCHS", str(MLP_EPOCHS))))
SEQUENCE_MLP_MAX_FEATURES = int(float(os.getenv("RAWSEQ_MH_SEQUENCE_MLP_MAX_FEATURES", "48")))
SEQUENCE_DATASET_MAX_FEATURES = int(
    float(os.getenv("RAWSEQ_MH_SEQUENCE_DATASET_MAX_FEATURES", str(SEQUENCE_MLP_MAX_FEATURES)))
)
WRITE_SEQUENCE_DATASETS = os.getenv("RAWSEQ_MH_WRITE_SEQUENCE_DATASETS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
TREE_MAX_FEATURES = int(float(os.getenv("RAWSEQ_MH_TREE_MAX_FEATURES", "40")))
BOOSTED_TREE_ROUNDS = int(float(os.getenv("RAWSEQ_MH_BOOSTED_TREE_ROUNDS", "12")))
BOOSTED_TREE_LEARNING_RATE = float(os.getenv("RAWSEQ_MH_BOOSTED_TREE_LEARNING_RATE", "0.1"))
TORCH_SEQUENCE_MODELS = [
    item.strip().lower()
    for item in os.getenv("RAWSEQ_MH_TORCH_SEQUENCE_MODELS", "tcn,gru,lstm,transformer").split(",")
    if item.strip()
]
TORCH_SEQUENCE_EPOCHS = int(float(os.getenv("RAWSEQ_MH_TORCH_SEQUENCE_EPOCHS", "5")))
TORCH_SEQUENCE_HIDDEN = int(float(os.getenv("RAWSEQ_MH_TORCH_SEQUENCE_HIDDEN", "32")))
TORCH_SEQUENCE_LR = float(os.getenv("RAWSEQ_MH_TORCH_SEQUENCE_LR", "0.001"))
CROSS_MARKET_SOURCES = {
    "BTC": os.getenv("RAWSEQ_MH_BTC_SOURCE_PATH", "").strip(),
    "ETH": os.getenv("RAWSEQ_MH_ETH_SOURCE_PATH", "").strip(),
}
MACD_SPEC = [
    int(float(item.strip()))
    for item in os.getenv("RAWSEQ_MH_MACD_SPEC", "12,26,9").split(",")
    if item.strip()
]
if len(MACD_SPEC) != 3:
    MACD_SPEC = [12, 26, 9]

NAMED_BASELINE_MODELS = {
    "zero_return_baseline",
    "training_mean_return_baseline",
    "rolling_mean_momentum_baseline",
    "mean_reversion_baseline",
}
ACCEPTANCE_BASELINE_MODELS = {
    "zero_return_baseline",
    "rolling_mean_momentum_baseline",
    "mean_reversion_baseline",
}

FEATURE_GROUP_FAMILIES = {
    "raw": {"raw"},
    "raw_trend": {"raw", "trend", "regime"},
    "raw_momentum": {"raw", "momentum", "regime"},
    "raw_volatility": {"raw", "volatility", "regime"},
    "raw_volume": {"raw", "volume", "regime"},
    "raw_cross_market": {"raw", "cross_market", "regime"},
    "all": {"raw", "trend", "momentum", "volatility", "breakout", "volume", "regime", "cross_market", "flow"},
}


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def iso_from_ms(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC).isoformat()
    except Exception:
        return ""


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def base_model_name(model: Any) -> str:
    text = str(model)
    return text.split("__", 1)[1] if "__" in text else text


def model_with_group(group: str, model: str) -> str:
    return f"{group}__{model}" if group and group != "legacy" else model


def infer_feature_family(feature: str) -> str:
    name = str(feature).lower()
    if any(token in name for token in ["macd", "sma_", "ema_", "slope", "donchian", "channel_position"]):
        return "trend"
    if any(token in name for token in ["rsi", "stochastic", "roc", "cci", "williams"]):
        return "momentum"
    if any(token in name for token in ["atr", "bollinger", "volatility", "parkinson", "compression", "realized"]):
        return "volatility"
    if any(token in name for token in ["recent_high", "recent_low", "zscore", "distance_to"]):
        return "breakout"
    if any(token in name for token in ["volume", "obv", "mfi", "vwap", "buy_sell", "pressure"]):
        return "volume"
    if any(token in name for token in ["time_of_day", "day_of_week", "regime", "session"]):
        return "regime"
    if any(token in name for token in ["btc", "eth", "sol_btc", "sol_eth", "cross_market"]):
        return "cross_market"
    if any(token in name for token in ["spread", "depth", "imbalance", "book"]):
        return "flow"
    return "raw"


def feature_family_manifest(feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "feature_family": infer_feature_family(feature),
                "causal_inputs_only": True,
            }
            for feature in feature_columns
        ]
    )


def selected_feature_columns_for_group(
    feature_columns: list[str],
    family_manifest: pd.DataFrame,
    group_spec: str,
) -> list[str]:
    group = str(group_spec).strip().lower()
    if group not in FEATURE_GROUP_FAMILIES:
        return feature_columns
    allowed = FEATURE_GROUP_FAMILIES[group]
    family_by_feature = dict(zip(family_manifest["feature"], family_manifest["feature_family"]))
    selected = [feature for feature in feature_columns if family_by_feature.get(feature, "raw") in allowed]
    # Keep a minimal raw return anchor even if a sparse group is requested.
    if "bucket_return_bps" in feature_columns and "bucket_return_bps" not in selected:
        selected.insert(0, "bucket_return_bps")
    return selected or feature_columns


def infer_column(frame: pd.DataFrame, choices: list[str], label: str, required: bool = True) -> str:
    for column in choices:
        if column in frame.columns:
            return column
    if required:
        raise SystemExit(f"Could not find {label} column. Tried: {choices}")
    return ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def estimate_bucket_seconds(frame: pd.DataFrame) -> float:
    diffs = pd.to_numeric(frame["decision_timestamp"], errors="coerce").diff().dropna().to_numpy(dtype=np.float64)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if len(diffs) == 0:
        return 10.0
    return float(np.median(diffs) / 1000.0)


def load_source(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Source path does not exist: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if MAX_ROWS_ENV:
        max_rows = int(float(MAX_ROWS_ENV))
        if max_rows > 0:
            frame = frame.tail(max_rows).copy()

    timestamp_col = infer_column(frame, ["timestamp", "time_ms", "ts"], "timestamp")
    close_col = infer_column(frame, ["close", "price", "mid_price", "last"], "price")
    open_col = infer_column(frame, ["open"], "open", required=False)
    high_col = infer_column(frame, ["high"], "high", required=False)
    low_col = infer_column(frame, ["low"], "low", required=False)
    bid_col = infer_column(frame, ["best_bid", "bid"], "bid", required=False)
    ask_col = infer_column(frame, ["best_ask", "ask"], "ask", required=False)
    volume_col = infer_column(frame, ["volume", "total_trade_volume_10s", "trade_volume"], "volume", required=False)
    buy_volume_col = infer_column(frame, ["market_buy_volume_10s", "buy_volume"], "buy volume", required=False)
    sell_volume_col = infer_column(frame, ["market_sell_volume_10s", "sell_volume"], "sell volume", required=False)
    trade_count_col = infer_column(frame, ["trade_count_10s", "trade_count"], "trade count", required=False)

    close = pd.to_numeric(frame[close_col], errors="coerce")
    open_values = pd.to_numeric(frame[open_col], errors="coerce") if open_col else close.shift(1).fillna(close)
    if high_col:
        high = pd.to_numeric(frame[high_col], errors="coerce")
    elif ask_col:
        high = pd.concat([close, pd.to_numeric(frame[ask_col], errors="coerce")], axis=1).max(axis=1)
    else:
        high = close
    if low_col:
        low = pd.to_numeric(frame[low_col], errors="coerce")
    elif bid_col:
        low = pd.concat([close, pd.to_numeric(frame[bid_col], errors="coerce")], axis=1).min(axis=1)
    else:
        low = close
    volume = pd.to_numeric(frame[volume_col], errors="coerce") if volume_col else pd.Series(0.0, index=frame.index)
    buy_volume = pd.to_numeric(frame[buy_volume_col], errors="coerce") if buy_volume_col else pd.Series(np.nan, index=frame.index)
    sell_volume = pd.to_numeric(frame[sell_volume_col], errors="coerce") if sell_volume_col else pd.Series(np.nan, index=frame.index)
    trade_count = pd.to_numeric(frame[trade_count_col], errors="coerce") if trade_count_col else pd.Series(np.nan, index=frame.index)

    out = pd.DataFrame(
        {
            "decision_timestamp": pd.to_numeric(frame[timestamp_col], errors="coerce"),
            "open": open_values,
            "high": high,
            "low": low,
            "close": close,
            "price": close,
            "volume": volume,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "trade_count": trade_count,
        }
    )
    if INCLUDE_FLOW_FEATURES:
        for column in [
            "spread_percent",
            "market_pressure_10s",
            "bid_depth_10bps",
            "ask_depth_10bps",
            "bid_depth_25bps",
            "ask_depth_25bps",
            "order_book_imbalance_10bps",
            "order_book_imbalance_25bps",
        ]:
            if column in frame.columns:
                out[column] = pd.to_numeric(frame[column], errors="coerce")
    out = out.dropna(subset=["decision_timestamp", "close"]).copy()
    out = out[(out["close"] > 0.0) & (out["high"] > 0.0) & (out["low"] > 0.0)]
    out = out.sort_values("decision_timestamp").drop_duplicates("decision_timestamp").reset_index(drop=True)
    out["decision_time_iso"] = out["decision_timestamp"].apply(iso_from_ms)

    source_meta = {
        "timestamp_column": timestamp_col,
        "close_column": close_col,
        "open_column": open_col or "previous_close_fallback",
        "high_column": high_col or ("ask_plus_close_fallback" if ask_col else "close_fallback"),
        "low_column": low_col or ("bid_plus_close_fallback" if bid_col else "close_fallback"),
        "volume_column": volume_col or "zero_fallback",
        "buy_volume_column": buy_volume_col,
        "sell_volume_column": sell_volume_col,
        "trade_count_column": trade_count_col,
    }
    return out, source_meta


def load_cross_market_source(path_text: str, prefix: str) -> pd.DataFrame:
    if not path_text:
        return pd.DataFrame()
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    timestamp_col = infer_column(frame, ["timestamp", "time_ms", "ts"], f"{prefix} timestamp")
    close_col = infer_column(frame, ["close", "price", "mid_price", "last"], f"{prefix} price")
    out = pd.DataFrame(
        {
            "decision_timestamp": pd.to_numeric(frame[timestamp_col], errors="coerce"),
            f"{prefix.lower()}_close": pd.to_numeric(frame[close_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["decision_timestamp", f"{prefix.lower()}_close"])
    out = out[out[f"{prefix.lower()}_close"] > 0.0]
    return out.sort_values("decision_timestamp").drop_duplicates("decision_timestamp").reset_index(drop=True)


def add_cross_market_columns(out: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base = out.sort_values("decision_timestamp").copy()
    added: list[str] = []
    for symbol, source_path in CROSS_MARKET_SOURCES.items():
        prefix = symbol.lower()
        cross = load_cross_market_source(source_path, symbol)
        if cross.empty:
            continue
        base = pd.merge_asof(
            base.sort_values("decision_timestamp"),
            cross.sort_values("decision_timestamp"),
            on="decision_timestamp",
            direction="backward",
            tolerance=None,
        )
        close_col = f"{prefix}_close"
        ret_col = f"{prefix}_return_bps"
        if close_col in base.columns:
            base[ret_col] = log_bps(base[close_col], base[close_col].shift(1)).fillna(0.0)
            added.append(ret_col)
            if prefix == "btc":
                base["sol_btc_relative_return_bps"] = base["bucket_return_bps"] - base[ret_col]
                added.append("sol_btc_relative_return_bps")
            if prefix == "eth":
                base["sol_eth_relative_return_bps"] = base["bucket_return_bps"] - base[ret_col]
                added.append("sol_eth_relative_return_bps")
    if {"btc_return_bps", "eth_return_bps"}.issubset(base.columns):
        base["btc_eth_relative_return_bps"] = base["btc_return_bps"] - base["eth_return_bps"]
        added.append("btc_eth_relative_return_bps")
    return base, list(dict.fromkeys(added))


def log_bps(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    values = 10_000.0 * np.log(pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(denominator, errors="coerce"))
    return pd.Series(values, index=numerator.index).replace([np.inf, -np.inf], np.nan)


def rolling_rsi(return_bps: pd.Series, window: int) -> pd.Series:
    gains = return_bps.clip(lower=0.0)
    losses = (-return_bps.clip(upper=0.0))
    avg_gain = gains.rolling(window, min_periods=window).mean()
    avg_loss = losses.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    both_flat = avg_gain.fillna(0.0).abs().lt(1e-12) & avg_loss.fillna(0.0).abs().lt(1e-12)
    rsi = rsi.where(~both_flat, 50.0)
    rsi = rsi.where(~(avg_loss.fillna(0.0).abs().lt(1e-12) & avg_gain.gt(0.0)), 100.0)
    return rsi


def add_feature_bank(table: pd.DataFrame, windows: list[int]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    out = table.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    open_values = pd.to_numeric(out["open"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    prev_close = close.shift(1)

    out["bucket_return_bps"] = log_bps(close, prev_close).fillna(0.0)
    out["log_return_bps"] = out["bucket_return_bps"]
    out["close_return_fraction"] = np.exp(out["bucket_return_bps"] / 10_000.0) - 1.0
    out["abs_bucket_return_bps"] = out["bucket_return_bps"].abs()
    out["raw_range_bps"] = log_bps(high, low).clip(lower=0.0)
    out["raw_body_bps"] = log_bps(close, open_values)
    out["upper_wick_bps"] = log_bps(high, pd.concat([open_values, close], axis=1).max(axis=1)).clip(lower=0.0)
    out["lower_wick_bps"] = log_bps(pd.concat([open_values, close], axis=1).min(axis=1), low).clip(lower=0.0)
    out["upper_wick_to_range"] = out["upper_wick_bps"] / out["raw_range_bps"].replace(0.0, np.nan)
    out["lower_wick_to_range"] = out["lower_wick_bps"] / out["raw_range_bps"].replace(0.0, np.nan)
    out["body_to_range"] = out["raw_body_bps"].abs() / out["raw_range_bps"].replace(0.0, np.nan)
    out["volume"] = volume
    out["trade_count"] = pd.to_numeric(out["trade_count"], errors="coerce")
    out["buy_sell_volume_imbalance"] = (
        (pd.to_numeric(out["buy_volume"], errors="coerce") - pd.to_numeric(out["sell_volume"], errors="coerce"))
        / volume.replace(0.0, np.nan)
    )
    out["buy_sell_volume_imbalance"] = out["buy_sell_volume_imbalance"].replace([np.inf, -np.inf], np.nan)

    out, cross_features = add_cross_market_columns(out)
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)

    timestamp_dt = pd.to_datetime(out["decision_timestamp"], unit="ms", utc=True, errors="coerce")
    seconds_in_day = (
        timestamp_dt.dt.hour.fillna(0) * 3600
        + timestamp_dt.dt.minute.fillna(0) * 60
        + timestamp_dt.dt.second.fillna(0)
    )
    day_angle = 2.0 * math.pi * seconds_in_day / 86_400.0
    out["time_of_day_sin"] = np.sin(day_angle)
    out["time_of_day_cos"] = np.cos(day_angle)
    week_angle = 2.0 * math.pi * timestamp_dt.dt.dayofweek.fillna(0) / 7.0
    out["day_of_week_sin"] = np.sin(week_angle)
    out["day_of_week_cos"] = np.cos(week_angle)

    feature_columns = [
        "bucket_return_bps",
        "log_return_bps",
        "close_return_fraction",
        "abs_bucket_return_bps",
        "raw_range_bps",
        "raw_body_bps",
        "upper_wick_bps",
        "lower_wick_bps",
        "upper_wick_to_range",
        "lower_wick_to_range",
        "body_to_range",
        "volume",
        "trade_count",
        "buy_sell_volume_imbalance",
        "time_of_day_sin",
        "time_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ]
    feature_columns.extend(cross_features)
    for column in [
        "spread_percent",
        "market_pressure_10s",
        "bid_depth_10bps",
        "ask_depth_10bps",
        "bid_depth_25bps",
        "ask_depth_25bps",
        "order_book_imbalance_10bps",
        "order_book_imbalance_25bps",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
            feature_columns.append(column)

    true_range = pd.concat(
        [
            log_bps(high, low).abs(),
            log_bps(high, prev_close).abs(),
            log_bps(low, prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    typical_price = (high + low + close) / 3.0
    obv = (np.sign(out["bucket_return_bps"].to_numpy(dtype=np.float64)) * volume.to_numpy(dtype=np.float64)).cumsum()
    out["obv"] = obv
    feature_columns.append("obv")

    for window in sorted(set(windows)):
        rolling_high = high.rolling(window, min_periods=window).max()
        rolling_low = low.rolling(window, min_periods=window).min()
        rolling_close_high = close.rolling(window, min_periods=window).max()
        rolling_close_low = close.rolling(window, min_periods=window).min()
        sma = close.rolling(window, min_periods=window).mean()
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        price_std = close.rolling(window, min_periods=window).std(ddof=0)
        typical_sma = typical_price.rolling(window, min_periods=window).mean()
        typical_mad = (typical_price - typical_sma).abs().rolling(window, min_periods=window).mean()
        return_mean = out["bucket_return_bps"].rolling(window, min_periods=window).mean()
        return_std = out["bucket_return_bps"].rolling(window, min_periods=window).std(ddof=0)
        volume_mean = volume.rolling(window, min_periods=window).mean()
        volume_std = volume.rolling(window, min_periods=window).std(ddof=0)
        vwap = (close * volume).rolling(window, min_periods=window).sum() / volume.rolling(window, min_periods=window).sum().replace(0.0, np.nan)
        positive_flow = (typical_price.where(typical_price > typical_price.shift(1), 0.0) * volume).rolling(window, min_periods=window).sum()
        negative_flow = (typical_price.where(typical_price < typical_price.shift(1), 0.0) * volume).rolling(window, min_periods=window).sum()
        money_ratio = positive_flow / negative_flow.replace(0.0, np.nan)
        bb_upper = sma + 2.0 * price_std
        bb_lower = sma - 2.0 * price_std
        channel_width = (rolling_high - rolling_low).replace(0.0, np.nan)
        bb_width_denominator = bb_lower.where(bb_lower > 0.0)
        rolling_range_bps = log_bps(rolling_high, rolling_low).clip(lower=0.0)
        parkinson_vol_bps = np.sqrt(
            out["raw_range_bps"].pow(2).rolling(window, min_periods=window).mean() / (4.0 * math.log(2.0))
        )
        volatility_regime = return_std / return_std.rolling(max(window * 4, window + 1), min_periods=window).median().replace(0.0, np.nan)
        compression_expansion = rolling_range_bps / rolling_range_bps.rolling(max(window * 4, window + 1), min_periods=window).mean().replace(0.0, np.nan)
        feature_map = {
            f"rolling_mean_return_bps_fw{window}": return_mean,
            f"sma_distance_bps_fw{window}": log_bps(close, sma),
            f"ema_distance_bps_fw{window}": log_bps(close, ema),
            f"sma_slope_bps_fw{window}": log_bps(sma, sma.shift(window)),
            f"ema_slope_bps_fw{window}": log_bps(ema, ema.shift(window)),
            f"roc_bps_fw{window}": log_bps(close, close.shift(window)),
            f"rolling_range_bps_fw{window}": rolling_range_bps,
            f"rolling_volatility_bps_fw{window}": return_std,
            f"realized_volatility_bps_fw{window}": return_std * math.sqrt(window),
            f"parkinson_volatility_bps_fw{window}": parkinson_vol_bps,
            f"distance_to_recent_high_bps_fw{window}": log_bps(close, rolling_close_high),
            f"distance_to_recent_low_bps_fw{window}": log_bps(close, rolling_close_low),
            f"donchian_position_fw{window}": (close - rolling_low) / channel_width,
            f"stochastic_k_fw{window}": 100.0 * (close - rolling_low) / channel_width,
            f"williams_r_fw{window}": -100.0 * (rolling_high - close) / channel_width,
            f"cci_fw{window}": (typical_price - typical_sma) / (0.015 * typical_mad.replace(0.0, np.nan)),
            f"rsi_fw{window}": rolling_rsi(out["bucket_return_bps"], window),
            f"atr_bps_fw{window}": true_range.rolling(window, min_periods=window).mean(),
            f"bollinger_width_bps_fw{window}": log_bps(bb_upper, bb_width_denominator),
            f"bollinger_percent_b_fw{window}": (close - bb_lower) / (bb_upper - bb_lower).replace(0.0, np.nan),
            f"rolling_mean_zscore_fw{window}": (close - sma) / price_std.replace(0.0, np.nan),
            f"volume_zscore_fw{window}": (volume - volume_mean) / volume_std.replace(0.0, np.nan),
            f"obv_roc_fw{window}": pd.Series(obv, index=out.index).diff(window),
            f"mfi_fw{window}": 100.0 - (100.0 / (1.0 + money_ratio)),
            f"vwap_distance_bps_fw{window}": log_bps(close, vwap),
            f"volatility_regime_fw{window}": volatility_regime,
            f"trend_regime_bps_fw{window}": log_bps(ema, ema.shift(window)),
            f"compression_expansion_fw{window}": compression_expansion,
        }
        for cross_return in ["btc_return_bps", "eth_return_bps"]:
            if cross_return in out.columns:
                prefix = cross_return.replace("_return_bps", "")
                feature_map[f"{prefix}_volatility_bps_fw{window}"] = (
                    pd.to_numeric(out[cross_return], errors="coerce").rolling(window, min_periods=window).std(ddof=0)
                )
        for name, values in feature_map.items():
            out[name] = pd.Series(values, index=out.index).replace([np.inf, -np.inf], np.nan)
            feature_columns.append(name)

    macd_fast, macd_slow, macd_signal = MACD_SPEC
    ema_fast = close.ewm(span=macd_fast, adjust=False, min_periods=macd_fast).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False, min_periods=macd_slow).mean()
    out[f"macd_bps_{macd_fast}_{macd_slow}"] = log_bps(ema_fast, ema_slow)
    out[f"macd_signal_bps_{macd_fast}_{macd_slow}_{macd_signal}"] = out[f"macd_bps_{macd_fast}_{macd_slow}"].ewm(
        span=macd_signal,
        adjust=False,
        min_periods=macd_signal,
    ).mean()
    out[f"macd_hist_bps_{macd_fast}_{macd_slow}_{macd_signal}"] = (
        out[f"macd_bps_{macd_fast}_{macd_slow}"]
        - out[f"macd_signal_bps_{macd_fast}_{macd_slow}_{macd_signal}"]
    )
    feature_columns += [
        f"macd_bps_{macd_fast}_{macd_slow}",
        f"macd_signal_bps_{macd_fast}_{macd_slow}_{macd_signal}",
        f"macd_hist_bps_{macd_fast}_{macd_slow}_{macd_signal}",
    ]

    feature_columns = list(dict.fromkeys(feature_columns))
    audit_rows = []
    missing_indicators: dict[str, pd.Series] = {}
    for column in feature_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing_column = f"{column}_missing"
        missing_indicators[missing_column] = out[column].isna().astype(int)
        values = out[column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        audit_rows.append(
            {
                "feature": column,
                "finite_rows": int(len(finite)),
                "missing_rows": int(len(values) - len(finite)),
                "minimum": float(np.min(finite)) if len(finite) else math.nan,
                "maximum": float(np.max(finite)) if len(finite) else math.nan,
                "mean": float(np.mean(finite)) if len(finite) else math.nan,
                "standard_deviation": float(np.std(finite)) if len(finite) else math.nan,
                "causal_inputs_only": True,
            }
        )
    if missing_indicators:
        out = pd.concat([out, pd.DataFrame(missing_indicators, index=out.index)], axis=1)
    return out, feature_columns, pd.DataFrame(audit_rows)


def future_extreme(price: pd.Series, offset: int, kind: str) -> pd.Series:
    shifted = price.shift(-1)
    reversed_shifted = shifted.iloc[::-1]
    if kind == "max":
        rolled = reversed_shifted.rolling(offset, min_periods=offset).max()
    elif kind == "min":
        rolled = reversed_shifted.rolling(offset, min_periods=offset).min()
    else:
        raise ValueError(kind)
    return rolled.iloc[::-1].reset_index(drop=True)


def add_targets(table: pd.DataFrame, horizons: list[int]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    out = table.copy()
    price = pd.to_numeric(out["close"], errors="coerce")
    rows = []
    target_columns = []
    for horizon in sorted(set(horizons)):
        future_price = price.shift(-horizon)
        future_high = future_extreme(price, horizon, "max")
        future_low = future_extreme(price, horizon, "min")
        target = f"future_return_bps_h{horizon}"
        mfe = f"mfe_bps_h{horizon}"
        mae = f"mae_bps_h{horizon}"
        label_ts = f"label_end_timestamp_h{horizon}"
        out[target] = log_bps(future_price, price)
        out[mfe] = log_bps(future_high, price).clip(lower=0.0)
        out[mae] = log_bps(future_low, price).clip(upper=0.0)
        out[label_ts] = out["decision_timestamp"].shift(-horizon)
        target_columns.append(target)
        rows.append(
            {
                "target_column": target,
                "horizon_buckets": horizon,
                "label_end_timestamp_column": label_ts,
                "mfe_column": mfe,
                "mae_column": mae,
                "target_type": "market_relative_cumulative_future_return_bps",
            }
        )
    out["label_end_timestamp"] = out[f"label_end_timestamp_h{max(horizons)}"]
    return out.replace([np.inf, -np.inf], np.nan), target_columns, pd.DataFrame(rows)


def assign_splits(table: pd.DataFrame, max_lookahead_rows: int, max_feature_window: int) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    n = len(table)
    purge_rows = int(PURGE_ROWS_ENV) if PURGE_ROWS_ENV else max(1, max_feature_window)
    embargo_rows = int(EMBARGO_ROWS_ENV) if EMBARGO_ROWS_ENV else max(1, max_lookahead_rows)
    holdout_start_raw = int(n * (1.0 - HOLDOUT_FRACTION))
    validation_start_raw = int(n * (1.0 - HOLDOUT_FRACTION - VALIDATION_FRACTION))
    validation_start_raw = max(1, min(validation_start_raw, n - 2))
    holdout_start_raw = max(validation_start_raw + 1, min(holdout_start_raw, n - 1))
    train_start = 0
    train_end = max(train_start, validation_start_raw - embargo_rows)
    validation_start = min(n, validation_start_raw + purge_rows)
    validation_end = max(validation_start, holdout_start_raw - embargo_rows)
    holdout_start = min(n, holdout_start_raw + purge_rows)
    holdout_end = n

    out = table.copy()
    out["split"] = "purge_embargo"
    out.loc[train_start: max(train_end - 1, train_start - 1), "split"] = "train"
    if validation_start < validation_end:
        out.loc[validation_start: validation_end - 1, "split"] = "validation"
    if holdout_start < holdout_end:
        out.loc[holdout_start: holdout_end - 1, "split"] = "untouched_holdout"

    split_rows = []
    for split_name in ["train", "validation", "untouched_holdout", "purge_embargo"]:
        subset = out[out["split"].eq(split_name)]
        split_rows.append(
            {
                "split": split_name,
                "rows": int(len(subset)),
                "start_timestamp": float(subset["decision_timestamp"].min()) if not subset.empty else math.nan,
                "end_timestamp": float(subset["decision_timestamp"].max()) if not subset.empty else math.nan,
                "start_iso": iso_from_ms(subset["decision_timestamp"].min()) if not subset.empty else "",
                "end_iso": iso_from_ms(subset["decision_timestamp"].max()) if not subset.empty else "",
            }
        )
    split_frame = pd.DataFrame(split_rows)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": SYMBOL,
        "venue": VENUE,
        "instrument": INSTRUMENT,
        "validation_fraction": VALIDATION_FRACTION,
        "holdout_fraction": HOLDOUT_FRACTION,
        "maximum_label_lookahead_rows": int(max_lookahead_rows),
        "maximum_feature_window_rows": int(max_feature_window),
        "purge_rows": int(purge_rows),
        "embargo_rows": int(embargo_rows),
        "raw_boundaries": {
            "validation_start_row": validation_start_raw,
            "holdout_start_row": holdout_start_raw,
            "validation_start_timestamp": iso_from_ms(table["decision_timestamp"].iloc[validation_start_raw])
            if validation_start_raw < n
            else "",
            "holdout_start_timestamp": iso_from_ms(table["decision_timestamp"].iloc[holdout_start_raw])
            if holdout_start_raw < n
            else "",
        },
        "effective_boundaries": {
            "train_start_row": train_start,
            "train_end_row": train_end,
            "validation_start_row": validation_start,
            "validation_end_row": validation_end,
            "holdout_start_row": holdout_start,
            "holdout_end_row": holdout_end,
        },
        "effective_sample_counts": {row["split"]: row["rows"] for row in split_rows},
        "split_rows": split_rows,
        "model_selection_stage": "validation_selects_model_threshold_policy",
        "holdout_usage": "untouched_holdout_final_evaluation_only",
    }
    return out, manifest, split_frame


def fit_preprocessor(train: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    values = train[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    means = np.nanmean(values, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    filled = np.where(np.isfinite(values), values, means)
    filled = np.where(np.isfinite(filled), filled, means)
    std = np.nanstd(filled, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return {"mean": means, "std": std}


def transform_features(frame: pd.DataFrame, columns: list[str], preprocessor: dict[str, np.ndarray]) -> np.ndarray:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    values = np.where(np.isfinite(values), values, preprocessor["mean"])
    out = (values - preprocessor["mean"]) / preprocessor["std"]
    out[~np.isfinite(out)] = 0.0
    return out


def target_matrix(frame: pd.DataFrame, target_columns: list[str]) -> np.ndarray:
    return frame[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


def fit_ridge_multi_output(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(X), dtype=np.float64), X])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ Y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def predict_ridge(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X), dtype=np.float64), X]) @ coef


def fit_elastic_net_multi_output(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(X), dtype=np.float64), X])
    coef = np.zeros((design.shape[1], Y.shape[1]), dtype=np.float64)
    y_scale = np.nanstd(Y, axis=0)
    y_scale = np.where(np.isfinite(y_scale) & (y_scale > 1e-12), y_scale, 1.0)
    y_scaled = np.where(np.isfinite(Y), Y, 0.0) / y_scale
    n = max(1, len(design))
    l1 = max(0.0, min(1.0, ELASTIC_NET_L1_RATIO))
    l2 = 1.0 - l1
    for _ in range(max(1, ELASTIC_NET_ITERATIONS)):
        pred = design @ coef
        err = pred - y_scaled
        grad = design.T @ err / n
        grad[1:] += alpha * l2 * coef[1:]
        grad[1:] += alpha * l1 * np.sign(coef[1:])
        coef -= ELASTIC_NET_LR * grad
    coef = coef * y_scale
    return coef


def rmse_matrix(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return math.nan
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2)))


def fit_tanh_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    all_x: np.ndarray,
    hidden_units: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    y_mean = np.nanmean(train_y, axis=0)
    y_std = np.nanstd(train_y, axis=0)
    y_mean = np.where(np.isfinite(y_mean), y_mean, 0.0)
    y_std = np.where(np.isfinite(y_std) & (y_std > 1e-12), y_std, 1.0)
    Y = (train_y - y_mean) / y_std
    Y = np.where(np.isfinite(Y), Y, 0.0)
    scale1 = 1.0 / math.sqrt(max(1, train_x.shape[1]))
    scale2 = 1.0 / math.sqrt(max(1, hidden_units))
    w1 = rng.normal(0.0, scale1, size=(train_x.shape[1], hidden_units))
    b1 = np.zeros(hidden_units, dtype=np.float64)
    w2 = rng.normal(0.0, scale2, size=(hidden_units, train_y.shape[1]))
    b2 = np.zeros(train_y.shape[1], dtype=np.float64)
    n = max(1, len(train_x))
    for _ in range(max(1, epochs)):
        hidden = np.tanh(train_x @ w1 + b1)
        pred = hidden @ w2 + b2
        err = (pred - Y) / n
        grad_w2 = hidden.T @ err
        grad_b2 = err.sum(axis=0)
        grad_hidden = (err @ w2.T) * (1.0 - hidden * hidden)
        grad_w1 = train_x.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2
        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
    pred_all = np.tanh(all_x @ w1 + b1) @ w2 + b2
    return pred_all * y_std + y_mean, {
        "model_type": "numpy_tanh_mlp",
        "hidden_units": hidden_units,
        "epochs": epochs,
        "seed": seed,
    }


def fit_mlp(train_x: np.ndarray, train_y: np.ndarray, all_x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    return fit_tanh_mlp(train_x, train_y, all_x, MLP_HIDDEN, MLP_EPOCHS, MLP_LR, MLP_SEED)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def fit_logistic_direction_model(X_train: np.ndarray, Y_train: np.ndarray, X_all: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    design_train = np.column_stack([np.ones(len(X_train), dtype=np.float64), X_train])
    design_all = np.column_stack([np.ones(len(X_all), dtype=np.float64), X_all])
    out = np.zeros((len(X_all), Y_train.shape[1]), dtype=np.float64)
    for horizon_idx in range(Y_train.shape[1]):
        y = (Y_train[:, horizon_idx] > 0.0).astype(np.float64)
        coef = np.zeros(design_train.shape[1], dtype=np.float64)
        if len(np.unique(y)) >= 2:
            for _ in range(120):
                pred = sigmoid(design_train @ coef)
                grad = design_train.T @ (pred - y) / max(1, len(y))
                coef -= 0.05 * grad
        else:
            base = np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)
            coef[0] = math.log(base / (1.0 - base))
        prob = sigmoid(design_all @ coef)
        scale = safe_float(np.nanmedian(np.abs(Y_train[:, horizon_idx])), 1.0)
        out[:, horizon_idx] = (prob - 0.5) * 2.0 * max(scale, 1e-6)
    return out, {"model_type": "logistic_direction_model", "selection_stage": "train_fit_direction"}


def fit_decision_stump_multi_output(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_all: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    feature_limit = min(TREE_MAX_FEATURES, X_train.shape[1])
    best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
    for feature_idx in range(feature_limit):
        values = X_train[:, feature_idx]
        threshold = safe_float(np.nanmedian(values))
        if not math.isfinite(threshold):
            continue
        left = values <= threshold
        right = ~left
        if left.sum() < 5 or right.sum() < 5:
            continue
        left_mean = np.nanmean(Y_train[left], axis=0)
        right_mean = np.nanmean(Y_train[right], axis=0)
        pred_val = np.where((X_val[:, [feature_idx]] <= threshold), left_mean, right_mean)
        score = rmse_matrix(Y_val, pred_val)
        if best is None or (math.isfinite(score) and score < best[0]):
            best = (score, feature_idx, threshold, left_mean, right_mean)
    if best is None:
        mean_pred = np.nanmean(Y_train, axis=0)
        return np.tile(mean_pred, (len(X_all), 1)), {"model_type": "decision_stump", "status": "fallback_train_mean"}
    score, feature_idx, threshold, left_mean, right_mean = best
    pred_all = np.where((X_all[:, [feature_idx]] <= threshold), left_mean, right_mean)
    return pred_all, {
        "model_type": "decision_stump",
        "selected_feature": feature_names[feature_idx] if feature_idx < len(feature_names) else str(feature_idx),
        "selected_threshold": threshold,
        "validation_combined_rmse": score,
    }


def best_residual_stump(
    X_train: np.ndarray,
    residual_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    pred_val: np.ndarray,
) -> tuple[float, int, float, np.ndarray, np.ndarray] | None:
    feature_limit = min(TREE_MAX_FEATURES, X_train.shape[1])
    best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
    for feature_idx in range(feature_limit):
        threshold = safe_float(np.nanmedian(X_train[:, feature_idx]))
        if not math.isfinite(threshold):
            continue
        left = X_train[:, feature_idx] <= threshold
        right = ~left
        if left.sum() < 5 or right.sum() < 5:
            continue
        left_mean = np.nanmean(residual_train[left], axis=0)
        right_mean = np.nanmean(residual_train[right], axis=0)
        update_val = np.where((X_val[:, [feature_idx]] <= threshold), left_mean, right_mean)
        score = rmse_matrix(Y_val, pred_val + BOOSTED_TREE_LEARNING_RATE * update_val)
        if best is None or (math.isfinite(score) and score < best[0]):
            best = (score, feature_idx, threshold, left_mean, right_mean)
    return best


def fit_boosted_stumps_multi_output(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_all: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    base = np.nanmean(Y_train, axis=0)
    base = np.where(np.isfinite(base), base, 0.0)
    pred_train = np.tile(base, (len(X_train), 1))
    pred_val = np.tile(base, (len(X_val), 1))
    pred_all = np.tile(base, (len(X_all), 1))
    rounds: list[dict[str, Any]] = []
    best_score = rmse_matrix(Y_val, pred_val)
    for round_idx in range(max(1, BOOSTED_TREE_ROUNDS)):
        residual_train = Y_train - pred_train
        stump = best_residual_stump(X_train, residual_train, X_val, Y_val, pred_val)
        if stump is None:
            break
        score, feature_idx, threshold, left_mean, right_mean = stump
        update_train = np.where((X_train[:, [feature_idx]] <= threshold), left_mean, right_mean)
        update_val = np.where((X_val[:, [feature_idx]] <= threshold), left_mean, right_mean)
        update_all = np.where((X_all[:, [feature_idx]] <= threshold), left_mean, right_mean)
        pred_train += BOOSTED_TREE_LEARNING_RATE * update_train
        pred_val += BOOSTED_TREE_LEARNING_RATE * update_val
        pred_all += BOOSTED_TREE_LEARNING_RATE * update_all
        best_score = score
        rounds.append(
            {
                "round": round_idx,
                "feature": feature_names[feature_idx] if feature_idx < len(feature_names) else str(feature_idx),
                "threshold": threshold,
                "validation_rmse": score,
            }
        )
    return pred_all, {
        "model_type": "boosted_decision_stumps",
        "boosting_rounds_completed": len(rounds),
        "boosting_learning_rate": BOOSTED_TREE_LEARNING_RATE,
        "validation_combined_rmse": best_score,
        "first_boost_feature": rounds[0]["feature"] if rounds else "",
    }


def build_sequence_arrays(
    X_all: np.ndarray,
    Y_all: np.ndarray,
    split_values: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequence_rows = []
    target_rows = []
    row_indices = []
    split_rows = []
    for idx in range(seq_len - 1, len(X_all)):
        seq = X_all[idx - seq_len + 1 : idx + 1]
        target = Y_all[idx]
        if not np.isfinite(target).all():
            continue
        sequence_rows.append(seq.reshape(-1))
        target_rows.append(target)
        row_indices.append(idx)
        split_rows.append(split_values[idx])
    if not sequence_rows:
        return (
            np.empty((0, seq_len * X_all.shape[1]), dtype=np.float64),
            np.empty((0, Y_all.shape[1]), dtype=np.float64),
            np.asarray([], dtype=int),
            np.asarray([], dtype=object),
        )
    return (
        np.asarray(sequence_rows, dtype=np.float64),
        np.asarray(target_rows, dtype=np.float64),
        np.asarray(row_indices, dtype=int),
        np.asarray(split_rows, dtype=object),
    )


def build_sequence_tensor_arrays(
    X_all: np.ndarray,
    Y_all: np.ndarray,
    split_values: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat_x, seq_y, row_indices, seq_splits = build_sequence_arrays(X_all, Y_all, split_values, seq_len)
    feature_count = int(X_all.shape[1]) if X_all.ndim == 2 else 0
    if len(flat_x) == 0:
        seq_x = np.empty((0, seq_len, feature_count), dtype=np.float64)
    else:
        seq_x = flat_x.reshape(len(flat_x), seq_len, feature_count)
    return seq_x, seq_y, row_indices, seq_splits


def safe_slug(value: str, max_len: int = 36) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip().lower()).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return (cleaned or "x")[:max_len]


def export_sequence_datasets(
    table: pd.DataFrame,
    feature_columns: list[str],
    family_manifest: pd.DataFrame,
    target_columns: list[str],
    run_dir: Path,
    group_specs: list[str],
    sequence_lens: list[int] | None = None,
    write_arrays: bool | None = None,
    max_features: int | None = None,
) -> pd.DataFrame:
    sequence_lens = sequence_lens if sequence_lens is not None else SEQUENCE_LENS
    write_arrays = WRITE_SEQUENCE_DATASETS if write_arrays is None else write_arrays
    max_features = SEQUENCE_DATASET_MAX_FEATURES if max_features is None else max_features
    dataset_dir = run_dir / "sequence_datasets"
    if write_arrays:
        dataset_dir.mkdir(parents=True, exist_ok=True)
    train = table[table["split"].astype(str).eq("train")]
    y_all = target_matrix(table, target_columns)
    split_values = table["split"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for group_spec in list(dict.fromkeys(group_specs)):
        selected_features = selected_feature_columns_for_group(feature_columns, family_manifest, group_spec)
        limited_features = selected_features[:max_features] if max_features and max_features > 0 else selected_features
        if not limited_features:
            for seq_len in sequence_lens:
                rows.append(
                    {
                        "feature_group": group_spec,
                        "seq_len": seq_len,
                        "status": "no_features",
                        "arrays_written": False,
                        "path_npz": "",
                    }
                )
            continue
        preprocessor = fit_preprocessor(train, limited_features)
        x_all = transform_features(table, limited_features, preprocessor)
        feature_schema_hash = stable_hash({"feature_group": group_spec, "features": limited_features})
        for seq_len in sequence_lens:
            seq_x, seq_y, row_indices, seq_splits = build_sequence_tensor_arrays(
                x_all,
                y_all,
                split_values,
                seq_len,
            )
            split_counts = {str(split): int((seq_splits == split).sum()) for split in sorted(set(seq_splits))}
            causal_ok = bool(len(row_indices) == 0 or np.all(row_indices >= (seq_len - 1)))
            dataset_hash = stable_hash(
                {
                    "feature_group": group_spec,
                    "seq_len": seq_len,
                    "feature_schema_hash": feature_schema_hash,
                    "targets": target_columns,
                }
            )[:8]
            filename = f"seq{seq_len}_{safe_slug(group_spec)}_{dataset_hash}.npz"
            path_npz = dataset_dir / filename
            arrays_written = False
            if write_arrays:
                timestamps = table.loc[row_indices, "decision_timestamp"].to_numpy(dtype=np.float64) if len(row_indices) else np.asarray([], dtype=np.float64)
                np.savez_compressed(
                    path_npz,
                    X=seq_x,
                    y=seq_y,
                    row_indices=row_indices,
                    splits=seq_splits.astype(str),
                    decision_timestamps=timestamps,
                    feature_columns=np.asarray(limited_features, dtype=str),
                    target_columns=np.asarray(target_columns, dtype=str),
                    seq_len=np.asarray([seq_len], dtype=int),
                    horizon_buckets=np.asarray(HORIZON_BUCKETS, dtype=int),
                )
                arrays_written = True
            rows.append(
                {
                    "feature_group": group_spec,
                    "seq_len": seq_len,
                    "status": "ok" if causal_ok else "causal_guard_failed",
                    "sequence_rows": int(len(seq_x)),
                    "train_sequence_rows": int(split_counts.get("train", 0)),
                    "validation_sequence_rows": int(split_counts.get("validation", 0)),
                    "holdout_sequence_rows": int(split_counts.get("untouched_holdout", 0)),
                    "feature_count": int(len(limited_features)),
                    "requested_feature_count": int(len(selected_features)),
                    "max_features": int(max_features or len(selected_features)),
                    "horizon_count": int(len(target_columns)),
                    "x_shape": json.dumps(list(seq_x.shape)),
                    "y_shape": json.dumps(list(seq_y.shape)),
                    "feature_schema_hash": feature_schema_hash,
                    "dataset_hash": dataset_hash,
                    "causal_sequence_status": "ok" if causal_ok else "failed",
                    "arrays_written": arrays_written,
                    "path_npz": str(path_npz) if arrays_written else "",
                    "sequence_input_shape": "[batch, seq_len, feature_count]",
                    "target_shape": "[batch, horizon_count]",
                }
            )
    return pd.DataFrame(rows)


def torch_status() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"torch_available": False, "cuda_available": False, "torch_version": ""}
    import torch  # type: ignore

    return {
        "torch_available": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }


def fit_torch_sequence_model(
    seq_x: np.ndarray,
    seq_y: np.ndarray,
    seq_splits: np.ndarray,
    row_indices: np.ndarray,
    full_rows: int,
    horizon_count: int,
    seq_len: int,
    feature_count: int,
    model_kind: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    status = torch_status()
    if not status["torch_available"]:
        raise RuntimeError("torch_unavailable")

    import torch  # type: ignore
    from torch import nn  # type: ignore

    train_mask = seq_splits == "train"
    if int(train_mask.sum()) < 10:
        raise RuntimeError("insufficient_sequence_train_rows")
    x_train = torch.tensor(seq_x[train_mask].reshape(-1, seq_len, feature_count), dtype=torch.float32)
    y_train = torch.tensor(seq_y[train_mask], dtype=torch.float32)
    x_all = torch.tensor(seq_x.reshape(-1, seq_len, feature_count), dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_all = x_all.to(device)

    class TcnModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(feature_count, TORCH_SEQUENCE_HIDDEN, kernel_size=3, padding=2, dilation=1),
                nn.ReLU(),
                nn.Conv1d(TORCH_SEQUENCE_HIDDEN, TORCH_SEQUENCE_HIDDEN, kernel_size=3, padding=4, dilation=2),
                nn.ReLU(),
            )
            self.head = nn.Linear(TORCH_SEQUENCE_HIDDEN, horizon_count)

        def forward(self, x: Any) -> Any:
            y = self.net(x.transpose(1, 2))[..., : x.shape[1]]
            return self.head(y[:, :, -1])

    class GruModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(feature_count, TORCH_SEQUENCE_HIDDEN, batch_first=True)
            self.head = nn.Linear(TORCH_SEQUENCE_HIDDEN, horizon_count)

        def forward(self, x: Any) -> Any:
            y, _ = self.rnn(x)
            return self.head(y[:, -1, :])

    class LstmModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.LSTM(feature_count, TORCH_SEQUENCE_HIDDEN, batch_first=True)
            self.head = nn.Linear(TORCH_SEQUENCE_HIDDEN, horizon_count)

        def forward(self, x: Any) -> Any:
            y, _ = self.rnn(x)
            return self.head(y[:, -1, :])

    class TransformerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(feature_count, TORCH_SEQUENCE_HIDDEN)
            layer = nn.TransformerEncoderLayer(
                d_model=TORCH_SEQUENCE_HIDDEN,
                nhead=4 if TORCH_SEQUENCE_HIDDEN % 4 == 0 else 1,
                dim_feedforward=max(TORCH_SEQUENCE_HIDDEN * 2, 8),
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(TORCH_SEQUENCE_HIDDEN, horizon_count)

        def forward(self, x: Any) -> Any:
            y = self.encoder(self.proj(x))
            return self.head(y[:, -1, :])

    kind = str(model_kind).strip().lower()
    if kind == "tcn":
        model = TcnModel()
    elif kind == "gru":
        model = GruModel()
    elif kind == "lstm":
        model = LstmModel()
    elif kind in {"transformer", "transformer_encoder"}:
        model = TransformerModel()
    else:
        raise RuntimeError(f"unsupported_torch_sequence_model={model_kind}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TORCH_SEQUENCE_LR)
    loss_fn = nn.HuberLoss()
    for _ in range(max(1, TORCH_SEQUENCE_EPOCHS)):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred_seq = model(x_all).detach().cpu().numpy()
    pred_full = np.full((full_rows, horizon_count), np.nan, dtype=np.float64)
    pred_full[row_indices] = pred_seq
    return pred_full, {
        **status,
        "model_type": f"torch_{kind}_sequence_model",
        "device": str(device),
        "seq_len": seq_len,
        "sequence_feature_count": feature_count,
        "epochs": TORCH_SEQUENCE_EPOCHS,
        "loss": "HuberLoss",
    }


def select_signal(values: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(values, dtype=np.float64)
    if INSTRUMENT in {"margin_long_short", "perpetual_long_short", "long_short"}:
        selected = np.abs(pred) > threshold
        direction = np.sign(pred)
    else:
        selected = pred > threshold
        direction = np.ones_like(pred, dtype=np.float64)
    return selected & np.isfinite(pred), direction


def max_dip(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan
    cumulative = np.cumsum(arr)
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def corr(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() < 3:
        return math.nan
    actual = actual[mask]
    pred = pred[mask]
    if np.std(actual) <= 1e-12 or np.std(pred) <= 1e-12:
        return math.nan
    return float(np.corrcoef(actual, pred)[0, 1])


def balanced_accuracy(actual_positive: np.ndarray, pred_positive: np.ndarray) -> float:
    actual_positive = np.asarray(actual_positive, dtype=bool)
    pred_positive = np.asarray(pred_positive, dtype=bool)
    pos = actual_positive
    neg = ~actual_positive
    tpr = float(np.mean(pred_positive[pos])) if pos.any() else math.nan
    tnr = float(np.mean(~pred_positive[neg])) if neg.any() else math.nan
    if math.isnan(tpr) and math.isnan(tnr):
        return math.nan
    if math.isnan(tpr):
        return tnr
    if math.isnan(tnr):
        return tpr
    return 0.5 * (tpr + tnr)


def non_overlapping_values(timestamps: np.ndarray, selected: np.ndarray, net: np.ndarray, horizon_ms: float) -> np.ndarray:
    order = np.argsort(timestamps)
    values = []
    next_allowed = -math.inf
    for idx in order:
        ts = safe_float(timestamps[idx])
        if not bool(selected[idx]) or not math.isfinite(ts) or ts < next_allowed:
            continue
        value = safe_float(net[idx])
        if math.isfinite(value):
            values.append(value)
            next_allowed = ts + horizon_ms
    return np.asarray(values, dtype=np.float64)


def policy_metrics_for_horizon(
    frame: pd.DataFrame,
    actual: np.ndarray,
    pred: np.ndarray,
    horizon: int,
    bucket_seconds: float,
    threshold: float,
    cost_bps: float,
) -> dict[str, Any]:
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    selected, direction = select_signal(pred, threshold)
    selected = selected & mask
    gross = direction * actual
    net = gross - cost_bps
    selected_net = net[selected]
    selected_gross = gross[selected]
    pred_positive = pred > 0.0
    actual_positive = actual > 0.0
    if INSTRUMENT in {"margin_long_short", "perpetual_long_short", "long_short"}:
        profitable_opp = np.abs(actual) > cost_bps
        selected_profitable_opp = selected & (gross > cost_bps)
    else:
        profitable_opp = actual > cost_bps
        selected_profitable_opp = selected & (actual > cost_bps)
    timestamps = pd.to_numeric(frame["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    horizon_ms = horizon * bucket_seconds * 1000.0
    non_overlap = non_overlapping_values(timestamps, selected, net, horizon_ms)
    span_ms = safe_float(np.nanmax(timestamps) - np.nanmin(timestamps), 0.0)
    exposure_fraction = min(len(non_overlap) * horizon_ms / max(span_ms, 1.0), 1.0) if span_ms > 0 else math.nan
    return {
        "threshold_bps": threshold,
        "cost_bps": cost_bps,
        "rows": int(mask.sum()),
        "mae": float(np.nanmean(np.abs(actual[mask] - pred[mask]))) if mask.any() else math.nan,
        "rmse": float(np.sqrt(np.nanmean((actual[mask] - pred[mask]) ** 2))) if mask.any() else math.nan,
        "correlation": corr(actual, pred),
        "directional_accuracy": float(np.mean(pred_positive[mask] == actual_positive[mask])) if mask.any() else math.nan,
        "balanced_accuracy": balanced_accuracy(actual_positive[mask], pred_positive[mask]) if mask.any() else math.nan,
        "selected_signal_count": int(selected.sum()),
        "selected_signal_precision_win_rate": float(np.mean(selected_net > 0.0)) if len(selected_net) else math.nan,
        "recall_profitable_opportunities": float(selected_profitable_opp.sum() / profitable_opp.sum()) if profitable_opp.any() else math.nan,
        "avg_selected_gross_bps": float(np.mean(selected_gross)) if len(selected_gross) else math.nan,
        "median_selected_gross_bps": float(np.median(selected_gross)) if len(selected_gross) else math.nan,
        "avg_selected_net_bps": float(np.mean(selected_net)) if len(selected_net) else math.nan,
        "median_selected_net_bps": float(np.median(selected_net)) if len(selected_net) else math.nan,
        "row_signal_cum_net_bps": float(np.sum(selected_net)) if len(selected_net) else 0.0,
        "row_signal_max_drawdown_bps": max_dip(selected_net),
        "non_overlapping_trade_count": int(len(non_overlap)),
        "non_overlapping_avg_net_bps": float(np.mean(non_overlap)) if len(non_overlap) else math.nan,
        "non_overlapping_cum_net_bps": float(np.sum(non_overlap)) if len(non_overlap) else 0.0,
        "non_overlapping_win_rate": float(np.mean(non_overlap > 0.0)) if len(non_overlap) else math.nan,
        "non_overlapping_max_drawdown_bps": max_dip(non_overlap),
        "position_trade_count": int(len(non_overlap)),
        "position_cum_net_bps": float(np.sum(non_overlap)) if len(non_overlap) else 0.0,
        "position_avg_net_bps": float(np.mean(non_overlap)) if len(non_overlap) else math.nan,
        "position_win_rate": float(np.mean(non_overlap > 0.0)) if len(non_overlap) else math.nan,
        "position_max_drawdown_bps": max_dip(non_overlap),
        "position_exposure_fraction": exposure_fraction,
        "position_average_hold_buckets": horizon if len(non_overlap) else 0,
        "position_turnover": float(len(non_overlap) / max(len(frame), 1)),
        "position_max_concurrent_exposure": 1 if len(non_overlap) else 0,
    }


def split_frame(table: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return table[table["split"].astype(str).eq(split_name)].copy().reset_index(drop=True)


def make_prediction_models(
    table: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    feature_group: str = "all",
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    train = split_frame(table, "train")
    validation = split_frame(table, "validation")
    all_features = table[feature_columns].copy()
    Y_train = target_matrix(train, target_columns)
    Y_val = target_matrix(validation, target_columns)
    train_target_mean = np.nanmean(Y_train, axis=0)
    train_target_mean = np.where(np.isfinite(train_target_mean), train_target_mean, 0.0)
    predictions: dict[str, np.ndarray] = {
        model_with_group(feature_group, "zero_return_baseline"): np.zeros((len(table), len(target_columns)), dtype=np.float64),
        model_with_group(feature_group, "training_mean_return_baseline"): np.tile(train_target_mean, (len(table), 1)),
    }
    model_rows = [
        {
            "model": model_with_group(feature_group, "zero_return_baseline"),
            "base_model": "zero_return_baseline",
            "feature_group": feature_group,
            "feature_count": len(feature_columns),
            "status": "ok",
            "selection_stage": "none",
        },
        {
            "model": model_with_group(feature_group, "training_mean_return_baseline"),
            "base_model": "training_mean_return_baseline",
            "feature_group": feature_group,
            "feature_count": len(feature_columns),
            "status": "ok",
            "selection_stage": "train_fit_constant",
        },
    ]

    momentum_candidates = []
    reversion_candidates = []
    for window in sorted(set(FEATURE_WINDOWS)):
        mean_col = f"rolling_mean_return_bps_fw{window}"
        ma_col = f"sma_distance_bps_fw{window}"
        if mean_col in table.columns and mean_col in feature_columns:
            pred = np.column_stack(
                [
                    pd.to_numeric(table[mean_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) * horizon
                    for horizon in HORIZON_BUCKETS
                ]
            )
            val_pred = pred[table["split"].astype(str).eq("validation").to_numpy()]
            momentum_candidates.append((rmse_matrix(Y_val, val_pred), window, pred))
        if ma_col in table.columns and ma_col in feature_columns:
            pred = np.column_stack(
                [
                    -pd.to_numeric(table[ma_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                    for _ in HORIZON_BUCKETS
                ]
            )
            val_pred = pred[table["split"].astype(str).eq("validation").to_numpy()]
            reversion_candidates.append((rmse_matrix(Y_val, val_pred), window, pred))
    if momentum_candidates:
        score, window, pred = sorted(momentum_candidates, key=lambda item: item[0] if math.isfinite(item[0]) else 1e18)[0]
        model_name = model_with_group(feature_group, "rolling_mean_momentum_baseline")
        predictions[model_name] = pred
        model_rows.append(
            {
                "model": model_name,
                "base_model": "rolling_mean_momentum_baseline",
                "feature_group": feature_group,
                "status": "ok",
                "selection_stage": "validation_selected_window",
                "selected_feature_window": window,
                "validation_combined_rmse": score,
                "feature_count": len(feature_columns),
            }
        )
    if reversion_candidates:
        score, window, pred = sorted(reversion_candidates, key=lambda item: item[0] if math.isfinite(item[0]) else 1e18)[0]
        model_name = model_with_group(feature_group, "mean_reversion_baseline")
        predictions[model_name] = pred
        model_rows.append(
            {
                "model": model_name,
                "base_model": "mean_reversion_baseline",
                "feature_group": feature_group,
                "status": "ok",
                "selection_stage": "validation_selected_window",
                "selected_feature_window": window,
                "validation_combined_rmse": score,
                "feature_count": len(feature_columns),
            }
        )

    pre = fit_preprocessor(train, feature_columns)
    X_all = transform_features(all_features, feature_columns, pre)
    X_train = X_all[table["split"].astype(str).eq("train").to_numpy()]
    X_val = X_all[table["split"].astype(str).eq("validation").to_numpy()]
    ridge_candidates = []
    for alpha in RIDGE_ALPHAS:
        coef = fit_ridge_multi_output(X_train, Y_train, alpha)
        val_pred = predict_ridge(X_val, coef)
        ridge_candidates.append((rmse_matrix(Y_val, val_pred), alpha, coef))
    if ridge_candidates:
        score, alpha, coef = sorted(ridge_candidates, key=lambda item: item[0] if math.isfinite(item[0]) else 1e18)[0]
        model_name = model_with_group(feature_group, "ridge_multi_output")
        predictions[model_name] = predict_ridge(X_all, coef)
        model_rows.append(
            {
                "model": model_name,
                "base_model": "ridge_multi_output",
                "feature_group": feature_group,
                "status": "ok",
                "selection_stage": "validation_selected_alpha",
                "selected_alpha": alpha,
                "validation_combined_rmse": score,
                "feature_count": len(feature_columns),
            }
        )
    elastic_candidates = []
    for alpha in ELASTIC_NET_ALPHAS:
        coef = fit_elastic_net_multi_output(X_train, Y_train, alpha)
        val_pred = predict_ridge(X_val, coef)
        elastic_candidates.append((rmse_matrix(Y_val, val_pred), alpha, coef))
    if elastic_candidates:
        score, alpha, coef = sorted(elastic_candidates, key=lambda item: item[0] if math.isfinite(item[0]) else 1e18)[0]
        model_name = model_with_group(feature_group, "elastic_net_multi_output")
        predictions[model_name] = predict_ridge(X_all, coef)
        model_rows.append(
            {
                "model": model_name,
                "base_model": "elastic_net_multi_output",
                "feature_group": feature_group,
                "status": "ok",
                "selection_stage": "validation_selected_alpha",
                "selected_alpha": alpha,
                "l1_ratio": ELASTIC_NET_L1_RATIO,
                "validation_combined_rmse": score,
                "feature_count": len(feature_columns),
            }
        )
    try:
        pred, meta = fit_logistic_direction_model(X_train, Y_train, X_all)
        model_name = model_with_group(feature_group, "logistic_direction_model")
        predictions[model_name] = pred
        model_rows.append(
            {
                "model": model_name,
                "base_model": "logistic_direction_model",
                "feature_group": feature_group,
                "status": "ok",
                "selection_stage": "train_fit_direction",
                "feature_count": len(feature_columns),
                **meta,
            }
        )
    except Exception as exc:
        model_rows.append(
            {
                "model": model_with_group(feature_group, "logistic_direction_model"),
                "base_model": "logistic_direction_model",
                "feature_group": feature_group,
                "status": "failed",
                "failure": str(exc),
            }
        )
    if ENABLE_TREE_BASELINE:
        try:
            pred, meta = fit_decision_stump_multi_output(X_train, Y_train, X_val, Y_val, X_all, feature_columns)
            model_name = model_with_group(feature_group, "small_tree_baseline")
            predictions[model_name] = pred
            model_rows.append(
                {
                    "model": model_name,
                    "base_model": "small_tree_baseline",
                    "feature_group": feature_group,
                    "status": "ok",
                    "selection_stage": "validation_selected_stump",
                    "feature_count": len(feature_columns),
                    **meta,
                }
            )
        except Exception as exc:
            model_rows.append(
                {
                    "model": model_with_group(feature_group, "small_tree_baseline"),
                    "base_model": "small_tree_baseline",
                    "feature_group": feature_group,
                    "status": "failed",
                    "failure": str(exc),
                }
            )
    if ENABLE_BOOSTED_TREE_BASELINE:
        try:
            pred, meta = fit_boosted_stumps_multi_output(X_train, Y_train, X_val, Y_val, X_all, feature_columns)
            model_name = model_with_group(feature_group, "boosted_tree_baseline")
            predictions[model_name] = pred
            model_rows.append(
                {
                    "model": model_name,
                    "base_model": "boosted_tree_baseline",
                    "feature_group": feature_group,
                    "status": "ok",
                    "selection_stage": "validation_selected_boosted_stumps",
                    "feature_count": len(feature_columns),
                    **meta,
                }
            )
        except Exception as exc:
            model_rows.append(
                {
                    "model": model_with_group(feature_group, "boosted_tree_baseline"),
                    "base_model": "boosted_tree_baseline",
                    "feature_group": feature_group,
                    "status": "failed",
                    "failure": str(exc),
                }
            )
    if ENABLE_MLP:
        try:
            pred, meta = fit_mlp(X_train, Y_train, X_all)
            model_name = model_with_group(feature_group, "mlp_multi_output")
            predictions[model_name] = pred
            model_rows.append(
                {
                    "model": model_name,
                    "base_model": "mlp_multi_output",
                    "feature_group": feature_group,
                    "status": "ok",
                    "selection_stage": "fixed_smoke_config",
                    "feature_count": len(feature_columns),
                    **meta,
                }
            )
        except Exception as exc:
            model_rows.append(
                {
                    "model": model_with_group(feature_group, "mlp_multi_output"),
                    "base_model": "mlp_multi_output",
                    "feature_group": feature_group,
                    "status": "failed",
                    "failure": str(exc),
                }
            )
    else:
        model_rows.append(
            {
                "model": model_with_group(feature_group, "mlp_multi_output"),
                "base_model": "mlp_multi_output",
                "feature_group": feature_group,
                "status": "disabled",
                "selection_stage": "not_run",
            }
        )
    if ENABLE_SEQUENCE_MLP:
        sequence_context: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None = None
        try:
            limited_feature_count = min(SEQUENCE_MLP_MAX_FEATURES, X_all.shape[1])
            X_limited = X_all[:, :limited_feature_count]
            Y_all = target_matrix(table, target_columns)
            seq_x, seq_y, row_indices, seq_splits = build_sequence_arrays(
                X_limited,
                Y_all,
                table["split"].astype(str).to_numpy(),
                SEQUENCE_LEN,
            )
            sequence_context = (seq_x, seq_y, row_indices, seq_splits, limited_feature_count)
            train_mask = seq_splits == "train"
            pred_full = np.full((len(table), len(target_columns)), np.nan, dtype=np.float64)
            if train_mask.sum() >= 10 and len(seq_x) > 0:
                pred_seq, meta = fit_tanh_mlp(
                    seq_x[train_mask],
                    seq_y[train_mask],
                    seq_x,
                    SEQUENCE_MLP_HIDDEN,
                    SEQUENCE_MLP_EPOCHS,
                    MLP_LR,
                    MLP_SEED,
                )
                pred_full[row_indices] = pred_seq
                model_name = model_with_group(feature_group, "sequence_mlp_multi_output")
                predictions[model_name] = pred_full
                model_rows.append(
                    {
                        "model": model_name,
                        "base_model": "sequence_mlp_multi_output",
                        "feature_group": feature_group,
                        "status": "ok",
                        "selection_stage": "fixed_sequence_config",
                        "seq_len": SEQUENCE_LEN,
                        "sequence_feature_count": limited_feature_count,
                        "sequence_train_rows": int(train_mask.sum()),
                        "sequence_total_rows": int(len(seq_x)),
                        **meta,
                    }
                )
            else:
                model_rows.append(
                    {
                        "model": model_with_group(feature_group, "sequence_mlp_multi_output"),
                        "base_model": "sequence_mlp_multi_output",
                        "feature_group": feature_group,
                        "status": "insufficient_sequence_rows",
                        "seq_len": SEQUENCE_LEN,
                        "sequence_train_rows": int(train_mask.sum()),
                    }
                )
        except Exception as exc:
            model_rows.append(
                {
                    "model": model_with_group(feature_group, "sequence_mlp_multi_output"),
                    "base_model": "sequence_mlp_multi_output",
                    "feature_group": feature_group,
                    "status": "failed",
                    "failure": str(exc),
                }
            )
    if ENABLE_TORCH_SEQUENCE_MODELS:
        if not ENABLE_SEQUENCE_MLP:
            sequence_context = None
            try:
                limited_feature_count = min(SEQUENCE_MLP_MAX_FEATURES, X_all.shape[1])
                X_limited = X_all[:, :limited_feature_count]
                Y_all = target_matrix(table, target_columns)
                seq_x, seq_y, row_indices, seq_splits = build_sequence_arrays(
                    X_limited,
                    Y_all,
                    table["split"].astype(str).to_numpy(),
                    SEQUENCE_LEN,
                )
                sequence_context = (seq_x, seq_y, row_indices, seq_splits, limited_feature_count)
            except Exception:
                sequence_context = None
        for torch_model in TORCH_SEQUENCE_MODELS:
            model_name = model_with_group(feature_group, f"torch_{torch_model}_sequence_model")
            if sequence_context is None:
                model_rows.append(
                    {
                        "model": model_name,
                        "base_model": f"torch_{torch_model}_sequence_model",
                        "feature_group": feature_group,
                        "status": "failed",
                        "failure": "sequence_context_build_failed",
                    }
                )
                continue
            seq_x, seq_y, row_indices, seq_splits, limited_feature_count = sequence_context
            try:
                pred_full, meta = fit_torch_sequence_model(
                    seq_x,
                    seq_y,
                    seq_splits,
                    row_indices,
                    len(table),
                    len(target_columns),
                    SEQUENCE_LEN,
                    limited_feature_count,
                    torch_model,
                )
                predictions[model_name] = pred_full
                model_rows.append(
                    {
                        "model": model_name,
                        "base_model": f"torch_{torch_model}_sequence_model",
                        "feature_group": feature_group,
                        "status": "ok",
                        "selection_stage": "fixed_torch_sequence_config",
                        "sequence_total_rows": int(len(seq_x)),
                        **meta,
                    }
                )
            except RuntimeError as exc:
                failure = str(exc)
                model_rows.append(
                    {
                        "model": model_name,
                        "base_model": f"torch_{torch_model}_sequence_model",
                        "feature_group": feature_group,
                        "status": "skipped_torch_unavailable"
                        if failure == "torch_unavailable"
                        else "skipped_sequence_unavailable",
                        "selection_stage": "not_run",
                        "failure": failure,
                        **torch_status(),
                    }
                )
            except Exception as exc:
                model_rows.append(
                    {
                        "model": model_name,
                        "base_model": f"torch_{torch_model}_sequence_model",
                        "feature_group": feature_group,
                        "status": "failed",
                        "failure": str(exc),
                        **torch_status(),
                    }
                )
    return predictions, pd.DataFrame(model_rows)


def build_metrics(
    table: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    target_columns: list[str],
    bucket_seconds: float,
    splits: list[str] | None = None,
    selected_policies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    splits = splits or ["train", "validation"]
    selected_lookup: dict[tuple[str, int], set[float]] = {}
    if selected_policies is not None and not selected_policies.empty:
        for _, selected in selected_policies.iterrows():
            key = (str(selected.get("model")), int(float(selected.get("horizon_buckets"))))
            selected_lookup.setdefault(key, set()).add(float(selected.get("selected_threshold_bps")))
    for model, pred_matrix in predictions.items():
        for split_name in splits:
            subset = split_frame(table, split_name)
            if subset.empty:
                continue
            split_mask = table["split"].astype(str).eq(split_name).to_numpy()
            for horizon_index, horizon in enumerate(HORIZON_BUCKETS):
                actual = pd.to_numeric(subset[target_columns[horizon_index]], errors="coerce").to_numpy(dtype=np.float64)
                pred = pred_matrix[split_mask, horizon_index]
                if split_name == "untouched_holdout":
                    thresholds = sorted(selected_lookup.get((model, int(horizon)), set()))
                    if not thresholds:
                        continue
                else:
                    thresholds = THRESHOLDS_BPS
                for threshold in thresholds:
                    for cost in COST_BPS:
                        metric = policy_metrics_for_horizon(
                            subset,
                            actual,
                            pred,
                            horizon,
                            bucket_seconds,
                            threshold,
                            cost,
                        )
                        metric.update(
                            {
                                "model": model,
                                "split": split_name,
                                "horizon_buckets": horizon,
                                "horizon_seconds": horizon * bucket_seconds,
                                "target_column": target_columns[horizon_index],
                                "instrument": INSTRUMENT,
                                "threshold_selection_split": "validation" if split_name == "untouched_holdout" else "",
                                "evaluation_stage": {
                                    "train": "train_diagnostic_grid",
                                    "validation": "validation_selection_grid",
                                    "untouched_holdout": "holdout_final_selected_policy",
                                }.get(split_name, split_name),
                                "holdout_threshold_source": "validation_selected"
                                if split_name == "untouched_holdout"
                                else "",
                            }
                        )
                        rows.append(metric)
    return pd.DataFrame(rows)


def selected_validation_policies(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    validation = metrics[
        metrics["split"].eq("validation")
        & metrics["cost_bps"].astype(float).sub(DECISION_COST_BPS).abs().lt(1e-12)
    ].copy()
    for (model, horizon), group in validation.groupby(["model", "horizon_buckets"], dropna=False):
        group = group.sort_values(
            ["position_cum_net_bps", "position_trade_count", "rmse"],
            ascending=[False, False, True],
        )
        selected = group.iloc[0]
        rows.append(
            {
                "model": model,
                "horizon_buckets": int(float(horizon)),
                "target_column": selected.get("target_column", ""),
                "selected_threshold_bps": float(selected.get("threshold_bps")),
                "decision_cost_bps": DECISION_COST_BPS,
                "selection_stage": "validation_selected",
                "validation_position_cum_net_bps": selected.get("position_cum_net_bps"),
                "validation_position_trade_count": selected.get("position_trade_count"),
            }
        )
    return pd.DataFrame(rows)


def build_leaderboard(metrics: pd.DataFrame, model_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    validation = metrics[
        metrics["split"].eq("validation")
        & metrics["cost_bps"].astype(float).sub(DECISION_COST_BPS).abs().lt(1e-12)
    ].copy()
    selected_policies = selected_validation_policies(metrics)
    for _, policy_row in selected_policies.iterrows():
        model = str(policy_row.get("model"))
        horizon = int(float(policy_row.get("horizon_buckets")))
        group = validation[
            validation["model"].eq(model)
            & validation["horizon_buckets"].astype(float).eq(float(horizon))
            & validation["threshold_bps"].astype(float).sub(float(policy_row["selected_threshold_bps"])).abs().lt(1e-12)
        ]
        if group.empty:
            continue
        selected = group.iloc[0]
        holdout = metrics[
            metrics["split"].eq("untouched_holdout")
            & metrics["model"].eq(model)
            & metrics["horizon_buckets"].astype(float).eq(float(horizon))
            & metrics["threshold_bps"].astype(float).sub(float(selected["threshold_bps"])).abs().lt(1e-12)
            & metrics["cost_bps"].astype(float).sub(DECISION_COST_BPS).abs().lt(1e-12)
        ]
        holdout_row = holdout.iloc[0] if not holdout.empty else pd.Series(dtype=object)
        validation_trades = int(safe_float(selected.get("position_trade_count"), 0))
        holdout_trades = int(safe_float(holdout_row.get("position_trade_count"), 0))
        validation_cum = safe_float(selected.get("position_cum_net_bps"))
        holdout_cum = safe_float(holdout_row.get("position_cum_net_bps"))
        if validation_trades < MIN_POSITION_TRADES:
            status = "insufficient_sample"
            reason = f"validation_position_trade_count<{MIN_POSITION_TRADES}"
        elif validation_cum <= 0.0:
            status = "training_candidate"
            reason = "validation_position_cum_net_not_positive"
        elif holdout_trades >= MIN_POSITION_TRADES and holdout_cum > 0.0:
            status = "holdout_survivor"
            reason = "validation_selected_threshold_survived_untouched_holdout"
        else:
            status = "validation_survivor"
            reason = "validation_positive_but_holdout_gate_not_met"
        model_meta = model_manifest[model_manifest["model"].eq(model)]
        meta = model_meta.iloc[0].to_dict() if not model_meta.empty else {}
        base_model = str(meta.get("base_model") or base_model_name(model))
        rows.append(
            {
                "model": model,
                "base_model": base_model,
                "feature_group": meta.get("feature_group", ""),
                "horizon_buckets": horizon,
                "horizon_seconds": selected.get("horizon_seconds"),
                "selection_stage": "validation_selected",
                "holdout_stage": "untouched_holdout_evaluated_once",
                "holdout_threshold_source": "validation_selected",
                "selected_threshold_bps": selected.get("threshold_bps"),
                "decision_cost_bps": DECISION_COST_BPS,
                "status": status,
                "status_reason": reason,
                "validation_rmse": selected.get("rmse"),
                "validation_correlation": selected.get("correlation"),
                "validation_directional_accuracy": selected.get("directional_accuracy"),
                "validation_balanced_accuracy": selected.get("balanced_accuracy"),
                "validation_position_trade_count": validation_trades,
                "validation_position_cum_net_bps": validation_cum,
                "validation_position_max_drawdown_bps": selected.get("position_max_drawdown_bps"),
                "validation_position_exposure_fraction": selected.get("position_exposure_fraction"),
                "holdout_rmse": holdout_row.get("rmse", math.nan),
                "holdout_correlation": holdout_row.get("correlation", math.nan),
                "holdout_directional_accuracy": holdout_row.get("directional_accuracy", math.nan),
                "holdout_balanced_accuracy": holdout_row.get("balanced_accuracy", math.nan),
                "holdout_position_trade_count": holdout_trades,
                "holdout_position_cum_net_bps": holdout_cum,
                "holdout_position_max_drawdown_bps": holdout_row.get("position_max_drawdown_bps", math.nan),
                "holdout_position_exposure_fraction": holdout_row.get("position_exposure_fraction", math.nan),
                "model_training_status": meta.get("status", ""),
                "model_selected_alpha": meta.get("selected_alpha", ""),
                "model_selected_feature_window": meta.get("selected_feature_window", ""),
                "model_type": meta.get("model_type", ""),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    status_priority = {
        "holdout_survivor": 0,
        "validation_survivor": 1,
        "training_candidate": 2,
        "insufficient_sample": 3,
    }
    out["status_priority"] = out["status"].map(status_priority).fillna(9)
    return out.sort_values(
        [
            "status_priority",
            "holdout_position_cum_net_bps",
            "validation_position_cum_net_bps",
            "holdout_rmse",
        ],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def best_row_by_holdout(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    ranked = frame.copy()
    ranked["status_is_holdout_survivor"] = ranked["status"].eq("holdout_survivor").astype(int)
    ranked = ranked.sort_values(
        [
            "status_is_holdout_survivor",
            "holdout_position_cum_net_bps",
            "holdout_position_trade_count",
            "holdout_rmse",
        ],
        ascending=[False, False, False, True],
    )
    return ranked.iloc[0].to_dict()


def build_horizon_summary(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame()
    rows = []
    for horizon, group in leaderboard.groupby("horizon_buckets", dropna=False):
        horizon = int(float(horizon))
        group = group.copy()
        if "base_model" not in group.columns:
            group["base_model"] = group["model"].map(base_model_name)
        baseline_rows = group[group["base_model"].isin(NAMED_BASELINE_MODELS)].copy()
        acceptance_baselines = group[group["base_model"].isin(ACCEPTANCE_BASELINE_MODELS)].copy()
        nonbaseline_rows = group[~group["base_model"].isin(NAMED_BASELINE_MODELS)].copy()
        best_any = best_row_by_holdout(group)
        best_nonbaseline = best_row_by_holdout(nonbaseline_rows)

        baseline_values: dict[str, float] = {}
        for model in ACCEPTANCE_BASELINE_MODELS:
            model_rows = group[group["base_model"].eq(model)]
            if model_rows.empty:
                baseline_values[model] = math.nan
            else:
                baseline_values[model] = safe_float(
                    best_row_by_holdout(model_rows).get("holdout_position_cum_net_bps")
                )
        best_acceptance_baseline = (
            float(np.nanmax(list(baseline_values.values())))
            if any(math.isfinite(value) for value in baseline_values.values())
            else math.nan
        )
        best_named_baseline_row = best_row_by_holdout(baseline_rows)
        best_acceptance_baseline_row = best_row_by_holdout(acceptance_baselines)
        nonbaseline_holdout = safe_float(best_nonbaseline.get("holdout_position_cum_net_bps"))
        nonbaseline_status = str(best_nonbaseline.get("status", ""))
        beats_zero = nonbaseline_holdout > baseline_values["zero_return_baseline"]
        beats_momentum = nonbaseline_holdout > baseline_values["rolling_mean_momentum_baseline"]
        beats_reversion = nonbaseline_holdout > baseline_values["mean_reversion_baseline"]
        beats_all_acceptance_baselines = bool(beats_zero and beats_momentum and beats_reversion)
        has_holdout_survivor = bool(group["status"].eq("holdout_survivor").any())
        nonbaseline_holdout_survivor = nonbaseline_status == "holdout_survivor"

        if nonbaseline_holdout_survivor and beats_all_acceptance_baselines:
            assessment = "promising_nonbaseline_holdout_survivor_beats_named_baselines"
        elif has_holdout_survivor:
            assessment = "holdout_survivor_exists_but_baseline_comparison_mixed"
        elif group["status"].eq("validation_survivor").any():
            assessment = "validation_only_no_holdout_survivor"
        else:
            assessment = "no_promising_holdout_signal"

        rows.append(
            {
                "horizon_buckets": horizon,
                "horizon_seconds": best_any.get("horizon_seconds", math.nan),
                "best_any_model": best_any.get("model", ""),
                "best_any_status": best_any.get("status", ""),
                "best_any_holdout_position_cum_net_bps": best_any.get("holdout_position_cum_net_bps", math.nan),
                "best_any_holdout_trade_count": best_any.get("holdout_position_trade_count", math.nan),
                "best_nonbaseline_model": best_nonbaseline.get("model", ""),
                "best_nonbaseline_status": nonbaseline_status,
                "best_nonbaseline_holdout_position_cum_net_bps": nonbaseline_holdout,
                "best_nonbaseline_holdout_trade_count": best_nonbaseline.get("holdout_position_trade_count", math.nan),
                "best_named_baseline_model": best_named_baseline_row.get("model", ""),
                "best_named_baseline_holdout_position_cum_net_bps": best_named_baseline_row.get("holdout_position_cum_net_bps", math.nan),
                "best_acceptance_baseline_model": best_acceptance_baseline_row.get("model", ""),
                "best_acceptance_baseline_holdout_position_cum_net_bps": best_acceptance_baseline,
                "zero_baseline_holdout_position_cum_net_bps": baseline_values["zero_return_baseline"],
                "momentum_baseline_holdout_position_cum_net_bps": baseline_values["rolling_mean_momentum_baseline"],
                "mean_reversion_baseline_holdout_position_cum_net_bps": baseline_values["mean_reversion_baseline"],
                "best_nonbaseline_beats_zero_holdout": bool(beats_zero),
                "best_nonbaseline_beats_momentum_holdout": bool(beats_momentum),
                "best_nonbaseline_beats_mean_reversion_holdout": bool(beats_reversion),
                "best_nonbaseline_beats_all_named_acceptance_baselines_holdout": beats_all_acceptance_baselines,
                "holdout_survivor_count": int(group["status"].eq("holdout_survivor").sum()),
                "nonbaseline_holdout_survivor_count": int(nonbaseline_rows["status"].eq("holdout_survivor").sum()),
                "horizon_assessment": assessment,
            }
        )
    out = pd.DataFrame(rows)
    assessment_priority = {
        "promising_nonbaseline_holdout_survivor_beats_named_baselines": 0,
        "holdout_survivor_exists_but_baseline_comparison_mixed": 1,
        "validation_only_no_holdout_survivor": 2,
        "no_promising_holdout_signal": 3,
    }
    out["assessment_priority"] = out["horizon_assessment"].map(assessment_priority).fillna(9)
    return out.sort_values(
        [
            "assessment_priority",
            "best_nonbaseline_holdout_position_cum_net_bps",
            "best_any_holdout_position_cum_net_bps",
            "best_nonbaseline_holdout_trade_count",
        ],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def build_feature_manifest(
    feature_columns: list[str],
    source_meta: dict[str, Any],
    bucket_seconds: float,
    family_manifest: pd.DataFrame,
) -> dict[str, Any]:
    family_counts = (
        family_manifest["feature_family"].value_counts().sort_index().to_dict()
        if not family_manifest.empty
        else {}
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": SYMBOL,
        "venue": VENUE,
        "source_path": str(SOURCE_PATH),
        "source_sha256": file_sha256(SOURCE_PATH),
        "source_columns": source_meta,
        "bucket_seconds": bucket_seconds,
        "feature_windows": FEATURE_WINDOWS,
        "macd_spec": MACD_SPEC,
        "include_flow_features": INCLUDE_FLOW_FEATURES,
        "cross_market_sources": {key: value for key, value in CROSS_MARKET_SOURCES.items() if value},
        "feature_group_specs": FEATURE_GROUP_SPECS,
        "feature_family_counts": family_counts,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "causality": "all indicators use current and historical rows only; rolling windows are not centered",
        "default_feature_scope": "OHLCV technical indicators; spread/depth/imbalance features are opt-in via RAWSEQ_MH_INCLUDE_FLOW_FEATURES",
        "feature_timestamp_max_rule": "feature_timestamp_max <= decision_timestamp",
        "missing_indicator_rule": "each feature has a companion <feature>_missing column in the training table",
    }


def validate_training_table(table: pd.DataFrame, feature_columns: list[str], target_columns: list[str]) -> dict[str, Any]:
    return {
        "feature_timestamp_max_lte_decision_timestamp": bool((table["feature_timestamp_max"] <= table["decision_timestamp"]).all()),
        "label_end_timestamp_gt_decision_timestamp": bool((table["label_end_timestamp"] > table["decision_timestamp"]).all()),
        "unique_symbol_venue_timestamp_keys": bool(not table.duplicated(["symbol", "venue", "decision_timestamp"]).any()),
        "sorted_chronological_order": bool(table["decision_timestamp"].is_monotonic_increasing),
        "feature_columns": len(feature_columns),
        "target_columns": target_columns,
        "paper_only": True,
        "promotion": False,
        "champion_mutation": False,
        "orders": False,
    }


def write_report(
    path: Path,
    run_dir: Path,
    table: pd.DataFrame,
    feature_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    model_manifest: pd.DataFrame,
    leaderboard: pd.DataFrame,
    horizon_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Multi-Horizon Technical Return Pipeline",
        "",
        f"Created at: {datetime.now(UTC).isoformat()}",
        f"Run dir: {run_dir}",
        f"Source: {SOURCE_PATH}",
        f"Symbol/venue: {SYMBOL}/{VENUE}",
        f"Instrument: {INSTRUMENT}",
        f"Rows: {len(table)}",
        f"Feature count: {feature_manifest.get('feature_count')}",
        f"Flow/order-book features enabled: {feature_manifest.get('include_flow_features')}",
        f"Horizon buckets: {HORIZON_BUCKETS}",
        f"Bucket seconds: {feature_manifest.get('bucket_seconds')}",
        "",
        "## Safeguards",
        f"- feature_timestamp_max <= decision_timestamp: {split_manifest.get('guards', {}).get('feature_timestamp_max_lte_decision_timestamp')}",
        f"- label_end_timestamp > decision_timestamp: {split_manifest.get('guards', {}).get('label_end_timestamp_gt_decision_timestamp')}",
        f"- purge_rows: {split_manifest.get('purge_rows')}",
        f"- embargo_rows: {split_manifest.get('embargo_rows')}",
        "- validation selects model/threshold/policy; untouched holdout is reported after selection.",
        "- holdout metrics are emitted only for validation-selected thresholds.",
        "- paper_only=true; private_api=false; orders=false; promotion=false; champion_mutation=false.",
        "",
        "## Models",
    ]
    for _, row in model_manifest.iterrows():
        lines.append(f"- {row.get('model')}: status={row.get('status')} selection={row.get('selection_stage')}")
    lines += ["", "## Most Promising Horizons"]
    if horizon_summary.empty:
        lines.append("No horizon summary rows were produced.")
    else:
        for _, row in horizon_summary.iterrows():
            lines.append(
                "- "
                f"h={row['horizon_buckets']} buckets ({fmt(row.get('horizon_seconds'), 2)}s): "
                f"{row['horizon_assessment']}; "
                f"best_nonbaseline={row.get('best_nonbaseline_model', '')} "
                f"status={row.get('best_nonbaseline_status', '')} "
                f"holdout_pos_net={fmt(row.get('best_nonbaseline_holdout_position_cum_net_bps'))}; "
                f"best_named_baseline={row.get('best_acceptance_baseline_model', '')} "
                f"holdout_pos_net={fmt(row.get('best_acceptance_baseline_holdout_position_cum_net_bps'))}"
            )
    baseline_beaters = (
        horizon_summary[horizon_summary["best_nonbaseline_beats_all_named_acceptance_baselines_holdout"].astype(bool)]
        if not horizon_summary.empty
        and "best_nonbaseline_beats_all_named_acceptance_baselines_holdout" in horizon_summary.columns
        else pd.DataFrame()
    )
    lines += ["", "## Holdout Baseline Comparison"]
    if baseline_beaters.empty:
        lines.append(
            "No non-baseline model beat zero, momentum, and mean-reversion baselines together on untouched holdout."
        )
    else:
        for _, row in baseline_beaters.iterrows():
            lines.append(
                "- "
                f"h={row['horizon_buckets']}: {row.get('best_nonbaseline_model', '')} beat "
                "zero/momentum/mean-reversion on holdout "
                f"(model={fmt(row.get('best_nonbaseline_holdout_position_cum_net_bps'))}, "
                f"best_baseline={fmt(row.get('best_acceptance_baseline_holdout_position_cum_net_bps'))})."
            )
    lines += ["", "## Top Validation-Selected Rows"]
    if leaderboard.empty:
        lines.append("No leaderboard rows were produced.")
    else:
        display = leaderboard.head(12)
        for _, row in display.iterrows():
            lines.append(
                "- "
                f"{row['model']} h={row['horizon_buckets']} "
                f"status={row['status']} threshold={fmt(row['selected_threshold_bps'])} "
                f"val_pos_net={fmt(row['validation_position_cum_net_bps'])} "
                f"holdout_pos_net={fmt(row['holdout_position_cum_net_bps'])} "
                f"holdout_trades={row['holdout_position_trade_count']} "
                f"reason={row['status_reason']}"
            )
    lines += [
        "",
        "## Interpretation",
        "This run is preliminary target/baseline research evidence only.",
        "A result may be described as a holdout_survivor only when it used a validation-selected threshold and survived untouched holdout.",
        "Do not call any row credible, clean, promotable, or champion-ready from this report.",
        "Stop here for review before broader runs, ensemble search, forward paper, or any freeze/promotion step.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    source_hash = file_sha256(SOURCE_PATH)
    base, source_meta = load_source(SOURCE_PATH)
    bucket_seconds = estimate_bucket_seconds(base)
    table, feature_columns, feature_audit = add_feature_bank(base, FEATURE_WINDOWS)
    family_manifest = feature_family_manifest(feature_columns)
    table, target_columns, target_manifest_csv = add_targets(table, HORIZON_BUCKETS)
    required_targets = ["label_end_timestamp", *target_columns]
    table = table.dropna(subset=required_targets).reset_index(drop=True)
    table["symbol"] = SYMBOL
    table["venue"] = VENUE
    table["instrument"] = INSTRUMENT
    table["feature_timestamp_max"] = table["decision_timestamp"]
    table["source_sha256"] = source_hash
    feature_manifest = build_feature_manifest(feature_columns, source_meta, bucket_seconds, family_manifest)
    target_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "target_family": "market_relative_cumulative_future_return_bps_array",
        "horizon_buckets": HORIZON_BUCKETS,
        "horizon_seconds": [horizon * bucket_seconds for horizon in HORIZON_BUCKETS],
        "target_columns": target_columns,
        "output_dim": len(target_columns),
        "label_end_timestamp": "label_end_timestamp_h<max_horizon>",
    }
    contract_hash = stable_hash({"feature_manifest": feature_manifest, "target_manifest": target_manifest})[:8]
    table["feature_schema_hash"] = stable_hash(feature_manifest)
    table["target_schema_hash"] = stable_hash(target_manifest)
    table, split_manifest, split_manifest_csv = assign_splits(table, max(HORIZON_BUCKETS), max(FEATURE_WINDOWS))
    guards = validate_training_table(table, feature_columns, target_columns)
    split_manifest.update(
        {
            "source_path": str(SOURCE_PATH),
            "source_sha256": source_hash,
            "source_max_timestamp": float(table["decision_timestamp"].max()) if not table.empty else math.nan,
            "source_max_iso": iso_from_ms(table["decision_timestamp"].max()) if not table.empty else "",
            "estimated_bucket_seconds": bucket_seconds,
            "feature_schema_hash": table["feature_schema_hash"].iloc[0] if not table.empty else "",
            "target_schema_hash": table["target_schema_hash"].iloc[0] if not table.empty else "",
            "data_hashes": {
                "source_sha256": source_hash,
                "feature_schema_hash": table["feature_schema_hash"].iloc[0] if not table.empty else "",
                "target_schema_hash": table["target_schema_hash"].iloc[0] if not table.empty else "",
            },
            "guards": guards,
            "paper_only": True,
            "private_api": False,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
        }
    )
    critical_guard_keys = [
        "feature_timestamp_max_lte_decision_timestamp",
        "label_end_timestamp_gt_decision_timestamp",
        "unique_symbol_venue_timestamp_keys",
        "sorted_chronological_order",
    ]
    if not all(bool(guards.get(key)) for key in critical_guard_keys):
        raise SystemExit(f"Training table guard failed: {guards}")

    predictions: dict[str, np.ndarray] = {}
    model_manifest_parts: list[pd.DataFrame] = []
    ablation_rows: list[dict[str, Any]] = []
    for group_spec in list(dict.fromkeys(FEATURE_GROUP_SPECS)):
        selected_features = selected_feature_columns_for_group(feature_columns, family_manifest, group_spec)
        group_predictions, group_manifest = make_prediction_models(
            table,
            selected_features,
            target_columns,
            feature_group=group_spec,
        )
        predictions.update(group_predictions)
        model_manifest_parts.append(group_manifest)
        ablation_rows.append(
            {
                "feature_group": group_spec,
                "selected_feature_count": len(selected_features),
                "selected_families": ",".join(sorted(set(family_manifest[family_manifest["feature"].isin(selected_features)]["feature_family"]))),
                "models_requested": ",".join(sorted({base_model_name(model) for model in group_predictions})),
            }
        )
    model_manifest = pd.concat(model_manifest_parts, ignore_index=True) if model_manifest_parts else pd.DataFrame()
    ablation_manifest = pd.DataFrame(ablation_rows)
    selection_metrics = build_metrics(table, predictions, target_columns, bucket_seconds, splits=["train", "validation"])
    selected_policies = selected_validation_policies(selection_metrics)
    holdout_metrics = build_metrics(
        table,
        predictions,
        target_columns,
        bucket_seconds,
        splits=["untouched_holdout"],
        selected_policies=selected_policies,
    )
    metrics = pd.concat([selection_metrics, holdout_metrics], ignore_index=True)
    leaderboard = build_leaderboard(metrics, model_manifest)
    horizon_summary = build_horizon_summary(leaderboard)

    run_dir = OUTPUT_ROOT / f"mh_indicator_{SYMBOL}_{VENUE}_{now_stamp()}_{contract_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    table_path = run_dir / "multi_horizon_training_table.csv"
    feature_manifest_json = run_dir / "feature_manifest.json"
    feature_manifest_csv = run_dir / "feature_manifest.csv"
    feature_family_manifest_path = run_dir / "feature_family_manifest.csv"
    target_manifest_json = run_dir / "target_manifest.json"
    target_manifest_csv_path = run_dir / "target_manifest.csv"
    split_manifest_json = run_dir / "split_manifest.json"
    split_manifest_csv_path = run_dir / "split_manifest.csv"
    metrics_path = run_dir / "per_horizon_metrics.csv"
    leaderboard_path = run_dir / "combined_leaderboard.csv"
    horizon_summary_path = run_dir / "horizon_summary.csv"
    selected_policies_path = run_dir / "validation_selected_policies.csv"
    ablation_manifest_path = run_dir / "feature_ablation_manifest.csv"
    sequence_manifest_path = run_dir / "sequence_manifest.json"
    sequence_dataset_manifest_path = run_dir / "sequence_dataset_manifest.csv"
    model_manifest_path = run_dir / "model_manifest.csv"
    feature_audit_path = run_dir / "feature_audit.csv"
    report_path = run_dir / "multi_horizon_indicator_report.txt"

    sequence_dataset_manifest = export_sequence_datasets(
        table,
        feature_columns,
        family_manifest,
        target_columns,
        run_dir,
        FEATURE_GROUP_SPECS,
    )

    table.to_csv(table_path, index=False)
    feature_manifest_json.write_text(json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame({"feature": feature_columns}).to_csv(feature_manifest_csv, index=False)
    family_manifest.to_csv(feature_family_manifest_path, index=False)
    target_manifest_json.write_text(json.dumps(target_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target_manifest_csv.to_csv(target_manifest_csv_path, index=False)
    split_manifest_json.write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split_manifest_csv.to_csv(split_manifest_csv_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)
    horizon_summary.to_csv(horizon_summary_path, index=False)
    selected_policies.to_csv(selected_policies_path, index=False)
    ablation_manifest.to_csv(ablation_manifest_path, index=False)
    sequence_dataset_manifest.to_csv(sequence_dataset_manifest_path, index=False)
    sequence_manifest_path.write_text(
        json.dumps(
            {
                "enabled": ENABLE_SEQUENCE_MLP,
                "torch_sequence_models_enabled": ENABLE_TORCH_SEQUENCE_MODELS,
                "torch_status": torch_status(),
                "torch_sequence_models_requested": TORCH_SEQUENCE_MODELS,
                "seq_len": SEQUENCE_LEN,
                "sequence_lens": SEQUENCE_LENS,
                "sequence_mlp_hidden": SEQUENCE_MLP_HIDDEN,
                "sequence_mlp_epochs": SEQUENCE_MLP_EPOCHS,
                "sequence_dataset_manifest": str(sequence_dataset_manifest_path),
                "write_sequence_datasets": WRITE_SEQUENCE_DATASETS,
                "sequence_dataset_max_features": SEQUENCE_DATASET_MAX_FEATURES,
                "sequence_dataset_rows": int(len(sequence_dataset_manifest)),
                "torch_sequence_hidden": TORCH_SEQUENCE_HIDDEN,
                "torch_sequence_epochs": TORCH_SEQUENCE_EPOCHS,
                "torch_sequence_loss": "HuberLoss when torch models run",
                "sequence_input_shape": "[batch, seq_len, feature_count]",
                "target_shape": "[batch, horizon_count]",
                "horizon_count": len(HORIZON_BUCKETS),
                "note": "Sequence rows use only the last seq_len feature rows ending at the decision row.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    model_manifest.to_csv(model_manifest_path, index=False)
    feature_audit.to_csv(feature_audit_path, index=False)
    write_report(report_path, run_dir, table, feature_manifest, split_manifest, model_manifest, leaderboard, horizon_summary)

    print("Rawseq multi-horizon indicator pipeline complete")
    print(f"Run dir: {run_dir}")
    print(f"Rows: {len(table)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Targets: {target_columns}")
    print(f"Per-horizon metrics: {metrics_path}")
    print(f"Leaderboard: {leaderboard_path}")
    print(f"Horizon summary: {horizon_summary_path}")
    print(f"Validation-selected policies: {selected_policies_path}")
    print(f"Feature-family manifest: {feature_family_manifest_path}")
    print(f"Feature ablation manifest: {ablation_manifest_path}")
    print(f"Sequence manifest: {sequence_manifest_path}")
    print(f"Sequence dataset manifest: {sequence_dataset_manifest_path}")
    print(f"Report: {report_path}")
    if not horizon_summary.empty:
        cols = [
            "horizon_buckets",
            "horizon_assessment",
            "best_nonbaseline_model",
            "best_nonbaseline_holdout_position_cum_net_bps",
            "best_acceptance_baseline_model",
            "best_acceptance_baseline_holdout_position_cum_net_bps",
            "best_nonbaseline_beats_all_named_acceptance_baselines_holdout",
        ]
        print(horizon_summary[cols].head(20).to_string(index=False))
    if not leaderboard.empty:
        cols = [
            "model",
            "horizon_buckets",
            "status",
            "selected_threshold_bps",
            "validation_position_cum_net_bps",
            "holdout_position_cum_net_bps",
            "holdout_position_trade_count",
        ]
        print(leaderboard[cols].head(20).to_string(index=False))
    print("Safety: paper_only=true private_api=false orders=false promotion=false champion_mutation=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
