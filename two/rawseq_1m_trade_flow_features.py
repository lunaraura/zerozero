#!/usr/bin/env python3
"""Reusable Binance trade-flow aggregation helpers for rawseq 1m research."""

from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAW_TRADE_COLUMNS = ["trade_id", "price", "quantity", "quote_quantity", "timestamp_ms", "is_buyer_maker", "is_best_match"]
AGG_TRADE_COLUMNS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "timestamp_ms", "is_buyer_maker", "is_best_match"]
NORMALIZED_TRADE_COLUMNS = ["timestamp_ms", "price", "size", "side"]
FEATURE_WINDOWS = [3, 5, 15, 30, 60]


@dataclass(frozen=True)
class TradeSource:
    path: Path
    source_type: str
    symbol: str
    compressed: bool
    format: str


def buyer_maker_to_aggressor_side(is_buyer_maker: Any) -> str:
    text = str(is_buyer_maker).strip().lower()
    if text in {"true", "1", "t", "yes"}:
        return "sell"
    if text in {"false", "0", "f", "no"}:
        return "buy"
    return "unknown"


def classify_trade_source(path: Path) -> TradeSource:
    name = path.name
    lower = str(path).lower()
    compressed = path.suffix.lower() == ".zip"
    source_type = "unknown"
    symbol = ""
    if "aggtrades" in lower:
        source_type = "binance_agg_trades"
        match = re.search(r"([A-Z0-9]+)-aggTrades-", name)
        symbol = match.group(1) if match else infer_symbol_from_path(path)
    elif "trades" in lower:
        source_type = "binance_raw_trades"
        match = re.search(r"([A-Z0-9]+)-trades-", name)
        symbol = match.group(1) if match else infer_symbol_from_path(path)
    elif "bookticker" in lower:
        source_type = "bookTicker"
        symbol = infer_symbol_from_path(path)
    elif "depth" in lower:
        source_type = "depth"
        symbol = infer_symbol_from_path(path)
    return TradeSource(path=path, source_type=source_type, symbol=symbol, compressed=compressed, format=path.suffix.lower().lstrip(".") or "unknown")


def infer_symbol_from_path(path: Path) -> str:
    for part in [path.stem, *path.parts[::-1]]:
        match = re.search(r"([A-Z]{2,10}USDT)", part)
        if match:
            return match.group(1)
    return ""


def read_csv_head(path: Path, nrows: int = 1000, names: list[str] | None = None, header: int | None | str = "infer") -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not members:
                return pd.DataFrame()
            with zf.open(members[0]) as handle:
                return pd.read_csv(handle, nrows=nrows, names=names, header=header)
    return pd.read_csv(path, nrows=nrows, names=names, header=header)


def normalize_raw_trades(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if list(data.columns[: len(RAW_TRADE_COLUMNS)]) != RAW_TRADE_COLUMNS:
        data = data.iloc[:, : len(RAW_TRADE_COLUMNS)]
        data.columns = RAW_TRADE_COLUMNS[: len(data.columns)]
    out = pd.DataFrame()
    out["trade_id"] = pd.to_numeric(data["trade_id"], errors="coerce")
    out["timestamp_ms"] = pd.to_numeric(data["timestamp_ms"], errors="coerce")
    out["price"] = pd.to_numeric(data["price"], errors="coerce")
    out["size"] = pd.to_numeric(data["quantity"], errors="coerce")
    out["quote_size"] = pd.to_numeric(data.get("quote_quantity", out["price"] * out["size"]), errors="coerce")
    out["is_buyer_maker"] = data["is_buyer_maker"].astype(str)
    out["side"] = out["is_buyer_maker"].map(buyer_maker_to_aggressor_side)
    out["source_type"] = "binance_raw_trades"
    return out


def normalize_agg_trades(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if list(data.columns[: len(AGG_TRADE_COLUMNS)]) != AGG_TRADE_COLUMNS:
        data = data.iloc[:, : len(AGG_TRADE_COLUMNS)]
        data.columns = AGG_TRADE_COLUMNS[: len(data.columns)]
    out = pd.DataFrame()
    out["trade_id"] = pd.to_numeric(data["agg_trade_id"], errors="coerce")
    out["timestamp_ms"] = pd.to_numeric(data["timestamp_ms"], errors="coerce")
    out["price"] = pd.to_numeric(data["price"], errors="coerce")
    out["size"] = pd.to_numeric(data["quantity"], errors="coerce")
    out["quote_size"] = out["price"] * out["size"]
    out["is_buyer_maker"] = data["is_buyer_maker"].astype(str)
    out["side"] = out["is_buyer_maker"].map(buyer_maker_to_aggressor_side)
    out["source_type"] = "binance_agg_trades"
    return out


def read_normalized_trade_file(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = [col for col in NORMALIZED_TRADE_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"missing normalized trade columns: {missing}")
    out = pd.DataFrame()
    out["trade_id"] = np.arange(len(data), dtype=np.int64)
    out["timestamp_ms"] = pd.to_numeric(data["timestamp_ms"], errors="coerce")
    out["price"] = pd.to_numeric(data["price"], errors="coerce")
    out["size"] = pd.to_numeric(data["size"], errors="coerce")
    out["quote_size"] = out["price"] * out["size"]
    out["side"] = data["side"].astype(str).str.lower()
    out["source_type"] = "normalized_binance_trades"
    return out


def canonicalize_trades(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["timestamp_ms", "price", "size", "quote_size"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[np.isfinite(out["timestamp_ms"]) & np.isfinite(out["price"]) & np.isfinite(out["size"])]
    out = out[(out["price"] > 0) & (out["size"] > 0)].copy()
    out["timestamp_ms"] = out["timestamp_ms"].round().astype("int64")
    out["minute_timestamp_ms"] = (out["timestamp_ms"] // 60000) * 60000
    out["side"] = out["side"].astype(str).str.lower()
    out["quote_size"] = out["quote_size"].fillna(out["price"] * out["size"])
    return out.sort_values(["timestamp_ms", "trade_id"], kind="mergesort").reset_index(drop=True)


def aggregate_trade_flow(trades: pd.DataFrame, candles: pd.DataFrame | None = None, windows: list[int] | None = None, large_window_minutes: int = 1440, large_quantile: float = 0.95) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    windows = windows or FEATURE_WINDOWS
    data = canonicalize_trades(trades)
    if data.empty:
        return pd.DataFrame(), feature_contract(windows, large_window_minutes, large_quantile)
    minute = data["minute_timestamp_ms"]
    buy = data["side"] == "buy"
    sell = data["side"] == "sell"
    grouped = data.groupby(minute, sort=True)
    out = pd.DataFrame({"timestamp_ms": sorted(grouped.groups.keys())})
    out = out.set_index("timestamp_ms")
    out["trade_count_1m"] = grouped.size()
    out["base_volume_1m"] = grouped["size"].sum()
    out["quote_volume_1m"] = grouped["quote_size"].sum()
    out["aggressive_buy_base_volume_1m"] = data.loc[buy].groupby("minute_timestamp_ms")["size"].sum()
    out["aggressive_sell_base_volume_1m"] = data.loc[sell].groupby("minute_timestamp_ms")["size"].sum()
    out["aggressive_buy_quote_volume_1m"] = data.loc[buy].groupby("minute_timestamp_ms")["quote_size"].sum()
    out["aggressive_sell_quote_volume_1m"] = data.loc[sell].groupby("minute_timestamp_ms")["quote_size"].sum()
    out = out.fillna(0.0)
    out["signed_base_volume_1m"] = out["aggressive_buy_base_volume_1m"] - out["aggressive_sell_base_volume_1m"]
    out["signed_quote_volume_1m"] = out["aggressive_buy_quote_volume_1m"] - out["aggressive_sell_quote_volume_1m"]
    out["trade_flow_imbalance_1m"] = out["signed_base_volume_1m"] / out["base_volume_1m"].replace(0, np.nan)
    out["aggressive_buy_fraction_1m"] = out["aggressive_buy_base_volume_1m"] / out["base_volume_1m"].replace(0, np.nan)
    out["trade_vwap_1m"] = out["quote_volume_1m"] / out["base_volume_1m"].replace(0, np.nan)
    out["first_trade_price"] = grouped["price"].first()
    out["last_trade_price"] = grouped["price"].last()
    out["high_trade_price"] = grouped["price"].max()
    out["low_trade_price"] = grouped["price"].min()
    out["intraminute_trade_range_bps"] = 10000.0 * np.log(out["high_trade_price"] / out["low_trade_price"].replace(0, np.nan))
    out["first_to_last_trade_return_bps"] = 10000.0 * np.log(out["last_trade_price"] / out["first_trade_price"].replace(0, np.nan))
    out["mean_trade_size"] = grouped["size"].mean()
    out["median_trade_size"] = grouped["size"].median()
    out["maximum_trade_size"] = grouped["size"].max()
    out["maximum_trade_quote_size"] = grouped["quote_size"].max()
    out["intraminute_trade_realized_volatility_bps"] = grouped["price"].apply(lambda s: realized_volatility_bps(s))
    threshold = out["maximum_trade_quote_size"].rolling(large_window_minutes, min_periods=max(30, min(large_window_minutes, 120))).quantile(large_quantile).shift(1)
    out["large_trade_quote_threshold"] = threshold
    data = data.merge(threshold.rename("large_trade_quote_threshold"), left_on="minute_timestamp_ms", right_index=True, how="left")
    large = data["quote_size"] >= data["large_trade_quote_threshold"]
    out["large_trade_count_1m"] = data.loc[large].groupby("minute_timestamp_ms").size()
    out["large_trade_quote_volume_1m"] = data.loc[large].groupby("minute_timestamp_ms")["quote_size"].sum()
    out["large_aggressive_buy_quote_volume_1m"] = data.loc[large & buy].groupby("minute_timestamp_ms")["quote_size"].sum()
    out["large_aggressive_sell_quote_volume_1m"] = data.loc[large & sell].groupby("minute_timestamp_ms")["quote_size"].sum()
    out = out.fillna({"large_trade_count_1m": 0.0, "large_trade_quote_volume_1m": 0.0, "large_aggressive_buy_quote_volume_1m": 0.0, "large_aggressive_sell_quote_volume_1m": 0.0})
    out["large_signed_quote_volume_1m"] = out["large_aggressive_buy_quote_volume_1m"] - out["large_aggressive_sell_quote_volume_1m"]
    if candles is not None and not candles.empty:
        c = candles[["timestamp_ms", "close"]].copy()
        c["timestamp_ms"] = pd.to_numeric(c["timestamp_ms"], errors="coerce").round().astype("Int64")
        out = out.reset_index().merge(c.dropna(subset=["timestamp_ms"]), on="timestamp_ms", how="left").set_index("timestamp_ms")
        out["close_to_trade_vwap_bps"] = 10000.0 * np.log(pd.to_numeric(out["close"], errors="coerce") / out["trade_vwap_1m"])
        out["last_trade_to_close_bps"] = 10000.0 * np.log(pd.to_numeric(out["close"], errors="coerce") / out["last_trade_price"])
    else:
        out["close_to_trade_vwap_bps"] = np.nan
        out["last_trade_to_close_bps"] = np.nan
    add_rolling_trade_flow_features(out, windows)
    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    return out, feature_contract(windows, large_window_minutes, large_quantile)


def realized_volatility_bps(prices: pd.Series) -> float:
    vals = pd.to_numeric(prices, errors="coerce").dropna()
    if len(vals) < 2:
        return 0.0
    returns = 10000.0 * np.log(vals / vals.shift(1))
    return float(returns.dropna().std(ddof=0)) if returns.notna().any() else 0.0


def add_rolling_trade_flow_features(out: pd.DataFrame, windows: list[int]) -> None:
    for window in windows:
        minp = window
        out[f"trade_count_sum_{window}m"] = out["trade_count_1m"].rolling(window, min_periods=minp).sum()
        out[f"trade_count_mean_{window}m"] = out["trade_count_1m"].rolling(window, min_periods=minp).mean()
        out[f"quote_volume_sum_{window}m"] = out["quote_volume_1m"].rolling(window, min_periods=minp).sum()
        out[f"signed_quote_volume_sum_{window}m"] = out["signed_quote_volume_1m"].rolling(window, min_periods=minp).sum()
        out[f"trade_flow_imbalance_mean_{window}m"] = out["trade_flow_imbalance_1m"].rolling(window, min_periods=minp).mean()
        out[f"trade_flow_imbalance_change_{window}m"] = out["trade_flow_imbalance_1m"] - out["trade_flow_imbalance_1m"].shift(window)
        out[f"signed_quote_volume_acceleration_{window}m"] = out["signed_quote_volume_1m"].diff().rolling(window, min_periods=minp).mean()
        out[f"trade_count_acceleration_{window}m"] = out["trade_count_1m"].diff().rolling(window, min_periods=minp).mean()
        out[f"cumulative_delta_{window}m"] = out["signed_base_volume_1m"].rolling(window, min_periods=minp).sum()
        out[f"positive_pressure_persistence_{window}m"] = (out["signed_quote_volume_1m"] > 0).astype(float).rolling(window, min_periods=minp).mean()
        out[f"negative_pressure_persistence_{window}m"] = (out["signed_quote_volume_1m"] < 0).astype(float).rolling(window, min_periods=minp).mean()
        shifted_mean = out["trade_count_1m"].rolling(window, min_periods=minp).mean().shift(1)
        shifted_std = out["trade_count_1m"].rolling(window, min_periods=minp).std(ddof=0).shift(1)
        out[f"trade_activity_zscore_{window}m"] = (out["trade_count_1m"] - shifted_mean) / shifted_std.replace(0, np.nan)


def feature_contract(windows: list[int], large_window_minutes: int, large_quantile: float) -> list[dict[str, Any]]:
    rows = [
        {"feature_name": "trade_count_1m", "formula": "count(trades in completed minute)", "source_columns": "timestamp_ms", "lookback": 1, "warmup": 0, "leakage_status": "causal"},
        {"feature_name": "signed_quote_volume_1m", "formula": "aggressive_buy_quote_volume_1m - aggressive_sell_quote_volume_1m", "source_columns": "price,size,side", "lookback": 1, "warmup": 0, "leakage_status": "causal"},
        {"feature_name": "large_trade_quote_threshold", "formula": f"shifted rolling {large_quantile:.2f} quantile of maximum_trade_quote_size over {large_window_minutes} minutes", "source_columns": "price,size", "lookback": large_window_minutes, "warmup": min(large_window_minutes, 120), "leakage_status": "shifted_causal"},
    ]
    for window in windows:
        rows.append({"feature_name": f"trade_count_sum_{window}m", "formula": f"rolling sum trade_count_1m over {window} completed minutes", "source_columns": "trade_count_1m", "lookback": window, "warmup": window, "leakage_status": "causal"})
        rows.append({"feature_name": f"trade_activity_zscore_{window}m", "formula": f"current trade count vs shifted rolling distribution over {window} minutes", "source_columns": "trade_count_1m", "lookback": window, "warmup": window + 1, "leakage_status": "shifted_causal"})
    return rows
