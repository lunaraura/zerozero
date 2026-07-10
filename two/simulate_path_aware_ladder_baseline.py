#!/usr/bin/env python3
"""Paper-only ladder/grid baselines with optional frozen rawseq path gating."""

from __future__ import annotations

import json
import math
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = Path(os.getenv("LADDER_SOURCE_PATH", PROJECT_ROOT / "data" / "realtime" / "kraken" / "SOLUSDT_10s_flow.csv"))
OUTPUT_ROOT = Path(os.getenv("LADDER_OUTPUT_DIR", PROJECT_ROOT / "data" / "research" / "ladder_baselines"))
PRICE_COLUMN_ENV = os.getenv("LADDER_PRICE_COLUMN", "").strip()
BUCKET_SECONDS = int(float(os.getenv("LADDER_BUCKET_SECONDS", "10")))
ANCHOR_MODE = os.getenv("LADDER_ANCHOR_MODE", "ema").strip().lower()
ANCHOR_WINDOWS = [int(float(x)) for x in os.getenv("LADDER_ANCHOR_WINDOWS", "60,150,300").split(",") if x.strip()]
VOL_WINDOWS = [int(float(x)) for x in os.getenv("LADDER_VOL_WINDOWS", "60,150,300").split(",") if x.strip()]
SPACING_MODE = os.getenv("LADDER_SPACING_MODE", "volatility_scaled").strip().lower()
MIN_SPACING_BPS = float(os.getenv("LADDER_MIN_SPACING_BPS", "10"))
MIN_SPACING_GRID_BPS = [float(x) for x in os.getenv("LADDER_MIN_SPACING_GRID_BPS", "10,20,40,80").split(",") if x.strip()]
MAX_SPACING_BPS = float(os.getenv("LADDER_MAX_SPACING_BPS", "80"))
VOL_MULTS = [float(x) for x in os.getenv("LADDER_VOL_MULTS", "1.0,1.5,2.0").split(",") if x.strip()]
SPACING_VOL_MULTS = [float(x) for x in os.getenv("LADDER_SPACING_VOL_MULTS", os.getenv("LADDER_VOL_MULTS", "1.0,1.5,2.0")).split(",") if x.strip()]
RUNG_COUNTS = [int(float(x)) for x in os.getenv("LADDER_RUNG_COUNTS", "3,5,8").split(",") if x.strip()]
TAKE_PROFIT_MULTS = [float(x) for x in os.getenv("LADDER_TAKE_PROFIT_MULTS", "1.0,1.25,1.5").split(",") if x.strip()]
TAKE_PROFIT_SPACING_MULTS = [float(x) for x in os.getenv("LADDER_TAKE_PROFIT_SPACING_MULTS", os.getenv("LADDER_TAKE_PROFIT_MULTS", "1.0,1.25,1.5")).split(",") if x.strip()]
DEFAULT_COST_BPS = float(os.getenv("LADDER_COST_BPS", "0.1"))
MAX_OPEN_UNITS = int(float(os.getenv("LADDER_MAX_OPEN_UNITS", "5")))
MAX_OPEN_UNITS_GRID = [int(float(x)) for x in os.getenv("LADDER_MAX_OPEN_UNITS_GRID", "1,2,3,5").split(",") if x.strip()]
STOP_LOSS_BPS_GRID = [float(x) for x in os.getenv("LADDER_STOP_LOSS_BPS_GRID", "40,80,120,200").split(",") if x.strip()]
STOP_LOSS_VOL_MULTS = [float(x) for x in os.getenv("LADDER_STOP_LOSS_VOL_MULTS", "4,8,12").split(",") if x.strip()]
MIN_SPACING_FLOOR_MODES = [x.strip().lower() for x in os.getenv("LADDER_MIN_SPACING_FLOOR_MODES", "fixed_bps").split(",") if x.strip()]
SPACING_SPREAD_MULTIPLE = float(os.getenv("LADDER_SPACING_SPREAD_MULTIPLE", "1.0"))
SPACING_TICK_MULTIPLE = float(os.getenv("LADDER_SPACING_TICK_MULTIPLE", "1.0"))
TICK_SIZE_ENV = os.getenv("LADDER_TICK_SIZE", "").strip()
TICK_SIZE = float(TICK_SIZE_ENV) if TICK_SIZE_ENV else math.nan
MAX_HOLD_BUCKETS_GRID = [int(float(x)) for x in os.getenv("LADDER_MAX_HOLD_BUCKETS_GRID", "60,180,360").split(",") if x.strip()]
COOLDOWN_AFTER_STOP_BUCKETS = int(float(os.getenv("LADDER_COOLDOWN_AFTER_STOP_BUCKETS", "30")))
DISABLE_BUYS_WHEN_PRICE_BELOW_ANCHOR_BPS = float(os.getenv("LADDER_DISABLE_BUYS_WHEN_PRICE_BELOW_ANCHOR_BPS", "120"))
DISABLE_BUYS_WHEN_EMA_SLOPE_BELOW_BPS = float(os.getenv("LADDER_DISABLE_BUYS_WHEN_EMA_SLOPE_BELOW_BPS", "-5"))
FORCE_FLAT_AT_END = os.getenv("LADDER_FORCE_FLAT_AT_END", "true").strip().lower() in {"1", "true", "yes", "y"}
REBALANCE_INTERVAL_BUCKETS = int(float(os.getenv("LADDER_REBALANCE_INTERVAL_BUCKETS", "180")))
REBALANCE_MODE = os.getenv("LADDER_REBALANCE_MODE", "fixed_interval").strip().lower()
REBALANCE_ANCHOR_DRIFT_BPS = float(os.getenv("LADDER_REBALANCE_ANCHOR_DRIFT_BPS", "20"))
REBALANCE_VOL_CHANGE_FRACTION = float(os.getenv("LADDER_REBALANCE_VOL_CHANGE_FRACTION", "0.25"))
MAX_DRAWDOWN_BPS = os.getenv("LADDER_MAX_DRAWDOWN_BPS", "").strip()
MAX_DRAWDOWN_BPS = float(MAX_DRAWDOWN_BPS) if MAX_DRAWDOWN_BPS else math.nan
RANGE_BREAK_BUFFER_BPS = float(os.getenv("LADDER_RANGE_BREAK_BUFFER_BPS", "20"))
SHADOW_DIR_ENV = os.getenv("LADDER_SHADOW_DIR", "").strip()
MODEL_GATE_MODE = os.getenv("LADDER_MODEL_GATE_MODE", "none").strip().lower()
MODEL_MIN_UPSIDE_ENV = os.getenv("LADDER_MODEL_MIN_UPSIDE_BPS", "").strip()
MODEL_MAX_DOWNSIDE_MULT = float(os.getenv("LADDER_MODEL_MAX_DOWNSIDE_MULT", "2.0"))
MAX_ROWS_ENV = os.getenv("LADDER_MAX_ROWS", "").strip()
MAX_CONFIGS_ENV = os.getenv("LADDER_MAX_CONFIGS", "").strip()
PRICE_CANDIDATES = ["price", "mid_price", "close", "last"]
SENSITIVITY_COSTS = [0.05, 0.1, 0.25]


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not math.isfinite(value) else value
    return value


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def load_price_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"LADDER_SOURCE_PATH not found: {path}")
    header = pd.read_csv(path, nrows=0)
    price_col = PRICE_COLUMN_ENV or first_existing(list(header.columns), PRICE_CANDIDATES)
    if not price_col:
        raise SystemExit(f"No price column found in {path}. Tried: {PRICE_CANDIDATES}")
    optional_cols = ["spread_percent", "spread_bps", "best_bid", "best_ask", "bid", "ask"]
    usecols = [column for column in ["timestamp", "time", price_col, *optional_cols] if column in header.columns]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame = frame.rename(columns={price_col: "price"})
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    for column in optional_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[np.isfinite(frame["price"]) & (frame["price"] > 0)].copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    else:
        frame["timestamp"] = np.arange(len(frame), dtype=np.float64) * BUCKET_SECONDS * 1000.0
    frame = frame.dropna(subset=["timestamp", "price"]).sort_values("timestamp").reset_index(drop=True)
    if MAX_ROWS_ENV:
        frame = frame.tail(int(float(MAX_ROWS_ENV))).reset_index(drop=True)
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    return frame


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def transform(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler.get("mean", 0.0), dtype=np.float64)
    std = np.asarray(scaler.get("std", 1.0), dtype=np.float64)
    std = np.where(np.abs(std) < 1e-12, 1.0, std)
    return (values - mean) / std


def unscale(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler.get("mean", 0.0), dtype=np.float64)
    std = np.asarray(scaler.get("std", 1.0), dtype=np.float64)
    return values * std + mean


def model_forward(weights: dict[str, Any], x: np.ndarray) -> np.ndarray:
    w1 = np.asarray(weights["W1"], dtype=np.float64)
    b1 = np.asarray(weights["b1"], dtype=np.float64)
    w2 = np.asarray(weights["W2"], dtype=np.float64)
    b2 = np.asarray(weights["b2"], dtype=np.float64)
    w3 = np.asarray(weights["W3"], dtype=np.float64)
    b3 = np.asarray(weights["b3"], dtype=np.float64)
    a1 = np.tanh(x @ w1 + b1)
    a2 = np.tanh(a1 @ w2 + b2)
    return a2 @ w3 + b3


def shadow_model_path(shadow_dir: Path) -> Path:
    direct = shadow_dir / "model.json"
    if direct.exists():
        return direct
    provenance = load_json(shadow_dir / "provenance.json")
    path = Path(str(provenance.get("model_path", "")))
    return path if path.exists() else direct


def build_shadow_gate(frame: pd.DataFrame, shadow_dir: Path) -> pd.DataFrame:
    model_path = shadow_model_path(shadow_dir)
    payload = load_json(model_path)
    if not payload:
        raise SystemExit(f"Could not load frozen shadow model: {model_path}")
    provenance = load_json(shadow_dir / "provenance.json")
    provenance_contract = provenance.get("contract") if isinstance(provenance.get("contract"), dict) else {}
    provenance_model_contract = provenance.get("model_contract") if isinstance(provenance.get("model_contract"), dict) else {}

    seq_len = int(payload.get("seq_len") or provenance_contract.get("seq_len") or provenance_model_contract.get("seq_len") or payload.get("architecture", {}).get("input_dim", 60))
    input_stride = int(payload.get("input_stride") or provenance_contract.get("input_stride") or provenance_model_contract.get("input_stride") or 1)
    input_feature = str(payload.get("input_feature") or provenance_contract.get("input_feature") or provenance_model_contract.get("input_feature") or "return").strip().lower()
    ma_window = int(float(payload.get("ma_window") or provenance_contract.get("ma_window") or provenance_model_contract.get("ma_window") or 0))
    prices = frame["price"].to_numpy(dtype=np.float64)

    if input_feature in {"return", "bucket_return", "signed_return", "signed_bucket_return_bps"}:
        values = np.zeros(len(prices), dtype=np.float64)
        values[1:] = 10_000.0 * np.log(prices[1:] / prices[:-1])
        values[~np.isfinite(values)] = 0.0
    elif input_feature in {"ma_distance", "price_vs_ma", "distance_to_ma"}:
        ma = pd.Series(prices).rolling(ma_window, min_periods=ma_window).mean().to_numpy(dtype=np.float64)
        values = 10_000.0 * np.log(prices / ma)
        values[~np.isfinite(values)] = np.nan
    elif input_feature in {"ma_slope", "slope_ma"}:
        ma = pd.Series(prices).rolling(ma_window, min_periods=ma_window).mean().to_numpy(dtype=np.float64)
        values = np.zeros(len(ma), dtype=np.float64)
        values[1:] = 10_000.0 * np.log(ma[1:] / ma[:-1])
        values[~np.isfinite(values)] = np.nan
    else:
        raise SystemExit(f"Unsupported frozen shadow input_feature={input_feature}")

    offsets = np.arange(seq_len - 1, -1, -1, dtype=np.int64) * input_stride
    start_i = max(seq_len, int(offsets[0]))
    rows = []
    indexes = []
    for i in range(start_i, len(frame)):
        x = values[i - offsets]
        if np.isfinite(x).all():
            rows.append(x)
            indexes.append(i)
    gate = pd.DataFrame(
        {
            "pred_final": np.nan,
            "pred_max_up": np.nan,
            "pred_max_down": np.nan,
            "pred_range": np.nan,
        },
        index=frame.index,
    )
    if not rows:
        return gate
    x = np.vstack(rows)
    expected = int(payload.get("architecture", {}).get("input_dim", x.shape[1]))
    if x.shape[1] != expected:
        raise SystemExit(f"Shadow gate input dim mismatch: built {x.shape[1]} expected {expected}")
    pred_scaled = model_forward(payload["weights"], transform(x, payload.get("x_scaler", {})))
    pred = unscale(pred_scaled, payload.get("y_scaler", {}))
    gate.loc[indexes, "pred_final"] = pred[:, -1]
    gate.loc[indexes, "pred_max_up"] = np.nanmax(pred, axis=1)
    gate.loc[indexes, "pred_max_down"] = np.nanmin(pred, axis=1)
    gate.loc[indexes, "pred_range"] = np.nanmax(pred, axis=1) - np.nanmin(pred, axis=1)
    return gate


def model_gate_allows(mode: str, pred_final: float, pred_max_up: float, pred_max_down: float, take_profit_bps: float) -> bool:
    if mode in {"", "none"}:
        return True
    if not all(math.isfinite(x) for x in [pred_final, pred_max_up, pred_max_down]):
        return False
    min_upside = float(MODEL_MIN_UPSIDE_ENV) if MODEL_MIN_UPSIDE_ENV else take_profit_bps
    downside_floor = -MODEL_MAX_DOWNSIDE_MULT * max(take_profit_bps, 1e-9)
    if mode == "pred_final_positive":
        return pred_final > 0
    if mode == "pred_max_up_gt_take_profit":
        return pred_max_up >= min_upside
    if mode == "pred_downside_limited":
        return pred_max_down >= downside_floor
    if mode == "pred_max_up_gt_take_profit_and_downside_limited":
        return pred_max_up >= min_upside and pred_max_down >= downside_floor
    raise SystemExit(f"Unsupported LADDER_MODEL_GATE_MODE={mode}")


def precompute_indicators(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    price = frame["price"].to_numpy(dtype=np.float64)
    returns = np.zeros(len(price), dtype=np.float64)
    returns[1:] = 10_000.0 * np.log(price[1:] / price[:-1])
    series = pd.Series(price)
    ret_series = pd.Series(returns)
    indicators = {}
    for window in sorted(set(ANCHOR_WINDOWS)):
        if ANCHOR_MODE == "rolling_ma":
            indicators[f"anchor_{window}"] = series.rolling(window, min_periods=window).mean().to_numpy(dtype=np.float64)
        elif ANCHOR_MODE == "ema":
            indicators[f"anchor_{window}"] = series.ewm(span=window, adjust=False, min_periods=window).mean().to_numpy(dtype=np.float64)
        else:
            raise SystemExit(f"Unsupported LADDER_ANCHOR_MODE={ANCHOR_MODE}")
    for window in sorted(set(VOL_WINDOWS)):
        indicators[f"vol_{window}"] = ret_series.rolling(window, min_periods=window).std().to_numpy(dtype=np.float64)
    return indicators


def max_drawdown(values: list[float]) -> float:
    if not values:
        return math.nan
    arr = np.asarray(values, dtype=np.float64)
    peak = np.maximum.accumulate(arr)
    return float(np.min(arr - peak))


def count_map_text(counts: dict[int, int]) -> str:
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def min_spacing_floor_series(frame: pd.DataFrame, price: np.ndarray, min_spacing_bps: float, floor_mode: str) -> np.ndarray:
    fixed = np.full(len(price), min_spacing_bps, dtype=np.float64)
    floor_mode = floor_mode.strip().lower()
    if floor_mode == "fixed_bps":
        return fixed

    spread_bps = np.full(len(price), np.nan, dtype=np.float64)
    if "spread_bps" in frame.columns:
        spread_bps = frame["spread_bps"].to_numpy(dtype=np.float64)
    elif "spread_percent" in frame.columns:
        spread_bps = frame["spread_percent"].to_numpy(dtype=np.float64) * 100.0
    elif "best_bid" in frame.columns and "best_ask" in frame.columns:
        bid = frame["best_bid"].to_numpy(dtype=np.float64)
        ask = frame["best_ask"].to_numpy(dtype=np.float64)
        mid = (bid + ask) / 2.0
        spread_bps = np.where(mid > 0, 10_000.0 * (ask - bid) / mid, np.nan)
    elif "bid" in frame.columns and "ask" in frame.columns:
        bid = frame["bid"].to_numpy(dtype=np.float64)
        ask = frame["ask"].to_numpy(dtype=np.float64)
        mid = (bid + ask) / 2.0
        spread_bps = np.where(mid > 0, 10_000.0 * (ask - bid) / mid, np.nan)
    spread_floor = SPACING_SPREAD_MULTIPLE * spread_bps
    spread_floor = np.where(np.isfinite(spread_floor) & (spread_floor > 0), spread_floor, min_spacing_bps)

    tick_floor = np.full(len(price), min_spacing_bps, dtype=np.float64)
    if math.isfinite(TICK_SIZE) and TICK_SIZE > 0:
        tick_bps = np.where(price > 0, 10_000.0 * np.log((price + TICK_SIZE) / price), np.nan)
        tick_floor = SPACING_TICK_MULTIPLE * tick_bps
        tick_floor = np.where(np.isfinite(tick_floor) & (tick_floor > 0), tick_floor, min_spacing_bps)

    if floor_mode in {"spread_multiple", "max_fixed_or_spread"}:
        return np.maximum(fixed, spread_floor)
    if floor_mode == "tick_multiple":
        return np.maximum(fixed, tick_floor)
    return fixed


def simulate_config(
    frame: pd.DataFrame,
    indicators: dict[str, np.ndarray],
    gate: pd.DataFrame | None,
    contract: dict[str, Any],
    save_paths: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    price = frame["price"].to_numpy(dtype=np.float64)
    timestamp = frame["timestamp"].to_numpy(dtype=np.float64)
    anchor = indicators[f"anchor_{contract['anchor_window']}"]
    vol = indicators[f"vol_{contract['vol_window']}"]
    if SPACING_MODE != "volatility_scaled":
        raise SystemExit(f"Unsupported LADDER_SPACING_MODE={SPACING_MODE}")
    min_spacing_bps = float(contract.get("min_spacing_bps", MIN_SPACING_BPS))
    min_spacing_floor_mode = str(contract.get("min_spacing_floor_mode", "fixed_bps")).strip().lower()
    spacing_vol_mult = float(contract.get("spacing_vol_mult", contract.get("vol_mult", 1.0)))
    stop_loss_vol_mult = float(contract.get("stop_loss_vol_mult", 8.0))
    take_profit_spacing_mult = float(contract.get("take_profit_spacing_mult", contract.get("take_profit_mult", 1.0)))
    max_open_units = int(contract.get("max_open_units", MAX_OPEN_UNITS))
    min_stop_floor_bps = float(contract.get("min_stop_floor_bps", contract.get("stop_loss_bps", 80.0)))
    max_hold_buckets = int(contract.get("max_hold_buckets", 180))
    force_flat = bool(contract.get("force_flat_at_end", FORCE_FLAT_AT_END))
    rebalance_interval_buckets = max(1, int(float(contract.get("rebalance_interval_buckets", REBALANCE_INTERVAL_BUCKETS))))
    rebalance_mode = str(contract.get("rebalance_mode", REBALANCE_MODE)).strip().lower()
    rebalance_anchor_drift_bps = float(contract.get("rebalance_anchor_drift_bps", REBALANCE_ANCHOR_DRIFT_BPS))
    rebalance_vol_change_fraction = float(contract.get("rebalance_vol_change_fraction", REBALANCE_VOL_CHANGE_FRACTION))
    allowed_rebalance_modes = {"fixed_interval", "anchor_drift_trigger", "volatility_trigger", "interval_or_trigger"}
    if rebalance_mode not in allowed_rebalance_modes:
        raise SystemExit(f"Unsupported LADDER_REBALANCE_MODE={rebalance_mode}")
    realized_vol_bps = np.asarray(vol, dtype=np.float64)
    min_spacing_floor = min_spacing_floor_series(frame, price, min_spacing_bps, min_spacing_floor_mode)
    spacing_vol_component = spacing_vol_mult * realized_vol_bps
    spacing = np.minimum(np.maximum(min_spacing_floor, spacing_vol_component), MAX_SPACING_BPS)
    stop_loss = np.maximum(min_stop_floor_bps, stop_loss_vol_mult * realized_vol_bps)
    take_profit = take_profit_spacing_mult * spacing
    pred_final = gate["pred_final"].to_numpy(dtype=np.float64) if gate is not None else np.full(len(frame), np.nan)
    pred_max_up = gate["pred_max_up"].to_numpy(dtype=np.float64) if gate is not None else np.full(len(frame), np.nan)
    pred_max_down = gate["pred_max_down"].to_numpy(dtype=np.float64) if gate is not None else np.full(len(frame), np.nan)

    open_entries: list[dict[str, float]] = []
    trades = []
    equity_rows = []
    realized_gross = 0.0
    realized_net = 0.0
    equity_curve = []
    exposure_steps = 0
    max_inventory = 0
    halted_by_drawdown = False
    last_price = price[0] if len(price) else math.nan
    valid_spacing = spacing[np.isfinite(spacing)]
    spacing_source = spacing_vol_component
    valid_source = spacing_source[np.isfinite(spacing_source)]
    spacing_clamp_max_fraction = float(np.mean(valid_source >= MAX_SPACING_BPS)) if len(valid_source) else math.nan
    floor_mask = np.isfinite(spacing_source) & np.isfinite(min_spacing_floor)
    spacing_clamp_min_fraction = float(np.mean(spacing_source[floor_mask] <= min_spacing_floor[floor_mask])) if np.any(floor_mask) else math.nan
    floor_used_fraction = spacing_clamp_min_fraction
    volatility_used_fraction = float(np.mean(spacing_source[floor_mask] > min_spacing_floor[floor_mask])) if np.any(floor_mask) else math.nan
    vol_ratio_mask = np.isfinite(realized_vol_bps) & (realized_vol_bps > 0) & np.isfinite(spacing) & np.isfinite(stop_loss)
    avg_realized_vol_bps = float(np.mean(realized_vol_bps[np.isfinite(realized_vol_bps)])) if np.any(np.isfinite(realized_vol_bps)) else math.nan
    avg_spacing_to_vol_ratio = float(np.mean(spacing[vol_ratio_mask] / realized_vol_bps[vol_ratio_mask])) if np.any(vol_ratio_mask) else math.nan
    avg_stop_to_vol_ratio = float(np.mean(stop_loss[vol_ratio_mask] / realized_vol_bps[vol_ratio_mask])) if np.any(vol_ratio_mask) else math.nan
    rung_trigger_by_rung = {rung: 0 for rung in range(1, contract["rung_count"] + 1)}
    buy_count_by_rung = {rung: 0 for rung in range(1, contract["rung_count"] + 1)}
    sell_count_by_rung = {rung: 0 for rung in range(1, contract["rung_count"] + 1)}
    anchor_cross_count = 0
    range_break_down_count = 0
    range_break_up_count = 0
    no_new_buy_due_to_range_break_count = 0
    no_new_buy_due_to_max_inventory_count = 0
    no_new_buy_due_to_gate_count = 0
    no_new_buy_due_to_no_rung_trigger_count = 0
    rows_seen = 0
    rows_with_model_prediction = 0
    rows_gate_passed = 0
    rows_rung_triggered = 0
    rows_buy_executed = 0
    take_profit_exit_count = 0
    stop_loss_exit_count = 0
    timeout_exit_count = 0
    final_liquidation_count = 0
    stop_loss_net_bps = 0.0
    timeout_net_bps = 0.0
    final_liquidation_net_bps = 0.0
    cooldown_block_count = 0
    trend_filter_block_count = 0
    anchor_distance_block_count = 0
    rebalance_count = 0
    anchor_drift_rebalance_count = 0
    volatility_rebalance_count = 0
    fixed_interval_rebalance_count = 0
    stale_ladder_entry_count = 0
    rebalance_blocked_count = 0
    rebalance_intervals: list[int] = []
    last_rebalance_i = -1
    ladder_anchor = math.nan
    ladder_spacing = math.nan
    ladder_take_profit = math.nan
    ladder_stop_loss = math.nan
    ladder_vol = math.nan
    ladder_rung_prices: dict[int, float] = {}
    cooldown_until_i = -1
    no_trade_reasons: dict[str, int] = {}

    def add_reason(reason: str) -> None:
        no_trade_reasons[reason] = no_trade_reasons.get(reason, 0) + 1

    def close_lot(unit: dict[str, float], exit_i: int, exit_price: float, exit_reason: str, tp_bps: float, stop_bps: float) -> float:
        nonlocal realized_gross, realized_net, take_profit_exit_count, stop_loss_exit_count
        nonlocal timeout_exit_count, final_liquidation_count, stop_loss_net_bps, timeout_net_bps
        nonlocal final_liquidation_net_bps, cooldown_until_i
        gross = 10_000.0 * math.log(exit_price / unit["entry_price"])
        net = gross - 2.0 * DEFAULT_COST_BPS
        realized_gross += gross
        realized_net += net
        if exit_reason == "take_profit":
            take_profit_exit_count += 1
        elif exit_reason == "stop_loss":
            stop_loss_exit_count += 1
            stop_loss_net_bps += net
            cooldown_until_i = max(cooldown_until_i, exit_i + COOLDOWN_AFTER_STOP_BUCKETS)
        elif exit_reason == "timeout":
            timeout_exit_count += 1
            timeout_net_bps += net
        elif exit_reason == "final_liquidation":
            final_liquidation_count += 1
            final_liquidation_net_bps += net
        trades.append(
            {
                "entry_timestamp": unit["entry_timestamp"],
                "exit_timestamp": timestamp[exit_i],
                "entry_bucket": int(unit["entry_bucket"]),
                "exit_bucket": exit_i,
                "hold_buckets": exit_i - int(unit["entry_bucket"]),
                "entry_price": unit["entry_price"],
                "exit_price": exit_price,
                "rung": int(unit["rung"]),
                "gross_bps": gross,
                "net_bps": net,
                "take_profit_bps": tp_bps,
                "stop_loss_bps": stop_bps,
                "max_hold_buckets": max_hold_buckets,
                "cost_bps": DEFAULT_COST_BPS,
                "exit_reason": exit_reason,
            }
        )
        return net

    def refresh_ladder(i: int, a: float, s: float, tp: float, sl: float, rv: float, reason: str) -> None:
        nonlocal rebalance_count, anchor_drift_rebalance_count, volatility_rebalance_count
        nonlocal fixed_interval_rebalance_count, last_rebalance_i, ladder_anchor, ladder_spacing
        nonlocal ladder_take_profit, ladder_stop_loss, ladder_vol, ladder_rung_prices
        if last_rebalance_i >= 0:
            rebalance_intervals.append(i - last_rebalance_i)
        rebalance_count += 1
        if reason == "fixed_interval":
            fixed_interval_rebalance_count += 1
        elif reason == "anchor_drift":
            anchor_drift_rebalance_count += 1
        elif reason == "volatility":
            volatility_rebalance_count += 1
        last_rebalance_i = i
        ladder_anchor = a
        ladder_spacing = s
        ladder_take_profit = tp
        ladder_stop_loss = sl
        ladder_vol = rv
        ladder_rung_prices = {
            rung: ladder_anchor * math.exp(-(rung * ladder_spacing) / 10_000.0)
            for rung in range(1, contract["rung_count"] + 1)
        }

    def rebalance_reason(i: int, a: float, rv: float) -> str:
        if last_rebalance_i < 0:
            return "fixed_interval"
        interval_due = i - last_rebalance_i >= rebalance_interval_buckets
        anchor_due = False
        if math.isfinite(ladder_anchor) and ladder_anchor > 0:
            anchor_due = abs(10_000.0 * math.log(a / ladder_anchor)) >= abs(rebalance_anchor_drift_bps)
        vol_due = False
        if math.isfinite(ladder_vol) and ladder_vol > 0 and math.isfinite(rv):
            vol_due = abs(rv - ladder_vol) / ladder_vol >= rebalance_vol_change_fraction
        elif math.isfinite(rv) and rv > 0 and not math.isfinite(ladder_vol):
            vol_due = True
        if rebalance_mode == "fixed_interval" and interval_due:
            return "fixed_interval"
        if rebalance_mode == "anchor_drift_trigger" and anchor_due:
            return "anchor_drift"
        if rebalance_mode == "volatility_trigger" and vol_due:
            return "volatility"
        if rebalance_mode == "interval_or_trigger":
            if interval_due:
                return "fixed_interval"
            if anchor_due:
                return "anchor_drift"
            if vol_due:
                return "volatility"
        return ""

    for i in range(1, len(price)):
        p = float(price[i])
        a = float(anchor[i])
        s = float(spacing[i])
        tp = float(take_profit[i])
        sl = float(stop_loss[i])
        rv = float(realized_vol_bps[i])
        if not all(math.isfinite(x) for x in [p, a, s, tp, sl, rv]) or s <= 0 or tp <= 0 or sl <= 0:
            if rebalance_reason(i, a, rv):
                rebalance_blocked_count += 1
            last_price = p
            continue
        reason = rebalance_reason(i, a, rv)
        if reason:
            refresh_ladder(i, a, s, tp, sl, rv, reason)
        rows_seen += 1
        has_model_prediction = gate is not None and all(math.isfinite(x) for x in [pred_final[i], pred_max_up[i], pred_max_down[i]])
        if has_model_prediction:
            rows_with_model_prediction += 1
        if last_price <= a < p or last_price >= a > p:
            anchor_cross_count += 1

        remaining = []
        for unit in open_entries:
            unit_tp = float(unit.get("take_profit_bps", tp))
            unit_sl = float(unit.get("stop_loss_bps", sl))
            exit_price = unit["entry_price"] * math.exp(unit_tp / 10_000.0)
            stop_price = unit["entry_price"] * math.exp(-unit_sl / 10_000.0)
            age = i - int(unit["entry_bucket"])
            if p <= stop_price:
                close_lot(unit, i, p, "stop_loss", unit_tp, unit_sl)
                rung_key = int(unit["rung"])
                sell_count_by_rung[rung_key] = sell_count_by_rung.get(rung_key, 0) + 1
            elif age >= max_hold_buckets:
                close_lot(unit, i, p, "timeout", unit_tp, unit_sl)
                rung_key = int(unit["rung"])
                sell_count_by_rung[rung_key] = sell_count_by_rung.get(rung_key, 0) + 1
            elif p >= exit_price:
                close_lot(unit, i, p, "take_profit", unit_tp, unit_sl)
                rung_key = int(unit["rung"])
                sell_count_by_rung[rung_key] = sell_count_by_rung.get(rung_key, 0) + 1
            else:
                remaining.append(unit)
        open_entries = remaining

        open_gross = sum(10_000.0 * math.log(p / unit["entry_price"]) for unit in open_entries)
        open_net = open_gross - DEFAULT_COST_BPS * len(open_entries)
        equity = realized_net + open_net
        equity_curve.append(equity)
        if save_paths:
            equity_rows.append(
                {
                    "timestamp": timestamp[i],
                    "price": p,
                    "anchor": ladder_anchor,
                    "spacing_bps": ladder_spacing,
                    "take_profit_bps": ladder_take_profit,
                    "inventory": len(open_entries),
                    "realized_net_bps": realized_net,
                    "unrealized_net_bps": open_net,
                    "equity_bps": equity,
                }
            )
        if len(open_entries) > 0:
            exposure_steps += 1
        if math.isfinite(MAX_DRAWDOWN_BPS) and equity - max(equity_curve) <= -abs(MAX_DRAWDOWN_BPS):
            halted_by_drawdown = True

        break_level = ladder_anchor * math.exp(-(contract["rung_count"] * ladder_spacing + RANGE_BREAK_BUFFER_BPS) / 10_000.0)
        upper_break_level = ladder_anchor * math.exp((contract["rung_count"] * ladder_spacing + RANGE_BREAK_BUFFER_BPS) / 10_000.0)
        if p < break_level:
            range_break_down_count += 1
        if p > upper_break_level:
            range_break_up_count += 1
        triggered_rungs = []
        for rung in range(1, contract["rung_count"] + 1):
            rung_price = ladder_rung_prices.get(rung, math.nan)
            if last_price > rung_price >= p:
                triggered_rungs.append(rung)
                rung_trigger_by_rung[rung] = rung_trigger_by_rung.get(rung, 0) + 1
        if triggered_rungs:
            rows_rung_triggered += 1
        anchor_distance_bps = 10_000.0 * math.log(p / a)
        anchor_slope_bps = 10_000.0 * math.log(a / anchor[i - 1]) if i > 0 and math.isfinite(anchor[i - 1]) and anchor[i - 1] > 0 else math.nan
        in_cooldown = i <= cooldown_until_i
        anchor_distance_block = anchor_distance_bps < -abs(DISABLE_BUYS_WHEN_PRICE_BELOW_ANCHOR_BPS)
        trend_filter_block = math.isfinite(anchor_slope_bps) and anchor_slope_bps < DISABLE_BUYS_WHEN_EMA_SLOPE_BELOW_BPS
        can_add = p >= break_level and not halted_by_drawdown and not in_cooldown and not anchor_distance_block and not trend_filter_block
        gate_ok = model_gate_allows(MODEL_GATE_MODE, pred_final[i], pred_max_up[i], pred_max_down[i], ladder_take_profit)
        if gate_ok:
            rows_gate_passed += 1
        bought = False
        if in_cooldown:
            if triggered_rungs:
                cooldown_block_count += 1
                add_reason("cooldown")
        elif anchor_distance_block:
            if triggered_rungs:
                anchor_distance_block_count += 1
                add_reason("anchor_distance")
        elif trend_filter_block:
            if triggered_rungs:
                trend_filter_block_count += 1
                add_reason("trend_filter")
        elif not can_add:
            if triggered_rungs:
                no_new_buy_due_to_range_break_count += 1
                add_reason("range_break")
        elif len(open_entries) >= max_open_units:
            if triggered_rungs:
                no_new_buy_due_to_max_inventory_count += 1
                add_reason("max_inventory")
        elif not gate_ok:
            if triggered_rungs:
                no_new_buy_due_to_gate_count += 1
                add_reason("gate")
        elif not triggered_rungs:
            no_new_buy_due_to_no_rung_trigger_count += 1
            add_reason("no_rung_trigger")
        else:
            for rung in triggered_rungs:
                if len(open_entries) >= max_open_units:
                    no_new_buy_due_to_max_inventory_count += 1
                    add_reason("max_inventory")
                    break
                open_entries.append(
                    {
                        "entry_price": p,
                        "entry_timestamp": timestamp[i],
                        "entry_bucket": float(i),
                        "rung": float(rung),
                        "take_profit_bps": ladder_take_profit,
                        "stop_loss_bps": ladder_stop_loss,
                        "ladder_rebalance_bucket": float(last_rebalance_i),
                    }
                )
                if i > last_rebalance_i:
                    stale_ladder_entry_count += 1
                buy_count_by_rung[rung] = buy_count_by_rung.get(rung, 0) + 1
                bought = True
            if bought:
                rows_buy_executed += 1
        max_inventory = max(max_inventory, len(open_entries))
        last_price = p

    final_i = len(price) - 1
    final_price = float(price[-1]) if len(price) else math.nan
    if force_flat and math.isfinite(final_price):
        final_tp = float(take_profit[final_i]) if len(take_profit) and math.isfinite(float(take_profit[final_i])) else math.nan
        final_sl = float(stop_loss[final_i]) if len(stop_loss) and math.isfinite(float(stop_loss[final_i])) else math.nan
        for unit in list(open_entries):
            close_lot(
                unit,
                final_i,
                final_price,
                "final_liquidation",
                float(unit.get("take_profit_bps", final_tp)),
                float(unit.get("stop_loss_bps", final_sl)),
            )
            rung_key = int(unit["rung"])
            sell_count_by_rung[rung_key] = sell_count_by_rung.get(rung_key, 0) + 1
        open_entries = []
    final_open_gross = sum(10_000.0 * math.log(final_price / unit["entry_price"]) for unit in open_entries) if math.isfinite(final_price) else 0.0
    final_open_net = final_open_gross - DEFAULT_COST_BPS * len(open_entries)
    force_flat_total_net = realized_net
    total_net = realized_net + final_open_net
    trade_nets = [trade["net_bps"] for trade in trades]
    gross_closed = sum(trade["gross_bps"] for trade in trades)
    sensitivity = {}
    for cost in SENSITIVITY_COSTS:
        sensitivity[f"total_net_cost_{str(cost).replace('.', '_')}_bps"] = gross_closed - 2.0 * cost * len(trades) + final_open_gross - cost * len(open_entries)

    result = {
        **contract,
        "spacing_vol_mult": spacing_vol_mult,
        "stop_loss_vol_mult": stop_loss_vol_mult,
        "take_profit_spacing_mult": take_profit_spacing_mult,
        "min_spacing_floor_mode": min_spacing_floor_mode,
        "min_stop_floor_bps": min_stop_floor_bps,
        "rebalance_interval_buckets": rebalance_interval_buckets,
        "rebalance_mode": rebalance_mode,
        "rebalance_anchor_drift_bps": rebalance_anchor_drift_bps,
        "rebalance_vol_change_fraction": rebalance_vol_change_fraction,
        "model_gate_mode": MODEL_GATE_MODE,
        "shadow_dir": str(SHADOW_DIR_ENV),
        "avg_spacing_bps": float(np.mean(valid_spacing)) if len(valid_spacing) else math.nan,
        "min_spacing_bps_observed": float(np.min(valid_spacing)) if len(valid_spacing) else math.nan,
        "max_spacing_bps_observed": float(np.max(valid_spacing)) if len(valid_spacing) else math.nan,
        "spacing_clamp_min_fraction": spacing_clamp_min_fraction,
        "spacing_clamp_max_fraction": spacing_clamp_max_fraction,
        "volatility_used_fraction": volatility_used_fraction,
        "floor_used_fraction": floor_used_fraction,
        "avg_realized_vol_bps": avg_realized_vol_bps,
        "avg_spacing_to_vol_ratio": avg_spacing_to_vol_ratio,
        "avg_stop_to_vol_ratio": avg_stop_to_vol_ratio,
        "rung_trigger_count_total": int(sum(rung_trigger_by_rung.values())),
        "rung_trigger_count_by_rung": count_map_text(rung_trigger_by_rung),
        "buy_count_by_rung": count_map_text(buy_count_by_rung),
        "sell_count_by_rung": count_map_text(sell_count_by_rung),
        "max_rung_reached": max([rung for rung, count in buy_count_by_rung.items() if count > 0], default=0),
        "anchor_cross_count": anchor_cross_count,
        "range_break_down_count": range_break_down_count,
        "range_break_up_count": range_break_up_count,
        "no_new_buy_due_to_range_break_count": no_new_buy_due_to_range_break_count,
        "no_new_buy_due_to_max_inventory_count": no_new_buy_due_to_max_inventory_count,
        "no_new_buy_due_to_gate_count": no_new_buy_due_to_gate_count,
        "no_new_buy_due_to_no_rung_trigger_count": no_new_buy_due_to_no_rung_trigger_count,
        "rows_seen": rows_seen,
        "rows_with_model_prediction": rows_with_model_prediction,
        "rows_gate_passed": rows_gate_passed,
        "rows_rung_triggered": rows_rung_triggered,
        "rows_buy_executed": rows_buy_executed,
        "gate_pass_fraction": rows_gate_passed / max(1, rows_seen),
        "most_common_no_trade_reason": max(no_trade_reasons.items(), key=lambda item: item[1])[0] if no_trade_reasons else "",
        "take_profit_exit_count": take_profit_exit_count,
        "stop_loss_exit_count": stop_loss_exit_count,
        "timeout_exit_count": timeout_exit_count,
        "final_liquidation_count": final_liquidation_count,
        "stop_loss_net_bps": stop_loss_net_bps,
        "timeout_net_bps": timeout_net_bps,
        "final_liquidation_net_bps": final_liquidation_net_bps,
        "cooldown_block_count": cooldown_block_count,
        "trend_filter_block_count": trend_filter_block_count,
        "anchor_distance_block_count": anchor_distance_block_count,
        "rebalance_count": rebalance_count,
        "avg_rebalance_interval_buckets": float(np.mean(rebalance_intervals)) if rebalance_intervals else math.nan,
        "anchor_drift_rebalance_count": anchor_drift_rebalance_count,
        "volatility_rebalance_count": volatility_rebalance_count,
        "fixed_interval_rebalance_count": fixed_interval_rebalance_count,
        "trades_per_rebalance": len(trades) / max(1, rebalance_count),
        "stale_ladder_entry_count": stale_ladder_entry_count,
        "rebalance_blocked_count": rebalance_blocked_count,
        "force_flat_total_net_bps": force_flat_total_net,
        "total_net_bps": total_net,
        "realized_net_bps": realized_net,
        "unrealized_net_bps": final_open_net,
        "max_drawdown_bps": max_drawdown(equity_curve),
        "number_of_trades": len(trades),
        "win_rate": float(np.mean(np.asarray(trade_nets) > 0)) if trade_nets else math.nan,
        "average_trade_net_bps": float(np.mean(trade_nets)) if trade_nets else math.nan,
        "max_inventory": max_inventory,
        "ending_inventory": len(open_entries),
        "exposure_time_fraction": exposure_steps / max(1, len(price)),
        "runaway_inventory_flag": max_inventory >= max_open_units and len(open_entries) >= max_open_units,
        "paper_only": True,
        "private_api": False,
        "orders": False,
        "training": False,
        "promotion": False,
        "champion_mutation": False,
        **sensitivity,
    }
    result["rank_score"] = rank_result(result)
    return result, pd.DataFrame(trades), pd.DataFrame(equity_rows)


def rank_result(row: dict[str, Any]) -> float:
    total = safe_float(row.get("force_flat_total_net_bps"), safe_float(row.get("total_net_bps"), 0.0))
    dd = abs(safe_float(row.get("max_drawdown_bps"), 0.0))
    trades = safe_float(row.get("number_of_trades"), 0.0)
    inv = safe_float(row.get("max_inventory"), 0.0)
    cost025 = safe_float(row.get("total_net_cost_0_25_bps"), -1e9)
    exposure = safe_float(row.get("exposure_time_fraction"), 0.0)
    stop_losses = safe_float(row.get("stop_loss_exit_count"), 0.0)
    score = total - 0.5 * dd + min(trades, 500.0) * 0.25 - max(0.0, inv - 3.0) * 25.0
    if safe_float(row.get("ending_inventory"), 0.0) > 0 and not bool(row.get("force_flat_at_end", FORCE_FLAT_AT_END)):
        score -= 10_000.0
    if exposure > 0.5:
        score -= (exposure - 0.5) * 1_000.0
    if stop_losses > max(5.0, trades * 0.35):
        score -= stop_losses * 10.0
    if total > 0 and safe_float(row.get("max_drawdown_bps"), 0.0) < -2.0 * total:
        score -= abs(safe_float(row.get("max_drawdown_bps"), 0.0)) * 0.5
    if bool(row.get("runaway_inventory_flag")):
        score -= 500.0
    if cost025 <= 0:
        score -= 250.0
    return score


def output_dir() -> Path:
    path = OUTPUT_ROOT / f"ladder_baseline_{stamp()}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def behavior_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("number_of_trades"),
        round(safe_float(row.get("total_net_bps"), 0.0), 8),
        round(safe_float(row.get("realized_net_bps"), 0.0), 8),
        round(safe_float(row.get("unrealized_net_bps"), 0.0), 8),
        row.get("max_inventory"),
        row.get("ending_inventory"),
        row.get("buy_count_by_rung"),
        row.get("sell_count_by_rung"),
        row.get("take_profit_exit_count"),
        row.get("stop_loss_exit_count"),
        row.get("timeout_exit_count"),
        row.get("final_liquidation_count"),
    )


def diagnostics_warnings(rows: list[dict[str, Any]]) -> list[str]:
    warnings = []
    signatures: dict[tuple[Any, ...], int] = {}
    for row in rows:
        signature = behavior_signature(row)
        signatures[signature] = signatures.get(signature, 0) + 1
        min_clamp = safe_float(row.get("spacing_clamp_min_fraction"), 0.0)
        max_clamp = safe_float(row.get("spacing_clamp_max_fraction"), 0.0)
        if max(min_clamp, max_clamp) > 0.80:
            warnings.append(
                f"spacing_clamped_over_80pct: anchor={row.get('anchor_window')} vol={row.get('vol_window')} "
                f"mult={row.get('spacing_vol_mult', row.get('vol_mult'))} min_frac={min_clamp:.3f} max_frac={max_clamp:.3f}"
            )
        if safe_float(row.get("max_rung_reached"), 0.0) <= 1 and safe_float(row.get("number_of_trades"), 0.0) > 0:
            warnings.append(
                f"only_rung_1_used: anchor={row.get('anchor_window')} vol={row.get('vol_window')} "
                f"mult={row.get('spacing_vol_mult', row.get('vol_mult'))} rungs={row.get('rung_count')}"
            )
        if row.get("model_gate_mode") not in {"", "none"} and safe_float(row.get("rows_gate_passed"), 0.0) == 0:
            warnings.append(
                f"model_gate_passes_zero_rows: mode={row.get('model_gate_mode')} "
                f"anchor={row.get('anchor_window')} vol={row.get('vol_window')} rungs={row.get('rung_count')}"
            )
    identical_max = max(signatures.values(), default=0)
    if rows and identical_max / len(rows) > 0.25:
        warnings.insert(0, f"many_configs_identical_trades_equity: largest_identical_group={identical_max} of {len(rows)}")
    return warnings


def write_diagnostics(path: Path, rows: list[dict[str, Any]], best: dict[str, Any]) -> None:
    warnings = diagnostics_warnings(rows)
    signature_counts: dict[tuple[Any, ...], int] = {}
    for row in rows:
        signature = behavior_signature(row)
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
    top_signatures = sorted(signature_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    lines = [
        "Ladder Diagnostics",
        "",
        f"Created at: {stamp()}",
        f"Configs: {len(rows)}",
        f"Model gate mode: {MODEL_GATE_MODE}",
        "",
        "Warnings:",
    ]
    if warnings:
        lines.extend(f"  {warning}" for warning in warnings[:50])
    else:
        lines.append("  none")
    lines.extend(["", "Most common behavior signatures:"])
    for signature, count in top_signatures:
        lines.append(f"  count={count} signature={signature}")
    lines.extend(
        [
            "",
            "Best config diagnostics:",
            f"  avg_spacing_bps={best.get('avg_spacing_bps')}",
            f"  spacing_clamp_min_fraction={best.get('spacing_clamp_min_fraction')}",
            f"  spacing_clamp_max_fraction={best.get('spacing_clamp_max_fraction')}",
            f"  volatility_used_fraction={best.get('volatility_used_fraction')}",
            f"  floor_used_fraction={best.get('floor_used_fraction')}",
            f"  avg_realized_vol_bps={best.get('avg_realized_vol_bps')}",
            f"  avg_spacing_to_vol_ratio={best.get('avg_spacing_to_vol_ratio')}",
            f"  avg_stop_to_vol_ratio={best.get('avg_stop_to_vol_ratio')}",
            f"  rebalance_count={best.get('rebalance_count')}",
            f"  avg_rebalance_interval_buckets={best.get('avg_rebalance_interval_buckets')}",
            f"  trades_per_rebalance={best.get('trades_per_rebalance')}",
            f"  stale_ladder_entry_count={best.get('stale_ladder_entry_count')}",
            f"  rung_trigger_count_by_rung={best.get('rung_trigger_count_by_rung')}",
            f"  buy_count_by_rung={best.get('buy_count_by_rung')}",
            f"  sell_count_by_rung={best.get('sell_count_by_rung')}",
            f"  max_rung_reached={best.get('max_rung_reached')}",
            f"  rows_seen={best.get('rows_seen')}",
            f"  rows_with_model_prediction={best.get('rows_with_model_prediction')}",
            f"  rows_gate_passed={best.get('rows_gate_passed')}",
            f"  rows_rung_triggered={best.get('rows_rung_triggered')}",
            f"  rows_buy_executed={best.get('rows_buy_executed')}",
            f"  gate_pass_fraction={best.get('gate_pass_fraction')}",
            f"  most_common_no_trade_reason={best.get('most_common_no_trade_reason')}",
            f"  force_flat_total_net_bps={best.get('force_flat_total_net_bps')}",
            f"  take_profit_exit_count={best.get('take_profit_exit_count')}",
            f"  stop_loss_exit_count={best.get('stop_loss_exit_count')}",
            f"  timeout_exit_count={best.get('timeout_exit_count')}",
            f"  final_liquidation_count={best.get('final_liquidation_count')}",
            f"  cooldown_block_count={best.get('cooldown_block_count')}",
            f"  trend_filter_block_count={best.get('trend_filter_block_count')}",
            f"  anchor_distance_block_count={best.get('anchor_distance_block_count')}",
            "",
            "Safety: paper only. No private API. No orders. No training. No promotion. No champion mutation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_text(path: Path, rows: list[dict[str, Any]], best: dict[str, Any]) -> None:
    lines = [
        "Path-Aware Ladder Baseline",
        "",
        f"Created at: {stamp()}",
        f"Source: {SOURCE_PATH}",
        f"Rows evaluated: {len(rows)} configs",
        f"Model gate mode: {MODEL_GATE_MODE}",
        f"Shadow dir: {SHADOW_DIR_ENV}",
        "",
        "Safety:",
        "  paper_only=true",
        "  private_api=false",
        "  orders=false",
        "  training=false",
        "  promotion=false",
        "  champion_mutation=false",
        "",
        "Top 20 ladder configs:",
    ]
    for row in rows[:20]:
        lines.append(
            "  "
            f"score={safe_float(row.get('rank_score')):.3f} force_flat={safe_float(row.get('force_flat_total_net_bps')):.3f} net={safe_float(row.get('total_net_bps')):.3f} "
            f"realized={safe_float(row.get('realized_net_bps')):.3f} unrealized={safe_float(row.get('unrealized_net_bps')):.3f} "
            f"dd={safe_float(row.get('max_drawdown_bps')):.3f} trades={row.get('number_of_trades')} "
            f"tp={row.get('take_profit_exit_count')} sl={row.get('stop_loss_exit_count')} timeout={row.get('timeout_exit_count')} final={row.get('final_liquidation_count')} "
            f"win={row.get('win_rate')} inv={row.get('max_inventory')} "
            f"avg_spacing={safe_float(row.get('avg_spacing_bps')):.3f} max_rung={row.get('max_rung_reached')} "
            f"vol_used={safe_float(row.get('volatility_used_fraction')):.3f} floor_used={safe_float(row.get('floor_used_fraction')):.3f} "
            f"rebalances={row.get('rebalance_count')} trades_per_rebalance={safe_float(row.get('trades_per_rebalance')):.3f} "
            f"gate_pass={safe_float(row.get('gate_pass_fraction')):.3f} reason={row.get('most_common_no_trade_reason')} "
            f"anchor={row.get('anchor_window')} vol={row.get('vol_window')} spacing_mult={row.get('spacing_vol_mult', row.get('vol_mult'))} "
            f"floor_mode={row.get('min_spacing_floor_mode')} rungs={row.get('rung_count')} "
            f"tp_spacing_mult={row.get('take_profit_spacing_mult', row.get('take_profit_mult'))} "
            f"stop_vol_mult={row.get('stop_loss_vol_mult')} stop_floor={row.get('min_stop_floor_bps', row.get('stop_loss_bps'))} hold={row.get('max_hold_buckets')}"
            f" rebalance={row.get('rebalance_mode')}@{row.get('rebalance_interval_buckets')}"
        )
    lines.extend(
        [
            "",
            "Best contract:",
            json.dumps(json_safe(best), indent=2, sort_keys=True),
            "",
            "Best config diagnostics:",
            f"  avg_spacing_bps={best.get('avg_spacing_bps')}",
            f"  min_spacing_bps_observed={best.get('min_spacing_bps_observed')}",
            f"  max_spacing_bps_observed={best.get('max_spacing_bps_observed')}",
            f"  spacing_clamp_min_fraction={best.get('spacing_clamp_min_fraction')}",
            f"  spacing_clamp_max_fraction={best.get('spacing_clamp_max_fraction')}",
            f"  volatility_used_fraction={best.get('volatility_used_fraction')}",
            f"  floor_used_fraction={best.get('floor_used_fraction')}",
            f"  avg_realized_vol_bps={best.get('avg_realized_vol_bps')}",
            f"  avg_spacing_to_vol_ratio={best.get('avg_spacing_to_vol_ratio')}",
            f"  avg_stop_to_vol_ratio={best.get('avg_stop_to_vol_ratio')}",
            f"  rebalance_count={best.get('rebalance_count')}",
            f"  avg_rebalance_interval_buckets={best.get('avg_rebalance_interval_buckets')}",
            f"  anchor_drift_rebalance_count={best.get('anchor_drift_rebalance_count')}",
            f"  volatility_rebalance_count={best.get('volatility_rebalance_count')}",
            f"  fixed_interval_rebalance_count={best.get('fixed_interval_rebalance_count')}",
            f"  trades_per_rebalance={best.get('trades_per_rebalance')}",
            f"  stale_ladder_entry_count={best.get('stale_ladder_entry_count')}",
            f"  rebalance_blocked_count={best.get('rebalance_blocked_count')}",
            f"  rung_trigger_count_by_rung={best.get('rung_trigger_count_by_rung')}",
            f"  buy_count_by_rung={best.get('buy_count_by_rung')}",
            f"  sell_count_by_rung={best.get('sell_count_by_rung')}",
            f"  most_common_no_trade_reason={best.get('most_common_no_trade_reason')}",
            f"  force_flat_total_net_bps={best.get('force_flat_total_net_bps')}",
            f"  take_profit_exit_count={best.get('take_profit_exit_count')}",
            f"  stop_loss_exit_count={best.get('stop_loss_exit_count')}",
            f"  timeout_exit_count={best.get('timeout_exit_count')}",
            f"  final_liquidation_count={best.get('final_liquidation_count')}",
            f"  cooldown_block_count={best.get('cooldown_block_count')}",
            f"  trend_filter_block_count={best.get('trend_filter_block_count')}",
            f"  anchor_distance_block_count={best.get('anchor_distance_block_count')}",
            "",
            "Warning: ladder baseline report only. No orders, promotion, training, or champion mutation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    frame = load_price_data(SOURCE_PATH)
    indicators = precompute_indicators(frame)
    gate = None
    if SHADOW_DIR_ENV and MODEL_GATE_MODE != "none":
        gate = build_shadow_gate(frame, Path(SHADOW_DIR_ENV))

    rows = []
    best_trades = pd.DataFrame()
    best_equity = pd.DataFrame()
    best_score = -math.inf
    best_contract: dict[str, Any] = {}
    contracts = [
        {
            "anchor_mode": ANCHOR_MODE,
            "anchor_window": anchor_window,
            "vol_window": vol_window,
            "spacing_mode": SPACING_MODE,
            "vol_mult": spacing_vol_mult,
            "spacing_vol_mult": spacing_vol_mult,
            "rung_count": rung_count,
            "take_profit_mult": take_profit_spacing_mult,
            "take_profit_spacing_mult": take_profit_spacing_mult,
            "min_spacing_bps": min_spacing_bps,
            "min_spacing_floor_mode": min_spacing_floor_mode,
            "cost_bps": DEFAULT_COST_BPS,
            "max_open_units": max_open_units,
            "stop_loss_bps": min_stop_floor_bps,
            "min_stop_floor_bps": min_stop_floor_bps,
            "stop_loss_vol_mult": stop_loss_vol_mult,
            "max_hold_buckets": max_hold_buckets,
            "cooldown_after_stop_buckets": COOLDOWN_AFTER_STOP_BUCKETS,
            "disable_buys_when_price_below_anchor_bps": DISABLE_BUYS_WHEN_PRICE_BELOW_ANCHOR_BPS,
            "disable_buys_when_ema_slope_below_bps": DISABLE_BUYS_WHEN_EMA_SLOPE_BELOW_BPS,
            "force_flat_at_end": FORCE_FLAT_AT_END,
            "range_break_buffer_bps": RANGE_BREAK_BUFFER_BPS,
            "rebalance_interval_buckets": REBALANCE_INTERVAL_BUCKETS,
            "rebalance_mode": REBALANCE_MODE,
            "rebalance_anchor_drift_bps": REBALANCE_ANCHOR_DRIFT_BPS,
            "rebalance_vol_change_fraction": REBALANCE_VOL_CHANGE_FRACTION,
        }
        for anchor_window in ANCHOR_WINDOWS
        for vol_window in VOL_WINDOWS
        for spacing_vol_mult in SPACING_VOL_MULTS
        for rung_count in RUNG_COUNTS
        for take_profit_spacing_mult in TAKE_PROFIT_SPACING_MULTS
        for min_spacing_bps in MIN_SPACING_GRID_BPS
        for min_spacing_floor_mode in MIN_SPACING_FLOOR_MODES
        for max_open_units in MAX_OPEN_UNITS_GRID
        for min_stop_floor_bps in STOP_LOSS_BPS_GRID
        for stop_loss_vol_mult in STOP_LOSS_VOL_MULTS
        for max_hold_buckets in MAX_HOLD_BUCKETS_GRID
    ]
    if MAX_CONFIGS_ENV:
        contracts = contracts[: int(float(MAX_CONFIGS_ENV))]
    for contract in contracts:
        result, _, _ = simulate_config(frame, indicators, gate, contract, save_paths=False)
        rows.append(result)
        score = safe_float(result.get("rank_score"), -math.inf)
        if score > best_score:
            best_score = score
            best_contract = contract

    best_result, best_trades, best_equity = simulate_config(frame, indicators, gate, best_contract, save_paths=True)
    rows = [best_result if row == next((r for r in rows if all(r.get(k) == best_contract.get(k) for k in best_contract)), None) else row for row in rows]
    rows = sorted(rows, key=lambda row: safe_float(row.get("rank_score"), -math.inf), reverse=True)

    out_dir = output_dir()
    results_path = out_dir / "ladder_grid_results.csv"
    text_path = out_dir / "ladder_grid_results.txt"
    diagnostics_path = out_dir / "ladder_diagnostics.txt"
    contract_path = out_dir / "best_ladder_contract.json"
    trades_path = out_dir / "trades.csv"
    equity_path = out_dir / "equity_curve.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    best_trades.to_csv(trades_path, index=False)
    best_equity.to_csv(equity_path, index=False)
    contract_payload = {**best_result, "source_path": str(SOURCE_PATH), "output_dir": str(out_dir)}
    contract_path.write_text(json.dumps(json_safe(contract_payload), indent=2, sort_keys=True), encoding="utf-8")
    write_text(text_path, rows, best_result)
    write_diagnostics(diagnostics_path, rows, best_result)

    print("Path-aware ladder baseline complete")
    print(f"Rows: {len(rows)}")
    print(f"Results CSV: {results_path}")
    print(f"Summary TXT: {text_path}")
    print(f"Diagnostics TXT: {diagnostics_path}")
    print(f"Best contract JSON: {contract_path}")
    print(f"Trades CSV: {trades_path}")
    print(f"Equity CSV: {equity_path}")
    print("Safety: paper only. No private API. No orders. No training. No promotion. No champion mutation.")
    print(pd.DataFrame(rows).head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
