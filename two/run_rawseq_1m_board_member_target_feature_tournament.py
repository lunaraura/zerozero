#!/usr/bin/env python3
"""Research-only target/feature tournament for new rawseq 1m board members."""

from __future__ import annotations

import math
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import (  # noqa: E402
    DEFAULT_SOURCE_PATH,
    SAFETY_FLAGS,
    build_features,
    env_path,
    load_candles,
    metric_row,
    now_stamp,
    parse_int_list,
    resolve_source_files,
    split_contract,
    stable_hash,
    write_csv,
    write_json,
)
from scripts.tiny.report_rawseq_1m_cross_asset_panel_scout import fit_hgb, fit_logistic, predict_model  # noqa: E402
from scripts.tiny.rawseq_1m_feature_evolution_runtime import (  # noqa: E402
    BoundedContentCache,
    MatrixTelemetry,
    extract_frame_matrix,
    extract_series_vector,
    take_rows,
)

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_board_member_target_feature_tournaments")
DEFAULT_SYMBOLS = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT"]
DEVELOPMENT_CUTOFF_MS = int(pd.Timestamp("2026-05-31T23:59:00Z").timestamp() * 1000)
QUOTE_COLUMNS = [
    "spread_percent",
    "best_bid",
    "best_ask",
    "mid_price",
    "bid_depth_10bps",
    "ask_depth_10bps",
    "bid_depth_25bps",
    "ask_depth_25bps",
    "order_book_imbalance_10bps",
    "order_book_imbalance_25bps",
]


@dataclass
class SymbolData:
    symbol: str
    candles: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
    folds: list[dict[str, Any]]
    available_families: set[str]


def env_float_list(name: str, default: list[float]) -> list[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def env_str_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def cap_tail(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows > 0 and len(frame) > max_rows:
        return frame.tail(max_rows).reset_index(drop=True)
    return frame.reset_index(drop=True)


def filter_development(frame: pd.DataFrame, cutoff_ms: int) -> pd.DataFrame:
    ts = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
    out = frame[ts <= cutoff_ms].copy()
    return out.sort_values("timestamp_ms").drop_duplicates("timestamp_ms", keep="last").reset_index(drop=True)


def future_extreme_return(close: pd.Series, price: pd.Series, horizon: int, kind: str) -> pd.Series:
    values = price.astype(float).reset_index(drop=True)
    close_values = close.astype(float).reset_index(drop=True)
    if kind == "high":
        future = values.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1].reset_index(drop=True)
        return 10000.0 * np.log(future / close_values)
    if kind == "low":
        future = values.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1].reset_index(drop=True)
        return 10000.0 * np.log(future / close_values)
    raise ValueError(kind)


def future_realized_vol(close: pd.Series, horizon: int) -> pd.Series:
    returns = 10000.0 * np.log(close.astype(float) / close.astype(float).shift(1))
    vals = []
    for idx in range(len(close)):
        chunk = returns.iloc[idx + 1 : idx + horizon + 1]
        vals.append(float(chunk.std(ddof=0)) if len(chunk) == horizon else math.nan)
    return pd.Series(vals, index=close.index)


def future_close_return_bps(close: pd.Series, horizon: int) -> pd.Series:
    close_values = close.astype(float).reset_index(drop=True)
    future = close_values.shift(-horizon)
    return 10000.0 * np.log(future / close_values)


def first_extreme_hit_step(close: pd.Series, price: pd.Series, horizon: int, threshold: pd.Series, side: str) -> pd.Series:
    close_values = close.astype(float).reset_index(drop=True)
    price_values = price.astype(float).reset_index(drop=True)
    threshold_values = pd.to_numeric(threshold, errors="coerce").reset_index(drop=True)
    hits: list[float] = []
    for idx in range(len(close_values)):
        threshold_bps = float(threshold_values.iloc[idx])
        if not math.isfinite(threshold_bps):
            hits.append(math.nan)
            continue
        hit_step = 0.0
        full_horizon = idx + horizon < len(close_values)
        if not full_horizon:
            hits.append(math.nan)
            continue
        for step in range(1, horizon + 1):
            j = idx + step
            if side == "up":
                move = 10000.0 * math.log(float(price_values.iloc[j]) / float(close_values.iloc[idx]))
            elif side == "down":
                move = max(0.0, -10000.0 * math.log(float(price_values.iloc[j]) / float(close_values.iloc[idx])))
            else:
                raise ValueError(side)
            if move > threshold_bps:
                hit_step = float(step)
                break
        hits.append(hit_step)
    return pd.Series(hits, index=close.index)


def future_range_bps(high: pd.Series, low: pd.Series, close: pd.Series, horizon: int) -> pd.Series:
    hi = high.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1].reset_index(drop=True)
    lo = low.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1].reset_index(drop=True)
    return 10000.0 * np.log(hi / lo)


def past_range_bps(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    hi = high.rolling(window, min_periods=window).max()
    lo = low.rolling(window, min_periods=window).min()
    return 10000.0 * np.log(hi / lo)


def build_target_lanes(frame: pd.DataFrame, horizons: list[int], vol_window: int, severity_levels: list[float]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    close = frame["close"].astype(float).reset_index(drop=True)
    high = frame["high"].astype(float).reset_index(drop=True)
    low = frame["low"].astype(float).reset_index(drop=True)
    returns = 10000.0 * np.log(close / close.shift(1))
    vol = returns.rolling(vol_window, min_periods=vol_window).std(ddof=0)
    out = pd.DataFrame({"timestamp_ms": frame["timestamp_ms"].reset_index(drop=True), f"trailing_volatility_bps_fw{vol_window}": vol})
    manifest: list[dict[str, Any]] = []
    for horizon in horizons:
        low_ret = future_extreme_return(close, low, horizon, "low")
        high_ret = future_extreme_return(close, high, horizon, "high")
        terminal_ret = future_close_return_bps(close, horizon)
        downside = np.maximum(0.0, -low_ret)
        upside = np.maximum(0.0, high_ret)
        down_col = f"downside_event_0p5vol_h{horizon}m_fw{vol_window}"
        up_col = f"upside_event_0p5vol_h{horizon}m_fw{vol_window}"
        directional_col = f"directional_return_positive_h{horizon}m"
        out[down_col] = (downside > 0.5 * vol).astype(float)
        out[up_col] = (upside > 0.5 * vol).astype(float)
        out[directional_col] = (terminal_ret > 0.0).astype(float)
        vol_values = pd.to_numeric(vol, errors="coerce").to_numpy(dtype=np.float64)
        downside_values = pd.to_numeric(downside, errors="coerce").to_numpy(dtype=np.float64)
        upside_values = pd.to_numeric(upside, errors="coerce").to_numpy(dtype=np.float64)
        terminal_values = pd.to_numeric(terminal_ret, errors="coerce").to_numpy(dtype=np.float64)
        out.loc[~np.isfinite(downside_values) | ~np.isfinite(vol_values), down_col] = np.nan
        out.loc[~np.isfinite(upside_values) | ~np.isfinite(vol_values), up_col] = np.nan
        out.loc[~np.isfinite(terminal_values), directional_col] = np.nan
        manifest.extend(
            [
                {"target_lane": "multi_horizon_downside", "target_name": down_col, "horizon_minutes": horizon, "threshold_vol_units": 0.5, "board_role": "downside_probability_member"},
                {"target_lane": "upside_excursion", "target_name": up_col, "horizon_minutes": horizon, "threshold_vol_units": 0.5, "board_role": "upside_probability_member"},
                {"target_lane": "directional_return", "target_name": directional_col, "horizon_minutes": horizon, "threshold_vol_units": math.nan, "board_role": "directional_probability_member"},
            ]
        )
        for level in severity_levels:
            sev_token = str(level).replace(".", "p")
            sev_col = f"downside_event_{sev_token}vol_h{horizon}m_fw{vol_window}"
            out[sev_col] = (downside > level * vol).astype(float)
            out.loc[~np.isfinite(downside_values) | ~np.isfinite(vol_values), sev_col] = np.nan
            manifest.append({"target_lane": "downside_severity", "target_name": sev_col, "horizon_minutes": horizon, "threshold_vol_units": level, "board_role": "downside_severity_member"})
    sorted_horizons = sorted(set(int(h) for h in horizons if int(h) > 0))
    for level in severity_levels:
        token = str(level).replace(".", "p")
        max_horizon = max(sorted_horizons) if sorted_horizons else 0
        if max_horizon <= 0:
            continue
        threshold = level * vol
        down_hit = first_extreme_hit_step(close, low, max_horizon, threshold, "down")
        up_hit = first_extreme_hit_step(close, high, max_horizon, threshold, "up")
        previous = 0
        for horizon in sorted_horizons:
            for side, hit_steps, role in [
                ("downside", down_hit, "downside_hazard_member"),
                ("upside", up_hit, "upside_hazard_member"),
            ]:
                col = f"{side}_interval_hazard_{token}vol_m{previous + 1}_to_{horizon}m_fw{vol_window}"
                hit_values = pd.to_numeric(hit_steps, errors="coerce").to_numpy(dtype=np.float64)
                out[col] = ((hit_steps > previous) & (hit_steps <= horizon)).astype(float)
                out.loc[~np.isfinite(hit_values), col] = np.nan
                manifest.append(
                    {
                        "target_lane": f"{side}_interval_hazard",
                        "target_name": col,
                        "horizon_minutes": horizon,
                        "interval_start_minutes": previous + 1,
                        "interval_end_minutes": horizon,
                        "threshold_vol_units": level,
                        "board_role": role,
                    }
                )
            previous = horizon
    for horizon in [5, 15]:
        current_vol = vol
        fut_vol = future_realized_vol(close, horizon)
        col = f"volatility_expansion_future_vol_gt_current_h{horizon}m_fw{vol_window}"
        out[col] = (fut_vol > current_vol).astype(float)
        fut_vol_values = pd.to_numeric(fut_vol, errors="coerce").to_numpy(dtype=np.float64)
        current_vol_values = pd.to_numeric(current_vol, errors="coerce").to_numpy(dtype=np.float64)
        out.loc[~np.isfinite(fut_vol_values) | ~np.isfinite(current_vol_values), col] = np.nan
        manifest.append({"target_lane": "volatility_expansion", "target_name": col, "horizon_minutes": horizon, "threshold_vol_units": math.nan, "board_role": "volatility_context_member"})
    range_h = 5
    future_range = future_range_bps(high, low, close, range_h)
    past_range = past_range_bps(high, low, range_h)
    recent_range = past_range.shift(1).rolling(vol_window, min_periods=vol_window).median()
    range_col = f"range_expansion_future_range_gt_recent_median_h{range_h}m_fw{vol_window}"
    out[range_col] = (future_range > recent_range).astype(float)
    future_range_values = pd.to_numeric(future_range, errors="coerce").to_numpy(dtype=np.float64)
    recent_range_values = pd.to_numeric(recent_range, errors="coerce").to_numpy(dtype=np.float64)
    out.loc[~np.isfinite(future_range_values) | ~np.isfinite(recent_range_values), range_col] = np.nan
    manifest.append({"target_lane": "volatility_expansion", "target_name": range_col, "horizon_minutes": range_h, "threshold_vol_units": math.nan, "board_role": "volatility_context_member"})
    barrier_h = max(horizons)
    for level in [0.5, 1.0]:
        token = str(level).replace(".", "p")
        col = f"barrier_first_up_{token}vol_before_down_{token}vol_h{barrier_h}m_fw{vol_window}"
        vals = []
        ambiguous_flags = []
        unresolved_flags = []
        for idx in range(len(frame)):
            threshold = vol.iloc[idx] * level
            if not math.isfinite(float(threshold)):
                vals.append(math.nan)
                ambiguous_flags.append(False)
                unresolved_flags.append(True)
                continue
            up_idx = down_idx = None
            ambiguous_same_minute = False
            for step in range(1, barrier_h + 1):
                j = idx + step
                if j >= len(frame):
                    break
                up_hit = 10000.0 * math.log(float(high.iloc[j]) / float(close.iloc[idx])) > threshold
                down_hit = max(0.0, -10000.0 * math.log(float(low.iloc[j]) / float(close.iloc[idx]))) > threshold
                if up_hit and up_idx is None:
                    up_idx = step
                if down_hit and down_idx is None:
                    down_idx = step
                if up_hit and down_hit and up_idx == down_idx:
                    up_idx = down_idx = -1
                    ambiguous_same_minute = True
                    break
                if up_idx is not None or down_idx is not None:
                    break
            if up_idx == -1 or (up_idx is None and down_idx is None):
                vals.append(math.nan)
                ambiguous_flags.append(ambiguous_same_minute)
                unresolved_flags.append(True)
            else:
                vals.append(float(up_idx is not None and (down_idx is None or up_idx < down_idx)))
                ambiguous_flags.append(False)
                unresolved_flags.append(False)
        out[col] = vals
        out[f"{col}_ambiguous_same_minute"] = np.asarray(ambiguous_flags, dtype=float)
        out[f"{col}_unresolved"] = np.asarray(unresolved_flags, dtype=float)
        valid_denominator = max(1, len(vals))
        manifest.append(
            {
                "target_lane": "barrier_first",
                "target_name": col,
                "horizon_minutes": barrier_h,
                "threshold_vol_units": level,
                "board_role": "directional_context_member",
                "barrier_ordering_source": "candle_only",
                "ambiguous_same_minute_fraction": float(np.mean(ambiguous_flags)) if ambiguous_flags else math.nan,
                "unresolved_fraction": float(np.mean(unresolved_flags)) if unresolved_flags else math.nan,
                "ambiguous_same_minute_rows": int(sum(ambiguous_flags)),
                "unresolved_rows": int(sum(unresolved_flags)),
                "supervised_ambiguous_rows_excluded": True,
                "ordered_trade_resolution_available": False,
            }
        )
    return out, manifest


def resolve_barrier_order_from_ordered_trades(*_args: Any, **_kwargs: Any) -> pd.Series:
    """Future interface for trade-sequenced barrier ordering.

    This phase intentionally does not reconstruct intraminute order. Callers
    must provide a tested ordered-trade resolver before using trade data to
    replace candle-only ambiguous labels.
    """
    raise NotImplementedError("ordered trade barrier resolution is not implemented in this phase")


def add_quote_spread_features(frame: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if not any(col in frame.columns for col in QUOTE_COLUMNS):
        return features, False
    out = features.copy()
    for col in QUOTE_COLUMNS:
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce")
    if "spread_percent" in frame.columns:
        spread = pd.to_numeric(frame["spread_percent"], errors="coerce")
        out["spread_bps"] = 10000.0 * spread
        out["spread_change_bps"] = out["spread_bps"].diff()
    if {"best_bid", "best_ask"}.issubset(frame.columns):
        bid = pd.to_numeric(frame["best_bid"], errors="coerce")
        ask = pd.to_numeric(frame["best_ask"], errors="coerce")
        mid = (bid + ask) / 2.0
        out["midprice_return_bps"] = 10000.0 * np.log(mid / mid.shift(1))
        out["bid_movement_bps"] = 10000.0 * np.log(bid / bid.shift(1))
        out["ask_movement_bps"] = 10000.0 * np.log(ask / ask.shift(1))
    return out.replace([np.inf, -np.inf], np.nan), True


def add_short_path_features(frame: pd.DataFrame, features: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = features.copy()
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    open_ = frame["open"].astype(float)
    vol = frame["volume"].astype(float)
    ret = 10000.0 * np.log(close / close.shift(1))
    out["return_acceleration_bps"] = ret.diff()
    out["body_to_range_ratio"] = (close - open_).abs() / (high - low).replace(0, np.nan)
    out["wick_asymmetry_bps"] = out["upper_wick_bps"] - out["lower_wick_bps"] if {"upper_wick_bps", "lower_wick_bps"}.issubset(out.columns) else np.nan
    signed = np.sign(ret.fillna(0.0).to_numpy())
    run = np.zeros(len(signed), dtype=float)
    for idx in range(1, len(signed)):
        run[idx] = run[idx - 1] + signed[idx] if signed[idx] and signed[idx] == signed[idx - 1] else signed[idx]
    out["consecutive_direction_minutes"] = run
    typical = (high + low + close) / 3.0
    for window in windows:
        rolling_vwap = (typical * vol).rolling(window, min_periods=window).sum() / vol.rolling(window, min_periods=window).sum()
        out[f"distance_to_local_vwap_bps_fw{window}"] = 10000.0 * np.log(close / rolling_vwap)
        rv = ret.rolling(window, min_periods=window).std(ddof=0)
        out[f"volatility_acceleration_bps_fw{window}"] = rv.diff()
        out[f"range_compression_ratio_fw{window}"] = out[f"candle_range_bps"] / out[f"rolling_range_bps_fw{window}"]
    return out.replace([np.inf, -np.inf], np.nan)


def add_regime_features(frame: pd.DataFrame, features: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = features.copy()
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    out["time_of_day_sin"] = np.sin(2.0 * np.pi * (ts.dt.hour * 60 + ts.dt.minute) / 1440.0)
    out["time_of_day_cos"] = np.cos(2.0 * np.pi * (ts.dt.hour * 60 + ts.dt.minute) / 1440.0)
    out["weekend_indicator"] = (ts.dt.dayofweek >= 5).astype(float)
    for window in windows:
        vol = out.get(f"rolling_volatility_bps_fw{window}", pd.Series(np.nan, index=out.index))
        rng = out.get(f"rolling_range_bps_fw{window}", pd.Series(np.nan, index=out.index))
        slope = out.get(f"ema_slope_bps_fw{window}", pd.Series(np.nan, index=out.index))
        out[f"volatility_regime_rank_fw{window}"] = vol.rolling(window * 4, min_periods=window).rank(pct=True)
        out[f"range_regime_rank_fw{window}"] = rng.rolling(window * 4, min_periods=window).rank(pct=True)
        out[f"trend_vs_range_state_fw{window}"] = slope.abs() / rng.replace(0, np.nan)
    if "spread_bps" in out.columns:
        out["spread_regime_rank_fw240"] = out["spread_bps"].rolling(240, min_periods=60).rank(pct=True)
    return out.replace([np.inf, -np.inf], np.nan)


def add_cross_asset_features(features_by_symbol: dict[str, pd.DataFrame]) -> None:
    close = {}
    for symbol, frame in features_by_symbol.items():
        close[symbol] = frame.set_index("timestamp_ms")["signed_bucket_return_bps"]
    panel = pd.DataFrame(close).sort_index()
    market_median = panel.median(axis=1)
    market_vol = panel.std(axis=1, ddof=0)
    dispersion = panel.max(axis=1) - panel.min(axis=1)
    same_direction = (panel > 0).sum(axis=1)
    for symbol, frame in features_by_symbol.items():
        aligned = frame["timestamp_ms"].map(market_median)
        frame["market_median_return_1m_bps"] = aligned
        frame["market_wide_volatility_1m_bps"] = frame["timestamp_ms"].map(market_vol)
        frame["cross_asset_dispersion_1m_bps"] = frame["timestamp_ms"].map(dispersion)
        frame["symbols_positive_return_count"] = frame["timestamp_ms"].map(same_direction)
        if "BTCUSDT" in panel.columns:
            frame["btc_return_1m_bps"] = frame["timestamp_ms"].map(panel["BTCUSDT"])
            frame["symbol_relative_to_btc_1m_bps"] = frame["signed_bucket_return_bps"] - frame["btc_return_1m_bps"]
        if "ETHUSDT" in panel.columns:
            frame["eth_return_1m_bps"] = frame["timestamp_ms"].map(panel["ETHUSDT"])
        for horizon in [5, 15]:
            if "BTCUSDT" in panel.columns:
                frame[f"btc_return_{horizon}m_bps"] = frame["timestamp_ms"].map(panel["BTCUSDT"].rolling(horizon, min_periods=horizon).sum())
            if "ETHUSDT" in panel.columns:
                frame[f"eth_return_{horizon}m_bps"] = frame["timestamp_ms"].map(panel["ETHUSDT"].rolling(horizon, min_periods=horizon).sum())


def read_symbol(symbol: str, source_root: Path, max_rows: int, cutoff_ms: int, feature_windows: list[int], horizons: list[int], vol_window: int, severity_levels: list[float]) -> tuple[SymbolData, list[dict[str, Any]], list[dict[str, Any]]]:
    candles = load_candles(resolve_source_files(source_root, symbol), max_rows=0)
    candles = filter_development(candles, cutoff_ms)
    candles = cap_tail(candles, max_rows)
    base_features, feature_audit, leakage = build_features(candles, feature_windows)
    quote_features, quote_available = add_quote_spread_features(candles, base_features)
    features = add_regime_features(candles, add_short_path_features(candles, quote_features, feature_windows), feature_windows)
    targets, target_manifest = build_target_lanes(candles, horizons, vol_window, severity_levels)
    split, folds, _ = split_contract(candles, feature_lookback=max(feature_windows + [vol_window]), max_horizon=max(horizons + [15]), fold_count=4)
    available = {"existing", "short_path", "regime"}
    if quote_available:
        available.add("quote_spread")
    if not candles.empty:
        available.add("cross_asset_pending")
    source_rows = [{"symbol": symbol, "rows": len(candles), "first_timestamp": str(candles["timestamp"].min()), "last_timestamp": str(candles["timestamp"].max()), "post_cutoff_rows_used": int((pd.to_numeric(candles["timestamp_ms"], errors="coerce") > cutoff_ms).sum()), **leakage}]
    return SymbolData(symbol, candles, features, targets, folds, available), feature_audit, [{**row, "symbol": symbol} for row in target_manifest] + source_rows


def feature_groups(all_columns: list[str], available: set[str]) -> dict[str, list[str]]:
    existing = [c for c in all_columns if any(tok in c for tok in ["signed_bucket", "candle_", "wick", "volume", "rolling_", "distance_to_recent", "close_to_ema", "ema_slope", "trailing_volatility"])]
    quote = [c for c in all_columns if any(tok in c for tok in ["spread", "bid", "ask", "depth", "imbalance", "midprice", "wall"])]
    short = [c for c in all_columns if any(tok in c for tok in ["acceleration", "compression", "consecutive", "vwap", "body_to_range", "wick_asymmetry"])]
    cross = [c for c in all_columns if any(tok in c for tok in ["btc_", "eth_", "market_", "cross_asset", "symbols_positive", "relative_to_btc"])]
    regime = [c for c in all_columns if any(tok in c for tok in ["regime", "time_of_day", "weekend", "trend_vs_range"])]
    groups = {
        "existing": existing,
        "existing_plus_quote_spread": existing + quote if "quote_spread" in available else [],
        "existing_plus_short_path": existing + short,
        "existing_plus_cross_asset": existing + cross if "cross_asset" in available else [],
        "existing_plus_regime": existing + regime,
        "all_challenger_features": sorted(set(existing + quote + short + cross + regime)) if ("quote_spread" in available or "cross_asset" in available) else sorted(set(existing + short + regime)),
    }
    for family, cols in {"minus_quote_spread": quote, "minus_short_path": short, "minus_cross_asset": cross, "minus_regime": regime}.items():
        groups[f"all_{family}"] = [c for c in groups["all_challenger_features"] if c not in set(cols)]
    return {k: [c for c in v if c in all_columns] for k, v in groups.items()}


def regime_mask(data: SymbolData, indices: np.ndarray, regime_name: str) -> tuple[np.ndarray, str]:
    if regime_name in {"", "all", "unrestricted"}:
        return np.ones(len(indices), dtype=bool), ""
    features = data.features
    if regime_name == "high_volatility":
        cols = [c for c in features.columns if c.startswith("volatility_regime_rank_fw")]
        if not cols:
            return np.zeros(len(indices), dtype=bool), "missing_volatility_regime_rank"
        vals = pd.to_numeric(features.iloc[indices][cols[0]], errors="coerce").to_numpy(dtype=np.float64)
        return vals >= 0.67, ""
    if regime_name == "low_volatility":
        cols = [c for c in features.columns if c.startswith("volatility_regime_rank_fw")]
        if not cols:
            return np.zeros(len(indices), dtype=bool), "missing_volatility_regime_rank"
        vals = pd.to_numeric(features.iloc[indices][cols[0]], errors="coerce").to_numpy(dtype=np.float64)
        return vals <= 0.33, ""
    if regime_name == "trend":
        cols = [c for c in features.columns if c.startswith("trend_vs_range_state_fw")]
        if not cols:
            return np.zeros(len(indices), dtype=bool), "missing_trend_vs_range_state"
        vals = pd.to_numeric(features.iloc[indices][cols[0]], errors="coerce").to_numpy(dtype=np.float64)
        threshold = np.nanmedian(vals)
        return vals >= threshold, ""
    if regime_name == "range":
        cols = [c for c in features.columns if c.startswith("trend_vs_range_state_fw")]
        if not cols:
            return np.zeros(len(indices), dtype=bool), "missing_trend_vs_range_state"
        vals = pd.to_numeric(features.iloc[indices][cols[0]], errors="coerce").to_numpy(dtype=np.float64)
        threshold = np.nanmedian(vals)
        return vals < threshold, ""
    if regime_name == "tight_spread":
        if "spread_regime_rank_fw240" not in features.columns:
            return np.zeros(len(indices), dtype=bool), "missing_spread_regime_rank"
        vals = pd.to_numeric(features.iloc[indices]["spread_regime_rank_fw240"], errors="coerce").to_numpy(dtype=np.float64)
        return vals <= 0.33, ""
    if regime_name == "wide_spread":
        if "spread_regime_rank_fw240" not in features.columns:
            return np.zeros(len(indices), dtype=bool), "missing_spread_regime_rank"
        vals = pd.to_numeric(features.iloc[indices]["spread_regime_rank_fw240"], errors="coerce").to_numpy(dtype=np.float64)
        return vals >= 0.67, ""
    return np.zeros(len(indices), dtype=bool), f"unknown_regime:{regime_name}"


def finite_xy(data: SymbolData, indices: np.ndarray, feature_cols: list[str], target_col: str, regime_name: str = "all") -> tuple[np.ndarray, np.ndarray, int, str]:
    x, y, regime_rows, regime_reason, _ = finite_xy_with_coverage(data, indices, feature_cols, target_col, regime_name)
    return x, y, regime_rows, regime_reason


def finite_xy_with_coverage(
    data: SymbolData,
    indices: np.ndarray,
    feature_cols: list[str],
    target_col: str,
    regime_name: str = "all",
    matrix_cache: BoundedContentCache | None = None,
    matrix_telemetry: MatrixTelemetry | None = None,
    matrix_dtype: np.dtype | type = np.float64,
    semantic_contract: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, str, dict[str, Any]]:
    context = {"symbol": data.symbol, "target_col": target_col, "feature_count": len(feature_cols), "regime_name": regime_name}
    full_x = extract_frame_matrix(data.features, feature_cols, matrix_dtype, matrix_cache, matrix_telemetry, context, semantic_contract)
    full_y = extract_series_vector(data.targets[target_col], matrix_dtype, matrix_cache, matrix_telemetry, context, semantic_contract)
    x = take_rows(full_x, indices, matrix_telemetry, {**context, "matrix_role": "candidate_feature_rows"})
    y = take_rows(full_y, indices, matrix_telemetry, {**context, "matrix_role": "candidate_target_rows"})
    rmask, regime_reason = regime_mask(data, indices, regime_name)
    finite_feature_mask = np.isfinite(x).all(axis=1)
    finite_target_mask = np.isfinite(y)
    finite_mask = finite_target_mask & finite_feature_mask
    mask = finite_target_mask & finite_feature_mask & rmask
    ts_values = data.features.iloc[indices]["timestamp_ms"].to_numpy(dtype=np.int64) if "timestamp_ms" in data.features.columns else indices
    matched_indices = indices[mask]
    matched_timestamps = ts_values[mask]
    coverage = {
        "symbol": data.symbol,
        "source_rows": int(len(indices)),
        "feature_finite_rows": int(np.sum(finite_feature_mask)),
        "target_labeled_rows": int(np.sum(finite_target_mask)),
        "regime_candidate_rows": int(np.sum(rmask)),
        "matched_rows": int(np.sum(mask)),
        "rows_removed_by_feature_filter": int(np.sum(~finite_feature_mask)),
        "rows_removed_by_target_filter": int(np.sum(~finite_target_mask)),
        "rows_removed_by_regime_filter": int(np.sum(~rmask)),
        "rows_removed_by_finite_filtering": int(np.sum(~finite_mask)),
        "rows_removed_by_purge": 0,
        "rows_removed_by_embargo": 0,
        "rows_removed_by_symbol_timestamp_mismatch": 0,
        "positive_rows": int(np.sum(y[mask] > 0.5)) if np.any(mask) else 0,
        "negative_rows": int(np.sum(y[mask] <= 0.5)) if np.any(mask) else 0,
        "source_index_sha256": stable_hash(indices.tolist()),
        "matched_index_sha256": stable_hash(matched_indices.tolist()),
        "source_timestamp_sha256": stable_hash(ts_values.tolist()),
        "matched_timestamp_sha256": stable_hash(matched_timestamps.tolist()),
        "regime_filter_reason": regime_reason,
    }
    return x[mask], y[mask], int(np.sum(rmask)), regime_reason, coverage


def annotate_matched_row_comparability(coverage_rows: list[dict[str, Any]], min_rows: int = 0) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], set[str]] = {}
    for row in coverage_rows:
        key = (
            row.get("target_lane"),
            row.get("target_name"),
            row.get("regime_name", "all"),
            row.get("model"),
            row.get("model_seed"),
            row.get("fold_id"),
            row.get("split"),
            row.get("symbol"),
        )
        groups.setdefault(key, set()).add(str(row.get("matched_timestamp_sha256", "")))
    out: list[dict[str, Any]] = []
    for row in coverage_rows:
        key = (
            row.get("target_lane"),
            row.get("target_name"),
            row.get("regime_name", "all"),
            row.get("model"),
            row.get("model_seed"),
            row.get("fold_id"),
            row.get("split"),
            row.get("symbol"),
        )
        if int(row.get("source_rows", 0) or 0) <= 0:
            status = "alignment_failure"
        elif int(row.get("matched_rows", 0) or 0) < int(min_rows):
            status = "insufficient_rows"
        elif len(groups.get(key, set())) > 1:
            status = "different_rows_disclosed"
        else:
            status = "identical_rows"
        row = dict(row)
        row["matched_row_contract_status"] = status
        row["feature_group_row_comparability_group_key"] = stable_hash([str(x) for x in key])
        out.append(row)
    return out


def stack_xy(
    by_symbol: dict[str, SymbolData],
    symbols: list[str],
    start: int,
    end: int,
    feature_cols: list[str],
    target_col: str,
    max_rows_per_symbol: int,
    regime_name: str = "all",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    x, y, s, regime_rows, regime_reason, _ = stack_xy_with_coverage(
        by_symbol,
        symbols,
        start,
        end,
        feature_cols,
        target_col,
        max_rows_per_symbol,
        regime_name,
    )
    return x, y, s, regime_rows, regime_reason


def stack_xy_with_coverage(
    by_symbol: dict[str, SymbolData],
    symbols: list[str],
    start: int,
    end: int,
    feature_cols: list[str],
    target_col: str,
    max_rows_per_symbol: int,
    regime_name: str = "all",
    matrix_cache: BoundedContentCache | None = None,
    matrix_telemetry: MatrixTelemetry | None = None,
    matrix_dtype: np.dtype | type = np.float64,
    semantic_contract: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str, list[dict[str, Any]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    regime_candidate_rows = 0
    regime_reasons: list[str] = []
    coverage_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        data = by_symbol[symbol]
        stop = min(end, len(data.features) - 1)
        idx = np.arange(start, stop + 1, dtype=np.int64)
        if max_rows_per_symbol > 0 and len(idx) > max_rows_per_symbol:
            idx = idx[-max_rows_per_symbol:]
        x, y, regime_rows, regime_reason, coverage = finite_xy_with_coverage(
            data,
            idx,
            feature_cols,
            target_col,
            regime_name,
            matrix_cache,
            matrix_telemetry,
            matrix_dtype,
            semantic_contract,
        )
        coverage_rows.append(coverage)
        regime_candidate_rows += regime_rows
        if regime_reason:
            regime_reasons.append(f"{symbol}:{regime_reason}")
        if len(y):
            xs.append(x)
            ys.append(y)
            ss.append(np.asarray([symbol] * len(y), dtype=object))
    if not ys:
        return np.empty((0, len(feature_cols))), np.empty(0), np.empty(0, dtype=object), regime_candidate_rows, ";".join(regime_reasons), coverage_rows
    x_out = np.vstack(xs)
    y_out = np.concatenate(ys)
    s_out = np.concatenate(ss)
    if matrix_telemetry:
        matrix_telemetry.record("vstack_features", x_out, {"symbols": ",".join(symbols), "target_col": target_col, "feature_count": len(feature_cols)})
        matrix_telemetry.record("concatenate_targets", y_out, {"symbols": ",".join(symbols), "target_col": target_col})
    return x_out, y_out, s_out, regime_candidate_rows, ";".join(regime_reasons), coverage_rows


def fit_logistic_with_diagnostics(
    train_x: np.ndarray,
    train_y: np.ndarray,
    max_iter: int = 300,
) -> tuple[Any, dict[str, Any]]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    diagnostics: dict[str, Any] = {
        "logistic_solver": "lbfgs",
        "logistic_regularization": "l2",
        "logistic_max_iter": int(max_iter),
        "logistic_iteration_count": 0,
        "logistic_convergence_status": "not_run",
        "logistic_warning_category": "",
        "logistic_warning_message": "",
        "logistic_train_rows": int(len(train_y)),
        "logistic_feature_count": int(train_x.shape[1]) if train_x.ndim == 2 else 0,
        "model_eligible_for_advancement": True,
    }
    if len(np.unique(train_y)) < 2:
        diagnostics["logistic_convergence_status"] = "constant_probability_single_class"
        diagnostics["model_eligible_for_advancement"] = False
        return {"constant_probability": float(np.mean(train_y)) if len(train_y) else 0.5}, diagnostics

    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=int(max_iter), solver="lbfgs"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_x, train_y.astype(int))
    warning = next((w for w in caught if issubclass(w.category, ConvergenceWarning)), None)
    lr = model.named_steps.get("logisticregression")
    n_iter = getattr(lr, "n_iter_", [0])
    diagnostics["logistic_iteration_count"] = int(np.max(n_iter)) if len(n_iter) else 0
    if warning is not None:
        diagnostics["logistic_convergence_status"] = "non_converged"
        diagnostics["logistic_warning_category"] = warning.category.__name__
        diagnostics["logistic_warning_message"] = str(warning.message)
        diagnostics["model_eligible_for_advancement"] = False
    else:
        diagnostics["logistic_convergence_status"] = "converged"
    return model, diagnostics


def fit_predict(model_name: str, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, model_seed: int = 1337) -> tuple[np.ndarray, bool, float]:
    pred, parity, diff, _ = fit_predict_with_diagnostics(model_name, train_x, train_y, val_x, model_seed)
    return pred, parity, diff


def fit_predict_with_diagnostics(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    model_seed: int = 1337,
) -> tuple[np.ndarray, bool, float, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "model_eligible_for_advancement": True,
        "logistic_convergence_status": "not_applicable",
        "logistic_iteration_count": math.nan,
        "logistic_max_iter": math.nan,
        "logistic_warning_category": "",
        "logistic_warning_message": "",
        "logistic_solver": "",
        "logistic_regularization": "",
        "logistic_feature_count": int(train_x.shape[1]) if train_x.ndim == 2 else 0,
        "logistic_train_rows": int(len(train_y)),
    }
    if model_name == "constant_prevalence":
        return np.full(len(val_x), np.clip(float(np.mean(train_y)), 1e-6, 1 - 1e-6)), True, 0.0, diagnostics
    if model_name == "regularized_logistic":
        model, diagnostics = fit_logistic_with_diagnostics(train_x, train_y)
    elif model_name == "shallow_hgb":
        contract = {"model_pipeline": [{"step": "HistGradientBoostingClassifier", "max_iter": 40, "max_leaf_nodes": 15, "learning_rate": 0.05, "l2_regularization": 0.01, "random_state": int(model_seed)}]}
        model = fit_hgb(contract, train_x, train_y)
    else:
        raise ValueError(model_name)
    pred = predict_model(model, val_x)
    loaded = pickle.loads(pickle.dumps(model))
    pred2 = predict_model(loaded, val_x)
    diff = float(np.nanmax(np.abs(pred - pred2))) if len(pred) else 0.0
    return pred, diff <= 1e-12, diff, diagnostics


def confidence_coverage_metrics(y: np.ndarray, p: np.ndarray, baseline_rate: float, coverage_levels: tuple[float, ...] = (0.10, 0.20, 0.40)) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    base = np.full(len(y), np.clip(float(baseline_rate), 1e-6, 1 - 1e-6))
    confidence = np.abs(p - float(baseline_rate))
    out: dict[str, Any] = {
        "coverage_grid": ",".join(str(level) for level in coverage_levels),
        "abstention_supported": True,
        "confidence_reference_probability": float(baseline_rate),
    }
    if not len(y):
        return out
    order = np.argsort(-confidence)
    for level in coverage_levels:
        rows = max(1, int(math.ceil(len(y) * level)))
        idx = order[:rows]
        metrics = metric_row(y[idx], p[idx], base[idx])
        token = str(level).replace(".", "p")
        out[f"coverage_{token}_rows"] = int(rows)
        out[f"coverage_{token}_event_prevalence"] = metrics["event_prevalence"]
        out[f"coverage_{token}_brier_skill"] = metrics["brier_skill_vs_prevalence"]
        out[f"coverage_{token}_pr_auc_lift"] = metrics["pr_auc_lift_over_event_prevalence"]
    return out


def evaluate_tournament(
    by_symbol: dict[str, SymbolData],
    target_rows: list[dict[str, Any]],
    max_rows_per_symbol: int,
    min_rows: int,
    min_prevalence: float,
    allowed_lanes: set[str],
    allowed_feature_groups: set[str],
    allowed_models: list[str],
    max_folds: int,
    regime_names: list[str] | None = None,
    model_seeds: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    symbols = sorted(by_symbol)
    regime_names = regime_names or ["all"]
    model_seeds = model_seeds or [1337]
    all_cols = sorted(set().union(*(set(d.features.columns) for d in by_symbol.values())) - {"timestamp", "timestamp_ms"})
    cross_available = all(c in all_cols for c in ["btc_return_1m_bps", "market_median_return_1m_bps"])
    quote_available = any(c in all_cols for c in ["spread_bps", "best_bid", "bid_depth_10bps"])
    available: set[str] = set()
    if cross_available:
        available.add("cross_asset")
    if quote_available:
        available.add("quote_spread")
    groups = feature_groups(all_cols, available)
    deduped_targets: dict[str, dict[str, Any]] = {}
    for row in target_rows:
        if "target_name" in row and str(row.get("target_lane")) in allowed_lanes:
            deduped_targets.setdefault(f"{row.get('target_lane')}::{row['target_name']}", row)
    target_meta = list(deduped_targets.values())
    rows: list[dict[str, Any]] = []
    for target in target_meta:
        target_col = str(target["target_name"])
        for group_name, feature_cols in groups.items():
            if group_name not in allowed_feature_groups:
                continue
            if not feature_cols:
                rows.append({**target, "feature_group": group_name, "model": "all", "status": "FEATURE_GROUP_UNAVAILABLE", "failure_reason": "required source columns unavailable", **SAFETY_FLAGS})
                continue
            for regime_name in regime_names:
                for fold_id in range(max_folds):
                    fold_symbols = [d for d in by_symbol.values() if len(d.folds) > fold_id]
                    if not fold_symbols:
                        for model_name in allowed_models:
                            seeds_for_model = model_seeds if model_name != "constant_prevalence" else [model_seeds[0]]
                            for model_seed in seeds_for_model:
                                rows.append(
                                    {
                                        **target,
                                        "feature_group": group_name,
                                        "regime_name": regime_name,
                                        "fold_id": fold_id,
                                        "feature_count": len(feature_cols),
                                        "train_rows": 0,
                                        "validation_rows": 0,
                                        "holdout_used_for_selection": False,
                                        "development_cutoff": "2026-05-31T23:59:00Z",
                                        "model": model_name,
                                        "model_seed": int(model_seed),
                                        "status": "DATA_FAILED",
                                        "failure_reason": "no development folds available after cutoff",
                                        **SAFETY_FLAGS,
                                    }
                                )
                        continue
                    train_end = min(d.folds[fold_id]["train_end_index"] for d in fold_symbols)
                    val_start = max(d.folds[fold_id]["validation_start_index"] for d in fold_symbols)
                    val_end = min(d.folds[fold_id]["validation_end_index"] for d in fold_symbols)
                    train_x, train_y, _, train_regime_rows, train_regime_reason = stack_xy(by_symbol, symbols, 0, train_end, feature_cols, target_col, max_rows_per_symbol, regime_name)
                    val_x, val_y, _, val_regime_rows, val_regime_reason = stack_xy(by_symbol, symbols, val_start, val_end, feature_cols, target_col, max_rows_per_symbol, regime_name)
                    base = {
                        **target,
                        "feature_group": group_name,
                        "regime_name": regime_name,
                        "fold_id": fold_id,
                        "feature_count": len(feature_cols),
                        "train_rows": int(len(train_y)),
                        "validation_rows": int(len(val_y)),
                        "train_regime_candidate_rows": train_regime_rows,
                        "validation_regime_candidate_rows": val_regime_rows,
                        "train_regime_coverage_fraction": float(len(train_y) / train_regime_rows) if train_regime_rows else math.nan,
                        "validation_regime_coverage_fraction": float(len(val_y) / val_regime_rows) if val_regime_rows else math.nan,
                        "regime_filter_reason": ";".join(x for x in [train_regime_reason, val_regime_reason] if x),
                        "holdout_used_for_selection": False,
                        "development_cutoff": "2026-05-31T23:59:00Z",
                        "validation_scope": "pooled_chronological_discovery_only",
                        "private_api": False,
                        "orders": False,
                        "promotion": False,
                        "champion_mutation": False,
                    }
                    if len(train_y) < min_rows or len(val_y) < min_rows or len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
                        rows.append({**base, "model": "all", "status": "DATA_FAILED", "failure_reason": "insufficient rows or class diversity", **SAFETY_FLAGS})
                        continue
                    baseline = np.full(len(val_y), np.clip(float(np.mean(train_y)), 1e-6, 1 - 1e-6))
                    for model_name in allowed_models:
                        seeds_for_model = model_seeds if model_name != "constant_prevalence" else [model_seeds[0]]
                        for model_seed in seeds_for_model:
                            try:
                                pred, parity, diff, diagnostics = fit_predict_with_diagnostics(model_name, train_x, train_y, val_x, int(model_seed))
                                if not diagnostics.get("model_eligible_for_advancement", True):
                                    row = {
                                        **base,
                                        "model": model_name,
                                        "model_seed": int(model_seed),
                                        "status": "TRAIN_FAILED",
                                        "failure_reason": "model_ineligible_for_advancement",
                                        **diagnostics,
                                        **SAFETY_FLAGS,
                                    }
                                else:
                                    row = {
                                        **base,
                                        "model": model_name,
                                        "model_seed": int(model_seed),
                                        "status": "OK",
                                        "failure_reason": "",
                                        "save_reload_parity": parity,
                                        "save_reload_max_abs_diff": diff,
                                        **diagnostics,
                                        **metric_row(val_y, pred, baseline),
                                        **confidence_coverage_metrics(val_y, pred, float(np.mean(train_y))),
                                    }
                            except Exception as exc:
                                row = {**base, "model": model_name, "model_seed": int(model_seed), "status": "TRAIN_FAILED", "failure_reason": repr(exc), **SAFETY_FLAGS}
                            rows.append(row)
    survival = survival_rows(rows, min_prevalence)
    return rows, survival


def evaluate_candidate_records(
    by_symbol: dict[str, SymbolData],
    candidates: list[dict[str, Any]],
    max_rows_per_symbol: int,
    min_rows: int,
    min_prevalence: float,
    max_folds: int,
    matrix_cache: BoundedContentCache | None = None,
    matrix_telemetry: MatrixTelemetry | None = None,
    matrix_dtype: np.dtype | type = np.float64,
    semantic_contract: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate explicit candidate identities without expanding siblings."""
    symbols = sorted(by_symbol)
    all_cols = sorted(set().union(*(set(d.features.columns) for d in by_symbol.values())) - {"timestamp", "timestamp_ms"})
    cross_available = all(c in all_cols for c in ["btc_return_1m_bps", "market_median_return_1m_bps"])
    quote_available = any(c in all_cols for c in ["spread_bps", "best_bid", "bid_depth_10bps"])
    available: set[str] = set()
    if cross_available:
        available.add("cross_asset")
    if quote_available:
        available.add("quote_spread")
    groups = feature_groups(all_cols, available)
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for candidate in sorted(candidates, key=lambda r: str(r.get("candidate_key", ""))):
        target_col = str(candidate.get("target_name"))
        group_name = str(candidate.get("feature_group"))
        feature_cols = groups.get(group_name, [])
        expected_folds = int(max_folds)
        if not feature_cols:
            rows.append(
                {
                    **candidate,
                    "expected_folds": expected_folds,
                    "feature_count": 0,
                    "model": candidate.get("model"),
                    "status": "FEATURE_GROUP_UNAVAILABLE",
                    "failure_reason": "required source columns unavailable",
                    "holdout_used_for_selection": False,
                    **SAFETY_FLAGS,
                }
            )
            continue
        regime_name = str(candidate.get("regime_name", "all") or "all")
        for fold_id in range(max_folds):
            fold_symbols = [d for d in by_symbol.values() if len(d.folds) > fold_id]
            if not fold_symbols:
                rows.append(
                    {
                        **candidate,
                        "expected_folds": expected_folds,
                        "fold_id": fold_id,
                        "feature_count": len(feature_cols),
                        "train_rows": 0,
                        "validation_rows": 0,
                        "status": "DATA_FAILED",
                        "failure_reason": "no development folds available after cutoff",
                        "holdout_used_for_selection": False,
                        **SAFETY_FLAGS,
                    }
                )
                continue
            train_end = min(d.folds[fold_id]["train_end_index"] for d in fold_symbols)
            val_start = max(d.folds[fold_id]["validation_start_index"] for d in fold_symbols)
            val_end = min(d.folds[fold_id]["validation_end_index"] for d in fold_symbols)
            train_x, train_y, _, train_regime_rows, train_regime_reason, train_coverage = stack_xy_with_coverage(
                by_symbol,
                symbols,
                0,
                train_end,
                feature_cols,
                target_col,
                max_rows_per_symbol,
                regime_name,
                matrix_cache,
                matrix_telemetry,
                matrix_dtype,
                semantic_contract,
            )
            val_x, val_y, _, val_regime_rows, val_regime_reason, val_coverage = stack_xy_with_coverage(
                by_symbol,
                symbols,
                val_start,
                val_end,
                feature_cols,
                target_col,
                max_rows_per_symbol,
                regime_name,
                matrix_cache,
                matrix_telemetry,
                matrix_dtype,
                semantic_contract,
            )
            for split_name, split_rows in [("train", train_coverage), ("validation", val_coverage)]:
                for cov in split_rows:
                    coverage_rows.append(
                        {
                            **candidate,
                            "fold_id": fold_id,
                            "split": split_name,
                            "feature_count": len(feature_cols),
                            **cov,
                            "train_rows": int(len(train_y)) if split_name == "train" else math.nan,
                            "calibration_rows": 0,
                            "eval_rows": int(len(val_y)) if split_name == "validation" else math.nan,
                            "matched_row_contract_status": "pending_comparability_annotation",
                            **SAFETY_FLAGS,
                        }
                    )
            base = {
                **candidate,
                "fold_id": fold_id,
                "expected_folds": expected_folds,
                "feature_count": len(feature_cols),
                "train_rows": int(len(train_y)),
                "validation_rows": int(len(val_y)),
                "rows": int(len(val_y)),
                "train_regime_candidate_rows": train_regime_rows,
                "validation_regime_candidate_rows": val_regime_rows,
                "train_regime_coverage_fraction": float(len(train_y) / train_regime_rows) if train_regime_rows else math.nan,
                "validation_regime_coverage_fraction": float(len(val_y) / val_regime_rows) if val_regime_rows else math.nan,
                "regime_filter_reason": ";".join(x for x in [train_regime_reason, val_regime_reason] if x),
                "holdout_used_for_selection": False,
                "development_cutoff": "2026-05-31T23:59:00Z",
                "validation_scope": "pooled_chronological_discovery_only",
                **SAFETY_FLAGS,
            }
            if len(train_y) < min_rows or len(val_y) < min_rows:
                rows.append({**base, "status": "DATA_FAILED", "failure_reason": "insufficient rows"})
                continue
            if len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
                rows.append({**base, "status": "DATA_FAILED", "failure_reason": "insufficient class diversity"})
                continue
            prevalence = float(np.mean(train_y))
            if not (min_prevalence <= prevalence <= 1.0 - min_prevalence):
                rows.append({**base, "status": "DATA_FAILED", "failure_reason": "train prevalence outside minimum gate", "event_prevalence": prevalence})
                continue
            baseline = np.full(len(val_y), np.clip(prevalence, 1e-6, 1 - 1e-6))
            model_name = str(candidate.get("model"))
            try:
                pred, parity, diff, diagnostics = fit_predict_with_diagnostics(model_name, train_x, train_y, val_x, int(candidate.get("model_seed", 1337)))
                if not diagnostics.get("model_eligible_for_advancement", True):
                    rows.append({**base, "status": "TRAIN_FAILED", "failure_reason": "model_ineligible_for_advancement", **diagnostics})
                    continue
                rows.append(
                    {
                        **base,
                        "status": "OK",
                        "failure_reason": "",
                        "train_event_prevalence": prevalence,
                        "save_reload_parity": parity,
                        "save_reload_max_abs_diff": diff,
                        **diagnostics,
                        **metric_row(val_y, pred, baseline),
                        **confidence_coverage_metrics(val_y, pred, prevalence),
                    }
                )
            except Exception as exc:
                rows.append({**base, "status": "TRAIN_FAILED", "failure_reason": repr(exc)})
    return rows, annotate_matched_row_comparability(coverage_rows, min_rows=min_rows)


def survival_rows(rows: list[dict[str, Any]], min_prevalence: float) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    ok = df[(df["status"] == "OK") & (df["model"] != "constant_prevalence")].copy()
    if "regime_name" not in ok.columns:
        ok["regime_name"] = "all"
    if "model_seed" not in ok.columns:
        ok["model_seed"] = 1337
    for (target_lane, target_name, feature_group, regime_name, model, model_seed), group in ok.groupby(["target_lane", "target_name", "feature_group", "regime_name", "model", "model_seed"], dropna=False):
        brier = pd.to_numeric(group["brier_skill_vs_prevalence"], errors="coerce")
        pr = pd.to_numeric(group["pr_auc_lift_over_event_prevalence"], errors="coerce")
        prev = pd.to_numeric(group["event_prevalence"], errors="coerce")
        required_positive_folds = min(3, len(group))
        gates = {
            "positive_median_brier_skill": bool(brier.median() > 0),
            "positive_worst_fold_brier_skill": bool(brier.min() > 0),
            "positive_median_pr_auc_lift": bool(pr.median() > 0),
            "sufficient_event_prevalence": bool(min_prevalence <= prev.median() <= 1.0 - min_prevalence),
            "save_reload_parity_all": bool(group["save_reload_parity"].astype(str).str.lower().isin(["true", "1"]).all()),
            "validation_fold_stability": bool((brier > 0).sum() >= required_positive_folds),
        }
        out.append(
            {
                "target_lane": target_lane,
                "target_name": target_name,
                "feature_group": feature_group,
                "regime_name": regime_name,
                "model": model,
                "model_seed": int(model_seed),
                "folds": int(len(group)),
                "rows": int(pd.to_numeric(group["rows"], errors="coerce").sum()),
                "median_event_prevalence": float(prev.median()),
                "median_brier_skill": float(brier.median()),
                "worst_fold_brier_skill": float(brier.min()),
                "median_log_loss_improvement": float(pd.to_numeric(group["log_loss_improvement_vs_prevalence"], errors="coerce").median()),
                "median_pr_auc_lift": float(pr.median()),
                **gates,
                "survives_validation_gate": all(gates.values()),
                "rejection_reasons": ";".join(key for key, val in gates.items() if not val),
                **SAFETY_FLAGS,
            }
        )
    out.sort(key=lambda r: (r["survives_validation_gate"], r["median_brier_skill"], r["worst_fold_brier_skill"], r["median_pr_auc_lift"]), reverse=True)
    return out


def text_summary(path: Path, survival: list[dict[str, Any]], rows: list[dict[str, Any]], contract: dict[str, Any]) -> None:
    passed = [r for r in survival if r.get("survives_validation_gate")]
    lines = [
        "RAWSEQ 1M BOARD-MEMBER TARGET/FEATURE TOURNAMENT",
        f"created_at={contract['created_at']}",
        "scope=research_only_cpu_baseline_tournament",
        "selection_data_end=2026-05-31T23:59:00Z",
        f"evaluated_rows={len(rows)}",
        f"surviving_rows={len(passed)}",
        "",
        "Top validation candidates:",
    ]
    for row in survival[:20]:
        lines.append(
            f"- status={'PASS' if row['survives_validation_gate'] else 'FAIL'} lane={row['target_lane']} target={row['target_name']} "
            f"features={row['feature_group']} model={row['model']} median_brier_skill={row['median_brier_skill']:.6f} "
            f"worst_fold={row['worst_fold_brier_skill']:.6f} reasons={row['rejection_reasons']}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- Frozen downside/dashboard models were not changed.",
            "- Feature groups marked FEATURE_GROUP_UNAVAILABLE lacked required development-source columns.",
            "- Passing rows are validation research candidates only; holdout/forward survival is required before freezing.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    source_root = env_path("RAWSEQ_BOARD_SOURCE_PATH", DEFAULT_SOURCE_PATH)
    output_root = env_path("RAWSEQ_BOARD_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    symbols = [s.strip().upper() for s in os.getenv("RAWSEQ_BOARD_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",") if s.strip()]
    feature_windows = parse_int_list(os.getenv("RAWSEQ_BOARD_FEATURE_WINDOWS", "60,240"), [60, 240])
    horizons = parse_int_list(os.getenv("RAWSEQ_BOARD_HORIZONS", "1,2,4,8"), [1, 2, 4, 8])
    severity_levels = env_float_list("RAWSEQ_BOARD_SEVERITY_LEVELS", [0.5, 1.0, 1.5, 2.0])
    vol_window = int(os.getenv("RAWSEQ_BOARD_VOL_WINDOW", "240"))
    max_rows = int(os.getenv("RAWSEQ_BOARD_MAX_ROWS_PER_SYMBOL", "50000"))
    eval_cap = int(os.getenv("RAWSEQ_BOARD_MAX_EVAL_ROWS_PER_SYMBOL", "15000"))
    min_rows = int(os.getenv("RAWSEQ_BOARD_MIN_ROWS", "500"))
    min_prevalence = float(os.getenv("RAWSEQ_BOARD_MIN_PREVALENCE", "0.01"))
    allowed_lanes = set(
        env_str_list(
            "RAWSEQ_BOARD_TARGET_LANES",
            ["multi_horizon_downside", "downside_severity", "upside_excursion", "volatility_expansion", "barrier_first"],
        )
    )
    allowed_feature_groups = set(
        env_str_list(
            "RAWSEQ_BOARD_FEATURE_GROUPS",
            [
                "existing",
                "existing_plus_quote_spread",
                "existing_plus_short_path",
                "existing_plus_cross_asset",
                "existing_plus_regime",
                "all_challenger_features",
                "all_minus_quote_spread",
                "all_minus_short_path",
                "all_minus_cross_asset",
                "all_minus_regime",
            ],
        )
    )
    allowed_models = env_str_list("RAWSEQ_BOARD_MODELS", ["constant_prevalence", "regularized_logistic", "shallow_hgb"])
    max_folds = int(os.getenv("RAWSEQ_BOARD_MAX_FOLDS", "4"))
    out_dir = output_root / f"rawseq_1m_board_member_target_feature_tournament_{now_stamp()}"
    by_symbol: dict[str, SymbolData] = {}
    feature_audit_rows: list[dict[str, Any]] = []
    target_manifest_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        data, audit, target_rows = read_symbol(symbol, source_root, max_rows, DEVELOPMENT_CUTOFF_MS, feature_windows, horizons, vol_window, severity_levels)
        by_symbol[symbol] = data
        feature_audit_rows.extend({"symbol": symbol, **row} for row in audit)
        target_manifest_rows.extend(target_rows)
    add_cross_asset_features({symbol: data.features for symbol, data in by_symbol.items()})
    rows, survival = evaluate_tournament(
        by_symbol,
        target_manifest_rows,
        eval_cap,
        min_rows,
        min_prevalence,
        allowed_lanes,
        allowed_feature_groups,
        allowed_models,
        max_folds,
    )
    contract = {
        "created_at": now_stamp(),
        "source_root": str(source_root),
        "symbols": symbols,
        "feature_windows": feature_windows,
        "horizons": horizons,
        "severity_levels": severity_levels,
        "volatility_window": vol_window,
        "development_cutoff_iso": "2026-05-31T23:59:00Z",
        "max_rows_per_symbol": max_rows,
        "max_eval_rows_per_symbol": eval_cap,
        "models": ["constant_prevalence", "regularized_logistic", "shallow_hgb"],
        "allowed_target_lanes": sorted(allowed_lanes),
        "allowed_feature_groups": sorted(allowed_feature_groups),
        "allowed_models": allowed_models,
        "max_folds": max_folds,
        "selection_rule": "validation_folds_only_positive_median_and_worst_brier_skill_plus_pr_lift",
        "validation_scope": "pooled_chronological_discovery_only",
        "scenario_validation_performed": False,
        "frozen_models_mutated": False,
        "dashboard_mutated": False,
        **SAFETY_FLAGS,
    }
    contract["contract_hash"] = stable_hash(contract)
    write_csv(out_dir / "board_member_target_feature_metrics.csv", rows)
    write_csv(out_dir / "board_member_target_feature_survivors.csv", survival)
    write_csv(out_dir / "board_member_feature_audit.csv", feature_audit_rows)
    write_csv(out_dir / "board_member_target_manifest.csv", target_manifest_rows)
    write_json(out_dir / "board_member_target_feature_tournament_contract.json", contract)
    text_summary(out_dir / "board_member_target_feature_tournament_summary.txt", survival, rows, contract)
    print(f"output_dir={out_dir}")
    print(f"metric_rows={len(rows)}")
    print(f"survivor_rows={len(survival)}")
    print(f"surviving_validation_candidates={sum(1 for row in survival if row.get('survives_validation_gate'))}")
    print(f"runtime_seconds={time.perf_counter() - start:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
