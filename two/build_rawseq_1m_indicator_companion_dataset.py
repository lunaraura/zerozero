#!/usr/bin/env python3
"""Build the rawseq 1m dual-timescale RSI/MA companion dataset.

This is a research-only dataset builder for a companion indicator model. It
does not modify or retrain the frozen pooled downside-risk candidate and it
filters development data at 2026-05-31T23:59:00Z so June/July remain untouched
for model development.
"""

from __future__ import annotations

import json
import math
import os
import sys
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
    SourceFile,
    build_features,
    env_path,
    file_sha256,
    load_candles,
    now_stamp,
    resolve_source_files,
    stable_hash,
    write_csv,
    write_json,
)

SYMBOLS = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT"]
DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_indicator_companion_scout")
DEFAULT_FROZEN_CONTRACT = Path(
    r"F:\rsio\rawseq_1m_cross_asset_scout\rawseq_1m_fixed_transfer_contract_20260712T131949Z\fixed_transfer_contract.json"
)
FROZEN_CANDIDATE_HASH = "8109480bd6b0c2575337cada9be991746199200d42feaeea1fd7b9dd0b3eb6bf"
DEVELOPMENT_CUTOFF = "2026-05-31T23:59:00Z"
SHORT_LEN = 60
LONG_LEN = 60
LONG_STRIDE = 4
MAX_HORIZON = 8
PURGE_ROWS = LONG_LEN * LONG_STRIDE
EMBARGO_ROWS = MAX_HORIZON
TEMPORAL_CHANNELS = [
    "signed_close_return_bps",
    "candle_range_bps",
    "candle_body_bps",
    "upper_wick_bps",
    "lower_wick_bps",
    "rolling_volatility_bps",
    "rolling_range_bps",
    "distance_to_recent_high_bps",
    "distance_to_recent_low_bps",
    "close_to_ema20_bps",
    "close_to_ema60_bps",
    "rsi14_normalized",
]
MA_CHANNELS = [
    "close_to_ema20_bps",
    "close_to_ema60_bps",
    "ema20_minus_ema60_bps",
    "ema20_slope_bps_per_minute",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI using ewm alpha=1/period after causal one-step deltas."""
    close = pd.to_numeric(close, errors="coerce").astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    return rsi


def filter_development_files(source_files: list[SourceFile], cutoff_ym: str = "2026-05") -> list[SourceFile]:
    return [item for item in source_files if item.year_month and item.year_month <= cutoff_ym]


def companion_indicator_frame(candles: pd.DataFrame) -> pd.DataFrame:
    close = candles["close"].astype(float).reset_index(drop=True)
    high = candles["high"].astype(float).reset_index(drop=True)
    low = candles["low"].astype(float).reset_index(drop=True)
    open_ = candles["open"].astype(float).reset_index(drop=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame(
            {
                "timestamp_ms": candles["timestamp_ms"].reset_index(drop=True),
                "timestamp": candles["timestamp"].reset_index(drop=True),
                "close": close,
                "signed_close_return_bps": 10000.0 * np.log(close / close.shift(1)),
                "candle_range_bps": 10000.0 * np.log(high / low),
                "candle_body_bps": 10000.0 * np.log(close / open_),
                "upper_wick_bps": 10000.0 * np.log(high / np.maximum(open_, close)),
                "lower_wick_bps": 10000.0 * np.log(np.minimum(open_, close) / low),
            }
        )
        returns = out["signed_close_return_bps"]
        rolling_high = high.rolling(60, min_periods=60).max()
        rolling_low = low.rolling(60, min_periods=60).min()
        ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
        ema60 = close.ewm(span=60, adjust=False, min_periods=60).mean()
        out["rolling_volatility_bps"] = returns.rolling(60, min_periods=60).std(ddof=0)
        out["rolling_range_bps"] = 10000.0 * np.log(rolling_high / rolling_low)
        out["distance_to_recent_high_bps"] = 10000.0 * np.log(close / rolling_high)
        out["distance_to_recent_low_bps"] = 10000.0 * np.log(close / rolling_low)
        out["ema20"] = ema20
        out["ema60"] = ema60
        out["close_to_ema20_bps"] = 10000.0 * np.log(close / ema20)
        out["close_to_ema60_bps"] = 10000.0 * np.log(close / ema60)
        out["ema20_minus_ema60_bps"] = 10000.0 * np.log(ema20 / ema60)
        out["ema20_slope_bps_per_minute"] = 10000.0 * np.log(ema20 / ema20.shift(1))
    out["rsi14"] = wilder_rsi(close)
    out["rsi14_normalized"] = (out["rsi14"] - 50.0) / 50.0
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def constant_price_ma_roll_forward(close: float, ema20: float, ema60: float, horizons: int = MAX_HORIZON) -> np.ndarray:
    alpha20 = 2.0 / (20.0 + 1.0)
    alpha60 = 2.0 / (60.0 + 1.0)
    current_ema20 = float(ema20)
    current_ema60 = float(ema60)
    rows = []
    for _ in range(horizons):
        prev20 = current_ema20
        current_ema20 = alpha20 * close + (1.0 - alpha20) * current_ema20
        current_ema60 = alpha60 * close + (1.0 - alpha60) * current_ema60
        rows.append(
            [
                10000.0 * math.log(close / current_ema20),
                10000.0 * math.log(close / current_ema60),
                10000.0 * math.log(current_ema20 / current_ema60),
                10000.0 * math.log(current_ema20 / prev20),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def split_for_symbol(n: int) -> np.ndarray:
    split = np.full(n, "train", dtype=object)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    split[train_end:validation_end] = "validation"
    split[validation_end:] = "test"
    for boundary in [train_end, validation_end]:
        lo = max(0, boundary - EMBARGO_ROWS)
        hi = min(n, boundary + PURGE_ROWS)
        split[lo:hi] = "purged"
    return split


def materialize_symbol(
    symbol: str,
    source_path: Path,
    static_feature_order: list[str],
    feature_windows: list[int],
    max_symbol_rows: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    files = filter_development_files(resolve_source_files(source_path, symbol))
    if not files:
        raise RuntimeError(f"No pre-June source files for {symbol}")
    candles = load_candles(files, max_rows=max_symbol_rows)
    cutoff = pd.Timestamp(DEVELOPMENT_CUTOFF)
    candles = candles[candles["timestamp"] <= cutoff].reset_index(drop=True)
    static_features, feature_audit_rows, _ = build_features(candles, feature_windows)
    targets = companion_indicator_frame(candles)
    static_features["trailing_volatility_bps_fw240"] = targets["signed_close_return_bps"].rolling(240, min_periods=240).std(ddof=0)
    static = static_features[static_feature_order].to_numpy(dtype=np.float32)
    static_values: list[np.ndarray] = []
    short_values: list[np.ndarray] = []
    long_values: list[np.ndarray] = []
    y_rsi_values: list[np.ndarray] = []
    y_ma_values: list[np.ndarray] = []
    ma_persist_values: list[np.ndarray] = []
    ma_constant_values: list[np.ndarray] = []
    rsi_slope_values: list[np.ndarray] = []
    timestamp_values: list[float] = []
    close_values: list[float] = []
    current_rsi_values: list[float] = []
    current_ema20_values: list[float] = []
    current_ema60_values: list[float] = []
    indices: list[int] = []
    symbols: list[str] = []
    temporal = targets[TEMPORAL_CHANNELS].to_numpy(dtype=np.float32)
    current_ma = targets[MA_CHANNELS].to_numpy(dtype=np.float32)
    rsi = targets["rsi14"].to_numpy(dtype=np.float32)
    for idx in range(PURGE_ROWS - 1, len(targets) - MAX_HORIZON):
        short_idx = np.arange(idx - SHORT_LEN + 1, idx + 1)
        long_idx = idx - np.arange(LONG_LEN - 1, -1, -1) * LONG_STRIDE
        future_idx = idx + np.arange(1, MAX_HORIZON + 1)
        if short_idx[0] < 0 or long_idx[0] < 0:
            continue
        y_rsi = rsi[future_idx] - rsi[idx]
        future_ma = current_ma[future_idx]
        values = [
            static[idx],
            temporal[short_idx],
            temporal[long_idx],
            y_rsi.astype(np.float32),
            future_ma.astype(np.float32),
            np.tile(current_ma[idx], (MAX_HORIZON, 1)).astype(np.float32),
        ]
        if any(not np.isfinite(value).all() for value in values):
            continue
        close = float(targets["close"].iloc[idx])
        ema20 = float(targets["ema20"].iloc[idx])
        ema60 = float(targets["ema60"].iloc[idx])
        if not all(math.isfinite(x) and x > 0 for x in [close, ema20, ema60]):
            continue
        static_values.append(static[idx])
        short_values.append(temporal[short_idx])
        long_values.append(temporal[long_idx])
        y_rsi_values.append(y_rsi.astype(np.float32))
        y_ma_values.append(future_ma.astype(np.float32))
        ma_persist_values.append(np.tile(current_ma[idx], (MAX_HORIZON, 1)).astype(np.float32))
        ma_constant_values.append(constant_price_ma_roll_forward(close, ema20, ema60))
        recent_slope = rsi[idx] - rsi[idx - 1] if idx > 0 and np.isfinite(rsi[idx - 1]) else 0.0
        rsi_slope_values.append((np.arange(1, MAX_HORIZON + 1, dtype=np.float32) * recent_slope).astype(np.float32))
        timestamp_values.append(float(targets["timestamp_ms"].iloc[idx]))
        close_values.append(close)
        current_rsi_values.append(float(rsi[idx]))
        current_ema20_values.append(ema20)
        current_ema60_values.append(ema60)
        indices.append(idx)
        symbols.append(symbol)
    n = len(static_values)
    if n == 0:
        raise RuntimeError(f"No finite companion rows for {symbol}")
    split = split_for_symbol(n)
    payload = {
        "x_static": np.asarray(static_values, dtype=np.float32),
        "x_short": np.asarray(short_values, dtype=np.float32),
        "x_long": np.asarray(long_values, dtype=np.float32),
        "y_rsi_delta": np.asarray(y_rsi_values, dtype=np.float32),
        "y_ma_state": np.asarray(y_ma_values, dtype=np.float32),
        "baseline_rsi_persistence": np.zeros((n, MAX_HORIZON), dtype=np.float32),
        "baseline_rsi_slope": np.asarray(rsi_slope_values, dtype=np.float32),
        "baseline_ma_persistence": np.asarray(ma_persist_values, dtype=np.float32),
        "baseline_ma_constant_price": np.asarray(ma_constant_values, dtype=np.float32),
        "timestamp_ms": np.asarray(timestamp_values, dtype=np.float64),
        "source_row_index": np.asarray(indices, dtype=np.int64),
        "symbol": np.asarray(symbols, dtype=object),
        "split": split,
        "current_close": np.asarray(close_values, dtype=np.float32),
        "current_rsi14": np.asarray(current_rsi_values, dtype=np.float32),
        "current_ema20": np.asarray(current_ema20_values, dtype=np.float32),
        "current_ema60": np.asarray(current_ema60_values, dtype=np.float32),
    }
    target_rows = target_audit(symbol, payload, targets)
    source_meta = {
        "symbol": symbol,
        "source_files": [str(item.path) for item in files],
        "source_file_count": len(files),
        "source_sha256": stable_hash([{"path": str(item.path), "sha256": file_sha256(item.path)} for item in files]),
        "raw_rows": int(len(candles)),
        "materialized_rows": n,
        "first_timestamp": pd.to_datetime(payload["timestamp_ms"].min(), unit="ms", utc=True).isoformat(),
        "last_timestamp": pd.to_datetime(payload["timestamp_ms"].max(), unit="ms", utc=True).isoformat(),
    }
    return payload, feature_audit_rows, target_rows, source_meta


def target_audit(symbol: str, payload: dict[str, np.ndarray], targets: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_rsi = payload["y_rsi_delta"]
    y_ma = payload["y_ma_state"]
    ts = pd.to_datetime(payload["timestamp_ms"], unit="ms", utc=True)
    base = {
        "symbol": symbol,
        "rows": int(len(y_rsi)),
        "date_range_start": ts.min().isoformat(),
        "date_range_end": ts.max().isoformat(),
        "rsi_minimum": safe_float(targets["rsi14"].min()),
        "rsi_maximum": safe_float(targets["rsi14"].max()),
        "nonfinite_count": int((~np.isfinite(y_rsi)).sum() + (~np.isfinite(y_ma)).sum()),
    }
    years = pd.Series(ts).dt.year.to_numpy()
    for horizon in range(MAX_HORIZON):
        vals = y_rsi[:, horizon]
        rows.append({**base, "target_group": "rsi_delta", "horizon": horizon + 1, "channel": "rsi_delta", **distribution(vals)})
    for channel_idx, channel in enumerate(MA_CHANNELS):
        for horizon in range(MAX_HORIZON):
            vals = y_ma[:, horizon, channel_idx]
            rows.append({**base, "target_group": "ma_state", "horizon": horizon + 1, "channel": channel, **distribution(vals)})
    for year in sorted(set(int(x) for x in years)):
        mask = years == year
        rows.append({**base, "target_group": "target_drift_by_year", "horizon": "all", "channel": "all", "year": year, **distribution(y_ma[mask].reshape(-1))})
    return rows


def distribution(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"mean": math.nan, "std": math.nan, "p05": math.nan, "p50": math.nan, "p95": math.nan}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def concat_payloads(payloads: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = list(payloads[0].keys())
    out: dict[str, np.ndarray] = {}
    for key in keys:
        out[key] = np.concatenate([payload[key] for payload in payloads], axis=0)
    return out


def fit_target_scaler(y_rsi: np.ndarray, y_ma: np.ndarray, split: np.ndarray) -> dict[str, Any]:
    mask = split == "train"
    rsi = y_rsi[mask]
    ma = y_ma[mask].reshape(int(mask.sum()), -1)
    return {
        "policy": "train_split_only",
        "rsi_delta_mean": np.mean(rsi, axis=0).astype(float).tolist(),
        "rsi_delta_std": np.maximum(np.std(rsi, axis=0, ddof=0), 1e-6).astype(float).tolist(),
        "ma_state_mean": np.mean(ma, axis=0).astype(float).tolist(),
        "ma_state_std": np.maximum(np.std(ma, axis=0, ddof=0), 1e-6).astype(float).tolist(),
    }


def main() -> int:
    source_path = env_path("RAWSEQ_INDICATOR_SOURCE_PATH", DEFAULT_SOURCE_PATH)
    output_root = env_path("RAWSEQ_INDICATOR_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    frozen_contract_path = env_path("RAWSEQ_INDICATOR_FROZEN_FEATURE_CONTRACT", DEFAULT_FROZEN_CONTRACT)
    symbols = [x.strip().upper() for x in os.getenv("RAWSEQ_INDICATOR_SYMBOLS", ",".join(SYMBOLS)).split(",") if x.strip()]
    max_symbol_rows = int(os.getenv("RAWSEQ_INDICATOR_MAX_SOURCE_ROWS_PER_SYMBOL", "0") or "0")
    run_dir = output_root / f"dual_timescale_indicator_companion_dataset_{now_stamp()}"
    frozen_contract = read_json(frozen_contract_path)
    static_feature_order = list(frozen_contract["model_feature_names_and_order"])
    if len(static_feature_order) != 31:
        raise RuntimeError(f"Expected frozen static feature contract length 31, got {len(static_feature_order)}")
    feature_windows = [int(x) for x in frozen_contract["feature_windows_minutes"]]
    payloads = []
    feature_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        payload, f_rows, t_rows, source_meta = materialize_symbol(symbol, source_path, static_feature_order, feature_windows, max_symbol_rows)
        payloads.append(payload)
        feature_rows.extend({**row, "symbol": symbol} for row in f_rows)
        target_rows.extend(t_rows)
        source_rows.append(source_meta)
    dataset = concat_payloads(payloads)
    order = np.argsort(dataset["timestamp_ms"], kind="mergesort")
    for key, value in dataset.items():
        dataset[key] = value[order]
    target_scaler = fit_target_scaler(dataset["y_rsi_delta"], dataset["y_ma_state"], dataset["split"])
    dataset_path = run_dir / "indicator_companion_dataset.npz"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **dataset)
    input_contract = {
        "model_name": "dual_timescale_indicator_companion_v1",
        "static_feature_contract_source": str(frozen_contract_path),
        "frozen_candidate_hash": FROZEN_CANDIDATE_HASH,
        "x_static_shape": [31],
        "x_short_shape": [SHORT_LEN, len(TEMPORAL_CHANNELS)],
        "x_long_shape": [LONG_LEN, len(TEMPORAL_CHANNELS)],
        "short_cadence_minutes": 1,
        "long_stride_minutes": LONG_STRIDE,
        "long_elapsed_coverage_minutes": LONG_LEN * LONG_STRIDE,
        "static_feature_order": static_feature_order,
        "temporal_channel_order": TEMPORAL_CHANNELS,
        "causality": "current_and_historical_inputs_only",
        "development_cutoff": DEVELOPMENT_CUTOFF,
        "june_holdout_used_for_development": False,
        "july_data_accessed": False,
        **SAFETY_FLAGS,
    }
    target_contract = {
        "rsi_formula": "Wilder RSI14 via ewm(alpha=1/14, adjust=False, min_periods=14) on one-minute closes",
        "output_heads": {"rsi_delta": [MAX_HORIZON], "ma_state": [MAX_HORIZON, len(MA_CHANNELS)]},
        "flattened_output_values": 40,
        "ma_channel_order": MA_CHANNELS,
        "horizons_minutes": list(range(1, MAX_HORIZON + 1)),
        "labels_from_actual_future_closes": True,
        "recursive_prediction_labels": False,
        "target_scaler": target_scaler,
        "target_scaling_policy": "train-only target mean/std recorded; models may normalize per fold",
        "max_target_horizon_minutes": MAX_HORIZON,
        **SAFETY_FLAGS,
    }
    purge_rows = []
    for symbol in symbols:
        for boundary in ["train_validation", "validation_test"]:
            purge_rows.append(
                {
                    "symbol": symbol,
                    "boundary": boundary,
                    "purge_rows": PURGE_ROWS,
                    "embargo_rows": EMBARGO_ROWS,
                    "input_long_lookback_rows": PURGE_ROWS,
                    "target_horizon_rows": MAX_HORIZON,
                    "purge_embargo_status": "PASS",
                }
            )
    manifest = {
        "created_at": now_stamp(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "symbols": symbols,
        "rows": int(len(dataset["split"])),
        "train_rows": int(np.sum(dataset["split"] == "train")),
        "validation_rows": int(np.sum(dataset["split"] == "validation")),
        "test_rows": int(np.sum(dataset["split"] == "test")),
        "purged_rows": int(np.sum(dataset["split"] == "purged")),
        "x_static_shape": list(dataset["x_static"].shape),
        "x_short_shape": list(dataset["x_short"].shape),
        "x_long_shape": list(dataset["x_long"].shape),
        "y_rsi_delta_shape": list(dataset["y_rsi_delta"].shape),
        "y_ma_state_shape": list(dataset["y_ma_state"].shape),
        "development_cutoff": DEVELOPMENT_CUTOFF,
        "june_files_opened": False,
        "july_files_opened": False,
        "paper_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "frozen_candidate_mutation": False,
    }
    hashes = {
        "dataset_sha256": manifest["dataset_sha256"],
        "input_contract_sha256": stable_hash(input_contract),
        "target_contract_sha256": stable_hash(target_contract),
        "frozen_feature_contract_sha256": file_sha256(frozen_contract_path),
        "source_manifest_sha256": stable_hash(source_rows),
    }
    write_json(run_dir / "indicator_companion_dataset_manifest.json", manifest)
    write_json(run_dir / "indicator_input_contract.json", input_contract)
    write_json(run_dir / "indicator_target_contract.json", target_contract)
    write_json(run_dir / "dataset_hashes.json", hashes)
    write_csv(run_dir / "indicator_target_audit.csv", target_rows)
    write_csv(run_dir / "indicator_feature_audit.csv", feature_rows)
    write_csv(run_dir / "purge_embargo_audit.csv", purge_rows)
    write_csv(run_dir / "source_manifest.csv", source_rows)
    print(f"dataset_dir={run_dir}")
    print(f"dataset_rows={manifest['rows']}")
    print(f"x_static_shape={manifest['x_static_shape']}")
    print(f"x_short_shape={manifest['x_short_shape']}")
    print(f"x_long_shape={manifest['x_long_shape']}")
    print(f"y_rsi_delta_shape={manifest['y_rsi_delta_shape']}")
    print(f"y_ma_state_shape={manifest['y_ma_state_shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
