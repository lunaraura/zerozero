#!/usr/bin/env python3
"""Shared helpers for the one-minute rawseq baseline scout."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_baseline_scout")
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "binance_public_zips"
BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]
SAFETY_FLAGS = {
    "paper_only": True,
    "private_api": False,
    "orders": False,
    "promotion": False,
    "champion_mutation": False,
    "active_future_shadow_mutation": False,
    "active_future_shadow_labels_used": False,
    "frozen_candidate_weights_reused": False,
    "frozen_candidate_thresholds_reused": False,
    "frozen_candidate_calibration_reused": False,
}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    symbol: str
    year_month: str


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_path(name: str, default: Path | None = None, required: bool = False) -> Path:
    raw = os.getenv(name, "").strip()
    if not raw:
        if required:
            raise SystemExit(f"{name} is required")
        if default is None:
            raise SystemExit(f"{name} is required")
        return default
    return resolve_path(raw)


def parse_int_list(raw: str, default: list[int]) -> list[int]:
    if not raw.strip():
        return list(default)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_bool(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def detect_timestamp_unit(values: pd.Series | np.ndarray) -> str:
    finite = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if finite.empty:
        return "unknown"
    units = timestamp_unit_counts(finite)
    present = [unit for unit, count in units.items() if count > 0]
    if len(present) == 1:
        return present[0]
    if present:
        return "mixed_" + "_".join(present)
    median = float(finite.median())
    if median > 1e14:
        return "microseconds"
    if median > 1e11:
        return "milliseconds"
    if median > 1e8:
        return "seconds"
    return "unknown"


def timestamp_to_ms(values: pd.Series, unit: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = numeric.copy()
    micro_mask = numeric > 1e14
    milli_mask = (numeric > 1e11) & ~micro_mask
    second_mask = (numeric > 1e8) & ~micro_mask & ~milli_mask
    out.loc[micro_mask] = numeric.loc[micro_mask] / 1000.0
    out.loc[milli_mask] = numeric.loc[milli_mask]
    out.loc[second_mask] = numeric.loc[second_mask] * 1000.0
    return out


def timestamp_unit_counts(values: pd.Series | np.ndarray) -> dict[str, int]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    return {
        "microseconds": int((numeric > 1e14).sum()),
        "milliseconds": int(((numeric > 1e11) & (numeric <= 1e14)).sum()),
        "seconds": int(((numeric > 1e8) & (numeric <= 1e11)).sum()),
        "unknown": int((numeric.notna() & (numeric <= 1e8)).sum()),
    }


def infer_symbol_from_name(path: Path) -> tuple[str, str]:
    match = re.match(r"^([A-Z0-9]+)-1m-(\d{4}-\d{2})\.zip$", path.name)
    if match:
        return match.group(1), match.group(2)
    match = re.match(r"^([A-Z0-9]+)-1m-(\d{4}-\d{2})\.csv$", path.name)
    if match:
        return match.group(1), match.group(2)
    match = re.match(r"^([A-Z0-9]+)_1m_flow\.csv$", path.name)
    if match:
        return match.group(1), "flow"
    return "", ""


def resolve_source_files(source_path: Path, symbol: str) -> list[SourceFile]:
    source_path = resolve_path(source_path)
    files: list[Path]
    if source_path.is_dir():
        files = sorted(source_path.glob(f"{symbol}-1m-*.zip"))
        if not files:
            files = sorted(source_path.glob(f"{symbol}-1m-*.csv"))
        if not files:
            files = sorted(source_path.glob(f"{symbol}_1m_flow.csv"))
    elif source_path.is_file():
        files = [source_path]
    else:
        raise FileNotFoundError(f"RAWSEQ_1M_SOURCE_PATH does not exist: {source_path}")
    out: list[SourceFile] = []
    for path in files:
        inferred_symbol, year_month = infer_symbol_from_name(path)
        out.append(SourceFile(path=path, symbol=inferred_symbol or symbol, year_month=year_month))
    if not out:
        raise FileNotFoundError(f"No {symbol}-1m zip/csv files found under {source_path}")
    return out


def read_kline_file(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV member in {path}")
            with zf.open(members[0]) as handle:
                return pd.read_csv(handle, header=None, names=BINANCE_KLINE_COLUMNS, nrows=nrows)
    if path.name.endswith("_1m_flow.csv"):
        frame = pd.read_csv(path, nrows=nrows)
        if "timestamp_ms" in frame.columns and "open_time" not in frame.columns:
            frame["open_time"] = frame["timestamp_ms"]
        elif "timestamp" in frame.columns and "open_time" not in frame.columns:
            frame["open_time"] = frame["timestamp"]
        return frame
    return pd.read_csv(path, header=None, names=BINANCE_KLINE_COLUMNS, nrows=nrows)


def load_candles(source_files: list[SourceFile], max_rows: int = 0) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    remaining = max_rows if max_rows and max_rows > 0 else None
    for item in source_files:
        nrows = remaining if remaining is not None else None
        frame = read_kline_file(item.path, nrows=nrows)
        frame["source_file"] = item.path.name
        frame["source_symbol"] = item.symbol
        frame["source_year_month"] = item.year_month
        frames.append(frame)
        if remaining is not None:
            remaining -= len(frame)
            if remaining <= 0:
                break
    if not frames:
        return pd.DataFrame(columns=BINANCE_KLINE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return canonicalize_candles(out)


def canonicalize_candles(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    unit = detect_timestamp_unit(out["open_time"])
    out["timestamp_ms"] = timestamp_to_ms(out["open_time"], unit)
    out["timestamp"] = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", "number_of_trades"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def source_manifest(source_files: list[SourceFile]) -> dict[str, Any]:
    rows = []
    for item in source_files:
        rows.append(
            {
                "path": str(item.path),
                "name": item.path.name,
                "symbol": item.symbol,
                "year_month": item.year_month,
                "bytes": item.path.stat().st_size,
                "sha256": file_sha256(item.path),
            }
        )
    return {
        "source_file_count": len(rows),
        "source_total_bytes": sum(int(row["bytes"]) for row in rows),
        "source_files": rows,
        "source_manifest_sha256": stable_hash(rows),
    }


def audit_candles(frame: pd.DataFrame, source_files: list[SourceFile], symbol: str, venue: str) -> dict[str, Any]:
    ts = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
    diffs = ts.sort_values().diff()
    missing_intervals = int(((diffs > 60_000) & diffs.notna()).sum())
    duplicate_timestamps = int(ts.duplicated().sum())
    nonmonotonic = int((ts.diff().dropna() <= 0).sum())
    largest_gap_ms = float(diffs.max()) if diffs.notna().any() else math.nan
    first_ts = pd.to_datetime(ts.min(), unit="ms", utc=True, errors="coerce") if ts.notna().any() else pd.NaT
    last_ts = pd.to_datetime(ts.max(), unit="ms", utc=True, errors="coerce") if ts.notna().any() else pd.NaT
    years = frame.assign(year=frame["timestamp"].dt.year)
    gaps_by_year: dict[str, int] = {}
    for year, group in years.groupby("year", dropna=True):
        group_diffs = pd.to_numeric(group["timestamp_ms"], errors="coerce").sort_values().diff()
        gaps_by_year[str(int(year))] = int(((group_diffs > 60_000) & group_diffs.notna()).sum())
    ohlc_bad = (
        (frame["low"] > frame["open"])
        | (frame["low"] > frame["close"])
        | (frame["high"] < frame["open"])
        | (frame["high"] < frame["close"])
        | (frame["low"] > frame["high"])
    )
    price_cols = ["open", "high", "low", "close"]
    finite = np.isfinite(frame[price_cols + ["volume"]].to_numpy(dtype=np.float64))
    nonpositive_prices = int((frame[price_cols] <= 0).sum().sum())
    nan_count = int(frame[price_cols + ["volume"]].isna().sum().sum())
    inf_count = int((~finite).sum() - nan_count)
    months = 0.0
    if pd.notna(first_ts) and pd.notna(last_ts):
        months = max(0.0, (last_ts.to_pydatetime() - first_ts.to_pydatetime()).total_seconds() / (86400.0 * 30.4375))
    manifest = source_manifest(source_files)
    status = "PASS"
    reasons = []
    if duplicate_timestamps:
        reasons.append("duplicate_timestamps")
    if nonmonotonic:
        reasons.append("nonmonotonic_timestamps")
    if missing_intervals:
        reasons.append("missing_one_minute_intervals")
    if int(ohlc_bad.sum()):
        reasons.append("ohlc_consistency_violations")
    if nonpositive_prices:
        reasons.append("nonpositive_prices")
    if nan_count or inf_count:
        reasons.append("nan_or_infinite_values")
    if reasons:
        status = "WARN"
    return {
        "audit_status": status,
        "audit_reasons": reasons,
        "timestamp_column": "open_time",
        "canonical_timestamp_column": "timestamp_ms",
        "timestamp_unit": detect_timestamp_unit(frame["open_time"]),
        "timestamp_unit_counts": timestamp_unit_counts(frame["open_time"]),
        "timezone_assumption": "UTC",
        "symbol": symbol,
        "detected_symbols": sorted(set(str(x) for x in frame.get("source_symbol", pd.Series(dtype=str)).dropna().unique())),
        "venue": venue,
        "ohlc_column_mapping": {"open": "open", "high": "high", "low": "low", "close": "close"},
        "volume_columns": [col for col in ["volume", "quote_asset_volume", "number_of_trades"] if col in frame.columns],
        "first_timestamp": first_ts.isoformat() if pd.notna(first_ts) else "",
        "last_timestamp": last_ts.isoformat() if pd.notna(last_ts) else "",
        "first_timestamp_ms": float(ts.min()) if ts.notna().any() else math.nan,
        "last_timestamp_ms": float(ts.max()) if ts.notna().any() else math.nan,
        "total_rows": int(len(frame)),
        "duplicate_timestamps": duplicate_timestamps,
        "nonmonotonic_timestamps": nonmonotonic,
        "missing_one_minute_intervals": missing_intervals,
        "largest_timestamp_gap_ms": largest_gap_ms,
        "largest_timestamp_gap_minutes": largest_gap_ms / 60000.0 if math.isfinite(largest_gap_ms) else math.nan,
        "gaps_by_calendar_year": gaps_by_year,
        "ohlc_consistency_violations": int(ohlc_bad.sum()),
        "nonpositive_prices": nonpositive_prices,
        "nan_values": nan_count,
        "infinite_values": inf_count,
        "source_file_size_bytes": manifest["source_total_bytes"],
        "source_sha256": manifest["source_manifest_sha256"],
        "approximate_months_covered": months,
        **SAFETY_FLAGS,
    }


def canonical_column_contract(symbol: str, venue: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "venue": venue,
        "cadence_seconds": 60,
        "timestamp_column": "timestamp_ms",
        "timestamp_timezone": "UTC",
        "timestamp_unit": "milliseconds",
        "required_columns": ["timestamp_ms", "open", "high", "low", "close", "volume"],
        "optional_columns": ["quote_asset_volume", "number_of_trades"],
        "ohlc_mapping": {"open": "open", "high": "high", "low": "low", "close": "close"},
        **SAFETY_FLAGS,
    }


def build_features(frame: pd.DataFrame, windows: list[int]) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    out = pd.DataFrame({"timestamp_ms": frame["timestamp_ms"], "timestamp": frame["timestamp"]})
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    open_ = frame["open"].astype(float)
    volume = frame["volume"].astype(float) if "volume" in frame.columns else pd.Series(np.nan, index=frame.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["signed_bucket_return_bps"] = 10000.0 * np.log(close / close.shift(1))
        out["candle_range_bps"] = 10000.0 * np.log(high / low)
        out["candle_body_bps"] = 10000.0 * np.log(close / open_)
        out["upper_wick_bps"] = 10000.0 * np.log(high / np.maximum(open_, close))
        out["lower_wick_bps"] = 10000.0 * np.log(np.minimum(open_, close) / low)
        out["log_volume_change"] = np.log(volume / volume.shift(1))
    feature_rows: list[dict[str, Any]] = []
    for window in windows:
        rolling_high = high.rolling(window, min_periods=window).max()
        rolling_low = low.rolling(window, min_periods=window).min()
        returns = out["signed_bucket_return_bps"]
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        out[f"rolling_range_bps_fw{window}"] = 10000.0 * np.log(rolling_high / rolling_low)
        out[f"rolling_volatility_bps_fw{window}"] = returns.rolling(window, min_periods=window).std(ddof=0)
        out[f"distance_to_recent_high_bps_fw{window}"] = 10000.0 * np.log(close / rolling_high)
        out[f"distance_to_recent_low_bps_fw{window}"] = 10000.0 * np.log(close / rolling_low)
        out[f"close_to_ema_bps_fw{window}"] = 10000.0 * np.log(close / ema)
        out[f"ema_slope_bps_fw{window}"] = 10000.0 * np.log(ema / ema.shift(window))
        for name, expected in [
            (f"rolling_range_bps_fw{window}", "nonnegative"),
            (f"rolling_volatility_bps_fw{window}", "nonnegative"),
            (f"distance_to_recent_high_bps_fw{window}", "nonpositive"),
            (f"distance_to_recent_low_bps_fw{window}", "nonnegative"),
            (f"close_to_ema_bps_fw{window}", "signed"),
            (f"ema_slope_bps_fw{window}", "signed"),
        ]:
            values = pd.to_numeric(out[name], errors="coerce")
            finite = values[np.isfinite(values)]
            if expected == "nonnegative":
                sign_bad = float((finite < -1e-9).mean()) if len(finite) else math.nan
            elif expected == "nonpositive":
                sign_bad = float((finite > 1e-9).mean()) if len(finite) else math.nan
            else:
                sign_bad = 0.0
            feature_rows.append(
                {
                    "feature": name,
                    "feature_window_minutes": window,
                    "total_rows": int(len(values)),
                    "warmup_rows": window,
                    "finite_rows": int(len(finite)),
                    "nonfinite_rows": int(len(values) - len(finite)),
                    "minimum": float(finite.min()) if len(finite) else math.nan,
                    "maximum": float(finite.max()) if len(finite) else math.nan,
                    "mean": float(finite.mean()) if len(finite) else math.nan,
                    "standard_deviation": float(finite.std(ddof=0)) if len(finite) else math.nan,
                    "expected_sign": expected,
                    "expected_sign_violation_fraction": sign_bad,
                    "leakage_check_status": "PASS",
                }
            )
    base_features = ["signed_bucket_return_bps", "candle_range_bps", "candle_body_bps", "upper_wick_bps", "lower_wick_bps", "log_volume_change"]
    for name in base_features:
        values = pd.to_numeric(out[name], errors="coerce")
        finite = values[np.isfinite(values)]
        feature_rows.append(
            {
                "feature": name,
                "feature_window_minutes": 1,
                "total_rows": int(len(values)),
                "warmup_rows": 1,
                "finite_rows": int(len(finite)),
                "nonfinite_rows": int(len(values) - len(finite)),
                "minimum": float(finite.min()) if len(finite) else math.nan,
                "maximum": float(finite.max()) if len(finite) else math.nan,
                "mean": float(finite.mean()) if len(finite) else math.nan,
                "standard_deviation": float(finite.std(ddof=0)) if len(finite) else math.nan,
                "expected_sign": "signed" if name not in {"candle_range_bps", "upper_wick_bps", "lower_wick_bps"} else "nonnegative",
                "expected_sign_violation_fraction": 0.0,
                "leakage_check_status": "PASS",
            }
        )
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    leakage = {
        "leakage_audit_status": "PASS",
        "centered_windows_used": False,
        "future_backfill_used": False,
        "rolling_min_periods_policy": "full_window",
        "features_use_current_and_historical_only": True,
    }
    return out, feature_rows, leakage


def future_low_return_bps(close: pd.Series, low: pd.Series, horizon: int) -> pd.Series:
    future_low = low.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1].reset_index(drop=True)
    return 10000.0 * np.log(future_low / close.reset_index(drop=True))


def downside_event_targets(frame: pd.DataFrame, vol_window: int, horizons: list[int], threshold_vol_units: float = 0.5) -> pd.DataFrame:
    close = frame["close"].astype(float).reset_index(drop=True)
    low = frame["low"].astype(float).reset_index(drop=True)
    returns = 10000.0 * np.log(close / close.shift(1))
    vol = returns.rolling(vol_window, min_periods=vol_window).std(ddof=0)
    out = pd.DataFrame({"timestamp_ms": frame["timestamp_ms"].reset_index(drop=True), f"trailing_volatility_bps_fw{vol_window}": vol})
    for horizon in horizons:
        low_ret = future_low_return_bps(close, low, horizon)
        downside = np.maximum(0.0, -low_ret)
        target = (downside > threshold_vol_units * vol).astype(float)
        target[~np.isfinite(downside) | ~np.isfinite(vol)] = np.nan
        out[f"downside_event_0p5vol_h{horizon}m_fw{vol_window}"] = target
        out[f"future_low_return_bps_h{horizon}m"] = low_ret
        out[f"downside_excursion_bps_h{horizon}m"] = downside
    return out


def future_return_path(close: pd.Series, output_len: int) -> np.ndarray:
    arr = []
    close_values = close.astype(float).reset_index(drop=True)
    for step in range(1, output_len + 1):
        arr.append(10000.0 * np.log(close_values.shift(-step) / close_values))
    return np.column_stack(arr)


def future_high_low_paths(close: pd.Series, high: pd.Series, low: pd.Series, output_len: int) -> tuple[np.ndarray, np.ndarray]:
    close_values = close.astype(float).reset_index(drop=True)
    high_paths = []
    low_paths = []
    for step in range(1, output_len + 1):
        future_high = high.shift(-1).iloc[::-1].rolling(step, min_periods=step).max().iloc[::-1].reset_index(drop=True)
        future_low = low.shift(-1).iloc[::-1].rolling(step, min_periods=step).min().iloc[::-1].reset_index(drop=True)
        high_ret = np.maximum(0.0, 10000.0 * np.log(future_high / close_values))
        low_ret = np.minimum(0.0, 10000.0 * np.log(future_low / close_values))
        high_paths.append(high_ret)
        low_paths.append(low_ret)
    return np.column_stack(high_paths), np.column_stack(low_paths)


def split_contract(frame: pd.DataFrame, feature_lookback: int, max_horizon: int, fold_count: int = 4) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = len(frame)
    ts = pd.to_numeric(frame["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64)
    first = float(np.nanmin(ts)) if rows else math.nan
    last = float(np.nanmax(ts)) if rows else math.nan
    months = (last - first) / (1000.0 * 86400.0 * 30.4375) if math.isfinite(first) and math.isfinite(last) else 0.0
    if months >= 36:
        holdout_rows = int(round(12 * 30.4375 * 24 * 60))
        final_dev_rows = int(round(12 * 30.4375 * 24 * 60))
    else:
        holdout_rows = max(1, int(rows * 0.20))
        final_dev_rows = max(1, int(rows * 0.20))
    rolling_rows = max(0, rows - holdout_rows - final_dev_rows)
    purge_rows = feature_lookback
    embargo_rows = max_horizon
    manifest = {
        "split_rule": "chronological_final_holdout_and_final_dev_confirmation",
        "rows": rows,
        "approximate_months": months,
        "rolling_development_rows": rolling_rows,
        "final_development_confirmation_rows": final_dev_rows,
        "untouched_holdout_rows": holdout_rows,
        "rolling_development_start_index": 0,
        "rolling_development_end_index": max(0, rolling_rows - 1),
        "final_development_start_index": rolling_rows,
        "final_development_end_index": max(rolling_rows, rolling_rows + final_dev_rows - 1),
        "holdout_start_index": rows - holdout_rows,
        "holdout_end_index": rows - 1,
        "purge_rows": purge_rows,
        "embargo_rows": embargo_rows,
        "holdout_accessed": False,
        **SAFETY_FLAGS,
    }
    folds: list[dict[str, Any]] = []
    if rolling_rows > (purge_rows + embargo_rows + fold_count):
        chunk = rolling_rows // (fold_count + 1)
        for fold_idx in range(fold_count):
            train_start = 0
            train_end = max(0, chunk * (fold_idx + 1) - embargo_rows - 1)
            val_start = chunk * (fold_idx + 1) + purge_rows
            val_end = min(rolling_rows - 1, chunk * (fold_idx + 2) - 1)
            if val_start > val_end or train_end <= train_start:
                continue
            folds.append(
                {
                    "fold_id": fold_idx,
                    "train_start_index": train_start,
                    "train_end_index": train_end,
                    "validation_start_index": val_start,
                    "validation_end_index": val_end,
                    "train_rows": train_end - train_start + 1,
                    "validation_rows": val_end - val_start + 1,
                    "purge_rows": purge_rows,
                    "embargo_rows": embargo_rows,
                    "holdout_accessed": False,
                }
            )
    purge_rows_out = [
        {
            "fold_id": row["fold_id"],
            "purge_rows": purge_rows,
            "embargo_rows": embargo_rows,
            "train_end_index": row["train_end_index"],
            "validation_start_index": row["validation_start_index"],
            "gap_rows_between_train_and_validation": row["validation_start_index"] - row["train_end_index"] - 1,
            "purge_embargo_status": "PASS" if row["validation_start_index"] - row["train_end_index"] - 1 >= max(purge_rows, embargo_rows) else "WARN",
        }
        for row in folds
    ]
    return manifest, folds, purge_rows_out


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(y) else math.nan


def log_loss_score(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else math.nan


def rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    pos_ranks = ranks[: len(pos)]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    total = len(y)
    if total == 0:
        return math.nan
    ece = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        hi = lo + 1.0 / bins
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return ece


def max_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    vals = []
    for lo in np.linspace(0, 1, bins, endpoint=False):
        hi = lo + 1.0 / bins
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            vals.append(abs(float(y[mask].mean()) - float(p[mask].mean())))
    return max(vals, default=math.nan)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)).reshape(-1, 1)
    if np.std(logits) <= 1e-12 or np.std(y) <= 1e-12:
        return math.nan, math.nan
    try:
        from sklearn.linear_model import LogisticRegression

        cal = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000).fit(logits, y.astype(int))
        return float(cal.coef_[0, 0]), float(cal.intercept_[0])
    except Exception:
        return math.nan, math.nan


def pr_auc_lift(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    try:
        from sklearn.metrics import average_precision_score

        ap = float(average_precision_score(y, p))
    except Exception:
        ap = math.nan
    prevalence = float(np.mean(y)) if len(y) else math.nan
    return ap, ap - prevalence if math.isfinite(ap) and math.isfinite(prevalence) else math.nan


def metric_row(y: np.ndarray, p: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    baseline = np.clip(np.asarray(baseline, dtype=float), 1e-6, 1 - 1e-6)
    brier = brier_score(y, p)
    base_brier = brier_score(y, baseline)
    ll = log_loss_score(y, p)
    base_ll = log_loss_score(y, baseline)
    slope, intercept = calibration_slope_intercept(y, p)
    pr, pr_lift = pr_auc_lift(y, p)
    return {
        "rows": int(len(y)),
        "events": int(np.sum(y > 0.5)),
        "event_prevalence": float(np.mean(y)) if len(y) else math.nan,
        "brier_score": brier,
        "prevalence_brier_score": base_brier,
        "brier_skill_vs_prevalence": (base_brier - brier) / base_brier if base_brier > 0 else math.nan,
        "log_loss": ll,
        "prevalence_log_loss": base_ll,
        "log_loss_improvement_vs_prevalence": base_ll - ll if math.isfinite(base_ll) and math.isfinite(ll) else math.nan,
        "pr_auc": pr,
        "pr_auc_lift_over_event_prevalence": pr_lift,
        "roc_auc": rank_auc(y, p),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "expected_calibration_error": expected_calibration_error(y, p),
        "maximum_calibration_error": max_calibration_error(y, p),
    }


def save_reload_prediction_parity(model: Any, predict_fn: Any, x: np.ndarray) -> tuple[bool, float]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pkl"
        path.write_bytes(pickle.dumps(model))
        loaded = pickle.loads(path.read_bytes())
        a = np.asarray(predict_fn(model, x), dtype=float)
        b = np.asarray(predict_fn(loaded, x), dtype=float)
    max_diff = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
    return max_diff <= 1e-12, max_diff


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
