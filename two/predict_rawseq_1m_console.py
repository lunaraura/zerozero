#!/usr/bin/env python3
"""Console predictor for the frozen rawseq 1m downside-risk candidate.

This is a thin command-line wrapper around
``run_rawseq_1m_live_paper_dashboard.py``. It reuses the frozen loading,
feature construction, hash verification, and inference functions from that
runner so the console view cannot drift from the dashboard implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.run_rawseq_1m_live_paper_dashboard import (  # noqa: E402
    DEFAULT_DOWNSIDE_DIR,
    DEFAULT_FIXED_CONTRACT,
    DEFAULT_INDICATOR_DIR,
    DEFAULT_SOURCE_DIR,
    EXPECTED_DOWNSIDE_HASH,
    EXPECTED_INDICATOR_HASH,
    SYMBOLS,
    WARMUP_ROWS_REQUIRED,
    append_ledger,
    iso_from_ms,
    load_candle_csv,
    load_frozen_packets,
    prediction_row,
    validate_completed_candles,
)

MODE_LATEST = "latest"
MODE_REPLAY = "replay"
MODE_WATCH = "watch"
SOURCE_KIND_1M_CANDLE = "1m_candle"
SOURCE_KIND_KRAKEN_10S_SNAPSHOT = "kraken_10s_snapshot"
INDICATOR_STATUS = "experimental_frozen_waiting_for_july_holdout"
FORBIDDEN_RECOMMENDATION_WORDS = ("buy", "sell", "long", "short", "safe", "guaranteed")
DEFAULT_KRAKEN_SOURCE_DIR = PROJECT_ROOT / "data" / "realtime" / "kraken"


def parse_utc_ms(value: str) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def utc_now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def source_lag_seconds(candle_open_ms: int, now_ms: int | None = None) -> float:
    now_ms = now_ms if now_ms is not None else utc_now_ms()
    return max(0.0, (float(now_ms) - float(candle_open_ms + 60_000)) / 1000.0)


def latest_completed_candle_frame(candles: pd.DataFrame, now_ms: int | None = None) -> pd.DataFrame:
    now_ms = now_ms if now_ms is not None else utc_now_ms()
    ts = pd.to_numeric(candles["timestamp_ms"], errors="coerce")
    completed = candles[ts + 60_000 <= now_ms].copy()
    if completed.empty:
        raise ValueError("no_completed_candles")
    return completed.reset_index(drop=True)


def source_integrity(candles: pd.DataFrame) -> dict[str, Any]:
    ts = pd.to_numeric(candles["timestamp_ms"], errors="coerce")
    diffs = ts.diff().dropna()
    suffix_len = 0
    if len(ts):
        suffix_len = 1
        for idx in range(len(ts) - 1, 0, -1):
            if int(ts.iloc[idx]) - int(ts.iloc[idx - 1]) == 60_000:
                suffix_len += 1
            else:
                break
    return {
        "source_rows_raw": int(len(candles)),
        "source_duplicate_timestamps": int(ts.duplicated().sum()),
        "source_missing_interval_count": int(((diffs > 60_000) & diffs.notna()).sum()),
        "source_out_of_order_count": int((diffs <= 0).sum()),
        "latest_strict_contiguous_minutes": int(suffix_len),
    }


def dedupe_sort_candles(candles: pd.DataFrame) -> pd.DataFrame:
    out = candles.copy()
    out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce")
    out = out.dropna(subset=["timestamp_ms"]).copy()
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    out = out.drop_duplicates(subset=["timestamp_ms"], keep="last").sort_values("timestamp_ms").reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True)
    return out


def is_historical_source_dir(source_dir: Path) -> bool:
    normalized = str(source_dir).replace("\\", "/").lower()
    return "binance_1m_candles_multi" in normalized


def symbol_list(symbol: str, symbols: str | None) -> list[str]:
    raw = symbols if symbols else symbol
    out = [item.strip().upper() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(out).difference(SYMBOLS))
    if unknown:
        raise ValueError(f"unsupported symbols: {','.join(unknown)}")
    return out or ["SOLUSDT"]


def load_packets(
    downside_dir: Path = DEFAULT_DOWNSIDE_DIR,
    indicator_dir: Path = DEFAULT_INDICATOR_DIR,
    fixed_contract_path: Path = DEFAULT_FIXED_CONTRACT,
) -> dict[str, Any]:
    return load_frozen_packets(downside_dir, indicator_dir, fixed_contract_path)


def snapshot_rows_to_minute_candles(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "mid_price"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"snapshot_source_missing_columns={','.join(missing)}")
    work = frame.copy()
    work["timestamp_ms"] = pd.to_numeric(work["timestamp"], errors="coerce")
    work["mid_price"] = pd.to_numeric(work["mid_price"], errors="coerce")
    if "total_trade_volume_10s" in work.columns:
        work["snapshot_volume"] = pd.to_numeric(work["total_trade_volume_10s"], errors="coerce").fillna(0.0)
    else:
        work["snapshot_volume"] = 0.0
    work = work.dropna(subset=["timestamp_ms", "mid_price"])
    work = work[work["mid_price"] > 0].copy()
    if work.empty:
        raise ValueError("snapshot_source_no_positive_mid_price_rows")
    work["minute_timestamp_ms"] = (work["timestamp_ms"].astype("int64") // 60_000) * 60_000
    grouped = work.groupby("minute_timestamp_ms", sort=True)
    candles = pd.DataFrame(
        {
            "timestamp_ms": grouped["minute_timestamp_ms"].first().astype("int64"),
            "open": grouped["mid_price"].first().astype(float),
            "high": grouped["mid_price"].max().astype(float),
            "low": grouped["mid_price"].min().astype(float),
            "close": grouped["mid_price"].last().astype(float),
            "volume": grouped["snapshot_volume"].sum().astype(float),
        }
    ).reset_index(drop=True)
    candles["timestamp"] = pd.to_datetime(candles["timestamp_ms"], unit="ms", utc=True)
    candles.attrs["source_kind"] = SOURCE_KIND_KRAKEN_10S_SNAPSHOT
    candles.attrs["source_state_timestamp_ms"] = int(work["timestamp_ms"].iloc[-1])
    candles.attrs["source_state_timestamp"] = iso_from_ms(int(work["timestamp_ms"].iloc[-1]))
    candles.attrs["source_adapter_note"] = "kraken 10s/1s snapshots resampled to minute OHLCV from mid_price"
    return candles[["timestamp_ms", "timestamp", "open", "high", "low", "close", "volume"]]


def load_symbol_candles(source_dir: Path, symbol: str, source_kind: str = SOURCE_KIND_1M_CANDLE) -> pd.DataFrame:
    if source_kind == SOURCE_KIND_1M_CANDLE:
        candles = load_candle_csv(source_dir / f"{symbol}_1m_flow.csv")
        candles.attrs["source_kind"] = SOURCE_KIND_1M_CANDLE
        return candles
    if source_kind == SOURCE_KIND_KRAKEN_10S_SNAPSHOT:
        frame = pd.read_csv(source_dir / f"{symbol}_10s_flow.csv")
        return snapshot_rows_to_minute_candles(frame)
    raise ValueError(f"unsupported_source_kind={source_kind}")


def _indicator_probabilities(row: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        "1m": float(row[f"{prefix}_ema_spread_narrowing_probability_1m"]),
        "2m": float(row[f"{prefix}_ema_spread_narrowing_probability_2m"]),
        "4m": float(row[f"{prefix}_ema_spread_narrowing_probability_4m"]),
        "8m": float(row[f"{prefix}_ema_spread_narrowing_probability_8m"]),
    }


def board_summary(row: dict[str, Any]) -> dict[str, Any]:
    members = np.asarray(
        [
            float(row["downside_pooled_logistic_probability"]),
            float(row["downside_regime_logistic_probability"]),
            float(row["downside_shallow_hgb_probability"]),
        ],
        dtype=float,
    )
    mean = float(np.mean(members))
    median = float(np.median(members))
    minimum = float(np.min(members))
    maximum = float(np.max(members))
    span = maximum - minimum
    std = float(np.std(members, ddof=0))
    baseline_probability = float(row.get("downside_event_prevalence_baseline", row.get("baseline_probability", 0.5)) or 0.5)
    above_reference = int(np.sum(members > baseline_probability))
    if above_reference == 0:
        consensus = "ALL_BELOW_BASELINE"
    elif above_reference == len(members):
        consensus = "ALL_ABOVE_BASELINE"
    else:
        consensus = "MIXED"
    data_degraded = (
        str(row.get("health_status", "")).upper() != "PASS"
        or str(row.get("monotonicity_health", "")).upper() != "PASS"
        or float(row.get("source_lag_seconds", 0.0) or 0.0) > 180.0
        or int(row.get("source_missing_interval_count", 0) or 0) > 0
        or int(row.get("source_duplicate_timestamps", 0) or 0) > 0
        or int(row.get("latest_strict_contiguous_minutes", 0) or 0) < 480
    )
    if data_degraded:
        board_state = "DEGRADED"
        operational_conclusion = "ABSTAIN"
    elif consensus == "MIXED" or std >= 0.08:
        board_state = "MIXED"
        operational_conclusion = "MONITOR"
    else:
        board_state = consensus
        operational_conclusion = "MONITOR"
    if std < 0.025 and span < 0.06:
        conviction = "LOW_DISAGREEMENT"
    elif std < 0.06:
        conviction = "MODERATE_DISAGREEMENT"
    else:
        conviction = "HIGH_DISAGREEMENT"
    return {
        "committee_mean": mean,
        "committee_median": median,
        "committee_min": minimum,
        "committee_max": maximum,
        "committee_range": span,
        "committee_standard_deviation": std,
        "committee_baseline_probability": baseline_probability,
        "committee_reference_label": "development_baseline_probability" if "downside_event_prevalence_baseline" in row or "baseline_probability" in row else "fallback_0p5_probability_not_operational_threshold",
        "committee_members_above_reference": above_reference,
        "committee_consensus": consensus,
        "data_quality_degraded": data_degraded,
        "board_state": board_state,
        "board_conviction": conviction,
        "operational_conclusion": operational_conclusion,
    }


def enrich_prediction_row(
    row: dict[str, Any],
    *,
    mode: str,
    historical_snapshot: bool,
    source_status: str,
    source_reasons: list[str] | None = None,
    source_integrity_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(row)
    out.update(
        {
            "prediction_mode": mode,
            "historical_snapshot": bool(historical_snapshot),
            "health_status": source_status,
            "health_reasons": ";".join(source_reasons or []),
            **(source_integrity_info or {}),
            "model_output": "P(next-minute downside excursion > 0.5 * trailing 240-minute volatility)",
            "indicator_status": INDICATOR_STATUS,
            "downside_candidate_hash_expected": EXPECTED_DOWNSIDE_HASH,
            "indicator_companion_hash_expected": EXPECTED_INDICATOR_HASH,
            "paper_only": True,
            "public_recorded_data_only": True,
            "public_market_data_only": True,
            "private_api": False,
            "orders": False,
            "execution": False,
            "promotion": False,
            "champion_mutation": False,
            "retraining": False,
            "recalibration": False,
            "candidate_mutation": False,
            "july_labels_computed": False,
            "july_metrics_computed": False,
        }
    )
    return out


def predict_from_candle_prefix(
    symbol: str,
    candles: pd.DataFrame,
    packets: dict[str, Any],
    *,
    mode: str,
    historical_snapshot: bool,
    now_ms: int | None = None,
    allow_open_latest: bool = False,
    allow_missing_intervals: bool = False,
    source_integrity_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(candles) < WARMUP_ROWS_REQUIRED:
        raise ValueError(f"warmup_failure rows={len(candles)} required={WARMUP_ROWS_REQUIRED}")
    check = validate_completed_candles(candles, now_ms=now_ms)
    if allow_open_latest and check["status"] != "PASS":
        reasons = [reason for reason in check["reasons"] if reason != "latest_candle_not_closed"]
        if not reasons:
            check = dict(check)
            check["status"] = "PASS"
            check["reasons"] = []
    if allow_missing_intervals and check["status"] != "PASS":
        reasons = [reason for reason in check["reasons"] if reason != "missing_one_minute_interval"]
        if not reasons:
            check = dict(check)
            check["status"] = "PASS"
            check["reasons"] = []
    if check["status"] != "PASS":
        raise ValueError(f"source_validation_failed reasons={','.join(check['reasons'])}")
    latest_ms = int(candles["timestamp_ms"].iloc[-1])
    row = prediction_row(symbol, candles, packets, source_lag_seconds(latest_ms, now_ms=now_ms))
    return enrich_prediction_row(
        row,
        mode=mode,
        historical_snapshot=historical_snapshot,
        source_status=str(check["status"]),
        source_reasons=list(check.get("reasons", [])),
        source_integrity_info=source_integrity_info,
    )


def predict_latest(
    symbol: str,
    source_dir: Path,
    packets: dict[str, Any],
    *,
    now_ms: int | None = None,
    source_kind: str = SOURCE_KIND_1M_CANDLE,
    allow_partial_latest: bool = False,
    allow_source_gaps: bool = False,
    historical_snapshot: bool | None = None,
) -> dict[str, Any]:
    candles = load_symbol_candles(source_dir, symbol, source_kind=source_kind)
    integrity = source_integrity(candles)
    if allow_source_gaps:
        candles = dedupe_sort_candles(candles)
    if not allow_partial_latest:
        candles = latest_completed_candle_frame(candles, now_ms=now_ms)
    row = predict_from_candle_prefix(
        symbol,
        candles,
        packets,
        mode=MODE_LATEST,
        historical_snapshot=is_historical_source_dir(source_dir) if historical_snapshot is None else historical_snapshot,
        now_ms=now_ms,
        allow_open_latest=allow_partial_latest,
        allow_missing_intervals=allow_source_gaps,
        source_integrity_info=integrity,
    )
    return add_source_metadata(row, candles, source_kind)


def replay_predictions(
    symbol: str,
    source_dir: Path,
    packets: dict[str, Any],
    *,
    start_ms: int,
    end_ms: int,
    max_rows: int = 0,
    now_ms: int | None = None,
    source_kind: str = SOURCE_KIND_1M_CANDLE,
) -> list[dict[str, Any]]:
    candles = load_symbol_candles(source_dir, symbol, source_kind=source_kind)
    ts = pd.to_numeric(candles["timestamp_ms"], errors="coerce")
    indexes = candles.index[(ts >= start_ms) & (ts <= end_ms)].tolist()
    if max_rows > 0:
        indexes = indexes[:max_rows]
    rows: list[dict[str, Any]] = []
    for idx in indexes:
        prefix = candles.iloc[: idx + 1].copy()
        rows.append(
            add_source_metadata(
                predict_from_candle_prefix(
                symbol,
                prefix,
                packets,
                mode=MODE_REPLAY,
                historical_snapshot=True,
                now_ms=now_ms,
                ),
                prefix,
                source_kind,
            )
        )
    return rows


def add_source_metadata(row: dict[str, Any], candles: pd.DataFrame, source_kind: str) -> dict[str, Any]:
    out = dict(row)
    out["source_kind"] = source_kind
    out["source_state_timestamp_ms"] = int(candles.attrs.get("source_state_timestamp_ms", out["candle_timestamp_ms"]))
    out["source_state_timestamp"] = str(candles.attrs.get("source_state_timestamp", out["candle_timestamp"]))
    out["source_adapter_note"] = str(candles.attrs.get("source_adapter_note", "native one-minute candle"))
    return out


def latest_ledger_timestamps(ledger_path: Path) -> dict[str, int]:
    if not ledger_path.exists() or ledger_path.stat().st_size == 0:
        return {}
    try:
        frame = pd.read_csv(ledger_path, on_bad_lines="skip")
    except Exception:
        return {}
    if "symbol" not in frame.columns or "candle_timestamp_ms" not in frame.columns:
        return {}
    if "source_kind" not in frame.columns:
        frame["source_kind"] = SOURCE_KIND_1M_CANDLE
    if "venue" not in frame.columns:
        frame["venue"] = ""
    if "price_basis" not in frame.columns:
        frame["price_basis"] = "close"
    frame["candle_timestamp_ms"] = pd.to_numeric(frame["candle_timestamp_ms"], errors="coerce")
    frame = frame.dropna(subset=["candle_timestamp_ms"])
    if frame.empty:
        return {}
    out: dict[str, int] = {}
    for key, group in frame.groupby(["venue", "source_kind", "price_basis", "symbol"], dropna=False):
        venue, kind, basis, symbol = key
        out[f"{venue}|{kind}|{basis}|{symbol}"] = int(group["candle_timestamp_ms"].max())
    return out


def prediction_identity_key(symbol: str, source_kind: str, venue: str = "", price_basis: str = "close") -> str:
    return f"{venue}|{source_kind}|{price_basis}|{symbol}"


def watch_once(
    symbols: list[str],
    source_dir: Path,
    packets: dict[str, Any],
    *,
    ledger_path: Path | None,
    last_predicted_ms: dict[str, int] | None = None,
    stale_seconds: float = 180.0,
    allow_stale_source: bool = False,
    now_ms: int | None = None,
    source_kind: str = SOURCE_KIND_1M_CANDLE,
    allow_partial_latest: bool = False,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    last_predicted_ms = dict(last_predicted_ms or {})
    if ledger_path is not None:
        for key, ts in latest_ledger_timestamps(ledger_path).items():
            last_predicted_ms[key] = max(last_predicted_ms.get(key, 0), ts)
    rows: list[dict[str, Any]] = []
    notices: list[str] = []
    for symbol in symbols:
        try:
            candles = load_symbol_candles(source_dir, symbol, source_kind=source_kind)
            check = validate_completed_candles(candles, now_ms=now_ms)
            if allow_partial_latest and check["status"] != "PASS":
                reasons = [reason for reason in check["reasons"] if reason != "latest_candle_not_closed"]
                if not reasons:
                    check = dict(check)
                    check["status"] = "PASS"
                    check["reasons"] = []
            latest_ms = int(candles["timestamp_ms"].iloc[-1])
            state_ms = int(candles.attrs.get("source_state_timestamp_ms", latest_ms))
            if check["status"] != "PASS":
                notices.append(f"{symbol} waiting source_status=FAIL reasons={','.join(check['reasons'])}")
                continue
            lag = float(check["source_lag_seconds"])
            if stale_seconds > 0 and lag > stale_seconds and not allow_stale_source:
                notices.append(f"{symbol} waiting stale_source lag_seconds={lag:.1f} threshold_seconds={stale_seconds:.1f}")
                continue
            key = prediction_identity_key(symbol, source_kind)
            if latest_ms <= int(last_predicted_ms.get(key, -1)):
                notices.append(f"{symbol} waiting unchanged latest={iso_from_ms(latest_ms)}")
                continue
            row = add_source_metadata(
                predict_from_candle_prefix(
                symbol,
                candles,
                packets,
                mode=MODE_WATCH,
                historical_snapshot=False,
                now_ms=now_ms,
                allow_open_latest=allow_partial_latest,
                ),
                candles,
                source_kind,
            )
            rows.append(row)
            last_predicted_ms[key] = latest_ms
        except Exception as exc:
            notices.append(f"{symbol} waiting inference_blocked reason={exc!r}")
    if ledger_path is not None and rows:
        append_ledger(ledger_path, rows)
    return rows, notices, last_predicted_ms


def json_projection(row: dict[str, Any]) -> dict[str, Any]:
    board = board_summary(row)
    return {
        "timestamp": row["candle_timestamp"],
        "timestamp_ms": int(row["candle_timestamp_ms"]),
        "symbol": row["symbol"],
        "close": float(row["close"]),
        "downside_probabilities": {
            "pooled_logistic": float(row["downside_pooled_logistic_probability"]),
            "pooled_regime_feature_logistic": float(row["downside_regime_logistic_probability"]),
            "pooled_shallow_hgb": float(row["downside_shallow_hgb_probability"]),
            "ensemble": float(row["downside_ensemble_probability"]),
        },
        "model_board": board,
        "indicator_raw_probabilities": _indicator_probabilities(row, "raw"),
        "indicator_corrected_probabilities": _indicator_probabilities(row, "corrected"),
        "current_ema20": float(row["current_ema20"]),
        "current_ema60": float(row["current_ema60"]),
        "current_ema_spread_bps": float(row["current_ema_spread_bps"]),
        "current_rsi14": float(row["current_rsi14"]),
        "downside_candidate_hash": row["downside_candidate_hash"],
        "indicator_companion_hash": row["indicator_companion_hash"],
        "fixed_feature_contract_hash": row["fixed_feature_contract_hash"],
        "prediction_mode": row["prediction_mode"],
        "historical_snapshot": bool(row["historical_snapshot"]),
        "source_kind": row.get("source_kind", SOURCE_KIND_1M_CANDLE),
        "source_state_timestamp": row.get("source_state_timestamp", row["candle_timestamp"]),
        "source_adapter_note": row.get("source_adapter_note", "native one-minute candle"),
        "source_rows_raw": int(row.get("source_rows_raw", 0) or 0),
        "source_duplicate_timestamps": int(row.get("source_duplicate_timestamps", 0) or 0),
        "source_missing_interval_count": int(row.get("source_missing_interval_count", 0) or 0),
        "latest_strict_contiguous_minutes": int(row.get("latest_strict_contiguous_minutes", 0) or 0),
        "source_lag_seconds": float(row["source_lag_seconds"]),
        "health_status": row["health_status"],
        "indicator_status": row["indicator_status"],
        "monotonicity_health": row["monotonicity_health"],
        "paper_only": True,
        "public_recorded_data_only": True,
        "public_market_data_only": True,
        "private_api": False,
        "orders": False,
        "execution": False,
        "promotion": False,
        "champion_mutation": False,
        "retraining": False,
        "recalibration": False,
        "candidate_mutation": False,
        "july_labels_computed": False,
        "july_metrics_computed": False,
    }


def format_readable(row: dict[str, Any]) -> str:
    raw = _indicator_probabilities(row, "raw")
    corrected = _indicator_probabilities(row, "corrected")
    board = board_summary(row)
    mode_text = f"{row['prediction_mode']} historical snapshot" if row["historical_snapshot"] else row["prediction_mode"]
    source_status = "historical snapshot" if row["historical_snapshot"] else str(row["health_status"])
    source_kind = str(row.get("source_kind", SOURCE_KIND_1M_CANDLE))
    return "\n".join(
        [
            "RAWSEQ MODEL BOARD",
            f"Mode: {mode_text}",
            f"Symbol: {row['symbol']}",
            f"Candle: {row['candle_timestamp']}",
            f"Close: {float(row['close']):.4f}",
            "",
            "DOWNSIDE COMMITTEE",
            f"Pooled logistic:  {float(row['downside_pooled_logistic_probability']):.2%}",
            f"Regime logistic:  {float(row['downside_regime_logistic_probability']):.2%}",
            f"Shallow HGB:      {float(row['downside_shallow_hgb_probability']):.2%}",
            f"Chair mean:       {board['committee_mean']:.2%}",
            f"Median:           {board['committee_median']:.2%}",
            f"Range:            {100.0 * board['committee_range']:.2f} pp",
            f"Disagreement:     {100.0 * board['committee_standard_deviation']:.2f} pp",
            f"Consensus:        {board['committee_consensus']}",
            "",
            f"MARKET-STRUCTURE MEMBER: {INDICATOR_STATUS}",
            f"Within 1m:        {corrected['1m']:.2%}",
            f"Within 2m:        {corrected['2m']:.2%}",
            f"Within 4m:        {corrected['4m']:.2%}",
            f"Within 8m:        {corrected['8m']:.2%}",
            f"Structural view:  {'STRONG' if corrected['8m'] >= 0.75 else ('MODERATE' if corrected['8m'] >= 0.55 else 'WEAK_OR_UNCERTAIN')}",
            "",
            "Current indicators:",
            f"EMA20:            {float(row['current_ema20']):.4f}",
            f"EMA60:            {float(row['current_ema60']):.4f}",
            f"EMA spread:       {float(row['current_ema_spread_bps']):.3f} bps",
            f"RSI14:            {float(row['current_rsi14']):.4f}",
            "",
            "Health:",
            f"Downside hash:    {'PASS' if row['downside_candidate_hash'] == EXPECTED_DOWNSIDE_HASH else 'FAIL'}",
            f"Indicator hash:   {'PASS' if row['indicator_companion_hash'] == EXPECTED_INDICATOR_HASH else 'FAIL'}",
            f"Monotonicity:     {row['monotonicity_health']}",
            "Warmup:           PASS",
            f"Source status:    {source_status}",
            f"Source kind:      {source_kind}",
            f"Source state:     {row.get('source_state_timestamp', row['candle_timestamp'])}",
            f"Source lag:       {float(row['source_lag_seconds']):.1f}s",
            (
                "Source integrity: "
                f"duplicates={int(row.get('source_duplicate_timestamps', 0) or 0)}, "
                f"gaps={int(row.get('source_missing_interval_count', 0) or 0)}, "
                f"strict_latest_minutes={int(row.get('latest_strict_contiguous_minutes', 0) or 0)}"
            ),
            "Paper only:       true",
            "public_recorded_data_only=true | private_api=false | orders=false | execution=false | promotion=false",
            "champion_mutation=false | candidate_mutation=false | July labels=false | July metrics=false",
            "",
            "BOARD SUMMARY",
            f"State:            {board['board_state']}",
            f"Conviction:       {board['board_conviction']}",
            f"Data quality:     {'DEGRADED' if board['data_quality_degraded'] else 'PASS'}",
            f"Operational conclusion: {board['operational_conclusion']}",
            "",
            "Primary model: P(next-minute downside excursion > 0.5 x trailing 240-minute volatility)",
            f"Hashes: downside={row['downside_candidate_hash']} indicator={row['indicator_companion_hash']}",
            (
                "Raw EMA-spread narrowing="
                f"1m {raw['1m']:.4%}, 2m {raw['2m']:.4%}, 4m {raw['4m']:.4%}, 8m {raw['8m']:.4%}"
            ),
        ]
    )


def format_compact(row: dict[str, Any]) -> str:
    corrected = _indicator_probabilities(row, "corrected")
    mode_text = "hist" if row["historical_snapshot"] else row["prediction_mode"]
    return (
        f"{row['candle_timestamp']} {row['symbol']} close={float(row['close']):.4f} "
        f"downside={float(row['downside_ensemble_probability']):.4f} "
        f"ema_narrow=[{corrected['1m']:.4f},{corrected['2m']:.4f},{corrected['4m']:.4f},{corrected['8m']:.4f}] "
        f"status={row['inference_status']} historical={str(bool(row['historical_snapshot'])).lower()} "
        f"mode={mode_text} "
        f"source_kind={row.get('source_kind', SOURCE_KIND_1M_CANDLE)} "
        f"dups={int(row.get('source_duplicate_timestamps', 0) or 0)} "
        f"gaps={int(row.get('source_missing_interval_count', 0) or 0)} "
        f"strict_mins={int(row.get('latest_strict_contiguous_minutes', 0) or 0)} "
        f"components=({float(row['downside_pooled_logistic_probability']):.6f},"
        f"{float(row['downside_regime_logistic_probability']):.6f},"
        f"{float(row['downside_shallow_hgb_probability']):.6f}) "
        f"lag_s={float(row['source_lag_seconds']):.1f} paper_only=true"
    )


def emit_row(row: dict[str, Any], *, compact: bool, json_lines: bool) -> None:
    if json_lines:
        print(json.dumps(json_projection(row), sort_keys=True, allow_nan=False))
    elif compact:
        print(format_compact(row))
    else:
        print(format_readable(row))
        print()


def write_output_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen rawseq 1m console predictor")
    parser.add_argument("--mode", choices=[MODE_LATEST, MODE_REPLAY, MODE_WATCH], default=MODE_LATEST)
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--source-kind", choices=[SOURCE_KIND_1M_CANDLE, SOURCE_KIND_KRAKEN_10S_SNAPSHOT], default=SOURCE_KIND_1M_CANDLE)
    parser.add_argument("--allow-partial-latest", action="store_true")
    parser.add_argument("--allow-source-gaps", action="store_true")
    parser.add_argument("--downside-dir", type=Path, default=DEFAULT_DOWNSIDE_DIR)
    parser.add_argument("--indicator-dir", type=Path, default=DEFAULT_INDICATOR_DIR)
    parser.add_argument("--fixed-contract", type=Path, default=DEFAULT_FIXED_CONTRACT)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_lines")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--step-seconds", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--stale-seconds", type=float, default=180.0)
    parser.add_argument("--allow-stale-source", action="store_true")
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=0, help="watch-mode test bound; 0 means run until interrupted")
    return parser


def run_latest(args: argparse.Namespace, packets: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        predict_latest(
            symbol,
            args.source_dir,
            packets,
            source_kind=args.source_kind,
            allow_partial_latest=args.allow_partial_latest,
            allow_source_gaps=args.allow_source_gaps,
            historical_snapshot=is_historical_source_dir(args.source_dir),
        )
        for symbol in symbol_list(args.symbol, args.symbols)
    ]
    for row in rows:
        emit_row(row, compact=args.compact, json_lines=args.json_lines)
    return rows


def run_replay(args: argparse.Namespace, packets: dict[str, Any]) -> list[dict[str, Any]]:
    if not args.start or not args.end:
        raise ValueError("--start and --end are required for replay mode")
    symbols = symbol_list(args.symbol, args.symbols)
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        symbol_rows = replay_predictions(
            symbol,
            args.source_dir,
            packets,
            start_ms=parse_utc_ms(args.start),
            end_ms=parse_utc_ms(args.end),
            max_rows=args.max_rows,
            source_kind=args.source_kind,
        )
        for row in symbol_rows:
            emit_row(row, compact=args.compact, json_lines=args.json_lines)
            if args.step_seconds > 0:
                time.sleep(args.step_seconds)
        rows.extend(symbol_rows)
    if args.output_csv:
        write_output_csv(args.output_csv, rows)
    return rows


def run_watch(args: argparse.Namespace, packets: dict[str, Any]) -> list[dict[str, Any]]:
    if args.allow_partial_latest and os.getenv("RAWSEQ_1M_ALLOW_PARTIAL_LATEST_WATCH_DIAGNOSTIC", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        raise ValueError("--allow-partial-latest is diagnostic-only and disabled in watch mode; set RAWSEQ_1M_ALLOW_PARTIAL_LATEST_WATCH_DIAGNOSTIC=true to override")
    symbols = symbol_list(args.symbol, args.symbols)
    last_predicted: dict[str, int] = {}
    emitted: list[dict[str, Any]] = []
    cycles = 0
    while True:
        rows, notices, last_predicted = watch_once(
            symbols,
            args.source_dir,
            packets,
            ledger_path=args.ledger_path,
            last_predicted_ms=last_predicted,
            stale_seconds=args.stale_seconds,
            allow_stale_source=args.allow_stale_source,
            source_kind=args.source_kind,
            allow_partial_latest=args.allow_partial_latest,
        )
        for row in rows:
            emit_row(row, compact=args.compact, json_lines=args.json_lines)
        if not rows and notices and not args.json_lines:
            for notice in notices:
                print(notice if args.compact else f"Watch status: {notice}")
        emitted.extend(rows)
        cycles += 1
        if args.max_cycles > 0 and cycles >= args.max_cycles:
            return emitted
        time.sleep(max(1.0, args.poll_seconds))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packets = load_packets(args.downside_dir, args.indicator_dir, args.fixed_contract)
    if args.mode == MODE_LATEST:
        rows = run_latest(args, packets)
    elif args.mode == MODE_REPLAY:
        rows = run_replay(args, packets)
    else:
        rows = run_watch(args, packets)
    if args.output_csv and args.mode != MODE_REPLAY:
        write_output_csv(args.output_csv, rows)
    return 0 if rows or args.mode == MODE_WATCH else 2


if __name__ == "__main__":
    raise SystemExit(main())
