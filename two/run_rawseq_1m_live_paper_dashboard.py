#!/usr/bin/env python3
"""Run frozen 1m downside-risk and indicator-event models in live-paper mode.

The runner consumes local public one-minute candle CSVs, normally the Kraken
files emitted by the realtime recorder. It writes an
append-only prediction ledger plus a compact dashboard state bundle. It never
trains, recalibrates, labels July outcomes, places orders, or mutates either
frozen model packet.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import signal
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.build_rawseq_1m_indicator_companion_dataset import (  # noqa: E402
    LONG_LEN,
    LONG_STRIDE,
    SHORT_LEN,
    TEMPORAL_CHANNELS,
    companion_indicator_frame,
)
from scripts.tiny.freeze_rawseq_1m_indicator_event_companion import cumulative_hazard_reconstruction  # noqa: E402
from scripts.tiny.rawseq_1m_baseline_utils import (  # noqa: E402
    build_features,
    file_sha256,
    now_stamp,
    stable_hash,
    write_json,
)
from scripts.tiny.report_rawseq_1m_cross_asset_panel_scout import predict_model, regime_feature_indices  # noqa: E402
from scripts.tiny.run_rawseq_1m_dual_timescale_indicator_scout import feature_matrix  # noqa: E402
from scripts.tiny.run_rawseq_1m_indicator_event_scout import predict_proba  # noqa: E402

SYMBOLS = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT"]
EXPECTED_DOWNSIDE_HASH = "8109480bd6b0c2575337cada9be991746199200d42feaeea1fd7b9dd0b3eb6bf"
EXPECTED_INDICATOR_HASH = "ce4136501bf53b5b05f53e613fc0bb5c15ee0cae5de13665ef4cf86494b10752"
EXPECTED_INDICATOR_MODEL_HASH = "b5bdb191e66b24061e278e1b74110f9ce7ad2afcfb26df8e8b9c8eed9ef29d8a"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "realtime" / "kraken"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_1m_live_paper_dashboard"
DEFAULT_DASHBOARD_STATE = PROJECT_ROOT / "docs" / "rawseq_1m_risk_dashboard_data.js"
DEFAULT_DOWNSIDE_DIR = Path(
    r"F:\rsio\rawseq_1m_cross_asset_scout\rawseq_1m_pooled_candidate_confirmation_20260712T150734Z\frozen_pooled_multisymbol_challenger"
)
DEFAULT_INDICATOR_DIR = Path(r"F:\rsio\rawseq_1m_indicator_companion_scout\frozen_indicator_event_companion_20260712T200053Z")
DEFAULT_FIXED_CONTRACT = Path(r"F:\rsio\rawseq_1m_cross_asset_scout\rawseq_1m_fixed_transfer_contract_20260712T131949Z\fixed_transfer_contract.json")
COMPONENTS = ["pooled_logistic", "pooled_regime_feature_logistic", "pooled_shallow_hgb"]
WARMUP_ROWS_REQUIRED = 480
DEFAULT_MAX_SOURCE_LAG_SECONDS = 180.0
DEFAULT_DASHBOARD_ROLLING_ROWS = 720
JULY_START_MS = int(pd.Timestamp("2026-07-01T00:00:00Z").timestamp() * 1000)
JULY_END_MS = int(pd.Timestamp("2026-08-01T00:00:00Z").timestamp() * 1000)
SHUTDOWN = False


def _handle_shutdown(signum: int, frame: Any) -> None:
    del signum, frame
    global SHUTDOWN
    SHUTDOWN = True


signal.signal(signal.SIGINT, _handle_shutdown)
signal.signal(signal.SIGTERM, _handle_shutdown)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    text = str(value)
    return default if text.lower() in {"nan", "nat", "none"} else text


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
    source_lag = safe_float(row.get("source_lag_seconds")) or 0.0
    missing_intervals = int(safe_float(row.get("source_missing_interval_count")) or 0)
    duplicate_timestamps = int(safe_float(row.get("source_duplicate_timestamps")) or 0)
    strict_minutes = int(safe_float(row.get("latest_strict_contiguous_minutes")) or 0)
    data_degraded = (
        str(row.get("health_status", row.get("inference_status", ""))).upper() != "PASS"
        or str(row.get("monotonicity_health", "")).upper() != "PASS"
        or source_lag > 180.0
        or missing_intervals > 0
        or duplicate_timestamps > 0
        or strict_minutes < 480
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


def iso_from_ms(timestamp_ms: int | float) -> str:
    return pd.to_datetime(int(timestamp_ms), unit="ms", utc=True).isoformat()


def verify_embedded_hash(payload: dict[str, Any], hash_key: str) -> tuple[bool, str]:
    expected = str(payload.get(hash_key, ""))
    copy = dict(payload)
    copy.pop(hash_key, None)
    actual = stable_hash(copy)
    return expected == actual, actual


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent), prefix=".tmp_", suffix=".txt") as handle:
        handle.write(text)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def load_candle_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} missing timestamp column")
    out = frame.copy()
    out["timestamp_ms"] = pd.to_numeric(out["timestamp"], errors="coerce").astype("Int64")
    out["timestamp"] = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            raise ValueError(f"{path} missing {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["timestamp_ms", "timestamp", "open", "high", "low", "close", "volume"]).reset_index(drop=True)


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


def validate_completed_candles(frame: pd.DataFrame, now_ms: int | None = None) -> dict[str, Any]:
    now_ms = now_ms or int(datetime.now(UTC).timestamp() * 1000)
    ts = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
    diffs = ts.diff().dropna()
    ohlc_bad = (
        (frame["low"] > frame["open"])
        | (frame["low"] > frame["close"])
        | (frame["high"] < frame["open"])
        | (frame["high"] < frame["close"])
        | (frame["low"] > frame["high"])
    )
    latest_open = int(ts.iloc[-1]) if len(ts) else 0
    latest_close = latest_open + 60_000
    reasons: list[str] = []
    if ts.duplicated().any():
        reasons.append("duplicate_timestamps")
    if not ts.is_monotonic_increasing:
        reasons.append("out_of_order_timestamps")
    if ((diffs > 60_000) & diffs.notna()).any():
        reasons.append("missing_one_minute_interval")
    if int(ohlc_bad.sum()):
        reasons.append("invalid_ohlc")
    if int((frame[["open", "high", "low", "close"]] <= 0).sum().sum()):
        reasons.append("nonpositive_price")
    if latest_close > now_ms:
        reasons.append("latest_candle_not_closed")
    return {
        "rows": int(len(frame)),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "out_of_order_count": int((ts.diff().dropna() <= 0).sum()),
        "missing_minute_count": int(((diffs > 60_000) & diffs.notna()).sum()),
        "latest_candle_open_ms": latest_open,
        "latest_candle_close_ms": latest_close,
        "latest_candle_closed": latest_close <= now_ms,
        "latest_timestamp": iso_from_ms(latest_open) if latest_open else "",
        "source_lag_seconds": max(0.0, (now_ms - latest_close) / 1000.0) if latest_open else math.nan,
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
    }


def prepare_feature_frames(candles: pd.DataFrame, fixed_contract: dict[str, Any], input_contract: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = [int(x) for x in fixed_contract["feature_windows_minutes"]]
    static_features, _, _ = build_features(candles, windows)
    indicators = companion_indicator_frame(candles)
    static_features["trailing_volatility_bps_fw240"] = indicators["signed_close_return_bps"].rolling(240, min_periods=240).std(ddof=0)
    for col in input_contract["static_feature_order"]:
        if col not in static_features.columns:
            raise ValueError(f"missing static feature {col}")
    for col in TEMPORAL_CHANNELS:
        if col not in indicators.columns:
            raise ValueError(f"missing temporal feature {col}")
    return static_features, indicators


def latest_companion_matrix(static_features: pd.DataFrame, indicators: pd.DataFrame, input_contract: dict[str, Any]) -> tuple[np.ndarray, int, dict[str, Any]]:
    idx = len(indicators) - 1
    short_idx = np.arange(idx - SHORT_LEN + 1, idx + 1)
    long_idx = idx - np.arange(LONG_LEN - 1, -1, -1) * LONG_STRIDE
    if short_idx[0] < 0 or long_idx[0] < 0:
        raise ValueError("insufficient_companion_context")
    static = static_features[input_contract["static_feature_order"]].to_numpy(dtype=np.float32)
    temporal = indicators[TEMPORAL_CHANNELS].to_numpy(dtype=np.float32)
    payload = {
        "x_static": static[idx : idx + 1],
        "x_short": temporal[short_idx][None, :, :],
        "x_long": temporal[long_idx][None, :, :],
    }
    if any(not np.isfinite(value).all() for value in payload.values()):
        raise ValueError("nonfinite_companion_features")
    x = feature_matrix(payload, np.asarray([0], dtype=int))
    meta = {
        "current_ema20": float(indicators["ema20"].iloc[idx]),
        "current_ema60": float(indicators["ema60"].iloc[idx]),
        "current_ema_spread_bps": float(indicators["ema20_minus_ema60_bps"].iloc[idx]),
        "current_rsi14": float(indicators["rsi14"].iloc[idx]),
    }
    return x, idx, meta


def latest_downside_matrix(static_features: pd.DataFrame, fixed_contract: dict[str, Any]) -> tuple[np.ndarray, int]:
    feature_cols = list(fixed_contract["model_feature_names_and_order"])
    idx = len(static_features) - 1
    x = static_features[feature_cols].iloc[[idx]].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite_downside_features")
    return x, idx


def load_frozen_packets(downside_dir: Path, indicator_dir: Path, fixed_contract_path: Path) -> dict[str, Any]:
    downside_contract = read_json(downside_dir / "pooled_candidate_contract.json")
    indicator_contract = read_json(indicator_dir / "indicator_event_companion_contract.json")
    fixed_contract = read_json(fixed_contract_path)
    input_contract = read_json(Path(indicator_contract["input_contract_path"]))
    downside_ok, downside_recomputed = verify_embedded_hash(downside_contract, "candidate_hash")
    indicator_ok, indicator_recomputed = verify_embedded_hash(indicator_contract, "companion_contract_hash")
    downside_model_path = Path(downside_contract["model_path"])
    indicator_model_path = Path(indicator_contract["model_artifact_path"])
    checks = {
        "downside_candidate_hash_expected": downside_contract.get("candidate_hash") == EXPECTED_DOWNSIDE_HASH,
        "downside_candidate_embedded_hash_ok": downside_ok,
        "indicator_companion_hash_expected": indicator_contract.get("companion_contract_hash") == EXPECTED_INDICATOR_HASH,
        "indicator_companion_embedded_hash_ok": indicator_ok,
        "indicator_model_artifact_hash_expected": file_sha256(indicator_model_path) == EXPECTED_INDICATOR_MODEL_HASH,
        "fixed_feature_contract_hash_expected": downside_contract.get("fixed_transfer_contract_hash") == fixed_contract.get("fixed_transfer_contract_hash"),
        "indicator_monotonic_method_expected": indicator_contract.get("selected_monotonic_correction_method") == "cumulative_hazard_reconstruction",
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen hash/contract verification failed: {checks}")
    return {
        "downside_contract": downside_contract,
        "indicator_contract": indicator_contract,
        "fixed_contract": fixed_contract,
        "input_contract": input_contract,
        "downside_models": pickle.loads(downside_model_path.read_bytes()),
        "indicator_payload": pickle.loads(indicator_model_path.read_bytes()),
        "hash_checks": checks,
        "recomputed_hashes": {
            "downside_candidate": downside_recomputed,
            "indicator_companion": indicator_recomputed,
            "indicator_model_artifact": file_sha256(indicator_model_path),
            "fixed_contract": file_sha256(fixed_contract_path),
            "input_contract": file_sha256(Path(indicator_contract["input_contract_path"])),
        },
    }


def predict_downside(models: dict[str, Any], contract: dict[str, Any], fixed_contract: dict[str, Any], x: np.ndarray) -> tuple[float, dict[str, float]]:
    weights = contract["component_weights"]
    regime_idx = regime_feature_indices(list(fixed_contract["model_feature_names_and_order"]))
    components = {
        "pooled_logistic": float(np.clip(predict_model(models["pooled_logistic"], x)[0], 1e-6, 1 - 1e-6)),
        "pooled_regime_feature_logistic": float(np.clip(predict_model(models["pooled_regime_feature_logistic"], x[:, regime_idx])[0], 1e-6, 1 - 1e-6)),
        "pooled_shallow_hgb": float(np.clip(predict_model(models["pooled_shallow_hgb"], x)[0], 1e-6, 1 - 1e-6)),
    }
    ensemble = float(sum(float(weights[name]) * components[name] for name in COMPONENTS))
    return float(np.clip(ensemble, 1e-6, 1 - 1e-6)), components


def predict_indicator(indicator_payload: dict[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    events = indicator_payload["selected_events"]
    raw = np.asarray([predict_proba(indicator_payload["models"][event], x)[0] for event in events], dtype=float)[None, :]
    corrected = cumulative_hazard_reconstruction(raw)
    monotonic = bool((np.diff(corrected, axis=1) >= -1e-12).all())
    return raw.reshape(-1), corrected.reshape(-1), monotonic


def ledger_identity(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("venue", "")),
        str(row.get("source_kind", "")),
        str(row.get("price_basis", "close")),
        str(row.get("symbol", "")),
        str(row.get("candle_timestamp", "")),
    )


def read_existing_ledger_keys(path: Path) -> set[tuple[str, str, str, str, str]]:
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return set()
    if frame.empty:
        return set()
    for col, default in [("venue", ""), ("source_kind", ""), ("price_basis", "close"), ("symbol", ""), ("candle_timestamp", "")]:
        if col not in frame.columns:
            frame[col] = default
    return set(
        zip(
            frame["venue"].astype(str),
            frame["source_kind"].astype(str),
            frame["price_basis"].astype(str),
            frame["symbol"].astype(str),
            frame["candle_timestamp"].astype(str),
        )
    )


def ledger_rows(path: Path, symbols: list[str], limit_per_symbol: int) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        frame = pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return []
    if frame.empty:
        return []
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame[frame["symbol"].isin(symbols)].copy()
    if frame.empty:
        return []
    frame["candle_timestamp_ms"] = pd.to_numeric(frame["candle_timestamp_ms"], errors="coerce")
    frame = frame.dropna(subset=["candle_timestamp_ms"]).sort_values(["symbol", "candle_timestamp_ms"])
    if limit_per_symbol > 0:
        frame = frame.groupby("symbol", group_keys=False).tail(limit_per_symbol)
    return frame.to_dict("records")


def append_ledger(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing_ledger_keys(path)
    for row in rows:
        row.setdefault("price_basis", "close")
    new_rows = [row for row in rows if ledger_identity(row) not in existing]
    if not new_rows:
        return 0
    fieldnames: list[str] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
    for row in new_rows:
        for key in row:
            if key not in fieldnames and not path.exists():
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = sorted({key for row in new_rows for key in row})
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in new_rows:
            writer.writerow(row)
    return len(new_rows)


def july_prediction_access(rows: list[dict[str, Any]]) -> dict[str, bool]:
    has_july_prediction = False
    for row in rows:
        try:
            ts = int(float(row.get("candle_timestamp_ms", 0)))
        except (TypeError, ValueError):
            ts = 0
        if JULY_START_MS <= ts < JULY_END_MS:
            has_july_prediction = True
            break
    return {
        "july_files_opened": has_july_prediction,
        "july_timestamps_enumerated": has_july_prediction,
        "july_features_computed": has_july_prediction,
        "july_predictions_computed": has_july_prediction,
        "july_labels_computed": False,
        "july_prevalence_computed": False,
        "july_metrics_computed": False,
        "july_acceptance_evaluated": False,
    }


def prediction_row(symbol: str, candles: pd.DataFrame, packets: dict[str, Any], source_lag_seconds: float) -> dict[str, Any]:
    static_features, indicators = prepare_feature_frames(candles, packets["fixed_contract"], packets["input_contract"])
    x_down, idx = latest_downside_matrix(static_features, packets["fixed_contract"])
    x_ind, ind_idx, indicator_meta = latest_companion_matrix(static_features, indicators, packets["input_contract"])
    if idx != ind_idx:
        raise ValueError("downside_indicator_row_mismatch")
    downside_ensemble, downside_components = predict_downside(
        packets["downside_models"], packets["downside_contract"], packets["fixed_contract"], x_down
    )
    raw_indicator, corrected_indicator, monotonic = predict_indicator(packets["indicator_payload"], x_ind)
    candle = candles.iloc[idx]
    now_iso = datetime.now(UTC).isoformat()
    row = {
        "prediction_timestamp": now_iso,
        "candle_timestamp": iso_from_ms(int(candle["timestamp_ms"])),
        "candle_timestamp_ms": int(candle["timestamp_ms"]),
        "symbol": symbol,
        "open": float(candle["open"]),
        "high": float(candle["high"]),
        "low": float(candle["low"]),
        "close": float(candle["close"]),
        "volume": float(candle["volume"]),
        "downside_pooled_logistic_probability": downside_components["pooled_logistic"],
        "downside_regime_logistic_probability": downside_components["pooled_regime_feature_logistic"],
        "downside_shallow_hgb_probability": downside_components["pooled_shallow_hgb"],
        "downside_ensemble_probability": downside_ensemble,
        "raw_ema_spread_narrowing_probability_1m": float(raw_indicator[0]),
        "raw_ema_spread_narrowing_probability_2m": float(raw_indicator[1]),
        "raw_ema_spread_narrowing_probability_4m": float(raw_indicator[2]),
        "raw_ema_spread_narrowing_probability_8m": float(raw_indicator[3]),
        "corrected_ema_spread_narrowing_probability_1m": float(corrected_indicator[0]),
        "corrected_ema_spread_narrowing_probability_2m": float(corrected_indicator[1]),
        "corrected_ema_spread_narrowing_probability_4m": float(corrected_indicator[2]),
        "corrected_ema_spread_narrowing_probability_8m": float(corrected_indicator[3]),
        "current_ema20": indicator_meta["current_ema20"],
        "current_ema60": indicator_meta["current_ema60"],
        "current_ema_spread_bps": indicator_meta["current_ema_spread_bps"],
        "current_rsi14": indicator_meta["current_rsi14"],
        "source_lag_seconds": source_lag_seconds,
        "source_kind": "1m_candle",
        "downside_candidate_hash": packets["downside_contract"]["candidate_hash"],
        "indicator_companion_hash": packets["indicator_contract"]["companion_contract_hash"],
        "fixed_feature_contract_hash": packets["fixed_contract"]["fixed_transfer_contract_hash"],
        "input_contract_sha256": packets["recomputed_hashes"]["input_contract"],
        "monotonicity_health": "PASS" if monotonic else "FAIL",
        "inference_status": "PASS" if monotonic else "FAIL",
        "paper_only": True,
        "public_market_data_only": True,
        "private_api": False,
        "orders": False,
        "execution": False,
        "promotion": False,
        "champion_mutation": False,
        "downside_candidate_mutation": False,
        "indicator_companion_mutation": False,
        "july_labels": False,
        "july_metrics": False,
    }
    return row


def state_payload(
    rows: list[dict[str, Any]],
    source_health: list[dict[str, Any]],
    packets: dict[str, Any],
    ledger_path: Path,
    current_prediction_rows: int | None = None,
) -> dict[str, Any]:
    series: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        board = board_summary(row)
        series.setdefault(safe_text(row.get("symbol"), "UNKNOWN"), []).append(
            {
                "timestamp": safe_text(row.get("candle_timestamp")),
                "open": safe_float(row["open"]),
                "high": safe_float(row["high"]),
                "low": safe_float(row["low"]),
                "close": safe_float(row["close"]),
                "volume": safe_float(row["volume"]),
                "ema20": safe_float(row["current_ema20"]),
                "ema60": safe_float(row["current_ema60"]),
                "ema_spread_bps": safe_float(row["current_ema_spread_bps"]),
                "ensemble": safe_float(row["downside_ensemble_probability"]),
                "pooled_logistic": safe_float(row["downside_pooled_logistic_probability"]),
                "pooled_regime_logistic": safe_float(row["downside_regime_logistic_probability"]),
                "pooled_shallow_hgb": safe_float(row["downside_shallow_hgb_probability"]),
                "indicator_raw_1m": safe_float(row["raw_ema_spread_narrowing_probability_1m"]),
                "indicator_raw_2m": safe_float(row["raw_ema_spread_narrowing_probability_2m"]),
                "indicator_raw_4m": safe_float(row["raw_ema_spread_narrowing_probability_4m"]),
                "indicator_raw_8m": safe_float(row["raw_ema_spread_narrowing_probability_8m"]),
                "indicator_corrected_1m": safe_float(row["corrected_ema_spread_narrowing_probability_1m"]),
                "indicator_corrected_2m": safe_float(row["corrected_ema_spread_narrowing_probability_2m"]),
                "indicator_corrected_4m": safe_float(row["corrected_ema_spread_narrowing_probability_4m"]),
                "indicator_corrected_8m": safe_float(row["corrected_ema_spread_narrowing_probability_8m"]),
                "source_kind": safe_text(row.get("source_kind"), "1m_candle"),
                "source_rows_raw": safe_float(row.get("source_rows_raw")),
                "source_duplicate_timestamps": safe_float(row.get("source_duplicate_timestamps")),
                "source_missing_interval_count": safe_float(row.get("source_missing_interval_count")),
                "latest_strict_contiguous_minutes": safe_float(row.get("latest_strict_contiguous_minutes")),
                "committee_mean": safe_float(board["committee_mean"]),
                "committee_median": safe_float(board["committee_median"]),
                "committee_min": safe_float(board["committee_min"]),
                "committee_max": safe_float(board["committee_max"]),
                "committee_range": safe_float(board["committee_range"]),
                "committee_standard_deviation": safe_float(board["committee_standard_deviation"]),
                "committee_members_above_reference": board["committee_members_above_reference"],
                "committee_consensus": board["committee_consensus"],
                "data_quality_degraded": board["data_quality_degraded"],
                "board_state": board["board_state"],
                "board_conviction": board["board_conviction"],
                "operational_conclusion": board["operational_conclusion"],
            }
        )
    symbols = sorted(series)
    july_access = july_prediction_access(rows)
    return {
        "created_at": now_stamp(),
        "mode": "live_paper",
        "dashboard_role": "Live-paper frozen downside-risk and indicator-event dashboard",
        "model_output": "P(next-minute downside excursion > 0.5 * trailing 240-minute volatility) plus P(EMA20-EMA60 spread narrows within horizon)",
        "live_inference_ready": bool(len(rows) if current_prediction_rows is None else current_prediction_rows),
        "symbols": symbols,
        "default_symbol": "SOLUSDT" if "SOLUSDT" in symbols else (symbols[0] if symbols else ""),
        "candidate_hash": packets["downside_contract"]["candidate_hash"],
        "indicator_companion_hash": packets["indicator_contract"]["companion_contract_hash"],
        "component_model_hashes": packets["hash_checks"],
        "ensemble_weights": packets["downside_contract"]["component_weights"],
        "weighting": packets["downside_contract"]["weighting"],
        "calibration": packets["downside_contract"]["calibration"],
        "ledger_path": str(ledger_path),
        "health": {
            "candidate_hash_match": packets["downside_contract"]["candidate_hash"] == EXPECTED_DOWNSIDE_HASH,
            "indicator_companion_hash_match": packets["indicator_contract"]["companion_contract_hash"] == EXPECTED_INDICATOR_HASH,
            "monotonicity_all_pass": all(row["monotonicity_health"] == "PASS" for row in rows),
            "component_probabilities_present": all(row["inference_status"] == "PASS" for row in rows),
            "source_freshness_all_pass": all(row.get("stale_source_status", "PASS") == "PASS" for row in source_health),
            "source_freshness_gate_pass": all(
                row.get("stale_source_status", "PASS") == "PASS" or bool(row.get("allow_stale_source_override", False))
                for row in source_health
            ),
            "warmup_all_ready": all(bool(row.get("prediction_ready", False)) for row in source_health) if source_health else False,
            **july_access,
            "paper_only": True,
            "public_market_data_only": True,
            "orders": False,
            "execution": False,
            "promotion": False,
        },
        "source_health": source_health,
        "series": series,
    }


def run_once() -> dict[str, Any]:
    started = time.perf_counter()
    source_dir = Path(os.getenv("RAWSEQ_1M_LIVE_SOURCE_DIR", str(DEFAULT_SOURCE_DIR)))
    downside_dir = Path(os.getenv("RAWSEQ_1M_LIVE_DOWNSIDE_DIR", str(DEFAULT_DOWNSIDE_DIR)))
    indicator_dir = Path(os.getenv("RAWSEQ_1M_LIVE_INDICATOR_DIR", str(DEFAULT_INDICATOR_DIR)))
    fixed_contract_path = Path(os.getenv("RAWSEQ_1M_LIVE_FIXED_CONTRACT", str(DEFAULT_FIXED_CONTRACT)))
    output_root = Path(os.getenv("RAWSEQ_1M_LIVE_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    ledger_path = Path(os.getenv("RAWSEQ_1M_LIVE_LEDGER_PATH", str(output_root / "rawseq_1m_live_paper_prediction_ledger.csv")))
    state_path = Path(os.getenv("RAWSEQ_1M_LIVE_DASHBOARD_STATE_PATH", str(DEFAULT_DASHBOARD_STATE)))
    symbols = [x.strip().upper() for x in os.getenv("RAWSEQ_1M_LIVE_SYMBOLS", ",".join(SYMBOLS)).split(",") if x.strip()]
    max_source_lag = float(os.getenv("RAWSEQ_1M_LIVE_MAX_SOURCE_LAG_SECONDS", str(DEFAULT_MAX_SOURCE_LAG_SECONDS)) or str(DEFAULT_MAX_SOURCE_LAG_SECONDS))
    allow_stale_source = os.getenv("RAWSEQ_1M_LIVE_ALLOW_STALE_SOURCE", "false").strip().lower() in {"1", "true", "yes", "on"}
    allow_source_gaps = os.getenv("RAWSEQ_1M_LIVE_ALLOW_SOURCE_GAPS", "false").strip().lower() in {"1", "true", "yes", "on"}
    rolling_rows = int(os.getenv("RAWSEQ_1M_LIVE_DASHBOARD_ROLLING_ROWS", str(DEFAULT_DASHBOARD_ROLLING_ROWS)) or str(DEFAULT_DASHBOARD_ROLLING_ROWS))
    max_symbols = int(os.getenv("RAWSEQ_1M_LIVE_MAX_SYMBOLS", "0") or "0")
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    packets = load_frozen_packets(downside_dir, indicator_dir, fixed_contract_path)
    rows: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        path = source_dir / f"{symbol}_1m_flow.csv"
        health: dict[str, Any] = {"symbol": symbol, "path": str(path), "file_exists": path.exists()}
        try:
            candles = load_candle_csv(path)
            integrity = source_integrity(candles)
            if allow_source_gaps:
                candles = dedupe_sort_candles(candles)
            check = validate_completed_candles(candles)
            if allow_source_gaps and check["status"] != "PASS":
                remaining_reasons = [reason for reason in check["reasons"] if reason != "missing_one_minute_interval"]
                if not remaining_reasons:
                    check = dict(check)
                    check["status"] = "PASS"
                    check["reasons"] = []
            health.update(check)
            health.update(integrity)
            health["warmup_rows_required"] = WARMUP_ROWS_REQUIRED
            health["warmup_rows_available"] = int(len(candles))
            health["latest_strict_contiguous_minutes_required"] = WARMUP_ROWS_REQUIRED
            health["allow_source_gaps"] = allow_source_gaps
            health["stale_source_threshold_seconds"] = max_source_lag
            health["allow_stale_source_override"] = allow_stale_source
            health["stale_source"] = bool(max_source_lag > 0 and float(check["source_lag_seconds"]) > max_source_lag)
            health["stale_source_status"] = "FAIL" if health["stale_source"] else "PASS"
            health["prediction_ready"] = bool(
                check["status"] == "PASS"
                and len(candles) >= WARMUP_ROWS_REQUIRED
                and int(integrity.get("latest_strict_contiguous_minutes", 0) or 0) >= WARMUP_ROWS_REQUIRED
                and (not health["stale_source"] or allow_stale_source)
            )
            if not health["prediction_ready"]:
                health["inference_status"] = "FAIL_CLOSED"
                health_rows.append(health)
                continue
            row = prediction_row(symbol, candles, packets, float(check["source_lag_seconds"]))
            row.update(integrity)
            rows.append(row)
            health["inference_status"] = row["inference_status"]
        except Exception as exc:
            health["inference_status"] = "FAIL_CLOSED"
            health["failure_reason"] = repr(exc)
        health_rows.append(health)
        if SHUTDOWN:
            break
    appended = append_ledger(ledger_path, rows)
    dashboard_rows = ledger_rows(ledger_path, symbols, rolling_rows) or rows
    payload = state_payload(dashboard_rows, health_rows, packets, ledger_path, current_prediction_rows=len(rows))
    atomic_write_text(state_path, "window.RAWSEQ_RISK_DASHBOARD_DATA = " + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";\n")
    peak_mb = math.nan
    try:
        import psutil  # type: ignore

        peak_mb = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        pass
    manifest = {
        "created_at": now_stamp(),
        "mode": "live_paper",
        "connection_state": "local_public_candle_file_polling",
        "reconnect_policy": "loop mode retries on the next interval; public recorder owns exchange reconnects",
        "source_dir": str(source_dir),
        "symbols_requested": symbols,
        "prediction_rows": len(rows),
        "dashboard_series_rows": len(dashboard_rows),
        "ledger_rows_appended": appended,
        "ledger_path": str(ledger_path),
        "dashboard_state_path": str(state_path),
        "max_source_lag_seconds": max_source_lag,
        "allow_stale_source_override": allow_stale_source,
        "allow_source_gaps": allow_source_gaps,
        "dashboard_rolling_rows_per_symbol": rolling_rows,
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_working_set_mb": peak_mb,
        "health": health_rows,
        "hash_checks": packets["hash_checks"],
        "final_recommendation": "live_paper_dashboard_ready_for_monitoring" if rows else "live_paper_dashboard_blocked",
        "safety": payload["health"],
    }
    out_dir = output_root / f"rawseq_1m_live_paper_dashboard_{now_stamp()}"
    write_json(out_dir / "rawseq_1m_live_paper_dashboard_manifest.json", manifest)
    write_json(state_path.with_suffix(".manifest.json"), {"dashboard_state_path": str(state_path), "sha256": file_sha256(state_path), **manifest})
    print(f"live_manifest_dir={out_dir}")
    print(f"ledger_path={ledger_path}")
    print(f"dashboard_state_path={state_path}")
    print(f"prediction_rows={len(rows)}")
    print(f"ledger_rows_appended={appended}")
    print(f"final_recommendation={manifest['final_recommendation']}")
    return manifest


def main() -> int:
    loop = os.getenv("RAWSEQ_1M_LIVE_LOOP", "false").strip().lower() in {"1", "true", "yes", "on"}
    interval = int(os.getenv("RAWSEQ_1M_LIVE_INTERVAL_SECONDS", "60") or "60")
    while True:
        manifest = run_once()
        if not loop or SHUTDOWN:
            return 0 if manifest["prediction_rows"] else 2
        time.sleep(max(1, interval))


if __name__ == "__main__":
    raise SystemExit(main())
