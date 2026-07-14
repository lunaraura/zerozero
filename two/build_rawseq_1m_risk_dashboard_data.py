#!/usr/bin/env python3
"""Build a compact data bundle for the rawseq 1m downside-risk dashboard.

The bundle is visualization-only. It reads the already-created June holdout
packet and prepared public candle CSVs, then writes a static JavaScript payload
for the GUI. It does not train, recalibrate, rescore, mutate the candidate, or
touch July data.
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

from scripts.tiny.rawseq_1m_baseline_utils import file_sha256, now_stamp, write_json

DEFAULT_PACKET_DIR = Path(r"F:\rsio\rawseq_1m_cross_asset_scout\rawseq_1m_june_holdout_evaluation_20260712T180529Z")
DEFAULT_CANDLE_DIR = PROJECT_ROOT / "data" / "realtime" / "binance_1m_candles_multi"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "rawseq_1m_risk_dashboard_data.js"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def rows_for_symbol(
    symbol: str,
    candles: pd.DataFrame,
    predictions: pd.DataFrame,
    row_limit: int,
    holdout_event_prevalence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    symbol_predictions = predictions[predictions["symbol"] == symbol].copy()
    frame = candles.merge(symbol_predictions, on="timestamp_ms", how="inner", suffixes=("", "_prediction"))
    expected_prediction_rows = int(len(symbol_predictions))
    matched_rows = int(len(frame))
    join_audit = {
        "symbol": symbol,
        "candle_rows": int(len(candles)),
        "expected_prediction_rows": expected_prediction_rows,
        "matched_rows": matched_rows,
        "unmatched_prediction_rows": int(max(0, expected_prediction_rows - matched_rows)),
        "unmatched_candle_rows": int(max(0, len(candles) - matched_rows)),
        "join_coverage_fraction": float(matched_rows / expected_prediction_rows) if expected_prediction_rows else 0.0,
        "join_status": "PASS" if expected_prediction_rows and matched_rows == expected_prediction_rows else "FAIL",
    }
    if join_audit["join_status"] != "PASS":
        raise RuntimeError(f"Dashboard timestamp join failed for {symbol}: {join_audit}")
    frame = frame.tail(row_limit).copy()
    out = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        vol_bps = safe_float(row.get("trailing_volatility_bps_fw240"))
        event_threshold_bps = 0.5 * vol_bps if vol_bps is not None else None
        pred = safe_float(row.get("prediction"))
        out.append(
            {
                "timestamp": pd.to_datetime(row["timestamp_ms"], unit="ms", utc=True).isoformat(),
                "open": safe_float(row.get("open")),
                "high": safe_float(row.get("high")),
                "low": safe_float(row.get("low")),
                "close": safe_float(row.get("close")),
                "volume": safe_float(row.get("volume")),
                "ensemble": pred,
                "pooled_logistic": safe_float(row.get("component_pooled_logistic")),
                "pooled_regime_logistic": safe_float(row.get("component_pooled_regime_feature_logistic")),
                "pooled_shallow_hgb": safe_float(row.get("component_pooled_shallow_hgb")),
                "actual_event": safe_float(row.get("actual")),
                "trailing_volatility_bps": vol_bps,
                "event_threshold_bps": event_threshold_bps,
                "above_holdout_event_prevalence": bool(pred is not None and pred >= holdout_event_prevalence),
                "high_risk": bool(pred is not None and pred >= holdout_event_prevalence),
            }
        )
    return out, join_audit


def normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    parsed = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if parsed.isna().any():
        raise RuntimeError("Prediction timestamp normalization failed")
    epoch = pd.Timestamp("1970-01-01T00:00:00Z")
    out["timestamp_ms"] = ((parsed - epoch) / pd.Timedelta(milliseconds=1)).round().astype("int64")
    return out


def normalize_candles(candles: pd.DataFrame) -> pd.DataFrame:
    out = candles.copy()
    out["timestamp_ms"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out["timestamp_dt"] = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True, errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = 10000.0 * np.log(close / close.shift(1))
    out["trailing_volatility_bps_fw240"] = returns.rolling(240, min_periods=240).std(ddof=0)
    if out["timestamp_dt"].isna().any():
        raise RuntimeError("Candle timestamp normalization failed")
    return out


def main() -> int:
    packet_dir = Path(os.getenv("RAWSEQ_RISK_DASHBOARD_PACKET_DIR", str(DEFAULT_PACKET_DIR)))
    candle_dir = Path(os.getenv("RAWSEQ_RISK_DASHBOARD_CANDLE_DIR", str(DEFAULT_CANDLE_DIR)))
    output_path = Path(os.getenv("RAWSEQ_RISK_DASHBOARD_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))
    row_limit = int(os.getenv("RAWSEQ_RISK_DASHBOARD_ROWS_PER_SYMBOL", "720") or "720")
    manifest = read_json(packet_dir / "june_holdout_evaluation_manifest.json")
    archive = read_json(packet_dir / "holdout_1_immutable_archive_manifest.json")
    evaluator = read_json(packet_dir / "june_holdout_evaluator_hashes.json")
    predictions = normalize_predictions(
        pd.read_csv(
            packet_dir / "june_holdout_predictions.csv",
            usecols=[
                "symbol",
                "timestamp",
                "actual",
                "prediction",
                "component_pooled_logistic",
                "component_pooled_regime_feature_logistic",
                "component_pooled_shallow_hgb",
            ],
        )
    )
    symbols = sorted(predictions["symbol"].dropna().unique().tolist())
    holdout_event_prevalence = float(manifest["row_weighted_aggregate"]["event_prevalence"])
    symbol_payload: dict[str, Any] = {}
    source_health = []
    join_audits = []
    for symbol in symbols:
        path = candle_dir / f"{symbol}_1m_flow.csv"
        candles = normalize_candles(pd.read_csv(path))
        rows, join_audit = rows_for_symbol(symbol, candles, predictions, row_limit, holdout_event_prevalence)
        symbol_payload[symbol] = rows
        join_audits.append(join_audit)
        source_health.append(
            {
                "symbol": symbol,
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": int(len(candles)),
                "latest_timestamp": candles["timestamp_dt"].iloc[-1].isoformat(),
                "data_through": candles["timestamp_dt"].iloc[-1].isoformat(),
                "source_lag_note": "historical replay packet; not live inference",
            }
        )
    payload = {
        "created_at": now_stamp(),
        "mode": "historical_replay",
        "dashboard_role": "June 2026 frozen-holdout replay and model-inspection dashboard",
        "model_output": "P(next-minute downside excursion > 0.5 * trailing 240-minute volatility)",
        "no_predicted_future_price_line": True,
        "holdout_event_prevalence": holdout_event_prevalence,
        "baseline_probability": holdout_event_prevalence,
        "risk_shading_threshold_probability": holdout_event_prevalence,
        "probability_reference_label": "Holdout event prevalence, not a frozen operational alert threshold",
        "live_inference_ready": False,
        "symbols": symbols,
        "default_symbol": "SOLUSDT" if "SOLUSDT" in symbols else symbols[0],
        "candidate_hash": manifest["candidate_hash"],
        "acceptance_rule_hash": manifest["acceptance_rule_hash"],
        "evaluation_packet_hash": manifest["june_evaluation_packet_hash"],
        "evaluator_script_sha256": evaluator["evaluator_script_sha256"],
        "component_model_hashes": manifest["component_model_hashes"],
        "component_model_pickle_sha256": manifest["component_model_pickle_sha256"],
        "holdout_1_status": archive["holdout_1_status"],
        "final_status": manifest["final_status"],
        "ensemble_weights": manifest["ensemble_weights"],
        "weighting": manifest["weighting"],
        "calibration": manifest["calibration"],
        "target": {
            "horizon_minutes": 1,
            "event": "next-minute downside excursion",
            "threshold": "0.5 * trailing_volatility_bps_fw240",
            "volatility_window_minutes": 240,
        },
        "health": {
            "candidate_hash_match": manifest["candidate_hash"] == archive["candidate_hash"],
            "acceptance_hash_match": manifest["acceptance_rule_hash"] == archive["acceptance_rule_hash"],
            "july_files_opened": bool(manifest.get("july_files_opened", False)),
            "july_timestamps_enumerated": bool(manifest.get("july_timestamps_enumerated", False)),
            "july_labels_computed": bool(manifest.get("july_labels_computed", False)),
            "july_predictions_computed": bool(manifest.get("july_predictions_computed", False)),
            "july_metrics_computed": bool(manifest.get("july_metrics_computed", False)),
            "timestamp_join_all_pass": all(row["join_status"] == "PASS" for row in join_audits),
            "component_probabilities_present": all(
                key in predictions.columns
                for key in ["component_pooled_logistic", "component_pooled_regime_feature_logistic", "component_pooled_shallow_hgb"]
            ),
        },
        "join_audit": join_audits,
        "source_health": source_health,
        "series": symbol_payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("window.RAWSEQ_RISK_DASHBOARD_DATA = " + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";\n", encoding="utf-8")
    write_json(output_path.with_suffix(".manifest.json"), {"output_path": str(output_path), "symbols": symbols, "rows_per_symbol": row_limit, "dashboard_data_sha256": file_sha256(output_path)})
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
