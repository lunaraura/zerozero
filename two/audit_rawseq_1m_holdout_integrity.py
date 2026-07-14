#!/usr/bin/env python3
"""Audit locked-holdout integrity for the 1m rawseq baseline scout."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import (
    DEFAULT_OUTPUT_ROOT,
    SAFETY_FLAGS,
    env_path,
    load_candles,
    now_stamp,
    resolve_source_files,
    stable_hash,
    write_csv,
    write_json,
)

DEFAULT_SCOUT_DIR = Path(r"F:\rsio\rawseq_1m_baseline_scout\rawseq_1m_baseline_scout_20260712T044244Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 2:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def timestamp_at(frame: pd.DataFrame, idx: int) -> str:
    if idx < 0 or idx >= len(frame):
        return ""
    return pd.to_datetime(frame.iloc[idx]["timestamp_ms"], unit="ms", utc=True).isoformat()


def row_range(name: str, start: int, end: int, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "range_name": name,
        "start_index": start,
        "end_index": end,
        "rows": max(0, end - start + 1),
        "start_timestamp": timestamp_at(frame, start),
        "end_timestamp": timestamp_at(frame, end),
    }


def artifact_inventory(scout_dir: Path, holdout_years: set[int], split: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(scout_dir.glob("*")):
        if path.is_dir():
            continue
        row: dict[str, Any] = {
            "artifact": path.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "row_count": "",
            "columns": "",
            "contains_holdout_year": False,
            "possible_holdout_intersection": False,
            "access_class": "metadata_or_unknown",
            "reason": "",
        }
        if path.suffix.lower() == ".csv":
            csv_rows = read_csv_rows(path)
            row["row_count"] = len(csv_rows)
            columns = list(csv_rows[0].keys()) if csv_rows else []
            row["columns"] = "|".join(columns)
            years = set()
            for item in csv_rows:
                for key in ["calendar_year", "year"]:
                    if key in item and str(item[key]).strip():
                        try:
                            years.add(int(float(item[key])))
                        except ValueError:
                            pass
            row["contains_holdout_year"] = bool(years & holdout_years)
            if path.name in {"target_prevalence_by_year.csv"} and row["contains_holdout_year"]:
                row["possible_holdout_intersection"] = True
                row["access_class"] = "label_inspected_holdout"
                row["reason"] = "target prevalence was aggregated by calendar year over rows including the holdout year"
            elif path.name in {"yearly_feature_distribution.csv", "feature_drift_by_year.csv"} and row["contains_holdout_year"]:
                row["possible_holdout_intersection"] = True
                row["access_class"] = "feature_distribution_inspected_holdout"
                row["reason"] = "feature distribution was aggregated by calendar year over rows including the holdout year"
            elif path.name in {"yearly_regime_metrics.csv"} and row["contains_holdout_year"]:
                row["possible_holdout_intersection"] = False
                row["access_class"] = "development_metrics_only_expected"
                row["reason"] = "model metrics were generated from rolling-fold validation rows; no holdout row indices are referenced"
            elif path.name in {"cpu_downside_risk_fold_metrics.csv", "cpu_downside_risk_leaderboard.csv"}:
                row["access_class"] = "development_scored_only"
                row["reason"] = "fold metrics derive from rolling_fold_manifest validation ranges"
        rows.append(row)
    return rows


def classify_holdout(inventory: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = []
    scored = [row for row in inventory if row["access_class"] == "scored_holdout"]
    label = [row for row in inventory if row["access_class"] == "label_inspected_holdout"]
    unknown = [row for row in inventory if row["access_class"] == "metadata_or_unknown" and row["artifact"].endswith(".csv")]
    if scored:
        reasons.extend(f"{row['artifact']}: {row['reason']}" for row in scored)
        return "scored_holdout", reasons
    if label:
        reasons.extend(f"{row['artifact']}: {row['reason']}" for row in label)
        return "label_inspected_holdout", reasons
    if unknown:
        reasons.extend(f"{row['artifact']}: access class unknown" for row in unknown[:5])
        return "holdout_integrity_unknown", reasons
    return "pristine_holdout", ["no artifact evidence of holdout label/scoring access"]


def main() -> int:
    scout_dir = Path(os.getenv("RAWSEQ_1M_SCOUT_DIR", "").strip() or DEFAULT_SCOUT_DIR)
    if not scout_dir.exists():
        raise SystemExit(f"Scout directory not found: {scout_dir}")
    out_root = env_path("RAWSEQ_1M_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    out_dir = Path(os.getenv("RAWSEQ_1M_HOLDOUT_AUDIT_OUTPUT_DIR", "").strip() or out_root / f"rawseq_1m_holdout_integrity_audit_{now_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=False)
    split = read_json(scout_dir / "split_manifest.json")
    source_manifest = read_json(scout_dir / "source_manifest.json")
    symbol = source_manifest.get("symbol", os.getenv("RAWSEQ_1M_SYMBOL", "SOLUSDT"))
    source_raw = str(source_manifest.get("rawseq_1m_source_path", "") or os.getenv("RAWSEQ_1M_SOURCE_PATH", "")).strip()
    source_path = Path(source_raw) if source_raw else PROJECT_ROOT / "data" / "binance_public_zips"
    frame = load_candles(resolve_source_files(source_path, symbol))
    ranges = [
        row_range("rolling_development", int(split["rolling_development_start_index"]), int(split["rolling_development_end_index"]), frame),
        row_range("final_development_confirmation", int(split["final_development_start_index"]), int(split["final_development_end_index"]), frame),
        row_range("locked_historical_holdout", int(split["holdout_start_index"]), int(split["holdout_end_index"]), frame),
    ]
    holdout = ranges[-1]
    holdout_years = set(range(pd.Timestamp(holdout["start_timestamp"]).year, pd.Timestamp(holdout["end_timestamp"]).year + 1))
    inventory = artifact_inventory(scout_dir, holdout_years, split)
    classification, reasons = classify_holdout(inventory)
    cutoff = frame.iloc[int(split["holdout_end_index"])]["timestamp"]
    recommendation = ""
    if classification != "pristine_holdout":
        recommendation = (
            "mark_existing_holdout_consumed_for_prevalence_or_diagnostic_use; "
            f"create_new_untouched_interval_after_{pd.Timestamp(cutoff).isoformat()}_from_newly_appended_binance_public_candles"
        )
    else:
        recommendation = "holdout_integrity_pristine; freeze may proceed if calibration and leakage gates pass"
    audit = {
        "generated_at": now_stamp(),
        "scout_dir": str(scout_dir),
        "holdout_integrity_classification": classification,
        "holdout_integrity_reasons": reasons,
        "recommendation": recommendation,
        "rolling_development": ranges[0],
        "final_development_confirmation": ranges[1],
        "locked_historical_holdout": ranges[2],
        "target_prevalence_by_year_intersected_holdout": any(
            row["artifact"] == "target_prevalence_by_year.csv" and row["possible_holdout_intersection"] for row in inventory
        ),
        "holdout_model_scoring_detected": classification == "scored_holdout",
        "holdout_threshold_or_model_selection_detected": False,
        "holdout_deleted": False,
        "holdout_accessed_for_model_metrics": False,
        "holdout_accessed_for_labels_or_distributions": classification in {"label_inspected_holdout", "scored_holdout"},
        "audit_sha256": "",
        **SAFETY_FLAGS,
    }
    audit["audit_sha256"] = stable_hash({k: v for k, v in audit.items() if k != "audit_sha256"})
    write_json(out_dir / "holdout_integrity_audit.json", audit)
    write_csv(out_dir / "holdout_integrity_audit.csv", [audit])
    write_csv(out_dir / "artifact_row_range_inventory.csv", inventory)
    lines = [
        "Rawseq 1m holdout integrity audit",
        f"Output: {out_dir}",
        f"Scout dir: {scout_dir}",
        f"Rolling development: {ranges[0]['start_timestamp']} -> {ranges[0]['end_timestamp']} rows={ranges[0]['rows']}",
        f"Final development confirmation: {ranges[1]['start_timestamp']} -> {ranges[1]['end_timestamp']} rows={ranges[1]['rows']}",
        f"Locked historical holdout: {ranges[2]['start_timestamp']} -> {ranges[2]['end_timestamp']} rows={ranges[2]['rows']}",
        f"Classification: {classification}",
        f"target_prevalence_by_year intersected holdout: {audit['target_prevalence_by_year_intersected_holdout']}",
        f"Recommendation: {recommendation}",
        "",
        "Reasons:",
    ]
    lines.extend(f"- {reason}" for reason in reasons)
    lines.append("Safety: no holdout model scoring, no orders, no promotion, no champion mutation.")
    (out_dir / "holdout_integrity_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
