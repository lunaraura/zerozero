#!/usr/bin/env python3
"""Data freshness and label-maturity report for downside-risk future shadow.

This report checks whether the recorded source table has advanced far enough to
label already-logged true-forward decisions. It is report-only and never trains,
recalibrates, changes thresholds, mutates champions, promotes, or places orders.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.freeze_rawseq_low_path_ridge_research_candidate import stable_hash
from scripts.tiny.run_rawseq_downside_risk_future_paper_shadow import add_derived_low_path_labels

DEFAULT_SHADOW_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_future_shadow"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_future_shadow_freshness"


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return resolve_path(raw) if raw else default


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 2:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def latest_cumulative_dir(root: Path) -> Path:
    explicit = os.getenv("RAWSEQ_DOWNSIDE_SHADOW_CUMULATIVE_DIR", "").strip()
    if explicit:
        return resolve_path(explicit)
    parent = root / "rawseq_downside_risk_future_shadow_cumulative"
    candidates = [p for p in parent.glob("*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No cumulative shadow directories found under {parent}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def is_true_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"true", "1", "yes"}:
            return True
        if stripped in {"false", "0", "no"}:
            return False
        return default
    return bool(value)


def infer_target_columns(decision_rows: list[dict[str, str]]) -> list[str]:
    columns: set[str] = set()
    for row in decision_rows:
        for key in row:
            if key.startswith("prob_"):
                columns.add(key[5:])
    return sorted(columns, key=lambda col: (horizon_seconds(col), col))


def horizon_seconds(target_column: str) -> int:
    match = re.search(r"_h(\d+)(?:$|_)", target_column)
    if not match:
        return 0
    return int(match.group(1))


def maturity_row(
    target_column: str,
    decision_rows: list[dict[str, str]],
    source_by_ts: dict[float, dict[str, Any]],
    source_max_timestamp: float,
) -> dict[str, Any]:
    horizon = horizon_seconds(target_column)
    horizon_ms = float(horizon * 1000)
    mature_cutoff = source_max_timestamp - horizon_ms
    true_rows = [row for row in decision_rows if is_true_value(row.get("true_forward_decision"), default=True)]
    mature_rows = []
    labeled_rows = []
    missing_rows = []
    for row in true_rows:
        ts = safe_float(row.get("decision_timestamp"))
        if not math.isfinite(ts):
            continue
        if ts <= mature_cutoff:
            mature_rows.append(row)
            source = source_by_ts.get(ts, {})
            label = safe_float(source.get(target_column))
            if math.isfinite(label):
                labeled_rows.append(row)
            else:
                missing_rows.append(row)
    latest_decision_ts = max([safe_float(row.get("decision_timestamp")) for row in true_rows], default=math.nan)
    next_label_source_timestamp = latest_decision_ts + horizon_ms if math.isfinite(latest_decision_ts) else math.nan
    return {
        "target_column": target_column,
        "horizon_seconds": horizon,
        "source_max_timestamp": source_max_timestamp,
        "label_mature_cutoff_timestamp": mature_cutoff,
        "true_forward_decisions": len(true_rows),
        "mature_true_forward_decisions": len(mature_rows),
        "labeled_true_forward_decisions": len(labeled_rows),
        "overdue_missing_label_rows": len(missing_rows),
        "label_fill_fraction_of_mature": len(labeled_rows) / max(len(mature_rows), 1),
        "latest_true_forward_decision_timestamp": latest_decision_ts,
        "source_timestamp_needed_to_label_latest_decision": next_label_source_timestamp,
        "source_timestamp_shortfall_for_latest_decision": max(0.0, next_label_source_timestamp - source_max_timestamp) if math.isfinite(next_label_source_timestamp) else math.nan,
        "freshness_status": "labels_overdue_missing" if missing_rows else ("labels_available" if mature_rows else "waiting_for_source_horizon"),
    }


def main() -> int:
    shadow_root = env_path("RAWSEQ_DOWNSIDE_SHADOW_ROOT", DEFAULT_SHADOW_ROOT)
    output_root = env_path("RAWSEQ_DOWNSIDE_SHADOW_FRESHNESS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    cumulative_dir = latest_cumulative_dir(shadow_root)
    contract = read_json(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_contract.json")
    feature_table = resolve_path(contract["feature_table"])
    decisions_path = cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_decisions.csv"
    decision_rows = read_csv_rows(decisions_path)
    target_columns = infer_target_columns(decision_rows)
    usecols = ["decision_timestamp", "price", "close", *[col for col in target_columns]]
    header = pd.read_csv(feature_table, nrows=0).columns.tolist()
    usecols = [col for col in usecols if col in header]
    source = pd.read_csv(feature_table, usecols=usecols)
    source = add_derived_low_path_labels(source, target_columns)
    source["decision_timestamp"] = pd.to_numeric(source["decision_timestamp"], errors="coerce")
    source_min = float(source["decision_timestamp"].min())
    source_max = float(source["decision_timestamp"].max())
    source_by_ts = {
        float(row["decision_timestamp"]): row.to_dict()
        for _, row in source.iterrows()
        if math.isfinite(float(row["decision_timestamp"]))
    }
    rows = [maturity_row(col, decision_rows, source_by_ts, source_max) for col in target_columns]
    consumed_cutoff = safe_float(contract.get("consumed_cutoff_timestamp"))
    source_rows_after_cutoff = int((source["decision_timestamp"] > consumed_cutoff).sum()) if math.isfinite(consumed_cutoff) else 0
    out_dir = output_root / f"rawseq_downside_risk_future_shadow_freshness_{now_stamp()}"
    summary = {
        "generated_at_iso": datetime.now(UTC).isoformat(),
        "cumulative_dir": str(cumulative_dir),
        "feature_table": str(feature_table),
        "source_min_timestamp": source_min,
        "source_max_timestamp": source_max,
        "consumed_cutoff_timestamp": consumed_cutoff,
        "source_rows": int(len(source)),
        "source_rows_after_consumed_cutoff": source_rows_after_cutoff,
        "cumulative_decision_rows": len(decision_rows),
        "target_columns": target_columns,
        "target_count": len(target_columns),
        "targets_waiting_for_source_horizon": sum(1 for row in rows if row["freshness_status"] == "waiting_for_source_horizon"),
        "targets_with_labels_overdue_missing": sum(1 for row in rows if row["freshness_status"] == "labels_overdue_missing"),
        "targets_with_labels_available": sum(1 for row in rows if row["freshness_status"] == "labels_available"),
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    summary["freshness_report_sha256"] = stable_hash(summary)
    write_json(out_dir / "rawseq_downside_risk_future_shadow_freshness.json", summary)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_label_maturity.csv", rows)
    lines = [
        "Rawseq downside-risk future shadow data freshness",
        f"Output: {out_dir}",
        f"Cumulative dir: {cumulative_dir}",
        f"Feature table: {feature_table}",
        f"Source max timestamp: {source_max}",
        f"Consumed cutoff timestamp: {consumed_cutoff}",
        f"Source rows after consumed cutoff: {source_rows_after_cutoff}",
        "",
        "Per-target maturity:",
    ]
    for row in rows:
        lines.append(
            "- {target_column}: status={freshness_status}, mature={mature_true_forward_decisions}, "
            "labeled={labeled_true_forward_decisions}, overdue_missing={overdue_missing_label_rows}, "
            "shortfall_ms={source_timestamp_shortfall_for_latest_decision}".format(**row)
        )
    lines.extend(
        [
            "",
            "Safety: report-only, paper_only=true, orders=false, promotion=false, champion_mutation=false.",
        ]
    )
    (out_dir / "rawseq_downside_risk_future_shadow_freshness.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
