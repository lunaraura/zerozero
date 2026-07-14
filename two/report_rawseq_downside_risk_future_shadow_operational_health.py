#!/usr/bin/env python3
"""Operational health report for the downside-risk future shadow.

This report checks recorder/source freshness, duplicate/gap symptoms, ledger
hashes, parity/status availability, disk space, and cadence state. It is
report-only and paper-only: no training, orders, promotion, champion mutation,
or threshold changes.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny import report_rawseq_downside_risk_future_shadow_status as status_report
from scripts.tiny import report_rawseq_downside_risk_future_shadow_data_freshness as data_freshness
from scripts.tiny import report_rawseq_downside_risk_shadow_contract_parity as parity_report
from scripts.tiny import report_rawseq_downside_risk_shadow_source_freshness as source_freshness
from scripts.tiny import run_rawseq_downside_risk_future_paper_shadow as shadow_logger
from scripts.tiny.freeze_rawseq_low_path_ridge_research_candidate import stable_hash
from scripts.tiny.rawseq_future_shadow_lock import (
    attach_implementation_lock,
    file_sha256,
    latest_cumulative_dir,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_future_shadow_operational_health"
DEFAULT_CYCLE_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_future_shadow_cycles"
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "realtime" / "kraken" / "SOLUSDT_10s_flow.csv"
DEFAULT_TAIL_ROWS = 5000
DEFAULT_STALE_SOURCE_SECONDS = 15 * 60
DEFAULT_LOW_DISK_FREE_GB = 5.0
DEFAULT_CADENCE_MIN_MINUTES = 60.0


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


def safe_int(value: Any, default: int = 0) -> int:
    out = safe_float(value)
    return int(out) if math.isfinite(out) else default


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def latest_dir(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    candidates = [p for p in root.glob(pattern) if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def latest_json(root: Path, pattern: str, filename: str) -> tuple[Path | None, dict[str, Any]]:
    run_dir = latest_dir(root, pattern)
    if not run_dir:
        return None, {}
    return run_dir, read_json(run_dir / filename)


def tail_csv_rows(path: Path, max_rows: int) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
    columns = next(csv.reader([header_line])) if header_line else []
    block_size = 1024 * 1024
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        data = b""
        pos = size
        while pos > 0 and data.count(b"\n") <= max_rows + 2:
            read_size = min(block_size, pos)
            pos -= read_size
            handle.seek(pos)
            data = handle.read(read_size) + data
    text = data.decode("utf-8", errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and lines[0].lstrip("\ufeff") == header_line.strip():
        lines = lines[1:]
    lines = lines[-max_rows:]
    reader = csv.DictReader([",".join(columns), *lines])
    return columns, [dict(row) for row in reader]


def timestamp_to_datetime(value: float) -> datetime | None:
    if not math.isfinite(value):
        return None
    if value > 1e12:
        return datetime.fromtimestamp(value / 1000.0, UTC)
    if value > 1e9:
        return datetime.fromtimestamp(value, UTC)
    return None


def source_tail_health(source_path: Path, tail_rows: int, stale_seconds: float) -> dict[str, Any]:
    columns, rows = tail_csv_rows(source_path, tail_rows)
    timestamp_col = next((col for col in ["decision_timestamp", "timestamp_ms", "timestamp", "time"] if col in columns), "")
    timestamps = [safe_float(row.get(timestamp_col)) for row in rows] if timestamp_col else []
    finite_ts = [ts for ts in timestamps if math.isfinite(ts)]
    latest_ts = max(finite_ts) if finite_ts else math.nan
    latest_dt = timestamp_to_datetime(latest_ts)
    now = datetime.now(UTC)
    sorted_ts = sorted(finite_ts)
    diffs_seconds = []
    for prev, cur in zip(sorted_ts, sorted_ts[1:]):
        scale = 1000.0 if max(abs(prev), abs(cur), 0.0) > 1e12 else 1.0
        diffs_seconds.append((cur - prev) / scale)
    positive_diffs = [d for d in diffs_seconds if d > 0]
    median_gap = sorted(positive_diffs)[len(positive_diffs) // 2] if positive_diffs else math.nan
    max_gap = max(positive_diffs, default=math.nan)
    expected_gap = median_gap if math.isfinite(median_gap) and median_gap > 0 else 10.0
    duplicate_count = len(sorted_ts) - len(set(sorted_ts))
    gap_count = sum(1 for diff in positive_diffs if diff > max(expected_gap * 3.0, expected_gap + 5.0))
    latest_lag = (now - latest_dt).total_seconds() if latest_dt else math.nan
    last_write_dt = datetime.fromtimestamp(source_path.stat().st_mtime, UTC) if source_path.exists() else None
    last_write_age = (now - last_write_dt).total_seconds() if last_write_dt else math.nan
    return {
        "source_path": str(source_path),
        "source_exists": source_path.exists(),
        "source_bytes": source_path.stat().st_size if source_path.exists() else 0,
        "source_last_write_iso": last_write_dt.isoformat() if last_write_dt else "",
        "source_last_write_age_seconds": last_write_age,
        "source_timestamp_column": timestamp_col,
        "source_tail_rows_inspected": len(rows),
        "source_latest_timestamp": latest_ts,
        "source_latest_timestamp_iso": latest_dt.isoformat() if latest_dt else "",
        "source_latest_lag_seconds": latest_lag,
        "source_duplicate_timestamp_count_tail": duplicate_count,
        "source_gap_count_tail": gap_count,
        "source_median_gap_seconds_tail": median_gap,
        "source_max_gap_seconds_tail": max_gap,
        "recorder_heartbeat_status": "fresh" if math.isfinite(latest_lag) and latest_lag <= stale_seconds else "stale_or_unknown",
    }


def ledger_hashes(cumulative_dir: Path | None) -> dict[str, Any]:
    if not cumulative_dir:
        return {}
    files = [
        "rawseq_downside_risk_future_shadow_cumulative_contract.json",
        "rawseq_downside_risk_future_shadow_cumulative_decisions.csv",
        "rawseq_downside_risk_future_shadow_true_forward_decisions.csv",
        "rawseq_downside_risk_future_shadow_cumulative_labeled_results.csv",
    ]
    hashes = {
        name: file_sha256(cumulative_dir / name) if (cumulative_dir / name).exists() else ""
        for name in files
    }
    hashes["combined_cumulative_ledger_hash"] = stable_hash(hashes)
    return hashes


def cadence_payload(latest_status_dir: Path | None, min_minutes: float) -> dict[str, Any]:
    if not latest_status_dir:
        return {
            "last_successful_status_dir": "",
            "last_successful_status_age_minutes": math.nan,
            "next_eligible_cycle_time_iso": "",
            "cadence_status": "due_no_prior_status",
        }
    now = datetime.now(UTC)
    mtime = datetime.fromtimestamp(latest_status_dir.stat().st_mtime, UTC)
    age = max(0.0, (now - mtime).total_seconds() / 60.0)
    next_time = mtime + timedelta(minutes=min_minutes)
    return {
        "last_successful_status_dir": str(latest_status_dir),
        "last_successful_status_age_minutes": age,
        "next_eligible_cycle_time_iso": next_time.isoformat(),
        "cadence_status": "due" if age >= min_minutes else "not_due",
    }


def health_status(summary: dict[str, Any], stale_seconds: float, low_disk_free_gb: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not summary.get("source_exists"):
        reasons.append("source file missing")
    if str(summary.get("recorder_heartbeat_status")) != "fresh":
        reasons.append(f"source heartbeat stale/unknown; lag_seconds={summary.get('source_latest_lag_seconds')}")
    if safe_int(summary.get("source_duplicate_timestamp_count_tail")) > 0:
        reasons.append("duplicate timestamps detected in source tail")
    if safe_int(summary.get("source_gap_count_tail")) > 0:
        reasons.append("timestamp gaps detected in source tail")
    if safe_float(summary.get("disk_free_gb"), 0.0) < low_disk_free_gb:
        reasons.append("disk free space below configured threshold")
    if str(summary.get("contract_parity_status", "")).upper() not in {"OK", "PASS"}:
        reasons.append("latest contract parity status is not OK/PASS")
    status = "OK" if not reasons else ("FAIL" if any("missing" in r or "disk" in r for r in reasons) else "WARN")
    return status, reasons


def main() -> int:
    output_root = env_path("RAWSEQ_DOWNSIDE_OPERATIONAL_HEALTH_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    source_path = env_path("RAWSEQ_DOWNSIDE_OPERATIONAL_HEALTH_SOURCE_PATH", DEFAULT_SOURCE_PATH)
    tail_rows = safe_int(os.getenv("RAWSEQ_DOWNSIDE_OPERATIONAL_HEALTH_TAIL_ROWS", ""), DEFAULT_TAIL_ROWS)
    stale_seconds = safe_float(os.getenv("RAWSEQ_DOWNSIDE_OPERATIONAL_HEALTH_STALE_SECONDS", ""), DEFAULT_STALE_SOURCE_SECONDS)
    low_disk_free_gb = safe_float(os.getenv("RAWSEQ_DOWNSIDE_OPERATIONAL_HEALTH_LOW_DISK_GB", ""), DEFAULT_LOW_DISK_FREE_GB)
    cadence_minutes = safe_float(os.getenv("RAWSEQ_DOWNSIDE_DUE_MIN_MINUTES", ""), DEFAULT_CADENCE_MIN_MINUTES)
    shadow_root = env_path("RAWSEQ_DOWNSIDE_SHADOW_ROOT", shadow_logger.DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / f"rawseq_downside_risk_future_shadow_operational_health_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    cumulative_dir = latest_cumulative_dir(shadow_root)
    contract = read_json(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_contract.json") if cumulative_dir else {}
    status_root = status_report.env_path("RAWSEQ_DOWNSIDE_SHADOW_STATUS_OUTPUT_ROOT", status_report.DEFAULT_OUTPUT_ROOT)
    latest_status_run, latest_status = latest_json(status_root, "rawseq_downside_risk_future_shadow_status_*", "rawseq_downside_risk_future_shadow_status.json")
    latest_parity_run, latest_parity = latest_json(
        parity_report.env_path("RAWSEQ_DOWNSIDE_PARITY_OUTPUT_ROOT", parity_report.DEFAULT_OUTPUT_ROOT),
        "rawseq_downside_risk_shadow_contract_parity_*",
        "rawseq_downside_risk_shadow_contract_parity.json",
    )
    latest_data_freshness_run, latest_data = latest_json(
        data_freshness.env_path("RAWSEQ_DOWNSIDE_SHADOW_FRESHNESS_OUTPUT_ROOT", data_freshness.DEFAULT_OUTPUT_ROOT),
        "rawseq_downside_risk_future_shadow_freshness_*",
        "rawseq_downside_risk_future_shadow_freshness.json",
    )
    latest_source_freshness_run, latest_source = latest_json(
        source_freshness.env_path("RAWSEQ_DOWNSIDE_SOURCE_FRESHNESS_OUTPUT_ROOT", source_freshness.DEFAULT_OUTPUT_ROOT),
        "rawseq_downside_risk_shadow_source_freshness_*",
        "rawseq_downside_risk_shadow_source_freshness.json",
    )
    latest_cycle_run = latest_dir(
        env_path("RAWSEQ_DOWNSIDE_SHADOW_CYCLE_OUTPUT_ROOT", DEFAULT_CYCLE_OUTPUT_ROOT),
        "rawseq_downside_risk_future_shadow_cycle_*",
    )
    disk = shutil.disk_usage(PROJECT_ROOT)
    source_health = source_tail_health(source_path, tail_rows, stale_seconds)
    summary = {
        "generated_at_iso": datetime.now(UTC).isoformat(),
        "operational_health_dir": str(out_dir),
        "cumulative_dir": str(cumulative_dir) if cumulative_dir else "",
        **source_health,
        "true_forward_decisions_total": safe_int(contract.get("cumulative_true_forward_decision_rows"), 0),
        "cumulative_labeled_rows": safe_int(contract.get("cumulative_labeled_rows"), 0),
        "non_overlapping_h480_rows": safe_int(contract.get("h480_non_overlapping_rows"), 0),
        "backfill_or_replay_exclusions": safe_int(contract.get("cumulative_backfill_or_replay_decision_rows"), 0),
        "latest_status_run": str(latest_status_run) if latest_status_run else "",
        "latest_status_final_status": latest_status.get("final_status", ""),
        "latest_parity_run": str(latest_parity_run) if latest_parity_run else "",
        "contract_parity_status": latest_parity.get("status", latest_parity.get("parity_status", "")),
        "latest_data_freshness_run": str(latest_data_freshness_run) if latest_data_freshness_run else "",
        "newly_matured_h480_labels": latest_data.get("newly_matured_h480_labels", latest_data.get("new_labeled_h480_rows", "")),
        "latest_source_freshness_run": str(latest_source_freshness_run) if latest_source_freshness_run else "",
        "source_freshness_status": latest_source.get("status", ""),
        "latest_successful_cycle_run": str(latest_cycle_run) if latest_cycle_run else "",
        "disk_total_gb": disk.total / (1024**3),
        "disk_free_gb": disk.free / (1024**3),
        "disk_low_threshold_gb": low_disk_free_gb,
        **ledger_hashes(cumulative_dir),
        **cadence_payload(latest_status_run, cadence_minutes),
        "paper_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "training": False,
        "recalibration": False,
        "threshold_changes": False,
    }
    status, reasons = health_status(summary, stale_seconds, low_disk_free_gb)
    summary["health_status"] = status
    summary["health_reasons"] = reasons
    attach_implementation_lock(
        summary,
        candidate_dir=contract.get("candidate_dir", shadow_logger.DEFAULT_CANDIDATE_DIR),
        cumulative_dir=cumulative_dir,
        feature_table=contract.get("feature_table", ""),
    )
    summary["operational_health_report_sha256"] = stable_hash(summary)

    write_json(out_dir / "rawseq_downside_risk_future_shadow_operational_health.json", summary)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_operational_health.csv", [summary])
    lines = [
        "Rawseq downside-risk future shadow operational health",
        f"Output: {out_dir}",
        f"Health status: {status}",
        f"Source latest timestamp: {summary['source_latest_timestamp_iso']}",
        f"Source lag seconds: {summary['source_latest_lag_seconds']}",
        f"Source gaps/duplicates tail: {summary['source_gap_count_tail']} / {summary['source_duplicate_timestamp_count_tail']}",
        f"True-forward decisions: {summary['true_forward_decisions_total']}",
        f"Labeled rows: {summary['cumulative_labeled_rows']}",
        f"Non-overlapping h480 rows: {summary['non_overlapping_h480_rows']}",
        f"Backfill/replay exclusions: {summary['backfill_or_replay_exclusions']}",
        f"Contract parity status: {summary['contract_parity_status']}",
        f"Disk free GB: {summary['disk_free_gb']:.3f}",
        f"Cadence status: {summary['cadence_status']}",
        f"Next eligible cycle: {summary['next_eligible_cycle_time_iso']}",
        f"Logger lock: {summary['logger_implementation_lock_sha256']}",
        "",
        "Reasons:",
    ]
    lines.extend(f"- {reason}" for reason in reasons)
    lines.append("Safety: paper_only=true, orders=false, promotion=false, champion_mutation=false.")
    (out_dir / "rawseq_downside_risk_future_shadow_operational_health.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
