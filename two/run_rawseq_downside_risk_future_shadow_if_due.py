#!/usr/bin/env python3
"""Run the frozen downside-risk future shadow only when cadence is due.

This is the safer scheduler/automation entry point. It checks the latest status
report age before calling the audited advance wrapper, so repeated invocations
do not churn feature refreshes every few minutes.

No training, no recalibration, no threshold changes, no orders, no private API,
no champion mutation, no promotion.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny import report_rawseq_downside_risk_future_shadow_status as shadow_status
from scripts.tiny import run_rawseq_downside_risk_future_shadow_advance as advance
from scripts.tiny.rawseq_future_shadow_lock import attach_implementation_lock
from scripts.tiny.freeze_rawseq_low_path_ridge_research_candidate import stable_hash

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_future_shadow_due_checks"
DEFAULT_MIN_MINUTES = 60.0


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return resolve_path(raw) if raw else default


def latest_status_dir(status_root: Path) -> Path | None:
    if not status_root.exists():
        return None
    candidates = [p for p in status_root.glob("rawseq_downside_risk_future_shadow_status_*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def status_age_minutes(status_dir: Path | None, now: datetime | None = None) -> float | None:
    if status_dir is None:
        return None
    now = now or datetime.now(UTC)
    mtime = datetime.fromtimestamp(status_dir.stat().st_mtime, UTC)
    return max(0.0, (now - mtime).total_seconds() / 60.0)


def should_run_advance(age_minutes: float | None, min_minutes: float, force: bool = False) -> tuple[bool, str]:
    if force:
        return True, "force_enabled"
    if age_minutes is None:
        return True, "no_prior_status"
    if age_minutes >= min_minutes:
        return True, "cadence_due"
    return False, "cadence_not_due"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    output_root = env_path("RAWSEQ_DOWNSIDE_DUE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    status_root = env_path("RAWSEQ_DOWNSIDE_SHADOW_STATUS_OUTPUT_ROOT", shadow_status.DEFAULT_OUTPUT_ROOT)
    min_minutes = safe_float(os.getenv("RAWSEQ_DOWNSIDE_DUE_MIN_MINUTES", ""), DEFAULT_MIN_MINUTES)
    force = parse_bool(os.getenv("RAWSEQ_DOWNSIDE_DUE_FORCE", "false"))
    due_dir = output_root / f"rawseq_downside_risk_future_shadow_due_check_{now_stamp()}"
    due_dir.mkdir(parents=True, exist_ok=False)

    started = datetime.now(UTC)
    latest_status = latest_status_dir(status_root)
    age = status_age_minutes(latest_status, started)
    run_advance, reason = should_run_advance(age, min_minutes, force)
    advance_exit_code: int | None = None

    status = "ADVANCE_DUE" if run_advance else "SKIPPED_NOT_DUE"
    if run_advance:
        advance_exit_code = int(advance.main())
        status = "ADVANCE_OK" if advance_exit_code == 0 else "ADVANCE_FAILED"

    finished = datetime.now(UTC)
    manifest = {
        "due_check_dir": str(due_dir),
        "started_at_iso": started.isoformat(),
        "finished_at_iso": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "status": status,
        "reason": reason,
        "min_minutes": min_minutes,
        "force": force,
        "latest_status_dir": str(latest_status) if latest_status else "",
        "latest_status_age_minutes": age,
        "advance_exit_code": advance_exit_code,
        "paper_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "training": False,
        "recalibration": False,
        "threshold_changes": False,
    }
    attach_implementation_lock(manifest)
    manifest["due_check_manifest_sha256"] = stable_hash(manifest)
    write_json(due_dir / "rawseq_downside_risk_future_shadow_due_check_manifest.json", manifest)
    print(f"Due check output: {due_dir}")
    print(f"Status: {status}")
    print(f"Reason: {reason}")
    print(f"Latest status: {manifest['latest_status_dir']}")
    print(f"Latest status age minutes: {age}")
    print(f"Min minutes: {min_minutes}")
    return 0 if status in {"SKIPPED_NOT_DUE", "ADVANCE_OK"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
