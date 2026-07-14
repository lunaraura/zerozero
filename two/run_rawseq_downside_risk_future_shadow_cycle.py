#!/usr/bin/env python3
"""Run one safe frozen downside-risk future-shadow cycle.

The cycle is intentionally small:
1. Run the frozen paper-shadow logger.
2. Run the conservative status reporter.
3. Write a cycle manifest tying both outputs together.

No training, no recalibration, no threshold changes, no orders, no private API,
no champion mutation, no promotion.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny import report_rawseq_downside_risk_future_shadow_status as status_report
from scripts.tiny import report_rawseq_downside_risk_future_shadow_data_freshness as freshness_report
from scripts.tiny import report_rawseq_downside_risk_future_shadow_operational_health as health_report
from scripts.tiny import report_rawseq_downside_risk_shadow_contract_parity as parity_report
from scripts.tiny import report_rawseq_downside_risk_shadow_source_freshness as source_freshness_report
from scripts.tiny import run_rawseq_downside_risk_future_paper_shadow as shadow_logger
from scripts.tiny.rawseq_future_shadow_lock import attach_implementation_lock
from scripts.tiny.freeze_rawseq_low_path_ridge_research_candidate import stable_hash

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_future_shadow_cycles"


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return resolve_path(raw) if raw else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_step(name: str, fn: Callable[[], int]) -> dict[str, Any]:
    stdout = io.StringIO()
    started = datetime.now(UTC)
    with contextlib.redirect_stdout(stdout):
        exit_code = int(fn())
    finished = datetime.now(UTC)
    return {
        "step": name,
        "exit_code": exit_code,
        "started_at_iso": started.isoformat(),
        "finished_at_iso": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "stdout": stdout.getvalue(),
    }


def latest_dir(root: Path, pattern: str) -> Path | None:
    candidates = [p for p in root.glob(pattern) if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    output_root = env_path("RAWSEQ_DOWNSIDE_SHADOW_CYCLE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    cycle_dir = output_root / f"rawseq_downside_risk_future_shadow_cycle_{now_stamp()}"
    cycle_dir.mkdir(parents=True, exist_ok=False)

    started = datetime.now(UTC)
    steps: list[dict[str, Any]] = []
    status = "OK"
    try:
        steps.append(run_step("future_shadow_contract_parity_report", parity_report.main))
        if steps[-1]["exit_code"] != 0:
            status = "CONTRACT_PARITY_FAILED"
        if status == "OK":
            steps.append(run_step("future_shadow_logger", shadow_logger.main))
        if status == "OK" and steps[-1]["exit_code"] != 0:
            status = "LOGGER_FAILED"
        if status == "OK":
            steps.append(run_step("future_shadow_source_freshness_report", source_freshness_report.main))
        if status == "OK" and steps[-1]["exit_code"] != 0:
            status = "SOURCE_FRESHNESS_REPORT_FAILED"
        if status == "OK":
            steps.append(run_step("future_shadow_data_freshness_report", freshness_report.main))
        if status == "OK" and steps[-1]["exit_code"] != 0:
            status = "FRESHNESS_REPORT_FAILED"
        if status == "OK":
            steps.append(run_step("future_shadow_status_report", status_report.main))
        if status == "OK" and steps[-1]["exit_code"] != 0:
            status = "STATUS_REPORT_FAILED"
        if status == "OK":
            steps.append(run_step("future_shadow_operational_health_report", health_report.main))
        if status == "OK" and steps[-1]["exit_code"] != 0:
            status = "OPERATIONAL_HEALTH_REPORT_FAILED"
    except Exception as exc:  # pragma: no cover - defensive manifest capture
        status = "EXCEPTION"
        steps.append(
            {
                "step": "exception",
                "exit_code": 1,
                "started_at_iso": datetime.now(UTC).isoformat(),
                "finished_at_iso": datetime.now(UTC).isoformat(),
                "elapsed_seconds": 0.0,
                "stdout": "",
                "exception": repr(exc),
            }
        )
    finished = datetime.now(UTC)

    shadow_root = shadow_logger.env_path("RAWSEQ_DOWNSIDE_SHADOW_OUTPUT_ROOT", shadow_logger.DEFAULT_OUTPUT_ROOT)
    status_root = status_report.env_path("RAWSEQ_DOWNSIDE_SHADOW_STATUS_OUTPUT_ROOT", status_report.DEFAULT_OUTPUT_ROOT)
    freshness_root = freshness_report.env_path("RAWSEQ_DOWNSIDE_SHADOW_FRESHNESS_OUTPUT_ROOT", freshness_report.DEFAULT_OUTPUT_ROOT)
    parity_root = parity_report.env_path("RAWSEQ_DOWNSIDE_PARITY_OUTPUT_ROOT", parity_report.DEFAULT_OUTPUT_ROOT)
    source_freshness_root = source_freshness_report.env_path("RAWSEQ_DOWNSIDE_SOURCE_FRESHNESS_OUTPUT_ROOT", source_freshness_report.DEFAULT_OUTPUT_ROOT)
    health_root = health_report.env_path("RAWSEQ_DOWNSIDE_OPERATIONAL_HEALTH_OUTPUT_ROOT", health_report.DEFAULT_OUTPUT_ROOT)
    latest_parity_run = latest_dir(parity_root, "rawseq_downside_risk_shadow_contract_parity_*")
    latest_source_freshness_run = latest_dir(source_freshness_root, "rawseq_downside_risk_shadow_source_freshness_*")
    latest_shadow_run = latest_dir(shadow_root, "rawseq_downside_risk_future_shadow_*")
    latest_freshness_run = latest_dir(freshness_root, "rawseq_downside_risk_future_shadow_freshness_*")
    latest_status_run = latest_dir(status_root, "rawseq_downside_risk_future_shadow_status_*")
    latest_health_run = latest_dir(health_root, "rawseq_downside_risk_future_shadow_operational_health_*")
    manifest = {
        "cycle_dir": str(cycle_dir),
        "started_at_iso": started.isoformat(),
        "finished_at_iso": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "status": status,
        "latest_parity_run": str(latest_parity_run) if latest_parity_run else "",
        "latest_source_freshness_run": str(latest_source_freshness_run) if latest_source_freshness_run else "",
        "latest_shadow_run": str(latest_shadow_run) if latest_shadow_run else "",
        "latest_freshness_run": str(latest_freshness_run) if latest_freshness_run else "",
        "latest_status_run": str(latest_status_run) if latest_status_run else "",
        "latest_health_run": str(latest_health_run) if latest_health_run else "",
        "steps": steps,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "training": False,
        "recalibration": False,
        "threshold_changes": False,
    }
    attach_implementation_lock(manifest)
    manifest["cycle_manifest_sha256"] = stable_hash(manifest)
    write_json(cycle_dir / "rawseq_downside_risk_future_shadow_cycle_manifest.json", manifest)
    (cycle_dir / "rawseq_downside_risk_future_shadow_cycle.log").write_text(
        "\n\n".join(f"## {step['step']}\n{step.get('stdout', '')}" for step in steps),
        encoding="utf-8",
    )
    print(f"Cycle output: {cycle_dir}")
    print(f"Status: {status}")
    print(f"Latest parity run: {manifest['latest_parity_run']}")
    print(f"Latest source freshness run: {manifest['latest_source_freshness_run']}")
    print(f"Latest shadow run: {manifest['latest_shadow_run']}")
    print(f"Latest freshness run: {manifest['latest_freshness_run']}")
    print(f"Latest status run: {manifest['latest_status_run']}")
    print(f"Latest operational health run: {manifest['latest_health_run']}")
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
