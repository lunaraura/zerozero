#!/usr/bin/env python3
"""Advance the frozen downside-risk future shadow by one audited step.

This wrapper chooses the latest feature table, checks whether the public source
has advanced, optionally refreshes the feature table to capture new
not-yet-labelable rows, and runs one frozen paper-shadow cycle.

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

from scripts.tiny import refresh_rawseq_downside_risk_shadow_feature_table as refresh_table
from scripts.tiny import report_rawseq_downside_risk_shadow_source_freshness as source_freshness
from scripts.tiny import run_rawseq_downside_risk_future_shadow_cycle as shadow_cycle
from scripts.tiny import run_rawseq_downside_risk_future_paper_shadow as shadow_logger
from scripts.tiny.rawseq_future_shadow_lock import attach_implementation_lock
from scripts.tiny.freeze_rawseq_low_path_ridge_research_candidate import stable_hash

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_future_shadow_advances"


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return resolve_path(raw) if raw else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def latest_dir(root: Path, pattern: str) -> Path | None:
    candidates = [p for p in root.glob(pattern) if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_feature_table(refresh_root: Path) -> Path | None:
    candidates: list[Path] = []
    for run_dir in refresh_root.glob("rawseq_downside_risk_shadow_feature_refresh_*"):
        table = run_dir / "multi_horizon_training_table.csv"
        if is_usable_feature_table(table):
            candidates.append(table)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def is_usable_feature_table(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def choose_starting_feature_table(refresh_root: Path) -> Path:
    explicit = os.getenv("RAWSEQ_DOWNSIDE_ADVANCE_FEATURE_TABLE", "").strip()
    if explicit:
        return resolve_path(explicit)
    latest = latest_feature_table(refresh_root)
    if latest:
        return latest
    return shadow_logger.env_path("RAWSEQ_DOWNSIDE_SHADOW_FEATURE_TABLE", shadow_logger.DEFAULT_FEATURE_TABLE)


@contextlib.contextmanager
def temporary_env(updates: dict[str, str]):
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_step(name: str, fn: Callable[[], int], env_updates: dict[str, str] | None = None) -> dict[str, Any]:
    stdout = io.StringIO()
    started = datetime.now(UTC)
    with temporary_env(env_updates or {}), contextlib.redirect_stdout(stdout):
        exit_code = int(fn())
    finished = datetime.now(UTC)
    return {
        "step": name,
        "exit_code": exit_code,
        "started_at_iso": started.isoformat(),
        "finished_at_iso": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "stdout": stdout.getvalue(),
        "env_updates": env_updates or {},
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    output_root = env_path("RAWSEQ_DOWNSIDE_ADVANCE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    refresh_root = env_path("RAWSEQ_DOWNSIDE_REFRESH_OUTPUT_ROOT", refresh_table.DEFAULT_OUTPUT_ROOT)
    auto_refresh = parse_bool(os.getenv("RAWSEQ_DOWNSIDE_ADVANCE_AUTO_REFRESH", "true"))
    advance_dir = output_root / f"rawseq_downside_risk_future_shadow_advance_{now_stamp()}"
    advance_dir.mkdir(parents=True, exist_ok=False)

    started = datetime.now(UTC)
    steps: list[dict[str, Any]] = []
    status = "OK"
    selected_feature_table = choose_starting_feature_table(refresh_root)
    refreshed_feature_table = ""
    freshness_report_dir = None
    refresh_run_dir = None
    cycle_run_dir = None
    freshness_payload: dict[str, Any] = {}

    try:
        steps.append(
            run_step(
                "source_freshness_precheck",
                source_freshness.main,
                {"RAWSEQ_DOWNSIDE_SHADOW_FEATURE_TABLE": str(selected_feature_table)},
            )
        )
        if steps[-1]["exit_code"] != 0:
            status = "SOURCE_FRESHNESS_PRECHECK_FAILED"
        freshness_report_dir = latest_dir(source_freshness.DEFAULT_OUTPUT_ROOT, "rawseq_downside_risk_shadow_source_freshness_*")
        freshness_payload = read_json(freshness_report_dir / "rawseq_downside_risk_shadow_source_freshness.json") if freshness_report_dir else {}
        should_refresh = bool(freshness_payload.get("prediction_logging_refresh_recommended")) or bool(
            freshness_payload.get("label_maturity_refresh_recommended")
        )
        if status == "OK" and auto_refresh and should_refresh:
            before_refresh = latest_feature_table(refresh_root)
            steps.append(run_step("feature_table_refresh", refresh_table.main))
            if steps[-1]["exit_code"] != 0:
                status = "FEATURE_REFRESH_FAILED"
            after_refresh = latest_feature_table(refresh_root)
            if status == "OK" and after_refresh and after_refresh != before_refresh:
                selected_feature_table = after_refresh
                refreshed_feature_table = str(after_refresh)
                refresh_run_dir = after_refresh.parent
        if status == "OK":
            steps.append(
                run_step(
                    "future_shadow_cycle",
                    shadow_cycle.main,
                    {"RAWSEQ_DOWNSIDE_SHADOW_FEATURE_TABLE": str(selected_feature_table)},
                )
            )
            if steps[-1]["exit_code"] != 0:
                status = "SHADOW_CYCLE_FAILED"
            cycle_run_dir = latest_dir(shadow_cycle.DEFAULT_OUTPUT_ROOT, "rawseq_downside_risk_future_shadow_cycle_*")
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
    manifest = {
        "advance_dir": str(advance_dir),
        "started_at_iso": started.isoformat(),
        "finished_at_iso": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "status": status,
        "auto_refresh": auto_refresh,
        "selected_feature_table": str(selected_feature_table),
        "refreshed_feature_table": refreshed_feature_table,
        "source_freshness_report_dir": str(freshness_report_dir) if freshness_report_dir else "",
        "source_freshness_status": freshness_payload.get("status", ""),
        "prediction_logging_refresh_recommended": freshness_payload.get("prediction_logging_refresh_recommended", ""),
        "label_maturity_refresh_recommended": freshness_payload.get("label_maturity_refresh_recommended", ""),
        "refresh_run_dir": str(refresh_run_dir) if refresh_run_dir else "",
        "cycle_run_dir": str(cycle_run_dir) if cycle_run_dir else "",
        "steps": steps,
        "paper_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "training": False,
        "recalibration": False,
        "threshold_changes": False,
    }
    attach_implementation_lock(manifest, feature_table=selected_feature_table)
    manifest["advance_manifest_sha256"] = stable_hash(manifest)
    write_json(advance_dir / "rawseq_downside_risk_future_shadow_advance_manifest.json", manifest)
    (advance_dir / "rawseq_downside_risk_future_shadow_advance.log").write_text(
        "\n\n".join(f"## {step['step']}\n{step.get('stdout', '')}" for step in steps),
        encoding="utf-8",
    )
    print(f"Advance output: {advance_dir}")
    print(f"Status: {status}")
    print(f"Selected feature table: {selected_feature_table}")
    print(f"Refreshed feature table: {refreshed_feature_table}")
    print(f"Source freshness report: {manifest['source_freshness_report_dir']}")
    print(f"Cycle run: {manifest['cycle_run_dir']}")
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
