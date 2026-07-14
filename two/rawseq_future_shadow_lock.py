#!/usr/bin/env python3
"""Implementation lock helpers for the downside-risk future shadow.

The frozen model and acceptance rule are not enough for prospective evidence.
These helpers bind each future-shadow artifact to the code and ledger schema
that produced it, without changing the frozen candidate, thresholds, or gates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_DIR = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "rawseq_downside_risk_cpu_candidates"
    / "rawseq_downside_risk_cpu_candidate_20260711T233404Z"
)
DEFAULT_SHADOW_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_future_shadow"
DEFAULT_FEATURE_TABLE = Path(
    r"F:\rsio\rawseq_target_tournament_coarse_1s_300k_retry\mh_indicator_SOLUSDT_kraken_20260711T145015Z_fba19c8d\multi_horizon_training_table.csv"
)
MAX_INLINE_FILE_HASH_BYTES = 100 * 1024 * 1024
LOGGER_AUDIT_TAG = "rawseq-downside-risk-shadow-logger-v1"

IMPLEMENTATION_FILES = {
    "due_wrapper": "scripts/tiny/run_rawseq_downside_risk_future_shadow_if_due.py",
    "advance_wrapper": "scripts/tiny/run_rawseq_downside_risk_future_shadow_advance.py",
    "cycle_wrapper": "scripts/tiny/run_rawseq_downside_risk_future_shadow_cycle.py",
    "paper_shadow_logger": "scripts/tiny/run_rawseq_downside_risk_future_paper_shadow.py",
    "label_derivation_and_feature_refresh": "scripts/tiny/refresh_rawseq_downside_risk_shadow_feature_table.py",
    "source_freshness_report": "scripts/tiny/report_rawseq_downside_risk_shadow_source_freshness.py",
    "data_freshness_report": "scripts/tiny/report_rawseq_downside_risk_future_shadow_data_freshness.py",
    "contract_parity_report": "scripts/tiny/report_rawseq_downside_risk_shadow_contract_parity.py",
    "acceptance_status_report": "scripts/tiny/report_rawseq_downside_risk_future_shadow_status.py",
    "implementation_lock_module": "scripts/tiny/rawseq_future_shadow_lock.py",
}

LEDGER_FILES = {
    "cumulative_contract": "rawseq_downside_risk_future_shadow_cumulative_contract.json",
    "cumulative_decisions": "rawseq_downside_risk_future_shadow_cumulative_decisions.csv",
    "true_forward_decisions": "rawseq_downside_risk_future_shadow_true_forward_decisions.csv",
    "labeled_results": "rawseq_downside_risk_future_shadow_cumulative_labeled_results.csv",
    "summary": "rawseq_downside_risk_future_shadow_cumulative_summary.csv",
    "threshold_utility": "rawseq_downside_risk_future_shadow_cumulative_threshold_utility.csv",
    "feature_snapshots": "rawseq_downside_risk_future_shadow_cumulative_feature_snapshots.csv",
}


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_file_sha256(path: Path, max_bytes: int = MAX_INLINE_FILE_HASH_BYTES) -> str:
    if not path.exists():
        return ""
    if path.stat().st_size > max_bytes:
        return f"skipped_large_file_gt_{max_bytes}_bytes"
    return file_sha256(path)


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def latest_cumulative_dir(shadow_root: Path = DEFAULT_SHADOW_ROOT) -> Path | None:
    parent = shadow_root / "rawseq_downside_risk_future_shadow_cumulative"
    if not parent.exists():
        return None
    candidates = [p for p in parent.glob("*") if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def git_command(args: list[str]) -> tuple[str, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - defensive only
        return "", repr(exc)
    return proc.stdout.strip(), proc.stderr.strip()


def git_info() -> dict[str, Any]:
    branch, branch_err = git_command(["branch", "--show-current"])
    commit, commit_err = git_command(["rev-parse", "HEAD"])
    tags, tags_err = git_command(["tag", "--points-at", "HEAD"])
    audit_tag_commit, audit_tag_err = git_command(["rev-list", "-n", "1", LOGGER_AUDIT_TAG])
    tracked, tracked_err = git_command(["diff", "--name-only", "--", "scripts", "tests", "docs", "configs", "MEGA_README.md", "FILE_CATALOG.md"])
    staged, staged_err = git_command(["diff", "--cached", "--name-only", "--", "scripts", "tests", "docs", "configs", "MEGA_README.md", "FILE_CATALOG.md"])
    untracked, untracked_err = git_command(
        ["ls-files", "--others", "--exclude-standard", "--", "scripts", "tests", "docs", "configs", "MEGA_README.md", "FILE_CATALOG.md"]
    )
    dirty_paths = sorted({line for block in [tracked, staged, untracked] for line in block.splitlines() if line})
    return {
        "git_branch": branch,
        "git_commit": commit,
        "git_commit_short": commit[:12] if commit else "",
        "git_dirty": bool(dirty_paths),
        "git_dirty_scope": "scripts_tests_docs_configs_catalogs",
        "git_dirty_path_count": len(dirty_paths),
        "git_dirty_paths_sha256": stable_hash(dirty_paths),
        "git_status_porcelain_sha256": stable_hash(dirty_paths),
        "git_tags_at_head": [line for line in tags.splitlines() if line],
        "logger_audit_tag": LOGGER_AUDIT_TAG,
        "logger_audit_tag_commit": audit_tag_commit,
        "logger_audit_tag_present": bool(audit_tag_commit),
        "git_errors": {
            "branch": branch_err,
            "commit": commit_err,
            "tags": tags_err,
            "logger_audit_tag": audit_tag_err,
            "tracked_diff": tracked_err,
            "staged_diff": staged_err,
            "untracked": untracked_err,
        },
    }


def implementation_file_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key, rel in IMPLEMENTATION_FILES.items():
        path = PROJECT_ROOT / rel
        rows[key] = {
            "relative_path": rel,
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "sha256": file_sha256(path) if path.exists() else "",
        }
    return rows


def csv_schema(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {"path": str(path), "exists": path.exists(), "columns": [], "column_count": 0, "columns_sha256": ""}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        columns = next(reader, [])
    return {
        "path": str(path),
        "exists": True,
        "columns": columns,
        "column_count": len(columns),
        "columns_sha256": stable_hash(columns),
    }


def ledger_schema(cumulative_dir: Path | None) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    if cumulative_dir is None:
        return rows
    for key, name in LEDGER_FILES.items():
        path = cumulative_dir / name
        if path.suffix.lower() == ".csv":
            rows[key] = csv_schema(path)
        else:
            rows[key] = {
                "path": str(path),
                "exists": path.exists(),
                "sha256": file_sha256(path) if path.exists() else "",
            }
    return rows


def stable_ledger_schema_identity(schema: dict[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for key, row in schema.items():
        if "columns_sha256" in row:
            identity[key] = {
                "exists": row.get("exists", False),
                "column_count": row.get("column_count", 0),
                "columns_sha256": row.get("columns_sha256", ""),
            }
        else:
            identity[key] = {"exists": row.get("exists", False)}
    return identity


def candidate_hashes(candidate_dir: Path) -> dict[str, Any]:
    contract_path = candidate_dir / "rawseq_downside_risk_cpu_candidate_contract.json"
    rule_path = candidate_dir / "rawseq_downside_risk_future_acceptance_rule.json"
    contract = safe_read_json(contract_path)
    rule = safe_read_json(rule_path)
    model_path = resolve_path(contract.get("model_path", candidate_dir / "rawseq_downside_risk_cpu_candidate_model.npz"))
    scaler_path = resolve_path(contract.get("scalers_path", candidate_dir / "rawseq_downside_risk_cpu_candidate_scalers.npz"))
    return {
        "candidate_dir": str(candidate_dir),
        "candidate_contract_path": str(contract_path),
        "candidate_contract_file_sha256": file_sha256(contract_path) if contract_path.exists() else "",
        "candidate_contract_sha256": contract.get("contract_sha256", ""),
        "acceptance_rule_path": str(rule_path),
        "acceptance_rule_file_sha256": file_sha256(rule_path) if rule_path.exists() else "",
        "acceptance_rule_sha256": rule.get("acceptance_rule_sha256", ""),
        "model_path": str(model_path),
        "model_file_sha256": file_sha256(model_path) if model_path.exists() else "",
        "scalers_path": str(scaler_path),
        "scalers_file_sha256": file_sha256(scaler_path) if scaler_path.exists() else "",
        "frozen_at_iso": rule.get("frozen_at_iso", contract.get("frozen_at_iso", "")),
    }


def build_implementation_lock(
    candidate_dir: str | Path | None = None,
    cumulative_dir: str | Path | None = None,
    feature_table: str | Path | None = None,
) -> dict[str, Any]:
    resolved_candidate = resolve_path(candidate_dir) if candidate_dir else DEFAULT_CANDIDATE_DIR
    resolved_cumulative = resolve_path(cumulative_dir) if cumulative_dir else latest_cumulative_dir()
    resolved_feature_table = resolve_path(feature_table) if feature_table else DEFAULT_FEATURE_TABLE
    files = implementation_file_rows()
    live_ledger_schema = ledger_schema(resolved_cumulative)
    stable_identity_payload = {
        "git": git_info(),
        "implementation_files": files,
        "candidate": candidate_hashes(resolved_candidate),
        "ledger_schema_identity": stable_ledger_schema_identity(live_ledger_schema),
    }
    lock = {
        "lock_version": "rawseq_future_shadow_lock_v1",
        "generated_at_iso": datetime.now(UTC).isoformat(),
        **stable_identity_payload["git"],
        "implementation_files": files,
        "implementation_files_sha256": stable_hash({k: v.get("sha256", "") for k, v in files.items()}),
        "candidate": stable_identity_payload["candidate"],
        "cumulative_dir": str(resolved_cumulative) if resolved_cumulative else "",
        "cumulative_ledger_schema": live_ledger_schema,
        "cumulative_ledger_schema_sha256": stable_hash(live_ledger_schema),
        "cumulative_ledger_schema_identity_sha256": stable_hash(stable_identity_payload["ledger_schema_identity"]),
        "feature_table": str(resolved_feature_table),
        "feature_table_exists": resolved_feature_table.exists(),
        "feature_table_bytes": resolved_feature_table.stat().st_size if resolved_feature_table.exists() else 0,
        "feature_table_sha256": optional_file_sha256(resolved_feature_table),
    }
    lock["code_revision_lock_sha256"] = stable_hash(stable_identity_payload)
    lock["logger_implementation_lock_sha256"] = stable_hash(lock)
    return lock


def attach_implementation_lock(
    manifest: dict[str, Any],
    candidate_dir: str | Path | None = None,
    cumulative_dir: str | Path | None = None,
    feature_table: str | Path | None = None,
) -> dict[str, Any]:
    lock = build_implementation_lock(candidate_dir=candidate_dir, cumulative_dir=cumulative_dir, feature_table=feature_table)
    manifest["logger_implementation_lock"] = lock
    manifest["logger_implementation_lock_sha256"] = lock["logger_implementation_lock_sha256"]
    manifest["code_revision_lock_sha256"] = lock["code_revision_lock_sha256"]
    return manifest


def main() -> int:
    lock = build_implementation_lock()
    print(json.dumps(lock, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
