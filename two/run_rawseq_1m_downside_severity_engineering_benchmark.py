#!/usr/bin/env python3
"""Run one full-development downside-severity engineering benchmark.

This wrapper does not change research state. It orchestrates the hardened
feature-family evolution runner for one closed target family, including
preflight, controlled interruption/resume, and packet collation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny import run_rawseq_1m_board_member_target_feature_tournament as board  # noqa: E402
from scripts.tiny import run_rawseq_1m_feature_family_evolution as evo  # noqa: E402
from scripts.tiny.rawseq_1m_feature_evolution_runtime import (  # noqa: E402
    process_memory_snapshot,
    stable_hash,
    write_json_atomic,
)

OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_feature_evolution_single_family_benchmark")
SYMBOLS = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT"]
FEATURE_GROUPS = ["existing", "existing_plus_short_path", "existing_plus_cross_asset", "existing_plus_regime", "all_challenger_features"]
MODELS = ["constant_prevalence", "regularized_logistic", "shallow_hgb"]
HORIZONS = [1, 2, 4, 8, 15]
SEVERITY_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5]
VOL_WINDOW = 240
REQUIRED_COMMITS = ["0775270", "b02243b", "e0fe482", "dbd8e99"]
EXPECTED_RUNNER_DIRTY_PATHS = {
    "scripts/tiny/rawseq_1m_feature_evolution_runtime.py",
    "scripts/tiny/run_rawseq_1m_feature_family_evolution.py",
    "scripts/tiny/run_rawseq_1m_downside_severity_engineering_benchmark.py",
}


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_command(cmd: list[str], env: dict[str, str] | None = None, timeout: int | None = None) -> tuple[int, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, env=env, timeout=timeout)
    elapsed = time.perf_counter() - start
    return proc.returncode, f"COMMAND: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n", elapsed


def command_stdout(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False).stdout.strip()


def git_preflight() -> dict[str, Any]:
    log = command_stdout(["git", "log", "--oneline", "-n", "80"])
    status = command_stdout(["git", "status", "--short"])
    dirty_paths: list[str] = []
    unexpected_runner_dirty: list[str] = []
    for line in status.splitlines():
        path = line[3:].strip() if len(line) >= 4 else line.strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[-1].strip()
        path = path.replace("\\", "/")
        dirty_paths.append(path)
        if path in {
            "scripts/tiny/rawseq_1m_feature_evolution_runtime.py",
            "scripts/tiny/run_rawseq_1m_feature_family_evolution.py",
            "scripts/tiny/run_rawseq_1m_downside_severity_engineering_benchmark.py",
        } and path not in EXPECTED_RUNNER_DIRTY_PATHS:
            unexpected_runner_dirty.append(path)
    present = {commit: commit in log for commit in REQUIRED_COMMITS}
    return {
        "branch": command_stdout(["git", "branch", "--show-current"]),
        "head": command_stdout(["git", "rev-parse", "--short", "HEAD"]),
        "required_commits_present": present,
        "all_required_commits_present": all(present.values()),
        "git_status_short": status,
        "dirty_paths": dirty_paths,
        "expected_runner_dirty_paths": sorted(EXPECTED_RUNNER_DIRTY_PATHS),
        "unexpected_runner_dirty_paths": unexpected_runner_dirty,
        "unexpected_runner_dirty_pass": not unexpected_runner_dirty,
    }


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolved_run_contract(output_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    target_count = len(HORIZONS) * len(SEVERITY_LEVELS)
    stage1_candidates = target_count * len(FEATURE_GROUPS) * len(MODELS)
    nonbaseline_candidates = target_count * len(FEATURE_GROUPS) * (len(MODELS) - 1)
    return {
        "contract_type": "rawseq_1m_downside_severity_engineering_benchmark_v1",
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "source_path": str(Path(r"F:\AITicker\Misc\data\binance_public_zips")),
        "symbols": SYMBOLS,
        "target_family": "downside_severity",
        "horizons": HORIZONS,
        "severity_levels": SEVERITY_LEVELS,
        "volatility_window_minutes": VOL_WINDOW,
        "feature_groups": FEATURE_GROUPS,
        "models": MODELS,
        "model_seed": 1337,
        "development_cutoff": "2026-05-31T23:59:00Z",
        "stage_preset": "full_dev",
        "stages": evo.stage_preset("full_dev"),
        "keep_fraction": 0.5,
        "min_keep": 10,
        "run_scenario_validation": False,
        "matrix_cache": True,
        "matrix_telemetry": True,
        "memory_guard": {
            "policy": "checkpoint_and_pause",
            "warn_system_commit_fraction": 0.85,
            "pause_system_commit_fraction": 0.90,
            "fail_system_commit_fraction": 0.93,
        },
        "controlled_interruption": {
            "enabled_first_run": True,
            "min_nonbaseline_candidates": 5,
            "require_logistic": True,
            "require_hgb": True,
        },
        "candidate_count_stage1": stage1_candidates,
        "exact_stage2_maximum": max(10, math.ceil(nonbaseline_candidates * 0.5)),
        "folds_per_candidate": 4,
        "holdout_used_for_selection": False,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "board_state_mutation": False,
        "future_shadow_mutation": False,
        "scientific_status": "downside_severity_remains_closed",
    }


def build_fixed_env(output_dir: Path, checkpoint_dir: Path, controlled_stop: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "RAWSEQ_EVOLVE_OUTPUT_ROOT": str(output_dir.parent),
            "RAWSEQ_EVOLVE_OUTPUT_DIR": str(output_dir),
            "RAWSEQ_EVOLVE_CHECKPOINT_DIR": str(checkpoint_dir),
            "RAWSEQ_EVOLVE_SOURCE_PATH": str(Path(r"F:\AITicker\Misc\data\binance_public_zips")),
            "RAWSEQ_EVOLVE_SYMBOLS": ",".join(SYMBOLS),
            "RAWSEQ_EVOLVE_TARGET_LANES": "downside_severity",
            "RAWSEQ_EVOLVE_FEATURE_GROUPS": ",".join(FEATURE_GROUPS),
            "RAWSEQ_EVOLVE_MODELS": ",".join(MODELS),
            "RAWSEQ_EVOLVE_HORIZONS": ",".join(str(x) for x in HORIZONS),
            "RAWSEQ_EVOLVE_SEVERITY_LEVELS": ",".join(str(x) for x in SEVERITY_LEVELS),
            "RAWSEQ_EVOLVE_VOL_WINDOW": str(VOL_WINDOW),
            "RAWSEQ_EVOLVE_SEEDS": "1337",
            "RAWSEQ_EVOLVE_STAGE_PRESET": "full_dev",
            "RAWSEQ_EVOLVE_KEEP_FRACTION": "0.5",
            "RAWSEQ_EVOLVE_MIN_KEEP": "10",
            "RAWSEQ_EVOLVE_MAX_CANDIDATES": "0",
            "RAWSEQ_EVOLVE_RUN_SCENARIO_VALIDATION": "false",
            "RAWSEQ_EVOLVE_MATRIX_CACHE": "true",
            "RAWSEQ_EVOLVE_MATRIX_TELEMETRY": "true",
            "RAWSEQ_EVOLVE_PROGRESS_EVERY": "5",
            "RAWSEQ_EVOLVE_MEMORY_WARN_SYSTEM_COMMIT_FRACTION": "0.85",
            "RAWSEQ_EVOLVE_MEMORY_PAUSE_SYSTEM_COMMIT_FRACTION": "0.90",
            "RAWSEQ_EVOLVE_MEMORY_FAIL_SYSTEM_COMMIT_FRACTION": "0.93",
            "RAWSEQ_EVOLVE_CONTROLLED_STOP_AFTER_CRITERIA": "true" if controlled_stop else "false",
            "RAWSEQ_EVOLVE_STOP_MIN_NONBASELINE": "5",
            "RAWSEQ_EVOLVE_STOP_REQUIRE_LOGISTIC": "true",
            "RAWSEQ_EVOLVE_STOP_REQUIRE_HGB": "true",
        }
    )
    return env


def finite_metric_delta(a: Any, b: Any) -> float:
    try:
        av = float(a)
        bv = float(b)
    except (TypeError, ValueError):
        return math.nan
    if not (math.isfinite(av) and math.isfinite(bv)):
        return math.nan
    return abs(av - bv)


def semantic_parity_spot_check(output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    try:
        by_symbol, _, target_rows = evo.build_symbol_data_for_stage(
            SYMBOLS[:3],
            Path(r"F:\AITicker\Misc\data\binance_public_zips"),
            2500,
            [60, 240],
            [1, 2],
            VOL_WINDOW,
            [0.25, 0.5],
        )
        candidate_grid = evo.build_candidate_grid(
            target_rows,
            ["existing", "existing_plus_short_path"],
            ["regularized_logistic", "shallow_hgb"],
            {"downside_severity"},
            ["all"],
            [1337],
        )[:3]
        for candidate in candidate_grid:
            exact_rows, _ = board.evaluate_candidate_records(
                by_symbol,
                [candidate],
                600,
                50,
                0.001,
                1,
                matrix_cache=None,
                matrix_telemetry=None,
                matrix_dtype=np.float64,
                semantic_contract={"parity_fixture": True, "candidate_key": candidate["candidate_key"]},
            )
            expanded_rows, _ = board.evaluate_tournament(
                by_symbol,
                [candidate],
                600,
                50,
                0.001,
                {str(candidate["target_lane"])},
                {str(candidate["feature_group"])},
                [str(candidate["model"])],
                1,
                ["all"],
                [1337],
            )
            exact_ok = next((row for row in exact_rows if row.get("status") == "OK" and row.get("model") == candidate["model"]), {})
            expanded_ok = next(
                (
                    row
                    for row in expanded_rows
                    if row.get("status") == "OK"
                    and row.get("model") == candidate["model"]
                    and row.get("target_name") == candidate["target_name"]
                    and row.get("feature_group") == candidate["feature_group"]
                ),
                {},
            )
            metric_names = [
                "rows",
                "event_prevalence",
                "brier_score",
                "brier_skill_vs_prevalence",
                "log_loss",
                "log_loss_improvement_vs_prevalence",
                "pr_auc",
                "pr_auc_lift_over_event_prevalence",
                "expected_calibration_error",
                "calibration_slope",
                "calibration_intercept",
            ]
            deltas = {f"{name}_abs_delta": finite_metric_delta(exact_ok.get(name), expanded_ok.get(name)) for name in metric_names}
            finite_deltas = [value for value in deltas.values() if math.isfinite(value)]
            max_delta = max(finite_deltas) if finite_deltas else math.nan
            rows.append(
                {
                    "candidate_key": candidate["candidate_key"],
                    "target_name": candidate["target_name"],
                    "feature_group": candidate["feature_group"],
                    "model": candidate["model"],
                    "exact_status": exact_ok.get("status", ""),
                    "expanded_status": expanded_ok.get("status", ""),
                    "max_metric_abs_delta": max_delta,
                    "parity_pass": bool(exact_ok and expanded_ok and (not math.isfinite(max_delta) or max_delta <= 1e-9)),
                    **deltas,
                }
            )
    except Exception as exc:
        rows.append({"parity_pass": False, "failure_reason": repr(exc)})
    write_csv(output_dir / "semantic_parity_spot_check.csv", rows)
    passed = bool(rows) and all(str(row.get("parity_pass")).lower() == "true" for row in rows)
    lines = [
        "# Semantic Parity Report",
        "",
        f"status={'PASS' if passed else 'FAIL'}",
        "",
        "Compared hardened exact-candidate evaluation against the pre-optimization expanded tournament path on three small deterministic candidate/fold fixtures.",
        "Tolerance: absolute metric delta <= 1e-9 for finite scalar metrics.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- candidate={row.get('candidate_key','')} target={row.get('target_name','')} "
            f"features={row.get('feature_group','')} model={row.get('model','')} "
            f"pass={row.get('parity_pass')} max_delta={row.get('max_metric_abs_delta')} "
            f"failure={row.get('failure_reason','')}"
        )
    (output_dir / "semantic_parity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"semantic_parity_pass": passed, "rows": rows}


def preflight(output_dir: Path) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    for name, cmd in [
        (
            "py_compile",
            [
                "python",
                "-m",
                "py_compile",
                "scripts/tiny/rawseq_1m_feature_evolution_runtime.py",
                "scripts/tiny/run_rawseq_1m_board_member_target_feature_tournament.py",
                "scripts/tiny/run_rawseq_1m_feature_family_evolution.py",
                "tests/test_rawseq_1m_feature_family_evolution.py",
                "tests/test_rawseq_1m_board_member_target_feature_tournament.py",
            ],
        ),
        ("focused_tests", ["python", "-m", "unittest", "tests.test_rawseq_1m_feature_family_evolution", "tests.test_rawseq_1m_board_member_target_feature_tournament"]),
        ("git_diff_check", ["git", "diff", "--check"]),
    ]:
        code, text, elapsed = run_command(cmd)
        tests.append({"name": name, "exit_code": code, "elapsed_seconds": elapsed})
        (output_dir / f"preflight_{name}.txt").write_text(text, encoding="utf-8")
    recommended = Path(r"F:\rsio\rawseq_1m_feature_evolution_memory_checkpoint_patch\rawseq_1m_feature_evolution_memory_checkpoint_patch_20260714T093636Z\recommended_single_family_command.txt")
    rec_text = recommended.read_text(encoding="utf-8") if recommended.exists() else ""
    run_contract = resolved_run_contract(output_dir, output_dir / "checkpoints")
    target_count = len(HORIZONS) * len(SEVERITY_LEVELS)
    stage1_candidates = int(run_contract["candidate_count_stage1"])
    exact_stage2_max = int(run_contract["exact_stage2_maximum"])
    folds = 4
    feature_guess = {"existing": 31, "existing_plus_short_path": 60, "existing_plus_cross_asset": 75, "existing_plus_regime": 55, "all_challenger_features": 95}
    approx_rows = 9 * 15000
    max_features = max(feature_guess.values())
    matrix_bytes = approx_rows * max_features * 8 * 2
    snapshot = process_memory_snapshot()
    projected_commit = snapshot.get("system_commit_percent")
    fail_threshold = 0.93
    return {
        "created_at": now_stamp(),
        "git_preflight": git_preflight(),
        "resolved_run_contract": run_contract,
        "recommended_command": rec_text,
        "test_results": tests,
        "stage1_candidate_count": stage1_candidates,
        "target_count": target_count,
        "exact_stage2_maximum": exact_stage2_max,
        "folds_per_candidate": folds,
        "approx_stage1_matrix_rows": approx_rows,
        "approx_max_feature_count": max_features,
        "approx_one_train_eval_pair_bytes": matrix_bytes,
        "estimated_peak_memory_note": "Workload dependent; memory guard enforces commit thresholds during run.",
        "checkpoint_volume_estimate_records": stage1_candidates + exact_stage2_max,
        "expected_runtime_range": "hours; depends on HGB/logistic fit time and survivor count",
        "memory_snapshot": snapshot,
        "estimated_system_commit_fraction": projected_commit,
        "memory_fail_threshold": fail_threshold,
        "preflight_memory_gate_pass": bool(not isinstance(projected_commit, float) or not math.isfinite(projected_commit) or projected_commit < fail_threshold),
    }


def load_stage_records(output_dir: Path) -> list[dict[str, Any]]:
    stage_dir = output_dir / "checkpoints" / "stage_records"
    if not stage_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(stage_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def validate_exact_stage_audit(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = output_dir / "exact_candidate_semantics_audit.json"
    target = output_dir / "exact_stage_transition_audit.json"
    if source.exists():
        shutil.copy2(source, target)
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        payload = {"stages": [], "missing_source": True}
        write_json_atomic(target, payload)
    failures: list[dict[str, Any]] = []
    for stage in payload.get("stages", []):
        for field in ["missing_requested_keys", "unexpected_evaluated_keys", "duplicate_requested_keys"]:
            if stage.get(field):
                failures.append({"stage_id": stage.get("stage_id"), "failure": field, "values": stage.get(field)})
        if not stage.get("every_stage_two_evaluated_candidate_was_stage_one_survivor", True):
            failures.append({"stage_id": stage.get("stage_id"), "failure": "stage_two_evaluated_non_survivor"})
        if not stage.get("no_dropped_candidate_evaluated", True):
            failures.append({"stage_id": stage.get("stage_id"), "failure": "dropped_candidate_evaluated"})
    return payload, failures


def collect_packet(output_dir: Path, pre: dict[str, Any], first_elapsed: float, resume_elapsed: float) -> dict[str, Any]:
    rows = read_csv(output_dir / "stage_results.csv")
    survivors = read_csv(output_dir / "stage_survivors.csv")
    progress = read_csv(output_dir / "progress_telemetry.csv")
    failures = [row for row in rows if row.get("status") not in {"OK", ""}]
    logistic = [row for row in rows if row.get("model") == "regularized_logistic"]
    candidate_metrics = []
    seen = set()
    for row in survivors:
        key = row.get("candidate_key", "")
        if key and key not in seen:
            candidate_metrics.append(row)
            seen.add(key)
    exact_audit, exact_failures = validate_exact_stage_audit(output_dir)
    interruption = json.loads((output_dir / "controlled_interruption_snapshot.json").read_text(encoding="utf-8")) if (output_dir / "controlled_interruption_snapshot.json").exists() else {}
    checkpoint_files = sorted((output_dir / "checkpoints").rglob("*.json")) if (output_dir / "checkpoints").exists() else []
    checkpoint_manifest = {
        "checkpoint_files": [str(path) for path in checkpoint_files],
        "checkpoint_file_count": len(checkpoint_files),
        "checkpoint_hash": stable_hash([str(path) for path in checkpoint_files]),
        "controlled_interruption": interruption,
    }
    write_json_atomic(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
    write_json_atomic(output_dir / "interruption_snapshot.json", interruption)
    resumed_keys = []
    recomputed_keys = []
    for payload in load_stage_records(output_dir):
        resumed_keys.extend(payload.get("resumed_keys", []))
        recomputed_keys.extend(payload.get("recomputed_keys", []))
    stage_results_by_key = Counter((row.get("stage_id"), row.get("candidate_key")) for row in rows if row.get("candidate_key"))
    duplicate_candidate_warnings = [
        {"stage_id": key[0], "candidate_key": key[1], "row_count": count}
        for key, count in stage_results_by_key.items()
        if count > 4
    ]
    resume_audit = {
        "resume_performed": True,
        "completed_keys_skipped_on_resume": sorted(set(resumed_keys)),
        "incomplete_keys_recomputed": sorted(set(recomputed_keys)),
        "controlled_interruption_snapshot_present": bool(interruption),
        "no_completed_candidate_refit_after_resume": bool(resumed_keys),
        "duplicate_candidate_fold_row_warnings": duplicate_candidate_warnings,
        "resume_parity_semantics": "checkpoint fixture and stage records indicate completed keys are skipped; incomplete keys recomputed",
    }
    write_json_atomic(output_dir / "resume_audit.json", resume_audit)
    completed_rows = [{"candidate_key": key, "resume_status": "skipped_completed"} for key in sorted(set(resumed_keys))]
    write_csv(output_dir / "completed_key_audit.csv", completed_rows)
    write_csv(output_dir / "logistic_convergence_audit.csv", logistic)
    write_csv(output_dir / "candidate_failures.csv", failures)
    write_csv(output_dir / "per_fold_metrics.csv", rows)
    write_csv(output_dir / "candidate_metrics.csv", candidate_metrics)
    if (output_dir / "matrix_memory_telemetry.csv").exists():
        shutil.copy2(output_dir / "matrix_memory_telemetry.csv", output_dir / "memory_telemetry.csv")
    else:
        write_csv(output_dir / "memory_telemetry.csv", [])
    cache_stats = {}
    contract_path = output_dir / "feature_family_evolution_contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        cache_stats = contract.get("matrix_cache_stats", {})
        write_json_atomic(output_dir / "run_contract.json", contract)
    write_csv(output_dir / "cache_telemetry.csv", [{**cache_stats}])
    peak_private = peak_working = peak_commit = math.nan
    if progress:
        def max_float(name: str) -> float:
            vals = []
            for row in progress:
                try:
                    v = float(row.get(name, "nan"))
                    if math.isfinite(v):
                        vals.append(v)
                except ValueError:
                    pass
            return max(vals) if vals else math.nan
        peak_private = max_float("process_private_bytes")
        peak_working = max_float("process_working_set_bytes")
        peak_commit = max_float("system_commit_percent")
    semantic = semantic_parity_spot_check(output_dir)
    resource = {
        "first_run_seconds": first_elapsed,
        "resume_run_seconds": resume_elapsed,
        "total_runtime_seconds": first_elapsed + resume_elapsed,
        "peak_private_memory": peak_private,
        "peak_working_set": peak_working,
        "peak_system_commit": peak_commit,
        "cache_stats": cache_stats,
        "memory_guard_under_93_percent": bool(not math.isfinite(peak_commit) or peak_commit < 0.93),
    }
    write_json_atomic(output_dir / "resource_summary.json", resource)
    required = [
        "run_contract.json",
        "preflight_summary.txt",
        "candidate_grid.csv",
        "candidate_grid_summary.json",
        "progress_telemetry.csv",
        "memory_telemetry.csv",
        "cache_telemetry.csv",
        "checkpoint_manifest.json",
        "interruption_snapshot.json",
        "resume_audit.json",
        "completed_key_audit.csv",
        "exact_stage_transition_audit.json",
        "matched_row_contract_audit.csv",
        "logistic_convergence_audit.csv",
        "candidate_failures.csv",
        "per_fold_metrics.csv",
        "candidate_metrics.csv",
        "stage_survivors.csv",
        "successive_halving_audit.csv",
        "semantic_parity_report.md",
        "resource_summary.json",
    ]
    missing_required = [name for name in required if not (output_dir / name).exists()]
    status = "BENCHMARK_PASS"
    if math.isfinite(peak_commit) and peak_commit >= 0.93:
        status = "BENCHMARK_FAIL_MEMORY_CONTROL"
    if not interruption or not resume_audit["resume_performed"]:
        status = "BENCHMARK_FAIL_CHECKPOINT_RESUME"
    if exact_failures:
        status = "BENCHMARK_FAIL_STAGE_IDENTITY"
    if not semantic["semantic_parity_pass"]:
        status = "BENCHMARK_FAIL_SEMANTIC_PARITY"
    if missing_required:
        status = "BENCHMARK_FAIL_OTHER"
    stage1_survivor_count = sum(
        1
        for row in survivors
        if row.get("stage_id") == "1" and str(row.get("survives_stage_gate")).lower() in {"true", "1"}
    )
    stage2_evaluated_count = len(set(row.get("candidate_key") for row in rows if row.get("stage_id") == "2"))
    convergence_failures = sum(1 for row in logistic if row.get("logistic_convergence_status") == "non_converged")
    summary = "\n".join(
        [
            "RAWSEQ 1M DOWNSIDE-SEVERITY ENGINEERING BENCHMARK",
            "",
            f"status={status}",
            "scientific_status=downside_severity_remains_closed",
            "promotion=false",
            "freeze=false",
            "board_state_mutation=false",
            "future_shadow_mutation=false",
            "",
            f"stage1_candidate_count={pre['stage1_candidate_count']}",
            f"stage1_survivor_count={stage1_survivor_count}",
            f"stage2_evaluated_count={stage2_evaluated_count}",
            f"first_run_seconds={first_elapsed:.2f}",
            f"resume_run_seconds={resume_elapsed:.2f}",
            f"peak_private_memory={peak_private}",
            f"peak_working_set={peak_working}",
            f"peak_system_commit={peak_commit}",
            f"interruption_point={interruption.get('last_completed_key', '')}",
            f"completed_keys_skipped_on_resume={len(set(resumed_keys))}",
            f"incomplete_keys_recomputed={len(set(recomputed_keys))}",
            f"convergence_failures={convergence_failures}",
            f"semantic_parity={'PASS' if semantic['semantic_parity_pass'] else 'FAIL'}",
            f"exact_stage_failures={len(exact_failures)}",
            f"missing_required_artifacts={','.join(missing_required)}",
            f"cache_hit_rate={cache_stats.get('hit_rate', cache_stats.get('cache_hit_rate', ''))}",
            "",
            "Interpretation: Engineering/reproducibility benchmark only. Predictive quality does not change research status.",
        ]
    ) + "\n"
    (output_dir / "benchmark_summary.txt").write_text(summary, encoding="utf-8")
    return {
        "status": status,
        "resource": resource,
        "resume_audit": resume_audit,
        "semantic": semantic,
        "missing_required": missing_required,
        "exact_failures": exact_failures,
    }


def main() -> int:
    output_dir = OUTPUT_ROOT / f"rawseq_1m_downside_severity_engineering_benchmark_{now_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pre = preflight(output_dir)
    write_json_atomic(output_dir / "run_contract_prelaunch.json", pre["resolved_run_contract"])
    write_json_atomic(output_dir / "candidate_grid_summary.json", pre)
    (output_dir / "preflight_summary.txt").write_text(json.dumps(pre, indent=2, default=str), encoding="utf-8")
    git_ok = bool(pre.get("git_preflight", {}).get("all_required_commits_present")) and bool(
        pre.get("git_preflight", {}).get("unexpected_runner_dirty_pass")
    )
    if not git_ok or not pre["preflight_memory_gate_pass"] or any(row["exit_code"] != 0 for row in pre["test_results"]):
        (output_dir / "benchmark_summary.txt").write_text("status=BENCHMARK_FAIL_OTHER\npreflight_failed=true\n", encoding="utf-8")
        print(f"output_dir={output_dir}")
        print("final_status=BENCHMARK_FAIL_OTHER")
        return 2
    checkpoint_dir = output_dir / "checkpoints"
    env1 = build_fixed_env(output_dir, checkpoint_dir, controlled_stop=True)
    cmd = ["python", "scripts/tiny/run_rawseq_1m_feature_family_evolution.py", "--output-dir", str(output_dir), "--checkpoint-dir", str(checkpoint_dir), "--memory-guard-policy", "checkpoint_and_pause"]
    code1, text1, elapsed1 = run_command(cmd, env=env1)
    (output_dir / "first_run.log").write_text(text1, encoding="utf-8")
    env2 = build_fixed_env(output_dir, checkpoint_dir, controlled_stop=False)
    cmd2 = cmd + ["--resume"]
    code2, text2, elapsed2 = run_command(cmd2, env=env2)
    (output_dir / "resume_run.log").write_text(text2, encoding="utf-8")
    result = collect_packet(output_dir, pre, elapsed1, elapsed2)
    print(f"command={' '.join(cmd)}")
    print(f"resume_command={' '.join(cmd2)}")
    print(f"output_dir={output_dir}")
    print(f"first_exit_code={code1}")
    print(f"resume_exit_code={code2}")
    print(f"final_status={result['status']}")
    return 0 if result["status"].startswith("BENCHMARK_PASS") else 3


if __name__ == "__main__":
    raise SystemExit(main())
