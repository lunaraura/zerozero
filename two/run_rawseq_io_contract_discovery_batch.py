#!/usr/bin/env python3
"""Run bounded rawseq walk-forward discovery from an I/O contract grid.

This is an orchestration wrapper around run_rawseq_recorded_walkforward_evolution.py.
It uses public/recorded source data only and writes research outputs. It never
uses private APIs, places orders, promotes models, or mutates champion folders.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WF_SCRIPT = PROJECT_ROOT / "scripts" / "tiny" / "run_rawseq_recorded_walkforward_evolution.py"
GRID_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_io_contract_grids"

GRID_PATH_ENV = os.getenv("RAWSEQ_IO_GRID_PATH", "").strip()
OUTPUT_ROOT_ENV = os.getenv(
    "RAWSEQ_IO_DISCOVERY_OUTPUT_ROOT",
    str(PROJECT_ROOT / "data" / "research" / "rawseq_io_contract_discovery_batches"),
).strip()
MAX_CONTRACTS = int(float(os.getenv("RAWSEQ_IO_DISCOVERY_MAX_CONTRACTS", "3")))
MAX_CANDIDATES = int(float(os.getenv("RAWSEQ_IO_DISCOVERY_MAX_CANDIDATES", "3")))
MAX_RUNTIME_SECONDS = float(os.getenv("RAWSEQ_IO_DISCOVERY_MAX_RUNTIME_SECONDS", "900"))
MAX_WINDOWS = int(float(os.getenv("RAWSEQ_IO_DISCOVERY_MAX_WINDOWS", "1")))
SEEDS = os.getenv("RAWSEQ_IO_DISCOVERY_SEEDS", os.getenv("RAWSEQ_WF_SEEDS", "900")).strip()
DRY_RUN = os.getenv("RAWSEQ_IO_DISCOVERY_DRY_RUN", "true").strip().lower() in {"1", "true", "yes", "y", "on"}

TRAIN_ROWS = os.getenv("RAWSEQ_IO_DISCOVERY_TRAIN_ROWS", os.getenv("RAWSEQ_WF_TRAIN_ROWS", "60000"))
VALIDATION_ROWS = os.getenv("RAWSEQ_IO_DISCOVERY_VALIDATION_ROWS", os.getenv("RAWSEQ_WF_VALIDATION_ROWS", "15000"))
TEST_ROWS = os.getenv("RAWSEQ_IO_DISCOVERY_TEST_ROWS", os.getenv("RAWSEQ_WF_TEST_ROWS", "15000"))
STEP_ROWS = os.getenv("RAWSEQ_IO_DISCOVERY_STEP_ROWS", os.getenv("RAWSEQ_WF_STEP_ROWS", TEST_ROWS))
POPULATION = os.getenv("RAWSEQ_IO_DISCOVERY_POPULATION", os.getenv("RAWSEQ_WF_POPULATION", "5"))
GENERATIONS = os.getenv("RAWSEQ_IO_DISCOVERY_GENERATIONS", os.getenv("RAWSEQ_WF_GENERATIONS", "3"))
EPOCHS = os.getenv("RAWSEQ_IO_DISCOVERY_EPOCHS", os.getenv("RAWSEQ_WF_EPOCHS", "35"))
DECISION_HORIZON_SECONDS = os.getenv(
    "RAWSEQ_IO_DISCOVERY_DECISION_HORIZON_SECONDS",
    os.getenv("RAWSEQ_WF_DECISION_HORIZON_SECONDS", "30"),
)
DECISION_THRESHOLD_BPS = os.getenv(
    "RAWSEQ_IO_DISCOVERY_DECISION_THRESHOLD_BPS",
    os.getenv("RAWSEQ_WF_DECISION_THRESHOLD_BPS", "0.0"),
)
FITNESS_POLICY = os.getenv("RAWSEQ_IO_DISCOVERY_FITNESS_POLICY", os.getenv("RAWSEQ_WF_FITNESS_POLICY", "direct_gt"))
FITNESS_THRESHOLD_BPS = os.getenv(
    "RAWSEQ_IO_DISCOVERY_FITNESS_THRESHOLD_BPS",
    os.getenv("RAWSEQ_WF_FITNESS_THRESHOLD_BPS", "0.0"),
)
MIN_FITNESS_TRADES = os.getenv(
    "RAWSEQ_IO_DISCOVERY_MIN_FITNESS_TRADES",
    os.getenv("RAWSEQ_WF_MIN_FITNESS_TRADES", "100"),
)
RUN_UNTIL_SUCCESS = os.getenv("RAWSEQ_DISCOVERY_RUN_UNTIL_SUCCESS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
SUCCESS_MODE = os.getenv("RAWSEQ_DISCOVERY_SUCCESS_MODE", "registered").strip().lower()
if SUCCESS_MODE not in {"registered", "quality_gate"}:
    raise SystemExit("RAWSEQ_DISCOVERY_SUCCESS_MODE must be registered or quality_gate")
RUN_MIN_OK_CANDIDATES = int(float(os.getenv("RAWSEQ_DISCOVERY_MIN_OK_CANDIDATES", "1")))
RUN_MAX_ATTEMPTS = int(float(os.getenv("RAWSEQ_DISCOVERY_MAX_ATTEMPTS", "100")))
RUN_MAX_RUNTIME_SECONDS = float(os.getenv("RAWSEQ_DISCOVERY_MAX_RUNTIME_SECONDS", "7200"))
RANDOMIZE_SEED = os.getenv("RAWSEQ_DISCOVERY_RANDOMIZE_SEED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
START_SEED = int(float(os.getenv("RAWSEQ_DISCOVERY_START_SEED", "900")))
RESUME_DIR_ENV = os.getenv("RAWSEQ_DISCOVERY_RESUME_DIR", "").strip()
QUALITY_MAX_TEST_RMSE = os.getenv("RAWSEQ_DISCOVERY_MAX_TEST_RMSE", "").strip()
QUALITY_MIN_TERMINAL_CORRELATION = os.getenv("RAWSEQ_DISCOVERY_MIN_TERMINAL_CORRELATION", "").strip()
QUALITY_MAX_MONOTONIC_VIOLATION = os.getenv("RAWSEQ_DISCOVERY_MAX_MONOTONIC_VIOLATION", "").strip()
QUALITY_MAX_ENVELOPE_ORDER_VIOLATION = os.getenv("RAWSEQ_DISCOVERY_MAX_ENVELOPE_ORDER_VIOLATION", "").strip()
MIN_MEAN_RMSE_IMPROVEMENT_FRACTION = float(os.getenv("RAWSEQ_DISCOVERY_MIN_MEAN_RMSE_IMPROVEMENT_FRACTION", "0.0"))
RETRYABLE_FAILURES = {"TRAIN_FAILED", "NONFINITE_MODEL", "LABEL_GATE_FAILED", "INSUFFICIENT_ROWS"}
FATAL_FAILURES = {
    "PATH_GUARD_FAILED",
    "METADATA_WRITE_FAILED",
    "ARCHIVE_FAILED",
    "CONFIGURATION_ERROR",
    "OUTPUT_DIM_MISMATCH",
    "LABEL_SHAPE_AUDIT_FAILED",
}
IS_WINDOWS = os.name == "nt"
WINDOWS_SAFE_PATH_LIMIT = 220
MAX_PATH_COMPONENT_LEN = 64
OUTPUT_LABEL_TOKENS = {
    "future_return_path": "frp",
    "future_high_from_now_bps_path": "fhigh",
    "future_low_from_now_bps_path": "flow",
    "future_range_envelope_path": "fenv",
    "barrier_hit_levels": "barrier",
    "tp_before_stop_by_rung": "tps",
}
INPUT_FEATURE_TOKENS = {
    "return": "ret",
    "signed_bucket_return_bps": "sret",
    "ma_distance": "mad",
    "ma_slope": "mas",
    "rolling_volatility_bps": "vol",
    "rolling_range_bps": "rng",
    "distance_to_recent_high_bps": "dh",
    "distance_to_recent_low_bps": "dl",
}


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(text: str | Path) -> Path:
    path = Path(text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def slugify(value: Any, default: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value).strip()).strip("_")
    return text or default


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def limit_component(value: Any, default: str = "item") -> str:
    slug = slugify(value, default)
    if len(slug) <= MAX_PATH_COMPONENT_LEN:
        return slug
    digest = stable_hash(slug)
    keep = max(1, MAX_PATH_COMPONENT_LEN - len(digest) - 1)
    return f"{slug[:keep].rstrip('_')}_{digest}"


def path_length(path: Path) -> int:
    return len(str(path))


def offending_component_length(path: Path) -> int:
    return max((len(part) for part in path.parts), default=0)


def safe_mkdir(path: Path, exist_ok: bool = True) -> None:
    try:
        path.mkdir(parents=True, exist_ok=exist_ok)
    except OSError as exc:
        payload = {
            "error": "mkdir_failed",
            "attempted_path": str(path),
            "attempted_path_length": path_length(path),
            "offending_component_length": offending_component_length(path),
            "exception": str(exc),
        }
        raise SystemExit(json.dumps(payload, indent=2, sort_keys=True)) from exc


def full_contract_slug(row: dict[str, str], index: int) -> str:
    return safe_str(row.get("contract_slug")) or slugify(
        "_".join(
            [
                safe_str(row.get("input_feature")) or "feature",
                f"ma{safe_str(row.get('ma_window'))}" if safe_str(row.get("ma_window")) else f"fw{safe_str(row.get('feature_window')) or 'NA'}",
                f"h{(safe_str(row.get('hidden')) or 'h').replace(',', 'x')}",
                f"is{safe_str(row.get('input_stride')) or '1'}",
                f"os{safe_str(row.get('output_stride')) or '1'}",
                safe_str(row.get("output_label")) or "future_return_path",
                f"idx{index:03d}",
            ]
        ),
        f"contract_{index:03d}",
    )


def filesystem_contract_slug(row: dict[str, str], index: int) -> str:
    feature = INPUT_FEATURE_TOKENS.get(safe_str(row.get("input_feature")), slugify(row.get("input_feature") or "feat")[:8])
    ma = f"ma{safe_str(row.get('ma_window'))}" if safe_str(row.get("ma_window")) else f"fw{safe_str(row.get('feature_window')) or 'NA'}"
    hidden = "h" + (safe_str(row.get("hidden")) or "h").replace(",", "x")
    input_stride = safe_str(row.get("input_stride")) or "1"
    output_stride = safe_str(row.get("output_stride")) or "1"
    output_label = safe_str(row.get("output_label")) or "future_return_path"
    label = OUTPUT_LABEL_TOKENS.get(output_label, slugify(output_label, "out")[:8])
    digest = stable_hash(row)
    return limit_component(f"c{index:03d}_{feature}_{ma}_{hidden}_i{input_stride}_o{output_stride}_{label}_{digest}", f"c{index:03d}")


def contract_path_info(
    batch_dir: Path,
    row: dict[str, str],
    index: int,
    seed_override: str = "",
    attempt_index: int = 0,
) -> dict[str, Any]:
    full_slug = full_contract_slug(row, index)
    fs_slug = filesystem_contract_slug(row, index)
    contract_dir = batch_dir / "contracts" / fs_slug
    shortened = fs_slug != full_slug
    if IS_WINDOWS and path_length(contract_dir) > WINDOWS_SAFE_PATH_LIMIT:
        output_label = safe_str(row.get("output_label")) or "future_return_path"
        label = OUTPUT_LABEL_TOKENS.get(output_label, "out")
        fs_slug = limit_component(f"c{index:03d}_{label}_{stable_hash(row)}", f"c{index:03d}")
        contract_dir = batch_dir / "contracts" / fs_slug
        shortened = True
    digest = stable_hash(row)
    seed = seed_override or (parse_seed_limit(SEEDS).split(",")[0] if parse_seed_limit(SEEDS) else "900")
    if seed_override or attempt_index:
        wf_run_id = limit_component(f"wf_{digest}_a{attempt_index:03d}_s{seed}", "wf")
    else:
        wf_run_id = limit_component(f"wf_{digest}", "wf")
    output_label = safe_str(row.get("output_label")) or "future_return_path"
    archive_label = OUTPUT_LABEL_TOKENS.get(output_label, "out")
    candidate_dir = contract_dir / "walkforward" / wf_run_id / "window_000" / limit_component(
        f"s{seed}_{stable_hash({'row': row, 'seed': seed, 'attempt_index': attempt_index})}",
        "candidate",
    )
    final_paths = [
        candidate_dir,
        candidate_dir / "contract.json",
        candidate_dir / "model_contract.json",
        candidate_dir / "selected_candidate_summary.json",
        candidate_dir.parent.parent / "candidates.csv",
        candidate_dir / "trainer_artifacts" / "rawseq_label_metric_summary.csv",
    ]
    temp_paths = [path.parent / ".t_12345678" for path in final_paths if path.suffix in {".json", ".csv"}]
    longest_artifact = max(final_paths, key=path_length)
    longest_temp = max(temp_paths, key=path_length)
    return {
        "contract_dir": contract_dir,
        "full_contract_slug": full_slug,
        "filesystem_contract_slug": fs_slug,
        "wf_run_id": wf_run_id,
        "filesystem_path_shortened": bool(shortened),
        "projected_path_length": path_length(contract_dir),
        "projected_candidate_dir_length": path_length(candidate_dir),
        "projected_longest_artifact_path_length": path_length(longest_artifact),
        "projected_longest_temp_path_length": path_length(longest_temp),
        "projected_longest_artifact_path": str(longest_artifact),
        "projected_longest_temp_path": str(longest_temp),
        "windows_path_guard_pass": (not IS_WINDOWS) or path_length(longest_temp) < 240,
        "offending_component_length": offending_component_length(contract_dir),
    }


def latest_grid_path() -> Path:
    if GRID_PATH_ENV:
        return resolve_path(GRID_PATH_ENV)
    if not GRID_ROOT.exists():
        raise SystemExit(f"Grid root not found: {GRID_ROOT}")
    candidates = sorted(
        GRID_ROOT.glob("*/rawseq_io_contract_grid.csv"),
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"No rawseq_io_contract_grid.csv files found under {GRID_ROOT}")
    return candidates[0]


def read_grid(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    supported = [row for row in rows if safe_str(row.get("support_status")).lower() == "supported"]
    return supported[: max(0, MAX_CONTRACTS)]


def parse_seed_limit(seeds_text: str) -> str:
    seeds = [item.strip() for item in seeds_text.split(",") if item.strip()]
    if not seeds:
        seeds = ["900"]
    if MAX_CANDIDATES > 0:
        seeds = seeds[:MAX_CANDIDATES]
    return ",".join(seeds)


def output_root() -> Path:
    root = resolve_path(OUTPUT_ROOT_ENV)
    safe_mkdir(root)
    path = root / f"rawseq_io_contract_discovery_{now_stamp()}_{uuid.uuid4().hex[:8]}"
    if IS_WINDOWS and path_length(path) > WINDOWS_SAFE_PATH_LIMIT:
        path = root / f"io_disc_{now_stamp()}_{uuid.uuid4().hex[:8]}"
    safe_mkdir(path, exist_ok=False)
    return path


def short_contract_dir_name(row: dict[str, str], index: int) -> str:
    return filesystem_contract_slug(row, index)


def run_contract(
    row: dict[str, str],
    batch_dir: Path,
    index: int,
    started_at: float,
    seed_override: str = "",
    attempt_index: int = 0,
) -> dict[str, Any]:
    path_info = contract_path_info(batch_dir, row, index, seed_override, attempt_index)
    contract_slug = path_info["full_contract_slug"]
    contract_dir = path_info["contract_dir"]
    wf_output_root = contract_dir / "walkforward"
    wf_run_id = path_info["wf_run_id"]
    contract_record = {
        **row,
        "full_contract_slug": path_info["full_contract_slug"],
        "filesystem_contract_slug": path_info["filesystem_contract_slug"],
        "filesystem_path_shortened": path_info["filesystem_path_shortened"],
        "projected_path_length": path_info["projected_path_length"],
        "projected_candidate_dir_length": path_info["projected_candidate_dir_length"],
        "projected_longest_artifact_path_length": path_info["projected_longest_artifact_path_length"],
        "projected_longest_temp_path_length": path_info["projected_longest_temp_path_length"],
        "windows_path_guard_pass": path_info["windows_path_guard_pass"],
        "attempt_index": attempt_index,
        "attempt_seed": seed_override or parse_seed_limit(SEEDS),
        "requested_population": POPULATION,
        "requested_generations": GENERATIONS,
        "requested_epochs": EPOCHS,
    }
    safe_mkdir(contract_dir)
    (contract_dir / "contract.json").write_text(json.dumps(contract_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if DRY_RUN:
        wf_run_dir = wf_output_root / wf_run_id
        safe_mkdir(wf_run_dir)
        dry_payload = {
            "dry_run": True,
            "contract_slug": contract_slug,
            "max_windows": max(1, MAX_WINDOWS),
            "seeds": seed_override or parse_seed_limit(SEEDS),
            "public_recorded_data_only": True,
            "private_api": False,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
        }
        (wf_run_dir / "dry_run_manifest.json").write_text(json.dumps(dry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_path = contract_dir / (f"walkforward_stdout_a{attempt_index:03d}_s{seed_override}.log" if attempt_index else "walkforward_stdout.log")
        log_path.write_text("DRY_RUN: walk-forward subprocess not executed by discovery batch.\n", encoding="utf-8")
        summary = base_summary(contract_record, contract_dir, "DRY_RUN", "", time.monotonic() - started_at, 0.0, str(log_path))
        summary["walkforward_run_dir"] = str(wf_run_dir)
        summary["candidate_rows"] = max(1, MAX_WINDOWS) * len((seed_override or parse_seed_limit(SEEDS)).split(","))
        return summary

    elapsed = time.monotonic() - started_at
    remaining = MAX_RUNTIME_SECONDS - elapsed
    if remaining <= 0:
        return base_summary(contract_record, contract_dir, "SKIPPED_MAX_RUNTIME", "", elapsed, 0.0, "")

    env = os.environ.copy()
    env.update(
        {
            "SYMBOL": safe_str(row.get("symbol")) or "SOLUSDT",
            "PRIMARY_VENUE": safe_str(row.get("venue")) or "kraken",
            "RAWSEQ_WF_SOURCE_PATH": safe_str(row.get("source_path")),
            "RAWSEQ_WF_OUTPUT_ROOT": str(wf_output_root),
            "RAWSEQ_WF_RUN_ID": wf_run_id,
            "RAWSEQ_WF_SHORT_PATHS": "true" if IS_WINDOWS else os.getenv("RAWSEQ_WF_SHORT_PATHS", "false"),
            "RAWSEQ_WF_INPUT_FEATURES": safe_str(row.get("input_feature")),
            "RAWSEQ_WF_MA_WINDOWS": safe_str(row.get("ma_window")) or "60",
            "RAWSEQ_WF_FEATURE_WINDOWS": safe_str(row.get("feature_window")) or safe_str(row.get("ma_window")) or "60",
            "RAWSEQ_WF_HIDDENS": safe_str(row.get("hidden")),
            "RAWSEQ_WF_SEEDS": seed_override or parse_seed_limit(SEEDS),
            "RAWSEQ_WF_INPUT_STRIDES": safe_str(row.get("input_stride")) or "1",
            "RAWSEQ_WF_OUTPUT_STRIDES": safe_str(row.get("output_stride")) or "1",
            "RAWSEQ_WF_OUTPUT_LABELS": safe_str(row.get("output_label")) or "future_return_path",
            "RAWSEQ_WF_BUCKET_SECONDS": safe_str(row.get("bucket_seconds")) or "10",
            "RAWSEQ_WF_SEQ_LEN": safe_str(row.get("seq_len")) or "60",
            "RAWSEQ_WF_MAX_WINDOWS": str(max(1, MAX_WINDOWS)),
            "RAWSEQ_WF_TRAIN_ROWS": TRAIN_ROWS,
            "RAWSEQ_WF_VALIDATION_ROWS": VALIDATION_ROWS,
            "RAWSEQ_WF_TEST_ROWS": TEST_ROWS,
            "RAWSEQ_WF_STEP_ROWS": STEP_ROWS,
            "RAWSEQ_WF_POPULATION": POPULATION,
            "RAWSEQ_WF_GENERATIONS": GENERATIONS,
            "RAWSEQ_WF_EPOCHS": EPOCHS,
            "RAWSEQ_POPULATION": POPULATION,
            "RAWSEQ_GENERATIONS": GENERATIONS,
            "RAWSEQ_EPOCHS": EPOCHS,
            "RAWSEQ_WF_DECISION_HORIZON_SECONDS": DECISION_HORIZON_SECONDS,
            "RAWSEQ_WF_DECISION_THRESHOLD_BPS": DECISION_THRESHOLD_BPS,
            "RAWSEQ_WF_FITNESS_POLICY": FITNESS_POLICY,
            "RAWSEQ_WF_FITNESS_THRESHOLD_BPS": FITNESS_THRESHOLD_BPS,
            "RAWSEQ_WF_MIN_FITNESS_TRADES": MIN_FITNESS_TRADES,
            "RAWSEQ_WF_DRY_RUN": "true" if DRY_RUN else "false",
            "PROMOTE_BEST": "false",
            "PRICE_TINY_AUTO_REGISTER_CHALLENGERS": "false",
            "TRAIN_PRICE_TINY_MODEL": "false",
        }
    )

    run_started = time.monotonic()
    log_path = contract_dir / (f"walkforward_stdout_a{attempt_index:03d}_s{seed_override}.log" if attempt_index else "walkforward_stdout.log")
    try:
        completed = subprocess.run(
            [sys.executable, str(WF_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1.0, remaining),
        )
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        runtime = time.monotonic() - run_started
        status = "DRY_RUN" if DRY_RUN and completed.returncode == 0 else "OK" if completed.returncode == 0 else "TRAIN_FAILED"
        return summarize_contract(contract_record, contract_dir, wf_output_root / wf_run_id, status, completed.returncode, runtime, str(log_path))
    except subprocess.TimeoutExpired as exc:
        log_path.write_text((exc.stdout or "") if isinstance(exc.stdout, str) else "", encoding="utf-8")
        runtime = time.monotonic() - run_started
        return base_summary(contract_record, contract_dir, "TIMEOUT", "timeout", time.monotonic() - started_at, runtime, str(log_path))


def base_summary(
    row: dict[str, str],
    contract_dir: Path,
    status: str,
    exit_code: Any,
    elapsed_batch_seconds: float,
    runtime_seconds: float,
    log_path: str,
) -> dict[str, Any]:
    return {
        "contract_slug": safe_str(row.get("contract_slug")),
        "full_contract_slug": safe_str(row.get("full_contract_slug") or row.get("contract_slug")),
        "filesystem_contract_slug": safe_str(row.get("filesystem_contract_slug")),
        "filesystem_path_shortened": safe_str(row.get("filesystem_path_shortened")),
        "projected_path_length": safe_str(row.get("projected_path_length")),
        "projected_candidate_dir_length": safe_str(row.get("projected_candidate_dir_length")),
        "projected_longest_artifact_path_length": safe_str(row.get("projected_longest_artifact_path_length")),
        "projected_longest_temp_path_length": safe_str(row.get("projected_longest_temp_path_length")),
        "actual_longest_artifact_path_length": safe_str(row.get("actual_longest_artifact_path_length")),
        "actual_longest_temp_path_length": safe_str(row.get("actual_longest_temp_path_length")),
        "windows_path_guard_pass": safe_str(row.get("windows_path_guard_pass")),
        "status": status,
        "exit_code": exit_code,
        "contract_dir": str(contract_dir),
        "walkforward_run_dir": "",
        "runtime_seconds": runtime_seconds,
        "elapsed_batch_seconds": elapsed_batch_seconds,
        "symbol": safe_str(row.get("symbol")),
        "venue": safe_str(row.get("venue")),
        "source_path": safe_str(row.get("source_path")),
        "input_feature": safe_str(row.get("input_feature")),
        "ma_window": safe_str(row.get("ma_window")),
        "feature_window": safe_str(row.get("feature_window")),
        "hidden": safe_str(row.get("hidden")),
        "bucket_seconds": safe_str(row.get("bucket_seconds")),
        "seq_len": safe_str(row.get("seq_len")),
        "input_stride": safe_str(row.get("input_stride")),
        "output_stride": safe_str(row.get("output_stride")),
        "output_label": safe_str(row.get("output_label")),
        "output_dim": safe_str(row.get("output_dim")),
        "output_orientation": safe_str(row.get("output_orientation")),
        "requested_population": safe_str(row.get("requested_population")),
        "requested_generations": safe_str(row.get("requested_generations")),
        "requested_epochs": safe_str(row.get("requested_epochs")),
        "input_window_seconds": safe_str(row.get("input_window_seconds")),
        "output_window_seconds": safe_str(row.get("output_window_seconds")),
        "candidate_rows": 0,
        "ok_candidate_rows": 0,
        "best_test_cumulative_return_bps": math.nan,
        "best_test_avg_return_bps": math.nan,
        "best_test_rows": 0,
        "best_contract_slug": "",
        "leaderboard_rows": 0,
        "log_path": log_path,
        "dry_run": DRY_RUN,
        "public_recorded_data_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }


def summarize_contract(
    row: dict[str, str],
    contract_dir: Path,
    wf_run_dir: Path,
    status: str,
    exit_code: Any,
    runtime_seconds: float,
    log_path: str,
) -> dict[str, Any]:
    summary = base_summary(row, contract_dir, status, exit_code, runtime_seconds, runtime_seconds, log_path)
    summary["walkforward_run_dir"] = str(wf_run_dir)
    candidates_path = wf_run_dir / "candidates.csv"
    leaderboard_path = wf_run_dir / "contract_leaderboard.csv"
    if candidates_path.exists():
        candidates = pd.read_csv(candidates_path, low_memory=False)
        summary["candidate_rows"] = int(len(candidates))
        if "status" in candidates.columns:
            summary["ok_candidate_rows"] = int(candidates["status"].astype(str).eq("OK").sum())
        for column in [
            "projected_candidate_dir_length",
            "projected_longest_artifact_path_length",
            "projected_longest_temp_path_length",
            "actual_longest_artifact_path_length",
            "actual_longest_temp_path_length",
        ]:
            if column in candidates.columns:
                summary[column] = safe_str(pd.to_numeric(candidates[column], errors="coerce").max())
        if "windows_path_guard_pass" in candidates.columns:
            summary["windows_path_guard_pass"] = bool(candidates["windows_path_guard_pass"].astype(str).str.lower().isin(["true", "1"]).all())
        for column in ["best_test_cumulative_return_bps", "best_test_avg_return_bps", "best_test_rows"]:
            if column in candidates.columns:
                candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
        if "best_test_cumulative_return_bps" in candidates.columns and candidates["best_test_cumulative_return_bps"].notna().any():
            best = candidates.sort_values("best_test_cumulative_return_bps", ascending=False).iloc[0]
            summary["best_test_cumulative_return_bps"] = safe_float(best.get("best_test_cumulative_return_bps"))
            summary["best_test_avg_return_bps"] = safe_float(best.get("best_test_avg_return_bps"))
            summary["best_test_rows"] = int(safe_float(best.get("best_test_rows"), 0.0))
            summary["best_contract_slug"] = safe_str(best.get("contract_slug"))
    if leaderboard_path.exists():
        try:
            leaderboard = pd.read_csv(leaderboard_path, low_memory=False)
            summary["leaderboard_rows"] = int(len(leaderboard))
        except pd.errors.EmptyDataError:
            summary["leaderboard_rows"] = 0
    return summary


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def latest_candidate(summary: dict[str, Any]) -> dict[str, Any]:
    wf_dir = Path(safe_str(summary.get("walkforward_run_dir")))
    candidates = read_csv_or_empty(wf_dir / "candidates.csv")
    if candidates.empty:
        return {}
    return candidates.iloc[-1].to_dict()


def classify_failure(summary: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, str]:
    status = safe_str(candidate.get("status")) or safe_str(summary.get("status"))
    archive_dir = Path(safe_str(candidate.get("archive_dir"))) if safe_str(candidate.get("archive_dir")) else None
    message = ""
    if archive_dir is not None:
        for name in ["metadata_write_error.txt", "run.log"]:
            path = archive_dir / name
            if path.exists():
                try:
                    message += "\n" + path.read_text(encoding="utf-8", errors="replace")[-4000:]
                except Exception:
                    pass
    lower = message.lower()
    if safe_str(candidate.get("windows_path_guard_pass")).lower() in {"false", "0"}:
        return "PATH_GUARD_FAILED", "Windows path guard failed."
    if status == "METADATA_WRITE_FAILED":
        return "METADATA_WRITE_FAILED", message.strip() or "Metadata write failed."
    if "output dim mismatch" in lower or "output_dim" in lower and "mismatch" in lower:
        return "OUTPUT_DIM_MISMATCH", message.strip()
    if "label_shape_audit_failed" in lower or "shape audit" in lower and "fail" in lower:
        return "LABEL_SHAPE_AUDIT_FAILED", message.strip()
    if "no rawseq rows built" in lower or "no walk-forward windows" in lower:
        return "INSUFFICIENT_ROWS", message.strip()
    if "nonfinite" in lower or "non-finite" in lower:
        return "NONFINITE_MODEL", message.strip()
    if status and status != "OK":
        return status, message.strip()
    return "", message.strip()


def registered_success(summary: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
    if safe_str(summary.get("status")) != "OK":
        return False, f"summary_status={summary.get('status')}"
    if safe_str(candidate.get("status")) != "OK":
        return False, f"candidate_status={candidate.get('status')}"
    archive_dir = Path(safe_str(candidate.get("archive_dir")))
    missing = [
        str(path)
        for path in [archive_dir / "model_contract.json", archive_dir / "model.json"]
        if not path.exists()
    ]
    if missing:
        return False, "missing_registered_artifacts=" + ";".join(missing)
    return True, ""


def optional_float(text: str) -> float:
    return safe_float(text) if safe_str(text) else math.nan


def safe_bool(value: Any) -> bool:
    return safe_str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def combined_label_rmse(row: Any) -> float:
    combined = safe_float(row.get("combined_path_rmse"))
    if math.isfinite(combined):
        return combined
    high = safe_float(row.get("high_path_rmse"))
    low = safe_float(row.get("low_path_rmse"))
    path = safe_float(row.get("path_rmse"))
    if math.isfinite(high) and math.isfinite(low):
        return 0.5 * (high + low)
    if math.isfinite(high):
        return high
    if math.isfinite(low):
        return low
    return path


def combined_label_mae(row: Any) -> float:
    high = safe_float(row.get("high_path_mae"))
    low = safe_float(row.get("low_path_mae"))
    path = safe_float(row.get("path_mae"))
    if math.isfinite(high) and math.isfinite(low):
        return 0.5 * (high + low)
    if math.isfinite(high):
        return high
    if math.isfinite(low):
        return low
    return path


def metric_baseline_guard(test_metrics: pd.DataFrame, model_row: Any) -> dict[str, Any]:
    model_rmse = combined_label_rmse(model_row)
    mean_rows = test_metrics[test_metrics["strategy"].astype(str).eq("training_mean_path_baseline")] if "strategy" in test_metrics.columns else pd.DataFrame()
    median_rows = test_metrics[test_metrics["strategy"].astype(str).eq("training_median_path_baseline")] if "strategy" in test_metrics.columns else pd.DataFrame()
    mean_rmse = combined_label_rmse(mean_rows.iloc[0]) if not mean_rows.empty else math.nan
    median_rmse = combined_label_rmse(median_rows.iloc[0]) if not median_rows.empty else math.nan
    model_mae = combined_label_mae(model_row)
    median_mae = combined_label_mae(median_rows.iloc[0]) if not median_rows.empty else math.nan
    mean_rmse_improvement = (mean_rmse - model_rmse) / mean_rmse if math.isfinite(model_rmse) and math.isfinite(mean_rmse) and mean_rmse > 1e-12 else math.nan
    median_mae_improvement = (median_mae - model_mae) / median_mae if math.isfinite(model_mae) and math.isfinite(median_mae) and median_mae > 1e-12 else math.nan
    pass_guard = bool(math.isfinite(mean_rmse_improvement) and mean_rmse_improvement >= MIN_MEAN_RMSE_IMPROVEMENT_FRACTION)
    if pass_guard:
        reason = f"mean_rmse_improvement_fraction>={MIN_MEAN_RMSE_IMPROVEMENT_FRACTION:g}"
    elif not math.isfinite(mean_rmse_improvement):
        reason = "mean_rmse_improvement_fraction_not_finite"
    else:
        reason = f"mean_rmse_improvement_fraction<{MIN_MEAN_RMSE_IMPROVEMENT_FRACTION:g}"
    return {
        "rmse_guard_pass": pass_guard,
        "rmse_guard_reason": reason,
        "model_rmse": model_rmse,
        "mean_rmse": mean_rmse,
        "median_rmse": median_rmse,
        "model_mae": model_mae,
        "median_mae": median_mae,
        "model_vs_mean_rmse_improvement_fraction": mean_rmse_improvement,
        "model_vs_median_mae_improvement_fraction": median_mae_improvement,
        "mae_baseline_diagnostic_pass": bool(math.isfinite(median_mae_improvement) and median_mae_improvement > 0.0),
    }


def quality_gate_success(candidate: dict[str, Any]) -> tuple[bool, str]:
    archive_dir = Path(safe_str(candidate.get("archive_dir")))
    metrics = read_csv_or_empty(archive_dir / "label_metric_summary.csv")
    audit = read_csv_or_empty(archive_dir / "label_shape_audit.csv")
    if audit.empty or not audit["status"].astype(str).str.upper().eq("PASS").all():
        return False, "LABEL_SHAPE_AUDIT_FAILED"
    if metrics.empty:
        return False, "LABEL_GATE_FAILED:no label_metric_summary.csv rows"
    test = metrics[metrics.get("split", pd.Series("", index=metrics.index)).astype(str).str.contains("test", case=False, na=False)].copy()
    if test.empty:
        test = metrics.tail(1).copy()
    if "strategy" in test.columns:
        model_test = test[test["strategy"].astype(str).eq("label_metric_summary")].copy()
        if not model_test.empty:
            test = model_test
    row = test.iloc[0]
    output_label = safe_str(row.get("output_label"))
    rmse = safe_float(row.get("path_rmse"))
    if output_label == "future_range_envelope_path":
        rmse = 0.5 * (safe_float(row.get("high_path_rmse")) + safe_float(row.get("low_path_rmse")))
    corr_values = [safe_float(row.get("terminal_high_correlation")), safe_float(row.get("terminal_low_correlation"))]
    corr_values = [value for value in corr_values if math.isfinite(value)]
    corr = sum(corr_values) / len(corr_values) if corr_values else math.nan
    mono = safe_float(row.get("monotonic_violation_fraction"))
    if output_label == "future_range_envelope_path":
        mono = 0.5 * (
            safe_float(row.get("high_monotonic_violation_fraction"))
            + safe_float(row.get("low_monotonic_violation_fraction"))
        )
    order = safe_float(row.get("envelope_order_violation_fraction"))
    failures = []
    max_rmse = optional_float(QUALITY_MAX_TEST_RMSE)
    min_corr = optional_float(QUALITY_MIN_TERMINAL_CORRELATION)
    max_mono = optional_float(QUALITY_MAX_MONOTONIC_VIOLATION)
    max_order = optional_float(QUALITY_MAX_ENVELOPE_ORDER_VIOLATION)
    if math.isfinite(max_rmse) and (not math.isfinite(rmse) or rmse > max_rmse):
        failures.append(f"rmse={rmse} > {max_rmse}")
    if math.isfinite(min_corr) and (not math.isfinite(corr) or corr < min_corr):
        failures.append(f"terminal_correlation={corr} < {min_corr}")
    if math.isfinite(max_mono) and (not math.isfinite(mono) or mono > max_mono):
        failures.append(f"monotonic_violation={mono} > {max_mono}")
    if math.isfinite(max_order) and output_label == "future_range_envelope_path" and (not math.isfinite(order) or order > max_order):
        failures.append(f"envelope_order_violation={order} > {max_order}")
    if output_label and output_label != "future_return_path":
        guard = metric_baseline_guard(
            metrics[metrics.get("split", pd.Series("", index=metrics.index)).astype(str).str.contains("test", case=False, na=False)].copy(),
            row,
        )
        if not guard["rmse_guard_pass"]:
            failures.append(
                "baseline_guard_failed:"
                f"{guard['rmse_guard_reason']}; "
                f"model_rmse={guard['model_rmse']}; "
                f"mean_rmse={guard['mean_rmse']}; "
                f"median_rmse={guard['median_rmse']}; "
                f"model_vs_mean_rmse_improvement_fraction={guard['model_vs_mean_rmse_improvement_fraction']}"
            )
    return (not failures), "; ".join(failures)


def seed_sequence(completed: set[int]):
    if RANDOMIZE_SEED:
        rng = random.Random(START_SEED)
        while True:
            seed = rng.randint(1, 2_147_483_000)
            if seed not in completed:
                yield seed
    else:
        seed = START_SEED
        while True:
            if seed not in completed:
                yield seed
            seed += 1


def write_run_until_state(batch_dir: Path, attempts: list[dict[str, Any]], started: float, status: str, stop_reason: str, current_seed: int | None) -> None:
    attempts_frame = pd.DataFrame(attempts)
    attempts_frame.to_csv(batch_dir / "attempts.csv", index=False)
    if not attempts_frame.empty:
        attempts_frame[attempts_frame["attempt_success"].astype(bool)].to_csv(batch_dir / "successful_candidates.csv", index=False)
        attempts_frame[~attempts_frame["attempt_success"].astype(bool)].to_csv(batch_dir / "failed_attempts.csv", index=False)
        completed = [str(int(seed)) for seed in attempts_frame["seed"].dropna().astype(int).tolist()]
    else:
        pd.DataFrame().to_csv(batch_dir / "successful_candidates.csv", index=False)
        pd.DataFrame().to_csv(batch_dir / "failed_attempts.csv", index=False)
        completed = []
    (batch_dir / "completed_seeds.txt").write_text("\n".join(completed) + ("\n" if completed else ""), encoding="utf-8")
    state = {
        "status": status,
        "attempts_completed": len(attempts),
        "ok_candidates": int(sum(1 for row in attempts if row.get("registered_success"))),
        "quality_gate_candidates": int(sum(1 for row in attempts if row.get("quality_gate_success"))),
        "current_seed": current_seed,
        "elapsed_seconds": time.monotonic() - started,
        "stop_reason": stop_reason,
        "resumable": status not in {"complete", "fatal"},
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    (batch_dir / "run_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_completed_seeds(batch_dir: Path) -> set[int]:
    seeds: set[int] = set()
    path = batch_dir / "completed_seeds.txt"
    if not path.exists():
        return seeds
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            seeds.add(int(line.strip()))
        except Exception:
            pass
    return seeds


def load_attempts(batch_dir: Path) -> list[dict[str, Any]]:
    path = batch_dir / "attempts.csv"
    if not path.exists():
        return []
    frame = read_csv_or_empty(path)
    return frame.to_dict(orient="records") if not frame.empty else []


def run_until_success(batch_dir: Path, contracts: list[dict[str, str]], grid_path: Path, started: float) -> list[dict[str, Any]]:
    attempts = load_attempts(batch_dir) if RESUME_DIR_ENV else []
    completed_seeds = load_completed_seeds(batch_dir)
    seed_iter = seed_sequence(completed_seeds)
    stop_reason = ""
    status = "running"
    current_seed: int | None = None
    write_run_until_state(batch_dir, attempts, started, status, stop_reason, current_seed)

    try:
        while True:
            ok_count = sum(1 for row in attempts if row.get("registered_success"))
            quality_count = sum(1 for row in attempts if row.get("quality_gate_success"))
            target_count = quality_count if SUCCESS_MODE == "quality_gate" else ok_count
            if target_count >= RUN_MIN_OK_CANDIDATES:
                status = "complete"
                stop_reason = "min_ok_candidates_reached"
                break
            if len(attempts) >= RUN_MAX_ATTEMPTS:
                status = "stopped"
                stop_reason = "max_attempts_reached"
                break
            elapsed = time.monotonic() - started
            if elapsed >= RUN_MAX_RUNTIME_SECONDS:
                status = "stopped"
                stop_reason = "max_runtime_reached"
                break

            current_seed = next(seed_iter)
            completed_seeds.add(current_seed)
            contract_index = (len(attempts) % len(contracts)) + 1
            row = contracts[contract_index - 1]
            attempt_number = len(attempts) + 1
            print(f"[attempt {attempt_number}] seed={current_seed} contract={safe_str(row.get('contract_slug'))}")
            summary = run_contract(row, batch_dir, contract_index, started, str(current_seed), attempt_number)
            candidate = latest_candidate(summary)
            reg_ok, reg_reason = registered_success(summary, candidate)
            quality_ok = False
            quality_reason = ""
            failure_class, failure_message = classify_failure(summary, candidate)
            if reg_ok:
                if SUCCESS_MODE == "quality_gate":
                    quality_ok, quality_reason = quality_gate_success(candidate)
                    if not quality_ok:
                        failure_class = "LABEL_GATE_FAILED" if quality_reason != "LABEL_SHAPE_AUDIT_FAILED" else "LABEL_SHAPE_AUDIT_FAILED"
                        failure_message = quality_reason
                else:
                    quality_ok = False
                if not failure_class:
                    failure_class = ""
            else:
                failure_message = failure_message or reg_reason
            attempt_success = reg_ok if SUCCESS_MODE == "registered" else (reg_ok and quality_ok)
            archive_dir = safe_str(candidate.get("archive_dir"))
            attempt_row = {
                **summary,
                "attempt_number": attempt_number,
                "seed": current_seed,
                "registered_success": bool(reg_ok),
                "quality_gate_success": bool(quality_ok),
                "attempt_success": bool(attempt_success),
                "failure_class": "" if attempt_success else (failure_class or "TRAIN_FAILED"),
                "failure_message": "" if attempt_success else failure_message,
                "model_path": safe_str(candidate.get("model_path")),
                "archived_model_path": safe_str(candidate.get("archived_model_path")),
                "archive_dir": archive_dir,
                "candidate_status": safe_str(candidate.get("status")),
                "output_label": safe_str(candidate.get("output_label") or summary.get("output_label")),
                "output_orientation": safe_str(
                    candidate.get("output_orientation")
                    or candidate.get("payload_output_orientation")
                    or summary.get("output_orientation")
                ),
                "output_dim": safe_str(candidate.get("payload_output_dim") or summary.get("output_dim")),
                "feature_window": safe_str(candidate.get("feature_window") or summary.get("feature_window")),
                "requested_feature_window": safe_str(candidate.get("requested_feature_window")),
                "resolved_feature_window": safe_str(candidate.get("resolved_feature_window")),
                "payload_feature_window": safe_str(candidate.get("payload_feature_window")),
                "requested_population": POPULATION,
                "requested_generations": GENERATIONS,
                "requested_epochs": EPOCHS,
                "resolved_population": safe_str(candidate.get("resolved_population")),
                "resolved_generations": safe_str(candidate.get("resolved_generations")),
                "resolved_epochs": safe_str(candidate.get("resolved_epochs")),
                "label_rank_score": safe_str(candidate.get("label_rank_score")),
                "unguarded_label_rank_score": safe_str(candidate.get("unguarded_label_rank_score")),
                "guarded_label_rank_score": safe_str(candidate.get("guarded_label_rank_score")),
                "baseline_guard_pass": safe_str(candidate.get("baseline_guard_pass")),
                "baseline_guard_reason": safe_str(candidate.get("baseline_guard_reason")),
                "rmse_guard_pass": safe_str(candidate.get("rmse_guard_pass")),
                "rmse_guard_reason": safe_str(candidate.get("rmse_guard_reason")),
                "mae_baseline_diagnostic_pass": safe_str(candidate.get("mae_baseline_diagnostic_pass")),
                "model_combined_test_rmse": safe_str(candidate.get("model_combined_test_rmse")),
                "mean_baseline_combined_test_rmse": safe_str(candidate.get("mean_baseline_combined_test_rmse")),
                "median_baseline_combined_test_rmse": safe_str(candidate.get("median_baseline_combined_test_rmse")),
                "model_combined_test_mae": safe_str(candidate.get("model_combined_test_mae")),
                "median_baseline_combined_test_mae": safe_str(candidate.get("median_baseline_combined_test_mae")),
                "model_vs_mean_rmse_improvement_fraction": safe_str(candidate.get("test_model_vs_mean_rmse_improvement_fraction")),
                "model_vs_median_rmse_improvement_fraction": safe_str(candidate.get("test_model_vs_median_rmse_improvement_fraction")),
                "model_vs_median_mae_improvement_fraction": safe_str(candidate.get("test_model_vs_median_mae_improvement_fraction")),
                "model_vs_zero_rmse_improvement_fraction": safe_str(candidate.get("test_model_vs_zero_rmse_improvement_fraction")),
                "model_beats_mean_baseline": safe_str(candidate.get("test_model_beats_mean_baseline")),
                "model_beats_median_baseline": safe_str(candidate.get("test_model_beats_median_baseline")),
                "model_beats_median_mae_baseline": safe_str(candidate.get("test_model_beats_median_mae_baseline")),
                "paper_only": True,
                "orders": False,
                "promotion": False,
                "champion_mutation": False,
            }
            attempts.append(attempt_row)
            if not attempt_success and attempt_row["failure_class"] in FATAL_FAILURES:
                status = "fatal"
                stop_reason = attempt_row["failure_class"]
                write_run_until_state(batch_dir, attempts, started, status, stop_reason, current_seed)
                break
            write_run_until_state(batch_dir, attempts, started, "running", "", current_seed)
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "user_interruption"
    finally:
        write_run_until_state(batch_dir, attempts, started, status, stop_reason, current_seed)

    summary_txt = batch_dir / "io_contract_discovery_summary.txt"
    summary_csv = batch_dir / "io_contract_discovery_summary.csv"
    pd.DataFrame(attempts).to_csv(summary_csv, index=False)
    text = render_summary(batch_dir, grid_path, attempts, started)
    text += (
        "\nRun Until Success\n"
        f"  enabled: true\n"
        f"  success_mode: {SUCCESS_MODE}\n"
        f"  status: {status}\n"
        f"  stop_reason: {stop_reason}\n"
        f"  attempts: {len(attempts)}\n"
        f"  registered_successes: {sum(1 for row in attempts if row.get('registered_success'))}\n"
        f"  quality_gate_successes: {sum(1 for row in attempts if row.get('quality_gate_success'))}\n"
    )
    summary_txt.write_text(text, encoding="utf-8")
    return attempts


def render_summary(batch_dir: Path, grid_path: Path, rows: list[dict[str, Any]], started: float) -> str:
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    lines = [
        "Rawseq I/O Contract Discovery Batch",
        "",
        f"Created at: {now_stamp()}",
        f"Grid path: {grid_path}",
        f"Batch dir: {batch_dir}",
        f"Dry run: {DRY_RUN}",
        f"Max contracts: {MAX_CONTRACTS}",
        f"Max candidates/seeds per contract: {MAX_CANDIDATES}",
        f"Max windows per contract: {MAX_WINDOWS}",
        f"Max runtime seconds: {MAX_RUNTIME_SECONDS:g}",
        f"Elapsed seconds: {time.monotonic() - started:.3f}",
        "",
        "Safety:",
        "  public_recorded_data_only=true",
        "  private_api=false",
        "  orders=false",
        "  promotion=false",
        "  champion_mutation=false",
        "",
        "Status counts:",
    ]
    for key in sorted(status_counts):
        lines.append(f"  {key}: {status_counts[key]}")
    lines.append("")
    lines.append("Contracts:")
    for row in rows:
        lines.append(
            "  "
            f"{row['status']} {row['contract_slug']} "
            f"candidates={row['candidate_rows']} ok={row['ok_candidate_rows']} "
            f"best_cum={row['best_test_cumulative_return_bps']} dir={row['contract_dir']}"
        )
    lines.append("")
    lines.append("Warning: no private API, no orders, no promotion, no champion mutation.")
    return "\n".join(lines) + "\n"


def main() -> int:
    if not WF_SCRIPT.exists():
        raise SystemExit(f"Missing walk-forward script: {WF_SCRIPT}")
    grid_path = latest_grid_path()
    contracts = read_grid(grid_path)
    if not contracts:
        raise SystemExit(f"No supported contracts selected from {grid_path}")
    if RESUME_DIR_ENV:
        batch_dir = resolve_path(RESUME_DIR_ENV)
        if not batch_dir.exists():
            raise SystemExit(f"RAWSEQ_DISCOVERY_RESUME_DIR does not exist: {batch_dir}")
    else:
        batch_dir = output_root()
    started = time.monotonic()
    if RUN_UNTIL_SUCCESS:
        rows = run_until_success(batch_dir, contracts, grid_path, started)
        metadata = {
            "created_at": now_stamp(),
            "grid_path": str(grid_path),
            "batch_dir": str(batch_dir),
            "run_until_success": True,
            "success_mode": SUCCESS_MODE,
            "min_ok_candidates": RUN_MIN_OK_CANDIDATES,
            "max_attempts": RUN_MAX_ATTEMPTS,
            "max_runtime_seconds": RUN_MAX_RUNTIME_SECONDS,
            "randomize_seed": RANDOMIZE_SEED,
            "start_seed": START_SEED,
            "public_recorded_data_only": True,
            "private_api": False,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
        }
        (batch_dir / "batch_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Rawseq I/O contract discovery run-until-success complete")
        print(f"Attempts: {len(rows)}")
        print(f"CSV: {batch_dir / 'attempts.csv'}")
        print(f"State: {batch_dir / 'run_state.json'}")
        print("Safety: public recorded data only. No private API. No orders. No champion mutation. No promotion.")
        return 0
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(contracts, start=1):
        if time.monotonic() - started >= MAX_RUNTIME_SECONDS:
            path_info = contract_path_info(batch_dir, row, index)
            skipped_row = {
                **row,
                "full_contract_slug": path_info["full_contract_slug"],
                "filesystem_contract_slug": path_info["filesystem_contract_slug"],
                "filesystem_path_shortened": path_info["filesystem_path_shortened"],
                "projected_path_length": path_info["projected_path_length"],
            }
            rows.append(base_summary(skipped_row, path_info["contract_dir"], "SKIPPED_MAX_RUNTIME", "", time.monotonic() - started, 0.0, ""))
            continue
        print(f"[{index}/{len(contracts)}] {safe_str(row.get('contract_slug'))}")
        rows.append(run_contract(row, batch_dir, index, started))

    summary_csv = batch_dir / "io_contract_discovery_summary.csv"
    summary_txt = batch_dir / "io_contract_discovery_summary.txt"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    summary_txt.write_text(render_summary(batch_dir, grid_path, rows, started), encoding="utf-8")
    metadata = {
        "created_at": now_stamp(),
        "grid_path": str(grid_path),
        "batch_dir": str(batch_dir),
        "dry_run": DRY_RUN,
        "max_contracts": MAX_CONTRACTS,
        "max_candidates": MAX_CANDIDATES,
        "max_windows": MAX_WINDOWS,
        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
        "public_recorded_data_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    (batch_dir / "batch_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Rawseq I/O contract discovery batch complete")
    print(f"Contracts attempted: {len(rows)}")
    print(f"CSV: {summary_csv}")
    print(f"TXT: {summary_txt}")
    print("Safety: public recorded data only. No private API. No orders. No champion mutation. No promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
