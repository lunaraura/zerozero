#!/usr/bin/env python3
"""Run bounded rawseq walk-forward discovery from an I/O contract grid.

This is an orchestration wrapper around run_rawseq_recorded_walkforward_evolution.py.
It uses public/recorded source data only and writes research outputs. It never
uses private APIs, places orders, promotes models, or mutates champion folders.
"""

from __future__ import annotations

import csv
import json
import math
import os
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


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
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
    path = root / f"rawseq_io_contract_discovery_{now_stamp()}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def short_contract_dir_name(row: dict[str, str], index: int) -> str:
    feature = safe_str(row.get("input_feature")) or "feature"
    ma = safe_str(row.get("ma_window")) or "NA"
    hidden = (safe_str(row.get("hidden")) or "h").replace(",", "x")
    input_stride = safe_str(row.get("input_stride")) or "1"
    output_stride = safe_str(row.get("output_stride")) or "1"
    return f"c{index:03d}_{feature}_ma{ma}_h{hidden}_is{input_stride}_os{output_stride}"


def run_contract(row: dict[str, str], batch_dir: Path, index: int, started_at: float) -> dict[str, Any]:
    contract_slug = safe_str(row.get("contract_slug")) or f"contract_{index:03d}"
    contract_dir = batch_dir / "contracts" / short_contract_dir_name(row, index)
    wf_output_root = contract_dir / "walkforward"
    wf_run_id = "wf"
    contract_dir.mkdir(parents=True, exist_ok=True)
    (contract_dir / "contract.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if DRY_RUN:
        wf_run_dir = wf_output_root / wf_run_id
        wf_run_dir.mkdir(parents=True, exist_ok=True)
        dry_payload = {
            "dry_run": True,
            "contract_slug": contract_slug,
            "max_windows": max(1, MAX_WINDOWS),
            "seeds": parse_seed_limit(SEEDS),
            "public_recorded_data_only": True,
            "private_api": False,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
        }
        (wf_run_dir / "dry_run_manifest.json").write_text(json.dumps(dry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (contract_dir / "walkforward_stdout.log").write_text("DRY_RUN: walk-forward subprocess not executed by discovery batch.\n", encoding="utf-8")
        summary = base_summary(row, contract_dir, "DRY_RUN", "", time.monotonic() - started_at, 0.0, str(contract_dir / "walkforward_stdout.log"))
        summary["walkforward_run_dir"] = str(wf_run_dir)
        summary["candidate_rows"] = max(1, MAX_WINDOWS) * len(parse_seed_limit(SEEDS).split(","))
        return summary

    elapsed = time.monotonic() - started_at
    remaining = MAX_RUNTIME_SECONDS - elapsed
    if remaining <= 0:
        return base_summary(row, contract_dir, "SKIPPED_MAX_RUNTIME", "", elapsed, 0.0, "")

    env = os.environ.copy()
    env.update(
        {
            "SYMBOL": safe_str(row.get("symbol")) or "SOLUSDT",
            "PRIMARY_VENUE": safe_str(row.get("venue")) or "kraken",
            "RAWSEQ_WF_SOURCE_PATH": safe_str(row.get("source_path")),
            "RAWSEQ_WF_OUTPUT_ROOT": str(wf_output_root),
            "RAWSEQ_WF_RUN_ID": wf_run_id,
            "RAWSEQ_WF_INPUT_FEATURES": safe_str(row.get("input_feature")),
            "RAWSEQ_WF_MA_WINDOWS": safe_str(row.get("ma_window")) or "60",
            "RAWSEQ_WF_HIDDENS": safe_str(row.get("hidden")),
            "RAWSEQ_WF_SEEDS": parse_seed_limit(SEEDS),
            "RAWSEQ_WF_INPUT_STRIDES": safe_str(row.get("input_stride")) or "1",
            "RAWSEQ_WF_OUTPUT_STRIDES": safe_str(row.get("output_stride")) or "1",
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
    log_path = contract_dir / "walkforward_stdout.log"
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
        status = "DRY_RUN" if DRY_RUN and completed.returncode == 0 else "OK" if completed.returncode == 0 else "FAILED"
        return summarize_contract(row, contract_dir, wf_output_root / wf_run_id, status, completed.returncode, runtime, str(log_path))
    except subprocess.TimeoutExpired as exc:
        log_path.write_text((exc.stdout or "") if isinstance(exc.stdout, str) else "", encoding="utf-8")
        runtime = time.monotonic() - run_started
        return base_summary(row, contract_dir, "TIMEOUT", "timeout", time.monotonic() - started_at, runtime, str(log_path))


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
        "hidden": safe_str(row.get("hidden")),
        "bucket_seconds": safe_str(row.get("bucket_seconds")),
        "seq_len": safe_str(row.get("seq_len")),
        "input_stride": safe_str(row.get("input_stride")),
        "output_stride": safe_str(row.get("output_stride")),
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
        leaderboard = pd.read_csv(leaderboard_path, low_memory=False)
        summary["leaderboard_rows"] = int(len(leaderboard))
    return summary


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
    batch_dir = output_root()
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(contracts, start=1):
        if time.monotonic() - started >= MAX_RUNTIME_SECONDS:
            rows.append(base_summary(row, batch_dir / "contracts" / safe_str(row.get("contract_slug")), "SKIPPED_MAX_RUNTIME", "", time.monotonic() - started, 0.0, ""))
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
