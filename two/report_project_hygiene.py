#!/usr/bin/env python3
"""Print a compact hygiene report for the tiny/rawseq workflow.

Read-only: this script only inspects files and git metadata.
"""

from __future__ import annotations

import csv
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TINY_DIR = PROJECT_ROOT / "scripts" / "tiny"
KRAKEN_DIR = PROJECT_ROOT / "data" / "realtime" / "kraken"
RAWSEQ_RUNS_DIR = PROJECT_ROOT / "data" / "rawseq_runs"
PAPER_CHAMPIONS_DIR = PROJECT_ROOT / "data" / "paper_champions"
SYMBOLS = ["SOLUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]

KEY_RAWSEQ_SCRIPTS = [
    "scripts/tiny_price_rawseq_path_v1.py",
    "scripts/tiny/run_rawseq_frozen_shadow.py",
    "scripts/tiny/freeze_rawseq_paper_champion.py",
    "scripts/tiny/evaluate_rawseq_frozen_shadow_costs.py",
    "scripts/tiny/evaluate_rawseq_frozen_shadow_rolling_costs.py",
    "scripts/tiny/evaluate_rawseq_frozen_shadow_overlay_filters.py",
    "scripts/tiny/summarize_rawseq_frozen_threshold_sweep.py",
    "scripts/tiny/report_rawseq_frozen_shadow_summary.py",
    "scripts/tiny/probe_rawseq_annotated_flow_join.py",
    "scripts/tiny/report_rawseq_run_health.py",
    "scripts/tiny/sweep_rawseq_policies_cost_aware.py",
]

ANALYSIS_CSV_PATTERNS = [
    "*rawseq_frozen_shadow*.csv",
    "*rawseq_annotated_flow_join_probe.csv",
    "rawseq_run_health_report.csv",
    "rawseq_policy_sweep_cost_aware*.csv",
]


def run_git(args: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.returncode, result.stdout.strip()
    except Exception as exc:
        return 1, f"git command failed: {exc}"


def format_mtime(path: Path) -> str:
    if not path.exists():
        return "missing"
    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return timestamp.strftime("%Y-%m-%d %H:%M:%SZ")


def file_size_mb(path: Path) -> str:
    if not path.exists():
        return ""
    return f"{path.stat().st_size / (1024 * 1024):.2f} MB"


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def git_status_short() -> list[str]:
    code, output = run_git(["status", "--short"])
    if code != 0:
        return [output or "git status unavailable"]
    return output.splitlines() if output else ["clean"]


def recent_commits(limit: int = 8) -> list[str]:
    code, output = run_git(["log", f"--max-count={limit}", "--oneline", "--decorate"])
    if code != 0:
        return [output or "git log unavailable"]
    return output.splitlines() if output else ["no commits found"]


def untracked_tiny_scripts(status_lines: list[str]) -> list[str]:
    scripts = []
    for line in status_lines:
        if not line.startswith("?? "):
            continue
        path_text = line[3:].replace("\\", "/")
        if path_text.startswith("scripts/tiny/") and path_text.endswith(".py"):
            scripts.append(path_text)
    return sorted(scripts)


def generated_analysis_untracked(status_lines: list[str]) -> list[str]:
    warnings = []
    for line in status_lines:
        if not line.startswith("?? "):
            continue
        path_text = line[3:].replace("\\", "/")
        path = PROJECT_ROOT / path_text
        if path.suffix.lower() != ".csv":
            continue
        name = path.name
        if "rawseq" in name and any(path.match(f"**/{pattern}") for pattern in ANALYSIS_CSV_PATTERNS):
            warnings.append(path_text)
    return sorted(warnings)


def latest_files(base: Path, patterns: list[str], limit: int = 12) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(base.glob(pattern))
    return sorted({path for path in files if path.is_file()}, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


def print_git_status(status_lines: list[str]) -> None:
    print_section("Git Status Short")
    for line in status_lines[:80]:
        print(line)
    if len(status_lines) > 80:
        print(f"... {len(status_lines) - 80} more lines")


def print_untracked_scripts(status_lines: list[str]) -> None:
    print_section("Untracked Scripts Under scripts/tiny")
    scripts = untracked_tiny_scripts(status_lines)
    if not scripts:
        print("none")
        return
    for script in scripts:
        print(script)


def print_recent_commits() -> None:
    print_section("Recent Commits")
    for line in recent_commits():
        print(line)


def print_key_scripts() -> None:
    print_section("Key Rawseq Scripts")
    for rel_path in KEY_RAWSEQ_SCRIPTS:
        path = PROJECT_ROOT / rel_path
        status = "present" if path.exists() else "missing"
        print(f"{status:8} {rel_path}")


def print_latest_champions() -> None:
    print_section("Latest Paper Champion Folders")
    if not PAPER_CHAMPIONS_DIR.exists():
        print(f"missing {PAPER_CHAMPIONS_DIR}")
        return
    folders = sorted(
        [path for path in PAPER_CHAMPIONS_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:10]
    if not folders:
        print("none")
        return
    for folder in folders:
        print(f"{format_mtime(folder)}  {folder.name}")


def print_latest_analysis_csvs() -> None:
    print_section("Latest Cost/Rolling/Threshold Summary CSVs")
    patterns = [
        "*rawseq_frozen_shadow_cost*.csv",
        "*rawseq_frozen_shadow_rolling*.csv",
        "*rawseq_frozen_shadow_threshold*.csv",
        "*rawseq_frozen_shadow_summary.csv",
        "*rawseq_frozen_shadow_overlay_filters.csv",
        "*rawseq_annotated_flow_join_probe.csv",
    ]
    files = latest_files(KRAKEN_DIR, patterns, limit=20)
    if not files:
        print("none")
        return
    for path in files:
        print(f"{format_mtime(path)}  {file_size_mb(path):>9}  {path.name}")


def print_health_counts() -> None:
    print_section("Rawseq Health Report Counts")
    health_path = RAWSEQ_RUNS_DIR / "rawseq_run_health_report.csv"
    if not health_path.exists():
        print(f"missing {health_path}")
        return
    counts: Counter[str] = Counter()
    try:
        with health_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                counts[str(row.get("status", "")).strip() or "unknown"] += 1
    except Exception as exc:
        print(f"failed to read {health_path}: {exc}")
        return
    print(f"{format_mtime(health_path)}  {health_path}")
    for status, count in sorted(counts.items()):
        print(f"{status:12} {count}")


def print_realtime_kraken_files() -> None:
    print_section("Realtime Kraken File Timestamps")
    suffixes = [
        "10s_flow.csv",
        "1m_flow.csv",
        "tiny_price_prediction_evaluation_rows_with_rawseq_path_v1.csv",
        "tiny_price_rawseq_path_v1_shadow_evaluation.csv",
    ]
    for symbol in SYMBOLS:
        print(symbol)
        for suffix in suffixes:
            path = KRAKEN_DIR / f"{symbol}_{suffix}"
            print(f"  {format_mtime(path)}  {file_size_mb(path):>9}  {path.name}")


def print_untracked_analysis_warnings(status_lines: list[str]) -> None:
    print_section("Warnings")
    warnings = generated_analysis_untracked(status_lines)
    if not warnings:
        print("no untracked generated analysis CSV warnings")
        return
    print("untracked generated analysis CSVs:")
    for path in warnings[:80]:
        print(f"  {path}")
    if len(warnings) > 80:
        print(f"  ... {len(warnings) - 80} more")


def main() -> None:
    print("Tiny Rawseq Project Hygiene")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}")
    status_lines = git_status_short()
    print_git_status(status_lines)
    print_untracked_scripts(status_lines)
    print_recent_commits()
    print_key_scripts()
    print_latest_champions()
    print_latest_analysis_csvs()
    print_health_counts()
    print_realtime_kraken_files()
    print_untracked_analysis_warnings(status_lines)


if __name__ == "__main__":
    main()
