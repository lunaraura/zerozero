#!/usr/bin/env python3
"""Cost-aware policy sweep for tiny rawseq run folders.

Read-only analysis script: it reads run artifacts and writes CSV reports only.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "rawseq_runs"
RUN_ROOT = Path(os.getenv("RAWSEQ_POLICY_SWEEP_ROOT", str(DEFAULT_ROOT)))
RUN_GLOB = os.getenv("RAWSEQ_POLICY_SWEEP_GLOB", "*")
COST_BPS = float(os.getenv("RAWSEQ_COST_BPS", "1.0"))
SYMBOL_FILTER_ENV = os.getenv("RAWSEQ_POLICY_SWEEP_SYMBOLS", "").strip()
SEED_FILTER_ENV = os.getenv("RAWSEQ_POLICY_SWEEP_SEEDS", "").strip()
STATUS_ALLOW_ENV = os.getenv("RAWSEQ_POLICY_SWEEP_STATUS_ALLOW", "valid,suspicious").strip()
MAX_RUNS_ENV = os.getenv("RAWSEQ_POLICY_SWEEP_MAX_RUNS", "").strip()
ALLOW_FULL_SCAN_ENV = os.getenv("RAWSEQ_POLICY_SWEEP_ALLOW_FULL_SCAN", "false").strip()
OUTPUT_PREFIX = Path(
    os.getenv(
        "RAWSEQ_POLICY_SWEEP_OUTPUT_PREFIX",
        str(DEFAULT_ROOT / "rawseq_policy_sweep_cost_aware"),
    )
)

if not RUN_ROOT.is_absolute():
    RUN_ROOT = PROJECT_ROOT / RUN_ROOT
if not OUTPUT_PREFIX.is_absolute():
    OUTPUT_PREFIX = PROJECT_ROOT / OUTPUT_PREFIX

ALL_RUNS_PATH = OUTPUT_PREFIX.with_name(f"{OUTPUT_PREFIX.name}_all_runs.csv")
ROLLUP_PATH = OUTPUT_PREFIX.with_name(f"{OUTPUT_PREFIX.name}_rollup.csv")
HEALTH_REPORT_PATH = DEFAULT_ROOT / "rawseq_run_health_report.csv"

PRED_COLUMN = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COLUMN = "rawseq_path_actual_horizon_return_bps"
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
POLICY_TYPES = ["direct_gt", "inverse_gt", "inverse_directional_abs_gt"]

SYMBOL_RE = re.compile(r"(?<![A-Z0-9])([A-Z0-9]{2,20}USDT)(?![A-Z0-9])")
SEED_RE = re.compile(r"(?:^|_)seed[_-]?(\d+)(?:_|$)", re.IGNORECASE)


def infer_symbol_from_name(name: str) -> str:
    match = SYMBOL_RE.search(name)
    return match.group(1).upper() if match else ""


def infer_seed_from_name(name: str) -> str:
    match = SEED_RE.search(name)
    return match.group(1) if match else ""


def parse_csv_set(text: str, uppercase: bool = False) -> set[str]:
    values = {item.strip() for item in text.split(",") if item.strip()}
    return {item.upper() for item in values} if uppercase else values


def parse_seed_filter(text: str) -> set[str]:
    seeds: set[str] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            try:
                start = int(left.strip())
                end = int(right.strip())
            except ValueError:
                seeds.add(item)
                continue
            if start <= end:
                seeds.update(str(seed) for seed in range(start, end + 1))
            else:
                seeds.update(str(seed) for seed in range(start, end - 1, -1))
        else:
            seeds.add(item)
    return seeds


def parse_max_runs(text: str) -> int | None:
    if not text:
        return None
    try:
        value = int(text)
        return value if value > 0 else None
    except ValueError:
        return None


def env_truthy(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_allowed_runs_from_health(status_allow: set[str]) -> set[str] | None:
    if not HEALTH_REPORT_PATH.exists() or not status_allow:
        return None
    try:
        health = pd.read_csv(HEALTH_REPORT_PATH, usecols=["run", "status"], dtype=str)
    except Exception:
        return None
    health["run"] = health["run"].fillna("").astype(str)
    health["status"] = health["status"].fillna("").astype(str).str.strip().str.lower()
    return set(health.loc[health["status"].isin(status_allow), "run"])


def discover_run_dirs() -> list[Path]:
    if not RUN_ROOT.exists():
        return []

    symbol_filter = parse_csv_set(SYMBOL_FILTER_ENV, uppercase=True)
    seed_filter = parse_seed_filter(SEED_FILTER_ENV)
    status_allow = {status.lower() for status in parse_csv_set(STATUS_ALLOW_ENV)}
    allowed_runs = load_allowed_runs_from_health(status_allow)
    max_runs = parse_max_runs(MAX_RUNS_ENV)
    if (
        RUN_GLOB == "*"
        and not symbol_filter
        and not seed_filter
        and max_runs is None
        and not env_truthy(ALLOW_FULL_SCAN_ENV)
    ):
        raise SystemExit(
            "Refusing default full rawseq corpus scan. Set RAWSEQ_POLICY_SWEEP_SYMBOLS, "
            "RAWSEQ_POLICY_SWEEP_SEEDS, RAWSEQ_POLICY_SWEEP_MAX_RUNS, or a narrower "
            "RAWSEQ_POLICY_SWEEP_GLOB; or explicitly set RAWSEQ_POLICY_SWEEP_ALLOW_FULL_SCAN=true."
        )

    run_dirs = []
    for path in sorted(path for path in RUN_ROOT.glob(RUN_GLOB) if path.is_dir()):
        if allowed_runs is not None and path.name not in allowed_runs:
            continue
        symbol = infer_symbol_from_name(path.name)
        seed = infer_seed_from_name(path.name)
        if symbol_filter and symbol not in symbol_filter:
            continue
        if seed_filter and seed not in seed_filter:
            continue
        run_dirs.append(path)
        if max_runs is not None and len(run_dirs) >= max_runs:
            break
    return run_dirs


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except Exception:
        return math.nan


def max_dip_bps(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return math.nan
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    return float(np.min(drawdown)) if len(drawdown) else math.nan


def metric_row(
    run_dir: Path,
    policy_type: str,
    threshold: float,
    pred: np.ndarray,
    actual: np.ndarray,
) -> dict[str, Any]:
    if policy_type == "direct_gt":
        mask = pred > threshold
        gross = actual[mask]
    elif policy_type == "inverse_gt":
        mask = pred > threshold
        gross = -actual[mask]
    elif policy_type == "inverse_directional_abs_gt":
        mask = np.abs(pred) > threshold
        gross = -np.sign(pred[mask]) * actual[mask]
    else:
        raise ValueError(f"unknown policy_type={policy_type}")

    gross = np.asarray(gross, dtype="float64")
    gross = gross[np.isfinite(gross)]
    net = gross - COST_BPS

    rows = int(len(gross))
    avg_gross = float(np.mean(gross)) if rows else math.nan
    avg_net = float(np.mean(net)) if rows else math.nan
    cum_gross = float(np.sum(gross)) if rows else 0.0
    cum_net = float(np.sum(net)) if rows else 0.0
    win_gross = float(np.mean(gross > 0.0)) if rows else math.nan
    win_net = float(np.mean(net > 0.0)) if rows else math.nan

    policy = f"{policy_type}_{threshold:g}"
    return {
        "run": run_dir.name,
        "symbol": infer_symbol_from_name(run_dir.name),
        "seed": infer_seed_from_name(run_dir.name),
        "policy": policy,
        "policy_type": policy_type,
        "threshold_bps": threshold,
        "rows": rows,
        "avg_gross_return_bps": avg_gross,
        "avg_net_return_bps": avg_net,
        "cumulative_gross_return_bps": cum_gross,
        "cumulative_net_return_bps": cum_net,
        "win_rate_gross": win_gross,
        "win_rate_net": win_net,
        "max_dip_gross_bps": max_dip_bps(gross),
        "max_dip_net_bps": max_dip_bps(net),
        "cost_bps": COST_BPS,
        "status": "ok",
        "issues": "",
    }


def count_csv_data_rows(path: Path) -> int:
    with path.open("rb") as handle:
        line_count = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))
    return max(0, line_count - 1)


def error_row(run_dir: Path, issue: str) -> dict[str, Any]:
    return {
        "run": run_dir.name,
        "symbol": infer_symbol_from_name(run_dir.name),
        "seed": infer_seed_from_name(run_dir.name),
        "policy": "",
        "policy_type": "",
        "threshold_bps": math.nan,
        "rows": 0,
        "avg_gross_return_bps": math.nan,
        "avg_net_return_bps": math.nan,
        "cumulative_gross_return_bps": 0.0,
        "cumulative_net_return_bps": 0.0,
        "win_rate_gross": math.nan,
        "win_rate_net": math.nan,
        "max_dip_gross_bps": math.nan,
        "max_dip_net_bps": math.nan,
        "cost_bps": COST_BPS,
        "status": "skipped",
        "issues": issue,
    }


def load_test_vectors(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    annotated_path = run_dir / "annotated.csv"
    if not annotated_path.exists():
        raise FileNotFoundError("missing annotated.csv")

    total_rows = count_csv_data_rows(annotated_path)
    if total_rows <= 0:
        raise ValueError("annotated.csv has no rows")
    split_at = int(total_rows * 0.8)

    try:
        test = pd.read_csv(
            annotated_path,
            usecols=[PRED_COLUMN, ACTUAL_COLUMN],
            skiprows=range(1, split_at + 1),
            low_memory=False,
        )
    except ValueError as exc:
        raise ValueError(f"annotated.csv missing required columns: {exc}") from exc

    test[PRED_COLUMN] = pd.to_numeric(test[PRED_COLUMN], errors="coerce")
    test[ACTUAL_COLUMN] = pd.to_numeric(test[ACTUAL_COLUMN], errors="coerce")
    test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=[PRED_COLUMN, ACTUAL_COLUMN])
    if test.empty:
        raise ValueError("test split has no finite prediction/actual rows")

    return test[PRED_COLUMN].to_numpy(dtype="float64"), test[ACTUAL_COLUMN].to_numpy(dtype="float64")


def evaluate_run(run_dir: Path) -> list[dict[str, Any]]:
    try:
        pred, actual = load_test_vectors(run_dir)
    except Exception as exc:
        return [error_row(run_dir, str(exc))]

    rows: list[dict[str, Any]] = []
    for policy_type in POLICY_TYPES:
        for threshold in THRESHOLDS:
            try:
                rows.append(metric_row(run_dir, policy_type, threshold, pred, actual))
            except Exception as exc:
                row = error_row(run_dir, f"{policy_type}_{threshold:g}: {exc}")
                row["policy"] = f"{policy_type}_{threshold:g}"
                row["policy_type"] = policy_type
                row["threshold_bps"] = threshold
                rows.append(row)
    return rows


def summarize_group(group: pd.DataFrame) -> pd.Series:
    active = group[group["rows"] > 0]
    return pd.Series(
        {
            "seeds": int(group["seed"].replace("", np.nan).nunique(dropna=True)),
            "active_seeds": int(active["seed"].replace("", np.nan).nunique(dropna=True)),
            "positive_net_active": int((active["avg_net_return_bps"] > 0.0).sum()),
            "median_rows": finite_or_nan(active["rows"].median()),
            "median_avg_net": finite_or_nan(active["avg_net_return_bps"].median()),
            "worst_avg_net": finite_or_nan(active["avg_net_return_bps"].min()),
            "best_avg_net": finite_or_nan(active["avg_net_return_bps"].max()),
            "median_cum_net": finite_or_nan(active["cumulative_net_return_bps"].median()),
            "median_win_net": finite_or_nan(active["win_rate_net"].median()),
            "median_max_dip_net": finite_or_nan(active["max_dip_net_bps"].median()),
            "worst_max_dip_net": finite_or_nan(active["max_dip_net_bps"].min()),
            "cost_bps": COST_BPS,
        }
    )


def build_rollup(all_runs: pd.DataFrame) -> pd.DataFrame:
    ok = all_runs[(all_runs["status"] == "ok") & (all_runs["symbol"].astype(str) != "")].copy()
    if ok.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "policy",
                "policy_type",
                "threshold_bps",
                "seeds",
                "active_seeds",
                "positive_net_active",
                "median_rows",
                "median_avg_net",
                "worst_avg_net",
                "best_avg_net",
                "median_cum_net",
                "median_win_net",
                "median_max_dip_net",
                "worst_max_dip_net",
                "cost_bps",
            ]
        )

    rollup = (
        ok.groupby(["symbol", "policy", "policy_type", "threshold_bps"], dropna=False)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )
    return rollup.sort_values(
        ["active_seeds", "positive_net_active", "median_avg_net"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def print_top_policies(rollup: pd.DataFrame) -> None:
    print("rawseq_policy_sweep_cost_aware")
    print(f"Root: {RUN_ROOT}")
    print(f"Glob: {RUN_GLOB}")
    print(f"Cost bps: {COST_BPS:g}")
    print(f"Symbols filter: {SYMBOL_FILTER_ENV or 'all'}")
    print(f"Seeds filter: {SEED_FILTER_ENV or 'all'}")
    print(f"Status allow: {STATUS_ALLOW_ENV or 'all'}")
    print(f"Max runs: {MAX_RUNS_ENV or 'none'}")
    print(f"All runs: {ALL_RUNS_PATH}")
    print(f"Rollup: {ROLLUP_PATH}")
    if rollup.empty:
        print("No rollup rows to display.")
        return

    print()
    columns = [
        "symbol",
        "policy",
        "active_seeds",
        "positive_net_active",
        "median_rows",
        "median_avg_net",
        "worst_avg_net",
        "median_cum_net",
        "median_win_net",
        "median_max_dip_net",
    ]
    widths = {
        "symbol": 10,
        "policy": 32,
        "active_seeds": 12,
        "positive_net_active": 19,
        "median_rows": 12,
        "median_avg_net": 14,
        "worst_avg_net": 13,
        "median_cum_net": 14,
        "median_win_net": 14,
        "median_max_dip_net": 18,
    }
    print(" ".join(column.ljust(widths[column]) for column in columns))
    print(" ".join("-" * widths[column] for column in columns))
    for _, row in rollup.groupby("symbol", sort=True).head(5).iterrows():
        values = {
            "symbol": row["symbol"],
            "policy": row["policy"],
            "active_seeds": int(row["active_seeds"]),
            "positive_net_active": int(row["positive_net_active"]),
            "median_rows": f"{row['median_rows']:.0f}",
            "median_avg_net": f"{row['median_avg_net']:.4f}",
            "worst_avg_net": f"{row['worst_avg_net']:.4f}",
            "median_cum_net": f"{row['median_cum_net']:.2f}",
            "median_win_net": f"{row['median_win_net']:.4f}",
            "median_max_dip_net": f"{row['median_max_dip_net']:.2f}",
        }
        print(" ".join(str(values[column]).ljust(widths[column]) for column in columns))


def main() -> None:
    rows: list[dict[str, Any]] = []
    for run_dir in discover_run_dirs():
        rows.extend(evaluate_run(run_dir))

    all_runs = pd.DataFrame(rows)
    if all_runs.empty:
        all_runs = pd.DataFrame(columns=list(error_row(Path(""), "empty").keys()))

    rollup = build_rollup(all_runs)

    OUTPUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(ALL_RUNS_PATH, index=False)
    rollup.to_csv(ROLLUP_PATH, index=False)
    print_top_policies(rollup)


if __name__ == "__main__":
    main()
