#!/usr/bin/env python3
"""Summarize frozen rawseq shadow cost-threshold sweep CSVs."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip()
BASE_DIR = PROJECT_ROOT / "data" / "realtime" / PRIMARY_VENUE
SWEEP_GLOB = os.getenv(
    "RAWSEQ_THRESHOLD_SWEEP_GLOB",
    f"{SYMBOL}_rawseq_frozen_shadow_cost_threshold_*.csv",
).strip()
COSTS_OF_INTEREST_ENV = os.getenv("RAWSEQ_THRESHOLD_SWEEP_COSTS_OF_INTEREST", "0.05,0.1,0.25")
OUTPUT_PATH = Path(
    os.getenv(
        "RAWSEQ_THRESHOLD_SWEEP_OUTPUT",
        BASE_DIR / f"{SYMBOL}_rawseq_frozen_shadow_threshold_sweep_summary.csv",
    )
)
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH

THRESHOLD_RE = re.compile(r"_threshold_([0-9]+(?:\.[0-9]+)?)\.csv$", re.IGNORECASE)
MIN_SELECTED_ROWS = int(os.getenv("RAWSEQ_THRESHOLD_SWEEP_MIN_SELECTED_ROWS", "500"))


def parse_costs(text: str) -> list[float]:
    costs: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            costs.append(float(item))
    return costs


def cost_label(cost: float) -> str:
    text = f"{cost:g}".replace(".", "p").replace("-", "m")
    return text


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except Exception:
        return math.nan


def infer_threshold(path: Path, frame: pd.DataFrame) -> float:
    if "threshold_bps" in frame.columns and frame["threshold_bps"].notna().any():
        return finite_or_nan(frame["threshold_bps"].dropna().iloc[0])
    match = THRESHOLD_RE.search(path.name)
    if match:
        return float(match.group(1))
    return math.nan


def row_for_cost(frame: pd.DataFrame, cost: float) -> pd.Series | None:
    if "cost_bps" not in frame.columns:
        return None
    cost_values = pd.to_numeric(frame["cost_bps"], errors="coerce")
    matches = frame[np.isclose(cost_values, cost, atol=1e-9, rtol=0.0)]
    if matches.empty:
        return None
    return matches.iloc[0]


def selected_rows_at_cost_zero(frame: pd.DataFrame) -> int:
    row = row_for_cost(frame, 0.0)
    if row is None:
        row = frame.iloc[0] if len(frame) else None
    if row is None:
        return 0
    return int(finite_or_nan(row.get("selected_rows", 0)) or 0)


def pass_flag(row: pd.Series | None, selected_rows: int) -> bool:
    if row is None or selected_rows < MIN_SELECTED_ROWS:
        return False
    return finite_or_nan(row.get("cum_net_bps")) > 0.0


def notes_for(summary: dict[str, Any], costs: list[float]) -> str:
    notes: list[str] = []
    selected_rows = int(summary.get("selected_rows_at_cost_0", 0) or 0)
    if selected_rows < MIN_SELECTED_ROWS:
        notes.append("sparse")
    cost_010 = "0p1"
    cum_010 = finite_or_nan(summary.get(f"cum_net_bps_cost_{cost_010}"))
    dip_010 = finite_or_nan(summary.get(f"max_dip_net_bps_cost_{cost_010}"))
    if math.isfinite(cum_010) and cum_010 <= 0:
        notes.append("negative at 0.10bps")
    if math.isfinite(cum_010) and math.isfinite(dip_010) and abs(dip_010) > max(cum_010, 0.0):
        notes.append("drawdown exceeds 0.10bps cum net")
    missing = [f"{cost:g}" for cost in costs if not math.isfinite(finite_or_nan(summary.get(f"cum_net_bps_cost_{cost_label(cost)}")))]
    if missing:
        notes.append("missing costs " + ",".join(missing))
    return "; ".join(notes)


def ranking_score(summary: dict[str, Any]) -> float:
    selected_rows = int(summary.get("selected_rows_at_cost_0", 0) or 0)
    cum_010 = finite_or_nan(summary.get("cum_net_bps_cost_0p1"))
    dip_010 = finite_or_nan(summary.get("max_dip_net_bps_cost_0p1"))
    avg_010 = finite_or_nan(summary.get("avg_net_bps_cost_0p1"))
    if not math.isfinite(cum_010):
        return -1e12
    sparse_penalty = max(0, MIN_SELECTED_ROWS - selected_rows) * 10.0
    drawdown_penalty = abs(dip_010) * 0.25 if math.isfinite(dip_010) else 1_000.0
    avg_bonus = avg_010 * 1_000.0 if math.isfinite(avg_010) else 0.0
    positive_bonus = 10_000.0 if cum_010 > 0.0 and selected_rows >= MIN_SELECTED_ROWS else 0.0
    return positive_bonus + cum_010 + avg_bonus - drawdown_penalty - sparse_penalty


def summarize_file(path: Path, costs: list[float]) -> dict[str, Any]:
    frame = pd.read_csv(path, low_memory=False)
    threshold = infer_threshold(path, frame)
    selected_rows = selected_rows_at_cost_zero(frame)
    summary: dict[str, Any] = {
        "threshold_bps": threshold,
        "selected_rows_at_cost_0": selected_rows,
        "source_path": str(path),
    }

    cost_rows = {cost: row_for_cost(frame, cost) for cost in costs}
    for cost, row in cost_rows.items():
        label = cost_label(cost)
        for metric in ["avg_net_bps", "cum_net_bps", "max_dip_net_bps", "win_rate_net"]:
            summary[f"{metric}_cost_{label}"] = finite_or_nan(row.get(metric)) if row is not None else math.nan

    summary["pass_0p05"] = pass_flag(cost_rows.get(0.05), selected_rows)
    summary["pass_0p10"] = pass_flag(cost_rows.get(0.10), selected_rows)
    summary["pass_0p25"] = pass_flag(cost_rows.get(0.25), selected_rows)
    summary["notes"] = notes_for(summary, costs)
    summary["rank_score"] = ranking_score(summary)
    return summary


def build_summary() -> pd.DataFrame:
    costs = parse_costs(COSTS_OF_INTEREST_ENV)
    if not costs:
        raise SystemExit("RAWSEQ_THRESHOLD_SWEEP_COSTS_OF_INTEREST did not contain any costs.")
    paths = sorted(BASE_DIR.glob(SWEEP_GLOB))
    if not paths:
        raise SystemExit(f"No threshold files matched: {BASE_DIR / SWEEP_GLOB}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            rows.append(summarize_file(path, costs))
        except Exception as exc:
            rows.append(
                {
                    "threshold_bps": math.nan,
                    "selected_rows_at_cost_0": 0,
                    "source_path": str(path),
                    "pass_0p05": False,
                    "pass_0p10": False,
                    "pass_0p25": False,
                    "notes": f"failed to read: {exc}",
                    "rank_score": -1e12,
                }
            )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["pass_0p10", "rank_score", "selected_rows_at_cost_0", "threshold_bps"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def print_table(summary: pd.DataFrame) -> None:
    print("rawseq_frozen_threshold_sweep_summary")
    print(f"Directory: {BASE_DIR}")
    print(f"Glob: {SWEEP_GLOB}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Minimum selected rows: {MIN_SELECTED_ROWS}")
    print()
    columns = [
        "threshold_bps",
        "selected_rows_at_cost_0",
        "cum_net_bps_cost_0p1",
        "avg_net_bps_cost_0p1",
        "max_dip_net_bps_cost_0p1",
        "pass_0p10",
        "notes",
    ]
    widths = {
        "threshold_bps": 13,
        "selected_rows_at_cost_0": 23,
        "cum_net_bps_cost_0p1": 20,
        "avg_net_bps_cost_0p1": 20,
        "max_dip_net_bps_cost_0p1": 25,
        "pass_0p10": 10,
        "notes": 44,
    }
    print(" ".join(column.ljust(widths[column]) for column in columns))
    print(" ".join("-" * widths[column] for column in columns))
    for _, row in summary.iterrows():
        values = {
            "threshold_bps": f"{finite_or_nan(row.get('threshold_bps')):.4g}",
            "selected_rows_at_cost_0": int(row.get("selected_rows_at_cost_0", 0) or 0),
            "cum_net_bps_cost_0p1": f"{finite_or_nan(row.get('cum_net_bps_cost_0p1')):.2f}",
            "avg_net_bps_cost_0p1": f"{finite_or_nan(row.get('avg_net_bps_cost_0p1')):.4f}",
            "max_dip_net_bps_cost_0p1": f"{finite_or_nan(row.get('max_dip_net_bps_cost_0p1')):.2f}",
            "pass_0p10": str(bool(row.get("pass_0p10"))).lower(),
            "notes": str(row.get("notes", ""))[:44],
        }
        print(" ".join(str(values[column]).ljust(widths[column]) for column in columns))


def main() -> None:
    summary = build_summary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)
    print_table(summary)


if __name__ == "__main__":
    main()
