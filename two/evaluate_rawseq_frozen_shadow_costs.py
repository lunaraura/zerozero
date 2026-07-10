#!/usr/bin/env python3
"""Evaluate cost sensitivity for a frozen rawseq shadow output.

Read-only except for writing the cost sensitivity CSV.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip()
POLICY = os.getenv("RAWSEQ_COST_POLICY", "inverse_gt").strip()
THRESHOLD_BPS = float(os.getenv("RAWSEQ_COST_THRESHOLD_BPS", "0.2"))
COST_BPS_LIST_ENV = os.getenv("RAWSEQ_COST_BPS_LIST", "0,0.05,0.1,0.25,0.5,1.0")
TEST_FRAC = float(os.getenv("RAWSEQ_COST_TEST_FRAC", "0.20"))

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "realtime"
    / PRIMARY_VENUE
    / f"{SYMBOL}_tiny_price_prediction_evaluation_rows_with_rawseq_path_v1.csv"
)
OUTPUT_PATH = Path(
    os.getenv(
        "RAWSEQ_COST_OUTPUT_PATH",
        PROJECT_ROOT
        / "data"
        / "realtime"
        / PRIMARY_VENUE
        / f"{SYMBOL}_rawseq_frozen_shadow_cost_sensitivity.csv",
    )
)
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH

REQUIRED_COLUMNS = [
    "timestamp",
    "time",
    "rawseq_path_pred_horizon_return_bps",
    "rawseq_path_actual_horizon_return_bps",
]
PRED_COLUMN = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COLUMN = "rawseq_path_actual_horizon_return_bps"


def parse_costs(text: str) -> list[float]:
    costs: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        costs.append(float(item))
    return costs


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


def load_test_frame() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input not found: {INPUT_PATH}")
    if not 0.0 < TEST_FRAC <= 1.0:
        raise SystemExit(f"RAWSEQ_COST_TEST_FRAC must be in (0, 1], got {TEST_FRAC}")

    try:
        frame = pd.read_csv(INPUT_PATH, usecols=REQUIRED_COLUMNS, low_memory=False)
    except ValueError as exc:
        raise SystemExit(f"Input missing required columns: {exc}") from exc

    if frame.empty:
        raise SystemExit(f"Input has no rows: {INPUT_PATH}")

    split_at = int(len(frame) * (1.0 - TEST_FRAC))
    test = frame.iloc[split_at:].copy()
    test[PRED_COLUMN] = pd.to_numeric(test[PRED_COLUMN], errors="coerce")
    test[ACTUAL_COLUMN] = pd.to_numeric(test[ACTUAL_COLUMN], errors="coerce")
    test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=[PRED_COLUMN, ACTUAL_COLUMN])
    if test.empty:
        raise SystemExit("Test split has no finite prediction/actual rows.")
    return test


def gross_returns_for_policy(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    if POLICY == "inverse_gt":
        mask = pred > THRESHOLD_BPS
        gross = -actual[mask]
    elif POLICY == "direct_gt":
        mask = pred > THRESHOLD_BPS
        gross = actual[mask]
    elif POLICY == "inverse_directional_abs_gt":
        mask = np.abs(pred) > THRESHOLD_BPS
        gross = -np.sign(pred[mask]) * actual[mask]
    else:
        raise SystemExit(
            "RAWSEQ_COST_POLICY must be one of: inverse_gt, direct_gt, inverse_directional_abs_gt"
        )
    gross = np.asarray(gross, dtype="float64")
    return gross[np.isfinite(gross)]


def summarize_cost(
    cost_bps: float,
    gross: np.ndarray,
    first_time: str,
    last_time: str,
    test_rows_total: int,
) -> dict[str, Any]:
    net = gross - cost_bps
    selected_rows = int(len(gross))
    return {
        "symbol": SYMBOL,
        "venue": PRIMARY_VENUE,
        "policy": POLICY,
        "threshold_bps": THRESHOLD_BPS,
        "cost_bps": cost_bps,
        "selected_rows": selected_rows,
        "avg_gross_bps": float(np.mean(gross)) if selected_rows else math.nan,
        "avg_net_bps": float(np.mean(net)) if selected_rows else math.nan,
        "cum_gross_bps": float(np.sum(gross)) if selected_rows else 0.0,
        "cum_net_bps": float(np.sum(net)) if selected_rows else 0.0,
        "win_rate_gross": float(np.mean(gross > 0.0)) if selected_rows else math.nan,
        "win_rate_net": float(np.mean(net > 0.0)) if selected_rows else math.nan,
        "max_dip_gross_bps": max_dip_bps(gross),
        "max_dip_net_bps": max_dip_bps(net),
        "first_time": first_time,
        "last_time": last_time,
        "test_rows_total": test_rows_total,
        "input_path": str(INPUT_PATH),
    }


def print_table(report: pd.DataFrame) -> None:
    print("rawseq_frozen_shadow_cost_sensitivity")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Symbol: {SYMBOL}")
    print(f"Venue: {PRIMARY_VENUE}")
    print(f"Policy: {POLICY}")
    print(f"Threshold bps: {THRESHOLD_BPS:g}")
    print(f"Test frac: {TEST_FRAC:g}")
    print()

    columns = [
        "cost_bps",
        "selected_rows",
        "avg_gross_bps",
        "avg_net_bps",
        "cum_net_bps",
        "win_rate_net",
        "max_dip_net_bps",
    ]
    widths = {
        "cost_bps": 9,
        "selected_rows": 13,
        "avg_gross_bps": 14,
        "avg_net_bps": 12,
        "cum_net_bps": 12,
        "win_rate_net": 12,
        "max_dip_net_bps": 16,
    }
    print(" ".join(column.ljust(widths[column]) for column in columns))
    print(" ".join("-" * widths[column] for column in columns))
    for _, row in report.iterrows():
        values = {
            "cost_bps": f"{row['cost_bps']:.4g}",
            "selected_rows": int(row["selected_rows"]),
            "avg_gross_bps": f"{finite_or_nan(row['avg_gross_bps']):.4f}",
            "avg_net_bps": f"{finite_or_nan(row['avg_net_bps']):.4f}",
            "cum_net_bps": f"{finite_or_nan(row['cum_net_bps']):.2f}",
            "win_rate_net": f"{finite_or_nan(row['win_rate_net']):.4f}",
            "max_dip_net_bps": f"{finite_or_nan(row['max_dip_net_bps']):.2f}",
        }
        print(" ".join(str(values[column]).ljust(widths[column]) for column in columns))


def main() -> None:
    costs = parse_costs(COST_BPS_LIST_ENV)
    if not costs:
        raise SystemExit("RAWSEQ_COST_BPS_LIST did not contain any costs.")

    test = load_test_frame()
    pred = test[PRED_COLUMN].to_numpy(dtype="float64")
    actual = test[ACTUAL_COLUMN].to_numpy(dtype="float64")
    gross = gross_returns_for_policy(pred, actual)
    first_time = str(test["time"].iloc[0])
    last_time = str(test["time"].iloc[-1])

    rows = [
        summarize_cost(
            cost_bps=cost,
            gross=gross,
            first_time=first_time,
            last_time=last_time,
            test_rows_total=len(test),
        )
        for cost in costs
    ]
    report = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)
    print_table(report)


if __name__ == "__main__":
    main()
