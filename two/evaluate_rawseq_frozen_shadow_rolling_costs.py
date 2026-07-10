#!/usr/bin/env python3
"""Evaluate frozen rawseq shadow predictions over rolling cost windows.

Read-only except for writing the rolling cost CSV.
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
POLICY = os.getenv("RAWSEQ_ROLLING_POLICY", "inverse_gt").strip()
THRESHOLD_BPS = float(os.getenv("RAWSEQ_ROLLING_THRESHOLD_BPS", "0.3"))
COST_BPS = float(os.getenv("RAWSEQ_ROLLING_COST_BPS", "0.1"))
WINDOW_HOURS_LIST_ENV = os.getenv("RAWSEQ_ROLLING_WINDOW_HOURS_LIST", "6,12,24")
TEST_FRAC = float(os.getenv("RAWSEQ_ROLLING_TEST_FRAC", "0.20"))

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "realtime"
    / PRIMARY_VENUE
    / f"{SYMBOL}_tiny_price_prediction_evaluation_rows_with_rawseq_path_v1.csv"
)
OUTPUT_PATH = Path(
    os.getenv(
        "RAWSEQ_ROLLING_OUTPUT_PATH",
        PROJECT_ROOT
        / "data"
        / "realtime"
        / PRIMARY_VENUE
        / f"{SYMBOL}_rawseq_frozen_shadow_rolling_costs.csv",
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


def parse_window_hours(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise SystemExit(f"Window hours must be positive, got {value}")
        values.append(value)
    return values


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
        raise SystemExit(f"RAWSEQ_ROLLING_TEST_FRAC must be in (0, 1], got {TEST_FRAC}")

    try:
        frame = pd.read_csv(INPUT_PATH, usecols=REQUIRED_COLUMNS, low_memory=False)
    except ValueError as exc:
        raise SystemExit(f"Input missing required columns: {exc}") from exc

    if frame.empty:
        raise SystemExit(f"Input has no rows: {INPUT_PATH}")

    split_at = int(len(frame) * (1.0 - TEST_FRAC))
    test = frame.iloc[split_at:].copy()
    test["timestamp"] = pd.to_numeric(test["timestamp"], errors="coerce")
    test[PRED_COLUMN] = pd.to_numeric(test[PRED_COLUMN], errors="coerce")
    test[ACTUAL_COLUMN] = pd.to_numeric(test[ACTUAL_COLUMN], errors="coerce")
    test = test.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["timestamp", PRED_COLUMN, ACTUAL_COLUMN]
    )
    if test.empty:
        raise SystemExit("Test split has no finite timestamp/prediction/actual rows.")
    return test.sort_values("timestamp").reset_index(drop=True)


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
            "RAWSEQ_ROLLING_POLICY must be one of: inverse_gt, direct_gt, inverse_directional_abs_gt"
        )
    gross = np.asarray(gross, dtype="float64")
    return gross[np.isfinite(gross)]


def summarize_window(window: pd.DataFrame, window_hours: float) -> dict[str, Any]:
    pred = window[PRED_COLUMN].to_numpy(dtype="float64")
    actual = window[ACTUAL_COLUMN].to_numpy(dtype="float64")
    gross = gross_returns_for_policy(pred, actual)
    net = gross - COST_BPS
    selected_rows = int(len(gross))
    return {
        "symbol": SYMBOL,
        "venue": PRIMARY_VENUE,
        "policy": POLICY,
        "threshold_bps": THRESHOLD_BPS,
        "cost_bps": COST_BPS,
        "window_hours": window_hours,
        "window_start_time": str(window["time"].iloc[0]),
        "window_end_time": str(window["time"].iloc[-1]),
        "total_rows": int(len(window)),
        "selected_rows": selected_rows,
        "avg_gross_bps": float(np.mean(gross)) if selected_rows else math.nan,
        "avg_net_bps": float(np.mean(net)) if selected_rows else math.nan,
        "cum_gross_bps": float(np.sum(gross)) if selected_rows else 0.0,
        "cum_net_bps": float(np.sum(net)) if selected_rows else 0.0,
        "win_rate_gross": float(np.mean(gross > 0.0)) if selected_rows else math.nan,
        "win_rate_net": float(np.mean(net > 0.0)) if selected_rows else math.nan,
        "max_dip_gross_bps": max_dip_bps(gross),
        "max_dip_net_bps": max_dip_bps(net),
    }


def build_report(test: pd.DataFrame, window_hours_list: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    first_timestamp = float(test["timestamp"].iloc[0])
    elapsed_ms = test["timestamp"] - first_timestamp

    for window_hours in window_hours_list:
        window_ms = window_hours * 60.0 * 60.0 * 1000.0
        window_ids = np.floor(elapsed_ms / window_ms).astype("int64")
        for _, window in test.groupby(window_ids, sort=True):
            if len(window) == 0:
                continue
            rows.append(summarize_window(window.reset_index(drop=True), window_hours))

    return pd.DataFrame(rows)


def print_summary(report: pd.DataFrame) -> None:
    print("rawseq_frozen_shadow_rolling_costs")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Symbol: {SYMBOL}")
    print(f"Venue: {PRIMARY_VENUE}")
    print(f"Policy: {POLICY}")
    print(f"Threshold bps: {THRESHOLD_BPS:g}")
    print(f"Cost bps: {COST_BPS:g}")
    print(f"Test frac: {TEST_FRAC:g}")
    print()

    columns = [
        "window_hours",
        "windows",
        "positive_windows",
        "median_avg_net",
        "total_cum_net",
        "worst_window_cum_net",
        "worst_window_max_dip",
        "total_selected_rows",
    ]
    widths = {
        "window_hours": 12,
        "windows": 8,
        "positive_windows": 16,
        "median_avg_net": 14,
        "total_cum_net": 13,
        "worst_window_cum_net": 20,
        "worst_window_max_dip": 20,
        "total_selected_rows": 19,
    }
    print(" ".join(column.ljust(widths[column]) for column in columns))
    print(" ".join("-" * widths[column] for column in columns))
    for window_hours, group in report.groupby("window_hours", sort=True):
        values = {
            "window_hours": f"{float(window_hours):g}",
            "windows": int(len(group)),
            "positive_windows": int((group["cum_net_bps"] > 0.0).sum()),
            "median_avg_net": f"{finite_or_nan(group['avg_net_bps'].median()):.4f}",
            "total_cum_net": f"{finite_or_nan(group['cum_net_bps'].sum()):.2f}",
            "worst_window_cum_net": f"{finite_or_nan(group['cum_net_bps'].min()):.2f}",
            "worst_window_max_dip": f"{finite_or_nan(group['max_dip_net_bps'].min()):.2f}",
            "total_selected_rows": int(group["selected_rows"].sum()),
        }
        print(" ".join(str(values[column]).ljust(widths[column]) for column in columns))


def main() -> None:
    window_hours_list = parse_window_hours(WINDOW_HOURS_LIST_ENV)
    if not window_hours_list:
        raise SystemExit("RAWSEQ_ROLLING_WINDOW_HOURS_LIST did not contain any windows.")

    test = load_test_frame()
    report = build_report(test, window_hours_list)
    if report.empty:
        raise SystemExit("No rolling windows were produced.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)
    print_summary(report)


if __name__ == "__main__":
    main()
