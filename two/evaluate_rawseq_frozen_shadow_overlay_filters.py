#!/usr/bin/env python3
"""Evaluate read-only overlay filters for frozen rawseq shadow predictions."""

from __future__ import annotations

import math
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip()
POLICY = os.getenv("RAWSEQ_FILTER_POLICY", "inverse_gt").strip()
THRESHOLD_BPS = float(os.getenv("RAWSEQ_FILTER_THRESHOLD_BPS", "0.3"))
COST_BPS = float(os.getenv("RAWSEQ_FILTER_COST_BPS", "0.1"))
TEST_FRAC = float(os.getenv("RAWSEQ_FILTER_TEST_FRAC", "0.20"))

COOLDOWN_LIST_ENV = os.getenv("RAWSEQ_FILTER_COOLDOWN_LIST", "3,5,10")
DENSITY_WINDOW_LIST_ENV = os.getenv("RAWSEQ_FILTER_DENSITY_WINDOW_LIST", "50,100,200")
DENSITY_MAX_LIST_ENV = os.getenv("RAWSEQ_FILTER_DENSITY_MAX_LIST", "5,10,20")
RECENT_SELECTED_WINDOW_LIST_ENV = os.getenv("RAWSEQ_FILTER_RECENT_SELECTED_WINDOW_LIST", "20,50,100")
RECENT_DRAWDOWN_MIN_LIST_ENV = os.getenv("RAWSEQ_FILTER_RECENT_DRAWDOWN_MIN_LIST", "-25,-50,-100")
OUTCOME_DELAY_ROWS = int(os.getenv("RAWSEQ_FILTER_OUTCOME_DELAY_ROWS", "3"))

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "realtime"
    / PRIMARY_VENUE
    / f"{SYMBOL}_tiny_price_prediction_evaluation_rows_with_rawseq_path_v1.csv"
)
OUTPUT_PATH = Path(
    os.getenv(
        "RAWSEQ_FILTER_OUTPUT_PATH",
        PROJECT_ROOT
        / "data"
        / "realtime"
        / PRIMARY_VENUE
        / f"{SYMBOL}_rawseq_frozen_shadow_overlay_filters.csv",
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
MIN_SELECTED_ROWS = int(os.getenv("RAWSEQ_FILTER_MIN_SELECTED_ROWS", "500"))


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
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
        raise SystemExit(f"RAWSEQ_FILTER_TEST_FRAC must be in (0, 1], got {TEST_FRAC}")

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
    return test.reset_index(drop=True)


def policy_mask_and_gross(pred: np.ndarray, actual: np.ndarray, threshold_bps: float) -> tuple[np.ndarray, np.ndarray]:
    if POLICY == "inverse_gt":
        mask = pred > threshold_bps
        gross_all = -actual
    elif POLICY == "direct_gt":
        mask = pred > threshold_bps
        gross_all = actual
    elif POLICY == "inverse_directional_abs_gt":
        mask = np.abs(pred) > threshold_bps
        gross_all = -np.sign(pred) * actual
    else:
        raise SystemExit(
            "RAWSEQ_FILTER_POLICY must be one of: inverse_gt, direct_gt, inverse_directional_abs_gt"
        )
    mask = np.asarray(mask, dtype=bool)
    gross_all = np.asarray(gross_all, dtype="float64")
    mask &= np.isfinite(gross_all)
    return mask, gross_all


def release_due_outcomes(pending_outcomes: deque[tuple[int, float]], idx: int) -> list[float]:
    released: list[float] = []
    while pending_outcomes and pending_outcomes[0][0] <= idx:
        _, net_return = pending_outcomes.popleft()
        released.append(net_return)
    return released


def cooldown_after_loss_mask(
    base_mask: np.ndarray,
    net_all: np.ndarray,
    cooldown_signals: int,
    outcome_delay_rows: int,
) -> np.ndarray:
    keep = np.zeros(len(base_mask), dtype=bool)
    cooldown_remaining = 0
    pending_outcomes: deque[tuple[int, float]] = deque()
    for idx, selected in enumerate(base_mask):
        for net_return in release_due_outcomes(pending_outcomes, idx):
            if net_return < 0.0:
                cooldown_remaining = max(cooldown_remaining, cooldown_signals)
        if not selected:
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        keep[idx] = True
        pending_outcomes.append((idx + outcome_delay_rows, float(net_all[idx])))
    return keep


def max_signal_density_mask(base_mask: np.ndarray, window_rows: int, max_signals: int) -> np.ndarray:
    keep = np.zeros(len(base_mask), dtype=bool)
    prior_selected = deque()
    for idx, selected in enumerate(base_mask):
        while prior_selected and prior_selected[0] <= idx - window_rows:
            prior_selected.popleft()
        if selected:
            if len(prior_selected) < max_signals:
                keep[idx] = True
            prior_selected.append(idx)
    return keep


def recent_policy_drawdown_guard_mask(
    base_mask: np.ndarray,
    net_all: np.ndarray,
    selected_window: int,
    drawdown_min_bps: float,
    outcome_delay_rows: int,
) -> np.ndarray:
    keep = np.zeros(len(base_mask), dtype=bool)
    recent_selected_net: deque[float] = deque(maxlen=selected_window)
    pending_outcomes: deque[tuple[int, float]] = deque()
    for idx, selected in enumerate(base_mask):
        for net_return in release_due_outcomes(pending_outcomes, idx):
            recent_selected_net.append(net_return)
        if not selected:
            continue
        recent_sum = sum(recent_selected_net)
        if len(recent_selected_net) >= selected_window and recent_sum < drawdown_min_bps:
            continue
        keep[idx] = True
        pending_outcomes.append((idx + outcome_delay_rows, float(net_all[idx])))
    return keep


def summarize_filter(
    filter_name: str,
    filter_params: str,
    selected_mask: np.ndarray,
    gross_all: np.ndarray,
    baseline_cum: float,
    baseline_dip: float,
) -> dict[str, Any]:
    gross = gross_all[selected_mask]
    gross = gross[np.isfinite(gross)]
    net = gross - COST_BPS
    selected_rows = int(len(gross))
    cum_net = float(np.sum(net)) if selected_rows else 0.0
    max_dip = max_dip_bps(net)
    return {
        "symbol": SYMBOL,
        "venue": PRIMARY_VENUE,
        "policy": POLICY,
        "threshold_bps": THRESHOLD_BPS,
        "cost_bps": COST_BPS,
        "filter_name": filter_name,
        "filter_params": filter_params,
        "selected_rows": selected_rows,
        "avg_gross_bps": float(np.mean(gross)) if selected_rows else math.nan,
        "avg_net_bps": float(np.mean(net)) if selected_rows else math.nan,
        "cum_net_bps": cum_net,
        "win_rate_net": float(np.mean(net > 0.0)) if selected_rows else math.nan,
        "max_dip_net_bps": max_dip,
        "improvement_vs_baseline_cum": cum_net - baseline_cum,
        "improvement_vs_baseline_dip": max_dip - baseline_dip if math.isfinite(max_dip) else math.nan,
        "enough_selected_rows": selected_rows >= MIN_SELECTED_ROWS,
    }


def build_report(test: pd.DataFrame) -> pd.DataFrame:
    pred = test[PRED_COLUMN].to_numpy(dtype="float64")
    actual = test[ACTUAL_COLUMN].to_numpy(dtype="float64")
    base_mask, gross_all = policy_mask_and_gross(pred, actual, THRESHOLD_BPS)
    net_all = gross_all - COST_BPS

    baseline_gross = gross_all[base_mask]
    baseline_net = baseline_gross - COST_BPS
    baseline_cum = float(np.sum(baseline_net)) if len(baseline_net) else 0.0
    baseline_dip = max_dip_bps(baseline_net)

    rows: list[dict[str, Any]] = [
        summarize_filter("baseline", "none", base_mask, gross_all, baseline_cum, baseline_dip)
    ]

    for cooldown in parse_int_list(COOLDOWN_LIST_ENV):
        selected = cooldown_after_loss_mask(base_mask, net_all, cooldown, OUTCOME_DELAY_ROWS)
        rows.append(
            summarize_filter(
                "cooldown_after_loss",
                f"cooldown_selected_signals={cooldown};outcome_delay_rows={OUTCOME_DELAY_ROWS}",
                selected,
                gross_all,
                baseline_cum,
                baseline_dip,
            )
        )

    for window_rows in parse_int_list(DENSITY_WINDOW_LIST_ENV):
        for max_signals in parse_int_list(DENSITY_MAX_LIST_ENV):
            selected = max_signal_density_mask(base_mask, window_rows, max_signals)
            rows.append(
                summarize_filter(
                    "max_signal_density",
                    f"prior_rows={window_rows};max_prior_selected={max_signals}",
                    selected,
                    gross_all,
                    baseline_cum,
                    baseline_dip,
                )
            )

    for selected_window in parse_int_list(RECENT_SELECTED_WINDOW_LIST_ENV):
        for drawdown_min in parse_float_list(RECENT_DRAWDOWN_MIN_LIST_ENV):
            selected = recent_policy_drawdown_guard_mask(
                base_mask,
                net_all,
                selected_window,
                drawdown_min,
                OUTCOME_DELAY_ROWS,
            )
            rows.append(
                summarize_filter(
                    "recent_policy_drawdown_guard",
                    (
                        f"recent_selected={selected_window};min_sum_net_bps={drawdown_min:g};"
                        f"outcome_delay_rows={OUTCOME_DELAY_ROWS}"
                    ),
                    selected,
                    gross_all,
                    baseline_cum,
                    baseline_dip,
                )
            )

    for threshold in [0.3, 0.5, 0.75]:
        threshold_mask, threshold_gross_all = policy_mask_and_gross(pred, actual, threshold)
        rows.append(
            summarize_filter(
                "min_pred_threshold",
                f"threshold_bps={threshold:g}",
                threshold_mask,
                threshold_gross_all,
                baseline_cum,
                baseline_dip,
            )
        )

    report = pd.DataFrame(rows)
    report["positive_cum_net"] = report["cum_net_bps"] > 0.0
    return report.sort_values(
        ["positive_cum_net", "enough_selected_rows", "max_dip_net_bps", "cum_net_bps", "selected_rows"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def print_top(report: pd.DataFrame, limit: int = 20) -> None:
    print("rawseq_frozen_shadow_overlay_filters")
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
        "filter_name",
        "filter_params",
        "selected_rows",
        "avg_net_bps",
        "cum_net_bps",
        "win_rate_net",
        "max_dip_net_bps",
        "improvement_vs_baseline_cum",
        "improvement_vs_baseline_dip",
    ]
    widths = {
        "filter_name": 28,
        "filter_params": 42,
        "selected_rows": 8,
        "avg_net_bps": 10,
        "cum_net_bps": 10,
        "win_rate_net": 10,
        "max_dip_net_bps": 12,
        "improvement_vs_baseline_cum": 14,
        "improvement_vs_baseline_dip": 14,
    }
    print(" ".join(column[: widths[column]].ljust(widths[column]) for column in columns))
    print(" ".join("-" * widths[column] for column in columns))
    for _, row in report.head(limit).iterrows():
        values = {
            "filter_name": str(row["filter_name"])[: widths["filter_name"]],
            "filter_params": str(row["filter_params"])[: widths["filter_params"]],
            "selected_rows": str(int(row["selected_rows"])),
            "avg_net_bps": f"{finite_or_nan(row['avg_net_bps']):.4f}",
            "cum_net_bps": f"{finite_or_nan(row['cum_net_bps']):.2f}",
            "win_rate_net": f"{finite_or_nan(row['win_rate_net']):.4f}",
            "max_dip_net_bps": f"{finite_or_nan(row['max_dip_net_bps']):.2f}",
            "improvement_vs_baseline_cum": f"{finite_or_nan(row['improvement_vs_baseline_cum']):.2f}",
            "improvement_vs_baseline_dip": f"{finite_or_nan(row['improvement_vs_baseline_dip']):.2f}",
        }
        print(" ".join(values[column].ljust(widths[column]) for column in columns))


def main() -> None:
    test = load_test_frame()
    report = build_report(test)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)
    print_top(report)


if __name__ == "__main__":
    main()
