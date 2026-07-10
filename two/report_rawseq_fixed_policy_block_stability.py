#!/usr/bin/env python3
"""Evaluate fixed rawseq paper-shadow policy over contiguous time blocks.

Read-only except for writing the block stability reports.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYMBOL = os.getenv("SYMBOL", os.getenv("RAWSEQ_BLOCK_SYMBOL", "SOLUSDT")).strip().upper()
VENUE = os.getenv("PRIMARY_VENUE", os.getenv("RAWSEQ_BLOCK_VENUE", "kraken")).strip().lower()
POLICY = os.getenv("RAWSEQ_BLOCK_POLICY", "inverse_gt").strip().lower()
THRESHOLD_BPS = float(os.getenv("RAWSEQ_BLOCK_THRESHOLD_BPS", "0.3"))
COST_BPS = float(os.getenv("RAWSEQ_BLOCK_COST_BPS", "0.10"))
TEST_FRAC = float(os.getenv("RAWSEQ_BLOCK_TEST_FRAC", "0.20"))
WINDOW_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0]

INPUT_PATH = Path(
    os.getenv(
        "RAWSEQ_BLOCK_INPUT_PATH",
        PROJECT_ROOT
        / "data"
        / "realtime"
        / VENUE
        / f"{SYMBOL}_tiny_price_prediction_evaluation_rows_with_rawseq_path_v1.csv",
    )
)
if not INPUT_PATH.is_absolute():
    INPUT_PATH = PROJECT_ROOT / INPUT_PATH

OUTPUT_PATH = Path(
    os.getenv(
        "RAWSEQ_BLOCK_OUTPUT_PATH",
        PROJECT_ROOT
        / "data"
        / "realtime"
        / VENUE
        / f"{SYMBOL}_rawseq_fixed_policy_block_stability.csv",
    )
)
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH
TEXT_OUTPUT_PATH = OUTPUT_PATH.with_suffix(".txt")

PRED_COLUMN = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COLUMN = "rawseq_path_actual_horizon_return_bps"
REQUIRED_COLUMNS = ["timestamp", "time", PRED_COLUMN, ACTUAL_COLUMN]


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except Exception:
        return math.nan


def max_dip_bps(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype="float64")
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return math.nan
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def load_test_frame() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input not found: {INPUT_PATH}")
    if not 0.0 < TEST_FRAC <= 1.0:
        raise SystemExit(f"RAWSEQ_BLOCK_TEST_FRAC must be in (0, 1], got {TEST_FRAC}")

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
    test["timestamp"] = test["timestamp"].astype("int64")
    return test.sort_values("timestamp").reset_index(drop=True)


def policy_gross_all(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred = frame[PRED_COLUMN].to_numpy(dtype="float64")
    actual = frame[ACTUAL_COLUMN].to_numpy(dtype="float64")
    if POLICY == "inverse_gt":
        mask = pred > THRESHOLD_BPS
        gross = -actual
    elif POLICY == "direct_gt":
        mask = pred > THRESHOLD_BPS
        gross = actual
    elif POLICY == "inverse_directional_abs_gt":
        mask = np.abs(pred) > THRESHOLD_BPS
        gross = -np.sign(pred) * actual
    else:
        raise SystemExit(
            "RAWSEQ_BLOCK_POLICY must be one of: inverse_gt, direct_gt, inverse_directional_abs_gt"
        )
    mask = np.asarray(mask, dtype=bool) & np.isfinite(gross)
    return mask, np.asarray(gross, dtype="float64")


def summarize_block(block: pd.DataFrame, block_type: str, block_id: str) -> dict[str, Any]:
    selected = block[block["selected"]].copy()
    net = selected["net_bps"].to_numpy(dtype="float64") if not selected.empty else np.array([])
    gross = selected["gross_bps"].to_numpy(dtype="float64") if not selected.empty else np.array([])
    selected_rows = int(len(selected))
    return {
        "symbol": SYMBOL,
        "venue": VENUE,
        "policy": POLICY,
        "threshold_bps": THRESHOLD_BPS,
        "cost_bps": COST_BPS,
        "test_frac": TEST_FRAC,
        "block_type": block_type,
        "block_id": block_id,
        "block_start_time": str(block["time"].iloc[0]),
        "block_end_time": str(block["time"].iloc[-1]),
        "block_start_timestamp": int(block["timestamp"].iloc[0]),
        "block_end_timestamp": int(block["timestamp"].iloc[-1]),
        "total_rows": int(len(block)),
        "selected_rows": selected_rows,
        "avg_gross_bps": float(np.mean(gross)) if selected_rows else math.nan,
        "avg_net_bps": float(np.mean(net)) if selected_rows else math.nan,
        "cum_gross_bps": float(np.sum(gross)) if selected_rows else 0.0,
        "cum_net_bps": float(np.sum(net)) if selected_rows else 0.0,
        "win_rate_net": float(np.mean(net > 0.0)) if selected_rows else math.nan,
        "max_dip_net_bps": max_dip_bps(net),
    }


def add_window_blocks(frame: pd.DataFrame, rows: list[dict[str, Any]], window_hours: float) -> None:
    first_ts = float(frame["timestamp"].iloc[0])
    window_ms = window_hours * 60.0 * 60.0 * 1000.0
    block_ids = np.floor((frame["timestamp"] - first_ts) / window_ms).astype("int64")
    for block_id, block in frame.groupby(block_ids, sort=True):
        rows.append(summarize_block(block.reset_index(drop=True), f"{window_hours:g}h", str(block_id)))


def add_day_blocks(frame: pd.DataFrame, rows: list[dict[str, Any]]) -> None:
    day_ids = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    for day_id, block in frame.groupby(day_ids, sort=True):
        rows.append(summarize_block(block.reset_index(drop=True), "utc_day", str(day_id)))


def build_blocks(test: pd.DataFrame) -> pd.DataFrame:
    frame = test.copy()
    selected, gross = policy_gross_all(frame)
    frame["selected"] = selected
    frame["gross_bps"] = gross
    frame["net_bps"] = frame["gross_bps"] - COST_BPS

    rows: list[dict[str, Any]] = []
    for window_hours in WINDOW_HOURS:
        add_window_blocks(frame, rows, window_hours)
    add_day_blocks(frame, rows)
    return pd.DataFrame(rows)


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    valid = group[group["selected_rows"] > 0].copy()
    total_cum = float(group["cum_net_bps"].sum()) if not group.empty else 0.0
    positive_blocks = int((group["cum_net_bps"] > 0.0).sum())
    blocks = int(len(group))
    positive_profit = group.loc[group["cum_net_bps"] > 0.0, "cum_net_bps"].sum()
    max_positive = group.loc[group["cum_net_bps"] > 0.0, "cum_net_bps"].max()
    contribution_fraction = (
        float(max_positive / positive_profit)
        if positive_profit and math.isfinite(float(max_positive))
        else math.nan
    )
    worst_block = float(group["cum_net_bps"].min()) if blocks else math.nan
    positive_fraction = positive_blocks / blocks if blocks else math.nan
    pass_block_type = (
        total_cum > 0.0
        and math.isfinite(positive_fraction)
        and positive_fraction > 0.5
        and (not math.isfinite(contribution_fraction) or contribution_fraction <= 0.50)
        and (not math.isfinite(worst_block) or abs(worst_block) < total_cum)
    )
    return {
        "block_type": str(group["block_type"].iloc[0]),
        "blocks": blocks,
        "active_blocks": int(len(valid)),
        "positive_blocks": positive_blocks,
        "positive_fraction": positive_fraction,
        "total_selected_rows": int(group["selected_rows"].sum()),
        "total_cum_net_bps": total_cum,
        "median_block_cum_net_bps": finite_or_nan(group["cum_net_bps"].median()),
        "worst_block_cum_net_bps": worst_block,
        "best_block_cum_net_bps": finite_or_nan(group["cum_net_bps"].max()),
        "max_positive_contribution_fraction": contribution_fraction,
        "status": "PASS" if pass_block_type else "FAIL",
    }


def build_summary(blocks: pd.DataFrame) -> pd.DataFrame:
    if blocks.empty:
        return pd.DataFrame()
    rows = [summarize_group(group) for _, group in blocks.groupby("block_type", sort=False)]
    return pd.DataFrame(rows)


def render_text(summary: pd.DataFrame) -> str:
    overall_status = "FAIL" if summary.empty or (summary["status"] == "FAIL").any() else "PASS"
    lines = [
        "Rawseq Fixed Policy Block Stability",
        "",
        f"Status: {overall_status}",
        f"Input: {INPUT_PATH}",
        f"Policy: {POLICY}",
        f"Threshold bps: {THRESHOLD_BPS:g}",
        f"Cost bps: {COST_BPS:g}",
        f"Test frac: {TEST_FRAC:g}",
        "",
        "Block Summary",
        "  block_type blocks positive_fraction total_cum_net worst_block max_contrib status",
        "  ---------- ------ ----------------- ------------- ----------- ----------- ------",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "  "
            + " ".join(
                [
                    str(row["block_type"])[:10].ljust(10),
                    str(int(row["blocks"])).rjust(6),
                    f"{finite_or_nan(row['positive_fraction']):.4f}".rjust(17),
                    f"{finite_or_nan(row['total_cum_net_bps']):.2f}".rjust(13),
                    f"{finite_or_nan(row['worst_block_cum_net_bps']):.2f}".rjust(11),
                    f"{finite_or_nan(row['max_positive_contribution_fraction']):.4f}".rjust(11),
                    str(row["status"]).ljust(6),
                ]
            )
        )
    lines.extend(["", f"CSV report: {OUTPUT_PATH}", f"Text report: {TEXT_OUTPUT_PATH}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    test = load_test_frame()
    blocks = build_blocks(test)
    summary = build_summary(blocks)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(OUTPUT_PATH, index=False)
    summary_path = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    text = render_text(summary)
    TEXT_OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
