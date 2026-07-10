#!/usr/bin/env python3
"""Evaluate frozen rawseq shadow predictions with timestamp-joined dynamic costs.

Read-only except for writing the dynamic cost reports.
"""

from __future__ import annotations

import math
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYMBOL = os.getenv("SYMBOL", os.getenv("RAWSEQ_DYNAMIC_SYMBOL", "SOLUSDT")).strip().upper()
VENUE = os.getenv("PRIMARY_VENUE", os.getenv("RAWSEQ_DYNAMIC_VENUE", "kraken")).strip().lower()
POLICY = os.getenv("RAWSEQ_DYNAMIC_POLICY", "inverse_gt").strip().lower()
THRESHOLD_BPS = float(os.getenv("RAWSEQ_DYNAMIC_THRESHOLD_BPS", "0.3"))
TEST_FRAC = float(os.getenv("RAWSEQ_DYNAMIC_TEST_FRAC", "0.20"))
FEE_BPS = float(os.getenv("RAWSEQ_DYNAMIC_FEE_BPS", "0.0"))
SPREAD_PERCENT_TO_BPS = float(os.getenv("RAWSEQ_DYNAMIC_SPREAD_PERCENT_TO_BPS", "100.0"))
MISSING_FLOW_PENALTY_BPS = float(os.getenv("RAWSEQ_DYNAMIC_MISSING_FLOW_PENALTY_BPS", "0.10"))
THIN_10BPS_DEPTH = float(os.getenv("RAWSEQ_DYNAMIC_THIN_10BPS_DEPTH", "1000"))
THIN_25BPS_DEPTH = float(os.getenv("RAWSEQ_DYNAMIC_THIN_25BPS_DEPTH", "5000"))
DEPTH_PENALTY_BPS = float(os.getenv("RAWSEQ_DYNAMIC_DEPTH_PENALTY_BPS", "0.05"))
OUTCOME_DELAY_ROWS = int(os.getenv("RAWSEQ_DYNAMIC_OUTCOME_DELAY_ROWS", "3"))
MIN_SELECTED_ROWS = int(os.getenv("RAWSEQ_DYNAMIC_MIN_SELECTED_ROWS", "500"))

ANNOTATED_PATH = Path(
    os.getenv(
        "RAWSEQ_DYNAMIC_ANNOTATED_PATH",
        PROJECT_ROOT
        / "data"
        / "realtime"
        / VENUE
        / f"{SYMBOL}_tiny_price_prediction_evaluation_rows_with_rawseq_path_v1.csv",
    )
)
FLOW_PATH = Path(
    os.getenv(
        "RAWSEQ_DYNAMIC_FLOW_PATH",
        PROJECT_ROOT / "data" / "realtime" / VENUE / f"{SYMBOL}_10s_flow.csv",
    )
)
OUTPUT_PATH = Path(
    os.getenv(
        "RAWSEQ_DYNAMIC_OUTPUT_PATH",
        PROJECT_ROOT
        / "data"
        / "realtime"
        / VENUE
        / f"{SYMBOL}_rawseq_frozen_shadow_dynamic_costs.csv",
    )
)
for path_name in ["ANNOTATED_PATH", "FLOW_PATH", "OUTPUT_PATH"]:
    path = globals()[path_name]
    if not path.is_absolute():
        globals()[path_name] = PROJECT_ROOT / path
TEXT_OUTPUT_PATH = OUTPUT_PATH.with_suffix(".txt")

PRED_COLUMN = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COLUMN = "rawseq_path_actual_horizon_return_bps"
ANNOTATED_COLUMNS = ["timestamp", "time", PRED_COLUMN, ACTUAL_COLUMN]
FLOW_COLUMNS = [
    "timestamp",
    "time",
    "spread_percent",
    "bid_depth_10bps",
    "ask_depth_10bps",
    "bid_depth_25bps",
    "ask_depth_25bps",
]


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except Exception:
        return math.nan


def max_dip_bps(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    cumulative = np.cumsum(values)
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def load_annotated_test() -> pd.DataFrame:
    if not ANNOTATED_PATH.exists():
        raise SystemExit(f"Annotated input not found: {ANNOTATED_PATH}")
    if not 0.0 < TEST_FRAC <= 1.0:
        raise SystemExit(f"RAWSEQ_DYNAMIC_TEST_FRAC must be in (0, 1], got {TEST_FRAC}")
    try:
        frame = pd.read_csv(ANNOTATED_PATH, usecols=ANNOTATED_COLUMNS, low_memory=False)
    except ValueError as exc:
        raise SystemExit(f"Annotated input missing required columns: {exc}") from exc
    if frame.empty:
        raise SystemExit(f"Annotated input has no rows: {ANNOTATED_PATH}")

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


def load_flow() -> pd.DataFrame:
    if not FLOW_PATH.exists():
        raise SystemExit(f"Flow input not found: {FLOW_PATH}")
    flow = pd.read_csv(FLOW_PATH, usecols=lambda column: column in FLOW_COLUMNS, low_memory=False)
    if "timestamp" not in flow.columns:
        raise SystemExit(f"Flow input missing timestamp column: {FLOW_PATH}")
    flow["timestamp"] = pd.to_numeric(flow["timestamp"], errors="coerce")
    flow = flow.dropna(subset=["timestamp"]).copy()
    flow["timestamp"] = flow["timestamp"].astype("int64")
    for column in FLOW_COLUMNS:
        if column not in flow.columns:
            flow[column] = np.nan
    return flow.drop_duplicates("timestamp", keep="last")


def join_flow(test: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    joined = test.merge(
        flow,
        on="timestamp",
        how="left",
        suffixes=("_annotated", "_flow"),
        indicator=True,
    )
    joined["flow_joined"] = joined["_merge"].eq("both")
    joined = joined.drop(columns=["_merge"])
    if "time_annotated" in joined.columns:
        joined["time"] = joined["time_annotated"]
    return joined


def policy_mask_and_gross(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
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
            "RAWSEQ_DYNAMIC_POLICY must be one of: inverse_gt, direct_gt, inverse_directional_abs_gt"
        )
    return np.asarray(mask, dtype=bool) & np.isfinite(gross), np.asarray(gross, dtype="float64")


def base_cost_components(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    spread_percent = pd.to_numeric(frame["spread_percent"], errors="coerce").clip(lower=0.0)
    out["half_spread_bps"] = spread_percent.fillna(0.0) * SPREAD_PERCENT_TO_BPS * 0.5
    out["missing_flow_penalty_bps"] = np.where(frame["flow_joined"], 0.0, MISSING_FLOW_PENALTY_BPS)

    bid10 = pd.to_numeric(frame["bid_depth_10bps"], errors="coerce")
    ask10 = pd.to_numeric(frame["ask_depth_10bps"], errors="coerce")
    bid25 = pd.to_numeric(frame["bid_depth_25bps"], errors="coerce")
    ask25 = pd.to_numeric(frame["ask_depth_25bps"], errors="coerce")
    min10 = pd.concat([bid10, ask10], axis=1).min(axis=1)
    min25 = pd.concat([bid25, ask25], axis=1).min(axis=1)
    out["thin_10bps_depth"] = min10.lt(THIN_10BPS_DEPTH) | min10.isna()
    out["thin_25bps_depth"] = min25.lt(THIN_25BPS_DEPTH) | min25.isna()
    out["depth_penalty_bps"] = (
        out["thin_10bps_depth"].astype(float) * DEPTH_PENALTY_BPS
        + out["thin_25bps_depth"].astype(float) * DEPTH_PENALTY_BPS
    )
    return out


def scenario_costs(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    components = base_cost_components(frame)
    fixed = FEE_BPS + components["half_spread_bps"] + components["missing_flow_penalty_bps"]
    return {
        "mild": (fixed + 0.02).to_numpy(dtype="float64"),
        "base": (fixed + 0.05).to_numpy(dtype="float64"),
        "harsh": (fixed + 0.10).to_numpy(dtype="float64"),
        "depth_stressed": (fixed + 0.05 + components["depth_penalty_bps"]).to_numpy(dtype="float64"),
    }


def release_due_outcomes(pending: deque[tuple[int, float]], idx: int) -> list[float]:
    released: list[float] = []
    while pending and pending[0][0] <= idx:
        _, net_return = pending.popleft()
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
    pending: deque[tuple[int, float]] = deque()
    for idx, selected in enumerate(base_mask):
        for net_return in release_due_outcomes(pending, idx):
            if net_return < 0.0:
                cooldown_remaining = max(cooldown_remaining, cooldown_signals)
        if not selected:
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        keep[idx] = True
        pending.append((idx + outcome_delay_rows, float(net_all[idx])))
    return keep


def max_signal_density_mask(base_mask: np.ndarray, window_rows: int, max_signals: int) -> np.ndarray:
    keep = np.zeros(len(base_mask), dtype=bool)
    prior_selected: deque[int] = deque()
    for idx, selected in enumerate(base_mask):
        while prior_selected and prior_selected[0] <= idx - window_rows:
            prior_selected.popleft()
        if selected:
            if len(prior_selected) < max_signals:
                keep[idx] = True
            prior_selected.append(idx)
    return keep


def overlay_masks(base_mask: np.ndarray, net_all: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    return [
        ("baseline", "none", base_mask),
        ("max_signal_density", "prior_rows=50;max_prior_selected=10", max_signal_density_mask(base_mask, 50, 10)),
        ("max_signal_density", "prior_rows=100;max_prior_selected=10", max_signal_density_mask(base_mask, 100, 10)),
        (
            "cooldown_after_loss",
            f"cooldown_selected_signals=5;outcome_delay_rows={OUTCOME_DELAY_ROWS}",
            cooldown_after_loss_mask(base_mask, net_all, 5, OUTCOME_DELAY_ROWS),
        ),
    ]


def summarize_selection(
    scenario: str,
    filter_name: str,
    filter_params: str,
    selected_mask: np.ndarray,
    gross_all: np.ndarray,
    cost_all: np.ndarray,
    joined: pd.DataFrame,
) -> dict[str, Any]:
    selected_mask = selected_mask & np.isfinite(gross_all) & np.isfinite(cost_all)
    gross = gross_all[selected_mask]
    costs = cost_all[selected_mask]
    net = gross - costs
    selected_rows = int(len(net))
    return {
        "symbol": SYMBOL,
        "venue": VENUE,
        "policy": POLICY,
        "threshold_bps": THRESHOLD_BPS,
        "scenario": scenario,
        "filter_name": filter_name,
        "filter_params": filter_params,
        "fee_bps": FEE_BPS,
        "selected_rows": selected_rows,
        "avg_gross_bps": float(np.mean(gross)) if selected_rows else math.nan,
        "avg_dynamic_cost_bps": float(np.mean(costs)) if selected_rows else math.nan,
        "p95_dynamic_cost_bps": float(np.quantile(costs, 0.95)) if selected_rows else math.nan,
        "avg_net_bps": float(np.mean(net)) if selected_rows else math.nan,
        "cum_net_bps": float(np.sum(net)) if selected_rows else 0.0,
        "win_rate_net": float(np.mean(net > 0.0)) if selected_rows else math.nan,
        "max_dip_net_bps": max_dip_bps(net),
        "flow_join_rate_all_test": float(joined["flow_joined"].mean()) if len(joined) else math.nan,
        "flow_missing_rows_all_test": int((~joined["flow_joined"]).sum()) if len(joined) else 0,
        "enough_selected_rows": selected_rows >= MIN_SELECTED_ROWS,
    }


def build_report(joined: pd.DataFrame) -> pd.DataFrame:
    base_mask, gross_all = policy_mask_and_gross(joined)
    rows: list[dict[str, Any]] = []
    for scenario, cost_all in scenario_costs(joined).items():
        net_all = gross_all - cost_all
        for filter_name, filter_params, selected_mask in overlay_masks(base_mask, net_all):
            rows.append(
                summarize_selection(
                    scenario,
                    filter_name,
                    filter_params,
                    selected_mask,
                    gross_all,
                    cost_all,
                    joined,
                )
            )
    return pd.DataFrame(rows)


def render_text(report: pd.DataFrame) -> str:
    base = report[report["scenario"].eq("base")].copy()
    pass_rows = base[(base["cum_net_bps"] > 0.0) & (base["enough_selected_rows"].astype(bool))]
    status = "PASS" if not pass_rows.empty else "FAIL"
    lines = [
        "Rawseq Frozen Shadow Dynamic Costs",
        "",
        f"Status: {status}",
        f"Annotated input: {ANNOTATED_PATH}",
        f"Flow input: {FLOW_PATH}",
        f"Policy: {POLICY}",
        f"Threshold bps: {THRESHOLD_BPS:g}",
        f"Fee bps: {FEE_BPS:g}",
        f"Spread percent to bps multiplier: {SPREAD_PERCENT_TO_BPS:g}",
        f"Missing flow penalty bps: {MISSING_FLOW_PENALTY_BPS:g}",
        f"Depth stress thresholds: 10bps<{THIN_10BPS_DEPTH:g}, 25bps<{THIN_25BPS_DEPTH:g}",
        "",
        "Top Rows By Scenario",
        "  scenario        filter                selected avg_cost avg_net cum_net max_dip status",
        "  --------------- --------------------- -------- -------- ------- ------- ------- ------",
    ]
    for _, row in report.sort_values(
        ["scenario", "cum_net_bps", "max_dip_net_bps"],
        ascending=[True, False, False],
    ).groupby("scenario", sort=True).head(4).iterrows():
        row_status = "PASS" if row["cum_net_bps"] > 0.0 and bool(row["enough_selected_rows"]) else "FAIL"
        lines.append(
            "  "
            + " ".join(
                [
                    str(row["scenario"])[:15].ljust(15),
                    str(row["filter_name"])[:21].ljust(21),
                    str(int(row["selected_rows"])).rjust(8),
                    f"{finite_or_nan(row['avg_dynamic_cost_bps']):.4f}".rjust(8),
                    f"{finite_or_nan(row['avg_net_bps']):.4f}".rjust(7),
                    f"{finite_or_nan(row['cum_net_bps']):.2f}".rjust(7),
                    f"{finite_or_nan(row['max_dip_net_bps']):.2f}".rjust(7),
                    row_status.ljust(6),
                ]
            )
        )
    lines.extend(["", f"CSV report: {OUTPUT_PATH}", f"Text report: {TEXT_OUTPUT_PATH}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    annotated = load_annotated_test()
    flow = load_flow()
    joined = join_flow(annotated, flow)
    report = build_report(joined)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)
    text = render_text(report)
    TEXT_OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
