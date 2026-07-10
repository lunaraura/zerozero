#!/usr/bin/env python3
"""Evaluate dynamic execution costs for an isolated rawseq candidate probe.

Reads only artifacts inside the selected probe folder and writes reports there.
No training, promotion, champion mutation, or orders.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_candidate_shadow_probes"

PROBE_DIR_ENV = os.getenv("RAWSEQ_PROBE_DIR", "").strip()
THRESHOLD_BPS = float(os.getenv("RAWSEQ_DYNAMIC_THRESHOLD_BPS", "0.1"))
POLICY = os.getenv("RAWSEQ_DYNAMIC_POLICY", "inverse_gt").strip().lower()
TEST_FRAC = float(os.getenv("RAWSEQ_DYNAMIC_TEST_FRAC", "0.20"))

PRED_COLUMN = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COLUMN = "rawseq_path_actual_horizon_return_bps"
FLOW_COLUMNS = [
    "spread_percent",
    "bid_depth_10bps",
    "ask_depth_10bps",
    "bid_depth_25bps",
    "ask_depth_25bps",
    "total_trade_volume_10s",
    "trade_count_10s",
    "market_pressure_10s",
    "order_book_imbalance_10bps",
    "order_book_imbalance_25bps",
]
REQUIRED_COLUMNS = ["timestamp", "time", PRED_COLUMN, ACTUAL_COLUMN]


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except Exception:
        return math.nan


def fmt(value: Any, digits: int = 6) -> str:
    value = safe_float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def max_dip_bps(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    cumulative = np.cumsum(values)
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def resolve_probe_dir() -> Path:
    if PROBE_DIR_ENV:
        path = Path(PROBE_DIR_ENV).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise SystemExit(f"RAWSEQ_PROBE_DIR does not exist: {path}")
        return path.resolve()

    if not PROBE_ROOT.exists():
        raise SystemExit(f"Probe root does not exist: {PROBE_ROOT}")
    candidates = [
        path
        for path in PROBE_ROOT.iterdir()
        if path.is_dir() and (path / "annotated.csv").exists() and (path / "model_contract.json").exists()
    ]
    if not candidates:
        raise SystemExit(f"No probe folders with annotated.csv found under {PROBE_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"Required file missing: {path}")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def load_test_frame(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    columns = [column for column in REQUIRED_COLUMNS + FLOW_COLUMNS if column in header.columns]
    missing_required = [column for column in REQUIRED_COLUMNS if column not in header.columns]
    if missing_required:
        raise SystemExit(f"annotated.csv missing required columns: {missing_required}")
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    if frame.empty:
        raise SystemExit(f"annotated.csv has no rows: {path}")
    if not 0.0 < TEST_FRAC <= 1.0:
        raise SystemExit(f"RAWSEQ_DYNAMIC_TEST_FRAC must be in (0, 1], got {TEST_FRAC}")
    split_at = int(len(frame) * (1.0 - TEST_FRAC))
    test = frame.iloc[split_at:].copy()
    for column in [PRED_COLUMN, ACTUAL_COLUMN, "timestamp"]:
        test[column] = pd.to_numeric(test[column], errors="coerce")
    for column in FLOW_COLUMNS:
        if column in test.columns:
            test[column] = pd.to_numeric(test[column], errors="coerce")
    test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=[PRED_COLUMN, ACTUAL_COLUMN])
    if test.empty:
        raise SystemExit("Test split has no finite prediction/actual rows.")
    return test.reset_index(drop=True)


def selected_gross(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pred = pd.to_numeric(frame[PRED_COLUMN], errors="coerce")
    actual = pd.to_numeric(frame[ACTUAL_COLUMN], errors="coerce")
    if POLICY == "inverse_gt":
        mask = pred > THRESHOLD_BPS
        gross = -actual
    elif POLICY == "direct_gt":
        mask = pred > THRESHOLD_BPS
        gross = actual
    elif POLICY == "inverse_directional_abs_gt":
        mask = pred.abs() > THRESHOLD_BPS
        gross = -np.sign(pred) * actual
    else:
        raise SystemExit(
            "RAWSEQ_DYNAMIC_POLICY must be one of: inverse_gt, direct_gt, inverse_directional_abs_gt"
        )
    mask = mask & np.isfinite(gross)
    return mask, pd.Series(gross, index=frame.index, dtype="float64")


def half_spread_bps(frame: pd.DataFrame, default_bps: float = 0.10) -> pd.Series:
    if "spread_percent" not in frame.columns:
        return pd.Series(default_bps, index=frame.index, dtype="float64")
    spread_percent = pd.to_numeric(frame["spread_percent"], errors="coerce")
    cost = spread_percent * 100.0 / 2.0
    return cost.where(np.isfinite(cost) & (cost >= 0.0), default_bps).astype("float64")


def thin_depth_penalty(frame: pd.DataFrame) -> pd.Series:
    if "bid_depth_10bps" not in frame.columns or "ask_depth_10bps" not in frame.columns:
        return pd.Series(0.15, index=frame.index, dtype="float64")
    bid10 = pd.to_numeric(frame["bid_depth_10bps"], errors="coerce")
    ask10 = pd.to_numeric(frame["ask_depth_10bps"], errors="coerce")
    min10 = pd.concat([bid10, ask10], axis=1).min(axis=1)
    valid = min10[np.isfinite(min10) & (min10 > 0.0)]
    if valid.empty:
        return pd.Series(0.15, index=frame.index, dtype="float64")
    q25 = float(valid.quantile(0.25))
    q10 = float(valid.quantile(0.10))
    penalty = pd.Series(0.0, index=frame.index, dtype="float64")
    penalty[min10 <= q25] += 0.05
    penalty[min10 <= q10] += 0.10
    penalty[~np.isfinite(min10)] += 0.15
    return penalty


def imbalance_penalty(frame: pd.DataFrame) -> pd.Series:
    columns = [column for column in ["order_book_imbalance_10bps", "order_book_imbalance_25bps"] if column in frame.columns]
    if not columns:
        return pd.Series(0.05, index=frame.index, dtype="float64")
    values = pd.concat([pd.to_numeric(frame[column], errors="coerce").abs() for column in columns], axis=1).max(axis=1)
    penalty = pd.Series(0.0, index=frame.index, dtype="float64")
    penalty[values >= 0.50] += 0.03
    penalty[values >= 0.75] += 0.07
    penalty[~np.isfinite(values)] += 0.05
    return penalty


def missing_liquidity_penalty(frame: pd.DataFrame) -> pd.Series:
    missing_count = sum(1 for column in FLOW_COLUMNS if column not in frame.columns)
    base = 0.25 if missing_count else 0.0
    penalty = pd.Series(base, index=frame.index, dtype="float64")
    for column in ["bid_depth_10bps", "ask_depth_10bps", "bid_depth_25bps", "ask_depth_25bps"]:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            penalty[~np.isfinite(values) | (values <= 0.0)] += 0.05
    return penalty


def scenario_costs(frame: pd.DataFrame) -> dict[str, pd.Series]:
    half = half_spread_bps(frame)
    depth = thin_depth_penalty(frame)
    imbalance = imbalance_penalty(frame)
    missing = missing_liquidity_penalty(frame)
    return {
        "fixed_0_05_bps": pd.Series(0.05, index=frame.index, dtype="float64"),
        "fixed_0_10_bps": pd.Series(0.10, index=frame.index, dtype="float64"),
        "fixed_0_25_bps": pd.Series(0.25, index=frame.index, dtype="float64"),
        "half_spread_bps": half,
        "half_spread_plus_0_05_bps": half + 0.05,
        "half_spread_plus_depth_penalty": half + depth,
        "half_spread_plus_depth_and_imbalance_penalty": half + depth + imbalance,
        "conservative_missing_liquidity_penalty": half + missing,
    }


def summarize_scenario(
    name: str,
    gross: pd.Series,
    costs: pd.Series,
    selected: pd.Series,
    contract: dict[str, Any],
    annotated_path: Path,
    missing_columns: list[str],
) -> dict[str, Any]:
    selected_gross_values = gross[selected].to_numpy(dtype="float64")
    selected_costs = costs[selected].to_numpy(dtype="float64")
    finite = np.isfinite(selected_gross_values) & np.isfinite(selected_costs)
    selected_gross_values = selected_gross_values[finite]
    selected_costs = selected_costs[finite]
    net = selected_gross_values - selected_costs
    rows = int(len(net))
    return {
        "scenario": name,
        "symbol": contract.get("symbol", ""),
        "venue": contract.get("venue", ""),
        "input_feature": contract.get("input_feature", ""),
        "hidden": contract.get("hidden", ""),
        "source_path_basename": contract.get("source_path_basename", ""),
        "seq_len": contract.get("seq_len", ""),
        "bucket_seconds": contract.get("bucket_seconds", ""),
        "input_stride": contract.get("input_stride", "1"),
        "output_stride": contract.get("output_stride", "1"),
        "policy": POLICY,
        "threshold_bps": THRESHOLD_BPS,
        "test_frac": TEST_FRAC,
        "selected_rows": rows,
        "avg_gross_bps": float(np.mean(selected_gross_values)) if rows else math.nan,
        "avg_dynamic_cost_bps": float(np.mean(selected_costs)) if rows else math.nan,
        "avg_net_bps": float(np.mean(net)) if rows else math.nan,
        "cum_net_bps": float(np.sum(net)) if rows else 0.0,
        "win_rate_net": float(np.mean(net > 0.0)) if rows else math.nan,
        "max_dip_net_bps": max_dip_bps(net),
        "cost_p50_bps": float(np.quantile(selected_costs, 0.50)) if rows else math.nan,
        "cost_p90_bps": float(np.quantile(selected_costs, 0.90)) if rows else math.nan,
        "cost_p99_bps": float(np.quantile(selected_costs, 0.99)) if rows else math.nan,
        "missing_flow_columns": ";".join(missing_columns),
        "available_flow_columns": ";".join(column for column in FLOW_COLUMNS if column not in missing_columns),
        "annotated_path": str(annotated_path),
        "paper_only": True,
        "training": False,
        "champion_mutation": False,
        "promotion": False,
        "orders": False,
    }


def render_text(probe_dir: Path, contract: dict[str, Any], report: pd.DataFrame, missing_columns: list[str]) -> str:
    lines = [
        "Rawseq Candidate Probe Dynamic Costs",
        "",
        f"Probe dir: {probe_dir}",
        f"Model path: {contract.get('model_path', '')}",
        (
            f"Contract: {contract.get('input_feature', '')} / {contract.get('hidden', '')} / "
            f"{contract.get('source_path_basename', '')} / seq={contract.get('seq_len', '')} "
            f"bucket={contract.get('bucket_seconds', '')} stride="
            f"{contract.get('input_stride', '1')}/{contract.get('output_stride', '1')}"
        ),
        f"Policy: {POLICY}",
        f"Threshold bps: {THRESHOLD_BPS:g}",
        f"Missing flow columns: {'; '.join(missing_columns) if missing_columns else 'none'}",
        "",
        "Scenario Summary",
        "  scenario selected avg_gross avg_cost avg_net cum_net win max_dip cost_p50 cost_p90 cost_p99",
        "  -------- -------- --------- -------- ------- ------- --- ------- -------- -------- --------",
    ]
    for _, row in report.iterrows():
        lines.append(
            f"  {row['scenario']} {int(row['selected_rows'])} "
            f"{fmt(row['avg_gross_bps'])} {fmt(row['avg_dynamic_cost_bps'])} "
            f"{fmt(row['avg_net_bps'])} {fmt(row['cum_net_bps'])} "
            f"{fmt(row['win_rate_net'])} {fmt(row['max_dip_net_bps'])} "
            f"{fmt(row['cost_p50_bps'])} {fmt(row['cost_p90_bps'])} {fmt(row['cost_p99_bps'])}"
        )

    fixed = report[report["scenario"].eq("fixed_0_10_bps")]
    harsher = report[report["scenario"].isin([
        "half_spread_bps",
        "half_spread_plus_depth_penalty",
        "half_spread_plus_depth_and_imbalance_penalty",
        "conservative_missing_liquidity_penalty",
    ])]
    hint = "research_candidate_or_reject"
    if not fixed.empty and safe_float(fixed.iloc[0]["cum_net_bps"]) > 0.0:
        if not harsher.empty and (pd.to_numeric(harsher["cum_net_bps"], errors="coerce") <= 0.0).any():
            hint = "keep_as_research_candidate_or_reject"
        else:
            hint = "dynamic_costs_do_not_immediately_reject"
    lines += [
        "",
        f"Decision hint: {hint}",
        "",
        "Safety",
        "  paper_only: true",
        "  training: false",
        "  champion_mutation: false",
        "  promotion: false",
        "  orders: false",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    probe_dir = resolve_probe_dir()
    annotated_path = require_file(probe_dir / "annotated.csv")
    contract = load_contract(require_file(probe_dir / "model_contract.json"))
    if str(contract.get("output_label", "future_return_path")) != "future_return_path":
        report = pd.DataFrame(
            [
                {
                    "scenario": "not_applicable",
                    "output_label": contract.get("output_label", ""),
                    "output_dim": contract.get("output_dim", ""),
                    "selected_rows": 0,
                    "cum_net_bps": math.nan,
                    "not_applicable_reason": "dynamic execution-cost trading evaluation is only valid for future_return_path",
                    "paper_only": True,
                    "training": False,
                    "champion_mutation": False,
                    "promotion": False,
                    "orders": False,
                }
            ]
        )
        output_csv = probe_dir / "dynamic_cost_summary.csv"
        output_txt = probe_dir / "dynamic_cost_summary.txt"
        report.to_csv(output_csv, index=False)
        text = (
            "Rawseq Candidate Dynamic Cost Evaluation\n\n"
            f"Probe dir: {probe_dir}\n"
            f"Output label: {contract.get('output_label', '')}\n"
            "Status: not_applicable\n"
            "Reason: dynamic execution-cost trading evaluation is only valid for future_return_path.\n\n"
            "Safety: paper-only. No training. No promotion. No champion mutation. No orders.\n"
        )
        output_txt.write_text(text, encoding="utf-8")
        print(text)
        print(f"Dynamic cost CSV: {output_csv}")
        print(f"Dynamic cost text: {output_txt}")
        return
    frame = load_test_frame(annotated_path)
    selected, gross = selected_gross(frame)
    missing_columns = [column for column in FLOW_COLUMNS if column not in frame.columns]

    costs_by_scenario = scenario_costs(frame)
    rows = [
        summarize_scenario(
            name,
            gross,
            costs,
            selected,
            contract,
            annotated_path,
            missing_columns,
        )
        for name, costs in costs_by_scenario.items()
    ]
    report = pd.DataFrame(rows)
    output_csv = probe_dir / "dynamic_cost_summary.csv"
    output_txt = probe_dir / "dynamic_cost_summary.txt"
    report.to_csv(output_csv, index=False)
    text = render_text(probe_dir, contract, report, missing_columns)
    output_txt.write_text(text, encoding="utf-8")
    print(text)
    print(f"Dynamic cost CSV: {output_csv}")
    print(f"Dynamic cost text: {output_txt}")
    print("Safety: read-only except reports. No training. No promotion. No orders.")


if __name__ == "__main__":
    main()
