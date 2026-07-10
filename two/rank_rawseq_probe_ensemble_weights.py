#!/usr/bin/env python3
"""Rank weighted prediction ensembles across rawseq probe folders.

This script averages prediction columns only. It never reads or averages model
weights, and it never trains, promotes, mutates champions, or places orders.
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_DIRS_ENV = os.getenv("RAWSEQ_ENSEMBLE_PROBE_DIRS", "").strip()
WEIGHT_GRID_ENV = os.getenv("RAWSEQ_ENSEMBLE_WEIGHT_GRID", "").strip()
THRESHOLDS_ENV = os.getenv("RAWSEQ_ENSEMBLE_THRESHOLDS", "0,0.1,0.2,0.3,0.5")
COSTS_ENV = os.getenv("RAWSEQ_ENSEMBLE_COSTS", "0.05,0.1,0.15,0.25")
POLICY = os.getenv("RAWSEQ_ENSEMBLE_POLICY", "inverse_gt").strip().lower()
OUTPUT_ROOT = Path(
    os.getenv(
        "RAWSEQ_ENSEMBLE_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_probe_ensemble_rankings"),
    )
)
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

PRED_COLUMN = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COLUMN = "rawseq_path_actual_horizon_return_bps"
ENSEMBLE_PRED_COLUMN = "ensemble_pred"
ROLLING_WINDOW_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0]
STATUS_PRIORITY = {
    "clean_candidate": 0,
    "research_candidate": 1,
    "fragile_candidate": 2,
    "reject": 3,
}
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


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def fmt(value: Any, digits: int = 6) -> str:
    value = safe_float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "nan"


def parse_float_list(text: str, label: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError as exc:
            raise SystemExit(f"{label} contains a non-float value: {item}") from exc
    if not values:
        raise SystemExit(f"{label} did not contain any values.")
    return values


def parse_probe_dirs() -> list[Path]:
    if not PROBE_DIRS_ENV:
        raise SystemExit("RAWSEQ_ENSEMBLE_PROBE_DIRS is required.")
    dirs: list[Path] = []
    for item in PROBE_DIRS_ENV.split(";"):
        text = item.strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise SystemExit(f"Probe folder does not exist: {path}")
        if not (path / "annotated.csv").exists():
            raise SystemExit(f"Probe folder missing annotated.csv: {path}")
        if not (path / "model_contract.json").exists():
            raise SystemExit(f"Probe folder missing model_contract.json: {path}")
        dirs.append(path.resolve())
    if len(dirs) < 2:
        raise SystemExit("RAWSEQ_ENSEMBLE_PROBE_DIRS must contain at least two probe folders.")
    return dirs


def normalize_weights(weights: list[float]) -> list[float] | None:
    total = float(sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        return None
    normalized = [float(value) / total for value in weights]
    if any(value < -1e-12 for value in normalized):
        return None
    return normalized


def parse_weight_grid(model_count: int) -> list[list[float]]:
    if WEIGHT_GRID_ENV:
        rows: list[list[float]] = []
        for item in re.split(r"[;\n]+", WEIGHT_GRID_ENV):
            item = item.strip()
            if not item:
                continue
            weights = parse_float_list(item, "RAWSEQ_ENSEMBLE_WEIGHT_GRID item")
            if len(weights) == 1 and model_count == 2:
                weights = [weights[0], 1.0 - weights[0]]
            if len(weights) != model_count:
                raise SystemExit(
                    f"Weight-grid item has {len(weights)} weights but there are {model_count} probes: {item}"
                )
            normalized = normalize_weights(weights)
            if normalized is None:
                raise SystemExit(f"Invalid non-positive weight-grid item: {item}")
            rows.append(normalized)
        if not rows:
            raise SystemExit("RAWSEQ_ENSEMBLE_WEIGHT_GRID did not contain any valid weights.")
        return rows
    if model_count == 2:
        return [[round(i * 0.05, 10), round(1.0 - i * 0.05, 10)] for i in range(21)]
    rows = [[1.0 / model_count] * model_count]
    for index in range(model_count):
        weights = [0.0] * model_count
        weights[index] = 1.0
        rows.append(weights)
    return rows


def load_contract(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def read_annotated(path: Path, label: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    required = ["timestamp", "time", PRED_COLUMN, ACTUAL_COLUMN]
    missing = [column for column in required if column not in header.columns]
    if missing:
        raise SystemExit(f"{path} missing required columns: {missing}")
    usecols = required + [column for column in FLOW_COLUMNS if column in header.columns]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    for column in ["timestamp", PRED_COLUMN, ACTUAL_COLUMN]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in FLOW_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", PRED_COLUMN, ACTUAL_COLUMN])
    if frame.empty:
        raise SystemExit(f"{path} has no finite timestamp/prediction/actual rows.")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    rename = {
        PRED_COLUMN: f"pred_{label}",
        ACTUAL_COLUMN: f"actual_{label}",
        "time": f"time_{label}",
    }
    flow_rename = {column: f"{column}_{label}" for column in FLOW_COLUMNS if column in frame.columns}
    return frame.rename(columns={**rename, **flow_rename}).reset_index(drop=True)


def max_dip_bps(returns: np.ndarray | pd.Series) -> float:
    values = np.asarray(returns, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    cumulative = np.cumsum(values)
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def selected_mask_and_gross(frame: pd.DataFrame, threshold_bps: float) -> tuple[pd.Series, pd.Series]:
    pred = pd.to_numeric(frame[ENSEMBLE_PRED_COLUMN], errors="coerce")
    actual = pd.to_numeric(frame[ACTUAL_COLUMN], errors="coerce")
    if POLICY == "inverse_gt":
        mask = pred > threshold_bps
        gross = -actual
    elif POLICY == "direct_gt":
        mask = pred > threshold_bps
        gross = actual
    elif POLICY == "inverse_directional_abs_gt":
        mask = pred.abs() > threshold_bps
        gross = -np.sign(pred) * actual
    else:
        raise SystemExit(
            "RAWSEQ_ENSEMBLE_POLICY must be one of: inverse_gt, direct_gt, inverse_directional_abs_gt"
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


def scenario_costs(frame: pd.DataFrame) -> dict[str, pd.Series]:
    half = half_spread_bps(frame)
    depth = thin_depth_penalty(frame)
    imbalance = imbalance_penalty(frame)
    return {
        "fixed_0_05_bps": pd.Series(0.05, index=frame.index, dtype="float64"),
        "fixed_0_10_bps": pd.Series(0.10, index=frame.index, dtype="float64"),
        "fixed_0_15_bps": pd.Series(0.15, index=frame.index, dtype="float64"),
        "fixed_0_25_bps": pd.Series(0.25, index=frame.index, dtype="float64"),
        "half_spread_plus_0_05_bps": half + 0.05,
        "half_spread_plus_depth_penalty": half + depth,
        "half_spread_plus_depth_and_imbalance_penalty": half + depth + imbalance,
    }


def build_aligned_frame(probe_dirs: list[Path]) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    frames: list[pd.DataFrame] = []
    contracts: list[dict[str, Any]] = []
    for index, probe_dir in enumerate(probe_dirs):
        label = f"p{index + 1}"
        contract = load_contract(probe_dir / "model_contract.json")
        contract["probe_dir"] = str(probe_dir)
        contracts.append(contract)
        frames.append(read_annotated(probe_dir / "annotated.csv", label))

    joined = frames[0]
    for frame in frames[1:]:
        joined = joined.merge(frame, on="timestamp", how="inner")
    if joined.empty:
        raise SystemExit("No overlapping timestamps across probe annotated.csv files.")

    pred_columns = [f"pred_p{i + 1}" for i in range(len(probe_dirs))]
    actual_columns = [f"actual_p{i + 1}" for i in range(len(probe_dirs))]
    time_columns = [f"time_p{i + 1}" for i in range(len(probe_dirs))]
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna(subset=pred_columns + actual_columns)
    if joined.empty:
        raise SystemExit("No finite aligned prediction/actual rows after merging.")

    actual_stack = joined[actual_columns].to_numpy(dtype="float64")
    actual_spread = np.nanmax(actual_stack, axis=1) - np.nanmin(actual_stack, axis=1)
    max_actual_diff = float(np.nanmax(np.abs(actual_spread))) if len(actual_spread) else 0.0
    if max_actual_diff > 1e-9:
        warnings.append(f"actual_horizon_return differs across probes; max_diff_bps={max_actual_diff:.12g}")
    joined[ACTUAL_COLUMN] = joined[actual_columns[0]]
    joined["time"] = joined[time_columns[0]]
    for flow_column in FLOW_COLUMNS:
        for index in range(len(probe_dirs)):
            candidate = f"{flow_column}_p{index + 1}"
            if candidate in joined.columns:
                joined[flow_column] = pd.to_numeric(joined[candidate], errors="coerce")
                break

    for field in ["symbol", "source_path_basename", "bucket_seconds"]:
        values = {safe_str(contract.get(field)) for contract in contracts if safe_str(contract.get(field))}
        if len(values) > 1:
            warnings.append(f"{field} differs across probes: {', '.join(sorted(values))}")
    return joined.sort_values("timestamp").reset_index(drop=True), contracts, warnings


def rolling_positive_fractions(selected: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    if selected.empty:
        for hours in ROLLING_WINDOW_HOURS:
            output[f"rolling_{hours:g}h_positive_fraction"] = math.nan
            output[f"rolling_{hours:g}h_windows"] = 0
        return output
    start = int(selected["timestamp"].min())
    for hours in ROLLING_WINDOW_HOURS:
        window_ms = int(hours * 3600 * 1000)
        work = selected[["timestamp", "net_bps"]].copy()
        work["window_id"] = ((work["timestamp"] - start) // window_ms).astype(int)
        cum = work.groupby("window_id", sort=True)["net_bps"].sum()
        output[f"rolling_{hours:g}h_positive_fraction"] = float((cum > 0.0).mean()) if len(cum) else math.nan
        output[f"rolling_{hours:g}h_windows"] = int(len(cum))
        output[f"rolling_{hours:g}h_worst_cum_bps"] = float(cum.min()) if len(cum) else math.nan
    return output


def evaluate_combo(frame: pd.DataFrame, weights: list[float], threshold: float, cost: float) -> dict[str, Any]:
    pred_columns = [column for column in frame.columns if re.fullmatch(r"pred_p\d+", column)]
    pred_matrix = frame[pred_columns].to_numpy(dtype="float64")
    frame[ENSEMBLE_PRED_COLUMN] = np.average(pred_matrix, axis=1, weights=weights)
    mask, gross_series = selected_mask_and_gross(frame, threshold)
    gross = gross_series[mask].to_numpy(dtype="float64")
    net = gross - cost
    selected = frame.loc[mask, ["timestamp"]].copy()
    selected["net_bps"] = net
    selected_rows = int(len(net))
    cum_net = float(np.sum(net)) if selected_rows else 0.0
    max_dip = max_dip_bps(net)
    dip_ratio = abs(max_dip) / max(cum_net, 1e-9) if selected_rows else math.nan
    row: dict[str, Any] = {
        "threshold_bps": threshold,
        "cost_bps": cost,
        "weights": ";".join(f"{weight:.12g}" for weight in weights),
        "selected_rows": selected_rows,
        "avg_gross_bps": float(np.mean(gross)) if selected_rows else math.nan,
        "avg_net_bps": float(np.mean(net)) if selected_rows else math.nan,
        "cum_net_bps": cum_net,
        "win_rate_net": float(np.mean(net > 0.0)) if selected_rows else math.nan,
        "max_dip_net_bps": max_dip,
        "max_dip_to_cum_net_ratio": dip_ratio,
    }
    row.update(rolling_positive_fractions(selected))
    scenarios = scenario_costs(frame)
    for name, costs in scenarios.items():
        selected_costs = costs[mask].to_numpy(dtype="float64")
        finite = np.isfinite(gross) & np.isfinite(selected_costs)
        scenario_net = gross[finite] - selected_costs[finite]
        row[f"{name}_selected_rows"] = int(len(scenario_net))
        row[f"{name}_cum_net_bps"] = float(np.sum(scenario_net)) if len(scenario_net) else 0.0
        row[f"{name}_avg_cost_bps"] = float(np.mean(selected_costs[finite])) if len(scenario_net) else math.nan
    return row


def score_row(row: dict[str, Any], threshold_zero_best: bool) -> tuple[float, list[str]]:
    reasons: list[str] = []
    selected_rows = safe_float(row.get("selected_rows"), 0.0)
    fixed_010 = safe_float(row.get("fixed_0_10_bps_cum_net_bps"), safe_float(row.get("cum_net_bps"), 0.0))
    fixed_025 = safe_float(row.get("fixed_0_25_bps_cum_net_bps"), 0.0)
    half_plus = safe_float(row.get("half_spread_plus_0_05_bps_cum_net_bps"), 0.0)
    dip_ratio = safe_float(row.get("max_dip_to_cum_net_ratio"), 999.0)
    rolling_1h = safe_float(row.get("rolling_1h_positive_fraction"), 0.0)
    rolling_3h = safe_float(row.get("rolling_3h_positive_fraction"), 0.0)
    score = 0.0
    score += max(fixed_010, -1000.0) * 1.0
    score += max(half_plus, -1000.0) * 1.5
    score += max(fixed_025, -1000.0) * 0.25
    score += min(selected_rows / 300.0, 3.0) * 50.0
    score += (rolling_1h + rolling_3h) * 150.0
    if math.isfinite(dip_ratio):
        score -= min(dip_ratio, 10.0) * 75.0
    if selected_rows < 100:
        score -= 500.0
        reasons.append("sparse_selected_rows_lt_100")
    elif selected_rows < 300:
        score -= 150.0
        reasons.append("selected_rows_lt_300")
    if fixed_010 <= 0.0:
        score -= 500.0
        reasons.append("fixed_0_10_nonpositive")
    if half_plus <= 0.0:
        score -= 350.0
        reasons.append("half_spread_plus_0_05_nonpositive")
    if fixed_025 < -250.0:
        score -= 200.0
        reasons.append("fixed_0_25_strongly_negative")
    if dip_ratio > 1.0:
        score -= 125.0
        reasons.append("drawdown_exceeds_cum_net")
    if threshold_zero_best and safe_float(row.get("threshold_bps"), 0.0) == 0.0:
        score -= 150.0
        reasons.append("threshold_0_only_behavior")
    return score, reasons


def classify_row(row: dict[str, Any]) -> tuple[str, list[str]]:
    fixed_010 = safe_float(row.get("fixed_0_10_bps_cum_net_bps"), 0.0)
    fixed_015 = safe_float(row.get("fixed_0_15_bps_cum_net_bps"), math.nan)
    fixed_025 = safe_float(row.get("fixed_0_25_bps_cum_net_bps"), 0.0)
    half_plus = safe_float(row.get("half_spread_plus_0_05_bps_cum_net_bps"), 0.0)
    selected_rows = safe_float(row.get("selected_rows"), 0.0)
    dip_ratio = safe_float(row.get("max_dip_to_cum_net_ratio"), math.inf)
    rolling_12h = safe_float(row.get("rolling_12h_positive_fraction"), 0.0)
    rolling_24h = safe_float(row.get("rolling_24h_positive_fraction"), 0.0)
    row["fixed_0_10_cum_net"] = fixed_010
    row["fixed_0_15_cum_net"] = fixed_015
    row["half_spread_plus_0_05_cum_net"] = half_plus
    row["fixed_0_25_cum_net"] = fixed_025
    row["rolling_12h_positive_fraction_for_status"] = rolling_12h
    row["rolling_24h_positive_fraction_for_status"] = rolling_24h

    reasons: list[str] = []
    if fixed_010 <= 0.0:
        reasons.append("fixed_0_10_cum_net_nonpositive")
    if selected_rows < 300:
        reasons.append("selected_rows_lt_300")
    if half_plus <= 0.0:
        reasons.append("half_spread_plus_0_05_cum_net_nonpositive")
    if fixed_025 <= 0.0:
        reasons.append("fixed_0_25_cum_net_nonpositive")
    if not math.isfinite(dip_ratio) or dip_ratio > 2.0:
        reasons.append("max_dip_to_cum_net_ratio_gt_2")
    if rolling_12h < 0.5:
        reasons.append("rolling_12h_positive_fraction_lt_0_5")
    if rolling_24h < 0.5:
        reasons.append("rolling_24h_positive_fraction_lt_0_5")

    if fixed_010 <= 0.0:
        return "reject", reasons
    if (
        half_plus > 0.0
        and selected_rows >= 300
        and math.isfinite(dip_ratio)
        and dip_ratio <= 2.0
        and rolling_12h >= 0.5
    ):
        return "clean_candidate", reasons
    if selected_rows >= 300 and (half_plus > 0.0 or fixed_025 > 0.0):
        return "research_candidate", reasons
    return "fragile_candidate", reasons


def make_contract(contracts: list[dict[str, Any]], warnings: list[str], best: pd.Series, output_dir: Path) -> dict[str, Any]:
    return {
        "ensemble_id": output_dir.name,
        "created_at": now_stamp(),
        "policy": POLICY,
        "probe_count": len(contracts),
        "probe_dirs": [contract.get("probe_dir", "") for contract in contracts],
        "weights": safe_str(best.get("weights")),
        "status": safe_str(best.get("status")),
        "rejection_reasons": safe_str(best.get("rejection_reasons")),
        "threshold_bps": safe_float(best.get("threshold_bps")),
        "cost_bps": safe_float(best.get("cost_bps")),
        "symbol": safe_str(contracts[0].get("symbol")) if contracts else "",
        "venue": safe_str(contracts[0].get("venue")) if contracts else "",
        "source_path_basename": safe_str(contracts[0].get("source_path_basename")) if contracts else "",
        "bucket_seconds": safe_str(contracts[0].get("bucket_seconds")) if contracts else "",
        "input_features": [safe_str(contract.get("input_feature")) for contract in contracts],
        "hiddens": [safe_str(contract.get("hidden")) for contract in contracts],
        "warnings": warnings,
        "paper_only": True,
        "training": False,
        "champion_mutation": False,
        "promotion": False,
        "orders": False,
    }


def render_ranked_row(rank: int, row: pd.Series) -> str:
    return (
        f"  {rank:02d}. status={row['status']} score={fmt(row['score'], 3)} weights={row['weights']} "
        f"thr={float(row['threshold_bps']):g} cost={float(row['cost_bps']):g} "
        f"rows={int(row['selected_rows'])} fixed010={fmt(row['fixed_0_10_cum_net'])} "
        f"half+005={fmt(row['half_spread_plus_0_05_cum_net'])} "
        f"fixed025={fmt(row['fixed_0_25_cum_net'])} "
        f"dip_ratio={fmt(row['max_dip_to_cum_net_ratio'], 3)} "
        f"roll12h={fmt(row['rolling_12h_positive_fraction'], 3)} "
        f"rejection_reasons={row['rejection_reasons'] or 'none'}"
    )


def render_text(ranking: pd.DataFrame, contracts: list[dict[str, Any]], warnings: list[str], output_dir: Path) -> str:
    eligible = ranking[ranking["status"].isin(["clean_candidate", "research_candidate", "fragile_candidate"])]
    clean_or_research = ranking[ranking["status"].isin(["clean_candidate", "research_candidate"])]
    rejects = ranking[ranking["status"] == "reject"]
    lines = [
        "Rawseq Probe Ensemble Weight Ranking",
        "",
        f"Output dir: {output_dir}",
        f"Probe count: {len(contracts)}",
        f"Policy: {POLICY}",
        "Probes:",
    ]
    for index, contract in enumerate(contracts):
        lines.append(
            f"  p{index + 1}: feature={contract.get('input_feature', '')} hidden={contract.get('hidden', '')} "
            f"source={contract.get('source_path_basename', '')} dir={contract.get('probe_dir', '')}"
        )
    lines += ["", "Warnings"]
    lines.extend([f"  - {warning}" for warning in warnings] or ["  none"])
    lines += ["", "Top Clean Or Research"]
    if clean_or_research.empty:
        lines.append("  none")
    else:
        for rank, (_, row) in enumerate(clean_or_research.head(20).iterrows(), start=1):
            lines.append(render_ranked_row(rank, row))
    lines += ["", "Top Eligible Candidates"]
    if eligible.empty:
        lines.append("  No eligible ensemble found.")
    else:
        for rank, (_, row) in enumerate(eligible.head(20).iterrows(), start=1):
            lines.append(render_ranked_row(rank, row))
    lines += ["", "Best Rejects For Diagnostics"]
    if rejects.empty:
        lines.append("  none")
    else:
        for rank, (_, row) in enumerate(rejects.head(20).iterrows(), start=1):
            lines.append(render_ranked_row(rank, row))
    lines += [
        "",
        f"Ranking CSV: {output_dir / 'ensemble_weight_ranking.csv'}",
        f"Best contract JSON: {output_dir / 'best_ensemble_contract.json'}",
        "Safety: no training. No champion mutation. No promotion. No orders.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    probe_dirs = parse_probe_dirs()
    thresholds = parse_float_list(THRESHOLDS_ENV, "RAWSEQ_ENSEMBLE_THRESHOLDS")
    costs = parse_float_list(COSTS_ENV, "RAWSEQ_ENSEMBLE_COSTS")
    weight_grid = parse_weight_grid(len(probe_dirs))
    aligned, contracts, warnings = build_aligned_frame(probe_dirs)
    rows: list[dict[str, Any]] = []
    for weights in weight_grid:
        threshold_cum: dict[float, float] = {}
        combo_rows: list[dict[str, Any]] = []
        for threshold in thresholds:
            for cost in costs:
                row = evaluate_combo(aligned.copy(), weights, threshold, cost)
                row["probe_count"] = len(probe_dirs)
                row["policy"] = POLICY
                row["weight_grid_index"] = len(rows) + len(combo_rows)
                row["symbol"] = safe_str(contracts[0].get("symbol")) if contracts else ""
                row["venue"] = safe_str(contracts[0].get("venue")) if contracts else ""
                row["source_path_basename"] = safe_str(contracts[0].get("source_path_basename")) if contracts else ""
                row["bucket_seconds"] = safe_str(contracts[0].get("bucket_seconds")) if contracts else ""
                row["input_features"] = ";".join(safe_str(contract.get("input_feature")) for contract in contracts)
                row["hiddens"] = ";".join(safe_str(contract.get("hidden")) for contract in contracts)
                combo_rows.append(row)
                if abs(cost - 0.10) < 1e-12:
                    threshold_cum[threshold] = safe_float(row.get("cum_net_bps"), 0.0)
        best_threshold = max(threshold_cum.items(), key=lambda item: item[1])[0] if threshold_cum else math.nan
        threshold_zero_best = abs(best_threshold - 0.0) < 1e-12 if math.isfinite(best_threshold) else False
        for row in combo_rows:
            score, reasons = score_row(row, threshold_zero_best)
            status, rejection_reasons = classify_row(row)
            row["score"] = score
            row["status"] = status
            row["status_priority"] = STATUS_PRIORITY[status]
            row["best_fixed_0_10_threshold_for_weights"] = best_threshold
            row["threshold_zero_best_for_weights"] = threshold_zero_best
            row["penalty_reasons"] = ";".join(reasons)
            row["rejection_reasons"] = ";".join(rejection_reasons)
        rows.extend(combo_rows)

    ranking = pd.DataFrame(rows).sort_values(
        ["status_priority", "score", "cum_net_bps", "selected_rows"],
        ascending=[True, False, False, False],
    )
    output_dir = OUTPUT_ROOT / f"rawseq_probe_ensemble_weight_ranking_{now_stamp()}_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    ranking.to_csv(output_dir / "ensemble_weight_ranking.csv", index=False)
    best_contract = make_contract(contracts, warnings, ranking.iloc[0], output_dir)
    (output_dir / "best_ensemble_contract.json").write_text(
        json.dumps(best_contract, indent=2, sort_keys=True), encoding="utf-8"
    )
    text = render_text(ranking, contracts, warnings, output_dir)
    (output_dir / "ensemble_weight_ranking.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
