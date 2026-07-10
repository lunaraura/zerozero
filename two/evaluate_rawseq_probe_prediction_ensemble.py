#!/usr/bin/env python3
"""Evaluate an ensemble made by averaging rawseq probe predictions.

This averages prediction columns only. It does not read, average, or mutate
model weights.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_DIRS_ENV = os.getenv("RAWSEQ_ENSEMBLE_PROBE_DIRS", "").strip()
WEIGHTS_ENV = os.getenv("RAWSEQ_ENSEMBLE_WEIGHTS", "").strip()
OUTPUT_ROOT = Path(
    os.getenv(
        "RAWSEQ_ENSEMBLE_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_probe_ensembles"),
    )
)
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

POLICY = os.getenv("RAWSEQ_ENSEMBLE_POLICY", "inverse_gt").strip().lower()
THRESHOLD_BPS_LIST_ENV = os.getenv("RAWSEQ_ENSEMBLE_THRESHOLD_BPS_LIST", "0.0,0.1,0.2,0.3,0.5")
COST_BPS_LIST_ENV = os.getenv("RAWSEQ_ENSEMBLE_COST_BPS_LIST", "0,0.05,0.1,0.25")
TEST_FRAC = float(os.getenv("RAWSEQ_ENSEMBLE_TEST_FRAC", "0.20"))
DYNAMIC_THRESHOLD_BPS = float(os.getenv("RAWSEQ_ENSEMBLE_DYNAMIC_THRESHOLD_BPS", "0.1"))

PRED_COLUMN = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COLUMN = "rawseq_path_actual_horizon_return_bps"
ENSEMBLE_PRED_COLUMN = "ensemble_pred"
ROLLING_WINDOW_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0]
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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
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
    dirs = []
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


def parse_weights(count: int) -> list[float]:
    if not WEIGHTS_ENV:
        return [1.0 / count] * count
    raw = parse_float_list(WEIGHTS_ENV, "RAWSEQ_ENSEMBLE_WEIGHTS")
    if len(raw) != count:
        raise SystemExit(
            f"RAWSEQ_ENSEMBLE_WEIGHTS length {len(raw)} does not match probe count {count}."
        )
    total = float(sum(raw))
    if not math.isfinite(total) or total <= 0.0:
        raise SystemExit("RAWSEQ_ENSEMBLE_WEIGHTS must sum to a positive finite value.")
    return [value / total for value in raw]


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


def selected_gross(frame: pd.DataFrame, threshold_bps: float) -> np.ndarray:
    pred = frame[ENSEMBLE_PRED_COLUMN].to_numpy(dtype="float64")
    actual = frame[ACTUAL_COLUMN].to_numpy(dtype="float64")
    if POLICY == "inverse_gt":
        mask = pred > threshold_bps
        gross = -actual[mask]
    elif POLICY == "direct_gt":
        mask = pred > threshold_bps
        gross = actual[mask]
    elif POLICY == "inverse_directional_abs_gt":
        mask = np.abs(pred) > threshold_bps
        gross = -np.sign(pred[mask]) * actual[mask]
    else:
        raise SystemExit(
            "RAWSEQ_ENSEMBLE_POLICY must be one of: inverse_gt, direct_gt, inverse_directional_abs_gt"
        )
    gross = np.asarray(gross, dtype="float64")
    return gross[np.isfinite(gross)]


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


def build_cost_threshold_summary(
    test: pd.DataFrame,
    contract: dict[str, Any],
    thresholds: list[float],
    costs: list[float],
    annotated_path: Path,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        gross = selected_gross(test, threshold)
        for cost in costs:
            net = gross - cost
            rows.append(
                {
                    "symbol": contract.get("symbol", ""),
                    "venue": contract.get("venue", ""),
                    "ensemble_id": contract.get("ensemble_id", ""),
                    "probe_count": contract.get("probe_count", ""),
                    "input_features": contract.get("input_features", ""),
                    "source_path_basename": contract.get("source_path_basename", ""),
                    "bucket_seconds": contract.get("bucket_seconds", ""),
                    "seq_len": contract.get("seq_len", ""),
                    "policy": POLICY,
                    "threshold_bps": threshold,
                    "cost_bps": cost,
                    "test_frac": TEST_FRAC,
                    "selected_rows": int(len(net)),
                    "avg_gross_bps": float(np.mean(gross)) if len(gross) else math.nan,
                    "avg_net_bps": float(np.mean(net)) if len(net) else math.nan,
                    "cum_gross_bps": float(np.sum(gross)) if len(gross) else 0.0,
                    "cum_net_bps": float(np.sum(net)) if len(net) else 0.0,
                    "win_rate_gross": float(np.mean(gross > 0.0)) if len(gross) else math.nan,
                    "win_rate_net": float(np.mean(net > 0.0)) if len(net) else math.nan,
                    "max_dip_gross_bps": max_dip_bps(gross),
                    "max_dip_net_bps": max_dip_bps(net),
                    "first_time": test["time"].iloc[0] if not test.empty else "",
                    "last_time": test["time"].iloc[-1] if not test.empty else "",
                    "test_rows_total": int(len(test)),
                    "annotated_path": str(annotated_path),
                    "paper_only": True,
                    "orders": False,
                    "promotion": False,
                    "champion_mutation": False,
                    "training": False,
                }
            )
    return pd.DataFrame(rows)


def build_rolling_summary(
    test: pd.DataFrame,
    contract: dict[str, Any],
    thresholds: list[float],
    costs: list[float],
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        mask, gross = selected_mask_and_gross(test, threshold)
        selected_base = test.loc[mask, ["timestamp", "time"]].copy()
        selected_base["gross_bps"] = gross[mask].to_numpy(dtype="float64")
        for cost in costs:
            selected = selected_base.copy()
            selected["net_bps"] = selected["gross_bps"] - cost
            for hours in ROLLING_WINDOW_HOURS:
                window_ms = int(hours * 3600 * 1000)
                if selected.empty:
                    continue
                start = int(selected["timestamp"].min())
                selected["window_id"] = ((selected["timestamp"] - start) // window_ms).astype(int)
                for window_id, group in selected.groupby("window_id", sort=True):
                    net = group["net_bps"].to_numpy(dtype="float64")
                    gross_values = group["gross_bps"].to_numpy(dtype="float64")
                    rows.append(
                        {
                            "symbol": contract.get("symbol", ""),
                            "venue": contract.get("venue", ""),
                            "ensemble_id": contract.get("ensemble_id", ""),
                            "policy": POLICY,
                            "threshold_bps": threshold,
                            "cost_bps": cost,
                            "window_hours": hours,
                            "window_id": int(window_id),
                            "window_start_time": group["time"].iloc[0],
                            "window_end_time": group["time"].iloc[-1],
                            "total_rows": int(len(group)),
                            "selected_rows": int(len(group)),
                            "avg_gross_bps": float(np.mean(gross_values)) if len(gross_values) else math.nan,
                            "avg_net_bps": float(np.mean(net)) if len(net) else math.nan,
                            "cum_gross_bps": float(np.sum(gross_values)) if len(gross_values) else 0.0,
                            "cum_net_bps": float(np.sum(net)) if len(net) else 0.0,
                            "win_rate_net": float(np.mean(net > 0.0)) if len(net) else math.nan,
                            "max_dip_net_bps": max_dip_bps(net),
                        }
                    )
    return pd.DataFrame(rows)


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


def build_dynamic_cost_summary(
    test: pd.DataFrame,
    contract: dict[str, Any],
    annotated_path: Path,
) -> pd.DataFrame:
    mask, gross = selected_mask_and_gross(test, DYNAMIC_THRESHOLD_BPS)
    missing_columns = [column for column in FLOW_COLUMNS if column not in test.columns]
    rows = []
    for name, costs in scenario_costs(test).items():
        selected_gross_values = gross[mask].to_numpy(dtype="float64")
        selected_costs = costs[mask].to_numpy(dtype="float64")
        finite = np.isfinite(selected_gross_values) & np.isfinite(selected_costs)
        selected_gross_values = selected_gross_values[finite]
        selected_costs = selected_costs[finite]
        net = selected_gross_values - selected_costs
        row_count = int(len(net))
        rows.append(
            {
                "scenario": name,
                "symbol": contract.get("symbol", ""),
                "venue": contract.get("venue", ""),
                "ensemble_id": contract.get("ensemble_id", ""),
                "policy": POLICY,
                "threshold_bps": DYNAMIC_THRESHOLD_BPS,
                "test_frac": TEST_FRAC,
                "selected_rows": row_count,
                "avg_gross_bps": float(np.mean(selected_gross_values)) if row_count else math.nan,
                "avg_dynamic_cost_bps": float(np.mean(selected_costs)) if row_count else math.nan,
                "avg_net_bps": float(np.mean(net)) if row_count else math.nan,
                "cum_net_bps": float(np.sum(net)) if row_count else 0.0,
                "win_rate_net": float(np.mean(net > 0.0)) if row_count else math.nan,
                "max_dip_net_bps": max_dip_bps(net),
                "cost_p50_bps": float(np.quantile(selected_costs, 0.50)) if row_count else math.nan,
                "cost_p90_bps": float(np.quantile(selected_costs, 0.90)) if row_count else math.nan,
                "cost_p99_bps": float(np.quantile(selected_costs, 0.99)) if row_count else math.nan,
                "missing_flow_columns": ";".join(missing_columns),
                "available_flow_columns": ";".join(column for column in FLOW_COLUMNS if column not in missing_columns),
                "annotated_path": str(annotated_path),
                "paper_only": True,
                "training": False,
                "champion_mutation": False,
                "promotion": False,
                "orders": False,
            }
        )
    return pd.DataFrame(rows)


def build_ensemble_frame(probe_dirs: list[Path], weights: list[float]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    warnings: list[str] = []
    frames = []
    contracts = []
    for index, probe_dir in enumerate(probe_dirs):
        label = f"p{index + 1}"
        contract = load_contract(probe_dir / "model_contract.json")
        contract["probe_dir"] = str(probe_dir)
        contracts.append(contract)
        frame = read_annotated(probe_dir / "annotated.csv", label)
        frames.append(frame)

    joined = frames[0]
    for frame in frames[1:]:
        joined = joined.merge(frame, on="timestamp", how="inner")
    if joined.empty:
        raise SystemExit("No overlapping timestamps across probe annotated.csv files.")

    pred_columns = [f"pred_p{i + 1}" for i in range(len(probe_dirs))]
    actual_columns = [f"actual_p{i + 1}" for i in range(len(probe_dirs))]
    time_columns = [f"time_p{i + 1}" for i in range(len(probe_dirs))]
    for column in pred_columns + actual_columns:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
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
    joined[ENSEMBLE_PRED_COLUMN] = np.average(joined[pred_columns].to_numpy(dtype="float64"), axis=1, weights=weights)

    # Use flow columns from the first probe that has each field.
    for flow_column in FLOW_COLUMNS:
        candidates = [f"{flow_column}_p{i + 1}" for i in range(len(probe_dirs))]
        for candidate in candidates:
            if candidate in joined.columns:
                joined[flow_column] = pd.to_numeric(joined[candidate], errors="coerce")
                break

    base_contract = contracts[0]
    for field in ["symbol", "source_path_basename", "bucket_seconds"]:
        values = {safe_str(contract.get(field)) for contract in contracts if safe_str(contract.get(field))}
        if len(values) > 1:
            warnings.append(f"{field} differs across probes: {', '.join(sorted(values))}")
    contract = {
        "ensemble_id": f"rawseq_probe_ensemble_{now_stamp()}_{uuid.uuid4().hex[:8]}",
        "symbol": safe_str(base_contract.get("symbol")),
        "venue": safe_str(base_contract.get("venue")),
        "source_path_basename": safe_str(base_contract.get("source_path_basename")),
        "bucket_seconds": safe_str(base_contract.get("bucket_seconds")),
        "seq_len": safe_str(base_contract.get("seq_len")),
        "probe_count": len(probe_dirs),
        "probe_dirs": ";".join(str(path) for path in probe_dirs),
        "weights": ";".join(f"{weight:.12g}" for weight in weights),
        "input_features": ";".join(safe_str(contract.get("input_feature")) for contract in contracts),
        "hiddens": ";".join(safe_str(contract.get("hidden")) for contract in contracts),
        "model_paths": ";".join(safe_str(contract.get("model_path")) for contract in contracts),
        "paper_only": True,
        "training": False,
        "champion_mutation": False,
        "promotion": False,
        "orders": False,
    }
    return joined, contract, warnings


def test_split(frame: pd.DataFrame) -> pd.DataFrame:
    if not 0.0 < TEST_FRAC <= 1.0:
        raise SystemExit(f"RAWSEQ_ENSEMBLE_TEST_FRAC must be in (0, 1], got {TEST_FRAC}")
    split_at = int(len(frame) * (1.0 - TEST_FRAC))
    test = frame.iloc[split_at:].copy()
    return test.sort_values("timestamp").reset_index(drop=True)


def render_text(
    contract: dict[str, Any],
    warnings: list[str],
    cost_summary: pd.DataFrame,
    rolling: pd.DataFrame,
    dynamic: pd.DataFrame,
    run_dir: Path,
    annotated_path: Path,
) -> str:
    base = cost_summary[
        (cost_summary["threshold_bps"].sub(0.1).abs() < 1e-12)
        & (cost_summary["cost_bps"].sub(0.1).abs() < 1e-12)
    ]
    base_row = base.iloc[0] if not base.empty else None
    best = cost_summary[cost_summary["selected_rows"] > 0].sort_values(
        ["cum_net_bps", "selected_rows"], ascending=[False, False]
    )
    best_row = best.iloc[0] if not best.empty else None
    lines = [
        "Rawseq Probe Prediction Ensemble",
        "",
        f"Output dir: {run_dir}",
        f"Probe count: {contract.get('probe_count')}",
        f"Policy: {POLICY}",
        f"Weights: {contract.get('weights')}",
        f"Symbol/source/bucket: {contract.get('symbol')} / {contract.get('source_path_basename')} / {contract.get('bucket_seconds')}",
        f"Annotated ensemble rows: {annotated_path}",
        "",
        "Warnings",
    ]
    lines.extend([f"  - {warning}" for warning in warnings] or ["  none"])
    lines += ["", "Fixed Cost Summary"]
    if base_row is not None:
        lines.append(
            f"  threshold=0.1 cost=0.1 rows={int(base_row['selected_rows'])} "
            f"avg_net={fmt(base_row['avg_net_bps'])} cum_net={fmt(base_row['cum_net_bps'])} "
            f"win_net={fmt(base_row['win_rate_net'])} max_dip={fmt(base_row['max_dip_net_bps'])}"
        )
    if best_row is not None:
        lines.append(
            f"  best_grid_by_cum_net: threshold={float(best_row['threshold_bps']):g} "
            f"cost={float(best_row['cost_bps']):g} rows={int(best_row['selected_rows'])} "
            f"cum_net={fmt(best_row['cum_net_bps'])}"
        )
    lines += ["", "Rolling Summary"]
    subset = rolling[(rolling["threshold_bps"].sub(0.1).abs() < 1e-12) & (rolling["cost_bps"].sub(0.1).abs() < 1e-12)]
    if subset.empty:
        lines.append("  no rolling rows at threshold=0.1 cost=0.1")
    else:
        for hours in ROLLING_WINDOW_HOURS:
            part = subset[subset["window_hours"].sub(hours).abs() < 1e-12]
            if part.empty:
                continue
            cum = pd.to_numeric(part["cum_net_bps"], errors="coerce")
            lines.append(
                f"  {hours:g}h: windows={len(part)} positive={int((cum > 0.0).sum())} "
                f"total_cum_net={fmt(cum.sum())} worst={fmt(cum.min())} "
                f"selected_rows={int(pd.to_numeric(part['selected_rows'], errors='coerce').sum())}"
            )
    lines += ["", "Dynamic Cost Summary"]
    if dynamic.empty:
        lines.append("  none")
    else:
        for _, row in dynamic.iterrows():
            lines.append(
                f"  {row['scenario']}: rows={int(row['selected_rows'])} "
                f"avg_cost={fmt(row['avg_dynamic_cost_bps'])} cum_net={fmt(row['cum_net_bps'])} "
                f"max_dip={fmt(row['max_dip_net_bps'])}"
            )
    lines += [
        "",
        f"Fixed summary CSV: {run_dir / 'ensemble_summary.csv'}",
        f"Rolling summary CSV: {run_dir / 'rolling_summary.csv'}",
        f"Dynamic cost CSV: {run_dir / 'dynamic_cost_summary.csv'}",
        "Safety: no training. No champion mutation. No promotion. No orders.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    probe_dirs = parse_probe_dirs()
    weights = parse_weights(len(probe_dirs))
    thresholds = parse_float_list(THRESHOLD_BPS_LIST_ENV, "RAWSEQ_ENSEMBLE_THRESHOLD_BPS_LIST")
    costs = parse_float_list(COST_BPS_LIST_ENV, "RAWSEQ_ENSEMBLE_COST_BPS_LIST")
    aligned, contract, warnings = build_ensemble_frame(probe_dirs, weights)
    run_dir = OUTPUT_ROOT / contract["ensemble_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    annotated_path = run_dir / "ensemble_annotated.csv"
    aligned.to_csv(annotated_path, index=False)
    (run_dir / "ensemble_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

    test = test_split(aligned)
    cost_summary = build_cost_threshold_summary(test, contract, thresholds, costs, annotated_path)
    rolling = build_rolling_summary(test, contract, thresholds, costs)
    dynamic = build_dynamic_cost_summary(test, contract, annotated_path)
    cost_summary.to_csv(run_dir / "ensemble_summary.csv", index=False)
    rolling.to_csv(run_dir / "rolling_summary.csv", index=False)
    dynamic.to_csv(run_dir / "dynamic_cost_summary.csv", index=False)
    text = render_text(contract, warnings, cost_summary, rolling, dynamic, run_dir, annotated_path)
    (run_dir / "ensemble_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
