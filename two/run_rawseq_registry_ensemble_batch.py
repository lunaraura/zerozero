#!/usr/bin/env python3
"""Batch-rank prediction ensembles from the rawseq threshold registry.

The registry candidate row is the input unit. This averages already-probed
prediction columns only; it never trains, mutates champions, promotes artifacts,
or places orders.
"""

from __future__ import annotations

import csv
import itertools
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
REGISTRY_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_probe_registry"
REGISTRY_DIR_ENV = os.getenv("RAWSEQ_REGISTRY_DIR", "").strip()
TOP_N = int(float(os.getenv("RAWSEQ_REGISTRY_TOP_N", "8")))
ALLOWED_STATUSES = {
    item.strip()
    for item in os.getenv(
        "RAWSEQ_REGISTRY_ALLOWED_STATUSES",
        "clean_shadow_candidate,robust_research_candidate",
    ).split(",")
    if item.strip()
}
MAX_SIZE = int(float(os.getenv("RAWSEQ_ENSEMBLE_MAX_SIZE", "3")))
OUTPUT_ROOT = Path(
    os.getenv(
        "RAWSEQ_ENSEMBLE_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_registry_ensemble_batches"),
    )
)
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT
POLICY = os.getenv("RAWSEQ_ENSEMBLE_POLICY", "inverse_gt").strip().lower()
WEIGHT_STEP = float(os.getenv("RAWSEQ_ENSEMBLE_WEIGHT_STEP", "0.1"))
THRESHOLDS = [float(item.strip()) for item in os.getenv("RAWSEQ_ENSEMBLE_THRESHOLDS", "0,0.1,0.2,0.3,0.5").split(",") if item.strip()]
COSTS = [float(item.strip()) for item in os.getenv("RAWSEQ_ENSEMBLE_COSTS", "0.05,0.1,0.15,0.25").split(",") if item.strip()]

PRED_COLUMN = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COLUMN = "rawseq_path_actual_horizon_return_bps"
ROLLING_WINDOW_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0]
GATES = [
    "none",
    "all_positive",
    "min_pred_above_0",
    "min_pred_above_-0.1",
    "primary_above_0.2_secondary_above_-0.1",
    "primary_above_0.3_secondary_above_0",
]
STATUS_PRIORITY = {
    "clean_shadow_ensemble": 0,
    "robust_research_ensemble": 1,
    "fragile_research_ensemble": 2,
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
OUTPUT_COLUMNS = [
    "status",
    "status_priority",
    "rank_score",
    "ensemble_id",
    "ensemble_size",
    "candidate_indices",
    "probe_dirs",
    "model_paths",
    "registry_thresholds",
    "input_features",
    "ma_windows",
    "hiddens",
    "seeds",
    "weights",
    "threshold_bps",
    "gate",
    "selected_rows",
    "avg_gross_bps",
    "avg_net_bps",
    "cum_net_bps",
    "win_rate_net",
    "max_dip_net_bps",
    "max_dip_to_cum_net_ratio",
    "positive_1h_window_fraction",
    "positive_3h_window_fraction",
    "positive_6h_window_fraction",
    "positive_12h_window_fraction",
    "positive_24h_window_fraction",
    "fixed_0_05_cum_net",
    "fixed_0_10_cum_net",
    "fixed_0_15_cum_net",
    "fixed_0_25_cum_net",
    "half_spread_plus_0_05_cum_net",
    "half_spread_plus_depth_penalty_cum_net",
    "half_spread_plus_depth_and_imbalance_penalty_cum_net",
    "conservative_missing_liquidity_penalty_cum_net",
    "rejection_reasons",
    "aligned_rows",
    "paper_only",
    "training",
    "champion_mutation",
    "promotion",
    "orders",
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
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def fmt(value: Any, digits: int = 6) -> str:
    value = safe_float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "nan"


def latest_registry_dir() -> Path:
    if REGISTRY_DIR_ENV:
        path = Path(REGISTRY_DIR_ENV)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    candidates = [path for path in REGISTRY_ROOT.iterdir() if path.is_dir()] if REGISTRY_ROOT.exists() else []
    if not candidates:
        raise SystemExit(f"No registry folders found under {REGISTRY_ROOT}")
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)[0]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def registry_dedupe_key(row: dict[str, Any]) -> str:
    threshold = safe_str(row.get("best_threshold_for_probe")) or safe_str(row.get("threshold_bps"))
    return "|".join([safe_str(row.get("probe_dir")), safe_str(row.get("model_path")), threshold])


def load_registry_candidates(registry_dir: Path) -> pd.DataFrame:
    path = registry_dir / "top_shadow_candidates.csv"
    if not path.exists():
        raise SystemExit(f"Missing top_shadow_candidates.csv in {registry_dir}")
    rows = [row for row in read_csv_rows(path) if safe_str(row.get("status")) in ALLOWED_STATUSES]
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, row in enumerate(rows, start=1):
        key = registry_dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        row["registry_rank"] = rank
        row["registry_dedupe_key"] = key
        deduped.append(row)
        if len(deduped) >= TOP_N:
            break
    if len(deduped) < 2:
        raise SystemExit("Need at least two allowed registry candidates after dedupe.")
    return pd.DataFrame(deduped)


def read_annotated(probe_dir: Path) -> pd.DataFrame:
    path = probe_dir / "annotated.csv"
    if not path.exists():
        raise SystemExit(f"Probe missing annotated.csv: {probe_dir}")
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
    return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def weight_grid(size: int) -> list[tuple[float, ...]]:
    if size == 2:
        steps = int(round(1.0 / WEIGHT_STEP))
        return [(round(i * WEIGHT_STEP, 10), round(1.0 - i * WEIGHT_STEP, 10)) for i in range(steps + 1)]
    if size == 3:
        steps = int(round(1.0 / WEIGHT_STEP))
        weights: list[tuple[float, ...]] = []
        for a in range(steps + 1):
            for b in range(steps + 1 - a):
                c = steps - a - b
                weights.append((round(a * WEIGHT_STEP, 10), round(b * WEIGHT_STEP, 10), round(c * WEIGHT_STEP, 10)))
        return weights
    return [tuple([1.0 / size] * size)]


def max_dip_bps(values: np.ndarray) -> float:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    cumulative = np.cumsum(values)
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def rolling_fractions(timestamp: np.ndarray, net: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if len(net) == 0:
        for hours in ROLLING_WINDOW_HOURS:
            output[f"positive_{hours:g}h_window_fraction"] = math.nan
        return output
    start = int(np.min(timestamp))
    for hours in ROLLING_WINDOW_HOURS:
        window_ms = int(hours * 3600 * 1000)
        window_ids = ((timestamp - start) // window_ms).astype(np.int64)
        sums = np.bincount(window_ids, weights=net)
        output[f"positive_{hours:g}h_window_fraction"] = float(np.mean(sums > 0.0)) if len(sums) else math.nan
    return output


def half_spread_bps(frame: pd.DataFrame, default_bps: float = 0.10) -> np.ndarray:
    if "spread_percent" not in frame.columns:
        return np.full(len(frame), default_bps, dtype="float64")
    spread_percent = pd.to_numeric(frame["spread_percent"], errors="coerce").to_numpy(dtype="float64")
    cost = spread_percent * 100.0 / 2.0
    return np.where(np.isfinite(cost) & (cost >= 0.0), cost, default_bps)


def thin_depth_penalty(frame: pd.DataFrame) -> np.ndarray:
    if "bid_depth_10bps" not in frame.columns or "ask_depth_10bps" not in frame.columns:
        return np.full(len(frame), 0.15, dtype="float64")
    bid10 = pd.to_numeric(frame["bid_depth_10bps"], errors="coerce").to_numpy(dtype="float64")
    ask10 = pd.to_numeric(frame["ask_depth_10bps"], errors="coerce").to_numpy(dtype="float64")
    min10 = np.fmin(bid10, ask10)
    valid = min10[np.isfinite(min10) & (min10 > 0.0)]
    if len(valid) == 0:
        return np.full(len(frame), 0.15, dtype="float64")
    q25 = float(np.quantile(valid, 0.25))
    q10 = float(np.quantile(valid, 0.10))
    penalty = np.zeros(len(frame), dtype="float64")
    penalty[min10 <= q25] += 0.05
    penalty[min10 <= q10] += 0.10
    penalty[~np.isfinite(min10)] += 0.15
    return penalty


def imbalance_penalty(frame: pd.DataFrame) -> np.ndarray:
    columns = [column for column in ["order_book_imbalance_10bps", "order_book_imbalance_25bps"] if column in frame.columns]
    if not columns:
        return np.full(len(frame), 0.05, dtype="float64")
    values = np.vstack([np.abs(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")) for column in columns])
    max_abs = np.nanmax(values, axis=0)
    penalty = np.zeros(len(frame), dtype="float64")
    penalty[max_abs >= 0.50] += 0.03
    penalty[max_abs >= 0.75] += 0.07
    penalty[~np.isfinite(max_abs)] += 0.05
    return penalty


def missing_liquidity_penalty(frame: pd.DataFrame) -> np.ndarray:
    missing_count = sum(1 for column in FLOW_COLUMNS if column not in frame.columns)
    penalty = np.full(len(frame), 0.25 if missing_count else 0.0, dtype="float64")
    for column in ["bid_depth_10bps", "ask_depth_10bps", "bid_depth_25bps", "ask_depth_25bps"]:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
            penalty[(~np.isfinite(values)) | (values <= 0.0)] += 0.05
    return penalty


def scenario_costs(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    half = half_spread_bps(frame)
    depth = thin_depth_penalty(frame)
    imbalance = imbalance_penalty(frame)
    missing = missing_liquidity_penalty(frame)
    costs = {
        "fixed_0_05": np.full(len(frame), 0.05, dtype="float64"),
        "fixed_0_10": np.full(len(frame), 0.10, dtype="float64"),
        "fixed_0_15": np.full(len(frame), 0.15, dtype="float64"),
        "fixed_0_25": np.full(len(frame), 0.25, dtype="float64"),
        "half_spread_plus_0_05": half + 0.05,
        "half_spread_plus_depth_penalty": half + depth,
        "half_spread_plus_depth_and_imbalance_penalty": half + depth + imbalance,
        "conservative_missing_liquidity_penalty": half + missing,
    }
    for cost in COSTS:
        key = f"fixed_{str(cost).replace('.', '_')}"
        costs.setdefault(key, np.full(len(frame), cost, dtype="float64"))
    return costs


def build_combo_frame(combo: tuple[int, ...], candidates: pd.DataFrame, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    joined: pd.DataFrame | None = None
    for local_index, candidate_index in enumerate(combo, start=1):
        probe_dir = Path(str(candidates.iloc[candidate_index]["probe_dir"]))
        frame = cache.setdefault(str(probe_dir), read_annotated(probe_dir))
        rename = {
            PRED_COLUMN: f"pred_{local_index}",
            ACTUAL_COLUMN: f"actual_{local_index}",
            "time": f"time_{local_index}",
        }
        flow_rename = {column: f"{column}_{local_index}" for column in FLOW_COLUMNS if column in frame.columns}
        part = frame.rename(columns={**rename, **flow_rename})
        joined = part if joined is None else joined.merge(part, on="timestamp", how="inner")
    if joined is None or joined.empty:
        raise SystemExit(f"No aligned rows for combo {combo}")
    actual_columns = [f"actual_{index}" for index in range(1, len(combo) + 1)]
    joined[ACTUAL_COLUMN] = joined[actual_columns[0]]
    joined["time"] = joined[f"time_1"]
    for flow_column in FLOW_COLUMNS:
        for index in range(1, len(combo) + 1):
            candidate = f"{flow_column}_{index}"
            if candidate in joined.columns:
                joined[flow_column] = pd.to_numeric(joined[candidate], errors="coerce")
                break
    return joined.replace([np.inf, -np.inf], np.nan).dropna(subset=[f"pred_{i}" for i in range(1, len(combo) + 1)] + [ACTUAL_COLUMN])


def gate_mask(pred_matrix: np.ndarray, gate: str) -> np.ndarray:
    if gate == "none":
        return np.ones(pred_matrix.shape[0], dtype=bool)
    if gate == "all_positive" or gate == "min_pred_above_0":
        return np.all(pred_matrix > 0.0, axis=1)
    if gate == "min_pred_above_-0.1":
        return np.min(pred_matrix, axis=1) > -0.1
    if gate == "primary_above_0.2_secondary_above_-0.1":
        return (pred_matrix[:, 0] > 0.2) & np.all(pred_matrix[:, 1:] > -0.1, axis=1)
    if gate == "primary_above_0.3_secondary_above_0":
        return (pred_matrix[:, 0] > 0.3) & np.all(pred_matrix[:, 1:] > 0.0, axis=1)
    raise SystemExit(f"Unknown gate: {gate}")


def policy_mask_and_gross(pred: np.ndarray, actual: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if POLICY == "inverse_gt":
        return pred > threshold, -actual
    if POLICY == "direct_gt":
        return pred > threshold, actual
    if POLICY == "inverse_directional_abs_gt":
        return np.abs(pred) > threshold, -np.sign(pred) * actual
    raise SystemExit("RAWSEQ_ENSEMBLE_POLICY must be inverse_gt, direct_gt, or inverse_directional_abs_gt")


def classify(row: dict[str, Any]) -> tuple[str, list[str]]:
    fixed_010 = safe_float(row.get("fixed_0_10_cum_net"), 0.0)
    half_plus = safe_float(row.get("half_spread_plus_0_05_cum_net"), math.nan)
    selected_rows = safe_float(row.get("selected_rows"), 0.0)
    dip_ratio = safe_float(row.get("max_dip_to_cum_net_ratio"), math.inf)
    rolling_12h = safe_float(row.get("positive_12h_window_fraction"), 0.0)
    rolling_24h = safe_float(row.get("positive_24h_window_fraction"), 0.0)
    reasons: list[str] = []
    if fixed_010 <= 0.0:
        reasons.append("fixed_0_10_cum_net_nonpositive")
    if not math.isfinite(half_plus) or half_plus <= 0.0:
        reasons.append("half_spread_plus_0_05_cum_net_nonpositive_or_missing")
    if selected_rows < 300:
        reasons.append("selected_rows_lt_300")
    if not math.isfinite(dip_ratio):
        reasons.append("max_dip_to_cum_net_ratio_missing")
    elif dip_ratio > 1.0:
        reasons.append("max_dip_to_cum_net_ratio_gt_1")
    if rolling_12h < 0.5:
        reasons.append("rolling_12h_positive_fraction_lt_0_5")
    if rolling_24h < 0.5:
        reasons.append("rolling_24h_positive_fraction_lt_0_5")
    if fixed_010 <= 0.0 or selected_rows < 300:
        return "reject", reasons
    if half_plus > 0.0 and dip_ratio <= 1.0 and rolling_12h >= 0.5 and rolling_24h >= 0.5:
        return "clean_shadow_ensemble", reasons
    if half_plus > 0.0 and dip_ratio <= 2.0:
        return "robust_research_ensemble", reasons
    return "fragile_research_ensemble", reasons


def rank_score(row: dict[str, Any]) -> float:
    return (
        safe_float(row.get("conservative_missing_liquidity_penalty_cum_net"), -1_000_000.0) * 2.0
        + safe_float(row.get("fixed_0_25_cum_net"), -1_000_000.0)
        + safe_float(row.get("half_spread_plus_0_05_cum_net"), -1_000_000.0)
        + safe_float(row.get("fixed_0_10_cum_net"), -1_000_000.0)
        - min(safe_float(row.get("max_dip_to_cum_net_ratio"), 999.0), 20.0) * 100.0
        + (safe_float(row.get("positive_12h_window_fraction"), 0.0) + safe_float(row.get("positive_24h_window_fraction"), 0.0)) * 250.0
        + min(safe_float(row.get("selected_rows"), 0.0) / 300.0, 10.0) * 10.0
    )


def evaluate_combo(combo: tuple[int, ...], candidates: pd.DataFrame, cache: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    frame = build_combo_frame(combo, candidates, cache)
    pred_columns = [f"pred_{index}" for index in range(1, len(combo) + 1)]
    pred_matrix = frame[pred_columns].to_numpy(dtype="float64")
    actual = pd.to_numeric(frame[ACTUAL_COLUMN], errors="coerce").to_numpy(dtype="float64")
    timestamp = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype="int64")
    scenarios = scenario_costs(frame)
    gate_masks = {gate: gate_mask(pred_matrix, gate) for gate in GATES}
    candidate_rows = [candidates.iloc[index] for index in combo]
    metadata = {
        "ensemble_size": len(combo),
        "candidate_indices": ";".join(str(index) for index in combo),
        "probe_dirs": ";".join(str(row["probe_dir"]) for row in candidate_rows),
        "model_paths": ";".join(safe_str(row.get("model_path")) for row in candidate_rows),
        "registry_thresholds": ";".join(safe_str(row.get("threshold_bps")) for row in candidate_rows),
        "input_features": ";".join(safe_str(row.get("input_feature")) for row in candidate_rows),
        "ma_windows": ";".join(safe_str(row.get("ma_window")) for row in candidate_rows),
        "hiddens": ";".join(safe_str(row.get("hidden")) for row in candidate_rows),
        "seeds": ";".join(safe_str(row.get("seed")) for row in candidate_rows),
        "aligned_rows": len(frame),
    }
    rows: list[dict[str, Any]] = []
    for weights in weight_grid(len(combo)):
        ensemble_pred = pred_matrix @ np.asarray(weights, dtype="float64")
        for threshold in THRESHOLDS:
            policy_mask, gross_all = policy_mask_and_gross(ensemble_pred, actual, threshold)
            finite_mask = policy_mask & np.isfinite(gross_all)
            for gate, gate_values in gate_masks.items():
                mask = finite_mask & gate_values
                gross = gross_all[mask]
                selected_rows = int(len(gross))
                row: dict[str, Any] = {
                    **metadata,
                    "ensemble_id": f"ens_{'_'.join(str(index) for index in combo)}",
                    "weights": ";".join(f"{weight:.10g}" for weight in weights),
                    "threshold_bps": threshold,
                    "gate": gate,
                    "selected_rows": selected_rows,
                    "avg_gross_bps": float(np.mean(gross)) if selected_rows else math.nan,
                }
                if selected_rows:
                    fixed010_net = gross - scenarios["fixed_0_10"][mask]
                    row["avg_net_bps"] = float(np.mean(fixed010_net))
                    row["cum_net_bps"] = float(np.sum(fixed010_net))
                    row["win_rate_net"] = float(np.mean(fixed010_net > 0.0))
                    row["max_dip_net_bps"] = max_dip_bps(fixed010_net)
                    row["max_dip_to_cum_net_ratio"] = (
                        abs(row["max_dip_net_bps"]) / row["cum_net_bps"]
                        if row["cum_net_bps"] > 0.0 and math.isfinite(row["max_dip_net_bps"])
                        else math.inf
                    )
                    row.update(rolling_fractions(timestamp[mask], fixed010_net))
                else:
                    row.update(
                        {
                            "avg_net_bps": math.nan,
                            "cum_net_bps": 0.0,
                            "win_rate_net": math.nan,
                            "max_dip_net_bps": math.nan,
                            "max_dip_to_cum_net_ratio": math.inf,
                        }
                    )
                    row.update(rolling_fractions(timestamp[mask], np.asarray([], dtype="float64")))
                for scenario_name, cost_values in scenarios.items():
                    scenario_net = gross - cost_values[mask]
                    row[f"{scenario_name}_cum_net"] = float(np.sum(scenario_net)) if selected_rows else 0.0
                status, reasons = classify(row)
                row["status"] = status
                row["status_priority"] = STATUS_PRIORITY[status]
                row["rank_score"] = rank_score(row)
                row["rejection_reasons"] = ";".join(reasons)
                row["paper_only"] = True
                row["training"] = False
                row["champion_mutation"] = False
                row["promotion"] = False
                row["orders"] = False
                rows.append(row)
    return rows


def sort_ranking(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    numeric_columns = [
        "status_priority",
        "rank_score",
        "conservative_missing_liquidity_penalty_cum_net",
        "fixed_0_25_cum_net",
        "half_spread_plus_0_05_cum_net",
        "fixed_0_10_cum_net",
        "max_dip_to_cum_net_ratio",
        "positive_12h_window_fraction",
        "positive_24h_window_fraction",
        "selected_rows",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(
        [
            "status_priority",
            "conservative_missing_liquidity_penalty_cum_net",
            "fixed_0_25_cum_net",
            "half_spread_plus_0_05_cum_net",
            "fixed_0_10_cum_net",
            "max_dip_to_cum_net_ratio",
            "positive_12h_window_fraction",
            "positive_24h_window_fraction",
            "selected_rows",
        ],
        ascending=[True, False, False, False, False, True, False, False, False],
    )[OUTPUT_COLUMNS]


def render_row(row: pd.Series) -> str:
    return (
        f"  {row.get('status', '')} size={row.get('ensemble_size', '')} "
        f"weights={row.get('weights', '')} threshold={fmt(row.get('threshold_bps'), 3)} gate={row.get('gate', '')} "
        f"rows={int(safe_float(row.get('selected_rows'), 0))} fixed010={fmt(row.get('fixed_0_10_cum_net'))} "
        f"fixed025={fmt(row.get('fixed_0_25_cum_net'))} half+005={fmt(row.get('half_spread_plus_0_05_cum_net'))} "
        f"missing_liq={fmt(row.get('conservative_missing_liquidity_penalty_cum_net'))} "
        f"dip_ratio={fmt(row.get('max_dip_to_cum_net_ratio'), 3)} "
        f"roll12/24={fmt(row.get('positive_12h_window_fraction'), 3)}/{fmt(row.get('positive_24h_window_fraction'), 3)} "
        f"features={row.get('input_features', '')} thresholds={row.get('registry_thresholds', '')}"
    )


def render_section(lines: list[str], title: str, frame: pd.DataFrame, limit: int = 20) -> None:
    lines += ["", title]
    if frame.empty:
        lines.append("  none")
        return
    for _, row in frame.head(limit).iterrows():
        lines.append(render_row(row))


def render_text(ranking: pd.DataFrame, candidates: pd.DataFrame, output_dir: Path, registry_dir: Path) -> str:
    lines = [
        "Rawseq Registry Ensemble Batch",
        "",
        f"Registry dir: {registry_dir}",
        f"Output dir: {output_dir}",
        f"Candidate inputs: {len(candidates)}",
        f"Rows evaluated: {len(ranking)}",
        "",
        "1. Candidate Inputs Used",
    ]
    for index, row in candidates.iterrows():
        lines.append(
            f"  {index}: status={row.get('status', '')} threshold={row.get('threshold_bps', '')} "
            f"feature={row.get('input_feature', '')} ma={row.get('ma_window', '')} hidden={row.get('hidden', '')} "
            f"seed={row.get('seed', '')} probe={Path(str(row.get('probe_dir', ''))).name}"
        )
    render_section(lines, "2. Clean Shadow Ensembles", ranking[ranking["status"] == "clean_shadow_ensemble"])
    render_section(lines, "3. Robust Research Ensembles", ranking[ranking["status"] == "robust_research_ensemble"])
    render_section(lines, "4. Fragile Research Ensembles", ranking[ranking["status"] == "fragile_research_ensemble"])
    render_section(lines, "5. Best Rejects For Diagnostics", ranking[ranking["status"] == "reject"])
    lines += ["", "6. Best Individual Registry Candidate For Comparison"]
    if candidates.empty:
        lines.append("  none")
    else:
        best = candidates.iloc[0]
        lines.append(
            f"  status={best.get('status', '')} threshold={best.get('threshold_bps', '')} "
            f"feature={best.get('input_feature', '')} ma={best.get('ma_window', '')} hidden={best.get('hidden', '')} "
            f"fixed010={best.get('fixed_0_10_cum_net', '')} half+005={best.get('half_spread_plus_0_05_cum_net', '')} "
            f"probe={Path(str(best.get('probe_dir', ''))).name}"
        )
    lines += [
        "",
        "7. Warning",
        "  No champion creation/mutation from this batch.",
        "  This is paper-only shadow research.",
        "  Any freeze requires a separate explicit freeze/audit script.",
        "",
        f"Ranking CSV: {output_dir / 'registry_ensemble_batch_ranking.csv'}",
        f"Best contract JSON: {output_dir / 'best_registry_ensemble_contract.json'}",
        "Safety: no training. No champion mutation. No promotion. No orders.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    registry_dir = latest_registry_dir()
    candidates = load_registry_candidates(registry_dir).reset_index(drop=True)
    combos: list[tuple[int, ...]] = []
    for size in range(2, min(MAX_SIZE, len(candidates)) + 1):
        combos.extend(itertools.combinations(range(len(candidates)), size))
    output_dir = OUTPUT_ROOT / f"rawseq_registry_ensemble_batch_{now_stamp()}_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates.to_csv(output_dir / "candidate_inputs.csv", index=False)
    cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for combo_index, combo in enumerate(combos, start=1):
        print(f"Evaluating combo {combo_index}/{len(combos)}: {combo}", flush=True)
        rows.extend(evaluate_combo(combo, candidates, cache))
    ranking = sort_ranking(rows)
    ranking.to_csv(output_dir / "registry_ensemble_batch_ranking.csv", index=False)
    best = ranking.iloc[0].to_dict()
    best_contract = {
        "created_at": now_stamp(),
        "registry_dir": str(registry_dir),
        "status": safe_str(best.get("status")),
        "ensemble_size": safe_str(best.get("ensemble_size")),
        "candidate_indices": safe_str(best.get("candidate_indices")),
        "probe_dirs": safe_str(best.get("probe_dirs")).split(";"),
        "model_paths": safe_str(best.get("model_paths")).split(";"),
        "weights": safe_str(best.get("weights")),
        "threshold_bps": safe_float(best.get("threshold_bps")),
        "gate": safe_str(best.get("gate")),
        "policy": POLICY,
        "paper_only": True,
        "training": False,
        "champion_mutation": False,
        "promotion": False,
        "orders": False,
    }
    (output_dir / "best_registry_ensemble_contract.json").write_text(
        json.dumps(best_contract, indent=2, sort_keys=True), encoding="utf-8"
    )
    text = render_text(ranking, candidates, output_dir, registry_dir)
    (output_dir / "registry_ensemble_batch_ranking.txt").write_text(text, encoding="utf-8")
    print(text)
    print("Top 20 Ensemble Rows")
    preview_columns = [
        "status",
        "ensemble_size",
        "weights",
        "threshold_bps",
        "gate",
        "selected_rows",
        "fixed_0_10_cum_net",
        "fixed_0_25_cum_net",
        "half_spread_plus_0_05_cum_net",
        "conservative_missing_liquidity_penalty_cum_net",
        "max_dip_to_cum_net_ratio",
        "positive_12h_window_fraction",
        "positive_24h_window_fraction",
        "input_features",
        "registry_thresholds",
    ]
    print(ranking[preview_columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
