#!/usr/bin/env python
"""Compare paper-only rawseq and ladder policies on identical recorded slices."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = Path(os.getenv("POLICY_COMPARE_SOURCE_PATH", PROJECT_ROOT / "data" / "realtime" / "kraken" / "SOLUSDT_10s_flow.csv"))
OUTPUT_ROOT = Path(os.getenv("POLICY_COMPARE_OUTPUT_DIR", PROJECT_ROOT / "data" / "research" / "policy_comparisons"))
SMOKE_MODE = os.getenv("POLICY_COMPARE_SMOKE_MODE", "").strip().lower() in {"1", "true", "yes", "y"}
SLICE_ROWS = int(float(os.getenv("POLICY_COMPARE_SLICE_ROWS", "20000" if SMOKE_MODE else "200000")))
STEP_ROWS = int(float(os.getenv("POLICY_COMPARE_STEP_ROWS", str(SLICE_ROWS if SMOKE_MODE else 200000))))
MAX_SLICES = int(float(os.getenv("POLICY_COMPARE_MAX_SLICES", "2" if SMOKE_MODE else "4")))
LADDER_CONTRACT_PATH_ENV = os.getenv("POLICY_COMPARE_LADDER_CONTRACT_PATH", "").strip()
RAWSEQ_SHADOW_DIRS_ENV = os.getenv("POLICY_COMPARE_RAWSEQ_SHADOW_DIRS", "").strip()
COST_BPS = float(os.getenv("POLICY_COMPARE_COST_BPS", "0.1"))
PROGRESS_EVERY = int(float(os.getenv("POLICY_COMPARE_PROGRESS_EVERY", "1")))
MIN_TRADE_COUNT_FOR_CANDIDATE = int(float(os.getenv("POLICY_COMPARE_MIN_TRADE_COUNT_FOR_CANDIDATE", "30")))
MIN_POSITIVE_SLICES_FOR_CANDIDATE = int(float(os.getenv("POLICY_COMPARE_MIN_POSITIVE_SLICES_FOR_CANDIDATE", "2")))
MIN_GATE_PASS_FRACTION_TEXT = os.getenv("POLICY_COMPARE_MIN_GATE_PASS_FRACTION", "").strip()
MIN_GATE_PASS_FRACTION = float(MIN_GATE_PASS_FRACTION_TEXT) if MIN_GATE_PASS_FRACTION_TEXT else math.nan

POLICIES = [
    "rawseq_model_only",
    "ladder_only",
    "rawseq_gate_ladder_strict",
    "rawseq_gate_ladder_medium",
    "rawseq_gate_ladder_loose",
    "rawseq_suppress_ladder",
    "rawseq_emergency_filter",
]


def load_simulator_module():
    path = PROJECT_ROOT / "scripts" / "tiny" / "simulate_path_aware_ladder_baseline.py"
    spec = importlib.util.spec_from_file_location("ladder_sim_policy_compare", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import ladder simulator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def utc_slug() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:8]
    return f"policy_comparison_{stamp}_{token}"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def latest_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if not matches:
        raise SystemExit(f"No {pattern} found under {root}")
    return matches[0]


def latest_shadow_dir() -> Path | None:
    root = PROJECT_ROOT / "data" / "research" / "rawseq_shadow_candidates"
    if not root.exists():
        return None
    matches = [p for p in root.iterdir() if p.is_dir() and (p / "model.json").exists()]
    if not matches:
        return None
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def shadow_dirs() -> list[Path]:
    if RAWSEQ_SHADOW_DIRS_ENV:
        return [Path(part.strip()) for part in RAWSEQ_SHADOW_DIRS_ENV.split(";") if part.strip()]
    latest = latest_shadow_dir()
    return [latest] if latest else []


def ladder_contract_path() -> Path:
    if LADDER_CONTRACT_PATH_ENV:
        return Path(LADDER_CONTRACT_PATH_ENV)
    sweep_root = PROJECT_ROOT / "data" / "research" / "ladder_risk_walkforward_sweeps"
    baseline_root = PROJECT_ROOT / "data" / "research" / "ladder_baselines"
    candidates: list[Path] = []
    if sweep_root.exists():
        candidates.extend(sweep_root.glob("*/best_ladder_walkforward_contract.json"))
    if baseline_root.exists():
        candidates.extend(baseline_root.glob("*/best_ladder_contract.json"))
    if not candidates:
        raise SystemExit("No ladder contract found. Set POLICY_COMPARE_LADDER_CONTRACT_PATH.")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def selected_threshold(shadow_dir: Path) -> float:
    provenance = load_json(shadow_dir / "provenance.json")
    registry_row = shadow_dir / "registry_row.csv"
    for key in ["threshold_bps", "selected_threshold_bps", "decision_threshold_bps"]:
        value = provenance.get(key)
        if value not in (None, ""):
            return float(value)
    if registry_row.exists():
        try:
            rows = list(csv.DictReader(registry_row.open("r", encoding="utf-8", newline="")))
            if rows:
                for key in ["threshold_bps", "best_threshold", "selected_threshold_bps"]:
                    value = rows[0].get(key)
                    if value not in (None, ""):
                        return float(value)
        except Exception:
            pass
    for path in sorted((shadow_dir / "reports").glob("decision_summary_threshold_*.txt")):
        stem = path.stem.rsplit("_", 1)[-1]
        try:
            return float(stem)
        except ValueError:
            continue
    return 0.1


def slice_bounds(total_rows: int) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    while start + SLICE_ROWS <= total_rows and len(bounds) < MAX_SLICES:
        bounds.append((start, start + SLICE_ROWS))
        start += STEP_ROWS
    return bounds


def max_dip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    cumulative = np.cumsum(values)
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def classify_policy(row: dict[str, Any]) -> tuple[str, str, bool]:
    total_net = row_metric(row, "force_flat_total_net_bps")
    cost025 = row_metric(row, "total_net_cost_0_25_bps")
    positive_slices = int(row_metric(row, "positive_slices"))
    positive_fraction = row_metric(row, "positive_slice_fraction")
    worst_slice = row_metric(row, "worst_slice_net_bps")
    max_drawdown = row_metric(row, "max_drawdown_bps")
    trade_count = int(row_metric(row, "trade_count"))
    gate_pass_fraction = row_metric(row, "rawseq_gate_pass_fraction", math.nan)
    reasons: list[str] = []
    if trade_count < MIN_TRADE_COUNT_FOR_CANDIDATE:
        reasons.append(f"trade_count<{MIN_TRADE_COUNT_FOR_CANDIDATE}")
    if positive_slices < MIN_POSITIVE_SLICES_FOR_CANDIDATE:
        reasons.append(f"positive_slices<{MIN_POSITIVE_SLICES_FOR_CANDIDATE}")
    if math.isfinite(MIN_GATE_PASS_FRACTION) and gate_pass_fraction < MIN_GATE_PASS_FRACTION:
        reasons.append(f"gate_pass_fraction<{MIN_GATE_PASS_FRACTION:g}")
    if reasons:
        return "insufficient_sample", ";".join(reasons), False
    meaningful = True
    drawdown_ok = max_drawdown >= -2.0 * max(total_net, 1.0)
    if (
        total_net > 0
        and cost025 > 0
        and positive_slices >= MIN_POSITIVE_SLICES_FOR_CANDIDATE
        and positive_fraction >= 0.75
        and worst_slice > -150
        and drawdown_ok
    ):
        return "stable_candidate", "", meaningful
    if total_net > 0:
        return "fragile_candidate", "", meaningful
    return "reject", "", meaningful


def row_metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        value = default
    return value if math.isfinite(value) else default


def rawseq_model_only(
    frame: pd.DataFrame,
    gate: pd.DataFrame,
    threshold: float,
    cost_bps: float,
    bucket_seconds: int,
) -> dict[str, Any]:
    prices = frame["price"].to_numpy(dtype=np.float64)
    pred_final = gate["pred_final"].to_numpy(dtype=np.float64)
    pred_max_up = gate["pred_max_up"].to_numpy(dtype=np.float64)
    pred_max_down = gate["pred_max_down"].to_numpy(dtype=np.float64)
    selected = np.isfinite(pred_final) & (pred_final > threshold)
    horizon = 1
    actual = np.full(len(frame), np.nan, dtype=np.float64)
    if len(frame) > horizon:
        actual[:-horizon] = 10_000.0 * np.log(prices[horizon:] / prices[:-horizon])
    net = np.where(selected & np.isfinite(actual), actual - cost_bps, np.nan)
    selected_net = net[np.isfinite(net)]
    selected_actual = actual[selected & np.isfinite(actual)]
    selected_rows = int(len(selected_net))
    cum_net = float(np.sum(selected_net)) if selected_rows else 0.0
    avg_net = float(np.mean(selected_net)) if selected_rows else math.nan
    avg_gross = float(np.mean(selected_actual)) if selected_rows else math.nan
    win_rate = float(np.mean(selected_net > 0)) if selected_rows else math.nan
    return {
        "total_net_bps": cum_net,
        "force_flat_total_net_bps": cum_net,
        "total_net_cost_0_25_bps": float(np.sum(selected_actual - 0.25)) if selected_rows else 0.0,
        "max_drawdown_bps": max_dip(selected_net),
        "trade_count": selected_rows,
        "rawseq_selected_rows": selected_rows,
        "rawseq_gate_pass_fraction": float(np.mean(selected)) if len(selected) else math.nan,
        "avg_net_bps": avg_net,
        "avg_gross_bps": avg_gross,
        "win_rate": win_rate,
        "exposure_time_fraction": selected_rows / max(1, len(frame)),
        "take_profit_exit_count": 0,
        "stop_loss_exit_count": 0,
        "timeout_exit_count": 0,
        "final_liquidation_count": 0,
        "timeout_churn_ratio": 0.0,
        "stop_loss_churn_ratio": 0.0,
        "emergency_rebalance_count": 0,
        "rungs_refreshed_by_emergency_count": 0,
        "bucket_seconds": bucket_seconds,
        "pred_max_up_mean": float(np.nanmean(pred_max_up)) if np.isfinite(pred_max_up).any() else math.nan,
        "pred_max_down_mean": float(np.nanmean(pred_max_down)) if np.isfinite(pred_max_down).any() else math.nan,
    }


def gate_mode(policy: str) -> str:
    if policy.endswith("_strict"):
        return "strict"
    if policy.endswith("_medium"):
        return "medium"
    if policy.endswith("_loose"):
        return "loose"
    if policy == "rawseq_suppress_ladder":
        return "suppress_only"
    return ""


def make_gate_fn(policy: str, stop_loss_bps: float) -> Callable[[str, float, float, float, float], bool]:
    mode_name = gate_mode(policy)

    def gate_fn(mode: str, pred_final: float, pred_max_up: float, pred_max_down: float, take_profit_bps: float) -> bool:
        if not all(math.isfinite(x) for x in [pred_final, pred_max_up, pred_max_down]):
            return False
        stop_loss_scale = max(stop_loss_bps, 2.0 * take_profit_bps, 1.0)
        if mode_name == "strict":
            return pred_max_up >= take_profit_bps and pred_max_down >= -stop_loss_scale and pred_final >= -0.5 * take_profit_bps
        if mode_name == "medium":
            return pred_max_up >= 0.75 * take_profit_bps and pred_max_down >= -1.25 * stop_loss_scale
        if mode_name == "loose":
            return pred_max_up >= 0.5 * take_profit_bps and pred_max_down >= -1.5 * stop_loss_scale
        if mode_name == "suppress_only":
            danger = pred_max_down < -1.5 * stop_loss_scale or pred_final < -1.0 * take_profit_bps
            return not danger
        if policy == "rawseq_emergency_filter":
            danger = pred_max_down < -1.5 * stop_loss_scale or pred_final < -1.0 * take_profit_bps
            return not danger
        return True

    return gate_fn


def run_ladder_policy(
    sim: Any,
    frame: pd.DataFrame,
    indicators: dict[str, np.ndarray],
    contract: dict[str, Any],
    gate: pd.DataFrame | None,
    policy: str,
) -> dict[str, Any]:
    local_contract = copy.deepcopy(contract)
    local_contract["cost_bps"] = COST_BPS
    if policy == "rawseq_emergency_filter":
        local_contract["emergency_rebalance_mode"] = "off"
    stop_loss_bps = float(local_contract.get("stop_loss_bps") or local_contract.get("min_stop_floor_bps") or 0.0)
    original_mode = getattr(sim, "MODEL_GATE_MODE", "none")
    original_gate_fn = sim.model_gate_allows
    original_anchor_windows = list(getattr(sim, "ANCHOR_WINDOWS", []))
    original_vol_windows = list(getattr(sim, "VOL_WINDOWS", []))
    try:
        anchor_window = int(float(local_contract.get("anchor_window", original_anchor_windows[0] if original_anchor_windows else 300)))
        vol_window = int(float(local_contract.get("vol_window", original_vol_windows[0] if original_vol_windows else 300)))
        sim.ANCHOR_WINDOWS = sorted(set([*original_anchor_windows, anchor_window]))
        sim.VOL_WINDOWS = sorted(set([*original_vol_windows, vol_window]))
        if policy == "ladder_only":
            sim.MODEL_GATE_MODE = "none"
            local_gate = None
        else:
            sim.MODEL_GATE_MODE = "policy_compare"
            sim.model_gate_allows = make_gate_fn(policy, stop_loss_bps)
            local_gate = gate
        result, _, _ = sim.simulate_config(frame, indicators, local_gate, local_contract, save_paths=False)
    finally:
        sim.MODEL_GATE_MODE = original_mode
        sim.model_gate_allows = original_gate_fn
        sim.ANCHOR_WINDOWS = original_anchor_windows
        sim.VOL_WINDOWS = original_vol_windows
    return result


def normalize_ladder_result(result: dict[str, Any]) -> dict[str, Any]:
    total_net = row_metric(result, "force_flat_total_net_bps", row_metric(result, "total_net_bps"))
    exits = sum(row_metric(result, key) for key in ["take_profit_exit_count", "stop_loss_exit_count", "timeout_exit_count", "final_liquidation_count"])
    return {
        "total_net_bps": total_net,
        "force_flat_total_net_bps": total_net,
        "total_net_cost_0_25_bps": row_metric(result, "total_net_cost_0_25_bps"),
        "max_drawdown_bps": row_metric(result, "max_drawdown_bps", math.nan),
        "trade_count": int(row_metric(result, "number_of_trades")),
        "exposure_time_fraction": row_metric(result, "exposure_time_fraction", math.nan),
        "take_profit_exit_count": int(row_metric(result, "take_profit_exit_count")),
        "stop_loss_exit_count": int(row_metric(result, "stop_loss_exit_count")),
        "timeout_exit_count": int(row_metric(result, "timeout_exit_count")),
        "final_liquidation_count": int(row_metric(result, "final_liquidation_count")),
        "timeout_churn_ratio": row_metric(result, "timeout_exit_count") / max(1.0, exits),
        "stop_loss_churn_ratio": row_metric(result, "stop_loss_exit_count") / max(1.0, exits),
        "emergency_rebalance_count": int(row_metric(result, "emergency_rebalance_count")),
        "rungs_refreshed_by_emergency_count": int(row_metric(result, "rungs_refreshed_by_emergency_count")),
        "rawseq_selected_rows": int(row_metric(result, "rows_gate_passed")),
        "rawseq_gate_pass_fraction": row_metric(result, "gate_pass_fraction", math.nan),
        "avg_net_bps": row_metric(result, "average_trade_net_bps", math.nan),
        "win_rate": row_metric(result, "win_rate", math.nan),
    }


def aggregate(policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    nets = np.asarray([row_metric(row, "force_flat_total_net_bps") for row in policy_rows], dtype=np.float64)
    cost025 = np.asarray([row_metric(row, "total_net_cost_0_25_bps") for row in policy_rows], dtype=np.float64)
    drawdowns = np.asarray([row_metric(row, "max_drawdown_bps", 0.0) for row in policy_rows], dtype=np.float64)
    total = float(np.nansum(nets))
    positive_fraction = float(np.nanmean(nets > 0)) if len(nets) else math.nan
    cost025_fraction = float(np.nanmean(cost025 > 0)) if len(cost025) else math.nan
    worst = float(np.nanmin(nets)) if len(nets) else math.nan
    max_drawdown = float(np.nanmin(drawdowns)) if len(drawdowns) else math.nan
    out = {
        "policy": policy_rows[0]["policy"] if policy_rows else "",
        "rawseq_shadow_dir": policy_rows[0].get("rawseq_shadow_dir", "") if policy_rows else "",
        "slices": len(policy_rows),
        "total_net_bps": total,
        "force_flat_total_net_bps": total,
        "total_net_cost_0_25_bps": float(np.nansum(cost025)),
        "positive_slice_fraction": positive_fraction,
        "positive_slices": int(np.nansum(nets > 0)),
        "worst_slice_net_bps": worst,
        "max_drawdown_bps": max_drawdown,
        "trade_count": int(sum(row_metric(row, "trade_count") for row in policy_rows)),
        "trades_per_slice": float(sum(row_metric(row, "trade_count") for row in policy_rows) / max(1, len(policy_rows))),
        "exposure_time_fraction": float(np.nanmean([row_metric(row, "exposure_time_fraction", math.nan) for row in policy_rows])),
        "take_profit_exit_count": int(sum(row_metric(row, "take_profit_exit_count") for row in policy_rows)),
        "stop_loss_exit_count": int(sum(row_metric(row, "stop_loss_exit_count") for row in policy_rows)),
        "timeout_exit_count": int(sum(row_metric(row, "timeout_exit_count") for row in policy_rows)),
        "final_liquidation_count": int(sum(row_metric(row, "final_liquidation_count") for row in policy_rows)),
        "timeout_churn_ratio": float(np.nanmean([row_metric(row, "timeout_churn_ratio", math.nan) for row in policy_rows])),
        "stop_loss_churn_ratio": float(np.nanmean([row_metric(row, "stop_loss_churn_ratio", math.nan) for row in policy_rows])),
        "emergency_rebalance_count": int(sum(row_metric(row, "emergency_rebalance_count") for row in policy_rows)),
        "rungs_refreshed_by_emergency_count": int(sum(row_metric(row, "rungs_refreshed_by_emergency_count") for row in policy_rows)),
        "rawseq_selected_rows": int(sum(row_metric(row, "rawseq_selected_rows") for row in policy_rows)),
        "rawseq_gate_pass_fraction": float(np.nanmean([row_metric(row, "rawseq_gate_pass_fraction", math.nan) for row in policy_rows])),
        "min_trade_count_for_candidate": MIN_TRADE_COUNT_FOR_CANDIDATE,
    }
    status, reason, meaningful = classify_policy(out)
    out["policy_classification"] = status
    out["insufficient_sample_reason"] = reason
    out["meaningful_candidate"] = meaningful
    out["risk_adjusted_score"] = risk_adjusted_score(out)
    return out


def risk_adjusted_score(row: dict[str, Any]) -> float:
    total = row_metric(row, "force_flat_total_net_bps")
    cost025 = row_metric(row, "total_net_cost_0_25_bps")
    pos_frac = row_metric(row, "positive_slice_fraction")
    worst = row_metric(row, "worst_slice_net_bps")
    drawdown = row_metric(row, "max_drawdown_bps")
    exposure = row_metric(row, "exposure_time_fraction")
    trade_count = row_metric(row, "trade_count")
    timeout_ratio = row_metric(row, "timeout_churn_ratio")
    stop_ratio = row_metric(row, "stop_loss_churn_ratio")
    score = total + 0.25 * cost025 + 75.0 * pos_frac + 0.25 * worst + 0.25 * drawdown
    score += min(trade_count, MIN_TRADE_COUNT_FOR_CANDIDATE) * 0.5
    score -= timeout_ratio * 50.0
    score -= stop_ratio * 75.0
    if trade_count < MIN_TRADE_COUNT_FOR_CANDIDATE:
        score -= (MIN_TRADE_COUNT_FOR_CANDIDATE - trade_count) * 5.0
    if exposure > 0.25:
        score -= (exposure - 0.25) * 200.0
    if total > 0 and drawdown < -2.0 * max(total, 1.0):
        score -= abs(drawdown + 2.0 * max(total, 1.0)) * 0.5
    if worst < -150:
        score -= abs(worst + 150.0) * 0.5
    return float(score)


def rank_score(row: dict[str, Any]) -> float:
    status_bonus = {
        "stable_candidate": 300.0,
        "fragile_candidate": 100.0,
        "insufficient_sample": -100.0,
        "reject": -200.0,
    }.get(str(row.get("policy_classification")), -200.0)
    return status_bonus + row_metric(row, "risk_adjusted_score")


def write_text(path: Path, summary_rows: list[dict[str, Any]], per_slice_rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    lines = [
        "Rawseq vs ladder policy comparison",
        "",
        f"source_path={meta['source_path']}",
        f"ladder_contract_path={meta['ladder_contract_path']}",
        f"rawseq_shadow_dirs={';'.join(meta['rawseq_shadow_dirs'])}",
        f"slice_rows={SLICE_ROWS}",
        f"step_rows={STEP_ROWS}",
        f"max_slices={MAX_SLICES}",
        f"smoke_mode={SMOKE_MODE}",
        f"progress_every={PROGRESS_EVERY}",
        f"runtime_seconds={meta.get('runtime_seconds')}",
        f"min_trade_count_for_candidate={MIN_TRADE_COUNT_FOR_CANDIDATE}",
        f"min_positive_slices_for_candidate={MIN_POSITIVE_SLICES_FOR_CANDIDATE}",
        f"min_gate_pass_fraction={MIN_GATE_PASS_FRACTION if math.isfinite(MIN_GATE_PASS_FRACTION) else 'none'}",
        f"cost_bps={COST_BPS}",
        "",
        "Safety:",
        "  paper_only=true",
        "  public_recorded_data_only=true",
        "  orders=false",
        "  training=false",
        "  promotion=false",
        "  champion_mutation=false",
        "",
        "Ranked policy summary:",
    ]
    for row in summary_rows:
        lines.append(
            "  "
            f"{row.get('policy')} shadow={Path(str(row.get('rawseq_shadow_dir', ''))).name or 'none'} "
            f"status={row.get('policy_classification')} net={row.get('force_flat_total_net_bps'):.3f} "
            f"cost025={row.get('total_net_cost_0_25_bps'):.3f} pos={row.get('positive_slice_fraction'):.3f} "
            f"worst={row.get('worst_slice_net_bps'):.3f} dd={row.get('max_drawdown_bps'):.3f} "
            f"trades={row.get('trade_count')} trades_per_slice={row.get('trades_per_slice'):.3f} "
            f"gate={row.get('rawseq_gate_pass_fraction'):.3f} meaningful={row.get('meaningful_candidate')} "
            f"risk_score={row.get('risk_adjusted_score'):.3f} reason={row.get('insufficient_sample_reason') or 'none'} "
            f"emergency={row.get('emergency_rebalance_count')} refreshed={row.get('rungs_refreshed_by_emergency_count')}"
        )
    lines.extend(["", "Per-slice row count:", f"  {len(per_slice_rows)}"])
    lines.extend(
        [
            "",
            "Notes:",
            "  rawseq_gate_ladder_strict requires predicted upside >= current take-profit, bounded downside, and no strongly adverse final prediction.",
            "  rawseq_gate_ladder_medium uses 0.75x take-profit upside and 1.25x stop-loss downside tolerance.",
            "  rawseq_gate_ladder_loose uses 0.5x take-profit upside and 1.5x stop-loss downside tolerance.",
            "  rawseq_suppress_ladder runs the ladder normally except when rawseq predicts adverse continuation.",
            "  rawseq_emergency_filter is implemented as a conservative comparison variant that disables emergency refreshes under the fixed rawseq adverse filter.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_progress(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame().to_csv(path, index=False)


def main() -> int:
    started = time.perf_counter()
    sim = load_simulator_module()
    frame = sim.load_price_data(SOURCE_PATH)
    bounds = slice_bounds(len(frame))
    if not bounds:
        raise SystemExit(f"Not enough rows for slices: rows={len(frame)} slice_rows={SLICE_ROWS}")

    contract_path = ladder_contract_path()
    contract = load_json(contract_path)
    if not contract:
        raise SystemExit(f"Could not load ladder contract: {contract_path}")

    shadows = shadow_dirs()
    if not shadows:
        raise SystemExit("No frozen rawseq shadow candidates found. Set POLICY_COMPARE_RAWSEQ_SHADOW_DIRS.")

    out_dir = OUTPUT_ROOT / utc_slug()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_slice_rows: list[dict[str, Any]] = []
    bucket_seconds = int(float(getattr(sim, "BUCKET_SECONDS", 10)))
    progress_path = out_dir / "policy_compare_progress.csv"
    gate_cache: dict[tuple[int, str], pd.DataFrame] = {}
    indicator_cache: dict[int, dict[str, np.ndarray]] = {}
    slice_cache: dict[int, pd.DataFrame] = {}
    completed = 0
    for shadow_dir in shadows:
        threshold = selected_threshold(shadow_dir)
        for slice_index, (start, end) in enumerate(bounds):
            if slice_index not in slice_cache:
                source_slice = frame.iloc[start:end].reset_index(drop=True).copy()
                slice_cache[slice_index] = source_slice
                contract_anchor = int(float(contract.get("anchor_window", 300)))
                contract_vol = int(float(contract.get("vol_window", 300)))
                sim.ANCHOR_WINDOWS = sorted(set([*getattr(sim, "ANCHOR_WINDOWS", []), contract_anchor]))
                sim.VOL_WINDOWS = sorted(set([*getattr(sim, "VOL_WINDOWS", []), contract_vol]))
                indicator_cache[slice_index] = sim.precompute_indicators(source_slice)
            source_slice = slice_cache[slice_index]
            gate_key = (slice_index, str(shadow_dir))
            if gate_key not in gate_cache:
                gate_cache[gate_key] = sim.build_shadow_gate(source_slice, shadow_dir)
            gate = gate_cache[gate_key]
            indicators = indicator_cache[slice_index]
            start_time = source_slice["timestamp"].iloc[0] if len(source_slice) else math.nan
            end_time = source_slice["timestamp"].iloc[-1] if len(source_slice) else math.nan
            for policy in POLICIES:
                if policy == "rawseq_model_only":
                    metrics = rawseq_model_only(source_slice, gate, threshold, COST_BPS, bucket_seconds)
                else:
                    ladder_result = run_ladder_policy(sim, source_slice, indicators, contract, gate, policy)
                    metrics = normalize_ladder_result(ladder_result)
                row = {
                    "policy": policy,
                    "slice_index": slice_index,
                    "slice_start_row": start,
                    "slice_end_row": end,
                    "start_time": start_time,
                    "end_time": end_time,
                    "rows": len(source_slice),
                    "rawseq_shadow_dir": str(shadow_dir),
                    "rawseq_shadow_name": shadow_dir.name,
                    "rawseq_threshold_bps": threshold,
                    "ladder_contract_path": str(contract_path),
                    "cost_bps": COST_BPS,
                    "paper_only": True,
                    "public_recorded_data_only": True,
                    "orders": False,
                    "training": False,
                    "promotion": False,
                    "champion_mutation": False,
                    **metrics,
                }
                per_slice_rows.append(row)
                completed += 1
                if PROGRESS_EVERY > 0 and completed % PROGRESS_EVERY == 0:
                    write_progress(progress_path, per_slice_rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_slice_rows:
        grouped.setdefault((row["policy"], row["rawseq_shadow_dir"]), []).append(row)
    summary_rows = [aggregate(rows) for rows in grouped.values()]
    for row in summary_rows:
        row["rank_score"] = rank_score(row)
    summary_rows = sorted(summary_rows, key=lambda row: row["rank_score"], reverse=True)

    per_slice_path = out_dir / "policy_comparison_per_slice.csv"
    summary_path = out_dir / "policy_comparison_summary.csv"
    text_path = out_dir / "policy_comparison.txt"
    best_path = out_dir / "best_policy_contract.json"
    pd.DataFrame(per_slice_rows).to_csv(per_slice_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    meta = {
        "source_path": str(SOURCE_PATH),
        "ladder_contract_path": str(contract_path),
        "rawseq_shadow_dirs": [str(path) for path in shadows],
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    write_text(text_path, summary_rows, per_slice_rows, meta)
    best = summary_rows[0] if summary_rows else {}
    best_payload = {
        **best,
        "source_path": str(SOURCE_PATH),
        "ladder_contract_path": str(contract_path),
        "rawseq_shadow_dirs": [str(path) for path in shadows],
        "output_dir": str(out_dir),
        "paper_only": True,
        "public_recorded_data_only": True,
        "orders": False,
        "training": False,
        "promotion": False,
        "champion_mutation": False,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    best_path.write_text(json.dumps(json_safe(best_payload), indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Wrote {per_slice_path}")
    print(f"Wrote {text_path}")
    print(f"Wrote {best_path}")
    print(f"Wrote {progress_path}")
    print(f"Runtime seconds: {time.perf_counter() - started:.3f}")
    print(pd.DataFrame(summary_rows).head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
