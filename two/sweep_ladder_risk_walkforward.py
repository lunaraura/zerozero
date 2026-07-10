#!/usr/bin/env python3
"""Sweep risk-controlled ladder configs across recorded walk-forward slices."""

from __future__ import annotations

import importlib.util
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
SOURCE_PATH = Path(os.getenv("LADDER_SOURCE_PATH", PROJECT_ROOT / "data" / "realtime" / "kraken" / "SOLUSDT_10s_flow.csv"))
OUTPUT_ROOT = Path(os.getenv("LADDER_SWEEP_OUTPUT_DIR", PROJECT_ROOT / "data" / "research" / "ladder_risk_walkforward_sweeps"))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def env_int(name: str, default: int, smoke_default: int | None = None) -> int:
    if name in os.environ:
        return int(float(os.environ[name]))
    if SMOKE_MODE and smoke_default is not None:
        return smoke_default
    return default


SMOKE_MODE = env_bool("LADDER_SWEEP_SMOKE_MODE", False)
SLICE_ROWS = env_int("LADDER_SLICE_ROWS", 200000, 50000)
STEP_ROWS = int(float(os.getenv("LADDER_STEP_ROWS", "200000")))
MAX_SLICES = env_int("LADDER_MAX_SLICES", 10, 2)
MAX_CONFIGS = env_int("LADDER_MAX_CONFIGS", 500, 25)
PROGRESS_EVERY = int(float(os.getenv("LADDER_SWEEP_PROGRESS_EVERY", "25")))
MAX_ROWS_PER_SLICE_ENV = os.getenv("LADDER_SWEEP_MAX_ROWS_PER_SLICE", "").strip()
MAX_ROWS_PER_SLICE = int(float(MAX_ROWS_PER_SLICE_ENV)) if MAX_ROWS_PER_SLICE_ENV else 0
SIM_PATH = PROJECT_ROOT / "scripts" / "tiny" / "simulate_path_aware_ladder_baseline.py"

ANCHOR_WINDOWS = [60, 150, 300]
VOL_WINDOWS = [60, 150, 300]
MIN_SPACING_BPS = [20.0, 40.0, 60.0, 80.0]
MIN_SPACING_FLOOR_MODES = ["fixed_bps", "spread_multiple", "tick_multiple", "max_fixed_or_spread"]
SPACING_VOL_MULTS = [1.0, 2.0, 4.0]
MAX_OPEN_UNITS = [1, 2]
STOP_LOSS_BPS = [60.0, 80.0, 120.0, 160.0, 240.0]
STOP_LOSS_VOL_MULTS = [4.0, 8.0, 12.0]
MAX_HOLD_BUCKETS = [60, 180, 360, 720]
TAKE_PROFIT_SPACING_MULTS = [1.0, 1.25, 1.5]
COOLDOWNS = [30, 60, 180]
ANCHOR_DISTANCE_LIMITS = [60.0, 120.0, 240.0]
EMA_SLOPE_LIMITS = [-2.0, 0.0, 2.0]
REBALANCE_INTERVAL_BUCKETS = [30, 90, 180, 360]
REBALANCE_MODES = ["fixed_interval"]


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def safe_int(value: Any, default: int = 0) -> int:
    value = safe_float(value)
    return int(value) if math.isfinite(value) else default


def load_simulator_module():
    spec = importlib.util.spec_from_file_location("ladder_sim", SIM_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load simulator module: {SIM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_dir() -> Path:
    path = OUTPUT_ROOT / f"ladder_risk_walkforward_sweep_{stamp()}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def timestamp_iso(value: Any) -> str:
    value = safe_float(value)
    if not math.isfinite(value):
        return ""
    unit = "ms" if value > 1e11 else "s"
    return pd.to_datetime(value, unit=unit, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_configs() -> list[dict[str, Any]]:
    configs = []
    for anchor_window in ANCHOR_WINDOWS:
        for vol_window in VOL_WINDOWS:
            for spacing_vol_mult in SPACING_VOL_MULTS:
                for min_spacing in MIN_SPACING_BPS:
                    for min_spacing_floor_mode in MIN_SPACING_FLOOR_MODES:
                        for max_units in MAX_OPEN_UNITS:
                            for stop_loss in STOP_LOSS_BPS:
                                for stop_loss_vol_mult in STOP_LOSS_VOL_MULTS:
                                    for max_hold in MAX_HOLD_BUCKETS:
                                        for take_profit_spacing_mult in TAKE_PROFIT_SPACING_MULTS:
                                            for cooldown in COOLDOWNS:
                                                for anchor_distance in ANCHOR_DISTANCE_LIMITS:
                                                    for ema_slope in EMA_SLOPE_LIMITS:
                                                        for rebalance_interval in REBALANCE_INTERVAL_BUCKETS:
                                                            for rebalance_mode in REBALANCE_MODES:
                                                                configs.append(
                                                                    {
                                                                        "anchor_mode": "ema",
                                                                        "anchor_window": anchor_window,
                                                                        "vol_window": vol_window,
                                                                        "spacing_mode": "volatility_scaled",
                                                                        "vol_mult": spacing_vol_mult,
                                                                        "spacing_vol_mult": spacing_vol_mult,
                                                                        "rung_count": 3,
                                                                        "take_profit_mult": take_profit_spacing_mult,
                                                                        "take_profit_spacing_mult": take_profit_spacing_mult,
                                                                        "min_spacing_bps": min_spacing,
                                                                        "min_spacing_floor_mode": min_spacing_floor_mode,
                                                                        "cost_bps": 0.1,
                                                                        "max_open_units": max_units,
                                                                        "stop_loss_bps": stop_loss,
                                                                        "min_stop_floor_bps": stop_loss,
                                                                        "stop_loss_vol_mult": stop_loss_vol_mult,
                                                                        "max_hold_buckets": max_hold,
                                                                        "cooldown_after_stop_buckets": cooldown,
                                                                        "disable_buys_when_price_below_anchor_bps": anchor_distance,
                                                                        "disable_buys_when_ema_slope_below_bps": ema_slope,
                                                                        "force_flat_at_end": True,
                                                                        "range_break_buffer_bps": 20.0,
                                                                        "rebalance_interval_buckets": rebalance_interval,
                                                                        "rebalance_mode": rebalance_mode,
                                                                        "rebalance_anchor_drift_bps": 20.0,
                                                                        "rebalance_vol_change_fraction": 0.25,
                                                                        "model_gate_mode": "none",
                                                                        "shadow_dir": "",
                                                                    }
                                                                )
                                                                if len(configs) >= MAX_CONFIGS:
                                                                    return configs
    return configs


def apply_contract_globals(sim: Any, contract: dict[str, Any]) -> None:
    sim.COOLDOWN_AFTER_STOP_BUCKETS = int(contract["cooldown_after_stop_buckets"])
    sim.DISABLE_BUYS_WHEN_PRICE_BELOW_ANCHOR_BPS = float(contract["disable_buys_when_price_below_anchor_bps"])
    sim.DISABLE_BUYS_WHEN_EMA_SLOPE_BELOW_BPS = float(contract["disable_buys_when_ema_slope_below_bps"])
    sim.FORCE_FLAT_AT_END = True
    sim.MODEL_GATE_MODE = "none"
    sim.SHADOW_DIR_ENV = ""


def slice_result_row(config_id: int, slice_index: int, source_slice: pd.DataFrame, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "slice_index": slice_index,
        "start_time": timestamp_iso(source_slice["timestamp"].iloc[0]),
        "end_time": timestamp_iso(source_slice["timestamp"].iloc[-1]),
        "rows": len(source_slice),
        "force_flat_total_net_bps": result.get("force_flat_total_net_bps"),
        "max_drawdown_bps": result.get("max_drawdown_bps"),
        "number_of_trades": result.get("number_of_trades"),
        "take_profit_exit_count": result.get("take_profit_exit_count"),
        "stop_loss_exit_count": result.get("stop_loss_exit_count"),
        "timeout_exit_count": result.get("timeout_exit_count"),
        "final_liquidation_count": result.get("final_liquidation_count"),
        "stop_loss_net_bps": result.get("stop_loss_net_bps"),
        "timeout_net_bps": result.get("timeout_net_bps"),
        "exposure_time_fraction": result.get("exposure_time_fraction"),
        "max_inventory": result.get("max_inventory"),
        "ending_inventory": result.get("ending_inventory"),
        "total_net_cost_0_25_bps": result.get("total_net_cost_0_25_bps"),
        "volatility_used_fraction": result.get("volatility_used_fraction"),
        "floor_used_fraction": result.get("floor_used_fraction"),
        "avg_realized_vol_bps": result.get("avg_realized_vol_bps"),
        "avg_spacing_to_vol_ratio": result.get("avg_spacing_to_vol_ratio"),
        "avg_stop_to_vol_ratio": result.get("avg_stop_to_vol_ratio"),
        "rebalance_count": result.get("rebalance_count"),
        "avg_rebalance_interval_buckets": result.get("avg_rebalance_interval_buckets"),
        "anchor_drift_rebalance_count": result.get("anchor_drift_rebalance_count"),
        "volatility_rebalance_count": result.get("volatility_rebalance_count"),
        "fixed_interval_rebalance_count": result.get("fixed_interval_rebalance_count"),
        "trades_per_rebalance": result.get("trades_per_rebalance"),
        "stale_ladder_entry_count": result.get("stale_ladder_entry_count"),
        "rebalance_blocked_count": result.get("rebalance_blocked_count"),
    }


def classify_and_score(row: dict[str, Any]) -> tuple[str, float]:
    total = safe_float(row.get("total_net_bps"), 0.0)
    pos_frac = safe_float(row.get("positive_slice_fraction"), 0.0)
    cost_frac = safe_float(row.get("cost_0_25_positive_slice_fraction"), 0.0)
    worst_slice = safe_float(row.get("worst_slice_net_bps"), -math.inf)
    worst_dd = safe_float(row.get("worst_drawdown_bps"), -math.inf)
    stop_ratio = safe_float(row.get("stop_loss_churn_ratio"), 1.0)
    exposure = safe_float(row.get("avg_exposure_time_fraction"), 0.0)
    all_flat = bool(row.get("all_slices_force_flat"))
    if (
        total > 0
        and pos_frac >= 0.75
        and cost_frac >= 0.5
        and worst_slice > -150
        and stop_ratio < 0.4
        and all_flat
    ):
        status = "stable_candidate"
    elif total > 0:
        status = "fragile_candidate"
    else:
        status = "reject"
    score = (
        total
        + pos_frac * 500.0
        + cost_frac * 250.0
        + worst_dd * 0.25
        - stop_ratio * 200.0
        - exposure * 100.0
    )
    return status, score


def aggregate_config(config_id: int, contract: dict[str, Any], slice_rows: list[dict[str, Any]]) -> dict[str, Any]:
    nets = np.asarray([safe_float(row.get("force_flat_total_net_bps")) for row in slice_rows], dtype=np.float64)
    drawdowns = np.asarray([safe_float(row.get("max_drawdown_bps")) for row in slice_rows], dtype=np.float64)
    cost025 = np.asarray([safe_float(row.get("total_net_cost_0_25_bps")) for row in slice_rows], dtype=np.float64)
    trades = np.asarray([safe_float(row.get("number_of_trades"), 0.0) for row in slice_rows], dtype=np.float64)
    tp = np.asarray([safe_float(row.get("take_profit_exit_count"), 0.0) for row in slice_rows], dtype=np.float64)
    sl = np.asarray([safe_float(row.get("stop_loss_exit_count"), 0.0) for row in slice_rows], dtype=np.float64)
    timeout = np.asarray([safe_float(row.get("timeout_exit_count"), 0.0) for row in slice_rows], dtype=np.float64)
    exposure = np.asarray([safe_float(row.get("exposure_time_fraction"), 0.0) for row in slice_rows], dtype=np.float64)
    ending_inventory = np.asarray([safe_float(row.get("ending_inventory"), 0.0) for row in slice_rows], dtype=np.float64)
    vol_used = np.asarray([safe_float(row.get("volatility_used_fraction")) for row in slice_rows], dtype=np.float64)
    floor_used = np.asarray([safe_float(row.get("floor_used_fraction")) for row in slice_rows], dtype=np.float64)
    avg_vol = np.asarray([safe_float(row.get("avg_realized_vol_bps")) for row in slice_rows], dtype=np.float64)
    spacing_to_vol = np.asarray([safe_float(row.get("avg_spacing_to_vol_ratio")) for row in slice_rows], dtype=np.float64)
    stop_to_vol = np.asarray([safe_float(row.get("avg_stop_to_vol_ratio")) for row in slice_rows], dtype=np.float64)
    rebalances = np.asarray([safe_float(row.get("rebalance_count"), 0.0) for row in slice_rows], dtype=np.float64)
    rebalance_intervals = np.asarray([safe_float(row.get("avg_rebalance_interval_buckets")) for row in slice_rows], dtype=np.float64)
    anchor_rebalances = np.asarray([safe_float(row.get("anchor_drift_rebalance_count"), 0.0) for row in slice_rows], dtype=np.float64)
    vol_rebalances = np.asarray([safe_float(row.get("volatility_rebalance_count"), 0.0) for row in slice_rows], dtype=np.float64)
    fixed_rebalances = np.asarray([safe_float(row.get("fixed_interval_rebalance_count"), 0.0) for row in slice_rows], dtype=np.float64)
    stale_entries = np.asarray([safe_float(row.get("stale_ladder_entry_count"), 0.0) for row in slice_rows], dtype=np.float64)
    rebalance_blocked = np.asarray([safe_float(row.get("rebalance_blocked_count"), 0.0) for row in slice_rows], dtype=np.float64)
    total_exits = float(np.nansum(tp + sl + timeout))
    out = {
        "config_id": config_id,
        **contract,
        "slices": len(slice_rows),
        "total_net_bps": float(np.nansum(nets)),
        "positive_slice_fraction": float(np.nanmean(nets > 0)),
        "median_slice_net_bps": float(np.nanmedian(nets)),
        "worst_slice_net_bps": float(np.nanmin(nets)),
        "average_slice_drawdown_bps": float(np.nanmean(drawdowns)),
        "worst_drawdown_bps": float(np.nanmin(drawdowns)),
        "total_trades": int(np.nansum(trades)),
        "total_take_profit_exits": int(np.nansum(tp)),
        "total_stop_loss_exits": int(np.nansum(sl)),
        "total_timeout_exits": int(np.nansum(timeout)),
        "stop_loss_churn_ratio": float(np.nansum(sl) / max(total_exits, 1.0)),
        "timeout_churn_ratio": float(np.nansum(timeout) / max(total_exits, 1.0)),
        "cost_0_25_positive_slice_fraction": float(np.nanmean(cost025 > 0)),
        "all_slices_force_flat": bool(np.nanmax(ending_inventory) == 0),
        "avg_exposure_time_fraction": float(np.nanmean(exposure)),
        "volatility_used_fraction": float(np.nanmean(vol_used)),
        "floor_used_fraction": float(np.nanmean(floor_used)),
        "avg_realized_vol_bps": float(np.nanmean(avg_vol)),
        "avg_spacing_to_vol_ratio": float(np.nanmean(spacing_to_vol)),
        "avg_stop_to_vol_ratio": float(np.nanmean(stop_to_vol)),
        "rebalance_count": int(np.nansum(rebalances)),
        "avg_rebalance_interval_buckets": float(np.nanmean(rebalance_intervals)),
        "anchor_drift_rebalance_count": int(np.nansum(anchor_rebalances)),
        "volatility_rebalance_count": int(np.nansum(vol_rebalances)),
        "fixed_interval_rebalance_count": int(np.nansum(fixed_rebalances)),
        "trades_per_rebalance": float(np.nansum(trades) / max(1.0, np.nansum(rebalances))),
        "stale_ladder_entry_count": int(np.nansum(stale_entries)),
        "rebalance_blocked_count": int(np.nansum(rebalance_blocked)),
    }
    out["classification"], out["rank_score"] = classify_and_score(out)
    return out


def status_priority(status: str) -> int:
    return {"stable_candidate": 0, "fragile_candidate": 1, "reject": 2}.get(status, 3)


def write_text(path: Path, rows: list[dict[str, Any]], configs: list[dict[str, Any]]) -> None:
    counts = pd.Series([row["classification"] for row in rows]).value_counts().to_dict()
    lines = [
        "Ladder Risk Walk-Forward Sweep",
        "",
        f"Created at: {stamp()}",
        f"Source: {SOURCE_PATH}",
        f"Configs evaluated: {len(configs)}",
        f"Smoke mode: {SMOKE_MODE}",
        f"Slice rows: {SLICE_ROWS}",
        f"Step rows: {STEP_ROWS}",
        f"Max slices: {MAX_SLICES}",
        f"Max rows per slice: {MAX_ROWS_PER_SLICE if MAX_ROWS_PER_SLICE else 'none'}",
        "",
        "Classification counts:",
    ]
    for status in ["stable_candidate", "fragile_candidate", "reject"]:
        lines.append(f"  {status}: {counts.get(status, 0)}")
    lines.extend(["", "Top 20:"])
    for row in rows[:20]:
        lines.append(
            "  "
            f"{row['classification']} score={row['rank_score']:.3f} total={row['total_net_bps']:.3f} "
            f"pos={row['positive_slice_fraction']:.3f} cost025={row['cost_0_25_positive_slice_fraction']:.3f} "
            f"worst_slice={row['worst_slice_net_bps']:.3f} worst_dd={row['worst_drawdown_bps']:.3f} "
            f"sl_ratio={row['stop_loss_churn_ratio']:.3f} exposure={row['avg_exposure_time_fraction']:.3f} "
            f"vol_used={safe_float(row.get('volatility_used_fraction')):.3f} floor_used={safe_float(row.get('floor_used_fraction')):.3f} "
            f"rebalances={row.get('rebalance_count')} trades_per_rebalance={safe_float(row.get('trades_per_rebalance')):.3f} "
            f"avg_vol={safe_float(row.get('avg_realized_vol_bps')):.3f} "
            f"spacing_mult={row.get('spacing_vol_mult', row.get('vol_mult'))} floor_mode={row.get('min_spacing_floor_mode')} "
            f"min_spacing={row['min_spacing_bps']} max_units={row['max_open_units']} "
            f"stop_mult={row.get('stop_loss_vol_mult')} stop_floor={row.get('min_stop_floor_bps', row.get('stop_loss_bps'))} "
            f"hold={row['max_hold_buckets']} tp_spacing={row.get('take_profit_spacing_mult', row.get('take_profit_mult'))} cooldown={row['cooldown_after_stop_buckets']} "
            f"rebalance={row.get('rebalance_mode')}@{row.get('rebalance_interval_buckets')} "
            f"anchor_dist={row['disable_buys_when_price_below_anchor_bps']} slope={row['disable_buys_when_ema_slope_below_bps']}"
        )
    lines.extend(
        [
            "",
            "Safety:",
            "  paper_only=true",
            "  public_recorded_data_only=true",
            "  private_api=false",
            "  orders=false",
            "  training=false",
            "  promotion=false",
            "  champion_mutation=false",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_progress(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame().to_csv(path, index=False)


def main() -> int:
    sim = load_simulator_module()
    source = sim.load_price_data(SOURCE_PATH)
    out_dir = output_dir()
    progress_path = out_dir / "sweep_progress.csv"
    slices: list[tuple[int, pd.DataFrame, dict[str, np.ndarray]]] = []
    for slice_index, start in enumerate(range(0, len(source) - SLICE_ROWS + 1, STEP_ROWS)):
        if slice_index >= MAX_SLICES:
            break
        source_slice = source.iloc[start : start + SLICE_ROWS].copy().reset_index(drop=True)
        if len(source_slice) < SLICE_ROWS:
            continue
        if MAX_ROWS_PER_SLICE > 0 and len(source_slice) > MAX_ROWS_PER_SLICE:
            source_slice = source_slice.iloc[:MAX_ROWS_PER_SLICE].copy().reset_index(drop=True)
        slices.append((slice_index, source_slice, sim.precompute_indicators(source_slice)))
    if not slices:
        raise SystemExit("No walk-forward slices produced.")

    configs = iter_configs()
    aggregate_rows = []
    per_slice_rows = []
    for config_id, contract in enumerate(configs):
        if PROGRESS_EVERY > 0 and (config_id == 0 or config_id % PROGRESS_EVERY == 0):
            print(f"Evaluating config {config_id + 1}/{len(configs)}", flush=True)
            write_progress(progress_path, aggregate_rows)
        apply_contract_globals(sim, contract)
        config_slice_rows = []
        for slice_index, source_slice, indicators in slices:
            result, _, _ = sim.simulate_config(source_slice, indicators, None, contract, save_paths=False)
            row = slice_result_row(config_id, slice_index, source_slice, result)
            config_slice_rows.append(row)
            per_slice_rows.append(row)
        aggregate_rows.append(aggregate_config(config_id, contract, config_slice_rows))
        if PROGRESS_EVERY > 0 and (config_id + 1) % PROGRESS_EVERY == 0:
            write_progress(progress_path, aggregate_rows)

    aggregate_rows = sorted(
        aggregate_rows,
        key=lambda row: (
            status_priority(row["classification"]),
            -safe_float(row.get("rank_score"), -math.inf),
            -safe_float(row.get("total_net_bps"), -math.inf),
            -safe_float(row.get("positive_slice_fraction"), -math.inf),
            -safe_float(row.get("cost_0_25_positive_slice_fraction"), -math.inf),
            safe_float(row.get("worst_drawdown_bps"), -math.inf),
            safe_float(row.get("stop_loss_churn_ratio"), math.inf),
            safe_float(row.get("avg_exposure_time_fraction"), math.inf),
        ),
    )

    sweep_path = out_dir / "ladder_risk_walkforward_sweep.csv"
    text_path = out_dir / "ladder_risk_walkforward_sweep.txt"
    stable_path = out_dir / "stable_candidates.csv"
    fragile_path = out_dir / "fragile_candidates.csv"
    rejects_path = out_dir / "rejects.csv"
    best_path = out_dir / "best_ladder_walkforward_contract.json"
    per_slice_path = out_dir / "per_slice_results.csv"
    frame = pd.DataFrame(aggregate_rows)
    write_progress(progress_path, aggregate_rows)
    frame.to_csv(sweep_path, index=False)
    frame[frame["classification"].eq("stable_candidate")].to_csv(stable_path, index=False)
    frame[frame["classification"].eq("fragile_candidate")].to_csv(fragile_path, index=False)
    frame[frame["classification"].eq("reject")].to_csv(rejects_path, index=False)
    pd.DataFrame(per_slice_rows).to_csv(per_slice_path, index=False)
    best = aggregate_rows[0] if aggregate_rows else {}
    best_path.write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")
    write_text(text_path, aggregate_rows, configs)

    print("Ladder risk walk-forward sweep complete")
    print(f"Configs: {len(configs)}")
    print(f"Slices: {len(slices)}")
    print(f"CSV: {sweep_path}")
    print(f"TXT: {text_path}")
    print(f"Best contract: {best_path}")
    print("Safety: paper only. Public recorded data only. No private API. No orders. No training. No promotion. No champion mutation.")
    print(frame.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
