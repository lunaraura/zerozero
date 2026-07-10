#!/usr/bin/env python3
"""Sweep risk-controlled ladder configs across recorded walk-forward slices."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import multiprocessing
import uuid
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
STAGE_MODE = os.getenv("LADDER_SWEEP_STAGE_MODE", "off").strip().lower()
STAGE_ROWS = int(float(os.getenv("LADDER_STAGE_ROWS", "20000")))
STAGE_KEEP_FRACTION = float(os.getenv("LADDER_STAGE_KEEP_FRACTION", "0.25"))
STAGE_MIN_KEEP = int(float(os.getenv("LADDER_STAGE_MIN_KEEP", "25")))
DISABLE_DEDUPE = env_bool("LADDER_SWEEP_DISABLE_DEDUPE", False)
WORKERS_TEXT = os.getenv("LADDER_SWEEP_WORKERS", "1").strip().lower()
SAVE_DETAIL_TOP_N = int(float(os.getenv("LADDER_SAVE_DETAIL_TOP_N", "5")))
RESUME_FROM_OUTPUT_DIR = os.getenv("LADDER_SWEEP_RESUME_FROM_OUTPUT_DIR", "").strip()
SIM_PATH = PROJECT_ROOT / "scripts" / "tiny" / "simulate_path_aware_ladder_baseline.py"

ANCHOR_MODES = ["ema", "rolling_ma"]
ANCHOR_WINDOWS = [300, 900, 1800, 3600]
ENTRY_ANCHOR_SOURCES = ["frozen_rebalance_anchor"]
RECENTER_MODES = ["fixed_interval", "never_while_inventory_open", "interval_only_when_flat"]
RUNG_CONSUMPTION_MODES = ["consume_until_rebalance", "consume_until_flat", "reusable_after_cooldown"]
RUNG_REUSE_COOLDOWN_BUCKETS = [180]
EMERGENCY_REBALANCE_MODES = ["off", "depleted_side_only", "depleted_side_if_anchor_accelerating"]
EMERGENCY_REMAINING_RUNG_THRESHOLDS = [1, 2]
EMERGENCY_MIN_INTERVAL_BUCKETS = [60, 180]
EMERGENCY_MAX_PER_GLOBAL_INTERVAL = [1]
VOL_WINDOWS = [60, 150, 300]
MIN_SPACING_BPS = [10.0, 20.0, 40.0]
MIN_SPACING_FLOOR_MODES = ["fixed_bps", "spread_multiple", "tick_multiple", "max_fixed_or_spread"]
SPACING_VOL_MULTS = [1.0, 2.0, 4.0]
MAX_OPEN_UNITS = [1, 2]
STOP_LOSS_BPS = [60.0, 80.0, 120.0]
STOP_LOSS_VOL_MULTS = [4.0, 6.0, 8.0]
MAX_HOLD_BUCKETS = [180, 360, 720, 1440]
TAKE_PROFIT_SPACING_MULTS = [0.25, 0.5, 0.75, 1.0]
MIN_TAKE_PROFIT_BPS = [5.0, 10.0, 15.0, 20.0]
TIMEOUT_EXIT_MODES = ["market", "breakeven_only", "trend_invalid_only", "disable_timeout"]
COOLDOWNS = [30, 60, 180]
ANCHOR_DISTANCE_LIMITS = [60.0, 120.0, 240.0]
EMA_SLOPE_LIMITS = [-2.0, 0.0, 2.0]
REBALANCE_INTERVAL_BUCKETS = [90, 180, 360]
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


def nanmean_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else math.nan


def load_simulator_module():
    spec = importlib.util.spec_from_file_location("ladder_sim", SIM_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load simulator module: {SIM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_dir() -> Path:
    if RESUME_FROM_OUTPUT_DIR:
        path = Path(RESUME_FROM_OUTPUT_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path
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
    axes = [
        ANCHOR_MODES,
        ANCHOR_WINDOWS,
        ENTRY_ANCHOR_SOURCES,
        RECENTER_MODES,
        RUNG_CONSUMPTION_MODES,
        RUNG_REUSE_COOLDOWN_BUCKETS,
        EMERGENCY_REBALANCE_MODES,
        EMERGENCY_REMAINING_RUNG_THRESHOLDS,
        EMERGENCY_MIN_INTERVAL_BUCKETS,
        EMERGENCY_MAX_PER_GLOBAL_INTERVAL,
        VOL_WINDOWS,
        SPACING_VOL_MULTS,
        MIN_SPACING_BPS,
        MIN_SPACING_FLOOR_MODES,
        MAX_OPEN_UNITS,
        STOP_LOSS_BPS,
        STOP_LOSS_VOL_MULTS,
        MAX_HOLD_BUCKETS,
        TAKE_PROFIT_SPACING_MULTS,
        MIN_TAKE_PROFIT_BPS,
        TIMEOUT_EXIT_MODES,
        COOLDOWNS,
        ANCHOR_DISTANCE_LIMITS,
        EMA_SLOPE_LIMITS,
        REBALANCE_INTERVAL_BUCKETS,
        REBALANCE_MODES,
    ]
    total = math.prod(len(axis) for axis in axes)
    if SMOKE_MODE:
        sample_count = min(MAX_CONFIGS, total)
        indexes = np.linspace(0, total - 1, sample_count, dtype=np.int64).tolist()
    else:
        indexes = list(range(min(MAX_CONFIGS, total)))

    configs = []
    for index in indexes:
        values = []
        remaining = int(index)
        for axis in reversed(axes):
            values.append(axis[remaining % len(axis)])
            remaining //= len(axis)
        (
            anchor_mode,
            anchor_window,
            entry_anchor_source,
            recenter_mode,
            rung_consumption_mode,
            rung_reuse_cooldown_buckets,
            emergency_rebalance_mode,
            emergency_remaining_rung_threshold,
            emergency_min_interval_buckets,
            emergency_max_per_global_interval,
            vol_window,
            spacing_vol_mult,
            min_spacing,
            min_spacing_floor_mode,
            max_units,
            stop_loss,
            stop_loss_vol_mult,
            max_hold,
            take_profit_spacing_mult,
            min_take_profit_bps,
            timeout_exit_mode,
            cooldown,
            anchor_distance,
            ema_slope,
            rebalance_interval,
            rebalance_mode,
        ) = reversed(values)
        configs.append(
            {
                "anchor_mode": anchor_mode,
                "entry_anchor_source": entry_anchor_source,
                "recenter_mode": recenter_mode,
                "rung_consumption_mode": rung_consumption_mode,
                "rung_reuse_cooldown_buckets": rung_reuse_cooldown_buckets,
                "emergency_rebalance_mode": emergency_rebalance_mode,
                "emergency_remaining_rung_threshold": emergency_remaining_rung_threshold,
                "emergency_min_interval_buckets": emergency_min_interval_buckets,
                "emergency_max_per_global_interval": emergency_max_per_global_interval,
                "emergency_anchor_slope_bps": 0.0,
                "emergency_anchor_accel_bps": 0.0,
                "emergency_disable_when_inventory_ge": max_units,
                "emergency_disable_when_drawdown_bps": 120.0,
                "anchor_window": anchor_window,
                "vol_window": vol_window,
                "spacing_mode": "volatility_scaled",
                "vol_mult": spacing_vol_mult,
                "spacing_vol_mult": spacing_vol_mult,
                "rung_count": 3,
                "take_profit_mult": take_profit_spacing_mult,
                "take_profit_spacing_mult": take_profit_spacing_mult,
                "min_take_profit_bps": min_take_profit_bps,
                "min_spacing_bps": min_spacing,
                "min_spacing_floor_mode": min_spacing_floor_mode,
                "cost_bps": 0.1,
                "max_open_units": max_units,
                "stop_loss_bps": stop_loss,
                "min_stop_floor_bps": stop_loss,
                "stop_loss_vol_mult": stop_loss_vol_mult,
                "max_hold_buckets": max_hold,
                "timeout_exit_mode": timeout_exit_mode,
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
    return configs


def apply_contract_globals(sim: Any, contract: dict[str, Any]) -> None:
    sim.COOLDOWN_AFTER_STOP_BUCKETS = int(contract["cooldown_after_stop_buckets"])
    sim.DISABLE_BUYS_WHEN_PRICE_BELOW_ANCHOR_BPS = float(contract["disable_buys_when_price_below_anchor_bps"])
    sim.DISABLE_BUYS_WHEN_EMA_SLOPE_BELOW_BPS = float(contract["disable_buys_when_ema_slope_below_bps"])
    sim.FORCE_FLAT_AT_END = True
    sim.MODEL_GATE_MODE = "none"
    sim.SHADOW_DIR_ENV = ""


def worker_count() -> int:
    if WORKERS_TEXT == "auto":
        return max(1, min(4, multiprocessing.cpu_count() or 1))
    try:
        return max(1, int(float(WORKERS_TEXT)))
    except ValueError:
        return 1


def behavior_signature(row: dict[str, Any]) -> str:
    fields = [
        "classification",
        "total_net_bps",
        "total_trades",
        "total_take_profit_exits",
        "total_stop_loss_exits",
        "total_timeout_exits",
        "worst_drawdown_bps",
        "timeout_churn_ratio",
        "stop_loss_churn_ratio",
        "emergency_rebalance_count",
        "rungs_refreshed_by_emergency_count",
    ]
    parts = []
    for field in fields:
        value = row.get(field, "")
        if isinstance(value, (int, str)):
            parts.append(f"{field}={value}")
        else:
            parts.append(f"{field}={safe_float(value, 0.0):.6f}")
    return "|".join(parts)


def load_completed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(int(line))
        except ValueError:
            continue
    return ids


def append_completed_id(path: Path, config_id: int) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{config_id}\n")


def simulate_one_config(args: tuple[int, dict[str, Any], list[tuple[int, pd.DataFrame, dict[str, np.ndarray]]]]) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    config_id, contract, slices = args
    sim = load_simulator_module()
    apply_contract_globals(sim, contract)
    config_slice_rows = []
    for slice_index, source_slice, indicators in slices:
        result, _, _ = sim.simulate_config(source_slice, indicators, None, contract, save_paths=False)
        config_slice_rows.append(slice_result_row(config_id, slice_index, source_slice, result))
    return config_id, config_slice_rows, aggregate_config(config_id, contract, config_slice_rows)


def evaluate_configs(
    sim: Any,
    configs: list[dict[str, Any]],
    slices: list[tuple[int, pd.DataFrame, dict[str, np.ndarray]]],
    out_dir: Path,
    progress_path: Path,
    completed_path: Path,
    completed_ids: set[int] | None = None,
    progress_prefix: str = "Evaluating",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed_ids = completed_ids or set()
    aggregate_rows: list[dict[str, Any]] = []
    per_slice_rows: list[dict[str, Any]] = []
    workers = worker_count()
    indexed = [(config_id, contract) for config_id, contract in enumerate(configs) if config_id not in completed_ids]
    if workers > 1 and indexed:
        tasks = [(config_id, contract, slices) for config_id, contract in indexed]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(simulate_one_config, task): task[0] for task in tasks}
            done = 0
            for future in as_completed(futures):
                config_id, config_slice_rows, aggregate = future.result()
                per_slice_rows.extend(config_slice_rows)
                aggregate_rows.append(aggregate)
                append_completed_id(completed_path, config_id)
                done += 1
                if PROGRESS_EVERY > 0 and (done == 1 or done % PROGRESS_EVERY == 0):
                    print(f"{progress_prefix} config {done}/{len(indexed)} with workers={workers}", flush=True)
                    write_progress(progress_path, aggregate_rows)
    else:
        for done, (config_id, contract) in enumerate(indexed, start=1):
            if PROGRESS_EVERY > 0 and (done == 1 or (done - 1) % PROGRESS_EVERY == 0):
                print(f"{progress_prefix} config {done}/{len(indexed)}", flush=True)
                write_progress(progress_path, aggregate_rows)
            apply_contract_globals(sim, contract)
            config_slice_rows = []
            for slice_index, source_slice, indicators in slices:
                result, _, _ = sim.simulate_config(source_slice, indicators, None, contract, save_paths=False)
                row = slice_result_row(config_id, slice_index, source_slice, result)
                config_slice_rows.append(row)
                per_slice_rows.append(row)
            aggregate_rows.append(aggregate_config(config_id, contract, config_slice_rows))
            append_completed_id(completed_path, config_id)
            if PROGRESS_EVERY > 0 and done % PROGRESS_EVERY == 0:
                write_progress(progress_path, aggregate_rows)
    write_progress(progress_path, aggregate_rows)
    return aggregate_rows, per_slice_rows


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
        "average_hold_buckets": result.get("average_hold_buckets"),
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
        "anchor_window_seconds": result.get("anchor_window_seconds"),
        "anchor_value_at_rebalance": result.get("anchor_value_at_rebalance"),
        "anchor_drift_since_last_rebalance_bps": result.get("anchor_drift_since_last_rebalance_bps"),
        "avg_anchor_drift_since_rebalance_bps": result.get("avg_anchor_drift_since_rebalance_bps"),
        "recenter_count": result.get("recenter_count"),
        "recenter_while_inventory_count": result.get("recenter_while_inventory_count"),
        "recenter_blocked_by_inventory_count": result.get("recenter_blocked_by_inventory_count"),
        "avg_anchor_distance_entry_bps": result.get("avg_anchor_distance_entry_bps"),
        "avg_anchor_distance_exit_bps": result.get("avg_anchor_distance_exit_bps"),
        "rung_consumed_count": result.get("rung_consumed_count"),
        "rung_reused_count": result.get("rung_reused_count"),
        "duplicate_rung_trigger_block_count": result.get("duplicate_rung_trigger_block_count"),
        "consumed_rung_block_count": result.get("consumed_rung_block_count"),
        "avg_rung_lifetime_buckets": result.get("avg_rung_lifetime_buckets"),
        "per_rung_consumed_count": result.get("per_rung_consumed_count"),
        "emergency_rebalance_count": result.get("emergency_rebalance_count"),
        "emergency_rebalance_buy_side_count": result.get("emergency_rebalance_buy_side_count"),
        "emergency_rebalance_sell_side_count": result.get("emergency_rebalance_sell_side_count"),
        "emergency_rebalance_blocked_cooldown_count": result.get("emergency_rebalance_blocked_cooldown_count"),
        "emergency_rebalance_blocked_inventory_count": result.get("emergency_rebalance_blocked_inventory_count"),
        "emergency_rebalance_blocked_drawdown_count": result.get("emergency_rebalance_blocked_drawdown_count"),
        "emergency_rebalance_blocked_slope_count": result.get("emergency_rebalance_blocked_slope_count"),
        "emergency_rebalance_blocked_interval_count": result.get("emergency_rebalance_blocked_interval_count"),
        "emergency_rebalance_blocked_max_per_global_count": result.get("emergency_rebalance_blocked_max_per_global_count"),
        "avg_remaining_rungs_before_emergency": result.get("avg_remaining_rungs_before_emergency"),
        "rungs_refreshed_by_emergency_count": result.get("rungs_refreshed_by_emergency_count"),
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
    timeout_ratio = safe_float(row.get("timeout_churn_ratio"), 1.0)
    timeout_net = safe_float(row.get("timeout_net_bps"), 0.0)
    exposure = safe_float(row.get("avg_exposure_time_fraction"), 0.0)
    all_flat = bool(row.get("all_slices_force_flat"))
    if timeout_ratio > 0.5 and total <= 0:
        status = "reject"
    elif (
        total > 0
        and pos_frac >= 0.75
        and cost_frac >= 0.5
        and worst_slice > -150
        and stop_ratio < 0.4
        and timeout_ratio < 0.35
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
        - timeout_ratio * 250.0
        + min(timeout_net, 0.0) * 0.25
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
    timeout_net = np.asarray([safe_float(row.get("timeout_net_bps"), 0.0) for row in slice_rows], dtype=np.float64)
    average_hold = np.asarray([safe_float(row.get("average_hold_buckets")) for row in slice_rows], dtype=np.float64)
    exposure = np.asarray([safe_float(row.get("exposure_time_fraction"), 0.0) for row in slice_rows], dtype=np.float64)
    ending_inventory = np.asarray([safe_float(row.get("ending_inventory"), 0.0) for row in slice_rows], dtype=np.float64)
    vol_used = np.asarray([safe_float(row.get("volatility_used_fraction")) for row in slice_rows], dtype=np.float64)
    floor_used = np.asarray([safe_float(row.get("floor_used_fraction")) for row in slice_rows], dtype=np.float64)
    avg_vol = np.asarray([safe_float(row.get("avg_realized_vol_bps")) for row in slice_rows], dtype=np.float64)
    spacing_to_vol = np.asarray([safe_float(row.get("avg_spacing_to_vol_ratio")) for row in slice_rows], dtype=np.float64)
    stop_to_vol = np.asarray([safe_float(row.get("avg_stop_to_vol_ratio")) for row in slice_rows], dtype=np.float64)
    rebalances = np.asarray([safe_float(row.get("rebalance_count"), 0.0) for row in slice_rows], dtype=np.float64)
    anchor_seconds = np.asarray([safe_float(row.get("anchor_window_seconds")) for row in slice_rows], dtype=np.float64)
    anchor_values = np.asarray([safe_float(row.get("anchor_value_at_rebalance")) for row in slice_rows], dtype=np.float64)
    anchor_drift = np.asarray([safe_float(row.get("anchor_drift_since_last_rebalance_bps")) for row in slice_rows], dtype=np.float64)
    avg_anchor_drift = np.asarray([safe_float(row.get("avg_anchor_drift_since_rebalance_bps")) for row in slice_rows], dtype=np.float64)
    recenter = np.asarray([safe_float(row.get("recenter_count"), 0.0) for row in slice_rows], dtype=np.float64)
    recenter_inventory = np.asarray([safe_float(row.get("recenter_while_inventory_count"), 0.0) for row in slice_rows], dtype=np.float64)
    recenter_blocked = np.asarray([safe_float(row.get("recenter_blocked_by_inventory_count"), 0.0) for row in slice_rows], dtype=np.float64)
    anchor_entry = np.asarray([safe_float(row.get("avg_anchor_distance_entry_bps")) for row in slice_rows], dtype=np.float64)
    anchor_exit = np.asarray([safe_float(row.get("avg_anchor_distance_exit_bps")) for row in slice_rows], dtype=np.float64)
    rung_consumed = np.asarray([safe_float(row.get("rung_consumed_count"), 0.0) for row in slice_rows], dtype=np.float64)
    rung_reused = np.asarray([safe_float(row.get("rung_reused_count"), 0.0) for row in slice_rows], dtype=np.float64)
    duplicate_rung_blocks = np.asarray([safe_float(row.get("duplicate_rung_trigger_block_count"), 0.0) for row in slice_rows], dtype=np.float64)
    consumed_rung_blocks = np.asarray([safe_float(row.get("consumed_rung_block_count"), 0.0) for row in slice_rows], dtype=np.float64)
    rung_lifetime = np.asarray([safe_float(row.get("avg_rung_lifetime_buckets")) for row in slice_rows], dtype=np.float64)
    emergency = np.asarray([safe_float(row.get("emergency_rebalance_count"), 0.0) for row in slice_rows], dtype=np.float64)
    emergency_buy = np.asarray([safe_float(row.get("emergency_rebalance_buy_side_count"), 0.0) for row in slice_rows], dtype=np.float64)
    emergency_sell = np.asarray([safe_float(row.get("emergency_rebalance_sell_side_count"), 0.0) for row in slice_rows], dtype=np.float64)
    emergency_cooldown = np.asarray([safe_float(row.get("emergency_rebalance_blocked_cooldown_count"), 0.0) for row in slice_rows], dtype=np.float64)
    emergency_inventory = np.asarray([safe_float(row.get("emergency_rebalance_blocked_inventory_count"), 0.0) for row in slice_rows], dtype=np.float64)
    emergency_drawdown = np.asarray([safe_float(row.get("emergency_rebalance_blocked_drawdown_count"), 0.0) for row in slice_rows], dtype=np.float64)
    emergency_slope = np.asarray([safe_float(row.get("emergency_rebalance_blocked_slope_count"), 0.0) for row in slice_rows], dtype=np.float64)
    emergency_interval = np.asarray([safe_float(row.get("emergency_rebalance_blocked_interval_count"), 0.0) for row in slice_rows], dtype=np.float64)
    emergency_max_global = np.asarray([safe_float(row.get("emergency_rebalance_blocked_max_per_global_count"), 0.0) for row in slice_rows], dtype=np.float64)
    emergency_remaining = np.asarray([safe_float(row.get("avg_remaining_rungs_before_emergency")) for row in slice_rows], dtype=np.float64)
    emergency_refreshed = np.asarray([safe_float(row.get("rungs_refreshed_by_emergency_count"), 0.0) for row in slice_rows], dtype=np.float64)
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
        "timeout_net_bps": float(np.nansum(timeout_net)),
        "average_hold_buckets": nanmean_or_nan(average_hold),
        "stop_loss_churn_ratio": float(np.nansum(sl) / max(total_exits, 1.0)),
        "timeout_churn_ratio": float(np.nansum(timeout) / max(total_exits, 1.0)),
        "cost_0_25_positive_slice_fraction": float(np.nanmean(cost025 > 0)),
        "all_slices_force_flat": bool(np.nanmax(ending_inventory) == 0),
        "avg_exposure_time_fraction": float(np.nanmean(exposure)),
        "volatility_used_fraction": nanmean_or_nan(vol_used),
        "floor_used_fraction": nanmean_or_nan(floor_used),
        "avg_realized_vol_bps": nanmean_or_nan(avg_vol),
        "avg_spacing_to_vol_ratio": nanmean_or_nan(spacing_to_vol),
        "avg_stop_to_vol_ratio": nanmean_or_nan(stop_to_vol),
        "rebalance_count": int(np.nansum(rebalances)),
        "anchor_window_seconds": nanmean_or_nan(anchor_seconds),
        "anchor_value_at_rebalance": nanmean_or_nan(anchor_values),
        "anchor_drift_since_last_rebalance_bps": nanmean_or_nan(anchor_drift),
        "avg_anchor_drift_since_rebalance_bps": nanmean_or_nan(avg_anchor_drift),
        "recenter_count": int(np.nansum(recenter)),
        "recenter_while_inventory_count": int(np.nansum(recenter_inventory)),
        "recenter_blocked_by_inventory_count": int(np.nansum(recenter_blocked)),
        "avg_anchor_distance_entry_bps": nanmean_or_nan(anchor_entry),
        "avg_anchor_distance_exit_bps": nanmean_or_nan(anchor_exit),
        "rung_consumed_count": int(np.nansum(rung_consumed)),
        "rung_reused_count": int(np.nansum(rung_reused)),
        "duplicate_rung_trigger_block_count": int(np.nansum(duplicate_rung_blocks)),
        "consumed_rung_block_count": int(np.nansum(consumed_rung_blocks)),
        "avg_rung_lifetime_buckets": nanmean_or_nan(rung_lifetime),
        "per_rung_consumed_count": ";".join(str(row.get("per_rung_consumed_count", "")) for row in slice_rows if row.get("per_rung_consumed_count")),
        "emergency_rebalance_count": int(np.nansum(emergency)),
        "emergency_rebalance_buy_side_count": int(np.nansum(emergency_buy)),
        "emergency_rebalance_sell_side_count": int(np.nansum(emergency_sell)),
        "emergency_rebalance_blocked_cooldown_count": int(np.nansum(emergency_cooldown)),
        "emergency_rebalance_blocked_inventory_count": int(np.nansum(emergency_inventory)),
        "emergency_rebalance_blocked_drawdown_count": int(np.nansum(emergency_drawdown)),
        "emergency_rebalance_blocked_slope_count": int(np.nansum(emergency_slope)),
        "emergency_rebalance_blocked_interval_count": int(np.nansum(emergency_interval)),
        "emergency_rebalance_blocked_max_per_global_count": int(np.nansum(emergency_max_global)),
        "avg_remaining_rungs_before_emergency": nanmean_or_nan(emergency_remaining),
        "rungs_refreshed_by_emergency_count": int(np.nansum(emergency_refreshed)),
        "avg_rebalance_interval_buckets": nanmean_or_nan(rebalance_intervals),
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
        f"Stage mode: {STAGE_MODE}",
        f"Stage rows: {STAGE_ROWS}",
        f"Stage keep fraction: {STAGE_KEEP_FRACTION}",
        f"Stage min keep: {STAGE_MIN_KEEP}",
        f"Dedupe disabled: {DISABLE_DEDUPE}",
        f"Workers: {worker_count()}",
        f"Save detail top N: {SAVE_DETAIL_TOP_N}",
        f"Resume from output dir: {RESUME_FROM_OUTPUT_DIR or 'none'}",
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
            f"timeout_ratio={row['timeout_churn_ratio']:.3f} timeout_net={safe_float(row.get('timeout_net_bps')):.3f} avg_hold={safe_float(row.get('average_hold_buckets')):.3f} "
            f"tp_exits={row.get('total_take_profit_exits')} timeout_exits={row.get('total_timeout_exits')} timeout_mode={row.get('timeout_exit_mode')} "
            f"anchor_seconds={safe_float(row.get('anchor_window_seconds')):.0f} recenter={row.get('recenter_count')} "
            f"recenter_blocked={row.get('recenter_blocked_by_inventory_count')} anchor_drift={safe_float(row.get('anchor_drift_since_last_rebalance_bps')):.3f} "
            f"rung_consumed={row.get('rung_consumed_count')} rung_blocks={row.get('duplicate_rung_trigger_block_count')} "
            f"emergency={row.get('emergency_rebalance_count')} emergency_rungs={row.get('rungs_refreshed_by_emergency_count')} "
            f"vol_used={safe_float(row.get('volatility_used_fraction')):.3f} floor_used={safe_float(row.get('floor_used_fraction')):.3f} "
            f"rebalances={row.get('rebalance_count')} trades_per_rebalance={safe_float(row.get('trades_per_rebalance')):.3f} "
            f"avg_vol={safe_float(row.get('avg_realized_vol_bps')):.3f} "
            f"spacing_mult={row.get('spacing_vol_mult', row.get('vol_mult'))} floor_mode={row.get('min_spacing_floor_mode')} "
            f"min_spacing={row['min_spacing_bps']} max_units={row['max_open_units']} "
            f"stop_mult={row.get('stop_loss_vol_mult')} stop_floor={row.get('min_stop_floor_bps', row.get('stop_loss_bps'))} "
            f"hold={row['max_hold_buckets']} tp_spacing={row.get('take_profit_spacing_mult', row.get('take_profit_mult'))} min_tp={row.get('min_take_profit_bps')} cooldown={row['cooldown_after_stop_buckets']} "
            f"anchor={row.get('anchor_mode')}:{row.get('anchor_window')} entry_anchor={row.get('entry_anchor_source')} recenter_mode={row.get('recenter_mode')} "
            f"rung_mode={row.get('rung_consumption_mode')} "
            f"emergency_mode={row.get('emergency_rebalance_mode')} "
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


def save_top_details(
    sim: Any,
    out_dir: Path,
    rows: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    slices: list[tuple[int, pd.DataFrame, dict[str, np.ndarray]]],
) -> None:
    if SAVE_DETAIL_TOP_N <= 0 or not rows:
        return
    detail_root = out_dir / "top_config_details"
    detail_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for rank, row in enumerate(rows[:SAVE_DETAIL_TOP_N], start=1):
        config_id = int(row.get("config_id", -1))
        if config_id < 0 or config_id >= len(configs):
            continue
        contract = configs[config_id]
        apply_contract_globals(sim, contract)
        config_dir = detail_root / f"rank_{rank:02d}_config_{config_id}"
        config_dir.mkdir(parents=True, exist_ok=True)
        for slice_index, source_slice, indicators in slices:
            result, trades, equity = sim.simulate_config(source_slice, indicators, None, contract, save_paths=True)
            trades_path = config_dir / f"slice_{slice_index}_trades.csv"
            equity_path = config_dir / f"slice_{slice_index}_equity_curve.csv"
            summary_path = config_dir / f"slice_{slice_index}_summary.json"
            trades.to_csv(trades_path, index=False)
            equity.to_csv(equity_path, index=False)
            summary_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
            manifest_rows.append(
                {
                    "rank": rank,
                    "config_id": config_id,
                    "slice_index": slice_index,
                    "trades_path": str(trades_path),
                    "equity_path": str(equity_path),
                    "summary_path": str(summary_path),
                    "paper_only": True,
                    "orders": False,
                    "training": False,
                    "promotion": False,
                    "champion_mutation": False,
                }
            )
    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(detail_root / "top_config_detail_manifest.csv", index=False)


def main() -> int:
    started = time.perf_counter()
    sim = load_simulator_module()
    sim.ANCHOR_WINDOWS = ANCHOR_WINDOWS
    sim.VOL_WINDOWS = VOL_WINDOWS
    source = sim.load_price_data(SOURCE_PATH)
    out_dir = output_dir()
    progress_path = out_dir / "sweep_progress.csv"
    completed_path = out_dir / "completed_config_ids.txt"
    stage_progress_path = out_dir / "stage1_sweep_progress.csv"
    stage_completed_path = out_dir / "stage1_completed_config_ids.txt"
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
    selected_configs = configs
    selected_original_ids = list(range(len(configs)))
    stage_rows: list[dict[str, Any]] = []
    if STAGE_MODE not in {"", "off", "false", "0"}:
        stage_slices: list[tuple[int, pd.DataFrame, dict[str, np.ndarray]]] = []
        for slice_index, source_slice, _ in slices:
            stage_slice = source_slice.iloc[: min(STAGE_ROWS, len(source_slice))].copy().reset_index(drop=True)
            stage_slices.append((slice_index, stage_slice, sim.precompute_indicators(stage_slice)))
        stage_aggregate_rows, _ = evaluate_configs(
            sim,
            configs,
            stage_slices,
            out_dir,
            stage_progress_path,
            stage_completed_path,
            completed_ids=set(),
            progress_prefix="Stage 1 evaluating",
        )
        for row in stage_aggregate_rows:
            row["behavior_signature"] = behavior_signature(row)
        stage_rows = sorted(
            stage_aggregate_rows,
            key=lambda row: (
                status_priority(row["classification"]),
                -safe_float(row.get("rank_score"), -math.inf),
                -safe_float(row.get("total_net_bps"), -math.inf),
            ),
        )
        keep_count = max(STAGE_MIN_KEEP, int(math.ceil(len(stage_rows) * STAGE_KEEP_FRACTION)))
        keep_count = min(len(stage_rows), keep_count)
        kept_ids: list[int] = []
        seen_signatures: set[str] = set()
        for row in stage_rows:
            signature = str(row.get("behavior_signature", ""))
            if not DISABLE_DEDUPE and signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            kept_ids.append(int(row["config_id"]))
            if len(kept_ids) >= keep_count:
                break
        selected_configs = [configs[idx] for idx in kept_ids]
        selected_original_ids = kept_ids
        pd.DataFrame(stage_rows).to_csv(out_dir / "stage1_results.csv", index=False)
        (out_dir / "stage1_kept_config_ids.txt").write_text("\n".join(str(idx) for idx in kept_ids) + "\n", encoding="utf-8")
        if RESUME_FROM_OUTPUT_DIR:
            completed_path.unlink(missing_ok=True)
    completed_ids = load_completed_ids(completed_path) if RESUME_FROM_OUTPUT_DIR else set()
    aggregate_rows, per_slice_rows = evaluate_configs(
        sim,
        selected_configs,
        slices,
        out_dir,
        progress_path,
        completed_path,
        completed_ids=completed_ids,
        progress_prefix="Final evaluating",
    )
    for row in aggregate_rows:
        row["behavior_signature"] = behavior_signature(row)
    if stage_rows:
        stage_by_id = {int(row["config_id"]): row for row in stage_rows}
        for row in aggregate_rows:
            final_id = int(row["config_id"])
            original_id = selected_original_ids[final_id] if final_id < len(selected_original_ids) else final_id
            row["stage_original_config_id"] = original_id
            row["stage_behavior_signature"] = stage_by_id.get(original_id, {}).get("behavior_signature", "")

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
    best_payload = {**best, "runtime_seconds": round(time.perf_counter() - started, 3)}
    best_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8")
    save_top_details(sim, out_dir, aggregate_rows, selected_configs, slices)
    write_text(text_path, aggregate_rows, configs)

    print("Ladder risk walk-forward sweep complete")
    print(f"Configs: {len(configs)}")
    print(f"Configs evaluated final: {len(selected_configs)}")
    print(f"Slices: {len(slices)}")
    print(f"Runtime seconds: {time.perf_counter() - started:.3f}")
    print(f"CSV: {sweep_path}")
    print(f"TXT: {text_path}")
    print(f"Best contract: {best_path}")
    print("Safety: paper only. Public recorded data only. No private API. No orders. No training. No promotion. No champion mutation.")
    print(frame.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
