#!/usr/bin/env python3
"""Validate a risk-controlled ladder baseline across recorded-data slices."""

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
BEST_CONTRACT_ENV = os.getenv("LADDER_BEST_CONTRACT_PATH", "").strip()
OUTPUT_ROOT = Path(
    os.getenv(
        "LADDER_VALIDATION_OUTPUT_DIR",
        PROJECT_ROOT / "data" / "research" / "ladder_walkforward_validation",
    )
)
SLICE_ROWS = int(float(os.getenv("LADDER_SLICE_ROWS", "200000")))
STEP_ROWS = int(float(os.getenv("LADDER_STEP_ROWS", "200000")))
MAX_SLICES = int(float(os.getenv("LADDER_MAX_SLICES", "10")))
CONFIG_SOURCE = os.getenv("LADDER_CONFIG_SOURCE", "best_contract").strip().lower()
FIXED_CONFIGS_ENV = os.getenv("LADDER_FIXED_CONFIGS", "").strip()
SIM_PATH = PROJECT_ROOT / "scripts" / "tiny" / "simulate_path_aware_ladder_baseline.py"


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def safe_int(value: Any, default: int = 0) -> int:
    value = safe_float(value, math.nan)
    return int(value) if math.isfinite(value) else default


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_simulator_module():
    spec = importlib.util.spec_from_file_location("ladder_sim", SIM_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load simulator module: {SIM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latest_best_contract_path() -> Path:
    if BEST_CONTRACT_ENV:
        path = Path(BEST_CONTRACT_ENV)
        return path if path.is_absolute() else PROJECT_ROOT / path
    candidates = sorted(
        (PROJECT_ROOT / "data" / "research" / "ladder_baselines").glob("ladder_baseline_*/best_ladder_contract.json"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        raise SystemExit("No best_ladder_contract.json found. Set LADDER_BEST_CONTRACT_PATH.")
    return candidates[0]


def parse_fixed_configs() -> list[dict[str, Any]]:
    if not FIXED_CONFIGS_ENV:
        raise SystemExit("LADDER_CONFIG_SOURCE=fixed_grid requires LADDER_FIXED_CONFIGS.")
    if FIXED_CONFIGS_ENV.lstrip().startswith("["):
        payload = json.loads(FIXED_CONFIGS_ENV)
        if not isinstance(payload, list):
            raise SystemExit("LADDER_FIXED_CONFIGS JSON must be a list of objects.")
        return [dict(item) for item in payload if isinstance(item, dict)]
    configs = []
    for chunk in FIXED_CONFIGS_ENV.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        row: dict[str, Any] = {}
        for part in chunk.split(","):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            number = safe_float(value)
            row[key] = number if math.isfinite(number) else value
        configs.append(row)
    if not configs:
        raise SystemExit("No fixed configs parsed from LADDER_FIXED_CONFIGS.")
    return configs


def contract_fields(contract: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "anchor_mode",
        "anchor_window",
        "vol_window",
        "spacing_mode",
        "vol_mult",
        "rung_count",
        "take_profit_mult",
        "min_spacing_bps",
        "cost_bps",
        "max_open_units",
        "stop_loss_bps",
        "max_hold_buckets",
        "cooldown_after_stop_buckets",
        "disable_buys_when_price_below_anchor_bps",
        "disable_buys_when_ema_slope_below_bps",
        "force_flat_at_end",
        "range_break_buffer_bps",
        "model_gate_mode",
        "shadow_dir",
    ]
    return {field: contract[field] for field in fields if field in contract}


def load_contracts() -> tuple[list[dict[str, Any]], str]:
    if CONFIG_SOURCE == "fixed_grid":
        return parse_fixed_configs(), "fixed_grid"
    if CONFIG_SOURCE != "best_contract":
        raise SystemExit("LADDER_CONFIG_SOURCE must be best_contract or fixed_grid.")
    path = latest_best_contract_path()
    contract = load_json(path)
    if not contract:
        raise SystemExit(f"Could not load best contract: {path}")
    return [contract_fields(contract)], str(path)


def timestamp_iso(value: Any) -> str:
    value = safe_float(value)
    if not math.isfinite(value):
        return ""
    unit = "ms" if value > 1e11 else "s"
    return pd.to_datetime(value, unit=unit, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")


def output_dir() -> Path:
    path = OUTPUT_ROOT / f"ladder_walkforward_validation_{stamp()}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def configure_simulator(sim: Any, contract: dict[str, Any]) -> None:
    sim.MODEL_GATE_MODE = str(contract.get("model_gate_mode") or sim.MODEL_GATE_MODE or "none").strip().lower()
    sim.SHADOW_DIR_ENV = str(contract.get("shadow_dir") or "")


def run_slice(sim: Any, source_slice: pd.DataFrame, contract: dict[str, Any], slice_index: int) -> dict[str, Any]:
    configure_simulator(sim, contract)
    indicators = sim.precompute_indicators(source_slice)
    gate = None
    if sim.SHADOW_DIR_ENV and sim.MODEL_GATE_MODE != "none":
        gate = sim.build_shadow_gate(source_slice, Path(sim.SHADOW_DIR_ENV))
    result, _, _ = sim.simulate_config(source_slice, indicators, gate, contract, save_paths=False)
    return {
        "slice_index": slice_index,
        "start_time": timestamp_iso(source_slice["timestamp"].iloc[0]),
        "end_time": timestamp_iso(source_slice["timestamp"].iloc[-1]),
        "rows": len(source_slice),
        "force_flat_total_net_bps": result.get("force_flat_total_net_bps"),
        "realized_net_bps": result.get("realized_net_bps"),
        "unrealized_net_bps": result.get("unrealized_net_bps"),
        "max_drawdown_bps": result.get("max_drawdown_bps"),
        "number_of_trades": result.get("number_of_trades"),
        "take_profit_exit_count": result.get("take_profit_exit_count"),
        "stop_loss_exit_count": result.get("stop_loss_exit_count"),
        "timeout_exit_count": result.get("timeout_exit_count"),
        "final_liquidation_count": result.get("final_liquidation_count"),
        "win_rate": result.get("win_rate"),
        "max_inventory": result.get("max_inventory"),
        "ending_inventory": result.get("ending_inventory"),
        "exposure_time_fraction": result.get("exposure_time_fraction"),
        "total_net_cost_0_05_bps": result.get("total_net_cost_0_05_bps"),
        "total_net_cost_0_10_bps": result.get("total_net_cost_0_1_bps"),
        "total_net_cost_0_25_bps": result.get("total_net_cost_0_25_bps"),
        "rank_score": result.get("rank_score"),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nets = np.asarray([safe_float(row.get("force_flat_total_net_bps")) for row in rows], dtype=np.float64)
    dd = np.asarray([safe_float(row.get("max_drawdown_bps")) for row in rows], dtype=np.float64)
    cost025 = np.asarray([safe_float(row.get("total_net_cost_0_25_bps")) for row in rows], dtype=np.float64)
    valid = np.isfinite(nets)
    if not valid.any():
        return {"classification": "reject", "reason": "no_valid_slices"}
    nets = nets[valid]
    dd = dd[np.isfinite(dd)]
    cost025 = cost025[np.isfinite(cost025)]
    positive_fraction = float(np.mean(nets > 0))
    total_net = float(np.sum(nets))
    worst_slice = float(np.min(nets))
    total_drawdown = float(np.min(dd)) if len(dd) else math.nan
    cost025_positive = float(np.mean(cost025 > 0)) if len(cost025) else math.nan
    failed_count = int(np.sum(nets <= 0))
    if total_net <= 0 or abs(worst_slice) > max(total_net, 1e-9):
        classification = "reject"
    elif positive_fraction >= 0.6 and (not math.isfinite(cost025_positive) or cost025_positive >= 0.5) and abs(total_drawdown) <= max(total_net * 2.0, 1e-9):
        classification = "stable_baseline"
    else:
        classification = "fragile_baseline"
    return {
        "classification": classification,
        "slices": int(len(nets)),
        "positive_slice_fraction": positive_fraction,
        "median_slice_net_bps": float(np.median(nets)),
        "worst_slice_net_bps": worst_slice,
        "total_net_bps": total_net,
        "total_max_drawdown_bps": total_drawdown,
        "cost_0_25_positive_slice_fraction": cost025_positive,
        "failed_slice_count": failed_count,
    }


def write_summary(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], config_source: str) -> None:
    lines = [
        "Ladder Walk-Forward Validation",
        "",
        f"Created at: {stamp()}",
        f"Source: {SOURCE_PATH}",
        f"Config source: {config_source}",
        f"Slice rows: {SLICE_ROWS}",
        f"Step rows: {STEP_ROWS}",
        f"Max slices: {MAX_SLICES}",
        "",
        "Aggregate:",
    ]
    for key, value in summary.items():
        lines.append(f"  {key}: {value}")
    lines.extend(["", "Slices:"])
    for row in rows:
        lines.append(
            "  "
            f"slice={row['slice_index']} rows={row['rows']} net={row['force_flat_total_net_bps']} "
            f"dd={row['max_drawdown_bps']} trades={row['number_of_trades']} "
            f"tp={row['take_profit_exit_count']} sl={row['stop_loss_exit_count']} timeout={row['timeout_exit_count']} "
            f"final={row['final_liquidation_count']} cost025={row['total_net_cost_0_25_bps']} "
            f"{row['start_time']}->{row['end_time']}"
        )
    lines.extend(
        [
            "",
            "Safety:",
            "  paper_only=true",
            "  private_api=false",
            "  orders=false",
            "  training=false",
            "  promotion=false",
            "  champion_mutation=false",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    sim = load_simulator_module()
    frame = sim.load_price_data(SOURCE_PATH)
    contracts, config_source = load_contracts()
    if len(contracts) != 1:
        raise SystemExit("This validator currently expects one fixed config or one best contract.")
    contract = contracts[0]
    contract["force_flat_at_end"] = True

    rows = []
    for slice_index, start in enumerate(range(0, len(frame) - SLICE_ROWS + 1, STEP_ROWS)):
        if slice_index >= MAX_SLICES:
            break
        source_slice = frame.iloc[start : start + SLICE_ROWS].copy().reset_index(drop=True)
        if len(source_slice) < SLICE_ROWS:
            continue
        rows.append(run_slice(sim, source_slice, contract, slice_index))
    if not rows:
        raise SystemExit("No validation slices produced.")
    summary = aggregate(rows)

    out_dir = output_dir()
    slices_path = out_dir / "ladder_walkforward_slices.csv"
    summary_csv_path = out_dir / "ladder_walkforward_summary.csv"
    summary_txt_path = out_dir / "ladder_walkforward_summary.txt"
    contract_path = out_dir / "validated_ladder_contract.json"
    pd.DataFrame(rows).to_csv(slices_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(summary_txt_path, summary, rows, config_source)

    print("Ladder walk-forward validation complete")
    print(f"Slices: {len(rows)}")
    print(f"Classification: {summary.get('classification')}")
    print(f"Slices CSV: {slices_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary TXT: {summary_txt_path}")
    print("Safety: paper only. No private API. No orders. No training. No promotion. No champion mutation.")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
