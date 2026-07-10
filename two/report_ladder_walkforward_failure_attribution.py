#!/usr/bin/env python3
"""Explain ladder walk-forward failures with market and exit attribution."""

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
VALIDATION_DIR_ENV = os.getenv("LADDER_VALIDATION_DIR", "").strip()
SOURCE_PATH = Path(os.getenv("LADDER_SOURCE_PATH", PROJECT_ROOT / "data" / "realtime" / "kraken" / "SOLUSDT_10s_flow.csv"))
BEST_CONTRACT_ENV = os.getenv("LADDER_BEST_CONTRACT_PATH", "").strip()
OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "ladder_failure_attribution"
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
    value = safe_float(value)
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


def latest_validation_dir() -> Path:
    if VALIDATION_DIR_ENV:
        path = Path(VALIDATION_DIR_ENV)
        return path if path.is_absolute() else PROJECT_ROOT / path
    candidates = sorted(
        (PROJECT_ROOT / "data" / "research" / "ladder_walkforward_validation").glob("ladder_walkforward_validation_*"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        raise SystemExit("No ladder validation directory found. Set LADDER_VALIDATION_DIR.")
    return candidates[0]


def latest_best_contract_path(validation_dir: Path) -> Path:
    if BEST_CONTRACT_ENV:
        path = Path(BEST_CONTRACT_ENV)
        return path if path.is_absolute() else PROJECT_ROOT / path
    local = validation_dir / "validated_ladder_contract.json"
    if local.exists():
        return local
    candidates = sorted(
        (PROJECT_ROOT / "data" / "research" / "ladder_baselines").glob("ladder_baseline_*/best_ladder_contract.json"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        raise SystemExit("No best_ladder_contract.json found. Set LADDER_BEST_CONTRACT_PATH.")
    return candidates[0]


def timestamp_ms(value: Any) -> float:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return math.nan
    return float(ts.timestamp() * 1000.0)


def timestamp_iso(value: Any) -> str:
    value = safe_float(value)
    if not math.isfinite(value):
        return ""
    unit = "ms" if value > 1e11 else "s"
    return pd.to_datetime(value, unit=unit, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")


def max_drawdown_bps(price: np.ndarray) -> float:
    if len(price) == 0:
        return math.nan
    peak = np.maximum.accumulate(price)
    drawdown = 10_000.0 * np.log(price / peak)
    return float(np.nanmin(drawdown))


def max_runup_bps(price: np.ndarray) -> float:
    if len(price) == 0:
        return math.nan
    trough = np.minimum.accumulate(price)
    runup = 10_000.0 * np.log(price / trough)
    return float(np.nanmax(runup))


def market_diagnostics(frame: pd.DataFrame, bucket_seconds: int) -> dict[str, Any]:
    price = frame["price"].to_numpy(dtype=np.float64)
    returns = np.zeros(len(price), dtype=np.float64)
    returns[1:] = 10_000.0 * np.log(price[1:] / price[:-1])
    ema60 = pd.Series(price).ewm(span=60, adjust=False, min_periods=60).mean()
    ema60_slope = 10_000.0 * np.log(ema60 / ema60.shift(1))
    elapsed_hours = max(len(price) * bucket_seconds / 3600.0, 1e-9)
    price_return = 10_000.0 * math.log(price[-1] / price[0]) if len(price) > 1 else math.nan
    realized_vol = float(np.nanstd(returns))
    range_bps = 10_000.0 * math.log(np.nanmax(price) / np.nanmin(price)) if len(price) else math.nan
    trend_slope = price_return / elapsed_hours if math.isfinite(price_return) else math.nan
    chop_score = range_bps / max(abs(price_return), realized_vol, 1e-9) if math.isfinite(range_bps) else math.nan
    trend_score = abs(price_return) / max(realized_vol * math.sqrt(max(len(price), 1)), 1e-9) if math.isfinite(price_return) else math.nan
    return {
        "price_return_bps": price_return,
        "max_price_drawdown_bps": max_drawdown_bps(price),
        "max_price_runup_bps": max_runup_bps(price),
        "realized_vol_bps": realized_vol,
        "range_bps": range_bps,
        "trend_slope_bps_per_hour": trend_slope,
        "ema60_slope_mean_bps": float(np.nanmean(ema60_slope)),
        "ema60_slope_negative_fraction": float(np.nanmean(ema60_slope < 0)),
        "time_below_ema60_fraction": float(np.nanmean(pd.Series(price) < ema60)),
        "chop_score": chop_score,
        "trend_score": trend_score,
    }


def dominant_loss_source(trades: pd.DataFrame) -> tuple[str, str, float, float, float]:
    if trades.empty or "exit_reason" not in trades.columns:
        return "", "", math.nan, math.nan, math.nan
    trades = trades.copy()
    trades["net_bps"] = pd.to_numeric(trades["net_bps"], errors="coerce")
    by_reason = trades.groupby("exit_reason")["net_bps"].sum().sort_values()
    worst_reason = str(by_reason.index[0]) if len(by_reason) else ""
    stop_loss_net = float(by_reason.get("stop_loss", 0.0))
    timeout_net = float(by_reason.get("timeout", 0.0))
    final_net = float(by_reason.get("final_liquidation", 0.0))
    if stop_loss_net <= timeout_net and stop_loss_net <= final_net and stop_loss_net < 0:
        dominant = "stop_loss"
    elif timeout_net <= stop_loss_net and timeout_net <= final_net and timeout_net < 0:
        dominant = "timeout"
    elif final_net < 0:
        dominant = "final_liquidation"
    else:
        dominant = "mixed_or_no_loss"
    return worst_reason, dominant, stop_loss_net, timeout_net, final_net


def classify(row: dict[str, Any]) -> str:
    trades = safe_int(row.get("number_of_trades"))
    net = safe_float(row.get("force_flat_total_net_bps"), 0.0)
    price_return = safe_float(row.get("price_return_bps"), 0.0)
    vol = safe_float(row.get("realized_vol_bps"), 0.0)
    chop = safe_float(row.get("chop_score"), 0.0)
    below = safe_float(row.get("time_below_ema60_fraction"), 0.0)
    timeout_count = safe_int(row.get("timeout_exit_count"))
    stop_count = safe_int(row.get("stop_loss_exit_count"))
    tp_count = safe_int(row.get("take_profit_exit_count"))
    if trades < 20:
        return "insufficient_trades"
    if price_return < -100 and below > 0.55:
        return "downtrend_failure"
    if timeout_count > max(tp_count, stop_count) and safe_float(row.get("timeout_net_bps"), 0.0) < 0:
        return "timeout_churn"
    if stop_count > max(tp_count, timeout_count) or safe_float(row.get("stop_loss_net_bps"), 0.0) < -abs(net):
        return "stop_loss_churn"
    if vol < 1.0 and trades < 50:
        return "low_vol_no_edge"
    if vol > 4.0 and chop > 5.0:
        return "high_vol_whipsaw"
    if abs(price_return) > 100 and safe_float(row.get("ema60_slope_negative_fraction"), 0.0) > 0.45:
        return "anchor_lag_failure"
    return "mixed_failure"


def output_dir() -> Path:
    path = OUTPUT_ROOT / f"ladder_failure_attribution_{stamp()}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_text(path: Path, rows: list[dict[str, Any]], summary: pd.DataFrame, validation_dir: Path, contract_path: Path) -> None:
    lines = [
        "Ladder Walk-Forward Failure Attribution",
        "",
        f"Created at: {stamp()}",
        f"Validation dir: {validation_dir}",
        f"Contract path: {contract_path}",
        f"Source: {SOURCE_PATH}",
        "",
        "Failure classes:",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "  "
            f"{row['failure_class']}: slices={row['slices']} total_net={row['total_net_bps']} "
            f"avg_trades={row['avg_trades']} avg_price_return={row['avg_price_return_bps']}"
        )
    lines.extend(["", "Slices:"])
    for row in rows:
        lines.append(
            "  "
            f"slice={row['slice_index']} class={row['failure_class']} net={row['force_flat_total_net_bps']} "
            f"price_ret={row['price_return_bps']:.3f} dd={row['max_drawdown_bps']} "
            f"vol={row['realized_vol_bps']:.3f} chop={row['chop_score']:.3f} "
            f"tp={row['take_profit_exit_count']} sl={row['stop_loss_exit_count']} timeout={row['timeout_exit_count']} "
            f"dominant={row['dominant_loss_source']} worst_exit={row['worst_exit_reason']}"
        )
    lines.extend(
        [
            "",
            "Safety:",
            "  report_only=true",
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
    validation_dir = latest_validation_dir()
    slices_path = validation_dir / "ladder_walkforward_slices.csv"
    if not slices_path.exists():
        raise SystemExit(f"Validation slices CSV not found: {slices_path}")
    validation = pd.read_csv(slices_path)
    contract_path = latest_best_contract_path(validation_dir)
    contract = load_json(contract_path)
    if not contract:
        raise SystemExit(f"Could not load contract: {contract_path}")

    sim = load_simulator_module()
    frame = sim.load_price_data(SOURCE_PATH)
    bucket_seconds = int(float(contract.get("bucket_seconds") or sim.BUCKET_SECONDS))
    rows = []
    for _, validation_row in validation.iterrows():
        start_ms = timestamp_ms(validation_row.get("start_time"))
        end_ms = timestamp_ms(validation_row.get("end_time"))
        source_slice = frame[(frame["timestamp"] >= start_ms) & (frame["timestamp"] <= end_ms)].copy().reset_index(drop=True)
        if source_slice.empty:
            continue
        indicators = sim.precompute_indicators(source_slice)
        gate = None
        sim.MODEL_GATE_MODE = str(contract.get("model_gate_mode") or "none").strip().lower()
        sim.SHADOW_DIR_ENV = str(contract.get("shadow_dir") or "")
        if sim.SHADOW_DIR_ENV and sim.MODEL_GATE_MODE != "none":
            gate = sim.build_shadow_gate(source_slice, Path(sim.SHADOW_DIR_ENV))
        result, trades, _ = sim.simulate_config(source_slice, indicators, gate, contract, save_paths=False)
        worst_exit, dominant, stop_net, timeout_net, final_net = dominant_loss_source(trades)
        hold = pd.to_numeric(trades.get("hold_buckets", pd.Series(dtype=float)), errors="coerce")
        row = {
            "slice_index": safe_int(validation_row.get("slice_index")),
            "start_time": validation_row.get("start_time"),
            "end_time": validation_row.get("end_time"),
            "rows": len(source_slice),
            **market_diagnostics(source_slice, bucket_seconds),
            "force_flat_total_net_bps": safe_float(validation_row.get("force_flat_total_net_bps")),
            "max_drawdown_bps": safe_float(validation_row.get("max_drawdown_bps")),
            "take_profit_exit_count": safe_int(validation_row.get("take_profit_exit_count")),
            "stop_loss_exit_count": safe_int(validation_row.get("stop_loss_exit_count")),
            "timeout_exit_count": safe_int(validation_row.get("timeout_exit_count")),
            "final_liquidation_count": safe_int(validation_row.get("final_liquidation_count")),
            "stop_loss_net_bps": stop_net,
            "timeout_net_bps": timeout_net,
            "final_liquidation_net_bps": final_net,
            "average_hold_buckets": float(hold.mean()) if len(hold.dropna()) else math.nan,
            "max_inventory": safe_int(validation_row.get("max_inventory")),
            "exposure_time_fraction": safe_float(validation_row.get("exposure_time_fraction")),
            "worst_exit_reason": worst_exit,
            "dominant_loss_source": dominant,
            "number_of_trades": safe_int(validation_row.get("number_of_trades")),
        }
        row["failure_class"] = classify(row)
        rows.append(row)
    if not rows:
        raise SystemExit("No attribution rows produced.")

    frame_out = pd.DataFrame(rows)
    summary = (
        frame_out.groupby("failure_class", dropna=False)
        .agg(
            slices=("slice_index", "count"),
            total_net_bps=("force_flat_total_net_bps", "sum"),
            avg_trades=("number_of_trades", "mean"),
            avg_price_return_bps=("price_return_bps", "mean"),
            avg_realized_vol_bps=("realized_vol_bps", "mean"),
        )
        .reset_index()
        .sort_values(["slices", "total_net_bps"], ascending=[False, True])
    )

    out_dir = output_dir()
    csv_path = out_dir / "ladder_failure_attribution.csv"
    txt_path = out_dir / "ladder_failure_attribution.txt"
    summary_path = out_dir / "regime_failure_summary.csv"
    frame_out.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_text(txt_path, rows, summary, validation_dir, contract_path)
    print("Ladder failure attribution complete")
    print(f"Rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"TXT: {txt_path}")
    print(f"Regime summary: {summary_path}")
    print("Safety: report only. Paper only. No private API. No orders. No training. No promotion. No champion mutation.")
    print(summary.to_string(index=False))
    print(frame_out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
