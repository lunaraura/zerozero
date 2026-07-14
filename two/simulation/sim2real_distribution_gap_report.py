#!/usr/bin/env python3
"""Compare synthetic and real market distributions for sim2real drift."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[3]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

from tiny.core import tiny_io, tiny_paths


ROOT = tiny_paths.ROOT
SYMBOL = tiny_paths.get_symbol()
REAL_VENUE = os.getenv("SIM2REAL_REAL_VENUE", tiny_paths.get_primary_venue())
SYNTHETIC_VENUE = os.getenv("SIM2REAL_SYNTHETIC_VENUE", "simulated")
REAL_PATH = Path(os.getenv("SIM2REAL_REAL_PATH", tiny_paths.symbol_flow_path(SYMBOL, REAL_VENUE, "10s")))
SYNTHETIC_PATH = Path(os.getenv("SIM2REAL_SYNTHETIC_PATH", tiny_paths.symbol_flow_path(SYMBOL, SYNTHETIC_VENUE, "10s")))
OUTPUT_PATH = Path(
    os.getenv(
        "SIM2REAL_GAP_REPORT_PATH",
        tiny_paths.report_path(SYMBOL, SYNTHETIC_VENUE, "sim2real_distribution_gap_report", "csv"),
    )
)
JSON_OUTPUT_PATH = OUTPUT_PATH.with_suffix(".json")

if not REAL_PATH.is_absolute():
    REAL_PATH = ROOT / REAL_PATH
if not SYNTHETIC_PATH.is_absolute():
    SYNTHETIC_PATH = ROOT / SYNTHETIC_PATH
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = ROOT / OUTPUT_PATH


def read_flow(path: Path, label: str) -> pd.DataFrame:
    try:
        return tiny_io.read_csv_required(path, label, chunksize=100_000)
    except tiny_io.TinyIOError:
        return pd.DataFrame()


def numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def has_simulation_run_groups(frame: pd.DataFrame) -> bool:
    if "simulation_run_id" not in frame.columns:
        return False
    run_id = frame["simulation_run_id"].fillna("").astype(str).str.strip()
    return bool(run_id.ne("").any())


def grouped_path_metrics(frame: pd.DataFrame, mid: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    returns = pd.Series(np.nan, index=frame.index, dtype="float64")
    drawdown = pd.Series(np.nan, index=frame.index, dtype="float64")
    volatility = pd.Series(np.nan, index=frame.index, dtype="float64")
    run_id = frame["simulation_run_id"].fillna("").astype(str).str.strip()
    for _, indexes in run_id.groupby(run_id, sort=False).groups.items():
        group_mid = mid.loc[indexes]
        group_returns = np.log(group_mid / group_mid.shift(1)).replace([np.inf, -np.inf], np.nan) * 10_000.0
        group_running_max = group_mid.cummax()
        returns.loc[indexes] = group_returns
        drawdown.loc[indexes] = (group_mid / group_running_max - 1.0) * 10_000.0
        volatility.loc[indexes] = group_returns.rolling(30, min_periods=3).std()
    return returns, drawdown, volatility


def metric_series(frame: pd.DataFrame, group_by_simulation_run: bool = False) -> dict[str, pd.Series]:
    mid_source = "mid_price" if "mid_price" in frame.columns else "close"
    mid = numeric(frame, mid_source).where(lambda value: value > 0)
    if group_by_simulation_run and has_simulation_run_groups(frame):
        returns, drawdown, volatility = grouped_path_metrics(frame, mid)
    else:
        returns = np.log(mid / mid.shift(1)).replace([np.inf, -np.inf], np.nan) * 10_000.0
        running_max = mid.cummax()
        drawdown = (mid / running_max - 1.0) * 10_000.0
        volatility = returns.rolling(30, min_periods=3).std()
    depth_10bps = numeric(frame, "bid_depth_10bps", 0.0).fillna(0.0) + numeric(frame, "ask_depth_10bps", 0.0).fillna(0.0)
    volume = numeric(frame, "total_trade_volume_10s")
    if volume.isna().all():
        volume = numeric(frame, "volume")
    return {
        "spread_bps": numeric(frame, "spread_percent") * 10_000.0,
        "depth_10bps": depth_10bps,
        "volume": volume,
        "imbalance_10bps": numeric(frame, "order_book_imbalance_10bps"),
        "return_bps": returns,
        "drawdown_bps": drawdown,
        "volatility_30row_bps": volatility,
    }


def summary(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {key: math.nan for key in ["count", "mean", "std", "p10", "p50", "p90"]}
    return {
        "count": float(len(clean)),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "p10": float(clean.quantile(0.10)),
        "p50": float(clean.quantile(0.50)),
        "p90": float(clean.quantile(0.90)),
    }


def build_report(real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    real_metrics = metric_series(real)
    synthetic_metrics = metric_series(synthetic, group_by_simulation_run=True)
    rows = []
    for metric in real_metrics:
        real_summary = summary(real_metrics[metric])
        synthetic_summary = summary(synthetic_metrics[metric])
        for stat in ["count", "mean", "std", "p10", "p50", "p90"]:
            real_value = real_summary[stat]
            synthetic_value = synthetic_summary[stat]
            absolute_gap = synthetic_value - real_value if np.isfinite([real_value, synthetic_value]).all() else math.nan
            relative_gap = absolute_gap / real_value if np.isfinite(absolute_gap) and abs(real_value) > 1e-12 else math.nan
            rows.append(
                {
                    "symbol": SYMBOL,
                    "metric": metric,
                    "stat": stat,
                    "real_value": real_value,
                    "synthetic_value": synthetic_value,
                    "absolute_gap": absolute_gap,
                    "relative_gap": relative_gap,
                    "real_rows": len(real),
                    "synthetic_rows": len(synthetic),
                    "real_path": str(REAL_PATH),
                    "synthetic_path": str(SYNTHETIC_PATH),
                    "paper_only": True,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    real = read_flow(REAL_PATH, "real 10s flow")
    synthetic = read_flow(SYNTHETIC_PATH, "synthetic 10s flow")
    report = build_report(real, synthetic)
    tiny_io.safe_write_csv_atomic(report, OUTPUT_PATH)
    tiny_io.safe_write_json_atomic(
        {
            "symbol": SYMBOL,
            "real_path": str(REAL_PATH),
            "synthetic_path": str(SYNTHETIC_PATH),
            "output_path": str(OUTPUT_PATH),
            "real_rows": int(len(real)),
            "synthetic_rows": int(len(synthetic)),
            "metrics": sorted(report["metric"].unique().tolist()) if len(report) else [],
            "paper_only": True,
        },
        JSON_OUTPUT_PATH,
    )
    print("Sim2real distribution gap report")
    print(f"Real rows: {len(real)} path={REAL_PATH}")
    print(f"Synthetic rows: {len(synthetic)} path={SYNTHETIC_PATH}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
