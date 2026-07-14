#!/usr/bin/env python3
"""Report horizon opportunity versus execution cost on recorded public data.

This is a paper-only feasibility report. It does not train, promote, mutate
champions, use private APIs, or place orders.
"""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "realtime" / "kraken" / "SOLUSDT_10s_flow.csv"

SOURCE_PATH = Path(os.getenv("RAWSEQ_HORIZON_SOURCE_PATH", str(DEFAULT_SOURCE))).expanduser()
if not SOURCE_PATH.is_absolute():
    SOURCE_PATH = PROJECT_ROOT / SOURCE_PATH
OUTPUT_DIR = Path(
    os.getenv(
        "RAWSEQ_HORIZON_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_horizon_feasibility"),
    )
).expanduser()
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper() or "SOLUSDT"
VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower() or "kraken"
INSTRUMENT = os.getenv("RAWSEQ_EXECUTION_INSTRUMENT", "inventory_spot_long_flat").strip().lower()
HORIZONS_SECONDS = [
    int(float(item.strip()))
    for item in os.getenv("RAWSEQ_HORIZONS_SECONDS", "30,60,300,900").split(",")
    if item.strip()
]
ROUND_TRIP_COSTS_BPS = [
    float(item.strip())
    for item in os.getenv("RAWSEQ_ROUND_TRIP_COST_BPS_LIST", "0.1,1,5,10").split(",")
    if item.strip()
]
MAX_ROWS_ENV = os.getenv("RAWSEQ_HORIZON_MAX_ROWS", "").strip()
REFERENCE_GROSS_EDGE_BPS = float(os.getenv("RAWSEQ_REFERENCE_GROSS_EDGE_BPS", "0.2"))


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def infer_column(frame: pd.DataFrame, choices: list[str], label: str) -> str:
    for column in choices:
        if column in frame.columns:
            return column
    raise SystemExit(f"Could not find {label} column. Tried: {choices}")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def load_price_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Source path does not exist: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if MAX_ROWS_ENV:
        max_rows = int(float(MAX_ROWS_ENV))
        if max_rows > 0:
            frame = frame.tail(max_rows).copy()
    timestamp_col = infer_column(frame, ["timestamp", "time_ms", "ts"], "timestamp")
    price_col = infer_column(frame, ["price", "mid_price", "close", "last"], "price")
    out = pd.DataFrame(
        {
            "timestamp": pd.to_numeric(frame[timestamp_col], errors="coerce"),
            "price": pd.to_numeric(frame[price_col], errors="coerce"),
        }
    )
    if "time" in frame.columns:
        out["time"] = frame["time"].astype(str)
    else:
        out["time"] = ""
    out = out.dropna(subset=["timestamp", "price"]).sort_values("timestamp").drop_duplicates("timestamp")
    return out.reset_index(drop=True)


def bucket_seconds(frame: pd.DataFrame) -> float:
    diffs = pd.to_numeric(frame["timestamp"], errors="coerce").diff().dropna().to_numpy(dtype=np.float64)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 10.0
    return float(np.median(diffs) / 1000.0)


def future_window_extreme(price: pd.Series, offset: int, kind: str) -> pd.Series:
    shifted = price.shift(-1)
    reversed_shifted = shifted.iloc[::-1]
    if kind == "max":
        rolled = reversed_shifted.rolling(offset, min_periods=offset).max()
    elif kind == "min":
        rolled = reversed_shifted.rolling(offset, min_periods=offset).min()
    else:
        raise ValueError(kind)
    return rolled.iloc[::-1].reset_index(drop=True)


def percentile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if len(values) else math.nan


def horizon_metrics(frame: pd.DataFrame, horizon_seconds: int, cost_bps: float, bucket_s: float) -> dict[str, Any]:
    offset = max(1, int(math.ceil(horizon_seconds / max(bucket_s, 1e-9))))
    price = pd.to_numeric(frame["price"], errors="coerce").reset_index(drop=True)
    future_price = price.shift(-offset)
    future_high = future_window_extreme(price, offset, "max")
    future_low = future_window_extreme(price, offset, "min")

    terminal = 10_000.0 * np.log(future_price / price)
    high_from_now = 10_000.0 * np.log(future_high / price)
    low_from_now = 10_000.0 * np.log(future_low / price)
    data = pd.DataFrame(
        {
            "terminal_return_bps": terminal,
            "long_mfe_bps": np.maximum(high_from_now, 0.0),
            "long_mae_bps": np.minimum(low_from_now, 0.0),
            "short_mfe_bps": np.maximum(-low_from_now, 0.0),
            "short_mae_bps": np.minimum(-high_from_now, 0.0),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return {
            "horizon_seconds": horizon_seconds,
            "cost_bps": cost_bps,
            "rows": 0,
            "status": "insufficient_rows",
        }

    terminal_arr = data["terminal_return_bps"].to_numpy(dtype=np.float64)
    abs_move = np.abs(terminal_arr)
    long_mfe = data["long_mfe_bps"].to_numpy(dtype=np.float64)
    short_mfe = data["short_mfe_bps"].to_numpy(dtype=np.float64)
    long_terminal_net = terminal_arr - cost_bps
    short_terminal_net = -terminal_arr - cost_bps
    best_direction_net = np.maximum(long_terminal_net, short_terminal_net)
    long_allowed = INSTRUMENT in {"kraken_spot_long_only", "inventory_spot_long_flat"}
    short_allowed = INSTRUMENT in {"margin_long_short", "perpetual_futures_long_short"}
    both_allowed = INSTRUMENT in {"margin_long_short", "perpetual_futures_long_short"}

    if both_allowed:
        opportunity_frequency = float(np.mean(best_direction_net > 0.0))
        theoretical_mfe_over_cost = float(np.mean(np.maximum(long_mfe, short_mfe) > cost_bps))
    elif long_allowed:
        opportunity_frequency = float(np.mean(long_terminal_net > 0.0))
        theoretical_mfe_over_cost = float(np.mean(long_mfe > cost_bps))
    elif short_allowed:
        opportunity_frequency = float(np.mean(short_terminal_net > 0.0))
        theoretical_mfe_over_cost = float(np.mean(short_mfe > cost_bps))
    else:
        opportunity_frequency = float(np.mean(abs_move > cost_bps))
        theoretical_mfe_over_cost = float(np.mean(np.maximum(long_mfe, short_mfe) > cost_bps))

    return {
        "symbol": SYMBOL,
        "venue": VENUE,
        "instrument": INSTRUMENT,
        "horizon_seconds": horizon_seconds,
        "horizon_offset_rows": offset,
        "estimated_bucket_seconds": bucket_s,
        "cost_bps": cost_bps,
        "rows": int(len(data)),
        "abs_move_median_bps": percentile(abs_move, 50),
        "abs_move_p75_bps": percentile(abs_move, 75),
        "abs_move_p90_bps": percentile(abs_move, 90),
        "abs_move_p95_bps": percentile(abs_move, 95),
        "abs_move_p99_bps": percentile(abs_move, 99),
        "terminal_return_mean_bps": float(np.mean(terminal_arr)),
        "long_terminal_opportunity_frequency": float(np.mean(long_terminal_net > 0.0)),
        "short_terminal_opportunity_frequency": float(np.mean(short_terminal_net > 0.0)),
        "instrument_opportunity_frequency": opportunity_frequency,
        "long_mfe_median_bps": percentile(long_mfe, 50),
        "long_mfe_p90_bps": percentile(long_mfe, 90),
        "long_mae_median_bps": percentile(data["long_mae_bps"].to_numpy(dtype=np.float64), 50),
        "short_mfe_median_bps": percentile(short_mfe, 50),
        "short_mfe_p90_bps": percentile(short_mfe, 90),
        "short_mae_median_bps": percentile(data["short_mae_bps"].to_numpy(dtype=np.float64), 50),
        "abs_terminal_over_cost_fraction": float(np.mean(abs_move > cost_bps)),
        "theoretical_mfe_over_cost_fraction": theoretical_mfe_over_cost,
        "median_abs_move_after_cost_bps": percentile(abs_move - cost_bps, 50),
        "p90_abs_move_after_cost_bps": percentile(abs_move - cost_bps, 90),
        "reference_gross_edge_bps": REFERENCE_GROSS_EDGE_BPS,
        "reference_edge_after_cost_bps": REFERENCE_GROSS_EDGE_BPS - cost_bps,
        "economic_feasibility_hint": (
            "cost_dominates_reference_edge"
            if REFERENCE_GROSS_EDGE_BPS - cost_bps <= 0.0
            else "reference_edge_exceeds_cost"
        ),
        "paper_only": True,
        "training": False,
        "promotion": False,
        "champion_mutation": False,
        "orders": False,
    }


def write_text(path: Path, rows: pd.DataFrame) -> None:
    lines = [
        "Rawseq Horizon Opportunity Versus Cost Report",
        "",
        f"Created at: {now_stamp()}",
        f"Source path: {SOURCE_PATH}",
        f"Symbol: {SYMBOL}",
        f"Venue: {VENUE}",
        f"Execution instrument: {INSTRUMENT}",
        f"Reference gross edge bps: {REFERENCE_GROSS_EDGE_BPS:g}",
        "",
        "Safety: paper_only=true training=false promotion=false champion_mutation=false orders=false",
        "",
        "Summary by horizon/cost:",
    ]
    for _, row in rows.iterrows():
        lines.append(
            "  "
            f"horizon={int(row['horizon_seconds'])}s cost={fmt(row['cost_bps'])} "
            f"rows={int(row['rows'])} median_abs={fmt(row['abs_move_median_bps'])} "
            f"p90_abs={fmt(row['abs_move_p90_bps'])} "
            f"long_opp={fmt(row['long_terminal_opportunity_frequency'])} "
            f"short_opp={fmt(row['short_terminal_opportunity_frequency'])} "
            f"mfe_over_cost={fmt(row['theoretical_mfe_over_cost_fraction'])} "
            f"hint={row['economic_feasibility_hint']}"
        )
    lines += [
        "",
        "Interpretation:",
        "  This report measures theoretical recorded-price opportunity only.",
        "  It is not model evidence and it does not account for queue position or fill probability.",
        "  If realistic round-trip costs dominate the reference gross edge, prefer longer horizons or abstention.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    frame = load_price_frame(SOURCE_PATH)
    bucket_s = bucket_seconds(frame)
    out_dir = OUTPUT_DIR / f"horizon_feasibility_{SYMBOL}_{VENUE}_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        horizon_metrics(frame, horizon, cost, bucket_s)
        for horizon in HORIZONS_SECONDS
        for cost in ROUND_TRIP_COSTS_BPS
    ]
    result = pd.DataFrame(rows)
    csv_path = out_dir / "horizon_opportunity_cost.csv"
    txt_path = out_dir / "horizon_opportunity_cost.txt"
    result.to_csv(csv_path, index=False)
    write_text(txt_path, result)
    print("Rawseq horizon opportunity-cost report complete")
    print(f"Rows: {len(result)}")
    print(f"CSV: {csv_path}")
    print(f"TXT: {txt_path}")
    print(result.head(20).to_string(index=False))
    print("Safety: paper_only=true training=false promotion=false champion_mutation=false orders=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
