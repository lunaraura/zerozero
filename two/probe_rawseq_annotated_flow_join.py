#!/usr/bin/env python3
"""Probe timestamp join quality between rawseq annotations and realtime 10s flow."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip()
TEST_FRAC = float(os.getenv("RAWSEQ_JOIN_TEST_FRAC", "0.20"))

BASE_DIR = PROJECT_ROOT / "data" / "realtime" / PRIMARY_VENUE
ANNOTATED_PATH = BASE_DIR / f"{SYMBOL}_tiny_price_prediction_evaluation_rows_with_rawseq_path_v1.csv"
FLOW_PATH = BASE_DIR / f"{SYMBOL}_10s_flow.csv"
OUTPUT_PATH = Path(
    os.getenv(
        "RAWSEQ_JOIN_OUTPUT_PATH",
        BASE_DIR / f"{SYMBOL}_rawseq_annotated_flow_join_probe.csv",
    )
)
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH
TEXT_OUTPUT_PATH = OUTPUT_PATH.with_suffix(".txt")

ANNOTATED_COLUMNS = [
    "timestamp",
    "time",
    "rawseq_path_pred_horizon_return_bps",
    "rawseq_path_actual_horizon_return_bps",
]
COST_COLUMNS = [
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
FLOW_COLUMNS = ["timestamp", "time", *COST_COLUMNS]


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except Exception:
        return math.nan


def load_csv_columns(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    try:
        frame = pd.read_csv(path, usecols=lambda column: column in columns, low_memory=False)
    except ValueError as exc:
        raise SystemExit(f"{label} missing required columns: {exc}") from exc
    if "timestamp" not in frame.columns:
        raise SystemExit(f"{label} missing timestamp column: {path}")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).copy()
    frame["timestamp"] = frame["timestamp"].astype("int64")
    return frame


def load_annotated_test_slice() -> pd.DataFrame:
    if not 0.0 < TEST_FRAC <= 1.0:
        raise SystemExit(f"RAWSEQ_JOIN_TEST_FRAC must be in (0, 1], got {TEST_FRAC}")
    frame = load_csv_columns(ANNOTATED_PATH, ANNOTATED_COLUMNS, "annotated rawseq rows")
    if frame.empty:
        raise SystemExit(f"annotated rawseq rows are empty: {ANNOTATED_PATH}")
    split_at = int(len(frame) * (1.0 - TEST_FRAC))
    return frame.iloc[split_at:].copy().reset_index(drop=True)


def timestamp_range(frame: pd.DataFrame) -> tuple[int | None, int | None]:
    if frame.empty:
        return None, None
    return int(frame["timestamp"].min()), int(frame["timestamp"].max())


def nearest_tolerance_diagnostics(annotated: pd.DataFrame, flow: pd.DataFrame) -> dict[str, Any]:
    if annotated.empty or flow.empty:
        return {
            "nearest_median_abs_ms": math.nan,
            "nearest_p95_abs_ms": math.nan,
            "nearest_within_1000ms": 0,
            "nearest_within_5000ms": 0,
            "nearest_within_10000ms": 0,
        }
    annotated_ts = np.sort(annotated["timestamp"].to_numpy(dtype="int64"))
    flow_ts = np.sort(flow["timestamp"].to_numpy(dtype="int64"))
    positions = np.searchsorted(flow_ts, annotated_ts)
    distances = np.full(len(annotated_ts), np.iinfo(np.int64).max, dtype="int64")
    valid_right = positions < len(flow_ts)
    distances[valid_right] = np.minimum(
        distances[valid_right],
        np.abs(flow_ts[positions[valid_right]] - annotated_ts[valid_right]),
    )
    valid_left = positions > 0
    distances[valid_left] = np.minimum(
        distances[valid_left],
        np.abs(flow_ts[positions[valid_left] - 1] - annotated_ts[valid_left]),
    )
    distances = distances[distances != np.iinfo(np.int64).max]
    if len(distances) == 0:
        return {
            "nearest_median_abs_ms": math.nan,
            "nearest_p95_abs_ms": math.nan,
            "nearest_within_1000ms": 0,
            "nearest_within_5000ms": 0,
            "nearest_within_10000ms": 0,
        }
    return {
        "nearest_median_abs_ms": float(np.median(distances)),
        "nearest_p95_abs_ms": float(np.quantile(distances, 0.95)),
        "nearest_within_1000ms": int((distances <= 1_000).sum()),
        "nearest_within_5000ms": int((distances <= 5_000).sum()),
        "nearest_within_10000ms": int((distances <= 10_000).sum()),
    }


def build_report() -> tuple[pd.DataFrame, str]:
    annotated = load_annotated_test_slice()
    flow = load_csv_columns(FLOW_PATH, FLOW_COLUMNS, "10s flow rows")
    if flow.empty:
        raise SystemExit(f"10s flow rows are empty: {FLOW_PATH}")

    joined = annotated.merge(
        flow,
        on="timestamp",
        how="left",
        suffixes=("_annotated", "_flow"),
        indicator=True,
    )
    joined_rows = int((joined["_merge"] == "both").sum())
    annotated_rows = int(len(annotated))
    flow_rows = int(len(flow))
    missing_rows = annotated_rows - joined_rows
    join_rate = joined_rows / annotated_rows if annotated_rows else math.nan

    annotated_min, annotated_max = timestamp_range(annotated)
    flow_min, flow_max = timestamp_range(flow)
    nearest = nearest_tolerance_diagnostics(annotated, flow)

    rows: list[dict[str, Any]] = []
    for column in COST_COLUMNS:
        available = column in flow.columns
        joined_non_null = int(joined[column].notna().sum()) if available and column in joined.columns else 0
        rows.append(
            {
                "symbol": SYMBOL,
                "venue": PRIMARY_VENUE,
                "annotated_path": str(ANNOTATED_PATH),
                "flow_path": str(FLOW_PATH),
                "annotated_rows": annotated_rows,
                "flow_rows": flow_rows,
                "joined_rows": joined_rows,
                "join_rate": join_rate,
                "missing_rows": missing_rows,
                "annotated_timestamp_min": annotated_min,
                "annotated_timestamp_max": annotated_max,
                "flow_timestamp_min": flow_min,
                "flow_timestamp_max": flow_max,
                "cost_column": column,
                "cost_column_available": available,
                "joined_non_null_rows": joined_non_null,
                "joined_non_null_rate": joined_non_null / joined_rows if joined_rows else math.nan,
                **nearest,
                "exact_join_only": True,
            }
        )

    report = pd.DataFrame(rows)
    text = render_text(report)
    return report, text


def render_text(report: pd.DataFrame) -> str:
    first = report.iloc[0]
    available = report[report["cost_column_available"].astype(bool)]["cost_column"].tolist()
    missing = report[~report["cost_column_available"].astype(bool)]["cost_column"].tolist()
    lines = [
        "Rawseq Annotated Flow Join Probe",
        "",
        f"Symbol: {SYMBOL}",
        f"Venue: {PRIMARY_VENUE}",
        f"Annotated rows: {int(first['annotated_rows'])}",
        f"Flow rows: {int(first['flow_rows'])}",
        f"Joined rows: {int(first['joined_rows'])}",
        f"Join rate: {finite_or_nan(first['join_rate']):.4f}",
        f"Missing rows: {int(first['missing_rows'])}",
        f"Annotated timestamp min/max: {first['annotated_timestamp_min']} / {first['annotated_timestamp_max']}",
        f"Flow timestamp min/max: {first['flow_timestamp_min']} / {first['flow_timestamp_max']}",
        "",
        "Nearest timestamp diagnostics, not used for output join:",
        f"  median_abs_ms: {finite_or_nan(first['nearest_median_abs_ms']):.0f}",
        f"  p95_abs_ms: {finite_or_nan(first['nearest_p95_abs_ms']):.0f}",
        f"  within_1000ms: {int(first['nearest_within_1000ms'])}",
        f"  within_5000ms: {int(first['nearest_within_5000ms'])}",
        f"  within_10000ms: {int(first['nearest_within_10000ms'])}",
        "",
        "Dynamic cost columns available:",
        "  " + (", ".join(available) if available else "none"),
        "Dynamic cost columns missing:",
        "  " + (", ".join(missing) if missing else "none"),
        "",
        f"CSV report: {OUTPUT_PATH}",
        f"Text report: {TEXT_OUTPUT_PATH}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    report, text = build_report()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)
    TEXT_OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
