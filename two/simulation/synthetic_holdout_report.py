#!/usr/bin/env python3
"""Summarize synthetic holdout behavior by scenario and hidden event type."""

from __future__ import annotations

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
HOLDOUT_VENUE = os.getenv("SYNTHETIC_HOLDOUT_VENUE", "sim_holdout")
INPUT_PATH = Path(
    os.getenv(
        "SYNTHETIC_HOLDOUT_INPUT_PATH",
        tiny_paths.report_path(SYMBOL, HOLDOUT_VENUE, "tiny_price_training_rows", "csv"),
    )
)
OUTPUT_PATH = Path(
    os.getenv(
        "SYNTHETIC_HOLDOUT_REPORT_PATH",
        tiny_paths.report_path(SYMBOL, HOLDOUT_VENUE, "synthetic_holdout_report", "csv"),
    )
)
JSON_OUTPUT_PATH = OUTPUT_PATH.with_suffix(".json")

if not INPUT_PATH.is_absolute():
    INPUT_PATH = ROOT / INPUT_PATH
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = ROOT / OUTPUT_PATH


def read_rows() -> pd.DataFrame:
    try:
        return tiny_io.read_csv_required(INPUT_PATH, "synthetic holdout rows", chunksize=100_000)
    except tiny_io.TinyIOError:
        return pd.DataFrame()


def numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def signed_prediction(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    prediction_column = first_existing(
        frame,
        [
            "predicted_direction",
            "pred_direction",
            "prediction_direction",
            "raw_prediction_direction",
        ],
    )
    if prediction_column is None:
        return pd.Series(0.0, index=frame.index, dtype="float64"), "none"
    return np.sign(numeric(frame, prediction_column, 0.0)).astype(float), prediction_column


def realized_return(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    column = first_existing(
        frame,
        [
            "realized_return_bps",
            "target_return_bps_30s",
            "target_next_mid_delta_bps_30s",
            "target_next_mid_delta_bps_1s",
        ],
    )
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype="float64"), "none"
    return numeric(frame, column), column


def target_direction(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    column = first_existing(
        frame,
        [
            "actual_direction",
            "target_direction_30s",
            "target_next_mid_direction_30s",
            "target_move_before_adverse_net_aware_30s",
            "target_next_mid_direction_1s",
        ],
    )
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype="float64"), "none"
    return np.sign(numeric(frame, column)).astype(float), column


def group_report(
    frame: pd.DataFrame,
    group_column: str,
    pred: pd.Series,
    target: pd.Series,
    returns: pd.Series,
    performance_source: str,
) -> list[dict[str, object]]:
    rows = []
    if group_column not in frame.columns:
        return rows
    for value, group in frame.groupby(frame[group_column].fillna("unknown").astype(str), sort=True):
        idx = group.index
        group_pred = pred.loc[idx]
        group_target = target.loc[idx]
        group_returns = returns.loc[idx]
        active = group_pred != 0
        comparable = active & group_target.notna() & (group_target != 0)
        strategy_returns = group_pred * group_returns
        rows.append(
            {
                "symbol": SYMBOL,
                "group_by": group_column,
                "group_value": value or "unknown",
                "rows": int(len(group)),
                "active_rows": int(active.sum()),
                "sign_accuracy": float((group_pred[comparable] == group_target[comparable]).mean()) if comparable.any() else math.nan,
                "avg_strategy_return_bps": float(strategy_returns[active].mean()) if active.any() and group_returns.notna().any() else math.nan,
                "avg_realized_return_bps": float(group_returns.mean()) if group_returns.notna().any() else math.nan,
                "positive_target_rate": float((group_target > 0).mean()) if group_target.notna().any() else math.nan,
                "negative_target_rate": float((group_target < 0).mean()) if group_target.notna().any() else math.nan,
                "performance_source": performance_source,
                "paper_only": True,
            }
        )
    return rows


def main() -> None:
    frame = read_rows()
    if len(frame) == 0:
        report = pd.DataFrame(
            [
                {
                    "symbol": SYMBOL,
                    "group_by": "none",
                    "group_value": "no_rows",
                    "rows": 0,
                    "paper_only": True,
                }
            ]
        )
        prediction_column = target_column = return_column = "none"
        performance_source = "no_rows"
    else:
        pred, prediction_column = signed_prediction(frame)
        target, target_column = target_direction(frame)
        returns, return_column = realized_return(frame)
        performance_source = "prediction_columns" if prediction_column != "none" else "target_return_diagnostics_only"
        rows = []
        for group_column in ["source_scenario", "hidden_active_event_type"]:
            rows.extend(group_report(frame, group_column, pred, target, returns, performance_source))
        report = pd.DataFrame(rows)

    tiny_io.safe_write_csv_atomic(report, OUTPUT_PATH)
    tiny_io.safe_write_json_atomic(
        {
            "symbol": SYMBOL,
            "input_path": str(INPUT_PATH),
            "output_path": str(OUTPUT_PATH),
            "rows": int(len(frame)),
            "prediction_column": prediction_column,
            "target_column": target_column,
            "return_column": return_column,
            "performance_source": performance_source,
            "group_columns": ["source_scenario", "hidden_active_event_type"],
            "paper_only": True,
        },
        JSON_OUTPUT_PATH,
    )
    print("Synthetic holdout report")
    print(f"Rows: {len(frame)} path={INPUT_PATH}")
    print(f"Prediction column: {prediction_column}")
    print(f"Target column: {target_column}")
    print(f"Return column: {return_column}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
