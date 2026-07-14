#!/usr/bin/env python3
"""Build a canonical rawseq training table and purged split manifest.

The table uses market-relative future-return labels. It is report/data prep
only: no model training, promotion, champion mutation, private API use, or
orders.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "realtime" / "kraken" / "SOLUSDT_10s_flow.csv"

SOURCE_PATH = Path(os.getenv("RAWSEQ_CANONICAL_SOURCE_PATH", str(DEFAULT_SOURCE))).expanduser()
if not SOURCE_PATH.is_absolute():
    SOURCE_PATH = PROJECT_ROOT / SOURCE_PATH
OUTPUT_ROOT = Path(
    os.getenv(
        "RAWSEQ_CANONICAL_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_canonical_tables"),
    )
).expanduser()
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper() or "SOLUSDT"
VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower() or "kraken"
INSTRUMENT = os.getenv("RAWSEQ_EXECUTION_INSTRUMENT", "inventory_spot_long_flat").strip().lower()
HORIZON_SECONDS = int(float(os.getenv("RAWSEQ_CANONICAL_HORIZON_SECONDS", "60")))
FEATURE_WINDOWS = [
    int(float(item.strip()))
    for item in os.getenv("RAWSEQ_CANONICAL_FEATURE_WINDOWS", "60,150,300").split(",")
    if item.strip()
]
MAX_ROWS_ENV = os.getenv("RAWSEQ_CANONICAL_MAX_ROWS", "").strip()
VALIDATION_FRACTION = float(os.getenv("RAWSEQ_CANONICAL_VALIDATION_FRACTION", "0.20"))
HOLDOUT_FRACTION = float(os.getenv("RAWSEQ_CANONICAL_HOLDOUT_FRACTION", "0.20"))
PURGE_ROWS_ENV = os.getenv("RAWSEQ_CANONICAL_PURGE_ROWS", "").strip()
EMBARGO_ROWS_ENV = os.getenv("RAWSEQ_CANONICAL_EMBARGO_ROWS", "").strip()
INCLUDE_MICROSTRUCTURE = os.getenv("RAWSEQ_CANONICAL_INCLUDE_MICROSTRUCTURE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
MICROSTRUCTURE_COLUMNS = [
    "spread_percent",
    "best_bid",
    "best_ask",
    "market_buy_volume_10s",
    "market_sell_volume_10s",
    "total_trade_volume_10s",
    "trade_count_10s",
    "market_pressure_10s",
    "bid_depth_10bps",
    "ask_depth_10bps",
    "bid_depth_25bps",
    "ask_depth_25bps",
    "order_book_imbalance_10bps",
    "order_book_imbalance_25bps",
    "bid_depth_change_10bps",
    "ask_depth_change_10bps",
    "imbalance_change_10bps",
    "large_bid_wall_distance",
    "large_ask_wall_distance",
    "large_bid_wall_size",
    "large_ask_wall_size",
]


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def iso_from_ms(value: float) -> str:
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC).isoformat()
    except Exception:
        return ""


def infer_column(frame: pd.DataFrame, choices: list[str], label: str) -> str:
    for column in choices:
        if column in frame.columns:
            return column
    raise SystemExit(f"Could not find {label} column. Tried: {choices}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_source(path: Path) -> pd.DataFrame:
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
            "decision_timestamp": pd.to_numeric(frame[timestamp_col], errors="coerce"),
            "price": pd.to_numeric(frame[price_col], errors="coerce"),
        }
    )
    if INCLUDE_MICROSTRUCTURE:
        for column in MICROSTRUCTURE_COLUMNS:
            if column in frame.columns:
                out[column] = pd.to_numeric(frame[column], errors="coerce")
    out = out.dropna(subset=["decision_timestamp", "price"]).sort_values("decision_timestamp")
    out = out.drop_duplicates("decision_timestamp").reset_index(drop=True)
    out["decision_time_iso"] = out["decision_timestamp"].apply(iso_from_ms)
    return out


def estimate_bucket_seconds(frame: pd.DataFrame) -> float:
    diffs = frame["decision_timestamp"].diff().dropna().to_numpy(dtype=np.float64)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 10.0
    return float(np.median(diffs) / 1000.0)


def future_extreme(price: pd.Series, offset: int, kind: str) -> pd.Series:
    shifted = price.shift(-1)
    reversed_shifted = shifted.iloc[::-1]
    if kind == "max":
        rolled = reversed_shifted.rolling(offset, min_periods=offset).max()
    elif kind == "min":
        rolled = reversed_shifted.rolling(offset, min_periods=offset).min()
    else:
        raise ValueError(kind)
    return rolled.iloc[::-1].reset_index(drop=True)


def add_features(table: pd.DataFrame, windows: list[int]) -> tuple[pd.DataFrame, list[str]]:
    out = table.copy()
    price = pd.to_numeric(out["price"], errors="coerce")
    out["bucket_return_bps"] = 10_000.0 * np.log(price / price.shift(1))
    out["bucket_return_bps"] = out["bucket_return_bps"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_columns = ["bucket_return_bps"]
    micro_features = [
        column
        for column in MICROSTRUCTURE_COLUMNS
        if column in out.columns and pd.api.types.is_numeric_dtype(out[column])
    ]
    for column in micro_features:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[f"{column}_missing"] = out[column].isna()
        feature_columns.append(column)
    for window in windows:
        rolling_high = price.rolling(window, min_periods=window).max()
        rolling_low = price.rolling(window, min_periods=window).min()
        rolling_mean = price.rolling(window, min_periods=window).mean()
        names = {
            f"ma_distance_bps_fw{window}": 10_000.0 * np.log(price / rolling_mean),
            f"rolling_range_bps_fw{window}": 10_000.0 * np.log(rolling_high / rolling_low),
            f"rolling_volatility_bps_fw{window}": out["bucket_return_bps"].rolling(window, min_periods=window).std(ddof=0),
            f"distance_to_recent_high_bps_fw{window}": 10_000.0 * np.log(price / rolling_high),
            f"distance_to_recent_low_bps_fw{window}": 10_000.0 * np.log(price / rolling_low),
        }
        for name, values in names.items():
            out[name] = values.replace([np.inf, -np.inf], np.nan)
            out[f"{name}_missing"] = out[name].isna()
            feature_columns.append(name)
    out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], np.nan)
    return out, feature_columns


def add_labels(table: pd.DataFrame, horizon_offset_rows: int) -> pd.DataFrame:
    out = table.copy()
    price = pd.to_numeric(out["price"], errors="coerce")
    future_price = price.shift(-horizon_offset_rows)
    future_high = future_extreme(price, horizon_offset_rows, "max")
    future_low = future_extreme(price, horizon_offset_rows, "min")
    out["label_end_timestamp"] = out["decision_timestamp"].shift(-horizon_offset_rows)
    out["gross_future_return_bps"] = 10_000.0 * np.log(future_price / price)
    high_from_now = 10_000.0 * np.log(future_high / price)
    low_from_now = 10_000.0 * np.log(future_low / price)
    out["mfe_bps"] = np.maximum(high_from_now, 0.0)
    out["mae_bps"] = np.minimum(low_from_now, 0.0)
    out["future_market_return_bps_horizon"] = out["gross_future_return_bps"]
    return out.replace([np.inf, -np.inf], np.nan)


def assign_splits(table: pd.DataFrame, lookahead_rows: int) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    n = len(table)
    purge_rows = int(PURGE_ROWS_ENV) if PURGE_ROWS_ENV else lookahead_rows
    embargo_rows = int(EMBARGO_ROWS_ENV) if EMBARGO_ROWS_ENV else lookahead_rows
    holdout_start_raw = int(n * (1.0 - HOLDOUT_FRACTION))
    validation_start_raw = int(n * (1.0 - HOLDOUT_FRACTION - VALIDATION_FRACTION))
    validation_start_raw = max(1, min(validation_start_raw, n - 2))
    holdout_start_raw = max(validation_start_raw + 1, min(holdout_start_raw, n - 1))

    train_start = 0
    train_end = max(train_start, validation_start_raw - embargo_rows)
    validation_start = min(n, validation_start_raw + purge_rows)
    validation_end = max(validation_start, holdout_start_raw - embargo_rows)
    holdout_start = min(n, holdout_start_raw + purge_rows)
    holdout_end = n

    out = table.copy()
    out["split"] = "purge_embargo"
    out.loc[train_start: max(train_end - 1, train_start - 1), "split"] = "train"
    if validation_start < validation_end:
        out.loc[validation_start: validation_end - 1, "split"] = "validation"
    if holdout_start < holdout_end:
        out.loc[holdout_start: holdout_end - 1, "split"] = "untouched_holdout"

    split_rows = []
    for split_name in ["train", "validation", "untouched_holdout", "purge_embargo"]:
        subset = out[out["split"].eq(split_name)]
        split_rows.append(
            {
                "split": split_name,
                "rows": int(len(subset)),
                "start_timestamp": float(subset["decision_timestamp"].min()) if not subset.empty else math.nan,
                "end_timestamp": float(subset["decision_timestamp"].max()) if not subset.empty else math.nan,
                "start_iso": iso_from_ms(subset["decision_timestamp"].min()) if not subset.empty else "",
                "end_iso": iso_from_ms(subset["decision_timestamp"].max()) if not subset.empty else "",
            }
        )
    split_frame = pd.DataFrame(split_rows)
    split_counts = {row["split"]: row["rows"] for row in split_rows}
    split_timestamp_boundaries = {
        row["split"]: {
            "start_timestamp": row["start_timestamp"],
            "end_timestamp": row["end_timestamp"],
            "start_iso": row["start_iso"],
            "end_iso": row["end_iso"],
        }
        for row in split_rows
    }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": SYMBOL,
        "venue": VENUE,
        "instrument": INSTRUMENT,
        "source_path": str(SOURCE_PATH),
        "horizon_seconds": HORIZON_SECONDS,
        "maximum_label_lookahead_rows": lookahead_rows,
        "purge_rows": purge_rows,
        "embargo_rows": embargo_rows,
        "raw_boundaries": {
            "validation_start_row": validation_start_raw,
            "holdout_start_row": holdout_start_raw,
            "validation_start_timestamp": iso_from_ms(table["decision_timestamp"].iloc[validation_start_raw])
            if 0 <= validation_start_raw < n
            else "",
            "holdout_start_timestamp": iso_from_ms(table["decision_timestamp"].iloc[holdout_start_raw])
            if 0 <= holdout_start_raw < n
            else "",
        },
        "effective_boundaries": {
            "train_start_row": train_start,
            "train_end_row": train_end,
            "validation_start_row": validation_start,
            "validation_end_row": validation_end,
            "holdout_start_row": holdout_start,
            "holdout_end_row": holdout_end,
            "train_start_timestamp": iso_from_ms(table["decision_timestamp"].iloc[train_start])
            if train_start < n
            else "",
            "train_end_timestamp": iso_from_ms(table["decision_timestamp"].iloc[train_end - 1])
            if train_end > train_start
            else "",
            "validation_start_timestamp": iso_from_ms(table["decision_timestamp"].iloc[validation_start])
            if validation_start < validation_end
            else "",
            "validation_end_timestamp": iso_from_ms(table["decision_timestamp"].iloc[validation_end - 1])
            if validation_end > validation_start
            else "",
            "holdout_start_timestamp": iso_from_ms(table["decision_timestamp"].iloc[holdout_start])
            if holdout_start < holdout_end
            else "",
            "holdout_end_timestamp": iso_from_ms(table["decision_timestamp"].iloc[holdout_end - 1])
            if holdout_end > holdout_start
            else "",
        },
        "split_timestamp_boundaries": split_timestamp_boundaries,
        "effective_sample_counts": split_counts,
        "purged_boundary_rows": {
            "train_validation_boundary": purge_rows,
            "validation_holdout_boundary": purge_rows,
        },
        "embargo_boundary_rows": {
            "train_validation_boundary": embargo_rows,
            "validation_holdout_boundary": embargo_rows,
        },
        "model_selection_stage": "validation_selects_model_seed_threshold_policy",
        "holdout_usage": "untouched_holdout_used_once_after_freeze",
        "split_rows": split_rows,
    }
    return out, manifest, split_frame


def validate_table(table: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    unique_keys = not table.duplicated(["symbol", "venue", "decision_timestamp"]).any()
    sorted_chronological = table["decision_timestamp"].is_monotonic_increasing
    guards = {
        "feature_timestamp_max_lte_decision_timestamp": bool(
            (table["feature_timestamp_max"] <= table["decision_timestamp"]).all()
        ),
        "label_end_timestamp_gt_decision_timestamp": bool(
            (table["label_end_timestamp"] > table["decision_timestamp"]).all()
        ),
        "unique_symbol_venue_timestamp_keys": bool(unique_keys),
        "sorted_chronological_order": bool(sorted_chronological),
        "feature_columns": feature_columns,
        "feature_missing_indicator_columns": [f"{column}_missing" for column in feature_columns if f"{column}_missing" in table.columns],
    }
    return guards


def write_summary(path: Path, manifest: dict[str, Any], guards: dict[str, Any], table_rows: int) -> None:
    lines = [
        "Rawseq Canonical Training Table",
        "",
        f"Created at: {manifest['created_at']}",
        f"Source path: {SOURCE_PATH}",
        f"Symbol: {SYMBOL}",
        f"Venue: {VENUE}",
        f"Instrument: {INSTRUMENT}",
        f"Horizon seconds: {HORIZON_SECONDS}",
        f"Rows: {table_rows}",
        f"Feature schema hash: {manifest['feature_schema_hash']}",
        f"Source SHA256: {manifest['source_sha256']}",
        "",
        "Split manifest:",
        f"  maximum_label_lookahead_rows: {manifest['maximum_label_lookahead_rows']}",
        f"  purge_rows: {manifest['purge_rows']}",
        f"  embargo_rows: {manifest['embargo_rows']}",
        f"  effective_boundaries: {manifest['effective_boundaries']}",
        "",
        "Guards:",
    ]
    for key, value in guards.items():
        if isinstance(value, list):
            lines.append(f"  {key}: {len(value)} columns")
        else:
            lines.append(f"  {key}: {value}")
    lines += [
        "",
        "Safety: paper_only=true training=false promotion=false champion_mutation=false orders=false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    source_hash = file_sha256(SOURCE_PATH)
    base = load_source(SOURCE_PATH)
    bucket_s = estimate_bucket_seconds(base)
    horizon_offset_rows = max(1, int(math.ceil(HORIZON_SECONDS / max(bucket_s, 1e-9))))
    table, feature_columns = add_features(base, FEATURE_WINDOWS)
    table = add_labels(table, horizon_offset_rows)
    table["symbol"] = SYMBOL
    table["venue"] = VENUE
    table["instrument"] = INSTRUMENT
    table["feature_timestamp_max"] = table["decision_timestamp"]
    feature_payload = {
        "features": feature_columns,
        "feature_windows": FEATURE_WINDOWS,
        "include_microstructure": INCLUDE_MICROSTRUCTURE,
        "microstructure_columns": [column for column in MICROSTRUCTURE_COLUMNS if column in table.columns],
        "target": f"future_market_return_bps_{HORIZON_SECONDS}s",
        "horizon_seconds": HORIZON_SECONDS,
        "instrument": INSTRUMENT,
    }
    feature_hash = schema_hash(feature_payload)
    table["feature_schema_hash"] = feature_hash
    table["source_sha256"] = source_hash
    table["source_max_timestamp"] = float(table["decision_timestamp"].max())
    table = table.dropna(subset=["label_end_timestamp", "gross_future_return_bps", "mfe_bps", "mae_bps"]).reset_index(drop=True)
    table, manifest, split_frame = assign_splits(table, horizon_offset_rows)
    guards = validate_table(table, feature_columns)
    manifest.update(
        {
            "source_sha256": source_hash,
            "source_max_timestamp": float(table["decision_timestamp"].max()) if not table.empty else math.nan,
            "source_max_iso": iso_from_ms(table["decision_timestamp"].max()) if not table.empty else "",
            "data_hashes": {
                "source_sha256": source_hash,
                "feature_schema_hash": feature_hash,
            },
            "estimated_bucket_seconds": bucket_s,
            "feature_schema": feature_payload,
            "feature_schema_hash": feature_hash,
            "guards": guards,
            "paper_only": True,
            "training": False,
            "promotion": False,
            "champion_mutation": False,
            "orders": False,
        }
    )

    out_dir = OUTPUT_ROOT / f"canonical_{SYMBOL}_{VENUE}_{HORIZON_SECONDS}s_{now_stamp()}_{feature_hash[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "canonical_training_table.csv"
    manifest_json_path = out_dir / "split_manifest.json"
    manifest_csv_path = out_dir / "split_manifest.csv"
    summary_path = out_dir / "canonical_training_table_summary.txt"
    table.to_csv(table_path, index=False)
    split_frame.to_csv(manifest_csv_path, index=False)
    manifest_json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(summary_path, manifest, guards, len(table))

    print("Rawseq canonical training table complete")
    print(f"Rows: {len(table)}")
    print(f"Output dir: {out_dir}")
    print(f"Table: {table_path}")
    print(f"Manifest JSON: {manifest_json_path}")
    print(f"Manifest CSV: {manifest_csv_path}")
    print(f"Summary: {summary_path}")
    print(split_frame.to_string(index=False))
    print("Safety: paper_only=true training=false promotion=false champion_mutation=false orders=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
