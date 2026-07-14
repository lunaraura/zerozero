#!/usr/bin/env python3
"""Inventory rawseq source CSV columns without loading full source files."""

from __future__ import annotations

import csv
import hashlib
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = [
    PROJECT_ROOT / "data" / "realtime" / "kraken" / "SOLUSDT_10s_flow.csv",
    PROJECT_ROOT / "data" / "realtime" / "kraken" / "BTCUSDT_10s_flow.csv",
    PROJECT_ROOT / "data" / "realtime" / "kraken" / "ETHUSDT_10s_flow.csv",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "research" / "rawseq_source_column_inventory"


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_sources(text: str | None = None) -> list[Path]:
    raw = text if text is not None else os.getenv("RAWSEQ_SOURCE_INVENTORY_PATHS", "")
    if raw.strip():
        return [resolve_path(item.strip()) for item in raw.split(";") if item.strip()]
    return DEFAULT_SOURCES


def symbol_from_path(path: Path) -> str:
    name = path.name.upper()
    for suffix in ["_10S_FLOW.CSV", "_ALL_FLOW_COMBINED.CSV", ".CSV"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return ""


def header_sha256(path: Path) -> tuple[list[str], str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    payload = ",".join(header).encode("utf-8")
    return header, hashlib.sha256(payload).hexdigest()


def normalize_column(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")


def infer_semantic_role(name: str) -> str:
    n = normalize_column(name)
    if n in {"symbol", "venue", "exchange"}:
        return "identity"
    if "timestamp" in n or n in {"time", "date", "datetime"}:
        return "timestamp"
    if n in {"price", "mid_price", "close", "last", "open", "high", "low"} or n.endswith("_price"):
        return "price"
    if "return" in n:
        return "return"
    if "spread" in n:
        return "spread"
    if "depth" in n or "book" in n:
        return "depth"
    if "imbalance" in n:
        return "imbalance"
    if "volume" in n:
        return "volume"
    if "trade_count" in n or n == "trades":
        return "trade_count"
    if "pressure" in n:
        return "pressure"
    if "volatility" in n or n == "vol":
        return "volatility"
    if "range" in n:
        return "range"
    if n.startswith("btc") or n.startswith("eth") or "cross" in n:
        return "cross_market"
    if "flag" in n or "quality" in n or "valid" in n:
        return "quality_flag"
    return "unknown"


def inferred_unit(name: str, role: str) -> str:
    n = normalize_column(name)
    if n.endswith("_bps") or "bps" in n:
        return "bps"
    if "percent" in n or n.endswith("_pct"):
        return "percent"
    if role == "timestamp":
        return "milliseconds_or_seconds"
    if role == "price":
        return "quote_currency"
    if role in {"volume", "depth"}:
        return "base_or_quote_units"
    if role == "trade_count":
        return "count"
    return "unknown"


def candidate_family(role: str) -> str:
    return {
        "price": "raw",
        "return": "raw",
        "spread": "liquidity",
        "depth": "order_book",
        "imbalance": "order_book",
        "trade_flow": "trade_flow",
        "volume": "volume",
        "trade_count": "volume",
        "pressure": "trade_flow",
        "volatility": "volatility",
        "range": "volatility",
        "cross_market": "cross_market",
        "quality_flag": "quality",
    }.get(role, "unknown")


def timestamp_unit(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "unknown"
    med = float(values.median())
    if med > 1e12:
        return "milliseconds"
    if med > 1e9:
        return "seconds"
    return "unknown"


def cadence_stats(sample: pd.DataFrame, header: list[str], declared_bucket_seconds: float) -> dict[str, Any]:
    ts_col = next((c for c in header if infer_semantic_role(c) == "timestamp"), "")
    result = {
        "timestamp_column": ts_col,
        "timestamp_unit": "unknown",
        "observed_bucket_seconds_median": math.nan,
        "observed_bucket_seconds_mode": math.nan,
        "declared_bucket_seconds": declared_bucket_seconds,
        "cadence_match": "",
        "cadence_warning": "",
        "duplicate_timestamp_count": 0,
        "negative_timestamp_count": 0,
    }
    if not ts_col or ts_col not in sample.columns:
        result["cadence_warning"] = "no_timestamp_column"
        return result
    values = pd.to_numeric(sample[ts_col], errors="coerce").dropna()
    result["timestamp_unit"] = timestamp_unit(values)
    result["duplicate_timestamp_count"] = int(values.duplicated().sum())
    result["negative_timestamp_count"] = int((values < 0).sum())
    diffs = values.sort_values().diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        result["cadence_warning"] = "no_positive_timestamp_differences"
        return result
    divisor = 1000.0 if result["timestamp_unit"] == "milliseconds" else 1.0
    seconds = diffs / divisor
    result["observed_bucket_seconds_median"] = float(seconds.median())
    try:
        result["observed_bucket_seconds_mode"] = float(seconds.mode().iloc[0])
    except Exception:
        result["observed_bucket_seconds_mode"] = math.nan
    if declared_bucket_seconds > 0 and math.isfinite(result["observed_bucket_seconds_median"]):
        result["cadence_match"] = abs(result["observed_bucket_seconds_median"] - declared_bucket_seconds) <= max(1.0, declared_bucket_seconds * 0.25)
    if result["cadence_match"] is False:
        result["cadence_warning"] = "observed_cadence_differs_from_declared"
    return result


def sample_column_stats(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    non_null_fraction = float(series.notna().mean()) if len(series) else math.nan
    finite_fraction = float(np.isfinite(numeric).mean()) if len(numeric) else math.nan
    return {
        "sample_dtype": str(series.dtype),
        "nullable": bool(series.isna().any()),
        "sample_non_null_fraction": non_null_fraction,
        "sample_finite_fraction": finite_fraction,
        "sample_min": float(finite.min()) if len(finite) else math.nan,
        "sample_max": float(finite.max()) if len(finite) else math.nan,
        "sample_mean": float(finite.mean()) if len(finite) else math.nan,
        "sample_std": float(finite.std(ddof=0)) if len(finite) else math.nan,
    }


def inventory_source(path: Path, source_role: str, declared_bucket_seconds: float, sample_rows: int) -> list[dict[str, Any]]:
    if not path.exists():
        return [
            {
                "source_path": str(path),
                "source_basename": path.name,
                "source_role": source_role,
                "symbol": symbol_from_path(path),
                "venue": path.parent.name,
                "status": "MISSING",
                "warning": "source_file_missing",
            }
        ]
    header, header_hash = header_sha256(path)
    sample = pd.read_csv(path, nrows=sample_rows)
    cadence = cadence_stats(sample, header, declared_bucket_seconds)
    rows: list[dict[str, Any]] = []
    for index, column in enumerate(header):
        role = infer_semantic_role(column)
        stats = sample_column_stats(sample[column]) if column in sample.columns else {}
        rows.append(
            {
                "source_path": str(path),
                "source_basename": path.name,
                "source_role": source_role,
                "symbol": symbol_from_path(path),
                "venue": path.parent.name,
                "file_bytes": path.stat().st_size,
                "file_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                "header_sha256": header_hash,
                "column_index": index,
                "column_name": column,
                "normalized_column_name": normalize_column(column),
                **stats,
                "declared_unit": "unknown",
                "inferred_unit": inferred_unit(column, role),
                "semantic_role": role,
                "timestamp_semantics": "event_or_bucket_timestamp" if role == "timestamp" else "",
                "native_or_derived": "declared" if role in {"timestamp", "price", "volume", "trade_count"} else "inferred",
                "candidate_feature_family": candidate_family(role),
                "known_aliases": "",
                "used_by_current_pipeline": role in {"timestamp", "price", "volume", "trade_count", "spread", "depth", "imbalance", "pressure"},
                "implementation_references": "scripts/tiny/run_rawseq_multi_horizon_indicator_pipeline.py;scripts/tiny_price_rawseq_path_v1.py",
                "status": "PASS" if role != "unknown" else "WARN",
                "warning": "" if role != "unknown" else "unknown_semantic_role",
                "notes": "confidence=inferred" if role != "unknown" else "confidence=unknown",
                **cadence,
            }
        )
    return rows


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "source_column_inventory.csv"
    txt_path = out_dir / "source_column_inventory.txt"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    summary = [
        "Rawseq source column inventory",
        f"Created at: {datetime.now(UTC).isoformat()}",
        f"Rows: {len(frame)}",
        f"Sources: {frame['source_path'].nunique() if not frame.empty and 'source_path' in frame else 0}",
        "",
        "Statuses:",
    ]
    if not frame.empty and "status" in frame:
        for status, count in frame["status"].value_counts(dropna=False).items():
            summary.append(f"  {status}: {count}")
    summary.extend(["", "Safety: report_only=true; no training; no orders; no promotion; no champion mutation."])
    txt_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return csv_path, txt_path


def main() -> int:
    sample_rows = int(float(os.getenv("RAWSEQ_SOURCE_INVENTORY_SAMPLE_ROWS", "2000")))
    declared_bucket_seconds = float(os.getenv("RAWSEQ_SOURCE_INVENTORY_DECLARED_BUCKET_SECONDS", "10"))
    output_root = resolve_path(os.getenv("RAWSEQ_SOURCE_INVENTORY_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    out_dir = output_root / f"rawseq_source_column_inventory_{now_stamp()}"
    rows: list[dict[str, Any]] = []
    for source in parse_sources():
        role = "primary" if "SOLUSDT" in source.name.upper() else "cross_market_context"
        rows.extend(inventory_source(source, role, declared_bucket_seconds, sample_rows))
    csv_path, txt_path = write_outputs(rows, out_dir)
    print("Rawseq source column inventory complete")
    print(f"CSV: {csv_path}")
    print(f"TXT: {txt_path}")
    print("Safety: no training. No orders. No promotion. No champion mutation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
