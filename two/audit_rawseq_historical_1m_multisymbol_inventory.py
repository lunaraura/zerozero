#!/usr/bin/env python3
"""Inventory all available Binance public one-minute candle symbols.

This is the source gate for the cross-asset 1m rawseq scout. It audits
continuity, OHLC validity, timestamp normalization, liquidity, and fixed
downside-target outcome sufficiency, but does not train or score any models.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import (
    DEFAULT_SOURCE_PATH,
    SAFETY_FLAGS,
    audit_candles,
    downside_event_targets,
    env_path,
    load_candles,
    now_stamp,
    resolve_source_files,
    source_manifest,
    stable_hash,
    timestamp_unit_counts,
    write_csv,
    write_json,
)

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_cross_asset_scout")
FIXED_HORIZON_MINUTES = 1
FIXED_VOL_WINDOW_MINUTES = 240


def discover_symbols(source_root: Path) -> list[str]:
    symbols = set()
    for path in source_root.glob("*-1m-*.zip"):
        match = re.match(r"^([A-Z0-9]+)-1m-\d{4}-\d{2}\.zip$", path.name)
        if match:
            symbols.add(match.group(1))
    for path in source_root.glob("*-1m-*.csv"):
        match = re.match(r"^([A-Z0-9]+)-1m-\d{4}-\d{2}\.csv$", path.name)
        if match:
            symbols.add(match.group(1))
    return sorted(symbols)


def month_is_complete(frame: pd.DataFrame, year_month: str) -> bool:
    if not year_month or frame.empty:
        return False
    month = pd.Period(year_month, freq="M")
    expected = month.days_in_month * 24 * 60
    return len(frame) >= expected


def median_daily_quote_volume(frame: pd.DataFrame) -> float:
    if "quote_asset_volume" not in frame.columns:
        return float("nan")
    tmp = frame[["timestamp", "quote_asset_volume"]].copy()
    tmp["day"] = tmp["timestamp"].dt.date
    daily = pd.to_numeric(tmp["quote_asset_volume"], errors="coerce").groupby(tmp["day"]).sum(min_count=1)
    return float(daily.median()) if daily.notna().any() else float("nan")


def target_sufficiency(frame: pd.DataFrame) -> dict[str, Any]:
    targets = downside_event_targets(
        frame,
        vol_window=FIXED_VOL_WINDOW_MINUTES,
        horizons=[FIXED_HORIZON_MINUTES],
    )
    col = f"downside_event_0p5vol_h{FIXED_HORIZON_MINUTES}m_fw{FIXED_VOL_WINDOW_MINUTES}"
    vals = pd.to_numeric(targets[col], errors="coerce").dropna()
    events = int((vals > 0.5).sum())
    nonevents = int((vals <= 0.5).sum())
    return {
        "fixed_target_rows": int(len(vals)),
        "fixed_target_events": events,
        "fixed_target_nonevents": nonevents,
        "fixed_target_event_prevalence": float(vals.mean()) if len(vals) else float("nan"),
        "sufficient_positive_and_negative_target_outcomes": events >= 100 and nonevents >= 100,
    }


def overlap_with_sol(row: dict[str, Any], sol_first_ms: float, sol_last_ms: float) -> dict[str, Any]:
    first = float(row.get("first_timestamp_ms", float("nan")))
    last = float(row.get("last_timestamp_ms", float("nan")))
    if not np.isfinite(first) or not np.isfinite(last) or not np.isfinite(sol_first_ms) or not np.isfinite(sol_last_ms):
        return {"overlap_with_SOLUSDT_start": "", "overlap_with_SOLUSDT_end": "", "overlap_with_SOLUSDT_rows_estimate": 0}
    start = max(first, sol_first_ms)
    end = min(last, sol_last_ms)
    rows = max(0, int((end - start) // 60_000) + 1) if end >= start else 0
    return {
        "overlap_with_SOLUSDT_start": pd.to_datetime(start, unit="ms", utc=True).isoformat() if rows else "",
        "overlap_with_SOLUSDT_end": pd.to_datetime(end, unit="ms", utc=True).isoformat() if rows else "",
        "overlap_with_SOLUSDT_rows_estimate": rows,
    }


def eligibility(row: dict[str, Any], min_complete_months: int, min_rows: int) -> tuple[bool, str]:
    reasons: list[str] = []
    if int(row.get("complete_months", 0)) < min_complete_months:
        reasons.append(f"complete_months<{min_complete_months}")
    if int(row.get("total_rows", 0)) < min_rows:
        reasons.append(f"rows<{min_rows}")
    if str(row.get("timestamp_unresolved", "")).lower() == "true":
        reasons.append("unresolved_timestamp_units")
    if int(row.get("ohlc_consistency_violations", 0)) > 0:
        reasons.append("ohlc_consistency_violations")
    if int(row.get("nonpositive_prices", 0)) > 0:
        reasons.append("nonpositive_prices")
    if int(row.get("duplicate_timestamps", 0)) > 0:
        reasons.append("duplicate_timestamps")
    if int(row.get("missing_one_minute_intervals", 0)) > max(10, int(row.get("total_rows", 0)) // 1000):
        reasons.append("severe_missing_data_concentration")
    if not bool(row.get("sufficient_positive_and_negative_target_outcomes", False)):
        reasons.append("insufficient_target_outcomes")
    return not reasons, ";".join(reasons)


def audit_symbol(source_root: Path, symbol: str, sol_bounds: tuple[float, float] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    files = resolve_source_files(source_root, symbol)
    manifest = source_manifest(files)
    frame = load_candles(files)
    audit = audit_candles(frame, files, symbol=symbol, venue="binance_public")
    file_months = [item.year_month for item in files if item.year_month]
    complete_months = sum(1 for item in files if month_is_complete(frame[frame["source_year_month"] == item.year_month], item.year_month))
    unit_counts = timestamp_unit_counts(frame["open_time"])
    fixed_target = target_sufficiency(frame)
    row = {
        "symbol": symbol,
        "monthly_files": len(files),
        "first_month": min(file_months) if file_months else "",
        "last_month": max(file_months) if file_months else "",
        "total_months": len(set(file_months)),
        "complete_months": complete_months,
        "total_rows": audit["total_rows"],
        "first_timestamp": audit["first_timestamp"],
        "last_timestamp": audit["last_timestamp"],
        "first_timestamp_ms": audit["first_timestamp_ms"],
        "last_timestamp_ms": audit["last_timestamp_ms"],
        "duplicate_timestamps": audit["duplicate_timestamps"],
        "missing_one_minute_intervals": audit["missing_one_minute_intervals"],
        "largest_gap_minutes": audit["largest_timestamp_gap_minutes"],
        "ohlc_consistency_violations": audit["ohlc_consistency_violations"],
        "ohlc_validity": audit["ohlc_consistency_violations"] == 0,
        "nonpositive_prices": audit["nonpositive_prices"],
        "volume_available": "volume" in frame.columns and pd.to_numeric(frame["volume"], errors="coerce").notna().any(),
        "quote_volume_available": "quote_asset_volume" in frame.columns and pd.to_numeric(frame["quote_asset_volume"], errors="coerce").notna().any(),
        "median_daily_quote_volume": median_daily_quote_volume(frame),
        "source_file_hashes_sha256": manifest["source_manifest_sha256"],
        "timestamp_units_encountered": ",".join(f"{k}:{v}" for k, v in unit_counts.items() if v),
        "mixed_millisecond_microsecond_normalization_required": unit_counts.get("milliseconds", 0) > 0 and unit_counts.get("microseconds", 0) > 0,
        "timestamp_unresolved": unit_counts.get("unknown", 0) > 0,
        **fixed_target,
        **SAFETY_FLAGS,
        "public_recorded_data_only": True,
        "frozen_kraken_candidate_reused": False,
        "future_data_accessed": False,
    }
    if sol_bounds:
        row.update(overlap_with_sol(row, sol_bounds[0], sol_bounds[1]))
    eligible, reason = eligibility(
        row,
        min_complete_months=int(os.getenv("RAWSEQ_1M_MIN_COMPLETE_MONTHS", "12")),
        min_rows=int(os.getenv("RAWSEQ_1M_MIN_VALID_ROWS", "300000")),
    )
    row["eligible_for_fixed_transfer_test"] = eligible
    row["exclusion_reason"] = reason
    return row, manifest


def main() -> int:
    source_root = env_path("RAWSEQ_1M_SOURCE_PATH", DEFAULT_SOURCE_PATH)
    output_root = env_path("RAWSEQ_1M_CROSS_ASSET_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    out_dir = Path(os.getenv("RAWSEQ_1M_MULTISYMBOL_INVENTORY_OUTPUT_DIR", "").strip() or output_root / f"rawseq_1m_multisymbol_inventory_{now_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=False)
    symbols = discover_symbols(source_root)
    if not symbols:
        raise SystemExit(f"No one-minute symbols found under {source_root}")
    sol_row: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    if "SOLUSDT" in symbols:
        sol_row, sol_manifest = audit_symbol(source_root, "SOLUSDT")
        manifests["SOLUSDT"] = sol_manifest
        sol_bounds = (float(sol_row["first_timestamp_ms"]), float(sol_row["last_timestamp_ms"]))
    else:
        sol_bounds = None
    for symbol in symbols:
        if symbol == "SOLUSDT" and sol_row is not None:
            row = dict(sol_row)
            row.update(overlap_with_sol(row, sol_bounds[0], sol_bounds[1]) if sol_bounds else {})
            eligible, reason = eligibility(
                row,
                min_complete_months=int(os.getenv("RAWSEQ_1M_MIN_COMPLETE_MONTHS", "12")),
                min_rows=int(os.getenv("RAWSEQ_1M_MIN_VALID_ROWS", "300000")),
            )
            row["eligible_for_fixed_transfer_test"] = eligible
            row["exclusion_reason"] = reason
            rows.append(row)
            continue
        row, manifest = audit_symbol(source_root, symbol, sol_bounds)
        rows.append(row)
        manifests[symbol] = manifest
    rows.sort(key=lambda r: r["symbol"])
    inventory = {
        "created_at": now_stamp(),
        "source_root": str(source_root),
        "symbol_count": len(rows),
        "eligible_symbol_count": sum(1 for row in rows if row["eligible_for_fixed_transfer_test"]),
        "symbols": [row["symbol"] for row in rows],
        "eligible_symbols": [row["symbol"] for row in rows if row["eligible_for_fixed_transfer_test"]],
        "safety": {
            **SAFETY_FLAGS,
            "public_recorded_data_only": True,
            "frozen_kraken_candidate_reused": False,
            "future_data_accessed": False,
        },
    }
    inventory["inventory_sha256"] = stable_hash({"rows": rows, "metadata": inventory})
    write_csv(out_dir / "multisymbol_inventory.csv", rows)
    write_json(out_dir / "multisymbol_inventory.json", inventory)
    write_json(out_dir / "multisymbol_source_manifest.json", manifests)
    lines = [
        "Rawseq 1m multisymbol inventory",
        f"Output: {out_dir}",
        f"Source root: {source_root}",
        f"Discovered symbols: {', '.join(row['symbol'] for row in rows)}",
        f"Eligible symbols: {', '.join(inventory['eligible_symbols']) if inventory['eligible_symbols'] else 'none'}",
        "",
        "Symbol coverage:",
    ]
    for row in rows:
        lines.append(
            f"- {row['symbol']}: months={row['total_months']} complete={row['complete_months']} "
            f"rows={row['total_rows']} first={row['first_month']} last={row['last_month']} "
            f"eligible={row['eligible_for_fixed_transfer_test']} reason={row['exclusion_reason'] or 'none'}"
        )
    lines.extend(
        [
            "",
            "Safety: inventory only; no model evaluation, no GPU, no private API, no orders, no promotion, no champion mutation.",
        ]
    )
    (out_dir / "multisymbol_inventory.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if inventory["eligible_symbol_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
