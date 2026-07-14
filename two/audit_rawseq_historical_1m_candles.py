#!/usr/bin/env python3
"""Audit and canonicalize public one-minute candle sources for rawseq research.

Input is required via RAWSEQ_1M_SOURCE_PATH and may be a directory of Binance
monthly zip files or a single zip/csv. This script is report-only and never
fills gaps or writes model/champion/shadow artifacts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import (
    DEFAULT_OUTPUT_ROOT,
    SAFETY_FLAGS,
    audit_candles,
    canonical_column_contract,
    env_path,
    load_candles,
    now_stamp,
    resolve_source_files,
    source_manifest,
    stable_hash,
    write_csv,
    write_json,
)


def output_dir(root: Path) -> Path:
    explicit = os.getenv("RAWSEQ_1M_AUDIT_OUTPUT_DIR", "").strip()
    if explicit:
        return Path(explicit)
    return root / f"rawseq_1m_candle_audit_{now_stamp()}"


def main() -> int:
    source_path = env_path("RAWSEQ_1M_SOURCE_PATH", required=True)
    symbol = os.getenv("RAWSEQ_1M_SYMBOL", "SOLUSDT").strip() or "SOLUSDT"
    venue = os.getenv("RAWSEQ_1M_VENUE", "binance_public").strip() or "unknown"
    out_root = env_path("RAWSEQ_1M_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    max_rows = int(os.getenv("RAWSEQ_1M_AUDIT_MAX_ROWS", "0") or "0")
    out_dir = output_dir(out_root)
    out_dir.mkdir(parents=True, exist_ok=False)

    files = resolve_source_files(source_path, symbol)
    frame = load_candles(files, max_rows=max_rows)
    audit = audit_candles(frame, files, symbol=symbol, venue=venue)
    audit["audit_output_dir"] = str(out_dir)
    audit["rawseq_1m_source_path"] = str(source_path)
    audit["max_rows"] = max_rows
    audit["audit_sha256"] = stable_hash(audit)
    manifest = source_manifest(files)
    manifest.update(
        {
            "rawseq_1m_source_path": str(source_path),
            "symbol": symbol,
            "venue": venue,
            "max_rows_loaded_for_audit": max_rows,
            **SAFETY_FLAGS,
        }
    )
    contract = canonical_column_contract(symbol=symbol, venue=venue)
    contract.update(
        {
            "source_path": str(source_path),
            "source_file_count": len(files),
            "audit_status": audit["audit_status"],
            "first_timestamp": audit["first_timestamp"],
            "last_timestamp": audit["last_timestamp"],
            "total_rows": audit["total_rows"],
        }
    )
    contract["contract_sha256"] = stable_hash(contract)

    write_json(out_dir / "candle_audit.json", audit)
    write_csv(out_dir / "candle_audit.csv", [audit])
    write_json(out_dir / "source_manifest.json", manifest)
    write_json(out_dir / "canonical_column_contract.json", contract)
    lines = [
        "Rawseq historical 1m candle audit",
        f"Output: {out_dir}",
        f"Source: {source_path}",
        f"Symbol: {symbol}",
        f"Venue: {venue}",
        f"Files: {len(files)}",
        f"Rows loaded: {audit['total_rows']}",
        f"First timestamp: {audit['first_timestamp']}",
        f"Last timestamp: {audit['last_timestamp']}",
        f"Approx months covered: {audit['approximate_months_covered']:.3f}",
        f"Audit status: {audit['audit_status']}",
        f"Audit reasons: {', '.join(audit['audit_reasons']) if audit['audit_reasons'] else 'none'}",
        f"Duplicate timestamps: {audit['duplicate_timestamps']}",
        f"Missing one-minute intervals: {audit['missing_one_minute_intervals']}",
        f"Largest gap minutes: {audit['largest_timestamp_gap_minutes']}",
        f"OHLC violations: {audit['ohlc_consistency_violations']}",
        f"Source manifest sha256: {manifest['source_manifest_sha256']}",
        "",
        "Safety: paper_only=true, private_api=false, orders=false, promotion=false, champion_mutation=false.",
    ]
    (out_dir / "candle_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if audit["audit_status"] in {"PASS", "WARN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
