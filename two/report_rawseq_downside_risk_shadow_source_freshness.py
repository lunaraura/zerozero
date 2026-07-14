#!/usr/bin/env python3
"""Source-vs-feature-table freshness report for downside-risk future shadow.

The frozen shadow logger scores the frozen feature table. This report checks
whether the underlying public/recorded source file has advanced beyond that
feature table enough to justify rebuilding features for continued prospective
logging.

Report-only: no training, no feature rebuild, no recalibration, no threshold
changes, no orders, no private API, no champion mutation, no promotion.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.freeze_rawseq_low_path_ridge_research_candidate import stable_hash
from scripts.tiny.run_rawseq_downside_risk_future_paper_shadow import DEFAULT_FEATURE_TABLE

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_shadow_source_freshness"
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "realtime" / "kraken" / "SOLUSDT_10s_flow.csv"
H480_SECONDS = 480


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return resolve_path(raw) if raw else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def timestamp_bounds(path: Path, timestamp_column: str) -> tuple[int, float, float]:
    if not path.exists() or not path.is_file():
        return 0, math.nan, math.nan
    frame = pd.read_csv(path, usecols=[timestamp_column])
    values = pd.to_numeric(frame[timestamp_column], errors="coerce")
    finite = values.dropna()
    if finite.empty:
        return len(frame), math.nan, math.nan
    return len(frame), float(finite.min()), float(finite.max())


def classify_source_freshness(source_lag_ms: float, h480_ms: float = H480_SECONDS * 1000.0) -> tuple[str, str]:
    if not math.isfinite(source_lag_ms):
        return "unknown", "timestamp bounds unavailable"
    if source_lag_ms <= 0:
        return "feature_table_current_or_ahead", "source has not advanced beyond feature table"
    if source_lag_ms < h480_ms:
        return "source_ahead_waiting_for_h480_horizon", f"source ahead by {source_lag_ms}ms but less than h480 horizon"
    return "feature_table_refresh_recommended", f"source ahead by {source_lag_ms}ms, at least one h480 horizon"


def refresh_recommendation_fields(
    status: str,
    source_lag_ms: float,
    source_rows_after_feature_max: int,
) -> dict[str, Any]:
    label_maturity_refresh_recommended = status == "feature_table_refresh_recommended"
    prediction_logging_refresh_recommended = (
        math.isfinite(source_lag_ms)
        and source_lag_ms > 0
        and source_rows_after_feature_max > 0
    )
    if prediction_logging_refresh_recommended and not label_maturity_refresh_recommended:
        prediction_logging_reason = "source advanced but h480 labels are not mature; refreshing now can log predictions before labels exist"
    elif prediction_logging_refresh_recommended:
        prediction_logging_reason = "source advanced; refresh can log new predictions, but some catch-up rows may already be backfill"
    else:
        prediction_logging_reason = "source has not advanced beyond the feature table"
    return {
        "label_maturity_refresh_recommended": label_maturity_refresh_recommended,
        "prediction_logging_refresh_recommended": prediction_logging_refresh_recommended,
        "prediction_logging_refresh_reason": prediction_logging_reason,
        "refresh_recommended": prediction_logging_refresh_recommended or label_maturity_refresh_recommended,
    }


def main() -> int:
    feature_table = env_path("RAWSEQ_DOWNSIDE_SHADOW_FEATURE_TABLE", DEFAULT_FEATURE_TABLE)
    output_root = env_path("RAWSEQ_DOWNSIDE_SOURCE_FRESHNESS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    feature_manifest = feature_table.parent / "feature_manifest.json"
    split_manifest = feature_table.parent / "split_manifest.json"
    manifest = read_json(feature_manifest)
    split = read_json(split_manifest)
    source_path = resolve_path(os.getenv("RAWSEQ_DOWNSIDE_SHADOW_SOURCE_PATH", "").strip() or manifest.get("source_path", "") or DEFAULT_SOURCE_PATH)
    source_columns = manifest.get("source_columns", {})
    source_timestamp_col = os.getenv("RAWSEQ_DOWNSIDE_SHADOW_SOURCE_TIMESTAMP_COLUMN", "").strip() or source_columns.get("timestamp_column", "timestamp")

    feature_rows, feature_min, feature_max = timestamp_bounds(feature_table, "decision_timestamp")
    source_rows, source_min, source_max = timestamp_bounds(source_path, source_timestamp_col)
    source_lag_ms = source_max - feature_max if math.isfinite(source_max) and math.isfinite(feature_max) else math.nan
    status, reason = classify_source_freshness(source_lag_ms)
    source_rows_after_feature_max = 0
    if math.isfinite(feature_max):
        source_ts = pd.to_numeric(pd.read_csv(source_path, usecols=[source_timestamp_col])[source_timestamp_col], errors="coerce")
        source_rows_after_feature_max = int((source_ts > feature_max).sum())
    refresh_fields = refresh_recommendation_fields(status, source_lag_ms, source_rows_after_feature_max)

    out_dir = output_root / f"rawseq_downside_risk_shadow_source_freshness_{now_stamp()}"
    row = {
        "status": status,
        "reason": reason,
        "feature_table": str(feature_table),
        "feature_rows": feature_rows,
        "feature_min_timestamp": feature_min,
        "feature_max_timestamp": feature_max,
        "feature_manifest": str(feature_manifest),
        "feature_manifest_source_sha256": manifest.get("source_sha256", ""),
        "split_manifest_source_sha256": split.get("source_sha256", ""),
        "source_path": str(source_path),
        "source_timestamp_column": source_timestamp_col,
        "source_rows": source_rows,
        "source_min_timestamp": source_min,
        "source_max_timestamp": source_max,
        "source_lag_ms": source_lag_ms,
        "source_lag_seconds": source_lag_ms / 1000.0 if math.isfinite(source_lag_ms) else math.nan,
        "source_lag_hours": source_lag_ms / 3_600_000.0 if math.isfinite(source_lag_ms) else math.nan,
        "source_rows_after_feature_max": source_rows_after_feature_max,
        "h480_ms": H480_SECONDS * 1000,
        **refresh_fields,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    row["source_freshness_report_sha256"] = stable_hash(row)
    write_json(out_dir / "rawseq_downside_risk_shadow_source_freshness.json", row)
    write_csv(out_dir / "rawseq_downside_risk_shadow_source_freshness.csv", [row])
    lines = [
        "Rawseq downside-risk shadow source freshness",
        f"Output: {out_dir}",
        f"Status: {status}",
        f"Reason: {reason}",
        f"Feature table max timestamp: {feature_max}",
        f"Source max timestamp: {source_max}",
        f"Source lag hours: {row['source_lag_hours']}",
        f"Source rows after feature max: {source_rows_after_feature_max}",
        f"Prediction logging refresh recommended: {row['prediction_logging_refresh_recommended']}",
        f"Prediction logging reason: {row['prediction_logging_refresh_reason']}",
        f"Label maturity refresh recommended: {row['label_maturity_refresh_recommended']}",
        f"Any refresh recommended: {row['refresh_recommended']}",
        "Safety: report-only, no feature rebuild, no training, no orders, no promotion.",
    ]
    (out_dir / "rawseq_downside_risk_shadow_source_freshness.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
