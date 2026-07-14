#!/usr/bin/env python3
"""Refresh the downside-risk shadow feature table without training.

This script materializes the same causal indicator feature bank used by the
frozen downside-risk candidate against the current public/recorded source file.
It writes a new immutable feature-table artifact for future-shadow scoring.

No model fitting, no baseline fitting, no recalibration, no threshold changes,
no orders, no private API, no champion mutation, no promotion.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.run_rawseq_downside_risk_future_paper_shadow import (
    DEFAULT_CANDIDATE_DIR,
    horizon_from_target_column,
)

DEFAULT_REFERENCE_INDICATOR_DIR = Path(
    r"F:\rsio\rawseq_target_tournament_coarse_1s_300k_retry\mh_indicator_SOLUSDT_kraken_20260711T145015Z_fba19c8d"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_shadow_feature_table_refresh"


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return resolve_path(raw) if raw else default


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def iso_from_ms(value: Any) -> str:
    numeric = safe_float(value)
    if not math.isfinite(numeric):
        return ""
    return datetime.fromtimestamp(numeric / 1000.0, tz=UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def set_pipeline_env(reference_manifest: dict[str, Any], source_path: Path, max_rows: str) -> None:
    os.environ["RAWSEQ_MH_SOURCE_PATH"] = str(source_path)
    os.environ["RAWSEQ_MH_FEATURE_WINDOWS"] = ",".join(str(int(x)) for x in reference_manifest.get("feature_windows", []))
    os.environ["RAWSEQ_MH_INCLUDE_FLOW_FEATURES"] = "true" if reference_manifest.get("include_flow_features") else "false"
    if max_rows:
        os.environ["RAWSEQ_MH_MAX_ROWS"] = max_rows
    for symbol, source in reference_manifest.get("cross_market_sources", {}).items():
        os.environ[f"RAWSEQ_MH_{symbol.upper()}_SOURCE_PATH"] = str(source)


def add_low_path_targets(table: pd.DataFrame, target_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scripts.tiny import run_rawseq_multi_horizon_indicator_pipeline as mh

    out = table.copy()
    price = pd.to_numeric(out["close"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for target in target_columns:
        horizon = horizon_from_target_column(target)
        if horizon <= 0:
            continue
        future_low = mh.future_extreme(price, horizon, "min")
        label_ts = f"label_end_timestamp_h{horizon}"
        values = mh.log_bps(future_low, price).clip(upper=0.0)
        out[target] = values
        out[label_ts] = out["decision_timestamp"].shift(-horizon)
        rows.append(
            {
                "target_column": target,
                "horizon_buckets": horizon,
                "label_end_timestamp_column": label_ts,
                "target_type": "market_relative_future_low_from_now_bps_path",
                "zero_inclusive": True,
            }
        )
    horizons = [horizon_from_target_column(col) for col in target_columns]
    max_horizon = max([h for h in horizons if h > 0], default=0)
    if max_horizon > 0:
        out["label_end_timestamp"] = out["decision_timestamp"].shift(-max_horizon)
    return out.replace([np.inf, -np.inf], np.nan), pd.DataFrame(rows)


def main() -> int:
    candidate_dir = env_path("RAWSEQ_DOWNSIDE_SHADOW_CANDIDATE_DIR", DEFAULT_CANDIDATE_DIR)
    reference_dir = env_path("RAWSEQ_DOWNSIDE_REFRESH_REFERENCE_INDICATOR_DIR", DEFAULT_REFERENCE_INDICATOR_DIR)
    output_root = env_path("RAWSEQ_DOWNSIDE_REFRESH_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    reference_manifest = read_json(reference_dir / "feature_manifest.json")
    source_path = env_path("RAWSEQ_DOWNSIDE_REFRESH_SOURCE_PATH", resolve_path(reference_manifest["source_path"]))
    max_rows = os.getenv("RAWSEQ_DOWNSIDE_REFRESH_MAX_ROWS", "300000").strip()

    contract = read_json(candidate_dir / "rawseq_downside_risk_cpu_candidate_contract.json")
    model_npz = np.load(contract["model_path"], allow_pickle=False)
    frozen_feature_columns = [str(x) for x in model_npz["feature_columns"]]
    frozen_target_columns = [str(x) for x in model_npz["target_columns"]]

    set_pipeline_env(reference_manifest, source_path, max_rows)
    mh = importlib.import_module("scripts.tiny.run_rawseq_multi_horizon_indicator_pipeline")

    source_hash = file_sha256(source_path)
    base, source_meta = mh.load_source(source_path)
    bucket_seconds = mh.estimate_bucket_seconds(base)
    table, feature_columns, feature_audit = mh.add_feature_bank(base, list(reference_manifest.get("feature_windows", [])))
    family_manifest = mh.feature_family_manifest(feature_columns)
    table, target_manifest_csv = add_low_path_targets(table, frozen_target_columns)
    table["symbol"] = reference_manifest.get("symbol", "SOLUSDT")
    table["venue"] = reference_manifest.get("venue", "kraken")
    table["instrument"] = f"{table['symbol'].iloc[0]}:{table['venue'].iloc[0]}" if not table.empty else ""
    table["feature_timestamp_max"] = table["decision_timestamp"]
    table["source_sha256"] = source_hash

    target_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "target_family": "market_relative_future_low_from_now_bps_path",
        "horizon_buckets": [horizon_from_target_column(col) for col in frozen_target_columns],
        "horizon_seconds": [horizon_from_target_column(col) * bucket_seconds for col in frozen_target_columns],
        "target_columns": frozen_target_columns,
        "output_dim": len(frozen_target_columns),
        "label_tail_rows_kept_for_true_forward_logging": True,
    }
    feature_manifest = mh.build_feature_manifest(feature_columns, source_meta, bucket_seconds, family_manifest)
    feature_manifest.update(
        {
            "reference_indicator_dir": str(reference_dir),
            "reference_feature_schema_hash": stable_hash(reference_manifest),
            "refresh_source_path": str(source_path),
            "refresh_source_sha256": source_hash,
            "refresh_max_rows": max_rows,
            "refresh_purpose": "future_shadow_feature_table_only",
            "training": False,
            "recalibration": False,
            "threshold_changes": False,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
        }
    )
    table["feature_schema_hash"] = stable_hash(feature_manifest)
    table["target_schema_hash"] = stable_hash(target_manifest)

    missing_frozen_features = [col for col in frozen_feature_columns if col not in table.columns]
    extra_feature_columns = [col for col in feature_columns if col not in frozen_feature_columns]
    schema_status = "PASS" if not missing_frozen_features else "MISSING_FROZEN_FEATURES"
    max_target_horizon = max([horizon_from_target_column(col) for col in frozen_target_columns], default=0)
    source_max = safe_float(table["decision_timestamp"].max()) if not table.empty else math.nan
    h480_labeled_cutoff = source_max - float(max_target_horizon * 1000)

    refresh_hash = stable_hash(
        {
            "candidate_contract_sha256": contract.get("contract_sha256"),
            "source_path": str(source_path),
            "source_sha256": source_hash,
            "feature_schema_hash": table["feature_schema_hash"].iloc[0] if not table.empty else "",
            "target_schema_hash": table["target_schema_hash"].iloc[0] if not table.empty else "",
            "max_rows": max_rows,
        }
    )[:12]
    run_dir = output_root / f"rawseq_downside_risk_shadow_feature_refresh_{now_stamp()}_{refresh_hash}"
    run_dir.mkdir(parents=True, exist_ok=False)

    table_path = run_dir / "multi_horizon_training_table.csv"
    table.to_csv(table_path, index=False)
    feature_audit.to_csv(run_dir / "feature_audit.csv", index=False)
    pd.DataFrame({"feature": feature_columns}).to_csv(run_dir / "feature_manifest.csv", index=False)
    family_manifest.to_csv(run_dir / "feature_family_manifest.csv", index=False)
    target_manifest_csv.to_csv(run_dir / "target_manifest.csv", index=False)
    write_json(run_dir / "feature_manifest.json", feature_manifest)
    write_json(run_dir / "target_manifest.json", target_manifest)

    refresh_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "feature_table": str(table_path),
        "candidate_dir": str(candidate_dir),
        "candidate_contract_sha256": contract.get("contract_sha256"),
        "reference_indicator_dir": str(reference_dir),
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "bucket_seconds": bucket_seconds,
        "rows": int(len(table)),
        "columns": int(len(table.columns)),
        "feature_count": int(len(feature_columns)),
        "frozen_feature_count": int(len(frozen_feature_columns)),
        "extra_feature_columns": int(len(extra_feature_columns)),
        "missing_frozen_features": missing_frozen_features,
        "schema_status": schema_status,
        "target_columns": frozen_target_columns,
        "source_min_timestamp": safe_float(table["decision_timestamp"].min()) if not table.empty else math.nan,
        "source_max_timestamp": source_max,
        "source_min_iso": iso_from_ms(table["decision_timestamp"].min()) if not table.empty else "",
        "source_max_iso": iso_from_ms(source_max),
        "max_target_horizon_buckets": max_target_horizon,
        "label_tail_rows_kept_for_true_forward_logging": True,
        "max_horizon_labeled_cutoff_timestamp": h480_labeled_cutoff,
        "max_horizon_labeled_cutoff_iso": iso_from_ms(h480_labeled_cutoff),
        "paper_only": True,
        "private_api": False,
        "training": False,
        "recalibration": False,
        "threshold_changes": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    refresh_manifest["refresh_manifest_sha256"] = stable_hash(refresh_manifest)
    write_json(run_dir / "rawseq_downside_risk_shadow_feature_refresh_manifest.json", refresh_manifest)

    lines = [
        "# Downside-Risk Shadow Feature Refresh",
        "",
        f"Run dir: {run_dir}",
        f"Feature table: {table_path}",
        f"Rows: {len(table)}",
        f"Source max: {refresh_manifest['source_max_iso']}",
        f"Schema status: {schema_status}",
        f"Missing frozen features: {len(missing_frozen_features)}",
        "",
        "Safety: paper_only=true, training=false, recalibration=false, threshold_changes=false, orders=false, promotion=false, champion_mutation=false.",
    ]
    (run_dir / "rawseq_downside_risk_shadow_feature_refresh_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Refresh output: {run_dir}")
    print(f"Feature table: {table_path}")
    print(f"Rows: {len(table)}")
    print(f"Source max: {refresh_manifest['source_max_iso']}")
    print(f"Schema status: {schema_status}")
    print(f"Missing frozen features: {len(missing_frozen_features)}")
    return 0 if schema_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
