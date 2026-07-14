#!/usr/bin/env python3
"""Build canonical rawseq sequence NPZs from locked feature bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ABLATION_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_feature_family_ablation"
DEFAULT_TARGET_TOURNAMENT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_target_lane_baseline_tournament"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_locked_bundle_sequence_datasets"
DEFAULT_INDICATOR_GLOB = "F:/rsio/rawseq_multi_horizon_indicator_full_contract_smoke/mh_indicator_*"


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def latest_dir(root_or_pattern: str | Path, child_glob: str | None = None) -> Path | None:
    if child_glob is not None:
        root = Path(root_or_pattern)
        paths = list(root.glob(child_glob)) if root.exists() else []
    else:
        pattern = str(root_or_pattern)
        if re.match(r"^[A-Za-z]:", pattern):
            import glob

            paths = [Path(p) for p in glob.glob(pattern)]
        else:
            paths = list(Path().glob(pattern))
    paths = [p for p in paths if p.exists() and p.is_dir()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def git_status_dirty() -> bool:
    try:
        return bool(subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=10).stdout.strip())
    except Exception:
        return True


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def column_hash(columns: list[str]) -> str:
    return stable_hash(list(columns))


def parse_int_list(text: str, default: list[int]) -> list[int]:
    if not text.strip():
        return default
    return [int(float(item.strip())) for item in text.split(",") if item.strip()]


def parse_csv_strings(text: str, default: list[str]) -> list[str]:
    raw = str(text or "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()] if raw else default


def log_bps(future: pd.Series | np.ndarray, now: pd.Series | np.ndarray) -> np.ndarray:
    future_arr = np.asarray(future, dtype=np.float64)
    now_arr = np.asarray(now, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10000.0 * np.log(future_arr / now_arr)
    out[~np.isfinite(out)] = np.nan
    return out


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


def horizon_from_target_column(column: str) -> int:
    match = re.search(r"h(\d+)$", str(column))
    if not match:
        raise ValueError(f"Could not infer horizon from target column: {column}")
    return int(match.group(1))


def horizon_buckets_for_target_columns(target_columns: list[str]) -> list[int]:
    return [horizon_from_target_column(column) for column in target_columns]


def selected_target_hash(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key != "selected_target_contract_hash"})


def expected_range_envelope_column_order(horizons: list[int]) -> list[str]:
    out: list[str] = []
    for horizon in sorted(set(horizons)):
        out.extend([f"future_range_high_bps_h{horizon}", f"future_range_low_bps_h{horizon}"])
    return out


def validate_selected_target_payload(payload: dict[str, Any]) -> dict[str, Any]:
    target_lane = str(payload.get("target_lane", ""))
    target_layout = str(payload.get("target_layout", ""))
    target_columns = [str(x) for x in payload.get("target_column_order", [])]
    if not target_columns:
        raise SystemExit("selected_target_manifest target_column_order is empty; refusing to infer target defaults.")
    derived_horizons = horizon_buckets_for_target_columns(target_columns)
    declared_horizons = [int(x) for x in payload.get("horizons", [])]
    declared_unique = [int(x) for x in payload.get("unique_horizon_buckets", declared_horizons)]
    declared_per_output = [int(x) for x in payload.get("target_horizon_buckets", derived_horizons)]
    declared_output_dim = int(payload.get("target_output_dim", len(target_columns)))
    if declared_per_output != derived_horizons:
        raise SystemExit(
            "selected_target_manifest horizon mismatch: "
            f"target_horizon_buckets={declared_per_output} target_column_suffixes={derived_horizons}"
        )
    if declared_unique and declared_unique != sorted(set(derived_horizons)):
        raise SystemExit(
            "selected_target_manifest unique horizon mismatch: "
            f"unique_horizon_buckets={declared_unique} derived={sorted(set(derived_horizons))}"
        )
    if declared_horizons and declared_horizons != sorted(set(derived_horizons)):
        raise SystemExit(
            "selected_target_manifest horizons mismatch: "
            f"horizons={declared_horizons} derived={sorted(set(derived_horizons))}"
        )
    if declared_output_dim != len(target_columns):
        raise SystemExit(
            "selected_target_manifest output dimension mismatch: "
            f"target_output_dim={declared_output_dim} columns={len(target_columns)}"
        )
    if target_lane == "future_range_envelope_path":
        expected_order = expected_range_envelope_column_order(sorted(set(derived_horizons)))
        if target_layout != "range_envelope_path":
            raise SystemExit(f"Range-envelope selected target has wrong layout: {target_layout}")
        if target_columns != expected_order:
            raise SystemExit(
                "Range-envelope selected target order mismatch: "
                f"expected={expected_order} actual={target_columns}"
            )
    if target_lane == "future_low_from_now_bps_path":
        expected_order = [f"future_range_low_bps_h{horizon}" for horizon in sorted(set(derived_horizons))]
        if target_layout != "scalar_path":
            raise SystemExit(f"Low-only selected target has wrong layout: {target_layout}")
        if target_columns != expected_order:
            raise SystemExit(
                "Low-only selected target order mismatch or high/range contamination: "
                f"expected={expected_order} actual={target_columns}"
            )
    declared_hash = str(payload.get("selected_target_contract_hash", "") or "")
    if declared_hash and declared_hash != selected_target_hash(payload):
        raise SystemExit(
            "selected_target_manifest selected_target_contract_hash mismatch: "
            f"declared={declared_hash} computed={selected_target_hash(payload)}"
        )
    return {
        "target_lane": target_lane,
        "target_layout": target_layout,
        "target_columns": target_columns,
        "target_horizon_buckets": derived_horizons,
        "selected_horizons": sorted(set(derived_horizons)),
        "target_output_dim": len(target_columns),
        "selected_target_contract_hash": selected_target_hash(payload),
    }


def label_timestamp_column_for_horizon(table: pd.DataFrame, horizon: int) -> str:
    existing = f"label_end_timestamp_h{horizon}"
    if existing in table.columns:
        return existing
    column = f"target_lane_label_end_timestamp_h{horizon}"
    if column not in table.columns:
        table[column] = pd.to_numeric(table["decision_timestamp"], errors="coerce").shift(-horizon)
    return column


def materialize_target_column(table: pd.DataFrame, column: str, target_lane: str) -> None:
    if column in table.columns:
        return
    price_col = "close" if "close" in table.columns else "price"
    if price_col not in table.columns:
        raise SystemExit("Training table must include close or price to materialize selected target lanes.")
    price = pd.to_numeric(table[price_col], errors="coerce")
    horizon = horizon_from_target_column(column)
    if target_lane == "coarse_return_vector" or column.startswith("coarse_return_bps_h"):
        table[column] = log_bps(price.shift(-horizon), price)
    elif target_lane == "future_high_from_now_bps_path" or column.startswith("future_high_from_now_bps_h") or column.startswith("future_range_high_bps_h"):
        table[column] = np.maximum(log_bps(future_extreme(price, horizon, "max"), price), 0.0)
    elif target_lane == "future_low_from_now_bps_path" or column.startswith("future_low_from_now_bps_h") or column.startswith("future_range_low_bps_h"):
        table[column] = np.minimum(log_bps(future_extreme(price, horizon, "min"), price), 0.0)
    else:
        raise SystemExit(f"Unsupported selected target lane/column: {target_lane} {column}")


def materialize_selected_targets(table: pd.DataFrame, target_lane: str, target_columns: list[str]) -> list[str]:
    label_columns = []
    for column in target_columns:
        materialize_target_column(table, column, target_lane)
        label_columns.append(label_timestamp_column_for_horizon(table, horizon_from_target_column(column)))
    return label_columns


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_manifest_path_from_env() -> Path | None:
    manifest_env = os.getenv("RAWSEQ_LOCKED_NPZ_SELECTED_TARGET_MANIFEST", "").strip()
    if manifest_env:
        return resolve_path(manifest_env)
    tournament_env = os.getenv("RAWSEQ_LOCKED_NPZ_TARGET_TOURNAMENT_DIR", "").strip()
    tournament_dir = resolve_path(tournament_env) if tournament_env else latest_dir(DEFAULT_TARGET_TOURNAMENT_ROOT, "rawseq_target_lane_baseline_tournament_*")
    if tournament_dir is None:
        return None
    path = tournament_dir / "selected_target_manifest.json"
    return path if path.exists() else None


def load_selected_target_manifest() -> tuple[Path | None, dict[str, Any] | None]:
    path = selected_manifest_path_from_env()
    if path is None or not path.exists():
        return None, None
    return path, load_json(path)


def load_bundle_from_sources(bundle_name: str, ablation_dir: Path | None, diag_dir: Path | None) -> tuple[Path, dict[str, Any]]:
    candidates = []
    if diag_dir is not None:
        candidates.append(diag_dir / f"feature_bundle_{bundle_name}.json")
    if ablation_dir is not None:
        candidates.append(ablation_dir / f"locked_feature_bundle_{bundle_name}.json")
    for path in candidates:
        if path.exists():
            return path, load_json(path)
    raise SystemExit(f"Could not find locked feature bundle JSON for {bundle_name}. Tried: {[str(p) for p in candidates]}")


def to_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


def fit_scaler(train_values: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.nanmean(train_values, axis=0)
    std = np.nanstd(train_values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}


def transform(values: np.ndarray, scaler: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    missing = ~np.isfinite(values)
    filled = np.where(np.isfinite(values), values, scaler["mean"])
    scaled = (filled - scaler["mean"]) / scaler["std"]
    scaled[~np.isfinite(scaled)] = 0.0
    return scaled.astype(np.float32), missing.astype(np.uint8)


def build_sequences(
    table: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    label_timestamp_columns: list[str],
    seq_len: int,
) -> dict[str, np.ndarray]:
    train_mask_rows = table["split"].astype(str).eq("train").to_numpy()
    raw_features = to_matrix(table, feature_columns)
    scaler = fit_scaler(raw_features[train_mask_rows])
    scaled_features, missing_features = transform(raw_features, scaler)
    targets = to_matrix(table, target_columns).astype(np.float32)
    target_scaler = fit_scaler(targets[train_mask_rows])
    timestamps = pd.to_numeric(table["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    label_end = to_matrix(table, label_timestamp_columns).astype(np.float64)
    splits = table["split"].astype(str).to_numpy()
    source_indices = np.arange(len(table), dtype=np.int64)
    x_rows = []
    missing_rows = []
    y_rows = []
    split_rows = []
    ts_rows = []
    label_ts_rows = []
    row_index_rows = []
    for idx in range(seq_len - 1, len(table)):
        start = idx - seq_len + 1
        if not np.isfinite(targets[idx]).all() or not np.isfinite(timestamps[idx]):
            continue
        x_rows.append(scaled_features[start : idx + 1])
        missing_rows.append(missing_features[start : idx + 1])
        y_rows.append(targets[idx])
        split_rows.append(splits[idx])
        ts_rows.append(timestamps[idx])
        label_ts_rows.append(label_end[idx])
        row_index_rows.append(source_indices[idx])
    return {
        "X": np.asarray(x_rows, dtype=np.float32),
        "y": np.asarray(y_rows, dtype=np.float32),
        "missing_mask": np.asarray(missing_rows, dtype=np.uint8),
        "splits": np.asarray(split_rows, dtype=str),
        "decision_timestamps": np.asarray(ts_rows, dtype=np.float64),
        "label_end_timestamps": np.asarray(label_ts_rows, dtype=np.float64),
        "source_row_indices": np.asarray(row_index_rows, dtype=np.int64),
        "row_indices": np.asarray(row_index_rows, dtype=np.int64),
        "feature_scaler_mean": scaler["mean"].astype(np.float32),
        "feature_scaler_std": scaler["std"].astype(np.float32),
        "target_scaler_mean": target_scaler["mean"].astype(np.float32),
        "target_scaler_std": target_scaler["std"].astype(np.float32),
    }


def split_counts(splits: np.ndarray) -> dict[str, int]:
    return {name: int(np.sum(splits.astype(str) == name)) for name in ["train", "validation", "untouched_holdout", "purge_embargo"]}


def limit_rows_per_split(table: pd.DataFrame, max_rows_per_split: int) -> pd.DataFrame:
    if max_rows_per_split <= 0:
        return table
    return table.groupby("split", sort=False, group_keys=False).head(max_rows_per_split).sort_index().copy()


def main() -> int:
    ablation_env = os.getenv("RAWSEQ_LOCKED_NPZ_ABLATION_DIR", "").strip()
    ablation_dir = resolve_path(ablation_env) if ablation_env else latest_dir(DEFAULT_ABLATION_ROOT, "rawseq_feature_family_ablation_*")
    indicator_env = os.getenv("RAWSEQ_LOCKED_NPZ_INDICATOR_RUN_DIR", "").strip()
    indicator_dir = resolve_path(indicator_env) if indicator_env else latest_dir(DEFAULT_INDICATOR_GLOB)
    if indicator_dir is None:
        raise SystemExit("Could not find indicator run directory.")
    output_root = resolve_path(os.getenv("RAWSEQ_LOCKED_NPZ_OUTPUT_DIR", str(DEFAULT_OUTPUT_ROOT)))
    out_dir = output_root / f"rawseq_locked_bundle_sequence_datasets_{now_stamp()}"
    dataset_dir = out_dir / "sequence_datasets"
    dataset_dir.mkdir(parents=True, exist_ok=False)
    seq_lens = parse_int_list(os.getenv("RAWSEQ_LOCKED_NPZ_SEQUENCE_LENS", ""), [60, 120, 240])
    requested_bundle_names = parse_csv_strings(os.getenv("RAWSEQ_LOCKED_NPZ_BUNDLES", ""), [])
    selected_manifest_path, selected_manifest = load_selected_target_manifest()
    selected_manifest_sha256 = file_sha256(selected_manifest_path) if selected_manifest_path else ""
    table = pd.read_csv(indicator_dir / "multi_horizon_training_table.csv")
    max_rows_per_split = int(float(os.getenv("RAWSEQ_LOCKED_NPZ_MAX_ROWS_PER_SPLIT", "0")))
    original_table_rows = int(len(table))
    table = limit_rows_per_split(table, max_rows_per_split)

    target_mode = "legacy_selected_horizons"
    selected_horizons: list[int] = []
    target_columns: list[str] = []
    label_timestamp_columns: list[str] = []
    target_horizon_buckets: list[int] = []
    target_lane = "coarse_return_vector"
    target_layout = "return_vector"
    target_manifest_payload: dict[str, Any] = {}
    selected_contract: dict[str, Any] = {}
    if selected_manifest is not None:
        target_mode = "selected_target_manifest"
        if selected_manifest.get("selection_status") != "selected" or not selected_manifest.get("selected_targets"):
            contract = {
                "created_at": now_iso(),
                "generator_path": "scripts/tiny/build_rawseq_locked_bundle_sequence_datasets.py",
                "git_head": git_head(),
                "git_status_dirty": git_status_dirty(),
                "selected_target_manifest_path": str(selected_manifest_path),
                "selection_status": selected_manifest.get("selection_status", "unknown"),
                "output_dir": str(out_dir),
                "dataset_count": 0,
                "stop_reason": "selected_target_manifest_has_no_selected_targets",
                "downstream_allowed": False,
                "paper_only": True,
                "orders": False,
                "promotion": False,
                "champion_mutation": False,
            }
            write_json(out_dir / "locked_bundle_sequence_dataset_contract.json", contract)
            pd.DataFrame([]).to_csv(out_dir / "sequence_dataset_manifest.csv", index=False)
            (out_dir / "locked_bundle_sequence_dataset_summary.txt").write_text(
                "# Rawseq Locked Bundle Sequence Datasets\n\n"
                "Selected target manifest abstained. No NPZ handoffs were built.\n\n"
                "Safety: no training; no Torch sequence model; no ensemble search; no freeze; no promotion; no champion mutation; no orders.\n",
                encoding="utf-8",
            )
            print("Selected target manifest abstained; no sequence datasets built.")
            print(f"Output: {out_dir}")
            return 0
        target_manifest_payload = selected_manifest["selected_targets"][0]
        selected_contract = validate_selected_target_payload(target_manifest_payload)
        target_lane = selected_contract["target_lane"]
        target_layout = selected_contract["target_layout"]
        target_columns = selected_contract["target_columns"]
        target_horizon_buckets = selected_contract["target_horizon_buckets"]
        selected_horizons = selected_contract["selected_horizons"]
        label_timestamp_columns = materialize_selected_targets(table, target_lane, target_columns)
        bundle_names = requested_bundle_names or list(dict.fromkeys(
            [str(target_manifest_payload.get("preferred_feature_bundle", ""))]
            + [str(x) for x in target_manifest_payload.get("qualifying_feature_bundles", [])]
            + ([str(target_manifest_payload.get("backup_bundle", ""))] if target_manifest_payload.get("backup_bundle") else [])
        ))
        bundle_names = [name for name in bundle_names if name]
        diag_dir_text = ""
        if selected_manifest_path is not None:
            contract_path = selected_manifest_path.parent / "target_lane_tournament_contract.json"
            if contract_path.exists():
                diag_dir_text = str(load_json(contract_path).get("diagnostics_dir", ""))
        diag_dir = resolve_path(diag_dir_text) if diag_dir_text else None
    else:
        if ablation_dir is None:
            raise SystemExit("Could not find rawseq feature-family ablation directory and no selected_target_manifest.json was provided.")
        selected = json.loads((ablation_dir / "selected_horizons.json").read_text(encoding="utf-8"))
        selected_horizons = [int(x) for x in selected.get("selected_horizons", [])]
        if not selected_horizons:
            raise SystemExit("No selected horizons found in ablation output.")
        target_columns = [f"future_return_bps_h{h}" for h in selected_horizons]
        target_horizon_buckets = horizon_buckets_for_target_columns(target_columns)
        label_timestamp_columns = [f"label_end_timestamp_h{h}" for h in selected_horizons]
        bundle_names = requested_bundle_names or parse_csv_strings(os.getenv("RAWSEQ_LOCKED_NPZ_BUNDLES", ""), ["minimal_core", "balanced_research", "full_registered"])
        diag_dir = None
    for column in [*target_columns, *label_timestamp_columns, "decision_timestamp", "split"]:
        if column not in table.columns:
            raise SystemExit(f"Missing required column {column} in {indicator_dir / 'multi_horizon_training_table.csv'}")

    manifest_rows = []
    bundle_rows = []
    for bundle_name in bundle_names:
        bundle_path, bundle = load_bundle_from_sources(bundle_name, ablation_dir, diag_dir)
        feature_columns = [str(x) for x in bundle["ordered_feature_columns"]]
        missing = [feature for feature in feature_columns if feature not in table.columns]
        if missing:
            raise SystemExit(f"Bundle {bundle_name} has features missing from training table: {missing[:5]}")
        bundle_rows.append(
            {
                "bundle_name": bundle_name,
                "feature_count": len(feature_columns),
                "feature_columns_sha256": column_hash(feature_columns),
                "source_bundle_path": str(bundle_path),
                "untouched_holdout_used_for_selection": False,
            }
        )
        for seq_len in seq_lens:
            arrays = build_sequences(table, feature_columns, target_columns, label_timestamp_columns, seq_len)
            contract = {
                "bundle_name": bundle_name,
                "seq_len": seq_len,
                "target_mode": target_mode,
                "target_lane": target_lane,
                "target_layout": target_layout,
                "target_horizons": selected_horizons,
                "unique_target_horizons": selected_horizons,
                "target_horizon_buckets": target_horizon_buckets,
                "target_column_order": target_columns,
                "target_output_dim": len(target_columns),
                "feature_columns_sha256": column_hash(feature_columns),
                "target_columns_sha256": column_hash(target_columns),
                "x_scaler_applied": True,
                "feature_scaler_fit_split": "train",
                "target_scaler_fit_split": "train_metadata_only",
                "source_ablation_dir": str(ablation_dir) if ablation_dir else "",
                "selected_target_manifest_path": str(selected_manifest_path) if selected_manifest_path else "",
                "selected_target_manifest_sha256": selected_manifest_sha256,
                "selected_target_contract_hash": selected_contract.get("selected_target_contract_hash", ""),
                "source_indicator_run_dir": str(indicator_dir),
                "original_source_rows": original_table_rows,
                "materialized_source_rows": int(len(table)),
                "max_rows_per_split": max_rows_per_split,
                "untouched_holdout_used_for_selection": False,
                "paper_only": True,
                "orders": False,
                "promotion": False,
                "champion_mutation": False,
            }
            contract["dataset_contract_hash"] = stable_hash(contract)
            dataset_hash = stable_hash({"contract": contract, "shape": list(arrays["X"].shape)})[:8]
            npz_path = dataset_dir / f"{bundle_name}_seq{seq_len}_{dataset_hash}.npz"
            np.savez_compressed(
                npz_path,
                X=arrays["X"],
                y=arrays["y"],
                splits=arrays["splits"],
                decision_timestamps=arrays["decision_timestamps"],
                label_end_timestamps=arrays["label_end_timestamps"],
                source_row_indices=arrays["source_row_indices"],
                row_indices=arrays["row_indices"],
                feature_columns=np.asarray(feature_columns, dtype=str),
                target_columns=np.asarray(target_columns, dtype=str),
                horizon_buckets=np.asarray(target_horizon_buckets, dtype=np.int64),
                unique_horizon_buckets=np.asarray(selected_horizons, dtype=np.int64),
                target_lane=np.asarray(target_lane, dtype=str),
                target_layout=np.asarray(target_layout, dtype=str),
                missing_mask=arrays["missing_mask"],
                feature_scaler_mean=arrays["feature_scaler_mean"],
                feature_scaler_std=arrays["feature_scaler_std"],
                target_scaler_mean=arrays["target_scaler_mean"],
                target_scaler_std=arrays["target_scaler_std"],
                dataset_contract=np.asarray(json.dumps(contract, sort_keys=True), dtype=str),
                dataset_contract_hash=np.asarray(contract["dataset_contract_hash"], dtype=str),
                selected_target_manifest_sha256=np.asarray(selected_manifest_sha256, dtype=str),
                selected_target_contract_hash=np.asarray(contract.get("selected_target_contract_hash", ""), dtype=str),
            )
            counts = split_counts(arrays["splits"])
            manifest_rows.append(
                {
                    "feature_group": bundle_name,
                    "bundle_name": bundle_name,
                    "seq_len": seq_len,
                    "status": "ok",
                    "arrays_written": True,
                    "path_npz": str(npz_path),
                    "sequence_rows": int(arrays["X"].shape[0]),
                    "train_sequence_rows": counts.get("train", 0),
                    "validation_sequence_rows": counts.get("validation", 0),
                    "holdout_sequence_rows": counts.get("untouched_holdout", 0),
                    "purge_embargo_sequence_rows": counts.get("purge_embargo", 0),
                    "feature_count": len(feature_columns),
                    "horizon_count": len(target_columns),
                    "target_mode": target_mode,
                    "target_lane": target_lane,
                    "target_layout": target_layout,
                    "target_horizons": ",".join(str(x) for x in selected_horizons),
                    "target_horizon_buckets": ",".join(str(x) for x in target_horizon_buckets),
                    "target_column_order": ";".join(target_columns),
                    "target_output_dim": len(target_columns),
                    "max_rows_per_split": max_rows_per_split,
                    "x_shape": json.dumps(list(arrays["X"].shape)),
                    "y_shape": json.dumps(list(arrays["y"].shape)),
                    "feature_columns_sha256": column_hash(feature_columns),
                    "target_columns_sha256": column_hash(target_columns),
                    "selected_target_manifest_sha256": selected_manifest_sha256,
                    "selected_target_contract_hash": contract.get("selected_target_contract_hash", ""),
                    "dataset_contract_hash": contract["dataset_contract_hash"],
                    "dataset_hash": dataset_hash,
                    "x_scaler_applied": True,
                    "complete_tensor_metadata": True,
                    "untouched_holdout_used_for_selection": False,
                }
            )
            write_json(npz_path.with_suffix(".contract.json"), contract)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "sequence_dataset_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    pd.DataFrame(bundle_rows).to_csv(out_dir / "locked_bundle_sequence_inputs.csv", index=False)
    baseline_reference_path = out_dir / "combined_leaderboard.csv"
    baseline_reference_copied = False
    baseline_reference_source = ablation_dir / "combined_leaderboard.csv" if ablation_dir is not None else Path()
    if ablation_dir is not None and baseline_reference_source.exists():
        shutil.copy2(baseline_reference_source, baseline_reference_path)
        baseline_reference_copied = True
    contract = {
        "created_at": now_iso(),
        "generator_path": "scripts/tiny/build_rawseq_locked_bundle_sequence_datasets.py",
        "git_head": git_head(),
        "git_status_dirty": git_status_dirty(),
        "ablation_dir": str(ablation_dir) if ablation_dir else "",
        "selected_target_manifest_path": str(selected_manifest_path) if selected_manifest_path else "",
        "selected_target_manifest_sha256": selected_manifest_sha256,
        "target_mode": target_mode,
        "target_lane": target_lane,
        "target_layout": target_layout,
        "indicator_run_dir": str(indicator_dir),
        "output_dir": str(out_dir),
        "original_source_rows": original_table_rows,
        "materialized_source_rows": int(len(table)),
        "max_rows_per_split": max_rows_per_split,
        "sequence_dataset_manifest": str(manifest_path),
        "baseline_reference_source": str(baseline_reference_source) if baseline_reference_source.exists() else "",
        "baseline_reference_path": str(baseline_reference_path) if baseline_reference_copied else "",
        "baseline_reference_copied": baseline_reference_copied,
        "bundle_names": bundle_names,
        "sequence_lens": seq_lens,
        "selected_horizons": selected_horizons,
        "target_horizon_buckets": target_horizon_buckets,
        "target_column_order": target_columns,
        "target_output_dim": len(target_columns),
        "dataset_count": len(manifest_rows),
        "untouched_holdout_used_for_selection": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    write_json(out_dir / "locked_bundle_sequence_dataset_contract.json", contract)
    lines = [
        "# Rawseq Locked Bundle Sequence Datasets",
        "",
        f"Created at: {contract['created_at']}",
        f"Ablation dir: `{ablation_dir}`",
        f"Selected target manifest: `{selected_manifest_path}`",
        f"Indicator run: `{indicator_dir}`",
        f"Output: `{out_dir}`",
        "",
        f"- bundles: {bundle_names}",
        f"- sequence lengths: {seq_lens}",
        f"- selected horizons: {selected_horizons}",
        f"- target mode: {target_mode}",
        f"- target lane: {target_lane}",
        f"- target layout: {target_layout}",
        f"- max rows per split: {max_rows_per_split if max_rows_per_split > 0 else 'unbounded'}",
        f"- dataset count: {len(manifest_rows)}",
        "- complete tensor metadata arrays: true",
        f"- baseline reference copied: {baseline_reference_copied}",
        "- untouched holdout used for selection: false",
        "",
        "Safety: no training; no Torch sequence model; no ensemble search; no freeze; no promotion; no champion mutation; no orders.",
    ]
    (out_dir / "locked_bundle_sequence_dataset_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Rawseq locked-bundle sequence datasets complete")
    print(f"Output: {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Datasets: {len(manifest_rows)}")
    print(f"Selected horizons: {selected_horizons}")
    print("Safety: no training. No Torch model. No orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
