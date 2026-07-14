#!/usr/bin/env python3
"""Report-only reconciliation for low-path CPU evidence vs bounded NPZ evidence.

This script explains whether the bounded 5,000-row sequence screen preserved the
CPU baseline evidence that opened the low/downside branch. It does not train GPU
models, select seeds, freeze candidates, promote artifacts, mutate champions, or
place orders.
"""

from __future__ import annotations

import csv
import hashlib
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

from scripts.tiny.build_rawseq_locked_bundle_sequence_datasets import materialize_selected_targets
from scripts.tiny.build_rawseq_locked_bundle_sequence_datasets import build_sequences
from scripts.tiny.report_rawseq_target_lane_baseline_tournament import (
    fill_targets,
    fit_elastic_net_proxy,
    fit_ridge,
    predict_linear,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_low_path_bounded_cpu_reconciliation"
DEFAULT_TOURNAMENT_GLOB = PROJECT_ROOT / "data" / "research" / "rawseq_target_lane_baseline_tournament"
DEFAULT_DIAG_GLOB = PROJECT_ROOT / "data" / "research" / "rawseq_feature_diagnostics"
DEFAULT_DATASET_GLOB = PROJECT_ROOT / "data" / "research" / "rawseq_locked_bundle_sequence_datasets"
DEFAULT_TORCH_ROOT = Path(r"F:\rsio\rawseq_low_path_gru_five_seed_smoke")


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def latest_dir(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No directories matched {root / pattern}")
    return candidates[0]


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
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def vector_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    per_target = []
    for idx in range(actual.shape[1]):
        mask = np.isfinite(actual[:, idx]) & np.isfinite(pred[:, idx])
        if mask.any():
            per_target.append(float(np.sqrt(np.mean((actual[mask, idx] - pred[mask, idx]) ** 2))))
    if not per_target:
        return math.nan
    return float(np.sqrt(np.mean(np.asarray(per_target, dtype=np.float64) ** 2)))


def vector_mae(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    per_target = []
    for idx in range(actual.shape[1]):
        mask = np.isfinite(actual[:, idx]) & np.isfinite(pred[:, idx])
        if mask.any():
            per_target.append(float(np.mean(np.abs(actual[mask, idx] - pred[mask, idx]))))
    return float(np.mean(per_target)) if per_target else math.nan


def array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def column_hash(columns: list[str]) -> str:
    return stable_hash(list(columns))


def improvement_vs_mean(model_rmse: float, mean_rmse: float) -> float:
    if not math.isfinite(model_rmse) or not math.isfinite(mean_rmse) or mean_rmse <= 0.0:
        return math.nan
    return (mean_rmse - model_rmse) / mean_rmse


def fit_feature_scaler(x_train: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}


def apply_feature_scaler(x: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, scaler["mean"])
    scaled = (filled - scaler["mean"]) / scaler["std"]
    scaled[~np.isfinite(scaled)] = 0.0
    return scaled.astype(np.float64)


def cpu_evaluate_arrays(
    case_id: str,
    train_x: np.ndarray,
    train_y_raw: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    feature_columns: list[str],
    target_columns: list[str],
    train_source_rows: np.ndarray,
    validation_source_rows: np.ndarray,
    train_timestamps: np.ndarray,
    validation_timestamps: np.ndarray,
    ridge_alphas: list[float],
    elastic_alphas: list[float],
    elastic_iterations: int,
    missing_value_handling: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_y, y_mean, y_median = fill_targets(train_y_raw.astype(np.float64))
    validation_y = validation_y.astype(np.float64)
    scaler = fit_feature_scaler(train_x.astype(np.float64))
    x_train = apply_feature_scaler(train_x.astype(np.float64), scaler)
    x_val = apply_feature_scaler(validation_x.astype(np.float64), scaler)
    predictions: dict[str, np.ndarray] = {
        "training_mean_baseline": np.tile(y_mean, (len(validation_y), 1)),
        "training_median_baseline": np.tile(y_median, (len(validation_y), 1)),
    }
    selected_hyperparameters: dict[str, Any] = {
        "training_mean_baseline": "",
        "training_median_baseline": "",
    }
    ridge_predictions: dict[float, np.ndarray] = {}
    for alpha in ridge_alphas:
        coef = fit_ridge(x_train, train_y, alpha)
        ridge_predictions[alpha] = predict_linear(x_val, coef)
    ridge_candidates = [
        (vector_rmse(validation_y, ridge_predictions[alpha]), alpha)
        for alpha in ridge_alphas
    ]
    best_ridge_alpha = sorted(ridge_candidates, key=lambda item: item[0] if math.isfinite(item[0]) else 1e18)[0][1]
    predictions["ridge_multi_output"] = ridge_predictions[best_ridge_alpha]
    selected_hyperparameters["ridge_multi_output"] = best_ridge_alpha
    elastic_predictions: dict[float, np.ndarray] = {}
    for alpha in elastic_alphas:
        coef = fit_elastic_net_proxy(x_train, train_y, alpha, 0.5, elastic_iterations, 0.02)
        elastic_predictions[alpha] = predict_linear(x_val, coef)
    elastic_candidates = [
        (vector_rmse(validation_y, elastic_predictions[alpha]), alpha)
        for alpha in elastic_alphas
    ]
    best_elastic_alpha = sorted(elastic_candidates, key=lambda item: item[0] if math.isfinite(item[0]) else 1e18)[0][1]
    predictions["elastic_net_multi_output"] = elastic_predictions[best_elastic_alpha]
    selected_hyperparameters["elastic_net_multi_output"] = best_elastic_alpha

    mean_rmse = vector_rmse(validation_y, predictions["training_mean_baseline"])
    mean_horizon_rmse = []
    mean_pred = predictions["training_mean_baseline"]
    for idx in range(validation_y.shape[1]):
        actual = validation_y[:, idx]
        target_pred = mean_pred[:, idx]
        mask = np.isfinite(actual) & np.isfinite(target_pred)
        mean_horizon_rmse.append(float(np.sqrt(np.mean((actual[mask] - target_pred[mask]) ** 2))) if mask.any() else math.nan)
    rows = []
    per_horizon = []
    for model, pred in predictions.items():
        rmse_value = vector_rmse(validation_y, pred)
        row = {
            "case_id": case_id,
            "model": model,
            "validation_vector_rmse": rmse_value,
            "validation_vector_mae": vector_mae(validation_y, pred),
            "validation_rmse_improvement_vs_training_mean": improvement_vs_mean(rmse_value, mean_rmse),
            "train_rows": int(len(train_y)),
            "validation_rows": int(len(validation_y)),
            "feature_column_order": ";".join(feature_columns),
            "target_column_order": ";".join(target_columns),
            "feature_columns_sha256": column_hash(feature_columns),
            "target_columns_sha256": column_hash(target_columns),
            "train_source_row_index_sha256": array_sha256(np.asarray(train_source_rows, dtype=np.int64)),
            "validation_source_row_index_sha256": array_sha256(np.asarray(validation_source_rows, dtype=np.int64)),
            "train_timestamp_sha256": array_sha256(np.asarray(train_timestamps, dtype=np.float64)),
            "validation_timestamp_sha256": array_sha256(np.asarray(validation_timestamps, dtype=np.float64)),
            "train_first_timestamp": float(train_timestamps[0]) if len(train_timestamps) else math.nan,
            "train_last_timestamp": float(train_timestamps[-1]) if len(train_timestamps) else math.nan,
            "validation_first_timestamp": float(validation_timestamps[0]) if len(validation_timestamps) else math.nan,
            "validation_last_timestamp": float(validation_timestamps[-1]) if len(validation_timestamps) else math.nan,
            "missing_value_handling": missing_value_handling,
            "feature_scaler_state_hash": stable_hash({"mean": scaler["mean"].tolist(), "std": scaler["std"].tolist()}),
            "feature_scaler_mean_sha256": array_sha256(scaler["mean"]),
            "feature_scaler_std_sha256": array_sha256(scaler["std"]),
            "target_scaler_state_hash": stable_hash({"mean": y_mean.tolist(), "median": y_median.tolist()}),
            "target_mean_sha256": array_sha256(y_mean),
            "target_median_sha256": array_sha256(y_median),
            "model_intercept_behavior": "explicit_intercept_unpenalized_for_ridge_and_elastic",
            "ridge_grid": ",".join(str(x) for x in ridge_alphas),
            "elastic_net_grid": ",".join(str(x) for x in elastic_alphas),
            "selected_hyperparameter": selected_hyperparameters.get(model, ""),
            "holdout_used_for_selection": False,
        }
        rows.append(row)
        for idx, target in enumerate(target_columns):
            actual = validation_y[:, idx]
            target_pred = pred[:, idx]
            mask = np.isfinite(actual) & np.isfinite(target_pred)
            per_horizon.append(
                {
                    "case_id": case_id,
                    "model": model,
                    "target_column": target,
                    "horizon_index": idx,
                    "validation_rmse": float(np.sqrt(np.mean((actual[mask] - target_pred[mask]) ** 2))) if mask.any() else math.nan,
                    "validation_mae": float(np.mean(np.abs(actual[mask] - target_pred[mask]))) if mask.any() else math.nan,
                    "mean_baseline_validation_rmse": mean_horizon_rmse[idx],
                    "validation_rmse_improvement_vs_training_mean": improvement_vs_mean(
                        float(np.sqrt(np.mean((actual[mask] - target_pred[mask]) ** 2))) if mask.any() else math.nan,
                        mean_horizon_rmse[idx],
                    ),
                    "validation_rows": int(mask.sum()),
                }
            )
    best = sorted(rows, key=lambda row: row["validation_vector_rmse"] if math.isfinite(row["validation_vector_rmse"]) else 1e18)[0]
    best_summary = {
        **best,
        "selected_cpu_model": best["model"],
        "training_mean_validation_vector_rmse": mean_rmse,
        "per_horizon_rows": per_horizon,
    }
    return rows, best_summary


def load_selected_payload(tournament_dir: Path) -> dict[str, Any]:
    manifest = json.loads((tournament_dir / "selected_target_manifest.json").read_text(encoding="utf-8"))
    selected = manifest.get("selected_targets") or []
    if not selected:
        raise SystemExit(f"{tournament_dir} selected_target_manifest.json has no selected_targets")
    return selected[0]


def load_feature_columns(diag_dir: Path, bundle: str) -> list[str]:
    payload = json.loads((diag_dir / f"feature_bundle_{bundle}.json").read_text(encoding="utf-8"))
    return [str(col) for col in payload.get("ordered_feature_columns", [])]


def load_training_table(indicator_dir: Path, feature_columns: list[str]) -> pd.DataFrame:
    source = indicator_dir / "multi_horizon_training_table.csv"
    header = pd.read_csv(source, nrows=0).columns.tolist()
    base_cols = ["split", "decision_timestamp"]
    for candidate in ["close", "price", "bucket_return_bps"]:
        if candidate in header:
            base_cols.append(candidate)
    usecols = list(dict.fromkeys([*base_cols, *[col for col in feature_columns if col in header]]))
    table = pd.read_csv(source, usecols=usecols)
    table["source_row_index"] = np.arange(len(table), dtype=np.int64)
    return table


def select_first_rows_per_split(table: pd.DataFrame, rows_per_split: int) -> pd.DataFrame:
    if rows_per_split <= 0:
        return table.copy()
    return table.groupby("split", sort=False, group_keys=False).head(rows_per_split).sort_index().copy()


def contiguous_split_window(table: pd.DataFrame, rows_per_split: int, fraction: float) -> pd.DataFrame:
    parts = []
    for _, group in table.groupby("split", sort=False):
        if rows_per_split <= 0 or len(group) <= rows_per_split:
            parts.append(group)
            continue
        start = int(round((len(group) - rows_per_split) * fraction))
        start = max(0, min(start, len(group) - rows_per_split))
        parts.append(group.iloc[start : start + rows_per_split])
    return pd.concat(parts, axis=0).sort_index().copy()


def split_coverage_rows(table: pd.DataFrame, scope: str, target_columns: list[str], feature_columns: list[str], source_sha: str) -> list[dict[str, Any]]:
    rows = []
    for split, group in table.groupby("split", sort=False):
        ts = pd.to_numeric(group["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        targets = group[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        features = group[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        finite_targets = targets[np.isfinite(targets)]
        finite_features = features[np.isfinite(features)]
        rows.append(
            {
                "scope": scope,
                "split": split,
                "rows": int(len(group)),
                "first_source_row_index": int(group["source_row_index"].iloc[0]) if len(group) else -1,
                "last_source_row_index": int(group["source_row_index"].iloc[-1]) if len(group) else -1,
                "first_timestamp": float(ts[0]) if len(ts) else math.nan,
                "last_timestamp": float(ts[-1]) if len(ts) else math.nan,
                "duration_seconds": float((ts[-1] - ts[0]) / 1000.0) if len(ts) > 1 else 0.0,
                "duration_hours": float((ts[-1] - ts[0]) / 3_600_000.0) if len(ts) > 1 else 0.0,
                "target_mean": float(np.mean(finite_targets)) if finite_targets.size else math.nan,
                "target_std": float(np.std(finite_targets)) if finite_targets.size else math.nan,
                "target_abs_p50": float(np.quantile(np.abs(finite_targets), 0.50)) if finite_targets.size else math.nan,
                "target_abs_p90": float(np.quantile(np.abs(finite_targets), 0.90)) if finite_targets.size else math.nan,
                "feature_mean": float(np.mean(finite_features)) if finite_features.size else math.nan,
                "feature_std": float(np.std(finite_features)) if finite_features.size else math.nan,
                "bucket_return_std": float(np.nanstd(pd.to_numeric(group.get("bucket_return_bps", pd.Series(dtype=float)), errors="coerce"))) if "bucket_return_bps" in group else math.nan,
                "source_artifact_sha256": source_sha,
            }
        )
    return rows


def evaluate_cpu_scope(
    table: pd.DataFrame,
    scope: str,
    target_columns: list[str],
    feature_columns: list[str],
    ridge_alphas: list[float],
    elastic_alphas: list[float],
    elastic_iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    predictions, manifest = build_baselines(
        table,
        target_columns,
        feature_columns,
        ridge_alphas=ridge_alphas,
        elastic_alphas=elastic_alphas,
        elastic_l1_ratio=0.5,
        elastic_iterations=elastic_iterations,
        elastic_lr=0.02,
    )
    validation = table["split"].astype(str).eq("validation").to_numpy()
    y = table[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    rows = []
    mean_rmse = vector_rmse(y[validation], predictions["training_mean_baseline"][validation])
    for model, pred in predictions.items():
        rmse_value = vector_rmse(y[validation], pred[validation])
        mae_value = vector_mae(y[validation], pred[validation])
        rows.append(
            {
                "scope": scope,
                "model": model,
                "validation_vector_rmse": rmse_value,
                "validation_vector_mae": mae_value,
                "validation_rmse_improvement_vs_training_mean": improvement_vs_mean(rmse_value, mean_rmse),
                "validation_rows": int(validation.sum()),
                "holdout_used_for_selection": False,
            }
        )
    best = sorted(rows, key=lambda row: row["validation_vector_rmse"] if math.isfinite(row["validation_vector_rmse"]) else 1e18)[0]
    best = {**best, "selected_cpu_model": best["model"], "training_mean_validation_vector_rmse": mean_rmse}
    manifest_rows = [{**row, "scope": scope} for row in manifest]
    return rows, manifest_rows, best


def arrays_from_frame(frame: pd.DataFrame, feature_columns: list[str], target_columns: list[str]) -> dict[str, np.ndarray]:
    return {
        "x": frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64),
        "y": frame[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64),
        "source_rows": frame["source_row_index"].to_numpy(dtype=np.int64),
        "timestamps": pd.to_numeric(frame["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64),
    }


def sequence_compatible_frame(table: pd.DataFrame, seq_len: int, target_columns: list[str]) -> pd.DataFrame:
    targets = table[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    timestamps = pd.to_numeric(table["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.arange(len(table)) >= max(0, seq_len - 1)
    mask &= np.isfinite(timestamps)
    mask &= np.isfinite(targets).all(axis=1)
    return table.loc[mask].copy()


def build_sequences_for_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    label_timestamp_columns: list[str],
    seq_len: int,
) -> dict[str, np.ndarray]:
    arrays = build_sequences(frame, feature_columns, target_columns, label_timestamp_columns, seq_len)
    return {
        "x": arrays["X"][:, -1, :].astype(np.float64),
        "y": arrays["y"].astype(np.float64),
        "splits": arrays["splits"].astype(str),
        "source_rows": arrays["source_row_indices"].astype(np.int64),
        "timestamps": arrays["decision_timestamps"].astype(np.float64),
        "feature_scaler_mean": arrays["feature_scaler_mean"],
        "feature_scaler_std": arrays["feature_scaler_std"],
        "target_scaler_mean": arrays["target_scaler_mean"],
        "target_scaler_std": arrays["target_scaler_std"],
    }


def arrays_from_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        x = data["X"][:, -1, :].astype(np.float64)
        return {
            "x": x,
            "y": data["y"].astype(np.float64),
            "splits": data["splits"].astype(str),
            "source_rows": (data["source_row_indices"] if "source_row_indices" in data.files else data["row_indices"]).astype(np.int64),
            "timestamps": data["decision_timestamps"].astype(np.float64),
            "feature_scaler_mean": data["feature_scaler_mean"] if "feature_scaler_mean" in data.files else np.asarray([]),
            "feature_scaler_std": data["feature_scaler_std"] if "feature_scaler_std" in data.files else np.asarray([]),
            "target_scaler_mean": data["target_scaler_mean"] if "target_scaler_mean" in data.files else np.asarray([]),
            "target_scaler_std": data["target_scaler_std"] if "target_scaler_std" in data.files else np.asarray([]),
        }


def evaluate_case_from_arrays(
    case_id: str,
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    feature_columns: list[str],
    target_columns: list[str],
    ridge_alphas: list[float],
    elastic_alphas: list[float],
    elastic_iterations: int,
    missing_value_handling: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return cpu_evaluate_arrays(
        case_id,
        train_arrays["x"],
        train_arrays["y"],
        validation_arrays["x"],
        validation_arrays["y"],
        feature_columns,
        target_columns,
        train_arrays["source_rows"],
        validation_arrays["source_rows"],
        train_arrays["timestamps"],
        validation_arrays["timestamps"],
        ridge_alphas,
        elastic_alphas,
        elastic_iterations,
        missing_value_handling,
    )


def split_arrays(arrays: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    mask = arrays["splits"].astype(str) == split
    return {
        "x": arrays["x"][mask],
        "y": arrays["y"][mask],
        "source_rows": arrays["source_rows"][mask],
        "timestamps": arrays["timestamps"][mask],
    }


POSITION_FRACTIONS = {
    "earliest_contiguous": 0.0,
    "early_middle_contiguous": 0.25,
    "middle_contiguous": 0.5,
    "late_middle_contiguous": 0.75,
    "latest_contiguous": 1.0,
}


def split_window(group: pd.DataFrame, rows: int, fraction: float) -> pd.DataFrame:
    if rows <= 0 or len(group) <= rows:
        return group.copy()
    start = int(round((len(group) - rows) * fraction))
    start = max(0, min(start, len(group) - rows))
    return group.iloc[start : start + rows].copy()


def sample_contiguous_train_validation(table: pd.DataFrame, rows_per_split: int, position: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    fraction = POSITION_FRACTIONS[position]
    parts = []
    blocks = []
    for split in ["train", "validation"]:
        group = table[table["split"].astype(str).eq(split)].copy()
        sampled = split_window(group, rows_per_split, fraction)
        sampled = sampled.copy()
        sampled["sampling_block_id"] = f"{split}_block_0"
        parts.append(sampled)
        blocks.append(block_record("contiguous", position, split, 0, sampled))
    return pd.concat(parts, axis=0).sort_index().copy(), blocks


def distributed_split_blocks(group: pd.DataFrame, total_rows: int, block_count: int) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    if total_rows <= 0 or len(group) <= total_rows:
        return group.copy(), [group.copy()]
    block_size = total_rows // block_count
    if block_size <= 0:
        raise ValueError("distributed block_size must be positive")
    usable_rows = block_size * block_count
    max_start = len(group) - block_size
    starts = np.linspace(0, max_start, block_count, dtype=int)
    blocks = []
    used_ranges: list[tuple[int, int]] = []
    for start in starts:
        end = int(start) + block_size
        if used_ranges and start < used_ranges[-1][1]:
            start = used_ranges[-1][1]
            end = start + block_size
        if end > len(group):
            end = len(group)
            start = max(0, end - block_size)
        used_ranges.append((int(start), int(end)))
        blocks.append(group.iloc[int(start) : int(end)].copy())
    sampled = pd.concat(blocks, axis=0).head(usable_rows).sort_index().copy()
    return sampled, blocks


def sample_distributed_train_validation(table: pd.DataFrame, rows_per_split: int, block_count: int = 5) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts = []
    records = []
    for split in ["train", "validation"]:
        group = table[table["split"].astype(str).eq(split)].copy()
        sampled, blocks = distributed_split_blocks(group, rows_per_split, block_count)
        block_parts = []
        for idx, block in enumerate(blocks):
            block = block.copy()
            block["sampling_block_id"] = f"{split}_block_{idx}"
            block_parts.append(block)
            records.append(block_record("distributed_blocks", f"{block_count}_blocks", split, idx, block))
        parts.append(pd.concat(block_parts, axis=0).sort_index().copy() if block_parts else sampled)
    return pd.concat(parts, axis=0).sort_index().copy(), records


def block_record(policy: str, position: str, split: str, block_index: int, frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) == 0:
        return {
            "sampling_policy": policy,
            "temporal_position": position,
            "split": split,
            "block_index": block_index,
            "rows": 0,
        }
    ts = pd.to_numeric(frame["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    return {
        "sampling_policy": policy,
        "temporal_position": position,
        "split": split,
        "block_index": block_index,
        "rows": int(len(frame)),
        "first_source_row_index": int(frame["source_row_index"].iloc[0]),
        "last_source_row_index": int(frame["source_row_index"].iloc[-1]),
        "first_timestamp": float(ts[0]),
        "last_timestamp": float(ts[-1]),
        "duration_hours": float((ts[-1] - ts[0]) / 3_600_000.0) if len(ts) > 1 else 0.0,
    }


def sample_arrays_from_frame(
    sample_id: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    target_lane: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    materialized = materialized_copy(frame, target_lane, target_columns)
    train = materialized[materialized["split"].astype(str).eq("train")].copy()
    validation = materialized[materialized["split"].astype(str).eq("validation")].copy()
    train_arrays = arrays_from_frame(train, feature_columns, target_columns)
    validation_arrays = arrays_from_frame(validation, feature_columns, target_columns)
    train_arrays["sample_id"] = np.asarray([sample_id])
    validation_arrays["sample_id"] = np.asarray([sample_id])
    return train_arrays, validation_arrays, materialized


def moment_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: math.nan for key in ["mean", "median", "std", "p10", "p25", "p75", "p90", "p99", "zero_fraction", "skewness", "kurtosis"]}
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    centered = finite - mean
    skew = float(np.mean((centered / std) ** 3)) if std > 1e-12 else math.nan
    kurt = float(np.mean((centered / std) ** 4) - 3.0) if std > 1e-12 else math.nan
    return {
        "mean": mean,
        "median": float(np.median(finite)),
        "std": std,
        "p10": float(np.quantile(finite, 0.10)),
        "p25": float(np.quantile(finite, 0.25)),
        "p75": float(np.quantile(finite, 0.75)),
        "p90": float(np.quantile(finite, 0.90)),
        "p99": float(np.quantile(finite, 0.99)),
        "zero_fraction": float(np.mean(np.isclose(finite, 0.0))),
        "skewness": skew,
        "kurtosis": kurt,
    }


def autocorr(values: np.ndarray, lag: int) -> float:
    finite = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(finite)
    finite = finite[mask]
    if finite.size <= lag + 2:
        return math.nan
    a = finite[:-lag]
    b = finite[lag:]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def row_nanmean_no_warn(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    counts = finite.sum(axis=1)
    sums = np.where(finite, array, 0.0).sum(axis=1)
    out = np.full(array.shape[0], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def regime_rows_for_sample(sample_id: str, frame: pd.DataFrame, target_columns: list[str], selected_cpu_model: str, selected_improvement: float) -> list[dict[str, Any]]:
    rows = []
    for split in ["train", "validation"]:
        part = frame[frame["split"].astype(str).eq(split)].copy()
        if len(part) == 0:
            continue
        targets = part[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        flat_stats = moment_stats(targets.ravel())
        base = {
            "sample_id": sample_id,
            "split": split,
            "target_column": "all",
            "selected_cpu_model": selected_cpu_model,
            "selected_model_improvement_vs_mean": selected_improvement,
            "rows": int(len(part)),
            **{f"target_{key}": value for key, value in flat_stats.items()},
        }
        returns = pd.to_numeric(part.get("bucket_return_bps", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64)
        ranges = pd.to_numeric(part.get("rolling_range_bps_fw60", part.get("raw_range_bps", pd.Series(dtype=float))), errors="coerce").to_numpy(dtype=np.float64)
        volume = pd.to_numeric(part.get("volume", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64)
        spread_proxy = pd.to_numeric(part.get("spread_percent", part.get("raw_range_bps", pd.Series(dtype=float))), errors="coerce").to_numpy(dtype=np.float64)
        base.update(
            {
                "realized_return_volatility": float(np.nanstd(returns)) if returns.size else math.nan,
                "rolling_range_mean": float(np.nanmean(ranges)) if ranges.size else math.nan,
                "rolling_range_p90": float(np.nanquantile(ranges, 0.90)) if np.isfinite(ranges).any() else math.nan,
                "volume_mean": float(np.nanmean(volume)) if volume.size else math.nan,
                "volume_p90": float(np.nanquantile(volume, 0.90)) if np.isfinite(volume).any() else math.nan,
                "spread_proxy_mean": float(np.nanmean(spread_proxy)) if spread_proxy.size else math.nan,
                "spread_proxy_p90": float(np.nanquantile(spread_proxy, 0.90)) if np.isfinite(spread_proxy).any() else math.nan,
                "lag1_target_autocorrelation": autocorr(row_nanmean_no_warn(targets), 1),
                "lag10_target_autocorrelation": autocorr(row_nanmean_no_warn(targets), 10),
                "lag60_target_autocorrelation": autocorr(row_nanmean_no_warn(targets), 60),
            }
        )
        rows.append(base)
        for idx, column in enumerate(target_columns):
            stats = moment_stats(targets[:, idx])
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "target_column": column,
                    "selected_cpu_model": selected_cpu_model,
                    "selected_model_improvement_vs_mean": selected_improvement,
                    "rows": int(np.isfinite(targets[:, idx]).sum()),
                    **{f"target_{key}": value for key, value in stats.items()},
                    "target_correlation_to_bucket_return": (
                        float(np.corrcoef(targets[:, idx], returns)[0, 1])
                        if returns.size == len(targets) and np.std(returns) > 1e-12 and np.std(targets[:, idx]) > 1e-12
                        else math.nan
                    ),
                    "lag1_target_autocorrelation": autocorr(targets[:, idx], 1),
                    "lag10_target_autocorrelation": autocorr(targets[:, idx], 10),
                    "lag60_target_autocorrelation": autocorr(targets[:, idx], 60),
                }
            )
    return rows


def add_train_validation_shift(rows: list[dict[str, Any]]) -> None:
    by_sample_target: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("sample_id")), str(row.get("target_column")))
        by_sample_target.setdefault(key, {})[str(row.get("split"))] = row
    for pair in by_sample_target.values():
        train = pair.get("train")
        validation = pair.get("validation")
        if not train or not validation:
            continue
        train_std = safe_float(train.get("target_std"))
        for row in [train, validation]:
            row["train_to_validation_normalized_mean_shift"] = (
                (safe_float(validation.get("target_mean")) - safe_float(train.get("target_mean"))) / train_std
                if math.isfinite(train_std) and train_std > 1e-12
                else math.nan
            )
            row["train_to_validation_std_ratio"] = (
                safe_float(validation.get("target_std")) / train_std
                if math.isfinite(train_std) and train_std > 1e-12
                else math.nan
            )
            row["train_to_validation_p90_shift"] = safe_float(validation.get("target_p90")) - safe_float(train.get("target_p90"))


def selected_surface_row(best: dict[str, Any], row_count_label: str, sampling_policy: str, temporal_position: str, total_rows_per_split: int) -> dict[str, Any]:
    model = str(best.get("selected_cpu_model", ""))
    return {
        **{key: value for key, value in best.items() if key != "per_horizon_rows"},
        "sample_id": str(best.get("case_id", "")),
        "row_count_label": row_count_label,
        "sampling_policy": sampling_policy,
        "temporal_position": temporal_position,
        "total_rows_per_split": total_rows_per_split,
        "learned_cpu_win": model in {"ridge_multi_output", "elastic_net_multi_output"} and safe_float(best.get("validation_rmse_improvement_vs_training_mean")) > 0.0,
    }


def summarize_surface(surface_rows: list[dict[str, Any]], per_horizon_rows: list[dict[str, Any]], target_columns: list[str]) -> dict[str, Any]:
    if not surface_rows:
        return {
            "learned_win_count": 0,
            "minimum_reliable_row_count": "",
            "best_contiguous_policy": "",
            "best_distributed_policy": "",
            "per_horizon_median_improvements": {},
            "recommendation": "stop_or_redesign_low_path_target_or_features",
        }
    frame = pd.DataFrame(surface_rows)
    learned = frame[frame["learned_cpu_win"].astype(bool)].copy()
    bounded_contiguous = frame[frame["sampling_policy"].eq("contiguous") & frame["row_count_label"].ne("full")]
    win_by_row = (
        bounded_contiguous.groupby("row_count_label")["learned_cpu_win"].sum().to_dict()
        if len(bounded_contiguous)
        else {}
    )
    learned_bounded_contiguous = bounded_contiguous[bounded_contiguous["learned_cpu_win"].astype(bool)].copy()
    median_by_row = (
        learned_bounded_contiguous.groupby("row_count_label")["validation_rmse_improvement_vs_training_mean"].median().to_dict()
        if len(learned_bounded_contiguous)
        else {}
    )
    reliable_rows = [
        row
        for row, count in win_by_row.items()
        if int(count) >= 3 and safe_float(median_by_row.get(row)) > 0.0
    ]
    numeric_reliable = sorted((int(row) for row in reliable_rows if str(row).isdigit()))
    minimum_reliable = str(numeric_reliable[0]) if numeric_reliable else ""
    best_contiguous = ""
    best_distributed = ""
    learned_frame = frame[frame["learned_cpu_win"].astype(bool)].copy()
    if len(learned_frame[learned_frame["sampling_policy"].eq("contiguous")]):
        best = learned_frame[learned_frame["sampling_policy"].eq("contiguous")].sort_values("validation_rmse_improvement_vs_training_mean", ascending=False).iloc[0]
        best_contiguous = str(best["sample_id"])
    if len(learned_frame[learned_frame["sampling_policy"].eq("distributed_blocks")]):
        best = learned_frame[learned_frame["sampling_policy"].eq("distributed_blocks")].sort_values("validation_rmse_improvement_vs_training_mean", ascending=False).iloc[0]
        best_distributed = str(best["sample_id"])
    horizon_frame = pd.DataFrame(per_horizon_rows)
    selected_model_by_sample = dict(zip(frame["sample_id"].astype(str), frame["selected_cpu_model"].astype(str)))
    learned_samples = set(learned_frame["sample_id"].astype(str))
    horizon_frame = horizon_frame[
        horizon_frame["case_id"].astype(str).isin(learned_samples)
    ].copy()
    if len(horizon_frame):
        horizon_frame = horizon_frame[
            horizon_frame.apply(lambda row: str(row["model"]) == selected_model_by_sample.get(str(row["case_id"]), ""), axis=1)
        ].copy()
    per_horizon_medians = {}
    positive_horizon_median_count = 0
    for target in target_columns:
        vals = horizon_frame[horizon_frame["target_column"].astype(str).eq(target)]["validation_rmse_improvement_vs_training_mean"]
        med = float(vals.median()) if len(vals) else math.nan
        per_horizon_medians[target] = med
        if math.isfinite(med) and med > 0.0:
            positive_horizon_median_count += 1
    adjacent_reproduction = False
    sorted_rows = sorted(int(row) for row in win_by_row if str(row).isdigit())
    for left, right in zip(sorted_rows, sorted_rows[1:]):
        if int(win_by_row.get(str(left), 0)) >= 3 and int(win_by_row.get(str(right), 0)) >= 3:
            adjacent_reproduction = True
    distributed_positive = bool(
        len(frame[(frame["sampling_policy"].eq("distributed_blocks")) & (frame["learned_cpu_win"].astype(bool))])
    )
    contiguous_positive = bool(
        len(frame[(frame["sampling_policy"].eq("contiguous")) & (frame["learned_cpu_win"].astype(bool))])
    )
    reproduces_two_modes = adjacent_reproduction or (distributed_positive and contiguous_positive)
    passes = bool(
        minimum_reliable
        and positive_horizon_median_count >= 3
        and reproduces_two_modes
    )
    return {
        "learned_win_count": int(frame["learned_cpu_win"].sum()),
        "evaluated_sampling_configurations": int(len(frame)),
        "learned_win_count_by_row_count": {str(k): int(v) for k, v in win_by_row.items()},
        "learned_win_count_by_temporal_position": {
            str(k): int(v)
            for k, v in frame.groupby("temporal_position")["learned_cpu_win"].sum().to_dict().items()
        },
        "median_improvement_by_row_count": {str(k): safe_float(v) for k, v in median_by_row.items()},
        "minimum_reliable_row_count": minimum_reliable,
        "best_contiguous_policy": best_contiguous,
        "best_distributed_policy": best_distributed,
        "per_horizon_median_improvements": per_horizon_medians,
        "positive_horizon_median_count": positive_horizon_median_count,
        "reproduces_two_adjacent_row_counts_or_sampling_modes": reproduces_two_modes,
        "recommendation": "build_low_path_residual_gru_screen" if passes else "stop_or_redesign_low_path_target_or_features",
    }


def regime_win_loss_contrast(regime_rows: list[dict[str, Any]], surface_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not regime_rows or not surface_rows:
        return {}
    surface = pd.DataFrame(surface_rows)
    win_by_sample = dict(zip(surface["sample_id"].astype(str), surface["learned_cpu_win"].astype(bool)))
    frame = pd.DataFrame(regime_rows)
    frame = frame[(frame["split"].astype(str) == "validation") & (frame["target_column"].astype(str) == "all")].copy()
    if len(frame) == 0:
        return {}
    frame["learned_cpu_win"] = frame["sample_id"].astype(str).map(win_by_sample).fillna(False)
    metrics = [
        "target_mean",
        "target_median",
        "target_std",
        "target_p90",
        "target_p99",
        "realized_return_volatility",
        "rolling_range_mean",
        "rolling_range_p90",
        "volume_mean",
        "spread_proxy_mean",
        "lag1_target_autocorrelation",
        "lag10_target_autocorrelation",
        "lag60_target_autocorrelation",
        "train_to_validation_normalized_mean_shift",
        "train_to_validation_std_ratio",
        "train_to_validation_p90_shift",
    ]
    out: dict[str, Any] = {
        "validation_win_samples": int(frame["learned_cpu_win"].sum()),
        "validation_loss_samples": int((~frame["learned_cpu_win"]).sum()),
    }
    diffs = []
    for metric in metrics:
        wins = pd.to_numeric(frame[frame["learned_cpu_win"]][metric], errors="coerce")
        losses = pd.to_numeric(frame[~frame["learned_cpu_win"]][metric], errors="coerce")
        win_mean = float(wins.mean()) if len(wins) else math.nan
        loss_mean = float(losses.mean()) if len(losses) else math.nan
        diff = win_mean - loss_mean if math.isfinite(win_mean) and math.isfinite(loss_mean) else math.nan
        out[f"{metric}_win_mean"] = win_mean
        out[f"{metric}_loss_mean"] = loss_mean
        out[f"{metric}_win_minus_loss"] = diff
        if math.isfinite(diff):
            diffs.append((abs(diff), metric, diff))
    out["largest_regime_differences"] = [
        {"metric": metric, "win_minus_loss": diff}
        for _, metric, diff in sorted(diffs, reverse=True)[:5]
    ]
    return out


def materialized_copy(table: pd.DataFrame, target_lane: str, target_columns: list[str]) -> pd.DataFrame:
    out = table.copy()
    materialize_selected_targets(out, target_lane, target_columns)
    return out


def load_npz_row_coverage(npz_path: Path, timestamp_to_source: dict[float, int] | None = None) -> list[dict[str, Any]]:
    with np.load(npz_path, allow_pickle=False) as data:
        splits = data["splits"].astype(str)
        timestamps = data["decision_timestamps"].astype(np.float64)
        label_end = data["label_end_timestamps"].astype(np.float64) if "label_end_timestamps" in data.files else np.empty((len(splits), 0))
        source_rows = data["source_row_indices"].astype(np.int64) if "source_row_indices" in data.files else data["row_indices"].astype(np.int64)
        y = data["y"].astype(np.float64)
    rows = []
    for split in ["train", "validation", "untouched_holdout", "purge_embargo"]:
        mask = splits == split
        if not mask.any():
            continue
        split_label_end = label_end[mask] if label_end.size else np.empty((int(mask.sum()), 0))
        split_timestamps = timestamps[mask]
        inferred_source = [
            timestamp_to_source.get(float(ts), -1) if timestamp_to_source is not None else -1
            for ts in split_timestamps
        ]
        rows.append(
            {
                "scope": "bounded_npz_sequence_rows",
                "split": split,
                "sequence_rows": int(mask.sum()),
                "first_npz_stored_row_index": int(source_rows[mask][0]),
                "last_npz_stored_row_index": int(source_rows[mask][-1]),
                "first_inferred_original_source_row_index": int(inferred_source[0]) if inferred_source else -1,
                "last_inferred_original_source_row_index": int(inferred_source[-1]) if inferred_source else -1,
                "npz_stored_row_index_basis": "post_cap_local_table_index",
                "first_timestamp": float(split_timestamps[0]),
                "last_timestamp": float(split_timestamps[-1]),
                "duration_seconds": float((split_timestamps[-1] - split_timestamps[0]) / 1000.0),
                "duration_hours": float((split_timestamps[-1] - split_timestamps[0]) / 3_600_000.0),
                "max_label_end_timestamp": float(np.nanmax(split_label_end)) if split_label_end.size else math.nan,
                "target_mean": float(np.nanmean(y[mask])),
                "target_std": float(np.nanstd(y[mask])),
            }
        )
    return rows


def read_selected_npz_path(dataset_dir: Path) -> Path:
    manifest = pd.read_csv(dataset_dir / "sequence_dataset_manifest.csv")
    row = manifest.iloc[0].to_dict()
    return Path(str(row["path_npz"]))


def load_locked_cpu_summary(torch_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    path = torch_dir / "rawseq_low_path_locked_cpu_reference.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") == "validation" and not row.get("target_name"):
                    rows.append(
                        {
                            "scope": "bounded_npz_locked_cpu_reference",
                            "model": row.get("cpu_model", ""),
                            "validation_vector_rmse": safe_float(row.get("vector_rmse")),
                            "validation_vector_mae": safe_float(row.get("vector_mae")),
                            "validation_rmse_improvement_vs_training_mean": math.nan,
                            "validation_rows": int(float(row.get("rows", 0) or 0)),
                            "holdout_used_for_selection": row.get("holdout_used_for_selection", "False"),
                        }
                    )
    contract = {}
    contract_path = torch_dir / "rawseq_low_path_locked_cpu_contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    mean_rmse = next((row["validation_vector_rmse"] for row in rows if row["model"] == "training_mean_baseline"), math.nan)
    for row in rows:
        row["validation_rmse_improvement_vs_training_mean"] = improvement_vs_mean(row["validation_vector_rmse"], mean_rmse)
    return rows, contract


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    tournament_dir = env_path(
        "RAWSEQ_LOW_RECON_TOURNAMENT_DIR",
        latest_dir(DEFAULT_TOURNAMENT_GLOB, "rawseq_target_lane_baseline_tournament_*"),
    )
    diag_dir = env_path("RAWSEQ_LOW_RECON_DIAG_DIR", latest_dir(DEFAULT_DIAG_GLOB, "rawseq_feature_diagnostics_*"))
    indicator_dir = env_path(
        "RAWSEQ_LOW_RECON_INDICATOR_DIR",
        Path(r"F:\rsio\rawseq_target_tournament_coarse_1s_300k_retry\mh_indicator_SOLUSDT_kraken_20260711T145015Z_fba19c8d"),
    )
    dataset_dir = env_path("RAWSEQ_LOW_RECON_DATASET_DIR", latest_dir(DEFAULT_DATASET_GLOB, "rawseq_locked_bundle_sequence_datasets_*"))
    torch_dir = env_path("RAWSEQ_LOW_RECON_TORCH_DIR", latest_dir(DEFAULT_TORCH_ROOT, "torch_sequence_benchmark_*"))
    output_root = env_path("RAWSEQ_LOW_RECON_OUTPUT_DIR", DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / f"rawseq_low_path_bounded_cpu_reconciliation_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    selected = load_selected_payload(tournament_dir)
    target_lane = str(selected["target_lane"])
    feature_bundle = str(selected["preferred_feature_bundle"])
    target_columns = [str(col) for col in selected["target_column_order"]]
    feature_columns = load_feature_columns(diag_dir, feature_bundle)
    source_csv = indicator_dir / "multi_horizon_training_table.csv"
    source_sha = file_sha256(source_csv)
    base_table = load_training_table(indicator_dir, feature_columns)
    available_features = [col for col in feature_columns if col in base_table.columns]
    if len(available_features) != len(feature_columns):
        missing = sorted(set(feature_columns) - set(available_features))
        raise SystemExit(f"Missing feature columns in table: {missing[:10]}")
    timestamp_to_source = {
        float(ts): int(idx)
        for ts, idx in zip(
            pd.to_numeric(base_table["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64),
            base_table["source_row_index"].to_numpy(dtype=np.int64),
        )
        if math.isfinite(float(ts))
    }
    table = materialized_copy(base_table, target_lane, target_columns)

    rows_per_split = int(float(os.getenv("RAWSEQ_LOW_RECON_ROWS_PER_SPLIT", "5000")))
    ridge_alphas = [float(x) for x in os.getenv("RAWSEQ_LOW_RECON_RIDGE_ALPHAS", "0.01,0.1,1,10,100").split(",")]
    elastic_alphas = [float(x) for x in os.getenv("RAWSEQ_LOW_RECON_ELASTIC_ALPHAS", "0.001,0.01,0.1").split(",")]
    elastic_iterations = int(float(os.getenv("RAWSEQ_LOW_RECON_ELASTIC_ITERATIONS", "40")))

    coverage_rows: list[dict[str, Any]] = []
    cpu_rows: list[dict[str, Any]] = []
    per_horizon_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    npz_path = read_selected_npz_path(dataset_dir)
    npz_arrays = arrays_from_npz(npz_path)
    npz_train = split_arrays(npz_arrays, "train")
    npz_validation = split_arrays(npz_arrays, "validation")

    label_timestamp_columns = [f"target_lane_label_end_timestamp_h{int(col.rsplit('h', 1)[1])}" for col in target_columns]
    builder_style_frame = materialized_copy(select_first_rows_per_split(base_table, rows_per_split), target_lane, target_columns)
    reconstructed_sequence = build_sequences_for_frame(builder_style_frame, available_features, target_columns, label_timestamp_columns, seq_len=60)
    reconstructed_train = split_arrays(reconstructed_sequence, "train")
    reconstructed_validation = split_arrays(reconstructed_sequence, "validation")

    full_sequence_compatible = sequence_compatible_frame(table, 60, target_columns)
    exact_npz_train_timestamps = set(float(x) for x in npz_train["timestamps"])
    exact_npz_validation_timestamps = set(float(x) for x in npz_validation["timestamps"])
    exact_npz_train_window = table[table["decision_timestamp"].astype(float).isin(exact_npz_train_timestamps)].copy()
    exact_npz_validation_window = table[table["decision_timestamp"].astype(float).isin(exact_npz_validation_timestamps)].copy()

    def frame_split_arrays(frame: pd.DataFrame, split: str) -> dict[str, np.ndarray]:
        return arrays_from_frame(frame[frame["split"].astype(str).eq(split)].copy(), available_features, target_columns)

    case_specs: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray], str]] = [
        (
            "A_full_tournament_train_to_full_validation",
            frame_split_arrays(table, "train"),
            frame_split_arrays(table, "validation"),
            "finite_fill_with_train_feature_mean_then_train_feature_standardization",
        ),
        (
            "B_full_sequence_compatible_train_to_validation",
            frame_split_arrays(full_sequence_compatible, "train"),
            frame_split_arrays(full_sequence_compatible, "validation"),
            "sequence_compatible_rows_only; finite_fill_with_train_feature_mean_then_train_feature_standardization",
        ),
        (
            "C_full_train_to_exact_npz_validation_window",
            frame_split_arrays(table, "train"),
            arrays_from_frame(exact_npz_validation_window, available_features, target_columns),
            "validation_window_by_npz_timestamps; targets_materialized_before_cap",
        ),
        (
            "D_exact_npz_train_window_to_full_sequence_validation",
            arrays_from_frame(exact_npz_train_window, available_features, target_columns),
            frame_split_arrays(full_sequence_compatible, "validation"),
            "train_window_by_npz_timestamps; targets_materialized_before_cap",
        ),
        (
            "E_reconstructed_builder_sequence_train_to_validation",
            reconstructed_train,
            reconstructed_validation,
            "builder_style_cap_before_targets; reconstructed_sequence_final_step_features",
        ),
        (
            "F_serialized_npz_arrays_train_to_validation",
            npz_train,
            npz_validation,
            "serialized_npz_final_step_features; finite_fill_with_train_feature_mean_then_train_feature_standardization",
        ),
    ]

    for case_id, train_arrays, validation_arrays, missing_handling in case_specs:
        rows, best = evaluate_case_from_arrays(
            case_id,
            train_arrays,
            validation_arrays,
            available_features,
            target_columns,
            ridge_alphas,
            elastic_alphas,
            elastic_iterations,
            missing_handling,
        )
        cpu_rows.extend(rows)
        per_horizon_rows.extend(best.pop("per_horizon_rows"))
        best_rows.append(best)

    scopes = {
        "full_train_validation_rows": table,
        f"bounded_first_{rows_per_split}_rows_per_split_targets_materialized_before_cap": select_first_rows_per_split(table, rows_per_split),
        f"bounded_first_{rows_per_split}_rows_per_split_builder_style_cap_before_targets": builder_style_frame,
    }
    window_specs = [
        ("early", 0.0),
        ("early_middle", 0.25),
        ("middle", 0.5),
        ("late_middle", 0.75),
        ("late", 1.0),
    ]
    for name, fraction in window_specs:
        scopes[f"window_{name}_{rows_per_split}_rows_per_split_targets_materialized_before_window"] = contiguous_split_window(table, rows_per_split, fraction)
        scopes[f"window_{name}_{rows_per_split}_rows_per_split_builder_style_window_before_targets"] = materialized_copy(
            contiguous_split_window(base_table, rows_per_split, fraction),
            target_lane,
            target_columns,
        )

    for scope, frame in scopes.items():
        coverage_rows.extend(split_coverage_rows(frame, scope, target_columns, available_features, source_sha))
    npz_coverage_rows = load_npz_row_coverage(npz_path, timestamp_to_source)
    locked_cpu_rows, locked_contract = load_locked_cpu_summary(torch_dir)
    if locked_cpu_rows:
        mean_rmse = next((row["validation_vector_rmse"] for row in locked_cpu_rows if row["model"] == "training_mean_baseline"), math.nan)
        best_locked = sorted(locked_cpu_rows, key=lambda row: row["validation_vector_rmse"] if math.isfinite(row["validation_vector_rmse"]) else 1e18)[0]
        best_rows.append({**best_locked, "case_id": "LOCKED_existing_artifact", "selected_cpu_model": best_locked["model"], "training_mean_validation_vector_rmse": mean_rmse})

    tournament_bundle_path = tournament_dir / "target_lane_bundle_metrics.csv"
    tournament_bundle = pd.read_csv(tournament_bundle_path)
    tournament_low = tournament_bundle[
        (tournament_bundle["target_lane"].astype(str) == target_lane)
        & (tournament_bundle["feature_bundle"].astype(str) == feature_bundle)
    ]
    tournament_summary = tournament_low.iloc[0].to_dict() if len(tournament_low) else {}

    write_csv(out_dir / "rawseq_low_path_cpu_reconciliation.csv", cpu_rows)
    write_csv(out_dir / "rawseq_low_path_cpu_per_horizon.csv", per_horizon_rows)
    write_csv(out_dir / "rawseq_low_path_cpu_best_by_scope.csv", best_rows)
    write_csv(out_dir / "rawseq_low_path_split_coverage.csv", coverage_rows)
    write_csv(out_dir / "rawseq_low_path_npz_sequence_coverage.csv", npz_coverage_rows)

    selected_by_case = {row.get("case_id", row.get("scope", "")): row for row in best_rows}
    case_a = selected_by_case.get("A_full_tournament_train_to_full_validation", {})
    case_b = selected_by_case.get("B_full_sequence_compatible_train_to_validation", {})
    case_c = selected_by_case.get("C_full_train_to_exact_npz_validation_window", {})
    case_d = selected_by_case.get("D_exact_npz_train_window_to_full_sequence_validation", {})
    case_e = selected_by_case.get("E_reconstructed_builder_sequence_train_to_validation", {})
    case_f = selected_by_case.get("F_serialized_npz_arrays_train_to_validation", {})
    locked_best = selected_by_case.get("LOCKED_existing_artifact", {})
    case_a_improvement = safe_float(case_a.get("validation_rmse_improvement_vs_training_mean"))
    case_e_rmse = safe_float(case_e.get("validation_vector_rmse"))
    locked_rmse = safe_float(locked_best.get("validation_vector_rmse"))
    case_a_pass = (
        case_a.get("selected_cpu_model") == str(tournament_summary.get("learned_baseline_model_used", "ridge_multi_output"))
        and abs(case_a_improvement - safe_float(tournament_summary.get("validation_baseline_improvement_fraction"))) <= 1e-9
    )
    case_e_pass = (
        case_e.get("selected_cpu_model") == locked_best.get("selected_cpu_model", locked_contract.get("selected_cpu_model", ""))
        and math.isfinite(case_e_rmse)
        and math.isfinite(locked_rmse)
        and abs(case_e_rmse - locked_rmse) <= 1e-6
    )
    parity_pass = bool(case_a_pass and case_e_pass)
    window_best: list[dict[str, Any]] = []
    learned_wins: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    surface_per_horizon_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    sampling_block_rows: list[dict[str, Any]] = []
    surface_summary = {
        "learned_win_count": 0,
        "evaluated_sampling_configurations": 0,
        "minimum_reliable_row_count": "",
        "best_contiguous_policy": "",
        "best_distributed_policy": "",
        "per_horizon_median_improvements": {},
        "recommendation": "stop_or_redesign_low_path_target_or_features",
    }
    if parity_pass:
        sample_configs: list[tuple[str, str, str, int, pd.DataFrame, list[dict[str, Any]]]] = []
        full_frame = table[table["split"].astype(str).isin(["train", "validation"])].copy()
        full_blocks = [
            block_record("full_split", "full", split, 0, full_frame[full_frame["split"].astype(str).eq(split)].copy())
            for split in ["train", "validation"]
        ]
        sample_configs.append(("full_full_split", "full", "full_split", 0, full_frame, full_blocks))
        for count in [5000, 10000, 20000, 50000]:
            for position in POSITION_FRACTIONS:
                sample_frame, blocks = sample_contiguous_train_validation(base_table, count, position)
                sample_configs.append((f"contiguous_{count}_{position}", str(count), "contiguous", count, sample_frame, blocks))
            if count in {20000, 50000}:
                sample_frame, blocks = sample_distributed_train_validation(base_table, count, 5)
                sample_configs.append((f"distributed_blocks_{count}_5x{count // 5}", str(count), "distributed_blocks", count, sample_frame, blocks))
        for sample_id, row_count_label, policy, total_rows, sample_frame, blocks in sample_configs:
            train_arrays, validation_arrays, materialized_sample = sample_arrays_from_frame(sample_id, sample_frame, available_features, target_columns, target_lane)
            rows, best = evaluate_case_from_arrays(
                sample_id,
                train_arrays,
                validation_arrays,
                available_features,
                target_columns,
                ridge_alphas,
                elastic_alphas,
                elastic_iterations,
                f"{policy};holdout_excluded;target_materialized_after_sampling",
            )
            cpu_rows.extend(rows)
            sample_per_horizon = best.pop("per_horizon_rows")
            per_horizon_rows.extend(sample_per_horizon)
            surface_per_horizon_rows.extend(sample_per_horizon)
            best_rows.append(best)
            temporal_position = "full" if policy == "full_split" else ("distributed_blocks" if policy == "distributed_blocks" else sample_id.split(f"contiguous_{total_rows}_", 1)[-1])
            surface_row = selected_surface_row(best, row_count_label, policy, temporal_position, total_rows)
            surface_rows.append(surface_row)
            if policy == "contiguous":
                window_best.append(surface_row)
            sampling_block_rows.extend([{**block, "sample_id": sample_id, "row_count_label": row_count_label, "sampling_policy": policy} for block in blocks])
            regime_rows.extend(regime_rows_for_sample(sample_id, materialized_sample, target_columns, str(best.get("selected_cpu_model", "")), safe_float(best.get("validation_rmse_improvement_vs_training_mean"))))
        learned_wins = [row for row in window_best if row.get("selected_cpu_model") not in {"training_mean_baseline", "training_median_baseline", "zero_baseline"} and safe_float(row.get("validation_rmse_improvement_vs_training_mean")) > 0]
        add_train_validation_shift(regime_rows)
        surface_summary = summarize_surface(surface_rows, surface_per_horizon_rows, target_columns)
        surface_summary["regime_win_loss_contrast"] = regime_win_loss_contrast(regime_rows, surface_rows)
        write_csv(out_dir / "rawseq_low_path_cpu_reconciliation.csv", cpu_rows)
        write_csv(out_dir / "rawseq_low_path_cpu_per_horizon.csv", per_horizon_rows)
        write_csv(out_dir / "rawseq_low_path_cpu_best_by_scope.csv", best_rows)
        write_csv(out_dir / "rawseq_low_path_cpu_learnability_surface.csv", surface_rows)
        write_csv(out_dir / "rawseq_low_path_cpu_regime_diagnostics.csv", regime_rows)
        write_csv(out_dir / "rawseq_low_path_cpu_sampling_blocks.csv", sampling_block_rows)
        write_csv(out_dir / "rawseq_low_path_cpu_surface_per_horizon.csv", surface_per_horizon_rows)
    case_f_rmse = safe_float(case_f.get("validation_vector_rmse"))
    case_b_differs = (
        case_a.get("selected_cpu_model") != case_b.get("selected_cpu_model")
        or abs(case_a_improvement - safe_float(case_b.get("validation_rmse_improvement_vs_training_mean"))) > 1e-4
    )
    case_c_passes = safe_float(case_c.get("validation_rmse_improvement_vs_training_mean")) > 0.0
    case_d_passes = safe_float(case_d.get("validation_rmse_improvement_vs_training_mean")) > 0.0
    case_e_f_differs = (
        case_e.get("selected_cpu_model") != case_f.get("selected_cpu_model")
        or (math.isfinite(case_e_rmse) and math.isfinite(case_f_rmse) and abs(case_e_rmse - case_f_rmse) > 1e-6)
    )
    interpretation_flags = []
    if not case_a_pass:
        interpretation_flags.append("A_failed_evaluator_parity_defect")
    if not case_e_pass:
        interpretation_flags.append("E_failed_locked_cpu_reproduction")
    if case_c_passes and not case_d_passes:
        interpretation_flags.append("training_cap_primary_cause")
    if not case_c_passes:
        interpretation_flags.append("validation_window_regime_primary_cause")
    if case_b_differs:
        interpretation_flags.append("sequence_eligibility_or_warmup_material")
    if case_e_f_differs:
        interpretation_flags.append("npz_preprocessing_or_serialization_defect")
    if parity_pass and not interpretation_flags:
        interpretation_flags.append("parity_passed_no_single_cause_flag")

    matrix_rows = [case_a, case_b, case_c, case_d, case_e, case_f]
    recommendation = {
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "gpu_training_rerun": False,
        "audit_status": "PARITY_PASSED" if parity_pass else "PARITY_FAILED",
        "target_lane": target_lane,
        "feature_bundle": feature_bundle,
        "row_cap_semantics": "groupby(split).head(max_rows_per_split), then sort_index",
        "full_tournament_reported_improvement": safe_float(tournament_summary.get("validation_baseline_improvement_fraction")),
        "case_a_pass": case_a_pass,
        "case_e_pass": case_e_pass,
        "case_a_selected_cpu": case_a.get("selected_cpu_model", ""),
        "case_a_improvement": case_a_improvement,
        "case_b_selected_cpu": case_b.get("selected_cpu_model", ""),
        "case_b_improvement": safe_float(case_b.get("validation_rmse_improvement_vs_training_mean")),
        "case_c_selected_cpu": case_c.get("selected_cpu_model", ""),
        "case_c_improvement": safe_float(case_c.get("validation_rmse_improvement_vs_training_mean")),
        "case_d_selected_cpu": case_d.get("selected_cpu_model", ""),
        "case_d_improvement": safe_float(case_d.get("validation_rmse_improvement_vs_training_mean")),
        "case_e_selected_cpu": case_e.get("selected_cpu_model", ""),
        "case_e_improvement": safe_float(case_e.get("validation_rmse_improvement_vs_training_mean")),
        "case_f_selected_cpu": case_f.get("selected_cpu_model", ""),
        "case_f_improvement": safe_float(case_f.get("validation_rmse_improvement_vs_training_mean")),
        "bounded_npz_locked_selected_cpu": locked_best.get("selected_cpu_model", locked_contract.get("selected_cpu_model", "")),
        "bounded_npz_locked_improvement": safe_float(locked_best.get("validation_rmse_improvement_vs_training_mean")),
        "interpretation_flags": interpretation_flags,
        "evaluated_sampling_configurations": int(surface_summary.get("evaluated_sampling_configurations", 0)),
        "learned_cpu_win_count": int(surface_summary.get("learned_win_count", 0)),
        "learned_win_count_by_row_count": surface_summary.get("learned_win_count_by_row_count", {}),
        "learned_win_count_by_temporal_position": surface_summary.get("learned_win_count_by_temporal_position", {}),
        "median_improvement_by_row_count": surface_summary.get("median_improvement_by_row_count", {}),
        "minimum_reliable_row_count": surface_summary.get("minimum_reliable_row_count", ""),
        "best_contiguous_policy": surface_summary.get("best_contiguous_policy", ""),
        "best_distributed_policy": surface_summary.get("best_distributed_policy", ""),
        "per_horizon_median_improvements": surface_summary.get("per_horizon_median_improvements", {}),
        "regime_win_loss_contrast": surface_summary.get("regime_win_loss_contrast", {}),
        "positive_horizon_median_count": surface_summary.get("positive_horizon_median_count", 0),
        "reproduces_two_adjacent_row_counts_or_sampling_modes": surface_summary.get("reproduces_two_adjacent_row_counts_or_sampling_modes", False),
        "recommended_next_step": (
            "fix_cpu_parity_before_window_replication"
            if not parity_pass
            else surface_summary.get("recommendation", "stop_or_redesign_low_path_target_or_features")
        ),
    }
    (out_dir / "rawseq_low_path_bounded_cpu_reconciliation.json").write_text(json.dumps(recommendation, indent=2, sort_keys=True), encoding="utf-8")
    surface_contract = {
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "holdout_used_for_selection": False,
        "gpu_training_rerun": False,
        "target_lane": target_lane,
        "feature_bundle": feature_bundle,
        "target_columns": target_columns,
        "feature_columns_sha256": column_hash(available_features),
        "target_columns_sha256": column_hash(target_columns),
        "row_counts_per_split": [5000, 10000, 20000, 50000, "full"],
        "temporal_positions": list(POSITION_FRACTIONS.keys()),
        "distributed_block_contracts": [
            {"total_rows_per_split": 20000, "blocks": 5, "rows_per_block": 4000},
            {"total_rows_per_split": 50000, "blocks": 5, "rows_per_block": 10000},
        ],
        "cpu_models": ["training_mean_baseline", "training_median_baseline", "ridge_multi_output", "elastic_net_multi_output"],
        "ridge_grid": ridge_alphas,
        "elastic_net_grid": elastic_alphas,
        "elastic_iterations": elastic_iterations,
        "parity_a_full_tournament_pass": case_a_pass,
        "parity_e_locked_npz_pass": case_e_pass,
        "recommendation_gate": {
            "learned_cpu_wins_at_least_3_of_5_positions": True,
            "median_learned_improvement_positive": True,
            "at_least_3_of_4_horizons_positive_median": True,
            "reproduce_two_adjacent_row_counts_or_sampling_modes": True,
            "npz_reconstruction_parity_required": True,
            "holdout_unused_required": True,
        },
        "surface_summary": surface_summary,
    }
    surface_contract["contract_sha256"] = stable_hash(surface_contract)
    (out_dir / "rawseq_low_path_cpu_learnability_contract.json").write_text(json.dumps(surface_contract, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "rawseq_low_path_cpu_sampling_recommendation.json").write_text(json.dumps(recommendation, indent=2, sort_keys=True), encoding="utf-8")

    surface_lines = [
        "Rawseq low-path CPU learnability surface",
        f"Output: {out_dir}",
        f"Evaluated sampling configurations: {recommendation['evaluated_sampling_configurations']}",
        f"Learned-CPU wins: {recommendation['learned_cpu_win_count']}",
        f"Learned wins by row count: {recommendation['learned_win_count_by_row_count']}",
        f"Learned wins by temporal position: {recommendation['learned_win_count_by_temporal_position']}",
        f"Median improvement by row count: {recommendation['median_improvement_by_row_count']}",
        f"Minimum reliable row count: {recommendation['minimum_reliable_row_count'] or 'none'}",
        f"Best contiguous policy: {recommendation['best_contiguous_policy'] or 'none'}",
        f"Best distributed policy: {recommendation['best_distributed_policy'] or 'none'}",
        f"Per-horizon median improvements: {recommendation['per_horizon_median_improvements']}",
        f"Key regime differences: {recommendation['regime_win_loss_contrast'].get('largest_regime_differences', []) if isinstance(recommendation['regime_win_loss_contrast'], dict) else []}",
        f"Holdout used for selection: false",
        f"Final recommendation: {recommendation['recommended_next_step']}",
    ]
    (out_dir / "rawseq_low_path_cpu_learnability_surface.txt").write_text("\n".join(surface_lines) + "\n", encoding="utf-8")

    lines = [
        "Rawseq low-path bounded CPU reconciliation",
        f"Output: {out_dir}",
        "",
        "Safety: paper_only=true private_api=false orders=false promotion=false champion_mutation=false gpu_training_rerun=false",
        "",
        f"Target lane: {target_lane}",
        f"Feature bundle: {feature_bundle}",
        f"Target columns: {', '.join(target_columns)}",
        f"Row-cap semantics: groupby(split).head({rows_per_split}), then sort_index",
        "",
        "A-F CPU parity matrix:",
        f"- Tournament reported full_registered improvement: {recommendation['full_tournament_reported_improvement']}",
        *[
            f"- {row.get('case_id')}: {row.get('selected_cpu_model')} improvement={safe_float(row.get('validation_rmse_improvement_vs_training_mean'))} rmse={safe_float(row.get('validation_vector_rmse'))}"
            for row in matrix_rows
        ],
        f"- Existing locked NPZ CPU artifact: {recommendation['bounded_npz_locked_selected_cpu']} improvement={recommendation['bounded_npz_locked_improvement']} rmse={locked_rmse}",
        f"- Parity status: {recommendation['audit_status']} case_A_pass={case_a_pass} case_E_pass={case_e_pass}",
        f"- Interpretation flags: {', '.join(interpretation_flags)}",
        f"- Sampling configurations evaluated: {recommendation['evaluated_sampling_configurations']}",
        f"- Learned-CPU wins: {recommendation['learned_cpu_win_count']}",
        f"- Minimum reliable row count: {recommendation['minimum_reliable_row_count'] or 'none'}",
        f"- Best contiguous policy: {recommendation['best_contiguous_policy'] or 'none'}",
        f"- Best distributed policy: {recommendation['best_distributed_policy'] or 'none'}",
        "",
        "Interpretation:",
        "Case A must reproduce tournament evidence; otherwise this audit has an evaluator parity defect.",
        "Case E must reproduce the locked NPZ CPU result; otherwise reconstructed sequence rows do not match the serialized bounded evidence.",
        "Deterministic temporal-window replication is only run when the parity matrix passes.",
        "",
        f"Recommended next step: {recommendation['recommended_next_step']}",
    ]
    (out_dir / "rawseq_low_path_bounded_cpu_reconciliation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0 if parity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
