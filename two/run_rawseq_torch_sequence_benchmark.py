#!/usr/bin/env python3
"""Paper-only torch sequence benchmarks for rawseq multi-horizon datasets.

This consumes sequence_dataset_manifest.csv files produced by
run_rawseq_multi_horizon_indicator_pipeline.py. The expected contract is:

    X: [batch, seq_len, feature_count]
    y: [batch, horizon_count]

When torch is unavailable the script writes an explicit skipped report instead
of treating the GPU stage as implicitly complete.

Safety: no private API, no orders, no promotion, and no champion mutation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_multi_horizon_indicator_returns"
BASELINE_BASE_MODELS = {
    "zero_return_baseline",
    "training_mean_return_baseline",
    "rolling_mean_momentum_baseline",
    "mean_reversion_baseline",
    "ridge_multi_output",
    "elastic_net_multi_output",
    "logistic_direction_model",
    "small_tree_baseline",
    "boosted_tree_baseline",
}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "true" if default else "false").strip().lower()
    return value in {"1", "true", "yes", "y"}


def parse_csv_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def parse_float_list(name: str, default: str) -> list[float]:
    return [float(item.strip()) for item in os.getenv(name, default).split(",") if item.strip()]


def parse_int_list(name: str, default: str) -> list[int]:
    return [int(float(item.strip())) for item in os.getenv(name, default).split(",") if item.strip()]


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ndarray_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def column_hash(columns: list[str]) -> str:
    return stable_hash(list(columns))


def horizon_from_target_column(column: str) -> int:
    import re

    match = re.search(r"h(\d+)$", str(column))
    if not match:
        raise ValueError(f"Could not infer horizon from target column: {column}")
    return int(match.group(1))


def horizon_buckets_for_target_columns(target_columns: list[str]) -> list[int]:
    return [horizon_from_target_column(column) for column in target_columns]


def expected_range_envelope_column_order(horizons: list[int]) -> list[str]:
    out: list[str] = []
    for horizon in sorted(set(horizons)):
        out.extend([f"future_range_high_bps_h{horizon}", f"future_range_low_bps_h{horizon}"])
    return out


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else math.nan


def finite_std(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.std(finite)) if finite.size else math.nan


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base or PROJECT_ROOT) / path
    return path


def latest_sequence_manifest() -> Path:
    candidates = sorted(
        DEFAULT_SEQUENCE_ROOT.glob("**/sequence_dataset_manifest.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            "RAWSEQ_TORCH_SEQUENCE_DATASET_MANIFEST is required; no default sequence_dataset_manifest.csv was found"
        )
    return candidates[0]


def torch_status() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"torch_available": False, "cuda_available": False, "torch_version": ""}
    import torch  # type: ignore

    return {
        "torch_available": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }


def bool_from_cell(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_dataset_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["manifest_path"] = str(path)
    frame["manifest_dir"] = str(path.parent)
    return frame


def resolve_dataset_path(path_value: Any, manifest_dir: Path) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = manifest_dir / path
    return path


def candidate_dataset_rows(
    manifest: pd.DataFrame,
    manifest_path: Path,
    feature_groups: set[str] | None = None,
    seq_lens: set[int] | None = None,
    max_datasets: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_dir = manifest_path.parent
    for idx, row in manifest.iterrows():
        feature_group = str(row.get("feature_group", ""))
        seq_len = int(float(row.get("seq_len", 0) or 0))
        if feature_groups and feature_group not in feature_groups:
            continue
        if seq_lens and seq_len not in seq_lens:
            continue
        dataset_path = resolve_dataset_path(row.get("path_npz", ""), manifest_dir)
        status = str(row.get("status", "")).lower()
        arrays_written = bool_from_cell(row.get("arrays_written", "false"))
        if status != "ok" or not arrays_written or not dataset_path.exists():
            rows.append(
                {
                    **row.to_dict(),
                    "dataset_index": idx,
                    "dataset_path": str(dataset_path),
                    "dataset_status": "unusable_dataset",
                    "failure": "status_not_ok_or_arrays_missing",
                }
            )
            continue
        rows.append(
            {
                **row.to_dict(),
                "dataset_index": idx,
                "dataset_path": str(dataset_path),
                "dataset_status": "ok",
                "failure": "",
            }
        )
        if max_datasets > 0 and len([item for item in rows if item["dataset_status"] == "ok"]) >= max_datasets:
            break
    return rows


def corr(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if int(mask.sum()) < 2:
        return math.nan
    a = actual[mask]
    p = pred[mask]
    if float(np.nanstd(a)) <= 1e-12 or float(np.nanstd(p)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, p)[0, 1])


def balanced_accuracy(actual_positive: np.ndarray, pred_positive: np.ndarray) -> float:
    actual_positive = np.asarray(actual_positive, dtype=bool)
    pred_positive = np.asarray(pred_positive, dtype=bool)
    positives = actual_positive
    negatives = ~actual_positive
    tpr = float(np.mean(pred_positive[positives])) if positives.any() else math.nan
    tnr = float(np.mean(~pred_positive[negatives])) if negatives.any() else math.nan
    values = [value for value in [tpr, tnr] if np.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def metric_row(actual: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return {
            "rows": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "correlation": math.nan,
            "directional_accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "positive_actual_fraction": math.nan,
            "positive_pred_fraction": math.nan,
        }
    actual_m = actual[mask]
    pred_m = pred[mask]
    actual_positive = actual_m > 0.0
    pred_positive = pred_m > 0.0
    return {
        "rows": int(mask.sum()),
        "mae": float(np.mean(np.abs(actual_m - pred_m))),
        "rmse": float(np.sqrt(np.mean((actual_m - pred_m) ** 2))),
        "correlation": corr(actual_m, pred_m),
        "directional_accuracy": float(np.mean(actual_positive == pred_positive)),
        "balanced_accuracy": balanced_accuracy(actual_positive, pred_positive),
        "positive_actual_fraction": float(np.mean(actual_positive)),
        "positive_pred_fraction": float(np.mean(pred_positive)),
    }


def parse_dataset_contract(raw: Any) -> dict[str, Any]:
    try:
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))
    except Exception:
        return {}


def validate_npz_target_contract(
    path: Path,
    y: np.ndarray,
    target_columns: list[str],
    horizon_buckets: list[int],
    target_lane: str,
    target_layout: str,
    contract: dict[str, Any],
) -> None:
    derived_horizons = horizon_buckets_for_target_columns(target_columns)
    if len(horizon_buckets) != len(target_columns):
        raise ValueError(
            f"{path} horizon_buckets length mismatch: "
            f"horizon_buckets={len(horizon_buckets)} target_columns={len(target_columns)}"
        )
    if horizon_buckets != derived_horizons:
        raise ValueError(
            f"{path} horizon_buckets do not match target column suffixes: "
            f"horizon_buckets={horizon_buckets} target_column_suffixes={derived_horizons}"
        )
    if int(y.shape[1]) != len(target_columns):
        raise ValueError(
            f"{path} output dimension mismatch: y_dim={int(y.shape[1])} target_columns={len(target_columns)}"
        )
    contract_columns = [str(x) for x in contract.get("target_column_order", [])]
    if contract_columns and contract_columns != target_columns:
        raise ValueError(f"{path} dataset_contract target_column_order mismatch")
    contract_horizons = [int(x) for x in contract.get("target_horizon_buckets", [])]
    if contract_horizons and contract_horizons != horizon_buckets:
        raise ValueError(f"{path} dataset_contract target_horizon_buckets mismatch")
    contract_unique = [int(x) for x in contract.get("unique_target_horizons", contract.get("target_horizons", []))]
    if contract_unique and contract_unique != sorted(set(horizon_buckets)):
        raise ValueError(f"{path} dataset_contract unique horizon mismatch")
    contract_output_dim = int(contract.get("target_output_dim", len(target_columns)))
    if contract_output_dim != len(target_columns):
        raise ValueError(f"{path} dataset_contract target_output_dim mismatch")
    contract_hash = str(contract.get("dataset_contract_hash", "") or "")
    if contract_hash:
        computed = stable_hash({key: value for key, value in contract.items() if key != "dataset_contract_hash"})
        if computed != contract_hash:
            raise ValueError(f"{path} dataset_contract_hash mismatch: declared={contract_hash} computed={computed}")
    if str(contract.get("target_columns_sha256", "") or "") and str(contract.get("target_columns_sha256")) != column_hash(target_columns):
        raise ValueError(f"{path} target_columns_sha256 mismatch")
    if target_lane == "future_range_envelope_path":
        expected = expected_range_envelope_column_order(sorted(set(horizon_buckets)))
        if target_layout != "range_envelope_path":
            raise ValueError(f"{path} range-envelope target layout mismatch: {target_layout}")
        if target_columns != expected:
            raise ValueError(f"{path} range-envelope interleaved target order mismatch")
    if target_lane == "future_low_from_now_bps_path":
        expected = [f"future_range_low_bps_h{horizon}" for horizon in sorted(set(horizon_buckets))]
        if target_layout != "scalar_path":
            raise ValueError(f"{path} low-only target layout mismatch: {target_layout}")
        if target_columns != expected:
            raise ValueError(f"{path} low-only target order mismatch or high/range contamination")


def infer_target_lane(target_columns: list[str], contract: dict[str, Any]) -> str:
    lane = str(contract.get("target_lane", "") or "").strip()
    if lane:
        return lane
    joined = ";".join(target_columns)
    has_range_high = any("future_range_high_bps" in str(column) for column in target_columns)
    has_range_low = any("future_range_low_bps" in str(column) for column in target_columns)
    if has_range_low and not has_range_high:
        return "future_low_from_now_bps_path"
    if has_range_high and has_range_low:
        return "future_range_envelope_path"
    if "future_high_from_now_bps" in joined:
        return "future_high_from_now_bps_path"
    if "future_low_from_now_bps" in joined:
        return "future_low_from_now_bps_path"
    if "coarse_return_bps" in joined:
        return "coarse_return_vector"
    return "future_return_path"


def infer_target_layout(target_lane: str, contract: dict[str, Any]) -> str:
    layout = str(contract.get("target_layout", "") or "").strip()
    if layout:
        return layout
    if target_lane == "future_range_envelope_path":
        return "range_envelope_path"
    if target_lane in {"future_high_from_now_bps_path", "future_low_from_now_bps_path"}:
        return "scalar_path"
    return "return_vector"


def split_target_indices(target_columns: list[str]) -> dict[str, list[int]]:
    out = {"return": [], "high": [], "low": []}
    for idx, column in enumerate(target_columns):
        text = str(column)
        if "high" in text:
            out["high"].append(idx)
        elif "low" in text:
            out["low"].append(idx)
        else:
            out["return"].append(idx)
    return out


def path_metric_row(actual: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return {"rows": 0, "mae": math.nan, "rmse": math.nan}
    err = actual[mask] - pred[mask]
    return {
        "rows": int(mask.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def flattened_pair(actual: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual_arr = np.asarray(actual, dtype=np.float64).reshape(-1)
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(actual_arr) & np.isfinite(pred_arr)
    return actual_arr[mask], pred_arr[mask]


def pearson_corr(actual: np.ndarray, pred: np.ndarray) -> float:
    a, p = flattened_pair(actual, pred)
    if len(a) < 2 or float(np.std(a)) <= 1e-12 or float(np.std(p)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, p)[0, 1])


def spearman_corr(actual: np.ndarray, pred: np.ndarray) -> float:
    a, p = flattened_pair(actual, pred)
    if len(a) < 2:
        return math.nan
    ar = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    pr = pd.Series(p).rank(method="average").to_numpy(dtype=np.float64)
    if float(np.std(ar)) <= 1e-12 or float(np.std(pr)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(ar, pr)[0, 1])


def std_ratio(actual: np.ndarray, pred: np.ndarray) -> float:
    a, p = flattened_pair(actual, pred)
    actual_std = float(np.std(a)) if len(a) else math.nan
    pred_std = float(np.std(p)) if len(p) else math.nan
    return pred_std / actual_std if math.isfinite(actual_std) and actual_std > 1e-12 and math.isfinite(pred_std) else math.nan


def calibration(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    a, p = flattened_pair(actual, pred)
    if len(a) < 2 or float(np.std(p)) <= 1e-12:
        return math.nan, math.nan
    slope, intercept = np.polyfit(p, a, 1)
    return float(slope), float(intercept)


def unique_fraction(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan
    rounded = np.round(arr, 8)
    return float(len(np.unique(rounded)) / len(rounded))


def top_decile_recall_precision(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    a, p = flattened_pair(actual, pred)
    if len(a) < 10 or len(p) < 10:
        return math.nan, math.nan
    actual_cut = float(np.quantile(a, 0.90))
    pred_cut = float(np.quantile(p, 0.90))
    actual_top = a >= actual_cut
    pred_top = p >= pred_cut
    tp = int(np.sum(actual_top & pred_top))
    recall = tp / int(np.sum(actual_top)) if int(np.sum(actual_top)) else math.nan
    precision = tp / int(np.sum(pred_top)) if int(np.sum(pred_top)) else math.nan
    return float(recall), float(precision)


def monotonic_violation_fraction(values: np.ndarray, direction: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return 0.0
    diffs = np.diff(arr, axis=1)
    if direction == "nondecreasing":
        violations = diffs < -1e-9
    else:
        violations = diffs > 1e-9
    finite = np.isfinite(diffs)
    return float(np.sum(violations & finite) / max(int(finite.sum()), 1))


def terminal_corr(actual: np.ndarray, pred: np.ndarray) -> float:
    if actual.ndim == 1 or actual.shape[1] == 0:
        return math.nan
    return corr(actual[:, -1], pred[:, -1])


def train_mean_baseline(y_true: np.ndarray, splits: np.ndarray) -> np.ndarray:
    split_values = splits.astype(str)
    train = y_true[split_values == "train"]
    mean = np.nanmean(train, axis=0) if len(train) else np.nanmean(y_true, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return np.tile(mean, (len(y_true), 1))


def train_median_baseline(y_true: np.ndarray, splits: np.ndarray) -> np.ndarray:
    split_values = splits.astype(str)
    train = y_true[split_values == "train"]
    median = np.nanmedian(train, axis=0) if len(train) else np.nanmedian(y_true, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    return np.tile(median, (len(y_true), 1))


def final_step_features(dataset: dict[str, Any]) -> np.ndarray:
    return np.asarray(dataset["X"], dtype=np.float64)[:, -1, :]


def fit_ridge_predictions(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, alpha: float) -> np.ndarray:
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_aug = np.c_[np.ones(len(x_train)), x_train]
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.pinv(x_aug.T @ x_aug + reg) @ x_aug.T @ y_train
    return np.c_[np.ones(len(x)), x] @ coef


def fit_elastic_net_predictions(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
    l1_ratio: float = 0.25,
    iterations: int = 250,
    lr: float = 0.02,
) -> np.ndarray:
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_std = np.where(np.isfinite(x_std) & (x_std > 1e-12), x_std, 1.0)
    xs_train = (x_train - x_mean) / x_std
    xs_all = (x - x_mean) / x_std
    xs_train = np.where(np.isfinite(xs_train), xs_train, 0.0)
    xs_all = np.where(np.isfinite(xs_all), xs_all, 0.0)
    y_mean = np.mean(y_train, axis=0)
    centered = y_train - y_mean
    weights = np.zeros((xs_train.shape[1], y_train.shape[1]), dtype=np.float64)
    n = max(1, len(xs_train))
    for _ in range(iterations):
        pred = xs_train @ weights
        grad = (xs_train.T @ (pred - centered)) / n
        grad += alpha * (1.0 - l1_ratio) * weights
        grad += alpha * l1_ratio * np.sign(weights)
        weights -= lr * grad
    return xs_all @ weights + y_mean


def vector_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return safe_float(path_metric_row(actual, pred).get("rmse"))


def build_locked_cpu_reference(dataset: dict[str, Any], dataset_meta: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    y = np.asarray(dataset["y"], dtype=np.float64)
    splits = dataset["splits"].astype(str)
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    x = final_step_features(dataset)
    predictions: dict[str, np.ndarray] = {
        "training_mean_baseline": train_mean_baseline(y, splits),
        "training_median_baseline": train_median_baseline(y, splits),
    }
    ridge_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    elastic_alphas = [0.001, 0.01, 0.1]
    for alpha in ridge_alphas:
        predictions[f"ridge_multi_output_alpha_{alpha:g}"] = fit_ridge_predictions(x, y, train_mask, alpha)
    for alpha in elastic_alphas:
        predictions[f"elastic_net_multi_output_alpha_{alpha:g}"] = fit_elastic_net_predictions(x, y, train_mask, alpha)
    rows: list[dict[str, Any]] = []
    for name, pred in predictions.items():
        for split in ["train", "validation", "untouched_holdout"]:
            mask = splits == split
            rows.append(
                {
                    **dataset_meta,
                    "cpu_model": name,
                    "split": split,
                    "selection_eligible": split in {"train", "validation"},
                    "rows": int(mask.sum()),
                    "vector_rmse": vector_rmse(y[mask], pred[mask]),
                    "vector_mae": safe_float(path_metric_row(y[mask], pred[mask]).get("mae")),
                    "holdout_used_for_selection": False,
                }
            )
            for idx, column in enumerate(dataset["target_columns"]):
                horizon = dataset["horizon_buckets"][idx] if idx < len(dataset["horizon_buckets"]) else idx + 1
                metrics = path_metric_row(y[mask, idx], pred[mask, idx])
                rows.append(
                    {
                        **dataset_meta,
                        "cpu_model": name,
                        "split": split,
                        "selection_eligible": split in {"train", "validation"},
                        "target_name": column,
                        "target_side": "low" if "low" in str(column) else "",
                        "horizon_buckets": horizon,
                        "rows": int(mask.sum()),
                        "vector_rmse": math.nan,
                        "vector_mae": math.nan,
                        "output_rmse": metrics.get("rmse", math.nan),
                        "output_mae": metrics.get("mae", math.nan),
                        "holdout_used_for_selection": False,
                    }
                )
    metrics_frame = pd.DataFrame(rows)
    validation_vectors = metrics_frame[
        metrics_frame["split"].astype(str).eq("validation") & metrics_frame["target_name"].isna()
    ].copy()
    selected_row = validation_vectors.sort_values("vector_rmse", ascending=True).iloc[0]
    selected_name = str(selected_row["cpu_model"])
    cpu_dir = output_dir / "locked_cpu_reference"
    cpu_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "rawseq_low_path_locked_cpu_reference.csv"
    predictions_path = output_dir / "rawseq_low_path_locked_cpu_predictions.npz"
    contract_path = output_dir / "rawseq_low_path_locked_cpu_contract.json"
    metrics_frame.to_csv(metrics_path, index=False)
    feature_scaler_mean = np.asarray(dataset.get("feature_scaler_mean", []), dtype=np.float32)
    feature_scaler_std = np.asarray(dataset.get("feature_scaler_std", []), dtype=np.float32)
    target_scaler_mean = np.asarray(dataset.get("target_scaler_mean", []), dtype=np.float32)
    target_scaler_std = np.asarray(dataset.get("target_scaler_std", []), dtype=np.float32)
    prediction_hashes = {
        name: ndarray_sha256(pred.astype(np.float32))
        for name, pred in predictions.items()
    }
    cpu_model_hashes = {
        name: stable_hash(
            {
                "cpu_model": name,
                "cpu_feature_view": "last_sequence_step",
                "prediction_sha256": prediction_hashes[name],
                "target_columns_sha256": dataset_meta.get("target_columns_sha256", ""),
            }
        )
        for name in predictions
    }
    np.savez_compressed(
        predictions_path,
        **{safe_filename(name): pred.astype(np.float32) for name, pred in predictions.items()},
        selected_cpu_prediction=predictions[selected_name].astype(np.float32),
        selected_cpu_model=np.asarray(selected_name, dtype=str),
        target_columns=np.asarray(dataset["target_columns"], dtype=str),
        horizon_buckets=np.asarray(dataset["horizon_buckets"], dtype=np.int64),
        splits=splits,
        decision_timestamps=np.asarray(dataset["decision_timestamps"], dtype=np.float64),
        feature_scaler_mean=feature_scaler_mean,
        feature_scaler_std=feature_scaler_std,
        target_scaler_mean=target_scaler_mean,
        target_scaler_std=target_scaler_std,
    )
    contract = {
        **dataset_meta,
        "cpu_feature_view": "last_sequence_step",
        "cpu_models": list(predictions.keys()),
        "cpu_model_hashes": cpu_model_hashes,
        "cpu_prediction_array_sha256": prediction_hashes,
        "selected_cpu_model": selected_name,
        "selected_cpu_model_hash": cpu_model_hashes[selected_name],
        "selected_cpu_prediction_sha256": prediction_hashes[selected_name],
        "selected_by": "validation_vector_rmse",
        "selected_validation_vector_rmse": safe_float(selected_row.get("vector_rmse")),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "metrics_sha256": file_sha256(metrics_path),
        "predictions_sha256": file_sha256(predictions_path),
        "feature_scaler_state_present": bool(feature_scaler_mean.size and feature_scaler_std.size),
        "target_scaler_state_present": bool(target_scaler_mean.size and target_scaler_std.size),
        "feature_scaler_mean_shape": list(feature_scaler_mean.shape),
        "feature_scaler_std_shape": list(feature_scaler_std.shape),
        "target_scaler_mean_shape": list(target_scaler_mean.shape),
        "target_scaler_std_shape": list(target_scaler_std.shape),
        "feature_scaler_mean_sha256": ndarray_sha256(feature_scaler_mean) if feature_scaler_mean.size else "",
        "feature_scaler_std_sha256": ndarray_sha256(feature_scaler_std) if feature_scaler_std.size else "",
        "target_scaler_mean_sha256": ndarray_sha256(target_scaler_mean) if target_scaler_mean.size else "",
        "target_scaler_std_sha256": ndarray_sha256(target_scaler_std) if target_scaler_std.size else "",
        "dataset_contract": dataset.get("dataset_contract", {}),
        "target_columns": dataset["target_columns"],
        "horizon_buckets": dataset["horizon_buckets"],
        "holdout_used_for_selection": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    contract["cpu_contract_hash"] = stable_hash(contract)
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return {
        "selected_name": selected_name,
        "selected_prediction": predictions[selected_name],
        "metrics": metrics_frame,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "contract_path": contract_path,
        "contract": contract,
    }


def evidence_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    evidence_type: str,
    selected_cpu_baseline_name: str = "training_mean_baseline",
) -> dict[str, Any]:
    metric_actual = -actual if evidence_type == "low" else actual
    metric_pred = -pred if evidence_type == "low" else pred
    metric_baseline = -baseline if evidence_type == "low" else baseline
    model_metrics = path_metric_row(metric_actual, metric_pred)
    baseline_metrics = path_metric_row(metric_actual, metric_baseline)
    base_rmse = safe_float(baseline_metrics.get("rmse"))
    model_rmse = safe_float(model_metrics.get("rmse"))
    improvement = (base_rmse - model_rmse) / base_rmse if math.isfinite(base_rmse) and base_rmse > 0.0 and math.isfinite(model_rmse) else math.nan
    cal_slope, cal_intercept = calibration(metric_actual, metric_pred)
    top_recall, top_precision = top_decile_recall_precision(metric_actual, metric_pred)
    pred_unique_fraction = unique_fraction(metric_pred)
    canonical_target_mean = finite_mean(actual)
    canonical_target_std = finite_std(actual)
    canonical_prediction_mean = finite_mean(pred)
    canonical_prediction_std = finite_std(pred)
    drawdown_target = -actual if evidence_type == "low" else np.full_like(np.asarray(actual, dtype=np.float64), np.nan)
    drawdown_prediction = -pred if evidence_type == "low" else np.full_like(np.asarray(pred, dtype=np.float64), np.nan)
    row = {
        **model_metrics,
        "model_rmse": model_metrics.get("rmse", math.nan),
        "model_mae": model_metrics.get("mae", math.nan),
        "canonical_target_mean": canonical_target_mean,
        "canonical_target_std": canonical_target_std,
        "canonical_prediction_mean": canonical_prediction_mean,
        "canonical_prediction_std": canonical_prediction_std,
        "drawdown_target_mean": finite_mean(drawdown_target),
        "drawdown_target_std": finite_std(drawdown_target),
        "drawdown_prediction_mean": finite_mean(drawdown_prediction),
        "drawdown_prediction_std": finite_std(drawdown_prediction),
        "target_mean": finite_mean(metric_actual),
        "target_std": finite_std(metric_actual),
        "prediction_mean": finite_mean(metric_pred),
        "prediction_std": finite_std(metric_pred),
        "selected_cpu_baseline_name": selected_cpu_baseline_name,
        "selected_cpu_baseline_rmse": base_rmse,
        "selected_cpu_baseline_mae": baseline_metrics.get("mae", math.nan),
        "rmse_improvement_vs_selected_cpu": improvement,
        "mae_improvement_vs_selected_cpu": (
            (safe_float(baseline_metrics.get("mae")) - safe_float(model_metrics.get("mae"))) / safe_float(baseline_metrics.get("mae"))
            if math.isfinite(safe_float(baseline_metrics.get("mae"))) and safe_float(baseline_metrics.get("mae")) > 0.0 and math.isfinite(safe_float(model_metrics.get("mae")))
            else math.nan
        ),
        "pearson_correlation": pearson_corr(metric_actual, metric_pred),
        "spearman_correlation": spearman_corr(metric_actual, metric_pred),
        "prediction_std_ratio": std_ratio(metric_actual, metric_pred),
        "prediction_std_to_target_std": std_ratio(metric_actual, metric_pred),
        "calibration_slope": cal_slope,
        "calibration_intercept": cal_intercept,
        "prediction_unique_fraction": pred_unique_fraction,
        "constant_prediction_flag": bool(math.isfinite(pred_unique_fraction) and pred_unique_fraction <= 0.01),
        "top_decile_drawdown_recall": top_recall if evidence_type == "low" else math.nan,
        "top_decile_drawdown_precision": top_precision if evidence_type == "low" else math.nan,
        "low_reported_as_drawdown_magnitude": evidence_type == "low",
        "train_mean_baseline_rmse": base_rmse,
        "train_mean_baseline_mae": baseline_metrics.get("mae", math.nan),
        "baseline_improvement_fraction": improvement,
    }
    if evidence_type == "high":
        row.update(
            {
                "terminal_high_correlation": terminal_corr(actual, pred),
                "high_sign_violation_fraction": float(np.nanmean(pred < -1e-9)) if pred.size else math.nan,
                "high_monotonic_violation_fraction": monotonic_violation_fraction(pred, "nondecreasing"),
            }
        )
    elif evidence_type == "low":
        row.update(
            {
                "terminal_low_correlation": terminal_corr(actual, pred),
                "low_sign_violation_fraction": float(np.nanmean(pred > 1e-9)) if pred.size else math.nan,
                "low_monotonic_violation_fraction": monotonic_violation_fraction(pred, "nonincreasing"),
            }
        )
    return row


def build_target_lane_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    splits: np.ndarray,
    horizon_buckets: list[int],
    target_columns: list[str],
    target_lane: str,
    target_layout: str,
    dataset_meta: dict[str, Any],
    model_kind: str,
    selected_cpu_prediction: np.ndarray | None = None,
    selected_cpu_baseline_name: str = "training_mean_baseline",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_values = splits.astype(str)
    baseline = (
        np.asarray(selected_cpu_prediction, dtype=np.float64)
        if selected_cpu_prediction is not None
        else train_mean_baseline(y_true, split_values)
    )
    indices = split_target_indices(target_columns)
    for split in ["train", "validation", "untouched_holdout"]:
        mask = split_values == split
        if target_lane in {"future_return_path", "coarse_return_vector"}:
            for horizon_idx in range(y_true.shape[1]):
                horizon = horizon_buckets[horizon_idx] if horizon_idx < len(horizon_buckets) else horizon_idx + 1
                base = metric_row(y_true[mask, horizon_idx], baseline[mask, horizon_idx])
                metrics = metric_row(y_true[mask, horizon_idx], y_pred[mask, horizon_idx])
                base_rmse = safe_float(base.get("rmse"))
                model_rmse = safe_float(metrics.get("rmse"))
                rows.append(
                    {
                        **dataset_meta,
                        "model_kind": model_kind,
                        "split": split,
                        "target_lane": target_lane,
                        "target_layout": target_layout,
                        "evidence_type": "horizon",
                        "target_side": "return",
                        "target_name": target_columns[horizon_idx] if horizon_idx < len(target_columns) else "",
                        "target_column": target_columns[horizon_idx] if horizon_idx < len(target_columns) else "",
                        "horizon_index": horizon_idx,
                        "horizon_buckets": horizon,
                        **metrics,
                        "train_mean_baseline_rmse": base_rmse,
                        "train_mean_baseline_mae": base.get("mae", math.nan),
                        "baseline_improvement_fraction": (base_rmse - model_rmse) / base_rmse if math.isfinite(base_rmse) and base_rmse > 0.0 and math.isfinite(model_rmse) else math.nan,
                    }
                )
            continue
        evidence_plan: list[tuple[str, list[int]]] = []
        if target_lane == "future_high_from_now_bps_path":
            evidence_plan = [("high", indices["high"] or list(range(y_true.shape[1])))]
        elif target_lane == "future_low_from_now_bps_path":
            evidence_plan = [("low", indices["low"] or list(range(y_true.shape[1])))]
        elif target_lane == "future_range_envelope_path":
            evidence_plan = [("high", indices["high"]), ("low", indices["low"]), ("combined", indices["high"] + indices["low"])]
        for evidence_type, cols in evidence_plan:
            if not cols:
                continue
            actual = y_true[mask][:, cols]
            pred = y_pred[mask][:, cols]
            base = baseline[mask][:, cols]
            metrics = evidence_metrics(actual, pred, base, evidence_type, selected_cpu_baseline_name)
            if evidence_type == "combined" and target_lane == "future_range_envelope_path":
                high_cols = indices["high"]
                low_cols = indices["low"]
                width = min(len(high_cols), len(low_cols))
                if width:
                    high_pred = y_pred[mask][:, high_cols[:width]]
                    low_pred = y_pred[mask][:, low_cols[:width]]
                    metrics["envelope_order_violation_fraction"] = float(np.nanmean(high_pred < low_pred - 1e-9))
                else:
                    metrics["envelope_order_violation_fraction"] = math.nan
            rows.append(
                {
                    **dataset_meta,
                    "model_kind": model_kind,
                    "split": split,
                    "target_lane": target_lane,
                    "target_layout": target_layout,
                    "evidence_type": evidence_type,
                    "target_side": evidence_type,
                    "target_name": evidence_type if evidence_type == "combined" else ";".join([target_columns[idx] for idx in cols]),
                    "target_column": ";".join([target_columns[idx] for idx in cols]),
                    "horizon_index": "",
                    "horizon_buckets": ",".join(str(horizon_buckets[idx]) for idx in cols if idx < len(horizon_buckets)),
                    **metrics,
                }
            )
            if target_lane == "future_low_from_now_bps_path" and evidence_type == "low":
                for col_idx in cols:
                    horizon = horizon_buckets[col_idx] if col_idx < len(horizon_buckets) else col_idx + 1
                    col_metrics = evidence_metrics(
                        y_true[mask][:, [col_idx]],
                        y_pred[mask][:, [col_idx]],
                        baseline[mask][:, [col_idx]],
                        "low",
                        selected_cpu_baseline_name,
                    )
                    rows.append(
                        {
                            **dataset_meta,
                            "model_kind": model_kind,
                            "split": split,
                            "target_lane": target_lane,
                            "target_layout": target_layout,
                            "evidence_type": "horizon",
                            "target_side": "low",
                            "target_name": target_columns[col_idx],
                            "target_column": target_columns[col_idx],
                            "horizon_index": col_idx,
                            "horizon_buckets": horizon,
                            **col_metrics,
                        }
                    )
    return pd.DataFrame(rows)


def apply_validation_target_lane_status(summary: pd.DataFrame, target_metrics: pd.DataFrame, min_improvement: float) -> pd.DataFrame:
    out = summary.copy()
    for column, default in [
        ("target_lane_status", ""),
        ("validation_target_guard_pass", False),
        ("validation_target_guard_reason", ""),
        ("weakest_validation_baseline_improvement", math.nan),
        ("validation_high_baseline_improvement", math.nan),
        ("validation_low_baseline_improvement", math.nan),
        ("validation_combined_baseline_improvement", math.nan),
    ]:
        if column not in out.columns:
            out[column] = default
    if target_metrics.empty:
        return out
    key_cols = ["dataset_index", "feature_group", "seq_len", "model_kind", "seed"]
    validation = target_metrics[target_metrics["split"].astype(str).eq("validation")].copy()
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, group in validation.groupby(key_cols, dropna=False):
        lane = str(group["target_lane"].dropna().iloc[0]) if "target_lane" in group and not group.empty else "future_return_path"
        improvements = pd.to_numeric(group["baseline_improvement_fraction"], errors="coerce")
        if lane == "future_range_envelope_path":
            required = {"high", "low", "combined"}
            by_type = {str(row["evidence_type"]): safe_float(row.get("baseline_improvement_fraction")) for _, row in group.iterrows()}
            missing = sorted(required - set(by_type))
            pass_flags = [math.isfinite(by_type.get(name, math.nan)) and by_type[name] >= min_improvement for name in required]
            passed = not missing and all(pass_flags)
            reason = "validation_high_low_combined_pass" if passed else "fails_validation_high_low_combined_guard"
            weakest = min([by_type.get(name, math.nan) for name in required if math.isfinite(by_type.get(name, math.nan))], default=math.nan)
            by_key[key if isinstance(key, tuple) else (key,)] = {
                "target_lane_status": "validation_target_lane_survivor" if passed else "fails_validation_target_lane_guard",
                "validation_target_guard_pass": passed,
                "validation_target_guard_reason": reason,
                "weakest_validation_baseline_improvement": weakest,
                "validation_high_baseline_improvement": by_type.get("high", math.nan),
                "validation_low_baseline_improvement": by_type.get("low", math.nan),
                "validation_combined_baseline_improvement": by_type.get("combined", math.nan),
            }
        elif lane == "future_low_from_now_bps_path":
            vector_rows = group[group["evidence_type"].astype(str).eq("low")]
            horizon_rows = group[group["evidence_type"].astype(str).eq("horizon")]
            vector_row = vector_rows.iloc[0] if not vector_rows.empty else pd.Series(dtype=object)
            vector_improvement = safe_float(vector_row.get("baseline_improvement_fraction"))
            horizon_improvements = pd.to_numeric(horizon_rows.get("baseline_improvement_fraction"), errors="coerce")
            positive_horizons = int((horizon_improvements > 0.0).sum()) if len(horizon_improvements) else 0
            median_horizon_improvement = safe_float(horizon_improvements.median()) if len(horizon_improvements) else math.nan
            spearman = safe_float(vector_row.get("spearman_correlation"))
            unique = safe_float(vector_row.get("prediction_unique_fraction"))
            constant = bool(vector_row.get("constant_prediction_flag")) if "constant_prediction_flag" in vector_row else True
            mono = safe_float(vector_row.get("low_monotonic_violation_fraction"), 1.0)
            passed = bool(
                math.isfinite(vector_improvement)
                and vector_improvement >= min_improvement
                and positive_horizons >= 3
                and math.isfinite(spearman)
                and spearman > 0.0
                and math.isfinite(unique)
                and unique > 0.01
                and not constant
                and math.isfinite(mono)
                and mono <= 0.05
            )
            failed = []
            if not (math.isfinite(vector_improvement) and vector_improvement >= min_improvement):
                failed.append("vector_rmse_not_above_locked_cpu")
            if positive_horizons < 3:
                failed.append("fewer_than_3_horizons_beat_locked_cpu")
            if not (math.isfinite(spearman) and spearman > 0.0):
                failed.append("spearman_not_positive")
            if not (math.isfinite(unique) and unique > 0.01) or constant:
                failed.append("constant_or_near_constant_prediction")
            if not (math.isfinite(mono) and mono <= 0.05):
                failed.append("low_path_monotonicity_violation")
            by_key[key if isinstance(key, tuple) else (key,)] = {
                "target_lane_status": "validation_target_lane_survivor" if passed else "fails_validation_target_lane_guard",
                "validation_target_guard_pass": passed,
                "validation_target_guard_reason": "low_path_locked_cpu_guard_pass" if passed else ";".join(failed),
                "weakest_validation_baseline_improvement": min(
                    [x for x in [vector_improvement, median_horizon_improvement] if math.isfinite(x)],
                    default=math.nan,
                ),
                "validation_low_baseline_improvement": vector_improvement,
                "validation_low_positive_horizon_count": positive_horizons,
                "validation_low_median_horizon_improvement": median_horizon_improvement,
                "validation_low_spearman_correlation": spearman,
                "validation_low_prediction_unique_fraction": unique,
                "validation_low_monotonic_violation_fraction": mono,
            }
        else:
            finite_improvements = improvements[np.isfinite(improvements)]
            weakest = float(finite_improvements.min()) if len(finite_improvements) else math.nan
            passed = bool(len(finite_improvements) > 0 and np.all(finite_improvements >= min_improvement))
            by_key[key if isinstance(key, tuple) else (key,)] = {
                "target_lane_status": "validation_target_lane_survivor" if passed else "fails_validation_target_lane_guard",
                "validation_target_guard_pass": passed,
                "validation_target_guard_reason": "validation_metric_pass" if passed else "fails_validation_metric_guard",
                "weakest_validation_baseline_improvement": weakest,
            }
    for idx, row in out.iterrows():
        key = tuple(row.get(col) for col in key_cols)
        metrics = by_key.get(key)
        if metrics:
            for name, value in metrics.items():
                out.at[idx, name] = value
            lane = str(row.get("target_lane", "future_return_path"))
            if lane != "future_return_path":
                out.at[idx, "sequence_model_status"] = metrics["target_lane_status"]
    return out


def max_dip(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan
    cumulative = np.cumsum(arr)
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def infer_bucket_seconds(timestamps: np.ndarray, fallback: float = 10.0) -> float:
    ts = np.asarray(timestamps, dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    if len(ts) < 2:
        return fallback
    diffs = np.diff(np.sort(ts))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if len(diffs) == 0:
        return fallback
    return float(np.median(diffs) / 1000.0)


def select_policy_signal(pred: np.ndarray, threshold: float, policy: str) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(pred, dtype=np.float64)
    normalized = str(policy).strip().lower()
    if normalized in {"inverse_gt", "inverse", "short_gt"}:
        selected = values > threshold
        direction = np.full_like(values, -1.0, dtype=np.float64)
        multiplier = -1.0
    elif normalized in {"direct_abs", "abs_gt", "long_short_abs"}:
        selected = np.abs(values) > threshold
        direction = np.sign(values)
        multiplier = math.nan
    else:
        selected = values > threshold
        direction = np.ones_like(values, dtype=np.float64)
        multiplier = 1.0
    return selected & np.isfinite(values), direction, multiplier


def non_overlapping_values(timestamps: np.ndarray, selected: np.ndarray, net: np.ndarray, horizon_ms: float) -> np.ndarray:
    order = np.argsort(timestamps)
    values = []
    next_allowed = -math.inf
    for idx in order:
        ts = safe_float(timestamps[idx])
        if not bool(selected[idx]) or not math.isfinite(ts) or ts < next_allowed:
            continue
        value = safe_float(net[idx])
        if math.isfinite(value):
            values.append(value)
            next_allowed = ts + horizon_ms
    return np.asarray(values, dtype=np.float64)


def policy_metrics_for_horizon(
    actual: np.ndarray,
    pred: np.ndarray,
    timestamps: np.ndarray,
    horizon: int,
    bucket_seconds: float,
    threshold: float,
    cost_bps: float,
    policy: str,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    mask = np.isfinite(actual) & np.isfinite(pred)
    selected, direction, multiplier = select_policy_signal(pred, threshold, policy)
    selected = selected & mask
    gross = direction * actual
    net = gross - cost_bps
    selected_net = net[selected]
    selected_gross = gross[selected]
    selected_cost = np.full_like(selected_net, cost_bps, dtype=np.float64)
    profitable_opp = actual > cost_bps if str(policy).strip().lower() not in {"inverse_gt", "inverse", "short_gt"} else -actual > cost_bps
    selected_profitable_opp = selected & (gross > cost_bps)
    horizon_ms = horizon * bucket_seconds * 1000.0
    non_overlap = non_overlapping_values(timestamps, selected, net, horizon_ms)
    finite_timestamps = timestamps[np.isfinite(timestamps)]
    span_ms = float(np.nanmax(finite_timestamps) - np.nanmin(finite_timestamps)) if len(finite_timestamps) else math.nan
    exposure_fraction = min(len(non_overlap) * horizon_ms / max(span_ms, 1.0), 1.0) if math.isfinite(span_ms) and span_ms > 0 else math.nan
    return {
        "policy": policy,
        "threshold_bps": threshold,
        "cost_bps": cost_bps,
        "policy_direction_multiplier": multiplier,
        "rows": int(mask.sum()),
        "selected_signal_count": int(selected.sum()),
        "selected_signal_precision_win_rate": float(np.mean(selected_net > 0.0)) if len(selected_net) else math.nan,
        "recall_profitable_opportunities": float(selected_profitable_opp.sum() / profitable_opp.sum()) if profitable_opp.any() else math.nan,
        "avg_selected_gross_bps": float(np.mean(selected_gross)) if len(selected_gross) else math.nan,
        "median_selected_gross_bps": float(np.median(selected_gross)) if len(selected_gross) else math.nan,
        "avg_selected_cost_bps": float(np.mean(selected_cost)) if len(selected_cost) else math.nan,
        "avg_selected_net_bps": float(np.mean(selected_net)) if len(selected_net) else math.nan,
        "median_selected_net_bps": float(np.median(selected_net)) if len(selected_net) else math.nan,
        "row_signal_cum_net_bps": float(np.sum(selected_net)) if len(selected_net) else 0.0,
        "row_signal_max_drawdown_bps": max_dip(selected_net),
        "non_overlapping_trade_count": int(len(non_overlap)),
        "non_overlapping_avg_net_bps": float(np.mean(non_overlap)) if len(non_overlap) else math.nan,
        "non_overlapping_cum_net_bps": float(np.sum(non_overlap)) if len(non_overlap) else 0.0,
        "non_overlapping_win_rate": float(np.mean(non_overlap > 0.0)) if len(non_overlap) else math.nan,
        "non_overlapping_max_drawdown_bps": max_dip(non_overlap),
        "position_trade_count": int(len(non_overlap)),
        "position_cum_net_bps": float(np.sum(non_overlap)) if len(non_overlap) else 0.0,
        "position_avg_net_bps": float(np.mean(non_overlap)) if len(non_overlap) else math.nan,
        "position_win_rate": float(np.mean(non_overlap > 0.0)) if len(non_overlap) else math.nan,
        "position_max_drawdown_bps": max_dip(non_overlap),
        "position_exposure_fraction": exposure_fraction,
        "position_average_hold_buckets": horizon if len(non_overlap) else 0,
        "position_turnover": float(len(non_overlap) / max(int(mask.sum()), 1)),
        "position_max_concurrent_exposure": 1 if len(non_overlap) else 0,
    }


def build_policy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    splits: np.ndarray,
    timestamps: np.ndarray,
    horizon_buckets: list[int],
    dataset_meta: dict[str, Any],
    model_kind: str,
    thresholds: list[float],
    costs: list[float],
    policy: str,
    bucket_seconds: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_values = splits.astype(str)
    for split in ["train", "validation", "untouched_holdout"]:
        split_mask = split_values == split
        for horizon_idx in range(y_true.shape[1]):
            horizon = horizon_buckets[horizon_idx] if horizon_idx < len(horizon_buckets) else horizon_idx + 1
            for threshold in thresholds:
                for cost in costs:
                    metrics = policy_metrics_for_horizon(
                        y_true[split_mask, horizon_idx],
                        y_pred[split_mask, horizon_idx],
                        timestamps[split_mask],
                        int(horizon),
                        bucket_seconds,
                        threshold,
                        cost,
                        policy,
                    )
                    rows.append(
                        {
                            **dataset_meta,
                            "model_kind": model_kind,
                            "split": split,
                            "horizon_index": horizon_idx,
                            "horizon_buckets": int(horizon),
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def select_validation_policy_metrics(policy_metrics: pd.DataFrame, decision_cost_bps: float) -> pd.DataFrame:
    if policy_metrics.empty:
        return pd.DataFrame()
    validation = policy_metrics[
        policy_metrics["split"].astype(str).eq("validation")
        & np.isclose(pd.to_numeric(policy_metrics["cost_bps"], errors="coerce"), decision_cost_bps)
    ].copy()
    holdout = policy_metrics[
        policy_metrics["split"].astype(str).eq("untouched_holdout")
        & np.isclose(pd.to_numeric(policy_metrics["cost_bps"], errors="coerce"), decision_cost_bps)
    ].copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["dataset_index", "feature_group", "seq_len", "model_kind", "seed", "horizon_buckets"]
    for key, group in validation.groupby(group_cols, dropna=False):
        selected = group.sort_values(
            ["position_cum_net_bps", "position_trade_count", "selected_signal_count"],
            ascending=[False, False, False],
            na_position="last",
        ).iloc[0]
        holdout_match = holdout[
            (holdout["dataset_index"].astype(str) == str(selected.get("dataset_index")))
            & (holdout["feature_group"].astype(str) == str(selected.get("feature_group")))
            & (holdout["seq_len"].astype(str) == str(selected.get("seq_len")))
            & (holdout["model_kind"].astype(str) == str(selected.get("model_kind")))
            & (holdout["seed"].astype(str) == str(selected.get("seed")))
            & (holdout["horizon_buckets"].astype(float) == float(selected.get("horizon_buckets")))
            & (holdout["threshold_bps"].astype(float) == float(selected.get("threshold_bps")))
        ]
        holdout_row = holdout_match.iloc[0] if not holdout_match.empty else pd.Series(dtype=object)
        rows.append(
            {
                "dataset_index": selected.get("dataset_index"),
                "feature_group": selected.get("feature_group"),
                "seq_len": selected.get("seq_len"),
                "model_kind": selected.get("model_kind"),
                "seed": selected.get("seed"),
                "horizon_buckets": selected.get("horizon_buckets"),
                "decision_cost_bps": decision_cost_bps,
                "selected_threshold_bps": selected.get("threshold_bps"),
                "selection_stage": "validation_selected_policy",
                "validation_position_trade_count": selected.get("position_trade_count"),
                "validation_position_cum_net_bps": selected.get("position_cum_net_bps"),
                "validation_non_overlapping_cum_net_bps": selected.get("non_overlapping_cum_net_bps"),
                "validation_selected_signal_precision_win_rate": selected.get("selected_signal_precision_win_rate"),
                "holdout_stage": "untouched_holdout_final_selected_policy",
                "holdout_position_trade_count": holdout_row.get("position_trade_count", math.nan),
                "holdout_position_cum_net_bps": holdout_row.get("position_cum_net_bps", math.nan),
                "holdout_non_overlapping_cum_net_bps": holdout_row.get("non_overlapping_cum_net_bps", math.nan),
                "holdout_selected_signal_precision_win_rate": holdout_row.get("selected_signal_precision_win_rate", math.nan),
                "holdout_position_max_drawdown_bps": holdout_row.get("position_max_drawdown_bps", math.nan),
                "holdout_position_exposure_fraction": holdout_row.get("position_exposure_fraction", math.nan),
            }
        )
    return pd.DataFrame(rows)


def apply_policy_status_to_summary(summary: pd.DataFrame, selected_policy: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for column, default in [
        ("policy_horizons_evaluated", 0),
        ("policy_horizons_holdout_positive", 0),
        ("best_holdout_position_cum_net_bps", math.nan),
        ("total_holdout_position_cum_net_bps", 0.0),
        ("best_holdout_position_trade_count", 0),
    ]:
        if column not in out.columns:
            out[column] = default
    if selected_policy.empty:
        return out
    group_cols = ["dataset_index", "feature_group", "seq_len", "model_kind", "seed"]
    by_key = {}
    for key, group in selected_policy.groupby(group_cols, dropna=False):
        holdout_cum = pd.to_numeric(group["holdout_position_cum_net_bps"], errors="coerce")
        holdout_trades = pd.to_numeric(group["holdout_position_trade_count"], errors="coerce")
        by_key[key if isinstance(key, tuple) else (key,)] = {
            "policy_horizons_evaluated": int(len(group)),
            "policy_horizons_holdout_positive": int((holdout_cum > 0.0).sum()),
            "best_holdout_position_cum_net_bps": safe_float(holdout_cum.max()),
            "total_holdout_position_cum_net_bps": safe_float(holdout_cum.sum(), 0.0),
            "best_holdout_position_trade_count": int(safe_float(holdout_trades.max(), 0.0)),
        }
    for idx, row in out.iterrows():
        key = tuple(row.get(col) for col in group_cols)
        metrics = by_key.get(key)
        if metrics:
            for name, value in metrics.items():
                out.at[idx, name] = value
    return out


def split_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    splits: np.ndarray,
    horizon_buckets: list[int],
    dataset_meta: dict[str, Any],
    model_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "untouched_holdout"]:
        mask = splits.astype(str) == split
        combined = metric_row(y_true[mask].reshape(-1), y_pred[mask].reshape(-1))
        summary_rows.append(
            {
                **dataset_meta,
                "model_kind": model_kind,
                "split": split,
                "combined_rows": combined["rows"],
                "combined_mae": combined["mae"],
                "combined_rmse": combined["rmse"],
                "combined_correlation": combined["correlation"],
                "combined_directional_accuracy": combined["directional_accuracy"],
                "combined_balanced_accuracy": combined["balanced_accuracy"],
            }
        )
        for horizon_idx in range(y_true.shape[1]):
            horizon = horizon_buckets[horizon_idx] if horizon_idx < len(horizon_buckets) else horizon_idx + 1
            metrics = metric_row(y_true[mask, horizon_idx], y_pred[mask, horizon_idx])
            horizon_rows.append(
                {
                    **dataset_meta,
                    "model_kind": model_kind,
                    "split": split,
                    "horizon_index": horizon_idx,
                    "horizon_buckets": horizon,
                    **metrics,
                }
            )
    return summary_rows, horizon_rows


def default_baseline_leaderboard_path(sequence_manifest_path: Path) -> Path:
    return sequence_manifest_path.parent / "combined_leaderboard.csv"


def empty_baseline_reference() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "horizon_buckets",
            "best_baseline_model_by_rmse",
            "best_baseline_holdout_rmse",
            "best_baseline_directional_model",
            "best_baseline_holdout_directional_accuracy",
            "best_baseline_position_model",
            "best_baseline_holdout_position_cum_net_bps",
            "baseline_reference_status",
        ]
    )


def build_baseline_reference(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty or "base_model" not in leaderboard.columns:
        return empty_baseline_reference()
    frame = leaderboard[leaderboard["base_model"].astype(str).isin(BASELINE_BASE_MODELS)].copy()
    if frame.empty:
        return empty_baseline_reference()
    rows: list[dict[str, Any]] = []
    for horizon, group in frame.groupby("horizon_buckets", dropna=False):
        finite_rmse = group[pd.to_numeric(group.get("holdout_rmse"), errors="coerce").apply(np.isfinite)].copy()
        finite_dir = group[
            pd.to_numeric(group.get("holdout_directional_accuracy"), errors="coerce").apply(np.isfinite)
        ].copy()
        finite_pos = group[
            pd.to_numeric(group.get("holdout_position_cum_net_bps"), errors="coerce").apply(np.isfinite)
        ].copy()
        rmse_row = finite_rmse.sort_values("holdout_rmse", ascending=True).head(1)
        dir_row = finite_dir.sort_values("holdout_directional_accuracy", ascending=False).head(1)
        pos_row = finite_pos.sort_values("holdout_position_cum_net_bps", ascending=False).head(1)
        rows.append(
            {
                "horizon_buckets": horizon,
                "best_baseline_model_by_rmse": rmse_row.iloc[0].get("model", "") if not rmse_row.empty else "",
                "best_baseline_holdout_rmse": safe_float(rmse_row.iloc[0].get("holdout_rmse")) if not rmse_row.empty else math.nan,
                "best_baseline_directional_model": dir_row.iloc[0].get("model", "") if not dir_row.empty else "",
                "best_baseline_holdout_directional_accuracy": safe_float(dir_row.iloc[0].get("holdout_directional_accuracy"))
                if not dir_row.empty
                else math.nan,
                "best_baseline_position_model": pos_row.iloc[0].get("model", "") if not pos_row.empty else "",
                "best_baseline_holdout_position_cum_net_bps": safe_float(pos_row.iloc[0].get("holdout_position_cum_net_bps"))
                if not pos_row.empty
                else math.nan,
                "baseline_reference_status": "ok" if not rmse_row.empty else "missing_finite_holdout_rmse",
            }
        )
    return pd.DataFrame(rows)


def load_baseline_reference(sequence_manifest_path: Path, explicit_path: str = "") -> tuple[pd.DataFrame, str, str]:
    path = resolve_path(explicit_path) if explicit_path else default_baseline_leaderboard_path(sequence_manifest_path)
    if not path.exists():
        return empty_baseline_reference(), str(path), "missing_baseline_leaderboard"
    try:
        leaderboard = pd.read_csv(path)
        reference = build_baseline_reference(leaderboard)
    except Exception as exc:
        reference = empty_baseline_reference()
        reference["baseline_reference_status"] = f"failed_to_read_baseline_leaderboard:{exc}"
        return reference, str(path), "failed_to_read_baseline_leaderboard"
    status = "ok" if not reference.empty and (reference["baseline_reference_status"].astype(str) == "ok").any() else "no_finite_baselines"
    return reference, str(path), status


def compare_horizon_metrics_to_baselines(
    horizon_metrics: pd.DataFrame,
    baseline_reference: pd.DataFrame,
    min_improvement_fraction: float,
    min_holdout_rows: int,
) -> pd.DataFrame:
    columns = [
        "dataset_index",
        "feature_group",
        "seq_len",
        "model_kind",
        "seed",
        "horizon_buckets",
        "rows",
        "rmse",
        "best_baseline_model_by_rmse",
        "best_baseline_holdout_rmse",
        "rmse_improvement_fraction_vs_best_baseline",
        "rmse_beats_best_baseline",
        "best_baseline_directional_model",
        "best_baseline_holdout_directional_accuracy",
        "directional_accuracy_delta_vs_best_baseline",
        "baseline_guard_pass",
        "baseline_guard_reason",
        "horizon_survivor_status",
    ]
    if horizon_metrics.empty:
        return pd.DataFrame(columns=columns)
    holdout = horizon_metrics[horizon_metrics["split"].astype(str).eq("untouched_holdout")].copy()
    if holdout.empty:
        return pd.DataFrame(columns=columns)
    if baseline_reference.empty:
        holdout["baseline_guard_pass"] = False
        holdout["baseline_guard_reason"] = "missing_baseline_reference"
        holdout["horizon_survivor_status"] = "missing_baseline_reference"
        return holdout.reindex(columns=columns)
    merged = holdout.merge(baseline_reference, on="horizon_buckets", how="left")
    improvements = []
    passes = []
    reasons = []
    statuses = []
    directional_deltas = []
    for _, row in merged.iterrows():
        model_rmse = safe_float(row.get("rmse"))
        baseline_rmse = safe_float(row.get("best_baseline_holdout_rmse"))
        rows = int(safe_float(row.get("rows"), 0))
        model_dir = safe_float(row.get("directional_accuracy"))
        baseline_dir = safe_float(row.get("best_baseline_holdout_directional_accuracy"))
        delta = model_dir - baseline_dir if math.isfinite(model_dir) and math.isfinite(baseline_dir) else math.nan
        directional_deltas.append(delta)
        if not math.isfinite(baseline_rmse) or baseline_rmse <= 0.0:
            improvement = math.nan
            passed = False
            reason = "missing_best_baseline_holdout_rmse"
        elif not math.isfinite(model_rmse):
            improvement = math.nan
            passed = False
            reason = "missing_model_holdout_rmse"
        else:
            improvement = (baseline_rmse - model_rmse) / baseline_rmse
            if rows < min_holdout_rows:
                passed = False
                reason = f"holdout_rows<{min_holdout_rows}"
            elif improvement >= min_improvement_fraction:
                passed = True
                reason = "beats_best_baseline_holdout_rmse"
            else:
                passed = False
                reason = "fails_best_baseline_holdout_rmse"
        improvements.append(improvement)
        passes.append(passed)
        reasons.append(reason)
        statuses.append("holdout_baseline_survivor" if passed else "fails_baseline_guard")
    merged["rmse_improvement_fraction_vs_best_baseline"] = improvements
    merged["rmse_beats_best_baseline"] = passes
    merged["directional_accuracy_delta_vs_best_baseline"] = directional_deltas
    merged["baseline_guard_pass"] = passes
    merged["baseline_guard_reason"] = reasons
    merged["horizon_survivor_status"] = statuses
    return merged.reindex(columns=columns)


def apply_baseline_status_to_summary(summary: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for column, default in [
        ("baseline_comparison_horizons", 0),
        ("holdout_horizons_beating_best_baseline", 0),
        ("best_rmse_improvement_fraction_vs_best_baseline", math.nan),
        ("all_holdout_horizons_beat_best_baseline", False),
        ("any_holdout_horizon_beats_best_baseline", False),
        ("sequence_model_status", ""),
    ]:
        if column not in out.columns:
            out[column] = default
    if comparison.empty:
        out["sequence_model_status"] = out.apply(
            lambda row: row.get("status", "") if row.get("status", "") != "ok" else "missing_baseline_comparison",
            axis=1,
        )
        return out
    key_cols = ["dataset_index", "feature_group", "seq_len", "model_kind", "seed"]
    grouped = comparison.groupby(key_cols, dropna=False)
    comparison_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, group in grouped:
        passes = group["baseline_guard_pass"].astype(bool)
        comparison_by_key[key if isinstance(key, tuple) else (key,)] = {
            "baseline_comparison_horizons": int(len(group)),
            "holdout_horizons_beating_best_baseline": int(passes.sum()),
            "best_rmse_improvement_fraction_vs_best_baseline": safe_float(
                pd.to_numeric(group["rmse_improvement_fraction_vs_best_baseline"], errors="coerce").max()
            ),
            "all_holdout_horizons_beat_best_baseline": bool(len(group) > 0 and passes.all()),
            "any_holdout_horizon_beats_best_baseline": bool(passes.any()),
        }
    statuses = []
    for idx, row in out.iterrows():
        key = tuple(row.get(col) for col in key_cols)
        status = str(row.get("status", ""))
        metrics = comparison_by_key.get(key)
        if metrics:
            for name, value in metrics.items():
                out.at[idx, name] = value
            if metrics["all_holdout_horizons_beat_best_baseline"]:
                statuses.append("multi_horizon_holdout_baseline_survivor")
            elif metrics["any_holdout_horizon_beats_best_baseline"]:
                statuses.append("single_horizon_holdout_baseline_survivor")
            else:
                statuses.append("fails_holdout_baseline_guard")
        elif status != "ok":
            statuses.append(status)
        else:
            statuses.append("missing_baseline_comparison")
    out["sequence_model_status"] = statuses
    return out


def no_torch_rows(
    datasets: list[dict[str, Any]],
    model_kinds: list[str],
    seeds: list[int],
    status: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    usable = [row for row in datasets if row.get("dataset_status") == "ok"]
    for dataset in usable:
        for model_kind in model_kinds:
            for seed in seeds:
                base = {
                    "dataset_index": dataset.get("dataset_index"),
                    "feature_group": dataset.get("feature_group", ""),
                    "seq_len": dataset.get("seq_len", ""),
                    "target_lane": dataset.get("target_lane", ""),
                    "target_layout": dataset.get("target_layout", ""),
                    "target_columns_sha256": dataset.get("target_columns_sha256", ""),
                    "dataset_path": dataset.get("dataset_path", ""),
                    "model_kind": model_kind,
                    "seed": seed,
                    **status,
                    "status": "skipped_torch_unavailable",
                    "failure": "torch_unavailable",
                    "paper_only": True,
                    "private_api": False,
                    "orders": False,
                    "promotion": False,
                    "champion_mutation": False,
                }
                summary_rows.append(base)
                status_rows.append(base)
    return summary_rows, status_rows


def load_npz_dataset(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        feature_columns = data["feature_columns"].astype(str).tolist()
        target_columns = data["target_columns"].astype(str).tolist()
        dataset_contract = parse_dataset_contract(data["dataset_contract"]) if "dataset_contract" in data.files else {}
        target_lane = str(data["target_lane"].item()) if "target_lane" in data.files else infer_target_lane(target_columns, dataset_contract)
        target_layout = str(data["target_layout"].item()) if "target_layout" in data.files else infer_target_layout(target_lane, dataset_contract)
        y = np.asarray(data["y"], dtype=np.float32)
        horizon_buckets = (
            data["horizon_buckets"].astype(int).tolist()
            if "horizon_buckets" in data.files
            else list(range(1, int(y.shape[1]) + 1))
        )
        validate_npz_target_contract(path, y, target_columns, horizon_buckets, target_lane, target_layout, dataset_contract)
        return {
            "X": np.asarray(data["X"], dtype=np.float32),
            "y": y,
            "splits": data["splits"].astype(str),
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "target_lane": target_lane,
            "target_layout": target_layout,
            "dataset_contract": dataset_contract,
            "horizon_buckets": horizon_buckets,
            "decision_timestamps": data["decision_timestamps"].astype(np.float64)
            if "decision_timestamps" in data.files
            else np.arange(int(data["y"].shape[0]), dtype=np.float64) * 10_000.0,
            "feature_scaler_mean": data["feature_scaler_mean"].astype(np.float32)
            if "feature_scaler_mean" in data.files
            else np.asarray([], dtype=np.float32),
            "feature_scaler_std": data["feature_scaler_std"].astype(np.float32)
            if "feature_scaler_std" in data.files
            else np.asarray([], dtype=np.float32),
            "target_scaler_mean": data["target_scaler_mean"].astype(np.float32)
            if "target_scaler_mean" in data.files
            else np.asarray([], dtype=np.float32),
            "target_scaler_std": data["target_scaler_std"].astype(np.float32)
            if "target_scaler_std" in data.files
            else np.asarray([], dtype=np.float32),
            "feature_columns_sha256": column_hash(feature_columns),
            "target_columns_sha256": column_hash(target_columns),
        }


def build_torch_model(kind: str, seq_len: int, feature_count: int, horizon_count: int, hidden: int, torch: Any, nn: Any) -> Any:
    kind = kind.strip().lower()

    class MlpSequenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_len * feature_count, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, horizon_count),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class TcnModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(feature_count, hidden, kernel_size=3, padding=2, dilation=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=2),
                nn.ReLU(),
            )
            self.head = nn.Linear(hidden, horizon_count)

        def forward(self, x: Any) -> Any:
            y = self.net(x.transpose(1, 2))[..., : x.shape[1]]
            return self.head(y[:, :, -1])

    class GruModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(feature_count, hidden, batch_first=True)
            self.head = nn.Linear(hidden, horizon_count)

        def forward(self, x: Any) -> Any:
            y, _ = self.rnn(x)
            return self.head(y[:, -1, :])

    class LstmModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.LSTM(feature_count, hidden, batch_first=True)
            self.head = nn.Linear(hidden, horizon_count)

        def forward(self, x: Any) -> Any:
            y, _ = self.rnn(x)
            return self.head(y[:, -1, :])

    class TransformerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(feature_count, hidden)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4 if hidden % 4 == 0 else 1,
                dim_feedforward=max(hidden * 2, 8),
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(hidden, horizon_count)

        def forward(self, x: Any) -> Any:
            y = self.encoder(self.proj(x))
            return self.head(y[:, -1, :])

    if kind in {"mlp", "sequence_mlp", "mlp_sequence"}:
        return MlpSequenceModel()
    if kind == "tcn":
        return TcnModel()
    if kind == "gru":
        return GruModel()
    if kind == "lstm":
        return LstmModel()
    if kind in {"transformer", "transformer_encoder"}:
        return TransformerModel()
    raise ValueError(f"unsupported_model_kind={kind}")


def predict_torch(model: Any, X: np.ndarray, batch_size: int, device: Any, torch: Any) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(batch).detach().cpu().numpy())
    if not outputs:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(outputs)


def train_torch_model(
    dataset: dict[str, Any],
    model_kind: str,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    require_cuda: bool,
    loss_name: str,
    checkpoint_path: Path | None = None,
    save_checkpoint: bool = True,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    status = torch_status()
    if not status["torch_available"]:
        return None, {"status": "skipped_torch_unavailable", "failure": "torch_unavailable", **status}

    import torch  # type: ignore
    from torch import nn  # type: ignore

    if require_cuda and not torch.cuda.is_available():
        return None, {"status": "skipped_cuda_unavailable", "failure": "cuda_unavailable", **status}

    X = dataset["X"]
    y = dataset["y"]
    splits = dataset["splits"].astype(str)
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    if int(train_mask.sum()) < 10:
        return None, {"status": "insufficient_train_rows", "failure": "train_rows_lt_10", **status}

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_mean = np.nanmean(y[train_mask], axis=0)
    y_std = np.nanstd(y[train_mask], axis=0)
    y_mean = np.where(np.isfinite(y_mean), y_mean, 0.0).astype(np.float32)
    y_std = np.where(np.isfinite(y_std) & (y_std > 1e-6), y_std, 1.0).astype(np.float32)
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)

    model_constructor_config = {
        "kind": model_kind,
        "seq_len": int(X.shape[1]),
        "feature_count": int(X.shape[2]),
        "horizon_count": int(y.shape[1]),
        "hidden": int(hidden),
    }
    model = build_torch_model(torch=torch, nn=nn, **model_constructor_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss() if loss_name.strip().lower() == "mse" else nn.HuberLoss()
    train_indices = np.where(train_mask)[0]
    losses: list[float] = []
    validation_losses: list[float] = []
    best_state: dict[str, Any] | None = None
    best_validation_loss = math.inf
    best_epoch = 0
    early_stop_reason = ""
    completed_epochs = 0
    patience_count = 0
    requested_epochs = max(1, epochs)
    for epoch_idx in range(requested_epochs):
        perm = train_indices[torch.randperm(len(train_indices)).cpu().numpy()]
        batch_losses = []
        model.train()
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            xb = torch.tensor(X[batch_idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_scaled[batch_idx], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        losses.append(float(np.mean(batch_losses)) if batch_losses else math.nan)
        completed_epochs = epoch_idx + 1
        if validation_mask.any():
            val_losses = []
            model.eval()
            with torch.no_grad():
                validation_indices = np.where(validation_mask)[0]
                for start in range(0, len(validation_indices), batch_size):
                    batch_idx = validation_indices[start : start + batch_size]
                    xb = torch.tensor(X[batch_idx], dtype=torch.float32, device=device)
                    yb = torch.tensor(y_scaled[batch_idx], dtype=torch.float32, device=device)
                    val_losses.append(float(loss_fn(model(xb), yb).detach().cpu().item()))
            current_validation_loss = float(np.mean(val_losses)) if val_losses else math.nan
            validation_losses.append(current_validation_loss)
            if math.isfinite(current_validation_loss) and current_validation_loss < best_validation_loss - early_stop_min_delta:
                best_validation_loss = current_validation_loss
                best_epoch = completed_epochs
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if early_stop_patience > 0 and patience_count >= early_stop_patience:
                    early_stop_reason = "validation_patience_exhausted"
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    elif validation_mask.any():
        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        best_epoch = completed_epochs
        best_validation_loss = validation_losses[-1] if validation_losses else math.nan
    pred_scaled = predict_torch(model, X, batch_size, device, torch)
    pred = (pred_scaled * y_std) + y_mean
    checkpoint_sha = ""
    checkpoint_status = "not_requested"
    roundtrip_status = "not_requested"
    roundtrip_max_abs_diff = math.nan
    if save_checkpoint and checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
            "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "model_constructor_config": model_constructor_config,
            "seed": int(seed),
            "target_scale_mean": y_mean,
            "target_scale_std": y_std,
            "feature_columns": list(dataset.get("feature_columns", [])),
            "target_columns": list(dataset.get("target_columns", [])),
            "target_lane": dataset.get("target_lane", "future_return_path"),
            "target_layout": dataset.get("target_layout", "return_vector"),
            "horizon_buckets": list(dataset.get("horizon_buckets", [])),
            "dataset_contract": dataset.get("dataset_contract", {}),
            "feature_columns_sha256": dataset.get("feature_columns_sha256", ""),
            "target_columns_sha256": dataset.get("target_columns_sha256", ""),
            "feature_scaler_mean": np.asarray(dataset.get("feature_scaler_mean", []), dtype=np.float32),
            "feature_scaler_std": np.asarray(dataset.get("feature_scaler_std", []), dtype=np.float32),
            "target_scaler_mean": np.asarray(dataset.get("target_scaler_mean", []), dtype=np.float32),
            "target_scaler_std": np.asarray(dataset.get("target_scaler_std", []), dtype=np.float32),
            "training_config": {
                "epochs_requested": int(requested_epochs),
                "epochs_completed": int(completed_epochs),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "loss_name": "MSELoss" if loss_name.strip().lower() == "mse" else "HuberLoss",
                "early_stop_patience": int(early_stop_patience),
                "early_stop_min_delta": float(early_stop_min_delta),
                "best_epoch": int(best_epoch),
                "best_validation_loss": float(best_validation_loss) if math.isfinite(best_validation_loss) else math.nan,
                "early_stop_reason": early_stop_reason,
            },
            "paper_only": True,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
        }
        try:
            torch.save(checkpoint_payload, checkpoint_path)
            checkpoint_sha = file_sha256(checkpoint_path)
            checkpoint_status = "saved"
            loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
            roundtrip_model = build_torch_model(torch=torch, nn=nn, **loaded["model_constructor_config"]).to(device)
            roundtrip_model.load_state_dict(loaded["model_state_dict"])
            roundtrip_scaled = predict_torch(roundtrip_model, X, batch_size, device, torch)
            loaded_mean = np.asarray(loaded["target_scale_mean"], dtype=np.float32)
            loaded_std = np.asarray(loaded["target_scale_std"], dtype=np.float32)
            roundtrip_pred = (roundtrip_scaled * loaded_std) + loaded_mean
            roundtrip_max_abs_diff = float(np.nanmax(np.abs(roundtrip_pred.astype(np.float64) - pred.astype(np.float64))))
            roundtrip_status = "ok" if math.isfinite(roundtrip_max_abs_diff) and roundtrip_max_abs_diff <= 1e-5 else "prediction_mismatch"
        except Exception as exc:
            checkpoint_status = "failed"
            roundtrip_status = f"failed:{exc}"
    return pred.astype(np.float64), {
        "status": "ok",
        "failure": "",
        **status,
        "device": str(device),
        "epochs": int(max(1, epochs)),
        "hidden": int(hidden),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "loss_name": "MSELoss" if loss_name.strip().lower() == "mse" else "HuberLoss",
        "final_train_loss": losses[-1] if losses else math.nan,
        "first_train_loss": losses[0] if losses else math.nan,
        "final_validation_loss": validation_losses[-1] if validation_losses else math.nan,
        "best_validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else math.nan,
        "best_epoch": int(best_epoch),
        "epochs_requested": int(requested_epochs),
        "epochs_completed": int(completed_epochs),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "early_stop_reason": early_stop_reason,
        "target_scaled": True,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_save_status": checkpoint_status,
        "checkpoint_roundtrip_status": roundtrip_status,
        "checkpoint_roundtrip_max_abs_diff": roundtrip_max_abs_diff,
        "model_constructor_config": json.dumps(model_constructor_config, sort_keys=True),
        "feature_columns_sha256": dataset.get("feature_columns_sha256", ""),
        "target_columns_sha256": dataset.get("target_columns_sha256", ""),
        "feature_scaler_state_present": bool(np.asarray(dataset.get("feature_scaler_mean", [])).size and np.asarray(dataset.get("feature_scaler_std", [])).size),
        "target_scaler_state_present": bool(np.asarray(dataset.get("target_scaler_mean", [])).size and np.asarray(dataset.get("target_scaler_std", [])).size),
    }


def write_report(
    path: Path,
    contract: dict[str, Any],
    summary: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    policy_metrics: pd.DataFrame,
    selected_policy_metrics: pd.DataFrame,
    target_lane_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# Rawseq Torch Sequence Benchmark",
        "",
        f"Created at: {contract['created_at']}",
        f"Dataset manifest: {contract['sequence_dataset_manifest']}",
        f"Output dir: {contract['output_dir']}",
        f"Torch status: {contract['torch_status']}",
        f"Models requested: {', '.join(contract['models_requested'])}",
        f"Seeds requested: {contract.get('seeds', [contract.get('seed')])}",
        f"Checkpoint saving: {contract.get('save_checkpoints')} dir={contract.get('checkpoint_dir', '')}",
        f"Early stopping: patience={contract.get('early_stop_patience')} min_delta={contract.get('early_stop_min_delta')}",
        f"Loss: {contract['loss_name']}",
        f"Baseline leaderboard: {contract.get('baseline_leaderboard_path', '')}",
        f"Baseline reference status: {contract.get('baseline_reference_status', '')}",
        f"Minimum baseline RMSE improvement fraction: {contract.get('min_baseline_improvement_fraction')}",
        f"Policy scoring: policy={contract.get('policy')} costs={contract.get('costs_bps')} thresholds={contract.get('thresholds_bps')}",
        "",
        "## Safety",
        "- paper_only=true",
        "- private_api=false",
        "- orders=false",
        "- promotion=false",
        "- champion_mutation=false",
        "",
        "## Summary",
    ]
    if summary.empty:
        lines.append("No model rows were produced.")
    else:
        cols = [
            col
            for col in [
                "feature_group",
                "seq_len",
                "model_kind",
                "seed",
                "status",
                "sequence_model_status",
                "target_lane",
                "target_lane_status",
                "weakest_validation_baseline_improvement",
                "validation_target_guard_reason",
                "validation_combined_rmse",
                "holdout_combined_rmse",
                "holdout_horizons_beating_best_baseline",
                "policy_horizons_holdout_positive",
                "best_holdout_position_cum_net_bps",
                "holdout_combined_correlation",
                "checkpoint_save_status",
                "checkpoint_roundtrip_status",
                "checkpoint_roundtrip_max_abs_diff",
                "device",
                "failure",
            ]
            if col in summary.columns
        ]
        lines.append(summary[cols].head(40).to_string(index=False))
    lines += ["", "## Target-Lane Metrics"]
    if target_lane_metrics.empty:
        lines.append("No target-lane-aware metrics were produced.")
    else:
        cols = [
            col
            for col in [
                "feature_group",
                "seq_len",
                "model_kind",
                "seed",
                "split",
                "target_lane",
                "evidence_type",
                "rmse",
                "train_mean_baseline_rmse",
                "baseline_improvement_fraction",
                "terminal_high_correlation",
                "terminal_low_correlation",
                "high_sign_violation_fraction",
                "low_sign_violation_fraction",
                "high_monotonic_violation_fraction",
                "low_monotonic_violation_fraction",
                "envelope_order_violation_fraction",
            ]
            if col in target_lane_metrics.columns
        ]
        lines.append(target_lane_metrics[cols].head(80).to_string(index=False))
    lines += ["", "## Per-Horizon Metrics"]
    if horizon_metrics.empty:
        lines.append("No per-horizon metrics were produced.")
    else:
        cols = [
            col
            for col in [
                "feature_group",
                "seq_len",
                "model_kind",
                "seed",
                "split",
                "horizon_buckets",
                "rmse",
                "mae",
                "correlation",
                "directional_accuracy",
                "balanced_accuracy",
            ]
            if col in horizon_metrics.columns
        ]
        lines.append(horizon_metrics[cols].head(60).to_string(index=False))
    lines += ["", "## Baseline Guard"]
    if baseline_comparison.empty:
        lines.append("No baseline comparison rows were produced.")
    else:
        cols = [
            col
            for col in [
                "feature_group",
                "seq_len",
                "model_kind",
                "seed",
                "horizon_buckets",
                "rmse",
                "best_baseline_model_by_rmse",
                "best_baseline_holdout_rmse",
                "rmse_improvement_fraction_vs_best_baseline",
                "baseline_guard_pass",
                "baseline_guard_reason",
            ]
            if col in baseline_comparison.columns
        ]
        lines.append(baseline_comparison[cols].head(60).to_string(index=False))
    lines += ["", "## Validation-Selected Policy Metrics"]
    if selected_policy_metrics.empty:
        lines.append("No validation-selected policy rows were produced.")
    else:
        cols = [
            col
            for col in [
                "feature_group",
                "seq_len",
                "model_kind",
                "seed",
                "horizon_buckets",
                "decision_cost_bps",
                "selected_threshold_bps",
                "validation_position_cum_net_bps",
                "holdout_position_cum_net_bps",
                "holdout_position_trade_count",
                "holdout_position_max_drawdown_bps",
            ]
            if col in selected_policy_metrics.columns
        ]
        lines.append(selected_policy_metrics[cols].head(60).to_string(index=False))
    lines += ["", "## Policy Grid Sample"]
    if policy_metrics.empty:
        lines.append("No policy grid rows were produced.")
    else:
        cols = [
            col
            for col in [
                "feature_group",
                "seq_len",
                "model_kind",
                "seed",
                "split",
                "horizon_buckets",
                "threshold_bps",
                "cost_bps",
                "selected_signal_count",
                "selected_signal_precision_win_rate",
                "non_overlapping_cum_net_bps",
                "position_cum_net_bps",
            ]
            if col in policy_metrics.columns
        ]
        lines.append(policy_metrics[cols].head(60).to_string(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_low_path_seed_stability_report(summary: pd.DataFrame, target_metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    validation = target_metrics[target_metrics["split"].astype(str).eq("validation")].copy() if not target_metrics.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        if validation.empty:
            group = pd.DataFrame()
        else:
            group = validation[
                validation["feature_group"].astype(str).eq(str(row.get("feature_group", "")))
                & validation["seq_len"].astype(str).eq(str(row.get("seq_len", "")))
                & validation["model_kind"].astype(str).eq(str(row.get("model_kind", "")))
                & validation["seed"].astype(str).eq(str(row.get("seed", "")))
            ].copy()
        vector = group[group["evidence_type"].astype(str).eq("low")].iloc[0] if not group.empty and group["evidence_type"].astype(str).eq("low").any() else pd.Series(dtype=object)
        horizons = group[group["evidence_type"].astype(str).eq("horizon")].copy() if not group.empty else pd.DataFrame()
        horizon_improvements = pd.to_numeric(horizons.get("baseline_improvement_fraction"), errors="coerce") if not horizons.empty else pd.Series(dtype=float)
        positive_horizons = int((horizon_improvements > 0.0).sum()) if len(horizon_improvements) else 0
        checkpoint_ok = str(row.get("checkpoint_roundtrip_status", "")).lower() == "ok"
        vector_improvement = safe_float(vector.get("baseline_improvement_fraction"))
        spearman = safe_float(vector.get("spearman_correlation"))
        unique = safe_float(vector.get("prediction_unique_fraction"))
        constant = bool(vector.get("constant_prediction_flag")) if "constant_prediction_flag" in vector else True
        monotonic = safe_float(vector.get("low_monotonic_violation_fraction"), 1.0)
        seed_pass = bool(
            str(row.get("sequence_model_status", "")) == "validation_target_lane_survivor"
            and checkpoint_ok
            and math.isfinite(vector_improvement)
            and vector_improvement > 0.0
            and positive_horizons >= 3
            and math.isfinite(spearman)
            and spearman > 0.0
            and math.isfinite(unique)
            and unique > 0.01
            and not constant
            and math.isfinite(monotonic)
            and monotonic <= 0.05
        )
        rows.append(
            {
                "feature_group": row.get("feature_group", ""),
                "seq_len": row.get("seq_len", ""),
                "model_kind": row.get("model_kind", ""),
                "seed": row.get("seed", ""),
                "sequence_model_status": row.get("sequence_model_status", ""),
                "seed_survival_pass": seed_pass,
                "vector_validation_rmse_improvement_vs_locked_cpu": vector_improvement,
                "median_horizon_improvement_vs_locked_cpu": safe_float(horizon_improvements.median()) if len(horizon_improvements) else math.nan,
                "positive_horizon_count": positive_horizons,
                "spearman_correlation": spearman,
                "pearson_correlation": safe_float(vector.get("pearson_correlation")),
                "prediction_unique_fraction": unique,
                "constant_prediction_flag": constant,
                "low_monotonic_violation_fraction": monotonic,
                "top_decile_drawdown_recall": safe_float(vector.get("top_decile_drawdown_recall")),
                "top_decile_drawdown_precision": safe_float(vector.get("top_decile_drawdown_precision")),
                "checkpoint_roundtrip_status": row.get("checkpoint_roundtrip_status", ""),
                "checkpoint_roundtrip_max_abs_diff": row.get("checkpoint_roundtrip_max_abs_diff", math.nan),
                "selected_cpu_baseline_name": row.get("selected_cpu_baseline_name", ""),
                "selected_cpu_contract_hash": row.get("selected_cpu_contract_hash", ""),
                "holdout_used_for_selection": False,
            }
        )
    frame = pd.DataFrame(rows)
    pass_count = int(frame["seed_survival_pass"].sum()) if not frame.empty else 0
    vector_improvements = pd.to_numeric(frame.get("vector_validation_rmse_improvement_vs_locked_cpu"), errors="coerce") if not frame.empty else pd.Series(dtype=float)
    positive_horizon_counts = pd.to_numeric(frame.get("positive_horizon_count"), errors="coerce") if not frame.empty else pd.Series(dtype=float)
    spearman_values = pd.to_numeric(frame.get("spearman_correlation"), errors="coerce") if not frame.empty else pd.Series(dtype=float)
    accepted = frame[frame["seed_survival_pass"].astype(bool)].copy() if not frame.empty else pd.DataFrame()
    recommend_larger = bool(
        pass_count >= 4
        and safe_float(vector_improvements.median()) > 0.0
        and safe_float(positive_horizon_counts.median()) >= 3
        and safe_float(spearman_values.median()) > 0.0
        and (accepted.empty or not accepted["constant_prediction_flag"].astype(bool).any())
        and (accepted.empty or accepted["checkpoint_roundtrip_status"].astype(str).eq("ok").all())
    )
    recommendation = {
        "recommended_next_step": "larger_replication" if recommend_larger else "stop_or_redesign_low_path_gru",
        "seed_survival_count": pass_count,
        "seed_count": int(len(frame)),
        "median_vector_validation_improvement_vs_locked_cpu": safe_float(vector_improvements.median()),
        "median_positive_horizon_count": safe_float(positive_horizon_counts.median()),
        "median_spearman_correlation": safe_float(spearman_values.median()),
        "holdout_used_for_selection": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    lines = [
        "# Rawseq Low-Path Seed Stability Report",
        "",
        f"Recommendation: {recommendation['recommended_next_step']}",
        f"Seed survival count: {pass_count}/{len(frame)}",
        f"Median vector validation improvement vs locked CPU: {recommendation['median_vector_validation_improvement_vs_locked_cpu']}",
        f"Median positive horizon count: {recommendation['median_positive_horizon_count']}",
        f"Median Spearman correlation: {recommendation['median_spearman_correlation']}",
        "Holdout used for selection: false",
    ]
    if not frame.empty:
        lines.extend(["", frame.to_string(index=False)])
    return frame, recommendation, lines


def main() -> int:
    manifest_env = os.getenv("RAWSEQ_TORCH_SEQUENCE_DATASET_MANIFEST", "").strip()
    manifest_path = resolve_path(manifest_env) if manifest_env else latest_sequence_manifest()
    output_root = resolve_path(
        os.getenv(
            "RAWSEQ_TORCH_SEQUENCE_OUTPUT_DIR",
            str(PROJECT_ROOT / "data" / "research" / "rawseq_torch_sequence_benchmarks"),
        )
    )
    model_kinds = [item.lower() for item in parse_csv_list("RAWSEQ_TORCH_SEQUENCE_MODELS", "mlp,tcn,gru,lstm,transformer")]
    feature_groups_env = parse_csv_list("RAWSEQ_TORCH_SEQUENCE_FEATURE_GROUPS", "")
    feature_groups = set(feature_groups_env) if feature_groups_env else None
    seq_lens_env = parse_csv_list("RAWSEQ_TORCH_SEQUENCE_LENS", "")
    seq_lens = {int(float(item)) for item in seq_lens_env} if seq_lens_env else None
    max_datasets = int(float(os.getenv("RAWSEQ_TORCH_SEQUENCE_MAX_DATASETS", "0")))
    epochs = int(float(os.getenv("RAWSEQ_TORCH_SEQUENCE_EPOCHS", "10")))
    hidden = int(float(os.getenv("RAWSEQ_TORCH_SEQUENCE_HIDDEN", "64")))
    batch_size = int(float(os.getenv("RAWSEQ_TORCH_SEQUENCE_BATCH_SIZE", "512")))
    learning_rate = float(os.getenv("RAWSEQ_TORCH_SEQUENCE_LR", "0.001"))
    seed = int(float(os.getenv("RAWSEQ_TORCH_SEQUENCE_SEED", "900")))
    seeds = parse_int_list("RAWSEQ_TORCH_SEQUENCE_SEEDS", str(seed))
    require_cuda = env_bool("RAWSEQ_TORCH_REQUIRE_CUDA", False)
    loss_name = os.getenv("RAWSEQ_TORCH_SEQUENCE_LOSS", "huber").strip().lower()
    write_predictions = env_bool("RAWSEQ_TORCH_WRITE_PREDICTIONS", False)
    save_checkpoints = env_bool("RAWSEQ_TORCH_SAVE_CHECKPOINTS", True)
    early_stop_patience = int(float(os.getenv("RAWSEQ_TORCH_EARLY_STOP_PATIENCE", "0")))
    early_stop_min_delta = float(os.getenv("RAWSEQ_TORCH_EARLY_STOP_MIN_DELTA", "0.0"))
    baseline_leaderboard_env = os.getenv("RAWSEQ_TORCH_BASELINE_LEADERBOARD_PATH", "").strip()
    min_baseline_improvement = float(os.getenv("RAWSEQ_TORCH_MIN_BASELINE_IMPROVEMENT_FRACTION", "0.0"))
    min_holdout_rows = int(float(os.getenv("RAWSEQ_TORCH_MIN_HOLDOUT_ROWS", "30")))
    policy = os.getenv("RAWSEQ_TORCH_POLICY", "direct_gt").strip().lower() or "direct_gt"
    thresholds = parse_float_list("RAWSEQ_TORCH_THRESHOLDS_BPS", "0,0.1,0.25,0.5,1,2")
    costs = parse_float_list("RAWSEQ_TORCH_COSTS_BPS", "0.1,1,5")
    decision_cost_bps = float(os.getenv("RAWSEQ_TORCH_DECISION_COST_BPS", str(costs[0] if costs else 0.1)))
    bucket_seconds_env = os.getenv("RAWSEQ_TORCH_BUCKET_SECONDS", "").strip()
    bucket_seconds_override = float(bucket_seconds_env) if bucket_seconds_env else math.nan

    manifest = load_dataset_manifest(manifest_path)
    baseline_reference, baseline_leaderboard_path, baseline_reference_status = load_baseline_reference(
        manifest_path,
        baseline_leaderboard_env,
    )
    dataset_rows = candidate_dataset_rows(
        manifest,
        manifest_path,
        feature_groups=feature_groups,
        seq_lens=seq_lens,
        max_datasets=max_datasets,
    )
    usable_datasets = [row for row in dataset_rows if row.get("dataset_status") == "ok"]
    run_hash = stable_hash(
        {
            "manifest": str(manifest_path),
            "models": model_kinds,
            "seq_lens": sorted(seq_lens) if seq_lens else "all",
            "feature_groups": sorted(feature_groups) if feature_groups else "all",
            "seeds": seeds,
        }
    )[:8]
    output_dir = output_root / f"torch_sequence_benchmark_{now_stamp()}_{run_hash}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    if write_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    if save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    status = torch_status()
    summary_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    target_metric_rows: list[pd.DataFrame] = []
    policy_rows: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []

    if not usable_datasets:
        status_rows.append(
            {
                "status": "no_usable_sequence_datasets",
                "failure": "no manifest rows with status=ok arrays_written=true path_npz_exists",
                **status,
            }
        )
    elif not status["torch_available"]:
        skipped_summary, skipped_status = no_torch_rows(usable_datasets, model_kinds, seeds, status)
        summary_rows.extend(skipped_summary)
        status_rows.extend(skipped_status)
    else:
        for dataset_row in usable_datasets:
            dataset_path = Path(str(dataset_row["dataset_path"]))
            dataset = load_npz_dataset(dataset_path)
            dataset_bucket_seconds = (
                bucket_seconds_override
                if math.isfinite(bucket_seconds_override) and bucket_seconds_override > 0
                else infer_bucket_seconds(dataset["decision_timestamps"])
            )
            dataset_meta = {
                "dataset_index": dataset_row.get("dataset_index"),
                "feature_group": dataset_row.get("feature_group", ""),
                "seq_len": int(float(dataset_row.get("seq_len", dataset["X"].shape[1]))),
                "feature_count": int(dataset["X"].shape[2]),
                "horizon_count": int(dataset["y"].shape[1]),
                "target_lane": dataset.get("target_lane", "future_return_path"),
                "target_layout": dataset.get("target_layout", "return_vector"),
                "target_columns_sha256": dataset.get("target_columns_sha256", ""),
                "bucket_seconds": float(dataset_bucket_seconds),
                "dataset_path": str(dataset_path),
            }
            cpu_reference = build_locked_cpu_reference(dataset, dataset_meta, output_dir)
            for model_kind in model_kinds:
                for current_seed in seeds:
                    seeded_meta = {**dataset_meta, "seed": current_seed}
                    checkpoint_path = (
                        checkpoint_dir
                        / f"{safe_filename(str(seeded_meta['feature_group']))}_seq{seeded_meta['seq_len']}_{model_kind}_s{current_seed}.pt"
                        if save_checkpoints
                        else None
                    )
                    try:
                        pred, meta = train_torch_model(
                            dataset,
                            model_kind,
                            hidden=hidden,
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            seed=current_seed,
                            require_cuda=require_cuda,
                            loss_name=loss_name,
                            checkpoint_path=checkpoint_path,
                            save_checkpoint=save_checkpoints,
                            early_stop_patience=early_stop_patience,
                            early_stop_min_delta=early_stop_min_delta,
                        )
                    except Exception as exc:
                        pred = None
                        meta = {"status": "failed", "failure": str(exc), **torch_status()}
                    status_row = {**seeded_meta, "model_kind": model_kind, **meta}
                    status_rows.append(status_row)
                    if pred is None or meta.get("status") != "ok":
                        summary_rows.append(status_row)
                        continue
                    pred_path = (
                        prediction_dir
                        / f"{safe_filename(str(seeded_meta['feature_group']))}_seq{seeded_meta['seq_len']}_{model_kind}_s{current_seed}.npz"
                        if write_predictions
                        else None
                    )
                    summary, per_horizon = split_metrics(
                        dataset["y"].astype(np.float64),
                        pred,
                        dataset["splits"].astype(str),
                        dataset["horizon_buckets"],
                        seeded_meta,
                        model_kind,
                    )
                    by_split = {row["split"]: row for row in summary}
                    summary_rows.append(
                        {
                            **status_row,
                            "prediction_path": str(pred_path) if pred_path is not None else "",
                            "selected_cpu_baseline_name": cpu_reference["selected_name"],
                            "selected_cpu_contract_hash": cpu_reference["contract"].get("cpu_contract_hash", ""),
                            "selected_cpu_validation_vector_rmse": cpu_reference["contract"].get("selected_validation_vector_rmse", math.nan),
                            "train_combined_rmse": by_split.get("train", {}).get("combined_rmse", math.nan),
                            "validation_combined_rmse": by_split.get("validation", {}).get("combined_rmse", math.nan),
                            "holdout_combined_rmse": by_split.get("untouched_holdout", {}).get("combined_rmse", math.nan),
                            "holdout_combined_correlation": by_split.get("untouched_holdout", {}).get(
                                "combined_correlation",
                                math.nan,
                            ),
                            "holdout_directional_accuracy": by_split.get("untouched_holdout", {}).get(
                                "combined_directional_accuracy",
                                math.nan,
                            ),
                        }
                    )
                    horizon_rows.extend(per_horizon)
                    target_metric_rows.append(
                        build_target_lane_metrics(
                            dataset["y"].astype(np.float64),
                            pred,
                            dataset["splits"].astype(str),
                            dataset["horizon_buckets"],
                            dataset["target_columns"],
                            dataset.get("target_lane", "future_return_path"),
                            dataset.get("target_layout", "return_vector"),
                            seeded_meta,
                            model_kind,
                            cpu_reference["selected_prediction"],
                            cpu_reference["selected_name"],
                        )
                    )
                    if dataset.get("target_lane", "future_return_path") in {"future_return_path", "coarse_return_vector"}:
                        policy_rows.append(
                            build_policy_metrics(
                                dataset["y"].astype(np.float64),
                                pred,
                                dataset["splits"].astype(str),
                                dataset["decision_timestamps"].astype(np.float64),
                                dataset["horizon_buckets"],
                                seeded_meta,
                                model_kind,
                                thresholds,
                                costs,
                                policy,
                                float(dataset_bucket_seconds),
                            )
                        )
                    if pred_path is not None:
                        actual_array = dataset["y"].astype(np.float32)
                        prediction_array = pred.astype(np.float32)
                        drawdown_target = np.full_like(actual_array, np.nan, dtype=np.float32)
                        drawdown_prediction = np.full_like(prediction_array, np.nan, dtype=np.float32)
                        low_indices = [
                            idx
                            for idx, column in enumerate(dataset["target_columns"])
                            if "low" in str(column)
                        ]
                        if low_indices:
                            drawdown_target[:, low_indices] = -actual_array[:, low_indices]
                            drawdown_prediction[:, low_indices] = -prediction_array[:, low_indices]
                        np.savez_compressed(
                            pred_path,
                            prediction=prediction_array,
                            actual=actual_array,
                            canonical_prediction=prediction_array,
                            canonical_target=actual_array,
                            drawdown_prediction=drawdown_prediction,
                            drawdown_target=drawdown_target,
                            splits=dataset["splits"].astype(str),
                            target_columns=np.asarray(dataset["target_columns"], dtype=str),
                            target_lane=np.asarray(dataset.get("target_lane", "future_return_path"), dtype=str),
                            target_layout=np.asarray(dataset.get("target_layout", "return_vector"), dtype=str),
                            horizon_buckets=np.asarray(dataset["horizon_buckets"], dtype=int),
                            selected_cpu_prediction=cpu_reference["selected_prediction"].astype(np.float32),
                            selected_cpu_model=np.asarray(cpu_reference["selected_name"], dtype=str),
                        )

    summary_df = pd.DataFrame(summary_rows)
    horizon_df = pd.DataFrame(horizon_rows)
    target_metric_df = pd.concat(target_metric_rows, ignore_index=True) if target_metric_rows else pd.DataFrame()
    policy_df = pd.concat(policy_rows, ignore_index=True) if policy_rows else pd.DataFrame()
    selected_policy_df = select_validation_policy_metrics(policy_df, decision_cost_bps)
    status_df = pd.DataFrame(status_rows)
    baseline_comparison_df = compare_horizon_metrics_to_baselines(
        horizon_df,
        baseline_reference,
        min_improvement_fraction=min_baseline_improvement,
        min_holdout_rows=min_holdout_rows,
    )
    summary_df = apply_baseline_status_to_summary(summary_df, baseline_comparison_df)
    summary_df = apply_policy_status_to_summary(summary_df, selected_policy_df)
    summary_df = apply_validation_target_lane_status(summary_df, target_metric_df, min_baseline_improvement)
    stability_df, stability_recommendation, stability_lines = build_low_path_seed_stability_report(summary_df, target_metric_df)
    if summary_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "feature_group",
                "seq_len",
                "model_kind",
                "seed",
                "status",
                "failure",
                "validation_combined_rmse",
                "holdout_combined_rmse",
                "checkpoint_path",
                "checkpoint_sha256",
            ]
        )
    if horizon_df.empty:
        horizon_df = pd.DataFrame(
            columns=[
                "feature_group",
                "seq_len",
                "model_kind",
                "seed",
                "split",
                "horizon_index",
                "horizon_buckets",
                "rows",
                "mae",
                "rmse",
                "correlation",
                "directional_accuracy",
                "balanced_accuracy",
            ]
        )
    if target_metric_df.empty:
        target_metric_df = pd.DataFrame(
            columns=[
                "dataset_index",
                "feature_group",
                "seq_len",
                "model_kind",
                "seed",
                "split",
                "target_lane",
                "target_layout",
                "evidence_type",
                "target_column",
                "rows",
                "mae",
                "rmse",
                "train_mean_baseline_rmse",
                "baseline_improvement_fraction",
                "terminal_high_correlation",
                "terminal_low_correlation",
                "high_sign_violation_fraction",
                "low_sign_violation_fraction",
                "high_monotonic_violation_fraction",
                "low_monotonic_violation_fraction",
                "envelope_order_violation_fraction",
            ]
        )
    if policy_df.empty:
        policy_df = pd.DataFrame(
            columns=[
                "dataset_index",
                "feature_group",
                "seq_len",
                "model_kind",
                "seed",
                "split",
                "horizon_index",
                "horizon_buckets",
                "policy",
                "threshold_bps",
                "cost_bps",
                "policy_direction_multiplier",
                "selected_signal_count",
                "selected_signal_precision_win_rate",
                "recall_profitable_opportunities",
                "avg_selected_gross_bps",
                "avg_selected_net_bps",
                "row_signal_cum_net_bps",
                "non_overlapping_trade_count",
                "non_overlapping_cum_net_bps",
                "position_trade_count",
                "position_cum_net_bps",
                "position_max_drawdown_bps",
                "position_exposure_fraction",
            ]
        )
    if selected_policy_df.empty:
        selected_policy_df = pd.DataFrame(
            columns=[
                "dataset_index",
                "feature_group",
                "seq_len",
                "model_kind",
                "seed",
                "horizon_buckets",
                "decision_cost_bps",
                "selected_threshold_bps",
                "selection_stage",
                "validation_position_trade_count",
                "validation_position_cum_net_bps",
                "holdout_stage",
                "holdout_position_trade_count",
                "holdout_position_cum_net_bps",
                "holdout_position_max_drawdown_bps",
                "holdout_position_exposure_fraction",
            ]
        )
    if status_df.empty:
        status_df = pd.DataFrame(columns=["status", "failure", "torch_available", "cuda_available"])
    if not summary_df.empty and "status" in summary_df.columns:
        if "weakest_validation_baseline_improvement" in summary_df.columns:
            status_priority = {
                "validation_target_lane_survivor": 0,
                "multi_horizon_holdout_baseline_survivor": 1,
                "single_horizon_holdout_baseline_survivor": 2,
                "fails_validation_target_lane_guard": 3,
                "fails_holdout_baseline_guard": 4,
            }
            summary_df["_sequence_status_priority"] = summary_df["sequence_model_status"].map(status_priority).fillna(9)
            sort_cols = [col for col in ["_sequence_status_priority", "weakest_validation_baseline_improvement", "validation_combined_rmse"] if col in summary_df.columns]
            ascending = [True if col != "weakest_validation_baseline_improvement" else False for col in sort_cols]
            summary_df = summary_df.sort_values(sort_cols, ascending=ascending, na_position="last").drop(columns=["_sequence_status_priority"])
        else:
            sort_cols = [col for col in ["status", "holdout_combined_rmse", "validation_combined_rmse"] if col in summary_df.columns]
            summary_df = summary_df.sort_values(sort_cols, ascending=[True] + [True] * (len(sort_cols) - 1), na_position="last")

    contract = {
        "created_at": datetime.now(UTC).isoformat(),
        "sequence_dataset_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "models_requested": model_kinds,
        "feature_groups_filter": sorted(feature_groups) if feature_groups else [],
        "seq_lens_filter": sorted(seq_lens) if seq_lens else [],
        "epochs": epochs,
        "hidden": hidden,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "seeds": seeds,
        "save_checkpoints": save_checkpoints,
        "checkpoint_dir": str(checkpoint_dir) if save_checkpoints else "",
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "require_cuda": require_cuda,
        "loss_name": loss_name,
        "torch_status": status,
        "usable_dataset_count": len(usable_datasets),
        "baseline_leaderboard_path": baseline_leaderboard_path,
        "baseline_reference_status": baseline_reference_status,
        "min_baseline_improvement_fraction": min_baseline_improvement,
        "min_holdout_rows": min_holdout_rows,
        "policy": policy,
        "thresholds_bps": thresholds,
        "costs_bps": costs,
        "decision_cost_bps": decision_cost_bps,
        "bucket_seconds_override": bucket_seconds_override if math.isfinite(bucket_seconds_override) else "",
        "target_lane_aware_metrics": True,
        "target_lane_guard_stage": "validation_only",
        "paper_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }

    summary_path = output_dir / "torch_sequence_model_summary.csv"
    horizon_path = output_dir / "torch_sequence_per_horizon_metrics.csv"
    target_metric_path = output_dir / "torch_sequence_target_lane_metrics.csv"
    policy_path = output_dir / "torch_sequence_policy_metrics.csv"
    selected_policy_path = output_dir / "torch_sequence_selected_policy_metrics.csv"
    baseline_reference_path = output_dir / "torch_sequence_baseline_reference.csv"
    baseline_comparison_path = output_dir / "torch_sequence_baseline_comparison.csv"
    status_path = output_dir / "torch_sequence_benchmark_status.csv"
    contract_path = output_dir / "torch_sequence_benchmark_contract.json"
    report_path = output_dir / "torch_sequence_benchmark_report.txt"
    stability_path = output_dir / "rawseq_low_path_seed_stability_report.csv"
    stability_txt_path = output_dir / "rawseq_low_path_seed_stability_report.txt"
    stability_json_path = output_dir / "rawseq_low_path_recommended_next_step.json"
    summary_df.to_csv(summary_path, index=False)
    horizon_df.to_csv(horizon_path, index=False)
    target_metric_df.to_csv(target_metric_path, index=False)
    policy_df.to_csv(policy_path, index=False)
    selected_policy_df.to_csv(selected_policy_path, index=False)
    baseline_reference.to_csv(baseline_reference_path, index=False)
    baseline_comparison_df.to_csv(baseline_comparison_path, index=False)
    status_df.to_csv(status_path, index=False)
    stability_df.to_csv(stability_path, index=False)
    stability_txt_path.write_text("\n".join(stability_lines) + "\n", encoding="utf-8")
    stability_json_path.write_text(json.dumps(stability_recommendation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_path, contract, summary_df, horizon_df, baseline_comparison_df, policy_df, selected_policy_df, target_metric_df)

    print("Rawseq torch sequence benchmark complete")
    print(f"Output dir: {output_dir}")
    print(f"Dataset manifest: {manifest_path}")
    print(f"Torch status: {status}")
    print(f"Usable datasets: {len(usable_datasets)}")
    print(f"Summary: {summary_path}")
    print(f"Per-horizon metrics: {horizon_path}")
    print(f"Target-lane metrics: {target_metric_path}")
    print(f"Policy metrics: {policy_path}")
    print(f"Validation-selected policy metrics: {selected_policy_path}")
    print(f"Baseline reference: {baseline_reference_path}")
    print(f"Baseline comparison: {baseline_comparison_path}")
    print(f"Status: {status_path}")
    print(f"Low-path seed stability: {stability_path}")
    print(f"Low-path recommended next step: {stability_json_path}")
    print("Safety: paper_only=true private_api=false orders=false promotion=false champion_mutation=false")
    if not summary_df.empty:
        cols = [
            col
            for col in [
                "feature_group",
                "seq_len",
                "model_kind",
                "seed",
                "status",
                "sequence_model_status",
                "validation_combined_rmse",
                "holdout_combined_rmse",
                "holdout_horizons_beating_best_baseline",
                "policy_horizons_holdout_positive",
                "best_holdout_position_cum_net_bps",
                "holdout_combined_correlation",
                "checkpoint_save_status",
                "checkpoint_roundtrip_status",
                "checkpoint_roundtrip_max_abs_diff",
                "failure",
            ]
            if col in summary_df.columns
        ]
        print(summary_df[cols].head(20).to_string(index=False))
    return 0


def safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
    return cleaned or "dataset"


if __name__ == "__main__":
    raise SystemExit(main())
