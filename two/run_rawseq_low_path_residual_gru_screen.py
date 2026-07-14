#!/usr/bin/env python3
"""Build and evaluate a bounded residual-GRU screen for the low/downside path.

This is research-only. It does not place orders, use private APIs, mutate or
promote champions, or use holdout for model/seed/configuration selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.build_rawseq_locked_bundle_sequence_datasets import materialize_selected_targets
from scripts.tiny.report_rawseq_low_path_bounded_cpu_reconciliation import (
    array_sha256,
    column_hash,
    cpu_evaluate_arrays,
    load_feature_columns,
    load_selected_payload,
    materialized_copy,
    safe_float,
    stable_hash,
    vector_mae,
    vector_rmse,
)
from scripts.tiny.report_rawseq_target_lane_baseline_tournament import (
    fill_targets,
    fit_elastic_net_proxy,
    fit_ridge,
    predict_linear,
)


DEFAULT_TOURNAMENT_DIR = PROJECT_ROOT / "data" / "research" / "rawseq_target_lane_baseline_tournament" / "rawseq_target_lane_baseline_tournament_20260711T175931Z"
DEFAULT_DIAG_DIR = PROJECT_ROOT / "data" / "research" / "rawseq_feature_diagnostics" / "rawseq_feature_diagnostics_20260711T150938Z"
DEFAULT_INDICATOR_DIR = Path(r"F:\rsio\rawseq_target_tournament_coarse_1s_300k_retry\mh_indicator_SOLUSDT_kraken_20260711T145015Z_fba19c8d")
DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_low_path_residual_gru_screens")


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return resolve_path(raw) if raw else default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "y"}


def parse_ints(name: str, default: str) -> list[int]:
    return [int(float(part.strip())) for part in os.getenv(name, default).split(",") if part.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def finite_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "rms": math.nan, "p95_abs": math.nan, "max_abs": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "rms": float(np.sqrt(np.mean(arr**2))),
        "p95_abs": float(np.quantile(np.abs(arr), 0.95)),
        "max_abs": float(np.max(np.abs(arr))),
    }


def pearson_corr(actual: np.ndarray, pred: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(p)
    if mask.sum() < 3:
        return math.nan
    a = a[mask]
    p = p[mask]
    if np.std(a) <= 1e-12 or np.std(p) <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, p)[0, 1])


def spearman_corr(actual: np.ndarray, pred: np.ndarray) -> float:
    a = pd.Series(np.asarray(actual, dtype=np.float64).ravel())
    p = pd.Series(np.asarray(pred, dtype=np.float64).ravel())
    mask = np.isfinite(a.to_numpy()) & np.isfinite(p.to_numpy())
    if mask.sum() < 3:
        return math.nan
    return pearson_corr(a[mask].rank(method="average").to_numpy(), p[mask].rank(method="average").to_numpy())


def calibration(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    a = np.asarray(actual, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(p)
    if mask.sum() < 3 or np.std(p[mask]) <= 1e-12:
        return math.nan, math.nan
    slope, intercept = np.polyfit(p[mask], a[mask], 1)
    return float(slope), float(intercept)


def low_monotonic_violation(pred: np.ndarray) -> float:
    arr = np.asarray(pred, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return math.nan
    diffs = np.diff(arr, axis=1)
    return float(np.mean(diffs > 1e-9))


def latest_contiguous_by_split(table: pd.DataFrame, rows_per_split: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts = []
    blocks = []
    for split in ["train", "validation", "untouched_holdout"]:
        group = table[table["split"].astype(str).eq(split)].copy()
        if len(group) == 0:
            continue
        sampled = group.tail(rows_per_split).copy()
        sampled["sampling_block_id"] = f"{split}_latest_contiguous"
        parts.append(sampled)
        ts = pd.to_numeric(sampled["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        blocks.append(
            {
                "split": split,
                "sampling_block_id": f"{split}_latest_contiguous",
                "rows": int(len(sampled)),
                "first_source_row_index": int(sampled["source_row_index"].iloc[0]),
                "last_source_row_index": int(sampled["source_row_index"].iloc[-1]),
                "first_timestamp": float(ts[0]),
                "last_timestamp": float(ts[-1]),
                "duration_hours": float((ts[-1] - ts[0]) / 3_600_000.0) if len(ts) > 1 else 0.0,
            }
        )
    return pd.concat(parts, axis=0).sort_index().copy(), blocks


def latest_contiguous_source_for_final_sequences(
    table: pd.DataFrame,
    final_sequence_rows_per_split: int,
    seq_len: int,
    target_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    warmup_rows = max(int(seq_len) - 1, 0)
    source_rows_per_split = int(final_sequence_rows_per_split) + warmup_rows
    parts = []
    blocks = []
    for split in ["train", "validation", "untouched_holdout"]:
        group = table[table["split"].astype(str).eq(split)].copy()
        if len(group) == 0:
            continue
        if target_columns:
            target_values = group[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            finite_endpoint_positions = np.flatnonzero(np.isfinite(target_values).all(axis=1))
            if len(finite_endpoint_positions) < final_sequence_rows_per_split:
                endpoint_start = int(finite_endpoint_positions[0]) if len(finite_endpoint_positions) else 0
            else:
                endpoint_start = int(finite_endpoint_positions[-final_sequence_rows_per_split])
            endpoint_end = int(finite_endpoint_positions[-1]) if len(finite_endpoint_positions) else len(group) - 1
            sample_start = max(0, endpoint_start - warmup_rows)
            sampled = group.iloc[sample_start : endpoint_end + 1].copy()
        else:
            sampled = group.tail(source_rows_per_split).copy()
        sampled["sampling_block_id"] = f"{split}_latest_contiguous"
        parts.append(sampled)
        ts = pd.to_numeric(sampled["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        blocks.append(
            {
                "split": split,
                "sampling_block_id": f"{split}_latest_contiguous",
                "rows": int(len(sampled)),
                "first_source_row_index": int(sampled["source_row_index"].iloc[0]),
                "last_source_row_index": int(sampled["source_row_index"].iloc[-1]),
                "first_timestamp": float(ts[0]),
                "last_timestamp": float(ts[-1]),
                "duration_hours": float((ts[-1] - ts[0]) / 3_600_000.0) if len(ts) > 1 else 0.0,
                "requested_final_sequence_rows": int(final_sequence_rows_per_split),
                "source_rows_requested_for_sequence_warmup": int(source_rows_per_split),
                "sequence_warmup_rows": int(warmup_rows),
                "target_finite_endpoint_sampling": bool(target_columns),
            }
        )
    return pd.concat(parts, axis=0).sort_index().copy(), blocks


def latest_contiguous_train_validation(table: pd.DataFrame, rows_per_split: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts = []
    blocks = []
    for split in ["train", "validation"]:
        group = table[table["split"].astype(str).eq(split)].copy()
        sampled = group.tail(rows_per_split).copy()
        sampled["sampling_block_id"] = f"{split}_latest_contiguous"
        parts.append(sampled)
        ts = pd.to_numeric(sampled["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        blocks.append(
            {
                "split": split,
                "sampling_block_id": f"{split}_latest_contiguous",
                "rows": int(len(sampled)),
                "first_source_row_index": int(sampled["source_row_index"].iloc[0]),
                "last_source_row_index": int(sampled["source_row_index"].iloc[-1]),
                "first_timestamp": float(ts[0]),
                "last_timestamp": float(ts[-1]),
                "duration_hours": float((ts[-1] - ts[0]) / 3_600_000.0) if len(ts) > 1 else 0.0,
            }
        )
    return pd.concat(parts, axis=0).sort_index().copy(), blocks


def fit_scaler(values: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}


def apply_scaler(values: np.ndarray, scaler: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    missing = ~np.isfinite(values)
    filled = np.where(np.isfinite(values), values, scaler["mean"])
    scaled = (filled - scaler["mean"]) / scaler["std"]
    scaled[~np.isfinite(scaled)] = 0.0
    return scaled.astype(np.float32), missing.astype(np.uint8)


def build_blocked_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    seq_len: int,
) -> dict[str, np.ndarray]:
    train_rows = frame[frame["split"].astype(str).eq("train")]
    raw_train = train_rows[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    feature_scaler = fit_scaler(raw_train)
    target_train = train_rows[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    target_scaler = fit_scaler(target_train)
    x_rows: list[np.ndarray] = []
    miss_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    split_rows: list[str] = []
    ts_rows: list[float] = []
    source_rows: list[int] = []
    block_rows: list[str] = []
    boundary_violations = 0
    for block_id, block in frame.groupby("sampling_block_id", sort=False):
        raw_features = block[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        scaled, missing = apply_scaler(raw_features, feature_scaler)
        targets = block[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        timestamps = pd.to_numeric(block["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        source_idx = block["source_row_index"].to_numpy(dtype=np.int64)
        splits = block["split"].astype(str).to_numpy()
        for idx in range(seq_len - 1, len(block)):
            start = idx - seq_len + 1
            if not np.isfinite(targets[idx]).all() or not np.isfinite(timestamps[idx]):
                continue
            if len(set(splits[start : idx + 1])) != 1:
                boundary_violations += 1
                continue
            x_rows.append(scaled[start : idx + 1])
            miss_rows.append(missing[start : idx + 1])
            y_rows.append(targets[idx])
            split_rows.append(str(splits[idx]))
            ts_rows.append(float(timestamps[idx]))
            source_rows.append(int(source_idx[idx]))
            block_rows.append(str(block_id))
    return {
        "X": np.asarray(x_rows, dtype=np.float32),
        "missing_mask": np.asarray(miss_rows, dtype=np.uint8),
        "y": np.asarray(y_rows, dtype=np.float32),
        "splits": np.asarray(split_rows, dtype=str),
        "decision_timestamps": np.asarray(ts_rows, dtype=np.float64),
        "source_row_indices": np.asarray(source_rows, dtype=np.int64),
        "sampling_block_ids": np.asarray(block_rows, dtype=str),
        "feature_scaler_mean": feature_scaler["mean"].astype(np.float32),
        "feature_scaler_std": feature_scaler["std"].astype(np.float32),
        "target_scaler_mean": target_scaler["mean"].astype(np.float32),
        "target_scaler_std": target_scaler["std"].astype(np.float32),
        "sequence_boundary_violations": int(boundary_violations),
    }


def split_arrays(arrays: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    mask = arrays["splits"].astype(str) == split
    return {
        "x": arrays["X"][mask, -1, :].astype(np.float64),
        "y": arrays["y"][mask].astype(np.float64),
        "source_rows": arrays["source_row_indices"][mask].astype(np.int64),
        "timestamps": arrays["decision_timestamps"][mask].astype(np.float64),
    }


def fit_cpu_model(
    model: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float | None,
) -> dict[str, Any]:
    scaler = fit_scaler(x_train)
    x_scaled, _ = apply_scaler(x_train, scaler)
    if model == "ridge_multi_output":
        coef = fit_ridge(x_scaled.astype(np.float64), y_train.astype(np.float64), float(alpha))
    elif model == "elastic_net_multi_output":
        coef = fit_elastic_net_proxy(x_scaled.astype(np.float64), y_train.astype(np.float64), float(alpha), 0.5, 40, 0.02)
    elif model == "training_mean_baseline":
        coef = np.nanmean(y_train, axis=0)
    elif model == "training_median_baseline":
        coef = np.nanmedian(y_train, axis=0)
    else:
        raise ValueError(f"Unsupported CPU model: {model}")
    return {"model": model, "alpha": alpha, "scaler": scaler, "coef": coef}


def predict_cpu(model_payload: dict[str, Any], x: np.ndarray) -> np.ndarray:
    model = model_payload["model"]
    if model in {"training_mean_baseline", "training_median_baseline"}:
        coef = np.asarray(model_payload["coef"], dtype=np.float64)
        return np.tile(coef, (len(x), 1))
    x_scaled, _ = apply_scaler(x, model_payload["scaler"])
    return predict_linear(x_scaled.astype(np.float64), np.asarray(model_payload["coef"], dtype=np.float64))


def ridge_pathology_audit(
    x_train: np.ndarray,
    y_train_raw: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    feature_columns: list[str],
    target_columns: list[str],
    ridge_alphas: list[float],
    mean_rmse: float,
    median_rmse: float,
) -> list[dict[str, Any]]:
    y_train, _, _ = fill_targets(y_train_raw.astype(np.float64))
    scaler = fit_scaler(x_train.astype(np.float64))
    x_train_scaled, _ = apply_scaler(x_train.astype(np.float64), scaler)
    x_validation_scaled, _ = apply_scaler(x_validation.astype(np.float64), scaler)
    target_std = float(np.nanstd(y_validation))
    feature_std = np.asarray(scaler["std"], dtype=np.float64)
    rows = []
    for alpha in ridge_alphas:
        coef = fit_ridge(x_train_scaled.astype(np.float64), y_train, float(alpha))
        pred = predict_linear(x_validation_scaled.astype(np.float64), coef)
        pred_finite = np.isfinite(pred)
        pred_std = float(np.nanstd(pred))
        pred_max_abs = float(np.nanmax(np.abs(pred))) if pred.size else math.nan
        rmse_value = vector_rmse(y_validation, pred)
        coef_no_intercept = coef[1:] if coef.ndim == 2 and coef.shape[0] > 1 else coef
        intercept = coef[0] if coef.ndim == 2 else np.asarray([])
        scale_ratio = pred_std / target_std if target_std > 1e-12 and math.isfinite(pred_std) else math.inf
        catastrophic_ratio = rmse_value / min(mean_rmse, median_rmse) if min(mean_rmse, median_rmse) > 0 and math.isfinite(rmse_value) else math.inf
        reasons = []
        if not bool(pred_finite.all()):
            reasons.append("nonfinite_predictions")
        if scale_ratio > 10.0 or pred_max_abs > max(1000.0, 50.0 * max(1.0, target_std)):
            reasons.append("prediction_scale_disproportionate")
        if catastrophic_ratio > 5.0:
            reasons.append("catastrophic_rmse_vs_constants")
        status = "pathological" if reasons else "ok"
        rows.append(
            {
                "alpha": alpha,
                "ridge_pathology_status": status,
                "ridge_pathology_reasons": ";".join(reasons),
                "validation_vector_rmse": rmse_value,
                "coefficient_l2_norm": float(np.linalg.norm(coef_no_intercept)),
                "intercept_norm": float(np.linalg.norm(intercept)),
                "prediction_mean": float(np.nanmean(pred)),
                "prediction_std": pred_std,
                "prediction_min": float(np.nanmin(pred)),
                "prediction_max": float(np.nanmax(pred)),
                "target_mean": float(np.nanmean(y_validation)),
                "target_std": target_std,
                "prediction_std_to_target_std": scale_ratio,
                "finite_prediction_fraction": float(np.mean(pred_finite)),
                "maximum_absolute_prediction": pred_max_abs,
                "feature_near_zero_variance_count": int(np.sum(feature_std <= 1e-8)),
                "feature_scaler_hash": stable_hash({"mean": scaler["mean"].tolist(), "std": scaler["std"].tolist()}),
                "target_scaler_hash": stable_hash({"target_columns": target_columns, "target_std": target_std}),
                "feature_columns_sha256": column_hash(feature_columns),
                "target_columns_sha256": column_hash(target_columns),
                "inverse_target_transformation_valid": True,
                "eligible_for_selection": status == "ok",
            }
        )
    return rows


def select_cpu_model(preflight_rows: list[dict[str, Any]], ridge_pathology_rows: list[dict[str, Any]] | None = None) -> tuple[str, float | None]:
    pathological_alphas = {
        float(row["alpha"])
        for row in (ridge_pathology_rows or [])
        if str(row.get("ridge_pathology_status")) != "ok"
    }
    eligible = []
    for row in preflight_rows:
        model = str(row["model"])
        if model not in {"ridge_multi_output", "elastic_net_multi_output"}:
            continue
        if model == "ridge_multi_output" and safe_float(row.get("selected_hyperparameter")) in pathological_alphas:
            continue
        eligible.append(row)
    if not eligible:
        raise RuntimeError("No nonpathological learned CPU candidates available for selection")
    best = sorted(eligible, key=lambda row: safe_float(row["validation_vector_rmse"]) if math.isfinite(safe_float(row["validation_vector_rmse"])) else 1e18)[0]
    alpha_raw = best.get("selected_hyperparameter", "")
    return str(best["model"]), (float(alpha_raw) if str(alpha_raw) not in {"", "nan", "None"} else None)


def generate_oof_cpu_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    selected_model: str,
    selected_alpha: float | None,
    folds: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n = len(y_train)
    pred = np.full_like(y_train, np.nan, dtype=np.float64)
    usable = np.zeros(n, dtype=bool)
    manifest = []
    bounds = np.linspace(0, n, folds + 1, dtype=int)
    for fold in range(folds):
        start, end = int(bounds[fold]), int(bounds[fold + 1])
        if start < max(200, y_train.shape[1] * 20):
            manifest.append(
                {
                    "fold": fold,
                    "status": "oof_unavailable_insufficient_prior_rows",
                    "train_start": 0,
                    "train_end_exclusive": start,
                    "predict_start": start,
                    "predict_end_exclusive": end,
                    "leakage_safe": True,
                }
            )
            continue
        payload = fit_cpu_model(selected_model, x_train[:start], y_train[:start], selected_alpha)
        pred[start:end] = predict_cpu(payload, x_train[start:end])
        usable[start:end] = True
        manifest.append(
            {
                "fold": fold,
                "status": "ok",
                "train_start": 0,
                "train_end_exclusive": start,
                "predict_start": start,
                "predict_end_exclusive": end,
                "leakage_safe": True,
                "selected_cpu_model": selected_model,
                "selected_alpha": selected_alpha,
            }
        )
    return pred, usable, manifest


@dataclass
class ResidualRun:
    seed: int
    final_prediction: np.ndarray
    residual_prediction: np.ndarray
    metrics: dict[str, Any]
    checkpoint_path: Path | None


def run_residual_gru_seed(
    seed: int,
    arrays: dict[str, np.ndarray],
    cpu_predictions: dict[str, np.ndarray],
    oof_train_prediction: np.ndarray,
    oof_usable: np.ndarray,
    output_dim: int,
    epochs: int,
    patience: int,
    require_cuda: bool,
    checkpoint_path: Path,
    correction_l2: float,
) -> ResidualRun:
    import torch  # type: ignore
    from torch import nn  # type: ignore

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but unavailable")
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = arrays["splits"].astype(str)
    train_mask_all = splits == "train"
    validation_mask = splits == "validation"
    train_indices_all = np.where(train_mask_all)[0]
    train_indices = train_indices_all[oof_usable]
    if len(train_indices) < 100:
        raise RuntimeError("Too few OOF-usable train rows for residual GRU")
    residual_target = arrays["y"].astype(np.float32) - oof_train_prediction.astype(np.float32)
    train_resid = residual_target[train_indices]
    resid_scale = np.nanstd(train_resid, axis=0)
    resid_scale = np.where(np.isfinite(resid_scale) & (resid_scale > 1e-6), resid_scale, 1.0).astype(np.float32)
    y_scaled = residual_target / resid_scale

    class ResidualGRU(nn.Module):
        def __init__(self, feature_count: int, hidden: int, output_dim: int):
            super().__init__()
            self.rnn = nn.GRU(feature_count, hidden, batch_first=True)
            self.head = nn.Linear(hidden, output_dim)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    model = ResidualGRU(arrays["X"].shape[2], 32, output_dim).to(device)
    x_val = torch.tensor(arrays["X"][validation_mask], dtype=torch.float32, device=device)
    with torch.no_grad():
        initial_resid = (model(x_val).cpu().numpy() * resid_scale).astype(np.float32)
    initial_cpu_reproduction_max_abs_diff = float(np.nanmax(np.abs(initial_resid)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.HuberLoss(reduction="none")
    best_state: dict[str, Any] | None = None
    best_val_loss = math.inf
    best_epoch = 0
    patience_count = 0
    batch_size = 256
    for epoch in range(1, epochs + 1):
        perm = train_indices[torch.randperm(len(train_indices)).cpu().numpy()]
        model.train()
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            xb = torch.tensor(arrays["X"][idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_scaled[idx], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pred_scaled = model(xb)
            loss = loss_fn(pred_scaled, yb).mean()
            if correction_l2 > 0:
                loss = loss + correction_l2 * torch.mean(pred_scaled**2)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_val_scaled = model(x_val)
            y_val_scaled = torch.tensor(
                ((arrays["y"][validation_mask] - cpu_predictions["validation"]) / resid_scale).astype(np.float32),
                dtype=torch.float32,
                device=device,
            )
            val_loss = float(loss_fn(pred_val_scaled, y_val_scaled).mean().cpu().item())
        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled_all = []
        for start in range(0, len(arrays["X"]), 512):
            xb = torch.tensor(arrays["X"][start : start + 512], dtype=torch.float32, device=device)
            pred_scaled_all.append(model(xb).cpu().numpy())
        residual_prediction = (np.vstack(pred_scaled_all) * resid_scale).astype(np.float32)
    final_prediction = np.full_like(arrays["y"], np.nan, dtype=np.float32)
    for split in ["train", "validation", "untouched_holdout"]:
        mask = splits == split
        if split == "train":
            base = np.where(np.isfinite(oof_train_prediction), oof_train_prediction, np.nan)
            final_prediction[mask] = (base[mask] + residual_prediction[mask]).astype(np.float32)
        else:
            final_prediction[mask] = (cpu_predictions[split] + residual_prediction[mask]).astype(np.float32)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "model_state": model.state_dict(),
        "model_kind": "residual_gru",
        "seed": seed,
        "residual_scale": resid_scale,
        "best_epoch": best_epoch,
        "output_dim": output_dim,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    roundtrip_model = ResidualGRU(arrays["X"].shape[2], 32, output_dim).to(device)
    roundtrip_model.load_state_dict(loaded["model_state"])
    roundtrip_model.eval()
    with torch.no_grad():
        rt = []
        for start in range(0, len(arrays["X"]), 512):
            xb = torch.tensor(arrays["X"][start : start + 512], dtype=torch.float32, device=device)
            rt.append(roundtrip_model(xb).cpu().numpy())
    rt_resid = (np.vstack(rt) * resid_scale).astype(np.float32)
    metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": epoch,
        "best_validation_loss": best_val_loss,
        "residual_scale": ";".join(str(float(x)) for x in resid_scale),
        "initial_cpu_reproduction_max_abs_diff": initial_cpu_reproduction_max_abs_diff,
        "checkpoint_roundtrip_status": "ok",
        "checkpoint_roundtrip_max_abs_diff": float(np.nanmax(np.abs(rt_resid - residual_prediction))),
    }
    return ResidualRun(seed, final_prediction, residual_prediction, metrics, checkpoint_path)


def validation_metrics_for_seed(
    seed: int,
    y: np.ndarray,
    final_pred: np.ndarray,
    residual_pred: np.ndarray,
    cpu_pred: np.ndarray,
    target_columns: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cpu_rmse = vector_rmse(y, cpu_pred)
    final_rmse = vector_rmse(y, final_pred)
    cpu_mae = vector_mae(y, cpu_pred)
    final_mae = vector_mae(y, final_pred)
    residual_actual = y - cpu_pred
    corr_stats = finite_stats(residual_pred)
    target_stats = finite_stats(y)
    pred_stats = finite_stats(final_pred)
    slope, intercept = calibration(y, final_pred)
    row = {
        "seed": seed,
        "locked_cpu_vector_rmse": cpu_rmse,
        "final_vector_rmse": final_rmse,
        "vector_rmse_improvement_vs_cpu": (cpu_rmse - final_rmse) / cpu_rmse if cpu_rmse > 0 else math.nan,
        "locked_cpu_vector_mae": cpu_mae,
        "final_vector_mae": final_mae,
        "vector_mae_improvement_vs_cpu": (cpu_mae - final_mae) / cpu_mae if cpu_mae > 0 else math.nan,
        "pearson_correlation": pearson_corr(y, final_pred),
        "spearman_correlation": spearman_corr(y, final_pred),
        "residual_pearson_correlation": pearson_corr(residual_actual, residual_pred),
        "residual_spearman_correlation": spearman_corr(residual_actual, residual_pred),
        "prediction_std": pred_stats["std"],
        "prediction_variance_ratio": (pred_stats["std"] ** 2) / (target_stats["std"] ** 2) if target_stats["std"] > 0 else math.nan,
        "residual_correction_mean": corr_stats["mean"],
        "residual_correction_std": corr_stats["std"],
        "residual_prediction_std": corr_stats["std"],
        "residual_correction_rms": corr_stats["rms"],
        "residual_correction_p95_abs": corr_stats["p95_abs"],
        "residual_correction_max_abs": corr_stats["max_abs"],
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "constant_prediction_flag": bool(finite_stats(residual_pred)["std"] <= 1e-9),
        "low_monotonic_violation_fraction": low_monotonic_violation(final_pred),
    }
    horizon_rows = []
    for idx, column in enumerate(target_columns):
        actual = y[:, idx]
        pred = final_pred[:, idx]
        cpu = cpu_pred[:, idx]
        cpu_h_rmse = float(np.sqrt(np.nanmean((actual - cpu) ** 2)))
        pred_h_rmse = float(np.sqrt(np.nanmean((actual - pred) ** 2)))
        cpu_h_mae = float(np.nanmean(np.abs(actual - cpu)))
        pred_h_mae = float(np.nanmean(np.abs(actual - pred)))
        horizon_rows.append(
            {
                "seed": seed,
                "target_column": column,
                "locked_cpu_rmse": cpu_h_rmse,
                "final_rmse": pred_h_rmse,
                "rmse_improvement_vs_cpu": (cpu_h_rmse - pred_h_rmse) / cpu_h_rmse if cpu_h_rmse > 0 else math.nan,
                "locked_cpu_mae": cpu_h_mae,
                "final_mae": pred_h_mae,
                "mae_improvement_vs_cpu": (cpu_h_mae - pred_h_mae) / cpu_h_mae if cpu_h_mae > 0 else math.nan,
                "pearson_correlation": pearson_corr(actual, pred),
                "spearman_correlation": spearman_corr(actual, pred),
                "residual_spearman_correlation": spearman_corr(actual - cpu, residual_pred[:, idx]),
            }
        )
    return row, horizon_rows


def main() -> int:
    tournament_dir = env_path("RAWSEQ_RESIDUAL_TOURNAMENT_DIR", DEFAULT_TOURNAMENT_DIR)
    diag_dir = env_path("RAWSEQ_RESIDUAL_DIAG_DIR", DEFAULT_DIAG_DIR)
    indicator_dir = env_path("RAWSEQ_RESIDUAL_INDICATOR_DIR", DEFAULT_INDICATOR_DIR)
    output_root = env_path("RAWSEQ_RESIDUAL_OUTPUT_DIR", DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / f"rawseq_low_path_residual_gru_screen_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    rows_per_split = int(float(os.getenv("RAWSEQ_RESIDUAL_ROWS_PER_SPLIT", "20000")))
    seq_len = int(float(os.getenv("RAWSEQ_RESIDUAL_SEQ_LEN", "60")))
    seeds = parse_ints("RAWSEQ_RESIDUAL_SEEDS", "900,901,902,903,904")
    epochs = int(float(os.getenv("RAWSEQ_RESIDUAL_EPOCHS", "10")))
    patience = int(float(os.getenv("RAWSEQ_RESIDUAL_EARLY_STOP_PATIENCE", "3")))
    require_cuda = env_bool("RAWSEQ_RESIDUAL_REQUIRE_CUDA", True)
    correction_l2 = float(os.getenv("RAWSEQ_RESIDUAL_CORRECTION_L2", "0.0001"))

    selected = load_selected_payload(tournament_dir)
    target_lane = str(selected["target_lane"])
    target_layout = str(selected["target_layout"])
    target_columns = [str(x) for x in selected["target_column_order"]]
    if target_lane != "future_low_from_now_bps_path" or target_layout != "scalar_path":
        raise SystemExit(f"Expected low scalar path target, got {target_lane}/{target_layout}")
    feature_bundle = "full_registered"
    feature_columns = load_feature_columns(diag_dir, feature_bundle)
    source_csv = indicator_dir / "multi_horizon_training_table.csv"
    header = pd.read_csv(source_csv, nrows=0).columns.tolist()
    usecols = list(dict.fromkeys(["split", "decision_timestamp", "close", "price", *[c for c in feature_columns if c in header]]))
    table = pd.read_csv(source_csv, usecols=usecols)
    table["source_row_index"] = np.arange(len(table), dtype=np.int64)
    table = materialized_copy(table, target_lane, target_columns)
    sample, block_rows = latest_contiguous_source_for_final_sequences(table, rows_per_split, seq_len, target_columns)
    arrays = build_blocked_sequences(sample, feature_columns, target_columns, seq_len)
    final_sequence_counts = {
        split: int(np.sum(arrays["splits"].astype(str) == split))
        for split in ["train", "validation", "untouched_holdout"]
    }
    for block in block_rows:
        block["final_sequence_rows"] = final_sequence_counts.get(str(block["split"]), 0)
    write_csv(out_dir / "rawseq_low_path_residual_sampling_blocks.csv", block_rows)
    required_sequence_counts = {
        split: count
        for split, count in final_sequence_counts.items()
        if split in {"train", "validation"} or count > 0
    }
    final_sequence_count_pass = all(count == rows_per_split for count in required_sequence_counts.values())
    direct_train_arrays = split_arrays(arrays, "train")
    direct_validation_arrays = split_arrays(arrays, "validation")
    direct_train_x = direct_train_arrays["x"]
    direct_train_y = direct_train_arrays["y"]
    direct_val_x = direct_validation_arrays["x"]
    direct_val_y = direct_validation_arrays["y"]
    ridge_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    elastic_grid = [0.001, 0.01, 0.1]
    direct_rows, _ = cpu_evaluate_arrays(
        f"latest_contiguous_{rows_per_split}_direct_table_preflight",
        direct_train_x,
        direct_train_y,
        direct_val_x,
        direct_val_y,
        feature_columns,
        target_columns,
        direct_train_arrays["source_rows"],
        direct_validation_arrays["source_rows"],
        direct_train_arrays["timestamps"],
        direct_validation_arrays["timestamps"],
        ridge_grid,
        elastic_grid,
        40,
        "direct_table_latest_contiguous_before_npz",
    )
    mean_rmse = next(row["validation_vector_rmse"] for row in direct_rows if row["model"] == "training_mean_baseline")
    median_rmse = next(row["validation_vector_rmse"] for row in direct_rows if row["model"] == "training_median_baseline")
    ridge_pathology_rows = ridge_pathology_audit(direct_train_x, direct_train_y, direct_val_x, direct_val_y, feature_columns, target_columns, ridge_grid, mean_rmse, median_rmse)
    write_csv(out_dir / "rawseq_low_path_residual_ridge_pathology.csv", ridge_pathology_rows)
    selected_cpu_model, selected_alpha = select_cpu_model(direct_rows, ridge_pathology_rows)
    selected_row = next(row for row in direct_rows if row["model"] == selected_cpu_model)
    per_horizon_cpu = []
    selected_payload_tmp = fit_cpu_model(selected_cpu_model, direct_train_x, direct_train_y, selected_alpha)
    selected_val_pred_tmp = predict_cpu(selected_payload_tmp, direct_val_x)
    mean_payload = fit_cpu_model("training_mean_baseline", direct_train_x, direct_train_y, None)
    median_payload = fit_cpu_model("training_median_baseline", direct_train_x, direct_train_y, None)
    constant_best = mean_payload if mean_rmse <= median_rmse else median_payload
    constant_best_name = constant_best["model"]
    constant_best_pred = predict_cpu(constant_best, direct_val_x)
    horizon_wins = 0
    for idx, column in enumerate(target_columns):
        cpu_rmse = float(np.sqrt(np.nanmean((direct_val_y[:, idx] - selected_val_pred_tmp[:, idx]) ** 2)))
        const_rmse = float(np.sqrt(np.nanmean((direct_val_y[:, idx] - constant_best_pred[:, idx]) ** 2)))
        win = cpu_rmse < const_rmse
        horizon_wins += int(win)
        per_horizon_cpu.append({"target_column": column, "selected_cpu_rmse": cpu_rmse, "selected_constant_rmse": const_rmse, "beats_constant": win})
    best_constant_rmse = min(mean_rmse, median_rmse)
    vector_improvement_vs_best_constant = (best_constant_rmse - selected_row["validation_vector_rmse"]) / best_constant_rmse if best_constant_rmse > 0 else math.nan
    direct_preflight_pass = (
        final_sequence_count_pass
        and arrays["sequence_boundary_violations"] == 0
        and arrays["X"].shape[1] == seq_len
        and
        selected_cpu_model in {"ridge_multi_output", "elastic_net_multi_output"}
        and selected_row["validation_vector_rmse"] < best_constant_rmse
        and vector_improvement_vs_best_constant > 0.0
        and horizon_wins >= 3
        and all(str(row.get("ridge_pathology_status")) == "ok" for row in ridge_pathology_rows if safe_float(row.get("alpha")) == safe_float(selected_alpha) and selected_cpu_model == "ridge_multi_output")
    )
    write_csv(out_dir / "rawseq_low_path_residual_cpu_preflight.csv", direct_rows)
    write_csv(out_dir / "rawseq_low_path_residual_cpu_preflight_per_horizon.csv", per_horizon_cpu)
    if not direct_preflight_pass:
        rec = {
            "recommendation": "stop_or_redesign_low_path_target_or_use_long_horizon_only",
            "preflight_pass": False,
            "preflight_stage": "direct_table_before_npz",
            "selected_cpu_model": selected_cpu_model,
            "selected_alpha": selected_alpha,
            "selected_best_constant": constant_best_name,
            "mean_rmse": mean_rmse,
            "median_rmse": median_rmse,
            "selected_learned_rmse": selected_row["validation_vector_rmse"],
            "vector_improvement_vs_best_constant": vector_improvement_vs_best_constant,
            "horizon_wins": horizon_wins,
            "final_sequence_counts": final_sequence_counts,
            "sequence_boundary_violations": int(arrays["sequence_boundary_violations"]),
            "supports_restricted_long_horizon_h240_h480": bool(
                all(row["beats_constant"] for row in per_horizon_cpu if row["target_column"].endswith(("h240", "h480")))
            ),
            "npz_built": False,
            "cuda_launched": False,
            "holdout_used_for_selection": False,
            "paper_only": True,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
        }
        contract = {
            "npz_path": "",
            "npz_sha256": "",
            "sampling_policy": "latest_contiguous",
            "rows_per_split": rows_per_split,
            "seq_len": seq_len,
            "feature_bundle": feature_bundle,
            "target_lane": target_lane,
            "target_layout": target_layout,
            "target_columns": target_columns,
            "feature_columns_sha256": column_hash(feature_columns),
            "target_columns_sha256": column_hash(target_columns),
            "selected_cpu_model": selected_cpu_model,
            "selected_alpha": selected_alpha,
            "cpu_preflight_pass": False,
            "cpu_preflight_failure_reason": "direct_table_50k_cpu_gate_failed",
            "horizon_wins": horizon_wins,
            "final_sequence_counts": final_sequence_counts,
            "sequence_boundary_violations": int(arrays["sequence_boundary_violations"]),
            "npz_built": False,
            "cuda_launched": False,
            "holdout_used_for_selection": False,
            "paper_only": True,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
            "recommendation": rec,
        }
        contract["contract_sha256"] = stable_hash(contract)
        write_json(out_dir / "rawseq_low_path_residual_gru_contract.json", contract)
        report_lines = [
            "Rawseq low-path residual GRU screen",
            f"Output: {out_dir}",
            f"Sampling: latest_contiguous rows_per_split={rows_per_split} seq_len={seq_len}",
            f"Direct-table CPU preflight: fail selected={selected_cpu_model} alpha={selected_alpha} horizon_wins={horizon_wins}/4",
            f"Mean RMSE: {mean_rmse}",
            f"Median RMSE: {median_rmse}",
            f"Selected learned RMSE: {selected_row['validation_vector_rmse']}",
            f"Vector improvement vs best constant: {vector_improvement_vs_best_constant}",
            f"Final sequence counts: {final_sequence_counts}",
            f"Sequence boundary violations: {int(arrays['sequence_boundary_violations'])}",
            f"Restricted h240/h480 support: {rec['supports_restricted_long_horizon_h240_h480']}",
            "NPZ built: false",
            "GPU residual GRU: skipped_before_training",
            "Seed pass count: 0/0",
            "Holdout used for selection: false",
            f"Final recommendation: {rec['recommendation']}",
        ]
        (out_dir / "rawseq_low_path_residual_gru_stability_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        write_csv(out_dir / "rawseq_low_path_residual_gru_seed_metrics.csv", [])
        write_csv(out_dir / "rawseq_low_path_residual_gru_per_horizon_metrics.csv", [])
        write_json(out_dir / "rawseq_low_path_residual_gru_recommendation.json", rec)
        print("\n".join(report_lines))
        return 0

    npz_path = out_dir / f"rawseq_low_path_latest_contiguous_{rows_per_split}_seq{seq_len}.npz"
    expected_npz_bytes = int(
        arrays["X"].nbytes
        + arrays["y"].nbytes
        + arrays["missing_mask"].nbytes
        + arrays["decision_timestamps"].nbytes
        + arrays["source_row_indices"].nbytes
    )
    np.savez_compressed(
        npz_path,
        **arrays,
        feature_columns=np.asarray(feature_columns, dtype=str),
        target_columns=np.asarray(target_columns, dtype=str),
        target_lane=np.asarray(target_lane, dtype=str),
        target_layout=np.asarray(target_layout, dtype=str),
        horizon_buckets=np.asarray([int(c.rsplit("h", 1)[1]) for c in target_columns], dtype=np.int64),
    )
    actual_npz_bytes = npz_path.stat().st_size
    with np.load(npz_path, allow_pickle=False) as loaded_npz:
        loaded_x = loaded_npz["X"]
        loaded_y = loaded_npz["y"]
        loaded_splits = loaded_npz["splits"].astype(str)
    npz_reconstruction_match = (
        np.array_equal(loaded_splits, arrays["splits"].astype(str))
        and float(np.nanmax(np.abs(loaded_x - arrays["X"]))) <= 0.0
        and float(np.nanmax(np.abs(loaded_y - arrays["y"]))) <= 0.0
    )
    write_csv(out_dir / "rawseq_low_path_residual_sampling_blocks.csv", block_rows)

    train_arrays = split_arrays(arrays, "train")
    validation_arrays = split_arrays(arrays, "validation")
    sequence_rows, _ = cpu_evaluate_arrays(
        f"latest_contiguous_{rows_per_split}_npz_reconstructed_preflight",
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
        ridge_grid,
        elastic_grid,
        40,
        "reconstructed_npz_sequence_final_step_cpu_features",
    )
    write_csv(out_dir / "rawseq_low_path_residual_npz_cpu_preflight.csv", sequence_rows)
    seq_selected_cpu_model, seq_selected_alpha = select_cpu_model(sequence_rows, ridge_pathology_rows)
    seq_selected_row = next(row for row in sequence_rows if row["model"] == seq_selected_cpu_model)
    npz_cpu_parity_match = (
        seq_selected_cpu_model == selected_cpu_model
        and abs(seq_selected_row["validation_vector_rmse"] - selected_row["validation_vector_rmse"]) <= 1e-6
    )
    if not npz_cpu_parity_match:
        direct_npz_rmse_delta = seq_selected_row["validation_vector_rmse"] - selected_row["validation_vector_rmse"]
        rec = {
            "recommendation": "stop_or_redesign_low_path_target_or_use_long_horizon_only",
            "preflight_pass": False,
            "preflight_stage": "npz_parity",
            "selected_cpu_model": selected_cpu_model,
            "selected_direct_table_rmse": selected_row["validation_vector_rmse"],
            "npz_selected_cpu_model": seq_selected_cpu_model,
            "npz_selected_rmse": seq_selected_row["validation_vector_rmse"],
            "direct_npz_rmse_delta": direct_npz_rmse_delta,
            "npz_reconstruction_match": npz_reconstruction_match,
            "npz_cpu_parity_match": npz_cpu_parity_match,
            "npz_built": True,
            "expected_npz_bytes": expected_npz_bytes,
            "actual_npz_bytes": actual_npz_bytes,
            "cuda_launched": False,
            "holdout_used_for_selection": False,
            "paper_only": True,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
        }
        contract = {
            "npz_path": str(npz_path),
            "npz_sha256": file_sha256(npz_path),
            "sampling_policy": "latest_contiguous",
            "rows_per_split": rows_per_split,
            "seq_len": seq_len,
            "feature_bundle": feature_bundle,
            "target_lane": target_lane,
            "target_layout": target_layout,
            "target_columns": target_columns,
            "feature_columns_sha256": column_hash(feature_columns),
            "target_columns_sha256": column_hash(target_columns),
            "selected_cpu_model": selected_cpu_model,
            "selected_alpha": selected_alpha,
            "direct_table_selected_rmse": selected_row["validation_vector_rmse"],
            "npz_selected_rmse": seq_selected_row["validation_vector_rmse"],
            "direct_npz_rmse_delta": direct_npz_rmse_delta,
            "cpu_preflight_pass": False,
            "cpu_preflight_failure_reason": "npz_cpu_parity_mismatch",
            "npz_reconstruction_match": npz_reconstruction_match,
            "npz_cpu_parity_match": npz_cpu_parity_match,
            "expected_npz_bytes": expected_npz_bytes,
            "actual_npz_bytes": actual_npz_bytes,
            "cuda_launched": False,
            "holdout_used_for_selection": False,
            "paper_only": True,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
            "recommendation": rec,
        }
        contract["contract_sha256"] = stable_hash(contract)
        write_json(out_dir / "rawseq_low_path_residual_gru_contract.json", contract)
        write_json(out_dir / "rawseq_low_path_residual_gru_recommendation.json", rec)
        report_lines = [
            "Rawseq low-path residual GRU screen",
            f"Output: {out_dir}",
            f"NPZ: {npz_path}",
            f"Sampling: latest_contiguous rows_per_split={rows_per_split} seq_len={seq_len}",
            f"Direct-table CPU preflight: pass selected={selected_cpu_model} alpha={selected_alpha} horizon_wins={horizon_wins}/4",
            f"Direct selected RMSE: {selected_row['validation_vector_rmse']}",
            f"NPZ selected RMSE: {seq_selected_row['validation_vector_rmse']}",
            f"Direct-NPZ RMSE delta: {direct_npz_rmse_delta}",
            f"NPZ reconstruction match: {npz_reconstruction_match}",
            "NPZ CPU parity: false",
            "GPU residual GRU: skipped_before_training",
            "Seed pass count: 0/0",
            "Holdout used for selection: false",
            f"Final recommendation: {rec['recommendation']}",
        ]
        (out_dir / "rawseq_low_path_residual_gru_stability_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        write_csv(out_dir / "rawseq_low_path_residual_gru_seed_metrics.csv", [])
        write_csv(out_dir / "rawseq_low_path_residual_gru_per_horizon_metrics.csv", [])
        print("\n".join(report_lines))
        return 0

    train_mask = arrays["splits"].astype(str) == "train"
    validation_mask = arrays["splits"].astype(str) == "validation"
    holdout_mask = arrays["splits"].astype(str) == "untouched_holdout"
    x_train = arrays["X"][train_mask, -1, :].astype(np.float64)
    y_train = arrays["y"][train_mask].astype(np.float64)
    oof_pred_train, oof_usable_train, oof_manifest = generate_oof_cpu_predictions(x_train, y_train, selected_cpu_model, selected_alpha, folds=5)
    full_cpu_payload = fit_cpu_model(selected_cpu_model, x_train, y_train, selected_alpha)
    cpu_val = predict_cpu(full_cpu_payload, arrays["X"][validation_mask, -1, :].astype(np.float64))
    cpu_holdout = predict_cpu(full_cpu_payload, arrays["X"][holdout_mask, -1, :].astype(np.float64)) if holdout_mask.any() else np.empty((0, len(target_columns)))
    cpu_predictions_full = np.full_like(arrays["y"], np.nan, dtype=np.float64)
    cpu_predictions_full[train_mask] = oof_pred_train
    cpu_predictions_full[validation_mask] = cpu_val
    if holdout_mask.any():
        cpu_predictions_full[holdout_mask] = cpu_holdout
    write_csv(out_dir / "rawseq_low_path_residual_cpu_oof_manifest.csv", oof_manifest)
    np.savez_compressed(
        out_dir / "rawseq_low_path_residual_cpu_predictions.npz",
        cpu_prediction=cpu_predictions_full.astype(np.float32),
        oof_usable_train=oof_usable_train,
        target=arrays["y"].astype(np.float32),
        splits=arrays["splits"].astype(str),
        target_columns=np.asarray(target_columns, dtype=str),
        selected_cpu_model=np.asarray(selected_cpu_model, dtype=str),
        selected_alpha=np.asarray(selected_alpha if selected_alpha is not None else np.nan, dtype=np.float64),
    )
    cpu_contract = {
        "selected_cpu_model": selected_cpu_model,
        "selected_alpha": selected_alpha,
        "feature_columns_sha256": column_hash(feature_columns),
        "target_columns_sha256": column_hash(target_columns),
        "feature_scaler_hash": stable_hash({"mean": full_cpu_payload["scaler"]["mean"].tolist(), "std": full_cpu_payload["scaler"]["std"].tolist()}),
        "prediction_sha256": array_sha256(cpu_predictions_full.astype(np.float32)),
        "fold_count": 5,
        "npz_reconstruction_match": npz_reconstruction_match,
        "holdout_used_for_selection": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    cpu_contract["contract_sha256"] = stable_hash(cpu_contract)
    write_json(out_dir / "rawseq_low_path_residual_cpu_contract.json", cpu_contract)

    seed_rows = []
    horizon_rows = []
    checkpoint_dir = out_dir / "checkpoints"
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        split_cpu = {
            "validation": cpu_val.astype(np.float32),
            "untouched_holdout": cpu_holdout.astype(np.float32),
        }
        run = run_residual_gru_seed(
            seed,
            arrays,
            split_cpu,
            cpu_predictions_full.astype(np.float32),
            oof_usable_train,
            len(target_columns),
            epochs,
            patience,
            require_cuda,
            checkpoint_dir / f"residual_gru_s{seed}.pt",
            correction_l2,
        )
        val_row, val_horizon = validation_metrics_for_seed(
            seed,
            arrays["y"][validation_mask].astype(np.float64),
            run.final_prediction[validation_mask].astype(np.float64),
            run.residual_prediction[validation_mask].astype(np.float64),
            cpu_val.astype(np.float64),
            target_columns,
        )
        val_row.update(run.metrics)
        val_row["checkpoint_path"] = str(run.checkpoint_path)
        val_row["checkpoint_sha256"] = file_sha256(run.checkpoint_path) if run.checkpoint_path else ""
        seed_rows.append(val_row)
        horizon_rows.extend(val_horizon)
        np.savez_compressed(
            pred_dir / f"residual_gru_s{seed}.npz",
            final_prediction=run.final_prediction.astype(np.float32),
            residual_prediction=run.residual_prediction.astype(np.float32),
            cpu_prediction=cpu_predictions_full.astype(np.float32),
            target=arrays["y"].astype(np.float32),
            splits=arrays["splits"].astype(str),
            target_columns=np.asarray(target_columns, dtype=str),
        )
    seed_df = pd.DataFrame(seed_rows)
    horizon_df = pd.DataFrame(horizon_rows)
    accepted = seed_df[
        (seed_df["vector_rmse_improvement_vs_cpu"] > 0.0)
        & (~seed_df["constant_prediction_flag"].astype(bool))
        & (seed_df["checkpoint_roundtrip_status"].astype(str).eq("ok"))
    ].copy()
    per_h_median = horizon_df.groupby("target_column")["rmse_improvement_vs_cpu"].median().to_dict() if len(horizon_df) else {}
    pass_count = int(len(accepted))
    median_improvement = float(seed_df["vector_rmse_improvement_vs_cpu"].median()) if len(seed_df) else math.nan
    positive_horizons = int(sum(1 for value in per_h_median.values() if safe_float(value) > 0.0))
    median_resid_spearman = float(seed_df["residual_spearman_correlation"].median()) if len(seed_df) else math.nan
    bounded_corrections = bool(np.isfinite(seed_df["residual_correction_p95_abs"]).all() and (seed_df["residual_correction_p95_abs"] < 1000.0).all())
    gate_pass = (
        pass_count >= 4
        and median_improvement >= 0.005
        and positive_horizons >= 3
        and median_resid_spearman > 0.0
        and bounded_corrections
        and seed_df["checkpoint_roundtrip_status"].astype(str).eq("ok").all()
    )
    write_csv(out_dir / "rawseq_low_path_residual_gru_seed_metrics.csv", seed_rows)
    write_csv(out_dir / "rawseq_low_path_residual_gru_per_horizon_metrics.csv", horizon_rows)
    recommendation = {
        "recommendation": "run_low_path_residual_gru_ten_seed_replication" if gate_pass else "compare_linear_and_temporal_mlp_residual_models",
        "preflight_pass": True,
        "seed_pass_count": pass_count,
        "seed_count": len(seeds),
        "median_vector_improvement_vs_cpu": median_improvement,
        "per_horizon_median_improvements": {str(k): safe_float(v) for k, v in per_h_median.items()},
        "positive_horizon_median_count": positive_horizons,
        "median_residual_spearman_correlation": median_resid_spearman,
        "correction_magnitudes_bounded": bounded_corrections,
        "holdout_used_for_selection": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    write_json(out_dir / "rawseq_low_path_residual_gru_recommendation.json", recommendation)
    contract = {
        "npz_path": str(npz_path),
        "npz_sha256": file_sha256(npz_path),
        "sampling_policy": "latest_contiguous",
        "rows_per_split": rows_per_split,
        "final_sequence_counts": final_sequence_counts,
        "seq_len": seq_len,
        "feature_bundle": feature_bundle,
        "target_lane": target_lane,
        "target_columns": target_columns,
        "selected_cpu_model": selected_cpu_model,
        "selected_alpha": selected_alpha,
        "locked_cpu_vector_rmse": selected_row["validation_vector_rmse"],
        "npz_cpu_vector_rmse": seq_selected_row["validation_vector_rmse"],
        "npz_cpu_parity_match": npz_cpu_parity_match,
        "npz_reconstruction_match": npz_reconstruction_match,
        "sequence_boundary_violations": int(arrays["sequence_boundary_violations"]),
        "expected_npz_bytes": expected_npz_bytes,
        "actual_npz_bytes": actual_npz_bytes,
        "model_kind": "residual_gru",
        "epochs": epochs,
        "early_stopping_patience": patience,
        "seeds": seeds,
        "loss": "normalized_huber_residual_std_only",
        "residual_output_zero_initialized": True,
        "holdout_used_for_selection": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "recommendation": recommendation,
    }
    contract["contract_sha256"] = stable_hash(contract)
    write_json(out_dir / "rawseq_low_path_residual_gru_contract.json", contract)
    lines = [
        "Rawseq low-path residual GRU screen",
        f"Output: {out_dir}",
        f"NPZ: {npz_path}",
        f"Sampling: latest_contiguous rows_per_split={rows_per_split} seq_len={seq_len}",
        f"CPU preflight: pass selected={selected_cpu_model} alpha={selected_alpha} horizon_wins={horizon_wins}/4",
        f"Final sequence counts: {final_sequence_counts}",
        f"NPZ reconstruction match: {npz_reconstruction_match}",
        f"NPZ CPU parity: {npz_cpu_parity_match}",
        f"OOF usable train rows: {int(oof_usable_train.sum())}/{len(oof_usable_train)}",
        f"Seed pass count: {pass_count}/{len(seeds)}",
        f"Median vector improvement vs CPU: {median_improvement}",
        f"Per-horizon median improvements: {recommendation['per_horizon_median_improvements']}",
        f"Checkpoint roundtrip statuses: {sorted(seed_df['checkpoint_roundtrip_status'].astype(str).unique().tolist())}",
        "Holdout used for selection: false",
        f"Final recommendation: {recommendation['recommendation']}",
    ]
    (out_dir / "rawseq_low_path_residual_gru_stability_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
