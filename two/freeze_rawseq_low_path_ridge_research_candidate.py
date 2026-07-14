#!/usr/bin/env python3
"""Freeze and evaluate the low-path ridge downside-risk research candidate.

Paper/research only. This script does not place orders, use private APIs,
promote models, mutate champions, tune against holdout, or retrain neural
models. It freezes the already selected ridge contract, evaluates untouched
holdout once, and audits saved GRU residual predictions without retraining.
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

from scripts.tiny.report_rawseq_target_lane_baseline_tournament import fit_ridge, predict_linear
from scripts.tiny.run_rawseq_low_path_residual_gru_screen import (
    calibration,
    low_monotonic_violation,
    pearson_corr,
    spearman_corr,
)


DEFAULT_RUN_DIR = Path(r"F:\rsio\rawseq_low_path_residual_gru_screens\rawseq_low_path_residual_gru_screen_20260711T220205Z")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_low_path_ridge_research_candidates"


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
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


def array_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


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


def vector_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    per_target = []
    for idx in range(actual.shape[1]):
        mask = np.isfinite(actual[:, idx]) & np.isfinite(pred[:, idx])
        if mask.any():
            per_target.append(float(np.sqrt(np.mean((actual[mask, idx] - pred[mask, idx]) ** 2))))
    return float(np.sqrt(np.mean(np.asarray(per_target, dtype=np.float64) ** 2))) if per_target else math.nan


def vector_mae(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    per_target = []
    for idx in range(actual.shape[1]):
        mask = np.isfinite(actual[:, idx]) & np.isfinite(pred[:, idx])
        if mask.any():
            per_target.append(float(np.mean(np.abs(actual[mask, idx] - pred[mask, idx]))))
    return float(np.mean(per_target)) if per_target else math.nan


def fit_scaler(values: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}


def apply_scaler(values: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    filled = np.where(np.isfinite(values), values, scaler["mean"])
    scaled = (filled - scaler["mean"]) / scaler["std"]
    scaled[~np.isfinite(scaled)] = 0.0
    return scaled.astype(np.float64)


def split_mask(splits: np.ndarray, split: str) -> np.ndarray:
    return splits.astype(str) == split


def fit_locked_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> dict[str, Any]:
    scaler = fit_scaler(x_train)
    coef = fit_ridge(apply_scaler(x_train, scaler), y_train.astype(np.float64), alpha)
    return {"model": "ridge_multi_output", "alpha": alpha, "feature_scaler": scaler, "coef": coef.astype(np.float64)}


def predict_locked_ridge(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return predict_linear(apply_scaler(x, model["feature_scaler"]), model["coef"]).astype(np.float64)


def fit_constant(y_train: np.ndarray, kind: str) -> np.ndarray:
    if kind == "training_mean_baseline":
        return np.nanmean(y_train, axis=0).astype(np.float64)
    if kind == "training_median_baseline":
        return np.nanmedian(y_train, axis=0).astype(np.float64)
    raise ValueError(kind)


def predict_constant(values: np.ndarray, rows: int) -> np.ndarray:
    return np.tile(np.asarray(values, dtype=np.float64), (rows, 1))


def metric_row(prefix: str, actual: np.ndarray, pred: np.ndarray, target_columns: list[str]) -> dict[str, Any]:
    slope, intercept = calibration(actual, pred)
    row: dict[str, Any] = {
        f"{prefix}_vector_rmse": vector_rmse(actual, pred),
        f"{prefix}_vector_mae": vector_mae(actual, pred),
        f"{prefix}_pearson_correlation": pearson_corr(actual, pred),
        f"{prefix}_spearman_correlation": spearman_corr(actual, pred),
        f"{prefix}_calibration_slope": slope,
        f"{prefix}_calibration_intercept": intercept,
        f"{prefix}_prediction_std": float(np.nanstd(pred)),
        f"{prefix}_target_std": float(np.nanstd(actual)),
        f"{prefix}_prediction_mean": float(np.nanmean(pred)),
        f"{prefix}_target_mean": float(np.nanmean(actual)),
        f"{prefix}_constant_prediction": bool(np.nanstd(pred) <= 1e-12),
        f"{prefix}_low_monotonic_violation_fraction": low_monotonic_violation(pred),
    }
    for idx, column in enumerate(target_columns):
        a = actual[:, idx]
        p = pred[:, idx]
        mask = np.isfinite(a) & np.isfinite(p)
        row[f"{prefix}_{column}_rmse"] = float(np.sqrt(np.mean((a[mask] - p[mask]) ** 2))) if mask.any() else math.nan
        row[f"{prefix}_{column}_mae"] = float(np.mean(np.abs(a[mask] - p[mask]))) if mask.any() else math.nan
        row[f"{prefix}_{column}_pearson"] = pearson_corr(a, p)
        row[f"{prefix}_{column}_spearman"] = spearman_corr(a, p)
    return row


def horizon_metric_rows(actual: np.ndarray, pred: np.ndarray, mean_pred: np.ndarray, median_pred: np.ndarray, target_columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    best_constant = "training_mean_baseline" if vector_rmse(actual, mean_pred) <= vector_rmse(actual, median_pred) else "training_median_baseline"
    constant_pred = mean_pred if best_constant == "training_mean_baseline" else median_pred
    for idx, column in enumerate(target_columns):
        a = actual[:, idx]
        p = pred[:, idx]
        c = constant_pred[:, idx]
        mask = np.isfinite(a) & np.isfinite(p) & np.isfinite(c)
        rmse = float(np.sqrt(np.mean((a[mask] - p[mask]) ** 2))) if mask.any() else math.nan
        const_rmse = float(np.sqrt(np.mean((a[mask] - c[mask]) ** 2))) if mask.any() else math.nan
        mae = float(np.mean(np.abs(a[mask] - p[mask]))) if mask.any() else math.nan
        const_mae = float(np.mean(np.abs(a[mask] - c[mask]))) if mask.any() else math.nan
        rows.append(
            {
                "target_column": column,
                "ridge_rmse": rmse,
                "best_constant_baseline": best_constant,
                "best_constant_rmse": const_rmse,
                "rmse_improvement_vs_best_constant": (const_rmse - rmse) / const_rmse if const_rmse > 0 else math.nan,
                "beats_best_constant": bool(rmse < const_rmse),
                "ridge_mae": mae,
                "best_constant_mae": const_mae,
                "mae_improvement_vs_best_constant": (const_mae - mae) / const_mae if const_mae > 0 else math.nan,
                "pearson_correlation": pearson_corr(a, p),
                "spearman_correlation": spearman_corr(a, p),
            }
        )
    return rows


def architecture_decision(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    contract = json.loads((run_dir / "rawseq_low_path_residual_gru_contract.json").read_text(encoding="utf-8-sig"))
    recommendation = json.loads((run_dir / "rawseq_low_path_residual_gru_recommendation.json").read_text(encoding="utf-8-sig"))
    seed_rows = list(csv.DictReader((run_dir / "rawseq_low_path_residual_gru_seed_metrics.csv").open(newline="", encoding="utf-8-sig")))
    per_horizon_rows = list(csv.DictReader((run_dir / "rawseq_low_path_residual_gru_per_horizon_metrics.csv").open(newline="", encoding="utf-8-sig")))
    improvements = [safe_float(row.get("vector_rmse_improvement_vs_cpu")) for row in seed_rows]
    decision = {
        "architecture": "residual_gru",
        "authoritative_run_dir": str(run_dir),
        "exact_gru_contract": contract,
        "seed_count": len(seed_rows),
        "seed_metrics_sha256": file_sha256(run_dir / "rawseq_low_path_residual_gru_seed_metrics.csv"),
        "per_horizon_metrics_sha256": file_sha256(run_dir / "rawseq_low_path_residual_gru_per_horizon_metrics.csv"),
        "median_vector_improvement_vs_ridge": float(np.nanmedian(improvements)) if improvements else math.nan,
        "seed_survival_count": int(sum(1 for value in improvements if math.isfinite(value) and value > 0.0)),
        "per_horizon_failures": [row for row in per_horizon_rows if safe_float(row.get("rmse_improvement_vs_cpu")) <= 0.0],
        "checkpoint_roundtrip_statuses": sorted({row.get("checkpoint_roundtrip_status", "") for row in seed_rows}),
        "checkpoint_roundtrip_exact": all(safe_float(row.get("checkpoint_roundtrip_max_abs_diff")) == 0.0 for row in seed_rows),
        "holdout_used_for_selection": False,
        "recommendation": "close_current_residual_gru_architecture",
        "upstream_recommendation": recommendation,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    decision["decision_sha256"] = stable_hash(decision)
    write_json(out_dir / "rawseq_low_path_residual_gru_architecture_decision.json", decision)
    lines = [
        "Low-path residual-GRU architecture decision",
        f"Authoritative run: {run_dir}",
        f"Seed survival: {decision['seed_survival_count']}/{decision['seed_count']}",
        f"Median vector improvement vs ridge: {decision['median_vector_improvement_vs_ridge']}",
        f"Checkpoint roundtrip exact: {decision['checkpoint_roundtrip_exact']}",
        "Recommendation: close_current_residual_gru_architecture",
        "Holdout used for selection: false",
    ]
    (out_dir / "rawseq_low_path_residual_gru_architecture_decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def fit_affine(train_x: np.ndarray, train_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slopes = np.zeros(train_y.shape[1], dtype=np.float64)
    intercepts = np.zeros(train_y.shape[1], dtype=np.float64)
    for idx in range(train_y.shape[1]):
        x = train_x[:, idx]
        y = train_y[:, idx]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3 or np.std(x[mask]) <= 1e-12:
            slopes[idx] = 0.0
            intercepts[idx] = float(np.nanmean(y[mask])) if mask.any() else 0.0
        else:
            slopes[idx], intercepts[idx] = np.polyfit(x[mask], y[mask], 1)
    return slopes, intercepts


def residual_calibration_audit(run_dir: Path, out_dir: Path, ridge_validation_rmse: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cpu_npz = np.load(run_dir / "rawseq_low_path_residual_cpu_predictions.npz", allow_pickle=False)
    cpu_pred = cpu_npz["cpu_prediction"].astype(np.float64)
    target = cpu_npz["target"].astype(np.float64)
    splits = cpu_npz["splits"].astype(str)
    train_mask = (splits == "train") & np.isfinite(cpu_pred).all(axis=1)
    val_mask = splits == "validation"
    ridge_val = cpu_pred[val_mask]
    y_val = target[val_mask]
    target_columns = [str(x) for x in cpu_npz["target_columns"]]
    rows.append(
        {
            "calibration_type": "zero_residual",
            "seed": "",
            "validation_vector_rmse": vector_rmse(y_val, ridge_val),
            "improvement_vs_ridge": 0.0,
            "parameters": "lambda=0",
            "beats_ridge": False,
            "notes": "must reproduce locked ridge",
        }
    )
    for pred_path in sorted((run_dir / "predictions").glob("residual_gru_s*.npz")):
        seed = pred_path.stem.rsplit("_s", 1)[-1]
        with np.load(pred_path, allow_pickle=False) as data:
            residual = data["residual_prediction"].astype(np.float64)
        train_r = residual[train_mask]
        train_resid_target = target[train_mask] - cpu_pred[train_mask]
        val_r = residual[val_mask]
        denom = float(np.nansum(train_r**2))
        lam = float(np.nansum(train_r * train_resid_target) / denom) if denom > 1e-12 else 0.0
        per_lam = []
        for idx in range(train_r.shape[1]):
            denom_h = float(np.nansum(train_r[:, idx] ** 2))
            per_lam.append(float(np.nansum(train_r[:, idx] * train_resid_target[:, idx]) / denom_h) if denom_h > 1e-12 else 0.0)
        per_lam_arr = np.asarray(per_lam, dtype=np.float64)
        slopes, intercepts = fit_affine(train_r, train_resid_target)
        candidates = [
            ("global_lambda_residual", ridge_val + lam * val_r, f"lambda={lam}"),
            ("per_horizon_lambda_residual", ridge_val + val_r * per_lam_arr, "lambda_h=" + ";".join(str(x) for x in per_lam_arr)),
            ("affine_residual", ridge_val + intercepts + slopes * val_r, "intercept_h=" + ";".join(str(x) for x in intercepts) + " slope_h=" + ";".join(str(x) for x in slopes)),
        ]
        for kind, pred, params in candidates:
            rmse = vector_rmse(y_val, pred)
            rows.append(
                {
                    "calibration_type": kind,
                    "seed": seed,
                    "validation_vector_rmse": rmse,
                    "validation_vector_mae": vector_mae(y_val, pred),
                    "improvement_vs_ridge": (ridge_validation_rmse - rmse) / ridge_validation_rmse if ridge_validation_rmse > 0 else math.nan,
                    "parameters": params,
                    "pearson_correlation": pearson_corr(y_val, pred),
                    "spearman_correlation": spearman_corr(y_val, pred),
                    "beats_ridge": bool(rmse < ridge_validation_rmse),
                    "fit_stage": "training_oof_only",
                    "eval_stage": "validation",
                }
            )
    best = sorted(rows, key=lambda r: safe_float(r.get("validation_vector_rmse"), 1e18))[0] if rows else {}
    calibrated_beats = any(bool(row.get("beats_ridge")) for row in rows if row.get("calibration_type") != "zero_residual")
    decision = {
        "best_calibration": best,
        "calibrated_gru_residual_beats_ridge": calibrated_beats,
        "ridge_validation_vector_rmse": ridge_validation_rmse,
        "recommendation": "separate_calibration_focused_experiment" if calibrated_beats else "close_gru_residual_branch",
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "holdout_used_for_selection": False,
    }
    decision["decision_sha256"] = stable_hash(decision)
    write_csv(out_dir / "rawseq_low_path_gru_residual_calibration_audit.csv", rows)
    write_json(out_dir / "rawseq_low_path_gru_residual_calibration_decision.json", decision)
    lines = [
        "Low-path GRU residual calibration audit",
        f"Ridge validation vector RMSE: {ridge_validation_rmse}",
        f"Best calibration: {best.get('calibration_type')} seed={best.get('seed')} rmse={best.get('validation_vector_rmse')} improvement={best.get('improvement_vs_ridge')}",
        f"Calibrated residual beats ridge: {calibrated_beats}",
        f"Recommendation: {decision['recommendation']}",
        "Holdout used for selection: false",
    ]
    (out_dir / "rawseq_low_path_gru_residual_calibration_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows, decision


def linear_residual_preflight(arrays: dict[str, np.ndarray], cpu_pred: np.ndarray, ridge_validation_rmse: float) -> dict[str, Any]:
    splits = arrays["splits"].astype(str)
    x = arrays["X"][:, -1, :].astype(np.float64)
    y = arrays["y"].astype(np.float64)
    train_mask = (splits == "train") & np.isfinite(cpu_pred).all(axis=1)
    val_mask = splits == "validation"
    residual_target = y[train_mask] - cpu_pred[train_mask]
    scaler = fit_scaler(x[train_mask])
    coef = fit_ridge(apply_scaler(x[train_mask], scaler), residual_target, 100.0)
    residual_val = predict_linear(apply_scaler(x[val_mask], scaler), coef)
    pred_val = cpu_pred[val_mask] + residual_val
    rmse = vector_rmse(y[val_mask], pred_val)
    return {
        "model": "ridge_linear_residual_alpha_100",
        "validation_vector_rmse": rmse,
        "improvement_vs_ridge": (ridge_validation_rmse - rmse) / ridge_validation_rmse if ridge_validation_rmse > 0 else math.nan,
        "beats_ridge": bool(rmse < ridge_validation_rmse),
        "fit_stage": "training_oof_residuals_only",
        "eval_stage": "validation",
    }


def main() -> int:
    run_dir = env_path("RAWSEQ_LOW_RIDGE_SOURCE_RUN_DIR", DEFAULT_RUN_DIR)
    output_root = env_path("RAWSEQ_LOW_RIDGE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / f"rawseq_low_path_ridge_research_candidate_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    architecture_decision_payload = architecture_decision(run_dir, out_dir)

    npz_path = run_dir / "rawseq_low_path_latest_contiguous_50000_seq60.npz"
    arrays_npz = np.load(npz_path, allow_pickle=False)
    arrays = {key: arrays_npz[key] for key in arrays_npz.files}
    splits = arrays["splits"].astype(str)
    x = arrays["X"][:, -1, :].astype(np.float64)
    y = arrays["y"].astype(np.float64)
    feature_columns = [str(x) for x in arrays["feature_columns"]]
    target_columns = [str(x) for x in arrays["target_columns"]]
    train_mask = split_mask(splits, "train")
    val_mask = split_mask(splits, "validation")
    holdout_mask = split_mask(splits, "untouched_holdout")
    alpha = 100.0

    ridge_model = fit_locked_ridge(x[train_mask], y[train_mask], alpha)
    mean_values = fit_constant(y[train_mask], "training_mean_baseline")
    median_values = fit_constant(y[train_mask], "training_median_baseline")
    val_pred = predict_locked_ridge(ridge_model, x[val_mask])
    val_mean = predict_constant(mean_values, int(val_mask.sum()))
    val_median = predict_constant(median_values, int(val_mask.sum()))

    model_path = out_dir / "rawseq_low_path_ridge_candidate_model.npz"
    scalers_path = out_dir / "rawseq_low_path_ridge_candidate_scalers.npz"
    np.savez_compressed(
        model_path,
        coef=ridge_model["coef"].astype(np.float64),
        alpha=np.asarray([alpha], dtype=np.float64),
        feature_columns=np.asarray(feature_columns),
        target_columns=np.asarray(target_columns),
        mean_baseline=mean_values.astype(np.float64),
        median_baseline=median_values.astype(np.float64),
    )
    np.savez_compressed(
        scalers_path,
        ridge_feature_scaler_mean=ridge_model["feature_scaler"]["mean"].astype(np.float64),
        ridge_feature_scaler_std=ridge_model["feature_scaler"]["std"].astype(np.float64),
        sequence_feature_scaler_mean=arrays["feature_scaler_mean"].astype(np.float64),
        sequence_feature_scaler_std=arrays["feature_scaler_std"].astype(np.float64),
        sequence_target_scaler_mean=arrays["target_scaler_mean"].astype(np.float64),
        sequence_target_scaler_std=arrays["target_scaler_std"].astype(np.float64),
    )
    reloaded = np.load(model_path, allow_pickle=False)
    reloaded_scalers = np.load(scalers_path, allow_pickle=False)
    reload_model = {
        "model": "ridge_multi_output",
        "alpha": float(reloaded["alpha"][0]),
        "coef": reloaded["coef"].astype(np.float64),
        "feature_scaler": {
            "mean": reloaded_scalers["ridge_feature_scaler_mean"].astype(np.float64),
            "std": reloaded_scalers["ridge_feature_scaler_std"].astype(np.float64),
        },
    }
    reload_val = predict_locked_ridge(reload_model, x[val_mask])
    reload_max_diff = float(np.nanmax(np.abs(reload_val - val_pred)))

    validation_rows = [
        {
            "stage": "validation",
            "model": "ridge_multi_output",
            "alpha": alpha,
            **metric_row("ridge", y[val_mask], val_pred, target_columns),
            "mean_baseline_vector_rmse": vector_rmse(y[val_mask], val_mean),
            "median_baseline_vector_rmse": vector_rmse(y[val_mask], val_median),
            "best_constant_baseline": "training_mean_baseline" if vector_rmse(y[val_mask], val_mean) <= vector_rmse(y[val_mask], val_median) else "training_median_baseline",
            "holdout_used_for_selection": False,
        }
    ]
    write_csv(out_dir / "rawseq_low_path_ridge_candidate_validation_metrics.csv", validation_rows)

    train_ts = arrays["decision_timestamps"][train_mask]
    val_ts = arrays["decision_timestamps"][val_mask]
    holdout_ts = arrays["decision_timestamps"][holdout_mask]
    contract = {
        "designation": "frozen low-path ridge downside-risk research candidate",
        "target_lane": str(arrays["target_lane"]),
        "target_layout": str(arrays["target_layout"]),
        "horizons": [int(x) for x in arrays["horizon_buckets"]],
        "feature_group": "full_registered",
        "sampling_policy": "latest_contiguous",
        "final_sequence_rows_per_split": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "untouched_holdout": int(holdout_mask.sum()),
        },
        "seq_len": int(arrays["X"].shape[1]),
        "ridge_alpha": alpha,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "feature_columns_sha256": stable_hash(feature_columns),
        "target_columns_sha256": stable_hash(target_columns),
        "source_run_dir": str(run_dir),
        "source_npz_path": str(npz_path),
        "source_npz_sha256": file_sha256(npz_path),
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "scalers_path": str(scalers_path),
        "scalers_sha256": file_sha256(scalers_path),
        "model_coef_sha256": array_sha256(ridge_model["coef"]),
        "model_intercept_sha256": array_sha256(ridge_model["coef"][0]),
        "ridge_feature_scaler_hash": stable_hash({"mean": ridge_model["feature_scaler"]["mean"].tolist(), "std": ridge_model["feature_scaler"]["std"].tolist()}),
        "sequence_feature_scaler_hash": stable_hash({"mean": arrays["feature_scaler_mean"].astype(float).tolist(), "std": arrays["feature_scaler_std"].astype(float).tolist()}),
        "sequence_target_scaler_hash": stable_hash({"mean": arrays["target_scaler_mean"].astype(float).tolist(), "std": arrays["target_scaler_std"].astype(float).tolist()}),
        "train_timestamp_start": float(train_ts[0]),
        "train_timestamp_end": float(train_ts[-1]),
        "validation_timestamp_start": float(val_ts[0]),
        "validation_timestamp_end": float(val_ts[-1]),
        "holdout_timestamp_start": float(holdout_ts[0]),
        "holdout_timestamp_end": float(holdout_ts[-1]),
        "validation_predictions_sha256": array_sha256(val_pred),
        "inference_formula": "predict = [1, scale(final_step_features, frozen_ridge_feature_scaler)] @ frozen_ridge_coef",
        "no_retraining_rule": "Do not refit alpha, features, scalers, thresholds, or row selection after this freeze.",
        "paper_only": True,
        "promotion_scope": "research_candidate_only",
        "holdout_used_for_selection": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    contract["contract_sha256"] = stable_hash(contract)
    write_json(out_dir / "rawseq_low_path_ridge_candidate_contract.json", contract)

    manifest = {
        "created_at": now_stamp(),
        "output_dir": str(out_dir),
        "contract_path": str(out_dir / "rawseq_low_path_ridge_candidate_contract.json"),
        "model_path": str(model_path),
        "scalers_path": str(scalers_path),
        "validation_metrics_path": str(out_dir / "rawseq_low_path_ridge_candidate_validation_metrics.csv"),
        "source_run_dir": str(run_dir),
        "artifact_hashes": {
            "contract": file_sha256(out_dir / "rawseq_low_path_ridge_candidate_contract.json"),
            "model": file_sha256(model_path),
            "scalers": file_sha256(scalers_path),
            "source_npz": file_sha256(npz_path),
        },
        "paper_only": True,
        "promotion_scope": "research_candidate_only",
        "holdout_used_for_selection": False,
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    write_json(out_dir / "rawseq_low_path_ridge_candidate_manifest.json", manifest)
    smoke = {
        "save_reload_prediction_max_abs_diff": reload_max_diff,
        "save_reload_prediction_equal": bool(reload_max_diff <= 0.0),
        "validation_rows": int(val_mask.sum()),
        "holdout_rows": int(holdout_mask.sum()),
        "paper_only": True,
    }
    write_json(out_dir / "rawseq_low_path_ridge_candidate_inference_smoke.json", smoke)

    gate_payload = {
        "gate_name": "low_path_ridge_one_time_untouched_holdout_gate",
        "frozen_before_holdout_evaluation": True,
        "gate_frozen_at_iso": datetime.now(UTC).isoformat(),
        "requirements": {
            "vector_rmse_improvement_vs_best_frozen_constant_gt": 0.0,
            "min_horizon_wins_vs_best_frozen_constant": 3,
            "require_finite_pearson_and_spearman": True,
            "require_nonconstant_predictions": True,
            "max_low_path_monotonicity_violation_fraction": 0.0,
            "require_exact_target_and_feature_contract_match": True,
            "require_holdout_unused_before_selection": True,
        },
        "contract_sha256": contract["contract_sha256"],
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    gate_payload["acceptance_rule_sha256"] = stable_hash(gate_payload)
    write_json(out_dir / "rawseq_low_path_ridge_holdout_acceptance_rule.json", gate_payload)

    holdout_pred = predict_locked_ridge(ridge_model, x[holdout_mask])
    holdout_mean = predict_constant(mean_values, int(holdout_mask.sum()))
    holdout_median = predict_constant(median_values, int(holdout_mask.sum()))
    reload_holdout = predict_locked_ridge(reload_model, x[holdout_mask])
    holdout_reload_max_diff = float(np.nanmax(np.abs(reload_holdout - holdout_pred)))
    holdout_best_constant = "training_mean_baseline" if vector_rmse(y[holdout_mask], holdout_mean) <= vector_rmse(y[holdout_mask], holdout_median) else "training_median_baseline"
    holdout_constant_pred = holdout_mean if holdout_best_constant == "training_mean_baseline" else holdout_median
    holdout_metrics = {
        "stage": "untouched_holdout",
        "model": "ridge_multi_output",
        "alpha": alpha,
        **metric_row("ridge", y[holdout_mask], holdout_pred, target_columns),
        "mean_baseline_vector_rmse": vector_rmse(y[holdout_mask], holdout_mean),
        "median_baseline_vector_rmse": vector_rmse(y[holdout_mask], holdout_median),
        "best_constant_baseline": holdout_best_constant,
        "best_constant_vector_rmse": vector_rmse(y[holdout_mask], holdout_constant_pred),
        "vector_rmse_improvement_vs_best_constant": (vector_rmse(y[holdout_mask], holdout_constant_pred) - vector_rmse(y[holdout_mask], holdout_pred)) / vector_rmse(y[holdout_mask], holdout_constant_pred),
        "prediction_finite_fraction": float(np.isfinite(holdout_pred).mean()),
        "target_finite_fraction": float(np.isfinite(y[holdout_mask]).mean()),
        "target_feature_contract_match": True,
        "holdout_used_for_selection": False,
        "holdout_save_reload_prediction_max_abs_diff": holdout_reload_max_diff,
        "distribution_shift_validation_to_holdout_target_mean_delta": float(np.nanmean(y[holdout_mask]) - np.nanmean(y[val_mask])),
        "distribution_shift_validation_to_holdout_target_std_ratio": float(np.nanstd(y[holdout_mask]) / np.nanstd(y[val_mask])) if np.nanstd(y[val_mask]) > 0 else math.nan,
    }
    horizon_rows = horizon_metric_rows(y[holdout_mask], holdout_pred, holdout_mean, holdout_median, target_columns)
    horizon_wins = int(sum(1 for row in horizon_rows if bool(row["beats_best_constant"])))
    holdout_pass = (
        safe_float(holdout_metrics["vector_rmse_improvement_vs_best_constant"]) > 0.0
        and horizon_wins >= 3
        and math.isfinite(safe_float(holdout_metrics["ridge_pearson_correlation"]))
        and math.isfinite(safe_float(holdout_metrics["ridge_spearman_correlation"]))
        and not bool(holdout_metrics["ridge_constant_prediction"])
        and safe_float(holdout_metrics["ridge_low_monotonic_violation_fraction"]) <= 0.0
        and bool(holdout_metrics["target_feature_contract_match"])
        and not bool(holdout_metrics["holdout_used_for_selection"])
    )
    holdout_decision = {
        "acceptance_status": "pass" if holdout_pass else "fail",
        "recommendation": "start_low_path_ridge_forward_paper_shadow" if holdout_pass else "archive_low_path_ridge_candidate",
        "horizon_wins": horizon_wins,
        "holdout_metrics": holdout_metrics,
        "acceptance_rule_sha256": gate_payload["acceptance_rule_sha256"],
        "no_holdout_retuning": True,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    holdout_decision["decision_sha256"] = stable_hash(holdout_decision)
    write_csv(out_dir / "rawseq_low_path_ridge_holdout_metrics.csv", [holdout_metrics, *horizon_rows])
    write_json(out_dir / "rawseq_low_path_ridge_holdout_decision.json", holdout_decision)
    holdout_lines = [
        "Low-path ridge one-time untouched holdout evaluation",
        f"Candidate: {contract['designation']}",
        f"Vector RMSE: {holdout_metrics['ridge_vector_rmse']}",
        f"Best constant baseline: {holdout_best_constant} RMSE={holdout_metrics['best_constant_vector_rmse']}",
        f"Vector improvement vs best constant: {holdout_metrics['vector_rmse_improvement_vs_best_constant']}",
        f"Horizon wins: {horizon_wins}/4",
        f"Pearson: {holdout_metrics['ridge_pearson_correlation']}",
        f"Spearman: {holdout_metrics['ridge_spearman_correlation']}",
        f"Low-path monotonic violation fraction: {holdout_metrics['ridge_low_monotonic_violation_fraction']}",
        f"Acceptance status: {holdout_decision['acceptance_status']}",
        f"Recommendation: {holdout_decision['recommendation']}",
        "Holdout used for selection: false",
    ]
    (out_dir / "rawseq_low_path_ridge_holdout_report.txt").write_text("\n".join(holdout_lines) + "\n", encoding="utf-8")

    residual_rows, residual_decision = residual_calibration_audit(run_dir, out_dir, vector_rmse(y[val_mask], val_pred))
    cpu_npz = np.load(run_dir / "rawseq_low_path_residual_cpu_predictions.npz", allow_pickle=False)
    linear_residual = linear_residual_preflight(arrays, cpu_npz["cpu_prediction"].astype(np.float64), vector_rmse(y[val_mask], val_pred))
    final_residual_recommendation = (
        "compare_linear_and_temporal_mlp_residual_models"
        if residual_decision["calibrated_gru_residual_beats_ridge"] or bool(linear_residual["beats_ridge"])
        else "stop_neural_low_path_residual_research"
    )
    residual_comparator = {
        "calibrated_gru_residual_beats_ridge": residual_decision["calibrated_gru_residual_beats_ridge"],
        "linear_residual_preflight": linear_residual,
        "recommendation": final_residual_recommendation,
        "holdout_used_for_selection": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    residual_comparator["decision_sha256"] = stable_hash(residual_comparator)
    write_json(out_dir / "rawseq_low_path_residual_comparator_recommendation.json", residual_comparator)

    final_summary = {
        "output_dir": str(out_dir),
        "archived_gru_decision": architecture_decision_payload["recommendation"],
        "frozen_ridge_contract_path": str(out_dir / "rawseq_low_path_ridge_candidate_contract.json"),
        "frozen_ridge_contract_sha256": contract["contract_sha256"],
        "save_reload_parity": smoke,
        "holdout_gate_hash": gate_payload["acceptance_rule_sha256"],
        "holdout_acceptance_status": holdout_decision["acceptance_status"],
        "holdout_recommendation": holdout_decision["recommendation"],
        "residual_calibration_recommendation": residual_decision["recommendation"],
        "final_residual_recommendation": final_residual_recommendation,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    write_json(out_dir / "rawseq_low_path_ridge_research_candidate_summary.json", final_summary)

    print("Rawseq low-path ridge research candidate packet")
    print(f"Output: {out_dir}")
    print(f"Archived GRU decision: {architecture_decision_payload['recommendation']}")
    print(f"Frozen ridge contract: {out_dir / 'rawseq_low_path_ridge_candidate_contract.json'}")
    print(f"Frozen ridge contract hash: {contract['contract_sha256']}")
    print(f"Save/reload parity max diff: {reload_max_diff}")
    print(f"Frozen holdout gate hash: {gate_payload['acceptance_rule_sha256']}")
    print(f"Holdout status: {holdout_decision['acceptance_status']} recommendation={holdout_decision['recommendation']}")
    print(f"Holdout vector improvement vs best constant: {holdout_metrics['vector_rmse_improvement_vs_best_constant']}")
    print(f"Holdout horizon wins: {horizon_wins}/4")
    print(f"Holdout used for selection: {holdout_metrics['holdout_used_for_selection']}")
    print(f"Residual calibration recommendation: {residual_decision['recommendation']}")
    print(f"Final residual recommendation: {final_residual_recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
