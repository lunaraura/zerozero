#!/usr/bin/env python3
"""Freeze CPU-only downside-risk exceedance research candidate.

Target:
    P(abs(future_range_low_bps_h) / recent_realized_volatility_bps > 0.5)

This script is paper/research only. It uses rolling-fold validation for model
selection, freezes the selected CPU candidate on pre-future-holdout development
data, writes a future acceptance gate before any future evaluation, and does not
touch champions, private APIs, orders, Torch, or GPU.
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

from scripts.tiny.freeze_rawseq_low_path_ridge_research_candidate import array_sha256, file_sha256, stable_hash
from scripts.tiny.report_rawseq_downside_target_redesign import (
    EPSILON,
    build_folds,
    calibration_error,
    make_targets,
    pr_auc,
    rank_auc,
    regime_labels,
    volatility_bucket_masks,
)


DEFAULT_REDESIGN_DIR = PROJECT_ROOT / "data" / "research" / "rawseq_downside_target_redesign" / "rawseq_downside_target_redesign_20260711T231208Z"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_cpu_candidates"
TARGET_NAME = "downside_exceedance_probability_gt_0_5vol"
TARGET_THRESHOLD_VOL_UNITS = 0.5
HORIZONS = [60, 120, 240, 480]


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
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


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fit_scaler(x: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}


def transform(x: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    out = (np.where(np.isfinite(x), x, scaler["mean"]) - scaler["mean"]) / scaler["std"]
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float64)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logistic_coef(x_scaled: np.ndarray, y: np.ndarray, l2: float = 0.01, iterations: int = 100) -> np.ndarray:
    y = (np.asarray(y, dtype=np.float64).ravel() > 0.5).astype(np.float64)
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    coef = np.zeros(design.shape[1], dtype=np.float64)
    if y.mean() > 0 and y.mean() < 1:
        coef[0] = math.log(y.mean() / (1 - y.mean()))
    for _ in range(iterations):
        p = sigmoid(design @ coef)
        grad = (design.T @ (p - y)) / max(len(y), 1)
        grad[1:] += l2 * coef[1:]
        coef -= 0.1 * grad
    return coef.astype(np.float64)


def predict_logistic_coef(x_scaled: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return sigmoid(np.column_stack([np.ones(len(x_scaled)), x_scaled]) @ coef)


def fit_multihorizon_logistic(x_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    scaler = fit_scaler(x_train)
    xs = transform(x_train, scaler)
    coefs = []
    for idx in range(y_train.shape[1]):
        coefs.append(fit_logistic_coef(xs, y_train[:, idx]))
    return {"scaler": scaler, "coef": np.column_stack(coefs)}


def predict_multihorizon_logistic(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    xs = transform(x, model["scaler"])
    preds = [predict_logistic_coef(xs, model["coef"][:, idx]) for idx in range(model["coef"].shape[1])]
    return np.column_stack(preds).astype(np.float64)


def fit_platt_from_scores(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slopes = np.ones(y.shape[1], dtype=np.float64)
    intercepts = np.zeros(y.shape[1], dtype=np.float64)
    for idx in range(y.shape[1]):
        coef = fit_logistic_coef(scores[:, [idx]], y[:, idx], l2=0.001, iterations=80)
        intercepts[idx] = coef[0]
        slopes[idx] = coef[1]
    return slopes, intercepts


def apply_platt(prob: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    scores = logit(prob)
    return sigmoid(scores * slopes + intercepts)


def reliability_rows(fold_id: str, model: str, target_column: str, y: np.ndarray, prob: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    rows = []
    y = np.asarray(y, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    for idx, lo in enumerate(np.linspace(0, 1, bins, endpoint=False)):
        hi = lo + 1.0 / bins
        mask = (prob >= lo) & (prob < hi if idx < bins - 1 else prob <= hi)
        rows.append(
            {
                "fold_id": fold_id,
                "model": model,
                "target_column": target_column,
                "bin_index": idx,
                "bin_low": lo,
                "bin_high": hi,
                "rows": int(mask.sum()),
                "predicted_probability_mean": float(np.mean(prob[mask])) if mask.any() else math.nan,
                "event_rate": float(np.mean(y[mask])) if mask.any() else math.nan,
                "absolute_calibration_error": abs(float(np.mean(prob[mask]) - np.mean(y[mask]))) if mask.any() else math.nan,
            }
        )
    return rows


def calibration_slope_intercept(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    y = (np.asarray(y, dtype=np.float64).ravel() > 0.5).astype(np.float64)
    scores = logit(np.asarray(prob, dtype=np.float64).ravel())[:, None]
    if y.min() == y.max() or np.nanstd(scores) <= 1e-12:
        return math.nan, math.nan
    coef = fit_logistic_coef(scores, y, l2=0.001, iterations=80)
    return float(coef[1]), float(coef[0])


def binary_metric_row(
    fold_id: str,
    model: str,
    y: np.ndarray,
    prob: np.ndarray,
    const_prob: np.ndarray,
    target_columns: list[str],
    feature_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    horizon_rows = []
    reliability = []
    skills = []
    pr_lifts = []
    cal_errors = []
    max_cal_errors = []
    slopes = []
    intercepts = []
    wins = 0
    for idx, col in enumerate(target_columns):
        yy = y[:, idx].astype(np.float64)
        pp = np.clip(prob[:, idx].astype(np.float64), 1e-6, 1 - 1e-6)
        cc = np.clip(const_prob[:, idx].astype(np.float64), 1e-6, 1 - 1e-6)
        brier = float(np.mean((pp - yy) ** 2))
        const_brier = float(np.mean((cc - yy) ** 2))
        skill = (const_brier - brier) / const_brier if const_brier > 0 else math.nan
        prevalence = float(np.mean(yy))
        pr = pr_auc(yy, pp)
        rel = reliability_rows(fold_id, model, col, yy, pp)
        max_cal = max([safe_float(r["absolute_calibration_error"], 0.0) for r in rel], default=math.nan)
        slope, intercept = calibration_slope_intercept(yy, pp)
        skills.append(skill)
        pr_lifts.append(pr - prevalence if math.isfinite(pr) else math.nan)
        cal_errors.append(calibration_error(yy, pp))
        max_cal_errors.append(max_cal)
        slopes.append(slope)
        intercepts.append(intercept)
        wins += int(skill > 0)
        reliability.extend(rel)
        horizon_rows.append(
            {
                "fold_id": fold_id,
                "model": model,
                "target_column": col,
                "event_prevalence": prevalence,
                "brier_score": brier,
                "constant_prevalence_brier_score": const_brier,
                "brier_skill_score": skill,
                "log_loss": float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))),
                "roc_auc": rank_auc(yy, pp),
                "pr_auc": pr,
                "pr_auc_prevalence_baseline": prevalence,
                "pr_auc_lift_over_prevalence": pr - prevalence if math.isfinite(pr) else math.nan,
                "expected_calibration_error": cal_errors[-1],
                "maximum_calibration_error": max_cal,
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                "probability_mean": float(np.mean(pp)),
                "probability_std": float(np.std(pp)),
                "prediction_unique_fraction": float(len(np.unique(np.round(pp, 8))) / len(pp)),
                "feature_hash": feature_hash,
                "prediction_hash": array_sha256(pp),
            }
        )
    row = {
        "fold_id": fold_id,
        "model": model,
        "target_name": TARGET_NAME,
        "event_prevalence": float(np.mean(y)),
        "fold_horizon_wins": wins,
        "mean_brier_skill_score": float(np.nanmean(skills)),
        "median_pr_auc_lift_over_prevalence": float(np.nanmedian(pr_lifts)),
        "mean_expected_calibration_error": float(np.nanmean(cal_errors)),
        "max_calibration_error": float(np.nanmax(max_cal_errors)),
        "median_calibration_slope": float(np.nanmedian(slopes)),
        "median_calibration_intercept": float(np.nanmedian(intercepts)),
        "probability_mean": float(np.mean(prob)),
        "probability_std": float(np.std(prob)),
        "prediction_unique_fraction": float(len(np.unique(np.round(prob.ravel(), 8))) / prob.size),
        "feature_hash": feature_hash,
        "prediction_hash": array_sha256(prob),
    }
    return row, horizon_rows, reliability


def model_predictions(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_idx: np.ndarray,
    vol_values: np.ndarray,
) -> np.ndarray:
    x_train = x[train_idx][:, feature_idx]
    x_val = x[val_idx][:, feature_idx]
    y_train = y[train_idx]
    if model_name == "constant_prevalence_baseline":
        prevalence = np.nanmean(y_train, axis=0)
        return np.tile(np.clip(prevalence, 1e-6, 1 - 1e-6), (len(val_idx), 1))
    if model_name in {"logistic_regression", "logistic_regression_with_continuous_regime_features"}:
        model = fit_multihorizon_logistic(x_train, y_train)
        return predict_multihorizon_logistic(model, x_val)
    if model_name in {"calibrated_logistic_regression", "calibrated_logistic_regression_with_continuous_regime_features"}:
        split = max(100, int(len(train_idx) * 0.8))
        inner_train = np.arange(0, split)
        inner_cal = np.arange(split, len(train_idx))
        base = fit_multihorizon_logistic(x_train[inner_train], y_train[inner_train])
        cal_prob = predict_multihorizon_logistic(base, x_train[inner_cal])
        slopes, intercepts = fit_platt_from_scores(logit(cal_prob), y_train[inner_cal])
        full_base = fit_multihorizon_logistic(x_train, y_train)
        return apply_platt(predict_multihorizon_logistic(full_base, x_val), slopes, intercepts)
    if model_name == "volatility_bucket_logistic":
        out = np.zeros((len(val_idx), y.shape[1]), dtype=np.float64)
        buckets = volatility_bucket_masks(vol_values, train_idx, val_idx)
        fallback = fit_multihorizon_logistic(x_train, y_train)
        fallback_pred = predict_multihorizon_logistic(fallback, x_val)
        out[:] = fallback_pred
        for _, (train_mask, val_mask) in buckets.items():
            if train_mask.sum() < 100 or val_mask.sum() == 0:
                continue
            bucket_y = y_train[train_mask]
            if np.any(np.nanstd(bucket_y, axis=0) <= 1e-12):
                continue
            model = fit_multihorizon_logistic(x_train[train_mask], bucket_y)
            out[val_mask] = predict_multihorizon_logistic(model, x_val[val_mask])
        return out
    raise ValueError(model_name)


def select_feature_indices(feature_columns: list[str], with_regime: bool) -> np.ndarray:
    regime_tokens = ["volatility", "range", "volume", "slope", "return", "bucket_return", "raw_range"]
    if with_regime:
        return np.arange(len(feature_columns))
    keep = [idx for idx, col in enumerate(feature_columns) if not any(tok in col.lower() for tok in regime_tokens)]
    return np.asarray(keep if keep else list(range(len(feature_columns))), dtype=np.int64)


def policy_threshold_rows(y: np.ndarray, prob: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    yy = y.ravel().astype(np.float64)
    pp = prob.ravel().astype(np.float64)
    event_count = max(float(np.sum(yy > 0.5)), 1.0)
    non_event_count = max(float(np.sum(yy <= 0.5)), 1.0)
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        reject = pp >= threshold
        tp = float(np.sum(reject & (yy > 0.5)))
        fp = float(np.sum(reject & (yy <= 0.5)))
        rows.append(
            {
                "threshold": threshold,
                "event_recall": tp / event_count,
                "event_precision": tp / max(float(reject.sum()), 1.0),
                "false_positive_rate": fp / non_event_count,
                "fraction_opportunities_rejected": float(np.mean(reject)),
                "adverse_events_avoided": tp,
                "favorable_opportunities_rejected": fp,
            }
        )
    return rows


def main() -> int:
    redesign_dir = env_path("RAWSEQ_DOWNSIDE_RISK_REDESIGN_DIR", DEFAULT_REDESIGN_DIR)
    output_root = env_path("RAWSEQ_DOWNSIDE_RISK_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / f"rawseq_downside_risk_cpu_candidate_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    redesign_contract = json.loads((redesign_dir / "rawseq_downside_target_redesign_contract.json").read_text(encoding="utf-8-sig"))
    npz_path = Path(redesign_contract["source_npz"])
    with np.load(npz_path, allow_pickle=False) as data:
        x_seq = data["X"].astype(np.float32)
        y_raw = data["y"].astype(np.float64)
        splits = data["splits"].astype(str)
        timestamps = data["decision_timestamps"].astype(np.float64)
        source_rows = data["source_row_indices"].astype(np.int64)
        feature_columns = [str(x) for x in data["feature_columns"]]
        target_columns = [str(x) for x in data["target_columns"]]
    x = x_seq[:, -1, :].astype(np.float64)
    pre_idx = np.flatnonzero(np.isin(splits, ["train", "validation"]))
    holdout_count = int(np.sum(splits == "untouched_holdout"))
    vol_col = "realized_volatility_bps_fw60" if "realized_volatility_bps_fw60" in feature_columns else "rolling_volatility_bps_fw60"
    vol = np.maximum(np.abs(x[:, feature_columns.index(vol_col)].astype(np.float64)), EPSILON)
    targets = make_targets(y_raw[pre_idx], vol[pre_idx], np.arange(0, 50000))
    y = targets[TARGET_NAME]["values"].astype(np.float64)
    x_pre = x[pre_idx]
    ts_pre = timestamps[pre_idx]
    source_pre = source_rows[pre_idx]
    folds = build_folds(len(pre_idx), ts_pre, source_pre)
    regimes = regime_labels(x, feature_columns)
    vol_pre = regimes["volatility"][pre_idx]
    candidate_models = [
        "constant_prevalence_baseline",
        "logistic_regression",
        "calibrated_logistic_regression",
        "logistic_regression_with_continuous_regime_features",
        "calibrated_logistic_regression_with_continuous_regime_features",
        "volatility_bucket_logistic",
    ]
    fold_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    oof_pred_by_model = {model: np.full_like(y, np.nan, dtype=np.float64) for model in candidate_models}
    for fold in folds:
        train_idx = fold["train_indices"]
        val_idx = fold["validation_indices"]
        const_pred = model_predictions("constant_prevalence_baseline", x_pre, y, train_idx, val_idx, np.arange(len(feature_columns)), vol_pre)
        for model in candidate_models:
            with_regime = "regime" in model or model == "volatility_bucket_logistic"
            feature_idx = select_feature_indices(feature_columns, with_regime)
            prob = const_pred if model == "constant_prevalence_baseline" else model_predictions(model, x_pre, y, train_idx, val_idx, feature_idx, vol_pre)
            oof_pred_by_model[model][val_idx] = prob
            row, hrows, rel = binary_metric_row(fold["fold_id"], model, y[val_idx], prob, const_pred, target_columns, array_sha256(x_pre[train_idx][:, feature_idx]))
            row.update(
                {
                    "feature_count": int(len(feature_idx)),
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(val_idx)),
                    "train_timestamp_start": float(ts_pre[train_idx[0]]),
                    "train_timestamp_end": float(ts_pre[train_idx[-1]]),
                    "validation_timestamp_start": float(ts_pre[val_idx[0]]),
                    "validation_timestamp_end": float(ts_pre[val_idx[-1]]),
                }
            )
            fold_rows.append(row)
            horizon_rows.extend(hrows)
            reliability.extend(rel)
    df = pd.DataFrame(fold_rows)
    ranking_rows = []
    for model, group in df[df["model"] != "constant_prevalence_baseline"].groupby("model"):
        skills = pd.to_numeric(group["mean_brier_skill_score"], errors="coerce")
        wins = int((skills > 0).sum())
        med_skill = float(skills.median())
        worst_skill = float(skills.min())
        cal_stability = float(pd.to_numeric(group["mean_expected_calibration_error"], errors="coerce").std())
        pr_lift = float(pd.to_numeric(group["median_pr_auc_lift_over_prevalence"], errors="coerce").median())
        finite_cal = bool(np.isfinite(pd.to_numeric(group["median_calibration_slope"], errors="coerce")).all())
        nonconstant = bool((pd.to_numeric(group["prediction_unique_fraction"], errors="coerce") > 0.001).all())
        passes = wins >= 4 and med_skill > 0 and worst_skill > -0.05 and cal_stability <= 0.05 and finite_cal and nonconstant
        ranking_rows.append(
            {
                "model": model,
                "fold_win_count": wins,
                "median_brier_skill_score": med_skill,
                "worst_fold_brier_skill_score": worst_skill,
                "calibration_stability_ece_std": cal_stability,
                "median_pr_auc_lift_over_prevalence": pr_lift,
                "finite_calibration_metrics": finite_cal,
                "nonconstant_probabilities": nonconstant,
                "selection_gate_pass": passes,
            }
        )
    ranking_rows = sorted(ranking_rows, key=lambda r: (int(r["selection_gate_pass"]), r["fold_win_count"], r["median_brier_skill_score"], r["worst_fold_brier_skill_score"], -r["calibration_stability_ece_std"], r["median_pr_auc_lift_over_prevalence"]), reverse=True)
    selected = ranking_rows[0]
    selected_model = selected["model"]
    selected_calibration = "platt_scaling" if selected_model.startswith("calibrated_") else "none"
    selected_regime_policy = "predefined_low_medium_high_volatility_models" if selected_model == "volatility_bucket_logistic" else ("continuous_regime_features" if "regime" in selected_model else "global")
    write_csv(out_dir / "rawseq_downside_risk_cpu_candidate_fold_metrics.csv", fold_rows)
    write_csv(out_dir / "rawseq_downside_risk_cpu_candidate_per_horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "rawseq_downside_risk_model_selection.csv", ranking_rows)
    write_csv(out_dir / "rawseq_downside_risk_reliability_bins.csv", reliability)
    write_csv(out_dir / "rawseq_downside_risk_calibration_metrics.csv", horizon_rows)
    calibration_contract = {
        "selected_calibration_method": selected_calibration,
        "available_methods_compared": ["none", "platt_scaling"],
        "isotonic_regression": "not_selected; reserved for later if sample sufficiency and leakage-safe implementation are approved",
        "calibration_fitting": "inner chronological calibration partition for calibrated candidates; none for selected uncalibrated candidate",
        "holdout_used_for_selection": False,
    }
    calibration_contract["calibration_contract_sha256"] = stable_hash(calibration_contract)
    write_json(out_dir / "rawseq_downside_risk_calibration_contract.json", calibration_contract)

    feature_idx = select_feature_indices(feature_columns, "regime" in selected_model or selected_model == "volatility_bucket_logistic")
    if selected_model == "volatility_bucket_logistic":
        final_model = fit_multihorizon_logistic(x_pre[:, feature_idx], y)
        # Store fallback/global model; bucket policy is preserved for next script.
    elif selected_model.startswith("calibrated_"):
        final_model = fit_multihorizon_logistic(x_pre[:, feature_idx], y)
    else:
        final_model = fit_multihorizon_logistic(x_pre[:, feature_idx], y)
    model_path = out_dir / "rawseq_downside_risk_cpu_candidate_model.npz"
    scalers_path = out_dir / "rawseq_downside_risk_cpu_candidate_scalers.npz"
    np.savez_compressed(
        model_path,
        coef=final_model["coef"],
        selected_model=np.asarray(selected_model),
        selected_calibration=np.asarray(selected_calibration),
        selected_feature_indices=feature_idx,
        feature_columns=np.asarray(feature_columns),
        target_columns=np.asarray(target_columns),
        target_threshold_vol_units=np.asarray([TARGET_THRESHOLD_VOL_UNITS], dtype=np.float64),
    )
    np.savez_compressed(scalers_path, feature_scaler_mean=final_model["scaler"]["mean"], feature_scaler_std=final_model["scaler"]["std"])
    reload_model = {
        "coef": np.load(model_path, allow_pickle=False)["coef"].astype(np.float64),
        "scaler": {
            "mean": np.load(scalers_path, allow_pickle=False)["feature_scaler_mean"].astype(np.float64),
            "std": np.load(scalers_path, allow_pickle=False)["feature_scaler_std"].astype(np.float64),
        },
    }
    dev_pred = predict_multihorizon_logistic(final_model, x_pre[:, feature_idx])
    reload_pred = predict_multihorizon_logistic(reload_model, x_pre[:, feature_idx])
    reload_diff = float(np.nanmax(np.abs(dev_pred - reload_pred)))
    selected_oof = oof_pred_by_model[selected_model]
    threshold_rows = policy_threshold_rows(y[np.isfinite(selected_oof).all(axis=1)], selected_oof[np.isfinite(selected_oof).all(axis=1)])
    threshold_rows_sorted = sorted(threshold_rows, key=lambda r: (r["event_recall"] >= 0.70, -r["false_positive_rate"], r["event_precision"]), reverse=True)
    primary = threshold_rows_sorted[0] if threshold_rows_sorted else {}
    backup = threshold_rows_sorted[1] if len(threshold_rows_sorted) > 1 else {}
    for row in threshold_rows:
        row["selected_primary"] = bool(row == primary)
        row["selected_backup"] = bool(row == backup)
    write_csv(out_dir / "rawseq_downside_risk_policy_thresholds.csv", threshold_rows)
    policy_contract = {
        "risk_score": "P(downside excursion over horizon exceeds 0.5 recent-volatility units)",
        "use_cases": ["reject proposed trades above fixed probability threshold", "reduce size as probability rises", "rank otherwise valid entries by downside risk", "compare risk score with realized adverse excursion"],
        "not_a_directional_entry_signal": True,
        "primary_threshold": primary,
        "backup_threshold": backup,
        "threshold_selection_source": "rolling-fold out-of-fold development predictions only",
        "holdout_used_for_selection": False,
    }
    policy_contract["policy_contract_sha256"] = stable_hash(policy_contract)
    write_json(out_dir / "rawseq_downside_risk_policy_contract.json", policy_contract)
    (out_dir / "rawseq_downside_risk_policy_report.txt").write_text(
        "\n".join(
            [
                "Downside-risk policy threshold report",
                f"Primary threshold: {primary}",
                f"Backup threshold: {backup}",
                "Policy scope: paper-only risk filter / sizing research, not directional entry.",
                "Thresholds selected from rolling-fold OOF predictions only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    contract = {
        "designation": "frozen CPU downside-risk research score candidate",
        "target_name": TARGET_NAME,
        "target_formula": "abs(future_range_low_bps_h) / max(recent_realized_volatility_bps, epsilon) > 0.5",
        "volatility_denominator_formula": f"{vol_col} at decision timestamp, epsilon={EPSILON}",
        "target_threshold_vol_units": TARGET_THRESHOLD_VOL_UNITS,
        "horizons": HORIZONS,
        "feature_columns": [feature_columns[i] for i in feature_idx],
        "feature_column_indices": feature_idx.tolist(),
        "selected_model": selected_model,
        "selected_hyperparameters": {"l2": 0.01, "iterations": 100, "optimizer": "batch_gradient_descent"},
        "selected_calibration_method": selected_calibration,
        "regime_policy": selected_regime_policy,
        "fold_evidence": selected,
        "training_cutoff_timestamp": float(ts_pre[-1]),
        "source_npz": str(npz_path),
        "source_npz_sha256": file_sha256(npz_path),
        "feature_columns_sha256": stable_hash([feature_columns[i] for i in feature_idx]),
        "target_hash": array_sha256(y),
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "scalers_path": str(scalers_path),
        "scalers_sha256": file_sha256(scalers_path),
        "prediction_hash": array_sha256(dev_pred),
        "holdout_used_for_selection": False,
        "consumed_holdout_reused": False,
        "paper_only": True,
        "promotion_scope": "downside_risk_research_score",
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    contract["contract_sha256"] = stable_hash(contract)
    write_json(out_dir / "rawseq_downside_risk_cpu_candidate_contract.json", contract)
    manifest = {
        "created_at": now_stamp(),
        "output_dir": str(out_dir),
        "contract_path": str(out_dir / "rawseq_downside_risk_cpu_candidate_contract.json"),
        "contract_sha256": contract["contract_sha256"],
        "model_sha256": contract["model_sha256"],
        "scalers_sha256": contract["scalers_sha256"],
        "holdout_used_for_selection": False,
        "paper_only": True,
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    write_json(out_dir / "rawseq_downside_risk_cpu_candidate_manifest.json", manifest)
    smoke = {"save_reload_prediction_max_abs_diff": reload_diff, "save_reload_prediction_equal": bool(reload_diff <= 0.0), "development_rows": int(len(pre_idx))}
    write_json(out_dir / "rawseq_downside_risk_cpu_candidate_inference_smoke.json", smoke)
    acceptance_rule = {
        "rule_name": "downside_risk_future_acceptance_gate",
        "frozen_at_iso": datetime.now(UTC).isoformat(),
        "requirements": {
            "positive_brier_skill_vs_frozen_prevalence_baseline": True,
            "positive_brier_skill_min_folds": "4/5 development folds already required before freeze",
            "min_log_loss_improvement": 0.0,
            "min_pr_auc_lift_over_prevalence": 0.01,
            "calibration_slope_min": 0.5,
            "calibration_slope_max": 1.5,
            "calibration_intercept_abs_max": 0.5,
            "expected_calibration_error_max": 0.08,
            "nonconstant_probability_predictions": True,
            "feature_and_target_contract_match": True,
            "distribution_shift_max_population_stability_index": 0.25,
            "minimum_event_count": 100,
            "minimum_effective_non_overlapping_observations": 300,
            "no_future_data_used_for_selection": True,
        },
        "contract_sha256": contract["contract_sha256"],
        "holdout_used_for_selection": False,
        "paper_only": True,
    }
    acceptance_rule["acceptance_rule_sha256"] = stable_hash(acceptance_rule)
    write_json(out_dir / "rawseq_downside_risk_future_acceptance_rule.json", acceptance_rule)
    gate_pass = bool(selected["selection_gate_pass"]) and reload_diff <= 0.0 and acceptance_rule["acceptance_rule_sha256"] and primary and not contract["holdout_used_for_selection"]
    recommendation = {
        "recommendation": "start_downside_risk_future_paper_shadow" if gate_pass else "continue_cpu_downside_risk_research",
        "selected_model": selected_model,
        "selected_calibration_method": selected_calibration,
        "fold_win_count": selected["fold_win_count"],
        "median_brier_skill_score": selected["median_brier_skill_score"],
        "worst_fold_brier_skill_score": selected["worst_fold_brier_skill_score"],
        "median_pr_auc_lift_over_prevalence": selected["median_pr_auc_lift_over_prevalence"],
        "save_reload_prediction_max_abs_diff": reload_diff,
        "future_acceptance_rule_hash": acceptance_rule["acceptance_rule_sha256"],
        "primary_policy_threshold": primary,
        "backup_policy_threshold": backup,
        "future_holdout_used_for_selection": False,
        "consumed_holdout_reused": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    recommendation["recommendation_sha256"] = stable_hash(recommendation)
    write_json(out_dir / "rawseq_downside_risk_cpu_candidate_recommendation.json", recommendation)
    print("Rawseq downside-risk CPU candidate freeze")
    print(f"Output: {out_dir}")
    print(f"Selected model: {selected_model}")
    print(f"Selected calibration: {selected_calibration}")
    print(f"Fold wins: {selected['fold_win_count']}/5")
    print(f"Median Brier skill: {selected['median_brier_skill_score']}")
    print(f"Worst-fold Brier skill: {selected['worst_fold_brier_skill_score']}")
    print(f"Median PR AUC lift over prevalence: {selected['median_pr_auc_lift_over_prevalence']}")
    print(f"Regime policy: {selected_regime_policy}")
    print(f"Contract hash: {contract['contract_sha256']}")
    print(f"Save/reload max diff: {reload_diff}")
    print(f"Future acceptance-rule hash: {acceptance_rule['acceptance_rule_sha256']}")
    print(f"Primary threshold: {primary}")
    print(f"Backup threshold: {backup}")
    print("Future/holdout used for selection: false")
    print(f"Final recommendation: {recommendation['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
