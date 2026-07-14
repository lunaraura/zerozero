#!/usr/bin/env python3
"""Close failed low-path lineage and run CPU-only downside target redesign.

This script is diagnostic/research-only. It does not use Torch/GPU, does not
reuse the consumed holdout as a promotion holdout, does not mutate champions,
does not place orders, and does not promote any model.
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

from scripts.tiny.report_rawseq_target_lane_baseline_tournament import fit_elastic_net_proxy, fit_ridge, predict_linear
from scripts.tiny.freeze_rawseq_low_path_ridge_research_candidate import (
    array_sha256,
    calibration,
    file_sha256,
    low_monotonic_violation,
    pearson_corr,
    spearman_corr,
    stable_hash,
    vector_mae,
    vector_rmse,
)


DEFAULT_FAILED_PACKET = PROJECT_ROOT / "data" / "research" / "rawseq_low_path_ridge_research_candidates" / "rawseq_low_path_ridge_research_candidate_20260711T223946Z"
DEFAULT_SOURCE_RUN = Path(r"F:\rsio\rawseq_low_path_residual_gru_screens\rawseq_low_path_residual_gru_screen_20260711T220205Z")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_target_redesign"
EPSILON = 1e-6
EXCEEDANCE_LEVELS = [0.5, 1.0, 2.0]
HORIZONS = [60, 120, 240, 480]


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return resolve_path(raw) if raw else default


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


def fit_linear_family(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    model: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = fit_scaler(x_train)
    xtr = transform(x_train, scaler)
    xva = transform(x_val, scaler)
    if model == "ridge":
        alpha = 100.0
        coef = fit_ridge(xtr, y_train, alpha)
        return predict_linear(xva, coef), {"alpha": alpha, "scaler_hash": stable_hash({"mean": scaler["mean"].tolist(), "std": scaler["std"].tolist()}), "coef_hash": array_sha256(coef)}
    if model == "elastic_net":
        alpha = 0.001
        coef = fit_elastic_net_proxy(xtr, y_train, alpha, 0.5, 40, 0.02)
        return predict_linear(xva, coef), {"alpha": alpha, "l1_ratio": 0.5, "scaler_hash": stable_hash({"mean": scaler["mean"].tolist(), "std": scaler["std"].tolist()}), "coef_hash": array_sha256(coef)}
    raise ValueError(model)


def rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask]
    score = score[mask]
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if y.size == 0 or np.sum(y > 0.5) == 0:
        return math.nan
    order = np.argsort(-score)
    labels = (y[order] > 0.5).astype(np.float64)
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(float(labels.sum()), 1.0)
    integrator = getattr(np, "trapezoid", None)
    if integrator is None:
        integrator = np.trapz
    return float(integrator(precision, recall))


def calibration_error(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(prob)
    y = y[mask]
    prob = np.clip(prob[mask], 1e-6, 1 - 1e-6)
    if y.size == 0:
        return math.nan
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0.1, 1.0, bins)):
        bmask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if bmask.any():
            total += float(bmask.mean()) * abs(float(prob[bmask].mean()) - float(y[bmask].mean()))
    return total


def fit_logistic_binary(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray) -> np.ndarray:
    y = (np.asarray(y_train, dtype=np.float64).ravel() > 0.5).astype(np.float64)
    scaler = fit_scaler(x_train)
    xtr = np.column_stack([np.ones(len(x_train)), transform(x_train, scaler)])
    xva = np.column_stack([np.ones(len(x_val)), transform(x_val, scaler)])
    coef = np.zeros(xtr.shape[1], dtype=np.float64)
    if y.mean() > 0 and y.mean() < 1:
        coef[0] = math.log(y.mean() / (1 - y.mean()))
    for _ in range(80):
        z = np.clip(xtr @ coef, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (xtr.T @ (p - y)) / max(len(y), 1)
        grad[1:] += 0.01 * coef[1:]
        coef -= 0.1 * grad
    return 1.0 / (1.0 + np.exp(-np.clip(xva @ coef, -30, 30)))


def regression_metrics(actual: np.ndarray, pred: np.ndarray, constant_pred: np.ndarray, target_columns: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    vr = vector_rmse(actual, pred)
    cr = vector_rmse(actual, constant_pred)
    row = {
        "vector_rmse": vr,
        "vector_mae": vector_mae(actual, pred),
        "best_constant_vector_rmse": cr,
        "vector_rmse_improvement_vs_best_constant": (cr - vr) / cr if cr > 0 else math.nan,
        "pearson_correlation": pearson_corr(actual, pred),
        "spearman_correlation": spearman_corr(actual, pred),
        "calibration_slope": calibration(actual, pred)[0],
        "calibration_intercept": calibration(actual, pred)[1],
        "prediction_variance_ratio": (float(np.nanstd(pred)) ** 2) / (float(np.nanstd(actual)) ** 2) if np.nanstd(actual) > 0 else math.nan,
        "constant_prediction": bool(np.nanstd(pred) <= 1e-12),
    }
    horizon_rows = []
    wins = 0
    for idx, col in enumerate(target_columns):
        a = actual[:, idx]
        p = pred[:, idx]
        c = constant_pred[:, idx]
        rmse = float(np.sqrt(np.nanmean((a - p) ** 2)))
        crmse = float(np.sqrt(np.nanmean((a - c) ** 2)))
        win = bool(rmse < crmse)
        wins += int(win)
        horizon_rows.append({"target_column": col, "rmse": rmse, "constant_rmse": crmse, "rmse_improvement": (crmse - rmse) / crmse if crmse > 0 else math.nan, "beats_constant": win})
    row["horizon_wins"] = wins
    return row, horizon_rows


def quantile_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {f"{prefix}_{k}": math.nan for k in ["mean", "median", "std", "q10", "q25", "q75", "q90"]}
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_q10": float(np.quantile(finite, 0.10)),
        f"{prefix}_q25": float(np.quantile(finite, 0.25)),
        f"{prefix}_q75": float(np.quantile(finite, 0.75)),
        f"{prefix}_q90": float(np.quantile(finite, 0.90)),
    }


def make_targets(raw_low: np.ndarray, vol: np.ndarray, train_indices: np.ndarray) -> dict[str, dict[str, Any]]:
    downside_abs = np.abs(raw_low.astype(np.float64))
    norm = downside_abs / np.maximum(vol[:, None].astype(np.float64), EPSILON)
    train_norm = norm[train_indices]
    q1 = np.nanquantile(train_norm, 1 / 3, axis=0)
    q2 = np.nanquantile(train_norm, 2 / 3, axis=0)
    bucket = np.zeros_like(norm)
    bucket += (norm > q1).astype(np.float64)
    bucket += (norm > q2).astype(np.float64)
    out = {
        "raw_downside_bps_path": {
            "task_type": "regression",
            "values": raw_low.astype(np.float64),
            "description": "Existing signed low-path target retained as comparator; values are usually <= 0 bps.",
            "units": "bps",
        },
        "volatility_normalized_downside_path": {
            "task_type": "regression",
            "values": norm,
            "description": "abs(future_range_low_bps_h) / max(causal_recent_realized_volatility_bps, epsilon)",
            "units": "volatility_units",
        },
        "downside_risk_bucket": {
            "task_type": "ordinal_regression",
            "values": bucket,
            "description": "Ordinal 0/1/2 buckets from training-only normalized-downside tertiles by horizon.",
            "units": "ordinal_bucket",
            "training_quantiles": {"q33": q1.tolist(), "q67": q2.tolist()},
        },
    }
    for level in EXCEEDANCE_LEVELS:
        out[f"downside_exceedance_probability_gt_{str(level).replace('.', '_')}vol"] = {
            "task_type": "binary",
            "values": (norm > level).astype(np.float64),
            "description": f"Binary target: normalized downside exceeds {level} volatility units.",
            "units": "probability",
            "threshold_vol_units": level,
        }
    return out


def build_folds(n: int, timestamps: np.ndarray, source_rows: np.ndarray) -> list[dict[str, Any]]:
    folds = []
    initial_train = n // 2
    val_size = (n - initial_train) // 5
    for idx in range(5):
        train_start = 0
        train_end = initial_train + idx * val_size
        val_start = train_end
        val_end = val_start + val_size
        train_idx = np.arange(train_start, train_end)
        val_idx = np.arange(val_start, val_end)
        folds.append(
            {
                "fold_id": f"fold_{idx+1:02d}",
                "train_indices": train_idx,
                "validation_indices": val_idx,
                "train_timestamp_start": float(timestamps[train_idx[0]]),
                "train_timestamp_end": float(timestamps[train_idx[-1]]),
                "validation_timestamp_start": float(timestamps[val_idx[0]]),
                "validation_timestamp_end": float(timestamps[val_idx[-1]]),
                "purge_rows": 60,
                "embargo_rows": 480,
                "train_source_row_hash": array_sha256(source_rows[train_idx]),
                "validation_source_row_hash": array_sha256(source_rows[val_idx]),
            }
        )
    return folds


def regime_labels(x: np.ndarray, feature_columns: list[str]) -> dict[str, np.ndarray]:
    def col(name: str) -> np.ndarray:
        if name in feature_columns:
            return x[:, feature_columns.index(name)].astype(np.float64)
        return np.full(len(x), np.nan)
    vol = col("realized_volatility_bps_fw60")
    rng = col("rolling_range_bps_fw60")
    volume = col("volume_zscore_fw60") if "volume_zscore_fw60" in feature_columns else col("volume")
    trend = col("ema_slope_bps_fw60") if "ema_slope_bps_fw60" in feature_columns else col("sma_slope_bps_fw60")
    spread_proxy = col("raw_range_bps")
    return {"volatility": vol, "range": rng, "volume_liquidity": volume, "trend_strength": trend, "spread_proxy": spread_proxy}


def regime_composition(values: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray) -> dict[str, Any]:
    train_vals = values[train_idx]
    eval_vals = values[eval_idx]
    finite_train = train_vals[np.isfinite(train_vals)]
    if finite_train.size < 3:
        return {"low_fraction": math.nan, "mid_fraction": math.nan, "high_fraction": math.nan}
    q1, q2 = np.quantile(finite_train, [1 / 3, 2 / 3])
    return {
        "low_fraction": float(np.mean(eval_vals <= q1)),
        "mid_fraction": float(np.mean((eval_vals > q1) & (eval_vals <= q2))),
        "high_fraction": float(np.mean(eval_vals > q2)),
    }


def volatility_bucket_masks(values: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train_vals = values[train_idx]
    finite = train_vals[np.isfinite(train_vals)]
    if finite.size < 3:
        return {}
    q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
    buckets = {
        "low": (values[train_idx] <= q1, values[eval_idx] <= q1),
        "medium": ((values[train_idx] > q1) & (values[train_idx] <= q2), (values[eval_idx] > q1) & (values[eval_idx] <= q2)),
        "high": (values[train_idx] > q2, values[eval_idx] > q2),
    }
    return buckets


def main() -> int:
    failed_packet = env_path("RAWSEQ_DOWNSIDE_FAILED_PACKET", DEFAULT_FAILED_PACKET)
    source_run = env_path("RAWSEQ_DOWNSIDE_SOURCE_RUN", DEFAULT_SOURCE_RUN)
    output_root = env_path("RAWSEQ_DOWNSIDE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / f"rawseq_downside_target_redesign_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    npz_path = source_run / "rawseq_low_path_latest_contiguous_50000_seq60.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        x_seq = data["X"].astype(np.float32)
        y_raw = data["y"].astype(np.float64)
        splits = data["splits"].astype(str)
        timestamps = data["decision_timestamps"].astype(np.float64)
        source_rows = data["source_row_indices"].astype(np.int64)
        feature_columns = [str(x) for x in data["feature_columns"]]
        target_columns = [str(x) for x in data["target_columns"]]
    x = x_seq[:, -1, :].astype(np.float64)
    pre_holdout_mask = np.isin(splits, ["train", "validation"])
    pre_indices = np.flatnonzero(pre_holdout_mask)
    holdout_indices = np.flatnonzero(splits == "untouched_holdout")
    x_pre = x[pre_indices]
    y_pre_raw = y_raw[pre_indices]
    ts_pre = timestamps[pre_indices]
    source_pre = source_rows[pre_indices]

    vol_col = "realized_volatility_bps_fw60" if "realized_volatility_bps_fw60" in feature_columns else "rolling_volatility_bps_fw60"
    vol = np.maximum(np.abs(x[:, feature_columns.index(vol_col)].astype(np.float64)), EPSILON)
    vol_pre = vol[pre_indices]
    targets = make_targets(y_pre_raw, vol_pre, np.arange(0, 50000))

    closure_json = json.loads((failed_packet / "rawseq_low_path_ridge_research_candidate_summary.json").read_text(encoding="utf-8-sig"))
    holdout_metrics = list(csv.DictReader((failed_packet / "rawseq_low_path_ridge_holdout_metrics.csv").open(newline="", encoding="utf-8-sig")))
    contract = json.loads((failed_packet / "rawseq_low_path_ridge_candidate_contract.json").read_text(encoding="utf-8-sig"))
    model_npz = np.load(failed_packet / "rawseq_low_path_ridge_candidate_model.npz", allow_pickle=False)
    coef = model_npz["coef"].astype(np.float64)
    lineage = {
        "status": "closed_failed_untouched_holdout",
        "failed_packet": str(failed_packet),
        "source_run": str(source_run),
        "tournament_artifacts": "rawseq_target_lane_baseline_tournament_20260711T175931Z",
        "bounded_sampling_audits": "rawseq_low_path_bounded_cpu_reconciliation and residual 50k parity screens",
        "failed_plain_gru": "historical low-path GRU screens failed baseline guard",
        "failed_residual_gru": str(source_run),
        "frozen_ridge_contract": str(failed_packet / "rawseq_low_path_ridge_candidate_contract.json"),
        "consumed_holdout_decision": str(failed_packet / "rawseq_low_path_ridge_holdout_decision.json"),
        "contract_sha256": contract.get("contract_sha256"),
        "model_coef_sha256": array_sha256(coef),
        "model_intercept_sha256": array_sha256(coef[0]),
        "failed_packet_summary": closure_json,
        "holdout_result": holdout_metrics[0] if holdout_metrics else {},
        "future_holdout_used_for_selection": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    lineage["lineage_closure_sha256"] = stable_hash(lineage)
    write_json(out_dir / "rawseq_low_path_lineage_closure.json", lineage)
    (out_dir / "rawseq_low_path_lineage_closure.txt").write_text(
        "\n".join(
            [
                "Rawseq low-path lineage closure",
                f"Status: {lineage['status']}",
                f"Failed packet: {failed_packet}",
                f"Frozen ridge contract: {lineage['frozen_ridge_contract']}",
                "Final status: closed_failed_untouched_holdout",
                "Forward paper shadow must not start.",
                "Consumed holdout must not be reused as promotion holdout.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    closure_rows = []
    ridge_pred_holdout = None
    # Use holdout metrics already frozen in failed packet for attribution, never for redesign selection.
    for row in holdout_metrics:
        if row.get("target_column"):
            closure_rows.append({**row, "failure_cause": "mixed_or_unknown;target_scale_shift;calibration_drift;relationship_absent"})
    if holdout_metrics:
        closure_rows.insert(0, {**holdout_metrics[0], "failure_cause": "mixed_or_unknown;target_scale_shift;calibration_drift;regime_composition_shift;relationship_absent"})
    write_csv(out_dir / "rawseq_low_path_holdout_failure_attribution.csv", closure_rows)

    regimes = regime_labels(x, feature_columns)
    regime_rows = []
    val_all = np.flatnonzero(splits == "validation")
    hold_all = np.flatnonzero(splits == "untouched_holdout")
    for name, values in regimes.items():
        for bucket, frac in regime_composition(values, val_all, hold_all).items():
            regime_rows.append({"regime_feature": name, "comparison": "holdout_vs_validation_training_quantiles", "bucket": bucket, "holdout_fraction": frac})
    write_csv(out_dir / "rawseq_low_path_holdout_regime_comparison.csv", regime_rows)

    registry_rows = []
    for name, spec in targets.items():
        constraints = {
            "finite_fraction": float(np.isfinite(spec["values"]).mean()),
            "target_hash": array_sha256(spec["values"]),
            "min": float(np.nanmin(spec["values"])),
            "max": float(np.nanmax(spec["values"])),
        }
        registry_rows.append(
            {
                "target_name": name,
                "task_type": spec["task_type"],
                "formula": spec["description"],
                "units": spec["units"],
                "sign_convention": "raw low path <=0; normalized/bucket/exceedance are nonnegative risk magnitudes",
                "causal_dependencies": f"final-step features and causal volatility column {vol_col}",
                "horizons": ",".join(str(x) for x in HORIZONS),
                "missing_value_behavior": "nonfinite volatility clipped to epsilon; model features train-mean filled by scaler",
                "normalization_epsilon": EPSILON,
                "target_hash": constraints["target_hash"],
                "target_constraint_checks": json.dumps(constraints, sort_keys=True),
            }
        )
    write_csv(out_dir / "rawseq_downside_target_redesign_registry.csv", registry_rows)

    contract_payload = {
        "created_at": now_stamp(),
        "source_npz": str(npz_path),
        "source_npz_sha256": file_sha256(npz_path),
        "failed_lineage_packet": str(failed_packet),
        "consumed_holdout_excluded_from_selection": True,
        "development_rows": int(len(pre_indices)),
        "reserved_later_period_required_for_future_one_time_test": True,
        "target_registry_rows": registry_rows,
        "fold_count": 5,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    contract_payload["contract_sha256"] = stable_hash(contract_payload)
    write_json(out_dir / "rawseq_downside_target_redesign_contract.json", contract_payload)

    folds = build_folds(len(pre_indices), ts_pre, source_pre)
    fold_manifest_rows = []
    metrics_rows: list[dict[str, Any]] = []
    regime_metric_rows: list[dict[str, Any]] = []
    for fold in folds:
        tr = fold["train_indices"]
        va = fold["validation_indices"]
        fold_manifest_rows.append(
            {
                **{k: v for k, v in fold.items() if not k.endswith("indices")},
                "feature_hash": array_sha256(x_pre[tr]),
                "raw_target_hash": array_sha256(y_pre_raw[tr]),
                "effective_non_overlapping_rows_h60": int(len(va) // 60),
                "effective_non_overlapping_rows_h120": int(len(va) // 120),
                "effective_non_overlapping_rows_h240": int(len(va) // 240),
                "effective_non_overlapping_rows_h480": int(len(va) // 480),
            }
        )
        for name, values in targets.items():
            y_all = values["values"]
            task = values["task_type"]
            y_train = y_all[tr]
            y_val = y_all[va]
            train_mean = np.nanmean(y_train, axis=0)
            train_median = np.nanmedian(y_train, axis=0)
            mean_pred = np.tile(train_mean, (len(va), 1))
            median_pred = np.tile(train_median, (len(va), 1))
            mean_rmse = vector_rmse(y_val, mean_pred)
            median_rmse = vector_rmse(y_val, median_pred)
            constant_pred = mean_pred if mean_rmse <= median_rmse else median_pred
            best_constant = "training_mean_baseline" if mean_rmse <= median_rmse else "training_median_baseline"
            if task in {"regression", "ordinal_regression"}:
                model_preds = {
                    "training_mean_baseline": mean_pred,
                    "training_median_baseline": median_pred,
                    "quantile_regression_p50_constant": median_pred,
                }
                for model in ["ridge", "elastic_net"]:
                    pred, meta = fit_linear_family(x_pre[tr], y_train, x_pre[va], model)
                    model_preds[model] = pred
                for model, pred in model_preds.items():
                    row, horizon_rows = regression_metrics(y_val, pred, constant_pred, target_columns)
                    rowspec = {
                        "fold_id": fold["fold_id"],
                        "target_name": name,
                        "task_type": task,
                        "model": model,
                        "best_constant_model": best_constant,
                        "validation_rows": len(va),
                        **row,
                    }
                    metrics_rows.append(rowspec)
                    for regime_name, values_regime in regimes.items():
                        comp = regime_composition(values_regime[pre_indices], tr, va)
                        regime_metric_rows.append({"fold_id": fold["fold_id"], "target_name": name, "model": model, "regime_feature": regime_name, **comp})
            elif task == "binary":
                for idx, col in enumerate(target_columns):
                    ytr = y_train[:, idx]
                    yva = y_val[:, idx]
                    prevalence = float(np.nanmean(ytr))
                    const_prob = np.full(len(va), min(max(prevalence, 1e-6), 1 - 1e-6))
                    logit_prob = fit_logistic_binary(x_pre[tr], ytr, x_pre[va])
                    for model, prob in [("constant_prevalence_baseline", const_prob), ("logistic_regression", logit_prob), ("calibrated_logistic_regression", logit_prob)]:
                        prob = np.clip(prob, 1e-6, 1 - 1e-6)
                        brier = float(np.mean((prob - yva) ** 2))
                        const_brier = float(np.mean((const_prob - yva) ** 2))
                        metrics_rows.append(
                            {
                                "fold_id": fold["fold_id"],
                                "target_name": name,
                                "task_type": task,
                                "target_column": col,
                                "model": model,
                                "validation_rows": len(va),
                                "prevalence": prevalence,
                                "brier_score": brier,
                                "constant_brier_score": const_brier,
                                "brier_improvement_vs_constant": (const_brier - brier) / const_brier if const_brier > 0 else math.nan,
                                "log_loss": float(-np.mean(yva * np.log(prob) + (1 - yva) * np.log(1 - prob))),
                                "roc_auc": rank_auc(yva, prob),
                                "pr_auc": pr_auc(yva, prob),
                                "calibration_error": calibration_error(yva, prob),
                            }
                        )
                    vol_values = regimes["volatility"][pre_indices]
                    for bucket_name, (train_bucket_mask, val_bucket_mask) in volatility_bucket_masks(vol_values, tr, va).items():
                        if int(train_bucket_mask.sum()) < 100 or int(val_bucket_mask.sum()) < 50:
                            continue
                        ytr_bucket = ytr[train_bucket_mask]
                        if np.nanmin(ytr_bucket) == np.nanmax(ytr_bucket):
                            continue
                        prob_bucket = fit_logistic_binary(x_pre[tr][train_bucket_mask], ytr_bucket, x_pre[va][val_bucket_mask])
                        yva_bucket = yva[val_bucket_mask]
                        prevalence_bucket = float(np.nanmean(ytr_bucket))
                        const_bucket = np.full(len(yva_bucket), min(max(prevalence_bucket, 1e-6), 1 - 1e-6))
                        brier = float(np.mean((prob_bucket - yva_bucket) ** 2))
                        const_brier = float(np.mean((const_bucket - yva_bucket) ** 2))
                        metrics_rows.append(
                            {
                                "fold_id": fold["fold_id"],
                                "target_name": name,
                                "task_type": task,
                                "target_column": col,
                                "model": f"logistic_regression_volatility_{bucket_name}_model",
                                "regime_conditioning": f"volatility_{bucket_name}",
                                "validation_rows": int(val_bucket_mask.sum()),
                                "prevalence": prevalence_bucket,
                                "brier_score": brier,
                                "constant_brier_score": const_brier,
                                "brier_improvement_vs_constant": (const_brier - brier) / const_brier if const_brier > 0 else math.nan,
                                "log_loss": float(-np.mean(yva_bucket * np.log(np.clip(prob_bucket, 1e-6, 1 - 1e-6)) + (1 - yva_bucket) * np.log(1 - np.clip(prob_bucket, 1e-6, 1 - 1e-6)))),
                                "roc_auc": rank_auc(yva_bucket, prob_bucket),
                                "pr_auc": pr_auc(yva_bucket, prob_bucket),
                                "calibration_error": calibration_error(yva_bucket, prob_bucket),
                                "selection_role": "regime_conditioning_diagnostic_not_main_ranking",
                            }
                        )
                    for bucket_name, (_, val_bucket_mask) in volatility_bucket_masks(vol_values, tr, va).items():
                        if int(val_bucket_mask.sum()) < 50:
                            continue
                        yva_bucket = yva[val_bucket_mask]
                        prob_bucket = logit_prob[val_bucket_mask]
                        regime_metric_rows.append(
                            {
                                "fold_id": fold["fold_id"],
                                "target_name": name,
                                "model": "global_logistic_regression_with_registered_regime_features",
                                "regime_feature": "volatility",
                                "regime_bucket": bucket_name,
                                "target_column": col,
                                "validation_rows": int(val_bucket_mask.sum()),
                                "brier_score": float(np.mean((prob_bucket - yva_bucket) ** 2)),
                                "roc_auc": rank_auc(yva_bucket, prob_bucket),
                                "pr_auc": pr_auc(yva_bucket, prob_bucket),
                                "calibration_error": calibration_error(yva_bucket, prob_bucket),
                            }
                        )
    write_csv(out_dir / "rawseq_downside_rolling_fold_manifest.csv", fold_manifest_rows)
    write_csv(out_dir / "rawseq_downside_cpu_fold_metrics.csv", metrics_rows)
    write_csv(out_dir / "rawseq_downside_regime_metrics.csv", regime_metric_rows)

    ranking_rows = []
    df = pd.DataFrame(metrics_rows)
    for name, group in df.groupby("target_name"):
        task = str(group["task_type"].iloc[0])
        if task in {"regression", "ordinal_regression"}:
            learned = group[group["model"].isin(["ridge", "elastic_net"])]
            if learned.empty:
                continue
            best_by_fold = learned.sort_values("vector_rmse", ascending=True).groupby("fold_id").head(1)
            win_count = int((pd.to_numeric(best_by_fold["vector_rmse_improvement_vs_best_constant"], errors="coerce") > 0).sum())
            median_improvement = float(pd.to_numeric(best_by_fold["vector_rmse_improvement_vs_best_constant"], errors="coerce").median())
            horizon_ok = int((pd.to_numeric(best_by_fold["horizon_wins"], errors="coerce") >= 3).sum())
            severe_collapse = bool((pd.to_numeric(best_by_fold["prediction_variance_ratio"], errors="coerce") < 1e-4).any())
            status = "advance_candidate" if win_count >= 4 and median_improvement > 0.02 and horizon_ok >= 4 and not severe_collapse else "reject_or_research_only"
            ranking_rows.append({"target_name": name, "task_type": task, "best_model_family": ",".join(sorted(set(best_by_fold["model"]))), "fold_win_count": win_count, "median_improvement_vs_best_constant": median_improvement, "horizon_positive_fold_count": horizon_ok, "severe_collapse": severe_collapse, "target_status": status})
        else:
            learned = group[group["model"].isin(["logistic_regression", "calibrated_logistic_regression"])]
            best_by_fold_target = learned.sort_values("brier_score", ascending=True).groupby(["fold_id", "target_column"]).head(1)
            win_count = int((pd.to_numeric(best_by_fold_target["brier_improvement_vs_constant"], errors="coerce") > 0).groupby(best_by_fold_target["fold_id"]).any().sum())
            median_improvement = float(pd.to_numeric(best_by_fold_target["brier_improvement_vs_constant"], errors="coerce").median())
            median_pr = float(pd.to_numeric(best_by_fold_target["pr_auc"], errors="coerce").median())
            median_cal = float(pd.to_numeric(best_by_fold_target["calibration_error"], errors="coerce").median())
            status = "advance_candidate" if win_count >= 4 and median_improvement > 0 and median_pr > 0 and median_cal < 0.15 else "reject_or_research_only"
            ranking_rows.append({"target_name": name, "task_type": task, "best_model_family": "logistic_regression", "fold_win_count": win_count, "median_improvement_vs_best_constant": median_improvement, "median_pr_auc": median_pr, "median_calibration_error": median_cal, "target_status": status})
    write_csv(out_dir / "rawseq_downside_cpu_target_ranking.csv", ranking_rows)

    advancing = [row for row in ranking_rows if row["target_status"] == "advance_candidate" and row["target_name"] != "raw_downside_bps_path"]
    recommendation = {
        "recommendation": "build_new_downside_risk_cpu_candidate" if advancing else "close_current_downside_target_research",
        "advancing_targets": advancing,
        "target_ranking_path": str(out_dir / "rawseq_downside_cpu_target_ranking.csv"),
        "future_holdout_used_for_selection": False,
        "consumed_holdout_reused_as_promotion_holdout": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    recommendation["recommendation_sha256"] = stable_hash(recommendation)
    write_json(out_dir / "rawseq_downside_target_recommendation.json", recommendation)

    lines = [
        "Rawseq downside target redesign report",
        f"Output: {out_dir}",
        f"Failed-lineage closure: {out_dir / 'rawseq_low_path_lineage_closure.json'}",
        "Principal holdout failure attribution: mixed_or_unknown with target_scale_shift, calibration_drift, regime_composition_shift, relationship_absent",
        "New targets: raw_downside_bps_path; volatility_normalized_downside_path; downside_exceedance_probability; downside_risk_bucket",
        "Folds: five expanding-origin train windows followed by non-overlapping validation windows before consumed holdout",
        "Future holdout used for selection: false",
        "CPU target ranking:",
    ]
    for row in ranking_rows:
        lines.append(f"- {row['target_name']}: status={row['target_status']} wins={row['fold_win_count']} median_improvement={row['median_improvement_vs_best_constant']}")
    lines.append(f"Final recommendation: {recommendation['recommendation']}")
    (out_dir / "rawseq_downside_target_redesign_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
