#!/usr/bin/env python3
"""Explain validation-baseline failures for completed rawseq GRU screens.

Report only. No training, no seed expansion, no architecture comparison, no
orders, no promotion, and no champion mutation.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_gru_locked_bundle_screen"
DEFAULT_SURVIVOR_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_gru_contract_survivors"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_validation_failure_attribution"
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
BORING_TARGET_BASELINES = {
    "zero_return_baseline",
    "training_mean_return_baseline",
    "rolling_mean_momentum_baseline",
    "mean_reversion_baseline",
}
PAIR_BUNDLES = [
    ("minimal_core", "balanced_research"),
    ("minimal_core", "full_registered"),
    ("balanced_research", "full_registered"),
]
PAIR_SEQS = [(60, 120), (60, 240), (120, 240)]
RECOMMENDATIONS = {
    "redesign_targets",
    "improve_feature_bundle",
    "run_training_capacity_test",
    "run_optimizer_scaling_test",
    "run_regime_conditioned_analysis",
    "stop_current_lane",
}


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def latest_dir(root: Path, pattern: str) -> Path | None:
    paths = [path for path in root.glob(pattern) if path.is_dir()] if root.exists() else []
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


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


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2))) if mask.any() else math.nan


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    return float(np.mean(np.abs(actual[mask] - pred[mask]))) if mask.any() else math.nan


def corr(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if int(mask.sum()) < 3:
        return math.nan
    a = actual[mask]
    p = pred[mask]
    if float(np.std(a)) <= 1e-12 or float(np.std(p)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, p)[0, 1])


def directional_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    return float(np.mean((actual[mask] > 0.0) == (pred[mask] > 0.0))) if mask.any() else math.nan


def quantile(values: np.ndarray, q: float) -> float:
    arr = finite_array(values)
    return float(np.quantile(arr, q)) if len(arr) else math.nan


def infer_dataset_dir(benchmark_dir: Path) -> Path:
    contract_path = benchmark_dir / "torch_sequence_benchmark_contract.json"
    if not contract_path.exists():
        raise SystemExit(f"Missing benchmark contract: {contract_path}")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = Path(str(payload.get("sequence_dataset_manifest", "")))
    if not manifest.exists():
        raise SystemExit(f"Missing sequence dataset manifest from benchmark contract: {manifest}")
    return manifest.parent


def infer_ablation_dir(dataset_dir: Path) -> Path:
    contract_path = dataset_dir / "locked_bundle_sequence_dataset_contract.json"
    if not contract_path.exists():
        raise SystemExit(f"Missing dataset contract: {contract_path}")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    ablation_dir = Path(str(payload.get("ablation_dir", "")))
    if not ablation_dir.exists():
        raise SystemExit(f"Missing ablation dir from dataset contract: {ablation_dir}")
    return ablation_dir


def validation_baseline_reference(ablation_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(ablation_dir / "ablation_metrics.csv")
    frame = metrics[metrics["base_model"].astype(str).isin(BASELINE_BASE_MODELS)].copy()
    rows: list[dict[str, Any]] = []
    for horizon, group in frame.groupby("horizon_buckets", dropna=False):
        group = group.copy()
        group["_rmse"] = pd.to_numeric(group["validation_rmse"], errors="coerce")
        finite = group[np.isfinite(group["_rmse"])].copy()
        if finite.empty:
            continue
        best = finite.sort_values(["_rmse", "model"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "horizon_buckets": int(float(horizon)),
                "best_validation_baseline": best.get("model", ""),
                "best_validation_baseline_base_model": best.get("base_model", ""),
                "best_validation_baseline_rmse": safe_float(best.get("validation_rmse")),
            }
        )
    return pd.DataFrame(rows)


def target_baselines_by_horizon(ablation_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(ablation_dir / "ablation_metrics.csv")
    frame = metrics[metrics["base_model"].astype(str).isin(BORING_TARGET_BASELINES)].copy()
    rows: list[dict[str, Any]] = []
    for horizon, group in frame.groupby("horizon_buckets", dropna=False):
        row: dict[str, Any] = {"horizon_buckets": int(float(horizon))}
        for base_name, out_name in [
            ("zero_return_baseline", "zero_baseline_rmse"),
            ("training_mean_return_baseline", "training_mean_baseline_rmse"),
            ("rolling_mean_momentum_baseline", "rolling_mean_baseline_rmse"),
            ("mean_reversion_baseline", "mean_reversion_baseline_rmse"),
        ]:
            base = group[group["base_model"].astype(str).eq(base_name)].copy()
            base["_rmse"] = pd.to_numeric(base["validation_rmse"], errors="coerce")
            finite = base[np.isfinite(base["_rmse"])]
            row[out_name] = safe_float(finite["_rmse"].min()) if not finite.empty else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_prediction_npz(path_value: Any) -> dict[str, Any] | None:
    path_text = str(path_value)
    if not path_text or path_text.lower() == "nan":
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {
            "prediction": np.asarray(data["prediction"], dtype=np.float64),
            "actual": np.asarray(data["actual"], dtype=np.float64),
            "splits": data["splits"].astype(str),
            "horizon_buckets": data["horizon_buckets"].astype(int).tolist(),
        }


def load_dataset_npz(path_value: Any) -> dict[str, Any]:
    path = Path(str(path_value))
    with np.load(path, allow_pickle=False) as data:
        return {
            "actual": np.asarray(data["y"], dtype=np.float64),
            "splits": data["splits"].astype(str),
            "timestamps": np.asarray(data["decision_timestamps"], dtype=np.float64),
            "horizon_buckets": data["horizon_buckets"].astype(int).tolist(),
        }


def ordered_split_arrays(dataset: dict[str, Any], split: str, horizon_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = dataset["splits"].astype(str) == split
    timestamps = np.asarray(dataset["timestamps"][mask], dtype=np.float64)
    actual = np.asarray(dataset["actual"][mask, horizon_idx], dtype=np.float64)
    order = np.argsort(timestamps)
    return actual[order], timestamps[order], order


def prediction_arrays(
    prediction_cache: dict[tuple[Any, ...], dict[str, Any] | None],
    row: pd.Series,
    split: str,
    horizon_idx: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    key = (
        row.get("dataset_index"),
        row.get("feature_group"),
        row.get("seq_len"),
        row.get("model_kind"),
        row.get("seed"),
    )
    payload = prediction_cache.get(key)
    if payload is None:
        return None, None
    mask = payload["splits"].astype(str) == split
    actual = np.asarray(payload["actual"][mask, horizon_idx], dtype=np.float64)
    pred = np.asarray(payload["prediction"][mask, horizon_idx], dtype=np.float64)
    return actual, pred


def target_stats(values: np.ndarray) -> dict[str, Any]:
    arr = finite_array(values)
    return {
        "target_mean": float(np.mean(arr)) if len(arr) else math.nan,
        "target_std": float(np.std(arr)) if len(arr) else math.nan,
        "target_positive_fraction": float(np.mean(arr > 0.0)) if len(arr) else math.nan,
        "target_abs_p50": quantile(np.abs(arr), 0.50),
        "target_abs_p90": quantile(np.abs(arr), 0.90),
        "target_abs_p99": quantile(np.abs(arr), 0.99),
    }


def prediction_stats(pred: np.ndarray | None, actual: np.ndarray) -> dict[str, Any]:
    if pred is None:
        return {
            "prediction_mean": math.nan,
            "prediction_std": math.nan,
            "prediction_to_target_variance_ratio": math.nan,
            "prediction_artifact_status": "missing_prediction_artifact",
        }
    pred_arr = finite_array(pred)
    target_arr = finite_array(actual)
    pred_std = float(np.std(pred_arr)) if len(pred_arr) else math.nan
    target_std = float(np.std(target_arr)) if len(target_arr) else math.nan
    ratio = (pred_std**2) / (target_std**2) if math.isfinite(pred_std) and math.isfinite(target_std) and target_std > 1e-12 else math.nan
    return {
        "prediction_mean": float(np.mean(pred_arr)) if len(pred_arr) else math.nan,
        "prediction_std": pred_std,
        "prediction_to_target_variance_ratio": ratio,
        "prediction_artifact_status": "ok",
    }


def effective_non_overlap_rows(validation_rows: int, horizon: int) -> int:
    return int(math.ceil(validation_rows / max(1, int(horizon)))) if validation_rows > 0 else 0


def classify_statuses(row: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    pred_ratio = safe_float(row.get("prediction_to_target_variance_ratio"))
    pred_std = safe_float(row.get("prediction_std"))
    train_rmse = safe_float(row.get("train_rmse"))
    val_rmse = safe_float(row.get("validation_rmse"))
    baseline_rmse = safe_float(row.get("best_validation_baseline_rmse"))
    train_corr = safe_float(row.get("train_correlation"))
    val_corr = safe_float(row.get("validation_correlation"))
    target_shift = safe_float(row.get("train_validation_target_shift_score"))
    degradation = safe_float(row.get("train_to_validation_degradation"))
    eff_rows = int(safe_float(row.get("effective_non_overlapping_validation_rows"), 0))
    improvement = safe_float(row.get("baseline_improvement_fraction"))
    baseline_name = str(row.get("best_validation_baseline", ""))
    prediction_missing = str(row.get("prediction_artifact_status", "")) != "ok"

    prediction_collapse = (math.isfinite(pred_ratio) and pred_ratio < 0.05) or (math.isfinite(pred_std) and pred_std < 1e-6)
    if prediction_missing:
        prediction_collapse_status = "unknown_prediction_artifact_missing"
    elif prediction_collapse:
        prediction_collapse_status = "prediction_collapse"
        reasons.append("prediction_variance_collapsed")
    else:
        prediction_collapse_status = "not_collapsed"

    underfit = math.isfinite(train_rmse) and math.isfinite(baseline_rmse) and train_rmse >= baseline_rmse * 0.98 and (not math.isfinite(train_corr) or abs(train_corr) < 0.10)
    underfit_status = "model_underfit" if underfit else "not_underfit"
    if underfit:
        reasons.append("train_fit_does_not_beat_baseline_proxy")

    overfit = math.isfinite(degradation) and degradation > 0.35 and math.isfinite(train_corr) and math.isfinite(val_corr) and train_corr > val_corr + 0.20
    overfit_status = "model_overfit" if overfit else "not_overfit"
    if overfit:
        reasons.append("train_validation_degradation_large")

    shift = math.isfinite(target_shift) and target_shift > 0.75
    shift_status = "train_validation_distribution_shift" if shift else "no_large_distribution_shift"
    if shift:
        reasons.append("target_distribution_shift")

    if eff_rows < 30:
        reasons.append("insufficient_effective_validation_rows")
    if math.isfinite(improvement) and improvement <= 0:
        reasons.append("fails_validation_baseline")
    if baseline_name and any(name in baseline_name for name in ["zero_return_baseline", "training_mean_return_baseline"]) and math.isfinite(improvement) and -0.05 <= improvement <= 0.0:
        reasons.append("baseline_too_strong_small_margin")
    if prediction_missing:
        reasons.append("prediction_artifact_missing")

    if "insufficient_effective_validation_rows" in reasons:
        failure_class = "insufficient_effective_rows"
    elif prediction_collapse:
        failure_class = "prediction_collapse"
    elif underfit:
        failure_class = "model_underfit"
    elif overfit:
        failure_class = "model_overfit"
    elif shift:
        failure_class = "train_validation_distribution_shift"
    elif "baseline_too_strong_small_margin" in reasons:
        failure_class = "baseline_too_strong"
    elif "fails_validation_baseline" in reasons and abs(safe_float(row.get("validation_correlation"), 0.0)) < 0.05:
        failure_class = "target_low_signal"
    elif len(reasons) > 1:
        failure_class = "mixed_failure"
    else:
        failure_class = "mixed_failure"
    return {
        "prediction_collapse_status": prediction_collapse_status,
        "underfit_status": underfit_status,
        "overfit_status": overfit_status,
        "train_validation_distribution_shift_status": shift_status,
        "failure_class": failure_class,
        "failure_reasons": ";".join(reasons),
    }


def build_attribution(
    summary: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    selected_policy: pd.DataFrame,
    validation_baseline: pd.DataFrame,
    dataset_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[Any, ...], dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    train = horizon_metrics[horizon_metrics["split"].astype(str).eq("train")].copy()
    validation = horizon_metrics[horizon_metrics["split"].astype(str).eq("validation")].copy()
    keys = ["dataset_index", "feature_group", "seq_len", "model_kind", "seed", "horizon_buckets"]
    train = train.rename(
        columns={
            "rmse": "train_rmse",
            "mae": "train_mae",
            "correlation": "train_correlation",
            "directional_accuracy": "train_directional_accuracy",
            "rows": "train_rows",
        }
    )
    validation = validation.rename(
        columns={
            "rmse": "validation_rmse",
            "mae": "validation_mae",
            "correlation": "validation_correlation",
            "directional_accuracy": "validation_directional_accuracy",
            "rows": "validation_rows",
        }
    )
    detail = validation.merge(
        train.reindex(columns=keys + ["train_rmse", "train_mae", "train_correlation", "train_directional_accuracy", "train_rows"]),
        on=keys,
        how="left",
    )
    detail = detail.merge(validation_baseline, on="horizon_buckets", how="left")
    detail = detail.merge(
        selected_policy.reindex(columns=keys + ["validation_position_trade_count", "validation_position_cum_net_bps", "selected_threshold_bps"]),
        on=keys,
        how="left",
    )
    detail = detail.merge(
        summary.reindex(columns=["dataset_index", "feature_group", "seq_len", "model_kind", "seed", "prediction_path"]),
        on=["dataset_index", "feature_group", "seq_len", "model_kind", "seed"],
        how="left",
    )
    dataset_map = dataset_manifest.set_index("dataset_index")["path_npz"].to_dict()
    prediction_cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}
    dataset_cache: dict[tuple[int, int], dict[str, Any]] = {}
    rows = []
    for _, row in detail.iterrows():
        dataset_index = int(float(row["dataset_index"]))
        horizon = int(float(row["horizon_buckets"]))
        dataset = dataset_cache.setdefault((dataset_index, 0), load_dataset_npz(dataset_map[dataset_index]))
        horizon_idx = list(dataset["horizon_buckets"]).index(horizon)
        val_target, _, _ = ordered_split_arrays(dataset, "validation", horizon_idx)
        train_target, _, _ = ordered_split_arrays(dataset, "train", horizon_idx)
        pred_key = (
            row.get("dataset_index"),
            row.get("feature_group"),
            row.get("seq_len"),
            row.get("model_kind"),
            row.get("seed"),
        )
        if pred_key not in prediction_cache:
            prediction_cache[pred_key] = load_prediction_npz(row.get("prediction_path"))
        _, val_pred = prediction_arrays(prediction_cache, row, "validation", horizon_idx)
        tstats = target_stats(val_target)
        pstats = prediction_stats(val_pred, val_target)
        train_stats = target_stats(train_target)
        train_to_val_shift = abs(tstats["target_mean"] - train_stats["target_mean"]) / max(train_stats["target_std"], 1e-9) if math.isfinite(train_stats["target_std"]) else math.nan
        train_to_validation_degradation = (
            (safe_float(row.get("validation_rmse")) - safe_float(row.get("train_rmse"))) / safe_float(row.get("train_rmse"))
            if safe_float(row.get("train_rmse")) > 1e-12
            else math.nan
        )
        baseline_improvement = (
            (safe_float(row.get("best_validation_baseline_rmse")) - safe_float(row.get("validation_rmse"))) / safe_float(row.get("best_validation_baseline_rmse"))
            if safe_float(row.get("best_validation_baseline_rmse")) > 1e-12
            else math.nan
        )
        out = {
            "feature_bundle": row.get("feature_group"),
            "seq_len": int(float(row.get("seq_len"))),
            "seed": int(float(row.get("seed"))),
            "horizon_buckets": horizon,
            "train_rmse": safe_float(row.get("train_rmse")),
            "train_mae": safe_float(row.get("train_mae")),
            "validation_rmse": safe_float(row.get("validation_rmse")),
            "validation_mae": safe_float(row.get("validation_mae")),
            "train_correlation": safe_float(row.get("train_correlation")),
            "validation_correlation": safe_float(row.get("validation_correlation")),
            "validation_directional_accuracy": safe_float(row.get("validation_directional_accuracy")),
            "best_validation_baseline": row.get("best_validation_baseline", ""),
            "best_validation_baseline_rmse": safe_float(row.get("best_validation_baseline_rmse")),
            "baseline_improvement_fraction": baseline_improvement,
            "train_to_validation_degradation": train_to_validation_degradation,
            **pstats,
            **tstats,
            "validation_rows": int(safe_float(row.get("validation_rows"), len(val_target))),
            "effective_non_overlapping_validation_rows": effective_non_overlap_rows(int(safe_float(row.get("validation_rows"), len(val_target))), horizon),
            "target_sign_balance": min(tstats["target_positive_fraction"], 1.0 - tstats["target_positive_fraction"]) if math.isfinite(tstats["target_positive_fraction"]) else math.nan,
            "train_target_mean": train_stats["target_mean"],
            "train_target_std": train_stats["target_std"],
            "train_validation_target_shift_score": train_to_val_shift,
            "validation_policy_trade_count": int(safe_float(row.get("validation_position_trade_count"), 0)),
            "selected_threshold_bps": safe_float(row.get("selected_threshold_bps")),
        }
        out.update(classify_statuses(out))
        rows.append(out)
    return pd.DataFrame(rows), prediction_cache, dataset_cache


def split_blocks(n: int, blocks: int = 4) -> list[tuple[int, int]]:
    bounds = np.linspace(0, n, blocks + 1, dtype=int)
    return [(int(bounds[idx]), int(bounds[idx + 1])) for idx in range(blocks)]


def build_block_attribution(
    attribution: pd.DataFrame,
    dataset_manifest: pd.DataFrame,
    prediction_cache: dict[tuple[Any, ...], dict[str, Any] | None],
) -> pd.DataFrame:
    dataset_map = dataset_manifest.set_index("dataset_index")["path_npz"].to_dict()
    dataset_cache = {idx: load_dataset_npz(path) for idx, path in dataset_map.items()}
    rows = []
    for _, row in attribution.iterrows():
        dataset_index = int(dataset_manifest[
            (dataset_manifest["feature_group"].astype(str) == str(row["feature_bundle"]))
            & (dataset_manifest["seq_len"].astype(int) == int(row["seq_len"]))
        ]["dataset_index"].iloc[0])
        dataset = dataset_cache[dataset_index]
        horizon = int(row["horizon_buckets"])
        horizon_idx = list(dataset["horizon_buckets"]).index(horizon)
        val_target, _, _ = ordered_split_arrays(dataset, "validation", horizon_idx)
        train_target, _, _ = ordered_split_arrays(dataset, "train", horizon_idx)
        pred_key = (dataset_index, row["feature_bundle"], row["seq_len"], "gru", row["seed"])
        payload = prediction_cache.get(pred_key)
        val_pred = None
        if payload is not None:
            mask = payload["splits"].astype(str) == "validation"
            val_pred = np.asarray(payload["prediction"][mask, horizon_idx], dtype=np.float64)
        train_mean = float(np.nanmean(train_target)) if len(train_target) else 0.0
        train_std = float(np.nanstd(train_target)) if len(train_target) else math.nan
        for block_index, (start, end) in enumerate(split_blocks(len(val_target), 4)):
            actual_block = val_target[start:end]
            baseline_pred = np.full(len(actual_block), train_mean, dtype=np.float64)
            pred_block = val_pred[start:end] if val_pred is not None else None
            block_target = target_stats(actual_block)
            block_pred = prediction_stats(pred_block, actual_block)
            base_rmse = rmse(actual_block, baseline_pred)
            gru_rmse = rmse(actual_block, pred_block) if pred_block is not None else math.nan
            rows.append(
                {
                    "feature_bundle": row["feature_bundle"],
                    "seq_len": int(row["seq_len"]),
                    "seed": int(row["seed"]),
                    "horizon_buckets": horizon,
                    "validation_block_index": block_index,
                    "block_start": start,
                    "block_end": end,
                    "block_rows": int(end - start),
                    "gru_rmse": gru_rmse,
                    "baseline_rmse": base_rmse,
                    "baseline_improvement": (base_rmse - gru_rmse) / base_rmse if math.isfinite(base_rmse) and base_rmse > 0 and math.isfinite(gru_rmse) else math.nan,
                    "target_mean": block_target["target_mean"],
                    "target_std": block_target["target_std"],
                    "prediction_mean": block_pred["prediction_mean"],
                    "prediction_std": block_pred["prediction_std"],
                    "correlation": corr(actual_block, pred_block) if pred_block is not None else math.nan,
                    "directional_accuracy": directional_accuracy(actual_block, pred_block) if pred_block is not None else math.nan,
                    "distribution_shift_from_training": abs(block_target["target_mean"] - train_mean) / max(train_std, 1e-9) if math.isfinite(train_std) else math.nan,
                    "prediction_artifact_status": block_pred["prediction_artifact_status"],
                }
            )
    return pd.DataFrame(rows)


def build_target_predictability(dataset_manifest: pd.DataFrame, baseline_table: pd.DataFrame) -> pd.DataFrame:
    first = dataset_manifest.sort_values(["seq_len", "feature_group"]).iloc[0]
    dataset = load_dataset_npz(first["path_npz"])
    rows = []
    for horizon_idx, horizon in enumerate(dataset["horizon_buckets"]):
        val_target, _, _ = ordered_split_arrays(dataset, "validation", horizon_idx)
        train_target, _, _ = ordered_split_arrays(dataset, "train", horizon_idx)
        train_mean = float(np.nanmean(train_target)) if len(train_target) else 0.0
        lag_autocorr = corr(val_target[:-1], val_target[1:]) if len(val_target) > 3 else math.nan
        block_means = []
        for start, end in split_blocks(len(val_target), 4):
            block_means.append(float(np.nanmean(val_target[start:end])) if end > start else math.nan)
        block_shift = float(np.nanmax(block_means) - np.nanmin(block_means)) if any(math.isfinite(x) for x in block_means) else math.nan
        base = baseline_table[baseline_table["horizon_buckets"].astype(int).eq(int(horizon))]
        row = base.iloc[0].to_dict() if not base.empty else {}
        best_rmse = min([safe_float(row.get(name)) for name in ["zero_baseline_rmse", "training_mean_baseline_rmse", "rolling_mean_baseline_rmse", "mean_reversion_baseline_rmse"] if math.isfinite(safe_float(row.get(name)))], default=math.nan)
        target_variance = float(np.nanvar(val_target)) if len(val_target) else math.nan
        explained_proxy = 1.0 - (best_rmse**2 / target_variance) if math.isfinite(best_rmse) and math.isfinite(target_variance) and target_variance > 1e-12 else math.nan
        rows.append(
            {
                "horizon_buckets": int(horizon),
                "zero_baseline_rmse": safe_float(row.get("zero_baseline_rmse")),
                "training_mean_baseline_rmse": safe_float(row.get("training_mean_baseline_rmse")),
                "rolling_mean_baseline_rmse": safe_float(row.get("rolling_mean_baseline_rmse")),
                "momentum_baseline_rmse": safe_float(row.get("rolling_mean_baseline_rmse")),
                "mean_reversion_baseline_rmse": safe_float(row.get("mean_reversion_baseline_rmse")),
                "lag_autocorrelation": lag_autocorr,
                "target_variance": target_variance,
                "validation_target_mean": float(np.nanmean(val_target)) if len(val_target) else math.nan,
                "validation_target_std": float(np.nanstd(val_target)) if len(val_target) else math.nan,
                "training_target_mean": train_mean,
                "block_to_block_target_shift": block_shift,
                "best_boring_baseline_explained_variance_proxy": explained_proxy,
            }
        )
    return pd.DataFrame(rows)


def matched_bundle_comparisons(attribution: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seq_len, seed, horizon in attribution[["seq_len", "seed", "horizon_buckets"]].drop_duplicates().itertuples(index=False):
        group = attribution[
            (attribution["seq_len"].astype(int) == int(seq_len))
            & (attribution["seed"].astype(int) == int(seed))
            & (attribution["horizon_buckets"].astype(int) == int(horizon))
        ]
        by_bundle = group.set_index("feature_bundle")
        for left, right in PAIR_BUNDLES:
            if left in by_bundle.index and right in by_bundle.index:
                l = by_bundle.loc[left]
                r = by_bundle.loc[right]
                rows.append(
                    {
                        "left_bundle": left,
                        "right_bundle": right,
                        "seq_len": int(seq_len),
                        "seed": int(seed),
                        "horizon_buckets": int(horizon),
                        "left_validation_improvement": safe_float(l["baseline_improvement_fraction"]),
                        "right_validation_improvement": safe_float(r["baseline_improvement_fraction"]),
                        "improvement_delta_right_minus_left": safe_float(r["baseline_improvement_fraction"]) - safe_float(l["baseline_improvement_fraction"]),
                        "matched_on": "seq_len,seed,horizon_buckets",
                    }
                )
    return pd.DataFrame(rows)


def matched_sequence_comparisons(attribution: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bundle, seed, horizon in attribution[["feature_bundle", "seed", "horizon_buckets"]].drop_duplicates().itertuples(index=False):
        group = attribution[
            (attribution["feature_bundle"].astype(str) == str(bundle))
            & (attribution["seed"].astype(int) == int(seed))
            & (attribution["horizon_buckets"].astype(int) == int(horizon))
        ]
        by_seq = group.set_index("seq_len")
        for left, right in PAIR_SEQS:
            if left in by_seq.index and right in by_seq.index:
                l = by_seq.loc[left]
                r = by_seq.loc[right]
                rows.append(
                    {
                        "feature_bundle": bundle,
                        "left_seq_len": left,
                        "right_seq_len": right,
                        "seed": int(seed),
                        "horizon_buckets": int(horizon),
                        "left_validation_improvement": safe_float(l["baseline_improvement_fraction"]),
                        "right_validation_improvement": safe_float(r["baseline_improvement_fraction"]),
                        "improvement_delta_right_minus_left": safe_float(r["baseline_improvement_fraction"]) - safe_float(l["baseline_improvement_fraction"]),
                        "matched_on": "feature_bundle,seed,horizon_buckets",
                    }
                )
    return pd.DataFrame(rows)


def choose_recommendation(
    attribution: pd.DataFrame,
    blocks: pd.DataFrame,
    predictability: pd.DataFrame,
    bundle_cmp: pd.DataFrame,
) -> tuple[str, list[str]]:
    reasons = []
    positive_rows = int((pd.to_numeric(attribution["baseline_improvement_fraction"], errors="coerce") > 0.0).sum())
    positive_blocks = int((pd.to_numeric(blocks["baseline_improvement"], errors="coerce") > 0.0).sum()) if not blocks.empty else 0
    if positive_rows == 0 and positive_blocks == 0:
        reasons.append("no_positive_validation_rows_or_blocks")
        return "stop_current_lane", reasons

    repeat_by_horizon = attribution[pd.to_numeric(attribution["baseline_improvement_fraction"], errors="coerce") > 0].groupby(["feature_bundle", "seq_len", "horizon_buckets"]).size()
    if repeat_by_horizon.empty or int(repeat_by_horizon.max()) < 2:
        reasons.append("no_repeatable_positive_validation_contract")

    block_repeat = blocks[pd.to_numeric(blocks["baseline_improvement"], errors="coerce") > 0].groupby(["horizon_buckets", "validation_block_index"]).size() if not blocks.empty else pd.Series(dtype=int)
    if not block_repeat.empty and int(block_repeat.max()) >= 3:
        reasons.append("specific_validation_blocks_repeat")
        return "run_regime_conditioned_analysis", reasons

    if not bundle_cmp.empty:
        wins = bundle_cmp.copy()
        wins["winner"] = np.where(pd.to_numeric(wins["improvement_delta_right_minus_left"], errors="coerce") > 0, wins["right_bundle"], wins["left_bundle"])
        share = wins["winner"].value_counts(normalize=True)
        if not share.empty and float(share.iloc[0]) >= 0.65:
            reasons.append("one_bundle_wins_matched_comparisons")
            return "improve_feature_bundle", reasons

    collapse_fraction = float(np.mean(attribution["prediction_collapse_status"].astype(str).eq("prediction_collapse"))) if len(attribution) else 0.0
    if collapse_fraction >= 0.50:
        reasons.append("prediction_collapse_dominates")
        return "run_optimizer_scaling_test", reasons

    underfit_fraction = float(np.mean(attribution["underfit_status"].astype(str).eq("model_underfit"))) if len(attribution) else 0.0
    if underfit_fraction >= 0.50 and positive_blocks > 0:
        reasons.append("underfit_with_some_positive_blocks")
        return "run_training_capacity_test", reasons

    weak_predictability = (
        not predictability.empty
        and (pd.to_numeric(predictability["lag_autocorrelation"], errors="coerce").abs().fillna(0.0) < 0.05).all()
        and (pd.to_numeric(predictability["best_boring_baseline_explained_variance_proxy"], errors="coerce").fillna(-1.0) <= 0.05).all()
    )
    shift_strong = (pd.to_numeric(attribution["train_validation_target_shift_score"], errors="coerce").fillna(0.0) > 0.75).mean() > 0.50
    if weak_predictability and shift_strong:
        reasons.append("weak_target_predictability_and_distribution_shift")
        return "redesign_targets", reasons

    reasons.append("no_repeatable_baseline_relative_evidence")
    return "stop_current_lane", reasons


def write_outputs(
    out_dir: Path,
    attribution: pd.DataFrame,
    blocks: pd.DataFrame,
    predictability: pd.DataFrame,
    bundle_cmp: pd.DataFrame,
    seq_cmp: pd.DataFrame,
    recommendation: str,
    recommendation_reasons: list[str],
    contract: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    attribution.to_csv(out_dir / "rawseq_validation_failure_attribution.csv", index=False)
    blocks.to_csv(out_dir / "rawseq_validation_block_attribution.csv", index=False)
    predictability.to_csv(out_dir / "rawseq_target_predictability.csv", index=False)
    bundle_cmp.to_csv(out_dir / "rawseq_bundle_pair_comparison.csv", index=False)
    seq_cmp.to_csv(out_dir / "rawseq_sequence_length_comparison.csv", index=False)
    rollup = attribution.groupby("failure_class", dropna=False).size().reset_index(name="rows").sort_values("rows", ascending=False)
    rollup.to_csv(out_dir / "rawseq_failure_class_rollup.csv", index=False)
    rec = {
        "recommendation": recommendation,
        "allowed_recommendations": sorted(RECOMMENDATIONS),
        "reasons": recommendation_reasons,
        "selection_basis": "validation_only",
        "holdout_used_for_recommendation": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    (out_dir / "recommended_next_experiment.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "rawseq_validation_failure_attribution_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Rawseq Validation Failure Attribution",
        "",
        f"Created at: {contract['created_at']}",
        f"Benchmark: `{contract['benchmark_dir']}`",
        f"Dataset dir: `{contract['dataset_dir']}`",
        f"Survivor dir: `{contract['survivor_dir']}`",
        "",
        "This is report-only. Recommendation logic uses validation data only; holdout is not used.",
        "",
        "## Failure Class Counts",
        rollup.to_string(index=False),
        "",
        "## Target Predictability",
        predictability.to_string(index=False),
        "",
        "## Matched Bundle Comparisons",
        bundle_cmp.groupby(["left_bundle", "right_bundle"], dropna=False)["improvement_delta_right_minus_left"].agg(["count", "mean", "median"]).reset_index().to_string(index=False)
        if not bundle_cmp.empty
        else "No matched bundle comparisons.",
        "",
        "## Matched Sequence-Length Comparisons",
        seq_cmp.groupby(["left_seq_len", "right_seq_len"], dropna=False)["improvement_delta_right_minus_left"].agg(["count", "mean", "median"]).reset_index().to_string(index=False)
        if not seq_cmp.empty
        else "No matched sequence comparisons.",
        "",
        f"## Recommendation\n{recommendation}",
        "",
        "Reasons: " + "; ".join(recommendation_reasons),
        "",
        "Safety: no training, seed expansion, architecture comparison, ensemble search, freeze, promotion, private API, champion mutation, or orders.",
    ]
    (out_dir / "rawseq_validation_failure_attribution.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    benchmark_env = os.getenv("RAWSEQ_VALIDATION_ATTRIBUTION_BENCHMARK_DIR", "").strip()
    benchmark_dir = resolve_path(benchmark_env) if benchmark_env else latest_dir(DEFAULT_BENCHMARK_ROOT, "torch_sequence_benchmark_*")
    if benchmark_dir is None:
        raise SystemExit("Could not find GRU benchmark directory.")
    survivor_env = os.getenv("RAWSEQ_VALIDATION_ATTRIBUTION_SURVIVOR_DIR", "").strip()
    survivor_dir = resolve_path(survivor_env) if survivor_env else latest_dir(DEFAULT_SURVIVOR_ROOT, "rawseq_gru_contract_survivors_*")
    if survivor_dir is None:
        raise SystemExit("Could not find survivor report directory.")
    dataset_env = os.getenv("RAWSEQ_VALIDATION_ATTRIBUTION_DATASET_DIR", "").strip()
    dataset_dir = resolve_path(dataset_env) if dataset_env else infer_dataset_dir(benchmark_dir)
    ablation_dir = infer_ablation_dir(dataset_dir)
    output_root = resolve_path(os.getenv("RAWSEQ_VALIDATION_ATTRIBUTION_OUTPUT_DIR", str(DEFAULT_OUTPUT_ROOT)))
    out_dir = output_root / f"rawseq_validation_failure_attribution_{now_stamp()}"

    summary = pd.read_csv(benchmark_dir / "torch_sequence_model_summary.csv")
    horizon_metrics = pd.read_csv(benchmark_dir / "torch_sequence_per_horizon_metrics.csv")
    selected_policy = pd.read_csv(benchmark_dir / "torch_sequence_selected_policy_metrics.csv")
    dataset_manifest = pd.read_csv(dataset_dir / "sequence_dataset_manifest.csv").reset_index().rename(columns={"index": "dataset_index"})
    validation_baseline = validation_baseline_reference(ablation_dir)
    baseline_table = target_baselines_by_horizon(ablation_dir)

    attribution, prediction_cache, _dataset_cache = build_attribution(summary, horizon_metrics, selected_policy, validation_baseline, dataset_manifest)
    blocks = build_block_attribution(attribution, dataset_manifest, prediction_cache)
    predictability = build_target_predictability(dataset_manifest, baseline_table)
    bundle_cmp = matched_bundle_comparisons(attribution)
    seq_cmp = matched_sequence_comparisons(attribution)
    recommendation, recommendation_reasons = choose_recommendation(attribution, blocks, predictability, bundle_cmp)
    if recommendation not in RECOMMENDATIONS:
        raise AssertionError(f"invalid recommendation={recommendation}")
    contract = {
        "created_at": now_iso(),
        "generator_path": "scripts/tiny/report_rawseq_validation_failure_attribution.py",
        "git_head": git_head(),
        "git_status_dirty": git_status_dirty(),
        "benchmark_dir": str(benchmark_dir),
        "survivor_dir": str(survivor_dir),
        "dataset_dir": str(dataset_dir),
        "ablation_dir": str(ablation_dir),
        "output_dir": str(out_dir),
        "selection_basis": "validation_only",
        "holdout_used_for_recommendation": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    write_outputs(out_dir, attribution, blocks, predictability, bundle_cmp, seq_cmp, recommendation, recommendation_reasons, contract)

    print("Rawseq validation failure attribution complete")
    print(f"Output: {out_dir}")
    print("Failure-class counts:")
    print(attribution.groupby("failure_class").size().reset_index(name="rows").sort_values("rows", ascending=False).to_string(index=False))
    print("Target predictability by horizon:")
    print(predictability.to_string(index=False))
    print("Matched bundle comparisons:")
    if bundle_cmp.empty:
        print("No matched bundle comparisons.")
    else:
        print(bundle_cmp.groupby(["left_bundle", "right_bundle"])["improvement_delta_right_minus_left"].agg(["count", "mean", "median"]).reset_index().to_string(index=False))
    print("Matched sequence-length comparisons:")
    if seq_cmp.empty:
        print("No matched sequence comparisons.")
    else:
        print(seq_cmp.groupby(["left_seq_len", "right_seq_len"])["improvement_delta_right_minus_left"].agg(["count", "mean", "median"]).reset_index().to_string(index=False))
    print(f"Final recommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
