#!/usr/bin/env python3
"""Baseline-first tournament for rawseq target lanes.

This report asks which target is learnable before any long GPU run. It builds
causal target lanes from the existing multi-horizon indicator table and scores
boring train/validation baselines only. Holdout rows are counted but never used
for lane selection.

Safety: paper/research only; no training beyond CPU baselines, no private API,
no orders, no promotion, and no champion mutation.
"""

from __future__ import annotations

import hashlib
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
DEFAULT_INDICATOR_GLOB = "F:/rsio/rawseq_multi_horizon_indicator_full_contract_smoke/mh_indicator_*"
DEFAULT_DIAG_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_feature_diagnostics"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_target_lane_baseline_tournament"

META_COLUMNS = [
    "schema_name",
    "schema_version",
    "schema_sha256",
    "created_at",
    "generator_path",
    "git_head",
    "git_status_dirty",
    "paper_only",
    "orders",
    "promotion",
    "champion_mutation",
]

TARGET_LANES = {
    "coarse_return_vector",
    "future_high_from_now_bps_path",
    "future_low_from_now_bps_path",
    "future_range_envelope_path",
}
BUNDLE_NAMES = ["minimal_core", "balanced_research", "full_registered"]
BORING_BASELINES = {
    "zero_baseline",
    "training_mean_baseline",
    "training_median_baseline",
    "momentum_baseline",
    "mean_reversion_baseline",
    "logistic_direction_baseline",
}
LEARNED_BASELINES = {"ridge_multi_output", "elastic_net_multi_output"}
LANE_STATUS_PRIORITY = {
    "viable_target_lane": 0,
    "regime_research_candidate": 1,
    "fragile_target_lane": 2,
    "baseline_only": 3,
    "reject": 4,
}
TARGET_LANE_SELECTION_PRIORITY = {
    "future_range_envelope_path": 0,
    "coarse_return_vector": 1,
    "future_high_from_now_bps_path": 2,
    "future_low_from_now_bps_path": 3,
}


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


def column_hash(columns: list[str]) -> str:
    return stable_hash(list(columns))


def horizon_buckets_for_target_columns(target_columns: list[str]) -> list[int]:
    return [split_target_meta(column)[0] for column in target_columns]


def infer_bucket_seconds(table: pd.DataFrame, fallback: float = 1.0) -> float:
    if "decision_timestamp" not in table.columns:
        return fallback
    values = pd.to_numeric(table["decision_timestamp"], errors="coerce").dropna().sort_values().to_numpy(dtype=np.float64)
    if len(values) < 2:
        return fallback
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return fallback
    median_diff = float(np.median(diffs))
    return median_diff / 1000.0 if median_diff > 100.0 else median_diff


def parse_csv_strings(text: str, default: list[str]) -> list[str]:
    raw = str(text or "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()] if raw else default


def parse_int_list(text: str, default: list[int]) -> list[int]:
    raw = str(text or "").strip()
    if not raw:
        return default
    return [int(float(item.strip())) for item in raw.split(",") if item.strip()]


def parse_float_list(text: str, default: list[float]) -> list[float]:
    raw = str(text or "").strip()
    if not raw:
        return default
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def schema_meta(name: str, version: str, schema_hash: str) -> dict[str, Any]:
    return {
        "schema_name": name,
        "schema_version": version,
        "schema_sha256": schema_hash,
        "created_at": now_iso(),
        "generator_path": "scripts/tiny/report_rawseq_target_lane_baseline_tournament.py",
        "git_head": git_head(),
        "git_status_dirty": git_status_dirty(),
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }


def with_meta(rows: list[dict[str, Any]], name: str, version: str, schema_hash: str) -> list[dict[str, Any]]:
    meta = schema_meta(name, version, schema_hash)
    return [{**meta, **row} for row in rows]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for col in META_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    frame.to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_schema_hashes() -> dict[str, str]:
    paths = [
        PROJECT_ROOT / "configs" / "rawseq" / "rawseq_feature_schema_v1.json",
        PROJECT_ROOT / "configs" / "rawseq" / "rawseq_label_schema_v1.json",
        PROJECT_ROOT / "configs" / "rawseq" / "rawseq_feature_groups_v1.json",
        PROJECT_ROOT / "configs" / "rawseq" / "rawseq_tensor_contracts_v1.json",
    ]
    return {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in paths if path.exists()}


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


def build_target_lane(table: pd.DataFrame, lane: str, horizons: list[int]) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    if lane not in TARGET_LANES:
        raise ValueError(f"Unsupported target lane: {lane}")
    out = table.copy()
    price = pd.to_numeric(out["close"] if "close" in out.columns else out["price"], errors="coerce")
    target_columns: list[str] = []
    manifest: list[dict[str, Any]] = []
    for horizon in sorted(set(horizons)):
        future_price = price.shift(-horizon)
        future_high = future_extreme(price, horizon, "max")
        future_low = future_extreme(price, horizon, "min")
        ret = log_bps(future_price, price)
        high = np.maximum(log_bps(future_high, price), 0.0)
        low = np.minimum(log_bps(future_low, price), 0.0)
        label_ts_col = f"target_lane_label_end_timestamp_h{horizon}"
        out[label_ts_col] = out["decision_timestamp"].shift(-horizon)
        if lane == "coarse_return_vector":
            items = [(f"coarse_return_bps_h{horizon}", ret, "market_relative_future_return_bps")]
        elif lane == "future_high_from_now_bps_path":
            items = [(f"future_high_from_now_bps_h{horizon}", high, "zero_inclusive_future_upper_envelope_bps")]
        elif lane == "future_low_from_now_bps_path":
            items = [(f"future_range_low_bps_h{horizon}", low, "zero_inclusive_future_lower_envelope_bps")]
        else:
            items = [
                (f"future_range_high_bps_h{horizon}", high, "zero_inclusive_future_upper_envelope_bps"),
                (f"future_range_low_bps_h{horizon}", low, "zero_inclusive_future_lower_envelope_bps"),
            ]
        for column, values, target_type in items:
            out[column] = values
            target_columns.append(column)
            manifest.append(
                {
                    "target_lane": lane,
                    "target_column": column,
                    "horizon_buckets": horizon,
                    "label_end_timestamp_column": label_ts_col,
                    "target_type": target_type,
                    "causal_label": True,
                    "holdout_used": False,
                }
            )
    return out.replace([np.inf, -np.inf], np.nan), target_columns, manifest


def to_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


def split_frame(table: pd.DataFrame, split: str) -> pd.DataFrame:
    return table[table["split"].astype(str).eq(split)].copy()


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2))) if mask.any() else math.nan


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    return float(np.mean(np.abs(actual[mask] - pred[mask]))) if mask.any() else math.nan


def corr(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() < 3:
        return math.nan
    a = actual[mask]
    p = pred[mask]
    if float(np.std(a)) <= 1e-12 or float(np.std(p)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, p)[0, 1])


def directional_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    return float(np.mean((actual[mask] > 0.0) == (pred[mask] > 0.0))) if mask.any() else math.nan


def fill_targets(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(y, axis=0)
    median = np.nanmedian(y, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    median = np.where(np.isfinite(median), median, 0.0)
    return np.where(np.isfinite(y), y, mean), mean, median


def fit_preprocessor(train: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    values = to_matrix(train, columns)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return {"mean": mean, "std": std}


def transform(frame: pd.DataFrame, columns: list[str], pre: dict[str, np.ndarray]) -> np.ndarray:
    values = to_matrix(frame, columns)
    values = np.where(np.isfinite(values), values, pre["mean"])
    out = (values - pre["mean"]) / pre["std"]
    out[~np.isfinite(out)] = 0.0
    return out


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def predict_linear(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x), dtype=np.float64), x]) @ coef


def fit_elastic_net_proxy(x: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float, iterations: int, learning_rate: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    coef = np.zeros((design.shape[1], y.shape[1]), dtype=np.float64)
    scale = np.nanstd(y, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    y_scaled = y / scale
    n = max(1, len(design))
    l1 = max(0.0, min(1.0, l1_ratio))
    l2 = 1.0 - l1
    for _ in range(max(1, iterations)):
        pred = design @ coef
        err = pred - y_scaled
        grad = design.T @ err / n
        grad[1:] += alpha * l2 * coef[1:]
        grad[1:] += alpha * l1 * np.sign(coef[1:])
        coef -= learning_rate * grad
    return coef * scale


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def fit_logistic_direction(x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray) -> np.ndarray:
    design_train = np.column_stack([np.ones(len(x_train), dtype=np.float64), x_train])
    design_all = np.column_stack([np.ones(len(x_all), dtype=np.float64), x_all])
    out = np.zeros((len(x_all), y_train.shape[1]), dtype=np.float64)
    for idx in range(y_train.shape[1]):
        y = (y_train[:, idx] > 0.0).astype(np.float64)
        coef = np.zeros(design_train.shape[1], dtype=np.float64)
        if len(np.unique(y)) >= 2:
            for _ in range(120):
                pred = sigmoid(design_train @ coef)
                grad = design_train.T @ (pred - y) / max(1, len(y))
                coef -= 0.05 * grad
        else:
            base = np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)
            coef[0] = math.log(base / (1.0 - base))
        prob = sigmoid(design_all @ coef)
        magnitude = float(np.nanmedian(np.abs(y_train[:, idx])))
        if not math.isfinite(magnitude) or magnitude <= 0.0:
            magnitude = 1.0
        out[:, idx] = (prob - 0.5) * 2.0 * magnitude
    return out


def recent_return(table: pd.DataFrame) -> np.ndarray:
    if "bucket_return_bps" in table.columns:
        values = pd.to_numeric(table["bucket_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        price = pd.to_numeric(table["close"] if "close" in table.columns else table["price"], errors="coerce")
        values = log_bps(price, price.shift(1))
    values = np.where(np.isfinite(values), values, 0.0)
    return values


def build_baselines(
    table: pd.DataFrame,
    target_columns: list[str],
    feature_columns: list[str],
    ridge_alphas: list[float],
    elastic_alphas: list[float],
    elastic_l1_ratio: float,
    elastic_iterations: int,
    elastic_lr: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    train = split_frame(table, "train")
    val = split_frame(table, "validation")
    train_mask = table["split"].astype(str).eq("train").to_numpy()
    val_mask = table["split"].astype(str).eq("validation").to_numpy()
    y_train_raw = to_matrix(train, target_columns)
    y_val = to_matrix(val, target_columns)
    y_train, y_mean, y_median = fill_targets(y_train_raw)
    predictions: dict[str, np.ndarray] = {
        "zero_baseline": np.zeros((len(table), len(target_columns)), dtype=np.float64),
        "training_mean_baseline": np.tile(y_mean, (len(table), 1)),
        "training_median_baseline": np.tile(y_median, (len(table), 1)),
    }
    manifest = [
        {"model": "zero_baseline", "base_model": "zero_baseline", "selection_stage": "none", "status": "ok"},
        {"model": "training_mean_baseline", "base_model": "training_mean_baseline", "selection_stage": "train_fit_constant", "status": "ok"},
        {"model": "training_median_baseline", "base_model": "training_median_baseline", "selection_stage": "train_fit_constant", "status": "ok"},
    ]
    recent = recent_return(table)
    scales = np.asarray([max(1.0, abs(float(re.search(r"h(\d+)$", col).group(1))) if re.search(r"h(\d+)$", col) else 1.0) for col in target_columns])
    predictions["momentum_baseline"] = recent[:, None] * np.sqrt(scales)[None, :]
    predictions["mean_reversion_baseline"] = -predictions["momentum_baseline"]
    manifest.extend(
        [
            {"model": "momentum_baseline", "base_model": "momentum_baseline", "selection_stage": "causal_recent_return", "status": "ok"},
            {"model": "mean_reversion_baseline", "base_model": "mean_reversion_baseline", "selection_stage": "causal_recent_return", "status": "ok"},
        ]
    )
    if not feature_columns:
        return predictions, manifest
    pre = fit_preprocessor(train, feature_columns)
    x_all = transform(table, feature_columns, pre)
    x_train = x_all[train_mask]
    x_val = x_all[val_mask]
    ridge_candidates = []
    for alpha in ridge_alphas:
        coef = fit_ridge(x_train, y_train, alpha)
        ridge_candidates.append((rmse(y_val, predict_linear(x_val, coef)), alpha, coef))
    score, alpha, coef = sorted(ridge_candidates, key=lambda item: item[0] if math.isfinite(item[0]) else 1e18)[0]
    predictions["ridge_multi_output"] = predict_linear(x_all, coef)
    manifest.append({"model": "ridge_multi_output", "base_model": "ridge_multi_output", "selection_stage": "validation_selected_alpha", "selected_alpha": alpha, "validation_combined_rmse": score, "status": "ok"})
    elastic_candidates = []
    for alpha in elastic_alphas:
        coef = fit_elastic_net_proxy(x_train, y_train, alpha, elastic_l1_ratio, elastic_iterations, elastic_lr)
        elastic_candidates.append((rmse(y_val, predict_linear(x_val, coef)), alpha, coef))
    score, alpha, coef = sorted(elastic_candidates, key=lambda item: item[0] if math.isfinite(item[0]) else 1e18)[0]
    predictions["elastic_net_multi_output"] = predict_linear(x_all, coef)
    manifest.append({"model": "elastic_net_multi_output", "base_model": "elastic_net_multi_output", "selection_stage": "validation_selected_alpha", "selected_alpha": alpha, "validation_combined_rmse": score, "status": "ok"})
    try:
        predictions["logistic_direction_baseline"] = fit_logistic_direction(x_train, y_train, x_all)
        manifest.append({"model": "logistic_direction_baseline", "base_model": "logistic_direction_baseline", "selection_stage": "train_fit_direction_diagnostic", "status": "ok"})
    except Exception as exc:
        manifest.append({"model": "logistic_direction_baseline", "base_model": "logistic_direction_baseline", "selection_stage": "train_fit_direction_diagnostic", "status": "failed", "failure": str(exc)})
    return predictions, manifest


def split_target_meta(column: str) -> tuple[int, str]:
    match = re.search(r"h(\d+)$", str(column))
    horizon = int(match.group(1)) if match else 1
    if "high" in column:
        side = "high"
    elif "low" in column:
        side = "low"
    else:
        side = "return"
    return horizon, side


def validation_block_scores(actual: np.ndarray, pred: np.ndarray, mean_pred: np.ndarray, blocks: int) -> dict[str, Any]:
    bounds = np.linspace(0, len(actual), blocks + 1, dtype=int)
    improvements = []
    for idx in range(blocks):
        sl = slice(bounds[idx], bounds[idx + 1])
        model_rmse = rmse(actual[sl], pred[sl])
        base_rmse = rmse(actual[sl], mean_pred[sl])
        if math.isfinite(model_rmse) and math.isfinite(base_rmse) and base_rmse > 0:
            improvements.append((base_rmse - model_rmse) / base_rmse)
    if not improvements:
        return {"positive_block_fraction": math.nan, "mean_block_rmse_improvement": math.nan, "worst_block_rmse_improvement": math.nan}
    arr = np.asarray(improvements, dtype=np.float64)
    return {
        "positive_block_fraction": float(np.mean(arr > 0.0)),
        "mean_block_rmse_improvement": float(np.mean(arr)),
        "worst_block_rmse_improvement": float(np.min(arr)),
    }


def target_diagnostics(actual: np.ndarray, horizon: int) -> dict[str, Any]:
    finite = actual[np.isfinite(actual)]
    if len(finite) == 0:
        return {}
    signs = np.sign(finite)
    lag = math.nan
    if len(finite) > 3 and np.std(finite[:-1]) > 1e-12 and np.std(finite[1:]) > 1e-12:
        lag = float(np.corrcoef(finite[:-1], finite[1:])[0, 1])
    return {
        "validation_rows": int(len(finite)),
        "effective_non_overlapping_validation_rows": int(math.floor(len(finite) / max(1, horizon))),
        "target_mean": float(np.mean(finite)),
        "target_std": float(np.std(finite)),
        "target_positive_fraction": float(np.mean(finite > 0.0)),
        "target_negative_fraction": float(np.mean(finite < 0.0)),
        "target_sign_balance": float(min(np.mean(signs > 0.0), np.mean(signs < 0.0))),
        "target_abs_p50": float(np.quantile(np.abs(finite), 0.50)),
        "target_abs_p90": float(np.quantile(np.abs(finite), 0.90)),
        "target_abs_p99": float(np.quantile(np.abs(finite), 0.99)),
        "lag1_autocorrelation": lag,
    }


def metric_rows(
    table: pd.DataFrame,
    target_lane: str,
    target_columns: list[str],
    predictions: dict[str, np.ndarray],
    block_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    predict_rows: list[dict[str, Any]] = []
    train_mask = table["split"].astype(str).eq("train").to_numpy()
    val_mask = table["split"].astype(str).eq("validation").to_numpy()
    y_train = to_matrix(table[train_mask], target_columns)
    y_val = to_matrix(table[val_mask], target_columns)
    mean_pred_val = predictions["training_mean_baseline"][val_mask]
    zero_pred_val = predictions["zero_baseline"][val_mask]
    median_pred_val = predictions["training_median_baseline"][val_mask]
    for model, pred in predictions.items():
        pred_val = pred[val_mask]
        pred_train = pred[train_mask]
        for idx, column in enumerate(target_columns):
            horizon, side = split_target_meta(column)
            train_actual = y_train[:, idx]
            val_actual = y_val[:, idx]
            val_pred = pred_val[:, idx]
            val_rmse = rmse(val_actual, val_pred)
            mean_rmse = rmse(val_actual, mean_pred_val[:, idx])
            zero_rmse = rmse(val_actual, zero_pred_val[:, idx])
            median_mae = mae(val_actual, median_pred_val[:, idx])
            val_mae = mae(val_actual, val_pred)
            improvement_mean = (mean_rmse - val_rmse) / mean_rmse if math.isfinite(mean_rmse) and mean_rmse > 0 else math.nan
            improvement_zero = (zero_rmse - val_rmse) / zero_rmse if math.isfinite(zero_rmse) and zero_rmse > 0 else math.nan
            improvement_median_mae = (median_mae - val_mae) / median_mae if math.isfinite(median_mae) and median_mae > 0 else math.nan
            block = validation_block_scores(val_actual, val_pred, mean_pred_val[:, idx], block_count)
            pred_std = float(np.nanstd(val_pred))
            target_std = float(np.nanstd(val_actual))
            variance_ratio = pred_std / target_std if math.isfinite(target_std) and target_std > 1e-12 else math.nan
            row = {
                "target_lane": target_lane,
                "target_column": column,
                "target_side": side,
                "horizon_buckets": horizon,
                "model": model,
                "base_model": model,
                "train_rmse": rmse(train_actual, pred_train[:, idx]),
                "validation_rmse": val_rmse,
                "train_mae": mae(train_actual, pred_train[:, idx]),
                "validation_mae": val_mae,
                "train_correlation": corr(train_actual, pred_train[:, idx]),
                "validation_correlation": corr(val_actual, val_pred),
                "validation_directional_accuracy": directional_accuracy(val_actual, val_pred),
                "mean_baseline_validation_rmse": mean_rmse,
                "zero_baseline_validation_rmse": zero_rmse,
                "median_baseline_validation_mae": median_mae,
                "validation_rmse_improvement_vs_mean": improvement_mean,
                "validation_rmse_improvement_vs_zero": improvement_zero,
                "validation_mae_improvement_vs_median": improvement_median_mae,
                "prediction_mean": float(np.nanmean(val_pred)),
                "prediction_std": pred_std,
                "prediction_to_target_std_ratio": variance_ratio,
                "holdout_used": False,
                **target_diagnostics(val_actual, horizon),
                **block,
            }
            rows.append(row)
            if model not in {"zero_baseline", "training_mean_baseline", "training_median_baseline"}:
                predict_rows.append(row)
            bounds = np.linspace(0, len(val_actual), block_count + 1, dtype=int)
            for block_idx in range(block_count):
                sl = slice(bounds[block_idx], bounds[block_idx + 1])
                model_rmse = rmse(val_actual[sl], val_pred[sl])
                base_rmse = rmse(val_actual[sl], mean_pred_val[sl, idx])
                block_rows.append(
                    {
                        "target_lane": target_lane,
                        "target_column": column,
                        "target_side": side,
                        "horizon_buckets": horizon,
                        "model": model,
                        "validation_block_index": block_idx,
                        "block_rows": int(bounds[block_idx + 1] - bounds[block_idx]),
                        "block_rmse": model_rmse,
                        "mean_baseline_block_rmse": base_rmse,
                        "training_mean_block_rmse": base_rmse,
                        "block_rmse_improvement_vs_training_mean": (base_rmse - model_rmse) / base_rmse if math.isfinite(base_rmse) and base_rmse > 0 else math.nan,
                        "block_rmse_improvement_vs_mean": (base_rmse - model_rmse) / base_rmse if math.isfinite(base_rmse) and base_rmse > 0 else math.nan,
                        "block_target_mean": float(np.nanmean(val_actual[sl])) if bounds[block_idx + 1] > bounds[block_idx] else math.nan,
                        "block_target_std": float(np.nanstd(val_actual[sl])) if bounds[block_idx + 1] > bounds[block_idx] else math.nan,
                        "block_prediction_mean": float(np.nanmean(val_pred[sl])) if bounds[block_idx + 1] > bounds[block_idx] else math.nan,
                        "block_prediction_std": float(np.nanstd(val_pred[sl])) if bounds[block_idx + 1] > bounds[block_idx] else math.nan,
                        "block_correlation": corr(val_actual[sl], val_pred[sl]),
                        "block_directional_accuracy": directional_accuracy(val_actual[sl], val_pred[sl]),
                        "holdout_used": False,
                    }
                )
    return rows, block_rows, predict_rows


def lane_decisions(metric_frame: pd.DataFrame, min_improvement: float, min_block_fraction: float, min_effective_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_models = metric_frame[~metric_frame["model"].isin(["zero_baseline", "training_mean_baseline", "training_median_baseline"])].copy()
    for lane, group in candidate_models.groupby("target_lane", dropna=False):
        best_by_target = []
        for _, target_group in group.groupby("target_column", dropna=False):
            sorted_group = target_group.sort_values(
                ["validation_rmse_improvement_vs_mean", "positive_block_fraction", "validation_correlation"],
                ascending=[False, False, False],
            )
            if not sorted_group.empty:
                best_by_target.append(sorted_group.iloc[0].to_dict())
        improvements = [float(row.get("validation_rmse_improvement_vs_mean", math.nan)) for row in best_by_target]
        blocks = [float(row.get("positive_block_fraction", math.nan)) for row in best_by_target]
        effective_rows = [float(row.get("effective_non_overlapping_validation_rows", math.nan)) for row in best_by_target]
        repeatable = [
            row
            for row in best_by_target
            if float(row.get("validation_rmse_improvement_vs_mean", math.nan)) >= min_improvement
            and float(row.get("positive_block_fraction", math.nan)) >= min_block_fraction
            and float(row.get("effective_non_overlapping_validation_rows", 0.0)) >= min_effective_rows
        ]
        if repeatable:
            status = "target_lane_baseline_survivor"
        elif any(math.isfinite(x) and x > 0.0 for x in improvements):
            status = "target_lane_research_only_unstable"
        else:
            status = "target_lane_rejected"
        best = sorted(best_by_target, key=lambda row: float(row.get("validation_rmse_improvement_vs_mean", -1e18)), reverse=True)
        best_row = best[0] if best else {}
        rows.append(
            {
                "target_lane": lane,
                "lane_status": status,
                "target_columns_evaluated": int(len(best_by_target)),
                "repeatable_positive_targets": int(len(repeatable)),
                "best_model": best_row.get("model", ""),
                "best_target_column": best_row.get("target_column", ""),
                "best_horizon_buckets": best_row.get("horizon_buckets", math.nan),
                "best_validation_rmse_improvement_vs_mean": best_row.get("validation_rmse_improvement_vs_mean", math.nan),
                "best_positive_block_fraction": best_row.get("positive_block_fraction", math.nan),
                "best_effective_non_overlapping_validation_rows": best_row.get("effective_non_overlapping_validation_rows", math.nan),
                "mean_best_target_improvement": float(np.nanmean(improvements)) if improvements else math.nan,
                "mean_best_target_positive_block_fraction": float(np.nanmean(blocks)) if blocks else math.nan,
                "min_effective_non_overlapping_validation_rows": float(np.nanmin(effective_rows)) if effective_rows else math.nan,
                "selection_stage": "train_validation_baseline_tournament",
                "holdout_used": False,
            }
        )
    return rows


def build_regime_rows(table: pd.DataFrame, target_lane: str, target_columns: list[str], predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    val_mask = table["split"].astype(str).eq("validation").to_numpy()
    validation = table[val_mask].copy()
    if validation.empty:
        return []
    y_val = to_matrix(validation, target_columns)
    mean_pred = predictions["training_mean_baseline"][val_mask]
    rows: list[dict[str, Any]] = []
    regime_specs: dict[str, pd.Series] = {}
    if "bucket_return_bps" in validation.columns:
        vol = pd.to_numeric(validation["bucket_return_bps"], errors="coerce").rolling(60, min_periods=10).std().bfill().ffill()
        regime_specs["volatility_tertile"] = pd.qcut(vol.rank(method="first"), 3, labels=["low", "mid", "high"])
    spread_col = next((col for col in ["spread_percent", "spread_bps"] if col in validation.columns), "")
    if spread_col:
        spread = pd.to_numeric(validation[spread_col], errors="coerce").bfill().ffill()
        regime_specs["liquidity_spread_tertile"] = pd.qcut(spread.rank(method="first"), 3, labels=["tight", "mid", "wide"])
    if "decision_timestamp" in validation.columns:
        hour = pd.to_datetime(pd.to_numeric(validation["decision_timestamp"], errors="coerce"), unit="ms", utc=True).dt.hour
        regime_specs["utc_session"] = pd.cut(hour, bins=[-1, 7, 15, 23], labels=["asia_us_night", "europe_us_morning", "us_afternoon"], include_lowest=True)
    best_model_name = next((name for name in predictions if name in {"ridge_multi_output", "elastic_net_multi_output", "momentum_baseline", "mean_reversion_baseline"}), "")
    if not best_model_name:
        return rows
    pred_val = predictions[best_model_name][val_mask]
    for regime_name, labels in regime_specs.items():
        for regime_value in sorted(labels.dropna().astype(str).unique()):
            mask = labels.astype(str).eq(regime_value).to_numpy()
            if not mask.any():
                continue
            for idx, column in enumerate(target_columns):
                horizon, side = split_target_meta(column)
                model_rmse = rmse(y_val[mask, idx], pred_val[mask, idx])
                base_rmse = rmse(y_val[mask, idx], mean_pred[mask, idx])
                rows.append(
                    {
                        "target_lane": target_lane,
                        "target_column": column,
                        "target_side": side,
                        "horizon_buckets": horizon,
                        "model": best_model_name,
                        "regime_name": regime_name,
                        "regime_value": regime_value,
                        "regime_rows": int(mask.sum()),
                        "regime_rmse": model_rmse,
                        "mean_baseline_regime_rmse": base_rmse,
                        "regime_rmse_improvement_vs_mean": (base_rmse - model_rmse) / base_rmse if math.isfinite(base_rmse) and base_rmse > 0 else math.nan,
                        "holdout_used": False,
                    }
                )
    return rows


def load_feature_columns(diag_dir: Path | None, bundle_name: str, table: pd.DataFrame, max_features: int) -> list[str]:
    if diag_dir is not None:
        bundle_path = diag_dir / f"feature_bundle_{bundle_name}.json"
        if bundle_path.exists():
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            columns = [col for col in bundle.get("ordered_feature_columns", []) if col in table.columns]
            return columns[:max_features] if max_features > 0 else columns
    excluded_prefixes = ("future_", "label_", "target_lane_", "coarse_return_")
    excluded = {"split", "decision_timestamp", "label_end_timestamp", "open", "high", "low", "close", "price"}
    columns = [
        col
        for col in table.columns
        if col not in excluded
        and not any(str(col).startswith(prefix) for prefix in excluded_prefixes)
        and pd.api.types.is_numeric_dtype(table[col])
    ]
    return columns[:max_features] if max_features > 0 else columns


def load_bundle_columns(diag_dir: Path | None, bundle_name: str, table: pd.DataFrame, max_features: int) -> tuple[list[str], str]:
    columns = load_feature_columns(diag_dir, bundle_name, table, max_features)
    if diag_dir is not None:
        bundle_path = diag_dir / f"feature_bundle_{bundle_name}.json"
        if bundle_path.exists():
            payload = json.loads(bundle_path.read_text(encoding="utf-8"))
            return columns, str(payload.get("feature_columns_sha256") or column_hash(columns))
    return columns, column_hash(columns)


def finite_fraction(values: np.ndarray) -> float:
    total = values.size
    return float(np.isfinite(values).sum() / total) if total else math.nan


def missing_fraction(values: np.ndarray) -> float:
    total = values.size
    return float(pd.isna(values).sum() / total) if total else math.nan


def vector_rmse_from_rows(frame: pd.DataFrame) -> float:
    values = pd.to_numeric(frame["validation_rmse"], errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if len(values) else math.nan


def vector_mae_from_rows(frame: pd.DataFrame) -> float:
    values = pd.to_numeric(frame["validation_mae"], errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else math.nan


def best_model_for_side(metrics: pd.DataFrame, side: str | None) -> dict[str, Any]:
    frame = metrics.copy()
    if side is not None:
        frame = frame[frame["target_side"].astype(str).eq(side)].copy()
    if frame.empty:
        return {"model": "", "rmse": math.nan, "mae": math.nan}
    rows = []
    for model, group in frame.groupby("model", dropna=False):
        rows.append({"model": str(model), "rmse": vector_rmse_from_rows(group), "mae": vector_mae_from_rows(group)})
    rows = sorted(rows, key=lambda row: row["rmse"] if math.isfinite(row["rmse"]) else 1e18)
    return rows[0] if rows else {"model": "", "rmse": math.nan, "mae": math.nan}


def best_boring_and_learned(metrics: pd.DataFrame, side: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = metrics.copy()
    if side is not None:
        frame = frame[frame["target_side"].astype(str).eq(side)].copy()
    boring = best_model_for_side(frame[frame["model"].isin(BORING_BASELINES)], None)
    learned = best_model_for_side(frame[frame["model"].isin(LEARNED_BASELINES)], None)
    return boring, learned


def improvement(baseline_rmse: float, model_rmse: float) -> float:
    return (baseline_rmse - model_rmse) / baseline_rmse if math.isfinite(baseline_rmse) and baseline_rmse > 0 and math.isfinite(model_rmse) else math.nan


def vector_rmse(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr**2))) if len(arr) else math.nan


def comparison_block_rows(
    blocks: pd.DataFrame,
    target_lane: str,
    feature_bundle: str,
    comparison_scope: str,
    learned_model: str,
    boring_model: str,
    side: str | None,
) -> list[dict[str, Any]]:
    learned_frame = blocks[blocks["model"].astype(str).eq(str(learned_model))].copy()
    boring_frame = blocks[blocks["model"].astype(str).eq(str(boring_model))].copy()
    mean_frame = blocks[blocks["model"].astype(str).eq("training_mean_baseline")].copy()
    if side is not None:
        learned_frame = learned_frame[learned_frame["target_side"].astype(str).eq(side)].copy()
        boring_frame = boring_frame[boring_frame["target_side"].astype(str).eq(side)].copy()
        mean_frame = mean_frame[mean_frame["target_side"].astype(str).eq(side)].copy()
    rows: list[dict[str, Any]] = []
    block_ids = sorted(set(learned_frame["validation_block_index"].dropna().tolist()))
    for block_idx in block_ids:
        learned_values = learned_frame[learned_frame["validation_block_index"].eq(block_idx)]["block_rmse"]
        boring_values = boring_frame[boring_frame["validation_block_index"].eq(block_idx)]["block_rmse"]
        mean_values = mean_frame[mean_frame["validation_block_index"].eq(block_idx)]["block_rmse"]
        learned_rmse = vector_rmse(learned_values)
        boring_rmse = vector_rmse(boring_values)
        mean_rmse = vector_rmse(mean_values)
        best_improvement = improvement(boring_rmse, learned_rmse)
        mean_improvement = improvement(mean_rmse, learned_rmse)
        beats = bool(math.isfinite(best_improvement) and best_improvement > 0.0)
        rows.append(
            {
                "target_lane": target_lane,
                "feature_bundle": feature_bundle,
                "comparison_scope": comparison_scope,
                "validation_block_index": int(block_idx),
                "best_boring_baseline": boring_model,
                "learned_model_used": learned_model,
                "learned_vector_block_rmse": learned_rmse,
                "best_boring_vector_block_rmse": boring_rmse,
                "block_rmse_improvement_vs_best_boring": best_improvement,
                "block_beats_best_boring": beats,
                "block_comparison_status": "pass" if beats else "fail",
                "block_rmse_improvement_vs_training_mean_diagnostic": mean_improvement,
                "holdout_used": False,
            }
        )
    return rows


def block_evidence_from_comparisons(comparison_rows: list[dict[str, Any]], comparison_scope: str) -> dict[str, Any]:
    values = np.asarray(
        [
            safe_float(row.get("block_rmse_improvement_vs_best_boring"))
            for row in comparison_rows
            if str(row.get("comparison_scope")) == comparison_scope
        ],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    return {
        "positive_block_count": int((values > 0.0).sum()) if len(values) else 0,
        "positive_block_fraction": float(np.mean(values > 0.0)) if len(values) else math.nan,
        "worst_block_improvement": float(np.min(values)) if len(values) else math.nan,
        "mean_block_improvement": float(np.mean(values)) if len(values) else math.nan,
    }


def target_distribution_summary(table: pd.DataFrame, target_columns: list[str], split: str) -> dict[str, Any]:
    frame = split_frame(table, split)
    values = to_matrix(frame, target_columns)
    finite = values[np.isfinite(values)]
    prefix = "train" if split == "train" else "validation"
    if len(finite) == 0:
        return {
            f"{prefix}_target_mean": math.nan,
            f"{prefix}_target_std": math.nan,
            f"{prefix}_target_p10": math.nan,
            f"{prefix}_target_p50": math.nan,
            f"{prefix}_target_p90": math.nan,
        }
    return {
        f"{prefix}_target_mean": float(np.mean(finite)),
        f"{prefix}_target_std": float(np.std(finite)),
        f"{prefix}_target_p10": float(np.quantile(finite, 0.10)),
        f"{prefix}_target_p50": float(np.quantile(finite, 0.50)),
        f"{prefix}_target_p90": float(np.quantile(finite, 0.90)),
    }


def target_shift(train: dict[str, Any], validation: dict[str, Any]) -> float:
    train_std = float(train.get("train_target_std", math.nan))
    train_mean = float(train.get("train_target_mean", math.nan))
    val_mean = float(validation.get("validation_target_mean", math.nan))
    if math.isfinite(train_std) and train_std > 1e-12 and math.isfinite(train_mean) and math.isfinite(val_mean):
        return abs(val_mean - train_mean) / train_std
    return math.nan


def constraint_summary(table: pd.DataFrame, target_lane: str, target_columns: list[str]) -> dict[str, Any]:
    values = table[target_columns].apply(pd.to_numeric, errors="coerce")
    high_cols = [col for col in target_columns if "high" in col]
    low_cols = [col for col in target_columns if "low" in col]
    out = {
        "target_constraint_status": "pass",
        "high_sign_violation_fraction": 0.0,
        "low_sign_violation_fraction": 0.0,
        "high_monotonic_violation_fraction": 0.0,
        "low_monotonic_violation_fraction": 0.0,
        "envelope_order_violation_fraction": 0.0,
    }
    if high_cols:
        high = values[high_cols].to_numpy(dtype=np.float64)
        out["high_sign_violation_fraction"] = float(np.nanmean(high < -1e-9))
        if high.shape[1] > 1:
            out["high_monotonic_violation_fraction"] = float(np.nanmean(np.diff(high, axis=1) < -1e-9))
    if low_cols:
        low = values[low_cols].to_numpy(dtype=np.float64)
        out["low_sign_violation_fraction"] = float(np.nanmean(low > 1e-9))
        if low.shape[1] > 1:
            out["low_monotonic_violation_fraction"] = float(np.nanmean(np.diff(low, axis=1) > 1e-9))
    if high_cols and low_cols:
        high = values[high_cols].to_numpy(dtype=np.float64)
        low = values[low_cols].to_numpy(dtype=np.float64)
        width = min(high.shape[1], low.shape[1])
        out["envelope_order_violation_fraction"] = float(np.nanmean(high[:, :width] < low[:, :width] - 1e-9))
    violation_keys = [
        "high_sign_violation_fraction",
        "low_sign_violation_fraction",
        "high_monotonic_violation_fraction",
        "low_monotonic_violation_fraction",
        "envelope_order_violation_fraction",
    ]
    if any(float(out[key]) > 1e-9 for key in violation_keys):
        out["target_constraint_status"] = "fail"
    return out


def terminal_correlation(metrics: pd.DataFrame, model: str, side: str) -> float:
    frame = metrics[(metrics["model"].astype(str).eq(str(model))) & (metrics["target_side"].astype(str).eq(side))].copy()
    if frame.empty:
        return math.nan
    frame = frame.sort_values("horizon_buckets")
    return float(frame.iloc[-1].get("validation_correlation", math.nan))


def lane_bundle_row(
    lane_table: pd.DataFrame,
    target_lane: str,
    target_layout: str,
    feature_bundle: str,
    feature_columns: list[str],
    feature_sha: str,
    target_columns: list[str],
    metrics: pd.DataFrame,
    blocks: pd.DataFrame,
    min_blocks: int,
    min_effective_rows: int,
    min_validation_improvement: float,
    max_target_shift: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    target_sha = column_hash(target_columns)
    train_summary = target_distribution_summary(lane_table, target_columns, "train")
    val_summary = target_distribution_summary(lane_table, target_columns, "validation")
    shift = target_shift(train_summary, val_summary)
    val_values = to_matrix(split_frame(lane_table, "validation"), target_columns)
    train_values = to_matrix(split_frame(lane_table, "train"), target_columns)
    missing = missing_fraction(val_values)
    nonfinite = 1.0 - finite_fraction(val_values)
    constraints = constraint_summary(lane_table, target_lane, target_columns)
    horizons = [split_target_meta(col)[0] for col in target_columns]
    effective_rows = int(min([math.floor(np.isfinite(val_values[:, idx]).sum() / max(1, horizons[idx])) for idx in range(len(target_columns))] or [0]))
    boring, learned = best_boring_and_learned(metrics)
    agg_improvement = improvement(float(boring["rmse"]), float(learned["rmse"]))
    comparison_rows = comparison_block_rows(
        blocks,
        target_lane,
        feature_bundle,
        "combined",
        str(learned.get("model", "")),
        str(boring.get("model", "")),
        None,
    )
    combined_blocks = block_evidence_from_comparisons(comparison_rows, "combined")
    side_rows: list[dict[str, Any]] = []
    evidence = {
        "high_validation_rmse": math.nan,
        "low_validation_rmse": math.nan,
        "combined_validation_rmse": float(learned.get("rmse", math.nan)),
        "high_baseline_improvement": math.nan,
        "low_baseline_improvement": math.nan,
        "combined_baseline_improvement": agg_improvement,
        "high_positive_block_count": 0,
        "low_positive_block_count": 0,
        "combined_positive_block_count": int(combined_blocks["positive_block_count"]),
        "terminal_high_correlation": math.nan,
        "terminal_low_correlation": math.nan,
    }
    side_gate_pass = True
    for side in ["high", "low"]:
        if side not in set(metrics["target_side"].astype(str)):
            continue
        side_boring, side_learned = best_boring_and_learned(metrics, side)
        side_improvement = improvement(float(side_boring["rmse"]), float(side_learned["rmse"]))
        side_comparisons = comparison_block_rows(
            blocks,
            target_lane,
            feature_bundle,
            side,
            str(side_learned.get("model", "")),
            str(side_boring.get("model", "")),
            side,
        )
        comparison_rows.extend(side_comparisons)
        side_blocks = block_evidence_from_comparisons(comparison_rows, side)
        evidence[f"{side}_validation_rmse"] = float(side_learned.get("rmse", math.nan))
        evidence[f"{side}_baseline_improvement"] = side_improvement
        evidence[f"{side}_positive_block_count"] = int(side_blocks["positive_block_count"])
        evidence[f"terminal_{side}_correlation"] = terminal_correlation(metrics, str(side_learned.get("model", "")), side)
        side_gate = side_improvement >= min_validation_improvement and int(side_blocks["positive_block_count"]) >= min_blocks
        side_gate_pass = side_gate_pass and side_gate
        side_rows.append(
            {
                "target_lane": target_lane,
                "feature_bundle": feature_bundle,
                "target_side": side,
                "best_boring_baseline": side_boring.get("model", ""),
                "learned_model_used": side_learned.get("model", ""),
                "validation_rmse": side_learned.get("rmse", math.nan),
                "baseline_rmse": side_boring.get("rmse", math.nan),
                "baseline_improvement": side_improvement,
                "positive_block_count": side_blocks["positive_block_count"],
                "positive_blocks_vs_best_boring": side_blocks["positive_block_count"],
                "worst_block_improvement": side_blocks["worst_block_improvement"],
                "worst_block_improvement_vs_best_boring": side_blocks["worst_block_improvement"],
                "side_gate_pass": side_gate,
                "holdout_used": False,
            }
        )
    gates = {
        "aggregate_improvement_positive": bool(math.isfinite(agg_improvement) and agg_improvement >= min_validation_improvement),
        "positive_blocks_pass": int(combined_blocks["positive_block_count"]) >= min_blocks,
        "effective_rows_pass": effective_rows >= min_effective_rows,
        "target_shift_pass": (not math.isfinite(shift)) or shift <= max_target_shift,
        "target_constraints_pass": constraints["target_constraint_status"] == "pass",
        "envelope_sides_pass": True if target_lane != "future_range_envelope_path" else side_gate_pass,
    }
    failed_gates = [key for key, value in gates.items() if not value]
    failed_gate_reasons = [f"failed_{key}" for key in failed_gates]
    if all(gates.values()):
        status = "viable_lane_bundle"
    elif gates["aggregate_improvement_positive"] and gates["target_constraints_pass"]:
        status = "fragile_lane_bundle"
    elif not gates["aggregate_improvement_positive"]:
        status = "baseline_only"
    else:
        status = "reject"
    status_explanation = "all_gates_passed" if not failed_gates else ";".join(failed_gate_reasons)
    row = {
        "target_lane": target_lane,
        "target_layout": target_layout,
        "feature_bundle": feature_bundle,
        "feature_count": len(feature_columns),
        "feature_columns_sha256": feature_sha,
        "target_columns_sha256": target_sha,
        "target_column_order": ";".join(target_columns),
        "train_rows": int(lane_table["split"].astype(str).eq("train").sum()),
        "validation_rows": int(lane_table["split"].astype(str).eq("validation").sum()),
        "effective_non_overlapping_validation_rows": effective_rows,
        "missing_fraction": missing,
        "nonfinite_fraction": nonfinite,
        "target_train_nonfinite_fraction": 1.0 - finite_fraction(train_values),
        "target_validation_nonfinite_fraction": nonfinite,
        **train_summary,
        **val_summary,
        "train_to_validation_target_shift": shift,
        "best_boring_baseline": boring.get("model", ""),
        "learned_baseline_model_used": learned.get("model", ""),
        "aggregate_validation_metric": learned.get("rmse", math.nan),
        "best_boring_validation_metric": boring.get("rmse", math.nan),
        "validation_baseline_improvement_fraction": agg_improvement,
        "positive_validation_block_count": combined_blocks["positive_block_count"],
        "positive_blocks_vs_best_boring": combined_blocks["positive_block_count"],
        "positive_validation_block_fraction": combined_blocks["positive_block_fraction"],
        "worst_validation_block_improvement": combined_blocks["worst_block_improvement"],
        "worst_block_improvement_vs_best_boring": combined_blocks["worst_block_improvement"],
        "validation_block_stability": "pass" if int(combined_blocks["positive_block_count"]) >= min_blocks else "fail",
        **constraints,
        **evidence,
        "lane_bundle_status": status,
        "failed_gates": ";".join(failed_gates),
        "failed_gate_reasons": ";".join(failed_gate_reasons),
        "failed_gate_count": len(failed_gates),
        "status_explanation": status_explanation,
        "reasons": status_explanation,
        "selection_stage": "train_validation_baseline_tournament",
        "holdout_used": False,
    }
    return row, side_rows, comparison_rows


def rank_target_lanes(bundle_rows: list[dict[str, Any]], min_viable_bundles: int) -> list[dict[str, Any]]:
    frame = pd.DataFrame(bundle_rows)
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for lane, group in frame.groupby("target_lane", dropna=False):
        viable = group[group["lane_bundle_status"].astype(str).eq("viable_lane_bundle")].copy()
        viable_unique = viable.drop_duplicates("feature_columns_sha256")
        qualifying_count = int(len(viable_unique))
        if qualifying_count >= min_viable_bundles:
            status = "viable_target_lane"
            qualifying = viable_unique
        elif qualifying_count == 1:
            status = "fragile_target_lane"
            qualifying = viable_unique
        elif (group["lane_bundle_status"].astype(str).eq("fragile_lane_bundle")).any():
            status = "fragile_target_lane"
            qualifying = group[group["lane_bundle_status"].astype(str).isin(["fragile_lane_bundle", "viable_lane_bundle"])]
        elif (group["lane_bundle_status"].astype(str).eq("baseline_only")).all():
            status = "baseline_only"
            qualifying = group
        else:
            status = "reject"
            qualifying = group
        if lane == "future_range_envelope_path":
            score_values = pd.to_numeric(group[["high_baseline_improvement", "low_baseline_improvement", "combined_baseline_improvement"]].min(axis=1), errors="coerce")
        else:
            score_values = pd.to_numeric(group["validation_baseline_improvement_fraction"], errors="coerce")
        q_improve = pd.to_numeric(qualifying["validation_baseline_improvement_fraction"], errors="coerce")
        q_blocks = pd.to_numeric(qualifying["positive_validation_block_count"], errors="coerce")
        q_rows = pd.to_numeric(qualifying["effective_non_overlapping_validation_rows"], errors="coerce")
        q_shift = pd.to_numeric(qualifying["train_to_validation_target_shift"], errors="coerce")
        preferred = group.sort_values(
            ["lane_bundle_status", "validation_baseline_improvement_fraction", "positive_validation_block_count", "effective_non_overlapping_validation_rows", "feature_bundle"],
            ascending=[True, False, False, False, True],
        ).iloc[0]
        rows.append(
            {
                "target_lane": str(lane),
                "target_lane_status": status,
                "qualifying_feature_bundle_count": qualifying_count,
                "qualifying_feature_bundles": ";".join(viable_unique["feature_bundle"].astype(str).tolist()),
                "preferred_feature_bundle": str(preferred.get("feature_bundle", "")),
                "backup_feature_bundle": next((b for b in viable_unique["feature_bundle"].astype(str).tolist() if b != str(preferred.get("feature_bundle", ""))), ""),
                "min_positive_block_count_across_qualifying": int(np.nanmin(q_blocks)) if len(q_blocks.dropna()) else 0,
                "min_baseline_improvement_across_qualifying": float(np.nanmin(q_improve)) if len(q_improve.dropna()) else math.nan,
                "mean_validation_baseline_improvement": float(np.nanmean(score_values)) if len(score_values.dropna()) else math.nan,
                "max_validation_baseline_improvement": float(np.nanmax(score_values)) if len(score_values.dropna()) else math.nan,
                "effective_non_overlapping_rows": int(np.nanmax(q_rows)) if len(q_rows.dropna()) else int(pd.to_numeric(group["effective_non_overlapping_validation_rows"], errors="coerce").max()),
                "target_distribution_shift": float(np.nanmin(q_shift)) if len(q_shift.dropna()) else float(pd.to_numeric(group["train_to_validation_target_shift"], errors="coerce").min()),
                "target_lane_selection_priority": TARGET_LANE_SELECTION_PRIORITY.get(str(lane), 99),
                "selection_stage": "train_validation_baseline_tournament",
                "holdout_used": False,
            }
        )
    rows.sort(
        key=lambda row: (
            LANE_STATUS_PRIORITY.get(row["target_lane_status"], 9),
            int(row.get("target_lane_selection_priority", 99)),
            -int(row["qualifying_feature_bundle_count"]),
            -int(row["min_positive_block_count_across_qualifying"]),
            -(row["min_baseline_improvement_across_qualifying"] if math.isfinite(float(row["min_baseline_improvement_across_qualifying"])) else -1e18),
            -(row["mean_validation_baseline_improvement"] if math.isfinite(float(row["mean_validation_baseline_improvement"])) else -1e18),
            -int(row["effective_non_overlapping_rows"]),
            row["target_distribution_shift"] if math.isfinite(float(row["target_distribution_shift"])) else 1e18,
            row["target_lane"],
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["validation_only_rank"] = idx
    return rows


def build_selected_manifest(
    ranking_rows: list[dict[str, Any]],
    bundle_rows: list[dict[str, Any]],
    target_manifests: list[dict[str, Any]],
    contract: dict[str, Any],
    required_target_lane: str = "",
) -> dict[str, Any]:
    viable = [row for row in ranking_rows if row["target_lane_status"] == "viable_target_lane"]
    if required_target_lane:
        required_viable = [row for row in viable if str(row.get("target_lane")) == required_target_lane]
        if not required_viable:
            return {
                "selection_status": "abstain_required_lane_not_viable",
                "selected_targets": [],
                "required_target_lane": required_target_lane,
                "reasons": ["required_target_lane_not_viable"],
                "validation_only_evidence_summary": ranking_rows,
                "holdout_used_for_selection": False,
                "paper_only": True,
                "orders": False,
                "promotion": False,
                "champion_mutation": False,
            }
        viable = required_viable
    if not viable:
        return {
            "selection_status": "abstain",
            "selected_targets": [],
            "reasons": ["no_viable_target_lane"],
            "validation_only_evidence_summary": ranking_rows,
            "holdout_used_for_selection": False,
            "paper_only": True,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
        }
    selected = viable[0]
    lane = selected["target_lane"]
    preferred_bundle = selected["preferred_feature_bundle"]
    bundle = next(row for row in bundle_rows if row["target_lane"] == lane and row["feature_bundle"] == preferred_bundle)
    target_cols = [col for col in str(bundle["target_column_order"]).split(";") if col]
    lane_manifest = [row for row in target_manifests if row["target_lane"] == lane and row["target_column"] in target_cols]
    target_horizon_buckets = horizon_buckets_for_target_columns(target_cols)
    unique_horizons = sorted(set(target_horizon_buckets))
    selected_target = {
        "target_lane": lane,
        "target_layout": bundle["target_layout"],
        "horizons": unique_horizons,
        "unique_horizon_buckets": unique_horizons,
        "target_horizon_buckets": target_horizon_buckets,
        "target_output_dim": len(target_cols),
        "target_column_order": target_cols,
        "target_columns_sha256": bundle["target_columns_sha256"],
        "qualifying_feature_bundles": selected["qualifying_feature_bundles"].split(";") if selected["qualifying_feature_bundles"] else [],
        "preferred_feature_bundle": preferred_bundle,
        "backup_bundle": selected.get("backup_feature_bundle", ""),
        "required_target_constraints": ["sign", "monotonicity", "envelope_order"],
        "validation_metrics": bundle,
        "positive_validation_blocks": bundle["positive_validation_block_count"],
        "effective_rows": bundle["effective_non_overlapping_validation_rows"],
        "bucket_seconds": contract.get("bucket_seconds", ""),
        "source_cutoff": contract.get("source_cutoff", ""),
        "split_contract": contract.get("split_contract", {}),
        "schema_hashes": contract.get("schema_hashes", {}),
        "selection_rule_version": contract.get("selection_rule_version"),
        "selection_rule_sha256": contract.get("selection_rule_sha256"),
        "holdout_used_for_selection": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    selected_target["selected_target_contract_hash"] = stable_hash(selected_target)
    return {
        "selection_status": "selected",
        "selected_targets": [selected_target],
        "holdout_used_for_selection": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }


def selection_payload_hash(ranking_rows: list[dict[str, Any]], manifest: dict[str, Any], gpu_contract: dict[str, Any]) -> str:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in sorted(value.items()) if "holdout" not in str(k).lower() and "created_at" not in str(k).lower()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    return stable_hash({"ranking": clean(ranking_rows), "manifest": clean(manifest), "gpu": clean(gpu_contract)})


def run_tournament(
    table: pd.DataFrame,
    diag_dir: Path | None,
    target_lanes: list[str],
    horizons: list[int],
    bundles: list[str],
    max_features: int,
    block_count: int,
    min_blocks: int,
    min_effective_rows: int,
    min_validation_improvement: float,
    max_target_shift: float,
    min_viable_bundles: int,
    ridge_alphas: list[float],
    elastic_alphas: list[float],
    elastic_l1_ratio: float,
    elastic_iterations: int,
    elastic_lr: float,
    contract_base: dict[str, Any],
    required_target_lane: str = "",
) -> dict[str, Any]:
    all_manifest: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    all_block_details: list[dict[str, Any]] = []
    all_block_comparisons: list[dict[str, Any]] = []
    all_target_manifest: list[dict[str, Any]] = []
    all_regimes: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []

    for bundle_name in bundles:
        feature_columns, feature_sha = load_bundle_columns(diag_dir, bundle_name, table, max_features)
        for lane in target_lanes:
            lane_table, target_columns, target_manifest = build_target_lane(table, lane, horizons)
            target_layout = (
                "return_vector"
                if lane == "coarse_return_vector"
                else ("range_envelope_path" if lane == "future_range_envelope_path" else "scalar_path")
            )
            predictions, manifest = build_baselines(
                lane_table,
                target_columns,
                feature_columns,
                ridge_alphas,
                elastic_alphas,
                elastic_l1_ratio,
                elastic_iterations,
                elastic_lr,
            )
            metrics, blocks, _ = metric_rows(lane_table, lane, target_columns, predictions, block_count)
            for row in metrics:
                row.update({"feature_bundle": bundle_name, "feature_count": len(feature_columns), "feature_columns_sha256": feature_sha, "target_columns_sha256": column_hash(target_columns)})
            for row in blocks:
                row.update({"feature_bundle": bundle_name, "feature_columns_sha256": feature_sha, "target_columns_sha256": column_hash(target_columns)})
            for row in manifest:
                row.update({"target_lane": lane, "feature_bundle": bundle_name, "feature_count": len(feature_columns), "feature_columns_sha256": feature_sha, "target_columns_sha256": column_hash(target_columns), "holdout_used": False})
            for row in target_manifest:
                row.update({"target_layout": target_layout, "target_columns_sha256": column_hash(target_columns)})
            metric_frame = pd.DataFrame(metrics)
            block_frame = pd.DataFrame(blocks)
            bundle_row, evidence_rows, comparison_rows = lane_bundle_row(
                lane_table,
                lane,
                target_layout,
                bundle_name,
                feature_columns,
                feature_sha,
                target_columns,
                metric_frame,
                block_frame,
                min_blocks,
                min_effective_rows,
                min_validation_improvement,
                max_target_shift,
            )
            registry_rows.append(
                {
                    "target_lane": lane,
                    "target_layout": target_layout,
                    "feature_bundle": bundle_name,
                    "feature_count": len(feature_columns),
                    "feature_columns_sha256": feature_sha,
                    "target_columns_sha256": column_hash(target_columns),
                    "target_column_order": ";".join(target_columns),
                    "primary_horizons": ",".join(str(x) for x in horizons),
                    "selection_stage": "train_validation_baseline_tournament",
                    "holdout_used": False,
                }
            )
            all_manifest.extend(manifest)
            all_metrics.extend(metrics)
            all_block_details.extend(blocks)
            all_block_comparisons.extend(comparison_rows)
            all_target_manifest.extend(target_manifest)
            all_regimes.extend(build_regime_rows(lane_table, lane, target_columns, predictions))
            bundle_rows.append(bundle_row)
            side_rows.extend(evidence_rows)

    ranking_rows = rank_target_lanes(bundle_rows, min_viable_bundles)
    selection_rule_payload = {
        "min_effective_rows": min_effective_rows,
        "min_positive_blocks": min_blocks,
        "validation_block_count": block_count,
        "min_validation_improvement": min_validation_improvement,
        "maximum_target_shift": max_target_shift,
        "minimum_viable_feature_bundles": min_viable_bundles,
        "envelope_requires_high_low_combined": True,
        "block_rmse_aggregation": "sqrt_mean_squared_per_target_rmse",
        "block_comparison_rows": True,
        "required_target_lane": required_target_lane,
    }
    contract = {
        **contract_base,
        "target_lanes": target_lanes,
        "horizon_buckets": horizons,
        "feature_bundles": bundles,
        "train_rows": int(table["split"].astype(str).eq("train").sum()),
        "validation_rows": int(table["split"].astype(str).eq("validation").sum()),
        "holdout_rows_used_for_selection": 0,
        "holdout_rows_present": int(table["split"].astype(str).eq("untouched_holdout").sum()),
        "selection_rule_version": "target_lane_gate_v1.0.0",
        "selection_rule_payload": selection_rule_payload,
        "selection_rule_sha256": stable_hash(selection_rule_payload),
        "schema_hashes": load_schema_hashes(),
        "min_effective_rows": min_effective_rows,
        "min_positive_blocks": min_blocks,
        "validation_block_count": block_count,
        "min_validation_improvement": min_validation_improvement,
        "maximum_target_shift": max_target_shift,
        "minimum_viable_feature_bundles": min_viable_bundles,
        "required_target_lane": required_target_lane,
        "split_contract": {
            "train_rows": int(table["split"].astype(str).eq("train").sum()),
            "validation_rows": int(table["split"].astype(str).eq("validation").sum()),
            "holdout_rows_used_for_selection": 0,
        },
        "source_cutoff": float(pd.to_numeric(table.loc[table["split"].astype(str).eq("validation"), "decision_timestamp"], errors="coerce").max()) if "decision_timestamp" in table.columns else "",
        "safety": {"paper_only": True, "orders": False, "promotion": False, "champion_mutation": False, "gpu_training": False, "ensemble_search": False},
    }
    selected_manifest = build_selected_manifest(ranking_rows, bundle_rows, all_target_manifest, contract, required_target_lane)
    viable = [row for row in ranking_rows if row["target_lane_status"] == "viable_target_lane"]
    gpu_contract = {
        "selection_status": "ready_for_bounded_gpu_screen" if selected_manifest.get("selection_status") == "selected" else "abstain",
        "runnable_gpu_contracts": selected_manifest.get("selected_targets", []) if selected_manifest.get("selection_status") == "selected" else [],
        "reason": ""
        if selected_manifest.get("selection_status") == "selected"
        else ";".join(str(x) for x in selected_manifest.get("reasons", ["no_selected_target"])),
        "holdout_used_for_selection": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    return {
        "target_manifest": all_target_manifest,
        "model_manifest": all_manifest,
        "metrics": all_metrics,
        "block_details": all_block_details,
        "blocks": all_block_comparisons,
        "regimes": all_regimes,
        "registry": registry_rows,
        "bundle_metrics": bundle_rows,
        "ranking": ranking_rows,
        "side_evidence": side_rows,
        "selected_manifest": selected_manifest,
        "recommended_gpu_screen": gpu_contract,
        "contract": contract,
        "selection_hash": selection_payload_hash(ranking_rows, selected_manifest, gpu_contract),
    }


def write_outputs(out_dir: Path, payload: dict[str, Any], version: str, schema_hash: str) -> None:
    ranking_rows = payload["ranking"]
    write_csv(out_dir / "target_lane_registry.csv", with_meta(payload["registry"], "rawseq_target_lane_registry", version, schema_hash))
    write_csv(out_dir / "target_lane_manifest.csv", with_meta(payload["target_manifest"], "rawseq_target_lane_manifest", version, schema_hash))
    write_csv(out_dir / "target_lane_baseline_model_manifest.csv", with_meta(payload["model_manifest"], "rawseq_target_lane_baseline_model_manifest", version, schema_hash))
    write_csv(out_dir / "target_lane_baseline_metrics.csv", with_meta(payload["metrics"], "rawseq_target_lane_baseline_metrics", version, schema_hash))
    write_csv(out_dir / "target_lane_validation_block_metrics.csv", with_meta(payload["blocks"], "rawseq_target_lane_validation_block_comparisons", version, schema_hash))
    write_csv(out_dir / "target_lane_validation_block_detail_metrics.csv", with_meta(payload["block_details"], "rawseq_target_lane_validation_block_details", version, schema_hash))
    write_csv(out_dir / "target_lane_regime_metrics.csv", with_meta(payload["regimes"], "rawseq_target_lane_regime_metrics", version, schema_hash))
    write_csv(out_dir / "target_lane_bundle_metrics.csv", with_meta(payload["bundle_metrics"], "rawseq_target_lane_bundle_metrics", version, schema_hash))
    write_csv(out_dir / "target_lane_ranking.csv", with_meta(ranking_rows, "rawseq_target_lane_ranking", version, schema_hash))
    write_csv(out_dir / "target_lane_decisions.csv", with_meta(ranking_rows, "rawseq_target_lane_decisions", version, schema_hash))
    write_csv(out_dir / "viable_target_lanes.csv", with_meta([r for r in ranking_rows if r["target_lane_status"] == "viable_target_lane"], "rawseq_viable_target_lanes", version, schema_hash))
    write_csv(out_dir / "regime_research_candidates.csv", with_meta([r for r in ranking_rows if r["target_lane_status"] == "regime_research_candidate"], "rawseq_regime_research_target_lanes", version, schema_hash))
    write_csv(out_dir / "fragile_target_lanes.csv", with_meta([r for r in ranking_rows if r["target_lane_status"] == "fragile_target_lane"], "rawseq_fragile_target_lanes", version, schema_hash))
    write_csv(out_dir / "baseline_only_lanes.csv", with_meta([r for r in ranking_rows if r["target_lane_status"] == "baseline_only"], "rawseq_baseline_only_target_lanes", version, schema_hash))
    write_csv(out_dir / "rejected_target_lanes.csv", with_meta([r for r in ranking_rows if r["target_lane_status"] == "reject"], "rawseq_rejected_target_lanes", version, schema_hash))
    write_csv(out_dir / "envelope_side_evidence.csv", with_meta(payload["side_evidence"], "rawseq_envelope_side_evidence", version, schema_hash))
    write_json(out_dir / "selected_target_manifest.json", payload["selected_manifest"])
    write_json(out_dir / "recommended_gpu_screen.json", payload["recommended_gpu_screen"])
    write_json(out_dir / "target_lane_tournament_contract.json", payload["contract"])
    write_json(out_dir / "recommended_next_target_lanes.json", {
        "recommendation": payload["recommended_gpu_screen"]["selection_status"],
        "recommended_target_lanes": [row["target_lane"] for row in ranking_rows if row["target_lane_status"] == "viable_target_lane"],
        "selection_stage": "train_validation_baseline_tournament",
        "holdout_used": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    })
    lines = [
        "# Rawseq Target-Lane Baseline Tournament",
        "",
        f"Created at: {payload['contract']['created_at']}",
        f"Output: `{out_dir}`",
        "",
        "## Scope",
        "- Validation-only target promotion gate.",
        "- Holdout rows are not used for lane, bundle, horizon, or manifest selection.",
        "- No GPU training, ensemble search, freeze, promotion, champion mutation, private API, or orders.",
        "",
        "## Target-Lane Ranking",
    ]
    for row in ranking_rows:
        lines.append(
            f"- {row['validation_only_rank']}. {row['target_lane']}: {row['target_lane_status']} "
            f"bundles={row['qualifying_feature_bundle_count']} "
            f"mean_improvement={float(row['mean_validation_baseline_improvement']):.6f} "
            f"effective_rows={row['effective_non_overlapping_rows']}"
        )
    if not [r for r in ranking_rows if r["target_lane_status"] == "viable_target_lane"]:
        lines.extend(
            [
                "",
                "## Downstream Stop Gate",
                "- viable_target_lanes.csv is empty.",
                "- Do not build target NPZ handoffs.",
                "- Do not launch GRU, TCN, LSTM, MLP, or Transformer screens.",
                "- Do not run architecture comparison.",
                "- Do not run ensemble search.",
                "- Do not freeze a candidate.",
            ]
        )
    (out_dir / "target_lane_tournament.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "target_lane_baseline_tournament_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    indicator_env = os.getenv("RAWSEQ_TARGET_TOURNAMENT_INDICATOR_RUN_DIR", "").strip()
    indicator_dir = resolve_path(indicator_env) if indicator_env else latest_dir(DEFAULT_INDICATOR_GLOB)
    if indicator_dir is None:
        raise SystemExit("Could not find indicator run directory.")
    diag_env = os.getenv("RAWSEQ_TARGET_TOURNAMENT_DIAG_DIR", "").strip()
    diag_dir = resolve_path(diag_env) if diag_env else latest_dir(DEFAULT_DIAG_ROOT, "rawseq_feature_diagnostics_*")
    output_root = resolve_path(os.getenv("RAWSEQ_TARGET_TOURNAMENT_OUTPUT_DIR", str(DEFAULT_OUTPUT_ROOT)))
    out_dir = output_root / f"rawseq_target_lane_baseline_tournament_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    target_lanes = parse_csv_strings(
        os.getenv("RAWSEQ_TARGET_TOURNAMENT_LANES", ""),
        ["coarse_return_vector", "future_high_from_now_bps_path", "future_low_from_now_bps_path", "future_range_envelope_path"],
    )
    horizons = parse_int_list(os.getenv("RAWSEQ_TARGET_TOURNAMENT_HORIZONS", ""), [60, 120, 240, 480])
    bundles = parse_csv_strings(os.getenv("RAWSEQ_TARGET_TOURNAMENT_BUNDLES", ""), BUNDLE_NAMES)
    block_count = int(float(os.getenv("RAWSEQ_TARGET_TOURNAMENT_VALIDATION_BLOCKS", "4")))
    min_blocks = int(float(os.getenv("RAWSEQ_TARGET_TOURNAMENT_MIN_POSITIVE_BLOCKS", "3")))
    min_effective_rows = int(float(os.getenv("RAWSEQ_TARGET_TOURNAMENT_MIN_EFFECTIVE_ROWS", "100")))
    min_validation_improvement = float(os.getenv("RAWSEQ_TARGET_TOURNAMENT_MIN_VALIDATION_IMPROVEMENT", "0.0"))
    min_viable_bundles = int(float(os.getenv("RAWSEQ_TARGET_TOURNAMENT_MIN_VIABLE_BUNDLES", "2")))
    max_target_shift = float(os.getenv("RAWSEQ_TARGET_TOURNAMENT_MAX_TARGET_SHIFT", "2.0"))
    required_target_lane = os.getenv("RAWSEQ_TARGET_TOURNAMENT_REQUIRED_SELECTED_LANE", "").strip()
    max_features = int(float(os.getenv("RAWSEQ_TARGET_TOURNAMENT_MAX_FEATURES", "0")))
    max_rows = int(float(os.getenv("RAWSEQ_TARGET_TOURNAMENT_MAX_ROWS", "0") or "0"))
    ridge_alphas = parse_float_list(os.getenv("RAWSEQ_TARGET_TOURNAMENT_RIDGE_ALPHAS", ""), [0.01, 0.1, 1.0, 10.0, 100.0])
    elastic_alphas = parse_float_list(os.getenv("RAWSEQ_TARGET_TOURNAMENT_ELASTIC_ALPHAS", ""), [0.001, 0.01, 0.1])
    elastic_l1_ratio = float(os.getenv("RAWSEQ_TARGET_TOURNAMENT_ELASTIC_L1_RATIO", "0.25"))
    elastic_iterations = int(float(os.getenv("RAWSEQ_TARGET_TOURNAMENT_ELASTIC_ITERATIONS", "120")))
    elastic_lr = float(os.getenv("RAWSEQ_TARGET_TOURNAMENT_ELASTIC_LR", "0.02"))
    table = pd.read_csv(indicator_dir / "multi_horizon_training_table.csv", low_memory=False)
    bucket_seconds = infer_bucket_seconds(table, fallback=1.0)
    if max_rows > 0:
        table = table.tail(max_rows).copy().reset_index(drop=True)
    if "split" not in table.columns:
        raise SystemExit("Indicator table must include split column.")
    schema_hash = stable_hash(
        {
            "indicator_dir": str(indicator_dir),
            "diag_dir": str(diag_dir) if diag_dir else "",
            "target_lanes": target_lanes,
            "horizons": horizons,
            "bundles": bundles,
            "selection_rule": "target_lane_gate_v1.0.0",
        }
    )
    version = "1.0.0"
    contract_base = {
        **schema_meta("rawseq_target_lane_baseline_tournament_contract", version, schema_hash),
        "indicator_run_dir": str(indicator_dir),
        "diagnostics_dir": str(diag_dir) if diag_dir else "",
        "output_dir": str(out_dir),
        "bucket_seconds": bucket_seconds,
    }
    print(f"Resolved indicator run: {indicator_dir}")
    print(f"Resolved diagnostics: {diag_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Target lanes: {target_lanes}")
    print(f"Bundles: {bundles}")
    print(f"Horizons: {horizons}")
    payload = run_tournament(
        table,
        diag_dir,
        target_lanes,
        horizons,
        bundles,
        max_features,
        block_count,
        min_blocks,
        min_effective_rows,
        min_validation_improvement,
        max_target_shift,
        min_viable_bundles,
        ridge_alphas,
        elastic_alphas,
        elastic_l1_ratio,
        elastic_iterations,
        elastic_lr,
        contract_base,
        required_target_lane,
    )
    write_outputs(out_dir, payload, version, schema_hash)
    ranking = pd.DataFrame(payload["ranking"])
    bundle_counts = pd.DataFrame(payload["bundle_metrics"])["lane_bundle_status"].value_counts().to_dict() if payload["bundle_metrics"] else {}
    print("Rawseq target-lane baseline tournament complete")
    print(f"Output: {out_dir}")
    print("Lane-bundle status counts:", bundle_counts)
    print("Target-lane ranking:")
    print(ranking[["target_lane", "target_lane_status", "qualifying_feature_bundle_count", "mean_validation_baseline_improvement", "effective_non_overlapping_rows"]].to_string(index=False))
    print("Selected manifest status:", payload["selected_manifest"].get("selection_status"))
    print("Recommended GPU screen status:", payload["recommended_gpu_screen"].get("selection_status"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
