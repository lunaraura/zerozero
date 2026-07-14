#!/usr/bin/env python3
"""Run train/validation-only CPU ablations over locked rawseq feature bundles."""

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
DEFAULT_DIAG_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_feature_diagnostics"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_feature_family_ablation"
DEFAULT_INDICATOR_GLOB = "F:/rsio/rawseq_multi_horizon_indicator_full_contract_smoke/mh_indicator_*"
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def column_hash(columns: list[str]) -> str:
    return stable_hash(list(columns))


def schema_meta(name: str, version: str, schema_hash: str) -> dict[str, Any]:
    return {
        "schema_name": name,
        "schema_version": version,
        "schema_sha256": schema_hash,
        "created_at": now_iso(),
        "generator_path": "scripts/tiny/run_rawseq_feature_family_ablation.py",
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


def parse_float_list(text: str, default: list[float]) -> list[float]:
    raw = text.strip()
    if not raw:
        return default
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_model_list(text: str) -> list[str]:
    raw = text.strip() or "zero,mean,ridge,elastic_net,logistic_direction"
    allowed = {"zero", "mean", "ridge", "elastic_net", "logistic_direction"}
    models = [item.strip() for item in raw.split(",") if item.strip()]
    return [model for model in models if model in allowed]


def to_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


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


def fill_targets(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(y, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return np.where(np.isfinite(y), y, mean), mean


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


def fit_elastic_net(x: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float, iterations: int, learning_rate: float) -> np.ndarray:
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
    for horizon_idx in range(y_train.shape[1]):
        y = (y_train[:, horizon_idx] > 0.0).astype(np.float64)
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
        magnitude = float(np.nanmedian(np.abs(y_train[:, horizon_idx])))
        if not math.isfinite(magnitude) or magnitude <= 0:
            magnitude = 1.0
        out[:, horizon_idx] = (prob - 0.5) * 2.0 * magnitude
    return out


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
    if float(np.std(a)) <= 0 or float(np.std(p)) <= 0:
        return math.nan
    return float(np.corrcoef(a, p)[0, 1])


def directional_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return math.nan
    return float(np.mean((actual[mask] > 0.0) == (pred[mask] > 0.0)))


def policy_metrics(actual: np.ndarray, pred: np.ndarray, policy: str, threshold: float, cost_bps: float, horizon: int) -> dict[str, Any]:
    selected = np.isfinite(actual) & np.isfinite(pred) & (pred > threshold)
    multiplier = -1.0 if policy == "inverse_gt" else 1.0
    gross = multiplier * actual[selected]
    net = gross - cost_bps
    non_overlap = []
    cooldown_until = -1
    selected_indices = np.flatnonzero(selected)
    for idx in selected_indices:
        if idx < cooldown_until:
            continue
        non_overlap.append(multiplier * actual[idx] - cost_bps)
        cooldown_until = idx + max(1, int(horizon))
    non_overlap_arr = np.asarray(non_overlap, dtype=np.float64)
    return {
        "policy": policy,
        "threshold_bps": threshold,
        "cost_bps": cost_bps,
        "selected_rows": int(selected.sum()),
        "avg_net_bps": float(np.mean(net)) if len(net) else math.nan,
        "cum_net_bps": float(np.sum(net)) if len(net) else 0.0,
        "win_rate_net": float(np.mean(net > 0.0)) if len(net) else math.nan,
        "non_overlap_selected_rows": int(len(non_overlap_arr)),
        "non_overlap_cum_net_bps": float(np.sum(non_overlap_arr)) if len(non_overlap_arr) else 0.0,
    }


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
        return {"validation_block_count": 0, "positive_block_fraction": math.nan, "mean_block_rmse_improvement": math.nan, "worst_block_rmse_improvement": math.nan}
    return {
        "validation_block_count": len(improvements),
        "positive_block_fraction": float(np.mean(np.asarray(improvements) > 0.0)),
        "mean_block_rmse_improvement": float(np.mean(improvements)),
        "worst_block_rmse_improvement": float(np.min(improvements)),
    }


def split_frame(table: pd.DataFrame, split: str) -> pd.DataFrame:
    return table[table["split"].astype(str).eq(split)].copy()


def load_bundle(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_bundle(bundle: dict[str, Any], diagnostics: pd.DataFrame) -> None:
    actions = diagnostics.set_index("materialized_feature_name")["recommended_action"].to_dict()
    missing = [feature for feature in bundle["ordered_feature_columns"] if feature not in actions]
    invalid = [feature for feature in bundle["ordered_feature_columns"] if actions.get(feature) in {"invalid", "unresolved"}]
    if missing:
        raise ValueError(f"Bundle {bundle.get('bundle_name')} has unregistered diagnostics columns: {missing[:5]}")
    if invalid:
        raise ValueError(f"Bundle {bundle.get('bundle_name')} includes invalid/unresolved columns: {invalid[:5]}")


def build_ablation_units(diagnostics: pd.DataFrame, bundles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = diagnostics[~diagnostics["recommended_action"].isin(["invalid", "unresolved"])].copy()
    rows = []
    for name in ["minimal_core", "balanced_research", "full_registered"]:
        cols = list(bundles[name]["ordered_feature_columns"])
        rows.append({"ablation_unit": name, "unit_type": "locked_bundle", "feature_count": len(cols), "feature_columns": ";".join(cols), "feature_columns_sha256": column_hash(cols)})
    raw = eligible[eligible["feature_family"].eq("raw")]["materialized_feature_name"].astype(str).tolist()
    families = ["trend", "momentum", "volatility", "breakout", "volume", "order_book", "regime", "cross_market"]
    for family in families:
        cols = sorted(set(raw + eligible[eligible["feature_family"].eq(family)]["materialized_feature_name"].astype(str).tolist()))
        rows.append({"ablation_unit": f"raw_plus_{family}", "unit_type": "raw_plus_family", "feature_count": len(cols), "feature_columns": ";".join(cols), "feature_columns_sha256": column_hash(cols)})
    all_cols = eligible.sort_values("materialized_feature_name")["materialized_feature_name"].astype(str).tolist()
    rows.append({"ablation_unit": "all_registered_features", "unit_type": "all_registered", "feature_count": len(all_cols), "feature_columns": ";".join(all_cols), "feature_columns_sha256": column_hash(all_cols)})
    for family in sorted(eligible["feature_family"].dropna().astype(str).unique()):
        cols = eligible[~eligible["feature_family"].astype(str).eq(family)].sort_values("materialized_feature_name")["materialized_feature_name"].astype(str).tolist()
        rows.append({"ablation_unit": f"all_minus_{family}", "unit_type": "all_minus_one_family", "feature_count": len(cols), "feature_columns": ";".join(cols), "feature_columns_sha256": column_hash(cols)})
    return rows


def train_models(
    table: pd.DataFrame,
    unit: dict[str, Any],
    target_columns: list[str],
    models: list[str],
    ridge_alphas: list[float],
    elastic_alphas: list[float],
    elastic_l1_ratio: float,
    elastic_iterations: int,
    elastic_lr: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    features = [item for item in str(unit["feature_columns"]).split(";") if item]
    train = split_frame(table, "train")
    validation = split_frame(table, "validation")
    y_train_raw = to_matrix(train, target_columns)
    y_val = to_matrix(validation, target_columns)
    y_train, y_mean = fill_targets(y_train_raw)
    all_pred: dict[str, np.ndarray] = {}
    manifest_rows = []
    split_mask_train = table["split"].astype(str).eq("train").to_numpy()
    split_mask_val = table["split"].astype(str).eq("validation").to_numpy()
    if "zero" in models:
        name = f"{unit['ablation_unit']}__zero_return_baseline"
        all_pred[name] = np.zeros((len(table), len(target_columns)), dtype=np.float64)
        manifest_rows.append({"model": name, "base_model": "zero_return_baseline", "ablation_unit": unit["ablation_unit"], "status": "ok", "selection_stage": "none"})
    if "mean" in models:
        name = f"{unit['ablation_unit']}__training_mean_return_baseline"
        all_pred[name] = np.tile(y_mean, (len(table), 1))
        manifest_rows.append({"model": name, "base_model": "training_mean_return_baseline", "ablation_unit": unit["ablation_unit"], "status": "ok", "selection_stage": "train_fit_constant"})
    if not features:
        return manifest_rows, all_pred
    pre = fit_preprocessor(train, features)
    x_all = transform(table, features, pre)
    x_train = x_all[split_mask_train]
    x_val = x_all[split_mask_val]
    if "ridge" in models:
        candidates = []
        for alpha in ridge_alphas:
            coef = fit_ridge(x_train, y_train, alpha)
            pred_val = predict_linear(x_val, coef)
            candidates.append((rmse(y_val, pred_val), alpha, coef))
        score, alpha, coef = sorted(candidates, key=lambda item: item[0] if math.isfinite(item[0]) else 1e18)[0]
        name = f"{unit['ablation_unit']}__ridge_multi_output"
        all_pred[name] = predict_linear(x_all, coef)
        manifest_rows.append({"model": name, "base_model": "ridge_multi_output", "ablation_unit": unit["ablation_unit"], "status": "ok", "selection_stage": "validation_selected_alpha", "selected_alpha": alpha, "validation_combined_rmse": score})
    if "elastic_net" in models:
        candidates = []
        for alpha in elastic_alphas:
            coef = fit_elastic_net(x_train, y_train, alpha, elastic_l1_ratio, elastic_iterations, elastic_lr)
            pred_val = predict_linear(x_val, coef)
            candidates.append((rmse(y_val, pred_val), alpha, coef))
        score, alpha, coef = sorted(candidates, key=lambda item: item[0] if math.isfinite(item[0]) else 1e18)[0]
        name = f"{unit['ablation_unit']}__elastic_net_multi_output"
        all_pred[name] = predict_linear(x_all, coef)
        manifest_rows.append({"model": name, "base_model": "elastic_net_multi_output", "ablation_unit": unit["ablation_unit"], "status": "ok", "selection_stage": "validation_selected_alpha", "selected_alpha": alpha, "l1_ratio": elastic_l1_ratio, "validation_combined_rmse": score})
    if "logistic_direction" in models:
        name = f"{unit['ablation_unit']}__logistic_direction_model"
        try:
            all_pred[name] = fit_logistic_direction(x_train, y_train, x_all)
            manifest_rows.append({"model": name, "base_model": "logistic_direction_model", "ablation_unit": unit["ablation_unit"], "status": "ok", "selection_stage": "train_fit_direction"})
        except Exception as exc:
            manifest_rows.append({"model": name, "base_model": "logistic_direction_model", "ablation_unit": unit["ablation_unit"], "status": "failed", "failure": str(exc)})
    return manifest_rows, all_pred


def metric_rows_for_predictions(
    table: pd.DataFrame,
    target_columns: list[str],
    predictions: dict[str, np.ndarray],
    horizons: list[int],
    block_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    block_rows = []
    train_mask = table["split"].astype(str).eq("train").to_numpy()
    val_mask = table["split"].astype(str).eq("validation").to_numpy()
    y_train = to_matrix(table[train_mask], target_columns)
    y_val = to_matrix(table[val_mask], target_columns)
    mean_name = next((name for name in predictions if name.endswith("__training_mean_return_baseline")), "")
    mean_pred_val = predictions[mean_name][val_mask] if mean_name else np.tile(np.nanmean(y_train, axis=0), (len(y_val), 1))
    for model, pred in predictions.items():
        for h_idx, target in enumerate(target_columns):
            train_actual = y_train[:, h_idx]
            val_actual = y_val[:, h_idx]
            train_pred = pred[train_mask, h_idx]
            val_pred = pred[val_mask, h_idx]
            val_rmse = rmse(val_actual, val_pred)
            mean_rmse = rmse(val_actual, mean_pred_val[:, h_idx])
            zero_rmse = rmse(val_actual, np.zeros_like(val_actual))
            improvement_mean = (mean_rmse - val_rmse) / mean_rmse if math.isfinite(mean_rmse) and mean_rmse > 0 else math.nan
            improvement_zero = (zero_rmse - val_rmse) / zero_rmse if math.isfinite(zero_rmse) and zero_rmse > 0 else math.nan
            block = validation_block_scores(val_actual, val_pred, mean_pred_val[:, h_idx], block_count)
            rows.append(
                {
                    "model": model,
                    "ablation_unit": model.split("__", 1)[0],
                    "base_model": model.split("__", 1)[1] if "__" in model else model,
                    "target_column": target,
                    "horizon_buckets": horizons[h_idx],
                    "train_rmse": rmse(train_actual, train_pred),
                    "validation_rmse": val_rmse,
                    "train_mae": mae(train_actual, train_pred),
                    "validation_mae": mae(val_actual, val_pred),
                    "train_correlation": corr(train_actual, train_pred),
                    "validation_correlation": corr(val_actual, val_pred),
                    "train_directional_accuracy": directional_accuracy(train_actual, train_pred),
                    "validation_directional_accuracy": directional_accuracy(val_actual, val_pred),
                    "validation_rmse_improvement_vs_mean": improvement_mean,
                    "validation_rmse_improvement_vs_zero": improvement_zero,
                    "holdout_used": False,
                    **block,
                }
            )
            bounds = np.linspace(0, len(val_actual), block_count + 1, dtype=int)
            for b_idx in range(block_count):
                sl = slice(bounds[b_idx], bounds[b_idx + 1])
                model_rmse = rmse(val_actual[sl], val_pred[sl])
                base_rmse = rmse(val_actual[sl], mean_pred_val[sl, h_idx])
                block_rows.append(
                    {
                        "model": model,
                        "ablation_unit": model.split("__", 1)[0],
                        "base_model": model.split("__", 1)[1] if "__" in model else model,
                        "target_column": target,
                        "horizon_buckets": horizons[h_idx],
                        "validation_block_index": b_idx,
                        "block_rows": int(bounds[b_idx + 1] - bounds[b_idx]),
                        "block_rmse": model_rmse,
                        "mean_baseline_block_rmse": base_rmse,
                        "block_rmse_improvement_vs_mean": (base_rmse - model_rmse) / base_rmse if math.isfinite(base_rmse) and base_rmse > 0 else math.nan,
                        "holdout_used": False,
                    }
                )
    return rows, block_rows


def policy_rows_for_predictions(
    table: pd.DataFrame,
    target_columns: list[str],
    predictions: dict[str, np.ndarray],
    horizons: list[int],
    thresholds: list[float],
    costs: list[float],
) -> list[dict[str, Any]]:
    rows = []
    val_mask = table["split"].astype(str).eq("validation").to_numpy()
    y_val = to_matrix(table[val_mask], target_columns)
    for model, pred in predictions.items():
        val_pred = pred[val_mask]
        for h_idx, target in enumerate(target_columns):
            for policy in ["direct_gt", "inverse_gt"]:
                for threshold in thresholds:
                    for cost in costs:
                        rows.append(
                            {
                                "model": model,
                                "ablation_unit": model.split("__", 1)[0],
                                "base_model": model.split("__", 1)[1] if "__" in model else model,
                                "target_column": target,
                                "horizon_buckets": horizons[h_idx],
                                "holdout_used": False,
                                **policy_metrics(y_val[:, h_idx], val_pred[:, h_idx], policy, threshold, cost, horizons[h_idx]),
                            }
                        )
    return rows


def holdout_reference_rows_for_predictions(
    table: pd.DataFrame,
    target_columns: list[str],
    predictions: dict[str, np.ndarray],
    horizons: list[int],
) -> list[dict[str, Any]]:
    rows = []
    holdout_mask = table["split"].astype(str).eq("untouched_holdout").to_numpy()
    if not holdout_mask.any():
        return rows
    y_holdout = to_matrix(table[holdout_mask], target_columns)
    for model, pred in predictions.items():
        holdout_pred = pred[holdout_mask]
        for h_idx, target in enumerate(target_columns):
            actual = y_holdout[:, h_idx]
            predicted = holdout_pred[:, h_idx]
            rows.append(
                {
                    "model": model,
                    "ablation_unit": model.split("__", 1)[0],
                    "base_model": model.split("__", 1)[1] if "__" in model else model,
                    "target_column": target,
                    "horizon_buckets": horizons[h_idx],
                    "holdout_rows": int((np.isfinite(actual) & np.isfinite(predicted)).sum()),
                    "holdout_rmse": rmse(actual, predicted),
                    "holdout_mae": mae(actual, predicted),
                    "holdout_correlation": corr(actual, predicted),
                    "holdout_directional_accuracy": directional_accuracy(actual, predicted),
                    "holdout_position_cum_net_bps": math.nan,
                    "selection_stage": "train_validation_selected_then_holdout_evaluated",
                    "holdout_used_for_selection": False,
                    "holdout_used": True,
                }
            )
    return rows


def select_horizons(metrics: pd.DataFrame, policy: pd.DataFrame, min_improvement: float, min_positive_block_fraction: float, min_directional_accuracy: float) -> list[dict[str, Any]]:
    rows = []
    non_constant = metrics[~metrics["base_model"].isin(["zero_return_baseline", "training_mean_return_baseline"])].copy()
    for horizon, group in non_constant.groupby("horizon_buckets"):
        group = group.sort_values(
            ["validation_rmse_improvement_vs_mean", "positive_block_fraction", "validation_directional_accuracy", "validation_correlation"],
            ascending=[False, False, False, False],
        )
        best = group.iloc[0].to_dict() if not group.empty else {}
        policy_group = policy[policy["horizon_buckets"].astype(int).eq(int(horizon))].copy()
        best_policy = policy_group.sort_values(["non_overlap_cum_net_bps", "cum_net_bps", "non_overlap_selected_rows"], ascending=[False, False, False]).head(1)
        best_policy_row = best_policy.iloc[0].to_dict() if not best_policy.empty else {}
        improvement = float(best.get("validation_rmse_improvement_vs_mean", math.nan))
        block_fraction = float(best.get("positive_block_fraction", math.nan))
        direction = float(best.get("validation_directional_accuracy", math.nan))
        if improvement >= min_improvement and block_fraction >= min_positive_block_fraction and direction >= min_directional_accuracy:
            status = "useful_horizon_candidate"
        elif improvement > 0:
            status = "research_horizon_unstable"
        else:
            status = "noisy_or_baseline_only"
        rows.append(
            {
                "horizon_buckets": int(horizon),
                "target_column": best.get("target_column", ""),
                "horizon_status": status,
                "best_validation_model": best.get("model", ""),
                "best_ablation_unit": best.get("ablation_unit", ""),
                "best_base_model": best.get("base_model", ""),
                "validation_rmse": best.get("validation_rmse", math.nan),
                "validation_rmse_improvement_vs_mean": improvement,
                "positive_block_fraction": block_fraction,
                "validation_directional_accuracy": direction,
                "validation_correlation": best.get("validation_correlation", math.nan),
                "best_policy_model": best_policy_row.get("model", ""),
                "best_policy": best_policy_row.get("policy", ""),
                "best_policy_threshold_bps": best_policy_row.get("threshold_bps", math.nan),
                "best_policy_cost_bps": best_policy_row.get("cost_bps", math.nan),
                "best_policy_non_overlap_selected_rows": best_policy_row.get("non_overlap_selected_rows", math.nan),
                "best_policy_non_overlap_cum_net_bps": best_policy_row.get("non_overlap_cum_net_bps", math.nan),
                "holdout_used": False,
            }
        )
    return rows


def main() -> int:
    diag_env = os.getenv("RAWSEQ_ABLATION_DIAG_DIR", "").strip()
    diag_dir = resolve_path(diag_env) if diag_env else latest_dir(DEFAULT_DIAG_ROOT, "rawseq_feature_diagnostics_*")
    if diag_dir is None:
        raise SystemExit("Could not find feature diagnostics directory.")
    indicator_env = os.getenv("RAWSEQ_ABLATION_INDICATOR_RUN_DIR", "").strip()
    indicator_dir = resolve_path(indicator_env) if indicator_env else latest_dir(DEFAULT_INDICATOR_GLOB)
    if indicator_dir is None:
        raise SystemExit("Could not find indicator run directory.")
    output_root = resolve_path(os.getenv("RAWSEQ_ABLATION_OUTPUT_DIR", str(DEFAULT_OUTPUT_ROOT)))
    out_dir = output_root / f"rawseq_feature_family_ablation_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    models = parse_model_list(os.getenv("RAWSEQ_ABLATION_MODELS", ""))
    ridge_alphas = parse_float_list(os.getenv("RAWSEQ_ABLATION_RIDGE_ALPHAS", ""), [0.01, 0.1, 1.0, 10.0, 100.0])
    elastic_alphas = parse_float_list(os.getenv("RAWSEQ_ABLATION_ELASTIC_ALPHAS", ""), [0.001, 0.01, 0.1])
    elastic_l1_ratio = float(os.getenv("RAWSEQ_ABLATION_ELASTIC_L1_RATIO", "0.25"))
    elastic_iterations = int(float(os.getenv("RAWSEQ_ABLATION_ELASTIC_ITERATIONS", "120")))
    elastic_lr = float(os.getenv("RAWSEQ_ABLATION_ELASTIC_LR", "0.02"))
    block_count = int(float(os.getenv("RAWSEQ_ABLATION_VALIDATION_BLOCKS", "4")))
    policy_thresholds = parse_float_list(os.getenv("RAWSEQ_ABLATION_POLICY_THRESHOLDS_BPS", ""), [0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    policy_costs = parse_float_list(os.getenv("RAWSEQ_ABLATION_POLICY_COSTS_BPS", ""), [0.1])
    min_improvement = float(os.getenv("RAWSEQ_ABLATION_MIN_RMSE_IMPROVEMENT", "0.0"))
    min_block_fraction = float(os.getenv("RAWSEQ_ABLATION_MIN_POSITIVE_BLOCK_FRACTION", "0.5"))
    min_direction = float(os.getenv("RAWSEQ_ABLATION_MIN_DIRECTIONAL_ACCURACY", "0.5"))

    diagnostics = pd.read_csv(diag_dir / "feature_diagnostics.csv")
    bundles = {name: load_bundle(diag_dir / f"feature_bundle_{name}.json") for name in ["minimal_core", "balanced_research", "full_registered"]}
    for bundle in bundles.values():
        validate_bundle(bundle, diagnostics)
        write_json(out_dir / f"locked_feature_bundle_{bundle['bundle_name']}.json", bundle)
    table = pd.read_csv(indicator_dir / "multi_horizon_training_table.csv")
    target_manifest = pd.read_csv(indicator_dir / "target_manifest.csv")
    target_columns = [str(x) for x in target_manifest["target_column"].astype(str) if str(x) in table.columns]
    horizons = [int(float(x)) for x in target_manifest[target_manifest["target_column"].isin(target_columns)]["horizon_buckets"]]
    units = build_ablation_units(diagnostics, bundles)

    schema_hash = stable_hash(
        {
            "diag_dir": str(diag_dir),
            "indicator_dir": str(indicator_dir),
            "bundle_hashes": {k: v.get("feature_columns_sha256") for k, v in bundles.items()},
            "models": models,
            "ridge_alphas": ridge_alphas,
            "elastic_alphas": elastic_alphas,
        }
    )
    version = "1.0.0"
    all_manifest: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    all_blocks: list[dict[str, Any]] = []
    all_policy: list[dict[str, Any]] = []
    all_holdout_reference: list[dict[str, Any]] = []
    print(f"Resolved diagnostics: {diag_dir}")
    print(f"Resolved indicator run: {indicator_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Ablation units: {len(units)}")
    for unit in units:
        if int(unit["feature_count"]) == 0:
            continue
        manifest, predictions = train_models(
            table,
            unit,
            target_columns,
            models,
            ridge_alphas,
            elastic_alphas,
            elastic_l1_ratio,
            elastic_iterations,
            elastic_lr,
        )
        for row in manifest:
            row.update({"feature_count": int(unit["feature_count"]), "feature_columns_sha256": unit["feature_columns_sha256"], "holdout_used": False})
        metrics, blocks = metric_rows_for_predictions(table, target_columns, predictions, horizons, block_count)
        policy_rows = policy_rows_for_predictions(table, target_columns, predictions, horizons, policy_thresholds, policy_costs)
        holdout_reference = holdout_reference_rows_for_predictions(table, target_columns, predictions, horizons)
        all_manifest.extend(manifest)
        all_metrics.extend(metrics)
        all_blocks.extend(blocks)
        all_policy.extend(policy_rows)
        all_holdout_reference.extend(holdout_reference)
        print(f"  {unit['ablation_unit']}: features={unit['feature_count']} models={len(predictions)}")

    metric_frame = pd.DataFrame(all_metrics)
    policy_frame = pd.DataFrame(all_policy)
    horizon_rows = select_horizons(metric_frame, policy_frame, min_improvement, min_block_fraction, min_direction)
    write_csv(out_dir / "locked_feature_bundles.csv", with_meta(
        [
            {
                "bundle_name": name,
                "feature_count": bundles[name]["feature_count"],
                "feature_columns_sha256": bundles[name]["feature_columns_sha256"],
                "source_bundle_path": str(diag_dir / f"feature_bundle_{name}.json"),
                "invalid_feature_count": 0,
                "unresolved_feature_count": 0,
                "holdout_used": False,
            }
            for name in bundles
        ],
        "rawseq_locked_feature_bundles",
        version,
        schema_hash,
    ))
    write_csv(out_dir / "ablation_units.csv", with_meta(units, "rawseq_feature_ablation_units", version, schema_hash))
    write_csv(out_dir / "ablation_model_manifest.csv", with_meta(all_manifest, "rawseq_feature_ablation_model_manifest", version, schema_hash))
    write_csv(out_dir / "ablation_metrics.csv", with_meta(all_metrics, "rawseq_feature_ablation_metrics", version, schema_hash))
    write_csv(out_dir / "ablation_validation_block_metrics.csv", with_meta(all_blocks, "rawseq_feature_ablation_validation_blocks", version, schema_hash))
    write_csv(out_dir / "ablation_policy_metrics.csv", with_meta(all_policy, "rawseq_feature_ablation_policy_metrics", version, schema_hash))
    write_csv(out_dir / "horizon_decisions.csv", with_meta(horizon_rows, "rawseq_feature_ablation_horizon_decisions", version, schema_hash))
    holdout_reference_with_meta = with_meta(
        all_holdout_reference,
        "rawseq_feature_ablation_holdout_reference",
        version,
        schema_hash,
    )
    write_csv(out_dir / "cpu_baseline_holdout_reference.csv", holdout_reference_with_meta)
    write_csv(out_dir / "combined_leaderboard.csv", holdout_reference_with_meta)
    useful = [row for row in horizon_rows if row["horizon_status"] == "useful_horizon_candidate"]
    write_json(out_dir / "selected_horizons.json", {
        "selected_horizons": [row["horizon_buckets"] for row in useful],
        "horizon_rows": horizon_rows,
        "selection_stage": "validation_only_cpu_ablation",
        "untouched_holdout_used": False,
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    })
    contract = {
        **schema_meta("rawseq_feature_family_ablation_contract", version, schema_hash),
        "diagnostics_dir": str(diag_dir),
        "indicator_run_dir": str(indicator_dir),
        "output_dir": str(out_dir),
        "models": models,
        "ablation_unit_count": len(units),
        "target_columns": target_columns,
        "horizon_buckets": horizons,
        "train_rows": int(table["split"].astype(str).eq("train").sum()),
        "validation_rows": int(table["split"].astype(str).eq("validation").sum()),
        "holdout_rows_used": 0,
        "holdout_reference_rows": int(len(all_holdout_reference)),
        "untouched_holdout_used": False,
        "useful_horizons": [row["horizon_buckets"] for row in useful],
        "safety": {"paper_only": True, "orders": False, "promotion": False, "champion_mutation": False, "training_torch": False, "ensemble_search": False},
    }
    write_json(out_dir / "feature_family_ablation_contract.json", contract)
    summary = [
        "# Rawseq Feature-Family CPU Ablation",
        "",
        f"Created at: {contract['created_at']}",
        f"Diagnostics: `{diag_dir}`",
        f"Indicator run: `{indicator_dir}`",
        f"Output: `{out_dir}`",
        "",
        "## Counts",
        f"- ablation units: {len(units)}",
        f"- models requested: {models}",
        f"- train rows used: {contract['train_rows']}",
        f"- validation rows used: {contract['validation_rows']}",
        "- untouched holdout rows used: 0",
        f"- holdout reference rows written: {contract['holdout_reference_rows']} (not used for selection)",
        f"- metric rows: {len(all_metrics)}",
        f"- policy rows: {len(all_policy)}",
        "",
        "## Horizon Decisions",
    ]
    for row in horizon_rows:
        summary.append(
            f"- h{row['horizon_buckets']}: {row['horizon_status']} "
            f"best={row['best_validation_model']} "
            f"improvement={row['validation_rmse_improvement_vs_mean']:.6f} "
            f"positive_blocks={row['positive_block_fraction']:.3f}"
        )
    summary.extend(
        [
            "",
            "Safety: CPU-only baseline/ablation. No Torch sequence training, no ensemble search, no holdout selection, no freeze, no promotion, no champion mutation, no orders.",
        ]
    )
    (out_dir / "feature_family_ablation_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("Rawseq feature-family CPU ablation complete")
    print(f"Output: {out_dir}")
    print(f"Useful horizons: {[row['horizon_buckets'] for row in useful]}")
    print("Safety: no Torch training. No ensemble search. No holdout selection. No orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
