#!/usr/bin/env python3
"""Train and evaluate canonical rawseq baseline models.

Baselines are fitted only on the canonical train split. Validation is used for
policy/threshold selection. Untouched holdout is evaluated after selection.

Safety: paper-only research. No private API, no orders, no promotion, and no
champion mutation.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rawseq_policy_scoring import expectancy_metrics, max_dip_bps, score_policy_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_canonical_tables"
OUTPUT_ROOT = Path(
    os.getenv(
        "RAWSEQ_BASELINE_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_canonical_baselines"),
    )
).expanduser()
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

TABLE_PATH_ENV = os.getenv("RAWSEQ_CANONICAL_TABLE_PATH", "").strip()
MANIFEST_PATH_ENV = os.getenv("RAWSEQ_CANONICAL_MANIFEST_PATH", "").strip()
MAX_ROWS_ENV = os.getenv("RAWSEQ_BASELINE_MAX_ROWS", "").strip()
THRESHOLDS = [
    float(item.strip())
    for item in os.getenv("RAWSEQ_BASELINE_THRESHOLDS_BPS", "0,0.1,0.2,0.5,1").split(",")
    if item.strip()
]
COSTS = [
    float(item.strip())
    for item in os.getenv("RAWSEQ_BASELINE_COSTS_BPS", "0.1,1,5").split(",")
    if item.strip()
]
DECISION_COST_BPS = float(os.getenv("RAWSEQ_BASELINE_DECISION_COST_BPS", "0.1"))
RIDGE_ALPHAS = [
    float(item.strip())
    for item in os.getenv("RAWSEQ_BASELINE_RIDGE_ALPHAS", "0.01,0.1,1,10,100").split(",")
    if item.strip()
]
LOGISTIC_ITERATIONS = int(float(os.getenv("RAWSEQ_BASELINE_LOGISTIC_ITERATIONS", "250")))
LOGISTIC_LR = float(os.getenv("RAWSEQ_BASELINE_LOGISTIC_LR", "0.05"))
LOGISTIC_L2 = float(os.getenv("RAWSEQ_BASELINE_LOGISTIC_L2", "0.1"))
BARRIER_BPS = float(os.getenv("RAWSEQ_BASELINE_BARRIER_BPS", "5"))
BOOTSTRAP_SAMPLES = int(float(os.getenv("RAWSEQ_BASELINE_BOOTSTRAP_SAMPLES", "300")))
BOOTSTRAP_SEED = int(float(os.getenv("RAWSEQ_BASELINE_BOOTSTRAP_SEED", "1729")))
MIN_POSITION_TRADES = int(float(os.getenv("RAWSEQ_BASELINE_MIN_POSITION_TRADES", "30")))

TARGET_COL = "gross_future_return_bps"
PRED_COL = "baseline_pred_bps"
ACTUAL_COL = "actual_return_bps"


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_latest_canonical_dir() -> Path:
    if not CANONICAL_ROOT.exists():
        raise SystemExit(f"Canonical root does not exist: {CANONICAL_ROOT}")
    candidates = [
        path
        for path in CANONICAL_ROOT.iterdir()
        if path.is_dir()
        and (path / "canonical_training_table.csv").exists()
        and (path / "split_manifest.json").exists()
    ]
    if not candidates:
        raise SystemExit(f"No canonical table folders found under {CANONICAL_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_inputs() -> tuple[Path, Path, Path]:
    if TABLE_PATH_ENV:
        table_path = Path(TABLE_PATH_ENV).expanduser()
        if not table_path.is_absolute():
            table_path = PROJECT_ROOT / table_path
        base_dir = table_path.parent
    else:
        base_dir = resolve_latest_canonical_dir()
        table_path = base_dir / "canonical_training_table.csv"

    if MANIFEST_PATH_ENV:
        manifest_path = Path(MANIFEST_PATH_ENV).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
    else:
        manifest_path = base_dir / "split_manifest.json"
    if not table_path.exists():
        raise SystemExit(f"Canonical table not found: {table_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Canonical manifest not found: {manifest_path}")
    return base_dir, table_path, manifest_path


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def feature_columns_from_manifest(manifest: dict[str, Any], frame: pd.DataFrame) -> list[str]:
    schema = manifest.get("feature_schema") if isinstance(manifest.get("feature_schema"), dict) else {}
    features = [str(item) for item in schema.get("features", []) if str(item) in frame.columns]
    missing = [f"{column}_missing" for column in features if f"{column}_missing" in frame.columns]
    columns = features + missing
    if columns:
        return columns
    excluded = {
        "decision_timestamp",
        "label_end_timestamp",
        "price",
        TARGET_COL,
        "future_market_return_bps_horizon",
        "mfe_bps",
        "mae_bps",
        "split",
        "source_max_timestamp",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
        and not column.endswith("_timestamp")
    ]


def split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return frame[frame["split"].astype(str).eq(split)].copy()


def fit_preprocessor(train: pd.DataFrame, feature_columns: list[str]) -> dict[str, np.ndarray]:
    raw = train[feature_columns].copy()
    for column in raw.columns:
        if raw[column].dtype == bool:
            raw[column] = raw[column].astype(float)
    values = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    means = np.nanmean(values, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    filled = np.where(np.isfinite(values), values, means)
    std = np.nanstd(filled, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return {"mean": means, "std": std}


def transform_features(frame: pd.DataFrame, feature_columns: list[str], preprocessor: dict[str, np.ndarray]) -> np.ndarray:
    raw = frame[feature_columns].copy()
    for column in raw.columns:
        if raw[column].dtype == bool:
            raw[column] = raw[column].astype(float)
    values = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    values = np.where(np.isfinite(values), values, preprocessor["mean"])
    out = (values - preprocessor["mean"]) / preprocessor["std"]
    out[~np.isfinite(out)] = 0.0
    return out


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(X), dtype=np.float64), X])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def predict_linear(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(X), dtype=np.float64), X])
    return design @ coef


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2))) if mask.any() else math.nan


def fit_logistic(X: np.ndarray, y: np.ndarray, iterations: int, lr: float, l2: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if len(np.unique(y[np.isfinite(y)])) < 2:
        base = np.clip(np.nanmean(y), 1e-6, 1 - 1e-6)
        coef = np.zeros(X.shape[1] + 1, dtype=np.float64)
        coef[0] = math.log(base / (1.0 - base))
        return coef
    design = np.column_stack([np.ones(len(X), dtype=np.float64), X])
    coef = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(iterations):
        pred = sigmoid(design @ coef)
        grad = design.T @ (pred - y) / max(1, len(y))
        reg = l2 * coef / max(1, len(y))
        reg[0] = 0.0
        coef -= lr * (grad + reg)
    return coef


def predict_logistic(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(X), dtype=np.float64), X])
    return sigmoid(design @ coef)


def correlation(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() < 3:
        return math.nan
    actual = actual[mask]
    pred = pred[mask]
    if np.std(actual) <= 1e-12 or np.std(pred) <= 1e-12:
        return math.nan
    return float(np.corrcoef(actual, pred)[0, 1])


def binary_precision_recall(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    return precision, recall


def balanced_accuracy(actual_positive: np.ndarray, predicted_positive: np.ndarray) -> float:
    actual_positive = np.asarray(actual_positive, dtype=bool)
    predicted_positive = np.asarray(predicted_positive, dtype=bool)
    positives = actual_positive
    negatives = ~actual_positive
    tpr = np.mean(predicted_positive[positives]) if positives.any() else math.nan
    tnr = np.mean(~predicted_positive[negatives]) if negatives.any() else math.nan
    if math.isfinite(tpr) and math.isfinite(tnr):
        return float(0.5 * (tpr + tnr))
    return math.nan


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2 or BOOTSTRAP_SAMPLES <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def prediction_metrics(frame: pd.DataFrame, model_name: str, split: str, pred_col: str = PRED_COL) -> dict[str, Any]:
    actual = pd.to_numeric(frame[ACTUAL_COL], errors="coerce").to_numpy(dtype=np.float64)
    pred = pd.to_numeric(frame[pred_col], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(actual) & np.isfinite(pred)
    actual_f = actual[finite]
    pred_f = pred[finite]
    abs_err = np.abs(actual_f - pred_f)
    direction_actual = actual_f > 0.0
    direction_pred = pred_f > 0.0
    up_precision, up_recall = binary_precision_recall(frame["mfe_bps"].to_numpy(dtype=np.float64) >= BARRIER_BPS, pred >= BARRIER_BPS)
    down_precision, down_recall = binary_precision_recall(frame["mae_bps"].to_numpy(dtype=np.float64) <= -BARRIER_BPS, pred <= -BARRIER_BPS)
    return {
        "model": model_name,
        "split": split,
        "rows": int(len(actual_f)),
        "mae_bps": float(np.mean(abs_err)) if len(abs_err) else math.nan,
        "rmse_bps": float(np.sqrt(np.mean((actual_f - pred_f) ** 2))) if len(actual_f) else math.nan,
        "return_correlation": correlation(actual_f, pred_f),
        "directional_accuracy": float(np.mean(direction_actual == direction_pred)) if len(actual_f) else math.nan,
        "balanced_accuracy": balanced_accuracy(direction_actual, direction_pred),
        "up_barrier_bps": BARRIER_BPS,
        "up_barrier_precision": up_precision,
        "up_barrier_recall": up_recall,
        "down_barrier_bps": BARRIER_BPS,
        "down_barrier_precision": down_precision,
        "down_barrier_recall": down_recall,
    }


def selected_non_overlapping(scored: pd.DataFrame, horizon_ms: int) -> np.ndarray:
    values = []
    next_allowed = -math.inf
    for _, row in scored.sort_values("decision_timestamp").iterrows():
        timestamp = safe_float(row.get("decision_timestamp"))
        if not math.isfinite(timestamp) or timestamp < next_allowed:
            continue
        net = safe_float(row.get("net_bps"))
        if math.isfinite(net):
            values.append(net)
            next_allowed = timestamp + horizon_ms
    return np.asarray(values, dtype=np.float64)


def position_values(scored: pd.DataFrame, horizon_ms: int) -> tuple[np.ndarray, float]:
    values = selected_non_overlapping(scored, horizon_ms)
    if scored.empty:
        return values, math.nan
    span = safe_float(scored["decision_timestamp"].max()) - safe_float(scored["decision_timestamp"].min())
    exposure = min(len(values) * horizon_ms / max(span, 1.0), 1.0) if math.isfinite(span) else math.nan
    return values, exposure


def policy_metric_rows(predictions: pd.DataFrame, model_name: str, split: str, horizon_ms: int, allowed_policies: list[str]) -> list[dict[str, Any]]:
    rows = []
    for policy in allowed_policies:
        for threshold in THRESHOLDS:
            for cost in COSTS:
                scored = score_policy_frame(
                    predictions,
                    PRED_COL,
                    ACTUAL_COL,
                    policy,
                    threshold,
                    cost_bps=cost,
                    selected_only=True,
                )
                row_values = scored["net_bps"].to_numpy(dtype=np.float64) if not scored.empty else np.asarray([])
                row_metrics = expectancy_metrics(row_values)
                non_overlap_values = selected_non_overlapping(scored, horizon_ms)
                non_overlap = expectancy_metrics(non_overlap_values)
                position_net, exposure = position_values(scored, horizon_ms)
                position = expectancy_metrics(position_net)
                ci_low, ci_high = bootstrap_ci(position_net)
                rows.append(
                    {
                        "model": model_name,
                        "split": split,
                        "policy": policy,
                        "threshold_bps": threshold,
                        "cost_bps": cost,
                        "row_signal_selected_rows": int(row_metrics["rows"]),
                        "row_signal_avg_net_bps": row_metrics["avg_net_bps"],
                        "row_signal_cum_net_bps": row_metrics["cum_net_bps"],
                        "row_signal_win_rate_net": row_metrics["win_rate_net"],
                        "row_signal_max_dip_net_bps": row_metrics["max_dip_net_bps"],
                        "non_overlapping_selected_rows": int(non_overlap["rows"]),
                        "non_overlapping_avg_net_bps": non_overlap["avg_net_bps"],
                        "non_overlapping_cum_net_bps": non_overlap["cum_net_bps"],
                        "non_overlapping_win_rate_net": non_overlap["win_rate_net"],
                        "non_overlapping_max_dip_net_bps": non_overlap["max_dip_net_bps"],
                        "position_trade_count": int(position["rows"]),
                        "position_avg_net_bps": position["avg_net_bps"],
                        "position_cum_net_bps": position["cum_net_bps"],
                        "position_win_rate_net": position["win_rate_net"],
                        "position_max_dip_net_bps": position["max_dip_net_bps"],
                        "position_payoff_ratio": position["payoff_ratio"],
                        "position_avg_net_ci95_low_bps": ci_low,
                        "position_avg_net_ci95_high_bps": ci_high,
                        "position_exposure_fraction": exposure,
                    }
                )
    return rows


def load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if MAX_ROWS_ENV:
        max_rows = int(float(MAX_ROWS_ENV))
        if max_rows > 0:
            frame = frame.tail(max_rows).copy()
    required = {"decision_timestamp", "label_end_timestamp", TARGET_COL, "mfe_bps", "mae_bps", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Canonical table missing required columns: {missing}")
    frame[ACTUAL_COL] = pd.to_numeric(frame[TARGET_COL], errors="coerce")
    frame = frame.dropna(subset=["decision_timestamp", "label_end_timestamp", ACTUAL_COL]).copy()
    return frame.reset_index(drop=True)


def choose_ma_reversal_feature(feature_columns: list[str]) -> str:
    candidates = [column for column in feature_columns if column.startswith("ma_distance_bps_fw")]
    if not candidates:
        return ""
    return sorted(candidates, key=lambda value: (len(value), value))[0]


def fit_baselines(frame: pd.DataFrame, manifest: dict[str, Any], feature_columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = split_frame(frame, "train")
    validation = split_frame(frame, "validation")
    if train.empty or validation.empty:
        raise SystemExit("Canonical table must contain non-empty train and validation splits.")
    y_train = train[ACTUAL_COL].to_numpy(dtype=np.float64)
    mean_return = float(np.nanmean(y_train))
    mean_abs_return = float(np.nanmean(np.abs(y_train)))

    predictions: list[pd.DataFrame] = []
    model_info: dict[str, Any] = {}

    def add_prediction(name: str, values_by_index: pd.Series | np.ndarray, info: dict[str, Any]) -> None:
        pred = frame[["decision_timestamp", "label_end_timestamp", "split", ACTUAL_COL, "mfe_bps", "mae_bps"]].copy()
        pred["model"] = name
        pred[PRED_COL] = np.asarray(values_by_index, dtype=np.float64)
        predictions.append(pred)
        model_info[name] = info

    add_prediction("zero_return_predictor", np.zeros(len(frame), dtype=np.float64), {"family": "constant", "fit_split": "train"})
    add_prediction(
        "training_mean_return_predictor",
        np.full(len(frame), mean_return, dtype=np.float64),
        {"family": "constant", "fit_split": "train", "train_mean_return_bps": mean_return},
    )
    persistence = pd.to_numeric(frame.get("bucket_return_bps", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    add_prediction(
        "previous_return_persistence_predictor",
        persistence.to_numpy(dtype=np.float64),
        {"family": "persistence", "fit_split": "none", "source_column": "bucket_return_bps"},
    )

    ma_feature = choose_ma_reversal_feature(feature_columns)
    if ma_feature:
        x = -pd.to_numeric(train[ma_feature], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        denom = float(np.dot(x, x))
        coef = float(np.dot(x, y_train) / denom) if denom > 1e-12 else 0.0
        all_x = -pd.to_numeric(frame[ma_feature], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        add_prediction(
            f"moving_average_reversal_rule_{ma_feature}",
            coef * all_x,
            {"family": "rule", "fit_split": "train", "source_column": ma_feature, "coefficient": coef},
        )

    preprocessor = fit_preprocessor(train, feature_columns)
    X_train = transform_features(train, feature_columns, preprocessor)
    X_val = transform_features(validation, feature_columns, preprocessor)
    y_val = validation[ACTUAL_COL].to_numpy(dtype=np.float64)
    ridge_candidates = []
    for alpha in RIDGE_ALPHAS:
        coef = fit_ridge(X_train, y_train, alpha)
        val_pred = predict_linear(X_val, coef)
        ridge_candidates.append((rmse(y_val, val_pred), alpha, coef))
    ridge_candidates.sort(key=lambda item: item[0] if math.isfinite(item[0]) else 1e18)
    ridge_rmse, ridge_alpha, ridge_coef = ridge_candidates[0]
    all_X = transform_features(frame, feature_columns, preprocessor)
    add_prediction(
        "ridge_regression_return_predictor",
        predict_linear(all_X, ridge_coef),
        {
            "family": "ridge",
            "fit_split": "train",
            "selected_by": "validation_rmse",
            "selected_alpha": ridge_alpha,
            "validation_rmse_bps": ridge_rmse,
            "feature_count": len(feature_columns),
        },
    )

    y_direction = (train[ACTUAL_COL].to_numpy(dtype=np.float64) > 0.0).astype(float)
    direction_coef = fit_logistic(X_train, y_direction, LOGISTIC_ITERATIONS, LOGISTIC_LR, LOGISTIC_L2)
    direction_prob = predict_logistic(all_X, direction_coef)
    add_prediction(
        "logistic_direction_predictor",
        (direction_prob - 0.5) * 2.0 * mean_abs_return,
        {
            "family": "logistic",
            "target": "direction_positive",
            "fit_split": "train",
            "iterations": LOGISTIC_ITERATIONS,
            "learning_rate": LOGISTIC_LR,
            "l2": LOGISTIC_L2,
            "proxy_scale_bps": mean_abs_return,
        },
    )

    up_target = (train["mfe_bps"].to_numpy(dtype=np.float64) >= BARRIER_BPS).astype(float)
    up_coef = fit_logistic(X_train, up_target, LOGISTIC_ITERATIONS, LOGISTIC_LR, LOGISTIC_L2)
    up_prob = predict_logistic(all_X, up_coef)
    add_prediction(
        f"logistic_up_barrier_{BARRIER_BPS:g}bps_predictor",
        (up_prob - 0.5) * 2.0 * BARRIER_BPS,
        {"family": "logistic", "target": "up_barrier", "barrier_bps": BARRIER_BPS, "fit_split": "train"},
    )
    down_target = (train["mae_bps"].to_numpy(dtype=np.float64) <= -BARRIER_BPS).astype(float)
    down_coef = fit_logistic(X_train, down_target, LOGISTIC_ITERATIONS, LOGISTIC_LR, LOGISTIC_L2)
    down_prob = predict_logistic(all_X, down_coef)
    add_prediction(
        f"logistic_down_barrier_{BARRIER_BPS:g}bps_predictor",
        -(down_prob - 0.5) * 2.0 * BARRIER_BPS,
        {"family": "logistic", "target": "down_barrier", "barrier_bps": BARRIER_BPS, "fit_split": "train"},
    )

    return pd.concat(predictions, ignore_index=True), model_info


def allowed_policies_for_instrument(instrument: str) -> list[str]:
    instrument = str(instrument).strip().lower()
    if instrument in {"kraken_spot_long_only", "inventory_spot_long_flat"}:
        return ["direct_gt"]
    return ["direct_gt", "inverse_gt"]


def evaluate_predictions(predictions: pd.DataFrame, manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    horizon_ms = int(
        max(
            1.0,
            np.nanmedian(
                pd.to_numeric(predictions["label_end_timestamp"], errors="coerce")
                - pd.to_numeric(predictions["decision_timestamp"], errors="coerce")
            ),
        )
    )
    instrument = manifest.get("instrument", "")
    allowed_policies = allowed_policies_for_instrument(str(instrument))
    pred_rows = []
    policy_rows = []
    for model_name, model_group in predictions.groupby("model", sort=True):
        for split in ["train", "validation", "untouched_holdout"]:
            split_data = model_group[model_group["split"].astype(str).eq(split)].copy()
            if split_data.empty:
                continue
            pred_rows.append(prediction_metrics(split_data, model_name, split))
            policy_rows.extend(policy_metric_rows(split_data, model_name, split, horizon_ms, allowed_policies))
    pred_metrics = pd.DataFrame(pred_rows)
    policy_metrics = pd.DataFrame(policy_rows)

    selected_rows = []
    if not policy_metrics.empty:
        validation = policy_metrics[
            policy_metrics["split"].eq("validation")
            & policy_metrics["cost_bps"].astype(float).sub(DECISION_COST_BPS).abs().lt(1e-12)
        ].copy()
        validation = validation.sort_values(
            ["model", "position_cum_net_bps", "non_overlapping_cum_net_bps", "position_trade_count"],
            ascending=[True, False, False, False],
        )
        for model_name, group in validation.groupby("model", sort=True):
            selected = group.iloc[0]
            holdout_match = policy_metrics[
                policy_metrics["model"].eq(model_name)
                & policy_metrics["split"].eq("untouched_holdout")
                & policy_metrics["policy"].eq(selected["policy"])
                & policy_metrics["threshold_bps"].astype(float).sub(float(selected["threshold_bps"])).abs().lt(1e-12)
                & policy_metrics["cost_bps"].astype(float).sub(DECISION_COST_BPS).abs().lt(1e-12)
            ]
            holdout = holdout_match.iloc[0] if not holdout_match.empty else pd.Series(dtype=object)
            validation_trades = int(safe_float(selected.get("position_trade_count"), 0))
            holdout_trades = int(safe_float(holdout.get("position_trade_count"), 0))
            if validation_trades < MIN_POSITION_TRADES:
                validation_status = "insufficient_sample"
                status_reason = f"validation_position_trade_count<{MIN_POSITION_TRADES}"
            elif safe_float(selected.get("position_cum_net_bps")) > 0:
                validation_status = "validation_survivor"
                status_reason = "validation_position_cum_net_positive"
            else:
                validation_status = "training_candidate"
                status_reason = "validation_position_cum_net_not_positive"
            holdout_status = (
                "holdout_survivor"
                if validation_status == "validation_survivor"
                and holdout_trades >= MIN_POSITION_TRADES
                and safe_float(holdout.get("position_cum_net_bps")) > 0
                else "validation_only_or_reject"
            )
            selected_rows.append(
                {
                    "model": model_name,
                    "selected_policy": selected["policy"],
                    "selected_threshold_bps": selected["threshold_bps"],
                    "decision_cost_bps": DECISION_COST_BPS,
                    "validation_position_cum_net_bps": selected["position_cum_net_bps"],
                    "validation_position_trade_count": selected["position_trade_count"],
                    "validation_position_max_dip_net_bps": selected["position_max_dip_net_bps"],
                    "holdout_position_cum_net_bps": holdout.get("position_cum_net_bps", math.nan),
                    "holdout_position_trade_count": holdout.get("position_trade_count", 0),
                    "holdout_position_max_dip_net_bps": holdout.get("position_max_dip_net_bps", math.nan),
                    "status": holdout_status if holdout_status == "holdout_survivor" else validation_status,
                    "status_reason": status_reason,
                    "min_position_trades": MIN_POSITION_TRADES,
                    "selection_stage": "validation_selected",
                    "holdout_stage": "untouched_holdout_evaluated",
                }
            )
    return pred_metrics, policy_metrics, pd.DataFrame(selected_rows)


def write_text(path: Path, selected: pd.DataFrame, pred_metrics: pd.DataFrame, policy_metrics: pd.DataFrame, manifest: dict[str, Any]) -> None:
    lines = [
        "Rawseq Canonical Baseline Evaluation",
        "",
        f"Created at: {datetime.now(UTC).isoformat()}",
        f"Canonical source: {manifest.get('source_path', '')}",
        f"Target: future_market_return_bps_{manifest.get('horizon_seconds', '')}s",
        f"Instrument: {manifest.get('instrument', '')}",
        f"Feature schema hash: {manifest.get('feature_schema_hash', '')}",
        "",
        "Safety: paper_only=true training=baseline_only promotion=false champion_mutation=false orders=false",
        "",
        "Selected validation policies evaluated on untouched holdout:",
    ]
    if selected.empty:
        lines.append("  none")
    else:
        for _, row in selected.sort_values(["status", "holdout_position_cum_net_bps"], ascending=[True, False]).iterrows():
            lines.append(
                "  "
                f"{row['model']} status={row['status']} policy={row['selected_policy']} "
                f"threshold={fmt(row['selected_threshold_bps'])} "
                f"val_pos_cum={fmt(row['validation_position_cum_net_bps'])} "
                f"val_trades={int(safe_float(row['validation_position_trade_count'], 0))} "
                f"holdout_pos_cum={fmt(row['holdout_position_cum_net_bps'])} "
                f"holdout_trades={int(safe_float(row['holdout_position_trade_count'], 0))} "
                f"reason={row.get('status_reason', '')}"
            )
    lines += ["", "Prediction metrics, validation split:"]
    validation_pred = pred_metrics[pred_metrics["split"].eq("validation")].copy()
    if validation_pred.empty:
        lines.append("  none")
    else:
        for _, row in validation_pred.sort_values("rmse_bps").iterrows():
            lines.append(
                "  "
                f"{row['model']} rmse={fmt(row['rmse_bps'])} mae={fmt(row['mae_bps'])} "
                f"corr={fmt(row['return_correlation'])} dir_acc={fmt(row['directional_accuracy'])} "
                f"bal_acc={fmt(row['balanced_accuracy'])}"
            )
    lines += [
        "",
        "Interpretation:",
        "  Baseline policies are selected on validation only.",
        "  Untouched holdout metrics are diagnostic evidence, not a promotion decision.",
        "  Rawseq candidates should beat these baselines on validation windows before ensemble work resumes.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    base_dir, table_path, manifest_path = resolve_inputs()
    manifest = load_json(manifest_path)
    frame = load_table(table_path)
    feature_columns = feature_columns_from_manifest(manifest, frame)
    if not feature_columns:
        raise SystemExit("No feature columns found for baseline training.")
    predictions, model_info = fit_baselines(frame, manifest, feature_columns)
    pred_metrics, policy_metrics, selected = evaluate_predictions(predictions, manifest)

    out_dir = OUTPUT_ROOT / f"canonical_baselines_{now_stamp()}_{str(manifest.get('feature_schema_hash', 'unknown'))[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "baseline_predictions.csv"
    pred_metrics_path = out_dir / "baseline_prediction_metrics.csv"
    policy_metrics_path = out_dir / "baseline_policy_metrics.csv"
    selected_path = out_dir / "baseline_selected_policies.csv"
    manifest_out_path = out_dir / "baseline_manifest.json"
    summary_path = out_dir / "baseline_summary.txt"

    predictions.to_csv(predictions_path, index=False)
    pred_metrics.to_csv(pred_metrics_path, index=False)
    policy_metrics.to_csv(policy_metrics_path, index=False)
    selected.to_csv(selected_path, index=False)
    baseline_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "canonical_dir": str(base_dir),
        "canonical_table_path": str(table_path),
        "canonical_manifest_path": str(manifest_path),
        "feature_columns": feature_columns,
        "feature_schema_hash": manifest.get("feature_schema_hash", ""),
        "target": f"future_market_return_bps_{manifest.get('horizon_seconds', '')}s",
        "costs_bps": COSTS,
        "thresholds_bps": THRESHOLDS,
        "decision_cost_bps": DECISION_COST_BPS,
        "ridge_alphas": RIDGE_ALPHAS,
        "logistic_iterations": LOGISTIC_ITERATIONS,
        "logistic_lr": LOGISTIC_LR,
        "logistic_l2": LOGISTIC_L2,
        "barrier_bps": BARRIER_BPS,
        "min_position_trades": MIN_POSITION_TRADES,
        "models": model_info,
        "paper_only": True,
        "training": "baseline_only",
        "promotion": False,
        "champion_mutation": False,
        "orders": False,
    }
    manifest_out_path.write_text(json.dumps(baseline_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_text(summary_path, selected, pred_metrics, policy_metrics, manifest)

    print("Rawseq canonical baseline evaluation complete")
    print(f"Rows: predictions={len(predictions)} prediction_metrics={len(pred_metrics)} policy_metrics={len(policy_metrics)}")
    print(f"Output dir: {out_dir}")
    print(f"Selected policies: {selected_path}")
    print(f"Summary: {summary_path}")
    if not selected.empty:
        print(selected.sort_values("holdout_position_cum_net_bps", ascending=False).head(20).to_string(index=False))
    print("Safety: paper_only=true training=baseline_only promotion=false champion_mutation=false orders=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
