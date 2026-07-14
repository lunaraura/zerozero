#!/usr/bin/env python3
"""Generate leakage-aware out-of-fold prediction tables from canonical data.

This script builds rolling train/validation/test folds before the untouched
holdout period. Each base family is trained only on its fold train rows,
selects internal ridge settings on validation rows, then predicts the next
unseen fold.

Safety: paper-only research. No private API, no orders, no promotion, and no
champion mutation.
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_canonical_tables"
OUTPUT_ROOT = Path(
    os.getenv(
        "RAWSEQ_OOF_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_canonical_oof_predictions"),
    )
).expanduser()
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

TABLE_PATH_ENV = os.getenv("RAWSEQ_CANONICAL_TABLE_PATH", "").strip()
MANIFEST_PATH_ENV = os.getenv("RAWSEQ_CANONICAL_MANIFEST_PATH", "").strip()
SMOKE_MODE = os.getenv("RAWSEQ_OOF_SMOKE_MODE", "false").strip().lower() in {"1", "true", "yes", "y"}
TRAIN_ROWS = int(float(os.getenv("RAWSEQ_OOF_TRAIN_ROWS", "10000" if SMOKE_MODE else "20000")))
VALIDATION_ROWS = int(float(os.getenv("RAWSEQ_OOF_VALIDATION_ROWS", "3000" if SMOKE_MODE else "5000")))
TEST_ROWS = int(float(os.getenv("RAWSEQ_OOF_TEST_ROWS", "3000" if SMOKE_MODE else "5000")))
STEP_ROWS = int(float(os.getenv("RAWSEQ_OOF_STEP_ROWS", str(TEST_ROWS))))
MAX_FOLDS = int(float(os.getenv("RAWSEQ_OOF_MAX_FOLDS", "1" if SMOKE_MODE else "5")))
RIDGE_ALPHAS = [
    float(item.strip())
    for item in os.getenv("RAWSEQ_OOF_RIDGE_ALPHAS", "0.01,0.1,1,10,100").split(",")
    if item.strip()
]
LOGISTIC_ITERATIONS = int(float(os.getenv("RAWSEQ_OOF_LOGISTIC_ITERATIONS", "80" if SMOKE_MODE else "200")))
LOGISTIC_LR = float(os.getenv("RAWSEQ_OOF_LOGISTIC_LR", "0.05"))
LOGISTIC_L2 = float(os.getenv("RAWSEQ_OOF_LOGISTIC_L2", "0.1"))

TARGET_COL = "gross_future_return_bps"


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


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


def load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"decision_timestamp", "label_end_timestamp", TARGET_COL, "mfe_bps", "mae_bps", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Canonical table missing required columns: {missing}")
    frame = frame.dropna(subset=["decision_timestamp", "label_end_timestamp", TARGET_COL]).copy()
    return frame.sort_values("decision_timestamp").reset_index(drop=True)


def feature_columns_from_manifest(manifest: dict[str, Any], frame: pd.DataFrame) -> list[str]:
    schema = manifest.get("feature_schema") if isinstance(manifest.get("feature_schema"), dict) else {}
    features = [str(item) for item in schema.get("features", []) if str(item) in frame.columns]
    return features


def family_feature_columns(frame: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, list[str]]:
    canonical_features = feature_columns_from_manifest(manifest, frame)
    rawseq_features = [
        column
        for column in canonical_features
        if any(
            token in column
            for token in [
                "bucket_return_bps",
                "ma_distance",
                "rolling_range",
                "rolling_volatility",
                "distance_to_recent_high",
                "distance_to_recent_low",
            ]
        )
    ]
    micro_tokens = [
        "spread_percent",
        "bid_depth_10bps",
        "ask_depth_10bps",
        "bid_depth_25bps",
        "ask_depth_25bps",
        "total_trade_volume_10s",
        "trade_count_10s",
        "market_pressure_10s",
        "order_book_imbalance_10bps",
        "order_book_imbalance_25bps",
    ]
    microstructure = [
        column
        for column in frame.columns
        if any(token in column for token in micro_tokens)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    regime = [
        column
        for column in canonical_features
        if any(token in column for token in ["fw300", "rolling_volatility", "rolling_range", "ma_distance"])
    ]
    if not regime:
        regime = rawseq_features
    return {
        "rawseq_alpha": rawseq_features,
        "microstructure_alpha": microstructure,
        "regime_gate": regime,
    }


def fit_preprocessor(train: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    values = train[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    means = np.nanmean(values, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    filled = np.where(np.isfinite(values), values, means)
    std = np.nanstd(filled, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return {"mean": means, "std": std}


def transform(frame: pd.DataFrame, columns: list[str], preprocessor: dict[str, np.ndarray]) -> np.ndarray:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
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
    return np.column_stack([np.ones(len(X), dtype=np.float64), X]) @ coef


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return math.nan
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2)))


def correlation(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() < 3:
        return math.nan
    actual = actual[mask]
    pred = pred[mask]
    if np.std(actual) <= 1e-12 or np.std(pred) <= 1e-12:
        return math.nan
    return float(np.corrcoef(actual, pred)[0, 1])


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(X), dtype=np.float64), X])
    if len(np.unique(y[np.isfinite(y)])) < 2:
        base = np.clip(np.nanmean(y), 1e-6, 1.0 - 1e-6)
        coef = np.zeros(design.shape[1], dtype=np.float64)
        coef[0] = math.log(base / (1.0 - base))
        return coef
    coef = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(LOGISTIC_ITERATIONS):
        pred = sigmoid(design @ coef)
        grad = design.T @ (pred - y) / max(1, len(y))
        reg = LOGISTIC_L2 * coef / max(1, len(y))
        reg[0] = 0.0
        coef -= LOGISTIC_LR * (grad + reg)
    return coef


def predict_logistic(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return sigmoid(np.column_stack([np.ones(len(X), dtype=np.float64), X]) @ coef)


def fit_ridge_family(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    if not columns:
        return np.full(len(test), np.nan, dtype=np.float64), {"status": "missing_features"}
    pre = fit_preprocessor(train, columns)
    X_train = transform(train, columns, pre)
    X_val = transform(validation, columns, pre)
    X_test = transform(test, columns, pre)
    y_train = train[TARGET_COL].to_numpy(dtype=np.float64)
    y_val = validation[TARGET_COL].to_numpy(dtype=np.float64)
    candidates = []
    for alpha in RIDGE_ALPHAS:
        coef = fit_ridge(X_train, y_train, alpha)
        pred_val = predict_linear(X_val, coef)
        candidates.append((rmse(y_val, pred_val), alpha, coef, pred_val))
    candidates.sort(key=lambda item: item[0] if math.isfinite(item[0]) else 1e18)
    score, alpha, coef, pred_val = candidates[0]
    pred_test = predict_linear(X_test, coef)
    return pred_test, {
        "status": "ok",
        "model_type": "ridge",
        "selected_alpha": alpha,
        "validation_rmse_bps": score,
        "validation_correlation": correlation(y_val, pred_val),
        "feature_count": len(columns),
    }


def fit_regime_gate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    if not columns:
        return np.full(len(test), np.nan, dtype=np.float64), {"status": "missing_features"}
    pre = fit_preprocessor(train, columns)
    X_train = transform(train, columns, pre)
    X_val = transform(validation, columns, pre)
    X_test = transform(test, columns, pre)
    y_train = (train[TARGET_COL].to_numpy(dtype=np.float64) > 0.0).astype(float)
    y_val = (validation[TARGET_COL].to_numpy(dtype=np.float64) > 0.0).astype(float)
    coef = fit_logistic(X_train, y_train)
    pred_val = predict_logistic(X_val, coef)
    pred_test = predict_logistic(X_test, coef)
    direction_pred = pred_val >= 0.5
    validation_accuracy = float(np.mean(direction_pred == y_val.astype(bool))) if len(y_val) else math.nan
    return pred_test, {
        "status": "ok",
        "model_type": "logistic_regime_gate",
        "validation_direction_accuracy": validation_accuracy,
        "feature_count": len(columns),
    }


def build_folds(frame: pd.DataFrame, manifest: dict[str, Any]) -> list[dict[str, int | str]]:
    discovery = frame[~frame["split"].astype(str).eq("untouched_holdout")].copy().reset_index()
    purge_rows = int(manifest.get("purge_rows") or manifest.get("maximum_label_lookahead_rows") or 0)
    embargo_rows = int(manifest.get("embargo_rows") or manifest.get("maximum_label_lookahead_rows") or 0)
    folds = []
    start = 0
    for fold_idx in range(MAX_FOLDS):
        train_start = start
        train_end = train_start + TRAIN_ROWS
        validation_start = train_end + embargo_rows + purge_rows
        validation_end = validation_start + VALIDATION_ROWS
        test_start = validation_end + embargo_rows + purge_rows
        test_end = test_start + TEST_ROWS
        if test_end > len(discovery):
            break
        folds.append(
            {
                "fold_id": f"fold_{fold_idx:03d}",
                "train_start": int(train_start),
                "train_end": int(train_end),
                "validation_start": int(validation_start),
                "validation_end": int(validation_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
                "purge_rows": int(purge_rows),
                "embargo_rows": int(embargo_rows),
            }
        )
        start += STEP_ROWS
    return folds


def fold_slice(discovery: pd.DataFrame, fold: dict[str, Any], start_key: str, end_key: str) -> pd.DataFrame:
    return discovery.iloc[int(fold[start_key]) : int(fold[end_key])].copy().reset_index(drop=True)


def build_oof(frame: pd.DataFrame, manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    discovery = frame[~frame["split"].astype(str).eq("untouched_holdout")].copy().reset_index(drop=True)
    folds = build_folds(frame, manifest)
    families = family_feature_columns(frame, manifest)
    prediction_rows = []
    metric_rows = []
    fold_rows = []
    for fold in folds:
        train = fold_slice(discovery, fold, "train_start", "train_end")
        validation = fold_slice(discovery, fold, "validation_start", "validation_end")
        test = fold_slice(discovery, fold, "test_start", "test_end")
        rawseq_pred, rawseq_meta = fit_ridge_family(train, validation, test, families["rawseq_alpha"])
        micro_pred, micro_meta = fit_ridge_family(train, validation, test, families["microstructure_alpha"])
        regime_prob, regime_meta = fit_regime_gate(train, validation, test, families["regime_gate"])
        fold_row = {
            **fold,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_start_timestamp": safe_float(train["decision_timestamp"].min()),
            "train_end_timestamp": safe_float(train["decision_timestamp"].max()),
            "validation_start_timestamp": safe_float(validation["decision_timestamp"].min()),
            "validation_end_timestamp": safe_float(validation["decision_timestamp"].max()),
            "test_start_timestamp": safe_float(test["decision_timestamp"].min()),
            "test_end_timestamp": safe_float(test["decision_timestamp"].max()),
            "base_model_training_end": safe_float(train["decision_timestamp"].max()),
            "rawseq_status": rawseq_meta.get("status", ""),
            "microstructure_status": micro_meta.get("status", ""),
            "regime_status": regime_meta.get("status", ""),
        }
        fold_row["fold_leakage_guard_pass"] = bool(
            fold_row["train_end"] <= fold_row["validation_start"] - purge_rows - embargo_rows
            and fold_row["validation_end"] <= fold_row["test_start"] - purge_rows - embargo_rows
            and safe_float(fold_row["base_model_training_end"]) < safe_float(fold_row["test_start_timestamp"])
        )
        fold_rows.append(fold_row)
        actual_test = test[TARGET_COL].to_numpy(dtype=np.float64)
        for name, pred, meta in [
            ("rawseq_alpha", rawseq_pred, rawseq_meta),
            ("microstructure_alpha", micro_pred, micro_meta),
            ("regime_gate", regime_prob, regime_meta),
        ]:
            metric_rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "family": name,
                    **meta,
                    "test_rows": len(test),
                    "test_rmse_bps": rmse(actual_test, pred) if name != "regime_gate" else math.nan,
                    "test_correlation": correlation(actual_test, pred) if name != "regime_gate" else math.nan,
                }
            )
        table = pd.DataFrame(
            {
                "timestamp": pd.to_numeric(test["decision_timestamp"], errors="coerce"),
                "actual_return_bps": pd.to_numeric(test[TARGET_COL], errors="coerce"),
                "rawseq_pred_bps": rawseq_pred,
                "microstructure_pred_bps": micro_pred,
                "regime_probability": regime_prob,
                "source_fold": fold["fold_id"],
                "base_model_training_end": safe_float(train["decision_timestamp"].max()),
                "feature_schema_hash": manifest.get("feature_schema_hash", ""),
                "target": f"future_market_return_bps_{manifest.get('horizon_seconds', '')}s",
                "rawseq_model_status": rawseq_meta.get("status", ""),
                "microstructure_model_status": micro_meta.get("status", ""),
                "regime_model_status": regime_meta.get("status", ""),
                "paper_only": True,
                "promotion": False,
                "champion_mutation": False,
                "orders": False,
            }
        )
        table["prediction_after_training_end"] = (
            pd.to_numeric(table["timestamp"], errors="coerce")
            > safe_float(train["decision_timestamp"].max())
        )
        table["selection_stage"] = "oof_fold_test_prediction"
        prediction_rows.append(table)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(metric_rows)


def leakage_guard_summary(predictions: pd.DataFrame, folds: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    holdout = frame[frame["split"].astype(str).eq("untouched_holdout")]
    if predictions.empty:
        holdout_hits = 0
        predictions_after_training_end = True
    else:
        pred_ts = pd.to_numeric(predictions["timestamp"], errors="coerce").dropna().astype("int64")
        holdout_ts = pd.to_numeric(holdout["decision_timestamp"], errors="coerce").dropna().astype("int64")
        holdout_hits = int(pred_ts.isin(set(holdout_ts.to_list())).sum()) if not holdout_ts.empty else 0
        predictions_after_training_end = bool(
            (pd.to_numeric(predictions["timestamp"], errors="coerce")
            > pd.to_numeric(predictions["base_model_training_end"], errors="coerce")).all()
        )
    fold_guards = bool(folds["fold_leakage_guard_pass"].all()) if "fold_leakage_guard_pass" in folds else False
    return {
        "oof_leakage_guard_pass": bool(fold_guards and predictions_after_training_end and holdout_hits == 0),
        "fold_leakage_guard_pass": fold_guards,
        "prediction_timestamps_after_training_end": predictions_after_training_end,
        "holdout_rows_in_predictions": holdout_hits,
        "untouched_holdout_excluded": True,
        "min_prediction_timestamp": safe_float(predictions["timestamp"].min()) if not predictions.empty else math.nan,
        "max_prediction_timestamp": safe_float(predictions["timestamp"].max()) if not predictions.empty else math.nan,
        "max_base_model_training_end": safe_float(predictions["base_model_training_end"].max())
        if not predictions.empty
        else math.nan,
    }


def write_text(
    path: Path,
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    metrics: pd.DataFrame,
    manifest: dict[str, Any],
    guards: dict[str, Any],
) -> None:
    lines = [
        "Rawseq Canonical Out-of-Fold Prediction Table",
        "",
        f"Created at: {datetime.now(UTC).isoformat()}",
        f"Target: future_market_return_bps_{manifest.get('horizon_seconds', '')}s",
        f"Feature schema hash: {manifest.get('feature_schema_hash', '')}",
        f"Rows: {len(predictions)}",
        f"Folds: {len(folds)}",
        f"Leakage guard pass: {guards.get('oof_leakage_guard_pass')}",
        f"Holdout rows in predictions: {guards.get('holdout_rows_in_predictions')}",
        "",
        "Safety: paper_only=true training=fold_baselines_only promotion=false champion_mutation=false orders=false",
        "",
        "Fold status:",
    ]
    if folds.empty:
        lines.append("  none")
    else:
        for _, row in folds.iterrows():
            lines.append(
                "  "
                f"{row['fold_id']} train={row['train_rows']} validation={row['validation_rows']} "
                f"test={row['test_rows']} rawseq={row['rawseq_status']} "
                f"microstructure={row['microstructure_status']} regime={row['regime_status']} "
                f"leakage_guard={row.get('fold_leakage_guard_pass', '')}"
            )
    lines += ["", "Model metrics:"]
    if metrics.empty:
        lines.append("  none")
    else:
        for _, row in metrics.iterrows():
            lines.append(
                "  "
                f"{row['fold_id']} {row['family']} status={row.get('status', '')} "
                f"val_rmse={fmt(row.get('validation_rmse_bps'))} "
                f"test_rmse={fmt(row.get('test_rmse_bps'))} "
                f"test_corr={fmt(row.get('test_correlation'))} "
                f"features={row.get('feature_count', '')}"
            )
    lines += [
        "",
        "Notes:",
        "  The final canonical untouched_holdout split is excluded from these OOF folds.",
        "  microstructure_pred_bps is populated only when canonical data includes spread/depth/flow columns.",
        "  This table is for stacker/ensemble training; final holdout must not tune ensemble thresholds.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    base_dir, table_path, manifest_path = resolve_inputs()
    manifest = load_json(manifest_path)
    frame = load_table(table_path)
    predictions, folds, metrics = build_oof(frame, manifest)
    guards = leakage_guard_summary(predictions, folds, frame)
    output_dir = OUTPUT_ROOT / f"canonical_oof_{now_stamp()}_{str(manifest.get('feature_schema_hash', 'unknown'))[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "canonical_oof_predictions.csv"
    folds_path = output_dir / "canonical_oof_fold_manifest.csv"
    metrics_path = output_dir / "canonical_oof_model_metrics.csv"
    summary_path = output_dir / "canonical_oof_summary.txt"
    manifest_out_path = output_dir / "canonical_oof_manifest.json"
    predictions.to_csv(predictions_path, index=False)
    folds.to_csv(folds_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    write_text(summary_path, predictions, folds, metrics, manifest, guards)
    manifest_out = {
        "created_at": datetime.now(UTC).isoformat(),
        "canonical_dir": str(base_dir),
        "canonical_table_path": str(table_path),
        "canonical_manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "train_rows": TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "test_rows": TEST_ROWS,
        "step_rows": STEP_ROWS,
        "max_folds": MAX_FOLDS,
        "folds_created": int(len(folds)),
        "feature_schema_hash": manifest.get("feature_schema_hash", ""),
        "target": f"future_market_return_bps_{manifest.get('horizon_seconds', '')}s",
        "untouched_holdout_excluded": True,
        "paper_only": True,
        "promotion": False,
        "champion_mutation": False,
        "orders": False,
    }
    manifest_out.update(guards)
    manifest_out_path.write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Rawseq canonical OOF prediction generation complete")
    print(f"Rows: {len(predictions)}")
    print(f"Folds: {len(folds)}")
    print(f"Output dir: {output_dir}")
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")
    if not folds.empty:
        print(folds.head(20).to_string(index=False))
    print("Safety: paper_only=true training=fold_baselines_only promotion=false champion_mutation=false orders=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
