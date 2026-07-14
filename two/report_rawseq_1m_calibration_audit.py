#!/usr/bin/env python3
"""Development-only calibration audit for the leading 1m downside-risk contract."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import (
    DEFAULT_OUTPUT_ROOT,
    SAFETY_FLAGS,
    build_features,
    downside_event_targets,
    env_path,
    load_candles,
    metric_row,
    now_stamp,
    resolve_source_files,
    save_reload_prediction_parity,
    stable_hash,
    write_csv,
    write_json,
)

DEFAULT_SCOUT_DIR = Path(r"F:\rsio\rawseq_1m_baseline_scout\rawseq_1m_baseline_scout_20260712T044244Z")
HORIZON_MINUTES = 1
VOL_WINDOW_MINUTES = 240


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite_xy(features: pd.DataFrame, target: pd.Series, start: int, end: int, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = features.iloc[start : end + 1]
    x = frame[feature_cols].to_numpy(dtype=np.float64)
    y = pd.to_numeric(target.iloc[start : end + 1], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[mask], y[mask]


def fit_hgb(train_x: np.ndarray, train_y: np.ndarray) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    if len(np.unique(train_y)) < 2:
        return {"constant_probability": float(np.mean(train_y)) if len(train_y) else 0.5}
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=0.01, random_state=1337),
    )
    model.fit(train_x, train_y.astype(int))
    return model


def predict_model(model: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(len(x), float(model.get("constant_probability", 0.5)))
    return model.predict_proba(x)[:, 1]


def fit_logistic(train_x: np.ndarray, train_y: np.ndarray) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(train_y)) < 2:
        return {"constant_probability": float(np.mean(train_y)) if len(train_y) else 0.5}
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300, solver="lbfgs"))
    model.fit(train_x, train_y.astype(int))
    return model


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(cal_p: np.ndarray, cal_y: np.ndarray) -> Any:
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(cal_y)) < 2:
        return {"constant_probability": float(np.mean(cal_y)) if len(cal_y) else 0.5}
    model = LogisticRegression(max_iter=300, solver="lbfgs")
    model.fit(logit(cal_p).reshape(-1, 1), cal_y.astype(int))
    return model


def predict_platt(model: Any, p: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(len(p), float(model.get("constant_probability", 0.5)))
    return model.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def fit_isotonic(cal_p: np.ndarray, cal_y: np.ndarray) -> Any:
    from sklearn.isotonic import IsotonicRegression

    if len(np.unique(cal_y)) < 2:
        return {"constant_probability": float(np.mean(cal_y)) if len(cal_y) else 0.5}
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(cal_p, cal_y)
    return model


def predict_isotonic(model: Any, p: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(len(p), float(model.get("constant_probability", 0.5)))
    return np.asarray(model.predict(p), dtype=np.float64)


def reliability_rows(y: np.ndarray, p: np.ndarray, fold_id: int, model: str, bins: int = 10) -> list[dict[str, Any]]:
    rows = []
    for idx, lo in enumerate(np.linspace(0, 1, bins, endpoint=False)):
        hi = lo + 1.0 / bins
        mask = (p >= lo) & (p < hi if idx < bins - 1 else p <= hi)
        rows.append(
            {
                "fold_id": fold_id,
                "model": model,
                "bin_index": idx,
                "bin_low": lo,
                "bin_high": hi,
                "rows": int(mask.sum()),
                "predicted_probability_mean": float(np.mean(p[mask])) if mask.any() else math.nan,
                "event_rate": float(np.mean(y[mask])) if mask.any() else math.nan,
            }
        )
    return rows


def saturation_fraction(p: np.ndarray) -> float:
    return float(((p <= 1e-4) | (p >= 1 - 1e-4)).mean()) if len(p) else math.nan


def evaluate_fold(fold: dict[str, str], features: pd.DataFrame, target: pd.Series, feature_cols: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_id = int(float(fold["fold_id"]))
    train_start = int(float(fold["train_start_index"]))
    train_end = int(float(fold["train_end_index"]))
    val_start = int(float(fold["validation_start_index"]))
    val_end = int(float(fold["validation_end_index"]))
    cal_size = max(1000, int((train_end - train_start + 1) * 0.20))
    fit_end = max(train_start, train_end - cal_size)
    train_x, train_y = finite_xy(features, target, train_start, train_end, feature_cols)
    fit_x, fit_y = finite_xy(features, target, train_start, fit_end, feature_cols)
    cal_x, cal_y = finite_xy(features, target, fit_end + 1, train_end, feature_cols)
    val_x, val_y = finite_xy(features, target, val_start, val_end, feature_cols)
    prevalence = float(np.mean(train_y)) if len(train_y) else 0.5
    baseline = np.full(len(val_y), np.clip(prevalence, 1e-6, 1 - 1e-6))
    rows: list[dict[str, Any]] = []
    rel: list[dict[str, Any]] = []

    raw_hgb = fit_hgb(train_x, train_y)
    raw_pred = predict_model(raw_hgb, val_x)
    raw_parity = save_reload_prediction_parity(raw_hgb, predict_model, val_x)

    base_for_cal = fit_hgb(fit_x, fit_y)
    cal_raw = predict_model(base_for_cal, cal_x)
    val_raw_for_cal = predict_model(base_for_cal, val_x)
    platt = fit_platt(cal_raw, cal_y)
    iso = fit_isotonic(cal_raw, cal_y)
    platt_pred = predict_platt(platt, val_raw_for_cal)
    iso_pred = predict_isotonic(iso, val_raw_for_cal)
    logistic = fit_logistic(train_x, train_y)
    logistic_pred = predict_model(logistic, val_x)
    logistic_parity = save_reload_prediction_parity(logistic, predict_model, val_x)
    models = {
        "raw_hist_gradient_boosting_shallow": (raw_pred, raw_parity, "full_train"),
        "fold_safe_platt_hgb": (platt_pred, (True, 0.0), "train_fit_then_train_calibration"),
        "fold_safe_isotonic_hgb": (iso_pred, (True, 0.0), "train_fit_then_train_calibration"),
        "global_regularized_logistic": (logistic_pred, logistic_parity, "full_train"),
    }
    rows.append(
        {
            "fold_id": fold_id,
            "model": "beta_calibration",
            "status": "not_implemented",
            "reason": "beta calibration not implemented in this repository without adding new leakage-sensitive dependency",
            "holdout_accessed": False,
            **SAFETY_FLAGS,
        }
    )
    for model_name, (pred, parity, calibration_source) in models.items():
        metrics = metric_row(val_y, pred, baseline)
        row = {
            "fold_id": fold_id,
            "horizon_minutes": HORIZON_MINUTES,
            "vol_window_minutes": VOL_WINDOW_MINUTES,
            "model": model_name,
            "status": "OK",
            "train_rows": int(len(train_y)),
            "calibration_rows": int(len(cal_y)) if "fold_safe" in model_name else 0,
            "validation_rows": int(len(val_y)),
            "calibration_source": calibration_source,
            "calibration_formula": "slope/intercept from linear fit: outcome = slope * logit(prediction) + intercept",
            "probability_minimum": float(np.min(pred)) if len(pred) else math.nan,
            "probability_maximum": float(np.max(pred)) if len(pred) else math.nan,
            "probability_mean": float(np.mean(pred)) if len(pred) else math.nan,
            "probability_standard_deviation": float(np.std(pred)) if len(pred) else math.nan,
            "prediction_saturation_fraction": saturation_fraction(pred),
            "save_reload_prediction_parity": parity[0],
            "save_reload_prediction_max_abs_diff": parity[1],
            "holdout_accessed": False,
            **metrics,
            **SAFETY_FLAGS,
        }
        rows.append(row)
        rel.extend(reliability_rows(val_y, pred, fold_id, model_name))
    return rows, rel


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "OK":
            grouped.setdefault(str(row["model"]), []).append(row)
    out = []
    for model, vals in grouped.items():
        skills = [float(row["brier_skill_vs_prevalence"]) for row in vals if math.isfinite(float(row["brier_skill_vs_prevalence"]))]
        log_improvements = [float(row["log_loss_improvement_vs_prevalence"]) for row in vals if math.isfinite(float(row["log_loss_improvement_vs_prevalence"]))]
        pr_lifts = [float(row["pr_auc_lift_over_event_prevalence"]) for row in vals if math.isfinite(float(row["pr_auc_lift_over_event_prevalence"]))]
        slopes = [float(row["calibration_slope"]) for row in vals if math.isfinite(float(row["calibration_slope"]))]
        wins = sum(1 for v in skills if v > 0)
        out.append(
            {
                "model": model,
                "folds": len(vals),
                "fold_wins": wins,
                "fold_win_fraction": wins / len(vals) if vals else math.nan,
                "median_brier_skill": float(np.median(skills)) if skills else math.nan,
                "worst_fold_brier_skill": float(np.min(skills)) if skills else math.nan,
                "median_log_loss_improvement": float(np.median(log_improvements)) if log_improvements else math.nan,
                "median_pr_auc_lift": float(np.median(pr_lifts)) if pr_lifts else math.nan,
                "median_calibration_slope": float(np.median(slopes)) if slopes else math.nan,
                "save_reload_parity_all": all(str(row.get("save_reload_prediction_parity")).lower() in {"true", "1"} for row in vals),
                "positive_worst_fold": min(skills) > 0 if skills else False,
                "holdout_accessed": False,
                **SAFETY_FLAGS,
            }
        )
    out.sort(
        key=lambda r: (
            not bool(r["positive_worst_fold"]),
            -float(r["median_brier_skill"]) if math.isfinite(float(r["median_brier_skill"])) else 999,
            -float(r["median_log_loss_improvement"]) if math.isfinite(float(r["median_log_loss_improvement"])) else 999,
            abs(float(r["median_calibration_slope"]) - 1.0) if math.isfinite(float(r["median_calibration_slope"])) else 999,
        )
    )
    for idx, row in enumerate(out, start=1):
        row["calibration_rank"] = idx
        row["selected_calibration_contract"] = idx == 1
    return out


def incremental_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model = {row["model"]: row for row in summary}
    top = by_model.get("raw_hist_gradient_boosting_shallow", {})
    rows = []
    for baseline in ["global_regularized_logistic", "fold_safe_platt_hgb", "fold_safe_isotonic_hgb"]:
        comp = by_model.get(baseline, {})
        if not top or not comp:
            continue
        rows.append(
            {
                "top_model": "raw_hist_gradient_boosting_shallow",
                "comparator": baseline,
                "median_brier_skill_difference": float(top["median_brier_skill"]) - float(comp["median_brier_skill"]),
                "median_pr_auc_lift_difference": float(top["median_pr_auc_lift"]) - float(comp["median_pr_auc_lift"]),
                "median_log_loss_improvement_difference": float(top["median_log_loss_improvement"]) - float(comp["median_log_loss_improvement"]),
                "worst_fold_brier_skill_difference": float(top["worst_fold_brier_skill"]) - float(comp["worst_fold_brier_skill"]),
                "meaningful_incremental_value": (float(top["worst_fold_brier_skill"]) - float(comp["worst_fold_brier_skill"])) > 0.001,
            }
        )
    return rows


def main() -> int:
    scout_dir = Path(os.getenv("RAWSEQ_1M_SCOUT_DIR", "").strip() or DEFAULT_SCOUT_DIR)
    out_root = env_path("RAWSEQ_1M_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    out_dir = Path(os.getenv("RAWSEQ_1M_CALIBRATION_OUTPUT_DIR", "").strip() or out_root / f"rawseq_1m_calibration_audit_{now_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=False)
    source_manifest = read_json(scout_dir / "source_manifest.json")
    symbol = source_manifest.get("symbol", os.getenv("RAWSEQ_1M_SYMBOL", "SOLUSDT"))
    source_raw = str(source_manifest.get("rawseq_1m_source_path", "") or os.getenv("RAWSEQ_1M_SOURCE_PATH", "")).strip()
    source_path = Path(source_raw) if source_raw else PROJECT_ROOT / "data" / "binance_public_zips"
    frame = load_candles(resolve_source_files(source_path, symbol))
    features, _, _ = build_features(frame, [15, 30, 60, 240])
    targets = downside_event_targets(frame, vol_window=VOL_WINDOW_MINUTES, horizons=[HORIZON_MINUTES])
    target_col = f"downside_event_0p5vol_h{HORIZON_MINUTES}m_fw{VOL_WINDOW_MINUTES}"
    features[f"trailing_volatility_bps_fw{VOL_WINDOW_MINUTES}"] = targets[f"trailing_volatility_bps_fw{VOL_WINDOW_MINUTES}"]
    feature_cols = [col for col in features.columns if col not in {"timestamp", "timestamp_ms"}]
    folds = read_csv_rows(scout_dir / "rolling_fold_manifest.csv")
    metric_rows: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    for fold in folds:
        rows, rel = evaluate_fold(fold, features, targets[target_col], feature_cols)
        metric_rows.extend(rows)
        reliability.extend(rel)
    summary = aggregate(metric_rows)
    increment = incremental_rows(summary)
    selected = next((row for row in summary if row.get("selected_calibration_contract")), {})
    contract = {
        "created_at": now_stamp(),
        "scout_dir": str(scout_dir),
        "horizon_minutes": HORIZON_MINUTES,
        "vol_window_minutes": VOL_WINDOW_MINUTES,
        "selected_calibration_contract": selected.get("model", ""),
        "selection_rule": "positive worst-fold Brier skill, median Brier skill, log-loss improvement, calibration stability, simplicity; development folds only",
        "holdout_accessed": False,
        "calibration_audit_sha256": "",
        **SAFETY_FLAGS,
    }
    contract["calibration_audit_sha256"] = stable_hash(contract)
    write_csv(out_dir / "calibration_fold_metrics.csv", metric_rows)
    write_csv(out_dir / "calibration_reliability_bins.csv", reliability)
    write_csv(out_dir / "calibration_leaderboard.csv", summary)
    write_csv(out_dir / "incremental_value_attribution.csv", increment)
    write_json(out_dir / "calibration_audit_contract.json", contract)
    lines = [
        "Rawseq 1m calibration audit",
        f"Output: {out_dir}",
        f"Scout dir: {scout_dir}",
        f"Leading contract: horizon={HORIZON_MINUTES}m vol_window={VOL_WINDOW_MINUTES}m",
        f"Selected calibration contract: {contract['selected_calibration_contract']}",
        "Calibration slope formula: outcome = slope * logit(prediction) + intercept.",
        "Holdout accessed: false",
        "",
        "Top calibration rows:",
    ]
    for row in summary[:5]:
        lines.append(
            f"- {row['model']}: worst_brier_skill={row['worst_fold_brier_skill']} "
            f"median_brier_skill={row['median_brier_skill']} median_log_loss_improvement={row['median_log_loss_improvement']}"
        )
    lines.append("Safety: CPU-only, public recorded data only, no orders, no promotion, no champion mutation.")
    (out_dir / "calibration_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
