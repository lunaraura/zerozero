#!/usr/bin/env python3
"""One-minute rawseq baseline scout report.

This script audits public Binance-style one-minute candles, defines split locks,
builds causal feature/target diagnostics, runs a bounded CPU downside-risk
baseline tournament for elapsed-time horizons, and writes research-only reports.

It does not train Torch/rawseq models, inspect final historical holdout metrics,
reuse frozen weights/scalers/thresholds, mutate future-shadow artifacts, place
orders, promote models, or create champions.
"""

from __future__ import annotations

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

from scripts.tiny.build_rawseq_1m_temporal_contract_grid import build_grid
from scripts.tiny.rawseq_1m_baseline_utils import (
    DEFAULT_OUTPUT_ROOT,
    SAFETY_FLAGS,
    audit_candles,
    build_features,
    canonical_column_contract,
    downside_event_targets,
    env_path,
    load_candles,
    metric_row,
    now_stamp,
    parse_bool,
    parse_int_list,
    resolve_source_files,
    save_reload_prediction_parity,
    source_manifest,
    split_contract,
    stable_hash,
    write_csv,
    write_json,
)


def cap_indices(indices: np.ndarray, max_rows: int, mode: str) -> np.ndarray:
    if max_rows <= 0 or len(indices) <= max_rows:
        return indices
    if mode == "tail":
        return indices[-max_rows:]
    return indices[:max_rows]


def finite_xy(features: pd.DataFrame, target: pd.Series, indices: np.ndarray, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = features.iloc[indices][feature_cols].to_numpy(dtype=np.float64)
    y = pd.to_numeric(target.iloc[indices], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[mask], y[mask], indices[mask]


def fit_logistic_predict(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, c: float = 1.0) -> tuple[np.ndarray, Any, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(train_y)) < 2:
        prob = float(np.mean(train_y)) if len(train_y) else 0.5
        return np.full(len(val_x), prob), {"constant_probability": prob}, lambda model, x: np.full(len(x), model["constant_probability"])
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=c, max_iter=300, solver="lbfgs"),
    )
    model.fit(train_x, train_y.astype(int))
    return model.predict_proba(val_x)[:, 1], model, lambda fitted, x: fitted.predict_proba(x)[:, 1]


def fit_hgb_predict(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, Any, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    if len(np.unique(train_y)) < 2:
        prob = float(np.mean(train_y)) if len(train_y) else 0.5
        return np.full(len(val_x), prob), {"constant_probability": prob}, lambda model, x: np.full(len(x), model["constant_probability"])
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=0.01, random_state=1337),
    )
    model.fit(train_x, train_y.astype(int))
    return model.predict_proba(val_x)[:, 1], model, lambda fitted, x: fitted.predict_proba(x)[:, 1]


def evaluate_cpu_contract(
    features: pd.DataFrame,
    target: pd.Series,
    folds: list[dict[str, Any]],
    feature_cols: list[str],
    vol_col: str,
    horizon: int,
    vol_window: int,
    max_train_rows: int,
    max_val_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    for fold in folds:
        train_idx = np.arange(fold["train_start_index"], fold["train_end_index"] + 1, dtype=np.int64)
        val_idx = np.arange(fold["validation_start_index"], fold["validation_end_index"] + 1, dtype=np.int64)
        train_idx = cap_indices(train_idx, max_train_rows, "tail")
        val_idx = cap_indices(val_idx, max_val_rows, "head")
        train_x, train_y, _ = finite_xy(features, target, train_idx, feature_cols)
        val_x, val_y, val_source_idx = finite_xy(features, target, val_idx, feature_cols)
        if len(train_y) < 100 or len(val_y) < 100:
            fold_rows.append(
                {
                    "horizon_minutes": horizon,
                    "vol_window_minutes": vol_window,
                    "fold_id": fold["fold_id"],
                    "model": "all",
                    "status": "insufficient_rows",
                    "train_rows": len(train_y),
                    "validation_rows": len(val_y),
                    "holdout_accessed": False,
                    **SAFETY_FLAGS,
                }
            )
            continue
        prevalence = float(np.mean(train_y))
        baseline = np.full(len(val_y), np.clip(prevalence, 1e-6, 1 - 1e-6))
        model_preds: dict[str, np.ndarray] = {"constant_training_prevalence": baseline}
        parity: dict[str, tuple[bool, float]] = {"constant_training_prevalence": (True, 0.0)}
        model_preds["volatility_only_logistic"], model, predict_fn = fit_logistic_predict(
            train_x[:, [feature_cols.index(vol_col)]], train_y, val_x[:, [feature_cols.index(vol_col)]]
        )
        parity["volatility_only_logistic"] = save_reload_prediction_parity(model, predict_fn, val_x[:, [feature_cols.index(vol_col)]])
        model_preds["global_regularized_logistic"], model, predict_fn = fit_logistic_predict(train_x, train_y, val_x)
        parity["global_regularized_logistic"] = save_reload_prediction_parity(model, predict_fn, val_x)
        regime_cols = [col for col in feature_cols if any(tok in col for tok in ["volatility", "range", "ema_slope", "volume"])]
        regime_idx = [feature_cols.index(col) for col in regime_cols] or list(range(train_x.shape[1]))
        model_preds["regime_feature_logistic"], model, predict_fn = fit_logistic_predict(train_x[:, regime_idx], train_y, val_x[:, regime_idx])
        parity["regime_feature_logistic"] = save_reload_prediction_parity(model, predict_fn, val_x[:, regime_idx])
        try:
            model_preds["hist_gradient_boosting_shallow"], model, predict_fn = fit_hgb_predict(train_x, train_y, val_x)
            parity["hist_gradient_boosting_shallow"] = save_reload_prediction_parity(model, predict_fn, val_x)
        except Exception as exc:
            fold_rows.append(
                {
                    "horizon_minutes": horizon,
                    "vol_window_minutes": vol_window,
                    "fold_id": fold["fold_id"],
                    "model": "hist_gradient_boosting_shallow",
                    "status": "skipped",
                    "skip_reason": repr(exc),
                    "train_rows": len(train_y),
                    "validation_rows": len(val_y),
                    "holdout_accessed": False,
                    **SAFETY_FLAGS,
                }
            )
        finite_model_names = [name for name, pred in model_preds.items() if name != "constant_training_prevalence" and np.isfinite(pred).all()]
        if finite_model_names:
            model_preds["conservative_probability_average"] = np.mean([model_preds[name] for name in finite_model_names], axis=0)
            parity["conservative_probability_average"] = (True, 0.0)
        for model_name, pred in model_preds.items():
            metrics = metric_row(val_y, pred, baseline)
            pass_parity, max_diff = parity.get(model_name, (False, math.nan))
            row = {
                "horizon_minutes": horizon,
                "vol_window_minutes": vol_window,
                "fold_id": fold["fold_id"],
                "model": model_name,
                "status": "OK",
                "train_rows": len(train_y),
                "validation_rows": len(val_y),
                "training_prevalence": prevalence,
                "save_reload_prediction_parity": pass_parity,
                "save_reload_prediction_max_abs_diff": max_diff,
                "holdout_accessed": False,
                **metrics,
                **SAFETY_FLAGS,
            }
            fold_rows.append(row)
            val_times = pd.to_datetime(features.iloc[val_source_idx]["timestamp_ms"], unit="ms", utc=True)
            for year in sorted(set(val_times.dt.year.dropna())):
                mask = val_times.dt.year.to_numpy() == year
                if int(mask.sum()) < 25:
                    continue
                y_year = val_y[mask]
                p_year = pred[mask]
                b_year = np.full(len(y_year), prevalence)
                year_metrics = metric_row(y_year, p_year, b_year)
                yearly_rows.append(
                    {
                        "horizon_minutes": horizon,
                        "vol_window_minutes": vol_window,
                        "fold_id": fold["fold_id"],
                        "model": model_name,
                        "calendar_year": int(year),
                        **year_metrics,
                        "holdout_accessed": False,
                    }
                )
    return fold_rows, yearly_rows


def aggregate_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [row for row in rows if row.get("status") == "OK"]
    grouped: dict[tuple[Any, Any, str], list[dict[str, Any]]] = {}
    for row in ok:
        grouped.setdefault((row["horizon_minutes"], row["vol_window_minutes"], row["model"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (horizon, vol_window, model), vals in grouped.items():
        skills = [float(row["brier_skill_vs_prevalence"]) for row in vals if math.isfinite(float(row["brier_skill_vs_prevalence"]))]
        pr_lifts = [float(row["pr_auc_lift_over_event_prevalence"]) for row in vals if math.isfinite(float(row["pr_auc_lift_over_event_prevalence"]))]
        fold_wins = sum(1 for value in skills if value > 0)
        row = {
            "horizon_minutes": horizon,
            "vol_window_minutes": vol_window,
            "model": model,
            "folds": len(vals),
            "fold_wins": fold_wins,
            "fold_win_fraction": fold_wins / len(vals) if vals else math.nan,
            "median_fold_brier_skill": float(np.median(skills)) if skills else math.nan,
            "worst_fold_brier_skill": float(np.min(skills)) if skills else math.nan,
            "median_pr_auc_lift": float(np.median(pr_lifts)) if pr_lifts else math.nan,
            "calibration_metrics_finite": all(math.isfinite(float(row.get("calibration_slope", math.nan))) or row["model"] == "constant_training_prevalence" for row in vals),
            "save_reload_parity_all_folds": all(str(row.get("save_reload_prediction_parity")).lower() in {"true", "1"} for row in vals),
            "holdout_accessed": False,
            **SAFETY_FLAGS,
        }
        row["advance_gate_pass"] = (
            row["median_fold_brier_skill"] > 0
            and row["fold_win_fraction"] >= 0.60
            and row["median_pr_auc_lift"] > 0
            and row["save_reload_parity_all_folds"]
        )
        out.append(row)
    out.sort(
        key=lambda r: (
            not bool(r["advance_gate_pass"]),
            -float(r["median_fold_brier_skill"]) if math.isfinite(float(r["median_fold_brier_skill"])) else 999,
            -float(r["fold_win_fraction"]) if math.isfinite(float(r["fold_win_fraction"])) else 999,
        )
    )
    return out


def recommendation_from_outputs(audit: dict[str, Any], folds: list[dict[str, Any]], leaderboard: list[dict[str, Any]]) -> str:
    if audit["total_rows"] <= 0 or not folds:
        return "one_minute_source_blocked"
    winners = [row for row in leaderboard if row.get("advance_gate_pass")]
    if not winners:
        return "no_one_minute_baseline_edge"
    if any(row["model"] != "volatility_only_logistic" for row in winners):
        return "continue_one_minute_cpu_baseline_research"
    return "continue_one_minute_cpu_baseline_research"


def main() -> int:
    source_path = env_path("RAWSEQ_1M_SOURCE_PATH", required=True)
    symbol = os.getenv("RAWSEQ_1M_SYMBOL", "SOLUSDT").strip() or "SOLUSDT"
    venue = os.getenv("RAWSEQ_1M_VENUE", "binance_public").strip() or "unknown"
    out_root = env_path("RAWSEQ_1M_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    out_dir = Path(os.getenv("RAWSEQ_1M_SCOUT_OUTPUT_DIR", "").strip() or out_root / f"rawseq_1m_baseline_scout_{now_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=False)
    feature_windows = parse_int_list(os.getenv("RAWSEQ_1M_FEATURE_WINDOWS", ""), [15, 30, 60, 240])
    horizons = parse_int_list(os.getenv("RAWSEQ_1M_DOWNSIDE_HORIZONS_MINUTES", ""), [1, 2, 4, 8])
    max_rows = int(os.getenv("RAWSEQ_1M_MAX_ROWS", "0") or "0")
    max_folds = int(os.getenv("RAWSEQ_1M_MAX_FOLDS", "4") or "4")
    cpu_max_contracts = int(os.getenv("RAWSEQ_1M_CPU_MAX_CONTRACTS", "16") or "16")
    max_train_rows = int(os.getenv("RAWSEQ_1M_CPU_MAX_TRAIN_ROWS_PER_FOLD", "100000") or "100000")
    max_val_rows = int(os.getenv("RAWSEQ_1M_CPU_MAX_VAL_ROWS_PER_FOLD", "100000") or "100000")
    run_cpu = parse_bool(os.getenv("RAWSEQ_1M_RUN_CPU_BASELINES", "true"))

    source_files = resolve_source_files(source_path, symbol)
    frame = load_candles(source_files, max_rows=max_rows)
    audit = audit_candles(frame, source_files, symbol=symbol, venue=venue)
    audit["scout_output_dir"] = str(out_dir)
    audit["rawseq_1m_source_path"] = str(source_path)
    audit["max_rows"] = max_rows
    manifest = source_manifest(source_files)
    contract = canonical_column_contract(symbol, venue)
    grid_rows = build_grid()
    features, feature_audit_rows, leakage = build_features(frame, feature_windows)
    split_manifest, fold_rows, purge_rows = split_contract(
        frame,
        feature_lookback=max(feature_windows),
        max_horizon=max(horizons),
        fold_count=max_folds,
    )
    holdout_lock = {
        "holdout_accessed": False,
        "holdout_start_index": split_manifest["holdout_start_index"],
        "holdout_end_index": split_manifest["holdout_end_index"],
        "holdout_rows": split_manifest["untouched_holdout_rows"],
        "selection_uses_holdout": False,
        **SAFETY_FLAGS,
    }

    write_json(out_dir / "candle_audit.json", audit)
    write_csv(out_dir / "candle_audit.csv", [audit])
    write_json(out_dir / "source_manifest.json", manifest)
    write_json(out_dir / "canonical_column_contract.json", contract)
    write_csv(out_dir / "one_minute_contract_grid.csv", grid_rows)
    write_json(out_dir / "feature_contract.json", {"feature_windows": feature_windows, "features": list(features.columns), **SAFETY_FLAGS})
    write_csv(out_dir / "feature_audit.csv", feature_audit_rows)
    write_json(out_dir / "leakage_audit.json", leakage)
    years = features.assign(year=features["timestamp"].dt.year)
    yearly_rows = []
    for year, group in years.groupby("year", dropna=True):
        row = {"calendar_year": int(year), "rows": len(group)}
        for col in [c for c in features.columns if c not in {"timestamp", "timestamp_ms"}][:20]:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else math.nan
            row[f"{col}_std"] = float(vals.std(ddof=0)) if vals.notna().any() else math.nan
        yearly_rows.append(row)
    write_csv(out_dir / "yearly_feature_distribution.csv", yearly_rows)
    write_csv(out_dir / "feature_drift_by_year.csv", yearly_rows)
    write_json(out_dir / "split_manifest.json", split_manifest)
    write_csv(out_dir / "rolling_fold_manifest.csv", fold_rows)
    write_csv(out_dir / "purge_embargo_audit.csv", purge_rows)
    write_json(out_dir / "holdout_lock.json", holdout_lock)

    cpu_fold_metrics: list[dict[str, Any]] = []
    yearly_regime_rows: list[dict[str, Any]] = []
    target_prevalence_rows: list[dict[str, Any]] = []
    audit_lock_pass = audit["total_rows"] > 0 and audit["ohlc_consistency_violations"] == 0 and audit["nonpositive_prices"] == 0
    split_lock_pass = bool(fold_rows) and not holdout_lock["holdout_accessed"]
    if run_cpu and audit_lock_pass and split_lock_pass:
        feature_cols = [col for col in features.columns if col not in {"timestamp", "timestamp_ms"}]
        contracts_run = 0
        for vol_window in feature_windows:
            targets = downside_event_targets(frame, vol_window=vol_window, horizons=horizons)
            features_for_contract = features.copy()
            vol_col = f"trailing_volatility_bps_fw{vol_window}"
            features_for_contract[vol_col] = targets[vol_col]
            this_features = feature_cols + [vol_col]
            for horizon in horizons:
                if contracts_run >= cpu_max_contracts:
                    break
                target_col = f"downside_event_0p5vol_h{horizon}m_fw{vol_window}"
                target = targets[target_col]
                finite_target = pd.to_numeric(target, errors="coerce")
                for year, group in frame.assign(target=finite_target, year=frame["timestamp"].dt.year).groupby("year", dropna=True):
                    vals = pd.to_numeric(group["target"], errors="coerce").dropna()
                    if len(vals):
                        target_prevalence_rows.append(
                            {
                                "horizon_minutes": horizon,
                                "vol_window_minutes": vol_window,
                                "calendar_year": int(year),
                                "rows": int(len(vals)),
                                "event_prevalence": float(vals.mean()),
                            }
                        )
                rows, year_rows = evaluate_cpu_contract(
                    features_for_contract,
                    target,
                    fold_rows,
                    this_features,
                    vol_col,
                    horizon,
                    vol_window,
                    max_train_rows=max_train_rows,
                    max_val_rows=max_val_rows,
                )
                cpu_fold_metrics.extend(rows)
                yearly_regime_rows.extend(year_rows)
                contracts_run += 1
            if contracts_run >= cpu_max_contracts:
                break
    cpu_leaderboard = aggregate_leaderboard(cpu_fold_metrics)
    recommendation = recommendation_from_outputs(audit, fold_rows, cpu_leaderboard)
    rawseq_rows = [
        {
            "status": "not_run_in_initial_cpu_scout",
            "reason": "rawseq architecture search requires explicit follow-up after audit/split/CPU review",
            "holdout_accessed": False,
            **SAFETY_FLAGS,
        }
    ]
    comparison = {
        "recommendation": recommendation,
        "bounded_smoke_mode": max_rows > 0 or cpu_max_contracts < 16 or max_folds < 4,
        "max_rows": max_rows,
        "max_folds": max_folds,
        "cpu_max_contracts": cpu_max_contracts,
        "audit_lock_pass": audit_lock_pass,
        "split_lock_pass": split_lock_pass,
        "holdout_accessed": False,
        "gpu_used": False,
        "cpu_baselines_run": run_cpu and audit_lock_pass and split_lock_pass,
        "cpu_fold_rows": len(cpu_fold_metrics),
        "cpu_leaderboard_rows": len(cpu_leaderboard),
        "top_cpu_contract": cpu_leaderboard[0] if cpu_leaderboard else {},
        "rawseq_search_status": rawseq_rows[0]["status"],
        **SAFETY_FLAGS,
    }
    write_csv(out_dir / "cpu_downside_risk_fold_metrics.csv", cpu_fold_metrics)
    write_csv(out_dir / "cpu_downside_risk_leaderboard.csv", cpu_leaderboard)
    write_csv(out_dir / "rawseq_path_fold_metrics.csv", rawseq_rows)
    write_csv(out_dir / "rawseq_path_leaderboard.csv", rawseq_rows)
    write_csv(out_dir / "yearly_regime_metrics.csv", yearly_regime_rows)
    write_csv(out_dir / "target_prevalence_by_year.csv", target_prevalence_rows)
    write_json(out_dir / "baseline_comparison.json", comparison)
    write_csv(out_dir / "recommended_next_contracts.csv", [row for row in cpu_leaderboard if row.get("advance_gate_pass")])
    strongest_horizon = ""
    if cpu_leaderboard:
        strongest_horizon = str(cpu_leaderboard[0]["horizon_minutes"])
    lines = [
        "Rawseq 1m baseline scout",
        f"Output: {out_dir}",
        f"Source: {source_path}",
        f"Rows loaded: {audit['total_rows']}",
        f"Coverage months: {audit['approximate_months_covered']:.3f}",
        f"Bounded smoke mode: {comparison['bounded_smoke_mode']}",
        f"Caps: max_rows={max_rows}, max_folds={max_folds}, cpu_max_contracts={cpu_max_contracts}",
        f"Audit status: {audit['audit_status']}",
        f"Split lock pass: {split_lock_pass}",
        f"Holdout accessed: {holdout_lock['holdout_accessed']}",
        "",
        "Questions:",
        "1. Is the 0.5-vol downside event predictable at one-minute cadence? "
        + (
            "smoke-positive only; not established until uncapped rolling folds pass"
            if comparison["bounded_smoke_mode"] and any(row.get("advance_gate_pass") for row in cpu_leaderboard)
            else ("yes, preliminarily across configured folds" if any(row.get("advance_gate_pass") for row in cpu_leaderboard) else "not established")
        ),
        f"2. Strongest elapsed-time horizon among tested CPU rows: {strongest_horizon or 'none'}",
        "3. Same-shape 60x60 rawseq contract beat simple path baselines? not run in this initial CPU scout",
        "4. Compact 60-input / 8-output contract performs better? not run in this initial CPU scout",
        f"5. Stable across years/regimes? {'see yearly_regime_metrics.csv' if yearly_regime_rows else 'not established'}",
        "6. Stronger than constant prevalence / volatility-only? see cpu_downside_risk_leaderboard.csv",
        "7. Concentrated in a small period? see yearly_regime_metrics.csv and target_prevalence_by_year.csv",
        "8. Rejected contracts: contracts failing advance_gate_pass in cpu_downside_risk_leaderboard.csv",
        "9. Larger development run candidates: recommended_next_contracts.csv",
        f"10. Untouched historical holdout accessed? {holdout_lock['holdout_accessed']}",
        "",
        f"Final recommendation: {recommendation}",
        "Safety: paper_only=true, private_api=false, orders=false, promotion=false, champion_mutation=false.",
    ]
    (out_dir / "baseline_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if audit_lock_pass and split_lock_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
