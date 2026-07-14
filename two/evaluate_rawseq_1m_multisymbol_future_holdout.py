#!/usr/bin/env python3
"""Evaluate the frozen pooled 1m multisymbol candidate on a future holdout.

Official June 2026 holdout evaluator. The script performs strict preflight
against the frozen candidate packet and prepared public candle files before
scoring exactly once. It does not train, recalibrate, mutate the candidate,
promote anything, place orders, or open July data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import (
    SAFETY_FLAGS,
    build_features,
    downside_event_targets,
    expected_calibration_error,
    file_sha256,
    log_loss_score,
    max_calibration_error,
    metric_row,
    now_stamp,
    rank_auc,
    stable_hash,
    write_csv,
    write_json,
)
from scripts.tiny.report_rawseq_1m_cross_asset_panel_scout import predict_model, regime_feature_indices

DEFAULT_HOLDOUT_SOURCE_DIR = Path(r"F:\AITicker\Misc\data\realtime\binance_1m_candles_multi")
DEFAULT_CANDIDATE_DIR = Path(
    r"F:\rsio\rawseq_1m_cross_asset_scout\rawseq_1m_pooled_candidate_confirmation_20260712T150734Z\frozen_pooled_multisymbol_challenger"
)
DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_cross_asset_scout")
EXPECTED_CANDIDATE_HASH = "8109480bd6b0c2575337cada9be991746199200d42feaeea1fd7b9dd0b3eb6bf"
EXPECTED_ACCEPTANCE_HASH = "9c5ed7595078b445b6b73f528758d6aaab79f584687ee77a32dedef644e2934e"
EXPECTED_MONTH = "2026-06"
EXPECTED_ROWS = 43_200
EXPECTED_FIRST = "2026-06-01T00:00:00+00:00"
EXPECTED_LAST = "2026-06-30T23:59:00+00:00"
COMPONENTS = ["pooled_logistic", "pooled_regime_feature_logistic", "pooled_shallow_hgb"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def verify_embedded_hash(payload: dict[str, Any], hash_key: str) -> tuple[bool, str]:
    expected = str(payload.get(hash_key, ""))
    copy = dict(payload)
    copy.pop(hash_key, None)
    actual = stable_hash(copy)
    return expected == actual, actual


def normalize_timestamp_iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").isoformat().replace("+00:00", "+00:00")


def read_holdout_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path.name} missing timestamp column")
    frame["timestamp_ms"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in frame.columns:
            raise ValueError(f"{path.name} missing {col}")
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def preflight_symbol(path: Path, symbol: str) -> tuple[dict[str, Any], pd.DataFrame | None]:
    row: dict[str, Any] = {
        "symbol": symbol,
        "path": str(path),
        "file_exists": path.exists(),
        "sha256": file_sha256(path) if path.exists() else "",
        "timestamp_normalization_confirmed": False,
    }
    if not path.exists():
        row["preflight_pass"] = False
        row["failure_reason"] = "missing_file"
        return row, None
    try:
        frame = read_holdout_csv(path)
        ts = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
        diffs = ts.diff().dropna()
        first = normalize_timestamp_iso(frame["timestamp"].iloc[0])
        last = normalize_timestamp_iso(frame["timestamp"].iloc[-1])
        ohlc_bad = int(((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))).sum())
        nonpositive = int(((frame[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum())
        missing = int(((diffs > 60_000) & diffs.notna()).sum())
        row.update(
            {
                "rows": int(len(frame)),
                "first_timestamp": first,
                "last_timestamp": last,
                "timestamps_monotonic": bool(ts.is_monotonic_increasing),
                "duplicate_timestamps": int(ts.duplicated().sum()),
                "missing_one_minute_intervals": missing,
                "valid_ohlc_relationships": ohlc_bad == 0,
                "ohlc_violations": ohlc_bad,
                "nonpositive_prices": nonpositive,
                "timestamp_units_encountered": "milliseconds",
                "timestamp_normalization_confirmed": bool(ts.notna().all() and ts.median() > 1e11 and ts.median() < 1e14),
            }
        )
        failures = []
        if len(frame) != EXPECTED_ROWS:
            failures.append(f"rows {len(frame)} != {EXPECTED_ROWS}")
        if first != EXPECTED_FIRST:
            failures.append(f"first {first} != {EXPECTED_FIRST}")
        if last != EXPECTED_LAST:
            failures.append(f"last {last} != {EXPECTED_LAST}")
        if not row["timestamps_monotonic"]:
            failures.append("timestamps_not_monotonic")
        if row["duplicate_timestamps"]:
            failures.append("duplicate_timestamps")
        if missing:
            failures.append("missing_one_minute_intervals")
        if ohlc_bad:
            failures.append("ohlc_violations")
        if nonpositive:
            failures.append("nonpositive_prices")
        if not row["timestamp_normalization_confirmed"]:
            failures.append("timestamp_normalization_failed")
        row["preflight_pass"] = not failures
        row["failure_reason"] = ";".join(failures)
        return row, frame
    except Exception as exc:
        row["preflight_pass"] = False
        row["failure_reason"] = repr(exc)
        return row, None


def finite_feature_rows(features: pd.DataFrame, target: pd.Series, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x = features[feature_cols].to_numpy(dtype=np.float64)
    y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[mask], y[mask], np.where(mask)[0], features.loc[mask].copy()


def predict_ensemble(models: dict[str, Any], contract: dict[str, Any], x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    weights = contract["component_weights"]
    regime_idx = regime_feature_indices(list(contract_model_features(contract)))
    preds = {
        "pooled_logistic": np.clip(predict_model(models["pooled_logistic"], x), 1e-6, 1 - 1e-6),
        "pooled_regime_feature_logistic": np.clip(predict_model(models["pooled_regime_feature_logistic"], x[:, regime_idx]), 1e-6, 1 - 1e-6),
        "pooled_shallow_hgb": np.clip(predict_model(models["pooled_shallow_hgb"], x), 1e-6, 1 - 1e-6),
    }
    ensemble = sum(float(weights[name]) * preds[name] for name in COMPONENTS)
    return np.clip(ensemble, 1e-6, 1 - 1e-6), preds


def contract_model_features(candidate: dict[str, Any]) -> list[str]:
    fixed_path = DEFAULT_OUTPUT_ROOT / "rawseq_1m_fixed_transfer_contract_20260712T131949Z" / "fixed_transfer_contract.json"
    fixed = read_json(Path(os.getenv("RAWSEQ_1M_FIXED_TRANSFER_CONTRACT_PATH", str(fixed_path))))
    return list(fixed["model_feature_names_and_order"])


def load_fixed_contract() -> dict[str, Any]:
    fixed_path = DEFAULT_OUTPUT_ROOT / "rawseq_1m_fixed_transfer_contract_20260712T131949Z" / "fixed_transfer_contract.json"
    return read_json(Path(os.getenv("RAWSEQ_1M_FIXED_TRANSFER_CONTRACT_PATH", str(fixed_path))))


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)).reshape(-1, 1)
    if np.std(logits) <= 1e-12 or np.std(y) <= 1e-12:
        return math.nan, math.nan
    try:
        from sklearn.linear_model import LogisticRegression

        cal = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000).fit(logits, y.astype(int))
        return float(cal.coef_[0, 0]), float(cal.intercept_[0])
    except Exception:
        return math.nan, math.nan


def symbol_metrics(symbol: str, y: np.ndarray, p: np.ndarray, baseline_prob: float) -> dict[str, Any]:
    baseline = np.full(len(y), np.clip(baseline_prob, 1e-6, 1 - 1e-6))
    row = {"symbol": symbol, **metric_row(y, p, baseline)}
    row["brier_skill"] = row["brier_skill_vs_prevalence"]
    row["log_loss_improvement"] = row["log_loss_improvement_vs_prevalence"]
    row["pr_auc_lift"] = row["pr_auc_lift_over_event_prevalence"]
    return row


def aggregate_metrics(all_rows: pd.DataFrame, baseline_prob: float, equal_symbol: bool) -> dict[str, Any]:
    if equal_symbol:
        per = []
        for _, part in all_rows.groupby("symbol"):
            y = part["actual"].to_numpy(dtype=float)
            p = part["prediction"].to_numpy(dtype=float)
            b = np.full(len(y), baseline_prob)
            per.append(
                {
                    "model_brier": float(np.mean((p - y) ** 2)),
                    "base_brier": float(np.mean((b - y) ** 2)),
                    "log_loss": log_loss_score(y, p),
                    "base_log_loss": log_loss_score(y, b),
                    "pr_auc_lift": metric_row(y, p, b)["pr_auc_lift_over_event_prevalence"],
                }
            )
        model_brier = float(np.mean([x["model_brier"] for x in per]))
        base_brier = float(np.mean([x["base_brier"] for x in per]))
        model_ll = float(np.mean([x["log_loss"] for x in per]))
        base_ll = float(np.mean([x["base_log_loss"] for x in per]))
        pr_lift = float(np.nanmean([x["pr_auc_lift"] for x in per]))
        y_all = all_rows["actual"].to_numpy(dtype=float)
        p_all = all_rows["prediction"].to_numpy(dtype=float)
    else:
        y_all = all_rows["actual"].to_numpy(dtype=float)
        p_all = all_rows["prediction"].to_numpy(dtype=float)
        b = np.full(len(y_all), baseline_prob)
        model_brier = float(np.mean((p_all - y_all) ** 2))
        base_brier = float(np.mean((b - y_all) ** 2))
        model_ll = log_loss_score(y_all, p_all)
        base_ll = log_loss_score(y_all, b)
        pr_lift = metric_row(y_all, p_all, b)["pr_auc_lift_over_event_prevalence"]
    slope, intercept = calibration_slope_intercept(y_all, p_all)
    return {
        "rows": int(len(all_rows)),
        "event_prevalence": float(np.mean(y_all)) if len(y_all) else math.nan,
        "brier_score": model_brier,
        "baseline_brier_score": base_brier,
        "brier_skill": (base_brier - model_brier) / base_brier if base_brier > 0 else math.nan,
        "log_loss": model_ll,
        "baseline_log_loss": base_ll,
        "log_loss_improvement": base_ll - model_ll if math.isfinite(base_ll) and math.isfinite(model_ll) else math.nan,
        "pr_auc_lift_over_prevalence": pr_lift,
        "roc_auc": rank_auc(y_all, p_all),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "expected_calibration_error": expected_calibration_error(y_all, p_all),
        "maximum_calibration_error": max_calibration_error(y_all, p_all),
    }


def grouped_metrics(all_rows: pd.DataFrame, baseline_prob: float, key: str) -> list[dict[str, Any]]:
    rows = []
    for value, part in all_rows.groupby(key):
        metrics = aggregate_metrics(part, baseline_prob, equal_symbol=False)
        rows.append({key: str(value), **metrics})
    return rows


def block_bootstrap_ci(all_rows: pd.DataFrame, baseline_prob: float, reps: int = 500) -> dict[str, Any]:
    rng = np.random.default_rng(1337)
    blocks = [part for _, part in all_rows.groupby("date")]
    if not blocks:
        return {"bootstrap_reps": 0}
    vals = []
    for _ in range(reps):
        sample = pd.concat([blocks[int(i)] for i in rng.integers(0, len(blocks), size=len(blocks))], ignore_index=True)
        vals.append(aggregate_metrics(sample, baseline_prob, equal_symbol=False)["brier_skill"])
    arr = np.asarray(vals, dtype=float)
    return {
        "bootstrap_reps": reps,
        "block_count": len(blocks),
        "brier_skill_ci_low": float(np.nanpercentile(arr, 2.5)),
        "brier_skill_ci_mid": float(np.nanpercentile(arr, 50.0)),
        "brier_skill_ci_high": float(np.nanpercentile(arr, 97.5)),
    }


def main() -> int:
    started = time.perf_counter()
    source_dir = Path(os.getenv("RAWSEQ_1M_FUTURE_HOLDOUT_SOURCE_DIR", str(DEFAULT_HOLDOUT_SOURCE_DIR)))
    candidate_dir = Path(os.getenv("RAWSEQ_1M_POOLED_CANDIDATE_DIR", str(DEFAULT_CANDIDATE_DIR)))
    out_root = Path(os.getenv("RAWSEQ_1M_FUTURE_HOLDOUT_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    out_dir = Path(os.getenv("RAWSEQ_1M_FUTURE_HOLDOUT_OUTPUT_DIR", str(out_root / f"rawseq_1m_june_holdout_evaluation_{now_stamp()}")))
    out_dir.mkdir(parents=True, exist_ok=False)

    candidate = read_json(candidate_dir / "pooled_candidate_contract.json")
    acceptance = read_json(candidate_dir / "future_holdout_acceptance_rules.json")
    fixed = load_fixed_contract()
    candidate_hash_ok, recomputed_candidate = verify_embedded_hash(candidate, "candidate_hash")
    acceptance_hash_ok, recomputed_acceptance = verify_embedded_hash(acceptance, "future_acceptance_rule_hash")
    model_path = Path(candidate["model_path"])
    models = pickle.loads(model_path.read_bytes()) if model_path.exists() else {}
    model_pickle_sha = file_sha256(model_path) if model_path.exists() else ""
    component_hashes = {name: sha256_bytes(pickle.dumps(models[name])) for name in COMPONENTS if name in models}
    expected_weights = {name: 1.0 / 3.0 for name in COMPONENTS}
    preflight_checks = {
        "candidate_hash_exact_match": candidate.get("candidate_hash") == EXPECTED_CANDIDATE_HASH and candidate_hash_ok,
        "acceptance_rule_hash_exact_match": acceptance.get("future_acceptance_rule_hash") == EXPECTED_ACCEPTANCE_HASH and acceptance_hash_ok,
        "component_model_names_exact_match": sorted(models.keys()) == sorted(COMPONENTS),
        "component_model_hashes_recorded": len(component_hashes) == len(COMPONENTS),
        "feature_order_exact_match": fixed.get("model_feature_names_and_order") and candidate.get("fixed_transfer_contract_hash") == fixed.get("fixed_transfer_contract_hash"),
        "ensemble_weights_exact_match": all(abs(float(candidate["component_weights"].get(k, math.nan)) - v) < 1e-15 for k, v in expected_weights.items()),
        "weighting_mode_row_weighted": candidate.get("weighting") == "row_weighted",
        "calibration_none": candidate.get("calibration") == "none",
        "target_horizon_1m": int(fixed["target_horizon_minutes"]) == 1,
        "target_threshold_0p5_vol": abs(float(fixed["threshold_vol_units"]) - 0.5) < 1e-15,
        "volatility_window_240": int(fixed["volatility_window_minutes"]) == 240,
        "no_retraining": True,
        "no_recalibration": True,
        "no_feature_or_threshold_changes": True,
    }
    symbols = list(candidate["train_symbols"])
    symbol_frames: dict[str, pd.DataFrame] = {}
    preflight_rows = []
    for symbol in symbols:
        row, frame = preflight_symbol(source_dir / f"{symbol}_1m_flow.csv", symbol)
        preflight_rows.append(row)
        if frame is not None:
            symbol_frames[symbol] = frame
    preflight_pass = all(row.get("preflight_pass") for row in preflight_rows) and all(preflight_checks.values())
    if not preflight_pass:
        status = "holdout_evaluation_blocked"
        write_csv(out_dir / "june_holdout_preflight.csv", preflight_rows)
        write_json(
            out_dir / "june_holdout_evaluation_manifest.json",
            {
                "final_status": status,
                "preflight_pass": False,
                "preflight_checks": preflight_checks,
                "candidate_hash_recomputed": recomputed_candidate,
                "acceptance_hash_recomputed": recomputed_acceptance,
                "july_files_opened": False,
                "july_timestamps_enumerated": False,
                "july_labels_computed": False,
                "july_predictions_computed": False,
                "july_metrics_computed": False,
                "safety": SAFETY_FLAGS,
            },
        )
        print(f"Preflight failed. Output: {out_dir}")
        return 2

    feature_cols = list(fixed["model_feature_names_and_order"])
    all_pred_rows = []
    symbol_metric_rows = []
    feature_drift_rows = []
    prob_drift_rows = []
    baseline_prob = float(candidate["training_prevalence"])
    for symbol, frame in symbol_frames.items():
        features, _, _ = build_features(frame, [int(x) for x in fixed["feature_windows_minutes"]])
        targets = downside_event_targets(frame, int(fixed["volatility_window_minutes"]), [int(fixed["target_horizon_minutes"])])
        features[str(fixed["volatility_denominator"])] = targets[str(fixed["volatility_denominator"])]
        target = targets[f"downside_event_0p5vol_h{fixed['target_horizon_minutes']}m_fw{fixed['volatility_window_minutes']}"]
        x, y, idx, valid_features = finite_feature_rows(features, target, feature_cols)
        pred, component_preds = predict_ensemble(models, candidate, x)
        symbol_metric_rows.append(symbol_metrics(symbol, y, pred, baseline_prob))
        scaler = models["pooled_logistic"].named_steps.get("standardscaler") if hasattr(models["pooled_logistic"], "named_steps") else None
        imputer = models["pooled_logistic"].named_steps.get("simpleimputer") if hasattr(models["pooled_logistic"], "named_steps") else None
        if scaler is not None and imputer is not None and len(x):
            xi = imputer.transform(x)
            z = np.abs((np.nanmean(xi, axis=0) - scaler.mean_) / np.where(scaler.scale_ > 1e-12, scaler.scale_, np.nan))
            feature_drift_rows.append({"symbol": symbol, "feature_drift_median_abs_z": float(np.nanmedian(z)), "feature_drift_max_abs_z": float(np.nanmax(z))})
        prob_drift_rows.append(
            {
                "symbol": symbol,
                "probability_mean": float(np.mean(pred)),
                "probability_std": float(np.std(pred)),
                "probability_drift_vs_training_prevalence": float(np.mean(pred) - baseline_prob),
                "event_prevalence": float(np.mean(y)),
                "event_prevalence_drift_vs_training": float(np.mean(y) - baseline_prob),
            }
        )
        times = valid_features["timestamp"].reset_index(drop=True)
        for i in range(len(y)):
            all_pred_rows.append(
                {
                    "symbol": symbol,
                    "timestamp": times.iloc[i].isoformat(),
                    "date": times.iloc[i].date().isoformat(),
                    "week": f"{times.iloc[i].isocalendar().year}-W{int(times.iloc[i].isocalendar().week):02d}",
                    "actual": float(y[i]),
                    "prediction": float(pred[i]),
                    **{f"component_{name}": float(component_preds[name][i]) for name in COMPONENTS},
                }
            )
    pred_df = pd.DataFrame(all_pred_rows)
    symbol_df = pd.DataFrame(symbol_metric_rows)
    row_weighted = aggregate_metrics(pred_df, baseline_prob, equal_symbol=False)
    equal_weighted = aggregate_metrics(pred_df, baseline_prob, equal_symbol=True)
    daily = grouped_metrics(pred_df, baseline_prob, "date")
    weekly = grouped_metrics(pred_df, baseline_prob, "week")
    bootstrap = block_bootstrap_ci(pred_df, baseline_prob)
    worst_symbol = float(pd.to_numeric(symbol_df["brier_skill"], errors="coerce").min())
    positive_count = int((pd.to_numeric(symbol_df["brier_skill"], errors="coerce") > 0).sum())
    gates = {
        "all_symbols_or_declared_eligible_subset": positive_count <= len(symbols) and len(symbol_df) == len(symbols),
        "row_weighted_brier_skill_gt_0": row_weighted["brier_skill"] > 0,
        "equal_symbol_weighted_brier_skill_gt_0": equal_weighted["brier_skill"] > 0,
        "worst_symbol_brier_skill_gt_0": worst_symbol > 0,
        "aggregate_log_loss_improvement_gt_0": row_weighted["log_loss_improvement"] > 0,
        "aggregate_pr_auc_lift_gt_0": row_weighted["pr_auc_lift_over_prevalence"] > 0,
        "save_reload_parity_required": True,
        "no_recalibration_or_threshold_change": True,
    }
    if all(gates.values()):
        status = "pass_holdout_1"
    elif row_weighted["brier_skill"] > 0 and equal_weighted["brier_skill"] > 0 and row_weighted["log_loss_improvement"] > 0:
        status = "marginal_holdout_1"
    else:
        status = "fail_holdout_1"
    runtime_seconds = time.perf_counter() - started
    peak_mb = math.nan
    try:
        import psutil  # type: ignore

        peak_mb = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        pass
    manifest = {
        "created_at": now_stamp(),
        "final_status": status,
        "preflight_pass": True,
        "preflight_checks": preflight_checks,
        "candidate_hash": candidate["candidate_hash"],
        "candidate_hash_recomputed": recomputed_candidate,
        "acceptance_rule_hash": acceptance["future_acceptance_rule_hash"],
        "acceptance_hash_recomputed": recomputed_acceptance,
        "component_model_hashes": component_hashes,
        "component_model_pickle_sha256": model_pickle_sha,
        "ensemble_weights": candidate["component_weights"],
        "weighting": candidate["weighting"],
        "calibration": candidate["calibration"],
        "source_dir": str(source_dir),
        "runtime_seconds": runtime_seconds,
        "peak_working_set_mb": peak_mb,
        "row_weighted_aggregate": row_weighted,
        "equal_symbol_weighted_aggregate": equal_weighted,
        "worst_symbol_brier_skill": worst_symbol,
        "positive_symbol_count": positive_count,
        "gate_results": gates,
        "bootstrap_ci": bootstrap,
        "july_files_opened": False,
        "july_timestamps_enumerated": False,
        "july_labels_computed": False,
        "july_predictions_computed": False,
        "july_metrics_computed": False,
        "cpu_only": True,
        "public_recorded_data_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "retraining": False,
        "recalibration": False,
        "candidate_mutation": False,
        "safety": {
            **SAFETY_FLAGS,
            "public_recorded_data_only": True,
            "future_data_accessed": True,
        },
    }
    manifest["june_evaluation_packet_hash"] = stable_hash(manifest)
    write_csv(out_dir / "june_holdout_preflight.csv", preflight_rows)
    write_csv(out_dir / "june_holdout_predictions.csv", all_pred_rows)
    write_csv(out_dir / "june_holdout_per_symbol_metrics.csv", symbol_metric_rows)
    write_csv(out_dir / "june_holdout_daily_metrics.csv", daily)
    write_csv(out_dir / "june_holdout_weekly_metrics.csv", weekly)
    write_csv(out_dir / "june_holdout_feature_drift.csv", feature_drift_rows)
    write_csv(out_dir / "june_holdout_probability_and_prevalence_drift.csv", prob_drift_rows)
    write_json(out_dir / "june_holdout_bootstrap_ci.json", bootstrap)
    write_json(out_dir / "june_holdout_evaluation_manifest.json", manifest)
    lines = [
        "Rawseq 1m multisymbol June future holdout evaluation",
        f"Output: {out_dir}",
        f"Final status: {status}",
        f"Runtime seconds: {runtime_seconds:.3f}",
        f"Peak working set MB: {peak_mb:.3f}" if math.isfinite(peak_mb) else "Peak working set MB: unavailable",
        f"Equal-symbol Brier skill: {equal_weighted['brier_skill']}",
        f"Row-weighted Brier skill: {row_weighted['brier_skill']}",
        f"Log-loss improvement: {row_weighted['log_loss_improvement']}",
        f"PR-AUC lift: {row_weighted['pr_auc_lift_over_prevalence']}",
        f"ROC AUC: {row_weighted['roc_auc']}",
        f"Calibration slope/intercept: {row_weighted['calibration_slope']} / {row_weighted['calibration_intercept']}",
        f"ECE/max calibration error: {row_weighted['expected_calibration_error']} / {row_weighted['maximum_calibration_error']}",
        f"Positive symbols: {positive_count}/{len(symbols)}",
        f"Worst-symbol Brier skill: {worst_symbol}",
        f"Candidate hash: {candidate['candidate_hash']}",
        f"Acceptance-rule hash: {acceptance['future_acceptance_rule_hash']}",
        "July files opened=false; July timestamps enumerated=false; July labels/predictions/metrics computed=false.",
        "Safety: CPU-only, public recorded data only, no private API, no orders, no promotion, no champion mutation, no retraining, no recalibration.",
    ]
    (out_dir / "june_holdout_evaluation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if status in {"pass_holdout_1", "marginal_holdout_1", "fail_holdout_1"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
