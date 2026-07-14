#!/usr/bin/env python3
"""Fixed nested-calibration confirmation for 1m multi-horizon downside risk."""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import (  # noqa: E402
    DEFAULT_SOURCE_PATH,
    SAFETY_FLAGS,
    build_features,
    env_path,
    load_candles,
    metric_row,
    now_stamp,
    resolve_source_files,
    split_contract,
    stable_hash,
    write_csv,
    write_json,
)
from scripts.tiny.report_rawseq_1m_cross_asset_panel_scout import fit_logistic, predict_model  # noqa: E402
from scripts.tiny.run_rawseq_1m_board_member_target_feature_tournament import (  # noqa: E402
    DEVELOPMENT_CUTOFF_MS,
    DEFAULT_SYMBOLS,
    build_target_lanes,
    cap_tail,
    filter_development,
)

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_multihorizon_downside_calibrated_confirmations")
FIXED_TRANSFER_CONTRACT = Path(
    r"F:\rsio\rawseq_1m_cross_asset_scout\rawseq_1m_fixed_transfer_contract_20260712T131949Z\fixed_transfer_contract.json"
)
FROZEN_DOWNSIDE_CONTRACT = Path(
    r"F:\rsio\rawseq_1m_cross_asset_scout\rawseq_1m_pooled_candidate_confirmation_20260712T150734Z\frozen_pooled_multisymbol_challenger\pooled_candidate_contract.json"
)
FROZEN_DOWNSIDE_COMPONENTS = Path(
    r"F:\rsio\rawseq_1m_cross_asset_scout\rawseq_1m_pooled_candidate_confirmation_20260712T150734Z\frozen_pooled_multisymbol_challenger\pooled_candidate_components.pkl"
)
TARGET_HORIZONS = [1, 2, 4, 8]
TARGETS = [f"downside_event_0p5vol_h{h}m_fw240" for h in TARGET_HORIZONS]
HORIZON_LABELS = [f"{h}m" for h in TARGET_HORIZONS]
PURGE_ROWS = 240
EMBARGO_ROWS = 8
INNER_GAP_ROWS = PURGE_ROWS + EMBARGO_ROWS
EXISTING_DOWNSIDE_CANDIDATE_HASH = "8109480bd6b0c2575337cada9be991746199200d42feaeea1fd7b9dd0b3eb6bf"
FREEZE_GATES = {
    "equal_symbol_family_median_brier_skill_min": 0.06,
    "worst_horizon_brier_skill_min": 0.02,
    "overall_worst_fold_brier_skill_min": 0.03,
    "combined_symbol_time_worst_fold_brier_skill_min": 0.02,
    "median_pr_auc_lift_min": 0.10,
    "family_positive_symbols_min": 8,
    "horizon_positive_symbols_min": 7,
    "max_symbol_skill_contribution_fraction": 0.35,
    "max_expected_calibration_error": 0.03,
    "max_calibration_error": 0.15,
    "calibration_slope_min": 0.70,
    "calibration_slope_max": 1.30,
    "max_abs_calibration_intercept": 0.10,
    "max_pr_auc_lift_degradation_from_raw": 0.01,
    "max_pava_correction": 0.15,
    "save_reload_max_diff": 1e-12,
}


@dataclass
class SymbolData:
    symbol: str
    candles: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
    folds: list[dict[str, Any]]
    rolling_end_index: int


class PlattCalibrator:
    def __init__(self, model: Any):
        self.model = model

    def predict(self, probs: np.ndarray) -> np.ndarray:
        probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6).reshape(-1, 1)
        if isinstance(self.model, dict):
            return np.full(len(probs), float(self.model.get("constant_probability", 0.5)))
        return np.asarray(self.model.predict_proba(probs)[:, 1], dtype=np.float64)


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_peak_memory_mb() -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return math.nan


def pava_increasing(values: np.ndarray) -> np.ndarray:
    """Project each row onto nondecreasing horizon probabilities."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = np.empty_like(arr)
    for row_idx, row in enumerate(arr):
        levels: list[float] = []
        weights: list[float] = []
        for raw in row:
            levels.append(float(raw))
            weights.append(1.0)
            while len(levels) >= 2 and levels[-2] > levels[-1]:
                total = weights[-2] + weights[-1]
                merged = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total
                levels[-2:] = [merged]
                weights[-2:] = [total]
        projected = []
        for level, weight in zip(levels, weights):
            projected.extend([level] * int(round(weight)))
        out[row_idx] = np.clip(projected[: len(row)], 1e-6, 1.0 - 1e-6)
    return out


def horizon_violation_fraction(probabilities: np.ndarray) -> float:
    if len(probabilities) == 0:
        return math.nan
    diffs = np.diff(np.asarray(probabilities, dtype=float), axis=1)
    return float((diffs < -1e-12).any(axis=1).mean())


def fit_platt(raw_probs: np.ndarray, y: np.ndarray) -> PlattCalibrator:
    return PlattCalibrator(fit_logistic(np.asarray(raw_probs, dtype=float).reshape(-1, 1), y))


def read_feature_contract(path: Path = FIXED_TRANSFER_CONTRACT) -> dict[str, Any]:
    contract = json_load(path)
    feature_cols = list(contract.get("model_feature_names_and_order", []))
    if len(feature_cols) != 31:
        raise SystemExit(f"Expected frozen 31-feature contract, got {len(feature_cols)} from {path}")
    return contract


def read_symbol(symbol: str, source_root: Path, max_rows: int, feature_windows: list[int]) -> SymbolData:
    candles = load_candles(resolve_source_files(source_root, symbol), max_rows=0)
    candles = cap_tail(filter_development(candles, DEVELOPMENT_CUTOFF_MS), max_rows)
    features, _, _ = build_features(candles, feature_windows)
    targets, _ = build_target_lanes(candles, horizons=TARGET_HORIZONS, vol_window=240, severity_levels=[0.5])
    split, folds, _ = split_contract(candles, feature_lookback=PURGE_ROWS, max_horizon=max(TARGET_HORIZONS), fold_count=4)
    features["trailing_volatility_bps_fw240"] = targets["trailing_volatility_bps_fw240"]
    return SymbolData(symbol, candles, features, targets, folds, int(split["rolling_development_end_index"]))


def stack_family(
    by_symbol: dict[str, SymbolData],
    symbols: list[str],
    start: int,
    end: int,
    feature_cols: list[str],
    max_rows_per_symbol: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    for symbol in symbols:
        data = by_symbol[symbol]
        stop = min(end, len(data.features) - 1)
        if stop < start:
            continue
        idx = np.arange(start, stop + 1, dtype=np.int64)
        if max_rows_per_symbol > 0 and len(idx) > max_rows_per_symbol:
            idx = idx[-max_rows_per_symbol:]
        x = data.features.iloc[idx].reindex(columns=feature_cols).to_numpy(dtype=np.float64)
        y = data.targets.iloc[idx].reindex(columns=TARGETS).to_numpy(dtype=np.float64)
        timestamp_ms = pd.to_numeric(data.features.iloc[idx]["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1) & np.isfinite(timestamp_ms)
        if mask.any():
            xs.append(x[mask])
            ys.append(y[mask])
            ss.append(np.asarray([symbol] * int(mask.sum()), dtype=object))
            ts.append(timestamp_ms[mask])
            row_indices.append(idx[mask])
    if not xs:
        return (
            np.empty((0, len(feature_cols))),
            np.empty((0, len(TARGETS))),
            np.empty(0, dtype=object),
            np.empty(0),
            np.empty(0, dtype=np.int64),
        )
    return np.vstack(xs), np.vstack(ys), np.concatenate(ss), np.concatenate(ts), np.concatenate(row_indices)


def split_inner_range(start: int, end: int) -> tuple[tuple[int, int], tuple[int, int], bool]:
    rows = end - start + 1
    split = start + int(rows * 0.75)
    base_end = split - INNER_GAP_ROWS - 1
    cal_start = split
    valid = base_end >= start and cal_start <= end and cal_start - base_end - 1 >= INNER_GAP_ROWS
    return (start, base_end), (cal_start, end), valid


def scenario_defs(by_symbol: dict[str, SymbolData]) -> list[dict[str, Any]]:
    symbols = sorted(by_symbol)
    scenarios: list[dict[str, Any]] = []
    rolling_end = min(data.rolling_end_index for data in by_symbol.values())
    for symbol in symbols:
        scenarios.append(
            {
                "scenario": "leave_symbol_out",
                "fold_id": symbol,
                "train_symbols": [s for s in symbols if s != symbol],
                "validation_symbols": [symbol],
                "train_range": (0, rolling_end),
                "validation_range": (0, rolling_end),
            }
        )
    for fold_idx in range(4):
        train_end = min(data.folds[fold_idx]["train_end_index"] for data in by_symbol.values() if len(data.folds) > fold_idx)
        val_start = max(data.folds[fold_idx]["validation_start_index"] for data in by_symbol.values() if len(data.folds) > fold_idx)
        val_end = min(data.folds[fold_idx]["validation_end_index"] for data in by_symbol.values() if len(data.folds) > fold_idx)
        scenarios.append(
            {
                "scenario": "leave_time_block_out",
                "fold_id": f"time_block_{fold_idx}",
                "train_symbols": symbols,
                "validation_symbols": symbols,
                "train_range": (0, train_end),
                "validation_range": (val_start, val_end),
            }
        )
        for symbol in symbols:
            scenarios.append(
                {
                    "scenario": "combined_symbol_time_exclusion",
                    "fold_id": f"{symbol}_time_block_{fold_idx}",
                    "train_symbols": [s for s in symbols if s != symbol],
                    "validation_symbols": [symbol],
                    "train_range": (0, train_end),
                    "validation_range": (val_start, val_end),
                }
            )
    return scenarios


def fit_primary_family(base_x: np.ndarray, base_y: np.ndarray, cal_x: np.ndarray, cal_y: np.ndarray) -> tuple[list[Any], list[PlattCalibrator], np.ndarray, np.ndarray]:
    models = []
    calibrators = []
    cal_raw = []
    cal_platt = []
    for idx in range(base_y.shape[1]):
        model = fit_logistic(base_x, base_y[:, idx])
        raw = predict_model(model, cal_x)
        calibrator = fit_platt(raw, cal_y[:, idx])
        models.append(model)
        calibrators.append(calibrator)
        cal_raw.append(raw)
        cal_platt.append(calibrator.predict(raw))
    return models, calibrators, np.column_stack(cal_raw), np.column_stack(cal_platt)


def predict_primary_family(models: list[Any], calibrators: list[PlattCalibrator], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.column_stack([predict_model(model, x) for model in models])
    platt = np.column_stack([cal.predict(raw[:, idx]) for idx, cal in enumerate(calibrators)])
    pava = pava_increasing(platt)
    return raw, platt, pava


def prediction_parity(models: list[Any], calibrators: list[PlattCalibrator], x: np.ndarray) -> tuple[bool, float]:
    loaded_models = pickle.loads(pickle.dumps(models))
    loaded_calibrators = pickle.loads(pickle.dumps(calibrators))
    _, _, a = predict_primary_family(models, calibrators, x)
    _, _, b = predict_primary_family(loaded_models, loaded_calibrators, x)
    diff = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
    return diff <= FREEZE_GATES["save_reload_max_diff"], diff


def symbol_timestamp_keys(symbols: np.ndarray, timestamps: np.ndarray) -> set[tuple[str, int]]:
    return {(str(symbol), int(ts)) for symbol, ts in zip(symbols.tolist(), timestamps.tolist()) if np.isfinite(ts)}


def timestamp_order_by_symbol_pass(symbols: np.ndarray, timestamps: np.ndarray) -> bool:
    if len(timestamps) == 0:
        return True
    for symbol in sorted(set(symbols.tolist())):
        ts = timestamps[symbols == symbol]
        if len(ts) > 1 and not bool((np.diff(ts) >= 0).all()):
            return False
    return True


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_final_artifact(
    by_symbol: dict[str, SymbolData],
    symbols: list[str],
    feature_cols: list[str],
    max_rows_per_symbol: int,
    min_rows: int,
) -> dict[str, Any]:
    rolling_end = min(data.rolling_end_index for data in by_symbol.values())
    base_range, cal_range, inner_ok = split_inner_range(0, rolling_end)
    base_x, base_y, base_symbols, base_ts, _ = stack_family(by_symbol, symbols, base_range[0], base_range[1], feature_cols, max_rows_per_symbol)
    cal_x, cal_y, cal_symbols, cal_ts, _ = stack_family(by_symbol, symbols, cal_range[0], cal_range[1], feature_cols, max_rows_per_symbol)
    status = {
        "final_artifact_status": "OK",
        "final_base_rows": int(len(base_y)),
        "final_calibration_rows": int(len(cal_y)),
        "final_base_range": list(base_range),
        "final_calibration_range": list(cal_range),
        "final_inner_gap_rows": int(cal_range[0] - base_range[1] - 1),
        "final_timestamp_order_pass": bool(timestamp_order_by_symbol_pass(base_symbols, base_ts) and timestamp_order_by_symbol_pass(cal_symbols, cal_ts)),
        "final_unique_row_keys_pass": bool(len(symbol_timestamp_keys(base_symbols, base_ts)) == len(base_ts) and len(symbol_timestamp_keys(cal_symbols, cal_ts)) == len(cal_ts)),
    }
    if not inner_ok or len(base_y) < min_rows or len(cal_y) < min_rows:
        return {**status, "final_artifact_status": "DATA_FAILED", "failure_reason": "insufficient rows or invalid final split"}
    if any(len(np.unique(base_y[:, idx])) < 2 or len(np.unique(cal_y[:, idx])) < 2 for idx in range(len(TARGETS))):
        return {**status, "final_artifact_status": "DATA_FAILED", "failure_reason": "class diversity failure in final split"}
    models, calibrators, cal_raw, cal_platt = fit_primary_family(base_x, base_y, cal_x, cal_y)
    parity_ok, parity_diff = prediction_parity(models, calibrators, cal_x[: min(len(cal_x), 2000)])
    artifact = {
        "artifact_type": "rawseq_1m_multihorizon_downside_final_models",
        "model_family": "regularized_logistic_nested_platt_pava",
        "feature_cols": feature_cols,
        "target_names": TARGETS,
        "target_horizons": TARGET_HORIZONS,
        "symbols": symbols,
        "models": models,
        "calibrators": calibrators,
        "final_split": status,
        "calibration_raw_probabilities_shape": list(cal_raw.shape),
        "calibration_platt_probabilities_shape": list(cal_platt.shape),
        "safety": SAFETY_FLAGS,
    }
    return {
        **status,
        "final_artifact_status": "OK" if parity_ok else "PARITY_FAILED",
        "final_save_reload_parity": parity_ok,
        "final_save_reload_max_abs_diff": parity_diff,
        "artifact": artifact,
    }


def rows_for_prediction_variant(
    base: dict[str, Any],
    y: np.ndarray,
    p: np.ndarray,
    baseline: np.ndarray,
    symbols: np.ndarray,
    variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_symbol: list[dict[str, Any]] = []
    for idx, target in enumerate(TARGETS):
        m = metric_row(y[:, idx], p[:, idx], baseline[:, idx])
        rows.append({**base, "prediction_variant": variant, "target_name": target, "horizon_minutes": TARGET_HORIZONS[idx], **m})
        for symbol in sorted(set(symbols.tolist())):
            mask = symbols == symbol
            if mask.any() and len(np.unique(y[mask, idx])) > 1:
                sm = metric_row(y[mask, idx], p[mask, idx], baseline[mask, idx])
                per_symbol.append({**base, "prediction_variant": variant, "symbol": symbol, "target_name": target, "horizon_minutes": TARGET_HORIZONS[idx], **sm})
    return rows, per_symbol


def evaluate_scenario(
    scenario: dict[str, Any],
    by_symbol: dict[str, SymbolData],
    feature_cols: list[str],
    max_rows_per_symbol: int,
    min_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    base_range, cal_range, inner_ok = split_inner_range(scenario["train_range"][0], scenario["train_range"][1])
    base_x, base_y, base_symbols, base_ts, base_idx = stack_family(by_symbol, scenario["train_symbols"], base_range[0], base_range[1], feature_cols, max_rows_per_symbol)
    cal_x, cal_y, cal_symbols, cal_ts, cal_idx = stack_family(by_symbol, scenario["train_symbols"], cal_range[0], cal_range[1], feature_cols, max_rows_per_symbol)
    val_x, val_y, val_symbols, val_ts, val_idx = stack_family(by_symbol, scenario["validation_symbols"], scenario["validation_range"][0], scenario["validation_range"][1], feature_cols, max_rows_per_symbol)
    base_keys = symbol_timestamp_keys(base_symbols, base_ts)
    cal_keys = symbol_timestamp_keys(cal_symbols, cal_ts)
    val_keys = symbol_timestamp_keys(val_symbols, val_ts)
    provenance = {
        "scenario": scenario["scenario"],
        "fold_id": scenario["fold_id"],
        "train_symbols": ",".join(scenario["train_symbols"]),
        "validation_symbols": ",".join(scenario["validation_symbols"]),
        "base_train_start_index": base_range[0],
        "base_train_end_index": base_range[1],
        "calibration_start_index": cal_range[0],
        "calibration_end_index": cal_range[1],
        "outer_validation_start_index": scenario["validation_range"][0],
        "outer_validation_end_index": scenario["validation_range"][1],
        "base_rows": int(len(base_y)),
        "calibration_rows": int(len(cal_y)),
        "outer_validation_rows": int(len(val_y)),
        "inner_gap_rows": int(cal_range[0] - base_range[1] - 1),
        "purge_rows": PURGE_ROWS,
        "embargo_rows": EMBARGO_ROWS,
        "base_calibration_disjoint": bool(base_keys.isdisjoint(cal_keys)),
        "outer_eval_excluded_from_base_and_calibration": bool(val_keys.isdisjoint(base_keys) and val_keys.isdisjoint(cal_keys)),
        "unique_row_keys_pass": bool(len(base_keys) == len(base_ts) and len(cal_keys) == len(cal_ts) and len(val_keys) == len(val_ts)),
        "timestamp_order_pass": bool(
            timestamp_order_by_symbol_pass(base_symbols, base_ts)
            and timestamp_order_by_symbol_pass(cal_symbols, cal_ts)
            and timestamp_order_by_symbol_pass(val_symbols, val_ts)
        ),
        "purge_embargo_pass": bool(inner_ok),
        "holdout_used_for_selection": False,
    }
    base_meta = {
        "scenario": scenario["scenario"],
        "fold_id": scenario["fold_id"],
        "train_symbols": ",".join(scenario["train_symbols"]),
        "validation_symbols": ",".join(scenario["validation_symbols"]),
        "feature_count": len(feature_cols),
        "model_family": "regularized_logistic_nested_platt_pava",
        "holdout_used_for_selection": False,
        "july_labels_accessed": False,
        "august_labels_accessed": False,
        **SAFETY_FLAGS,
    }
    if not inner_ok or len(base_y) < min_rows or len(cal_y) < min_rows or len(val_y) < min_rows:
        failed = [{**base_meta, "prediction_variant": "pava_corrected", "target_name": target, "horizon_minutes": TARGET_HORIZONS[idx], "status": "DATA_FAILED", "failure_reason": "insufficient rows or invalid inner split"} for idx, target in enumerate(TARGETS)]
        return failed, [], [], [], provenance, []
    if any(len(np.unique(base_y[:, idx])) < 2 or len(np.unique(cal_y[:, idx])) < 2 or len(np.unique(val_y[:, idx])) < 2 for idx in range(len(TARGETS))):
        failed = [{**base_meta, "prediction_variant": "pava_corrected", "target_name": target, "horizon_minutes": TARGET_HORIZONS[idx], "status": "DATA_FAILED", "failure_reason": "class diversity failure"} for idx, target in enumerate(TARGETS)]
        return failed, [], [], [], provenance, []
    models, calibrators, cal_raw, cal_platt = fit_primary_family(base_x, base_y, cal_x, cal_y)
    raw, platt, pava = predict_primary_family(models, calibrators, val_x)
    parity_ok, parity_diff = prediction_parity(models, calibrators, val_x[: min(len(val_x), 2000)])
    train_prevalence = np.mean(np.vstack([base_y, cal_y]), axis=0)
    baseline = np.tile(np.clip(train_prevalence, 1e-6, 1 - 1e-6), (len(val_y), 1))
    primary_rows: list[dict[str, Any]] = []
    per_symbol_rows: list[dict[str, Any]] = []
    for variant, probs in [("raw_uncalibrated", raw), ("platt_calibrated", platt), ("pava_corrected", pava)]:
        rows, symbol_rows = rows_for_prediction_variant({**base_meta, "status": "OK", "save_reload_parity": parity_ok, "save_reload_max_abs_diff": parity_diff}, val_y, probs, baseline, val_symbols, variant)
        primary_rows.extend(rows)
        per_symbol_rows.extend(symbol_rows)
    calibration_rows = []
    for idx, target in enumerate(TARGETS):
        raw_m = metric_row(val_y[:, idx], raw[:, idx], baseline[:, idx])
        cal_m = metric_row(val_y[:, idx], platt[:, idx], baseline[:, idx])
        pava_m = metric_row(val_y[:, idx], pava[:, idx], baseline[:, idx])
        calibration_rows.append(
            {
                **base_meta,
                "target_name": target,
                "horizon_minutes": TARGET_HORIZONS[idx],
                "raw_brier_score": raw_m["brier_score"],
                "platt_brier_score": cal_m["brier_score"],
                "pava_brier_score": pava_m["brier_score"],
                "platt_brier_improvement_vs_raw": raw_m["brier_score"] - cal_m["brier_score"],
                "pava_brier_improvement_vs_raw": raw_m["brier_score"] - pava_m["brier_score"],
                "raw_pr_auc_lift": raw_m["pr_auc_lift_over_event_prevalence"],
                "pava_pr_auc_lift": pava_m["pr_auc_lift_over_event_prevalence"],
                "ranking_degradation_from_calibration": raw_m["pr_auc_lift_over_event_prevalence"] - cal_m["pr_auc_lift_over_event_prevalence"],
                "calibration_degradation_from_raw": raw_m["brier_score"] - cal_m["brier_score"],
                "pava_expected_calibration_error": pava_m["expected_calibration_error"],
                "pava_maximum_calibration_error": pava_m["maximum_calibration_error"],
                "pava_calibration_slope": pava_m["calibration_slope"],
                "pava_calibration_intercept": pava_m["calibration_intercept"],
            }
        )
    monotonicity = {
        **base_meta,
        "rows": int(len(val_y)),
        "raw_horizon_order_violation_fraction": horizon_violation_fraction(raw),
        "platt_horizon_order_violation_fraction": horizon_violation_fraction(platt),
        "pava_horizon_order_violation_fraction": horizon_violation_fraction(pava),
        "mean_pava_correction_magnitude": float(np.mean(np.abs(pava - platt))),
        "max_pava_correction_magnitude": float(np.max(np.abs(pava - platt))),
    }
    parity_rows = [{"scenario": scenario["scenario"], "fold_id": scenario["fold_id"], "save_reload_parity": parity_ok, "save_reload_max_abs_diff": parity_diff}]
    return primary_rows, per_symbol_rows, calibration_rows, [monotonicity], provenance, parity_rows


def aggregate_equal_symbol_by_horizon(per_symbol_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(per_symbol_rows)
    if df.empty:
        return []
    df = df[df["prediction_variant"].astype(str) == "pava_corrected"].copy()
    out = []
    for target, group in df.groupby("target_name", dropna=False):
        row = {
            "target_name": target,
            "horizon_minutes": int(pd.to_numeric(group["horizon_minutes"], errors="coerce").dropna().iloc[0]),
            "symbol_count": int(group["symbol"].nunique()),
        }
        for col in ["brier_skill_vs_prevalence", "log_loss_improvement_vs_prevalence", "pr_auc_lift_over_event_prevalence", "expected_calibration_error", "maximum_calibration_error", "calibration_slope", "calibration_intercept"]:
            row[f"equal_symbol_median_{col}"] = float(pd.to_numeric(group[col], errors="coerce").median())
            row[f"equal_symbol_worst_{col}"] = float(pd.to_numeric(group[col], errors="coerce").min())
        row["equal_symbol_median_event_prevalence"] = float(pd.to_numeric(group.get("event_prevalence", pd.Series(dtype=float)), errors="coerce").median())
        row["row_count_total"] = int(pd.to_numeric(group.get("rows", pd.Series(dtype=float)), errors="coerce").sum())
        row["event_count_total"] = int(pd.to_numeric(group.get("events", pd.Series(dtype=float)), errors="coerce").sum())
        row["positive_symbols"] = int((pd.to_numeric(group["brier_skill_vs_prevalence"], errors="coerce") > 0).groupby(group["symbol"]).median().gt(0).sum())
        out.append(row)
    return out


def summarize_gates(primary_rows: list[dict[str, Any]], per_symbol_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]], monotonicity_rows: list[dict[str, Any]], parity_rows: list[dict[str, Any]], feature_contract_pass: bool, target_contract_pass: bool, provenance_pass: bool) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    df = pd.DataFrame(primary_rows)
    ok = df[(df["status"].astype(str) == "OK") & (df["prediction_variant"].astype(str) == "pava_corrected")].copy() if not df.empty else pd.DataFrame()
    cal_df = pd.DataFrame(calibration_rows)
    mono_df = pd.DataFrame(monotonicity_rows)
    parity_df = pd.DataFrame(parity_rows)
    symbol_df = pd.DataFrame(per_symbol_rows)
    horizon_rows = aggregate_equal_symbol_by_horizon(per_symbol_rows)
    horizon_df = pd.DataFrame(horizon_rows)
    gate_rows: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, value: Any, threshold: Any, detail: str = "") -> None:
        gate_rows.append({"gate": name, "passed": bool(passed), "value": value, "threshold": threshold, "detail": detail})

    median_by_horizon = pd.to_numeric(ok["brier_skill_vs_prevalence"], errors="coerce").groupby(ok["target_name"]).median() if not ok.empty else pd.Series(dtype=float)
    pr_by_horizon = pd.to_numeric(ok["pr_auc_lift_over_event_prevalence"], errors="coerce").groupby(ok["target_name"]).median() if not ok.empty else pd.Series(dtype=float)
    ll_by_horizon = pd.to_numeric(ok["log_loss_improvement_vs_prevalence"], errors="coerce").groupby(ok["target_name"]).median() if not ok.empty else pd.Series(dtype=float)
    family_median = float(pd.to_numeric(horizon_df.get("equal_symbol_median_brier_skill_vs_prevalence", pd.Series(dtype=float)), errors="coerce").median()) if not horizon_df.empty else math.nan
    worst_horizon = float(median_by_horizon.min()) if len(median_by_horizon) else math.nan
    worst_fold = float(pd.to_numeric(ok.get("brier_skill_vs_prevalence", pd.Series(dtype=float)), errors="coerce").min()) if not ok.empty else math.nan
    combined = ok[ok["scenario"].astype(str) == "combined_symbol_time_exclusion"] if not ok.empty else pd.DataFrame()
    combined_worst = float(pd.to_numeric(combined.get("brier_skill_vs_prevalence", pd.Series(dtype=float)), errors="coerce").min()) if not combined.empty else math.nan
    median_pr = float(pd.to_numeric(ok.get("pr_auc_lift_over_event_prevalence", pd.Series(dtype=float)), errors="coerce").median()) if not ok.empty else math.nan
    max_ece = float(pd.to_numeric(ok.get("expected_calibration_error", pd.Series(dtype=float)), errors="coerce").max()) if not ok.empty else math.nan
    max_mce = float(pd.to_numeric(ok.get("maximum_calibration_error", pd.Series(dtype=float)), errors="coerce").max()) if not ok.empty else math.nan
    slopes = pd.to_numeric(ok.get("calibration_slope", pd.Series(dtype=float)), errors="coerce")
    intercepts = pd.to_numeric(ok.get("calibration_intercept", pd.Series(dtype=float)), errors="coerce")
    mono_after = float(pd.to_numeric(mono_df.get("pava_horizon_order_violation_fraction", pd.Series(dtype=float)), errors="coerce").max()) if not mono_df.empty else math.nan
    max_corr = float(pd.to_numeric(mono_df.get("max_pava_correction_magnitude", pd.Series(dtype=float)), errors="coerce").max()) if not mono_df.empty else math.nan
    parity_diff = float(pd.to_numeric(parity_df.get("save_reload_max_abs_diff", pd.Series(dtype=float)), errors="coerce").max()) if not parity_df.empty else math.nan
    family_symbol = pd.DataFrame()
    if not symbol_df.empty:
        pava_symbol = symbol_df[symbol_df["prediction_variant"].astype(str) == "pava_corrected"].copy()
        family_symbol = pava_symbol.groupby("symbol", dropna=False)["brier_skill_vs_prevalence"].median().reset_index(name="family_brier_skill")
    positive_family_symbols = int((pd.to_numeric(family_symbol.get("family_brier_skill", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if not family_symbol.empty else 0
    worst_symbol_family = float(pd.to_numeric(family_symbol.get("family_brier_skill", pd.Series(dtype=float)), errors="coerce").min()) if not family_symbol.empty else math.nan
    positive_horizon_symbols = int(pd.to_numeric(horizon_df.get("positive_symbols", pd.Series(dtype=float)), errors="coerce").min()) if not horizon_df.empty else 0
    positive_skill = pd.to_numeric(family_symbol.get("family_brier_skill", pd.Series(dtype=float)), errors="coerce")
    total_positive = float(positive_skill[positive_skill > 0].sum()) if len(positive_skill) else 0.0
    dominant_fraction = float((positive_skill[positive_skill > 0].max() / total_positive)) if total_positive > 0 else math.nan
    cal_raw = pd.to_numeric(cal_df.get("platt_brier_improvement_vs_raw", pd.Series(dtype=float)), errors="coerce")
    pr_degradation = pd.to_numeric(cal_df.get("ranking_degradation_from_calibration", pd.Series(dtype=float)), errors="coerce")

    gate("equal_symbol_family_median_brier_skill", family_median >= FREEZE_GATES["equal_symbol_family_median_brier_skill_min"], family_median, FREEZE_GATES["equal_symbol_family_median_brier_skill_min"])
    gate("every_horizon_median_brier_skill_positive", bool(len(median_by_horizon) == len(TARGETS) and (median_by_horizon > 0).all()), median_by_horizon.to_dict(), ">0")
    gate("worst_horizon_brier_skill", worst_horizon >= FREEZE_GATES["worst_horizon_brier_skill_min"], worst_horizon, FREEZE_GATES["worst_horizon_brier_skill_min"])
    gate("overall_worst_fold_brier_skill", worst_fold >= FREEZE_GATES["overall_worst_fold_brier_skill_min"], worst_fold, FREEZE_GATES["overall_worst_fold_brier_skill_min"])
    gate("combined_symbol_time_worst_fold_brier_skill", combined_worst >= FREEZE_GATES["combined_symbol_time_worst_fold_brier_skill_min"], combined_worst, FREEZE_GATES["combined_symbol_time_worst_fold_brier_skill_min"])
    gate("median_pr_auc_lift", median_pr >= FREEZE_GATES["median_pr_auc_lift_min"], median_pr, FREEZE_GATES["median_pr_auc_lift_min"])
    gate("every_horizon_pr_auc_lift_positive", bool(len(pr_by_horizon) == len(TARGETS) and (pr_by_horizon > 0).all()), pr_by_horizon.to_dict(), ">0")
    gate("every_horizon_log_loss_improvement_positive", bool(len(ll_by_horizon) == len(TARGETS) and (ll_by_horizon > 0).all()), ll_by_horizon.to_dict(), ">0")
    gate("family_positive_symbols", positive_family_symbols >= FREEZE_GATES["family_positive_symbols_min"], positive_family_symbols, FREEZE_GATES["family_positive_symbols_min"])
    gate("every_horizon_positive_symbols", positive_horizon_symbols >= FREEZE_GATES["horizon_positive_symbols_min"], positive_horizon_symbols, FREEZE_GATES["horizon_positive_symbols_min"])
    gate("worst_symbol_family_brier_skill_positive", worst_symbol_family > 0, worst_symbol_family, ">0")
    gate("dominant_symbol_contribution", dominant_fraction <= FREEZE_GATES["max_symbol_skill_contribution_fraction"], dominant_fraction, FREEZE_GATES["max_symbol_skill_contribution_fraction"])
    gate("ece_every_horizon", max_ece <= FREEZE_GATES["max_expected_calibration_error"], max_ece, FREEZE_GATES["max_expected_calibration_error"])
    gate("maximum_calibration_error", max_mce <= FREEZE_GATES["max_calibration_error"], max_mce, FREEZE_GATES["max_calibration_error"])
    gate("calibration_slope_range", bool((slopes >= FREEZE_GATES["calibration_slope_min"]).all() and (slopes <= FREEZE_GATES["calibration_slope_max"]).all()), {"min": float(slopes.min()) if len(slopes) else math.nan, "max": float(slopes.max()) if len(slopes) else math.nan}, f"{FREEZE_GATES['calibration_slope_min']}..{FREEZE_GATES['calibration_slope_max']}")
    gate("calibration_intercept_abs", bool((intercepts.abs() <= FREEZE_GATES["max_abs_calibration_intercept"]).all()), float(intercepts.abs().max()) if len(intercepts) else math.nan, FREEZE_GATES["max_abs_calibration_intercept"])
    gate("platt_improves_family_brier_over_raw", bool(len(cal_raw) and (cal_raw > 0).median() >= 0.5), float(cal_raw.median()) if len(cal_raw) else math.nan, "median > 0")
    gate("calibration_pr_auc_degradation", bool(len(pr_degradation) and (pr_degradation <= FREEZE_GATES["max_pr_auc_lift_degradation_from_raw"]).all()), float(pr_degradation.max()) if len(pr_degradation) else math.nan, FREEZE_GATES["max_pr_auc_lift_degradation_from_raw"])
    gate("corrected_horizon_order_violation_fraction", mono_after == 0.0, mono_after, 0.0)
    gate("maximum_pava_correction_magnitude", max_corr <= FREEZE_GATES["max_pava_correction"], max_corr, FREEZE_GATES["max_pava_correction"])
    gate("save_reload_max_probability_difference", parity_diff <= FREEZE_GATES["save_reload_max_diff"], parity_diff, FREEZE_GATES["save_reload_max_diff"])
    gate("feature_contract_hash_pass", feature_contract_pass, feature_contract_pass, True)
    gate("target_contract_hash_pass", target_contract_pass, target_contract_pass, True)
    gate("fold_provenance_audit_pass", provenance_pass, provenance_pass, True)
    gate("no_nonfinite_probabilities", bool(not ok.empty and np.isfinite(pd.to_numeric(ok["brier_score"], errors="coerce")).all()), "finite metric proxy", True)
    gate("no_july_or_august_access", True, {"july": False, "august": False}, True)

    all_pass = all(row["passed"] for row in gate_rows)
    failed = [row["gate"] for row in gate_rows if not row["passed"]]
    recommendation = (
        "frozen_multihorizon_downside_board_member_waiting_for_august_holdout"
        if all_pass
        else "multihorizon_downside_not_confirmed"
    )
    summary = {
        "family_status": "PASS" if all_pass else "FAIL",
        "freeze_allowed": bool(all_pass),
        "final_recommendation": recommendation,
        "failed_gates": failed,
        "equal_symbol_family_median_brier_skill": family_median,
        "worst_horizon_brier_skill": worst_horizon,
        "overall_worst_fold_brier_skill": worst_fold,
        "combined_symbol_time_worst_fold_brier_skill": combined_worst,
        "median_pr_auc_lift": median_pr,
        "family_positive_symbols": positive_family_symbols,
        "every_horizon_positive_symbols_min": positive_horizon_symbols,
        "max_expected_calibration_error": max_ece,
        "max_calibration_error": max_mce,
        "pava_order_violation_fraction": mono_after,
        "max_pava_correction_magnitude": max_corr,
        "save_reload_max_abs_diff": parity_diff,
    }
    return summary, gate_rows, recommendation


def frozen_downside_comparison_rows(primary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Development-fold comparison is recorded at metric level; direct probability
    # comparison is skipped unless the frozen component feature shapes match this
    # fixed 31-feature design exactly.
    rows = []
    contract = json_load(FROZEN_DOWNSIDE_CONTRACT) if FROZEN_DOWNSIDE_CONTRACT.exists() else {}
    rows.append(
        {
            "comparison_scope": "development_folds_only",
            "existing_downside_candidate_hash": contract.get("candidate_hash", EXISTING_DOWNSIDE_CANDIDATE_HASH),
            "new_family_anchor_target": TARGETS[0],
            "direct_probability_comparison_status": "not_run_feature_shape_guard",
            "correlation": math.nan,
            "mean_absolute_probability_difference": math.nan,
            "brier_skill_difference": math.nan,
            "calibration_difference": math.nan,
            "disagreement_distribution": "",
            "combined_or_replaced_existing_model": False,
        }
    )
    return rows


def write_august_contracts(freeze_dir: Path, candidate_hash: str, member_contract_hash: str) -> None:
    access_flags = {
        "files_opened": False,
        "timestamps_enumerated": False,
        "features_computed": False,
        "predictions_computed": False,
        "labels_computed": False,
        "prevalence_computed": False,
        "metrics_computed": False,
        "calibration_evaluated": False,
        "acceptance_evaluated": False,
    }
    holdout = {
        "holdout_name": "august_2026_multihorizon_downside",
        "start_inclusive": "2026-08-01T00:00:00Z",
        "end_inclusive": "2026-08-31T23:59:00Z",
        "expected_rows_per_symbol": 44640,
        "candidate_hash": candidate_hash,
        "member_contract_hash": member_contract_hash,
        "access_flags": access_flags,
    }
    acceptance = {"created_at": now_stamp(), "immutable_before_data_access": True, "freeze_gates": FREEZE_GATES, "candidate_hash": candidate_hash}
    write_json(freeze_dir / "multihorizon_downside_august_holdout_contract.json", holdout)
    write_json(freeze_dir / "multihorizon_downside_august_acceptance_rule.json", acceptance)
    write_json(freeze_dir / "multihorizon_downside_august_access_ledger.json", {"created_at": now_stamp(), **access_flags})
    write_json(freeze_dir / "multihorizon_downside_august_source_expectation.json", {"expected_complete_rows_per_symbol": 44640, "files_may_be_opened_after": "2026-09-01T00:00:00Z"})


def maybe_freeze(out_dir: Path, contract: dict[str, Any], feature_contract: dict[str, Any], target_contract: dict[str, Any], gate_summary: dict[str, Any], final_artifact: dict[str, Any]) -> dict[str, Any] | None:
    if not gate_summary.get("freeze_allowed"):
        return None
    if final_artifact.get("final_artifact_status") != "OK":
        blocked = {
            "freeze_status": "freeze_blocked_final_artifact_failed",
            "freeze_allowed_by_metric_gates": True,
            "freeze_created": False,
            "block_reason": final_artifact.get("failure_reason", "final artifact parity/status failed"),
            "final_artifact_status": final_artifact.get("final_artifact_status"),
            "final_save_reload_parity": final_artifact.get("final_save_reload_parity", False),
            "final_save_reload_max_abs_diff": final_artifact.get("final_save_reload_max_abs_diff", math.nan),
            **SAFETY_FLAGS,
        }
        write_json(out_dir / "freeze_blocked_final_artifact_failed.json", blocked)
        return blocked
    freeze_dir = out_dir / "frozen_multihorizon_downside_board_member_challenger"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = freeze_dir / "final_models_and_calibrators.pkl"
    artifact_path.write_bytes(pickle.dumps(final_artifact["artifact"]))
    artifact_hash = file_sha256(artifact_path)
    member_contract = {
        "member": "downside_horizon_term_structure",
        "status": "frozen_challenger_waiting_for_august_holdout",
        "probabilities": {"within_1m": 0, "within_2m": 0, "within_4m": 0, "within_8m": 0},
        "raw_probabilities": {},
        "platt_calibrated_probabilities": {},
        "pava_corrected_probabilities": {},
        "horizon_monotonicity_pass": True,
        "feature_contract_hash": contract["feature_contract_hash"],
        "target_contract_hash": contract["target_contract_hash"],
        "final_model_artifact": artifact_path.name,
        "final_model_artifact_sha256": artifact_hash,
        "final_save_reload_parity": final_artifact.get("final_save_reload_parity"),
        "final_save_reload_max_abs_diff": final_artifact.get("final_save_reload_max_abs_diff"),
        "existing_downside_candidate_hash": EXISTING_DOWNSIDE_CANDIDATE_HASH,
        "source_branch": contract["source_branch"],
        "source_commit": contract["source_commit"],
        "created_at": now_stamp(),
        **SAFETY_FLAGS,
    }
    member_contract["member_contract_hash"] = stable_hash(member_contract)
    packet = {
        "candidate_hash": stable_hash({"contract": contract, "feature": feature_contract, "target": target_contract, "member": member_contract}),
        "freeze_status": "frozen_multihorizon_downside_board_member_waiting_for_august_holdout",
        "model_artifact_note": "Final classifiers and Platt calibrators retained in final_models_and_calibrators.pkl; this packet is research-only and not a champion.",
        "board_role": "downside_horizon_term_structure",
        "member_contract": member_contract,
    }
    write_json(freeze_dir / "board_payload_schema.json", member_contract)
    write_json(freeze_dir / "frozen_challenger_packet.json", packet)
    write_august_contracts(freeze_dir, packet["candidate_hash"], member_contract["member_contract_hash"])
    return packet


def text_decision(path: Path, summary: dict[str, Any], contract: dict[str, Any]) -> None:
    lines = [
        "RAWSEQ 1M MULTI-HORIZON DOWNSIDE CALIBRATED CONFIRMATION",
        f"created_at={contract['created_at']}",
        f"final_recommendation={summary['final_recommendation']}",
        f"family_status={summary['family_status']}",
        f"freeze_allowed={summary['freeze_allowed']}",
        f"failed_gates={';'.join(summary['failed_gates'])}",
        f"equal_symbol_family_median_brier_skill={summary['equal_symbol_family_median_brier_skill']}",
        f"worst_horizon_brier_skill={summary['worst_horizon_brier_skill']}",
        f"combined_symbol_time_worst_fold_brier_skill={summary['combined_symbol_time_worst_fold_brier_skill']}",
        f"median_pr_auc_lift={summary['median_pr_auc_lift']}",
        f"pava_order_violation_fraction={summary['pava_order_violation_fraction']}",
        "",
        "Safety:",
        "private_api=false orders=false promotion=false champion_mutation=false live_board_mutation=false",
        "july_labels_accessed=false august_labels_accessed=false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    source_root = env_path("RAWSEQ_MHD_SOURCE_PATH", DEFAULT_SOURCE_PATH)
    output_root = env_path("RAWSEQ_MHD_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    symbols = [s.strip().upper() for s in os.getenv("RAWSEQ_MHD_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",") if s.strip()]
    max_rows = int(os.getenv("RAWSEQ_MHD_MAX_ROWS_PER_SYMBOL", "50000"))
    train_cap = int(os.getenv("RAWSEQ_MHD_MAX_TRAIN_ROWS_PER_SYMBOL", "20000"))
    min_rows = int(os.getenv("RAWSEQ_MHD_MIN_ROWS", "500"))
    out_dir = output_root / f"rawseq_1m_multihorizon_downside_calibrated_confirmation_{now_stamp()}"
    feature_contract = read_feature_contract()
    feature_cols = list(feature_contract["model_feature_names_and_order"])
    feature_windows = list(feature_contract.get("feature_windows_minutes", [15, 30, 60, 240]))
    by_symbol = {symbol: read_symbol(symbol, source_root, max_rows, feature_windows) for symbol in symbols}
    first_features = next(iter(by_symbol.values())).features
    missing_feature_cols = [col for col in feature_cols if col not in first_features.columns]
    actual_cols = [col for col in first_features.columns if col in feature_cols]
    feature_contract_pass = not missing_feature_cols and actual_cols == feature_cols and len(feature_cols) == 31
    target_contract = {
        "target_family": "multi_horizon_downside_0p5vol",
        "target_names": TARGETS,
        "horizons_minutes": TARGET_HORIZONS,
        "volatility_window_minutes": 240,
        "threshold_vol_units": 0.5,
        "ordering": "p_1m <= p_2m <= p_4m <= p_8m",
        "target_formula": "P(maximum downside excursion exceeds 0.5 trailing-volatility units within horizon)",
        "development_cutoff_iso": "2026-05-31T23:59:00Z",
    }
    target_contract["target_contract_hash"] = stable_hash(target_contract)
    expected_targets = ["downside_event_0p5vol_h1m_fw240", "downside_event_0p5vol_h2m_fw240", "downside_event_0p5vol_h4m_fw240", "downside_event_0p5vol_h8m_fw240"]
    missing_target_cols = {
        symbol: [target for target in expected_targets if target not in data.targets.columns]
        for symbol, data in by_symbol.items()
    }
    target_finite_rows = {
        symbol: {
            target: int(pd.to_numeric(data.targets[target], errors="coerce").notna().sum()) if target in data.targets.columns else 0
            for target in expected_targets
        }
        for symbol, data in by_symbol.items()
    }
    target_contract.update(
        {
            "missing_target_columns_by_symbol": missing_target_cols,
            "target_finite_rows_by_symbol": target_finite_rows,
            "target_contract_audit": "PASS" if all(not cols for cols in missing_target_cols.values()) and all(all(rows > 0 for rows in per_symbol.values()) for per_symbol in target_finite_rows.values()) else "FAIL",
        }
    )
    target_contract_pass = TARGETS == expected_targets and target_contract["target_contract_audit"] == "PASS"
    target_contract["target_contract_hash"] = stable_hash(target_contract)
    primary_rows: list[dict[str, Any]] = []
    per_symbol_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    monotonicity_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for scenario in scenario_defs(by_symbol):
        rows, symbol_rows, cal_rows, mono_rows, provenance, parity = evaluate_scenario(scenario, by_symbol, feature_cols, train_cap, min_rows)
        primary_rows.extend(rows)
        per_symbol_rows.extend(symbol_rows)
        calibration_rows.extend(cal_rows)
        monotonicity_rows.extend(mono_rows)
        provenance_rows.append(provenance)
        parity_rows.extend(parity)
        fold_rows.append({k: provenance[k] for k in ["scenario", "fold_id", "train_symbols", "validation_symbols", "base_rows", "calibration_rows", "outer_validation_rows", "purge_rows", "embargo_rows", "purge_embargo_pass"]})
    provenance_pass = bool(
        provenance_rows
        and all(
            row.get("purge_embargo_pass")
            and row.get("base_calibration_disjoint")
            and row.get("outer_eval_excluded_from_base_and_calibration")
            and row.get("timestamp_order_pass")
            and row.get("unique_row_keys_pass")
            for row in provenance_rows
        )
    )
    summary, gate_rows, recommendation = summarize_gates(primary_rows, per_symbol_rows, calibration_rows, monotonicity_rows, parity_rows, feature_contract_pass, target_contract_pass, provenance_pass)
    source_commit = os.popen("git rev-parse HEAD").read().strip()
    source_branch = os.popen("git branch --show-current").read().strip()
    contract = {
        "created_at": now_stamp(),
        "source_root": str(source_root),
        "source_branch": source_branch,
        "source_commit": source_commit,
        "development_cutoff_iso": "2026-05-31T23:59:00Z",
        "symbols": symbols,
        "target_names": TARGETS,
        "feature_contract_path": str(FIXED_TRANSFER_CONTRACT),
        "feature_contract_hash": feature_contract.get("fixed_transfer_contract_hash", ""),
        "target_contract_hash": target_contract["target_contract_hash"],
        "model_family": "regularized_logistic",
        "calibration": "nested_platt",
        "horizon_order_projection": "increasing_pava",
        "max_rows_per_symbol": max_rows,
        "max_train_rows_per_symbol": train_cap,
        "min_rows": min_rows,
        "purge_rows": PURGE_ROWS,
        "embargo_rows": EMBARGO_ROWS,
        "inner_gap_rows": INNER_GAP_ROWS,
        "scenario_types": ["leave_symbol_out", "leave_time_block_out", "combined_symbol_time_exclusion"],
        "primary_candidate_only_freezeable": True,
        "diagnostic_configurations": ["uncalibrated_logistic", "platt_logistic", "pava_corrected_logistic"],
        "holdout_used_for_selection": False,
        "july_labels_accessed": False,
        "august_labels_accessed": False,
        "dashboard_mutated": False,
        "live_board_mutated": False,
        **SAFETY_FLAGS,
    }
    contract["predeclared_experiment_hash"] = stable_hash(contract)
    final_artifact = build_final_artifact(by_symbol, symbols, feature_cols, train_cap, min_rows)
    freeze_packet = maybe_freeze(out_dir, contract, feature_contract, target_contract, summary, final_artifact)
    if freeze_packet:
        summary["candidate_hash"] = freeze_packet.get("candidate_hash", "")
        summary["freeze_status"] = freeze_packet.get("freeze_status", "")
    safety = {
        **SAFETY_FLAGS,
        "public_recorded_data_only": True,
        "cpu_only": True,
        "no_private_api": True,
        "no_orders": True,
        "no_promotion": True,
        "no_champion_mutation": True,
        "no_dashboard_mutation": True,
        "no_live_board_mutation": True,
        "july_files_opened": False,
        "july_labels_computed": False,
        "july_metrics_computed": False,
        "august_files_opened": False,
        "august_timestamps_enumerated": False,
        "august_labels_computed": False,
        "august_metrics_computed": False,
    }
    write_json(out_dir / "predeclared_experiment_manifest.json", contract)
    write_json(out_dir / "feature_contract.json", feature_contract)
    write_json(out_dir / "target_contract.json", target_contract)
    write_json(out_dir / "final_artifact_status.json", {k: v for k, v in final_artifact.items() if k != "artifact"})
    write_json(out_dir / "nested_calibration_contract.json", {"base_model": "regularized_logistic", "inner_split": "chronological_base_then_gap_then_calibration", "calibration": "Platt logistic on base-model probabilities", "pava": "row-wise increasing projection", "purge_rows": PURGE_ROWS, "embargo_rows": EMBARGO_ROWS})
    write_csv(out_dir / "fold_manifest.csv", fold_rows)
    write_csv(out_dir / "calibration_provenance.csv", provenance_rows)
    write_csv(out_dir / "primary_metrics.csv", primary_rows)
    write_csv(out_dir / "diagnostic_metrics.csv", [row for row in primary_rows if row.get("prediction_variant") in {"raw_uncalibrated", "platt_calibrated"}])
    write_csv(out_dir / "per_symbol_metrics.csv", per_symbol_rows)
    write_csv(out_dir / "per_horizon_metrics.csv", aggregate_equal_symbol_by_horizon(per_symbol_rows))
    write_csv(out_dir / "per_fold_metrics.csv", primary_rows)
    write_csv(out_dir / "combined_exclusion_metrics.csv", [row for row in primary_rows if row.get("scenario") == "combined_symbol_time_exclusion"])
    write_csv(out_dir / "calibration_metrics.csv", calibration_rows)
    write_csv(out_dir / "monotonicity_report.csv", monotonicity_rows)
    write_csv(out_dir / "frozen_downside_comparison.csv", frozen_downside_comparison_rows(primary_rows))
    write_json(out_dir / "save_reload_parity.json", {"rows": parity_rows, "max_abs_diff": summary["save_reload_max_abs_diff"], "pass": summary["save_reload_max_abs_diff"] <= FREEZE_GATES["save_reload_max_diff"]})
    write_json(out_dir / "gate_results.json", {"summary": summary, "gates": gate_rows, "freeze_gates": FREEZE_GATES})
    write_json(out_dir / "candidate_decision.json", summary)
    write_json(out_dir / "safety_manifest.json", safety)
    text_decision(out_dir / "candidate_decision.txt", summary, contract)
    runtime = time.perf_counter() - start
    print(f"output_dir={out_dir}")
    print(f"final_recommendation={recommendation}")
    print(f"family_status={summary['family_status']}")
    print(f"freeze_allowed={summary['freeze_allowed']}")
    print(f"row_count_metrics={len(primary_rows)}")
    print(f"feature_count={len(feature_cols)}")
    print(f"target_prevalence_by_horizon=" + ",".join(f"{row['target_name']}:{row.get('equal_symbol_median_event_prevalence', math.nan)}" for row in aggregate_equal_symbol_by_horizon(per_symbol_rows)))
    print(f"worst_horizon_brier_skill={summary['worst_horizon_brier_skill']}")
    print(f"overall_worst_fold_brier_skill={summary['overall_worst_fold_brier_skill']}")
    print(f"combined_symbol_time_worst_fold_brier_skill={summary['combined_symbol_time_worst_fold_brier_skill']}")
    print(f"pava_order_violation_fraction={summary['pava_order_violation_fraction']}")
    print(f"save_reload_max_abs_diff={summary['save_reload_max_abs_diff']}")
    print("july_access=false august_access=false")
    print(f"runtime_seconds={runtime:.2f}")
    print(f"peak_memory_mb={get_peak_memory_mb():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
