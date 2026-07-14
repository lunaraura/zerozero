#!/usr/bin/env python3
"""CPU-only staged feature-family evolution for rawseq 1m board research."""

from __future__ import annotations

import math
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny import run_rawseq_1m_board_member_target_feature_tournament as board  # noqa: E402
from scripts.tiny.rawseq_1m_baseline_utils import (  # noqa: E402
    DEFAULT_SOURCE_PATH,
    SAFETY_FLAGS,
    env_path,
    now_stamp,
    parse_int_list,
    stable_hash,
    write_csv,
    write_json,
)
from scripts.tiny.rawseq_1m_feature_evolution_runtime import (  # noqa: E402
    BoundedContentCache,
    CheckpointStore,
    HeartbeatEmitter,
    MatrixTelemetry,
    MemoryGuard,
    StagePreparationCache,
    file_identity,
    format_progress_line,
    process_memory_snapshot,
    progress_payload,
    stable_hash as runtime_stable_hash,
    truthy_env,
    write_json_atomic,
)

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_feature_family_evolution")
DEFAULT_SYMBOLS = board.DEFAULT_SYMBOLS
DEFAULT_TARGET_LANES = [
    "multi_horizon_downside",
    "downside_severity",
    "upside_excursion",
    "volatility_expansion",
    "barrier_first",
    "directional_return",
    "downside_interval_hazard",
    "upside_interval_hazard",
]
DEFAULT_FEATURE_GROUPS = [
    "existing",
    "existing_plus_quote_spread",
    "existing_plus_short_path",
    "existing_plus_cross_asset",
    "existing_plus_regime",
    "all_challenger_features",
    "all_minus_quote_spread",
    "all_minus_short_path",
    "all_minus_cross_asset",
    "all_minus_regime",
]
DEFAULT_MODELS = ["constant_prevalence", "regularized_logistic", "shallow_hgb"]
DEFAULT_REGIME_NAMES = ["all"]
PREPROCESSING_CONTRACT = "rawseq1m_board_features_dev_cutoff_2026_05_31_v1"
SPLIT_CONTRACT = "chronological_development_folds_no_holdout_v1"
CALIBRATION_CONTRACT = "none"


def env_str_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def env_float_list(name: str, default: list[float]) -> list[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def clamp_keep_fraction(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return min(1.0, max(0.05, value))


def stage_preset(name: str) -> list[dict[str, Any]]:
    preset = name.strip().lower()
    if preset == "dry_run":
        return [{"stage_id": 0, "stage_name": "dry_run", "source_rows_per_symbol": 0, "eval_rows_per_symbol": 0, "folds": 0, "dry_run": True}]
    if preset == "smoke":
        return [{"stage_id": 1, "stage_name": "smoke", "source_rows_per_symbol": 2500, "eval_rows_per_symbol": 750, "folds": 1, "dry_run": False}]
    if preset == "overnight":
        return [
            {"stage_id": 1, "stage_name": "coarse_screen", "source_rows_per_symbol": 25000, "eval_rows_per_symbol": 5000, "folds": 2, "dry_run": False},
            {"stage_id": 2, "stage_name": "expanded_screen", "source_rows_per_symbol": 50000, "eval_rows_per_symbol": 15000, "folds": 4, "dry_run": False},
        ]
    if preset == "full_dev":
        return [
            {"stage_id": 1, "stage_name": "coarse_screen", "source_rows_per_symbol": 50000, "eval_rows_per_symbol": 15000, "folds": 4, "dry_run": False},
            {"stage_id": 2, "stage_name": "full_development", "source_rows_per_symbol": 0, "eval_rows_per_symbol": 0, "folds": 4, "dry_run": False},
        ]
    raise ValueError(f"unknown RAWSEQ_EVOLVE_STAGE_PRESET={name!r}")


def candidate_key(row: dict[str, Any]) -> str:
    payload = {
        "target_lane": row.get("target_lane"),
        "target_name": row.get("target_name"),
        "horizon_minutes": row.get("horizon_minutes"),
        "interval_start_minutes": row.get("interval_start_minutes"),
        "interval_end_minutes": row.get("interval_end_minutes"),
        "threshold_vol_units": row.get("threshold_vol_units"),
        "feature_group": row.get("feature_group"),
        "regime_name": row.get("regime_name", "all"),
        "model_seed": row.get("model_seed", 1337),
        "model": row.get("model"),
        "preprocessing_contract": row.get("preprocessing_contract", PREPROCESSING_CONTRACT),
        "split_contract": row.get("split_contract", SPLIT_CONTRACT),
        "calibration_contract": row.get("calibration_contract", CALIBRATION_CONTRACT),
    }
    return stable_hash(payload)[:16]


def result_key(row: dict[str, Any]) -> str:
    return candidate_key(row)


def build_candidate_grid(
    target_rows: list[dict[str, Any]],
    feature_groups: list[str],
    models: list[str],
    allowed_lanes: set[str],
    regime_names: list[str] | None = None,
    model_seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    deduped_targets: dict[str, dict[str, Any]] = {}
    regime_names = regime_names or ["all"]
    model_seeds = model_seeds or [1337]
    for row in target_rows:
        if "target_name" in row and str(row.get("target_lane")) in allowed_lanes:
            deduped_targets.setdefault(f"{row.get('target_lane')}::{row['target_name']}", row)
    out: list[dict[str, Any]] = []
    for target_key in sorted(deduped_targets):
        target = deduped_targets[target_key]
        for feature_group in feature_groups:
            for regime_name in regime_names:
                for model in models:
                    seeds_for_model = model_seeds if model != "constant_prevalence" else [model_seeds[0]]
                    for model_seed in seeds_for_model:
                        row = {
                            "candidate_key": "",
                            "parent_candidate_key": stable_hash({"target_lane": target.get("target_lane"), "target_name": target.get("target_name"), "model": model})[:16],
                            "mutation_family": "feature_family" if regime_name in {"", "all", "unrestricted"} else "feature_family_plus_regime_specialist",
                            "mutation_detail": f"{feature_group}|regime={regime_name}|seed={model_seed}",
                            "lineage_generation": 0,
                            "target_lane": target.get("target_lane"),
                            "target_name": target.get("target_name"),
                            "horizon_minutes": target.get("horizon_minutes"),
                            "interval_start_minutes": target.get("interval_start_minutes"),
                            "interval_end_minutes": target.get("interval_end_minutes"),
                            "threshold_vol_units": target.get("threshold_vol_units"),
                            "board_role": target.get("board_role"),
                            "feature_group": feature_group,
                            "regime_name": regime_name,
                            "model_seed": int(model_seed),
                            "model": model,
                            "preprocessing_contract": PREPROCESSING_CONTRACT,
                            "split_contract": SPLIT_CONTRACT,
                            "calibration_contract": CALIBRATION_CONTRACT,
                            "candidate_status": "pending",
                            "holdout_used_for_selection": False,
                            **SAFETY_FLAGS,
                        }
                        row["candidate_key"] = candidate_key(row)
                        out.append(row)
    assert_unique_candidate_keys(out)
    out.sort(key=candidate_order_tuple)
    return out


def candidate_order_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("target_lane", "")),
        str(row.get("target_name", "")),
        str(row.get("horizon_minutes", "")),
        str(row.get("threshold_vol_units", "")),
        str(row.get("feature_group", "")),
        str(row.get("regime_name", "all")),
        str(row.get("model", "")),
        int(row.get("model_seed", 1337) or 1337),
        str(row.get("candidate_key", "")),
    )


def assert_unique_candidate_keys(candidates: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in candidates:
        key = str(row.get("candidate_key") or candidate_key(row))
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError("duplicate candidate_key values: " + ",".join(sorted(set(duplicates))))


def candidate_records_for_active_keys(candidate_grid: list[dict[str, Any]], active_keys: set[str] | None) -> list[dict[str, Any]]:
    rows = list(candidate_grid) if active_keys is None else [row for row in candidate_grid if str(row.get("candidate_key")) in active_keys]
    assert_unique_candidate_keys(rows)
    return sorted(rows, key=candidate_order_tuple)


def safe_median(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    return float(vals.median()) if vals.notna().any() else math.nan


def safe_min(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    return float(vals.min()) if vals.notna().any() else math.nan


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fitness_score(row: dict[str, Any]) -> float:
    median_brier = finite_float(row.get("median_brier_skill"))
    worst_brier = finite_float(row.get("worst_fold_brier_skill"))
    pr_lift = finite_float(row.get("median_pr_auc_lift"))
    log_loss = finite_float(row.get("median_log_loss_improvement"))
    ece = finite_float(row.get("median_expected_calibration_error"))
    feature_count = finite_float(row.get("median_feature_count"))
    folds = max(1.0, finite_float(row.get("folds"), 1.0))
    positive_folds = finite_float(row.get("positive_brier_folds"))
    stability = positive_folds / folds
    negative_worst_penalty = abs(min(0.0, worst_brier)) * 2.0
    complexity_penalty = min(0.02, feature_count / 10000.0)
    calibration_penalty = min(0.05, max(0.0, ece - 0.05))
    return (
        median_brier
        + 0.75 * worst_brier
        + 0.25 * pr_lift
        + 0.10 * log_loss
        + 0.05 * stability
        - negative_worst_penalty
        - complexity_penalty
        - calibration_penalty
    )


def aggregate_stage_results(rows: list[dict[str, Any]], min_prevalence: float) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    ok = df[(df["status"] == "OK") & (df["model"] != "constant_prevalence")].copy()
    out: list[dict[str, Any]] = []
    if "regime_name" not in ok.columns:
        ok["regime_name"] = "all"
    if "model_seed" not in ok.columns:
        ok["model_seed"] = 1337
    group_cols = ["target_lane", "target_name", "feature_group", "regime_name", "model", "model_seed"]
    for keys, group in ok.groupby(group_cols, dropna=False):
        target_lane, target_name, feature_group, regime_name, model, model_seed = keys
        brier = pd.to_numeric(group["brier_skill_vs_prevalence"], errors="coerce")
        pr = pd.to_numeric(group["pr_auc_lift_over_event_prevalence"], errors="coerce")
        prev = pd.to_numeric(group["event_prevalence"], errors="coerce")
        ece = pd.to_numeric(group["expected_calibration_error"], errors="coerce")
        folds = int(len(group))
        expected_folds = int(pd.to_numeric(group.get("expected_folds", pd.Series([folds])), errors="coerce").max()) if folds else 0
        positive_brier_folds = int((brier > 0).sum())
        finite_required_metrics = all(
            math.isfinite(x)
            for x in [
                safe_median(brier),
                safe_min(brier),
                safe_median(pr),
                safe_median(prev),
            ]
        )
        row = {
            "candidate_key": "",
            "target_lane": target_lane,
            "target_name": target_name,
            "feature_group": feature_group,
            "regime_name": regime_name,
            "model": model,
            "model_seed": int(model_seed),
            "folds": folds,
            "expected_folds": expected_folds,
            "positive_brier_folds": positive_brier_folds,
            "rows": int(pd.to_numeric(group["rows"], errors="coerce").sum()),
            "median_feature_count": safe_median(group["feature_count"]),
            "median_event_prevalence": safe_median(prev),
            "median_brier_skill": safe_median(brier),
            "worst_fold_brier_skill": safe_min(brier),
            "median_pr_auc_lift": safe_median(pr),
            "median_log_loss_improvement": safe_median(group["log_loss_improvement_vs_prevalence"]),
            "median_expected_calibration_error": safe_median(ece),
            "save_reload_parity_all": bool(group["save_reload_parity"].astype(str).str.lower().isin(["true", "1"]).all()),
            "sufficient_event_prevalence": bool(min_prevalence <= safe_median(prev) <= 1.0 - min_prevalence),
            "holdout_used_for_selection": False,
            **SAFETY_FLAGS,
        }
        row["candidate_key"] = candidate_key(row)
        row["fitness_score"] = fitness_score(row)
        row["survives_stage_gate"] = bool(
            row["save_reload_parity_all"]
            and row["sufficient_event_prevalence"]
            and finite_required_metrics
            and folds >= expected_folds
            and row["median_brier_skill"] > 0
            and row["median_pr_auc_lift"] > 0
            and positive_brier_folds >= max(1, math.ceil(folds / 2))
        )
        reasons = []
        if not finite_required_metrics:
            reasons.append("nonfinite_required_metrics")
        if folds < expected_folds:
            reasons.append("missing_or_failed_folds")
        for name in ["save_reload_parity_all", "sufficient_event_prevalence"]:
            if not row[name]:
                reasons.append(name)
        if row["median_brier_skill"] <= 0:
            reasons.append("nonpositive_median_brier_skill")
        if row["median_pr_auc_lift"] <= 0:
            reasons.append("nonpositive_median_pr_auc_lift")
        if positive_brier_folds < max(1, math.ceil(folds / 2)):
            reasons.append("insufficient_positive_folds")
        row["stage_rejection_reasons"] = ";".join(reasons)
        out.append(row)
    out.sort(
        key=lambda r: (
            -int(bool(r.get("survives_stage_gate"))),
            -finite_float(r.get("fitness_score"), -1e9),
            -finite_float(r.get("worst_fold_brier_skill"), -1e9),
            -finite_float(r.get("median_brier_skill"), -1e9),
            str(r.get("target_lane", "")),
            str(r.get("target_name", "")),
            str(r.get("feature_group", "")),
            str(r.get("regime_name", "all")),
            str(r.get("model", "")),
            int(r.get("model_seed", 1337) or 1337),
            str(r.get("candidate_key", "")),
        )
    )
    return out


def candidate_feature_columns(by_symbol: dict[str, board.SymbolData], feature_group: str) -> tuple[list[str], str]:
    all_cols = sorted(set().union(*(set(d.features.columns) for d in by_symbol.values())) - {"timestamp", "timestamp_ms"})
    cross_available = all(c in all_cols for c in ["btc_return_1m_bps", "market_median_return_1m_bps"])
    quote_available = any(c in all_cols for c in ["spread_bps", "best_bid", "bid_depth_10bps"])
    available: set[str] = set()
    if cross_available:
        available.add("cross_asset")
    if quote_available:
        available.add("quote_spread")
    groups = board.feature_groups(all_cols, available)
    cols = groups.get(feature_group, [])
    if not cols:
        return [], "feature_group_unavailable"
    missing = sorted(c for c in cols if any(c not in d.features.columns for d in by_symbol.values()))
    if missing:
        return [], "missing_feature_columns:" + ",".join(missing[:20])
    return cols, ""


def stack_symbol_indices(
    by_symbol: dict[str, board.SymbolData],
    symbols: list[str],
    start: int,
    end: int,
    feature_cols: list[str],
    target_col: str,
    max_rows_per_symbol: int,
    regime_name: str = "all",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    return board.stack_xy(by_symbol, symbols, start, end, feature_cols, target_col, max_rows_per_symbol, regime_name)


def scenario_metric_row(
    candidate: dict[str, Any],
    scenario_type: str,
    scenario_id: str,
    fold_id: int,
    train_symbols: list[str],
    validation_symbols: list[str],
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    min_rows: int,
    min_prevalence: float,
    train_regime_candidate_rows: int = 0,
    validation_regime_candidate_rows: int = 0,
    regime_filter_reason: str = "",
) -> dict[str, Any]:
    base = {
        "candidate_key": candidate_key(candidate),
        "target_lane": candidate.get("target_lane"),
        "target_name": candidate.get("target_name"),
        "feature_group": candidate.get("feature_group"),
        "model": candidate.get("model"),
        "scenario_type": scenario_type,
        "scenario_id": scenario_id,
        "fold_id": fold_id,
        "train_symbols": ",".join(train_symbols),
        "validation_symbols": ",".join(validation_symbols),
        "train_rows": int(len(train_y)),
        "validation_rows": int(len(val_y)),
        "regime_name": candidate.get("regime_name", "all"),
        "model_seed": int(candidate.get("model_seed", 1337)),
        "train_regime_candidate_rows": train_regime_candidate_rows,
        "validation_regime_candidate_rows": validation_regime_candidate_rows,
        "train_regime_coverage_fraction": float(len(train_y) / train_regime_candidate_rows) if train_regime_candidate_rows else math.nan,
        "validation_regime_coverage_fraction": float(len(val_y) / validation_regime_candidate_rows) if validation_regime_candidate_rows else math.nan,
        "regime_filter_reason": regime_filter_reason,
        "min_rows": min_rows,
        "min_prevalence": min_prevalence,
        "validation_scope": "scenario_validation_loso_time_combined",
        "scenario_validation_performed": True,
        "holdout_used_for_selection": False,
        **SAFETY_FLAGS,
    }
    if len(train_y) < min_rows or len(val_y) < min_rows:
        return {**base, "status": "DATA_FAILED", "failure_reason": "insufficient rows"}
    if len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
        return {**base, "status": "DATA_FAILED", "failure_reason": "insufficient class diversity"}
    prevalence = float(np.mean(train_y))
    if not (min_prevalence <= prevalence <= 1.0 - min_prevalence):
        return {**base, "status": "DATA_FAILED", "failure_reason": "train prevalence outside minimum gate", "train_event_prevalence": prevalence}
    try:
        pred, parity, diff, diagnostics = board.fit_predict_with_diagnostics(str(candidate.get("model")), train_x, train_y, val_x, int(candidate.get("model_seed", 1337)))
        if not diagnostics.get("model_eligible_for_advancement", True):
            return {**base, "status": "TRAIN_FAILED", "failure_reason": "model_ineligible_for_advancement", **diagnostics}
        baseline = np.full(len(val_y), np.clip(prevalence, 1e-6, 1 - 1e-6))
        return {
            **base,
            "status": "OK",
            "failure_reason": "",
            "train_event_prevalence": prevalence,
            "save_reload_parity": parity,
            "save_reload_max_abs_diff": diff,
            **diagnostics,
            **board.metric_row(val_y, pred, baseline),
            **board.confidence_coverage_metrics(val_y, pred, prevalence),
        }
    except Exception as exc:
        return {**base, "status": "TRAIN_FAILED", "failure_reason": repr(exc)}


def scenario_validation_rows(
    by_symbol: dict[str, board.SymbolData],
    candidates: list[dict[str, Any]],
    max_rows_per_symbol: int,
    min_rows: int,
    min_prevalence: float,
    max_folds: int,
    max_loso_symbols: int,
) -> list[dict[str, Any]]:
    symbols = sorted(by_symbol)
    rows: list[dict[str, Any]] = []
    passing = [row for row in candidates if row.get("survives_stage_gate") and str(row.get("model")) != "constant_prevalence"]
    for candidate in passing:
        target_col = str(candidate.get("target_name"))
        regime_name = str(candidate.get("regime_name", "all") or "all")
        feature_cols, failure = candidate_feature_columns(by_symbol, str(candidate.get("feature_group")))
        if failure:
            rows.append({**candidate, "scenario_type": "all", "scenario_id": "feature_contract", "status": "FEATURE_GROUP_UNAVAILABLE", "failure_reason": failure, **SAFETY_FLAGS})
            continue
        fold_count = min(max_folds, max((len(d.folds) for d in by_symbol.values()), default=0))
        for fold_id in range(fold_count):
            fold_symbols = [d for d in by_symbol.values() if len(d.folds) > fold_id]
            if not fold_symbols:
                continue
            train_end = min(d.folds[fold_id]["train_end_index"] for d in fold_symbols)
            val_start = max(d.folds[fold_id]["validation_start_index"] for d in fold_symbols)
            val_end = min(d.folds[fold_id]["validation_end_index"] for d in fold_symbols)
            train_x, train_y, _, train_regime_rows, train_regime_reason = stack_symbol_indices(by_symbol, symbols, 0, train_end, feature_cols, target_col, max_rows_per_symbol, regime_name)
            val_x, val_y, _, val_regime_rows, val_regime_reason = stack_symbol_indices(by_symbol, symbols, val_start, val_end, feature_cols, target_col, max_rows_per_symbol, regime_name)
            rows.append(
                scenario_metric_row(
                    candidate,
                    "leave_time_block_out",
                    f"time_fold_{fold_id}",
                    fold_id,
                    symbols,
                    symbols,
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    min_rows,
                    min_prevalence,
                    train_regime_rows,
                    val_regime_rows,
                    ";".join(x for x in [train_regime_reason, val_regime_reason] if x),
                )
            )
            for left_symbol in symbols[:max_loso_symbols]:
                train_symbols = [s for s in symbols if s != left_symbol]
                train_x, train_y, _, train_regime_rows, train_regime_reason = stack_symbol_indices(by_symbol, train_symbols, 0, train_end, feature_cols, target_col, max_rows_per_symbol, regime_name)
                val_x, val_y, _, val_regime_rows, val_regime_reason = stack_symbol_indices(by_symbol, [left_symbol], val_start, val_end, feature_cols, target_col, max_rows_per_symbol, regime_name)
                rows.append(
                    scenario_metric_row(
                        candidate,
                        "leave_one_symbol_out",
                        left_symbol,
                        fold_id,
                        train_symbols,
                        [left_symbol],
                        train_x,
                        train_y,
                        val_x,
                        val_y,
                        min_rows,
                        min_prevalence,
                        train_regime_rows,
                        val_regime_rows,
                        ";".join(x for x in [train_regime_reason, val_regime_reason] if x),
                    )
                )
                rows.append(
                    scenario_metric_row(
                        candidate,
                        "combined_symbol_time",
                        f"{left_symbol}_time_fold_{fold_id}",
                        fold_id,
                        train_symbols,
                        [left_symbol],
                        train_x,
                        train_y,
                        val_x,
                        val_y,
                        min_rows,
                        min_prevalence,
                        train_regime_rows,
                        val_regime_rows,
                        ";".join(x for x in [train_regime_reason, val_regime_reason] if x),
                    )
                )
    return rows


def aggregate_scenario_results(rows: list[dict[str, Any]], min_prevalence: float) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    ok = df[(df["status"] == "OK") & (df["model"] != "constant_prevalence")].copy()
    out: list[dict[str, Any]] = []
    if "regime_name" not in ok.columns:
        ok["regime_name"] = "all"
    if "model_seed" not in ok.columns:
        ok["model_seed"] = 1337
    group_cols = ["target_lane", "target_name", "feature_group", "regime_name", "model", "model_seed"]
    for keys, group in ok.groupby(group_cols, dropna=False):
        target_lane, target_name, feature_group, regime_name, model, model_seed = keys
        brier = pd.to_numeric(group["brier_skill_vs_prevalence"], errors="coerce")
        pr = pd.to_numeric(group["pr_auc_lift_over_event_prevalence"], errors="coerce")
        prev = pd.to_numeric(group["event_prevalence"], errors="coerce")
        by_scenario: dict[str, dict[str, Any]] = {}
        for scenario, sgroup in group.groupby("scenario_type", dropna=False):
            sbrier = pd.to_numeric(sgroup["brier_skill_vs_prevalence"], errors="coerce")
            by_scenario[str(scenario)] = {
                "rows": int(len(sgroup)),
                "positive_rows": int((sbrier > 0).sum()),
                "median_brier_skill": safe_median(sbrier),
                "worst_brier_skill": safe_min(sbrier),
            }
        row = {
            "candidate_key": "",
            "target_lane": target_lane,
            "target_name": target_name,
            "feature_group": feature_group,
            "regime_name": regime_name,
            "model": model,
            "model_seed": int(model_seed),
            "scenario_validation_performed": True,
            "validation_scope": "scenario_validation_loso_time_combined",
            "scenario_rows": int(len(group)),
            "scenario_ok_rows": int(len(group)),
            "median_brier_skill": safe_median(brier),
            "worst_scenario_brier_skill": safe_min(brier),
            "median_pr_auc_lift": safe_median(pr),
            "median_event_prevalence": safe_median(prev),
            "scenario_positive_brier_rows": int((brier > 0).sum()),
            "loso_worst_brier_skill": by_scenario.get("leave_one_symbol_out", {}).get("worst_brier_skill", math.nan),
            "time_block_worst_brier_skill": by_scenario.get("leave_time_block_out", {}).get("worst_brier_skill", math.nan),
            "combined_worst_brier_skill": by_scenario.get("combined_symbol_time", {}).get("worst_brier_skill", math.nan),
            "holdout_used_for_selection": False,
            **SAFETY_FLAGS,
        }
        row["candidate_key"] = candidate_key(row)
        row["scenario_survives_gate"] = bool(
            row["median_brier_skill"] > 0
            and row["worst_scenario_brier_skill"] > 0
            and row["median_pr_auc_lift"] > 0
            and min_prevalence <= row["median_event_prevalence"] <= 1.0 - min_prevalence
        )
        reasons = []
        if row["median_brier_skill"] <= 0:
            reasons.append("nonpositive_scenario_median_brier_skill")
        if row["worst_scenario_brier_skill"] <= 0:
            reasons.append("nonpositive_worst_scenario_brier_skill")
        if row["median_pr_auc_lift"] <= 0:
            reasons.append("nonpositive_scenario_pr_auc_lift")
        if not (min_prevalence <= row["median_event_prevalence"] <= 1.0 - min_prevalence):
            reasons.append("scenario_prevalence_outside_gate")
        row["scenario_rejection_reasons"] = ";".join(reasons)
        row["fitness_score"] = fitness_score(
            {
                "median_brier_skill": row["median_brier_skill"],
                "worst_fold_brier_skill": row["worst_scenario_brier_skill"],
                "median_pr_auc_lift": row["median_pr_auc_lift"],
                "median_log_loss_improvement": 0.0,
                "median_expected_calibration_error": 0.0,
                "median_feature_count": 0.0,
                "folds": row["scenario_rows"],
                "positive_brier_folds": row["scenario_positive_brier_rows"],
            }
        )
        out.append(row)
    out.sort(key=lambda r: (r["scenario_survives_gate"], r["fitness_score"], r["worst_scenario_brier_skill"]), reverse=True)
    return out


def select_survivor_keys(stage_survivors: list[dict[str, Any]], keep_fraction: float, min_keep: int, advance_failed_for_diagnostics: bool = False) -> tuple[set[str], list[dict[str, Any]]]:
    eligible = [row for row in stage_survivors if row.get("survives_stage_gate")]
    ranked = eligible if eligible or not advance_failed_for_diagnostics else list(stage_survivors)
    keep_n = min(len(ranked), max(min_keep, int(math.ceil(len(ranked) * keep_fraction)))) if ranked else 0
    keep = set(str(row["candidate_key"]) for row in ranked[:keep_n])
    audit: list[dict[str, Any]] = []
    for idx, row in enumerate(stage_survivors, start=1):
        kept = row["candidate_key"] in keep
        audit.append(
            {
                "candidate_key": row["candidate_key"],
                "rank": idx,
                "kept_for_next_stage": kept,
                "drop_reason": "" if kept else ("stage_gate_failed" if not row.get("survives_stage_gate") else "successive_halving_cut"),
                "fitness_score": row.get("fitness_score"),
                "target_lane": row.get("target_lane"),
                "target_name": row.get("target_name"),
                "feature_group": row.get("feature_group"),
                "model": row.get("model"),
                **SAFETY_FLAGS,
            }
        )
    return keep, audit


def stage_rows_for_active_targets(
    target_manifest_rows: list[dict[str, Any]],
    active_keys: set[str] | None,
    feature_groups: list[str],
    models: list[str],
    allowed_lanes: set[str],
    regime_names: list[str] | None = None,
    model_seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    if active_keys is None:
        return [row for row in target_manifest_rows if "target_name" in row and str(row.get("target_lane")) in allowed_lanes]
    regime_names = regime_names or ["all"]
    model_seeds = model_seeds or [1337]
    target_names = set()
    for row in target_manifest_rows:
        if "target_name" not in row:
            continue
        for feature_group in feature_groups:
            for regime_name in regime_names:
                for model in models:
                    seeds_for_model = model_seeds if model != "constant_prevalence" else [model_seeds[0]]
                    for model_seed in seeds_for_model:
                        probe = {**row, "feature_group": feature_group, "regime_name": regime_name, "model": model, "model_seed": int(model_seed)}
                        if candidate_key(probe) in active_keys:
                            target_names.add(str(row["target_name"]))
    return [row for row in target_manifest_rows if "target_name" in row and str(row.get("target_name")) in target_names]


def row_matches_active_candidate(row: dict[str, Any], active_keys: set[str], model_seeds: list[int] | None = None) -> bool:
    if result_key(row) in active_keys:
        return True
    if str(row.get("model")) != "all":
        return False
    model_seeds = model_seeds or [1337]
    for model in DEFAULT_MODELS:
        seeds_for_model = model_seeds if model != "constant_prevalence" else [model_seeds[0]]
        for model_seed in seeds_for_model:
            probe = {**row, "model": model, "model_seed": int(model_seed)}
            if candidate_key(probe) in active_keys:
                return True
    return False


def exact_candidate_stage_audit(
    stage_id: int,
    stage_name: str,
    prior_evaluated_keys: set[str],
    requested_candidates: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    survivor_keys: set[str] | None = None,
) -> dict[str, Any]:
    requested_keys = [str(row.get("candidate_key")) for row in requested_candidates]
    requested_set = set(requested_keys)
    evaluated_keys = sorted(set(str(row.get("candidate_key")) for row in result_rows if row.get("candidate_key")))
    duplicates = sorted(key for key in requested_set if requested_keys.count(key) > 1)
    return {
        "stage_id": stage_id,
        "stage_name": stage_name,
        "stage_one_evaluated_keys": sorted(prior_evaluated_keys),
        "survivor_keys": sorted(survivor_keys or set()),
        "incoming_survivor_keys_used_as_stage_request": sorted(set(requested_keys)) if prior_evaluated_keys else [],
        "requested_keys": sorted(requested_set),
        "actually_evaluated_keys": evaluated_keys,
        "missing_requested_keys": sorted(requested_set - set(evaluated_keys)),
        "unexpected_evaluated_keys": sorted(set(evaluated_keys) - requested_set),
        "duplicate_requested_keys": duplicates,
        "requested_count": len(requested_keys),
        "actually_evaluated_count": len(evaluated_keys),
        "every_stage_two_evaluated_candidate_was_stage_one_survivor": bool(not prior_evaluated_keys or set(evaluated_keys).issubset(requested_set)),
        "no_dropped_candidate_evaluated": bool(not prior_evaluated_keys or not (set(evaluated_keys) - requested_set)),
        "no_stage_two_candidate_appears_twice": not duplicates,
        "exact_candidate_filter_before_matrix_construction": True,
    }


def assert_exact_candidate_stage_audit_passes(audit: dict[str, Any]) -> None:
    failures = []
    if audit.get("missing_requested_keys"):
        failures.append("missing_requested_keys")
    if audit.get("unexpected_evaluated_keys"):
        failures.append("unexpected_evaluated_keys")
    if audit.get("duplicate_requested_keys"):
        failures.append("duplicate_requested_keys")
    if not audit.get("every_stage_two_evaluated_candidate_was_stage_one_survivor", True):
        failures.append("stage_two_evaluated_non_survivor")
    if not audit.get("no_dropped_candidate_evaluated", True):
        failures.append("dropped_candidate_evaluated")
    if not audit.get("no_stage_two_candidate_appears_twice", True):
        failures.append("duplicate_stage_two_candidate")
    if failures:
        raise AssertionError("exact candidate stage audit failed: " + ",".join(failures))


def build_symbol_data_for_stage(
    symbols: list[str],
    source_root: Path,
    source_rows_per_symbol: int,
    feature_windows: list[int],
    horizons: list[int],
    vol_window: int,
    severity_levels: list[float],
    stage_cache: StagePreparationCache | None = None,
    heartbeat: HeartbeatEmitter | None = None,
    stage_context: dict[str, Any] | None = None,
) -> tuple[dict[str, board.SymbolData], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_context = stage_context or {}
    source_files = {}
    for symbol in symbols:
        try:
            source_files[symbol] = [file_identity(path) for path in board.resolve_source_files(source_root, symbol)]
        except Exception as exc:
            source_files[symbol] = [{"symbol": symbol, "resolve_error": repr(exc)}]
    cache_contract = {
        "contract_type": "rawseq_1m_stage_preparation_v1",
        "symbols": list(symbols),
        "source_root": str(source_root),
        "source_files": source_files,
        "source_rows_per_symbol": int(source_rows_per_symbol),
        "cutoff_ms": int(board.DEVELOPMENT_CUTOFF_MS),
        "feature_windows": list(feature_windows),
        "horizons": list(horizons),
        "vol_window": int(vol_window),
        "severity_levels": [float(x) for x in severity_levels],
        "stage_context": stage_context,
        "feature_builder": "rawseq_1m_board_member_target_feature_tournament.read_symbol_add_cross_asset_features",
    }
    if heartbeat:
        heartbeat.emit("stage_cache_lookup", **stage_context, cache_status="lookup")
    if stage_cache:
        cached = stage_cache.load(cache_contract)
        if cached is not None:
            by_symbol, feature_audit_rows, target_manifest_rows, manifest = cached
            if heartbeat:
                heartbeat.emit("stage_cache_hit", **stage_context, cache_status="hit", cache_key=manifest.get("cache_key", ""))
            return by_symbol, feature_audit_rows, target_manifest_rows
        if heartbeat:
            heartbeat.emit("stage_cache_miss", **stage_context, cache_status="miss")
    by_symbol: dict[str, board.SymbolData] = {}
    feature_audit_rows: list[dict[str, Any]] = []
    target_manifest_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        if heartbeat:
            heartbeat.emit("source_loading", **stage_context, symbol=symbol)
        data, audit, target_rows = board.read_symbol(
            symbol,
            source_root,
            source_rows_per_symbol,
            board.DEVELOPMENT_CUTOFF_MS,
            feature_windows,
            horizons,
            vol_window,
            severity_levels,
        )
        if heartbeat:
            heartbeat.emit("feature_construction", **stage_context, symbol=symbol, feature_columns=len(data.features.columns))
            heartbeat.emit("target_construction", **stage_context, symbol=symbol, target_columns=len(data.targets.columns))
            heartbeat.emit("fold_construction", **stage_context, symbol=symbol, folds=len(data.folds))
        by_symbol[symbol] = data
        feature_audit_rows.extend({"symbol": symbol, **row} for row in audit)
        target_manifest_rows.extend(target_rows)
    if heartbeat:
        heartbeat.emit("cross_asset_feature_construction", **stage_context, symbol="ALL")
    board.add_cross_asset_features({symbol: data.features for symbol, data in by_symbol.items()})
    if stage_cache:
        manifest = stage_cache.write(cache_contract, by_symbol, feature_audit_rows, target_manifest_rows)
        if heartbeat:
            heartbeat.emit("stage_cache_write", **stage_context, cache_status="write", cache_key=manifest.get("cache_key", ""))
    return by_symbol, feature_audit_rows, target_manifest_rows


def rollup(rows: list[dict[str, Any]], by: str) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty or by not in df.columns:
        return []
    out = []
    for name, group in df.groupby(by, dropna=False):
        passed = group[group["survives_stage_gate"].astype(bool)] if "survives_stage_gate" in group else group.iloc[0:0]
        out.append(
            {
                by: name,
                "candidate_rows": int(len(group)),
                "surviving_rows": int(len(passed)),
                "best_fitness_score": safe_median(group.nlargest(1, "fitness_score")["fitness_score"]) if "fitness_score" in group else math.nan,
                "median_brier_skill": safe_median(group["median_brier_skill"]) if "median_brier_skill" in group else math.nan,
                "worst_fold_brier_skill": safe_min(group["worst_fold_brier_skill"]) if "worst_fold_brier_skill" in group else math.nan,
                **SAFETY_FLAGS,
            }
        )
    out.sort(key=lambda r: (r["surviving_rows"], r["best_fitness_score"]), reverse=True)
    return out


def write_summary(path: Path, contract: dict[str, Any], final_survivors: list[dict[str, Any]], dropped: list[dict[str, Any]]) -> None:
    lines = [
        "RAWSEQ 1M FEATURE-FAMILY EVOLUTION",
        f"created_at={contract['created_at']}",
        "scope=research_only_cpu_feature_family_evolution",
        "selection_data_end=2026-05-31T23:59:00Z",
        f"stage_preset={contract['stage_preset']}",
        f"candidate_grid_rows={contract['candidate_grid_rows']}",
        f"final_survivor_rows={sum(1 for row in final_survivors if row.get('survives_stage_gate'))}",
        f"scenario_validation_performed={contract.get('scenario_validation_performed')}",
        f"scenario_validation_scope={contract.get('validation_scope')}",
        "",
        "Top final candidates:",
    ]
    for row in final_survivors[:20]:
        lines.append(
            f"- status={'PASS' if row.get('survives_stage_gate') else 'FAIL'} fitness={float(row.get('fitness_score', 0.0)):.6f} "
            f"lane={row.get('target_lane')} target={row.get('target_name')} features={row.get('feature_group')} model={row.get('model')} "
            f"median_brier={float(row.get('median_brier_skill', math.nan)):.6f} worst_fold={float(row.get('worst_fold_brier_skill', math.nan)):.6f} "
            f"scenario_pass={row.get('scenario_survives_gate')} scenario_worst={float(row.get('scenario_worst_brier_skill', math.nan)):.6f} "
            f"reasons={row.get('stage_rejection_reasons', '')} scenario_reasons={row.get('scenario_rejection_reasons', '')}"
        )
    lines.extend(
        [
            "",
            "Successive halving:",
            f"- dropped_candidate_rows={len([row for row in dropped if row.get('drop_reason')])}",
            "",
            "Interpretation:",
            "- This is preliminary validation research, not a freeze or board promotion.",
            "- Frozen downside/dashboard/future-shadow artifacts were not changed.",
            "- Survivors require a separate fixed confirmation script before any frozen challenger packet.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rawseq 1m feature-family evolution.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint directory after validating the run contract.")
    parser.add_argument("--output-dir", default=os.getenv("RAWSEQ_EVOLVE_OUTPUT_DIR", ""), help="Use a fixed output directory instead of creating a timestamped one.")
    parser.add_argument("--checkpoint-dir", default=os.getenv("RAWSEQ_EVOLVE_CHECKPOINT_DIR", ""), help="Checkpoint directory. Defaults to <output_dir>/checkpoints.")
    parser.add_argument("--checkpoint-every-candidates", type=int, default=int(os.getenv("RAWSEQ_EVOLVE_CHECKPOINT_EVERY_CANDIDATES", "1")))
    parser.add_argument("--memory-guard-policy", default=os.getenv("RAWSEQ_EVOLVE_MEMORY_GUARD_POLICY", "fail_closed"), choices=["warn", "checkpoint_and_pause", "fail_closed"])
    return parser.parse_args(argv)


def code_version_info() -> dict[str, Any]:
    import subprocess

    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False).stdout.strip()
        except Exception:
            return ""

    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_branch": run(["git", "branch", "--show-current"]),
        "git_dirty_status_hash": runtime_stable_hash(run(["git", "status", "--short"])),
    }


def build_checkpoint_run_contract(
    config: dict[str, Any],
    candidate_grid: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "contract_type": "rawseq_1m_feature_family_evolution_checkpoint_v1",
        "configuration": config,
        "candidate_grid_hash": runtime_stable_hash(candidate_grid),
        "candidate_count": len(candidate_grid),
        "code_version": code_version_info(),
        "target_contract": {"target_lanes": config["allowed_target_lanes"], "horizons": config["horizons"], "severity_levels": config["severity_levels"], "volatility_window": config["volatility_window"]},
        "feature_contract": {"feature_windows": config["feature_windows"], "feature_groups": config["allowed_feature_groups"]},
        "split_contract": SPLIT_CONTRACT,
        "preprocessing_contract": PREPROCESSING_CONTRACT,
        "calibration_contract": CALIBRATION_CONTRACT,
        "model_contract": {"models": config["allowed_models"], "model_seeds": config["model_seeds"]},
        "deterministic_ordering_contract": "candidate_order_tuple_then_stage_order_then_fold_order",
    }


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    source_root = env_path("RAWSEQ_EVOLVE_SOURCE_PATH", DEFAULT_SOURCE_PATH)
    output_root = env_path("RAWSEQ_EVOLVE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    symbols = [s.strip().upper() for s in os.getenv("RAWSEQ_EVOLVE_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",") if s.strip()]
    feature_windows = parse_int_list(os.getenv("RAWSEQ_EVOLVE_FEATURE_WINDOWS", "60,240"), [60, 240])
    horizons = parse_int_list(os.getenv("RAWSEQ_EVOLVE_HORIZONS", "1,2,4,8"), [1, 2, 4, 8])
    severity_levels = env_float_list("RAWSEQ_EVOLVE_SEVERITY_LEVELS", [0.5, 1.0, 1.5, 2.0])
    vol_window = int(os.getenv("RAWSEQ_EVOLVE_VOL_WINDOW", "240"))
    min_rows = int(os.getenv("RAWSEQ_EVOLVE_MIN_ROWS", "500"))
    min_prevalence = float(os.getenv("RAWSEQ_EVOLVE_MIN_PREVALENCE", "0.01"))
    allowed_lanes = set(env_str_list("RAWSEQ_EVOLVE_TARGET_LANES", DEFAULT_TARGET_LANES))
    allowed_feature_groups = env_str_list("RAWSEQ_EVOLVE_FEATURE_GROUPS", DEFAULT_FEATURE_GROUPS)
    allowed_models = env_str_list("RAWSEQ_EVOLVE_MODELS", DEFAULT_MODELS)
    regime_names = env_str_list("RAWSEQ_EVOLVE_REGIMES", DEFAULT_REGIME_NAMES)
    model_seeds = parse_int_list(os.getenv("RAWSEQ_EVOLVE_SEEDS", "1337"), [1337])
    preset_name = os.getenv("RAWSEQ_EVOLVE_STAGE_PRESET", "overnight").strip().lower()
    stages = stage_preset(preset_name)
    keep_fraction = clamp_keep_fraction(float(os.getenv("RAWSEQ_EVOLVE_KEEP_FRACTION", "0.5")))
    min_keep = int(os.getenv("RAWSEQ_EVOLVE_MIN_KEEP", "10"))
    max_candidates = int(os.getenv("RAWSEQ_EVOLVE_MAX_CANDIDATES", "0"))
    matrix_dtype = np.dtype(os.getenv("RAWSEQ_EVOLVE_MATRIX_DTYPE", "float64"))
    matrix_telemetry_enabled = truthy_env("RAWSEQ_EVOLVE_MATRIX_TELEMETRY", False)
    cache_enabled = truthy_env("RAWSEQ_EVOLVE_MATRIX_CACHE", True)
    cache_max_entries = int(os.getenv("RAWSEQ_EVOLVE_CACHE_MAX_ENTRIES", "128"))
    cache_max_bytes = int(os.getenv("RAWSEQ_EVOLVE_CACHE_MAX_BYTES", str(512 * 1024 * 1024)))
    stage_cache_enabled = truthy_env("RAWSEQ_EVOLVE_STAGE_PREP_CACHE", True)
    stage_cache_root = env_path("RAWSEQ_EVOLVE_STAGE_PREP_CACHE_DIR", output_root / "_stage_prep_cache")
    stage_cache_max_entries = int(os.getenv("RAWSEQ_EVOLVE_STAGE_PREP_CACHE_MAX_ENTRIES", "24"))
    heartbeat_enabled = truthy_env("RAWSEQ_EVOLVE_HEARTBEAT", True)
    advance_failed_for_diagnostics = os.getenv("RAWSEQ_EVOLVE_ADVANCE_FAILED_FOR_DIAGNOSTICS", "false").strip().lower() in {"1", "true", "yes", "on"}
    scenario_validation_enabled = os.getenv("RAWSEQ_EVOLVE_RUN_SCENARIO_VALIDATION", "true").strip().lower() in {"1", "true", "yes", "on"}
    max_loso_symbols = int(os.getenv("RAWSEQ_EVOLVE_SCENARIO_MAX_LOSO_SYMBOLS", str(len(symbols))))
    dry_only = all(stage.get("dry_run") for stage in stages)
    progress_every = max(1, int(os.getenv("RAWSEQ_EVOLVE_PROGRESS_EVERY", "10")))

    out_dir = Path(args.output_dir) if args.output_dir else output_root / f"rawseq_1m_feature_family_evolution_{now_stamp()}"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "checkpoints"
    checkpoint_store = CheckpointStore(checkpoint_dir)
    heartbeat = HeartbeatEmitter(enabled=heartbeat_enabled)
    stage_cache = StagePreparationCache(stage_cache_root, stage_cache_max_entries) if stage_cache_enabled else None
    max_source_rows = 0 if any(int(stage["source_rows_per_symbol"]) <= 0 and not stage.get("dry_run") for stage in stages) else max(int(stage["source_rows_per_symbol"]) for stage in stages)
    if dry_only:
        max_source_rows = max(int(os.getenv("RAWSEQ_EVOLVE_DRY_RUN_SOURCE_ROWS", "2000")), max_source_rows)

    _, feature_audit_rows, target_manifest_rows = build_symbol_data_for_stage(
        symbols,
        source_root,
        max_source_rows,
        feature_windows,
        horizons,
        vol_window,
        severity_levels,
        stage_cache=stage_cache,
        heartbeat=heartbeat,
        stage_context={"stage_id": 0, "stage_name": "manifest", "source_rows_per_symbol": max_source_rows},
    )

    candidate_grid = build_candidate_grid(target_manifest_rows, allowed_feature_groups, allowed_models, allowed_lanes, regime_names, model_seeds)
    if max_candidates > 0:
        candidate_grid = candidate_grid[:max_candidates]
    checkpoint_config = {
        "source_root": str(source_root),
        "symbols": symbols,
        "feature_windows": feature_windows,
        "horizons": horizons,
        "severity_levels": severity_levels,
        "volatility_window": vol_window,
        "min_rows": min_rows,
        "min_prevalence": min_prevalence,
        "allowed_target_lanes": sorted(allowed_lanes),
        "allowed_feature_groups": allowed_feature_groups,
        "allowed_models": allowed_models,
        "allowed_regimes": regime_names,
        "model_seeds": model_seeds,
        "stage_preset": preset_name,
        "stages": stages,
        "keep_fraction": keep_fraction,
        "min_keep": min_keep,
        "max_candidates": max_candidates,
        "matrix_dtype": matrix_dtype.name,
        "stage_prep_cache_enabled": stage_cache_enabled,
        "stage_prep_cache_root": str(stage_cache_root),
        "stage_prep_cache_max_entries": stage_cache_max_entries,
        "heartbeat_enabled": heartbeat_enabled,
        "preprocessing_contract": PREPROCESSING_CONTRACT,
        "split_contract": SPLIT_CONTRACT,
        "calibration_contract": CALIBRATION_CONTRACT,
    }
    checkpoint_run_contract = build_checkpoint_run_contract(checkpoint_config, candidate_grid)
    checkpoint_contract_hash = checkpoint_store.write_or_validate_run_contract(checkpoint_run_contract, resume=bool(args.resume))
    active_keys: set[str] | None = set(row["candidate_key"] for row in candidate_grid)
    all_stage_rows: list[dict[str, Any]] = []
    all_stage_survivors: list[dict[str, Any]] = []
    matched_row_contract_rows: list[dict[str, Any]] = []
    exact_stage_audits: list[dict[str, Any]] = []
    halving_audit: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    prior_evaluated_keys: set[str] = set()
    matrix_cache = BoundedContentCache(cache_max_entries, cache_max_bytes) if cache_enabled else None
    matrix_telemetry = MatrixTelemetry(enabled=matrix_telemetry_enabled)
    memory_guard = MemoryGuard.from_env(args.memory_guard_policy)
    progress_rows: list[dict[str, Any]] = []
    stop_after_criteria = os.getenv("RAWSEQ_EVOLVE_CONTROLLED_STOP_AFTER_CRITERIA", "false").strip().lower() in {"1", "true", "yes", "on"}
    stop_min_nonbaseline = int(os.getenv("RAWSEQ_EVOLVE_STOP_MIN_NONBASELINE", "5"))
    stop_require_logistic = os.getenv("RAWSEQ_EVOLVE_STOP_REQUIRE_LOGISTIC", "true").strip().lower() in {"1", "true", "yes", "on"}
    stop_require_hgb = os.getenv("RAWSEQ_EVOLVE_STOP_REQUIRE_HGB", "true").strip().lower() in {"1", "true", "yes", "on"}
    completed_nonbaseline = 0
    completed_logistic = 0
    completed_hgb = 0

    if dry_only:
        active_keys = set(row["candidate_key"] for row in candidate_grid)
    else:
        for stage in stages:
            if stage.get("dry_run"):
                continue
            by_symbol, stage_feature_audit_rows, stage_target_manifest_rows = build_symbol_data_for_stage(
                symbols,
                source_root,
                int(stage["source_rows_per_symbol"]),
                feature_windows,
                horizons,
                vol_window,
                severity_levels,
                stage_cache=stage_cache,
                heartbeat=heartbeat,
                stage_context={"stage_id": int(stage["stage_id"]), "stage_name": str(stage["stage_name"]), "source_rows_per_symbol": int(stage["source_rows_per_symbol"])},
            )
            for row in stage_feature_audit_rows:
                row["stage_id"] = stage["stage_id"]
                row["stage_name"] = stage["stage_name"]
                row["source_rows_per_symbol"] = stage["source_rows_per_symbol"]
            feature_audit_rows.extend(stage_feature_audit_rows)
            requested_candidates = candidate_records_for_active_keys(candidate_grid, active_keys)
            rows: list[dict[str, Any]] = []
            coverage_rows: list[dict[str, Any]] = []
            resumed_keys: list[str] = []
            recomputed_keys: list[str] = []
            completed_keys = checkpoint_store.completed_candidate_keys(int(stage["stage_id"])) if args.resume else set()
            candidate_loop_start = time.perf_counter()
            for candidate_index, candidate in enumerate(requested_candidates, start=1):
                candidate_key_value = str(candidate["candidate_key"])
                guard = memory_guard.check(
                    {
                        "stage_id": stage["stage_id"],
                        "stage_name": stage["stage_name"],
                        "candidate_index": candidate_index,
                        "candidate_total": len(requested_candidates),
                        "candidate_key": candidate_key_value,
                    }
                )
                if guard["memory_guard_status"] != "MEMORY_GUARD_OK":
                    write_json_atomic(out_dir / "memory_guard_status.json", guard)
                    print(guard["memory_guard_status"])
                    if guard["memory_guard_should_stop"]:
                        return 3 if guard["memory_guard_status"] == "MEMORY_GUARD_FAILED" else 0
                checkpoint_payload = checkpoint_store.read_candidate_record(int(stage["stage_id"]), candidate_key_value) if candidate_key_value in completed_keys else None
                if checkpoint_payload:
                    heartbeat.emit(
                        "candidate_checkpoint_reuse",
                        stage_id=stage["stage_id"],
                        stage_name=stage["stage_name"],
                        candidate_key=candidate_key_value,
                        target_name=candidate.get("target_name"),
                        feature_group=candidate.get("feature_group"),
                        model=candidate.get("model"),
                        cache_status="checkpoint_hit",
                    )
                    candidate_rows = list(checkpoint_payload.get("stage_rows", []))
                    candidate_coverage = list(checkpoint_payload.get("coverage_rows", []))
                    resumed_keys.append(candidate_key_value)
                else:
                    heartbeat.emit(
                        "matched_mask_construction",
                        stage_id=stage["stage_id"],
                        stage_name=stage["stage_name"],
                        candidate_key=candidate_key_value,
                        target_name=candidate.get("target_name"),
                        feature_group=candidate.get("feature_group"),
                        model=candidate.get("model"),
                        fold="all",
                    )
                    heartbeat.emit(
                        "candidate_matrix_construction",
                        stage_id=stage["stage_id"],
                        stage_name=stage["stage_name"],
                        candidate_key=candidate_key_value,
                        target_name=candidate.get("target_name"),
                        feature_group=candidate.get("feature_group"),
                        model=candidate.get("model"),
                        fold="all",
                    )
                    heartbeat.emit(
                        "model_fitting",
                        stage_id=stage["stage_id"],
                        stage_name=stage["stage_name"],
                        candidate_key=candidate_key_value,
                        target_name=candidate.get("target_name"),
                        feature_group=candidate.get("feature_group"),
                        model=candidate.get("model"),
                        fold="all",
                    )
                    candidate_rows, candidate_coverage = board.evaluate_candidate_records(
                        by_symbol,
                        [candidate],
                        int(stage["eval_rows_per_symbol"]),
                        min_rows,
                        min_prevalence,
                        int(stage["folds"]),
                        matrix_cache=matrix_cache,
                        matrix_telemetry=matrix_telemetry,
                        matrix_dtype=matrix_dtype,
                        semantic_contract={
                            "preprocessing_contract": PREPROCESSING_CONTRACT,
                            "split_contract": SPLIT_CONTRACT,
                            "calibration_contract": CALIBRATION_CONTRACT,
                            "stage_id": stage["stage_id"],
                            "source_rows_per_symbol": stage["source_rows_per_symbol"],
                            "eval_rows_per_symbol": stage["eval_rows_per_symbol"],
                            "candidate_key": candidate_key_value,
                        },
                    )
                    checkpoint_store.write_candidate_record(
                        int(stage["stage_id"]),
                        candidate_key_value,
                        {
                            "stage_rows": candidate_rows,
                            "coverage_rows": candidate_coverage,
                            "candidate_index": candidate_index,
                            "candidate_count": len(requested_candidates),
                            "contract_hash": checkpoint_contract_hash,
                            "status": "complete",
                        },
                    )
                    for fold_row in candidate_rows:
                        checkpoint_store.write_fold_record(
                            int(stage["stage_id"]),
                            candidate_key_value,
                            fold_row.get("fold_id", "unknown"),
                            {
                                "metric_row": fold_row,
                                "contract_hash": checkpoint_contract_hash,
                                "status": fold_row.get("status", ""),
                                "failure_reason": fold_row.get("failure_reason", ""),
                                "large_matrices_stored": False,
                            },
                        )
                    recomputed_keys.append(candidate_key_value)
                    if str(candidate.get("model")) != "constant_prevalence":
                        completed_nonbaseline += 1
                    if str(candidate.get("model")) == "regularized_logistic":
                        completed_logistic += 1
                    if str(candidate.get("model")) == "shallow_hgb":
                        completed_hgb += 1
                if candidate_index == 1 or candidate_index % progress_every == 0 or candidate_index == len(requested_candidates):
                    elapsed = max(1e-9, time.perf_counter() - candidate_loop_start)
                    rate = candidate_index / elapsed
                    remaining = (len(requested_candidates) - candidate_index) / rate if rate > 0 else math.nan
                    progress = progress_payload(
                        {
                            "stage_index": int(stage["stage_id"]),
                            "stage_total": len([s for s in stages if not s.get("dry_run")]),
                            "candidate_index": candidate_index,
                            "candidate_total": len(requested_candidates),
                            "candidate_key": candidate_key_value,
                            "target_name": candidate.get("target_name"),
                            "target_lane": candidate.get("target_lane"),
                            "feature_group": candidate.get("feature_group"),
                            "model": candidate.get("model"),
                            "model_seed": candidate.get("model_seed"),
                            "fold": "all",
                            "elapsed_seconds": round(time.perf_counter() - start, 3),
                            "recent_candidate_rate_per_sec": round(rate, 6),
                            "estimated_remaining_seconds": round(remaining, 3) if math.isfinite(remaining) else math.nan,
                        },
                        matrix_cache.stats() if matrix_cache is not None else {},
                        str(checkpoint_store.candidate_record_path(int(stage["stage_id"]), candidate_key_value)),
                    )
                    progress_rows.append(progress)
                    print(format_progress_line(progress))
                rows.extend(candidate_rows)
                coverage_rows.extend(candidate_coverage)
                if stop_after_criteria and not args.resume:
                    meets_stop = (
                        completed_nonbaseline >= stop_min_nonbaseline
                        and (completed_logistic > 0 or not stop_require_logistic)
                        and (completed_hgb > 0 or not stop_require_hgb)
                    )
                    if meets_stop:
                        next_candidate = requested_candidates[candidate_index] if candidate_index < len(requested_candidates) else {}
                        snapshot = {
                            "controlled_interruption": True,
                            "stage_id": stage["stage_id"],
                            "stage_name": stage["stage_name"],
                            "last_completed_key": candidate_key_value,
                            "active_incomplete_key": next_candidate.get("candidate_key", ""),
                            "checkpoint_path": str(checkpoint_store.candidate_record_path(int(stage["stage_id"]), candidate_key_value)),
                            "checkpoint_contract_hash": checkpoint_contract_hash,
                            "completed_nonbaseline_candidates": completed_nonbaseline,
                            "completed_logistic_candidates": completed_logistic,
                            "completed_hgb_candidates": completed_hgb,
                            "memory_state": process_memory_snapshot(),
                            "resume_command_requires_same_contract": True,
                        }
                        write_json_atomic(out_dir / "controlled_interruption_snapshot.json", snapshot)
                        print("CONTROLLED_INTERRUPTION_AFTER_CHECKPOINT")
                        return 0
            for row in rows:
                row["stage_id"] = stage["stage_id"]
                row["stage_name"] = stage["stage_name"]
                row["candidate_key"] = str(row.get("candidate_key") or result_key(row))
                row["source_rows_per_symbol"] = stage["source_rows_per_symbol"]
                row["eval_rows_per_symbol"] = stage["eval_rows_per_symbol"]
                row["validation_scope"] = "pooled_chronological_discovery_only"
                row["checkpoint_resume_source"] = "resumed" if row["candidate_key"] in set(resumed_keys) else "computed"
            for row in coverage_rows:
                row["stage_id"] = stage["stage_id"]
                row["stage_name"] = stage["stage_name"]
                row["source_rows_per_symbol"] = stage["source_rows_per_symbol"]
                row["eval_rows_per_symbol"] = stage["eval_rows_per_symbol"]
                row["validation_scope"] = "pooled_chronological_discovery_only"
                row["checkpoint_resume_source"] = "resumed" if str(row.get("candidate_key")) in set(resumed_keys) else "computed"
            matched_row_contract_rows.extend(coverage_rows)
            evaluated_keys_this_stage = set(str(row.get("candidate_key")) for row in rows if row.get("candidate_key"))
            stage_survivors = aggregate_stage_results(rows, min_prevalence)
            for row in stage_survivors:
                row["stage_id"] = stage["stage_id"]
                row["stage_name"] = stage["stage_name"]
                row["source_rows_per_symbol"] = stage["source_rows_per_symbol"]
                row["eval_rows_per_symbol"] = stage["eval_rows_per_symbol"]
            keep, audit = select_survivor_keys(stage_survivors, keep_fraction, min_keep, advance_failed_for_diagnostics)
            stage_exact_audit = exact_candidate_stage_audit(
                int(stage["stage_id"]),
                str(stage["stage_name"]),
                prior_evaluated_keys,
                requested_candidates,
                rows,
                keep,
            )
            assert_exact_candidate_stage_audit_passes(stage_exact_audit)
            exact_stage_audits.append(stage_exact_audit)
            prior_evaluated_keys = evaluated_keys_this_stage
            for row in audit:
                row["stage_id"] = stage["stage_id"]
                row["stage_name"] = stage["stage_name"]
            all_stage_rows.extend(rows)
            all_stage_survivors.extend(stage_survivors)
            halving_audit.extend(audit)
            dropped.extend([row for row in audit if row.get("drop_reason")])
            checkpoint_store.write_stage_record(
                int(stage["stage_id"]),
                {
                    "stage_name": stage["stage_name"],
                    "stage_survivors": stage_survivors,
                    "survivor_keys": sorted(keep),
                    "halving_audit": audit,
                    "resumed_keys": sorted(resumed_keys),
                    "recomputed_keys": sorted(recomputed_keys),
                    "contract_hash": checkpoint_contract_hash,
                },
            )
            active_keys = keep
            if not active_keys:
                break

    final_stage_id = max([row.get("stage_id", 0) for row in all_stage_survivors], default=0)
    final_survivors = [row for row in all_stage_survivors if row.get("stage_id") == final_stage_id]
    scenario_rows: list[dict[str, Any]] = []
    scenario_survivors: list[dict[str, Any]] = []
    if scenario_validation_enabled and final_survivors and any(row.get("survives_stage_gate") for row in final_survivors):
        final_stage = next((stage for stage in reversed(stages) if int(stage.get("stage_id", -1)) == int(final_stage_id)), None)
        if final_stage and not final_stage.get("dry_run"):
            by_symbol, scenario_feature_audit_rows, _ = build_symbol_data_for_stage(
                symbols,
                source_root,
                int(final_stage["source_rows_per_symbol"]),
                feature_windows,
                horizons,
                vol_window,
                severity_levels,
                stage_cache=stage_cache,
                heartbeat=heartbeat,
                stage_context={"stage_id": int(final_stage["stage_id"]), "stage_name": str(final_stage["stage_name"]), "source_rows_per_symbol": int(final_stage["source_rows_per_symbol"]), "scenario_validation": True},
            )
            for row in scenario_feature_audit_rows:
                row["stage_id"] = final_stage["stage_id"]
                row["stage_name"] = final_stage["stage_name"]
                row["scenario_validation_feature_audit"] = True
            feature_audit_rows.extend(scenario_feature_audit_rows)
            scenario_rows = scenario_validation_rows(
                by_symbol,
                final_survivors,
                int(final_stage["eval_rows_per_symbol"]),
                min_rows,
                min_prevalence,
                int(final_stage["folds"]),
                max_loso_symbols,
            )
            scenario_survivors = aggregate_scenario_results(scenario_rows, min_prevalence)
            scenario_by_key = {row.get("candidate_key"): row for row in scenario_survivors}
            for row in final_survivors:
                scen = scenario_by_key.get(row.get("candidate_key"))
                row["scenario_validation_performed"] = bool(scen)
                row["scenario_survives_gate"] = bool(scen and scen.get("scenario_survives_gate"))
                row["scenario_validation_scope"] = "scenario_validation_loso_time_combined" if scen else "scenario_validation_missing"
                row["scenario_worst_brier_skill"] = scen.get("worst_scenario_brier_skill") if scen else math.nan
                row["scenario_rejection_reasons"] = scen.get("scenario_rejection_reasons") if scen else "scenario_validation_missing"
    else:
        for row in final_survivors:
            row["scenario_validation_performed"] = False
            row["scenario_survives_gate"] = False
            row["scenario_validation_scope"] = "pooled_chronological_discovery_only"
            row["scenario_worst_brier_skill"] = math.nan
            row["scenario_rejection_reasons"] = "scenario_validation_not_run"
    recommendations = [
        {
            "candidate_key": row.get("candidate_key"),
            "recommended_action": (
                "fixed_confirmation_required"
                if row.get("survives_stage_gate") and (not scenario_validation_enabled or row.get("scenario_survives_gate"))
                else "do_not_confirm"
            ),
            "target_lane": row.get("target_lane"),
            "target_name": row.get("target_name"),
            "feature_group": row.get("feature_group"),
            "model": row.get("model"),
            "fitness_score": row.get("fitness_score"),
            "scenario_validation_performed": row.get("scenario_validation_performed"),
            "scenario_survives_gate": row.get("scenario_survives_gate"),
            "scenario_validation_scope": row.get("scenario_validation_scope"),
            "scenario_worst_brier_skill": row.get("scenario_worst_brier_skill"),
            "scenario_rejection_reasons": row.get("scenario_rejection_reasons"),
            **SAFETY_FLAGS,
        }
        for row in final_survivors[:20]
    ]
    contract = {
        "created_at": now_stamp(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "symbols": symbols,
        "feature_windows": feature_windows,
        "horizons": horizons,
        "severity_levels": severity_levels,
        "volatility_window": vol_window,
        "development_cutoff_iso": "2026-05-31T23:59:00Z",
        "stage_preset": preset_name,
        "stages": stages,
        "keep_fraction": keep_fraction,
        "min_keep": min_keep,
        "advance_failed_candidates_for_diagnostics": advance_failed_for_diagnostics,
        "candidate_grid_rows": len(candidate_grid),
        "allowed_target_lanes": sorted(allowed_lanes),
        "allowed_feature_groups": allowed_feature_groups,
        "allowed_models": allowed_models,
        "allowed_regimes": regime_names,
        "model_seeds": model_seeds,
        "coverage_grid": "0.10,0.20,0.40",
        "abstention_diagnostics_supported": True,
        "matrix_dtype": matrix_dtype.name,
        "matrix_cache_enabled": cache_enabled,
        "matrix_cache_max_entries": cache_max_entries,
        "matrix_cache_max_bytes": cache_max_bytes,
        "matrix_cache_stats": matrix_cache.stats() if matrix_cache is not None else {},
        "stage_prep_cache_enabled": stage_cache_enabled,
        "stage_prep_cache_root": str(stage_cache_root),
        "stage_prep_cache_max_entries": stage_cache_max_entries,
        "stage_prep_cache_stats": stage_cache.stats() if stage_cache is not None else {},
        "matrix_telemetry_enabled": matrix_telemetry_enabled,
        "heartbeat_enabled": heartbeat_enabled,
        "memory_snapshot_at_summary": process_memory_snapshot(),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_contract_hash": checkpoint_contract_hash,
        "checkpoint_resume_requested": bool(args.resume),
        "checkpoint_every_candidates": int(args.checkpoint_every_candidates),
        "memory_guard_policy": args.memory_guard_policy,
        "progress_every_candidates": progress_every,
        "candidate_lineage_fields": ["parent_candidate_key", "mutation_family", "mutation_detail", "lineage_generation"],
        "candidate_identity_fields": [
            "target_lane",
            "target_name",
            "horizon_minutes",
            "interval_start_minutes",
            "interval_end_minutes",
            "threshold_vol_units",
            "feature_group",
            "regime_name",
            "model",
            "model_seed",
            "preprocessing_contract",
            "split_contract",
            "calibration_contract",
        ],
        "exact_successive_halving_semantics": "stage_survivor_candidate_keys_are_filtered_before_matrix_construction",
        "longer_horizon_directional_targets_supported": "set RAWSEQ_EVOLVE_TARGET_LANES=directional_return and RAWSEQ_EVOLVE_HORIZONS=15,30,60,120",
        "selection_rule": "successive_halving_validation_only_stability_aware_feature_family_evolution_v1",
        "validation_scope": "scenario_validation_loso_time_combined" if scenario_rows else "pooled_chronological_discovery_only",
        "scenario_validation_enabled": scenario_validation_enabled,
        "scenario_validation_performed": bool(scenario_rows),
        "scenario_max_loso_symbols": max_loso_symbols,
        "holdout_used_for_selection": False,
        "frozen_models_mutated": False,
        "dashboard_mutated": False,
        "active_future_shadow_mutated": False,
        **SAFETY_FLAGS,
    }
    contract["contract_hash"] = stable_hash(contract)
    logistic_policy = {
        "regularized_logistic_convergence_required_for_advancement": True,
        "non_converged_status": "TRAIN_FAILED",
        "telemetry_fields": [
            "logistic_convergence_status",
            "logistic_iteration_count",
            "logistic_max_iter",
            "logistic_warning_category",
            "logistic_warning_message",
            "logistic_solver",
            "logistic_regularization",
            "logistic_feature_count",
            "logistic_train_rows",
            "model_eligible_for_advancement",
        ],
    }
    failed_candidate_policy = {
        "failed_candidates_can_advance_by_default": False,
        "data_failed_can_advance": False,
        "train_failed_can_advance": False,
        "feature_group_unavailable_can_advance": False,
        "nan_metric_candidates_can_advance": False,
        "constant_prevalence_role": "baseline_only_not_survivor",
        "diagnostic_override_env": "RAWSEQ_EVOLVE_ADVANCE_FAILED_FOR_DIAGNOSTICS",
    }
    barrier_ordering_policy = {
        "barrier_first_source": "one_minute_candles_only",
        "same_candle_up_and_down_hit": "ambiguous_same_minute_nan",
        "trade_reconstruction_implemented_here": False,
        "fail_closed_when_order_unresolved": True,
    }
    determinism_audit = {
        "candidate_grid_sort": "candidate_order_tuple",
        "survivor_sort": "status_then_score_then_metrics_then_identity_fields",
        "fold_order": "ascending_fold_id",
        "symbol_order": "sorted_symbol_names",
        "candidate_key_hash": "stable_hash_first_16_hex_over_full_identity_contract",
        "model_seed_recorded": True,
        "holdout_used_for_selection": False,
    }

    write_csv(out_dir / "candidate_grid.csv", candidate_grid)
    write_csv(out_dir / "stage_results.csv", all_stage_rows)
    write_csv(out_dir / "stage_survivors.csv", all_stage_survivors)
    write_csv(out_dir / "scenario_results.csv", scenario_rows)
    write_csv(out_dir / "scenario_survivors.csv", scenario_survivors)
    write_csv(out_dir / "feature_family_rollup.csv", rollup(final_survivors, "feature_group"))
    write_csv(out_dir / "target_family_rollup.csv", rollup(final_survivors, "target_lane"))
    write_csv(out_dir / "successive_halving_audit.csv", halving_audit)
    write_csv(out_dir / "matched_row_contract_audit.csv", matched_row_contract_rows)
    write_csv(out_dir / "progress_telemetry.csv", progress_rows)
    heartbeat.write_csv(out_dir / "heartbeat_telemetry.csv")
    if stage_cache is not None:
        stage_cache.write_events_csv(out_dir / "stage_prep_cache_telemetry.csv")
    matrix_telemetry.write_csv(out_dir / "matrix_memory_telemetry.csv")
    write_csv(out_dir / "dropped_candidates.csv", dropped)
    write_csv(out_dir / "feature_audit.csv", feature_audit_rows)
    write_csv(out_dir / "target_manifest.csv", target_manifest_rows)
    write_json(out_dir / "recommended_next_confirmations.json", recommendations)
    write_json(out_dir / "exact_candidate_semantics_audit.json", {"stages": exact_stage_audits})
    write_json(out_dir / "logistic_convergence_policy.json", logistic_policy)
    write_json(out_dir / "failed_candidate_policy.json", failed_candidate_policy)
    write_json(out_dir / "barrier_ordering_policy.json", barrier_ordering_policy)
    write_json(out_dir / "determinism_audit.json", determinism_audit)
    write_json(out_dir / "feature_family_evolution_contract.json", contract)
    write_json_atomic(
        out_dir / "cache_contract.json",
        {
            "matrix_cache_enabled": cache_enabled,
            "matrix_cache_max_entries": cache_max_entries,
            "matrix_cache_max_bytes": cache_max_bytes,
            "cache_key_fields": [
                "source/frame identity",
                "symbol",
                "cutoff via frame identity",
                "feature columns",
                "target column",
                "preprocessing contract",
                "split contract",
                "calibration contract",
                "dtype",
            ],
            "fitted_preprocessors_cached_across_folds": False,
            "cache_stats": matrix_cache.stats() if matrix_cache is not None else {},
            "stage_prep_cache_enabled": stage_cache_enabled,
            "stage_prep_cache_root": str(stage_cache_root),
            "stage_prep_cache_stats": stage_cache.stats() if stage_cache is not None else {},
        },
    )
    write_summary(out_dir / "evolution_summary.txt", contract, final_survivors, dropped)

    print(f"output_dir={out_dir}")
    print(f"candidate_grid_rows={len(candidate_grid)}")
    print(f"stage_result_rows={len(all_stage_rows)}")
    print(f"final_survivor_rows={len(final_survivors)}")
    print(f"passing_final_survivors={sum(1 for row in final_survivors if row.get('survives_stage_gate'))}")
    print(f"scenario_result_rows={len(scenario_rows)}")
    print(f"passing_scenario_survivors={sum(1 for row in scenario_survivors if row.get('scenario_survives_gate'))}")
    print(f"runtime_seconds={time.perf_counter() - start:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
