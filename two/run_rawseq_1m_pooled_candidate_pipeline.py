#!/usr/bin/env python3
"""Confirm and freeze the selected 1m pooled multisymbol challenger.

This wrapper does not run a new model search. It resolves the existing selected
panel row (`conservative_probability_ensemble`, `row_weighted`) and reruns only
that fixed ensemble with a larger deterministic sample. If the predeclared gates
pass, it freezes a research candidate packet and creates June/July 2026 future
holdout contracts without opening those future candle files.
"""

from __future__ import annotations

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

from scripts.tiny.rawseq_1m_baseline_utils import DEFAULT_SOURCE_PATH, SAFETY_FLAGS, env_path, now_stamp, stable_hash, write_csv, write_json
from scripts.tiny.report_rawseq_1m_cross_asset_fixed_transfer import latest_dir, load_eligible_symbols, read_json, safe_float, truthy
from scripts.tiny.report_rawseq_1m_cross_asset_panel_scout import (
    SymbolData,
    evaluate_predictions,
    fit_hgb,
    fit_logistic,
    prepare_symbol,
    predict_model,
    regime_feature_indices,
    save_reload_parity,
    stack_data,
)

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_cross_asset_scout")
SELECTED_MODEL = "conservative_probability_ensemble"
SELECTED_WEIGHTING = "row_weighted"
COMPONENT_MODELS = ["pooled_logistic", "pooled_regime_feature_logistic", "pooled_shallow_hgb"]
COMPONENT_WEIGHTS = {"pooled_logistic": 1.0 / 3.0, "pooled_regime_feature_logistic": 1.0 / 3.0, "pooled_shallow_hgb": 1.0 / 3.0}


def selected_panel_row(panel_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(panel_dir / "pooled_panel_leaderboard.csv")
    rows = df[
        (df["scenario"] == "combined_symbol_time_holdout")
        & (df["model"] == SELECTED_MODEL)
        & (df["weighting"] == SELECTED_WEIGHTING)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one selected panel row, found {len(rows)}")
    row = rows.iloc[0].to_dict()
    if not (safe_float(row.get("worst_fold_brier_skill")) > 0 and int(row.get("fold_wins", 0)) == int(row.get("folds", -1))):
        raise RuntimeError("Selected panel row does not satisfy the original combined symbol/time gate")
    return row


def fit_selected_components(
    contract: dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, tuple[bool, float]]]:
    regime_idx = regime_feature_indices(list(contract["model_feature_names_and_order"]))
    models = {
        "pooled_logistic": (fit_logistic(train_x, train_y), val_x),
        "pooled_regime_feature_logistic": (fit_logistic(train_x[:, regime_idx], train_y), val_x[:, regime_idx]),
        "pooled_shallow_hgb": (fit_hgb(contract, train_x, train_y), val_x),
    }
    predictions: dict[str, np.ndarray] = {}
    parity: dict[str, tuple[bool, float]] = {}
    fitted: dict[str, Any] = {}
    for name, (model, x_val) in models.items():
        fitted[name] = model
        predictions[name] = np.clip(predict_model(model, x_val), 1e-6, 1 - 1e-6)
        parity[name] = save_reload_parity(model, x_val)
    return fitted, predictions, parity


def evaluate_selected_scenario(
    scenario: str,
    fold_id: str,
    train_symbols: list[str],
    validation_symbols: list[str],
    train_range: tuple[int, int],
    validation_range: tuple[int, int],
    by_symbol: dict[str, SymbolData],
    contract: dict[str, Any],
    max_train_rows_per_symbol: int,
    max_validation_rows_per_symbol: int,
) -> list[dict[str, Any]]:
    feature_cols = list(contract["model_feature_names_and_order"])
    train_x, train_y, _ = stack_data(by_symbol, train_symbols, train_range[0], train_range[1], feature_cols, max_train_rows_per_symbol, "tail")
    val_x, val_y, _ = stack_data(by_symbol, validation_symbols, validation_range[0], validation_range[1], feature_cols, max_validation_rows_per_symbol, "head")
    base = {
        "scenario": scenario,
        "fold_id": fold_id,
        "train_symbols": ",".join(train_symbols),
        "validation_symbols": ",".join(validation_symbols),
        "train_start_index": train_range[0],
        "train_end_index": train_range[1],
        "validation_start_index": validation_range[0],
        "validation_end_index": validation_range[1],
        "train_rows": int(len(train_y)),
        "validation_rows": int(len(val_y)),
        "train_symbol_count": len(set(train_symbols)),
        "validation_symbol_count": len(set(validation_symbols)),
        "training_prevalence": float(np.mean(train_y)) if len(train_y) else math.nan,
        "fixed_transfer_contract_hash": contract["fixed_transfer_contract_hash"],
        "selected_candidate_model": SELECTED_MODEL,
        "selected_candidate_weighting": SELECTED_WEIGHTING,
        "symbol_identifier_used": False,
        "holdout_accessed": False,
        "gpu_used": False,
        **SAFETY_FLAGS,
        "public_recorded_data_only": True,
        "frozen_kraken_candidate_reused": False,
        "future_data_accessed": False,
    }
    if len(train_y) < 100 or len(val_y) < 100 or len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
        return [{**base, "model": SELECTED_MODEL, "weighting": SELECTED_WEIGHTING, "status": "DATA_FAILED", "failure_reason": "insufficient rows or class diversity"}]
    baseline = np.full(len(val_y), np.clip(float(np.mean(train_y)), 1e-6, 1 - 1e-6))
    _, predictions, parity = fit_selected_components(contract, train_x, train_y, val_x)
    rows = []
    for name, pred in predictions.items():
        rows.append(evaluate_predictions({**base, "model": name, "weighting": SELECTED_WEIGHTING, "component_weight": COMPONENT_WEIGHTS[name]}, val_y, pred, baseline, parity[name]))
    ensemble_pred = sum(COMPONENT_WEIGHTS[name] * predictions[name] for name in COMPONENT_MODELS)
    rows.append(
        evaluate_predictions(
            {**base, "model": SELECTED_MODEL, "weighting": SELECTED_WEIGHTING, "component_weights_json": json.dumps(COMPONENT_WEIGHTS, sort_keys=True)},
            val_y,
            np.clip(ensemble_pred, 1e-6, 1 - 1e-6),
            baseline,
            (all(parity[name][0] for name in COMPONENT_MODELS), max(parity[name][1] for name in COMPONENT_MODELS)),
        )
    )
    return rows


def aggregate_selected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "OK":
            grouped.setdefault((str(row["scenario"]), str(row["model"])), []).append(row)
    out = []
    for (scenario, model), vals in grouped.items():
        skills = [safe_float(row.get("brier_skill_vs_prevalence")) for row in vals]
        log_imps = [safe_float(row.get("log_loss_improvement_vs_prevalence")) for row in vals]
        pr_lifts = [safe_float(row.get("pr_auc_lift_over_event_prevalence")) for row in vals]
        out.append(
            {
                "scenario": scenario,
                "model": model,
                "weighting": SELECTED_WEIGHTING,
                "folds": len(vals),
                "fold_wins": sum(1 for x in skills if x > 0),
                "fold_win_fraction": sum(1 for x in skills if x > 0) / len(vals) if vals else math.nan,
                "median_brier_skill": float(np.nanmedian(skills)) if skills else math.nan,
                "worst_fold_brier_skill": float(np.nanmin(skills)) if skills else math.nan,
                "median_log_loss_improvement": float(np.nanmedian(log_imps)) if log_imps else math.nan,
                "median_pr_auc_lift": float(np.nanmedian(pr_lifts)) if pr_lifts else math.nan,
                "save_reload_parity_all": all(truthy(row.get("save_reload_prediction_parity")) for row in vals),
                "symbol_identifier_used": False,
                "holdout_accessed": False,
                "gpu_used": False,
                **SAFETY_FLAGS,
            }
        )
    out.sort(key=lambda r: (r["scenario"], r["model"]))
    return out


def confirmation_pass(leaderboard: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    failures = []
    required = {
        "leave_one_symbol_out": 9,
        "leave_one_time_block_out": 4,
        "combined_symbol_time_holdout": 9,
    }
    by_scenario = {row["scenario"]: row for row in leaderboard if row["model"] == SELECTED_MODEL}
    for scenario, expected_folds in required.items():
        row = by_scenario.get(scenario)
        if not row:
            failures.append(f"{scenario}: missing selected ensemble row")
            continue
        if int(row["folds"]) != expected_folds:
            failures.append(f"{scenario}: folds {row['folds']} != {expected_folds}")
        if int(row["fold_wins"]) != int(row["folds"]):
            failures.append(f"{scenario}: wins {row['fold_wins']} != folds {row['folds']}")
        if safe_float(row["median_brier_skill"]) <= 0:
            failures.append(f"{scenario}: median_brier_skill <= 0")
        if safe_float(row["worst_fold_brier_skill"]) <= 0:
            failures.append(f"{scenario}: worst_fold_brier_skill <= 0")
        if safe_float(row["median_log_loss_improvement"]) <= 0:
            failures.append(f"{scenario}: median_log_loss_improvement <= 0")
        if safe_float(row["median_pr_auc_lift"]) <= 0:
            failures.append(f"{scenario}: median_pr_auc_lift <= 0")
        if not truthy(row["save_reload_parity_all"]):
            failures.append(f"{scenario}: save_reload_parity_all failed")
    return not failures, failures


def train_final_candidate(
    by_symbol: dict[str, SymbolData],
    symbols: list[str],
    contract: dict[str, Any],
    max_train_rows_per_symbol: int,
) -> dict[str, Any]:
    feature_cols = list(contract["model_feature_names_and_order"])
    common_end = min(data.rolling_end_index for data in by_symbol.values())
    train_x, train_y, train_s = stack_data(by_symbol, symbols, 0, common_end, feature_cols, max_train_rows_per_symbol, "tail")
    regime_idx = regime_feature_indices(feature_cols)
    models = {
        "pooled_logistic": fit_logistic(train_x, train_y),
        "pooled_regime_feature_logistic": fit_logistic(train_x[:, regime_idx], train_y),
        "pooled_shallow_hgb": fit_hgb(contract, train_x, train_y),
    }
    return {
        "models": models,
        "regime_feature_indices": regime_idx,
        "train_rows": int(len(train_y)),
        "train_symbols": symbols,
        "train_symbol_rows": {symbol: int((train_s == symbol).sum()) for symbol in symbols},
        "training_prevalence": float(np.mean(train_y)) if len(train_y) else math.nan,
        "common_rolling_end_index": common_end,
    }


def write_freeze_packet(
    out_dir: Path,
    final_fit: dict[str, Any],
    contract: dict[str, Any],
    confirmation_hash: str,
    source_root: Path,
) -> tuple[str, str]:
    freeze_dir = out_dir / "frozen_pooled_multisymbol_challenger"
    freeze_dir.mkdir(parents=True, exist_ok=False)
    model_path = freeze_dir / "pooled_candidate_components.pkl"
    model_path.write_bytes(pickle.dumps(final_fit["models"]))
    candidate_contract = {
        "created_at": now_stamp(),
        "candidate_kind": "pooled_multisymbol_1m_downside_risk_challenger",
        "status": "frozen_research_candidate_before_future_holdout",
        "confirmation_hash": confirmation_hash,
        "fixed_transfer_contract_hash": contract["fixed_transfer_contract_hash"],
        "model_path": str(model_path),
        "component_models": COMPONENT_MODELS,
        "component_weights": COMPONENT_WEIGHTS,
        "weighting": SELECTED_WEIGHTING,
        "calibration": "none",
        "symbol_identifier_used": False,
        "train_rows": final_fit["train_rows"],
        "train_symbols": final_fit["train_symbols"],
        "train_symbol_rows": final_fit["train_symbol_rows"],
        "training_prevalence": final_fit["training_prevalence"],
        "common_rolling_end_index": final_fit["common_rolling_end_index"],
        "source_root": str(source_root),
        "source_data_end_inclusive": "2026-05-31T23:59:00+00:00",
        "post_may_2026_data_opened": False,
        "june_or_july_evaluated": False,
        "safety": {
            **SAFETY_FLAGS,
            "public_recorded_data_only": True,
            "active_future_shadow_mutation": False,
            "active_future_shadow_labels_used": False,
            "frozen_kraken_candidate_reused": False,
            "future_data_accessed": False,
        },
    }
    candidate_contract["candidate_hash"] = stable_hash(candidate_contract)
    write_json(freeze_dir / "pooled_candidate_contract.json", candidate_contract)
    acceptance = {
        "created_at": now_stamp(),
        "candidate_hash": candidate_contract["candidate_hash"],
        "future_holdout_months": ["2026-06", "2026-07"],
        "future_holdout_contracts": [
            {
                "month": "2026-06",
                "source_pattern": str(source_root / "{symbol}-1m-2026-06.zip"),
                "must_not_be_opened_before_freeze": True,
            },
            {
                "month": "2026-07",
                "source_pattern": str(source_root / "{symbol}-1m-2026-07.zip"),
                "must_not_be_opened_before_freeze": True,
            },
        ],
        "acceptance_rules": {
            "all_symbols_or_declared_eligible_subset": True,
            "loso_worst_fold_brier_skill_gt": 0.0,
            "time_block_worst_fold_brier_skill_gt": 0.0,
            "combined_symbol_time_worst_fold_brier_skill_gt": 0.0,
            "median_log_loss_improvement_gt": 0.0,
            "median_pr_auc_lift_gt": 0.0,
            "save_reload_parity_required": True,
            "no_recalibration_or_threshold_change": True,
        },
        "june_or_july_files_opened": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    acceptance["future_acceptance_rule_hash"] = stable_hash(acceptance)
    write_json(freeze_dir / "future_holdout_acceptance_rules.json", acceptance)
    return candidate_contract["candidate_hash"], acceptance["future_acceptance_rule_hash"]


def main() -> int:
    started = time.perf_counter()
    output_root = env_path("RAWSEQ_1M_CROSS_ASSET_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    source_root = env_path("RAWSEQ_1M_SOURCE_PATH", DEFAULT_SOURCE_PATH)
    contract_dir = Path(os.getenv("RAWSEQ_1M_TRANSFER_CONTRACT_DIR", "").strip() or latest_dir(output_root, "rawseq_1m_fixed_transfer_contract_*"))
    inventory_dir = Path(os.getenv("RAWSEQ_1M_MULTISYMBOL_INVENTORY_DIR", "").strip() or latest_dir(output_root, "rawseq_1m_multisymbol_inventory_*"))
    panel_dir = Path(os.getenv("RAWSEQ_1M_PANEL_DIR", "").strip() or latest_dir(output_root, "rawseq_1m_panel_scout_*"))
    out_dir = Path(os.getenv("RAWSEQ_1M_POOLED_CONFIRM_OUTPUT_DIR", "").strip() or output_root / f"rawseq_1m_pooled_candidate_confirmation_{now_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=False)
    contract = read_json(contract_dir / "fixed_transfer_contract.json")
    selected = selected_panel_row(panel_dir)
    symbols = load_eligible_symbols(inventory_dir)
    max_rows = int(os.getenv("RAWSEQ_1M_POOLED_CONFIRM_MAX_SOURCE_ROWS", "0") or "0")
    max_train_rows_per_symbol = int(os.getenv("RAWSEQ_1M_POOLED_CONFIRM_ROWS_PER_SYMBOL", "50000") or "50000")
    max_validation_rows_per_symbol = int(os.getenv("RAWSEQ_1M_POOLED_CONFIRM_ROWS_PER_SYMBOL", "50000") or "50000")
    by_symbol = {}
    for symbol in symbols:
        print(f"[prepare] {symbol}", flush=True)
        by_symbol[symbol] = prepare_symbol(symbol, source_root, contract, max_rows)
    common_end = min(data.rolling_end_index for data in by_symbol.values())
    folds = next(iter(by_symbol.values())).folds
    latest_fold = max(folds, key=lambda row: int(row["fold_id"]))
    rows: list[dict[str, Any]] = []
    for excluded in symbols:
        print(f"[confirm] LOSO excluded={excluded}", flush=True)
        rows.extend(evaluate_selected_scenario("leave_one_symbol_out", excluded, [s for s in symbols if s != excluded], [excluded], (0, common_end), (0, common_end), by_symbol, contract, max_train_rows_per_symbol, max_validation_rows_per_symbol))
    for fold in folds:
        print(f"[confirm] time_block fold={fold['fold_id']}", flush=True)
        rows.extend(evaluate_selected_scenario("leave_one_time_block_out", str(fold["fold_id"]), symbols, symbols, (int(fold["train_start_index"]), int(fold["train_end_index"])), (int(fold["validation_start_index"]), int(fold["validation_end_index"])), by_symbol, contract, max_train_rows_per_symbol, max_validation_rows_per_symbol))
    for excluded in symbols:
        print(f"[confirm] combined excluded={excluded}", flush=True)
        rows.extend(evaluate_selected_scenario("combined_symbol_time_holdout", excluded, [s for s in symbols if s != excluded], [excluded], (int(latest_fold["train_start_index"]), int(latest_fold["train_end_index"])), (int(latest_fold["validation_start_index"]), int(latest_fold["validation_end_index"])), by_symbol, contract, max_train_rows_per_symbol, max_validation_rows_per_symbol))
    leaderboard = aggregate_selected(rows)
    passed, failures = confirmation_pass(leaderboard)
    runtime_seconds = time.perf_counter() - started
    peak_mb = math.nan
    try:
        import psutil  # type: ignore

        peak_mb = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        pass
    manifest = {
        "created_at": now_stamp(),
        "selected_panel_dir": str(panel_dir),
        "selected_panel_row": selected,
        "contract_dir": str(contract_dir),
        "inventory_dir": str(inventory_dir),
        "source_root": str(source_root),
        "symbols": symbols,
        "component_models": COMPONENT_MODELS,
        "component_weights": COMPONENT_WEIGHTS,
        "selected_model": SELECTED_MODEL,
        "selected_weighting": SELECTED_WEIGHTING,
        "rows_per_symbol_per_scenario_leg": max_train_rows_per_symbol,
        "runtime_seconds": runtime_seconds,
        "peak_working_set_mb": peak_mb,
        "confirmation_pass": passed,
        "confirmation_failures": failures,
        "post_may_2026_data_opened": False,
        "june_or_july_evaluated": False,
        "cpu_only": True,
        "gpu_used": False,
        "safety": {
            **SAFETY_FLAGS,
            "public_recorded_data_only": True,
            "active_future_shadow_mutation": False,
            "active_future_shadow_labels_used": False,
            "frozen_kraken_candidate_reused": False,
            "future_data_accessed": False,
        },
    }
    manifest["confirmation_hash"] = stable_hash({"manifest": manifest, "leaderboard": leaderboard})
    candidate_hash = ""
    future_hash = ""
    if passed:
        final_fit = train_final_candidate(by_symbol, symbols, contract, max_train_rows_per_symbol)
        candidate_hash, future_hash = write_freeze_packet(out_dir, final_fit, contract, manifest["confirmation_hash"], source_root)
        manifest["candidate_hash"] = candidate_hash
        manifest["future_acceptance_rule_hash"] = future_hash
        manifest["freeze_status"] = "frozen"
    else:
        manifest["freeze_status"] = "not_frozen"
        write_json(out_dir / "failure_attribution.json", {"confirmation_failures": failures, "leaderboard": leaderboard})
    write_csv(out_dir / "pooled_candidate_confirmation_fold_metrics.csv", rows)
    write_csv(out_dir / "pooled_candidate_confirmation_leaderboard.csv", leaderboard)
    write_json(out_dir / "pooled_candidate_confirmation_manifest.json", manifest)
    lines = [
        "Rawseq 1m pooled candidate confirmation",
        f"Output: {out_dir}",
        f"Selected: {SELECTED_MODEL} / {SELECTED_WEIGHTING}",
        f"Rows per symbol per scenario leg: {max_train_rows_per_symbol}",
        f"Confirmation pass: {passed}",
        f"Runtime seconds: {runtime_seconds:.3f}",
        f"Peak working set MB: {peak_mb:.3f}" if math.isfinite(peak_mb) else "Peak working set MB: unavailable",
        f"Candidate hash: {candidate_hash or 'not_frozen'}",
        f"Future acceptance-rule hash: {future_hash or 'not_created'}",
        f"Post-May-2026 data opened: {manifest['post_may_2026_data_opened']}",
        "",
        "Selected ensemble leaderboard:",
    ]
    for row in [r for r in leaderboard if r["model"] == SELECTED_MODEL]:
        lines.append(
            f"- {row['scenario']}: wins={row['fold_wins']}/{row['folds']} median_brier_skill={row['median_brier_skill']} "
            f"worst={row['worst_fold_brier_skill']} log_loss_improvement={row['median_log_loss_improvement']} pr_auc_lift={row['median_pr_auc_lift']}"
        )
    if failures:
        lines.append("Failures: " + "; ".join(failures))
    lines.append("Safety: CPU-only, paper-only, no private API, no orders, no promotion, no champion mutation, no June/July evaluation.")
    (out_dir / "pooled_candidate_confirmation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
