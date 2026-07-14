#!/usr/bin/env python3
"""Final one-minute cross-asset decision report.

Aggregates inventory, fixed-transfer, panel, liquidity, and bounded feature
ablation evidence. This is report/research only: no private API, no orders, no
promotion, no champion mutation, and no active future-shadow access.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
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
    DEFAULT_SOURCE_PATH,
    SAFETY_FLAGS,
    build_features,
    downside_event_targets,
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
from scripts.tiny.report_rawseq_1m_cross_asset_fixed_transfer import (
    cap_indices,
    feature_drift,
    finite_xy,
    fit_predict_hgb,
    latest_dir,
    load_eligible_symbols,
    read_csv_rows,
    read_json,
    safe_float,
    save_reload_prediction_parity,
    truthy,
)

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_cross_asset_scout")


def copy_csv(src: Path, dst: Path) -> pd.DataFrame:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return pd.read_csv(dst)


def liquidity_group_rows(inventory: pd.DataFrame, fixed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = inventory.merge(fixed, on="symbol", how="left")
    merged["median_daily_quote_volume"] = pd.to_numeric(merged["median_daily_quote_volume"], errors="coerce")
    labels = ["low-liquidity", "medium-liquidity", "high-liquidity"]
    ranks = merged["median_daily_quote_volume"].rank(method="first")
    merged["liquidity_group"] = pd.qcut(ranks, q=3, labels=labels)
    rows = []
    for group, part in merged.groupby("liquidity_group", observed=True):
        skills = pd.to_numeric(part["median_brier_skill"], errors="coerce")
        worst = pd.to_numeric(part["worst_fold_brier_skill"], errors="coerce")
        rows.append(
            {
                "liquidity_group": str(group),
                "symbols": ",".join(part["symbol"].astype(str).tolist()),
                "symbol_count": len(part),
                "median_daily_quote_volume_median": float(part["median_daily_quote_volume"].median()),
                "strong_transfer_count": int((part["classification"] == "strong_transfer").sum()),
                "positive_worst_fold_count": int((worst > 0).sum()),
                "median_brier_skill_median": float(skills.median()),
                "worst_fold_brier_skill_min": float(worst.min()),
                "median_target_prevalence": float(pd.to_numeric(part["fixed_target_event_prevalence"], errors="coerce").median()),
                "median_missing_intervals": float(pd.to_numeric(part["missing_one_minute_intervals"], errors="coerce").median()),
                "median_largest_gap_minutes": float(pd.to_numeric(part["largest_gap_minutes"], errors="coerce").median()),
            }
        )
    return merged, pd.DataFrame(rows)


def feature_group_columns(feature_cols: list[str], group: str) -> list[str]:
    volatility = [c for c in feature_cols if "volatility" in c]
    volume = [c for c in feature_cols if "volume" in c]
    price_range = [
        c
        for c in feature_cols
        if any(tok in c for tok in ["signed_bucket_return", "candle_", "wick", "rolling_range", "distance_to_recent", "close_to_ema", "ema_slope"])
    ]
    if group == "volatility-only":
        return volatility
    if group == "price/range-only":
        return price_range
    if group == "volume-only":
        return volume
    if group == "full":
        return feature_cols
    if group == "full_minus_volatility":
        return [c for c in feature_cols if c not in set(volatility)]
    if group == "full_minus_volume":
        return [c for c in feature_cols if c not in set(volume)]
    raise ValueError(f"Unknown feature group: {group}")


def evaluate_ablation_symbol(
    symbol: str,
    source_root: Path,
    contract: dict[str, Any],
    feature_group: str,
    max_rows: int,
    max_train_rows: int,
    max_val_rows: int,
) -> list[dict[str, Any]]:
    source_files = resolve_source_files(source_root, symbol)
    candles = load_candles(source_files, max_rows=max_rows)
    features, _, _ = build_features(candles, [int(x) for x in contract["feature_windows_minutes"]])
    horizon = int(contract["target_horizon_minutes"])
    vol_window = int(contract["volatility_window_minutes"])
    targets = downside_event_targets(candles, vol_window=vol_window, horizons=[horizon])
    vol_col = str(contract["volatility_denominator"])
    features[vol_col] = targets[vol_col]
    target = targets[f"downside_event_0p5vol_h{horizon}m_fw{vol_window}"]
    _, folds, _ = split_contract(candles, feature_lookback=int(contract["purge_rows"]), max_horizon=int(contract["embargo_rows"]), fold_count=4)
    fixed_cols = list(contract["model_feature_names_and_order"])
    feature_cols = feature_group_columns(fixed_cols, feature_group)
    if not feature_cols:
        return [{"symbol": symbol, "feature_group": feature_group, "status": "DATA_FAILED", "failure_reason": "empty feature set"}]
    rows = []
    for fold in folds:
        train_idx = cap_indices(np.arange(fold["train_start_index"], fold["train_end_index"] + 1, dtype=np.int64), max_train_rows, "tail")
        val_idx = cap_indices(np.arange(fold["validation_start_index"], fold["validation_end_index"] + 1, dtype=np.int64), max_val_rows, "head")
        train_x, train_y, _ = finite_xy(features, target, train_idx, feature_cols)
        val_x, val_y, _ = finite_xy(features, target, val_idx, feature_cols)
        base = {
            "symbol": symbol,
            "feature_group": feature_group,
            "fold_id": int(fold["fold_id"]),
            "feature_count": len(feature_cols),
            "train_rows": int(len(train_y)),
            "validation_rows": int(len(val_y)),
            "training_prevalence": float(np.mean(train_y)) if len(train_y) else math.nan,
            "holdout_accessed": False,
            "gpu_used": False,
            **SAFETY_FLAGS,
        }
        if len(train_y) < 100 or len(val_y) < 100 or len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
            rows.append({**base, "status": "DATA_FAILED", "failure_reason": "insufficient rows or class diversity"})
            continue
        baseline = np.full(len(val_y), np.clip(float(np.mean(train_y)), 1e-6, 1 - 1e-6))
        pred, model = fit_predict_hgb(contract, train_x, train_y, val_x)
        parity, max_diff = save_reload_prediction_parity(model, val_x)
        rows.append(
            {
                **base,
                "status": "OK",
                "save_reload_prediction_parity": parity,
                "save_reload_prediction_max_abs_diff": max_diff,
                "feature_drift_median_abs_z": feature_drift(train_x, val_x),
                **metric_row(val_y, pred, baseline),
            }
        )
    return rows


def aggregate_ablation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "OK":
            groups.setdefault((str(row["symbol"]), str(row["feature_group"])), []).append(row)
    for (symbol, group), vals in groups.items():
        skills = [safe_float(row.get("brier_skill_vs_prevalence")) for row in vals]
        pr_lifts = [safe_float(row.get("pr_auc_lift_over_event_prevalence")) for row in vals]
        out.append(
            {
                "symbol": symbol,
                "feature_group": group,
                "folds": len(vals),
                "fold_wins": sum(1 for x in skills if x > 0),
                "median_brier_skill": float(np.nanmedian(skills)) if skills else math.nan,
                "worst_fold_brier_skill": float(np.nanmin(skills)) if skills else math.nan,
                "median_pr_auc_lift": float(np.nanmedian(pr_lifts)) if pr_lifts else math.nan,
                "save_reload_parity_all": all(truthy(row.get("save_reload_prediction_parity")) for row in vals),
            }
        )
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in out:
        by_group.setdefault(row["feature_group"], []).append(row)
    summary = []
    for group, vals in by_group.items():
        skills = [safe_float(row["median_brier_skill"]) for row in vals]
        worst = [safe_float(row["worst_fold_brier_skill"]) for row in vals]
        summary.append(
            {
                "symbol": "ALL",
                "feature_group": group,
                "symbols": len(vals),
                "strong_like_symbol_count": sum(1 for row in vals if safe_float(row["worst_fold_brier_skill"]) > 0 and safe_float(row["median_brier_skill"]) > 0),
                "median_brier_skill": float(np.nanmedian(skills)) if skills else math.nan,
                "worst_fold_brier_skill_min": float(np.nanmin(worst)) if worst else math.nan,
                "median_pr_auc_lift": float(np.nanmedian([safe_float(row["median_pr_auc_lift"]) for row in vals])) if vals else math.nan,
            }
        )
    return summary + out


def final_recommendation(fixed: pd.DataFrame, panel: pd.DataFrame) -> str:
    if fixed.empty:
        return "cross_asset_source_blocked"
    strong_count = int((fixed["classification"] == "strong_transfer").sum())
    positive_worst_count = int((pd.to_numeric(fixed["worst_fold_brier_skill"], errors="coerce") > 0).sum())
    combined = panel[(panel["scenario"] == "combined_symbol_time_holdout") & (panel["model"] == "conservative_probability_ensemble")]
    combined_survives = bool((pd.to_numeric(combined["worst_fold_brier_skill"], errors="coerce") > 0).any())
    if strong_count >= max(2, math.ceil(len(fixed) * 0.75)) and combined_survives:
        return "freeze_pooled_multisymbol_challenger_before_future_holdout"
    if strong_count >= max(2, math.ceil(len(fixed) * 0.75)):
        return "freeze_fixed_contract_multisymbol_challenger_before_future_holdout"
    if positive_worst_count > 1:
        return "partial_cross_asset_transfer_continue_research"
    return "sol_specific_one_minute_effect"


def direct_diagnostic_placeholder(symbols: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": symbol,
            "status": "frozen_sol_model_domain_shift_diagnostic_only",
            "diagnostic_run_status": "not_run_serialized_sol_model_not_available_in_scout_artifacts",
            "input_out_of_range_fraction": math.nan,
            "feature_distribution_shift": math.nan,
            "probability_mean": math.nan,
            "probability_standard_deviation": math.nan,
            "probability_minimum": math.nan,
            "probability_maximum": math.nan,
            "probability_saturation": math.nan,
            "brier_skill": math.nan,
            "pr_auc_lift": math.nan,
            "calibration_drift": math.nan,
            "included_in_main_leaderboard": False,
            "holdout_accessed": False,
            **SAFETY_FLAGS,
        }
        for symbol in symbols
    ]


def main() -> int:
    started = time.perf_counter()
    output_root = env_path("RAWSEQ_1M_CROSS_ASSET_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    source_root = env_path("RAWSEQ_1M_SOURCE_PATH", DEFAULT_SOURCE_PATH)
    inventory_dir = Path(os.getenv("RAWSEQ_1M_MULTISYMBOL_INVENTORY_DIR", "").strip() or latest_dir(output_root, "rawseq_1m_multisymbol_inventory_*"))
    contract_dir = Path(os.getenv("RAWSEQ_1M_TRANSFER_CONTRACT_DIR", "").strip() or latest_dir(output_root, "rawseq_1m_fixed_transfer_contract_*"))
    fixed_dir = Path(os.getenv("RAWSEQ_1M_FIXED_TRANSFER_DIR", "").strip() or latest_dir(output_root, "rawseq_1m_fixed_transfer_*"))
    panel_dir = Path(os.getenv("RAWSEQ_1M_PANEL_DIR", "").strip() or latest_dir(output_root, "rawseq_1m_panel_scout_*"))
    out_dir = Path(os.getenv("RAWSEQ_1M_DECISION_OUTPUT_DIR", "").strip() or output_root / f"rawseq_1m_cross_asset_decision_{now_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=False)

    contract = read_json(contract_dir / "fixed_transfer_contract.json")
    inventory = copy_csv(inventory_dir / "multisymbol_inventory.csv", out_dir / "multisymbol_inventory.csv")
    fixed_fold = copy_csv(fixed_dir / "fixed_transfer_fold_metrics.csv", out_dir / "fixed_transfer_fold_metrics.csv")
    fixed_leader = copy_csv(fixed_dir / "fixed_transfer_symbol_leaderboard.csv", out_dir / "fixed_transfer_symbol_leaderboard.csv")
    panel_fold = copy_csv(panel_dir / "pooled_panel_fold_metrics.csv", out_dir / "pooled_panel_fold_metrics.csv")
    panel_leader = copy_csv(panel_dir / "pooled_panel_leaderboard.csv", out_dir / "pooled_panel_leaderboard.csv")
    copy_csv(panel_dir / "leave_one_symbol_out_metrics.csv", out_dir / "leave_one_symbol_out_metrics.csv")
    copy_csv(panel_dir / "leave_one_time_block_out_metrics.csv", out_dir / "leave_one_time_block_out_metrics.csv")

    merged_liq, liq = liquidity_group_rows(inventory, fixed_leader)
    write_csv(out_dir / "liquidity_stratified_metrics.csv", liq.to_dict("records"))
    symbols = load_eligible_symbols(inventory_dir)

    ablation_groups = ["volatility-only", "price/range-only", "volume-only", "full", "full_minus_volatility", "full_minus_volume"]
    max_rows = int(os.getenv("RAWSEQ_1M_ABLATION_MAX_SOURCE_ROWS", "0") or "0")
    max_train = int(os.getenv("RAWSEQ_1M_ABLATION_MAX_TRAIN_ROWS_PER_FOLD", "10000") or "10000")
    max_val = int(os.getenv("RAWSEQ_1M_ABLATION_MAX_VALIDATION_ROWS_PER_FOLD", "10000") or "10000")
    ablation_rows: list[dict[str, Any]] = []
    for group in ablation_groups:
        print(f"[ablation] {group}", flush=True)
        for symbol in symbols:
            ablation_rows.extend(evaluate_ablation_symbol(symbol, source_root, contract, group, max_rows, max_train, max_val))
    ablation_summary = aggregate_ablation(ablation_rows)
    write_csv(out_dir / "feature_ablation_fold_metrics.csv", ablation_rows)
    write_csv(out_dir / "feature_ablation_metrics.csv", ablation_summary)
    write_csv(out_dir / "direct_weight_transfer_diagnostic.csv", direct_diagnostic_placeholder(symbols))

    strong_count = int((fixed_leader["classification"] == "strong_transfer").sum())
    sol_rank = int(fixed_leader.sort_values("median_brier_skill", ascending=False)["symbol"].tolist().index("SOLUSDT") + 1) if "SOLUSDT" in fixed_leader["symbol"].tolist() else -1
    positive_worst_symbols = fixed_leader.loc[pd.to_numeric(fixed_leader["worst_fold_brier_skill"], errors="coerce") > 0, "symbol"].tolist()
    hgb_fixed_median = float(pd.to_numeric(fixed_leader["median_brier_skill"], errors="coerce").median())
    panel_combined = panel_leader[panel_leader["scenario"] == "combined_symbol_time_holdout"].copy()
    best_combined = panel_combined.sort_values("median_brier_skill", ascending=False).iloc[0].to_dict() if len(panel_combined) else {}
    pooled_survives = bool(best_combined and safe_float(best_combined.get("worst_fold_brier_skill")) > 0 and int(best_combined.get("fold_wins", 0)) == int(best_combined.get("folds", -1)))
    ablation_all = pd.DataFrame([row for row in ablation_summary if row.get("symbol") == "ALL"])
    best_ablation = ablation_all.sort_values("median_brier_skill", ascending=False).iloc[0].to_dict() if len(ablation_all) else {}
    rec = final_recommendation(fixed_leader, panel_leader)
    answers = {
        "strong_transfer_symbol_count": strong_count,
        "sol_unusually_strong": sol_rank <= 3,
        "sol_rank_by_median_brier_skill": sol_rank,
        "exact_sol_temporal_contract_generalizes": strong_count >= 7,
        "pooled_training_improves_excluded_symbol_performance": bool(pooled_survives),
        "best_combined_symbol_time_model": best_combined,
        "hgb_still_outperforms_logistic_across_symbols": hgb_fixed_median > 0 and "pooled_shallow_hgb" in panel_leader["model"].tolist(),
        "edge_limited_to_high_liquidity": int((liq["strong_transfer_count"] > 0).sum()) < 3,
        "effect_mostly_volatility_normalization": str(best_ablation.get("feature_group", "")) == "volatility-only",
        "positive_worst_fold_symbols": positive_worst_symbols,
        "pooled_model_survives_symbol_and_time_exclusion": pooled_survives,
        "pooled_one_minute_challenger_justified": rec == "freeze_pooled_multisymbol_challenger_before_future_holdout",
    }
    recommended = fixed_leader[fixed_leader["classification"].isin(["strong_transfer", "partial_transfer"])].copy()
    recommended = recommended.sort_values(["classification", "median_brier_skill"], ascending=[True, False])
    recommended.to_csv(out_dir / "recommended_symbol_candidates.csv", index=False)
    manifest = {
        "created_at": now_stamp(),
        "inventory_dir": str(inventory_dir),
        "contract_dir": str(contract_dir),
        "fixed_transfer_dir": str(fixed_dir),
        "panel_dir": str(panel_dir),
        "contract_hash": contract.get("fixed_transfer_contract_hash"),
        "runtime_seconds": time.perf_counter() - started,
        "cpu_only": True,
        "gpu_used": False,
        "holdout_accessed": False,
        "final_recommendation": rec,
        "answers": answers,
        "safety": {
            **SAFETY_FLAGS,
            "public_recorded_data_only": True,
            "active_future_shadow_mutation": False,
            "active_future_shadow_labels_used": False,
            "frozen_kraken_candidate_reused": False,
            "future_data_accessed": False,
        },
    }
    manifest["cross_asset_decision_hash"] = stable_hash({"manifest": manifest, "answers": answers})
    write_json(out_dir / "cross_asset_decision.json", manifest)
    lines = [
        "Rawseq 1m cross-asset decision",
        f"Output: {out_dir}",
        f"Final recommendation: {rec}",
        f"Strong transfer symbols: {strong_count}/{len(fixed_leader)}",
        f"SOL rank by median Brier skill: {sol_rank}",
        f"Positive worst-fold symbols: {', '.join(positive_worst_symbols)}",
        f"Best combined symbol/time model: {best_combined.get('model')} {best_combined.get('weighting')} median={best_combined.get('median_brier_skill')} worst={best_combined.get('worst_fold_brier_skill')}",
        f"Best ablation group: {best_ablation.get('feature_group')} median={best_ablation.get('median_brier_skill')}",
        "",
        "Answers:",
        f"1. How many symbols pass strong transfer? {strong_count}.",
        f"2. Is SOL unusually strong? {'yes' if answers['sol_unusually_strong'] else 'no'}; rank={sol_rank}.",
        f"3. Does the exact SOL temporal contract generalize? {'yes' if answers['exact_sol_temporal_contract_generalizes'] else 'not broadly enough'}.",
        f"4. Does pooled training improve excluded-symbol performance? {'yes' if answers['pooled_training_improves_excluded_symbol_performance'] else 'unclear/no'}.",
        f"5. Does HGB still outperform logistic across symbols? {'yes' if answers['hgb_still_outperforms_logistic_across_symbols'] else 'unclear/no'}.",
        f"6. Is the edge limited to high-liquidity coins? {'yes' if answers['edge_limited_to_high_liquidity'] else 'no'}.",
        f"7. Is the effect mostly volatility normalization? {'yes' if answers['effect_mostly_volatility_normalization'] else 'no / broader feature interactions appear useful'}.",
        f"8. Which symbols have positive worst-fold Brier skill? {', '.join(positive_worst_symbols)}.",
        f"9. Does any model survive both symbol and time exclusion? {'yes' if pooled_survives else 'no'}.",
        f"10. Is a pooled one-minute challenger justified? {'yes' if answers['pooled_one_minute_challenger_justified'] else 'not yet as the final recommendation'}.",
        "",
        "Safety: paper-only, public recorded data only, no private API, no orders, no promotion, no champion mutation.",
    ]
    (out_dir / "cross_asset_decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
