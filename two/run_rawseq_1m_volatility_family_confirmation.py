#!/usr/bin/env python3
"""Focused confirmation for a predeclared volatility-family board member."""

from __future__ import annotations

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
    add_cross_asset_features,
    add_regime_features,
    add_short_path_features,
    build_target_lanes,
    cap_tail,
    filter_development,
)

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_volatility_family_confirmations")
PRIMARY_TARGET = "volatility_expansion_future_vol_gt_current_h15m_fw240"
SECONDARY_TARGET = "range_expansion_future_range_gt_recent_median_h5m_fw240"
TARGETS = [PRIMARY_TARGET, SECONDARY_TARGET]
FREEZE_GATE = {
    "primary_median_brier_skill_min": 0.08,
    "primary_worst_brier_skill_min": 0.05,
    "primary_combined_worst_brier_skill_min": 0.03,
    "positive_symbols_min": 8,
    "primary_pr_auc_lift_min": 0.15,
    "secondary_median_brier_skill_min": 0.0,
    "secondary_worst_brier_skill_min": 0.0,
    "expanded_feature_median_brier_delta_min": 0.01,
    "expanded_feature_worst_brier_delta_min": 0.0,
    "max_expected_calibration_error": 0.08,
    "max_calibration_error": 0.25,
    "dominant_symbol_contribution_max_fraction": 0.30,
    "dominant_time_block_contribution_max_fraction": 0.40,
}


@dataclass
class SymbolData:
    symbol: str
    candles: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
    folds: list[dict[str, Any]]
    rolling_end_index: int


def read_symbol(symbol: str, source_root: Path, max_rows: int, feature_windows: list[int]) -> SymbolData:
    candles = load_candles(resolve_source_files(source_root, symbol), max_rows=0)
    candles = cap_tail(filter_development(candles, DEVELOPMENT_CUTOFF_MS), max_rows)
    features, _, _ = build_features(candles, feature_windows)
    features = add_regime_features(candles, add_short_path_features(candles, features, feature_windows), feature_windows)
    targets, _ = build_target_lanes(candles, horizons=[1, 2, 4, 8], vol_window=240, severity_levels=[0.5])
    split, folds, _ = split_contract(candles, feature_lookback=max(feature_windows + [240]), max_horizon=15, fold_count=4)
    features["trailing_volatility_bps_fw240"] = targets["trailing_volatility_bps_fw240"]
    return SymbolData(symbol, candles, features, targets, folds, int(split["rolling_development_end_index"]))


def feature_columns(features: pd.DataFrame, feature_group: str) -> list[str]:
    tokens = {
        "existing": [
            "signed_bucket",
            "candle_",
            "wick",
            "volume",
            "rolling_",
            "distance_to_recent",
            "close_to_ema",
            "ema_slope",
            "trailing_volatility",
        ],
        "expanded": [
            "signed_bucket",
            "candle_",
            "wick",
            "volume",
            "rolling_",
            "distance_to_recent",
            "close_to_ema",
            "ema_slope",
            "trailing_volatility",
            "acceleration",
            "compression",
            "consecutive",
            "vwap",
            "body_to_range",
            "wick_asymmetry",
            "regime",
            "time_of_day",
            "weekend",
            "trend_vs_range",
            "btc_",
            "eth_",
            "market_",
            "cross_asset",
            "symbols_positive",
            "relative_to_btc",
        ],
    }[feature_group]
    return sorted([c for c in features.columns if c not in {"timestamp", "timestamp_ms"} and any(tok in c for tok in tokens)])


def finite_xy(data: SymbolData, indices: np.ndarray, feature_cols: list[str], target_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = data.features.iloc[indices].reindex(columns=feature_cols).to_numpy(dtype=np.float64)
    y = pd.to_numeric(data.targets.iloc[indices][target_col], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[mask], y[mask]


def stack_xy(
    by_symbol: dict[str, SymbolData],
    symbols: list[str],
    start: int,
    end: int,
    feature_cols: list[str],
    target_col: str,
    max_rows_per_symbol: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    for symbol in symbols:
        data = by_symbol[symbol]
        stop = min(end, len(data.features) - 1)
        if stop < start:
            continue
        idx = np.arange(start, stop + 1, dtype=np.int64)
        if max_rows_per_symbol > 0 and len(idx) > max_rows_per_symbol:
            idx = idx[-max_rows_per_symbol:]
        x, y = finite_xy(data, idx, feature_cols, target_col)
        if len(y):
            xs.append(x)
            ys.append(y)
            ss.append(np.asarray([symbol] * len(y), dtype=object))
    if not ys:
        return np.empty((0, len(feature_cols))), np.empty(0), np.empty(0, dtype=object)
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ss)


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


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, bool, float, Any]:
    model = fit_logistic(train_x, train_y)
    pred = predict_model(model, val_x)
    loaded = pickle.loads(pickle.dumps(model))
    pred2 = predict_model(loaded, val_x)
    diff = float(np.nanmax(np.abs(pred - pred2))) if len(pred) else 0.0
    return pred, diff <= 1e-12, diff, model


def evaluate_scenario(
    scenario: dict[str, Any],
    by_symbol: dict[str, SymbolData],
    feature_cols: list[str],
    feature_group: str,
    max_rows_per_symbol: int,
    min_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, target in [("primary", PRIMARY_TARGET), ("secondary_diagnostic", SECONDARY_TARGET)]:
        train_x, train_y, _ = stack_xy(by_symbol, scenario["train_symbols"], scenario["train_range"][0], scenario["train_range"][1], feature_cols, target, max_rows_per_symbol)
        val_x, val_y, _ = stack_xy(by_symbol, scenario["validation_symbols"], scenario["validation_range"][0], scenario["validation_range"][1], feature_cols, target, max_rows_per_symbol)
        base = {
            "scenario": scenario["scenario"],
            "fold_id": scenario["fold_id"],
            "train_symbols": ",".join(scenario["train_symbols"]),
            "validation_symbols": ",".join(scenario["validation_symbols"]),
            "target_role": role,
            "target_name": target,
            "feature_group": feature_group,
            "feature_count": len(feature_cols),
            "model": "regularized_logistic",
            "holdout_used_for_selection": False,
            "private_api": False,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
            **SAFETY_FLAGS,
        }
        if len(train_y) < min_rows or len(val_y) < min_rows or len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
            rows.append({**base, "status": "DATA_FAILED", "failure_reason": "insufficient rows or class diversity"})
            continue
        baseline = np.full(len(val_y), np.clip(float(np.mean(train_y)), 1e-6, 1 - 1e-6))
        pred, parity, diff, _ = fit_predict(train_x, train_y, val_x)
        rows.append(
            {
                **base,
                "status": "OK",
                "train_rows": int(len(train_y)),
                "validation_rows": int(len(val_y)),
                "training_prevalence": float(np.mean(train_y)),
                "save_reload_parity": parity,
                "save_reload_max_abs_diff": diff,
                **metric_row(val_y, pred, baseline),
            }
        )
    return rows


def contribution_fraction(group: pd.DataFrame, key: str) -> float:
    vals = group.copy()
    vals["improvement_mass"] = pd.to_numeric(vals["brier_skill_vs_prevalence"], errors="coerce").clip(lower=0.0) * pd.to_numeric(vals["rows"], errors="coerce")
    total = float(vals["improvement_mass"].sum())
    if total <= 0:
        return math.nan
    return float(vals.groupby(key)["improvement_mass"].sum().max() / total)


def target_summary(metrics: list[dict[str, Any]], feature_group: str, target_name: str) -> dict[str, Any]:
    df = pd.DataFrame([r for r in metrics if r.get("feature_group") == feature_group and r.get("target_name") == target_name and r.get("status") == "OK"])
    if df.empty:
        return {"feature_group": feature_group, "target_name": target_name, "status": "NO_ROWS"}
    brier = pd.to_numeric(df["brier_skill_vs_prevalence"], errors="coerce")
    pr = pd.to_numeric(df["pr_auc_lift_over_event_prevalence"], errors="coerce")
    ll = pd.to_numeric(df["log_loss_improvement_vs_prevalence"], errors="coerce")
    ece = pd.to_numeric(df["expected_calibration_error"], errors="coerce")
    mce = pd.to_numeric(df["maximum_calibration_error"], errors="coerce")
    combined = df[df["scenario"] == "combined_symbol_time_exclusion"]
    loso = df[df["scenario"] == "leave_symbol_out"]
    positive_symbols = int((pd.to_numeric(loso["brier_skill_vs_prevalence"], errors="coerce") > 0).sum())
    return {
        "feature_group": feature_group,
        "target_name": target_name,
        "scenario_rows": int(len(df)),
        "median_brier_skill": float(brier.median()),
        "worst_brier_skill": float(brier.min()),
        "combined_worst_brier_skill": float(pd.to_numeric(combined["brier_skill_vs_prevalence"], errors="coerce").min()),
        "positive_symbols": positive_symbols,
        "median_pr_auc_lift": float(pr.median()),
        "median_log_loss_improvement": float(ll.median()),
        "max_expected_calibration_error": float(ece.max()),
        "max_calibration_error": float(mce.max()),
        "save_reload_parity_all": bool(df["save_reload_parity"].astype(str).str.lower().isin(["true", "1"]).all()),
        "dominant_symbol_contribution_fraction": contribution_fraction(loso, "validation_symbols"),
        "dominant_time_block_contribution_fraction": contribution_fraction(df[df["scenario"] == "leave_time_block_out"], "fold_id"),
    }


def apply_freeze_gate(summaries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(row["feature_group"], row["target_name"]): row for row in summaries}
    existing_primary = by_key.get(("existing", PRIMARY_TARGET), {})
    expanded_primary = by_key.get(("expanded", PRIMARY_TARGET), {})
    existing_secondary = by_key.get(("existing", SECONDARY_TARGET), {})
    expanded_secondary = by_key.get(("expanded", SECONDARY_TARGET), {})
    gated: list[dict[str, Any]] = []
    for group_name in ["existing", "expanded"]:
        primary = by_key.get((group_name, PRIMARY_TARGET), {})
        secondary = by_key.get((group_name, SECONDARY_TARGET), {})
        complexity_ok = True
        complexity_reason = ""
        if group_name == "expanded":
            delta_median = float(primary.get("median_brier_skill", math.nan)) - float(existing_primary.get("median_brier_skill", math.nan))
            delta_worst = float(primary.get("worst_brier_skill", math.nan)) - float(existing_primary.get("worst_brier_skill", math.nan))
            complexity_ok = delta_median >= FREEZE_GATE["expanded_feature_median_brier_delta_min"] and delta_worst >= FREEZE_GATE["expanded_feature_worst_brier_delta_min"]
            complexity_reason = f"expanded_delta_median={delta_median};expanded_delta_worst={delta_worst}"
        gates = {
            "primary_median_brier_skill": float(primary.get("median_brier_skill", -math.inf)) >= FREEZE_GATE["primary_median_brier_skill_min"],
            "primary_worst_brier_skill": float(primary.get("worst_brier_skill", -math.inf)) >= FREEZE_GATE["primary_worst_brier_skill_min"],
            "primary_combined_worst_brier_skill": float(primary.get("combined_worst_brier_skill", -math.inf)) >= FREEZE_GATE["primary_combined_worst_brier_skill_min"],
            "positive_symbols": int(primary.get("positive_symbols", 0)) >= FREEZE_GATE["positive_symbols_min"],
            "primary_pr_auc_lift": float(primary.get("median_pr_auc_lift", -math.inf)) >= FREEZE_GATE["primary_pr_auc_lift_min"],
            "log_loss_improvement": float(primary.get("median_log_loss_improvement", -math.inf)) > 0,
            "expected_calibration_error": float(primary.get("max_expected_calibration_error", math.inf)) <= FREEZE_GATE["max_expected_calibration_error"],
            "maximum_calibration_error": float(primary.get("max_calibration_error", math.inf)) <= FREEZE_GATE["max_calibration_error"],
            "save_reload_parity": bool(primary.get("save_reload_parity_all", False)),
            "dominant_symbol": float(primary.get("dominant_symbol_contribution_fraction", math.inf)) <= FREEZE_GATE["dominant_symbol_contribution_max_fraction"],
            "dominant_time_block": float(primary.get("dominant_time_block_contribution_fraction", math.inf)) <= FREEZE_GATE["dominant_time_block_contribution_max_fraction"],
            "secondary_median_brier_skill": float(secondary.get("median_brier_skill", -math.inf)) >= FREEZE_GATE["secondary_median_brier_skill_min"],
            "secondary_worst_brier_skill": float(secondary.get("worst_brier_skill", -math.inf)) >= FREEZE_GATE["secondary_worst_brier_skill_min"],
            "feature_complexity": complexity_ok,
        }
        gated.append(
            {
                "feature_group": group_name,
                **{f"primary_{k}": v for k, v in primary.items() if k not in {"feature_group", "target_name"}},
                "secondary_median_brier_skill": secondary.get("median_brier_skill", math.nan),
                "secondary_worst_brier_skill": secondary.get("worst_brier_skill", math.nan),
                "freeze_gate_pass": all(gates.values()),
                "freeze_gate_failures": ";".join(name for name, ok in gates.items() if not ok),
                "feature_complexity_reason": complexity_reason,
            }
        )
    winners = [row for row in gated if row["freeze_gate_pass"]]
    if winners:
        winners.sort(key=lambda row: (row["feature_group"] == "existing", row["primary_median_brier_skill"]), reverse=True)
        chosen = winners[0]["feature_group"]
    else:
        chosen = ""
    family = {
        "family_status": "PASS" if bool(chosen) else "FAIL",
        "freeze_allowed": bool(chosen),
        "chosen_feature_group": chosen,
        "primary_target": PRIMARY_TARGET,
        "secondary_target": SECONDARY_TARGET,
        "freeze_gate": FREEZE_GATE,
        "existing_primary_median_brier_skill": existing_primary.get("median_brier_skill", math.nan),
        "expanded_primary_median_brier_skill": expanded_primary.get("median_brier_skill", math.nan),
        "existing_secondary_median_brier_skill": existing_secondary.get("median_brier_skill", math.nan),
        "expanded_secondary_median_brier_skill": expanded_secondary.get("median_brier_skill", math.nan),
    }
    return gated, family


def final_train_artifact(by_symbol: dict[str, SymbolData], feature_cols: list[str], max_rows_per_symbol: int) -> dict[str, Any]:
    symbols = sorted(by_symbol)
    rolling_end = min(data.rolling_end_index for data in by_symbol.values())
    artifact: dict[str, Any] = {"feature_cols": feature_cols, "targets": TARGETS, "models": {}}
    for target in TARGETS:
        x, y, _ = stack_xy(by_symbol, symbols, 0, rolling_end, feature_cols, target, max_rows_per_symbol)
        artifact["models"][target] = fit_logistic(x, y)
        artifact["train_rows"] = int(len(y))
    return artifact


def write_freeze_packet(out_dir: Path, artifact: dict[str, Any], contract: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
    freeze_dir = out_dir / "frozen_volatility_family_challenger"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    model_path = freeze_dir / "volatility_family_models.pkl"
    model_path.write_bytes(pickle.dumps(artifact))
    packet = {
        **contract,
        "family_summary": family,
        "model_path": str(model_path),
        "model_sha256": stable_hash({"model_bytes": model_path.read_bytes().hex()}),
        "freeze_status": "frozen_research_challenger",
        "board_member_role": "volatility_context_member",
    }
    packet["candidate_hash"] = stable_hash(packet)
    write_json(freeze_dir / "volatility_family_candidate_contract.json", packet)
    return packet


def text_report(path: Path, gated: list[dict[str, Any]], family: dict[str, Any], contract: dict[str, Any]) -> None:
    lines = [
        "RAWSEQ 1M VOLATILITY-FAMILY CONFIRMATION",
        f"created_at={contract['created_at']}",
        f"family_status={family['family_status']}",
        f"freeze_allowed={family['freeze_allowed']}",
        f"chosen_feature_group={family['chosen_feature_group']}",
        f"primary_target={PRIMARY_TARGET}",
        f"secondary_diagnostic={SECONDARY_TARGET}",
        "",
        "Freeze gates by feature group:",
    ]
    for row in gated:
        lines.append(
            f"- feature_group={row['feature_group']} pass={row['freeze_gate_pass']} "
            f"primary_median={row.get('primary_median_brier_skill')} "
            f"primary_worst={row.get('primary_worst_brier_skill')} "
            f"secondary_median={row.get('secondary_median_brier_skill')} "
            f"failures={row['freeze_gate_failures']} {row['feature_complexity_reason']}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- This is a predeclared volatility-family confirmation, not a broad target tournament.",
            "- Primary target controls freeze eligibility; secondary target is diagnostic but must not fail basic stability.",
            "- Expanded features must beat existing features by the predeclared margin without worsening worst-fold skill.",
            "- Frozen dashboard/downside models were not changed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    source_root = env_path("RAWSEQ_VOL_SOURCE_PATH", DEFAULT_SOURCE_PATH)
    output_root = env_path("RAWSEQ_VOL_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    symbols = [s.strip().upper() for s in os.getenv("RAWSEQ_VOL_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",") if s.strip()]
    max_rows = int(os.getenv("RAWSEQ_VOL_MAX_ROWS_PER_SYMBOL", "50000"))
    train_cap = int(os.getenv("RAWSEQ_VOL_MAX_TRAIN_ROWS_PER_SYMBOL", "20000"))
    min_rows = int(os.getenv("RAWSEQ_VOL_MIN_ROWS", "500"))
    feature_windows = [60, 240]
    out_dir = output_root / f"rawseq_1m_volatility_family_confirmation_{now_stamp()}"
    by_symbol = {symbol: read_symbol(symbol, source_root, max_rows, feature_windows) for symbol in symbols}
    add_cross_asset_features({symbol: data.features for symbol, data in by_symbol.items()})
    metrics: list[dict[str, Any]] = []
    for scenario in scenario_defs(by_symbol):
        for group_name in ["existing", "expanded"]:
            cols = feature_columns(next(iter(by_symbol.values())).features, group_name)
            metrics.extend(evaluate_scenario(scenario, by_symbol, cols, group_name, train_cap, min_rows))
    summaries: list[dict[str, Any]] = []
    for group_name in ["existing", "expanded"]:
        for target in TARGETS:
            summaries.append(target_summary(metrics, group_name, target))
    gated, family = apply_freeze_gate(summaries)
    contract = {
        "created_at": now_stamp(),
        "source_root": str(source_root),
        "development_cutoff_iso": "2026-05-31T23:59:00Z",
        "symbols": symbols,
        "primary_target": PRIMARY_TARGET,
        "secondary_diagnostic_target": SECONDARY_TARGET,
        "feature_groups": ["existing", "expanded"],
        "feature_windows": feature_windows,
        "model_family": "regularized_logistic",
        "scenario_types": ["leave_symbol_out", "leave_time_block_out", "combined_symbol_time_exclusion"],
        "freeze_gate": FREEZE_GATE,
        "max_rows_per_symbol": max_rows,
        "max_train_rows_per_symbol": train_cap,
        "holdout_used_for_selection": False,
        "dashboard_mutated": False,
        "frozen_downside_model_mutated": False,
        **SAFETY_FLAGS,
    }
    contract["contract_hash"] = stable_hash(contract)
    write_csv(out_dir / "volatility_family_confirmation_metrics.csv", metrics)
    write_csv(out_dir / "volatility_family_target_summary.csv", summaries)
    write_csv(out_dir / "volatility_family_freeze_gates.csv", gated)
    write_json(out_dir / "volatility_family_confirmation_contract.json", contract)
    write_json(out_dir / "volatility_family_confirmation_summary.json", family)
    if family["freeze_allowed"]:
        cols = feature_columns(next(iter(by_symbol.values())).features, family["chosen_feature_group"])
        packet = write_freeze_packet(out_dir, final_train_artifact(by_symbol, cols, train_cap), contract, family)
        family["candidate_hash"] = packet["candidate_hash"]
        write_json(out_dir / "volatility_family_confirmation_summary.json", family)
    text_report(out_dir / "volatility_family_confirmation_report.txt", gated, family, contract)
    print(f"output_dir={out_dir}")
    print(f"family_status={family['family_status']}")
    print(f"freeze_allowed={family['freeze_allowed']}")
    print(f"chosen_feature_group={family['chosen_feature_group']}")
    if "candidate_hash" in family:
        print(f"candidate_hash={family['candidate_hash']}")
    print(f"runtime_seconds={time.perf_counter() - start:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
