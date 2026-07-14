#!/usr/bin/env python3
"""Focused confirmation for a 1m downside-severity probability family."""

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
    build_target_lanes,
    cap_tail,
    filter_development,
)

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_downside_severity_family_confirmations")
THRESHOLDS = [0.5, 1.0, 1.5, 2.0]
TARGETS = [f"downside_event_{str(level).replace('.', 'p')}vol_h1m_fw240" for level in THRESHOLDS]


@dataclass
class SymbolData:
    symbol: str
    candles: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
    folds: list[dict[str, Any]]
    rolling_end_index: int


def severity_token(level: float) -> str:
    return str(level).replace(".", "p")


def monotonic_project(probabilities: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(np.asarray(probabilities, dtype=float), axis=1)


def monotonic_violation_fraction(probabilities: np.ndarray) -> float:
    if len(probabilities) == 0:
        return math.nan
    diffs = np.diff(np.asarray(probabilities, dtype=float), axis=1)
    return float((diffs > 1e-12).any(axis=1).mean())


def read_symbol(symbol: str, source_root: Path, max_rows: int, feature_windows: list[int]) -> SymbolData:
    candles = load_candles(resolve_source_files(source_root, symbol), max_rows=0)
    candles = cap_tail(filter_development(candles, DEVELOPMENT_CUTOFF_MS), max_rows)
    features, _, _ = build_features(candles, feature_windows)
    targets, _ = build_target_lanes(candles, horizons=[1], vol_window=240, severity_levels=THRESHOLDS)
    split, folds, _ = split_contract(candles, feature_lookback=max(feature_windows + [240]), max_horizon=1, fold_count=4)
    features["trailing_volatility_bps_fw240"] = targets["trailing_volatility_bps_fw240"]
    return SymbolData(
        symbol=symbol,
        candles=candles,
        features=features,
        targets=targets,
        folds=folds,
        rolling_end_index=int(split["rolling_development_end_index"]),
    )


def existing_feature_columns(features: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in features.columns
        if c not in {"timestamp", "timestamp_ms"}
        and any(
            tok in c
            for tok in [
                "signed_bucket",
                "candle_",
                "wick",
                "volume",
                "rolling_",
                "distance_to_recent",
                "close_to_ema",
                "ema_slope",
                "trailing_volatility",
            ]
        )
    ]
    return sorted(cols)


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


def fit_family(train_x: np.ndarray, train_ys: dict[str, np.ndarray]) -> dict[str, Any]:
    return {target: fit_logistic(train_x, train_ys[target]) for target in TARGETS}


def predict_family(models: dict[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.column_stack([predict_model(models[target], x) for target in TARGETS])
    return raw, monotonic_project(raw)


def evaluate_scenario(
    scenario: dict[str, Any],
    by_symbol: dict[str, SymbolData],
    feature_cols: list[str],
    max_rows_per_symbol: int,
    min_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    train_x: np.ndarray | None = None
    val_x: np.ndarray | None = None
    train_ys: dict[str, np.ndarray] = {}
    val_ys: dict[str, np.ndarray] = {}
    val_symbols: np.ndarray | None = None
    for target in TARGETS:
        tx, ty, _ = stack_xy(
            by_symbol,
            scenario["train_symbols"],
            scenario["train_range"][0],
            scenario["train_range"][1],
            feature_cols,
            target,
            max_rows_per_symbol,
        )
        vx, vy, vs = stack_xy(
            by_symbol,
            scenario["validation_symbols"],
            scenario["validation_range"][0],
            scenario["validation_range"][1],
            feature_cols,
            target,
            max_rows_per_symbol,
        )
        if train_x is None:
            train_x = tx
            val_x = vx
            val_symbols = vs
        train_ys[target] = ty
        val_ys[target] = vy
    assert train_x is not None and val_x is not None and val_symbols is not None
    base = {
        "scenario": scenario["scenario"],
        "fold_id": scenario["fold_id"],
        "train_symbols": ",".join(scenario["train_symbols"]),
        "validation_symbols": ",".join(scenario["validation_symbols"]),
        "train_start_index": scenario["train_range"][0],
        "train_end_index": scenario["train_range"][1],
        "validation_start_index": scenario["validation_range"][0],
        "validation_end_index": scenario["validation_range"][1],
        "feature_count": len(feature_cols),
        "model_family": "regularized_logistic_independent_thresholds_monotonic_projected",
        "holdout_used_for_selection": False,
        **SAFETY_FLAGS,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    if any(len(train_ys[target]) < min_rows or len(val_ys[target]) < min_rows or len(np.unique(train_ys[target])) < 2 or len(np.unique(val_ys[target])) < 2 for target in TARGETS):
        return (
            [{**base, "threshold_vol_units": level, "target_name": target, "status": "DATA_FAILED", "failure_reason": "insufficient rows or class diversity"} for level, target in zip(THRESHOLDS, TARGETS)],
            {**base, "status": "DATA_FAILED", "raw_monotonic_violation_fraction": math.nan, "projected_monotonic_violation_fraction": math.nan},
            None,
        )
    models = fit_family(train_x, train_ys)
    raw_pred, pred = predict_family(models, val_x)
    mono = {
        **base,
        "status": "OK",
        "validation_rows": int(len(val_x)),
        "raw_monotonic_violation_fraction": monotonic_violation_fraction(raw_pred),
        "projected_monotonic_violation_fraction": monotonic_violation_fraction(pred),
    }
    rows: list[dict[str, Any]] = []
    for idx, (level, target) in enumerate(zip(THRESHOLDS, TARGETS)):
        y = val_ys[target]
        baseline = np.full(len(y), np.clip(float(np.mean(train_ys[target])), 1e-6, 1 - 1e-6))
        rows.append(
            {
                **base,
                "threshold_vol_units": level,
                "target_name": target,
                "status": "OK",
                "train_rows": int(len(train_ys[target])),
                "validation_rows": int(len(y)),
                "training_prevalence": float(np.mean(train_ys[target])),
                "projected_probability_used": True,
                **metric_row(y, pred[:, idx], baseline),
            }
        )
    artifact = {"models": models, "feature_cols": feature_cols}
    return rows, mono, artifact


def summarize(metrics: list[dict[str, Any]], monotonicity: list[dict[str, Any]], min_positive_symbols: int, max_ece: float, max_mce: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = pd.DataFrame(metrics)
    mono_df = pd.DataFrame(monotonicity)
    target_rows: list[dict[str, Any]] = []
    for target, group in df[df["status"] == "OK"].groupby("target_name", dropna=False):
        loso = group[group["scenario"] == "leave_symbol_out"]
        positive_symbols = int((pd.to_numeric(loso["brier_skill_vs_prevalence"], errors="coerce") > 0).sum())
        combined = group[group["scenario"] == "combined_symbol_time_exclusion"]
        brier = pd.to_numeric(group["brier_skill_vs_prevalence"], errors="coerce")
        ece = pd.to_numeric(group["expected_calibration_error"], errors="coerce")
        mce = pd.to_numeric(group["maximum_calibration_error"], errors="coerce")
        gates = {
            "worst_fold_brier_skill_positive": bool(brier.min() > 0),
            "positive_symbol_count_gate": bool(positive_symbols >= min_positive_symbols),
            "combined_worst_brier_skill_positive": bool(pd.to_numeric(combined["brier_skill_vs_prevalence"], errors="coerce").min() > 0),
            "calibration_ece_gate": bool(ece.max() <= max_ece),
            "calibration_mce_gate": bool(mce.max() <= max_mce),
        }
        target_rows.append(
            {
                "target_name": target,
                "threshold_vol_units": float(group["threshold_vol_units"].iloc[0]),
                "scenario_rows": int(len(group)),
                "median_brier_skill": float(brier.median()),
                "worst_brier_skill": float(brier.min()),
                "positive_symbols": positive_symbols,
                "worst_combined_brier_skill": float(pd.to_numeric(combined["brier_skill_vs_prevalence"], errors="coerce").min()),
                "max_expected_calibration_error": float(ece.max()),
                "max_calibration_error": float(mce.max()),
                **gates,
                "target_passes": all(gates.values()),
                "failure_reasons": ";".join(name for name, ok in gates.items() if not ok),
            }
        )
    mono_projected = pd.to_numeric(mono_df.get("projected_monotonic_violation_fraction", pd.Series(dtype=float)), errors="coerce")
    mono_raw = pd.to_numeric(mono_df.get("raw_monotonic_violation_fraction", pd.Series(dtype=float)), errors="coerce")
    all_targets_pass = all(row["target_passes"] for row in target_rows) and len(target_rows) == len(TARGETS)
    family = {
        "family_status": "PASS" if all_targets_pass and float(mono_projected.max()) <= 0.0 else "FAIL",
        "target_count": len(target_rows),
        "targets_passing": sum(1 for row in target_rows if row["target_passes"]),
        "raw_monotonic_violation_fraction_max": float(mono_raw.max()) if len(mono_raw) else math.nan,
        "projected_monotonic_violation_fraction_max": float(mono_projected.max()) if len(mono_projected) else math.nan,
        "monotonic_projection_method": "cumulative_nonincreasing_minimum_accumulate",
        "freeze_allowed": bool(all_targets_pass and float(mono_projected.max()) <= 0.0),
    }
    return target_rows, family


def final_train_artifact(by_symbol: dict[str, SymbolData], feature_cols: list[str], max_rows_per_symbol: int) -> dict[str, Any]:
    symbols = sorted(by_symbol)
    train_ys = {}
    train_x = None
    rolling_end = min(data.rolling_end_index for data in by_symbol.values())
    for target in TARGETS:
        x, y, _ = stack_xy(by_symbol, symbols, 0, rolling_end, feature_cols, target, max_rows_per_symbol)
        if train_x is None:
            train_x = x
        train_ys[target] = y
    assert train_x is not None
    return {"models": fit_family(train_x, train_ys), "feature_cols": feature_cols, "train_rows": int(len(train_x)), "target_names": TARGETS}


def write_freeze_packet(out_dir: Path, artifact: dict[str, Any], contract: dict[str, Any], family_summary: dict[str, Any]) -> dict[str, Any]:
    freeze_dir = out_dir / "frozen_downside_severity_family_challenger"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    model_path = freeze_dir / "downside_severity_family_models.pkl"
    model_path.write_bytes(pickle.dumps(artifact))
    packet = {
        **contract,
        "family_summary": family_summary,
        "model_path": str(model_path),
        "model_sha256": stable_hash({"model_bytes": model_path.read_bytes().hex()}),
        "freeze_status": "frozen_research_challenger",
        "board_member_role": "downside_severity_member",
    }
    packet["candidate_hash"] = stable_hash(packet)
    write_json(freeze_dir / "downside_severity_family_candidate_contract.json", packet)
    return packet


def text_report(path: Path, target_summary: list[dict[str, Any]], family: dict[str, Any], contract: dict[str, Any]) -> None:
    lines = [
        "RAWSEQ 1M DOWNSIDE-SEVERITY FAMILY CONFIRMATION",
        f"created_at={contract['created_at']}",
        f"family_status={family['family_status']}",
        f"freeze_allowed={family['freeze_allowed']}",
        f"raw_monotonic_violation_fraction_max={family['raw_monotonic_violation_fraction_max']}",
        f"projected_monotonic_violation_fraction_max={family['projected_monotonic_violation_fraction_max']}",
        "",
        "Threshold gates:",
    ]
    for row in target_summary:
        lines.append(
            f"- threshold={row['threshold_vol_units']} target={row['target_name']} pass={row['target_passes']} "
            f"median_brier_skill={row['median_brier_skill']:.6f} worst_brier_skill={row['worst_brier_skill']:.6f} "
            f"positive_symbols={row['positive_symbols']} max_ece={row['max_expected_calibration_error']:.6f} "
            f"max_mce={row['max_calibration_error']:.6f} reasons={row['failure_reasons']}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- This is a focused confirmation only, not a broad target tournament.",
            "- Frozen dashboard/downside models were not changed.",
            "- A frozen challenger packet is created only when the whole ordered family passes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    source_root = env_path("RAWSEQ_SEVERITY_SOURCE_PATH", DEFAULT_SOURCE_PATH)
    output_root = env_path("RAWSEQ_SEVERITY_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    symbols = [s.strip().upper() for s in os.getenv("RAWSEQ_SEVERITY_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",") if s.strip()]
    max_rows = int(os.getenv("RAWSEQ_SEVERITY_MAX_ROWS_PER_SYMBOL", "50000"))
    train_cap = int(os.getenv("RAWSEQ_SEVERITY_MAX_TRAIN_ROWS_PER_SYMBOL", "20000"))
    min_rows = int(os.getenv("RAWSEQ_SEVERITY_MIN_ROWS", "500"))
    min_positive_symbols = int(os.getenv("RAWSEQ_SEVERITY_MIN_POSITIVE_SYMBOLS", "7"))
    max_ece = float(os.getenv("RAWSEQ_SEVERITY_MAX_ECE", "0.08"))
    max_mce = float(os.getenv("RAWSEQ_SEVERITY_MAX_MCE", "0.25"))
    feature_windows = [60, 240]
    out_dir = output_root / f"rawseq_1m_downside_severity_family_confirmation_{now_stamp()}"
    by_symbol = {symbol: read_symbol(symbol, source_root, max_rows, feature_windows) for symbol in symbols}
    feature_cols = existing_feature_columns(next(iter(by_symbol.values())).features)
    metrics: list[dict[str, Any]] = []
    mono_rows: list[dict[str, Any]] = []
    for scenario in scenario_defs(by_symbol):
        rows, mono, _ = evaluate_scenario(scenario, by_symbol, feature_cols, train_cap, min_rows)
        metrics.extend(rows)
        mono_rows.append(mono)
    target_summary, family = summarize(metrics, mono_rows, min_positive_symbols, max_ece, max_mce)
    contract = {
        "created_at": now_stamp(),
        "source_root": str(source_root),
        "development_cutoff_iso": "2026-05-31T23:59:00Z",
        "symbols": symbols,
        "thresholds_vol_units": THRESHOLDS,
        "target_names": TARGETS,
        "horizon_minutes": 1,
        "volatility_window_minutes": 240,
        "feature_group": "existing",
        "feature_windows": feature_windows,
        "model_family": "regularized_logistic_independent_thresholds_monotonic_projected",
        "scenario_types": ["leave_symbol_out", "leave_time_block_out", "combined_symbol_time_exclusion"],
        "min_positive_symbols": min_positive_symbols,
        "max_expected_calibration_error": max_ece,
        "max_calibration_error": max_mce,
        "max_rows_per_symbol": max_rows,
        "max_train_rows_per_symbol": train_cap,
        "holdout_used_for_selection": False,
        "dashboard_mutated": False,
        "frozen_downside_model_mutated": False,
        **SAFETY_FLAGS,
    }
    contract["contract_hash"] = stable_hash(contract)
    write_csv(out_dir / "downside_severity_family_confirmation_metrics.csv", metrics)
    write_csv(out_dir / "downside_severity_family_target_gates.csv", target_summary)
    write_csv(out_dir / "downside_severity_family_monotonicity.csv", mono_rows)
    write_json(out_dir / "downside_severity_family_confirmation_contract.json", contract)
    write_json(out_dir / "downside_severity_family_confirmation_summary.json", family)
    if family["freeze_allowed"]:
        packet = write_freeze_packet(out_dir, final_train_artifact(by_symbol, feature_cols, train_cap), contract, family)
        family["candidate_hash"] = packet["candidate_hash"]
        write_json(out_dir / "downside_severity_family_confirmation_summary.json", family)
    text_report(out_dir / "downside_severity_family_confirmation_report.txt", target_summary, family, contract)
    print(f"output_dir={out_dir}")
    print(f"family_status={family['family_status']}")
    print(f"freeze_allowed={family['freeze_allowed']}")
    print(f"targets_passing={family['targets_passing']}/{family['target_count']}")
    print(f"raw_monotonic_violation_fraction_max={family['raw_monotonic_violation_fraction_max']}")
    print(f"projected_monotonic_violation_fraction_max={family['projected_monotonic_violation_fraction_max']}")
    if "candidate_hash" in family:
        print(f"candidate_hash={family['candidate_hash']}")
    print(f"runtime_seconds={time.perf_counter() - start:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
