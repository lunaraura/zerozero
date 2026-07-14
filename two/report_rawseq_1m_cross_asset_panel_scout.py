#!/usr/bin/env python3
"""Pooled one-minute cross-asset panel scout.

This is a separate challenger lineage from the fixed per-symbol transfer test.
It trains pooled CPU baselines on public Binance 1m candles with no symbol ID in
the primary models, then validates leave-one-symbol-out, leave-one-time-block-
out, and combined symbol/time exclusions.
"""

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
    latest_dir,
    read_json,
    load_eligible_symbols,
    read_csv_rows,
    safe_float,
    truthy,
)

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_cross_asset_scout")


@dataclass
class SymbolData:
    symbol: str
    frame: pd.DataFrame
    target: pd.Series
    folds: list[dict[str, Any]]
    rolling_end_index: int


def cap_indices(indices: np.ndarray, max_rows: int, mode: str) -> np.ndarray:
    if max_rows <= 0 or len(indices) <= max_rows:
        return indices
    return indices[-max_rows:] if mode == "tail" else indices[:max_rows]


def fit_logistic(train_x: np.ndarray, train_y: np.ndarray, sample_weight: np.ndarray | None = None) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(train_y)) < 2:
        return {"constant_probability": float(np.mean(train_y)) if len(train_y) else 0.5}
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300, solver="lbfgs"))
    fit_kwargs = {"logisticregression__sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(train_x, train_y.astype(int), **fit_kwargs)
    return model


def fit_hgb(contract: dict[str, Any], train_x: np.ndarray, train_y: np.ndarray, sample_weight: np.ndarray | None = None) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    if len(np.unique(train_y)) < 2:
        return {"constant_probability": float(np.mean(train_y)) if len(train_y) else 0.5}
    hgb_step = next(step for step in contract["model_pipeline"] if step.get("step") == "HistGradientBoostingClassifier")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=int(hgb_step["max_iter"]),
            max_leaf_nodes=int(hgb_step["max_leaf_nodes"]),
            learning_rate=float(hgb_step["learning_rate"]),
            l2_regularization=float(hgb_step["l2_regularization"]),
            random_state=int(hgb_step["random_state"]),
        ),
    )
    fit_kwargs = {"histgradientboostingclassifier__sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(train_x, train_y.astype(int), **fit_kwargs)
    return model


def predict_model(model: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(len(x), float(model.get("constant_probability", 0.5)))
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def save_reload_parity(model: Any, x: np.ndarray) -> tuple[bool, float]:
    loaded = pickle.loads(pickle.dumps(model))
    a = predict_model(model, x)
    b = predict_model(loaded, x)
    diff = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
    return diff <= 1e-12, diff


def equal_symbol_weights(symbols: np.ndarray) -> np.ndarray:
    weights = np.ones(len(symbols), dtype=np.float64)
    unique = sorted(set(symbols.tolist()))
    for symbol in unique:
        mask = symbols == symbol
        if mask.any():
            weights[mask] = len(symbols) / (len(unique) * float(mask.sum()))
    return weights


def prepare_symbol(symbol: str, source_root: Path, contract: dict[str, Any], max_rows: int = 0) -> SymbolData:
    source_files = resolve_source_files(source_root, symbol)
    candles = load_candles(source_files, max_rows=max_rows)
    feature_windows = [int(x) for x in contract["feature_windows_minutes"]]
    features, _, _ = build_features(candles, feature_windows)
    horizon = int(contract["target_horizon_minutes"])
    vol_window = int(contract["volatility_window_minutes"])
    targets = downside_event_targets(candles, vol_window=vol_window, horizons=[horizon])
    features[str(contract["volatility_denominator"])] = targets[str(contract["volatility_denominator"])]
    target = targets[f"downside_event_0p5vol_h{horizon}m_fw{vol_window}"]
    split, folds, _ = split_contract(
        candles,
        feature_lookback=int(contract["purge_rows"]),
        max_horizon=int(contract["embargo_rows"]),
        fold_count=4,
    )
    return SymbolData(symbol=symbol, frame=features, target=target, folds=folds, rolling_end_index=int(split["rolling_development_end_index"]))


def finite_symbol_xy(data: SymbolData, indices: np.ndarray, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = data.frame.iloc[indices][feature_cols].to_numpy(dtype=np.float64)
    y = pd.to_numeric(data.target.iloc[indices], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[mask], y[mask], np.asarray([data.symbol] * int(mask.sum()), dtype=object)


def stack_data(
    by_symbol: dict[str, SymbolData],
    symbols: list[str],
    start_index: int,
    end_index: int,
    feature_cols: list[str],
    max_rows_per_symbol: int,
    cap_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    for symbol in symbols:
        indices = np.arange(start_index, min(end_index, len(by_symbol[symbol].frame) - 1) + 1, dtype=np.int64)
        indices = cap_indices(indices, max_rows_per_symbol, cap_mode)
        x, y, s = finite_symbol_xy(by_symbol[symbol], indices, feature_cols)
        if len(y):
            xs.append(x)
            ys.append(y)
            ss.append(s)
    if not ys:
        return np.empty((0, len(feature_cols))), np.empty(0), np.empty(0, dtype=object)
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ss)


def regime_feature_indices(feature_cols: list[str]) -> list[int]:
    cols = [idx for idx, col in enumerate(feature_cols) if any(tok in col for tok in ["volatility", "range", "ema_slope", "volume"])]
    return cols or list(range(len(feature_cols)))


def evaluate_predictions(
    base: dict[str, Any],
    val_y: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    parity: tuple[bool, float] = (True, 0.0),
) -> dict[str, Any]:
    return {
        **base,
        "status": "OK",
        "save_reload_prediction_parity": parity[0],
        "save_reload_prediction_max_abs_diff": parity[1],
        **metric_row(val_y, pred, baseline),
    }


def per_symbol_constant_prediction(train_y: np.ndarray, train_s: np.ndarray, val_s: np.ndarray) -> np.ndarray:
    global_prev = float(np.mean(train_y)) if len(train_y) else 0.5
    out = np.full(len(val_s), global_prev)
    for symbol in sorted(set(train_s.tolist())):
        mask_train = train_s == symbol
        mask_val = val_s == symbol
        if mask_train.any() and mask_val.any():
            out[mask_val] = float(np.mean(train_y[mask_train]))
    return np.clip(out, 1e-6, 1 - 1e-6)


def evaluate_scenario(
    scenario_name: str,
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
    train_x, train_y, train_s = stack_data(by_symbol, train_symbols, train_range[0], train_range[1], feature_cols, max_train_rows_per_symbol, "tail")
    val_x, val_y, val_s = stack_data(by_symbol, validation_symbols, validation_range[0], validation_range[1], feature_cols, max_validation_rows_per_symbol, "head")
    base = {
        "scenario": scenario_name,
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
        "symbol_identifier_used": False,
        "holdout_accessed": False,
        "gpu_used": False,
        **SAFETY_FLAGS,
        "public_recorded_data_only": True,
        "frozen_kraken_candidate_reused": False,
        "future_data_accessed": False,
    }
    if len(train_y) < 100 or len(val_y) < 100 or len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
        return [{**base, "model": "all", "weighting": "none", "status": "DATA_FAILED", "failure_reason": "insufficient rows or class diversity"}]
    global_baseline = np.full(len(val_y), np.clip(float(np.mean(train_y)), 1e-6, 1 - 1e-6))
    rows = [
        evaluate_predictions({**base, "model": "global_pooled_prevalence", "weighting": "row_weighted"}, val_y, global_baseline, global_baseline),
        evaluate_predictions({**base, "model": "constant_per_symbol_prevalence", "weighting": "row_weighted"}, val_y, per_symbol_constant_prediction(train_y, train_s, val_s), global_baseline),
    ]
    regime_idx = regime_feature_indices(feature_cols)
    model_predictions: dict[tuple[str, str], np.ndarray] = {}
    for weighting, weights in [("row_weighted", None), ("equal_symbol_weighted", equal_symbol_weights(train_s))]:
        for model_name, fit_fn, x_train, x_val in [
            ("pooled_logistic", fit_logistic, train_x, val_x),
            ("pooled_regime_feature_logistic", fit_logistic, train_x[:, regime_idx], val_x[:, regime_idx]),
            ("pooled_shallow_hgb", lambda x, y, w=None: fit_hgb(contract, x, y, w), train_x, val_x),
        ]:
            model = fit_fn(x_train, train_y, weights)
            pred = np.clip(predict_model(model, x_val), 1e-6, 1 - 1e-6)
            model_predictions[(model_name, weighting)] = pred
            rows.append(evaluate_predictions({**base, "model": model_name, "weighting": weighting}, val_y, pred, global_baseline, save_reload_parity(model, x_val)))
        ens_parts = [model_predictions[(name, weighting)] for name in ["pooled_logistic", "pooled_regime_feature_logistic", "pooled_shallow_hgb"] if (name, weighting) in model_predictions]
        if ens_parts:
            rows.append(
                evaluate_predictions(
                    {**base, "model": "conservative_probability_ensemble", "weighting": weighting},
                    val_y,
                    np.mean(ens_parts, axis=0),
                    global_baseline,
                )
            )
    if set(validation_symbols).issubset(set(train_symbols)):
        pred = np.full(len(val_y), np.nan)
        parity_ok = True
        max_diff = 0.0
        for symbol in sorted(set(validation_symbols)):
            tr_mask = train_s == symbol
            va_mask = val_s == symbol
            if not tr_mask.any() or not va_mask.any():
                continue
            model = fit_logistic(train_x[tr_mask], train_y[tr_mask])
            pred[va_mask] = predict_model(model, val_x[va_mask])
            parity, diff = save_reload_parity(model, val_x[va_mask])
            parity_ok = parity_ok and parity
            max_diff = max(max_diff, diff)
        mask = np.isfinite(pred)
        if mask.all():
            rows.append(evaluate_predictions({**base, "model": "per_symbol_logistic", "weighting": "per_symbol"}, val_y, np.clip(pred, 1e-6, 1 - 1e-6), global_baseline, (parity_ok, max_diff)))
    else:
        rows.append({**base, "model": "per_symbol_logistic", "weighting": "per_symbol", "status": "SKIPPED", "failure_reason": "validation symbol excluded from training"})
    return rows


def aggregate_panel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "OK":
            grouped.setdefault((str(row["scenario"]), str(row["model"]), str(row["weighting"])), []).append(row)
    out = []
    for (scenario, model, weighting), vals in grouped.items():
        skills = [safe_float(row.get("brier_skill_vs_prevalence")) for row in vals]
        pr_lifts = [safe_float(row.get("pr_auc_lift_over_event_prevalence")) for row in vals]
        wins = sum(1 for value in skills if value > 0)
        out.append(
            {
                "scenario": scenario,
                "model": model,
                "weighting": weighting,
                "folds": len(vals),
                "fold_wins": wins,
                "fold_win_fraction": wins / len(vals) if vals else math.nan,
                "median_brier_skill": float(np.nanmedian(skills)) if skills else math.nan,
                "worst_fold_brier_skill": float(np.nanmin(skills)) if skills else math.nan,
                "median_pr_auc_lift": float(np.nanmedian(pr_lifts)) if pr_lifts else math.nan,
                "save_reload_parity_all": all(truthy(row.get("save_reload_prediction_parity")) for row in vals),
                "symbol_identifier_used": any(truthy(row.get("symbol_identifier_used")) for row in vals),
                "holdout_accessed": False,
                "gpu_used": False,
                **SAFETY_FLAGS,
            }
        )
    out.sort(key=lambda r: (r["scenario"], -safe_float(r["median_brier_skill"], -999), -safe_float(r["worst_fold_brier_skill"], -999)))
    return out


def rows_per_symbol(by_symbol: dict[str, SymbolData]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": symbol,
            "finite_rows": int(pd.to_numeric(data.target, errors="coerce").notna().sum()),
            "source_rows": len(data.frame),
            "rolling_end_index": data.rolling_end_index,
        }
        for symbol, data in sorted(by_symbol.items())
    ]


def main() -> int:
    started = time.perf_counter()
    output_root = env_path("RAWSEQ_1M_CROSS_ASSET_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    source_root = env_path("RAWSEQ_1M_SOURCE_PATH", DEFAULT_SOURCE_PATH)
    contract_dir = Path(os.getenv("RAWSEQ_1M_TRANSFER_CONTRACT_DIR", "").strip() or latest_dir(output_root, "rawseq_1m_fixed_transfer_contract_*"))
    inventory_dir = Path(os.getenv("RAWSEQ_1M_MULTISYMBOL_INVENTORY_DIR", "").strip() or latest_dir(output_root, "rawseq_1m_multisymbol_inventory_*"))
    out_dir = Path(os.getenv("RAWSEQ_1M_PANEL_OUTPUT_DIR", "").strip() or output_root / f"rawseq_1m_panel_scout_{now_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=False)
    contract = read_json(contract_dir / "fixed_transfer_contract.json")
    symbols = load_eligible_symbols(inventory_dir)
    raw_symbols = [x.strip().upper() for x in os.getenv("RAWSEQ_1M_PANEL_SYMBOLS", "").split(",") if x.strip()]
    if raw_symbols:
        symbols = [symbol for symbol in symbols if symbol in set(raw_symbols)]
    max_symbols = int(os.getenv("RAWSEQ_1M_PANEL_MAX_SYMBOLS", "0") or "0")
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    max_source_rows = int(os.getenv("RAWSEQ_1M_PANEL_MAX_SOURCE_ROWS", "0") or "0")
    max_train_rows_per_symbol = int(os.getenv("RAWSEQ_1M_PANEL_MAX_TRAIN_ROWS_PER_SYMBOL", "10000") or "10000")
    max_validation_rows_per_symbol = int(os.getenv("RAWSEQ_1M_PANEL_MAX_VALIDATION_ROWS_PER_SYMBOL", "10000") or "10000")
    by_symbol = {}
    for symbol in symbols:
        print(f"[prepare] {symbol}", flush=True)
        by_symbol[symbol] = prepare_symbol(symbol, source_root, contract, max_source_rows)
    if not by_symbol:
        raise SystemExit("No eligible symbols for panel scout")
    common_rolling_end = min(data.rolling_end_index for data in by_symbol.values())
    reference_folds = next(iter(by_symbol.values())).folds
    latest_fold = max(reference_folds, key=lambda row: int(row["fold_id"]))
    rows: list[dict[str, Any]] = []
    for excluded in symbols:
        print(f"[scenario] leave_one_symbol_out excluded={excluded}", flush=True)
        train_symbols = [symbol for symbol in symbols if symbol != excluded]
        rows.extend(
            evaluate_scenario(
                "leave_one_symbol_out",
                excluded,
                train_symbols,
                [excluded],
                (0, common_rolling_end),
                (0, common_rolling_end),
                by_symbol,
                contract,
                max_train_rows_per_symbol,
                max_validation_rows_per_symbol,
            )
        )
    for fold in reference_folds:
        print(f"[scenario] leave_one_time_block_out fold={fold['fold_id']}", flush=True)
        rows.extend(
            evaluate_scenario(
                "leave_one_time_block_out",
                str(fold["fold_id"]),
                symbols,
                symbols,
                (int(fold["train_start_index"]), int(fold["train_end_index"])),
                (int(fold["validation_start_index"]), int(fold["validation_end_index"])),
                by_symbol,
                contract,
                max_train_rows_per_symbol,
                max_validation_rows_per_symbol,
            )
        )
    for excluded in symbols:
        print(f"[scenario] combined_symbol_time_holdout excluded={excluded}", flush=True)
        train_symbols = [symbol for symbol in symbols if symbol != excluded]
        rows.extend(
            evaluate_scenario(
                "combined_symbol_time_holdout",
                excluded,
                train_symbols,
                [excluded],
                (int(latest_fold["train_start_index"]), int(latest_fold["train_end_index"])),
                (int(latest_fold["validation_start_index"]), int(latest_fold["validation_end_index"])),
                by_symbol,
                contract,
                max_train_rows_per_symbol,
                max_validation_rows_per_symbol,
            )
        )
    leaderboard = aggregate_panel(rows)
    runtime_seconds = time.perf_counter() - started
    peak_working_set_mb = math.nan
    try:
        import psutil  # type: ignore

        peak_working_set_mb = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        pass
    manifest = {
        "created_at": now_stamp(),
        "contract_dir": str(contract_dir),
        "contract_hash": contract.get("fixed_transfer_contract_hash"),
        "inventory_dir": str(inventory_dir),
        "symbols": symbols,
        "common_rolling_end_index": common_rolling_end,
        "max_source_rows": max_source_rows,
        "max_train_rows_per_symbol": max_train_rows_per_symbol,
        "max_validation_rows_per_symbol": max_validation_rows_per_symbol,
        "primary_models_use_symbol_identifier": False,
        "row_weighted_and_equal_symbol_weighted_compared": True,
        "runtime_seconds": runtime_seconds,
        "peak_working_set_mb": peak_working_set_mb,
        "cpu_only": True,
        "gpu_used": False,
        "holdout_accessed": False,
        "safety": {
            **SAFETY_FLAGS,
            "public_recorded_data_only": True,
            "active_future_shadow_mutation": False,
            "active_future_shadow_labels_used": False,
            "frozen_kraken_candidate_reused": False,
            "future_data_accessed": False,
        },
    }
    manifest["panel_run_hash"] = stable_hash({"manifest": manifest, "leaderboard": leaderboard})
    write_csv(out_dir / "pooled_panel_fold_metrics.csv", rows)
    write_csv(out_dir / "pooled_panel_leaderboard.csv", leaderboard)
    write_csv(out_dir / "leave_one_symbol_out_metrics.csv", [row for row in rows if row.get("scenario") == "leave_one_symbol_out"])
    write_csv(out_dir / "leave_one_time_block_out_metrics.csv", [row for row in rows if row.get("scenario") == "leave_one_time_block_out"])
    write_csv(out_dir / "combined_symbol_time_holdout_metrics.csv", [row for row in rows if row.get("scenario") == "combined_symbol_time_holdout"])
    write_csv(out_dir / "panel_rows_per_symbol.csv", rows_per_symbol(by_symbol))
    write_json(out_dir / "panel_scout_manifest.json", manifest)
    lines = [
        "Rawseq 1m cross-asset panel scout",
        f"Output: {out_dir}",
        f"Symbols: {', '.join(symbols)}",
        f"Contract hash: {manifest['contract_hash']}",
        f"Runtime seconds: {runtime_seconds:.3f}",
        f"Peak working set MB: {peak_working_set_mb:.3f}" if math.isfinite(peak_working_set_mb) else "Peak working set MB: unavailable",
        "",
        "Top rows by scenario:",
    ]
    for scenario in ["leave_one_symbol_out", "leave_one_time_block_out", "combined_symbol_time_holdout"]:
        subset = [row for row in leaderboard if row["scenario"] == scenario][:5]
        lines.append(f"{scenario}:")
        for row in subset:
            lines.append(
                f"- {row['model']} {row['weighting']}: folds={row['folds']} wins={row['fold_wins']} "
                f"median_brier_skill={row['median_brier_skill']} worst={row['worst_fold_brier_skill']}"
            )
    lines.append("")
    lines.append("Safety: CPU/paper-only; no symbol-ID primary model, no private API, no orders, no promotion, no champion mutation.")
    (out_dir / "panel_scout_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
