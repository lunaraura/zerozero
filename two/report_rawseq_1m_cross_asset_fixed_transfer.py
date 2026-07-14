#!/usr/bin/env python3
"""Evaluate the fixed SOL one-minute downside-risk contract across symbols.

This is a fixed-contract transfer test. It rebuilds the frozen SOL feature and
target contract for each eligible Binance public symbol, fits model weights
separately per symbol, and applies the same development-fold metrics. It does
not tune features, horizons, volatility windows, hyperparameters, calibration,
or split rules per symbol.
"""

from __future__ import annotations

import csv
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
    DEFAULT_SOURCE_PATH,
    SAFETY_FLAGS,
    audit_candles,
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

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_cross_asset_scout")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def latest_dir(root: Path, pattern: str) -> Path:
    dirs = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No {pattern} directories found under {root}")
    return dirs[0]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def cap_indices(indices: np.ndarray, max_rows: int, mode: str) -> np.ndarray:
    if max_rows <= 0 or len(indices) <= max_rows:
        return indices
    if mode == "tail":
        return indices[-max_rows:]
    return indices[:max_rows]


def finite_xy(
    features: pd.DataFrame,
    target: pd.Series,
    indices: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = features.iloc[indices][feature_cols].to_numpy(dtype=np.float64)
    y = pd.to_numeric(target.iloc[indices], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[mask], y[mask], indices[mask]


def make_hgb_model(contract: dict[str, Any]) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    hgb_step = next(step for step in contract["model_pipeline"] if step.get("step") == "HistGradientBoostingClassifier")
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=int(hgb_step["max_iter"]),
            max_leaf_nodes=int(hgb_step["max_leaf_nodes"]),
            learning_rate=float(hgb_step["learning_rate"]),
            l2_regularization=float(hgb_step["l2_regularization"]),
            random_state=int(hgb_step["random_state"]),
        ),
    )


def fit_predict_hgb(contract: dict[str, Any], train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, Any]:
    if len(np.unique(train_y)) < 2:
        prob = float(np.mean(train_y)) if len(train_y) else 0.5
        return np.full(len(val_x), prob), {"constant_probability": prob}
    model = make_hgb_model(contract)
    model.fit(train_x, train_y.astype(int))
    return model.predict_proba(val_x)[:, 1], model


def predict_model(model: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(len(x), float(model.get("constant_probability", 0.5)))
    return model.predict_proba(x)[:, 1]


def save_reload_prediction_parity(model: Any, x: np.ndarray) -> tuple[bool, float]:
    loaded = pickle.loads(pickle.dumps(model))
    a = np.asarray(predict_model(model, x), dtype=float)
    b = np.asarray(predict_model(loaded, x), dtype=float)
    diff = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
    return diff <= 1e-12, diff


def timestamp_range(features: pd.DataFrame, indices: np.ndarray) -> tuple[str, str]:
    if len(indices) == 0:
        return "", ""
    ts = pd.to_datetime(features.iloc[indices]["timestamp_ms"], unit="ms", utc=True, errors="coerce")
    return ts.min().isoformat(), ts.max().isoformat()


def feature_drift(train_x: np.ndarray, val_x: np.ndarray) -> float:
    if train_x.size == 0 or val_x.size == 0:
        return math.nan
    train_mean = np.nanmean(train_x, axis=0)
    val_mean = np.nanmean(val_x, axis=0)
    train_std = np.nanstd(train_x, axis=0)
    denom = np.where(train_std > 1e-12, train_std, np.nan)
    drift = np.abs(val_mean - train_mean) / denom
    return float(np.nanmedian(drift)) if np.isfinite(drift).any() else math.nan


def classify_symbol(rows: list[dict[str, Any]], rules: dict[str, Any]) -> str:
    ok = [row for row in rows if row.get("status") == "OK"]
    if not ok:
        return "no_transfer"
    skills = [safe_float(row.get("brier_skill_vs_prevalence")) for row in ok]
    pr_lifts = [safe_float(row.get("pr_auc_lift_over_event_prevalence")) for row in ok]
    fold_wins = sum(1 for value in skills if value > 0)
    median_skill = float(np.nanmedian(skills)) if skills else math.nan
    worst_skill = float(np.nanmin(skills)) if skills else math.nan
    median_pr = float(np.nanmedian(pr_lifts)) if pr_lifts else math.nan
    parity = all(truthy(row.get("save_reload_prediction_parity")) for row in ok)
    strong = rules["classification_rules"]["strong_transfer"]
    if (
        fold_wins >= int(strong["fold_wins_min"])
        and median_skill > float(strong["median_brier_skill_gt"])
        and worst_skill > float(strong["worst_fold_brier_skill_gt"])
        and median_pr > float(strong["median_pr_auc_lift_gt"])
        and (parity or not strong["save_reload_parity_required"])
    ):
        return "strong_transfer"
    if median_skill > 0:
        return "partial_transfer"
    return "no_transfer"


def evaluate_symbol(
    symbol: str,
    source_root: Path,
    contract: dict[str, Any],
    rules: dict[str, Any],
    max_rows: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_files = resolve_source_files(source_root, symbol)
    frame = load_candles(source_files, max_rows=max_rows)
    audit = audit_candles(frame, source_files, symbol=symbol, venue="binance_public")
    feature_windows = [int(x) for x in contract["feature_windows_minutes"]]
    features, _, leakage = build_features(frame, feature_windows)
    horizon = int(contract["target_horizon_minutes"])
    vol_window = int(contract["volatility_window_minutes"])
    targets = downside_event_targets(frame, vol_window=vol_window, horizons=[horizon])
    vol_col = str(contract["volatility_denominator"])
    features[vol_col] = targets[vol_col]
    target_col = f"downside_event_0p5vol_h{horizon}m_fw{vol_window}"
    target = targets[target_col]
    split, folds, purge_rows = split_contract(
        frame,
        feature_lookback=int(contract["purge_rows"]),
        max_horizon=int(contract["embargo_rows"]),
        fold_count=int(rules["classification_rules"]["strong_transfer"]["folds_expected"]),
    )
    feature_cols = list(contract["model_feature_names_and_order"])
    if feature_cols != list(contract["model_feature_names_and_order"]):
        raise RuntimeError("Feature order unexpectedly changed")
    missing = [col for col in feature_cols if col not in features.columns]
    if missing:
        raise RuntimeError(f"{symbol} missing fixed contract features: {missing}")

    rows: list[dict[str, Any]] = []
    train_cap = int(contract["training_row_cap_per_fold"])
    val_cap = int(contract["validation_row_cap_per_fold"])
    for fold in folds:
        train_idx_all = np.arange(fold["train_start_index"], fold["train_end_index"] + 1, dtype=np.int64)
        val_idx_all = np.arange(fold["validation_start_index"], fold["validation_end_index"] + 1, dtype=np.int64)
        train_idx = cap_indices(train_idx_all, train_cap, contract["cap_policy"]["train"])
        val_idx = cap_indices(val_idx_all, val_cap, contract["cap_policy"]["validation"])
        train_x, train_y, train_source_idx = finite_xy(features, target, train_idx, feature_cols)
        val_x, val_y, val_source_idx = finite_xy(features, target, val_idx, feature_cols)
        train_start, train_end = timestamp_range(features, train_source_idx)
        val_start, val_end = timestamp_range(features, val_source_idx)
        base = {
            "symbol": symbol,
            "fold_id": int(fold["fold_id"]),
            "target_horizon_minutes": horizon,
            "vol_window_minutes": vol_window,
            "model": contract["model_family"],
            "train_start": train_start,
            "train_end": train_end,
            "validation_start": val_start,
            "validation_end": val_end,
            "train_rows": int(len(train_y)),
            "validation_rows": int(len(val_y)),
            "training_prevalence": float(np.mean(train_y)) if len(train_y) else math.nan,
            "fixed_transfer_contract_hash": contract["fixed_transfer_contract_hash"],
            "feature_order_sha256": stable_hash(feature_cols),
            "no_per_symbol_tuning": True,
            "holdout_accessed": False,
            "gpu_used": False,
            **SAFETY_FLAGS,
            "public_recorded_data_only": True,
            "frozen_kraken_candidate_reused": False,
            "future_data_accessed": False,
        }
        if len(train_y) < 100 or len(val_y) < 100 or len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
            rows.append({**base, "status": "DATA_FAILED", "failure_reason": "insufficient rows or class diversity"})
            continue
        baseline = np.full(len(val_y), np.clip(float(np.mean(train_y)), 1e-6, 1 - 1e-6))
        pred, model = fit_predict_hgb(contract, train_x, train_y, val_x)
        parity, max_diff = save_reload_prediction_parity(model, val_x)
        metrics = metric_row(val_y, pred, baseline)
        rows.append(
            {
                **base,
                "status": "OK",
                "save_reload_prediction_parity": parity,
                "save_reload_prediction_max_abs_diff": max_diff,
                "feature_drift_median_abs_z": feature_drift(train_x, val_x),
                "target_prevalence_drift": metrics["event_prevalence"] - float(np.mean(train_y)),
                **metrics,
            }
        )

    audit_row = {
        "symbol": symbol,
        "source_rows": audit["total_rows"],
        "folds_built": len(folds),
        "purge_rows": split.get("purge_rows"),
        "embargo_rows": split.get("embargo_rows"),
        "purge_embargo_all_pass": all(row["purge_embargo_status"] == "PASS" for row in purge_rows),
        "leakage_audit_status": leakage.get("leakage_audit_status"),
        "classification": classify_symbol(rows, rules),
    }
    return rows, audit_row


def aggregate_symbol(rows: list[dict[str, Any]], audit_row: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "OK"]
    skills = [safe_float(row.get("brier_skill_vs_prevalence")) for row in ok]
    pr_lifts = [safe_float(row.get("pr_auc_lift_over_event_prevalence")) for row in ok]
    log_improvements = [safe_float(row.get("log_loss_improvement_vs_prevalence")) for row in ok]
    eces = [safe_float(row.get("expected_calibration_error")) for row in ok]
    drift = [safe_float(row.get("feature_drift_median_abs_z")) for row in ok]
    target_drift = [abs(safe_float(row.get("target_prevalence_drift"))) for row in ok]
    fold_wins = sum(1 for value in skills if value > 0)
    return {
        "symbol": audit_row["symbol"],
        "classification": classify_symbol(rows, rules),
        "folds": len(ok),
        "fold_wins": fold_wins,
        "fold_win_fraction": fold_wins / len(ok) if ok else math.nan,
        "median_brier_skill": float(np.nanmedian(skills)) if skills else math.nan,
        "worst_fold_brier_skill": float(np.nanmin(skills)) if skills else math.nan,
        "median_pr_auc_lift": float(np.nanmedian(pr_lifts)) if pr_lifts else math.nan,
        "median_log_loss_improvement": float(np.nanmedian(log_improvements)) if log_improvements else math.nan,
        "calibration_stability_ece_median": float(np.nanmedian(eces)) if eces else math.nan,
        "feature_drift_median_abs_z": float(np.nanmedian(drift)) if drift else math.nan,
        "target_prevalence_drift_abs_median": float(np.nanmedian(target_drift)) if target_drift else math.nan,
        "advance_gate_pass": audit_row["classification"] == "strong_transfer",
        "save_reload_parity_all_folds": all(truthy(row.get("save_reload_prediction_parity")) for row in ok) if ok else False,
        "fixed_transfer_contract_hash": rows[0].get("fixed_transfer_contract_hash") if rows else "",
        "no_per_symbol_tuning": True,
        "holdout_accessed": False,
        "gpu_used": False,
        **SAFETY_FLAGS,
    }


def load_eligible_symbols(inventory_dir: Path) -> list[str]:
    rows = read_csv_rows(inventory_dir / "multisymbol_inventory.csv")
    return [row["symbol"] for row in rows if truthy(row.get("eligible_for_fixed_transfer_test"))]


def main() -> int:
    started = time.perf_counter()
    output_root = env_path("RAWSEQ_1M_CROSS_ASSET_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    source_root = env_path("RAWSEQ_1M_SOURCE_PATH", DEFAULT_SOURCE_PATH)
    contract_dir = Path(os.getenv("RAWSEQ_1M_TRANSFER_CONTRACT_DIR", "").strip() or latest_dir(output_root, "rawseq_1m_fixed_transfer_contract_*"))
    inventory_dir = Path(os.getenv("RAWSEQ_1M_MULTISYMBOL_INVENTORY_DIR", "").strip() or latest_dir(output_root, "rawseq_1m_multisymbol_inventory_*"))
    out_dir = Path(os.getenv("RAWSEQ_1M_FIXED_TRANSFER_OUTPUT_DIR", "").strip() or output_root / f"rawseq_1m_fixed_transfer_{now_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=False)
    contract = read_json(contract_dir / "fixed_transfer_contract.json")
    rules = read_json(contract_dir / "transfer_acceptance_rules.json")
    symbols = load_eligible_symbols(inventory_dir)
    raw_symbols = [x.strip().upper() for x in os.getenv("RAWSEQ_1M_FIXED_TRANSFER_SYMBOLS", "").split(",") if x.strip()]
    if raw_symbols:
        symbols = [symbol for symbol in symbols if symbol in set(raw_symbols)]
    max_symbols = int(os.getenv("RAWSEQ_1M_FIXED_TRANSFER_MAX_SYMBOLS", "0") or "0")
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    max_rows = int(os.getenv("RAWSEQ_1M_FIXED_TRANSFER_MAX_ROWS", "0") or "0")

    fold_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            rows, audit_row = evaluate_symbol(symbol, source_root, contract, rules, max_rows=max_rows)
            fold_rows.extend(rows)
            audit_rows.append(audit_row)
            print(f"{symbol}: {audit_row['classification']} folds={len([r for r in rows if r.get('status') == 'OK'])}")
        except Exception as exc:
            audit_rows.append({"symbol": symbol, "classification": "DATA_FAILED", "failure_reason": repr(exc)})
            print(f"{symbol}: DATA_FAILED {exc!r}")
    leaderboard = [aggregate_symbol([row for row in fold_rows if row.get("symbol") == audit["symbol"]], audit, rules) for audit in audit_rows if audit.get("classification") != "DATA_FAILED"]
    leaderboard.sort(
        key=lambda row: (
            {"strong_transfer": 0, "partial_transfer": 1, "no_transfer": 2}.get(str(row["classification"]), 3),
            -safe_float(row.get("median_brier_skill"), -999),
            -safe_float(row.get("worst_fold_brier_skill"), -999),
        )
    )
    runtime_seconds = time.perf_counter() - started
    peak_working_set_mb = math.nan
    try:
        import psutil  # type: ignore

        peak_working_set_mb = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        peak_working_set_mb = math.nan
    payload = {
        "created_at": now_stamp(),
        "contract_dir": str(contract_dir),
        "contract_hash": contract.get("fixed_transfer_contract_hash"),
        "inventory_dir": str(inventory_dir),
        "source_root": str(source_root),
        "symbols_evaluated": symbols,
        "symbol_count": len(symbols),
        "max_rows": max_rows,
        "runtime_seconds": runtime_seconds,
        "cpu_only": True,
        "peak_working_set_mb": peak_working_set_mb,
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
    payload["fixed_transfer_run_hash"] = stable_hash({"payload": payload, "leaderboard": leaderboard})
    write_csv(out_dir / "fixed_transfer_fold_metrics.csv", fold_rows)
    write_csv(out_dir / "fixed_transfer_symbol_leaderboard.csv", leaderboard)
    write_csv(out_dir / "fixed_transfer_symbol_audit.csv", audit_rows)
    write_json(out_dir / "fixed_transfer_run_manifest.json", payload)
    lines = [
        "Rawseq 1m fixed-contract cross-asset transfer",
        f"Output: {out_dir}",
        f"Contract hash: {payload['contract_hash']}",
        f"Symbols evaluated: {', '.join(symbols) if symbols else 'none'}",
        f"Runtime seconds: {runtime_seconds:.3f}",
        f"Peak working set MB: {peak_working_set_mb:.3f}" if math.isfinite(peak_working_set_mb) else "Peak working set MB: unavailable",
        "",
        "Symbol classifications:",
    ]
    for row in leaderboard:
        lines.append(
            f"- {row['symbol']}: {row['classification']} folds={row['folds']} wins={row['fold_wins']} "
            f"median_brier_skill={row['median_brier_skill']} worst={row['worst_fold_brier_skill']}"
        )
    lines.append("")
    lines.append("Safety: CPU/paper-only; no GPU, no private API, no orders, no promotion, no champion mutation.")
    (out_dir / "fixed_transfer_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if symbols else 1


if __name__ == "__main__":
    raise SystemExit(main())
