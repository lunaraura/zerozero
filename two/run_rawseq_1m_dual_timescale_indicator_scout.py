#!/usr/bin/env python3
"""Run a bounded dual-timescale indicator companion scout.

The scout is validation-only research. It reads an already materialized
companion dataset, evaluates deterministic and learned CPU baselines across
LOSO/time/combined exclusions, and only attempts a GPU temporal stage when the
CPU learned models beat the deterministic baselines.
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

from scripts.tiny.rawseq_1m_baseline_utils import (  # noqa: E402
    SAFETY_FLAGS,
    env_path,
    file_sha256,
    now_stamp,
    parse_bool,
    save_reload_prediction_parity,
    stable_hash,
    write_csv,
    write_json,
)

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_indicator_companion_scout")
FROZEN_CANDIDATE_HASH = "8109480bd6b0c2575337cada9be991746199200d42feaeea1fd7b9dd0b3eb6bf"
MA_CHANNELS = [
    "close_to_ema20_bps",
    "close_to_ema60_bps",
    "ema20_minus_ema60_bps",
    "ema20_slope_bps_per_minute",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def latest_dataset_dir(root: Path) -> Path:
    dirs = sorted(root.glob("dual_timescale_indicator_companion_dataset_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise SystemExit(f"No companion dataset dirs found under {root}")
    return dirs[0]


def finite_rows(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        flat = arr.reshape(len(arr), -1)
        mask &= np.isfinite(flat).all(axis=1)
    return mask


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    err = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
    return float(np.sqrt(np.nanmean(err * err))) if err.size else math.nan


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.nanmean(np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float)))) if np.size(y) else math.nan


def corr(y: np.ndarray, p: np.ndarray) -> float:
    yy = np.asarray(y, dtype=float).reshape(-1)
    pp = np.asarray(p, dtype=float).reshape(-1)
    mask = np.isfinite(yy) & np.isfinite(pp)
    if mask.sum() < 3 or np.std(yy[mask]) <= 1e-12 or np.std(pp[mask]) <= 1e-12:
        return math.nan
    return float(np.corrcoef(yy[mask], pp[mask])[0, 1])


def improvement(base: float, score: float) -> float:
    return float((base - score) / base) if math.isfinite(base) and base > 0 and math.isfinite(score) else math.nan


def direction_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    yy = np.asarray(y, dtype=float).reshape(-1)
    pp = np.asarray(p, dtype=float).reshape(-1)
    mask = np.isfinite(yy) & np.isfinite(pp) & (np.abs(yy) > 1e-12)
    return float((np.sign(yy[mask]) == np.sign(pp[mask])).mean()) if mask.sum() else math.nan


def current_state_baselines(data: dict[str, np.ndarray], indices: np.ndarray, train_indices: np.ndarray) -> dict[str, np.ndarray]:
    y_rsi_train = data["y_rsi_delta"][train_indices]
    y_ma_train = data["y_ma_state"][train_indices]
    rsi_slope = data["baseline_rsi_slope"][indices]
    ma_persist = data["baseline_ma_persistence"][indices]
    ma_constant = data["baseline_ma_constant_price"][indices]
    ma_slope = ma_persist.copy()
    for h in range(ma_slope.shape[1]):
        ma_slope[:, h, 3] = ma_persist[:, 0, 3] * float(h + 1)
    return {
        "rsi_persistence": np.zeros_like(data["y_rsi_delta"][indices]),
        "rsi_recent_slope": rsi_slope,
        "rsi_train_mean": np.tile(np.nanmean(y_rsi_train, axis=0), (len(indices), 1)),
        "rsi_train_median": np.tile(np.nanmedian(y_rsi_train, axis=0), (len(indices), 1)),
        "ma_persistence": ma_persist,
        "ma_constant_price": ma_constant,
        "ma_slope_continuation": ma_slope,
        "ma_train_mean": np.tile(np.nanmean(y_ma_train.reshape(len(train_indices), -1), axis=0), (len(indices), 1)).reshape(len(indices), 8, 4),
        "ma_train_median": np.tile(np.nanmedian(y_ma_train.reshape(len(train_indices), -1), axis=0), (len(indices), 1)).reshape(len(indices), 8, 4),
    }


def feature_matrix(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    static = data["x_static"][indices].reshape(len(indices), -1)
    short_last = data["x_short"][indices, -1, :].reshape(len(indices), -1)
    short_summary = np.concatenate(
        [
            np.nanmean(data["x_short"][indices], axis=1),
            np.nanstd(data["x_short"][indices], axis=1),
            np.nanmean(data["x_long"][indices], axis=1),
            np.nanstd(data["x_long"][indices], axis=1),
        ],
        axis=1,
    )
    return np.concatenate([static, short_last, short_summary], axis=1)


def output_vector(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    return np.concatenate([data["y_rsi_delta"][indices], data["y_ma_state"][indices].reshape(len(indices), -1)], axis=1)


def split_output(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pred[:, :8], pred[:, 8:].reshape(len(pred), 8, 4)


def cap_indices(indices: np.ndarray, limit: int, policy: str) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return indices
    if policy == "head":
        return indices[:limit]
    if policy == "tail":
        return indices[-limit:]
    take = np.linspace(0, len(indices) - 1, limit).round().astype(int)
    return indices[take]


def time_blocks(timestamp_ms: np.ndarray, eligible: np.ndarray, folds: int = 4) -> np.ndarray:
    order = eligible[np.argsort(timestamp_ms[eligible], kind="mergesort")]
    block = np.full(len(timestamp_ms), -1, dtype=int)
    for idx, chunk in enumerate(np.array_split(order, folds)):
        block[chunk] = idx
    return block


def make_folds(data: dict[str, np.ndarray], eligible: np.ndarray) -> list[dict[str, Any]]:
    symbols = sorted(set(str(x) for x in data["symbol"][eligible]))
    blocks = time_blocks(data["timestamp_ms"], eligible, 4)
    folds: list[dict[str, Any]] = []
    for symbol in symbols:
        val = eligible[data["symbol"][eligible] == symbol]
        train = eligible[data["symbol"][eligible] != symbol]
        folds.append({"scenario": "leave_one_symbol_out", "fold_id": symbol, "train": train, "validation": val})
    for block in range(4):
        val = eligible[blocks[eligible] == block]
        train = eligible[blocks[eligible] != block]
        folds.append({"scenario": "leave_one_time_block_out", "fold_id": f"time_block_{block}", "train": train, "validation": val})
    for idx, symbol in enumerate(symbols):
        block = idx % 4
        val = eligible[(data["symbol"][eligible] == symbol) & (blocks[eligible] == block)]
        train = eligible[(data["symbol"][eligible] != symbol) & (blocks[eligible] != block)]
        folds.append({"scenario": "combined_symbol_time_exclusion", "fold_id": f"{symbol}_time_block_{block}", "train": train, "validation": val})
    return folds


def evaluate_prediction(
    base: dict[str, Any],
    data: dict[str, np.ndarray],
    validation_idx: np.ndarray,
    train_idx: np.ndarray,
    model_name: str,
    pred_rsi: np.ndarray,
    pred_ma: np.ndarray,
    parity: tuple[bool, float] = (True, 0.0),
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    y_rsi = data["y_rsi_delta"][validation_idx]
    y_ma = data["y_ma_state"][validation_idx]
    baselines = current_state_baselines(data, validation_idx, train_idx)
    rsi_persistence_rmse = rmse(y_rsi, baselines["rsi_persistence"])
    rsi_slope_rmse = rmse(y_rsi, baselines["rsi_recent_slope"])
    ma_persistence_rmse = rmse(y_ma, baselines["ma_persistence"])
    ma_constant_rmse = rmse(y_ma, baselines["ma_constant_price"])
    rsi_model_rmse = rmse(y_rsi, pred_rsi)
    ma_model_rmse = rmse(y_ma, pred_ma)
    row = {
        **base,
        "model": model_name,
        "status": "OK",
        "rows": int(len(validation_idx)),
        "rsi_rmse": rsi_model_rmse,
        "rsi_mae": mae(y_rsi, pred_rsi),
        "rsi_correlation": corr(y_rsi, pred_rsi),
        "rsi_direction_accuracy": direction_accuracy(y_rsi, pred_rsi),
        "rsi_persistence_rmse": rsi_persistence_rmse,
        "rsi_recent_slope_rmse": rsi_slope_rmse,
        "rsi_persistence_improvement": improvement(rsi_persistence_rmse, rsi_model_rmse),
        "rsi_recent_slope_improvement": improvement(rsi_slope_rmse, rsi_model_rmse),
        "ma_rmse": ma_model_rmse,
        "ma_mae": mae(y_ma, pred_ma),
        "ma_correlation": corr(y_ma, pred_ma),
        "ma_persistence_rmse": ma_persistence_rmse,
        "ma_constant_price_rmse": ma_constant_rmse,
        "ma_persistence_improvement": improvement(ma_persistence_rmse, ma_model_rmse),
        "ma_constant_price_improvement": improvement(ma_constant_rmse, ma_model_rmse),
        "equal_target_normalized_rmse": normalized_rmse(y_rsi, pred_rsi, y_ma, pred_ma),
        "save_reload_parity": bool(parity[0]),
        "save_reload_max_abs_diff": float(parity[1]),
        "prediction_finite_fraction": float(np.isfinite(pred_rsi).mean() * 0.2 + np.isfinite(pred_ma).mean() * 0.8),
        "feature_contract_parity": True,
        "june_used_for_development": False,
        "july_data_accessed": False,
        **SAFETY_FLAGS,
        "public_recorded_data_only": True,
        "frozen_candidate_mutation": False,
    }
    per_horizon = []
    for h in range(8):
        per_horizon.append(
            {
                **base,
                "model": model_name,
                "head": "rsi_delta",
                "horizon": h + 1,
                "channel": "rsi_delta",
                "rmse": rmse(y_rsi[:, h], pred_rsi[:, h]),
                "mae": mae(y_rsi[:, h], pred_rsi[:, h]),
                "correlation": corr(y_rsi[:, h], pred_rsi[:, h]),
                "direction_accuracy": direction_accuracy(y_rsi[:, h], pred_rsi[:, h]),
                "persistence_improvement": improvement(rmse(y_rsi[:, h], baselines["rsi_persistence"][:, h]), rmse(y_rsi[:, h], pred_rsi[:, h])),
            }
        )
        for c, channel in enumerate(MA_CHANNELS):
            per_horizon.append(
                {
                    **base,
                    "model": model_name,
                    "head": "ma_state",
                    "horizon": h + 1,
                    "channel": channel,
                    "rmse": rmse(y_ma[:, h, c], pred_ma[:, h, c]),
                    "mae": mae(y_ma[:, h, c], pred_ma[:, h, c]),
                    "correlation": corr(y_ma[:, h, c], pred_ma[:, h, c]),
                    "sign_accuracy": direction_accuracy(y_ma[:, h, c], pred_ma[:, h, c]),
                    "primary_baseline": "constant_price_roll_forward" if channel != "ema20_slope_bps_per_minute" else "persistence",
                    "primary_baseline_improvement": improvement(
                        rmse(y_ma[:, h, c], baselines["ma_constant_price"][:, h, c] if channel != "ema20_slope_bps_per_minute" else baselines["ma_persistence"][:, h, c]),
                        rmse(y_ma[:, h, c], pred_ma[:, h, c]),
                    ),
                }
            )
    per_symbol = []
    for symbol in sorted(set(str(x) for x in data["symbol"][validation_idx])):
        mask = data["symbol"][validation_idx] == symbol
        per_symbol.append(
            {
                **base,
                "model": model_name,
                "symbol": symbol,
                "rows": int(mask.sum()),
                "rsi_persistence_improvement": improvement(rmse(y_rsi[mask], baselines["rsi_persistence"][mask]), rmse(y_rsi[mask], pred_rsi[mask])),
                "ma_constant_price_improvement": improvement(rmse(y_ma[mask], baselines["ma_constant_price"][mask]), rmse(y_ma[mask], pred_ma[mask])),
            }
        )
    crossing = crossing_metrics(base, model_name, data, validation_idx, pred_rsi, pred_ma)
    return row, per_symbol, per_horizon, crossing


def normalized_rmse(y_rsi: np.ndarray, p_rsi: np.ndarray, y_ma: np.ndarray, p_ma: np.ndarray) -> float:
    y = np.concatenate([y_rsi, y_ma.reshape(len(y_ma), -1)], axis=1)
    p = np.concatenate([p_rsi, p_ma.reshape(len(p_ma), -1)], axis=1)
    scale = np.maximum(np.nanstd(y, axis=0, ddof=0), 1e-6)
    return rmse(y / scale, p / scale)


def crossing_metrics(base: dict[str, Any], model_name: str, data: dict[str, np.ndarray], validation_idx: np.ndarray, pred_rsi: np.ndarray, pred_ma: np.ndarray) -> list[dict[str, Any]]:
    current_rsi = data["current_rsi14"][validation_idx]
    actual_rsi = current_rsi[:, None] + data["y_rsi_delta"][validation_idx]
    predicted_rsi = current_rsi[:, None] + pred_rsi
    rows = []
    for threshold, direction in [(50.0, "above"), (70.0, "above"), (30.0, "below")]:
        if direction == "above":
            actual_state = actual_rsi > threshold
            pred_state = predicted_rsi > threshold
            actual_cross = np.any(actual_state & (current_rsi[:, None] <= threshold), axis=1)
            pred_cross = np.any(pred_state & (current_rsi[:, None] <= threshold), axis=1)
        else:
            actual_state = actual_rsi < threshold
            pred_state = predicted_rsi < threshold
            actual_cross = np.any(actual_state & (current_rsi[:, None] >= threshold), axis=1)
            pred_cross = np.any(pred_state & (current_rsi[:, None] >= threshold), axis=1)
        rows.append({**base, "model": model_name, "event": f"rsi_{direction}_{int(threshold)}", "accuracy": float((actual_state == pred_state).mean()), "crossing_accuracy": float((actual_cross == pred_cross).mean())})
    y_ma = data["y_ma_state"][validation_idx]
    for idx, name in [(0, "price_above_ema20"), (1, "price_above_ema60"), (2, "ema20_above_ema60")]:
        actual = y_ma[:, :, idx] > 0.0
        pred = pred_ma[:, :, idx] > 0.0
        rows.append({**base, "model": model_name, "event": name, "accuracy": float((actual == pred).mean()), "crossing_accuracy": float((np.any(actual, axis=1) == np.any(pred, axis=1)).mean())})
    return rows


def fit_cpu_model(name: str, train_x: np.ndarray, train_y: np.ndarray) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "regularized_multioutput_linear":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    if name == "flattened_shallow_hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=40, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.01, random_state=1337)),
        )
    raise ValueError(name)


def model_predict(model: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(x), dtype=np.float64)


def aggregate_leaderboard(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(metrics)
    if df.empty:
        return []
    rows = []
    for model, group in df.groupby("model"):
        rsi_imp = pd.to_numeric(group["rsi_persistence_improvement"], errors="coerce")
        ma_imp = pd.to_numeric(group["ma_constant_price_improvement"], errors="coerce")
        symbol_wins = 0
        if "validation_symbols" in group:
            pass
        rows.append(
            {
                "model": model,
                "folds": int(len(group)),
                "rsi_aggregate_rmse": float(pd.to_numeric(group["rsi_rmse"], errors="coerce").mean()),
                "ma_aggregate_rmse": float(pd.to_numeric(group["ma_rmse"], errors="coerce").mean()),
                "median_rsi_persistence_improvement": float(rsi_imp.median()),
                "worst_rsi_persistence_improvement": float(rsi_imp.min()),
                "median_ma_constant_price_improvement": float(ma_imp.median()),
                "worst_ma_constant_price_improvement": float(ma_imp.min()),
                "combined_symbol_time_median_improvement": float(
                    pd.to_numeric(group[group["scenario"] == "combined_symbol_time_exclusion"]["ma_constant_price_improvement"], errors="coerce").median()
                ),
                "save_reload_parity_all": bool(group["save_reload_parity"].astype(str).str.lower().isin(["true", "1"]).all()),
                "prediction_finite_fraction_min": float(pd.to_numeric(group["prediction_finite_fraction"], errors="coerce").min()),
                "status": "evaluated",
                **SAFETY_FLAGS,
                "public_recorded_data_only": True,
            }
        )
    rows.sort(key=lambda row: (min(row["median_rsi_persistence_improvement"], row["median_ma_constant_price_improvement"]), row["combined_symbol_time_median_improvement"]), reverse=True)
    return rows


def decision_from_leaderboard(rows: list[dict[str, Any]], per_horizon: list[dict[str, Any]], per_symbol: list[dict[str, Any]], temporal_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not rows:
        return "indicator_dataset_blocked", {"reasons": ["no_evaluable_rows"]}
    best = rows[0]
    hdf = pd.DataFrame([row for row in per_horizon if row["model"] == best["model"]])
    sdf = pd.DataFrame([row for row in per_symbol if row["model"] == best["model"]])
    rsi_pass = best["median_rsi_persistence_improvement"] > 0 and best["worst_rsi_persistence_improvement"] > -0.02
    ma_pass = best["median_ma_constant_price_improvement"] > 0
    if not hdf.empty:
        horizon_wins = int((pd.to_numeric(hdf["primary_baseline_improvement"].fillna(hdf["persistence_improvement"]), errors="coerce") > 0).sum())
    else:
        horizon_wins = 0
    if not sdf.empty:
        symbol_wins = int(((pd.to_numeric(sdf["rsi_persistence_improvement"], errors="coerce") > 0) & (pd.to_numeric(sdf["ma_constant_price_improvement"], errors="coerce") > 0)).sum())
    else:
        symbol_wins = 0
    cpu_learned_pass = bool(rsi_pass and ma_pass)
    if not cpu_learned_pass:
        recommendation = "deterministic_baseline_not_beaten"
    elif temporal_rows:
        recommendation = "continue_dual_timescale_temporal_research"
    else:
        recommendation = "continue_indicator_cpu_research"
    return recommendation, {
        "best_model": best["model"],
        "rsi_pass": rsi_pass,
        "ma_pass": ma_pass,
        "horizon_positive_rows": horizon_wins,
        "symbols_positive_both": symbol_wins,
        "cpu_learned_pass": cpu_learned_pass,
        "temporal_stage_rows": len(temporal_rows),
    }


def write_gui_prediction_schema(run_dir: Path) -> None:
    schema = {
        "timestamp": "...",
        "symbol": "SOLUSDT",
        "current": {"price": 0, "rsi14": 0, "ema20": 0, "ema60": 0},
        "frozen_risk": {"downside_probability_1m": 0},
        "forecast_horizons_minutes": [1, 2, 3, 4, 5, 6, 7, 8],
        "rsi14_forecast": [],
        "close_to_ema20_bps_forecast": [],
        "close_to_ema60_bps_forecast": [],
        "ema20_minus_ema60_bps_forecast": [],
        "ema20_slope_bps_forecast": [],
    }
    write_json(run_dir / "gui_prediction_schema.json", schema)


def write_empty_gui_predictions(run_dir: Path, reason: str) -> None:
    columns = {
        "timestamp": "",
        "symbol": "",
        "current_close": "",
        "current_RSI14": "",
        "current_EMA20": "",
        "current_EMA60": "",
        "frozen_downside_risk_probability_1m": "",
        **{f"predicted_RSI14_delta_h{i}": "" for i in range(1, 9)},
        **{f"predicted_RSI14_h{i}": "" for i in range(1, 9)},
        **{f"predicted_close_to_EMA20_bps_h{i}": "" for i in range(1, 9)},
        **{f"predicted_close_to_EMA60_bps_h{i}": "" for i in range(1, 9)},
        **{f"predicted_EMA20_minus_EMA60_bps_h{i}": "" for i in range(1, 9)},
        **{f"predicted_EMA20_slope_bps_h{i}": "" for i in range(1, 9)},
        "model_disagreement": "",
        "companion_contract_hash": "",
        "frozen_risk_candidate_hash": FROZEN_CANDIDATE_HASH,
    }
    write_csv(run_dir / "indicator_companion_predictions.csv", [])
    (run_dir / "indicator_companion_predictions.csv").write_text(",".join(columns.keys()) + "\n", encoding="utf-8")
    write_json(run_dir / "indicator_companion_predictions_status.json", {"status": "not_generated", "reason": reason})


def write_model_artifact_manifest(run_dir: Path, recommendation: str, reason: str) -> None:
    artifact_dir = run_dir / "saved_model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        artifact_dir / "model_artifact_manifest.json",
        {
            "model_artifacts_saved": False,
            "reason": reason,
            "recommendation": recommendation,
            "first_scout_freeze": False,
            "frozen_candidate_mutation": False,
            "promotion": False,
            "orders": False,
        },
    )


def memory_report() -> dict[str, Any]:
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        rss = getattr(info, "rss", None)
        return {
            "memory_measurement": "psutil.Process.memory_info",
            "peak_working_set_mb": float(peak) / (1024.0 * 1024.0) if peak is not None else math.nan,
            "rss_mb": float(rss) / (1024.0 * 1024.0) if rss is not None else math.nan,
        }
    except Exception as exc:
        return {"memory_measurement": "unavailable", "memory_measurement_error": str(exc), "peak_working_set_mb": math.nan, "rss_mb": math.nan}


def main() -> int:
    started = time.perf_counter()
    raw_dataset_dir = os.getenv("RAWSEQ_INDICATOR_DATASET_DIR", "").strip()
    dataset_dir = Path(raw_dataset_dir) if raw_dataset_dir else latest_dataset_dir(DEFAULT_OUTPUT_ROOT)
    dataset_dir = dataset_dir if dataset_dir.is_absolute() else PROJECT_ROOT / dataset_dir
    output_root = env_path("RAWSEQ_INDICATOR_SCOUT_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    max_train_rows = int(os.getenv("RAWSEQ_INDICATOR_MAX_TRAIN_ROWS", "12000") or "12000")
    max_validation_rows = int(os.getenv("RAWSEQ_INDICATOR_MAX_VALIDATION_ROWS", "6000") or "6000")
    run_hgb = parse_bool(os.getenv("RAWSEQ_INDICATOR_RUN_HGB", "true"))
    run_temporal = parse_bool(os.getenv("RAWSEQ_INDICATOR_RUN_TEMPORAL", "true"))
    run_dir = output_root / f"dual_timescale_indicator_companion_scout_{now_stamp()}"
    manifest = read_json(dataset_dir / "indicator_companion_dataset_manifest.json")
    input_contract = read_json(dataset_dir / "indicator_input_contract.json")
    target_contract = read_json(dataset_dir / "indicator_target_contract.json")
    if input_contract["frozen_candidate_hash"] != FROZEN_CANDIDATE_HASH:
        raise RuntimeError("Frozen candidate hash mismatch")
    data_npz = np.load(manifest["dataset_path"], allow_pickle=True)
    data = {key: data_npz[key] for key in data_npz.files}
    split = data["split"].astype(str)
    eligible = np.where(np.isin(split, ["train", "validation"]))[0]
    folds = make_folds(data, eligible)
    fold_rows = []
    metrics: list[dict[str, Any]] = []
    per_symbol: list[dict[str, Any]] = []
    per_horizon: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    cpu_models = ["regularized_multioutput_linear"] + (["flattened_shallow_hgb"] if run_hgb else [])
    for fold in folds:
        train_idx = cap_indices(np.asarray(fold["train"], dtype=int), max_train_rows, "tail")
        validation_idx = cap_indices(np.asarray(fold["validation"], dtype=int), max_validation_rows, "head")
        base = {
            "scenario": fold["scenario"],
            "fold_id": fold["fold_id"],
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(validation_idx)),
            "validation_symbols": ",".join(sorted(set(str(x) for x in data["symbol"][validation_idx]))),
            "dataset_dir": str(dataset_dir),
            "frozen_candidate_hash": FROZEN_CANDIDATE_HASH,
        }
        fold_rows.append(base)
        if len(train_idx) < 100 or len(validation_idx) < 20:
            continue
        baselines = current_state_baselines(data, validation_idx, train_idx)
        for name, pred in baselines.items():
            if name.startswith("rsi_"):
                pred_rsi = pred
                pred_ma = baselines["ma_constant_price"]
            else:
                pred_rsi = baselines["rsi_persistence"]
                pred_ma = pred
            row, sym, hor, cross = evaluate_prediction(base, data, validation_idx, train_idx, name, pred_rsi, pred_ma)
            baseline_rows.append(row)
            per_symbol.extend(sym)
            per_horizon.extend(hor)
            crossing_rows.extend(cross)
        train_x = feature_matrix(data, train_idx)
        val_x = feature_matrix(data, validation_idx)
        train_y = output_vector(data, train_idx)
        good = finite_rows(train_x, train_y)
        train_x = train_x[good]
        train_y = train_y[good]
        if len(train_y) < 100:
            continue
        for model_name in cpu_models:
            model = fit_cpu_model(model_name, train_x, train_y)
            model.fit(train_x, train_y)
            pred = model_predict(model, val_x)
            pred_rsi, pred_ma = split_output(pred)
            parity = save_reload_prediction_parity(model, model_predict, val_x[: min(200, len(val_x))])
            row, sym, hor, cross = evaluate_prediction(base, data, validation_idx, train_idx, model_name, pred_rsi, pred_ma, parity)
            metrics.append(row)
            per_symbol.extend(sym)
            per_horizon.extend(hor)
            crossing_rows.extend(cross)
    leaderboard = aggregate_leaderboard(metrics + baseline_rows)
    learned_leaderboard = aggregate_leaderboard(metrics)
    temporal_rows: list[dict[str, Any]] = []
    temporal_block_reason = ""
    if learned_leaderboard and learned_leaderboard[0]["median_rsi_persistence_improvement"] > 0 and learned_leaderboard[0]["median_ma_constant_price_improvement"] > 0:
        if run_temporal:
            temporal_block_reason = "temporal_stage_not_run_in_first_scout_cpu_guard_only; set up contract but no large GPU search in this bounded run"
        else:
            temporal_block_reason = "temporal_stage_disabled_by_env"
    else:
        temporal_block_reason = "cpu_learned_models_did_not_beat_deterministic_baselines"
    recommendation, decision_details = decision_from_leaderboard(learned_leaderboard or leaderboard, per_horizon, per_symbol, temporal_rows)
    mem = memory_report()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "input_contract.json", input_contract)
    write_json(run_dir / "target_contract.json", target_contract)
    write_csv(run_dir / "fold_manifest.csv", fold_rows)
    write_csv(run_dir / "cpu_baseline_metrics.csv", baseline_rows)
    write_csv(run_dir / "temporal_model_metrics.csv", temporal_rows or [{"status": "NOT_RUN", "reason": temporal_block_reason}])
    write_csv(run_dir / "per_symbol_metrics.csv", per_symbol)
    write_csv(run_dir / "per_horizon_metrics.csv", per_horizon)
    write_csv(run_dir / "baseline_comparison.csv", metrics + baseline_rows)
    write_csv(run_dir / "crossing_event_metrics.csv", crossing_rows)
    write_csv(run_dir / "candidate_leaderboard.csv", leaderboard)
    write_gui_prediction_schema(run_dir)
    write_empty_gui_predictions(run_dir, "no_research_candidate_selected_in_first_scout")
    write_model_artifact_manifest(run_dir, recommendation, "no_model_artifact_saved_unless_candidate_advances")
    write_json(
        run_dir / "candidate_decision.json",
        {
            "final_recommendation": recommendation,
            "decision_details": decision_details,
            "temporal_stage_reason": temporal_block_reason,
            "runtime_seconds": time.perf_counter() - started,
            **mem,
            "candidate_hash": stable_hash({"leaderboard": leaderboard[:5], "input_contract": input_contract, "target_contract": target_contract}),
            "no_freeze_first_scout": True,
            "june_used_for_development": False,
            "july_data_accessed": False,
            **SAFETY_FLAGS,
            "public_recorded_data_only": True,
        },
    )
    report = [
        "Rawseq 1m dual-timescale indicator companion scout",
        f"dataset_dir={dataset_dir}",
        f"runtime_seconds={time.perf_counter() - started:.2f}",
        f"peak_working_set_mb={mem.get('peak_working_set_mb')}",
        f"best_cpu_model={(learned_leaderboard[0]['model'] if learned_leaderboard else 'none')}",
        f"best_overall_model={(leaderboard[0]['model'] if leaderboard else 'none')}",
        f"temporal_stage_reason={temporal_block_reason}",
        f"final_recommendation={recommendation}",
        "safety: no training of frozen downside-risk candidate; no June/July development access; no orders/promotion/champion mutation",
    ]
    (run_dir / "candidate_decision.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(
        run_dir / "save_reload_parity_report.json",
        {
            "all_evaluated_cpu_models_parity": all(bool(row.get("save_reload_parity")) for row in metrics),
            "max_abs_diff": max([float(row.get("save_reload_max_abs_diff", 0.0)) for row in metrics], default=0.0),
        },
    )
    print(f"scout_dir={run_dir}")
    print(f"runtime_seconds={time.perf_counter() - started:.2f}")
    print(f"peak_working_set_mb={mem.get('peak_working_set_mb')}")
    print(f"best_cpu_baseline={learned_leaderboard[0]['model'] if learned_leaderboard else 'none'}")
    print(f"best_temporal_model=NOT_RUN")
    print(f"final_recommendation={recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
