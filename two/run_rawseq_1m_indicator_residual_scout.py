#!/usr/bin/env python3
"""CPU residual scout for the 1m indicator companion lane."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import SAFETY_FLAGS, file_sha256, now_stamp, parse_bool, save_reload_prediction_parity, stable_hash, write_csv, write_json  # noqa: E402
from scripts.tiny.run_rawseq_1m_dual_timescale_indicator_scout import (  # noqa: E402
    cap_indices,
    corr,
    direction_accuracy,
    feature_matrix,
    finite_rows,
    make_folds,
    mae,
    model_predict,
    rmse,
)

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_indicator_companion_scout")
FROZEN_CANDIDATE_HASH = "8109480bd6b0c2575337cada9be991746199200d42feaeea1fd7b9dd0b3eb6bf"
MA_CHANNELS = ["close_to_ema20_bps", "close_to_ema60_bps", "ema20_minus_ema60_bps", "ema20_slope_bps_per_minute"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def latest_residual_dataset(root: Path) -> Path:
    dirs = sorted(root.glob("indicator_residual_event_dataset_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise SystemExit(f"No residual/event datasets found under {root}")
    return dirs[0]


def improvement(base: float, score: float) -> float:
    return float((base - score) / base) if math.isfinite(base) and base > 0 and math.isfinite(score) else math.nan


def residual_y(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            data["rsi_residual_vs_persistence"][indices],
            data["ma_residual_vs_constant_price"][indices].reshape(len(indices), -1),
        ],
        axis=1,
    )


def split_residual(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pred[:, :8], pred[:, 8:].reshape(len(pred), 8, 4)


def fit_cpu_model(name: str, train_x: np.ndarray, train_y: np.ndarray) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "regularized_multioutput_linear":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    if name == "shallow_hgb_residual":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=40, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.01, random_state=1337)),
        )
    raise ValueError(name)


def evaluate_residual_prediction(
    base: dict[str, Any],
    data: dict[str, np.ndarray],
    validation_idx: np.ndarray,
    model: str,
    pred_residual: np.ndarray,
    parity: tuple[bool, float] = (True, 0.0),
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    actual_rsi = data["actual_rsi_path"][validation_idx]
    actual_ma = data["actual_ma_state_path"][validation_idx]
    rsi_base = data["rsi_persistence_baseline_path"][validation_idx]
    ma_base = data["ma_constant_price_baseline_path"][validation_idx]
    pred_rsi_resid, pred_ma_resid = split_residual(pred_residual)
    pred_rsi = rsi_base + pred_rsi_resid
    pred_ma = ma_base + pred_ma_resid
    det_rsi_rmse = rmse(actual_rsi, rsi_base)
    det_ma_rmse = rmse(actual_ma, ma_base)
    model_rsi_rmse = rmse(actual_rsi, pred_rsi)
    model_ma_rmse = rmse(actual_ma, pred_ma)
    zero_resid = np.zeros_like(pred_residual)
    y_resid = residual_y(data, validation_idx)
    row = {
        **base,
        "model": model,
        "rows": int(len(validation_idx)),
        "residual_rmse": rmse(y_resid, pred_residual),
        "residual_mae": mae(y_resid, pred_residual),
        "zero_residual_rmse": rmse(y_resid, zero_resid),
        "residual_improvement_over_zero": improvement(rmse(y_resid, zero_resid), rmse(y_resid, pred_residual)),
        "rsi_reconstructed_rmse": model_rsi_rmse,
        "rsi_deterministic_baseline_rmse": det_rsi_rmse,
        "rsi_reconstruction_improvement": improvement(det_rsi_rmse, model_rsi_rmse),
        "ma_reconstructed_rmse": model_ma_rmse,
        "ma_deterministic_baseline_rmse": det_ma_rmse,
        "ma_reconstruction_improvement": improvement(det_ma_rmse, model_ma_rmse),
        "correlation": corr(y_resid, pred_residual),
        "sign_accuracy": direction_accuracy(y_resid, pred_residual),
        "save_reload_parity": bool(parity[0]),
        "save_reload_max_abs_diff": float(parity[1]),
        "prediction_finite_fraction": float(np.isfinite(pred_residual).mean()),
        "june_development_access": False,
        "july_access": False,
        **SAFETY_FLAGS,
        "public_recorded_data_only": True,
        "frozen_candidate_mutation": False,
    }
    per_symbol = []
    for symbol in sorted(set(str(x) for x in data["symbol"][validation_idx])):
        mask = data["symbol"][validation_idx] == symbol
        per_symbol.append(
            {
                **base,
                "model": model,
                "symbol": symbol,
                "rows": int(mask.sum()),
                "rsi_reconstruction_improvement": improvement(rmse(actual_rsi[mask], rsi_base[mask]), rmse(actual_rsi[mask], pred_rsi[mask])),
                "ma_reconstruction_improvement": improvement(rmse(actual_ma[mask], ma_base[mask]), rmse(actual_ma[mask], pred_ma[mask])),
            }
        )
    per_horizon = []
    for h in range(8):
        per_horizon.append(
            {
                **base,
                "model": model,
                "target": "rsi_path",
                "horizon": h + 1,
                "rmse": rmse(actual_rsi[:, h], pred_rsi[:, h]),
                "deterministic_rmse": rmse(actual_rsi[:, h], rsi_base[:, h]),
                "improvement": improvement(rmse(actual_rsi[:, h], rsi_base[:, h]), rmse(actual_rsi[:, h], pred_rsi[:, h])),
            }
        )
        for c, channel in enumerate(MA_CHANNELS):
            per_horizon.append(
                {
                    **base,
                    "model": model,
                    "target": "ma_state",
                    "channel": channel,
                    "horizon": h + 1,
                    "rmse": rmse(actual_ma[:, h, c], pred_ma[:, h, c]),
                    "deterministic_rmse": rmse(actual_ma[:, h, c], ma_base[:, h, c]),
                    "improvement": improvement(rmse(actual_ma[:, h, c], ma_base[:, h, c]), rmse(actual_ma[:, h, c], pred_ma[:, h, c])),
                }
            )
    return row, per_symbol, per_horizon


def aggregate(rows: list[dict[str, Any]], per_symbol: list[dict[str, Any]], per_horizon: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    sdf = pd.DataFrame(per_symbol)
    hdf = pd.DataFrame(per_horizon)
    out = []
    for model, group in df.groupby("model"):
        sym = sdf[sdf["model"] == model] if not sdf.empty else pd.DataFrame()
        hor = hdf[hdf["model"] == model] if not hdf.empty else pd.DataFrame()
        sym_by = sym.groupby("symbol").agg(
            rsi=("rsi_reconstruction_improvement", lambda x: pd.to_numeric(x, errors="coerce").median()),
            ma=("ma_reconstruction_improvement", lambda x: pd.to_numeric(x, errors="coerce").median()),
        ) if not sym.empty else pd.DataFrame()
        out.append(
            {
                "model": model,
                "folds": int(len(group)),
                "median_residual_improvement_over_zero": float(pd.to_numeric(group["residual_improvement_over_zero"], errors="coerce").median()),
                "median_rsi_reconstruction_improvement": float(pd.to_numeric(group["rsi_reconstruction_improvement"], errors="coerce").median()),
                "worst_rsi_reconstruction_improvement": float(pd.to_numeric(group["rsi_reconstruction_improvement"], errors="coerce").min()),
                "median_ma_reconstruction_improvement": float(pd.to_numeric(group["ma_reconstruction_improvement"], errors="coerce").median()),
                "worst_ma_reconstruction_improvement": float(pd.to_numeric(group["ma_reconstruction_improvement"], errors="coerce").min()),
                "combined_symbol_time_median_improvement": float(pd.to_numeric(group[group["scenario"] == "combined_symbol_time_exclusion"]["ma_reconstruction_improvement"], errors="coerce").median()),
                "symbols_positive_both": int(((sym_by["rsi"] > 0) & (sym_by["ma"] > 0)).sum()) if not sym_by.empty else 0,
                "horizon_positive_count": int((pd.to_numeric(hor["improvement"], errors="coerce") > 0).sum()) if not hor.empty else 0,
                "save_reload_parity_all": bool(group["save_reload_parity"].astype(str).str.lower().isin(["true", "1"]).all()),
                "prediction_finite_fraction_min": float(pd.to_numeric(group["prediction_finite_fraction"], errors="coerce").min()),
            }
        )
    out.sort(key=lambda row: (min(row["median_rsi_reconstruction_improvement"], row["median_ma_reconstruction_improvement"]), row["combined_symbol_time_median_improvement"]), reverse=True)
    return out


def final_recommendation(leaderboard: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not leaderboard:
        return "residual_event_dataset_blocked", {"reason": "no_evaluable_rows"}
    row = leaderboard[0]
    passed = (
        row["median_rsi_reconstruction_improvement"] > 0
        and row["median_ma_reconstruction_improvement"] > 0
        and row["horizon_positive_count"] >= 6
        and row["symbols_positive_both"] >= 7
        and row["combined_symbol_time_median_improvement"] > 0
        and row["worst_ma_reconstruction_improvement"] > -0.05
        and row["save_reload_parity_all"]
    )
    return ("continue_indicator_residual_research" if passed else "deterministic_indicators_only"), {"best_model": row["model"], "residual_gate_pass": passed, **row}


def memory_report() -> dict[str, Any]:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return {"peak_working_set_mb": float(getattr(info, "peak_wset", math.nan)) / (1024.0 * 1024.0), "rss_mb": float(info.rss) / (1024.0 * 1024.0)}
    except Exception as exc:
        return {"peak_working_set_mb": math.nan, "rss_mb": math.nan, "memory_error": str(exc)}


def main() -> int:
    started = time.perf_counter()
    root = Path(os.getenv("RAWSEQ_RESIDUAL_EVENT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    dataset_dir = Path(os.getenv("RAWSEQ_RESIDUAL_DATASET_DIR", "") or latest_residual_dataset(root))
    manifest = read_json(dataset_dir / "residual_dataset_manifest.json")
    npz = np.load(manifest["dataset_path"], allow_pickle=True)
    data = {key: npz[key] for key in npz.files}
    split = data["split"].astype(str)
    eligible = np.where(np.isin(split, ["train", "validation"]))[0]
    folds = make_folds(data, eligible)
    max_train_rows = int(os.getenv("RAWSEQ_RESIDUAL_MAX_TRAIN_ROWS", "2000") or "2000")
    max_validation_rows = int(os.getenv("RAWSEQ_RESIDUAL_MAX_VALIDATION_ROWS", "1000") or "1000")
    run_hgb = parse_bool(os.getenv("RAWSEQ_RESIDUAL_RUN_HGB", "true"))
    run_dir = root / f"indicator_residual_scout_{now_stamp()}"
    metrics: list[dict[str, Any]] = []
    per_symbol: list[dict[str, Any]] = []
    per_horizon: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    models = ["zero_residual", "regularized_multioutput_linear"] + (["shallow_hgb_residual"] if run_hgb else [])
    for fold in folds:
        train_idx = cap_indices(np.asarray(fold["train"], dtype=int), max_train_rows, "tail")
        val_idx = cap_indices(np.asarray(fold["validation"], dtype=int), max_validation_rows, "head")
        base = {"scenario": fold["scenario"], "fold_id": fold["fold_id"], "train_rows": int(len(train_idx)), "validation_rows": int(len(val_idx)), "validation_symbols": ",".join(sorted(set(str(x) for x in data["symbol"][val_idx])))}
        fold_rows.append(base)
        if len(train_idx) < 100 or len(val_idx) < 20:
            continue
        train_x = feature_matrix(data, train_idx)
        val_x = feature_matrix(data, val_idx)
        train_y = residual_y(data, train_idx)
        good = finite_rows(train_x, train_y)
        train_x, train_y = train_x[good], train_y[good]
        preds: dict[str, np.ndarray] = {"zero_residual": np.zeros((len(val_idx), 40), dtype=np.float32)}
        parities: dict[str, tuple[bool, float]] = {"zero_residual": (True, 0.0)}
        fitted_predictions = []
        for model_name in [m for m in models if m != "zero_residual"]:
            model = fit_cpu_model(model_name, train_x, train_y)
            model.fit(train_x, train_y)
            pred = model_predict(model, val_x)
            preds[model_name] = pred
            parities[model_name] = save_reload_prediction_parity(model, model_predict, val_x[: min(200, len(val_x))])
            fitted_predictions.append(pred)
        if len(fitted_predictions) >= 2:
            preds["conservative_linear_hgb_average"] = np.mean(fitted_predictions, axis=0)
            parities["conservative_linear_hgb_average"] = (all(parities[m][0] for m in preds if m not in {"zero_residual", "conservative_linear_hgb_average"}), max(parities[m][1] for m in parities if m not in {"zero_residual", "conservative_linear_hgb_average"}))
        for model_name, pred in preds.items():
            row, sym, hor = evaluate_residual_prediction(base, data, val_idx, model_name, pred, parities[model_name])
            metrics.append(row)
            per_symbol.extend(sym)
            per_horizon.extend(hor)
    leaderboard = aggregate(metrics, per_symbol, per_horizon)
    rec, details = final_recommendation([row for row in leaderboard if row["model"] != "zero_residual"] or leaderboard)
    mem = memory_report()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "residual_fold_metrics.csv", metrics)
    write_csv(run_dir / "residual_per_symbol_metrics.csv", per_symbol)
    write_csv(run_dir / "residual_per_horizon_metrics.csv", per_horizon)
    write_csv(run_dir / "residual_candidate_leaderboard.csv", leaderboard)
    write_csv(run_dir / "fold_manifest.csv", fold_rows)
    decision = {
        "final_recommendation": rec,
        "decision_details": details,
        "dataset_dir": str(dataset_dir),
        "runtime_seconds": time.perf_counter() - started,
        **mem,
        "june_development_access": False,
        "july_access": False,
        "cpu_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "frozen_candidate_mutation": False,
        "candidate_hash": stable_hash(leaderboard[:5]),
    }
    write_json(run_dir / "residual_candidate_decision.json", decision)
    write_json(run_dir / "save_reload_parity_report.json", {"all_parity": all(bool(row.get("save_reload_parity")) for row in metrics), "max_abs_diff": max([float(row.get("save_reload_max_abs_diff", 0.0)) for row in metrics], default=0.0)})
    (run_dir / "residual_candidate_decision.txt").write_text(
        "\n".join(
            [
                "Rawseq 1m indicator residual scout",
                f"dataset_dir={dataset_dir}",
                f"runtime_seconds={decision['runtime_seconds']:.2f}",
                f"peak_working_set_mb={decision.get('peak_working_set_mb')}",
                f"best_residual_model={(leaderboard[0]['model'] if leaderboard else 'none')}",
                f"final_recommendation={rec}",
                "safety: CPU only; no June/July development access; no frozen candidate mutation; no orders/promotion",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"residual_scout_dir={run_dir}")
    print(f"runtime_seconds={decision['runtime_seconds']:.2f}")
    print(f"peak_working_set_mb={decision.get('peak_working_set_mb')}")
    print(f"best_residual_model={(leaderboard[0]['model'] if leaderboard else 'none')}")
    print(f"final_recommendation={rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
