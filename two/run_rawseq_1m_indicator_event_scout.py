#!/usr/bin/env python3
"""CPU event-probability scout for indicator companion event targets."""

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

from scripts.tiny.rawseq_1m_baseline_utils import (  # noqa: E402
    SAFETY_FLAGS,
    expected_calibration_error,
    file_sha256,
    log_loss_score,
    max_calibration_error,
    now_stamp,
    parse_bool,
    pr_auc_lift,
    rank_auc,
    save_reload_prediction_parity,
    stable_hash,
    write_csv,
    write_json,
)
from scripts.tiny.run_rawseq_1m_dual_timescale_indicator_scout import cap_indices, feature_matrix, finite_rows, make_folds  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_indicator_companion_scout")
FROZEN_CANDIDATE_HASH = "8109480bd6b0c2575337cada9be991746199200d42feaeea1fd7b9dd0b3eb6bf"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def latest_residual_dataset(root: Path) -> Path:
    dirs = sorted(root.glob("indicator_residual_event_dataset_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise SystemExit(f"No residual/event datasets found under {root}")
    return dirs[0]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, dtype=float) - np.asarray(p, dtype=float)) ** 2)) if len(y) else math.nan


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=float)
    logits = np.log(p / (1.0 - p))
    if len(y) < 3 or np.std(logits) <= 1e-12 or np.std(y) <= 1e-12:
        return math.nan, math.nan
    slope, intercept = np.polyfit(logits, y, 1)
    return float(slope), float(intercept)


def metric_row(base: dict[str, Any], y: np.ndarray, p: np.ndarray, baseline: np.ndarray, parity: tuple[bool, float]) -> dict[str, Any]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    baseline = np.clip(np.asarray(baseline, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=float)
    base_brier = brier(y, baseline)
    score = brier(y, p)
    base_ll = log_loss_score(y, baseline)
    ll = log_loss_score(y, p)
    ap, lift = pr_auc_lift(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        **base,
        "rows": int(len(y)),
        "events": int(np.sum(y > 0.5)),
        "event_prevalence": float(np.mean(y)) if len(y) else math.nan,
        "brier_score": score,
        "prevalence_brier_score": base_brier,
        "brier_skill": (base_brier - score) / base_brier if base_brier > 0 else math.nan,
        "log_loss": ll,
        "prevalence_log_loss": base_ll,
        "log_loss_improvement": base_ll - ll if math.isfinite(base_ll) and math.isfinite(ll) else math.nan,
        "pr_auc": ap,
        "pr_auc_lift_over_prevalence": lift,
        "roc_auc": rank_auc(y, p),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "expected_calibration_error": expected_calibration_error(y, p),
        "maximum_calibration_error": max_calibration_error(y, p),
        "save_reload_parity": bool(parity[0]),
        "save_reload_max_abs_diff": float(parity[1]),
        "prediction_finite_fraction": float(np.isfinite(p).mean()),
        "june_development_access": False,
        "july_access": False,
        **SAFETY_FLAGS,
        "public_recorded_data_only": True,
        "frozen_candidate_mutation": False,
    }


def fit_classifier(name: str, x: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name in {"regularized_logistic", "regime_feature_logistic"}:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=300, solver="lbfgs"))
    if name == "shallow_hgb":
        return make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=40, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.01, random_state=1337))
    raise ValueError(name)


def predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(model[-1].predict_proba(model[:-1].transform(x))[:, 1], dtype=float)


def regime_indices(cols: int) -> np.ndarray:
    # Static 31 + short last 12 + summaries. Vol/range-like columns are stable
    # positions from the temporal feature summaries and enough for a secondary
    # no-symbol-ID regime-feature model.
    return np.asarray([idx for idx in range(cols) if idx >= 31], dtype=int)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    out = []
    for (model, group_name), group in df.groupby(["model", "event_group"], dropna=False):
        brier_skill = pd.to_numeric(group["brier_skill"], errors="coerce")
        out.append(
            {
                "model": model,
                "event_group": group_name,
                "rows": int(pd.to_numeric(group["rows"], errors="coerce").sum()),
                "targets": int(group["event_name"].nunique()),
                "fold_rows": int(len(group)),
                "fold_wins": int((brier_skill > 0).sum()),
                "fold_win_fraction": float((brier_skill > 0).mean()),
                "median_brier_skill": float(brier_skill.median()),
                "worst_fold_brier_skill": float(brier_skill.min()),
                "median_log_loss_improvement": float(pd.to_numeric(group["log_loss_improvement"], errors="coerce").median()),
                "median_pr_auc_lift": float(pd.to_numeric(group["pr_auc_lift_over_prevalence"], errors="coerce").median()),
                "save_reload_parity_all": bool(group["save_reload_parity"].astype(str).str.lower().isin(["true", "1"]).all()),
            }
        )
    out.sort(key=lambda row: (row["median_brier_skill"], row["worst_fold_brier_skill"], row["median_log_loss_improvement"]), reverse=True)
    return out


def event_survival_rows(rows: list[dict[str, Any]], min_prevalence: float) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    out = []
    learned = df[df["model"] != "constant_prevalence"].copy()
    for (model, event_name), group in learned.groupby(["model", "event_name"], dropna=False):
        brier_skill = pd.to_numeric(group["brier_skill"], errors="coerce")
        pr_lift = pd.to_numeric(group["pr_auc_lift_over_prevalence"], errors="coerce")
        prevalence = pd.to_numeric(group["event_prevalence"], errors="coerce")
        combined = group[group["scenario"] == "combined_symbol_time_exclusion"]
        combined_brier = pd.to_numeric(combined["brier_skill"], errors="coerce")
        loso = group[group["scenario"] == "leave_one_symbol_out"]
        symbol_rows = []
        for symbol, sym_group in loso.groupby("validation_symbols", dropna=False):
            sym_brier = pd.to_numeric(sym_group["brier_skill"], errors="coerce")
            symbol_rows.append({"symbol": symbol, "positive": bool(sym_brier.median() > 0)})
        positive_symbols = sum(1 for row in symbol_rows if row["positive"])
        sufficient_event_prevalence = bool(prevalence.median() >= min_prevalence and prevalence.median() <= (1.0 - min_prevalence))
        gate_results = {
            "positive_median_brier_skill": bool(brier_skill.median() > 0),
            "positive_worst_fold_brier_skill": bool(brier_skill.min() > 0),
            "positive_pr_auc_lift": bool(pr_lift.median() > 0),
            "sufficient_event_prevalence": sufficient_event_prevalence,
            "at_least_7_of_9_symbols_positive": bool(positive_symbols >= 7),
            "save_reload_parity": bool(group["save_reload_parity"].astype(str).str.lower().isin(["true", "1"]).all()),
            "combined_symbol_time_stability": bool(len(combined_brier) > 0 and combined_brier.median() > 0 and combined_brier.min() > 0),
        }
        out.append(
            {
                "model": model,
                "event_name": event_name,
                "event_group": str(group["event_group"].iloc[0]),
                "horizon": int(group["horizon"].iloc[0]),
                "fold_rows": int(len(group)),
                "rows": int(pd.to_numeric(group["rows"], errors="coerce").sum()),
                "median_event_prevalence": float(prevalence.median()),
                "median_brier_skill": float(brier_skill.median()),
                "worst_fold_brier_skill": float(brier_skill.min()),
                "median_log_loss_improvement": float(pd.to_numeric(group["log_loss_improvement"], errors="coerce").median()),
                "median_pr_auc_lift": float(pr_lift.median()),
                "roc_auc_median": float(pd.to_numeric(group["roc_auc"], errors="coerce").median()),
                "positive_symbols": int(positive_symbols),
                "combined_symbol_time_median_brier_skill": float(combined_brier.median()) if len(combined_brier) else math.nan,
                "combined_symbol_time_worst_brier_skill": float(combined_brier.min()) if len(combined_brier) else math.nan,
                **gate_results,
                "survives_event_gate": all(gate_results.values()),
                "rejection_reasons": ";".join(key for key, passed in gate_results.items() if not passed),
            }
        )
    out.sort(
        key=lambda row: (
            row["survives_event_gate"],
            row["median_brier_skill"],
            row["worst_fold_brier_skill"],
            row["positive_symbols"],
            row["median_pr_auc_lift"],
        ),
        reverse=True,
    )
    return out


def final_recommendation(leaderboard: list[dict[str, Any]], survival: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    survivors = [row for row in survival if row.get("survives_event_gate")]
    if survivors:
        return "freeze_indicator_event_companion_before_future_holdout", {
            "best_event_model": survivors[0]["model"],
            "best_event_name": survivors[0]["event_name"],
            "best_event_group": survivors[0]["event_group"],
            "surviving_event_count": len(survivors),
            **survivors[0],
        }
    learned = [row for row in leaderboard if row["model"] != "constant_prevalence"]
    if not learned:
        return "residual_event_dataset_blocked", {"reason": "no_learned_event_rows"}
    return "deterministic_indicators_only", {"best_event_model": learned[0]["model"], "best_event_group": learned[0]["event_group"], **learned[0]}


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
    dataset_dir = Path(os.getenv("RAWSEQ_EVENT_DATASET_DIR", "") or latest_residual_dataset(root))
    manifest = read_json(dataset_dir / "residual_dataset_manifest.json")
    npz = np.load(manifest["dataset_path"], allow_pickle=True)
    data = {key: npz[key] for key in npz.files}
    split = data["split"].astype(str)
    eligible = np.where(np.isin(split, ["train", "validation"]))[0]
    folds = make_folds(data, eligible)
    max_train_rows = int(os.getenv("RAWSEQ_EVENT_MAX_TRAIN_ROWS", "2000") or "2000")
    max_validation_rows = int(os.getenv("RAWSEQ_EVENT_MAX_VALIDATION_ROWS", "1000") or "1000")
    run_hgb = parse_bool(os.getenv("RAWSEQ_EVENT_RUN_HGB", "true"))
    max_events = int(os.getenv("RAWSEQ_EVENT_MAX_EVENTS", "0") or "0")
    min_event_prevalence = float(os.getenv("RAWSEQ_EVENT_MIN_PREVALENCE", "0.01") or "0.01")
    event_names = [str(x) for x in data["event_names"]]
    event_groups = [str(x) for x in data["event_groups"]]
    event_horizons = [int(x) for x in data["event_horizons"]]
    event_indices = list(range(len(event_names)))[: max_events or None]
    run_dir = root / f"indicator_event_scout_{now_stamp()}"
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    model_names = ["regularized_logistic", "regime_feature_logistic"] + (["shallow_hgb"] if run_hgb else [])
    for fold in folds:
        train_idx = cap_indices(np.asarray(fold["train"], dtype=int), max_train_rows, "tail")
        val_idx = cap_indices(np.asarray(fold["validation"], dtype=int), max_validation_rows, "head")
        fold_rows.append({"scenario": fold["scenario"], "fold_id": fold["fold_id"], "train_rows": int(len(train_idx)), "validation_rows": int(len(val_idx))})
        if len(train_idx) < 100 or len(val_idx) < 20:
            continue
        train_x = feature_matrix(data, train_idx)
        val_x = feature_matrix(data, val_idx)
        good = finite_rows(train_x)
        train_x = train_x[good]
        train_idx = train_idx[good]
        ridx = regime_indices(train_x.shape[1])
        for event_idx in event_indices:
            y_train = data["event_targets"][train_idx, event_idx].astype(float)
            y_val = data["event_targets"][val_idx, event_idx].astype(float)
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue
            prevalence = float(np.mean(y_train))
            baseline = np.full(len(y_val), np.clip(prevalence, 1e-6, 1.0 - 1e-6))
            base = {
                "scenario": fold["scenario"],
                "fold_id": fold["fold_id"],
                "event_name": event_names[event_idx],
                "event_group": event_groups[event_idx],
                "horizon": event_horizons[event_idx],
                "train_rows": int(len(y_train)),
                "validation_rows": int(len(y_val)),
                "validation_symbols": ",".join(sorted(set(str(x) for x in data["symbol"][val_idx]))),
            }
            rows.append(metric_row({**base, "model": "constant_prevalence"}, y_val, baseline, baseline, (True, 0.0)))
            preds = []
            parity_values = []
            for model_name in model_names:
                x_fit = train_x[:, ridx] if model_name == "regime_feature_logistic" else train_x
                x_val = val_x[:, ridx] if model_name == "regime_feature_logistic" else val_x
                model = fit_classifier(model_name, x_fit, y_train)
                model.fit(x_fit, y_train)
                pred = np.clip(predict_proba(model, x_val), 1e-6, 1.0 - 1e-6)
                parity = save_reload_prediction_parity(model, predict_proba, x_val[: min(200, len(x_val))])
                rows.append(metric_row({**base, "model": model_name}, y_val, pred, baseline, parity))
                preds.append(pred)
                parity_values.append(parity)
            if len(preds) >= 2:
                avg = np.mean(preds, axis=0)
                rows.append(metric_row({**base, "model": "conservative_probability_average"}, y_val, avg, baseline, (all(x[0] for x in parity_values), max(x[1] for x in parity_values))))
    leaderboard = aggregate(rows)
    survival = event_survival_rows(rows, min_event_prevalence)
    rec, details = final_recommendation(leaderboard, survival)
    mem = memory_report()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "event_fold_metrics.csv", rows)
    write_csv(run_dir / "event_candidate_leaderboard.csv", leaderboard)
    write_csv(run_dir / "event_target_survival.csv", survival)
    write_csv(run_dir / "fold_manifest.csv", fold_rows)
    survivors = [row for row in survival if row.get("survives_event_gate")]
    branch_closure = {
        "residual_lane_archive_status": "closed_deterministic_baseline_not_beaten",
        "indicator_event_survivor_count": len(survivors),
        "indicator_ml_branch_status": "event_survivor_found" if survivors else "closed_deterministic_baseline_not_beaten",
        "closed_lanes": [
            "future_RSI_regression",
            "future_EMA_state_regression",
            "RSI_residual_paths",
            "EMA_residual_paths",
            "GRU_or_temporal_models_for_indicator_targets",
        ],
        "gui_policy": "use deterministic indicators plus frozen validated downside-risk probability unless an event family is separately frozen",
        "next_research_direction": "multi_horizon_multi_severity_downside_risk_outputs",
        "june_development_access": False,
        "july_access": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "frozen_candidate_mutation": False,
    }
    decision = {
        "final_recommendation": rec,
        "decision_details": details,
        "dataset_dir": str(dataset_dir),
        "runtime_seconds": time.perf_counter() - started,
        **mem,
        "event_targets_evaluated": len(event_indices),
        "june_development_access": False,
        "july_access": False,
        "cpu_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "frozen_candidate_mutation": False,
        "candidate_hash": stable_hash(leaderboard[:5]),
        "surviving_event_count": len(survivors),
        "branch_closure": branch_closure,
    }
    write_json(run_dir / "event_candidate_decision.json", decision)
    write_json(run_dir / "indicator_ml_branch_closure.json", branch_closure)
    write_json(run_dir / "save_reload_parity_report.json", {"all_parity": all(bool(row.get("save_reload_parity")) for row in rows), "max_abs_diff": max([float(row.get("save_reload_max_abs_diff", 0.0)) for row in rows], default=0.0)})
    (run_dir / "event_candidate_decision.txt").write_text(
        "\n".join(
            [
                "Rawseq 1m indicator event scout",
                f"dataset_dir={dataset_dir}",
                f"runtime_seconds={decision['runtime_seconds']:.2f}",
                f"peak_working_set_mb={decision.get('peak_working_set_mb')}",
                f"best_event_classifier={(leaderboard[0]['model'] if leaderboard else 'none')}",
                f"best_event_group={(leaderboard[0]['event_group'] if leaderboard else 'none')}",
                f"surviving_event_count={len(survivors)}",
                f"indicator_ml_branch_status={branch_closure['indicator_ml_branch_status']}",
                f"final_recommendation={rec}",
                "safety: CPU only; no June/July development access; no frozen candidate mutation; no orders/promotion",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"event_scout_dir={run_dir}")
    print(f"runtime_seconds={decision['runtime_seconds']:.2f}")
    print(f"peak_working_set_mb={decision.get('peak_working_set_mb')}")
    print(f"best_event_classifier={(leaderboard[0]['model'] if leaderboard else 'none')}")
    print(f"best_event_group={(leaderboard[0]['event_group'] if leaderboard else 'none')}")
    print(f"surviving_event_count={len(survivors)}")
    print(f"indicator_ml_branch_status={branch_closure['indicator_ml_branch_status']}")
    print(f"final_recommendation={rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
