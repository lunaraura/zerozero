#!/usr/bin/env python3
"""Development-only challenger tournament for downside-risk exceedance models.

This report runs while the frozen future-shadow candidate accumulates evidence.
It uses only the frozen candidate's development source NPZ train/validation
rows. It never reads future-shadow labels or untouched holdout rows for model,
hyperparameter, threshold, or advancement selection.

No training outside this development report, no recalibration of the frozen
candidate, no threshold changes, no orders, no private API, no champion
mutation, no promotion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from datetime import timedelta
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.freeze_rawseq_downside_risk_cpu_candidate import (
    TARGET_NAME,
    TARGET_THRESHOLD_VOL_UNITS,
    binary_metric_row,
    calibration_error,
    fit_logistic_coef,
    fit_scaler,
    predict_logistic_coef,
    pr_auc,
    rank_auc,
    stable_hash,
    transform,
)
from scripts.tiny.report_rawseq_downside_target_redesign import EPSILON, build_folds, make_targets

DEFAULT_CANDIDATE_DIR = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_cpu_candidates" / "rawseq_downside_risk_cpu_candidate_20260711T233404Z"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_challenger_tournaments"
HORIZONS = [60, 120, 240, 480]


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return resolve_path(raw) if raw else default


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def frozen_shadow_start_iso(candidate_dir: Path) -> str:
    rule_path = candidate_dir / "rawseq_downside_risk_future_acceptance_rule.json"
    if not rule_path.exists():
        return ""
    rule = read_json(rule_path)
    return str(rule.get("frozen_at_iso", ""))


def challenger_development_cutoff_iso(candidate_dir: Path) -> str:
    explicit = os.getenv("RAWSEQ_CHALLENGER_DEVELOPMENT_CUTOFF_ISO", "").strip()
    if explicit:
        return explicit
    frozen_at = frozen_shadow_start_iso(candidate_dir)
    if not frozen_at:
        return ""
    parsed = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    parsed = parsed.replace(tzinfo=parsed.tzinfo or UTC)
    return (parsed - timedelta(microseconds=1)).isoformat()


def sklearn_available() -> tuple[bool, str]:
    try:
        import sklearn  # type: ignore

        return True, str(sklearn.__version__)
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, repr(exc)


def load_development_data(candidate_dir: Path, max_dev_rows: int = 0) -> dict[str, Any]:
    contract = read_json(candidate_dir / "rawseq_downside_risk_cpu_candidate_contract.json")
    npz_path = resolve_path(contract["source_npz"])
    with np.load(npz_path, allow_pickle=False) as data:
        x_seq = data["X"].astype(np.float32)
        y_raw = data["y"].astype(np.float64)
        splits = data["splits"].astype(str)
        timestamps = data["decision_timestamps"].astype(np.float64)
        source_rows = data["source_row_indices"].astype(np.int64)
        feature_columns = [str(x) for x in data["feature_columns"]]
        target_columns = [str(x) for x in data["target_columns"]]
    x = x_seq[:, -1, :].astype(np.float64)
    pre_idx = np.flatnonzero(np.isin(splits, ["train", "validation"]))
    if max_dev_rows and max_dev_rows > 0:
        pre_idx = pre_idx[-min(max_dev_rows, len(pre_idx)) :]
    holdout_rows_present = int(np.sum(splits == "untouched_holdout"))
    vol_col = "realized_volatility_bps_fw60" if "realized_volatility_bps_fw60" in feature_columns else "rolling_volatility_bps_fw60"
    vol = np.maximum(np.abs(x[:, feature_columns.index(vol_col)].astype(np.float64)), EPSILON)
    targets = make_targets(y_raw[pre_idx], vol[pre_idx], np.arange(0, max(1, int(np.sum(splits[pre_idx] == "train")))))
    y = targets[TARGET_NAME]["values"].astype(np.float64)
    return {
        "contract": contract,
        "npz_path": npz_path,
        "x": x[pre_idx],
        "y": y,
        "timestamps": timestamps[pre_idx],
        "source_rows": source_rows[pre_idx],
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "holdout_rows_present": holdout_rows_present,
        "pre_idx_rows": int(len(pre_idx)),
        "source_npz_sha256": file_sha256(npz_path),
        "vol_values": vol[pre_idx],
    }


def select_feature_indices(feature_columns: list[str], mode: str, train_x: np.ndarray | None = None, train_y: np.ndarray | None = None) -> np.ndarray:
    if mode == "all":
        return np.arange(len(feature_columns), dtype=np.int64)
    if mode == "volatility_only":
        names = [
            "realized_volatility_bps_fw60",
            "rolling_volatility_bps_fw60",
            "atr_bps_fw60",
            "rolling_range_bps_fw60",
        ]
        return np.asarray([feature_columns.index(name) for name in names if name in feature_columns], dtype=np.int64)
    if mode == "oracle_top20":
        if train_x is None or train_y is None:
            return np.arange(min(20, len(feature_columns)), dtype=np.int64)
        scores = []
        for idx in range(train_x.shape[1]):
            values = train_x[:, idx].astype(np.float64)
            if np.nanstd(values) <= 1e-12:
                scores.append(0.0)
                continue
            per_target = []
            for target_idx in range(train_y.shape[1]):
                yy = train_y[:, target_idx]
                if np.nanstd(yy) <= 1e-12:
                    continue
                corr = np.corrcoef(np.nan_to_num(values), np.nan_to_num(yy))[0, 1]
                if math.isfinite(corr):
                    per_target.append(abs(corr))
            scores.append(max(per_target, default=0.0))
        return np.asarray(np.argsort(scores)[-20:], dtype=np.int64)
    raise ValueError(mode)


def fit_predict_logistic(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, l2: float, iterations: int = 100) -> np.ndarray:
    scaler = fit_scaler(train_x)
    train_scaled = transform(train_x, scaler)
    val_scaled = transform(val_x, scaler)
    preds = []
    for idx in range(train_y.shape[1]):
        coef = fit_logistic_coef(train_scaled, train_y[:, idx], l2=l2, iterations=iterations)
        preds.append(predict_logistic_coef(val_scaled, coef))
    return np.column_stack(preds)


def fit_predict_hgb(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    preds = []
    for idx in range(train_y.shape[1]):
        y = (train_y[:, idx] > 0.5).astype(int)
        if y.min() == y.max():
            preds.append(np.full(len(val_x), float(y.mean())))
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=0.01, random_state=1337),
        )
        model.fit(train_x, y)
        preds.append(model.predict_proba(val_x)[:, 1])
    return np.column_stack(preds)


def fit_predict_extra_trees(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesClassifier  # type: ignore
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore

    preds = []
    for idx in range(train_y.shape[1]):
        y = (train_y[:, idx] > 0.5).astype(int)
        if y.min() == y.max():
            preds.append(np.full(len(val_x), float(y.mean())))
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=80, max_depth=4, min_samples_leaf=100, random_state=1337, n_jobs=1),
        )
        model.fit(train_x, y)
        preds.append(model.predict_proba(val_x)[:, 1])
    return np.column_stack(preds)


def predict_model(model_name: str, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, sklearn_ok: bool) -> tuple[np.ndarray | None, str]:
    if model_name == "constant_prevalence_baseline":
        prevalence = np.nanmean(train_y, axis=0)
        return np.tile(np.clip(prevalence, 1e-6, 1 - 1e-6), (len(val_x), 1)), ""
    if model_name == "logistic_l2_0_001":
        return fit_predict_logistic(train_x, train_y, val_x, l2=0.001), ""
    if model_name == "logistic_l2_0_01":
        return fit_predict_logistic(train_x, train_y, val_x, l2=0.01), ""
    if model_name == "logistic_l2_0_1":
        return fit_predict_logistic(train_x, train_y, val_x, l2=0.1), ""
    if model_name == "hist_gradient_boosting":
        if not sklearn_ok:
            return None, "sklearn unavailable"
        return fit_predict_hgb(train_x, train_y, val_x), ""
    if model_name == "shallow_extra_trees":
        if not sklearn_ok:
            return None, "sklearn unavailable"
        return fit_predict_extra_trees(train_x, train_y, val_x), ""
    return None, f"model not implemented: {model_name}"


def aggregate_model_rows(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in fold_rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    ranking = []
    for model, rows in by_model.items():
        skills = [safe_float(row.get("mean_brier_skill_score")) for row in rows if math.isfinite(safe_float(row.get("mean_brier_skill_score")))]
        pr_lifts = [safe_float(row.get("pr_auc_lift_over_prevalence")) for row in rows if math.isfinite(safe_float(row.get("pr_auc_lift_over_prevalence")))]
        if not pr_lifts:
            pr_lifts = [safe_float(row.get("median_pr_auc_lift_over_prevalence")) for row in rows if math.isfinite(safe_float(row.get("median_pr_auc_lift_over_prevalence")))]
        eces = [safe_float(row.get("expected_calibration_error")) for row in rows if math.isfinite(safe_float(row.get("expected_calibration_error")))]
        if not eces:
            eces = [safe_float(row.get("mean_expected_calibration_error")) for row in rows if math.isfinite(safe_float(row.get("mean_expected_calibration_error")))]
        wins = int(sum(1 for val in skills if val > 0))
        median_skill = float(np.median(skills)) if skills else math.nan
        worst_skill = float(np.min(skills)) if skills else math.nan
        median_pr = float(np.median(pr_lifts)) if pr_lifts else math.nan
        ece_std = float(np.std(eces)) if eces else math.nan
        selection_score = (
            (median_skill if math.isfinite(median_skill) else -999.0)
            + 0.5 * (worst_skill if math.isfinite(worst_skill) else -999.0)
            + 0.25 * (median_pr if math.isfinite(median_pr) else -999.0)
            - 0.1 * (ece_std if math.isfinite(ece_std) else 999.0)
            + 0.01 * wins
        )
        ranking.append(
            {
                "model": model,
                "folds": len(rows),
                "positive_brier_skill_folds": wins,
                "median_brier_skill_score": median_skill,
                "worst_fold_brier_skill_score": worst_skill,
                "median_pr_auc_lift_over_prevalence": median_pr,
                "calibration_stability_ece_std": ece_std,
                "selection_score": selection_score,
                "selection_gate_pass": bool(wins >= 4 and median_skill > 0 and worst_skill > -0.02 and median_pr > 0.0),
            }
        )
    return sorted(
        ranking,
        key=lambda row: (
            int(bool(row["selection_gate_pass"])),
            safe_float(row["selection_score"], -999),
            safe_float(row["median_brier_skill_score"], -999),
            safe_float(row["worst_fold_brier_skill_score"], -999),
        ),
        reverse=True,
    )


def target_label_stability(y_raw: np.ndarray, vol: np.ndarray) -> list[dict[str, Any]]:
    base = (np.abs(y_raw) / np.maximum(vol[:, None], EPSILON)) > TARGET_THRESHOLD_VOL_UNITS
    rows = []
    scenarios = {
        "threshold_minus_0_05": (np.abs(y_raw) / np.maximum(vol[:, None], EPSILON)) > (TARGET_THRESHOLD_VOL_UNITS - 0.05),
        "threshold_plus_0_05": (np.abs(y_raw) / np.maximum(vol[:, None], EPSILON)) > (TARGET_THRESHOLD_VOL_UNITS + 0.05),
        "vol_denominator_plus_5pct": (np.abs(y_raw) / np.maximum(vol[:, None] * 1.05, EPSILON)) > TARGET_THRESHOLD_VOL_UNITS,
        "vol_denominator_minus_5pct": (np.abs(y_raw) / np.maximum(vol[:, None] * 0.95, EPSILON)) > TARGET_THRESHOLD_VOL_UNITS,
    }
    for name, values in scenarios.items():
        flips = values != base
        rows.append(
            {
                "scenario": name,
                "rows": int(base.size),
                "flip_fraction": float(np.mean(flips)),
                "base_event_fraction": float(np.mean(base)),
                "scenario_event_fraction": float(np.mean(values)),
            }
        )
    return rows


def main() -> int:
    candidate_dir = env_path("RAWSEQ_DOWNSIDE_CHALLENGER_CANDIDATE_DIR", DEFAULT_CANDIDATE_DIR)
    output_root = env_path("RAWSEQ_DOWNSIDE_CHALLENGER_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    max_dev_rows = int(float(os.getenv("RAWSEQ_DOWNSIDE_CHALLENGER_MAX_DEV_ROWS", "100000")))
    include_sklearn = parse_bool(os.getenv("RAWSEQ_DOWNSIDE_CHALLENGER_INCLUDE_SKLEARN", "true"))
    out_dir = output_root / f"rawseq_downside_risk_challenger_tournament_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    data = load_development_data(candidate_dir, max_dev_rows=max_dev_rows)
    sklearn_ok, sklearn_status = sklearn_available()
    sklearn_ok = sklearn_ok and include_sklearn
    folds = build_folds(len(data["x"]), data["timestamps"], data["source_rows"])
    models = [
        "constant_prevalence_baseline",
        "volatility_only_logistic",
        "logistic_l2_0_001",
        "logistic_l2_0_01",
        "logistic_l2_0_1",
        "hist_gradient_boosting",
        "shallow_extra_trees",
        "oracle_top20_logistic_diagnostic",
        "generalized_additive_model_skipped",
    ]
    fold_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    oof_by_model: dict[str, np.ndarray] = {model: np.full_like(data["y"], np.nan, dtype=np.float64) for model in models}

    for fold in folds:
        train_idx = fold["train_indices"]
        val_idx = fold["validation_indices"]
        train_y = data["y"][train_idx]
        const_prob = np.tile(np.clip(np.nanmean(train_y, axis=0), 1e-6, 1 - 1e-6), (len(val_idx), 1))
        for model_name in models:
            if model_name == "generalized_additive_model_skipped":
                skipped_rows.append({"model": model_name, "reason": "GAM dependency not installed/standardized for this repository"})
                continue
            if model_name == "volatility_only_logistic":
                feature_idx = select_feature_indices(data["feature_columns"], "volatility_only")
            elif model_name == "oracle_top20_logistic_diagnostic":
                feature_idx = select_feature_indices(data["feature_columns"], "oracle_top20", data["x"][train_idx], train_y)
            else:
                feature_idx = select_feature_indices(data["feature_columns"], "all")
            if feature_idx.size == 0:
                skipped_rows.append({"model": model_name, "fold_id": fold["fold_id"], "reason": "no feature columns selected"})
                continue
            actual_model = "logistic_l2_0_01" if model_name in {"volatility_only_logistic", "oracle_top20_logistic_diagnostic"} else model_name
            pred, skip_reason = predict_model(actual_model, data["x"][train_idx][:, feature_idx], train_y, data["x"][val_idx][:, feature_idx], sklearn_ok)
            if pred is None:
                skipped_rows.append({"model": model_name, "fold_id": fold["fold_id"], "reason": skip_reason})
                continue
            oof_by_model[model_name][val_idx] = pred
            row, hrows, rel = binary_metric_row(
                fold["fold_id"],
                model_name,
                data["y"][val_idx],
                pred,
                const_prob,
                data["target_columns"],
                stable_hash([data["feature_columns"][int(i)] for i in feature_idx]),
            )
            row.update(
                {
                    "feature_count": int(feature_idx.size),
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(val_idx)),
                    "train_timestamp_start": float(data["timestamps"][train_idx[0]]),
                    "train_timestamp_end": float(data["timestamps"][train_idx[-1]]),
                    "validation_timestamp_start": float(data["timestamps"][val_idx[0]]),
                    "validation_timestamp_end": float(data["timestamps"][val_idx[-1]]),
                    "holdout_used_for_selection": False,
                    "future_shadow_labels_used": False,
                }
            )
            fold_rows.append(row)
            horizon_rows.extend(hrows)
            reliability_rows.extend(rel)

    learned_models = [name for name in oof_by_model if name not in {"constant_prevalence_baseline", "generalized_additive_model_skipped"}]
    finite_models = [name for name in learned_models if np.isfinite(oof_by_model[name]).all(axis=1).any()]
    if finite_models:
        stacked = np.stack([oof_by_model[name] for name in finite_models], axis=0)
        valid_count = np.isfinite(stacked).sum(axis=0)
        ensemble = np.divide(
            np.nansum(stacked, axis=0),
            valid_count,
            out=np.full_like(oof_by_model[finite_models[0]], np.nan, dtype=np.float64),
            where=valid_count > 0,
        )
        oof_by_model["conservative_oof_probability_ensemble"] = ensemble
        for fold in folds:
            val_idx = fold["validation_indices"]
            pred = ensemble[val_idx]
            if not np.isfinite(pred).all():
                continue
            train_y = data["y"][fold["train_indices"]]
            const_prob = np.tile(np.clip(np.nanmean(train_y, axis=0), 1e-6, 1 - 1e-6), (len(val_idx), 1))
            row, hrows, rel = binary_metric_row(
                fold["fold_id"],
                "conservative_oof_probability_ensemble",
                data["y"][val_idx],
                pred,
                const_prob,
                data["target_columns"],
                stable_hash(finite_models),
            )
            row.update({"feature_count": "ensemble", "train_rows": int(len(fold["train_indices"])), "validation_rows": int(len(val_idx)), "holdout_used_for_selection": False, "future_shadow_labels_used": False})
            fold_rows.append(row)
            horizon_rows.extend(hrows)
            reliability_rows.extend(rel)

    ranking_rows = aggregate_model_rows(fold_rows)
    stability_rows = target_label_stability(data["y"], data["vol_values"])
    frozen_at_iso = frozen_shadow_start_iso(candidate_dir)
    development_cutoff_iso = challenger_development_cutoff_iso(candidate_dir)
    contract = {
        "created_at": datetime.now(UTC).isoformat(),
        "candidate_dir": str(candidate_dir),
        "frozen_shadow_start_iso": frozen_at_iso,
        "challenger_development_cutoff_iso": development_cutoff_iso,
        "challenger_cutoff_strictly_before_frozen_shadow_start": bool(
            frozen_at_iso and development_cutoff_iso and development_cutoff_iso < frozen_at_iso
        ),
        "source_npz": str(data["npz_path"]),
        "source_npz_sha256": data["source_npz_sha256"],
        "target_name": TARGET_NAME,
        "target_threshold_vol_units": TARGET_THRESHOLD_VOL_UNITS,
        "horizons": HORIZONS,
        "dev_rows": data["pre_idx_rows"],
        "max_dev_rows": max_dev_rows,
        "holdout_rows_present_but_unused": data["holdout_rows_present"],
        "holdout_used_for_selection": False,
        "future_shadow_labels_used": False,
        "future_shadow_dirs_read": [],
        "models_requested": models,
        "sklearn_status": sklearn_status,
        "sklearn_models_enabled": sklearn_ok,
        "selection_rule": "median_brier_skill + worst_fold_brier_skill + calibration stability + PR-AUC lift + simplicity; validation folds only",
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "training": "development_only_cpu_challenger_models",
        "recalibration_of_frozen_candidate": False,
        "threshold_changes": False,
        "feature_columns_sha256": stable_hash(data["feature_columns"]),
        "target_columns_sha256": stable_hash(data["target_columns"]),
        "target_values_sha256": array_sha256(data["y"]),
    }
    contract["contract_sha256"] = stable_hash(contract)
    write_csv(out_dir / "rawseq_downside_risk_challenger_fold_metrics.csv", fold_rows)
    write_csv(out_dir / "rawseq_downside_risk_challenger_per_horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "rawseq_downside_risk_challenger_model_ranking.csv", ranking_rows)
    write_csv(out_dir / "rawseq_downside_risk_challenger_reliability_bins.csv", reliability_rows)
    write_csv(out_dir / "rawseq_downside_risk_challenger_skipped_models.csv", skipped_rows)
    write_csv(out_dir / "rawseq_downside_risk_target_label_stability.csv", stability_rows)
    write_json(out_dir / "rawseq_downside_risk_challenger_tournament_contract.json", contract)
    best = ranking_rows[0] if ranking_rows else {}
    recommendation = {
        "recommendation": "keep_collecting_frozen_future_shadow_and_review_challengers_later",
        "best_development_challenger": best.get("model", ""),
        "best_selection_gate_pass": best.get("selection_gate_pass", False),
        "frozen_shadow_start_iso": frozen_at_iso,
        "challenger_development_cutoff_iso": development_cutoff_iso,
        "challenger_cutoff_strictly_before_frozen_shadow_start": contract["challenger_cutoff_strictly_before_frozen_shadow_start"],
        "holdout_used_for_selection": False,
        "future_shadow_labels_used": False,
        "promotion": False,
        "orders": False,
        "champion_mutation": False,
    }
    write_json(out_dir / "rawseq_downside_risk_challenger_recommendation.json", recommendation)
    lines = [
        "Rawseq downside-risk challenger tournament",
        f"Output: {out_dir}",
        f"Development rows: {data['pre_idx_rows']}",
        f"Holdout rows present but unused: {data['holdout_rows_present']}",
        f"Best challenger: {best.get('model', '')}",
        f"Best median Brier skill: {best.get('median_brier_skill_score', '')}",
        f"Best worst-fold Brier skill: {best.get('worst_fold_brier_skill_score', '')}",
        f"Best selection gate pass: {best.get('selection_gate_pass', '')}",
        "Future-shadow labels used: false",
        "Holdout used for selection: false",
        "No orders, no promotion, no champion mutation.",
    ]
    (out_dir / "rawseq_downside_risk_challenger_tournament.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
