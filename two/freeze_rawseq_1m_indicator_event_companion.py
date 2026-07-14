#!/usr/bin/env python3
"""Freeze a selected indicator-event companion family before July holdout."""

from __future__ import annotations

import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import file_sha256, metric_row, now_stamp, save_reload_prediction_parity, stable_hash, write_csv, write_json  # noqa: E402
from scripts.tiny.run_rawseq_1m_dual_timescale_indicator_scout import feature_matrix  # noqa: E402
from scripts.tiny.run_rawseq_1m_indicator_event_scout import fit_classifier, predict_proba  # noqa: E402

DEFAULT_ROOT = Path(r"F:\rsio\rawseq_1m_indicator_companion_scout")
DEFAULT_DATASET = DEFAULT_ROOT / "indicator_residual_event_dataset_20260712T191440Z"
FROZEN_DOWNSIDE_HASH = "8109480bd6b0c2575337cada9be991746199200d42feaeea1fd7b9dd0b3eb6bf"
PROB_CLIP = [1e-6, 1.0 - 1e-6]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def latest_selection(root: Path) -> Path:
    dirs = sorted(root.glob("indicator_event_family_selection_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise SystemExit("No family selection packet found; run report_rawseq_1m_indicator_event_family_selection.py first")
    return dirs[0]


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def event_indices_for_family(data: dict[str, np.ndarray], event_names: list[str], selected_events: list[str]) -> list[int]:
    lookup = {name: idx for idx, name in enumerate(event_names)}
    return [lookup[name] for name in selected_events if name in lookup]


def monotonic_violation_fraction(probs: np.ndarray) -> float:
    if probs.shape[1] <= 1:
        return 0.0
    return float((np.diff(probs, axis=1) < -1e-12).any(axis=1).mean())


def isotonic_projection(row: np.ndarray) -> np.ndarray:
    """Nondecreasing least-squares projection via simple PAVA."""
    levels = [float(x) for x in row]
    weights = [1.0 for _ in levels]
    i = 0
    while i < len(levels) - 1:
        if levels[i] <= levels[i + 1] + 1e-15:
            i += 1
            continue
        total_w = weights[i] + weights[i + 1]
        avg = (levels[i] * weights[i] + levels[i + 1] * weights[i + 1]) / total_w
        levels[i] = avg
        weights[i] = total_w
        del levels[i + 1]
        del weights[i + 1]
        i = max(0, i - 1)
    expanded: list[float] = []
    for level, weight in zip(levels, weights):
        expanded.extend([level] * int(round(weight)))
    return np.asarray(expanded[: len(row)], dtype=float)


def apply_isotonic(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.vstack([isotonic_projection(row) for row in probs]), PROB_CLIP[0], PROB_CLIP[1])


def cumulative_hazard_reconstruction(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, PROB_CLIP[0], PROB_CLIP[1])
    out = np.zeros_like(probs)
    survival = np.ones(len(probs), dtype=float)
    prev = np.zeros(len(probs), dtype=float)
    for idx in range(probs.shape[1]):
        hazard = np.clip((probs[:, idx] - prev) / np.maximum(survival, 1e-6), 0.0, 1.0)
        out[:, idx] = 1.0 - survival * (1.0 - hazard)
        survival *= 1.0 - hazard
        prev = out[:, idx]
    return np.clip(out, PROB_CLIP[0], PROB_CLIP[1])


def brier_matrix(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((labels - probs) ** 2))


def short_horizon_name(event_name: str) -> str:
    if "_within_" not in event_name:
        return event_name
    return event_name.rsplit("_within_", 1)[-1]


def choose_monotonic_method(raw_probs: np.ndarray, labels: np.ndarray) -> tuple[str, dict[str, Any], np.ndarray]:
    candidates = {
        "raw_independent_probabilities": np.clip(raw_probs, PROB_CLIP[0], PROB_CLIP[1]),
        "isotonic_monotonic_projection": apply_isotonic(raw_probs),
        "cumulative_hazard_reconstruction": cumulative_hazard_reconstruction(raw_probs),
    }
    rows = []
    for name, probs in candidates.items():
        rows.append(
            {
                "method": name,
                "brier": brier_matrix(labels, probs),
                "monotonicity_violation_fraction": monotonic_violation_fraction(probs),
                "mean_abs_correction": float(np.mean(np.abs(probs - raw_probs))),
                "max_abs_correction": float(np.max(np.abs(probs - raw_probs))),
            }
        )
    valid = [row for row in rows if row["monotonicity_violation_fraction"] <= 1e-12]
    selected = min(valid or rows, key=lambda r: (r["monotonicity_violation_fraction"] > 0, r["brier"], r["mean_abs_correction"]))
    return selected["method"], {"methods": rows, "selected": selected}, candidates[selected["method"]]


def main() -> int:
    root = Path(os.getenv("RAWSEQ_EVENT_FREEZE_ROOT", str(DEFAULT_ROOT)))
    selection_dir = Path(os.getenv("RAWSEQ_EVENT_SELECTION_DIR", "") or latest_selection(root))
    dataset_dir = Path(os.getenv("RAWSEQ_EVENT_DATASET_DIR", str(DEFAULT_DATASET)))
    selection = read_json(selection_dir / "selected_indicator_event_family.json")
    selected = selection.get("selected_family") or {}
    if selection.get("recommendation") != "freeze_indicator_event_companion_before_future_holdout" or not selected:
        print("freeze_status=no_indicator_event_family_survives")
        return 0
    if selected.get("model") != "shallow_hgb":
        raise RuntimeError(f"Freeze script currently expects selected model shallow_hgb, got {selected.get('model')}")
    manifest = read_json(dataset_dir / "residual_dataset_manifest.json")
    contracts = {
        "residual_target_contract": read_json(dataset_dir / "residual_target_contract.json"),
        "deterministic_baseline_contract": read_json(dataset_dir / "deterministic_baseline_contract.json"),
    }
    input_contract_path = Path(manifest["source_companion_dataset_dir"]) / "indicator_input_contract.json"
    input_contract = read_json_if_exists(input_contract_path)
    formula_contract_path = Path(str(input_contract.get("static_feature_contract_source", "")))
    formula_contract = read_json_if_exists(formula_contract_path)
    npz = np.load(manifest["dataset_path"], allow_pickle=True)
    data = {key: npz[key] for key in npz.files}
    split = data["split"].astype(str)
    dev_idx = np.where(np.isin(split, ["train", "validation"]))[0]
    event_names = [str(x) for x in data["event_names"]]
    selected_events = str(selected["included_events"]).split("|")
    event_indices = event_indices_for_family(data, event_names, selected_events)
    if len(event_indices) != len(selected_events):
        raise RuntimeError("Selected event names not all present in residual/event dataset")
    feature_x = feature_matrix(data, dev_idx)
    models: dict[str, Any] = {}
    raw_probs = []
    parity_rows = []
    model_rows = []
    run_dir = root / f"frozen_indicator_event_companion_{now_stamp()}"
    artifact_dir = run_dir / "model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for event_idx in event_indices:
        event_name = event_names[event_idx]
        y = data["event_targets"][dev_idx, event_idx].astype(float)
        if len(np.unique(y)) < 2:
            raise RuntimeError(f"Selected event has no class diversity: {event_name}")
        model = fit_classifier("shallow_hgb", feature_x, y)
        model.fit(feature_x, y)
        probs = predict_proba(model, feature_x)
        parity = save_reload_prediction_parity(model, predict_proba, feature_x[: min(500, len(feature_x))])
        models[event_name] = model
        raw_probs.append(probs)
        parity_rows.append({"event_name": event_name, "save_reload_parity": parity[0], "save_reload_max_abs_diff": parity[1]})
        model_rows.append({"event_name": event_name, "model_family": "shallow_hgb", "event_index": event_idx, "training_rows": int(len(y)), "event_prevalence": float(np.mean(y))})
    raw = np.vstack(raw_probs).T
    labels = data["event_targets"][dev_idx][:, event_indices].astype(float)
    method, monotonic_contract, corrected = choose_monotonic_method(raw, labels)
    scout_dir = Path(selection["source_scout_dir"])
    survival = pd.read_csv(scout_dir / "event_target_survival.csv")
    fold_metrics = pd.read_csv(scout_dir / "event_fold_metrics.csv")
    fold_manifest = pd.read_csv(scout_dir / "fold_manifest.csv")
    selected_survival = survival[
        (survival["model"].astype(str) == "shallow_hgb")
        & (survival["event_name"].astype(str).isin(selected_events))
    ].copy()
    selected_fold_metrics = fold_metrics[
        (fold_metrics["model"].astype(str) == "shallow_hgb")
        & (fold_metrics["event_name"].astype(str).isin(selected_events))
    ].copy()
    symbol_values = data["symbol"][dev_idx].astype(str)
    per_symbol_rows: list[dict[str, Any]] = []
    for col_idx, event_name in enumerate(selected_events):
        for symbol in sorted(set(symbol_values)):
            mask = symbol_values == symbol
            y_symbol = labels[mask, col_idx]
            raw_symbol = raw[mask, col_idx]
            corrected_symbol = corrected[mask, col_idx]
            baseline = np.full(len(y_symbol), float(np.mean(y_symbol)) if len(y_symbol) else math.nan)
            corrected_metrics = metric_row(y_symbol, corrected_symbol, baseline)
            raw_metrics = metric_row(y_symbol, raw_symbol, baseline)
            per_symbol_rows.append(
                {
                    "event_name": event_name,
                    "horizon": short_horizon_name(event_name),
                    "symbol": symbol,
                    "rows": corrected_metrics["rows"],
                    "events": corrected_metrics["events"],
                    "event_prevalence": corrected_metrics["event_prevalence"],
                    "corrected_brier_skill_vs_prevalence": corrected_metrics["brier_skill_vs_prevalence"],
                    "corrected_log_loss_improvement_vs_prevalence": corrected_metrics["log_loss_improvement_vs_prevalence"],
                    "corrected_pr_auc_lift_over_event_prevalence": corrected_metrics["pr_auc_lift_over_event_prevalence"],
                    "corrected_roc_auc": corrected_metrics["roc_auc"],
                    "corrected_calibration_slope": corrected_metrics["calibration_slope"],
                    "corrected_calibration_intercept": corrected_metrics["calibration_intercept"],
                    "raw_brier_skill_vs_prevalence": raw_metrics["brier_skill_vs_prevalence"],
                    "raw_log_loss_improvement_vs_prevalence": raw_metrics["log_loss_improvement_vs_prevalence"],
                    "raw_pr_auc_lift_over_event_prevalence": raw_metrics["pr_auc_lift_over_event_prevalence"],
                    "raw_roc_auc": raw_metrics["roc_auc"],
                }
            )
    probability_rows: list[dict[str, Any]] = []
    for row_pos, src_idx in enumerate(dev_idx):
        row: dict[str, Any] = {
            "timestamp_ms": int(data["timestamp_ms"][src_idx]),
            "source_row_index": int(data["source_row_index"][src_idx]),
            "split": str(data["split"][src_idx]),
            "symbol": str(data["symbol"][src_idx]),
        }
        for col_idx, event_name in enumerate(selected_events):
            suffix = short_horizon_name(event_name)
            row[f"{suffix}_event_name"] = event_name
            row[f"{suffix}_label"] = float(labels[row_pos, col_idx])
            row[f"{suffix}_raw_probability"] = float(raw[row_pos, col_idx])
            row[f"{suffix}_corrected_probability"] = float(corrected[row_pos, col_idx])
        probability_rows.append(row)
    model_path = artifact_dir / "indicator_event_family_models.pkl"
    model_payload = {
        "models": models,
        "selected_events": selected_events,
        "event_indices": event_indices,
        "monotonic_correction_method": method,
        "probability_clip": PROB_CLIP,
    }
    model_path.write_bytes(pickle.dumps(model_payload))
    selected_horizon_metrics_path = run_dir / "selected_event_horizon_metrics.csv"
    selected_fold_metrics_path = run_dir / "selected_event_fold_metrics.csv"
    fold_manifest_path = run_dir / "source_fold_manifest.csv"
    per_symbol_metrics_path = run_dir / "selected_event_per_symbol_development_metrics.csv"
    probability_path = run_dir / "selected_event_development_probabilities.csv"
    write_csv(selected_horizon_metrics_path, selected_survival.to_dict("records"))
    write_csv(selected_fold_metrics_path, selected_fold_metrics.to_dict("records"))
    write_csv(fold_manifest_path, fold_manifest.to_dict("records"))
    write_csv(per_symbol_metrics_path, per_symbol_rows)
    write_csv(probability_path, probability_rows)
    contract = {
        "status": "frozen_indicator_event_companion_waiting_for_july_holdout",
        "created_at": now_stamp(),
        "family": selected["family"],
        "model_family": "shallow_hgb",
        "event_definitions": selected_events,
        "horizons_minutes": [int(x) for x in selected["included_horizons"]],
        "feature_order_source": str(dataset_dir / "residual_target_contract.json"),
        "input_contract_path": str(input_contract_path),
        "input_contract_sha256": file_sha256(input_contract_path) if input_contract_path.exists() else "",
        "static_feature_contract_source": input_contract.get("static_feature_contract_source", ""),
        "static_feature_contract_sha256": file_sha256(formula_contract_path) if formula_contract_path.exists() else "",
        "static_feature_order": input_contract.get("static_feature_order", []),
        "temporal_channel_order": input_contract.get("temporal_channel_order", []),
        "feature_formulas": formula_contract.get("feature_formulas", {}),
        "formula_source": formula_contract.get("formula_source", {}),
        "input_shapes": {
            "x_static": [31],
            "x_short": [60, 12],
            "x_long": [60, 12],
        },
        "hgb_hyperparameters": {
            "max_iter": 40,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "l2_regularization": 0.01,
            "random_state": 1337,
        },
        "monotonic_correction_contract": monotonic_contract,
        "selected_monotonic_correction_method": method,
        "probability_clipping": PROB_CLIP,
        "source_dataset_dir": str(dataset_dir),
        "source_dataset_sha256": manifest["dataset_sha256"],
        "source_scout_dir": selection["source_scout_dir"],
        "selection_dir": str(selection_dir),
        "fold_manifest_path": str(fold_manifest_path),
        "selected_event_horizon_metrics_path": str(selected_horizon_metrics_path),
        "selected_event_fold_metrics_path": str(selected_fold_metrics_path),
        "selected_event_per_symbol_development_metrics_path": str(per_symbol_metrics_path),
        "selected_event_development_probabilities_path": str(probability_path),
        "model_artifact_path": str(model_path),
        "model_artifact_sha256": file_sha256(model_path),
        "freeze_artifact_sha256": {
            "source_fold_manifest": file_sha256(fold_manifest_path),
            "selected_event_horizon_metrics": file_sha256(selected_horizon_metrics_path),
            "selected_event_fold_metrics": file_sha256(selected_fold_metrics_path),
            "selected_event_per_symbol_development_metrics": file_sha256(per_symbol_metrics_path),
            "selected_event_development_probabilities": file_sha256(probability_path),
        },
        "branch": git_value(["branch", "--show-current"]),
        "commit": git_value(["rev-parse", "HEAD"]),
        "dirty_state": bool(git_value(["status", "--short"])),
        "frozen_downside_candidate_hash": FROZEN_DOWNSIDE_HASH,
        "paper_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "downside_candidate_mutation": False,
        "june_event_evaluation": False,
        "july_access": False,
    }
    contract["companion_contract_hash"] = stable_hash({k: v for k, v in contract.items() if k != "companion_contract_hash"})
    gui_payload = {
        "ema_spread_narrowing_probability": {f"{h}m": 0 for h in contract["horizons_minutes"]},
        "probability_monotonicity_corrected": method != "raw_independent_probabilities",
        "event_companion_contract_hash": contract["companion_contract_hash"],
        "frozen_downside_candidate_hash": FROZEN_DOWNSIDE_HASH,
    }
    write_json(run_dir / "indicator_event_companion_contract.json", contract)
    write_json(run_dir / "indicator_event_companion_gui_schema.json", gui_payload)
    write_json(run_dir / "monotonic_correction_contract.json", monotonic_contract)
    write_csv(run_dir / "model_artifacts.csv", model_rows)
    write_csv(run_dir / "save_reload_parity.csv", parity_rows)
    write_csv(run_dir / "monotonic_development_predictions_audit.csv", [
        {
            "raw_violation_fraction": monotonic_violation_fraction(raw),
            "corrected_violation_fraction": monotonic_violation_fraction(corrected),
            "selected_method": method,
            "mean_abs_correction": float(np.mean(np.abs(corrected - raw))),
            "max_abs_correction": float(np.max(np.abs(corrected - raw))),
        }
    ])
    report = [
        "Rawseq 1m indicator-event companion freeze",
        f"status={contract['status']}",
        f"family={contract['family']}",
        f"model_family={contract['model_family']}",
        f"horizons={','.join(str(x) for x in contract['horizons_minutes'])}",
        f"selected_monotonic_correction_method={method}",
        f"companion_contract_hash={contract['companion_contract_hash']}",
        f"model_artifact_sha256={contract['model_artifact_sha256']}",
        "june_event_evaluation=false",
        "july_access=false",
    ]
    (run_dir / "indicator_event_companion_freeze_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"freeze_dir={run_dir}")
    print(f"status={contract['status']}")
    print(f"family={contract['family']}")
    print(f"model_family={contract['model_family']}")
    print(f"horizons={','.join(str(x) for x in contract['horizons_minutes'])}")
    print(f"selected_monotonic_correction_method={method}")
    print(f"companion_contract_hash={contract['companion_contract_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
