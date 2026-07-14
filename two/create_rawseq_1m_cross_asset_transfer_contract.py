#!/usr/bin/env python3
"""Freeze the fixed SOL one-minute contract for cross-asset transfer tests.

This script is report-only. It reads the completed SOL development artifacts,
checks that the same contract is selected by the scout and calibration audit,
and writes a fixed transfer contract that downstream cross-asset reports must
use without per-symbol tuning.
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny import rawseq_1m_baseline_utils as baseline_utils
from scripts.tiny.rawseq_1m_baseline_utils import SAFETY_FLAGS, env_path, file_sha256, now_stamp, stable_hash, write_json

DEFAULT_SCOUT_DIR = Path(r"F:\rsio\rawseq_1m_baseline_scout\rawseq_1m_baseline_scout_20260712T044244Z")
DEFAULT_CALIBRATION_DIR = Path(r"F:\rsio\rawseq_1m_baseline_scout\rawseq_1m_calibration_audit_20260712T053558Z")
DEFAULT_FREEZE_GATE_DIR = Path(r"F:\rsio\rawseq_1m_baseline_scout\rawseq_1m_cpu_challenger_freeze_gate_20260712T053659Z")
DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_cross_asset_scout")

SELECTED_SCOUT_MODEL = "hist_gradient_boosting_shallow"
SELECTED_CALIBRATION_MODEL = "raw_hist_gradient_boosting_shallow"
SELECTED_HORIZON_MINUTES = 1
SELECTED_VOL_WINDOW_MINUTES = 240
THRESHOLD_VOL_UNITS = 0.5


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required SOL artifact missing: {path}")
    return path


def selected_scout_row(scout_dir: Path) -> dict[str, str]:
    rows = read_csv_rows(require_file(scout_dir / "cpu_downside_risk_leaderboard.csv"))
    matches = [
        row
        for row in rows
        if int(float(row.get("horizon_minutes", -1))) == SELECTED_HORIZON_MINUTES
        and int(float(row.get("vol_window_minutes", -1))) == SELECTED_VOL_WINDOW_MINUTES
        and row.get("model") == SELECTED_SCOUT_MODEL
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one selected SOL scout row, found {len(matches)}")
    row = matches[0]
    if not truthy(row.get("advance_gate_pass")):
        raise RuntimeError("Selected SOL scout row does not pass the development advance gate")
    return row


def selected_calibration_row(calibration_dir: Path) -> dict[str, str]:
    rows = read_csv_rows(require_file(calibration_dir / "calibration_leaderboard.csv"))
    selected = [row for row in rows if truthy(row.get("selected_calibration_contract"))]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one selected calibration row, found {len(selected)}")
    row = selected[0]
    if row.get("model") != SELECTED_CALIBRATION_MODEL:
        raise RuntimeError(f"Unexpected selected calibration model: {row.get('model')}")
    return row


def feature_formula_for(name: str) -> str:
    if name == "timestamp_ms":
        return "Binance open_time normalized to milliseconds."
    if name == "timestamp":
        return "UTC timestamp derived from timestamp_ms."
    formulas = {
        "signed_bucket_return_bps": "10000 * log(close / close.shift(1))",
        "candle_range_bps": "10000 * log(high / low)",
        "candle_body_bps": "10000 * log(close / open)",
        "upper_wick_bps": "10000 * log(high / max(open, close))",
        "lower_wick_bps": "10000 * log(min(open, close) / low)",
        "log_volume_change": "log(volume / volume.shift(1))",
    }
    if name in formulas:
        return formulas[name]
    if name.startswith("rolling_range_bps_fw"):
        window = name.rsplit("fw", 1)[-1]
        return f"10000 * log(rolling_max(high,{window}) / rolling_min(low,{window})); min_periods={window}"
    if name.startswith("rolling_volatility_bps_fw"):
        window = name.rsplit("fw", 1)[-1]
        return f"std(signed_bucket_return_bps,{window},ddof=0); min_periods={window}"
    if name.startswith("distance_to_recent_high_bps_fw"):
        window = name.rsplit("fw", 1)[-1]
        return f"10000 * log(close / rolling_max(high,{window})); min_periods={window}"
    if name.startswith("distance_to_recent_low_bps_fw"):
        window = name.rsplit("fw", 1)[-1]
        return f"10000 * log(close / rolling_min(low,{window})); min_periods={window}"
    if name.startswith("close_to_ema_bps_fw"):
        window = name.rsplit("fw", 1)[-1]
        return f"10000 * log(close / ema(close,span={window},adjust=False)); min_periods={window}"
    if name.startswith("ema_slope_bps_fw"):
        window = name.rsplit("fw", 1)[-1]
        return f"10000 * log(ema(close,span={window}) / ema(close,span={window}).shift({window}))"
    if name.startswith("trailing_volatility_bps_fw"):
        window = name.rsplit("fw", 1)[-1]
        return f"std(10000 * log(close / close.shift(1)),{window},ddof=0); min_periods={window}"
    return "unresolved_formula"


def source_function_hashes() -> dict[str, str]:
    functions = {
        "build_features": baseline_utils.build_features,
        "future_low_return_bps": baseline_utils.future_low_return_bps,
        "downside_event_targets": baseline_utils.downside_event_targets,
        "split_contract": baseline_utils.split_contract,
        "metric_row": baseline_utils.metric_row,
    }
    return {name: stable_hash(inspect.getsource(func)) for name, func in functions.items()}


def build_contract(
    scout_dir: Path,
    calibration_dir: Path,
    freeze_gate_dir: Path,
    inventory_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scout_row = selected_scout_row(scout_dir)
    calibration_row = selected_calibration_row(calibration_dir)
    calibration_contract = read_json(require_file(calibration_dir / "calibration_audit_contract.json"))
    baseline_comparison = read_json(require_file(scout_dir / "baseline_comparison.json"))
    feature_contract = read_json(require_file(scout_dir / "feature_contract.json"))
    split_manifest = read_json(require_file(scout_dir / "split_manifest.json"))
    leakage_audit = read_json(require_file(scout_dir / "leakage_audit.json"))
    holdout_lock = read_json(require_file(scout_dir / "holdout_lock.json"))
    source_manifest = read_json(require_file(scout_dir / "source_manifest.json"))
    freeze_decision = read_json(require_file(freeze_gate_dir / "rawseq_1m_cpu_challenger_freeze_decision.json"))
    purge_rows = read_csv_rows(require_file(scout_dir / "purge_embargo_audit.csv"))
    fold_manifest = read_csv_rows(require_file(scout_dir / "rolling_fold_manifest.csv"))

    if calibration_contract.get("selected_calibration_contract") != SELECTED_CALIBRATION_MODEL:
        raise RuntimeError("Calibration contract and leaderboard disagree on selected calibration model")
    top = baseline_comparison.get("top_cpu_contract", {})
    if top.get("model") != SELECTED_SCOUT_MODEL or int(top.get("horizon_minutes", -1)) != SELECTED_HORIZON_MINUTES:
        raise RuntimeError("Baseline comparison top contract does not match the expected SOL transfer contract")
    if truthy(freeze_decision.get("freeze_created")):
        raise RuntimeError("This transfer freezer must not consume a created frozen candidate")

    base_feature_names = list(feature_contract.get("features", []))
    if not base_feature_names or base_feature_names[:2] != ["timestamp_ms", "timestamp"]:
        raise RuntimeError("Feature contract is missing canonical timestamp columns")
    model_feature_names = [name for name in base_feature_names if name not in {"timestamp_ms", "timestamp"}]
    vol_feature_name = f"trailing_volatility_bps_fw{SELECTED_VOL_WINDOW_MINUTES}"
    if vol_feature_name not in model_feature_names:
        model_feature_names.append(vol_feature_name)

    artifact_paths = {
        "scout_cpu_leaderboard": scout_dir / "cpu_downside_risk_leaderboard.csv",
        "scout_fold_metrics": scout_dir / "cpu_downside_risk_fold_metrics.csv",
        "feature_contract": scout_dir / "feature_contract.json",
        "split_manifest": scout_dir / "split_manifest.json",
        "purge_embargo_audit": scout_dir / "purge_embargo_audit.csv",
        "calibration_contract": calibration_dir / "calibration_audit_contract.json",
        "calibration_leaderboard": calibration_dir / "calibration_leaderboard.csv",
        "freeze_gate_decision": freeze_gate_dir / "rawseq_1m_cpu_challenger_freeze_decision.json",
    }
    if inventory_dir:
        artifact_paths["multisymbol_inventory"] = inventory_dir / "multisymbol_inventory.csv"

    contract: dict[str, Any] = {
        "created_at": now_stamp(),
        "contract_kind": "fixed_sol_1m_downside_risk_cross_asset_transfer",
        "source_symbol": "SOLUSDT",
        "source_venue": "binance_public",
        "cadence_seconds": 60,
        "target_name": "downside_event_0p5vol",
        "target_formula": "1 if max(0, -future_low_return_bps_h1m) > 0.5 * trailing_volatility_bps_fw240 else 0",
        "future_low_formula": "future low uses future lows from steps 1..horizon only; label construction is causal for training rows but unavailable at decision time",
        "target_horizon_minutes": SELECTED_HORIZON_MINUTES,
        "threshold_vol_units": THRESHOLD_VOL_UNITS,
        "volatility_window_minutes": SELECTED_VOL_WINDOW_MINUTES,
        "volatility_denominator": vol_feature_name,
        "feature_windows_minutes": feature_contract.get("feature_windows", []),
        "materialized_feature_names_and_order": base_feature_names,
        "model_feature_names_and_order": model_feature_names,
        "feature_formulas": {name: feature_formula_for(name) for name in model_feature_names},
        "formula_source": {
            "module": "scripts.tiny.rawseq_1m_baseline_utils",
            "function_hashes": source_function_hashes(),
            "leakage_audit": leakage_audit,
        },
        "model_family": "hist_gradient_boosting_shallow",
        "model_library": "sklearn.ensemble.HistGradientBoostingClassifier",
        "model_pipeline": [
            {"step": "SimpleImputer", "strategy": "median"},
            {
                "step": "HistGradientBoostingClassifier",
                "max_iter": 80,
                "max_leaf_nodes": 15,
                "learning_rate": 0.05,
                "l2_regularization": 0.01,
                "random_state": 1337,
            },
        ],
        "missing_value_policy": {
            "row_filter_before_fit": "drop rows where target or any selected model input is nonfinite",
            "model_input_imputation": "SimpleImputer(strategy='median') fit on training fold only",
        },
        "scaling_policy": "none_for_hgb_pipeline",
        "calibration_policy": {
            "selected_calibration": "none",
            "selected_calibration_contract": SELECTED_CALIBRATION_MODEL,
            "calibration_source": "raw HGB probabilities; no Platt/isotonic transform selected",
        },
        "chronological_split_rules": {
            "split_manifest": split_manifest,
            "rolling_fold_manifest": fold_manifest,
            "purge_embargo_audit": purge_rows,
        },
        "purge_rows": int(split_manifest.get("purge_rows", 0)),
        "embargo_rows": int(split_manifest.get("embargo_rows", 0)),
        "training_row_cap_per_fold": 100000,
        "validation_row_cap_per_fold": 100000,
        "cap_policy": {"train": "tail", "validation": "head"},
        "selection_stage": "development_folds_only",
        "holdout_policy": {
            "holdout_accessed_for_selection": False,
            "do_not_reopen_label_inspected_sol_holdout_as_untouched": True,
            "holdout_lock": holdout_lock,
            "freeze_gate_recommendation": freeze_decision.get("final_recommendation"),
            "freeze_gate_block_reasons": freeze_decision.get("freeze_block_reasons", []),
        },
        "selected_scout_metrics": scout_row,
        "selected_calibration_metrics": calibration_row,
        "acceptance_metric_sources": {
            "development_scout": str(scout_dir),
            "calibration_audit": str(calibration_dir),
            "freeze_gate": str(freeze_gate_dir),
        },
        "source_artifacts": {key: str(path) for key, path in artifact_paths.items()},
        "source_artifact_sha256": {key: file_sha256(path) for key, path in artifact_paths.items() if path.exists()},
        "source_manifest_sha256": source_manifest.get("source_manifest_sha256", ""),
        "safety": {
            **SAFETY_FLAGS,
            "public_recorded_data_only": True,
            "active_future_shadow_mutation": False,
            "active_future_shadow_labels_used": False,
            "frozen_kraken_candidate_reused": False,
            "future_data_accessed": False,
        },
    }
    contract["fixed_transfer_contract_hash"] = stable_hash(contract)

    acceptance_rules = {
        "created_at": contract["created_at"],
        "applies_to_contract_hash": contract["fixed_transfer_contract_hash"],
        "classification_rules": {
            "strong_transfer": {
                "fold_wins_min": 3,
                "folds_expected": 4,
                "median_brier_skill_gt": 0.0,
                "worst_fold_brier_skill_gt": 0.0,
                "median_pr_auc_lift_gt": 0.0,
                "save_reload_parity_required": True,
            },
            "partial_transfer": {
                "median_brier_skill_gt": 0.0,
                "stability_or_worst_fold_may_fail": True,
            },
            "no_transfer": {
                "median_brier_skill_lte": 0.0,
                "or_most_folds_fail_prevalence": True,
            },
        },
        "do_not_tune_per_symbol": [
            "horizon_minutes",
            "vol_window_minutes",
            "feature_names_and_order",
            "feature_windows",
            "model_hyperparameters",
            "calibration_policy",
            "chronological_split_rule",
            "purge_rows",
            "embargo_rows",
        ],
        "safety": contract["safety"],
    }
    acceptance_rules["transfer_acceptance_rules_hash"] = stable_hash(acceptance_rules)
    return contract, acceptance_rules


def main() -> int:
    scout_dir = env_path("RAWSEQ_1M_SOL_SCOUT_DIR", DEFAULT_SCOUT_DIR)
    calibration_dir = env_path("RAWSEQ_1M_SOL_CALIBRATION_DIR", DEFAULT_CALIBRATION_DIR)
    freeze_gate_dir = env_path("RAWSEQ_1M_SOL_FREEZE_GATE_DIR", DEFAULT_FREEZE_GATE_DIR)
    output_root = env_path("RAWSEQ_1M_CROSS_ASSET_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    inventory_raw = os.getenv("RAWSEQ_1M_MULTISYMBOL_INVENTORY_DIR", "").strip()
    inventory_dir = Path(inventory_raw) if inventory_raw else None
    out_dir = Path(os.getenv("RAWSEQ_1M_TRANSFER_CONTRACT_OUTPUT_DIR", "").strip() or output_root / f"rawseq_1m_fixed_transfer_contract_{now_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=False)

    contract, acceptance_rules = build_contract(scout_dir, calibration_dir, freeze_gate_dir, inventory_dir)
    write_json(out_dir / "fixed_transfer_contract.json", contract)
    (out_dir / "fixed_transfer_contract_hash.txt").write_text(contract["fixed_transfer_contract_hash"] + "\n", encoding="utf-8")
    write_json(out_dir / "transfer_acceptance_rules.json", acceptance_rules)
    lines = [
        "Rawseq 1m fixed SOL transfer contract",
        f"Output: {out_dir}",
        f"Contract hash: {contract['fixed_transfer_contract_hash']}",
        f"Acceptance rules hash: {acceptance_rules['transfer_acceptance_rules_hash']}",
        f"Selected model: {contract['model_family']}",
        f"Target: horizon={contract['target_horizon_minutes']}m threshold={contract['threshold_vol_units']} vol_window={contract['volatility_window_minutes']}m",
        f"Model feature count: {len(contract['model_feature_names_and_order'])}",
        "Status: fixed contract only; no model evaluation, no tuning, no promotion, no orders.",
    ]
    (out_dir / "fixed_transfer_contract.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
