#!/usr/bin/env python3
"""Contract parity report for frozen downside-risk future shadow.

Verifies that the frozen candidate files, model payload, scaler payload, policy
thresholds, acceptance rule, and current feature table agree before prospective
paper-shadow evidence is interpreted.

No training, no recalibration, no threshold changes, no orders, no private API,
no champion mutation, no promotion.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.freeze_rawseq_low_path_ridge_research_candidate import file_sha256, stable_hash
from scripts.tiny.run_rawseq_downside_risk_future_paper_shadow import DEFAULT_CANDIDATE_DIR, DEFAULT_FEATURE_TABLE

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_shadow_contract_parity"
EXPECTED_TARGET_COLUMNS = [
    "future_range_low_bps_h60",
    "future_range_low_bps_h120",
    "future_range_low_bps_h240",
    "future_range_low_bps_h480",
]


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return resolve_path(raw) if raw else default


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


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def check_row(name: str, passed: bool, expected: Any, actual: Any, severity: str = "error") -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(passed),
        "severity": severity,
        "expected": expected,
        "actual": actual,
    }


def infer_selected_features(feature_columns: list[str], selected_indices: np.ndarray) -> list[str]:
    return [feature_columns[int(idx)] for idx in selected_indices]


def main() -> int:
    candidate_dir = env_path("RAWSEQ_DOWNSIDE_SHADOW_CANDIDATE_DIR", DEFAULT_CANDIDATE_DIR)
    feature_table = env_path("RAWSEQ_DOWNSIDE_SHADOW_FEATURE_TABLE", DEFAULT_FEATURE_TABLE)
    output_root = env_path("RAWSEQ_DOWNSIDE_PARITY_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / f"rawseq_downside_risk_shadow_contract_parity_{now_stamp()}"

    contract_path = candidate_dir / "rawseq_downside_risk_cpu_candidate_contract.json"
    policy_path = candidate_dir / "rawseq_downside_risk_policy_contract.json"
    rule_path = candidate_dir / "rawseq_downside_risk_future_acceptance_rule.json"
    contract = read_json(contract_path)
    policy = read_json(policy_path)
    rule = read_json(rule_path)
    model_path = resolve_path(contract["model_path"])
    scalers_path = resolve_path(contract["scalers_path"])
    source_npz = resolve_path(contract["source_npz"])

    model_hash = file_sha256(model_path)
    scalers_hash = file_sha256(scalers_path)
    source_npz_hash = file_sha256(source_npz) if source_npz.exists() else ""
    with np.load(model_path, allow_pickle=False) as model:
        coef_shape = tuple(int(x) for x in model["coef"].shape)
        model_selected = str(model["selected_model"])
        model_calibration = str(model["selected_calibration"])
        model_features = [str(x) for x in model["feature_columns"]]
        model_targets = [str(x) for x in model["target_columns"]]
        selected_indices = model["selected_feature_indices"].astype(np.int64)
        selected_features = infer_selected_features(model_features, selected_indices)
        model_threshold = float(model["target_threshold_vol_units"][0])
    with np.load(scalers_path, allow_pickle=False) as scalers:
        scaler_mean_shape = tuple(int(x) for x in scalers["feature_scaler_mean"].shape)
        scaler_std_shape = tuple(int(x) for x in scalers["feature_scaler_std"].shape)

    table_header = pd.read_csv(feature_table, nrows=0).columns.tolist()
    missing_selected_features = [feature for feature in selected_features if feature not in table_header]
    target_columns_materialized = [target for target in model_targets if target in table_header]
    required_derivation_columns = ["price", "close", "realized_volatility_bps_fw60"]
    missing_derivation_columns = [column for column in required_derivation_columns if column not in table_header]

    checks = [
        check_row("model_hash_matches_contract", model_hash == contract.get("model_sha256"), contract.get("model_sha256"), model_hash),
        check_row("scalers_hash_matches_contract", scalers_hash == contract.get("scalers_sha256"), contract.get("scalers_sha256"), scalers_hash),
        check_row("source_npz_hash_matches_contract", source_npz_hash == contract.get("source_npz_sha256"), contract.get("source_npz_sha256"), source_npz_hash),
        check_row("selected_model_is_frozen_logistic_regression", model_selected == "logistic_regression" == contract.get("selected_model"), "logistic_regression", model_selected),
        check_row("selected_calibration_is_none", model_calibration == "none" == contract.get("selected_calibration_method"), "none", model_calibration),
        check_row("target_name_matches_contract", contract.get("target_name") == "downside_exceedance_probability_gt_0_5vol", "downside_exceedance_probability_gt_0_5vol", contract.get("target_name")),
        check_row("target_threshold_is_0_5_vol_units", abs(model_threshold - 0.5) < 1e-12 and abs(float(contract.get("target_threshold_vol_units")) - 0.5) < 1e-12, 0.5, model_threshold),
        check_row("target_columns_match_expected_order", model_targets == EXPECTED_TARGET_COLUMNS, ";".join(EXPECTED_TARGET_COLUMNS), ";".join(model_targets)),
        check_row("coef_shape_matches_selected_features_and_targets", coef_shape == (len(selected_features) + 1, len(model_targets)), (len(selected_features) + 1, len(model_targets)), coef_shape),
        check_row("scaler_shapes_match_selected_features", scaler_mean_shape == scaler_std_shape == (len(selected_features),), (len(selected_features),), {"mean": scaler_mean_shape, "std": scaler_std_shape}),
        check_row("feature_table_has_all_selected_features", not missing_selected_features, "no missing selected features", ";".join(missing_selected_features)),
        check_row("feature_table_can_materialize_targets", not missing_derivation_columns, "price/close/realized_volatility available", ";".join(missing_derivation_columns)),
        check_row("primary_threshold_is_0_80", abs(safe_float(policy.get("primary_threshold", {}).get("threshold")) - 0.8) < 1e-12, 0.8, policy.get("primary_threshold", {}).get("threshold")),
        check_row("backup_threshold_is_0_70", abs(safe_float(policy.get("backup_threshold", {}).get("threshold")) - 0.7) < 1e-12, 0.7, policy.get("backup_threshold", {}).get("threshold")),
        check_row("acceptance_rule_hash_matches_contract", rule.get("acceptance_rule_sha256") == contract.get("contract_sha256") or rule.get("contract_sha256") == contract.get("contract_sha256"), contract.get("contract_sha256"), rule.get("contract_sha256"), "warning"),
        check_row("paper_only_flags", contract.get("paper_only") is True and contract.get("orders") is False and contract.get("promotion") is False and contract.get("champion_mutation") is False, "paper_only true/orders false/promotion false/champion false", {k: contract.get(k) for k in ["paper_only", "orders", "promotion", "champion_mutation"]}),
    ]
    error_failures = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warning_failures = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    status = "PASS" if not error_failures else "FAIL"
    summary = {
        "generated_at_iso": datetime.now(UTC).isoformat(),
        "candidate_dir": str(candidate_dir),
        "feature_table": str(feature_table),
        "status": status,
        "error_failures": len(error_failures),
        "warning_failures": len(warning_failures),
        "contract_sha256": contract.get("contract_sha256"),
        "model_sha256": model_hash,
        "scalers_sha256": scalers_hash,
        "source_npz_sha256": source_npz_hash,
        "selected_feature_count": len(selected_features),
        "target_columns": model_targets,
        "target_columns_materialized_in_feature_table": target_columns_materialized,
        "target_label_source": "materialized_columns" if len(target_columns_materialized) == len(model_targets) else "derived_from_price_path",
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    summary["parity_report_sha256"] = stable_hash(summary)
    write_json(out_dir / "rawseq_downside_risk_shadow_contract_parity.json", summary)
    write_csv(out_dir / "rawseq_downside_risk_shadow_contract_parity_checks.csv", checks)
    lines = [
        "Rawseq downside-risk shadow contract parity",
        f"Output: {out_dir}",
        f"Status: {status}",
        f"Candidate: {candidate_dir}",
        f"Feature table: {feature_table}",
        f"Selected features: {len(selected_features)}",
        f"Target columns: {';'.join(model_targets)}",
        f"Target label source: {summary['target_label_source']}",
        f"Error failures: {len(error_failures)}",
        f"Warning failures: {len(warning_failures)}",
        "",
        "Failures:",
    ]
    failures = error_failures + warning_failures
    lines.extend(f"- {row['severity']} {row['check']}: expected={row['expected']} actual={row['actual']}" for row in failures)
    if not failures:
        lines.append("- none")
    lines.append("Safety: report-only, paper_only=true, orders=false, promotion=false, champion_mutation=false.")
    (out_dir / "rawseq_downside_risk_shadow_contract_parity.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
