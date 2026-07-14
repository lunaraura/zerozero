#!/usr/bin/env python3
"""Conditional freeze gate for the one-minute CPU challenger.

This script refuses to freeze if holdout integrity is not pristine and no new
unseen interval plan exists. It never evaluates holdout model performance.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import DEFAULT_OUTPUT_ROOT, SAFETY_FLAGS, env_path, now_stamp, stable_hash, write_csv, write_json


def latest_dir(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    candidates = [p for p in root.glob(pattern) if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists() or path.stat().st_size <= 2:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    out_root = env_path("RAWSEQ_1M_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    holdout_dir = Path(os.getenv("RAWSEQ_1M_HOLDOUT_AUDIT_DIR", "").strip() or latest_dir(out_root, "rawseq_1m_holdout_integrity_audit_*") or "")
    calibration_dir = Path(os.getenv("RAWSEQ_1M_CALIBRATION_AUDIT_DIR", "").strip() or latest_dir(out_root, "rawseq_1m_calibration_audit_*") or "")
    out_dir = Path(os.getenv("RAWSEQ_1M_FREEZE_OUTPUT_DIR", "").strip() or out_root / f"rawseq_1m_cpu_challenger_freeze_gate_{now_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=False)
    holdout = read_json(holdout_dir / "holdout_integrity_audit.json")
    calibration = read_json(calibration_dir / "calibration_audit_contract.json")
    calibration_rows = read_csv_rows(calibration_dir / "calibration_leaderboard.csv")
    selected = next((row for row in calibration_rows if str(row.get("selected_calibration_contract", "")).lower() in {"true", "1"}), {})
    holdout_class = holdout.get("holdout_integrity_classification", "holdout_integrity_unknown")
    new_plan = "create_new_untouched_interval_after_" in str(holdout.get("recommendation", ""))
    leakage_pass = True
    parity_pass = str(selected.get("save_reload_parity_all", "")).lower() in {"true", "1"}
    all_folds_win = int(float(selected.get("fold_wins", 0) or 0)) >= int(float(selected.get("folds", 999) or 999))
    worst_positive = float(selected.get("worst_fold_brier_skill", "-1") or -1) > 0
    freeze_valid = (
        holdout_class == "pristine_holdout"
        and leakage_pass
        and parity_pass
        and all_folds_win
        and worst_positive
        and bool(selected)
    )
    if freeze_valid:
        final_recommendation = "freeze_one_minute_cpu_challenger_before_holdout"
    elif holdout_class != "pristine_holdout" and new_plan:
        final_recommendation = "create_new_unseen_one_minute_holdout_then_freeze"
    elif selected and worst_positive:
        final_recommendation = "continue_one_minute_calibration_research"
    else:
        final_recommendation = "reject_current_one_minute_lane"
    packet = {
        "created_at": now_stamp(),
        "freeze_created": False,
        "final_recommendation": final_recommendation,
        "holdout_integrity_classification": holdout_class,
        "holdout_recommendation": holdout.get("recommendation", ""),
        "selected_calibration_contract": calibration.get("selected_calibration_contract", ""),
        "candidate_contract_hash": "",
        "holdout_acceptance_rule_hash": "",
        "freeze_block_reasons": [],
        "holdout_model_scoring": False,
        "holdout_accessed": False,
        **SAFETY_FLAGS,
    }
    if not freeze_valid:
        if holdout_class != "pristine_holdout":
            packet["freeze_block_reasons"].append(f"holdout is {holdout_class}, not pristine")
        if not parity_pass:
            packet["freeze_block_reasons"].append("save/reload parity did not pass for selected calibration contract")
        if not all_folds_win:
            packet["freeze_block_reasons"].append("selected calibration contract did not win all folds")
        if not worst_positive:
            packet["freeze_block_reasons"].append("selected calibration contract worst-fold Brier skill is not positive")
    else:
        # The current artifact path should not reach this branch for the supplied
        # scout because the holdout was label-inspected by yearly prevalence.
        contract_payload = {
            "horizon_minutes": 1,
            "vol_window_minutes": 240,
            "model_family": "hist_gradient_boosting_shallow",
            "calibration_contract": calibration.get("selected_calibration_contract", ""),
            "holdout_integrity_classification": holdout_class,
            "holdout_accessed": False,
            **SAFETY_FLAGS,
        }
        packet["freeze_created"] = True
        packet["candidate_contract_hash"] = stable_hash(contract_payload)
        packet["holdout_acceptance_rule_hash"] = stable_hash(
            {
                "brier_skill_gt_0": True,
                "log_loss_improvement_gt_0": True,
                "pr_auc_lift_gt_0": True,
                "finite_calibration_metrics": True,
                "nonconstant_probability_distribution": True,
                "sufficient_event_and_nonevent_counts": True,
                "no_feature_contract_mismatch": True,
            }
        )
    packet["freeze_gate_sha256"] = stable_hash(packet)
    write_json(out_dir / "rawseq_1m_cpu_challenger_freeze_decision.json", packet)
    write_csv(out_dir / "rawseq_1m_cpu_challenger_freeze_decision.csv", [packet])
    lines = [
        "Rawseq 1m CPU challenger freeze gate",
        f"Output: {out_dir}",
        f"Final recommendation: {final_recommendation}",
        f"Freeze created: {packet['freeze_created']}",
        f"Holdout integrity: {holdout_class}",
        f"Selected calibration contract: {packet['selected_calibration_contract']}",
        f"Candidate hash: {packet['candidate_contract_hash']}",
        "Reasons:",
    ]
    lines.extend(f"- {reason}" for reason in packet["freeze_block_reasons"])
    lines.append("Safety: CPU-only, no holdout scoring, no orders, no promotion, no champion mutation.")
    (out_dir / "rawseq_1m_cpu_challenger_freeze_decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
