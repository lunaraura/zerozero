#!/usr/bin/env python3
"""Report rawseq 1m artifacts superseded by methodology fixes."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import now_stamp, stable_hash, write_csv, write_json  # noqa: E402

DEFAULT_BOARD_ROOT = Path(r"F:\rsio\rawseq_1m_board_member_target_feature_tournaments")
DEFAULT_MHD_ROOT = Path(r"F:\rsio\rawseq_1m_multihorizon_downside_calibrated_confirmations")
DEFAULT_UPSIDE_ROOT = Path(r"F:\rsio\rawseq_1m_upside_excursion_calibrated_confirmations")
DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_methodology_supersession")

METHODOLOGY_FIXES = {
    "range_expansion_target_reference": {
        "status": "fixed_in_code",
        "supersession_reason": "range expansion target previously compared future range to rolling median of future-derived ranges",
        "new_rule": "future 5m range is compared to shifted rolling median of past 5m candle ranges",
    },
    "calibration_slope_intercept": {
        "status": "fixed_in_code",
        "supersession_reason": "calibration slope/intercept previously used linear least squares on binary targets",
        "new_rule": "calibration slope/intercept uses logistic regression on prediction logits",
    },
    "calibrated_freeze_artifacts": {
        "status": "fixed_in_code",
        "supersession_reason": "older calibrated confirmation packets did not save final fitted classifiers/calibrators when freeze gates passed",
        "new_rule": "freeze packet requires final classifiers/calibrators artifact, reload parity, and artifact hash",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def artifact_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)


def board_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in artifact_dirs(root):
        contract = read_json(run_dir / "board_member_target_feature_tournament_contract.json")
        metrics_path = run_dir / "board_member_target_feature_metrics.csv"
        target_path = run_dir / "board_member_target_manifest.csv"
        range_rows = 0
        if metrics_path.exists():
            try:
                metrics = pd.read_csv(metrics_path, usecols=lambda c: c in {"target_name", "target_lane"})
                range_rows = int(metrics["target_name"].astype(str).str.contains("range_expansion_future_range_gt_recent_median", na=False).sum())
            except Exception:
                range_rows = -1
        target_range_rows = 0
        if target_path.exists():
            try:
                targets = pd.read_csv(target_path, usecols=lambda c: c in {"target_name", "target_lane"})
                target_range_rows = int(targets["target_name"].astype(str).str.contains("range_expansion_future_range_gt_recent_median", na=False).sum())
            except Exception:
                target_range_rows = -1
        superseded = range_rows != 0 or target_range_rows != 0
        rows.append(
            {
                "artifact_type": "board_target_feature_tournament",
                "artifact_dir": str(run_dir),
                "created_at": contract.get("created_at", ""),
                "contract_hash": contract.get("contract_hash", ""),
                "methodology_issue": "range_expansion_target_reference" if superseded else "",
                "supersession_status": "superseded_do_not_reuse_range_expansion_rows" if superseded else "current_or_not_affected",
                "range_metric_rows": range_rows,
                "range_target_manifest_rows": target_range_rows,
                "required_action": "regenerate range-expansion rows with causal past-range reference before reuse" if superseded else "",
                "safe_to_use_for_new_freeze": False if superseded else True,
            }
        )
    return rows


def confirmation_rows(root: Path, family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in artifact_dirs(root):
        decision = read_json(run_dir / "candidate_decision.json")
        contract = read_json(run_dir / "predeclared_experiment_manifest.json")
        final_status = read_json(run_dir / "final_artifact_status.json")
        has_final_artifact_status = bool(final_status)
        freeze_dirs = list(run_dir.glob("frozen_*_board_member_challenger/final_models_and_calibrators.pkl"))
        freeze_allowed = bool(decision.get("freeze_allowed", False))
        methodology_issues = ["calibration_slope_intercept"]
        if freeze_allowed and not freeze_dirs:
            methodology_issues.append("calibrated_freeze_artifacts")
        status = "methodology_superseded_reconfirm_required"
        if has_final_artifact_status and decision.get("freeze_status") not in {"freeze_blocked_missing_final_model_artifacts", ""}:
            status = "post_fix_packet"
        rows.append(
            {
                "artifact_type": f"{family}_calibrated_confirmation",
                "artifact_dir": str(run_dir),
                "created_at": contract.get("created_at", ""),
                "predeclared_experiment_hash": contract.get("predeclared_experiment_hash", ""),
                "candidate_hash": decision.get("candidate_hash", ""),
                "freeze_allowed": freeze_allowed,
                "freeze_status": decision.get("freeze_status", ""),
                "methodology_issue": ";".join(methodology_issues),
                "supersession_status": status,
                "has_final_artifact_status": has_final_artifact_status,
                "has_final_model_artifact": bool(freeze_dirs),
                "required_action": "rerun confirmation under logistic calibration slope and final-artifact freeze code before reuse",
                "safe_to_use_for_new_freeze": status == "post_fix_packet" and bool(freeze_dirs),
            }
        )
    return rows


def text_summary(path: Path, rows: list[dict[str, Any]], contract: dict[str, Any]) -> None:
    df = pd.DataFrame(rows)
    superseded = int((df["supersession_status"].astype(str).str.contains("superseded|reconfirm", regex=True)).sum()) if not df.empty else 0
    safe = int(df["safe_to_use_for_new_freeze"].astype(bool).sum()) if not df.empty and "safe_to_use_for_new_freeze" in df else 0
    lines = [
        "RAWSEQ 1M METHODOLOGY SUPERSESSION REPORT",
        f"created_at={contract['created_at']}",
        f"artifact_rows={len(rows)}",
        f"superseded_or_reconfirm_rows={superseded}",
        f"safe_to_use_for_new_freeze_rows={safe}",
        "",
        "Methodology fixes:",
    ]
    for key, payload in METHODOLOGY_FIXES.items():
        lines.append(f"- {key}: {payload['supersession_reason']} -> {payload['new_rule']}")
    lines.extend(
        [
            "",
            "Policy:",
            "- Superseded rows are historical diagnostics only.",
            "- Do not freeze, promote, or rank old affected packets without regeneration.",
            "- This report is read-only and does not mutate old artifact folders.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(board_root: Path, mhd_root: Path, upside_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    rows.extend(board_rows(board_root))
    rows.extend(confirmation_rows(mhd_root, "multihorizon_downside"))
    rows.extend(confirmation_rows(upside_root, "upside_excursion"))
    contract = {
        "created_at": now_stamp(),
        "board_root": str(board_root),
        "multihorizon_downside_root": str(mhd_root),
        "upside_excursion_root": str(upside_root),
        "methodology_fixes": METHODOLOGY_FIXES,
        "report_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "old_artifact_mutation": False,
    }
    contract["methodology_supersession_report_hash"] = stable_hash({"rows": rows, "contract": contract})
    return rows, contract


def main() -> int:
    board_root = Path(os.getenv("RAWSEQ_SUPERSESSION_BOARD_ROOT", str(DEFAULT_BOARD_ROOT)))
    mhd_root = Path(os.getenv("RAWSEQ_SUPERSESSION_MHD_ROOT", str(DEFAULT_MHD_ROOT)))
    upside_root = Path(os.getenv("RAWSEQ_SUPERSESSION_UPSIDE_ROOT", str(DEFAULT_UPSIDE_ROOT)))
    output_root = Path(os.getenv("RAWSEQ_SUPERSESSION_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    out_dir = output_root / f"rawseq_1m_methodology_supersession_{now_stamp()}"
    rows, contract = build_report(board_root, mhd_root, upside_root)
    write_csv(out_dir / "rawseq_1m_methodology_supersession.csv", rows)
    write_json(out_dir / "rawseq_1m_methodology_supersession_contract.json", contract)
    text_summary(out_dir / "rawseq_1m_methodology_supersession.txt", rows, contract)
    print(f"output_dir={out_dir}")
    print(f"artifact_rows={len(rows)}")
    print(f"superseded_rows={sum(1 for row in rows if 'superseded' in str(row.get('supersession_status')) or 'reconfirm' in str(row.get('supersession_status')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
