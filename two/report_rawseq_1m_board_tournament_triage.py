#!/usr/bin/env python3
"""Triage broad rawseq 1m board-member tournaments into next actions."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import now_stamp, stable_hash, write_csv, write_json  # noqa: E402

DEFAULT_TOURNAMENT_ROOT = Path(r"F:\rsio\rawseq_1m_board_member_target_feature_tournaments")
DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_board_tournament_triage")
DEFAULT_RESEARCH_STATE_ROOT = Path(r"F:\rsio\rawseq_1m_board_research_state")
SEVERITY_CONFIRMATION_PACKET = Path(
    r"F:\rsio\rawseq_1m_downside_severity_family_confirmations\rawseq_1m_downside_severity_family_confirmation_20260713T045526Z"
)
VOLATILITY_CONFIRMATION_PACKET = Path(
    r"F:\rsio\rawseq_1m_volatility_family_confirmations\rawseq_1m_volatility_family_confirmation_20260713T061822Z"
)
MULTIHORIZON_DOWNSIDE_CONFIRMATION_PACKET = Path(
    r"F:\rsio\rawseq_1m_multihorizon_downside_calibrated_confirmations\rawseq_1m_multihorizon_downside_calibrated_confirmation_20260713T073838Z"
)
UPSIDE_CONFIRMATION_PACKET = Path(
    r"F:\rsio\rawseq_1m_upside_excursion_calibrated_confirmations\rawseq_1m_upside_excursion_calibrated_confirmation_20260713T083805Z"
)
SUPPORTED_FAMILY_STATES = [
    "discovery_pending",
    "discovery_survivor",
    "confirmation_pending",
    "confirmation_failed_closed",
    "confirmation_passed",
    "frozen_challenger",
    "prospective_holdout_pending",
    "prospective_holdout_failed",
    "prospective_holdout_passed",
    "shadow_monitoring",
    "promoted_board_member",
]
STAGE_PRIORITY = {"freeze_candidate": 3, "confirmation_candidate": 2, "discovery_candidate": 1, "reject": 0}
STATE_PRIORITY = {
    "discovery_pending": 0,
    "discovery_survivor": 1,
    "confirmation_pending": 2,
    "confirmation_failed_closed": 3,
    "confirmation_passed": 4,
    "frozen_challenger": 5,
    "prospective_holdout_pending": 6,
    "prospective_holdout_failed": 7,
    "prospective_holdout_passed": 8,
    "shadow_monitoring": 9,
    "promoted_board_member": 10,
    "retired": 11,
}
FAMILY_ORDER = [
    "volatility_expansion_family",
    "downside_severity_family",
    "downside_0p5_multi_horizon",
    "upside_excursion_family",
    "barrier_first_family",
    "market_structure_family",
]
FAMILY_STATE_KEY_BY_FAMILY = {
    "downside_severity_family": "downside_severity",
    "volatility_expansion_family": "volatility_expansion",
    "downside_0p5_multi_horizon": "multi_horizon_downside",
    "upside_excursion_family": "upside_excursion",
    "barrier_first_family": "barrier_first",
}
FAMILY_BY_STATE_KEY = {value: key for key, value in FAMILY_STATE_KEY_BY_FAMILY.items()}
STATE_RECOMMENDATION_BY_KEY = {
    "upside_excursion": "confirm_upside_excursion_with_nested_calibration",
}


def latest_dir(root: Path) -> Path:
    dirs = sorted([p for p in root.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise SystemExit(f"No tournament directories found under {root}")
    return dirs[0]


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def family_name(target_lane: str, target_name: str) -> str:
    if target_lane == "multi_horizon_downside":
        return "downside_0p5_multi_horizon"
    if target_lane == "downside_severity":
        return "downside_severity_family"
    if target_lane == "upside_excursion":
        return "upside_excursion_family"
    if target_lane == "volatility_expansion":
        return "volatility_expansion_family"
    if target_lane == "barrier_first":
        return "barrier_first_family"
    if "ema" in target_name or "structure" in target_name:
        return "market_structure_family"
    return f"{target_lane}_family"


def target_role(target_name: str, target_lane: str) -> str:
    if target_name == "volatility_expansion_future_vol_gt_current_h15m_fw240":
        return "primary"
    if target_name == "range_expansion_future_range_gt_recent_median_h5m_fw240":
        return "secondary"
    if target_lane in {"multi_horizon_downside", "downside_severity", "upside_excursion"} and "_h1m_" in target_name:
        return "primary"
    return "member"


def feature_complexity_penalty(feature_group: str) -> float:
    if feature_group == "existing":
        return 0.0
    if feature_group.startswith("all_") or feature_group == "all_challenger_features":
        return 0.02
    return 0.01


def staged_status(row: dict[str, Any]) -> str:
    median = safe_float(row.get("median_brier_skill"))
    worst = safe_float(row.get("worst_fold_brier_skill"))
    pr = safe_float(row.get("median_pr_auc_lift"))
    rows = safe_float(row.get("rows"), 0.0)
    survives = boolish(row.get("survives_validation_gate"))
    if median >= 0.08 and worst >= 0.03 and pr >= 0.10 and rows >= 10000:
        return "freeze_candidate"
    if median >= 0.02 and worst >= 0.0 and pr >= 0.03 and survives:
        return "confirmation_candidate"
    if survives or (median > 0 and pr > 0):
        return "discovery_candidate"
    return "reject"


def ranking_score(row: dict[str, Any]) -> float:
    median = safe_float(row.get("median_brier_skill"), 0.0)
    pr = safe_float(row.get("median_pr_auc_lift"), 0.0)
    log_loss = safe_float(row.get("median_log_loss_improvement"), 0.0)
    worst = safe_float(row.get("worst_fold_brier_skill"), 0.0)
    calibration = safe_float(row.get("max_expected_calibration_error"), 0.0)
    complexity = feature_complexity_penalty(str(row.get("feature_group", "")))
    worst_penalty = abs(min(0.0, worst))
    return median + 0.25 * pr + 0.25 * log_loss - 2.0 * worst_penalty - calibration - complexity


def failure_flags(row: dict[str, Any], metrics: pd.DataFrame) -> dict[str, Any]:
    key = (
        str(row.get("target_name")),
        str(row.get("feature_group")),
        str(row.get("model")),
    )
    group = metrics[
        (metrics["target_name"].astype(str) == key[0])
        & (metrics["feature_group"].astype(str) == key[1])
        & (metrics["model"].astype(str) == key[2])
        & (metrics["status"].astype(str) == "OK")
    ].copy()
    brier = pd.to_numeric(group.get("brier_skill_vs_prevalence", pd.Series(dtype=float)), errors="coerce")
    ece = pd.to_numeric(group.get("expected_calibration_error", pd.Series(dtype=float)), errors="coerce")
    mce = pd.to_numeric(group.get("maximum_calibration_error", pd.Series(dtype=float)), errors="coerce")
    scenario_available = "scenario" in group.columns
    combined = group[group["scenario"].astype(str).eq("combined_symbol_time_exclusion")] if "scenario" in group else pd.DataFrame()
    time_block = group[group["scenario"].astype(str).eq("leave_time_block_out")] if "scenario" in group else pd.DataFrame()
    symbol = group[group["scenario"].astype(str).eq("leave_symbol_out")] if "scenario" in group else pd.DataFrame()
    return {
        "scenario_validation_scope": "full_scenario_ladder" if scenario_available else "pooled_chronological_discovery_only",
        "scenario_validation_missing": not scenario_available,
        "failed_by_symbol": bool(len(symbol) and (pd.to_numeric(symbol["brier_skill_vs_prevalence"], errors="coerce") <= 0).any()),
        "failed_by_time_block": bool(len(time_block) and (pd.to_numeric(time_block["brier_skill_vs_prevalence"], errors="coerce") <= 0).any()),
        "failed_by_combined_symbol_time": bool(len(combined) and (pd.to_numeric(combined["brier_skill_vs_prevalence"], errors="coerce") <= 0).any()),
        "symbol_failure_applicable": scenario_available and len(symbol) > 0,
        "time_block_failure_applicable": scenario_available and len(time_block) > 0,
        "combined_symbol_time_failure_applicable": scenario_available and len(combined) > 0,
        "failed_by_calibration": bool((ece > 0.08).any() or (mce > 0.25).any()),
        "worst_metric_brier_skill": float(brier.min()) if len(brier) else math.nan,
        "max_expected_calibration_error": float(ece.max()) if len(ece) else math.nan,
        "max_calibration_error": float(mce.max()) if len(mce) else math.nan,
    }


def repair_path(row: dict[str, Any]) -> str:
    if row.get("stage_status") == "reject":
        return "stop_lane"
    if row.get("failed_by_calibration") and safe_float(row.get("median_brier_skill")) > 0.05 and safe_float(row.get("median_pr_auc_lift")) > 0.10:
        return "run_calibration_repair"
    if row.get("failed_by_combined_symbol_time") or row.get("failed_by_time_block"):
        return "run_regime_or_time_stability_diagnostic"
    if row.get("failed_by_symbol"):
        return "run_symbol_failure_attribution"
    if row.get("failed_by_complexity_penalty"):
        return "prefer_existing_or_test_feature_family_ablation"
    return "confirm_family"


def load_inputs(tournament_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    survivors_path = tournament_dir / "board_member_target_feature_survivors.csv"
    metrics_path = tournament_dir / "board_member_target_feature_metrics.csv"
    if not survivors_path.exists() or not metrics_path.exists():
        raise SystemExit(
            "Tournament directory is missing required broad tournament artifacts: "
            f"tournament_dir={tournament_dir} survivors_exists={survivors_path.exists()} "
            f"metrics_exists={metrics_path.exists()}"
        )
    survivors = pd.read_csv(survivors_path)
    metrics = pd.read_csv(metrics_path)
    return survivors, metrics


def resolve_tournament_dir() -> Path:
    explicit = os.getenv("RAWSEQ_TRIAGE_TOURNAMENT_DIR")
    if explicit and explicit.strip():
        return Path(explicit).expanduser()
    root = Path(os.getenv("RAWSEQ_TRIAGE_TOURNAMENT_ROOT", str(DEFAULT_TOURNAMENT_ROOT))).expanduser()
    return latest_dir(root)


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"json_error": f"could not decode {path}"}


def confirmation_packet_state(packet: Path, summary_name: str) -> tuple[str, dict[str, Any]]:
    summary = read_json_if_exists(packet / summary_name)
    if summary.get("family_status") == "FAIL" and summary.get("freeze_allowed") is False:
        return "confirmation_failed_closed", summary
    if summary.get("final_recommendation") in {"multihorizon_downside_not_confirmed", "upside_excursion_not_confirmed"} and summary.get("freeze_allowed") is False:
        return "confirmation_failed_closed", summary
    if summary.get("freeze_allowed") is True:
        return "confirmation_passed", summary
    return "confirmation_pending", summary


def default_research_state_entries() -> dict[str, dict[str, Any]]:
    severity_state, severity_summary = confirmation_packet_state(
        SEVERITY_CONFIRMATION_PACKET,
        "downside_severity_family_confirmation_summary.json",
    )
    volatility_state, volatility_summary = confirmation_packet_state(
        VOLATILITY_CONFIRMATION_PACKET,
        "volatility_family_confirmation_summary.json",
    )
    multihorizon_state, multihorizon_summary = confirmation_packet_state(
        MULTIHORIZON_DOWNSIDE_CONFIRMATION_PACKET,
        "candidate_decision.json",
    )
    upside_state, upside_summary = confirmation_packet_state(
        UPSIDE_CONFIRMATION_PACKET,
        "candidate_decision.json",
    )
    return {
        "downside_severity": {
            "family": "downside_severity_family",
            "state": severity_state,
            "reopen_allowed": False,
            "source_packet": str(SEVERITY_CONFIRMATION_PACKET),
            "source_packet_exists": SEVERITY_CONFIRMATION_PACKET.exists(),
            "source_family_status": severity_summary.get("family_status", ""),
            "source_freeze_allowed": severity_summary.get("freeze_allowed", ""),
        },
        "volatility_expansion": {
            "family": "volatility_expansion_family",
            "state": volatility_state,
            "reopen_allowed": False,
            "source_packet": str(VOLATILITY_CONFIRMATION_PACKET),
            "source_packet_exists": VOLATILITY_CONFIRMATION_PACKET.exists(),
            "source_family_status": volatility_summary.get("family_status", ""),
            "source_freeze_allowed": volatility_summary.get("freeze_allowed", ""),
        },
        "multi_horizon_downside": {
            "family": "downside_0p5_multi_horizon",
            "state": multihorizon_state,
            "reopen_allowed": False,
            "source_packet": str(MULTIHORIZON_DOWNSIDE_CONFIRMATION_PACKET),
            "source_packet_exists": MULTIHORIZON_DOWNSIDE_CONFIRMATION_PACKET.exists(),
            "source_family_status": multihorizon_summary.get("family_status", ""),
            "source_freeze_allowed": multihorizon_summary.get("freeze_allowed", ""),
            "source_final_recommendation": multihorizon_summary.get("final_recommendation", ""),
            "source_equal_symbol_family_median_brier_skill": multihorizon_summary.get("equal_symbol_family_median_brier_skill", ""),
            "source_worst_fold_brier_skill": multihorizon_summary.get("overall_worst_fold_brier_skill", ""),
            "source_combined_worst_fold_brier_skill": multihorizon_summary.get("combined_symbol_time_worst_fold_brier_skill", ""),
        },
        "upside_excursion": {
            "family": "upside_excursion_family",
            "state": upside_state,
            "reopen_allowed": False if upside_state == "confirmation_failed_closed" else True,
            "source_packet": str(UPSIDE_CONFIRMATION_PACKET),
            "source_packet_exists": UPSIDE_CONFIRMATION_PACKET.exists(),
            "source_family_status": upside_summary.get("family_status", ""),
            "source_freeze_allowed": upside_summary.get("freeze_allowed", ""),
            "source_final_recommendation": upside_summary.get("final_recommendation", ""),
            "source_equal_symbol_family_median_brier_skill": upside_summary.get("equal_symbol_family_median_brier_skill", ""),
            "source_worst_fold_brier_skill": upside_summary.get("overall_worst_fold_brier_skill", ""),
            "source_combined_worst_fold_brier_skill": upside_summary.get("combined_symbol_time_worst_fold_brier_skill", ""),
        },
        "barrier_first": {
            "family": "barrier_first_family",
            "state": "discovery_survivor",
            "reopen_allowed": True,
            "confirmation_priority": "after_upside_excursion",
            "source_packet": "",
            "source_packet_exists": "",
            "source_family_status": "",
            "source_freeze_allowed": "",
        },
    }


def validate_research_state_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    families = registry.get("families", {})
    if not isinstance(families, dict):
        return ["registry.families must be an object"]
    for key, entry in families.items():
        state = str(entry.get("state", ""))
        if state not in SUPPORTED_FAMILY_STATES:
            errors.append(f"{key}: unsupported state {state}")
        if state.endswith("_closed") and boolish(entry.get("reopen_allowed")):
            errors.append(f"{key}: closed state cannot set reopen_allowed=true")
    return errors


def load_research_state_registry(state_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry_path = state_root / "board_research_state_registry.json"
    existing = read_json_if_exists(registry_path)
    families = existing.get("families", {}) if isinstance(existing.get("families"), dict) else {}
    defaults = default_research_state_entries()
    transitions: list[dict[str, Any]] = []
    for key, default_entry in defaults.items():
        previous = dict(families.get(key, {})) if isinstance(families.get(key), dict) else {}
        default_has_downstream_evidence = bool(default_entry.get("source_packet_exists")) and str(default_entry.get("state", "")) != "confirmation_pending"
        previous_priority = STATE_PRIORITY.get(str(previous.get("state", "discovery_pending")), 0)
        default_priority = STATE_PRIORITY.get(str(default_entry.get("state", "discovery_pending")), 0)
        if default_has_downstream_evidence or default_priority >= previous_priority:
            merged = {**previous, **default_entry, "updated_at": now_stamp()}
        else:
            merged = {**default_entry, **previous, "updated_at": now_stamp()}
        if previous.get("state") != merged.get("state") or previous.get("reopen_allowed") != merged.get("reopen_allowed"):
            transitions.append(
                {
                    "family_key": key,
                    "family": merged.get("family"),
                    "previous_state": previous.get("state", ""),
                    "new_state": merged.get("state"),
                    "previous_reopen_allowed": previous.get("reopen_allowed", ""),
                    "new_reopen_allowed": merged.get("reopen_allowed"),
                    "source_packet": merged.get("source_packet", ""),
                    "transition_reason": "downstream_confirmation_ingested" if merged.get("source_packet") else "predeclared_pending_confirmation",
                    "created_at": now_stamp(),
                }
            )
        families[key] = merged
    registry = {
        "created_at": existing.get("created_at", now_stamp()),
        "updated_at": now_stamp(),
        "supported_states": SUPPORTED_FAMILY_STATES,
        "families": families,
        "registry_hash": stable_hash(families),
    }
    errors = validate_research_state_registry(registry)
    if errors:
        raise SystemExit("Invalid board research state registry: " + "; ".join(errors))
    write_json(registry_path, registry)
    return registry, transitions


def state_for_family(registry: dict[str, Any], family: str) -> dict[str, Any]:
    key = FAMILY_STATE_KEY_BY_FAMILY.get(family, family.replace("_family", ""))
    families = registry.get("families", {})
    if isinstance(families, dict) and isinstance(families.get(key), dict):
        return families[key]
    return {
        "family": family,
        "state": "discovery_survivor",
        "reopen_allowed": True,
        "source_packet": "",
    }


def recommendation_for_state(family: str, raw_recommendation: str, registry: dict[str, Any]) -> tuple[str, bool, str]:
    state_entry = state_for_family(registry, family)
    state = str(state_entry.get("state", ""))
    reopen_allowed = boolish(state_entry.get("reopen_allowed"))
    family_key = FAMILY_STATE_KEY_BY_FAMILY.get(family, "")
    if state == "confirmation_failed_closed" and not reopen_allowed:
        return "stop_lane", raw_recommendation != "stop_lane", "family_confirmation_failed_closed"
    if state == "confirmation_pending" and family_key in STATE_RECOMMENDATION_BY_KEY:
        return STATE_RECOMMENDATION_BY_KEY[family_key], raw_recommendation != STATE_RECOMMENDATION_BY_KEY[family_key], "state_registry_pending_confirmation"
    return raw_recommendation, False, "unchanged"


def apply_research_state_to_families(families: list[dict[str, Any]], registry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    updated: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for family in families:
        raw = str(family.get("recommended_next", ""))
        state_entry = state_for_family(registry, str(family.get("family", "")))
        updated_rec, suppressed, reason = recommendation_for_state(str(family.get("family", "")), raw, registry)
        out = {
            **family,
            "raw_recommended_next": raw,
            "recommended_next": updated_rec,
            "research_state": state_entry.get("state", "discovery_survivor"),
            "reopen_allowed": state_entry.get("reopen_allowed", True),
            "state_source_packet": state_entry.get("source_packet", ""),
            "state_recommendation_reason": reason,
        }
        updated.append(out)
        if suppressed:
            stale.append(
                {
                    "family": family.get("family"),
                    "research_state": state_entry.get("state", ""),
                    "reopen_allowed": state_entry.get("reopen_allowed", ""),
                    "stale_recommendation": raw,
                    "updated_recommendation": updated_rec,
                    "suppression_reason": reason,
                    "source_packet": state_entry.get("source_packet", ""),
                }
            )
    def recommendation_priority(row: dict[str, Any]) -> int:
        if row.get("recommended_next") == "confirm_upside_excursion_with_nested_calibration":
            return 3
        if row.get("recommended_next") == "stop_lane":
            return 0
        return 1

    updated.sort(key=lambda r: (recommendation_priority(r), STAGE_PRIORITY.get(str(r["best_stage_status"]), 0), safe_float(r["best_ranking_score"])), reverse=True)
    return updated, stale


def top_discovery_rows(survivors: pd.DataFrame, metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in survivors.to_dict("records"):
        row = dict(raw)
        row["family"] = family_name(str(row.get("target_lane")), str(row.get("target_name")))
        row["target_role"] = target_role(str(row.get("target_name")), str(row.get("target_lane")))
        row["stage_status"] = staged_status(row)
        row["ranking_score"] = ranking_score(row)
        row.update(failure_flags(row, metrics))
        row["failed_by_secondary_target"] = False
        row["failed_by_complexity_penalty"] = False
        row["repair_path"] = repair_path(row)
        rows.append(row)
    rows.sort(key=lambda r: (STAGE_PRIORITY.get(str(r["stage_status"]), 0), safe_float(r["ranking_score"])), reverse=True)
    return rows


def feature_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("feature_group") == "existing":
            by_key[(str(row.get("target_name")), str(row.get("model")))] = row
    for row in rows:
        base = by_key.get((str(row.get("target_name")), str(row.get("model"))))
        if not base or row.get("feature_group") == "existing":
            continue
        delta_median = safe_float(row.get("median_brier_skill")) - safe_float(base.get("median_brier_skill"))
        delta_worst = safe_float(row.get("worst_fold_brier_skill")) - safe_float(base.get("worst_fold_brier_skill"))
        delta_pr = safe_float(row.get("median_pr_auc_lift")) - safe_float(base.get("median_pr_auc_lift"))
        failed_complexity = not (delta_median >= 0.01 and delta_worst >= 0.0)
        out.append(
            {
                "target_lane": row.get("target_lane"),
                "target_name": row.get("target_name"),
                "model": row.get("model"),
                "feature_group": row.get("feature_group"),
                "baseline_feature_group": "existing",
                "expanded_delta_median_brier": delta_median,
                "expanded_delta_worst_fold": delta_worst,
                "expanded_delta_pr_auc_lift": delta_pr,
                "failed_by_complexity_penalty": failed_complexity,
            }
        )
    out.sort(key=lambda r: (not r["failed_by_complexity_penalty"], safe_float(r["expanded_delta_median_brier"])), reverse=True)
    return out


def family_rollup_rows(rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    delta_df = pd.DataFrame(deltas)
    out: list[dict[str, Any]] = []
    for family, group in df.groupby("family", dropna=False):
        stages = group["stage_status"].astype(str)
        best = group.sort_values(["ranking_score"], ascending=False).iloc[0].to_dict()
        primary = group[group["target_role"].astype(str) == "primary"]
        secondary = group[group["target_role"].astype(str) == "secondary"]
        family_deltas = delta_df[delta_df["target_name"].isin(group["target_name"].astype(str))] if not delta_df.empty else pd.DataFrame()
        out.append(
            {
                "family": family,
                "candidate_rows": int(len(group)),
                "best_stage_status": max(stages, key=lambda s: STAGE_PRIORITY.get(s, 0)),
                "confirmation_candidates": int((stages == "confirmation_candidate").sum()),
                "freeze_candidates": int((stages == "freeze_candidate").sum()),
                "best_target_name": best.get("target_name"),
                "best_feature_group": best.get("feature_group"),
                "best_model": best.get("model"),
                "best_ranking_score": best.get("ranking_score"),
                "best_median_brier_skill": best.get("median_brier_skill"),
                "worst_member_brier_skill": float(pd.to_numeric(group["worst_fold_brier_skill"], errors="coerce").min()),
                "primary_target_count": int(len(primary)),
                "secondary_target_count": int(len(secondary)),
                "best_feature_delta_median": float(pd.to_numeric(family_deltas.get("expanded_delta_median_brier", pd.Series(dtype=float)), errors="coerce").max()) if not family_deltas.empty else math.nan,
                "recommended_next": family_recommendation(str(family), group),
            }
        )
    out.sort(key=lambda r: (STAGE_PRIORITY.get(str(r["best_stage_status"]), 0), safe_float(r["best_ranking_score"])), reverse=True)
    return out


def family_recommendation(family: str, group: pd.DataFrame) -> str:
    best_status = max(group["stage_status"].astype(str), key=lambda s: STAGE_PRIORITY.get(s, 0))
    if best_status == "freeze_candidate":
        if family == "volatility_expansion_family":
            return "confirm_volatility_family"
        if family == "downside_severity_family":
            return "confirm_downside_severity_family"
        return "confirm_family"
    if (group["repair_path"].astype(str) == "run_calibration_repair").any():
        return "run_calibration_repair"
    if best_status == "confirmation_candidate":
        if family == "volatility_expansion_family":
            return "confirm_volatility_family"
        if family == "downside_severity_family":
            return "confirm_downside_severity_family"
        return "confirm_family"
    if (pd.to_numeric(group["median_brier_skill"], errors="coerce") > 0).any():
        return "test_feature_family_ablation"
    return "stop_lane"


def failure_attribution_rows(rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    delta_map = {(str(r["target_name"]), str(r["model"]), str(r["feature_group"])): r for r in deltas}
    out: list[dict[str, Any]] = []
    for row in rows:
        delta = delta_map.get((str(row.get("target_name")), str(row.get("model")), str(row.get("feature_group"))))
        failed_complexity = bool(delta and delta.get("failed_by_complexity_penalty"))
        reasons = []
        for key in ["failed_by_symbol", "failed_by_time_block", "failed_by_combined_symbol_time", "failed_by_calibration"]:
            if row.get(key):
                reasons.append(key)
        if failed_complexity:
            reasons.append("failed_by_complexity_penalty")
        out.append(
            {
                "family": row.get("family"),
                "target_lane": row.get("target_lane"),
                "target_name": row.get("target_name"),
                "feature_group": row.get("feature_group"),
                "model": row.get("model"),
                "stage_status": row.get("stage_status"),
                "failure_class": classify_failure(row, failed_complexity),
                "failure_reasons": ";".join(reasons),
                "repair_path": repair_path({**row, "failed_by_complexity_penalty": failed_complexity}),
                "worst_metric_brier_skill": row.get("worst_metric_brier_skill"),
                "max_expected_calibration_error": row.get("max_expected_calibration_error"),
                "max_calibration_error": row.get("max_calibration_error"),
            }
        )
    out.sort(key=lambda r: (STAGE_PRIORITY.get(str(r["stage_status"]), 0), r["failure_class"]), reverse=True)
    return out


def classify_failure(row: dict[str, Any], failed_complexity: bool) -> str:
    if row.get("stage_status") == "reject":
        return "low_signal_or_no_edge"
    if failed_complexity:
        return "feature_complexity_not_worth_it"
    if row.get("failed_by_calibration") and safe_float(row.get("median_brier_skill")) > 0.05:
        return "good_discrimination_bad_calibration"
    if row.get("failed_by_combined_symbol_time") or row.get("failed_by_time_block"):
        return "high_lift_unstable"
    if row.get("failed_by_symbol"):
        return "symbol_specific"
    return "stable_but_low_lift"


def recommendation_payload(families: list[dict[str, Any]], top_rows: list[dict[str, Any]], state_aware: bool = False) -> dict[str, Any]:
    recs = []
    for family in families:
        rec = str(family.get("recommended_next"))
        if rec != "stop_lane":
            recs.append({"family": family["family"], "recommended_next": rec, "best_target_name": family.get("best_target_name"), "best_feature_group": family.get("best_feature_group"), "best_model": family.get("best_model")})
    if not recs:
        recs.append({"family": "all", "recommended_next": "stop_lane"})
    first_actionable_family = next((str(row["family"]) for row in recs if row.get("recommended_next") != "stop_lane"), "")
    top_candidate = next((row for row in top_rows if str(row.get("family")) == first_actionable_family), top_rows[0] if top_rows else {})
    return {
        "created_at": now_stamp(),
        "recommendations": recs,
        "top_candidate": top_candidate,
        "selection_warning": "This triage reads validation tournament artifacts only. It does not freeze, promote, or use holdout/future labels.",
        "state_aware": state_aware,
        "triage_hash": stable_hash({"families": families, "top": top_rows[:20]}),
    }


def text_summary(path: Path, tournament_dir: Path, families: list[dict[str, Any]], top_rows: list[dict[str, Any]], state_aware: bool = False) -> None:
    lines = [
        "RAWSEQ 1M BOARD TOURNAMENT TRIAGE",
        f"source_tournament={tournament_dir}",
        "",
        "Top candidates:",
    ]
    for row in top_rows[:20]:
        lines.append(
            f"- stage={row['stage_status']} family={row['family']} target={row['target_name']} features={row['feature_group']} "
            f"model={row['model']} score={safe_float(row['ranking_score']):.6f} median_brier={safe_float(row['median_brier_skill']):.6f} "
            f"worst={safe_float(row['worst_fold_brier_skill']):.6f} repair={row['repair_path']}"
        )
    lines.extend(["", "Family recommendations:"])
    for family in families:
        lines.append(
            f"- family={family['family']} status={family['best_stage_status']} next={family['recommended_next']} "
            f"best={family['best_target_name']} features={family['best_feature_group']}"
            + (f" state={family.get('research_state')} reason={family.get('state_recommendation_reason')}" if state_aware else "")
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- discovery_candidate is not freezeable.",
            "- confirmation_candidate means a focused confirmation script may be justified.",
            "- freeze_candidate is only a pre-confirmation triage label; strict family confirmation still controls freeze.",
            "- updated_* files are state-aware and suppress closed downstream confirmations.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    tournament_dir = resolve_tournament_dir()
    output_root = Path(os.getenv("RAWSEQ_TRIAGE_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))).expanduser()
    out_dir = output_root / f"rawseq_1m_board_tournament_triage_{now_stamp()}"
    state_root = Path(os.getenv("RAWSEQ_BOARD_RESEARCH_STATE_ROOT", str(DEFAULT_RESEARCH_STATE_ROOT))).expanduser()
    survivors, metrics = load_inputs(tournament_dir)
    top_rows = top_discovery_rows(survivors, metrics)
    deltas = feature_delta_rows(top_rows)
    failures = failure_attribution_rows(top_rows, deltas)
    families = family_rollup_rows(top_rows, deltas)
    recommendations = recommendation_payload(families, top_rows)
    registry, transitions = load_research_state_registry(state_root)
    updated_families, stale_recommendations = apply_research_state_to_families(families, registry)
    updated_recommendations = recommendation_payload(updated_families, top_rows, state_aware=True)
    write_csv(out_dir / "top_discovery_candidates.csv", top_rows)
    write_csv(out_dir / "family_rollup.csv", families)
    write_csv(out_dir / "feature_group_delta.csv", deltas)
    write_csv(out_dir / "failure_attribution.csv", failures)
    write_json(out_dir / "recommended_confirmations.json", recommendations)
    write_csv(out_dir / "state_transition_audit.csv", transitions)
    write_csv(out_dir / "stale_recommendation_audit.csv", stale_recommendations)
    write_csv(out_dir / "updated_family_rollup.csv", updated_families)
    write_json(out_dir / "updated_recommended_confirmations.json", updated_recommendations)
    text_summary(out_dir / "board_tournament_triage_summary.txt", tournament_dir, families, top_rows)
    text_summary(out_dir / "updated_board_tournament_triage_summary.txt", tournament_dir, updated_families, top_rows, state_aware=True)
    print(f"output_dir={out_dir}")
    print(f"source_tournament={tournament_dir}")
    print(f"top_rows={len(top_rows)}")
    print(f"families={len(families)}")
    print("recommendations=" + ",".join(row["recommended_next"] for row in updated_recommendations["recommendations"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
