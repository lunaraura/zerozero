#!/usr/bin/env python3
"""Status report for frozen downside-risk future paper shadow.

This report is deliberately conservative. It reads the cumulative future-shadow
ledger, enforces the prospective sample gates from the project objective, and
only evaluates the frozen acceptance gate when enough true-forward evidence has
accumulated.

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_future_shadow_lock import attach_implementation_lock
from scripts.tiny.freeze_rawseq_low_path_ridge_research_candidate import stable_hash

DEFAULT_SHADOW_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_future_shadow"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_future_shadow_status"
GOAL_MIN_CALENDAR_DAYS = 30
GOAL_MIN_NON_OVERLAP_H480 = 5000
GOAL_MIN_EVENTS = 100
GOAL_MIN_NON_EVENTS = 100
PRIMARY_POLICY_THRESHOLD = 0.8
BACKUP_POLICY_THRESHOLD = 0.7
REGIME_MIN_ROWS = 100
MIN_RISK_REDUCTION_PER_1PCT_COVERAGE_LOST = 0.01
MAX_PRIMARY_REJECTED_PERCENTAGE = 99.0


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


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


def safe_int(value: Any, default: int = 0) -> int:
    out = safe_float(value)
    return int(out) if math.isfinite(out) else default


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 2:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def latest_cumulative_dir(root: Path) -> Path:
    explicit = os.getenv("RAWSEQ_DOWNSIDE_SHADOW_CUMULATIVE_DIR", "").strip()
    if explicit:
        return resolve_path(explicit)
    parent = root / "rawseq_downside_risk_future_shadow_cumulative"
    candidates = [p for p in parent.glob("*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No cumulative shadow directories found under {parent}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def best_or_worst_summary(summary_rows: list[dict[str, str]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in summary_rows:
        for key in [
            "brier_skill_score",
            "log_loss_improvement",
            "pr_auc_lift_over_prevalence",
            "roc_auc",
            "expected_calibration_error",
            "maximum_calibration_error",
            "calibration_slope",
            "calibration_intercept",
            "prediction_unique_fraction",
        ]:
            val = safe_float(row.get(key))
            if math.isfinite(val):
                values.setdefault(key, []).append(val)
    out: dict[str, float] = {}
    for key, vals in values.items():
        if key in {"expected_calibration_error", "maximum_calibration_error"}:
            out[f"max_{key}"] = max(vals)
        elif key in {"calibration_slope", "calibration_intercept"}:
            out[f"median_{key}"] = sorted(vals)[len(vals) // 2]
        else:
            out[f"min_{key}"] = min(vals)
    return out


def status_from_gates(contract: dict[str, Any], summary_rows: list[dict[str, str]]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    h480_row = next((row for row in summary_rows if str(row.get("target_column", "")).endswith("_h480")), {})
    days = safe_float(h480_row.get("calendar_days"), safe_float(contract.get("h480_calendar_days"), 0.0))
    non_overlap = safe_int(h480_row.get("non_overlapping_h480_rows"), safe_int(contract.get("h480_non_overlapping_rows"), 0))
    true_forward_rows = safe_int(contract.get("cumulative_true_forward_decision_rows"), 0)
    labeled_rows = safe_int(contract.get("cumulative_labeled_rows"), 0)
    backfill_rows = safe_int(contract.get("cumulative_backfill_or_replay_decision_rows"), 0)
    if days < GOAL_MIN_CALENDAR_DAYS:
        reasons.append(f"calendar_days {days:.3f} < required {GOAL_MIN_CALENDAR_DAYS}")
    if non_overlap < GOAL_MIN_NON_OVERLAP_H480:
        reasons.append(f"non_overlapping_h480_rows {non_overlap} < required {GOAL_MIN_NON_OVERLAP_H480}")
    if labeled_rows <= 0:
        reasons.append("no true-forward labels are available yet")
    if true_forward_rows <= 0:
        reasons.append("no true-forward decisions are logged")
    if backfill_rows > 0:
        reasons.append(f"{backfill_rows} backfill/replay rows exist and are excluded")
    if not h480_row:
        reasons.append("h480 true-forward labels are not available yet")
    for row in summary_rows:
        col = row.get("target_column", "unknown")
        events = safe_int(row.get("events"), 0)
        non_events = safe_int(row.get("non_events"), 0)
        if events < GOAL_MIN_EVENTS:
            reasons.append(f"{col} events {events} < required {GOAL_MIN_EVENTS}")
        if non_events < GOAL_MIN_NON_EVENTS:
            reasons.append(f"{col} non_events {non_events} < required {GOAL_MIN_NON_EVENTS}")
    if reasons:
        return "accumulating_not_ready_for_acceptance", reasons
    return "ready_for_acceptance_review", []


def threshold_matches(value: Any, expected: float, tolerance: float = 1e-9) -> bool:
    numeric = safe_float(value)
    return math.isfinite(numeric) and abs(numeric - expected) <= tolerance


def bootstrap_gate(bootstrap_rows: list[dict[str, str]], target_columns: set[str]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    marginal: list[str] = []
    rows_by_target = {row.get("target_column", ""): row for row in bootstrap_rows}
    for col in sorted(target_columns):
        row = rows_by_target.get(col)
        if not row:
            failures.append(f"{col} bootstrap CI row missing")
            continue
        lower = safe_float(row.get("brier_skill_ci_low"))
        if not math.isfinite(lower):
            failures.append(f"{col} brier_skill lower CI missing")
        elif lower <= 0:
            failures.append(f"{col} brier_skill lower CI <= 0")
        blocks = safe_int(row.get("bootstrap_blocks"), 0)
        if blocks < 5:
            marginal.append(f"{col} bootstrap_blocks {blocks} < 5")
    return failures, marginal


def regime_gate(regime_rows: list[dict[str, str]], target_columns: set[str]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    marginal: list[str] = []
    required_features = {"volatility", "range", "volume_liquidity", "trend", "daytime_overnight", "weekday_weekend"}
    seen: set[tuple[str, str]] = set()
    for row in regime_rows:
        col = row.get("target_column", "")
        feature = row.get("regime_feature", "")
        if col not in target_columns or feature not in required_features:
            continue
        rows = safe_int(row.get("rows"), 0)
        seen.add((col, feature))
        if rows < REGIME_MIN_ROWS:
            marginal.append(f"{col} {feature}/{row.get('regime_bucket', '')} rows {rows} < {REGIME_MIN_ROWS}")
            continue
        if safe_float(row.get("brier_skill_score")) <= 0:
            failures.append(f"{col} {feature}/{row.get('regime_bucket', '')} brier_skill_score <= 0")
        if safe_float(row.get("log_loss_improvement")) <= 0:
            failures.append(f"{col} {feature}/{row.get('regime_bucket', '')} log_loss_improvement <= 0")
    for col in sorted(target_columns):
        for feature in sorted(required_features):
            if (col, feature) not in seen:
                marginal.append(f"{col} {feature} regime evidence missing")
    return failures, marginal


def threshold_utility_gate(threshold_rows: list[dict[str, str]], target_columns: set[str]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    marginal: list[str] = []
    for col in sorted(target_columns):
        primary = next(
            (row for row in threshold_rows if row.get("target_column") == col and threshold_matches(row.get("threshold"), PRIMARY_POLICY_THRESHOLD)),
            {},
        )
        backup = next(
            (row for row in threshold_rows if row.get("target_column") == col and threshold_matches(row.get("threshold"), BACKUP_POLICY_THRESHOLD)),
            {},
        )
        if not primary:
            failures.append(f"{col} primary threshold {PRIMARY_POLICY_THRESHOLD} utility row missing")
            continue
        if not backup:
            marginal.append(f"{col} backup threshold {BACKUP_POLICY_THRESHOLD} utility row missing")
        if safe_float(primary.get("adverse_events_caught")) <= 0:
            failures.append(f"{col} primary threshold catches no adverse events")
        if safe_float(primary.get("risk_reduction_per_1pct_coverage_lost")) <= MIN_RISK_REDUCTION_PER_1PCT_COVERAGE_LOST:
            failures.append(f"{col} primary threshold risk reduction per 1pct coverage lost <= {MIN_RISK_REDUCTION_PER_1PCT_COVERAGE_LOST}")
        rejected = safe_float(primary.get("percentage_all_opportunities_rejected"))
        if math.isfinite(rejected) and rejected >= MAX_PRIMARY_REJECTED_PERCENTAGE:
            failures.append(f"{col} primary threshold rejects {rejected:.3f}% >= {MAX_PRIMARY_REJECTED_PERCENTAGE}%")
    return failures, marginal


def acceptance_gate(
    rule: dict[str, Any],
    summary_rows: list[dict[str, str]],
    bootstrap_rows: list[dict[str, str]],
    regime_rows: list[dict[str, str]],
    threshold_rows: list[dict[str, str]],
) -> tuple[str, list[str], list[str]]:
    if not summary_rows:
        return "not_evaluated", ["summary rows are not available"], []
    requirements = rule.get("requirements", {})
    failures: list[str] = []
    marginal: list[str] = []
    min_log_loss = safe_float(requirements.get("min_log_loss_improvement"), 0.0)
    min_pr_lift = safe_float(requirements.get("min_pr_auc_lift_over_prevalence"), 0.01)
    max_ece = safe_float(requirements.get("expected_calibration_error_max"), 0.08)
    slope_min = safe_float(requirements.get("calibration_slope_min"), 0.5)
    slope_max = safe_float(requirements.get("calibration_slope_max"), 1.5)
    intercept_abs = safe_float(requirements.get("calibration_intercept_abs_max"), 0.5)
    target_columns = {row.get("target_column", "unknown") for row in summary_rows}
    for row in summary_rows:
        col = row.get("target_column", "unknown")
        if safe_float(row.get("brier_skill_score")) <= 0:
            failures.append(f"{col} brier_skill_score <= 0")
        if safe_float(row.get("log_loss_improvement")) <= min_log_loss:
            failures.append(f"{col} log_loss_improvement <= {min_log_loss}")
        if safe_float(row.get("pr_auc_lift_over_prevalence")) < min_pr_lift:
            failures.append(f"{col} pr_auc_lift_over_prevalence < {min_pr_lift}")
        if safe_float(row.get("expected_calibration_error")) > max_ece:
            failures.append(f"{col} expected_calibration_error > {max_ece}")
        slope = safe_float(row.get("calibration_slope"))
        if math.isfinite(slope) and not (slope_min <= slope <= slope_max):
            failures.append(f"{col} calibration_slope outside [{slope_min}, {slope_max}]")
        intercept = safe_float(row.get("calibration_intercept"))
        if math.isfinite(intercept) and abs(intercept) > intercept_abs:
            failures.append(f"{col} abs(calibration_intercept) > {intercept_abs}")
        if safe_float(row.get("prediction_unique_fraction")) <= 0.01:
            failures.append(f"{col} probabilities are near-constant")
    bootstrap_failures, bootstrap_marginal = bootstrap_gate(bootstrap_rows, target_columns)
    regime_failures, regime_marginal = regime_gate(regime_rows, target_columns)
    utility_failures, utility_marginal = threshold_utility_gate(threshold_rows, target_columns)
    failures.extend(bootstrap_failures)
    failures.extend(regime_failures)
    failures.extend(utility_failures)
    marginal.extend(bootstrap_marginal)
    marginal.extend(regime_marginal)
    marginal.extend(utility_marginal)
    if failures:
        return "fail", failures, marginal
    if marginal:
        return "marginal", [], marginal
    return "pass", [], []


def main() -> int:
    shadow_root = env_path("RAWSEQ_DOWNSIDE_SHADOW_ROOT", DEFAULT_SHADOW_ROOT)
    output_root = env_path("RAWSEQ_DOWNSIDE_SHADOW_STATUS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    cumulative_dir = latest_cumulative_dir(shadow_root)
    out_dir = output_root / f"rawseq_downside_risk_future_shadow_status_{now_stamp()}"
    contract = read_json(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_contract.json")
    candidate_dir = resolve_path(contract.get("candidate_dir", ".")) if contract.get("candidate_dir") else Path(".")
    rule = read_json(candidate_dir / "rawseq_downside_risk_future_acceptance_rule.json")
    summary_rows = read_csv_rows(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_summary.csv")
    bootstrap_rows = read_csv_rows(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_bootstrap_ci.csv")
    threshold_rows = read_csv_rows(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_threshold_utility.csv")
    prediction_drift_rows = read_csv_rows(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_prediction_drift.csv")
    feature_drift_rows = read_csv_rows(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_feature_drift.csv")
    regime_rows = read_csv_rows(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_regime_metrics.csv")

    readiness_status, readiness_reasons = status_from_gates(contract, summary_rows)
    if readiness_status == "ready_for_acceptance_review":
        gate_status, gate_reasons, marginal_reasons = acceptance_gate(rule, summary_rows, bootstrap_rows, regime_rows, threshold_rows)
        final_status = f"acceptance_{gate_status}"
    else:
        gate_status = "not_evaluated_until_sample_gates_pass"
        gate_reasons = ["sample gates are not met"]
        marginal_reasons = []
        final_status = readiness_status
    if final_status == "acceptance_pass":
        future_period_outcome = "pass"
        recommended_next_action = "freeze_as_paper_risk_overlay_candidate_after_human_review"
    elif final_status == "acceptance_marginal":
        future_period_outcome = "marginal"
        recommended_next_action = "collect_another_untouched_period_without_changing_model"
    elif final_status == "acceptance_fail":
        future_period_outcome = "fail"
        recommended_next_action = "archive_candidate_and_develop_challengers_from_new_development_lineage"
    else:
        future_period_outcome = "not_ready"
        recommended_next_action = "continue_accumulating_true_forward_shadow_evidence"

    summary = {
        "generated_at_iso": datetime.now(UTC).isoformat(),
        "cumulative_dir": str(cumulative_dir),
        "candidate_dir": str(candidate_dir),
        "contract_sha256": contract.get("contract_sha256"),
        "acceptance_rule_sha256": contract.get("acceptance_rule_sha256"),
        "readiness_status": readiness_status,
        "acceptance_gate_status": gate_status,
        "final_status": final_status,
        "future_period_outcome": future_period_outcome,
        "recommended_next_action": recommended_next_action,
        "calendar_days": safe_float(next((row.get("calendar_days") for row in summary_rows if str(row.get("target_column", "")).endswith("_h480")), contract.get("h480_calendar_days")), 0.0),
        "calendar_days_required": GOAL_MIN_CALENDAR_DAYS,
        "non_overlapping_h480_rows": safe_int(next((row.get("non_overlapping_h480_rows") for row in summary_rows if str(row.get("target_column", "")).endswith("_h480")), contract.get("h480_non_overlapping_rows")), 0),
        "non_overlapping_h480_required": GOAL_MIN_NON_OVERLAP_H480,
        "cumulative_decision_rows": safe_int(contract.get("cumulative_decision_rows"), 0),
        "cumulative_true_forward_decision_rows": safe_int(contract.get("cumulative_true_forward_decision_rows"), 0),
        "cumulative_backfill_or_replay_decision_rows": safe_int(contract.get("cumulative_backfill_or_replay_decision_rows"), 0),
        "cumulative_labeled_rows": safe_int(contract.get("cumulative_labeled_rows"), 0),
        "sample_gate_reasons": readiness_reasons,
        "acceptance_gate_reasons": gate_reasons,
        "marginal_reasons": marginal_reasons,
        "metric_extrema": best_or_worst_summary(summary_rows),
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "holdout_used_for_selection": False,
    }

    status_rows = [
        {
            "final_status": final_status,
            "future_period_outcome": future_period_outcome,
            "recommended_next_action": recommended_next_action,
            "readiness_status": readiness_status,
            "acceptance_gate_status": gate_status,
            "calendar_days": summary["calendar_days"],
            "calendar_days_required": GOAL_MIN_CALENDAR_DAYS,
            "non_overlapping_h480_rows": summary["non_overlapping_h480_rows"],
            "non_overlapping_h480_required": GOAL_MIN_NON_OVERLAP_H480,
            "true_forward_decisions": summary["cumulative_true_forward_decision_rows"],
            "backfill_or_replay_decisions": summary["cumulative_backfill_or_replay_decision_rows"],
            "labeled_rows": summary["cumulative_labeled_rows"],
            "reason_count": len(readiness_reasons) + len(gate_reasons) + len(marginal_reasons),
        }
    ]
    reason_rows = [{"reason_type": "sample_gate", "reason": reason} for reason in readiness_reasons]
    reason_rows.extend({"reason_type": "acceptance_gate", "reason": reason} for reason in gate_reasons)
    reason_rows.extend({"reason_type": "marginal_gate", "reason": reason} for reason in marginal_reasons)

    attach_implementation_lock(summary, candidate_dir=candidate_dir, cumulative_dir=cumulative_dir, feature_table=contract.get("feature_table", ""))
    summary["status_report_sha256"] = stable_hash(summary)
    status_rows[0]["logger_implementation_lock_sha256"] = summary.get("logger_implementation_lock_sha256", "")
    status_rows[0]["code_revision_lock_sha256"] = summary.get("code_revision_lock_sha256", "")

    write_json(out_dir / "rawseq_downside_risk_future_shadow_status.json", summary)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_status.csv", status_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_status_reasons.csv", reason_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_threshold_utility_snapshot.csv", threshold_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_prediction_drift_snapshot.csv", prediction_drift_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_feature_drift_snapshot.csv", feature_drift_rows)

    lines = [
        "Rawseq downside-risk future shadow status",
        f"Output: {out_dir}",
        f"Cumulative dir: {cumulative_dir}",
        f"Final status: {final_status}",
        f"Future period outcome: {future_period_outcome}",
        f"Recommended next action: {recommended_next_action}",
        f"Readiness status: {readiness_status}",
        f"Acceptance gate status: {gate_status}",
        f"Calendar days: {summary['calendar_days']} / {GOAL_MIN_CALENDAR_DAYS}",
        f"Non-overlapping h480 rows: {summary['non_overlapping_h480_rows']} / {GOAL_MIN_NON_OVERLAP_H480}",
        f"True-forward decisions: {summary['cumulative_true_forward_decision_rows']}",
        f"Backfill/replay decisions excluded: {summary['cumulative_backfill_or_replay_decision_rows']}",
        f"Labeled rows: {summary['cumulative_labeled_rows']}",
        "",
        "Reasons:",
    ]
    lines.extend(f"- {row['reason_type']}: {row['reason']}" for row in reason_rows)
    lines.extend(
        [
            "",
            "Safety: paper_only=true, orders=false, promotion=false, champion_mutation=false.",
            "Do not interpret accumulating status as pass/fail evidence.",
        ]
    )
    (out_dir / "rawseq_downside_risk_future_shadow_status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
