#!/usr/bin/env python3
"""Decision report for an isolated rawseq candidate shadow probe.

Read-only except for writing decision_summary.txt/csv into the probe folder.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_candidate_shadow_probes"

PROBE_DIR_ENV = os.getenv("RAWSEQ_PROBE_DIR", "").strip()
DECISION_THRESHOLD_BPS = float(os.getenv("RAWSEQ_DECISION_THRESHOLD_BPS", "0.1"))
DECISION_COST_BPS = float(os.getenv("RAWSEQ_DECISION_COST_BPS", "0.1"))
MIN_SELECTED_ROWS = int(os.getenv("RAWSEQ_MIN_SELECTED_ROWS", "300"))

COSTS_TO_COMPARE = [0.0, 0.05, 0.1, 0.25]
DISPLAY_COSTS = {0.05, 0.1, 0.25}
THRESHOLDS_TO_COMPARE = [0.0, 0.1, 0.2, 0.3]
WINDOWS_TO_COMPARE = [1.0, 3.0, 6.0, 12.0, 24.0]


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except Exception:
        return math.nan


def safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def fmt_float(value: Any, digits: int = 6) -> str:
    value = safe_float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def resolve_probe_dir() -> Path:
    if PROBE_DIR_ENV:
        path = Path(PROBE_DIR_ENV).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise SystemExit(f"RAWSEQ_PROBE_DIR does not exist: {path}")
        return path.resolve()

    if not PROBE_ROOT.exists():
        raise SystemExit(f"Probe root does not exist: {PROBE_ROOT}")
    candidates = [
        path
        for path in PROBE_ROOT.iterdir()
        if path.is_dir()
        and (path / "model_contract.json").exists()
    ]
    if not candidates:
        raise SystemExit(f"No complete probe folders found under {PROBE_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"Required probe artifact missing: {path}")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def nearest_row(frame: pd.DataFrame, threshold: float, cost: float) -> pd.Series | None:
    if frame.empty:
        return None
    subset = frame[
        frame["threshold_bps"].astype(float).sub(threshold).abs().lt(1e-12)
        & frame["cost_bps"].astype(float).sub(cost).abs().lt(1e-12)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def ratio(numerator: float, denominator: float) -> float:
    numerator = safe_float(numerator)
    denominator = safe_float(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return math.nan
    return numerator / denominator


def dip_to_cum_ratio(max_dip: float, cum_net: float) -> float:
    max_dip = safe_float(max_dip)
    cum_net = safe_float(cum_net)
    if not math.isfinite(max_dip) or not math.isfinite(cum_net):
        return math.nan
    return abs(max_dip) / max(cum_net, 1e-12)


def abs_ratio(numerator: float, denominator: float) -> float:
    numerator = safe_float(numerator)
    denominator = safe_float(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return math.nan
    return abs(numerator) / abs(denominator)


def rolling_metrics(rolling: pd.DataFrame, threshold: float, cost: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if rolling.empty:
        return out
    subset = rolling[
        rolling["threshold_bps"].astype(float).sub(threshold).abs().lt(1e-12)
        & rolling["cost_bps"].astype(float).sub(cost).abs().lt(1e-12)
    ].copy()
    for window in WINDOWS_TO_COMPARE:
        group = subset[subset["window_hours"].astype(float).sub(window).abs().lt(1e-12)]
        windows = int(len(group))
        positive = int((pd.to_numeric(group["cum_net_bps"], errors="coerce") > 0.0).sum()) if windows else 0
        total = safe_float(pd.to_numeric(group["cum_net_bps"], errors="coerce").sum()) if windows else math.nan
        worst = safe_float(pd.to_numeric(group["cum_net_bps"], errors="coerce").min()) if windows else math.nan
        selected_rows = (
            safe_int(pd.to_numeric(group["selected_rows"], errors="coerce").sum()) if windows else 0
        )
        out[f"{window:g}h"] = {
            "windows": windows,
            "positive_windows": positive,
            "positive_fraction": positive / windows if windows else math.nan,
            "total_cum_net_bps": total,
            "worst_cum_net_bps": worst,
            "selected_rows": selected_rows,
        }
    return out


def sensitivity_rows(summary: pd.DataFrame, threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cost_rows: list[dict[str, Any]] = []
    for cost in COSTS_TO_COMPARE:
        row = nearest_row(summary, threshold, cost)
        cost_rows.append(
            {
                "cost_bps": cost,
                "selected_rows": safe_int(row["selected_rows"]) if row is not None else 0,
                "avg_net_bps": safe_float(row["avg_net_bps"]) if row is not None else math.nan,
                "cum_net_bps": safe_float(row["cum_net_bps"]) if row is not None else math.nan,
                "win_rate_net": safe_float(row["win_rate_net"]) if row is not None else math.nan,
            }
        )

    threshold_rows: list[dict[str, Any]] = []
    for candidate_threshold in THRESHOLDS_TO_COMPARE:
        row = nearest_row(summary, candidate_threshold, DECISION_COST_BPS)
        threshold_rows.append(
            {
                "threshold_bps": candidate_threshold,
                "selected_rows": safe_int(row["selected_rows"]) if row is not None else 0,
                "avg_net_bps": safe_float(row["avg_net_bps"]) if row is not None else math.nan,
                "cum_net_bps": safe_float(row["cum_net_bps"]) if row is not None else math.nan,
                "win_rate_net": safe_float(row["win_rate_net"]) if row is not None else math.nan,
            }
        )
    return cost_rows, threshold_rows


def decide(
    contract_pass: bool,
    decision_row: pd.Series | None,
    roll: dict[str, dict[str, Any]],
    cost_rows: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not contract_pass:
        return "reject", ["contract audit failed"]
    if decision_row is None:
        return "reject", ["decision threshold/cost row missing"]

    selected_rows = safe_int(decision_row["selected_rows"])
    cum_net = safe_float(decision_row["cum_net_bps"])
    max_dip = safe_float(decision_row["max_dip_net_bps"])
    one_hour_positive = safe_float(roll.get("1h", {}).get("positive_fraction"))
    three_hour_positive = safe_float(roll.get("3h", {}).get("positive_fraction"))
    max_dip_ratio = dip_to_cum_ratio(max_dip, cum_net)
    cost_000 = next((row for row in cost_rows if abs(row["cost_bps"] - 0.0) < 1e-12), {})
    cost_025 = next((row for row in cost_rows if abs(row["cost_bps"] - 0.25) < 1e-12), {})
    cum_000 = safe_float(cost_000.get("cum_net_bps"))
    cum_025 = safe_float(cost_025.get("cum_net_bps"))
    cost_025_strong_negative = math.isfinite(cum_025) and cum_net > 0.0 and cum_025 < -0.5 * cum_net
    threshold_only_works_at_cost_zero = (
        math.isfinite(cum_000)
        and cum_000 > 0.0
        and math.isfinite(cum_net)
        and cum_net <= 0.0
    )

    if selected_rows < 100:
        reasons.append("selected_rows below 100 at decision threshold")
    if not math.isfinite(cum_net) or cum_net <= 0.0:
        reasons.append("cum_net is not positive at decision cost")
    if threshold_only_works_at_cost_zero:
        reasons.append("threshold only works at cost=0")
    if reasons:
        return "reject", reasons

    research_flags: list[str] = []
    if math.isfinite(max_dip_ratio) and max_dip_ratio > 1.0:
        research_flags.append("max_dip/cum_net ratio above 1")
    if not math.isfinite(one_hour_positive) or one_hour_positive < 0.50:
        research_flags.append("positive 1h rolling windows below 50%")
    if math.isfinite(cum_025) and cum_025 < 0.0:
        research_flags.append("0.25 bps cost is negative")

    clean = (
        selected_rows >= MIN_SELECTED_ROWS
        and cum_net > 0.0
        and math.isfinite(max_dip_ratio)
        and max_dip_ratio <= 1.0
        and math.isfinite(one_hour_positive)
        and one_hour_positive >= 0.50
        and math.isfinite(three_hour_positive)
        and three_hour_positive >= 0.50
        and not cost_025_strong_negative
    )
    if clean:
        return "clean_champion_candidate", ["all clean champion candidate gates passed"]

    reasons.extend(research_flags)
    if selected_rows < MIN_SELECTED_ROWS:
        reasons.append(f"selected_rows below {MIN_SELECTED_ROWS} clean-candidate gate")
    if cost_025_strong_negative:
        reasons.append("0.25 bps cost is strongly negative")
    if not reasons:
        reasons.append("passes reject gates but misses at least one clean-candidate gate")
    return "research_candidate", reasons


def render_text(
    probe_dir: Path,
    contract: dict[str, Any],
    contract_pass: bool,
    decision: str,
    reasons: list[str],
    decision_row: pd.Series | None,
    roll: dict[str, dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
) -> str:
    selected_rows = safe_int(decision_row["selected_rows"]) if decision_row is not None else 0
    avg_gross = safe_float(decision_row["avg_gross_bps"]) if decision_row is not None else math.nan
    avg_net = safe_float(decision_row["avg_net_bps"]) if decision_row is not None else math.nan
    cum_net = safe_float(decision_row["cum_net_bps"]) if decision_row is not None else math.nan
    win_rate = safe_float(decision_row["win_rate_net"]) if decision_row is not None else math.nan
    max_dip = safe_float(decision_row["max_dip_net_bps"]) if decision_row is not None else math.nan
    max_dip_ratio = dip_to_cum_ratio(max_dip, cum_net)

    lines = [
        "Rawseq Candidate Probe Decision",
        "",
        f"Decision: {decision}",
        f"Probe dir: {probe_dir}",
        f"Model path: {contract.get('model_path', '')}",
        (
            f"Contract: {contract.get('input_feature', '')} / {contract.get('hidden', '')} / "
            f"{contract.get('source_path_basename', '')} / seq={contract.get('seq_len', '')} "
            f"bucket={contract.get('bucket_seconds', '')} stride="
            f"{contract.get('input_stride', '1')}/{contract.get('output_stride', '1')}"
        ),
        f"Contract audit: {'PASS' if contract_pass else 'FAIL'}",
        "",
        "Decision Metrics",
        f"  threshold_bps: {DECISION_THRESHOLD_BPS:g}",
        f"  cost_bps: {DECISION_COST_BPS:g}",
        f"  min_selected_rows: {MIN_SELECTED_ROWS}",
        f"  selected_rows: {selected_rows}",
        f"  avg_gross_bps: {fmt_float(avg_gross)}",
        f"  avg_net_bps: {fmt_float(avg_net)}",
        f"  cum_net_bps: {fmt_float(cum_net)}",
        f"  win_rate_net: {fmt_float(win_rate)}",
        f"  max_dip_net_bps: {fmt_float(max_dip)}",
        f"  max_dip_to_cum_net_ratio: {fmt_float(max_dip_ratio)}",
        "",
        "Reasons",
    ]
    lines.extend(f"  - {reason}" for reason in reasons)
    lines += ["", "Rolling Windows"]
    for window in WINDOWS_TO_COMPARE:
        key = f"{window:g}h"
        item = roll.get(key, {})
        worst_ratio = dip_to_cum_ratio(item.get("worst_cum_net_bps"), cum_net)
        lines.append(
            f"  {key}: windows={safe_int(item.get('windows'))} "
            f"positive_windows={safe_int(item.get('positive_windows'))} "
            f"positive_fraction={fmt_float(item.get('positive_fraction'))} "
            f"total_cum_net={fmt_float(item.get('total_cum_net_bps'))} "
            f"worst_cum_net={fmt_float(item.get('worst_cum_net_bps'))} "
            f"selected_rows={safe_int(item.get('selected_rows'))} "
            f"worst_window_to_cum_net_ratio={fmt_float(worst_ratio)}"
        )

    lines += ["", "Cost Sensitivity"]
    for row in cost_rows:
        if row["cost_bps"] not in DISPLAY_COSTS:
            continue
        lines.append(
            f"  cost={row['cost_bps']:g}: rows={row['selected_rows']} "
            f"avg_net={fmt_float(row['avg_net_bps'])} "
            f"cum_net={fmt_float(row['cum_net_bps'])} "
            f"win={fmt_float(row['win_rate_net'])}"
        )

    lines += ["", "Threshold Sensitivity"]
    for row in threshold_rows:
        lines.append(
            f"  threshold={row['threshold_bps']:g}: rows={row['selected_rows']} "
            f"avg_net={fmt_float(row['avg_net_bps'])} "
            f"cum_net={fmt_float(row['cum_net_bps'])} "
            f"win={fmt_float(row['win_rate_net'])}"
        )

    lines += [
        "",
        "Safety",
        "  paper_only: true",
        "  training: false",
        "  champion_mutation: false",
        "  promotion: false",
        "  orders: false",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    probe_dir = resolve_probe_dir()
    contract_path = require_file(probe_dir / "model_contract.json")
    audit_path = require_file(probe_dir / "contract_audit.csv")

    contract = load_contract(contract_path)
    audit = pd.read_csv(audit_path, low_memory=False)
    if str(contract.get("output_label", "future_return_path")) != "future_return_path":
        require_file(probe_dir / "label_metric_summary.csv")
        require_file(probe_dir / "label_shape_audit.csv")
        row = {
            "probe_dir": str(probe_dir),
            "decision": "label_metric_only",
            "reasons": "direct_gt/inverse_gt trading decision is only valid for future_return_path",
            "model_path": contract.get("model_path", ""),
            "symbol": contract.get("symbol", ""),
            "venue": contract.get("venue", ""),
            "input_feature": contract.get("input_feature", ""),
            "source_path_basename": contract.get("source_path_basename", ""),
            "bucket_seconds": contract.get("bucket_seconds", ""),
            "seq_len": contract.get("seq_len", ""),
            "output_label": contract.get("output_label", ""),
            "output_dim": contract.get("output_dim", ""),
            "input_stride": contract.get("input_stride", "1"),
            "output_stride": contract.get("output_stride", "1"),
            "hidden": contract.get("hidden", ""),
            "contract_pass": (not audit.empty) and audit["status"].astype(str).str.upper().eq("PASS").all(),
            "decision_threshold_bps": DECISION_THRESHOLD_BPS,
            "decision_cost_bps": DECISION_COST_BPS,
            "selected_rows": 0,
            "cum_net_bps": math.nan,
            "paper_only": True,
            "training": False,
            "champion_mutation": False,
            "promotion": False,
            "orders": False,
        }
        text = (
            "Rawseq Candidate Probe Decision\n\n"
            f"Probe dir: {probe_dir}\n"
            "Decision: label_metric_only\n"
            f"Output label: {contract.get('output_label', '')}\n"
            "Reason: direct_gt/inverse_gt trading decision is only valid for future_return_path.\n\n"
            "Safety: paper-only. No training. No promotion. No champion mutation. No orders.\n"
        )
        (probe_dir / "decision_summary.txt").write_text(text, encoding="utf-8")
        pd.DataFrame([row]).to_csv(probe_dir / "decision_summary.csv", index=False)
        print(text)
        print(f"Decision summary: {probe_dir / 'decision_summary.txt'}")
        print(f"Decision CSV: {probe_dir / 'decision_summary.csv'}")
        return
    summary_path = require_file(probe_dir / "cost_threshold_summary.csv")
    rolling_path = require_file(probe_dir / "rolling_summary.csv")
    summary = pd.read_csv(summary_path, low_memory=False)
    rolling = pd.read_csv(rolling_path, low_memory=False)

    contract_pass = (not audit.empty) and audit["status"].astype(str).str.upper().eq("PASS").all()
    decision_row = nearest_row(summary, DECISION_THRESHOLD_BPS, DECISION_COST_BPS)
    roll = rolling_metrics(rolling, DECISION_THRESHOLD_BPS, DECISION_COST_BPS)
    cost_rows, threshold_rows = sensitivity_rows(summary, DECISION_THRESHOLD_BPS)
    decision, reasons = decide(contract_pass, decision_row, roll, cost_rows)

    selected_rows = safe_int(decision_row["selected_rows"]) if decision_row is not None else 0
    cum_net = safe_float(decision_row["cum_net_bps"]) if decision_row is not None else math.nan
    max_dip = safe_float(decision_row["max_dip_net_bps"]) if decision_row is not None else math.nan
    row = {
        "probe_dir": str(probe_dir),
        "decision": decision,
        "reasons": "; ".join(reasons),
        "model_path": contract.get("model_path", ""),
        "symbol": contract.get("symbol", ""),
        "venue": contract.get("venue", ""),
        "input_feature": contract.get("input_feature", ""),
        "source_path_basename": contract.get("source_path_basename", ""),
        "bucket_seconds": contract.get("bucket_seconds", ""),
        "seq_len": contract.get("seq_len", ""),
        "input_stride": contract.get("input_stride", "1"),
        "output_stride": contract.get("output_stride", "1"),
        "hidden": contract.get("hidden", ""),
        "contract_pass": contract_pass,
        "decision_threshold_bps": DECISION_THRESHOLD_BPS,
        "decision_cost_bps": DECISION_COST_BPS,
        "min_selected_rows": MIN_SELECTED_ROWS,
        "selected_rows": selected_rows,
        "avg_gross_bps": safe_float(decision_row["avg_gross_bps"]) if decision_row is not None else math.nan,
        "avg_net_bps": safe_float(decision_row["avg_net_bps"]) if decision_row is not None else math.nan,
        "cum_net_bps": cum_net,
        "win_rate_net": safe_float(decision_row["win_rate_net"]) if decision_row is not None else math.nan,
        "max_dip_net_bps": max_dip,
        "max_dip_to_cum_net_ratio": dip_to_cum_ratio(max_dip, cum_net),
    }
    for window in WINDOWS_TO_COMPARE:
        key = f"{window:g}h"
        item = roll.get(key, {})
        row[f"positive_{key}_window_fraction"] = safe_float(item.get("positive_fraction"))
        row[f"{key}_windows"] = safe_int(item.get("windows"))
        row[f"{key}_positive_windows"] = safe_int(item.get("positive_windows"))
        row[f"{key}_total_cum_net_bps"] = safe_float(item.get("total_cum_net_bps"))
        row[f"{key}_selected_rows"] = safe_int(item.get("selected_rows"))
        row[f"worst_{key}_window_cum_net_bps"] = safe_float(item.get("worst_cum_net_bps"))
        row[f"worst_{key}_window_to_cum_net_ratio"] = dip_to_cum_ratio(
            item.get("worst_cum_net_bps"), cum_net
        )
    for cost_row in cost_rows:
        label = str(cost_row["cost_bps"]).replace(".", "p")
        row[f"cost_{label}_cum_net_bps"] = cost_row["cum_net_bps"]
    for threshold_row in threshold_rows:
        label = str(threshold_row["threshold_bps"]).replace(".", "p")
        row[f"threshold_{label}_cum_net_bps"] = threshold_row["cum_net_bps"]

    text = render_text(
        probe_dir,
        contract,
        contract_pass,
        decision,
        reasons,
        decision_row,
        roll,
        cost_rows,
        threshold_rows,
    )
    (probe_dir / "decision_summary.txt").write_text(text, encoding="utf-8")
    pd.DataFrame([row]).to_csv(probe_dir / "decision_summary.csv", index=False)
    print(text)
    print(f"Decision summary: {probe_dir / 'decision_summary.txt'}")
    print(f"Decision CSV: {probe_dir / 'decision_summary.csv'}")
    print("Safety: read-only except report outputs. No training. No promotion. No orders.")


if __name__ == "__main__":
    main()
