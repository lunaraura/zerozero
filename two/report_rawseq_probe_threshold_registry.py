#!/usr/bin/env python3
"""Build a threshold-aware registry of full-source rawseq candidate probes.

Paper-only report generator. It does not train, promote, mutate champion
folders, or place orders.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_ROOT = Path(
    os.getenv(
        "RAWSEQ_PROBE_ROOT",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_candidate_shadow_probes"),
    )
)
if not PROBE_ROOT.is_absolute():
    PROBE_ROOT = PROJECT_ROOT / PROBE_ROOT

OUTPUT_ROOT = Path(
    os.getenv(
        "RAWSEQ_REGISTRY_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_probe_registry"),
    )
)
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

MIN_SELECTED_ROWS = int(float(os.getenv("RAWSEQ_MIN_SELECTED_ROWS", "300")))
DECISION_COST_BPS = float(os.getenv("RAWSEQ_DECISION_COST_BPS", "0.1"))
STATUS_PRIORITY = {
    "clean_shadow_candidate": 0,
    "robust_research_candidate": 1,
    "fragile_research_candidate": 2,
    "reject": 3,
}
CONTRACT_FIELDS = [
    "symbol",
    "venue",
    "source_path_basename",
    "input_feature",
    "ma_window",
    "hidden",
    "seq_len",
    "bucket_seconds",
    "input_stride",
    "output_stride",
    "seed",
    "model_path",
]
DYNAMIC_SCENARIOS = {
    "fixed_0_05_bps": "fixed_0_05_cum_net",
    "fixed_0_10_bps": "fixed_0_10_cum_net",
    "fixed_0_15_bps": "fixed_0_15_cum_net",
    "fixed_0_25_bps": "fixed_0_25_cum_net",
    "half_spread_plus_0_05_bps": "half_spread_plus_0_05_cum_net",
    "half_spread_plus_depth_penalty": "half_spread_plus_depth_penalty_cum_net",
    "half_spread_plus_depth_and_imbalance_penalty": "half_spread_plus_depth_and_imbalance_penalty_cum_net",
    "conservative_missing_liquidity_penalty": "conservative_missing_liquidity_penalty_cum_net",
}
OUTPUT_COLUMNS = [
    "status",
    "status_priority",
    "rank_score",
    "probe_threshold_key",
    "duplicate_count",
    "prior_reject_thresholds",
    "prior_reject_count_same_probe",
    "contract_reject_count_all_thresholds",
    "best_threshold_for_probe",
    "best_status_for_probe",
    "status_explanation",
    "probe_dir",
    "threshold_bps",
    "decision_cost_bps",
    *CONTRACT_FIELDS,
    "decision",
    "decision_source",
    "dynamic_source",
    "selected_rows",
    "avg_net_bps",
    "cum_net_bps",
    "win_rate_net",
    "max_dip_net_bps",
    "max_dip_to_cum_net_ratio",
    "positive_1h_window_fraction",
    "positive_3h_window_fraction",
    "positive_6h_window_fraction",
    "positive_12h_window_fraction",
    "positive_24h_window_fraction",
    "fixed_0_05_cum_net",
    "fixed_0_10_cum_net",
    "fixed_0_15_cum_net",
    "fixed_0_25_cum_net",
    "half_spread_plus_0_05_cum_net",
    "half_spread_plus_depth_penalty_cum_net",
    "half_spread_plus_depth_and_imbalance_penalty_cum_net",
    "conservative_missing_liquidity_penalty_cum_net",
    "conservative_dynamic_survival",
    "conservative_missing_liquidity_penalty_positive",
    "fixed_0_25_positive",
    "selected_rows_ge_900",
    "rejection_reasons",
    "paper_only",
    "training",
    "champion_mutation",
    "promotion",
    "orders",
]


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def safe_int(value: Any, default: int = 0) -> int:
    value = safe_float(value)
    return int(value) if math.isfinite(value) else default


def fmt(value: Any, digits: int = 6) -> str:
    value = safe_float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "nan"


def norm_threshold(value: Any) -> str:
    number = safe_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.10g}"


def threshold_from_name(path: Path) -> str:
    match = re.search(r"threshold_([0-9]+(?:\.[0-9]+)?)", path.stem)
    return norm_threshold(match.group(1)) if match else ""


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def set_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if safe_str(value):
        target[key] = value


def contract_from_probe(probe_dir: Path) -> dict[str, Any]:
    contract = read_json(probe_dir / "model_contract.json")
    if contract and not safe_str(contract.get("source_path_basename")):
        source_path = safe_str(contract.get("source_path"))
        if source_path:
            contract["source_path_basename"] = Path(source_path).name
    if not safe_str(contract.get("ma_window")):
        search_text = f"{probe_dir.name} {safe_str(contract.get('model_path'))}"
        match = re.search(r"ma(?:_distance|_slope)?_?ma?([0-9]+)", search_text)
        if not match:
            match = re.search(r"_ma([0-9]+)_", search_text)
        if match:
            contract["ma_window"] = match.group(1)
    return {field: safe_str(contract.get(field)) for field in CONTRACT_FIELDS}


def parse_decision_csv(probe_dir: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for path in sorted(probe_dir.glob("decision_summary*.csv")):
        for csv_row in read_csv_rows(path):
            threshold = norm_threshold(
                csv_row.get("decision_threshold_bps")
                or csv_row.get("threshold_bps")
                or threshold_from_name(path)
            )
            if not threshold:
                continue
            row = decisions.setdefault(threshold, {})
            row["decision_source"] = path.name
            mapping = {
                "decision": "decision",
                "decision_cost_bps": "decision_cost_bps",
                "selected_rows": "selected_rows",
                "avg_net_bps": "avg_net_bps",
                "cum_net_bps": "cum_net_bps",
                "win_rate_net": "win_rate_net",
                "max_dip_net_bps": "max_dip_net_bps",
                "max_dip_to_cum_net_ratio": "max_dip_to_cum_net_ratio",
                "positive_1h_window_fraction": "positive_1h_window_fraction",
                "positive_3h_window_fraction": "positive_3h_window_fraction",
                "positive_6h_window_fraction": "positive_6h_window_fraction",
                "positive_12h_window_fraction": "positive_12h_window_fraction",
                "positive_24h_window_fraction": "positive_24h_window_fraction",
            }
            for src, dst in mapping.items():
                set_if_present(row, dst, csv_row.get(src))
            for field in CONTRACT_FIELDS:
                set_if_present(row, field, csv_row.get(field))
            for cost_key, output_key in [
                ("cost_0p05_cum_net_bps", "fixed_0_05_cum_net"),
                ("cost_0p1_cum_net_bps", "fixed_0_10_cum_net"),
                ("cost_0p15_cum_net_bps", "fixed_0_15_cum_net"),
                ("cost_0p25_cum_net_bps", "fixed_0_25_cum_net"),
            ]:
                set_if_present(row, output_key, csv_row.get(cost_key))
    return decisions


def parse_decision_txt(path: Path) -> tuple[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    threshold = threshold_from_name(path)
    row: dict[str, Any] = {"decision_source": path.name}
    for line in text:
        stripped = line.strip()
        if stripped.startswith("Decision:"):
            row["decision"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Model path:"):
            row["model_path"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Contract:"):
            parts = [part.strip() for part in stripped.split(":", 1)[1].split("/")]
            if len(parts) >= 3:
                row["input_feature"] = parts[0]
                row["hidden"] = parts[1]
                row["source_path_basename"] = parts[2]
            match = re.search(r"seq=([0-9.]+)\s+bucket=([0-9.]+)\s+stride=([0-9.]+)/([0-9.]+)", stripped)
            if match:
                row["seq_len"], row["bucket_seconds"], row["input_stride"], row["output_stride"] = match.groups()
        elif re.match(r"^(threshold_bps|cost_bps|selected_rows|avg_net_bps|cum_net_bps|win_rate_net|max_dip_net_bps|max_dip_to_cum_net_ratio):", stripped):
            key, value = stripped.split(":", 1)
            if key == "threshold_bps":
                threshold = norm_threshold(value)
            elif key == "cost_bps":
                row["decision_cost_bps"] = value.strip()
            else:
                row[key] = value.strip()
        else:
            rolling = re.match(
                r"^([0-9]+)h:\s+windows=([0-9]+)\s+positive_windows=([0-9]+)\s+"
                r"positive_fraction=([-+0-9.eE]+)",
                stripped,
            )
            if rolling:
                hours = rolling.group(1)
                row[f"positive_{hours}h_window_fraction"] = rolling.group(4)
            cost = re.match(r"^cost=([0-9.]+):.*cum_net=([-+0-9.eE]+)", stripped)
            if cost:
                key = {
                    "0.05": "fixed_0_05_cum_net",
                    "0.1": "fixed_0_10_cum_net",
                    "0.10": "fixed_0_10_cum_net",
                    "0.15": "fixed_0_15_cum_net",
                    "0.25": "fixed_0_25_cum_net",
                }.get(cost.group(1))
                if key:
                    row[key] = cost.group(2)
    return threshold, row


def parse_decision_txts(probe_dir: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for path in sorted(probe_dir.glob("decision_summary_threshold_*.txt")):
        threshold, parsed = parse_decision_txt(path)
        if threshold:
            decisions.setdefault(threshold, {}).update(parsed)
    return decisions


def parse_cost_threshold_csv(probe_dir: Path, decision_cost_bps: float) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(probe_dir.glob("cost_threshold_summary.csv")):
        for csv_row in read_csv_rows(path):
            threshold = norm_threshold(csv_row.get("threshold_bps"))
            if not threshold:
                continue
            row = rows.setdefault(threshold, {})
            cost = safe_float(csv_row.get("cost_bps"))
            if abs(cost - decision_cost_bps) < 1e-9:
                row["cost_threshold_source"] = path.name
                for key in [
                    "selected_rows",
                    "avg_net_bps",
                    "cum_net_bps",
                    "win_rate_net",
                    "max_dip_net_bps",
                ]:
                    set_if_present(row, key, csv_row.get(key))
                cum = safe_float(csv_row.get("cum_net_bps"))
                dip = safe_float(csv_row.get("max_dip_net_bps"))
                if math.isfinite(cum) and cum > 0.0 and math.isfinite(dip):
                    row["max_dip_to_cum_net_ratio"] = abs(dip) / cum
                for field in CONTRACT_FIELDS:
                    set_if_present(row, field, csv_row.get(field))
            scenario_key = {
                0.05: "fixed_0_05_cum_net",
                0.10: "fixed_0_10_cum_net",
                0.15: "fixed_0_15_cum_net",
                0.25: "fixed_0_25_cum_net",
            }.get(round(cost, 2))
            if scenario_key:
                set_if_present(row, scenario_key, csv_row.get("cum_net_bps"))
    return rows


def parse_dynamic_csv(probe_dir: Path) -> dict[str, dict[str, Any]]:
    dynamic: dict[str, dict[str, Any]] = {}
    for path in sorted(probe_dir.glob("dynamic_cost_summary*.csv")):
        for csv_row in read_csv_rows(path):
            threshold = norm_threshold(csv_row.get("threshold_bps") or threshold_from_name(path))
            scenario = safe_str(csv_row.get("scenario"))
            if not threshold or scenario not in DYNAMIC_SCENARIOS:
                continue
            row = dynamic.setdefault(threshold, {})
            row["dynamic_source"] = path.name
            row[DYNAMIC_SCENARIOS[scenario]] = csv_row.get("cum_net_bps")
            if scenario == "fixed_0_10_bps":
                for key in [
                    "selected_rows",
                    "avg_net_bps",
                    "cum_net_bps",
                    "win_rate_net",
                    "max_dip_net_bps",
                ]:
                    set_if_present(row, key, csv_row.get(key))
            for field in CONTRACT_FIELDS:
                set_if_present(row, field, csv_row.get(field))
    return dynamic


def parse_dynamic_txt(path: Path) -> tuple[str, dict[str, Any]]:
    threshold = threshold_from_name(path)
    row: dict[str, Any] = {"dynamic_source": path.name}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("Threshold bps:"):
            threshold = norm_threshold(stripped.split(":", 1)[1])
            continue
        if stripped.startswith("Contract:"):
            parts = [part.strip() for part in stripped.split(":", 1)[1].split("/")]
            if len(parts) >= 3:
                row["input_feature"] = parts[0]
                row["hidden"] = parts[1]
                row["source_path_basename"] = parts[2]
            match = re.search(r"seq=([0-9.]+)\s+bucket=([0-9.]+)\s+stride=([0-9.]+)/([0-9.]+)", stripped)
            if match:
                row["seq_len"], row["bucket_seconds"], row["input_stride"], row["output_stride"] = match.groups()
            continue
        parts = stripped.split()
        if len(parts) >= 8 and parts[0] in DYNAMIC_SCENARIOS:
            row[DYNAMIC_SCENARIOS[parts[0]]] = parts[5]
            if parts[0] == "fixed_0_10_bps":
                row["selected_rows"] = parts[1]
                row["avg_net_bps"] = parts[4]
                row["cum_net_bps"] = parts[5]
                row["win_rate_net"] = parts[6]
                row["max_dip_net_bps"] = parts[7]
    return threshold, row


def parse_dynamic_txts(probe_dir: Path) -> dict[str, dict[str, Any]]:
    dynamic: dict[str, dict[str, Any]] = {}
    for path in sorted(probe_dir.glob("dynamic_cost_summary_threshold_*.txt")):
        threshold, parsed = parse_dynamic_txt(path)
        if threshold:
            dynamic.setdefault(threshold, {}).update(parsed)
    return dynamic


def merge_threshold_maps(*maps: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in maps:
        for threshold, row in item.items():
            target = merged.setdefault(threshold, {})
            for key, value in row.items():
                if safe_str(value) or key not in target:
                    target[key] = value
    return merged


def classify(row: dict[str, Any]) -> tuple[str, list[str]]:
    fixed_010 = safe_float(row.get("fixed_0_10_cum_net"), safe_float(row.get("cum_net_bps"), 0.0))
    half_plus = safe_float(row.get("half_spread_plus_0_05_cum_net"), math.nan)
    selected_rows = safe_int(row.get("selected_rows"), 0)
    dip_ratio = safe_float(row.get("max_dip_to_cum_net_ratio"), math.inf)
    rolling_12h = safe_float(row.get("positive_12h_window_fraction"), 0.0)
    rolling_24h = safe_float(row.get("positive_24h_window_fraction"), 0.0)
    reasons: list[str] = []
    if fixed_010 <= 0.0:
        reasons.append("fixed_0_10_cum_net_nonpositive")
    if selected_rows < MIN_SELECTED_ROWS:
        reasons.append("selected_rows_lt_min")
    if not math.isfinite(half_plus) or half_plus <= 0.0:
        reasons.append("half_spread_plus_0_05_cum_net_nonpositive_or_missing")
    if not math.isfinite(dip_ratio):
        reasons.append("max_dip_to_cum_net_ratio_missing")
    elif dip_ratio > 1.0:
        reasons.append("max_dip_to_cum_net_ratio_gt_1")
    if rolling_12h < 0.5:
        reasons.append("rolling_12h_positive_fraction_lt_0_5")
    if rolling_24h < 0.5:
        reasons.append("rolling_24h_positive_fraction_lt_0_5")

    if fixed_010 <= 0.0 or selected_rows < MIN_SELECTED_ROWS:
        return "reject", reasons
    if (
        half_plus > 0.0
        and dip_ratio <= 1.0
        and rolling_12h >= 0.5
        and rolling_24h >= 0.5
    ):
        return "clean_shadow_candidate", reasons
    if half_plus > 0.0 and dip_ratio <= 2.0:
        return "robust_research_candidate", reasons
    return "fragile_research_candidate", reasons


def conservative_survival(row: dict[str, Any]) -> float:
    values = [
        safe_float(row.get(key))
        for key in [
            "half_spread_plus_0_05_cum_net",
            "half_spread_plus_depth_penalty_cum_net",
            "half_spread_plus_depth_and_imbalance_penalty_cum_net",
            "conservative_missing_liquidity_penalty_cum_net",
        ]
    ]
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else math.nan


def rank_score(row: dict[str, Any]) -> float:
    fixed_010 = safe_float(row.get("fixed_0_10_cum_net"), 0.0)
    fixed_025 = safe_float(row.get("fixed_0_25_cum_net"), math.nan)
    conservative_missing = safe_float(row.get("conservative_missing_liquidity_penalty_cum_net"), math.nan)
    conservative = safe_float(row.get("conservative_dynamic_survival"), -1_000_000.0)
    dip_ratio = safe_float(row.get("max_dip_to_cum_net_ratio"), 999.0)
    rolling_12h = safe_float(row.get("positive_12h_window_fraction"), 0.0)
    rolling_24h = safe_float(row.get("positive_24h_window_fraction"), 0.0)
    selected_rows = safe_float(row.get("selected_rows"), 0.0)
    conservative_bonus = 500.0 if math.isfinite(conservative_missing) and conservative_missing > 0.0 else 0.0
    fixed_025_bonus = 250.0 if math.isfinite(fixed_025) and fixed_025 > 0.0 else 0.0
    row_depth_bonus = 150.0 if selected_rows >= 900.0 else 0.0
    return (
        conservative * 1.5
        + fixed_010
        + conservative_bonus
        + fixed_025_bonus
        + row_depth_bonus
        - min(dip_ratio, 20.0) * 50.0
        + (rolling_12h + rolling_24h) * 250.0
        + min(selected_rows / 300.0, 10.0) * 10.0
    )


def contract_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field, "") for field in [
        "input_feature",
        "ma_window",
        "hidden",
        "seq_len",
        "bucket_seconds",
        "input_stride",
        "output_stride",
        "source_path_basename",
    ])


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not PROBE_ROOT.exists():
        raise SystemExit(f"Probe root does not exist: {PROBE_ROOT}")
    for probe_dir in sorted(path for path in PROBE_ROOT.iterdir() if path.is_dir()):
        contract = contract_from_probe(probe_dir)
        decision_csv = parse_decision_csv(probe_dir)
        decision_txt = parse_decision_txts(probe_dir)
        cost_threshold = parse_cost_threshold_csv(probe_dir, DECISION_COST_BPS)
        dynamic_csv = parse_dynamic_csv(probe_dir)
        dynamic_txt = parse_dynamic_txts(probe_dir)
        threshold_rows = merge_threshold_maps(cost_threshold, decision_csv, decision_txt, dynamic_csv, dynamic_txt)
        for threshold, data in sorted(threshold_rows.items(), key=lambda item: safe_float(item[0], 999.0)):
            row = {"probe_dir": str(probe_dir), "threshold_bps": threshold, **contract}
            row.update(data)
            for field in CONTRACT_FIELDS:
                if not safe_str(row.get(field)):
                    row[field] = contract.get(field, "")
            row["fixed_0_10_cum_net"] = safe_float(
                row.get("fixed_0_10_cum_net"), safe_float(row.get("cum_net_bps"), math.nan)
            )
            row["decision_cost_bps"] = safe_float(row.get("decision_cost_bps"), DECISION_COST_BPS)
            row["conservative_dynamic_survival"] = conservative_survival(row)
            status, reasons = classify(row)
            row["status"] = status
            row["status_priority"] = STATUS_PRIORITY[status]
            row["rejection_reasons"] = ";".join(reasons)
            row["conservative_missing_liquidity_penalty_positive"] = (
                safe_float(row.get("conservative_missing_liquidity_penalty_cum_net"), math.nan) > 0.0
            )
            row["fixed_0_25_positive"] = safe_float(row.get("fixed_0_25_cum_net"), math.nan) > 0.0
            row["selected_rows_ge_900"] = safe_float(row.get("selected_rows"), 0.0) >= 900.0
            row["rank_score"] = rank_score(row)
            row["probe_threshold_key"] = probe_threshold_key(row)
            row["paper_only"] = True
            row["training"] = False
            row["champion_mutation"] = False
            row["promotion"] = False
            row["orders"] = False
            rows.append(row)
    return rows


def probe_threshold_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            safe_str(row.get("probe_dir")),
            safe_str(row.get("model_path")),
            norm_threshold(row.get("threshold_bps")),
            norm_threshold(row.get("decision_cost_bps")),
        ]
    )


def source_priority(row: dict[str, Any]) -> int:
    decision_source = safe_str(row.get("decision_source"))
    dynamic_source = safe_str(row.get("dynamic_source"))
    if "threshold_" in decision_source:
        return 3
    if "threshold_" in dynamic_source:
        return 2
    if decision_source:
        return 1
    return 0


def deduplicate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row["probe_threshold_key"] = probe_threshold_key(row)
        grouped.setdefault(row["probe_threshold_key"], []).append(row)
    deduped: list[dict[str, Any]] = []
    removed = 0
    for key, candidates in grouped.items():
        candidates = sorted(
            candidates,
            key=lambda item: (
                source_priority(item),
                -safe_float(item.get("status_priority"), 999.0),
                safe_float(item.get("rank_score"), -1_000_000.0),
            ),
            reverse=True,
        )
        keep = candidates[0]
        keep["probe_threshold_key"] = key
        keep["duplicate_count"] = len(candidates)
        removed += max(0, len(candidates) - 1)
        deduped.append(keep)
    return deduped, removed


def probe_key(row: dict[str, Any]) -> tuple[str, str]:
    return safe_str(row.get("probe_dir")), safe_str(row.get("model_path"))


def row_sort_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        safe_float(row.get("status_priority"), 999.0),
        not bool(row.get("conservative_missing_liquidity_penalty_positive")),
        not bool(row.get("fixed_0_25_positive")),
        not bool(row.get("selected_rows_ge_900")),
        -safe_float(row.get("conservative_dynamic_survival"), -1_000_000.0),
        -safe_float(row.get("fixed_0_10_cum_net"), -1_000_000.0),
        safe_float(row.get("max_dip_to_cum_net_ratio"), 999.0),
        -safe_float(row.get("positive_12h_window_fraction"), -1_000_000.0),
        -safe_float(row.get("positive_24h_window_fraction"), -1_000_000.0),
        -safe_float(row.get("selected_rows"), 0.0),
    )


def annotate_threshold_history(rows: list[dict[str, Any]]) -> None:
    by_probe: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_contract: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        by_probe.setdefault(probe_key(row), []).append(row)
        by_contract.setdefault(contract_key(row), []).append(row)

    best_by_probe: dict[tuple[str, str], dict[str, Any]] = {}
    for key, probe_rows in by_probe.items():
        best_by_probe[key] = sorted(probe_rows, key=row_sort_tuple)[0]

    contract_reject_counts = {
        key: sum(1 for row in contract_rows if row.get("status") == "reject")
        for key, contract_rows in by_contract.items()
    }

    for row in rows:
        threshold = safe_float(row.get("threshold_bps"), math.nan)
        probe_rows = by_probe.get(probe_key(row), [])
        prior_rejects = sorted(
            safe_float(candidate.get("threshold_bps"), math.nan)
            for candidate in probe_rows
            if candidate.get("status") == "reject"
            and math.isfinite(threshold)
            and safe_float(candidate.get("threshold_bps"), math.inf) < threshold
        )
        row["prior_reject_thresholds"] = ";".join(f"{value:.10g}" for value in prior_rejects)
        row["prior_reject_count_same_probe"] = len(prior_rejects)
        row["contract_reject_count_all_thresholds"] = contract_reject_counts.get(contract_key(row), 0)
        best = best_by_probe.get(probe_key(row), row)
        row["best_threshold_for_probe"] = norm_threshold(best.get("threshold_bps"))
        row["best_status_for_probe"] = safe_str(best.get("status"))
        status = safe_str(row.get("status"))
        reasons = safe_str(row.get("rejection_reasons"))
        if prior_rejects and status != "reject":
            row["status_explanation"] = (
                f"{status}; threshold_sensitive_revived_after_lower_rejects="
                f"{row['prior_reject_thresholds']}; local_reasons={reasons or 'none'}"
            )
        elif status == "reject":
            row["status_explanation"] = f"reject; reasons={reasons or 'none'}"
        else:
            row["status_explanation"] = f"{status}; local_reasons={reasons or 'none'}"


def sorted_registry(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    numeric_columns = [
        "status_priority",
        "rank_score",
        "threshold_bps",
        "decision_cost_bps",
        "duplicate_count",
        "prior_reject_count_same_probe",
        "contract_reject_count_all_thresholds",
        "best_threshold_for_probe",
        "selected_rows",
        "fixed_0_10_cum_net",
        "fixed_0_25_cum_net",
        "conservative_missing_liquidity_penalty_cum_net",
        "conservative_dynamic_survival",
        "max_dip_to_cum_net_ratio",
        "positive_12h_window_fraction",
        "positive_24h_window_fraction",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(
        [
            "status_priority",
            "conservative_missing_liquidity_penalty_positive",
            "fixed_0_25_positive",
            "selected_rows_ge_900",
            "conservative_dynamic_survival",
            "fixed_0_10_cum_net",
            "max_dip_to_cum_net_ratio",
            "positive_12h_window_fraction",
            "positive_24h_window_fraction",
            "selected_rows",
        ],
        ascending=[True, False, False, False, False, False, True, False, False, False],
    )[OUTPUT_COLUMNS]


def best_per_contract(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = []
    seen: set[tuple[Any, ...]] = set()
    for _, row in frame.iterrows():
        key = contract_key(row.to_dict())
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return pd.DataFrame(rows)


def render_row(row: pd.Series) -> str:
    contract = (
        f"{row.get('input_feature', '')}/ma{row.get('ma_window', '')}/h{row.get('hidden', '')}/"
        f"seq{row.get('seq_len', '')}/stride{row.get('input_stride', '')}-{row.get('output_stride', '')}"
    )
    return (
        f"  {row.get('status', '')} threshold={fmt(row.get('threshold_bps'), 3)} "
        f"rows={int(safe_float(row.get('selected_rows'), 0))} fixed010={fmt(row.get('fixed_0_10_cum_net'))} "
        f"fixed025={fmt(row.get('fixed_0_25_cum_net'))} "
        f"half+005={fmt(row.get('half_spread_plus_0_05_cum_net'))} "
        f"missing_liq={fmt(row.get('conservative_missing_liquidity_penalty_cum_net'))} "
        f"conservative={fmt(row.get('conservative_dynamic_survival'))} "
        f"dip_ratio={fmt(row.get('max_dip_to_cum_net_ratio'), 3)} "
        f"roll12/24={fmt(row.get('positive_12h_window_fraction'), 3)}/{fmt(row.get('positive_24h_window_fraction'), 3)} "
        f"prior_rejects={row.get('prior_reject_thresholds', '') or 'none'} "
        f"contract={contract} probe={Path(str(row.get('probe_dir', ''))).name}"
    )


def render_section(lines: list[str], title: str, frame: pd.DataFrame, limit: int = 20) -> None:
    lines += ["", title]
    if frame.empty:
        lines.append("  none")
        return
    for _, row in frame.head(limit).iterrows():
        lines.append(render_row(row))


def render_text(frame: pd.DataFrame, output_dir: Path, top_frame: pd.DataFrame, duplicate_rows_removed: int) -> str:
    lines = [
        "Rawseq Probe Threshold Registry",
        "",
        f"Probe root: {PROBE_ROOT}",
        f"Output dir: {output_dir}",
        f"Rows after dedupe: {len(frame)}",
        f"Duplicate rows removed: {duplicate_rows_removed}",
        f"Min selected rows: {MIN_SELECTED_ROWS}",
        f"Decision cost bps: {DECISION_COST_BPS}",
    ]
    render_section(lines, "1. Clean Shadow Candidates, Deduped", frame[frame["status"] == "clean_shadow_candidate"])
    render_section(lines, "2. Robust Research Candidates, Deduped", frame[frame["status"] == "robust_research_candidate"])
    revived = frame[
        (frame["status"].isin(["clean_shadow_candidate", "robust_research_candidate", "fragile_research_candidate"]))
        & (pd.to_numeric(frame["prior_reject_count_same_probe"], errors="coerce").fillna(0) > 0)
    ]
    render_section(lines, "3. Candidates Revived At Stricter Threshold After Lower-Threshold Reject", revived)
    lines += ["", "4. Best Threshold Per Probe"]
    best_probe_rows = []
    seen_probe_keys: set[tuple[str, str]] = set()
    for _, row in frame.iterrows():
        key = (safe_str(row.get("probe_dir")), safe_str(row.get("model_path")))
        if key in seen_probe_keys:
            continue
        seen_probe_keys.add(key)
        best_probe_rows.append(row)
    if not best_probe_rows:
        lines.append("  none")
    else:
        for row in best_probe_rows[:30]:
            lines.append(render_row(row))
    lines += ["", "5. Duplicate Rows Removed"]
    lines.append(f"  duplicate_rows_removed: {duplicate_rows_removed}")
    duplicated = frame[pd.to_numeric(frame["duplicate_count"], errors="coerce").fillna(0) > 1]
    if duplicated.empty:
        lines.append("  no duplicate probe-threshold keys retained")
    else:
        for _, row in duplicated.head(20).iterrows():
            lines.append(
                f"  kept duplicate_count={int(safe_float(row.get('duplicate_count'), 0))} "
                f"key={row.get('probe_threshold_key', '')} source={row.get('decision_source', '')}"
            )
    render_section(lines, "Rejects, Deduped", frame[frame["status"] == "reject"], limit=20)
    lines += ["", "Best Candidate Per Contract"]
    best_contracts = best_per_contract(frame)
    if best_contracts.empty:
        lines.append("  none")
    else:
        for _, row in best_contracts.head(30).iterrows():
            lines.append(render_row(row))
    lines += ["", "Top Shadow Candidates"]
    if top_frame.empty:
        lines.append("  none")
    else:
        for _, row in top_frame.head(20).iterrows():
            lines.append(render_row(row))
    lines += [
        "",
        "6. Warning",
        "  Threshold-specific clean status is not a champion freeze.",
        "  Do not create or mutate champion folders from this report alone.",
        "  This is a paper-only shadow registry. Promotion requires a separate explicit freeze/audit step.",
        "",
        f"Registry CSV: {output_dir / 'rawseq_probe_threshold_registry.csv'}",
        f"Top candidates CSV: {output_dir / 'top_shadow_candidates.csv'}",
        "Safety: no training. No champion mutation. No promotion. No orders.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = build_rows()
    rows, duplicate_rows_removed = deduplicate_rows(rows)
    annotate_threshold_history(rows)
    output_dir = OUTPUT_ROOT / f"rawseq_probe_threshold_registry_{now_stamp()}_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = sorted_registry(rows)
    top_frame = frame[frame["status"] != "reject"].copy()
    frame.to_csv(output_dir / "rawseq_probe_threshold_registry.csv", index=False)
    top_frame.to_csv(output_dir / "top_shadow_candidates.csv", index=False)
    text = render_text(frame, output_dir, top_frame, duplicate_rows_removed)
    (output_dir / "rawseq_probe_threshold_registry.txt").write_text(text, encoding="utf-8")
    (output_dir / "top_shadow_candidates.txt").write_text(
        "\n".join(["Top Shadow Candidates", ""] + [render_row(row) for _, row in top_frame.head(50).iterrows()])
        + "\n",
        encoding="utf-8",
    )
    print(text)
    print("Top 20 Registry Rows")
    preview_columns = [
        "status",
        "threshold_bps",
        "input_feature",
        "ma_window",
        "hidden",
        "seed",
        "selected_rows",
        "fixed_0_10_cum_net",
        "half_spread_plus_0_05_cum_net",
        "fixed_0_25_cum_net",
        "conservative_missing_liquidity_penalty_cum_net",
        "conservative_dynamic_survival",
        "max_dip_to_cum_net_ratio",
        "positive_12h_window_fraction",
        "positive_24h_window_fraction",
        "prior_reject_thresholds",
        "duplicate_count",
        "probe_dir",
    ]
    print(frame[preview_columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
