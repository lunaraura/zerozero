#!/usr/bin/env python3
"""Compare frozen rawseq shadow candidates by latest forward paper run."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHADOW_ROOT = Path(
    os.getenv(
        "RAWSEQ_SHADOW_COMPARISON_ROOT",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_shadow_candidates"),
    )
)
OUTPUT_PATH_ENV = os.getenv("RAWSEQ_SHADOW_COMPARISON_OUTPUT_PATH", "").strip()
MIN_SELECTED_ROWS = int(float(os.getenv("RAWSEQ_FORWARD_MIN_SELECTED_ROWS", "100")))
COMPARISON_MODE = os.getenv("RAWSEQ_FORWARD_COMPARISON_MODE", "best_available").strip().lower()
INCLUDE_REPLAY = os.getenv("RAWSEQ_FORWARD_INCLUDE_REPLAY_IN_COMPARISON", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
STATUS_PRIORITY = {
    "tracking_ok": 0,
    "degraded": 1,
    "insufficient_sample": 2,
    "failed_forward": 3,
}


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_first_csv_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return next(reader, {})


def truthy(value: Any) -> bool:
    return safe_str(value).lower() in {"1", "true", "yes", "y", "on"}


def forward_evidence_type(run_dir: Path, summary: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> str:
    summary = summary or read_first_csv_row(run_dir / "forward_summary.csv")
    metadata = metadata or load_json(run_dir / "forward_run_metadata.json")
    if truthy(summary.get("replay")) or truthy(metadata.get("replay")) or safe_str(metadata.get("replay_mode")) == "replay_window":
        return "replay"
    if safe_str(summary.get("evidence_type")):
        return safe_str(summary.get("evidence_type"))
    if safe_str(metadata.get("evidence_type")):
        return safe_str(metadata.get("evidence_type"))
    if truthy(summary.get("run_is_incremental")) or truthy(metadata.get("run_is_incremental")):
        if truthy(summary.get("forward_cutoff_enforced")) or truthy(metadata.get("forward_cutoff_enforced")):
            return "true_forward"
        return "backfill_risk"
    return "legacy_unknown"


def policy_sign_status(summary: dict[str, Any], metadata: dict[str, Any]) -> tuple[str, str, str]:
    version = safe_str(
        metadata.get("policy_sign_semantics_version")
        or summary.get("policy_sign_semantics_version")
    )
    policy = safe_str(metadata.get("policy") or summary.get("policy")).lower()
    if version == "gross_bps_policy_multiplier_v1":
        return version, "true", ""
    if policy == "inverse_gt":
        return version, "false", "legacy_inverse_gt_forward_scoring_may_have_used_wrong_sign"
    return version, "unknown", "legacy_run_missing_policy_sign_semantics_version"


def latest_forward_run(shadow_dir: Path) -> Path | None:
    root = shadow_dir / "forward_paper_runs"
    if not root.exists():
        return None
    runs = [path for path in root.iterdir() if path.is_dir() and (path / "forward_summary.csv").exists()]
    if not runs:
        return None
    return sorted(runs, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)[0]


def forward_runs(shadow_dir: Path) -> list[Path]:
    root = shadow_dir / "forward_paper_runs"
    if not root.exists():
        return []
    runs = [path for path in root.iterdir() if path.is_dir() and (path / "forward_summary.csv").exists()]
    return sorted(runs, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def selected_rows(summary: dict[str, Any]) -> int:
    return int(safe_float(summary.get("forward_selected_rows") or summary.get("selected_rows"), 0.0))


def key_value(value: Any) -> str:
    text = safe_str(value)
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def duplicate_shadow_group_key(row: dict[str, Any]) -> str:
    fields = [
        "model_path",
        "threshold_bps",
        "input_feature",
        "ma_window",
        "hidden",
        "seed",
        "seq_len",
        "bucket_seconds",
        "input_stride",
        "output_stride",
    ]
    parts = []
    for field in fields:
        value = key_value(row.get(field))
        if field == "model_path":
            value = value.replace("/", "\\").lower()
        parts.append(value)
    return "|".join(parts)


def max_dip_bps(values: pd.Series) -> float:
    net = pd.to_numeric(values, errors="coerce").dropna()
    if net.empty:
        return math.nan
    cumulative = net.cumsum()
    peak = cumulative.cummax()
    return float((cumulative - peak).min())


def classify_forward(selected: int, cum_net: float, max_dip: float, registry_cum: float) -> tuple[str, bool, float]:
    ratio = cum_net / registry_cum if math.isfinite(cum_net) and registry_cum > 0 else math.nan
    if selected >= 20 and math.isfinite(cum_net) and cum_net < 0:
        return "failed_forward", selected < 100, ratio
    if selected < 100:
        return "insufficient_sample", False, ratio
    if math.isfinite(cum_net) and cum_net < 0:
        return "failed_forward", False, ratio
    if math.isfinite(ratio) and ratio < 0.25:
        return "degraded", False, ratio
    if math.isfinite(cum_net) and cum_net > 0 and (not math.isfinite(max_dip) or abs(max_dip) <= max(cum_net * 2.0, 1e-9)):
        return "tracking_ok", False, ratio
    return "degraded", False, ratio


def aggregate_summary(run_dirs: list[Path], registry_cum: float) -> dict[str, Any]:
    frames = []
    for run_dir in run_dirs:
        if not INCLUDE_REPLAY and forward_evidence_type(run_dir) != "true_forward":
            continue
        path = run_dir / "forward_labeled_results.csv"
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if "timestamp" in frame.columns:
            frame["source_forward_run"] = str(run_dir)
            frames.append(frame)
    if not frames:
        return {}
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["timestamp"] = pd.to_numeric(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")
    data = data.drop_duplicates(subset=["timestamp"], keep="last")
    if "selected" in data.columns:
        selected_mask = data["selected"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        selected_mask = pd.Series(False, index=data.index)
    selected = data[selected_mask].copy()
    net = pd.to_numeric(selected.get("net_bps", pd.Series(dtype=float)), errors="coerce").dropna()
    gross = pd.to_numeric(selected.get("rawseq_path_actual_horizon_return_bps", pd.Series(dtype=float)), errors="coerce").dropna()
    selected_count = int(len(selected))
    cum_net = float(net.sum()) if len(net) else math.nan
    avg_net = float(net.mean()) if len(net) else math.nan
    win_rate = float((net > 0).mean()) if len(net) else math.nan
    max_dip = max_dip_bps(net)
    status, preliminary, ratio = classify_forward(selected_count, cum_net, max_dip, registry_cum)
    return {
        "forward_status": status,
        "forward_status_preliminary": str(preliminary),
        "forward_selected_rows": str(selected_count),
        "forward_cum_net_bps": str(cum_net),
        "forward_avg_net_bps": str(avg_net),
        "forward_win_rate": str(win_rate),
        "forward_max_dip_bps": str(max_dip),
        "forward_vs_registry_cum_ratio": str(ratio),
        "avg_gross_bps": str(float(gross.mean()) if len(gross) else math.nan),
    }


def normalize_forward_summary(summary: dict[str, Any], registry_cum: float) -> dict[str, Any]:
    if not summary:
        return summary
    normalized = dict(summary)
    selected = selected_rows(normalized)
    cum_net = safe_float(normalized.get("forward_cum_net_bps") or normalized.get("cumulative_net_bps"), math.nan)
    avg_net = safe_float(normalized.get("forward_avg_net_bps") or normalized.get("avg_net_bps"), math.nan)
    win_rate = safe_float(normalized.get("forward_win_rate") or normalized.get("win_rate_net"), math.nan)
    max_dip = safe_float(normalized.get("forward_max_dip_bps") or normalized.get("max_dip_bps"), math.nan)
    if not safe_str(normalized.get("forward_status")):
        status, preliminary, ratio = classify_forward(selected, cum_net, max_dip, registry_cum)
        normalized["forward_status"] = status
        normalized["forward_status_preliminary"] = str(preliminary)
        normalized["forward_vs_registry_cum_ratio"] = str(ratio)
    if not safe_str(normalized.get("forward_selected_rows")):
        normalized["forward_selected_rows"] = str(selected)
    if not safe_str(normalized.get("forward_cum_net_bps")):
        normalized["forward_cum_net_bps"] = str(cum_net)
    if not safe_str(normalized.get("forward_avg_net_bps")):
        normalized["forward_avg_net_bps"] = str(avg_net)
    if not safe_str(normalized.get("forward_win_rate")):
        normalized["forward_win_rate"] = str(win_rate)
    if not safe_str(normalized.get("forward_max_dip_bps")):
        normalized["forward_max_dip_bps"] = str(max_dip)
    return normalized


def choose_forward_run(run_dirs: list[Path]) -> tuple[Path | None, dict[str, Any], str, dict[str, Any], int, int]:
    if not run_dirs:
        return None, {}, "no_forward_runs", {}, 0, 0
    run_summaries = [(run_dir, read_first_csv_row(run_dir / "forward_summary.csv")) for run_dir in run_dirs]
    latest_run, latest_summary = run_summaries[0]
    latest_rows = selected_rows(latest_summary)
    comparable = [
        (run_dir, summary)
        for run_dir, summary in run_summaries
        if INCLUDE_REPLAY or forward_evidence_type(run_dir, summary) == "true_forward"
    ]
    if not comparable:
        best_run, best_summary = max(run_summaries, key=lambda item: selected_rows(item[1]))
        return best_run, best_summary, "no_true_forward_runs_available_diagnostic_only", latest_summary, selected_rows(best_summary), latest_rows

    best_run, best_summary = max(comparable, key=lambda item: selected_rows(item[1]))
    best_rows = selected_rows(best_summary)

    if COMPARISON_MODE == "latest":
        if INCLUDE_REPLAY or forward_evidence_type(latest_run, latest_summary) == "true_forward":
            return latest_run, latest_summary, "latest_mode", latest_summary, best_rows, latest_rows
        return best_run, best_summary, "latest_mode_excluded_replay_or_backfill_used_best_true_forward", latest_summary, best_rows, latest_rows
    eligible = [(run_dir, summary) for run_dir, summary in comparable if selected_rows(summary) >= MIN_SELECTED_ROWS]
    if eligible:
        chosen_run, chosen_summary = eligible[0]
        return chosen_run, chosen_summary, f"latest_with_selected_rows_ge_{MIN_SELECTED_ROWS}", latest_summary, best_rows, latest_rows
    return best_run, best_summary, "largest_selected_rows_below_minimum", latest_summary, best_rows, latest_rows


def output_base() -> Path:
    if OUTPUT_PATH_ENV:
        path = resolve_path(OUTPUT_PATH_ENV)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path.with_suffix("")
        path.mkdir(parents=True, exist_ok=True)
        return path / "shadow_forward_comparison"
    out_dir = PROJECT_ROOT / "data" / "research" / "rawseq_shadow_forward_comparisons" / f"shadow_forward_comparison_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "shadow_forward_comparison"


def candidate_row(shadow_dir: Path) -> dict[str, Any]:
    provenance = load_json(shadow_dir / "provenance.json")
    contract = provenance.get("contract") if isinstance(provenance.get("contract"), dict) else {}
    registry = provenance.get("metrics") if isinstance(provenance.get("metrics"), dict) else {}
    run_dirs = forward_runs(shadow_dir)
    latest_run = run_dirs[0] if run_dirs else None
    registry_cum = safe_float(registry.get("fixed_0_10_cum_net"), math.nan)
    if COMPARISON_MODE == "aggregate":
        summary = aggregate_summary(run_dirs, registry_cum)
        run_dir = latest_run
        latest_summary = read_first_csv_row(latest_run / "forward_summary.csv") if latest_run else {}
        best_rows = max([selected_rows(read_first_csv_row(path / "forward_summary.csv")) for path in run_dirs], default=0)
        latest_rows = selected_rows(latest_summary)
        selection_reason = "aggregate_all_non_duplicate_timestamps"
    else:
        run_dir, summary, selection_reason, latest_summary, best_rows, latest_rows = choose_forward_run(run_dirs)
        summary = normalize_forward_summary(summary, registry_cum)
        latest_summary = normalize_forward_summary(latest_summary, registry_cum)
    metadata = load_json(run_dir / "forward_run_metadata.json") if run_dir else {}
    selected_evidence_type = forward_evidence_type(run_dir, summary, metadata) if run_dir else "none"
    latest_evidence_type = forward_evidence_type(latest_run, latest_summary) if latest_run else "none"
    evidence_counts = Counter(forward_evidence_type(path) for path in run_dirs)
    sign_version, sign_valid, sign_warning = policy_sign_status(summary, metadata)
    forward_status = safe_str(summary.get("forward_status")) or "missing_forward"
    model_contract = provenance.get("model_contract") if isinstance(provenance.get("model_contract"), dict) else {}
    model_path = safe_str(provenance.get("model_path") or model_contract.get("model_path"))
    return {
        "shadow_dir": str(shadow_dir),
        "candidate": shadow_dir.name,
        "forward_run_selection_reason": selection_reason,
        "forward_run_count": len(run_dirs),
        "best_available_selected_rows": best_rows,
        "latest_selected_rows": latest_rows,
        "latest_forward_run": str(latest_run or ""),
        "selected_forward_run": str(run_dir or ""),
        "selected_forward_evidence_type": selected_evidence_type,
        "latest_forward_evidence_type": latest_evidence_type,
        "true_forward_run_count": evidence_counts.get("true_forward", 0),
        "replay_run_count": evidence_counts.get("replay", 0),
        "backfill_or_legacy_run_count": sum(
            count
            for key, count in evidence_counts.items()
            if key not in {"true_forward", "replay"}
        ),
        "excluded_replay_from_ranking": str(not INCLUDE_REPLAY),
        "policy_sign_semantics_version": sign_version,
        "policy_sign_valid": sign_valid,
        "policy_sign_warning": sign_warning,
        "dry_run": safe_str(metadata.get("dry_run")),
        "status_priority": STATUS_PRIORITY.get(forward_status, 9),
        "forward_status": forward_status,
        "forward_status_preliminary": safe_str(summary.get("forward_status_preliminary")),
        "symbol": safe_str(contract.get("symbol")),
        "venue": safe_str(contract.get("venue")),
        "input_feature": safe_str(contract.get("input_feature")),
        "ma_window": safe_str(contract.get("ma_window")),
        "hidden": safe_str(contract.get("hidden")),
        "seed": safe_str(contract.get("seed")),
        "seq_len": safe_str(contract.get("seq_len") or model_contract.get("seq_len")),
        "bucket_seconds": safe_str(contract.get("bucket_seconds") or model_contract.get("bucket_seconds")),
        "input_stride": safe_str(contract.get("input_stride") or model_contract.get("input_stride") or "1"),
        "output_stride": safe_str(contract.get("output_stride") or model_contract.get("output_stride") or "1"),
        "model_path": model_path,
        "threshold_bps": safe_str(provenance.get("threshold_bps")),
        "forward_selected_rows": safe_str(summary.get("forward_selected_rows") or summary.get("selected_rows")),
        "forward_cum_net_bps": safe_str(summary.get("forward_cum_net_bps") or summary.get("cumulative_net_bps")),
        "forward_avg_net_bps": safe_str(summary.get("forward_avg_net_bps") or summary.get("avg_net_bps")),
        "forward_win_rate": safe_str(summary.get("forward_win_rate") or summary.get("win_rate_net")),
        "forward_max_dip_bps": safe_str(summary.get("forward_max_dip_bps") or summary.get("max_dip_bps")),
        "forward_vs_registry_cum_ratio": safe_str(summary.get("forward_vs_registry_cum_ratio")),
        "latest_forward_selected_rows": safe_str(latest_summary.get("forward_selected_rows") or latest_summary.get("selected_rows")),
        "latest_forward_cum_net_bps": safe_str(latest_summary.get("forward_cum_net_bps") or latest_summary.get("cumulative_net_bps")),
        "latest_forward_status": safe_str(latest_summary.get("forward_status")),
        "registry_selected_rows": safe_str(summary.get("registry_selected_rows") or registry.get("selected_rows")),
        "registry_fixed_0_10_cum_net": safe_str(summary.get("registry_fixed_0_10_cum_net") or registry.get("fixed_0_10_cum_net")),
        "registry_fixed_0_25_cum_net": safe_str(summary.get("registry_fixed_0_25_cum_net") or registry.get("fixed_0_25_cum_net")),
        "registry_half_spread_plus_0_05_cum_net": safe_str(summary.get("registry_half_spread_plus_0_05_cum_net") or registry.get("half_spread_plus_0_05_cum_net")),
        "registry_conservative_missing_liquidity_penalty_cum_net": safe_str(
            summary.get("registry_conservative_missing_liquidity_penalty_cum_net")
            or registry.get("conservative_missing_liquidity_penalty_cum_net")
        ),
        "registry_max_dip_to_cum_net_ratio": safe_str(summary.get("registry_max_dip_to_cum_net_ratio") or registry.get("max_dip_to_cum_net_ratio")),
        "registry_positive_12h_window_fraction": safe_str(summary.get("registry_positive_12h_window_fraction") or registry.get("positive_12h_window_fraction")),
        "registry_positive_24h_window_fraction": safe_str(summary.get("registry_positive_24h_window_fraction") or registry.get("positive_24h_window_fraction")),
        "paper_only": "True",
        "orders": "False",
        "promotion": "False",
        "champion_mutation": "False",
    }


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("status_priority", 9)),
            -safe_float(row.get("forward_cum_net_bps"), -1e18),
            -safe_float(row.get("forward_selected_rows"), -1e18),
            safe_float(row.get("forward_max_dip_bps"), 1e18),
        ),
    )


def group_best_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    return (
        safe_float(row.get("forward_selected_rows"), -1.0),
        safe_float(row.get("forward_cum_net_bps"), -1e18),
    )


def apply_duplicate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = duplicate_shadow_group_key(row)
        row["duplicate_shadow_group_key"] = key
        groups.setdefault(key, []).append(row)

    for key, group_rows in groups.items():
        best = max(group_rows, key=group_best_sort_key)
        for row in group_rows:
            row["duplicate_shadow_count"] = len(group_rows)
            row["group_best_candidate"] = "True" if row is best else "False"
    return rows


def grouped_best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rank_rows([row for row in rows if safe_str(row.get("group_best_candidate")) == "True"])


def write_text(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped_rows = grouped_best_rows(rows)
    duplicate_groups = [row for row in grouped_rows if safe_float(row.get("duplicate_shadow_count"), 0.0) > 1]
    lines = [
        "Shadow Candidate Forward Comparison",
        "",
        f"Created at: {now_stamp()}",
        f"Shadow root: {SHADOW_ROOT}",
        f"Mode: {COMPARISON_MODE}",
        f"Minimum selected rows: {MIN_SELECTED_ROWS}",
        f"Include replay/backfill in ranking: {INCLUDE_REPLAY}",
        f"Frozen candidates: {len(rows)}",
        f"Grouped candidates: {len(grouped_rows)}",
        f"Duplicate groups: {len(duplicate_groups)}",
        "",
        "Safety:",
        "  paper_only=true",
        "  orders=false",
        "  promotion=false",
        "  champion_mutation=false",
        "",
        "Candidates:",
    ]
    for row in rows:
        lines.append(
            "  "
            f"{row['forward_status']} prelim={row['forward_status_preliminary']} "
            f"rows={row['forward_selected_rows']} cum={row['forward_cum_net_bps']} "
            f"ratio={row['forward_vs_registry_cum_ratio']} "
            f"evidence={row['selected_forward_evidence_type']} "
            f"policy_sign_valid={row.get('policy_sign_valid', '')} "
            f"group_best={row['group_best_candidate']} dup_count={row['duplicate_shadow_count']} "
            f"reason={row['forward_run_selection_reason']} runs={row['forward_run_count']} "
            f"true_forward_runs={row['true_forward_run_count']} replay_runs={row['replay_run_count']} "
            f"backfill_or_legacy_runs={row['backfill_or_legacy_run_count']} "
            f"{row['input_feature']} ma={row['ma_window']} h={row['hidden']} seed={row['seed']} "
            f"selected_run={row['selected_forward_run']} latest_run={row['latest_forward_run']}"
        )
    lines.append("")
    lines.append("Replay/backfill separation:")
    for row in rows:
        if row.get("selected_forward_evidence_type") != "true_forward" or safe_float(row.get("replay_run_count"), 0.0) > 0:
            lines.append(
                "  "
                f"{row['candidate']}: selected_evidence={row['selected_forward_evidence_type']} "
                f"latest_evidence={row['latest_forward_evidence_type']} "
                f"true_forward={row['true_forward_run_count']} replay={row['replay_run_count']} "
                f"backfill_or_legacy={row['backfill_or_legacy_run_count']} "
                f"reason={row['forward_run_selection_reason']}"
            )
    lines.append("")
    lines.append("Policy sign warnings:")
    invalid_sign_rows = [row for row in rows if safe_str(row.get("policy_sign_valid")).lower() == "false"]
    if invalid_sign_rows:
        for row in invalid_sign_rows:
            lines.append(
                "  "
                f"{row['candidate']}: {row.get('policy_sign_warning', '')} "
                f"selected_run={row.get('selected_forward_run', '')}"
            )
    else:
        lines.append("  none")
    lines.append("")
    lines.append("Grouped candidates:")
    for row in grouped_rows:
        duplicate_note = " duplicate_group" if safe_float(row.get("duplicate_shadow_count"), 0.0) > 1 else ""
        lines.append(
            "  "
            f"{row['forward_status']} rows={row['forward_selected_rows']} cum={row['forward_cum_net_bps']} "
            f"count={row['duplicate_shadow_count']}{duplicate_note} "
            f"{row['input_feature']} ma={row['ma_window']} h={row['hidden']} seed={row['seed']} "
            f"threshold={row['threshold_bps']} model={row['model_path']}"
        )
    lines.append("")
    lines.append("Duplicate frozen candidate groups:")
    if duplicate_groups:
        for row in duplicate_groups:
            members = [candidate["candidate"] for candidate in rows if candidate.get("duplicate_shadow_group_key") == row.get("duplicate_shadow_group_key")]
            lines.append(
                "  "
                f"best={row['candidate']} count={row['duplicate_shadow_count']} "
                f"rows={row['forward_selected_rows']} cum={row['forward_cum_net_bps']} "
                f"members={';'.join(members)}"
            )
    else:
        lines.append("  none")
    lines.extend(
        [
            "",
            "Warning:",
            "  No champion creation/mutation from this report.",
            "  This is paper-only forward comparison research.",
            "  Promotion requires a separate explicit freeze/audit step.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = resolve_path(SHADOW_ROOT)
    if not root.exists():
        raise SystemExit(f"Shadow candidate root not found: {root}")
    rows = [candidate_row(path) for path in sorted(root.iterdir()) if path.is_dir()]
    if not rows:
        raise SystemExit(f"No shadow candidate folders found under {root}")
    rows = apply_duplicate_groups(rows)
    rows = rank_rows(rows)
    grouped_rows = grouped_best_rows(rows)
    base = output_base()
    csv_path = base.with_suffix(".csv")
    grouped_csv_path = base.with_name(base.name + "_grouped").with_suffix(".csv")
    txt_path = base.with_suffix(".txt")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(grouped_rows).to_csv(grouped_csv_path, index=False)
    write_text(txt_path, rows)
    print("Shadow candidate forward comparison complete")
    print(f"Rows: {len(rows)}")
    print(f"Grouped rows: {len(grouped_rows)}")
    print(f"CSV: {csv_path}")
    print(f"Grouped CSV: {grouped_csv_path}")
    print(f"TXT: {txt_path}")
    print("Safety: no training. No model mutation. No promotion. No champion mutation. No orders.")
    print(pd.DataFrame(rows).head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
