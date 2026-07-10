#!/usr/bin/env python3
"""Aggregate rawseq I/O contract evidence into a research leaderboard."""

from __future__ import annotations

import csv
import math
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(
    os.getenv(
        "RAWSEQ_IO_LEADERBOARD_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_io_contract_leaderboards"),
    )
)
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

REGISTRY_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_probe_registry"
FORWARD_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_shadow_forward_comparisons"
WALKFORWARD_ROOT = PROJECT_ROOT / "data" / "rawseq_walkforward"
DISCOVERY_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_io_contract_discovery_batches"

STATUS_PRIORITY = {
    "tracking_ok": 4,
    "degraded": 2,
    "insufficient_sample": 1,
    "failed_forward": -3,
    "missing_forward": 0,
}


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def safe_int(value: Any) -> int:
    value = safe_float(value, 0.0)
    return int(value) if math.isfinite(value) else 0


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def latest_files(root: Path, pattern: str, limit: int | None = None) -> list[Path]:
    if not root.exists():
        return []
    files = sorted(root.rglob(pattern), key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)
    return files[:limit] if limit else files


def slug_part(value: Any, fallback: str = "NA") -> str:
    text = safe_str(value) or fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or fallback


def norm_int_text(value: Any, default: str) -> str:
    text = safe_str(value)
    if not text:
        return default
    try:
        number = float(text)
    except Exception:
        return text
    if math.isfinite(number) and abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return text


def norm_ma_text(value: Any) -> str:
    text = safe_str(value)
    if not text or text.lower() == "nan":
        return "NA"
    return norm_int_text(text, text)


def infer_ma_window(row: dict[str, Any] | pd.Series) -> tuple[str, str, str]:
    get = row.get if hasattr(row, "get") else lambda key, default="": default
    explicit = norm_ma_text(get("ma_window"))
    if explicit != "NA":
        return explicit, "explicit_ma_window", explicit

    registry = norm_ma_text(get("registry_ma_window"))
    if registry != "NA":
        return registry, "registry_ma_window", registry

    path_fields = [
        "model_path",
        "probe_dir",
        "best_candidate_probe_dir",
        "best_candidate_model_path",
        "survivor_representative_model_path",
        "representative_archived_model_path",
        "contract_slug",
        "io_contract_slug",
    ]
    for field in path_fields:
        text = safe_str(get(field))
        if not text:
            continue
        match = re.search(r"(?:^|[_\\/.-])ma(?:_distance_)?(?P<ma>\d{2,4})(?:[_\\/.-]|$)", text, re.IGNORECASE)
        if match:
            value = norm_int_text(match.group("ma"), match.group("ma"))
            return value, f"path_pattern:{field}", value
    return "NA", "missing", ""


def contract_key(row: dict[str, Any] | pd.Series) -> str:
    get = row.get if hasattr(row, "get") else lambda key, default="": default
    input_feature = safe_str(get("input_feature"))
    ma_window, _, _ = infer_ma_window(row) if input_feature == "ma_distance" else (norm_ma_text(get("ma_window")), "non_ma_distance", "")
    hidden = safe_str(get("hidden")).replace(",", "x") or "NA"
    seq_len = norm_int_text(get("seq_len") or get("payload_seq_len"), "60")
    bucket_seconds = norm_int_text(get("bucket_seconds") or get("payload_bucket_seconds"), "10")
    input_stride = norm_int_text(get("input_stride") or get("payload_input_stride"), "1")
    output_stride = norm_int_text(get("output_stride") or get("payload_output_stride"), "1")
    return "|".join([input_feature, ma_window, hidden, seq_len, bucket_seconds, input_stride, output_stride])


def raw_contract_key(row: dict[str, Any] | pd.Series) -> str:
    get = row.get if hasattr(row, "get") else lambda key, default="": default
    input_feature = safe_str(get("input_feature"))
    ma_window = norm_ma_text(get("ma_window"))
    hidden = safe_str(get("hidden")).replace(",", "x") or "NA"
    seq_len = norm_int_text(get("seq_len") or get("payload_seq_len"), "60")
    bucket_seconds = norm_int_text(get("bucket_seconds") or get("payload_bucket_seconds"), "10")
    input_stride = norm_int_text(get("input_stride") or get("payload_input_stride"), "1")
    output_stride = norm_int_text(get("output_stride") or get("payload_output_stride"), "1")
    return "|".join([input_feature, ma_window, hidden, seq_len, bucket_seconds, input_stride, output_stride])


def contract_slug_from_key(key: str) -> str:
    feature, ma, hidden, seq_len, bucket, istride, ostride = key.split("|")
    return "_".join(
        [
            slug_part(feature, "feature"),
            f"ma{slug_part(ma)}",
            f"h{slug_part(hidden)}",
            f"seq{slug_part(seq_len)}",
            f"b{slug_part(bucket)}",
            f"is{slug_part(istride)}",
            f"os{slug_part(ostride)}",
        ]
    )


def empty_record(key: str) -> dict[str, Any]:
    feature, ma, hidden, seq_len, bucket, istride, ostride = key.split("|")
    return {
        "io_contract_key": key,
        "io_contract_slug": contract_slug_from_key(key),
        "input_feature": feature,
        "ma_window": "" if ma == "NA" else ma,
        "ma_window_source": "key",
        "ma_window_inferred": "" if ma == "NA" else ma,
        "ma_window_missing_warning": feature == "ma_distance" and ma == "NA",
        "hidden": hidden.replace("x", ","),
        "seq_len": seq_len,
        "bucket_seconds": bucket,
        "input_stride": istride,
        "output_stride": ostride,
        "clean_shadow_candidates": 0,
        "robust_research_candidates": 0,
        "fragile_research_candidates": 0,
        "reject_candidates": 0,
        "best_fixed_0_10_cum_net": math.nan,
        "best_half_spread_plus_0_05_cum_net": math.nan,
        "best_conservative_missing_liquidity_cum_net": math.nan,
        "best_selected_rows": 0,
        "best_drawdown_ratio": math.nan,
        "best_rolling_12h_fraction": math.nan,
        "best_rolling_24h_fraction": math.nan,
        "best_candidate_status": "",
        "best_candidate_probe_dir": "",
        "best_candidate_model_path": "",
        "best_candidate_threshold_bps": "",
        "frozen_shadow_candidate_exists": False,
        "forward_status": "missing_forward",
        "forward_selected_rows": 0,
        "forward_cum_net_bps": math.nan,
        "forward_vs_registry_cum_ratio": math.nan,
        "forward_run": "",
        "walkforward_runs": 0,
        "walkforward_windows": 0,
        "walkforward_total_rows": 0,
        "walkforward_total_cum_bps": math.nan,
        "walkforward_positive_window_fraction": math.nan,
        "survivor_status": "",
        "survivor_positive_window_fraction": math.nan,
        "survivor_total_cum_bps": math.nan,
        "survivor_representative_model_path": "",
        "discovery_runs": 0,
        "discovery_statuses": "",
        "_raw_contract_keys": set(),
    }


def get_record(records: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    if key not in records:
        records[key] = empty_record(key)
    return records[key]


def apply_ma_metadata(record: dict[str, Any], row: dict[str, Any] | pd.Series) -> None:
    raw_keys = record.get("_raw_contract_keys")
    if isinstance(raw_keys, set):
        raw_keys.add(raw_contract_key(row))
    if safe_str(record.get("input_feature")) != "ma_distance":
        return
    ma, source, inferred = infer_ma_window(row)
    if ma != "NA":
        record["ma_window"] = ma
        record["ma_window_inferred"] = inferred
        if safe_str(record.get("ma_window_source")) in {"", "key", "missing"} or source.startswith("explicit"):
            record["ma_window_source"] = source
        record["ma_window_missing_warning"] = False
        record["io_contract_key"] = contract_key(row)
        record["io_contract_slug"] = contract_slug_from_key(record["io_contract_key"])
    elif not safe_str(record.get("ma_window")):
        record["ma_window_source"] = "missing"
        record["ma_window_missing_warning"] = True


def update_max(record: dict[str, Any], field: str, value: Any) -> None:
    value = safe_float(value)
    current = safe_float(record.get(field))
    if math.isfinite(value) and (not math.isfinite(current) or value > current):
        record[field] = value


def ingest_registries(records: dict[str, dict[str, Any]], sources: list[str]) -> None:
    for path in latest_files(REGISTRY_ROOT, "rawseq_probe_threshold_registry.csv"):
        frame = read_csv(path)
        if frame.empty:
            continue
        sources.append(str(path))
        for _, row in frame.iterrows():
            key = contract_key(row)
            if not key.startswith("|"):
                record = get_record(records, key)
                apply_ma_metadata(record, row)
            else:
                continue
            status = safe_str(row.get("status"))
            if status == "clean_shadow_candidate":
                record["clean_shadow_candidates"] += 1
            elif status == "robust_research_candidate":
                record["robust_research_candidates"] += 1
            elif status == "fragile_research_candidate":
                record["fragile_research_candidates"] += 1
            elif status == "reject":
                record["reject_candidates"] += 1
            update_max(record, "best_fixed_0_10_cum_net", row.get("fixed_0_10_cum_net"))
            update_max(record, "best_half_spread_plus_0_05_cum_net", row.get("half_spread_plus_0_05_cum_net"))
            update_max(record, "best_conservative_missing_liquidity_cum_net", row.get("conservative_missing_liquidity_penalty_cum_net"))
            update_max(record, "best_selected_rows", row.get("selected_rows"))
            update_max(record, "best_rolling_12h_fraction", row.get("positive_12h_window_fraction"))
            update_max(record, "best_rolling_24h_fraction", row.get("positive_24h_window_fraction"))
            ratio = safe_float(row.get("max_dip_to_cum_net_ratio"))
            current_ratio = safe_float(record.get("best_drawdown_ratio"))
            if math.isfinite(ratio) and (not math.isfinite(current_ratio) or ratio < current_ratio):
                record["best_drawdown_ratio"] = ratio
            fixed = safe_float(row.get("fixed_0_10_cum_net"), -math.inf)
            current_best = safe_float(record.get("_best_registry_rank_metric"), -math.inf)
            if fixed > current_best:
                record["_best_registry_rank_metric"] = fixed
                record["best_candidate_status"] = status
                record["best_candidate_probe_dir"] = safe_str(row.get("probe_dir"))
                record["best_candidate_model_path"] = safe_str(row.get("model_path"))
                record["best_candidate_threshold_bps"] = safe_str(row.get("threshold_bps"))


def ingest_forward(records: dict[str, dict[str, Any]], sources: list[str]) -> None:
    for path in latest_files(FORWARD_ROOT, "shadow_forward_comparison.csv"):
        frame = read_csv(path)
        if frame.empty:
            continue
        sources.append(str(path))
        for _, row in frame.iterrows():
            key = contract_key(row)
            record = get_record(records, key)
            apply_ma_metadata(record, row)
            record["frozen_shadow_candidate_exists"] = True
            selected = safe_int(row.get("forward_selected_rows"))
            current_selected = safe_int(record.get("forward_selected_rows"))
            status = safe_str(row.get("forward_status")) or "missing_forward"
            status_score = STATUS_PRIORITY.get(status, 0)
            current_score = STATUS_PRIORITY.get(safe_str(record.get("forward_status")), 0)
            if selected > current_selected or (selected == current_selected and status_score > current_score):
                record["forward_status"] = status
                record["forward_selected_rows"] = selected
                record["forward_cum_net_bps"] = safe_float(row.get("forward_cum_net_bps"))
                record["forward_vs_registry_cum_ratio"] = safe_float(row.get("forward_vs_registry_cum_ratio"))
                record["forward_run"] = safe_str(row.get("selected_forward_run") or row.get("latest_forward_run"))


def ingest_walkforward(records: dict[str, dict[str, Any]], sources: list[str]) -> None:
    for path in latest_files(WALKFORWARD_ROOT, "contract_leaderboard.csv"):
        frame = read_csv(path)
        if frame.empty:
            continue
        sources.append(str(path))
        for _, row in frame.iterrows():
            key = contract_key(row)
            record = get_record(records, key)
            apply_ma_metadata(record, row)
            record["walkforward_runs"] += safe_int(row.get("runs"))
            record["walkforward_windows"] += safe_int(row.get("windows"))
            record["walkforward_total_rows"] += safe_int(row.get("total_test_rows"))
            current_cum = safe_float(record.get("walkforward_total_cum_bps"), 0.0)
            add_cum = safe_float(row.get("total_test_cumulative_return_bps"), 0.0)
            record["walkforward_total_cum_bps"] = current_cum + add_cum
            update_max(record, "walkforward_positive_window_fraction", row.get("positive_test_window_fraction"))


def ingest_survivors(records: dict[str, dict[str, Any]], sources: list[str]) -> None:
    for path in latest_files(PROJECT_ROOT / "data", "walkforward_contract_survivors.csv"):
        frame = read_csv(path)
        if frame.empty:
            continue
        sources.append(str(path))
        for _, row in frame.iterrows():
            key = contract_key(row)
            record = get_record(records, key)
            apply_ma_metadata(record, row)
            score = safe_float(row.get("contract_score"), -math.inf)
            current = safe_float(record.get("_survivor_score"), -math.inf)
            if score > current:
                record["_survivor_score"] = score
                record["survivor_status"] = safe_str(row.get("survivor_status"))
                record["survivor_positive_window_fraction"] = safe_float(row.get("positive_window_fraction"))
                record["survivor_total_cum_bps"] = safe_float(row.get("total_cumulative_return_bps"))
                record["survivor_representative_model_path"] = safe_str(row.get("representative_archived_model_path"))


def ingest_discovery(records: dict[str, dict[str, Any]], sources: list[str]) -> None:
    for path in latest_files(DISCOVERY_ROOT, "io_contract_discovery_summary.csv"):
        frame = read_csv(path)
        if frame.empty:
            continue
        sources.append(str(path))
        for _, row in frame.iterrows():
            key = contract_key(row)
            record = get_record(records, key)
            apply_ma_metadata(record, row)
            record["discovery_runs"] += 1
            statuses = {item for item in safe_str(record.get("discovery_statuses")).split(";") if item}
            statuses.add(safe_str(row.get("status")))
            record["discovery_statuses"] = ";".join(sorted(statuses))


def rank_record(record: dict[str, Any]) -> float:
    score = 0.0
    score += record["clean_shadow_candidates"] * 1000.0
    score += record["robust_research_candidates"] * 500.0
    score += record["fragile_research_candidates"] * 100.0
    score -= record["reject_candidates"] * 5.0
    score += max(0.0, safe_float(record.get("best_fixed_0_10_cum_net"), 0.0))
    score += 0.5 * max(0.0, safe_float(record.get("best_half_spread_plus_0_05_cum_net"), 0.0))
    score += 0.25 * max(0.0, safe_float(record.get("best_conservative_missing_liquidity_cum_net"), 0.0))
    score += 100.0 * safe_float(record.get("best_rolling_12h_fraction"), 0.0)
    score += 100.0 * safe_float(record.get("best_rolling_24h_fraction"), 0.0)
    score += min(safe_int(record.get("best_selected_rows")), 2000) / 10.0
    ratio = safe_float(record.get("best_drawdown_ratio"))
    if math.isfinite(ratio):
        score -= min(max(ratio, 0.0), 5.0) * 50.0
    score += STATUS_PRIORITY.get(safe_str(record.get("forward_status")), 0) * 100.0
    score += max(0.0, safe_float(record.get("forward_cum_net_bps"), 0.0))
    score += 100.0 * safe_float(record.get("survivor_positive_window_fraction"), 0.0)
    score += 0.05 * max(0.0, safe_float(record.get("walkforward_total_cum_bps"), 0.0))
    return score


def recommendation(record: dict[str, Any]) -> str:
    good = record["clean_shadow_candidates"] > 0 or record["robust_research_candidates"] > 0
    frozen = bool(record.get("frozen_shadow_candidate_exists"))
    forward_rows = safe_int(record.get("forward_selected_rows"))
    forward_status = safe_str(record.get("forward_status"))
    if good and not frozen:
        return "freeze_best_shadow_candidate"
    if frozen and forward_rows < 100:
        return "continue_forward_paper_until_sample_ge_100"
    if forward_status == "failed_forward" and forward_rows >= 100:
        return "deprioritize_or_replace_shadow_candidate"
    if record["fragile_research_candidates"] > 0 and not good:
        return "probe_more_thresholds_or_keep_research_only"
    if not good:
        return "ignore_or_expand_grid_later"
    if forward_status == "tracking_ok":
        return "eligible_for_explicit_freeze_audit"
    return "ignore_or_expand_grid_later"


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    record = dict(record)
    record["rank_score"] = rank_record(record)
    record["recommended_next_action"] = recommendation(record)
    raw_keys = record.get("_raw_contract_keys")
    if isinstance(raw_keys, set):
        record["duplicate_source_key_count"] = len(raw_keys)
        record["duplicate_source_keys"] = ";".join(sorted(raw_keys))
    for key in list(record.keys()):
        if key.startswith("_"):
            record.pop(key, None)
    return record


def output_dir() -> Path:
    path = OUTPUT_ROOT / f"rawseq_io_contract_leaderboard_{now_stamp()}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_text(path: Path, rows: list[dict[str, Any]], sources: list[str]) -> None:
    duplicate_rows = [row for row in rows if safe_int(row.get("duplicate_source_key_count")) > 1]
    ma_warning_rows = [
        row
        for row in rows
        if safe_str(row.get("input_feature")) == "ma_distance" and str(row.get("ma_window_missing_warning")).lower() == "true"
    ]
    lines = [
        "Rawseq I/O Contract Leaderboard",
        "",
        f"Created at: {now_stamp()}",
        f"Contracts: {len(rows)}",
        f"Duplicate source contracts merged: {len(duplicate_rows)}",
        f"ma_distance ma_window inference warnings: {len(ma_warning_rows)}",
        "",
        "Safety:",
        "  report_only=true",
        "  training=false",
        "  promotion=false",
        "  champion_mutation=false",
        "  orders=false",
        "",
        "Top contracts:",
    ]
    for row in rows[:20]:
        lines.append(
            "  "
            f"score={safe_float(row['rank_score']):.3f} "
            f"clean={row['clean_shadow_candidates']} robust={row['robust_research_candidates']} "
            f"fixed010={row['best_fixed_0_10_cum_net']} half05={row['best_half_spread_plus_0_05_cum_net']} "
            f"forward={row['forward_status']} action={row['recommended_next_action']} "
            f"{row['io_contract_slug']}"
        )
    lines.append("")
    lines.append("Best candidate per I/O contract:")
    for row in rows[:20]:
        lines.append(
            "  "
            f"{row['io_contract_slug']}: status={row['best_candidate_status']} "
            f"threshold={row['best_candidate_threshold_bps']} "
            f"model={row['best_candidate_model_path'] or row['survivor_representative_model_path']}"
        )
    lines.append("")
    lines.append("Duplicate contracts merged:")
    if duplicate_rows:
        for row in duplicate_rows[:20]:
            lines.append(
                "  "
                f"{row['io_contract_slug']}: raw_keys={row.get('duplicate_source_keys', '')}"
            )
    else:
        lines.append("  none")
    lines.append("")
    lines.append("ma_distance ma_window inference warnings:")
    if ma_warning_rows:
        for row in ma_warning_rows[:20]:
            lines.append(
                "  "
                f"{row['io_contract_slug']}: source={row.get('ma_window_source', '')} "
                f"best_probe={row.get('best_candidate_probe_dir', '')}"
            )
    else:
        lines.append("  none")
    lines.append("")
    lines.append("Sources read:")
    for source in sorted(set(sources)):
        lines.append(f"  {source}")
    lines.append("")
    lines.append("Warning: report only. No training, promotion, champion mutation, or orders.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    records: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    ingest_walkforward(records, sources)
    ingest_survivors(records, sources)
    ingest_registries(records, sources)
    ingest_forward(records, sources)
    ingest_discovery(records, sources)
    rows = [clean_record(record) for record in records.values()]
    rows = sorted(
        rows,
        key=lambda row: (
            safe_float(row.get("rank_score"), -math.inf),
            safe_float(row.get("best_fixed_0_10_cum_net"), -math.inf),
            safe_int(row.get("best_selected_rows")),
        ),
        reverse=True,
    )
    out_dir = output_dir()
    csv_path = out_dir / "io_contract_leaderboard.csv"
    txt_path = out_dir / "io_contract_leaderboard.txt"
    duplicate_path = out_dir / "duplicate_contracts_merged.csv"
    ma_warning_path = out_dir / "ma_window_inference_warnings.csv"
    duplicate_rows = [row for row in rows if safe_int(row.get("duplicate_source_key_count")) > 1]
    ma_warning_rows = [
        row
        for row in rows
        if safe_str(row.get("input_feature")) == "ma_distance" and str(row.get("ma_window_missing_warning")).lower() == "true"
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(duplicate_rows).to_csv(duplicate_path, index=False)
    pd.DataFrame(ma_warning_rows).to_csv(ma_warning_path, index=False)
    write_text(txt_path, rows, sources)
    print("Rawseq I/O contract leaderboard complete")
    print(f"Contracts: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"TXT: {txt_path}")
    print(f"Duplicate contracts merged CSV: {duplicate_path}")
    print(f"ma_window inference warnings CSV: {ma_warning_path}")
    print("Safety: report only. No training. No promotion. No champion mutation. No orders.")
    print(pd.DataFrame(rows).head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
