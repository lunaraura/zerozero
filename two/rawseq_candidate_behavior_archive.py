#!/usr/bin/env python3
"""Build and query behavioral archives for rawseq candidate probes."""

from __future__ import annotations

import csv
import json
import math
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_ROOT = Path(
    os.getenv(
        "RAWSEQ_ARCHIVE_PROBE_ROOT",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_candidate_shadow_probes"),
    )
)
REGISTRY_ROOT = Path(
    os.getenv(
        "RAWSEQ_ARCHIVE_REGISTRY_ROOT",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_probe_registry"),
    )
)
OUTPUT_ROOT = Path(
    os.getenv(
        "RAWSEQ_ARCHIVE_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_behavior_archive"),
    )
)
CANDIDATE_PROBE_DIR = os.getenv("RAWSEQ_CANDIDATE_PROBE_DIR", "").strip()
MAX_ANNOTATED_ROWS = int(float(os.getenv("RAWSEQ_ARCHIVE_MAX_ANNOTATED_ROWS", "250000")))

PRED_COL = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COL = "rawseq_path_actual_horizon_return_bps"
FEATURE_COLUMNS = [
    "pred_mean",
    "pred_std",
    "pred_skew",
    "pred_q01",
    "pred_q05",
    "pred_q25",
    "pred_q50",
    "pred_q75",
    "pred_q95",
    "pred_q99",
    "pred_actual_corr",
    "path_final_mean",
    "path_min_mean",
    "path_max_mean",
    "max_selected_rows",
    "min_selected_rows",
    "best_fixed_0_10_cum_net",
    "best_half_spread_plus_0_05_cum_net",
    "best_conservative_missing_liquidity_cum_net",
    "best_rolling_12h_fraction",
    "best_rolling_24h_fraction",
    "best_max_dip_to_cum_net_ratio",
]


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def latest_registry_rows() -> pd.DataFrame:
    if not REGISTRY_ROOT.exists():
        return pd.DataFrame()
    frames = []
    for path in REGISTRY_ROOT.rglob("rawseq_probe_threshold_registry.csv"):
        frame = read_csv(path)
        if not frame.empty:
            frame["registry_source"] = str(path)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def threshold_from_name(path: Path) -> str:
    name = path.stem
    if "_threshold_" in name:
        return name.rsplit("_threshold_", 1)[1]
    return ""


def read_first_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return next(csv.DictReader(handle), {})
    except Exception:
        return {}


def load_contract(probe_dir: Path) -> dict[str, Any]:
    path = probe_dir / "model_contract.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def annotated_fingerprint(probe_dir: Path) -> dict[str, Any]:
    path = probe_dir / "annotated.csv"
    result = {column: math.nan for column in FEATURE_COLUMNS[:14]}
    result["annotated_rows_sampled"] = 0
    if not path.exists():
        return result
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return result
    columns = [column for column in [PRED_COL, ACTUAL_COL] if column in header.columns]
    y_cols = [column for column in header.columns if column.startswith("rawseq_pred_fwd_") and column.endswith("_bps")]
    usecols = columns + y_cols[:16]
    if not usecols:
        return result
    try:
        frame = pd.read_csv(path, usecols=usecols, nrows=MAX_ANNOTATED_ROWS, low_memory=False)
    except Exception:
        return result
    result["annotated_rows_sampled"] = int(len(frame))
    if PRED_COL in frame.columns:
        pred = pd.to_numeric(frame[PRED_COL], errors="coerce").dropna()
        if not pred.empty:
            result.update(
                {
                    "pred_mean": float(pred.mean()),
                    "pred_std": float(pred.std(ddof=0)),
                    "pred_skew": float(pred.skew()),
                    "pred_q01": float(pred.quantile(0.01)),
                    "pred_q05": float(pred.quantile(0.05)),
                    "pred_q25": float(pred.quantile(0.25)),
                    "pred_q50": float(pred.quantile(0.50)),
                    "pred_q75": float(pred.quantile(0.75)),
                    "pred_q95": float(pred.quantile(0.95)),
                    "pred_q99": float(pred.quantile(0.99)),
                }
            )
    if PRED_COL in frame.columns and ACTUAL_COL in frame.columns:
        pair = frame[[PRED_COL, ACTUAL_COL]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 3:
            result["pred_actual_corr"] = float(pair[PRED_COL].corr(pair[ACTUAL_COL]))
    if y_cols:
        path_frame = frame[y_cols[:16]].apply(pd.to_numeric, errors="coerce")
        result["path_final_mean"] = float(path_frame.iloc[:, -1].mean())
        result["path_min_mean"] = float(path_frame.min(axis=1).mean())
        result["path_max_mean"] = float(path_frame.max(axis=1).mean())
    return result


def probe_report_rows(probe_dir: Path) -> pd.DataFrame:
    frames = []
    for path in [probe_dir / "cost_threshold_summary.csv", probe_dir / "dynamic_cost_summary.csv"]:
        frame = read_csv(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def registry_for_probe(registry: pd.DataFrame, probe_dir: Path) -> pd.DataFrame:
    if registry.empty or "probe_dir" not in registry.columns:
        return pd.DataFrame()
    probe_text = str(probe_dir)
    mask = registry["probe_dir"].astype(str).eq(probe_text)
    if not mask.any():
        mask = registry["probe_dir"].astype(str).str.endswith(probe_dir.name, na=False)
    return registry[mask].copy()


def fixed_rows(report_rows: pd.DataFrame, cost: float) -> pd.DataFrame:
    if report_rows.empty:
        return pd.DataFrame()
    if "cost_bps" in report_rows.columns:
        cost_values = pd.to_numeric(report_rows["cost_bps"], errors="coerce")
        return report_rows[cost_values.sub(cost).abs() < 1e-9].copy()
    if "scenario" in report_rows.columns:
        label = f"fixed_{str(cost).replace('.', '_')}_bps"
        return report_rows[report_rows["scenario"].astype(str).eq(label)].copy()
    return pd.DataFrame()


def threshold_sensitivity(report_rows: pd.DataFrame) -> dict[str, Any]:
    result = {
        "thresholds_tested": "",
        "threshold_0_fixed_0_10_cum_net": math.nan,
        "positive_fixed_0_10_thresholds": 0,
        "threshold_0_only_behavior": False,
        "max_selected_rows": 0,
        "min_selected_rows": 0,
        "best_fixed_0_10_cum_net": math.nan,
        "best_fixed_0_10_threshold_bps": "",
    }
    rows = fixed_rows(report_rows, 0.1)
    if rows.empty or "threshold_bps" not in rows.columns:
        return result
    rows = rows.copy()
    rows["threshold_bps_num"] = pd.to_numeric(rows["threshold_bps"], errors="coerce")
    rows["cum_net_num"] = pd.to_numeric(rows.get("cum_net_bps"), errors="coerce")
    rows["selected_rows_num"] = pd.to_numeric(rows.get("selected_rows"), errors="coerce")
    thresholds = sorted({safe_float(value) for value in rows["threshold_bps_num"].dropna()})
    result["thresholds_tested"] = ";".join(f"{value:g}" for value in thresholds)
    result["positive_fixed_0_10_thresholds"] = int((rows["cum_net_num"] > 0).sum())
    result["max_selected_rows"] = int(rows["selected_rows_num"].max()) if rows["selected_rows_num"].notna().any() else 0
    result["min_selected_rows"] = int(rows["selected_rows_num"].min()) if rows["selected_rows_num"].notna().any() else 0
    zero = rows[rows["threshold_bps_num"].abs() < 1e-9]
    if not zero.empty:
        result["threshold_0_fixed_0_10_cum_net"] = safe_float(zero.iloc[0].get("cum_net_bps"))
    if rows["cum_net_num"].notna().any():
        best = rows.sort_values("cum_net_num", ascending=False).iloc[0]
        result["best_fixed_0_10_cum_net"] = safe_float(best.get("cum_net_bps"))
        result["best_fixed_0_10_threshold_bps"] = safe_str(best.get("threshold_bps"))
        result["threshold_0_only_behavior"] = (
            safe_float(best.get("threshold_bps")) == 0.0
            and int((rows[rows["threshold_bps_num"] > 0]["cum_net_num"] > 0).sum()) == 0
        )
    return result


def dynamic_sensitivity(report_rows: pd.DataFrame) -> dict[str, Any]:
    result = {
        "best_half_spread_plus_0_05_cum_net": math.nan,
        "best_conservative_missing_liquidity_cum_net": math.nan,
        "fixed_0_25_best_cum_net": math.nan,
    }
    if report_rows.empty:
        return result
    if "scenario" in report_rows.columns:
        scenario = report_rows["scenario"].astype(str)
        for scenario_name, key in [
            ("half_spread_plus_0_05_bps", "best_half_spread_plus_0_05_cum_net"),
            ("half_spread_plus_0_05", "best_half_spread_plus_0_05_cum_net"),
            ("conservative_missing_liquidity_penalty", "best_conservative_missing_liquidity_cum_net"),
            ("fixed_0_25_bps", "fixed_0_25_best_cum_net"),
        ]:
            rows = report_rows[scenario.eq(scenario_name)]
            if not rows.empty and "cum_net_bps" in rows.columns:
                result[key] = max(safe_float(result[key]), safe_float(pd.to_numeric(rows["cum_net_bps"], errors="coerce").max()))
    if "cost_bps" in report_rows.columns:
        rows = fixed_rows(report_rows, 0.25)
        if not rows.empty and "cum_net_bps" in rows.columns:
            result["fixed_0_25_best_cum_net"] = safe_float(pd.to_numeric(rows["cum_net_bps"], errors="coerce").max())
    return result


def rolling_from_registry(rows: pd.DataFrame) -> dict[str, Any]:
    result = {
        "best_rolling_12h_fraction": math.nan,
        "best_rolling_24h_fraction": math.nan,
        "best_max_dip_to_cum_net_ratio": math.nan,
    }
    if rows.empty:
        return result
    for source, target in [
        ("positive_12h_window_fraction", "best_rolling_12h_fraction"),
        ("positive_24h_window_fraction", "best_rolling_24h_fraction"),
    ]:
        if source in rows.columns:
            result[target] = safe_float(pd.to_numeric(rows[source], errors="coerce").max())
    if "max_dip_to_cum_net_ratio" in rows.columns:
        vals = pd.to_numeric(rows["max_dip_to_cum_net_ratio"], errors="coerce").dropna()
        if not vals.empty:
            result["best_max_dip_to_cum_net_ratio"] = float(vals.min())
    return result


def archive_bucket(row: dict[str, Any]) -> tuple[str, str]:
    registry_statuses = set(safe_str(row.get("registry_statuses")).split(";"))
    if "clean_shadow_candidate" in registry_statuses or "robust_research_candidate" in registry_statuses:
        return "good", "registry_clean_or_robust"
    fixed = safe_float(row.get("best_fixed_0_10_cum_net"))
    half = safe_float(row.get("best_half_spread_plus_0_05_cum_net"))
    conservative = safe_float(row.get("best_conservative_missing_liquidity_cum_net"))
    dip = safe_float(row.get("best_max_dip_to_cum_net_ratio"))
    roll12 = safe_float(row.get("best_rolling_12h_fraction"))
    roll24 = safe_float(row.get("best_rolling_24h_fraction"))
    selected = safe_int(row.get("max_selected_rows"))
    threshold0 = bool(row.get("threshold_0_only_behavior"))
    if fixed <= 0:
        return "bad", "fixed_0_10_nonpositive_all_useful_thresholds"
    if threshold0:
        return "bad", "threshold_0_only_behavior"
    if math.isfinite(dip) and dip > 3:
        return "bad", "severe_drawdown_ratio"
    if math.isfinite(roll12) and math.isfinite(roll24) and roll12 < 0.35 and roll24 < 0.35:
        return "bad", "rolling_windows_consistently_negative"
    if selected < 100 or (selected > 10000 and fixed <= 0):
        return "bad", "selected_rows_too_sparse_or_dense_without_edge"
    if fixed > 0 and (half <= 0 or conservative <= 0 or (math.isfinite(dip) and dip > 1.5)):
        return "near_miss", "fixed_positive_but_dynamic_or_drawdown_fails"
    return "near_miss", "some_fixed_positive_but_not_registry_clean"


def fingerprint_probe(probe_dir: Path, registry: pd.DataFrame) -> list[dict[str, Any]]:
    contract = load_contract(probe_dir)
    report_rows = probe_report_rows(probe_dir)
    registry_rows = registry_for_probe(registry, probe_dir)
    ann = annotated_fingerprint(probe_dir)
    thresh = threshold_sensitivity(report_rows)
    dyn = dynamic_sensitivity(report_rows)
    roll = rolling_from_registry(registry_rows)
    thresholds = set()
    if not registry_rows.empty and "threshold_bps" in registry_rows.columns:
        thresholds.update(safe_str(value) for value in registry_rows["threshold_bps"].dropna())
    if thresh["thresholds_tested"]:
        thresholds.update(thresh["thresholds_tested"].split(";"))
    if not thresholds:
        thresholds.add("")

    statuses = sorted({safe_str(value) for value in registry_rows.get("status", pd.Series(dtype=str)).dropna() if safe_str(value)})
    base = {
        "probe_dir": str(probe_dir),
        "probe_name": probe_dir.name,
        "symbol": safe_str(contract.get("symbol")),
        "venue": safe_str(contract.get("venue")),
        "input_feature": safe_str(contract.get("input_feature")),
        "ma_window": safe_str(contract.get("ma_window")),
        "hidden": safe_str(contract.get("hidden")),
        "seq_len": safe_str(contract.get("seq_len")),
        "bucket_seconds": safe_str(contract.get("bucket_seconds")),
        "input_stride": safe_str(contract.get("input_stride")),
        "output_stride": safe_str(contract.get("output_stride")),
        "source_path_basename": safe_str(contract.get("source_path_basename")),
        "model_path": safe_str(contract.get("model_path")),
        "registry_statuses": ";".join(statuses),
        **ann,
        **thresh,
        **dyn,
        **roll,
        "paper_only": True,
        "training": False,
        "promotion": False,
        "champion_mutation": False,
        "orders": False,
    }
    rows = []
    for threshold in sorted(thresholds, key=lambda item: safe_float(item, 1e9)):
        row = dict(base)
        row["threshold_bps"] = threshold
        if not registry_rows.empty and threshold:
            subset = registry_rows[registry_rows["threshold_bps"].astype(str).eq(str(threshold))]
            if not subset.empty:
                first = subset.iloc[0]
                row["registry_status"] = safe_str(first.get("status"))
                row["registry_fixed_0_10_cum_net"] = safe_float(first.get("fixed_0_10_cum_net"))
                row["registry_selected_rows"] = safe_int(first.get("selected_rows"))
        bucket, reason = archive_bucket(row)
        row["archive_bucket"] = bucket
        row["archive_reason"] = reason
        rows.append(row)
    return rows


def output_dir() -> Path:
    path = resolve_path(OUTPUT_ROOT) / f"rawseq_behavior_archive_{now_stamp()}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_summary(path: Path, rows: list[dict[str, Any]], compare_rows: list[dict[str, Any]] | None = None) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["archive_bucket"]] = counts.get(row["archive_bucket"], 0) + 1
    lines = [
        "Rawseq Candidate Behavior Archive",
        "",
        f"Created at: {now_stamp()}",
        f"Rows: {len(rows)}",
        "",
        "Safety:",
        "  report_only=true",
        "  training=false",
        "  promotion=false",
        "  champion_mutation=false",
        "  orders=false",
        "",
        "Archive buckets:",
    ]
    for key in sorted(counts):
        lines.append(f"  {key}: {counts[key]}")
    if compare_rows is not None:
        lines.append("")
        lines.append("Candidate comparison:")
        for row in compare_rows[:15]:
            lines.append(
                "  "
                f"bucket={row['archive_bucket']} distance={row['distance']:.6f} "
                f"novelty={row['novelty_score']:.6f} prune={row['prune_recommendation']} "
                f"probe={row['nearest_probe_dir']}"
            )
    lines.append("")
    lines.append("Warning: report only. No training, promotion, champion mutation, or orders.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def numeric_vector(row: dict[str, Any], means: dict[str, float], stds: dict[str, float]) -> np.ndarray:
    values = []
    for column in FEATURE_COLUMNS:
        value = safe_float(row.get(column), means.get(column, 0.0))
        mean = means.get(column, 0.0)
        std = stds.get(column, 1.0) or 1.0
        if not math.isfinite(value):
            value = mean
        values.append((value - mean) / std)
    return np.asarray(values, dtype=np.float64)


def compare_candidate(candidate_rows: list[dict[str, Any]], archive_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(archive_rows)
    if frame.empty or not candidate_rows:
        return []
    means = {column: safe_float(pd.to_numeric(frame.get(column), errors="coerce").mean(), 0.0) for column in FEATURE_COLUMNS}
    stds = {column: safe_float(pd.to_numeric(frame.get(column), errors="coerce").std(ddof=0), 1.0) or 1.0 for column in FEATURE_COLUMNS}
    archive_vectors = [(row, numeric_vector(row, means, stds)) for row in archive_rows]
    candidate = candidate_rows[0]
    candidate_vec = numeric_vector(candidate, means, stds)
    results = []
    for row, vec in archive_vectors:
        if row.get("probe_dir") == candidate.get("probe_dir"):
            continue
        distance = float(np.linalg.norm(candidate_vec - vec) / math.sqrt(len(FEATURE_COLUMNS)))
        novelty = distance
        bucket = safe_str(row.get("archive_bucket"))
        if bucket == "bad" and distance < 0.75:
            prune = "prune_similar_to_bad"
        elif bucket == "near_miss" and distance < 0.75:
            prune = "allow_with_caution_near_miss"
        elif distance > 1.5:
            prune = "novel_enough_to_test"
        else:
            prune = "test_if_budget_allows"
        results.append(
            {
                "candidate_probe_dir": candidate.get("probe_dir"),
                "archive_bucket": bucket,
                "nearest_probe_dir": row.get("probe_dir"),
                "distance": distance,
                "novelty_score": novelty,
                "prune_recommendation": prune,
            }
        )
    deduped: dict[str, dict[str, Any]] = {}
    for row in sorted(results, key=lambda item: item["distance"]):
        key = safe_str(row.get("nearest_probe_dir"))
        if key and key not in deduped:
            deduped[key] = row
    return list(deduped.values())


def main() -> int:
    probe_root = resolve_path(PROBE_ROOT)
    registry = latest_registry_rows()
    probe_dirs = sorted([path for path in probe_root.iterdir() if path.is_dir()]) if probe_root.exists() else []
    rows: list[dict[str, Any]] = []
    for probe_dir in probe_dirs:
        rows.extend(fingerprint_probe(probe_dir, registry))
    out_dir = output_dir()
    frame = pd.DataFrame(rows)
    archive_path = out_dir / "behavior_archive.csv"
    frame.to_csv(archive_path, index=False)
    for bucket, filename in [
        ("bad", "bad_fingerprints.csv"),
        ("near_miss", "near_miss_fingerprints.csv"),
        ("good", "good_fingerprints.csv"),
    ]:
        frame[frame["archive_bucket"].eq(bucket)].to_csv(out_dir / filename, index=False)

    compare_rows: list[dict[str, Any]] | None = None
    if CANDIDATE_PROBE_DIR:
        candidate_dir = resolve_path(CANDIDATE_PROBE_DIR)
        candidate_rows = fingerprint_probe(candidate_dir, registry)
        compare_rows = compare_candidate(candidate_rows, rows)
        pd.DataFrame(compare_rows).to_csv(out_dir / "candidate_archive_comparison.csv", index=False)
    write_summary(out_dir / "behavior_archive_summary.txt", rows, compare_rows)
    print("Rawseq behavior archive complete")
    print(f"Rows: {len(rows)}")
    print(f"Output dir: {out_dir}")
    print(f"Archive CSV: {archive_path}")
    if compare_rows is not None:
        print(f"Comparison CSV: {out_dir / 'candidate_archive_comparison.csv'}")
        print(pd.DataFrame(compare_rows).head(15).to_string(index=False))
    print("Safety: report only. No training. No promotion. No champion mutation. No orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
