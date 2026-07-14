#!/usr/bin/env python3
"""Prospective paper-shadow logger for frozen downside-risk CPU candidate.

This script runs the frozen logistic downside-risk score exactly as saved. It
starts strictly after the consumed development/holdout data, records predictions
separately from labels, and evaluates only rows whose future horizon label is
already available in the provided public/recorded feature table.

No training, no recalibration, no threshold changes, no orders, no private API,
no champion mutation, no promotion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.freeze_rawseq_downside_risk_cpu_candidate import (
    policy_threshold_rows,
    predict_logistic_coef,
    rank_auc,
    pr_auc,
    calibration_error,
    calibration_slope_intercept,
)
from scripts.tiny.freeze_rawseq_low_path_ridge_research_candidate import file_sha256, stable_hash, array_sha256
from scripts.tiny.rawseq_future_shadow_lock import attach_implementation_lock


DEFAULT_CANDIDATE_DIR = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_cpu_candidates" / "rawseq_downside_risk_cpu_candidate_20260711T233404Z"
DEFAULT_FEATURE_TABLE = Path(r"F:\rsio\rawseq_target_tournament_coarse_1s_300k_retry\mh_indicator_SOLUSDT_kraken_20260711T145015Z_fba19c8d\multi_horizon_training_table.csv")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_downside_risk_future_shadow"
HORIZONS = [60, 120, 240, 480]
H480_SECONDS = 480
MIN_CALENDAR_DAYS = 30
MIN_NON_OVERLAP_H480 = 5000
MIN_EVENTS = 100
MIN_NON_EVENTS = 100
REGIME_FEATURE_COLUMNS = [
    "realized_volatility_bps_fw60",
    "rolling_range_bps_fw60",
    "volume_zscore_fw60",
    "ema_slope_bps_fw60",
]


def horizon_from_target_column(target_column: str) -> int:
    match = re.search(r"_h(\d+)(?:$|_)", target_column)
    return int(match.group(1)) if match else 0


def add_derived_low_path_labels(table: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    out = table.copy()
    price_col = "close" if "close" in out.columns else ("price" if "price" in out.columns else "")
    if not price_col:
        return out
    price = pd.to_numeric(out[price_col], errors="coerce")
    shifted = price.shift(-1)
    reversed_shifted = shifted.iloc[::-1]
    for col in target_columns:
        if col in out.columns and pd.to_numeric(out[col], errors="coerce").notna().any():
            continue
        if not col.startswith("future_range_low_bps_h"):
            continue
        horizon = horizon_from_target_column(col)
        if horizon <= 0:
            continue
        future_low = reversed_shifted.rolling(horizon, min_periods=horizon).min().iloc[::-1].reset_index(drop=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = 10000.0 * np.log(future_low.to_numpy(dtype=np.float64) / price.to_numpy(dtype=np.float64))
        values = np.minimum(values, 0.0)
        values[~np.isfinite(values)] = np.nan
        out[col] = values
    return out


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size <= 2:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def merge_decisions(existing: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ts: dict[float, dict[str, Any]] = {}
    for row in existing:
        ts = safe_float(row.get("decision_timestamp"))
        if math.isfinite(ts):
            migrated = dict(row)
            if "true_forward_decision" not in migrated:
                migrated["true_forward_decision"] = True
                migrated["backfill_or_replay_decision"] = False
                migrated["label_availability_at_logging"] = "legacy_assumed_none_available"
            by_ts[ts] = migrated
    for row in new_rows:
        ts = safe_float(row.get("decision_timestamp"))
        if not math.isfinite(ts):
            continue
        if ts not in by_ts:
            by_ts[ts] = row
        else:
            existing_row = by_ts[ts]
            merged = dict(existing_row)
            for key, value in row.items():
                if key.startswith("last_") or key in {"last_seen_at_iso", "last_run_id"}:
                    merged[key] = value
                elif key not in merged or merged[key] in {"", None}:
                    merged[key] = value
            if "first_logged_at_iso" not in merged and "logged_at_iso" in merged:
                merged["first_logged_at_iso"] = merged["logged_at_iso"]
            by_ts[ts] = merged
    return [by_ts[ts] for ts in sorted(by_ts)]


def merge_labeled_rows(existing: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ts: dict[float, dict[str, Any]] = {}
    for row in [*existing, *new_rows]:
        ts = safe_float(row.get("decision_timestamp"))
        if not math.isfinite(ts):
            continue
        merged = dict(by_ts.get(ts, {}))
        for key, value in row.items():
            if value in {"", None}:
                continue
            merged[key] = value
        by_ts[ts] = merged
    return [by_ts[ts] for ts in sorted(by_ts)]


def merge_timestamp_rows(existing: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ts: dict[float, dict[str, Any]] = {}
    for row in [*existing, *new_rows]:
        ts = safe_float(row.get("decision_timestamp"))
        if not math.isfinite(ts):
            continue
        merged = dict(by_ts.get(ts, {}))
        for key, value in row.items():
            if value in {"", None}:
                continue
            merged[key] = value
        by_ts[ts] = merged
    return [by_ts[ts] for ts in sorted(by_ts)]


def label_availability_payload(row: pd.Series, target_columns: list[str]) -> dict[str, Any]:
    available = 0
    for col in target_columns:
        if col in row.index and math.isfinite(safe_float(row.get(col))):
            available += 1
    if available == 0:
        state = "none_available"
    elif available == len(target_columns):
        state = "all_available"
    else:
        state = "partial_available"
    return {
        "target_labels_available_at_logging": available,
        "target_labels_expected": len(target_columns),
        "label_availability_at_logging": state,
        "true_forward_decision": available == 0,
        "backfill_or_replay_decision": available > 0,
    }


def true_forward_decision_rows(decision_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in decision_rows:
        value = row.get("true_forward_decision", True)
        if isinstance(value, str):
            is_true = value.strip().lower() in {"true", "1", "yes"}
        else:
            is_true = bool(value)
        if is_true:
            rows.append(row)
    return rows


def horizon_summary(summary_rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    suffix = f"_h{horizon}"
    for row in summary_rows:
        if str(row.get("target_column", "")).endswith(suffix):
            return row
    return {}


def transform(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = (np.where(np.isfinite(values), values, mean) - mean) / std
    out[~np.isfinite(out)] = 0.0
    return out


def predict_multihorizon(x: np.ndarray, mean: np.ndarray, std: np.ndarray, coef: np.ndarray) -> np.ndarray:
    xs = transform(x, mean, std)
    preds = []
    for idx in range(coef.shape[1]):
        preds.append(predict_logistic_coef(xs, coef[:, idx]))
    return np.column_stack(preds)


def consumed_cutoff_from_source_npz(source_npz: Path) -> float:
    with np.load(source_npz, allow_pickle=False) as data:
        return float(np.nanmax(data["decision_timestamps"].astype(np.float64)))


def non_overlapping_mask(timestamps_ms: np.ndarray, min_gap_seconds: int = H480_SECONDS) -> np.ndarray:
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.float64)
    order = np.argsort(timestamps_ms)
    keep = np.zeros(len(timestamps_ms), dtype=bool)
    last = -math.inf
    min_gap_ms = float(min_gap_seconds * 1000)
    for idx in order:
        ts = timestamps_ms[idx]
        if not math.isfinite(ts):
            continue
        if ts - last >= min_gap_ms:
            keep[idx] = True
            last = ts
    return keep


def reliability_rows(y: np.ndarray, prob: np.ndarray, target_column: str, bins: int = 10) -> list[dict[str, Any]]:
    rows = []
    for idx, lo in enumerate(np.linspace(0, 1, bins, endpoint=False)):
        hi = lo + 1.0 / bins
        mask = (prob >= lo) & (prob < hi if idx < bins - 1 else prob <= hi)
        rows.append(
            {
                "target_column": target_column,
                "bin_index": idx,
                "bin_low": lo,
                "bin_high": hi,
                "rows": int(mask.sum()),
                "predicted_probability_mean": float(np.mean(prob[mask])) if mask.any() else math.nan,
                "event_rate": float(np.mean(y[mask])) if mask.any() else math.nan,
                "absolute_calibration_error": abs(float(np.mean(prob[mask]) - np.mean(y[mask]))) if mask.any() else math.nan,
            }
        )
    return rows


def frozen_development_prevalence(policy: dict[str, Any]) -> float:
    direct = safe_float(policy.get("development_event_prevalence"))
    if math.isfinite(direct):
        return direct
    for key in ["primary_threshold", "backup_threshold"]:
        threshold_policy = policy.get(key, {})
        caught = safe_float(threshold_policy.get("adverse_events_avoided"))
        recall = safe_float(threshold_policy.get("event_recall"))
        rejected_fraction = safe_float(threshold_policy.get("fraction_opportunities_rejected"))
        safe_rejected = safe_float(threshold_policy.get("favorable_opportunities_rejected"))
        if caught > 0 and recall > 0 and rejected_fraction > 0:
            rejected = caught + max(safe_rejected, 0.0)
            total = rejected / rejected_fraction
            events = caught / recall
            if total > 0 and events > 0:
                return events / total
    return math.nan


def binary_metrics(y: np.ndarray, prob: np.ndarray, prevalence: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    const = np.full_like(prob, np.clip(prevalence, 1e-6, 1 - 1e-6))
    brier = float(np.mean((prob - y) ** 2)) if len(y) else math.nan
    const_brier = float(np.mean((const - y) ** 2)) if len(y) else math.nan
    log_loss = float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))) if len(y) else math.nan
    const_log_loss = float(-np.mean(y * np.log(const) + (1 - y) * np.log(1 - const))) if len(y) else math.nan
    slope, intercept = calibration_slope_intercept(y, prob) if len(y) else (math.nan, math.nan)
    pr = pr_auc(y, prob) if len(y) else math.nan
    rel = reliability_rows(y, prob, "metric")
    max_cal_error = max([safe_float(row.get("absolute_calibration_error"), 0.0) for row in rel], default=math.nan)
    return {
        "rows": int(len(y)),
        "events": int(np.sum(y > 0.5)) if len(y) else 0,
        "non_events": int(np.sum(y <= 0.5)) if len(y) else 0,
        "event_prevalence": float(np.mean(y)) if len(y) else math.nan,
        "frozen_prevalence_baseline": prevalence,
        "brier_score": brier,
        "constant_prevalence_brier_score": const_brier,
        "brier_skill_score": (const_brier - brier) / const_brier if const_brier > 0 else math.nan,
        "log_loss": log_loss,
        "constant_prevalence_log_loss": const_log_loss,
        "log_loss_improvement": const_log_loss - log_loss if math.isfinite(const_log_loss) and math.isfinite(log_loss) else math.nan,
        "roc_auc": rank_auc(y, prob) if len(y) else math.nan,
        "pr_auc": pr,
        "pr_auc_lift_over_prevalence": pr - float(np.mean(y)) if math.isfinite(pr) and len(y) else math.nan,
        "expected_calibration_error": calibration_error(y, prob) if len(y) else math.nan,
        "maximum_calibration_error": max_cal_error,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "probability_mean": float(np.mean(prob)) if len(y) else math.nan,
        "probability_std": float(np.std(prob)) if len(y) else math.nan,
        "prediction_unique_fraction": float(len(np.unique(np.round(prob, 8))) / len(prob)) if len(y) else math.nan,
    }


def detailed_policy_threshold_rows(y: np.ndarray, prob: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    yy = np.asarray(y, dtype=np.float64).ravel()
    pp = np.asarray(prob, dtype=np.float64).ravel()
    event_mask = yy > 0.5
    safe_mask = ~event_mask
    events = max(float(event_mask.sum()), 1.0)
    safe = max(float(safe_mask.sum()), 1.0)
    for row in policy_threshold_rows(yy, pp):
        threshold = safe_float(row.get("threshold"))
        reject = pp >= threshold
        retained = ~reject
        adverse_caught = float(np.sum(reject & event_mask))
        adverse_missed = float(np.sum(retained & event_mask))
        safe_rejected = float(np.sum(reject & safe_mask))
        safe_retained = float(np.sum(retained & safe_mask))
        rejected_fraction = float(np.mean(reject)) if len(reject) else math.nan
        event_recall = adverse_caught / events
        enriched = dict(row)
        enriched.update(
            {
                "adverse_events_caught": adverse_caught,
                "adverse_events_missed": adverse_missed,
                "safe_opportunities_rejected": safe_rejected,
                "safe_opportunities_retained": safe_retained,
                "percentage_all_opportunities_rejected": rejected_fraction * 100.0 if math.isfinite(rejected_fraction) else math.nan,
                "safe_rejection_rate": safe_rejected / safe,
                "risk_reduction_per_1pct_coverage_lost": event_recall / max(rejected_fraction * 100.0, 1e-12) if math.isfinite(rejected_fraction) else math.nan,
            }
        )
        rows.append(enriched)
    return rows


def bootstrap_ci(y: np.ndarray, prob: np.ndarray, prevalence: float, block_ids: np.ndarray, iterations: int = 300) -> dict[str, Any]:
    unique = np.asarray(sorted(set(block_ids.tolist())))
    if len(unique) < 5:
        return {"brier_skill_ci_low": math.nan, "brier_skill_ci_high": math.nan, "bootstrap_blocks": int(len(unique))}
    rng = np.random.default_rng(1337)
    values = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        mask = np.isin(block_ids, sampled)
        values.append(binary_metrics(y[mask], prob[mask], prevalence)["brier_skill_score"])
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return {
        "brier_skill_ci_low": float(np.quantile(finite, 0.025)) if finite.size else math.nan,
        "brier_skill_ci_high": float(np.quantile(finite, 0.975)) if finite.size else math.nan,
        "bootstrap_blocks": int(len(unique)),
    }


def regime_bucket(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return np.asarray(["unknown"] * len(values), dtype=object)
    q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
    out = np.asarray(["medium"] * len(values), dtype=object)
    out[values <= q1] = "low"
    out[values > q2] = "high"
    out[~np.isfinite(values)] = "unknown"
    return out


def calendar_bucket(timestamps_ms: np.ndarray, kind: str) -> np.ndarray:
    out: list[str] = []
    for value in timestamps_ms:
        if not math.isfinite(float(value)):
            out.append("unknown")
            continue
        dt = datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
        if kind == "daytime_overnight":
            out.append("daytime_utc" if 8 <= dt.hour < 20 else "overnight_utc")
        elif kind == "weekday_weekend":
            out.append("weekend" if dt.weekday() >= 5 else "weekday")
        else:
            out.append("unknown")
    return np.asarray(out, dtype=object)


def metrics_from_labeled_rows(
    labeled_rows: list[dict[str, Any]],
    target_columns: list[str],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    if not labeled_rows:
        return summary_rows, reliability, regime_rows, bootstrap_rows, threshold_rows
    timestamps = np.asarray([safe_float(row.get("decision_timestamp")) for row in labeled_rows], dtype=np.float64)
    for col in target_columns:
        yy_all = np.asarray([safe_float(row.get(f"event_{col}")) for row in labeled_rows], dtype=np.float64)
        pp_all = np.asarray([safe_float(row.get(f"prob_{col}")) for row in labeled_rows], dtype=np.float64)
        mask = np.isfinite(timestamps) & np.isfinite(yy_all) & np.isfinite(pp_all)
        if not mask.any():
            continue
        timestamps_arr = timestamps[mask]
        yy = yy_all[mask]
        pp = pp_all[mask]
        nonoverlap = non_overlapping_mask(timestamps_arr, H480_SECONDS)
        block_ids = np.floor((timestamps_arr - timestamps_arr.min()) / (H480_SECONDS * 1000)).astype(np.int64)
        prevalence = frozen_development_prevalence(policy)
        metrics = binary_metrics(yy, pp, prevalence)
        metrics.update(
            {
                "target_column": col,
                "horizon_seconds": horizon_from_target_column(col),
                "non_overlapping_h480_rows": int(nonoverlap.sum()),
                "calendar_days": float((timestamps_arr.max() - timestamps_arr.min()) / 86_400_000.0) if len(timestamps_arr) else 0.0,
                "label_source": "durable_cumulative_labeled_results",
            }
        )
        summary_rows.append(metrics)
        reliability.extend(reliability_rows(yy, pp, col))
        bootstrap = bootstrap_ci(yy[nonoverlap], pp[nonoverlap], prevalence, block_ids[nonoverlap])
        bootstrap["target_column"] = col
        bootstrap_rows.append(bootstrap)
        threshold_eval = detailed_policy_threshold_rows(yy, pp)
        for trow in threshold_eval:
            trow["target_column"] = col
        threshold_rows.extend(threshold_eval)
        masked_rows = [row for idx, row in enumerate(labeled_rows) if bool(mask[idx])]
        regime_specs = {
            "volatility": "realized_volatility_bps_fw60",
            "range": "rolling_range_bps_fw60",
            "volume_liquidity": "volume_zscore_fw60",
            "trend": "ema_slope_bps_fw60",
        }
        for regime_name, feature in regime_specs.items():
            values = np.asarray([safe_float(row.get(feature)) for row in masked_rows], dtype=np.float64)
            buckets = regime_bucket(values)
            for bucket in ["low", "medium", "high"]:
                bucket_mask = buckets == bucket
                if not bucket_mask.any():
                    continue
                m = binary_metrics(yy[bucket_mask], pp[bucket_mask], float(np.mean(yy)))
                m.update({"regime_feature": regime_name, "regime_bucket": bucket, "target_column": col})
                regime_rows.append(m)
        for regime_name, buckets_expected in {
            "daytime_overnight": ["daytime_utc", "overnight_utc"],
            "weekday_weekend": ["weekday", "weekend"],
        }.items():
            buckets = calendar_bucket(timestamps_arr, regime_name)
            for bucket in buckets_expected:
                bucket_mask = buckets == bucket
                if not bucket_mask.any():
                    continue
                m = binary_metrics(yy[bucket_mask], pp[bucket_mask], float(np.mean(yy)))
                m.update({"regime_feature": regime_name, "regime_bucket": bucket, "target_column": col})
                regime_rows.append(m)
    return summary_rows, reliability, regime_rows, bootstrap_rows, threshold_rows


def prediction_drift_rows(decision_rows: list[dict[str, Any]], target_columns: list[str], policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    primary = safe_float(policy.get("primary_threshold", {}).get("threshold"), 0.8)
    backup = safe_float(policy.get("backup_threshold", {}).get("threshold"), 0.7)
    prevalence = frozen_development_prevalence(policy)
    for col in target_columns:
        values = np.asarray([safe_float(row.get(f"prob_{col}")) for row in decision_rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "target_column": col,
                "rows": int(len(values)),
                "finite_rows": int(len(finite)),
                "development_event_prevalence": prevalence,
                "probability_mean": float(np.mean(finite)) if finite.size else math.nan,
                "probability_std": float(np.std(finite)) if finite.size else math.nan,
                "probability_p01": float(np.quantile(finite, 0.01)) if finite.size else math.nan,
                "probability_p50": float(np.quantile(finite, 0.50)) if finite.size else math.nan,
                "probability_p99": float(np.quantile(finite, 0.99)) if finite.size else math.nan,
                "mean_minus_development_prevalence": float(np.mean(finite) - prevalence) if finite.size and math.isfinite(prevalence) else math.nan,
                "primary_reject_fraction": float(np.mean(finite >= primary)) if finite.size else math.nan,
                "backup_reject_fraction": float(np.mean(finite >= backup)) if finite.size else math.nan,
                "near_zero_probability_fraction": float(np.mean(finite <= 1e-6)) if finite.size else math.nan,
                "near_one_probability_fraction": float(np.mean(finite >= 1 - 1e-6)) if finite.size else math.nan,
                "prediction_unique_fraction": float(len(np.unique(np.round(finite, 8))) / len(finite)) if finite.size else math.nan,
            }
        )
    return rows


def feature_drift_rows(
    decision_rows: list[dict[str, Any]],
    table: pd.DataFrame,
    selected_features: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not decision_rows or not selected_features:
        return rows
    ts_values = {safe_float(row.get("decision_timestamp")) for row in decision_rows}
    ts_values = {ts for ts in ts_values if math.isfinite(ts)}
    if not ts_values:
        return rows
    working = table.copy()
    working["decision_timestamp"] = pd.to_numeric(working["decision_timestamp"], errors="coerce")
    matched = working[working["decision_timestamp"].isin(ts_values)]
    if matched.empty:
        return rows
    for idx, feature in enumerate(selected_features):
        if feature not in matched.columns:
            rows.append({"feature": feature, "rows": 0, "status": "missing_from_feature_table"})
            continue
        values = pd.to_numeric(matched[feature], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        frozen_mean = float(mean[idx]) if idx < len(mean) else math.nan
        frozen_std = float(std[idx]) if idx < len(std) else math.nan
        if finite.size and math.isfinite(frozen_mean) and math.isfinite(frozen_std):
            z = (finite - frozen_mean) / max(abs(frozen_std), 1e-12)
        else:
            z = np.asarray([], dtype=np.float64)
        rows.append(
            {
                "feature": feature,
                "rows": int(len(values)),
                "finite_rows": int(len(finite)),
                "nonfinite_fraction": float(1.0 - len(finite) / max(len(values), 1)),
                "frozen_scaler_mean": frozen_mean,
                "frozen_scaler_std": frozen_std,
                "future_mean": float(np.mean(finite)) if finite.size else math.nan,
                "future_std": float(np.std(finite)) if finite.size else math.nan,
                "future_min": float(np.min(finite)) if finite.size else math.nan,
                "future_max": float(np.max(finite)) if finite.size else math.nan,
                "z_mean_vs_frozen_scaler": float(np.mean(z)) if z.size else math.nan,
                "z_std_vs_frozen_scaler": float(np.std(z)) if z.size else math.nan,
                "abs_z_gt_3_fraction": float(np.mean(np.abs(z) > 3.0)) if z.size else math.nan,
                "abs_z_gt_5_fraction": float(np.mean(np.abs(z) > 5.0)) if z.size else math.nan,
                "status": "ok",
            }
        )
    return rows


def feature_snapshot_rows(
    decision_rows: list[dict[str, Any]],
    table: pd.DataFrame,
    selected_features: list[str],
) -> list[dict[str, Any]]:
    if not decision_rows or not selected_features:
        return []
    ts_values = {safe_float(row.get("decision_timestamp")) for row in decision_rows}
    ts_values = {ts for ts in ts_values if math.isfinite(ts)}
    if not ts_values:
        return []
    working = table.copy()
    working["decision_timestamp"] = pd.to_numeric(working["decision_timestamp"], errors="coerce")
    matched = working[working["decision_timestamp"].isin(ts_values)]
    rows: list[dict[str, Any]] = []
    for _, source in matched.sort_values("decision_timestamp").iterrows():
        row = {"decision_timestamp": safe_float(source.get("decision_timestamp"))}
        for feature in selected_features:
            if feature in source.index:
                row[feature] = safe_float(source.get(feature))
        rows.append(row)
    return rows


def feature_drift_from_snapshots(
    snapshot_rows: list[dict[str, Any]],
    selected_features: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, feature in enumerate(selected_features):
        values = np.asarray([safe_float(row.get(feature)) for row in snapshot_rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        frozen_mean = float(mean[idx]) if idx < len(mean) else math.nan
        frozen_std = float(std[idx]) if idx < len(std) else math.nan
        if finite.size and math.isfinite(frozen_mean) and math.isfinite(frozen_std):
            z = (finite - frozen_mean) / max(abs(frozen_std), 1e-12)
        else:
            z = np.asarray([], dtype=np.float64)
        rows.append(
            {
                "feature": feature,
                "rows": int(len(values)),
                "finite_rows": int(len(finite)),
                "nonfinite_fraction": float(1.0 - len(finite) / max(len(values), 1)),
                "frozen_scaler_mean": frozen_mean,
                "frozen_scaler_std": frozen_std,
                "future_mean": float(np.mean(finite)) if finite.size else math.nan,
                "future_std": float(np.std(finite)) if finite.size else math.nan,
                "future_min": float(np.min(finite)) if finite.size else math.nan,
                "future_max": float(np.max(finite)) if finite.size else math.nan,
                "z_mean_vs_frozen_scaler": float(np.mean(z)) if z.size else math.nan,
                "z_std_vs_frozen_scaler": float(np.std(z)) if z.size else math.nan,
                "abs_z_gt_3_fraction": float(np.mean(np.abs(z) > 3.0)) if z.size else math.nan,
                "abs_z_gt_5_fraction": float(np.mean(np.abs(z) > 5.0)) if z.size else math.nan,
                "status": "ok",
                "source": "durable_cumulative_feature_snapshots",
            }
        )
    return rows


def evaluate_decision_rows(
    decision_rows: list[dict[str, Any]],
    table: pd.DataFrame,
    target_columns: list[str],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    labeled_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    if not decision_rows or "realized_volatility_bps_fw60" not in table.columns:
        return labeled_rows, summary_rows, reliability, regime_rows, bootstrap_rows, threshold_rows
    working = add_derived_low_path_labels(table, target_columns)
    working["decision_timestamp"] = pd.to_numeric(working["decision_timestamp"], errors="coerce")
    by_ts = {
        float(row["decision_timestamp"]): (int(pos), row)
        for pos, (_, row) in enumerate(working.iterrows())
        if math.isfinite(float(row["decision_timestamp"]))
    }
    by_target: dict[str, dict[str, list[Any]]] = {
        col: {"timestamps": [], "events": [], "probs": [], "matched_positions": []}
        for col in target_columns
    }
    for decision in decision_rows:
        ts = safe_float(decision.get("decision_timestamp"))
        source_item = by_ts.get(ts)
        if source_item is None:
            continue
        matched_pos, source = source_item
        vol = safe_float(source.get("realized_volatility_bps_fw60"))
        if not math.isfinite(vol) or vol <= 0:
            continue
        payload = {"decision_timestamp": ts}
        for feature in REGIME_FEATURE_COLUMNS:
            payload[feature] = safe_float(source.get(feature))
        has_label = False
        for col in target_columns:
            target = safe_float(source.get(col))
            prob = safe_float(decision.get(f"prob_{col}"))
            if not math.isfinite(target) or not math.isfinite(prob):
                continue
            event = float(abs(target) / max(vol, 1e-6) > 0.5)
            by_target[col]["timestamps"].append(ts)
            by_target[col]["events"].append(event)
            by_target[col]["probs"].append(prob)
            by_target[col]["matched_positions"].append(matched_pos)
            payload[f"event_{col}"] = int(event)
            payload[f"prob_{col}"] = prob
            payload[f"target_value_{col}"] = target
            payload[f"target_vol_units_{col}"] = abs(target) / max(vol, 1e-6)
            payload[f"derived_label_{col}"] = col not in table.columns
            has_label = True
        if has_label:
            labeled_rows.append(payload)
    if not labeled_rows:
        return labeled_rows, summary_rows, reliability, regime_rows, bootstrap_rows, threshold_rows
    summary_rows, reliability, regime_rows, bootstrap_rows, threshold_rows = metrics_from_labeled_rows(
        labeled_rows,
        target_columns,
        policy,
    )
    return labeled_rows, summary_rows, reliability, regime_rows, bootstrap_rows, threshold_rows


def main() -> int:
    candidate_dir = env_path("RAWSEQ_DOWNSIDE_SHADOW_CANDIDATE_DIR", DEFAULT_CANDIDATE_DIR)
    feature_table = env_path("RAWSEQ_DOWNSIDE_SHADOW_FEATURE_TABLE", DEFAULT_FEATURE_TABLE)
    output_root = env_path("RAWSEQ_DOWNSIDE_SHADOW_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / f"rawseq_downside_risk_future_shadow_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    contract = json.loads((candidate_dir / "rawseq_downside_risk_cpu_candidate_contract.json").read_text(encoding="utf-8-sig"))
    rule = json.loads((candidate_dir / "rawseq_downside_risk_future_acceptance_rule.json").read_text(encoding="utf-8-sig"))
    policy = json.loads((candidate_dir / "rawseq_downside_risk_policy_contract.json").read_text(encoding="utf-8-sig"))
    model_npz = np.load(contract["model_path"], allow_pickle=False)
    scaler_npz = np.load(contract["scalers_path"], allow_pickle=False)
    feature_columns = [str(x) for x in model_npz["feature_columns"]]
    target_columns = [str(x) for x in model_npz["target_columns"]]
    selected_indices = model_npz["selected_feature_indices"].astype(np.int64)
    selected_features = [feature_columns[int(i)] for i in selected_indices]
    coef = model_npz["coef"].astype(np.float64)
    mean = scaler_npz["feature_scaler_mean"].astype(np.float64)
    std = scaler_npz["feature_scaler_std"].astype(np.float64)
    consumed_cutoff = consumed_cutoff_from_source_npz(Path(contract["source_npz"]))
    header = pd.read_csv(feature_table, nrows=0).columns.tolist()
    usecols = [
        c
        for c in [
            "decision_timestamp",
            "price",
            "close",
            *selected_features,
            *target_columns,
            "realized_volatility_bps_fw60",
            "rolling_range_bps_fw60",
            "volume_zscore_fw60",
            "ema_slope_bps_fw60",
            "time_of_day_sin",
            "day_of_week_sin",
        ]
        if c in header
    ]
    table = pd.read_csv(feature_table, usecols=usecols)
    label_table = add_derived_low_path_labels(table, target_columns)
    timestamps = pd.to_numeric(table["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    future_mask = timestamps > consumed_cutoff
    future = table.loc[future_mask].copy()
    future_labels = label_table.loc[future_mask].copy()
    decisions: list[dict[str, Any]] = []
    run_id = out_dir.name
    logged_at_iso = datetime.now(UTC).isoformat()
    if not future.empty:
        x = future[selected_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        prob = predict_multihorizon(x, mean, std, coef)
        future_ts = pd.to_numeric(future["decision_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        primary = float(policy["primary_threshold"]["threshold"])
        backup = float(policy["backup_threshold"]["threshold"])
        for row_idx, ts in enumerate(future_ts):
            source_row = future_labels.iloc[row_idx]
            payload = {
                "decision_timestamp": float(ts),
                "first_logged_at_iso": logged_at_iso,
                "last_seen_at_iso": logged_at_iso,
                "first_run_id": run_id,
                "last_run_id": run_id,
                "paper_only": True,
                "orders": False,
                "promotion": False,
            }
            payload.update(label_availability_payload(source_row, target_columns))
            for idx, col in enumerate(target_columns):
                payload[f"prob_{col}"] = float(prob[row_idx, idx])
                payload[f"primary_reject_{col}"] = bool(prob[row_idx, idx] >= primary)
                payload[f"backup_reject_{col}"] = bool(prob[row_idx, idx] >= backup)
            decisions.append(payload)
    cumulative_dir = output_root / "rawseq_downside_risk_future_shadow_cumulative" / str(contract["contract_sha256"])[:12]
    cumulative_decisions_path = cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_decisions.csv"
    existing_decisions = read_csv_rows(cumulative_decisions_path)
    cumulative_decisions = merge_decisions(existing_decisions, decisions)
    duplicate_rows_seen = max(0, len(existing_decisions) + len(decisions) - len(cumulative_decisions))
    unique_new_decision_rows_logged = max(0, len(cumulative_decisions) - len(existing_decisions))
    true_forward_decisions = true_forward_decision_rows(decisions)
    cumulative_true_forward_decisions = true_forward_decision_rows(cumulative_decisions)
    backfill_decisions_this_run = len(decisions) - len(true_forward_decisions)
    cumulative_backfill_decisions = len(cumulative_decisions) - len(cumulative_true_forward_decisions)
    labeled_rows, summary_rows, reliability, regime_rows, bootstrap_rows, threshold_rows = evaluate_decision_rows(true_forward_decisions, label_table, target_columns, policy)
    cumulative_labeled_path = cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_labeled_results.csv"
    existing_cumulative_labeled_rows = read_csv_rows(cumulative_labeled_path)
    newly_available_labeled_rows, *_ = evaluate_decision_rows(cumulative_true_forward_decisions, label_table, target_columns, policy)
    cumulative_labeled_rows = merge_labeled_rows(existing_cumulative_labeled_rows, newly_available_labeled_rows)
    (
        cumulative_summary_rows,
        cumulative_reliability,
        cumulative_regime_rows,
        cumulative_bootstrap_rows,
        cumulative_threshold_rows,
    ) = metrics_from_labeled_rows(cumulative_labeled_rows, target_columns, policy)
    prediction_drift = prediction_drift_rows(true_forward_decisions, target_columns, policy)
    cumulative_prediction_drift = prediction_drift_rows(cumulative_true_forward_decisions, target_columns, policy)
    feature_drift = feature_drift_rows(true_forward_decisions, table, selected_features, mean, std)
    feature_snapshots = feature_snapshot_rows(true_forward_decisions, table, selected_features)
    cumulative_feature_snapshots_path = cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_feature_snapshots.csv"
    existing_feature_snapshots = read_csv_rows(cumulative_feature_snapshots_path)
    cumulative_feature_snapshots = merge_timestamp_rows(existing_feature_snapshots, feature_snapshots)
    cumulative_feature_drift = feature_drift_from_snapshots(cumulative_feature_snapshots, selected_features, mean, std)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_decisions.csv", decisions)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_true_forward_decisions.csv", true_forward_decisions)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_backfill_or_replay_decisions.csv", [row for row in decisions if row not in true_forward_decisions])
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_labeled_results.csv", labeled_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_summary.csv", summary_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_reliability_bins.csv", reliability)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_regime_metrics.csv", regime_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_bootstrap_ci.csv", bootstrap_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_threshold_utility.csv", threshold_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_prediction_drift.csv", prediction_drift)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_feature_drift.csv", feature_drift)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_feature_snapshots.csv", feature_snapshots)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_cumulative_decisions.csv", cumulative_decisions)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_cumulative_true_forward_decisions.csv", cumulative_true_forward_decisions)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_cumulative_backfill_or_replay_decisions.csv", [row for row in cumulative_decisions if row not in cumulative_true_forward_decisions])
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_cumulative_labeled_results.csv", cumulative_labeled_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_cumulative_summary.csv", cumulative_summary_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_cumulative_reliability_bins.csv", cumulative_reliability)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_cumulative_regime_metrics.csv", cumulative_regime_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_cumulative_bootstrap_ci.csv", cumulative_bootstrap_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_cumulative_threshold_utility.csv", cumulative_threshold_rows)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_cumulative_prediction_drift.csv", cumulative_prediction_drift)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_cumulative_feature_drift.csv", cumulative_feature_drift)
    write_csv(out_dir / "rawseq_downside_risk_future_shadow_cumulative_feature_snapshots.csv", cumulative_feature_snapshots)
    write_csv(cumulative_decisions_path, cumulative_decisions)
    write_csv(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_true_forward_decisions.csv", cumulative_true_forward_decisions)
    write_csv(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_backfill_or_replay_decisions.csv", [row for row in cumulative_decisions if row not in cumulative_true_forward_decisions])
    write_csv(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_labeled_results.csv", cumulative_labeled_rows)
    write_csv(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_summary.csv", cumulative_summary_rows)
    write_csv(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_reliability_bins.csv", cumulative_reliability)
    write_csv(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_regime_metrics.csv", cumulative_regime_rows)
    write_csv(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_bootstrap_ci.csv", cumulative_bootstrap_rows)
    write_csv(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_threshold_utility.csv", cumulative_threshold_rows)
    write_csv(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_prediction_drift.csv", cumulative_prediction_drift)
    write_csv(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_feature_drift.csv", cumulative_feature_drift)
    write_csv(cumulative_feature_snapshots_path, cumulative_feature_snapshots)
    rows = len(decisions)
    labeled = len(labeled_rows)
    cumulative_rows = len(cumulative_decisions)
    true_forward_rows = len(true_forward_decisions)
    cumulative_true_forward_rows = len(cumulative_true_forward_decisions)
    cumulative_labeled = len(cumulative_labeled_rows)
    max_nonoverlap = max([int(r.get("non_overlapping_h480_rows", 0)) for r in cumulative_summary_rows], default=0)
    max_days = max([safe_float(r.get("calendar_days"), 0.0) for r in cumulative_summary_rows], default=0.0)
    h480_summary = horizon_summary(cumulative_summary_rows, 480)
    h480_nonoverlap = int(h480_summary.get("non_overlapping_h480_rows", 0)) if h480_summary else 0
    h480_days = safe_float(h480_summary.get("calendar_days"), 0.0) if h480_summary else 0.0
    h480_events = int(h480_summary.get("events", 0)) if h480_summary else 0
    h480_non_events = int(h480_summary.get("non_events", 0)) if h480_summary else 0
    sufficient_events = all(int(r.get("events", 0)) >= MIN_EVENTS and int(r.get("non_events", 0)) >= MIN_NON_EVENTS for r in cumulative_summary_rows) if cumulative_summary_rows else False
    status = "accumulating"
    h480_sufficient_events = h480_events >= MIN_EVENTS and h480_non_events >= MIN_NON_EVENTS
    if h480_days >= MIN_CALENDAR_DAYS and h480_nonoverlap >= MIN_NON_OVERLAP_H480 and h480_sufficient_events:
        status = "ready_for_future_acceptance_evaluation"
    run_contract = {
        "candidate_dir": str(candidate_dir),
        "contract_sha256": contract["contract_sha256"],
        "acceptance_rule_sha256": rule["acceptance_rule_sha256"],
        "feature_table": str(feature_table),
        "feature_table_sha256": file_sha256(feature_table) if feature_table.exists() and feature_table.stat().st_size < 2_000_000_000 else "skipped_large_file",
        "consumed_cutoff_timestamp": consumed_cutoff,
        "future_rows_seen": rows,
        "true_forward_rows_scored_this_run": true_forward_rows,
        "backfill_or_replay_rows_scored_this_run": backfill_decisions_this_run,
        "labeled_rows": labeled,
        "cumulative_dir": str(cumulative_dir),
        "cumulative_decision_rows": cumulative_rows,
        "cumulative_true_forward_decision_rows": cumulative_true_forward_rows,
        "cumulative_backfill_or_replay_decision_rows": cumulative_backfill_decisions,
        "cumulative_labeled_rows": cumulative_labeled,
        "prior_cumulative_labeled_rows": len(existing_cumulative_labeled_rows),
        "newly_available_labeled_rows_seen": len(newly_available_labeled_rows),
        "feature_snapshot_rows_this_run": len(feature_snapshots),
        "cumulative_feature_snapshot_rows": len(cumulative_feature_snapshots),
        "prior_cumulative_feature_snapshot_rows": len(existing_feature_snapshots),
        "duplicate_rows_seen": duplicate_rows_seen,
        "prior_cumulative_decision_rows": len(existing_decisions),
        "rows_scored_this_run": rows,
        "unique_new_decision_rows_logged": unique_new_decision_rows_logged,
        "max_calendar_days": max_days,
        "max_non_overlapping_h480_rows": max_nonoverlap,
        "h480_calendar_days": h480_days,
        "h480_non_overlapping_rows": h480_nonoverlap,
        "h480_event_rows": h480_events,
        "h480_non_event_rows": h480_non_events,
        "min_events_required": MIN_EVENTS,
        "min_non_events_required": MIN_NON_EVENTS,
        "sufficient_event_and_non_event_examples": sufficient_events,
        "h480_sufficient_event_and_non_event_examples": h480_sufficient_events,
        "ready_conditions_met": status == "ready_for_future_acceptance_evaluation",
        "status": status,
        "min_calendar_days_required": MIN_CALENDAR_DAYS,
        "min_non_overlapping_h480_required": MIN_NON_OVERLAP_H480,
        "holdout_used_for_selection": False,
        "backfill_rows_excluded_from_acceptance": True,
        "acceptance_metrics_source": "true_forward_decisions_only",
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    attach_implementation_lock(run_contract, candidate_dir=candidate_dir, cumulative_dir=cumulative_dir, feature_table=feature_table)
    run_contract["run_contract_sha256"] = stable_hash(run_contract)
    write_json(out_dir / "rawseq_downside_risk_future_shadow_run_contract.json", run_contract)
    write_json(cumulative_dir / "rawseq_downside_risk_future_shadow_cumulative_contract.json", run_contract)
    lines = [
        "Rawseq downside-risk future paper shadow",
        f"Output: {out_dir}",
        f"Cumulative output: {cumulative_dir}",
        f"Candidate: {candidate_dir}",
        f"Consumed cutoff timestamp: {consumed_cutoff}",
        f"Future rows seen this run: {rows}",
        f"True-forward rows scored this run: {true_forward_rows}",
        f"Backfill/replay rows scored this run: {backfill_decisions_this_run}",
        f"Labeled rows this run: {labeled}",
        f"Cumulative decision rows: {cumulative_rows}",
        f"Cumulative true-forward decisions: {cumulative_true_forward_rows}",
        f"Cumulative backfill/replay decisions: {cumulative_backfill_decisions}",
        f"Cumulative labeled rows: {cumulative_labeled}",
        f"Unique new cumulative rows logged: {unique_new_decision_rows_logged}",
        f"Duplicate rows seen: {duplicate_rows_seen}",
        f"Max calendar days: {max_days}",
        f"Max non-overlapping h480 rows: {max_nonoverlap}",
        f"H480 calendar days: {h480_days}",
        f"H480 non-overlapping rows: {h480_nonoverlap}",
        f"H480 events/non-events: {h480_events}/{h480_non_events}",
        f"Sufficient event/non-event examples: {sufficient_events}",
        f"Status: {status}",
        "No retraining, recalibration, threshold changes, orders, or promotion.",
    ]
    (out_dir / "rawseq_downside_risk_future_shadow_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
