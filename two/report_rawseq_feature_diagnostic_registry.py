#!/usr/bin/env python3
"""Build a report-only diagnostic registry for materialized rawseq features."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_schema_contracts"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_feature_diagnostic_registry"
DEFAULT_INDICATOR_GLOB = "F:/rsio/rawseq_multi_horizon_indicator_full_contract_smoke/mh_indicator_*"


META_COLUMNS = [
    "schema_name",
    "schema_version",
    "schema_sha256",
    "created_at",
    "generator_path",
    "git_head",
    "git_status_dirty",
    "paper_only",
    "orders",
    "promotion",
    "champion_mutation",
]


@dataclass(frozen=True)
class DiagnosticThresholds:
    near_constant_variance: float = 1e-12
    high_missing_fraction: float = 0.25
    high_nonfinite_fraction: float = 0.01
    high_extreme_fraction: float = 0.01
    high_validation_out_of_range_fraction: float = 0.05
    unstable_block_cv: float = 2.0
    redundancy_correlation: float = 0.98


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def latest_dir(root_or_pattern: Path | str, child_glob: str | None = None) -> Path | None:
    if child_glob is None:
        pattern = str(root_or_pattern)
        if re.match(r"^[A-Za-z]:", pattern):
            import glob

            paths = [Path(p) for p in glob.glob(pattern)]
        else:
            paths = list(Path().glob(pattern))
    else:
        root = Path(root_or_pattern)
        paths = list(root.glob(child_glob)) if root.exists() else []
    paths = [p for p in paths if p.exists() and p.is_dir()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def git_status_dirty() -> bool:
    try:
        return bool(subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=10).stdout.strip())
    except Exception:
        return True


def stable_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def schema_meta(schema_name: str, schema_version: str, schema_sha256: str) -> dict[str, Any]:
    return {
        "schema_name": schema_name,
        "schema_version": schema_version,
        "schema_sha256": schema_sha256,
        "created_at": now_iso(),
        "generator_path": "scripts/tiny/report_rawseq_feature_diagnostic_registry.py",
        "git_head": git_head(),
        "git_status_dirty": git_status_dirty(),
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }


def with_meta(rows: list[dict[str, Any]], schema_name: str, schema_version: str, schema_sha256: str) -> list[dict[str, Any]]:
    meta = schema_meta(schema_name, schema_version, schema_sha256)
    return [{**meta, **row} for row in rows]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for col in META_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    frame.to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return math.nan
    if float(pair.iloc[:, 0].std(ddof=0)) <= 0 or float(pair.iloc[:, 1].std(ddof=0)) <= 0:
        return math.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def univariate_linear_metrics(train_x: pd.Series, train_y: pd.Series, val_x: pd.Series, val_y: pd.Series) -> dict[str, float]:
    train = pd.concat([pd.to_numeric(train_x, errors="coerce"), pd.to_numeric(train_y, errors="coerce")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    val = pd.concat([pd.to_numeric(val_x, errors="coerce"), pd.to_numeric(val_y, errors="coerce")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(train) < 5 or len(val) < 2:
        return {
            "train_univariate_slope": math.nan,
            "train_univariate_intercept": math.nan,
            "train_univariate_rmse": math.nan,
            "validation_univariate_rmse": math.nan,
            "train_univariate_r2": math.nan,
            "validation_univariate_r2": math.nan,
        }
    x = train.iloc[:, 0].to_numpy(dtype=float)
    y = train.iloc[:, 1].to_numpy(dtype=float)
    x_var = float(np.var(x))
    if x_var <= 0:
        slope = 0.0
    else:
        slope = float(np.cov(x, y, ddof=0)[0, 1] / x_var)
    intercept = float(np.mean(y) - slope * np.mean(x))
    train_pred = intercept + slope * x
    vx = val.iloc[:, 0].to_numpy(dtype=float)
    vy = val.iloc[:, 1].to_numpy(dtype=float)
    val_pred = intercept + slope * vx
    train_rmse = float(np.sqrt(np.mean((train_pred - y) ** 2)))
    val_rmse = float(np.sqrt(np.mean((val_pred - vy) ** 2)))
    train_base = float(np.mean(y))
    val_base = float(np.mean(y))
    train_sse = float(np.sum((train_pred - y) ** 2))
    val_sse = float(np.sum((val_pred - vy) ** 2))
    train_tss = float(np.sum((y - train_base) ** 2))
    val_tss = float(np.sum((vy - val_base) ** 2))
    return {
        "train_univariate_slope": slope,
        "train_univariate_intercept": intercept,
        "train_univariate_rmse": train_rmse,
        "validation_univariate_rmse": val_rmse,
        "train_univariate_r2": float(1.0 - train_sse / train_tss) if train_tss > 0 else math.nan,
        "validation_univariate_r2": float(1.0 - val_sse / val_tss) if val_tss > 0 else math.nan,
    }


def block_stability(frame: pd.DataFrame, feature: str, target: str, blocks: int = 6) -> dict[str, Any]:
    data = frame[[feature, target]].copy().replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < blocks * 5:
        return {
            "time_block_count": 0,
            "time_block_mean_cv": math.nan,
            "time_block_target_corr_sign_stability": math.nan,
            "time_block_abs_corr_mean": math.nan,
            "time_block_status": "insufficient_rows",
        }
    boundaries = np.linspace(0, len(data), blocks + 1, dtype=int)
    chunks = [data.iloc[boundaries[i] : boundaries[i + 1]] for i in range(blocks) if boundaries[i + 1] > boundaries[i]]
    means = []
    corrs = []
    for chunk in chunks:
        means.append(float(pd.to_numeric(chunk[feature], errors="coerce").mean()))
        corrs.append(safe_corr(chunk[feature], chunk[target]))
    mean_abs = float(np.mean(np.abs(means))) if means else math.nan
    mean_cv = float(np.std(means) / max(mean_abs, 1e-12)) if means else math.nan
    finite_corrs = np.asarray([c for c in corrs if np.isfinite(c)], dtype=float)
    if finite_corrs.size:
        majority_sign = 1.0 if np.nanmean(finite_corrs) >= 0 else -1.0
        sign_stability = float(np.mean(np.sign(finite_corrs) == majority_sign))
        abs_corr_mean = float(np.mean(np.abs(finite_corrs)))
    else:
        sign_stability = math.nan
        abs_corr_mean = math.nan
    status = "stable"
    if np.isfinite(mean_cv) and mean_cv > 2.0:
        status = "unstable_mean"
    if np.isfinite(sign_stability) and sign_stability < 0.5:
        status = "unstable_correlation"
    return {
        "time_block_count": len(chunks),
        "time_block_mean_cv": mean_cv,
        "time_block_target_corr_sign_stability": sign_stability,
        "time_block_abs_corr_mean": abs_corr_mean,
        "time_block_status": status,
    }


def find_redundancy_clusters(train_frame: pd.DataFrame, features: list[str], threshold: float) -> dict[str, dict[str, Any]]:
    if not features:
        return {}
    corr = train_frame[features].replace([np.inf, -np.inf], np.nan).corr().abs().fillna(0.0)
    parent = {feature: feature for feature in features}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            if float(corr.loc[left, right]) >= threshold:
                union(left, right)
    groups: dict[str, list[str]] = {}
    for feature in features:
        groups.setdefault(find(feature), []).append(feature)
    result = {}
    for idx, members in enumerate(sorted(groups.values(), key=lambda xs: (len(xs), xs[0]), reverse=True), start=1):
        members = sorted(members)
        representative = members[0]
        cluster_id = f"cluster_{idx:04d}" if len(members) > 1 else f"singleton_{idx:04d}"
        for feature in members:
            result[feature] = {
                "redundancy_cluster": cluster_id,
                "redundancy_cluster_size": len(members),
                "redundancy_representative": representative,
                "redundancy_cluster_members": ";".join(members),
            }
    return result


def recommended_action(row: dict[str, Any], thresholds: DiagnosticThresholds) -> tuple[str, str]:
    reasons = []
    if str(row.get("implementation_status", "")).lower() == "unresolved":
        reasons.append("formula_or_lineage_unresolved")
        return "unresolved", ";".join(reasons)
    if row.get("constant_status") in {"constant", "near_constant"}:
        reasons.append(str(row.get("constant_status")))
        return "invalid", ";".join(reasons)
    if float(row.get("nonfinite_fraction", 0) or 0) > thresholds.high_nonfinite_fraction:
        reasons.append("high_nonfinite_fraction")
    if float(row.get("missing_fraction", 0) or 0) > thresholds.high_missing_fraction:
        reasons.append("high_missing_fraction")
    if str(row.get("causal_status", "PASS")) != "PASS":
        reasons.append("causal_status_not_pass")
    if reasons:
        return "invalid", ";".join(reasons)
    if float(row.get("validation_out_of_range_fraction", 0) or 0) > thresholds.high_validation_out_of_range_fraction:
        reasons.append("validation_out_of_range")
    if float(row.get("extreme_value_rate", 0) or 0) > thresholds.high_extreme_fraction:
        reasons.append("extreme_value_rate")
    if str(row.get("time_block_status", "")) not in {"stable", ""}:
        reasons.append(str(row.get("time_block_status")))
    if reasons:
        return "unstable", ";".join(reasons)
    if int(float(row.get("redundancy_cluster_size", 1) or 1)) > 1 and row.get("materialized_feature_name") != row.get("redundancy_representative"):
        return "redundant", "high_train_correlation_cluster"
    if str(row.get("implementation_status", "")).lower() != "implemented":
        return "keep_for_ablation", "registered_but_not_fully_documented"
    if abs(float(row.get("max_abs_train_target_correlation", 0) or 0)) < 0.01 and float(row.get("best_validation_univariate_r2", -999) or -999) <= 0:
        return "keep_for_ablation", "weak_univariate_signal"
    return "keep", "passes_diagnostics"


def load_inputs() -> tuple[Path, Path, Path, Path]:
    indicator_env = os.getenv("RAWSEQ_FEATURE_DIAG_INDICATOR_RUN_DIR", "").strip()
    schema_env = os.getenv("RAWSEQ_FEATURE_DIAG_SCHEMA_DIR", "").strip()
    output_env = os.getenv("RAWSEQ_FEATURE_DIAG_OUTPUT_DIR", "").strip()
    training_table_env = os.getenv("RAWSEQ_FEATURE_DIAG_TRAINING_TABLE", "").strip()
    indicator_dir = resolve_path(indicator_env) if indicator_env else latest_dir(DEFAULT_INDICATOR_GLOB)
    if indicator_dir is None:
        raise SystemExit("Could not resolve RAWSEQ_FEATURE_DIAG_INDICATOR_RUN_DIR and no default indicator artifact was found.")
    schema_dir = resolve_path(schema_env) if schema_env else latest_dir(DEFAULT_SCHEMA_ROOT, "rawseq_schema_contract_*")
    if schema_dir is None:
        raise SystemExit("Could not resolve RAWSEQ_FEATURE_DIAG_SCHEMA_DIR and no schema contract report was found.")
    training_table = resolve_path(training_table_env) if training_table_env else indicator_dir / "multi_horizon_training_table.csv"
    output_root = resolve_path(output_env) if output_env else DEFAULT_OUTPUT_ROOT
    return indicator_dir, schema_dir, training_table, output_root


def build_ablation_units(features: pd.DataFrame) -> list[dict[str, Any]]:
    registered = features[features["recommended_action"].isin(["keep", "keep_for_ablation", "redundant", "unstable"])].copy()
    freezeable = registered[~registered["recommended_action"].isin(["unresolved", "invalid"])].copy()
    family_map = {fam: sorted(group["materialized_feature_name"].astype(str).tolist()) for fam, group in freezeable.groupby("feature_family")}
    rows = []
    raw_features = family_map.get("raw", [])
    def add(name: str, mode: str, selected: list[str], notes: str) -> None:
        rows.append(
            {
                "ablation_unit": name,
                "selection_mode": mode,
                "feature_count": len(selected),
                "features": ";".join(selected),
                "included_families": ";".join(sorted({str(features.loc[features["materialized_feature_name"].eq(f), "feature_family"].iloc[0]) for f in selected if not features.loc[features["materialized_feature_name"].eq(f)].empty})),
                "notes": notes,
                "freezeable_contract_candidate": not any(
                    str(features.loc[features["materialized_feature_name"].eq(f), "recommended_action"].iloc[0]) in {"unresolved", "invalid"}
                    for f in selected
                    if not features.loc[features["materialized_feature_name"].eq(f)].empty
                ),
            }
        )
    add("raw", "family", raw_features, "raw-only baseline unit")
    for family in ["trend", "momentum", "volatility", "breakout", "volume", "order_book", "regime", "cross_market"]:
        selected = sorted(set(raw_features + family_map.get(family, [])))
        add(f"raw_plus_{family}", "raw_plus_family", selected, f"raw plus {family} family")
    all_features = sorted(freezeable["materialized_feature_name"].astype(str).tolist())
    add("all_registered_features", "all_freezeable_registered", all_features, "excludes invalid and unresolved formula/lineage rows")
    for family in sorted(family_map):
        selected = sorted([f for fam, members in family_map.items() if fam != family for f in members])
        add(f"all_minus_{family}", "all_minus_one_family", selected, f"all freezeable registered features except {family}")
    return rows


def main() -> int:
    indicator_dir, schema_dir, training_table, output_root = load_inputs()
    out_dir = output_root / f"rawseq_feature_diagnostic_registry_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    thresholds = DiagnosticThresholds(
        near_constant_variance=float(os.getenv("RAWSEQ_FEATURE_DIAG_NEAR_CONSTANT_VARIANCE", "1e-12")),
        high_missing_fraction=float(os.getenv("RAWSEQ_FEATURE_DIAG_HIGH_MISSING_FRACTION", "0.25")),
        high_nonfinite_fraction=float(os.getenv("RAWSEQ_FEATURE_DIAG_HIGH_NONFINITE_FRACTION", "0.01")),
        high_extreme_fraction=float(os.getenv("RAWSEQ_FEATURE_DIAG_HIGH_EXTREME_FRACTION", "0.01")),
        high_validation_out_of_range_fraction=float(os.getenv("RAWSEQ_FEATURE_DIAG_HIGH_VALIDATION_OOR_FRACTION", "0.05")),
        unstable_block_cv=float(os.getenv("RAWSEQ_FEATURE_DIAG_UNSTABLE_BLOCK_CV", "2.0")),
        redundancy_correlation=float(os.getenv("RAWSEQ_FEATURE_DIAG_REDUNDANCY_CORRELATION", "0.98")),
    )
    block_count = int(float(os.getenv("RAWSEQ_FEATURE_DIAG_TIME_BLOCKS", "6")))

    print(f"Resolved indicator run: {indicator_dir}")
    print(f"Resolved schema dir: {schema_dir}")
    print(f"Resolved training table: {training_table}")
    print(f"Output dir: {out_dir}")

    feature_schema = pd.read_csv(schema_dir / "materialized_feature_columns.csv")
    target_schema = pd.read_csv(schema_dir / "materialized_target_columns.csv")
    family_manifest = pd.read_csv(indicator_dir / "feature_family_manifest.csv")
    table = pd.read_csv(training_table)

    feature_cols = [str(x) for x in feature_schema["materialized_feature_name"].astype(str) if str(x) in table.columns]
    target_cols = [str(x) for x in target_schema["materialized_target_column"].astype(str) if str(x) in table.columns]
    if not feature_cols:
        raise SystemExit("No materialized feature columns from schema were found in the training table.")
    if not target_cols:
        raise SystemExit("No materialized target columns from schema were found in the training table.")

    train = table[table["split"].eq("train")].copy() if "split" in table else table.iloc[: int(len(table) * 0.7)].copy()
    validation = table[table["split"].eq("validation")].copy() if "split" in table else table.iloc[int(len(table) * 0.7) : int(len(table) * 0.85)].copy()
    all_eval = table[table["split"].isin(["train", "validation", "untouched_holdout"])].copy() if "split" in table else table.copy()

    redundancy = find_redundancy_clusters(train, feature_cols, thresholds.redundancy_correlation)
    family_lookup = dict(zip(family_manifest["feature"].astype(str), family_manifest["feature_family"].astype(str)))
    schema_lookup = feature_schema.set_index("materialized_feature_name").to_dict("index")

    registry_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []

    for feature in feature_cols:
        series_all = numeric_series(table, feature)
        series_train = numeric_series(train, feature)
        series_val = numeric_series(validation, feature) if feature in validation else pd.Series(dtype=float)
        finite_all = np.isfinite(series_all.to_numpy(dtype=float, na_value=np.nan))
        finite_train = np.isfinite(series_train.to_numpy(dtype=float, na_value=np.nan))
        total_rows = int(len(series_all))
        missing_col = f"{feature}_missing"
        missing_fraction = float(pd.to_numeric(table[missing_col], errors="coerce").fillna(0).mean()) if missing_col in table.columns else float(series_all.isna().mean())
        nonfinite_fraction = float(1.0 - finite_all.mean()) if total_rows else math.nan
        variance = float(np.nanvar(series_train.to_numpy(dtype=float, na_value=np.nan))) if len(series_train) else math.nan
        constant_status = "variable"
        finite_unique = pd.Series(series_train[finite_train]).nunique(dropna=True) if len(series_train) else 0
        if finite_unique <= 1:
            constant_status = "constant"
        elif np.isfinite(variance) and variance <= thresholds.near_constant_variance:
            constant_status = "near_constant"
        train_mean = float(np.nanmean(series_train.to_numpy(dtype=float, na_value=np.nan))) if len(series_train) else math.nan
        train_std = float(np.nanstd(series_train.to_numpy(dtype=float, na_value=np.nan))) if len(series_train) else math.nan
        train_min = float(np.nanmin(series_train.to_numpy(dtype=float, na_value=np.nan))) if finite_train.any() else math.nan
        train_max = float(np.nanmax(series_train.to_numpy(dtype=float, na_value=np.nan))) if finite_train.any() else math.nan
        if np.isfinite(train_mean) and np.isfinite(train_std) and train_std > 0:
            extreme_value_rate = float((np.abs((series_all - train_mean) / train_std) > 6.0).mean())
        else:
            extreme_value_rate = 0.0
        if len(series_val) and np.isfinite(train_min) and np.isfinite(train_max):
            val_finite = series_val[np.isfinite(series_val.to_numpy(dtype=float, na_value=np.nan))]
            validation_out_of_range_fraction = float(((val_finite < train_min) | (val_finite > train_max)).mean()) if len(val_finite) else math.nan
        else:
            validation_out_of_range_fraction = math.nan

        best_corr = 0.0
        best_corr_target = ""
        best_val_r2 = -math.inf
        best_val_r2_target = ""
        corr_by_horizon: dict[str, float] = {}
        for target in target_cols:
            target_train = numeric_series(train, target)
            target_val = numeric_series(validation, target) if target in validation else pd.Series(dtype=float)
            train_corr = safe_corr(series_train, target_train)
            val_corr = safe_corr(series_val, target_val) if len(series_val) else math.nan
            corr_by_horizon[f"{target}_train_corr"] = train_corr
            if np.isfinite(train_corr) and abs(train_corr) > abs(best_corr):
                best_corr = train_corr
                best_corr_target = target
            corr_rows.append(
                {
                    "materialized_feature_name": feature,
                    "target_column": target,
                    "horizon_buckets": target_schema.loc[target_schema["materialized_target_column"].eq(target), "horizon_buckets"].iloc[0],
                    "train_correlation": train_corr,
                    "validation_correlation": val_corr,
                    "abs_train_correlation": abs(train_corr) if np.isfinite(train_corr) else math.nan,
                    "abs_validation_correlation": abs(val_corr) if np.isfinite(val_corr) else math.nan,
                }
            )
            metrics = univariate_linear_metrics(series_train, target_train, series_val, target_val)
            if np.isfinite(metrics["validation_univariate_r2"]) and metrics["validation_univariate_r2"] > best_val_r2:
                best_val_r2 = metrics["validation_univariate_r2"]
                best_val_r2_target = target
            baseline_rows.append(
                {
                    "materialized_feature_name": feature,
                    "target_column": target,
                    "horizon_buckets": target_schema.loc[target_schema["materialized_target_column"].eq(target), "horizon_buckets"].iloc[0],
                    **metrics,
                }
            )
        stability = block_stability(all_eval, feature, best_corr_target or target_cols[0], blocks=block_count)
        block_rows.append({"materialized_feature_name": feature, "stability_target_column": best_corr_target or target_cols[0], **stability})
        schema_row = schema_lookup.get(feature, {})
        base = {
            "materialized_feature_name": feature,
            "feature_name": schema_row.get("feature_name", feature),
            "feature_family": family_lookup.get(feature, schema_row.get("feature_family", "")),
            "feature_group_membership": schema_row.get("feature_group_membership", ""),
            "implementation_status": schema_row.get("implementation_status", ""),
            "formula": schema_row.get("formula", ""),
            "causal_status": schema_row.get("causal_status", ""),
            "availability_status": schema_row.get("availability_status", ""),
            "window_buckets": schema_row.get("window_buckets", ""),
            "source_columns": schema_row.get("source_columns", ""),
            "total_rows": total_rows,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "variance": variance,
            "nonfinite_fraction": nonfinite_fraction,
            "missing_fraction": missing_fraction,
            "constant_status": constant_status,
            "extreme_value_rate": extreme_value_rate,
            "train_scaler_mean": train_mean,
            "train_scaler_std": train_std,
            "train_scaler_min": train_min,
            "train_scaler_max": train_max,
            "validation_out_of_range_fraction": validation_out_of_range_fraction,
            "max_abs_train_target_correlation": abs(best_corr) if np.isfinite(best_corr) else math.nan,
            "best_train_target_correlation": best_corr,
            "best_correlation_target": best_corr_target,
            "best_validation_univariate_r2": best_val_r2 if np.isfinite(best_val_r2) else math.nan,
            "best_validation_univariate_target": best_val_r2_target,
            **redundancy.get(feature, {}),
            **stability,
        }
        base.update(corr_by_horizon)
        action, reason = recommended_action(base, thresholds)
        base["recommended_action"] = action
        base["recommended_action_reason"] = reason
        base["freezeable_contract_eligible"] = action not in {"invalid", "unresolved"}
        registry_rows.append(base)

    registry_frame = pd.DataFrame(registry_rows)
    ablation_rows = build_ablation_units(registry_frame)
    cluster_rows = []
    for cluster_id, group in registry_frame.groupby("redundancy_cluster", dropna=False):
        members = sorted(group["materialized_feature_name"].astype(str).tolist())
        cluster_rows.append(
            {
                "redundancy_cluster": cluster_id,
                "cluster_size": len(members),
                "representative": group["redundancy_representative"].iloc[0] if "redundancy_representative" in group else "",
                "members": ";".join(members),
                "families": ";".join(sorted(set(group["feature_family"].astype(str)))),
            }
        )

    schema_hash = stable_sha(
        {
            "indicator_dir": str(indicator_dir),
            "schema_dir": str(schema_dir),
            "training_table": str(training_table),
            "features": feature_cols,
            "targets": target_cols,
            "thresholds": thresholds.__dict__,
        }
    )
    schema_version = "1.0.0"
    write_csv(out_dir / "feature_diagnostic_registry.csv", with_meta(registry_rows, "rawseq_feature_diagnostic_registry", schema_version, schema_hash))
    write_csv(out_dir / "feature_target_correlations.csv", with_meta(corr_rows, "rawseq_feature_target_correlations", schema_version, schema_hash))
    write_csv(out_dir / "feature_univariate_baselines.csv", with_meta(baseline_rows, "rawseq_feature_univariate_baselines", schema_version, schema_hash))
    write_csv(out_dir / "feature_time_block_stability.csv", with_meta(block_rows, "rawseq_feature_time_block_stability", schema_version, schema_hash))
    write_csv(out_dir / "feature_redundancy_clusters.csv", with_meta(cluster_rows, "rawseq_feature_redundancy_clusters", schema_version, schema_hash))
    write_csv(out_dir / "feature_family_ablation_units.csv", with_meta(ablation_rows, "rawseq_feature_family_ablation_units", schema_version, schema_hash))

    action_counts = registry_frame["recommended_action"].value_counts().to_dict()
    family_counts = registry_frame.groupby(["feature_family", "recommended_action"]).size().reset_index(name="count").to_dict("records")
    contract = {
        **schema_meta("rawseq_feature_diagnostic_contract", schema_version, schema_hash),
        "indicator_run_dir": str(indicator_dir),
        "schema_dir": str(schema_dir),
        "training_table": str(training_table),
        "output_dir": str(out_dir),
        "feature_count": len(feature_cols),
        "target_count": len(target_cols),
        "thresholds": thresholds.__dict__,
        "action_counts": action_counts,
        "family_action_counts": family_counts,
        "safety": {"paper_only": True, "orders": False, "promotion": False, "champion_mutation": False, "training": False},
    }
    write_json(out_dir / "feature_diagnostic_contract.json", contract)
    summary = [
        "# Rawseq Feature Diagnostic Registry",
        "",
        f"Created at: {contract['created_at']}",
        f"Indicator run: `{indicator_dir}`",
        f"Schema dir: `{schema_dir}`",
        f"Training table: `{training_table}`",
        "",
        "## Counts",
        f"- materialized features diagnosed: {len(feature_cols)}",
        f"- target horizons diagnosed: {len(target_cols)}",
        f"- target-correlation rows: {len(corr_rows)}",
        f"- univariate baseline rows: {len(baseline_rows)}",
        f"- redundancy clusters: {len(cluster_rows)}",
        f"- ablation units: {len(ablation_rows)}",
        "",
        "## Recommended Actions",
    ]
    for action, count in sorted(action_counts.items()):
        summary.append(f"- {action}: {count}")
    summary.extend(
        [
            "",
            "## Policy",
            "- `invalid` and `unresolved` features should be excluded from freezeable model contracts.",
            "- `redundant`, `unstable`, and `keep_for_ablation` features may remain in research ablations but should not be promoted blindly.",
            "- Existing schema WARN items do not block diagnostics; they identify documentation or metadata work before freezeable contracts.",
            "",
            "Safety: report_only=true; no training; no discovery; no ensemble search; no freeze; no promotion; no champion mutation; no orders.",
        ]
    )
    text = "\n".join(summary) + "\n"
    (out_dir / "feature_diagnostic_summary.md").write_text(text, encoding="utf-8")
    (out_dir / "feature_diagnostic_summary.txt").write_text(text, encoding="utf-8")

    print("Rawseq feature diagnostic registry complete")
    print(f"Output: {out_dir}")
    print(f"Features: {len(feature_cols)}")
    print(f"Targets: {len(target_cols)}")
    print(f"Actions: {action_counts}")
    print("Safety: no training. No discovery. No orders. No promotion. No champion mutation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
