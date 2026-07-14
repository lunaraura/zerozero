#!/usr/bin/env python3
"""Train/validation-only diagnostics for rawseq materialized feature columns."""

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
CONFIG_DIR = PROJECT_ROOT / "configs" / "rawseq"
DEFAULT_SCHEMA_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_schema_contracts"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_feature_diagnostics"
DEFAULT_INDICATOR_GLOB = "F:/rsio/rawseq_multi_horizon_indicator_full_contract_smoke/mh_indicator_*"
ALPHA = 1.0
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
class Thresholds:
    correlation_threshold: float
    validation_blocks: int
    min_train_finite_fraction: float
    min_validation_finite_fraction: float
    near_constant_std: float


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def latest_dir(pattern_or_root: str | Path, child_glob: str | None = None) -> Path | None:
    if child_glob is not None:
        root = Path(pattern_or_root)
        paths = list(root.glob(child_glob)) if root.exists() else []
    else:
        pattern = str(pattern_or_root)
        if re.match(r"^[A-Za-z]:", pattern):
            import glob

            paths = [Path(p) for p in glob.glob(pattern)]
        else:
            paths = list(Path().glob(pattern))
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def column_hash(columns: list[str]) -> str:
    return stable_json_sha256(list(columns))


def schema_meta(name: str, version: str, schema_hash: str) -> dict[str, Any]:
    return {
        "schema_name": name,
        "schema_version": version,
        "schema_sha256": schema_hash,
        "created_at": now_iso(),
        "generator_path": "scripts/tiny/report_rawseq_feature_diagnostics.py",
        "git_head": git_head(),
        "git_status_dirty": git_status_dirty(),
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }


def with_meta(rows: list[dict[str, Any]], name: str, version: str, schema_hash: str) -> list[dict[str, Any]]:
    meta = schema_meta(name, version, schema_hash)
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


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def finite_fraction(series: pd.Series) -> float:
    values = to_num(series)
    return float(np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).mean()) if len(values) else math.nan


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([to_num(x), to_num(y)], axis=1).dropna()
    if len(pair) < 3:
        return math.nan
    if float(pair.iloc[:, 0].std(ddof=0)) <= 0 or float(pair.iloc[:, 1].std(ddof=0)) <= 0:
        return math.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def split_frame(table: pd.DataFrame, split: str) -> pd.DataFrame:
    return table[table["split"].astype(str).eq(split)].copy()


def scaler_params(train_series: pd.Series) -> dict[str, float]:
    x = to_num(train_series)
    return {
        "train_scaler_center": float(x.mean()) if len(x.dropna()) else math.nan,
        "train_scaler_scale": float(x.std(ddof=0)) if len(x.dropna()) else math.nan,
    }


def ridge_univariate_metrics(train_x: pd.Series, train_y: pd.Series, val_x: pd.Series, val_y: pd.Series, alpha: float = ALPHA) -> dict[str, float]:
    train = pd.concat([to_num(train_x), to_num(train_y)], axis=1).dropna()
    val = pd.concat([to_num(val_x), to_num(val_y)], axis=1).dropna()
    if len(train) < 5 or len(val) < 2:
        return {
            "ridge_alpha": alpha,
            "ridge_intercept": math.nan,
            "ridge_slope": math.nan,
            "train_rmse": math.nan,
            "validation_rmse": math.nan,
            "train_r2": math.nan,
            "validation_r2": math.nan,
        }
    x = train.iloc[:, 0].to_numpy(dtype=float)
    y = train.iloc[:, 1].to_numpy(dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    xc = x - x_mean
    yc = y - y_mean
    slope = float(np.sum(xc * yc) / (np.sum(xc * xc) + alpha))
    intercept = y_mean - slope * x_mean
    train_pred = intercept + slope * x
    vx = val.iloc[:, 0].to_numpy(dtype=float)
    vy = val.iloc[:, 1].to_numpy(dtype=float)
    val_pred = intercept + slope * vx
    return {
        "ridge_alpha": alpha,
        "ridge_intercept": float(intercept),
        "ridge_slope": float(slope),
        "train_rmse": rmse(y, train_pred),
        "validation_rmse": rmse(vy, val_pred),
        "train_r2": r2_score(y, train_pred),
        "validation_r2": r2_score(vy, val_pred),
    }


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2))) if len(actual) else math.nan


def r2_score(actual: np.ndarray, pred: np.ndarray) -> float:
    if not len(actual):
        return math.nan
    denom = float(np.sum((actual - np.mean(actual)) ** 2))
    return float(1.0 - np.sum((pred - actual) ** 2) / denom) if denom > 0 else math.nan


def validation_block_metrics(validation: pd.DataFrame, feature: str, targets: list[str], blocks: int) -> list[dict[str, Any]]:
    if validation.empty:
        return []
    bounds = np.linspace(0, len(validation), blocks + 1, dtype=int)
    rows = []
    for block_index in range(blocks):
        block = validation.iloc[bounds[block_index] : bounds[block_index + 1]]
        if block.empty:
            continue
        values = to_num(block[feature])
        for target in targets:
            rows.append(
                {
                    "materialized_feature_name": feature,
                    "validation_block_index": block_index,
                    "block_rows": int(len(block)),
                    "target_column": target,
                    "block_feature_mean": float(values.mean()) if len(values.dropna()) else math.nan,
                    "block_feature_std": float(values.std(ddof=0)) if len(values.dropna()) else math.nan,
                    "block_target_correlation": safe_corr(block[feature], block[target]),
                }
            )
    return rows


def stability_score(block_rows: list[dict[str, Any]]) -> tuple[float, str]:
    corrs = np.asarray([row["block_target_correlation"] for row in block_rows if np.isfinite(row.get("block_target_correlation", math.nan))], dtype=float)
    means = np.asarray([row["block_feature_mean"] for row in block_rows if np.isfinite(row.get("block_feature_mean", math.nan))], dtype=float)
    if not len(means):
        return math.nan, "insufficient_validation_blocks"
    mean_cv = float(np.std(means) / max(float(np.mean(np.abs(means))), 1e-12))
    if len(corrs):
        sign_share = float(max(np.mean(corrs >= 0), np.mean(corrs < 0)))
    else:
        sign_share = 0.0
    score = float(max(0.0, min(1.0, sign_share / (1.0 + min(mean_cv, 10.0)))))
    status = "stable" if score >= 0.25 else "unstable"
    return score, status


def formula_status(row: dict[str, Any]) -> str:
    impl = str(row.get("implementation_status", "")).lower()
    formula = str(row.get("formula", "")).strip().lower()
    if impl == "unresolved" or formula in {"", "unknown", "nan"}:
        return "unresolved"
    if "implementation_specific" in formula or "inspect implementation" in formula:
        return "implementation_specific"
    return "resolved"


def resolve_formula(row: dict[str, Any]) -> str:
    text = str(row.get("formula", ""))
    window = row.get("window_buckets", "")
    if pd.notna(window) and str(window) not in {"", "nan"}:
        text = text.replace("window", str(int(float(window))) if str(window).replace(".", "", 1).isdigit() else str(window))
    return text


def choose_representative(candidates: pd.DataFrame) -> str:
    def key(row: pd.Series) -> tuple[Any, ...]:
        formula_ok = 0 if row.get("formula_status") == "resolved" and row.get("causal_status") == "PASS" else 1
        missing_raw = row.get("train_missing_fraction", 1.0)
        missing = float(missing_raw) if pd.notna(missing_raw) else 1.0
        stability_raw = row.get("validation_stability_score", 0.0)
        stability = -float(stability_raw) if pd.notna(stability_raw) else 0.0
        window = row.get("window_buckets", 10**9)
        try:
            window_value = int(float(window)) if str(window) not in {"", "nan"} else 0
        except Exception:
            window_value = 0
        return (formula_ok, missing, stability, window_value, str(row.get("materialized_feature_name", "")))

    ordered = sorted([row for _, row in candidates.iterrows()], key=key)
    return str(ordered[0].get("materialized_feature_name", "")) if ordered else ""


def redundancy(train: pd.DataFrame, diag_rows: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    valid_features = [row["materialized_feature_name"] for row in diag_rows if row["recommended_action_pre_redundancy"] not in {"invalid", "unresolved"}]
    if not valid_features:
        return [], [], {}
    corr = train[valid_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).corr().abs().fillna(0.0)
    pairs = []
    parent = {feature: feature for feature in valid_features}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, left in enumerate(valid_features):
        for right in valid_features[i + 1 :]:
            value = float(corr.loc[left, right])
            if value >= threshold:
                pairs.append({"left_feature": left, "right_feature": right, "train_abs_correlation": value})
                union(left, right)
    group_members: dict[str, list[str]] = {}
    for feature in valid_features:
        group_members.setdefault(find(feature), []).append(feature)
    diag_frame = pd.DataFrame(diag_rows)
    clusters = []
    feature_to_rep = {}
    for index, members in enumerate(sorted(group_members.values(), key=lambda xs: (len(xs), xs[0]), reverse=True), start=1):
        cluster_id = f"cluster_{index:04d}" if len(members) > 1 else f"singleton_{index:04d}"
        sub = diag_frame[diag_frame["materialized_feature_name"].isin(members)].copy()
        rep = choose_representative(sub)
        for feature in members:
            feature_to_rep[feature] = rep
        clusters.append(
            {
                "redundancy_cluster_id": cluster_id,
                "cluster_size": len(members),
                "representative_feature": rep,
                "members": ";".join(sorted(members)),
            }
        )
    cluster_by_feature = {}
    for cluster in clusters:
        for member in str(cluster["members"]).split(";"):
            cluster_by_feature[member] = cluster["redundancy_cluster_id"]
    for row in pairs:
        row["left_cluster_id"] = cluster_by_feature.get(row["left_feature"], "")
        row["right_cluster_id"] = cluster_by_feature.get(row["right_feature"], "")
    return pairs, clusters, feature_to_rep


def initial_action(row: dict[str, Any], thresholds: Thresholds) -> tuple[str, str]:
    reasons = []
    if row["formula_status"] == "unresolved":
        return "unresolved", "formula_or_implementation_lineage_unresolved"
    if row["causal_status"] != "PASS":
        reasons.append("causal_status_not_pass")
    if row["train_finite_fraction"] < thresholds.min_train_finite_fraction:
        reasons.append("low_train_finite_fraction")
    if row["validation_finite_fraction"] < thresholds.min_validation_finite_fraction:
        reasons.append("low_validation_finite_fraction")
    if row["near_constant"]:
        reasons.append("near_constant")
    if reasons:
        return "invalid", ";".join(reasons)
    if row["validation_stability_status"] == "unstable" or row["distribution_shift_score"] > 5.0 or row["extreme_value_warning"]:
        return "unstable", "distribution_or_block_instability"
    if row["formula_status"] != "resolved":
        return "keep_for_ablation", "formula_implementation_specific"
    return "keep", "passes_train_validation_diagnostics"


def load_sequence_feature_indexes(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result = {}
    for path in paths:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            features = [str(x) for x in data["feature_columns"].astype(str)] if "feature_columns" in data.files else []
            targets = [str(x) for x in data["target_columns"].astype(str)] if "target_columns" in data.files else []
            x_shape = list(data["X"].shape) if "X" in data.files else []
            y_shape = list(data["y"].shape) if "y" in data.files else []
        for idx, feature in enumerate(features):
            result.setdefault(
                feature,
                {
                    "sequence_dataset_path": str(path),
                    "feature_index": idx,
                    "sequence_x_shape": json.dumps(x_shape),
                    "sequence_y_shape": json.dumps(y_shape),
                    "sequence_target_columns": ";".join(targets),
                },
            )
    return result


def parse_sequence_dataset_env(text: str, sequence_manifest: pd.DataFrame) -> list[Path]:
    if text.strip():
        return [resolve_path(item.strip()) for item in text.split(";") if item.strip()]
    paths = []
    for _, row in sequence_manifest.iterrows():
        path = Path(str(row.get("path_npz", "")))
        paths.append(path if path.is_absolute() else path.resolve())
    return paths


def build_bundle(name: str, rows: pd.DataFrame, selected: list[str], schema_hashes: dict[str, str], evidence: dict[str, Any]) -> dict[str, Any]:
    selected = [feature for feature in selected if feature in set(rows["materialized_feature_name"])]
    return {
        "bundle_name": name,
        "ordered_feature_columns": selected,
        "feature_count": len(selected),
        "feature_columns_sha256": column_hash(selected),
        "schema_hashes": schema_hashes,
        "selection_evidence": evidence,
        "untouched_holdout_used": False,
        "untouched_holdout_statement": "Untouched holdout rows were not used for diagnostics, ranking, clustering, or bundle construction.",
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }


def select_feature_bundles(diag_frame: pd.DataFrame, schema_hashes: dict[str, str], evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    eligible = diag_frame[~diag_frame["recommended_action"].isin(["invalid", "unresolved"])].copy()
    keep = diag_frame[diag_frame["recommended_action"].eq("keep")].copy()
    raw_keep = keep[keep["feature_family"].eq("raw")]["materialized_feature_name"].tolist()
    minimal = sorted(
        set(
            raw_keep
            + keep.sort_values(["feature_family", "materialized_feature_name"])
            .groupby("feature_family")
            .head(3)["materialized_feature_name"]
            .tolist()
        )
    )
    balanced = sorted(
        set(
            minimal
            + diag_frame[diag_frame["recommended_action"].isin(["keep", "keep_for_ablation"])]
            .groupby("feature_family")
            .head(8)["materialized_feature_name"]
            .tolist()
        )
    )
    full = eligible.sort_values("materialized_feature_name")["materialized_feature_name"].tolist()
    return {
        "minimal_core": build_bundle("minimal_core", diag_frame, minimal, schema_hashes, evidence),
        "balanced_research": build_bundle("balanced_research", diag_frame, balanced, schema_hashes, evidence),
        "full_registered": build_bundle("full_registered", diag_frame, full, schema_hashes, evidence),
    }


def main() -> int:
    indicator_env = os.getenv("RAWSEQ_FEATURE_DIAG_INDICATOR_RUN_DIR", "").strip()
    indicator_dir = resolve_path(indicator_env) if indicator_env else latest_dir(DEFAULT_INDICATOR_GLOB)
    if indicator_dir is None:
        raise SystemExit("Could not resolve RAWSEQ_FEATURE_DIAG_INDICATOR_RUN_DIR.")
    output_root = resolve_path(os.getenv("RAWSEQ_FEATURE_DIAG_OUTPUT_DIR", str(DEFAULT_OUTPUT_ROOT)))
    out_dir = output_root / f"rawseq_feature_diagnostics_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    schema_dir = latest_dir(DEFAULT_SCHEMA_ROOT, "rawseq_schema_contract_*")
    if schema_dir is None:
        raise SystemExit("No schema packet found under data/research/rawseq_schema_contracts.")
    thresholds = Thresholds(
        correlation_threshold=float(os.getenv("RAWSEQ_FEATURE_DIAG_CORRELATION_THRESHOLD", "0.995")),
        validation_blocks=int(float(os.getenv("RAWSEQ_FEATURE_DIAG_VALIDATION_BLOCKS", "4"))),
        min_train_finite_fraction=float(os.getenv("RAWSEQ_FEATURE_DIAG_MIN_TRAIN_FINITE_FRACTION", "0.99")),
        min_validation_finite_fraction=float(os.getenv("RAWSEQ_FEATURE_DIAG_MIN_VALIDATION_FINITE_FRACTION", "0.99")),
        near_constant_std=float(os.getenv("RAWSEQ_FEATURE_DIAG_NEAR_CONSTANT_STD", "1e-10")),
    )
    max_rows_text = os.getenv("RAWSEQ_FEATURE_DIAG_MAX_ROWS", "").strip()
    max_rows = int(float(max_rows_text)) if max_rows_text else None

    schemas = {
        "feature_schema": CONFIG_DIR / "rawseq_feature_schema_v1.json",
        "label_schema": CONFIG_DIR / "rawseq_label_schema_v1.json",
        "feature_groups": CONFIG_DIR / "rawseq_feature_groups_v1.json",
        "tensor_contracts": CONFIG_DIR / "rawseq_tensor_contracts_v1.json",
    }
    schema_hashes = {name: file_sha256(path) for name, path in schemas.items()}
    feature_manifest_json = json.loads((indicator_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    family_manifest = pd.read_csv(indicator_dir / "feature_family_manifest.csv")
    sequence_manifest = pd.read_csv(indicator_dir / "sequence_dataset_manifest.csv")
    materialized = pd.read_csv(schema_dir / "materialized_feature_columns.csv")
    lineage = pd.read_csv(schema_dir / "rawseq_column_lineage.csv")
    target_schema = pd.read_csv(schema_dir / "materialized_target_columns.csv")
    table_path = indicator_dir / "multi_horizon_training_table.csv"
    table = pd.read_csv(table_path, nrows=max_rows)
    sequence_paths = parse_sequence_dataset_env(os.getenv("RAWSEQ_FEATURE_DIAG_SEQUENCE_DATASET", ""), sequence_manifest)
    sequence_index = load_sequence_feature_indexes(sequence_paths)

    features = [str(x) for x in materialized["materialized_feature_name"].astype(str) if str(x) in table.columns]
    targets = [str(x) for x in target_schema["materialized_target_column"].astype(str) if str(x) in table.columns]
    train = split_frame(table, "train")
    validation = split_frame(table, "validation")
    if train.empty or validation.empty:
        raise SystemExit("Train and validation rows are required; holdout-only diagnostics are not allowed.")
    family_lookup = dict(zip(family_manifest["feature"].astype(str), family_manifest["feature_family"].astype(str)))
    mat_lookup = materialized.set_index("materialized_feature_name").to_dict("index")
    lineage_lookup = lineage.groupby("materialized_feature_name")["source_column"].apply(lambda s: ";".join(sorted(set(str(x) for x in s if str(x) != "nan")))).to_dict() if "materialized_feature_name" in lineage.columns else {}

    diag_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    ridge_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    formula_rows: list[dict[str, Any]] = []
    for feature in features:
        row = mat_lookup.get(feature, {})
        x_train = to_num(train[feature])
        x_val = to_num(validation[feature])
        train_std = float(x_train.std(ddof=0)) if len(x_train.dropna()) else math.nan
        train_q01 = float(x_train.quantile(0.01)) if len(x_train.dropna()) else math.nan
        train_q99 = float(x_train.quantile(0.99)) if len(x_train.dropna()) else math.nan
        val_outside_minmax = float(((x_val < float(x_train.min())) | (x_val > float(x_train.max()))).mean()) if len(x_train.dropna()) and len(x_val.dropna()) else math.nan
        val_outside_q = float(((x_val < train_q01) | (x_val > train_q99)).mean()) if np.isfinite(train_q01) and np.isfinite(train_q99) else math.nan
        shift = abs(float(x_val.mean()) - float(x_train.mean())) / max(abs(train_std), 1e-12) if np.isfinite(train_std) else math.nan
        missing_col = f"{feature}_missing"
        train_missing = float(pd.to_numeric(train[missing_col], errors="coerce").fillna(0).mean()) if missing_col in train.columns else float(x_train.isna().mean())
        val_missing = float(pd.to_numeric(validation[missing_col], errors="coerce").fillna(0).mean()) if missing_col in validation.columns else float(x_val.isna().mean())
        f_status = formula_status(row)
        block_feature_rows = validation_block_metrics(validation, feature, targets, thresholds.validation_blocks)
        score, stable_status = stability_score(block_feature_rows)
        block_rows.extend(block_feature_rows)
        best_abs_corr = 0.0
        for target in targets:
            train_corr = safe_corr(train[feature], train[target])
            val_corr = safe_corr(validation[feature], validation[target])
            if np.isfinite(train_corr):
                best_abs_corr = max(best_abs_corr, abs(train_corr))
            corr_rows.append(
                {
                    "materialized_feature_name": feature,
                    "target_column": target,
                    "horizon_buckets": target_schema.loc[target_schema["materialized_target_column"].eq(target), "horizon_buckets"].iloc[0],
                    "train_correlation": train_corr,
                    "validation_correlation": val_corr,
                    "holdout_used": False,
                }
            )
            ridge_rows.append(
                {
                    "materialized_feature_name": feature,
                    "target_column": target,
                    "horizon_buckets": target_schema.loc[target_schema["materialized_target_column"].eq(target), "horizon_buckets"].iloc[0],
                    **ridge_univariate_metrics(train[feature], train[target], validation[feature], validation[target]),
                    "holdout_used": False,
                }
            )
        base = {
            "feature_name": row.get("feature_name", feature),
            "materialized_feature_name": feature,
            "feature_index": sequence_index.get(feature, {}).get("feature_index", ""),
            "feature_family": family_lookup.get(feature, row.get("feature_family", "")),
            "source_columns": lineage_lookup.get(feature, row.get("source_columns", "")),
            "formula": resolve_formula(row),
            "formula_status": f_status,
            "implementation_path": row.get("implementation_path", ""),
            "implementation_function": row.get("implementation_function", ""),
            "window_buckets": row.get("window_buckets", ""),
            "lag_buckets": row.get("lag_buckets", ""),
            "causal_status": row.get("causal_status", ""),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "train_finite_fraction": finite_fraction(train[feature]),
            "validation_finite_fraction": finite_fraction(validation[feature]),
            "train_missing_fraction": train_missing,
            "validation_missing_fraction": val_missing,
            "train_mean": float(x_train.mean()) if len(x_train.dropna()) else math.nan,
            "train_std": train_std,
            "train_min": float(x_train.min()) if len(x_train.dropna()) else math.nan,
            "train_max": float(x_train.max()) if len(x_train.dropna()) else math.nan,
            "train_q01": train_q01,
            "train_q50": float(x_train.quantile(0.5)) if len(x_train.dropna()) else math.nan,
            "train_q99": train_q99,
            "validation_mean": float(x_val.mean()) if len(x_val.dropna()) else math.nan,
            "validation_std": float(x_val.std(ddof=0)) if len(x_val.dropna()) else math.nan,
            "validation_min": float(x_val.min()) if len(x_val.dropna()) else math.nan,
            "validation_max": float(x_val.max()) if len(x_val.dropna()) else math.nan,
            "validation_outside_train_minmax_fraction": val_outside_minmax,
            "validation_outside_train_q01_q99_fraction": val_outside_q,
            **scaler_params(train[feature]),
            "near_constant": bool(np.isfinite(train_std) and train_std <= thresholds.near_constant_std),
            "extreme_value_warning": bool(np.isfinite(val_outside_q) and val_outside_q > 0.25),
            "distribution_shift_score": shift,
            "validation_stability_score": score,
            "validation_stability_status": stable_status,
            "maximum_abs_correlation_with_another_feature": math.nan,
            "nearest_redundant_feature": "",
            "redundancy_cluster_id": "",
            "sequence_dataset_path": sequence_index.get(feature, {}).get("sequence_dataset_path", ""),
            "sequence_x_shape": sequence_index.get(feature, {}).get("sequence_x_shape", ""),
            "source_feature_tensor_index_aligned": feature not in sequence_index or sequence_index.get(feature, {}).get("feature_index", "") != "",
            "max_abs_train_target_correlation": best_abs_corr,
            "holdout_used": False,
        }
        action, reason = initial_action(base, thresholds)
        base["recommended_action_pre_redundancy"] = action
        base["recommended_action"] = action
        base["recommendation_reasons"] = reason
        diag_rows.append(base)
        formula_rows.append(
            {
                "materialized_feature_name": feature,
                "feature_name": base["feature_name"],
                "formula_status": f_status,
                "formula": base["formula"],
                "implementation_path": base["implementation_path"],
                "implementation_function": base["implementation_function"],
                "can_enter_freezeable_contract": f_status == "resolved",
            }
        )

    pairs, clusters, reps = redundancy(train, diag_rows, thresholds.correlation_threshold)
    cluster_by_member = {}
    for cluster in clusters:
        for member in str(cluster["members"]).split(";"):
            cluster_by_member[member] = cluster["redundancy_cluster_id"]
    pair_lookup: dict[str, tuple[str, float]] = {}
    for pair in pairs:
        left, right, value = pair["left_feature"], pair["right_feature"], float(pair["train_abs_correlation"])
        if value > pair_lookup.get(left, ("", -1.0))[1]:
            pair_lookup[left] = (right, value)
        if value > pair_lookup.get(right, ("", -1.0))[1]:
            pair_lookup[right] = (left, value)
    for row in diag_rows:
        feature = row["materialized_feature_name"]
        nearest, max_corr = pair_lookup.get(feature, ("", math.nan))
        row["maximum_abs_correlation_with_another_feature"] = max_corr
        row["nearest_redundant_feature"] = nearest
        row["redundancy_cluster_id"] = cluster_by_member.get(feature, "")
        if row["recommended_action"] not in {"invalid", "unresolved"} and reps.get(feature, feature) != feature:
            row["recommended_action"] = "redundant"
            row["recommendation_reasons"] = "redundant_with_representative_" + reps.get(feature, "")

    diag_frame = pd.DataFrame(diag_rows)
    evidence = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "holdout_rows_used": 0,
        "correlation_threshold": thresholds.correlation_threshold,
    }
    bundles = select_feature_bundles(diag_frame, schema_hashes, evidence)

    schema_hash = stable_json_sha256({"schemas": schema_hashes, "indicator": str(indicator_dir), "thresholds": thresholds.__dict__})
    version = "1.0.0"
    write_csv(out_dir / "feature_diagnostics.csv", with_meta(diag_rows, "rawseq_feature_diagnostics", version, schema_hash))
    write_csv(out_dir / "feature_target_correlations.csv", with_meta(corr_rows, "rawseq_feature_target_correlations", version, schema_hash))
    write_csv(out_dir / "feature_univariate_metrics.csv", with_meta(ridge_rows, "rawseq_feature_univariate_metrics", version, schema_hash))
    write_csv(out_dir / "feature_validation_block_metrics.csv", with_meta(block_rows, "rawseq_feature_validation_block_metrics", version, schema_hash))
    write_csv(out_dir / "feature_redundancy_pairs.csv", with_meta(pairs, "rawseq_feature_redundancy_pairs", version, schema_hash))
    write_csv(out_dir / "feature_redundancy_clusters.csv", with_meta(clusters, "rawseq_feature_redundancy_clusters", version, schema_hash))
    write_csv(out_dir / "formula_resolution_report.csv", with_meta(formula_rows, "rawseq_formula_resolution_report", version, schema_hash))
    for action in ["invalid", "unstable", "unresolved"]:
        write_csv(out_dir / f"{action}_features.csv", with_meta([row for row in diag_rows if row["recommended_action"] == action], f"rawseq_{action}_features", version, schema_hash))
    bundle_compare = []
    for name, payload in bundles.items():
        write_json(out_dir / f"feature_bundle_{name}.json", payload)
        bundle_compare.append(
            {
                "bundle_name": name,
                "feature_count": payload["feature_count"],
                "feature_columns_sha256": payload["feature_columns_sha256"],
                "untouched_holdout_used": False,
                "invalid_feature_count": int(diag_frame[diag_frame["materialized_feature_name"].isin(payload["ordered_feature_columns"])]["recommended_action"].eq("invalid").sum()),
                "unresolved_feature_count": int(diag_frame[diag_frame["materialized_feature_name"].isin(payload["ordered_feature_columns"])]["recommended_action"].eq("unresolved").sum()),
            }
        )
    write_csv(out_dir / "feature_bundle_comparison.csv", with_meta(bundle_compare, "rawseq_feature_bundle_comparison", version, schema_hash))
    action_counts = diag_frame["recommended_action"].value_counts().to_dict()
    family_counts = diag_frame.groupby(["feature_family", "recommended_action"]).size().reset_index(name="count").to_dict("records")
    contract = {
        **schema_meta("rawseq_feature_diagnostics_contract", version, schema_hash),
        "indicator_run_dir": str(indicator_dir),
        "schema_packet_dir": str(schema_dir),
        "sequence_datasets_read": [str(p) for p in sequence_paths],
        "training_table": str(table_path),
        "output_dir": str(out_dir),
        "thresholds": thresholds.__dict__,
        "max_rows": max_rows,
        "feature_count": len(features),
        "target_count": len(targets),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "holdout_rows_used": 0,
        "untouched_holdout_used": False,
        "recommended_action_counts": action_counts,
        "feature_family_action_counts": family_counts,
        "bundle_counts": {name: payload["feature_count"] for name, payload in bundles.items()},
        "unresolved_formula_count": int((pd.DataFrame(formula_rows)["formula_status"] == "unresolved").sum()),
        "implementation_specific_formula_count": int((pd.DataFrame(formula_rows)["formula_status"] == "implementation_specific").sum()),
        "safety": {"paper_only": True, "orders": False, "promotion": False, "champion_mutation": False, "training": False, "ensemble_search": False},
    }
    write_json(out_dir / "feature_diagnostics_contract.json", contract)
    summary = [
        "# Rawseq Feature Diagnostics",
        "",
        f"Created at: {contract['created_at']}",
        f"Indicator artifact: `{indicator_dir}`",
        f"Schema packet: `{schema_dir}`",
        f"Output: `{out_dir}`",
        "",
        "## Counts",
        f"- features: {len(features)}",
        f"- targets: {len(targets)}",
        f"- train rows used: {len(train)}",
        f"- validation rows used: {len(validation)}",
        "- untouched holdout rows used: 0",
        f"- redundancy pairs at threshold {thresholds.correlation_threshold}: {len(pairs)}",
        "",
        "## Recommended Actions",
    ]
    for action, count in sorted(action_counts.items()):
        summary.append(f"- {action}: {count}")
    summary.extend(["", "## Bundles"])
    for name, payload in bundles.items():
        summary.append(f"- {name}: {payload['feature_count']} features, sha256={payload['feature_columns_sha256']}")
    summary.extend(
        [
            "",
            "Formula warnings do not block diagnostics, but unresolved features are excluded from freezeable bundles.",
            "Safety: report_only=true; no training; no Torch sequence models; no ensemble search; no freeze; no promotion; no champion mutation; no private API; no orders.",
        ]
    )
    (out_dir / "feature_diagnostics_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("Rawseq feature diagnostics complete")
    print(f"Output: {out_dir}")
    print(f"Actions: {action_counts}")
    print(f"Bundles: {contract['bundle_counts']}")
    print("Safety: no training. No ensemble search. No orders. No promotion. No champion mutation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
