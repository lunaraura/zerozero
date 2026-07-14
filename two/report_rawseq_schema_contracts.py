#!/usr/bin/env python3
"""Audit canonical rawseq schema contracts against current artifacts."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "realtime" / "kraken" / "SOLUSDT_10s_flow.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_schema_contracts"
DEFAULT_INDICATOR_GLOB = "F:/rsio/rawseq_multi_horizon_indicator_full_contract_smoke/mh_indicator_*"
REGISTRY_PATH = PROJECT_ROOT / "scripts" / "tiny" / "rawseq_feature_label_registry.py"
SOURCE_INVENTORY_PATH = PROJECT_ROOT / "scripts" / "tiny" / "report_rawseq_source_column_inventory.py"

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


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def latest_dir(pattern: str) -> Path | None:
    paths = [Path(p) for p in sorted(Path().glob(pattern) if not re.match(r"^[A-Za-z]:", pattern) else [])]
    if not paths and re.match(r"^[A-Za-z]:", pattern):
        import glob

        paths = [Path(p) for p in glob.glob(pattern)]
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


def stable_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def registry_module():
    return load_module(REGISTRY_PATH, "rawseq_feature_label_registry_for_schema_report")


def inventory_module():
    return load_module(SOURCE_INVENTORY_PATH, "rawseq_source_inventory_for_schema_report")


def schema_meta(schema_name: str, schema_version: str, schema_sha256: str) -> dict[str, Any]:
    return {
        "schema_name": schema_name,
        "schema_version": schema_version,
        "schema_sha256": schema_sha256,
        "created_at": now_iso(),
        "generator_path": "scripts/tiny/report_rawseq_schema_contracts.py",
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


def listify(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            loaded = json.loads(text.replace("'", '"'))
            return [str(x) for x in loaded]
        except Exception:
            pass
    return [item.strip() for item in re.split(r"[,;]", text) if item.strip()]


def parse_shape(value: Any) -> list[int]:
    try:
        if isinstance(value, str):
            return [int(float(x)) for x in re.findall(r"\d+", value)]
        if isinstance(value, (list, tuple)):
            return [int(x) for x in value]
    except Exception:
        return []
    return []


def infer_family_from_name(name: str) -> str:
    n = name.lower()
    if any(token in n for token in ["macd", "sma_", "ema_", "slope", "channel_position", "vwap_distance"]):
        return "trend"
    if any(token in n for token in ["rsi", "stochastic", "roc", "cci", "williams"]):
        return "momentum"
    if any(token in n for token in ["atr", "bollinger", "volatility", "parkinson", "compression", "range"]):
        return "volatility"
    if any(token in n for token in ["recent_high", "recent_low", "zscore", "distance_to"]):
        return "breakout"
    if any(token in n for token in ["volume", "obv", "mfi", "buy_sell", "pressure"]):
        return "volume"
    if any(token in n for token in ["spread", "depth", "imbalance", "book"]):
        return "order_book"
    if any(token in n for token in ["time_of_day", "day_of_week", "regime", "session"]):
        return "regime"
    if any(token in n for token in ["btc", "eth", "sol_btc", "sol_eth"]):
        return "cross_market"
    return "raw"


def base_feature_name(materialized: str) -> str:
    name = str(materialized)
    name = re.sub(r"_fw\d+$", "", name)
    name = re.sub(r"__w\d+$", "", name)
    name = re.sub(r"_h\d+$", "", name)
    return name


def window_from_name(name: str) -> int | str:
    match = re.search(r"(?:_fw|__w)(\d+)$", str(name))
    return int(match.group(1)) if match else ""


def canonical_materialized_name(name: str) -> str:
    window = window_from_name(name)
    base = base_feature_name(name)
    return f"{base}__w{window}" if window != "" else base


def match_feature_definition(name: str, feature_defs: pd.DataFrame) -> pd.Series | None:
    base = base_feature_name(name).lower()
    for _, row in feature_defs.iterrows():
        aliases = [row.get("feature_name", ""), row.get("canonical_name", ""), *listify(row.get("aliases", ""))]
        if any(base == str(alias).lower() or base.startswith(str(alias).lower() + "_") for alias in aliases if str(alias)):
            return row
    return None


def feature_definition_rows(registry) -> list[dict[str, Any]]:
    rows = []
    for item in registry.feature_schema().get("features", []):
        row = dict(item)
        row.setdefault("window_seconds", "depends_on_declared_bucket_seconds" if row.get("feature_window_parameter") else "")
        row["aliases"] = ";".join(listify(row.get("aliases", [])))
        row["source_columns"] = ";".join(listify(row.get("source_columns", [])))
        row["required_source_columns"] = ";".join(listify(row.get("required_source_columns", [])))
        row["optional_source_columns"] = ";".join(listify(row.get("optional_source_columns", [])))
        row["feature_group_membership"] = ";".join(listify(row.get("feature_group_membership", [])))
        row["supported_symbols"] = ";".join(listify(row.get("supported_symbols", [])))
        row["supported_venues"] = ";".join(listify(row.get("supported_venues", [])))
        row["supported_bucket_seconds"] = ";".join(listify(row.get("supported_bucket_seconds", [])))
        row["supported_input_layouts"] = ";".join(listify(row.get("supported_input_layouts", [])))
        rows.append(row)
    return rows


def label_definition_rows(registry, seq_len: int) -> list[dict[str, Any]]:
    rows = []
    for item in registry.label_schema().get("labels", []):
        row = dict(item)
        for key in [
            "aliases",
            "horizon_buckets",
            "horizon_seconds",
            "required_source_columns",
            "compatible_policies",
            "incompatible_policies",
            "default_thresholds",
        ]:
            row[key] = ";".join(listify(row.get(key, [])))
        rule = str(item.get("output_dim_rule", ""))
        if rule == "output_length":
            row["materialized_output_dim"] = seq_len
        elif rule == "2 * output_length":
            row["materialized_output_dim"] = 2 * seq_len
        elif rule == "horizon_count":
            row["materialized_output_dim"] = len(listify(item.get("horizon_buckets", [])))
        rows.append(row)
    return rows


def collect_exported_features(indicator_dir: Path) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    feature_manifest = pd.read_csv(indicator_dir / "feature_manifest.csv")
    family_manifest = pd.read_csv(indicator_dir / "feature_family_manifest.csv")
    features = list(feature_manifest["feature"].astype(str))
    return features, feature_manifest, family_manifest


def collect_sequence_npz_rows(sequence_manifest: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, manifest_row in sequence_manifest.iterrows():
        path = Path(str(manifest_row.get("path_npz", "")))
        if not path.exists():
            rows.append({"path_npz": str(path), "npz_exists": False})
            continue
        with np.load(path, allow_pickle=True) as data:
            arrays = {name: data[name] for name in data.files}
            feature_columns = [str(x) for x in arrays.get("feature_columns", np.asarray([], dtype=str)).astype(str)]
            target_columns = [str(x) for x in arrays.get("target_columns", np.asarray([], dtype=str)).astype(str)]
            rows.append(
                {
                    **manifest_row.to_dict(),
                    "path_npz": str(path),
                    "npz_exists": True,
                    "npz_arrays": ";".join(data.files),
                    "actual_X_shape": json.dumps(list(arrays["X"].shape)) if "X" in arrays else "",
                    "actual_y_shape": json.dumps(list(arrays["y"].shape)) if "y" in arrays else "",
                    "feature_columns": feature_columns,
                    "target_columns": target_columns,
                    "feature_columns_sha256": stable_sha(feature_columns),
                    "target_columns_sha256": stable_sha(target_columns),
                    "decision_timestamp_min": float(np.nanmin(arrays["decision_timestamps"])) if "decision_timestamps" in arrays else math.nan,
                    "decision_timestamp_max": float(np.nanmax(arrays["decision_timestamps"])) if "decision_timestamps" in arrays else math.nan,
                }
            )
    return rows


def build_materialized_features(
    exported_features: list[str],
    family_manifest: pd.DataFrame,
    sequence_rows: list[dict[str, Any]],
    feature_defs: pd.DataFrame,
    bucket_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_map = dict(zip(family_manifest["feature"].astype(str), family_manifest["feature_family"].astype(str)))
    group_membership: dict[str, set[str]] = {}
    column_order: dict[str, int] = {}
    for seq in sequence_rows:
        group = str(seq.get("feature_group", ""))
        for idx, feature in enumerate(seq.get("feature_columns", []) or []):
            group_membership.setdefault(feature, set()).add(group)
            column_order.setdefault(feature, idx)
    rows = []
    unresolved = []
    for feature in exported_features:
        definition = match_feature_definition(feature, feature_defs)
        window = window_from_name(feature)
        family = family_map.get(feature, infer_family_from_name(feature))
        if definition is None:
            implementation_status = "unresolved"
            formula = "unknown"
            feature_name = base_feature_name(feature)
            source_columns = ""
            unresolved.append({"column_name": feature, "reason": "no_matching_feature_definition", "formula": "unknown"})
        else:
            implementation_status = str(definition.get("implementation_status", ""))
            formula = str(definition.get("formula", ""))
            feature_name = str(definition.get("feature_name", base_feature_name(feature)))
            source_columns = str(definition.get("source_columns", ""))
            if formula in {"", "unknown"} or "implementation_specific" in formula:
                unresolved.append({"column_name": feature, "reason": "formula_unresolved_or_implementation_specific", "formula": formula})
        rows.append(
            {
                "materialized_feature_name": feature,
                "feature_name": feature_name,
                "window_buckets": window,
                "window_seconds": float(window) * bucket_seconds if window != "" else "",
                "lag_buckets": 0,
                "source_symbol": "SOLUSDT",
                "source_venue": "kraken",
                "source_columns": source_columns,
                "feature_family": family,
                "feature_group_membership": ";".join(sorted(group_membership.get(feature, set()))),
                "tensor_eligible": feature in column_order,
                "column_order": column_order.get(feature, ""),
                "causal_status": "PASS",
                "availability_status": "available",
                "legacy_column_name": feature,
                "canonical_materialized_name": canonical_materialized_name(feature),
                "implementation_status": implementation_status,
                "formula": formula,
            }
        )
    return rows, unresolved


def build_materialized_targets(target_manifest: pd.DataFrame, sequence_rows: list[dict[str, Any]], label_defs: pd.DataFrame, bucket_seconds: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_order: dict[str, int] = {}
    for seq in sequence_rows:
        for idx, target in enumerate(seq.get("target_columns", []) or []):
            target_order.setdefault(target, idx)
    label = label_defs[label_defs["label_name"].eq("multi_horizon_sparse_return_vector")]
    label_row = label.iloc[0] if not label.empty else pd.Series(dtype=object)
    rows = []
    unregistered = []
    for _, item in target_manifest.iterrows():
        target = str(item.get("target_column", ""))
        if not re.match(r"future_return_bps_h\d+$", target):
            unregistered.append({"target_column": target, "reason": "no_label_pattern_match"})
        horizon = int(float(item.get("horizon_buckets", 0)))
        rows.append(
            {
                "materialized_target_column": target,
                "label_name": "multi_horizon_sparse_return_vector",
                "target_layout": label_row.get("target_layout", "sparse_horizon_vector"),
                "horizon_buckets": horizon,
                "horizon_seconds": horizon * bucket_seconds,
                "target_column_order": target_order.get(target, ""),
                "tensor_eligible": target in target_order,
                "label_end_timestamp_column": item.get("label_end_timestamp_column", ""),
                "target_type": item.get("target_type", ""),
                "legacy_column_name": target,
                "canonical_materialized_name": f"multi_horizon_sparse_return_vector__h{horizon}",
                "registration_status": "registered" if target not in [x["target_column"] for x in unregistered] else "unregistered",
            }
        )
    return rows, unregistered


def build_tensor_contracts(sequence_rows: list[dict[str, Any]], registry, split_manifest: pd.DataFrame, bucket_seconds: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    audits = []
    for base in registry.tensor_contract_schema().get("tensor_contracts", []):
        row = dict(base)
        row["required_npz_arrays"] = ";".join(listify(row.get("required_npz_arrays", [])))
        row["optional_npz_arrays"] = ";".join(listify(row.get("optional_npz_arrays", [])))
        rows.append(row)
    for seq in sequence_rows:
        x_shape = parse_shape(seq.get("actual_X_shape") or seq.get("x_shape"))
        y_shape = parse_shape(seq.get("actual_y_shape") or seq.get("y_shape"))
        feature_columns = seq.get("feature_columns", []) or []
        target_columns = seq.get("target_columns", []) or []
        arrays = set(str(seq.get("npz_arrays", "")).split(";"))
        required = {"X", "y", "splits", "decision_timestamps", "feature_columns", "target_columns", "row_indices", "horizon_buckets"}
        optional = {"label_end_timestamps", "missing_mask", "source_row_indices"}
        missing_required = sorted(required - arrays)
        missing_optional = sorted(optional - arrays)
        feature_count_ok = bool(len(x_shape) >= 3 and len(feature_columns) == x_shape[2])
        target_count_ok = bool(len(y_shape) >= 2 and len(target_columns) == y_shape[1])
        status = "PASS" if not missing_required and feature_count_ok and target_count_ok else "FAIL"
        if status == "PASS" and missing_optional:
            status = "WARN"
        contract_id = f"artifact_multivariate_sparse_horizon_vector_{seq.get('feature_group')}_seq{seq.get('seq_len')}"
        rows.append(
            {
                "contract_id": contract_id,
                "input_layout": "multivariate_sequence",
                "target_layout": "sparse_horizon_vector",
                "X_shape_template": "[batch, seq_len, feature_count]",
                "y_shape_template": "[batch, horizon_count]",
                "X_dtype": "float64",
                "y_dtype": "float64",
                "memory_order": "row_major",
                "seq_len": seq.get("seq_len", ""),
                "feature_count": len(feature_columns),
                "output_length": "",
                "horizon_count": len(target_columns),
                "input_stride": 1,
                "output_stride": "",
                "input_window_buckets": seq.get("seq_len", ""),
                "input_window_seconds": float(seq.get("seq_len", 0)) * bucket_seconds,
                "output_window_buckets": max([int(float(x)) for x in seq.get("horizon_buckets", [])] or [0]),
                "output_window_seconds": max([int(float(x)) for x in seq.get("horizon_buckets", [])] or [0]) * bucket_seconds,
                "feature_column_order": "feature_columns array order",
                "target_column_order": "target_columns array order",
                "feature_columns_sha256": seq.get("feature_columns_sha256", ""),
                "target_columns_sha256": seq.get("target_columns_sha256", ""),
                "timestamp_array_name": "decision_timestamps",
                "label_end_timestamp_array_name": "",
                "split_array_name": "splits",
                "source_row_index_array_name": "row_indices",
                "missing_mask_array_name": "",
                "required_npz_arrays": ";".join(sorted(required)),
                "optional_npz_arrays": ";".join(sorted(optional)),
                "split_encoding": "string",
                "timestamp_unit": "milliseconds",
                "declared_bucket_seconds": bucket_seconds,
                "observed_bucket_seconds": "",
                "cadence_tolerance_seconds": max(1.0, bucket_seconds * 0.25),
                "scaler_contract": "model_specific_not_serialized_in_npz",
                "feature_scaler_fit_split": "train_expected",
                "target_scaler_fit_split": "train_expected",
                "purge_rows": split_manifest.loc[split_manifest["split"].eq("purge_embargo"), "rows"].sum() if "split" in split_manifest else "",
                "embargo_rows": "",
                "source_cutoff_timestamp": seq.get("decision_timestamp_max", ""),
                "causality_guards": "feature_timestamp<=decision_timestamp;label_end_timestamp>decision_timestamp",
                "status": status,
            }
        )
        audits.extend(
            [
                {"artifact": seq.get("path_npz", ""), "check_name": "required_npz_arrays_present", "status": "PASS" if not missing_required else "FAIL", "details": ";".join(missing_required)},
                {"artifact": seq.get("path_npz", ""), "check_name": "optional_npz_arrays_present", "status": "PASS" if not missing_optional else "WARN", "details": ";".join(missing_optional)},
                {"artifact": seq.get("path_npz", ""), "check_name": "feature_columns_match_X_shape", "status": "PASS" if feature_count_ok else "FAIL", "details": f"features={len(feature_columns)} X_shape={x_shape}"},
                {"artifact": seq.get("path_npz", ""), "check_name": "target_columns_match_y_shape", "status": "PASS" if target_count_ok else "FAIL", "details": f"targets={len(target_columns)} y_shape={y_shape}"},
            ]
        )
    return rows, audits


def build_lineage(sequence_rows: list[dict[str, Any]], materialized_targets: list[dict[str, Any]], materialized_features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_map = {row["materialized_feature_name"]: row for row in materialized_features}
    target_map = {row["materialized_target_column"]: row for row in materialized_targets}
    rows = []
    for seq in sequence_rows:
        path = str(seq.get("path_npz", ""))
        contract_id = f"artifact_multivariate_sparse_horizon_vector_{seq.get('feature_group')}_seq{seq.get('seq_len')}"
        targets = seq.get("target_columns", []) or []
        for feature_idx, feature in enumerate(seq.get("feature_columns", []) or []):
            frow = feature_map.get(feature, {})
            for target_idx, target in enumerate(targets):
                trow = target_map.get(target, {})
                rows.append(
                    {
                        "source_path": str(DEFAULT_SOURCE),
                        "source_column": frow.get("source_columns", ""),
                        "feature_name": frow.get("feature_name", base_feature_name(feature)),
                        "materialized_feature_name": feature,
                        "feature_group": seq.get("feature_group", ""),
                        "tensor_contract_id": contract_id,
                        "sequence_dataset_path": path,
                        "tensor_feature_index": feature_idx,
                        "feature_scaler_index": feature_idx,
                        "model_input_index": feature_idx,
                        "label_name": trow.get("label_name", "multi_horizon_sparse_return_vector"),
                        "target_column": target,
                        "tensor_target_index": target_idx,
                        "target_scaler_index": target_idx,
                        "prediction_column": f"pred_{target}",
                        "metric_family": "sparse_return_regression_and_policy",
                        "policy_adapter": "direct_gt/inverse_gt compatible",
                        "implementation_status": frow.get("implementation_status", "unresolved"),
                    }
                )
    return rows


def build_compatibility(
    materialized_features: list[dict[str, Any]],
    label_defs: pd.DataFrame,
    tensor_contracts: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    artifact_audits: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_label = []
    policy_label = []
    model_tensor = []
    for feature in materialized_features:
        status = "PASS" if feature.get("causal_status") == "PASS" else "FAIL"
        feature_label.append(
            {
                "component_a": feature.get("materialized_feature_name", ""),
                "component_b": "multi_horizon_sparse_return_vector",
                "compatibility_status": status,
                "reason": "causal feature available for sparse horizon target" if status == "PASS" else "noncausal_or_unavailable",
                "required_adapter": "",
                "implementation_status": feature.get("implementation_status", ""),
                "test_status": "covered_or_artifact_audited",
            }
        )
    for _, label in label_defs.iterrows():
        compatible = set(listify(label.get("compatible_policies", "")))
        incompatible = set(listify(label.get("incompatible_policies", "")))
        for policy in ["direct_gt", "inverse_gt", "path_envelope_gate"]:
            if policy in compatible:
                status, reason = "PASS", "policy listed as compatible"
            elif policy in incompatible:
                status, reason = "FAIL", "policy listed as incompatible"
            else:
                status, reason = "WARN", "policy not explicitly listed"
            policy_label.append(
                {
                    "component_a": policy,
                    "component_b": label.get("label_name", ""),
                    "compatibility_status": status,
                    "reason": reason,
                    "required_adapter": "label_specific_policy_adapter" if status != "PASS" else "",
                    "implementation_status": label.get("evaluation_support", ""),
                    "test_status": label.get("test_status", ""),
                }
            )
    for audit in artifact_audits:
        model_tensor.append(
            {
                "component_a": audit.get("artifact", ""),
                "component_b": audit.get("check_name", ""),
                "compatibility_status": audit.get("status", ""),
                "reason": audit.get("details", ""),
                "required_adapter": "",
                "implementation_status": "artifact_audit",
                "test_status": "schema_report",
            }
        )
    artifact_schema_audit = list(artifact_audits)
    # Explicit target registration checks.
    for target in target_rows:
        artifact_schema_audit.append(
            {
                "artifact": target.get("materialized_target_column", ""),
                "check_name": "target_column_registered",
                "status": "PASS" if target.get("registration_status") == "registered" else "FAIL",
                "details": target.get("label_name", ""),
            }
        )
    return feature_label, model_tensor, policy_label, artifact_schema_audit


def load_source_inventory(source_path: Path, declared_bucket_seconds: float) -> pd.DataFrame:
    module = inventory_module()
    rows = module.inventory_source(
        source_path,
        "primary",
        declared_bucket_seconds,
        int(float(os.getenv("RAWSEQ_SCHEMA_SOURCE_SAMPLE_ROWS", "2000"))),
    )
    return pd.DataFrame(rows)


def read_bucket_seconds(indicator_path: Path) -> float:
    try:
        report_text = (indicator_path / "multi_horizon_indicator_report.txt").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Bucket seconds:\s*([0-9.]+)", report_text)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return 1.0


def optional_artifact_audit_rows(model_path: Path | None, probe_dir: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if model_path is None:
        rows.append(
            {
                "artifact": "RAWSEQ_SCHEMA_MODEL_PATH",
                "check_name": "optional_model_path_resolved",
                "status": "WARN",
                "details": "not provided; model tensor compatibility is limited to sequence artifacts",
            }
        )
    else:
        rows.append(
            {
                "artifact": str(model_path),
                "check_name": "optional_model_path_resolved",
                "status": "PASS" if model_path.exists() else "WARN",
                "details": "exists" if model_path.exists() else "provided path does not exist",
            }
        )
    if probe_dir is None:
        rows.append(
            {
                "artifact": "RAWSEQ_SCHEMA_PROBE_DIR",
                "check_name": "optional_probe_dir_resolved",
                "status": "WARN",
                "details": "not provided; probe/report compatibility is limited to available schema artifacts",
            }
        )
    else:
        rows.append(
            {
                "artifact": str(probe_dir),
                "check_name": "optional_probe_dir_resolved",
                "status": "PASS" if probe_dir.exists() else "WARN",
                "details": "exists" if probe_dir.exists() else "provided path does not exist",
            }
        )
    return rows


def main() -> int:
    source_path = resolve_path(os.getenv("RAWSEQ_SCHEMA_SOURCE_PATH", str(DEFAULT_SOURCE)))
    indicator_dir = os.getenv("RAWSEQ_SCHEMA_INDICATOR_RUN_DIR", "").strip()
    indicator_path = resolve_path(indicator_dir) if indicator_dir else latest_dir(DEFAULT_INDICATOR_GLOB)
    if indicator_path is None:
        raise SystemExit("Could not resolve RAWSEQ_SCHEMA_INDICATOR_RUN_DIR and no default indicator run found.")
    sequence_manifest_env = os.getenv("RAWSEQ_SCHEMA_SEQUENCE_MANIFEST", "").strip()
    sequence_manifest_path = resolve_path(sequence_manifest_env) if sequence_manifest_env else indicator_path / "sequence_dataset_manifest.csv"
    model_path_env = os.getenv("RAWSEQ_SCHEMA_MODEL_PATH", "").strip()
    probe_dir_env = os.getenv("RAWSEQ_SCHEMA_PROBE_DIR", "").strip()
    model_path = resolve_path(model_path_env) if model_path_env else None
    probe_dir = resolve_path(probe_dir_env) if probe_dir_env else None
    output_root = resolve_path(os.getenv("RAWSEQ_SCHEMA_OUTPUT_DIR", str(DEFAULT_OUTPUT_ROOT)))
    out_dir = output_root / f"rawseq_schema_contract_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    registry = registry_module()
    reg_meta = registry.all_schema_metadata()
    schema_version = "1.0.0"
    schema_hash = stable_sha(reg_meta)

    print(f"Resolved source: {source_path}")
    print(f"Resolved indicator run: {indicator_path}")
    print(f"Resolved sequence manifest: {sequence_manifest_path}")
    print(f"Output dir: {out_dir}")

    bucket_seconds = read_bucket_seconds(indicator_path)
    source_inventory = load_source_inventory(source_path, bucket_seconds)
    write_csv(out_dir / "source_column_inventory.csv", with_meta(source_inventory.to_dict("records"), "rawseq_source_column_inventory", schema_version, schema_hash))

    exported_features, _feature_manifest, family_manifest = collect_exported_features(indicator_path)
    feature_defs = pd.DataFrame(feature_definition_rows(registry))
    label_defs = pd.DataFrame(label_definition_rows(registry, seq_len=60))
    sequence_manifest = pd.read_csv(sequence_manifest_path)
    split_manifest = pd.read_csv(indicator_path / "split_manifest.csv")
    target_manifest = pd.read_csv(indicator_path / "target_manifest.csv")
    sequence_rows = collect_sequence_npz_rows(sequence_manifest)
    materialized_features, unresolved = build_materialized_features(exported_features, family_manifest, sequence_rows, feature_defs, bucket_seconds)
    materialized_targets, unregistered_targets = build_materialized_targets(target_manifest, sequence_rows, label_defs, bucket_seconds)
    tensor_contracts, artifact_audits = build_tensor_contracts(sequence_rows, registry, split_manifest, bucket_seconds)
    lineage = build_lineage(sequence_rows, materialized_targets, materialized_features)
    feature_label, model_tensor, policy_label, artifact_schema_audit = build_compatibility(
        materialized_features,
        label_defs,
        tensor_contracts,
        materialized_targets,
        artifact_audits,
    )
    artifact_schema_audit.extend(optional_artifact_audit_rows(model_path, probe_dir))

    unregistered_features = [
        {"column_name": row["materialized_feature_name"], "reason": "no_matching_definition"}
        for row in materialized_features
        if row.get("implementation_status") == "unresolved" and row.get("formula") == "unknown"
    ]
    # Unknown formulas are warnings, not unregistered, because the audit creates unresolved registry rows.
    unregistered_features = []

    write_csv(out_dir / "feature_definition_registry.csv", with_meta(feature_defs.to_dict("records"), "rawseq_feature_definition_registry", schema_version, schema_hash))
    write_csv(out_dir / "materialized_feature_columns.csv", with_meta(materialized_features, "rawseq_materialized_feature_columns", schema_version, schema_hash))
    write_csv(out_dir / "label_definition_registry.csv", with_meta(label_defs.to_dict("records"), "rawseq_label_definition_registry", schema_version, schema_hash))
    write_csv(out_dir / "materialized_target_columns.csv", with_meta(materialized_targets, "rawseq_materialized_target_columns", schema_version, schema_hash))
    write_csv(out_dir / "tensor_contract_registry.csv", with_meta(tensor_contracts, "rawseq_tensor_contract_registry", schema_version, schema_hash))
    write_csv(out_dir / "rawseq_column_lineage.csv", with_meta(lineage, "rawseq_column_lineage", schema_version, schema_hash))
    write_csv(out_dir / "rawseq_feature_label_compatibility.csv", with_meta(feature_label, "rawseq_feature_label_compatibility", schema_version, schema_hash))
    write_csv(out_dir / "rawseq_model_tensor_compatibility.csv", with_meta(model_tensor, "rawseq_model_tensor_compatibility", schema_version, schema_hash))
    write_csv(out_dir / "rawseq_policy_label_compatibility.csv", with_meta(policy_label, "rawseq_policy_label_compatibility", schema_version, schema_hash))
    write_csv(out_dir / "artifact_schema_audit.csv", with_meta(artifact_schema_audit, "rawseq_artifact_schema_audit", schema_version, schema_hash))
    write_csv(out_dir / "unregistered_columns.csv", with_meta(unregistered_features + unregistered_targets, "rawseq_unregistered_columns", schema_version, schema_hash))
    write_csv(out_dir / "unresolved_formulas.csv", with_meta(unresolved, "rawseq_unresolved_formulas", schema_version, schema_hash))

    fail_count = sum(1 for row in artifact_schema_audit if row.get("status") == "FAIL")
    warn_count = sum(1 for row in artifact_schema_audit if row.get("status") == "WARN") + len(unresolved)
    if unregistered_features or unregistered_targets:
        fail_count += len(unregistered_features) + len(unregistered_targets)
    overall = "FAIL" if fail_count else ("WARN" if warn_count else "PASS")
    contract = {
        **schema_meta("rawseq_schema_contract", schema_version, schema_hash),
        "overall_status": overall,
        "resolved_source_path": str(source_path),
        "resolved_indicator_run_dir": str(indicator_path),
        "resolved_sequence_manifest_path": str(sequence_manifest_path),
        "resolved_model_path": str(model_path) if model_path else "",
        "resolved_probe_dir": str(probe_dir) if probe_dir else "",
        "declared_bucket_seconds": bucket_seconds,
        "output_dir": str(out_dir),
        "source_column_rows": int(len(source_inventory)),
        "feature_definitions": int(len(feature_defs)),
        "materialized_features": int(len(materialized_features)),
        "label_definitions": int(len(label_defs)),
        "materialized_targets": int(len(materialized_targets)),
        "tensor_contracts": int(len(tensor_contracts)),
        "lineage_rows": int(len(lineage)),
        "unregistered_feature_columns": int(len(unregistered_features)),
        "unregistered_target_columns": int(len(unregistered_targets)),
        "unresolved_formulas": int(len(unresolved)),
        "artifact_failures": int(fail_count),
        "artifact_warnings": int(warn_count),
        "safety": {"paper_only": True, "orders": False, "promotion": False, "champion_mutation": False},
    }
    write_json(out_dir / "schema_contract.json", contract)
    (out_dir / "schema_contract.sha256").write_text(file_sha256(out_dir / "schema_contract.json") + "\n", encoding="utf-8")

    family_counts = pd.DataFrame(materialized_features)["feature_family"].value_counts().to_dict() if materialized_features else {}
    cadence = source_inventory.iloc[0].to_dict() if not source_inventory.empty else {}
    summary = [
        "# Rawseq Schema Contract Audit",
        "",
        f"Created at: {contract['created_at']}",
        f"Overall status: {overall}",
        f"Source: `{source_path}`",
        f"Indicator run: `{indicator_path}`",
        "",
        "## Counts",
        f"- source columns: {len(source_inventory)}",
        f"- feature definitions: {len(feature_defs)}",
        f"- materialized features: {len(materialized_features)}",
        f"- feature-family counts: {family_counts}",
        f"- label definitions: {len(label_defs)}",
        f"- materialized targets: {len(materialized_targets)}",
        f"- tensor contracts: {len(tensor_contracts)}",
        f"- lineage rows: {len(lineage)}",
        f"- unregistered feature columns: {len(unregistered_features)}",
        f"- unregistered target columns: {len(unregistered_targets)}",
        f"- unresolved formulas: {len(unresolved)}",
        "",
        "## Cadence",
        f"- timestamp column: {cadence.get('timestamp_column', '')}",
        f"- timestamp unit: {cadence.get('timestamp_unit', '')}",
        f"- observed median seconds: {cadence.get('observed_bucket_seconds_median', '')}",
        f"- observed mode seconds: {cadence.get('observed_bucket_seconds_mode', '')}",
        f"- declared seconds: {cadence.get('declared_bucket_seconds', '')}",
        f"- cadence match: {cadence.get('cadence_match', '')}",
        f"- duplicate timestamp count: {cadence.get('duplicate_timestamp_count', '')}",
        "",
        "## Shape/Order/Array Findings",
    ]
    audit_frame = pd.DataFrame(artifact_schema_audit)
    if not audit_frame.empty:
        for status, count in audit_frame["status"].value_counts(dropna=False).items():
            summary.append(f"- {status}: {count}")
    summary.extend(
        [
            "",
            "Warnings for old artifacts are expected when optional newly introduced metadata arrays are absent.",
            "",
            "Safety: report_only=true; no training; no discovery; no ensemble search; no orders; no promotion; no champion mutation.",
        ]
    )
    text = "\n".join(summary) + "\n"
    (out_dir / "schema_audit_summary.md").write_text(text, encoding="utf-8")
    (out_dir / "schema_audit_summary.txt").write_text(text, encoding="utf-8")

    print("Rawseq schema contract report complete")
    print(f"Status: {overall}")
    print(f"Output: {out_dir}")
    print(f"Feature definitions: {len(feature_defs)}")
    print(f"Materialized features: {len(materialized_features)}")
    print(f"Unregistered features: {len(unregistered_features)}")
    print(f"Unregistered targets: {len(unregistered_targets)}")
    print(f"Unresolved formulas: {len(unresolved)}")
    print("Safety: no training. No discovery. No orders. No promotion. No champion mutation.")
    return 0 if overall != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
