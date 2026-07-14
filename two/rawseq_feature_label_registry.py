#!/usr/bin/env python
"""Canonical rawseq feature, label, group, and tensor schema registry.

The versioned JSON files under configs/rawseq are the authoritative source.
This module keeps the earlier Python API (`input_features`, `output_labels`,
`registry_rows`) available for older scripts while emitting deterministic
versioned manifests for schema/audit tooling.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "rawseq"
OUTPUT_DIR = Path(os.getenv("RAWSEQ_REGISTRY_OUTPUT_DIR", PROJECT_ROOT / "data" / "research" / "rawseq_feature_label_registry"))
DEFAULT_SEQ_LEN = int(float(os.getenv("RAWSEQ_REGISTRY_SEQ_LEN", "60")))

FEATURE_SCHEMA_PATH = CONFIG_DIR / "rawseq_feature_schema_v1.json"
LABEL_SCHEMA_PATH = CONFIG_DIR / "rawseq_label_schema_v1.json"
FEATURE_GROUP_SCHEMA_PATH = CONFIG_DIR / "rawseq_feature_groups_v1.json"
TENSOR_CONTRACT_SCHEMA_PATH = CONFIG_DIR / "rawseq_tensor_contracts_v1.json"


@dataclass(frozen=True)
class InputFeatureDefinition:
    feature_name: str
    task_role: str
    required_source_columns: list[str]
    default_windows: list[int] = field(default_factory=list)
    default_thresholds: list[float] = field(default_factory=list)
    leakage_warning: str = ""
    compatible_labels: list[str] = field(default_factory=list)
    description: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        row = dict(self.raw)
        row.update(
            {
                "registry_type": "input_feature",
                "name": self.feature_name,
                "label_name": "",
                "feature_name": self.feature_name,
                "task_type": "",
                "task_role": self.task_role,
                "output_dim": "",
                "required_source_columns": ",".join(self.required_source_columns),
                "required_horizon_buckets": "",
                "default_windows": ",".join(str(x) for x in self.default_windows),
                "default_thresholds": ",".join(str(x) for x in self.default_thresholds),
                "compatible_policies": "",
                "compatible_labels": ",".join(self.compatible_labels),
                "leakage_warning": self.leakage_warning,
                "description": self.description,
            }
        )
        return stringify_row(row)


@dataclass(frozen=True)
class OutputLabelDefinition:
    label_name: str
    task_type: str
    output_dim: int
    required_source_columns: list[str]
    required_horizon_buckets: int
    leakage_warning: str
    compatible_policies: list[str]
    default_thresholds: list[float | str]
    description: str
    raw: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        row = dict(self.raw)
        row.update(
            {
                "registry_type": "output_label",
                "name": self.label_name,
                "label_name": self.label_name,
                "feature_name": "",
                "task_type": self.task_type,
                "task_role": "",
                "output_dim": self.output_dim,
                "required_source_columns": ",".join(self.required_source_columns),
                "required_horizon_buckets": self.required_horizon_buckets,
                "default_windows": "",
                "default_thresholds": ",".join(str(x) for x in self.default_thresholds),
                "compatible_policies": ",".join(str(x) for x in self.compatible_policies),
                "compatible_labels": "",
                "leakage_warning": self.leakage_warning,
                "description": self.description,
            }
        )
        return stringify_row(row)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


def git_status_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except Exception:
        return True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def schema_metadata(schema: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "schema_name": schema.get("schema_name", path.stem),
        "schema_version": schema.get("schema_version", ""),
        "schema_sha256": file_sha256(path),
        "created_at": now_iso(),
        "generator_path": schema.get("generator_path", "scripts/tiny/rawseq_feature_label_registry.py"),
        "git_head": git_head(),
        "git_status_dirty": git_status_dirty(),
        "paper_only": bool(schema.get("paper_only", True)),
        "orders": bool(schema.get("orders", False)),
        "promotion": bool(schema.get("promotion", False)),
        "champion_mutation": bool(schema.get("champion_mutation", False)),
    }


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def stringify_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: stringify_value(value) for key, value in row.items()}


def canonical_materialized_feature_name(feature_name: str, window_buckets: int | str | None = None) -> str:
    """Return the canonical materialized column name without renaming legacy artifacts."""
    base = str(feature_name).strip()
    if window_buckets in {None, ""}:
        return base
    return f"{base}__w{int(float(window_buckets))}"


def column_order_sha256(columns: list[str]) -> str:
    """Stable hash for feature/target column order."""
    return stable_json_sha256([str(column) for column in columns])


def validate_unique_feature_keys(features: list[dict[str, Any]]) -> None:
    """Reject duplicate feature IDs, canonical names, and aliases."""
    seen_ids: dict[str, str] = {}
    seen_names: dict[str, str] = {}
    seen_aliases: dict[str, str] = {}
    for item in features:
        feature_id = str(item.get("feature_id", "")).strip().lower()
        canonical = str(item.get("canonical_name", "")).strip().lower()
        if feature_id:
            if feature_id in seen_ids:
                raise ValueError(f"duplicate feature_id={feature_id}")
            seen_ids[feature_id] = feature_id
        if canonical:
            if canonical in seen_names:
                raise ValueError(f"duplicate canonical_name={canonical}")
            seen_names[canonical] = feature_id
        aliases = item.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        for alias in aliases:
            alias_key = str(alias).strip().lower()
            if not alias_key:
                continue
            if alias_key in seen_aliases and seen_aliases[alias_key] != feature_id:
                raise ValueError(f"duplicate alias={alias_key}")
            seen_aliases[alias_key] = feature_id


def validate_unique_label_keys(labels: list[dict[str, Any]]) -> None:
    seen_names: set[str] = set()
    for item in labels:
        name = str(item.get("label_name", item.get("label_id", ""))).strip().lower()
        if not name:
            continue
        if name in seen_names:
            raise ValueError(f"duplicate label_name={name}")
        seen_names.add(name)


def feature_schema() -> dict[str, Any]:
    schema = load_json(FEATURE_SCHEMA_PATH)
    validate_unique_feature_keys(schema.get("features", []))
    return schema


def label_schema() -> dict[str, Any]:
    schema = load_json(LABEL_SCHEMA_PATH)
    validate_unique_label_keys(schema.get("labels", []))
    return schema


def feature_group_schema() -> dict[str, Any]:
    return load_json(FEATURE_GROUP_SCHEMA_PATH)


def tensor_contract_schema() -> dict[str, Any]:
    return load_json(TENSOR_CONTRACT_SCHEMA_PATH)


def all_schema_metadata() -> dict[str, dict[str, Any]]:
    return {
        "feature": schema_metadata(feature_schema(), FEATURE_SCHEMA_PATH),
        "label": schema_metadata(label_schema(), LABEL_SCHEMA_PATH),
        "feature_group": schema_metadata(feature_group_schema(), FEATURE_GROUP_SCHEMA_PATH),
        "tensor_contract": schema_metadata(tensor_contract_schema(), TENSOR_CONTRACT_SCHEMA_PATH),
    }


def input_features() -> list[InputFeatureDefinition]:
    common_labels = [item["label_name"] for item in label_schema().get("labels", [])]
    definitions = []
    for item in feature_schema().get("features", []):
        windows = item.get("window_buckets")
        default_windows: list[int] = []
        if isinstance(windows, int):
            default_windows = [windows]
        elif item.get("feature_window_parameter") in {"ma_window", "feature_window"}:
            default_windows = [60, 150, 300]
        definitions.append(
            InputFeatureDefinition(
                feature_name=str(item.get("feature_name", item.get("feature_id", ""))),
                task_role=str(item.get("subfamily", item.get("feature_family", ""))),
                required_source_columns=list(item.get("required_source_columns", [])),
                default_windows=default_windows,
                default_thresholds=[0.0, 0.1, 0.25, 0.5],
                leakage_warning=str(item.get("leakage_warning", "")),
                compatible_labels=common_labels,
                description=str(item.get("description", "")),
                raw=item,
            )
        )
    return definitions


def _materialized_output_dim(item: dict[str, Any], seq_len: int) -> int:
    rule = str(item.get("output_dim_rule", ""))
    if rule == "output_length":
        return seq_len
    if rule == "2 * output_length":
        return 2 * seq_len
    if rule == "horizon_count":
        horizons = item.get("horizon_buckets") or []
        return len(horizons)
    try:
        return int(item.get("materialized_output_dim", 0))
    except Exception:
        return 0


def _required_horizon_buckets(item: dict[str, Any], seq_len: int) -> int:
    horizons = item.get("horizon_buckets") or []
    if horizons:
        return int(max(horizons))
    if str(item.get("required_future_rows", "")).startswith("output_length"):
        return seq_len
    try:
        return int(float(item.get("required_future_rows", seq_len)))
    except Exception:
        return seq_len


def output_labels(seq_len: int = DEFAULT_SEQ_LEN) -> list[OutputLabelDefinition]:
    definitions = []
    for item in label_schema().get("labels", []):
        definitions.append(
            OutputLabelDefinition(
                label_name=str(item.get("label_name", item.get("label_id", ""))),
                task_type=str(item.get("task_type", "")),
                output_dim=_materialized_output_dim(item, seq_len),
                required_source_columns=list(item.get("required_source_columns", [])),
                required_horizon_buckets=_required_horizon_buckets(item, seq_len),
                leakage_warning=str(item.get("leakage_warning", "")),
                compatible_policies=list(item.get("compatible_policies", [])),
                default_thresholds=list(item.get("default_thresholds", [])),
                description=str(item.get("description", "")),
                raw=item,
            )
        )
    return definitions


def registry_rows(seq_len: int = DEFAULT_SEQ_LEN) -> list[dict[str, Any]]:
    rows = [definition.row() for definition in input_features()]
    rows.extend(definition.row() for definition in output_labels(seq_len))
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return [{key: row.get(key, "") for key in fieldnames} for row in rows]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, features: list[InputFeatureDefinition], labels: list[OutputLabelDefinition]) -> None:
    lines = [
        "Rawseq canonical feature and label registry",
        "",
        "Canonical source: configs/rawseq/*.json",
        "",
        "Safety:",
        "  report_only=true",
        "  training=false",
        "  promotion=false",
        "  champion_mutation=false",
        "  orders=false",
        "",
        "Input feature definitions:",
    ]
    for item in features:
        raw = item.raw
        lines.append(
            f"  {item.feature_name}: family={raw.get('feature_family', '')} "
            f"status={raw.get('implementation_status', '')} columns={','.join(item.required_source_columns)}"
        )
        lines.append(f"    {item.description}")
    lines.extend(["", "Output label definitions:"])
    for item in labels:
        lines.append(
            f"  {item.label_name}: layout={item.raw.get('target_layout', '')} "
            f"type={item.task_type} output_dim={item.output_dim} status={item.raw.get('status', '')}"
        )
        lines.append(f"    policies={','.join(item.compatible_policies)}")
        lines.append(f"    {item.description}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    features = input_features()
    labels = output_labels(DEFAULT_SEQ_LEN)
    rows = registry_rows(DEFAULT_SEQ_LEN)
    metadata = all_schema_metadata()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "rawseq_feature_label_registry.csv"
    txt_path = OUTPUT_DIR / "rawseq_feature_label_registry.txt"
    json_path = OUTPUT_DIR / "rawseq_feature_label_registry_metadata.json"
    write_csv(csv_path, rows)
    write_text(txt_path, features, labels)
    write_json(json_path, metadata)

    print("Input features:")
    for item in features:
        print(f"  {item.feature_name}: {item.description}")
    print("")
    print("Output labels:")
    for item in labels:
        print(f"  {item.label_name}: type={item.task_type} output_dim={item.output_dim} status={item.raw.get('status', '')}")
    print("")
    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
