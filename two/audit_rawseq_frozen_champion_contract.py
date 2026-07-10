#!/usr/bin/env python3
"""Audit the frozen rawseq champion contract without running inference.

Read-only except for writing the audit reports.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAMPION_DIR = (
    PROJECT_ROOT
    / "data"
    / "paper_champions"
    / "rawseq_fade_ma_distance_60_h2x2_v1_seed906"
)

SYMBOL = os.getenv("RAWSEQ_AUDIT_SYMBOL", "SOLUSDT").strip().upper()
VENUE = os.getenv("RAWSEQ_AUDIT_VENUE", "kraken").strip().lower()
STRICT_EXIT = os.getenv("RAWSEQ_AUDIT_STRICT_EXIT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}

CHAMPION_DIR = Path(os.getenv("RAWSEQ_AUDIT_CHAMPION_DIR", str(DEFAULT_CHAMPION_DIR)))
if not CHAMPION_DIR.is_absolute():
    CHAMPION_DIR = PROJECT_ROOT / CHAMPION_DIR

DEFAULT_TEXT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "realtime"
    / VENUE
    / f"{SYMBOL}_rawseq_frozen_champion_contract_audit.txt"
)
OUTPUT_PATH_ENV = os.getenv("RAWSEQ_AUDIT_OUTPUT_PATH", "").strip()
if OUTPUT_PATH_ENV:
    output_path = Path(OUTPUT_PATH_ENV)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    if output_path.suffix.lower() == ".csv":
        CSV_OUTPUT_PATH = output_path
        TEXT_OUTPUT_PATH = output_path.with_suffix(".txt")
    else:
        TEXT_OUTPUT_PATH = output_path
        CSV_OUTPUT_PATH = output_path.with_suffix(".csv")
else:
    TEXT_OUTPUT_PATH = DEFAULT_TEXT_OUTPUT_PATH
    CSV_OUTPUT_PATH = DEFAULT_TEXT_OUTPUT_PATH.with_suffix(".csv")

RUN_DEFAULTS_SCRIPT = PROJECT_ROOT / "scripts" / "tiny" / "run_rawseq_frozen_shadow.py"

MODEL_PATH = CHAMPION_DIR / "model.json"
CHAMPION_SPEC_PATH = CHAMPION_DIR / "champion_spec.txt"
SOURCE_RUN_META_PATH = CHAMPION_DIR / "source_run_meta.txt"
FROZEN_SHADOW_SUMMARY_PATH = CHAMPION_DIR / "frozen_shadow_summary.csv"
FROZEN_SHADOW_RUNS_DIR = CHAMPION_DIR / "frozen_shadow_runs"

STATUS_ORDER = {"FAIL": 0, "WARN": 1, "PASS": 2}
RELEVANT_MODEL_KEYWORDS = (
    "feature",
    "input",
    "hidden",
    "architecture",
    "scaler",
    "normal",
    "seq",
    "bucket",
    "ma_",
    "source",
    "symbol",
    "venue",
    "policy",
    "threshold",
)


@dataclass
class Check:
    status: str
    name: str
    expected: str = ""
    actual: str = ""
    detail: str = ""
    sources: str = ""


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def rel_or_abs(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def read_text(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", "missing"
    try:
        return path.read_text(encoding="utf-8", errors="replace"), ""
    except Exception as exc:
        return "", f"read failed: {exc}"


def parse_key_value_text(path: Path) -> dict[str, str]:
    text, issue = read_text(path)
    if issue:
        return {}
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def read_csv_first_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                return {safe_str(k): safe_str(v) for k, v in row.items() if k is not None}
    except Exception:
        return {}
    return {}


def read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            return [safe_str(column) for column in next(reader, [])]
    except Exception:
        return []


def latest_shadow_run_dir() -> Path | None:
    if not FROZEN_SHADOW_RUNS_DIR.exists():
        return None
    run_dirs = [path for path in FROZEN_SHADOW_RUNS_DIR.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)[0]


def normalize_hidden(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(safe_str(item) for item in value if safe_str(item))
    text = safe_str(value)
    if not text:
        return ""
    text = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return ",".join(parts) if parts else text


def normalize_value(field: str, value: Any) -> str:
    text = normalize_hidden(value) if field == "hidden" else safe_str(value)
    if not text:
        return ""
    if field in {"symbol"}:
        return text.upper()
    if field in {"venue", "input_feature"}:
        return text.lower()
    if field == "source_path":
        return Path(text.replace("\\", "/")).name.lower()
    if field in {
        "bucket_seconds",
        "seq_len",
        "input_stride",
        "output_stride",
        "input_span_buckets",
        "output_span_buckets",
        "input_span_seconds",
        "output_span_seconds",
        "ma_window",
        "input_dim",
        "output_dim",
    }:
        try:
            return str(int(float(text)))
        except Exception:
            return text
    if field == "hidden":
        return text.replace(" ", "")
    if field in {"paper_only", "promotion", "orders"}:
        return text.lower()
    return text


def parse_os_getenv_defaults(script_path: Path) -> dict[str, str]:
    text, issue = read_text(script_path)
    if issue:
        return {}
    matches = re.findall(
        r'os\.getenv\(\s*"([^"]+)"\s*,\s*"([^"]*)"',
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    env_defaults = {env_name: default for env_name, default in matches}
    return {
        "symbol": env_defaults.get("SYMBOL", ""),
        "venue": env_defaults.get("PRIMARY_VENUE", ""),
        "champion_name": env_defaults.get("RAWSEQ_SHADOW_CHAMPION_NAME", ""),
        "bucket_seconds": env_defaults.get("RAWSEQ_BUCKET_SECONDS", ""),
        "seq_len": env_defaults.get("RAWSEQ_LEN", ""),
        "input_stride": env_defaults.get("RAWSEQ_INPUT_STRIDE", "1"),
        "output_stride": env_defaults.get("RAWSEQ_OUTPUT_STRIDE", "1"),
        "input_feature": env_defaults.get("RAWSEQ_INPUT_FEATURE", ""),
        "ma_window": env_defaults.get("RAWSEQ_MA_WINDOW", ""),
        "hidden": env_defaults.get("RAWSEQ_HIDDEN", ""),
        "policy": env_defaults.get("RAWSEQ_SHADOW_POLICY", ""),
        "threshold_bps": env_defaults.get("RAWSEQ_SHADOW_THRESHOLD_BPS", ""),
    }


def parse_run_log(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    patterns = {
        "symbol": r"^Symbol:\s*(.+?)\s*$",
        "venue": r"^Primary venue:\s*(.+?)\s*$",
        "bucket_seconds": r"^Bucket seconds:\s*(\d+)\s*$",
        "seq_len": r"^Sequence length:\s*(\d+)\s*$",
        "source_path": r"^Source:\s*(.+?)\s*$",
        "loaded_model_path": r"^Loaded frozen model:\s*(.+?)\s*$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            out[key] = match.group(1).strip()

    arch_match = re.search(
        r"^Architecture:\s*(\d+)\s*->\s*(\d+)\s*->\s*(\d+)\s*->\s*(\d+)\s*$",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if arch_match:
        out["input_dim"] = arch_match.group(1)
        out["hidden"] = f"{arch_match.group(2)},{arch_match.group(3)}"
        out["output_dim"] = arch_match.group(4)
        out["architecture"] = "->".join(arch_match.groups())
    stride_match = re.search(
        r"^Input stride:\s*(\d+).*?output stride:\s*(\d+)",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if stride_match:
        out["input_stride"] = stride_match.group(1)
        out["output_stride"] = stride_match.group(2)
    return out


def load_model_payload(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, "missing"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return {}, f"read failed: {exc}"
    if not isinstance(payload, dict):
        return {}, "model payload root is not an object"
    return payload, ""


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return ""


def matrix_shape(value: Any) -> tuple[int | None, int | None]:
    if not isinstance(value, list):
        return None, None
    rows = len(value)
    if rows == 0:
        return 0, 0
    first_row = value[0]
    if not isinstance(first_row, list):
        return rows, None
    return rows, len(first_row)


def vector_len(value: Any) -> int | None:
    return len(value) if isinstance(value, list) else None


def describe_model_payload(payload: dict[str, Any]) -> dict[str, str]:
    arch = payload.get("architecture") if isinstance(payload.get("architecture"), dict) else {}
    weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else {}

    w1_rows, w1_cols = matrix_shape(weights.get("W1"))
    w2_rows, w2_cols = matrix_shape(weights.get("W2"))
    w3_rows, w3_cols = matrix_shape(weights.get("W3"))
    b1_len = vector_len(weights.get("b1"))
    b2_len = vector_len(weights.get("b2"))
    b3_len = vector_len(weights.get("b3"))

    hidden_decl = ""
    hidden_1 = first_present(arch, ["hidden_1", "h1", "hidden1"])
    hidden_2 = first_present(arch, ["hidden_2", "h2", "hidden2"])
    if hidden_1 != "" and hidden_2 != "":
        hidden_decl = normalize_hidden([hidden_1, hidden_2])
    else:
        hidden_decl = normalize_hidden(first_present(payload, ["hidden", "rawseq_hidden"]))

    inferred_hidden = ""
    if w1_cols is not None and w2_cols is not None:
        inferred_hidden = normalize_hidden([w1_cols, w2_cols])

    scaler_parts = []
    for scaler_name in ["x_scaler", "y_scaler", "scaler", "normalizer", "normalization"]:
        scaler = payload.get(scaler_name)
        if isinstance(scaler, dict):
            bits = []
            for field in ["mean", "std", "scale", "center"]:
                length = vector_len(scaler.get(field))
                if length is not None:
                    bits.append(f"{field}[{length}]")
            scaler_parts.append(f"{scaler_name}: " + (", ".join(bits) if bits else "object"))

    return {
        "symbol": safe_str(first_present(payload, ["symbol", "asset"])),
        "venue": safe_str(first_present(payload, ["primary_venue", "venue"])),
        "bucket_seconds": safe_str(first_present(payload, ["bucket_seconds", "rawseq_bucket_seconds"])),
        "seq_len": safe_str(first_present(payload, ["seq_len", "rawseq_len", "sequence_length"])),
        "input_stride": safe_str(first_present(payload, ["input_stride", "rawseq_input_stride"]) or "1"),
        "output_stride": safe_str(first_present(payload, ["output_stride", "rawseq_output_stride"]) or "1"),
        "input_span_seconds": safe_str(first_present(payload, ["input_span_seconds", "rawseq_input_span_seconds"])),
        "output_span_seconds": safe_str(first_present(payload, ["output_span_seconds", "rawseq_output_span_seconds"])),
        "input_feature": safe_str(first_present(payload, ["input_feature", "rawseq_input_feature"])),
        "ma_window": safe_str(first_present(payload, ["ma_window", "rawseq_ma_window"])),
        "source_path": safe_str(first_present(payload, ["source_path"])),
        "input_dim_declared": safe_str(first_present(arch, ["input_dim"])),
        "output_dim_declared": safe_str(first_present(arch, ["output_dim"])),
        "hidden_declared": hidden_decl,
        "input_dim_inferred": safe_str(w1_rows),
        "hidden_inferred": inferred_hidden,
        "output_dim_inferred": safe_str(w3_cols),
        "w1_shape": f"{w1_rows}x{w1_cols}" if w1_rows is not None else "",
        "w2_shape": f"{w2_rows}x{w2_cols}" if w2_rows is not None else "",
        "w3_shape": f"{w3_rows}x{w3_cols}" if w3_rows is not None else "",
        "b1_len": safe_str(b1_len),
        "b2_len": safe_str(b2_len),
        "b3_len": safe_str(b3_len),
        "scaler_fields": "; ".join(scaler_parts),
    }


def flatten_relevant_keys(
    value: Any,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 3,
) -> list[tuple[str, str]]:
    if depth > max_depth:
        return []
    if isinstance(value, dict):
        rows: list[tuple[str, str]] = []
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            low_path = path.lower()
            if any(token in low_path for token in RELEVANT_MODEL_KEYWORDS):
                item = value[key]
                if isinstance(item, (dict, list)):
                    rows.append((path, type(item).__name__))
                else:
                    rows.append((path, safe_str(item)[:160]))
            rows.extend(flatten_relevant_keys(value[key], path, depth + 1, max_depth))
        return rows
    return []


def add_check(
    checks: list[Check],
    status: str,
    name: str,
    expected: Any = "",
    actual: Any = "",
    detail: str = "",
    sources: str = "",
) -> None:
    checks.append(
        Check(
            status=status,
            name=name,
            expected=safe_str(expected),
            actual=safe_str(actual),
            detail=detail,
            sources=sources,
        )
    )


def check_file_exists(checks: list[Check], name: str, path: Path, missing_status: str) -> None:
    add_check(
        checks,
        "PASS" if path.exists() else missing_status,
        name,
        expected="exists",
        actual=rel_or_abs(path) if path.exists() else "missing",
        sources=rel_or_abs(path),
    )


def compare_declared_values(
    checks: list[Check],
    field: str,
    values_by_source: dict[str, Any],
    missing_status: str = "WARN",
) -> None:
    normalized = {
        source: normalize_value(field, value)
        for source, value in values_by_source.items()
        if normalize_value(field, value)
    }
    sources = "; ".join(f"{source}={value}" for source, value in normalized.items())
    if len(normalized) < 2:
        add_check(
            checks,
            missing_status,
            f"{field} agreement",
            expected="at least two declarations",
            actual=sources or "none",
            detail="Not enough declarations to compare.",
            sources=sources,
        )
        return

    unique_values = sorted(set(normalized.values()))
    if len(unique_values) == 1:
        add_check(
            checks,
            "PASS",
            f"{field} agreement",
            expected=unique_values[0],
            actual=sources,
            sources=sources,
        )
        return

    grouped = []
    for value in unique_values:
        source_names = [source for source, current in normalized.items() if current == value]
        grouped.append(f"{value}: {', '.join(source_names)}")
    add_check(
        checks,
        "FAIL",
        f"{field} mismatch",
        expected="one shared value",
        actual="; ".join(grouped),
        detail=f"{field} mismatch across champion contract sources.",
        sources=sources,
    )


def check_equal(
    checks: list[Check],
    name: str,
    expected: Any,
    actual: Any,
    field: str,
    missing_status: str = "WARN",
    detail: str = "",
) -> None:
    expected_norm = normalize_value(field, expected)
    actual_norm = normalize_value(field, actual)
    if not expected_norm or not actual_norm:
        add_check(
            checks,
            missing_status,
            name,
            expected=expected_norm or "available",
            actual=actual_norm or "missing",
            detail=detail or "Could not compare because one side is missing.",
        )
    elif expected_norm == actual_norm:
        add_check(checks, "PASS", name, expected=expected_norm, actual=actual_norm, detail=detail)
    else:
        add_check(checks, "FAIL", name, expected=expected_norm, actual=actual_norm, detail=detail)


def build_report() -> tuple[dict[str, str], list[Check], list[tuple[str, str]], list[Path]]:
    checks: list[Check] = []
    paths_inspected: list[Path] = [
        RUN_DEFAULTS_SCRIPT,
        CHAMPION_DIR,
        MODEL_PATH,
        CHAMPION_SPEC_PATH,
        SOURCE_RUN_META_PATH,
        FROZEN_SHADOW_SUMMARY_PATH,
    ]

    latest_run = latest_shadow_run_dir()
    latest_result_path = latest_run / "result.csv" if latest_run else None
    latest_run_log_path = latest_run / "run.log" if latest_run else None
    latest_annotated_path = latest_run / "annotated.csv" if latest_run else None
    latest_rows_path = latest_run / "rows.csv" if latest_run else None
    for maybe_path in [
        latest_run,
        latest_result_path,
        latest_run_log_path,
        latest_annotated_path,
        latest_rows_path,
    ]:
        if maybe_path is not None:
            paths_inspected.append(maybe_path)

    run_defaults = parse_os_getenv_defaults(RUN_DEFAULTS_SCRIPT)
    champion_spec = parse_key_value_text(CHAMPION_SPEC_PATH)
    source_run_meta = parse_key_value_text(SOURCE_RUN_META_PATH)
    model_payload, model_issue = load_model_payload(MODEL_PATH)
    model_meta = describe_model_payload(model_payload) if model_payload else {}
    latest_result = read_csv_first_row(latest_result_path) if latest_result_path else {}

    run_log_text = ""
    if latest_run_log_path:
        run_log_text, _ = read_text(latest_run_log_path)
    run_log_meta = parse_run_log(run_log_text)

    annotated_header = read_csv_header(latest_annotated_path) if latest_annotated_path else []
    rows_header = read_csv_header(latest_rows_path) if latest_rows_path else []

    check_file_exists(checks, "model.json exists", MODEL_PATH, "FAIL")
    check_file_exists(checks, "champion_spec.txt exists", CHAMPION_SPEC_PATH, "WARN")
    check_file_exists(checks, "source_run_meta.txt exists", SOURCE_RUN_META_PATH, "WARN")
    add_check(
        checks,
        "PASS" if latest_run else "FAIL",
        "latest frozen shadow run exists",
        expected="exists",
        actual=rel_or_abs(latest_run) if latest_run else "missing",
        sources=rel_or_abs(FROZEN_SHADOW_RUNS_DIR),
    )
    if latest_result_path:
        check_file_exists(checks, "latest result.csv exists", latest_result_path, "FAIL")
    if latest_run_log_path:
        check_file_exists(checks, "latest run.log exists", latest_run_log_path, "WARN")
    if model_issue:
        add_check(checks, "FAIL", "model.json parse", expected="valid JSON object", actual=model_issue)
    elif model_payload:
        add_check(checks, "PASS", "model.json parse", expected="valid JSON object", actual="ok")

    compare_declared_values(
        checks,
        "symbol",
        {
            "audit_env": SYMBOL,
            "run_defaults": run_defaults.get("symbol"),
            "model_json": model_meta.get("symbol"),
            "latest_result": latest_result.get("symbol"),
            "run_log": run_log_meta.get("symbol"),
        },
    )
    compare_declared_values(
        checks,
        "venue",
        {
            "audit_env": VENUE,
            "run_defaults": run_defaults.get("venue"),
            "champion_spec": champion_spec.get("primary_venue") or champion_spec.get("venue"),
            "source_run_meta": source_run_meta.get("primary_venue") or source_run_meta.get("venue"),
            "model_json": model_meta.get("venue"),
            "latest_result": latest_result.get("primary_venue") or latest_result.get("venue"),
            "run_log": run_log_meta.get("venue"),
        },
    )
    compare_declared_values(
        checks,
        "bucket_seconds",
        {
            "run_defaults": run_defaults.get("bucket_seconds"),
            "champion_spec": champion_spec.get("bucket_seconds"),
            "source_run_meta": source_run_meta.get("bucket_seconds"),
            "model_json": model_meta.get("bucket_seconds"),
            "latest_result": latest_result.get("bucket_seconds"),
            "run_log": run_log_meta.get("bucket_seconds"),
        },
    )
    compare_declared_values(
        checks,
        "seq_len",
        {
            "run_defaults": run_defaults.get("seq_len"),
            "champion_spec": champion_spec.get("seq_len"),
            "source_run_meta": source_run_meta.get("seq_len"),
            "model_json": model_meta.get("seq_len"),
            "latest_result": latest_result.get("seq_len"),
            "run_log": run_log_meta.get("seq_len"),
        },
    )
    compare_declared_values(
        checks,
        "input_stride",
        {
            "run_defaults": run_defaults.get("input_stride"),
            "champion_spec": champion_spec.get("input_stride") or champion_spec.get("rawseq_input_stride"),
            "source_run_meta": source_run_meta.get("input_stride") or source_run_meta.get("rawseq_input_stride"),
            "model_json": model_meta.get("input_stride"),
            "latest_result": latest_result.get("input_stride") or latest_result.get("rawseq_input_stride"),
            "run_log": run_log_meta.get("input_stride"),
        },
    )
    compare_declared_values(
        checks,
        "output_stride",
        {
            "run_defaults": run_defaults.get("output_stride"),
            "champion_spec": champion_spec.get("output_stride") or champion_spec.get("rawseq_output_stride"),
            "source_run_meta": source_run_meta.get("output_stride") or source_run_meta.get("rawseq_output_stride"),
            "model_json": model_meta.get("output_stride"),
            "latest_result": latest_result.get("output_stride") or latest_result.get("rawseq_output_stride"),
            "run_log": run_log_meta.get("output_stride"),
        },
    )
    compare_declared_values(
        checks,
        "input_feature",
        {
            "run_defaults": run_defaults.get("input_feature"),
            "champion_spec": champion_spec.get("input_feature"),
            "source_run_meta": source_run_meta.get("input_feature"),
            "model_json": model_meta.get("input_feature"),
            "latest_result": latest_result.get("input_feature"),
        },
    )
    compare_declared_values(
        checks,
        "ma_window",
        {
            "run_defaults": run_defaults.get("ma_window"),
            "champion_spec": champion_spec.get("ma_window"),
            "source_run_meta": source_run_meta.get("ma_window"),
            "model_json": model_meta.get("ma_window"),
            "latest_result": latest_result.get("ma_window"),
        },
    )
    compare_declared_values(
        checks,
        "hidden",
        {
            "run_defaults": run_defaults.get("hidden"),
            "champion_spec": champion_spec.get("hidden"),
            "source_run_meta": source_run_meta.get("hidden"),
            "model_json_declared": model_meta.get("hidden_declared"),
            "model_json_inferred": model_meta.get("hidden_inferred"),
            "latest_result": latest_result.get("hidden"),
            "run_log": run_log_meta.get("hidden"),
        },
    )
    compare_declared_values(
        checks,
        "source_path",
        {
            "champion_spec": champion_spec.get("source") or champion_spec.get("source_path"),
            "source_run_meta": source_run_meta.get("source_path") or source_run_meta.get("source"),
            "model_json": model_meta.get("source_path"),
            "latest_result": latest_result.get("source_path"),
            "run_log": run_log_meta.get("source_path"),
        },
    )

    check_equal(
        checks,
        "declared input_dim matches inferred W1 rows",
        model_meta.get("input_dim_declared"),
        model_meta.get("input_dim_inferred"),
        "input_dim",
    )
    check_equal(
        checks,
        "declared hidden matches inferred weights",
        model_meta.get("hidden_declared"),
        model_meta.get("hidden_inferred"),
        "hidden",
    )
    check_equal(
        checks,
        "declared output_dim matches inferred W3 columns",
        model_meta.get("output_dim_declared"),
        model_meta.get("output_dim_inferred"),
        "output_dim",
    )
    check_equal(
        checks,
        "inferred input dimension matches seq_len",
        model_meta.get("seq_len") or champion_spec.get("seq_len") or run_defaults.get("seq_len"),
        model_meta.get("input_dim_inferred"),
        "seq_len",
    )
    check_equal(
        checks,
        "inferred output dimension matches seq_len",
        model_meta.get("seq_len") or champion_spec.get("seq_len") or run_defaults.get("seq_len"),
        model_meta.get("output_dim_inferred"),
        "seq_len",
    )

    if model_meta.get("w1_shape") and model_meta.get("w2_shape"):
        w1_cols = model_meta.get("hidden_inferred", "").split(",")[0] if model_meta.get("hidden_inferred") else ""
        w2_rows = model_meta.get("w2_shape", "").split("x")[0]
        check_equal(checks, "W1 columns match W2 rows", w1_cols, w2_rows, "input_dim")
    if model_meta.get("w2_shape") and model_meta.get("w3_shape"):
        hidden_parts = model_meta.get("hidden_inferred", "").split(",")
        w2_cols = hidden_parts[1] if len(hidden_parts) > 1 else ""
        w3_rows = model_meta.get("w3_shape", "").split("x")[0]
        check_equal(checks, "W2 columns match W3 rows", w2_cols, w3_rows, "input_dim")

    check_equal(
        checks,
        "latest result input_feature matches model payload",
        model_meta.get("input_feature"),
        latest_result.get("input_feature"),
        "input_feature",
        detail="Latest frozen-shadow result must not claim a different feature than the loaded model.",
    )
    check_equal(
        checks,
        "latest result hidden matches model payload",
        model_meta.get("hidden_declared") or model_meta.get("hidden_inferred"),
        latest_result.get("hidden"),
        "hidden",
        detail="Latest frozen-shadow result must not claim a different hidden architecture than the loaded model.",
    )
    check_equal(
        checks,
        "latest result input_stride matches model payload",
        model_meta.get("input_stride"),
        latest_result.get("input_stride") or latest_result.get("rawseq_input_stride"),
        "input_stride",
        detail="Latest frozen-shadow result must not claim a different input stride than the loaded model.",
    )
    check_equal(
        checks,
        "latest result output_stride matches model payload",
        model_meta.get("output_stride"),
        latest_result.get("output_stride") or latest_result.get("rawseq_output_stride"),
        "output_stride",
        detail="Latest frozen-shadow result must not claim a different output stride than the loaded model.",
    )
    check_equal(
        checks,
        "run.log architecture hidden matches model payload",
        model_meta.get("hidden_declared") or model_meta.get("hidden_inferred"),
        run_log_meta.get("hidden"),
        "hidden",
        detail="Latest frozen-shadow run.log should not contradict the loaded model architecture.",
    )

    for flag_name, expected in [("paper_only", "true"), ("promotion", "false"), ("orders", "false")]:
        check_equal(
            checks,
            f"latest result {flag_name} safety flag",
            expected,
            latest_result.get(flag_name),
            flag_name,
            missing_status="FAIL",
        )

    if latest_result.get("source_path") and run_defaults.get("symbol"):
        result_source = latest_result.get("source_path", "")
        expected_fragment = f"{run_defaults.get('symbol')}_10s_flow.csv"
        add_check(
            checks,
            "WARN" if expected_fragment not in result_source else "PASS",
            "latest result source_path resembles frozen-shadow default source",
            expected=expected_fragment,
            actual=result_source,
            detail="Training source and inference source may differ; this warns on unexpected frozen-shadow source files.",
        )

    field_rows: dict[str, str] = {
        "champion_dir": rel_or_abs(CHAMPION_DIR),
        "champion_folder_name": CHAMPION_DIR.name,
        "model_path": rel_or_abs(MODEL_PATH),
        "model_exists": bool_text(MODEL_PATH.exists()),
        "champion_spec_exists": bool_text(CHAMPION_SPEC_PATH.exists()),
        "source_run_meta_exists": bool_text(SOURCE_RUN_META_PATH.exists()),
        "latest_frozen_shadow_run_exists": bool_text(latest_run is not None),
        "latest_frozen_shadow_run_dir": rel_or_abs(latest_run),
        "symbol_audit": SYMBOL,
        "venue_audit": VENUE,
        "symbol_model_payload": normalize_value("symbol", model_meta.get("symbol")),
        "venue_model_payload": normalize_value("venue", model_meta.get("venue")),
        "bucket_seconds_model_payload": normalize_value("bucket_seconds", model_meta.get("bucket_seconds")),
        "seq_len_model_payload": normalize_value("seq_len", model_meta.get("seq_len")),
        "input_stride_model_payload": normalize_value("input_stride", model_meta.get("input_stride")),
        "output_stride_model_payload": normalize_value("output_stride", model_meta.get("output_stride")),
        "input_span_seconds_model_payload": normalize_value("input_span_seconds", model_meta.get("input_span_seconds")),
        "output_span_seconds_model_payload": normalize_value("output_span_seconds", model_meta.get("output_span_seconds")),
        "input_feature_model_payload": normalize_value("input_feature", model_meta.get("input_feature")),
        "ma_window_model_payload": normalize_value("ma_window", model_meta.get("ma_window")),
        "hidden_model_payload_declared": normalize_value("hidden", model_meta.get("hidden_declared")),
        "hidden_model_payload_inferred": normalize_value("hidden", model_meta.get("hidden_inferred")),
        "model_payload_input_dim_declared": normalize_value("input_dim", model_meta.get("input_dim_declared")),
        "model_payload_input_dim_inferred": normalize_value("input_dim", model_meta.get("input_dim_inferred")),
        "model_payload_output_dim_declared": normalize_value("output_dim", model_meta.get("output_dim_declared")),
        "model_payload_output_dim_inferred": normalize_value("output_dim", model_meta.get("output_dim_inferred")),
        "model_weight_shapes": ", ".join(
            item
            for item in [
                f"W1={model_meta.get('w1_shape', '')}",
                f"W2={model_meta.get('w2_shape', '')}",
                f"W3={model_meta.get('w3_shape', '')}",
            ]
            if item.split("=", 1)[1]
        ),
        "scaler_normalization_fields": model_meta.get("scaler_fields", ""),
        "source_path_model_payload": model_meta.get("source_path", ""),
        "source_path_model_payload_basename": normalize_value("source_path", model_meta.get("source_path")),
        "source_path_latest_result": latest_result.get("source_path", ""),
        "source_path_latest_result_basename": normalize_value("source_path", latest_result.get("source_path")),
        "source_path_source_run_meta": source_run_meta.get("source_path", ""),
        "source_path_source_run_meta_basename": normalize_value("source_path", source_run_meta.get("source_path")),
        "source_path_champion_spec": champion_spec.get("source") or champion_spec.get("source_path", ""),
        "source_path_champion_spec_basename": normalize_value(
            "source_path",
            champion_spec.get("source") or champion_spec.get("source_path"),
        ),
        "source_timestamp_min": latest_result.get("source_timestamp_min", ""),
        "source_timestamp_max": latest_result.get("source_timestamp_max", ""),
        "test_timestamp_min": latest_result.get("test_timestamp_min", ""),
        "test_timestamp_max": latest_result.get("test_timestamp_max", ""),
        "latest_shadow_policy": latest_result.get("policy", ""),
        "latest_shadow_threshold": latest_result.get("threshold_bps", ""),
        "latest_shadow_paper_only": latest_result.get("paper_only", ""),
        "latest_shadow_promotion": latest_result.get("promotion", ""),
        "latest_shadow_orders": latest_result.get("orders", ""),
        "run_defaults_input_feature": normalize_value("input_feature", run_defaults.get("input_feature")),
        "run_defaults_hidden": normalize_value("hidden", run_defaults.get("hidden")),
        "run_defaults_input_stride": normalize_value("input_stride", run_defaults.get("input_stride")),
        "run_defaults_output_stride": normalize_value("output_stride", run_defaults.get("output_stride")),
        "champion_spec_input_feature": normalize_value("input_feature", champion_spec.get("input_feature")),
        "champion_spec_hidden": normalize_value("hidden", champion_spec.get("hidden")),
        "champion_spec_input_stride": normalize_value(
            "input_stride", champion_spec.get("input_stride") or champion_spec.get("rawseq_input_stride")
        ),
        "champion_spec_output_stride": normalize_value(
            "output_stride", champion_spec.get("output_stride") or champion_spec.get("rawseq_output_stride")
        ),
        "source_run_meta_input_feature": normalize_value("input_feature", source_run_meta.get("input_feature")),
        "source_run_meta_hidden": normalize_value("hidden", source_run_meta.get("hidden")),
        "source_run_meta_input_stride": normalize_value(
            "input_stride", source_run_meta.get("input_stride") or source_run_meta.get("rawseq_input_stride")
        ),
        "source_run_meta_output_stride": normalize_value(
            "output_stride", source_run_meta.get("output_stride") or source_run_meta.get("rawseq_output_stride")
        ),
        "latest_result_input_feature": normalize_value("input_feature", latest_result.get("input_feature")),
        "latest_result_hidden": normalize_value("hidden", latest_result.get("hidden")),
        "latest_result_input_stride": normalize_value(
            "input_stride", latest_result.get("input_stride") or latest_result.get("rawseq_input_stride")
        ),
        "latest_result_output_stride": normalize_value(
            "output_stride", latest_result.get("output_stride") or latest_result.get("rawseq_output_stride")
        ),
        "run_log_hidden": normalize_value("hidden", run_log_meta.get("hidden")),
        "run_log_input_stride": normalize_value("input_stride", run_log_meta.get("input_stride")),
        "run_log_output_stride": normalize_value("output_stride", run_log_meta.get("output_stride")),
        "latest_annotated_header_columns": str(len(annotated_header)),
        "latest_annotated_has_rawseq_input_feature": bool_text("rawseq_input_feature" in annotated_header),
        "latest_rows_header_columns": str(len(rows_header)),
        "latest_rows_has_rawseq_input_feature": bool_text("rawseq_input_feature" in rows_header),
    }

    parsed_relevant_paths = {
        "architecture.input_dim",
        "architecture.hidden_1",
        "architecture.hidden_2",
        "architecture.output_dim",
        "input_feature",
        "input_span_seconds",
        "input_stride",
        "ma_window",
        "output_span_seconds",
        "output_stride",
        "primary_venue",
        "seq_len",
        "source_path",
        "symbol",
        "x_scaler",
        "y_scaler",
    }
    unknown_relevant = [
        (path, value)
        for path, value in flatten_relevant_keys(model_payload)
        if path not in parsed_relevant_paths and not path.startswith("weights")
    ]

    return field_rows, checks, unknown_relevant, paths_inspected


def summarize_checks(checks: list[Check]) -> dict[str, int | str]:
    failures = sum(1 for check in checks if check.status == "FAIL")
    warnings = sum(1 for check in checks if check.status == "WARN")
    passes = sum(1 for check in checks if check.status == "PASS")
    return {
        "status": "FAIL" if failures else "PASS",
        "passes": passes,
        "warnings": warnings,
        "failures": failures,
    }


def compact_table(checks: list[Check], include_passes: bool = False) -> list[str]:
    rows = [
        check
        for check in sorted(checks, key=lambda item: (STATUS_ORDER.get(item.status, 9), item.name))
        if include_passes or check.status != "PASS"
    ]
    if not rows:
        return ["  no mismatches or warnings"]
    widths = {"status": 6, "name": 46, "expected": 30, "actual": 68}
    lines = [
        "  "
        + " ".join(
            header.ljust(widths[header])
            for header in ["status", "name", "expected", "actual"]
        ),
        "  "
        + " ".join(
            ("-" * widths[header])
            for header in ["status", "name", "expected", "actual"]
        ),
    ]
    for check in rows:
        values = {
            "status": check.status,
            "name": check.name,
            "expected": check.expected,
            "actual": check.actual,
        }
        line = "  " + " ".join(
            safe_str(values[header])[: widths[header]].ljust(widths[header])
            for header in ["status", "name", "expected", "actual"]
        )
        lines.append(line)
    return lines


def render_text_report(
    field_rows: dict[str, str],
    checks: list[Check],
    unknown_relevant: list[tuple[str, str]],
    paths_inspected: list[Path],
) -> str:
    summary = summarize_checks(checks)
    lines = [
        "Rawseq Frozen Champion Contract Audit",
        "",
        f"Status: {summary['status']}",
        f"Checks passed: {summary['passes']}",
        f"Warnings: {summary['warnings']}",
        f"Failures: {summary['failures']}",
        f"Strict exit: {bool_text(STRICT_EXIT)}",
        "",
        "Key Fields",
        f"  champion_dir: {field_rows.get('champion_dir', '')}",
        f"  champion_folder_name: {field_rows.get('champion_folder_name', '')}",
        f"  model_path: {field_rows.get('model_path', '')}",
        f"  model payload feature: {field_rows.get('input_feature_model_payload', '')}",
        f"  model payload hidden declared/inferred: "
        f"{field_rows.get('hidden_model_payload_declared', '')} / "
        f"{field_rows.get('hidden_model_payload_inferred', '')}",
        f"  model payload input/output stride: "
        f"{field_rows.get('input_stride_model_payload', '')} / "
        f"{field_rows.get('output_stride_model_payload', '')}",
        f"  run defaults feature/hidden: "
        f"{field_rows.get('run_defaults_input_feature', '')} / "
        f"{field_rows.get('run_defaults_hidden', '')}",
        f"  run defaults input/output stride: "
        f"{field_rows.get('run_defaults_input_stride', '')} / "
        f"{field_rows.get('run_defaults_output_stride', '')}",
        f"  champion spec feature/hidden: "
        f"{field_rows.get('champion_spec_input_feature', '')} / "
        f"{field_rows.get('champion_spec_hidden', '')}",
        f"  champion spec input/output stride: "
        f"{field_rows.get('champion_spec_input_stride', '')} / "
        f"{field_rows.get('champion_spec_output_stride', '')}",
        f"  source meta feature/hidden: "
        f"{field_rows.get('source_run_meta_input_feature', '')} / "
        f"{field_rows.get('source_run_meta_hidden', '')}",
        f"  source meta input/output stride: "
        f"{field_rows.get('source_run_meta_input_stride', '')} / "
        f"{field_rows.get('source_run_meta_output_stride', '')}",
        f"  latest result feature/hidden: "
        f"{field_rows.get('latest_result_input_feature', '')} / "
        f"{field_rows.get('latest_result_hidden', '')}",
        f"  latest result input/output stride: "
        f"{field_rows.get('latest_result_input_stride', '')} / "
        f"{field_rows.get('latest_result_output_stride', '')}",
        f"  latest policy/threshold: "
        f"{field_rows.get('latest_shadow_policy', '')} / "
        f"{field_rows.get('latest_shadow_threshold', '')}",
        "",
        "Mismatches And Warnings",
        *compact_table(checks),
        "",
        "Paths Inspected",
    ]
    for path in paths_inspected:
        lines.append(f"  {rel_or_abs(path)}")

    lines.extend(["", "Relevant Model Keys Not Specifically Parsed"])
    if unknown_relevant:
        for path, value in unknown_relevant[:80]:
            lines.append(f"  {path}: {value}")
        if len(unknown_relevant) > 80:
            lines.append(f"  ... {len(unknown_relevant) - 80} more")
    else:
        lines.append("  none")

    lines.extend(
        [
            "",
            f"Text report: {rel_or_abs(TEXT_OUTPUT_PATH)}",
            f"CSV report: {rel_or_abs(CSV_OUTPUT_PATH)}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv_report(field_rows: dict[str, str], checks: list[Check]) -> None:
    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_checks(checks)
    rows: list[dict[str, str]] = [
        {
            "record_type": "summary",
            "name": "status",
            "status": safe_str(summary["status"]),
            "expected": "",
            "actual": safe_str(summary["status"]),
            "detail": (
                f"passes={summary['passes']}; warnings={summary['warnings']}; "
                f"failures={summary['failures']}"
            ),
            "sources": "",
        }
    ]
    for name in sorted(field_rows):
        rows.append(
            {
                "record_type": "field",
                "name": name,
                "status": "",
                "expected": "",
                "actual": field_rows[name],
                "detail": "",
                "sources": "",
            }
        )
    for check in sorted(checks, key=lambda item: (STATUS_ORDER.get(item.status, 9), item.name)):
        rows.append(
            {
                "record_type": "check",
                "name": check.name,
                "status": check.status,
                "expected": check.expected,
                "actual": check.actual,
                "detail": check.detail,
                "sources": check.sources,
            }
        )

    with CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_type", "name", "status", "expected", "actual", "detail", "sources"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    field_rows, checks, unknown_relevant, paths_inspected = build_report()
    text_report = render_text_report(field_rows, checks, unknown_relevant, paths_inspected)

    TEXT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEXT_OUTPUT_PATH.write_text(text_report, encoding="utf-8")
    write_csv_report(field_rows, checks)

    print(text_report)
    summary = summarize_checks(checks)
    if STRICT_EXIT and int(summary["failures"]) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
