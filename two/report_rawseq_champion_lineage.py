#!/usr/bin/env python3
"""Report lineage/provenance for a frozen rawseq paper champion.

Read-only except for writing the lineage reports.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAMPION_DIR = (
    PROJECT_ROOT
    / "data"
    / "paper_champions"
    / "rawseq_fade_ma_distance_60_h2x2_v1_seed906"
)
DEFAULT_RUN_ROOT = PROJECT_ROOT / "data" / "rawseq_runs"

SYMBOL = os.getenv("RAWSEQ_LINEAGE_SYMBOL", "SOLUSDT").strip().upper()
VENUE = os.getenv("RAWSEQ_LINEAGE_VENUE", "kraken").strip().lower()
CHAMPION_DIR = Path(os.getenv("RAWSEQ_LINEAGE_CHAMPION_DIR", str(DEFAULT_CHAMPION_DIR)))
RUN_ROOT = Path(os.getenv("RAWSEQ_LINEAGE_RUN_ROOT", str(DEFAULT_RUN_ROOT)))
if not CHAMPION_DIR.is_absolute():
    CHAMPION_DIR = PROJECT_ROOT / CHAMPION_DIR
if not RUN_ROOT.is_absolute():
    RUN_ROOT = PROJECT_ROOT / RUN_ROOT

DEFAULT_TEXT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "realtime"
    / VENUE
    / f"{SYMBOL}_rawseq_champion_lineage.txt"
)
OUTPUT_PATH_ENV = os.getenv("RAWSEQ_LINEAGE_OUTPUT_PATH", "").strip()
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

MODEL_PATH = CHAMPION_DIR / "model.json"
CHAMPION_SPEC_PATH = CHAMPION_DIR / "champion_spec.txt"
SOURCE_RUN_META_PATH = CHAMPION_DIR / "source_run_meta.txt"


@dataclass
class Check:
    status: str
    name: str
    expected: str = ""
    actual: str = ""
    detail: str = ""


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def abs_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def parse_key_value_text(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    rows: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        rows[key.strip()] = value.strip()
    return rows


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def normalize_hidden(value: Any) -> str:
    text = safe_str(value).replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    return ",".join(part.strip() for part in text.split(",") if part.strip())


def normalize(field: str, value: Any) -> str:
    text = normalize_hidden(value) if field == "hidden" else safe_str(value)
    if not text:
        return ""
    if field in {"symbol"}:
        return text.upper()
    if field in {"venue", "input_feature"}:
        return text.lower()
    if field == "source_path":
        return Path(text.replace("\\", "/")).name.lower()
    if field in {"bucket_seconds", "seq_len", "ma_window", "seed", "population", "generations", "epochs"}:
        try:
            return str(int(float(text)))
        except Exception:
            return text
    if field == "hidden":
        return text.replace(" ", "")
    return text


def add_check(
    checks: list[Check],
    status: str,
    name: str,
    expected: Any = "",
    actual: Any = "",
    detail: str = "",
) -> None:
    checks.append(Check(status, name, safe_str(expected), safe_str(actual), detail))


def check_exists(checks: list[Check], name: str, path: Path, missing_status: str = "FAIL") -> None:
    add_check(
        checks,
        "PASS" if path.exists() else missing_status,
        name,
        "exists",
        abs_text(path) if path.exists() else "missing",
    )


def check_agreement(
    checks: list[Check],
    field: str,
    values_by_source: dict[str, Any],
    missing_status: str = "WARN",
) -> None:
    normalized = {
        source: normalize(field, value)
        for source, value in values_by_source.items()
        if normalize(field, value)
    }
    if len(normalized) < 2:
        add_check(
            checks,
            missing_status,
            f"{field} agreement",
            "at least two declarations",
            "; ".join(f"{k}={v}" for k, v in normalized.items()) or "none",
        )
        return
    unique = sorted(set(normalized.values()))
    if len(unique) == 1:
        add_check(checks, "PASS", f"{field} agreement", unique[0], "; ".join(f"{k}={v}" for k, v in normalized.items()))
        return
    grouped = []
    for value in unique:
        grouped.append(f"{value}: {', '.join(k for k, v in normalized.items() if v == value)}")
    add_check(checks, "FAIL", f"{field} mismatch", "one shared value", "; ".join(grouped))


def count_csv_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return max(0, sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b"")) - 1)


def csv_timestamp_range(path: Path) -> tuple[int, str, str]:
    if not path.exists():
        return 0, "", ""
    try:
        row_count = 0
        min_ts: float | None = None
        max_ts: float | None = None
        for chunk in pd.read_csv(path, usecols=["timestamp"], chunksize=250_000, low_memory=False):
            values = pd.to_numeric(chunk["timestamp"], errors="coerce").dropna()
            row_count += len(chunk)
            if values.empty:
                continue
            current_min = float(values.min())
            current_max = float(values.max())
            min_ts = current_min if min_ts is None else min(min_ts, current_min)
            max_ts = current_max if max_ts is None else max(max_ts, current_max)
        return row_count, "" if min_ts is None else str(int(min_ts)), "" if max_ts is None else str(int(max_ts))
    except Exception:
        return count_csv_data_rows(path), "", ""


def git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as exc:
        return f"git unavailable: {exc}"
    return result.stdout.strip()


def infer_source_run(champion_spec: dict[str, str], source_meta: dict[str, str]) -> str:
    return safe_str(source_meta.get("tag") or champion_spec.get("source_run"))


def build_report() -> tuple[dict[str, str], list[Check]]:
    checks: list[Check] = []
    champion_spec = parse_key_value_text(CHAMPION_SPEC_PATH)
    source_meta = parse_key_value_text(SOURCE_RUN_META_PATH)
    model = load_json(MODEL_PATH)
    model_arch = model.get("architecture") if isinstance(model.get("architecture"), dict) else {}
    model_population = model.get("population_settings") if isinstance(model.get("population_settings"), dict) else {}

    check_exists(checks, "champion directory exists", CHAMPION_DIR)
    check_exists(checks, "model.json exists", MODEL_PATH)
    check_exists(checks, "champion_spec.txt exists", CHAMPION_SPEC_PATH)
    check_exists(checks, "source_run_meta.txt exists", SOURCE_RUN_META_PATH)

    source_run = infer_source_run(champion_spec, source_meta)
    source_run_dir = RUN_ROOT / source_run if source_run else None
    if source_run:
        check_exists(checks, "source run directory exists", source_run_dir)
    else:
        add_check(checks, "FAIL", "source run declared", "source_run/tag", "missing")

    source_meta_path = source_run_dir / "meta.txt" if source_run_dir else None
    source_annotated = source_run_dir / "annotated.csv" if source_run_dir else None
    source_rows = source_run_dir / "rows.csv" if source_run_dir else None
    source_evaluation = source_run_dir / "evaluation.csv" if source_run_dir else None
    source_run_log = source_run_dir / "run.log" if source_run_dir else None
    run_meta = parse_key_value_text(source_meta_path) if source_meta_path else {}

    for name, path in [
        ("source run meta.txt exists", source_meta_path),
        ("source annotated.csv exists", source_annotated),
        ("source rows.csv exists", source_rows),
        ("source evaluation.csv exists", source_evaluation),
        ("source run.log exists", source_run_log),
    ]:
        if path is not None:
            check_exists(checks, name, path)

    for field in [
        "venue",
        "source_path",
        "bucket_seconds",
        "seq_len",
        "input_feature",
        "ma_window",
        "hidden",
        "seed",
        "population",
        "generations",
        "epochs",
    ]:
        model_value: Any = ""
        if field == "venue":
            model_value = model.get("primary_venue")
        elif field == "hidden":
            if model_arch.get("hidden_1") is not None and model_arch.get("hidden_2") is not None:
                model_value = f"{model_arch.get('hidden_1')},{model_arch.get('hidden_2')}"
        elif field == "seed":
            model_value = model_population.get("seed")
        else:
            model_value = model.get(field)

        check_agreement(
            checks,
            field,
            {
                "champion_spec": champion_spec.get(field)
                or champion_spec.get("primary_venue" if field == "venue" else "")
                or champion_spec.get("source" if field == "source_path" else ""),
                "source_run_meta": source_meta.get(field)
                or source_meta.get("primary_venue" if field == "venue" else ""),
                "source_run_meta_txt": run_meta.get(field)
                or run_meta.get("primary_venue" if field == "venue" else ""),
                "model_json": model_value,
            },
        )

    annotated_rows, annotated_min, annotated_max = csv_timestamp_range(source_annotated) if source_annotated else (0, "", "")
    rawseq_rows, rows_min, rows_max = csv_timestamp_range(source_rows) if source_rows else (0, "", "")

    git_head = git_output(["rev-parse", "--short", "HEAD"])
    git_status = git_output(["status", "--short", "--", abs_text(CHAMPION_DIR), abs_text(source_run_dir) if source_run_dir else ""])

    fields = {
        "status": "FAIL" if any(check.status == "FAIL" for check in checks) else "PASS",
        "champion_dir": abs_text(CHAMPION_DIR),
        "champion_folder": CHAMPION_DIR.name,
        "champion_spec_path": abs_text(CHAMPION_SPEC_PATH),
        "source_run_meta_path": abs_text(SOURCE_RUN_META_PATH),
        "model_path": abs_text(MODEL_PATH),
        "source_run": source_run,
        "source_run_dir": abs_text(source_run_dir),
        "source_annotated_path": abs_text(source_annotated),
        "source_rows_path": abs_text(source_rows),
        "source_evaluation_path": abs_text(source_evaluation),
        "source_run_log_path": abs_text(source_run_log),
        "source_annotated_rows": str(annotated_rows),
        "source_annotated_timestamp_min": annotated_min,
        "source_annotated_timestamp_max": annotated_max,
        "source_rows_rows": str(rawseq_rows),
        "source_rows_timestamp_min": rows_min,
        "source_rows_timestamp_max": rows_max,
        "training_config": "; ".join(
            f"{key}={source_meta.get(key) or run_meta.get(key) or champion_spec.get(key, '')}"
            for key in [
                "primary_venue",
                "source_path",
                "bucket_seconds",
                "seq_len",
                "input_feature",
                "ma_window",
                "hidden",
                "seed",
                "population",
                "generations",
                "epochs",
                "frozen_policy",
            ]
        ),
        "git_head": git_head,
        "git_status_for_paths": git_status or "clean for inspected paths",
        "checks_passed": str(sum(1 for check in checks if check.status == "PASS")),
        "warnings": str(sum(1 for check in checks if check.status == "WARN")),
        "failures": str(sum(1 for check in checks if check.status == "FAIL")),
    }
    return fields, checks


def render_text(fields: dict[str, str], checks: list[Check]) -> str:
    lines = [
        "Rawseq Champion Lineage Report",
        "",
        f"Status: {fields['status']}",
        f"Checks passed: {fields['checks_passed']}",
        f"Warnings: {fields['warnings']}",
        f"Failures: {fields['failures']}",
        "",
        "Lineage",
        f"  champion_dir: {fields['champion_dir']}",
        f"  source_run: {fields['source_run']}",
        f"  source_run_dir: {fields['source_run_dir']}",
        f"  source_annotated_rows: {fields['source_annotated_rows']}",
        f"  source_annotated_timestamp_min/max: "
        f"{fields['source_annotated_timestamp_min']} / {fields['source_annotated_timestamp_max']}",
        f"  source_rows_rows: {fields['source_rows_rows']}",
        f"  source_rows_timestamp_min/max: {fields['source_rows_timestamp_min']} / {fields['source_rows_timestamp_max']}",
        f"  git_head: {fields['git_head']}",
        f"  git_status_for_paths: {fields['git_status_for_paths']}",
        "",
        "Training Config",
        f"  {fields['training_config']}",
        "",
        "Checks",
        "  status name                                     expected                       actual",
        "  ------ ---------------------------------------- ------------------------------ ------------------------------",
    ]
    for check in checks:
        lines.append(
            "  "
            + " ".join(
                [
                    check.status[:6].ljust(6),
                    check.name[:40].ljust(40),
                    check.expected[:30].ljust(30),
                    check.actual[:60],
                ]
            )
        )
    lines.extend(
        [
            "",
            f"Text report: {abs_text(TEXT_OUTPUT_PATH)}",
            f"CSV report: {abs_text(CSV_OUTPUT_PATH)}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(fields: dict[str, str], checks: list[Check]) -> None:
    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_type", "name", "status", "expected", "actual", "detail"],
        )
        writer.writeheader()
        for key in sorted(fields):
            writer.writerow(
                {
                    "record_type": "field",
                    "name": key,
                    "status": "",
                    "expected": "",
                    "actual": fields[key],
                    "detail": "",
                }
            )
        for check in checks:
            writer.writerow(
                {
                    "record_type": "check",
                    "name": check.name,
                    "status": check.status,
                    "expected": check.expected,
                    "actual": check.actual,
                    "detail": check.detail,
                }
            )


def main() -> None:
    fields, checks = build_report()
    text = render_text(fields, checks)
    TEXT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEXT_OUTPUT_PATH.write_text(text, encoding="utf-8")
    write_csv(fields, checks)
    print(text)


if __name__ == "__main__":
    main()
