#!/usr/bin/env python3
"""Report health for tiny rawseq run folders.

This script is intentionally read-only with respect to models and champions. It
only reads run artifacts and writes a CSV report.
"""

from __future__ import annotations

import csv
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "rawseq_runs"
RUN_ROOT = Path(os.getenv("RAWSEQ_RUN_HEALTH_ROOT", str(DEFAULT_ROOT)))
RUN_GLOB = os.getenv("RAWSEQ_RUN_HEALTH_GLOB", "*")
OUTPUT_PATH = Path(
    os.getenv(
        "RAWSEQ_RUN_HEALTH_OUTPUT",
        str(DEFAULT_ROOT / "rawseq_run_health_report.csv"),
    )
)

if not RUN_ROOT.is_absolute():
    RUN_ROOT = PROJECT_ROOT / RUN_ROOT
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH


FIELDS = [
    "run",
    "symbol",
    "seed",
    "has_run_log",
    "has_annotated_csv",
    "has_evaluation_csv",
    "has_rows_csv",
    "has_candidate_model",
    "candidate_model_path",
    "loaded_frozen_model",
    "inference_only_complete",
    "training_generations_seen",
    "candidate_model_saved",
    "zero_trade_policy_rows",
    "suspicious_zero_rows",
    "symbol_path_mismatch",
    "status",
    "issues",
]

SYMBOL_RE = re.compile(r"(?<![A-Z0-9])([A-Z0-9]{2,20}USDT)(?![A-Z0-9])")
SEED_RE = re.compile(r"(?:^|_)seed[_-]?(\d+)(?:_|$)", re.IGNORECASE)
LOG_SYMBOL_RE = re.compile(r"^Symbol:\s*([A-Z0-9]+)\s*$", re.IGNORECASE | re.MULTILINE)
CANDIDATE_MODEL_RE = re.compile(r"^Candidate model:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
MODEL_ROWS_RE = re.compile(r"\bmodel=\d+\b[^\n]*\brows=([0-9]+|nan)\b", re.IGNORECASE)
WINNER_RE = re.compile(r"\bwinner=model_", re.IGNORECASE)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def read_text(path: Path) -> tuple[str, str | None]:
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except Exception as exc:
        return "", f"could not read {path.name}: {exc}"


def infer_symbol_from_name(name: str) -> str:
    match = SYMBOL_RE.search(name)
    return match.group(1).upper() if match else ""


def infer_seed_from_name(name: str) -> str:
    match = SEED_RE.search(name)
    return match.group(1) if match else ""


def infer_symbol_from_log(log_text: str) -> str:
    match = LOG_SYMBOL_RE.search(log_text)
    return match.group(1).upper() if match else ""


def candidate_model_from_log(log_text: str) -> str:
    matches = CANDIDATE_MODEL_RE.findall(log_text)
    return matches[-1].strip() if matches else ""


def resolve_logged_path(path_text: str, run_dir: Path) -> Path:
    path = Path(path_text.strip().strip('"'))
    return path if path.is_absolute() else run_dir / path


def looks_like_training_confirmation_run(run_name: str) -> bool:
    lower = run_name.lower()
    return "confirm" in lower and "infer" not in lower


def parse_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return float(text)
    except Exception:
        return None


def rows_values_from_evaluation(path: Path) -> tuple[list[float], str | None]:
    values: list[float] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "rows" not in reader.fieldnames:
                return values, None
            for row in reader:
                value = parse_float(row.get("rows"))
                if value is not None:
                    values.append(value)
    except Exception as exc:
        return values, f"could not read evaluation.csv: {exc}"
    return values, None


def zero_row_flags(log_text: str, evaluation_path: Path) -> tuple[bool, bool, list[str]]:
    issues: list[str] = []
    values: list[float] = []

    for raw_value in MODEL_ROWS_RE.findall(log_text):
        value = parse_float(raw_value)
        if value is not None:
            values.append(value)

    if evaluation_path.exists():
        eval_values, issue = rows_values_from_evaluation(evaluation_path)
        values.extend(eval_values)
        if issue:
            issues.append(issue)

    if not values:
        return False, False, issues

    zero_count = sum(1 for value in values if value == 0)
    zero_trade_policy_rows = zero_count > 0
    suspicious_zero_rows = zero_count / len(values) >= 0.5
    return zero_trade_policy_rows, suspicious_zero_rows, issues


def symbol_mismatch(folder_symbol: str, log_text: str) -> bool:
    if not folder_symbol:
        return False
    log_symbols = {symbol.upper() for symbol in SYMBOL_RE.findall(log_text)}
    log_symbol = infer_symbol_from_log(log_text)
    if log_symbol:
        log_symbols.add(log_symbol)
    return bool(log_symbols and any(symbol != folder_symbol for symbol in log_symbols))


def inspect_run(run_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    run_name = run_dir.name
    run_log = run_dir / "run.log"
    annotated_csv = run_dir / "annotated.csv"
    evaluation_csv = run_dir / "evaluation.csv"
    rows_csv = run_dir / "rows.csv"

    has_run_log = run_log.exists()
    log_text = ""
    if has_run_log:
        log_text, issue = read_text(run_log)
        if issue:
            issues.append(issue)
    else:
        issues.append("missing run.log")

    folder_symbol = infer_symbol_from_name(run_name)
    log_symbol = infer_symbol_from_log(log_text)
    symbol = folder_symbol or log_symbol
    seed = infer_seed_from_name(run_name)

    loaded_frozen_model = "Loaded frozen model" in log_text
    inference_only_complete = "Inference-only complete" in log_text
    training_generations_seen = len(WINNER_RE.findall(log_text))
    training_confirmation = looks_like_training_confirmation_run(run_name)

    candidate_model_text = candidate_model_from_log(log_text)
    candidate_model_path = ""
    has_candidate_model = False
    if candidate_model_text:
        candidate_path = resolve_logged_path(candidate_model_text, run_dir)
        candidate_model_path = str(candidate_path)
        has_candidate_model = candidate_path.exists()

    candidate_model_saved = bool(candidate_model_text)
    has_annotated_csv = annotated_csv.exists()
    has_evaluation_csv = evaluation_csv.exists()
    has_rows_csv = rows_csv.exists()

    if not has_annotated_csv:
        issues.append("missing annotated.csv")
    if not has_evaluation_csv:
        issues.append("missing evaluation.csv")
    if not has_rows_csv:
        issues.append("missing rows.csv")

    zero_trade_policy_rows, suspicious_zero_rows, zero_issues = zero_row_flags(log_text, evaluation_csv)
    issues.extend(zero_issues)
    if suspicious_zero_rows:
        issues.append("most policy/evaluation rows are zero")

    symbol_path_mismatch = symbol_mismatch(folder_symbol, log_text)
    if symbol_path_mismatch:
        issues.append("folder symbol disagrees with run.log paths")

    invalid = False
    incomplete = False
    suspicious = False

    if training_confirmation and loaded_frozen_model:
        invalid = True
        issues.append("training confirmation run loaded a frozen model")
    if training_confirmation and inference_only_complete:
        invalid = True
        issues.append("training confirmation run completed inference-only")

    looks_training = (
        not inference_only_complete
        and not loaded_frozen_model
        and (training_generations_seen > 0 or training_confirmation or "train" in run_name.lower())
    )
    if looks_training and not candidate_model_saved:
        incomplete = True
        issues.append("training run did not report a candidate model")
    if looks_training and candidate_model_saved and not has_candidate_model:
        incomplete = True
        issues.append("reported candidate model file is missing")

    if not has_run_log or not has_annotated_csv or not has_evaluation_csv or not has_rows_csv:
        incomplete = True
    if suspicious_zero_rows or symbol_path_mismatch:
        suspicious = True

    if invalid:
        status = "invalid"
    elif incomplete:
        status = "incomplete"
    elif suspicious:
        status = "suspicious"
    else:
        status = "valid"

    return {
        "run": run_name,
        "symbol": symbol,
        "seed": seed,
        "has_run_log": bool_text(has_run_log),
        "has_annotated_csv": bool_text(has_annotated_csv),
        "has_evaluation_csv": bool_text(has_evaluation_csv),
        "has_rows_csv": bool_text(has_rows_csv),
        "has_candidate_model": bool_text(has_candidate_model),
        "candidate_model_path": candidate_model_path,
        "loaded_frozen_model": bool_text(loaded_frozen_model),
        "inference_only_complete": bool_text(inference_only_complete),
        "training_generations_seen": training_generations_seen,
        "candidate_model_saved": bool_text(candidate_model_saved),
        "zero_trade_policy_rows": bool_text(zero_trade_policy_rows),
        "suspicious_zero_rows": bool_text(suspicious_zero_rows),
        "symbol_path_mismatch": bool_text(symbol_path_mismatch),
        "status": status,
        "issues": "; ".join(dict.fromkeys(issues)),
    }


def malformed_run_row(run_dir: Path, exc: Exception) -> dict[str, Any]:
    return {
        "run": run_dir.name,
        "symbol": infer_symbol_from_name(run_dir.name),
        "seed": infer_seed_from_name(run_dir.name),
        "has_run_log": bool_text((run_dir / "run.log").exists()),
        "has_annotated_csv": bool_text((run_dir / "annotated.csv").exists()),
        "has_evaluation_csv": bool_text((run_dir / "evaluation.csv").exists()),
        "has_rows_csv": bool_text((run_dir / "rows.csv").exists()),
        "has_candidate_model": "false",
        "candidate_model_path": "",
        "loaded_frozen_model": "false",
        "inference_only_complete": "false",
        "training_generations_seen": 0,
        "candidate_model_saved": "false",
        "zero_trade_policy_rows": "false",
        "suspicious_zero_rows": "false",
        "symbol_path_mismatch": "false",
        "status": "invalid",
        "issues": f"malformed run folder: {exc}",
    }


def discover_run_dirs() -> list[Path]:
    if not RUN_ROOT.exists():
        return []
    return sorted(path for path in RUN_ROOT.glob(RUN_GLOB) if path.is_dir())


def write_report(rows: list[dict[str, Any]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def truncate(value: Any, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def print_summary(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["status"] for row in rows)
    print("rawseq_run_health")
    print(f"Root: {RUN_ROOT}")
    print(f"Glob: {RUN_GLOB}")
    print(f"Runs inspected: {len(rows)}")
    print(
        "Status: "
        + ", ".join(
            f"{status}={counts.get(status, 0)}"
            for status in ["valid", "suspicious", "incomplete", "invalid"]
        )
    )
    print(f"Report: {OUTPUT_PATH}")

    if not rows:
        return

    status_rank = {"invalid": 0, "incomplete": 1, "suspicious": 2, "valid": 3}
    display_rows = sorted(rows, key=lambda row: (status_rank.get(row["status"], 9), row["run"]))
    limit = int(os.getenv("RAWSEQ_RUN_HEALTH_PRINT_LIMIT", "80"))
    display_rows = display_rows[:limit]

    headers = ["status", "run", "symbol", "seed", "issues"]
    widths = {"status": 10, "run": 48, "symbol": 10, "seed": 8, "issues": 80}
    print()
    print(" ".join(header.ljust(widths[header]) for header in headers))
    print(" ".join("-" * widths[header] for header in headers))
    for row in display_rows:
        print(
            " ".join(
                truncate(row.get(header, ""), widths[header]).ljust(widths[header])
                for header in headers
            )
        )
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows in CSV")


def main() -> None:
    rows: list[dict[str, Any]] = []
    for run_dir in discover_run_dirs():
        try:
            rows.append(inspect_run(run_dir))
        except Exception as exc:
            rows.append(malformed_run_row(run_dir, exc))

    rows.sort(key=lambda row: row["run"])
    write_report(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
