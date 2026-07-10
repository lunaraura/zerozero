#!/usr/bin/env python3
"""Triage archived rawseq grid runs for follow-up probe review.

Read-only except for writing triage reports.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRID_ROOT = Path(os.getenv("RAWSEQ_GRID_ROOT", str(PROJECT_ROOT / "data" / "rawseq_runs")))
if not GRID_ROOT.is_absolute():
    GRID_ROOT = PROJECT_ROOT / GRID_ROOT
SUMMARY_GLOB = os.getenv("RAWSEQ_GRID_SUMMARY_GLOB", "overnight_rawseq_grid_summary_*.csv")

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "realtime" / "kraken" / "rawseq_overnight_grid_triage.csv"
OUTPUT_PATH = Path(os.getenv("RAWSEQ_TRIAGE_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH
ROLLUP_PATH = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_rollup.csv")
TEXT_PATH = OUTPUT_PATH.with_suffix(".txt")

ERROR_PATTERNS = [
    "traceback",
    "systemexit",
    "error",
    "exception",
    "failed",
    "exit code 1",
    "unknown rawseq_input_feature",
]
BENIGN_LOG_PATTERNS = [
    "performancewarning",
    "nativecommanderror",
]
SAFETY_BAD_PATTERNS = [
    "orders: true",
    "orders=true",
    "promotion: true",
    "promotion=true",
    "champion_replacement: true",
    "champion_replacement=true",
    "private_api: true",
    "private_api=true",
]

MODEL_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[^\r\n\"']*models[\\/]+candidates[^\r\n\"']*model\.json)",
    re.IGNORECASE,
)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except Exception:
        return math.nan


def safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def normalize_int(value: Any) -> str:
    text = safe_str(value)
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def matrix_shape(value: Any) -> tuple[int | None, int | None]:
    if not isinstance(value, list):
        return None, None
    rows = len(value)
    if rows == 0:
        return 0, 0
    if not isinstance(value[0], list):
        return rows, None
    return rows, len(value[0])


def normalize_hidden(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(normalize_int(item) for item in value if safe_str(item))
    text = safe_str(value).replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    return ",".join(part.strip() for part in text.split(",") if part.strip()).replace(" ", "")


def parse_meta(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    except Exception:
        return {}
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def read_log(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    except Exception:
        return ""


def resolve_logged_model_path(text: str) -> Path | None:
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("candidate_model_path="):
            candidate = stripped.split("=", 1)[1].strip().strip('"')
            path = Path(candidate)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                return path.resolve()
        if lower.startswith("candidate model:"):
            candidate = stripped.split(":", 1)[1].strip().strip('"')
            if candidate:
                path = Path(candidate)
                if not path.is_absolute():
                    path = PROJECT_ROOT / path
                if path.exists():
                    return path.resolve()
    for match in MODEL_PATH_RE.finditer(text):
        candidate = match.group("path").strip().strip('"').strip("'")
        path = Path(candidate)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return path.resolve()
    return None


def parse_payload_contract(path: Path | None) -> dict[str, str]:
    empty = {
        "candidate_model_path": str(path) if path else "",
        "payload_input_feature": "",
        "payload_hidden": "",
        "payload_seq_len": "",
        "payload_bucket_seconds": "",
        "payload_input_stride": "",
        "payload_output_stride": "",
        "payload_source_path_basename": "",
        "payload_seed": "",
        "payload_created_at": "",
        "payload_w1_shape": "",
        "payload_w2_shape": "",
        "payload_w3_shape": "",
    }
    if path is None or not path.exists():
        return empty
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return empty
    if not isinstance(payload, dict):
        return empty
    arch = payload.get("architecture") if isinstance(payload.get("architecture"), dict) else {}
    weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else {}
    pop = payload.get("population_settings") if isinstance(payload.get("population_settings"), dict) else {}
    w1_rows, w1_cols = matrix_shape(weights.get("W1"))
    w2_rows, w2_cols = matrix_shape(weights.get("W2"))
    w3_rows, w3_cols = matrix_shape(weights.get("W3"))
    hidden = ""
    if arch.get("hidden_1") is not None and arch.get("hidden_2") is not None:
        hidden = normalize_hidden([arch.get("hidden_1"), arch.get("hidden_2")])
    elif w1_cols is not None and w2_cols is not None:
        hidden = normalize_hidden([w1_cols, w2_cols])
    source_path = safe_str(payload.get("source_path"))
    return {
        "candidate_model_path": str(path),
        "payload_input_feature": safe_str(payload.get("input_feature")),
        "payload_hidden": hidden,
        "payload_seq_len": normalize_int(payload.get("seq_len") or arch.get("input_dim") or w1_rows),
        "payload_bucket_seconds": normalize_int(payload.get("bucket_seconds")),
        "payload_input_stride": normalize_int(payload.get("input_stride") or payload.get("rawseq_input_stride") or 1),
        "payload_output_stride": normalize_int(payload.get("output_stride") or payload.get("rawseq_output_stride") or 1),
        "payload_source_path_basename": Path(source_path.replace("\\", "/")).name if source_path else "",
        "payload_seed": normalize_int(pop.get("seed") or payload.get("seed")),
        "payload_created_at": safe_str(payload.get("created_at")),
        "payload_w1_shape": f"{w1_rows}x{w1_cols}" if w1_rows is not None else "",
        "payload_w2_shape": f"{w2_rows}x{w2_cols}" if w2_rows is not None else "",
        "payload_w3_shape": f"{w3_rows}x{w3_cols}" if w3_rows is not None else "",
    }


def log_has_errors(text: str) -> bool:
    lower = safe_str(text).lower()
    if "performancewarning" in lower and not any(
        pattern in lower for pattern in ["traceback", "unknown rawseq_input_feature", "exit code 1"]
    ):
        return False
    return any(pattern in lower for pattern in ERROR_PATTERNS)


def safety_flags(text: str, evaluation: pd.DataFrame | None) -> dict[str, Any]:
    lower = safe_str(text).lower()
    bad_log = any(pattern in lower for pattern in SAFETY_BAD_PATTERNS)
    eval_bad = False
    eval_ok = False
    if evaluation is not None and not evaluation.empty:
        safety_columns = [column for column in ["paper_only", "promotion", "champion_replacement", "private_api", "orders"] if column in evaluation.columns]
        if safety_columns:
            eval_ok = True
            for column in safety_columns:
                values = evaluation[column].astype(str).str.lower().str.strip()
                if column == "paper_only":
                    eval_bad = eval_bad or values.isin(["false", "0", "no"]).any()
                else:
                    eval_bad = eval_bad or values.isin(["true", "1", "yes"]).any()
    text_mentions_paper = "paper-only" in lower or "paper_only" in lower
    return {
        "paper_only_no_promotion_no_orders": bool(not bad_log and not eval_bad and (eval_ok or text_mentions_paper)),
        "safety_warning": bool(bad_log or eval_bad),
    }


def load_grid_summaries() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in sorted(GRID_ROOT.glob(SUMMARY_GLOB)):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    tag = safe_str(row.get("tag"))
                    archive = safe_str(row.get("archive"))
                    if tag:
                        item = {str(k): safe_str(v) for k, v in row.items()}
                        item["_summary_path"] = str(path)
                        rows[tag] = item
                    if archive:
                        item = {str(k): safe_str(v) for k, v in row.items()}
                        item["_summary_path"] = str(path)
                        rows[Path(archive).name] = item
        except Exception:
            continue
    return rows


def read_evaluation(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def score_evaluation(evaluation: pd.DataFrame | None) -> dict[str, Any]:
    empty = {
        "best_split": "",
        "best_strategy": "",
        "best_fitness": math.nan,
        "best_rows": 0,
        "best_avg_return_bps": math.nan,
        "best_cumulative_return_bps": math.nan,
        "best_win_rate": math.nan,
        "best_max_dip_bps": math.nan,
        "test_rows_total": 0,
        "validation_rows_total": 0,
        "generation_rows_total": 0,
        "score": -math.inf,
    }
    if evaluation is None or evaluation.empty:
        return empty

    frame = evaluation.copy()
    for column in ["fitness", "rows", "avg_return_bps", "cumulative_return_bps", "win_rate", "max_dip_bps"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    split_text = frame["split"].astype(str).str.lower() if "split" in frame.columns else pd.Series("", index=frame.index)
    test = frame[split_text.str.contains("test", na=False)].copy()
    validation = frame[split_text.str.contains("validation|val", na=False)].copy()
    generation = frame[frame.get("fitness", pd.Series(index=frame.index, dtype=float)).notna()].copy()

    candidates = test if not test.empty else validation if not validation.empty else generation
    if candidates.empty:
        candidates = frame

    if "cumulative_return_bps" in candidates.columns and candidates["cumulative_return_bps"].notna().any():
        best_idx = candidates["cumulative_return_bps"].idxmax()
        score = safe_float(candidates.loc[best_idx, "cumulative_return_bps"])
    elif "fitness" in candidates.columns and candidates["fitness"].notna().any():
        best_idx = candidates["fitness"].idxmax()
        score = safe_float(candidates.loc[best_idx, "fitness"])
    elif "avg_return_bps" in candidates.columns and candidates["avg_return_bps"].notna().any():
        best_idx = candidates["avg_return_bps"].idxmax()
        score = safe_float(candidates.loc[best_idx, "avg_return_bps"])
    else:
        best_idx = candidates.index[0]
        score = -math.inf
    best = candidates.loc[best_idx]
    return {
        "best_split": safe_str(best.get("split")),
        "best_strategy": safe_str(best.get("strategy")),
        "best_fitness": safe_float(best.get("fitness")),
        "best_rows": safe_int(best.get("rows")),
        "best_avg_return_bps": safe_float(best.get("avg_return_bps")),
        "best_cumulative_return_bps": safe_float(best.get("cumulative_return_bps")),
        "best_win_rate": safe_float(best.get("win_rate")),
        "best_max_dip_bps": safe_float(best.get("max_dip_bps")),
        "test_rows_total": int(pd.to_numeric(test.get("rows", pd.Series(dtype=float)), errors="coerce").sum()) if not test.empty else 0,
        "validation_rows_total": int(pd.to_numeric(validation.get("rows", pd.Series(dtype=float)), errors="coerce").sum()) if not validation.empty else 0,
        "generation_rows_total": int(pd.to_numeric(generation.get("rows", pd.Series(dtype=float)), errors="coerce").sum()) if not generation.empty else 0,
        "score": score,
    }


def infer_from_tag(tag: str) -> dict[str, str]:
    text = tag.lower()
    hidden = ""
    match = re.search(r"_h(\d+)x(\d+)", text)
    if match:
        hidden = f"{match.group(1)},{match.group(2)}"
    seed = ""
    match = re.search(r"seed[_-]?(\d+)", text)
    if match:
        seed = match.group(1)
    ma_window = ""
    match = re.search(r"ma_distance_(\d+)|_w(\d+)_", text)
    if match:
        ma_window = next(group for group in match.groups() if group)
    feature = ""
    if "ma_distance" in text:
        feature = "ma_distance"
    elif "signed_return" in text or "_return_" in text:
        feature = "return"
    return {"input_feature": feature, "ma_window": ma_window, "hidden": hidden, "seed": seed}


def parse_run(run_dir: Path, summaries: dict[str, dict[str, str]]) -> dict[str, Any]:
    meta = parse_meta(run_dir / "meta.txt")
    summary = summaries.get(run_dir.name, {})
    inferred = infer_from_tag(run_dir.name)
    evaluation = read_evaluation(run_dir / "evaluation.csv")
    metrics = score_evaluation(evaluation)
    log_text = read_log(run_dir / "run.log")
    flags = safety_flags(log_text, evaluation)
    candidate_model_path = resolve_logged_model_path(log_text)
    payload_contract = parse_payload_contract(candidate_model_path)

    annotated_path = run_dir / "annotated.csv"
    rows_path = run_dir / "rows.csv"
    model_path = run_dir / "model.json"
    eval_path = run_dir / "evaluation.csv"
    source_path = safe_str(meta.get("source_path") or summary.get("source"))
    source_basename = Path(source_path.replace("\\", "/")).name if source_path else ""
    status = safe_str(summary.get("status"))
    if not status:
        if log_has_errors(log_text):
            status = "FAILED"
        elif eval_path.exists() and annotated_path.exists():
            status = "OK"
        elif eval_path.exists():
            status = "PARTIAL"
        else:
            status = "MISSING"

    issues: list[str] = []
    if status.upper() not in {"OK", "SUCCESS", "COMPLETE", "COMPLETED", "PARTIAL"}:
        issues.append(f"status={status}")
    if not eval_path.exists():
        issues.append("missing_evaluation")
    if not annotated_path.exists():
        issues.append("missing_annotated")
    if not rows_path.exists():
        issues.append("missing_rows")
    if not (run_dir / "run.log").exists():
        issues.append("missing_run_log")
    if log_has_errors(log_text):
        issues.append("log_errors")
    if flags["safety_warning"]:
        issues.append("safety_warning")

    archive_input_feature = safe_str(meta.get("input_feature") or summary.get("feature") or inferred["input_feature"])
    archive_hidden = safe_str(meta.get("hidden") or summary.get("hidden") or inferred["hidden"])
    archive_source_basename = source_basename
    archive_input_stride_raw = meta.get("input_stride") or meta.get("rawseq_input_stride") or summary.get("input_stride")
    archive_output_stride_raw = meta.get("output_stride") or meta.get("rawseq_output_stride") or summary.get("output_stride")
    archive_input_stride = normalize_int(archive_input_stride_raw or "1")
    archive_output_stride = normalize_int(archive_output_stride_raw or "1")
    payload_available = bool(payload_contract.get("payload_input_feature"))
    contract_mismatch = False
    if payload_available:
        checks = [
            (archive_input_feature, payload_contract.get("payload_input_feature")),
            (archive_hidden, payload_contract.get("payload_hidden")),
            (archive_source_basename, payload_contract.get("payload_source_path_basename")),
        ]
        if archive_input_stride_raw:
            checks.append((archive_input_stride, payload_contract.get("payload_input_stride")))
        if archive_output_stride_raw:
            checks.append((archive_output_stride, payload_contract.get("payload_output_stride")))
        contract_mismatch = any(safe_str(left) and safe_str(right) and safe_str(left) != safe_str(right) for left, right in checks)
        if contract_mismatch:
            issues.append("contract_mismatch")

    row = {
        "tag": safe_str(meta.get("tag") or summary.get("tag") or run_dir.name),
        "run_dir": str(run_dir.resolve()),
        "status": status,
        "exit_code": safe_str(summary.get("exit_code")),
        "input_feature": archive_input_feature,
        "ma_window": normalize_int(meta.get("ma_window") or summary.get("window") or inferred["ma_window"]),
        "hidden": archive_hidden,
        "seed": normalize_int(meta.get("seed") or summary.get("seed") or inferred["seed"]),
        "source_path": source_path,
        "source_path_basename": source_basename,
        "bucket_seconds": normalize_int(meta.get("bucket_seconds")),
        "seq_len": normalize_int(meta.get("seq_len")),
        "input_stride": archive_input_stride,
        "output_stride": archive_output_stride,
        "population": normalize_int(meta.get("population")),
        "generations": normalize_int(meta.get("generations")),
        "epochs": normalize_int(meta.get("epochs")),
        "evaluation_exists": eval_path.exists(),
        "annotated_exists": annotated_path.exists(),
        "rows_exists": rows_path.exists(),
        "model_exists": model_path.exists(),
        "run_log_exists": (run_dir / "run.log").exists(),
        "annotated_size_bytes": annotated_path.stat().st_size if annotated_path.exists() else 0,
        "rows_size_bytes": rows_path.stat().st_size if rows_path.exists() else 0,
        "log_contains_errors": log_has_errors(log_text),
        "paper_only_no_promotion_no_orders": flags["paper_only_no_promotion_no_orders"],
        "safety_warning": flags["safety_warning"],
        "contract_mismatch": contract_mismatch,
        "issues": ";".join(issues),
        "triage_class": "",
        **payload_contract,
        **metrics,
    }
    row["effective_input_feature"] = row["payload_input_feature"] or row["input_feature"]
    row["effective_hidden"] = row["payload_hidden"] or row["hidden"]
    row["effective_seq_len"] = row["payload_seq_len"] or row["seq_len"]
    row["effective_bucket_seconds"] = row["payload_bucket_seconds"] or row["bucket_seconds"]
    row["effective_input_stride"] = row["payload_input_stride"] or row["input_stride"]
    row["effective_output_stride"] = row["payload_output_stride"] or row["output_stride"]
    row["effective_source_path_basename"] = row["payload_source_path_basename"] or row["source_path_basename"]
    row["effective_seed"] = row["payload_seed"] or row["seed"]
    row["contract_group"] = "|".join(
        [
            row["effective_input_feature"],
            row["ma_window"],
            row["effective_hidden"],
            row["effective_seq_len"],
            row["effective_bucket_seconds"],
            row["effective_input_stride"],
            row["effective_output_stride"],
            row["effective_source_path_basename"],
        ]
    )
    row["probe_candidate_score"] = safe_float(row["score"])
    return row


def classify_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    classes: list[str] = []
    for _, row in out.iterrows():
        issues = safe_str(row.get("issues"))
        score = safe_float(row.get("probe_candidate_score"))
        best_rows = safe_int(row.get("best_rows"))
        status = safe_str(row.get("status")).upper()
        if row.get("safety_warning"):
            classes.append("ignore_safety_warning")
        elif "missing_evaluation" in issues or "log_errors" in issues or status == "FAILED":
            classes.append("failed_or_stale")
        elif not math.isfinite(score):
            classes.append("ignore_no_score")
        elif best_rows < 25:
            classes.append("ignore_too_sparse")
        elif score > 0 or safe_float(row.get("best_avg_return_bps")) > 0:
            classes.append("probe_candidate")
        else:
            classes.append("ignore_nonpositive")
    out["triage_class"] = classes
    return out


def build_rollup(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [
        "effective_input_feature",
        "ma_window",
        "effective_hidden",
        "effective_seq_len",
        "effective_bucket_seconds",
        "effective_input_stride",
        "effective_output_stride",
        "effective_source_path_basename",
    ]
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        ranked = group.sort_values(
            ["probe_candidate_score", "best_cumulative_return_bps", "best_avg_return_bps"],
            ascending=[False, False, False],
        )
        best = ranked.iloc[0]
        rows.append(
            {
                "input_feature": keys[0],
                "ma_window": keys[1],
                "hidden": keys[2],
                "seq_len": keys[3],
                "bucket_seconds": keys[4],
                "input_stride": keys[5],
                "output_stride": keys[6],
                "source_path_basename": keys[7],
                "runs": int(len(group)),
                "ok_or_partial_runs": int(group["status"].astype(str).str.upper().isin(["OK", "PARTIAL", "SUCCESS", "COMPLETE", "COMPLETED"]).sum()),
                "failed_or_stale_runs": int(group["triage_class"].astype(str).str.contains("failed|stale|ignore_no_score").sum()),
                "probe_candidates": int(group["triage_class"].eq("probe_candidate").sum()),
                "safety_warnings": int(group["safety_warning"].astype(bool).sum()),
                "best_tag": best["tag"],
                "best_run_dir": best["run_dir"],
                "best_seed": best["seed"],
                "best_effective_seed": best["effective_seed"],
                "contract_mismatches": int(group["contract_mismatch"].astype(bool).sum()),
                "best_score": best["probe_candidate_score"],
                "best_rows": best["best_rows"],
                "best_avg_return_bps": best["best_avg_return_bps"],
                "best_cumulative_return_bps": best["best_cumulative_return_bps"],
                "best_max_dip_bps": best["best_max_dip_bps"],
                "best_triage_class": best["triage_class"],
            }
        )
    rollup = pd.DataFrame(rows)
    if not rollup.empty:
        rollup = rollup.sort_values(
            ["probe_candidates", "best_score", "best_avg_return_bps"],
            ascending=[False, False, False],
        )
    return rollup


def render_text(frame: pd.DataFrame, rollup: pd.DataFrame) -> str:
    safety_count = int(frame["safety_warning"].astype(bool).sum()) if not frame.empty else 0
    mismatch_count = int(frame["contract_mismatch"].astype(bool).sum()) if not frame.empty else 0
    lines = [
        "Rawseq Overnight Grid Triage",
        "",
        f"Grid root: {GRID_ROOT}",
        f"Runs scanned: {len(frame)}",
        f"Contract groups: {len(rollup)}",
        f"Safety warnings: {safety_count}",
        f"Contract mismatches: {mismatch_count}",
        "",
        "1. Failed/Stale/Missing Runs",
    ]
    non_contract_issues = frame["issues"].astype(str).str.replace("contract_mismatch", "", regex=False).str.strip(";")
    failed = frame[
        non_contract_issues.ne("")
        | frame["status"].astype(str).str.upper().isin(["FAILED", "MISSING"])
        | frame["log_contains_errors"].astype(bool)
        | frame["safety_warning"].astype(bool)
    ].copy()
    if failed.empty:
        lines.append("  none")
    else:
        for _, row in failed.head(25).iterrows():
            lines.append(f"  {row['tag']} status={row['status']} issues={row['issues'] or 'none'}")
        if len(failed) > 25:
            lines.append(f"  ... {len(failed) - 25} more")

    lines += ["", "2. Best Candidate Per Contract Group"]
    if rollup.empty:
        lines.append("  none")
    else:
        for _, row in rollup.head(25).iterrows():
            lines.append(
                f"  {row['input_feature']} ma={row['ma_window']} hidden={row['hidden']} "
                f"seq={row['seq_len']} bucket={row['bucket_seconds']} "
                f"stride={row['input_stride']}/{row['output_stride']} "
                f"source={row['source_path_basename']} best={row['best_tag']} "
                f"score={safe_float(row['best_score']):.6g} rows={int(row['best_rows'])} "
                f"class={row['best_triage_class']}"
            )

    lines += ["", "2b. Payload Contract Mismatch Warnings"]
    mismatches = frame[frame["contract_mismatch"].astype(bool)].copy()
    if mismatches.empty:
        lines.append("  none")
    else:
        for _, row in mismatches.head(30).iterrows():
            lines.append(
                f"  {row['tag']} archive={row['input_feature']}/{row['hidden']}/{row['source_path_basename']} "
                f"stride={row['input_stride']}/{row['output_stride']} "
                f"payload={row['payload_input_feature']}/{row['payload_hidden']}/{row['payload_source_path_basename']} "
                f"payload_stride={row['payload_input_stride']}/{row['payload_output_stride']} "
                f"model={row['candidate_model_path']}"
            )
        if len(mismatches) > 30:
            lines.append(f"  ... {len(mismatches) - 30} more")

    probe = frame[frame["triage_class"].eq("probe_candidate")].sort_values(
        ["probe_candidate_score", "best_avg_return_bps"], ascending=[False, False]
    )
    lines += ["", "3. Candidates Worth Running Through run_rawseq_candidate_shadow_probe.py"]
    if probe.empty:
        lines.append("  none")
    else:
        for _, row in probe.head(30).iterrows():
            lines.append(
                f"  {row['tag']} feature={row['effective_input_feature']} ma={row['ma_window']} "
                f"hidden={row['effective_hidden']} seed={row['effective_seed']} score={safe_float(row['probe_candidate_score']):.6g} "
                f"stride={row['effective_input_stride']}/{row['effective_output_stride']} "
                f"avg={safe_float(row['best_avg_return_bps']):.6g} rows={int(row['best_rows'])} "
                f"run_dir={row['run_dir']}"
            )

    ignore = frame[~frame["triage_class"].isin(["probe_candidate", "failed_or_stale"])].copy()
    lines += ["", "4. Candidates To Ignore"]
    if ignore.empty:
        lines.append("  none")
    else:
        for _, row in ignore.head(30).iterrows():
            lines.append(f"  {row['tag']} class={row['triage_class']} score={safe_float(row['probe_candidate_score']):.6g} issues={row['issues'] or 'none'}")
        if len(ignore) > 30:
            lines.append(f"  ... {len(ignore) - 30} more")

    lines += ["", "5. Safety Warning"]
    if safety_count:
        safety = frame[frame["safety_warning"].astype(bool)]
        lines.append("  WARNING: at least one run appears to mention promotion/champion mutation/private API/orders.")
        for _, row in safety.head(20).iterrows():
            lines.append(f"  {row['tag']} issues={row['issues']}")
    else:
        lines.append("  No promotion/champion mutation/private API/orders warning detected.")

    lines += [
        "",
        "Safety: read-only except reports. No training. No champion mutation. No promotion. No orders.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if not GRID_ROOT.exists():
        raise SystemExit(f"RAWSEQ_GRID_ROOT does not exist: {GRID_ROOT}")
    summaries = load_grid_summaries()
    run_dirs = sorted(path for path in GRID_ROOT.iterdir() if path.is_dir())
    rows = [parse_run(path, summaries) for path in run_dirs]
    frame = classify_rows(pd.DataFrame(rows))
    rollup = build_rollup(frame)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    rollup.to_csv(ROLLUP_PATH, index=False)
    text = render_text(frame, rollup)
    TEXT_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"Triage CSV: {OUTPUT_PATH}")
    print(f"Triage rollup: {ROLLUP_PATH}")
    print(f"Triage text: {TEXT_PATH}")


if __name__ == "__main__":
    main()
