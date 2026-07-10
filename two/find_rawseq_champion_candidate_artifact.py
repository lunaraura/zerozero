#!/usr/bin/env python3
"""Find candidate model artifacts matching the expected rawseq champion contract.

Read-only except for writing the search reports.
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
SYMBOL = os.getenv("RAWSEQ_FIND_SYMBOL", "SOLUSDT").strip().upper()
VENUE = os.getenv("RAWSEQ_FIND_VENUE", "kraken").strip().lower()
SOURCE_RUN_DIR = Path(
    os.getenv(
        "RAWSEQ_FIND_SOURCE_RUN_DIR",
        PROJECT_ROOT / "data" / "rawseq_runs" / "hist_b10s_ma_distance_60_h2x2_p6_g4_e35_seed_906",
    )
)
CANDIDATE_ROOT = Path(
    os.getenv(
        "RAWSEQ_FIND_CANDIDATE_ROOT",
        PROJECT_ROOT / "models" / "candidates" / SYMBOL / "tiny_price_rawseq_path_v1" / VENUE,
    )
)
if not SOURCE_RUN_DIR.is_absolute():
    SOURCE_RUN_DIR = PROJECT_ROOT / SOURCE_RUN_DIR
if not CANDIDATE_ROOT.is_absolute():
    CANDIDATE_ROOT = PROJECT_ROOT / CANDIDATE_ROOT

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "realtime"
    / VENUE
    / f"{SYMBOL}_rawseq_champion_candidate_artifact_search.csv"
)
OUTPUT_TXT = OUTPUT_CSV.with_suffix(".txt")

EXPECTED = {
    "symbol": SYMBOL,
    "venue": VENUE,
    "source_path_basename": "SOLUSDT_all_flow_combined.csv",
    "bucket_seconds": "10",
    "seq_len": "60",
    "input_stride": "1",
    "output_stride": "1",
    "input_feature": "ma_distance",
    "ma_window": "60",
    "hidden": "2,2",
    "seed": "906",
    "population": "6",
    "generations": "4",
    "epochs": "35",
}

RUN_LOG_PATH = SOURCE_RUN_DIR / "run.log"
META_PATH = SOURCE_RUN_DIR / "meta.txt"
EVALUATION_PATH = SOURCE_RUN_DIR / "evaluation.csv"
MODEL_PATH_RE = re.compile(
    r"Candidate model:\s*([A-Za-z]:\\.+?model\.json)",
    re.IGNORECASE | re.DOTALL,
)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def abs_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def normalize_hidden(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(safe_str(item) for item in value if safe_str(item))
    text = safe_str(value).replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    return ",".join(part.strip() for part in text.split(",") if part.strip())


def normalize_int_text(value: Any) -> str:
    text = safe_str(value)
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def normalize_field(field: str, value: Any) -> str:
    text = normalize_hidden(value) if field == "hidden" else safe_str(value)
    if not text:
        return ""
    if field == "symbol":
        return text.upper()
    if field in {"venue", "input_feature"}:
        return text.lower()
    if field == "source_path_basename":
        return Path(text.replace("\\", "/")).name
    if field in {
        "bucket_seconds",
        "seq_len",
        "input_stride",
        "output_stride",
        "ma_window",
        "seed",
        "population",
        "generations",
        "epochs",
    }:
        return normalize_int_text(text)
    if field == "hidden":
        return text.replace(" ", "")
    return text


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


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    except Exception:
        return ""


def candidate_from_run_log() -> Path | None:
    text = read_text(RUN_LOG_PATH)
    matches = MODEL_PATH_RE.findall(text)
    if not matches:
        return None
    path_text = re.sub(r"\s+", "", matches[-1].strip())
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def matrix_shape(value: Any) -> tuple[int | None, int | None]:
    if not isinstance(value, list):
        return None, None
    rows = len(value)
    if rows == 0:
        return 0, 0
    if not isinstance(value[0], list):
        return rows, None
    return rows, len(value[0])


def vector_len(value: Any) -> int | None:
    return len(value) if isinstance(value, list) else None


def load_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return {}, str(exc)
    return payload if isinstance(payload, dict) else {}, "root is not an object"


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return ""


def parse_model(path: Path, source_run_candidate_path: Path | None) -> dict[str, Any]:
    payload, issue = load_json(path)
    if issue and not payload:
        return {
            "model_path": abs_text(path),
            "status": "error",
            "issues": issue,
            "match_class": "unreadable",
            "match_score": 0,
            "is_source_run_candidate_path": abs_text(path) == abs_text(source_run_candidate_path),
        }

    arch = payload.get("architecture") if isinstance(payload.get("architecture"), dict) else {}
    pop = payload.get("population_settings") if isinstance(payload.get("population_settings"), dict) else {}
    weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else {}

    w1_rows, w1_cols = matrix_shape(weights.get("W1"))
    w2_rows, w2_cols = matrix_shape(weights.get("W2"))
    w3_rows, w3_cols = matrix_shape(weights.get("W3"))
    hidden_declared = ""
    if arch.get("hidden_1") is not None and arch.get("hidden_2") is not None:
        hidden_declared = normalize_hidden([arch.get("hidden_1"), arch.get("hidden_2")])
    hidden_inferred = normalize_hidden([w1_cols, w2_cols]) if w1_cols is not None and w2_cols is not None else ""

    row = {
        "model_path": abs_text(path),
        "status": "ok",
        "issues": issue,
        "is_source_run_candidate_path": abs_text(path) == abs_text(source_run_candidate_path),
        "symbol": safe_str(payload.get("symbol")),
        "venue": safe_str(payload.get("primary_venue") or payload.get("venue")),
        "source_path": safe_str(payload.get("source_path")),
        "source_path_basename": Path(safe_str(payload.get("source_path")).replace("\\", "/")).name,
        "bucket_seconds": normalize_int_text(payload.get("bucket_seconds")),
        "seq_len": normalize_int_text(payload.get("seq_len")),
        "input_stride": normalize_int_text(payload.get("input_stride") or payload.get("rawseq_input_stride") or 1),
        "output_stride": normalize_int_text(payload.get("output_stride") or payload.get("rawseq_output_stride") or 1),
        "input_feature": safe_str(payload.get("input_feature")),
        "ma_window": normalize_int_text(payload.get("ma_window") or payload.get("rawseq_ma_window")),
        "hidden_declared": hidden_declared,
        "hidden_inferred": hidden_inferred,
        "input_dim_declared": normalize_int_text(arch.get("input_dim")),
        "output_dim_declared": normalize_int_text(arch.get("output_dim")),
        "input_dim_inferred": safe_str(w1_rows),
        "output_dim_inferred": safe_str(w3_cols),
        "w1_shape": f"{w1_rows}x{w1_cols}" if w1_rows is not None else "",
        "w2_shape": f"{w2_rows}x{w2_cols}" if w2_rows is not None else "",
        "w3_shape": f"{w3_rows}x{w3_cols}" if w3_rows is not None else "",
        "b1_len": safe_str(vector_len(weights.get("b1"))),
        "b2_len": safe_str(vector_len(weights.get("b2"))),
        "b3_len": safe_str(vector_len(weights.get("b3"))),
        "seed": normalize_int_text(pop.get("seed") or payload.get("seed")),
        "population": normalize_int_text(pop.get("population") or payload.get("population")),
        "generations": normalize_int_text(pop.get("generations") or payload.get("generations")),
        "epochs": normalize_int_text(pop.get("epochs_per_generation") or payload.get("epochs")),
        "created_at": safe_str(payload.get("created_at")),
        "best_validation_fitness": safe_str(payload.get("best_validation_fitness")),
        "fitness_policy": safe_str(payload.get("fitness_policy")),
        "fitness_threshold_bps": safe_str(payload.get("fitness_threshold_bps")),
        "decision_horizon_seconds": safe_str(payload.get("decision_horizon_seconds")),
        "decision_threshold_bps": safe_str(payload.get("decision_threshold_bps")),
    }

    compared = {
        "symbol": row["symbol"],
        "venue": row["venue"],
        "source_path_basename": row["source_path_basename"],
        "bucket_seconds": row["bucket_seconds"],
        "seq_len": row["seq_len"],
        "input_stride": row["input_stride"],
        "output_stride": row["output_stride"],
        "input_feature": row["input_feature"],
        "ma_window": row["ma_window"],
        "hidden": row["hidden_declared"] or row["hidden_inferred"],
        "seed": row["seed"],
        "population": row["population"],
        "generations": row["generations"],
        "epochs": row["epochs"],
    }
    mismatches = []
    missing = []
    score = 0
    for field, expected in EXPECTED.items():
        actual = normalize_field(field, compared.get(field, ""))
        expected_norm = normalize_field(field, expected)
        if not actual:
            missing.append(field)
        elif actual == expected_norm:
            score += 1
        else:
            mismatches.append(f"{field}: expected={expected_norm} actual={actual}")

    shape_mismatches = []
    if row["input_dim_inferred"] and row["input_dim_inferred"] != EXPECTED["seq_len"]:
        shape_mismatches.append(f"input_dim_inferred={row['input_dim_inferred']}")
    if row["output_dim_inferred"] and row["output_dim_inferred"] != EXPECTED["seq_len"]:
        shape_mismatches.append(f"output_dim_inferred={row['output_dim_inferred']}")
    if row["hidden_inferred"] and row["hidden_inferred"] != EXPECTED["hidden"]:
        shape_mismatches.append(f"hidden_inferred={row['hidden_inferred']}")

    exact_required = not mismatches and not shape_mismatches and not missing
    wrong_feature = normalize_field("input_feature", row["input_feature"]) != EXPECTED["input_feature"]
    wrong_hidden = normalize_field("hidden", row["hidden_declared"] or row["hidden_inferred"]) != EXPECTED["hidden"]
    if exact_required:
        match_class = "exact_match"
    elif wrong_feature and wrong_hidden:
        match_class = "wrong_feature_and_hidden"
    elif wrong_feature:
        match_class = "wrong_feature"
    elif wrong_hidden:
        match_class = "wrong_hidden"
    elif score >= 8:
        match_class = "near_match"
    else:
        match_class = "other_mismatch"

    row["match_score"] = score
    row["match_class"] = match_class
    row["mismatches"] = "; ".join(mismatches)
    row["missing_fields"] = "; ".join(missing)
    row["shape_mismatches"] = "; ".join(shape_mismatches)
    return row


def discover_model_paths(source_run_candidate_path: Path | None) -> list[Path]:
    paths = []
    if source_run_candidate_path and source_run_candidate_path.exists():
        paths.append(source_run_candidate_path)
    if CANDIDATE_ROOT.exists():
        paths.extend(CANDIDATE_ROOT.glob("**/model.json"))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[abs_text(path)] = path
    return [unique[key] for key in sorted(unique)]


def read_evaluation_summary() -> dict[str, Any]:
    if not EVALUATION_PATH.exists():
        return {"source_evaluation_exists": False}
    try:
        frame = pd.read_csv(EVALUATION_PATH, low_memory=False)
    except Exception as exc:
        return {"source_evaluation_exists": True, "source_evaluation_error": str(exc)}
    out: dict[str, Any] = {
        "source_evaluation_exists": True,
        "source_evaluation_rows": len(frame),
        "source_evaluation_columns": ",".join(frame.columns.astype(str).tolist()),
    }
    for column in ["split", "strategy", "rows", "avg_return_bps", "cumulative_return_bps", "max_dip_bps"]:
        if column in frame.columns:
            out[f"source_evaluation_{column}_sample"] = safe_str(frame[column].dropna().head(3).tolist())
    return out


def recommendation(report: pd.DataFrame, source_run_candidate_path: Path | None) -> tuple[str, str]:
    exact = report[report["match_class"].eq("exact_match")]
    if not exact.empty:
        preferred = exact.sort_values(
            ["is_source_run_candidate_path", "created_at"],
            ascending=[False, False],
        ).iloc[0]
        return (
            "A",
            "recover exact model artifact into a new clean champion folder: "
            + str(preferred["model_path"]),
        )

    source_rows = report[report["is_source_run_candidate_path"].astype(bool)]
    if not source_rows.empty:
        row = source_rows.iloc[0]
        if row["match_class"] in {"wrong_feature", "wrong_hidden", "wrong_feature_and_hidden"}:
            return (
                "C",
                "source-run declared candidate exists but does not match the expected contract; retrain/refreeze required",
            )
        return (
            "B",
            "source-run candidate path exists but has metadata gaps/near match; verify whether current champion metadata is wrong",
        )

    return ("C", "no correct artifact found; retrain/refreeze required")


def render_text(
    report: pd.DataFrame,
    source_run_candidate_path: Path | None,
    eval_summary: dict[str, Any],
    rec_code: str,
    rec_text: str,
) -> str:
    counts = report["match_class"].value_counts().to_dict() if not report.empty else {}
    lines = [
        "Rawseq Champion Candidate Artifact Search",
        "",
        f"Source run dir: {abs_text(SOURCE_RUN_DIR)}",
        f"Candidate root: {abs_text(CANDIDATE_ROOT)}",
        f"Run-log candidate path: {abs_text(source_run_candidate_path)}",
        "",
        "Expected Contract",
        *[f"  {key}: {value}" for key, value in EXPECTED.items()],
        "",
        "Counts",
        *[f"  {key}: {counts.get(key, 0)}" for key in [
            "exact_match",
            "near_match",
            "wrong_feature",
            "wrong_hidden",
            "wrong_feature_and_hidden",
            "other_mismatch",
            "unreadable",
        ]],
        "",
        f"Recommendation: {rec_code}. {rec_text}",
        "",
        "Top Candidates",
        "  class                    score source_run path",
        "  ------------------------ ----- ---------- ------------------------------------------------------------",
    ]
    if report.empty:
        lines.append("  none")
    else:
        display = report.sort_values(
            ["match_class", "match_score", "is_source_run_candidate_path", "created_at"],
            ascending=[True, False, False, False],
        )
        class_rank = {
            "exact_match": 0,
            "near_match": 1,
            "wrong_feature": 2,
            "wrong_hidden": 3,
            "wrong_feature_and_hidden": 4,
            "other_mismatch": 5,
            "unreadable": 6,
        }
        display = display.assign(_rank=display["match_class"].map(class_rank).fillna(9))
        display = display.sort_values(["_rank", "match_score", "is_source_run_candidate_path"], ascending=[True, False, False])
        for _, row in display.head(20).iterrows():
            lines.append(
                "  "
                + " ".join(
                    [
                        str(row["match_class"])[:24].ljust(24),
                        str(row["match_score"]).rjust(5),
                        ("yes" if row["is_source_run_candidate_path"] else "no").ljust(10),
                        str(row["model_path"])[:100],
                    ]
                )
            )
            details = "; ".join(
                part
                for part in [
                    safe_str(row.get("mismatches")),
                    safe_str(row.get("shape_mismatches")),
                    safe_str(row.get("missing_fields")),
                ]
                if part
            )
            if details:
                lines.append(f"    {details[:180]}")

    lines.extend(
        [
            "",
            "Source Run Inputs",
            f"  run.log exists: {RUN_LOG_PATH.exists()}",
            f"  meta.txt exists: {META_PATH.exists()}",
            f"  evaluation.csv exists: {EVALUATION_PATH.exists()}",
            f"  evaluation rows: {eval_summary.get('source_evaluation_rows', '')}",
            "",
            f"CSV report: {abs_text(OUTPUT_CSV)}",
            f"Text report: {abs_text(OUTPUT_TXT)}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    source_run_candidate_path = candidate_from_run_log()
    paths = discover_model_paths(source_run_candidate_path)
    rows = [parse_model(path, source_run_candidate_path) for path in paths]
    report = pd.DataFrame(rows)
    if report.empty:
        report = pd.DataFrame(
            columns=[
                "model_path",
                "status",
                "match_class",
                "match_score",
                "is_source_run_candidate_path",
            ]
        )

    eval_summary = read_evaluation_summary()
    rec_code, rec_text = recommendation(report, source_run_candidate_path)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_CSV, index=False)
    text = render_text(report, source_run_candidate_path, eval_summary, rec_code, rec_text)
    OUTPUT_TXT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
