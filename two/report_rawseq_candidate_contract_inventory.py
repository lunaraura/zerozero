#!/usr/bin/env python3
"""Inventory SOLUSDT rawseq candidate model contracts.

Read-only except for writing inventory reports.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYMBOL = os.getenv("RAWSEQ_INVENTORY_SYMBOL", "SOLUSDT").strip().upper()
VENUE = os.getenv("RAWSEQ_INVENTORY_VENUE", "kraken").strip().lower()
DEFAULT_CANDIDATE_ROOT = (
    PROJECT_ROOT / "models" / "candidates" / SYMBOL / "tiny_price_rawseq_path_v1" / VENUE
)
CANDIDATE_ROOT = Path(os.getenv("RAWSEQ_INVENTORY_CANDIDATE_ROOT", str(DEFAULT_CANDIDATE_ROOT)))
if not CANDIDATE_ROOT.is_absolute():
    CANDIDATE_ROOT = PROJECT_ROOT / CANDIDATE_ROOT

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "realtime"
    / VENUE
    / f"{SYMBOL}_rawseq_candidate_contract_inventory.csv"
)
OUTPUT_PATH = Path(os.getenv("RAWSEQ_INVENTORY_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH
ROLLUP_PATH = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_rollup.csv")
TEXT_OUTPUT_PATH = OUTPUT_PATH.with_suffix(".txt")

CONTAMINATED_CHAMPION_DIR = (
    PROJECT_ROOT / "data" / "paper_champions" / "rawseq_fade_ma_distance_60_h2x2_v1_seed906"
)
CONTAMINATED_MODEL_PATH = CONTAMINATED_CHAMPION_DIR / "model.json"
CONTAMINATED_SPEC_PATH = CONTAMINATED_CHAMPION_DIR / "champion_spec.txt"

METADATA_CONTRACT = {
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
}


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
    return ",".join(part.strip() for part in text.split(",") if part.strip()).replace(" ", "")


def normalize_int(value: Any) -> str:
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
        "input_span_buckets",
        "output_span_buckets",
        "input_span_seconds",
        "output_span_seconds",
        "ma_window",
        "seed",
        "population",
        "generations",
        "epochs",
    }:
        return normalize_int(text)
    if field == "hidden":
        return text.replace(" ", "")
    return text


def parse_key_value_text(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
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


def load_model(path: Path) -> tuple[dict[str, Any], str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return {}, str(exc)
    if not isinstance(payload, dict):
        return {}, "model payload root is not an object"
    return payload, ""


def payload_contract(payload: dict[str, Any]) -> dict[str, str]:
    arch = payload.get("architecture") if isinstance(payload.get("architecture"), dict) else {}
    weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else {}
    pop = payload.get("population_settings") if isinstance(payload.get("population_settings"), dict) else {}
    w1_rows, w1_cols = matrix_shape(weights.get("W1"))
    w2_rows, w2_cols = matrix_shape(weights.get("W2"))
    w3_rows, w3_cols = matrix_shape(weights.get("W3"))
    hidden_declared = ""
    if arch.get("hidden_1") is not None and arch.get("hidden_2") is not None:
        hidden_declared = normalize_hidden([arch.get("hidden_1"), arch.get("hidden_2")])
    hidden_inferred = normalize_hidden([w1_cols, w2_cols]) if w1_cols is not None and w2_cols is not None else ""
    bucket_seconds = normalize_field("bucket_seconds", payload.get("bucket_seconds"))
    seq_len = normalize_field("seq_len", payload.get("seq_len"))
    input_stride = normalize_field("input_stride", payload.get("input_stride") or payload.get("rawseq_input_stride") or 1)
    output_stride = normalize_field("output_stride", payload.get("output_stride") or payload.get("rawseq_output_stride") or 1)
    input_span_buckets = normalize_field(
        "input_span_buckets",
        payload.get("input_span_buckets") or payload.get("rawseq_input_span_buckets"),
    )
    output_span_buckets = normalize_field(
        "output_span_buckets",
        payload.get("output_span_buckets") or payload.get("rawseq_output_span_buckets"),
    )
    input_span_seconds = normalize_field(
        "input_span_seconds",
        payload.get("input_span_seconds") or payload.get("rawseq_input_span_seconds"),
    )
    output_span_seconds = normalize_field(
        "output_span_seconds",
        payload.get("output_span_seconds") or payload.get("rawseq_output_span_seconds"),
    )
    if not input_span_buckets and seq_len and input_stride:
        input_span_buckets = str(int(seq_len) * int(input_stride))
    if not output_span_buckets and seq_len and output_stride:
        output_span_buckets = str(int(seq_len) * int(output_stride))
    if not input_span_seconds and input_span_buckets and bucket_seconds:
        input_span_seconds = str(int(input_span_buckets) * int(bucket_seconds))
    if not output_span_seconds and output_span_buckets and bucket_seconds:
        output_span_seconds = str(int(output_span_buckets) * int(bucket_seconds))
    return {
        "symbol": normalize_field("symbol", payload.get("symbol")),
        "venue": normalize_field("venue", payload.get("primary_venue") or payload.get("venue")),
        "source_path": safe_str(payload.get("source_path")),
        "source_path_basename": normalize_field("source_path_basename", payload.get("source_path")),
        "bucket_seconds": bucket_seconds,
        "seq_len": seq_len,
        "input_stride": input_stride,
        "output_stride": output_stride,
        "input_span_buckets": input_span_buckets,
        "output_span_buckets": output_span_buckets,
        "input_span_seconds": input_span_seconds,
        "output_span_seconds": output_span_seconds,
        "input_feature": normalize_field("input_feature", payload.get("input_feature")),
        "ma_window": normalize_field("ma_window", payload.get("ma_window") or payload.get("rawseq_ma_window")),
        "hidden_declared": hidden_declared,
        "hidden_inferred": hidden_inferred,
        "hidden": hidden_declared or hidden_inferred,
        "input_dim_declared": normalize_int(arch.get("input_dim")),
        "output_dim_declared": normalize_int(arch.get("output_dim")),
        "input_dim_inferred": safe_str(w1_rows),
        "output_dim_inferred": safe_str(w3_cols),
        "w1_shape": f"{w1_rows}x{w1_cols}" if w1_rows is not None else "",
        "w2_shape": f"{w2_rows}x{w2_cols}" if w2_rows is not None else "",
        "w3_shape": f"{w3_rows}x{w3_cols}" if w3_rows is not None else "",
        "b1_len": safe_str(vector_len(weights.get("b1"))),
        "b2_len": safe_str(vector_len(weights.get("b2"))),
        "b3_len": safe_str(vector_len(weights.get("b3"))),
        "seed": normalize_field("seed", pop.get("seed") or payload.get("seed")),
        "population": normalize_field("population", pop.get("population") or payload.get("population")),
        "generations": normalize_field("generations", pop.get("generations") or payload.get("generations")),
        "epochs": normalize_field("epochs", pop.get("epochs_per_generation") or payload.get("epochs")),
        "created_at": safe_str(payload.get("created_at")),
        "best_validation_fitness": safe_str(payload.get("best_validation_fitness")),
        "fitness_policy": safe_str(payload.get("fitness_policy")),
        "fitness_threshold_bps": safe_str(payload.get("fitness_threshold_bps")),
        "decision_horizon_seconds": safe_str(payload.get("decision_horizon_seconds")),
        "decision_threshold_bps": safe_str(payload.get("decision_threshold_bps")),
        "validation_rows": safe_str(payload.get("split", {}).get("validation_rows"))
        if isinstance(payload.get("split"), dict)
        else "",
        "test_rows": safe_str(payload.get("split", {}).get("test_rows")) if isinstance(payload.get("split"), dict) else "",
    }


def parse_model(path: Path, contaminated_payload_contract: dict[str, str]) -> dict[str, Any]:
    payload, issue = load_model(path)
    folder = path.parent.name
    if issue and not payload:
        return {
            "path": abs_text(path),
            "timestamp_folder": folder,
            "status": "error",
            "issues": issue,
        }
    contract = payload_contract(payload)
    row: dict[str, Any] = {
        "path": abs_text(path),
        "timestamp_folder": folder,
        "status": "ok",
        "issues": issue,
        **contract,
    }
    row["matches_contaminated_payload"] = contract_matches(contract, contaminated_payload_contract)
    row["matches_contaminated_metadata_contract"] = contract_matches(contract, METADATA_CONTRACT)
    return row


def contract_matches(contract: dict[str, str], expected: dict[str, str]) -> bool:
    for field, expected_value in expected.items():
        if normalize_field(field, contract.get(field, "")) != normalize_field(field, expected_value):
            return False
    return True


def contaminated_payload_contract() -> dict[str, str]:
    payload, _ = load_model(CONTAMINATED_MODEL_PATH)
    if not payload:
        return {}
    contract = payload_contract(payload)
    return {
        "symbol": contract.get("symbol", ""),
        "venue": contract.get("venue", ""),
        "source_path_basename": contract.get("source_path_basename", ""),
        "bucket_seconds": contract.get("bucket_seconds", ""),
        "seq_len": contract.get("seq_len", ""),
        "input_stride": contract.get("input_stride", "1"),
        "output_stride": contract.get("output_stride", "1"),
        "input_feature": contract.get("input_feature", ""),
        "ma_window": contract.get("ma_window", ""),
        "hidden": contract.get("hidden", ""),
    }


def contaminated_metadata_contract() -> dict[str, str]:
    spec = parse_key_value_text(CONTAMINATED_SPEC_PATH)
    return {
        "symbol": SYMBOL,
        "venue": spec.get("primary_venue", VENUE),
        "source_path_basename": spec.get("source") or spec.get("source_path", ""),
        "bucket_seconds": spec.get("bucket_seconds", ""),
        "seq_len": spec.get("seq_len", ""),
        "input_stride": spec.get("input_stride") or spec.get("rawseq_input_stride", "1"),
        "output_stride": spec.get("output_stride") or spec.get("rawseq_output_stride", "1"),
        "input_feature": spec.get("input_feature", ""),
        "ma_window": spec.get("ma_window", ""),
        "hidden": spec.get("hidden", ""),
    }


def discover_models() -> list[Path]:
    if not CANDIDATE_ROOT.exists():
        return []
    return sorted(CANDIDATE_ROOT.glob("**/model.json"))


def build_inventory() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str]]:
    payload_expected = contaminated_payload_contract()
    metadata_expected = contaminated_metadata_contract()
    # Use champion spec when present, while preserving documented defaults.
    for key, value in METADATA_CONTRACT.items():
        metadata_expected.setdefault(key, value)
        if not metadata_expected[key]:
            metadata_expected[key] = value

    rows = [parse_model(path, payload_expected) for path in discover_models()]
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        inventory = pd.DataFrame(columns=["path", "timestamp_folder", "status"])
    for column in [
        "input_feature",
        "source_path_basename",
        "hidden",
        "seq_len",
        "bucket_seconds",
        "input_stride",
        "output_stride",
        "best_validation_fitness",
    ]:
        if column not in inventory.columns:
            inventory[column] = ""

    ok = inventory[inventory["status"].eq("ok")].copy()
    if ok.empty:
        rollup = pd.DataFrame()
    else:
        ok["_fitness_num"] = pd.to_numeric(ok["best_validation_fitness"], errors="coerce")
        rollup = (
            ok.groupby(
                [
                    "input_feature",
                    "source_path_basename",
                    "hidden",
                    "seq_len",
                    "bucket_seconds",
                    "input_stride",
                    "output_stride",
                ],
                dropna=False,
            )
            .agg(
                models=("path", "count"),
                min_created_at=("created_at", "min"),
                max_created_at=("created_at", "max"),
                seeds=("seed", lambda values: ",".join(sorted({str(v) for v in values if str(v)}))),
                best_validation_fitness_min=("_fitness_num", "min"),
                best_validation_fitness_median=("_fitness_num", "median"),
                best_validation_fitness_max=("_fitness_num", "max"),
                matches_contaminated_payload=("matches_contaminated_payload", "sum"),
                matches_contaminated_metadata_contract=("matches_contaminated_metadata_contract", "sum"),
            )
            .reset_index()
            .sort_values(["models", "input_feature", "hidden"], ascending=[False, True, True])
        )
    return inventory, rollup, payload_expected, metadata_expected


def lane_summary(
    inventory: pd.DataFrame,
    input_feature: str,
    hidden: str,
    source_basename: str,
) -> tuple[pd.DataFrame, str]:
    if inventory.empty:
        return inventory, "0 candidates"
    mask = (
        inventory["status"].eq("ok")
        & inventory["input_feature"].eq(input_feature)
        & inventory["hidden"].eq(hidden)
        & inventory["source_path_basename"].eq(source_basename)
    )
    lane = inventory[mask].copy()
    if lane.empty:
        return lane, "0 candidates"
    fitness = pd.to_numeric(lane["best_validation_fitness"], errors="coerce")
    best_idx = fitness.idxmax() if fitness.notna().any() else lane.index[0]
    best = lane.loc[best_idx]
    return lane, (
        f"{len(lane)} candidates; best_by_fitness={best.get('timestamp_folder', '')} "
        f"fitness={best.get('best_validation_fitness', '')} seed={best.get('seed', '')}"
    )


def best_match_text(inventory: pd.DataFrame, column: str) -> str:
    if column not in inventory.columns:
        return "none"
    matches = inventory[inventory[column].astype(bool)].copy()
    if matches.empty:
        return "none"
    fitness = pd.to_numeric(matches["best_validation_fitness"], errors="coerce")
    if fitness.notna().any():
        row = matches.loc[fitness.idxmax()]
    else:
        row = matches.iloc[0]
    return f"{row.get('timestamp_folder', '')} {row.get('path', '')}"


def render_contract(contract: dict[str, str]) -> str:
    keys = [
        "symbol",
        "venue",
        "source_path_basename",
        "bucket_seconds",
        "seq_len",
        "input_stride",
        "output_stride",
        "input_feature",
        "ma_window",
        "hidden",
    ]
    return "; ".join(f"{key}={contract.get(key, '')}" for key in keys)


def render_text(
    inventory: pd.DataFrame,
    rollup: pd.DataFrame,
    payload_expected: dict[str, str],
    metadata_expected: dict[str, str],
) -> str:
    lanes = [
        (
            "ma_distance / 2,2 candidates",
            "ma_distance",
            "2,2",
            "SOLUSDT_all_flow_combined.csv",
        ),
        (
            "signed_bucket_return_bps / 2,2 candidates using SOLUSDT_all_flow_combined.csv",
            "signed_bucket_return_bps",
            "2,2",
            "SOLUSDT_all_flow_combined.csv",
        ),
        (
            "signed_bucket_return_bps / 4,4 candidates using SOLUSDT_all_flow_combined.csv",
            "signed_bucket_return_bps",
            "4,4",
            "SOLUSDT_all_flow_combined.csv",
        ),
    ]
    lines = [
        "Rawseq Candidate Contract Inventory",
        "",
        f"Candidate root: {abs_text(CANDIDATE_ROOT)}",
        f"Models inspected: {len(inventory)}",
        f"Contract groups: {len(rollup)}",
        "",
        "Contaminated Champion Comparisons",
        f"  payload contract: {render_contract(payload_expected)}",
        f"  metadata contract: {render_contract(metadata_expected)}",
        f"  candidate matching contaminated payload: {best_match_text(inventory, 'matches_contaminated_payload')}",
        f"  candidate matching contaminated metadata contract: {best_match_text(inventory, 'matches_contaminated_metadata_contract')}",
        "",
        "Requested Lanes",
    ]
    for title, feature, hidden, source in lanes:
        lane, summary = lane_summary(inventory, feature, hidden, source)
        lines.append(f"  {title}: {summary}")
        if not lane.empty:
            display = lane.copy()
            display["_fitness"] = pd.to_numeric(display["best_validation_fitness"], errors="coerce")
            display = display.sort_values(["_fitness", "created_at"], ascending=[False, False]).head(5)
            for _, row in display.iterrows():
                lines.append(
                    "    "
                    + f"{row.get('timestamp_folder', '')} seed={row.get('seed', '')} "
                    + f"stride={row.get('input_stride', '1')}/{row.get('output_stride', '1')} "
                    + f"fitness={row.get('best_validation_fitness', '')} path={row.get('path', '')}"
                )

    lines.extend(
        [
            "",
            "Top Contract Groups",
            "  models feature                    source                         hidden seq bucket in out payload metadata",
            "  ------ -------------------------- ------------------------------ ------ --- ------ -- --- ------- --------",
        ]
    )
    if rollup.empty:
        lines.append("  none")
    else:
        for _, row in rollup.head(20).iterrows():
            lines.append(
                "  "
                + " ".join(
                    [
                        str(int(row["models"])).rjust(6),
                        str(row["input_feature"])[:26].ljust(26),
                        str(row["source_path_basename"])[:30].ljust(30),
                        str(row["hidden"])[:6].ljust(6),
                        str(row["seq_len"])[:3].rjust(3),
                        str(row["bucket_seconds"])[:6].rjust(6),
                        str(row["input_stride"])[:2].rjust(2),
                        str(row["output_stride"])[:3].rjust(3),
                        str(int(row["matches_contaminated_payload"])).rjust(7),
                        str(int(row["matches_contaminated_metadata_contract"])).rjust(8),
                    ]
                )
            )

    lines.extend(
        [
            "",
            "Recommendation",
            "  - retire contaminated champion",
            "  - create truthful signed_bucket_return_bps/2,2 champion if we want to test source-run lineage",
            "  - create truthful signed_bucket_return_bps/4,4 champion if we want to test current payload lineage",
            "  - separately test later ma_distance/2,2 candidates only as new candidates, not as the old champion",
            "",
            f"CSV inventory: {abs_text(OUTPUT_PATH)}",
            f"Rollup CSV: {abs_text(ROLLUP_PATH)}",
            f"Text summary: {abs_text(TEXT_OUTPUT_PATH)}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    inventory, rollup, payload_expected, metadata_expected = build_inventory()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUTPUT_PATH, index=False)
    rollup.to_csv(ROLLUP_PATH, index=False)
    text = render_text(inventory, rollup, payload_expected, metadata_expected)
    TEXT_OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
