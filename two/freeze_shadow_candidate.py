#!/usr/bin/env python3
"""Freeze a registry-selected rawseq candidate into a research shadow folder.

This is a paper-only archival helper. It copies an already-selected model and
threshold-specific probe reports into data/research, writes provenance and safety
metadata, and marks copied files read-only. It never writes to paper_champions.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_probe_registry"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_shadow_candidates"

REGISTRY_DIR_ENV = os.getenv("RAWSEQ_SHADOW_REGISTRY_DIR", "").strip()
REGISTRY_ROW_INDEX = os.getenv("RAWSEQ_SHADOW_REGISTRY_ROW_INDEX", "0").strip()
REGISTRY_KEY = os.getenv("RAWSEQ_SHADOW_REGISTRY_KEY", "").strip()
OUTPUT_ROOT_ENV = os.getenv("RAWSEQ_SHADOW_OUTPUT_ROOT", "").strip()
OUTPUT_SLUG_ENV = os.getenv("RAWSEQ_SHADOW_OUTPUT_SLUG", "").strip()
ALLOW_OVERWRITE = os.getenv("RAWSEQ_SHADOW_ALLOW_OVERWRITE", "0").strip().lower() in {"1", "true", "yes"}


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_slug(value: str, fallback: str = "shadow_candidate") -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("._")
    return value or fallback


def latest_registry_dir() -> Path:
    if REGISTRY_DIR_ENV:
        return resolve_project_path(REGISTRY_DIR_ENV)
    if not REGISTRY_ROOT.exists():
        raise SystemExit(f"Registry root not found: {REGISTRY_ROOT}")
    candidates = [path for path in REGISTRY_ROOT.iterdir() if path.is_dir()]
    if not candidates:
        raise SystemExit(f"No registry folders found under {REGISTRY_ROOT}")
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)[0]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_registry_row(registry_dir: Path) -> dict[str, str]:
    path = registry_dir / "top_shadow_candidates.csv"
    if not path.exists():
        raise SystemExit(f"Missing top_shadow_candidates.csv: {path}")
    rows = read_csv_rows(path)
    if not rows:
        raise SystemExit(f"No candidate rows found in {path}")

    if REGISTRY_KEY:
        for row in rows:
            key = safe_str(row.get("probe_threshold_key")) or "|".join(
                [
                    safe_str(row.get("probe_dir")),
                    safe_str(row.get("model_path")),
                    safe_str(row.get("threshold_bps")),
                    safe_str(row.get("decision_cost_bps")),
                ]
            )
            if key == REGISTRY_KEY:
                return row
        raise SystemExit("RAWSEQ_SHADOW_REGISTRY_KEY did not match any registry row.")

    try:
        index = int(REGISTRY_ROW_INDEX)
    except ValueError as exc:
        raise SystemExit(f"RAWSEQ_SHADOW_REGISTRY_ROW_INDEX must be an integer: {REGISTRY_ROW_INDEX}") from exc
    if index < 0 or index >= len(rows):
        raise SystemExit(f"Registry row index {index} outside 0..{len(rows) - 1}")
    return rows[index]


def threshold_token(row: dict[str, str]) -> str:
    value = safe_str(row.get("best_threshold_for_probe")) or safe_str(row.get("threshold_bps"))
    if not value:
        raise SystemExit("Selected registry row has no threshold.")
    return value


def threshold_file_token(value: str) -> str:
    numeric = value.strip()
    if "." in numeric:
        numeric = numeric.rstrip("0").rstrip(".")
    return numeric or value.strip()


def copy_file(src: Path, dst: Path, copied: list[dict[str, str]], required: bool = True) -> None:
    if not src.exists():
        if required:
            raise SystemExit(f"Required source file missing: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append({"source": str(src), "destination": str(dst), "bytes": str(dst.stat().st_size)})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_read_only(path: Path) -> None:
    if path.is_file():
        path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def make_tree_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        make_read_only(path)


def output_root() -> Path:
    if OUTPUT_ROOT_ENV:
        return resolve_project_path(OUTPUT_ROOT_ENV)
    return DEFAULT_OUTPUT_ROOT


def output_slug(row: dict[str, str]) -> str:
    if OUTPUT_SLUG_ENV:
        return safe_slug(OUTPUT_SLUG_ENV)
    parts = [
        safe_str(row.get("symbol")) or "symbol",
        safe_str(row.get("venue")) or "venue",
        safe_str(row.get("input_feature")) or "feature",
        f"ma{safe_str(row.get('ma_window')) or 'NA'}",
        f"h{safe_str(row.get('hidden')).replace(',', 'x') or 'NA'}",
        f"seed{safe_str(row.get('seed')) or 'NA'}",
        f"thr{safe_slug(threshold_token(row).replace('.', 'p'))}",
    ]
    return safe_slug("_".join(parts))


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    registry_dir = latest_registry_dir()
    row = select_registry_row(registry_dir)
    probe_dir = resolve_project_path(safe_str(row.get("probe_dir")))
    model_path = resolve_project_path(safe_str(row.get("model_path")))
    threshold = threshold_token(row)
    threshold_file = threshold_file_token(threshold)

    if not probe_dir.exists():
        raise SystemExit(f"Probe directory not found: {probe_dir}")
    if not model_path.exists():
        raise SystemExit(f"Model path not found: {model_path}")

    dest_root = output_root() / f"{output_slug(row)}_{now_stamp()}_{uuid.uuid4().hex[:8]}"
    if dest_root.exists() and not ALLOW_OVERWRITE:
        raise SystemExit(f"Output folder already exists: {dest_root}")
    dest_root.mkdir(parents=True, exist_ok=ALLOW_OVERWRITE)

    copied: list[dict[str, str]] = []
    copy_file(model_path, dest_root / "model.json", copied, required=True)

    report_files = [
        "model_contract.json",
        "contract_audit.csv",
        "cost_threshold_summary.csv",
        "rolling_summary.csv",
        "evaluation.csv",
        "summary.txt",
        "run.log",
        f"decision_summary_threshold_{threshold_file}.txt",
        f"decision_summary_threshold_{threshold_file}.csv",
        f"dynamic_cost_summary_threshold_{threshold_file}.txt",
        f"dynamic_cost_summary_threshold_{threshold_file}.csv",
    ]
    for name in report_files:
        copy_file(probe_dir / name, dest_root / "reports" / name, copied, required=False)

    registry_row_path = dest_root / "registry_row.csv"
    with registry_row_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    copied.append({"source": str(registry_dir / "top_shadow_candidates.csv"), "destination": str(registry_row_path), "bytes": str(registry_row_path.stat().st_size)})

    model_contract = load_json_if_exists(probe_dir / "model_contract.json")
    provenance = {
        "created_at": now_stamp(),
        "purpose": "read_only_research_shadow_candidate_freeze",
        "registry_dir": str(registry_dir),
        "registry_source": str(registry_dir / "top_shadow_candidates.csv"),
        "probe_threshold_key": safe_str(row.get("probe_threshold_key")),
        "threshold_bps": threshold,
        "decision_cost_bps": safe_str(row.get("decision_cost_bps")),
        "status": safe_str(row.get("status")),
        "decision": safe_str(row.get("decision")),
        "status_explanation": safe_str(row.get("status_explanation")),
        "probe_dir": str(probe_dir),
        "model_path": str(model_path),
        "contract": {
            "symbol": safe_str(row.get("symbol")),
            "venue": safe_str(row.get("venue")),
            "source_path_basename": safe_str(row.get("source_path_basename")),
            "input_feature": safe_str(row.get("input_feature")),
            "ma_window": safe_str(row.get("ma_window")),
            "hidden": safe_str(row.get("hidden")),
            "seq_len": safe_str(row.get("seq_len")),
            "bucket_seconds": safe_str(row.get("bucket_seconds")),
            "input_stride": safe_str(row.get("input_stride")),
            "output_stride": safe_str(row.get("output_stride")),
            "seed": safe_str(row.get("seed")),
        },
        "metrics": {
            "selected_rows": safe_str(row.get("selected_rows")),
            "fixed_0_10_cum_net": safe_str(row.get("fixed_0_10_cum_net")),
            "fixed_0_25_cum_net": safe_str(row.get("fixed_0_25_cum_net")),
            "half_spread_plus_0_05_cum_net": safe_str(row.get("half_spread_plus_0_05_cum_net")),
            "conservative_missing_liquidity_penalty_cum_net": safe_str(row.get("conservative_missing_liquidity_penalty_cum_net")),
            "max_dip_to_cum_net_ratio": safe_str(row.get("max_dip_to_cum_net_ratio")),
            "positive_12h_window_fraction": safe_str(row.get("positive_12h_window_fraction")),
            "positive_24h_window_fraction": safe_str(row.get("positive_24h_window_fraction")),
        },
        "model_contract": model_contract,
        "copied_files": copied,
    }
    safety = {
        "paper_only": True,
        "read_only_research_shadow_candidate": True,
        "training": False,
        "champion_mutation": False,
        "paper_champions_mutation": False,
        "promotion": False,
        "orders": False,
        "private_api": False,
        "live_trading": False,
        "notes": [
            "This folder is a research shadow archive only.",
            "No champion folder was created or mutated.",
            "Promotion or freeze into paper_champions requires a separate explicit script and audit.",
        ],
    }

    write_json(dest_root / "provenance.json", provenance)
    write_json(dest_root / "safety_metadata.json", safety)
    write_text(
        dest_root / "README.txt",
        [
            "Rawseq Research Shadow Candidate",
            "",
            f"Created at: {provenance['created_at']}",
            f"Status: {provenance['status']}",
            f"Threshold bps: {threshold}",
            f"Probe dir: {probe_dir}",
            f"Model path: {model_path}",
            "",
            "Safety:",
            "- paper_only=true",
            "- read_only_research_shadow_candidate=true",
            "- training=false",
            "- champion_mutation=false",
            "- promotion=false",
            "- orders=false",
            "",
            "This is not a paper champion and must not be ranked as one without a separate freeze/audit step.",
        ],
    )

    make_tree_read_only(dest_root)

    print("Rawseq shadow candidate freeze complete")
    print(f"Registry dir: {registry_dir}")
    print(f"Probe dir: {probe_dir}")
    print(f"Model path: {model_path}")
    print(f"Threshold bps: {threshold}")
    print(f"Output dir: {dest_root}")
    print("Safety: no training. No paper_champions mutation. No promotion. No orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
