#!/usr/bin/env python3
"""Prepare a portable GPU handoff package for rawseq sequence benchmarks.

This is a report/package step only. It validates the exported sequence dataset
manifest, records shapes and optional hashes, and writes the exact PowerShell
commands needed on a torch/CUDA-capable machine.

Safety: no training, no private API, no orders, no promotion, and no champion
mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_gpu_sequence_handoffs"
REQUIRED_SEQUENCE_LENS = [60, 120, 240]
REQUIRED_FEATURE_GROUPS = ["raw", "raw_trend", "raw_momentum", "raw_volatility", "raw_volume", "raw_cross_market", "all"]


def resolve_path(value: str | Path, default: Path | None = None) -> Path:
    path = Path(value or default or ".").expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def latest_sequence_manifest() -> Path:
    roots = [
        Path("F:/rsio/rawseq_multi_horizon_indicator_full_contract_smoke"),
        PROJECT_ROOT / "data" / "research" / "rawseq_multi_horizon_indicator_returns",
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("**/sequence_dataset_manifest.csv"))
    if not candidates:
        raise SystemExit("RAWSEQ_GPU_HANDOFF_SEQUENCE_MANIFEST is required; no sequence_dataset_manifest.csv found")
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quick_file_fingerprint(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if size > 1024 * 1024:
            handle.seek(max(size - 1024 * 1024, 0))
            digest.update(handle.read(1024 * 1024))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "path_npz" not in frame.columns:
        raise SystemExit(f"Missing path_npz column in {path}")
    return frame


def validate_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        x_shape = list(data["X"].shape)
        y_shape = list(data["y"].shape)
        splits = data["splits"].astype(str) if "splits" in data.files else np.asarray([], dtype=str)
        horizons = data["horizon_buckets"].astype(int).tolist() if "horizon_buckets" in data.files else []
        feature_columns = data["feature_columns"].astype(str).tolist() if "feature_columns" in data.files else []
        target_columns = data["target_columns"].astype(str).tolist() if "target_columns" in data.files else []
        timestamps = data["decision_timestamps"].astype(float) if "decision_timestamps" in data.files else np.asarray([])
    return {
        "x_shape_actual": json.dumps(x_shape),
        "y_shape_actual": json.dumps(y_shape),
        "split_values": ",".join(sorted(set(splits.astype(str)))) if len(splits) else "",
        "horizon_buckets": ",".join(str(item) for item in horizons),
        "feature_column_count": len(feature_columns),
        "target_column_count": len(target_columns),
        "timestamp_min": float(np.nanmin(timestamps)) if len(timestamps) else math.nan,
        "timestamp_max": float(np.nanmax(timestamps)) if len(timestamps) else math.nan,
        "shape_guard_pass": bool(len(x_shape) == 3 and len(y_shape) == 2 and x_shape[0] == y_shape[0]),
    }


def build_dataset_inventory(
    manifest: pd.DataFrame,
    manifest_path: Path,
    hash_mode: str,
    max_hash_files: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    manifest_dir = manifest_path.parent
    hashed = 0
    for idx, row in manifest.iterrows():
        path = Path(str(row.get("path_npz", "")))
        if not path.is_absolute():
            path = manifest_dir / path
        exists = path.exists()
        base = {
            "dataset_index": idx,
            "feature_group": row.get("feature_group", ""),
            "seq_len": int(safe_float(row.get("seq_len"), -1)),
            "manifest_status": row.get("status", ""),
            "arrays_written": parse_bool(row.get("arrays_written", False)),
            "path_npz": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else 0,
            "checksum_mode": hash_mode,
            "sha256": "",
            "quick_fingerprint": "",
            "validation_status": "missing_file" if not exists else "ok",
        }
        if exists:
            try:
                base.update(validate_npz(path))
            except Exception as exc:
                base["validation_status"] = "npz_validation_failed"
                base["validation_error"] = str(exc)
            if hash_mode == "full" and (max_hash_files <= 0 or hashed < max_hash_files):
                base["sha256"] = file_sha256(path)
                hashed += 1
            elif hash_mode in {"quick", "full"}:
                base["quick_fingerprint"] = quick_file_fingerprint(path)
        rows.append(base)
    return pd.DataFrame(rows)


def build_powershell_script(
    sequence_manifest: Path,
    output_root: Path,
    audit_output_root: Path,
    epochs: int,
    hidden: int,
    batch_size: int,
    require_cuda: bool,
) -> str:
    require_cuda_value = "true" if require_cuda else "false"
    return "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"$env:RAWSEQ_TORCH_SEQUENCE_DATASET_MANIFEST = '{sequence_manifest}'",
            f"$env:RAWSEQ_TORCH_SEQUENCE_OUTPUT_DIR = '{output_root}'",
            "$env:RAWSEQ_TORCH_SEQUENCE_MODELS = 'mlp,tcn,gru,lstm,transformer'",
            f"$env:RAWSEQ_TORCH_SEQUENCE_EPOCHS = '{epochs}'",
            f"$env:RAWSEQ_TORCH_SEQUENCE_HIDDEN = '{hidden}'",
            f"$env:RAWSEQ_TORCH_SEQUENCE_BATCH_SIZE = '{batch_size}'",
            "$env:RAWSEQ_TORCH_SEQUENCE_LOSS = 'huber'",
            "$env:RAWSEQ_TORCH_POLICY = 'direct_gt'",
            "$env:RAWSEQ_TORCH_THRESHOLDS_BPS = '0,0.1,0.25,0.5,1,2'",
            "$env:RAWSEQ_TORCH_COSTS_BPS = '0.1,1,5'",
            f"$env:RAWSEQ_TORCH_REQUIRE_CUDA = '{require_cuda_value}'",
            "python scripts/tiny/run_rawseq_torch_sequence_benchmark.py",
            "",
            "$LatestTorchRun = Get-ChildItem $env:RAWSEQ_TORCH_SEQUENCE_OUTPUT_DIR -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1",
            "if (-not $LatestTorchRun) { throw 'No torch benchmark output directory found.' }",
            f"$env:RAWSEQ_GOAL_AUDIT_INDICATOR_RUN_DIR = '{sequence_manifest.parent}'",
            "$env:RAWSEQ_GOAL_AUDIT_TORCH_RUN_DIR = $LatestTorchRun.FullName",
            f"$env:RAWSEQ_GOAL_AUDIT_OUTPUT_DIR = '{audit_output_root}'",
            "python scripts/tiny/report_rawseq_multi_horizon_goal_audit.py",
            "",
        ]
    )


def build_report_lines(summary: dict[str, Any], inventory: pd.DataFrame) -> list[str]:
    missing = inventory[~inventory["exists"].astype(bool)] if not inventory.empty else pd.DataFrame()
    invalid = inventory[inventory["validation_status"].astype(str).ne("ok")] if not inventory.empty else pd.DataFrame()
    lines = [
        "# Rawseq GPU Sequence Handoff",
        "",
        f"Created at: {summary['created_at']}",
        f"Sequence manifest: {summary['sequence_manifest']}",
        f"Output dir: {summary['output_dir']}",
        f"Dataset rows: {summary['dataset_count']}",
        f"Total dataset GB: {summary['total_dataset_gb']:.3f}",
        f"Validation status: {summary['validation_status']}",
        "",
        "## Required Coverage",
        f"- Feature groups observed: {summary['feature_groups_observed']}",
        f"- Sequence lengths observed: {summary['sequence_lens_observed']}",
        f"- Missing feature groups: {summary['missing_feature_groups']}",
        f"- Missing sequence lengths: {summary['missing_sequence_lens']}",
        "",
        "## Transfer Notes",
        "- Copy or mount the full indicator run directory containing sequence_dataset_manifest.csv and sequence_datasets/.",
        "- Run generated run_gpu_sequence_benchmark.ps1 from the repository root on a torch/CUDA-capable machine.",
        "- No ensemble search is authorized by this handoff; wait for the audit to report survivor evidence.",
        "",
        "## Problems",
        f"- Missing files: {len(missing)}",
        f"- Invalid datasets: {len(invalid)}",
        "",
        "## Safety",
        "- no_training=true for this handoff preparation step",
        "- private_api=false",
        "- orders=false",
        "- promotion=false",
        "- champion_mutation=false",
        "- ensemble_search=false",
    ]
    return lines


def main() -> int:
    manifest_env = os.getenv("RAWSEQ_GPU_HANDOFF_SEQUENCE_MANIFEST", "").strip()
    manifest_path = resolve_path(manifest_env) if manifest_env else latest_sequence_manifest()
    output_root = resolve_path(os.getenv("RAWSEQ_GPU_HANDOFF_OUTPUT_DIR", ""), DEFAULT_OUTPUT_ROOT)
    output_dir = output_root / f"rawseq_gpu_sequence_handoff_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    hash_mode = os.getenv("RAWSEQ_GPU_HANDOFF_HASH_MODE", "quick").strip().lower()
    if hash_mode not in {"none", "quick", "full"}:
        hash_mode = "quick"
    max_hash_files = int(float(os.getenv("RAWSEQ_GPU_HANDOFF_MAX_HASH_FILES", "0")))
    epochs = int(float(os.getenv("RAWSEQ_GPU_HANDOFF_EPOCHS", "10")))
    hidden = int(float(os.getenv("RAWSEQ_GPU_HANDOFF_HIDDEN", "64")))
    batch_size = int(float(os.getenv("RAWSEQ_GPU_HANDOFF_BATCH_SIZE", "512")))
    require_cuda = parse_bool(os.getenv("RAWSEQ_GPU_HANDOFF_REQUIRE_CUDA", "true"))

    manifest = load_manifest(manifest_path)
    inventory = build_dataset_inventory(manifest, manifest_path, hash_mode, max_hash_files)
    feature_groups = sorted(set(inventory["feature_group"].astype(str))) if not inventory.empty else []
    seq_lens = sorted({int(safe_float(value, -1)) for value in inventory.get("seq_len", []) if safe_float(value, -1) > 0})
    missing_feature_groups = [group for group in REQUIRED_FEATURE_GROUPS if group not in feature_groups]
    missing_seq_lens = [seq_len for seq_len in REQUIRED_SEQUENCE_LENS if seq_len not in seq_lens]
    invalid_count = int((inventory["validation_status"].astype(str) != "ok").sum()) if not inventory.empty else 0
    missing_count = int((~inventory["exists"].astype(bool)).sum()) if not inventory.empty else 0
    validation_status = (
        "ok"
        if not missing_feature_groups and not missing_seq_lens and invalid_count == 0 and missing_count == 0
        else "incomplete"
    )

    torch_output_root = resolve_path(os.getenv("RAWSEQ_GPU_HANDOFF_TORCH_OUTPUT_DIR", "F:/rsio/rawseq_torch_sequence_benchmark_full_gpu"))
    audit_output_root = resolve_path(os.getenv("RAWSEQ_GPU_HANDOFF_AUDIT_OUTPUT_DIR", "F:/rsio/rawseq_goal_audits"))
    ps1_text = build_powershell_script(
        manifest_path,
        torch_output_root,
        audit_output_root,
        epochs=epochs,
        hidden=hidden,
        batch_size=batch_size,
        require_cuda=require_cuda,
    )

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "sequence_manifest": str(manifest_path),
        "indicator_run_dir": str(manifest_path.parent),
        "output_dir": str(output_dir),
        "dataset_count": int(len(inventory)),
        "total_dataset_bytes": int(inventory["size_bytes"].sum()) if not inventory.empty else 0,
        "total_dataset_gb": float(inventory["size_bytes"].sum() / 1e9) if not inventory.empty else 0.0,
        "feature_groups_observed": feature_groups,
        "sequence_lens_observed": seq_lens,
        "missing_feature_groups": missing_feature_groups,
        "missing_sequence_lens": missing_seq_lens,
        "validation_status": validation_status,
        "hash_mode": hash_mode,
        "gpu_command_script": str(output_dir / "run_gpu_sequence_benchmark.ps1"),
        "paper_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "ensemble_search": False,
    }

    inventory_path = output_dir / "gpu_handoff_dataset_inventory.csv"
    summary_path = output_dir / "gpu_handoff_manifest.json"
    report_path = output_dir / "gpu_handoff_report.txt"
    ps1_path = output_dir / "run_gpu_sequence_benchmark.ps1"
    inventory.to_csv(inventory_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(build_report_lines(summary, inventory)) + "\n", encoding="utf-8")
    ps1_path.write_text(ps1_text, encoding="utf-8")

    print("Rawseq GPU sequence handoff prepared")
    print(f"Output dir: {output_dir}")
    print(f"Validation status: {validation_status}")
    print(f"Datasets: {len(inventory)}")
    print(f"Total dataset GB: {summary['total_dataset_gb']:.3f}")
    print(f"PowerShell script: {ps1_path}")
    print(f"Inventory: {inventory_path}")
    print("Safety: no_training=true private_api=false orders=false promotion=false champion_mutation=false ensemble_search=false")
    print(inventory[["feature_group", "seq_len", "exists", "size_bytes", "validation_status", "shape_guard_pass"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
