#!/usr/bin/env python3
"""Run rawseq evolution over recorded Kraken walk-forward windows.

This is an orchestration wrapper around scripts/tiny_price_rawseq_path_v1.py.
It uses only recorded/public source files, archives each run, and never mutates
paper champions, promotes, or places orders.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAWSEQ_SCRIPT = PROJECT_ROOT / "scripts" / "tiny_price_rawseq_path_v1.py"

SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper() or "SOLUSDT"
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower() or "kraken"
SOURCE_PATH = Path(
    os.getenv(
        "RAWSEQ_WF_SOURCE_PATH",
        str(PROJECT_ROOT / "data" / "realtime" / PRIMARY_VENUE / f"{SYMBOL}_10s_flow.csv"),
    )
)
if not SOURCE_PATH.is_absolute():
    SOURCE_PATH = PROJECT_ROOT / SOURCE_PATH

OUTPUT_ROOT = Path(
    os.getenv("RAWSEQ_WF_OUTPUT_ROOT", str(PROJECT_ROOT / "data" / "rawseq_walkforward"))
)
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

INPUT_FEATURES = os.getenv("RAWSEQ_WF_INPUT_FEATURES", "return,ma_distance,ma_slope")
MA_WINDOWS = os.getenv("RAWSEQ_WF_MA_WINDOWS", "60,150,300")
FEATURE_WINDOWS = os.getenv("RAWSEQ_WF_FEATURE_WINDOWS", os.getenv("RAWSEQ_IO_FEATURE_WINDOWS", MA_WINDOWS))
HIDDENS = os.getenv("RAWSEQ_WF_HIDDENS", "2,2;3,3;4,4")
SEEDS = os.getenv("RAWSEQ_WF_SEEDS", "900,901,902")
INPUT_STRIDES = os.getenv("RAWSEQ_WF_INPUT_STRIDES", os.getenv("RAWSEQ_INPUT_STRIDE", "1"))
OUTPUT_STRIDES = os.getenv("RAWSEQ_WF_OUTPUT_STRIDES", os.getenv("RAWSEQ_OUTPUT_STRIDE", "1"))
OUTPUT_LABELS = os.getenv("RAWSEQ_WF_OUTPUT_LABELS", os.getenv("RAWSEQ_OUTPUT_LABEL", "future_return_path"))

TRAIN_ROWS = int(os.getenv("RAWSEQ_WF_TRAIN_ROWS", "60000"))
VALIDATION_ROWS = int(os.getenv("RAWSEQ_WF_VALIDATION_ROWS", "15000"))
TEST_ROWS = int(os.getenv("RAWSEQ_WF_TEST_ROWS", "15000"))
STEP_ROWS = int(os.getenv("RAWSEQ_WF_STEP_ROWS", "15000"))
MAX_WINDOWS = int(os.getenv("RAWSEQ_WF_MAX_WINDOWS", "10"))

BUCKET_SECONDS = int(os.getenv("RAWSEQ_WF_BUCKET_SECONDS", os.getenv("RAWSEQ_BUCKET_SECONDS", "10")))
SEQ_LEN = int(os.getenv("RAWSEQ_WF_SEQ_LEN", os.getenv("RAWSEQ_LEN", "60")))
POPULATION = int(os.getenv("RAWSEQ_WF_POPULATION", os.getenv("RAWSEQ_POPULATION", os.getenv("RAWSEQ_IO_DISCOVERY_POPULATION", "5"))))
GENERATIONS = int(os.getenv("RAWSEQ_WF_GENERATIONS", os.getenv("RAWSEQ_GENERATIONS", os.getenv("RAWSEQ_IO_DISCOVERY_GENERATIONS", "3"))))
EPOCHS = int(os.getenv("RAWSEQ_WF_EPOCHS", os.getenv("RAWSEQ_EPOCHS", os.getenv("RAWSEQ_IO_DISCOVERY_EPOCHS", "35"))))
DECISION_HORIZON_SECONDS = int(
    os.getenv("RAWSEQ_WF_DECISION_HORIZON_SECONDS", os.getenv("RAWSEQ_DECISION_HORIZON_SECONDS", "30"))
)
DECISION_THRESHOLD_BPS = float(
    os.getenv("RAWSEQ_WF_DECISION_THRESHOLD_BPS", os.getenv("RAWSEQ_DECISION_THRESHOLD_BPS", "0.0"))
)
FITNESS_POLICY = os.getenv("RAWSEQ_WF_FITNESS_POLICY", os.getenv("RAWSEQ_FITNESS_POLICY", "direct_gt"))
FITNESS_THRESHOLD_BPS = float(
    os.getenv("RAWSEQ_WF_FITNESS_THRESHOLD_BPS", os.getenv("RAWSEQ_FITNESS_THRESHOLD_BPS", "0.0"))
)
MIN_FITNESS_TRADES = int(
    os.getenv("RAWSEQ_WF_MIN_FITNESS_TRADES", os.getenv("RAWSEQ_MIN_FITNESS_TRADES", "100"))
)
MIN_MEAN_RMSE_IMPROVEMENT_FRACTION = float(
    os.getenv("RAWSEQ_DISCOVERY_MIN_MEAN_RMSE_IMPROVEMENT_FRACTION", "0.0")
)
DRY_RUN = os.getenv("RAWSEQ_WF_DRY_RUN", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
RUN_ID = os.getenv("RAWSEQ_WF_RUN_ID", "").strip()
IS_WINDOWS = os.name == "nt"
SHORT_PATHS = os.getenv("RAWSEQ_WF_SHORT_PATHS", "true" if IS_WINDOWS else "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
WINDOWS_SAFE_PATH_LIMIT = 220
MAX_PATH_COMPONENT_LEN = 64
OUTPUT_LABEL_TOKENS = {
    "future_return_path": "frp",
    "future_high_from_now_bps_path": "fhigh",
    "future_low_from_now_bps_path": "flow",
    "future_range_envelope_path": "fenv",
    "barrier_hit_levels": "barrier",
    "tp_before_stop_by_rung": "tps",
}
INPUT_FEATURE_TOKENS = {
    "return": "ret",
    "signed_bucket_return_bps": "sret",
    "bucket_return": "ret",
    "signed_return": "sret",
    "ma_distance": "mad",
    "ma_slope": "mas",
    "rolling_volatility_bps": "vol",
    "rolling_range_bps": "rng",
    "distance_to_recent_high_bps": "dh",
    "distance_to_recent_low_bps": "dl",
}

MODEL_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[^\r\n\"']*models[\\/]+candidates[^\r\n\"']*model\.json)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    index: int
    start_row: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class ContractSpec:
    input_feature: str
    ma_window: str
    feature_window: str
    hidden: str
    seed: str
    input_stride: str
    output_stride: str
    output_label: str


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_hiddens(text: str) -> list[str]:
    values = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if len(parts) != 2:
            raise SystemExit(f"RAWSEQ_WF_HIDDENS entry must look like 2,2; got {item}")
        values.append(",".join(parts))
    return values


def slugify(value: Any, default: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value).strip()).strip("_")
    return text or default


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def limit_component(value: Any, default: str = "item") -> str:
    slug = slugify(value, default)
    if len(slug) <= MAX_PATH_COMPONENT_LEN:
        return slug
    digest = stable_hash(slug)
    keep = max(1, MAX_PATH_COMPONENT_LEN - len(digest) - 1)
    return f"{slug[:keep].rstrip('_')}_{digest}"


def path_length(path: Path) -> int:
    return len(str(path))


def offending_component_length(path: Path) -> int:
    return max((len(part) for part in path.parts), default=0)


def safe_mkdir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        payload = {
            "error": "mkdir_failed",
            "attempted_path": str(path),
            "attempted_path_length": path_length(path),
            "offending_component_length": offending_component_length(path),
            "exception": str(exc),
        }
        raise SystemExit(json.dumps(payload, indent=2, sort_keys=True)) from exc


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".t_{uuid.uuid4().hex[:8]}"
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception as exc:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        payload = {
            "error": "atomic_write_csv_failed",
            "target_path": str(path),
            "target_path_length": path_length(path),
            "temporary_path": str(tmp),
            "temporary_path_length": path_length(tmp),
            "parent_exists": path.parent.exists(),
            "original_exception": str(exc),
        }
        raise RuntimeError(json.dumps(payload, indent=2, sort_keys=True)) from exc


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".t_{uuid.uuid4().hex[:8]}"
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        detail = {
            "error": "atomic_write_json_failed",
            "target_path": str(path),
            "target_path_length": path_length(path),
            "temporary_path": str(tmp),
            "temporary_path_length": path_length(tmp),
            "parent_exists": path.parent.exists(),
            "original_exception": str(exc),
        }
        raise RuntimeError(json.dumps(detail, indent=2, sort_keys=True)) from exc


def load_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"RAWSEQ_WF_SOURCE_PATH does not exist: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise SystemExit(f"RAWSEQ_WF_SOURCE_PATH has no rows: {path}")
    if "timestamp" in frame.columns:
        frame["_wf_timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame = frame.sort_values("_wf_timestamp", kind="mergesort").drop(columns=["_wf_timestamp"])
    return frame.reset_index(drop=True)


def timestamp_bounds(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty:
        return "", ""
    if "time" in frame.columns:
        return safe_str(frame["time"].iloc[0]), safe_str(frame["time"].iloc[-1])
    if "timestamp" in frame.columns:
        start = pd.to_datetime(frame["timestamp"].iloc[0], unit="ms", utc=True, errors="coerce")
        end = pd.to_datetime(frame["timestamp"].iloc[-1], unit="ms", utc=True, errors="coerce")
        start_text = "" if pd.isna(start) else start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_text = "" if pd.isna(end) else end.strftime("%Y-%m-%dT%H:%M:%SZ")
        return start_text, end_text
    return "", ""


def build_windows(total_rows: int) -> list[WindowSpec]:
    window_rows = TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS
    if window_rows <= 0:
        raise SystemExit("Train/validation/test row counts must sum to a positive number.")
    if STEP_ROWS <= 0:
        raise SystemExit("RAWSEQ_WF_STEP_ROWS must be positive.")

    windows: list[WindowSpec] = []
    start = 0
    index = 0
    while start + window_rows <= total_rows and len(windows) < MAX_WINDOWS:
        train_start = start
        train_end = train_start + TRAIN_ROWS
        validation_start = train_end
        validation_end = validation_start + VALIDATION_ROWS
        test_start = validation_end
        test_end = test_start + TEST_ROWS
        windows.append(
            WindowSpec(
                window_id=f"window_{index:03d}",
                index=index,
                start_row=start,
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        start += STEP_ROWS
        index += 1
    return windows


def build_contracts() -> list[ContractSpec]:
    features = parse_csv_list(INPUT_FEATURES)
    ma_windows = parse_csv_list(MA_WINDOWS)
    configured_feature_windows = parse_csv_list(FEATURE_WINDOWS)
    hiddens = parse_hiddens(HIDDENS)
    seeds = parse_csv_list(SEEDS)
    input_strides = parse_csv_list(INPUT_STRIDES)
    output_strides = parse_csv_list(OUTPUT_STRIDES)
    output_labels = parse_csv_list(OUTPUT_LABELS)
    contracts: list[ContractSpec] = []
    seen: set[tuple[str, str, str, str, str, str, str, str]] = set()
    for feature in features:
        feature = feature.strip().lower()
        if feature in {"ma_distance", "ma_slope"}:
            window_pairs = [(window, window) for window in ma_windows]
        elif feature in {"rolling_range_bps", "rolling_volatility_bps", "distance_to_recent_high_bps", "distance_to_recent_low_bps"}:
            window_pairs = [("", window) for window in configured_feature_windows]
        else:
            window_pairs = [("", "")]
        for ma_window, feature_window in window_pairs:
            for hidden in hiddens:
                for seed in seeds:
                    for input_stride in input_strides:
                        for output_stride in output_strides:
                            for output_label in output_labels:
                                output_label = output_label.strip().lower()
                                key = (feature, ma_window, feature_window, hidden, seed, input_stride, output_stride, output_label)
                                if key in seen:
                                    continue
                                seen.add(key)
                                contracts.append(
                                    ContractSpec(
                                        feature,
                                        ma_window,
                                        feature_window,
                                        hidden,
                                        seed,
                                        str(int(float(input_stride))),
                                        str(int(float(output_stride))),
                                        output_label,
                                    )
                                )
    return contracts


def contract_slug(contract: ContractSpec) -> str:
    ma = f"ma{contract.ma_window}" if contract.ma_window else (f"fw{contract.feature_window}" if contract.feature_window else "maNA")
    hidden = "h" + contract.hidden.replace(",", "x")
    stride = f"is{contract.input_stride}_os{contract.output_stride}"
    return slugify(f"{contract.input_feature}_{ma}_{hidden}_{stride}_{contract.output_label}_seed{contract.seed}", "contract")


def full_contract_dict(contract: ContractSpec) -> dict[str, Any]:
    return {
        "symbol": SYMBOL,
        "venue": PRIMARY_VENUE,
        "source_path": str(SOURCE_PATH),
        "bucket_seconds": BUCKET_SECONDS,
        "seq_len": SEQ_LEN,
        "input_feature": contract.input_feature,
        "ma_window": contract.ma_window,
        "feature_window": contract.feature_window,
        "hidden": contract.hidden,
        "seed": contract.seed,
        "input_stride": contract.input_stride,
        "output_stride": contract.output_stride,
        "output_label": contract.output_label,
        "population": POPULATION,
        "generations": GENERATIONS,
        "epochs": EPOCHS,
    }


def filesystem_contract_slug(contract: ContractSpec) -> str:
    feature = INPUT_FEATURE_TOKENS.get(contract.input_feature, slugify(contract.input_feature, "feat")[:8])
    ma = f"ma{contract.ma_window}" if contract.ma_window else (f"fw{contract.feature_window}" if contract.feature_window else "maNA")
    hidden = "h" + contract.hidden.replace(",", "x")
    label = OUTPUT_LABEL_TOKENS.get(contract.output_label, slugify(contract.output_label, "out")[:8])
    digest = stable_hash(full_contract_dict(contract))
    return limit_component(
        f"c_{feature}_{ma}_{hidden}_i{contract.input_stride}_o{contract.output_stride}_{label}_s{contract.seed}_{digest}",
        "contract",
    )


def archive_path_info(parent_dir: Path, contract: ContractSpec) -> dict[str, Any]:
    full_slug = contract_slug(contract)
    compact_slug = filesystem_contract_slug(contract)
    digest = stable_hash(full_contract_dict(contract))
    short_archive_slug = limit_component(f"s{contract.seed}_{digest}", "contract")
    preferred_component = short_archive_slug if SHORT_PATHS else limit_component(full_slug, "contract")
    candidate = parent_dir / preferred_component
    shortened = preferred_component != full_slug
    if IS_WINDOWS and path_length(candidate) > WINDOWS_SAFE_PATH_LIMIT:
        label = OUTPUT_LABEL_TOKENS.get(contract.output_label, "out")
        preferred_component = limit_component(f"c_{label}_{digest}", "contract")
        candidate = parent_dir / preferred_component
        shortened = True
    return {
        "path": candidate,
        "full_contract_slug": full_slug,
        "filesystem_contract_slug": preferred_component,
        "filesystem_path_shortened": bool(shortened),
        "projected_path_length": path_length(candidate),
        "offending_component_length": offending_component_length(candidate),
    }


def projected_path_metrics(archive_dir: Path) -> dict[str, Any]:
    final_paths = [
        archive_dir,
        archive_dir / "contract.json",
        archive_dir / "model_contract.json",
        archive_dir / "selected_candidate_summary.json",
        archive_dir / "selected_candidate_summary.csv",
        archive_dir.parent.parent / "candidates.csv",
        archive_dir / "rows.csv",
        archive_dir / "annotated.csv",
        archive_dir / "evaluation.csv",
        archive_dir / "history.csv",
        archive_dir / "label_metric_summary.csv",
        archive_dir / "label_shape_audit.csv",
        archive_dir / "trainer_artifacts",
        archive_dir / "trainer_artifacts" / "rawseq_rows.csv",
        archive_dir / "trainer_artifacts" / "rawseq_annotated.csv",
        archive_dir / "trainer_artifacts" / "rawseq_evaluation.csv",
        archive_dir / "trainer_artifacts" / "rawseq_history.csv",
        archive_dir / "trainer_artifacts" / "rawseq_label_metric_summary.csv",
        archive_dir / "trainer_artifacts" / "rawseq_label_shape_audit.csv",
    ]
    temp_paths = [path.parent / ".t_12345678" for path in final_paths if path.suffix in {".json", ".csv"}]
    longest_artifact = max(final_paths, key=path_length)
    longest_temp = max(temp_paths, key=path_length)
    longest = max(final_paths + temp_paths, key=path_length)
    return {
        "projected_candidate_dir_length": path_length(archive_dir),
        "projected_longest_artifact_path_length": path_length(longest_artifact),
        "projected_longest_temp_path_length": path_length(longest_temp),
        "projected_longest_path": str(longest),
        "windows_path_guard_pass": (not IS_WINDOWS) or path_length(longest_temp) < 240,
    }


def compact_run_id(run_id: str) -> str:
    component = limit_component(run_id, "rawseq_wf")
    candidate = OUTPUT_ROOT / component
    if IS_WINDOWS and path_length(candidate) > WINDOWS_SAFE_PATH_LIMIT:
        component = limit_component(f"wf_{now_stamp()}_{stable_hash(run_id)}", "rawseq_wf")
    return component


def write_window_source(source: pd.DataFrame, window: WindowSpec, path: Path) -> pd.DataFrame:
    slice_frame = source.iloc[window.train_start : window.test_end].copy()
    slice_frame["rawseq_wf_split"] = "train"
    rel_validation_start = window.validation_start - window.train_start
    rel_test_start = window.test_start - window.train_start
    slice_frame.loc[slice_frame.index[rel_validation_start:rel_test_start], "rawseq_wf_split"] = "validation"
    slice_frame.loc[slice_frame.index[rel_test_start:], "rawseq_wf_split"] = "test"
    path.parent.mkdir(parents=True, exist_ok=True)
    slice_frame.to_csv(path, index=False)
    return slice_frame


def parse_candidate_model_path(text: str) -> Path | None:
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
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


def matrix_shape(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    rows = len(value)
    if rows == 0:
        return "0x0"
    if not isinstance(value[0], list):
        return f"{rows}"
    return f"{rows}x{len(value[0])}"


def payload_contract(model_path: Path | None, requested: ContractSpec) -> dict[str, Any]:
    requested_output_dim = "120" if requested.output_label == "future_range_envelope_path" else "60"
    requested_orientation = "side_relative" if requested.output_label == "future_return_path" else "market_relative"
    base = {
        "requested_input_feature": requested.input_feature,
        "requested_ma_window": requested.ma_window,
        "requested_feature_window": requested.feature_window,
        "requested_hidden": requested.hidden,
        "requested_seed": requested.seed,
        "requested_input_stride": requested.input_stride,
        "requested_output_stride": requested.output_stride,
        "requested_output_label": requested.output_label,
        "model_path": str(model_path) if model_path else "",
        "output_label": requested.output_label,
        "output_dim": requested_output_dim,
        "output_orientation": requested_orientation,
        "payload_input_feature": "",
        "payload_ma_window": "",
        "payload_feature_window": "",
        "payload_hidden": "",
        "payload_seq_len": "",
        "payload_bucket_seconds": "",
        "payload_input_stride": "",
        "payload_output_stride": "",
        "payload_output_label": "",
        "payload_output_orientation": "",
        "payload_output_dim": "",
        "payload_source_path_basename": "",
        "payload_seed": "",
        "payload_created_at": "",
        "payload_w1_shape": "",
        "payload_w2_shape": "",
        "payload_w3_shape": "",
        "best_validation_fitness": math.nan,
    }
    if model_path is None or not model_path.exists():
        return base
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception:
        return base
    if not isinstance(payload, dict):
        return base
    arch = payload.get("architecture") if isinstance(payload.get("architecture"), dict) else {}
    weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else {}
    pop = payload.get("population_settings") if isinstance(payload.get("population_settings"), dict) else {}
    hidden = ""
    if arch.get("hidden_1") is not None and arch.get("hidden_2") is not None:
        hidden = f"{safe_int(arch.get('hidden_1'))},{safe_int(arch.get('hidden_2'))}"
    source_path = safe_str(payload.get("source_path"))
    base.update(
        {
            "payload_input_feature": safe_str(payload.get("input_feature")),
            "payload_ma_window": safe_str(payload.get("ma_window")),
            "payload_feature_window": safe_str(payload.get("feature_window") or payload.get("resolved_feature_window")),
            "payload_hidden": hidden,
            "payload_seq_len": safe_str(payload.get("seq_len") or arch.get("input_dim")),
            "payload_bucket_seconds": safe_str(payload.get("bucket_seconds")),
            "payload_input_stride": safe_str(payload.get("input_stride") or payload.get("rawseq_input_stride") or "1"),
            "payload_output_stride": safe_str(payload.get("output_stride") or payload.get("rawseq_output_stride") or "1"),
            "payload_output_label": safe_str(payload.get("output_label") or payload.get("output_target")),
            "payload_output_orientation": safe_str(payload.get("output_orientation")),
            "output_orientation": safe_str(payload.get("output_orientation")),
            "payload_output_dim": safe_str(payload.get("output_dim") or arch.get("output_dim")),
            "output_label": safe_str(payload.get("output_label") or payload.get("output_target")),
            "output_dim": safe_str(payload.get("output_dim") or arch.get("output_dim")),
            "payload_source_path_basename": Path(source_path.replace("\\", "/")).name if source_path else "",
            "payload_seed": safe_str(pop.get("seed") or payload.get("seed")),
            "payload_created_at": safe_str(payload.get("created_at")),
            "payload_w1_shape": matrix_shape(weights.get("W1")),
            "payload_w2_shape": matrix_shape(weights.get("W2")),
            "payload_w3_shape": matrix_shape(weights.get("W3")),
            "best_validation_fitness": safe_float(payload.get("best_validation_fitness")),
            "fitness_family": safe_str(payload.get("fitness_family")),
            "primary_fitness_metric": safe_str(payload.get("primary_fitness_metric")),
            "validation_path_mae": safe_float(payload.get("validation_path_mae")),
            "validation_path_rmse": safe_float(payload.get("validation_path_rmse")),
            "terminal_correlation": safe_float(payload.get("terminal_correlation")),
            "monotonic_violation_fraction": safe_float(payload.get("monotonic_violation_fraction")),
            "envelope_order_violation_fraction": safe_float(payload.get("envelope_order_violation_fraction")),
            "resolved_population": safe_str(payload.get("resolved_population") or pop.get("population")),
            "resolved_generations": safe_str(payload.get("resolved_generations") or pop.get("generations")),
            "resolved_epochs": safe_str(payload.get("resolved_epochs") or pop.get("epochs_per_generation")),
            "resolved_seed": safe_str(payload.get("resolved_seed") or pop.get("seed")),
            "resolved_feature_window": safe_str(payload.get("resolved_feature_window") or payload.get("feature_window")),
            "resolved_output_label": safe_str(payload.get("resolved_output_label") or payload.get("output_label")),
            "resolved_output_dim": safe_str(payload.get("resolved_output_dim") or payload.get("output_dim")),
            "generations_requested": safe_str(pop.get("generations_requested") or pop.get("generations")),
            "generations_completed": safe_str(pop.get("generations_completed")),
            "early_stop_reason": safe_str(pop.get("early_stop_reason")),
            "unique_population_fingerprints_per_generation": json.dumps(pop.get("unique_population_fingerprints_per_generation", [])),
        }
    )
    return base


def run_rawseq(
    window: WindowSpec,
    contract: ContractSpec,
    source_slice_path: Path,
    archive_dir: Path,
) -> subprocess.CompletedProcess[str] | None:
    if DRY_RUN:
        return None

    train_frac = TRAIN_ROWS / max(TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS, 1)
    val_frac = VALIDATION_ROWS / max(TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS, 1)
    env = os.environ.copy()
    env.update(
        {
            "SYMBOL": SYMBOL,
            "PRIMARY_VENUE": PRIMARY_VENUE,
            "RAWSEQ_SOURCE_PATH": str(source_slice_path),
            "RAWSEQ_BUCKET_SECONDS": str(BUCKET_SECONDS),
            "RAWSEQ_LEN": str(SEQ_LEN),
            "RAWSEQ_INPUT_STRIDE": contract.input_stride,
            "RAWSEQ_OUTPUT_STRIDE": contract.output_stride,
            "RAWSEQ_OUTPUT_LABEL": contract.output_label,
            "RAWSEQ_INPUT_FEATURE": contract.input_feature,
            "RAWSEQ_HIDDEN": contract.hidden,
            "RAWSEQ_SEED": contract.seed,
            "RAWSEQ_TRAIN_FRAC": f"{train_frac:.12g}",
            "RAWSEQ_VAL_FRAC": f"{val_frac:.12g}",
            "RAWSEQ_POPULATION": str(POPULATION),
            "RAWSEQ_GENERATIONS": str(GENERATIONS),
            "RAWSEQ_EPOCHS": str(EPOCHS),
            "RAWSEQ_WF_POPULATION": str(POPULATION),
            "RAWSEQ_WF_GENERATIONS": str(GENERATIONS),
            "RAWSEQ_WF_EPOCHS": str(EPOCHS),
            "RAWSEQ_DECISION_HORIZON_SECONDS": str(DECISION_HORIZON_SECONDS),
            "RAWSEQ_DECISION_THRESHOLD_BPS": str(DECISION_THRESHOLD_BPS),
            "RAWSEQ_FITNESS_POLICY": FITNESS_POLICY,
            "RAWSEQ_FITNESS_THRESHOLD_BPS": str(FITNESS_THRESHOLD_BPS),
            "RAWSEQ_MIN_FITNESS_TRADES": str(MIN_FITNESS_TRADES),
            "RAWSEQ_ARTIFACT_OUTPUT_DIR": str(archive_dir / "trainer_artifacts"),
            "RAWSEQ_ARTIFACT_PREFIX": "rawseq",
            "RAWSEQ_INCLUDE_WINDOW_GUIDE": "false",
            "PROMOTE_BEST": "false",
            "PRICE_TINY_AUTO_REGISTER_CHALLENGERS": "false",
            "TRAIN_PRICE_TINY_MODEL": "false",
        }
    )
    if contract.ma_window:
        env["RAWSEQ_MA_WINDOW"] = contract.ma_window
    if contract.feature_window:
        env["RAWSEQ_FEATURE_WINDOW"] = contract.feature_window

    completed = subprocess.run(
        [sys.executable, str(RAWSEQ_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    safe_mkdir(archive_dir)
    (archive_dir / "run.log").write_text(completed.stdout or "", encoding="utf-8")
    return completed


def archive_outputs(archive_dir: Path, completed: subprocess.CompletedProcess[str] | None) -> dict[str, str]:
    artifact_dir = archive_dir / "trainer_artifacts"
    realtime_dir = artifact_dir if artifact_dir.exists() else PROJECT_ROOT / "data" / "realtime" / PRIMARY_VENUE
    paths = {
        "rows_path": realtime_dir / "rawseq_rows.csv",
        "annotated_path": realtime_dir / "rawseq_annotated.csv",
        "evaluation_path": realtime_dir / "rawseq_evaluation.csv",
        "history_path": realtime_dir / "rawseq_history.csv",
        "label_metric_summary_path": realtime_dir / "rawseq_label_metric_summary.csv",
        "label_shape_audit_path": realtime_dir / "rawseq_label_shape_audit.csv",
        "feature_audit_path": realtime_dir / "rawseq_feature_audit.csv",
    }
    archived: dict[str, str] = {}
    if completed is None:
        return archived
    if completed.returncode != 0:
        archived["model_path"] = ""
        archived["archived_model_path"] = ""
        return archived
    for key, source in paths.items():
        if not source.exists():
            archived[key] = ""
            continue
        short_name = key.replace("_path", "")
        target = archive_dir / f"{short_name}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        archived[key] = str(target)
    candidate_model = parse_candidate_model_path(completed.stdout or "")
    if candidate_model is not None and candidate_model.exists():
        shutil.copy2(candidate_model, archive_dir / "model.json")
        archived["model_path"] = str(candidate_model)
        archived["archived_model_path"] = str(archive_dir / "model.json")
    else:
        archived["model_path"] = ""
        archived["archived_model_path"] = ""
    return archived


def best_test_row(evaluation_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not evaluation_path.exists():
        return {}, []
    try:
        frame = pd.read_csv(evaluation_path, low_memory=False)
    except Exception:
        return {}, []
    if frame.empty:
        return {}, []
    split_text = frame["split"].astype(str).str.lower() if "split" in frame.columns else pd.Series("", index=frame.index)
    test = frame[split_text.str.contains("test", na=False)].copy()
    if test.empty:
        return {}, []
    for column in [
        "rows",
        "avg_return_bps",
        "cumulative_return_bps",
        "win_rate",
        "max_dip_bps",
        "threshold_bps",
        "path_rmse",
        "path_mae",
        "high_path_rmse",
        "low_path_rmse",
        "terminal_high_correlation",
        "terminal_low_correlation",
        "monotonic_violation_fraction",
        "high_monotonic_violation_fraction",
        "low_monotonic_violation_fraction",
        "envelope_order_violation_fraction",
        "combined_path_rmse",
        "predicted_range_mae",
        "model_vs_mean_rmse_improvement_fraction",
        "model_vs_median_rmse_improvement_fraction",
        "model_vs_zero_rmse_improvement_fraction",
        "barrier_up_5bps_accuracy",
        "barrier_down_5bps_accuracy",
    ]:
        if column in test.columns:
            test[column] = pd.to_numeric(test[column], errors="coerce")
    output_label = safe_str(test["output_label"].dropna().iloc[0]) if "output_label" in test.columns and test["output_label"].notna().any() else "future_return_path"
    if output_label and output_label != "future_return_path":
        metric_rows = test[test["strategy"].astype(str).eq("label_metric_summary")].copy() if "strategy" in test.columns else test.copy()
        if metric_rows.empty:
            metric_rows = test.copy()
        baseline_rows = test[test["strategy"].astype(str).isin(["training_mean_path_baseline", "training_median_path_baseline"])].copy() if "strategy" in test.columns else pd.DataFrame()
        metric_rows = apply_label_baseline_guard(metric_rows, baseline_rows)
        for column in [
            "unguarded_label_rank_score",
            "guarded_label_rank_score",
            "label_rank_score",
            "baseline_guard_pass",
            "baseline_guard_reason",
            "rmse_guard_pass",
            "rmse_guard_reason",
            "mae_baseline_diagnostic_pass",
            "model_combined_test_rmse",
            "mean_baseline_combined_test_rmse",
            "median_baseline_combined_test_rmse",
            "model_combined_test_mae",
            "median_baseline_combined_test_mae",
            "model_vs_mean_rmse_improvement_fraction",
            "model_vs_median_mae_improvement_fraction",
            "model_beats_mean_baseline",
            "model_beats_median_baseline",
            "model_beats_median_mae_baseline",
        ]:
            test.loc[metric_rows.index, column] = metric_rows[column]
        best_idx = metric_rows["guarded_label_rank_score"].idxmax()
    elif "cumulative_return_bps" in test.columns and test["cumulative_return_bps"].notna().any():
        best_idx = test["cumulative_return_bps"].idxmax()
    elif "avg_return_bps" in test.columns and test["avg_return_bps"].notna().any():
        best_idx = test["avg_return_bps"].idxmax()
    else:
        best_idx = test.index[0]
    records = test.to_dict(orient="records")
    return test.loc[best_idx].to_dict(), records


def combined_test_rmse(row: pd.Series | dict[str, Any]) -> float:
    combined = safe_float(row.get("combined_path_rmse"))
    if math.isfinite(combined):
        return combined
    high = safe_float(row.get("high_path_rmse"))
    low = safe_float(row.get("low_path_rmse"))
    path = safe_float(row.get("path_rmse"))
    if math.isfinite(high) and math.isfinite(low):
        return 0.5 * (high + low)
    if math.isfinite(high):
        return high
    if math.isfinite(low):
        return low
    return path


def combined_test_mae(row: pd.Series | dict[str, Any]) -> float:
    high = safe_float(row.get("high_path_mae"))
    low = safe_float(row.get("low_path_mae"))
    path = safe_float(row.get("path_mae"))
    if math.isfinite(high) and math.isfinite(low):
        return 0.5 * (high + low)
    if math.isfinite(high):
        return high
    if math.isfinite(low):
        return low
    return path


def apply_label_baseline_guard(metric_rows: pd.DataFrame, baseline_rows: pd.DataFrame) -> pd.DataFrame:
    guarded = metric_rows.copy()
    mean_rows = baseline_rows[baseline_rows["strategy"].astype(str).eq("training_mean_path_baseline")] if not baseline_rows.empty and "strategy" in baseline_rows.columns else pd.DataFrame()
    median_rows = baseline_rows[baseline_rows["strategy"].astype(str).eq("training_median_path_baseline")] if not baseline_rows.empty and "strategy" in baseline_rows.columns else pd.DataFrame()
    mean_rmse = combined_test_rmse(mean_rows.iloc[0]) if not mean_rows.empty else math.nan
    median_rmse = combined_test_rmse(median_rows.iloc[0]) if not median_rows.empty else math.nan
    median_mae = combined_test_mae(median_rows.iloc[0]) if not median_rows.empty else math.nan
    for idx, row in guarded.iterrows():
        model_rmse = combined_test_rmse(row)
        model_mae = combined_test_mae(row)
        unguarded_score = label_rank_score(row)
        mean_rmse_improvement = (mean_rmse - model_rmse) / mean_rmse if math.isfinite(model_rmse) and math.isfinite(mean_rmse) and mean_rmse > 1e-12 else math.nan
        median_mae_improvement = (median_mae - model_mae) / median_mae if math.isfinite(model_mae) and math.isfinite(median_mae) and median_mae > 1e-12 else math.nan
        rmse_guard_pass = bool(math.isfinite(mean_rmse_improvement) and mean_rmse_improvement >= MIN_MEAN_RMSE_IMPROVEMENT_FRACTION)
        median_mae_pass = bool(math.isfinite(median_mae_improvement) and median_mae_improvement > 0.0)
        if rmse_guard_pass:
            reason = f"mean_rmse_improvement_fraction>={MIN_MEAN_RMSE_IMPROVEMENT_FRACTION:g}"
        elif not math.isfinite(mean_rmse_improvement):
            reason = "mean_rmse_improvement_fraction_not_finite"
        else:
            reason = f"mean_rmse_improvement_fraction<{MIN_MEAN_RMSE_IMPROVEMENT_FRACTION:g}"
        guarded_score = unguarded_score if rmse_guard_pass else -1_000_000_000.0
        if not rmse_guard_pass and guarded_score > -999_999_999.0:
            raise AssertionError("baseline_guard_pass=false candidate cannot have guarded_label_rank_score > -999999999")
        guarded.loc[idx, "unguarded_label_rank_score"] = unguarded_score
        guarded.loc[idx, "guarded_label_rank_score"] = guarded_score
        guarded.loc[idx, "label_rank_score"] = guarded_score
        guarded.loc[idx, "baseline_guard_pass"] = rmse_guard_pass
        guarded.loc[idx, "baseline_guard_reason"] = reason
        guarded.loc[idx, "rmse_guard_pass"] = rmse_guard_pass
        guarded.loc[idx, "rmse_guard_reason"] = reason
        guarded.loc[idx, "mae_baseline_diagnostic_pass"] = median_mae_pass
        guarded.loc[idx, "model_combined_test_rmse"] = model_rmse
        guarded.loc[idx, "mean_baseline_combined_test_rmse"] = mean_rmse
        guarded.loc[idx, "median_baseline_combined_test_rmse"] = median_rmse
        guarded.loc[idx, "model_combined_test_mae"] = model_mae
        guarded.loc[idx, "median_baseline_combined_test_mae"] = median_mae
        guarded.loc[idx, "model_vs_mean_rmse_improvement_fraction"] = mean_rmse_improvement
        guarded.loc[idx, "model_vs_median_mae_improvement_fraction"] = median_mae_improvement
        guarded.loc[idx, "model_beats_mean_baseline"] = rmse_guard_pass
        guarded.loc[idx, "model_beats_median_baseline"] = median_mae_pass
        guarded.loc[idx, "model_beats_median_mae_baseline"] = median_mae_pass
    return guarded


def label_rank_score(row: pd.Series) -> float:
    output_label = safe_str(row.get("output_label"))
    if output_label == "future_high_from_now_bps_path":
        rmse = safe_float(row.get("path_rmse"))
        mae = safe_float(row.get("path_mae"))
        corr = safe_float(row.get("terminal_high_correlation"))
        mono = safe_float(row.get("monotonic_violation_fraction"))
        barrier = safe_float(row.get("barrier_up_5bps_accuracy"))
        return -rmse - 0.25 * mae + corr + 0.5 * barrier - 10.0 * mono
    if output_label == "future_low_from_now_bps_path":
        rmse = safe_float(row.get("path_rmse"))
        mae = safe_float(row.get("path_mae"))
        corr = safe_float(row.get("terminal_low_correlation"))
        mono = safe_float(row.get("monotonic_violation_fraction"))
        barrier = safe_float(row.get("barrier_down_5bps_accuracy"))
        return -rmse - 0.25 * mae + corr + 0.5 * barrier - 10.0 * mono
    if output_label == "future_range_envelope_path":
        high_rmse = safe_float(row.get("high_path_rmse"))
        low_rmse = safe_float(row.get("low_path_rmse"))
        rmse = 0.5 * (high_rmse + low_rmse)
        corr = 0.5 * (safe_float(row.get("terminal_high_correlation")) + safe_float(row.get("terminal_low_correlation")))
        mono = 0.5 * (
            safe_float(row.get("high_monotonic_violation_fraction"))
            + safe_float(row.get("low_monotonic_violation_fraction"))
        )
        order = safe_float(row.get("envelope_order_violation_fraction"))
        barrier = 0.5 * (safe_float(row.get("barrier_up_5bps_accuracy")) + safe_float(row.get("barrier_down_5bps_accuracy")))
        return -rmse + corr + 0.5 * barrier - 10.0 * mono - 10.0 * order
    return math.nan


def candidate_row(
    run_id: str,
    window: WindowSpec,
    contract: ContractSpec,
    archive_dir: Path,
    completed: subprocess.CompletedProcess[str] | None,
    archived: dict[str, str],
    best_row: dict[str, Any],
    contract_payload: dict[str, Any],
    path_info: dict[str, Any],
    status_override: str = "",
) -> dict[str, Any]:
    status = status_override or ("DRY_RUN" if completed is None else "OK" if completed.returncode == 0 else "TRAIN_FAILED")
    trainer_dir = archive_dir / "trainer_artifacts"
    actual_paths = [p for p in archive_dir.rglob("*") if p.is_file()] if archive_dir.exists() else []
    actual_longest = max(actual_paths, key=path_length) if actual_paths else archive_dir
    actual_temp_paths = [p for p in actual_paths if p.name.startswith(".t_")]
    actual_longest_temp = max(actual_temp_paths, key=path_length) if actual_temp_paths else None
    projected = projected_path_metrics(archive_dir)
    return {
        "run_id": run_id,
        "window_id": window.window_id,
        "contract_slug": contract_slug(contract),
        "full_contract_slug": safe_str(path_info.get("full_contract_slug")),
        "filesystem_contract_slug": safe_str(path_info.get("filesystem_contract_slug")),
        "filesystem_path_shortened": bool(path_info.get("filesystem_path_shortened")),
        "projected_path_length": safe_int(path_info.get("projected_path_length")),
        **projected,
        "actual_longest_artifact_path_length": path_length(actual_longest),
        "actual_longest_artifact_path": str(actual_longest),
        "actual_longest_temp_path_length": path_length(actual_longest_temp) if actual_longest_temp is not None else 0,
        "actual_longest_temp_path": str(actual_longest_temp) if actual_longest_temp is not None else "",
        "trainer_artifacts_path": str(trainer_dir) if trainer_dir.exists() else "",
        "trainer_artifacts_retained": trainer_dir.exists(),
        "trainer_artifacts_cleanup_status": "retained" if trainer_dir.exists() else "not_created_or_cleaned",
        "status": status,
        "exit_code": "" if completed is None else completed.returncode,
        "symbol": SYMBOL,
        "venue": PRIMARY_VENUE,
        "input_feature": contract.input_feature,
        "ma_window": contract.ma_window,
        "feature_window": contract.feature_window,
        "hidden": contract.hidden,
        "seed": contract.seed,
        "bucket_seconds": BUCKET_SECONDS,
        "seq_len": SEQ_LEN,
        "input_stride": contract.input_stride,
        "output_stride": contract.output_stride,
        "output_label": contract.output_label,
        "output_orientation": safe_str(
            contract_payload.get("output_orientation")
            or contract_payload.get("payload_output_orientation")
        ),
        "requested_feature_window": contract.feature_window,
        "resolved_feature_window": safe_str(contract_payload.get("resolved_feature_window") or contract_payload.get("payload_feature_window") or contract.feature_window),
        "payload_feature_window": safe_str(contract_payload.get("payload_feature_window")),
        "payload_ma_window": safe_str(contract_payload.get("payload_ma_window")),
        "train_rows_requested": TRAIN_ROWS,
        "validation_rows_requested": VALIDATION_ROWS,
        "test_rows_requested": TEST_ROWS,
        "archive_dir": str(archive_dir),
        "model_path": archived.get("model_path", ""),
        "archived_model_path": archived.get("archived_model_path", ""),
        "best_test_strategy": safe_str(best_row.get("strategy")),
        "best_test_rows": safe_int(best_row.get("rows")),
        "best_test_avg_return_bps": safe_float(best_row.get("avg_return_bps")),
        "best_test_cumulative_return_bps": safe_float(best_row.get("cumulative_return_bps")),
        "best_test_win_rate": safe_float(best_row.get("win_rate")),
        "best_test_max_dip_bps": safe_float(best_row.get("max_dip_bps")),
        "best_test_threshold_bps": safe_float(best_row.get("threshold_bps")),
        "label_rank_score": safe_float(best_row.get("label_rank_score")),
        "unguarded_label_rank_score": safe_float(best_row.get("unguarded_label_rank_score")),
        "guarded_label_rank_score": safe_float(best_row.get("guarded_label_rank_score")),
        "baseline_guard_pass": safe_str(best_row.get("baseline_guard_pass")),
        "baseline_guard_reason": safe_str(best_row.get("baseline_guard_reason")),
        "rmse_guard_pass": safe_str(best_row.get("rmse_guard_pass")),
        "rmse_guard_reason": safe_str(best_row.get("rmse_guard_reason")),
        "mae_baseline_diagnostic_pass": safe_str(best_row.get("mae_baseline_diagnostic_pass")),
        "model_combined_test_rmse": safe_float(best_row.get("model_combined_test_rmse")),
        "mean_baseline_combined_test_rmse": safe_float(best_row.get("mean_baseline_combined_test_rmse")),
        "median_baseline_combined_test_rmse": safe_float(best_row.get("median_baseline_combined_test_rmse")),
        "model_combined_test_mae": safe_float(best_row.get("model_combined_test_mae")),
        "median_baseline_combined_test_mae": safe_float(best_row.get("median_baseline_combined_test_mae")),
        "model_vs_mean_rmse_improvement_fraction": safe_float(best_row.get("model_vs_mean_rmse_improvement_fraction")),
        "model_vs_median_mae_improvement_fraction": safe_float(best_row.get("model_vs_median_mae_improvement_fraction")),
        "test_path_mae": safe_float(best_row.get("path_mae")),
        "test_path_rmse": safe_float(best_row.get("path_rmse")),
        "test_combined_path_rmse": safe_float(best_row.get("combined_path_rmse")),
        "test_predicted_range_mae": safe_float(best_row.get("predicted_range_mae")),
        "test_high_path_rmse": safe_float(best_row.get("high_path_rmse")),
        "test_low_path_rmse": safe_float(best_row.get("low_path_rmse")),
        "test_model_vs_mean_rmse_improvement_fraction": safe_float(best_row.get("model_vs_mean_rmse_improvement_fraction")),
        "test_model_vs_median_rmse_improvement_fraction": safe_float(best_row.get("model_vs_median_rmse_improvement_fraction")),
        "test_model_vs_median_mae_improvement_fraction": safe_float(best_row.get("model_vs_median_mae_improvement_fraction")),
        "test_model_vs_zero_rmse_improvement_fraction": safe_float(best_row.get("model_vs_zero_rmse_improvement_fraction")),
        "test_model_beats_mean_baseline": safe_str(best_row.get("model_beats_mean_baseline")),
        "test_model_beats_median_baseline": safe_str(best_row.get("model_beats_median_baseline")),
        "test_model_beats_median_mae_baseline": safe_str(best_row.get("model_beats_median_mae_baseline")),
        "test_terminal_high_correlation": safe_float(best_row.get("terminal_high_correlation")),
        "test_terminal_low_correlation": safe_float(best_row.get("terminal_low_correlation")),
        "test_monotonic_violation_fraction": safe_float(best_row.get("monotonic_violation_fraction")),
        "test_high_monotonic_violation_fraction": safe_float(best_row.get("high_monotonic_violation_fraction")),
        "test_low_monotonic_violation_fraction": safe_float(best_row.get("low_monotonic_violation_fraction")),
        "test_envelope_order_violation_fraction": safe_float(best_row.get("envelope_order_violation_fraction")),
        "test_derived_tp_before_downside_risk_accuracy": safe_float(best_row.get("derived_tp_before_downside_risk_accuracy")),
        "test_derived_tp_before_downside_risk_precision": safe_float(best_row.get("derived_tp_before_downside_risk_precision")),
        "test_derived_tp_before_downside_risk_recall": safe_float(best_row.get("derived_tp_before_downside_risk_recall")),
        "test_derived_tp_before_downside_risk_coverage": safe_float(best_row.get("derived_tp_before_downside_risk_coverage")),
        "paper_only": True,
        "promotion": False,
        "champion_mutation": False,
        "orders": False,
        **contract_payload,
    }


def window_row(window: WindowSpec, slice_frame: pd.DataFrame, source_slice_path: Path) -> dict[str, Any]:
    train = slice_frame[slice_frame["rawseq_wf_split"].eq("train")]
    validation = slice_frame[slice_frame["rawseq_wf_split"].eq("validation")]
    test = slice_frame[slice_frame["rawseq_wf_split"].eq("test")]
    start_time, end_time = timestamp_bounds(slice_frame)
    test_start, test_end = timestamp_bounds(test)
    return {
        "window_id": window.window_id,
        "window_index": window.index,
        "source_path": str(SOURCE_PATH),
        "source_slice_path": str(source_slice_path),
        "start_row": window.start_row,
        "end_row": window.test_end,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "start_time": start_time,
        "end_time": end_time,
        "test_start_time": test_start,
        "test_end_time": test_end,
    }


def build_selected_by_window(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    valid = candidates[candidates["status"].eq("OK")].copy()
    if valid.empty:
        return pd.DataFrame()
    if "output_label" in valid.columns and valid["output_label"].astype(str).ne("future_return_path").any():
        guard = valid.get("baseline_guard_pass", pd.Series(False, index=valid.index)).astype(str).str.lower().isin(["true", "1", "yes", "y"])
        valid = valid[guard].copy()
        if valid.empty:
            return pd.DataFrame()
        valid["rank_score"] = pd.to_numeric(valid.get("guarded_label_rank_score", valid.get("label_rank_score")), errors="coerce")
    else:
        valid["rank_score"] = pd.to_numeric(valid["best_test_cumulative_return_bps"], errors="coerce")
        valid["rank_score"] = valid["rank_score"].fillna(pd.to_numeric(valid["best_test_avg_return_bps"], errors="coerce"))
    rows = []
    for window_id, group in valid.groupby("window_id", sort=True):
        ranked = group.sort_values(["rank_score", "best_test_rows"], ascending=[False, False])
        rows.append(ranked.iloc[0].to_dict())
    return pd.DataFrame(rows)


def build_leaderboard(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    valid = candidates[candidates["status"].eq("OK")].copy()
    if valid.empty:
        return pd.DataFrame()
    for column in [
        "best_test_rows",
        "best_test_avg_return_bps",
        "best_test_cumulative_return_bps",
        "label_rank_score",
        "unguarded_label_rank_score",
        "guarded_label_rank_score",
        "model_combined_test_rmse",
        "mean_baseline_combined_test_rmse",
        "median_baseline_combined_test_rmse",
        "test_path_rmse",
        "test_high_path_rmse",
        "test_low_path_rmse",
        "test_terminal_high_correlation",
        "test_terminal_low_correlation",
    ]:
        if column in valid.columns:
            valid[column] = pd.to_numeric(valid[column], errors="coerce")
        else:
            valid[column] = math.nan
    if "baseline_guard_pass" not in valid.columns:
        valid["baseline_guard_pass"] = ""
    label_mask = valid["output_label"].astype(str).ne("future_return_path") if "output_label" in valid.columns else pd.Series(False, index=valid.index)
    guard_mask = valid["baseline_guard_pass"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    valid = valid[(~label_mask) | guard_mask].copy()
    if valid.empty:
        return pd.DataFrame()
    group_cols = ["input_feature", "ma_window", "feature_window", "hidden", "input_stride", "output_stride", "output_label"]
    rows = []
    for keys, group in valid.groupby(group_cols, dropna=False, sort=True):
        is_return = safe_str(keys[6]) in {"", "future_return_path"}
        positive = group["best_test_cumulative_return_bps"] > 0.0 if is_return else group["guarded_label_rank_score"].notna()
        best_sort_column = "best_test_cumulative_return_bps" if is_return else "guarded_label_rank_score"
        rows.append(
            {
                "input_feature": keys[0],
                "ma_window": keys[1],
                "feature_window": keys[2],
                "hidden": keys[3],
                "input_stride": keys[4],
                "output_stride": keys[5],
                "output_label": keys[6],
                "runs": int(len(group)),
                "windows": int(group["window_id"].nunique()),
                "seeds": int(group["seed"].nunique()),
                "total_test_rows": int(group["best_test_rows"].fillna(0).sum()),
                "total_test_cumulative_return_bps": float(group["best_test_cumulative_return_bps"].fillna(0).sum()),
                "mean_test_avg_return_bps": float(group["best_test_avg_return_bps"].mean()),
                "median_test_avg_return_bps": float(group["best_test_avg_return_bps"].median()),
                "positive_test_windows": int(positive.sum()),
                "positive_test_window_fraction": float(positive.mean()) if len(group) else math.nan,
                "mean_label_rank_score": float(group["label_rank_score"].mean()) if not is_return else math.nan,
                "best_label_rank_score": float(group["guarded_label_rank_score"].max()) if not is_return else math.nan,
                "mean_unguarded_label_rank_score": float(group["unguarded_label_rank_score"].mean()) if not is_return else math.nan,
                "valid_rank_windows": int(group["baseline_guard_pass"].astype(str).str.lower().isin(["true", "1", "yes", "y"]).sum()) if not is_return else int(positive.sum()),
                "baseline_guard_pass_windows": int(group["baseline_guard_pass"].astype(str).str.lower().isin(["true", "1", "yes", "y"]).sum()) if not is_return else math.nan,
                "mean_model_combined_test_rmse": float(group["model_combined_test_rmse"].mean()) if not is_return else math.nan,
                "mean_mean_baseline_combined_test_rmse": float(group["mean_baseline_combined_test_rmse"].mean()) if not is_return else math.nan,
                "mean_median_baseline_combined_test_rmse": float(group["median_baseline_combined_test_rmse"].mean()) if not is_return else math.nan,
                "mean_test_path_rmse": float(group["test_path_rmse"].mean()) if "test_path_rmse" in group else math.nan,
                "mean_test_high_path_rmse": float(group["test_high_path_rmse"].mean()) if "test_high_path_rmse" in group else math.nan,
                "mean_test_low_path_rmse": float(group["test_low_path_rmse"].mean()) if "test_low_path_rmse" in group else math.nan,
                "mean_terminal_high_correlation": float(group["test_terminal_high_correlation"].mean()) if "test_terminal_high_correlation" in group else math.nan,
                "mean_terminal_low_correlation": float(group["test_terminal_low_correlation"].mean()) if "test_terminal_low_correlation" in group else math.nan,
                "ranking_mode": "return_policy" if is_return else "label_metrics",
                "best_run_id": safe_str(group.sort_values(best_sort_column, ascending=False).iloc[0].get("contract_slug")),
            }
        )
    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            [
                "ranking_mode",
                "positive_test_window_fraction",
                "best_label_rank_score",
                "total_test_cumulative_return_bps",
                "mean_test_avg_return_bps",
                "total_test_rows",
            ],
            ascending=[True, False, False, False, False, False],
        )
    return leaderboard


def render_summary(
    run_dir: Path,
    windows: pd.DataFrame,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> str:
    lines = [
        "Rawseq Recorded Walk-Forward Evolution",
        "",
        f"Run dir: {run_dir}",
        f"Symbol: {SYMBOL}",
        f"Venue: {PRIMARY_VENUE}",
        f"Source: {SOURCE_PATH}",
        f"Dry run: {DRY_RUN}",
        f"Windows: {len(windows)}",
        f"Candidate runs: {len(candidates)}",
        f"Population/generations/epochs: {POPULATION}/{GENERATIONS}/{EPOCHS}",
        "",
        "Leaderboard",
    ]
    if leaderboard.empty:
        lines.append("  none")
    else:
        for _, row in leaderboard.head(20).iterrows():
            if safe_str(row.get("ranking_mode")) == "label_metrics":
                lines.append(
                    f"  {row['input_feature']} ma={row['ma_window']} fw={row.get('feature_window', '')} hidden={row['hidden']} "
                    f"stride={row['input_stride']}/{row['output_stride']} label={row['output_label']} "
                    f"ranking=label_metrics windows={int(row['windows'])} seeds={int(row['seeds'])} "
                    f"best_label_score={safe_float(row.get('best_label_rank_score')):.6f} "
                    f"mean_rmse={safe_float(row.get('mean_test_path_rmse')):.6f}"
                )
            else:
                lines.append(
                    f"  {row['input_feature']} ma={row['ma_window']} fw={row.get('feature_window', '')} hidden={row['hidden']} "
                    f"stride={row['input_stride']}/{row['output_stride']} "
                    f"windows={int(row['windows'])} seeds={int(row['seeds'])} "
                    f"positive_fraction={safe_float(row['positive_test_window_fraction']):.4f} "
                    f"total_cum={safe_float(row['total_test_cumulative_return_bps']):.4f} "
                    f"mean_avg={safe_float(row['mean_test_avg_return_bps']):.6f}"
                )
    lines += ["", "Selected By Window"]
    if selected.empty:
        lines.append("  none")
    else:
        for _, row in selected.head(30).iterrows():
            if safe_str(row.get("output_label")) != "future_return_path":
                lines.append(
                    f"  {row['window_id']} {row['contract_slug']} "
                    f"label_score={safe_float(row.get('label_rank_score')):.6f} "
                    f"rmse={safe_float(row.get('test_path_rmse')):.6f} "
                    f"rows={safe_int(row['best_test_rows'])}"
                )
            else:
                lines.append(
                    f"  {row['window_id']} {row['contract_slug']} "
                    f"cum={safe_float(row['best_test_cumulative_return_bps']):.4f} "
                    f"avg={safe_float(row['best_test_avg_return_bps']):.6f} "
                    f"rows={safe_int(row['best_test_rows'])}"
                )
    lines += [
        "",
        "Safety: public/recorded data only. Paper-only. No promotion. No champion mutation. No orders.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if not RAWSEQ_SCRIPT.exists():
        raise SystemExit(f"Missing rawseq trainer: {RAWSEQ_SCRIPT}")
    source = load_source(SOURCE_PATH)
    windows = build_windows(len(source))
    contracts = build_contracts()
    if not windows:
        raise SystemExit(
            f"No walk-forward windows available: source_rows={len(source)} "
            f"needed={TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS}"
        )
    if not contracts:
        raise SystemExit("No walk-forward contracts configured.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = compact_run_id(RUN_ID or f"{SYMBOL.lower()}_{PRIMARY_VENUE}_rawseq_wf_{now_stamp()}")
    run_dir = OUTPUT_ROOT / run_id
    safe_mkdir(run_dir)

    window_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    test_score_rows: list[dict[str, Any]] = []

    for window in windows:
        window_dir = run_dir / window.window_id
        source_slice_path = window_dir / "source_slice.csv"
        slice_frame = write_window_source(source, window, source_slice_path)
        window_rows.append(window_row(window, slice_frame, source_slice_path))

        for contract in contracts:
            path_info = archive_path_info(window_dir, contract)
            archive_dir = path_info["path"]
            path_info.update(projected_path_metrics(archive_dir))
            safe_mkdir(archive_dir)
            completed = run_rawseq(window, contract, source_slice_path, archive_dir)
            archived = archive_outputs(archive_dir, completed)
            status_override = ""
            if completed is not None and completed.returncode != 0:
                status_override = "TRAIN_FAILED"
            if completed is None:
                (archive_dir / "run.log").write_text("DRY_RUN: rawseq evolution not executed.\n", encoding="utf-8")
            evaluation_path = Path(archived.get("evaluation_path", "")) if archived.get("evaluation_path") else archive_dir / "evaluation.csv"
            best_row, test_records = best_test_row(evaluation_path)
            model_path = Path(archived["model_path"]) if archived.get("model_path") else None
            contract_payload = payload_contract(model_path, contract)
            contract_payload.update(
                {
                    "window_id": window.window_id,
                    "symbol": SYMBOL,
                    "venue": PRIMARY_VENUE,
                    "contract": full_contract_dict(contract),
                    "contract_slug": contract_slug(contract),
                    "full_contract_slug": path_info["full_contract_slug"],
                    "filesystem_contract_slug": path_info["filesystem_contract_slug"],
                    "filesystem_path_shortened": path_info["filesystem_path_shortened"],
                    "projected_path_length": path_info["projected_path_length"],
                    "source_slice_path": str(source_slice_path),
                    "paper_only": True,
                    "promotion": False,
                    "champion_mutation": False,
                    "orders": False,
                }
            )
            try:
                if os.getenv("RAWSEQ_WF_SIMULATE_METADATA_WRITE_FAILED", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
                    raise RuntimeError("Simulated metadata write failure for discovery verification.")
                atomic_write_json(
                    {
                        **full_contract_dict(contract),
                        "full_contract_slug": path_info["full_contract_slug"],
                        "filesystem_contract_slug": path_info["filesystem_contract_slug"],
                        "filesystem_path_shortened": path_info["filesystem_path_shortened"],
                        "projected_path_length": path_info["projected_path_length"],
                        **projected_path_metrics(archive_dir),
                        "archive_dir": str(archive_dir),
                        "paper_only": True,
                        "promotion": False,
                        "champion_mutation": False,
                        "orders": False,
                    },
                    archive_dir / "contract.json",
                )
                atomic_write_json(contract_payload, archive_dir / "model_contract.json")
            except Exception as exc:
                status_override = "METADATA_WRITE_FAILED"
                (archive_dir / "metadata_write_error.txt").write_text(str(exc), encoding="utf-8")
            candidate = candidate_row(
                run_id,
                window,
                contract,
                archive_dir,
                completed,
                archived,
                best_row,
                contract_payload,
                path_info,
                status_override,
            )
            candidate_rows.append(candidate)
            try:
                atomic_write_json(candidate, archive_dir / "selected_candidate_summary.json")
                atomic_write_csv(pd.DataFrame([candidate]), archive_dir / "selected_candidate_summary.csv")
            except Exception as exc:
                candidate["status"] = "METADATA_WRITE_FAILED"
                (archive_dir / "metadata_write_error.txt").write_text(str(exc), encoding="utf-8")
            for record in test_records:
                record.update(
                    {
                        "run_id": run_id,
                        "window_id": window.window_id,
                        "contract_slug": contract_slug(contract),
                        "full_contract_slug": path_info["full_contract_slug"],
                        "filesystem_contract_slug": path_info["filesystem_contract_slug"],
                        "filesystem_path_shortened": path_info["filesystem_path_shortened"],
                        "projected_path_length": path_info["projected_path_length"],
                        "input_feature": contract.input_feature,
                        "ma_window": contract.ma_window,
                        "feature_window": contract.feature_window,
                        "hidden": contract.hidden,
                        "seed": contract.seed,
                        "input_stride": contract.input_stride,
                        "output_stride": contract.output_stride,
                        "archive_dir": str(archive_dir),
                    }
                )
                test_score_rows.append(record)

    windows_frame = pd.DataFrame(window_rows)
    candidates_frame = pd.DataFrame(candidate_rows)
    test_scores_frame = pd.DataFrame(test_score_rows)
    selected_frame = build_selected_by_window(candidates_frame)
    leaderboard_frame = build_leaderboard(candidates_frame)

    atomic_write_csv(windows_frame, run_dir / "windows.csv")
    atomic_write_csv(candidates_frame, run_dir / "candidates.csv")
    atomic_write_csv(candidates_frame, run_dir / "all_window_candidates.csv")
    atomic_write_csv(selected_frame, run_dir / "selected_by_window.csv")
    atomic_write_csv(test_scores_frame, run_dir / "test_scores.csv")
    atomic_write_csv(leaderboard_frame, run_dir / "contract_leaderboard.csv")
    summary = render_summary(run_dir, windows_frame, candidates_frame, selected_frame, leaderboard_frame)
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"windows.csv: {run_dir / 'windows.csv'}")
    print(f"candidates.csv: {run_dir / 'candidates.csv'}")
    print(f"all_window_candidates.csv: {run_dir / 'all_window_candidates.csv'}")
    print(f"selected_by_window.csv: {run_dir / 'selected_by_window.csv'}")
    print(f"test_scores.csv: {run_dir / 'test_scores.csv'}")
    print(f"contract_leaderboard.csv: {run_dir / 'contract_leaderboard.csv'}")
    print(f"summary.txt: {run_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
