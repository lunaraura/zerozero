#!/usr/bin/env python3
"""Run rawseq evolution over recorded Kraken walk-forward windows.

This is an orchestration wrapper around scripts/tiny_price_rawseq_path_v1.py.
It uses only recorded/public source files, archives each run, and never mutates
paper champions, promotes, or places orders.
"""

from __future__ import annotations

import json
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
HIDDENS = os.getenv("RAWSEQ_WF_HIDDENS", "2,2;3,3;4,4")
SEEDS = os.getenv("RAWSEQ_WF_SEEDS", "900,901,902")
INPUT_STRIDES = os.getenv("RAWSEQ_WF_INPUT_STRIDES", os.getenv("RAWSEQ_INPUT_STRIDE", "1"))
OUTPUT_STRIDES = os.getenv("RAWSEQ_WF_OUTPUT_STRIDES", os.getenv("RAWSEQ_OUTPUT_STRIDE", "1"))

TRAIN_ROWS = int(os.getenv("RAWSEQ_WF_TRAIN_ROWS", "60000"))
VALIDATION_ROWS = int(os.getenv("RAWSEQ_WF_VALIDATION_ROWS", "15000"))
TEST_ROWS = int(os.getenv("RAWSEQ_WF_TEST_ROWS", "15000"))
STEP_ROWS = int(os.getenv("RAWSEQ_WF_STEP_ROWS", "15000"))
MAX_WINDOWS = int(os.getenv("RAWSEQ_WF_MAX_WINDOWS", "10"))

BUCKET_SECONDS = int(os.getenv("RAWSEQ_WF_BUCKET_SECONDS", os.getenv("RAWSEQ_BUCKET_SECONDS", "10")))
SEQ_LEN = int(os.getenv("RAWSEQ_WF_SEQ_LEN", os.getenv("RAWSEQ_LEN", "60")))
POPULATION = int(os.getenv("RAWSEQ_WF_POPULATION", os.getenv("RAWSEQ_POPULATION", "5")))
GENERATIONS = int(os.getenv("RAWSEQ_WF_GENERATIONS", os.getenv("RAWSEQ_GENERATIONS", "3")))
EPOCHS = int(os.getenv("RAWSEQ_WF_EPOCHS", os.getenv("RAWSEQ_EPOCHS", "35")))
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
DRY_RUN = os.getenv("RAWSEQ_WF_DRY_RUN", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
RUN_ID = os.getenv("RAWSEQ_WF_RUN_ID", "").strip()

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
    hidden: str
    seed: str
    input_stride: str
    output_stride: str


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


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


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
    hiddens = parse_hiddens(HIDDENS)
    seeds = parse_csv_list(SEEDS)
    input_strides = parse_csv_list(INPUT_STRIDES)
    output_strides = parse_csv_list(OUTPUT_STRIDES)
    contracts: list[ContractSpec] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for feature in features:
        feature = feature.strip().lower()
        feature_windows = [""] if feature in {"return", "bucket_return", "signed_return"} else ma_windows
        for ma_window in feature_windows:
            for hidden in hiddens:
                for seed in seeds:
                    for input_stride in input_strides:
                        for output_stride in output_strides:
                            key = (feature, ma_window, hidden, seed, input_stride, output_stride)
                            if key in seen:
                                continue
                            seen.add(key)
                            contracts.append(
                                ContractSpec(
                                    feature,
                                    ma_window,
                                    hidden,
                                    seed,
                                    str(int(float(input_stride))),
                                    str(int(float(output_stride))),
                                )
                            )
    return contracts


def contract_slug(contract: ContractSpec) -> str:
    ma = f"ma{contract.ma_window}" if contract.ma_window else "maNA"
    hidden = "h" + contract.hidden.replace(",", "x")
    stride = f"is{contract.input_stride}_os{contract.output_stride}"
    return slugify(f"{contract.input_feature}_{ma}_{hidden}_{stride}_seed{contract.seed}", "contract")


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
    base = {
        "requested_input_feature": requested.input_feature,
        "requested_ma_window": requested.ma_window,
        "requested_hidden": requested.hidden,
        "requested_seed": requested.seed,
        "requested_input_stride": requested.input_stride,
        "requested_output_stride": requested.output_stride,
        "model_path": str(model_path) if model_path else "",
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
            "payload_hidden": hidden,
            "payload_seq_len": safe_str(payload.get("seq_len") or arch.get("input_dim")),
            "payload_bucket_seconds": safe_str(payload.get("bucket_seconds")),
            "payload_input_stride": safe_str(payload.get("input_stride") or payload.get("rawseq_input_stride") or "1"),
            "payload_output_stride": safe_str(payload.get("output_stride") or payload.get("rawseq_output_stride") or "1"),
            "payload_source_path_basename": Path(source_path.replace("\\", "/")).name if source_path else "",
            "payload_seed": safe_str(pop.get("seed") or payload.get("seed")),
            "payload_created_at": safe_str(payload.get("created_at")),
            "payload_w1_shape": matrix_shape(weights.get("W1")),
            "payload_w2_shape": matrix_shape(weights.get("W2")),
            "payload_w3_shape": matrix_shape(weights.get("W3")),
            "best_validation_fitness": safe_float(payload.get("best_validation_fitness")),
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
            "RAWSEQ_INPUT_FEATURE": contract.input_feature,
            "RAWSEQ_HIDDEN": contract.hidden,
            "RAWSEQ_SEED": contract.seed,
            "RAWSEQ_TRAIN_FRAC": f"{train_frac:.12g}",
            "RAWSEQ_VAL_FRAC": f"{val_frac:.12g}",
            "RAWSEQ_POPULATION": str(POPULATION),
            "RAWSEQ_GENERATIONS": str(GENERATIONS),
            "RAWSEQ_EPOCHS": str(EPOCHS),
            "RAWSEQ_DECISION_HORIZON_SECONDS": str(DECISION_HORIZON_SECONDS),
            "RAWSEQ_DECISION_THRESHOLD_BPS": str(DECISION_THRESHOLD_BPS),
            "RAWSEQ_FITNESS_POLICY": FITNESS_POLICY,
            "RAWSEQ_FITNESS_THRESHOLD_BPS": str(FITNESS_THRESHOLD_BPS),
            "RAWSEQ_MIN_FITNESS_TRADES": str(MIN_FITNESS_TRADES),
            "RAWSEQ_INCLUDE_WINDOW_GUIDE": "false",
            "PROMOTE_BEST": "false",
            "PRICE_TINY_AUTO_REGISTER_CHALLENGERS": "false",
            "TRAIN_PRICE_TINY_MODEL": "false",
        }
    )
    if contract.ma_window:
        env["RAWSEQ_MA_WINDOW"] = contract.ma_window

    completed = subprocess.run(
        [sys.executable, str(RAWSEQ_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "run.log").write_text(completed.stdout or "", encoding="utf-8")
    return completed


def archive_outputs(archive_dir: Path, completed: subprocess.CompletedProcess[str] | None) -> dict[str, str]:
    realtime_dir = PROJECT_ROOT / "data" / "realtime" / PRIMARY_VENUE
    paths = {
        "rows_path": realtime_dir / f"{SYMBOL}_tiny_price_rawseq_path_v1_rows.csv",
        "annotated_path": realtime_dir / f"{SYMBOL}_tiny_price_prediction_evaluation_rows_with_rawseq_path_v1.csv",
        "evaluation_path": realtime_dir / f"{SYMBOL}_tiny_price_rawseq_path_v1_shadow_evaluation.csv",
        "history_path": realtime_dir / f"{SYMBOL}_tiny_price_rawseq_path_v1_history.csv",
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
        target = archive_dir / source.name
        shutil.copy2(source, target)
        short_name = key.replace("_path", "")
        named_target = archive_dir / f"{short_name}.csv"
        if named_target != target:
            shutil.copy2(source, named_target)
        archived[key] = str(named_target)
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
    for column in ["rows", "avg_return_bps", "cumulative_return_bps", "win_rate", "max_dip_bps", "threshold_bps"]:
        if column in test.columns:
            test[column] = pd.to_numeric(test[column], errors="coerce")
    if "cumulative_return_bps" in test.columns and test["cumulative_return_bps"].notna().any():
        best_idx = test["cumulative_return_bps"].idxmax()
    elif "avg_return_bps" in test.columns and test["avg_return_bps"].notna().any():
        best_idx = test["avg_return_bps"].idxmax()
    else:
        best_idx = test.index[0]
    records = test.to_dict(orient="records")
    return test.loc[best_idx].to_dict(), records


def candidate_row(
    run_id: str,
    window: WindowSpec,
    contract: ContractSpec,
    archive_dir: Path,
    completed: subprocess.CompletedProcess[str] | None,
    archived: dict[str, str],
    best_row: dict[str, Any],
    contract_payload: dict[str, Any],
) -> dict[str, Any]:
    status = "DRY_RUN" if completed is None else "OK" if completed.returncode == 0 else "FAILED"
    return {
        "run_id": run_id,
        "window_id": window.window_id,
        "contract_slug": contract_slug(contract),
        "status": status,
        "exit_code": "" if completed is None else completed.returncode,
        "symbol": SYMBOL,
        "venue": PRIMARY_VENUE,
        "input_feature": contract.input_feature,
        "ma_window": contract.ma_window,
        "hidden": contract.hidden,
        "seed": contract.seed,
        "bucket_seconds": BUCKET_SECONDS,
        "seq_len": SEQ_LEN,
        "input_stride": contract.input_stride,
        "output_stride": contract.output_stride,
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
    if candidates.empty or "best_test_cumulative_return_bps" not in candidates.columns:
        return pd.DataFrame()
    valid = candidates[candidates["status"].eq("OK")].copy()
    if valid.empty:
        return pd.DataFrame()
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
    for column in ["best_test_rows", "best_test_avg_return_bps", "best_test_cumulative_return_bps"]:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    group_cols = ["input_feature", "ma_window", "hidden", "input_stride", "output_stride"]
    rows = []
    for keys, group in valid.groupby(group_cols, dropna=False, sort=True):
        positive = group["best_test_cumulative_return_bps"] > 0.0
        rows.append(
            {
                "input_feature": keys[0],
                "ma_window": keys[1],
                "hidden": keys[2],
                "input_stride": keys[3],
                "output_stride": keys[4],
                "runs": int(len(group)),
                "windows": int(group["window_id"].nunique()),
                "seeds": int(group["seed"].nunique()),
                "total_test_rows": int(group["best_test_rows"].fillna(0).sum()),
                "total_test_cumulative_return_bps": float(group["best_test_cumulative_return_bps"].fillna(0).sum()),
                "mean_test_avg_return_bps": float(group["best_test_avg_return_bps"].mean()),
                "median_test_avg_return_bps": float(group["best_test_avg_return_bps"].median()),
                "positive_test_windows": int(positive.sum()),
                "positive_test_window_fraction": float(positive.mean()) if len(group) else math.nan,
                "best_run_id": safe_str(group.sort_values("best_test_cumulative_return_bps", ascending=False).iloc[0].get("contract_slug")),
            }
        )
    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            [
                "positive_test_window_fraction",
                "total_test_cumulative_return_bps",
                "mean_test_avg_return_bps",
                "total_test_rows",
            ],
            ascending=[False, False, False, False],
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
            lines.append(
                f"  {row['input_feature']} ma={row['ma_window']} hidden={row['hidden']} "
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

    run_id = RUN_ID or f"{SYMBOL.lower()}_{PRIMARY_VENUE}_rawseq_wf_{now_stamp()}"
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    window_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    test_score_rows: list[dict[str, Any]] = []

    for window in windows:
        window_dir = run_dir / window.window_id
        source_slice_path = window_dir / "source_slice.csv"
        slice_frame = write_window_source(source, window, source_slice_path)
        window_rows.append(window_row(window, slice_frame, source_slice_path))

        for contract in contracts:
            archive_dir = window_dir / contract_slug(contract)
            archive_dir.mkdir(parents=True, exist_ok=True)
            completed = run_rawseq(window, contract, source_slice_path, archive_dir)
            archived = archive_outputs(archive_dir, completed)
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
                    "source_slice_path": str(source_slice_path),
                    "paper_only": True,
                    "promotion": False,
                    "champion_mutation": False,
                    "orders": False,
                }
            )
            atomic_write_json(contract_payload, archive_dir / "model_contract.json")
            candidate = candidate_row(
                run_id,
                window,
                contract,
                archive_dir,
                completed,
                archived,
                best_row,
                contract_payload,
            )
            candidate_rows.append(candidate)
            atomic_write_json(candidate, archive_dir / "selected_candidate_summary.json")
            pd.DataFrame([candidate]).to_csv(archive_dir / "selected_candidate_summary.csv", index=False)
            for record in test_records:
                record.update(
                    {
                        "run_id": run_id,
                        "window_id": window.window_id,
                        "contract_slug": contract_slug(contract),
                        "input_feature": contract.input_feature,
                        "ma_window": contract.ma_window,
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
    atomic_write_csv(selected_frame, run_dir / "selected_by_window.csv")
    atomic_write_csv(test_scores_frame, run_dir / "test_scores.csv")
    atomic_write_csv(leaderboard_frame, run_dir / "contract_leaderboard.csv")
    summary = render_summary(run_dir, windows_frame, candidates_frame, selected_frame, leaderboard_frame)
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"windows.csv: {run_dir / 'windows.csv'}")
    print(f"candidates.csv: {run_dir / 'candidates.csv'}")
    print(f"selected_by_window.csv: {run_dir / 'selected_by_window.csv'}")
    print(f"test_scores.csv: {run_dir / 'test_scores.csv'}")
    print(f"contract_leaderboard.csv: {run_dir / 'contract_leaderboard.csv'}")
    print(f"summary.txt: {run_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
