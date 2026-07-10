#!/usr/bin/env python3
"""Probe a rawseq candidate model using the model payload contract.

Read-only except for writing isolated probe reports/artifacts under
data/research/rawseq_candidate_shadow_probes by default.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, UTC
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAWSEQ_SCRIPT = PROJECT_ROOT / "scripts" / "tiny_price_rawseq_path_v1.py"

MODEL_PATH_ENV = os.getenv("RAWSEQ_PROBE_MODEL_PATH", "").strip()
SYMBOL_ENV = os.getenv("SYMBOL", "SOLUSDT").strip().upper() or "SOLUSDT"
VENUE_ENV = os.getenv("PRIMARY_VENUE", "kraken").strip().lower() or "kraken"
SOURCE_PATH_ENV = os.getenv("RAWSEQ_PROBE_SOURCE_PATH", "").strip()
POLICY = os.getenv("RAWSEQ_PROBE_POLICY", "inverse_gt").strip().lower()
THRESHOLD_BPS_LIST_ENV = os.getenv("RAWSEQ_PROBE_THRESHOLD_BPS_LIST", "0.0,0.1,0.2,0.3,0.5")
COST_BPS_LIST_ENV = os.getenv("RAWSEQ_PROBE_COST_BPS_LIST", "0,0.05,0.1,0.25")
TEST_FRAC = float(os.getenv("RAWSEQ_PROBE_TEST_FRAC", "0.20"))
OUTPUT_ROOT = Path(
    os.getenv(
        "RAWSEQ_PROBE_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_candidate_shadow_probes"),
    )
)
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

PRED_COLUMN = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COLUMN = "rawseq_path_actual_horizon_return_bps"
REQUIRED_ANNOTATED_COLUMNS = ["timestamp", "time", PRED_COLUMN, ACTUAL_COLUMN]
ROLLING_WINDOW_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0]


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_float_list(text: str, label: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError as exc:
            raise SystemExit(f"{label} contains a non-float value: {item}") from exc
    if not values:
        raise SystemExit(f"{label} did not contain any values.")
    return values


def normalize_int(value: Any) -> str:
    text = safe_str(value)
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def normalize_hidden(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(normalize_int(item) for item in value if safe_str(item))
    text = safe_str(value).replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    return ",".join(part.strip() for part in text.split(",") if part.strip()).replace(" ", "")


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


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except Exception:
        return math.nan


def int_contract(contract: dict[str, Any], key: str, default: int = 1) -> int:
    try:
        return int(float(safe_str(contract.get(key)) or default))
    except Exception:
        return default


def max_dip_bps(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype="float64")
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return math.nan
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def safe_slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:180] or "rawseq_candidate_probe"


def resolve_existing_path(path_text: str, label: str) -> Path:
    if not path_text:
        raise SystemExit(f"{label} is required.")
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    return path.resolve()


def load_model(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise SystemExit(f"Could not parse model.json: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"model.json root is not an object: {path}")
    return payload


def resolve_source_path(payload: dict[str, Any], symbol: str, venue: str) -> Path:
    source_text = SOURCE_PATH_ENV or safe_str(payload.get("source_path"))
    if source_text:
        path = Path(source_text).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return path.resolve()
        basename = Path(source_text.replace("\\", "/")).name
        fallback = PROJECT_ROOT / "data" / "realtime" / venue / basename
        if fallback.exists():
            return fallback.resolve()
        raise SystemExit(f"Source path does not exist: {path}")

    fallback = PROJECT_ROOT / "data" / "realtime" / venue / f"{symbol}_10s_flow.csv"
    if fallback.exists():
        return fallback.resolve()
    raise SystemExit("No RAWSEQ_PROBE_SOURCE_PATH and model.json has no source_path.")


def extract_contract(payload: dict[str, Any], model_path: Path) -> dict[str, Any]:
    arch = payload.get("architecture") if isinstance(payload.get("architecture"), dict) else {}
    weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else {}
    y_scaler = payload.get("y_scaler") if isinstance(payload.get("y_scaler"), dict) else {}
    pop = payload.get("population_settings") if isinstance(payload.get("population_settings"), dict) else {}

    w1_rows, w1_cols = matrix_shape(weights.get("W1"))
    w2_rows, w2_cols = matrix_shape(weights.get("W2"))
    w3_rows, w3_cols = matrix_shape(weights.get("W3"))

    symbol = safe_str(payload.get("symbol")).upper() or SYMBOL_ENV
    venue = (safe_str(payload.get("primary_venue") or payload.get("venue")).lower() or VENUE_ENV)
    bucket_seconds = normalize_int(payload.get("bucket_seconds"))
    seq_len = normalize_int(payload.get("seq_len") or arch.get("input_dim") or w1_rows)
    output_label = safe_str(payload.get("output_label") or payload.get("output_target") or "future_return_path").lower()
    if output_label == "future_signed_cumulative_return_path_bps":
        output_label = "future_return_path"
    output_dim = normalize_int(payload.get("output_dim") or arch.get("output_dim") or w3_cols)
    input_feature = safe_str(payload.get("input_feature")).lower()
    ma_window = normalize_int(payload.get("ma_window") or payload.get("rawseq_ma_window"))
    input_stride = normalize_int(payload.get("input_stride") or payload.get("rawseq_input_stride") or 1)
    output_stride = normalize_int(payload.get("output_stride") or payload.get("rawseq_output_stride") or 1)

    hidden_declared = ""
    if arch.get("hidden_1") is not None and arch.get("hidden_2") is not None:
        hidden_declared = normalize_hidden([arch.get("hidden_1"), arch.get("hidden_2")])
    hidden_inferred = ""
    if w1_cols is not None and w2_cols is not None:
        hidden_inferred = normalize_hidden([w1_cols, w2_cols])
    hidden = hidden_declared or hidden_inferred

    source_path = resolve_source_path(payload, symbol, venue)

    contract = {
        "model_path": str(model_path),
        "symbol": symbol,
        "venue": venue,
        "source_path": str(source_path),
        "source_path_basename": source_path.name,
        "bucket_seconds": bucket_seconds,
        "seq_len": seq_len,
        "output_label": output_label,
        "task_type": safe_str(payload.get("task_type") or "regression"),
        "output_dim": output_dim,
        "label_required_horizon_buckets": normalize_int(payload.get("label_required_horizon_buckets") or seq_len),
        "input_stride": input_stride,
        "output_stride": output_stride,
        "input_span_buckets": normalize_int(
            payload.get("input_span_buckets") or payload.get("rawseq_input_span_buckets")
        ),
        "output_span_buckets": normalize_int(
            payload.get("output_span_buckets") or payload.get("rawseq_output_span_buckets")
        ),
        "input_span_seconds": normalize_int(
            payload.get("input_span_seconds") or payload.get("rawseq_input_span_seconds")
        ),
        "output_span_seconds": normalize_int(
            payload.get("output_span_seconds") or payload.get("rawseq_output_span_seconds")
        ),
        "input_feature": input_feature,
        "ma_window": ma_window,
        "hidden_declared": hidden_declared,
        "hidden_inferred": hidden_inferred,
        "hidden": hidden,
        "architecture_input_dim": normalize_int(arch.get("input_dim")),
        "architecture_output_dim": normalize_int(arch.get("output_dim")),
        "w1_shape": f"{w1_rows}x{w1_cols}" if w1_rows is not None else "",
        "w2_shape": f"{w2_rows}x{w2_cols}" if w2_rows is not None else "",
        "w3_shape": f"{w3_rows}x{w3_cols}" if w3_rows is not None else "",
        "b1_len": safe_str(vector_len(weights.get("b1"))),
        "b2_len": safe_str(vector_len(weights.get("b2"))),
        "b3_len": safe_str(vector_len(weights.get("b3"))),
        "y_scaler_mean_len": safe_str(vector_len(y_scaler.get("mean"))),
        "y_scaler_std_len": safe_str(vector_len(y_scaler.get("std"))),
        "seed": normalize_int(pop.get("seed") or payload.get("seed")),
        "created_at": safe_str(payload.get("created_at")),
        "best_validation_fitness": safe_str(payload.get("best_validation_fitness")),
        "fitness_policy": safe_str(payload.get("fitness_policy")),
        "fitness_threshold_bps": safe_str(payload.get("fitness_threshold_bps")),
        "decision_horizon_seconds": normalize_int(payload.get("decision_horizon_seconds")) or "30",
        "decision_threshold_bps": safe_str(payload.get("decision_threshold_bps")),
    }
    if not contract["input_span_buckets"] and contract["seq_len"] and contract["input_stride"]:
        contract["input_span_buckets"] = str(int(contract["seq_len"]) * int(contract["input_stride"]))
    if not contract["output_span_buckets"] and contract["seq_len"] and contract["output_stride"]:
        contract["output_span_buckets"] = str(int(contract["seq_len"]) * int(contract["output_stride"]))
    if not contract["input_span_seconds"] and contract["input_span_buckets"] and contract["bucket_seconds"]:
        contract["input_span_seconds"] = str(int(contract["input_span_buckets"]) * int(contract["bucket_seconds"]))
    if not contract["output_span_seconds"] and contract["output_span_buckets"] and contract["bucket_seconds"]:
        contract["output_span_seconds"] = str(int(contract["output_span_buckets"]) * int(contract["bucket_seconds"]))
    return contract


def validate_contract(contract: dict[str, Any]) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(name: str, expected: str, actual: str) -> None:
        checks.append(
            {
                "check": name,
                "expected": expected,
                "actual": actual,
                "status": "PASS" if safe_str(expected) == safe_str(actual) else "FAIL",
            }
        )

    add("seq_len_vs_w1_rows", contract["seq_len"], contract["w1_shape"].split("x")[0])
    add("output_dim_vs_w3_cols", contract["output_dim"], contract["w3_shape"].split("x")[-1])
    if contract.get("output_label") == "future_return_path":
        add("seq_len_vs_output_dim", contract["seq_len"], contract["output_dim"])
    add("hidden_declared_vs_inferred", contract["hidden_declared"], contract["hidden_inferred"])
    if contract["w2_shape"]:
        w2_rows, w2_cols = contract["w2_shape"].split("x", 1)
        hidden_parts = contract["hidden"].split(",")
        if len(hidden_parts) == 2:
            add("hidden_1_vs_w2_rows", hidden_parts[0], w2_rows)
            add("hidden_2_vs_w2_cols", hidden_parts[1], w2_cols)
    if contract["b1_len"]:
        add("hidden_1_vs_b1_len", contract["hidden"].split(",")[0], contract["b1_len"])
    if contract["b2_len"] and "," in contract["hidden"]:
        add("hidden_2_vs_b2_len", contract["hidden"].split(",")[1], contract["b2_len"])
    if contract["b3_len"]:
        add("output_dim_vs_b3_len", contract["output_dim"], contract["b3_len"])
    if contract.get("y_scaler_mean_len"):
        add("output_dim_vs_y_scaler_mean_len", contract["output_dim"], contract["y_scaler_mean_len"])
    if contract.get("y_scaler_std_len"):
        add("output_dim_vs_y_scaler_std_len", contract["output_dim"], contract["y_scaler_std_len"])

    required_fields = [
        "symbol",
        "venue",
        "source_path",
        "bucket_seconds",
        "seq_len",
        "input_stride",
        "output_stride",
        "input_feature",
        "hidden",
        "output_label",
        "output_dim",
        "task_type",
    ]
    for field in required_fields:
        checks.append(
            {
                "check": f"required_{field}",
                "expected": "nonempty",
                "actual": safe_str(contract.get(field)),
                "status": "PASS" if safe_str(contract.get(field)) else "FAIL",
            }
        )
    return pd.DataFrame(checks)


def build_probe_dir(contract: dict[str, Any], model_path: Path) -> Path:
    slug = safe_slug(
        "_".join(
            [
                contract["symbol"],
                contract["venue"],
                contract["input_feature"] or "feature_unknown",
                f"seq{contract['seq_len'] or 'unknown'}",
                f"is{contract['input_stride'] or '1'}",
                f"os{contract['output_stride'] or '1'}",
                contract.get("output_label") or "future_return_path",
                f"h{(contract['hidden'] or 'unknown').replace(',', 'x')}",
                f"b{contract['bucket_seconds'] or 'unknown'}",
                Path(contract["source_path_basename"]).stem or "source_unknown",
                model_path.parent.name,
                now_stamp(),
            ]
        )
    )
    return OUTPUT_ROOT / slug


def run_inference(contract: dict[str, Any], model_path: Path, run_dir: Path) -> dict[str, Path]:
    venue = contract["venue"]
    symbol = contract["symbol"]
    realtime_dir = PROJECT_ROOT / "data" / "realtime" / venue
    expected_outputs = {
        "rows": realtime_dir / f"{symbol}_tiny_price_rawseq_path_v1_rows.csv",
        "annotated": realtime_dir / f"{symbol}_tiny_price_prediction_evaluation_rows_with_rawseq_path_v1.csv",
        "evaluation": realtime_dir / f"{symbol}_tiny_price_rawseq_path_v1_shadow_evaluation.csv",
        "label_metric_summary": realtime_dir / f"{symbol}_tiny_price_rawseq_path_v1_label_metric_summary.csv",
        "label_shape_audit": realtime_dir / f"{symbol}_tiny_price_rawseq_path_v1_label_shape_audit.csv",
    }

    env = os.environ.copy()
    env.update(
        {
            "SYMBOL": symbol,
            "PRIMARY_VENUE": venue,
            "RAWSEQ_SOURCE_PATH": contract["source_path"],
            "RAWSEQ_BUCKET_SECONDS": contract["bucket_seconds"],
            "RAWSEQ_LEN": contract["seq_len"],
            "RAWSEQ_INPUT_STRIDE": contract["input_stride"],
            "RAWSEQ_OUTPUT_STRIDE": contract["output_stride"],
            "RAWSEQ_INPUT_FEATURE": contract["input_feature"],
            "RAWSEQ_OUTPUT_LABEL": contract.get("output_label") or "future_return_path",
            "RAWSEQ_HIDDEN": contract["hidden"],
            "RAWSEQ_INCLUDE_WINDOW_GUIDE": "false",
            "RAWSEQ_INFERENCE_ONLY": "true",
            "RAWSEQ_LOAD_MODEL_PATH": str(model_path),
            "RAWSEQ_DECISION_HORIZON_SECONDS": contract["decision_horizon_seconds"],
        }
    )
    if contract["ma_window"]:
        env["RAWSEQ_MA_WINDOW"] = contract["ma_window"]

    log_path = run_dir / "run.log"
    proc = subprocess.run(
        [sys.executable, str(RAWSEQ_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        if "Unknown RAWSEQ_INPUT_FEATURE=signed_bucket_return_bps" in proc.stdout:
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\nProbe fallback: direct signed_bucket_return_bps inference.\n")
            return run_direct_inference(contract, model_path, run_dir)
        raise SystemExit(f"Inference failed with exit code {proc.returncode}. See {log_path}")

    archived = {}
    for key, src in expected_outputs.items():
        if not src.exists():
            raise SystemExit(f"Expected inference output missing: {src}")
        dst = run_dir / f"{key}.csv"
        shutil.copy2(src, dst)
        archived[key] = dst
    return archived


def load_rawseq_module(env_updates: dict[str, str]):
    old_env = {key: os.environ.get(key) for key in env_updates}
    os.environ.update(env_updates)
    module_name = f"_rawseq_probe_{now_stamp()}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, RAWSEQ_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load rawseq script module: {RAWSEQ_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


def run_direct_inference(contract: dict[str, Any], model_path: Path, run_dir: Path) -> dict[str, Path]:
    """Run probe-only inference for legacy payload feature aliases.

    The production script currently treats "return" as signed bucket returns but
    does not accept the explicit payload label "signed_bucket_return_bps".
    This fallback keeps the payload contract visible while reusing the same
    rawseq row/scaler/MLP implementation and writing only into the probe folder.
    """

    rawseq_feature = contract["input_feature"]
    module_feature = "return" if rawseq_feature == "signed_bucket_return_bps" else rawseq_feature
    env_updates = {
        "SYMBOL": contract["symbol"],
        "PRIMARY_VENUE": contract["venue"],
        "RAWSEQ_SOURCE_PATH": contract["source_path"],
        "RAWSEQ_BUCKET_SECONDS": contract["bucket_seconds"],
        "RAWSEQ_LEN": contract["seq_len"],
        "RAWSEQ_INPUT_STRIDE": contract["input_stride"],
        "RAWSEQ_OUTPUT_STRIDE": contract["output_stride"],
        "RAWSEQ_INPUT_FEATURE": module_feature,
        "RAWSEQ_OUTPUT_LABEL": contract.get("output_label") or "future_return_path",
        "RAWSEQ_HIDDEN": contract["hidden"],
        "RAWSEQ_INCLUDE_WINDOW_GUIDE": "false",
        "RAWSEQ_INFERENCE_ONLY": "true",
        "RAWSEQ_LOAD_MODEL_PATH": str(model_path),
        "RAWSEQ_DECISION_HORIZON_SECONDS": contract["decision_horizon_seconds"],
    }
    if contract["ma_window"]:
        env_updates["RAWSEQ_MA_WINDOW"] = contract["ma_window"]

    rawseq = load_rawseq_module(env_updates)
    if rawseq_feature == "signed_bucket_return_bps":
        def probe_signed_return_feature(bucketed: pd.DataFrame):
            values = np.array(
                pd.to_numeric(bucketed["bucket_return_bps"], errors="coerce").to_numpy(dtype=np.float64),
                dtype=np.float64,
                copy=True,
            )
            values[~np.isfinite(values)] = 0.0
            return values, "signed_bucket_return_bps", {"input_feature": rawseq_feature, "ma_window": ""}

        rawseq.build_input_feature = probe_signed_return_feature

    payload = load_model(model_path)
    model = {key: np.asarray(value, dtype="float64") for key, value in payload["weights"].items()}

    x_scaler = {
        "mean": np.asarray(payload["x_scaler"]["mean"], dtype="float64"),
        "std": np.asarray(payload["x_scaler"]["std"], dtype="float64"),
    }
    y_scaler = {
        "mean": np.asarray(payload["y_scaler"]["mean"], dtype="float64"),
        "std": np.asarray(payload["y_scaler"]["std"], dtype="float64"),
    }

    source = rawseq.load_source()
    bucketed = rawseq.bucketize(source)
    rows, X, Y = rawseq.build_rawseq_rows(bucketed)
    rows["rawseq_input_feature"] = rawseq_feature
    rows["rawseq_payload_input_feature"] = rawseq_feature
    rows["rawseq_output_label"] = contract.get("output_label") or "future_return_path"

    expected_input_dim = int(payload["architecture"]["input_dim"])
    expected_output_dim = int(payload.get("output_dim") or payload["architecture"]["output_dim"])
    if X.shape[1] != expected_input_dim:
        raise SystemExit(
            f"Input dim mismatch. Fresh X has {X.shape[1]}, "
            f"but model expects {expected_input_dim}."
        )
    if Y.shape[1] != expected_output_dim:
        raise SystemExit(
            f"Y output dim mismatch. Fresh Y has {Y.shape[1]}, but model expects {expected_output_dim}."
        )
    rawseq.assert_model_output_contract(model, y_scaler, expected_output_dim, "probe-direct")

    split = rawseq.chronological_split(len(rows))
    timestamps = rows["timestamp"].to_numpy(dtype=np.float64)
    Xs = rawseq.transform(X, x_scaler)
    pred_all_scaled, _ = rawseq.forward(model, Xs)
    if pred_all_scaled.shape[1] != expected_output_dim:
        raise SystemExit(
            f"Model output dim mismatch. Predictions have {pred_all_scaled.shape[1]}, expected {expected_output_dim}."
        )
    pred_all = rawseq.unscale_y(pred_all_scaled, y_scaler)

    val_eval, _ = rawseq.evaluate_model(
        model,
        Xs[split.val],
        Y[split.val],
        y_scaler,
        timestamps[split.val],
        "validation_probe_inference_only",
    )
    test_eval, _ = rawseq.evaluate_model(
        model,
        Xs[split.test],
        Y[split.test],
        y_scaler,
        timestamps[split.test],
        "test_probe_inference_only",
    )
    evaluation = pd.concat([val_eval, test_eval], ignore_index=True, sort=False)

    output_step_seconds = int_contract(contract, "bucket_seconds") * int_contract(contract, "output_stride")
    horizon_idx = min(
        max(1, int(contract["decision_horizon_seconds"]) // output_step_seconds),
        int(contract["seq_len"]),
    ) - 1
    annotated = rawseq.add_label_annotation_columns(rows.copy(), pred_all, Y)
    label_metrics = pd.DataFrame(
        rawseq.label_metric_rows(Y[split.val], pred_all[split.val], "validation_probe_inference_only")
        + rawseq.label_metric_rows(Y[split.test], pred_all[split.test], "test_probe_inference_only")
    )
    label_audit = pd.DataFrame(rawseq.label_shape_audit_rows(Y, pred_all, y_scaler, "probe_direct"))

    artifacts = {
        "rows": run_dir / "rows.csv",
        "annotated": run_dir / "annotated.csv",
        "evaluation": run_dir / "evaluation.csv",
        "label_metric_summary": run_dir / "label_metric_summary.csv",
        "label_shape_audit": run_dir / "label_shape_audit.csv",
    }
    rows.to_csv(artifacts["rows"], index=False)
    annotated.to_csv(artifacts["annotated"], index=False)
    evaluation.to_csv(artifacts["evaluation"], index=False)
    label_metrics.to_csv(artifacts["label_metric_summary"], index=False)
    label_audit.to_csv(artifacts["label_shape_audit"], index=False)
    with (run_dir / "run.log").open("a", encoding="utf-8") as log:
        log.write(f"Probe direct rows: {artifacts['rows']}\n")
        log.write(f"Probe direct annotated: {artifacts['annotated']}\n")
        log.write(f"Probe direct evaluation: {artifacts['evaluation']}\n")
        log.write(f"Probe direct label metrics: {artifacts['label_metric_summary']}\n")
        log.write(f"Probe direct label shape audit: {artifacts['label_shape_audit']}\n")
        log.write("Probe direct inference complete: no training, no promotion, no orders.\n")
    return artifacts


def load_test_frame(annotated_path: Path) -> pd.DataFrame:
    if not 0.0 < TEST_FRAC <= 1.0:
        raise SystemExit(f"RAWSEQ_PROBE_TEST_FRAC must be in (0, 1], got {TEST_FRAC}")
    try:
        frame = pd.read_csv(annotated_path, usecols=REQUIRED_ANNOTATED_COLUMNS, low_memory=False)
    except ValueError as exc:
        raise SystemExit(f"Annotated rows missing required columns: {exc}") from exc
    if frame.empty:
        raise SystemExit(f"Annotated rows are empty: {annotated_path}")
    split_at = int(len(frame) * (1.0 - TEST_FRAC))
    test = frame.iloc[split_at:].copy()
    test["timestamp"] = pd.to_numeric(test["timestamp"], errors="coerce")
    test[PRED_COLUMN] = pd.to_numeric(test[PRED_COLUMN], errors="coerce")
    test[ACTUAL_COLUMN] = pd.to_numeric(test[ACTUAL_COLUMN], errors="coerce")
    test = test.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["timestamp", PRED_COLUMN, ACTUAL_COLUMN]
    )
    if test.empty:
        raise SystemExit("Test split has no finite timestamp/prediction/actual rows.")
    return test.sort_values("timestamp").reset_index(drop=True)


def selected_gross(frame: pd.DataFrame, threshold_bps: float) -> np.ndarray:
    pred = frame[PRED_COLUMN].to_numpy(dtype="float64")
    actual = frame[ACTUAL_COLUMN].to_numpy(dtype="float64")
    if POLICY == "inverse_gt":
        mask = pred > threshold_bps
        gross = -actual[mask]
    elif POLICY == "direct_gt":
        mask = pred > threshold_bps
        gross = actual[mask]
    elif POLICY == "inverse_directional_abs_gt":
        mask = np.abs(pred) > threshold_bps
        gross = -np.sign(pred[mask]) * actual[mask]
    else:
        raise SystemExit(
            "RAWSEQ_PROBE_POLICY must be one of: inverse_gt, direct_gt, inverse_directional_abs_gt"
        )
    gross = np.asarray(gross, dtype="float64")
    return gross[np.isfinite(gross)]


def build_cost_threshold_summary(
    test: pd.DataFrame,
    contract: dict[str, Any],
    thresholds: list[float],
    costs: list[float],
    annotated_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        gross = selected_gross(test, threshold)
        for cost in costs:
            net = gross - cost
            rows.append(
                {
                    "symbol": contract["symbol"],
                    "venue": contract["venue"],
                    "model_path": contract["model_path"],
                    "input_feature": contract["input_feature"],
                    "source_path_basename": contract["source_path_basename"],
                    "bucket_seconds": contract["bucket_seconds"],
                    "seq_len": contract["seq_len"],
                    "output_label": contract.get("output_label", ""),
                    "output_dim": contract.get("output_dim", ""),
                    "task_type": contract.get("task_type", ""),
                    "input_stride": contract["input_stride"],
                    "output_stride": contract["output_stride"],
                    "hidden": contract["hidden"],
                    "policy": POLICY,
                    "threshold_bps": threshold,
                    "cost_bps": cost,
                    "test_frac": TEST_FRAC,
                    "selected_rows": int(len(gross)),
                    "avg_gross_bps": float(np.mean(gross)) if len(gross) else math.nan,
                    "avg_net_bps": float(np.mean(net)) if len(gross) else math.nan,
                    "cum_gross_bps": float(np.sum(gross)) if len(gross) else 0.0,
                    "cum_net_bps": float(np.sum(net)) if len(gross) else 0.0,
                    "win_rate_gross": float(np.mean(gross > 0.0)) if len(gross) else math.nan,
                    "win_rate_net": float(np.mean(net > 0.0)) if len(gross) else math.nan,
                    "max_dip_gross_bps": max_dip_bps(gross),
                    "max_dip_net_bps": max_dip_bps(net),
                    "first_time": str(test["time"].iloc[0]),
                    "last_time": str(test["time"].iloc[-1]),
                    "test_rows_total": int(len(test)),
                    "annotated_path": str(annotated_path),
                    "paper_only": True,
                    "orders": False,
                    "promotion": False,
                }
            )
    return pd.DataFrame(rows)


def build_rolling_summary(
    test: pd.DataFrame,
    contract: dict[str, Any],
    thresholds: list[float],
    costs: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    first_timestamp = float(test["timestamp"].iloc[0])
    elapsed_ms = test["timestamp"] - first_timestamp
    for threshold in thresholds:
        for cost in costs:
            for window_hours in ROLLING_WINDOW_HOURS:
                window_ms = window_hours * 60.0 * 60.0 * 1000.0
                window_ids = np.floor(elapsed_ms / window_ms).astype("int64")
                for window_id, window in test.groupby(window_ids, sort=True):
                    gross = selected_gross(window.reset_index(drop=True), threshold)
                    net = gross - cost
                    rows.append(
                        {
                            "symbol": contract["symbol"],
                            "venue": contract["venue"],
                            "input_feature": contract["input_feature"],
                            "source_path_basename": contract["source_path_basename"],
                            "bucket_seconds": contract["bucket_seconds"],
                            "seq_len": contract["seq_len"],
                            "output_label": contract.get("output_label", ""),
                            "output_dim": contract.get("output_dim", ""),
                            "task_type": contract.get("task_type", ""),
                            "input_stride": contract["input_stride"],
                            "output_stride": contract["output_stride"],
                            "hidden": contract["hidden"],
                            "policy": POLICY,
                            "threshold_bps": threshold,
                            "cost_bps": cost,
                            "window_hours": window_hours,
                            "window_id": int(window_id),
                            "window_start_time": str(window["time"].iloc[0]),
                            "window_end_time": str(window["time"].iloc[-1]),
                            "total_rows": int(len(window)),
                            "selected_rows": int(len(gross)),
                            "avg_gross_bps": float(np.mean(gross)) if len(gross) else math.nan,
                            "avg_net_bps": float(np.mean(net)) if len(gross) else math.nan,
                            "cum_gross_bps": float(np.sum(gross)) if len(gross) else 0.0,
                            "cum_net_bps": float(np.sum(net)) if len(gross) else 0.0,
                            "win_rate_net": float(np.mean(net > 0.0)) if len(gross) else math.nan,
                            "max_dip_net_bps": max_dip_bps(net),
                        }
                    )
    return pd.DataFrame(rows)


def build_label_policy_placeholder(contract: dict[str, Any], thresholds: list[float], costs: list[float], annotated_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for cost in costs:
            rows.append(
                {
                    "symbol": contract["symbol"],
                    "venue": contract["venue"],
                    "model_path": contract["model_path"],
                    "input_feature": contract["input_feature"],
                    "source_path_basename": contract["source_path_basename"],
                    "bucket_seconds": contract["bucket_seconds"],
                    "seq_len": contract["seq_len"],
                    "output_label": contract.get("output_label", ""),
                    "output_dim": contract.get("output_dim", ""),
                    "task_type": contract.get("task_type", ""),
                    "input_stride": contract["input_stride"],
                    "output_stride": contract["output_stride"],
                    "hidden": contract["hidden"],
                    "policy": "label_specific_metrics_only",
                    "threshold_bps": threshold,
                    "cost_bps": cost,
                    "test_frac": TEST_FRAC,
                    "selected_rows": 0,
                    "avg_gross_bps": math.nan,
                    "avg_net_bps": math.nan,
                    "cum_gross_bps": math.nan,
                    "cum_net_bps": math.nan,
                    "win_rate_gross": math.nan,
                    "win_rate_net": math.nan,
                    "max_dip_gross_bps": math.nan,
                    "max_dip_net_bps": math.nan,
                    "annotated_path": str(annotated_path),
                    "not_applicable_reason": "direct_gt/inverse_gt trading evaluation is only valid for future_return_path",
                    "paper_only": True,
                    "orders": False,
                    "promotion": False,
                }
            )
    return pd.DataFrame(rows)


def best_rows(summary: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    base = summary[
        (summary["threshold_bps"].sub(0.3).abs() < 1e-12)
        & (summary["cost_bps"].sub(0.1).abs() < 1e-12)
    ]
    base_row = base.iloc[0] if not base.empty else None
    candidates = summary[summary["selected_rows"] > 0].copy()
    if candidates.empty:
        return base_row, None
    candidates = candidates.sort_values(
        ["cum_net_bps", "avg_net_bps", "selected_rows"],
        ascending=[False, False, False],
    )
    return base_row, candidates.iloc[0]


def render_text(
    contract: dict[str, Any],
    audit: pd.DataFrame,
    cost_summary: pd.DataFrame,
    rolling: pd.DataFrame,
    run_dir: Path,
    artifacts: dict[str, Path],
) -> str:
    base_row, best_row = best_rows(cost_summary)
    audit_status = "PASS" if not audit.empty and audit["status"].eq("PASS").all() else "FAIL"
    lines = [
        "Rawseq Candidate Shadow Probe",
        "",
        f"Status: {'PASS' if audit_status == 'PASS' else 'FAIL'}",
        f"Probe dir: {run_dir}",
        "",
        "Contract",
        f"  model_path: {contract['model_path']}",
        f"  symbol: {contract['symbol']}",
        f"  venue: {contract['venue']}",
        f"  source_path: {contract['source_path']}",
        f"  bucket_seconds: {contract['bucket_seconds']}",
        f"  seq_len: {contract['seq_len']}",
        f"  output_label: {contract.get('output_label', '')}",
        f"  task_type: {contract.get('task_type', '')}",
        f"  output_dim: {contract.get('output_dim', '')}",
        f"  label_required_horizon_buckets: {contract.get('label_required_horizon_buckets', '')}",
        f"  input_stride: {contract['input_stride']} (span {contract['input_span_seconds']}s)",
        f"  output_stride: {contract['output_stride']} (span {contract['output_span_seconds']}s)",
        f"  input_feature: {contract['input_feature']}",
        f"  ma_window: {contract['ma_window']}",
        f"  hidden_declared: {contract['hidden_declared']}",
        f"  hidden_inferred: {contract['hidden_inferred']}",
        f"  W1/W2/W3: {contract['w1_shape']} / {contract['w2_shape']} / {contract['w3_shape']}",
        f"  seed: {contract['seed']}",
        f"  created_at: {contract['created_at']}",
        f"  best_validation_fitness: {contract['best_validation_fitness']}",
        "",
        "Archived Inference Outputs",
        f"  rows: {artifacts.get('rows', '')}",
        f"  annotated: {artifacts.get('annotated', '')}",
        f"  evaluation: {artifacts.get('evaluation', '')}",
        f"  label_metric_summary: {artifacts.get('label_metric_summary', '')}",
        f"  label_shape_audit: {artifacts.get('label_shape_audit', '')}",
        "",
        "Contract Audit",
        f"  status: {audit_status}",
    ]
    if not audit.empty:
        for _, row in audit.iterrows():
            if row["status"] != "PASS":
                lines.append(f"  FAIL {row['check']}: expected={row['expected']} actual={row['actual']}")

    lines += [
        "",
        "Fixed Cost/Threshold Summary",
        f"  policy: {POLICY}",
        f"  thresholds: {THRESHOLD_BPS_LIST_ENV}",
        f"  costs: {COST_BPS_LIST_ENV}",
        f"  test_frac: {TEST_FRAC:g}",
    ]
    if contract.get("output_label", "future_return_path") != "future_return_path":
        lines.append("  not_applicable: direct_gt/inverse_gt trading grids are skipped for high/low/envelope labels.")
    if base_row is not None:
        lines.append(
            "  threshold=0.3 cost=0.1: "
            f"rows={int(base_row['selected_rows'])} "
            f"avg_net={finite_or_nan(base_row['avg_net_bps']):.6f} "
            f"cum_net={finite_or_nan(base_row['cum_net_bps']):.6f} "
            f"win_net={finite_or_nan(base_row['win_rate_net']):.6f} "
            f"max_dip_net={finite_or_nan(base_row['max_dip_net_bps']):.6f}"
        )
    else:
        lines.append("  threshold=0.3 cost=0.1: not in requested grid")
    if best_row is not None:
        lines.append(
            "  best_grid_by_cum_net: "
            f"threshold={best_row['threshold_bps']:g} cost={best_row['cost_bps']:g} "
            f"rows={int(best_row['selected_rows'])} "
            f"avg_net={finite_or_nan(best_row['avg_net_bps']):.6f} "
            f"cum_net={finite_or_nan(best_row['cum_net_bps']):.6f}"
        )

    lines += [
        "",
        "Rolling Summary",
    ]
    if rolling.empty:
        lines.append("  no rolling rows")
    else:
        primary_threshold = 0.3 if any(abs(t - 0.3) < 1e-12 for t in cost_summary["threshold_bps"]) else float(cost_summary["threshold_bps"].iloc[0])
        primary_cost = 0.1 if any(abs(c - 0.1) < 1e-12 for c in cost_summary["cost_bps"]) else float(cost_summary["cost_bps"].iloc[0])
        subset = rolling[
            (rolling["threshold_bps"].sub(primary_threshold).abs() < 1e-12)
            & (rolling["cost_bps"].sub(primary_cost).abs() < 1e-12)
        ]
        lines.append(f"  displayed threshold={primary_threshold:g} cost={primary_cost:g}")
        for window_hours, group in subset.groupby("window_hours", sort=True):
            lines.append(
                f"  {float(window_hours):g}h: windows={len(group)} "
                f"positive={int((group['cum_net_bps'] > 0.0).sum())} "
                f"total_cum_net={finite_or_nan(group['cum_net_bps'].sum()):.6f} "
                f"worst={finite_or_nan(group['cum_net_bps'].min()):.6f} "
                f"selected_rows={int(group['selected_rows'].sum())}"
            )

    lines += [
        "",
        "Safety",
        "  paper_only: true",
        "  orders: false",
        "  promotion: false",
        "  champion_mutation: false",
        "  training: false",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    model_path = resolve_existing_path(MODEL_PATH_ENV, "RAWSEQ_PROBE_MODEL_PATH")
    payload = load_model(model_path)
    contract = extract_contract(payload, model_path)
    audit = validate_contract(contract)

    run_dir = build_probe_dir(contract, model_path)
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "model_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    audit.to_csv(run_dir / "contract_audit.csv", index=False)

    artifacts = run_inference(contract, model_path, run_dir)
    thresholds = parse_float_list(THRESHOLD_BPS_LIST_ENV, "RAWSEQ_PROBE_THRESHOLD_BPS_LIST")
    costs = parse_float_list(COST_BPS_LIST_ENV, "RAWSEQ_PROBE_COST_BPS_LIST")

    if contract.get("output_label", "future_return_path") == "future_return_path":
        test = load_test_frame(artifacts["annotated"])
        cost_summary = build_cost_threshold_summary(test, contract, thresholds, costs, artifacts["annotated"])
        rolling = build_rolling_summary(test, contract, thresholds, costs)
    else:
        cost_summary = build_label_policy_placeholder(contract, thresholds, costs, artifacts["annotated"])
        rolling = pd.DataFrame()
    cost_summary.to_csv(run_dir / "cost_threshold_summary.csv", index=False)
    rolling.to_csv(run_dir / "rolling_summary.csv", index=False)

    text = render_text(contract, audit, cost_summary, rolling, run_dir, artifacts)
    summary_path = run_dir / "summary.txt"
    summary_path.write_text(text, encoding="utf-8")

    print(text)
    print(f"Probe folder: {run_dir}")
    print("Safety: paper-only. No training. No promotion. No champion mutation. No orders.")


if __name__ == "__main__":
    main()
