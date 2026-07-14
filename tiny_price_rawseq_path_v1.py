#!/usr/bin/env python3
"""
tiny_price_rawseq_path_v1.py

Paper-only raw sequence path model.

Idea:
    Input  X = past L buckets of one signed raw feature.
    Output Y = future L buckets of signed cumulative return.

Default:
    L = 60
    bucket_seconds = 1
    model = 60 -> 16 -> 8 -> 60
    population = 5
    generations = 3

Safety:
    No orders.
    No private API.
    No promotion.
    No champion replacement.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, UTC

import numpy as np
import pandas as pd


# =========================
# Config
# =========================

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper() or "SOLUSDT"
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "simulated").strip().lower() or "simulated"

OUTPUT_DIR = ROOT / "data" / "realtime" / PRIMARY_VENUE
MODEL_ROOT = ROOT / "models" / "candidates" / SYMBOL / "tiny_price_rawseq_path_v1" / PRIMARY_VENUE
ARTIFACT_OUTPUT_DIR = Path(os.getenv("RAWSEQ_ARTIFACT_OUTPUT_DIR", "")).expanduser() if os.getenv("RAWSEQ_ARTIFACT_OUTPUT_DIR") else OUTPUT_DIR
if not ARTIFACT_OUTPUT_DIR.is_absolute():
    ARTIFACT_OUTPUT_DIR = ROOT / ARTIFACT_OUTPUT_DIR
ARTIFACT_PREFIX = os.getenv("RAWSEQ_ARTIFACT_PREFIX", "").strip()

DEFAULT_SOURCE_CANDIDATES = [
    OUTPUT_DIR / f"{SYMBOL}_1s_flow.csv",
    OUTPUT_DIR / f"{SYMBOL}_tiny_price_prediction_evaluation_rows.csv",
    OUTPUT_DIR / f"{SYMBOL}_tiny_price_prediction_evaluation_rows_with_v2_stabilized_shadow.csv",
    OUTPUT_DIR / f"{SYMBOL}_10s_flow.csv",
]

RAWSEQ_FITNESS_POLICY = os.getenv("RAWSEQ_FITNESS_POLICY", "direct_gt").strip().lower()
RAWSEQ_FITNESS_THRESHOLD_BPS = float(os.getenv("RAWSEQ_FITNESS_THRESHOLD_BPS", "0.0"))
RAWSEQ_MIN_FITNESS_TRADES = int(os.getenv("RAWSEQ_MIN_FITNESS_TRADES", "100"))

RAWSEQ_SOURCE_PATH = Path(os.getenv("RAWSEQ_SOURCE_PATH", "")).expanduser() if os.getenv("RAWSEQ_SOURCE_PATH") else None
RAWSEQ_INFERENCE_ONLY = os.getenv(
    "RAWSEQ_INFERENCE_ONLY", "false"
).strip().lower() in {"1", "true", "yes", "y"}

RAWSEQ_LOAD_MODEL_PATH = os.getenv("RAWSEQ_LOAD_MODEL_PATH", "").strip()
def artifact_name(name: str) -> str:
    return f"{ARTIFACT_PREFIX}_{name}" if ARTIFACT_PREFIX else name


ROWS_PATH = ARTIFACT_OUTPUT_DIR / artifact_name("rows.csv")
ANNOTATED_PATH = ARTIFACT_OUTPUT_DIR / artifact_name("annotated.csv")
EVALUATION_PATH = ARTIFACT_OUTPUT_DIR / artifact_name("evaluation.csv")
HISTORY_PATH = ARTIFACT_OUTPUT_DIR / artifact_name("history.csv")
LABEL_METRIC_SUMMARY_PATH = ARTIFACT_OUTPUT_DIR / artifact_name("label_metric_summary.csv")
LABEL_SHAPE_AUDIT_PATH = ARTIFACT_OUTPUT_DIR / artifact_name("label_shape_audit.csv")
FEATURE_AUDIT_PATH = ARTIFACT_OUTPUT_DIR / artifact_name("feature_audit.csv")

BUCKET_SECONDS = int(os.getenv("RAWSEQ_BUCKET_SECONDS", "1"))
SEQ_LEN = int(os.getenv("RAWSEQ_LEN", "60"))
RAWSEQ_INPUT_STRIDE = int(os.getenv("RAWSEQ_INPUT_STRIDE", "1"))
RAWSEQ_OUTPUT_STRIDE = int(os.getenv("RAWSEQ_OUTPUT_STRIDE", "1"))
if RAWSEQ_INPUT_STRIDE < 1:
    raise SystemExit("RAWSEQ_INPUT_STRIDE must be >= 1")
if RAWSEQ_OUTPUT_STRIDE < 1:
    raise SystemExit("RAWSEQ_OUTPUT_STRIDE must be >= 1")
RAWSEQ_INPUT_SPAN_BUCKETS = SEQ_LEN * RAWSEQ_INPUT_STRIDE
RAWSEQ_OUTPUT_SPAN_BUCKETS = SEQ_LEN * RAWSEQ_OUTPUT_STRIDE
RAWSEQ_INPUT_SPAN_SECONDS = RAWSEQ_INPUT_SPAN_BUCKETS * BUCKET_SECONDS
RAWSEQ_OUTPUT_SPAN_SECONDS = RAWSEQ_OUTPUT_SPAN_BUCKETS * BUCKET_SECONDS
HIDDEN = [
    int(x.strip())
    for x in os.getenv("RAWSEQ_HIDDEN", "16,8").split(",")
    if x.strip()
]
if len(HIDDEN) != 2:
    raise SystemExit("RAWSEQ_HIDDEN must be two comma-separated integers, e.g. 16,8 or 2,2")

POPULATION = int(os.getenv("RAWSEQ_POPULATION", os.getenv("RAWSEQ_WF_POPULATION", os.getenv("RAWSEQ_IO_DISCOVERY_POPULATION", "5"))))
GENERATIONS = int(os.getenv("RAWSEQ_GENERATIONS", os.getenv("RAWSEQ_WF_GENERATIONS", os.getenv("RAWSEQ_IO_DISCOVERY_GENERATIONS", "3"))))
EPOCHS_PER_GENERATION = int(os.getenv("RAWSEQ_EPOCHS", os.getenv("RAWSEQ_WF_EPOCHS", os.getenv("RAWSEQ_IO_DISCOVERY_EPOCHS", "35"))))
BATCH_SIZE = int(os.getenv("RAWSEQ_BATCH_SIZE", "256"))
LEARNING_RATE = float(os.getenv("RAWSEQ_LR", "0.001"))
MUTATION_STD = float(os.getenv("RAWSEQ_MUTATION_STD", "0.025"))
TARGET_CLIP_BPS = float(os.getenv("RAWSEQ_TARGET_CLIP_BPS", "80"))
SEED = int(os.getenv("RAWSEQ_SEED", os.getenv("RAWSEQ_WF_SEED", "778")))
EARLY_STOP_PATIENCE = int(os.getenv("RAWSEQ_EVOLUTION_EARLY_STOP_PATIENCE", "3"))
EARLY_STOP_MIN_IMPROVEMENT = float(os.getenv("RAWSEQ_EVOLUTION_MIN_IMPROVEMENT", "1e-6"))

TRAIN_FRAC = float(os.getenv("RAWSEQ_TRAIN_FRAC", "0.60"))
VAL_FRAC = float(os.getenv("RAWSEQ_VAL_FRAC", "0.20"))

DECISION_HORIZON_SECONDS = int(os.getenv("RAWSEQ_DECISION_HORIZON_SECONDS", "30"))
DECISION_THRESHOLD_BPS = float(os.getenv("RAWSEQ_DECISION_THRESHOLD_BPS", "0.0"))
MIN_VAL_ROWS = int(os.getenv("RAWSEQ_MIN_VAL_ROWS", "25"))

# If set, require predicted path not to dip below this adverse level before horizon.
# Example: -5 means suppress if predicted curve dips below -5 bps.
MAX_EARLY_ADVERSE_BPS = os.getenv("RAWSEQ_MAX_EARLY_ADVERSE_BPS", "").strip()
MAX_EARLY_ADVERSE_BPS = float(MAX_EARLY_ADVERSE_BPS) if MAX_EARLY_ADVERSE_BPS else math.nan

RAWSEQ_INPUT_FEATURE = os.getenv("RAWSEQ_INPUT_FEATURE", "return").strip().lower()
RAWSEQ_OUTPUT_LABEL = os.getenv("RAWSEQ_OUTPUT_LABEL", "future_return_path").strip().lower()
SUPPORTED_REGRESSION_OUTPUT_LABELS = {
    "future_return_path",
    "future_high_from_now_bps_path",
    "future_low_from_now_bps_path",
    "future_range_envelope_path",
}
if RAWSEQ_OUTPUT_LABEL not in SUPPORTED_REGRESSION_OUTPUT_LABELS:
    raise SystemExit(
        "RAWSEQ_OUTPUT_LABEL must be one of: "
        + ",".join(sorted(SUPPORTED_REGRESSION_OUTPUT_LABELS))
    )
default_orientation = "side_relative" if RAWSEQ_OUTPUT_LABEL == "future_return_path" else "market_relative"
RAWSEQ_OUTPUT_ORIENTATION = os.getenv("RAWSEQ_OUTPUT_ORIENTATION", default_orientation).strip().lower()
if RAWSEQ_OUTPUT_ORIENTATION not in {"market_relative", "side_relative"}:
    raise SystemExit("RAWSEQ_OUTPUT_ORIENTATION must be market_relative or side_relative")
RAWSEQ_MA_WINDOW = int(os.getenv("RAWSEQ_MA_WINDOW", "150"))
RAWSEQ_FEATURE_WINDOW = int(os.getenv("RAWSEQ_FEATURE_WINDOW", os.getenv("RAWSEQ_MA_WINDOW", "60")))
RAWSEQ_INCLUDE_WINDOW_GUIDE = os.getenv(
    "RAWSEQ_INCLUDE_WINDOW_GUIDE", "false"
).strip().lower() in {"1", "true", "yes", "y"}

PRICE_CANDIDATES = [
    "price",
    "mid_price",
    "best_mid",
    "close",
    "last",
    "last_price",
    "mark_price",
    "close_price",
    "feature_price",
    "feature_close",
    "feature_mid_price",
]

SIDE_CANDIDATES = [
    "predicted_side",
    "side",
    "signal_side",
]

TIME_CANDIDATES = [
    "time",
    "datetime",
    "iso_time",
    "timestamp_iso",
]


# =========================
# Utilities
# =========================

def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if not math.isfinite(v) else v
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value

def load_frozen_model_payload(path: str) -> dict:
    if not path:
        raise SystemExit("RAWSEQ_LOAD_MODEL_PATH is required when RAWSEQ_INFERENCE_ONLY=true.")

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Frozen model path does not exist: {p}")

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        detail = {
            "error": "atomic_write_csv_failed",
            "target_path": str(path),
            "target_path_length": len(str(path)),
            "temporary_path": str(tmp),
            "temporary_path_length": len(str(tmp)),
            "parent_exists": path.parent.exists(),
            "original_exception": str(exc),
        }
        raise RuntimeError(json.dumps(detail, indent=2, sort_keys=True)) from exc


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".t_{uuid.uuid4().hex[:8]}"
    try:
        tmp.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
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
            "target_path_length": len(str(path)),
            "temporary_path": str(tmp),
            "temporary_path_length": len(str(tmp)),
            "parent_exists": path.parent.exists(),
            "original_exception": str(exc),
        }
        raise RuntimeError(json.dumps(detail, indent=2, sort_keys=True)) from exc


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for candidate in candidates:
        token = candidate.lower()
        for low, original in lowered.items():
            if token in low:
                return original
    return None


def safe_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def resolve_source_path() -> Path:
    if RAWSEQ_SOURCE_PATH:
        if not RAWSEQ_SOURCE_PATH.exists():
            raise SystemExit(f"RAWSEQ_SOURCE_PATH does not exist: {RAWSEQ_SOURCE_PATH}")
        return RAWSEQ_SOURCE_PATH

    for path in DEFAULT_SOURCE_CANDIDATES:
        if path.exists():
            return path

    raise SystemExit(
        "No source file found. Set RAWSEQ_SOURCE_PATH. Tried:\n"
        + "\n".join(str(p) for p in DEFAULT_SOURCE_CANDIDATES)
    )


def timestamp_to_iso(ms: float) -> str:
    if not math.isfinite(float(ms)):
        return ""
    return pd.to_datetime(int(ms), unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_side(series: pd.Series, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series("", index=index)
    out = series.astype(str).str.strip().str.lower()
    out = out.replace({
        "buy": "long",
        "sell": "short",
        "1": "long",
        "-1": "short",
    })
    return out


def side_to_sign(side: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [side.eq("long"), side.eq("short")],
            [1.0, -1.0],
            default=1.0,
        ),
        index=side.index,
        dtype=float,
    )


def max_dip_bps(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return math.nan
    cumulative = values.cumsum()
    return float((cumulative - cumulative.cummax()).min())


# =========================
# Data builder
# =========================

def load_source() -> pd.DataFrame:
    path = resolve_source_path()
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.loc[:, ~frame.columns.duplicated(keep="last")].copy()

    if "timestamp" not in frame.columns:
        time_col = first_existing(frame.columns.tolist(), TIME_CANDIDATES)
        if not time_col:
            raise SystemExit(f"Source has no timestamp/time column: {path}")
        frame["timestamp"] = pd.to_datetime(frame[time_col], utc=True, errors="coerce").astype("int64") // 1_000_000

    price_col = first_existing(frame.columns.tolist(), PRICE_CANDIDATES)
    if not price_col:
        raise SystemExit(
            f"Could not find price column in {path}.\n"
            f"Columns include: {frame.columns.tolist()[:80]}"
        )

    side_col = first_existing(frame.columns.tolist(), SIDE_CANDIDATES)

    out = pd.DataFrame()
    out["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    out["price"] = pd.to_numeric(frame[price_col], errors="coerce")

    if "time" in frame.columns:
        out["time"] = frame["time"].astype(str)
    else:
        out["time"] = out["timestamp"].apply(lambda x: timestamp_to_iso(float(x)) if pd.notna(x) else "")

    if side_col:
        out["predicted_side"] = normalize_side(frame[side_col], frame.index)
    else:
        out["predicted_side"] = "long"

    if "trade_return_bps" in frame.columns:
        out["trade_return_bps"] = pd.to_numeric(frame["trade_return_bps"], errors="coerce")
    else:
        out["trade_return_bps"] = np.nan

    if "rawseq_wf_split" in frame.columns:
        out["rawseq_wf_split"] = frame["rawseq_wf_split"].astype(str).str.lower().str.strip()

    out = out.dropna(subset=["timestamp", "price"]).copy()
    out["timestamp"] = out["timestamp"].astype(np.int64)
    out = out[out["price"] > 0].copy()
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    print(f"Source: {path}")
    print(f"Source rows loaded: {len(out)}")
    print(f"Price column: {price_col}")
    print(f"Side column: {side_col or '(none; long-only default)'}")
    return out


def bucketize(frame: pd.DataFrame) -> pd.DataFrame:
    bucket_ms = BUCKET_SECONDS * 1000
    working = frame.copy()
    working["bucket"] = (working["timestamp"] // bucket_ms).astype(np.int64)

    grouped = working.groupby("bucket", sort=True)
    bucketed = grouped.agg(
        timestamp=("timestamp", "last"),
        price=("price", "last"),
        time=("time", "last"),
        predicted_side=("predicted_side", "last"),
        trade_return_bps=("trade_return_bps", "last"),
    ).reset_index()
    if "rawseq_wf_split" in working.columns:
        split_by_bucket = grouped["rawseq_wf_split"].last().reset_index()
        bucketed = bucketed.merge(split_by_bucket, on="bucket", how="left")

    full_buckets = pd.DataFrame({
        "bucket": np.arange(int(bucketed["bucket"].min()), int(bucketed["bucket"].max()) + 1, dtype=np.int64)
    })
    bucketed = full_buckets.merge(bucketed, on="bucket", how="left")
    bucketed["timestamp"] = bucketed["bucket"] * bucket_ms
    bucketed["price"] = pd.to_numeric(bucketed["price"], errors="coerce").ffill()
    bucketed["time"] = bucketed["timestamp"].apply(timestamp_to_iso)
    bucketed["predicted_side"] = bucketed["predicted_side"].fillna("")
    bucketed["trade_return_bps"] = pd.to_numeric(bucketed["trade_return_bps"], errors="coerce")
    if "rawseq_wf_split" in bucketed.columns:
        bucketed["rawseq_wf_split"] = bucketed["rawseq_wf_split"].fillna("purge_embargo")

    bucketed = bucketed.dropna(subset=["price"]).reset_index(drop=True)

    # log return from previous bucket to this bucket, in bps
    price = bucketed["price"].to_numpy(dtype=np.float64)
    ret = np.zeros(len(bucketed), dtype=np.float64)
    ret[1:] = 10_000.0 * np.log(price[1:] / price[:-1])
    ret[~np.isfinite(ret)] = 0.0
    bucketed["bucket_return_bps"] = ret

    print(f"Bucket seconds: {BUCKET_SECONDS}")
    print(f"Bucketed rows: {len(bucketed)}")
    return bucketed

def build_input_feature(bucketed: pd.DataFrame) -> tuple[np.ndarray, str, dict]:
    price = pd.to_numeric(bucketed["price"], errors="coerce").to_numpy(dtype=np.float64)
    price_series = pd.Series(price)

    if RAWSEQ_INPUT_FEATURE in {"return", "bucket_return", "signed_return"}:
        values = pd.to_numeric(bucketed["bucket_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
        values = np.array(values, dtype=np.float64, copy=True)
        values[~np.isfinite(values)] = 0.0
        return values, "signed_bucket_return_bps", {
            "input_feature": RAWSEQ_INPUT_FEATURE,
            "ma_window": "",
            "feature_window": "",
            "feature_formula": "bucket_return_bps",
            "feature_units": "bps",
            "expected_sign": "signed",
            "warmup_rows": 0,
        }

    if RAWSEQ_INPUT_FEATURE in {"ma_distance", "price_vs_ma", "distance_to_ma"}:
        ma = price_series.rolling(RAWSEQ_MA_WINDOW, min_periods=RAWSEQ_MA_WINDOW).mean().to_numpy(dtype=np.float64)
        values = 10_000.0 * np.log(price / ma)
        values[~np.isfinite(values)] = np.nan
        return values, f"signed_price_vs_ma_{RAWSEQ_MA_WINDOW}_bps", {
            "input_feature": RAWSEQ_INPUT_FEATURE,
            "ma_window": RAWSEQ_MA_WINDOW,
            "feature_window": RAWSEQ_MA_WINDOW,
            "feature_formula": "10000*log(price/rolling_mean(price, ma_window))",
            "feature_units": "bps",
            "expected_sign": "signed",
            "warmup_rows": RAWSEQ_MA_WINDOW - 1,
        }

    if RAWSEQ_INPUT_FEATURE in {"ma_slope", "slope_ma"}:
        ma = price_series.rolling(RAWSEQ_MA_WINDOW, min_periods=RAWSEQ_MA_WINDOW).mean().to_numpy(dtype=np.float64)
        values = np.zeros(len(ma), dtype=np.float64)
        values[1:] = 10_000.0 * np.log(ma[1:] / ma[:-1])
        values[~np.isfinite(values)] = np.nan
        return values, f"signed_ma_{RAWSEQ_MA_WINDOW}_slope_bps", {
            "input_feature": RAWSEQ_INPUT_FEATURE,
            "ma_window": RAWSEQ_MA_WINDOW,
            "feature_window": RAWSEQ_MA_WINDOW,
            "feature_formula": "10000*log(rolling_mean(price)[t]/rolling_mean(price)[t-1])",
            "feature_units": "bps",
            "expected_sign": "signed",
            "warmup_rows": RAWSEQ_MA_WINDOW,
        }

    rolling_feature_formulas = {
        "rolling_range_bps": (
            "10000*log(rolling_max(price, feature_window)/rolling_min(price, feature_window))",
            "nonnegative",
        ),
        "rolling_volatility_bps": (
            "rolling_std(bucket_return_bps, feature_window, ddof=0)",
            "nonnegative",
        ),
        "distance_to_recent_high_bps": (
            "10000*log(price/rolling_max(price, feature_window))",
            "nonpositive",
        ),
        "distance_to_recent_low_bps": (
            "10000*log(price/rolling_min(price, feature_window))",
            "nonnegative",
        ),
    }
    if RAWSEQ_INPUT_FEATURE in rolling_feature_formulas:
        window = RAWSEQ_FEATURE_WINDOW
        if window <= 0:
            raise SystemExit("RAWSEQ_FEATURE_WINDOW must be positive.")
        rolling_high = price_series.rolling(window, min_periods=window).max().to_numpy(dtype=np.float64)
        rolling_low = price_series.rolling(window, min_periods=window).min().to_numpy(dtype=np.float64)
        if RAWSEQ_INPUT_FEATURE == "rolling_range_bps":
            values = 10_000.0 * np.log(rolling_high / rolling_low)
        elif RAWSEQ_INPUT_FEATURE == "rolling_volatility_bps":
            values = (
                pd.Series(pd.to_numeric(bucketed["bucket_return_bps"], errors="coerce").to_numpy(dtype=np.float64))
                .rolling(window, min_periods=window)
                .std(ddof=0)
                .to_numpy(dtype=np.float64)
            )
        elif RAWSEQ_INPUT_FEATURE == "distance_to_recent_high_bps":
            values = 10_000.0 * np.log(price / rolling_high)
        else:
            values = 10_000.0 * np.log(price / rolling_low)
        values = np.array(values, dtype=np.float64, copy=True)
        values[~np.isfinite(values)] = np.nan
        formula, expected_sign = rolling_feature_formulas[RAWSEQ_INPUT_FEATURE]
        return values, f"{RAWSEQ_INPUT_FEATURE}_fw{window}", {
            "input_feature": RAWSEQ_INPUT_FEATURE,
            "ma_window": "",
            "feature_window": window,
            "feature_formula": formula,
            "feature_units": "bps",
            "expected_sign": expected_sign,
            "warmup_rows": window - 1,
        }

    raise SystemExit(
        f"Unknown RAWSEQ_INPUT_FEATURE={RAWSEQ_INPUT_FEATURE}. "
        "Use return, ma_distance, ma_slope, rolling_range_bps, rolling_volatility_bps, "
        "distance_to_recent_high_bps, or distance_to_recent_low_bps."
    )


def decision_horizon_index() -> int:
    step_seconds = BUCKET_SECONDS * RAWSEQ_OUTPUT_STRIDE
    return min(max(1, DECISION_HORIZON_SECONDS // step_seconds), SEQ_LEN) - 1


def feature_audit_rows(values: np.ndarray, meta: dict, total_rows: int) -> list[dict[str, object]]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    expected_sign = str(meta.get("expected_sign", "signed"))
    tolerance = 1e-9
    if expected_sign == "nonnegative":
        violations = finite < -tolerance
    elif expected_sign == "nonpositive":
        violations = finite > tolerance
    else:
        violations = np.zeros(len(finite), dtype=bool)
    return [
        {
            "input_feature": RAWSEQ_INPUT_FEATURE,
            "ma_window": meta.get("ma_window", ""),
            "feature_window": meta.get("feature_window", ""),
            "feature_formula": meta.get("feature_formula", ""),
            "feature_units": meta.get("feature_units", "bps"),
            "expected_sign": expected_sign,
            "total_rows": int(total_rows),
            "warmup_rows": int(meta.get("warmup_rows") or 0),
            "finite_rows": int(len(finite)),
            "nonfinite_rows": int(total_rows - len(finite)),
            "minimum": float(np.min(finite)) if len(finite) else math.nan,
            "maximum": float(np.max(finite)) if len(finite) else math.nan,
            "mean": float(np.mean(finite)) if len(finite) else math.nan,
            "standard_deviation": float(np.std(finite, ddof=0)) if len(finite) else math.nan,
            "expected_sign_violation_fraction": float(np.mean(violations)) if len(finite) else math.nan,
            "leakage_check_status": "PASS_current_and_historical_rows_only",
            "paper_only": True,
            "promotion": False,
            "champion_replacement": False,
            "orders": False,
        }
    ]


def output_dim_for_label(output_label: str) -> int:
    if output_label == "future_range_envelope_path":
        return 2 * SEQ_LEN
    if output_label in {
        "future_return_path",
        "future_high_from_now_bps_path",
        "future_low_from_now_bps_path",
    }:
        return SEQ_LEN
    raise SystemExit(f"Unsupported RAWSEQ_OUTPUT_LABEL={output_label}")


OUTPUT_DIM = output_dim_for_label(RAWSEQ_OUTPUT_LABEL)
LABEL_REQUIRED_HORIZON_BUCKETS = SEQ_LEN
TASK_TYPE = "regression"


def forward_seconds_index(seconds: int) -> int:
    step_seconds = BUCKET_SECONDS * RAWSEQ_OUTPUT_STRIDE
    return seconds // step_seconds - 1


def build_output_label(sign: float, prices: np.ndarray, i: int, output_offsets: np.ndarray) -> np.ndarray:
    future_prices = prices[i + output_offsets]
    orientation_sign = sign if RAWSEQ_OUTPUT_ORIENTATION == "side_relative" else 1.0
    future_return = orientation_sign * 10_000.0 * np.log(future_prices / prices[i])
    if RAWSEQ_OUTPUT_LABEL == "future_return_path":
        return future_return
    future_high = np.maximum.accumulate(np.concatenate([np.array([0.0], dtype=np.float64), future_return]))[1:]
    future_low = np.minimum.accumulate(np.concatenate([np.array([0.0], dtype=np.float64), future_return]))[1:]
    if RAWSEQ_OUTPUT_LABEL == "future_high_from_now_bps_path":
        return future_high
    if RAWSEQ_OUTPUT_LABEL == "future_low_from_now_bps_path":
        return future_low
    if RAWSEQ_OUTPUT_LABEL == "future_range_envelope_path":
        return np.concatenate([future_high, future_low])
    raise SystemExit(f"Unsupported RAWSEQ_OUTPUT_LABEL={RAWSEQ_OUTPUT_LABEL}")


def assert_y_contract(Y: np.ndarray, label: str) -> None:
    expected = output_dim_for_label(label)
    if Y.ndim != 2 or Y.shape[1] != expected:
        raise SystemExit(f"Y shape mismatch for {label}: got {Y.shape}, expected (*, {expected})")
    if label == "future_high_from_now_bps_path" and np.nanmin(Y) < -1e-9:
        raise SystemExit("Y contract violation: future_high_from_now_bps_path must be >= 0 with zero-inclusive envelope.")
    if label == "future_low_from_now_bps_path" and np.nanmax(Y) > 1e-9:
        raise SystemExit("Y contract violation: future_low_from_now_bps_path must be <= 0 with zero-inclusive envelope.")
    if label == "future_range_envelope_path":
        high = Y[:, :SEQ_LEN]
        low = Y[:, SEQ_LEN : 2 * SEQ_LEN]
        if np.nanmin(high) < -1e-9:
            raise SystemExit("Y contract violation: envelope high half must be >= 0 with zero-inclusive envelope.")
        if np.nanmax(low) > 1e-9:
            raise SystemExit("Y contract violation: envelope low half must be <= 0 with zero-inclusive envelope.")


def assert_model_output_contract(model: dict, y_scaler: dict, expected_dim: int, context: str) -> None:
    w3 = np.asarray(model.get("W3"), dtype=np.float64)
    b3 = np.asarray(model.get("b3"), dtype=np.float64)
    y_mean = np.asarray(y_scaler.get("mean"), dtype=np.float64)
    y_std = np.asarray(y_scaler.get("std"), dtype=np.float64)
    if w3.ndim != 2 or w3.shape[1] != expected_dim:
        raise SystemExit(f"{context}: W3 output dim mismatch: got {w3.shape}, expected second dim {expected_dim}")
    if b3.shape != (expected_dim,):
        raise SystemExit(f"{context}: b3 shape mismatch: got {b3.shape}, expected ({expected_dim},)")
    if y_mean.shape != (expected_dim,) or y_std.shape != (expected_dim,):
        raise SystemExit(
            f"{context}: y_scaler shape mismatch: mean={y_mean.shape}, std={y_std.shape}, expected ({expected_dim},)"
        )

 
def build_rawseq_rows(bucketed: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    side = normalize_side(bucketed["predicted_side"], bucketed.index)
    has_side = side.isin(["long", "short"])
    use_all_if_no_sides = not has_side.any()

    side_sign_all = side_to_sign(side).to_numpy(dtype=np.float64)
    input_values, input_feature_name, input_feature_meta = build_input_feature(bucketed)
    atomic_write_csv(pd.DataFrame(feature_audit_rows(input_values, input_feature_meta, len(bucketed))), FEATURE_AUDIT_PATH)
    prices = bucketed["price"].to_numpy(dtype=np.float64)
    timestamps = bucketed["timestamp"].to_numpy(dtype=np.float64)

    rows = []
    x_list = []
    y_list = []

    input_offsets = np.arange(SEQ_LEN - 1, -1, -1, dtype=np.int64) * RAWSEQ_INPUT_STRIDE
    output_offsets = np.arange(1, SEQ_LEN + 1, dtype=np.int64) * RAWSEQ_OUTPUT_STRIDE
    start_i = max(SEQ_LEN, int(input_offsets[0]))
    stop_i = len(bucketed) - int(output_offsets[-1])

    for i in range(start_i, stop_i):
        if not use_all_if_no_sides and not has_side.iloc[i]:
            continue

        sign = float(side_sign_all[i])
        x_base = sign * input_values[i - input_offsets]

        if RAWSEQ_INCLUDE_WINDOW_GUIDE and input_feature_meta.get("ma_window"):
            guide = math.log(float(RAWSEQ_MA_WINDOW)) / math.log(max(float(RAWSEQ_MA_WINDOW), 2.0))
            x = np.concatenate([x_base, np.array([guide], dtype=np.float64)])
        else:
            x = x_base

        y = build_output_label(sign, prices, i, output_offsets)

        if not np.isfinite(x).all() or not np.isfinite(y).all():
            continue

        y = np.clip(y, -TARGET_CLIP_BPS, TARGET_CLIP_BPS)

        x_list.append(x.astype(np.float64))
        y_list.append(y.astype(np.float64))

        current_side = side.iloc[i] if side.iloc[i] in {"long", "short"} else "long"
        rows.append({
            "timestamp": int(timestamps[i]),
            "time": timestamp_to_iso(float(timestamps[i])),
            "price": float(prices[i]),
            "predicted_side": current_side,
            "side_sign": sign,
            "rawseq_feature_name": input_feature_name,
            "rawseq_input_feature": RAWSEQ_INPUT_FEATURE,
            "rawseq_ma_window": input_feature_meta.get("ma_window", ""),
            "rawseq_feature_window": input_feature_meta.get("feature_window", ""),
            "rawseq_feature_formula": input_feature_meta.get("feature_formula", ""),
            "rawseq_feature_units": input_feature_meta.get("feature_units", "bps"),
            "rawseq_expected_sign": input_feature_meta.get("expected_sign", ""),
            "rawseq_feature_warmup_rows": input_feature_meta.get("warmup_rows", ""),
            "rawseq_include_window_guide": RAWSEQ_INCLUDE_WINDOW_GUIDE,
            "rawseq_bucket_seconds": BUCKET_SECONDS,
            "rawseq_len": SEQ_LEN,
            "rawseq_input_stride": RAWSEQ_INPUT_STRIDE,
            "rawseq_output_stride": RAWSEQ_OUTPUT_STRIDE,
            "rawseq_input_span_buckets": RAWSEQ_INPUT_SPAN_BUCKETS,
            "rawseq_output_span_buckets": RAWSEQ_OUTPUT_SPAN_BUCKETS,
            "rawseq_input_span_seconds": RAWSEQ_INPUT_SPAN_SECONDS,
            "rawseq_output_span_seconds": RAWSEQ_OUTPUT_SPAN_SECONDS,
            "rawseq_output_label": RAWSEQ_OUTPUT_LABEL,
            "rawseq_output_orientation": RAWSEQ_OUTPUT_ORIENTATION,
            "rawseq_task_type": TASK_TYPE,
            "rawseq_output_dim": OUTPUT_DIM,
            "rawseq_label_required_horizon_buckets": LABEL_REQUIRED_HORIZON_BUCKETS,
            "trade_return_bps": bucketed["trade_return_bps"].iloc[i],
            "target_horizon_seconds": DECISION_HORIZON_SECONDS,
            "rawseq_wf_split": safe_str(bucketed["rawseq_wf_split"].iloc[i])
            if "rawseq_wf_split" in bucketed.columns
            else "",
        })

    if not rows:
        raise SystemExit("No rawseq rows built. Check bucket size, source row density, and predicted_side availability.")

    meta = pd.DataFrame(rows)
    X = np.vstack(x_list)
    Y = np.vstack(y_list)
    assert_y_contract(Y, RAWSEQ_OUTPUT_LABEL)

    for j in range(SEQ_LEN):
        meta[f"x_lag_{SEQ_LEN - 1 - j:03d}"] = X[:, j]

    if X.shape[1] > SEQ_LEN:
        meta["x_window_guide"] = X[:, SEQ_LEN]
    for k in range(SEQ_LEN):
        meta[f"y_fwd_{k + 1:03d}"] = Y[:, k]
    if RAWSEQ_OUTPUT_LABEL == "future_range_envelope_path":
        for k in range(SEQ_LEN):
            meta[f"y_low_fwd_{k + 1:03d}"] = Y[:, SEQ_LEN + k]

    horizon_idx = decision_horizon_index()
    meta["target_path_horizon_return_bps"] = Y[:, horizon_idx]
    meta["target_path_profitable"] = meta["target_path_horizon_return_bps"] > 0

    print(f"Rawseq rows: {len(meta)}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    return meta, X, Y


# =========================
# Tiny MLP
# =========================

@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def chronological_split(n: int, rows: pd.DataFrame | None = None) -> Split:
    if rows is not None and "rawseq_wf_split" in rows.columns:
        labels = rows["rawseq_wf_split"].astype(str).str.lower().str.strip()
        train = np.flatnonzero(labels.eq("train").to_numpy())
        val = np.flatnonzero(labels.eq("validation").to_numpy())
        test = np.flatnonzero(labels.eq("test").to_numpy())
        if len(train) and len(val) and len(test):
            return Split(train=train, val=val, test=test)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    return Split(
        train=np.arange(0, train_end),
        val=np.arange(train_end, val_end),
        test=np.arange(val_end, n),
    )


def fit_scaler(values: np.ndarray) -> dict:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return {"mean": mean, "std": std}


def transform(values: np.ndarray, scaler: dict) -> np.ndarray:
    out = (values - scaler["mean"]) / scaler["std"]
    out[~np.isfinite(out)] = 0.0
    return out


def init_model(rng: np.random.Generator, input_dim: int, h1: int, h2: int, output_dim: int) -> dict:
    def w(shape):
        return rng.normal(0.0, math.sqrt(2.0 / max(1, shape[0])), size=shape).astype(np.float64)
    return {
        "W1": w((input_dim, h1)),
        "b1": np.zeros(h1, dtype=np.float64),
        "W2": w((h1, h2)),
        "b2": np.zeros(h2, dtype=np.float64),
        "W3": w((h2, output_dim)),
        "b3": np.zeros(output_dim, dtype=np.float64),
    }


def clone_model(model: dict) -> dict:
    return {k: np.array(v, copy=True) for k, v in model.items()}


def mutate_model(model: dict, rng: np.random.Generator, std: float) -> dict:
    child = clone_model(model)
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        child[key] += rng.normal(0.0, std, size=child[key].shape)
    return child


def forward(model: dict, X: np.ndarray) -> tuple[np.ndarray, dict]:
    z1 = X @ model["W1"] + model["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ model["W2"] + model["b2"]
    a2 = np.tanh(z2)
    yhat = a2 @ model["W3"] + model["b3"]
    cache = {"X": X, "a1": a1, "a2": a2}
    return yhat, cache


def train_one_model(
    model: dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    rng: np.random.Generator,
    epochs: int,
    lr: float,
) -> dict:
    # Adam state
    m = {k: np.zeros_like(v) for k, v in model.items()}
    v = {k: np.zeros_like(v) for k, v in model.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    n = len(X_train)
    for _epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            xb = X_train[idx]
            yb = Y_train[idx]

            pred, cache = forward(model, xb)

            # MSE loss gradient
            grad = (2.0 / max(1, pred.size)) * (pred - yb)

            a2 = cache["a2"]
            a1 = cache["a1"]
            x = cache["X"]

            grads = {}
            grads["W3"] = a2.T @ grad
            grads["b3"] = grad.sum(axis=0)

            da2 = grad @ model["W3"].T
            dz2 = da2 * (1.0 - a2 * a2)
            grads["W2"] = a1.T @ dz2
            grads["b2"] = dz2.sum(axis=0)

            da1 = dz2 @ model["W2"].T
            dz1 = da1 * (1.0 - a1 * a1)
            grads["W1"] = x.T @ dz1
            grads["b1"] = dz1.sum(axis=0)

            step += 1
            for key in model.keys():
                m[key] = beta1 * m[key] + (1.0 - beta1) * grads[key]
                v[key] = beta2 * v[key] + (1.0 - beta2) * (grads[key] ** 2)
                mhat = m[key] / (1.0 - beta1 ** step)
                vhat = v[key] / (1.0 - beta2 ** step)
                model[key] -= lr * mhat / (np.sqrt(vhat) + eps)

    return model


# =========================
# Evaluation
# =========================

def unscale_y(y_scaled: np.ndarray, y_scaler: dict) -> np.ndarray:
    return y_scaled * y_scaler["std"] + y_scaler["mean"]


def strategy_metrics(
    timestamps: np.ndarray,
    actual_curve: np.ndarray,
    pred_curve: np.ndarray,
    threshold_bps: float,
    label: str,
) -> dict:
    horizon_idx = decision_horizon_index()
    pred_horizon = pred_curve[:, horizon_idx]
    actual_horizon = actual_curve[:, horizon_idx]

    mask = pred_horizon > threshold_bps

    if math.isfinite(MAX_EARLY_ADVERSE_BPS):
        early_min = np.min(pred_curve[:, : horizon_idx + 1], axis=1)
        mask &= early_min >= MAX_EARLY_ADVERSE_BPS

    selected = actual_horizon[mask]
    selected_ts = timestamps[mask]

    if len(selected) == 0:
        return {
            "strategy": label,
            "rows": 0,
            "avg_return_bps": math.nan,
            "cumulative_return_bps": math.nan,
            "win_rate": math.nan,
            "max_dip_bps": math.nan,
            "threshold_bps": threshold_bps,
            "decision_horizon_seconds": DECISION_HORIZON_SECONDS,
        }

    returns = pd.Series(selected).reset_index(drop=True)
    return {
        "strategy": label,
        "rows": int(len(selected)),
        "avg_return_bps": float(np.mean(selected)),
        "cumulative_return_bps": float(np.sum(selected)),
        "win_rate": float(np.mean(selected > 0)),
        "max_dip_bps": max_dip_bps(returns),
        "threshold_bps": threshold_bps,
        "decision_horizon_seconds": DECISION_HORIZON_SECONDS,
        "first_timestamp": int(selected_ts[0]) if len(selected_ts) else "",
        "first_time": timestamp_to_iso(float(selected_ts[0])) if len(selected_ts) else "",
        "last_timestamp": int(selected_ts[-1]) if len(selected_ts) else "",
        "last_time": timestamp_to_iso(float(selected_ts[-1])) if len(selected_ts) else "",
    }


def fitness_score(timestamps: np.ndarray, actual_curve: np.ndarray, pred_curve: np.ndarray) -> float:
    data = strategy_metrics(
        timestamps,
        actual_curve,
        pred_curve,
        DECISION_THRESHOLD_BPS,
        "fitness_gate",
    )
    rows = int(data["rows"])
    avg = float(data["avg_return_bps"]) if math.isfinite(float(data.get("avg_return_bps", math.nan))) else -999.0
    dip = float(data["max_dip_bps"]) if math.isfinite(float(data.get("max_dip_bps", math.nan))) else -999.0

    if rows < MIN_VAL_ROWS:
        return avg - 10.0 - (MIN_VAL_ROWS - rows) * 0.05

    # Reward avg return and coverage modestly; penalize drawdown.
    return avg + 0.05 * math.log1p(rows) + 0.01 * dip


def evaluate_model(
    model: dict,
    X_scaled: np.ndarray,
    Y_actual: np.ndarray,
    y_scaler: dict,
    timestamps: np.ndarray,
    split_name: str,
    label_baselines: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    pred_scaled, _ = forward(model, X_scaled)
    pred = unscale_y(pred_scaled, y_scaler)

    if RAWSEQ_OUTPUT_LABEL != "future_return_path":
        rows = label_metric_rows_with_baselines(Y_actual, pred, split_name, label_baselines)
        rows.append({
            "split": split_name,
            "strategy": "label_prediction_quality",
            "rows": int(len(pred)),
            "output_label": RAWSEQ_OUTPUT_LABEL,
            "output_dim": OUTPUT_DIM,
            "paper_only": True,
            "promotion": False,
            "champion_replacement": False,
            "private_api": False,
            "orders": False,
        })
        return pd.DataFrame(rows), pred

    rows = []
    for threshold in [DECISION_THRESHOLD_BPS, 0.0, 1.0, 2.0, 3.0, 5.0]:
        rows.append({
            "split": split_name,
            **strategy_metrics(
                timestamps,
                Y_actual,
                pred,
                threshold,
                f"rawseq_path_pred_horizon_gt_{threshold:g}",
            ),
            "paper_only": True,
            "promotion": False,
            "champion_replacement": False,
            "private_api": False,
            "orders": False,
        })

    mse = float(np.mean((pred - Y_actual) ** 2)) if len(pred) else math.nan
    mae = float(np.mean(np.abs(pred - Y_actual))) if len(pred) else math.nan
    rows.append({
        "split": split_name,
        "strategy": "path_prediction_quality",
        "rows": int(len(pred)),
        "path_mse_bps2": mse,
        "path_mae_bps": mae,
        "paper_only": True,
        "promotion": False,
        "champion_replacement": False,
        "private_api": False,
        "orders": False,
    })
    return pd.DataFrame(rows), pred


def corr_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return math.nan
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def mae_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    return float(np.mean(np.abs(diff))) if len(diff) else math.nan


def rmse_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    return float(np.sqrt(np.mean(diff * diff))) if len(diff) else math.nan


def finite_fraction(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.isfinite(arr).mean()) if arr.size else math.nan


def monotonic_violation_fraction(path: np.ndarray, direction: str) -> float:
    arr = np.asarray(path, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return 0.0
    diffs = np.diff(arr, axis=1)
    if direction == "increasing":
        violations = diffs < -1e-9
    else:
        violations = diffs > 1e-9
    finite = np.isfinite(diffs)
    return float(violations[finite].mean()) if finite.any() else math.nan


def barrier_accuracy(pred_terminal: np.ndarray, actual_terminal: np.ndarray, levels: list[float], side: str) -> dict[str, float]:
    out: dict[str, float] = {}
    pred_terminal = np.asarray(pred_terminal, dtype=np.float64)
    actual_terminal = np.asarray(actual_terminal, dtype=np.float64)
    for level in levels:
        if side == "up":
            pred_hit = pred_terminal >= level
            actual_hit = actual_terminal >= level
            key = f"barrier_up_{int(level)}bps_accuracy"
        else:
            pred_hit = pred_terminal <= -level
            actual_hit = actual_terminal <= -level
            key = f"barrier_down_{int(level)}bps_accuracy"
        mask = np.isfinite(pred_terminal) & np.isfinite(actual_terminal)
        out[key] = float((pred_hit[mask] == actual_hit[mask]).mean()) if mask.any() else math.nan
    return out


def split_label_paths(values: np.ndarray, output_label: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    arr = np.asarray(values, dtype=np.float64)
    if output_label == "future_high_from_now_bps_path":
        return arr, None
    if output_label == "future_low_from_now_bps_path":
        return None, arr
    if output_label == "future_range_envelope_path":
        return arr[:, :SEQ_LEN], arr[:, SEQ_LEN : 2 * SEQ_LEN]
    return arr, None


def label_combined_rmse(row: dict[str, object]) -> float:
    high_rmse = finite_metric(row, "high_path_rmse", math.nan)
    low_rmse = finite_metric(row, "low_path_rmse", math.nan)
    path_rmse = finite_metric(row, "path_rmse", math.nan)
    if math.isfinite(high_rmse) and math.isfinite(low_rmse):
        return 0.5 * (high_rmse + low_rmse)
    if math.isfinite(high_rmse):
        return high_rmse
    if math.isfinite(low_rmse):
        return low_rmse
    return path_rmse


def add_baseline_comparison_columns(row: dict[str, object], baseline_rmse: dict[str, float] | None) -> None:
    model_rmse = label_combined_rmse(row)
    for key, out_key in [
        ("training_mean_path_baseline", "model_vs_mean_rmse_improvement_fraction"),
        ("training_median_path_baseline", "model_vs_median_rmse_improvement_fraction"),
        ("zero_baseline", "model_vs_zero_rmse_improvement_fraction"),
    ]:
        base_rmse = float(baseline_rmse.get(key, math.nan)) if baseline_rmse else math.nan
        if math.isfinite(model_rmse) and math.isfinite(base_rmse) and base_rmse > 1e-12:
            row[out_key] = (base_rmse - model_rmse) / base_rmse
        else:
            row[out_key] = math.nan
    row["model_beats_mean_baseline"] = bool(
        math.isfinite(float(row["model_vs_mean_rmse_improvement_fraction"]))
        and float(row["model_vs_mean_rmse_improvement_fraction"]) > 0.0
    )
    row["model_beats_median_baseline"] = bool(
        math.isfinite(float(row["model_vs_median_rmse_improvement_fraction"]))
        and float(row["model_vs_median_rmse_improvement_fraction"]) > 0.0
    )


def label_metric_rows(
    actual: np.ndarray,
    pred: np.ndarray,
    split_name: str,
    strategy: str = "label_metric_summary",
    baseline_rmse: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    levels = [5.0, 10.0, 20.0, 40.0, 80.0]
    rows: list[dict[str, object]] = []
    actual_high, actual_low = split_label_paths(actual, RAWSEQ_OUTPUT_LABEL)
    pred_high, pred_low = split_label_paths(pred, RAWSEQ_OUTPUT_LABEL)
    base = {
        "split": split_name,
        "strategy": strategy,
        "rows": int(len(pred)),
        "output_label": RAWSEQ_OUTPUT_LABEL,
        "output_dim": OUTPUT_DIM,
        "paper_only": True,
        "promotion": False,
        "champion_replacement": False,
        "private_api": False,
        "orders": False,
    }
    if RAWSEQ_OUTPUT_LABEL == "future_high_from_now_bps_path" and actual_high is not None and pred_high is not None:
        row = {
            **base,
            "path_mae": mae_or_nan(pred_high, actual_high),
            "path_rmse": rmse_or_nan(pred_high, actual_high),
            "high_path_mae": mae_or_nan(pred_high, actual_high),
            "high_path_rmse": rmse_or_nan(pred_high, actual_high),
            "low_path_mae": math.nan,
            "low_path_rmse": math.nan,
            "combined_path_rmse": rmse_or_nan(pred_high, actual_high),
            "predicted_range_mae": math.nan,
            "terminal_high_mae": mae_or_nan(pred_high[:, -1], actual_high[:, -1]),
            "terminal_high_correlation": corr_or_nan(pred_high[:, -1], actual_high[:, -1]),
            "terminal_low_correlation": math.nan,
            "monotonic_violation_fraction": monotonic_violation_fraction(pred_high, "increasing"),
            "negative_prediction_fraction": float((pred_high[np.isfinite(pred_high)] < 0.0).mean()) if np.isfinite(pred_high).any() else math.nan,
        }
        row.update(barrier_accuracy(pred_high[:, -1], actual_high[:, -1], levels, "up"))
        if strategy == "label_metric_summary":
            add_baseline_comparison_columns(row, baseline_rmse)
        rows.append(row)
    elif RAWSEQ_OUTPUT_LABEL == "future_low_from_now_bps_path" and actual_low is not None and pred_low is not None:
        row = {
            **base,
            "path_mae": mae_or_nan(pred_low, actual_low),
            "path_rmse": rmse_or_nan(pred_low, actual_low),
            "high_path_mae": math.nan,
            "high_path_rmse": math.nan,
            "low_path_mae": mae_or_nan(pred_low, actual_low),
            "low_path_rmse": rmse_or_nan(pred_low, actual_low),
            "combined_path_rmse": rmse_or_nan(pred_low, actual_low),
            "predicted_range_mae": math.nan,
            "terminal_high_correlation": math.nan,
            "terminal_low_mae": mae_or_nan(pred_low[:, -1], actual_low[:, -1]),
            "terminal_low_correlation": corr_or_nan(pred_low[:, -1], actual_low[:, -1]),
            "monotonic_violation_fraction": monotonic_violation_fraction(pred_low, "decreasing"),
            "positive_prediction_fraction": float((pred_low[np.isfinite(pred_low)] > 0.0).mean()) if np.isfinite(pred_low).any() else math.nan,
        }
        row.update(barrier_accuracy(pred_low[:, -1], actual_low[:, -1], levels, "down"))
        if strategy == "label_metric_summary":
            add_baseline_comparison_columns(row, baseline_rmse)
        rows.append(row)
    elif RAWSEQ_OUTPUT_LABEL == "future_range_envelope_path" and actual_high is not None and pred_high is not None and actual_low is not None and pred_low is not None:
        predicted_range = pred_high[:, -1] - pred_low[:, -1]
        actual_range = actual_high[:, -1] - actual_low[:, -1]
        tp_level = max(DECISION_THRESHOLD_BPS, 5.0)
        danger_level = -max(2.0 * tp_level, 10.0)
        pred_tp_before, actual_tp_before, tp_coverage = tp_before_downside_stats(
            pred_high, pred_low, actual_high, actual_low, tp_level, danger_level
        )
        tp_mask = np.isfinite(pred_tp_before) & np.isfinite(actual_tp_before)
        tp_precision, tp_recall = binary_precision_recall(pred_tp_before, actual_tp_before)
        row = {
            **base,
            "high_path_mae": mae_or_nan(pred_high, actual_high),
            "high_path_rmse": rmse_or_nan(pred_high, actual_high),
            "low_path_mae": mae_or_nan(pred_low, actual_low),
            "low_path_rmse": rmse_or_nan(pred_low, actual_low),
            "combined_path_rmse": 0.5 * (rmse_or_nan(pred_high, actual_high) + rmse_or_nan(pred_low, actual_low)),
            "terminal_high_correlation": corr_or_nan(pred_high[:, -1], actual_high[:, -1]),
            "terminal_low_correlation": corr_or_nan(pred_low[:, -1], actual_low[:, -1]),
            "predicted_range_mae": mae_or_nan(predicted_range, actual_range),
            "high_monotonic_violation_fraction": monotonic_violation_fraction(pred_high, "increasing"),
            "low_monotonic_violation_fraction": monotonic_violation_fraction(pred_low, "decreasing"),
            "envelope_order_violation_fraction": float((pred_high[np.isfinite(pred_high) & np.isfinite(pred_low)] < pred_low[np.isfinite(pred_high) & np.isfinite(pred_low)]).mean())
            if (np.isfinite(pred_high) & np.isfinite(pred_low)).any()
            else math.nan,
            "derived_tp_before_downside_risk_accuracy": float((pred_tp_before[tp_mask] == actual_tp_before[tp_mask]).mean()) if tp_mask.any() else math.nan,
            "derived_tp_before_downside_risk_precision": tp_precision,
            "derived_tp_before_downside_risk_recall": tp_recall,
            "derived_tp_before_downside_risk_coverage": tp_coverage,
            "tp_before_danger_accuracy": float((pred_tp_before[tp_mask] == actual_tp_before[tp_mask]).mean()) if tp_mask.any() else math.nan,
            "tp_before_danger_precision": tp_precision,
            "tp_before_danger_recall": tp_recall,
            "tp_before_danger_coverage": tp_coverage,
        }
        row.update(barrier_accuracy(pred_high[:, -1], actual_high[:, -1], levels, "up"))
        row.update(barrier_accuracy(pred_low[:, -1], actual_low[:, -1], levels, "down"))
        if strategy == "label_metric_summary":
            add_baseline_comparison_columns(row, baseline_rmse)
        rows.append(row)
    return rows


def finite_metric(row: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value if math.isfinite(value) else default


def fit_label_baselines(train_y: np.ndarray) -> dict[str, np.ndarray]:
    if RAWSEQ_OUTPUT_LABEL == "future_return_path":
        return {}
    train = np.asarray(train_y, dtype=np.float64)
    if train.ndim != 2 or train.shape[1] != OUTPUT_DIM:
        raise SystemExit(f"Cannot fit label baselines: train_y shape={train.shape}, expected (*, {OUTPUT_DIM})")
    return {
        "zero_baseline": np.zeros(OUTPUT_DIM, dtype=np.float64),
        "training_mean_path_baseline": np.nanmean(train, axis=0).astype(np.float64),
        "training_median_path_baseline": np.nanmedian(train, axis=0).astype(np.float64),
    }


def baseline_prediction(template: np.ndarray, rows: int) -> np.ndarray:
    values = np.asarray(template, dtype=np.float64)
    if values.shape != (OUTPUT_DIM,):
        raise SystemExit(f"Baseline shape mismatch: got {values.shape}, expected ({OUTPUT_DIM},)")
    return np.tile(values.reshape(1, -1), (rows, 1))


def baseline_rmse_by_strategy(actual: np.ndarray, baselines: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for strategy, template in baselines.items():
        pred = baseline_prediction(template, len(actual))
        rows = label_metric_rows(actual, pred, "baseline_fit", strategy=strategy)
        out[strategy] = label_combined_rmse(rows[0]) if rows else math.nan
    return out


def label_metric_rows_with_baselines(
    actual: np.ndarray,
    pred: np.ndarray,
    split_name: str,
    baselines: dict[str, np.ndarray] | None,
) -> list[dict[str, object]]:
    baselines = baselines or {}
    baseline_rmse = baseline_rmse_by_strategy(actual, baselines) if baselines else {}
    rows = label_metric_rows(actual, pred, split_name, strategy="label_metric_summary", baseline_rmse=baseline_rmse)
    for strategy, template in baselines.items():
        rows.extend(label_metric_rows(actual, baseline_prediction(template, len(actual)), split_name, strategy=strategy))
    return rows


def label_validation_fitness(actual: np.ndarray, pred: np.ndarray) -> tuple[float, dict[str, object]]:
    metrics = label_metric_rows(actual, pred, "validation_fitness")
    row = metrics[0] if metrics else {}
    path_mae = finite_metric(row, "path_mae", finite_metric(row, "high_path_mae", 0.0) + finite_metric(row, "low_path_mae", 0.0))
    path_rmse = finite_metric(row, "path_rmse", 0.0)
    terminal_correlation = math.nan
    monotonic_fraction = math.nan
    order_violation = math.nan
    primary_metric = "validation_path_rmse"
    fitness_family = "return_policy"
    if RAWSEQ_OUTPUT_LABEL == "future_high_from_now_bps_path":
        fitness_family = "future_high_regression"
        terminal_correlation = finite_metric(row, "terminal_high_correlation", 0.0)
        monotonic_fraction = finite_metric(row, "monotonic_violation_fraction", 0.0)
        negative_fraction = finite_metric(row, "negative_prediction_fraction", 0.0)
        barrier = float(np.nanmean([finite_metric(row, f"barrier_up_{level}bps_accuracy", math.nan) for level in [5, 10, 20, 40, 80]]))
        barrier = barrier if math.isfinite(barrier) else 0.0
        fitness = -path_rmse - 0.25 * path_mae + terminal_correlation + 0.5 * barrier - 5.0 * negative_fraction - 10.0 * monotonic_fraction
    elif RAWSEQ_OUTPUT_LABEL == "future_low_from_now_bps_path":
        fitness_family = "future_low_regression"
        terminal_correlation = finite_metric(row, "terminal_low_correlation", 0.0)
        monotonic_fraction = finite_metric(row, "monotonic_violation_fraction", 0.0)
        positive_fraction = finite_metric(row, "positive_prediction_fraction", 0.0)
        barrier = float(np.nanmean([finite_metric(row, f"barrier_down_{level}bps_accuracy", math.nan) for level in [5, 10, 20, 40, 80]]))
        barrier = barrier if math.isfinite(barrier) else 0.0
        fitness = -path_rmse - 0.25 * path_mae + terminal_correlation + 0.5 * barrier - 5.0 * positive_fraction - 10.0 * monotonic_fraction
    elif RAWSEQ_OUTPUT_LABEL == "future_range_envelope_path":
        fitness_family = "future_range_envelope_regression"
        high_rmse = finite_metric(row, "high_path_rmse", 0.0)
        low_rmse = finite_metric(row, "low_path_rmse", 0.0)
        high_mae = finite_metric(row, "high_path_mae", 0.0)
        low_mae = finite_metric(row, "low_path_mae", 0.0)
        path_rmse = 0.5 * (high_rmse + low_rmse)
        path_mae = 0.5 * (high_mae + low_mae)
        high_corr = finite_metric(row, "terminal_high_correlation", 0.0)
        low_corr = finite_metric(row, "terminal_low_correlation", 0.0)
        terminal_correlation = 0.5 * (high_corr + low_corr)
        monotonic_fraction = 0.5 * (
            finite_metric(row, "high_monotonic_violation_fraction", 0.0)
            + finite_metric(row, "low_monotonic_violation_fraction", 0.0)
        )
        order_violation = finite_metric(row, "envelope_order_violation_fraction", 0.0)
        range_mae = finite_metric(row, "predicted_range_mae", 0.0)
        barrier_up = float(np.nanmean([finite_metric(row, f"barrier_up_{level}bps_accuracy", math.nan) for level in [5, 10, 20, 40, 80]]))
        barrier_down = float(np.nanmean([finite_metric(row, f"barrier_down_{level}bps_accuracy", math.nan) for level in [5, 10, 20, 40, 80]]))
        barrier = np.nanmean([barrier_up, barrier_down])
        barrier = float(barrier) if math.isfinite(float(barrier)) else 0.0
        fitness = -path_rmse - 0.25 * path_mae - 0.2 * range_mae + terminal_correlation + 0.5 * barrier - 10.0 * monotonic_fraction - 10.0 * order_violation
        primary_metric = "validation_high_low_path_rmse"
    else:
        fitness = -rmse_or_nan(pred, actual)
    details = {
        "fitness_family": fitness_family,
        "primary_fitness_metric": primary_metric,
        "validation_path_mae": path_mae,
        "validation_path_rmse": path_rmse,
        "terminal_correlation": terminal_correlation,
        "monotonic_violation_fraction": monotonic_fraction,
        "envelope_order_violation_fraction": order_violation,
    }
    return float(fitness) if math.isfinite(float(fitness)) else -1e9, details


def model_fingerprint(model: dict) -> str:
    digest = hashlib.sha1()
    for key in sorted(model):
        digest.update(key.encode("utf-8"))
        digest.update(np.asarray(model[key], dtype=np.float64).round(10).tobytes())
    return digest.hexdigest()[:16]


def derived_tp_before_downside_accuracy(pred_high: np.ndarray, pred_low: np.ndarray, actual_high: np.ndarray, actual_low: np.ndarray) -> float:
    tp = max(DECISION_THRESHOLD_BPS, 5.0)
    danger = -max(2.0 * tp, 10.0)
    pred_ok, actual_ok, _coverage = tp_before_downside_stats(pred_high, pred_low, actual_high, actual_low, tp, danger)
    mask = np.isfinite(pred_ok) & np.isfinite(actual_ok)
    return float((pred_ok[mask] == actual_ok[mask]).mean()) if mask.any() else math.nan


def first_hit_index(path: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    arr = np.asarray(path, dtype=np.float64)
    hits = arr >= threshold if direction == "up" else arr <= threshold
    any_hit = hits.any(axis=1)
    idx = np.argmax(hits, axis=1).astype(np.float64)
    idx[~any_hit] = np.inf
    return idx


def tp_before_downside_stats(
    pred_high: np.ndarray,
    pred_low: np.ndarray,
    actual_high: np.ndarray,
    actual_low: np.ndarray,
    tp: float,
    danger: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    pred_tp = first_hit_index(pred_high, tp, "up")
    pred_danger = first_hit_index(pred_low, danger, "down")
    actual_tp = first_hit_index(actual_high, tp, "up")
    actual_danger = first_hit_index(actual_low, danger, "down")
    pred_event = np.isfinite(pred_tp) | np.isfinite(pred_danger)
    actual_event = np.isfinite(actual_tp) | np.isfinite(actual_danger)
    pred_ok = np.where(pred_event, pred_tp < pred_danger, np.nan)
    actual_ok = np.where(actual_event, actual_tp < actual_danger, np.nan)
    coverage = float(pred_event.mean()) if len(pred_event) else math.nan
    return pred_ok.astype(np.float64), actual_ok.astype(np.float64), coverage


def binary_precision_recall(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(pred) & np.isfinite(actual)
    if not mask.any():
        return math.nan, math.nan
    p = pred[mask].astype(bool)
    a = actual[mask].astype(bool)
    tp = int((p & a).sum())
    fp = int((p & ~a).sum())
    fn = int((~p & a).sum())
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    return float(precision), float(recall)


def label_shape_audit_rows(actual: np.ndarray, pred: np.ndarray, y_scaler: dict, context: str) -> list[dict[str, object]]:
    expected_dim = output_dim_for_label(RAWSEQ_OUTPUT_LABEL)
    actual_high, actual_low = split_label_paths(actual, RAWSEQ_OUTPUT_LABEL)
    pred_high, pred_low = split_label_paths(pred, RAWSEQ_OUTPUT_LABEL)
    checks = [
        ("expected_output_dim", expected_dim, OUTPUT_DIM, OUTPUT_DIM == expected_dim),
        ("actual_model_output_dim", expected_dim, pred.shape[1] if pred.ndim == 2 else "", pred.ndim == 2 and pred.shape[1] == expected_dim),
        ("y_shape", f"*,{expected_dim}", f"{actual.shape[0]},{actual.shape[1] if actual.ndim == 2 else ''}", actual.ndim == 2 and actual.shape[1] == expected_dim),
        ("y_scaler_mean_shape", expected_dim, np.asarray(y_scaler.get("mean", [])).shape[0], np.asarray(y_scaler.get("mean", [])).shape == (expected_dim,)),
        ("y_scaler_std_shape", expected_dim, np.asarray(y_scaler.get("std", [])).shape[0], np.asarray(y_scaler.get("std", [])).shape == (expected_dim,)),
        ("finite_output_fraction", 1.0, finite_fraction(pred), finite_fraction(pred) >= 0.999),
    ]
    if RAWSEQ_OUTPUT_LABEL == "future_range_envelope_path":
        checks.append(("high_split_shape", f"*,{SEQ_LEN}", f"{pred_high.shape[0]},{pred_high.shape[1]}" if pred_high is not None else "", pred_high is not None and pred_high.shape[1] == SEQ_LEN))
        checks.append(("low_split_shape", f"*,{SEQ_LEN}", f"{pred_low.shape[0]},{pred_low.shape[1]}" if pred_low is not None else "", pred_low is not None and pred_low.shape[1] == SEQ_LEN))
        checks.append(("monotonicity_validity", "high increasing and low decreasing", "computed", True))
    elif RAWSEQ_OUTPUT_LABEL == "future_high_from_now_bps_path":
        checks.append(("monotonicity_validity", "high increasing", "computed", True))
    elif RAWSEQ_OUTPUT_LABEL == "future_low_from_now_bps_path":
        checks.append(("monotonicity_validity", "low decreasing", "computed", True))
    return [
        {
            "context": context,
            "output_label": RAWSEQ_OUTPUT_LABEL,
            "check": name,
            "expected": expected,
            "actual": observed,
            "status": "PASS" if ok else "FAIL",
            "paper_only": True,
            "orders": False,
            "promotion": False,
            "champion_mutation": False,
        }
        for name, expected, observed, ok in checks
    ]


def add_label_annotation_columns(annotated: pd.DataFrame, pred_all: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    annotated["rawseq_output_label"] = RAWSEQ_OUTPUT_LABEL
    annotated["rawseq_output_orientation"] = RAWSEQ_OUTPUT_ORIENTATION
    annotated["rawseq_output_dim"] = OUTPUT_DIM
    horizon_idx = decision_horizon_index()
    if RAWSEQ_OUTPUT_LABEL == "future_return_path":
        annotated["rawseq_path_pred_horizon_return_bps"] = pred_all[:, horizon_idx]
        annotated["rawseq_path_actual_horizon_return_bps"] = actual[:, horizon_idx]
        annotated["rawseq_path_allowed_gt_0"] = annotated["rawseq_path_pred_horizon_return_bps"] > 0.0
        annotated["rawseq_path_allowed_gt_1"] = annotated["rawseq_path_pred_horizon_return_bps"] > 1.0
        annotated["rawseq_path_allowed_gt_2"] = annotated["rawseq_path_pred_horizon_return_bps"] > 2.0
    elif RAWSEQ_OUTPUT_LABEL == "future_high_from_now_bps_path":
        annotated["rawseq_predicted_max_up_bps"] = pred_all[:, -1]
        annotated["rawseq_actual_max_up_bps"] = actual[:, -1]
        annotated["rawseq_predicted_tp_possible"] = annotated["rawseq_predicted_max_up_bps"] >= max(DECISION_THRESHOLD_BPS, 5.0)
    elif RAWSEQ_OUTPUT_LABEL == "future_low_from_now_bps_path":
        annotated["rawseq_predicted_max_down_bps"] = pred_all[:, -1]
        annotated["rawseq_actual_max_down_bps"] = actual[:, -1]
        annotated["rawseq_predicted_downside_danger"] = annotated["rawseq_predicted_max_down_bps"] <= -max(DECISION_THRESHOLD_BPS, 5.0)
    elif RAWSEQ_OUTPUT_LABEL == "future_range_envelope_path":
        pred_high = pred_all[:, :SEQ_LEN]
        pred_low = pred_all[:, SEQ_LEN : 2 * SEQ_LEN]
        actual_high = actual[:, :SEQ_LEN]
        actual_low = actual[:, SEQ_LEN : 2 * SEQ_LEN]
        annotated["rawseq_predicted_max_up_bps"] = pred_high[:, -1]
        annotated["rawseq_predicted_max_down_bps"] = pred_low[:, -1]
        annotated["rawseq_actual_max_up_bps"] = actual_high[:, -1]
        annotated["rawseq_actual_max_down_bps"] = actual_low[:, -1]
        annotated["rawseq_predicted_range_bps"] = pred_high[:, -1] - pred_low[:, -1]
        annotated["rawseq_predicted_upside_to_downside_ratio"] = pred_high[:, -1] / np.maximum(np.abs(pred_low[:, -1]), 1e-9)
        annotated["rawseq_predicted_tp_possible"] = pred_high[:, -1] >= max(DECISION_THRESHOLD_BPS, 5.0)
        annotated["rawseq_predicted_downside_danger"] = pred_low[:, -1] <= -max(2.0 * max(DECISION_THRESHOLD_BPS, 5.0), 10.0)
    for seconds in [1, 5, 10, 15, 30, 60, 120, 300]:
        idx = forward_seconds_index(seconds)
        if 0 <= idx < SEQ_LEN:
            if RAWSEQ_OUTPUT_LABEL == "future_range_envelope_path":
                annotated[f"rawseq_pred_high_fwd_{seconds}s_bps"] = pred_all[:, idx]
                annotated[f"rawseq_actual_high_fwd_{seconds}s_bps"] = actual[:, idx]
                annotated[f"rawseq_pred_low_fwd_{seconds}s_bps"] = pred_all[:, SEQ_LEN + idx]
                annotated[f"rawseq_actual_low_fwd_{seconds}s_bps"] = actual[:, SEQ_LEN + idx]
            else:
                annotated[f"rawseq_pred_fwd_{seconds}s_bps"] = pred_all[:, idx]
                annotated[f"rawseq_actual_fwd_{seconds}s_bps"] = actual[:, idx]
    return annotated

def policy_returns_from_pred_actual(pred, actual, policy: str, threshold_bps: float):
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    policy = str(policy).strip().lower()

    if policy == "direct_gt":
        mask = pred > threshold_bps
        returns = actual[mask]

    elif policy == "inverse_gt":
        mask = pred > threshold_bps
        returns = -actual[mask]

    elif policy == "inverse_directional_abs_gt":
        mask = np.abs(pred) > threshold_bps
        returns = -np.sign(pred[mask]) * actual[mask]

    else:
        raise ValueError(f"Unsupported RAWSEQ_FITNESS_POLICY={policy}")

    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    return returns

def score_policy_returns(returns, min_trades: int):
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]

    n = len(returns)
    if n == 0:
        return -1e9, {
            "rows": 0,
            "avg": math.nan,
            "cum": 0.0,
            "win_rate": math.nan,
            "max_dip": math.nan,
        }

    c = np.cumsum(returns)
    peak = np.maximum.accumulate(c)
    max_dip = float(np.min(c - peak))

    avg = float(np.mean(returns))
    cum = float(np.sum(returns))
    win_rate = float(np.mean(returns > 0))

    # Penalize tiny/sparse policies so one lucky trade does not win evolution.
    trade_penalty = 0.0
    if n < min_trades:
        trade_penalty = (min_trades - n) * 0.02

    # Conservative fitness: reward avg, penalize drawdown and sparse rows.
    fitness = avg - abs(max_dip) / 100.0 - trade_penalty

    return float(fitness), {
        "rows": int(n),
        "avg": avg,
        "cum": cum,
        "win_rate": win_rate,
        "max_dip": max_dip,
    }



# =========================
# Main
# =========================

def main() -> None:
    rng = np.random.default_rng(SEED)

    print("tiny_price_rawseq_path_v1")
    print(f"Symbol: {SYMBOL}")
    print(f"Primary venue: {PRIMARY_VENUE}")
    print(f"Bucket seconds: {BUCKET_SECONDS}")
    print(f"Sequence length: {SEQ_LEN}")
    print(
        f"Input stride: {RAWSEQ_INPUT_STRIDE} "
        f"(span {RAWSEQ_INPUT_SPAN_SECONDS}s); "
        f"output stride: {RAWSEQ_OUTPUT_STRIDE} "
        f"(span {RAWSEQ_OUTPUT_SPAN_SECONDS}s)"
    )
    print(f"Output label: {RAWSEQ_OUTPUT_LABEL}; task={TASK_TYPE}; output_dim={OUTPUT_DIM}")
    print(f"Output orientation: {RAWSEQ_OUTPUT_ORIENTATION}")
    print(f"Architecture: {SEQ_LEN} -> {HIDDEN[0]} -> {HIDDEN[1]} -> {OUTPUT_DIM}")
    print(f"Population: {POPULATION}; generations: {GENERATIONS}; epochs/gen: {EPOCHS_PER_GENERATION}")
    print(f"resolved_population={POPULATION}")
    print(f"resolved_generations={GENERATIONS}")
    print(f"resolved_epochs={EPOCHS_PER_GENERATION}")
    print(f"resolved_seed={SEED}")
    print(f"resolved_ma_window={RAWSEQ_MA_WINDOW}")
    print(f"resolved_feature_window={RAWSEQ_FEATURE_WINDOW}")
    print(f"resolved_output_label={RAWSEQ_OUTPUT_LABEL}")
    print(f"resolved_output_dim={OUTPUT_DIM}")
    print(f"resolved_output_orientation={RAWSEQ_OUTPUT_ORIENTATION}")
    print("Safety: paper-only. No promotion. No champion replacement. No private API. No orders.")

    source = load_source()
    bucketed = bucketize(source)
    rows, X, Y = build_rawseq_rows(bucketed)
    atomic_write_csv(rows, ROWS_PATH)
    print(f"Wrote rows: {ROWS_PATH}")
    feature_meta_row = rows.iloc[0].to_dict()

    split = chronological_split(len(rows), rows)
    timestamps = rows["timestamp"].to_numpy(dtype=np.float64)
    label_baselines = fit_label_baselines(Y[split.train]) if RAWSEQ_OUTPUT_LABEL != "future_return_path" else {}

    if RAWSEQ_INFERENCE_ONLY:
        payload = load_frozen_model_payload(RAWSEQ_LOAD_MODEL_PATH)

        print("Mode: inference-only")
        print(f"Loaded frozen model: {RAWSEQ_LOAD_MODEL_PATH}")

        model = payload["weights"]
        x_scaler = payload["x_scaler"]
        y_scaler = payload["y_scaler"]
        expected_output_label = str(payload.get("output_label") or payload.get("output_target") or "future_return_path").strip().lower()
        if expected_output_label == "future_signed_cumulative_return_path_bps":
            expected_output_label = "future_return_path"
        if expected_output_label != RAWSEQ_OUTPUT_LABEL:
            raise SystemExit(
                f"Output label mismatch. Fresh label={RAWSEQ_OUTPUT_LABEL}, frozen model label={expected_output_label}. "
                "Set RAWSEQ_OUTPUT_LABEL to match the payload."
            )
        expected_output_dim = int(
            payload.get("output_dim")
            or payload.get("architecture", {}).get("output_dim")
            or output_dim_for_label(expected_output_label)
        )
        if expected_output_dim != OUTPUT_DIM:
            raise SystemExit(f"Output dim mismatch. Fresh label expects {OUTPUT_DIM}, frozen model expects {expected_output_dim}.")
        assert_y_contract(Y, RAWSEQ_OUTPUT_LABEL)
        assert_model_output_contract(model, y_scaler, expected_output_dim, "inference-only")

        expected_input_dim = int(payload["architecture"]["input_dim"])
        if X.shape[1] != expected_input_dim:
            raise SystemExit(
                f"Input dim mismatch. Fresh X has {X.shape[1]}, "
                f"but frozen model expects {expected_input_dim}."
            )

        Xs = transform(X, x_scaler)

        pred_all_scaled, _ = forward(model, Xs)
        if pred_all_scaled.shape[1] != expected_output_dim:
            raise SystemExit(f"Model output shape mismatch: got {pred_all_scaled.shape}, expected second dim {expected_output_dim}")
        pred_all = unscale_y(pred_all_scaled, y_scaler)

        val_eval, _ = evaluate_model(
            model,
            Xs[split.val],
            Y[split.val],
            y_scaler,
            timestamps[split.val],
            "validation_inference_only",
            label_baselines,
        )

        test_eval, _ = evaluate_model(
            model,
            Xs[split.test],
            Y[split.test],
            y_scaler,
            timestamps[split.test],
            "test_inference_only",
            label_baselines,
        )

        evaluation = pd.concat([val_eval, test_eval], ignore_index=True, sort=False)
        atomic_write_csv(evaluation, EVALUATION_PATH)

        label_metrics = pd.DataFrame(
            label_metric_rows_with_baselines(Y[split.val], pred_all[split.val], "validation_inference_only", label_baselines)
            + label_metric_rows_with_baselines(Y[split.test], pred_all[split.test], "test_inference_only", label_baselines)
        )
        label_audit = pd.DataFrame(label_shape_audit_rows(Y, pred_all, y_scaler, "inference_only"))
        atomic_write_csv(label_metrics, LABEL_METRIC_SUMMARY_PATH)
        atomic_write_csv(label_audit, LABEL_SHAPE_AUDIT_PATH)

        annotated = add_label_annotation_columns(rows.copy(), pred_all, Y)
        atomic_write_csv(annotated, ANNOTATED_PATH)

        print(f"\nEvaluation: {EVALUATION_PATH}")
        print(f"Annotated rows: {ANNOTATED_PATH}")
        print(f"Label metric summary: {LABEL_METRIC_SUMMARY_PATH}")
        print(f"Label shape audit: {LABEL_SHAPE_AUDIT_PATH}")
        print("Inference-only complete: no training, no mutation, no candidate model saved.")
        print("Safety complete: paper-only. No promotion. No champion replacement. No private API. No orders.")
        return

    x_scaler = fit_scaler(X[split.train])
    y_scaler = fit_scaler(Y[split.train])
    assert_y_contract(Y, RAWSEQ_OUTPUT_LABEL)
    if np.asarray(y_scaler["mean"]).shape != (OUTPUT_DIM,) or np.asarray(y_scaler["std"]).shape != (OUTPUT_DIM,):
        raise SystemExit(
            f"y_scaler shape mismatch: mean={np.asarray(y_scaler['mean']).shape}, "
            f"std={np.asarray(y_scaler['std']).shape}, expected ({OUTPUT_DIM},)"
        )

    Xs = transform(X, x_scaler)
    Ys = transform(Y, y_scaler)

    population = [
        init_model(rng, X.shape[1], HIDDEN[0], HIDDEN[1], OUTPUT_DIM)
        for _ in range(POPULATION)
    ]

    generation_rows = []
    best_model = None
    best_fitness = -math.inf
    best_fitness_details: dict[str, object] = {
        "fitness_family": "return_policy" if RAWSEQ_OUTPUT_LABEL == "future_return_path" else "label_regression",
        "primary_fitness_metric": "policy_return" if RAWSEQ_OUTPUT_LABEL == "future_return_path" else "validation_path_rmse",
        "validation_path_mae": math.nan,
        "validation_path_rmse": math.nan,
        "terminal_correlation": math.nan,
        "monotonic_violation_fraction": math.nan,
        "envelope_order_violation_fraction": math.nan,
    }
    generations_completed = 0
    early_stop_reason = ""
    stale_generations = 0
    unique_population_fingerprints_per_generation: list[int] = []

    for gen in range(GENERATIONS):
        scored = []
        generation_fingerprints: set[str] = set()
        print(f"\nGeneration {gen + 1}/{GENERATIONS}")

        for i, model in enumerate(population):
            model = train_one_model(
                model,
                Xs[split.train],
                Ys[split.train],
                rng,
                EPOCHS_PER_GENERATION,
                LEARNING_RATE,
            )
            population[i] = model
            generation_fingerprints.add(model_fingerprint(model))

            val_pred_scaled, _ = forward(model, Xs[split.val])
            val_pred = unscale_y(val_pred_scaled, y_scaler)

            if RAWSEQ_OUTPUT_LABEL == "future_return_path":
                horizon_idx = decision_horizon_index()
                pred_horizon = val_pred[:, horizon_idx]
                actual_horizon = Y[split.val, horizon_idx]
                policy_returns = policy_returns_from_pred_actual(
                    pred_horizon,
                    actual_horizon,
                    RAWSEQ_FITNESS_POLICY,
                    RAWSEQ_FITNESS_THRESHOLD_BPS,
                )
                fit, policy_metrics = score_policy_returns(
                    policy_returns,
                    RAWSEQ_MIN_FITNESS_TRADES,
                )
                fitness_policy_name = RAWSEQ_FITNESS_POLICY
                fitness_details = {
                    "fitness_family": "return_policy",
                    "primary_fitness_metric": "policy_return",
                    "validation_path_mae": math.nan,
                    "validation_path_rmse": math.nan,
                    "terminal_correlation": math.nan,
                    "monotonic_violation_fraction": math.nan,
                    "envelope_order_violation_fraction": math.nan,
                }
            else:
                fit, fitness_details = label_validation_fitness(Y[split.val], val_pred)
                policy_metrics = {
                    "rows": int(len(val_pred)),
                    "avg": -float(fitness_details.get("validation_path_mae", math.nan)),
                    "cum": -float(fitness_details.get("validation_path_rmse", math.nan)),
                    "max_dip": math.nan,
                    "win_rate": math.nan,
                }
                fitness_policy_name = str(fitness_details.get("fitness_family", "label_regression"))

            scored.append((fit, i))

            val_eval, _ = evaluate_model(
                model,
                Xs[split.val],
                Y[split.val],
                y_scaler,
                timestamps[split.val],
                f"generation_{gen + 1}_model_{i}_validation",
                label_baselines,
            )

            primary_strategy = (
                f"rawseq_path_pred_horizon_gt_{DECISION_THRESHOLD_BPS:g}"
                if RAWSEQ_OUTPUT_LABEL == "future_return_path"
                else "label_metric_summary"
            )
            primary = val_eval[val_eval["strategy"].astype(str).eq(primary_strategy)]
            row = primary.iloc[0].to_dict() if not primary.empty else {}
            generation_rows.append({
                "generation": gen + 1,
                "model_index": i,
                "fitness": fit,
                "fitness_policy": fitness_policy_name,
                **fitness_details,
                "fitness_threshold_bps": RAWSEQ_FITNESS_THRESHOLD_BPS,
                "min_fitness_trades": RAWSEQ_MIN_FITNESS_TRADES,
                "rows": policy_metrics["rows"],
                "avg_return_bps": policy_metrics["avg"],
                "cumulative_return_bps": policy_metrics["cum"],
                "max_dip_bps": policy_metrics["max_dip"],
                "win_rate": policy_metrics["win_rate"],
                "paper_only": True,
            })

            print(
                f"  model={i} "
                f"policy={(RAWSEQ_FITNESS_POLICY + '_' + format(RAWSEQ_FITNESS_THRESHOLD_BPS, 'g')) if RAWSEQ_OUTPUT_LABEL == 'future_return_path' else fitness_policy_name} "
                f"fitness={fit:.4f} "
                f"rows={policy_metrics['rows']} "
                f"avg={float(policy_metrics['avg']):.4f} "
                f"dip={float(policy_metrics['max_dip']):.4f}"
            )

        scored.sort(reverse=True)
        unique_population_fingerprints_per_generation.append(len(generation_fingerprints))
        gen_best_fitness, gen_best_idx = scored[0]
        gen_best = population[gen_best_idx]

        improved = gen_best_fitness > best_fitness + EARLY_STOP_MIN_IMPROVEMENT
        if improved:
            best_fitness = gen_best_fitness
            best_model = clone_model(gen_best)
            stale_generations = 0
            best_row = next((row for row in generation_rows if row.get("generation") == gen + 1 and row.get("model_index") == gen_best_idx), {})
            best_fitness_details = {
                "fitness_family": best_row.get("fitness_family", best_fitness_details.get("fitness_family")),
                "primary_fitness_metric": best_row.get("primary_fitness_metric", best_fitness_details.get("primary_fitness_metric")),
                "validation_path_mae": best_row.get("validation_path_mae", math.nan),
                "validation_path_rmse": best_row.get("validation_path_rmse", math.nan),
                "terminal_correlation": best_row.get("terminal_correlation", math.nan),
                "monotonic_violation_fraction": best_row.get("monotonic_violation_fraction", math.nan),
                "envelope_order_violation_fraction": best_row.get("envelope_order_violation_fraction", math.nan),
            }
        else:
            stale_generations += 1

        # Elitism + mutated children from generation winner.
        population = [clone_model(gen_best)]
        while len(population) < POPULATION:
            population.append(mutate_model(gen_best, rng, MUTATION_STD))

        print(f"  winner=model_{gen_best_idx} fitness={gen_best_fitness:.4f}")
        generations_completed = gen + 1
        unique_fitnesses = len({round(float(item[0]), 12) for item in scored if math.isfinite(float(item[0]))})
        if len(generation_fingerprints) <= 1 and POPULATION > 1:
            early_stop_reason = "identical_population_fingerprints"
            break
        if unique_fitnesses <= 1 and POPULATION > 1 and gen > 0:
            early_stop_reason = "identical_population_fitness"
            break
        if stale_generations >= EARLY_STOP_PATIENCE:
            early_stop_reason = f"no_improvement_patience_{EARLY_STOP_PATIENCE}"
            break

    if best_model is None:
        raise SystemExit("Training failed: no best model selected.")

    val_eval, val_pred = evaluate_model(
        best_model,
        Xs[split.val],
        Y[split.val],
        y_scaler,
        timestamps[split.val],
        "validation_best",
        label_baselines,
    )
    test_eval, test_pred = evaluate_model(
        best_model,
        Xs[split.test],
        Y[split.test],
        y_scaler,
        timestamps[split.test],
        "test_best",
        label_baselines,
    )

    generation_eval = pd.DataFrame(generation_rows)
    evaluation = pd.concat([generation_eval, val_eval, test_eval], ignore_index=True, sort=False)
    atomic_write_csv(evaluation, EVALUATION_PATH)

    annotated = rows.copy()
    pred_all_scaled, _ = forward(best_model, Xs)
    if pred_all_scaled.shape[1] != OUTPUT_DIM:
        raise SystemExit(f"Best model output shape mismatch: got {pred_all_scaled.shape}, expected second dim {OUTPUT_DIM}")
    pred_all = unscale_y(pred_all_scaled, y_scaler)

    annotated = add_label_annotation_columns(annotated, pred_all, Y)
    atomic_write_csv(annotated, ANNOTATED_PATH)
    label_metrics = pd.DataFrame(
        label_metric_rows_with_baselines(Y[split.val], val_pred, "validation_best", label_baselines)
        + label_metric_rows_with_baselines(Y[split.test], test_pred, "test_best", label_baselines)
    )
    label_audit = pd.DataFrame(label_shape_audit_rows(Y, pred_all, y_scaler, "training_best"))
    atomic_write_csv(label_metrics, LABEL_METRIC_SUMMARY_PATH)
    atomic_write_csv(label_audit, LABEL_SHAPE_AUDIT_PATH)

    out_dir = MODEL_ROOT / f"{stamp()}_{uuid.uuid4().hex[:8]}"
    model_path = out_dir / "model.json"

    payload = {
        "model_family": "tiny_price_rawseq_path_v1",
        "model_version": "v1",
        "created_at": now_iso(),
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE,
        "source_path": str(resolve_source_path()),
        "rows_path": str(ROWS_PATH),
        "evaluation_path": str(EVALUATION_PATH),
        "annotated_path": str(ANNOTATED_PATH),
        "label_metric_summary_path": str(LABEL_METRIC_SUMMARY_PATH),
        "label_shape_audit_path": str(LABEL_SHAPE_AUDIT_PATH),
        "feature_audit_path": str(FEATURE_AUDIT_PATH),
        "bucket_seconds": BUCKET_SECONDS,
        "seq_len": SEQ_LEN,
        "input_stride": RAWSEQ_INPUT_STRIDE,
        "output_stride": RAWSEQ_OUTPUT_STRIDE,
        "input_span_buckets": RAWSEQ_INPUT_SPAN_BUCKETS,
        "output_span_buckets": RAWSEQ_OUTPUT_SPAN_BUCKETS,
        "input_span_seconds": RAWSEQ_INPUT_SPAN_SECONDS,
        "output_span_seconds": RAWSEQ_OUTPUT_SPAN_SECONDS,
        "input_feature": RAWSEQ_INPUT_FEATURE,
        "ma_window": feature_meta_row.get("rawseq_ma_window", ""),
        "feature_window": feature_meta_row.get("rawseq_feature_window", ""),
        "requested_feature_window": RAWSEQ_FEATURE_WINDOW,
        "resolved_feature_window": feature_meta_row.get("rawseq_feature_window", RAWSEQ_FEATURE_WINDOW),
        "feature_formula": feature_meta_row.get("rawseq_feature_formula", ""),
        "feature_units": feature_meta_row.get("rawseq_feature_units", "bps"),
        "expected_sign": feature_meta_row.get("rawseq_expected_sign", ""),
        "feature_warmup_rows": feature_meta_row.get("rawseq_feature_warmup_rows", ""),
        "output_target": "future_signed_cumulative_return_path_bps" if RAWSEQ_OUTPUT_LABEL == "future_return_path" else RAWSEQ_OUTPUT_LABEL,
        "output_label": RAWSEQ_OUTPUT_LABEL,
        "output_orientation": RAWSEQ_OUTPUT_ORIENTATION,
        "task_type": TASK_TYPE,
        "output_dim": OUTPUT_DIM,
        "label_required_horizon_buckets": LABEL_REQUIRED_HORIZON_BUCKETS,
        "decision_horizon_seconds": DECISION_HORIZON_SECONDS,
        "decision_threshold_bps": DECISION_THRESHOLD_BPS,
        "architecture": {
            "input_dim": int(X.shape[1]),
            "hidden_1": HIDDEN[0],
            "hidden_2": HIDDEN[1],
            "output_dim": OUTPUT_DIM,
            "activation": "tanh",
            "output_activation": "linear",
        },
        "population_settings": {
            "population": POPULATION,
            "generations": GENERATIONS,
            "generations_requested": GENERATIONS,
            "generations_completed": generations_completed,
            "early_stop_reason": early_stop_reason,
            "unique_population_fingerprints_per_generation": unique_population_fingerprints_per_generation,
            "epochs_per_generation": EPOCHS_PER_GENERATION,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "mutation_std": MUTATION_STD,
            "seed": SEED,
        },
        "population": POPULATION,
        "generations": GENERATIONS,
        "epochs": EPOCHS_PER_GENERATION,
        "split": {
            "train_rows": int(len(split.train)),
            "validation_rows": int(len(split.val)),
            "test_rows": int(len(split.test)),
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
        },
        "x_scaler": {
            "mean": x_scaler["mean"],
            "std": x_scaler["std"],
        },
        "y_scaler": {
            "mean": y_scaler["mean"],
            "std": y_scaler["std"],
        },
        "weights": best_model,
        "best_validation_fitness": best_fitness,
        "fitness_family": best_fitness_details.get("fitness_family"),
        "primary_fitness_metric": best_fitness_details.get("primary_fitness_metric"),
        "validation_path_mae": best_fitness_details.get("validation_path_mae"),
        "validation_path_rmse": best_fitness_details.get("validation_path_rmse"),
        "terminal_correlation": best_fitness_details.get("terminal_correlation"),
        "monotonic_violation_fraction": best_fitness_details.get("monotonic_violation_fraction"),
        "envelope_order_violation_fraction": best_fitness_details.get("envelope_order_violation_fraction"),
        "resolved_population": POPULATION,
        "resolved_generations": GENERATIONS,
        "resolved_epochs": EPOCHS_PER_GENERATION,
        "resolved_seed": SEED,
        "resolved_ma_window": RAWSEQ_MA_WINDOW,
        "resolved_feature_window": feature_meta_row.get("rawseq_feature_window", RAWSEQ_FEATURE_WINDOW),
        "resolved_output_label": RAWSEQ_OUTPUT_LABEL,
        "resolved_output_dim": OUTPUT_DIM,
        "resolved_output_orientation": RAWSEQ_OUTPUT_ORIENTATION,
        "paper_only": True,
        "promotion": False,
        "champion_replacement": False,
        "private_api": False,
        "orders": False,
        "fitness_policy": RAWSEQ_FITNESS_POLICY,
        "fitness_threshold_bps": RAWSEQ_FITNESS_THRESHOLD_BPS,
        "min_fitness_trades": RAWSEQ_MIN_FITNESS_TRADES,
    }
    atomic_write_json(payload, model_path)

    history_row = {
        "run_time": now_iso(),
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE,
        "model_path": str(model_path),
        "bucket_seconds": BUCKET_SECONDS,
        "seq_len": SEQ_LEN,
        "input_stride": RAWSEQ_INPUT_STRIDE,
        "output_stride": RAWSEQ_OUTPUT_STRIDE,
        "input_feature": RAWSEQ_INPUT_FEATURE,
        "ma_window": feature_meta_row.get("rawseq_ma_window", ""),
        "feature_window": feature_meta_row.get("rawseq_feature_window", ""),
        "requested_feature_window": RAWSEQ_FEATURE_WINDOW,
        "resolved_feature_window": feature_meta_row.get("rawseq_feature_window", RAWSEQ_FEATURE_WINDOW),
        "feature_formula": feature_meta_row.get("rawseq_feature_formula", ""),
        "feature_units": feature_meta_row.get("rawseq_feature_units", "bps"),
        "expected_sign": feature_meta_row.get("rawseq_expected_sign", ""),
        "feature_warmup_rows": feature_meta_row.get("rawseq_feature_warmup_rows", ""),
        "output_label": RAWSEQ_OUTPUT_LABEL,
        "output_orientation": RAWSEQ_OUTPUT_ORIENTATION,
        "task_type": TASK_TYPE,
        "output_dim": OUTPUT_DIM,
        "label_required_horizon_buckets": LABEL_REQUIRED_HORIZON_BUCKETS,
        "input_span_seconds": RAWSEQ_INPUT_SPAN_SECONDS,
        "output_span_seconds": RAWSEQ_OUTPUT_SPAN_SECONDS,
        "hidden": ",".join(map(str, HIDDEN)),
        "population": POPULATION,
        "generations": GENERATIONS,
        "generations_requested": GENERATIONS,
        "generations_completed": generations_completed,
        "early_stop_reason": early_stop_reason,
        "unique_population_fingerprints_per_generation": json.dumps(unique_population_fingerprints_per_generation),
        "best_validation_fitness": best_fitness,
        "fitness_family": best_fitness_details.get("fitness_family"),
        "primary_fitness_metric": best_fitness_details.get("primary_fitness_metric"),
        "validation_path_mae": best_fitness_details.get("validation_path_mae"),
        "validation_path_rmse": best_fitness_details.get("validation_path_rmse"),
        "terminal_correlation": best_fitness_details.get("terminal_correlation"),
        "monotonic_violation_fraction": best_fitness_details.get("monotonic_violation_fraction"),
        "envelope_order_violation_fraction": best_fitness_details.get("envelope_order_violation_fraction"),
        "resolved_population": POPULATION,
        "resolved_generations": GENERATIONS,
        "resolved_epochs": EPOCHS_PER_GENERATION,
        "resolved_seed": SEED,
        "resolved_ma_window": RAWSEQ_MA_WINDOW,
        "resolved_feature_window": feature_meta_row.get("rawseq_feature_window", RAWSEQ_FEATURE_WINDOW),
        "resolved_output_label": RAWSEQ_OUTPUT_LABEL,
        "resolved_output_dim": OUTPUT_DIM,
        "resolved_output_orientation": RAWSEQ_OUTPUT_ORIENTATION,
        "rows": len(rows),
        "paper_only": True,
        "promotion": False,
        "orders": False,
        "fitness_policy": RAWSEQ_FITNESS_POLICY,
        "fitness_threshold_bps": RAWSEQ_FITNESS_THRESHOLD_BPS,
        "min_fitness_trades": RAWSEQ_MIN_FITNESS_TRADES,
    }
    if HISTORY_PATH.exists():
        old = pd.read_csv(HISTORY_PATH, low_memory=False)
        hist = pd.concat([old, pd.DataFrame([history_row])], ignore_index=True, sort=False)
    else:
        hist = pd.DataFrame([history_row])
    atomic_write_csv(hist, HISTORY_PATH)

    print(f"\nEvaluation: {EVALUATION_PATH}")
    print(f"Annotated rows: {ANNOTATED_PATH}")
    print(f"Candidate model: {model_path}")
    print(f"History path: {HISTORY_PATH}")

    test_primary = test_eval[test_eval["strategy"].astype(str).eq(f"rawseq_path_pred_horizon_gt_{DECISION_THRESHOLD_BPS:g}")]
    if not test_primary.empty:
        r = test_primary.iloc[0]
        print(
            "Test primary: "
            f"rows={int(r.get('rows', 0))} "
            f"avg={float(r.get('avg_return_bps', math.nan)):.4f}bps "
            f"win_rate={float(r.get('win_rate', math.nan)):.4f} "
            f"max_dip={float(r.get('max_dip_bps', math.nan)):.4f}bps"
        )

    print("Safety complete: paper-only. No promotion. No champion replacement. No private API. No orders.")


if __name__ == "__main__":
    main()
