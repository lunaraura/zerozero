#!/usr/bin/env python3
"""Run a frozen rawseq research shadow candidate on forward recorded data.

Paper-only forward monitor. It loads a frozen shadow candidate folder from
data/research/rawseq_shadow_candidates, runs inference on public/recorded flow,
appends no model state, places no orders, and writes only forward paper reports.
"""

from __future__ import annotations

import json
import math
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rawseq_policy_scoring import (
    policy_direction_multiplier as shared_policy_direction_multiplier,
    selected_mask as shared_selected_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHADOW_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_shadow_candidates"
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "realtime" / "kraken" / "SOLUSDT_10s_flow.csv"

SHADOW_DIR_ENV = os.getenv("RAWSEQ_SHADOW_DIR", "").strip()
SOURCE_PATH_ENV = os.getenv("RAWSEQ_FORWARD_SOURCE_PATH", str(DEFAULT_SOURCE)).strip()
OUTPUT_DIR_ENV = os.getenv("RAWSEQ_FORWARD_OUTPUT_DIR", "").strip()
LOOKBACK_ROWS_ENV = os.getenv("RAWSEQ_FORWARD_LOOKBACK_ROWS", "").strip()
POLICY_ENV = os.getenv("RAWSEQ_FORWARD_POLICY", "").strip().lower()
COST_BPS = float(os.getenv("RAWSEQ_FORWARD_COST_BPS", "0.1"))
DRY_RUN = os.getenv("RAWSEQ_FORWARD_DRY_RUN", "0").strip().lower() in {"1", "true", "yes"}
REPLAY_MODE = os.getenv("RAWSEQ_FORWARD_REPLAY_MODE", "incremental").strip().lower()
if REPLAY_MODE not in {"incremental", "replay_window"}:
    raise SystemExit("RAWSEQ_FORWARD_REPLAY_MODE must be incremental or replay_window")
RUN_IS_INCREMENTAL = REPLAY_MODE == "incremental"

PRED_COLUMN = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COLUMN = "rawseq_path_actual_horizon_return_bps"
ROLLING_WINDOW_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0]
FROZEN_AT_TIMESTAMP_KEYS = [
    "frozen_at_timestamp_ms",
    "created_at_timestamp_ms",
    "created_timestamp_ms",
]
BOOTSTRAP_SEED = 1729
BOOTSTRAP_SAMPLES = 500


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def timestamp_to_iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat()


def iso_to_timestamp_ms(text: str) -> float:
    text = safe_str(text)
    if not text:
        return math.nan
    try:
        normalized = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return float(parsed.timestamp() * 1000.0)
    except Exception:
        return math.nan


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def safe_int(value: Any, default: int) -> int:
    try:
        return int(float(safe_str(value)))
    except Exception:
        return default


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def latest_shadow_dir() -> Path:
    if SHADOW_DIR_ENV:
        return resolve_path(SHADOW_DIR_ENV)
    if not SHADOW_ROOT.exists():
        raise SystemExit(f"Shadow root not found: {SHADOW_ROOT}")
    candidates = [path for path in SHADOW_ROOT.iterdir() if path.is_dir()]
    if not candidates:
        raise SystemExit(f"No shadow candidate folders found under {SHADOW_ROOT}")
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)[0]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not parse JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root is not an object: {path}")
    return payload


def normalize_side(series: pd.Series, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series("long", index=index)
    text = series.astype(str).str.lower().str.strip()
    mapped = text.replace({"buy": "long", "bid": "long", "sell": "short", "ask": "short"})
    return mapped.where(mapped.isin(["long", "short"]), "")


def side_to_sign(side: pd.Series) -> pd.Series:
    return side.map({"long": 1.0, "short": -1.0}).fillna(1.0)


def infer_column(frame: pd.DataFrame, choices: list[str], label: str) -> str:
    for column in choices:
        if column in frame.columns:
            return column
    raise SystemExit(f"Could not find {label} column. Tried: {choices}")


def load_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Forward source path does not exist: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if LOOKBACK_ROWS_ENV:
        rows = safe_int(LOOKBACK_ROWS_ENV, 0)
        if rows > 0:
            frame = frame.tail(rows).copy()
    timestamp_col = infer_column(frame, ["timestamp", "time_ms", "ts"], "timestamp")
    price_col = infer_column(frame, ["price", "mid_price", "close", "last"], "price")
    frame["timestamp"] = pd.to_numeric(frame[timestamp_col], errors="coerce")
    frame["price"] = pd.to_numeric(frame[price_col], errors="coerce")
    if "time" not in frame.columns:
        frame["time"] = frame["timestamp"].apply(lambda value: timestamp_to_iso(value) if pd.notna(value) else "")
    if "predicted_side" not in frame.columns:
        frame["predicted_side"] = "long"
    return frame.dropna(subset=["timestamp", "price"]).sort_values("timestamp").reset_index(drop=True)


def bucket_flow(frame: pd.DataFrame, bucket_seconds: int) -> pd.DataFrame:
    bucket_ms = bucket_seconds * 1000
    working = frame.copy()
    working["bucket"] = (working["timestamp"] // bucket_ms).astype(np.int64)
    grouped = working.groupby("bucket", sort=True)
    bucketed = grouped.agg(
        timestamp=("timestamp", "last"),
        price=("price", "last"),
        time=("time", "last"),
        predicted_side=("predicted_side", "last"),
    ).reset_index()
    full = pd.DataFrame({"bucket": np.arange(int(bucketed["bucket"].min()), int(bucketed["bucket"].max()) + 1)})
    bucketed = full.merge(bucketed, on="bucket", how="left")
    bucketed["timestamp"] = bucketed["bucket"] * bucket_ms
    bucketed["price"] = pd.to_numeric(bucketed["price"], errors="coerce").ffill()
    bucketed["time"] = bucketed["timestamp"].apply(timestamp_to_iso)
    bucketed["predicted_side"] = bucketed["predicted_side"].fillna("")
    bucketed = bucketed.dropna(subset=["price"]).reset_index(drop=True)
    price = bucketed["price"].to_numpy(dtype=np.float64)
    ret = np.zeros(len(bucketed), dtype=np.float64)
    ret[1:] = 10_000.0 * np.log(price[1:] / price[:-1])
    ret[~np.isfinite(ret)] = 0.0
    bucketed["bucket_return_bps"] = ret
    return bucketed


def build_input_values(bucketed: pd.DataFrame, input_feature: str, ma_window: int) -> np.ndarray:
    price = pd.to_numeric(bucketed["price"], errors="coerce").to_numpy(dtype=np.float64)
    feature = input_feature.lower().strip()
    if feature in {"return", "bucket_return", "signed_return", "signed_bucket_return_bps"}:
        values = pd.to_numeric(bucketed["bucket_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
        values = np.array(values, dtype=np.float64, copy=True)
        values[~np.isfinite(values)] = 0.0
        return values
    if feature in {"ma_distance", "price_vs_ma", "distance_to_ma"}:
        ma = pd.Series(price).rolling(ma_window, min_periods=ma_window).mean().to_numpy(dtype=np.float64)
        values = 10_000.0 * np.log(price / ma)
        values[~np.isfinite(values)] = np.nan
        return values
    if feature in {"ma_slope", "slope_ma"}:
        ma = pd.Series(price).rolling(ma_window, min_periods=ma_window).mean().to_numpy(dtype=np.float64)
        values = np.zeros(len(ma), dtype=np.float64)
        values[1:] = 10_000.0 * np.log(ma[1:] / ma[:-1])
        values[~np.isfinite(values)] = np.nan
        return values
    raise SystemExit(f"Unsupported frozen input_feature={input_feature}")


def as_array(weights: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(weights[key], dtype=np.float64)


def forward(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    a1 = np.tanh(x @ model["W1"] + model["b1"])
    a2 = np.tanh(a1 @ model["W2"] + model["b2"])
    return a2 @ model["W3"] + model["b3"]


def unscale_y(y_scaled: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler.get("mean", 0.0), dtype=np.float64)
    std = np.asarray(scaler.get("std", 1.0), dtype=np.float64)
    return y_scaled * std + mean


def max_dip_bps(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    cumulative = np.cumsum(values)
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def select_mask(pred: np.ndarray, policy: str, threshold: float) -> np.ndarray:
    try:
        return shared_selected_mask(pred, policy, threshold)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def policy_direction_multiplier(policy: str, pred: float | np.ndarray) -> float | np.ndarray:
    try:
        direction = shared_policy_direction_multiplier(policy, np.asarray(pred, dtype=np.float64))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if np.ndim(direction) == 0:
        return float(direction)
    if np.asarray(direction).shape == ():
        return float(direction)
    if np.asarray(direction).size == 1 and np.asarray(pred).shape == ():
        return float(np.asarray(direction).reshape(-1)[0])
    return direction


def score_policy_gross(actual: float, pred: float, policy: str) -> tuple[float, float]:
    if not math.isfinite(actual):
        return math.nan, math.nan
    multiplier = policy_direction_multiplier(policy, pred)
    try:
        multiplier_float = float(multiplier)
    except Exception:
        multiplier_float = math.nan
    if not math.isfinite(multiplier_float):
        return math.nan, math.nan
    return multiplier_float * actual, multiplier_float


def extract_frozen_at_timestamp_ms(provenance: dict[str, Any], payload: dict[str, Any]) -> float:
    candidates: list[Any] = []
    candidates.extend(provenance.get(key) for key in FROZEN_AT_TIMESTAMP_KEYS)
    candidates.extend(payload.get(key) for key in FROZEN_AT_TIMESTAMP_KEYS)
    safety = provenance.get("safety_metadata") if isinstance(provenance.get("safety_metadata"), dict) else {}
    candidates.extend(safety.get(key) for key in FROZEN_AT_TIMESTAMP_KEYS)
    for value in candidates:
        numeric = safe_float(value, math.nan)
        if math.isfinite(numeric):
            return numeric

    for value in [
        provenance.get("frozen_at_iso"),
        provenance.get("created_at"),
        payload.get("frozen_at_iso"),
        payload.get("created_at"),
    ]:
        numeric = iso_to_timestamp_ms(safe_str(value))
        if math.isfinite(numeric):
            return numeric
    return math.nan


def previous_seen_timestamps(output_dir: Path) -> tuple[set[int], int]:
    seen: set[int] = set()
    prior_real_runs = 0
    for path in sorted(output_dir.glob("*/forward_decisions.csv")):
        run_dir = path.parent
        metadata_path = run_dir / "forward_run_metadata.json"
        if metadata_path.exists():
            metadata = load_json(metadata_path)
            if bool(metadata.get("dry_run")):
                continue
            if metadata.get("replay_mode") == "replay_window" or bool(metadata.get("replay")):
                continue
        else:
            summary_path = run_dir / "forward_summary.txt"
            if summary_path.exists() and "Dry run/report mode: True" in summary_path.read_text(encoding="utf-8", errors="ignore"):
                continue
        prior_real_runs += 1
        try:
            for chunk in pd.read_csv(path, usecols=["timestamp"], chunksize=100_000):
                seen.update(pd.to_numeric(chunk["timestamp"], errors="coerce").dropna().astype(np.int64).tolist())
        except Exception:
            continue
    return seen, prior_real_runs


def build_forward_rows(
    bucketed: pd.DataFrame,
    payload: dict[str, Any],
    provenance: dict[str, Any],
    threshold: float,
    policy: str,
    cost_bps: float,
    seen: set[int],
    frozen_at_timestamp_ms: float,
) -> tuple[pd.DataFrame, int]:
    contract = provenance.get("contract") if isinstance(provenance.get("contract"), dict) else {}
    seq_len = safe_int(payload.get("seq_len") or contract.get("seq_len"), 60)
    bucket_seconds = safe_int(payload.get("bucket_seconds") or contract.get("bucket_seconds"), 10)
    input_stride = safe_int(payload.get("input_stride") or contract.get("input_stride"), 1)
    output_stride = safe_int(payload.get("output_stride") or contract.get("output_stride"), 1)
    horizon_seconds = safe_int(payload.get("decision_horizon_seconds"), 30)
    horizon_offset = max(1, horizon_seconds // max(1, bucket_seconds * output_stride)) * output_stride
    input_feature = safe_str(payload.get("input_feature") or contract.get("input_feature"))
    ma_window = safe_int(payload.get("ma_window") or contract.get("ma_window"), 60)

    values = build_input_values(bucketed, input_feature, ma_window)
    side = normalize_side(bucketed["predicted_side"], bucketed.index)
    has_side = side.isin(["long", "short"])
    use_all_if_no_sides = not has_side.any()
    signs = side_to_sign(side).to_numpy(dtype=np.float64)
    prices = bucketed["price"].to_numpy(dtype=np.float64)
    timestamps = bucketed["timestamp"].to_numpy(dtype=np.float64)
    input_offsets = np.arange(seq_len - 1, -1, -1, dtype=np.int64) * input_stride
    start_i = max(seq_len, int(input_offsets[0]))

    weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else {}
    model = {key: as_array(weights, key) for key in ["W1", "b1", "W2", "b2", "W3", "b3"]}
    y_scaler = payload.get("y_scaler") if isinstance(payload.get("y_scaler"), dict) else {"mean": 0.0, "std": 1.0}

    rows: list[dict[str, Any]] = []
    skipped_prior_rows = 0
    skipped_pre_freeze_rows = 0
    for i in range(start_i, len(bucketed)):
        timestamp = int(timestamps[i])
        if RUN_IS_INCREMENTAL and math.isfinite(frozen_at_timestamp_ms) and timestamp <= frozen_at_timestamp_ms:
            skipped_pre_freeze_rows += 1
            continue
        if timestamp in seen:
            skipped_prior_rows += 1
            continue
        if not use_all_if_no_sides and not has_side.iloc[i]:
            continue
        sign = float(signs[i])
        x = sign * values[i - input_offsets]
        if not np.isfinite(x).all():
            continue
        pred_curve = unscale_y(forward(model, x.reshape(1, -1)), y_scaler)[0]
        horizon_idx = min(max(1, horizon_seconds // max(1, bucket_seconds * output_stride)), seq_len) - 1
        pred = float(pred_curve[horizon_idx])
        selected = bool(select_mask(np.asarray([pred]), policy, threshold)[0])
        actual = math.nan
        label_available = i + horizon_offset < len(bucketed)
        if label_available:
            actual = float(sign * 10_000.0 * math.log(prices[i + horizon_offset] / prices[i]))
        gross, direction_multiplier = score_policy_gross(actual, pred, policy) if selected else (math.nan, math.nan)
        net = gross - cost_bps if selected and math.isfinite(gross) else math.nan
        rows.append(
            {
                "timestamp": timestamp,
                "time": timestamp_to_iso(timestamp),
                "price": float(prices[i]),
                "predicted_side": side.iloc[i] if side.iloc[i] in {"long", "short"} else "long",
                "side_sign": sign,
                PRED_COLUMN: pred,
                "policy": policy,
                "threshold_bps": threshold,
                "selected": selected,
                "label_available": label_available,
                ACTUAL_COLUMN: actual,
                "policy_direction_multiplier": direction_multiplier,
                "gross_bps": gross,
                "cost_bps": cost_bps,
                "net_bps": net,
                "evidence_type": "true_forward" if RUN_IS_INCREMENTAL else "replay",
                "replay": not RUN_IS_INCREMENTAL,
                "paper_only": True,
                "orders": False,
                "promotion": False,
                "champion_mutation": False,
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["skipped_pre_freeze_rows"] = skipped_pre_freeze_rows
    return result, skipped_prior_rows


def rolling_summary(labeled_selected: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if labeled_selected.empty:
        return rows
    data = labeled_selected.sort_values("timestamp").copy()
    for hours in ROLLING_WINDOW_HOURS:
        window_ms = int(hours * 60 * 60 * 1000)
        chunks = []
        start = int(data["timestamp"].min())
        end = int(data["timestamp"].max())
        cursor = start
        while cursor <= end:
            stop = cursor + window_ms
            chunk = data[(data["timestamp"] >= cursor) & (data["timestamp"] < stop)]
            if not chunk.empty:
                chunks.append(float(chunk["net_bps"].sum()))
            cursor = stop
        chunks_arr = np.asarray(chunks, dtype=np.float64)
        rows.append(
            {
                "window_hours": hours,
                "windows": int(len(chunks_arr)),
                "positive_windows": int((chunks_arr > 0).sum()) if len(chunks_arr) else 0,
                "positive_fraction": float((chunks_arr > 0).mean()) if len(chunks_arr) else math.nan,
                "total_cum_net_bps": float(chunks_arr.sum()) if len(chunks_arr) else math.nan,
                "worst_window_bps": float(chunks_arr.min()) if len(chunks_arr) else math.nan,
            }
        )
    return rows


def summary_rows(decisions: pd.DataFrame, provenance: dict[str, Any]) -> dict[str, Any]:
    labeled = decisions[decisions["label_available"].astype(bool)].copy() if not decisions.empty else decisions
    selected = labeled[labeled["selected"].astype(bool)].copy() if not labeled.empty else labeled
    net = pd.to_numeric(selected["net_bps"], errors="coerce").dropna().to_numpy(dtype=np.float64) if not selected.empty else np.asarray([])
    gross = pd.to_numeric(selected.get("gross_bps", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=np.float64) if not selected.empty else np.asarray([])
    actual = pd.to_numeric(labeled.get(ACTUAL_COLUMN, pd.Series(dtype=float)), errors="coerce") if not labeled.empty else pd.Series(dtype=float)
    pred = pd.to_numeric(labeled.get(PRED_COLUMN, pd.Series(dtype=float)), errors="coerce") if not labeled.empty else pd.Series(dtype=float)
    registry = provenance.get("metrics") if isinstance(provenance.get("metrics"), dict) else {}
    selected_rows = int(len(selected))
    avg_net = float(np.mean(net)) if len(net) else math.nan
    cum_net = float(np.sum(net)) if len(net) else math.nan
    win_rate = float((net > 0).mean()) if len(net) else math.nan
    max_dip = max_dip_bps(net)
    registry_cum = safe_float(registry.get("fixed_0_10_cum_net"), math.nan)
    forward_vs_registry_cum_ratio = cum_net / registry_cum if math.isfinite(cum_net) and registry_cum > 0 else math.nan
    if selected_rows >= 20 and math.isfinite(cum_net) and cum_net < 0:
        forward_status = "failed_forward"
        preliminary = selected_rows < 100
    elif selected_rows < 100:
        forward_status = "insufficient_sample"
        preliminary = False
    elif math.isfinite(cum_net) and cum_net < 0:
        forward_status = "failed_forward"
        preliminary = False
    elif math.isfinite(forward_vs_registry_cum_ratio) and forward_vs_registry_cum_ratio < 0.25:
        forward_status = "degraded"
        preliminary = False
    elif math.isfinite(cum_net) and cum_net > 0 and (not math.isfinite(max_dip) or abs(max_dip) <= max(cum_net * 2.0, 1e-9)):
        forward_status = "tracking_ok"
        preliminary = False
    else:
        forward_status = "degraded"
        preliminary = False

    return {
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "decision_rows": int(len(decisions)),
        "labeled_rows": int(len(labeled)),
        "selected_rows": selected_rows,
        "avg_gross_bps": float(np.mean(gross)) if len(gross) else math.nan,
        "cum_gross_bps": float(np.sum(gross)) if len(gross) else math.nan,
        "avg_net_bps": avg_net,
        "cumulative_net_bps": cum_net,
        "win_rate_net": win_rate,
        "max_dip_bps": max_dip,
        "directional_accuracy_all_rows": directional_accuracy(pred, actual),
        "signal_precision_net": win_rate,
        "payoff_ratio_gross": payoff_ratio(gross),
        "edge_source": edge_source(win_rate, gross),
        "registry_selected_rows": safe_str(registry.get("selected_rows")),
        "registry_fixed_0_10_cum_net": safe_str(registry.get("fixed_0_10_cum_net")),
        "registry_fixed_0_25_cum_net": safe_str(registry.get("fixed_0_25_cum_net")),
        "registry_half_spread_plus_0_05_cum_net": safe_str(registry.get("half_spread_plus_0_05_cum_net")),
        "registry_conservative_missing_liquidity_penalty_cum_net": safe_str(registry.get("conservative_missing_liquidity_penalty_cum_net")),
        "registry_max_dip_to_cum_net_ratio": safe_str(registry.get("max_dip_to_cum_net_ratio")),
        "registry_positive_12h_window_fraction": safe_str(registry.get("positive_12h_window_fraction")),
        "registry_positive_24h_window_fraction": safe_str(registry.get("positive_24h_window_fraction")),
        "forward_selected_rows": selected_rows,
        "forward_cum_net_bps": cum_net,
        "forward_avg_net_bps": avg_net,
        "forward_win_rate": win_rate,
        "forward_max_dip_bps": max_dip,
        "forward_vs_registry_cum_ratio": forward_vs_registry_cum_ratio,
        "forward_status": forward_status,
        "forward_status_preliminary": preliminary,
    }


def directional_accuracy(pred: pd.Series, actual: pd.Series) -> float:
    frame = pd.DataFrame({"pred": pred, "actual": actual}).replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[(frame["pred"] != 0.0) & (frame["actual"] != 0.0)]
    if frame.empty:
        return math.nan
    return float((np.sign(frame["pred"]) == np.sign(frame["actual"])).mean())


def payoff_ratio(gross: np.ndarray) -> float:
    values = np.asarray(gross, dtype=np.float64)
    values = values[np.isfinite(values)]
    wins = values[values > 0.0]
    losses = values[values < 0.0]
    if len(wins) == 0 or len(losses) == 0:
        return math.nan
    return float(np.mean(wins) / abs(np.mean(losses)))


def edge_source(win_rate: float, gross: np.ndarray) -> str:
    ratio = payoff_ratio(gross)
    if math.isfinite(win_rate) and win_rate >= 0.5:
        return "win_rate"
    if math.isfinite(ratio) and ratio > 1.0:
        return "payoff_magnitude"
    if math.isfinite(win_rate):
        return "mixed_or_unclear"
    return "insufficient_sample"


def bootstrap_avg_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for idx in range(BOOTSTRAP_SAMPLES):
        sample = rng.choice(values, size=len(values), replace=True)
        means[idx] = float(np.mean(sample))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def return_metrics(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    low, high = bootstrap_avg_ci(values)
    return {
        "rows": int(len(values)),
        "avg_net_bps": float(np.mean(values)) if len(values) else math.nan,
        "cum_net_bps": float(np.sum(values)) if len(values) else math.nan,
        "win_rate_net": float(np.mean(values > 0.0)) if len(values) else math.nan,
        "max_dip_net_bps": max_dip_bps(values),
        "avg_net_ci95_low_bps": low,
        "avg_net_ci95_high_bps": high,
    }


def non_overlapping_metrics(labeled_selected: pd.DataFrame, horizon_ms: int) -> pd.DataFrame:
    chosen = []
    next_allowed = -math.inf
    data = labeled_selected.sort_values("timestamp").copy()
    for _, row in data.iterrows():
        timestamp = safe_float(row.get("timestamp"))
        if not math.isfinite(timestamp) or timestamp < next_allowed:
            continue
        net = safe_float(row.get("net_bps"))
        if math.isfinite(net):
            chosen.append(net)
            next_allowed = timestamp + horizon_ms
    metric = return_metrics(np.asarray(chosen, dtype=np.float64))
    return pd.DataFrame(
        [
            {
                "metric_family": "non_overlapping_decision_metrics",
                "cooldown_ms": horizon_ms,
                "selected_rows_before_cooldown": int(len(data)),
                "selected_rows": metric["rows"],
                **{key: value for key, value in metric.items() if key != "rows"},
                "paper_only": True,
                "orders": False,
                "promotion": False,
                "champion_mutation": False,
            }
        ]
    )


def position_sim_metrics(labeled_selected: pd.DataFrame, horizon_ms: int) -> pd.DataFrame:
    exits = []
    holds = []
    exposure_ms = 0.0
    next_flat = -math.inf
    data = labeled_selected.sort_values("timestamp").copy()
    span_ms = safe_float(data["timestamp"].max()) - safe_float(data["timestamp"].min()) if not data.empty else math.nan
    span_ms = max(span_ms, 1.0) if math.isfinite(span_ms) else math.nan
    for _, row in data.iterrows():
        timestamp = safe_float(row.get("timestamp"))
        if not math.isfinite(timestamp) or timestamp < next_flat:
            continue
        net = safe_float(row.get("net_bps"))
        if math.isfinite(net):
            exits.append(net)
            holds.append(horizon_ms)
            exposure_ms += horizon_ms
            next_flat = timestamp + horizon_ms
    metric = return_metrics(np.asarray(exits, dtype=np.float64))
    return pd.DataFrame(
        [
            {
                "metric_family": "position_sim_metrics",
                "position_cost_model": "round_trip_cost_bps_from_forward_cost",
                "trade_count": metric["rows"],
                "selected_rows": metric["rows"],
                **{key: value for key, value in metric.items() if key != "rows"},
                "avg_hold_ms": float(np.mean(holds)) if holds else math.nan,
                "turnover": int(len(exits)),
                "max_concurrent_positions": 1 if exits else 0,
                "exposure_time_fraction": float(min(exposure_ms / span_ms, 1.0)) if math.isfinite(span_ms) else math.nan,
                "paper_only": True,
                "orders": False,
                "promotion": False,
                "champion_mutation": False,
            }
        ]
    )


def write_summary_text(
    path: Path,
    summary: dict[str, Any],
    rolling: list[dict[str, Any]],
    non_overlap: pd.DataFrame,
    position_sim: pd.DataFrame,
    shadow_dir: Path,
    source_path: Path,
    dry_run: bool,
) -> None:
    lines = [
        "Frozen Rawseq Shadow Candidate Forward Paper Run",
        "",
        f"Created at: {now_stamp()}",
        f"Shadow dir: {shadow_dir}",
        f"Source path: {source_path}",
        f"Dry run/report mode: {dry_run}",
        f"Replay mode: {REPLAY_MODE}",
        f"Run is incremental: {RUN_IS_INCREMENTAL}",
        "",
        "Safety:",
        "  paper_only=true",
        "  orders=false",
        "  promotion=false",
        "  champion_mutation=false",
        "",
        "Forward metrics:",
    ]
    for key, value in summary.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Rolling net stability:")
    for row in rolling:
        lines.append(
            "  "
            f"{row['window_hours']:g}h windows={row['windows']} "
            f"positive_fraction={row['positive_fraction']} "
            f"worst_window_bps={row['worst_window_bps']}"
        )
    lines.append("")
    lines.append("Non-overlapping decision metrics:")
    for _, row in non_overlap.iterrows():
        lines.append(
            "  "
            f"rows={row.get('selected_rows')} avg_net={row.get('avg_net_bps')} "
            f"cum_net={row.get('cum_net_bps')} win={row.get('win_rate_net')} "
            f"ci95=[{row.get('avg_net_ci95_low_bps')},{row.get('avg_net_ci95_high_bps')}]"
        )
    lines.append("")
    lines.append("Position simulation metrics:")
    for _, row in position_sim.iterrows():
        lines.append(
            "  "
            f"trades={row.get('trade_count')} avg_net={row.get('avg_net_bps')} "
            f"cum_net={row.get('cum_net_bps')} exposure={row.get('exposure_time_fraction')} "
            f"turnover={row.get('turnover')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    shadow_dir = latest_shadow_dir()
    provenance = load_json(shadow_dir / "provenance.json")
    payload = load_json(shadow_dir / "model.json")
    source_path = resolve_path(SOURCE_PATH_ENV)
    output_root = resolve_path(OUTPUT_DIR_ENV) if OUTPUT_DIR_ENV else shadow_dir / "forward_paper_runs"
    run_dir = output_root / f"forward_paper_{now_stamp()}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)

    threshold = safe_float(provenance.get("threshold_bps") or payload.get("decision_threshold_bps"), 0.0)
    policy = POLICY_ENV or safe_str(provenance.get("selected_policy") or provenance.get("policy") or payload.get("fitness_policy") or "inverse_gt").lower()
    frozen_at_timestamp_ms = extract_frozen_at_timestamp_ms(provenance, payload)
    frozen_at_iso = timestamp_to_iso(frozen_at_timestamp_ms) if math.isfinite(frozen_at_timestamp_ms) else ""

    prior_seen, prior_real_runs_seen = previous_seen_timestamps(output_root)
    seen = prior_seen if RUN_IS_INCREMENTAL else set()
    frame = load_source(source_path)
    contract = provenance.get("contract") if isinstance(provenance.get("contract"), dict) else {}
    bucket_seconds = safe_int(payload.get("bucket_seconds") or contract.get("bucket_seconds"), 10)
    bucketed = bucket_flow(frame, bucket_seconds)
    decisions, skipped_prior_rows = build_forward_rows(
        bucketed,
        payload,
        provenance,
        threshold,
        policy,
        COST_BPS,
        seen,
        frozen_at_timestamp_ms,
    )
    skipped_pre_freeze_rows = int(decisions.attrs.get("skipped_pre_freeze_rows", 0))

    if DRY_RUN and len(decisions) > 5000:
        decisions = decisions.tail(5000).copy()

    labeled = decisions[decisions["label_available"].astype(bool)].copy() if not decisions.empty else decisions.copy()
    selected_labeled = labeled[labeled["selected"].astype(bool)].copy() if not labeled.empty else labeled.copy()
    selected_labeled["equity_bps"] = pd.to_numeric(selected_labeled.get("net_bps", pd.Series(dtype=float)), errors="coerce").fillna(0.0).cumsum()
    horizon_seconds = safe_int(payload.get("decision_horizon_seconds"), 30)
    horizon_ms = horizon_seconds * 1000

    summary = summary_rows(decisions, provenance)
    summary["replay_mode"] = REPLAY_MODE
    summary["replay"] = REPLAY_MODE == "replay_window"
    summary["skipped_prior_rows"] = skipped_prior_rows
    summary["skipped_pre_freeze_rows"] = skipped_pre_freeze_rows
    summary["prior_real_forward_runs_seen"] = prior_real_runs_seen
    summary["run_is_incremental"] = RUN_IS_INCREMENTAL
    summary["evidence_type"] = "true_forward" if RUN_IS_INCREMENTAL else "replay"
    summary["frozen_at_timestamp_ms"] = frozen_at_timestamp_ms
    summary["frozen_at_iso"] = frozen_at_iso
    summary["forward_cutoff_enforced"] = RUN_IS_INCREMENTAL and math.isfinite(frozen_at_timestamp_ms)
    summary["policy_sign_semantics_version"] = "gross_bps_policy_multiplier_v1"
    rolling = rolling_summary(selected_labeled)
    non_overlap = non_overlapping_metrics(selected_labeled, horizon_ms)
    position_sim = position_sim_metrics(selected_labeled, horizon_ms)
    if not non_overlap.empty:
        row = non_overlap.iloc[0]
        summary["non_overlapping_selected_rows"] = row.get("selected_rows")
        summary["non_overlapping_cum_net_bps"] = row.get("cum_net_bps")
        summary["non_overlapping_avg_net_bps"] = row.get("avg_net_bps")
        summary["non_overlapping_win_rate_net"] = row.get("win_rate_net")
        summary["non_overlapping_avg_net_ci95_low_bps"] = row.get("avg_net_ci95_low_bps")
        summary["non_overlapping_avg_net_ci95_high_bps"] = row.get("avg_net_ci95_high_bps")
    if not position_sim.empty:
        row = position_sim.iloc[0]
        summary["position_trade_count"] = row.get("trade_count")
        summary["position_cum_net_bps"] = row.get("cum_net_bps")
        summary["position_avg_net_bps"] = row.get("avg_net_bps")
        summary["position_win_rate_net"] = row.get("win_rate_net")
        summary["position_exposure_time_fraction"] = row.get("exposure_time_fraction")

    decisions.to_csv(run_dir / "forward_decisions.csv", index=False)
    labeled.to_csv(run_dir / "forward_labeled_results.csv", index=False)
    equity_columns = [
        column
        for column in ["timestamp", "time", "gross_bps", "cost_bps", "net_bps", "policy_direction_multiplier", "equity_bps"]
        if column in selected_labeled.columns
    ]
    selected_labeled[equity_columns].to_csv(run_dir / "forward_equity_curve.csv", index=False)
    pd.DataFrame([summary]).to_csv(run_dir / "forward_summary.csv", index=False)
    pd.DataFrame(rolling).to_csv(run_dir / "forward_rolling_summary.csv", index=False)
    non_overlap.to_csv(run_dir / "forward_non_overlapping_decision_metrics.csv", index=False)
    position_sim.to_csv(run_dir / "forward_position_sim_metrics.csv", index=False)
    write_summary_text(run_dir / "forward_summary.txt", summary, rolling, non_overlap, position_sim, shadow_dir, source_path, DRY_RUN)
    (run_dir / "forward_run_metadata.json").write_text(
        json.dumps(
            {
                "created_at": now_stamp(),
                "dry_run": DRY_RUN,
                "replay": REPLAY_MODE == "replay_window",
                "replay_mode": REPLAY_MODE,
                "run_is_incremental": RUN_IS_INCREMENTAL,
                "skipped_prior_rows": skipped_prior_rows,
                "skipped_pre_freeze_rows": skipped_pre_freeze_rows,
                "prior_real_forward_runs_seen": prior_real_runs_seen,
                "frozen_at_timestamp_ms": frozen_at_timestamp_ms,
                "frozen_at_iso": frozen_at_iso,
                "forward_cutoff_enforced": RUN_IS_INCREMENTAL and math.isfinite(frozen_at_timestamp_ms),
                "evidence_type": "true_forward" if RUN_IS_INCREMENTAL else "replay",
                "paper_only": True,
                "orders": False,
                "promotion": False,
                "champion_mutation": False,
                "shadow_dir": str(shadow_dir),
                "source_path": str(source_path),
                "policy": policy,
                "threshold_bps": threshold,
                "cost_bps": COST_BPS,
                "policy_sign_semantics_version": "gross_bps_policy_multiplier_v1",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Frozen shadow candidate forward paper run complete")
    print(f"Shadow dir: {shadow_dir}")
    print(f"Source path: {source_path}")
    print(f"Output dir: {run_dir}")
    print(f"Policy: {policy}")
    print(f"Threshold bps: {threshold:g}")
    print(f"Cost bps: {COST_BPS:g}")
    print(f"Replay mode: {REPLAY_MODE}")
    print(f"Run is incremental: {RUN_IS_INCREMENTAL}")
    print(f"Frozen at timestamp ms: {frozen_at_timestamp_ms}")
    print(f"Prior real forward runs seen: {prior_real_runs_seen}")
    print(f"Skipped prior rows: {skipped_prior_rows}")
    print(f"Skipped pre-freeze rows: {skipped_pre_freeze_rows}")
    print(f"Decision rows: {summary['decision_rows']}")
    print(f"Labeled rows: {summary['labeled_rows']}")
    print(f"Selected rows: {summary['selected_rows']}")
    print("Safety: paper_only=true orders=false promotion=false champion_mutation=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
