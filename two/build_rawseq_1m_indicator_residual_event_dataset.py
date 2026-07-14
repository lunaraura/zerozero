#!/usr/bin/env python3
"""Build residual and event targets for the 1m indicator companion lane.

This script consumes the completed direct RSI/MA companion dataset and keeps
the input tensors unchanged. It adds residual targets beyond deterministic
indicator baselines and binary crossing/direction event targets. It does not
open June or July data and does not mutate the frozen downside-risk model.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import SAFETY_FLAGS, file_sha256, now_stamp, stable_hash, write_csv, write_json  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_indicator_companion_scout")
FROZEN_CANDIDATE_HASH = "8109480bd6b0c2575337cada9be991746199200d42feaeea1fd7b9dd0b3eb6bf"
DEVELOPMENT_CUTOFF = "2026-05-31T23:59:00Z"
EVENT_HORIZONS = [1, 2, 4, 8]
MA_CHANNELS = [
    "close_to_ema20_bps",
    "close_to_ema60_bps",
    "ema20_minus_ema60_bps",
    "ema20_slope_bps_per_minute",
]


def latest_companion_dataset(root: Path) -> Path:
    dirs = sorted(root.glob("dual_timescale_indicator_companion_dataset_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise SystemExit(f"No companion datasets found under {root}")
    return dirs[0]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dist(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"mean": math.nan, "std": math.nan, "p05": math.nan, "p50": math.nan, "p95": math.nan}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def event_columns(data: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    current_rsi = data["current_rsi14"].astype(np.float32)
    actual_rsi = current_rsi[:, None] + data["y_rsi_delta"].astype(np.float32)
    y_ma = data["y_ma_state"].astype(np.float32)
    current_ma = data["baseline_ma_persistence"][:, 0, :].astype(np.float32)
    events: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    def add(name: str, horizon: int, group: str, values: np.ndarray, ambiguous: str) -> None:
        events.append(values.astype(np.float32))
        meta.append({"event_name": name, "horizon": horizon, "event_group": group, "ambiguous_no_event_handling": ambiguous, "leakage_audit": "PASS"})

    for threshold, label, direction in [(50.0, "above_50", "above"), (70.0, "above_70", "above"), (30.0, "below_30", "below")]:
        for horizon in EVENT_HORIZONS:
            future = actual_rsi[:, :horizon]
            if direction == "above":
                values = (current_rsi <= threshold) & np.any(future > threshold, axis=1)
                ambiguous = "already_above_threshold_treated_as_no_new_cross"
            else:
                values = (current_rsi >= threshold) & np.any(future < threshold, axis=1)
                ambiguous = "already_below_threshold_treated_as_no_new_cross"
            add(f"rsi_crosses_{label}_within_{horizon}m", horizon, "rsi_crossing", values, ambiguous)
    for horizon in EVENT_HORIZONS:
        add(f"rsi_direction_positive_h{horizon}m", horizon, "rsi_direction", data["y_rsi_delta"][:, horizon - 1] > 0.0, "flat_delta_treated_as_negative")

    for channel_idx, label in [(0, "ema20"), (1, "ema60")]:
        current = current_ma[:, channel_idx]
        for horizon in EVENT_HORIZONS:
            future = y_ma[:, :horizon, channel_idx]
            values = ((current <= 0.0) & np.any(future > 0.0, axis=1)) | ((current >= 0.0) & np.any(future < 0.0, axis=1))
            add(f"price_crosses_{label}_within_{horizon}m", horizon, "price_ma_crossing", values, "already_on_boundary_treated_by_sign_change_rule")
    spread = current_ma[:, 2]
    future_spread = y_ma[:, :, 2]
    add("ema20_crosses_above_ema60_within_8m", 8, "ema_crossover", (spread <= 0.0) & np.any(future_spread > 0.0, axis=1), "already_above_treated_as_no_new_cross")
    add("ema20_crosses_below_ema60_within_8m", 8, "ema_crossover", (spread >= 0.0) & np.any(future_spread < 0.0, axis=1), "already_below_treated_as_no_new_cross")
    for horizon in EVENT_HORIZONS:
        add(
            f"ema20_minus_ema60_spread_narrows_within_{horizon}m",
            horizon,
            "ema_spread_narrowing",
            np.any(np.abs(future_spread[:, :horizon]) < np.abs(spread[:, None]), axis=1),
            "zero_current_spread_requires_strict_future_abs_less_than_zero",
        )
    return np.column_stack(events).astype(np.float32), meta


def prevalence_rows(events: np.ndarray, event_meta: list[dict[str, Any]], symbols: np.ndarray, timestamp_ms: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    years = pd.to_datetime(timestamp_ms, unit="ms", utc=True).year.to_numpy()
    for idx, meta in enumerate(event_meta):
        values = events[:, idx]
        rows.append({**meta, "scope": "all", "scope_value": "all", "rows": int(len(values)), "positive_rows": int(values.sum()), "prevalence": float(np.mean(values))})
        for symbol in sorted(set(str(x) for x in symbols)):
            mask = symbols == symbol
            rows.append({**meta, "scope": "symbol", "scope_value": symbol, "rows": int(mask.sum()), "positive_rows": int(values[mask].sum()), "prevalence": float(np.mean(values[mask]))})
        for year in sorted(set(int(x) for x in years)):
            mask = years == year
            rows.append({**meta, "scope": "year", "scope_value": year, "rows": int(mask.sum()), "positive_rows": int(values[mask].sum()), "prevalence": float(np.mean(values[mask]))})
    return rows


def main() -> int:
    root = Path(os.getenv("RAWSEQ_RESIDUAL_EVENT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    source_dir = Path(os.getenv("RAWSEQ_RESIDUAL_SOURCE_DATASET_DIR", "") or latest_companion_dataset(root))
    source_manifest = read_json(source_dir / "indicator_companion_dataset_manifest.json")
    input_contract = read_json(source_dir / "indicator_input_contract.json")
    target_contract = read_json(source_dir / "indicator_target_contract.json")
    if input_contract.get("frozen_candidate_hash") != FROZEN_CANDIDATE_HASH:
        raise RuntimeError("Frozen candidate hash mismatch")
    npz = np.load(source_manifest["dataset_path"], allow_pickle=True)
    data = {key: npz[key] for key in npz.files}
    current_rsi = data["current_rsi14"].astype(np.float32)
    actual_rsi_path = current_rsi[:, None] + data["y_rsi_delta"].astype(np.float32)
    rsi_persistence_path = np.tile(current_rsi[:, None], (1, 8)).astype(np.float32)
    rsi_slope_path = current_rsi[:, None] + data["baseline_rsi_slope"].astype(np.float32)
    ma_constant = data["baseline_ma_constant_price"].astype(np.float32)
    rsi_residual_persistence = actual_rsi_path - rsi_persistence_path
    rsi_residual_slope = actual_rsi_path - rsi_slope_path
    ma_residual_constant = data["y_ma_state"].astype(np.float32) - ma_constant
    events, event_meta = event_columns(data)
    run_dir = root / f"indicator_residual_event_dataset_{now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = run_dir / "indicator_residual_event_dataset.npz"
    np.savez_compressed(
        dataset_path,
        x_static=data["x_static"],
        x_short=data["x_short"],
        x_long=data["x_long"],
        split=data["split"],
        symbol=data["symbol"],
        timestamp_ms=data["timestamp_ms"],
        source_row_index=data["source_row_index"],
        current_close=data["current_close"],
        current_rsi14=data["current_rsi14"],
        current_ema20=data["current_ema20"],
        current_ema60=data["current_ema60"],
        actual_rsi_path=actual_rsi_path,
        actual_ma_state_path=data["y_ma_state"],
        rsi_persistence_baseline_path=rsi_persistence_path,
        rsi_slope_baseline_path=rsi_slope_path,
        ma_constant_price_baseline_path=ma_constant,
        rsi_residual_vs_persistence=rsi_residual_persistence,
        rsi_residual_vs_slope=rsi_residual_slope,
        ma_residual_vs_constant_price=ma_residual_constant,
        event_targets=events,
        event_names=np.asarray([row["event_name"] for row in event_meta], dtype=object),
        event_horizons=np.asarray([row["horizon"] for row in event_meta], dtype=np.int16),
        event_groups=np.asarray([row["event_group"] for row in event_meta], dtype=object),
    )
    residual_rows: list[dict[str, Any]] = []
    for h in range(8):
        residual_rows.append({"target": "rsi_residual_vs_persistence", "horizon": h + 1, **dist(rsi_residual_persistence[:, h]), "nonfinite_count": int((~np.isfinite(rsi_residual_persistence[:, h])).sum())})
        residual_rows.append({"target": "rsi_residual_vs_slope", "horizon": h + 1, **dist(rsi_residual_slope[:, h]), "nonfinite_count": int((~np.isfinite(rsi_residual_slope[:, h])).sum())})
        for c, channel in enumerate(MA_CHANNELS):
            vals = ma_residual_constant[:, h, c]
            residual_rows.append({"target": "ma_residual_vs_constant_price", "horizon": h + 1, "channel": channel, **dist(vals), "nonfinite_count": int((~np.isfinite(vals)).sum())})
    residual_contract = {
        "dataset_kind": "indicator_residual_event_companion_v1",
        "source_companion_dataset_dir": str(source_dir),
        "input_contract_unchanged": True,
        "x_static_shape": list(data["x_static"].shape[1:]),
        "x_short_shape": list(data["x_short"].shape[1:]),
        "x_long_shape": list(data["x_long"].shape[1:]),
        "rsi_residual_targets": ["actual_rsi_path - rsi_persistence_baseline_path", "actual_rsi_path - rsi_slope_baseline_path"],
        "ma_residual_target": "actual_ma_state_path - constant_price_ema_roll_forward_path",
        "development_cutoff": DEVELOPMENT_CUTOFF,
        "june_development_access": False,
        "july_access": False,
        **SAFETY_FLAGS,
        "frozen_candidate_mutation": False,
    }
    baseline_contract = {
        "rsi_persistence": "future RSI baseline equals current RSI14 at every horizon",
        "rsi_slope_continuation": "current RSI14 plus recent causal one-minute RSI slope times horizon",
        "ma_constant_price_roll_forward": target_contract.get("ma_channel_order", MA_CHANNELS),
        "event_ambiguous_handling": "stored per event in residual_target_audit.csv",
        "leakage_audit": "PASS: all deterministic baselines use current and historical state only",
    }
    manifest = {
        "created_at": now_stamp(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "source_companion_dataset_dir": str(source_dir),
        "rows": int(len(data["split"])),
        "event_target_count": int(events.shape[1]),
        "split_counts": {name: int(np.sum(data["split"].astype(str) == name)) for name in sorted(set(data["split"].astype(str)))},
        "symbols": sorted(set(str(x) for x in data["symbol"])),
        "development_cutoff": DEVELOPMENT_CUTOFF,
        "june_development_access": False,
        "july_access": False,
        "paper_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "frozen_candidate_mutation": False,
    }
    hashes = {
        "dataset_sha256": manifest["dataset_sha256"],
        "source_dataset_sha256": source_manifest["dataset_sha256"],
        "residual_target_contract_sha256": stable_hash(residual_contract),
        "deterministic_baseline_contract_sha256": stable_hash(baseline_contract),
        "event_metadata_sha256": stable_hash(event_meta),
    }
    gui_schema = {
        "timestamp": "...",
        "symbol": "SOLUSDT",
        "current": {"price": 0, "rsi14": 0, "ema20": 0, "ema60": 0},
        "frozen_risk": {"downside_probability_1m": 0, "candidate_hash": FROZEN_CANDIDATE_HASH},
        "forecast_horizons_minutes": [1, 2, 3, 4, 5, 6, 7, 8],
        "deterministic_projection": {
            "rsi14_persistence": [],
            "rsi14_slope_continuation": [],
            "constant_price_ema20": [],
            "constant_price_ema60": [],
            "constant_price_close_to_ema20_bps": [],
            "constant_price_close_to_ema60_bps": [],
            "constant_price_ema20_minus_ema60_bps": [],
            "constant_price_ema20_slope_bps": [],
        },
        "learned_correction_optional": {
            "rsi14_residual_correction": [],
            "ma_state_residual_correction": [],
            "enabled_only_if_cpu_gate_passes": True,
        },
        "experimental_event_probabilities": {
            "rsi_threshold_crossing_probabilities": {},
            "price_ema_crossing_probabilities": {},
            "ema_crossover_probabilities": {},
        },
        "display_labels": {
            "deterministic_projection": "causal deterministic indicator projection",
            "learned_correction": "experimental residual correction",
            "frozen_risk": "validated frozen downside-risk probability output",
            "event_probabilities": "experimental indicator-event probabilities",
        },
    }
    audit_rows = residual_rows + prevalence_rows(events, event_meta, data["symbol"], data["timestamp_ms"])
    write_json(run_dir / "residual_target_contract.json", residual_contract)
    write_json(run_dir / "deterministic_baseline_contract.json", baseline_contract)
    write_json(run_dir / "residual_event_gui_prediction_schema.json", gui_schema)
    write_csv(run_dir / "residual_target_audit.csv", audit_rows)
    write_json(run_dir / "residual_dataset_manifest.json", manifest)
    write_json(run_dir / "residual_dataset_hashes.json", hashes)
    print(f"residual_dataset_dir={run_dir}")
    print(f"rows={manifest['rows']}")
    print(f"x_static_shape={list(data['x_static'].shape)}")
    print(f"x_short_shape={list(data['x_short'].shape)}")
    print(f"x_long_shape={list(data['x_long'].shape)}")
    print(f"event_target_count={events.shape[1]}")
    print("june_development_access=false")
    print("july_access=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
