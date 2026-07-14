#!/usr/bin/env python3
"""Shared paper-only rawseq policy scoring helpers.

This module contains no model loading, training, promotion, private API access,
or order placement. It only converts predictions plus realized returns into
explicit gross/cost/net policy metrics.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_POLICIES = {
    "direct_gt",
    "inverse_gt",
    "direct_lt",
    "inverse_lt",
    "gt",
    "pred_gt",
    "lt",
    "pred_lt",
    "inverse_directional_abs_gt",
}


def normalize_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized not in SUPPORTED_POLICIES:
        raise ValueError(
            "policy must be one of: " + ", ".join(sorted(SUPPORTED_POLICIES))
        )
    return normalized


def selected_mask(pred: Any, policy: str, threshold_bps: float) -> np.ndarray:
    pred_arr = np.asarray(pred, dtype=np.float64)
    policy = normalize_policy(policy)
    if policy in {"direct_gt", "inverse_gt", "gt", "pred_gt"}:
        mask = pred_arr > threshold_bps
    elif policy in {"direct_lt", "inverse_lt", "lt", "pred_lt"}:
        mask = pred_arr < -threshold_bps
    elif policy == "inverse_directional_abs_gt":
        mask = np.abs(pred_arr) > threshold_bps
    else:  # pragma: no cover - normalize_policy guards this.
        raise ValueError(f"Unsupported policy={policy}")
    return mask & np.isfinite(pred_arr)


def policy_direction_multiplier(policy: str, pred: Any) -> np.ndarray:
    pred_arr = np.asarray(pred, dtype=np.float64)
    policy = normalize_policy(policy)
    if policy in {"inverse_gt", "inverse_lt"}:
        return np.full(pred_arr.shape, -1.0, dtype=np.float64)
    if policy in {"direct_gt", "direct_lt", "gt", "pred_gt", "lt", "pred_lt"}:
        return np.full(pred_arr.shape, 1.0, dtype=np.float64)
    if policy == "inverse_directional_abs_gt":
        return -np.sign(pred_arr).astype(np.float64)
    raise ValueError(f"Unsupported policy={policy}")


def score_policy_arrays(
    pred: Any,
    actual_bps: Any,
    policy: str,
    threshold_bps: float,
    cost_bps: float = 0.0,
) -> dict[str, np.ndarray]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    actual_arr = np.asarray(actual_bps, dtype=np.float64)
    if pred_arr.shape != actual_arr.shape:
        raise ValueError(f"pred and actual shape mismatch: {pred_arr.shape} != {actual_arr.shape}")
    mask = selected_mask(pred_arr, policy, threshold_bps) & np.isfinite(actual_arr)
    direction = policy_direction_multiplier(policy, pred_arr)
    gross = direction * actual_arr
    net = gross - float(cost_bps)
    return {
        "selected": mask,
        "policy_direction_multiplier": direction,
        "gross_bps": gross,
        "cost_bps": np.full(pred_arr.shape, float(cost_bps), dtype=np.float64),
        "net_bps": net,
    }


def score_policy_frame(
    frame: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    policy: str,
    threshold_bps: float,
    cost_bps: float = 0.0,
    selected_only: bool = True,
) -> pd.DataFrame:
    pred = pd.to_numeric(frame[pred_col], errors="coerce").to_numpy(dtype=np.float64)
    actual = pd.to_numeric(frame[actual_col], errors="coerce").to_numpy(dtype=np.float64)
    scored = score_policy_arrays(pred, actual, policy, threshold_bps, cost_bps)
    out = frame.copy()
    out["selected"] = scored["selected"]
    out["policy_direction_multiplier"] = scored["policy_direction_multiplier"]
    out["gross_bps"] = scored["gross_bps"]
    out["cost_bps"] = scored["cost_bps"]
    out["net_bps"] = scored["net_bps"]
    out = out.replace([np.inf, -np.inf], np.nan)
    if selected_only:
        out = out[out["selected"].astype(bool)].copy()
    out = out.dropna(subset=[pred_col, actual_col, "gross_bps"])
    if "timestamp" in out.columns:
        out = out.sort_values("timestamp")
    return out.reset_index(drop=True)


def max_dip_bps(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan
    cumulative = np.cumsum(arr)
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def payoff_ratio(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    wins = arr[arr > 0.0]
    losses = arr[arr < 0.0]
    if len(wins) == 0 or len(losses) == 0:
        return math.nan
    return float(np.mean(wins) / abs(np.mean(losses)))


def expectancy_metrics(values: Any) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "rows": 0,
            "avg_net_bps": math.nan,
            "cum_net_bps": 0.0,
            "win_rate_net": math.nan,
            "max_dip_net_bps": math.nan,
            "payoff_ratio": math.nan,
        }
    return {
        "rows": float(len(arr)),
        "avg_net_bps": float(np.mean(arr)),
        "cum_net_bps": float(np.sum(arr)),
        "win_rate_net": float(np.mean(arr > 0.0)),
        "max_dip_net_bps": max_dip_bps(arr),
        "payoff_ratio": payoff_ratio(arr),
    }
