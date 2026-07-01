import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from log_hierarchy_forecast_snapshot import (
    LOG_PATH,
    SNAPSHOT_PATH,
    SYMBOL,
    VENUE_TAG,
    read_csv,
    update_realized_outcomes,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = LOG_PATH.with_name(f"{SYMBOL}_hierarchy_forecast_evaluation.csv")
RETURN_COLUMN = os.getenv("HIERARCHY_EVAL_RETURN_COLUMN", "realized_return_60s")
RETURN_COLUMNS = [
    "realized_return_10s",
    "realized_return_30s",
    "realized_return_60s",
]
MIN_BEST_FILTER_ROWS = int(os.getenv("HIERARCHY_MIN_BEST_FILTER_ROWS", "100"))


def numeric(frame, column, default=np.nan):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def percent(value):
    try:
        numeric_value = float(value)
    except Exception:
        return "unavailable"
    if not np.isfinite(numeric_value):
        return "unavailable"
    return f"{numeric_value:.2%}"


def finite_number(value):
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def safe_float(value, default=np.nan):
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if np.isfinite(parsed) else default


def safe_int(value, default=0):
    parsed = safe_float(value, np.nan)
    return int(parsed) if np.isfinite(parsed) else default


def direction_from_return(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values > 0, 1, np.where(values < 0, -1, 0))


def flow_direction(frame):
    labels = frame.get("flow_1s_class", pd.Series("", index=frame.index)).astype(str).str.lower()
    return np.where(labels.str.contains("buy"), 1, np.where(labels.str.contains("sell"), -1, 0))


def flow_pressure_baseline_direction(frame):
    pressure = numeric(frame, "flow_1s_pred_pressure", 0.0).to_numpy(dtype=np.float64)
    return np.where(pressure > 0, 1, np.where(pressure < 0, -1, 0))


def recent_logged_return_baseline_direction(frame):
    working = frame.copy()
    working["timestamp"] = pd.to_numeric(working.get("timestamp"), errors="coerce")
    working["current_mid"] = pd.to_numeric(working.get("current_mid"), errors="coerce")
    working = working.sort_values("timestamp")
    recent_return = working["current_mid"].pct_change()
    direction = pd.Series(
        np.where(recent_return > 0, 1, np.where(recent_return < 0, -1, 0)),
        index=working.index,
    )
    return direction.reindex(frame.index).fillna(0).astype(int)


def add_report_row(rows, section, metric, value, extra=None):
    row = {"section": section, "metric": metric, "value": value}
    if extra:
        row.update(extra)
    rows.append(row)


def baseline_stats(frame):
    stats = {}
    for column in RETURN_COLUMNS:
        values = numeric(frame, column).dropna()
        if len(values) == 0:
            stats[column] = {
                "avg_return": np.nan,
                "win_rate": np.nan,
                "majority_win_rate": np.nan,
                "always_long_win_rate": np.nan,
                "always_short_win_rate": np.nan,
            }
        else:
            long_win_rate = float((values > 0).mean())
            short_win_rate = float((values < 0).mean())
            stats[column] = {
                "avg_return": float(values.mean()),
                "win_rate": long_win_rate,
                "majority_win_rate": max(long_win_rate, short_win_rate),
                "always_long_win_rate": long_win_rate,
                "always_short_win_rate": short_win_rate,
            }
    return stats


def summarize_filter(rows, section, metric, subset, baselines=None, predicted_direction=0):
    baselines = baselines or {}
    primary_values = numeric(subset, RETURN_COLUMN).dropna()
    payload = {
        "rows": int(len(primary_values)),
        "predicted_direction": int(predicted_direction),
    }
    if len(primary_values) == 0:
        add_report_row(rows, section, metric, np.nan, payload)
        return payload

    for column in RETURN_COLUMNS:
        values = numeric(subset.loc[primary_values.index], column).dropna()
        horizon = column.replace("realized_return_", "")
        if len(values) == 0:
            payload[f"avg_return_{horizon}"] = np.nan
            payload[f"win_rate_{horizon}"] = np.nan
            payload[f"baseline_avg_return_{horizon}"] = baselines.get(column, {}).get("avg_return", np.nan)
            payload[f"baseline_win_rate_{horizon}"] = baselines.get(column, {}).get("win_rate", np.nan)
            payload[f"avg_return_improvement_{horizon}"] = np.nan
            payload[f"win_rate_improvement_{horizon}"] = np.nan
            continue
        baseline_avg = baselines.get(column, {}).get("avg_return", np.nan)
        baseline_win = baselines.get(column, {}).get("win_rate", np.nan)
        majority_win = baselines.get(column, {}).get("majority_win_rate", np.nan)
        always_long_win = baselines.get(column, {}).get("always_long_win_rate", np.nan)
        always_short_win = baselines.get(column, {}).get("always_short_win_rate", np.nan)
        win_rate = float((values > 0).mean())
        payload[f"avg_return_{horizon}"] = float(values.mean())
        payload[f"win_rate_{horizon}"] = win_rate
        payload[f"baseline_avg_return_{horizon}"] = baseline_avg
        payload[f"baseline_win_rate_{horizon}"] = baseline_win
        payload[f"majority_baseline_win_rate_{horizon}"] = majority_win
        payload[f"always_long_baseline_win_rate_{horizon}"] = always_long_win
        payload[f"always_short_baseline_win_rate_{horizon}"] = always_short_win
        payload[f"avg_return_improvement_{horizon}"] = (
            float(values.mean() - baseline_avg) if np.isfinite(baseline_avg) else np.nan
        )
        payload[f"win_rate_improvement_{horizon}"] = (
            float(win_rate - baseline_win) if np.isfinite(baseline_win) else np.nan
        )
        payload[f"win_lift_vs_majority_{horizon}"] = (
            float(win_rate - majority_win) if np.isfinite(majority_win) else np.nan
        )
        payload[f"win_lift_vs_always_long_{horizon}"] = (
            float(win_rate - always_long_win) if np.isfinite(always_long_win) else np.nan
        )
        payload[f"win_lift_vs_always_short_{horizon}"] = (
            float(win_rate - always_short_win) if np.isfinite(always_short_win) else np.nan
        )
        if predicted_direction != 0:
            actual = direction_from_return(values.to_numpy(dtype=np.float64))
            payload[f"directional_accuracy_{horizon}"] = float((actual == predicted_direction).mean())
            directional_returns = values * predicted_direction
            payload[f"directional_avg_return_{horizon}"] = float(directional_returns.mean())
            payload[f"directional_win_rate_{horizon}"] = float((directional_returns > 0).mean())

    payload["avg_return"] = payload.get(f"avg_return_{RETURN_COLUMN.replace('realized_return_', '')}", float(primary_values.mean()))
    payload["win_rate_positive_return"] = payload.get(
        f"win_rate_{RETURN_COLUMN.replace('realized_return_', '')}",
        float((primary_values > 0).mean()),
    )
    payload["avg_return_improvement"] = payload.get(
        f"avg_return_improvement_{RETURN_COLUMN.replace('realized_return_', '')}",
        np.nan,
    )
    payload["win_rate_improvement"] = payload.get(
        f"win_rate_improvement_{RETURN_COLUMN.replace('realized_return_', '')}",
        np.nan,
    )
    payload["win_lift_vs_majority"] = payload.get(
        f"win_lift_vs_majority_{RETURN_COLUMN.replace('realized_return_', '')}",
        np.nan,
    )
    payload["win_lift_vs_always_long"] = payload.get(
        f"win_lift_vs_always_long_{RETURN_COLUMN.replace('realized_return_', '')}",
        np.nan,
    )
    payload["win_lift_vs_always_short"] = payload.get(
        f"win_lift_vs_always_short_{RETURN_COLUMN.replace('realized_return_', '')}",
        np.nan,
    )
    payload["avg_max_runup_60s"] = float(numeric(subset.loc[primary_values.index], "realized_max_runup_60s").mean())
    payload["avg_max_drawdown_60s"] = float(numeric(subset.loc[primary_values.index], "realized_max_drawdown_60s").mean())
    add_report_row(rows, section, metric, payload["avg_return"], payload)
    return payload


def summarize_subset(rows, section, metric, subset, direction=None):
    returns = numeric(subset, RETURN_COLUMN).dropna()
    if len(returns) == 0:
        add_report_row(rows, section, metric, np.nan, {"rows": 0})
        return
    payload = {
        "rows": int(len(returns)),
        "avg_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "win_rate_positive_return": float((returns > 0).mean()),
        "avg_max_runup_60s": float(numeric(subset.loc[returns.index], "realized_max_runup_60s").mean()),
        "avg_max_drawdown_60s": float(numeric(subset.loc[returns.index], "realized_max_drawdown_60s").mean()),
    }
    if direction is not None:
        actual = direction_from_return(returns.to_numpy(dtype=np.float64))
        predicted = np.asarray(direction.loc[returns.index], dtype=np.int64)
        active = predicted != 0
        payload["directional_rows"] = int(active.sum())
        payload["directional_accuracy"] = float((predicted[active] == actual[active]).mean()) if active.any() else np.nan
    add_report_row(rows, section, metric, payload["avg_return"], payload)


def summarize_direction_strategy(rows, section, metric, frame, predicted_direction):
    observed = numeric(frame, RETURN_COLUMN).dropna()
    if len(observed) == 0:
        add_report_row(rows, section, metric, np.nan, {"rows": 0})
        return
    predicted = pd.Series(predicted_direction, index=frame.index).reindex(observed.index).fillna(0).astype(int)
    actual = pd.Series(direction_from_return(observed.to_numpy(dtype=np.float64)), index=observed.index)
    active = predicted != 0
    strategy_returns = observed * predicted
    strategy_returns = strategy_returns.where(active, 0.0)
    payload = {
        "rows": int(len(observed)),
        "active_rows": int(active.sum()),
        "avg_return": float(strategy_returns.mean()),
        "median_return": float(strategy_returns.median()),
        "win_rate_positive_return": float((strategy_returns[active] > 0).mean()) if active.any() else np.nan,
        "directional_accuracy": float((predicted[active] == actual[active]).mean()) if active.any() else np.nan,
        "avg_observed_return": float(observed.mean()),
    }
    add_report_row(rows, section, metric, payload["avg_return"], payload)


def bucket_label(value, buckets):
    for low, high in buckets:
        if value >= low and value < high:
            return f"{low:.2f}-{high:.2f}"
    if value >= buckets[-1][1]:
        return f">={buckets[-1][1]:.2f}"
    return f"<{buckets[0][0]:.2f}"


def summarize_buckets(rows, section, frame, column, buckets):
    values = numeric(frame, column)
    working = frame.copy()
    working["_bucket"] = [
        bucket_label(value, buckets) if np.isfinite(value) else "missing"
        for value in values
    ]
    for label, subset in working.groupby("_bucket", sort=False):
        summarize_subset(rows, section, str(label), subset)


def abstain_contains(frame, pattern):
    reasons = frame.get("abstain_reason", pd.Series("", index=frame.index)).fillna("").astype(str)
    return reasons.str.contains(pattern, case=False, regex=True)


def micro_upside_score(frame):
    return numeric(frame, "micro_upside_scare_prob", 0.0)


def micro_downside_score(frame):
    return numeric(frame, "micro_downside_scare_prob", 0.0)


def max_liquidity_drop_score(frame):
    return pd.concat(
        [
            numeric(frame, "micro_bid_liquidity_drop_prob", 0.0),
            numeric(frame, "micro_ask_liquidity_drop_prob", 0.0),
        ],
        axis=1,
    ).max(axis=1)


def max_micro_scare_score(frame):
    return pd.concat(
        [
            micro_upside_score(frame),
            micro_downside_score(frame),
            numeric(frame, "micro_scare_score", 0.0),
        ],
        axis=1,
    ).max(axis=1)


def flow_class_pressure_disagreement_masks(frame):
    labels = frame.get("flow_1s_class", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    pressure = numeric(frame, "flow_1s_pred_pressure", 0.0)
    neutral_high_pressure = labels.eq("neutral") & (pressure.abs() >= 0.50)
    buy_negative = labels.eq("buy_dominant") & (pressure < 0)
    sell_positive = labels.eq("sell_dominant") & (pressure > 0)
    disagreement = neutral_high_pressure | buy_negative | sell_positive
    return {
        "1s_class_pressure_agree": ~disagreement,
        "1s_class_pressure_disagree": disagreement,
        "neutral_class_high_pressure": neutral_high_pressure,
    }


def add_abstain_reason_filters(rows, evaluated, baselines):
    filters = [
        ("spread_expansion_risk_high", abstain_contains(evaluated, "spread_expansion_risk_high")),
        ("liquidity_risk_high", abstain_contains(evaluated, "liquidity(?:_drop)?_risk_high")),
        ("3m_optional_unavailable_or_stale", abstain_contains(evaluated, "3m_optional_unavailable_or_stale")),
        ("regime_optional_unavailable_or_stale", abstain_contains(evaluated, "(?:15m|30m)_optional_unavailable_or_stale")),
        ("htf_optional_unavailable_or_stale", abstain_contains(evaluated, "htf_optional_unavailable_or_stale")),
        ("no_abstain_reason", evaluated.get("abstain_reason", pd.Series("", index=evaluated.index)).fillna("").astype(str).str.len() == 0),
    ]
    for label, mask in filters:
        summarize_filter(rows, "abstain_reason", label, evaluated[mask], baselines)


def add_directional_filters(rows, evaluated, baselines):
    pressure = numeric(evaluated, "flow_1s_pred_pressure", 0.0)
    up = micro_upside_score(evaluated)
    down = micro_downside_score(evaluated)
    filters = [
        ("micro_upside_scare_gt_downside", up > down, 1),
        ("micro_downside_scare_gt_upside", down > up, -1),
        ("flow_1s_pred_pressure_gt_0", pressure > 0, 1),
        ("flow_1s_pred_pressure_lt_0", pressure < 0, -1),
        ("both_agree_bullish", (pressure > 0) & (up > down), 1),
        ("both_agree_bearish", (pressure < 0) & (down > up), -1),
        ("flow_bullish_micro_bearish_conflict", (pressure > 0) & (down > up), 0),
        ("flow_bearish_micro_bullish_conflict", (pressure < 0) & (up > down), 0),
    ]
    for label, mask, predicted_direction in filters:
        summarize_filter(rows, "directional_filter", label, evaluated[mask], baselines, predicted_direction)


def add_class_pressure_disagreement_filters(rows, evaluated, baselines):
    for label, mask in flow_class_pressure_disagreement_masks(evaluated).items():
        summarize_filter(rows, "class_pressure_disagreement", label, evaluated[mask], baselines)


def add_threshold_sweep(rows, evaluated, baselines):
    sweeps = [
        (
            "abs_pred_market_pressure_1s",
            numeric(evaluated, "flow_1s_pred_pressure", 0.0).abs(),
            [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80],
        ),
        (
            "max_micro_scare_probability",
            max_micro_scare_score(evaluated),
            [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        ),
        (
            "spread_expansion_probability",
            numeric(evaluated, "micro_spread_expansion_prob", 0.0),
            [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        ),
        (
            "bid_or_ask_liquidity_drop_probability",
            max_liquidity_drop_score(evaluated),
            [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        ),
    ]
    for sweep_name, values, thresholds in sweeps:
        for threshold in thresholds:
            summarize_filter(
                rows,
                "threshold_sweep",
                f"{sweep_name}>={threshold:.2f}",
                evaluated[values >= threshold],
                baselines,
            )


def add_combined_filters(rows, evaluated, baselines):
    pressure_abs = numeric(evaluated, "flow_1s_pred_pressure", 0.0).abs()
    pressure = numeric(evaluated, "flow_1s_pred_pressure", 0.0)
    scare = max_micro_scare_score(evaluated)
    spread = numeric(evaluated, "micro_spread_expansion_prob", 0.0)
    up = micro_upside_score(evaluated)
    down = micro_downside_score(evaluated)
    both_agree_bullish = (pressure > 0) & (up > down)
    combined = [
        (
            "abs_pressure>=0.30_AND_max_scare>=0.70",
            (pressure_abs >= 0.30) & (scare >= 0.70),
            0,
        ),
        (
            "abs_pressure>=0.50_AND_max_scare>=0.70",
            (pressure_abs >= 0.50) & (scare >= 0.70),
            0,
        ),
        (
            "abs_pressure>=0.65_AND_max_scare>=0.70",
            (pressure_abs >= 0.65) & (scare >= 0.70),
            0,
        ),
        (
            "abs_pressure>=0.50_AND_spread_expansion>=0.70",
            (pressure_abs >= 0.50) & (spread >= 0.70),
            0,
        ),
        (
            "both_agree_bullish_AND_abs_pressure>=0.30",
            both_agree_bullish & (pressure_abs >= 0.30),
            1,
        ),
        (
            "both_agree_bullish_AND_max_scare>=0.70",
            both_agree_bullish & (scare >= 0.70),
            1,
        ),
        (
            "micro_upside_gt_downside_AND_max_scare>=0.70",
            (up > down) & (scare >= 0.70),
            1,
        ),
        (
            "micro_downside_gt_upside_AND_max_scare>=0.70",
            (down > up) & (scare >= 0.70),
            -1,
        ),
    ]
    for label, mask, predicted_direction in combined:
        summarize_filter(rows, "combined_filter", label, evaluated[mask], baselines, predicted_direction)


def add_best_filters(rows):
    candidates = [
        row for row in rows
        if row.get("section") in {"abstain_reason", "directional_filter", "threshold_sweep", "risk", "agreement"}
        and safe_int(row.get("rows"), 0) >= MIN_BEST_FILTER_ROWS
        and finite_number(row.get("avg_return_improvement", np.nan))
        and finite_number(row.get("win_rate_improvement", np.nan))
        and safe_float(row.get("avg_return_improvement"), 0.0) > 0.0
        and safe_float(row.get("win_rate_improvement"), 0.0) > 0.0
    ]
    candidates.sort(
        key=lambda row: (
            safe_float(row.get("avg_return_improvement", -np.inf), -np.inf),
            safe_float(row.get("win_rate_improvement", -np.inf), -np.inf),
            safe_int(row.get("rows"), 0),
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(candidates[:25], start=1):
        payload = {
            "rank": rank,
            "source_section": candidate.get("section"),
            "source_metric": candidate.get("metric"),
            "rows": candidate.get("rows"),
            "avg_return": candidate.get("avg_return"),
            "win_rate_positive_return": candidate.get("win_rate_positive_return"),
            "avg_return_improvement": candidate.get("avg_return_improvement"),
            "win_rate_improvement": candidate.get("win_rate_improvement"),
            "avg_return_10s": candidate.get("avg_return_10s"),
            "avg_return_30s": candidate.get("avg_return_30s"),
            "avg_return_60s": candidate.get("avg_return_60s"),
            "win_rate_10s": candidate.get("win_rate_10s"),
            "win_rate_30s": candidate.get("win_rate_30s"),
            "win_rate_60s": candidate.get("win_rate_60s"),
        }
        add_report_row(
            rows,
            "best_filters",
            f"#{rank} {candidate.get('section')}::{candidate.get('metric')}",
            payload["avg_return"],
            payload,
        )


def add_best_combined_filters(rows):
    candidates = [
        row for row in rows
        if row.get("section") == "combined_filter"
        and safe_int(row.get("rows"), 0) >= MIN_BEST_FILTER_ROWS
    ]
    candidates.sort(
        key=lambda row: (
            safe_float(row.get("avg_return_improvement", -np.inf), -np.inf),
            safe_float(row.get("win_rate_improvement", -np.inf), -np.inf),
            safe_float(row.get("win_lift_vs_majority", -np.inf), -np.inf),
            safe_int(row.get("rows"), 0),
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(candidates, start=1):
        payload = {
            "rank": rank,
            "source_section": candidate.get("section"),
            "source_metric": candidate.get("metric"),
            "rows": candidate.get("rows"),
            "avg_return": candidate.get("avg_return"),
            "win_rate_positive_return": candidate.get("win_rate_positive_return"),
            "avg_return_improvement": candidate.get("avg_return_improvement"),
            "win_rate_improvement": candidate.get("win_rate_improvement"),
            "win_lift_vs_majority": candidate.get("win_lift_vs_majority"),
            "win_lift_vs_always_long": candidate.get("win_lift_vs_always_long"),
            "win_lift_vs_always_short": candidate.get("win_lift_vs_always_short"),
            "avg_return_10s": candidate.get("avg_return_10s"),
            "avg_return_30s": candidate.get("avg_return_30s"),
            "avg_return_60s": candidate.get("avg_return_60s"),
            "win_rate_10s": candidate.get("win_rate_10s"),
            "win_rate_30s": candidate.get("win_rate_30s"),
            "win_rate_60s": candidate.get("win_rate_60s"),
        }
        add_report_row(
            rows,
            "best_combined_filters",
            f"#{rank} {candidate.get('metric')}",
            payload["avg_return"],
            payload,
        )


def baseline_summaries(rows, frame):
    returns = numeric(frame, RETURN_COLUMN).dropna()
    actual_direction = pd.Series(direction_from_return(returns.to_numpy(dtype=np.float64)), index=returns.index)
    if len(actual_direction) and len(actual_direction.mode()):
        majority_direction = int(actual_direction.mode().iloc[0])
    else:
        majority_direction = 0
    summarize_direction_strategy(rows, "baseline", "no_trade_zero_return", frame, pd.Series(0, index=frame.index))
    summarize_direction_strategy(rows, "baseline", "majority_realized_direction", frame, pd.Series(majority_direction, index=frame.index))
    summarize_direction_strategy(rows, "baseline", "always_long_direction", frame, pd.Series(1, index=frame.index))
    summarize_direction_strategy(rows, "baseline", "always_short_direction", frame, pd.Series(-1, index=frame.index))
    summarize_direction_strategy(rows, "baseline", "recent_logged_mid_return_sign", frame, recent_logged_return_baseline_direction(frame))
    summarize_direction_strategy(rows, "baseline", "follow_1s_pred_pressure_sign", frame, pd.Series(flow_pressure_baseline_direction(frame), index=frame.index))


def build_report(frame):
    rows = []
    evaluated = frame.dropna(subset=[RETURN_COLUMN]).copy()
    add_report_row(rows, "overall", "total_logged_rows", int(len(frame)))
    add_report_row(rows, "overall", "total_evaluated_rows", int(len(evaluated)))
    if len(evaluated) == 0:
        return pd.DataFrame(rows)

    baselines = baseline_stats(evaluated)
    flow_dir = pd.Series(flow_direction(evaluated), index=evaluated.index)
    summarize_filter(rows, "overall", "all_evaluated", evaluated, baselines)
    summarize_filter(rows, "agreement", "flow_1s_and_micro_10s_agree", evaluated[numeric(evaluated, "flow_micro_agree", 0) >= 0.5], baselines)
    summarize_filter(rows, "agreement", "flow_1s_and_micro_10s_conflict", evaluated[numeric(evaluated, "flow_micro_conflict", 0) >= 0.5], baselines)

    risk = (
        (numeric(evaluated, "micro_spread_expansion_prob", 0.0) >= 0.70)
        | (numeric(evaluated, "micro_bid_liquidity_drop_prob", 0.0) >= 0.75)
        | (numeric(evaluated, "micro_ask_liquidity_drop_prob", 0.0) >= 0.75)
    )
    summarize_filter(rows, "risk", "high_10s_spread_or_liquidity_risk", evaluated[risk], baselines)
    summarize_filter(rows, "risk", "low_10s_spread_or_liquidity_risk", evaluated[~risk], baselines)

    confidence_buckets = [(0.00, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    summarize_buckets(rows, "flow_1s_confidence_bucket", evaluated, "flow_1s_confidence", confidence_buckets)
    scare_buckets = [(0.00, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 1.01)]
    summarize_buckets(rows, "micro_10s_scare_bucket", evaluated, "micro_scare_score", scare_buckets)
    agreement_buckets = [(0.00, 0.34), (0.34, 0.67), (0.67, 1.01)]
    summarize_buckets(rows, "agreement_score_bucket", evaluated, "agreement_score", agreement_buckets)
    add_abstain_reason_filters(rows, evaluated, baselines)
    add_directional_filters(rows, evaluated, baselines)
    add_class_pressure_disagreement_filters(rows, evaluated, baselines)
    add_threshold_sweep(rows, evaluated, baselines)
    add_combined_filters(rows, evaluated, baselines)
    baseline_summaries(rows, evaluated)
    add_best_filters(rows)
    add_best_combined_filters(rows)
    return pd.DataFrame(rows)


def print_report(report):
    print("Hierarchy forecast evaluation")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {VENUE_TAG}")
    print(f"Log path: {LOG_PATH}")
    print(f"Report path: {REPORT_PATH}")
    print(f"Return column: {RETURN_COLUMN}")
    printable_sections = {
        "overall",
        "agreement",
        "risk",
        "abstain_reason",
        "directional_filter",
        "class_pressure_disagreement",
        "threshold_sweep",
        "combined_filter",
        "baseline",
        "best_filters",
        "best_combined_filters",
        "flow_1s_confidence_bucket",
        "micro_10s_scare_bucket",
        "agreement_score_bucket",
    }
    for _, row in report.iterrows():
        section = row.get("section")
        metric = row.get("metric")
        rows = row.get("rows", "")
        avg_return = row.get("avg_return", row.get("value"))
        win_rate = row.get("win_rate_positive_return", np.nan)
        if section in printable_sections:
            displayed_rows = rows if finite_number(rows) else row.get("value")
            lift = row.get("avg_return_improvement", np.nan)
            win_lift = row.get("win_rate_improvement", np.nan)
            extra = ""
            if section == "best_filters":
                extra = (
                    f", source={row.get('source_section')}::{row.get('source_metric')}, "
                    f"return_lift={percent(lift)}, win_lift={percent(win_lift)}"
                )
            elif section == "best_combined_filters":
                extra = (
                    f", source={row.get('source_metric')}, "
                    f"return_lift={percent(lift)}, win_lift={percent(win_lift)}, "
                    f"win_lift_vs_majority={percent(row.get('win_lift_vs_majority', np.nan))}, "
                    f"win_lift_vs_long={percent(row.get('win_lift_vs_always_long', np.nan))}, "
                    f"win_lift_vs_short={percent(row.get('win_lift_vs_always_short', np.nan))}"
                )
            elif finite_number(lift):
                extra = (
                    f", return_lift={percent(lift)}, win_lift={percent(win_lift)}, "
                    f"win_lift_vs_majority={percent(row.get('win_lift_vs_majority', np.nan))}"
                )
            print(
                f"- [{section}] {metric}: rows={displayed_rows}, "
                f"avg_return={percent(avg_return)}, win_rate={percent(win_rate)}{extra}"
            )
    print("No trades/orders/private API behavior.")


def main():
    frame = read_csv(LOG_PATH)
    if len(frame) == 0:
        report = pd.DataFrame([
            {"section": "overall", "metric": "total_logged_rows", "value": 0},
            {"section": "warning", "metric": "missing_log", "value": str(LOG_PATH)},
        ])
        write_csv(report, REPORT_PATH)
        print_report(report)
        return
    frame, updated = update_realized_outcomes(frame, read_csv(SNAPSHOT_PATH))
    write_csv(frame, LOG_PATH)
    report = build_report(frame)
    write_csv(report, REPORT_PATH)
    print_report(report)
    print(f"Realized outcome cells updated before evaluation: {updated}")


if __name__ == "__main__":
    main()
