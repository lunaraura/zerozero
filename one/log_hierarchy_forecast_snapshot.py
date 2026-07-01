import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from show_live_market_stack import (
    MAX_3M_AGE_MS,
    MAX_FLOW_1S_AGE_MS,
    MAX_HTF_AGE_MS,
    MAX_MICRO_AGE_MS,
    MAX_REGIME_AGE_MS,
    VENUE_OUTPUT_DIR,
    as_float,
    flow_1s_label,
    flow_1s_class_pressure_disagreement,
    flow_1s_pred_pressure,
    latest_1s_order_flow_prediction,
    latest_3m_prediction,
    latest_htf_context,
    latest_micro_prediction,
    latest_regime,
    path_label,
    regime_label,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
VENUE_TAG = PRIMARY_VENUE or "legacy"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
MAX_SNAPSHOT_AGE_SECONDS = float(os.getenv("HIERARCHY_LOG_MAX_SNAPSHOT_AGE_SECONDS", "120"))
HIERARCHY_LOG_LOOP = os.getenv("HIERARCHY_LOG_LOOP", "false").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
} or "--loop" in sys.argv
HIERARCHY_LOG_INTERVAL_SECONDS = float(os.getenv("HIERARCHY_LOG_INTERVAL_SECONDS", "5"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
SNAPSHOT_PATH = VENUE_DIR / f"{SYMBOL}_10s_flow.csv"
LOG_PATH = VENUE_DIR / f"{SYMBOL}_hierarchy_forecast_log.csv"

REALIZED_COLUMNS = [
    "realized_return_10s",
    "realized_return_30s",
    "realized_return_60s",
    "realized_max_runup_60s",
    "realized_max_drawdown_60s",
]


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def normalize_timestamps(frame):
    if len(frame) == 0 or "timestamp" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame["timestamp"] = np.where(
        frame["timestamp"] < 10_000_000_000,
        frame["timestamp"] * 1000,
        frame["timestamp"],
    ).astype(np.int64)
    return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def price_from_row(row):
    for column in ["mid_price", "close"]:
        value = as_float(row.get(column), np.nan)
        if np.isfinite(value) and value > 0:
            return value
    bid = as_float(row.get("best_bid"), np.nan)
    ask = as_float(row.get("best_ask"), np.nan)
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return np.nan


def latest_snapshot():
    snapshots = normalize_timestamps(read_csv(SNAPSHOT_PATH))
    if len(snapshots) == 0:
        return None, "missing primary venue 10s snapshot"
    latest = snapshots.iloc[-1].to_dict()
    timestamp = int(latest["timestamp"])
    age_seconds = max(0.0, (time.time() * 1000.0 - timestamp) / 1000.0)
    if age_seconds > MAX_SNAPSHOT_AGE_SECONDS:
        return latest, f"stale primary snapshot age={age_seconds:.1f}s"
    return latest, "fresh"


def current_log_status():
    log_frame = normalize_timestamps(read_csv(LOG_PATH))
    if len(log_frame) == 0:
        return {
            "latest_logged_timestamp": "",
            "row_count": 0,
            "latest_abstain_reason": "",
        }
    latest_row = log_frame.iloc[-1]
    return {
        "latest_logged_timestamp": int(latest_row["timestamp"]),
        "row_count": int(len(log_frame)),
        "latest_abstain_reason": str(latest_row.get("abstain_reason", "")),
    }


def signed_direction_from_label(label):
    label = str(label or "").lower()
    if "buy" in label or "long" in label or "bull" in label or "up" in label:
        return 1
    if "sell" in label or "short" in label or "bear" in label or "down" in label:
        return -1
    return 0


def signed_direction_from_micro(micro):
    if micro is None:
        return 0
    up = max(
        as_float(micro.get("prob_moderate_upside_scare"), 0.0),
        as_float(micro.get("prob_aggressive_buy_burst_10s"), 0.0),
    )
    down = max(
        as_float(micro.get("prob_moderate_downside_scare"), 0.0),
        as_float(micro.get("prob_aggressive_sell_burst_10s"), 0.0),
    )
    if max(up, down) < 0.50:
        return 0
    return 1 if up > down else -1


def confidence_1s(flow_1s):
    if flow_1s is None:
        return np.nan
    return max(
        as_float(flow_1s.get("prob_sell_dominant_1s"), 0.0),
        as_float(flow_1s.get("prob_neutral_1s"), 0.0),
        as_float(flow_1s.get("prob_buy_dominant_1s"), 0.0),
    )


def scare_score_10s(micro):
    if micro is None:
        return np.nan
    return max(
        as_float(micro.get("prob_moderate_upside_scare"), 0.0),
        as_float(micro.get("prob_moderate_downside_scare"), 0.0),
        as_float(micro.get("prob_upside_scare_event_10s"), 0.0),
        as_float(micro.get("prob_downside_scare_event_10s"), 0.0),
    )


def freshness_age(reference_timestamp, row):
    if row is None:
        return np.nan
    timestamp = as_float(row.get("timestamp", row.get("_timestamp")), np.nan)
    if not np.isfinite(timestamp):
        return np.nan
    return max(0.0, (reference_timestamp - int(timestamp)) / 1000.0)


def is_fresh_status(status):
    return status in {None, "fresh"}


def build_abstain_reason(flow_status, micro_status, path_status, regime_15_status, regime_30_status, htf_status, micro):
    reasons = []
    if flow_status != "fresh":
        reasons.append(f"1s_not_fresh:{flow_status}")
    if micro_status not in {None, "fresh"}:
        reasons.append(f"10s_not_fresh:{micro_status}")
    if micro is not None:
        if micro.get("regression_sanity_failed"):
            reasons.append("10s_regression_sanity_failed")
        if micro.get("event_sanity_failed"):
            reasons.append("10s_event_sanity_failed")
        if as_float(micro.get("prob_spread_expansion_event_10s"), 0.0) >= 0.75:
            reasons.append("spread_expansion_risk_high")
        if max(
            as_float(micro.get("prob_bid_liquidity_drop_10s"), 0.0),
            as_float(micro.get("prob_ask_liquidity_drop_10s"), 0.0),
        ) >= 0.80:
            reasons.append("liquidity_drop_risk_high")
    for label, status in [
        ("3m", path_status),
        ("15m", regime_15_status),
        ("30m", regime_30_status),
        ("htf", htf_status),
    ]:
        if status not in {None, "fresh"}:
            reasons.append(f"{label}_optional_unavailable_or_stale")
    return ";".join(reasons)


def update_realized_outcomes(log_frame, snapshots):
    if len(log_frame) == 0 or len(snapshots) == 0:
        return log_frame, 0
    log_frame = log_frame.copy()
    snapshots = normalize_timestamps(snapshots)
    snapshots["_price"] = snapshots.apply(price_from_row, axis=1)
    snapshots = snapshots.dropna(subset=["_price"])
    if len(snapshots) == 0:
        return log_frame, 0
    for column in REALIZED_COLUMNS:
        if column not in log_frame.columns:
            log_frame[column] = np.nan
    updated = 0
    latest_snapshot_timestamp = int(snapshots["timestamp"].max())
    for index, row in log_frame.iterrows():
        timestamp = as_float(row.get("timestamp"), np.nan)
        entry = as_float(row.get("current_mid"), np.nan)
        if not np.isfinite(timestamp) or not np.isfinite(entry) or entry <= 0:
            continue
        timestamp = int(timestamp)
        future = snapshots[snapshots["timestamp"] > timestamp]
        if len(future) == 0:
            continue
        for horizon_seconds in [10, 30, 60]:
            column = f"realized_return_{horizon_seconds}s"
            if pd.notna(row.get(column)):
                continue
            target_timestamp = timestamp + horizon_seconds * 1000
            if latest_snapshot_timestamp < target_timestamp:
                continue
            horizon_rows = future[future["timestamp"] <= target_timestamp]
            if len(horizon_rows) == 0:
                continue
            exit_price = float(horizon_rows.iloc[-1]["_price"])
            log_frame.loc[index, column] = exit_price / entry - 1.0
            updated += 1
        if pd.isna(row.get("realized_max_runup_60s")) or pd.isna(row.get("realized_max_drawdown_60s")):
            target_timestamp = timestamp + 60_000
            if latest_snapshot_timestamp >= target_timestamp:
                horizon_rows = future[future["timestamp"] <= target_timestamp]
                if len(horizon_rows):
                    prices = horizon_rows["_price"].to_numpy(dtype=np.float64)
                    log_frame.loc[index, "realized_max_runup_60s"] = float(prices.max() / entry - 1.0)
                    log_frame.loc[index, "realized_max_drawdown_60s"] = float(prices.min() / entry - 1.0)
                    updated += 1
    return log_frame, updated


def build_log_row(snapshot, flow_1s, micro, path_3m, regime_15m, regime_30m, htf, statuses):
    timestamp = int(snapshot["timestamp"])
    current_mid = price_from_row(snapshot)
    flow_label = flow_1s_label(flow_1s)
    flow_pressure = flow_1s_pred_pressure(flow_1s)
    flow_pressure_disagreement, flow_pressure_disagreement_reason = flow_1s_class_pressure_disagreement(flow_1s)
    path3_label = path_label(path_3m) if path_3m is not None else ""
    regime15_label = regime_label(regime_15m) if regime_15m is not None else ""
    regime30_label = regime_label(regime_30m) if regime_30m is not None else ""
    flow_dir = signed_direction_from_label(flow_label)
    micro_dir = signed_direction_from_micro(micro)
    path_dir = signed_direction_from_label(path3_label)
    regime15_dir = signed_direction_from_label(regime15_label)
    regime30_dir = signed_direction_from_label(regime30_label)
    directional_votes = [value for value in [flow_dir, micro_dir, path_dir, regime15_dir, regime30_dir] if value != 0]
    agreement_score = (
        abs(sum(directional_votes)) / len(directional_votes)
        if directional_votes
        else 0.0
    )
    contradiction_score = 1.0 - agreement_score if directional_votes else 0.0
    row = {
        "timestamp": timestamp,
        "time": snapshot.get("time", ""),
        "symbol": SYMBOL,
        "primary_venue": VENUE_TAG,
        "current_mid": current_mid,
        "snapshot_path": str(SNAPSHOT_PATH),
        "flow_1s_status": statuses["flow_1s"],
        "micro_10s_status": statuses["micro"],
        "path_3m_status": statuses["path_3m"],
        "regime_15m_status": statuses["regime_15m"],
        "regime_30m_status": statuses["regime_30m"],
        "htf_status": statuses["htf"],
        "flow_1s_age_seconds": as_float(flow_1s.get("_context_age_ms"), np.nan) / 1000.0 if flow_1s is not None else np.nan,
        "micro_10s_age_seconds": freshness_age(timestamp, micro),
        "micro_10s_event_only": bool(micro.get("event_only", False)) if micro is not None else False,
        "path_3m_age_seconds": freshness_age(timestamp, path_3m),
        "regime_15m_age_seconds": freshness_age(timestamp, regime_15m),
        "regime_30m_age_seconds": freshness_age(timestamp, regime_30m),
        "htf_age_seconds": freshness_age(timestamp, htf),
        "flow_1s_model_id": flow_1s.get("model_id", "") if flow_1s is not None else "",
        "micro_10s_model_id": micro.get("model_id", "") if micro is not None else "",
        "path_3m_model_id": path_3m.get("model_id", "") if path_3m is not None else "",
        "flow_1s_class": flow_label,
        "flow_1s_prob_sell": flow_1s.get("prob_sell_dominant_1s", np.nan) if flow_1s is not None else np.nan,
        "flow_1s_prob_neutral": flow_1s.get("prob_neutral_1s", np.nan) if flow_1s is not None else np.nan,
        "flow_1s_prob_buy": flow_1s.get("prob_buy_dominant_1s", np.nan) if flow_1s is not None else np.nan,
        "flow_1s_confidence": confidence_1s(flow_1s),
        "flow_1s_pred_pressure": flow_pressure,
        "flow_1s_class_pressure_disagreement": int(flow_pressure_disagreement),
        "flow_1s_class_pressure_disagreement_reason": flow_pressure_disagreement_reason,
        "flow_1s_buy_burst_prob": flow_1s.get("buy_burst_prob_1s", np.nan) if flow_1s is not None else np.nan,
        "flow_1s_sell_burst_prob": flow_1s.get("sell_burst_prob_1s", np.nan) if flow_1s is not None else np.nan,
        "micro_upside_scare_prob": micro.get("prob_upside_scare_event_10s", micro.get("prob_moderate_upside_scare", np.nan)) if micro is not None else np.nan,
        "micro_downside_scare_prob": micro.get("prob_downside_scare_event_10s", micro.get("prob_moderate_downside_scare", np.nan)) if micro is not None else np.nan,
        "micro_scare_score": scare_score_10s(micro),
        "micro_buy_burst_prob": micro.get("prob_aggressive_buy_burst_10s", np.nan) if micro is not None else np.nan,
        "micro_sell_burst_prob": micro.get("prob_aggressive_sell_burst_10s", np.nan) if micro is not None else np.nan,
        "micro_bid_liquidity_drop_prob": micro.get("prob_bid_liquidity_drop_10s", np.nan) if micro is not None else np.nan,
        "micro_ask_liquidity_drop_prob": micro.get("prob_ask_liquidity_drop_10s", np.nan) if micro is not None else np.nan,
        "micro_spread_expansion_prob": micro.get("prob_spread_expansion_event_10s", np.nan) if micro is not None else np.nan,
        "micro_continuation_30s_prob": micro.get("prob_continuation_30s", np.nan) if micro is not None else np.nan,
        "micro_continuation_60s_prob": micro.get("prob_continuation_60s", np.nan) if micro is not None else np.nan,
        "micro_direction_flip_10s_prob": micro.get("prob_direction_flip_10s", np.nan) if micro is not None else np.nan,
        "micro_reversal_after_upside_scare_30s_prob": micro.get("prob_reversal_after_upside_scare_30s", np.nan) if micro is not None else np.nan,
        "micro_reversal_after_downside_scare_30s_prob": micro.get("prob_reversal_after_downside_scare_30s", np.nan) if micro is not None else np.nan,
        "micro_reversal_after_upside_scare_60s_prob": micro.get("prob_reversal_after_upside_scare_60s", np.nan) if micro is not None else np.nan,
        "micro_reversal_after_downside_scare_60s_prob": micro.get("prob_reversal_after_downside_scare_60s", np.nan) if micro is not None else np.nan,
        "path_3m_class": path3_label,
        "path_3m_prob_short": path_3m.get("prob_short", np.nan) if path_3m is not None else np.nan,
        "path_3m_prob_neutral": path_3m.get("prob_neutral", np.nan) if path_3m is not None else np.nan,
        "path_3m_prob_long": path_3m.get("prob_long", np.nan) if path_3m is not None else np.nan,
        "regime_15m_label": regime15_label,
        "regime_30m_label": regime30_label,
        "htf_bull_prob": htf.get("hourly_bull_prob", np.nan) if htf is not None else np.nan,
        "htf_bear_prob": htf.get("hourly_bear_prob", np.nan) if htf is not None else np.nan,
        "htf_chop_prob": htf.get("hourly_chop_prob", np.nan) if htf is not None else np.nan,
        "agreement_score": agreement_score,
        "contradiction_score": contradiction_score,
        "flow_micro_agree": int(flow_dir != 0 and flow_dir == micro_dir),
        "flow_micro_conflict": int(flow_dir != 0 and micro_dir != 0 and flow_dir != micro_dir),
        "abstain_reason": build_abstain_reason(
            statuses["flow_1s"],
            statuses["micro"],
            statuses["path_3m"],
            statuses["regime_15m"],
            statuses["regime_30m"],
            statuses["htf"],
            micro,
        ),
    }
    return row


def log_once(verbose=True):
    status = current_log_status()
    snapshot, snapshot_status = latest_snapshot()
    if snapshot is None or snapshot_status != "fresh":
        result = {
            **status,
            "logged": False,
            "reason": snapshot_status,
            "flow_1s_status": "not_checked",
            "micro_10s_status": "not_checked",
            "abstain_reason": status["latest_abstain_reason"],
            "realized_updates": 0,
        }
        if verbose:
            print("Hierarchy forecast log skipped.")
            print(f"Reason: {snapshot_status}")
            print("No trades/orders/private API behavior.")
        return result
    reference_timestamp = int(snapshot["timestamp"])
    flow_1s, flow_status = latest_1s_order_flow_prediction(SYMBOL, reference_timestamp)
    if flow_1s is None or flow_status != "fresh":
        result = {
            **status,
            "logged": False,
            "reason": f"1s order-flow prediction not fresh ({flow_status})",
            "flow_1s_status": flow_status,
            "micro_10s_status": "not_checked",
            "abstain_reason": status["latest_abstain_reason"],
            "realized_updates": 0,
        }
        if verbose:
            print("Hierarchy forecast log skipped.")
            print(f"Reason: {result['reason']}")
            print("No trades/orders/private API behavior.")
        return result
    micro, micro_status = latest_micro_prediction(SYMBOL)
    if micro is None:
        result = {
            **status,
            "logged": False,
            "reason": f"10s microstructure unavailable ({micro_status})",
            "flow_1s_status": flow_status,
            "micro_10s_status": micro_status,
            "abstain_reason": status["latest_abstain_reason"],
            "realized_updates": 0,
        }
        if verbose:
            print("Hierarchy forecast log skipped.")
            print(f"Reason: {result['reason']}")
            print("No trades/orders/private API behavior.")
        return result
    micro_age_ms = reference_timestamp - int(micro["timestamp"])
    if micro_age_ms > MAX_MICRO_AGE_MS:
        result = {
            **status,
            "logged": False,
            "reason": f"10s microstructure stale context_age_ms={micro_age_ms}",
            "flow_1s_status": flow_status,
            "micro_10s_status": f"stale context_age_ms={micro_age_ms}",
            "abstain_reason": status["latest_abstain_reason"],
            "realized_updates": 0,
        }
        if verbose:
            print("Hierarchy forecast log skipped.")
            print(f"Reason: {result['reason']}")
            print("No trades/orders/private API behavior.")
        return result

    path_3m, path_status = latest_3m_prediction(SYMBOL, reference_timestamp)
    if path_status != "fresh":
        path_3m = None
    regime_15m, regime_15_status = latest_regime(SYMBOL, "15m", reference_timestamp)
    if regime_15_status not in {None, "fresh"}:
        regime_15m = None
    regime_30m, regime_30_status = latest_regime(SYMBOL, "30m", reference_timestamp)
    if regime_30_status not in {None, "fresh"}:
        regime_30m = None
    htf, htf_status = latest_htf_context(SYMBOL, reference_timestamp)
    if htf_status not in {None, "fresh"}:
        htf = None

    statuses = {
        "flow_1s": flow_status,
        "micro": micro_status or "fresh",
        "path_3m": path_status or "fresh",
        "regime_15m": regime_15_status or "fresh",
        "regime_30m": regime_30_status or "fresh",
        "htf": htf_status or "fresh",
    }
    new_row = build_log_row(snapshot, flow_1s, micro, path_3m, regime_15m, regime_30m, htf, statuses)
    log_frame = read_csv(LOG_PATH)
    log_frame = pd.concat([log_frame, pd.DataFrame([new_row])], ignore_index=True)
    log_frame = normalize_timestamps(log_frame).drop_duplicates("timestamp", keep="last")
    log_frame, updated = update_realized_outcomes(log_frame, read_csv(SNAPSHOT_PATH))
    write_csv(log_frame, LOG_PATH)
    result = {
        "logged": True,
        "reason": "logged",
        "latest_logged_timestamp": reference_timestamp,
        "row_count": int(len(log_frame)),
        "flow_1s_status": flow_status,
        "micro_10s_status": micro_status or "fresh",
        "abstain_reason": new_row["abstain_reason"],
        "agreement_score": new_row["agreement_score"],
        "contradiction_score": new_row["contradiction_score"],
        "realized_updates": updated,
    }
    if verbose:
        print("Hierarchy forecast snapshot logged.")
        print(f"SYMBOL: {SYMBOL}")
        print(f"PRIMARY_VENUE: {VENUE_TAG}")
        print(f"timestamp: {reference_timestamp}")
        print(f"agreement_score: {new_row['agreement_score']:.3f}")
        print(f"contradiction_score: {new_row['contradiction_score']:.3f}")
        print(f"abstain_reason: {new_row['abstain_reason'] or 'none'}")
        print(f"output: {LOG_PATH}")
        print(f"realized outcome cells updated: {updated}")
        print("No trades/orders/private API behavior.")
    return result


def print_loop_summary(result):
    print(
        "Hierarchy log loop | "
        f"latest_logged_timestamp={result.get('latest_logged_timestamp', '') or 'none'} | "
        f"row_count={result.get('row_count', 0)} | "
        f"1s_status={result.get('flow_1s_status', 'unknown')} | "
        f"10s_status={result.get('micro_10s_status', 'unknown')} | "
        f"abstain_reason={result.get('abstain_reason') or 'none'} | "
        f"reason={result.get('reason', '')} | "
        f"realized_updates={result.get('realized_updates', 0)}",
        flush=True,
    )


def run_loop():
    print("Hierarchy forecast logging loop starting.")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {VENUE_TAG}")
    print(f"interval_seconds: {HIERARCHY_LOG_INTERVAL_SECONDS}")
    print(f"output: {LOG_PATH}")
    print("Paper-only: no trades/orders/private API behavior.")
    try:
        while True:
            result = log_once(verbose=False)
            print_loop_summary(result)
            time.sleep(max(0.25, HIERARCHY_LOG_INTERVAL_SECONDS))
    except KeyboardInterrupt:
        print("Hierarchy forecast logging loop stopped.")


def main():
    if HIERARCHY_LOG_LOOP:
        run_loop()
        return
    log_once(verbose=True)


if __name__ == "__main__":
    main()
