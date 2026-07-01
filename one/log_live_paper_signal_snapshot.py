import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from show_live_market_stack import (
    MAX_FLOW_1S_AGE_MS,
    MAX_MICRO_AGE_MS,
    VENUE_OUTPUT_DIR,
    as_float,
    latest_1s_order_flow_prediction,
    latest_micro_prediction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
VENUE_TAG = PRIMARY_VENUE or "legacy"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
PAPER_SIGNAL_TAG = os.getenv("PAPER_SIGNAL_TAG", "").strip()
PAPER_SIGNAL_MAX_STALENESS_SECONDS = float(os.getenv("PAPER_SIGNAL_MAX_STALENESS_SECONDS", "120"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
SNAPSHOT_PATH = VENUE_DIR / f"{SYMBOL}_10s_flow.csv"
FLOW_1S_PREDICTION_PATH = VENUE_DIR / f"{SYMBOL}_1s_order_flow_predictions.csv"
LOG_PATH = VENUE_DIR / f"{SYMBOL}_paper_signal_log.csv"


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
    frame = frame.copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    return frame.sort_values("timestamp").reset_index(drop=True)


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


def latest_fresh_snapshot():
    snapshots = read_csv(SNAPSHOT_PATH)
    if len(snapshots) == 0 or "timestamp" not in snapshots.columns:
        return None, "missing primary venue 10s snapshot"
    snapshots = normalize_timestamps(snapshots)
    latest = snapshots.iloc[-1].to_dict()
    timestamp = int(latest["timestamp"])
    age_seconds = max(0.0, (time.time() * 1000.0 - timestamp) / 1000.0)
    if age_seconds > PAPER_SIGNAL_MAX_STALENESS_SECONDS:
        return latest, f"stale primary snapshot age={age_seconds:.1f}s"
    return latest, "fresh"


def copy_prefixed(row, prefix, columns):
    output = {}
    if row is None:
        return output
    for column in columns:
        if column in row:
            output[f"{prefix}{column}"] = row.get(column)
    return output


def build_log_row(snapshot, flow_1s, micro):
    timestamp = int(snapshot["timestamp"])
    row = {
        "timestamp": timestamp,
        "time": snapshot.get("time", ""),
        "symbol": SYMBOL,
        "primary_venue": VENUE_TAG,
        "paper_signal_tag": PAPER_SIGNAL_TAG,
        "mid_price": price_from_row(snapshot),
        "snapshot_file": str(SNAPSHOT_PATH),
        "flow_1s_prediction_file": str(FLOW_1S_PREDICTION_PATH),
    }

    flow_columns = [
        "decoded_flow_class_1s",
        "prob_sell_dominant_1s",
        "prob_neutral_1s",
        "prob_buy_dominant_1s",
        "pred_market_buy_volume_1s",
        "pred_market_sell_volume_1s",
        "pred_market_pressure_1s",
        "pred_pressure_magnitude_1s",
        "pred_trade_count_1s",
        "buy_burst_prob_1s",
        "sell_burst_prob_1s",
        "model_id",
        "feature_schema_hash",
        "trained_until_timestamp",
        "_prediction_file_age_seconds",
        "_context_age_ms",
    ]
    row.update(copy_prefixed(flow_1s, "flow_1s_", flow_columns))

    if micro is not None:
        row["micro_model_id"] = micro.get("model_id", "")
        row["micro_model_path"] = micro.get("model_path", "")
        row["micro_selected_feature_group"] = micro.get("selected_feature_group", "")
        row["micro_feature_group_used"] = micro.get("feature_group_used", "")
        row["micro_regression_sanity_status"] = micro.get("regression_sanity_status", "")
        row["micro_event_sanity_status"] = micro.get("event_sanity_status", "")
        row["micro_event_saturation_fraction"] = micro.get("event_saturation_fraction", np.nan)
        for key, value in micro.items():
            if key.startswith("prob_") or key.startswith("pred_"):
                row[f"micro_{key}"] = value
    return row


def update_realized_outcomes(log_frame, snapshots):
    if len(log_frame) == 0 or len(snapshots) == 0:
        return log_frame, 0
    log_frame = log_frame.copy()
    snapshots = normalize_timestamps(snapshots)
    snapshots["_price"] = snapshots.apply(price_from_row, axis=1)
    snapshots = snapshots.dropna(subset=["_price"])
    if len(snapshots) == 0:
        return log_frame, 0

    for column in [
        "actual_return_10s",
        "actual_return_30s",
        "actual_return_60s",
        "actual_max_runup_60s",
        "actual_max_drawdown_60s",
    ]:
        if column not in log_frame.columns:
            log_frame[column] = np.nan

    updated = 0
    for index, row in log_frame.iterrows():
        timestamp = as_float(row.get("timestamp"), np.nan)
        entry = as_float(row.get("mid_price"), np.nan)
        if not np.isfinite(timestamp) or not np.isfinite(entry) or entry <= 0:
            continue
        timestamp = int(timestamp)
        future = snapshots[snapshots["timestamp"] > timestamp]
        if len(future) == 0:
            continue

        for horizon_seconds in [10, 30, 60]:
            column = f"actual_return_{horizon_seconds}s"
            if pd.notna(row.get(column)):
                continue
            target_timestamp = timestamp + horizon_seconds * 1000
            if int(snapshots["timestamp"].max()) < target_timestamp:
                continue
            horizon_rows = future[future["timestamp"] <= target_timestamp]
            if len(horizon_rows) == 0:
                continue
            exit_price = float(horizon_rows.iloc[-1]["_price"])
            log_frame.loc[index, column] = exit_price / entry - 1.0
            updated += 1

        if (
            pd.isna(row.get("actual_max_runup_60s"))
            or pd.isna(row.get("actual_max_drawdown_60s"))
        ):
            target_timestamp = timestamp + 60_000
            if int(snapshots["timestamp"].max()) >= target_timestamp:
                horizon_rows = future[future["timestamp"] <= target_timestamp]
                if len(horizon_rows) > 0:
                    prices = horizon_rows["_price"].to_numpy(dtype=np.float64)
                    log_frame.loc[index, "actual_max_runup_60s"] = float(prices.max() / entry - 1.0)
                    log_frame.loc[index, "actual_max_drawdown_60s"] = float(prices.min() / entry - 1.0)
                    updated += 1
    return log_frame, updated


def main():
    snapshot, snapshot_status = latest_fresh_snapshot()
    if snapshot is None or snapshot_status != "fresh":
        print("Paper signal log skipped.")
        print(f"SYMBOL: {SYMBOL}")
        print(f"PRIMARY_VENUE: {VENUE_TAG}")
        print(f"Reason: {snapshot_status}")
        print("No trades/orders/private API behavior.")
        return

    reference_timestamp = int(snapshot["timestamp"])
    flow_1s, flow_status = latest_1s_order_flow_prediction(SYMBOL, reference_timestamp)
    if flow_1s is None or flow_status != "fresh":
        print("Paper signal log skipped.")
        print(f"Reason: 1s order-flow prediction not fresh ({flow_status})")
        print("No trades/orders/private API behavior.")
        return

    micro, micro_status = latest_micro_prediction(SYMBOL)
    if micro is None:
        print("Paper signal log skipped.")
        print(f"Reason: 10s microstructure prediction unavailable ({micro_status})")
        print("No trades/orders/private API behavior.")
        return
    micro_age_ms = reference_timestamp - int(micro["timestamp"])
    if micro_age_ms > MAX_MICRO_AGE_MS:
        print("Paper signal log skipped.")
        print(f"Reason: 10s microstructure prediction stale context_age_ms={micro_age_ms}")
        print("No trades/orders/private API behavior.")
        return

    log_frame = read_csv(LOG_PATH)
    new_row = build_log_row(snapshot, flow_1s, micro)
    log_frame = pd.concat([log_frame, pd.DataFrame([new_row])], ignore_index=True)
    log_frame = normalize_timestamps(log_frame)
    log_frame = log_frame.drop_duplicates(subset=["timestamp"], keep="last")

    snapshots = read_csv(SNAPSHOT_PATH)
    log_frame, updated_outcomes = update_realized_outcomes(log_frame, snapshots)
    write_csv(log_frame, LOG_PATH)

    print("Paper signal snapshot logged.")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {VENUE_TAG}")
    print(f"timestamp: {reference_timestamp}")
    print(f"mid_price: {new_row.get('mid_price')}")
    print(f"PAPER_SIGNAL_TAG: {PAPER_SIGNAL_TAG or 'none'}")
    print(f"output: {LOG_PATH}")
    print(f"realized outcome cells updated: {updated_outcomes}")
    print("No trades/orders/private API behavior.")


if __name__ == "__main__":
    main()
