import os
from pathlib import Path

import numpy as np
import pandas as pd

from microstructure_model_utils import (
    atomic_write_csv,
    first_value,
    infer_snapshot_step_seconds,
    is_valid_current_row,
    last_value,
    load_snapshot_rows,
    mean_value,
    percent,
    previous_row_window,
    row_window,
    safe_ratio,
    volume_sum,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
MAX_NEXT_SECOND_GAP_MS = int(os.getenv("MAX_NEXT_SECOND_GAP_MS", "1500"))
FLOW_1S_PRESSURE_CLASS_THRESHOLD = float(
    os.getenv("FLOW_1S_PRESSURE_CLASS_THRESHOLD", os.getenv("FLOW_PRESSURE_THRESHOLD", "0.20"))
)
# Backward-compatible alias used by older evaluator/GUI scripts.
FLOW_PRESSURE_THRESHOLD = FLOW_1S_PRESSURE_CLASS_THRESHOLD
FLOW_1S_MIN_DIRECTIONAL_VOLUME = float(os.getenv("FLOW_1S_MIN_DIRECTIONAL_VOLUME", "0"))
FLOW_1S_MIN_DIRECTIONAL_TRADE_COUNT = float(os.getenv("FLOW_1S_MIN_DIRECTIONAL_TRADE_COUNT", "0"))
BURST_QUANTILE = float(os.getenv("FLOW_1S_BURST_QUANTILE", "0.90"))
BURST_LOOKBACK_ROWS = int(os.getenv("FLOW_1S_BURST_LOOKBACK_ROWS", "300"))
MIN_BURST_LOOKBACK_ROWS = int(os.getenv("FLOW_1S_MIN_BURST_LOOKBACK_ROWS", "60"))
MAX_FLOW_1S_SNAPSHOTS = int(os.getenv("MAX_FLOW_1S_SNAPSHOTS", "0"))
MAX_1S_FLOW_TRAINING_ROWS = int(os.getenv("MAX_1S_FLOW_TRAINING_ROWS", "50000"))

WINDOW_SECONDS = [1, 2, 3, 5, 10, 30, 60]
CLASS_NAMES = {0: "sell_dominant", 1: "neutral", 2: "buy_dominant"}

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
INPUT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_10s_flow.csv"
OUTPUT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_1s_order_flow_training_rows.csv"


def optional_float_env(name):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return None
    return float(raw)


MIN_BUY_BURST_VOLUME = optional_float_env("FLOW_1S_MIN_BUY_BURST_VOLUME")
MIN_SELL_BURST_VOLUME = optional_float_env("FLOW_1S_MIN_SELL_BURST_VOLUME")


def numeric_value(row, column, default=0.0):
    value = pd.to_numeric(pd.Series([row.get(column, default)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) and np.isfinite(value) else default


def pressure_from_volumes(buy_volume, sell_volume):
    return safe_ratio(buy_volume - sell_volume, buy_volume + sell_volume)


def flow_class_from_activity(pressure, total_volume, trade_count):
    if total_volume < FLOW_1S_MIN_DIRECTIONAL_VOLUME:
        return 1
    if trade_count < FLOW_1S_MIN_DIRECTIONAL_TRADE_COUNT:
        return 1
    if pressure <= -FLOW_1S_PRESSURE_CLASS_THRESHOLD:
        return 0
    if pressure >= FLOW_1S_PRESSURE_CLASS_THRESHOLD:
        return 2
    return 1


def future_window(frame, index, horizon_seconds):
    if index + horizon_seconds >= len(frame):
        return None, f"missing future {horizon_seconds}s rows"
    current_timestamp = int(frame.loc[index, "timestamp"])
    rows = []
    previous_timestamp = current_timestamp
    for offset in range(1, horizon_seconds + 1):
        row = frame.loc[index + offset]
        timestamp = int(row["timestamp"])
        if timestamp - previous_timestamp > MAX_NEXT_SECOND_GAP_MS:
            return None, f"future {horizon_seconds}s timestamp gap too large"
        previous_timestamp = timestamp
        rows.append(row)
    if int(rows[-1]["timestamp"]) - current_timestamp > horizon_seconds * MAX_NEXT_SECOND_GAP_MS:
        return None, f"future {horizon_seconds}s window too wide"
    return pd.DataFrame(rows), None


def aggregate_future_flow(window):
    buy_volume = volume_sum(window, "market_buy_volume_10s")
    sell_volume = volume_sum(window, "market_sell_volume_10s")
    total_volume = volume_sum(window, "total_trade_volume_10s")
    trade_count = volume_sum(window, "trade_count_10s")
    pressure = pressure_from_volumes(buy_volume, sell_volume)
    return {
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "total_volume": total_volume,
        "trade_count": trade_count,
        "pressure": pressure,
        "flow_class": flow_class_from_activity(pressure, total_volume, trade_count),
    }


def rolling_positive_volume_threshold(values, end_index, explicit_minimum):
    start = max(0, end_index - BURST_LOOKBACK_ROWS)
    window = pd.to_numeric(values.iloc[start:end_index], errors="coerce").dropna()
    if len(window) < MIN_BURST_LOOKBACK_ROWS:
        return None
    positive = window[window > 0]
    if len(positive) == 0:
        # With no positive historical flow, there is no sane adaptive threshold.
        # Use an explicit floor if supplied; otherwise no next row can qualify.
        return {
            "threshold": float(explicit_minimum) if explicit_minimum is not None else float("inf"),
            "minimum": float(explicit_minimum) if explicit_minimum is not None else float("inf"),
            "positive_history_count": 0,
            "adaptive_minimum_used": explicit_minimum is None,
        }

    quantile_threshold = float(positive.quantile(BURST_QUANTILE))
    adaptive_minimum = float(positive.median())
    minimum = float(explicit_minimum) if explicit_minimum is not None else adaptive_minimum
    return {
        "threshold": max(quantile_threshold, minimum),
        "minimum": minimum,
        "positive_history_count": int(len(positive)),
        "adaptive_minimum_used": explicit_minimum is None,
    }


def is_burst(future_volume, threshold_info):
    if threshold_info is None:
        return False
    threshold = threshold_info["threshold"]
    minimum = threshold_info["minimum"]
    if not np.isfinite(threshold) or not np.isfinite(minimum):
        return False
    return (
        future_volume > 0
        and future_volume >= threshold
        and future_volume >= minimum
    )


def build_feature_row(frame, end_index, snapshot_step_seconds):
    row = frame.loc[end_index]
    if not is_valid_current_row(row):
        return None, "invalid current book row"

    history_60s = row_window(frame, end_index, 60)
    if len(history_60s) == 0:
        return None, "missing 60s history"

    current_mid = numeric_value(row, "mid_price")
    if current_mid <= 0:
        return None, "invalid current mid price"

    feature_row = {
        "timestamp": int(row["timestamp"]),
        "time": row.get("time", ""),
        "feature_ready": True,
        "snapshot_step_seconds": snapshot_step_seconds,
    }

    past_60_volume = max(volume_sum(history_60s, "total_trade_volume_10s"), 1e-9)
    past_60_trades = max(volume_sum(history_60s, "trade_count_10s"), 1e-9)
    past_60_spread = max(mean_value(history_60s, "spread_percent"), 1e-9)

    previous_pressure = 0.0
    previous_velocity = 0.0
    for seconds in WINDOW_SECONDS:
        window = row_window(frame, end_index, seconds)
        previous = previous_row_window(frame, end_index, seconds)
        if len(window) == 0:
            return None, f"empty {seconds}s feature window"

        start_mid = first_value(window, "mid_price", current_mid)
        end_mid = last_value(window, "mid_price", current_mid)
        window_return = safe_ratio(end_mid - start_mid, start_mid)
        velocity = safe_ratio(window_return, max(1, seconds))
        acceleration = velocity - previous_velocity
        previous_velocity = velocity

        buy_volume = volume_sum(window, "market_buy_volume_10s")
        sell_volume = volume_sum(window, "market_sell_volume_10s")
        total_volume = volume_sum(window, "total_trade_volume_10s")
        trade_count = volume_sum(window, "trade_count_10s")
        pressure = pressure_from_volumes(buy_volume, sell_volume)
        pressure_change = pressure - previous_pressure
        previous_pressure = pressure

        prev_volume = max(volume_sum(previous, "total_trade_volume_10s"), 1e-9)
        prev_trade_count = max(volume_sum(previous, "trade_count_10s"), 1e-9)
        prev_spread = max(mean_value(previous, "spread_percent", past_60_spread), 1e-9)

        bid_depth_start = first_value(window, "bid_depth_10bps", 0.0)
        ask_depth_start = first_value(window, "ask_depth_10bps", 0.0)
        bid_depth_end = last_value(window, "bid_depth_10bps", 0.0)
        ask_depth_end = last_value(window, "ask_depth_10bps", 0.0)
        imbalance_start = first_value(window, "order_book_imbalance_10bps", 0.0)
        imbalance_end = last_value(window, "order_book_imbalance_10bps", 0.0)
        spread_now = last_value(window, "spread_percent", 0.0)
        spread_mean = mean_value(window, "spread_percent", 0.0)

        prefix = f"feature_{seconds}s_"
        feature_row[prefix + "return"] = window_return
        feature_row[prefix + "price_velocity"] = velocity
        feature_row[prefix + "price_acceleration"] = acceleration
        feature_row[prefix + "market_buy_volume"] = buy_volume
        feature_row[prefix + "market_sell_volume"] = sell_volume
        feature_row[prefix + "total_trade_volume"] = total_volume
        feature_row[prefix + "trade_count"] = trade_count
        feature_row[prefix + "market_pressure"] = pressure
        feature_row[prefix + "market_pressure_change"] = pressure_change
        feature_row[prefix + "volume_burst_vs_60s"] = safe_ratio(total_volume, past_60_volume * seconds / 60.0)
        feature_row[prefix + "trade_burst_vs_60s"] = safe_ratio(trade_count, past_60_trades * seconds / 60.0)
        feature_row[prefix + "volume_change_vs_previous"] = safe_ratio(total_volume - prev_volume, prev_volume)
        feature_row[prefix + "trade_change_vs_previous"] = safe_ratio(trade_count - prev_trade_count, prev_trade_count)
        feature_row[prefix + "spread_level"] = spread_mean
        feature_row[prefix + "spread_expansion"] = safe_ratio(spread_now - prev_spread, prev_spread)
        feature_row[prefix + "bid_depth_change"] = safe_ratio(bid_depth_end - bid_depth_start, bid_depth_start)
        feature_row[prefix + "ask_depth_change"] = safe_ratio(ask_depth_end - ask_depth_start, ask_depth_start)
        feature_row[prefix + "order_book_imbalance"] = mean_value(window, "order_book_imbalance_10bps")
        feature_row[prefix + "imbalance_change"] = imbalance_end - imbalance_start

    feature_row["feature_current_mid_price"] = current_mid
    feature_row["feature_current_spread_percent"] = numeric_value(row, "spread_percent")
    feature_row["feature_current_bid_depth_10bps"] = numeric_value(row, "bid_depth_10bps")
    feature_row["feature_current_ask_depth_10bps"] = numeric_value(row, "ask_depth_10bps")
    feature_row["feature_current_order_book_imbalance_10bps"] = numeric_value(row, "order_book_imbalance_10bps")
    feature_row["feature_current_market_pressure_10s"] = numeric_value(row, "market_pressure_10s")
    return feature_row, None


def build_target_row(frame, index):
    current_timestamp = int(frame.loc[index, "timestamp"])
    next_window, next_reason = future_window(frame, index, 1)
    if next_window is None:
        return None, next_reason
    next_row = next_window.iloc[-1]
    next_timestamp = int(next_row["timestamp"])

    buy_threshold_info = rolling_positive_volume_threshold(
        frame["market_buy_volume_10s"],
        index,
        MIN_BUY_BURST_VOLUME,
    )
    sell_threshold_info = rolling_positive_volume_threshold(
        frame["market_sell_volume_10s"],
        index,
        MIN_SELL_BURST_VOLUME,
    )
    if buy_threshold_info is None or sell_threshold_info is None:
        return None, "not enough rolling burst history"

    one_second = aggregate_future_flow(next_window)
    buy_volume = one_second["buy_volume"]
    sell_volume = one_second["sell_volume"]
    total_volume = one_second["total_volume"]
    trade_count = one_second["trade_count"]
    pressure = one_second["pressure"]
    flow_class = one_second["flow_class"]

    targets = {
        "future_market_buy_volume_1s": buy_volume,
        "future_market_sell_volume_1s": sell_volume,
        "future_market_pressure_1s": pressure,
        "future_trade_count_1s": trade_count,
        "future_log_market_buy_volume_1s": float(np.log1p(max(0.0, buy_volume))),
        "future_log_market_sell_volume_1s": float(np.log1p(max(0.0, sell_volume))),
        "future_log_trade_count_1s": float(np.log1p(max(0.0, trade_count))),
        "next_1s_flow_class": flow_class,
        "future_aggressive_buy_burst_1s": int(is_burst(buy_volume, buy_threshold_info)),
        "future_aggressive_sell_burst_1s": int(is_burst(sell_volume, sell_threshold_info)),
        "buy_burst_threshold_1s": (
            buy_threshold_info["threshold"]
            if np.isfinite(buy_threshold_info["threshold"])
            else np.nan
        ),
        "sell_burst_threshold_1s": (
            sell_threshold_info["threshold"]
            if np.isfinite(sell_threshold_info["threshold"])
            else np.nan
        ),
        "buy_burst_minimum_1s": (
            buy_threshold_info["minimum"]
            if np.isfinite(buy_threshold_info["minimum"])
            else np.nan
        ),
        "sell_burst_minimum_1s": (
            sell_threshold_info["minimum"]
            if np.isfinite(sell_threshold_info["minimum"])
            else np.nan
        ),
        "buy_burst_positive_history_count": buy_threshold_info["positive_history_count"],
        "sell_burst_positive_history_count": sell_threshold_info["positive_history_count"],
        "next_timestamp": int(next_row["timestamp"]),
        "next_time": next_row.get("time", ""),
        "future_total_trade_volume_1s": total_volume,
    }
    for horizon_seconds in [3, 5]:
        window, reason = future_window(frame, index, horizon_seconds)
        if window is None:
            return None, reason
        aggregate = aggregate_future_flow(window)
        targets[f"future_market_buy_volume_{horizon_seconds}s"] = aggregate["buy_volume"]
        targets[f"future_market_sell_volume_{horizon_seconds}s"] = aggregate["sell_volume"]
        targets[f"future_market_pressure_{horizon_seconds}s"] = aggregate["pressure"]
        targets[f"future_trade_count_{horizon_seconds}s"] = aggregate["trade_count"]
        targets[f"future_total_trade_volume_{horizon_seconds}s"] = aggregate["total_volume"]
        targets[f"future_log_market_buy_volume_{horizon_seconds}s"] = float(
            np.log1p(max(0.0, aggregate["buy_volume"]))
        )
        targets[f"future_log_market_sell_volume_{horizon_seconds}s"] = float(
            np.log1p(max(0.0, aggregate["sell_volume"]))
        )
        targets[f"future_log_trade_count_{horizon_seconds}s"] = float(
            np.log1p(max(0.0, aggregate["trade_count"]))
        )
        targets[f"future_{horizon_seconds}s_flow_class"] = aggregate["flow_class"]
    return targets, None


def build_training_rows(frame):
    snapshot_step_seconds = infer_snapshot_step_seconds(frame)
    rows = []
    skipped = {}
    next_row_gaps = []
    feature_ready_true = 0
    feature_ready_false = 0
    for index in range(len(frame)):
        if index + 1 < len(frame):
            current_timestamp = int(frame.loc[index, "timestamp"])
            next_timestamp = int(frame.loc[index + 1, "timestamp"])
            next_row_gaps.append(next_timestamp - current_timestamp)

        features, feature_reason = build_feature_row(frame, index, snapshot_step_seconds)
        if features is None:
            feature_ready_false += 1
            skipped[feature_reason] = skipped.get(feature_reason, 0) + 1
            continue
        feature_ready_true += 1
        targets, target_reason = build_target_row(frame, index)
        if targets is None:
            skipped[target_reason] = skipped.get(target_reason, 0) + 1
            continue
        rows.append({**features, **targets})

    diagnostics = {
        "feature_ready_true": feature_ready_true,
        "feature_ready_false": feature_ready_false,
        "next_row_gaps": next_row_gaps,
    }
    return pd.DataFrame(rows), skipped, snapshot_step_seconds, diagnostics


def feature_columns(frame):
    return sorted(
        column for column in frame.columns
        if isinstance(column, str) and column.startswith("feature_") and column != "feature_ready"
    )


def print_distribution(rows):
    print("Target distributions")
    if len(rows) == 0:
        print("- no rows")
        return
    total = len(rows)
    print("flow_class_1s:")
    for class_id, name in CLASS_NAMES.items():
        count = int((rows["next_1s_flow_class"] == class_id).sum())
        print(f"- {name}: {count}/{total} ({count / total:.2%})")
    for horizon_seconds in [3, 5]:
        column = f"future_{horizon_seconds}s_flow_class"
        if column not in rows.columns:
            continue
        print(f"flow_class_{horizon_seconds}s:")
        for class_id, name in CLASS_NAMES.items():
            count = int((rows[column] == class_id).sum())
            print(f"- {name}: {count}/{total} ({count / total:.2%})")
    burst_columns = [
        ("buy_burst_1s", "future_aggressive_buy_burst_1s"),
        ("sell_burst_1s", "future_aggressive_sell_burst_1s"),
    ]
    for label, column in burst_columns:
        count = int(pd.to_numeric(rows[column], errors="coerce").fillna(0).sum())
        positive_rate = count / total
        print(f"{label}: {count}/{total} positive ({positive_rate:.2%})")
        if positive_rate < 0.01:
            print(f"WARNING: {label} positive rate is below 1%; target may be too rare.")
        if positive_rate > 0.30:
            print(f"WARNING: {label} positive rate is above 30%; burst threshold may be too loose.")


def print_next_gap_stats(next_row_gaps):
    print("Next-row gap stats")
    if not next_row_gaps:
        print("- no adjacent rows available")
        return
    gaps = pd.Series(next_row_gaps, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(gaps) == 0:
        print("- no numeric adjacent gaps available")
        return
    too_large = int((gaps > MAX_NEXT_SECOND_GAP_MS).sum())
    print(f"- count: {len(gaps)}")
    print(f"- min ms: {float(gaps.min()):.0f}")
    print(f"- median ms: {float(gaps.median()):.0f}")
    print(f"- p90 ms: {float(gaps.quantile(0.90)):.0f}")
    print(f"- p99 ms: {float(gaps.quantile(0.99)):.0f}")
    print(f"- max ms: {float(gaps.max()):.0f}")
    print(f"- gaps > MAX_NEXT_SECOND_GAP_MS: {too_large}")


def main():
    snapshots = load_snapshot_rows(INPUT_PATH)
    original_count = len(snapshots)
    if MAX_FLOW_1S_SNAPSHOTS > 0 and len(snapshots) > MAX_FLOW_1S_SNAPSHOTS:
        snapshots = snapshots.tail(MAX_FLOW_1S_SNAPSHOTS).reset_index(drop=True)
    rows, skipped, snapshot_step_seconds, diagnostics = build_training_rows(snapshots)
    rows_before_cap = len(rows)
    if MAX_1S_FLOW_TRAINING_ROWS > 0 and len(rows) > MAX_1S_FLOW_TRAINING_ROWS:
        rows = rows.tail(MAX_1S_FLOW_TRAINING_ROWS).reset_index(drop=True)
    if len(rows):
        atomic_write_csv(rows, OUTPUT_PATH)

    print("1s order-flow training row builder")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Input snapshot path: {INPUT_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Input rows available: {original_count}")
    print(
        f"Input rows used: {len(snapshots)}"
        + (f" (MAX_FLOW_1S_SNAPSHOTS={MAX_FLOW_1S_SNAPSHOTS})" if MAX_FLOW_1S_SNAPSHOTS > 0 else "")
    )
    print(f"Inferred snapshot step seconds: {snapshot_step_seconds:.3g}")
    print(f"MAX_NEXT_SECOND_GAP_MS: {MAX_NEXT_SECOND_GAP_MS}")
    print(f"FLOW_1S_PRESSURE_CLASS_THRESHOLD: {FLOW_1S_PRESSURE_CLASS_THRESHOLD:.3f}")
    print(f"FLOW_1S_MIN_DIRECTIONAL_VOLUME: {FLOW_1S_MIN_DIRECTIONAL_VOLUME:.8g}")
    print(f"FLOW_1S_MIN_DIRECTIONAL_TRADE_COUNT: {FLOW_1S_MIN_DIRECTIONAL_TRADE_COUNT:.8g}")
    print(f"FLOW_1S_BURST_QUANTILE: {BURST_QUANTILE:.2f}")
    print(f"FLOW_1S_BURST_LOOKBACK_ROWS: {BURST_LOOKBACK_ROWS}")
    print(f"FLOW_1S_MIN_BURST_LOOKBACK_ROWS: {MIN_BURST_LOOKBACK_ROWS}")
    print(
        "FLOW_1S_MIN_BUY_BURST_VOLUME: "
        f"{MIN_BUY_BURST_VOLUME if MIN_BUY_BURST_VOLUME is not None else 'adaptive rolling positive median'}"
    )
    print(
        "FLOW_1S_MIN_SELL_BURST_VOLUME: "
        f"{MIN_SELL_BURST_VOLUME if MIN_SELL_BURST_VOLUME is not None else 'adaptive rolling positive median'}"
    )
    print(f"MAX_1S_FLOW_TRAINING_ROWS: {MAX_1S_FLOW_TRAINING_ROWS}")
    print("Feature_ready counts")
    print(f"- true: {diagnostics['feature_ready_true']}")
    print(f"- false: {diagnostics['feature_ready_false']}")
    print(f"Generated rows before cap: {rows_before_cap}")
    print(f"Generated rows: {len(rows)}")
    if len(rows):
        print(f"First timestamp: {int(rows['timestamp'].min())}")
        print(f"Last timestamp: {int(rows['timestamp'].max())}")
        print(f"Feature count: {len(feature_columns(rows))}")
        print_distribution(rows)
    print_next_gap_stats(diagnostics["next_row_gaps"])
    print("Skip reasons")
    if skipped:
        for reason, count in sorted(skipped.items()):
            print(f"- {reason}: {count}")
    else:
        print("- none")
    print("No trades were placed. No orders were sent.")


if __name__ == "__main__":
    main()
