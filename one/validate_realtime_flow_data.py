import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

TEN_SECOND_PATH = OUTPUT_DIR / f"{SYMBOL}_10s_flow.csv"
ONE_MINUTE_PATH = OUTPUT_DIR / f"{SYMBOL}_1m_flow.csv"

PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

KEY_NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_sell_volume",
    "taker_buy_ratio",
    "mid_price",
    "best_bid",
    "best_ask",
    "spread_percent",
    "market_buy_volume_10s",
    "market_sell_volume_10s",
    "total_trade_volume_10s",
    "trade_count_10s",
    "market_pressure_10s",
    "bid_depth_10bps",
    "ask_depth_10bps",
    "bid_depth_25bps",
    "ask_depth_25bps",
    "order_book_imbalance_10bps",
    "order_book_imbalance_25bps",
    "snapshot_spread_percent_mean",
    "snapshot_spread_percent_last",
    "snapshot_imbalance_10bps_mean",
    "snapshot_imbalance_10bps_last",
    "snapshot_bid_depth_10bps_mean",
    "snapshot_bid_depth_10bps_last",
    "snapshot_ask_depth_10bps_mean",
    "snapshot_ask_depth_10bps_last",
    "snapshot_market_pressure_10s_mean",
    "snapshot_market_pressure_10s_last",
]


def percent(value):
    return f"{value * 100:.2f}%"


def print_section(title):
    print(f"\n{title}")
    print("-" * len(title))


def load_csv(path):
    if not path.exists():
        print(f"Missing file: {path}")
        return None

    if path.stat().st_size == 0:
        print(f"Empty file: {path}")
        return None

    return pd.read_csv(path)


def coerce_numeric_columns(frame):
    numeric_frame = frame.copy()

    for column in numeric_frame.columns:
        if column == "time":
            continue
        numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")

    return numeric_frame


def print_empty_nan_counts(frame):
    print("\nEmpty/NaN count by column:")
    any_empty = False

    for column in frame.columns:
        empty_count = frame[column].isna().sum()

        if frame[column].dtype == object:
            empty_count += (frame[column].astype(str).str.strip() == "").sum()

        if empty_count > 0:
            any_empty = True
            print(f"- {column}: {int(empty_count)}")

    if not any_empty:
        print("- none detected")


def print_numeric_stats(frame):
    print("\nNumeric min/max/mean for key columns:")
    printed = False

    for column in KEY_NUMERIC_COLUMNS:
        if column not in frame.columns:
            continue

        values = pd.to_numeric(frame[column], errors="coerce").dropna()

        if len(values) == 0:
            continue

        printed = True
        print(
            f"- {column}: min={values.min():.10g}, "
            f"max={values.max():.10g}, mean={values.mean():.10g}"
        )

    if not printed:
        print("- no key numeric columns found")


def print_percentiles(frame, label, candidate_columns):
    column = next((name for name in candidate_columns if name in frame.columns), None)

    if column is None:
        print(f"\n{label} percentiles: column not found")
        return

    values = pd.to_numeric(frame[column], errors="coerce").dropna()

    if len(values) == 0:
        print(f"\n{label} percentiles ({column}): no numeric values")
        return

    print(f"\n{label} percentiles ({column}):")
    for percentile_value in PERCENTILES:
        value = np.percentile(values, percentile_value)
        print(f"- p{percentile_value}: {value:.10g}")


def timestamp_diagnostics(frame, expected_interval_ms):
    if "timestamp" not in frame.columns:
        print("Timestamp column missing.")
        return

    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce").dropna()

    if len(timestamps) == 0:
        print("No valid timestamps.")
        return

    timestamps = timestamps.astype(np.int64).sort_values()
    duplicate_count = int(timestamps.duplicated().sum())
    unique_timestamps = timestamps.drop_duplicates()
    diffs = unique_timestamps.diff().dropna()
    gap_diffs = diffs[diffs > expected_interval_ms]
    missing_intervals = int(
        np.maximum(np.round(gap_diffs / expected_interval_ms).astype(int) - 1, 0).sum()
    )

    print(f"First timestamp: {pd.to_datetime(timestamps.iloc[0], unit='ms', utc=True)}")
    print(f"Last timestamp: {pd.to_datetime(timestamps.iloc[-1], unit='ms', utc=True)}")
    print(f"Duplicate timestamps: {duplicate_count}")
    print(f"Missing interval gaps: {len(gap_diffs)}")
    print(f"Estimated missing rows inside gaps: {missing_intervals}")

    if len(gap_diffs) > 0:
        print("First gaps:")
        gap_positions = np.flatnonzero(diffs.to_numpy() > expected_interval_ms)

        for position in gap_positions[:5]:
            gap = diffs.iloc[position]
            before = unique_timestamps.iloc[position]
            after = unique_timestamps.iloc[position + 1]
            print(f"- gap_ms={int(gap)}, before={before}, after={after}")


def count_zero_volume_or_trades(frame):
    checks = []

    if "volume" in frame.columns:
        checks.append(("volume", pd.to_numeric(frame["volume"], errors="coerce") <= 0))
    if "trade_count" in frame.columns:
        checks.append(
            ("trade_count", pd.to_numeric(frame["trade_count"], errors="coerce") <= 0)
        )
    if "total_trade_volume_10s" in frame.columns:
        checks.append(
            (
                "total_trade_volume_10s",
                pd.to_numeric(frame["total_trade_volume_10s"], errors="coerce") <= 0,
            )
        )
    if "trade_count_10s" in frame.columns:
        checks.append(
            (
                "trade_count_10s",
                pd.to_numeric(frame["trade_count_10s"], errors="coerce") <= 0,
            )
        )

    print("\nRows with zero volume/trade_count:")
    if not checks:
        print("- no volume/trade count columns found")
        return

    for column, mask in checks:
        print(f"- {column}: {int(mask.sum())}")


def count_zero_depth(frame):
    print("\nRows with zero bid or ask depth:")
    printed = False

    for bid_column, ask_column in [
        ("bid_depth_10bps", "ask_depth_10bps"),
        ("bid_depth_25bps", "ask_depth_25bps"),
    ]:
        if bid_column not in frame.columns or ask_column not in frame.columns:
            continue

        bid = pd.to_numeric(frame[bid_column], errors="coerce")
        ask = pd.to_numeric(frame[ask_column], errors="coerce")
        zero_count = int(((bid <= 0) | (ask <= 0)).sum())
        printed = True
        print(f"- {bid_column}/{ask_column}: {zero_count}")

    if not printed:
        print("- depth columns not found")


def invalid_taker_buy_ratio_count(frame):
    if "taker_buy_ratio" not in frame.columns:
        return None

    values = pd.to_numeric(frame["taker_buy_ratio"], errors="coerce")
    return int(((values < 0) | (values > 1)).sum())


def invalid_market_pressure_count(frame):
    pressure_columns = [
        column for column in frame.columns if "market_pressure" in column
    ]

    if not pressure_columns:
        return None

    invalid_rows = pd.Series(False, index=frame.index)

    for column in pressure_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid_rows = invalid_rows | (values < -1) | (values > 1)

    return int(invalid_rows.sum())


def validate_file(path, expected_interval_ms, label):
    print_section(f"{label}: {path}")
    raw_frame = load_csv(path)

    if raw_frame is None:
        return

    numeric_frame = coerce_numeric_columns(raw_frame)

    print(f"Row count: {len(raw_frame)}")
    timestamp_diagnostics(numeric_frame, expected_interval_ms)
    print_empty_nan_counts(raw_frame)
    print_numeric_stats(numeric_frame)

    print_percentiles(numeric_frame, "spread_percent", ["spread_percent"])
    print_percentiles(
        numeric_frame,
        "market_pressure",
        [
            "market_pressure_10s",
            "snapshot_market_pressure_10s_last",
            "snapshot_market_pressure_10s_mean",
        ],
    )
    print_percentiles(
        numeric_frame,
        "order_book_imbalance_10bps",
        ["order_book_imbalance_10bps", "snapshot_imbalance_10bps_last"],
    )

    count_zero_volume_or_trades(numeric_frame)
    count_zero_depth(numeric_frame)

    if label == "1m flow":
        invalid_ratio = invalid_taker_buy_ratio_count(numeric_frame)
        if invalid_ratio is not None:
            print(f"\nInvalid taker_buy_ratio outside 0..1: {invalid_ratio}")

    invalid_pressure = invalid_market_pressure_count(numeric_frame)
    if invalid_pressure is not None:
        print(f"Invalid market_pressure outside -1..1: {invalid_pressure}")


def main():
    print("Realtime flow data validator")
    print(f"Symbol: {SYMBOL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Read-only diagnostics. No training will run.")

    validate_file(TEN_SECOND_PATH, 10_000, "10s flow")
    validate_file(ONE_MINUTE_PATH, 60_000, "1m flow")


if __name__ == "__main__":
    main()
