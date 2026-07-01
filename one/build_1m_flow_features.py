import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
INPUT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_1m_flow.csv"
OUTPUT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_1m_flow_features.csv"

EPSILON = 1e-12
ROLLING_WINDOW = 20
DEPTH_CHANGE_RATIO_CLIP = float(os.getenv("DEPTH_CHANGE_RATIO_CLIP", "5"))
CHANGE_CLIP = float(os.getenv("CHANGE_CLIP", "2"))
STRICT_ZERO_VOLUME_READY = os.getenv("STRICT_ZERO_VOLUME_READY", "false").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}

REQUIRED_COLUMNS = [
    "timestamp",
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "taker_buy_volume",
    "taker_sell_volume",
    "spread_percent",
    "best_bid",
    "best_ask",
    "mid_price",
    "bid_depth_10bps",
    "ask_depth_10bps",
    "bid_depth_25bps",
    "ask_depth_25bps",
    "order_book_imbalance_10bps",
    "order_book_imbalance_25bps",
]

DERIVED_FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "range_percent",
    "close_position_in_range",
    "volume_zscore_20",
    "trade_count_zscore_20",
    "spread_zscore_20",
    "market_pressure",
    "market_pressure_change",
    "order_book_imbalance_10bps_change",
    "order_book_imbalance_25bps_change",
    "bid_depth_change_ratio_10bps",
    "ask_depth_change_ratio_10bps",
    "bid_depth_change_ratio_25bps",
    "ask_depth_change_ratio_25bps",
    "flow_pressure_index",
    "liquidity_support_index",
    "liquidity_resistance_index",
    "breakout_pressure_index",
    "absorption_index",
]


def safe_divide(numerator, denominator):
    numerator = pd.Series(numerator, copy=False)
    denominator = pd.Series(denominator, copy=False)
    result = pd.Series(0.0, index=numerator.index, dtype="float64")
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator.abs() > EPSILON)
    )
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def rolling_zscore(values, window=ROLLING_WINDOW):
    mean = values.rolling(window=window, min_periods=window).mean()
    std = values.rolling(window=window, min_periods=window).std(ddof=0)
    zscore = safe_divide(values - mean, std)
    zscore[mean.isna() | std.isna()] = np.nan
    return zscore


def clip_series(values, lower, upper):
    return values.clip(lower=lower, upper=upper)


def load_flow_rows():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}. Run npm run record-realtime first."
        )

    frame = pd.read_csv(INPUT_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    frame = frame.reset_index(drop=True)

    for column in frame.columns:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def add_optional_column(frame, column, default_value=0.0):
    if column not in frame.columns:
        frame[column] = default_value


def build_features(frame):
    result = frame.copy()

    add_optional_column(result, "snapshot_market_pressure_10s_last", 0.0)
    add_optional_column(result, "snapshot_imbalance_10bps_last", 0.0)
    add_optional_column(result, "snapshot_imbalance_25bps_last", 0.0)

    candle_range = result["high"] - result["low"]
    previous_close = result["close"].shift(1)

    result["return_1"] = safe_divide(result["close"] - previous_close, previous_close)
    result["return_3"] = safe_divide(
        result["close"] - result["close"].shift(3),
        result["close"].shift(3),
    )
    result["range_percent"] = safe_divide(candle_range, result["close"])
    result["close_position_in_range"] = safe_divide(
        result["close"] - result["low"],
        candle_range,
    ).fillna(0.5)
    result["close_position_in_range"] = clip_series(
        result["close_position_in_range"],
        0.0,
        1.0,
    )

    result["volume_zscore_20"] = rolling_zscore(result["volume"])
    result["trade_count_zscore_20"] = rolling_zscore(result["trade_count"])
    result["spread_zscore_20"] = rolling_zscore(result["spread_percent"])

    result["market_pressure"] = safe_divide(
        result["taker_buy_volume"] - result["taker_sell_volume"],
        result["volume"],
    )
    result["market_pressure"] = clip_series(result["market_pressure"], -1.0, 1.0)
    result["market_pressure_change"] = clip_series(
        result["market_pressure"].diff().fillna(0.0),
        -CHANGE_CLIP,
        CHANGE_CLIP,
    )
    result["order_book_imbalance_10bps_change"] = (
        clip_series(
            result["order_book_imbalance_10bps"].diff().fillna(0.0),
            -CHANGE_CLIP,
            CHANGE_CLIP,
        )
    )
    result["order_book_imbalance_25bps_change"] = (
        clip_series(
            result["order_book_imbalance_25bps"].diff().fillna(0.0),
            -CHANGE_CLIP,
            CHANGE_CLIP,
        )
    )

    result["bid_depth_change_ratio_10bps"] = clip_series(
        safe_divide(
            result["bid_depth_10bps"] - result["bid_depth_10bps"].shift(1),
            result["bid_depth_10bps"].shift(1),
        ),
        -DEPTH_CHANGE_RATIO_CLIP,
        DEPTH_CHANGE_RATIO_CLIP,
    )
    result["ask_depth_change_ratio_10bps"] = clip_series(
        safe_divide(
            result["ask_depth_10bps"] - result["ask_depth_10bps"].shift(1),
            result["ask_depth_10bps"].shift(1),
        ),
        -DEPTH_CHANGE_RATIO_CLIP,
        DEPTH_CHANGE_RATIO_CLIP,
    )
    result["bid_depth_change_ratio_25bps"] = clip_series(
        safe_divide(
            result["bid_depth_25bps"] - result["bid_depth_25bps"].shift(1),
            result["bid_depth_25bps"].shift(1),
        ),
        -DEPTH_CHANGE_RATIO_CLIP,
        DEPTH_CHANGE_RATIO_CLIP,
    )
    result["ask_depth_change_ratio_25bps"] = clip_series(
        safe_divide(
            result["ask_depth_25bps"] - result["ask_depth_25bps"].shift(1),
            result["ask_depth_25bps"].shift(1),
        ),
        -DEPTH_CHANGE_RATIO_CLIP,
        DEPTH_CHANGE_RATIO_CLIP,
    )

    depth_total_10bps = result["bid_depth_10bps"] + result["ask_depth_10bps"]
    depth_total_25bps = result["bid_depth_25bps"] + result["ask_depth_25bps"]
    bid_share_10bps = safe_divide(result["bid_depth_10bps"], depth_total_10bps)
    ask_share_10bps = safe_divide(result["ask_depth_10bps"], depth_total_10bps)
    bid_share_25bps = safe_divide(result["bid_depth_25bps"], depth_total_25bps)
    ask_share_25bps = safe_divide(result["ask_depth_25bps"], depth_total_25bps)
    directional_close_position = result["close_position_in_range"] * 2.0 - 1.0
    snapshot_pressure = result["snapshot_market_pressure_10s_last"].fillna(0.0)
    snapshot_imbalance_10 = result["snapshot_imbalance_10bps_last"].fillna(0.0)
    snapshot_imbalance_25 = result["snapshot_imbalance_25bps_last"].fillna(0.0)

    result["flow_pressure_index"] = (
        result["market_pressure"]
        + result["order_book_imbalance_10bps"]
        + result["order_book_imbalance_25bps"]
        + snapshot_pressure
    ) / 4.0
    result["liquidity_support_index"] = (bid_share_10bps + bid_share_25bps) / 2.0
    result["liquidity_resistance_index"] = (ask_share_10bps + ask_share_25bps) / 2.0
    result["breakout_pressure_index"] = (
        result["flow_pressure_index"]
        + directional_close_position
        + result["market_pressure_change"]
        + result["order_book_imbalance_10bps_change"]
    ) / 4.0
    result["absorption_index"] = (
        result["liquidity_resistance_index"] * clip_series(result["market_pressure"], 0, 1)
        + result["liquidity_support_index"] * clip_series(-result["market_pressure"], 0, 1)
        + (1.0 - result["range_percent"].rank(pct=True).fillna(0.5))
        + (snapshot_imbalance_10.abs() + snapshot_imbalance_25.abs()) / 2.0
    ) / 4.0

    readiness_reasons = {
        "insufficient_rolling_history": result.index < ROLLING_WINDOW - 1,
        "missing_derived_features": result[DERIVED_FEATURE_COLUMNS].isna().any(axis=1),
        "zero_volume_or_trade_count": result["volume"].isna()
        | result["trade_count"].isna()
        | (result["volume"] <= 0)
        | (result["trade_count"] <= 0),
        "invalid_best_bid_ask_or_mid": result["best_bid"].isna()
        | result["best_ask"].isna()
        | result["mid_price"].isna()
        | (result["best_bid"] <= 0)
        | (result["best_ask"] <= 0)
        | (result["mid_price"] <= 0),
        "zero_10bps_bid_or_ask_depth": result["bid_depth_10bps"].isna()
        | result["ask_depth_10bps"].isna()
        | (result["bid_depth_10bps"] <= 0)
        | (result["ask_depth_10bps"] <= 0),
        "after_timestamp_gap_gt_60s": result["timestamp"].diff().fillna(60_000)
        > 60_000,
    }
    not_ready_mask = pd.Series(False, index=result.index)

    non_blocking_reasons = set()
    if not STRICT_ZERO_VOLUME_READY:
        # A quiet one-minute candle with no trades is a real market state. It
        # should be visible to the model as zero flow instead of breaking every
        # contiguous lookback window. We still report it for health diagnostics.
        non_blocking_reasons.add("zero_volume_or_trade_count")

    for reason, mask in readiness_reasons.items():
        result[f"not_ready_{reason}"] = mask
        if reason not in non_blocking_reasons:
            not_ready_mask = not_ready_mask | mask

    result["feature_ready"] = ~not_ready_mask
    result.attrs["readiness_reason_counts"] = {
        reason: int(mask.sum()) for reason, mask in readiness_reasons.items()
    }

    return result


def print_missing_values(frame):
    missing = frame.isna().sum()
    missing = missing[missing > 0]

    print("\nMissing values:")
    if len(missing) == 0:
        print("- none")
        return

    for column, count in missing.items():
        print(f"- {column}: {int(count)}")


def print_feature_summary(frame):
    print("\nDerived feature summary:")
    summary = frame[DERIVED_FEATURE_COLUMNS].describe().transpose()

    for column, row in summary.iterrows():
        print(
            f"- {column}: "
            f"min={row['min']:.10g}, "
            f"mean={row['mean']:.10g}, "
            f"max={row['max']:.10g}, "
            f"std={row['std']:.10g}"
        )


def main():
    frame = load_flow_rows()
    featured = build_features(frame)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(OUTPUT_PATH, index=False)

    ready_count = int(featured["feature_ready"].sum())
    not_ready_count = len(featured) - ready_count

    print("1m realtime flow feature builder")
    print(f"Symbol: {SYMBOL}")
    print(f"Primary venue: {PRIMARY_VENUE or 'legacy'}")
    print(f"Input path: {INPUT_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Input row count: {len(frame)}")
    print(f"Output row count: {len(featured)}")
    print(f"Feature-ready rows: {ready_count}")
    print(f"Rows marked not ready: {not_ready_count}")
    print(f"Rolling window: {ROLLING_WINDOW}")
    print(f"Depth change ratio clip: +/-{DEPTH_CHANGE_RATIO_CLIP}")
    print(f"Pressure/imbalance change clip: +/-{CHANGE_CLIP}")
    print(f"STRICT_ZERO_VOLUME_READY: {STRICT_ZERO_VOLUME_READY}")
    print("Rows marked not ready by reason:")
    for reason, count in featured.attrs.get("readiness_reason_counts", {}).items():
        print(f"- {reason}: {count}")
    print_missing_values(featured)
    print_feature_summary(featured)
    print("\nNo training was run.")


if __name__ == "__main__":
    main()
