import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE_FALLBACK = os.getenv("VENUE", os.getenv("VENUES", "")).split(",")[0].strip().lower()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", PRIMARY_VENUE_FALLBACK).strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
INPUT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_1m_flow.csv"
OUTPUT_15M_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_15m_regime_features.csv"
OUTPUT_30M_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_30m_regime_features.csv"
EPSILON = 1e-8


def safe_ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return 0.0
    if abs(denominator) < EPSILON:
        return 0.0
    return float(numerator / denominator)


def ema(values, period):
    result = np.full(len(values), np.nan, dtype=np.float64)
    multiplier = 2.0 / (period + 1.0)
    previous = np.nan
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if not np.isfinite(previous):
            if index < period - 1:
                continue
            seed = values[index - period + 1 : index + 1]
            if not np.all(np.isfinite(seed)):
                continue
            previous = float(np.mean(seed))
        else:
            previous = value * multiplier + previous * (1.0 - multiplier)
        result[index] = previous
    return result


def rsi(close, period=14):
    result = np.full(len(close), np.nan, dtype=np.float64)
    for index in range(period, len(close)):
        changes = np.diff(close[index - period : index + 1])
        gains = np.where(changes > 0, changes, 0.0).sum()
        losses = np.where(changes < 0, -changes, 0.0).sum()
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss < EPSILON:
            result[index] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[index] = 100.0 - 100.0 / (1.0 + rs)
    return result


def atr_percent(high, low, close, period=14):
    true_range = np.zeros(len(close), dtype=np.float64)
    result = np.full(len(close), np.nan, dtype=np.float64)
    for index in range(len(close)):
        previous_close = close[index - 1] if index > 0 else close[index]
        true_range[index] = max(
            high[index] - low[index],
            abs(high[index] - previous_close),
            abs(low[index] - previous_close),
        )
        if index >= period - 1:
            result[index] = safe_ratio(np.mean(true_range[index - period + 1 : index + 1]), close[index])
    return result


def rolling_volatility(close, period=20):
    returns = np.full(len(close), np.nan, dtype=np.float64)
    result = np.full(len(close), np.nan, dtype=np.float64)
    for index in range(1, len(close)):
        returns[index] = safe_ratio(close[index] - close[index - 1], close[index - 1])
    for index in range(period, len(close)):
        result[index] = np.std(returns[index - period + 1 : index + 1])
    return result


def assign_regime(row):
    trend_score = row["trend_score"]
    chop_score = row["chop_score"]
    volatility = row["rolling_volatility_20"]
    if not np.isfinite(trend_score) or not np.isfinite(chop_score):
        return "unknown"
    if volatility > 0.008 and abs(trend_score) < 0.5:
        return "high_volatility_chop"
    if trend_score >= 1.0:
        return "bullish"
    if trend_score <= -1.0:
        return "bearish"
    return "chop"


def load_1m_rows():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing realtime 1m file: {INPUT_PATH}")
    frame = pd.read_csv(INPUT_PATH)
    required = ["timestamp", "time", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Input 1m CSV is missing required columns: {missing}")
    for column in frame.columns:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).sort_values("timestamp").drop_duplicates("timestamp")
    return frame.reset_index(drop=True)


def aggregate_closed_1m(frame, timeframe_minutes):
    interval_ms = timeframe_minutes * 60 * 1000
    expected_rows = timeframe_minutes
    latest_timestamp = int(frame["timestamp"].max()) if len(frame) else 0
    latest_closed_open = ((latest_timestamp + 60_000) // interval_ms) * interval_ms - interval_ms

    frame = frame.copy()
    frame[f"timestamp_{timeframe_minutes}m"] = (frame["timestamp"] // interval_ms) * interval_ms
    frame = frame[frame[f"timestamp_{timeframe_minutes}m"] <= latest_closed_open]
    grouped = frame.groupby(f"timestamp_{timeframe_minutes}m", sort=True)
    result = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        source_rows=("timestamp", "count"),
    ).reset_index()
    result = result[result["source_rows"] == expected_rows].copy()
    result = result.rename(columns={f"timestamp_{timeframe_minutes}m": "timestamp"})
    result["close_timestamp"] = result["timestamp"] + interval_ms
    result["time"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )
    return result


def add_regime_features(candles):
    candles = candles.copy().sort_values("timestamp").reset_index(drop=True)
    if len(candles) == 0:
        return candles
    close = candles["close"].to_numpy(dtype=np.float64)
    high = candles["high"].to_numpy(dtype=np.float64)
    low = candles["low"].to_numpy(dtype=np.float64)
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    candles["return_1"] = pd.Series(close).pct_change(1).fillna(0.0)
    candles["return_4"] = pd.Series(close).pct_change(4).fillna(0.0)
    candles["return_12"] = pd.Series(close).pct_change(12).fillna(0.0)
    candles["ema20"] = ema20
    candles["ema50"] = ema50
    candles["ema20_distance"] = (close - ema20) / close
    candles["ema50_distance"] = (close - ema50) / close
    candles["ema20_slope_4"] = pd.Series(ema20).pct_change(4)
    candles["ema50_slope_4"] = pd.Series(ema50).pct_change(4)
    candles["rsi14"] = rsi(close, 14)
    candles["atr14_percent"] = atr_percent(high, low, close, 14)
    candles["rolling_volatility_20"] = rolling_volatility(close, 20)

    bullish_points = (
        (candles["close"] > candles["ema20"]).astype(float)
        + (candles["ema20"] > candles["ema50"]).astype(float)
        + (candles["ema20_slope_4"] > 0).astype(float)
        + (candles["rsi14"] > 52).astype(float)
    )
    bearish_points = (
        (candles["close"] < candles["ema20"]).astype(float)
        + (candles["ema20"] < candles["ema50"]).astype(float)
        + (candles["ema20_slope_4"] < 0).astype(float)
        + (candles["rsi14"] < 48).astype(float)
    )
    candles["trend_score"] = bullish_points - bearish_points
    candles["chop_score"] = 4.0 - candles["trend_score"].abs()
    candles["regime"] = candles.apply(assign_regime, axis=1)
    candles = candles.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return candles


def write_regime(frame, path, timeframe_minutes):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    gaps = int((frame["timestamp"].diff().dropna() != timeframe_minutes * 60 * 1000).sum()) if len(frame) else 0
    print(f"{timeframe_minutes}m output path: {path}")
    print(f"{timeframe_minutes}m regime rows: {len(frame)}")
    print(f"{timeframe_minutes}m missing gaps: {gaps}")
    if len(frame):
        print(f"{timeframe_minutes}m first row: {frame.iloc[0]['time']}")
        print(f"{timeframe_minutes}m last row: {frame.iloc[-1]['time']}")
        print(f"{timeframe_minutes}m regime distribution:")
        print(frame["regime"].value_counts())


def main():
    rows_1m = load_1m_rows()
    candles_15m = add_regime_features(aggregate_closed_1m(rows_1m, 15))
    candles_30m = add_regime_features(aggregate_closed_1m(rows_1m, 30))
    write_regime(candles_15m, OUTPUT_15M_PATH, 15)
    write_regime(candles_30m, OUTPUT_30M_PATH, 30)
    print("Live regime updater complete.")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Input path: {INPUT_PATH}")
    print(f"Input 1m rows: {len(rows_1m)}")
    print("Only completed 15m/30m candles were used. No trades were placed.")


if __name__ == "__main__":
    main()
