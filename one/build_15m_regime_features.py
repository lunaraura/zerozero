import os
import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "BTCUSDT").strip().upper()
INPUT_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_5m_imported.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_15m_regime_features.csv"
LEGACY_INPUT_PATH = PROJECT_ROOT / "data" / "btc_5m_imported.csv"

EPSILON = 1e-8


def safe_ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return 0.0
    return numerator / max(abs(denominator), EPSILON)


def ema(values, period):
    result = np.full(len(values), np.nan, dtype=np.float64)
    multiplier = 2.0 / (period + 1.0)
    previous = np.nan

    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue

        if not np.isfinite(previous):
            if i < period - 1:
                continue
            seed = values[i - period + 1 : i + 1]
            if not np.all(np.isfinite(seed)):
                continue
            previous = float(np.mean(seed))
        else:
            previous = value * multiplier + previous * (1.0 - multiplier)

        result[i] = previous

    return result


def rsi(close, period=14):
    result = np.full(len(close), np.nan, dtype=np.float64)

    for i in range(period, len(close)):
        changes = np.diff(close[i - period : i + 1])
        gains = np.where(changes > 0, changes, 0.0).sum()
        losses = np.where(changes < 0, -changes, 0.0).sum()

        avg_gain = gains / period
        avg_loss = losses / period

        if avg_loss < EPSILON:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)

    return result


def atr_percent(high, low, close, period=14):
    tr = np.zeros(len(close), dtype=np.float64)
    result = np.full(len(close), np.nan, dtype=np.float64)

    for i in range(len(close)):
        previous_close = close[i - 1] if i > 0 else close[i]
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - previous_close),
            abs(low[i] - previous_close),
        )

        if i >= period - 1:
            atr = np.mean(tr[i - period + 1 : i + 1])
            result[i] = safe_ratio(atr, close[i])

    return result


def rolling_volatility(close, period=20):
    returns = np.full(len(close), np.nan, dtype=np.float64)
    result = np.full(len(close), np.nan, dtype=np.float64)

    for i in range(1, len(close)):
        returns[i] = safe_ratio(close[i] - close[i - 1], close[i - 1])

    for i in range(period, len(close)):
        result[i] = np.std(returns[i - period + 1 : i + 1])

    return result


def aggregate_5m_to_15m(candles):
    candles = candles.copy()
    candles["timestamp"] = candles["timestamp"].astype(np.int64)

    # Binance timestamps are milliseconds. Group by 15-minute open timestamp.
    interval_ms = 15 * 60 * 1000
    candles["timestamp_15m"] = (candles["timestamp"] // interval_ms) * interval_ms

    grouped = candles.groupby("timestamp_15m", sort=True)

    result = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        source_rows=("timestamp", "count"),
    ).reset_index()

    result = result[result["source_rows"] == 3].copy()
    result = result.rename(columns={"timestamp_15m": "timestamp"})
    result["close_timestamp"] = result["timestamp"] + interval_ms
    result["time"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )

    return result


def assign_regime(row):
    trend_score = row["trend_score"]
    chop_score = row["chop_score"]
    vol = row["rolling_volatility_20"]

    if not np.isfinite(trend_score) or not np.isfinite(chop_score):
        return "unknown"

    if vol > 0.008 and abs(trend_score) < 0.5:
        return "high_volatility_chop"

    if trend_score >= 1.0:
        return "bullish"

    if trend_score <= -1.0:
        return "bearish"

    return "chop"


def main():
    input_path = INPUT_PATH
    if not input_path.exists() and SYMBOL == "BTCUSDT" and LEGACY_INPUT_PATH.exists():
        input_path = LEGACY_INPUT_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    print(f"Symbol: {SYMBOL}")
    print(f"Input path: {input_path}")
    print(f"Output path: {OUTPUT_PATH}")

    candles_5m = pd.read_csv(input_path)
    required = ["time", "timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in candles_5m.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    candles_5m = candles_5m.sort_values("timestamp").drop_duplicates("timestamp")
    candles_15m = aggregate_5m_to_15m(candles_5m)

    close = candles_15m["close"].to_numpy(dtype=np.float64)
    high = candles_15m["high"].to_numpy(dtype=np.float64)
    low = candles_15m["low"].to_numpy(dtype=np.float64)

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    rsi14 = rsi(close, 14)
    atr14_pct = atr_percent(high, low, close, 14)
    vol20 = rolling_volatility(close, 20)

    candles_15m["return_1"] = pd.Series(close).pct_change(1).fillna(0.0)
    candles_15m["return_4"] = pd.Series(close).pct_change(4).fillna(0.0)
    candles_15m["return_12"] = pd.Series(close).pct_change(12).fillna(0.0)

    candles_15m["ema20"] = ema20
    candles_15m["ema50"] = ema50
    candles_15m["ema20_distance"] = (close - ema20) / close
    candles_15m["ema50_distance"] = (close - ema50) / close
    candles_15m["ema20_slope_4"] = pd.Series(ema20).pct_change(4)
    candles_15m["ema50_slope_4"] = pd.Series(ema50).pct_change(4)
    candles_15m["rsi14"] = rsi14
    candles_15m["atr14_percent"] = atr14_pct
    candles_15m["rolling_volatility_20"] = vol20

    bullish_points = (
        (candles_15m["close"] > candles_15m["ema20"]).astype(float)
        + (candles_15m["ema20"] > candles_15m["ema50"]).astype(float)
        + (candles_15m["ema20_slope_4"] > 0).astype(float)
        + (candles_15m["rsi14"] > 52).astype(float)
    )

    bearish_points = (
        (candles_15m["close"] < candles_15m["ema20"]).astype(float)
        + (candles_15m["ema20"] < candles_15m["ema50"]).astype(float)
        + (candles_15m["ema20_slope_4"] < 0).astype(float)
        + (candles_15m["rsi14"] < 48).astype(float)
    )

    candles_15m["trend_score"] = bullish_points - bearish_points
    candles_15m["chop_score"] = 4.0 - candles_15m["trend_score"].abs()
    candles_15m["regime"] = candles_15m.apply(assign_regime, axis=1)

    # Remove early rows where indicators are not ready.
    candles_15m = candles_15m.replace([np.inf, -np.inf], np.nan)
    candles_15m = candles_15m.dropna().reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candles_15m.to_csv(OUTPUT_PATH, index=False)

    missing_gaps = int(
        ((candles_15m["timestamp"].diff().dropna()) != 15 * 60 * 1000).sum()
    )

    print("15m regime features built")
    print(f"Input 5m rows: {len(candles_5m)}")
    print(f"Output 15m rows: {len(candles_15m)}")
    print(f"First timestamp: {candles_15m.iloc[0]['time'] if len(candles_15m) else 'n/a'}")
    print(f"Last timestamp: {candles_15m.iloc[-1]['time'] if len(candles_15m) else 'n/a'}")
    print(f"Missing 15m gaps: {missing_gaps}")
    print("Regime distribution:")
    print(candles_15m["regime"].value_counts())


if __name__ == "__main__":
    main()
