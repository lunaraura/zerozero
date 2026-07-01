import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
INPUT_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_5m_imported.csv"
LEGACY_INPUT_PATH = PROJECT_ROOT / "data" / "btc_5m_imported.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_htf_context_features.csv"

EPSILON = 1e-12


def safe_ratio(numerator, denominator, default=0.0):
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return default
    if abs(denominator) < EPSILON:
        return default
    return float(numerator / denominator)


def ema(values, period):
    values = np.asarray(values, dtype=np.float64)
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
            previous = float(seed.mean())
        else:
            previous = value * multiplier + previous * (1.0 - multiplier)
        result[index] = previous
    return result


def rsi(close, period=14):
    close = np.asarray(close, dtype=np.float64)
    result = np.full(len(close), np.nan, dtype=np.float64)
    for index in range(period, len(close)):
        changes = np.diff(close[index - period : index + 1])
        gains = np.where(changes > 0, changes, 0.0).sum()
        losses = np.where(changes < 0, -changes, 0.0).sum()
        if losses < EPSILON:
            result[index] = 100.0
        else:
            rs = gains / losses
            result[index] = 100.0 - 100.0 / (1.0 + rs)
    return result


def rolling_volatility(close, period=20):
    returns = pd.Series(close, dtype="float64").pct_change()
    return returns.rolling(period, min_periods=period).std().to_numpy(dtype=np.float64)


def atr_percent(high, low, close, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    true_range = np.zeros(len(close), dtype=np.float64)
    for index in range(len(close)):
        previous_close = close[index - 1] if index > 0 else close[index]
        true_range[index] = max(
            high[index] - low[index],
            abs(high[index] - previous_close),
            abs(low[index] - previous_close),
        )
    atr = pd.Series(true_range).rolling(period, min_periods=period).mean()
    return (atr / pd.Series(close).replace(0, np.nan)).to_numpy(dtype=np.float64)


def softmax3(bear_score, chop_score, bull_score):
    values = np.asarray([bear_score, chop_score, bull_score], dtype=np.float64)
    values = values - np.nanmax(values)
    exp_values = np.exp(np.clip(values, -40, 40))
    probs = exp_values / max(exp_values.sum(), EPSILON)
    return probs[0], probs[1], probs[2]


def load_candles():
    path = INPUT_PATH
    if not path.exists() and SYMBOL == "BTCUSDT" and LEGACY_INPUT_PATH.exists():
        path = LEGACY_INPUT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing 5m imported data: {INPUT_PATH}")

    candles = pd.read_csv(path)
    required = ["time", "timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in candles.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    for column in required:
        if column != "time":
            candles[column] = pd.to_numeric(candles[column], errors="coerce")
    candles = candles.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    candles["timestamp"] = np.where(
        candles["timestamp"] < 10_000_000_000,
        candles["timestamp"] * 1000,
        candles["timestamp"],
    ).astype(np.int64)
    candles = candles.sort_values("timestamp").drop_duplicates("timestamp")
    return candles.reset_index(drop=True), path


def aggregate_5m_to_hourly(candles):
    interval_ms = 60 * 60 * 1000
    working = candles.copy()
    working["timestamp_hour"] = (working["timestamp"] // interval_ms) * interval_ms
    grouped = working.groupby("timestamp_hour", sort=True)
    hourly = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_like_volume=("close", lambda values: np.nan),
        source_rows=("timestamp", "count"),
    ).reset_index()
    hourly = hourly[hourly["source_rows"] == 12].copy()
    hourly = hourly.rename(columns={"timestamp_hour": "timestamp"})
    hourly["close_timestamp"] = hourly["timestamp"] + interval_ms
    hourly["time"] = pd.to_datetime(hourly["timestamp"], unit="ms", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )
    return hourly.reset_index(drop=True)


def add_context_features(hourly):
    close = hourly["close"].to_numpy(dtype=np.float64)
    high = hourly["high"].to_numpy(dtype=np.float64)
    low = hourly["low"].to_numpy(dtype=np.float64)
    volume = hourly["volume"].to_numpy(dtype=np.float64)

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    rsi14 = rsi(close, 14)
    vol20 = rolling_volatility(close, 20)
    atr14 = atr_percent(high, low, close, 14)

    frame = hourly.copy()
    frame["hourly_return_1"] = pd.Series(close).pct_change(1)
    frame["hourly_return_4"] = pd.Series(close).pct_change(4)
    frame["hourly_return_12"] = pd.Series(close).pct_change(12)
    frame["hourly_return_24"] = pd.Series(close).pct_change(24)
    frame["hourly_ema20_distance"] = (close - ema20) / close
    frame["hourly_ema50_distance"] = (close - ema50) / close
    frame["hourly_ema20_slope_4"] = pd.Series(ema20).pct_change(4)
    frame["hourly_ema50_slope_4"] = pd.Series(ema50).pct_change(4)
    frame["hourly_rsi14"] = rsi14
    frame["hourly_atr14_percent"] = atr14
    frame["hourly_rolling_volatility_20"] = vol20

    rolling_vwap_numerator = pd.Series(close * volume).rolling(24, min_periods=12).sum()
    rolling_vwap_denominator = pd.Series(volume).rolling(24, min_periods=12).sum()
    htf_vwap = rolling_vwap_numerator / rolling_vwap_denominator.replace(0, np.nan)
    frame["distance_from_htf_vwap"] = (pd.Series(close) - htf_vwap) / pd.Series(close)

    rolling_high = pd.Series(high).rolling(24, min_periods=12).max()
    rolling_low = pd.Series(low).rolling(24, min_periods=12).min()
    frame["rolling_daily_range_position"] = (
        (pd.Series(close) - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
    ).clip(0.0, 1.0)

    bullish_points = (
        (pd.Series(close) > pd.Series(ema20)).astype(float)
        + (pd.Series(ema20) > pd.Series(ema50)).astype(float)
        + (frame["hourly_ema20_slope_4"] > 0).astype(float)
        + (frame["hourly_rsi14"] > 52).astype(float)
    )
    bearish_points = (
        (pd.Series(close) < pd.Series(ema20)).astype(float)
        + (pd.Series(ema20) < pd.Series(ema50)).astype(float)
        + (frame["hourly_ema20_slope_4"] < 0).astype(float)
        + (frame["hourly_rsi14"] < 48).astype(float)
    )
    trend_score = bullish_points - bearish_points
    chop_score = 4.0 - trend_score.abs()
    volatility_median = pd.Series(vol20).rolling(120, min_periods=30).median()
    volatility_state = pd.Series(vol20) / volatility_median.replace(0, np.nan)

    frame["htf_trend_score"] = trend_score
    frame["htf_volatility_state"] = volatility_state.clip(0.0, 10.0)

    probs = [
        softmax3(
            bear_score=max(0.0, -float(trend)) + max(0.0, float(vol_state) - 1.5) * 0.15,
            chop_score=max(0.0, float(chop)) * 0.45,
            bull_score=max(0.0, float(trend)) + max(0.0, float(vol_state) - 1.5) * 0.15,
        )
        if np.isfinite(trend) and np.isfinite(vol_state)
        else (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        for trend, chop, vol_state in zip(trend_score, chop_score, frame["htf_volatility_state"])
    ]
    frame["hourly_bear_prob"] = [item[0] for item in probs]
    frame["hourly_chop_prob"] = [item[1] for item in probs]
    frame["hourly_bull_prob"] = [item[2] for item in probs]

    keep = [
        "timestamp",
        "close_timestamp",
        "time",
        "source_rows",
        "hourly_return_1",
        "hourly_return_4",
        "hourly_return_12",
        "hourly_return_24",
        "hourly_ema20_distance",
        "hourly_ema50_distance",
        "hourly_ema20_slope_4",
        "hourly_ema50_slope_4",
        "hourly_rsi14",
        "hourly_atr14_percent",
        "hourly_rolling_volatility_20",
        "htf_trend_score",
        "htf_volatility_state",
        "hourly_bull_prob",
        "hourly_bear_prob",
        "hourly_chop_prob",
        "distance_from_htf_vwap",
        "rolling_daily_range_position",
    ]
    frame = frame[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return frame


def main():
    candles, input_path = load_candles()
    hourly = aggregate_5m_to_hourly(candles)
    context = add_context_features(hourly)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    context.to_csv(OUTPUT_PATH, index=False)

    print("Higher-timeframe hourly/daily context builder")
    print(f"SYMBOL: {SYMBOL}")
    print(f"Input path: {input_path}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Input 5m rows: {len(candles)}")
    print(f"Completed hourly rows: {len(hourly)}")
    print(f"Output context rows: {len(context)}")
    if len(context):
        print(f"First timestamp: {context.iloc[0]['time']}")
        print(f"Last timestamp: {context.iloc[-1]['time']}")
        print("Hourly broad-state probability means:")
        print(f"- hourly_bull_prob: {context['hourly_bull_prob'].mean():.4f}")
        print(f"- hourly_bear_prob: {context['hourly_bear_prob'].mean():.4f}")
        print(f"- hourly_chop_prob: {context['hourly_chop_prob'].mean():.4f}")
    print("No trades were placed. Only completed hourly candles are used.")


if __name__ == "__main__":
    main()
