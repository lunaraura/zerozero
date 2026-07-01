import os
from pathlib import Path

import numpy as np
import pandas as pd

from build_htf_context_features import add_context_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE_FALLBACK = os.getenv("VENUE", os.getenv("VENUES", "")).split(",")[0].strip().lower()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", PRIMARY_VENUE_FALLBACK).strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
INPUT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_1m_flow.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_htf_context_features.csv"


def load_realtime_1m_rows():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing realtime 1m flow CSV: {INPUT_PATH}")

    frame = pd.read_csv(INPUT_PATH)
    required = ["timestamp", "time", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Realtime 1m CSV is missing required columns: {missing}")

    for column in required:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["timestamp"] = np.where(
        frame["timestamp"] < 10_000_000_000,
        frame["timestamp"] * 1000,
        frame["timestamp"],
    ).astype("float64")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame.reset_index(drop=True)


def aggregate_1m_to_hourly(rows):
    interval_ms = 60 * 60 * 1000
    working = rows.copy()
    working["timestamp_hour"] = (working["timestamp"] // interval_ms) * interval_ms
    grouped = working.groupby("timestamp_hour", sort=True)
    hourly = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        source_rows=("timestamp", "count"),
    ).reset_index()

    # Use only fully completed hourly candles: exactly sixty completed 1m rows.
    hourly = hourly[hourly["source_rows"] == 60].copy()
    hourly = hourly.rename(columns={"timestamp_hour": "timestamp"})
    hourly["close_timestamp"] = hourly["timestamp"] + interval_ms
    hourly["time"] = pd.to_datetime(hourly["timestamp"], unit="ms", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )
    return hourly.reset_index(drop=True)


def main():
    rows = load_realtime_1m_rows()
    hourly = aggregate_1m_to_hourly(rows)
    context = add_context_features(hourly) if len(hourly) else pd.DataFrame()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    context.to_csv(OUTPUT_PATH, index=False)

    print("Live hourly HTF context updater")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Input realtime 1m path: {INPUT_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Realtime 1m rows: {len(rows)}")
    print(f"Completed hourly rows: {len(hourly)}")
    print(f"Output context rows: {len(context)}")
    if len(rows):
        print(f"Latest raw 1m timestamp: {int(rows['timestamp'].max())}")
    if len(context):
        print(f"First context time: {context.iloc[0]['time']}")
        print(f"Last context time: {context.iloc[-1]['time']}")
        print("Hourly broad-state probability means:")
        print(f"- hourly_bull_prob: {context['hourly_bull_prob'].mean():.4f}")
        print(f"- hourly_bear_prob: {context['hourly_bear_prob'].mean():.4f}")
        print(f"- hourly_chop_prob: {context['hourly_chop_prob'].mean():.4f}")
    print("No trades were placed. Only completed hourly candles are used.")


if __name__ == "__main__":
    main()
