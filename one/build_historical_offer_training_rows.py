import os
from pathlib import Path

import numpy as np
import pandas as pd

from live_3m_model_utils import atomic_write_csv
from hierarchical_context import attach_hierarchical_context, print_context_diagnostics
from offer_model_utils import replay_offer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
os.environ.setdefault("SYMBOL", SYMBOL)

import train_python_model as old_features


RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))
MAX_SNAPSHOTS = int(os.getenv("MAX_SNAPSHOTS", "50000"))
LOOKBACK = int(os.getenv("LOOKBACK", "60"))
TAKER_ROUND_TRIP_COST = float(os.getenv("TAKER_ROUND_TRIP_COST", os.getenv("ROUND_TRIP_COST", "0.0062")))
MAKER_ROUND_TRIP_COST = float(os.getenv("MAKER_ROUND_TRIP_COST", "0.0010"))
ESTIMATED_SLIPPAGE = float(os.getenv("ESTIMATED_SLIPPAGE", "0.0000"))
SPREAD_COST = float(os.getenv("SPREAD_COST", "0.0000"))
COST_PROFILE = os.getenv("COST_PROFILE", "taker").strip().lower()
ROUND_TRIP_COST = (
    float(os.getenv("ROUND_TRIP_COST"))
    if os.getenv("ROUND_TRIP_COST") is not None
    else (
        (MAKER_ROUND_TRIP_COST if COST_PROFILE == "maker" else TAKER_ROUND_TRIP_COST)
        + ESTIMATED_SLIPPAGE
        + SPREAD_COST
    )
)
ADVERSE_PENALTY = float(os.getenv("ADVERSE_PENALTY", "0.75"))
TIME_PENALTY = float(os.getenv("TIME_PENALTY", "0.00005"))
ALLOCATION_FAVORABLE_WEIGHT = float(os.getenv("ALLOCATION_FAVORABLE_WEIGHT", "0.50"))
ALLOCATION_ADVERSE_WEIGHT = float(os.getenv("ALLOCATION_ADVERSE_WEIGHT", "1.00"))
ALLOCATION_VELOCITY_WEIGHT = float(os.getenv("ALLOCATION_VELOCITY_WEIGHT", "0.25"))
SAMPLING_MODE = os.getenv("SAMPLING_MODE", "random").strip().lower()

DATA_PATH = PROJECT_ROOT / "data" / f"{SYMBOL}_5m_imported.csv"
LEGACY_DATA_PATH = PROJECT_ROOT / "data" / "btc_5m_imported.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "training"
    / f"{SYMBOL}_historical_5m_offer_training_rows.csv"
)

HORIZONS = {
    10: 2,
    15: 3,
    25: 5,
    30: 6,
}


def normalize_timestamp_series(values):
    timestamps = pd.to_numeric(values, errors="coerce")
    return np.where(timestamps < 10_000_000_000, timestamps * 1000, timestamps)


def load_candles():
    path = DATA_PATH
    if not path.exists() and SYMBOL == "BTCUSDT" and LEGACY_DATA_PATH.exists():
        path = LEGACY_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing historical 5m data: {DATA_PATH}. "
            "Run the Binance downloader/importer first."
        )

    candles = pd.read_csv(path)
    required = ["time", "timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in candles.columns]
    if missing:
        raise ValueError(f"Historical 5m CSV is missing required columns: {missing}")

    for column in [
        "quoteVolume",
        "numberOfTrades",
        "takerBuyBaseVolume",
        "takerBuyQuoteVolume",
    ]:
        if column not in candles.columns:
            candles[column] = np.nan

    candles["timestamp"] = normalize_timestamp_series(candles["timestamp"])
    candles = candles.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    candles["timestamp"] = candles["timestamp"].astype(np.int64)
    for column in ["open", "high", "low", "close", "volume", "numberOfTrades", "takerBuyBaseVolume"]:
        candles[column] = pd.to_numeric(candles[column], errors="coerce")
    candles = candles.sort_values("timestamp").drop_duplicates("timestamp")
    return candles.reset_index(drop=True), path


def load_optional_old_predictions():
    path = PROJECT_ROOT / "data" / f"{SYMBOL}_model_predictions.csv"
    if not path.exists():
        return None, None
    frame = pd.read_csv(path)
    required = ["timestamp", "prob_down", "prob_neutral", "prob_up"]
    if any(column not in frame.columns for column in required):
        return None, path
    frame["timestamp"] = normalize_timestamp_series(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).copy()
    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    return frame[required], path


def old_context_for_timestamp(old_predictions, timestamp):
    if old_predictions is None or len(old_predictions) == 0:
        return {
            "old_prob_down": 1.0 / 3.0,
            "old_prob_neutral": 1.0 / 3.0,
            "old_prob_up": 1.0 / 3.0,
            "old_directional_confidence": 0.0,
            "old_context_available": 0,
        }
    timestamps = old_predictions["timestamp"].to_numpy(dtype=np.int64)
    index = np.searchsorted(timestamps, timestamp, side="right") - 1
    if index < 0:
        return {
            "old_prob_down": 1.0 / 3.0,
            "old_prob_neutral": 1.0 / 3.0,
            "old_prob_up": 1.0 / 3.0,
            "old_directional_confidence": 0.0,
            "old_context_available": 0,
        }
    row = old_predictions.iloc[index]
    return {
        "old_prob_down": float(row["prob_down"]),
        "old_prob_neutral": float(row["prob_neutral"]),
        "old_prob_up": float(row["prob_up"]),
        "old_directional_confidence": float(max(row["prob_down"], row["prob_up"]) - row["prob_neutral"]),
        "old_context_available": 1,
    }


def select_snapshot_indexes(candles, indicators):
    max_horizon_candles = max(HORIZONS.values())
    first_valid = max(LOOKBACK - 1, old_features.FIRST_VALID_FEATURE_INDEX)
    last_valid = len(candles) - max_horizon_candles - 1
    valid = []
    for index in range(first_valid, last_valid + 1):
        window_start = index - LOOKBACK + 1
        if window_start < 0:
            continue
        if not old_features.has_enough_feature_history(index, indicators):
            continue
        valid.append(index)

    valid = np.asarray(valid, dtype=np.int64)
    if MAX_SNAPSHOTS > 0 and len(valid) > MAX_SNAPSHOTS:
        if SAMPLING_MODE in {"all", "chronological"}:
            sampled = valid[:MAX_SNAPSHOTS]
        else:
            rng = np.random.default_rng(RANDOM_SEED)
            sampled = np.sort(rng.choice(valid, size=MAX_SNAPSHOTS, replace=False))
    else:
        sampled = valid
    return valid, sampled


def future_window(candles, index, horizon_candles):
    start = index + 1
    end = index + horizon_candles
    if end >= len(candles):
        return None
    return candles.iloc[start : end + 1].copy()


def build_rows(candles, sampled_indexes, arrays, indicators, old_predictions):
    rows = []
    for index in sampled_indexes:
        feature_values = old_features.features_for_candle(
            candles,
            arrays,
            indicators,
            int(index),
        )
        if feature_values is None:
            continue

        base = {
            "timestamp": int(candles.loc[index, "timestamp"]),
            "time": candles.loc[index, "time"],
            "symbol": SYMBOL,
            "entry_price": float(candles.loc[index, "close"]),
            **{
                name: float(value)
                for name, value in zip(old_features.FEATURE_NAMES, feature_values)
            },
            **old_context_for_timestamp(
                old_predictions,
                int(candles.loc[index, "timestamp"]),
            ),
        }

        for horizon_minutes, horizon_candles in HORIZONS.items():
            future = future_window(candles, int(index), horizon_candles)
            if future is None:
                continue
            for side in ["LONG", "SHORT"]:
                outcome = replay_offer(
                    entry_price=base["entry_price"],
                    side=side,
                    take_profit=999.0,
                    stop_loss=999.0,
                    future_rows=future,
                    round_trip_cost=ROUND_TRIP_COST,
                    adverse_penalty=ADVERSE_PENALTY,
                    time_penalty=TIME_PENALTY,
                    allocation_favorable_weight=ALLOCATION_FAVORABLE_WEIGHT,
                    allocation_adverse_weight=ALLOCATION_ADVERSE_WEIGHT,
                    allocation_velocity_weight=ALLOCATION_VELOCITY_WEIGHT,
                )
                row = {
                    **base,
                    "offer_side": side,
                    "offer_horizon_minutes": horizon_minutes,
                    "offer_horizon_candles": horizon_candles,
                    "offer_take_profit": 0.0,
                    "offer_stop_loss": 0.0,
                    **outcome,
                }
                rows.append(row)

    return pd.DataFrame(rows)


def main():
    old_features.LOOKBACK = LOOKBACK
    candles, data_path = load_candles()
    arrays = old_features.create_arrays(candles)
    indicators = old_features.build_indicators(candles)
    old_predictions, old_path = load_optional_old_predictions()
    valid_indexes, sampled_indexes = select_snapshot_indexes(candles, indicators)
    output = build_rows(candles, sampled_indexes, arrays, indicators, old_predictions)
    if len(output) > 0:
        output, context_diagnostics = attach_hierarchical_context(
            output,
            PROJECT_ROOT,
            SYMBOL,
            layers=("htf", "regime15", "regime30"),
        )
    else:
        context_diagnostics = {}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output, OUTPUT_PATH)

    print("Historical 5m paper-only offer training row builder")
    print(f"SYMBOL: {SYMBOL}")
    print(f"Input path: {data_path}")
    print(f"Optional old prediction path: {old_path if old_path else 'not available'}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")
    print(f"MAX_SNAPSHOTS: {MAX_SNAPSHOTS}")
    print(f"SAMPLING_MODE: {SAMPLING_MODE}")
    print(f"LOOKBACK: {LOOKBACK}")
    print(f"COST_PROFILE: {COST_PROFILE}")
    print(f"TAKER_ROUND_TRIP_COST: {TAKER_ROUND_TRIP_COST:.4%}")
    print(f"MAKER_ROUND_TRIP_COST: {MAKER_ROUND_TRIP_COST:.4%}")
    print(f"ESTIMATED_SLIPPAGE: {ESTIMATED_SLIPPAGE:.4%}")
    print(f"SPREAD_COST: {SPREAD_COST:.4%}")
    print(f"ROUND_TRIP_COST: {ROUND_TRIP_COST:.4%}")
    print(f"ADVERSE_PENALTY: {ADVERSE_PENALTY}")
    print(f"TIME_PENALTY: {TIME_PENALTY}")
    print(f"ALLOCATION_FAVORABLE_WEIGHT: {ALLOCATION_FAVORABLE_WEIGHT}")
    print(f"ALLOCATION_ADVERSE_WEIGHT: {ALLOCATION_ADVERSE_WEIGHT}")
    print(f"ALLOCATION_VELOCITY_WEIGHT: {ALLOCATION_VELOCITY_WEIGHT}")
    print(f"Total candles: {len(candles)}")
    print(f"Valid snapshots: {len(valid_indexes)}")
    print(f"Sampled snapshots: {len(sampled_indexes)}")
    print(f"Generated offer rows: {len(output)}")
    if len(output) > 0:
        print_context_diagnostics(context_diagnostics)
        print(
            "Date range used: "
            f"{output.iloc[0]['time']} -> {output.iloc[-1]['time']}"
        )
        print("accept_target distribution:")
        counts = output["accept_target"].value_counts().sort_index()
        for value in [0, 1]:
            count = int(counts.get(value, 0))
            print(f"- {value}: {count} ({count / len(output):.2%})")

        print("\nAverage opportunity_score by side/horizon:")
        grouped = output.groupby(["offer_side", "offer_horizon_minutes"])["opportunity_score"].mean()
        for (side, horizon), value in grouped.items():
            print(f"- {side} {int(horizon)}m: {value:.6f}")

        print("\nAverage final_return by side/horizon:")
        grouped_return = output.groupby(["offer_side", "offer_horizon_minutes"])["final_return"].mean()
        for (side, horizon), value in grouped_return.items():
            print(f"- {side} {int(horizon)}m: {value:.4%}")

        print("\nTarget allocation bucket distribution:")
        bucket_counts = output["target_allocation_bucket"].value_counts().sort_index()
        for bucket, count in bucket_counts.items():
            print(f"- bucket {int(bucket)}: {int(count)} ({count / len(output):.2%})")

    print("No trades were placed. No future candles were used as inputs.")


if __name__ == "__main__":
    main()
