import os
from pathlib import Path

import numpy as np
import pandas as pd

from live_3m_model_utils import atomic_write_csv, parse_bool
from offer_model_utils import (
    OFFER_SPECS,
    attach_old_context,
    choose_snapshot_feature_columns,
    load_old_context,
    prepare_feature_rows,
    read_csv_sorted,
    replay_offer,
)
from hierarchical_context import (
    attach_hierarchical_context,
    print_context_availability_summary,
    print_context_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
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
USE_OLD_MODEL_CONTEXT = parse_bool(os.getenv("USE_OLD_MODEL_CONTEXT", "true"))
MAX_OLD_PREDICTION_AGE_MS = int(os.getenv("MAX_OLD_PREDICTION_AGE_MS", "300000"))
ADVERSE_PENALTY = float(os.getenv("ADVERSE_PENALTY", "0.75"))
TIME_PENALTY = float(os.getenv("TIME_PENALTY", "0.00005"))
ALLOCATION_FAVORABLE_WEIGHT = float(os.getenv("ALLOCATION_FAVORABLE_WEIGHT", "0.50"))
ALLOCATION_ADVERSE_WEIGHT = float(os.getenv("ALLOCATION_ADVERSE_WEIGHT", "1.00"))
ALLOCATION_VELOCITY_WEIGHT = float(os.getenv("ALLOCATION_VELOCITY_WEIGHT", "0.25"))
EXCLUDE_AMBIGUOUS_OFFERS = parse_bool(os.getenv("EXCLUDE_AMBIGUOUS_OFFERS", "false"))

REALTIME_DIR = PROJECT_ROOT / "data" / "realtime"
VENUE_REALTIME_DIR = REALTIME_DIR / PRIMARY_VENUE if PRIMARY_VENUE else REALTIME_DIR
RAW_1M_PATH = VENUE_REALTIME_DIR / f"{SYMBOL}_1m_flow.csv"
FEATURES_PATH = VENUE_REALTIME_DIR / f"{SYMBOL}_1m_flow_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "live_training" / f"{SYMBOL}_offer_training_rows.csv"


def load_inputs():
    raw = read_csv_sorted(
        RAW_1M_PATH,
        ["timestamp", "time", "high", "low", "close"],
        "raw 1m flow CSV",
    )
    features = read_csv_sorted(
        FEATURES_PATH,
        ["timestamp", "time", "feature_ready", "close"],
        "1m flow feature CSV",
    )
    features = prepare_feature_rows(features)
    return raw, features


def complete_future_window(timestamp, horizon_minutes, raw_by_timestamp):
    future_timestamps = [
        int(timestamp + step * 60_000)
        for step in range(1, horizon_minutes + 1)
    ]
    if any(value not in raw_by_timestamp.index for value in future_timestamps):
        return None
    return raw_by_timestamp.loc[future_timestamps].copy()


def entry_price_for_row(row):
    if "mid_price" in row.index and pd.notna(row["mid_price"]) and float(row["mid_price"]) > 0:
        return float(row["mid_price"])
    return float(row["close"])


def main():
    raw, features = load_inputs()
    raw_by_timestamp = raw.set_index("timestamp", drop=False)
    snapshot_feature_columns = choose_snapshot_feature_columns(features)
    old_context, old_path = load_old_context(
        PROJECT_ROOT,
        SYMBOL,
        USE_OLD_MODEL_CONTEXT,
        MAX_OLD_PREDICTION_AGE_MS,
    )

    output_rows = []
    skipped_feature_not_ready = 0
    skipped_incomplete_future_window = 0
    ambiguous_hit_count = 0
    eligible_feature_rows = 0

    for _, feature_row in features.iterrows():
        timestamp = int(feature_row["timestamp"])
        if not bool(feature_row["feature_ready"]):
            skipped_feature_not_ready += 1
            continue

        eligible_feature_rows += 1
        entry_price = entry_price_for_row(feature_row)

        for side, horizon, take_profit, stop_loss in OFFER_SPECS:
            future = complete_future_window(timestamp, horizon, raw_by_timestamp)
            if future is None:
                skipped_incomplete_future_window += 1
                continue

            outcome = replay_offer(
                entry_price=entry_price,
                side=side,
                take_profit=take_profit,
                stop_loss=stop_loss,
                future_rows=future,
                round_trip_cost=ROUND_TRIP_COST,
                adverse_penalty=ADVERSE_PENALTY,
                time_penalty=TIME_PENALTY,
                allocation_favorable_weight=ALLOCATION_FAVORABLE_WEIGHT,
                allocation_adverse_weight=ALLOCATION_ADVERSE_WEIGHT,
                allocation_velocity_weight=ALLOCATION_VELOCITY_WEIGHT,
            )
            if outcome["hit_result"] == "ambiguous":
                ambiguous_hit_count += 1
                if EXCLUDE_AMBIGUOUS_OFFERS:
                    continue

            row = {
                "timestamp": timestamp,
                "time": feature_row["time"],
                "symbol": SYMBOL,
                "offer_side": side,
                "offer_horizon_minutes": horizon,
                "offer_take_profit": take_profit,
                "offer_stop_loss": stop_loss,
                "entry_price": entry_price,
                **outcome,
            }
            for column in snapshot_feature_columns:
                row[column] = feature_row[column]
            output_rows.append(row)

    output = pd.DataFrame(output_rows)
    if len(output) > 0:
        output = attach_old_context(
            output,
            old_context,
            MAX_OLD_PREDICTION_AGE_MS,
        )
        output, context_diagnostics = attach_hierarchical_context(
            output,
            PROJECT_ROOT,
            SYMBOL,
            layers=("htf", "regime15", "regime30", "flow3m", "flow1s", "micro10s"),
        )
        output = output.sort_values(["timestamp", "offer_horizon_minutes", "offer_side"])
    else:
        context_diagnostics = {}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output, OUTPUT_PATH)

    print("Paper-only live offer training row builder")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Raw 1m path: {RAW_1M_PATH}")
    print(f"Feature path: {FEATURES_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Old model context enabled: {USE_OLD_MODEL_CONTEXT}")
    print(f"Old model context path: {old_path if old_path else 'not available'}")
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
    print("accept_target = 1 only when opportunity_score > 0")
    print(f"Total feature rows: {len(features)}")
    print(f"Eligible feature rows: {eligible_feature_rows}")
    print(f"Rows skipped due feature_ready=false: {skipped_feature_not_ready}")
    print(f"Rows skipped due incomplete future window: {skipped_incomplete_future_window}")
    print(f"Generated offer rows: {len(output)}")
    print(f"Ambiguous hit count: {ambiguous_hit_count}")

    if len(output) > 0:
        print_context_diagnostics(context_diagnostics)
        print_context_availability_summary(output)
        print("accept_target distribution:")
        counts = output["accept_target"].value_counts().sort_index()
        for value in [0, 1]:
            count = int(counts.get(value, 0))
            print(f"- {value}: {count} ({count / len(output):.2%})")

        print("\nAverage opportunity_score by offer_side/horizon/TP/SL:")
        grouped = (
            output.groupby(
                [
                    "offer_side",
                    "offer_horizon_minutes",
                    "offer_take_profit",
                    "offer_stop_loss",
                ]
            )["opportunity_score"]
            .mean()
            .reset_index()
        )
        for _, row in grouped.iterrows():
            print(
                f"- {row['offer_side']} {int(row['offer_horizon_minutes'])}m "
                f"TP={row['offer_take_profit']:.2%} SL={row['offer_stop_loss']:.2%}: "
                f"opportunity_score={row['opportunity_score']:.6f}"
            )

    print("No trades were placed. No exchange order APIs were used.")


if __name__ == "__main__":
    main()
