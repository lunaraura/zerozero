import os
from pathlib import Path

import numpy as np
import pandas as pd

from live_3m_model_utils import parse_bool
from offer_model_utils import (
    MODEL_TARGET_COLUMNS,
    OFFER_SPECS,
    allocation_fraction_to_bucket,
    allocation_score_to_fraction,
    attach_old_context,
    load_model,
    load_old_context,
    offer_prices,
    predict_artifact_full,
    prepare_feature_rows,
    read_csv_sorted,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
USE_OLD_MODEL_CONTEXT = parse_bool(os.getenv("USE_OLD_MODEL_CONTEXT", "true"))
MAX_OLD_PREDICTION_AGE_MS = int(os.getenv("MAX_OLD_PREDICTION_AGE_MS", "300000"))
ACCEPT_THRESHOLD = float(os.getenv("ACCEPT_THRESHOLD", "0.60"))
ROUND_TRIP_COST = float(os.getenv("ROUND_TRIP_COST", "0.0062"))

REALTIME_DIR = PROJECT_ROOT / "data" / "realtime"
VENUE_REALTIME_DIR = REALTIME_DIR / PRIMARY_VENUE if PRIMARY_VENUE else REALTIME_DIR
FEATURES_PATH = VENUE_REALTIME_DIR / f"{SYMBOL}_1m_flow_features.csv"
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "active" / f"{SYMBOL}_offer_acceptance_model.json"


def latest_feature_row():
    features = read_csv_sorted(
        FEATURES_PATH,
        ["timestamp", "time", "feature_ready", "close"],
        "1m flow feature CSV",
    )
    features = prepare_feature_rows(features)
    ready = features[features["feature_ready"]].copy()
    if len(ready) == 0:
        raise RuntimeError("No feature_ready rows are available for paper offer scoring.")
    return ready.iloc[-1], features


def build_current_offers(row):
    entry_price = float(row["mid_price"]) if "mid_price" in row.index and pd.notna(row["mid_price"]) and float(row["mid_price"]) > 0 else float(row["close"])
    offers = []
    for side, horizon, take_profit, stop_loss in OFFER_SPECS:
        tp_price, sl_price = offer_prices(entry_price, side, take_profit, stop_loss)
        offer = {
            "timestamp": int(row["timestamp"]),
            "time": row["time"],
            "symbol": SYMBOL,
            "offer_side": side,
            "offer_horizon_minutes": horizon,
            "offer_take_profit": take_profit,
            "offer_stop_loss": stop_loss,
            "entry_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }
        for column, value in row.items():
            if column not in offer:
                offer[column] = value
        offers.append(offer)
    return pd.DataFrame(offers)


def bootstrap_scores(offers):
    pressure = pd.to_numeric(offers.get("market_pressure", 0.0), errors="coerce").fillna(0.0)
    breakout = pd.to_numeric(offers.get("breakout_pressure_index", 0.0), errors="coerce").fillna(0.0)
    imbalance = pd.to_numeric(offers.get("order_book_imbalance_10bps", 0.0), errors="coerce").fillna(0.0)
    signal = ((pressure + breakout + imbalance) / 3.0).clip(-1.0, 1.0)
    side_multiplier = np.where(offers["offer_side"] == "LONG", 1.0, -1.0)
    aligned_signal = signal.to_numpy() * side_multiplier
    accept_probability = np.clip(0.35 + aligned_signal * 0.25, 0.05, 0.75)
    expected_final_return = (
        aligned_signal * offers["offer_take_profit"].to_numpy()
        - (1.0 - np.maximum(aligned_signal, 0.0)) * offers["offer_stop_loss"].to_numpy() * 0.5
    )
    expected_favorable = np.maximum(expected_final_return, 0.0) + offers["offer_take_profit"].to_numpy() * 0.25
    expected_adverse = -offers["offer_stop_loss"].to_numpy() * (1.0 - np.maximum(aligned_signal, 0.0))
    expected_favorable_velocity = expected_favorable / offers["offer_horizon_minutes"].to_numpy()
    expected_opportunity_score = (
        expected_favorable
        - 0.75 * np.abs(expected_adverse)
        - ROUND_TRIP_COST
        - 0.00005 * offers["offer_horizon_minutes"].to_numpy()
    )
    expected_net_return = expected_final_return - ROUND_TRIP_COST
    predicted_allocation_fraction = np.asarray(
        [allocation_score_to_fraction(value) for value in expected_opportunity_score],
        dtype=np.float64,
    )
    predicted_bucket = np.asarray(
        [allocation_fraction_to_bucket(value) for value in predicted_allocation_fraction],
        dtype=np.int64,
    )
    bucket_probabilities = np.zeros((len(offers), 6), dtype=np.float64)
    bucket_probabilities[np.arange(len(offers)), predicted_bucket] = 1.0
    return accept_probability, bucket_probabilities, np.column_stack(
        [
            predicted_allocation_fraction,
            expected_opportunity_score,
            expected_adverse,
            expected_net_return,
        ]
    )


def main():
    latest_row, _ = latest_feature_row()
    offers = build_current_offers(latest_row)
    old_context, old_path = load_old_context(
        PROJECT_ROOT,
        SYMBOL,
        USE_OLD_MODEL_CONTEXT,
        MAX_OLD_PREDICTION_AGE_MS,
    )
    offers = attach_old_context(offers, old_context, MAX_OLD_PREDICTION_AGE_MS)
    model = load_model(ACTIVE_MODEL_PATH)
    model_has_allocation_head = (
        model is not None
        and "w_bucket" in model.get("model", {})
        and model.get("target_columns") == MODEL_TARGET_COLUMNS
    )

    if model_has_allocation_head:
        model_source = "active_offer_model"
        accept_probability, bucket_probabilities, regression = predict_artifact_full(model, offers)
        target_columns = model.get("target_columns", MODEL_TARGET_COLUMNS)
    else:
        model_source = "bootstrap" if model is None else "bootstrap_old_model_schema"
        accept_probability, bucket_probabilities, regression = bootstrap_scores(offers)
        target_columns = MODEL_TARGET_COLUMNS

    scored = offers.copy()
    scored["paper_accept_probability"] = accept_probability
    scored["predicted_allocation_bucket"] = np.argmax(bucket_probabilities, axis=1)
    for index, column in enumerate(target_columns):
        scored[f"predicted_{column}"] = regression[:, index]
    for column in MODEL_TARGET_COLUMNS:
        if f"predicted_{column}" not in scored.columns:
            scored[f"predicted_{column}"] = np.nan
    scored["predicted_allocation_fraction"] = scored[
        "predicted_target_allocation_fraction"
    ].clip(lower=0.0, upper=0.20)
    scored["predicted_opportunity_score"] = scored["predicted_opportunity_score"]
    scored["predicted_net_return"] = scored["predicted_net_return"]
    scored["predicted_max_adverse_excursion"] = scored["predicted_max_adverse_excursion"]
    scored = scored.sort_values(
        ["predicted_allocation_fraction", "paper_accept_probability", "predicted_opportunity_score"],
        ascending=False,
    ).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    best = scored.iloc[0]
    best_action = (
        "paper_best_offer"
        if best["predicted_allocation_fraction"] > 0
        and best["paper_accept_probability"] >= ACCEPT_THRESHOLD
        and best["predicted_opportunity_score"] > 0
        else "NO_ACTION"
    )

    print("Latest paper-only offer scores")
    print(f"symbol: {SYMBOL}")
    print(f"primary_venue: {PRIMARY_VENUE or 'legacy'}")
    print(f"feature_path: {FEATURES_PATH}")
    print(f"time: {latest_row['time']}")
    print(f"model_source: {model_source}")
    print(f"active_model_path: {ACTIVE_MODEL_PATH if model is not None else 'none'}")
    print(f"old_5m_context_path: {old_path if old_path else 'not available'}")
    print(
        "old_5m_context: "
        f"available={bool(best.get('old_context_available', 0))}, "
        f"down={best.get('old_prob_down', 1/3):.3f}, "
        f"neutral={best.get('old_prob_neutral', 1/3):.3f}, "
        f"up={best.get('old_prob_up', 1/3):.3f}, "
        f"age_ms={best.get('old_prediction_age_ms', np.nan)}"
    )
    print("No order placed. This is a paper-only scenario report.")
    print("\nOffers:")
    for _, row in scored.iterrows():
        print(
            f"rank={int(row['rank'])} | side={row['offer_side']} | "
            f"horizon={int(row['offer_horizon_minutes'])}m | "
            f"TP={row['offer_take_profit']:.2%} | SL={row['offer_stop_loss']:.2%} | "
            f"paper_accept_probability={row['paper_accept_probability']:.2%} | "
            f"predicted_allocation_fraction={row['predicted_allocation_fraction']:.2%} | "
            f"predicted_allocation_bucket={int(row['predicted_allocation_bucket'])} | "
            f"predicted_opportunity_score={row['predicted_opportunity_score']:.6f} | "
            f"predicted_net_return={row['predicted_net_return']:.4%} | "
            f"predicted_max_adverse_excursion={row['predicted_max_adverse_excursion']:.4%}"
        )

    print("\nSummary:")
    print(f"best_action={best_action}")
    if best_action == "NO_ACTION":
        print(
            "paper_best_offer=none; no offer exceeds the paper acceptance threshold."
        )
    else:
        print(
            f"paper_best_offer={best['offer_side']} "
            f"{int(best['offer_horizon_minutes'])}m "
            f"TP={best['offer_take_profit']:.2%} SL={best['offer_stop_loss']:.2%}"
        )
        print(f"paper_allocation={best['predicted_allocation_fraction']:.2%}")
    print("no order placed")


if __name__ == "__main__":
    main()
