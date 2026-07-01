import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "ensemble" / f"{SYMBOL}_ensemble_predictions.csv"

LONG_THRESHOLD = float(os.getenv("LONG_THRESHOLD", "0.60"))
SHORT_THRESHOLD = float(os.getenv("SHORT_THRESHOLD", "0.60"))
MAX_NEUTRAL_THRESHOLD = float(os.getenv("MAX_NEUTRAL_THRESHOLD", "0.30"))
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

CLASS_NAMES = {
    0: "short_win",
    1: "neutral",
    2: "long_win",
}


def percent(value):
    return f"{value * 100:.2f}%"


def load_predictions():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing ensemble predictions: {PREDICTIONS_PATH}. "
            "Run scripts/train_ensemble_meta_model.py first."
        )

    frame = pd.read_csv(PREDICTIONS_PATH)
    required = [
        "time",
        "timestamp",
        "actual_class",
        "prob_short",
        "prob_neutral",
        "prob_long",
        "future_return_3",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")

    for column in frame.columns:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)


def signal_for_row(row):
    if row["prob_long"] >= LONG_THRESHOLD and row["prob_neutral"] <= MAX_NEUTRAL_THRESHOLD:
        return "LONG"
    if (
        row["prob_short"] >= SHORT_THRESHOLD
        and row["prob_neutral"] <= MAX_NEUTRAL_THRESHOLD
    ):
        return "SHORT"
    return "NONE"


def create_trades(frame):
    trades = []
    skipped = 0
    for _, row in frame.iterrows():
        signal = signal_for_row(row)
        if signal == "NONE":
            skipped += 1
            continue

        gross_return = row["future_return_3"] if signal == "LONG" else -row["future_return_3"]
        net_return = gross_return - ROUND_TRIP_COST
        trades.append(
            {
                "time": row["time"],
                "timestamp": row["timestamp"],
                "signal": signal,
                "actual_class": int(row["actual_class"]),
                "prob_short": row["prob_short"],
                "prob_neutral": row["prob_neutral"],
                "prob_long": row["prob_long"],
                "gross_return": gross_return,
                "net_return": net_return,
            }
        )

    return pd.DataFrame(trades), skipped


def compounded_return(returns):
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = (equity / peak) - 1.0
        max_drawdown = min(max_drawdown, drawdown)
    return equity - 1.0, max_drawdown


def print_class_distribution(classes):
    counts = pd.Series(classes).value_counts().sort_index()
    total = len(classes) or 1
    for class_id in [0, 1, 2]:
        count = int(counts.get(class_id, 0))
        print(f"- class {class_id} {CLASS_NAMES[class_id]}: {count} ({count / total:.2%})")


def print_threshold_table(frame, side):
    probability_column = "prob_long" if side == "LONG" else "prob_short"
    print(f"\nThreshold table for {side}")
    print("threshold | rows | avg gross | win before costs | win after costs")
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        subset = frame[
            (frame[probability_column] >= threshold)
            & (frame["prob_neutral"] <= MAX_NEUTRAL_THRESHOLD)
        ]
        gross = subset["future_return_3"] if side == "LONG" else -subset["future_return_3"]
        net = gross - ROUND_TRIP_COST
        count = len(subset)
        print(
            f"{threshold:.2f} | {count} | "
            f"{percent(gross.mean() if count else 0)} | "
            f"{percent((gross > 0).mean() if count else 0)} | "
            f"{percent((net > 0).mean() if count else 0)}"
        )


def main():
    frame = load_predictions()
    trades, skipped = create_trades(frame)

    print("Stacked ensemble prediction backtest")
    print(f"SYMBOL: {SYMBOL}")
    print(f"Prediction path: {PREDICTIONS_PATH}")
    print(f"LONG_THRESHOLD: {LONG_THRESHOLD}")
    print(f"SHORT_THRESHOLD: {SHORT_THRESHOLD}")
    print(f"MAX_NEUTRAL_THRESHOLD: {MAX_NEUTRAL_THRESHOLD}")
    print(f"COST_PROFILE: {COST_PROFILE}")
    print(f"TAKER_ROUND_TRIP_COST: {TAKER_ROUND_TRIP_COST:.4%}")
    print(f"MAKER_ROUND_TRIP_COST: {MAKER_ROUND_TRIP_COST:.4%}")
    print(f"ESTIMATED_SLIPPAGE: {ESTIMATED_SLIPPAGE:.4%}")
    print(f"SPREAD_COST: {SPREAD_COST:.4%}")
    print(f"ROUND_TRIP_COST: {ROUND_TRIP_COST:.4%}")
    print("No trades are placed; this is an offline paper backtest.")

    print(f"\nTotal prediction rows: {len(frame)}")
    print(f"Total trades: {len(trades)}")
    print(f"Long trades: {int((trades['signal'] == 'LONG').sum()) if len(trades) else 0}")
    print(f"Short trades: {int((trades['signal'] == 'SHORT').sum()) if len(trades) else 0}")
    print(f"Skipped rows: {skipped}")

    if len(trades) == 0:
        print("\nNo trades passed the thresholds.")
        print_threshold_table(frame, "LONG")
        print_threshold_table(frame, "SHORT")
        return

    before_cost_wins = trades["gross_return"] > 0
    after_cost_wins = trades["net_return"] > 0
    total_return, max_drawdown = compounded_return(trades["net_return"])
    winners = trades[after_cost_wins]
    losers = trades[~after_cost_wins]

    print(f"Win rate before costs: {percent(before_cost_wins.mean())}")
    print(f"Win rate after costs: {percent(after_cost_wins.mean())}")
    print(f"Average gross return: {percent(trades['gross_return'].mean())}")
    print(f"Average net return: {percent(trades['net_return'].mean())}")
    print(f"Compounded return: {percent(total_return)}")
    print(f"Max drawdown: {percent(max_drawdown)}")
    print("\nActual class distribution among executed trades:")
    print_class_distribution(trades["actual_class"])
    print(
        "Average prob_long on winners / losers: "
        f"{winners['prob_long'].mean() if len(winners) else 0:.4f} / "
        f"{losers['prob_long'].mean() if len(losers) else 0:.4f}"
    )
    print(
        "Average prob_short on winners / losers: "
        f"{winners['prob_short'].mean() if len(winners) else 0:.4f} / "
        f"{losers['prob_short'].mean() if len(losers) else 0:.4f}"
    )

    print_threshold_table(frame, "LONG")
    print_threshold_table(frame, "SHORT")


if __name__ == "__main__":
    main()
