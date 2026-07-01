import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

PREDICTIONS_PATH = OUTPUT_DIR / f"{SYMBOL}_1m_flow_predictions.csv"

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
RETURN_HORIZON = int(os.getenv("RETURN_HORIZON", "3"))

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
            f"Missing prediction file: {PREDICTIONS_PATH}. "
            "Run scripts/train_1m_flow_model.py first."
        )

    frame = pd.read_csv(PREDICTIONS_PATH)
    required = [
        "time",
        "timestamp",
        "actual_class",
        "prob_short",
        "prob_neutral",
        "prob_long",
        f"actual_return_{RETURN_HORIZON}",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")

    for column in frame.columns:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=required).reset_index(drop=True)
    return frame


def signal_for_row(row):
    if (
        row["prob_long"] >= LONG_THRESHOLD
        and row["prob_neutral"] <= MAX_NEUTRAL_THRESHOLD
    ):
        return "LONG"

    if (
        row["prob_short"] >= SHORT_THRESHOLD
        and row["prob_neutral"] <= MAX_NEUTRAL_THRESHOLD
    ):
        return "SHORT"

    return "NONE"


def create_trades(frame):
    trades = []
    skipped_rows = 0
    return_column = f"actual_return_{RETURN_HORIZON}"

    for _, row in frame.iterrows():
        signal = signal_for_row(row)
        if signal == "NONE":
            skipped_rows += 1
            continue

        gross_return = (
            row[return_column] if signal == "LONG" else -row[return_column]
        )
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

    return pd.DataFrame(trades), skipped_rows


def compounded_return(returns):
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0


def class_distribution(classes):
    counts = {class_id: int((classes == class_id).sum()) for class_id in [0, 1, 2]}
    total = len(classes) or 1
    return {
        class_id: (counts[class_id], counts[class_id] / total)
        for class_id in counts
    }


def print_class_distribution(title, classes):
    print(title)
    distribution = class_distribution(classes)
    for class_id in [0, 1, 2]:
        count, ratio = distribution[class_id]
        print(f"- class {class_id} {CLASS_NAMES[class_id]}: {count} ({percent(ratio)})")


def correlation(actual, predicted):
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")
    valid = actual.notna() & predicted.notna()

    if valid.sum() < 2:
        return np.nan

    return float(np.corrcoef(actual[valid], predicted[valid])[0, 1])


def mae(actual, predicted):
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")
    valid = actual.notna() & predicted.notna()

    if valid.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(actual[valid] - predicted[valid])))


def print_threshold_table(frame, side):
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    probability_column = "prob_long" if side == "LONG" else "prob_short"
    return_column = f"actual_return_{RETURN_HORIZON}"

    print(f"\nThreshold table for {probability_column}")
    print("threshold | rows | avg gross return | win rate before costs | win rate after costs")

    for threshold in thresholds:
        subset = frame[
            (frame[probability_column] >= threshold)
            & (frame["prob_neutral"] <= MAX_NEUTRAL_THRESHOLD)
        ].copy()

        if side == "LONG":
            gross_returns = subset[return_column]
        else:
            gross_returns = -subset[return_column]

        net_returns = gross_returns - ROUND_TRIP_COST
        before_cost_wins = int((gross_returns > 0).sum())
        after_cost_wins = int((net_returns > 0).sum())
        count = len(subset)

        print(
            f"{threshold:.2f} | {count} | "
            f"{percent(gross_returns.mean() if count else 0)} | "
            f"{percent(before_cost_wins / count if count else 0)} | "
            f"{percent(after_cost_wins / count if count else 0)}"
        )


def print_return_diagnostics(frame):
    print("\nPredicted vs actual return diagnostics")
    print("horizon | correlation | MAE")

    for horizon in [1, 2, 3]:
        actual_column = f"actual_return_{horizon}"
        predicted_column = f"pred_return_{horizon}"

        if actual_column not in frame.columns or predicted_column not in frame.columns:
            print(f"{horizon} | missing columns | missing columns")
            continue

        print(
            f"{horizon} | "
            f"{correlation(frame[actual_column], frame[predicted_column]):.6f} | "
            f"{mae(frame[actual_column], frame[predicted_column]):.10g}"
        )


def agreement_rate(signal_values, future_returns):
    signal_values = pd.to_numeric(signal_values, errors="coerce")
    future_returns = pd.to_numeric(future_returns, errors="coerce")
    valid = signal_values.notna() & future_returns.notna()
    valid = valid & (signal_values != 0) & (future_returns != 0)

    if valid.sum() == 0:
        return np.nan, 0

    agreement = np.sign(signal_values[valid]) == np.sign(future_returns[valid])
    return float(agreement.mean()), int(valid.sum())


def print_flow_direction_diagnostics(frame):
    future_return = frame["actual_return_1"]
    checks = [
        ("market_pressure", "pred_market_pressure_1"),
        ("imbalance_10bps", "pred_imbalance_10bps_1"),
        ("breakout_pressure_index", "pred_breakout_pressure_index_1"),
    ]

    print("\nPredicted flow direction agreement with actual future return_1")
    for label, column in checks:
        if column not in frame.columns:
            print(f"- {label}: missing column {column}")
            continue

        rate, count = agreement_rate(frame[column], future_return)
        if np.isnan(rate):
            print(f"- {label}: no non-zero comparable rows")
        else:
            print(f"- {label}: {percent(rate)} agreement over {count} rows")


def print_winner_loser_probabilities(trades):
    winners = trades[trades["net_return"] > 0]
    losers = trades[trades["net_return"] <= 0]

    print("\nAverage probabilities on winners vs losers:")
    if len(winners) == 0:
        print("- winners: none")
    else:
        print(
            f"- winners avg prob_long: {percent(winners['prob_long'].mean())}, "
            f"avg prob_short: {percent(winners['prob_short'].mean())}"
        )

    if len(losers) == 0:
        print("- losers: none")
    else:
        print(
            f"- losers avg prob_long: {percent(losers['prob_long'].mean())}, "
            f"avg prob_short: {percent(losers['prob_short'].mean())}"
        )


def main():
    frame = load_predictions()
    trades, skipped_rows = create_trades(frame)
    long_trades = trades[trades["signal"] == "LONG"] if len(trades) else trades
    short_trades = trades[trades["signal"] == "SHORT"] if len(trades) else trades
    gross_returns = trades["gross_return"] if len(trades) else pd.Series(dtype=float)
    net_returns = trades["net_return"] if len(trades) else pd.Series(dtype=float)
    before_cost_wins = int((gross_returns > 0).sum())
    after_cost_wins = int((net_returns > 0).sum())

    print("1m flow prediction backtest")
    print(f"Symbol: {SYMBOL}")
    print(f"Prediction file: {PREDICTIONS_PATH}")
    print(f"LONG_THRESHOLD: {LONG_THRESHOLD}")
    print(f"SHORT_THRESHOLD: {SHORT_THRESHOLD}")
    print(f"MAX_NEUTRAL_THRESHOLD: {MAX_NEUTRAL_THRESHOLD}")
    print(f"RETURN_HORIZON: {RETURN_HORIZON}")
    print(f"COST_PROFILE: {COST_PROFILE}")
    print(f"TAKER_ROUND_TRIP_COST: {TAKER_ROUND_TRIP_COST:.4%}")
    print(f"MAKER_ROUND_TRIP_COST: {MAKER_ROUND_TRIP_COST:.4%}")
    print(f"ESTIMATED_SLIPPAGE: {ESTIMATED_SLIPPAGE:.4%}")
    print(f"SPREAD_COST: {SPREAD_COST:.4%}")
    print(f"ROUND_TRIP_COST: {ROUND_TRIP_COST:.4%}")
    print("No trades are placed.")

    print(f"\nTotal prediction rows: {len(frame)}")
    print(f"Total trades: {len(trades)}")
    print(f"Long trades: {len(long_trades)}")
    print(f"Short trades: {len(short_trades)}")
    print(f"Skipped rows: {skipped_rows}")
    print(
        f"Win rate before costs: "
        f"{percent(before_cost_wins / len(trades) if len(trades) else 0)}"
    )
    print(
        f"Win rate after costs: "
        f"{percent(after_cost_wins / len(trades) if len(trades) else 0)}"
    )
    print(f"Average gross return: {percent(gross_returns.mean() if len(trades) else 0)}")
    print(f"Average net return: {percent(net_returns.mean() if len(trades) else 0)}")
    print(f"Compounded return: {percent(compounded_return(net_returns))}")

    if len(trades):
        print_class_distribution(
            "\nActual class distribution among executed trades:",
            trades["actual_class"],
        )
        print_winner_loser_probabilities(trades)
    else:
        print("\nActual class distribution among executed trades: no trades")
        print("\nAverage probabilities on winners vs losers: no trades")

    print_threshold_table(frame, "LONG")
    print_threshold_table(frame, "SHORT")
    print_return_diagnostics(frame)
    print_flow_direction_diagnostics(frame)


if __name__ == "__main__":
    main()
