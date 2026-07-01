import os
from pathlib import Path

import numpy as np
import pandas as pd

from live_3m_model_utils import parse_bool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
ROWS = int(os.getenv("ROWS", "10"))

PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "live_predictions" / f"{SYMBOL}_live_3m_predictions.csv"
)

CLASS_NAMES = {
    0: "bearish",
    1: "neutral",
    2: "bullish",
}


def load_predictions():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing live prediction file: {PREDICTIONS_PATH}. "
            "Run scripts/run_live_3m_prediction_loop.py first."
        )

    frame = pd.read_csv(PREDICTIONS_PATH)
    required = ["timestamp", "time", "prob_short", "prob_neutral", "prob_long"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")

    for column in frame.columns:
        if column not in {"time", "model_source", "label_ready"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["label_ready"] = (
        frame["label_ready"].apply(parse_bool)
        if "label_ready" in frame.columns
        else False
    )
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    return frame.reset_index(drop=True)


def value(row, column, default=np.nan):
    return row[column] if column in row.index else default


def format_number(number, decimals=6):
    if pd.isna(number):
        return "blank"
    return f"{float(number):.{decimals}f}"


def format_percent(number):
    if pd.isna(number):
        return "blank"
    return f"{float(number) * 100:.2f}%"


def predicted_class(row):
    probabilities = [
        float(row["prob_short"]),
        float(row["prob_neutral"]),
        float(row["prob_long"]),
    ]
    class_id = int(np.argmax(probabilities))
    return class_id, CLASS_NAMES[class_id], probabilities[class_id]


def confidence_label(confidence):
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


def usefulness_note(row, predicted_class_id):
    label_ready = bool(row["label_ready"])
    if not label_ready:
        return "pending; useful for paper tracking only until the 3m label arrives"

    actual_class = value(row, "actual_path_class")
    if pd.isna(actual_class):
        return "complete flag is set, but actual_path_class is blank"

    actual_class = int(actual_class)
    if predicted_class_id == actual_class:
        return "correct versus labeled path class"
    return f"incorrect; actual was {CLASS_NAMES.get(actual_class, actual_class)}"


def print_prediction(row):
    class_id, class_name, confidence = predicted_class(row)
    label_ready = bool(row["label_ready"])

    print("-" * 72)
    print(f"time: {value(row, 'time', 'blank')}")
    print(f"model_source: {value(row, 'model_source', 'blank')}")
    print(f"prob_short: {format_percent(value(row, 'prob_short'))}")
    print(f"prob_neutral: {format_percent(value(row, 'prob_neutral'))}")
    print(f"prob_long: {format_percent(value(row, 'prob_long'))}")
    print(f"predicted_class: {class_name}")
    print(f"confidence: {format_percent(confidence)}")
    print(f"pred_future_return_3: {format_percent(value(row, 'pred_future_return_3'))}")
    print(
        "pred_future_volume_delta_3m: "
        f"{format_number(value(row, 'pred_future_volume_delta_3m'))}"
    )
    print(
        "pred_future_market_pressure_3m: "
        f"{format_number(value(row, 'pred_future_market_pressure_3m'))}"
    )
    print(
        "pred_future_order_book_imbalance_10bps_3m: "
        f"{format_number(value(row, 'pred_future_order_book_imbalance_10bps_3m'))}"
    )
    print(
        "pred_future_spread_percent_3m: "
        f"{format_percent(value(row, 'pred_future_spread_percent_3m'))}"
    )
    print(f"label_ready: {label_ready}")
    print(f"actual_path_class: {format_number(value(row, 'actual_path_class'), 0)}")
    print(
        f"actual_future_return_3: "
        f"{format_percent(value(row, 'actual_future_return_3'))}"
    )
    print(
        f"actual_volume_delta_3m: "
        f"{format_number(value(row, 'actual_volume_delta_3m'))}"
    )
    print(
        f"actual_market_pressure_3m: "
        f"{format_number(value(row, 'actual_market_pressure_3m'))}"
    )

    print("Interpretation:")
    print(f"  Bias: {class_name}")
    print(f"  Confidence: {confidence_label(confidence)}")
    print(f"  Label status: {'complete' if label_ready else 'pending'}")
    print(f"  Usefulness note: {usefulness_note(row, class_id)}")


def main():
    frame = load_predictions()
    rows = frame.tail(max(1, ROWS))

    print("Latest live 3m paper predictions")
    print(f"SYMBOL: {SYMBOL}")
    print(f"Prediction file: {PREDICTIONS_PATH}")
    print(f"Rows in file: {len(frame)}")
    print(f"Rows shown: {len(rows)}")
    print("No trades are placed. This is a read-only paper prediction view.")

    for _, row in rows.iterrows():
        print_prediction(row)


if __name__ == "__main__":
    main()
