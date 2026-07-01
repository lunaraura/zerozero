import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from build_1s_order_flow_training_rows import (
    CLASS_NAMES,
    FLOW_1S_MIN_DIRECTIONAL_TRADE_COUNT,
    FLOW_1S_MIN_DIRECTIONAL_VOLUME,
    FLOW_1S_PRESSURE_CLASS_THRESHOLD,
    flow_class_from_activity,
    pressure_from_volumes,
)
from microstructure_model_utils import atomic_write_csv, percent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
VENUE_TAG = PRIMARY_VENUE or "legacy"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
FLOW_1S_DIRECTIONAL_MIN_PROB = float(os.getenv("FLOW_1S_DIRECTIONAL_MIN_PROB", "0.45"))
FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN = float(os.getenv("FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN", "0.05"))
MAX_NEXT_SECOND_GAP_MS = int(os.getenv("MAX_NEXT_SECOND_GAP_MS", "1500"))
FLOW_1S_EVALUATION_TARGET_HORIZON_SECONDS = int(os.getenv("FLOW_1S_EVALUATION_TARGET_HORIZON_SECONDS", "1"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
PREDICTION_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_1s_order_flow_predictions.csv"
SNAPSHOT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_10s_flow.csv"
PAPER_SIGNAL_LOG_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_paper_signal_log.csv"
REPORT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_1s_forecast_evaluation.csv"

CLASS_TO_ID = {value: key for key, value in CLASS_NAMES.items()}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), f"missing: {path}"
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(), f"empty: {path}"
    except Exception as error:
        return pd.DataFrame(), f"read error for {path}: {error}"
    if len(frame) == 0:
        return pd.DataFrame(), f"empty: {path}"
    return frame, None


def normalize_timestamps(frame):
    frame = frame.copy()
    if "timestamp" not in frame.columns:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    return frame.sort_values("timestamp").reset_index(drop=True)


def numeric_value(row, column, default=np.nan):
    try:
        value = float(row.get(column, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def mid_price(row):
    for column in ["mid_price", "close"]:
        value = numeric_value(row, column)
        if np.isfinite(value) and value > 0:
            return value
    bid = numeric_value(row, "best_bid")
    ask = numeric_value(row, "best_ask")
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return np.nan


def threshold_decode(probabilities, min_prob=FLOW_1S_DIRECTIONAL_MIN_PROB, neutral_margin=FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    decoded = np.ones(len(probabilities), dtype=np.int64)
    sell = probabilities[:, 0]
    neutral = probabilities[:, 1]
    buy = probabilities[:, 2]
    sell_candidate = (sell >= min_prob) & (sell >= neutral + neutral_margin)
    buy_candidate = (buy >= min_prob) & (buy >= neutral + neutral_margin)
    decoded[sell_candidate & ~buy_candidate] = 0
    decoded[buy_candidate & ~sell_candidate] = 2
    both = sell_candidate & buy_candidate
    decoded[both & (buy > sell)] = 2
    decoded[both & (sell >= buy)] = 0
    return decoded


def prediction_target_horizon(prediction):
    value = numeric_value(
        prediction,
        "model_target_horizon_seconds",
        numeric_value(prediction, "target_horizon_seconds", np.nan),
    )
    if np.isfinite(value) and int(value) in {1, 3, 5}:
        return int(value)
    return FLOW_1S_EVALUATION_TARGET_HORIZON_SECONDS


def future_snapshot_window(snapshots, timestamp, horizon_seconds):
    timestamps = snapshots["timestamp"].to_numpy(dtype=np.int64)
    future_index = int(np.searchsorted(timestamps, timestamp, side="right"))
    if future_index + horizon_seconds > len(snapshots):
        return pd.DataFrame()
    rows = snapshots.iloc[future_index:future_index + horizon_seconds].copy()
    previous_timestamp = timestamp
    for _, row in rows.iterrows():
        row_timestamp = int(row["timestamp"])
        if row_timestamp <= previous_timestamp or row_timestamp - previous_timestamp > MAX_NEXT_SECOND_GAP_MS:
            return pd.DataFrame()
        previous_timestamp = row_timestamp
    return rows


def sum_future_column(frame, column):
    if column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return float(values.sum())


def attach_paper_signal_tags(evaluated):
    log, _ = read_csv(PAPER_SIGNAL_LOG_PATH)
    if len(log) == 0 or "timestamp" not in log.columns or "paper_signal_tag" not in log.columns:
        evaluated["paper_signal_tag"] = ""
        return evaluated
    log = normalize_timestamps(log)
    tag_map = {
        int(row["timestamp"]): str(row.get("paper_signal_tag", "") or "")
        for _, row in log.iterrows()
    }
    evaluated["paper_signal_tag"] = evaluated["timestamp"].map(tag_map).fillna("")
    return evaluated


def build_evaluation_rows():
    warnings = []
    predictions, error = read_csv(PREDICTION_PATH)
    if error:
        warnings.append(error)
    snapshots, error = read_csv(SNAPSHOT_PATH)
    if error:
        warnings.append(error)
    predictions = normalize_timestamps(predictions)
    snapshots = normalize_timestamps(snapshots)
    if len(predictions) == 0 or len(snapshots) == 0:
        return pd.DataFrame(), warnings

    required_prediction_columns = [
        "prob_sell_dominant_1s",
        "prob_neutral_1s",
        "prob_buy_dominant_1s",
    ]
    missing = [column for column in required_prediction_columns if column not in predictions.columns]
    if missing:
        warnings.append(f"missing prediction columns: {missing}")
        return pd.DataFrame(), warnings

    snapshot_timestamps = snapshots["timestamp"].to_numpy(dtype=np.int64)
    rows = []
    for _, prediction in predictions.iterrows():
        timestamp = int(prediction["timestamp"])
        current_index = int(np.searchsorted(snapshot_timestamps, timestamp, side="right") - 1)
        horizon_seconds = prediction_target_horizon(prediction)
        if current_index < 0:
            continue
        current = snapshots.iloc[current_index]
        future_window = future_snapshot_window(snapshots, timestamp, horizon_seconds)
        if len(future_window) == 0:
            continue
        future = future_window.iloc[-1]
        future_timestamp = int(future["timestamp"])

        buy_volume = sum_future_column(future_window, "market_buy_volume_10s")
        sell_volume = sum_future_column(future_window, "market_sell_volume_10s")
        total_volume = sum_future_column(future_window, "total_trade_volume_10s")
        trade_count = sum_future_column(future_window, "trade_count_10s")
        pressure = pressure_from_volumes(buy_volume, sell_volume)
        actual_class = flow_class_from_activity(pressure, total_volume, trade_count)
        current_mid = mid_price(current)
        future_mid = mid_price(future)
        realized_return = future_mid / current_mid - 1.0 if current_mid > 0 and future_mid > 0 else np.nan

        probs = [
            numeric_value(prediction, "prob_sell_dominant_1s", 0.0),
            numeric_value(prediction, "prob_neutral_1s", 0.0),
            numeric_value(prediction, "prob_buy_dominant_1s", 0.0),
        ]
        raw_argmax = int(np.argmax(probs))
        decoded_label = str(prediction.get("decoded_flow_class_1s", "") or "")
        csv_thresholded = CLASS_TO_ID.get(decoded_label)
        thresholded = int(threshold_decode(np.asarray([probs]))[0])
        if csv_thresholded is not None:
            thresholded = int(csv_thresholded)

        rows.append(
            {
                "timestamp": timestamp,
                "future_timestamp": future_timestamp,
                "target_horizon_seconds": horizon_seconds,
                "prob_sell": probs[0],
                "prob_neutral": probs[1],
                "prob_buy": probs[2],
                "confidence": max(probs),
                "raw_argmax_class": raw_argmax,
                "thresholded_class": thresholded,
                "actual_class": actual_class,
                "actual_pressure": pressure,
                "actual_return_1s": realized_return,
                "predicted_pressure": numeric_value(prediction, "pred_market_pressure_1s"),
                "model_id": str(prediction.get("model_id", "") or ""),
            }
        )

    evaluated = pd.DataFrame(rows)
    if len(evaluated) > 0:
        evaluated = attach_paper_signal_tags(evaluated)
    return evaluated, warnings


def confusion_matrix(actual, predicted):
    matrix = np.zeros((3, 3), dtype=np.int64)
    for a, p in zip(actual, predicted):
        if 0 <= int(a) <= 2 and 0 <= int(p) <= 2:
            matrix[int(a), int(p)] += 1
    return matrix


def class_precision_recall(actual, predicted, class_id):
    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    tp = int(((actual == class_id) & (predicted == class_id)).sum())
    fp = int(((actual != class_id) & (predicted == class_id)).sum())
    fn = int(((actual == class_id) & (predicted != class_id)).sum())
    return {
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": int((actual == class_id).sum()),
    }


def add_report_row(rows, section, metric, value, extra=None):
    payload = {"section": section, "metric": metric, "value": value}
    if extra:
        payload.update(extra)
    rows.append(payload)


def summarize_overall(evaluated, report_rows):
    if len(evaluated) == 0:
        add_report_row(report_rows, "overall", "total_evaluated_rows", 0)
        return
    actual = evaluated["actual_class"].to_numpy(dtype=np.int64)
    raw = evaluated["raw_argmax_class"].to_numpy(dtype=np.int64)
    thresholded = evaluated["thresholded_class"].to_numpy(dtype=np.int64)
    majority_count = pd.Series(actual).value_counts().max()
    directional = evaluated[evaluated["thresholded_class"] != 1]

    add_report_row(report_rows, "overall", "total_evaluated_rows", len(evaluated))
    if "target_horizon_seconds" in evaluated.columns:
        add_report_row(
            report_rows,
            "overall",
            "target_horizon_seconds_mode",
            float(evaluated["target_horizon_seconds"].mode().iloc[0]) if len(evaluated["target_horizon_seconds"].mode()) else np.nan,
        )
    add_report_row(report_rows, "overall", "raw_argmax_accuracy", float((actual == raw).mean()))
    add_report_row(report_rows, "overall", "thresholded_class_accuracy", float((actual == thresholded).mean()))
    add_report_row(report_rows, "overall", "majority_baseline", float(majority_count / max(1, len(evaluated))))
    add_report_row(
        report_rows,
        "overall",
        "directional_accuracy_excluding_neutral",
        float((directional["actual_class"] == directional["thresholded_class"]).mean()) if len(directional) else np.nan,
    )
    for class_id, label in [(2, "buy"), (0, "sell")]:
        metrics = class_precision_recall(actual, thresholded, class_id)
        add_report_row(report_rows, "overall", f"{label}_precision", metrics["precision"], metrics)
        add_report_row(report_rows, "overall", f"{label}_recall", metrics["recall"], metrics)

    matrix = confusion_matrix(actual, thresholded)
    for actual_id, actual_label in CLASS_NAMES.items():
        for predicted_id, predicted_label in CLASS_NAMES.items():
            add_report_row(
                report_rows,
                "confusion_matrix",
                f"actual_{actual_label}_predicted_{predicted_label}",
                int(matrix[actual_id, predicted_id]),
                {"actual_class": actual_label, "predicted_class": predicted_label},
            )


def summarize_confidence_buckets(evaluated, report_rows):
    buckets = [(0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.000001)]
    for low, high in buckets:
        subset = evaluated[(evaluated["confidence"] >= low) & (evaluated["confidence"] < high)]
        label = f"{low:.2f}-{min(high, 1.0):.2f}"
        accuracy = (
            float((subset["actual_class"] == subset["thresholded_class"]).mean())
            if len(subset)
            else np.nan
        )
        add_report_row(
            report_rows,
            "confidence_bucket",
            label,
            accuracy,
            {
                "bucket_low": low,
                "bucket_high": min(high, 1.0),
                "rows": int(len(subset)),
                "accuracy": accuracy,
                "avg_confidence": float(subset["confidence"].mean()) if len(subset) else np.nan,
            },
        )


def summarize_predicted_class_outcomes(evaluated, report_rows):
    for class_id, label in CLASS_NAMES.items():
        subset = evaluated[evaluated["thresholded_class"] == class_id]
        add_report_row(
            report_rows,
            "predicted_class_outcomes",
            label,
            int(len(subset)),
            {
                "predicted_class": label,
                "rows": int(len(subset)),
                "avg_realized_return_1s": float(subset["actual_return_1s"].mean()) if len(subset) else np.nan,
                "avg_realized_pressure": float(subset["actual_pressure"].mean()) if len(subset) else np.nan,
                "accuracy": float((subset["actual_class"] == subset["thresholded_class"]).mean()) if len(subset) else np.nan,
            },
        )


def summarize_tags(evaluated, report_rows):
    if "paper_signal_tag" not in evaluated.columns:
        return
    tagged = evaluated[evaluated["paper_signal_tag"].astype(str).str.len() > 0]
    if len(tagged) == 0:
        return
    for tag, subset in tagged.groupby("paper_signal_tag"):
        add_report_row(
            report_rows,
            "paper_signal_tag",
            str(tag),
            int(len(subset)),
            {
                "tag": str(tag),
                "rows": int(len(subset)),
                "accuracy": float((subset["actual_class"] == subset["thresholded_class"]).mean()),
                "avg_realized_return_1s": float(subset["actual_return_1s"].mean()),
                "avg_realized_pressure": float(subset["actual_pressure"].mean()),
                "avg_confidence": float(subset["confidence"].mean()),
            },
        )


def threshold_decode_for_sweep(evaluated, min_prob, neutral_margin=FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN):
    probs = evaluated[["prob_sell", "prob_neutral", "prob_buy"]].to_numpy(dtype=np.float64)
    return threshold_decode(probs, min_prob=min_prob, neutral_margin=neutral_margin)


def summarize_abstention(evaluated, report_rows):
    actual = evaluated["actual_class"].to_numpy(dtype=np.int64)
    actual_directional = actual != 1
    for threshold in np.arange(0.40, 0.91, 0.05):
        decoded = threshold_decode_for_sweep(evaluated, threshold)
        kept = decoded != 1
        correct_directional = kept & (decoded == actual) & actual_directional
        false_directional = kept & (decoded != actual)
        actual_directional_total = max(1, int(actual_directional.sum()))
        directional_precision = float(correct_directional.sum() / max(1, int(kept.sum())))
        directional_recall = float(correct_directional.sum() / actual_directional_total)
        false_directional_rate = float(false_directional.sum() / max(1, int(kept.sum())))
        add_report_row(
            report_rows,
            "abstention",
            f"threshold_{threshold:.2f}",
            directional_precision,
            {
                "threshold": round(float(threshold), 2),
                "rows_kept": int(kept.sum()),
                "keep_rate": float(kept.mean()),
                "directional_precision": directional_precision,
                "directional_recall": directional_recall,
                "false_directional_rate": false_directional_rate,
            },
        )


def recommended_threshold_policy(evaluated, report_rows):
    best = None
    for threshold in np.arange(0.40, 0.91, 0.05):
        decoded = threshold_decode_for_sweep(evaluated, threshold)
        kept = decoded != 1
        if kept.sum() < 20:
            continue
        actual = evaluated["actual_class"].to_numpy(dtype=np.int64)
        actual_directional = actual != 1
        correct_directional = kept & (decoded == actual) & actual_directional
        false_directional = kept & (decoded != actual)
        precision = float(correct_directional.sum() / max(1, int(kept.sum())))
        recall = float(correct_directional.sum() / max(1, int(actual_directional.sum())))
        false_rate = float(false_directional.sum() / max(1, int(kept.sum())))
        score = precision + 0.5 * recall - 0.75 * false_rate
        candidate = {
            "threshold": round(float(threshold), 2),
            "neutral_margin": FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN,
            "score": score,
            "directional_precision": precision,
            "directional_recall": recall,
            "false_directional_rate": false_rate,
            "rows_kept": int(kept.sum()),
        }
        if best is None or score > best["score"]:
            best = candidate
    if best is None:
        best = {
            "threshold": FLOW_1S_DIRECTIONAL_MIN_PROB,
            "neutral_margin": FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN,
            "score": np.nan,
            "directional_precision": np.nan,
            "directional_recall": np.nan,
            "false_directional_rate": np.nan,
            "rows_kept": 0,
        }
    add_report_row(
        report_rows,
        "recommended_policy",
        "FLOW_1S_DIRECTIONAL_MIN_PROB",
        best["threshold"],
        best,
    )
    add_report_row(
        report_rows,
        "recommended_policy",
        "FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN",
        best["neutral_margin"],
        best,
    )
    return best


def build_report(evaluated, warnings):
    report_rows = []
    for warning in warnings:
        add_report_row(report_rows, "warning", "file_warning", warning)
    summarize_overall(evaluated, report_rows)
    if len(evaluated) > 0:
        summarize_confidence_buckets(evaluated, report_rows)
        summarize_predicted_class_outcomes(evaluated, report_rows)
        summarize_tags(evaluated, report_rows)
        summarize_abstention(evaluated, report_rows)
        best = recommended_threshold_policy(evaluated, report_rows)
    else:
        best = None
    return pd.DataFrame(report_rows), best


def print_confusion_matrix_from_report(report):
    subset = report[report["section"] == "confusion_matrix"]
    if len(subset) == 0:
        return
    labels = [CLASS_NAMES[index] for index in [0, 1, 2]]
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    for _, row in subset.iterrows():
        matrix.loc[row["actual_class"], row["predicted_class"]] = int(row["value"])
    print("Confusion matrix (thresholded, rows=actual, columns=predicted)")
    print(matrix.to_string())


def main():
    evaluated, warnings = build_evaluation_rows()
    report, best = build_report(evaluated, warnings)
    atomic_write_csv(report, REPORT_PATH)

    print("1s forecast log evaluation")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {VENUE_TAG}")
    print(f"Predictions: {PREDICTION_PATH}")
    print(f"Snapshots: {SNAPSHOT_PATH}")
    print(f"Paper signal log: {PAPER_SIGNAL_LOG_PATH}")
    print(f"FLOW_1S_PRESSURE_CLASS_THRESHOLD: {FLOW_1S_PRESSURE_CLASS_THRESHOLD:.3f}")
    print(f"FLOW_1S_MIN_DIRECTIONAL_VOLUME: {FLOW_1S_MIN_DIRECTIONAL_VOLUME:.8g}")
    print(f"FLOW_1S_MIN_DIRECTIONAL_TRADE_COUNT: {FLOW_1S_MIN_DIRECTIONAL_TRADE_COUNT:.8g}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Total evaluated rows: {len(evaluated)}")
    overall = report[report["section"] == "overall"]
    for _, row in overall.iterrows():
        value = row["value"]
        if isinstance(value, float) and np.isfinite(value) and "rows" not in str(row["metric"]):
            print(f"- {row['metric']}: {percent(value)}")
        else:
            print(f"- {row['metric']}: {value}")
    print_confusion_matrix_from_report(report)
    if best:
        print("Recommended threshold policy")
        print(f"- FLOW_1S_DIRECTIONAL_MIN_PROB={best['threshold']:.2f}")
        print(f"- FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN={best['neutral_margin']:.2f}")
        print(
            "- based on directional precision="
            f"{percent(best['directional_precision'])}, recall={percent(best['directional_recall'])}, "
            f"false directional rate={percent(best['false_directional_rate'])}, rows_kept={best['rows_kept']}"
        )
    print(f"Saved report: {REPORT_PATH}")
    print("No trades/orders/private API behavior.")


if __name__ == "__main__":
    main()
