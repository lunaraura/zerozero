import datetime as dt
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from build_1s_order_flow_training_rows import CLASS_NAMES, feature_columns
from microstructure_model_utils import atomic_write_csv, atomic_write_json, feature_schema_hash, percent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
VENUE_TAG = PRIMARY_VENUE or "legacy"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.70"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.15"))
EMBARGO_ROWS = int(os.getenv("EMBARGO_ROWS", "60"))
EPOCHS = int(os.getenv("EPOCHS", "40"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
HIDDEN_UNITS = int(os.getenv("HIDDEN_UNITS", "64"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))
REGRESSION_LOSS_WEIGHT = float(os.getenv("REGRESSION_LOSS_WEIGHT", "0.20"))
BURST_LOSS_WEIGHT = float(os.getenv("BURST_LOSS_WEIGHT", "0.70"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "200"))
PROMOTE_BEST = os.getenv("PROMOTE_BEST", "false").strip().lower() in {"1", "true", "yes", "y"}
FLOW_1S_CLASS_WEIGHT_MODE = os.getenv("FLOW_1S_CLASS_WEIGHT_MODE", "balanced").strip().lower()
FLOW_1S_MAX_NEUTRAL_RATIO_RAW = os.getenv("FLOW_1S_MAX_NEUTRAL_RATIO", "").strip()
FLOW_1S_MAX_NEUTRAL_RATIO = (
    float(FLOW_1S_MAX_NEUTRAL_RATIO_RAW)
    if FLOW_1S_MAX_NEUTRAL_RATIO_RAW
    else None
)
FLOW_1S_MIN_DIRECTIONAL_MACRO_F1 = float(os.getenv("FLOW_1S_MIN_DIRECTIONAL_MACRO_F1", "0.05"))
FLOW_1S_DIRECTIONAL_MIN_PROB = float(os.getenv("FLOW_1S_DIRECTIONAL_MIN_PROB", "0.45"))
FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN = float(os.getenv("FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN", "0.05"))
FLOW_1S_CLASS_TARGET = os.getenv("FLOW_1S_CLASS_TARGET", "next_1s_flow_class").strip()
FLOW_1S_TARGET_HORIZON_SECONDS = int(os.getenv("FLOW_1S_TARGET_HORIZON_SECONDS", "1"))
VALID_CLASS_TARGETS = {"next_1s_flow_class", "future_3s_flow_class", "future_5s_flow_class"}
VALID_TARGET_HORIZONS = {1, 3, 5}

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
TRAINING_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_1s_order_flow_training_rows.csv"
PREDICTIONS_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_1s_order_flow_predictions.csv"
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "order_flow_1s" / VENUE_TAG
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "active" / SYMBOL / "order_flow_1s" / VENUE_TAG / "model.json"

def regression_targets_for_horizon(horizon_seconds):
    suffix = f"{horizon_seconds}s"
    return [
        f"future_log_market_buy_volume_{suffix}",
        f"future_log_market_sell_volume_{suffix}",
        f"future_market_pressure_{suffix}",
        f"future_log_trade_count_{suffix}",
    ]


def raw_regression_diagnostics_for_horizon(horizon_seconds):
    suffix = f"{horizon_seconds}s"
    return [
        f"future_market_buy_volume_{suffix}",
        f"future_market_sell_volume_{suffix}",
        f"future_trade_count_{suffix}",
    ]


if FLOW_1S_CLASS_TARGET not in VALID_CLASS_TARGETS:
    raise ValueError(
        "FLOW_1S_CLASS_TARGET must be one of: "
        + ", ".join(sorted(VALID_CLASS_TARGETS))
    )
if FLOW_1S_TARGET_HORIZON_SECONDS not in VALID_TARGET_HORIZONS:
    raise ValueError("FLOW_1S_TARGET_HORIZON_SECONDS must be 1, 3, or 5.")

REGRESSION_TARGET_COLUMNS = regression_targets_for_horizon(FLOW_1S_TARGET_HORIZON_SECONDS)
RAW_REGRESSION_DIAGNOSTIC_COLUMNS = raw_regression_diagnostics_for_horizon(FLOW_1S_TARGET_HORIZON_SECONDS)
BURST_TARGET_COLUMNS = [
    "future_aggressive_buy_burst_1s",
    "future_aggressive_sell_burst_1s",
]
CLASS_TARGET_COLUMN = FLOW_1S_CLASS_TARGET
LIVE_1S_PREDICTION_COLUMNS = [
    "timestamp",
    "time",
    "symbol",
    "primary_venue",
    "class_target_column",
    "target_horizon_seconds",
    "model_target_horizon_seconds",
    "prob_sell_dominant_1s",
    "prob_neutral_1s",
    "prob_buy_dominant_1s",
    "decoded_flow_class_1s",
    "pred_market_buy_volume_1s",
    "pred_market_sell_volume_1s",
    "pred_market_pressure_1s",
    "pred_pressure_magnitude_1s",
    "pred_trade_count_1s",
    "buy_burst_prob_1s",
    "sell_burst_prob_1s",
    "model_id",
    "feature_schema_hash",
    "trained_until_timestamp",
]


def softmax(values):
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(np.clip(values, -40.0, 40.0))
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def initialize_model(input_size, hidden_units, regression_outputs, burst_outputs, rng):
    return {
        "w1": rng.normal(0, np.sqrt(2.0 / max(1, input_size)), (input_size, hidden_units)),
        "b1": np.zeros(hidden_units),
        "w_class": rng.normal(0, np.sqrt(2.0 / max(1, hidden_units)), (hidden_units, 3)),
        "b_class": np.zeros(3),
        "w_burst": rng.normal(0, np.sqrt(2.0 / max(1, hidden_units)), (hidden_units, burst_outputs)),
        "b_burst": np.zeros(burst_outputs),
        "w_reg": rng.normal(0, np.sqrt(2.0 / max(1, hidden_units)), (hidden_units, regression_outputs)),
        "b_reg": np.zeros(regression_outputs),
    }


def forward(model, x):
    hidden_pre = x @ model["w1"] + model["b1"]
    hidden = np.maximum(hidden_pre, 0.0)
    class_probabilities = softmax(hidden @ model["w_class"] + model["b_class"])
    burst_probabilities = sigmoid(hidden @ model["w_burst"] + model["b_burst"])
    regression = hidden @ model["w_reg"] + model["b_reg"]
    return hidden_pre, hidden, class_probabilities, burst_probabilities, regression


def standardize(train_values, other_values):
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std < 1e-9] = 1.0
    return (train_values - mean) / std, (other_values - mean) / std, mean, std


def load_training_rows():
    if not TRAINING_PATH.exists():
        raise FileNotFoundError(f"Missing training rows: {TRAINING_PATH}. Run npm run flow1s-build first.")
    frame = pd.read_csv(TRAINING_PATH)
    for column in frame.columns:
        if column != "time" and column != "next_time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    log_fallbacks = {}
    for horizon_seconds in [1, 3, 5]:
        suffix = f"{horizon_seconds}s"
        log_fallbacks[f"future_log_market_buy_volume_{suffix}"] = f"future_market_buy_volume_{suffix}"
        log_fallbacks[f"future_log_market_sell_volume_{suffix}"] = f"future_market_sell_volume_{suffix}"
        log_fallbacks[f"future_log_trade_count_{suffix}"] = f"future_trade_count_{suffix}"
    derived_log_columns = []
    for log_column, raw_column in log_fallbacks.items():
        if log_column not in frame.columns and raw_column in frame.columns:
            raw_values = pd.to_numeric(frame[raw_column], errors="coerce").clip(lower=0.0)
            frame[log_column] = np.log1p(raw_values)
            derived_log_columns.append(log_column)
    if derived_log_columns:
        print(
            "Derived missing log1p regression targets from raw nonnegative columns: "
            + ", ".join(derived_log_columns)
        )
    columns = feature_columns(frame)
    required = ["timestamp", CLASS_TARGET_COLUMN, *REGRESSION_TARGET_COLUMNS, *BURST_TARGET_COLUMNS, *columns]
    frame = frame.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)
    frame[CLASS_TARGET_COLUMN] = frame[CLASS_TARGET_COLUMN].astype(int)
    return frame, columns


def split_frame(frame):
    total = len(frame)
    train_end = int(total * TRAIN_SPLIT)
    validation_end = int(total * (TRAIN_SPLIT + VALIDATION_SPLIT))
    train_end = max(1, min(train_end, total - 2))
    validation_end = max(train_end + 1, min(validation_end, total - 1))
    validation_start = min(total, train_end + EMBARGO_ROWS)
    test_start = min(total, validation_end + EMBARGO_ROWS)
    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[validation_start:validation_end].copy()
    test = frame.iloc[test_start:].copy()
    return train, validation, test, {
        "train_validation_embargo_rows": max(0, validation_start - train_end),
        "validation_test_embargo_rows": max(0, test_start - validation_end),
    }


def downsample_neutral_training_rows(train):
    if FLOW_1S_MAX_NEUTRAL_RATIO is None:
        return train.copy(), {
            "enabled": False,
            "before_rows": int(len(train)),
            "after_rows": int(len(train)),
            "neutral_before": int((train[CLASS_TARGET_COLUMN] == 1).sum()),
            "neutral_after": int((train[CLASS_TARGET_COLUMN] == 1).sum()),
            "directional_rows": int((train[CLASS_TARGET_COLUMN] != 1).sum()),
            "dropped_neutral_rows": 0,
        }

    neutral = train[train[CLASS_TARGET_COLUMN] == 1]
    directional = train[train[CLASS_TARGET_COLUMN] != 1]
    directional_count = len(directional)
    max_neutral = int(np.floor(max(1, directional_count) * FLOW_1S_MAX_NEUTRAL_RATIO))

    if len(neutral) <= max_neutral:
        sampled_neutral = neutral
    elif max_neutral <= 0:
        sampled_neutral = neutral.iloc[0:0]
    else:
        # Keep a deterministic chronological spread of neutral examples.
        keep_positions = np.linspace(0, len(neutral) - 1, max_neutral).round().astype(int)
        sampled_neutral = neutral.iloc[sorted(set(keep_positions))]

    output = pd.concat([directional, sampled_neutral], ignore_index=False)
    output = output.sort_index().reset_index(drop=True)
    return output, {
        "enabled": True,
        "before_rows": int(len(train)),
        "after_rows": int(len(output)),
        "neutral_before": int(len(neutral)),
        "neutral_after": int((output[CLASS_TARGET_COLUMN] == 1).sum()),
        "directional_rows": int(directional_count),
        "dropped_neutral_rows": int(len(train) - len(output)),
        "max_neutral_ratio": FLOW_1S_MAX_NEUTRAL_RATIO,
    }


def class_distribution(values):
    counts = pd.Series(values).value_counts().sort_index()
    total = len(values) or 1
    return {
        CLASS_NAMES[class_id]: {
            "count": int(counts.get(class_id, 0)),
            "pct": float(counts.get(class_id, 0) / total),
        }
        for class_id in [0, 1, 2]
    }


def compute_class_weights(y_class):
    if FLOW_1S_CLASS_WEIGHT_MODE not in {"none", "balanced"}:
        raise ValueError("FLOW_1S_CLASS_WEIGHT_MODE must be none or balanced.")
    if FLOW_1S_CLASS_WEIGHT_MODE == "none":
        return np.ones(3, dtype=np.float64)

    y_class = np.asarray(y_class, dtype=np.int64)
    counts = np.bincount(y_class, minlength=3).astype(np.float64)
    present = counts > 0
    weights = np.zeros(3, dtype=np.float64)
    if present.any():
        weights[present] = len(y_class) / (present.sum() * counts[present])
        sample_average = weights[y_class].mean()
        if sample_average > 0:
            weights = weights / sample_average
    weights[~present] = 0.0
    return weights


def class_weights_dict(weights):
    return {
        CLASS_NAMES[class_id]: float(weights[class_id])
        for class_id in [0, 1, 2]
    }


def burst_distribution(frame):
    output = {}
    total = len(frame) or 1
    for column in BURST_TARGET_COLUMNS:
        positives = int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())
        output[column] = {
            "positive_count": positives,
            "negative_count": int(len(frame) - positives),
            "positive_pct": float(positives / total),
        }
    return output


def train_model(
    x_train,
    y_class,
    y_burst,
    y_reg,
    x_validation,
    y_class_validation,
    y_burst_validation,
    y_reg_validation,
    class_weights,
):
    rng = np.random.default_rng(RANDOM_SEED)
    model = initialize_model(x_train.shape[1], HIDDEN_UNITS, y_reg.shape[1], y_burst.shape[1], rng)
    class_one_hot = np.zeros((len(y_class), 3))
    class_one_hot[np.arange(len(y_class)), y_class] = 1.0
    best_model = {name: value.copy() for name, value in model.items()}
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        for start in range(0, len(x_train), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(x_train))
            xb = x_train[start:end]
            ycb = class_one_hot[start:end]
            ybb = y_burst[start:end]
            yrb = y_reg[start:end]
            hidden_pre, hidden, class_prob, burst_prob, regression = forward(model, xb)
            sample_class_weights = class_weights[y_class[start:end]]
            class_weight_denominator = max(float(sample_class_weights.sum()), 1e-9)
            d_class = (
                (class_prob - ycb)
                * sample_class_weights[:, None]
                / class_weight_denominator
            )
            d_burst = BURST_LOSS_WEIGHT * (burst_prob - ybb) / max(1, len(xb) * ybb.shape[1])
            d_reg = REGRESSION_LOSS_WEIGHT * 2.0 * (regression - yrb) / max(1, len(xb) * yrb.shape[1])

            gradients = {
                "w_class": hidden.T @ d_class,
                "b_class": d_class.sum(axis=0),
                "w_burst": hidden.T @ d_burst,
                "b_burst": d_burst.sum(axis=0),
                "w_reg": hidden.T @ d_reg,
                "b_reg": d_reg.sum(axis=0),
            }
            d_hidden = d_class @ model["w_class"].T + d_burst @ model["w_burst"].T + d_reg @ model["w_reg"].T
            d_hidden[hidden_pre <= 0] = 0.0
            gradients["w1"] = xb.T @ d_hidden
            gradients["b1"] = d_hidden.sum(axis=0)
            for name, gradient in gradients.items():
                model[name] -= LEARNING_RATE * gradient

        _, _, class_validation, burst_validation, reg_validation = forward(model, x_validation)
        validation_weights = class_weights[y_class_validation]
        validation_weight_denominator = max(float(validation_weights.sum()), 1e-9)
        class_loss = -float(
            np.sum(
                validation_weights
                * np.log(
                    np.clip(
                        class_validation[np.arange(len(y_class_validation)), y_class_validation],
                        1e-9,
                        1.0,
                    )
                )
            )
            / validation_weight_denominator
        )
        burst_loss = -np.mean(
            y_burst_validation * np.log(np.clip(burst_validation, 1e-9, 1.0))
            + (1.0 - y_burst_validation) * np.log(np.clip(1.0 - burst_validation, 1e-9, 1.0))
        )
        reg_loss = np.mean((reg_validation - y_reg_validation) ** 2)
        total_loss = class_loss + BURST_LOSS_WEIGHT * burst_loss + REGRESSION_LOSS_WEIGHT * reg_loss
        if total_loss < best_loss:
            best_loss = total_loss
            best_model = {name: value.copy() for name, value in model.items()}
        if epoch == 1 or epoch == EPOCHS or epoch % 10 == 0:
            acc = float((np.argmax(class_validation, axis=1) == y_class_validation).mean())
            print(f"epoch {epoch:03d} | validation loss {total_loss:.6f} class acc {acc:.2%}")
    return best_model


def predict_with_artifact(artifact, frame_or_values):
    if isinstance(frame_or_values, pd.DataFrame):
        x = frame_or_values[artifact["feature_columns"]].to_numpy(dtype=np.float64)
    else:
        x = np.asarray(frame_or_values, dtype=np.float64)
    x = (x - artifact["feature_mean"]) / artifact["feature_std"]
    _, _, class_prob, burst_prob, regression_scaled = forward(artifact["model"], x)
    regression = regression_scaled * artifact["target_std"] + artifact["target_mean"]
    return class_prob, burst_prob, regression


def threshold_decode_class_probabilities(
    class_prob,
    min_prob=FLOW_1S_DIRECTIONAL_MIN_PROB,
    neutral_margin=FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN,
):
    class_prob = np.asarray(class_prob, dtype=np.float64)
    decoded = np.ones(len(class_prob), dtype=np.int64)
    sell = class_prob[:, 0]
    neutral = class_prob[:, 1]
    buy = class_prob[:, 2]
    sell_candidate = (sell >= min_prob) & (sell >= neutral + neutral_margin)
    buy_candidate = (buy >= min_prob) & (buy >= neutral + neutral_margin)
    decoded[sell_candidate & ~buy_candidate] = 0
    decoded[buy_candidate & ~sell_candidate] = 2
    both = sell_candidate & buy_candidate
    decoded[both & (buy > sell)] = 2
    decoded[both & (sell >= buy)] = 0
    return decoded


def precision_recall(actual, probability, threshold=0.5):
    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(probability >= threshold, dtype=np.int64)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    return tp / max(1, tp + fp), tp / max(1, tp + fn), tp, fp, fn


def confusion_matrix(actual_class, predicted_class):
    matrix = np.zeros((3, 3), dtype=np.int64)
    for actual, predicted in zip(actual_class, predicted_class):
        if 0 <= int(actual) <= 2 and 0 <= int(predicted) <= 2:
            matrix[int(actual), int(predicted)] += 1
    return matrix


def per_class_precision_recall(actual_class, predicted_class):
    output = {}
    actual_class = np.asarray(actual_class, dtype=np.int64)
    predicted_class = np.asarray(predicted_class, dtype=np.int64)
    for class_id, name in CLASS_NAMES.items():
        tp = int(((actual_class == class_id) & (predicted_class == class_id)).sum())
        fp = int(((actual_class != class_id) & (predicted_class == class_id)).sum())
        fn = int(((actual_class == class_id) & (predicted_class != class_id)).sum())
        support = int((actual_class == class_id).sum())
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
        output[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return output


def top_enrichment_by_score(score, actual, fractions=(0.01, 0.05, 0.10)):
    score = np.asarray(score, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(actual)
    score = score[valid]
    actual = actual[valid]
    if len(score) == 0:
        return {}

    order = np.argsort(-score)
    baseline_actual = float(actual.mean())
    output = {}
    for fraction in fractions:
        count = max(1, int(np.ceil(len(score) * fraction)))
        selected = order[:count]
        average_actual = float(actual[selected].mean())
        output[f"top_{int(fraction * 100)}pct"] = {
            "selected_count": int(count),
            "average_predicted_score": float(score[selected].mean()),
            "average_actual_value": average_actual,
            "baseline_actual_value": baseline_actual,
            "enrichment_ratio": float(average_actual / max(abs(baseline_actual), 1e-12)),
        }
    return output


def enrichment_diagnostics(frame, burst_prob, regression):
    pressure_column = f"future_market_pressure_{FLOW_1S_TARGET_HORIZON_SECONDS}s"
    buy_volume_column = f"future_market_buy_volume_{FLOW_1S_TARGET_HORIZON_SECONDS}s"
    sell_volume_column = f"future_market_sell_volume_{FLOW_1S_TARGET_HORIZON_SECONDS}s"
    return {
        "predicted_buy_burst_score_vs_actual_future_buy_volume": top_enrichment_by_score(
            burst_prob[:, 0],
            frame[buy_volume_column].to_numpy(dtype=np.float64),
        ),
        "predicted_sell_burst_score_vs_actual_future_sell_volume": top_enrichment_by_score(
            burst_prob[:, 1],
            frame[sell_volume_column].to_numpy(dtype=np.float64),
        ),
        "predicted_pressure_magnitude_vs_actual_absolute_pressure": top_enrichment_by_score(
            np.abs(regression[:, REGRESSION_TARGET_COLUMNS.index(pressure_column)]),
            np.abs(frame[pressure_column].to_numpy(dtype=np.float64)),
        ),
    }


def evaluate(frame, class_prob, burst_prob, regression):
    actual_class = frame[CLASS_TARGET_COLUMN].to_numpy(dtype=np.int64)
    raw_predicted_class = np.argmax(class_prob, axis=1)
    thresholded_predicted_class = threshold_decode_class_probabilities(class_prob)
    majority = np.bincount(actual_class, minlength=3).max() / max(1, len(actual_class))
    raw_matrix = confusion_matrix(actual_class, raw_predicted_class)
    thresholded_matrix = confusion_matrix(actual_class, thresholded_predicted_class)
    raw_per_class = per_class_precision_recall(actual_class, raw_predicted_class)
    per_class = per_class_precision_recall(actual_class, thresholded_predicted_class)
    sell_recall = per_class[CLASS_NAMES[0]]["recall"]
    buy_recall = per_class[CLASS_NAMES[2]]["recall"]
    directional_macro_f1 = (
        per_class[CLASS_NAMES[0]]["f1"]
        + per_class[CLASS_NAMES[2]]["f1"]
    ) / 2.0
    metrics = {
        "rows": int(len(frame)),
        "raw_argmax_class_accuracy": float((raw_predicted_class == actual_class).mean()) if len(frame) else 0.0,
        "thresholded_class_accuracy": float((thresholded_predicted_class == actual_class).mean()) if len(frame) else 0.0,
        "class_accuracy": float((thresholded_predicted_class == actual_class).mean()) if len(frame) else 0.0,
        "majority_baseline_accuracy": float(majority),
        "raw_argmax_confusion_matrix": raw_matrix.tolist(),
        "thresholded_confusion_matrix": thresholded_matrix.tolist(),
        "confusion_matrix": thresholded_matrix.tolist(),
        "raw_argmax_per_class_precision_recall": raw_per_class,
        "per_class_precision_recall": per_class,
        "sell_dominant_recall": float(sell_recall),
        "buy_dominant_recall": float(buy_recall),
        "directional_macro_f1": float(directional_macro_f1),
        "threshold_decode": {
            "directional_min_prob": FLOW_1S_DIRECTIONAL_MIN_PROB,
            "directional_neutral_margin": FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN,
        },
    }
    for index, column in enumerate(REGRESSION_TARGET_COLUMNS):
        metrics[f"{column}_mae"] = float(np.mean(np.abs(regression[:, index] - frame[column].to_numpy(dtype=np.float64))))
        if column.startswith("future_log_market_buy_volume_"):
            raw_column = column.replace("future_log_", "future_")
            if raw_column not in frame.columns:
                continue
            converted = np.expm1(np.clip(regression[:, index], 0.0, None))
            actual = frame[raw_column].to_numpy(dtype=np.float64)
            metrics[f"{raw_column}_converted_mae"] = float(np.mean(np.abs(converted - actual)))
        if column.startswith("future_log_market_sell_volume_"):
            raw_column = column.replace("future_log_", "future_")
            if raw_column not in frame.columns:
                continue
            converted = np.expm1(np.clip(regression[:, index], 0.0, None))
            actual = frame[raw_column].to_numpy(dtype=np.float64)
            metrics[f"{raw_column}_converted_mae"] = float(np.mean(np.abs(converted - actual)))
        if column.startswith("future_log_trade_count_"):
            raw_column = column.replace("future_log_", "future_")
            if raw_column not in frame.columns:
                continue
            converted = np.expm1(np.clip(regression[:, index], 0.0, None))
            actual = frame[raw_column].to_numpy(dtype=np.float64)
            metrics[f"{raw_column}_converted_mae"] = float(np.mean(np.abs(converted - actual)))
    for index, column in enumerate(BURST_TARGET_COLUMNS):
        precision, recall, tp, fp, fn = precision_recall(frame[column], burst_prob[:, index])
        metrics[f"{column}_precision"] = precision
        metrics[f"{column}_recall"] = recall
        metrics[f"{column}_tp"] = tp
        metrics[f"{column}_fp"] = fp
        metrics[f"{column}_fn"] = fn
    return metrics


def print_confusion_matrix(matrix, title="Confusion matrix"):
    print(f"{title} (rows=actual, columns=predicted)")
    labels = [CLASS_NAMES[index] for index in [0, 1, 2]]
    print("actual \\ predicted | " + " | ".join(labels))
    for index, label in enumerate(labels):
        row = matrix[index] if isinstance(matrix, list) else matrix[index].tolist()
        print(f"{label} | " + " | ".join(str(int(value)) for value in row))


def print_per_class_metrics(metrics, key="per_class_precision_recall", title="Per-class precision/recall"):
    print(title)
    per_class = metrics.get(key, {})
    for name in [CLASS_NAMES[index] for index in [0, 1, 2]]:
        info = per_class.get(name, {})
        print(
            f"- {name}: precision={info.get('precision', 0.0):.2%}, "
            f"recall={info.get('recall', 0.0):.2%}, "
            f"f1={info.get('f1', 0.0):.2%}, support={info.get('support', 0)}"
        )


def print_burst_metrics(metrics):
    print("Buy/sell burst precision/recall")
    for column in BURST_TARGET_COLUMNS:
        print(
            f"- {column}: precision={metrics.get(column + '_precision', 0.0):.2%}, "
            f"recall={metrics.get(column + '_recall', 0.0):.2%}, "
            f"tp={metrics.get(column + '_tp', 0)}, fp={metrics.get(column + '_fp', 0)}, "
            f"fn={metrics.get(column + '_fn', 0)}"
        )


def print_enrichment_diagnostics(title, diagnostics):
    print(title)
    if not diagnostics:
        print("- unavailable")
        return
    for group, values in diagnostics.items():
        print(f"- {group}")
        if not values:
            print("  - unavailable")
            continue
        for slice_name, info in values.items():
            print(
                f"  - {slice_name}: n={info['selected_count']}, "
                f"avg_pred_score={info['average_predicted_score']:.6g}, "
                f"avg_actual={info['average_actual_value']:.6g}, "
                f"baseline_actual={info['baseline_actual_value']:.6g}, "
                f"enrichment={info['enrichment_ratio']:.3f}x"
            )


def predictions_frame(frame, class_prob, burst_prob, regression):
    output = frame[["timestamp", "time"]].copy()
    output["symbol"] = SYMBOL
    output["primary_venue"] = VENUE_TAG
    output["class_target_column"] = CLASS_TARGET_COLUMN
    output["target_horizon_seconds"] = FLOW_1S_TARGET_HORIZON_SECONDS
    output["model_target_horizon_seconds"] = FLOW_1S_TARGET_HORIZON_SECONDS
    output["prob_sell_dominant"] = class_prob[:, 0]
    output["prob_neutral"] = class_prob[:, 1]
    output["prob_buy_dominant"] = class_prob[:, 2]
    output["prob_sell_dominant_1s"] = class_prob[:, 0]
    output["prob_neutral_1s"] = class_prob[:, 1]
    output["prob_buy_dominant_1s"] = class_prob[:, 2]
    output["raw_argmax_next_1s_flow_class"] = np.argmax(class_prob, axis=1)
    output[f"raw_argmax_{CLASS_TARGET_COLUMN}"] = np.argmax(class_prob, axis=1)
    thresholded = threshold_decode_class_probabilities(class_prob)
    output["thresholded_next_1s_flow_class"] = thresholded
    output[f"thresholded_{CLASS_TARGET_COLUMN}"] = thresholded
    output["decoded_flow_class_1s"] = [
        CLASS_NAMES.get(int(value), str(int(value)))
        for value in thresholded
    ]
    output["prob_aggressive_buy_burst_1s"] = burst_prob[:, 0]
    output["prob_aggressive_sell_burst_1s"] = burst_prob[:, 1]
    output["buy_burst_prob_1s"] = burst_prob[:, 0]
    output["sell_burst_prob_1s"] = burst_prob[:, 1]
    for index, column in enumerate(REGRESSION_TARGET_COLUMNS):
        output[f"pred_{column}"] = regression[:, index]
        output[f"actual_{column}"] = frame[column].to_numpy()
    suffix = f"{FLOW_1S_TARGET_HORIZON_SECONDS}s"
    output[f"pred_future_market_buy_volume_{suffix}"] = np.expm1(
        output[f"pred_future_log_market_buy_volume_{suffix}"].clip(lower=0.0)
    )
    output[f"pred_future_market_sell_volume_{suffix}"] = np.expm1(
        output[f"pred_future_log_market_sell_volume_{suffix}"].clip(lower=0.0)
    )
    output[f"pred_future_trade_count_{suffix}"] = np.expm1(
        output[f"pred_future_log_trade_count_{suffix}"].clip(lower=0.0)
    )
    output["pred_market_buy_volume_1s"] = output[f"pred_future_market_buy_volume_{suffix}"]
    output["pred_market_sell_volume_1s"] = output[f"pred_future_market_sell_volume_{suffix}"]
    output["pred_market_pressure_1s"] = output[f"pred_future_market_pressure_{suffix}"]
    output["pred_pressure_magnitude_1s"] = output["pred_market_pressure_1s"].abs()
    output["pred_trade_count_1s"] = output[f"pred_future_trade_count_{suffix}"]
    for column in RAW_REGRESSION_DIAGNOSTIC_COLUMNS:
        if column in frame.columns:
            output[f"actual_{column}"] = frame[column].to_numpy()
    output["actual_next_1s_flow_class"] = frame[CLASS_TARGET_COLUMN].to_numpy()
    output[f"actual_{CLASS_TARGET_COLUMN}"] = frame[CLASS_TARGET_COLUMN].to_numpy()
    output["actual_future_aggressive_buy_burst_1s"] = frame["future_aggressive_buy_burst_1s"].to_numpy()
    output["actual_future_aggressive_sell_burst_1s"] = frame["future_aggressive_sell_burst_1s"].to_numpy()
    return output


def serializable_artifact(artifact):
    output = dict(artifact)
    output["model"] = {name: value.tolist() for name, value in artifact["model"].items()}
    for key in ["feature_mean", "feature_std", "target_mean", "target_std"]:
        output[key] = np.asarray(artifact[key]).tolist()
    return output


def save_model(path, artifact):
    atomic_write_json(serializable_artifact(artifact), path)


def load_model(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    artifact["model"] = {name: np.asarray(value, dtype=np.float64) for name, value in artifact["model"].items()}
    for key in ["feature_mean", "feature_std", "target_mean", "target_std"]:
        artifact[key] = np.asarray(artifact[key], dtype=np.float64)
        artifact[key][artifact[key] == 0] = 1.0 if key.endswith("std") else 0.0
    artifact["feature_std"][artifact["feature_std"] < 1e-9] = 1.0
    artifact["target_std"][artifact["target_std"] < 1e-9] = 1.0
    return artifact


def main():
    frame, columns = load_training_rows()
    print("1s order-flow model trainer")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Training path: {TRAINING_PATH}")
    print(f"Rows: {len(frame)}")
    print(f"Feature count: {len(columns)}")
    print(f"EMBARGO_ROWS: {EMBARGO_ROWS}")
    print(f"FLOW_1S_CLASS_WEIGHT_MODE: {FLOW_1S_CLASS_WEIGHT_MODE}")
    print(f"FLOW_1S_DIRECTIONAL_MIN_PROB: {FLOW_1S_DIRECTIONAL_MIN_PROB:.2%}")
    print(f"FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN: {FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN:.2%}")
    print(f"FLOW_1S_CLASS_TARGET: {CLASS_TARGET_COLUMN}")
    print(f"FLOW_1S_TARGET_HORIZON_SECONDS: {FLOW_1S_TARGET_HORIZON_SECONDS}")
    print("Regression targets:")
    for column in REGRESSION_TARGET_COLUMNS:
        print(f"- {column}")
    print(
        "FLOW_1S_MAX_NEUTRAL_RATIO: "
        f"{FLOW_1S_MAX_NEUTRAL_RATIO if FLOW_1S_MAX_NEUTRAL_RATIO is not None else 'off'}"
    )
    print(f"FLOW_1S_MIN_DIRECTIONAL_MACRO_F1: {FLOW_1S_MIN_DIRECTIONAL_MACRO_F1:.2%}")
    print(f"PROMOTE_BEST: {PROMOTE_BEST}")
    if len(frame) < MIN_TRAIN_ROWS:
        print(f"Training skipped: rows {len(frame)} < MIN_TRAIN_ROWS {MIN_TRAIN_ROWS}")
        print("No trades were placed. No orders were sent.")
        return

    train, validation, test, embargo = split_frame(frame)
    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        print("Training skipped: split is empty after embargo.")
        print(f"Dropped train/validation embargo rows: {embargo['train_validation_embargo_rows']}")
        print(f"Dropped validation/test embargo rows: {embargo['validation_test_embargo_rows']}")
        print("No trades were placed. No orders were sent.")
        return
    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Test rows: {len(test)}")
    print(f"Dropped train/validation embargo rows: {embargo['train_validation_embargo_rows']}")
    print(f"Dropped validation/test embargo rows: {embargo['validation_test_embargo_rows']}")
    print("Train class distribution:")
    for name, info in class_distribution(train[CLASS_TARGET_COLUMN]).items():
        print(f"- {name}: {info['count']} ({info['pct']:.2%})")
    train, neutral_downsampling = downsample_neutral_training_rows(train)
    if neutral_downsampling["enabled"]:
        print("Neutral downsampling")
        print(f"- before rows: {neutral_downsampling['before_rows']}")
        print(f"- after rows: {neutral_downsampling['after_rows']}")
        print(f"- neutral before: {neutral_downsampling['neutral_before']}")
        print(f"- neutral after: {neutral_downsampling['neutral_after']}")
        print(f"- directional rows kept: {neutral_downsampling['directional_rows']}")
        print(f"- dropped neutral rows: {neutral_downsampling['dropped_neutral_rows']}")
        print(f"- max neutral ratio: {neutral_downsampling['max_neutral_ratio']}")
        print("Train class distribution after neutral downsampling:")
        for name, info in class_distribution(train[CLASS_TARGET_COLUMN]).items():
            print(f"- {name}: {info['count']} ({info['pct']:.2%})")
    print("Validation class distribution:")
    for name, info in class_distribution(validation[CLASS_TARGET_COLUMN]).items():
        print(f"- {name}: {info['count']} ({info['pct']:.2%})")
    validation_majority = max(
        (info["pct"] for info in class_distribution(validation[CLASS_TARGET_COLUMN]).values()),
        default=0.0,
    )
    print(f"Validation majority-class baseline: {validation_majority:.2%}")

    x_train = train[columns].to_numpy(dtype=np.float64)
    x_validation_raw = validation[columns].to_numpy(dtype=np.float64)
    x_test_raw = test[columns].to_numpy(dtype=np.float64)
    x_train, x_validation, feature_mean, feature_std = standardize(x_train, x_validation_raw)
    x_test = (x_test_raw - feature_mean) / feature_std
    y_reg_train_raw = train[REGRESSION_TARGET_COLUMNS].to_numpy(dtype=np.float64)
    y_reg_validation_raw = validation[REGRESSION_TARGET_COLUMNS].to_numpy(dtype=np.float64)
    y_reg_train, y_reg_validation, target_mean, target_std = standardize(y_reg_train_raw, y_reg_validation_raw)
    y_reg_test = (test[REGRESSION_TARGET_COLUMNS].to_numpy(dtype=np.float64) - target_mean) / target_std
    y_class_train = train[CLASS_TARGET_COLUMN].to_numpy(dtype=np.int64)
    y_class_validation = validation[CLASS_TARGET_COLUMN].to_numpy(dtype=np.int64)
    class_weights = compute_class_weights(y_class_train)
    print("Class weights")
    for name, weight in class_weights_dict(class_weights).items():
        print(f"- {name}: {weight:.6g}")
    y_burst_train = train[BURST_TARGET_COLUMNS].to_numpy(dtype=np.float64)
    y_burst_validation = validation[BURST_TARGET_COLUMNS].to_numpy(dtype=np.float64)
    y_burst_test = test[BURST_TARGET_COLUMNS].to_numpy(dtype=np.float64)

    model = train_model(
        x_train,
        y_class_train,
        y_burst_train,
        y_reg_train,
        x_validation,
        y_class_validation,
        y_burst_validation,
        y_reg_validation,
        class_weights,
    )
    schema_hash = feature_schema_hash(columns)
    created_at = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact = {
        "model_type": "paper_only_1s_order_flow_numpy_mlp",
        "symbol": SYMBOL,
        "model_symbol": SYMBOL,
        "primary_venue": VENUE_TAG,
        "created_at": created_at,
        "model_id": f"{SYMBOL}_{VENUE_TAG}_order_flow_1s_{created_at}_{schema_hash}",
        "trained_until_timestamp": int(train["timestamp"].max()),
        "feature_columns": columns,
        "feature_count": len(columns),
        "feature_schema_hash": schema_hash,
        "regression_target_columns": REGRESSION_TARGET_COLUMNS,
        "burst_target_columns": BURST_TARGET_COLUMNS,
        "class_target_column": CLASS_TARGET_COLUMN,
        "target_horizon_seconds": FLOW_1S_TARGET_HORIZON_SECONDS,
        "target_configuration": {
            "class_target_column": CLASS_TARGET_COLUMN,
            "regression_target_horizon_seconds": FLOW_1S_TARGET_HORIZON_SECONDS,
            "regression_target_columns": REGRESSION_TARGET_COLUMNS,
            "backward_compatible_prediction_columns": True,
        },
        "class_names": CLASS_NAMES,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "hidden_units": HIDDEN_UNITS,
        "model": model,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "class_weight_mode": FLOW_1S_CLASS_WEIGHT_MODE,
        "class_weights": class_weights_dict(class_weights),
        "neutral_downsampling": neutral_downsampling,
        "promotion_requirements": {
            "accuracy_must_beat_majority_baseline": True,
            "or_directional_macro_f1_minimum": FLOW_1S_MIN_DIRECTIONAL_MACRO_F1,
            "directional_macro_f1_uses_thresholded_decode": True,
        },
        "threshold_decode": {
            "directional_min_prob": FLOW_1S_DIRECTIONAL_MIN_PROB,
            "directional_neutral_margin": FLOW_1S_DIRECTIONAL_NEUTRAL_MARGIN,
        },
        "target_distributions": {
            "class": {
                "train": class_distribution(train[CLASS_TARGET_COLUMN]),
                "validation": class_distribution(validation[CLASS_TARGET_COLUMN]),
                "test": class_distribution(test[CLASS_TARGET_COLUMN]),
            },
            "burst": {
                "train": burst_distribution(train),
                "validation": burst_distribution(validation),
                "test": burst_distribution(test),
            },
        },
        "context_availability": {},
    }

    val_class_prob, val_burst_prob, val_reg = predict_with_artifact(artifact, validation)
    test_class_prob, test_burst_prob, test_reg = predict_with_artifact(artifact, test)
    validation_metrics = evaluate(validation, val_class_prob, val_burst_prob, val_reg)
    test_metrics = evaluate(test, test_class_prob, test_burst_prob, test_reg)
    validation_enrichment = enrichment_diagnostics(validation, val_burst_prob, val_reg)
    test_enrichment = enrichment_diagnostics(test, test_burst_prob, test_reg)
    artifact["validation_metrics"] = validation_metrics
    artifact["test_metrics"] = test_metrics
    artifact["validation_enrichment_diagnostics"] = validation_enrichment
    artifact["test_enrichment_diagnostics"] = test_enrichment
    accuracy_gate_passed = validation_metrics["class_accuracy"] > validation_metrics["majority_baseline_accuracy"]
    directional_gate_passed = validation_metrics["directional_macro_f1"] >= FLOW_1S_MIN_DIRECTIONAL_MACRO_F1
    artifact["validation_promotion_gate_results"] = {
        "accuracy_gate_passed": bool(accuracy_gate_passed),
        "directional_macro_f1_gate_passed": bool(directional_gate_passed),
        "class_accuracy": float(validation_metrics["class_accuracy"]),
        "majority_baseline_accuracy": float(validation_metrics["majority_baseline_accuracy"]),
        "directional_macro_f1": float(validation_metrics["directional_macro_f1"]),
        "minimum_directional_macro_f1": float(FLOW_1S_MIN_DIRECTIONAL_MACRO_F1),
    }

    prediction_frame = pd.concat(
        [
            predictions_frame(validation, val_class_prob, val_burst_prob, val_reg).assign(split="validation"),
            predictions_frame(test, test_class_prob, test_burst_prob, test_reg).assign(split="test"),
        ],
        ignore_index=True,
    )
    prediction_frame["model_id"] = artifact["model_id"]
    prediction_frame["model_trained_until_timestamp"] = artifact["trained_until_timestamp"]
    prediction_frame["trained_until_timestamp"] = artifact["trained_until_timestamp"]
    prediction_frame["feature_schema_hash"] = artifact["feature_schema_hash"]
    prediction_frame["training_row_count"] = artifact["training_rows"]
    prediction_frame["validation_score"] = validation_metrics["class_accuracy"]
    for column in LIVE_1S_PREDICTION_COLUMNS:
        if column not in prediction_frame.columns:
            prediction_frame[column] = ""
    prediction_frame = prediction_frame[LIVE_1S_PREDICTION_COLUMNS]

    tag = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate_dir = CANDIDATE_ROOT / tag
    candidate_path = candidate_dir / "model.json"
    save_model(candidate_path, artifact)
    atomic_write_csv(prediction_frame, PREDICTIONS_PATH)
    atomic_write_csv(pd.DataFrame([validation_metrics, test_metrics], index=["validation", "test"]), candidate_dir / "validation_metrics.csv")

    print("Validation metrics")
    for key, value in validation_metrics.items():
        if key in {"confusion_matrix", "per_class_precision_recall"}:
            continue
        print(f"- {key}: {value:.6g}" if isinstance(value, float) else f"- {key}: {value}")
    print_confusion_matrix(validation_metrics["raw_argmax_confusion_matrix"], "Raw argmax confusion matrix")
    print_confusion_matrix(validation_metrics["thresholded_confusion_matrix"], "Thresholded confusion matrix")
    print_per_class_metrics(
        validation_metrics,
        key="raw_argmax_per_class_precision_recall",
        title="Raw argmax per-class precision/recall",
    )
    print_per_class_metrics(
        validation_metrics,
        key="per_class_precision_recall",
        title="Thresholded per-class precision/recall",
    )
    print_burst_metrics(validation_metrics)
    print_enrichment_diagnostics("Validation top-score enrichment diagnostics", validation_enrichment)
    print("Test metrics")
    for key, value in test_metrics.items():
        if key in {"confusion_matrix", "per_class_precision_recall"}:
            continue
        print(f"- {key}: {value:.6g}" if isinstance(value, float) else f"- {key}: {value}")
    print_confusion_matrix(test_metrics["raw_argmax_confusion_matrix"], "Raw argmax confusion matrix")
    print_confusion_matrix(test_metrics["thresholded_confusion_matrix"], "Thresholded confusion matrix")
    print_per_class_metrics(
        test_metrics,
        key="raw_argmax_per_class_precision_recall",
        title="Raw argmax per-class precision/recall",
    )
    print_per_class_metrics(
        test_metrics,
        key="per_class_precision_recall",
        title="Thresholded per-class precision/recall",
    )
    print_burst_metrics(test_metrics)
    print_enrichment_diagnostics("Test top-score enrichment diagnostics", test_enrichment)
    print("Validation promotion gates")
    print(
        f"- accuracy > majority baseline: {accuracy_gate_passed} "
        f"({validation_metrics['class_accuracy']:.2%} vs "
        f"{validation_metrics['majority_baseline_accuracy']:.2%})"
    )
    print(
        f"- directional_macro_f1 >= minimum: {directional_gate_passed} "
        f"({validation_metrics['directional_macro_f1']:.2%} vs "
        f"{FLOW_1S_MIN_DIRECTIONAL_MACRO_F1:.2%})"
    )

    promoted = False
    if PROMOTE_BEST and (accuracy_gate_passed or directional_gate_passed):
        ACTIVE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, ACTIVE_MODEL_PATH)
        promoted = True
        print(f"Candidate promoted to active model: {ACTIVE_MODEL_PATH}")
    else:
        print("Candidate not promoted.")
        if not PROMOTE_BEST:
            print("- PROMOTE_BEST is false")
        else:
            print("- validation class accuracy did not beat majority baseline")
            print(
                "- validation directional_macro_f1 did not reach "
                f"{FLOW_1S_MIN_DIRECTIONAL_MACRO_F1:.2%}"
            )
    print(f"Candidate model saved to: {candidate_path}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Candidate promoted: {'yes' if promoted else 'no'}")
    print("No trades were placed. No orders were sent.")


if __name__ == "__main__":
    main()
