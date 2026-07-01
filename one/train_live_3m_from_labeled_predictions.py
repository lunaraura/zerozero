import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from live_3m_model_utils import (
    CLASS_LONG,
    CLASS_NEUTRAL,
    CLASS_SHORT,
    CLASS_NAMES,
    REGRESSION_TARGET_COLUMNS,
    atomic_write_csv,
    build_input_window,
    choose_feature_columns,
    coerce_feature_ready,
    coerce_numeric_columns,
    compounded_return_and_drawdown,
    forward,
    initialize_model,
    load_model,
    parse_bool,
    predict_with_model,
    save_model,
)
from hierarchical_context import attach_hierarchical_context, print_context_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
LOOKBACK = int(os.getenv("LOOKBACK", "30"))
EMBARGO_ROWS = int(os.getenv("EMBARGO_ROWS", str(max(LOOKBACK, 3))))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.70"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.15"))
TEST_SPLIT = float(os.getenv("TEST_SPLIT", "0.15"))
EPOCHS = int(os.getenv("EPOCHS", "40"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
HIDDEN_UNITS = int(os.getenv("HIDDEN_UNITS", "48"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))
MIN_LABELED_ROWS = int(os.getenv("MIN_LABELED_ROWS", "200"))
MIN_VALIDATION_ROWS = int(os.getenv("MIN_VALIDATION_ROWS", "50"))
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
LONG_THRESHOLD = float(os.getenv("LONG_THRESHOLD", "0.60"))
SHORT_THRESHOLD = float(os.getenv("SHORT_THRESHOLD", "0.60"))
MAX_NEUTRAL_THRESHOLD = float(os.getenv("MAX_NEUTRAL_THRESHOLD", "0.35"))
MIN_EXPECTED_RETURN = float(os.getenv("MIN_EXPECTED_RETURN", "0.0"))
MIN_REGIME_CONFIDENCE = float(os.getenv("MIN_REGIME_CONFIDENCE", "0.0"))
MAX_SPREAD = float(os.getenv("MAX_SPREAD", "1.0"))
MAX_NEUTRAL_TRADE_RATIO = float(os.getenv("MAX_NEUTRAL_TRADE_RATIO", "0.60"))
MAX_DRAWDOWN_WORSE_ALLOWANCE = float(
    os.getenv("MAX_DRAWDOWN_WORSE_ALLOWANCE", "0.02")
)
REGRESSION_LOSS_WEIGHT = float(os.getenv("REGRESSION_LOSS_WEIGHT", "0.25"))
TRAINING_TARGET_STAGE = int(os.getenv("TRAINING_TARGET_STAGE", "1"))
CLASS_TARGET = os.getenv("CLASS_TARGET", "actual_future_return_class").strip()
SAMPLE_WEIGHT_MODE = os.getenv("SAMPLE_WEIGHT_MODE", "tradeability").strip().lower()
PROMOTE_BEST = os.getenv("PROMOTE_BEST", "true").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
AUTO_REBUILD_FEATURES = os.getenv("AUTO_REBUILD_FEATURES", "true").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
MIN_PROMOTION_VALIDATION_ROWS = int(os.getenv("MIN_PROMOTION_VALIDATION_ROWS", "200"))
MIN_PROMOTION_TRADES = int(os.getenv("MIN_PROMOTION_TRADES", "20"))
MIN_CLASS_ACCURACY_EDGE = float(os.getenv("MIN_CLASS_ACCURACY_EDGE", "0.02"))
MIN_WINNER_PROBABILITY_EDGE = float(os.getenv("MIN_WINNER_PROBABILITY_EDGE", "0.02"))
MIN_AVERAGE_NET_RETURN = float(os.getenv("MIN_AVERAGE_NET_RETURN", "0.0"))
MAX_PROMOTION_DRAWDOWN = float(os.getenv("MAX_PROMOTION_DRAWDOWN", "-0.08"))
MIN_TOP5_NET_RETURN = float(os.getenv("MIN_TOP5_NET_RETURN", "0.0"))
MIN_TOP5_WIN_AFTER_COST = float(os.getenv("MIN_TOP5_WIN_AFTER_COST", "0.52"))
MAX_LONG_SHORT_TRADE_RATIO = float(os.getenv("MAX_LONG_SHORT_TRADE_RATIO", "4.0"))
RUN_WALK_FORWARD = os.getenv("RUN_WALK_FORWARD", "true").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
MIN_WALK_FORWARD_PERIODS = int(os.getenv("MIN_WALK_FORWARD_PERIODS", "2"))
WALK_FORWARD_MIN_PERIOD_ROWS = int(
    os.getenv("WALK_FORWARD_MIN_PERIOD_ROWS", str(MIN_PROMOTION_VALIDATION_ROWS))
)

LABELED_PATH = (
    PROJECT_ROOT / "data" / "live_training" / f"{SYMBOL}_labeled_3m_training_rows.csv"
)
PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "live_predictions" / f"{SYMBOL}_live_3m_predictions.csv"
)
REALTIME_DIR = PROJECT_ROOT / "data" / "realtime"
VENUE_REALTIME_DIR = REALTIME_DIR / PRIMARY_VENUE if PRIMARY_VENUE else REALTIME_DIR
FEATURES_PATH = VENUE_REALTIME_DIR / f"{SYMBOL}_1m_flow_features.csv"
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "active" / f"{SYMBOL}_live_3m_model.json"
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL

STAGE_TARGET_COLUMNS = {
    1: ["actual_future_return_3"],
    2: [
        "actual_future_return_3",
        "actual_future_range_percent_3",
        "actual_market_pressure_3m",
        "actual_breakout_pressure_3m",
    ],
    3: REGRESSION_TARGET_COLUMNS,
}
TARGET_COLUMNS = STAGE_TARGET_COLUMNS.get(TRAINING_TARGET_STAGE, STAGE_TARGET_COLUMNS[1])


def load_labeled_rows():
    source_path = PREDICTIONS_PATH if PREDICTIONS_PATH.exists() else LABELED_PATH
    if not source_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(source_path)
    if "label_ready" in frame.columns:
        frame["label_ready"] = frame["label_ready"].apply(parse_bool)
    for column in frame.columns:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "actual_path_event_class" not in frame.columns and "actual_path_class" in frame.columns:
        frame["actual_path_event_class"] = frame["actual_path_class"]
    if "actual_future_return_class" not in frame.columns and "actual_future_return_3" in frame.columns:
        threshold = float(os.getenv("RETURN_CLASS_THRESHOLD", "0.0005"))
        future_return = pd.to_numeric(frame["actual_future_return_3"], errors="coerce")
        frame["actual_future_return_class"] = np.where(
            future_return > threshold,
            2,
            np.where(future_return < -threshold, 0, 1),
        )

    required = ["timestamp", CLASS_TARGET] + TARGET_COLUMNS
    frame = frame.dropna(subset=required)
    frame = frame[frame[CLASS_TARGET].isin([0, 1, 2])].copy()
    frame[CLASS_TARGET] = frame[CLASS_TARGET].astype(int)
    return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def load_feature_rows():
    if AUTO_REBUILD_FEATURES:
        env = os.environ.copy()
        env["SYMBOL"] = SYMBOL
        env["OUTPUT_DIR"] = str(REALTIME_DIR)
        if PRIMARY_VENUE:
            env["PRIMARY_VENUE"] = PRIMARY_VENUE
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "build_1m_flow_features.py")],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print("Feature rebuild failed; continuing with any existing feature file.")
            print(result.stdout.strip())
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature file: {FEATURES_PATH}. Run scripts/build_1m_flow_features.py first."
        )
    frame = pd.read_csv(FEATURES_PATH)
    frame = coerce_numeric_columns(frame)
    frame = coerce_feature_ready(frame)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    frame, diagnostics = attach_hierarchical_context(
        frame,
        PROJECT_ROOT,
        SYMBOL,
        layers=("htf", "regime15", "regime30"),
    )
    print_context_diagnostics(diagnostics)
    return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def build_examples(labeled, feature_frame, feature_columns):
    feature_index_by_timestamp = {
        int(timestamp): index
        for index, timestamp in enumerate(feature_frame["timestamp"].to_numpy())
    }
    inputs = []
    classes = []
    regression_targets = []
    sample_weights = []
    spreads = []
    regime_confidences = []
    market_pressures = []
    breakout_pressures = []
    times = []
    timestamps = []

    for _, row in labeled.iterrows():
        timestamp = int(row["timestamp"])
        if timestamp not in feature_index_by_timestamp:
            continue
        end_index = feature_index_by_timestamp[timestamp]
        window, _, _ = build_input_window(
            feature_frame,
            end_index,
            feature_columns,
            LOOKBACK,
        )
        if window is None:
            continue
        inputs.append(window.reshape(-1))
        classes.append(int(row[CLASS_TARGET]))
        regression_targets.append(row[TARGET_COLUMNS].to_numpy(dtype=np.float64))
        future_return = float(row.get("actual_future_return_3", 0.0) or 0.0)
        if SAMPLE_WEIGHT_MODE == "path":
            net_after_costs = max(0.0, abs(future_return) - ROUND_TRIP_COST)
            sample_weight = 1.0 + 4.0 * abs(net_after_costs)
        elif SAMPLE_WEIGHT_MODE == "off":
            sample_weight = 1.0
        else:
            sample_weight = min(5.0, abs(future_return) / max(ROUND_TRIP_COST, 1e-8))
        sample_weights.append(sample_weight)
        current_feature_row = feature_frame.loc[end_index]
        spreads.append(float(current_feature_row.get("spread_percent", 0.0) or 0.0))
        regime_confidence = max(
            abs(float(current_feature_row.get("trend_score", 0.0) or 0.0)) / 4.0,
            float(current_feature_row.get("regime_confidence", 0.0) or 0.0),
        )
        regime_confidences.append(min(max(regime_confidence, 0.0), 1.0))
        market_pressures.append(float(current_feature_row.get("market_pressure", 0.0) or 0.0))
        breakout_pressures.append(float(current_feature_row.get("breakout_pressure_index", 0.0) or 0.0))
        times.append(row.get("time", ""))
        timestamps.append(timestamp)

    return {
        "inputs": np.asarray(inputs, dtype=np.float64),
        "classes": np.asarray(classes, dtype=np.int64),
        "regression_targets": np.asarray(regression_targets, dtype=np.float64),
        "sample_weights": np.asarray(sample_weights, dtype=np.float64),
        "spreads": np.asarray(spreads, dtype=np.float64),
        "regime_confidences": np.asarray(regime_confidences, dtype=np.float64),
        "market_pressures": np.asarray(market_pressures, dtype=np.float64),
        "breakout_pressures": np.asarray(breakout_pressures, dtype=np.float64),
        "times": np.asarray(times, dtype=object),
        "timestamps": np.asarray(timestamps, dtype=np.int64),
    }


def split_train_validation_test(examples):
    total = len(examples["inputs"])
    train_end = int(total * TRAIN_SPLIT)
    validation_end = int(total * (TRAIN_SPLIT + VALIDATION_SPLIT))
    train_end = max(1, min(train_end, total - 2))
    validation_end = max(train_end + 1, min(validation_end, total - 1))
    validation_start = min(total, train_end + EMBARGO_ROWS)
    test_start = min(total, validation_end + EMBARGO_ROWS)
    return (
        {key: value[:train_end] for key, value in examples.items()},
        {key: value[validation_start:validation_end] for key, value in examples.items()},
        {key: value[test_start:] for key, value in examples.items()},
    )


def embargo_counts(total):
    train_end = int(total * TRAIN_SPLIT)
    validation_end = int(total * (TRAIN_SPLIT + VALIDATION_SPLIT))
    train_end = max(1, min(train_end, total - 2))
    validation_end = max(train_end + 1, min(validation_end, total - 1))
    validation_start = min(total, train_end + EMBARGO_ROWS)
    test_start = min(total, validation_end + EMBARGO_ROWS)
    return {
        "train_validation_embargo_rows": max(0, validation_start - train_end),
        "validation_test_embargo_rows": max(0, test_start - validation_end),
    }


def standardize(train_values, validation_values):
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train_values - mean) / std, (validation_values - mean) / std, mean, std


def one_hot(classes):
    encoded = np.zeros((len(classes), 3))
    encoded[np.arange(len(classes)), classes] = 1.0
    return encoded


def train_model(
    x_train,
    y_class_train,
    y_reg_train,
    sample_weight_train,
    x_validation,
    y_class_validation,
    y_reg_validation,
):
    rng = np.random.default_rng(RANDOM_SEED)
    model = initialize_model(x_train.shape[1], HIDDEN_UNITS, y_reg_train.shape[1], rng)
    y_class_one_hot = one_hot(y_class_train)
    class_counts = np.bincount(y_class_train, minlength=3).astype(float)
    class_weights = len(y_class_train) / (3.0 * np.maximum(class_counts, 1.0))

    best_model = {name: value.copy() for name, value in model.items()}
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        for start in range(0, len(x_train), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(x_train))
            xb = x_train[start:end]
            ycb = y_class_train[start:end]
            ycb_one_hot = y_class_one_hot[start:end]
            yrb = y_reg_train[start:end]
            sample_weights = sample_weight_train[start:end][:, None]
            weights = class_weights[ycb][:, None] * sample_weights

            hidden_pre, hidden, probabilities, regression = forward(model, xb)
            d_class = (probabilities - ycb_one_hot) * weights / len(xb)
            d_reg = (
                REGRESSION_LOSS_WEIGHT
                * 2.0
                * (regression - yrb)
                * sample_weights
                / max(1, len(xb) * yrb.shape[1])
            )

            gradients = {
                "w_class": hidden.T @ d_class,
                "b_class": d_class.sum(axis=0),
                "w_reg": hidden.T @ d_reg,
                "b_reg": d_reg.sum(axis=0),
            }
            d_hidden = d_class @ model["w_class"].T + d_reg @ model["w_reg"].T
            d_hidden[hidden_pre <= 0] = 0.0
            gradients["w1"] = xb.T @ d_hidden
            gradients["b1"] = d_hidden.sum(axis=0)

            for name, gradient in gradients.items():
                model[name] -= LEARNING_RATE * gradient

        _, _, validation_probs, validation_reg = forward(model, x_validation)
        class_loss = -np.mean(
            np.log(np.clip(validation_probs[np.arange(len(y_class_validation)), y_class_validation], 1e-8, 1.0))
        )
        reg_loss = np.mean((validation_reg - y_reg_validation) ** 2)
        total_loss = class_loss + REGRESSION_LOSS_WEIGHT * reg_loss
        if total_loss < best_loss:
            best_loss = total_loss
            best_model = {name: value.copy() for name, value in model.items()}

        if epoch == 1 or epoch == EPOCHS or epoch % 10 == 0:
            accuracy = np.mean(np.argmax(validation_probs, axis=1) == y_class_validation)
            print(
                f"epoch {epoch:03d} | validation loss {total_loss:.5f} "
                f"class acc {accuracy:.2%}"
            )

    return best_model


def predict_artifact_on_examples(artifact, inputs):
    x = (inputs - artifact["feature_mean"]) / artifact["feature_std"]
    _, _, probabilities, regression_scaled = forward(artifact["model"], x)
    regression = regression_scaled * artifact["target_std"] + artifact["target_mean"]
    return probabilities, regression


def default_threshold_set():
    return {
        "long_threshold": LONG_THRESHOLD,
        "short_threshold": SHORT_THRESHOLD,
        "max_neutral_threshold": MAX_NEUTRAL_THRESHOLD,
        "min_expected_return": MIN_EXPECTED_RETURN,
        "min_regime_confidence": MIN_REGIME_CONFIDENCE,
        "max_spread": MAX_SPREAD,
    }


def signal_for_row(probability, predicted_return, spread, regime_confidence, threshold_set):
    if spread > threshold_set["max_spread"]:
        return "NONE"
    if regime_confidence < threshold_set["min_regime_confidence"]:
        return "NONE"
    if (
        probability[CLASS_LONG] >= threshold_set["long_threshold"]
        and probability[CLASS_NEUTRAL] <= threshold_set["max_neutral_threshold"]
        and predicted_return >= threshold_set["min_expected_return"]
    ):
        return "LONG"
    if (
        probability[CLASS_SHORT] >= threshold_set["short_threshold"]
        and probability[CLASS_NEUTRAL] <= threshold_set["max_neutral_threshold"]
        and predicted_return <= -threshold_set["min_expected_return"]
    ):
        return "SHORT"
    return "NONE"


def metrics_from_trades(trades, actual_classes):
    trades = pd.DataFrame(trades)
    if len(trades) == 0:
        return {
            "trade_count": 0,
            "long_trades": 0,
            "short_trades": 0,
            "average_net_return": -1.0,
            "average_gross_return": 0.0,
            "compounded_return": 0.0,
            "max_drawdown": 0.0,
            "winner_avg_probability": 0.0,
            "loser_avg_probability": 0.0,
            "neutral_trade_ratio": 1.0,
            "win_rate": 0.0,
            "top5_avg_net_return": -1.0,
            "top5_win_rate": 0.0,
            "long_short_trade_ratio": float("inf"),
        }

    winners = trades[trades["net_return"] > 0]
    losers = trades[trades["net_return"] <= 0]
    compounded, drawdown = compounded_return_and_drawdown(trades["net_return"])
    top_count = max(1, int(len(trades) * 0.05))
    top = trades.sort_values("probability", ascending=False).head(top_count)
    long_count = int((trades["signal"] == "LONG").sum())
    short_count = int((trades["signal"] == "SHORT").sum())
    return {
        "trade_count": int(len(trades)),
        "long_trades": long_count,
        "short_trades": short_count,
        "average_net_return": float(trades["net_return"].mean()),
        "average_gross_return": float(trades["gross_return"].mean()),
        "compounded_return": float(compounded),
        "max_drawdown": float(drawdown),
        "winner_avg_probability": float(winners["probability"].mean()) if len(winners) else 0.0,
        "loser_avg_probability": float(losers["probability"].mean()) if len(losers) else 0.0,
        "neutral_trade_ratio": float((trades["actual_class"] == CLASS_NEUTRAL).mean()),
        "win_rate": float((trades["net_return"] > 0).mean()),
        "top5_avg_net_return": float(top["net_return"].mean()) if len(top) else -1.0,
        "top5_win_rate": float((top["net_return"] > 0).mean()) if len(top) else 0.0,
        "long_short_trade_ratio": float(max(long_count, short_count) / max(1, min(long_count, short_count))),
    }


def backtest_probabilities(
    probabilities,
    actual_classes,
    future_returns,
    predicted_returns=None,
    spreads=None,
    regime_confidences=None,
    threshold_set=None,
):
    threshold_set = threshold_set or default_threshold_set()
    predicted_returns = (
        np.asarray(predicted_returns, dtype=np.float64)
        if predicted_returns is not None
        else np.zeros(len(future_returns), dtype=np.float64)
    )
    spreads = (
        np.asarray(spreads, dtype=np.float64)
        if spreads is not None
        else np.zeros(len(future_returns), dtype=np.float64)
    )
    regime_confidences = (
        np.asarray(regime_confidences, dtype=np.float64)
        if regime_confidences is not None
        else np.zeros(len(future_returns), dtype=np.float64)
    )
    trades = []
    for probability, actual_class, future_return, predicted_return, spread, regime_confidence in zip(
        probabilities,
        actual_classes,
        future_returns,
        predicted_returns,
        spreads,
        regime_confidences,
    ):
        signal = signal_for_row(
            probability,
            predicted_return,
            spread,
            regime_confidence,
            threshold_set,
        )
        if signal == "NONE":
            continue
        gross = future_return if signal == "LONG" else -future_return
        net = gross - ROUND_TRIP_COST
        trade_probability = probability[CLASS_LONG] if signal == "LONG" else probability[CLASS_SHORT]
        trades.append(
            {
                "signal": signal,
                "actual_class": int(actual_class),
                "gross_return": gross,
                "net_return": net,
                "probability": trade_probability,
            }
        )
    return metrics_from_trades(trades, actual_classes)


def threshold_score(metrics):
    if metrics["trade_count"] == 0:
        return -1e9
    return (
        metrics["average_net_return"]
        + 0.25 * metrics["win_rate"]
        + 0.10 * metrics["top5_avg_net_return"]
        + 0.05 * metrics["max_drawdown"]
    )


def calibrate_thresholds(probabilities, regression, validation):
    future_return_index = TARGET_COLUMNS.index("actual_future_return_3")
    predicted_returns = regression[:, future_return_index]
    actual_returns = validation["regression_targets"][:, future_return_index]
    spread_values = validation["spreads"]
    finite_spreads = spread_values[np.isfinite(spread_values)]
    spread_grid = [MAX_SPREAD]
    if len(finite_spreads):
        spread_grid.extend(
            sorted(set(float(np.quantile(finite_spreads, q)) for q in [0.50, 0.75, 0.90, 0.98]))
        )
    regime_grid = [0.0]
    if np.nanmax(validation["regime_confidences"]) > 0:
        regime_grid.extend([0.25, 0.50, 0.75])
    min_trade_count = max(5, int(len(actual_returns) * 0.03))
    best = None
    rows = []
    for long_threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for short_threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            for max_neutral in [0.25, 0.35, 0.45, 0.55, 0.70]:
                for min_expected in [0.0, ROUND_TRIP_COST * 0.25, ROUND_TRIP_COST * 0.50, ROUND_TRIP_COST]:
                    for min_regime in regime_grid:
                        for max_spread in spread_grid:
                            threshold_set = {
                                "long_threshold": long_threshold,
                                "short_threshold": short_threshold,
                                "max_neutral_threshold": max_neutral,
                                "min_expected_return": min_expected,
                                "min_regime_confidence": min_regime,
                                "max_spread": max_spread,
                            }
                            metrics = backtest_probabilities(
                                probabilities,
                                validation["classes"],
                                actual_returns,
                                predicted_returns,
                                spread_values,
                                validation["regime_confidences"],
                                threshold_set,
                            )
                            score = threshold_score(metrics)
                            if metrics["trade_count"] < min_trade_count:
                                score -= 10.0
                            row = {**threshold_set, **metrics, "score": score}
                            rows.append(row)
                            if best is None or score > best["score"]:
                                best = row
    if best is None:
        best = {**default_threshold_set(), "score": -1e9, "trade_count": 0}
    return {
        key: best[key]
        for key in [
            "long_threshold",
            "short_threshold",
            "max_neutral_threshold",
            "min_expected_return",
            "min_regime_confidence",
            "max_spread",
        ]
    }, pd.DataFrame(rows).sort_values("score", ascending=False)


def baseline_metrics(validation):
    future_return_index = TARGET_COLUMNS.index("actual_future_return_3")
    actual_returns = validation["regression_targets"][:, future_return_index]
    baselines = {}

    def directional_metrics(name, signals):
        trades = []
        for signal, actual_class, future_return in zip(signals, validation["classes"], actual_returns):
            if signal == "NONE":
                continue
            gross = future_return if signal == "LONG" else -future_return
            trades.append(
                {
                    "signal": signal,
                    "actual_class": int(actual_class),
                    "gross_return": gross,
                    "net_return": gross - ROUND_TRIP_COST,
                    "probability": 1.0,
                }
            )
        baselines[name] = metrics_from_trades(trades, validation["classes"])

    directional_metrics("always_neutral", ["NONE"] * len(actual_returns))
    directional_metrics("always_long", ["LONG"] * len(actual_returns))
    directional_metrics("always_short", ["SHORT"] * len(actual_returns))
    directional_metrics(
        "follow_market_pressure_sign",
        [
            "LONG" if value > 0 else "SHORT" if value < 0 else "NONE"
            for value in validation["market_pressures"]
        ],
    )
    directional_metrics(
        "follow_breakout_pressure_sign",
        [
            "LONG" if value > 0 else "SHORT" if value < 0 else "NONE"
            for value in validation["breakout_pressures"]
        ],
    )
    # These are intentionally explicit placeholders until their input streams
    # are stable enough to be used without causing as-of/staleness row loss.
    baselines["follow_old_5m_model"] = {"available": False, "reason": "not attached to live 3m examples"}
    baselines["ensemble_agreement_only"] = {"available": False, "reason": "ensemble should wait until base models are stable"}
    return baselines


def best_available_baseline(baselines):
    available = {
        name: metrics
        for name, metrics in baselines.items()
        if metrics.get("available", True) is not False and "average_net_return" in metrics
    }
    if not available:
        return "none", {"average_net_return": -1.0, "max_drawdown": 0.0, "trade_count": 0}
    return max(available.items(), key=lambda item: item[1]["average_net_return"])


def evaluate_candidate(artifact, evaluation_rows, threshold_set=None):
    probabilities, regression = predict_artifact_on_examples(artifact, evaluation_rows["inputs"])
    target_columns = artifact.get("regression_target_columns", TARGET_COLUMNS)
    future_return_index = target_columns.index("actual_future_return_3")
    metrics = backtest_probabilities(
        probabilities,
        evaluation_rows["classes"],
        evaluation_rows["regression_targets"][:, future_return_index],
        regression[:, future_return_index],
        evaluation_rows["spreads"],
        evaluation_rows["regime_confidences"],
        threshold_set,
    )
    metrics["class_accuracy"] = float(np.mean(np.argmax(probabilities, axis=1) == evaluation_rows["classes"]))
    class_counts = np.bincount(evaluation_rows["classes"], minlength=3)
    metrics["majority_class_accuracy"] = float(class_counts.max() / max(1, len(evaluation_rows["classes"])))
    metrics["class_accuracy_edge"] = metrics["class_accuracy"] - metrics["majority_class_accuracy"]
    metrics["return_mae"] = float(
        np.mean(np.abs(regression[:, future_return_index] - evaluation_rows["regression_targets"][:, future_return_index]))
    )
    return metrics


def confusion_matrix(actual, predicted):
    matrix = np.zeros((3, 3), dtype=int)
    for actual_class, predicted_class in zip(actual, predicted):
        matrix[int(actual_class), int(predicted_class)] += 1
    return matrix


def print_confusion_matrix(matrix):
    print("confusion matrix rows=actual cols=predicted [short, neutral, long]")
    print("      pred_short pred_neutral pred_long")
    for class_id, label in enumerate(["actual_short", "actual_neutral", "actual_long"]):
        print(
            f"{label:14s} "
            f"{matrix[class_id, 0]:10d} {matrix[class_id, 1]:12d} {matrix[class_id, 2]:9d}"
        )


def print_per_class_precision_recall(actual, predicted):
    print("per-class precision/recall")
    for class_id in [CLASS_SHORT, CLASS_NEUTRAL, CLASS_LONG]:
        tp = int(((actual == class_id) & (predicted == class_id)).sum())
        fp = int(((actual != class_id) & (predicted == class_id)).sum())
        fn = int(((actual == class_id) & (predicted != class_id)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        print(f"- {CLASS_NAMES[class_id]}: precision={precision:.2%}, recall={recall:.2%}, tp={tp}, fp={fp}, fn={fn}")


def print_probability_calibration_buckets(probabilities, actual, bucket_count=10):
    predicted = np.argmax(probabilities, axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == actual
    bins = np.linspace(0.0, 1.0, bucket_count + 1)
    print("probability calibration buckets")
    print("bucket | rows | avg confidence | accuracy")
    for start, end in zip(bins[:-1], bins[1:]):
        if end >= 1.0:
            mask = (confidence >= start) & (confidence <= end)
        else:
            mask = (confidence >= start) & (confidence < end)
        rows = int(mask.sum())
        if rows == 0:
            print(f"{start:.1f}-{end:.1f} | 0 | n/a | n/a")
            continue
        print(
            f"{start:.1f}-{end:.1f} | {rows} | "
            f"{float(confidence[mask].mean()):.2%} | {float(correct[mask].mean()):.2%}"
        )


def print_winner_loser_probability(probabilities, actual):
    predicted = np.argmax(probabilities, axis=1)
    confidence = probabilities.max(axis=1)
    winners = confidence[predicted == actual]
    losers = confidence[predicted != actual]
    winner_avg = float(winners.mean()) if len(winners) else 0.0
    loser_avg = float(losers.mean()) if len(losers) else 0.0
    print(f"winner average probability: {winner_avg:.2%}")
    print(f"loser average probability: {loser_avg:.2%}")


def print_model_diagnostics(title, artifact, rows):
    probabilities, regression = predict_artifact_on_examples(artifact, rows["inputs"])
    predicted = np.argmax(probabilities, axis=1)
    actual = rows["classes"]
    future_return_index = artifact.get("regression_target_columns", TARGET_COLUMNS).index("actual_future_return_3")
    class_counts = np.bincount(actual, minlength=3)
    majority_accuracy = float(class_counts.max() / max(1, len(actual)))
    class_accuracy = float((predicted == actual).mean()) if len(actual) else 0.0
    return_mae = float(np.mean(np.abs(regression[:, future_return_index] - rows["regression_targets"][:, future_return_index]))) if len(actual) else 0.0

    print(f"\nA. Model diagnostics - {title}")
    print("class distribution")
    print_class_distribution(actual)
    print(f"majority baseline accuracy: {majority_accuracy:.2%}")
    print(f"class accuracy: {class_accuracy:.2%}")
    print_confusion_matrix(confusion_matrix(actual, predicted))
    print_per_class_precision_recall(actual, predicted)
    print(f"return MAE: {return_mae:.6g}")
    print_probability_calibration_buckets(probabilities, actual)
    print_winner_loser_probability(probabilities, actual)


def print_cost_profile():
    print("cost profile used")
    print(f"- COST_PROFILE: {COST_PROFILE}")
    print(f"- TAKER_ROUND_TRIP_COST: {TAKER_ROUND_TRIP_COST:.4%}")
    print(f"- MAKER_ROUND_TRIP_COST: {MAKER_ROUND_TRIP_COST:.4%}")
    print(f"- ESTIMATED_SLIPPAGE: {ESTIMATED_SLIPPAGE:.4%}")
    print(f"- SPREAD_COST: {SPREAD_COST:.4%}")
    print(f"- ROUND_TRIP_COST: {ROUND_TRIP_COST:.4%}")


def print_economic_diagnostics(title, metrics, threshold_set, baselines, best_baseline_name, best_baseline_metrics):
    print(f"\nB. Economic diagnostics - {title}")
    print("threshold set")
    for key, value in threshold_set.items():
        print(f"- {key}: {value:.6g}" if isinstance(value, float) else f"- {key}: {value}")
    print_cost_profile()
    print(f"trade count: {metrics['trade_count']}")
    print(f"gross return avg: {metrics['average_gross_return']:.6g}")
    print(f"net return avg: {metrics['average_net_return']:.6g}")
    print(f"compounded return: {metrics['compounded_return']:.6g}")
    print(f"drawdown: {metrics['max_drawdown']:.6g}")
    print(f"win rate after costs: {metrics['win_rate']:.2%}")
    print(f"top 5% avg net return: {metrics['top5_avg_net_return']:.6g}")
    print(f"top 5% win-after-cost: {metrics['top5_win_rate']:.2%}")
    print("baseline comparisons")
    for name, baseline in baselines.items():
        if baseline.get("available", True) is False:
            print(f"- {name}: unavailable ({baseline.get('reason', 'not available')})")
            continue
        print(
            f"- {name}: trades={baseline['trade_count']}, "
            f"avg_net={baseline['average_net_return']:.6g}, "
            f"drawdown={baseline['max_drawdown']:.6g}, "
            f"win_after_cost={baseline['win_rate']:.2%}"
        )
    print(
        f"best baseline: {best_baseline_name} "
        f"avg_net={best_baseline_metrics['average_net_return']:.6g}, "
        f"drawdown={best_baseline_metrics['max_drawdown']:.6g}, "
        f"trades={best_baseline_metrics['trade_count']}"
    )


def evaluate_active_on_validation(active, validation, feature_frame):
    if active is None:
        return None
    try:
        # Active and candidate may have different feature columns. Rebuild
        # validation windows using the active model's own schema.
        labeled = pd.DataFrame(
            {
                "timestamp": validation["timestamps"],
                "time": validation["times"],
                CLASS_TARGET: validation["classes"],
            }
        )
        active_target_columns = active.get("regression_target_columns", REGRESSION_TARGET_COLUMNS)
        if active.get("class_target_column", CLASS_TARGET) != CLASS_TARGET:
            return None
        if any(column not in TARGET_COLUMNS for column in active_target_columns):
            return None
        for index, column in enumerate(TARGET_COLUMNS):
            labeled[column] = validation["regression_targets"][:, index]
        examples = build_examples(labeled, feature_frame, active["feature_columns"])
        if len(examples["inputs"]) < MIN_VALIDATION_ROWS:
            return None
        return evaluate_candidate(active, examples)
    except Exception as error:
        print(f"Could not evaluate active model on current validation rows: {error}")
        return None


def print_class_distribution(classes):
    counts = pd.Series(classes).value_counts().sort_index()
    total = len(classes) or 1
    for class_id in [0, 1, 2]:
        count = int(counts.get(class_id, 0))
        print(f"- class {class_id} {CLASS_NAMES[class_id]}: {count} ({count / total:.2%})")


def feature_schema_hash(feature_columns):
    payload = json.dumps(feature_columns, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def model_id_for(timestamp, schema_hash):
    return f"{SYMBOL}_live_3m_{timestamp}_{schema_hash}"


def subset_examples(examples, start, end):
    return {key: value[start:end] for key, value in examples.items()}


def walk_forward_validation(examples, feature_columns):
    if not RUN_WALK_FORWARD:
        return pd.DataFrame()

    total = len(examples["inputs"])
    edges = np.linspace(0, total, 5, dtype=int)
    periods = [subset_examples(examples, int(edges[index]), int(edges[index + 1])) for index in range(4)]
    period_lengths = [len(period["inputs"]) for period in periods]

    if min(period_lengths) < WALK_FORWARD_MIN_PERIOD_ROWS:
        print("\nWalk-forward validation skipped:")
        print(
            "- each of the four chronological periods needs at least "
            f"{WALK_FORWARD_MIN_PERIOD_ROWS} usable rows"
        )
        print(f"- period row counts: {period_lengths}")
        return pd.DataFrame()

    fold_specs = [
        ("A_to_B_to_C", 0, 1, 2),
        ("B_to_C_to_D", 1, 2, 3),
    ]
    rows = []
    schema_hash = feature_schema_hash(feature_columns)

    print("\nWalk-forward validation")
    print("Rule: train one older period, calibrate on the next, then test on the next unseen period.")
    for fold_name, train_index, validation_index, test_index in fold_specs:
        fold_train = periods[train_index]
        fold_validation = periods[validation_index]
        fold_test = periods[test_index]

        x_train, x_validation, feature_mean, feature_std = standardize(
            fold_train["inputs"],
            fold_validation["inputs"],
        )
        y_reg_train, y_reg_validation, target_mean, target_std = standardize(
            fold_train["regression_targets"],
            fold_validation["regression_targets"],
        )
        model = train_model(
            x_train,
            fold_train["classes"],
            y_reg_train,
            fold_train["sample_weights"],
            x_validation,
            fold_validation["classes"],
            y_reg_validation,
        )
        artifact = {
            "model_type": "live_3m_numpy_mlp_walk_forward",
            "symbol": SYMBOL,
            "model_id": f"{SYMBOL}_walk_forward_{fold_name}_{schema_hash}",
            "lookback": LOOKBACK,
            "feature_columns": feature_columns,
            "feature_schema_hash": schema_hash,
            "regression_target_columns": TARGET_COLUMNS,
            "class_target_column": CLASS_TARGET,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "model": model,
        }
        validation_probabilities, validation_regression = predict_artifact_on_examples(
            artifact,
            fold_validation["inputs"],
        )
        thresholds, _ = calibrate_thresholds(
            validation_probabilities,
            validation_regression,
            fold_validation,
        )
        validation_metrics = evaluate_candidate(artifact, fold_validation, thresholds)
        test_metrics = evaluate_candidate(artifact, fold_test, thresholds)
        validation_baselines = baseline_metrics(fold_validation)
        test_baselines = baseline_metrics(fold_test)
        validation_baseline_name, validation_baseline_metrics = best_available_baseline(validation_baselines)
        test_baseline_name, test_baseline_metrics = best_available_baseline(test_baselines)
        passed = (
            test_metrics["trade_count"] >= MIN_PROMOTION_TRADES
            and test_metrics["average_net_return"] > test_baseline_metrics["average_net_return"]
            and test_metrics["average_net_return"] > MIN_AVERAGE_NET_RETURN
            and test_metrics["max_drawdown"] >= MAX_PROMOTION_DRAWDOWN
            and test_metrics["top5_avg_net_return"] >= MIN_TOP5_NET_RETURN
            and test_metrics["top5_win_rate"] >= MIN_TOP5_WIN_AFTER_COST
        )
        rows.append(
            {
                "fold": fold_name,
                "train_rows": len(fold_train["inputs"]),
                "validation_rows": len(fold_validation["inputs"]),
                "test_rows": len(fold_test["inputs"]),
                "train_start_timestamp": int(fold_train["timestamps"].min()),
                "train_end_timestamp": int(fold_train["timestamps"].max()),
                "validation_start_timestamp": int(fold_validation["timestamps"].min()),
                "validation_end_timestamp": int(fold_validation["timestamps"].max()),
                "test_start_timestamp": int(fold_test["timestamps"].min()),
                "test_end_timestamp": int(fold_test["timestamps"].max()),
                "validation_baseline": validation_baseline_name,
                "test_baseline": test_baseline_name,
                "validation_average_net_return": validation_metrics["average_net_return"],
                "test_average_net_return": test_metrics["average_net_return"],
                "test_baseline_average_net_return": test_baseline_metrics["average_net_return"],
                "test_trade_count": test_metrics["trade_count"],
                "test_win_rate": test_metrics["win_rate"],
                "test_max_drawdown": test_metrics["max_drawdown"],
                "test_top5_avg_net_return": test_metrics["top5_avg_net_return"],
                "test_top5_win_rate": test_metrics["top5_win_rate"],
                "passed": bool(passed),
            }
        )
        print(
            f"- {fold_name}: test avg net {test_metrics['average_net_return']:.6g}, "
            f"baseline {test_baseline_name} {test_baseline_metrics['average_net_return']:.6g}, "
            f"trades {test_metrics['trade_count']}, passed={passed}"
        )

    return pd.DataFrame(rows)


def should_promote(
    validation_metrics,
    test_metrics,
    validation_baseline_name,
    validation_baseline_metrics,
    test_baseline_name,
    test_baseline_metrics,
    active_metrics,
    validation_count,
    test_count,
    walk_forward_results=None,
):
    reasons = []
    if validation_count < MIN_VALIDATION_ROWS:
        reasons.append(
            f"validation row count {validation_count} < MIN_VALIDATION_ROWS {MIN_VALIDATION_ROWS}"
        )
    if validation_count < MIN_PROMOTION_VALIDATION_ROWS:
        reasons.append(
            f"validation row count {validation_count} < MIN_PROMOTION_VALIDATION_ROWS {MIN_PROMOTION_VALIDATION_ROWS}"
        )
    if test_count < MIN_VALIDATION_ROWS:
        reasons.append(f"test row count {test_count} < MIN_VALIDATION_ROWS {MIN_VALIDATION_ROWS}")
    if validation_metrics["trade_count"] == 0:
        reasons.append("candidate produced zero validation trades")
    if validation_metrics["trade_count"] < MIN_PROMOTION_TRADES:
        reasons.append(
            f"candidate validation trade count {validation_metrics['trade_count']} < MIN_PROMOTION_TRADES {MIN_PROMOTION_TRADES}"
        )
    if test_metrics["trade_count"] < MIN_PROMOTION_TRADES:
        reasons.append(
            f"candidate test trade count {test_metrics['trade_count']} < MIN_PROMOTION_TRADES {MIN_PROMOTION_TRADES}"
        )
    if validation_metrics["average_net_return"] <= MIN_AVERAGE_NET_RETURN:
        reasons.append(
            f"validation average net return {validation_metrics['average_net_return']:.6g} <= MIN_AVERAGE_NET_RETURN {MIN_AVERAGE_NET_RETURN:.6g}"
        )
    if test_metrics["average_net_return"] <= MIN_AVERAGE_NET_RETURN:
        reasons.append(
            f"test average net return {test_metrics['average_net_return']:.6g} <= MIN_AVERAGE_NET_RETURN {MIN_AVERAGE_NET_RETURN:.6g}"
        )
    if validation_metrics["average_net_return"] <= validation_baseline_metrics["average_net_return"]:
        reasons.append(
            f"validation net return did not beat best baseline {validation_baseline_name}"
        )
    if test_metrics["average_net_return"] <= test_baseline_metrics["average_net_return"]:
        reasons.append(f"test net return did not beat best baseline {test_baseline_name}")
    if validation_metrics["max_drawdown"] < MAX_PROMOTION_DRAWDOWN:
        reasons.append(
            f"validation max drawdown {validation_metrics['max_drawdown']:.2%} < MAX_PROMOTION_DRAWDOWN {MAX_PROMOTION_DRAWDOWN:.2%}"
        )
    if test_metrics["max_drawdown"] < MAX_PROMOTION_DRAWDOWN:
        reasons.append(
            f"test max drawdown {test_metrics['max_drawdown']:.2%} < MAX_PROMOTION_DRAWDOWN {MAX_PROMOTION_DRAWDOWN:.2%}"
        )
    if validation_metrics["top5_avg_net_return"] < MIN_TOP5_NET_RETURN:
        reasons.append("validation top 5% average net return is below gate")
    if test_metrics["top5_avg_net_return"] < MIN_TOP5_NET_RETURN:
        reasons.append("test top 5% average net return is below gate")
    if validation_metrics["top5_win_rate"] < MIN_TOP5_WIN_AFTER_COST:
        reasons.append("validation top 5% win-after-cost is below gate")
    if test_metrics["top5_win_rate"] < MIN_TOP5_WIN_AFTER_COST:
        reasons.append("test top 5% win-after-cost is below gate")
    if validation_metrics["class_accuracy_edge"] < MIN_CLASS_ACCURACY_EDGE:
        reasons.append(
            f"validation class accuracy edge {validation_metrics['class_accuracy_edge']:.2%} < MIN_CLASS_ACCURACY_EDGE {MIN_CLASS_ACCURACY_EDGE:.2%}"
        )
    if test_metrics["class_accuracy_edge"] < MIN_CLASS_ACCURACY_EDGE:
        reasons.append(
            f"test class accuracy edge {test_metrics['class_accuracy_edge']:.2%} < MIN_CLASS_ACCURACY_EDGE {MIN_CLASS_ACCURACY_EDGE:.2%}"
        )
    if validation_metrics["winner_avg_probability"] <= validation_metrics["loser_avg_probability"]:
        reasons.append("winner average probability is not greater than loser average probability")
    if (
        validation_metrics["winner_avg_probability"]
        - validation_metrics["loser_avg_probability"]
        < MIN_WINNER_PROBABILITY_EDGE
    ):
        reasons.append(
            "winner average probability edge is below "
            f"MIN_WINNER_PROBABILITY_EDGE {MIN_WINNER_PROBABILITY_EDGE:.2%}"
        )
    if validation_metrics["neutral_trade_ratio"] > MAX_NEUTRAL_TRADE_RATIO:
        reasons.append("validation directional trades are mostly actual neutral rows")
    if test_metrics["neutral_trade_ratio"] > MAX_NEUTRAL_TRADE_RATIO:
        reasons.append("test directional trades are mostly actual neutral rows")
    if validation_metrics["long_short_trade_ratio"] > MAX_LONG_SHORT_TRADE_RATIO:
        reasons.append("validation long/short performance is too asymmetric")
    if test_metrics["long_short_trade_ratio"] > MAX_LONG_SHORT_TRADE_RATIO:
        reasons.append("test long/short performance is too asymmetric")

    if active_metrics is not None:
        if test_metrics["average_net_return"] <= active_metrics["average_net_return"]:
            reasons.append("average net return did not improve versus active model")
        allowed_drawdown = active_metrics["max_drawdown"] - MAX_DRAWDOWN_WORSE_ALLOWANCE
        if test_metrics["max_drawdown"] < allowed_drawdown:
            reasons.append("max drawdown is worse than active model allowance")
    else:
        if test_metrics["average_net_return"] <= 0:
            reasons.append("no active model exists and candidate average net return is not positive")

    if RUN_WALK_FORWARD:
        if walk_forward_results is None or len(walk_forward_results) == 0:
            reasons.append("walk-forward validation was unavailable or skipped")
        else:
            completed = walk_forward_results.copy()
            passed_values = completed["passed"].astype(bool).tolist()
            if len(passed_values) < MIN_WALK_FORWARD_PERIODS:
                reasons.append(
                    f"walk-forward completed {len(passed_values)} periods < MIN_WALK_FORWARD_PERIODS {MIN_WALK_FORWARD_PERIODS}"
                )
            elif not all(passed_values[-MIN_WALK_FORWARD_PERIODS:]):
                reasons.append(
                    f"candidate did not beat baseline on the latest {MIN_WALK_FORWARD_PERIODS} walk-forward periods"
                )

    return len(reasons) == 0, reasons


def main():
    labeled = load_labeled_rows()
    print("Live 3m online trainer")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Labeled row path: {LABELED_PATH}")
    print(f"Live prediction source path: {PREDICTIONS_PATH}")
    print(f"Feature path: {FEATURES_PATH}")
    print(f"Total labeled rows: {len(labeled)}")
    print(f"TRAINING_TARGET_STAGE: {TRAINING_TARGET_STAGE}")
    print(f"CLASS_TARGET: {CLASS_TARGET}")
    print(f"Regression targets: {', '.join(TARGET_COLUMNS)}")
    print(f"SAMPLE_WEIGHT_MODE: {SAMPLE_WEIGHT_MODE}")
    print(f"PROMOTE_BEST: {PROMOTE_BEST}")
    print(f"AUTO_REBUILD_FEATURES: {AUTO_REBUILD_FEATURES}")
    print(f"MIN_LABELED_ROWS: {MIN_LABELED_ROWS}")
    print(f"EMBARGO_ROWS: {EMBARGO_ROWS}")
    print_cost_profile()
    if len(labeled) < MIN_LABELED_ROWS:
        print("Training skipped: not enough labeled rows yet.")
        print("Candidate promoted: no")
        print("No trades were placed.")
        return

    feature_frame = load_feature_rows()
    feature_columns = choose_feature_columns(feature_frame)
    examples = build_examples(labeled, feature_frame, feature_columns)
    if len(examples["inputs"]) < MIN_LABELED_ROWS:
        print(
            "Training skipped: not enough labeled rows also have valid contiguous feature windows."
        )
        print(f"Usable examples: {len(examples['inputs'])}")
        print("Candidate promoted: no")
        print("No trades were placed.")
        return

    train, validation, test = split_train_validation_test(examples)
    embargo = embargo_counts(len(examples["inputs"]))
    print(f"Usable examples: {len(examples['inputs'])}")
    print(f"Train rows: {len(train['inputs'])}")
    print(f"Validation rows: {len(validation['inputs'])}")
    print(f"Test rows: {len(test['inputs'])}")
    print(f"Dropped train/validation embargo rows: {embargo['train_validation_embargo_rows']}")
    print(f"Dropped validation/test embargo rows: {embargo['validation_test_embargo_rows']}")
    print(f"Feature count per row: {len(feature_columns)}")
    print(f"Flattened input size: {train['inputs'].shape[1]}")
    print(
        "Sample weight range: "
        f"{examples['sample_weights'].min():.4g} to {examples['sample_weights'].max():.4g}"
    )
    print("Class distribution of labeled rows:")
    print_class_distribution(examples["classes"])
    if len(train["inputs"]) == 0 or len(validation["inputs"]) == 0 or len(test["inputs"]) == 0:
        print("Training skipped: train/validation/test split is empty after embargo.")
        print("Candidate promoted: no")
        print("No trades were placed.")
        return

    x_train, x_validation, feature_mean, feature_std = standardize(
        train["inputs"],
        validation["inputs"],
    )
    x_test = (test["inputs"] - feature_mean) / feature_std
    y_reg_train, y_reg_validation, target_mean, target_std = standardize(
        train["regression_targets"],
        validation["regression_targets"],
    )

    model = train_model(
        x_train,
        train["classes"],
        y_reg_train,
        train["sample_weights"],
        x_validation,
        validation["classes"],
        y_reg_validation,
    )

    validation_for_eval = dict(validation)
    test_for_eval = dict(test)

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    schema_hash = feature_schema_hash(feature_columns)
    current_model_id = model_id_for(timestamp, schema_hash)
    candidate_artifact = {
        "model_type": "live_3m_numpy_mlp",
        "symbol": SYMBOL,
        "created_at": (
            dt.datetime.now(dt.UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        "model_id": current_model_id,
        "model_trained_until_timestamp": int(train["timestamps"].max()) if len(train["timestamps"]) else None,
        "lookback": LOOKBACK,
        "feature_columns": feature_columns,
        "feature_schema_hash": schema_hash,
        "regression_target_columns": TARGET_COLUMNS,
        "class_target_column": CLASS_TARGET,
        "training_target_stage": TRAINING_TARGET_STAGE,
        "sample_weight_mode": SAMPLE_WEIGHT_MODE,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "model": model,
        "class_names": CLASS_NAMES,
        "training_rows": int(len(train["inputs"])),
        "training_row_count": int(len(train["inputs"])),
        "validation_rows": int(len(validation["inputs"])),
        "test_rows": int(len(test["inputs"])),
        "cost_profile": {
            "profile": COST_PROFILE,
            "taker_round_trip_cost": TAKER_ROUND_TRIP_COST,
            "maker_round_trip_cost": MAKER_ROUND_TRIP_COST,
            "estimated_slippage": ESTIMATED_SLIPPAGE,
            "spread_cost": SPREAD_COST,
            "effective_round_trip_cost": ROUND_TRIP_COST,
        },
    }

    validation_probabilities, validation_regression = predict_artifact_on_examples(
        candidate_artifact,
        validation_for_eval["inputs"],
    )
    calibrated_thresholds, threshold_sweep = calibrate_thresholds(
        validation_probabilities,
        validation_regression,
        validation_for_eval,
    )
    candidate_artifact["calibrated_thresholds"] = calibrated_thresholds

    validation_metrics = evaluate_candidate(candidate_artifact, validation_for_eval, calibrated_thresholds)
    test_metrics = evaluate_candidate(candidate_artifact, test_for_eval, calibrated_thresholds)
    validation_baselines = baseline_metrics(validation_for_eval)
    test_baselines = baseline_metrics(test_for_eval)
    validation_baseline_name, validation_baseline_metrics = best_available_baseline(validation_baselines)
    test_baseline_name, test_baseline_metrics = best_available_baseline(test_baselines)
    active = load_model(ACTIVE_MODEL_PATH)
    active_metrics = evaluate_active_on_validation(active, test_for_eval, feature_frame)
    walk_forward_results = walk_forward_validation(examples, feature_columns)
    candidate_artifact["validation_score"] = float(threshold_score(validation_metrics))
    candidate_artifact["validation_average_net_return"] = float(
        validation_metrics["average_net_return"]
    )
    candidate_artifact["test_average_net_return"] = float(test_metrics["average_net_return"])
    candidate_artifact["best_validation_baseline"] = validation_baseline_name
    candidate_artifact["best_test_baseline"] = test_baseline_name
    promote, rejection_reasons = should_promote(
        validation_metrics,
        test_metrics,
        validation_baseline_name,
        validation_baseline_metrics,
        test_baseline_name,
        test_baseline_metrics,
        active_metrics,
        len(validation["inputs"]),
        len(test["inputs"]),
        walk_forward_results,
    )

    candidate_dir = CANDIDATE_ROOT / timestamp
    candidate_model_path = candidate_dir / "model.json"
    save_model(candidate_model_path, candidate_artifact)

    metrics_frame = pd.DataFrame(
        [
            {"model": "candidate_validation", **validation_metrics},
            {"model": "candidate_test", **test_metrics},
            {"model": f"validation_baseline_{validation_baseline_name}", **validation_baseline_metrics},
            {"model": f"test_baseline_{test_baseline_name}", **test_baseline_metrics},
            *([{"model": "active", **active_metrics}] if active_metrics else []),
        ]
    )
    atomic_write_csv(metrics_frame, candidate_dir / "validation_metrics.csv")
    atomic_write_csv(threshold_sweep.head(250), candidate_dir / "threshold_sweep_top.csv")
    atomic_write_csv(
        pd.DataFrame([{**calibrated_thresholds, "model_id": current_model_id}]),
        candidate_dir / "calibrated_thresholds.csv",
    )
    baseline_rows = []
    for split_name, baselines in [
        ("validation", validation_baselines),
        ("test", test_baselines),
    ]:
        for baseline_name, metrics in baselines.items():
            baseline_rows.append({"split": split_name, "baseline": baseline_name, **metrics})
    atomic_write_csv(pd.DataFrame(baseline_rows), candidate_dir / "baseline_metrics.csv")
    if len(walk_forward_results) > 0:
        atomic_write_csv(walk_forward_results, candidate_dir / "walk_forward_metrics.csv")

    print_model_diagnostics("validation", candidate_artifact, validation_for_eval)
    print_model_diagnostics("newest holdout/test", candidate_artifact, test_for_eval)
    print_economic_diagnostics(
        "validation",
        validation_metrics,
        calibrated_thresholds,
        validation_baselines,
        validation_baseline_name,
        validation_baseline_metrics,
    )
    print_economic_diagnostics(
        "newest holdout/test",
        test_metrics,
        calibrated_thresholds,
        test_baselines,
        test_baseline_name,
        test_baseline_metrics,
    )
    if active_metrics:
        print("\nActive model economic comparison on newest holdout/test rows:")
        for key, value in active_metrics.items():
            print(f"- {key}: {value:.6g}" if isinstance(value, float) else f"- {key}: {value}")
    else:
        print("\nNo active model was available for comparison.")

    if PROMOTE_BEST and promote:
        ACTIVE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_model_path, ACTIVE_MODEL_PATH)
        print(f"\nCandidate promoted to active model: {ACTIVE_MODEL_PATH}")
    else:
        print("\nCandidate rejected.")
        if not PROMOTE_BEST:
            print("- PROMOTE_BEST is false")
        for reason in rejection_reasons:
            print(f"- {reason}")

    print(f"Candidate model saved to: {candidate_model_path}")
    print(f"Candidate promoted: {'yes' if PROMOTE_BEST and promote else 'no'}")
    print("No trades were placed.")


if __name__ == "__main__":
    main()
