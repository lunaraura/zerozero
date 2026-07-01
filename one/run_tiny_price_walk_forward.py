import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
TRAINING_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_training_rows.csv"
RESULTS_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_walk_forward_results.csv"
SUMMARY_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_walk_forward_summary.csv"

FEATURE_SETS = [v.strip().lower() for v in os.getenv("WALK_FORWARD_TINY_PRICE_FEATURE_SETS", "tiny_price_v1").split(",") if v.strip()]
LOOKBACK_PROFILES = [v.strip().lower() for v in os.getenv("WALK_FORWARD_TINY_PRICE_LOOKBACK_PROFILES", "long").split(",") if v.strip()]
HORIZONS = [int(v.strip()) for v in os.getenv("WALK_FORWARD_TINY_PRICE_HORIZONS", "28,30,32,35").split(",") if v.strip()]
MODEL_TYPES = [v.strip().lower() for v in os.getenv("WALK_FORWARD_TINY_PRICE_MODEL_TYPES", "logistic_regression,ridge_regression").split(",") if v.strip()]
THRESHOLDS = [float(v.strip()) for v in os.getenv("WALK_FORWARD_TINY_PRICE_THRESHOLDS", "0.50,0.55,0.60").split(",") if v.strip()]

TRAIN_ROWS = int(os.getenv("WALK_FORWARD_TRAIN_ROWS", "30000"))
VALIDATION_ROWS = int(os.getenv("WALK_FORWARD_VALIDATION_ROWS", "10000"))
TEST_ROWS = int(os.getenv("WALK_FORWARD_TEST_ROWS", "5000"))
STEP_ROWS = int(os.getenv("WALK_FORWARD_STEP_ROWS", "5000"))
MIN_WINDOWS = int(os.getenv("WALK_FORWARD_MIN_WINDOWS", "5"))

PRICE_TINY_FLAT_BPS = float(os.getenv("PRICE_TINY_FLAT_BPS", "0.10"))
RIDGE_L2 = float(os.getenv("PRICE_TINY_RIDGE_L2", "0.001"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))
LOGISTIC_EPOCHS = int(os.getenv("WALK_FORWARD_LOGISTIC_EPOCHS", "80"))
CALIBRATION_INVERSION_MARGIN = float(os.getenv("PRICE_TINY_CALIBRATION_INVERSION_MARGIN", "0.02"))
MIN_PRED_DELTA_STD_BPS = float(os.getenv("PRICE_TINY_MIN_PRED_DELTA_STD_BPS", "0.001"))

GATE_NAMES = [
    "no_gate",
    "long_side_only",
    "short_side_only",
    "suppress_longs_when_bid_imbalance_high",
    "suppress_longs_when_low_volatility_and_bid_imbalance_high",
    "suppress_longs_when_bid_depth_dominates_ask_depth",
    "suppress_all_when_signal_coverage_low",
]


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def write_csv(rows, path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(path)


def run_builder(feature_set, horizon, lookback_profile):
    env = os.environ.copy()
    env["SYMBOL"] = SYMBOL
    env["PRIMARY_VENUE"] = PRIMARY_VENUE
    env["PRICE_TINY_FEATURE_SET"] = feature_set
    env["PRICE_TINY_HORIZON_SECONDS"] = str(horizon)
    env["PRICE_TINY_LOOKBACK_PROFILE"] = lookback_profile
    env["PROMOTE_BEST"] = "false"
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "build_tiny_price_training_rows.py")]
    print(f"Building rows: feature_set={feature_set}, horizon={horizon}s, lookback_profile={lookback_profile}")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"build_tiny_price_training_rows.py failed with exit code {result.returncode}")


def softmax(values):
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(np.clip(values, -40, 40))
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)


def standardize(train_x, validation_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (train_x - mean) / std, (validation_x - mean) / std, (test_x - mean) / std


def direction_to_class(values):
    values = np.asarray(values, dtype=np.int64)
    return np.where(values < 0, 0, np.where(values > 0, 2, 1))


def class_to_direction(classes):
    classes = np.asarray(classes, dtype=np.int64)
    return np.where(classes == 0, -1, np.where(classes == 2, 1, 0))


def direction_from_delta(delta_bps):
    delta_bps = np.asarray(delta_bps, dtype=np.float64)
    return np.where(delta_bps > PRICE_TINY_FLAT_BPS, 1, np.where(delta_bps < -PRICE_TINY_FLAT_BPS, -1, 0))


def confidence_from_delta(pred_delta):
    scale = max(PRICE_TINY_FLAT_BPS * 4.0, 1e-6)
    return np.clip(np.abs(np.asarray(pred_delta, dtype=np.float64)) / scale, 0.0, 1.0)


def train_ridge(x, y):
    x_augmented = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(x_augmented.shape[1]) * RIDGE_L2
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(x_augmented.T @ x_augmented + penalty) @ x_augmented.T @ y
    return weights[1:], float(weights[0])


def train_softmax_logistic(x, classes):
    rng = np.random.default_rng(RANDOM_SEED)
    w = rng.normal(0, 0.01, (x.shape[1], 3))
    b = np.zeros(3)
    y = np.eye(3)[classes]
    counts = np.maximum(y.sum(axis=0), 1.0)
    class_weights = len(y) / (3.0 * counts)
    for _ in range(max(20, LOGISTIC_EPOCHS)):
        p = softmax(x @ w + b)
        weighted_error = (p - y) * class_weights.reshape(1, -1)
        w -= 0.05 * ((x.T @ weighted_error) / len(x) + RIDGE_L2 * w)
        b -= 0.05 * weighted_error.mean(axis=0)
    return w, b


def prepare_frame(frame, horizon):
    target_delta = f"target_next_mid_delta_bps_{horizon}s"
    target_direction = f"target_next_mid_direction_{horizon}s"
    if target_delta not in frame.columns:
        raise RuntimeError(f"Missing required target column: {target_delta}")
    if target_direction not in frame.columns:
        raise RuntimeError(f"Missing required target column: {target_direction}")
    feature_columns = sorted(c for c in frame.columns if c.startswith("feature_") and c != "feature_set_name")
    required = ["timestamp", target_delta, target_direction, *feature_columns]
    frame = frame.copy()
    frame[required] = frame[required].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)
    return frame, feature_columns, target_delta, target_direction


def train_models(train, validation, test, feature_columns, target_delta, target_direction):
    x_train_raw = train[feature_columns].to_numpy(dtype=np.float64)
    x_validation_raw = validation[feature_columns].to_numpy(dtype=np.float64)
    x_test_raw = test[feature_columns].to_numpy(dtype=np.float64)
    x_train, x_validation, x_test = standardize(x_train_raw, x_validation_raw, x_test_raw)
    y_delta_train = train[target_delta].to_numpy(dtype=np.float64)
    class_train = direction_to_class(train[target_direction].to_numpy(dtype=np.int64))

    models = {}
    if "ridge_regression" in MODEL_TYPES:
        weights, bias = train_ridge(x_train, y_delta_train)
        models["ridge_regression"] = {"weights": weights, "bias": bias}
    if "logistic_regression" in MODEL_TYPES:
        weights, bias = train_softmax_logistic(x_train, class_train)
        models["logistic_regression"] = {"weights": weights, "bias": bias}
    predictions = {}
    for split_name, x in [("validation", x_validation), ("test", x_test)]:
        predictions[split_name] = {}
        if "ridge_regression" in models:
            pred_delta = x @ models["ridge_regression"]["weights"] + models["ridge_regression"]["bias"]
            predictions[split_name]["ridge_regression"] = {
                "pred_delta": pred_delta,
                "pred_direction": direction_from_delta(pred_delta),
                "confidence": confidence_from_delta(pred_delta),
            }
        if "logistic_regression" in models:
            probs = softmax(x @ models["logistic_regression"]["weights"] + models["logistic_regression"]["bias"])
            predictions[split_name]["logistic_regression"] = {
                "pred_delta": np.zeros(len(x), dtype=np.float64),
                "pred_direction": class_to_direction(np.argmax(probs, axis=1)),
                "confidence": probs.max(axis=1),
            }
    return predictions


def majority_direction(train, target_direction):
    if len(train) == 0:
        return 0
    return int(train[target_direction].mode().iloc[0])


def calibration_inverted(confidence, pred_direction, actual_direction):
    low = confidence < 0.60
    high = confidence >= 0.70
    low_active = low & (pred_direction != 0)
    high_active = high & (pred_direction != 0)
    if low_active.sum() < 30 or high_active.sum() < 30:
        return False, False
    low_acc = float((pred_direction[low_active] == actual_direction[low_active]).mean())
    high_acc = float((pred_direction[high_active] == actual_direction[high_active]).mean())
    inverted = high_acc + CALIBRATION_INVERSION_MARGIN < low_acc
    severe = high_acc + (CALIBRATION_INVERSION_MARGIN * 2.0) < low_acc
    return bool(inverted), bool(severe)


def finite_quantile(values, quantile, default):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return default
    return float(np.quantile(values, quantile))


def feature_array(frame, column, default=0.0):
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    return np.where(np.isfinite(values), values, default)


def depth_array(frame, column):
    values = feature_array(frame, column, default=0.0)
    return np.maximum(0.0, np.expm1(values))


def depth_ratio(numerator, denominator):
    return numerator / np.maximum(denominator, 1e-9)


def gate_variables(frame):
    bid10 = depth_array(frame, "feature_bid_depth_10bps_log1p")
    ask10 = depth_array(frame, "feature_ask_depth_10bps_log1p")
    bid25 = depth_array(frame, "feature_bid_depth_25bps_log1p")
    ask25 = depth_array(frame, "feature_ask_depth_25bps_log1p")
    return {
        "imbalance10": feature_array(frame, "feature_imbalance10", 0.0),
        "imbalance25": feature_array(frame, "feature_imbalance25", 0.0),
        "rolling_volatility_60s": feature_array(frame, "feature_rolling_volatility_60s", 0.0),
        "range_60s": feature_array(frame, "feature_recent_high_low_range_60s", 0.0),
        "bid_depth_ratio_10bps": depth_ratio(bid10, ask10),
        "bid_depth_ratio_25bps": depth_ratio(bid25, ask25),
    }


def base_active_mask(prediction, threshold):
    confidence = np.asarray(prediction["confidence"], dtype=np.float64)
    pred_direction = np.asarray(prediction["pred_direction"], dtype=np.int64)
    return (confidence >= threshold) & (pred_direction != 0)


def apply_gate(frame, prediction, threshold, gate_config):
    gate_name = gate_config.get("gate_name", "no_gate")
    pred_direction = np.asarray(prediction["pred_direction"], dtype=np.int64)
    confidence = np.asarray(prediction["confidence"], dtype=np.float64)
    allowed = np.ones(len(frame), dtype=bool)
    variables = gate_variables(frame)
    if gate_name == "long_side_only":
        return pred_direction > 0
    if gate_name == "short_side_only":
        return pred_direction < 0
    if gate_name == "suppress_longs_when_bid_imbalance_high":
        high_imbalance = (variables["imbalance10"] >= gate_config["imbalance10_threshold"]) | (
            variables["imbalance25"] >= gate_config["imbalance25_threshold"]
        )
        allowed[(pred_direction > 0) & high_imbalance] = False
        return allowed
    if gate_name == "suppress_longs_when_low_volatility_and_bid_imbalance_high":
        high_imbalance = (variables["imbalance10"] >= gate_config["imbalance10_threshold"]) | (
            variables["imbalance25"] >= gate_config["imbalance25_threshold"]
        )
        low_volatility = variables["rolling_volatility_60s"] <= gate_config["volatility60_threshold"]
        low_range = variables["range_60s"] <= gate_config["range60_threshold"]
        allowed[(pred_direction > 0) & high_imbalance & low_volatility & low_range] = False
        return allowed
    if gate_name == "suppress_longs_when_bid_depth_dominates_ask_depth":
        bid_dominates = (variables["bid_depth_ratio_10bps"] >= gate_config["depth_ratio10_threshold"]) | (
            variables["bid_depth_ratio_25bps"] >= gate_config["depth_ratio25_threshold"]
        )
        allowed[(pred_direction > 0) & bid_dominates] = False
        return allowed
    if gate_name == "suppress_all_when_signal_coverage_low":
        coverage = float(base_active_mask(prediction, threshold).mean()) if len(frame) else 0.0
        if coverage < gate_config["min_signal_coverage"]:
            return np.zeros(len(frame), dtype=bool)
        return allowed
    return allowed


def gate_threshold_string(gate_config):
    payload = {k: v for k, v in gate_config.items() if k != "gate_name"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def evaluate_prediction(frame, train, target_delta, target_direction, prediction, threshold, gate_config=None):
    gate_config = gate_config or {"gate_name": "no_gate"}
    actual_delta = frame[target_delta].to_numpy(dtype=np.float64)
    actual_direction = frame[target_direction].to_numpy(dtype=np.int64)
    pred_delta = np.asarray(prediction["pred_delta"], dtype=np.float64)
    pred_direction = np.asarray(prediction["pred_direction"], dtype=np.int64)
    confidence = np.asarray(prediction["confidence"], dtype=np.float64)
    active = base_active_mask(prediction, threshold) & apply_gate(frame, prediction, threshold, gate_config)
    active_rows = int(active.sum())
    majority = majority_direction(train, target_direction)
    majority_on_active = np.full(active_rows, majority, dtype=np.int64) if active_rows else np.asarray([], dtype=np.int64)
    calibration_bad, high_conf_bad = calibration_inverted(confidence, pred_direction, actual_direction)
    mae = float(np.mean(np.abs(pred_delta - actual_delta))) if len(frame) else np.nan
    rmse = float(np.sqrt(np.mean((pred_delta - actual_delta) ** 2))) if len(frame) else np.nan
    zero_rmse = float(np.sqrt(np.mean(actual_delta ** 2))) if len(frame) else np.nan
    zero_mae = float(np.mean(np.abs(actual_delta))) if len(frame) else np.nan
    pred_std = float(np.nanstd(pred_delta)) if len(pred_delta) else 0.0
    price_candidate_useful = bool(mae < zero_mae and rmse <= zero_rmse * 1.10 and pred_std >= MIN_PRED_DELTA_STD_BPS)
    if active_rows:
        strategy_return = actual_delta[active] * np.sign(pred_direction[active])
        directional_accuracy = float((pred_direction[active] == actual_direction[active]).mean())
        avg_return = float(strategy_return.mean())
        up_mask = active & (pred_direction > 0)
        down_mask = active & (pred_direction < 0)
        up_return = float(actual_delta[up_mask].mean()) if up_mask.any() else np.nan
        down_return = float(actual_delta[down_mask].mean()) if down_mask.any() else np.nan
        majority_acc = float((majority_on_active == actual_direction[active]).mean()) if len(majority_on_active) else np.nan
        always_up_return = float(actual_delta[active].mean())
        always_down_return = float((-actual_delta[active]).mean())
        lift_vs_majority = float(directional_accuracy - majority_acc)
        lift_vs_always_up = float(avg_return - always_up_return)
        lift_vs_always_down = float(avg_return - always_down_return)
    else:
        directional_accuracy = np.nan
        avg_return = np.nan
        up_return = np.nan
        down_return = np.nan
        lift_vs_majority = np.nan
        lift_vs_always_up = np.nan
        lift_vs_always_down = np.nan
    predicted_up_all = pred_direction > 0
    predicted_down_all = pred_direction < 0
    predicted_up_active = active & (pred_direction > 0)
    predicted_down_active = active & (pred_direction < 0)
    direction_candidate_useful = bool(
        active_rows >= 300
        and np.isfinite(avg_return)
        and avg_return > 0
        and np.isfinite(lift_vs_majority)
        and lift_vs_majority > 0
        and (not np.isfinite(up_return) or not np.isfinite(down_return) or up_return > down_return)
        and not high_conf_bad
    )
    return {
        "test_rows": int(len(frame)),
        "active_rows_at_threshold": active_rows,
        "coverage": float(active_rows / len(frame)) if len(frame) else 0.0,
        "directional_accuracy": directional_accuracy,
        "avg_return_bps": avg_return,
        "up_return_bps": up_return,
        "down_return_bps": down_return,
        "lift_vs_majority": lift_vs_majority,
        "lift_vs_always_up": lift_vs_always_up,
        "lift_vs_always_down": lift_vs_always_down,
        "calibration_inverted": calibration_bad,
        "high_confidence_inverted": high_conf_bad,
        "price_candidate_useful": price_candidate_useful,
        "direction_candidate_useful": direction_candidate_useful,
        "failure_window": bool(np.isfinite(avg_return) and avg_return < 0),
        "predicted_up_active_count": int(predicted_up_active.sum()),
        "predicted_down_active_count": int(predicted_down_active.sum()),
        "predicted_up_avg_return_bps": float(actual_delta[predicted_up_active].mean()) if predicted_up_active.any() else np.nan,
        "predicted_down_avg_return_bps": float((-actual_delta[predicted_down_active]).mean()) if predicted_down_active.any() else np.nan,
        "predicted_up_win_rate": float((actual_delta[predicted_up_active] > 0).mean()) if predicted_up_active.any() else np.nan,
        "predicted_down_win_rate": float((-actual_delta[predicted_down_active] > 0).mean()) if predicted_down_active.any() else np.nan,
        "average_absolute_future_move_bps": float(np.mean(np.abs(actual_delta))) if len(actual_delta) else np.nan,
        "average_predicted_confidence": float(np.mean(confidence)) if len(confidence) else np.nan,
        "predicted_up_count": int(predicted_up_all.sum()),
        "predicted_down_count": int(predicted_down_all.sum()),
        "avg_return_for_predicted_up_bps": float(actual_delta[predicted_up_all].mean()) if predicted_up_all.any() else np.nan,
        "avg_return_for_predicted_down_bps": float((-actual_delta[predicted_down_all]).mean()) if predicted_down_all.any() else np.nan,
    }


def select_on_validation(validation, train, target_delta, target_direction, validation_predictions):
    candidates = []
    for model_type, prediction in validation_predictions.items():
        for threshold in THRESHOLDS:
            metrics = evaluate_prediction(validation, train, target_delta, target_direction, prediction, threshold)
            stable = metrics["active_rows_at_threshold"] >= 300 and np.isfinite(metrics["avg_return_bps"]) and metrics["avg_return_bps"] > 0
            score = (
                1 if stable else 0,
                metrics["avg_return_bps"] if np.isfinite(metrics["avg_return_bps"]) else -1e9,
                metrics["lift_vs_majority"] if np.isfinite(metrics["lift_vs_majority"]) else -1e9,
                metrics["active_rows_at_threshold"],
            )
            candidates.append((score, model_type, threshold, metrics))
    if not candidates:
        raise RuntimeError("No validation candidates were produced.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2], candidates[0][3]


def unique_configs(configs):
    seen = set()
    output = []
    for config in configs:
        key = gate_threshold_string(config)
        full_key = (config.get("gate_name"), key)
        if full_key in seen:
            continue
        seen.add(full_key)
        output.append(config)
    return output


def candidate_gate_configs(frame, prediction, threshold, gate_name):
    variables = gate_variables(frame)
    if gate_name in {"no_gate", "long_side_only", "short_side_only"}:
        return [{"gate_name": gate_name}]
    if gate_name == "suppress_longs_when_bid_imbalance_high":
        configs = []
        for quantile in [0.60, 0.70, 0.80]:
            configs.append(
                {
                    "gate_name": gate_name,
                    "imbalance10_threshold": finite_quantile(variables["imbalance10"], quantile, 0.20),
                    "imbalance25_threshold": finite_quantile(variables["imbalance25"], quantile, 0.20),
                }
            )
        return unique_configs(configs)
    if gate_name == "suppress_longs_when_low_volatility_and_bid_imbalance_high":
        configs = []
        for imbalance_quantile in [0.60, 0.70, 0.80]:
            for low_quantile in [0.30, 0.50]:
                configs.append(
                    {
                        "gate_name": gate_name,
                        "imbalance10_threshold": finite_quantile(variables["imbalance10"], imbalance_quantile, 0.20),
                        "imbalance25_threshold": finite_quantile(variables["imbalance25"], imbalance_quantile, 0.20),
                        "volatility60_threshold": finite_quantile(variables["rolling_volatility_60s"], low_quantile, 0.0),
                        "range60_threshold": finite_quantile(variables["range_60s"], low_quantile, 0.0),
                    }
                )
        return unique_configs(configs)
    if gate_name == "suppress_longs_when_bid_depth_dominates_ask_depth":
        configs = []
        for quantile in [0.60, 0.70, 0.80]:
            configs.append(
                {
                    "gate_name": gate_name,
                    "depth_ratio10_threshold": finite_quantile(variables["bid_depth_ratio_10bps"], quantile, 1.25),
                    "depth_ratio25_threshold": finite_quantile(variables["bid_depth_ratio_25bps"], quantile, 1.25),
                }
            )
        return unique_configs(configs)
    if gate_name == "suppress_all_when_signal_coverage_low":
        base_coverage = float(base_active_mask(prediction, threshold).mean()) if len(frame) else 0.0
        candidates = sorted(set([0.02, 0.05, 0.10, 0.20, round(base_coverage * 0.75, 4)]))
        return [{"gate_name": gate_name, "min_signal_coverage": value} for value in candidates if value >= 0]
    return [{"gate_name": "no_gate"}]


def select_gate_configs(validation, train, target_delta, target_direction, prediction, threshold):
    selected = []
    for gate_name in GATE_NAMES:
        candidates = []
        for gate_config in candidate_gate_configs(validation, prediction, threshold, gate_name):
            metrics = evaluate_prediction(validation, train, target_delta, target_direction, prediction, threshold, gate_config)
            predicted_up_loss_penalty = (
                metrics["predicted_up_avg_return_bps"]
                if np.isfinite(metrics["predicted_up_avg_return_bps"])
                else -1e9
            )
            score = (
                1 if metrics["active_rows_at_threshold"] >= 300 else 0,
                metrics["avg_return_bps"] if np.isfinite(metrics["avg_return_bps"]) else -1e9,
                metrics["lift_vs_majority"] if np.isfinite(metrics["lift_vs_majority"]) else -1e9,
                predicted_up_loss_penalty,
                metrics["active_rows_at_threshold"],
            )
            candidates.append((score, gate_config, metrics))
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_config, best_metrics = candidates[0]
        selected.append((best_config, best_metrics))
    return selected


def timestamp_at(frame, index):
    return int(frame["timestamp"].iloc[index])


def mean_column(frame, column):
    if column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.mean()) if values.notna().any() else np.nan


def median_column(frame, column):
    if column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.median()) if values.notna().any() else np.nan


def depth_from_log_column(frame, column):
    if column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not values.notna().any():
        return np.nan
    return float(np.expm1(values).mean())


def window_regime_diagnostics(frame, target_direction):
    labels = pd.to_numeric(frame[target_direction], errors="coerce").fillna(0).astype(int)
    rows = max(1, len(labels))
    return {
        "avg_spread_percent": mean_column(frame, "feature_spread_percent"),
        "median_spread_percent": median_column(frame, "feature_spread_percent"),
        "avg_rolling_volatility_30s": mean_column(frame, "feature_rolling_volatility_30s"),
        "avg_rolling_volatility_60s": mean_column(frame, "feature_rolling_volatility_60s"),
        "avg_rolling_volatility_120s": mean_column(frame, "feature_rolling_volatility_120s"),
        "avg_range_30s": mean_column(frame, "feature_recent_high_low_range_30s"),
        "avg_range_60s": mean_column(frame, "feature_recent_high_low_range_60s"),
        "avg_range_120s": mean_column(frame, "feature_recent_high_low_range_120s"),
        "avg_imbalance10": mean_column(frame, "feature_imbalance10"),
        "avg_imbalance25": mean_column(frame, "feature_imbalance25"),
        "avg_bid_depth_10bps": depth_from_log_column(frame, "feature_bid_depth_10bps_log1p"),
        "avg_ask_depth_10bps": depth_from_log_column(frame, "feature_ask_depth_10bps_log1p"),
        "avg_bid_depth_25bps": depth_from_log_column(frame, "feature_bid_depth_25bps_log1p"),
        "avg_ask_depth_25bps": depth_from_log_column(frame, "feature_ask_depth_25bps_log1p"),
        "percent_flat_labels": float((labels == 0).sum() / rows),
        "percent_up_labels": float((labels > 0).sum() / rows),
        "percent_down_labels": float((labels < 0).sum() / rows),
    }


def run_combo(feature_set, horizon, lookback_profile):
    run_builder(feature_set, horizon, lookback_profile)
    frame = read_csv(TRAINING_PATH)
    frame, feature_columns, target_delta, target_direction = prepare_frame(frame, horizon)
    total_needed = TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS
    if len(frame) < total_needed:
        print(f"Skipping combo; not enough rows: {len(frame)} < {total_needed}")
        return []
    rows = []
    window_index = 0
    start = 0
    while start + total_needed <= len(frame):
        train = frame.iloc[start : start + TRAIN_ROWS].copy()
        validation = frame.iloc[start + TRAIN_ROWS : start + TRAIN_ROWS + VALIDATION_ROWS].copy()
        test_start = start + TRAIN_ROWS + VALIDATION_ROWS
        test = frame.iloc[test_start : test_start + TEST_ROWS].copy()
        predictions = train_models(train, validation, test, feature_columns, target_delta, target_direction)
        selected_model, selected_threshold, validation_metrics = select_on_validation(
            validation,
            train,
            target_delta,
            target_direction,
            predictions["validation"],
        )
        selected_gate_configs = select_gate_configs(
            validation,
            train,
            target_delta,
            target_direction,
            predictions["validation"][selected_model],
            selected_threshold,
        )
        regime_diagnostics = window_regime_diagnostics(test, target_direction)
        gate_rows = []
        for gate_config, validation_gate_metrics in selected_gate_configs:
            test_metrics = evaluate_prediction(
                test,
                train,
                target_delta,
                target_direction,
                predictions["test"][selected_model],
                selected_threshold,
                gate_config,
            )
            row = {
                "window_index": window_index,
                "train_start_timestamp": timestamp_at(train, 0),
                "train_end_timestamp": timestamp_at(train, len(train) - 1),
                "validation_start_timestamp": timestamp_at(validation, 0),
                "validation_end_timestamp": timestamp_at(validation, len(validation) - 1),
                "test_start_timestamp": timestamp_at(test, 0),
                "test_end_timestamp": timestamp_at(test, len(test) - 1),
                "feature_set": feature_set,
                "horizon_seconds": horizon,
                "lookback_profile": lookback_profile,
                "model_type": selected_model,
                "selected_threshold": selected_threshold,
                "gate_name": gate_config.get("gate_name", "no_gate"),
                "gate_thresholds": gate_threshold_string(gate_config),
                **test_metrics,
                **regime_diagnostics,
                "validation_active_rows_at_threshold": validation_gate_metrics["active_rows_at_threshold"],
                "validation_avg_return_bps": validation_gate_metrics["avg_return_bps"],
                "validation_lift_vs_majority": validation_gate_metrics["lift_vs_majority"],
                "validation_predicted_up_avg_return_bps": validation_gate_metrics["predicted_up_avg_return_bps"],
                "validation_predicted_down_avg_return_bps": validation_gate_metrics["predicted_down_avg_return_bps"],
            }
            gate_rows.append(row)
        rows.extend(gate_rows)
        no_gate_metrics = next((row for row in gate_rows if row["gate_name"] == "no_gate"), gate_rows[0])
        best_gate_metrics = max(
            gate_rows,
            key=lambda row: row["avg_return_bps"] if np.isfinite(row["avg_return_bps"]) else -1e9,
        )
        print(
            f"Window {window_index}: selected {selected_model} threshold={selected_threshold:.2f} "
            f"no_gate_active={no_gate_metrics['active_rows_at_threshold']} no_gate_return={no_gate_metrics['avg_return_bps']:.4f}bps "
            f"best_gate={best_gate_metrics['gate_name']} best_return={best_gate_metrics['avg_return_bps']:.4f}bps"
        )
        window_index += 1
        start += STEP_ROWS
    return rows


def bool_mean(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def summarize_group(rows):
    windows = len(rows)
    avg_returns = np.asarray([r["avg_return_bps"] for r in rows if np.isfinite(r["avg_return_bps"])], dtype=np.float64)
    active_rows = np.asarray([r["active_rows_at_threshold"] for r in rows], dtype=np.float64)
    coverage = np.asarray([r["coverage"] for r in rows], dtype=np.float64)
    predicted_up_returns = np.asarray(
        [r["predicted_up_avg_return_bps"] for r in rows if np.isfinite(r["predicted_up_avg_return_bps"])],
        dtype=np.float64,
    )
    positive = [r["avg_return_bps"] > 0 if np.isfinite(r["avg_return_bps"]) else False for r in rows]
    failure = [bool(r.get("failure_window", False)) for r in rows]
    beating_majority = [r["lift_vs_majority"] > 0 if np.isfinite(r["lift_vs_majority"]) else False for r in rows]
    severe_inversion_rate = bool_mean(r["high_confidence_inverted"] for r in rows)
    percent_positive = bool_mean(positive)
    percent_beating_majority = bool_mean(beating_majority)
    median_return = float(np.median(avg_returns)) if len(avg_returns) else np.nan
    worst_return = float(np.min(avg_returns)) if len(avg_returns) else np.nan
    median_active = float(np.median(active_rows)) if len(active_rows) else 0.0
    stability_score = float(
        0.35 * percent_positive
        + 0.35 * percent_beating_majority
        + 0.15 * (1.0 if median_return > 0 else 0.0)
        + 0.10 * (1.0 if median_active >= 300 else 0.0)
        + 0.05 * (1.0 - severe_inversion_rate)
    )
    stable = bool(
        windows >= MIN_WINDOWS
        and percent_positive >= 0.60
        and percent_beating_majority >= 0.60
        and np.isfinite(median_return)
        and median_return > 0
        and np.isfinite(worst_return)
        and worst_return > -1.0
        and median_active >= 300
        and severe_inversion_rate <= 0.20
    )
    return {
        "feature_set": rows[0]["feature_set"],
        "horizon_seconds": rows[0]["horizon_seconds"],
        "lookback_profile": rows[0]["lookback_profile"],
        "gate_name": rows[0].get("gate_name", "no_gate"),
        "selected_model_counts": ";".join(f"{k}:{v}" for k, v in pd.Series([r["model_type"] for r in rows]).value_counts().to_dict().items()),
        "windows": windows,
        "positive_windows": int(sum(positive)),
        "percent_positive_windows": percent_positive,
        "failure_windows": int(sum(failure)),
        "percent_failure_windows": bool_mean(failure),
        "windows_beating_majority": int(sum(beating_majority)),
        "percent_windows_beating_majority": percent_beating_majority,
        "mean_avg_return_bps": float(np.mean(avg_returns)) if len(avg_returns) else np.nan,
        "median_avg_return_bps": median_return,
        "worst_avg_return_bps": worst_return,
        "mean_active_rows": float(np.mean(active_rows)) if len(active_rows) else 0.0,
        "median_active_rows": median_active,
        "mean_coverage": float(np.mean(coverage)) if len(coverage) else 0.0,
        "mean_predicted_up_avg_return_bps": float(np.mean(predicted_up_returns)) if len(predicted_up_returns) else np.nan,
        "predicted_up_loss_windows": int(sum(np.isfinite(r["predicted_up_avg_return_bps"]) and r["predicted_up_avg_return_bps"] < 0 for r in rows)),
        "high_confidence_inverted_windows": int(sum(bool(r["high_confidence_inverted"]) for r in rows)),
        "high_confidence_inverted_rate": severe_inversion_rate,
        "stability_score": stability_score,
        "walk_forward_stable": stable,
        "gate_helpful": False,
        "gate_helpful_reason": "",
    }


def summarize_gate_groups(rows):
    summaries = []
    grouped = {}
    for row in rows:
        key = (row["feature_set"], row["horizon_seconds"], row["lookback_profile"], row.get("gate_name", "no_gate"))
        grouped.setdefault(key, []).append(row)
    for group_rows in grouped.values():
        summaries.append(summarize_group(group_rows))
    baseline_by_combo = {
        (row["feature_set"], row["horizon_seconds"], row["lookback_profile"]): row
        for row in summaries
        if row["gate_name"] == "no_gate"
    }
    for row in summaries:
        combo = (row["feature_set"], row["horizon_seconds"], row["lookback_profile"])
        baseline = baseline_by_combo.get(combo)
        if row["gate_name"] == "no_gate" or baseline is None:
            row["gate_helpful"] = False
            row["gate_helpful_reason"] = "baseline"
            continue
        median_improves = row["median_avg_return_bps"] > baseline["median_avg_return_bps"]
        worst_improves = row["worst_avg_return_bps"] > baseline["worst_avg_return_bps"]
        positive_not_lower = row["percent_positive_windows"] >= baseline["percent_positive_windows"]
        enough_rows = row["median_active_rows"] >= 300
        up_loss_reduced = (
            row["predicted_up_loss_windows"] < baseline["predicted_up_loss_windows"]
            or (
                np.isfinite(row["mean_predicted_up_avg_return_bps"])
                and np.isfinite(baseline["mean_predicted_up_avg_return_bps"])
                and row["mean_predicted_up_avg_return_bps"] > baseline["mean_predicted_up_avg_return_bps"]
            )
        )
        row["gate_helpful"] = bool(median_improves and worst_improves and positive_not_lower and enough_rows and up_loss_reduced)
        failed = []
        if not median_improves:
            failed.append("median_not_improved")
        if not worst_improves:
            failed.append("worst_not_improved")
        if not positive_not_lower:
            failed.append("positive_window_rate_lower")
        if not enough_rows:
            failed.append("median_active_rows_below_300")
        if not up_loss_reduced:
            failed.append("predicted_up_losses_not_reduced")
        row["gate_helpful_reason"] = "ok" if row["gate_helpful"] else ";".join(failed)
    return summaries


RESULT_FIELDS = [
    "window_index",
    "train_start_timestamp",
    "train_end_timestamp",
    "validation_start_timestamp",
    "validation_end_timestamp",
    "test_start_timestamp",
    "test_end_timestamp",
    "feature_set",
    "horizon_seconds",
    "lookback_profile",
    "model_type",
    "selected_threshold",
    "gate_name",
    "gate_thresholds",
    "test_rows",
    "active_rows_at_threshold",
    "coverage",
    "directional_accuracy",
    "avg_return_bps",
    "up_return_bps",
    "down_return_bps",
    "lift_vs_majority",
    "lift_vs_always_up",
    "lift_vs_always_down",
    "calibration_inverted",
    "high_confidence_inverted",
    "price_candidate_useful",
    "direction_candidate_useful",
    "failure_window",
    "predicted_up_active_count",
    "predicted_down_active_count",
    "predicted_up_avg_return_bps",
    "predicted_down_avg_return_bps",
    "predicted_up_win_rate",
    "predicted_down_win_rate",
    "avg_spread_percent",
    "median_spread_percent",
    "avg_rolling_volatility_30s",
    "avg_rolling_volatility_60s",
    "avg_rolling_volatility_120s",
    "avg_range_30s",
    "avg_range_60s",
    "avg_range_120s",
    "avg_imbalance10",
    "avg_imbalance25",
    "avg_bid_depth_10bps",
    "avg_ask_depth_10bps",
    "avg_bid_depth_25bps",
    "avg_ask_depth_25bps",
    "percent_flat_labels",
    "percent_up_labels",
    "percent_down_labels",
    "average_absolute_future_move_bps",
    "average_predicted_confidence",
    "predicted_up_count",
    "predicted_down_count",
    "avg_return_for_predicted_up_bps",
    "avg_return_for_predicted_down_bps",
    "validation_active_rows_at_threshold",
    "validation_avg_return_bps",
    "validation_lift_vs_majority",
    "validation_predicted_up_avg_return_bps",
    "validation_predicted_down_avg_return_bps",
]

SUMMARY_FIELDS = [
    "feature_set",
    "horizon_seconds",
    "lookback_profile",
    "gate_name",
    "selected_model_counts",
    "windows",
    "positive_windows",
    "percent_positive_windows",
    "failure_windows",
    "percent_failure_windows",
    "windows_beating_majority",
    "percent_windows_beating_majority",
    "mean_avg_return_bps",
    "median_avg_return_bps",
    "worst_avg_return_bps",
    "mean_active_rows",
    "median_active_rows",
    "mean_coverage",
    "mean_predicted_up_avg_return_bps",
    "predicted_up_loss_windows",
    "high_confidence_inverted_windows",
    "high_confidence_inverted_rate",
    "stability_score",
    "walk_forward_stable",
    "gate_helpful",
    "gate_helpful_reason",
]


COMPARISON_COLUMNS = [
    "avg_spread_percent",
    "median_spread_percent",
    "avg_rolling_volatility_30s",
    "avg_rolling_volatility_60s",
    "avg_rolling_volatility_120s",
    "avg_range_30s",
    "avg_range_60s",
    "avg_range_120s",
    "avg_imbalance10",
    "avg_imbalance25",
    "avg_bid_depth_10bps",
    "avg_ask_depth_10bps",
    "avg_bid_depth_25bps",
    "avg_ask_depth_25bps",
    "percent_flat_labels",
    "percent_up_labels",
    "percent_down_labels",
    "average_absolute_future_move_bps",
    "average_predicted_confidence",
    "predicted_up_count",
    "predicted_down_count",
    "avg_return_for_predicted_up_bps",
    "avg_return_for_predicted_down_bps",
    "active_rows_at_threshold",
    "coverage",
]


def safe_mean(rows, column):
    values = [float(row[column]) for row in rows if column in row and np.isfinite(row[column])]
    return float(np.mean(values)) if values else np.nan


def print_positive_vs_failure_comparison(rows):
    print("")
    print("Positive windows vs failure windows")
    if not rows:
        print("- no rows")
        return
    positive = [row for row in rows if np.isfinite(row.get("avg_return_bps", np.nan)) and row["avg_return_bps"] > 0]
    failure = [row for row in rows if row.get("failure_window", False)]
    print(f"- positive windows: {len(positive)}")
    print(f"- failure windows: {len(failure)}")
    if not positive or not failure:
        print("- comparison unavailable because one side has no rows")
        return
    print("- column: positive_mean | failure_mean | difference")
    for column in COMPARISON_COLUMNS:
        pos_mean = safe_mean(positive, column)
        fail_mean = safe_mean(failure, column)
        diff = pos_mean - fail_mean if np.isfinite(pos_mean) and np.isfinite(fail_mean) else np.nan
        print(f"  {column}: {pos_mean:.6f} | {fail_mean:.6f} | {diff:.6f}")


def main():
    print("Tiny price rolling walk-forward evaluation")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Feature sets: {FEATURE_SETS}")
    print(f"Lookback profiles: {LOOKBACK_PROFILES}")
    print(f"Horizons: {HORIZONS}")
    print(f"Model types: {MODEL_TYPES}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Windows: train={TRAIN_ROWS}, validation={VALIDATION_ROWS}, test={TEST_ROWS}, step={STEP_ROWS}")
    all_rows = []
    summary_rows = []
    for feature_set in FEATURE_SETS:
        for lookback_profile in LOOKBACK_PROFILES:
            for horizon in HORIZONS:
                print("")
                print(f"=== Walk-forward combo: feature_set={feature_set}, horizon={horizon}s, lookback_profile={lookback_profile} ===")
                rows = run_combo(feature_set, horizon, lookback_profile)
                all_rows.extend(rows)
                if rows:
                    summary_rows.extend(summarize_gate_groups(rows))
                    write_csv(all_rows, RESULTS_PATH, RESULT_FIELDS)
                    write_csv(summary_rows, SUMMARY_PATH, SUMMARY_FIELDS)
    write_csv(all_rows, RESULTS_PATH, RESULT_FIELDS)
    write_csv(summary_rows, SUMMARY_PATH, SUMMARY_FIELDS)
    print("")
    print(f"Walk-forward results: {RESULTS_PATH}")
    print(f"Walk-forward summary: {SUMMARY_PATH}")
    print(f"Result rows: {len(all_rows)}")
    print("Summary")
    for row in sorted(summary_rows, key=lambda r: r["stability_score"], reverse=True):
        print(
            f"- {row['feature_set']} h={row['horizon_seconds']}s lookback={row['lookback_profile']} gate={row['gate_name']}: "
            f"windows={row['windows']} positive={row['percent_positive_windows']:.2%} "
            f"beat_majority={row['percent_windows_beating_majority']:.2%} "
            f"median_return={row['median_avg_return_bps']:.4f}bps "
            f"median_active={row['median_active_rows']:.0f} "
            f"stable={row['walk_forward_stable']} helpful={row['gate_helpful']}"
        )
    print_positive_vs_failure_comparison([row for row in all_rows if row.get("gate_name", "no_gate") == "no_gate"])
    print("No promotion. Paper-only. No trades/orders/private API.")


if __name__ == "__main__":
    main()
