import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
VENUE_TAG = PRIMARY_VENUE or "legacy"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
WINDOW_MODE = os.getenv("WALK_FORWARD_MODE", "expanding").strip().lower()
HORIZON = os.getenv("WALK_FORWARD_HORIZON", "60s").strip().lower()
TRAIN_ROWS = int(os.getenv("WALK_FORWARD_TRAIN_ROWS", os.getenv("train_rows", "10000")))
VALIDATION_ROWS = int(os.getenv("WALK_FORWARD_VALIDATION_ROWS", os.getenv("validation_rows", "2000")))
TEST_ROWS = int(os.getenv("WALK_FORWARD_TEST_ROWS", os.getenv("test_rows", "2000")))
STEP_ROWS = int(os.getenv("WALK_FORWARD_STEP_ROWS", os.getenv("step_rows", "1000")))
MIN_TEST_ROWS = int(os.getenv("WALK_FORWARD_MIN_TEST_ROWS", os.getenv("min_test_rows", "500")))
AUTO_SHRINK_WINDOWS = os.getenv("WALK_FORWARD_AUTO_SHRINK_WINDOWS", "true").strip().lower() in {"1", "true", "yes", "y"}
THRESHOLD = float(os.getenv("WALK_FORWARD_THRESHOLD", "0.55"))
EPOCHS = int(os.getenv("WALK_FORWARD_EPOCHS", "400"))
LEARNING_RATE = float(os.getenv("WALK_FORWARD_LEARNING_RATE", "0.05"))
L2 = float(os.getenv("WALK_FORWARD_L2", "0.001"))
SEVERE_SATURATION_RATE = float(os.getenv("WALK_FORWARD_SEVERE_SATURATION_RATE", "0.25"))
WORST_AVG_RETURN_TOLERANCE = float(os.getenv("WALK_FORWARD_WORST_AVG_RETURN_TOLERANCE", "-0.005"))
TIME_BASED_KEYS = ["train_hours", "validation_hours", "test_hours", "step_hours"]

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
HIERARCHY_LOG_PATH = VENUE_DIR / f"{SYMBOL}_hierarchy_forecast_log.csv"
MEDIATOR_ROWS_PATH = VENUE_DIR / f"{SYMBOL}_hierarchy_mediator_training_rows.csv"
TINY_ROWS_PATH = VENUE_DIR / f"{SYMBOL}_tiny_direction_training_rows.csv"
OUTPUT_DETAIL_PATH = VENUE_DIR / f"{SYMBOL}_walk_forward_model_evaluation.csv"
OUTPUT_SUMMARY_PATH = VENUE_DIR / f"{SYMBOL}_walk_forward_model_summary.csv"

CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL
MEDIATOR_CANDIDATE_DIR = CANDIDATE_ROOT / "hierarchy_mediator" / VENUE_TAG
TINY_CANDIDATE_DIR = CANDIDATE_ROOT / "tiny_direction" / VENUE_TAG
SHADOW_CANDIDATE_DIR = CANDIDATE_ROOT / "shadow_pool" / VENUE_TAG
ACTIVE_TINY_PATH = PROJECT_ROOT / "models" / "active" / SYMBOL / "tiny_direction" / VENUE_TAG / "model.json"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(np.clip(values, -40.0, 40.0))
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)


def standardize(train_x, *others):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    output = [(train_x - mean) / std]
    output.extend((x - mean) / std for x in others)
    return output, mean, std


def train_logistic(x, y):
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    y = y.astype(np.float64)
    positive = max(float(y.sum()), 1.0)
    negative = max(float(len(y) - y.sum()), 1.0)
    sample_weights = np.where(y > 0.5, len(y) / (2.0 * positive), len(y) / (2.0 * negative))
    for _ in range(EPOCHS):
        probability = sigmoid(x @ weights + bias)
        error = (probability - y) * sample_weights
        weights -= LEARNING_RATE * ((x.T @ error) / max(1, len(x)) + L2 * weights)
        bias -= LEARNING_RATE * float(error.mean())
    return weights, bias


def direction_from_probability(probability, threshold=THRESHOLD):
    probability = np.asarray(probability, dtype=np.float64)
    return np.where(probability >= threshold, 1, np.where(probability <= 1.0 - threshold, -1, 0))


def numeric(frame, column, default=0.0):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def target_columns(frame, horizon):
    return_column_options = [
        f"target_realized_return_{horizon}",
        f"realized_return_{horizon}",
        f"future_return_{horizon}",
        "actual_future_return_3",
        "future_return",
    ]
    direction_column_options = [
        f"target_realized_direction_{horizon}",
        f"realized_direction_{horizon}",
        f"future_direction_{horizon}",
        "actual_direction",
    ]
    return_column = next((column for column in return_column_options if column in frame.columns), None)
    direction_column = next((column for column in direction_column_options if column in frame.columns), None)
    return return_column, direction_column


def prepare_training_frame(path, label):
    frame = read_csv(path)
    if len(frame) == 0:
        return pd.DataFrame(), f"{label}: missing or empty: {path}"
    if "timestamp" not in frame.columns:
        return pd.DataFrame(), f"{label}: missing timestamp column"
    feature_columns = sorted(column for column in frame.columns if column.startswith("feature_"))
    if not feature_columns:
        return pd.DataFrame(), f"{label}: no feature_* columns found"
    return_column, direction_column = target_columns(frame, HORIZON)
    if return_column is None:
        return pd.DataFrame(), f"{label}: no realized/future return target found for horizon {HORIZON}"
    frame = frame.copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["_target_return"] = pd.to_numeric(frame[return_column], errors="coerce")
    if direction_column and direction_column in frame.columns:
        frame["_target_direction"] = pd.to_numeric(frame[direction_column], errors="coerce")
    else:
        frame["_target_direction"] = np.where(frame["_target_return"] > 0, 1, np.where(frame["_target_return"] < 0, -1, 0))
    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["timestamp", "_target_return", "_target_direction", *feature_columns])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    frame.attrs["feature_columns"] = feature_columns
    frame.attrs["source_label"] = label
    return frame, f"{label}: loaded {len(frame)} rows, features={len(feature_columns)}, target={return_column}"


def best_combined_filter_direction(frame):
    pressure = numeric(frame, "feature_flow_1s_pred_pressure")
    both_agree_bullish = numeric(frame, "feature_both_agree_bullish")
    return np.where((both_agree_bullish >= 0.5) & (pressure.abs() >= 0.30), 1, 0)


def follow_pressure_direction(frame):
    pressure = numeric(frame, "feature_flow_1s_pred_pressure")
    return np.where(pressure > 0, 1, np.where(pressure < 0, -1, 0))


def strategy_metrics(actual_return, direction):
    actual_return = np.asarray(actual_return, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    active = direction != 0
    strategy_return = np.where(active, actual_return * direction, 0.0)
    active_returns = strategy_return[active]
    equity = np.cumprod(1.0 + np.clip(strategy_return, -0.99, 10.0)) if len(strategy_return) else np.asarray([])
    if len(equity):
        peak = np.maximum.accumulate(equity)
        drawdown = equity / np.maximum(peak, 1e-12) - 1.0
        max_drawdown = float(drawdown.min())
    else:
        max_drawdown = 0.0
    return {
        "rows": int(len(actual_return)),
        "active_rows": int(active.sum()),
        "coverage": float(active.mean()) if len(active) else 0.0,
        "win_rate": float((active_returns > 0).mean()) if len(active_returns) else np.nan,
        "avg_return": float(strategy_return.mean()) if len(strategy_return) else 0.0,
        "median_return": float(np.median(active_returns)) if len(active_returns) else np.nan,
        "min_strategy_return": float(strategy_return.min()) if len(strategy_return) else 0.0,
        "max_drawdown": max_drawdown,
    }


def baseline_directions(train, test):
    train_direction = train["_target_direction"].to_numpy(dtype=np.int64)
    up_count = int((train_direction > 0).sum())
    down_count = int((train_direction < 0).sum())
    majority_direction = 1 if up_count >= down_count else -1
    return {
        "majority": np.full(len(test), majority_direction),
        "always_long": np.ones(len(test)),
        "always_short": -np.ones(len(test)),
        "follow_1s_pressure": follow_pressure_direction(test),
        "best_combined_filter": best_combined_filter_direction(test),
    }


def latest_model_path(directory):
    directory = Path(directory)
    if not directory.exists():
        return None
    paths = sorted(directory.glob("*/model.json"), reverse=True)
    return paths[0] if paths else None


def artifact_baseline_direction(path, test, threshold=THRESHOLD):
    if path is None or not Path(path).exists():
        return None, "missing"
    try:
        artifact = load_json(path)
    except Exception as error:
        return None, f"load failed: {error}"
    trained_until = artifact_timestamp(artifact)
    if trained_until is not None and int(test["timestamp"].min()) <= int(trained_until):
        return None, "test window is not after artifact trained_until_timestamp"
    probability, reason = predict_artifact_probability(artifact, test)
    if probability is None:
        return None, reason
    return direction_from_probability(probability, float(artifact.get("prediction_threshold", threshold))), "ok"


def calibration_buckets(probability, actual_direction):
    probability = np.asarray(probability, dtype=np.float64)
    actual_up = np.asarray(actual_direction, dtype=np.float64) > 0
    buckets = []
    edges = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high)
        if not mask.any():
            buckets.append({"bucket": f"{low:.1f}-{min(high, 1.0):.1f}", "rows": 0})
            continue
        buckets.append(
            {
                "bucket": f"{low:.1f}-{min(high, 1.0):.1f}",
                "rows": int(mask.sum()),
                "avg_probability": float(probability[mask].mean()),
                "actual_up_rate": float(actual_up[mask].mean()),
            }
        )
    return buckets


def evaluate_probability(name, kind, window, test, train, probability, threshold=THRESHOLD, validation_probability=None, validation=None):
    actual = test["_target_return"].to_numpy(dtype=np.float64)
    direction = direction_from_probability(probability, threshold)
    metrics = strategy_metrics(actual, direction)
    baselines = {
        key: strategy_metrics(actual, value)
        for key, value in baseline_directions(train, test).items()
    }
    hierarchy_path = latest_model_path(MEDIATOR_CANDIDATE_DIR)
    hierarchy_direction, hierarchy_reason = artifact_baseline_direction(hierarchy_path, test, threshold)
    if hierarchy_direction is not None:
        baselines["hierarchy_mediator"] = strategy_metrics(actual, hierarchy_direction)
    active_tiny_direction, active_tiny_reason = artifact_baseline_direction(ACTIVE_TINY_PATH, test, threshold)
    if active_tiny_direction is not None:
        baselines["active_tiny_baseline"] = strategy_metrics(actual, active_tiny_direction)
    validation_metrics = {}
    if validation is not None and validation_probability is not None and len(validation):
        validation_metrics = strategy_metrics(
            validation["_target_return"].to_numpy(dtype=np.float64),
            direction_from_probability(validation_probability, threshold),
        )
    saturation_rate = float(((probability <= 0.01) | (probability >= 0.99)).mean()) if len(probability) else 0.0
    row = {
        **window,
        "model_name": name,
        "model_kind": kind,
        "horizon": HORIZON,
        "threshold": float(threshold),
        "test_rows": int(len(test)),
        "prediction_rows_above_threshold": int(metrics["active_rows"]),
        "coverage": metrics["coverage"],
        "win_rate": metrics["win_rate"],
        "avg_return": metrics["avg_return"],
        "median_return": metrics["median_return"],
        "max_adverse_window_result": metrics["min_strategy_return"],
        "max_drawdown": metrics["max_drawdown"],
        "probability_saturation_rate": saturation_rate,
        "validation_win_rate": validation_metrics.get("win_rate", np.nan),
        "validation_avg_return": validation_metrics.get("avg_return", np.nan),
        "calibration_buckets": json.dumps(calibration_buckets(probability, test["_target_direction"].to_numpy())),
        "hierarchy_mediator_baseline_status": hierarchy_reason,
        "active_tiny_baseline_status": active_tiny_reason,
    }
    for baseline_name, baseline_metrics in baselines.items():
        row[f"{baseline_name}_avg_return"] = baseline_metrics["avg_return"]
        row[f"{baseline_name}_win_rate"] = baseline_metrics["win_rate"]
        row[f"lift_vs_{baseline_name}"] = metrics["avg_return"] - baseline_metrics["avg_return"]
    return row


def row_windows(frame):
    n = len(frame)
    train_rows, validation_rows, test_rows, step_rows = TRAIN_ROWS, VALIDATION_ROWS, TEST_ROWS, STEP_ROWS
    if n < train_rows + validation_rows + MIN_TEST_ROWS and AUTO_SHRINK_WINDOWS:
        train_rows = max(1, int(n * 0.60))
        validation_rows = max(1, int(n * 0.20))
        test_rows = max(MIN_TEST_ROWS, n - train_rows - validation_rows)
        if train_rows + validation_rows + test_rows > n:
            test_rows = max(1, n - train_rows - validation_rows)
        step_rows = max(1, min(step_rows, test_rows))
    windows = []
    if test_rows < MIN_TEST_ROWS:
        return windows
    if WINDOW_MODE == "rolling":
        start = 0
        while start + train_rows + validation_rows + MIN_TEST_ROWS <= n:
            train_start = start
            train_end = start + train_rows
            validation_start = train_end
            validation_end = validation_start + validation_rows
            test_start = validation_end
            test_end = min(n, test_start + test_rows)
            if test_end - test_start >= MIN_TEST_ROWS:
                windows.append((train_start, train_end, validation_start, validation_end, test_start, test_end))
            start += step_rows
    else:
        train_end = train_rows
        while train_end + validation_rows + MIN_TEST_ROWS <= n:
            validation_start = train_end
            validation_end = validation_start + validation_rows
            test_start = validation_end
            test_end = min(n, test_start + test_rows)
            if test_end - test_start >= MIN_TEST_ROWS:
                windows.append((0, train_end, validation_start, validation_end, test_start, test_end))
            train_end += step_rows
    return windows


def time_windows(frame):
    if not all(os.getenv(key) for key in TIME_BASED_KEYS):
        return None
    train_ms = float(os.getenv("train_hours")) * 3600_000
    validation_ms = float(os.getenv("validation_hours")) * 3600_000
    test_ms = float(os.getenv("test_hours")) * 3600_000
    step_ms = float(os.getenv("step_hours")) * 3600_000
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
    start_ts = int(timestamps.min())
    end_ts = int(timestamps.max())
    windows = []
    cursor = start_ts
    while cursor + train_ms + validation_ms < end_ts:
        train_start_ts = start_ts if WINDOW_MODE == "expanding" else cursor
        train_end_ts = cursor + train_ms
        validation_end_ts = train_end_ts + validation_ms
        test_end_ts = validation_end_ts + test_ms
        train_start = int(np.searchsorted(timestamps, train_start_ts, side="left"))
        train_end = int(np.searchsorted(timestamps, train_end_ts, side="right"))
        validation_start = train_end
        validation_end = int(np.searchsorted(timestamps, validation_end_ts, side="right"))
        test_start = validation_end
        test_end = int(np.searchsorted(timestamps, test_end_ts, side="right"))
        if test_end - test_start >= MIN_TEST_ROWS:
            windows.append((train_start, train_end, validation_start, validation_end, test_start, test_end))
        cursor += step_ms
    return windows


def make_window_dict(frame, indexes, window_index):
    train_start, train_end, validation_start, validation_end, test_start, test_end = indexes
    return {
        "window_index": int(window_index),
        "window_mode": WINDOW_MODE,
        "train_start_timestamp": int(frame.iloc[train_start]["timestamp"]),
        "train_end_timestamp": int(frame.iloc[train_end - 1]["timestamp"]),
        "validation_start_timestamp": int(frame.iloc[validation_start]["timestamp"]),
        "validation_end_timestamp": int(frame.iloc[validation_end - 1]["timestamp"]),
        "test_start_timestamp": int(frame.iloc[test_start]["timestamp"]),
        "test_end_timestamp": int(frame.iloc[test_end - 1]["timestamp"]),
    }


def evaluate_retrained_variant(frame, model_name, model_kind):
    feature_columns = frame.attrs["feature_columns"]
    windows = time_windows(frame) or row_windows(frame)
    rows = []
    for index, window_indexes in enumerate(windows):
        train_start, train_end, validation_start, validation_end, test_start, test_end = window_indexes
        train = frame.iloc[train_start:train_end].copy()
        validation = frame.iloc[validation_start:validation_end].copy()
        test = frame.iloc[test_start:test_end].copy()
        if min(len(train), len(validation), len(test)) <= 0:
            continue
        x_train_raw = train[feature_columns].to_numpy(dtype=np.float64)
        x_validation_raw = validation[feature_columns].to_numpy(dtype=np.float64)
        x_test_raw = test[feature_columns].to_numpy(dtype=np.float64)
        (x_train, x_validation, x_test), _, _ = standardize(x_train_raw, x_validation_raw, x_test_raw)
        y_train = (train["_target_direction"].to_numpy(dtype=np.int64) > 0).astype(float)
        weights, bias = train_logistic(x_train, y_train)
        probability = sigmoid(x_test @ weights + bias)
        validation_probability = sigmoid(x_validation @ weights + bias)
        rows.append(
            evaluate_probability(
                model_name,
                model_kind,
                make_window_dict(frame, window_indexes, index),
                test,
                train,
                probability,
                validation_probability=validation_probability,
                validation=validation,
            )
        )
    return rows, len(windows)


def artifact_timestamp(artifact):
    return artifact.get("trained_until_timestamp") or artifact.get("model_trained_until_timestamp")


def predict_artifact_probability(artifact, frame):
    feature_columns = list(artifact.get("feature_columns", []))
    if not feature_columns or any(column not in frame.columns for column in feature_columns):
        return None, "missing artifact feature columns"
    x = frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    mean = np.asarray(artifact.get("feature_mean", np.zeros(x.shape[1])), dtype=np.float64)
    std = np.asarray(artifact.get("feature_std", np.ones(x.shape[1])), dtype=np.float64)
    if len(mean) != x.shape[1] or len(std) != x.shape[1]:
        return None, "feature scaler length mismatch"
    std = np.where(np.abs(std) < 1e-9, 1.0, std)
    x = (x - mean) / std
    if "direction_models" in artifact:
        direction_models = artifact["direction_models"]
        if HORIZON not in direction_models:
            return None, f"artifact missing direction model for {HORIZON}"
        model = direction_models[HORIZON]
        weights = np.asarray(model["weights"], dtype=np.float64)
        return sigmoid(x @ weights + float(model.get("bias", 0.0))), "ok"
    model = artifact.get("model", {})
    if isinstance(model, dict) and {"w1", "b1", "w_class", "b_class"}.issubset(model):
        w1 = np.asarray(model["w1"], dtype=np.float64)
        b1 = np.asarray(model["b1"], dtype=np.float64)
        w_class = np.asarray(model["w_class"], dtype=np.float64)
        b_class = np.asarray(model["b_class"], dtype=np.float64)
        hidden = np.maximum(x @ w1 + b1, 0.0)
        probabilities = softmax(hidden @ w_class + b_class)
        class_names = [str(name).lower() for name in artifact.get("class_names", [])]
        up_index = 2 if probabilities.shape[1] >= 3 else probabilities.shape[1] - 1
        for candidate in ["long", "bullish", "buy_dominant", "up"]:
            if candidate in class_names:
                up_index = class_names.index(candidate)
                break
        return probabilities[:, up_index], "ok"
    return None, "unsupported artifact model format"


def discover_candidate_artifacts():
    sources = [
        ("hierarchy_mediator_candidate", MEDIATOR_CANDIDATE_DIR),
        ("tiny_direction_candidate", TINY_CANDIDATE_DIR),
        ("shadow_pool_candidate", SHADOW_CANDIDATE_DIR),
    ]
    artifacts = []
    if ACTIVE_TINY_PATH.exists():
        artifacts.append(("active_tiny_baseline", ACTIVE_TINY_PATH))
    for kind, directory in sources:
        if not directory.exists():
            print(f"{kind}: skipped, missing directory {directory}")
            continue
        for path in sorted(directory.glob("*/model.json")):
            artifacts.append((kind, path))
    return artifacts


def evaluate_artifact_variant(frame, artifact_kind, path):
    try:
        artifact = load_json(path)
    except Exception as error:
        return [], f"{path}: load failed: {error}"
    trained_until = artifact_timestamp(artifact)
    trained_until = int(trained_until) if trained_until is not None else None
    windows = time_windows(frame) or row_windows(frame)
    rows = []
    skipped = 0
    for index, window_indexes in enumerate(windows):
        train_start, train_end, validation_start, validation_end, test_start, test_end = window_indexes
        train = frame.iloc[train_start:train_end].copy()
        validation = frame.iloc[validation_start:validation_end].copy()
        test = frame.iloc[test_start:test_end].copy()
        if trained_until is not None and int(test["timestamp"].min()) <= trained_until:
            skipped += 1
            continue
        probability, reason = predict_artifact_probability(artifact, test)
        if probability is None:
            return rows, f"{path}: {reason}"
        validation_probability, _ = predict_artifact_probability(artifact, validation)
        model_id = artifact.get("model_id", path.parent.name)
        rows.append(
            evaluate_probability(
                str(model_id),
                artifact_kind,
                make_window_dict(frame, window_indexes, index),
                test,
                train,
                probability,
                threshold=float(artifact.get("prediction_threshold", THRESHOLD)),
                validation_probability=validation_probability,
                validation=validation,
            )
        )
    return rows, f"{path}: evaluated_windows={len(rows)}, skipped_non_future_windows={skipped}"


def summarize(results):
    if len(results) == 0:
        return pd.DataFrame()
    frame = pd.DataFrame(results)
    rows = []
    for (model_name, model_kind, horizon), group in frame.groupby(["model_name", "model_kind", "horizon"], dropna=False):
        beat_majority = group["lift_vs_majority"] > 0
        beat_best = group["lift_vs_best_combined_filter"] > 0
        positive = group["avg_return"] > 0
        avg_return = float(group["avg_return"].mean())
        worst_idx = group["avg_return"].idxmin()
        best_idx = group["avg_return"].idxmax()
        saturation = float(group["probability_saturation_rate"].mean())
        pct_beat_majority = float(beat_majority.mean())
        pct_beat_best = float(beat_best.mean())
        pct_positive = float(positive.mean())
        stability_score = float(
            np.clip(
                100.0
                * (
                    0.40 * pct_beat_majority
                    + 0.30 * pct_beat_best
                    + 0.20 * pct_positive
                    + 0.10 * max(0.0, 1.0 - saturation / max(SEVERE_SATURATION_RATE, 1e-9))
                ),
                0.0,
                100.0,
            )
        )
        validation_good = pd.to_numeric(group["validation_avg_return"], errors="coerce").fillna(0.0) > 0
        forward_bad = group["avg_return"] <= 0
        overfit_warning = bool((validation_good & forward_bad).mean() >= 0.50)
        walk_forward_stable = bool(
            len(group) >= 5
            and pct_beat_majority >= 0.60
            and pct_beat_best >= 0.50
            and avg_return > 0
            and saturation < SEVERE_SATURATION_RATE
            and float(group["avg_return"].min()) >= WORST_AVG_RETURN_TOLERANCE
        )
        rows.append(
            {
                "model_name": model_name,
                "model_kind": model_kind,
                "horizon": horizon,
                "windows": int(len(group)),
                "average_win_rate": float(group["win_rate"].mean(skipna=True)),
                "average_return": avg_return,
                "median_window_return": float(group["avg_return"].median()),
                "percent_windows_beating_majority": pct_beat_majority,
                "percent_windows_beating_best_combined_filter": pct_beat_best,
                "percent_positive_windows": pct_positive,
                "average_probability_saturation_rate": saturation,
                "worst_window_index": int(group.loc[worst_idx, "window_index"]),
                "worst_window_avg_return": float(group.loc[worst_idx, "avg_return"]),
                "best_window_index": int(group.loc[best_idx, "window_index"]),
                "best_window_avg_return": float(group.loc[best_idx, "avg_return"]),
                "stability_score": stability_score,
                "overfit_warning": overfit_warning,
                "walk_forward_stable": walk_forward_stable,
            }
        )
    return pd.DataFrame(rows).sort_values(["walk_forward_stable", "stability_score", "average_return"], ascending=[False, False, False])


def main():
    print("Rolling walk-forward model evaluation")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {VENUE_TAG}")
    print(f"Window mode: {WINDOW_MODE}")
    print(f"Horizon: {HORIZON}")
    print(f"Hierarchy log: {HIERARCHY_LOG_PATH} ({'exists' if HIERARCHY_LOG_PATH.exists() else 'missing'})")
    results = []
    mediator_frame, message = prepare_training_frame(MEDIATOR_ROWS_PATH, "hierarchy_mediator_rows")
    print(message)
    if len(mediator_frame):
        rows, window_count = evaluate_retrained_variant(
            mediator_frame,
            "rolling_retrained_hierarchy_mediator",
            "rolling_retrained",
        )
        print(f"rolling_retrained_hierarchy_mediator: windows={window_count}, evaluated={len(rows)}")
        results.extend(rows)
        for artifact_kind, path in discover_candidate_artifacts():
            artifact_rows, artifact_message = evaluate_artifact_variant(mediator_frame, artifact_kind, path)
            print(artifact_message)
            results.extend(artifact_rows)
    tiny_frame, tiny_message = prepare_training_frame(TINY_ROWS_PATH, "tiny_direction_rows")
    print(tiny_message)
    if len(tiny_frame):
        rows, window_count = evaluate_retrained_variant(
            tiny_frame,
            "rolling_retrained_tiny_direction",
            "rolling_retrained",
        )
        print(f"rolling_retrained_tiny_direction: windows={window_count}, evaluated={len(rows)}")
        results.extend(rows)

    detail = pd.DataFrame(results)
    summary = summarize(results)
    atomic_write_csv(detail, OUTPUT_DETAIL_PATH)
    atomic_write_csv(summary, OUTPUT_SUMMARY_PATH)

    print(f"\nPer-window output: {OUTPUT_DETAIL_PATH}")
    print(f"Summary output: {OUTPUT_SUMMARY_PATH}")
    print(f"Per-window rows: {len(detail)}")
    if len(summary):
        print("\nAggregate summary")
        for _, row in summary.iterrows():
            print(
                f"- {row['model_name']} ({row['model_kind']}): "
                f"windows={int(row['windows'])}, "
                f"avg_return={row['average_return']:.4%}, "
                f"avg_win={row['average_win_rate']:.2%}, "
                f"beat_majority={row['percent_windows_beating_majority']:.2%}, "
                f"beat_best_filter={row['percent_windows_beating_best_combined_filter']:.2%}, "
                f"stability={row['stability_score']:.1f}, "
                f"walk_forward_stable={bool(row['walk_forward_stable'])}, "
                f"overfit_warning={bool(row['overfit_warning'])}"
            )
    else:
        print("No models/windows were evaluated.")
    print("Paper-only. No trades/orders/private API. No automatic promotion.")


if __name__ == "__main__":
    main()
