import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from tiny_price_feature_utils import (
    assert_numeric_feature_columns,
    feature_schema_hash,
    select_model_feature_columns,
    slugify,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR

DEFAULT_TARGETS = "move_before_adverse_30s,instability_30s,direction_30s,return_bps_30s"


def parse_csv_env(*names, default=""):
    text = ""
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            text = str(value)
            break
    if not text:
        text = default
    return [value.strip() for value in str(text).split(",") if value.strip()]


TARGET_SPECS = [
    value.lower()
    for value in parse_csv_env(
        "PRICE_TINY_WALK_FORWARD_TARGET_SPECS",
        "PRICE_TINY_TARGET_SPEC",
        default=DEFAULT_TARGETS,
    )
]
FEATURE_GROUPS = parse_csv_env("PRICE_TINY_FEATURE_GROUPS", default="base_tiny_price_v1")
MODEL_SPECS_ENV = [value.lower() for value in parse_csv_env("PRICE_TINY_MODEL_SPECS", default="")]

TRAIN_ROWS = int(os.getenv("PRICE_TINY_WALK_FORWARD_TRAIN_ROWS", os.getenv("WALK_FORWARD_TRAIN_ROWS", "30000")))
VALIDATION_ROWS = int(os.getenv("PRICE_TINY_WALK_FORWARD_VALIDATION_ROWS", os.getenv("WALK_FORWARD_VALIDATION_ROWS", "10000")))
TEST_ROWS = int(os.getenv("PRICE_TINY_WALK_FORWARD_TEST_ROWS", os.getenv("WALK_FORWARD_TEST_ROWS", "5000")))
STEP_ROWS = int(os.getenv("PRICE_TINY_WALK_FORWARD_STEP_ROWS", os.getenv("WALK_FORWARD_STEP_ROWS", "5000")))
MIN_TEST_ROWS = int(os.getenv("PRICE_TINY_WALK_FORWARD_MIN_TEST_ROWS", "500"))
MIN_ACTIVE_ROWS = int(os.getenv("PRICE_TINY_WALK_FORWARD_MIN_ACTIVE_ROWS", "100"))
MAX_WINDOWS = int(os.getenv("PRICE_TINY_WALK_FORWARD_MAX_WINDOWS", "0") or "0")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))
RIDGE_L2 = float(os.getenv("PRICE_TINY_RIDGE_L2", "0.001"))
LOGISTIC_EPOCHS = int(os.getenv("PRICE_TINY_WALK_FORWARD_LOGISTIC_EPOCHS", "80"))
PRICE_TINY_FLAT_BPS = float(os.getenv("PRICE_TINY_FLAT_BPS", "0.10"))
CONFIDENCE_THRESHOLDS = [
    float(value.strip())
    for value in os.getenv("PRICE_TINY_WALK_FORWARD_CONFIDENCE_THRESHOLDS", "0.50,0.55,0.60,0.65,0.70").split(",")
    if value.strip()
]
AUTO_BUILD_ROWS = os.getenv("PRICE_TINY_WALK_FORWARD_AUTO_BUILD", "true").strip().lower() in {"1", "true", "yes", "y"}
REQUIRE_EXACT_FEATURE_GROUPS = os.getenv("PRICE_TINY_REQUIRE_EXACT_FEATURE_GROUPS", "false").strip().lower() in {"1", "true", "yes", "y"}
ESTIMATED_FEE_BPS = float(os.getenv("PRICE_TINY_ESTIMATED_FEE_BPS", "0"))
ESTIMATED_SLIPPAGE_BPS = float(os.getenv("PRICE_TINY_ESTIMATED_SLIPPAGE_BPS", "0"))
CHARGE_HALF_SPREAD = os.getenv("PRICE_TINY_CHARGE_HALF_SPREAD", "true").strip().lower() in {"1", "true", "yes", "y"}
OUTPUT_SUFFIX = slugify(os.getenv("PRICE_TINY_WALK_FORWARD_OUTPUT_SUFFIX", "").strip(), "")

WALK_FORWARD_MODEL_ROOT = PROJECT_ROOT / "models" / "walk_forward" / SYMBOL / "tiny_price" / (PRIMARY_VENUE or "legacy")


def walk_forward_output_paths(target_spec):
    suffix = f"__{OUTPUT_SUFFIX}" if OUTPUT_SUFFIX else ""
    result_path = VENUE_DIR / f"{SYMBOL}_tiny_price_walk_forward_{target_slug(target_spec)}{suffix}.csv"
    summary_path = VENUE_DIR / f"{SYMBOL}_tiny_price_walk_forward_summary_{target_slug(target_spec)}{suffix}.csv"
    return result_path, summary_path


def read_csv(path, nrows=None):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, nrows=nrows)
    except EmptyDataError:
        return pd.DataFrame()


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
    tmp_path.replace(path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def target_horizon_seconds(target_spec):
    for part in reversed(str(target_spec).split("_")):
        if part.endswith("s") and part[:-1].isdigit():
            return int(part[:-1])
    return int(os.getenv("PRICE_TINY_HORIZON_SECONDS", "30"))


def target_method(target_spec):
    target_spec = str(target_spec).lower()
    if target_spec.startswith("move_before_adverse") and "net_aware" in target_spec:
        return "move_before_adverse_net_aware"
    if target_spec.startswith("move_before_adverse"):
        return "move_before_adverse"
    if target_spec.startswith("instability"):
        return "instability"
    if target_spec.startswith("return_bps"):
        return "return_bps"
    if target_spec.startswith("next_mid_delta_bps"):
        return "next_mid_delta_bps"
    if target_spec.startswith("next_mid_log_return"):
        return "next_mid_log_return"
    return "direction"


def selected_target_column(method, horizon):
    if method == "move_before_adverse_net_aware":
        return f"target_move_before_adverse_net_aware_{horizon}s"
    if method == "move_before_adverse":
        return f"target_move_before_adverse_{horizon}s"
    if method == "instability":
        return f"target_instability_{horizon}s"
    if method == "return_bps":
        return f"target_return_bps_{horizon}s"
    if method == "next_mid_delta_bps":
        return f"target_next_mid_delta_bps_{horizon}s"
    if method == "next_mid_log_return":
        return f"target_next_mid_log_return_{horizon}s"
    return f"target_direction_{horizon}s"


def realized_return_column(horizon):
    return f"target_return_bps_{horizon}s"


def is_regression_method(method):
    return method in {"return_bps", "next_mid_delta_bps", "next_mid_log_return"}


def target_slug(target_spec):
    return slugify(target_spec, "target")


def feature_group_matches(frame):
    if not FEATURE_GROUPS:
        return True
    requested = {value.strip().lower() for value in FEATURE_GROUPS if value.strip()}
    if not requested:
        return True
    for column in ["feature_groups", "enabled_feature_groups"]:
        if column in frame.columns and len(frame[column].dropna()):
            observed = {
                value.strip().lower()
                for value in str(frame[column].dropna().iloc[0]).replace("+", ",").split(",")
                if value.strip()
            }
            if REQUIRE_EXACT_FEATURE_GROUPS:
                return requested == observed
            return requested.issubset(observed)
    return True


def estimated_costs_for_frame(frame, active_mask):
    active_mask = np.asarray(active_mask, dtype=bool)
    if not active_mask.any():
        return np.asarray([], dtype=np.float64)
    costs = np.full(int(active_mask.sum()), ESTIMATED_FEE_BPS + ESTIMATED_SLIPPAGE_BPS, dtype=np.float64)
    if CHARGE_HALF_SPREAD and "feature_spread_percent" in frame.columns:
        spread_ratio = pd.to_numeric(frame.loc[active_mask, "feature_spread_percent"], errors="coerce").to_numpy(dtype=np.float64)
        spread_bps = np.where(np.isfinite(spread_ratio) & (spread_ratio >= 0), spread_ratio * 10000.0, 0.0)
        costs += spread_bps
    return costs


def matching_training_files(target_spec):
    slug = target_slug(target_spec)
    files = sorted(
        VENUE_DIR.glob(f"{SYMBOL}_tiny_price_training_rows__*__{slug}__*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    matches = []
    for path in files:
        sample = read_csv(path, nrows=5)
        if len(sample) == 0:
            continue
        if "target_spec_name" in sample.columns and len(sample["target_spec_name"].dropna()):
            if str(sample["target_spec_name"].dropna().iloc[0]).strip().lower() != target_spec:
                continue
        if not feature_group_matches(sample):
            continue
        matches.append(path)
    return matches


def build_training_rows(target_spec):
    if not AUTO_BUILD_ROWS:
        raise FileNotFoundError(
            f"No matching tiny-price training rows found for {target_spec}. "
            "Set PRICE_TINY_WALK_FORWARD_AUTO_BUILD=true or run npm run tiny-price-build."
        )
    env = os.environ.copy()
    env["SYMBOL"] = SYMBOL
    env["PRIMARY_VENUE"] = PRIMARY_VENUE
    env["PRICE_TINY_TARGET_SPEC"] = target_spec
    env["PRICE_TINY_HORIZON_SECONDS"] = str(target_horizon_seconds(target_spec))
    env["PRICE_TINY_FEATURE_GROUPS"] = ",".join(FEATURE_GROUPS)
    env["PROMOTE_BEST"] = "false"
    env["PRICE_TINY_AUTO_REGISTER_CHALLENGERS"] = "false"
    print(f"Building training rows for {target_spec} with feature_groups={','.join(FEATURE_GROUPS)}")
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "build_tiny_price_training_rows.py")]
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"build_tiny_price_training_rows.py failed for {target_spec} with code {result.returncode}")


def resolve_training_rows(target_spec):
    explicit = os.getenv("PRICE_TINY_WALK_FORWARD_TRAINING_PATH", "").strip() or os.getenv("PRICE_TINY_TRAINING_PATH", "").strip()
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else PROJECT_ROOT / path
    matches = matching_training_files(target_spec)
    if not matches:
        build_training_rows(target_spec)
        matches = matching_training_files(target_spec)
    if not matches:
        raise FileNotFoundError(f"No matching training rows found for {target_spec}")
    return matches[0]


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(np.clip(values, -40, 40))
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)


def direction_to_class(values):
    values = np.asarray(values, dtype=np.int64)
    return np.where(values < 0, 0, np.where(values > 0, 2, 1))


def class_to_direction(classes):
    classes = np.asarray(classes, dtype=np.int64)
    return np.where(classes == 0, -1, np.where(classes == 2, 1, 0))


def direction_from_return_bps(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values > PRICE_TINY_FLAT_BPS, 1, np.where(values < -PRICE_TINY_FLAT_BPS, -1, 0))


def confidence_from_return_bps(values, scale=None):
    values = np.asarray(values, dtype=np.float64)
    if scale is None or not np.isfinite(scale) or scale <= 0:
        scale = max(PRICE_TINY_FLAT_BPS * 8.0, 1e-6)
    return np.clip(np.abs(values) / scale, 0.0, 1.0)


def standardize(train_x, validation_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (train_x - mean) / std, (validation_x - mean) / std, (test_x - mean) / std, mean, std


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


def model_specs_for_target(method):
    if MODEL_SPECS_ENV:
        return MODEL_SPECS_ENV
    if is_regression_method(method):
        return ["ridge_regression"]
    return ["ridge_logistic"]


def prepare_frame(frame, target_spec):
    method = target_method(target_spec)
    horizon = target_horizon_seconds(target_spec)
    selected_col = selected_target_column(method, horizon)
    realized_col = realized_return_column(horizon)
    if selected_col not in frame.columns:
        raise RuntimeError(f"Missing selected target column: {selected_col}")
    if realized_col not in frame.columns:
        raise RuntimeError(f"Missing realized return column: {realized_col}")
    feature_columns = select_model_feature_columns(frame)
    assert_numeric_feature_columns(frame, feature_columns)
    required = ["timestamp", selected_col, realized_col, *feature_columns]
    if "time" in frame.columns:
        required.append("time")
    frame = frame.copy()
    for column in required:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[column for column in required if column != "time"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame, feature_columns, selected_col, realized_col, method, horizon


def target_distribution(frame, selected_col, method):
    values = pd.to_numeric(frame[selected_col], errors="coerce").dropna()
    if len(values) == 0:
        return {}
    if is_regression_method(method):
        return {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=0)),
            "p05": float(values.quantile(0.05)),
            "p95": float(values.quantile(0.95)),
            "positive": int((values > PRICE_TINY_FLAT_BPS).sum()),
            "negative": int((values < -PRICE_TINY_FLAT_BPS).sum()),
            "flat": int((values.abs() <= PRICE_TINY_FLAT_BPS).sum()),
        }
    counts = values.astype(int).value_counts().sort_index().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def majority_baseline(train, selected_col, method):
    if is_regression_method(method):
        train_direction = direction_from_return_bps(train[selected_col].to_numpy(dtype=np.float64))
    else:
        train_direction = train[selected_col].to_numpy(dtype=np.int64)
    if len(train_direction) == 0:
        return 0
    return int(pd.Series(train_direction).mode().iloc[0])


def train_window_models(train, validation, test, feature_columns, selected_col, realized_col, method, model_specs):
    train_x_raw = train[feature_columns].to_numpy(dtype=np.float64)
    validation_x_raw = validation[feature_columns].to_numpy(dtype=np.float64)
    test_x_raw = test[feature_columns].to_numpy(dtype=np.float64)
    train_x, validation_x, test_x, mean, std = standardize(train_x_raw, validation_x_raw, test_x_raw)
    models = {}
    predictions = {"validation": {}, "test": {}}
    regression_scale = max(float(np.nanstd(train[realized_col].to_numpy(dtype=np.float64))), PRICE_TINY_FLAT_BPS * 8.0)

    if "ridge_regression" in model_specs and is_regression_method(method):
        weights, bias = train_ridge(train_x, train[selected_col].to_numpy(dtype=np.float64))
        models["ridge_regression"] = {"weights": weights, "bias": bias, "feature_mean": mean, "feature_std": std}
        for split_name, x in [("validation", validation_x), ("test", test_x)]:
            pred_return = x @ weights + bias
            predictions[split_name]["ridge_regression"] = {
                "pred_return_bps": pred_return,
                "pred_direction": direction_from_return_bps(pred_return),
                "confidence": confidence_from_return_bps(pred_return, regression_scale),
            }

    wants_logistic = any(spec in {"ridge_logistic", "logistic_regression"} for spec in model_specs)
    if wants_logistic and not is_regression_method(method):
        classes = direction_to_class(train[selected_col].to_numpy(dtype=np.int64))
        weights, bias = train_softmax_logistic(train_x, classes)
        models["ridge_logistic"] = {"weights": weights, "bias": bias, "feature_mean": mean, "feature_std": std}
        for split_name, x in [("validation", validation_x), ("test", test_x)]:
            probs = softmax(x @ weights + bias)
            predictions[split_name]["ridge_logistic"] = {
                "pred_return_bps": np.zeros(len(x), dtype=np.float64),
                "pred_direction": class_to_direction(np.argmax(probs, axis=1)),
                "confidence": probs.max(axis=1),
                "prob_down": probs[:, 0],
                "prob_flat": probs[:, 1],
                "prob_up": probs[:, 2],
            }

    unsupported = [
        spec for spec in model_specs
        if spec not in {"ridge_regression", "ridge_logistic", "logistic_regression"}
    ]
    if unsupported:
        print(f"WARNING: unsupported walk-forward model specs skipped: {unsupported}")
    if not models:
        raise RuntimeError(f"No trainable model specs for target method={method}: requested={model_specs}")
    return models, predictions


def actual_direction(frame, selected_col, method):
    if is_regression_method(method):
        return direction_from_return_bps(frame[selected_col].to_numpy(dtype=np.float64))
    return frame[selected_col].to_numpy(dtype=np.int64)


def evaluate_predictions(frame, train, selected_col, realized_col, method, prediction, threshold):
    realized_return = frame[realized_col].to_numpy(dtype=np.float64)
    actual_dir = actual_direction(frame, selected_col, method)
    pred_dir = np.asarray(prediction["pred_direction"], dtype=np.int64)
    confidence = np.asarray(prediction["confidence"], dtype=np.float64)
    active = (confidence >= threshold) & (pred_dir != 0)
    majority = majority_baseline(train, selected_col, method)
    majority_accuracy = float((np.full(len(frame), majority) == actual_dir).mean()) if len(frame) else np.nan
    sign_accuracy_all = float((pred_dir == actual_dir).mean()) if len(frame) else np.nan

    if active.any() and method != "instability":
        strategy_return = realized_return[active] * np.sign(pred_dir[active])
        estimated_costs = estimated_costs_for_frame(frame, active)
        net_strategy_return = strategy_return - estimated_costs
        sign_accuracy = float((pred_dir[active] == actual_dir[active]).mean())
        avg_strategy_return = float(strategy_return.mean())
        estimated_net_avg_strategy_return = float(net_strategy_return.mean()) if len(net_strategy_return) else np.nan
        average_estimated_cost = float(estimated_costs.mean()) if len(estimated_costs) else np.nan
    elif active.any() and method == "instability":
        strategy_return = np.asarray([], dtype=np.float64)
        sign_accuracy = float(((pred_dir[active] > 0) == (actual_dir[active] > 0)).mean())
        avg_strategy_return = np.nan
        estimated_net_avg_strategy_return = np.nan
        average_estimated_cost = np.nan
    else:
        sign_accuracy = np.nan
        avg_strategy_return = np.nan
        estimated_net_avg_strategy_return = np.nan
        average_estimated_cost = np.nan

    pred_return = np.asarray(prediction.get("pred_return_bps", np.zeros(len(frame))), dtype=np.float64)
    mae = float(np.mean(np.abs(pred_return - realized_return))) if is_regression_method(method) and len(frame) else np.nan
    rmse = float(np.sqrt(np.mean((pred_return - realized_return) ** 2))) if is_regression_method(method) and len(frame) else np.nan
    zero_mae = float(np.mean(np.abs(realized_return))) if is_regression_method(method) and len(frame) else np.nan
    correlation = np.nan
    if is_regression_method(method) and len(frame) > 2 and np.nanstd(pred_return) > 1e-12 and np.nanstd(realized_return) > 1e-12:
        correlation = float(np.corrcoef(pred_return, realized_return)[0, 1])

    predicted_up = int((pred_dir > 0).sum())
    predicted_down = int((pred_dir < 0).sum())
    one_sided = bool((predicted_up == 0 or predicted_down == 0) and (predicted_up + predicted_down) > 0)
    active_rows = int(active.sum())
    active_long_count = int(((pred_dir > 0) & active).sum())
    active_short_count = int(((pred_dir < 0) & active).sum())
    failure_reasons = []
    if active_rows < MIN_ACTIVE_ROWS:
        failure_reasons.append("too_few_active_rows")
    if method != "instability" and np.isfinite(avg_strategy_return) and avg_strategy_return <= 0:
        failure_reasons.append("non_positive_avg_strategy_return")
    if np.isfinite(sign_accuracy) and np.isfinite(majority_accuracy) and sign_accuracy <= majority_accuracy:
        failure_reasons.append("does_not_beat_majority")
    if one_sided:
        failure_reasons.append("one_sided_predictions")
    if is_regression_method(method):
        if np.isfinite(mae) and np.isfinite(zero_mae) and mae >= zero_mae:
            failure_reasons.append("mae_not_better_than_zero_return")
        if np.isfinite(correlation) and correlation <= 0:
            failure_reasons.append("non_positive_correlation")
    if method == "instability":
        failure_reasons.append("risk_target_not_directional_strategy")

    return {
        "active_rows": active_rows,
        "coverage": float(active_rows / len(frame)) if len(frame) else 0.0,
        "sign_accuracy": sign_accuracy,
        "sign_accuracy_all_rows": sign_accuracy_all,
        "majority_baseline": majority_accuracy,
        "avg_strategy_return_bps": avg_strategy_return,
        "estimated_net_avg_strategy_return_bps": estimated_net_avg_strategy_return,
        "average_estimated_cost_bps": average_estimated_cost,
        "mae_bps": mae,
        "rmse_bps": rmse,
        "zero_return_baseline_mae_bps": zero_mae,
        "correlation": correlation,
        "predicted_up_count": predicted_up,
        "predicted_down_count": predicted_down,
        "active_long_count": active_long_count,
        "active_short_count": active_short_count,
        "one_sided_prediction_warning": one_sided,
        "failure_reason": ";".join(failure_reasons),
        "target_role_note": "risk/gating target; strategy return is descriptive only" if method == "instability" else "",
    }


def confidence_threshold_table(frame, train, selected_col, realized_col, method, prediction):
    rows = []
    for threshold in CONFIDENCE_THRESHOLDS:
        metrics = evaluate_predictions(frame, train, selected_col, realized_col, method, prediction, threshold)
        rows.append(
            {
                "threshold": threshold,
                "active_rows": metrics["active_rows"],
                "sign_accuracy": metrics["sign_accuracy"],
                "avg_strategy_return_bps": metrics["avg_strategy_return_bps"],
                "coverage": metrics["coverage"],
            }
        )
    return rows


def select_model_on_validation(validation, train, selected_col, realized_col, method, validation_predictions):
    candidates = []
    for model_name, prediction in validation_predictions.items():
        for threshold in CONFIDENCE_THRESHOLDS:
            metrics = evaluate_predictions(validation, train, selected_col, realized_col, method, prediction, threshold)
            if is_regression_method(method):
                score = (
                    -metrics["mae_bps"] if np.isfinite(metrics["mae_bps"]) else -1e9,
                    metrics["correlation"] if np.isfinite(metrics["correlation"]) else -1e9,
                    metrics["active_rows"],
                )
            elif method == "instability":
                score = (
                    metrics["sign_accuracy"] if np.isfinite(metrics["sign_accuracy"]) else -1e9,
                    metrics["active_rows"],
                )
            else:
                score = (
                    metrics["avg_strategy_return_bps"] if np.isfinite(metrics["avg_strategy_return_bps"]) else -1e9,
                    metrics["sign_accuracy"] if np.isfinite(metrics["sign_accuracy"]) else -1e9,
                    metrics["active_rows"],
                )
            candidates.append((score, model_name, threshold, metrics))
    if not candidates:
        raise RuntimeError("No validation candidates produced.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2], candidates[0][3]


def window_regime_summary(frame):
    def mean_col(*names):
        for name in names:
            if name in frame.columns:
                values = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
                if values.notna().any():
                    return float(values.mean())
        return np.nan

    return {
        "avg_spread_percent": mean_col("feature_spread_percent"),
        "avg_rolling_volatility_30s": mean_col("feature_rolling_volatility_30s", "feature_regime_realized_volatility_60s"),
        "avg_rolling_volatility_60s": mean_col("feature_rolling_volatility_60s", "feature_regime_realized_volatility_60s"),
        "avg_rolling_volatility_120s": mean_col("feature_rolling_volatility_120s"),
        "avg_range_30s": mean_col("feature_recent_high_low_range_30s", "feature_regime_high_low_range_60s"),
        "avg_range_60s": mean_col("feature_recent_high_low_range_60s", "feature_regime_high_low_range_60s"),
        "avg_range_120s": mean_col("feature_recent_high_low_range_120s"),
    }


def timestamp_at(frame, index):
    return int(frame["timestamp"].iloc[index])


def save_walk_forward_artifact(target_spec, window_index, model_name, model, feature_columns, selected_col, realized_col, method, threshold, metrics):
    artifact_dir = WALK_FORWARD_MODEL_ROOT / target_slug(target_spec)
    artifact_path = artifact_dir / f"window_{window_index:04d}_{model_name}.json"
    serializable_model = {}
    for key, value in model.items():
        if isinstance(value, np.ndarray):
            serializable_model[key] = value.tolist()
        else:
            serializable_model[key] = value
    payload = {
        "artifact_type": "paper_only_tiny_price_walk_forward",
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE,
        "target_spec": target_spec,
        "target_method": method,
        "selected_target_column": selected_col,
        "realized_return_column": realized_col,
        "model_name": model_name,
        "selected_threshold": threshold,
        "feature_columns": feature_columns,
        "feature_schema_hash": feature_schema_hash(feature_columns),
        "validation_metrics": metrics,
        "model": serializable_model,
        "paper_only": True,
        "created_at_unix_ms": int(time.time() * 1000),
    }
    atomic_write_json(payload, artifact_path)
    return artifact_path


def run_target(target_spec):
    training_path = resolve_training_rows(target_spec)
    print(f"Walk-forward target={target_spec} training_rows={training_path}")
    raw = read_csv(training_path)
    frame, feature_columns, selected_col, realized_col, method, horizon = prepare_frame(raw, target_spec)
    model_specs = model_specs_for_target(method)
    total_needed = TRAIN_ROWS + VALIDATION_ROWS + max(TEST_ROWS, MIN_TEST_ROWS)
    if len(frame) < total_needed:
        raise RuntimeError(
            f"Not enough rows for {target_spec}: {len(frame)} < {total_needed}. "
            "Reduce PRICE_TINY_WALK_FORWARD_*_ROWS or collect/build more rows."
        )

    rows = []
    window_index = 0
    start = 0
    while start + TRAIN_ROWS + VALIDATION_ROWS + MIN_TEST_ROWS <= len(frame):
        test_size = min(TEST_ROWS, len(frame) - (start + TRAIN_ROWS + VALIDATION_ROWS))
        if test_size < MIN_TEST_ROWS:
            break
        train = frame.iloc[start : start + TRAIN_ROWS].copy()
        validation = frame.iloc[start + TRAIN_ROWS : start + TRAIN_ROWS + VALIDATION_ROWS].copy()
        test_start = start + TRAIN_ROWS + VALIDATION_ROWS
        test = frame.iloc[test_start : test_start + test_size].copy()
        models, predictions = train_window_models(train, validation, test, feature_columns, selected_col, realized_col, method, model_specs)
        selected_model, selected_threshold, validation_metrics = select_model_on_validation(
            validation,
            train,
            selected_col,
            realized_col,
            method,
            predictions["validation"],
        )
        test_prediction = predictions["test"][selected_model]
        test_metrics = evaluate_predictions(test, train, selected_col, realized_col, method, test_prediction, selected_threshold)
        table = confidence_threshold_table(test, train, selected_col, realized_col, method, test_prediction)
        artifact_path = save_walk_forward_artifact(
            target_spec,
            window_index,
            selected_model,
            models[selected_model],
            feature_columns,
            selected_col,
            realized_col,
            method,
            selected_threshold,
            validation_metrics,
        )
        row = {
            "target_spec": target_spec,
            "target_method": method,
            "window_index": window_index,
            "train_start_timestamp": timestamp_at(train, 0),
            "train_end_timestamp": timestamp_at(train, len(train) - 1),
            "validation_start_timestamp": timestamp_at(validation, 0),
            "validation_end_timestamp": timestamp_at(validation, len(validation) - 1),
            "test_start_timestamp": timestamp_at(test, 0),
            "test_end_timestamp": timestamp_at(test, len(test) - 1),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "feature_count": int(len(feature_columns)),
            "feature_schema_hash": feature_schema_hash(feature_columns),
            "selected_target_column": selected_col,
            "realized_return_column": realized_col,
            "selected_model": selected_model,
            "selected_threshold": selected_threshold,
            "train_target_distribution": json.dumps(target_distribution(train, selected_col, method), sort_keys=True),
            "validation_target_distribution": json.dumps(target_distribution(validation, selected_col, method), sort_keys=True),
            "test_target_distribution": json.dumps(target_distribution(test, selected_col, method), sort_keys=True),
            "validation_selection_metrics": json.dumps(validation_metrics, sort_keys=True),
            "confidence_threshold_table": json.dumps(table, sort_keys=True),
            "walk_forward_artifact_path": str(artifact_path),
            **test_metrics,
            **window_regime_summary(test),
        }
        rows.append(row)
        print(
            f"target={target_spec} window={window_index} model={selected_model} "
            f"threshold={selected_threshold:.2f} active={test_metrics['active_rows']} "
            f"sign_acc={test_metrics['sign_accuracy']} avg_return={test_metrics['avg_strategy_return_bps']}"
        )
        window_index += 1
        if MAX_WINDOWS and window_index >= MAX_WINDOWS:
            break
        start += STEP_ROWS

    result_path, summary_path = walk_forward_output_paths(target_spec)
    write_csv(rows, result_path)
    write_csv([summary_for_target(target_spec, rows, training_path)], summary_path)
    print(f"Walk-forward rows: {result_path}")
    print(f"Walk-forward summary: {summary_path}")
    return result_path, summary_path


def summary_for_target(target_spec, rows, training_path):
    if not rows:
        return {
            "symbol": SYMBOL,
            "primary_venue": PRIMARY_VENUE,
            "target_spec": target_spec,
            "training_rows_path": str(training_path),
            "windows": 0,
            "paper_only": True,
        }
    frame = pd.DataFrame(rows)
    avg_return = pd.to_numeric(frame["avg_strategy_return_bps"], errors="coerce")
    sign_accuracy = pd.to_numeric(frame["sign_accuracy"], errors="coerce")
    majority = pd.to_numeric(frame["majority_baseline"], errors="coerce")
    useful = frame["failure_reason"].fillna("").astype(str).str.len() == 0
    return {
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE,
        "target_spec": target_spec,
        "training_rows_path": str(training_path),
        "windows": int(len(frame)),
        "train_rows": TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "test_rows": TEST_ROWS,
        "step_rows": STEP_ROWS,
        "mean_sign_accuracy": float(sign_accuracy.mean()) if sign_accuracy.notna().any() else np.nan,
        "mean_majority_baseline": float(majority.mean()) if majority.notna().any() else np.nan,
        "mean_avg_strategy_return_bps": float(avg_return.mean()) if avg_return.notna().any() else np.nan,
        "median_avg_strategy_return_bps": float(avg_return.median()) if avg_return.notna().any() else np.nan,
        "positive_return_windows": int((avg_return > 0).sum()) if avg_return.notna().any() else 0,
        "percent_positive_return_windows": float((avg_return > 0).mean()) if avg_return.notna().any() else np.nan,
        "windows_beating_majority": int((sign_accuracy > majority).sum()) if sign_accuracy.notna().any() and majority.notna().any() else 0,
        "percent_windows_beating_majority": float((sign_accuracy > majority).mean()) if sign_accuracy.notna().any() and majority.notna().any() else np.nan,
        "useful_windows": int(useful.sum()),
        "percent_useful_windows": float(useful.mean()),
        "paper_only": True,
        "no_promotion": True,
    }


def main():
    print("Tiny-price walk-forward evaluator")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Targets: {', '.join(TARGET_SPECS)}")
    print(f"Feature groups: {', '.join(FEATURE_GROUPS)}")
    print(f"Model specs: {', '.join(MODEL_SPECS_ENV) if MODEL_SPECS_ENV else '(target defaults)'}")
    print(f"Windows: train={TRAIN_ROWS} validation={VALIDATION_ROWS} test={TEST_ROWS} step={STEP_ROWS}")
    for target_spec in TARGET_SPECS:
        run_target(target_spec)
    print("Research/evaluation only. No challenger registration, promotion, orders, or live prediction writes.")


if __name__ == "__main__":
    main()
