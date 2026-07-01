import os
from pathlib import Path

import numpy as np
import pandas as pd

from microstructure_model_utils import (
    DIAGNOSTIC_TARGET_COLUMNS,
    EVENT_TARGET_COLUMNS,
    REGRESSION_TARGET_COLUMNS,
    atomic_write_csv,
    copy_if_promoted,
    current_utc_tag,
    feature_schema_hash,
    forward,
    get_micro_feature_columns,
    initialize_model,
    is_optional_context_feature,
    load_model,
    percent,
    precision_recall,
    predictions_frame,
    predict_with_artifact,
    regression_clip_bounds,
    regression_scalers_from_arrays,
    save_model,
    standardize,
    required_micro_feature_columns,
)
from hierarchical_context import print_context_availability_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.80"))
EMBARGO_ROWS = int(os.getenv("EMBARGO_ROWS", "60"))
EPOCHS = int(os.getenv("EPOCHS", "40"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
HIDDEN_UNITS = int(os.getenv("HIDDEN_UNITS", "64"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))
REGRESSION_LOSS_WEIGHT = float(os.getenv("REGRESSION_LOSS_WEIGHT", "0.25"))
EVENT_LOSS_WEIGHT = float(os.getenv("EVENT_LOSS_WEIGHT", "1.0"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "200"))
MIN_VALIDATION_ROWS = int(os.getenv("MIN_VALIDATION_ROWS", "50"))
PROMOTE_BEST = os.getenv("PROMOTE_BEST", "false").strip().lower() in {"1", "true", "yes", "y"}
MIN_SCARE_RECALL = float(os.getenv("MIN_SCARE_RECALL", "0.05"))
EVENT_TARGET_MIN_TRAIN_POSITIVES = int(os.getenv("EVENT_TARGET_MIN_TRAIN_POSITIVES", "50"))
EVENT_TARGET_MIN_VALIDATION_POSITIVES = int(os.getenv("EVENT_TARGET_MIN_VALIDATION_POSITIVES", "20"))
MICRO_RUN_FLOW_1S_ABLATION = os.getenv("MICRO_RUN_FLOW_1S_ABLATION", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
MICRO_ALLOW_GROUP_DISAGREEMENT = os.getenv("MICRO_ALLOW_GROUP_DISAGREEMENT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
MICRO_MAX_ABS_RETURN_TARGET = float(os.getenv("MICRO_MAX_ABS_RETURN_TARGET", "0.02"))
MICRO_MAX_ABS_SPREAD_TARGET = float(os.getenv("MICRO_MAX_ABS_SPREAD_TARGET", "0.01"))
MICRO_MAX_ABS_LOG_DEPTH_TARGET = float(os.getenv("MICRO_MAX_ABS_LOG_DEPTH_TARGET", "5.0"))
MICRO_TRAIN_EVENT_ONLY = os.getenv("MICRO_TRAIN_EVENT_ONLY", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
TRAINING_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_10s_microstructure_training_rows.csv"
PREDICTIONS_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_10s_microstructure_predictions.csv"
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "microstructure_10s"
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "active" / SYMBOL / "microstructure_10s" / "model.json"

EVENT_TARGET_GROUPS = {
    "scare_targets": [
        "upside_scare_event_10s",
        "downside_scare_event_10s",
    ],
    "burst_targets": [
        "aggressive_buy_burst_10s",
        "aggressive_sell_burst_10s",
    ],
    "liquidity_targets": [
        "bid_liquidity_drop_10s",
        "ask_liquidity_drop_10s",
        "spread_expansion_event_10s",
    ],
    "path_targets": [
        "continuation_30s",
        "continuation_60s",
        "direction_flip_10s",
    ],
}


def fill_optional_context_defaults(frame):
    frame = frame.copy()
    for column in frame.columns:
        if not is_optional_context_feature(column):
            continue
        if column.endswith("_context_available"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        elif column.endswith("_context_age_ms"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(-1.0)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def print_nan_drop_diagnostics(frame, required_columns, title, top_n=12):
    if len(frame) == 0:
        print(f"{title}: frame is empty")
        return
    counts = frame[required_columns].isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False).head(top_n)
    print(title)
    if len(counts) == 0:
        print("- no NaN drops from required columns")
        return
    for column, count in counts.items():
        print(f"- {column}: {int(count)}")


def target_bound_for_column(column):
    if column == "future_spread_expansion_10s":
        return MICRO_MAX_ABS_SPREAD_TARGET
    if "log_depth_change" in column:
        return MICRO_MAX_ABS_LOG_DEPTH_TARGET
    if column in {
        "future_return_10s",
        "max_runup_10s",
        "max_drawdown_10s",
        "upside_velocity_10s",
        "downside_velocity_10s",
    }:
        return MICRO_MAX_ABS_RETURN_TARGET
    return None


def print_regression_target_stats(frame, title):
    if MICRO_TRAIN_EVENT_ONLY:
        print(f"\n{title}")
        print("- skipped because MICRO_TRAIN_EVENT_ONLY=true")
        return
    print(f"\n{title}")
    for column in REGRESSION_TARGET_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            raise ValueError(f"Non-finite regression target found after filtering: {column}")
        print(
            f"- {column}: "
            f"min={values.min():.8g}, "
            f"p01={values.quantile(0.01):.8g}, "
            f"median={values.median():.8g}, "
            f"mean={values.mean():.8g}, "
            f"p99={values.quantile(0.99):.8g}, "
            f"max={values.max():.8g}, "
            f"std={values.std(ddof=0):.8g}"
        )


def apply_regression_target_bounds(frame):
    if MICRO_TRAIN_EVENT_ONLY:
        print("\nRegression target bounds")
        print("- skipped because MICRO_TRAIN_EVENT_ONLY=true")
        return frame.reset_index(drop=True), {}
    frame = frame.copy()
    keep = pd.Series(True, index=frame.index)
    dropped_counts = {}
    for column in REGRESSION_TARGET_COLUMNS:
        bound = target_bound_for_column(column)
        if bound is None:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values) | (values.abs() > bound)
        dropped_counts[column] = int(invalid.sum())
        keep &= ~invalid
    filtered = frame[keep].reset_index(drop=True)
    print("\nRegression target bounds")
    print(f"- MICRO_MAX_ABS_RETURN_TARGET: {MICRO_MAX_ABS_RETURN_TARGET:.4%}")
    print(f"- MICRO_MAX_ABS_SPREAD_TARGET: {MICRO_MAX_ABS_SPREAD_TARGET:.4%}")
    print(f"- MICRO_MAX_ABS_LOG_DEPTH_TARGET: {MICRO_MAX_ABS_LOG_DEPTH_TARGET}")
    print(f"- rows before target bounds: {len(frame)}")
    print(f"- rows after target bounds: {len(filtered)}")
    print("- dropped rows by target:")
    for column, count in dropped_counts.items():
        print(f"  - {column}: {count}")
    if len(filtered) == 0:
        raise ValueError("All rows were dropped by micro regression target bounds.")
    return filtered, dropped_counts


def load_training_rows():
    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            f"Missing training rows: {TRAINING_PATH}. "
            "Run scripts/build_10s_microstructure_training_rows.py first."
        )
    frame = pd.read_csv(TRAINING_PATH)
    raw_row_count = len(frame)
    raw_feature_ready_count = (
        int(frame["feature_ready"].astype(str).str.lower().isin({"true", "1", "1.0"}).sum())
        if "feature_ready" in frame.columns
        else raw_row_count
    )

    for column in frame.columns:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    rows_after_numeric_conversion = int(frame.dropna(subset=["timestamp"]).shape[0]) if "timestamp" in frame.columns else 0

    if "feature_ready" in frame.columns:
        frame = frame[frame["feature_ready"].fillna(0.0).astype(float) > 0.0].copy()
    rows_after_feature_ready_filter = len(frame)

    frame = fill_optional_context_defaults(frame)
    required = ["timestamp", *EVENT_TARGET_COLUMNS]
    if not MICRO_TRAIN_EVENT_ONLY:
        required.extend(REGRESSION_TARGET_COLUMNS)
    missing_targets = [column for column in required if column not in frame.columns]
    if missing_targets:
        raise ValueError(
            "Training rows are missing required target columns. "
            "Run npm run micro-build before npm run micro-train. "
            f"Missing: {missing_targets[:20]}"
        )
    columns = get_micro_feature_columns(frame)
    required_micro_features = required_micro_feature_columns(columns)
    required.extend(required_micro_features)

    target_required = ["timestamp", *EVENT_TARGET_COLUMNS]
    if not MICRO_TRAIN_EVENT_ONLY:
        target_required.extend(REGRESSION_TARGET_COLUMNS)
    target_filtered = frame.dropna(subset=target_required).copy()
    rows_after_target_filtering = len(target_filtered)

    print("\n10s training row load diagnostics")
    print(f"- raw row count: {raw_row_count}")
    print(f"- rows with feature_ready=true before numeric conversion: {raw_feature_ready_count}")
    print(f"- rows after numeric conversion with valid timestamp: {rows_after_numeric_conversion}")
    print(f"- rows after feature_ready filter: {rows_after_feature_ready_filter}")
    print(f"- rows after target filtering: {rows_after_target_filtering}")
    print(f"- required micro feature columns: {len(required_micro_features)}")
    print(f"- optional context feature columns default-filled: {len(columns) - len(required_micro_features)}")
    print_nan_drop_diagnostics(frame, target_required, "- top columns causing target NaN drops")
    print_nan_drop_diagnostics(target_filtered, required_micro_features, "- top columns causing micro feature NaN drops")

    frame = target_filtered.dropna(subset=required_micro_features).sort_values("timestamp").reset_index(drop=True)
    frame, dropped_counts = apply_regression_target_bounds(frame)
    print_regression_target_stats(frame, "Regression target stats after final filtering")
    print(f"- final usable rows: {len(frame)}")
    return frame, columns


def split_time_ordered(frame):
    split_index = int(len(frame) * TRAIN_SPLIT)
    split_index = max(1, min(split_index, len(frame) - 1))
    validation_start = min(len(frame), split_index + EMBARGO_ROWS)
    dropped = max(0, validation_start - split_index)
    return frame.iloc[:split_index].copy(), frame.iloc[validation_start:].copy(), dropped


def event_class_weights(y_events):
    positives = y_events.sum(axis=0)
    negatives = len(y_events) - positives
    positive_weights = negatives / np.maximum(positives, 1.0)
    positive_weights = np.clip(positive_weights, 1.0, 50.0)
    return positive_weights


def standardize_regression_targets(train_values, validation_values, regression_columns):
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    for index, column in enumerate(regression_columns):
        if not np.isfinite(mean[index]) or not np.isfinite(std[index]) or std[index] <= 0:
            raise ValueError(
                f"Invalid regression target scaler for {column}: "
                f"mean={mean[index]}, std={std[index]}"
            )
    return (train_values - mean) / std, (validation_values - mean) / std, mean, std


def train_model(
    x_train,
    y_reg_train,
    y_event_train,
    x_validation,
    y_reg_validation,
    y_event_validation,
    active_event_mask,
):
    rng = np.random.default_rng(RANDOM_SEED)
    regression_output_count = int(y_reg_train.shape[1]) if y_reg_train.ndim == 2 else 0
    model = initialize_model(
        x_train.shape[1],
        HIDDEN_UNITS,
        regression_output_count,
        y_event_train.shape[1],
        rng,
    )
    positive_weights = event_class_weights(y_event_train)
    active_event_mask = np.asarray(active_event_mask, dtype=np.float64).reshape(1, -1)
    active_event_count = max(1.0, float(active_event_mask.sum()))
    best_model = {name: value.copy() for name, value in model.items()}
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        for start in range(0, len(x_train), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(x_train))
            xb = x_train[start:end]
            yrb = y_reg_train[start:end]
            yeb = y_event_train[start:end]

            hidden_pre, hidden, regression, event_probabilities = forward(model, xb)
            if regression_output_count:
                d_reg = (
                    REGRESSION_LOSS_WEIGHT
                    * 2.0
                    * (regression - yrb)
                    / max(1, len(xb) * yrb.shape[1])
                )
            else:
                d_reg = np.zeros_like(regression)
            event_weights = np.where(yeb == 1.0, positive_weights, 1.0)
            d_event = (
                EVENT_LOSS_WEIGHT
                * (event_probabilities - yeb)
                * event_weights
                * active_event_mask
                / max(1.0, len(xb) * active_event_count)
            )

            gradients = {
                "w_event": hidden.T @ d_event,
                "b_event": d_event.sum(axis=0),
            }
            if regression_output_count:
                gradients["w_reg"] = hidden.T @ d_reg
                gradients["b_reg"] = d_reg.sum(axis=0)
            d_hidden = d_reg @ model["w_reg"].T + d_event @ model["w_event"].T
            d_hidden[hidden_pre <= 0] = 0.0
            gradients["w1"] = xb.T @ d_hidden
            gradients["b1"] = d_hidden.sum(axis=0)

            for name, gradient in gradients.items():
                model[name] -= LEARNING_RATE * gradient

        _, _, validation_regression, validation_events = forward(model, x_validation)
        reg_loss = (
            float(np.mean((validation_regression - y_reg_validation) ** 2))
            if regression_output_count
            else 0.0
        )
        event_bce_matrix = -(
            y_event_validation * np.log(np.clip(validation_events, 1e-8, 1.0))
            + (1.0 - y_event_validation) * np.log(np.clip(1.0 - validation_events, 1e-8, 1.0))
        )
        event_loss = float(
            (event_bce_matrix * active_event_mask).sum()
            / max(1.0, len(y_event_validation) * active_event_count)
        )
        total_loss = (
            (REGRESSION_LOSS_WEIGHT * reg_loss if regression_output_count else 0.0)
            + EVENT_LOSS_WEIGHT * event_loss
        )
        if total_loss < best_loss:
            best_loss = total_loss
            best_model = {name: value.copy() for name, value in model.items()}

        if epoch == 1 or epoch == EPOCHS or epoch % 10 == 0:
            print(
                f"epoch {epoch:03d} | validation loss {total_loss:.6f} "
                f"event BCE {event_loss:.6f} regression MSE {reg_loss:.6f}"
            )

    return best_model


def event_support_table(train, validation):
    rows = []
    active_columns = []
    for column in EVENT_TARGET_COLUMNS:
        train_positives = int(pd.to_numeric(train[column], errors="coerce").fillna(0).sum())
        validation_positives = int(pd.to_numeric(validation[column], errors="coerce").fillna(0).sum())
        supported = (
            train_positives >= EVENT_TARGET_MIN_TRAIN_POSITIVES
            and validation_positives >= EVENT_TARGET_MIN_VALIDATION_POSITIVES
        )
        if supported:
            active_columns.append(column)
        rows.append(
            {
                "event_target": column,
                "train_positives": train_positives,
                "validation_positives": validation_positives,
                "used_in_event_loss": supported,
                "validation_metric_status": "ok" if supported else "insufficient_support",
            }
        )
    return pd.DataFrame(rows), active_columns


def print_event_support(support_frame):
    print("\nEvent support gating")
    print(f"EVENT_TARGET_MIN_TRAIN_POSITIVES: {EVENT_TARGET_MIN_TRAIN_POSITIVES}")
    print(f"EVENT_TARGET_MIN_VALIDATION_POSITIVES: {EVENT_TARGET_MIN_VALIDATION_POSITIVES}")
    for _, row in support_frame.iterrows():
        status = "used" if bool(row["used_in_event_loss"]) else "excluded_from_loss"
        print(
            f"- {row['event_target']}: train_pos={int(row['train_positives'])}, "
            f"validation_pos={int(row['validation_positives'])}, {status}, "
            f"metrics={row['validation_metric_status']}"
        )


def event_bce_loss(actual, probability, active_event_columns=None):
    active_event_columns = (
        list(EVENT_TARGET_COLUMNS)
        if active_event_columns is None
        else list(active_event_columns)
    )
    if not active_event_columns:
        return 0.0
    losses = []
    for column in active_event_columns:
        y = np.asarray(actual[column], dtype=np.float64)
        p = np.asarray(probability[f"prob_{column}"], dtype=np.float64)
        losses.append(
            -np.mean(
                y * np.log(np.clip(p, 1e-8, 1.0))
                + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-8, 1.0))
            )
        )
    return float(np.mean(losses)) if losses else 0.0


def event_bce_for_target(actual, probability, column):
    if column not in actual.columns or f"prob_{column}" not in probability.columns:
        return np.nan
    y = np.asarray(actual[column], dtype=np.float64)
    p = np.asarray(probability[f"prob_{column}"], dtype=np.float64)
    return float(
        -np.mean(
            y * np.log(np.clip(p, 1e-8, 1.0))
            + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-8, 1.0))
        )
    )


def regression_mse(validation, prediction_frame, regression_columns=None):
    regression_columns = list(regression_columns or REGRESSION_TARGET_COLUMNS)
    losses = []
    for target in regression_columns:
        pred_column = f"pred_{target}"
        if target not in validation.columns or pred_column not in prediction_frame.columns:
            continue
        pred = prediction_frame[pred_column].to_numpy(dtype=np.float64)
        actual = validation[target].to_numpy(dtype=np.float64)
        losses.append(np.mean((pred - actual) ** 2))
    return float(np.mean(losses)) if losses else 0.0


def regression_mae_mean(validation, prediction_frame, regression_columns=None):
    regression_columns = list(regression_columns or REGRESSION_TARGET_COLUMNS)
    values = []
    for target in regression_columns:
        pred_column = f"pred_{target}"
        if target not in validation.columns or pred_column not in prediction_frame.columns:
            continue
        pred = prediction_frame[pred_column].to_numpy(dtype=np.float64)
        actual = validation[target].to_numpy(dtype=np.float64)
        values.append(np.mean(np.abs(pred - actual)))
    return float(np.mean(values)) if values else np.nan


def top_bucket_rows(validation, prediction_frame, event, fraction):
    probability = pd.to_numeric(prediction_frame[f"prob_{event}"], errors="coerce").fillna(0.0)
    count = max(1, int(np.ceil(len(validation) * fraction)))
    return validation.assign(_probability=probability).sort_values("_probability", ascending=False).head(count)


def top_bucket_enrichment(validation, prediction_frame, event, fraction):
    if len(validation) == 0:
        return {"rows": 0, "bucket_rate": 0.0, "baseline_rate": 0.0, "enrichment": 0.0}
    subset = top_bucket_rows(validation, prediction_frame, event, fraction)
    baseline_rate = float(pd.to_numeric(validation[event], errors="coerce").fillna(0.0).mean())
    bucket_rate = float(pd.to_numeric(subset[event], errors="coerce").fillna(0.0).mean())
    enrichment = bucket_rate / baseline_rate if baseline_rate > 0 else np.nan
    return {
        "rows": int(len(subset)),
        "bucket_rate": bucket_rate,
        "baseline_rate": baseline_rate,
        "enrichment": float(enrichment) if np.isfinite(enrichment) else np.nan,
    }


def extra_bucket_comparison(validation, subset, event):
    comparisons = {
        "upside_scare_event_10s": ("max_runup_10s", "avg max_runup"),
        "downside_scare_event_10s": ("max_drawdown_10s", "avg max_drawdown"),
        "bid_liquidity_drop_10s": ("future_bid_log_depth_change_10s", "avg bid log-depth-change"),
        "ask_liquidity_drop_10s": ("future_ask_log_depth_change_10s", "avg ask log-depth-change"),
        "spread_expansion_event_10s": ("future_spread_expansion_10s", "avg spread expansion"),
    }
    if event not in comparisons:
        return None
    column, label = comparisons[event]
    if column not in validation.columns or column not in subset.columns:
        return None
    return {
        "label": label,
        "bucket_average": float(pd.to_numeric(subset[column], errors="coerce").mean()),
        "overall_average": float(pd.to_numeric(validation[column], errors="coerce").mean()),
    }


def print_top_bucket_diagnostics(validation, prediction_frame):
    print("\nPer-target top-bucket diagnostics")
    for event in EVENT_TARGET_COLUMNS:
        if f"prob_{event}" not in prediction_frame.columns or event not in validation.columns:
            continue
        print(f"- {event}")
        for fraction in [0.01, 0.05, 0.10]:
            metrics = top_bucket_enrichment(validation, prediction_frame, event, fraction)
            subset = top_bucket_rows(validation, prediction_frame, event, fraction)
            extra = extra_bucket_comparison(validation, subset, event)
            line = (
                f"  top {fraction:.0%}: rows={metrics['rows']}, "
                f"actual_rate={percent(metrics['bucket_rate'])}, "
                f"baseline={percent(metrics['baseline_rate'])}, "
                f"enrichment={metrics['enrichment']:.3g}"
                if np.isfinite(metrics["enrichment"])
                else (
                    f"  top {fraction:.0%}: rows={metrics['rows']}, "
                    f"actual_rate={percent(metrics['bucket_rate'])}, "
                    f"baseline={percent(metrics['baseline_rate'])}, enrichment=n/a"
                )
            )
            if extra is not None:
                line += (
                    f", {extra['label']} bucket={extra['bucket_average']:.6g}, "
                    f"overall={extra['overall_average']:.6g}"
                )
            print(line)


def print_event_probability_distribution(prediction_frame):
    print("\nValidation event probability distribution")
    for event in EVENT_TARGET_COLUMNS:
        column = f"prob_{event}"
        if column not in prediction_frame.columns:
            continue
        values = pd.to_numeric(prediction_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) == 0:
            print(f"- {event}: unavailable")
            continue
        below = float((values < 0.01).mean())
        above = float((values > 0.99).mean())
        print(
            f"- {event}: "
            f"min={values.min():.6g}, "
            f"p01={values.quantile(0.01):.6g}, "
            f"p10={values.quantile(0.10):.6g}, "
            f"median={values.median():.6g}, "
            f"p90={values.quantile(0.90):.6g}, "
            f"p99={values.quantile(0.99):.6g}, "
            f"max={values.max():.6g}, "
            f"frac_below_0.01={below:.2%}, "
            f"frac_above_0.99={above:.2%}"
        )
        if below > 0.30 or above > 0.30:
            print("  WARNING: event head saturated.")


def event_saturation_warning_for_frame(prediction_frame):
    warnings = []
    for event in EVENT_TARGET_COLUMNS:
        column = f"prob_{event}"
        if column not in prediction_frame.columns:
            warnings.append(f"{event}: missing probability column")
            continue
        values = pd.to_numeric(prediction_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) == 0:
            warnings.append(f"{event}: no finite probabilities")
            continue
        below = float((values < 0.01).mean())
        above = float((values > 0.99).mean())
        if below > 0.30 or above > 0.30:
            warnings.append(
                f"{event}: saturated low={below:.2%}, high={above:.2%}"
            )
    return warnings


def regression_sanity_failures_for_frame(prediction_frame, regression_columns):
    failures = []
    bounds = regression_clip_bounds(
        regression_columns,
        max_abs_return=MICRO_MAX_ABS_RETURN_TARGET,
        max_abs_spread=MICRO_MAX_ABS_SPREAD_TARGET,
        max_abs_log_depth=MICRO_MAX_ABS_LOG_DEPTH_TARGET,
    )
    for target in regression_columns:
        column = f"pred_{target}"
        if column not in prediction_frame.columns:
            failures.append(f"{target}: missing prediction column")
            continue
        values = pd.to_numeric(prediction_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            failures.append(f"{target}: non-finite prediction values")
            continue
        bound = bounds.get(target)
        if bound is None:
            continue
        extreme = values.abs() > float(bound) * 3.0
        if bool(extreme.any()):
            failures.append(
                f"{target}: {int(extreme.sum())} predictions exceed 3x sane bound"
            )
    return failures


def summarize_enrichment(validation, prediction_frame, events):
    values = []
    for event in events:
        if event not in validation.columns or f"prob_{event}" not in prediction_frame.columns:
            continue
        metrics = top_bucket_enrichment(validation, prediction_frame, event, 0.05)
        if np.isfinite(metrics["enrichment"]):
            values.append(metrics["enrichment"])
    return float(np.mean(values)) if values else np.nan


def summarize_metric_values(metrics, events, suffix):
    values = []
    for event in events:
        value = metrics.get(f"{event}_{suffix}", np.nan)
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else np.nan


def group_metric_summary(metrics):
    summary = {}
    for group_name, events in EVENT_TARGET_GROUPS.items():
        summary[group_name] = {
            "event_bce": summarize_metric_values(metrics, events, "bce"),
            "top5_enrichment": summarize_metric_values(metrics, events, "top_5pct_enrichment"),
            "top10_enrichment": summarize_metric_values(metrics, events, "top_10pct_enrichment"),
            "precision": summarize_metric_values(metrics, events, "precision"),
            "recall": summarize_metric_values(metrics, events, "recall"),
            "support": int(
                sum(
                    metrics.get(f"{event}_validation_support", 0)
                    for event in events
                    if event in EVENT_TARGET_COLUMNS
                )
            ),
        }
    return summary


def finite_delta(a, b):
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return float(a - b)


def evaluate_predictions(validation, prediction_frame, active_event_columns=None, regression_columns=None):
    active_event_columns = (
        list(EVENT_TARGET_COLUMNS)
        if active_event_columns is None
        else list(active_event_columns)
    )
    regression_columns = (
        list(REGRESSION_TARGET_COLUMNS)
        if regression_columns is None
        else list(regression_columns)
    )
    metrics = {}
    for event in EVENT_TARGET_COLUMNS:
        support = int(pd.to_numeric(validation[event], errors="coerce").fillna(0.0).sum())
        metrics[f"{event}_validation_support"] = support
        metrics[f"{event}_bce"] = event_bce_for_target(validation, prediction_frame, event)
        precision, recall, tp, fp, fn = precision_recall(
            validation[event],
            prediction_frame[f"prob_{event}"],
            threshold=0.5,
        )
        status = "ok" if event in active_event_columns else "insufficient_support"
        metrics[f"{event}_metric_status"] = status
        metrics[f"{event}_precision"] = precision if status == "ok" else np.nan
        metrics[f"{event}_recall"] = recall if status == "ok" else np.nan
        metrics[f"{event}_tp"] = tp
        metrics[f"{event}_fp"] = fp
        metrics[f"{event}_fn"] = fn
        for fraction in [0.01, 0.05, 0.10]:
            bucket = top_bucket_enrichment(validation, prediction_frame, event, fraction)
            prefix = f"{event}_top_{int(fraction * 100)}pct"
            metrics[f"{prefix}_event_rate"] = bucket["bucket_rate"]
            metrics[f"{prefix}_baseline_rate"] = bucket["baseline_rate"]
            metrics[f"{prefix}_enrichment"] = bucket["enrichment"]

    for target in regression_columns:
        pred = prediction_frame[f"pred_{target}"].to_numpy(dtype=np.float64)
        actual = validation[target].to_numpy(dtype=np.float64)
        metrics[f"{target}_mae"] = float(np.mean(np.abs(pred - actual)))

    metrics["validation_event_bce"] = event_bce_loss(validation, prediction_frame, active_event_columns)
    metrics["validation_regression_mse"] = regression_mse(validation, prediction_frame, regression_columns)
    metrics["validation_regression_mae_mean"] = regression_mae_mean(validation, prediction_frame, regression_columns)
    metrics["scare_top5_enrichment"] = summarize_enrichment(
        validation,
        prediction_frame,
        ["upside_scare_event_10s", "downside_scare_event_10s"],
    )
    metrics["aggressive_burst_top5_enrichment"] = summarize_enrichment(
        validation,
        prediction_frame,
        ["aggressive_buy_burst_10s", "aggressive_sell_burst_10s"],
    )
    metrics["liquidity_drop_top5_enrichment"] = summarize_enrichment(
        validation,
        prediction_frame,
        ["bid_liquidity_drop_10s", "ask_liquidity_drop_10s"],
    )

    combined_scare_probability = prediction_frame[
        ["prob_upside_scare_event_10s", "prob_downside_scare_event_10s"]
    ].max(axis=1).to_numpy(dtype=np.float64)
    top_count = max(1, int(len(validation) * 0.05))
    top = validation.assign(_prob=combined_scare_probability).sort_values("_prob", ascending=False).head(top_count)
    overall_runup = float(validation["max_runup_10s"].mean())
    top_runup = float(top["max_runup_10s"].mean())
    overall_drawdown = float(validation["max_drawdown_10s"].mean())
    top_drawdown = float(top["max_drawdown_10s"].mean())
    metrics["top_5pct_avg_runup"] = top_runup
    metrics["overall_avg_runup"] = overall_runup
    metrics["top_5pct_avg_drawdown"] = top_drawdown
    metrics["overall_avg_drawdown"] = overall_drawdown
    for group_name, group_values in group_metric_summary(metrics).items():
        for key, value in group_values.items():
            metrics[f"{group_name}_{key}"] = value
    return metrics


def should_promote(metrics, validation_rows):
    reasons = []
    if validation_rows < MIN_VALIDATION_ROWS:
        reasons.append(f"validation rows {validation_rows} < MIN_VALIDATION_ROWS {MIN_VALIDATION_ROWS}")
    upside_recall = metrics.get("upside_scare_event_10s_recall", 0.0)
    downside_recall = metrics.get("downside_scare_event_10s_recall", 0.0)
    upside_recall = 0.0 if not np.isfinite(upside_recall) else upside_recall
    downside_recall = 0.0 if not np.isfinite(downside_recall) else downside_recall
    if upside_recall < MIN_SCARE_RECALL and downside_recall < MIN_SCARE_RECALL:
        reasons.append("both upside and downside scare recall are below the minimum gate")
    top_abs = abs(metrics["top_5pct_avg_runup"]) + abs(metrics["top_5pct_avg_drawdown"])
    overall_abs = abs(metrics["overall_avg_runup"]) + abs(metrics["overall_avg_drawdown"])
    if top_abs <= overall_abs:
        reasons.append("top predicted scare bucket is not more extreme than the overall validation set")
    return len(reasons) == 0, reasons


def print_event_distribution(frame, title):
    print(f"\n{title}")
    total = len(frame) or 1
    for column in EVENT_TARGET_COLUMNS:
        positives = int(frame[column].sum())
        print(f"- {column}: {positives}/{len(frame)} ({percent(positives / total)})")


def feature_group_columns(columns, group_name):
    columns = list(columns)
    required_micro = set(required_micro_feature_columns(columns))
    flow1s_context = {
        column
        for column in columns
        if str(column).startswith("feature_context_flow_1s_")
    }

    if group_name == "full_with_flow1s_context":
        selected = columns
    elif group_name == "without_flow1s_context":
        selected = [
            column
            for column in columns
            if not str(column).startswith("feature_context_flow_1s_")
        ]
    elif group_name == "flow1s_only_context_plus_required_micro":
        selected = [
            column
            for column in columns
            if column in required_micro or column in flow1s_context
        ]
    elif group_name == "no_context_required_micro_only":
        selected = [
            column
            for column in columns
            if column in required_micro
        ]
    else:
        raise ValueError(f"Unknown micro feature group: {group_name}")

    # Preserve the canonical feature order from get_micro_feature_columns().
    return [column for column in columns if column in set(selected)]


def available_ablation_feature_groups(columns):
    groups = [
        "full_with_flow1s_context",
        "without_flow1s_context",
        "flow1s_only_context_plus_required_micro",
        "no_context_required_micro_only",
    ]
    flow1s_count = sum(1 for column in columns if str(column).startswith("feature_context_flow_1s_"))
    if flow1s_count == 0:
        return ["full_with_flow1s_context", "no_context_required_micro_only"]
    return groups


def run_training_experiment(
    name,
    train,
    validation,
    columns,
    active_event_columns,
    active_event_mask,
):
    print(f"\nTraining experiment: {name}")
    print(f"- feature count: {len(columns)}")
    print(f"- event_only: {MICRO_TRAIN_EVENT_ONLY}")
    x_train = train[columns].to_numpy(dtype=np.float64)
    x_validation = validation[columns].to_numpy(dtype=np.float64)
    regression_columns = [] if MICRO_TRAIN_EVENT_ONLY else list(REGRESSION_TARGET_COLUMNS)
    if regression_columns:
        y_reg_train = train[regression_columns].to_numpy(dtype=np.float64)
        y_reg_validation = validation[regression_columns].to_numpy(dtype=np.float64)
    else:
        y_reg_train = np.zeros((len(train), 0), dtype=np.float64)
        y_reg_validation = np.zeros((len(validation), 0), dtype=np.float64)
    y_event_train = train[EVENT_TARGET_COLUMNS].to_numpy(dtype=np.float64)
    y_event_validation = validation[EVENT_TARGET_COLUMNS].to_numpy(dtype=np.float64)

    x_train, x_validation, feature_mean, feature_std = standardize(x_train, x_validation)
    if regression_columns:
        y_reg_train_scaled, y_reg_validation_scaled, target_mean, target_std = standardize_regression_targets(
            y_reg_train,
            y_reg_validation,
            regression_columns,
        )
        clip_bounds = regression_clip_bounds(
            regression_columns,
            max_abs_return=MICRO_MAX_ABS_RETURN_TARGET,
            max_abs_spread=MICRO_MAX_ABS_SPREAD_TARGET,
            max_abs_log_depth=MICRO_MAX_ABS_LOG_DEPTH_TARGET,
        )
        regression_target_scalers = regression_scalers_from_arrays(
            regression_columns,
            target_mean,
            target_std,
            clip_bounds,
        )
    else:
        y_reg_train_scaled = y_reg_train
        y_reg_validation_scaled = y_reg_validation
        target_mean = np.asarray([], dtype=np.float64)
        target_std = np.asarray([], dtype=np.float64)
        regression_target_scalers = None

    model = train_model(
        x_train,
        y_reg_train_scaled,
        y_event_train,
        x_validation,
        y_reg_validation_scaled,
        y_event_validation,
        active_event_mask,
    )

    artifact = {
        "feature_columns": columns,
        "feature_count": int(len(columns)),
        "event_only": bool(MICRO_TRAIN_EVENT_ONLY),
        "regression_target_columns": regression_columns,
        "event_target_columns": EVENT_TARGET_COLUMNS,
        "active_event_target_columns": active_event_columns,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "model": model,
    }
    if regression_target_scalers is not None:
        artifact["regression_target_scalers"] = regression_target_scalers
    regression, event_probabilities = predict_with_artifact(artifact, validation)
    prediction_frame = predictions_frame(
        validation,
        regression,
        event_probabilities,
        regression_columns,
        EVENT_TARGET_COLUMNS,
    )
    for column in [
        *REGRESSION_TARGET_COLUMNS,
        *DIAGNOSTIC_TARGET_COLUMNS,
        *EVENT_TARGET_COLUMNS,
        "future_return_30s",
        "future_return_60s",
    ]:
        if column in validation.columns:
            prediction_frame[column] = validation[column].to_numpy()
    metrics = evaluate_predictions(
        validation.reset_index(drop=True),
        prediction_frame.reset_index(drop=True),
        active_event_columns,
        regression_columns,
    )
    regression_failures = (
        [] if MICRO_TRAIN_EVENT_ONLY else regression_sanity_failures_for_frame(prediction_frame, regression_columns)
    )
    event_saturation_warnings = event_saturation_warning_for_frame(prediction_frame)
    return {
        "name": name,
        "feature_group_used": name,
        "columns": columns,
        "model": model,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "regression_target_scalers": regression_target_scalers,
        "regression_target_columns": regression_columns,
        "prediction_frame": prediction_frame,
        "metrics": metrics,
        "regression_sanity_failures": regression_failures,
        "event_saturation_warnings": event_saturation_warnings,
    }


def result_metric(result, key):
    value = result.get("metrics", {}).get(key, np.nan)
    return float(value) if isinstance(value, (int, float, np.floating)) else np.nan


def format_float(value, digits=6):
    if value is None or not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.{digits}g}"


def build_context_group_ablation_summary(results):
    by_name = {result["name"]: result for result in results}
    summary = {
        "experiments": [],
        "event_targets": {},
        "target_groups": {},
        "comparisons": {},
        "group_target_disagreement": False,
    }

    for result in results:
        metrics = result["metrics"]
        summary["experiments"].append(
            {
                "name": result["name"],
                "feature_count": int(len(result["columns"])),
                "validation_event_bce": result_metric(result, "validation_event_bce"),
                "validation_regression_mse": result_metric(result, "validation_regression_mse"),
                "validation_regression_mae_mean": result_metric(result, "validation_regression_mae_mean"),
            }
        )
        for event in EVENT_TARGET_COLUMNS:
            summary["event_targets"].setdefault(event, {})[result["name"]] = {
                "validation_bce": metrics.get(f"{event}_bce", np.nan),
                "top5_enrichment": metrics.get(f"{event}_top_5pct_enrichment", np.nan),
                "top10_enrichment": metrics.get(f"{event}_top_10pct_enrichment", np.nan),
                "precision_at_0_5": metrics.get(f"{event}_precision", np.nan),
                "recall_at_0_5": metrics.get(f"{event}_recall", np.nan),
                "validation_support": int(metrics.get(f"{event}_validation_support", 0)),
                "metric_status": metrics.get(f"{event}_metric_status", "unknown"),
            }
        for group_name in EVENT_TARGET_GROUPS:
            summary["target_groups"].setdefault(group_name, {})[result["name"]] = {
                "event_bce": metrics.get(f"{group_name}_event_bce", np.nan),
                "top5_enrichment": metrics.get(f"{group_name}_top5_enrichment", np.nan),
                "top10_enrichment": metrics.get(f"{group_name}_top10_enrichment", np.nan),
                "precision": metrics.get(f"{group_name}_precision", np.nan),
                "recall": metrics.get(f"{group_name}_recall", np.nan),
                "support": int(metrics.get(f"{group_name}_support", 0)),
                "decoded_regression_mae": metrics.get("validation_regression_mae_mean", np.nan),
            }

    comparisons = [
        ("full_vs_without_flow1s", "full_with_flow1s_context", "without_flow1s_context"),
        (
            "flow1s_required_micro_vs_required_micro_only",
            "flow1s_only_context_plus_required_micro",
            "no_context_required_micro_only",
        ),
    ]
    disagreement_votes = []
    for comparison_name, with_name, without_name in comparisons:
        if with_name not in by_name or without_name not in by_name:
            continue
        with_metrics = by_name[with_name]["metrics"]
        without_metrics = by_name[without_name]["metrics"]
        comparison = {}
        for group_name in EVENT_TARGET_GROUPS:
            delta_bce = finite_delta(
                with_metrics.get(f"{group_name}_event_bce", np.nan),
                without_metrics.get(f"{group_name}_event_bce", np.nan),
            )
            delta_top5 = finite_delta(
                with_metrics.get(f"{group_name}_top5_enrichment", np.nan),
                without_metrics.get(f"{group_name}_top5_enrichment", np.nan),
            )
            delta_mae = finite_delta(
                with_metrics.get("validation_regression_mae_mean", np.nan),
                without_metrics.get("validation_regression_mae_mean", np.nan),
            )
            # Lower BCE/MAE is better; higher top-bucket enrichment is better.
            helps_bce = np.isfinite(delta_bce) and delta_bce < 0
            helps_top5 = np.isfinite(delta_top5) and delta_top5 > 0
            hurts_bce = np.isfinite(delta_bce) and delta_bce > 0
            hurts_top5 = np.isfinite(delta_top5) and delta_top5 < 0
            if helps_bce and helps_top5:
                verdict = "helps"
            elif hurts_bce and hurts_top5:
                verdict = "hurts"
            elif (helps_bce or helps_top5) and (hurts_bce or hurts_top5):
                verdict = "mixed"
            elif helps_bce or helps_top5:
                verdict = "slightly_helps"
            elif hurts_bce or hurts_top5:
                verdict = "slightly_hurts"
            else:
                verdict = "flat_or_unclear"
            comparison[group_name] = {
                "delta_event_bce": delta_bce,
                "delta_top5_enrichment": delta_top5,
                "delta_decoded_regression_mae": delta_mae,
                "verdict": verdict,
            }
            if verdict in {"helps", "slightly_helps"}:
                disagreement_votes.append("helps")
            elif verdict in {"hurts", "slightly_hurts"}:
                disagreement_votes.append("hurts")
            elif verdict == "mixed":
                disagreement_votes.append("mixed")
        summary["comparisons"][comparison_name] = comparison

    summary["group_target_disagreement"] = (
        "mixed" in disagreement_votes
        or ("helps" in disagreement_votes and "hurts" in disagreement_votes)
    )
    return summary


def print_per_target_ablation_metrics(results):
    print("\nPer-event target ablation metrics")
    print(
        "event_target | experiment | validation BCE | top5 enrich | top10 enrich | "
        "precision@0.5 | recall@0.5 | validation support"
    )
    for event in EVENT_TARGET_COLUMNS:
        for result in results:
            metrics = result["metrics"]
            print(
                f"{event} | {result['name']} | "
                f"{format_float(metrics.get(f'{event}_bce'))} | "
                f"{format_float(metrics.get(f'{event}_top_5pct_enrichment'))} | "
                f"{format_float(metrics.get(f'{event}_top_10pct_enrichment'))} | "
                f"{format_float(metrics.get(f'{event}_precision'))} | "
                f"{format_float(metrics.get(f'{event}_recall'))} | "
                f"{int(metrics.get(f'{event}_validation_support', 0))}"
            )


def print_group_ablation_metrics(results, summary):
    print("\nGrouped target ablation metrics")
    print(
        "target_group | experiment | event BCE | top5 enrich | top10 enrich | "
        "precision | recall | support | decoded regression MAE"
    )
    for group_name in EVENT_TARGET_GROUPS:
        for result in results:
            metrics = result["metrics"]
            print(
                f"{group_name} | {result['name']} | "
                f"{format_float(metrics.get(f'{group_name}_event_bce'))} | "
                f"{format_float(metrics.get(f'{group_name}_top5_enrichment'))} | "
                f"{format_float(metrics.get(f'{group_name}_top10_enrichment'))} | "
                f"{format_float(metrics.get(f'{group_name}_precision'))} | "
                f"{format_float(metrics.get(f'{group_name}_recall'))} | "
                f"{int(metrics.get(f'{group_name}_support', 0))} | "
                f"{format_float(metrics.get('validation_regression_mae_mean'))}"
            )

    print("\nDoes 1s context help each target group?")
    if not summary.get("comparisons"):
        print("- no comparable 1s context experiments were run")
        return
    for comparison_name, groups in summary["comparisons"].items():
        print(f"- {comparison_name}")
        for group_name, values in groups.items():
            print(
                f"  - {group_name}: "
                f"delta_event_bce={format_float(values['delta_event_bce'])}, "
                f"delta_top5_enrichment={format_float(values['delta_top5_enrichment'])}, "
                f"delta_decoded_regression_mae={format_float(values['delta_decoded_regression_mae'])}, "
                f"verdict={values['verdict']}"
            )
    if summary.get("group_target_disagreement"):
        print("- WARNING: target groups disagree; promotion cannot rely on aggregate metrics alone.")


def print_ablation_summary(results):
    if not results or len(results) < 2:
        return build_context_group_ablation_summary(results)
    summary = build_context_group_ablation_summary(results)
    print("\nFeature-group ablation summary")
    print(
        "experiment | validation event BCE | regression MSE | scare top5 enrich | "
        "aggressive burst enrich | liquidity-drop enrich | decoded regression MAE"
    )
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['name']} | "
            f"{metrics.get('validation_event_bce', np.nan):.6g} | "
            f"{metrics.get('validation_regression_mse', np.nan):.6g} | "
            f"{metrics.get('scare_top5_enrichment', np.nan):.6g} | "
            f"{metrics.get('aggressive_burst_top5_enrichment', np.nan):.6g} | "
            f"{metrics.get('liquidity_drop_top5_enrichment', np.nan):.6g} | "
            f"{metrics.get('validation_regression_mae_mean', np.nan):.6g}"
        )
    print_per_target_ablation_metrics(results)
    print_group_ablation_metrics(results, summary)
    return summary


def safe_metric(metrics, key, default=np.nan):
    value = metrics.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def feature_group_selection_score(metrics):
    event_bce = safe_metric(metrics, "validation_event_bce", np.inf)
    decoded_mae = 0.0 if MICRO_TRAIN_EVENT_ONLY else safe_metric(metrics, "validation_regression_mae_mean", np.inf)
    scare_bonus = safe_metric(metrics, "scare_top5_enrichment", 0.0)
    liquidity_bonus = safe_metric(metrics, "liquidity_drop_top5_enrichment", 0.0)
    path_bonus = safe_metric(metrics, "path_targets_top5_enrichment", 0.0)
    return float(
        -event_bce
        - decoded_mae
        + scare_bonus
        + liquidity_bonus
        + path_bonus
    )


def selection_baseline_mae(results):
    if MICRO_TRAIN_EVENT_ONLY:
        return 0.0
    by_name = {result["name"]: result for result in results}
    baseline = by_name.get("no_context_required_micro_only") or by_name.get("without_flow1s_context") or results[0]
    return safe_metric(baseline["metrics"], "validation_regression_mae_mean", np.inf)


def select_preferred_feature_group(results, summary):
    baseline_mae = selection_baseline_mae(results)
    rows = []
    for result in results:
        metrics = result["metrics"]
        score = feature_group_selection_score(metrics)
        reasons = []
        if (not MICRO_TRAIN_EVENT_ONLY) and result.get("regression_sanity_failures"):
            reasons.extend(
                f"regression sanity failure: {reason}"
                for reason in result["regression_sanity_failures"][:5]
            )
        if result.get("event_saturation_warnings"):
            reasons.extend(
                f"event saturation warning: {reason}"
                for reason in result["event_saturation_warnings"][:5]
            )
        scare_enrichment = safe_metric(metrics, "scare_top5_enrichment", 0.0)
        if (not MICRO_TRAIN_EVENT_ONLY) and scare_enrichment < 1.2:
            reasons.append(
                f"scare_top5_enrichment {scare_enrichment:.6g} < 1.2"
            )
        decoded_mae = safe_metric(metrics, "validation_regression_mae_mean", np.inf)
        if (not MICRO_TRAIN_EVENT_ONLY) and np.isfinite(baseline_mae) and decoded_mae > baseline_mae * 1.10:
            reasons.append(
                f"decoded regression MAE {decoded_mae:.6g} is worse than baseline {baseline_mae:.6g} by more than 10%"
            )
        if (not MICRO_TRAIN_EVENT_ONLY) and summary.get("group_target_disagreement") and not MICRO_ALLOW_GROUP_DISAGREEMENT:
            reasons.append(
                "target groups disagree; MICRO_ALLOW_GROUP_DISAGREEMENT is false, so promotion is blocked"
            )
        rows.append(
            {
                "result": result,
                "score": score,
                "reasons": reasons,
                "hard_rejected": bool(reasons),
            }
        )

    eligible = [row for row in rows if not row["hard_rejected"]]
    if eligible:
        selected = max(eligible, key=lambda row: row["score"])
    else:
        # Fail-soft for paper research: keep the highest-scoring candidate artifact so
        # diagnostics can continue, but record that it failed selection gates.
        selected = max(rows, key=lambda row: row["score"])
        selected["reasons"].append("no feature group passed all hard selection gates")

    print("\nFeature-group candidate selection")
    print("feature_group | score | selected | hard_rejected | selection notes")
    for row in sorted(rows, key=lambda item: item["score"], reverse=True):
        selected_marker = "YES" if row is selected else "no"
        notes = "; ".join(row["reasons"]) if row["reasons"] else "passed selection gates"
        print(
            f"{row['result']['name']} | "
            f"{format_float(row['score'])} | "
            f"{selected_marker} | "
            f"{row['hard_rejected']} | "
            f"{notes}"
        )

    selected_result = selected["result"]
    selection_reason = {
        "selected_feature_group": selected_result["name"],
        "selection_score": selected["score"],
        "baseline_feature_group_for_mae": "no_context_required_micro_only",
        "baseline_decoded_regression_mae": baseline_mae,
        "micro_allow_group_disagreement": MICRO_ALLOW_GROUP_DISAGREEMENT,
        "selection_notes": selected["reasons"] or ["passed selection gates"],
        "all_candidates": [
            {
                "feature_group": row["result"]["name"],
                "score": row["score"],
                "hard_rejected": bool(row["hard_rejected"]),
                "notes": row["reasons"],
            }
            for row in rows
        ],
    }
    return selected_result, selection_reason


def main():
    frame, columns = load_training_rows()
    print("10s microstructure event model trainer")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Training rows path: {TRAINING_PATH}")
    print(f"Rows: {len(frame)}")
    print(f"Feature count: {len(columns)}")
    print(f"PROMOTE_BEST: {PROMOTE_BEST}")
    print(f"MICRO_RUN_FLOW_1S_ABLATION: {MICRO_RUN_FLOW_1S_ABLATION}")
    print(f"MICRO_TRAIN_EVENT_ONLY: {MICRO_TRAIN_EVENT_ONLY}")
    print("No trades are placed.")
    print_context_availability_summary(frame)

    if len(frame) < MIN_TRAIN_ROWS:
        print(f"Training skipped: rows {len(frame)} < MIN_TRAIN_ROWS {MIN_TRAIN_ROWS}.")
        return

    train, validation, train_validation_embargo = split_time_ordered(frame)
    if len(validation) < 1:
        print("Training skipped: validation split has no rows.")
        print(f"Dropped train/validation embargo rows: {train_validation_embargo}")
        print("Dropped validation/test embargo rows: 0 (10s trainer has no separate test split)")
        return

    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"EMBARGO_ROWS: {EMBARGO_ROWS}")
    print(f"Dropped train/validation embargo rows: {train_validation_embargo}")
    print("Dropped validation/test embargo rows: 0 (10s trainer has no separate test split)")
    print(f"Train timestamp range: {int(train['timestamp'].min())} -> {int(train['timestamp'].max())}")
    print(f"Validation timestamp range: {int(validation['timestamp'].min())} -> {int(validation['timestamp'].max())}")
    print_event_distribution(train, "Train event distributions")
    print_event_distribution(validation, "Validation event distributions")
    if MICRO_TRAIN_EVENT_ONLY:
        print("Default regression training targets: skipped because MICRO_TRAIN_EVENT_ONLY=true")
    else:
        print(f"Default regression training targets: {', '.join(REGRESSION_TARGET_COLUMNS)}")
    print(
        "Diagnostic-only raw depth targets: "
        + ", ".join(column for column in DIAGNOSTIC_TARGET_COLUMNS if column in frame.columns)
    )

    event_support, active_event_columns = event_support_table(train, validation)
    print_event_support(event_support)
    active_event_mask = np.asarray(
        [1.0 if column in active_event_columns else 0.0 for column in EVENT_TARGET_COLUMNS],
        dtype=np.float64,
    )

    full_columns = feature_group_columns(columns, "full_with_flow1s_context")
    full_result = run_training_experiment(
        "full_with_flow1s_context",
        train,
        validation,
        full_columns,
        active_event_columns,
        active_event_mask,
    )
    ablation_results = [full_result]
    if MICRO_RUN_FLOW_1S_ABLATION:
        for group_name in available_ablation_feature_groups(columns):
            if group_name == "full_with_flow1s_context":
                continue
            group_columns = feature_group_columns(columns, group_name)
            if len(group_columns) == 0:
                print(f"\nFeature-group ablation skipped for {group_name}: no columns selected.")
                continue
            ablation_results.append(
                run_training_experiment(
                    group_name,
                    train,
                    validation,
                    group_columns,
                    active_event_columns,
                    active_event_mask,
                )
            )
    context_group_ablation_summary = print_ablation_summary(ablation_results)
    selected_result, selection_reason = select_preferred_feature_group(
        ablation_results,
        context_group_ablation_summary,
    )
    selected_candidate_selection = next(
        (
            row for row in selection_reason.get("all_candidates", [])
            if row.get("feature_group") == selected_result["name"]
        ),
        {},
    )
    selected_hard_rejected = bool(selected_candidate_selection.get("hard_rejected", False))
    selection_rejection_reasons = [
        str(reason)
        for reason in selected_candidate_selection.get("notes", [])
        if str(reason).strip()
    ]
    regression_sanity_status = (
        "not_applicable"
        if MICRO_TRAIN_EVENT_ONLY
        else ("fail" if selected_result.get("regression_sanity_failures") else "ok")
    )
    event_sanity_status = (
        "fail" if selected_result.get("event_saturation_warnings") else "ok"
    )

    selected_columns = selected_result["columns"]
    schema_hash = feature_schema_hash(selected_columns)
    created_at = current_utc_tag()
    feature_mean = selected_result["feature_mean"]
    feature_std = selected_result["feature_std"]
    target_mean = selected_result["target_mean"]
    target_std = selected_result["target_std"]
    regression_target_scalers = selected_result.get("regression_target_scalers")
    selected_regression_columns = list(selected_result.get("regression_target_columns", []))
    artifact = {
        "model_type": "paper_only_10s_microstructure_numpy_mlp",
        "event_only": bool(MICRO_TRAIN_EVENT_ONLY),
        "symbol": SYMBOL,
        "model_symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE or "legacy",
        "created_at": created_at,
        "model_id": f"{SYMBOL}_microstructure_10s_{created_at}_{schema_hash}",
        "trained_until_timestamp": int(train["timestamp"].max()) if len(train) else None,
        "feature_columns": selected_columns,
        "feature_count": int(len(selected_columns)),
        "feature_schema_hash": schema_hash,
        "selected_feature_group": selected_result["name"],
        "feature_group_used": selected_result["feature_group_used"],
        "context_group_ablation_summary": context_group_ablation_summary,
        "selection_reason": selection_reason,
        "regression_target_columns": selected_regression_columns,
        "event_target_columns": EVENT_TARGET_COLUMNS,
        "active_event_target_columns": active_event_columns,
        "event_support": event_support.to_dict(orient="records"),
        "event_target_min_train_positives": EVENT_TARGET_MIN_TRAIN_POSITIVES,
        "event_target_min_validation_positives": EVENT_TARGET_MIN_VALIDATION_POSITIVES,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "micro_max_abs_return_target": MICRO_MAX_ABS_RETURN_TARGET,
        "micro_max_abs_spread_target": MICRO_MAX_ABS_SPREAD_TARGET,
        "micro_max_abs_log_depth_target": MICRO_MAX_ABS_LOG_DEPTH_TARGET,
        "event_probability_temperature": 1.0,
        "hidden_units": HIDDEN_UNITS,
        "train_split": TRAIN_SPLIT,
        "model": selected_result["model"],
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
    }
    if regression_target_scalers is not None:
        artifact["regression_target_scalers"] = regression_target_scalers

    prediction_frame = selected_result["prediction_frame"]
    for column in [
        *REGRESSION_TARGET_COLUMNS,
        *DIAGNOSTIC_TARGET_COLUMNS,
        *EVENT_TARGET_COLUMNS,
        "future_return_30s",
        "future_return_60s",
    ]:
        if column in validation.columns:
            prediction_frame[column] = validation[column].to_numpy()

    metrics = selected_result["metrics"]
    promote, rejection_reasons = should_promote(metrics, len(validation))
    event_only_group_gate_failed = bool(
        MICRO_TRAIN_EVENT_ONLY
        and context_group_ablation_summary.get("group_target_disagreement")
        and not MICRO_ALLOW_GROUP_DISAGREEMENT
    )
    if selected_hard_rejected:
        promote = False
        rejection_reasons.extend(selection_rejection_reasons)
    if context_group_ablation_summary.get("group_target_disagreement"):
        promote = False
        rejection_reasons.append(
            "target groups disagree across feature-group ablations; aggregate metrics are not enough for promotion"
        )
    for note in selection_reason.get("selection_notes", []):
        if isinstance(note, str) and "promotion is blocked" in note:
            promote = False
            rejection_reasons.append(note)
    rejection_reasons = list(dict.fromkeys(str(reason) for reason in rejection_reasons if str(reason).strip()))
    if MICRO_TRAIN_EVENT_ONLY:
        candidate_promotable = bool(
            not selected_hard_rejected
            and event_sanity_status == "ok"
            and not event_only_group_gate_failed
        )
    else:
        candidate_promotable = bool(promote and not selected_hard_rejected)
    artifact_rejection_reasons = (
        [] if MICRO_TRAIN_EVENT_ONLY and candidate_promotable else rejection_reasons
    )

    tag = current_utc_tag()
    candidate_dir = CANDIDATE_ROOT / tag
    candidate_path = candidate_dir / "model.json"
    artifact.update(
        {
            "hard_rejected": bool(selected_hard_rejected),
            "promotable": bool(candidate_promotable),
            "rejection_reasons": artifact_rejection_reasons,
            "promotion_gate_notes": rejection_reasons,
            "regression_sanity_status": regression_sanity_status,
            "event_sanity_status": event_sanity_status,
            "validation_event_bce": safe_metric(metrics, "validation_event_bce", np.nan),
            "validation_regression_mae_mean": safe_metric(metrics, "validation_regression_mae_mean", np.nan),
        }
    )
    save_model(candidate_path, artifact)
    atomic_write_csv(prediction_frame, PREDICTIONS_PATH)
    atomic_write_csv(pd.DataFrame([metrics]), candidate_dir / "validation_metrics.csv")
    atomic_write_csv(event_support, candidate_dir / "event_support.csv")

    print("\nValidation metrics")
    for key, value in metrics.items():
        print(f"- {key}: {value:.6g}" if isinstance(value, float) and np.isfinite(value) else f"- {key}: {value}")
    print_event_probability_distribution(prediction_frame.reset_index(drop=True))
    print_top_bucket_diagnostics(validation.reset_index(drop=True), prediction_frame.reset_index(drop=True))

    if PROMOTE_BEST and promote:
        copy_if_promoted(candidate_path, ACTIVE_MODEL_PATH)
        print(f"\nCandidate promoted to active paper model: {ACTIVE_MODEL_PATH}")
    else:
        print("\nCandidate not promoted.")
        if not PROMOTE_BEST:
            print("- PROMOTE_BEST is false")
        for reason in rejection_reasons:
            print(f"- {reason}")

    print(f"Candidate model saved to: {candidate_path}")
    print(f"Validation predictions saved to: {PREDICTIONS_PATH}")
    print("No trades were placed.")


if __name__ == "__main__":
    main()
