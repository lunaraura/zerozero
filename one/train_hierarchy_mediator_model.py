import datetime as dt
import hashlib
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
EPOCHS = int(os.getenv("MEDIATOR_EPOCHS", "500"))
LEARNING_RATE = float(os.getenv("MEDIATOR_LEARNING_RATE", "0.05"))
L2 = float(os.getenv("MEDIATOR_L2", "0.001"))
MIN_ROWS = int(os.getenv("MEDIATOR_MIN_ROWS", "300"))
PREDICTION_THRESHOLD = float(os.getenv("MEDIATOR_PREDICTION_THRESHOLD", "0.55"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
TRAINING_PATH = VENUE_DIR / f"{SYMBOL}_hierarchy_mediator_training_rows.csv"
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "hierarchy_mediator" / VENUE_TAG

HORIZONS = ["10s", "30s", "60s"]
THRESHOLD_SWEEP = [0.50, 0.525, 0.55, 0.575, 0.60, 0.65]


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def current_utc_tag():
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def standardize(train_x, validation_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (train_x - mean) / std, (validation_x - mean) / std, (test_x - mean) / std, mean, std


def train_logistic(x, y):
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    y = y.astype(np.float64)
    positive = max(float(y.sum()), 1.0)
    negative = max(float(len(y) - y.sum()), 1.0)
    sample_weights = np.where(y > 0.5, len(y) / (2 * positive), len(y) / (2 * negative))
    for _ in range(EPOCHS):
        probability = sigmoid(x @ weights + bias)
        error = (probability - y) * sample_weights
        grad_w = (x.T @ error) / len(x) + L2 * weights
        grad_b = float(error.mean())
        weights -= LEARNING_RATE * grad_w
        bias -= LEARNING_RATE * grad_b
    return weights, bias


def train_linear_ridge(x, y):
    x_augmented = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(x_augmented.shape[1]) * L2
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(x_augmented.T @ x_augmented + penalty) @ x_augmented.T @ y
    return weights[1:], float(weights[0])


def split_chronological(frame):
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    n = len(frame)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    return frame.iloc[:train_end].copy(), frame.iloc[train_end:validation_end].copy(), frame.iloc[validation_end:].copy()


def strategy_metrics(actual_return, direction):
    actual_return = np.asarray(actual_return, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    active = direction != 0
    strategy_return = actual_return * direction
    strategy_return = np.where(active, strategy_return, 0.0)
    return {
        "rows": int(len(actual_return)),
        "active_rows": int(active.sum()),
        "avg_return": float(strategy_return.mean()) if len(strategy_return) else 0.0,
        "win_rate": float((strategy_return[active] > 0).mean()) if active.any() else np.nan,
        "coverage": float(active.mean()) if len(active) else 0.0,
    }


def direction_from_probability(probability, threshold=PREDICTION_THRESHOLD):
    probability = np.asarray(probability, dtype=np.float64)
    return np.where(probability >= threshold, 1, np.where(probability <= 1.0 - threshold, -1, 0))


def best_combined_filter_direction(frame):
    pressure = pd.to_numeric(frame.get("feature_flow_1s_pred_pressure"), errors="coerce").fillna(0.0)
    both_agree_bullish = pd.to_numeric(frame.get("feature_both_agree_bullish"), errors="coerce").fillna(0.0)
    # Mirrors the strongest current diagnostic filter:
    # both_agree_bullish AND abs_pressure >= 0.30.
    return np.where((both_agree_bullish >= 0.5) & (pressure.abs() >= 0.30), 1, 0)


def follow_pressure_direction(frame):
    pressure = pd.to_numeric(frame.get("feature_flow_1s_pred_pressure"), errors="coerce").fillna(0.0)
    return np.where(pressure > 0, 1, np.where(pressure < 0, -1, 0))


def evaluate_baselines(train, test, horizon):
    actual = test[f"target_realized_return_{horizon}"].to_numpy(dtype=np.float64)
    train_direction = train[f"target_realized_direction_{horizon}"].to_numpy(dtype=np.int64)
    up_rate = float((train_direction > 0).mean())
    majority_direction = 1 if up_rate >= 0.5 else -1
    return {
        "no_trade": strategy_metrics(actual, np.zeros(len(test))),
        "always_long": strategy_metrics(actual, np.ones(len(test))),
        "always_short": strategy_metrics(actual, -np.ones(len(test))),
        "majority_direction": strategy_metrics(actual, np.full(len(test), majority_direction)),
        "follow_1s_pressure": strategy_metrics(actual, follow_pressure_direction(test)),
        "best_combined_filter": strategy_metrics(actual, best_combined_filter_direction(test)),
    }


def evaluate_model_on_segment(segment, x_segment, direction_weights, direction_bias, train, horizon, threshold):
    actual = segment[f"target_realized_return_{horizon}"].to_numpy(dtype=np.float64)
    probability = sigmoid(x_segment @ direction_weights + direction_bias)
    direction = direction_from_probability(probability, threshold)
    mediator = strategy_metrics(actual, direction)
    baselines = evaluate_baselines(train, segment, horizon)
    majority = baselines["majority_direction"]
    best_filter = baselines["best_combined_filter"]
    return {
        "rows": int(len(segment)),
        "threshold": float(threshold),
        "mediator": mediator,
        "baselines": baselines,
        "lift_vs_majority_avg_return": float(mediator["avg_return"] - majority["avg_return"]),
        "lift_vs_majority_win_rate": (
            float(mediator["win_rate"] - majority["win_rate"])
            if np.isfinite(mediator["win_rate"]) and np.isfinite(majority["win_rate"])
            else np.nan
        ),
        "lift_vs_best_combined_filter_avg_return": float(mediator["avg_return"] - best_filter["avg_return"]),
        "lift_vs_best_combined_filter_win_rate": (
            float(mediator["win_rate"] - best_filter["win_rate"])
            if np.isfinite(mediator["win_rate"]) and np.isfinite(best_filter["win_rate"])
            else np.nan
        ),
    }


def threshold_sweep(segment, x_segment, direction_weights, direction_bias, train, horizon):
    rows = []
    for threshold in THRESHOLD_SWEEP:
        report = evaluate_model_on_segment(
            segment,
            x_segment,
            direction_weights,
            direction_bias,
            train,
            horizon,
            threshold,
        )
        mediator = report["mediator"]
        rows.append(
            {
                "threshold": float(threshold),
                "rows": int(mediator["rows"]),
                "rows_predicted": int(mediator["active_rows"]),
                "coverage": float(mediator["coverage"]),
                "win_rate": float(mediator["win_rate"]) if np.isfinite(mediator["win_rate"]) else np.nan,
                "avg_return": float(mediator["avg_return"]),
                "lift_vs_majority_avg_return": float(report["lift_vs_majority_avg_return"]),
                "lift_vs_majority_win_rate": float(report["lift_vs_majority_win_rate"]) if np.isfinite(report["lift_vs_majority_win_rate"]) else np.nan,
                "lift_vs_best_combined_filter_avg_return": float(report["lift_vs_best_combined_filter_avg_return"]),
                "lift_vs_best_combined_filter_win_rate": float(report["lift_vs_best_combined_filter_win_rate"]) if np.isfinite(report["lift_vs_best_combined_filter_win_rate"]) else np.nan,
            }
        )
    return rows


def feature_group(name):
    if "flow_1s" in name:
        return "1s_order_flow"
    if name.startswith("feature_micro_") or "scare" in name or "liquidity" in name:
        return "10s_microstructure"
    if "regime" in name or "htf" in name or "path_3m" in name:
        return "context_availability"
    if "agree" in name or "conflict" in name or "agreement" in name or "contradiction" in name:
        return "agreement_conflict"
    if "risk" in name or "abstain" in name:
        return "risk_abstention"
    return "other"


def top_feature_importances(feature_columns, weights_by_horizon, limit=25):
    combined = np.zeros(len(feature_columns), dtype=np.float64)
    for weights in weights_by_horizon.values():
        combined += np.abs(np.asarray(weights, dtype=np.float64))
    order = np.argsort(-combined)[:limit]
    return [
        {
            "feature": feature_columns[index],
            "importance": float(combined[index]),
            "group": feature_group(feature_columns[index]),
        }
        for index in order
    ]


def accountability(feature_columns, weights):
    rows = {}
    for column, weight in zip(feature_columns, weights):
        group = feature_group(column)
        rows.setdefault(group, {"abs_weight": 0.0, "signed_weight": 0.0, "top_features": []})
        rows[group]["abs_weight"] += float(abs(weight))
        rows[group]["signed_weight"] += float(weight)
        rows[group]["top_features"].append((column, float(weight)))
    for group in rows.values():
        group["top_features"] = [
            {"feature": name, "weight": weight}
            for name, weight in sorted(group["top_features"], key=lambda item: abs(item[1]), reverse=True)[:5]
        ]
    return rows


def feature_weight_report(feature_columns, weights, near_zero_threshold=1e-4, limit=15):
    items = [
        {
            "feature": column,
            "weight": float(weight),
            "abs_weight": float(abs(weight)),
            "group": feature_group(column),
        }
        for column, weight in zip(feature_columns, weights)
    ]
    positive = sorted([item for item in items if item["weight"] > 0], key=lambda item: item["weight"], reverse=True)
    negative = sorted([item for item in items if item["weight"] < 0], key=lambda item: item["weight"])
    near_zero = sorted(
        [item for item in items if abs(item["weight"]) <= near_zero_threshold],
        key=lambda item: item["feature"],
    )
    return {
        "top_positive_features": positive[:limit],
        "top_negative_features": negative[:limit],
        "near_zero_features": near_zero,
        "near_zero_threshold": float(near_zero_threshold),
    }


def group_help_harm(feature_columns, weights):
    groups = {}
    for column, weight in zip(feature_columns, weights):
        group = feature_group(column)
        groups.setdefault(group, {"positive_weight": 0.0, "negative_weight": 0.0, "net_weight": 0.0, "features": []})
        if weight >= 0:
            groups[group]["positive_weight"] += float(weight)
        else:
            groups[group]["negative_weight"] += float(weight)
        groups[group]["net_weight"] += float(weight)
        groups[group]["features"].append({"feature": column, "weight": float(weight)})
    for values in groups.values():
        values["helpful_or_harmful"] = "helpful_to_up_probability" if values["net_weight"] > 0 else "harmful_to_up_probability"
        values["features"] = sorted(values["features"], key=lambda item: abs(item["weight"]), reverse=True)[:8]
    return groups


def flag_series(frame, column):
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float) > 0.5


def numeric_series(frame, column):
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def subset_strategy_metrics(frame, actual_return, direction, mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"rows": 0, "active_rows": 0, "avg_return": np.nan, "win_rate": np.nan, "coverage": 0.0}
    return strategy_metrics(actual_return[mask], direction[mask])


def bucket_accountability(segment, x_segment, direction_weights, direction_bias, horizon, threshold=PREDICTION_THRESHOLD):
    actual = segment[f"target_realized_return_{horizon}"].to_numpy(dtype=np.float64)
    probability = sigmoid(x_segment @ direction_weights + direction_bias)
    direction = direction_from_probability(probability, threshold)
    pressure = numeric_series(segment, "feature_flow_1s_pred_pressure")
    max_scare = numeric_series(segment, "feature_max_micro_scare_probability")
    max_liquidity = numeric_series(segment, "feature_max_liquidity_drop_probability")
    buckets = {
        "1s_pressure_positive": pressure > 0,
        "1s_pressure_negative": pressure < 0,
        "1s_high_abs_pressure": pressure.abs() >= 0.50,
        "1s_class_pressure_disagreement": flag_series(segment, "feature_flow_1s_class_pressure_disagreement"),
        "10s_high_scare": max_scare >= 0.70,
        "10s_high_liquidity_drop": max_liquidity >= 0.70,
        "10s_spread_risk": flag_series(segment, "feature_spread_expansion_risk_high"),
        "both_agree_bullish": flag_series(segment, "feature_both_agree_bullish"),
        "both_agree_bearish": flag_series(segment, "feature_both_agree_bearish"),
        "flow_bullish_micro_bearish_conflict": flag_series(segment, "feature_flow_bullish_micro_bearish_conflict"),
        "flow_bearish_micro_bullish_conflict": flag_series(segment, "feature_flow_bearish_micro_bullish_conflict"),
        "path_3m_stale_or_unavailable": flag_series(segment, "feature_path_3m_stale_or_unavailable"),
        "abstain_3m_unavailable": flag_series(segment, "feature_abstain_3m_unavailable"),
        "abstain_regime_unavailable": flag_series(segment, "feature_abstain_regime_unavailable"),
        "abstain_htf_unavailable": flag_series(segment, "feature_abstain_htf_unavailable"),
    }
    overall = strategy_metrics(actual, direction)
    rows = {}
    for name, mask in buckets.items():
        metrics = subset_strategy_metrics(segment, actual, direction, mask)
        metrics["rate"] = float(np.asarray(mask, dtype=bool).mean()) if len(segment) else 0.0
        metrics["lift_vs_overall_avg_return"] = (
            float(metrics["avg_return"] - overall["avg_return"])
            if np.isfinite(metrics["avg_return"])
            else np.nan
        )
        metrics["helpful_or_harmful"] = (
            "helpful"
            if np.isfinite(metrics["avg_return"]) and metrics["avg_return"] > overall["avg_return"]
            else "harmful_or_neutral"
        )
        rows[name] = metrics
    return {"overall": overall, "buckets": rows}


def stale_unavailable_rates(frame):
    candidates = [
        column
        for column in frame.columns
        if column.startswith("feature_")
        and (
            "available" in column
            or "unavailable" in column
            or "stale" in column
            or "abstain" in column
            or "event_only" in column
        )
    ]
    rates = {}
    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
        rates[column] = float((values > 0.5).mean()) if len(values) else 0.0
    return rates


def print_metrics_line(label, metrics):
    win_rate = metrics["win_rate"]
    win_text = f"{win_rate:.2%}" if np.isfinite(win_rate) else "n/a"
    print(
        f"- {label}: rows={metrics['rows']}, active={metrics['active_rows']}, "
        f"coverage={metrics['coverage']:.2%}, win={win_text}, avg_return={metrics['avg_return']:.4%}"
    )


def main():
    frame = read_csv(TRAINING_PATH)
    if len(frame) < MIN_ROWS:
        print("Hierarchy mediator training skipped.")
        print(f"Rows: {len(frame)} < MEDIATOR_MIN_ROWS={MIN_ROWS}")
        print(f"Training path: {TRAINING_PATH}")
        print("No trades/orders/private API.")
        return
    feature_columns = sorted(column for column in frame.columns if column.startswith("feature_"))
    required_targets = [f"target_realized_return_{h}" for h in HORIZONS]
    required_targets += [f"target_realized_direction_{h}" for h in HORIZONS]
    frame = frame.dropna(subset=["timestamp", *feature_columns, *required_targets]).copy()
    train, validation, test = split_chronological(frame)
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("Chronological split produced an empty train/validation/test segment.")

    x_train = train[feature_columns].to_numpy(dtype=np.float64)
    x_validation = validation[feature_columns].to_numpy(dtype=np.float64)
    x_test = test[feature_columns].to_numpy(dtype=np.float64)
    x_train, x_validation, x_test, feature_mean, feature_std = standardize(x_train, x_validation, x_test)

    direction_models = {}
    return_models = {}
    predictions = []
    weights_by_horizon = {}
    mediator_report = {}
    segment_frames = {
        "train": train,
        "validation": validation,
        "forward_test": test,
    }
    segment_matrices = {
        "train": x_train,
        "validation": x_validation,
        "forward_test": x_test,
    }
    for horizon in HORIZONS:
        y_train_direction = (train[f"target_realized_direction_{horizon}"].to_numpy(dtype=np.int64) > 0).astype(float)
        y_test_return = test[f"target_realized_return_{horizon}"].to_numpy(dtype=np.float64)
        direction_weights, direction_bias = train_logistic(x_train, y_train_direction)
        return_weights, return_bias = train_linear_ridge(
            x_train,
            train[f"target_realized_return_{horizon}"].to_numpy(dtype=np.float64),
        )
        direction_models[horizon] = {
            "weights": direction_weights.tolist(),
            "bias": float(direction_bias),
        }
        return_models[horizon] = {
            "weights": return_weights.tolist(),
            "bias": float(return_bias),
        }
        weights_by_horizon[horizon] = direction_weights
        test_probability = sigmoid(x_test @ direction_weights + direction_bias)
        test_predicted_return = x_test @ return_weights + return_bias
        mediator_direction = direction_from_probability(test_probability)
        segment_reports = {
            name: evaluate_model_on_segment(
                segment_frames[name],
                segment_matrices[name],
                direction_weights,
                direction_bias,
                train,
                horizon,
                PREDICTION_THRESHOLD,
            )
            for name in ["train", "validation", "forward_test"]
        }
        mediator_report[horizon] = {
            "segments": segment_reports,
            "forward_test_threshold_sweep": threshold_sweep(
                test,
                x_test,
                direction_weights,
                direction_bias,
                train,
                horizon,
            ),
            "feature_weight_report": feature_weight_report(feature_columns, direction_weights),
            "group_help_harm": group_help_harm(feature_columns, direction_weights),
            "forward_test_bucket_accountability": bucket_accountability(
                test,
                x_test,
                direction_weights,
                direction_bias,
                horizon,
                PREDICTION_THRESHOLD,
            ),
        }
        for row_index, (_, row) in enumerate(test.reset_index(drop=True).iterrows()):
            if horizon == "60s":
                predictions.append(
                    {
                        "timestamp": int(row["timestamp"]),
                        "time": row.get("time", ""),
                        "prob_up_60s": float(test_probability[row_index]),
                        "predicted_return_60s": float(test_predicted_return[row_index]),
                        "predicted_direction_60s": int(mediator_direction[row_index]),
                        "actual_return_60s": float(y_test_return[row_index]),
                        "actual_direction_60s": int(row[f"target_realized_direction_{horizon}"]),
                    }
                )

    tag = current_utc_tag()
    schema_hash = hashlib.sha256("\n".join(feature_columns).encode("utf-8")).hexdigest()[:16]
    artifact = {
        "model_type": "paper_only_hierarchy_mediator_numpy_logistic",
        "symbol": SYMBOL,
        "primary_venue": VENUE_TAG,
        "created_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model_id": f"{SYMBOL}_{VENUE_TAG}_hierarchy_mediator_{tag}_{schema_hash}",
        "trained_until_timestamp": int(train["timestamp"].max()),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "direction_models": direction_models,
        "return_models": return_models,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "threshold_sweep_values": THRESHOLD_SWEEP,
        "split": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "train_start": int(train["timestamp"].min()),
            "train_end": int(train["timestamp"].max()),
            "validation_start": int(validation["timestamp"].min()),
            "validation_end": int(validation["timestamp"].max()),
            "test_start": int(test["timestamp"].min()),
            "test_end": int(test["timestamp"].max()),
        },
        "mediator_report": mediator_report,
        "test_report": mediator_report,
        "top_feature_importances": top_feature_importances(feature_columns, weights_by_horizon),
        "accountability_60s": accountability(feature_columns, weights_by_horizon["60s"]),
        "stale_unavailable_rates": stale_unavailable_rates(frame),
        "promoted": False,
        "paper_only": True,
    }
    candidate_dir = CANDIDATE_ROOT / tag
    atomic_write_json(artifact, candidate_dir / "model.json")
    atomic_write_csv(pd.DataFrame(predictions), candidate_dir / "test_predictions.csv")

    print("Hierarchy mediator model trained")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {VENUE_TAG}")
    print(f"Training rows: {len(frame)}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Candidate model: {candidate_dir / 'model.json'}")
    for horizon in HORIZONS:
        report = mediator_report[horizon]
        print(f"\n===== Hierarchy mediator diagnostics: {horizon} =====")
        for segment_name in ["train", "validation", "forward_test"]:
            segment_report = report["segments"][segment_name]
            print(f"\nSegment: {segment_name}")
            print_metrics_line("mediator", segment_report["mediator"])
            for baseline_name in [
                "majority_direction",
                "always_long",
                "always_short",
                "follow_1s_pressure",
                "best_combined_filter",
            ]:
                print_metrics_line(baseline_name, segment_report["baselines"][baseline_name])
            print(f"- lift vs majority avg return: {segment_report['lift_vs_majority_avg_return']:.4%}")
            print(f"- lift vs best combined filter avg return: {segment_report['lift_vs_best_combined_filter_avg_return']:.4%}")

        print("\nForward-test probability threshold sweep")
        print("threshold | rows_predicted | win_rate | avg_return | lift_vs_majority | lift_vs_best_filter")
        for row in report["forward_test_threshold_sweep"]:
            win_text = f"{row['win_rate']:.2%}" if np.isfinite(row["win_rate"]) else "n/a"
            print(
                f"{row['threshold']:.3f} | "
                f"{row['rows_predicted']}/{row['rows']} | "
                f"{win_text} | "
                f"{row['avg_return']:.4%} | "
                f"{row['lift_vs_majority_avg_return']:.4%} | "
                f"{row['lift_vs_best_combined_filter_avg_return']:.4%}"
            )

        if horizon == "60s":
            weight_report = report["feature_weight_report"]
            print("\n60s feature attribution: top positive features")
            for item in weight_report["top_positive_features"][:12]:
                print(f"- {item['feature']}: weight={item['weight']:+.6g} ({item['group']})")
            print("\n60s feature attribution: top negative features")
            for item in weight_report["top_negative_features"][:12]:
                print(f"- {item['feature']}: weight={item['weight']:+.6g} ({item['group']})")
            print("\n60s near-zero / unstable low-signal features")
            near_zero = weight_report["near_zero_features"]
            if not near_zero:
                print("- none at current near-zero threshold")
            for item in near_zero[:20]:
                print(f"- {item['feature']}: weight={item['weight']:+.6g}")

            print("\n60s model accountability: group weights")
            for group, values in sorted(report["group_help_harm"].items(), key=lambda item: abs(item[1]["net_weight"]), reverse=True):
                print(
                    f"- {group}: net={values['net_weight']:+.6g}, "
                    f"positive={values['positive_weight']:+.6g}, "
                    f"negative={values['negative_weight']:+.6g}, "
                    f"{values['helpful_or_harmful']}"
                )

            print("\n60s model accountability: forward-test buckets")
            bucket_report = report["forward_test_bucket_accountability"]
            print_metrics_line("overall mediator", bucket_report["overall"])
            for bucket_name, metrics in bucket_report["buckets"].items():
                win_text = f"{metrics['win_rate']:.2%}" if np.isfinite(metrics["win_rate"]) else "n/a"
                avg_text = f"{metrics['avg_return']:.4%}" if np.isfinite(metrics["avg_return"]) else "n/a"
                lift_text = (
                    f"{metrics['lift_vs_overall_avg_return']:.4%}"
                    if np.isfinite(metrics["lift_vs_overall_avg_return"])
                    else "n/a"
                )
                print(
                    f"- {bucket_name}: rows={metrics['rows']}, rate={metrics['rate']:.2%}, "
                    f"active={metrics['active_rows']}, win={win_text}, "
                    f"avg_return={avg_text}, lift_vs_overall={lift_text}, "
                    f"{metrics['helpful_or_harmful']}"
                )

            print("\nStale/unavailable/context rates")
            rates = artifact["stale_unavailable_rates"]
            if not rates:
                print("- no stale/unavailable feature columns found")
            for column, rate in sorted(rates.items()):
                print(f"- {column}: {rate:.2%}")

    print("\nTop cross-horizon feature importances")
    for item in artifact["top_feature_importances"][:15]:
        print(f"- {item['feature']}: {item['importance']:.6g} ({item['group']})")
    print("\nModel accountability 60s")
    for group, values in sorted(artifact["accountability_60s"].items(), key=lambda item: item[1]["abs_weight"], reverse=True):
        direction = "helped bullish trust" if values["signed_weight"] > 0 else "hurt bullish trust"
        print(f"- {group}: abs_weight={values['abs_weight']:.6g}, signed_weight={values['signed_weight']:.6g}, {direction}")
    print("No trades/orders/private API. No promotion.")


if __name__ == "__main__":
    main()
