import datetime as dt
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from live_3m_model_utils import atomic_write_csv
from offer_model_utils import (
    MODEL_TARGET_COLUMNS,
    add_offer_input_columns,
    choose_model_feature_columns,
    compounded_return_and_drawdown,
    forward,
    initialize_model,
    load_model,
    predict_artifact,
    predict_artifact_full,
    save_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "live_training" / f"{SYMBOL}_offer_training_rows.csv"
INPUT_PATH = Path(os.getenv("OFFER_TRAINING_PATH", str(DEFAULT_INPUT_PATH)))
if not INPUT_PATH.is_absolute():
    INPUT_PATH = PROJECT_ROOT / INPUT_PATH
MODEL_OUTPUT_TAG = os.getenv("MODEL_OUTPUT_TAG", "live").strip() or "live"
PROMOTE_BEST = os.getenv("PROMOTE_BEST", "false").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "active" / f"{SYMBOL}_offer_acceptance_model.json"
CANDIDATE_ROOT = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / SYMBOL
    / "offer_acceptance"
    / MODEL_OUTPUT_TAG
)

MIN_OFFER_ROWS = int(os.getenv("MIN_OFFER_ROWS", "1000"))
MIN_VALIDATION_ROWS = int(os.getenv("MIN_VALIDATION_ROWS", "200"))
MIN_ACCEPTED_VALIDATION_ROWS = int(os.getenv("MIN_ACCEPTED_VALIDATION_ROWS", "20"))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.8"))
EPOCHS = int(os.getenv("EPOCHS", "60"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
HIDDEN_UNITS = int(os.getenv("HIDDEN_UNITS", "48"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))
REGRESSION_LOSS_WEIGHT = float(os.getenv("REGRESSION_LOSS_WEIGHT", "0.25"))
ACCEPT_THRESHOLD = float(os.getenv("ACCEPT_THRESHOLD", "0.60"))
MAX_ACCEPT_RATE = float(os.getenv("MAX_ACCEPT_RATE", "0.80"))
MIN_ACCEPT_RATE = float(os.getenv("MIN_ACCEPT_RATE", "0.01"))
MAX_AMBIGUOUS_ACCEPT_RATE = float(os.getenv("MAX_AMBIGUOUS_ACCEPT_RATE", "0.35"))
MAX_SIDE_DOMINANCE = float(os.getenv("MAX_SIDE_DOMINANCE", "0.85"))
MAX_DRAWDOWN_WORSE_ALLOWANCE = float(os.getenv("MAX_DRAWDOWN_WORSE_ALLOWANCE", "0.02"))


def load_rows():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing offer training rows: {INPUT_PATH}. Run npm run offer-build first."
        )
    frame = pd.read_csv(INPUT_PATH)
    required = [
        "timestamp",
        "offer_side",
        "accept_target",
        "target_allocation_bucket",
        *MODEL_TARGET_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Offer training CSV is missing required columns: {missing}")
    for column in frame.columns:
        if column not in {"time", "symbol", "offer_side", "hit_result"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).copy()
    frame = frame[frame["accept_target"].isin([0, 1])]
    frame = frame[frame["target_allocation_bucket"].isin([0, 1, 2, 3, 4, 5])]
    frame["target_allocation_bucket"] = frame["target_allocation_bucket"].astype(int)
    return frame.sort_values("timestamp").reset_index(drop=True)


def split_time_ordered(frame):
    split_index = int(len(frame) * TRAIN_SPLIT)
    split_index = max(1, min(split_index, len(frame) - 1))
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def standardize(train_values, validation_values):
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train_values - mean) / std, (validation_values - mean) / std, mean, std


def one_hot(values, class_count):
    result = np.zeros((len(values), class_count))
    result[np.arange(len(values)), values.astype(int)] = 1.0
    return result


def train_model(
    x_train,
    y_accept_train,
    y_bucket_train,
    y_reg_train,
    x_validation,
    y_accept_validation,
    y_bucket_validation,
    y_reg_validation,
):
    rng = np.random.default_rng(RANDOM_SEED)
    model = initialize_model(x_train.shape[1], HIDDEN_UNITS, y_reg_train.shape[1], rng)
    y_accept_one_hot = one_hot(y_accept_train, 2)
    y_bucket_one_hot = one_hot(y_bucket_train, 6)
    class_counts = np.bincount(y_accept_train.astype(int), minlength=2).astype(float)
    class_weights = len(y_accept_train) / (2.0 * np.maximum(class_counts, 1.0))
    bucket_counts = np.bincount(y_bucket_train.astype(int), minlength=6).astype(float)
    bucket_weights = len(y_bucket_train) / (6.0 * np.maximum(bucket_counts, 1.0))
    best_model = {name: value.copy() for name, value in model.items()}
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        for start in range(0, len(x_train), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(x_train))
            xb = x_train[start:end]
            yb = y_accept_train[start:end].astype(int)
            ybucket = y_bucket_train[start:end].astype(int)
            yb_one_hot = y_accept_one_hot[start:end]
            ybucket_one_hot = y_bucket_one_hot[start:end]
            yr = y_reg_train[start:end]
            weights = class_weights[yb][:, None]
            bucket_batch_weights = bucket_weights[ybucket][:, None]

            hidden_pre, hidden, probabilities, bucket_probabilities, regression = forward(model, xb)
            d_class = (probabilities - yb_one_hot) * weights / len(xb)
            d_bucket = (bucket_probabilities - ybucket_one_hot) * bucket_batch_weights / len(xb)
            d_reg = REGRESSION_LOSS_WEIGHT * 2.0 * (regression - yr) / max(1, len(xb) * yr.shape[1])
            gradients = {
                "w_class": hidden.T @ d_class,
                "b_class": d_class.sum(axis=0),
                "w_bucket": hidden.T @ d_bucket,
                "b_bucket": d_bucket.sum(axis=0),
                "w_reg": hidden.T @ d_reg,
                "b_reg": d_reg.sum(axis=0),
            }
            d_hidden = (
                d_class @ model["w_class"].T
                + d_bucket @ model["w_bucket"].T
                + d_reg @ model["w_reg"].T
            )
            d_hidden[hidden_pre <= 0] = 0.0
            gradients["w1"] = xb.T @ d_hidden
            gradients["b1"] = d_hidden.sum(axis=0)
            for name, gradient in gradients.items():
                model[name] -= LEARNING_RATE * gradient

        _, _, validation_probabilities, validation_bucket_probabilities, validation_regression = forward(model, x_validation)
        class_loss = -np.mean(
            np.log(
                np.clip(
                    validation_probabilities[np.arange(len(y_accept_validation)), y_accept_validation.astype(int)],
                    1e-8,
                    1.0,
                )
            )
        )
        bucket_loss = -np.mean(
            np.log(
                np.clip(
                    validation_bucket_probabilities[np.arange(len(y_bucket_validation)), y_bucket_validation.astype(int)],
                    1e-8,
                    1.0,
                )
            )
        )
        regression_loss = np.mean((validation_regression - y_reg_validation) ** 2)
        total_loss = class_loss + bucket_loss + REGRESSION_LOSS_WEIGHT * regression_loss
        if total_loss < best_loss:
            best_loss = total_loss
            best_model = {name: value.copy() for name, value in model.items()}

        if epoch == 1 or epoch == EPOCHS or epoch % 10 == 0:
            accuracy = np.mean(np.argmax(validation_probabilities, axis=1) == y_accept_validation)
            bucket_accuracy = np.mean(np.argmax(validation_bucket_probabilities, axis=1) == y_bucket_validation)
            print(
                f"epoch {epoch:03d} | validation loss {total_loss:.6f} | "
                f"accept accuracy {accuracy:.2%} | bucket accuracy {bucket_accuracy:.2%}"
            )

    return best_model


def metrics_for_probabilities(frame, accept_probability):
    accepted = frame[accept_probability >= ACCEPT_THRESHOLD].copy()
    rejected_count = len(frame) - len(accepted)
    if len(accepted) == 0:
        return {
            "accepted_count": 0,
            "rejected_count": rejected_count,
            "accept_rate": 0.0,
            "average_opportunity_score": -1e9,
            "average_net_return": -1e9,
            "win_rate_after_costs": 0.0,
            "max_drawdown": 0.0,
            "winner_avg_accept_probability": 0.0,
            "loser_avg_accept_probability": 0.0,
            "ambiguous_accept_rate": 0.0,
            "max_side_share": 1.0,
        }
    accepted["accept_probability"] = accept_probability[accept_probability >= ACCEPT_THRESHOLD]
    winners = accepted[accepted["net_return"] > 0]
    losers = accepted[accepted["net_return"] <= 0]
    _, max_drawdown = compounded_return_and_drawdown(accepted["net_return"])
    side_counts = accepted["offer_side"].value_counts(normalize=True)
    return {
        "accepted_count": int(len(accepted)),
        "rejected_count": int(rejected_count),
        "accept_rate": float(len(accepted) / len(frame)),
        "average_opportunity_score": float(accepted["opportunity_score"].mean()),
        "average_net_return": float(accepted["net_return"].mean()),
        "win_rate_after_costs": float((accepted["net_return"] > 0).mean()),
        "max_drawdown": float(max_drawdown),
        "winner_avg_accept_probability": float(winners["accept_probability"].mean()) if len(winners) else 0.0,
        "loser_avg_accept_probability": float(losers["accept_probability"].mean()) if len(losers) else 0.0,
        "ambiguous_accept_rate": float((accepted["hit_result"] == "ambiguous").mean()) if "hit_result" in accepted.columns else 0.0,
        "max_side_share": float(side_counts.max()) if len(side_counts) else 1.0,
    }


def evaluate_artifact(artifact, validation_frame):
    accept_probability, bucket_probabilities, regression = predict_artifact_full(artifact, validation_frame)
    metrics = metrics_for_probabilities(validation_frame, accept_probability)
    opportunity_index = artifact["target_columns"].index("opportunity_score")
    metrics["regression_opportunity_mae"] = float(
        np.mean(
            np.abs(
                regression[:, opportunity_index]
                - validation_frame["opportunity_score"].to_numpy()
            )
        )
    )
    metrics["allocation_bucket_accuracy"] = float(
        np.mean(np.argmax(bucket_probabilities, axis=1) == validation_frame["target_allocation_bucket"].to_numpy(dtype=int))
    )
    return metrics


def should_promote(candidate_metrics, active_metrics, validation):
    reasons = []
    accepted_examples = int(validation["accept_target"].sum())
    rejected_examples = len(validation) - accepted_examples
    if len(validation) < MIN_VALIDATION_ROWS:
        reasons.append("validation rows below MIN_VALIDATION_ROWS")
    if accepted_examples < MIN_ACCEPTED_VALIDATION_ROWS:
        reasons.append("accepted validation labels below MIN_ACCEPTED_VALIDATION_ROWS")
    if rejected_examples < MIN_ACCEPTED_VALIDATION_ROWS:
        reasons.append("rejected validation labels below minimum balance check")
    if candidate_metrics["accepted_count"] == 0:
        reasons.append("zero accepted validation offers")
    if candidate_metrics["accept_rate"] > MAX_ACCEPT_RATE:
        reasons.append("model accepts almost everything")
    if candidate_metrics["accept_rate"] < MIN_ACCEPT_RATE:
        reasons.append("model rejects everything")
    if candidate_metrics["average_net_return"] <= 0:
        reasons.append("validation net_return is negative or zero")
    if candidate_metrics["winner_avg_accept_probability"] <= candidate_metrics["loser_avg_accept_probability"]:
        reasons.append("winner average accept_probability is not greater than loser average")
    if candidate_metrics["ambiguous_accept_rate"] > MAX_AMBIGUOUS_ACCEPT_RATE:
        reasons.append("accepted offers are mostly ambiguous hit rows")
    if candidate_metrics["max_side_share"] > MAX_SIDE_DOMINANCE:
        reasons.append("accepted offers are too concentrated on one side")

    if active_metrics is not None:
        if candidate_metrics["average_opportunity_score"] <= active_metrics["average_opportunity_score"]:
            reasons.append("candidate average opportunity_score did not improve versus active model")
        allowed_drawdown = active_metrics["max_drawdown"] - MAX_DRAWDOWN_WORSE_ALLOWANCE
        if candidate_metrics["max_drawdown"] < allowed_drawdown:
            reasons.append("candidate max drawdown is worse than allowance")

    return len(reasons) == 0, reasons


def main():
    frame = load_rows()
    print("Paper-only offer acceptance trainer")
    print(f"SYMBOL: {SYMBOL}")
    print(f"Input path: {INPUT_PATH}")
    print(f"MODEL_OUTPUT_TAG: {MODEL_OUTPUT_TAG}")
    print(f"PROMOTE_BEST: {PROMOTE_BEST}")
    print(f"Rows loaded: {len(frame)}")
    print(f"MIN_OFFER_ROWS: {MIN_OFFER_ROWS}")
    if len(frame) < MIN_OFFER_ROWS:
        print("Training skipped: not enough offer rows.")
        print("Candidate promoted: no")
        print("No trades were placed.")
        return

    feature_columns = choose_model_feature_columns(frame)
    frame = add_offer_input_columns(frame)
    train, validation = split_time_ordered(frame)
    x_train_raw = train[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    x_validation_raw = validation[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    y_accept_train = train["accept_target"].to_numpy(dtype=int)
    y_accept_validation = validation["accept_target"].to_numpy(dtype=int)
    y_bucket_train = train["target_allocation_bucket"].to_numpy(dtype=int)
    y_bucket_validation = validation["target_allocation_bucket"].to_numpy(dtype=int)
    y_reg_train_raw = train[MODEL_TARGET_COLUMNS].to_numpy(dtype=np.float64)
    y_reg_validation_raw = validation[MODEL_TARGET_COLUMNS].to_numpy(dtype=np.float64)
    x_train, x_validation, feature_mean, feature_std = standardize(x_train_raw, x_validation_raw)
    y_reg_train, y_reg_validation, target_mean, target_std = standardize(y_reg_train_raw, y_reg_validation_raw)

    print(f"Feature count: {len(feature_columns)}")
    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print("No rows are shuffled; validation is later in time.")
    print("Validation accept_target distribution:")
    print(validation["accept_target"].value_counts().sort_index())
    print("Validation target_allocation_bucket distribution:")
    print(validation["target_allocation_bucket"].value_counts().sort_index())

    model = train_model(
        x_train,
        y_accept_train,
        y_bucket_train,
        y_reg_train,
        x_validation,
        y_accept_validation,
        y_bucket_validation,
        y_reg_validation,
    )
    artifact = {
        "model_type": "paper_offer_acceptance_numpy_mlp",
        "symbol": SYMBOL,
        "created_at": (
            dt.datetime.now(dt.UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        "feature_columns": feature_columns,
        "target_columns": MODEL_TARGET_COLUMNS,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "model": model,
        "accept_threshold": ACCEPT_THRESHOLD,
    }
    candidate_metrics = evaluate_artifact(artifact, validation)
    active = load_model(ACTIVE_MODEL_PATH) if PROMOTE_BEST else None
    if active is not None and active.get("target_columns") == MODEL_TARGET_COLUMNS:
        active_metrics = evaluate_artifact(active, validation)
    else:
        active_metrics = None
        if active is not None:
            print("Active offer model uses an older target schema; skipping active comparison.")
    promote, reasons = should_promote(candidate_metrics, active_metrics, validation)
    if not PROMOTE_BEST:
        promote = False
        reasons = ["PROMOTE_BEST is false; candidate saved but active model is not overwritten"]

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate_dir = CANDIDATE_ROOT / timestamp
    candidate_path = candidate_dir / "model.json"
    save_model(candidate_path, artifact)
    metrics = pd.DataFrame(
        [
            {"model": "candidate", **candidate_metrics},
            *([{"model": "active", **active_metrics}] if active_metrics else []),
        ]
    )
    atomic_write_csv(metrics, candidate_dir / "validation_metrics.csv")

    print("\nCandidate validation metrics:")
    for key, value in candidate_metrics.items():
        print(f"- {key}: {value:.6g}" if isinstance(value, float) else f"- {key}: {value}")
    if active_metrics:
        print("\nActive validation metrics:")
        for key, value in active_metrics.items():
            print(f"- {key}: {value:.6g}" if isinstance(value, float) else f"- {key}: {value}")
    else:
        print("\nNo active offer model was available.")

    if promote:
        ACTIVE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, ACTIVE_MODEL_PATH)
        print(f"\nCandidate promoted to active offer model: {ACTIVE_MODEL_PATH}")
    else:
        print("\nCandidate rejected.")
        for reason in reasons:
            print(f"- {reason}")

    print(f"Candidate model saved to: {candidate_path}")
    print(f"Candidate promoted: {'yes' if promote else 'no'}")
    print("No trades were placed.")


if __name__ == "__main__":
    main()
