import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
INPUT_PATH = PROJECT_ROOT / "data" / "ensemble" / f"{SYMBOL}_ensemble_training_rows.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "ensemble" / f"{SYMBOL}_ensemble_predictions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "ensemble" / SYMBOL / "ensemble_meta_model.json"

TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.80"))
EPOCHS = int(os.getenv("EPOCHS", "80"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
HIDDEN_UNITS = int(os.getenv("HIDDEN_UNITS", "32"))
L2 = float(os.getenv("L2", "0.0001"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))

CLASS_NAMES = {
    0: "short_win",
    1: "neutral",
    2: "long_win",
}

EXCLUDED_FEATURE_COLUMNS = {
    "time",
    "timestamp",
    "actual_class",
    "future_return_3",
    "old_prediction_timestamp",
    "flow_prediction_timestamp",
    "feature_timestamp",
    "regime_close_timestamp",
}


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(logits, -40, 40))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def one_hot(classes):
    output = np.zeros((len(classes), 3), dtype=np.float64)
    output[np.arange(len(classes)), classes.astype(int)] = 1.0
    return output


def choose_feature_columns(frame):
    columns = []
    for column in frame.columns:
        if column in EXCLUDED_FEATURE_COLUMNS:
            continue
        if column.endswith("_timestamp") or "timestamp" in column:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def load_dataset():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing ensemble rows: {INPUT_PATH}. "
            "Run scripts/build_ensemble_training_rows.py first."
        )

    frame = pd.read_csv(INPUT_PATH)
    required = ["time", "timestamp", "actual_class", "future_return_3"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Ensemble rows are missing required columns: {missing}")

    for column in frame.columns:
        if column != "time":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    feature_columns = choose_feature_columns(frame)
    if not feature_columns:
        raise RuntimeError("No numeric ensemble feature columns found.")

    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    usable = frame.dropna(subset=feature_columns + ["actual_class", "future_return_3"])
    usable = usable[usable["actual_class"].isin([0, 1, 2])].copy()
    usable["actual_class"] = usable["actual_class"].astype(int)
    usable = usable.reset_index(drop=True)
    if len(usable) < 10:
        raise RuntimeError(
            f"Only {len(usable)} usable ensemble rows found. "
            "The meta-model needs more aligned prediction rows."
        )

    return usable, feature_columns


def split_time_ordered(frame):
    split_index = int(len(frame) * TRAIN_SPLIT)
    split_index = max(1, min(split_index, len(frame) - 1))
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def standardize(train_values, validation_values):
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_values - mean) / std, (validation_values - mean) / std, mean, std


def initialize_model(input_size, rng):
    return {
        "w1": rng.normal(0, np.sqrt(2 / max(1, input_size)), (input_size, HIDDEN_UNITS)),
        "b1": np.zeros(HIDDEN_UNITS),
        "w2": rng.normal(0, np.sqrt(2 / max(1, HIDDEN_UNITS)), (HIDDEN_UNITS, 3)),
        "b2": np.zeros(3),
    }


def forward(model, x):
    hidden_pre = x @ model["w1"] + model["b1"]
    hidden = np.maximum(0, hidden_pre)
    logits = hidden @ model["w2"] + model["b2"]
    probabilities = softmax(logits)
    return hidden_pre, hidden, logits, probabilities


def loss_and_accuracy(model, x, classes):
    _, _, _, probabilities = forward(model, x)
    clipped = np.clip(probabilities[np.arange(len(classes)), classes], 1e-8, 1.0)
    loss = float(-np.mean(np.log(clipped)))
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predictions == classes))
    return loss, accuracy


def train_model(x_train, y_train, x_validation, y_validation):
    rng = np.random.default_rng(RANDOM_SEED)
    model = initialize_model(x_train.shape[1], rng)
    y_train_one_hot = one_hot(y_train)
    best_model = json.loads(json.dumps({key: value.tolist() for key, value in model.items()}))
    best_validation_loss = float("inf")

    class_counts = np.bincount(y_train, minlength=3).astype(np.float64)
    class_weights = len(y_train) / (3.0 * np.maximum(class_counts, 1.0))

    for epoch in range(1, EPOCHS + 1):
        for start in range(0, len(x_train), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(x_train))
            xb = x_train[start:end]
            yb = y_train[start:end]
            yb_one_hot = y_train_one_hot[start:end]
            weights = class_weights[yb][:, None]

            hidden_pre, hidden, _, probabilities = forward(model, xb)
            dlogits = (probabilities - yb_one_hot) * weights / len(xb)
            dw2 = hidden.T @ dlogits + L2 * model["w2"]
            db2 = dlogits.sum(axis=0)
            dhidden = dlogits @ model["w2"].T
            dhidden[hidden_pre <= 0] = 0
            dw1 = xb.T @ dhidden + L2 * model["w1"]
            db1 = dhidden.sum(axis=0)

            model["w2"] -= LEARNING_RATE * dw2
            model["b2"] -= LEARNING_RATE * db2
            model["w1"] -= LEARNING_RATE * dw1
            model["b1"] -= LEARNING_RATE * db1

        train_loss, train_accuracy = loss_and_accuracy(model, x_train, y_train)
        validation_loss, validation_accuracy = loss_and_accuracy(
            model, x_validation, y_validation
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model = {key: value.tolist() for key, value in model.items()}

        if epoch == 1 or epoch == EPOCHS or epoch % 10 == 0:
            print(
                f"epoch {epoch:03d} | train loss {train_loss:.5f} "
                f"acc {train_accuracy:.2%} | validation loss {validation_loss:.5f} "
                f"acc {validation_accuracy:.2%}"
            )

    return {key: np.array(value) for key, value in best_model.items()}


def confusion_matrix(actual, predicted):
    matrix = np.zeros((3, 3), dtype=int)
    for actual_class, predicted_class in zip(actual, predicted):
        matrix[int(actual_class), int(predicted_class)] += 1
    return matrix


def print_metrics(actual, probabilities):
    predicted = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predicted == actual))
    majority_class = int(pd.Series(actual).mode().iloc[0])
    majority_accuracy = float(np.mean(actual == majority_class))

    print(f"\nValidation accuracy: {accuracy:.2%}")
    print(
        f"Majority-class baseline: class {majority_class} "
        f"({CLASS_NAMES[majority_class]}) at {majority_accuracy:.2%}"
    )
    print("\nConfusion matrix rows=actual, columns=predicted")
    print(confusion_matrix(actual, predicted))
    print("\nPer-class precision / recall")
    for class_id in [0, 1, 2]:
        true_positive = int(((predicted == class_id) & (actual == class_id)).sum())
        predicted_positive = int((predicted == class_id).sum())
        actual_positive = int((actual == class_id).sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        print(
            f"- class {class_id} {CLASS_NAMES[class_id]}: "
            f"precision {precision:.2%}, recall {recall:.2%}"
        )


def print_class_distribution(label, classes):
    print(label)
    counts = pd.Series(classes).value_counts().sort_index()
    total = len(classes) or 1
    for class_id in [0, 1, 2]:
        count = int(counts.get(class_id, 0))
        print(f"- class {class_id} {CLASS_NAMES[class_id]}: {count} ({count / total:.2%})")


def save_predictions(validation_frame, probabilities):
    output = pd.DataFrame(
        {
            "time": validation_frame["time"].values,
            "timestamp": validation_frame["timestamp"].values,
            "actual_class": validation_frame["actual_class"].astype(int).values,
            "pred_class": np.argmax(probabilities, axis=1),
            "prob_short": probabilities[:, 0],
            "prob_neutral": probabilities[:, 1],
            "prob_long": probabilities[:, 2],
            "future_return_3": validation_frame["future_return_3"].values,
        }
    )
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS_PATH, index=False)


def main():
    frame, feature_columns = load_dataset()
    train_frame, validation_frame = split_time_ordered(frame)

    x_train = train_frame[feature_columns].to_numpy(dtype=np.float64)
    x_validation = validation_frame[feature_columns].to_numpy(dtype=np.float64)
    y_train = train_frame["actual_class"].to_numpy(dtype=int)
    y_validation = validation_frame["actual_class"].to_numpy(dtype=int)
    x_train_scaled, x_validation_scaled, mean, std = standardize(x_train, x_validation)

    print("Stacked ensemble meta-model")
    print(f"SYMBOL: {SYMBOL}")
    print(f"Input path: {INPUT_PATH}")
    print(f"Prediction output path: {PREDICTIONS_PATH}")
    print(f"Model output path: {MODEL_PATH}")
    print(f"Rows loaded: {len(frame)}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Train rows: {len(train_frame)}")
    print(f"Validation rows: {len(validation_frame)}")
    print(
        f"Train range: {train_frame['time'].iloc[0]} -> {train_frame['time'].iloc[-1]}"
    )
    print(
        "Validation range: "
        f"{validation_frame['time'].iloc[0]} -> {validation_frame['time'].iloc[-1]}"
    )
    print("No rows are shuffled; validation is the later time segment.")
    print("No trades are placed.")
    print_class_distribution("\nTrain class distribution", y_train)
    print_class_distribution("\nValidation class distribution", y_validation)

    model = train_model(x_train_scaled, y_train, x_validation_scaled, y_validation)
    _, _, _, probabilities = forward(model, x_validation_scaled)
    print_metrics(y_validation, probabilities)
    save_predictions(validation_frame, probabilities)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "symbol": SYMBOL,
                "feature_columns": feature_columns,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "model": {key: value.tolist() for key, value in model.items()},
                "class_names": CLASS_NAMES,
            },
            handle,
            indent=2,
        )

    print(f"\nSaved validation predictions to: {PREDICTIONS_PATH}")
    print(f"Saved ensemble meta-model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
