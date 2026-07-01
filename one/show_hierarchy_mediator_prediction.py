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
MODEL_PATH_ENV = os.getenv("HIERARCHY_MEDIATOR_MODEL_PATH", "").strip()
PREDICTION_THRESHOLD = float(os.getenv("MEDIATOR_PREDICTION_THRESHOLD", "0.55"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
TRAINING_PATH = VENUE_DIR / f"{SYMBOL}_hierarchy_mediator_training_rows.csv"
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "hierarchy_mediator" / VENUE_TAG
HORIZONS = ["10s", "30s", "60s"]


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_model_path():
    if MODEL_PATH_ENV:
        return Path(MODEL_PATH_ENV)
    if not CANDIDATE_ROOT.exists():
        return None
    candidates = sorted(CANDIDATE_ROOT.glob("*/model.json"), key=lambda path: path.parent.name, reverse=True)
    return candidates[0] if candidates else None


def decode_direction(probability, threshold):
    if probability >= threshold:
        return "bullish"
    if probability <= 1.0 - threshold:
        return "bearish"
    return "neutral / abstain"


def top_contributors(row_vector, feature_columns, weights, limit=12):
    contribution = row_vector * np.asarray(weights, dtype=np.float64)
    order = np.argsort(-np.abs(contribution))[:limit]
    return [
        (feature_columns[index], float(contribution[index]))
        for index in order
        if abs(contribution[index]) > 0
    ]


def main():
    model_path = latest_model_path()
    if model_path is None or not model_path.exists():
        print(f"No hierarchy mediator model found for {SYMBOL}/{VENUE_TAG}")
        print(f"Expected candidates under: {CANDIDATE_ROOT}")
        print("Paper-only. No trades/orders/private API.")
        return

    artifact = load_json(model_path)
    model_symbol = str(artifact.get("symbol", "")).upper()
    model_venue = str(artifact.get("primary_venue", "")).lower()
    if model_symbol != SYMBOL or model_venue != VENUE_TAG:
        print("Hierarchy mediator model metadata mismatch. Refusing prediction.")
        print(f"Requested: {SYMBOL}/{VENUE_TAG}")
        print(f"Model: {model_symbol}/{model_venue}")
        print(f"Model path: {model_path}")
        print("Paper-only. No trades/orders/private API.")
        return

    rows = read_csv(TRAINING_PATH)
    if len(rows) == 0:
        print("No hierarchy mediator rows available.")
        print(f"Run: npm run hierarchy-mediator-build")
        print(f"Training row path: {TRAINING_PATH}")
        print("Paper-only. No trades/orders/private API.")
        return

    rows["timestamp"] = pd.to_numeric(rows.get("timestamp"), errors="coerce")
    rows = rows.dropna(subset=["timestamp"]).sort_values("timestamp")
    latest = rows.iloc[-1]
    feature_columns = artifact["feature_columns"]
    missing = [column for column in feature_columns if column not in rows.columns]
    if missing:
        print("Current mediator row schema is missing model features. Refusing prediction.")
        print(f"Missing columns count: {len(missing)}")
        print(f"First missing columns: {missing[:20]}")
        print(f"Model path: {model_path}")
        print("Paper-only. No trades/orders/private API.")
        return

    raw = pd.to_numeric(latest[feature_columns], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
    std = np.asarray(artifact["feature_std"], dtype=np.float64)
    std = np.where(std < 1e-9, 1.0, std)
    x = (raw - mean) / std
    threshold = float(artifact.get("prediction_threshold", PREDICTION_THRESHOLD))

    print("Hierarchy mediator paper prediction")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {VENUE_TAG}")
    print(f"Model path: {model_path}")
    print(f"Model id: {artifact.get('model_id', '')}")
    print(f"Trained until timestamp: {artifact.get('trained_until_timestamp', '')}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Latest row timestamp: {int(latest['timestamp'])}")
    print(f"Latest row time: {latest.get('time', '')}")
    print(f"Decision threshold: {threshold:.2f}")

    for horizon in HORIZONS:
        direction_model = artifact["direction_models"][horizon]
        return_model = artifact["return_models"][horizon]
        direction_weights = np.asarray(direction_model["weights"], dtype=np.float64)
        return_weights = np.asarray(return_model["weights"], dtype=np.float64)
        prob_up = float(sigmoid(np.array([x @ direction_weights + float(direction_model["bias"])]))[0])
        predicted_return = float(x @ return_weights + float(return_model["bias"]))
        direction = decode_direction(prob_up, threshold)
        print(f"\n{horizon} mediator view")
        print(f"- probability up: {prob_up:.2%}")
        print(f"- predicted return: {predicted_return:.4%}")
        print(f"- paper bias: {direction}")

    print("\nTop current contributors for 60s probability")
    sixty_weights = artifact["direction_models"]["60s"]["weights"]
    for feature, contribution in top_contributors(x, feature_columns, sixty_weights):
        sign = "supports bullish trust" if contribution > 0 else "supports bearish/abstain trust"
        print(f"- {feature}: {contribution:+.6f} ({sign})")

    print("\nAccountability summary from training")
    for group, values in sorted(artifact.get("accountability_60s", {}).items(), key=lambda item: item[1].get("abs_weight", 0), reverse=True):
        print(f"- {group}: abs_weight={values.get('abs_weight', 0):.6g}, signed_weight={values.get('signed_weight', 0):+.6g}")

    print("\nPaper-only mediator. No trades/orders/private API. No promotion.")


if __name__ == "__main__":
    main()
