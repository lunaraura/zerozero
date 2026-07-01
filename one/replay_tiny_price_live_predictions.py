import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from train_tiny_price_model import predict_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
MODEL_PATH_ENV = os.getenv("PRICE_TINY_MODEL_PATH", "").strip()
PRICE_TINY_THRESHOLD = float(os.getenv("PRICE_TINY_THRESHOLD", "0.55"))
PRICE_TINY_REGIME_GATE = os.getenv("PRICE_TINY_REGIME_GATE", "no_gate").strip().lower()
PRICE_TINY_HORIZON_SECONDS_ENV = os.getenv("PRICE_TINY_HORIZON_SECONDS", "").strip()
ALLOW_HORIZON_MISMATCH = os.getenv("PRICE_TINY_ALLOW_HORIZON_MISMATCH", "false").strip().lower() in {"1", "true", "yes", "on"}
REPLAY_START_TIMESTAMP = os.getenv("PRICE_TINY_REPLAY_START_TIMESTAMP", "").strip()
REPLAY_END_TIMESTAMP = os.getenv("PRICE_TINY_REPLAY_END_TIMESTAMP", "").strip()
REPLAY_MAX_ROWS = os.getenv("PRICE_TINY_REPLAY_MAX_ROWS", "").strip()
REPLAY_SOURCE = os.getenv("PRICE_TINY_REPLAY_SOURCE", "training_rows").strip().lower()
REPLAY_OUTPUT_PATH_ENV = os.getenv("PRICE_TINY_REPLAY_OUTPUT_PATH", "").strip()

TRAINING_ROWS_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_training_rows.csv"
if REPLAY_OUTPUT_PATH_ENV:
    # When the user provides an output path, write to exactly that path.
    # A relative path remains relative to the process working directory.
    OUTPUT_PATH = Path(REPLAY_OUTPUT_PATH_ENV)
else:
    OUTPUT_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_replay_predictions.csv"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
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


def resolve_model_path():
    if not MODEL_PATH_ENV:
        raise RuntimeError("PRICE_TINY_MODEL_PATH is required for replay.")
    path = Path(MODEL_PATH_ENV)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_optional_int(value):
    if value == "":
        return None
    return int(float(value))


def depth_from_log(row, column):
    value = float(row.get(column, 0.0))
    if not np.isfinite(value):
        return 0.0
    return max(0.0, float(np.expm1(value)))


def gate_status(feature_row, predicted_direction):
    if PRICE_TINY_REGIME_GATE in {"", "none", "no_gate"}:
        return True, "no_gate"
    imbalance10 = float(feature_row.get("feature_imbalance10", 0.0))
    imbalance25 = float(feature_row.get("feature_imbalance25", 0.0))
    volatility60 = float(feature_row.get("feature_rolling_volatility_60s", np.nan))
    range60 = float(feature_row.get("feature_recent_high_low_range_60s", np.nan))
    bid10 = depth_from_log(feature_row, "feature_bid_depth_10bps_log1p")
    ask10 = depth_from_log(feature_row, "feature_ask_depth_10bps_log1p")
    bid25 = depth_from_log(feature_row, "feature_bid_depth_25bps_log1p")
    ask25 = depth_from_log(feature_row, "feature_ask_depth_25bps_log1p")
    imbalance10_high = float(os.getenv("PRICE_TINY_GATE_IMBALANCE10_HIGH", "0.20"))
    imbalance25_high = float(os.getenv("PRICE_TINY_GATE_IMBALANCE25_HIGH", "0.20"))
    low_volatility60 = float(os.getenv("PRICE_TINY_GATE_LOW_VOLATILITY60", "0.00008"))
    low_range60 = float(os.getenv("PRICE_TINY_GATE_LOW_RANGE60", "0.00080"))
    depth_ratio10_high = float(os.getenv("PRICE_TINY_GATE_DEPTH_RATIO10_HIGH", "1.25"))
    depth_ratio25_high = float(os.getenv("PRICE_TINY_GATE_DEPTH_RATIO25_HIGH", "1.25"))
    bid_depth_ratio10 = bid10 / max(ask10, 1e-9)
    bid_depth_ratio25 = bid25 / max(ask25, 1e-9)
    if PRICE_TINY_REGIME_GATE == "long_side_only":
        return predicted_direction > 0, "passes only long-side signals"
    if PRICE_TINY_REGIME_GATE == "short_side_only":
        return predicted_direction < 0, "passes only short-side signals"
    if PRICE_TINY_REGIME_GATE == "suppress_longs_when_bid_imbalance_high":
        blocked = predicted_direction > 0 and (imbalance10 >= imbalance10_high or imbalance25 >= imbalance25_high)
        return not blocked, f"imbalance10={imbalance10:.4f}, imbalance25={imbalance25:.4f}"
    if PRICE_TINY_REGIME_GATE == "suppress_longs_when_low_volatility_and_bid_imbalance_high":
        high_imbalance = imbalance10 >= imbalance10_high or imbalance25 >= imbalance25_high
        low_vol = np.isfinite(volatility60) and volatility60 <= low_volatility60
        low_range = np.isfinite(range60) and range60 <= low_range60
        blocked = predicted_direction > 0 and high_imbalance and low_vol and low_range
        return not blocked, f"imbalance10={imbalance10:.4f}, imbalance25={imbalance25:.4f}, vol60={volatility60:.8f}, range60={range60:.8f}"
    if PRICE_TINY_REGIME_GATE == "suppress_longs_when_bid_depth_dominates_ask_depth":
        blocked = predicted_direction > 0 and (bid_depth_ratio10 >= depth_ratio10_high or bid_depth_ratio25 >= depth_ratio25_high)
        return not blocked, f"bid_depth_ratio10={bid_depth_ratio10:.4f}, bid_depth_ratio25={bid_depth_ratio25:.4f}"
    return True, f"unknown gate '{PRICE_TINY_REGIME_GATE}' treated as pass"


def load_replay_source():
    if REPLAY_SOURCE != "training_rows":
        raise RuntimeError(f"Unsupported PRICE_TINY_REPLAY_SOURCE={REPLAY_SOURCE}. Only training_rows is currently supported.")
    frame = read_csv(TRAINING_ROWS_PATH)
    if len(frame) == 0:
        raise FileNotFoundError(f"Missing replay source: {TRAINING_ROWS_PATH}")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    start = parse_optional_int(REPLAY_START_TIMESTAMP)
    end = parse_optional_int(REPLAY_END_TIMESTAMP)
    if start is not None:
        frame = frame[frame["timestamp"] >= start]
    if end is not None:
        frame = frame[frame["timestamp"] <= end]
    max_rows = parse_optional_int(REPLAY_MAX_ROWS)
    if max_rows is not None and max_rows > 0:
        frame = frame.head(max_rows)
    return frame.reset_index(drop=True)


def actual_direction_from_delta(delta_bps):
    if not np.isfinite(delta_bps):
        return np.nan
    if delta_bps > 0:
        return 1
    if delta_bps < 0:
        return -1
    return 0


def summarize(output):
    rows = len(output)
    active = output["paper_signal_direction"] != 0 if rows else pd.Series(dtype=bool)
    longs = output["paper_signal_direction"] > 0 if rows else pd.Series(dtype=bool)
    shorts = output["paper_signal_direction"] < 0 if rows else pd.Series(dtype=bool)
    threshold_passed = output["threshold_passed"].astype(bool) if rows else pd.Series(dtype=bool)
    gate_passed = output["regime_gate_passed"].astype(bool) if rows else pd.Series(dtype=bool)
    predicted_direction = pd.to_numeric(output["predicted_direction"], errors="coerce").fillna(0) if rows else pd.Series(dtype=float)
    actual_delta = pd.to_numeric(output["actual_next_mid_delta_bps"], errors="coerce") if rows else pd.Series(dtype=float)
    gate_blocked = threshold_passed & ~gate_passed if rows else pd.Series(dtype=bool)
    gate_blocked_longs = gate_blocked & (predicted_direction > 0) if rows else pd.Series(dtype=bool)
    gate_blocked_shorts = gate_blocked & (predicted_direction < 0) if rows else pd.Series(dtype=bool)
    active_returns = pd.to_numeric(output.loc[active, "paper_signal_return_bps"], errors="coerce") if rows else pd.Series(dtype=float)
    long_returns = pd.to_numeric(output.loc[longs, "paper_signal_return_bps"], errors="coerce") if rows else pd.Series(dtype=float)
    short_returns = pd.to_numeric(output.loc[shorts, "paper_signal_return_bps"], errors="coerce") if rows else pd.Series(dtype=float)
    gate_blocked_equivalent_returns = actual_delta[gate_blocked] * predicted_direction[gate_blocked] if rows else pd.Series(dtype=float)
    gate_blocked_long_returns = actual_delta[gate_blocked_longs] if rows else pd.Series(dtype=float)
    gate_blocked_short_returns = -actual_delta[gate_blocked_shorts] if rows else pd.Series(dtype=float)
    no_gate_equivalent_active = threshold_passed & (predicted_direction != 0) if rows else pd.Series(dtype=bool)
    no_gate_equivalent_returns = actual_delta[no_gate_equivalent_active] * predicted_direction[no_gate_equivalent_active] if rows else pd.Series(dtype=float)
    no_gate_equivalent_longs = no_gate_equivalent_active & (predicted_direction > 0) if rows else pd.Series(dtype=bool)
    no_gate_equivalent_shorts = no_gate_equivalent_active & (predicted_direction < 0) if rows else pd.Series(dtype=bool)
    print("Replay summary")
    print(f"- rows: {rows}")
    print(f"- active rows: {int(active.sum()) if rows else 0}")
    print(f"- coverage: {(float(active.mean()) if rows else 0.0):.2%}")
    print(f"- avg_return_bps on active paper signals: {float(active_returns.mean()) if len(active_returns) else np.nan:.4f}")
    print(f"- win rate on active paper signals: {float((active_returns > 0).mean()) if len(active_returns) else np.nan:.2%}")
    print(f"- long active rows: {int(longs.sum()) if rows else 0}")
    print(f"- long avg return: {float(long_returns.mean()) if len(long_returns) else np.nan:.4f} bps")
    print(f"- short active rows: {int(shorts.sum()) if rows else 0}")
    print(f"- short avg return: {float(short_returns.mean()) if len(short_returns) else np.nan:.4f} bps")
    print(f"- threshold_passed rows: {int(threshold_passed.sum()) if rows else 0}")
    print(f"- gate_blocked_rows: {int(gate_blocked.sum()) if rows else 0}")
    print(f"- gate_blocked_avg_actual_return_bps: {float(gate_blocked_equivalent_returns.mean()) if len(gate_blocked_equivalent_returns) else np.nan:.4f}")
    print(f"- gate_blocked_long_rows: {int(gate_blocked_longs.sum()) if rows else 0}")
    print(f"- gate_blocked_long_avg_actual_return_bps: {float(gate_blocked_long_returns.mean()) if len(gate_blocked_long_returns) else np.nan:.4f}")
    print(f"- gate_blocked_short_rows: {int(gate_blocked_shorts.sum()) if rows else 0}")
    print(f"- gate_blocked_short_avg_actual_return_bps: {float(gate_blocked_short_returns.mean()) if len(gate_blocked_short_returns) else np.nan:.4f}")
    print(f"- no_gate_equivalent_active_rows: {int(no_gate_equivalent_active.sum()) if rows else 0}")
    print(f"- no_gate_equivalent_active_return_bps: {float(no_gate_equivalent_returns.mean()) if len(no_gate_equivalent_returns) else np.nan:.4f}")
    print(f"- no_gate_equivalent_long_rows: {int(no_gate_equivalent_longs.sum()) if rows else 0}")
    print(f"- no_gate_equivalent_short_rows: {int(no_gate_equivalent_shorts.sum()) if rows else 0}")
    print(f"- abstain rows: {int((~active).sum()) if rows else 0}")


def main():
    model_path = resolve_model_path()
    artifact = load_json(model_path)
    if str(artifact.get("symbol", "")).upper() != SYMBOL:
        raise RuntimeError(f"Model symbol mismatch: requested {SYMBOL}, artifact has {artifact.get('symbol')}")
    if str(artifact.get("primary_venue", "")).lower() != (PRIMARY_VENUE or "legacy"):
        raise RuntimeError(f"Model venue mismatch: requested {PRIMARY_VENUE or 'legacy'}, artifact has {artifact.get('primary_venue')}")
    model_horizon = int(artifact.get("horizon_seconds", 1))
    requested_horizon = int(PRICE_TINY_HORIZON_SECONDS_ENV) if PRICE_TINY_HORIZON_SECONDS_ENV else model_horizon
    horizon_matches = requested_horizon == model_horizon
    if not horizon_matches and not ALLOW_HORIZON_MISMATCH:
        raise RuntimeError(
            f"Requested horizon {requested_horizon}s does not match model horizon {model_horizon}s. "
            "Set PRICE_TINY_ALLOW_HORIZON_MISMATCH=true to replay anyway."
        )
    frame = load_replay_source()
    feature_columns = artifact["feature_columns"]
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Replay source missing model feature columns: {missing[:20]}")
    x = frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
    std = np.asarray(artifact["feature_std"], dtype=np.float64)
    std = np.where(std < 1e-9, 1.0, std)
    x = (x - mean) / std
    selected = artifact["selected_model_name"]
    pred_delta, pred_log, pred_direction, confidence, probabilities = predict_model(
        selected,
        artifact["models"][selected],
        x,
        float(artifact.get("delta_target_mean", 0.0)),
        float(artifact.get("delta_target_std", 1.0)),
    )
    target_delta_column = f"target_next_mid_delta_bps_{model_horizon}s"
    target_direction_column = f"target_next_mid_direction_{model_horizon}s"
    rows = []
    for index, source in frame.iterrows():
        predicted = int(pred_direction[index])
        conf = float(confidence[index])
        threshold_passed = conf >= PRICE_TINY_THRESHOLD
        gate_passed, gate_reason = gate_status(source, predicted)
        paper_signal = predicted if threshold_passed and gate_passed else 0
        actual_delta = float(source.get(target_delta_column, np.nan))
        actual_direction = int(source.get(target_direction_column, actual_direction_from_delta(actual_delta))) if np.isfinite(actual_delta) else np.nan
        paper_return = actual_delta * paper_signal if paper_signal != 0 and np.isfinite(actual_delta) else np.nan
        no_gate_equivalent_signal = predicted if threshold_passed else 0
        no_gate_equivalent_return = actual_delta * no_gate_equivalent_signal if no_gate_equivalent_signal != 0 and np.isfinite(actual_delta) else np.nan
        gate_blocked = bool(threshold_passed and not gate_passed and predicted != 0)
        prediction_correct = bool(predicted == actual_direction) if np.isfinite(actual_direction) else False
        rows.append(
            {
                "timestamp": int(source["timestamp"]),
                "time": source.get("time", ""),
                "symbol": SYMBOL,
                "primary_venue": PRIMARY_VENUE or "legacy",
                "model_path": str(model_path),
                "model_id": artifact.get("model_id", ""),
                "feature_set_name": artifact.get("feature_set_name", ""),
                "lookback_profile": artifact.get("lookback_profile", "short"),
                "model_horizon_seconds": model_horizon,
                "requested_horizon_seconds": requested_horizon,
                "horizon_matches_model": horizon_matches,
                "confidence": conf,
                "confidence_type": artifact.get("confidence_type", "class_probability"),
                "predicted_direction": predicted,
                "predicted_return_bps": float(pred_delta[index]),
                "predicted_next_mid_delta_bps": float(pred_delta[index]),
                "price_tiny_threshold": PRICE_TINY_THRESHOLD,
                "threshold_passed": threshold_passed,
                "price_tiny_regime_gate": PRICE_TINY_REGIME_GATE,
                "regime_gate_passed": gate_passed,
                "regime_gate_reason": gate_reason,
                "paper_signal_direction": paper_signal,
                "gate_blocked": gate_blocked,
                "no_gate_equivalent_signal_direction": no_gate_equivalent_signal,
                "no_gate_equivalent_return_bps": no_gate_equivalent_return,
                "replay_mode": True,
                "actual_next_mid_delta_bps": actual_delta,
                "actual_direction": actual_direction,
                "paper_signal_return_bps": paper_return,
                "prediction_correct": prediction_correct,
            }
        )
    output = pd.DataFrame(rows)
    atomic_write_csv(output, OUTPUT_PATH)
    print("Tiny price historical live replay")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Model path: {model_path}")
    print(f"Model horizon seconds: {model_horizon}")
    print(f"Requested horizon seconds: {requested_horizon}")
    print(f"Threshold: {PRICE_TINY_THRESHOLD:.2f}")
    print(f"Regime gate: {PRICE_TINY_REGIME_GATE}")
    print(f"Replay source: {REPLAY_SOURCE}")
    print(f"Rows replayed: {len(output)}")
    print(f"Output: {OUTPUT_PATH.resolve()}")
    summarize(output)
    print("Important: this is a replay/backtest harness, not proof of live performance if the interval was used to train or select the model.")
    print("No promotion. Paper-only. No trades/orders/private API.")


if __name__ == "__main__":
    main()
