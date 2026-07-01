import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_tiny_price_training_rows import (
    add_asof_context,
    add_btc_lead_lag_context,
    add_calendar_session_features,
    add_depth_acceleration_features,
    add_pressure_change_features,
    add_regime_context_features,
    add_snapshot_freshness_features,
    add_spread_volatility_features,
    get_lookback_profile,
    normalize_input,
    past_index,
)
from train_tiny_price_model import predict_model
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
MODEL_PATH_ENV = os.getenv("PRICE_TINY_MODEL_PATH", "").strip()
MAX_STALENESS_SECONDS = int(os.getenv("PRICE_TINY_MAX_STALENESS_SECONDS", "120"))
PRICE_TINY_THRESHOLD = float(os.getenv("PRICE_TINY_THRESHOLD", "0.55"))
PRICE_TINY_REGIME_GATE = os.getenv("PRICE_TINY_REGIME_GATE", "no_gate").strip().lower()
PRICE_TINY_HORIZON_SECONDS_ENV = os.getenv("PRICE_TINY_HORIZON_SECONDS", "").strip()

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
SNAPSHOT_PATH = VENUE_DIR / f"{SYMBOL}_10s_flow.csv"
LIVE_PREDICTIONS_PATH = Path(
    os.getenv("PRICE_TINY_LIVE_PREDICTIONS_PATH", VENUE_DIR / f"{SYMBOL}_tiny_price_live_predictions.csv")
)
if not LIVE_PREDICTIONS_PATH.is_absolute():
    LIVE_PREDICTIONS_PATH = PROJECT_ROOT / LIVE_PREDICTIONS_PATH
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "tiny_price" / (PRIMARY_VENUE or "legacy")
SELECTED_ROOT = PROJECT_ROOT / "models" / "selected" / SYMBOL / "tiny_price" / (PRIMARY_VENUE or "legacy")
SELECTED_MODEL_PATH = SELECTED_ROOT / "selected_model.json"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_model_path():
    if MODEL_PATH_ENV:
        path = Path(MODEL_PATH_ENV)
        return path if path.is_absolute() else PROJECT_ROOT / path
    if SELECTED_MODEL_PATH.exists():
        selected = load_json(SELECTED_MODEL_PATH)
        selected_path = selected.get("champion_model_path") or selected.get("model_path")
        if selected_path:
            path = Path(selected_path)
            return path if path.is_absolute() else PROJECT_ROOT / path
    return None


def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def build_current_features(frame, artifact):
    frame = normalize_input(frame)
    if len(frame) == 0:
        return pd.DataFrame()
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
    mid = frame["mid_price"].to_numpy(dtype=np.float64)
    index = len(frame) - 1
    current_mid = float(mid[index])
    source = frame.iloc[index]
    profile = get_lookback_profile(str(artifact.get("lookback_profile", "short")))
    row = {
        "timestamp": int(timestamps[index]),
        "time": source.get("time", ""),
        "current_mid_price": current_mid,
    }
    rolling_mean_seconds = int(profile["rolling_mean_seconds"])
    rolling_start = past_index(timestamps, index, rolling_mean_seconds)
    rolling_mid = mid[rolling_start : index + 1]
    rolling_mean = float(np.nanmean(rolling_mid)) if len(rolling_mid) else current_mid
    row[f"feature_mid_vs_rolling_mean_{rolling_mean_seconds}s"] = current_mid / max(rolling_mean, 1e-12) - 1.0
    row["feature_spread_percent"] = float(source.get("spread_percent", 0.0))
    row["feature_bid_distance_to_mid"] = (current_mid - float(source.get("best_bid", current_mid))) / current_mid
    row["feature_ask_distance_to_mid"] = (float(source.get("best_ask", current_mid)) - current_mid) / current_mid
    for column in ["bid_depth_10bps", "ask_depth_10bps", "bid_depth_25bps", "ask_depth_25bps"]:
        row[f"feature_{column}_log1p"] = float(np.log1p(max(0.0, float(source.get(column, 0.0)))))
    row["feature_imbalance10"] = float(source.get("order_book_imbalance_10bps", 0.0))
    row["feature_imbalance25"] = float(source.get("order_book_imbalance_25bps", 0.0))
    for seconds in profile["return_seconds"]:
        p_index = past_index(timestamps, index, seconds)
        past_mid = mid[p_index]
        row[f"feature_mid_return_{seconds}s"] = float(np.log(current_mid / past_mid)) if past_mid > 0 else 0.0
    for seconds in profile["volatility_seconds"]:
        p_index = past_index(timestamps, index, seconds)
        window_mid = mid[p_index : index + 1]
        returns = np.diff(np.log(window_mid)) if len(window_mid) > 1 else np.asarray([0.0])
        row[f"feature_rolling_volatility_{seconds}s"] = float(np.nanstd(returns))
    for seconds in profile["range_seconds"]:
        p_index = past_index(timestamps, index, seconds)
        window_mid = mid[p_index : index + 1]
        row[f"feature_recent_high_low_range_{seconds}s"] = float((np.nanmax(window_mid) / max(np.nanmin(window_mid), 1e-12)) - 1.0)
    buy = float(source.get("market_buy_volume_10s", 0.0))
    sell = float(source.get("market_sell_volume_10s", 0.0))
    total = buy + sell
    row["feature_market_buy_volume_log1p"] = float(np.log1p(max(0.0, buy)))
    row["feature_market_sell_volume_log1p"] = float(np.log1p(max(0.0, sell)))
    row["feature_trade_count_log1p"] = float(np.log1p(max(0.0, float(source.get("trade_count_10s", 0.0)))))
    row["feature_buy_sell_imbalance"] = float((buy - sell) / max(total, 1e-12))
    row["feature_aggressive_flow_pressure"] = float(source.get("market_pressure_10s", row["feature_buy_sell_imbalance"]))
    feature_groups = artifact.get("feature_spec", {}).get("enabled_feature_groups", [])
    if isinstance(feature_groups, str):
        feature_groups = [value.strip() for value in feature_groups.split(",") if value.strip()]
    if "pressure_change_features" in feature_groups:
        add_pressure_change_features(row, frame, timestamps, index)
    if "spread_volatility_features" in feature_groups:
        add_spread_volatility_features(row, frame, timestamps, index)
    if "depth_acceleration_features" in feature_groups:
        add_depth_acceleration_features(row, frame, timestamps, index)
    if "snapshot_freshness_features" in feature_groups:
        add_snapshot_freshness_features(row, timestamps, index)
    if "regime_context_features" in feature_groups:
        add_regime_context_features(row, frame, timestamps, index)
    output = pd.DataFrame([row])
    if "calendar_session_features" in feature_groups:
        output = add_calendar_session_features(output)
    feature_set = str(artifact.get("feature_set_name", "tiny_price_v1"))
    if feature_set in {"tiny_price_v3", "tiny_price_v4"}:
        output = add_asof_context(
            output,
            VENUE_DIR / f"{SYMBOL}_1s_order_flow_predictions.csv",
            "flow1s",
            [
                "prob_sell_dominant_1s",
                "prob_neutral_1s",
                "prob_buy_dominant_1s",
                "pred_market_pressure_1s",
                "buy_burst_prob_1s",
                "sell_burst_prob_1s",
            ],
        )
    if feature_set == "tiny_price_v4":
        output = add_asof_context(
            output,
            VENUE_DIR / f"{SYMBOL}_10s_microstructure_predictions.csv",
            "micro10s",
            [
                "prob_upside_scare_event_10s",
                "prob_downside_scare_event_10s",
                "prob_spread_expansion_event_10s",
                "prob_bid_liquidity_drop_10s",
                "prob_ask_liquidity_drop_10s",
            ],
        )
    if "cross_venue_features" in feature_groups:
        output = add_asof_context(
            output,
            OUTPUT_DIR / f"{SYMBOL}_cross_venue_features.csv",
            "crossvenue",
            [
                "venue_count",
                "venue_mid_diff_bps",
                "venue_spread_diff_bps",
                "log_bid_depth_ratio_10bps",
                "log_ask_depth_ratio_10bps",
                "venue_imbalance_diff_10bps",
                "leading_venue_return_1s",
                "leading_venue_return_3s",
                "leading_venue_return_10s",
                "cross_venue_pressure_agreement",
                "cross_venue_pressure_divergence",
            ],
        )
    if "regime_context_features" in feature_groups:
        output = add_btc_lead_lag_context(output)
    for column in artifact["feature_columns"]:
        if column not in output.columns:
            output[column] = 0.0
    return output


def depth_from_log(row, column):
    value = float(row.get(column, 0.0))
    if not np.isfinite(value):
        return 0.0
    return max(0.0, float(np.expm1(value)))


def live_gate_status(feature_row, predicted_direction):
    if PRICE_TINY_REGIME_GATE in {"", "none", "no_gate"}:
        return True, "no_gate"
    row = feature_row.iloc[0]
    imbalance10 = float(row.get("feature_imbalance10", 0.0))
    imbalance25 = float(row.get("feature_imbalance25", 0.0))
    volatility60 = float(row.get("feature_rolling_volatility_60s", np.nan))
    range60 = float(row.get("feature_recent_high_low_range_60s", np.nan))
    bid10 = depth_from_log(row, "feature_bid_depth_10bps_log1p")
    ask10 = depth_from_log(row, "feature_ask_depth_10bps_log1p")
    bid25 = depth_from_log(row, "feature_bid_depth_25bps_log1p")
    ask25 = depth_from_log(row, "feature_ask_depth_25bps_log1p")
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


def append_prediction(row):
    existing = read_csv(LIVE_PREDICTIONS_PATH)
    output = pd.concat([existing, pd.DataFrame([row])], ignore_index=True) if len(existing) else pd.DataFrame([row])
    output["timestamp"] = pd.to_numeric(output["timestamp"], errors="coerce")
    output = output.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    atomic_write_csv(output, LIVE_PREDICTIONS_PATH)


def main():
    model_path = latest_model_path()
    if model_path is None or not model_path.exists():
        print(f"No pinned tiny price champion found for {SYMBOL}/{PRIMARY_VENUE or 'legacy'}")
        print("Set PRICE_TINY_MODEL_PATH or create:")
        print(f"  {SELECTED_MODEL_PATH}")
        print("Refusing to auto-load latest_candidate for live-style tiny-price prediction.")
        print("Paper-only. No trades/orders/private API.")
        return
    artifact = load_json(model_path)
    if str(artifact.get("symbol", "")).upper() != SYMBOL or str(artifact.get("primary_venue", "")).lower() != (PRIMARY_VENUE or "legacy"):
        print("Tiny price model metadata mismatch; refusing prediction.")
        print(f"Requested: {SYMBOL}/{PRIMARY_VENUE or 'legacy'}")
        print(f"Model: {artifact.get('symbol')}/{artifact.get('primary_venue')}")
        return
    snapshots = read_csv(SNAPSHOT_PATH)
    feature_row = build_current_features(snapshots, artifact)
    if len(feature_row) == 0:
        print(f"No valid snapshot rows available: {SNAPSHOT_PATH}")
        print("Paper-only. No trades/orders/private API.")
        return
    feature_columns = artifact["feature_columns"]
    x = feature_row[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
    std = np.asarray(artifact["feature_std"], dtype=np.float64)
    std = np.where(std < 1e-9, 1.0, std)
    x = (x - mean) / std
    selected = artifact["selected_model_name"]
    model = artifact["models"][selected]
    pred_delta, pred_log, pred_direction, confidence, probabilities = predict_model(
        selected,
        model,
        x,
        float(artifact.get("delta_target_mean", 0.0)),
        float(artifact.get("delta_target_std", 1.0)),
    )
    timestamp = int(feature_row["timestamp"].iloc[0])
    age_seconds = max(0.0, time.time() - timestamp / 1000.0)
    status = "fresh" if age_seconds <= MAX_STALENESS_SECONDS else "stale"
    confidence_passed = bool(float(confidence[0]) >= PRICE_TINY_THRESHOLD)
    gate_passed, gate_reason = live_gate_status(feature_row, int(pred_direction[0]))
    paper_signal_direction = int(pred_direction[0]) if confidence_passed and gate_passed else 0
    artifact_horizon = int(artifact.get("horizon_seconds", 1))
    requested_horizon = int(PRICE_TINY_HORIZON_SECONDS_ENV) if PRICE_TINY_HORIZON_SECONDS_ENV else artifact_horizon
    horizon_match = requested_horizon == artifact_horizon
    row = {
        "timestamp": timestamp,
        "time": feature_row["time"].iloc[0],
        "predicted_return_bps": float(pred_delta[0]),
        "predicted_next_mid_delta_bps": float(pred_delta[0]),
        "predicted_next_mid_log_return": float(pred_log[0]),
        "predicted_direction": int(pred_direction[0]),
        "confidence": float(confidence[0]),
        "confidence_type": artifact.get("confidence_type", "class_probability"),
        "model_id": artifact["model_id"],
        "feature_set_name": artifact.get("feature_set_name", ""),
        "horizon_seconds": artifact_horizon,
        "requested_horizon_seconds": requested_horizon,
        "horizon_matches_model": horizon_match,
        "lookback_profile": artifact.get("lookback_profile", "short"),
        "price_tiny_threshold": PRICE_TINY_THRESHOLD,
        "threshold_passed": confidence_passed,
        "price_tiny_regime_gate": PRICE_TINY_REGIME_GATE,
        "regime_gate_passed": gate_passed,
        "regime_gate_reason": gate_reason,
        "paper_signal_direction": paper_signal_direction,
        "freshness_status": status,
    }
    append_prediction(row)
    print("Tiny price paper prediction")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Snapshot path: {SNAPSHOT_PATH}")
    print(f"Model path: {model_path}")
    print(f"Selected model: {selected}")
    print(f"Feature set: {artifact.get('feature_set_name')}")
    print(f"Lookback profile: {artifact.get('lookback_profile', 'short')}")
    print(f"Model horizon seconds: {artifact_horizon}")
    print(f"Requested horizon seconds: {requested_horizon}")
    if not horizon_match:
        print("WARNING: PRICE_TINY_HORIZON_SECONDS does not match the loaded model artifact horizon.")
    print(f"Live threshold: {PRICE_TINY_THRESHOLD:.2f} passed={confidence_passed}")
    print(f"Regime gate: {PRICE_TINY_REGIME_GATE} passed={gate_passed}")
    print(f"Regime gate reason: {gate_reason}")
    print(f"Timestamp: {timestamp}")
    print(f"Freshness: {status} age={age_seconds:.1f}s")
    print(f"Predicted next mid delta: {pred_delta[0]:+.4f} bps")
    print(f"Predicted return bps: {pred_delta[0]:+.4f}")
    print(f"Predicted log return: {pred_log[0]:+.8f}")
    print(f"Predicted direction: {int(pred_direction[0])} (-1 down, 0 flat, 1 up)")
    confidence_type = artifact.get("confidence_type", "class_probability")
    print(f"Confidence/score: {confidence[0]:.2%} ({confidence_type})")
    if confidence_type == "absolute_predicted_return_magnitude":
        print("NOTE: regression score is derived from absolute predicted return magnitude; it is not a class probability.")
    print(f"Paper signal direction after threshold/gate: {paper_signal_direction}")
    print(f"Live prediction output: {LIVE_PREDICTIONS_PATH}")
    print("Paper-only. No trades/orders/private API.")


if __name__ == "__main__":
    main()
