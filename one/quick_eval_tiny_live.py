from pathlib import Path
import os
import pandas as pd
import numpy as np

base = Path(r"C:\Users\Mrkin\OneDrive\Documents\Misc")
symbol = os.getenv("SYMBOL", "SOLUSDT")
venue = os.getenv("PRIMARY_VENUE", "kraken")

pred_path = Path(os.getenv(
    "PRICE_TINY_LIVE_PREDICTIONS_PATH",
    base / "data" / "realtime" / venue / f"{symbol}_tiny_price_live_predictions.csv"
))
snap_path = base / "data" / "realtime" / venue / f"{symbol}_10s_flow.csv"
out_path = base / "data" / "realtime" / venue / f"{symbol}_tiny_price_live_prediction_evaluation_quick.csv"

def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

pred = pd.read_csv(pred_path)
snap = pd.read_csv(snap_path)

pred_ts_col = find_col(pred, ["timestamp_ms", "timestamp", "prediction_timestamp_ms", "snapshot_timestamp_ms"])
snap_ts_col = find_col(snap, ["timestamp_ms", "timestamp", "snapshot_timestamp_ms"])
signal_col = find_col(pred, ["paper_signal_direction", "signal_direction", "predicted_direction"])
horizon_col = find_col(pred, ["horizon_seconds", "model_horizon_seconds", "requested_horizon_seconds"])
conf_col = find_col(pred, ["confidence", "model_confidence"])

if pred_ts_col is None:
    pred_ts_col = pred.columns[0]
if snap_ts_col is None:
    snap_ts_col = snap.columns[0]
if signal_col is None:
    raise SystemExit(f"Could not find signal column. Prediction columns: {list(pred.columns)}")

mid_col = find_col(snap, ["mid", "mid_price", "best_mid"])
bid_col = find_col(snap, ["best_bid", "bid"])
ask_col = find_col(snap, ["best_ask", "ask"])

snap["_ts"] = pd.to_numeric(snap[snap_ts_col], errors="coerce")

if mid_col:
    snap["_mid"] = pd.to_numeric(snap[mid_col], errors="coerce")
elif bid_col and ask_col:
    snap["_mid"] = (
        pd.to_numeric(snap[bid_col], errors="coerce") +
        pd.to_numeric(snap[ask_col], errors="coerce")
    ) / 2.0
else:
    raise SystemExit(f"Could not infer mid price. Snapshot columns: {list(snap.columns)}")

pred["_pred_ts"] = pd.to_numeric(pred[pred_ts_col], errors="coerce")
pred["_signal"] = pd.to_numeric(pred[signal_col], errors="coerce").fillna(0).astype(int)

if horizon_col:
    pred["_horizon_seconds"] = pd.to_numeric(pred[horizon_col], errors="coerce").fillna(30)
else:
    pred["_horizon_seconds"] = 30

pred["_target_ts"] = pred["_pred_ts"] + pred["_horizon_seconds"] * 1000

snap2 = snap[["_ts", "_mid"]].dropna().sort_values("_ts")
p = pred.dropna(subset=["_pred_ts"]).sort_values("_pred_ts")

cur = pd.merge_asof(
    p,
    snap2.rename(columns={"_ts": "_current_snap_ts", "_mid": "current_mid"}),
    left_on="_pred_ts",
    right_on="_current_snap_ts",
    direction="backward",
    tolerance=15000,
)

full = pd.merge_asof(
    cur.sort_values("_target_ts"),
    snap2.rename(columns={"_ts": "_future_snap_ts", "_mid": "future_mid"}),
    left_on="_target_ts",
    right_on="_future_snap_ts",
    direction="forward",
    tolerance=15000,
)

full["realized_return_bps"] = (full["future_mid"] / full["current_mid"] - 1.0) * 10000.0
full["actual_direction"] = np.sign(full["realized_return_bps"]).fillna(0).astype(int)
full["directional_return_bps"] = full["_signal"] * full["realized_return_bps"]

evaluated = full.dropna(subset=["realized_return_bps"]).copy()
active = evaluated[(evaluated["_signal"] != 0) & (evaluated["actual_direction"] != 0)].copy()

full.to_csv(out_path, index=False)

print(f"Predictions: {len(pred)}")
print(f"Evaluated:   {len(evaluated)}")
print(f"Pending:     {len(pred) - len(evaluated)}")
print(f"Output:      {out_path}")

if len(active) == 0:
    print("No active evaluated predictions yet. Wait another 30-60 seconds and rerun.")
else:
    active["correct"] = active["_signal"] == active["actual_direction"]
    print("")
    print("Active evaluated signals")
    print(f"- count: {len(active)}")
    print(f"- sign accuracy: {active['correct'].mean() * 100:.2f}%")
    print(f"- directional win rate: {(active['directional_return_bps'] > 0).mean() * 100:.2f}%")
    print(f"- avg directional return: {active['directional_return_bps'].mean():.4f} bps")
    print(f"- median directional return: {active['directional_return_bps'].median():.4f} bps")

    if conf_col:
        active["_confidence"] = pd.to_numeric(active[conf_col], errors="coerce")
        print(f"- avg confidence: {active['_confidence'].mean() * 100:.2f}%")

    print("")
    print("Latest evaluated rows:")
    cols = [pred_ts_col, signal_col, "current_mid", "future_mid", "realized_return_bps", "actual_direction", "directional_return_bps"]
    if conf_col:
        cols.insert(2, conf_col)
    print(active[cols].tail(10).to_string(index=False))
