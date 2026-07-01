import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
HORIZON_SECONDS = int(os.getenv("PRICE_TINY_HORIZON_SECONDS", "1"))
MAX_FUTURE_GAP_MS = int(os.getenv("PRICE_TINY_MAX_FUTURE_GAP_MS", str(max(1500, HORIZON_SECONDS * 1500))))
CALIBRATION_INVERSION_MARGIN = float(os.getenv("PRICE_TINY_CALIBRATION_INVERSION_MARGIN", "0.02"))
CALIBRATION_MIN_BUCKET_ROWS = int(os.getenv("PRICE_TINY_CALIBRATION_MIN_BUCKET_ROWS", "30"))
MIN_THRESHOLD_ROWS_INTERESTING = int(os.getenv("PRICE_TINY_MIN_THRESHOLD_ROWS_INTERESTING", "100"))
MIN_THRESHOLD_ROWS_STABLE = int(os.getenv("PRICE_TINY_MIN_THRESHOLD_ROWS_STABLE", "300"))
CONFIDENCE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
EXPECTED_TARGET_SPEC = os.getenv("PRICE_TINY_TARGET_SPEC", "").strip().lower()
REGRESSION_OUTPUT_SEMANTICS = {"return_bps", "next_mid_delta_bps", "next_mid_log_return"}

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
SNAPSHOT_PATH = VENUE_DIR / f"{SYMBOL}_10s_flow.csv"
PREDICTIONS_PATH = Path(
    os.getenv("PRICE_TINY_EVALUATION_PREDICTIONS_PATH", VENUE_DIR / f"{SYMBOL}_tiny_price_forward_test_predictions.csv")
)
if not PREDICTIONS_PATH.is_absolute():
    PREDICTIONS_PATH = PROJECT_ROOT / PREDICTIONS_PATH
EVALUATION_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_evaluation.csv"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def normalize_snapshots(frame):
    if len(frame) == 0:
        return frame
    frame = frame.copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["mid_price"] = pd.to_numeric(frame["mid_price"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "mid_price"])
    frame = frame[frame["mid_price"] > 0].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame.reset_index(drop=True)


def attach_actuals(predictions, snapshots):
    predictions = predictions.copy()
    predictions["timestamp"] = pd.to_numeric(predictions["timestamp"], errors="coerce")
    predictions = predictions.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    snapshots = normalize_snapshots(snapshots)
    if len(predictions) == 0 or len(snapshots) == 0:
        return pd.DataFrame()
    ts = snapshots["timestamp"].to_numpy(dtype=np.int64)
    mid = snapshots["mid_price"].to_numpy(dtype=np.float64)
    current_index = np.searchsorted(ts, predictions["timestamp"].to_numpy(dtype=np.int64), side="left")
    rows = []
    for row_index, pred in predictions.iterrows():
        index = int(current_index[row_index])
        if index >= len(ts) or int(ts[index]) != int(pred["timestamp"]):
            index = int(np.searchsorted(ts, int(pred["timestamp"]), side="right") - 1)
        if index < 0 or index >= len(ts):
            continue
        target = int(pred["timestamp"]) + int(pred.get("horizon_seconds", HORIZON_SECONDS)) * 1000
        f_index = int(np.searchsorted(ts, target, side="left"))
        if f_index >= len(ts) or ts[f_index] - target > MAX_FUTURE_GAP_MS:
            continue
        current_mid = mid[index]
        future_mid = mid[f_index]
        actual_delta = (future_mid / current_mid - 1.0) * 10000.0
        actual_log = np.log(future_mid / current_mid)
        future_path = mid[index + 1 : f_index + 1]
        future_path = future_path[np.isfinite(future_path) & (future_path > 0)]
        if len(future_path):
            actual_future_range_bps = float((np.nanmax(future_path) / current_mid - np.nanmin(future_path) / current_mid) * 10000.0)
        else:
            actual_future_range_bps = float(abs(actual_delta))
        price_direction = 1 if actual_delta > 0 else (-1 if actual_delta < 0 else 0)
        out = pred.to_dict()
        out["actual_next_mid_delta_bps"] = float(actual_delta)
        out["actual_next_mid_log_return"] = float(actual_log)
        out["actual_future_range_bps"] = actual_future_range_bps
        output_semantics = str(pred.get("output_semantics", "")).strip().lower()
        if output_semantics == "instability" and pd.notna(pred.get("actual_direction", np.nan)):
            out["actual_instability"] = int(float(pred.get("actual_direction", 0)))
            out["actual_direction_price"] = price_direction
            out["actual_direction"] = int(float(pred.get("actual_direction", 0)))
        else:
            out["actual_direction"] = price_direction
        rows.append(out)
    return pd.DataFrame(rows)


def expected_selected_target_column(target_spec):
    target_spec = str(target_spec or "").strip().lower()
    horizon = HORIZON_SECONDS
    for part in reversed(target_spec.split("_")):
        if part.endswith("s") and part[:-1].isdigit():
            horizon = int(part[:-1])
            break
    if target_spec.startswith("return_bps"):
        return f"target_return_bps_{horizon}s", "return_bps"
    if target_spec.startswith("next_mid_delta_bps"):
        return f"target_next_mid_delta_bps_{horizon}s", "next_mid_delta_bps"
    if target_spec.startswith("next_mid_log_return"):
        return f"target_next_mid_log_return_{horizon}s", "next_mid_log_return"
    if target_spec.startswith("instability"):
        return f"target_instability_{horizon}s", "instability"
    if target_spec.startswith("move_before_adverse"):
        return f"target_move_before_adverse_{horizon}s", "move_before_adverse"
    if target_spec.startswith("first_touch"):
        return f"target_first_touch_direction_{horizon}s", "first_touch"
    return f"target_direction_{horizon}s", "direction"


def prediction_metadata_matches(predictions):
    if len(predictions) == 0 or not EXPECTED_TARGET_SPEC:
        return True, ""
    expected_column, expected_semantics = expected_selected_target_column(EXPECTED_TARGET_SPEC)
    selected_columns = ""
    if "selected_target_columns" in predictions.columns and len(predictions["selected_target_columns"].dropna()):
        selected_columns = str(predictions["selected_target_columns"].dropna().iloc[0])
    output_semantics = ""
    if "output_semantics" in predictions.columns and len(predictions["output_semantics"].dropna()):
        output_semantics = str(predictions["output_semantics"].dropna().iloc[0]).strip().lower()
    if expected_column and selected_columns and expected_column not in selected_columns:
        return False, f"selected_target_columns mismatch: expected {expected_column}, found {selected_columns}"
    if output_semantics and expected_semantics and output_semantics != expected_semantics:
        return False, f"output_semantics mismatch: expected {expected_semantics}, found {output_semantics}"
    return True, ""


def regression_correlation(predicted_bps, actual_bps):
    predicted_bps = np.asarray(predicted_bps, dtype=np.float64)
    actual_bps = np.asarray(actual_bps, dtype=np.float64)
    mask = np.isfinite(predicted_bps) & np.isfinite(actual_bps)
    if mask.sum() < 2:
        return np.nan
    if np.nanstd(predicted_bps[mask]) < 1e-12 or np.nanstd(actual_bps[mask]) < 1e-12:
        return np.nan
    return float(np.corrcoef(predicted_bps[mask], actual_bps[mask])[0, 1])


def print_regression_threshold_report(evaluated, pred_return, actual_delta):
    pred_direction = np.where(pred_return > 0, 1, np.where(pred_return < 0, -1, 0))
    print("Absolute predicted-return threshold report")
    for threshold in [1.0, 2.0, 5.0, 8.0]:
        mask = np.abs(pred_return) >= threshold
        if not mask.any():
            print(f"- abs_pred_return>={threshold:.1f}bps: rows=0")
            continue
        up_mask = mask & (pred_direction > 0)
        down_mask = mask & (pred_direction < 0)
        strategy_return = actual_delta[mask] * np.sign(pred_direction[mask])
        sign_accuracy = float((np.sign(pred_direction[mask]) == np.sign(actual_delta[mask])).mean())
        print(
            f"- abs_pred_return>={threshold:.1f}bps: rows={int(mask.sum())} "
            f"sign_acc={sign_accuracy:.2%} "
            f"strategy_avg_return={float(strategy_return.mean()):.4f}bps "
            f"avg_realized={float(actual_delta[mask].mean()):.4f}bps "
            f"up_rows={int(up_mask.sum())} "
            f"up_return={float(actual_delta[up_mask].mean()) if up_mask.any() else np.nan:.4f}bps "
            f"down_rows={int(down_mask.sum())} "
            f"down_return={float(actual_delta[down_mask].mean()) if down_mask.any() else np.nan:.4f}bps"
        )


def calibration_buckets(frame):
    rows = []
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0).to_numpy()
    pred = pd.to_numeric(frame["predicted_direction"], errors="coerce").fillna(0).to_numpy()
    actual = pd.to_numeric(frame["actual_direction"], errors="coerce").fillna(0).to_numpy()
    for low, high in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]:
        mask = (confidence >= low) & (confidence < high)
        if not mask.any():
            continue
        rows.append(
            {
                "bucket": f"{low:.1f}-{min(high, 1.0):.1f}",
                "rows": int(mask.sum()),
                "accuracy": float((pred[mask] == actual[mask]).mean()),
                "avg_confidence": float(confidence[mask].mean()),
                "avg_actual_delta_bps": float(frame.loc[mask, "actual_next_mid_delta_bps"].mean()),
            }
        )
    return rows


def weighted_bucket_accuracy(buckets, low_threshold=None, high_threshold=None):
    total = 0
    weighted = 0.0
    for row in buckets:
        if row.get("rows", 0) < CALIBRATION_MIN_BUCKET_ROWS or "accuracy" not in row:
            continue
        low = float(row["bucket"].split("-", 1)[0])
        high = float(row["bucket"].split("-", 1)[1])
        if low_threshold is not None and high > low_threshold:
            continue
        if high_threshold is not None and low < high_threshold:
            continue
        total += int(row["rows"])
        weighted += int(row["rows"]) * float(row["accuracy"])
    if total == 0:
        return np.nan, 0
    return weighted / total, total


def calibration_is_inverted(buckets):
    low_accuracy, low_rows = weighted_bucket_accuracy(buckets, low_threshold=0.60)
    high_accuracy, high_rows = weighted_bucket_accuracy(buckets, high_threshold=0.70)
    if low_rows < CALIBRATION_MIN_BUCKET_ROWS or high_rows < CALIBRATION_MIN_BUCKET_ROWS:
        return False
    return bool(high_accuracy + CALIBRATION_INVERSION_MARGIN < low_accuracy)


def confidence_threshold_directional_report(frame):
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0).to_numpy()
    pred_dir = pd.to_numeric(frame["predicted_direction"], errors="coerce").fillna(0).astype(int).to_numpy()
    actual_dir = pd.to_numeric(frame["actual_direction"], errors="coerce").fillna(0).astype(int).to_numpy()
    actual_delta = pd.to_numeric(frame["actual_next_mid_delta_bps"], errors="coerce").fillna(0.0).to_numpy()
    majority_direction = pd.Series(actual_dir).mode().iloc[0] if len(actual_dir) else 0
    majority_accuracy = float((np.full(len(actual_dir), majority_direction) == actual_dir).mean()) if len(actual_dir) else np.nan
    rows = []
    for threshold in CONFIDENCE_THRESHOLDS:
        mask = (confidence >= threshold) & (pred_dir != 0)
        if not mask.any():
            rows.append(
                {
                    "threshold": threshold,
                    "rows_kept": 0,
                    "directional_accuracy": np.nan,
                    "avg_realized_return_bps": np.nan,
                    "predicted_up_return_bps": np.nan,
                    "predicted_down_return_bps": np.nan,
                    "lift_vs_majority": np.nan,
                    "threshold_interesting": False,
                    "threshold_stable_candidate": False,
                }
            )
            continue
        up_mask = mask & (pred_dir > 0)
        down_mask = mask & (pred_dir < 0)
        accuracy = float((pred_dir[mask] == actual_dir[mask]).mean())
        rows_kept = int(mask.sum())
        avg_return = float((actual_delta[mask] * np.sign(pred_dir[mask])).mean())
        lift = float(accuracy - majority_accuracy)
        rows.append(
            {
                "threshold": threshold,
                "rows_kept": rows_kept,
                "directional_accuracy": accuracy,
                "avg_realized_return_bps": avg_return,
                "predicted_up_return_bps": float(actual_delta[up_mask].mean()) if up_mask.any() else np.nan,
                "predicted_down_return_bps": float(actual_delta[down_mask].mean()) if down_mask.any() else np.nan,
                "lift_vs_majority": lift,
                "threshold_interesting": bool(rows_kept >= MIN_THRESHOLD_ROWS_INTERESTING and avg_return > 0),
                "threshold_stable_candidate": bool(rows_kept >= MIN_THRESHOLD_ROWS_STABLE and avg_return > 0 and lift > 0),
            }
        )
    return rows


def best_threshold_summary(rows, min_rows):
    eligible = [
        row
        for row in rows
        if int(row.get("rows_kept", 0)) >= min_rows and np.isfinite(row.get("avg_realized_return_bps", np.nan))
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row.get("avg_realized_return_bps", float("-inf"))),
            float(row.get("lift_vs_majority", float("-inf"))),
            int(row.get("rows_kept", 0)),
        ),
    )


def print_best_threshold(label, row):
    if not row:
        print(f"- {label}: none")
        return
    print(
        f"- {label}: threshold>={row['threshold']:.2f} rows={row['rows_kept']} "
        f"avg_return={row['avg_realized_return_bps']:.4f}bps "
        f"dir_acc={row['directional_accuracy']:.2%} "
        f"lift_vs_majority={row['lift_vs_majority']:.2%} "
        f"interesting={row['threshold_interesting']} "
        f"stable={row['threshold_stable_candidate']}"
    )


def main():
    predictions = read_csv(PREDICTIONS_PATH)
    metadata_ok, metadata_reason = prediction_metadata_matches(predictions)
    if not metadata_ok:
        print("Tiny price prediction evaluation blocked")
        print(f"Predictions: {PREDICTIONS_PATH}")
        print(f"Expected target spec: {EXPECTED_TARGET_SPEC}")
        print(f"Blocking reason: {metadata_reason}")
        print("Refusing to evaluate potentially stale/mismatched predictions.")
        print("Paper-only. No trades/orders/private API.")
        return
    snapshots = read_csv(SNAPSHOT_PATH)
    evaluated = attach_actuals(predictions, snapshots)
    atomic_write_csv(evaluated, EVALUATION_PATH)
    print("Tiny price prediction evaluation")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Snapshots: {SNAPSHOT_PATH}")
    print(f"Evaluation output: {EVALUATION_PATH}")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Evaluated rows: {len(evaluated)}")
    if len(evaluated):
        pred_delta = pd.to_numeric(evaluated["predicted_next_mid_delta_bps"], errors="coerce").fillna(0.0)
        pred_return = (
            pd.to_numeric(evaluated["predicted_return_bps"], errors="coerce").fillna(pred_delta)
            if "predicted_return_bps" in evaluated.columns
            else pred_delta
        )
        actual_delta = pd.to_numeric(evaluated["actual_next_mid_delta_bps"], errors="coerce").fillna(0.0)
        pred_dir = pd.to_numeric(evaluated["predicted_direction"], errors="coerce").fillna(0).astype(int)
        actual_dir = pd.to_numeric(evaluated["actual_direction"], errors="coerce").fillna(0).astype(int)
        output_semantics = str(evaluated["output_semantics"].dropna().iloc[0]).strip().lower() if "output_semantics" in evaluated.columns and len(evaluated["output_semantics"].dropna()) else ""
        if output_semantics in REGRESSION_OUTPUT_SEMANTICS:
            sign_accuracy = float((np.sign(pred_return) == np.sign(actual_delta)).mean()) if len(evaluated) else np.nan
            corr = regression_correlation(pred_return, actual_delta)
            print("Regression return evaluation")
            print(f"Output semantics: {output_semantics}")
            print("Predicted return is continuous bps; confidence is magnitude score, not class probability.")
            print(f"MAE bps: {float((pred_return - actual_delta).abs().mean()):.4f}")
            print(f"RMSE bps: {float(np.sqrt(((pred_return - actual_delta) ** 2).mean())):.4f}")
            print(f"Sign accuracy: {sign_accuracy:.2%}")
            print(f"Correlation predicted vs realized return: {corr:.4f}")
            print(f"Average realized bps when predicted up: {float(actual_delta[pred_return > 0].mean()) if (pred_return > 0).any() else np.nan:.4f}")
            print(f"Average realized bps when predicted down: {float(actual_delta[pred_return < 0].mean()) if (pred_return < 0).any() else np.nan:.4f}")
            if (np.isfinite(corr) and corr < 0) or (np.isfinite(sign_accuracy) and sign_accuracy < 0.5):
                print("WARNING: Regression sign may be inverted.")
            print_regression_threshold_report(evaluated, pred_return.to_numpy(dtype=np.float64), actual_delta.to_numpy(dtype=np.float64))
            print("Paper-only. No trades/orders/private API.")
            return
        if output_semantics == "instability":
            actual_instability = (
                pd.to_numeric(evaluated["actual_instability"], errors="coerce").fillna(0).astype(int)
                if "actual_instability" in evaluated.columns
                else actual_dir
            )
            predicted_instability = (pred_dir > 0).astype(int)
            actual_bool = actual_instability > 0
            pred_bool = predicted_instability > 0
            tp = int((pred_bool & actual_bool).sum())
            fp = int((pred_bool & ~actual_bool).sum())
            fn = int((~pred_bool & actual_bool).sum())
            tn = int((~pred_bool & ~actual_bool).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            stable = ~actual_bool
            unstable = actual_bool
            actual_range = (
                pd.to_numeric(evaluated["actual_future_range_bps"], errors="coerce").fillna(np.nan)
                if "actual_future_range_bps" in evaluated.columns
                else actual_delta.abs()
            )
            print("Instability risk/gating evaluation")
            print(f"Class distribution: {actual_instability.value_counts().sort_index().to_dict()}")
            print(f"Percent unstable: {float(actual_bool.mean()):.2%}")
            print(f"Accuracy: {float((pred_bool == actual_bool).mean()):.2%}")
            print(f"Precision: {precision:.2%}")
            print(f"Recall: {recall:.2%}")
            print(f"F1: {f1:.2%}")
            print(f"Confusion: tp={tp} fp={fp} tn={tn} fn={fn}")
            print(f"Average realized return bps stable: {float(actual_delta[stable].mean()) if stable.any() else np.nan:.4f}")
            print(f"Average realized return bps unstable: {float(actual_delta[unstable].mean()) if unstable.any() else np.nan:.4f}")
            print(f"Average future range bps stable: {float(actual_range[stable].mean()) if stable.any() else np.nan:.4f}")
            print(f"Average future range bps unstable: {float(actual_range[unstable].mean()) if unstable.any() else np.nan:.4f}")
            print("WARNING: Instability is a paper-only risk/gating target, not a direction model.")
            return
        nonflat = actual_dir != 0
        directional = pred_dir != 0
        print(f"MAE bps: {float((pred_delta - actual_delta).abs().mean()):.4f}")
        print(f"RMSE bps: {float(np.sqrt(((pred_delta - actual_delta) ** 2).mean())):.4f}")
        print(f"Sign accuracy excluding flat: {float((pred_dir[nonflat] == actual_dir[nonflat]).mean()) if nonflat.any() else np.nan:.2%}")
        print(f"Directional win rate: {float((np.sign(pred_dir[directional]) == np.sign(actual_delta[directional])).mean()) if directional.any() else np.nan:.2%}")
        print(f"Average realized bps when predicted up: {float(actual_delta[pred_dir > 0].mean()) if (pred_dir > 0).any() else np.nan:.4f}")
        print(f"Average realized bps when predicted down: {float(actual_delta[pred_dir < 0].mean()) if (pred_dir < 0).any() else np.nan:.4f}")
        print("Calibration buckets")
        buckets = calibration_buckets(evaluated)
        for row in buckets:
            print(
                f"- {row['bucket']}: rows={row['rows']} "
                f"accuracy={row['accuracy']:.2%} "
                f"avg_conf={row['avg_confidence']:.2%} "
                f"avg_actual_bps={row['avg_actual_delta_bps']:.4f}"
            )
        if calibration_is_inverted(buckets):
            print("WARNING: Model is overconfident/inverted.")
        print("Confidence-threshold directional report")
        threshold_rows = confidence_threshold_directional_report(evaluated)
        for row in threshold_rows:
            print(
                f"- threshold>={row['threshold']:.2f}: rows={row['rows_kept']} "
                f"dir_acc={row['directional_accuracy']:.2%} "
                f"avg_return={row['avg_realized_return_bps']:.4f}bps "
                f"up_return={row['predicted_up_return_bps']:.4f}bps "
                f"down_return={row['predicted_down_return_bps']:.4f}bps "
                f"lift_vs_majority={row['lift_vs_majority']:.2%} "
                f"interesting={row['threshold_interesting']} "
                f"stable={row['threshold_stable_candidate']}"
            )
        print("Best confidence thresholds")
        print_best_threshold("best_threshold_any_rows", best_threshold_summary(threshold_rows, 1))
        print_best_threshold("best_threshold_min_100_rows", best_threshold_summary(threshold_rows, MIN_THRESHOLD_ROWS_INTERESTING))
        print_best_threshold("best_threshold_min_300_rows", best_threshold_summary(threshold_rows, MIN_THRESHOLD_ROWS_STABLE))
    print("Paper-only. No trades/orders/private API.")


if __name__ == "__main__":
    main()
