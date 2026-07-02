import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
HORIZON_SECONDS = int(os.getenv("PRICE_TINY_ENSEMBLE_EVAL_HORIZON_SECONDS", "30"))
MAX_FUTURE_GAP_MS = int(
    os.getenv(
        "PRICE_TINY_ENSEMBLE_EVAL_MAX_FUTURE_GAP_MS",
        str(max(1500, HORIZON_SECONDS * 1500)),
    )
)
MOVE_CONFIDENCE_THRESHOLDS = [
    float(value.strip())
    for value in os.getenv("PRICE_TINY_ENSEMBLE_EVAL_MOVE_THRESHOLDS", "0.70,0.65,0.60,0.55").split(",")
    if value.strip()
]
INSTABILITY_THRESHOLDS = [
    float(value.strip())
    for value in os.getenv("PRICE_TINY_ENSEMBLE_EVAL_INSTABILITY_THRESHOLDS", "0.70,0.75,0.80").split(",")
    if value.strip()
]
ESTIMATED_FEE_BPS = float(os.getenv("PRICE_TINY_ESTIMATED_FEE_BPS", "0"))
ESTIMATED_SLIPPAGE_BPS = float(os.getenv("PRICE_TINY_ESTIMATED_SLIPPAGE_BPS", "0"))
CHARGE_HALF_SPREAD = os.getenv("PRICE_TINY_CHARGE_HALF_SPREAD", "true").strip().lower() in {"1", "true", "yes", "y"}

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
PREDICTIONS_PATH = Path(
    os.getenv(
        "PRICE_TINY_ENSEMBLE_LIVE_PREDICTIONS_PATH",
        VENUE_DIR / f"{SYMBOL}_tiny_price_ensemble_live_predictions.csv",
    )
)
SNAPSHOT_PATH = Path(
    os.getenv(
        "PRICE_TINY_ENSEMBLE_EVAL_SNAPSHOT_PATH",
        VENUE_DIR / f"{SYMBOL}_10s_flow.csv",
    )
)
EVALUATED_ROWS_PATH = Path(
    os.getenv(
        "PRICE_TINY_ENSEMBLE_LIVE_EVALUATED_ROWS_PATH",
        VENUE_DIR / f"{SYMBOL}_tiny_price_ensemble_live_evaluated_rows.csv",
    )
)
SUMMARY_PATH = Path(
    os.getenv(
        "PRICE_TINY_ENSEMBLE_LIVE_EVALUATION_PATH",
        VENUE_DIR / f"{SYMBOL}_tiny_price_ensemble_live_evaluation.csv",
    )
)
EVALUATION_BY_RUN_PATH = Path(
    os.getenv(
        "PRICE_TINY_ENSEMBLE_LIVE_EVALUATION_BY_RUN_PATH",
        VENUE_DIR / f"{SYMBOL}_tiny_price_ensemble_live_evaluation_by_run.csv",
    )
)
THRESHOLD_SENSITIVITY_PATH = Path(
    os.getenv(
        "PRICE_TINY_ENSEMBLE_THRESHOLD_SENSITIVITY_PATH",
        VENUE_DIR / f"{SYMBOL}_tiny_price_ensemble_threshold_sensitivity.csv",
    )
)
SIDE_REGIME_DIAGNOSTICS_PATH = Path(
    os.getenv(
        "PRICE_TINY_ENSEMBLE_SIDE_REGIME_DIAGNOSTICS_PATH",
        VENUE_DIR / f"{SYMBOL}_tiny_price_ensemble_side_regime_diagnostics.csv",
    )
)

for path_name in [
    "PREDICTIONS_PATH",
    "SNAPSHOT_PATH",
    "EVALUATED_ROWS_PATH",
    "SUMMARY_PATH",
    "EVALUATION_BY_RUN_PATH",
    "THRESHOLD_SENSITIVITY_PATH",
    "SIDE_REGIME_DIAGNOSTICS_PATH",
]:
    path = globals()[path_name]
    if not path.is_absolute():
        globals()[path_name] = PROJECT_ROOT / path


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


def numeric(frame, column, default=np.nan):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def text(frame, column, default=""):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str)


def truthy_series(frame, column, default=False):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].fillna(default).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def normalize_predictions(frame):
    if len(frame) == 0:
        return frame
    frame = frame.copy()
    frame["timestamp"] = numeric(frame, "timestamp")
    frame = frame.dropna(subset=["timestamp"]).copy()
    frame["timestamp"] = frame["timestamp"].astype("int64")
    if "symbol" in frame.columns:
        frame = frame[text(frame, "symbol").str.upper() == SYMBOL].copy()
    if "primary_venue" in frame.columns:
        frame = frame[text(frame, "primary_venue").str.lower() == (PRIMARY_VENUE or "legacy")].copy()
    if "run_id" not in frame.columns:
        frame["run_id"] = "legacy_missing_run_id"
    frame["run_id"] = text(frame, "run_id").where(text(frame, "run_id").str.len() > 0, "legacy_missing_run_id")
    for column in [
        "move_model_id",
        "instability_model_id",
        "move_model_path",
        "instability_model_path",
        "model_pinning_status",
        "models_pinned",
    ]:
        if column not in frame.columns:
            frame[column] = "" if column != "models_pinned" else False
    frame["move_model_id"] = text(frame, "move_model_id")
    frame["instability_model_id"] = text(frame, "instability_model_id")
    frame["move_model_path"] = text(frame, "move_model_path")
    frame["instability_model_path"] = text(frame, "instability_model_path")
    frame["model_pinning_status"] = text(frame, "model_pinning_status", "unknown")
    frame["models_pinned"] = truthy_series(frame, "models_pinned", False)
    if "active_move_confidence_threshold" not in frame.columns:
        frame["active_move_confidence_threshold"] = numeric(frame, "move_confidence_threshold", np.nan)
    else:
        frame["active_move_confidence_threshold"] = numeric(frame, "active_move_confidence_threshold").fillna(
            numeric(frame, "move_confidence_threshold", np.nan)
        )
    if "active_instability_threshold" not in frame.columns:
        frame["active_instability_threshold"] = numeric(frame, "instability_max_probability", np.nan)
    else:
        frame["active_instability_threshold"] = numeric(frame, "active_instability_threshold").fillna(
            numeric(frame, "instability_max_probability", np.nan)
        )
    if "regression_enabled" not in frame.columns:
        frame["regression_enabled"] = False
    frame["regression_enabled"] = truthy_series(frame, "regression_enabled", False)
    if "active_allowed_sides" not in frame.columns:
        frame["active_allowed_sides"] = text(frame, "allowed_sides", "both")
    else:
        frame["active_allowed_sides"] = text(frame, "active_allowed_sides").where(
            text(frame, "active_allowed_sides").str.len() > 0,
            text(frame, "allowed_sides", "both"),
        )
    frame["active_allowed_sides"] = (
        frame["active_allowed_sides"]
        .str.strip()
        .str.lower()
        .where(lambda values: values.isin(["long", "short", "both"]), "both")
    )
    frame["allowed_sides"] = frame["active_allowed_sides"]
    move_hash = text(frame, "move_before_adverse_schema_hash")
    instability_hash = text(frame, "instability_schema_hash")
    direction_hash = text(frame, "direction_model_schema_hash")
    direction_hash = direction_hash.where(direction_hash.str.len() > 0, text(frame, "direction_schema_hash"))
    regression_hash = text(frame, "regression_schema_hash")
    computed_required_schema_match = (move_hash.str.len() > 0) & (instability_hash.str.len() > 0) & (move_hash == instability_hash)
    if "required_schema_match" not in frame.columns:
        frame["required_schema_match"] = computed_required_schema_match
    else:
        raw_required_schema_match = text(frame, "required_schema_match")
        explicit_required_schema_match = truthy_series(frame, "required_schema_match", False)
        frame["required_schema_match"] = explicit_required_schema_match.where(
            raw_required_schema_match.str.len() > 0,
            computed_required_schema_match,
        )
    computed_optional_direction_schema_match = (
        direction_hash.str.len() == 0
    ) | ((move_hash.str.len() > 0) & (direction_hash == move_hash))
    computed_optional_regression_schema_match = (
        regression_hash.str.len() == 0
    ) | ((move_hash.str.len() > 0) & (regression_hash == move_hash))
    if "optional_direction_schema_match" not in frame.columns:
        frame["optional_direction_schema_match"] = computed_optional_direction_schema_match
    else:
        raw_optional_direction = text(frame, "optional_direction_schema_match")
        frame["optional_direction_schema_match"] = truthy_series(frame, "optional_direction_schema_match", False).where(
            raw_optional_direction.str.len() > 0,
            computed_optional_direction_schema_match,
        )
    if "optional_regression_schema_match" not in frame.columns:
        frame["optional_regression_schema_match"] = computed_optional_regression_schema_match
    else:
        raw_optional_regression = text(frame, "optional_regression_schema_match")
        frame["optional_regression_schema_match"] = truthy_series(frame, "optional_regression_schema_match", False).where(
            raw_optional_regression.str.len() > 0,
            computed_optional_regression_schema_match,
        )
    if "optional_direction_ignored_reason" not in frame.columns:
        frame["optional_direction_ignored_reason"] = ""
    if "optional_regression_ignored_reason" not in frame.columns:
        frame["optional_regression_ignored_reason"] = ""
    frame = frame.sort_values("timestamp").drop_duplicates(
        ["timestamp", "symbol", "primary_venue", "run_id", "move_model_id", "instability_model_id"],
        keep="last",
    )
    return frame.reset_index(drop=True)


def normalize_snapshots(frame):
    if len(frame) == 0:
        return frame
    frame = frame.copy()
    frame["timestamp"] = numeric(frame, "timestamp")
    frame["mid_price"] = numeric(frame, "mid_price")
    frame = frame.dropna(subset=["timestamp", "mid_price"])
    frame = frame[frame["mid_price"] > 0].copy()
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame.reset_index(drop=True)


def signal_direction(value):
    value = str(value or "").strip().lower()
    if value == "long":
        return 1
    if value == "short":
        return -1
    return 0


def snapshot_spread_bps(snapshots, index):
    if index < 0 or index >= len(snapshots):
        return np.nan
    row = snapshots.iloc[index]
    spread_percent = pd.to_numeric(pd.Series([row.get("spread_percent", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(spread_percent) and spread_percent >= 0:
        # Recorder stores spread_percent as a price ratio, so bps = ratio * 10000.
        return float(spread_percent * 10000.0)
    best_bid = pd.to_numeric(pd.Series([row.get("best_bid", np.nan)]), errors="coerce").iloc[0]
    best_ask = pd.to_numeric(pd.Series([row.get("best_ask", np.nan)]), errors="coerce").iloc[0]
    mid_price = pd.to_numeric(pd.Series([row.get("mid_price", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(best_bid) and np.isfinite(best_ask) and np.isfinite(mid_price) and best_bid > 0 and best_ask > 0 and mid_price > 0:
        return float(((best_ask - best_bid) / mid_price) * 10000.0)
    return np.nan


def total_cost_bps(entry_spread_bps, exit_spread_bps):
    if CHARGE_HALF_SPREAD:
        if np.isfinite(entry_spread_bps) and np.isfinite(exit_spread_bps):
            spread_cost = 0.5 * float(entry_spread_bps) + 0.5 * float(exit_spread_bps)
        elif np.isfinite(entry_spread_bps):
            spread_cost = float(entry_spread_bps)
        else:
            spread_cost = np.nan
    else:
        spread_cost = 0.0
    fee_cost = float(ESTIMATED_FEE_BPS)
    slippage_cost = float(ESTIMATED_SLIPPAGE_BPS)
    if not np.isfinite(spread_cost):
        return spread_cost, fee_cost, slippage_cost, np.nan
    return spread_cost, fee_cost, slippage_cost, spread_cost + fee_cost + slippage_cost


def session_bucket_from_timestamp(timestamp_ms):
    try:
        hour = pd.to_datetime(int(timestamp_ms), unit="ms", utc=True).hour
    except Exception:
        return "unknown"
    if 0 <= hour < 7:
        return "asia_utc_00_06"
    if 7 <= hour < 13:
        return "london_utc_07_12"
    if 13 <= hour < 21:
        return "us_utc_13_20"
    return "late_us_utc_21_23"


def snapshot_context(snapshots, index):
    if index < 0 or index >= len(snapshots):
        return {}
    current = snapshots.iloc[index]
    current_ts = int(current.get("timestamp", 0) or 0)
    mid_values = pd.to_numeric(snapshots["mid_price"], errors="coerce").to_numpy(dtype=np.float64)
    ts_values = pd.to_numeric(snapshots["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    context = {
        "diagnostic_hour_utc": pd.to_datetime(current_ts, unit="ms", utc=True).hour if current_ts else np.nan,
        "diagnostic_session_bucket": session_bucket_from_timestamp(current_ts),
    }
    for column, output_column in [
        ("order_book_imbalance_10bps", "diagnostic_imbalance10"),
        ("order_book_imbalance_25bps", "diagnostic_imbalance25"),
        ("bid_depth_10bps", "diagnostic_bid_depth_10bps"),
        ("ask_depth_10bps", "diagnostic_ask_depth_10bps"),
    ]:
        value = pd.to_numeric(pd.Series([current.get(column, np.nan)]), errors="coerce").iloc[0]
        context[output_column] = float(value) if np.isfinite(value) else np.nan
    for seconds in [30, 60, 120]:
        start_ts = current_ts - seconds * 1000
        window_mask = (ts_values >= start_ts) & (ts_values <= current_ts) & np.isfinite(mid_values) & (mid_values > 0)
        window_mid = mid_values[window_mask]
        if len(window_mid) >= 2:
            returns = np.diff(np.log(window_mid))
            context[f"diagnostic_rolling_volatility_{seconds}s_bps"] = float(np.std(returns) * 10000.0)
            context[f"diagnostic_range_{seconds}s_bps"] = float((np.max(window_mid) / np.min(window_mid) - 1.0) * 10000.0)
        else:
            context[f"diagnostic_rolling_volatility_{seconds}s_bps"] = np.nan
            context[f"diagnostic_range_{seconds}s_bps"] = np.nan
    return context


def attach_realized_outcomes(predictions, snapshots):
    predictions = normalize_predictions(predictions)
    snapshots = normalize_snapshots(snapshots)
    spread_fields_available = (
        "spread_percent" in snapshots.columns
        or {"best_bid", "best_ask", "mid_price"}.issubset(set(snapshots.columns))
    )
    if len(predictions) == 0 or len(snapshots) == 0:
        return pd.DataFrame(), {
            "missing_future_rows": int(len(predictions)),
            "missing_current_snapshot_rows": int(len(predictions)),
            "spread_fields_available": bool(spread_fields_available),
        }

    ts = snapshots["timestamp"].to_numpy(dtype=np.int64)
    mid = snapshots["mid_price"].to_numpy(dtype=np.float64)
    rows = []
    missing_future_rows = 0
    missing_current_snapshot_rows = 0

    for _, pred in predictions.iterrows():
        pred_ts = int(pred["timestamp"])
        current_index = int(np.searchsorted(ts, pred_ts, side="right") - 1)
        if current_index < 0 or current_index >= len(ts):
            missing_current_snapshot_rows += 1
            continue
        current_mid = float(mid[current_index])
        if not np.isfinite(current_mid) or current_mid <= 0:
            missing_current_snapshot_rows += 1
            continue

        target_ts = pred_ts + HORIZON_SECONDS * 1000
        future_index = int(np.searchsorted(ts, target_ts, side="left"))
        if future_index >= len(ts) or int(ts[future_index]) - target_ts > MAX_FUTURE_GAP_MS:
            missing_future_rows += 1
            continue
        future_mid = float(mid[future_index])
        if not np.isfinite(future_mid) or future_mid <= 0:
            missing_future_rows += 1
            continue

        path = mid[current_index + 1 : future_index + 1]
        path = path[np.isfinite(path) & (path > 0)]
        realized_return_bps = (future_mid / current_mid - 1.0) * 10000.0
        max_runup_bps = np.nan
        max_drawdown_bps = np.nan
        if len(path):
            max_runup_bps = (float(np.max(path)) / current_mid - 1.0) * 10000.0
            max_drawdown_bps = (float(np.min(path)) / current_mid - 1.0) * 10000.0
        entry_spread_bps = snapshot_spread_bps(snapshots, current_index)
        exit_spread_bps = snapshot_spread_bps(snapshots, future_index)
        spread_cost_bps, fee_bps, slippage_bps, total_cost = total_cost_bps(entry_spread_bps, exit_spread_bps)

        output = pred.to_dict()
        output["current_mid_price"] = current_mid
        output["future_mid_price"] = future_mid
        output["future_timestamp"] = int(ts[future_index])
        output["entry_spread_bps"] = float(entry_spread_bps)
        output["exit_spread_bps"] = float(exit_spread_bps)
        output["diagnostic_spread_bps"] = float(entry_spread_bps)
        output.update(snapshot_context(snapshots, current_index))
        output["realized_30s_return_bps"] = float(realized_return_bps)
        output["realized_30s_direction"] = 1 if realized_return_bps > 0 else (-1 if realized_return_bps < 0 else 0)
        output["realized_30s_max_runup_bps"] = float(max_runup_bps)
        output["realized_30s_max_drawdown_bps"] = float(max_drawdown_bps)
        output["paper_signal_direction_from_text"] = signal_direction(output.get("final_paper_signal", ""))
        direction = int(output.get("final_paper_signal_direction", 0) or 0)
        if direction == 0:
            direction = output["paper_signal_direction_from_text"]
        output["paper_signal_direction_eval"] = int(direction)
        gross_strategy_return_bps = float(realized_return_bps * direction) if direction else np.nan
        net_strategy_return_bps = float(gross_strategy_return_bps - total_cost) if direction and np.isfinite(total_cost) else np.nan
        output["paper_signal_return_bps"] = gross_strategy_return_bps
        output["gross_strategy_return_bps"] = gross_strategy_return_bps
        output["estimated_spread_cost_bps"] = float(spread_cost_bps)
        output["estimated_fee_bps"] = float(fee_bps)
        output["estimated_slippage_bps"] = float(slippage_bps)
        output["estimated_total_cost_bps"] = float(total_cost)
        output["net_strategy_return_bps"] = net_strategy_return_bps
        output["spread_cost_available"] = bool(np.isfinite(spread_cost_bps))
        output["prediction_correct"] = bool(direction != 0 and np.sign(direction) == np.sign(realized_return_bps))
        output["net_positive"] = bool(direction != 0 and np.isfinite(net_strategy_return_bps) and net_strategy_return_bps > 0)
        if direction > 0:
            output["paper_signal_mfe_bps"] = float(max_runup_bps)
            output["paper_signal_mae_bps"] = float(max_drawdown_bps)
        elif direction < 0:
            output["paper_signal_mfe_bps"] = float(-max_drawdown_bps) if np.isfinite(max_drawdown_bps) else np.nan
            output["paper_signal_mae_bps"] = float(-max_runup_bps) if np.isfinite(max_runup_bps) else np.nan
        else:
            output["paper_signal_mfe_bps"] = np.nan
            output["paper_signal_mae_bps"] = np.nan
        rows.append(output)

    return pd.DataFrame(rows), {
        "missing_future_rows": int(missing_future_rows),
        "missing_current_snapshot_rows": int(missing_current_snapshot_rows),
        "spread_fields_available": bool(spread_fields_available),
    }


def average(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def count_no_trade_reasons(frame):
    if len(frame) == 0:
        return pd.DataFrame(columns=["no_trade_reason", "rows"])
    reasons = text(frame, "no_trade_reason")
    reasons = reasons.where(reasons.str.len() > 0, text(frame, "decision_reason"))
    reasons = reasons.where(reasons.str.len() > 0, "(blank)")
    exploded = []
    for value in reasons:
        parts = [part.strip() for part in str(value).split(";") if part.strip()]
        exploded.extend(parts or ["(blank)"])
    if not exploded:
        return pd.DataFrame(columns=["no_trade_reason", "rows"])
    return (
        pd.Series(exploded)
        .value_counts()
        .rename_axis("no_trade_reason")
        .reset_index(name="rows")
    )


def summarize(predictions, evaluated, attach_stats):
    total_rows = int(len(predictions))
    evaluated_rows = int(len(evaluated))
    final_signal = text(predictions, "final_paper_signal").str.lower()
    evaluated_signal = text(evaluated, "final_paper_signal").str.lower()
    signal_rows = evaluated[evaluated["paper_signal_direction_eval"] != 0].copy() if len(evaluated) else pd.DataFrame()

    stale_count = int((text(predictions, "snapshot_freshness").str.lower() != "fresh").sum()) if len(predictions) else 0
    missing_count = int(
        text(predictions, "decision_reason").str.contains("missing_", case=False, regex=False).sum()
        if len(predictions)
        else 0
    )
    required_schema_mismatch_count = int((~truthy_series(predictions, "required_schema_match", False)).sum()) if len(predictions) else 0
    optional_direction_schema_mismatch_count = int((~truthy_series(predictions, "optional_direction_schema_match", True)).sum()) if len(predictions) else 0
    optional_regression_schema_mismatch_count = int((~truthy_series(predictions, "optional_regression_schema_match", True)).sum()) if len(predictions) else 0
    all_model_schema_mismatch_count = (
        required_schema_mismatch_count
        + optional_direction_schema_mismatch_count
        + optional_regression_schema_mismatch_count
    )
    no_trade_count = int((final_signal == "no_trade").sum()) if len(predictions) else 0
    long_count = int((final_signal == "long").sum()) if len(predictions) else 0
    short_count = int((final_signal == "short").sum()) if len(predictions) else 0
    evaluated_long_count = int((evaluated_signal == "long").sum()) if len(evaluated) else 0
    evaluated_short_count = int((evaluated_signal == "short").sum()) if len(evaluated) else 0

    sign_accuracy = np.nan
    net_positive_rate = np.nan
    if len(signal_rows):
        sign_accuracy = float(signal_rows["prediction_correct"].mean())
        net_positive_rate = float(truthy_series(signal_rows, "net_positive", False).mean())

    summary = {
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE,
        "prediction_path": str(PREDICTIONS_PATH),
        "snapshot_path": str(SNAPSHOT_PATH),
        "horizon_seconds": HORIZON_SECONDS,
        "allowed_sides_values": ",".join(sorted(set(text(predictions, "active_allowed_sides", "both")))) if len(predictions) else "",
        "total_rows": total_rows,
        "evaluated_rows": evaluated_rows,
        "paper_signal_rows": int(len(signal_rows)),
        "no_trade_rows": no_trade_count,
        "long_count": long_count,
        "short_count": short_count,
        "evaluated_long_count": evaluated_long_count,
        "evaluated_short_count": evaluated_short_count,
        "realized_30s_avg_return_for_paper_signals_bps": average(signal_rows["paper_signal_return_bps"]) if len(signal_rows) else np.nan,
        "realized_30s_avg_raw_return_for_long_signals_bps": average(signal_rows.loc[signal_rows["paper_signal_direction_eval"] > 0, "realized_30s_return_bps"]) if len(signal_rows) else np.nan,
        "realized_30s_avg_short_strategy_return_bps": average(signal_rows.loc[signal_rows["paper_signal_direction_eval"] < 0, "paper_signal_return_bps"]) if len(signal_rows) else np.nan,
        "sign_accuracy": sign_accuracy,
        "gross_avg_strategy_return_bps": average(signal_rows["gross_strategy_return_bps"]) if len(signal_rows) else np.nan,
        "net_avg_strategy_return_bps": average(signal_rows["net_strategy_return_bps"]) if len(signal_rows) else np.nan,
        "gross_sign_accuracy": sign_accuracy,
        "net_positive_rate": net_positive_rate,
        "average_spread_cost_bps": average(signal_rows["estimated_spread_cost_bps"]) if len(signal_rows) else np.nan,
        "average_fee_bps": average(signal_rows["estimated_fee_bps"]) if len(signal_rows) else np.nan,
        "average_slippage_bps": average(signal_rows["estimated_slippage_bps"]) if len(signal_rows) else np.nan,
        "average_total_cost_bps": average(signal_rows["estimated_total_cost_bps"]) if len(signal_rows) else np.nan,
        "spread_cost_missing_signal_rows": int((~truthy_series(signal_rows, "spread_cost_available", False)).sum()) if len(signal_rows) else 0,
        "avg_mfe_bps": average(signal_rows["paper_signal_mfe_bps"]) if len(signal_rows) else np.nan,
        "avg_mae_bps": average(signal_rows["paper_signal_mae_bps"]) if len(signal_rows) else np.nan,
        "stale_count": stale_count,
        "missing_count": missing_count,
        "schema_mismatch_count": required_schema_mismatch_count,
        "required_schema_mismatch_rows": required_schema_mismatch_count,
        "optional_direction_schema_mismatch_rows": optional_direction_schema_mismatch_count,
        "optional_regression_schema_mismatch_rows": optional_regression_schema_mismatch_count,
        "all_model_schema_mismatch_count": all_model_schema_mismatch_count,
        "missing_future_rows": int(attach_stats.get("missing_future_rows", 0)),
        "missing_current_snapshot_rows": int(attach_stats.get("missing_current_snapshot_rows", 0)),
        "spread_fields_available": bool(attach_stats.get("spread_fields_available", False)),
    }
    return summary


def summarize_by_run(predictions, evaluated):
    if len(predictions) == 0:
        return pd.DataFrame()

    group_columns = [
        "run_id",
        "move_model_id",
        "instability_model_id",
        "active_move_confidence_threshold",
        "active_instability_threshold",
        "active_allowed_sides",
        "regression_enabled",
    ]
    evaluated_lookup = evaluated.copy() if len(evaluated) else pd.DataFrame()
    rows = []
    grouped = predictions.groupby(group_columns, dropna=False, sort=False)
    for group_values, group in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_key = dict(zip(group_columns, group_values))
        matched = evaluated_lookup
        for column, value in group_key.items():
            if len(matched) == 0 or column not in matched.columns:
                matched = pd.DataFrame()
                break
            if pd.isna(value):
                matched = matched[matched[column].isna()].copy()
            else:
                matched = matched[matched[column] == value].copy()

        final_signal = text(group, "final_paper_signal").str.lower()
        signal_rows = matched[matched["paper_signal_direction_eval"] != 0].copy() if len(matched) else pd.DataFrame()
        sign_accuracy = float(signal_rows["prediction_correct"].mean()) if len(signal_rows) else np.nan
        net_positive_rate = float(truthy_series(signal_rows, "net_positive", False).mean()) if len(signal_rows) else np.nan
        stale_rows = int((text(group, "snapshot_freshness").str.lower() != "fresh").sum())
        required_schema_mismatch_rows = int((~truthy_series(group, "required_schema_match", False)).sum())
        optional_direction_schema_mismatch_rows = int((~truthy_series(group, "optional_direction_schema_match", True)).sum())
        optional_regression_schema_mismatch_rows = int((~truthy_series(group, "optional_regression_schema_match", True)).sum())
        rows.append(
            {
                "symbol": SYMBOL,
                "primary_venue": PRIMARY_VENUE,
                "run_id": str(group_key.get("run_id", "")),
                "move_model_id": str(group_key.get("move_model_id", "")),
                "instability_model_id": str(group_key.get("instability_model_id", "")),
                "move_model_path": text(group, "move_model_path").iloc[-1] if "move_model_path" in group.columns and len(group) else "",
                "instability_model_path": text(group, "instability_model_path").iloc[-1] if "instability_model_path" in group.columns and len(group) else "",
                "model_pinning_status": text(group, "model_pinning_status").iloc[-1] if "model_pinning_status" in group.columns and len(group) else "unknown",
                "models_pinned": bool(truthy_series(group, "models_pinned", False).all()) if len(group) else False,
                "active_move_confidence_threshold": group_key.get("active_move_confidence_threshold", np.nan),
                "active_instability_threshold": group_key.get("active_instability_threshold", np.nan),
                "active_allowed_sides": group_key.get("active_allowed_sides", "both"),
                "regression_enabled": bool(group_key.get("regression_enabled", False)),
                "total_rows": int(len(group)),
                "evaluated_rows": int(len(matched)),
                "paper_signal_rows": int(len(signal_rows)),
                "long_count": int((final_signal == "long").sum()),
                "short_count": int((final_signal == "short").sum()),
                "sign_accuracy": sign_accuracy,
                "average_realized_30s_strategy_return_bps": average(signal_rows["paper_signal_return_bps"]) if len(signal_rows) else np.nan,
                "gross_avg_strategy_return_bps": average(signal_rows["gross_strategy_return_bps"]) if len(signal_rows) else np.nan,
                "net_avg_strategy_return_bps": average(signal_rows["net_strategy_return_bps"]) if len(signal_rows) else np.nan,
                "gross_sign_accuracy": sign_accuracy,
                "net_positive_rate": net_positive_rate,
                "average_spread_cost_bps": average(signal_rows["estimated_spread_cost_bps"]) if len(signal_rows) else np.nan,
                "average_fee_bps": average(signal_rows["estimated_fee_bps"]) if len(signal_rows) else np.nan,
                "average_slippage_bps": average(signal_rows["estimated_slippage_bps"]) if len(signal_rows) else np.nan,
                "average_total_cost_bps": average(signal_rows["estimated_total_cost_bps"]) if len(signal_rows) else np.nan,
                "average_mfe_bps": average(signal_rows["paper_signal_mfe_bps"]) if len(signal_rows) else np.nan,
                "average_mae_bps": average(signal_rows["paper_signal_mae_bps"]) if len(signal_rows) else np.nan,
                "stale_rows": stale_rows,
                "missing_future_rows": max(0, int(len(group)) - int(len(matched))),
                "required_schema_mismatch_rows": required_schema_mismatch_rows,
                "optional_direction_schema_mismatch_rows": optional_direction_schema_mismatch_rows,
                "optional_regression_schema_mismatch_rows": optional_regression_schema_mismatch_rows,
                "all_model_schema_mismatch_rows": (
                    required_schema_mismatch_rows
                    + optional_direction_schema_mismatch_rows
                    + optional_regression_schema_mismatch_rows
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["total_rows", "run_id"], ascending=[False, True]).reset_index(drop=True)


def side_specific_excursions(frame, direction):
    max_runup = pd.to_numeric(frame["realized_30s_max_runup_bps"], errors="coerce").to_numpy(dtype=np.float64)
    max_drawdown = pd.to_numeric(frame["realized_30s_max_drawdown_bps"], errors="coerce").to_numpy(dtype=np.float64)
    direction = np.asarray(direction, dtype=np.int64)
    mfe = np.where(direction > 0, max_runup, np.where(direction < 0, -max_drawdown, np.nan))
    mae = np.where(direction > 0, max_drawdown, np.where(direction < 0, -max_runup, np.nan))
    return mfe, mae


def allowed_side_mask(directions, allowed_sides):
    directions = pd.to_numeric(directions, errors="coerce").fillna(0).astype(int)
    allowed_sides = pd.Series(allowed_sides, index=directions.index if not hasattr(allowed_sides, "index") else allowed_sides.index)
    allowed_sides = allowed_sides.fillna("both").astype(str).str.strip().str.lower()
    allowed_sides = allowed_sides.where(allowed_sides.isin(["long", "short", "both"]), "both")
    return (
        (allowed_sides == "both")
        | ((allowed_sides == "long") & (directions > 0))
        | ((allowed_sides == "short") & (directions < 0))
    )


def threshold_sensitivity(predictions, evaluated):
    if len(predictions) == 0:
        return pd.DataFrame()

    predictions = predictions.copy()
    evaluated = evaluated.copy()
    move_confidence = numeric(predictions, "move_before_adverse_confidence")
    instability_probability = numeric(predictions, "instability_probability")
    move_direction = numeric(predictions, "move_before_adverse_direction", 0).fillna(0).astype(int)
    stale = text(predictions, "snapshot_freshness").str.lower() != "fresh"
    side_allowed = allowed_side_mask(move_direction, text(predictions, "active_allowed_sides", "both"))

    has_required_values = (
        np.isfinite(move_confidence.to_numpy(dtype=np.float64))
        & np.isfinite(instability_probability.to_numpy(dtype=np.float64))
        & (move_direction.to_numpy(dtype=np.int64) != 0)
    )

    evaluated_by_timestamp = set()
    if len(evaluated):
        evaluated_by_timestamp = set(pd.to_numeric(evaluated["timestamp"], errors="coerce").dropna().astype("int64").tolist())

    rows = []
    for move_threshold in MOVE_CONFIDENCE_THRESHOLDS:
        for instability_threshold in INSTABILITY_THRESHOLDS:
            gate = (
                has_required_values
                & (move_confidence.to_numpy(dtype=np.float64) >= move_threshold)
                & (instability_probability.to_numpy(dtype=np.float64) < instability_threshold)
            )
            side_blocked_rows_excluded = int((gate & (~side_allowed.to_numpy(dtype=bool))).sum())
            kept_mask = gate & side_allowed.to_numpy(dtype=bool) & (~stale.to_numpy(dtype=bool))
            kept_predictions = predictions.loc[kept_mask].copy()
            kept_timestamps = set(pd.to_numeric(kept_predictions["timestamp"], errors="coerce").dropna().astype("int64").tolist())

            stale_rows_excluded = int((gate & stale.to_numpy(dtype=bool)).sum())
            evaluated_kept = pd.DataFrame()
            if len(evaluated) and kept_timestamps:
                evaluated_kept = evaluated[evaluated["timestamp"].astype("int64").isin(kept_timestamps)].copy()

            rows_kept = int(len(kept_predictions))
            evaluated_rows = int(len(evaluated_kept))
            missing_future_rows_excluded = max(0, rows_kept - evaluated_rows)
            long_count = int((numeric(kept_predictions, "move_before_adverse_direction", 0) > 0).sum()) if len(kept_predictions) else 0
            short_count = int((numeric(kept_predictions, "move_before_adverse_direction", 0) < 0).sum()) if len(kept_predictions) else 0

            sign_accuracy = np.nan
            avg_strategy_return = np.nan
            net_positive_rate = np.nan
            avg_net_strategy_return = np.nan
            avg_spread_cost = np.nan
            avg_fee = np.nan
            avg_slippage = np.nan
            avg_total_cost = np.nan
            avg_mfe = np.nan
            avg_mae = np.nan
            if evaluated_rows:
                hypo_direction = numeric(evaluated_kept, "move_before_adverse_direction", 0).fillna(0).astype(int).to_numpy(dtype=np.int64)
                realized_return = pd.to_numeric(evaluated_kept["realized_30s_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
                finite = np.isfinite(realized_return) & (hypo_direction != 0)
                if finite.any():
                    strategy_return = realized_return[finite] * hypo_direction[finite]
                    total_cost = pd.to_numeric(evaluated_kept.loc[finite, "estimated_total_cost_bps"], errors="coerce").to_numpy(dtype=np.float64)
                    net_return = strategy_return - total_cost
                    sign_accuracy = float((np.sign(hypo_direction[finite]) == np.sign(realized_return[finite])).mean())
                    avg_strategy_return = float(np.mean(strategy_return))
                    avg_net_strategy_return = float(np.nanmean(net_return)) if np.isfinite(net_return).any() else np.nan
                    net_positive_rate = float(np.nanmean(net_return > 0)) if np.isfinite(net_return).any() else np.nan
                    avg_spread_cost = average(evaluated_kept.loc[finite, "estimated_spread_cost_bps"])
                    avg_fee = average(evaluated_kept.loc[finite, "estimated_fee_bps"])
                    avg_slippage = average(evaluated_kept.loc[finite, "estimated_slippage_bps"])
                    avg_total_cost = average(evaluated_kept.loc[finite, "estimated_total_cost_bps"])
                    mfe, mae = side_specific_excursions(evaluated_kept.loc[finite], hypo_direction[finite])
                    avg_mfe = float(np.nanmean(mfe)) if np.isfinite(mfe).any() else np.nan
                    avg_mae = float(np.nanmean(mae)) if np.isfinite(mae).any() else np.nan

            rows.append(
                {
                    "symbol": SYMBOL,
                    "primary_venue": PRIMARY_VENUE,
                    "move_confidence_threshold": float(move_threshold),
                    "instability_probability_threshold": float(instability_threshold),
                    "rows_kept": rows_kept,
                    "evaluated_rows": evaluated_rows,
                    "long_count": long_count,
                    "short_count": short_count,
                    "sign_accuracy": sign_accuracy,
                    "average_realized_30s_strategy_return_bps": avg_strategy_return,
                    "gross_avg_strategy_return_bps": avg_strategy_return,
                    "net_avg_strategy_return_bps": avg_net_strategy_return,
                    "gross_sign_accuracy": sign_accuracy,
                    "net_positive_rate": net_positive_rate,
                    "average_spread_cost_bps": avg_spread_cost,
                    "average_fee_bps": avg_fee,
                    "average_slippage_bps": avg_slippage,
                    "average_total_cost_bps": avg_total_cost,
                    "average_mfe_bps": avg_mfe,
                    "average_mae_bps": avg_mae,
                    "stale_rows_excluded": stale_rows_excluded,
                    "side_blocked_rows_excluded": side_blocked_rows_excluded,
                    "missing_future_rows_excluded": missing_future_rows_excluded,
                }
            )
    return pd.DataFrame(rows)


def bucket_fixed(values, bins, labels, missing_label="missing"):
    values = pd.to_numeric(values, errors="coerce")
    output = pd.Series(missing_label, index=values.index, dtype="object")
    finite = values.notna()
    if finite.any():
        output.loc[finite] = pd.cut(values.loc[finite], bins=bins, labels=labels, include_lowest=True).astype("object")
    return output.fillna(missing_label).astype(str)


def bucket_quantile(values, label_prefix):
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    output = pd.Series("missing", index=values.index, dtype="object")
    finite = values.notna()
    unique_count = values.loc[finite].nunique()
    if unique_count >= 4:
        labels = [f"{label_prefix}_p00_p25", f"{label_prefix}_p25_p50", f"{label_prefix}_p50_p75", f"{label_prefix}_p75_p100"]
        try:
            output.loc[finite] = pd.qcut(values.loc[finite], q=4, labels=labels, duplicates="drop").astype("object")
        except ValueError:
            output.loc[finite] = f"{label_prefix}_single"
    elif finite.any():
        output.loc[finite] = f"{label_prefix}_single"
    return output.fillna("missing").astype(str)


def first_available_numeric(frame, columns):
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def add_live_bucket_columns(frame):
    frame = frame.copy()
    frame["bucket_spread_percentile"] = bucket_quantile(first_available_numeric(frame, ["diagnostic_spread_bps", "entry_spread_bps", "estimated_spread_cost_bps"]), "spread")
    frame["bucket_volatility_range"] = bucket_quantile(
        first_available_numeric(
            frame,
            [
                "diagnostic_rolling_volatility_60s_bps",
                "diagnostic_rolling_volatility_30s_bps",
                "diagnostic_range_60s_bps",
                "diagnostic_range_30s_bps",
            ],
        ),
        "vol_range",
    )
    frame["bucket_instability_probability"] = bucket_fixed(
        numeric(frame, "instability_probability"),
        [-np.inf, 0.50, 0.60, 0.70, 0.80, np.inf],
        ["instability_<0.50", "instability_0.50_0.60", "instability_0.60_0.70", "instability_0.70_0.80", "instability_>=0.80"],
    )
    frame["bucket_move_confidence"] = bucket_fixed(
        numeric(frame, "move_before_adverse_confidence"),
        [-np.inf, 0.55, 0.60, 0.65, 0.70, np.inf],
        ["move_conf_<0.55", "move_conf_0.55_0.60", "move_conf_0.60_0.65", "move_conf_0.65_0.70", "move_conf_>=0.70"],
    )
    if "diagnostic_session_bucket" in frame.columns:
        frame["bucket_hour_session"] = text(frame, "diagnostic_session_bucket").where(text(frame, "diagnostic_session_bucket").str.len() > 0, "missing")
    else:
        frame["bucket_hour_session"] = frame["timestamp"].apply(session_bucket_from_timestamp)
    imbalance = first_available_numeric(frame, ["diagnostic_imbalance10", "diagnostic_imbalance25"])
    frame["bucket_bid_ask_imbalance"] = bucket_fixed(
        imbalance,
        [-np.inf, -0.20, -0.05, 0.05, 0.20, np.inf],
        ["ask_heavy", "mild_ask_heavy", "balanced", "mild_bid_heavy", "bid_heavy"],
    )
    return frame


def side_regime_metric_row(frame, side_label, meta, bucket_type="all", bucket_value="all"):
    if len(frame) == 0:
        direction_value = 1 if side_label == "long" else -1
        rows = frame
    else:
        direction_value = 1 if side_label == "long" else -1
        rows = frame[pd.to_numeric(frame["diagnostic_direction"], errors="coerce").fillna(0).astype(int) == direction_value].copy()
    if len(rows) == 0:
        return {
            **meta,
            "bucket_type": bucket_type,
            "bucket_value": bucket_value,
            "side": side_label,
            "rows": 0,
            "sign_accuracy": np.nan,
            "gross_avg_return_bps": np.nan,
            "net_avg_return_bps": np.nan,
            "average_mfe_bps": np.nan,
            "average_mae_bps": np.nan,
            "average_spread_cost_bps": np.nan,
            "net_positive_rate": np.nan,
            "paper_only": True,
            "no_promotion": True,
        }
    direction = np.full(len(rows), direction_value, dtype=np.int64)
    realized = pd.to_numeric(rows["realized_30s_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
    gross = pd.to_numeric(rows["diagnostic_gross_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
    net = pd.to_numeric(rows["diagnostic_net_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
    sign_accuracy = float((np.sign(realized) == np.sign(direction)).mean()) if len(rows) else np.nan
    return {
        **meta,
        "bucket_type": bucket_type,
        "bucket_value": bucket_value,
        "side": side_label,
        "rows": int(len(rows)),
        "sign_accuracy": sign_accuracy,
        "gross_avg_return_bps": average(pd.Series(gross)),
        "net_avg_return_bps": average(pd.Series(net)),
        "average_mfe_bps": average(rows["diagnostic_mfe_bps"]),
        "average_mae_bps": average(rows["diagnostic_mae_bps"]),
        "average_spread_cost_bps": average(rows["estimated_spread_cost_bps"]),
        "net_positive_rate": float(np.nanmean(net > 0)) if np.isfinite(net).any() else np.nan,
        "paper_only": True,
        "no_promotion": True,
    }


def add_hypothetical_strategy_columns(frame, direction_column):
    frame = frame.copy()
    direction = pd.to_numeric(frame[direction_column], errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int64)
    realized = pd.to_numeric(frame["realized_30s_return_bps"], errors="coerce").to_numpy(dtype=np.float64)
    total_cost = pd.to_numeric(frame["estimated_total_cost_bps"], errors="coerce").to_numpy(dtype=np.float64)
    frame["diagnostic_direction"] = direction
    frame["diagnostic_gross_return_bps"] = realized * direction
    frame["diagnostic_net_return_bps"] = frame["diagnostic_gross_return_bps"] - total_cost
    mfe, mae = side_specific_excursions(frame, direction)
    frame["diagnostic_mfe_bps"] = mfe
    frame["diagnostic_mae_bps"] = mae
    return frame


def append_side_and_bucket_rows(rows, frame, meta):
    if len(frame) == 0:
        for side in ["long", "short"]:
            rows.append(side_regime_metric_row(frame, side, meta))
        return
    frame = add_live_bucket_columns(frame)
    bucket_columns = [
        ("all", None),
        ("spread_percentile_bucket", "bucket_spread_percentile"),
        ("volatility_range_bucket", "bucket_volatility_range"),
        ("instability_probability_bucket", "bucket_instability_probability"),
        ("move_confidence_bucket", "bucket_move_confidence"),
        ("hour_session_bucket", "bucket_hour_session"),
        ("bid_ask_imbalance_bucket", "bucket_bid_ask_imbalance"),
    ]
    for bucket_type, column in bucket_columns:
        if column is None:
            for side in ["long", "short"]:
                rows.append(side_regime_metric_row(frame, side, meta, "all", "all"))
            continue
        if column not in frame.columns:
            continue
        for bucket_value, bucket_frame in frame.groupby(column, dropna=False, sort=True):
            for side in ["long", "short"]:
                rows.append(side_regime_metric_row(bucket_frame, side, meta, bucket_type, str(bucket_value)))


def side_regime_diagnostics(predictions, evaluated):
    rows = []
    if len(evaluated):
        group_columns = [
            "run_id",
            "move_model_id",
            "instability_model_id",
            "active_move_confidence_threshold",
            "active_instability_threshold",
            "active_allowed_sides",
            "regression_enabled",
        ]
        for group_values, group in evaluated.groupby(group_columns, dropna=False, sort=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            meta = dict(zip(group_columns, group_values))
            signal_rows = group[pd.to_numeric(group["paper_signal_direction_eval"], errors="coerce").fillna(0).astype(int) != 0].copy()
            signal_rows = add_hypothetical_strategy_columns(signal_rows, "paper_signal_direction_eval") if len(signal_rows) else signal_rows
            append_side_and_bucket_rows(
                rows,
                signal_rows,
                {
                    "diagnostic_source": "live_rule",
                    "rule_name": "live_rule",
                    "symbol": SYMBOL,
                    "primary_venue": PRIMARY_VENUE,
                    **meta,
                },
            )

    if len(evaluated):
        evaluated = evaluated.copy()
        evaluated["move_before_adverse_direction_for_gate"] = numeric(evaluated, "move_before_adverse_direction", 0).fillna(0).astype(int)
        stale = text(evaluated, "snapshot_freshness").str.lower() != "fresh"
        required_schema_ok = truthy_series(evaluated, "required_schema_match", False)
        side_allowed = allowed_side_mask(
            evaluated["move_before_adverse_direction_for_gate"],
            text(evaluated, "active_allowed_sides", "both"),
        )
        for move_threshold in MOVE_CONFIDENCE_THRESHOLDS:
            for instability_threshold in INSTABILITY_THRESHOLDS:
                base_gate = (
                    (~stale)
                    & required_schema_ok
                    & (numeric(evaluated, "move_before_adverse_confidence") >= move_threshold)
                    & (numeric(evaluated, "instability_probability") < instability_threshold)
                    & (numeric(evaluated, "move_before_adverse_direction", 0).fillna(0).astype(int) != 0)
                    & side_allowed
                )
                gated = evaluated.loc[base_gate].copy()
                gated = add_hypothetical_strategy_columns(gated, "move_before_adverse_direction_for_gate") if len(gated) else gated
                for run_id, run_frame in gated.groupby("run_id", dropna=False, sort=False) if len(gated) else []:
                    append_side_and_bucket_rows(
                        rows,
                        run_frame,
                        {
                            "diagnostic_source": "threshold_sensitivity",
                            "rule_name": f"move>={move_threshold:.2f}_and_instability<{instability_threshold:.2f}",
                            "symbol": SYMBOL,
                            "primary_venue": PRIMARY_VENUE,
                            "run_id": run_id,
                            "active_move_confidence_threshold": move_threshold,
                            "active_instability_threshold": instability_threshold,
                            "regression_enabled": "hypothetical",
                        },
                    )
    return pd.DataFrame(rows)


def print_threshold_sensitivity(frame):
    print("")
    print("Threshold sensitivity")
    print(
        "Hypothetical rule: move_confidence >= threshold AND "
        "instability_probability < threshold; side = original move_before_adverse direction."
    )
    if len(frame) == 0:
        print("- no rows")
        return
    for _, row in frame.iterrows():
        sign_accuracy = row["sign_accuracy"]
        avg_return = row["average_realized_30s_strategy_return_bps"]
        avg_mfe = row["average_mfe_bps"]
        avg_mae = row["average_mae_bps"]
        print(
            f"- move>={row['move_confidence_threshold']:.2f} "
            f"instability<{row['instability_probability_threshold']:.2f}: "
            f"kept={int(row['rows_kept'])} "
            f"evaluated={int(row['evaluated_rows'])} "
            f"long={int(row['long_count'])} "
            f"short={int(row['short_count'])} "
            f"sign_acc={sign_accuracy:.2%}" if np.isfinite(sign_accuracy) else
            f"- move>={row['move_confidence_threshold']:.2f} "
            f"instability<{row['instability_probability_threshold']:.2f}: "
            f"kept={int(row['rows_kept'])} "
            f"evaluated={int(row['evaluated_rows'])} "
            f"long={int(row['long_count'])} "
            f"short={int(row['short_count'])} "
            f"sign_acc=n/a",
            end="",
        )
        print(
            f" avg_return={avg_return:.4f}bps" if np.isfinite(avg_return) else " avg_return=n/a",
            end="",
        )
        net_return = row.get("net_avg_strategy_return_bps", np.nan)
        total_cost = row.get("average_total_cost_bps", np.nan)
        print(
            f" net={net_return:.4f}bps" if np.isfinite(net_return) else " net=n/a",
            end="",
        )
        print(
            f" cost={total_cost:.4f}bps" if np.isfinite(total_cost) else " cost=n/a",
            end="",
        )
        net_positive_rate = row.get("net_positive_rate", np.nan)
        print(
            f" net_pos={net_positive_rate:.2%}" if np.isfinite(net_positive_rate) else " net_pos=n/a",
            end="",
        )
        print(
            f" avg_mfe={avg_mfe:.4f}bps" if np.isfinite(avg_mfe) else " avg_mfe=n/a",
            end="",
        )
        print(
            f" avg_mae={avg_mae:.4f}bps" if np.isfinite(avg_mae) else " avg_mae=n/a",
            end="",
        )
        print(
            f" stale_excluded={int(row['stale_rows_excluded'])} "
            f"side_blocked={int(row.get('side_blocked_rows_excluded', 0))} "
            f"missing_future_excluded={int(row['missing_future_rows_excluded'])}"
        )
    print(f"Threshold sensitivity output: {THRESHOLD_SENSITIVITY_PATH}")


def print_summary(summary, no_trade_reasons):
    print("Tiny-price live ensemble evaluator")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Snapshots: {SNAPSHOT_PATH}")
    print(f"Horizon: {HORIZON_SECONDS}s")
    print(
        "Cost assumptions: "
        f"fee={ESTIMATED_FEE_BPS:.4f}bps "
        f"slippage={ESTIMATED_SLIPPAGE_BPS:.4f}bps "
        f"charge_half_spread={CHARGE_HALF_SPREAD}"
    )
    print("")
    print("Summary")
    print(f"- total rows: {summary['total_rows']}")
    print(f"- allowed sides values: {summary.get('allowed_sides_values', '')}")
    print(f"- evaluated rows with realized {HORIZON_SECONDS}s outcome: {summary['evaluated_rows']}")
    print(f"- paper signal rows: {summary['paper_signal_rows']}")
    print(f"- long count: {summary['long_count']}")
    print(f"- short count: {summary['short_count']}")
    print(f"- evaluated long count: {summary['evaluated_long_count']}")
    print(f"- evaluated short count: {summary['evaluated_short_count']}")
    print(f"- no_trade rows: {summary['no_trade_rows']}")
    print(f"- realized 30s avg return for paper signals: {summary['realized_30s_avg_return_for_paper_signals_bps']:.4f} bps")
    print(f"- gross avg strategy return: {summary['gross_avg_strategy_return_bps']:.4f} bps" if np.isfinite(summary["gross_avg_strategy_return_bps"]) else "- gross avg strategy return: n/a")
    print(f"- net avg strategy return: {summary['net_avg_strategy_return_bps']:.4f} bps" if np.isfinite(summary["net_avg_strategy_return_bps"]) else "- net avg strategy return: n/a")
    print(f"- sign accuracy: {summary['sign_accuracy']:.2%}" if np.isfinite(summary["sign_accuracy"]) else "- sign accuracy: n/a")
    print(f"- net positive rate: {summary['net_positive_rate']:.2%}" if np.isfinite(summary["net_positive_rate"]) else "- net positive rate: n/a")
    print(f"- average spread cost: {summary['average_spread_cost_bps']:.4f} bps" if np.isfinite(summary["average_spread_cost_bps"]) else "- average spread cost: n/a")
    print(f"- average fee: {summary['average_fee_bps']:.4f} bps" if np.isfinite(summary["average_fee_bps"]) else "- average fee: n/a")
    print(f"- average slippage: {summary['average_slippage_bps']:.4f} bps" if np.isfinite(summary["average_slippage_bps"]) else "- average slippage: n/a")
    print(f"- average total cost: {summary['average_total_cost_bps']:.4f} bps" if np.isfinite(summary["average_total_cost_bps"]) else "- average total cost: n/a")
    print(f"- spread fields available: {summary['spread_fields_available']}")
    if CHARGE_HALF_SPREAD and not summary["spread_fields_available"]:
        print("- WARNING: half-spread cost is enabled but spread/bid/ask fields are missing from snapshots")
    if CHARGE_HALF_SPREAD and summary["spread_cost_missing_signal_rows"] > 0:
        print(f"- WARNING: spread cost missing for {summary['spread_cost_missing_signal_rows']} signal rows")
    print(f"- average MFE: {summary['avg_mfe_bps']:.4f} bps" if np.isfinite(summary["avg_mfe_bps"]) else "- average MFE: n/a")
    print(f"- average MAE: {summary['avg_mae_bps']:.4f} bps" if np.isfinite(summary["avg_mae_bps"]) else "- average MAE: n/a")
    print("")
    print("Failure/quality counts")
    print(f"- stale rows: {summary['stale_count']}")
    print(f"- missing required/model rows: {summary['missing_count']}")
    print(f"- required move/instability schema mismatch rows: {summary['schema_mismatch_count']}")
    print(f"- optional direction schema mismatch rows: {summary['optional_direction_schema_mismatch_rows']}")
    print(f"- optional regression schema mismatch rows: {summary['optional_regression_schema_mismatch_rows']}")
    print(f"- all-model schema mismatch rows, diagnostic total: {summary['all_model_schema_mismatch_count']}")
    print(f"- missing current snapshot rows: {summary['missing_current_snapshot_rows']}")
    print(f"- missing future outcome rows: {summary['missing_future_rows']}")
    print("")
    print("No-trade reasons")
    if len(no_trade_reasons) == 0:
        print("- none")
    else:
        for _, row in no_trade_reasons.head(25).iterrows():
            print(f"- {row['no_trade_reason']}: {int(row['rows'])}")
    print("")
    print(f"Evaluated rows output: {EVALUATED_ROWS_PATH}")
    print(f"Evaluation summary output: {SUMMARY_PATH}")
    print(f"Evaluation by run output: {EVALUATION_BY_RUN_PATH}")
    print("Paper-only. No trades/orders/private API.")


def print_grouped_summary(grouped):
    print("")
    print("Grouped summaries by run/threshold/regression")
    if len(grouped) == 0:
        print("- no groups")
        return
    for _, row in grouped.head(25).iterrows():
        sign_accuracy = row["sign_accuracy"]
        avg_return = row["average_realized_30s_strategy_return_bps"]
        print(
            f"- run_id={row['run_id']} "
            f"move_model={str(row.get('move_model_id', ''))[:18]} "
            f"inst_model={str(row.get('instability_model_id', ''))[:18]} "
            f"pinning={row.get('model_pinning_status', 'unknown')} "
            f"move={row['active_move_confidence_threshold']} "
            f"instability={row['active_instability_threshold']} "
            f"sides={row.get('active_allowed_sides', 'both')} "
            f"regression={row['regression_enabled']} "
            f"rows={int(row['total_rows'])} "
            f"eval={int(row['evaluated_rows'])} "
            f"signals={int(row['paper_signal_rows'])} "
            f"long={int(row['long_count'])} "
            f"short={int(row['short_count'])} "
            f"sign_acc={sign_accuracy:.2%}" if np.isfinite(sign_accuracy) else
            f"- run_id={row['run_id']} "
            f"move_model={str(row.get('move_model_id', ''))[:18]} "
            f"inst_model={str(row.get('instability_model_id', ''))[:18]} "
            f"pinning={row.get('model_pinning_status', 'unknown')} "
            f"move={row['active_move_confidence_threshold']} "
            f"instability={row['active_instability_threshold']} "
            f"sides={row.get('active_allowed_sides', 'both')} "
            f"regression={row['regression_enabled']} "
            f"rows={int(row['total_rows'])} "
            f"eval={int(row['evaluated_rows'])} "
            f"signals={int(row['paper_signal_rows'])} "
            f"long={int(row['long_count'])} "
            f"short={int(row['short_count'])} "
            f"sign_acc=n/a",
            end="",
        )
        print(
            f" avg_return={avg_return:.4f}bps" if np.isfinite(avg_return) else " avg_return=n/a",
            end="",
        )
        net_return = row.get("net_avg_strategy_return_bps", np.nan)
        total_cost = row.get("average_total_cost_bps", np.nan)
        print(
            f" net={net_return:.4f}bps" if np.isfinite(net_return) else " net=n/a",
            end="",
        )
        print(
            f" cost={total_cost:.4f}bps" if np.isfinite(total_cost) else " cost=n/a",
            end="",
        )
        net_positive_rate = row.get("net_positive_rate", np.nan)
        print(
            f" net_pos={net_positive_rate:.2%}" if np.isfinite(net_positive_rate) else " net_pos=n/a",
            end="",
        )
        print(
            f" stale={int(row['stale_rows'])} "
            f"missing_future={int(row['missing_future_rows'])} "
            f"required_schema_mismatch={int(row['required_schema_mismatch_rows'])} "
            f"optional_direction_mismatch={int(row.get('optional_direction_schema_mismatch_rows', 0))} "
            f"optional_regression_mismatch={int(row.get('optional_regression_schema_mismatch_rows', 0))}"
        )


def main():
    predictions = normalize_predictions(read_csv(PREDICTIONS_PATH))
    snapshots = normalize_snapshots(read_csv(SNAPSHOT_PATH))
    evaluated, attach_stats = attach_realized_outcomes(predictions, snapshots)
    no_trade_reasons = count_no_trade_reasons(predictions[predictions["final_paper_signal"].astype(str).str.lower() == "no_trade"]) if len(predictions) and "final_paper_signal" in predictions.columns else pd.DataFrame()
    summary = summarize(predictions, evaluated, attach_stats)
    grouped_summary = summarize_by_run(predictions, evaluated)
    sensitivity = threshold_sensitivity(predictions, evaluated)
    side_regime = side_regime_diagnostics(predictions, evaluated)

    if len(evaluated):
        atomic_write_csv(evaluated, EVALUATED_ROWS_PATH)
    atomic_write_csv(pd.DataFrame([summary]), SUMMARY_PATH)
    atomic_write_csv(grouped_summary, EVALUATION_BY_RUN_PATH)
    atomic_write_csv(sensitivity, THRESHOLD_SENSITIVITY_PATH)
    atomic_write_csv(side_regime, SIDE_REGIME_DIAGNOSTICS_PATH)
    print_summary(summary, no_trade_reasons)
    print_grouped_summary(grouped_summary)
    print_threshold_sensitivity(sensitivity)
    print("")
    print(f"Side/regime diagnostics output: {SIDE_REGIME_DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
