import os
import shutil
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from build_1s_order_flow_training_rows import CLASS_NAMES, build_feature_row
from microstructure_model_utils import feature_schema_hash, infer_snapshot_step_seconds, load_snapshot_rows
from train_1s_order_flow_model import load_model, predict_with_artifact, threshold_decode_class_probabilities


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
VENUE_TAG = PRIMARY_VENUE or "legacy"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
FLOW_1S_LOOP_SECONDS = float(os.getenv("FLOW_1S_LOOP_SECONDS", "1"))
FLOW_1S_MODEL_SELECTION = os.getenv("FLOW_1S_MODEL_SELECTION", "latest_candidate").strip().lower()
FLOW_1S_MODEL_PATH = os.getenv("FLOW_1S_MODEL_PATH", "").strip()
RUN_ONCE = os.getenv("RUN_ONCE", "false").strip().lower() in {"1", "true", "yes", "y"}
FLOW_1S_ON_SCHEMA_MISMATCH = os.getenv("FLOW_1S_ON_SCHEMA_MISMATCH", "fail").strip().lower()
FLOW_1S_MAX_CANDIDATE_ROWS_PER_TICK = int(os.getenv("FLOW_1S_MAX_CANDIDATE_ROWS_PER_TICK", "5000"))
VALID_MODEL_SELECTIONS = {"latest_candidate", "active_only"}
VALID_SCHEMA_ACTIONS = {"fail", "archive"}

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
SNAPSHOT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_10s_flow.csv"
PREDICTION_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_1s_order_flow_predictions.csv"
DIAGNOSTICS_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_1s_order_flow_loop_diagnostics.json"
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "active" / SYMBOL / "order_flow_1s" / VENUE_TAG / "model.json"
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "order_flow_1s" / VENUE_TAG

OUTPUT_COLUMNS = [
    "timestamp",
    "time",
    "symbol",
    "primary_venue",
    "class_target_column",
    "target_horizon_seconds",
    "model_target_horizon_seconds",
    "prob_sell_dominant_1s",
    "prob_neutral_1s",
    "prob_buy_dominant_1s",
    "decoded_flow_class_1s",
    "pred_market_buy_volume_1s",
    "pred_market_sell_volume_1s",
    "pred_market_pressure_1s",
    "pred_pressure_magnitude_1s",
    "pred_trade_count_1s",
    "buy_burst_prob_1s",
    "sell_burst_prob_1s",
    "model_id",
    "feature_schema_hash",
    "trained_until_timestamp",
]
OPTIONAL_OUTPUT_COLUMNS = {
    "class_target_column",
    "target_horizon_seconds",
    "model_target_horizon_seconds",
}


def newest_candidate_model_path():
    if not CANDIDATE_ROOT.exists():
        return None
    candidates = sorted(CANDIDATE_ROOT.glob("*/model.json"))
    return candidates[-1] if candidates else None


def path_is_allowed_for_symbol_and_venue(path):
    path = Path(path).resolve()
    if path == ACTIVE_MODEL_PATH.resolve():
        return True
    try:
        path.relative_to(CANDIDATE_ROOT.resolve())
        return path.name == "model.json"
    except ValueError:
        return False


def choose_model_path():
    if FLOW_1S_MODEL_PATH:
        path = Path(FLOW_1S_MODEL_PATH)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        return path if path_is_allowed_for_symbol_and_venue(path) else None

    if FLOW_1S_MODEL_SELECTION not in VALID_MODEL_SELECTIONS:
        print(
            "Invalid FLOW_1S_MODEL_SELECTION="
            f"{FLOW_1S_MODEL_SELECTION!r}; expected latest_candidate or active_only.",
            flush=True,
        )
        return None

    if FLOW_1S_MODEL_SELECTION == "active_only":
        return ACTIVE_MODEL_PATH if ACTIVE_MODEL_PATH.exists() else None

    candidate = newest_candidate_model_path()
    if candidate is not None:
        return candidate
    return ACTIVE_MODEL_PATH if ACTIVE_MODEL_PATH.exists() else None


def model_symbol(artifact):
    return str(artifact.get("model_symbol", artifact.get("symbol", ""))).strip().upper()


def model_venue(artifact):
    return str(artifact.get("primary_venue", "legacy")).strip().lower() or "legacy"


def is_optional_context_feature(column):
    return isinstance(column, str) and column.startswith("feature_context_")


def optional_context_default(column):
    if str(column).endswith("_context_age_ms"):
        return -1.0
    return 0.0


def invalid_current_book_columns(row):
    positive_columns = [
        "mid_price",
        "best_bid",
        "best_ask",
        "bid_depth_10bps",
        "ask_depth_10bps",
        "bid_depth_25bps",
        "ask_depth_25bps",
    ]
    finite_columns = [
        "spread_percent",
        "order_book_imbalance_10bps",
        "order_book_imbalance_25bps",
    ]
    invalid = []
    values = {}
    for column in positive_columns:
        value = pd.to_numeric(pd.Series([row.get(column, np.nan)]), errors="coerce").iloc[0]
        values[column] = None if pd.isna(value) else float(value)
        if pd.isna(value) or not np.isfinite(value) or value <= 0:
            invalid.append(column)
    for column in finite_columns:
        value = pd.to_numeric(pd.Series([row.get(column, np.nan)]), errors="coerce").iloc[0]
        values[column] = None if pd.isna(value) else float(value)
        if pd.isna(value) or not np.isfinite(value):
            invalid.append(column)
    best_bid = values.get("best_bid")
    best_ask = values.get("best_ask")
    if best_bid is not None and best_ask is not None and best_ask <= best_bid:
        invalid.append("best_ask<=best_bid")
    return invalid, values


def add_optional_defaults(frame, model_columns):
    frame = frame.copy()
    for column in model_columns:
        if column not in frame.columns and is_optional_context_feature(column):
            frame[column] = optional_context_default(column)
        elif column in frame.columns and is_optional_context_feature(column):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(
                optional_context_default(column)
            )
    return frame


def read_existing_prediction_state():
    if not PREDICTION_PATH.exists():
        return None, OUTPUT_COLUMNS, {"status": "ok", "missing": [], "extra": [], "archived_path": None, "message": ""}
    try:
        frame = pd.read_csv(PREDICTION_PATH)
    except EmptyDataError:
        return None, OUTPUT_COLUMNS, {"status": "ok", "missing": [], "extra": [], "archived_path": None, "message": ""}
    except Exception as error:
        return None, OUTPUT_COLUMNS, {
            "status": "failed",
            "missing": [],
            "extra": [],
            "archived_path": None,
            "message": f"could not read existing predictions: {error}",
        }

    existing_columns = list(frame.columns)
    missing_required = [
        column for column in OUTPUT_COLUMNS
        if column not in existing_columns and column not in OPTIONAL_OUTPUT_COLUMNS
    ]
    extra_columns = [column for column in existing_columns if column not in OUTPUT_COLUMNS]
    if missing_required or extra_columns:
        schema_info = handle_schema_mismatch(existing_columns, missing_required, extra_columns)
        if schema_info["status"] != "archived":
            return None, existing_columns, schema_info
        return None, OUTPUT_COLUMNS, schema_info

    if len(frame) == 0 or "timestamp" not in frame.columns:
        return None, existing_columns, {"status": "ok", "missing": [], "extra": [], "archived_path": None, "message": ""}
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce").dropna()
    if len(timestamps) == 0:
        return None, existing_columns, {"status": "ok", "missing": [], "extra": [], "archived_path": None, "message": ""}
    return int(timestamps.max()), existing_columns, {"status": "ok", "missing": [], "extra": [], "archived_path": None, "message": ""}


def current_utc_tag():
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def write_empty_prediction_file():
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(PREDICTION_PATH, index=False)


def handle_schema_mismatch(existing_columns, missing_required, extra_columns):
    if FLOW_1S_ON_SCHEMA_MISMATCH not in VALID_SCHEMA_ACTIONS:
        return {
            "status": "failed",
            "missing": missing_required,
            "extra": extra_columns,
            "archived_path": None,
            "message": (
                "Invalid FLOW_1S_ON_SCHEMA_MISMATCH="
                f"{FLOW_1S_ON_SCHEMA_MISMATCH!r}; expected fail or archive."
            ),
        }

    message = (
        "existing prediction file has incompatible columns; "
        f"missing={missing_required}, extra={extra_columns}, "
        f"expected={OUTPUT_COLUMNS}, existing={existing_columns}"
    )
    if FLOW_1S_ON_SCHEMA_MISMATCH == "fail":
        return {
            "status": "failed",
            "missing": missing_required,
            "extra": extra_columns,
            "archived_path": None,
            "message": message,
        }

    archive_path = PREDICTION_PATH.with_name(
        f"{PREDICTION_PATH.name}.old_schema.{current_utc_tag()}.csv"
    )
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(PREDICTION_PATH), str(archive_path))
    write_empty_prediction_file()
    return {
        "status": "archived",
        "missing": missing_required,
        "extra": extra_columns,
        "archived_path": str(archive_path),
        "message": message,
    }


def output_columns_for_existing_file():
    if not PREDICTION_PATH.exists():
        return OUTPUT_COLUMNS, None
    try:
        header = pd.read_csv(PREDICTION_PATH, nrows=0)
    except EmptyDataError:
        return OUTPUT_COLUMNS, None
    existing_columns = list(header.columns)
    missing_required = [
        column for column in OUTPUT_COLUMNS
        if column not in existing_columns and column not in OPTIONAL_OUTPUT_COLUMNS
    ]
    extra_columns = [column for column in existing_columns if column not in OUTPUT_COLUMNS]
    if missing_required or extra_columns:
        return existing_columns, (
            "existing prediction file has incompatible columns; "
            f"missing={missing_required}, extra={extra_columns}. Not appending mixed-schema rows."
        )
    return existing_columns, None


def safe_latest_snapshot_timestamp():
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        frame = pd.read_csv(SNAPSHOT_PATH, usecols=["timestamp"])
    except Exception:
        return None
    if len(frame) == 0:
        return None
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce").dropna()
    return int(timestamps.max()) if len(timestamps) else None


def write_diagnostics(payload):
    payload = dict(payload)
    payload.setdefault("symbol", SYMBOL)
    payload.setdefault("primary_venue", VENUE_TAG)
    payload.setdefault("snapshot_path", str(SNAPSHOT_PATH))
    payload.setdefault("prediction_path", str(PREDICTION_PATH))
    payload["updated_at_ms"] = int(time.time() * 1000)
    payload["updated_at"] = (
        dt.datetime.now(dt.UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    DIAGNOSTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DIAGNOSTICS_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json_dumps(payload), encoding="utf-8")
    tmp_path.replace(DIAGNOSTICS_PATH)


def json_dumps(payload):
    import json

    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def zero_write_reason(skipped, feature_rows_count=0, append_error=None):
    if append_error:
        return f"append blocked: {append_error}"
    candidate_rows = int(skipped.get("candidate rows newer than latest prediction", 0))
    if candidate_rows == 0:
        return "no new snapshot rows after latest prediction timestamp"
    if feature_rows_count == 0:
        ranked = [
            ("invalid current book row", "all/most new rows have invalid book fields"),
            ("missing 60s history", "not enough 60s lookback history yet"),
            ("missing required features", "model feature columns are missing"),
            ("non-finite model features", "model feature values are non-finite"),
            ("feature not ready", "feature builder rejected all new rows"),
        ]
        for key, message in ranked:
            if int(skipped.get(key, 0)) > 0:
                return message
        return "no feature-ready rows were produced"
    return "prediction output rows were empty"


def print_blocked_tick(
    reason,
    latest_prediction_timestamp,
    schema_status="ok",
    schema_info=None,
    skipped_reason="blocked",
):
    schema_info = schema_info or {}
    print("1s order-flow live loop", flush=True)
    print(f"- SYMBOL: {SYMBOL}", flush=True)
    print(f"- PRIMARY_VENUE: {VENUE_TAG}", flush=True)
    print(f"- Input snapshot path: {SNAPSHOT_PATH}", flush=True)
    print(f"- Prediction output path: {PREDICTION_PATH}", flush=True)
    print(f"- Latest snapshot timestamp: {safe_latest_snapshot_timestamp()}", flush=True)
    print(f"- Latest prediction timestamp: {latest_prediction_timestamp}", flush=True)
    print("- Candidate rows newer than latest prediction: 0", flush=True)
    print("- Newly written predictions: 0", flush=True)
    print(f"- Schema status: {schema_status}", flush=True)
    if schema_info:
        print(f"- Schema missing columns: {schema_info.get('missing', [])}", flush=True)
        print(f"- Schema extra columns: {schema_info.get('extra', [])}", flush=True)
    print(f"- Blocking reason: {reason}", flush=True)
    print("- Skipped rows by reason:", flush=True)
    print(f"  - {skipped_reason}: 1", flush=True)
    print("- No trades/orders.", flush=True)
    write_diagnostics(
        {
            "status": "blocked",
            "blocking_reason": reason,
            "latest_snapshot_timestamp": safe_latest_snapshot_timestamp(),
            "latest_prediction_timestamp": latest_prediction_timestamp,
            "candidate_rows_newer_than_latest_prediction": 0,
            "feature_ready_skipped_rows": 0,
            "non_finite_feature_skipped_rows": 0,
            "invalid_book_skipped_rows": 0,
            "newly_written_predictions": 0,
            "schema_status": schema_status,
            "skipped_rows_by_reason": {skipped_reason: 1},
        }
    )


def current_feature_columns(frame):
    return sorted(
        column
        for column in frame.columns
        if isinstance(column, str)
        and column.startswith("feature_")
        and column != "feature_ready"
    )


def build_new_feature_rows(snapshots, latest_prediction_timestamp, artifact):
    snapshot_step_seconds = infer_snapshot_step_seconds(snapshots)
    model_columns = list(artifact.get("feature_columns", []))
    model_set = set(model_columns)
    rows = []
    skipped = {
        "already predicted": 0,
        "feature not ready": 0,
        "missing required features": 0,
        "non-finite model features": 0,
        "invalid current book row": 0,
        "candidate rows newer than latest prediction": 0,
    }
    debug_details = {
        "latest_invalid_book_timestamp": None,
        "latest_invalid_book_columns": [],
        "latest_invalid_book_values": {},
        "latest_missing_required_feature_columns": [],
        "latest_non_finite_feature_timestamp": None,
        "latest_non_finite_feature_columns": [],
    }

    start_index = 0
    if latest_prediction_timestamp is not None and len(snapshots):
        timestamps = pd.to_numeric(snapshots["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        start_index = int(np.searchsorted(timestamps, latest_prediction_timestamp, side="right"))
        skipped["already predicted"] = max(0, start_index)
    end_index = len(snapshots)
    if FLOW_1S_MAX_CANDIDATE_ROWS_PER_TICK > 0:
        end_index = min(end_index, start_index + FLOW_1S_MAX_CANDIDATE_ROWS_PER_TICK)

    for index in range(start_index, end_index):
        timestamp = int(snapshots.loc[index, "timestamp"])
        skipped["candidate rows newer than latest prediction"] += 1

        feature_row, reason = build_feature_row(snapshots, index, snapshot_step_seconds)
        if feature_row is None:
            skipped["feature not ready"] += 1
            skipped[reason or "unknown feature reason"] = skipped.get(reason or "unknown feature reason", 0) + 1
            if reason == "invalid current book row":
                skipped["invalid current book row"] += 1
                invalid_columns, invalid_values = invalid_current_book_columns(snapshots.loc[index])
                debug_details["latest_invalid_book_timestamp"] = timestamp
                debug_details["latest_invalid_book_columns"] = invalid_columns
                debug_details["latest_invalid_book_values"] = invalid_values
            continue

        frame = pd.DataFrame([feature_row])
        frame = add_optional_defaults(frame, model_columns)
        available_columns = set(current_feature_columns(frame))
        missing_required = [
            column
            for column in model_columns
            if column not in available_columns and not is_optional_context_feature(column)
        ]
        if missing_required:
            skipped["missing required features"] += 1
            debug_details["latest_missing_required_feature_columns"] = missing_required[:50]
            continue

        model_input = frame[model_columns].replace([np.inf, -np.inf], np.nan)
        if model_input.isna().any().any():
            skipped["non-finite model features"] += 1
            non_finite_columns = [
                column for column in model_input.columns
                if model_input[column].isna().any()
            ]
            debug_details["latest_non_finite_feature_timestamp"] = timestamp
            debug_details["latest_non_finite_feature_columns"] = non_finite_columns[:50]
            continue

        rows.append(frame)

    if not rows:
        return pd.DataFrame(), skipped, snapshot_step_seconds, debug_details
    feature_frame = pd.concat(rows, ignore_index=True)
    # Guard against a stale artifact that lists duplicate columns.
    feature_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated()]
    unused_current = sorted(set(current_feature_columns(feature_frame)) - model_set)
    if unused_current:
        skipped["extra current features ignored"] = len(unused_current)
    return feature_frame, skipped, snapshot_step_seconds, debug_details


def regression_value(artifact, regression_row, target_name, log_target_name=None):
    columns = list(artifact.get("regression_target_columns", []))
    if log_target_name and log_target_name in columns:
        value = float(regression_row[columns.index(log_target_name)])
        return max(0.0, float(np.expm1(value)))
    if target_name in columns:
        value = float(regression_row[columns.index(target_name)])
        if target_name in {
            "future_market_buy_volume_1s",
            "future_market_sell_volume_1s",
            "future_trade_count_1s",
        }:
            return max(0.0, value)
        return value
    return np.nan


def artifact_target_horizon_seconds(artifact):
    return int(artifact.get("target_horizon_seconds", 1) or 1)


def horizon_target_names(artifact):
    horizon = artifact_target_horizon_seconds(artifact)
    suffix = f"{horizon}s"
    return {
        "horizon": horizon,
        "buy": f"future_market_buy_volume_{suffix}",
        "buy_log": f"future_log_market_buy_volume_{suffix}",
        "sell": f"future_market_sell_volume_{suffix}",
        "sell_log": f"future_log_market_sell_volume_{suffix}",
        "pressure": f"future_market_pressure_{suffix}",
        "trade_count": f"future_trade_count_{suffix}",
        "trade_count_log": f"future_log_trade_count_{suffix}",
    }


def pressure_value(artifact, regression_row):
    columns = list(artifact.get("regression_target_columns", []))
    target_names = horizon_target_names(artifact)
    if target_names["pressure"] in columns:
        return float(regression_row[columns.index(target_names["pressure"])])
    if "future_market_pressure_1s" in columns:
        return float(regression_row[columns.index("future_market_pressure_1s")])
    return np.nan


def prediction_rows_from_outputs(artifact, feature_frame, class_prob, burst_prob, regression):
    threshold_decode = artifact.get("threshold_decode", {})
    directional_min_prob = float(threshold_decode.get("directional_min_prob", 0.45))
    directional_neutral_margin = float(threshold_decode.get("directional_neutral_margin", 0.05))
    decoded = threshold_decode_class_probabilities(
        class_prob,
        min_prob=directional_min_prob,
        neutral_margin=directional_neutral_margin,
    )

    rows = []
    target_names = horizon_target_names(artifact)
    class_target_column = str(artifact.get("class_target_column", "next_1s_flow_class"))
    for index, feature_row in feature_frame.reset_index(drop=True).iterrows():
        pressure = pressure_value(artifact, regression[index])
        decoded_class = CLASS_NAMES.get(int(decoded[index]), str(int(decoded[index])))
        rows.append(
            {
                "timestamp": int(feature_row["timestamp"]),
                "time": feature_row.get("time", ""),
                "symbol": SYMBOL,
                "primary_venue": VENUE_TAG,
                "class_target_column": class_target_column,
                "target_horizon_seconds": target_names["horizon"],
                "model_target_horizon_seconds": target_names["horizon"],
                "prob_sell_dominant_1s": float(class_prob[index, 0]),
                "prob_neutral_1s": float(class_prob[index, 1]),
                "prob_buy_dominant_1s": float(class_prob[index, 2]),
                "decoded_flow_class_1s": decoded_class,
                "pred_market_buy_volume_1s": regression_value(
                    artifact,
                    regression[index],
                    target_names["buy"],
                    target_names["buy_log"],
                ),
                "pred_market_sell_volume_1s": regression_value(
                    artifact,
                    regression[index],
                    target_names["sell"],
                    target_names["sell_log"],
                ),
                "pred_market_pressure_1s": pressure,
                "pred_pressure_magnitude_1s": abs(pressure) if np.isfinite(pressure) else np.nan,
                "pred_trade_count_1s": regression_value(
                    artifact,
                    regression[index],
                    target_names["trade_count"],
                    target_names["trade_count_log"],
                ),
                "buy_burst_prob_1s": float(burst_prob[index, 0]),
                "sell_burst_prob_1s": float(burst_prob[index, 1]),
                "model_id": artifact.get("model_id", ""),
                "feature_schema_hash": artifact.get("feature_schema_hash", ""),
                "trained_until_timestamp": artifact.get("trained_until_timestamp", ""),
            }
        )
    return pd.DataFrame(rows)


def append_predictions(rows):
    if len(rows) == 0:
        return False, None
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_columns, error = output_columns_for_existing_file()
    if error:
        return False, error
    rows = rows.copy()
    for column in output_columns:
        if column not in rows.columns:
            rows[column] = ""
    rows = rows[output_columns]
    write_header = not PREDICTION_PATH.exists() or PREDICTION_PATH.stat().st_size == 0
    rows.to_csv(PREDICTION_PATH, mode="a", header=write_header, index=False)
    return True, None


def load_validated_artifact():
    model_path = choose_model_path()
    if model_path is None:
        return None, None, f"No 1s order-flow model found for {SYMBOL} on {VENUE_TAG}"
    artifact = load_model(model_path)
    if artifact is None:
        return None, model_path, f"Could not load model: {model_path}"
    loaded_symbol = model_symbol(artifact)
    loaded_venue = model_venue(artifact)
    if loaded_symbol != SYMBOL:
        return None, model_path, (
            f"model symbol mismatch: requested={SYMBOL} model={loaded_symbol or 'missing'}"
        )
    if loaded_venue != VENUE_TAG:
        return None, model_path, (
            f"model venue mismatch: requested={VENUE_TAG} model={loaded_venue or 'missing'}"
        )
    saved_columns_hash = feature_schema_hash(list(artifact.get("feature_columns", [])))
    artifact_hash = str(artifact.get("feature_schema_hash", ""))
    if artifact_hash != saved_columns_hash:
        return None, model_path, (
            f"feature schema hash mismatch: artifact={artifact_hash or 'missing'} "
            f"columns={saved_columns_hash}"
        )
    return artifact, model_path, None


def run_once():
    latest_prediction_timestamp, _, schema_info = read_existing_prediction_state()
    schema_status = schema_info.get("status", "ok")
    if schema_status == "failed":
        print_blocked_tick(
            schema_info.get("message", ""),
            latest_prediction_timestamp,
            schema_status="failed",
            schema_info=schema_info,
            skipped_reason="schema mismatch",
        )
        return
    if schema_status == "archived":
        print("LOUD WARNING: archived incompatible 1s order-flow prediction CSV.", flush=True)
        print(f"- Archive path: {schema_info.get('archived_path')}", flush=True)
        print(f"- Missing columns: {schema_info.get('missing', [])}", flush=True)
        print(f"- Extra columns: {schema_info.get('extra', [])}", flush=True)
        print("- A new prediction CSV with the current expected schema was created.", flush=True)

    artifact, model_path, model_error = load_validated_artifact()
    if model_error:
        print_blocked_tick(
            model_error,
            latest_prediction_timestamp,
            schema_status=schema_status,
            schema_info=schema_info,
            skipped_reason="missing/invalid model",
        )
        return

    try:
        snapshots = load_snapshot_rows(SNAPSHOT_PATH)
    except FileNotFoundError:
        print_blocked_tick(
            f"missing snapshot file {SNAPSHOT_PATH}",
            latest_prediction_timestamp,
            schema_status=schema_status,
            schema_info=schema_info,
            skipped_reason="missing snapshot file",
        )
        return
    except ValueError as error:
        print_blocked_tick(
            f"invalid snapshot file: {error}",
            latest_prediction_timestamp,
            schema_status=schema_status,
            schema_info=schema_info,
            skipped_reason="invalid snapshot file",
        )
        return

    feature_frame, skipped, snapshot_step_seconds, debug_details = build_new_feature_rows(
        snapshots,
        latest_prediction_timestamp,
        artifact,
    )
    if len(feature_frame) == 0:
        reason = zero_write_reason(skipped, feature_rows_count=0)
        print("1s order-flow live loop", flush=True)
        print(f"- SYMBOL: {SYMBOL}", flush=True)
        print(f"- PRIMARY_VENUE: {VENUE_TAG}", flush=True)
        print(f"- Snapshot path: {SNAPSHOT_PATH}", flush=True)
        print(f"- Prediction path: {PREDICTION_PATH}", flush=True)
        print(f"- Input snapshot path: {SNAPSHOT_PATH}", flush=True)
        print(f"- Prediction output path: {PREDICTION_PATH}", flush=True)
        print(f"- Latest snapshot timestamp: {int(snapshots['timestamp'].max()) if len(snapshots) else None}", flush=True)
        print(f"- Latest prediction timestamp: {latest_prediction_timestamp}", flush=True)
        print(
            "- Candidate rows newer than latest prediction: "
            f"{skipped.get('candidate rows newer than latest prediction', 0)}",
            flush=True,
        )
        print(f"- Inferred snapshot step seconds: {snapshot_step_seconds:.3g}", flush=True)
        print("- Newly written predictions: 0", flush=True)
        print(f"- Blocking reason: {reason}", flush=True)
        print(f"- Schema status: {schema_status}", flush=True)
        print("- Skipped rows by reason:", flush=True)
        for reason, count in sorted(skipped.items()):
            print(f"  - {reason}: {count}", flush=True)
        if debug_details.get("latest_invalid_book_columns"):
            print(
                "- Latest rejected book row: "
                f"timestamp={debug_details.get('latest_invalid_book_timestamp')} "
                f"invalid_columns={debug_details.get('latest_invalid_book_columns')} "
                f"values={debug_details.get('latest_invalid_book_values')}",
                flush=True,
            )
        if debug_details.get("latest_non_finite_feature_columns"):
            print(
                "- Latest non-finite model feature row: "
                f"timestamp={debug_details.get('latest_non_finite_feature_timestamp')} "
                f"columns={debug_details.get('latest_non_finite_feature_columns')}",
                flush=True,
            )
        if debug_details.get("latest_missing_required_feature_columns"):
            print(
                "- Latest missing required model feature columns: "
                f"{debug_details.get('latest_missing_required_feature_columns')}",
                flush=True,
            )
        print("- No trades/orders.", flush=True)
        write_diagnostics(
            {
                "status": "blocked",
                "blocking_reason": zero_write_reason(skipped, feature_rows_count=0),
                "latest_snapshot_timestamp": int(snapshots["timestamp"].max()) if len(snapshots) else None,
                "latest_prediction_timestamp": latest_prediction_timestamp,
                "candidate_rows_newer_than_latest_prediction": int(skipped.get("candidate rows newer than latest prediction", 0)),
                "feature_ready_skipped_rows": int(skipped.get("feature not ready", 0)),
                "non_finite_feature_skipped_rows": int(skipped.get("non-finite model features", 0)),
                "invalid_book_skipped_rows": int(skipped.get("invalid current book row", 0)),
                "missing_required_feature_skipped_rows": int(skipped.get("missing required features", 0)),
                "candidate_feature_rows": 0,
                "newly_written_predictions": 0,
                "schema_status": schema_status,
                "skipped_rows_by_reason": skipped,
                "latest_invalid_book_timestamp": debug_details.get("latest_invalid_book_timestamp"),
                "latest_invalid_book_columns": debug_details.get("latest_invalid_book_columns", []),
                "latest_invalid_book_values": debug_details.get("latest_invalid_book_values", {}),
                "latest_non_finite_feature_timestamp": debug_details.get("latest_non_finite_feature_timestamp"),
                "latest_non_finite_feature_columns": debug_details.get("latest_non_finite_feature_columns", []),
                "latest_missing_required_feature_columns": debug_details.get("latest_missing_required_feature_columns", []),
                "model_id": artifact.get("model_id", ""),
                "model_path": str(model_path),
            }
        )
        return

    try:
        class_prob, burst_prob, regression = predict_with_artifact(artifact, feature_frame)
        output_rows = prediction_rows_from_outputs(
            artifact,
            feature_frame,
            class_prob,
            burst_prob,
            regression,
        )
    except Exception as error:
        reason = f"model prediction failed: {error}"
        print("1s order-flow live loop", flush=True)
        print(f"- SYMBOL: {SYMBOL}", flush=True)
        print(f"- PRIMARY_VENUE: {VENUE_TAG}", flush=True)
        print(f"- Latest snapshot timestamp: {int(snapshots['timestamp'].max()) if len(snapshots) else None}", flush=True)
        print(f"- Latest prediction timestamp: {latest_prediction_timestamp}", flush=True)
        print(f"- Candidate rows newer than latest prediction: {skipped.get('candidate rows newer than latest prediction', 0)}", flush=True)
        print(f"- Candidate feature rows: {len(feature_frame)}", flush=True)
        print("- Newly written predictions: 0", flush=True)
        print(f"- Blocking reason: {reason}", flush=True)
        print("- No trades/orders.", flush=True)
        write_diagnostics(
            {
                "status": "blocked",
                "blocking_reason": reason,
                "latest_snapshot_timestamp": int(snapshots["timestamp"].max()) if len(snapshots) else None,
                "latest_prediction_timestamp": latest_prediction_timestamp,
                "candidate_rows_newer_than_latest_prediction": int(skipped.get("candidate rows newer than latest prediction", 0)),
                "feature_ready_skipped_rows": int(skipped.get("feature not ready", 0)),
                "non_finite_feature_skipped_rows": int(skipped.get("non-finite model features", 0)),
                "invalid_book_skipped_rows": int(skipped.get("invalid current book row", 0)),
                "missing_required_feature_skipped_rows": int(skipped.get("missing required features", 0)),
                "candidate_feature_rows": int(len(feature_frame)),
                "newly_written_predictions": 0,
                "schema_status": schema_status,
                "skipped_rows_by_reason": skipped,
                "latest_invalid_book_timestamp": debug_details.get("latest_invalid_book_timestamp"),
                "latest_invalid_book_columns": debug_details.get("latest_invalid_book_columns", []),
                "latest_invalid_book_values": debug_details.get("latest_invalid_book_values", {}),
                "latest_non_finite_feature_timestamp": debug_details.get("latest_non_finite_feature_timestamp"),
                "latest_non_finite_feature_columns": debug_details.get("latest_non_finite_feature_columns", []),
                "latest_missing_required_feature_columns": debug_details.get("latest_missing_required_feature_columns", []),
                "model_id": artifact.get("model_id", ""),
                "model_path": str(model_path),
            }
        )
        return
    written, append_error = append_predictions(output_rows)
    if append_error:
        reason = zero_write_reason(skipped, feature_rows_count=len(feature_frame), append_error=append_error)
        print(f"1s order-flow live loop blocked: {append_error}", flush=True)
        print(f"- Newly written predictions: 0", flush=True)
        print(f"- Blocking reason: {reason}", flush=True)
        write_diagnostics(
            {
                "status": "blocked",
                "blocking_reason": reason,
                "latest_snapshot_timestamp": int(snapshots["timestamp"].max()) if len(snapshots) else None,
                "latest_prediction_timestamp": latest_prediction_timestamp,
                "candidate_rows_newer_than_latest_prediction": int(skipped.get("candidate rows newer than latest prediction", 0)),
                "feature_ready_skipped_rows": int(skipped.get("feature not ready", 0)),
                "non_finite_feature_skipped_rows": int(skipped.get("non-finite model features", 0)),
                "invalid_book_skipped_rows": int(skipped.get("invalid current book row", 0)),
                "missing_required_feature_skipped_rows": int(skipped.get("missing required features", 0)),
                "candidate_feature_rows": int(len(feature_frame)),
                "newly_written_predictions": 0,
                "schema_status": schema_status,
                "skipped_rows_by_reason": skipped,
                "latest_invalid_book_timestamp": debug_details.get("latest_invalid_book_timestamp"),
                "latest_invalid_book_columns": debug_details.get("latest_invalid_book_columns", []),
                "latest_invalid_book_values": debug_details.get("latest_invalid_book_values", {}),
                "latest_non_finite_feature_timestamp": debug_details.get("latest_non_finite_feature_timestamp"),
                "latest_non_finite_feature_columns": debug_details.get("latest_non_finite_feature_columns", []),
                "latest_missing_required_feature_columns": debug_details.get("latest_missing_required_feature_columns", []),
                "model_id": artifact.get("model_id", ""),
                "model_path": str(model_path),
            }
        )
        return
    if not written or len(output_rows) == 0:
        reason = zero_write_reason(skipped, feature_rows_count=len(feature_frame))
        print("- Newly written predictions: 0", flush=True)
        print(f"- Blocking reason: {reason}", flush=True)

    print("1s order-flow live loop", flush=True)
    print(f"- SYMBOL: {SYMBOL}", flush=True)
    print(f"- PRIMARY_VENUE: {VENUE_TAG}", flush=True)
    print(f"- FLOW_1S_MODEL_SELECTION: {FLOW_1S_MODEL_SELECTION}", flush=True)
    print(f"- Model path: {model_path}", flush=True)
    print(f"- Model id: {artifact.get('model_id', 'missing')}", flush=True)
    print(f"- Model class target: {artifact.get('class_target_column', 'next_1s_flow_class')}", flush=True)
    print(f"- Model target horizon seconds: {artifact_target_horizon_seconds(artifact)}", flush=True)
    print(f"- Snapshot path: {SNAPSHOT_PATH}", flush=True)
    print(f"- Prediction path: {PREDICTION_PATH}", flush=True)
    print(f"- Input snapshot path: {SNAPSHOT_PATH}", flush=True)
    print(f"- Prediction output path: {PREDICTION_PATH}", flush=True)
    print(f"- Snapshot rows: {len(snapshots)}", flush=True)
    print(f"- Latest snapshot timestamp: {int(snapshots['timestamp'].max()) if len(snapshots) else None}", flush=True)
    print(f"- Latest prediction timestamp: {latest_prediction_timestamp}", flush=True)
    print(
        "- Candidate rows newer than latest prediction: "
        f"{skipped.get('candidate rows newer than latest prediction', 0)}",
        flush=True,
    )
    print(f"- Inferred snapshot step seconds: {snapshot_step_seconds:.3g}", flush=True)
    print(f"- Candidate feature rows: {len(feature_frame)}", flush=True)
    print(f"- Newly written predictions: {len(output_rows) if written else 0}", flush=True)
    print(f"- Schema status: {schema_status}", flush=True)
    if len(output_rows):
        print(f"- First written timestamp: {int(output_rows['timestamp'].min())}", flush=True)
        print(f"- Last written timestamp: {int(output_rows['timestamp'].max())}", flush=True)
    print("- Skipped rows by reason:", flush=True)
    for reason, count in sorted(skipped.items()):
        print(f"  - {reason}: {count}", flush=True)
    if debug_details.get("latest_invalid_book_columns"):
        print(
            "- Latest rejected book row: "
            f"timestamp={debug_details.get('latest_invalid_book_timestamp')} "
            f"invalid_columns={debug_details.get('latest_invalid_book_columns')} "
            f"values={debug_details.get('latest_invalid_book_values')}",
            flush=True,
        )
    if debug_details.get("latest_non_finite_feature_columns"):
        print(
            "- Latest non-finite model feature row: "
            f"timestamp={debug_details.get('latest_non_finite_feature_timestamp')} "
            f"columns={debug_details.get('latest_non_finite_feature_columns')}",
            flush=True,
        )
    if debug_details.get("latest_missing_required_feature_columns"):
        print(
            "- Latest missing required model feature columns: "
            f"{debug_details.get('latest_missing_required_feature_columns')}",
            flush=True,
        )
    print("- No trades/orders.", flush=True)
    write_diagnostics(
        {
            "status": "ok" if written and len(output_rows) else "blocked",
            "blocking_reason": "" if written and len(output_rows) else zero_write_reason(skipped, feature_rows_count=len(feature_frame)),
            "latest_snapshot_timestamp": int(snapshots["timestamp"].max()) if len(snapshots) else None,
            "latest_prediction_timestamp": int(output_rows["timestamp"].max()) if written and len(output_rows) else latest_prediction_timestamp,
            "previous_latest_prediction_timestamp": latest_prediction_timestamp,
            "candidate_rows_newer_than_latest_prediction": int(skipped.get("candidate rows newer than latest prediction", 0)),
            "feature_ready_skipped_rows": int(skipped.get("feature not ready", 0)),
            "non_finite_feature_skipped_rows": int(skipped.get("non-finite model features", 0)),
            "invalid_book_skipped_rows": int(skipped.get("invalid current book row", 0)),
            "missing_required_feature_skipped_rows": int(skipped.get("missing required features", 0)),
            "candidate_feature_rows": int(len(feature_frame)),
            "newly_written_predictions": int(len(output_rows) if written else 0),
            "schema_status": schema_status,
            "skipped_rows_by_reason": skipped,
            "latest_invalid_book_timestamp": debug_details.get("latest_invalid_book_timestamp"),
            "latest_invalid_book_columns": debug_details.get("latest_invalid_book_columns", []),
            "latest_invalid_book_values": debug_details.get("latest_invalid_book_values", {}),
            "latest_non_finite_feature_timestamp": debug_details.get("latest_non_finite_feature_timestamp"),
            "latest_non_finite_feature_columns": debug_details.get("latest_non_finite_feature_columns", []),
            "latest_missing_required_feature_columns": debug_details.get("latest_missing_required_feature_columns", []),
            "first_written_timestamp": int(output_rows["timestamp"].min()) if written and len(output_rows) else None,
            "last_written_timestamp": int(output_rows["timestamp"].max()) if written and len(output_rows) else None,
            "model_id": artifact.get("model_id", ""),
            "model_path": str(model_path),
        }
    )


def main():
    print("Starting paper-only 1s order-flow prediction loop", flush=True)
    print(f"SYMBOL: {SYMBOL}", flush=True)
    print(f"PRIMARY_VENUE: {VENUE_TAG}", flush=True)
    print(f"SNAPSHOT_PATH: {SNAPSHOT_PATH}", flush=True)
    print(f"PREDICTION_PATH: {PREDICTION_PATH}", flush=True)
    print(f"FLOW_1S_LOOP_SECONDS: {FLOW_1S_LOOP_SECONDS}", flush=True)
    print(f"FLOW_1S_ON_SCHEMA_MISMATCH: {FLOW_1S_ON_SCHEMA_MISMATCH}", flush=True)
    print(f"FLOW_1S_MAX_CANDIDATE_ROWS_PER_TICK: {FLOW_1S_MAX_CANDIDATE_ROWS_PER_TICK}", flush=True)
    print("No trades/orders.", flush=True)
    while True:
        try:
            run_once()
        except Exception as error:
            reason = f"unhandled loop error: {error}"
            print("1s order-flow live loop", flush=True)
            print(f"- SYMBOL: {SYMBOL}", flush=True)
            print(f"- PRIMARY_VENUE: {VENUE_TAG}", flush=True)
            print("- Newly written predictions: 0", flush=True)
            print(f"- Blocking reason: {reason}", flush=True)
            print("- Loop will continue after sleep. No trades/orders.", flush=True)
            write_diagnostics(
                {
                    "status": "error",
                    "blocking_reason": reason,
                    "latest_snapshot_timestamp": safe_latest_snapshot_timestamp(),
                    "latest_prediction_timestamp": read_existing_prediction_state()[0],
                    "candidate_rows_newer_than_latest_prediction": 0,
                    "feature_ready_skipped_rows": 0,
                    "non_finite_feature_skipped_rows": 0,
                    "invalid_book_skipped_rows": 0,
                    "newly_written_predictions": 0,
                }
            )
        if RUN_ONCE:
            break
        time.sleep(max(0.1, FLOW_1S_LOOP_SECONDS))


if __name__ == "__main__":
    main()
