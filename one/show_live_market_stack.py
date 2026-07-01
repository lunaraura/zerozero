import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from microstructure_model_utils import (
    EVENT_TARGET_COLUMNS,
    REGRESSION_TARGET_COLUMNS,
    add_missing_optional_context_columns,
    build_latest_feature_only,
    fill_optional_context_feature_values,
    feature_schema_hash,
    get_micro_feature_columns,
    load_model,
    load_snapshot_rows,
    micro_schema_diagnostics,
    percent,
    predict_with_artifact,
    regression_sanity_report,
    event_saturation_report,
    validate_regression_scalers,
    required_micro_feature_columns,
)
from hierarchical_context import attach_hierarchical_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
PRIMARY_VENUE_FALLBACK = os.getenv("VENUE", os.getenv("VENUES", "")).split(",")[0].strip().lower()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", PRIMARY_VENUE_FALLBACK).strip().lower()
VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR

SYMBOLS = [
    value.strip().upper()
    for value in os.getenv("SYMBOLS", os.getenv("SYMBOL", "SOLUSDT")).split(",")
    if value.strip()
]
STACK_LOOP_SECONDS = int(os.getenv("STACK_LOOP_SECONDS", os.getenv("LOOP_SECONDS", "60")))
STACK_RUN_ONCE = os.getenv("STACK_RUN_ONCE", "true").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
MAX_REGIME_AGE_MS = int(os.getenv("MAX_REGIME_AGE_MS", str(60 * 60 * 1000)))
MAX_HTF_AGE_MS = int(os.getenv("MAX_HTF_AGE_MS", str(3 * 60 * 60 * 1000)))
MAX_3M_AGE_MS = int(os.getenv("MAX_3M_AGE_MS", str(10 * 60 * 1000)))
MAX_MICRO_AGE_MS = int(os.getenv("MAX_MICRO_AGE_MS", str(2 * 60 * 1000)))
MAX_FLOW_1S_AGE_MS = int(os.getenv("MAX_FLOW_1S_AGE_MS", str(2 * 60 * 1000)))
MICROSTRUCTURE_MODEL_PATH = os.getenv("MICROSTRUCTURE_MODEL_PATH", "").strip()
MICRO_MODEL_SELECTION = os.getenv("MICRO_MODEL_SELECTION", "latest_candidate").strip().lower()
MICRO_ALLOW_REJECTED_CANDIDATES = os.getenv("MICRO_ALLOW_REJECTED_CANDIDATES", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
MICRO_EVENT_PROB_TEMPERATURE = float(os.getenv("MICRO_EVENT_PROB_TEMPERATURE", "1.0"))


def as_float(value, default=np.nan):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def format_percent(value):
    return "unavailable" if pd.isna(value) else percent(float(value))


def format_number(value, decimals=4):
    return "unavailable" if pd.isna(value) else f"{float(value):.{decimals}f}"


def load_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    string_columns = {
        "time",
        "symbol",
        "primary_venue",
        "decoded_flow_class_1s",
        "model_id",
        "feature_schema_hash",
    }
    for column in frame.columns:
        if column not in string_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("timestamp" if "timestamp" in frame.columns else frame.columns[0]).reset_index(drop=True)


def latest_timestamp_from_csv(path, timestamp_candidates=("timestamp", "close_timestamp")):
    path = Path(path)
    if not path.exists():
        return None, "missing file"
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return None, "empty"
    except Exception as error:
        return None, f"read error: {error}"
    if len(frame) == 0:
        return None, "empty file"
    timestamp_column = next((column for column in timestamp_candidates if column in frame.columns), None)
    if timestamp_column is None:
        return None, "no timestamp column"
    values = pd.to_numeric(frame[timestamp_column], errors="coerce").dropna()
    if len(values) == 0:
        return None, "no numeric timestamps"
    timestamp = int(values.max())
    if timestamp < 10_000_000_000:
        timestamp *= 1000
    return timestamp, "ok"


def age_status(timestamp, max_age_ms):
    if timestamp is None:
        return "stale"
    age_ms = int(time.time() * 1000) - int(timestamp)
    return "stale" if age_ms > max_age_ms else "fresh"


def print_freshness_line(label, path, max_age_ms, timestamp_candidates=("timestamp", "close_timestamp")):
    timestamp, status = latest_timestamp_from_csv(path, timestamp_candidates)
    if timestamp is None:
        print(f"- {label}: unavailable ({status}) path={path}")
        return
    age_ms = int(time.time() * 1000) - int(timestamp)
    stale_flag = "STALE" if age_ms > max_age_ms else "fresh"
    print(
        f"- {label}: timestamp={timestamp}, age={age_ms / 1000:.1f}s, "
        f"status={stale_flag}, path={path}"
    )


def realtime_1m_path(symbol):
    return VENUE_OUTPUT_DIR / f"{symbol}_1m_flow.csv"


def read_realtime_1m_timestamps(symbol):
    path = realtime_1m_path(symbol)
    if not path.exists():
        return pd.DataFrame(), f"missing file: realtime 1m input {path}"
    try:
        frame = pd.read_csv(path, usecols=["timestamp"])
    except EmptyDataError:
        return pd.DataFrame(), f"empty file: realtime 1m input {path}"
    except ValueError:
        return pd.DataFrame(), f"realtime 1m input has no timestamp column: {path}"
    except Exception as error:
        return pd.DataFrame(), f"could not read realtime 1m input {path}: {error}"
    if len(frame) == 0:
        return pd.DataFrame(), f"empty file: realtime 1m input {path}"
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    if len(frame) == 0:
        return pd.DataFrame(), f"realtime 1m input has no numeric timestamps: {path}"
    frame["timestamp"] = np.where(
        frame["timestamp"] < 10_000_000_000,
        frame["timestamp"] * 1000,
        frame["timestamp"],
    ).astype(np.int64)
    return frame.reset_index(drop=True), None


def completed_interval_count(timestamps, timeframe_minutes):
    if len(timestamps) == 0:
        return 0
    interval_ms = timeframe_minutes * 60 * 1000
    latest_timestamp = int(timestamps["timestamp"].max())
    latest_closed_open = ((latest_timestamp + 60_000) // interval_ms) * interval_ms - interval_ms
    working = timestamps.copy()
    working["bucket"] = (working["timestamp"] // interval_ms) * interval_ms
    working = working[working["bucket"] <= latest_closed_open]
    if len(working) == 0:
        return 0
    counts = working.groupby("bucket")["timestamp"].count()
    return int((counts == timeframe_minutes).sum())


def regime_unavailable_reason(symbol, timeframe, output_path):
    minutes = int(str(timeframe).replace("m", ""))
    timestamps, input_reason = read_realtime_1m_timestamps(symbol)
    if input_reason:
        return f"missing file: {output_path}; {input_reason}"
    completed = completed_interval_count(timestamps, minutes)
    minimum_completed_for_indicators = 50
    minimum_rows_for_one_candle = minutes
    if len(timestamps) < minimum_rows_for_one_candle:
        return (
            f"insufficient 1m rows: have {len(timestamps)}, "
            f"need at least {minimum_rows_for_one_candle} for one {timeframe} candle"
        )
    if completed == 0:
        return f"no completed candle: no complete {timeframe} candle from {len(timestamps)} realtime 1m rows"
    if completed < minimum_completed_for_indicators:
        return (
            "insufficient 1m rows for regime indicators: "
            f"have {completed} completed {timeframe} candles, need about {minimum_completed_for_indicators}"
        )
    return f"missing file: {output_path}; run npm run regime-update or enable ENABLE_REGIME_UPDATERS=true"


def htf_unavailable_reason(symbol, output_path):
    timestamps, input_reason = read_realtime_1m_timestamps(symbol)
    if input_reason:
        return f"missing file: {output_path}; {input_reason}"
    completed = completed_interval_count(timestamps, 60)
    minimum_completed_for_indicators = 50
    if len(timestamps) < 60:
        return f"insufficient 1m rows: have {len(timestamps)}, need at least 60 for one hourly candle"
    if completed == 0:
        return f"no completed candle: no complete hourly candle from {len(timestamps)} realtime 1m rows"
    if completed < minimum_completed_for_indicators:
        return (
            "insufficient 1m rows for HTF indicators: "
            f"have {completed} completed hourly candles, need about {minimum_completed_for_indicators}"
        )
    return f"missing file: {output_path}; run npm run htf-update or enable ENABLE_HTF_UPDATERS=true"


def print_file_freshness(symbol):
    primary_dir = VENUE_OUTPUT_DIR
    print("File freshness")
    print_freshness_line(
        "primary venue 10s snapshot",
        primary_dir / f"{symbol}_10s_flow.csv",
        MAX_MICRO_AGE_MS,
    )
    print_freshness_line(
        "binanceus 10s snapshot",
        OUTPUT_DIR / "binanceus" / f"{symbol}_10s_flow.csv",
        MAX_MICRO_AGE_MS,
    )
    print_freshness_line(
        "kraken 10s snapshot",
        OUTPUT_DIR / "kraken" / f"{symbol}_10s_flow.csv",
        MAX_MICRO_AGE_MS,
    )
    print_freshness_line(
        "primary venue 1m flow",
        primary_dir / f"{symbol}_1m_flow.csv",
        3 * 60 * 1000,
    )
    print_freshness_line(
        "live 1m features",
        primary_dir / f"{symbol}_1m_flow_features.csv",
        3 * 60 * 1000,
    )
    print_freshness_line(
        "3m predictions",
        PROJECT_ROOT / "data" / "live_predictions" / f"{symbol}_live_3m_predictions.csv",
        MAX_3M_AGE_MS,
    )
    print_freshness_line(
        "15m regime",
        PROJECT_ROOT / "data" / f"{symbol}_15m_regime_features.csv",
        MAX_REGIME_AGE_MS,
        timestamp_candidates=("close_timestamp", "timestamp"),
    )
    print_freshness_line(
        "30m regime",
        PROJECT_ROOT / "data" / f"{symbol}_30m_regime_features.csv",
        MAX_REGIME_AGE_MS,
        timestamp_candidates=("close_timestamp", "timestamp"),
    )
    print_freshness_line(
        "HTF context",
        PROJECT_ROOT / "data" / f"{symbol}_htf_context_features.csv",
        MAX_HTF_AGE_MS,
        timestamp_candidates=("close_timestamp", "timestamp"),
    )
    print_freshness_line(
        "10s micro predictions",
        primary_dir / f"{symbol}_10s_microstructure_predictions.csv",
        MAX_MICRO_AGE_MS,
    )
    print_freshness_line(
        "1s order-flow predictions",
        primary_dir / f"{symbol}_1s_order_flow_predictions.csv",
        MAX_FLOW_1S_AGE_MS,
    )


def print_micro_1s_context_health(symbol):
    path = VENUE_OUTPUT_DIR / f"{symbol}_10s_microstructure_training_rows.csv"
    frame = load_csv(path)
    if len(frame) == 0:
        print(f"- 10s micro rows can attach 1s context: unavailable (missing/empty) path={path}")
        return
    required_columns = {
        "timestamp",
        "feature_context_flow_1s_context_available",
        "feature_context_flow_1s_context_age_ms",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        print(
            "- 10s micro rows can attach 1s context: unavailable "
            f"(missing columns: {missing}) path={path}"
        )
        return
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if len(frame) == 0:
        print(f"- 10s micro rows can attach 1s context: unavailable (no timestamped rows) path={path}")
        return
    latest = frame.iloc[-1]
    availability = pd.to_numeric(
        frame["feature_context_flow_1s_context_available"],
        errors="coerce",
    ).fillna(0.0)
    available_rows = int((availability >= 0.5).sum())
    latest_available = as_float(latest.get("feature_context_flow_1s_context_available"), 0.0) >= 0.5
    latest_age_ms = int(time.time() * 1000) - int(latest["timestamp"])
    stale_flag = "STALE" if latest_age_ms > MAX_MICRO_AGE_MS else "fresh"
    context_age = as_float(latest.get("feature_context_flow_1s_context_age_ms"), np.nan)
    print(
        "- 10s micro rows can attach 1s context: "
        f"latest_available={latest_available}, "
        f"availability={available_rows}/{len(frame)}, "
        f"latest_row_age={latest_age_ms / 1000:.1f}s ({stale_flag}), "
        f"latest_context_age_ms={format_number(context_age, 1)}, "
        f"path={path}"
    )


def latest_raw_timestamp(symbol):
    path = VENUE_OUTPUT_DIR / f"{symbol}_10s_flow.csv"
    if not path.exists():
        return None, None
    frame = pd.read_csv(path, usecols=["timestamp", "time"])
    if len(frame) == 0:
        return None, None
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if len(frame) == 0:
        return None, None
    row = frame.iloc[-1]
    return int(row["timestamp"]), row.get("time", "")


def micro_artifact_timestamp(artifact):
    return (
        artifact.get("created_at")
        or artifact.get("model_created_at")
        or artifact.get("trained_until_timestamp")
        or "missing"
    )


def micro_scaler_exists(artifact):
    return isinstance(artifact.get("regression_target_scalers"), dict)


def micro_artifact_event_only(artifact):
    return bool(artifact.get("event_only", False))


def micro_model_symbol(artifact):
    return str(artifact.get("model_symbol", artifact.get("symbol", ""))).strip().upper()


def micro_artifact_rejection_notes(artifact):
    notes = []
    notes.extend(str(value) for value in artifact.get("rejection_reasons", []) if str(value).strip())
    selection = artifact.get("selection_reason", {})
    if isinstance(selection, dict):
        notes.extend(str(value) for value in selection.get("selection_notes", []) if str(value).strip())
    return notes


def micro_infer_hard_rejected(artifact):
    if "hard_rejected" in artifact:
        return bool(artifact.get("hard_rejected"))
    joined = " | ".join(micro_artifact_rejection_notes(artifact)).lower()
    hard_markers = [
        "no feature group passed",
        "regression sanity failure",
        "event saturation warning",
        "target groups disagree",
        "promotion is blocked",
    ]
    return any(marker in joined for marker in hard_markers)


def micro_artifact_promotable(artifact):
    if "promotable" in artifact:
        return bool(artifact.get("promotable"))
    return not micro_infer_hard_rejected(artifact)


def micro_artifact_regression_sanity_status(artifact):
    if artifact.get("regression_sanity_status"):
        return str(artifact.get("regression_sanity_status")).strip().lower()
    return "fail" if artifact.get("regression_sanity_failures") else "ok"


def micro_artifact_event_sanity_status(artifact):
    if artifact.get("event_sanity_status"):
        return str(artifact.get("event_sanity_status")).strip().lower()
    return "fail" if artifact.get("event_saturation_warnings") else "ok"


def validate_micro_model_candidate(path, symbol):
    try:
        artifact = load_model(path)
    except Exception as error:
        return None, f"could not load artifact: {error}"
    if artifact is None:
        return None, "could not load artifact"
    model_symbol = micro_model_symbol(artifact)
    if model_symbol != symbol:
        return None, f"model symbol mismatch: model={model_symbol or 'missing'} requested={symbol}"
    event_only = micro_artifact_event_only(artifact)
    if (not event_only) and not micro_scaler_exists(artifact):
        return None, "missing regression_target_scalers"
    regression_columns = list(artifact.get("regression_target_columns", REGRESSION_TARGET_COLUMNS))
    if event_only:
        regression_columns = []
    else:
        try:
            validate_regression_scalers(artifact, regression_columns, allow_legacy=False)
        except ValueError as error:
            return None, str(error)
    hard_rejected = micro_infer_hard_rejected(artifact)
    promotable = micro_artifact_promotable(artifact)
    regression_status = micro_artifact_regression_sanity_status(artifact)
    event_status = micro_artifact_event_sanity_status(artifact)
    if not MICRO_ALLOW_REJECTED_CANDIDATES:
        if hard_rejected:
            return None, "hard_rejected=true"
        if not promotable:
            return None, "promotable=false"
        if (not event_only) and regression_status != "ok":
            return None, f"regression_sanity_status={regression_status}"
        if event_status != "ok":
            return None, f"event_sanity_status={event_status}"
    return artifact, "ok"


def choose_micro_model_artifact(symbol):
    diagnostics = {
        "selection_mode": MICRO_MODEL_SELECTION,
        "selected_path": None,
        "rejections": [],
    }
    if MICROSTRUCTURE_MODEL_PATH:
        path = Path(MICROSTRUCTURE_MODEL_PATH)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        active_path = PROJECT_ROOT / "models" / "active" / symbol / "microstructure_10s" / "model.json"
        candidate_root = PROJECT_ROOT / "models" / "candidates" / symbol / "microstructure_10s"
        try:
            allowed_candidate = path.resolve().relative_to(candidate_root.resolve())
            if path.name == "model.json" and len(allowed_candidate.parts) >= 2:
                artifact, reason = validate_micro_model_candidate(path, symbol)
                if artifact is None:
                    diagnostics["rejections"].append({"path": str(path), "reason": reason})
                    return None, None, diagnostics
                diagnostics["selected_path"] = str(path)
                return path, artifact, diagnostics
        except ValueError:
            pass
        if path.resolve() == active_path.resolve():
            artifact, reason = validate_micro_model_candidate(path, symbol)
            if artifact is None:
                diagnostics["rejections"].append({"path": str(path), "reason": reason})
                return None, None, diagnostics
            diagnostics["selected_path"] = str(path)
            return path, artifact, diagnostics
        diagnostics["rejections"].append({"path": str(path), "reason": "path is not allowed for requested symbol"})
        return None, None, diagnostics
    active_path = PROJECT_ROOT / "models" / "active" / symbol / "microstructure_10s" / "model.json"
    if MICRO_MODEL_SELECTION == "active_only":
        if not active_path.exists():
            return None, None, diagnostics
        print(f"[{symbol} micro model selection] active_only path before loading: {active_path}")
        artifact, reason = validate_micro_model_candidate(active_path, symbol)
        if artifact is None:
            diagnostics["rejections"].append({"path": str(active_path), "reason": reason})
            return None, None, diagnostics
        diagnostics["selected_path"] = str(active_path)
        return active_path, artifact, diagnostics
    candidate_root = PROJECT_ROOT / "models" / "candidates" / symbol / "microstructure_10s"
    candidates = sorted(candidate_root.glob("*/model.json"), reverse=True) if candidate_root.exists() else []
    if MICRO_MODEL_SELECTION == "latest_candidate":
        for candidate in candidates:
            print(f"[{symbol} micro model selection] latest_candidate path before loading: {candidate}")
            artifact, reason = validate_micro_model_candidate(candidate, symbol)
            if artifact is not None:
                diagnostics["selected_path"] = str(candidate)
                return candidate, artifact, diagnostics
            diagnostics["rejections"].append({"path": str(candidate), "reason": reason})
            print(f"[{symbol} micro model selection] skipped candidate: {candidate} reason={reason}")
    return None, None, diagnostics


def micro_candidate_paths_for_selection(symbol):
    diagnostics = {
        "selection_mode": MICRO_MODEL_SELECTION,
        "selected_path": None,
        "rejections": [],
        "live_rejections": [],
    }
    active_path = PROJECT_ROOT / "models" / "active" / symbol / "microstructure_10s" / "model.json"
    candidate_root = PROJECT_ROOT / "models" / "candidates" / symbol / "microstructure_10s"
    if MICROSTRUCTURE_MODEL_PATH:
        path = Path(MICROSTRUCTURE_MODEL_PATH)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        return [path], diagnostics
    if MICRO_MODEL_SELECTION == "active_only":
        return ([active_path] if active_path.exists() else []), diagnostics
    if MICRO_MODEL_SELECTION == "latest_candidate":
        candidates = sorted(candidate_root.glob("*/model.json"), reverse=True) if candidate_root.exists() else []
        return candidates, diagnostics
    return [], diagnostics


def live_sanity_statuses(regression_sanity, event_sanity):
    regression_status = "fail" if regression_sanity["failed"] else ("warning" if regression_sanity["warning"] else "ok")
    event_status = "fail" if event_sanity["failed"] else ("warning" if event_sanity["warning"] else "ok")
    clipped_count = sum(
        1 for warning in regression_sanity.get("warnings", [])
        if "clipped from" in str(warning) or "non-finite" in str(warning)
    )
    return regression_status, event_status, clipped_count


def predict_micro_candidate_for_live_row(symbol, model_path, artifact, feature_frame, snapshot_step_seconds):
    model_symbol = micro_model_symbol(artifact)
    model_columns = list(artifact.get("feature_columns", []))
    latest = feature_frame.tail(1).copy()
    current_columns = get_micro_feature_columns(latest)
    schema_diagnostics = micro_schema_diagnostics(model_columns, current_columns)
    latest = add_missing_optional_context_columns(latest, model_columns)
    latest = fill_optional_context_feature_values(latest, model_columns)
    saved_columns_hash = feature_schema_hash(model_columns)
    model_schema_hash = artifact.get("feature_schema_hash", "")
    if model_schema_hash and model_schema_hash != saved_columns_hash:
        return None, f"micro feature schema mismatch: model={model_schema_hash} saved_columns={saved_columns_hash}"
    missing_required = [
        column
        for column in required_micro_feature_columns(model_columns)
        if column not in current_columns
    ]
    if missing_required:
        return None, f"missing required micro features: {missing_required[:10]}"
    event_only = micro_artifact_event_only(artifact)
    regression_columns = [] if event_only else list(artifact.get("regression_target_columns", REGRESSION_TARGET_COLUMNS))
    event_columns = list(artifact.get("event_target_columns", EVENT_TARGET_COLUMNS))
    if not event_only:
        validate_regression_scalers(artifact, regression_columns, allow_legacy=False)
    regression, event_probabilities = predict_with_artifact(
        artifact,
        latest,
        event_temperature=MICRO_EVENT_PROB_TEMPERATURE,
    )
    regression_sanity = (
        {
            "ok": True,
            "warning": False,
            "failed": False,
            "clipped_values": {},
            "warnings": [],
        }
        if event_only
        else regression_sanity_report(regression[0], regression_columns, artifact)
    )
    event_sanity = event_saturation_report(event_probabilities[0], event_columns)
    live_regression_status, live_event_status, clipped_count = live_sanity_statuses(
        regression_sanity,
        event_sanity,
    )
    if regression_sanity["failed"] or event_sanity["failed"]:
        return None, (
            "selected model failed live-input sanity: "
            f"live_regression_sanity_status={live_regression_status}, "
            f"live_event_sanity_status={live_event_status}, "
            f"live_event_saturation_fraction={event_sanity['fraction_saturated']:.4f}, "
            f"live_regression_clipped_count={clipped_count}"
        )
    result = {
        "timestamp": int(latest["timestamp"].iloc[0]),
        "time": latest["time"].iloc[0],
        "model_path": str(model_path),
        "model_id": artifact.get("model_id", ""),
        "model_selection_mode": MICRO_MODEL_SELECTION,
        "selected_candidate_path": str(model_path),
        "model_timestamp": micro_artifact_timestamp(artifact),
        "regression_target_scalers_exists": micro_scaler_exists(artifact),
        "event_only": event_only,
        "micro_mode": "event-only" if event_only else "full",
        "model_symbol": model_symbol,
        "hard_rejected": micro_infer_hard_rejected(artifact),
        "promotable": micro_artifact_promotable(artifact),
        "rejection_reasons": micro_artifact_rejection_notes(artifact),
        "research_only_loaded": MICRO_ALLOW_REJECTED_CANDIDATES
        and (micro_infer_hard_rejected(artifact) or not micro_artifact_promotable(artifact)),
        "selected_feature_group": artifact.get("selected_feature_group", artifact.get("feature_group_used", "")),
        "feature_group_used": artifact.get("feature_group_used", ""),
        "snapshot_step_seconds": snapshot_step_seconds,
        "schema_required_missing_count": len(schema_diagnostics["required_missing_columns"]),
        "schema_optional_filled_count": len(schema_diagnostics["optional_missing_columns_filled"]),
        "schema_extra_ignored_count": len(schema_diagnostics["extra_in_current"]),
        "schema_artifact_feature_schema_hash": schema_diagnostics["artifact_feature_schema_hash"],
        "schema_current_required_feature_hash": schema_diagnostics["current_required_feature_hash"],
        "artifact_regression_sanity_status": micro_artifact_regression_sanity_status(artifact),
        "artifact_event_sanity_status": micro_artifact_event_sanity_status(artifact),
        "live_regression_sanity_status": "not_applicable" if event_only else live_regression_status,
        "live_event_sanity_status": live_event_status,
        "live_event_saturation_fraction": event_sanity["fraction_saturated"],
        "live_regression_clipped_count": int(clipped_count),
        "regression_sanity_status": "not_applicable" if event_only else live_regression_status,
        "regression_sanity_failed": False if event_only else regression_sanity["failed"],
        "regression_sanity_warnings": regression_sanity["warnings"],
        "event_sanity_status": live_event_status,
        "event_sanity_failed": event_sanity["failed"],
        "event_saturation_fraction": event_sanity["fraction_saturated"],
        "event_saturated_events": event_sanity["saturated_events"],
    }
    for index, column in enumerate(regression_columns):
        result[f"raw_pred_{column}"] = float(regression[0, index])
        result[f"pred_{column}"] = float(regression_sanity["clipped_values"].get(column, regression[0, index]))
    for index, column in enumerate(event_columns):
        result[f"prob_{column}"] = float(event_probabilities[0, index])
    add_scare_severity_probabilities(result)
    return result, None


def latest_micro_prediction(symbol):
    snapshot_path = VENUE_OUTPUT_DIR / f"{symbol}_10s_flow.csv"
    if not snapshot_path.exists():
        return None, "missing 10s snapshot file"
    try:
        snapshots = load_snapshot_rows(snapshot_path)
        feature_frame, snapshot_step_seconds = build_latest_feature_only(snapshots)
        if len(feature_frame) == 0:
            return None, "no feature-ready 10s snapshot rows"
        feature_frame, _ = attach_hierarchical_context(
            feature_frame,
            PROJECT_ROOT,
            symbol,
            layers=("htf", "regime15", "regime30", "flow3m", "flow1s"),
            as_model_features=True,
        )
    except Exception as error:
        return None, f"micro prediction failed: {error}"
    candidate_paths, selection_diagnostics = micro_candidate_paths_for_selection(symbol)
    if not candidate_paths:
        return None, "missing valid 10s model (no candidate/active model found)"
    for model_path in candidate_paths:
        print(f"[{symbol} micro model selection] checking artifact for live row: {model_path}")
        artifact, reason = validate_micro_model_candidate(model_path, symbol)
        if artifact is None:
            selection_diagnostics["rejections"].append({"path": str(model_path), "reason": reason})
            print(f"[{symbol} micro model selection] skipped artifact candidate: {model_path} reason={reason}")
            continue
        try:
            result, live_reason = predict_micro_candidate_for_live_row(
                symbol,
                model_path,
                artifact,
                feature_frame,
                snapshot_step_seconds,
            )
        except Exception as error:
            result, live_reason = None, f"micro prediction failed: {error}"
        if result is not None:
            selection_diagnostics["selected_path"] = str(model_path)
            result["model_selection_rejections"] = selection_diagnostics.get("rejections", [])
            result["model_selection_live_rejections"] = selection_diagnostics.get("live_rejections", [])
            return result, None
        selection_diagnostics["live_rejections"].append({"path": str(model_path), "reason": live_reason})
        print(f"[{symbol} micro model selection] skipped live candidate: {model_path} reason={live_reason}")
        if MICRO_MODEL_SELECTION != "latest_candidate":
            break
    reason_text = "; ".join(
        f"{item['path']} => {item['reason']}"
        for item in selection_diagnostics.get("live_rejections", [])[:5]
    )
    return None, f"all valid candidates failed live-input sanity ({reason_text or 'no live-valid candidate'})"


def severity_probability(base_probability, predicted_excursion, threshold, direction):
    if direction == "down":
        distance = max(0.0, -predicted_excursion)
    else:
        distance = max(0.0, predicted_excursion)
    # The trained event probability is for MICRO_MOVE_THRESHOLD=0.003. These
    # minor/major views are descriptive derived scores, not separate model heads.
    shape_score = 1.0 / (1.0 + np.exp(-(distance - threshold) / max(threshold, 1e-6)))
    return float(np.clip(max(base_probability * 0.35, shape_score * base_probability), 0.0, 1.0))


def add_scare_severity_probabilities(result):
    up_base = as_float(result.get("prob_upside_scare_event_10s"), 0.0)
    down_base = as_float(result.get("prob_downside_scare_event_10s"), 0.0)
    up_runup = as_float(result.get("pred_max_runup_10s"), 0.0)
    down_drawdown = as_float(result.get("pred_max_drawdown_10s"), 0.0)
    result["prob_minor_upside_scare"] = severity_probability(up_base, up_runup, 0.0010, "up")
    result["prob_moderate_upside_scare"] = up_base
    result["prob_major_upside_scare"] = severity_probability(up_base, up_runup, 0.0060, "up")
    result["prob_minor_downside_scare"] = severity_probability(down_base, down_drawdown, 0.0010, "down")
    result["prob_moderate_downside_scare"] = down_base
    result["prob_major_downside_scare"] = severity_probability(down_base, down_drawdown, 0.0060, "down")


def latest_3m_prediction(symbol, reference_timestamp):
    path = PROJECT_ROOT / "data" / "live_predictions" / f"{symbol}_live_3m_predictions.csv"
    frame = load_csv(path)
    if len(frame) == 0 or "timestamp" not in frame.columns:
        return None, "missing 3m prediction file"
    if reference_timestamp is not None:
        frame = frame[frame["timestamp"] <= reference_timestamp]
    if len(frame) == 0:
        return None, "no 3m prediction at or before latest timestamp"
    row = frame.iloc[-1].to_dict()
    if reference_timestamp is not None and reference_timestamp - int(row["timestamp"]) > MAX_3M_AGE_MS:
        return row, "stale"
    return row, None


def latest_1s_order_flow_prediction(symbol, reference_timestamp):
    path = VENUE_OUTPUT_DIR / f"{symbol}_1s_order_flow_predictions.csv"
    diagnostics = load_1s_loop_diagnostics(symbol)
    frame = load_csv(path)
    if len(frame) == 0 or "timestamp" not in frame.columns:
        reason = diagnostics.get("blocking_reason") if diagnostics else ""
        return None, f"missing 1s prediction context" + (f"; loop reason: {reason}" if reason else "")
    file_latest_timestamp = int(pd.to_numeric(frame["timestamp"], errors="coerce").max())
    prediction_file_age_ms = int(time.time() * 1000) - file_latest_timestamp
    prediction_file_stale = prediction_file_age_ms > MAX_FLOW_1S_AGE_MS
    if reference_timestamp is not None:
        frame = frame[frame["timestamp"] <= reference_timestamp]
    if len(frame) == 0:
        return None, "no 1s prediction at or before latest timestamp"
    row = frame.iloc[-1].to_dict()
    timestamp = int(row["timestamp"])
    if reference_timestamp is not None:
        context_age_ms = int(reference_timestamp) - timestamp
    else:
        context_age_ms = int(time.time() * 1000) - timestamp
    row["_prediction_file_age_seconds"] = prediction_file_age_ms / 1000.0
    row["_context_age_ms"] = float(context_age_ms)
    row["_prediction_file_stale"] = bool(prediction_file_stale)
    if diagnostics:
        row["_loop_diagnostics"] = diagnostics
    if prediction_file_stale:
        reason = diagnostics.get("blocking_reason", "")
        written = diagnostics.get("newly_written_predictions", "unknown")
        candidate = diagnostics.get("candidate_rows_newer_than_latest_prediction", "unknown")
        return row, (
            f"STALE prediction_file_age_seconds={prediction_file_age_ms / 1000:.1f}; "
            f"loop_status={diagnostics.get('status', 'unknown')}; "
            f"latest_snapshot={diagnostics.get('latest_snapshot_timestamp', 'unknown')}; "
            f"latest_prediction={diagnostics.get('latest_prediction_timestamp', 'unknown')}; "
            f"candidate_newer={candidate}; newly_written={written}; "
            f"blocking_reason={reason or 'none'}"
        )
    if context_age_ms > MAX_FLOW_1S_AGE_MS:
        return row, f"STALE context_age_ms={context_age_ms}"
    return row, "fresh"


def load_1s_loop_diagnostics(symbol):
    path = VENUE_OUTPUT_DIR / f"{symbol}_1s_order_flow_loop_diagnostics.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        return payload
    except Exception as error:
        return {"status": "read_error", "blocking_reason": f"could not read diagnostics: {error}", "_path": str(path)}


def regime_timestamp_column(frame):
    for column in ["close_timestamp", "timestamp", "time_close", "open_timestamp"]:
        if column in frame.columns:
            return column
    return None


def latest_regime(symbol, timeframe, reference_timestamp):
    path = PROJECT_ROOT / "data" / f"{symbol}_{timeframe}_regime_features.csv"
    if not path.exists():
        return None, regime_unavailable_reason(symbol, timeframe, path)
    frame = load_csv(path)
    if len(frame) == 0:
        return None, regime_unavailable_reason(symbol, timeframe, path)
    timestamp_column = regime_timestamp_column(frame)
    if timestamp_column is None:
        return None, f"{timeframe} regime file has no timestamp column"
    frame[timestamp_column] = pd.to_numeric(frame[timestamp_column], errors="coerce")
    if reference_timestamp is not None:
        frame = frame[frame[timestamp_column] <= reference_timestamp]
    frame = frame.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
    if len(frame) == 0:
        return None, f"no completed candle: no completed {timeframe} regime row in {path}"
    row = frame.iloc[-1].to_dict()
    row["_timestamp_column"] = timestamp_column
    row["_timestamp"] = int(row[timestamp_column])
    if reference_timestamp is not None and reference_timestamp - int(row[timestamp_column]) > MAX_REGIME_AGE_MS:
        age_seconds = (reference_timestamp - int(row[timestamp_column])) / 1000.0
        return row, f"stale file: latest {timeframe} context age={age_seconds:.1f}s"
    return row, None


def latest_htf_context(symbol, reference_timestamp):
    path = PROJECT_ROOT / "data" / f"{symbol}_htf_context_features.csv"
    if not path.exists():
        return None, htf_unavailable_reason(symbol, path)
    frame = load_csv(path)
    if len(frame) == 0:
        return None, htf_unavailable_reason(symbol, path)
    timestamp_column = "close_timestamp" if "close_timestamp" in frame.columns else "timestamp"
    if timestamp_column not in frame.columns:
        return None, "HTF context file has no timestamp column"
    if reference_timestamp is not None:
        frame = frame[frame[timestamp_column] <= reference_timestamp]
    frame = frame.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
    if len(frame) == 0:
        return None, f"no completed candle: no completed HTF context row at or before latest timestamp in {path}"
    row = frame.iloc[-1].to_dict()
    row["_timestamp"] = int(row[timestamp_column])
    if reference_timestamp is not None and reference_timestamp - int(row[timestamp_column]) > MAX_HTF_AGE_MS:
        age_seconds = (reference_timestamp - int(row[timestamp_column])) / 1000.0
        return row, f"stale file: latest HTF context age={age_seconds:.1f}s"
    return row, None


def regime_label(row):
    if row is None:
        return "unavailable"
    if "regime" in row and isinstance(row["regime"], str):
        return row["regime"]
    for column in ["regime_bullish", "regime_bearish", "regime_high_volatility_chop", "regime_chop"]:
        if as_float(row.get(column), 0.0) >= 0.5:
            return column.replace("regime_", "").replace("_", " ")
    trend_score = as_float(row.get("trend_score"), 0.0)
    chop_score = as_float(row.get("chop_score"), 0.0)
    if chop_score >= 0.6:
        return "chop"
    if trend_score > 0.25:
        return "bullish"
    if trend_score < -0.25:
        return "bearish"
    return "neutral/chop"


def path_label(row):
    if row is None:
        return "unavailable"
    probs = {
        "short": as_float(row.get("prob_short"), 0.0),
        "neutral": as_float(row.get("prob_neutral"), 0.0),
        "long": as_float(row.get("prob_long"), 0.0),
    }
    return max(probs, key=probs.get)


def flow_1s_label(row):
    if row is None:
        return "unavailable"
    if "decoded_flow_class_1s" in row and pd.notna(row.get("decoded_flow_class_1s")):
        value = row.get("decoded_flow_class_1s")
        if isinstance(value, str):
            return value
    if "thresholded_next_1s_flow_class" in row and pd.notna(row.get("thresholded_next_1s_flow_class")):
        class_id = int(row.get("thresholded_next_1s_flow_class"))
        return {
            0: "sell_dominant",
            1: "neutral",
            2: "buy_dominant",
        }.get(class_id, str(class_id))
    probs = {
        "sell_dominant": as_float(row.get("prob_sell_dominant"), 0.0),
        "neutral": as_float(row.get("prob_neutral"), 0.0),
        "buy_dominant": as_float(row.get("prob_buy_dominant"), 0.0),
    }
    return max(probs, key=probs.get)


def row_first(row, names, default=np.nan):
    for name in names:
        if name in row and pd.notna(row.get(name)):
            return row.get(name)
    return default


def flow_1s_pred_pressure(row):
    if row is None:
        return np.nan
    return as_float(
        row_first(row, ["pred_market_pressure_1s", "pred_future_market_pressure_1s"]),
        np.nan,
    )


def flow_1s_class_pressure_disagreement(row):
    label = flow_1s_label(row).lower()
    pressure = flow_1s_pred_pressure(row)
    if not np.isfinite(pressure):
        return False, ""
    if label == "neutral" and abs(pressure) >= 0.50:
        return True, f"neutral_class_high_pressure_abs={abs(pressure):.4f}"
    if label == "buy_dominant" and pressure < 0:
        return True, f"buy_dominant_with_negative_pressure={pressure:.4f}"
    if label == "sell_dominant" and pressure > 0:
        return True, f"sell_dominant_with_positive_pressure={pressure:.4f}"
    return False, ""


def pressure_side(micro):
    if micro is None:
        return "unknown"
    up = max(
        as_float(micro.get("prob_moderate_upside_scare"), 0.0),
        as_float(micro.get("prob_aggressive_buy_burst_10s"), 0.0),
    )
    down = max(
        as_float(micro.get("prob_moderate_downside_scare"), 0.0),
        as_float(micro.get("prob_aggressive_sell_burst_10s"), 0.0),
    )
    if max(up, down) < 0.50:
        return "neutral"
    return "upside" if up > down else "downside"


def final_interpretation(regime_15m, regime_30m, path_3m, micro):
    if micro is None:
        return "calm" if path_3m is None else "pressure building"
    if micro.get("regression_sanity_failed") or micro.get("event_sanity_failed"):
        return "chop / unreliable signal"

    up_scare = max(
        as_float(micro.get("prob_moderate_upside_scare"), 0.0),
        as_float(micro.get("prob_major_upside_scare"), 0.0),
    )
    down_scare = max(
        as_float(micro.get("prob_moderate_downside_scare"), 0.0),
        as_float(micro.get("prob_major_downside_scare"), 0.0),
    )
    buy_burst = as_float(micro.get("prob_aggressive_buy_burst_10s"), 0.0)
    sell_burst = as_float(micro.get("prob_aggressive_sell_burst_10s"), 0.0)
    bid_drop = as_float(micro.get("prob_bid_liquidity_drop_10s"), 0.0)
    ask_drop = as_float(micro.get("prob_ask_liquidity_drop_10s"), 0.0)
    spread_expansion = as_float(micro.get("prob_spread_expansion_event_10s"), 0.0)
    side = pressure_side(micro)

    regime_text = f"{regime_label(regime_15m)} {regime_label(regime_30m)}".lower()
    bullish_regime = "bullish" in regime_text
    bearish_regime = "bearish" in regime_text
    chop_regime = "chop" in regime_text or "neutral" in regime_text

    if spread_expansion >= 0.70 or bid_drop >= 0.75 or ask_drop >= 0.75:
        if max(up_scare, down_scare, buy_burst, sell_burst) >= 0.65:
            return "liquidity unstable"
    if up_scare >= 0.70 or (buy_burst >= 0.75 and ask_drop >= 0.55):
        if bearish_regime:
            return "counter-regime pressure"
        if bullish_regime:
            return "regime-aligned pressure"
        return "upside squeeze risk"
    if down_scare >= 0.70 or (sell_burst >= 0.75 and bid_drop >= 0.55):
        if bullish_regime:
            return "counter-regime pressure"
        if bearish_regime:
            return "regime-aligned pressure"
        return "downside scare risk"
    if chop_regime and side != "neutral":
        return "chop / unreliable signal"
    if max(up_scare, down_scare, buy_burst, sell_burst) >= 0.55:
        return "pressure building"
    return "calm"


def print_regime_line(label, row, status):
    if row is None:
        print(f"- {label}: unavailable ({status})")
        return
    details = [
        f"regime={regime_label(row)}",
        f"trend_score={format_number(row.get('trend_score'))}",
        f"chop_score={format_number(row.get('chop_score'))}",
        f"timestamp={row.get('_timestamp', 'unknown')}",
    ]
    if status:
        details.append(f"status={status}")
    print(f"- {label}: " + ", ".join(details))


def print_htf_line(row, status):
    if row is None:
        print(f"- daily/hourly broad state: unavailable ({status})")
        return
    details = [
        f"bull={format_percent(row.get('hourly_bull_prob'))}",
        f"bear={format_percent(row.get('hourly_bear_prob'))}",
        f"chop={format_percent(row.get('hourly_chop_prob'))}",
        f"trend_score={format_number(row.get('htf_trend_score'))}",
        f"vol_state={format_number(row.get('htf_volatility_state'))}",
        f"vwap_dist={format_percent(row.get('distance_from_htf_vwap'))}",
        f"daily_range_pos={format_number(row.get('rolling_daily_range_position'))}",
        f"timestamp={row.get('_timestamp', 'unknown')}",
    ]
    if status:
        details.append(f"status={status}")
    print("- daily/hourly broad state: " + ", ".join(details))


def print_3m_line(row, status):
    if row is None:
        print(f"- 3m path/return: unavailable ({status})")
        return
    details = [
        f"class={path_label(row)}",
        f"short={format_percent(row.get('prob_short'))}",
        f"neutral={format_percent(row.get('prob_neutral'))}",
        f"long={format_percent(row.get('prob_long'))}",
        f"pred_return={format_percent(row.get('pred_future_return_3'))}",
        f"timestamp={int(row.get('timestamp')) if pd.notna(row.get('timestamp')) else 'unknown'}",
    ]
    if status:
        details.append(f"status={status}")
    print("- 3m path/return: " + ", ".join(details))


def print_1s_flow_line(row, status):
    if row is None:
        print(f"- 1s order-flow context: unavailable ({status})")
        return
    disagreement, disagreement_reason = flow_1s_class_pressure_disagreement(row)
    details = [
        f"class={flow_1s_label(row)}",
        f"sell={format_percent(row_first(row, ['prob_sell_dominant_1s', 'prob_sell_dominant']))}",
        f"neutral={format_percent(row_first(row, ['prob_neutral_1s', 'prob_neutral']))}",
        f"buy={format_percent(row_first(row, ['prob_buy_dominant_1s', 'prob_buy_dominant']))}",
        f"buy_burst={format_percent(row_first(row, ['buy_burst_prob_1s', 'prob_aggressive_buy_burst_1s']))}",
        f"sell_burst={format_percent(row_first(row, ['sell_burst_prob_1s', 'prob_aggressive_sell_burst_1s']))}",
        f"pred_pressure={format_percent(row_first(row, ['pred_market_pressure_1s', 'pred_future_market_pressure_1s']))}",
        f"target_horizon={int(row_first(row, ['model_target_horizon_seconds', 'target_horizon_seconds'], 1))}s",
        f"timestamp={int(row.get('timestamp')) if pd.notna(row.get('timestamp')) else 'unknown'}",
        f"prediction_file_age_seconds={format_number(row.get('_prediction_file_age_seconds'), 1)}",
        f"context_age_ms={format_number(row.get('_context_age_ms'), 0)}",
        f"1s_class_pressure_disagreement={str(disagreement).lower()}",
        f"disagreement_reason={disagreement_reason or 'none'}",
    ]
    if "raw_argmax_next_1s_flow_class" in row and pd.notna(row.get("raw_argmax_next_1s_flow_class")):
        raw_class_id = int(row.get("raw_argmax_next_1s_flow_class"))
        raw_label = {
            0: "sell_dominant",
            1: "neutral",
            2: "buy_dominant",
        }.get(raw_class_id, str(raw_class_id))
        details.append(f"raw_argmax={raw_label}")
    if "model_id" in row and pd.notna(row.get("model_id")):
        details.append(f"model_id={row.get('model_id')}")
    diagnostics = row.get("_loop_diagnostics") if isinstance(row, dict) else None
    if isinstance(diagnostics, dict) and diagnostics:
        details.append(f"loop_status={diagnostics.get('status', 'unknown')}")
        details.append(f"loop_reason={diagnostics.get('blocking_reason', 'none') or 'none'}")
        details.append(f"candidate_newer={diagnostics.get('candidate_rows_newer_than_latest_prediction', 'unknown')}")
        details.append(f"feature_not_ready={diagnostics.get('feature_ready_skipped_rows', 'unknown')}")
        details.append(f"non_finite={diagnostics.get('non_finite_feature_skipped_rows', 'unknown')}")
        details.append(f"invalid_book={diagnostics.get('invalid_book_skipped_rows', 'unknown')}")
    if status:
        details.append(f"status={status}")
    print("- 1s order-flow context: " + ", ".join(details))


def print_micro_lines(micro, status):
    if micro is None:
        print(f"- 10s microstructure: unavailable ({status})")
        return
    print(
        "- 10s microstructure: "
        f"timestamp={micro['timestamp']}, "
        f"step={micro['snapshot_step_seconds']:.3g}s, "
        f"model={micro['model_path']}"
    )
    print(
        "  model selection: "
        f"mode={micro.get('model_selection_mode', MICRO_MODEL_SELECTION)}, "
        f"10s_micro_mode={micro.get('micro_mode', 'event-only' if micro.get('event_only') else 'full')}, "
        f"selected_candidate_path={micro.get('selected_candidate_path', micro.get('model_path'))}, "
        f"model_timestamp={micro.get('model_timestamp', 'missing')}, "
        f"regression_target_scalers_exists={micro.get('regression_target_scalers_exists', False)}, "
        f"event_only={micro.get('event_only', False)}, "
        f"hard_rejected={micro.get('hard_rejected', 'unknown')}, "
        f"promotable={micro.get('promotable', 'unknown')}, "
        f"research_only_loaded={micro.get('research_only_loaded', False)}, "
        f"artifact_regression_sanity={micro.get('artifact_regression_sanity_status', 'unknown')}, "
        f"artifact_event_sanity={micro.get('artifact_event_sanity_status', 'unknown')}, "
        f"selected_feature_group={micro.get('selected_feature_group', micro.get('feature_group_used', 'missing'))}"
    )
    if micro.get("rejection_reasons"):
        print("  artifact rejection reasons:")
        for reason in micro.get("rejection_reasons", [])[:6]:
            print(f"  - {reason}")
    rejections = micro.get("model_selection_rejections") or []
    if rejections:
        print("  rejected newer/bad micro candidates:")
        for item in rejections:
            print(f"  - {item.get('path')}: {item.get('reason')}")
    live_rejections = micro.get("model_selection_live_rejections") or []
    if live_rejections:
        print("  skipped-live micro candidates:")
        for item in live_rejections:
            print(f"  - {item.get('path')}: {item.get('reason')}")
    print(
        "  schema: "
        f"required_missing={micro.get('schema_required_missing_count', 'unknown')}, "
        f"optional_filled={micro.get('schema_optional_filled_count', 'unknown')}, "
        f"extra_ignored={micro.get('schema_extra_ignored_count', 'unknown')}, "
        f"artifact_hash={micro.get('schema_artifact_feature_schema_hash', 'unknown')}, "
        f"current_required_hash={micro.get('schema_current_required_feature_hash', 'unknown')}"
    )
    if status:
        print(f"  status={status}")
    print(
        "  sanity: "
        f"regression={micro.get('regression_sanity_status', 'unknown')}, "
        f"events={micro.get('event_sanity_status', 'unknown')}, "
        f"event_saturation={format_percent(micro.get('event_saturation_fraction'))}, "
        f"live_regression={micro.get('live_regression_sanity_status', 'unknown')}, "
        f"live_events={micro.get('live_event_sanity_status', 'unknown')}, "
        f"live_event_saturation={format_percent(micro.get('live_event_saturation_fraction'))}, "
        f"live_regression_clipped_count={micro.get('live_regression_clipped_count', 'unknown')}"
    )
    if micro.get("regression_sanity_warnings"):
        print("  WARNING: regression output clipped for display; model may be extrapolating.")
        for warning in micro.get("regression_sanity_warnings", [])[:6]:
            print(f"  - {warning}")
    if micro.get("event_sanity_status") in {"warning", "fail"}:
        saturated = micro.get("event_saturated_events", [])
        print("  WARNING: event probabilities are saturated.")
        print(f"  - saturated events: {', '.join(saturated) if saturated else 'none'}")
    print(
        "  upside scare derived: "
        f"minor={format_percent(micro.get('prob_minor_upside_scare'))}, "
        f"moderate={format_percent(micro.get('prob_moderate_upside_scare'))}, "
        f"major={format_percent(micro.get('prob_major_upside_scare'))}"
    )
    print(
        "  downside scare derived: "
        f"minor={format_percent(micro.get('prob_minor_downside_scare'))}, "
        f"moderate={format_percent(micro.get('prob_moderate_downside_scare'))}, "
        f"major={format_percent(micro.get('prob_major_downside_scare'))}"
    )
    print(
        "  bursts/liquidity: "
        f"buy_burst={format_percent(micro.get('prob_aggressive_buy_burst_10s'))}, "
        f"sell_burst={format_percent(micro.get('prob_aggressive_sell_burst_10s'))}, "
        f"bid_drop={format_percent(micro.get('prob_bid_liquidity_drop_10s'))}, "
        f"ask_drop={format_percent(micro.get('prob_ask_liquidity_drop_10s'))}, "
        f"spread_expansion={format_percent(micro.get('prob_spread_expansion_event_10s'))}"
    )
    print(
        "  continuation/reversal: "
        f"cont_30s={format_percent(micro.get('prob_continuation_30s'))}, "
        f"cont_60s={format_percent(micro.get('prob_continuation_60s'))}, "
        f"rev_up_30s={format_percent(micro.get('prob_reversal_after_upside_scare_30s'))}, "
        f"rev_down_30s={format_percent(micro.get('prob_reversal_after_downside_scare_30s'))}, "
        f"rev_up_60s={format_percent(micro.get('prob_reversal_after_upside_scare_60s'))}, "
        f"rev_down_60s={format_percent(micro.get('prob_reversal_after_downside_scare_60s'))}"
    )


def print_symbol_stack(symbol):
    latest_timestamp, latest_time = latest_raw_timestamp(symbol)
    htf, htf_status = latest_htf_context(symbol, latest_timestamp)
    regime_15m, regime_15m_status = latest_regime(symbol, "15m", latest_timestamp)
    regime_30m, regime_30m_status = latest_regime(symbol, "30m", latest_timestamp)
    path_3m, path_3m_status = latest_3m_prediction(symbol, latest_timestamp)
    flow_1s, flow_1s_status = latest_1s_order_flow_prediction(symbol, latest_timestamp)
    micro, micro_status = latest_micro_prediction(symbol)
    if micro and latest_timestamp is not None and latest_timestamp - int(micro["timestamp"]) > MAX_MICRO_AGE_MS:
        micro_status = "stale"

    interpretation = final_interpretation(regime_15m, regime_30m, path_3m, micro)

    print("=" * 88)
    print(f"{symbol} live paper market stack")
    print(f"Latest timestamp: {latest_timestamp if latest_timestamp is not None else 'unavailable'}")
    print(f"Latest time: {latest_time if latest_time else 'unavailable'}")
    print_file_freshness(symbol)
    print_htf_line(htf, htf_status)
    print_regime_line("15m regime context", regime_15m, regime_15m_status)
    print_regime_line("30m regime context", regime_30m, regime_30m_status)
    print_3m_line(path_3m, path_3m_status)
    print_1s_flow_line(flow_1s, flow_1s_status)
    print_micro_1s_context_health(symbol)
    print_micro_lines(micro, micro_status)
    if flow_1s is not None and bool(flow_1s.get("_prediction_file_stale", False)):
        print("Final paper-only interpretation: suppressed; 1s order-flow prediction file is stale.")
    elif micro and (micro.get("regression_sanity_failed") or micro.get("event_sanity_failed")):
        print("Final paper-only interpretation: suppressed; microstructure sanity failed.")
    else:
        print(f"Final paper-only interpretation: {interpretation}")
    print("No buy/sell command. No trade sizing. No trades are placed.")


def print_dashboard_once():
    print("\nLive paper market stack dashboard")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"10s snapshot directory: {VENUE_OUTPUT_DIR}")
    print("Architecture: daily/hourly prior + 15m/30m regime + 3m flow + 1s order-flow + 10s microstructure.")
    for symbol in SYMBOLS:
        try:
            print_symbol_stack(symbol)
        except Exception as error:
            print("=" * 88)
            print(f"{symbol} live paper market stack")
            print(f"Dashboard error: {error}")
            print("No trades are placed.")


def main():
    if not SYMBOLS:
        raise RuntimeError("No symbols configured.")
    while True:
        print_dashboard_once()
        if STACK_RUN_ONCE:
            break
        time.sleep(STACK_LOOP_SECONDS)


if __name__ == "__main__":
    main()
