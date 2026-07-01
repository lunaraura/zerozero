import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
    print_micro_schema_diagnostics,
    predict_with_artifact,
    regression_sanity_report,
    event_saturation_report,
    validate_regression_scalers,
    required_micro_feature_columns,
)
from hierarchical_context import attach_hierarchical_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
MODEL_PATH_ENV = os.getenv("MICROSTRUCTURE_MODEL_PATH", "").strip()
MICRO_MODEL_SELECTION = os.getenv("MICRO_MODEL_SELECTION", "latest_candidate").strip().lower()
MICRO_ALLOW_REJECTED_CANDIDATES = os.getenv("MICRO_ALLOW_REJECTED_CANDIDATES", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
MICRO_SHOW_MAX_STALENESS_SECONDS = int(os.getenv("MICRO_SHOW_MAX_STALENESS_SECONDS", "120"))
MICRO_DEPTH_DISPLAY_THRESHOLD = float(os.getenv("MICRO_DEPTH_DISPLAY_THRESHOLD", "5.0"))
MICRO_EVENT_PROB_TEMPERATURE = float(os.getenv("MICRO_EVENT_PROB_TEMPERATURE", "1.0"))
VALID_MODEL_SELECTIONS = {"latest_candidate", "active_only"}

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
INPUT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_10s_flow.csv"
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "active" / SYMBOL / "microstructure_10s" / "model.json"
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "microstructure_10s"


def candidate_model_paths_newest_first():
    if not CANDIDATE_ROOT.exists():
        return []
    return sorted(CANDIDATE_ROOT.glob("*/model.json"), reverse=True)


def path_is_allowed_for_symbol(path):
    path = Path(path).resolve()
    allowed_roots = [
        ACTIVE_MODEL_PATH.resolve(),
        CANDIDATE_ROOT.resolve(),
    ]
    if path == allowed_roots[0]:
        return True
    try:
        path.relative_to(allowed_roots[1])
        return path.name == "model.json"
    except ValueError:
        return False


def artifact_timestamp(artifact):
    return (
        artifact.get("created_at")
        or artifact.get("model_created_at")
        or artifact.get("trained_until_timestamp")
        or "missing"
    )


def scaler_exists(artifact):
    return isinstance(artifact.get("regression_target_scalers"), dict)


def artifact_event_only(artifact):
    return bool(artifact.get("event_only", False))


def artifact_rejection_notes(artifact):
    notes = []
    notes.extend(str(value) for value in artifact.get("rejection_reasons", []) if str(value).strip())
    selection = artifact.get("selection_reason", {})
    if isinstance(selection, dict):
        notes.extend(str(value) for value in selection.get("selection_notes", []) if str(value).strip())
    return notes


def infer_hard_rejected(artifact):
    if "hard_rejected" in artifact:
        return bool(artifact.get("hard_rejected"))
    joined = " | ".join(artifact_rejection_notes(artifact)).lower()
    hard_markers = [
        "no feature group passed",
        "regression sanity failure",
        "event saturation warning",
        "target groups disagree",
        "promotion is blocked",
    ]
    return any(marker in joined for marker in hard_markers)


def artifact_promotable(artifact):
    if "promotable" in artifact:
        return bool(artifact.get("promotable"))
    return not infer_hard_rejected(artifact)


def artifact_regression_sanity_status(artifact):
    if artifact.get("regression_sanity_status"):
        return str(artifact.get("regression_sanity_status")).strip().lower()
    return "fail" if artifact.get("regression_sanity_failures") else "ok"


def artifact_event_sanity_status(artifact):
    if artifact.get("event_sanity_status"):
        return str(artifact.get("event_sanity_status")).strip().lower()
    return "fail" if artifact.get("event_saturation_warnings") else "ok"


def model_symbol(artifact):
    return str(artifact.get("model_symbol", artifact.get("symbol", ""))).strip().upper()


def validate_model_candidate(path, expected_symbol):
    try:
        artifact = load_model(path)
    except Exception as error:
        return None, f"could not load artifact: {error}"
    if artifact is None:
        return None, "could not load artifact"
    loaded_symbol = model_symbol(artifact)
    if loaded_symbol != expected_symbol:
        return None, f"symbol mismatch: model={loaded_symbol or 'missing'} requested={expected_symbol}"
    event_only = artifact_event_only(artifact)
    if (not event_only) and not scaler_exists(artifact):
        return None, "missing regression_target_scalers"
    regression_columns = list(artifact.get("regression_target_columns", REGRESSION_TARGET_COLUMNS))
    if event_only:
        regression_columns = []
    else:
        try:
            validate_regression_scalers(artifact, regression_columns, allow_legacy=False)
        except ValueError as error:
            return None, str(error)
    hard_rejected = infer_hard_rejected(artifact)
    promotable = artifact_promotable(artifact)
    regression_status = artifact_regression_sanity_status(artifact)
    event_status = artifact_event_sanity_status(artifact)
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


def choose_model_artifact():
    diagnostics = {
        "selection_mode": MICRO_MODEL_SELECTION,
        "selected_path": None,
        "rejections": [],
    }
    if MODEL_PATH_ENV:
        path = Path(MODEL_PATH_ENV)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path_is_allowed_for_symbol(path):
            diagnostics["rejections"].append({"path": str(path), "reason": "path is not allowed for requested symbol"})
            return None, None, diagnostics
        artifact, reason = validate_model_candidate(path, SYMBOL)
        if artifact is None:
            diagnostics["rejections"].append({"path": str(path), "reason": reason})
            return None, None, diagnostics
        diagnostics["selected_path"] = str(path)
        return path, artifact, diagnostics
    if MICRO_MODEL_SELECTION not in VALID_MODEL_SELECTIONS:
        print(
            "Invalid MICRO_MODEL_SELECTION="
            f"{MICRO_MODEL_SELECTION!r}; expected latest_candidate or active_only."
        )
        return None, None, diagnostics
    if MICRO_MODEL_SELECTION == "active_only":
        if not ACTIVE_MODEL_PATH.exists():
            return None, None, diagnostics
        print(f"Micro model selection active_only path before loading: {ACTIVE_MODEL_PATH}")
        artifact, reason = validate_model_candidate(ACTIVE_MODEL_PATH, SYMBOL)
        if artifact is None:
            diagnostics["rejections"].append({"path": str(ACTIVE_MODEL_PATH), "reason": reason})
            return None, None, diagnostics
        diagnostics["selected_path"] = str(ACTIVE_MODEL_PATH)
        return ACTIVE_MODEL_PATH, artifact, diagnostics

    for candidate in candidate_model_paths_newest_first():
        print(f"Micro model selection latest_candidate path before loading: {candidate}")
        artifact, reason = validate_model_candidate(candidate, SYMBOL)
        if artifact is not None:
            diagnostics["selected_path"] = str(candidate)
            return candidate, artifact, diagnostics
        diagnostics["rejections"].append({"path": str(candidate), "reason": reason})
        print(f"Micro model selection skipped candidate: {candidate} reason={reason}")

    return None, None, diagnostics


def risk_level(probability):
    if probability >= 0.80:
        return "high"
    if probability >= 0.60:
        return "medium"
    return "low"


def direction_bias(row):
    up = float(row.get("prob_upside_scare_event_10s", 0.0))
    down = float(row.get("prob_downside_scare_event_10s", 0.0))
    if max(up, down) < 0.50:
        return "neutral / no strong scare signal"
    if up > down:
        return "upside pressure / upside scare risk"
    if down > up:
        return "downside pressure / downside scare risk"
    return "mixed pressure"


def format_regression_value(column, value):
    if value is None or not pd.notna(value):
        return "unavailable"
    value = float(value)
    if "log" in column and "depth" in column:
        return f"{value:.6g} log-depth-change"
    if "depth_change" in column and abs(value) > MICRO_DEPTH_DISPLAY_THRESHOLD:
        return "unstable/out-of-range"
    return percent(value)


def main():
    from show_live_market_stack import latest_micro_prediction as select_live_micro_prediction

    micro, live_status = select_live_micro_prediction(SYMBOL)
    if micro is None:
        print(f"No live-sane microstructure model found for {SYMBOL}")
        print(f"SYMBOL: {SYMBOL}")
        print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
        print(f"Snapshot input path: {INPUT_PATH}")
        print(f"MICRO_MODEL_SELECTION: {MICRO_MODEL_SELECTION}")
        print(f"Reason: {live_status}")
        print("10s_microstructure unavailable; fail-closed for this live row.")
        print("No trades are placed.")
        return
    model_path = Path(micro["model_path"])
    artifact = load_model(model_path)
    event_only = artifact_event_only(artifact)
    regression_columns = [] if event_only else list(artifact.get("regression_target_columns", REGRESSION_TARGET_COLUMNS))
    event_columns = list(artifact.get("event_target_columns", EVENT_TARGET_COLUMNS))
    latest_feature_timestamp = int(micro["timestamp"])
    staleness_seconds = max(0.0, (time.time() * 1000.0 - latest_feature_timestamp) / 1000.0)
    is_stale = staleness_seconds > MICRO_SHOW_MAX_STALENESS_SECONDS
    scare_probability = max(
        float(micro.get("prob_upside_scare_event_10s", 0.0)),
        float(micro.get("prob_downside_scare_event_10s", 0.0)),
    )
    print("Latest 10s microstructure paper prediction")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Snapshot input path: {INPUT_PATH}")
    print(f"Latest feature timestamp: {latest_feature_timestamp}")
    print(f"Latest feature staleness seconds: {staleness_seconds:.1f}")
    if is_stale:
        print(
            "WARNING: latest feature is stale beyond "
            f"MICRO_SHOW_MAX_STALENESS_SECONDS={MICRO_SHOW_MAX_STALENESS_SECONDS}. "
            "Current-live interpretation suppressed."
        )
    print(f"Model symbol from metadata: {micro.get('model_symbol', 'missing')}")
    print(f"Model path: {model_path}")
    print(f"MICRO_MODEL_SELECTION: {MICRO_MODEL_SELECTION}")
    print(f"Selected candidate path: {micro.get('selected_candidate_path')}")
    print(f"Model timestamp: {micro.get('model_timestamp', 'missing')}")
    print(f"regression_target_scalers exists: {micro.get('regression_target_scalers_exists', False)}")
    print(f"event_only: {event_only}")
    print(f"10s micro mode: {'event-only' if event_only else 'full'}")
    print(f"hard_rejected: {micro.get('hard_rejected', 'unknown')}")
    print(f"promotable: {micro.get('promotable', 'unknown')}")
    print(f"research_only_loaded: {micro.get('research_only_loaded', False)}")
    print(f"selected_feature_group: {micro.get('selected_feature_group', micro.get('feature_group_used', 'missing'))}")
    print(f"artifact_regression_sanity_status: {micro.get('artifact_regression_sanity_status', 'unknown')}")
    print(f"artifact_event_sanity_status: {micro.get('artifact_event_sanity_status', 'unknown')}")
    print(f"live_regression_sanity_status: {micro.get('live_regression_sanity_status', 'unknown')}")
    print(f"live_event_sanity_status: {micro.get('live_event_sanity_status', 'unknown')}")
    print(f"live_event_saturation_fraction: {micro.get('live_event_saturation_fraction', np.nan):.2%}")
    print(f"live_regression_clipped_count: {micro.get('live_regression_clipped_count', 'unknown')}")
    if micro.get("model_selection_rejections"):
        print("Rejected hard/bad artifact candidates:")
        for item in micro.get("model_selection_rejections", []):
            print(f"- {item.get('path')}: {item.get('reason')}")
    if micro.get("model_selection_live_rejections"):
        print("Skipped-live candidates:")
        for item in micro.get("model_selection_live_rejections", []):
            print(f"- {item.get('path')}: {item.get('reason')}")
    if micro.get("rejection_reasons"):
        print(f"artifact rejection_reasons: {'; '.join(micro.get('rejection_reasons', [])[:8])}")
    print(f"Model id: {micro.get('model_id', 'missing')}")
    print(f"Model trained_until_timestamp: {artifact.get('trained_until_timestamp', 'missing')}")
    print(f"Model feature count: {artifact.get('feature_count', len(artifact.get('feature_columns', [])))}")
    print(f"feature_schema_hash: {artifact.get('feature_schema_hash', 'missing')}")
    print(f"MICRO_EVENT_PROB_TEMPERATURE: {MICRO_EVENT_PROB_TEMPERATURE}")
    print("No trades are placed.")
    print("\nScare/event probabilities")
    for column in event_columns:
        if f"prob_{column}" in micro:
            print(f"- {column}: {percent(micro[f'prob_{column}'])}")
    print("\n10s regression forecasts")
    if event_only:
        print("- hidden/unavailable because event_only=true")
    else:
        for column in regression_columns:
            if f"pred_{column}" in micro:
                print(f"- {column}: {format_regression_value(column, micro[f'pred_{column}'])}")
    if not is_stale:
        print("\nInterpretation")
        print(f"- Bias: {direction_bias(micro)}")
        print(f"- Overall scare risk: {risk_level(scare_probability)} ({percent(scare_probability)})")
        print(
            "- Slippage risk note: watch spread expansion, bid/ask depth drops, "
            "and aggressive buy/sell burst probabilities together."
        )
        print("- Paper-only event scoring; no buy/sell command is produced.")
    else:
        print("\nInterpretation suppressed because the live feature row is stale.")
    return

    model_path, artifact, selection_diagnostics = choose_model_artifact()
    if model_path is None or artifact is None:
        print(f"No microstructure model found for {SYMBOL}")
        print(f"SYMBOL: {SYMBOL}")
        print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
        print(f"Snapshot input path: {INPUT_PATH}")
        print(f"MICRO_MODEL_SELECTION: {MICRO_MODEL_SELECTION}")
        if selection_diagnostics.get("rejections"):
            print("Rejected micro model candidates:")
            for item in selection_diagnostics["rejections"]:
                print(f"- {item['path']}: {item['reason']}")
        print("No trades are placed.")
        return
    loaded_model_symbol = model_symbol(artifact)
    if loaded_model_symbol != SYMBOL:
        print(
            "Model symbol mismatch. "
            f"Requested SYMBOL={SYMBOL}, model_symbol={loaded_model_symbol or 'missing'}."
        )
        print(f"Rejected model path: {model_path}")
        print("No prediction shown. No trades are placed.")
        return

    try:
        snapshots = load_snapshot_rows(INPUT_PATH)
    except FileNotFoundError:
        print(f"Missing snapshot input file: {INPUT_PATH}")
        print(f"SYMBOL: {SYMBOL}")
        print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
        print("No prediction shown. No trades are placed.")
        return
    except ValueError as error:
        print(f"Invalid snapshot input file: {error}")
        print(f"Snapshot input path: {INPUT_PATH}")
        print("No prediction shown. No trades are placed.")
        return
    feature_frame, snapshot_step_seconds = build_latest_feature_only(snapshots)
    if len(feature_frame) == 0:
        print("No feature-ready snapshot rows are available yet.")
        print(f"SYMBOL: {SYMBOL}")
        print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
        print(f"Snapshot input path: {INPUT_PATH}")
        print("No prediction shown. No trades are placed.")
        return
    feature_frame, _ = attach_hierarchical_context(
        feature_frame,
        PROJECT_ROOT,
        SYMBOL,
        layers=("htf", "regime15", "regime30", "flow3m", "flow1s"),
        as_model_features=True,
    )

    model_feature_columns = list(artifact.get("feature_columns", []))
    latest = feature_frame.tail(1).copy()
    current_feature_columns = get_micro_feature_columns(latest)
    schema_diagnostics = micro_schema_diagnostics(
        model_feature_columns,
        current_feature_columns,
    )
    latest = add_missing_optional_context_columns(latest, model_feature_columns)
    latest = fill_optional_context_feature_values(latest, model_feature_columns)
    model_schema_hash = artifact.get("feature_schema_hash", "")
    saved_columns_hash = feature_schema_hash(model_feature_columns)
    if model_schema_hash != saved_columns_hash:
        print("Feature schema mismatch.")
        print(f"SYMBOL: {SYMBOL}")
        print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
        print(f"Snapshot input path: {INPUT_PATH}")
        print(f"Model symbol from metadata: {loaded_model_symbol}")
        print(f"Model path: {model_path}")
        print(f"Model feature_schema_hash: {model_schema_hash or 'missing'}")
        print(f"Saved feature_columns hash: {saved_columns_hash}")
        print_micro_schema_diagnostics(schema_diagnostics)
        print("Rebuild micro training rows and retrain the model for this symbol/venue.")
        print("No prediction shown. No trades are placed.")
        return
    model_feature_count = int(artifact.get("feature_count", len(model_feature_columns)))
    if model_feature_count != len(model_feature_columns):
        print(
            "Feature count mismatch. "
            f"model={model_feature_count} saved_columns={len(model_feature_columns)}."
        )
        print(f"Model path: {model_path}")
        print("Rebuild micro training rows and retrain the model for this symbol/venue.")
        print("No prediction shown. No trades are placed.")
        return
    missing_required = [
        column
        for column in required_micro_feature_columns(model_feature_columns)
        if column not in current_feature_columns
    ]
    if missing_required:
        print("Feature schema mismatch.")
        print(f"SYMBOL: {SYMBOL}")
        print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
        print(f"Snapshot input path: {INPUT_PATH}")
        print(f"Model path: {model_path}")
        print("Missing required real microstructure feature columns.")
        print_micro_schema_diagnostics(schema_diagnostics)
        print("No prediction shown. No trades are placed.")
        return

    event_only = artifact_event_only(artifact)
    regression_columns = [] if event_only else list(artifact.get("regression_target_columns", REGRESSION_TARGET_COLUMNS))
    event_columns = list(artifact.get("event_target_columns", EVENT_TARGET_COLUMNS))
    try:
        if not event_only:
            validate_regression_scalers(artifact, regression_columns, allow_legacy=False)
        regression, event_probabilities = predict_with_artifact(
            artifact,
            latest,
            event_temperature=MICRO_EVENT_PROB_TEMPERATURE,
        )
    except ValueError as error:
        print("Microstructure model output scaling validation failed.")
        print(f"Reason: {error}")
        print(f"Model path: {model_path}")
        print("Retrain the 10s microstructure model before showing live predictions.")
        print("No prediction shown. No trades are placed.")
        return
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
    result = {
        "timestamp": int(latest["timestamp"].iloc[0]),
        "time": latest["time"].iloc[0],
        "_regression_columns": regression_columns,
        "_event_columns": event_columns,
    }
    for index, column in enumerate(regression_columns):
        result[f"raw_pred_{column}"] = float(regression[0, index])
        result[f"pred_{column}"] = float(regression_sanity["clipped_values"].get(column, regression[0, index]))
    for index, column in enumerate(event_columns):
        result[f"prob_{column}"] = float(event_probabilities[0, index])

    result = pd.Series(result)
    scare_probability = max(
        float(result.get("prob_upside_scare_event_10s", 0.0)),
        float(result.get("prob_downside_scare_event_10s", 0.0)),
    )

    latest_snapshot_timestamp = int(pd.to_numeric(snapshots["timestamp"], errors="coerce").max())
    latest_feature_timestamp = int(result["timestamp"])
    staleness_seconds = max(0.0, (time.time() * 1000.0 - latest_feature_timestamp) / 1000.0)
    is_stale = staleness_seconds > MICRO_SHOW_MAX_STALENESS_SECONDS

    print("Latest 10s microstructure paper prediction")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Snapshot input path: {INPUT_PATH}")
    print(f"Latest snapshot timestamp: {latest_snapshot_timestamp}")
    print(f"Latest feature timestamp: {latest_feature_timestamp}")
    print(f"Latest feature staleness seconds: {staleness_seconds:.1f}")
    if is_stale:
        print(
            "WARNING: latest feature is stale beyond "
            f"MICRO_SHOW_MAX_STALENESS_SECONDS={MICRO_SHOW_MAX_STALENESS_SECONDS}. "
            "Current-live interpretation suppressed."
        )
    print(f"Model symbol from metadata: {loaded_model_symbol}")
    print(f"Model path: {model_path}")
    print(f"MICRO_MODEL_SELECTION: {MICRO_MODEL_SELECTION}")
    print(f"Selected candidate path: {selection_diagnostics.get('selected_path')}")
    print(f"Model timestamp: {artifact_timestamp(artifact)}")
    print(f"regression_target_scalers exists: {scaler_exists(artifact)}")
    print(f"event_only: {event_only}")
    print(f"hard_rejected: {infer_hard_rejected(artifact)}")
    print(f"promotable: {artifact_promotable(artifact)}")
    print(f"regression_sanity_status: {artifact_regression_sanity_status(artifact)}")
    print(f"event_sanity_status: {artifact_event_sanity_status(artifact)}")
    print(f"research_only_loaded: {MICRO_ALLOW_REJECTED_CANDIDATES and (infer_hard_rejected(artifact) or not artifact_promotable(artifact))}")
    if artifact_rejection_notes(artifact):
        print(f"rejection_reasons: {'; '.join(artifact_rejection_notes(artifact)[:8])}")
    print(f"selected_feature_group: {artifact.get('selected_feature_group', artifact.get('feature_group_used', 'missing'))}")
    print(f"10s micro mode: {'event-only' if event_only else 'full'}")
    if selection_diagnostics.get("rejections"):
        print("Rejected newer/bad micro candidates:")
        for item in selection_diagnostics["rejections"]:
            print(f"- {item['path']}: {item['reason']}")
    print(f"Model id: {artifact.get('model_id', 'missing')}")
    print(f"Model trained_until_timestamp: {artifact.get('trained_until_timestamp', 'missing')}")
    print(f"Model feature count: {model_feature_count}")
    print(f"feature_schema_hash: {model_schema_hash}")
    print(f"MICRO_EVENT_PROB_TEMPERATURE: {MICRO_EVENT_PROB_TEMPERATURE}")
    print(f"Current available feature count: {len(current_feature_columns)}")
    print("Schema validation")
    print(f"- required missing columns: {len(schema_diagnostics['required_missing_columns'])}")
    print(f"- optional missing columns filled: {len(schema_diagnostics['optional_missing_columns_filled'])}")
    print(f"- extra current columns ignored: {len(schema_diagnostics['extra_in_current'])}")
    print(f"- artifact feature_schema_hash: {schema_diagnostics['artifact_feature_schema_hash']}")
    print(f"- current required-feature hash: {schema_diagnostics['current_required_feature_hash']}")
    print(f"Inferred snapshot step seconds: {snapshot_step_seconds:.3g}")
    print(f"Latest feature time: {result['time']}")
    print("No trades are placed.")
    print("Regression sanity")
    if event_only:
        print("- status: not_applicable (event-only artifact)")
    else:
        print(f"- status: {'fail' if regression_sanity['failed'] else ('warning' if regression_sanity['warning'] else 'ok')}")
        if regression_sanity["warning"]:
            print("WARNING: regression output clipped for display; model may be extrapolating.")
            for warning in regression_sanity["warnings"]:
                print(f"- {warning}")
    print("Event probability sanity")
    print(f"- saturated fraction: {event_sanity['fraction_saturated']:.2%}")
    print(f"- status: {'fail' if event_sanity['failed'] else ('warning' if event_sanity['warning'] else 'ok')}")
    if event_sanity["warning"]:
        print("WARNING: event probabilities are saturated.")
        print(f"- saturated events: {', '.join(event_sanity['saturated_events']) if event_sanity['saturated_events'] else 'none'}")

    print("\nScare/event probabilities")
    for column in event_columns:
        print(f"- {column}: {percent(result[f'prob_{column}'])}")

    if not event_only:
        print("\n10s regression forecasts")
        for column in result.get("_regression_columns", REGRESSION_TARGET_COLUMNS):
            if f"pred_{column}" in result:
                print(f"- {column}: {format_regression_value(column, result[f'pred_{column}'])}")
    else:
        print("\n10s regression forecasts")
        print("- hidden/unavailable because event_only=true")

    if not is_stale and not regression_sanity["failed"] and not event_sanity["failed"]:
        print("\nInterpretation")
        print(f"- Bias: {direction_bias(result)}")
        print(f"- Overall scare risk: {risk_level(scare_probability)} ({percent(scare_probability)})")
        print(
            "- Slippage risk note: watch spread expansion, bid/ask depth drops, "
            "and aggressive buy/sell burst probabilities together."
        )
        print("- Paper-only event scoring; no buy/sell command is produced.")
    else:
        reasons = []
        if is_stale:
            reasons.append("latest feature is stale")
        if regression_sanity["failed"]:
            reasons.append("regression sanity failed")
        if event_sanity["failed"]:
            reasons.append("event probability saturation is extreme")
        print(f"\nInterpretation suppressed because {', '.join(reasons)}.")
        print("- Paper-only event scoring; no buy/sell command is produced.")


if __name__ == "__main__":
    main()
