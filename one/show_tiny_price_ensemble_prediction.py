import json
import os
import re
import secrets
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from show_tiny_price_prediction import (
    atomic_write_csv,
    build_current_features,
    load_json,
    read_csv,
)
from train_tiny_price_model import predict_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower() or "legacy"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))

MOVE_TARGET_SPEC = os.getenv("PRICE_TINY_ENSEMBLE_MOVE_TARGET_SPEC", "move_before_adverse_30s").strip()
INSTABILITY_TARGET_SPEC = os.getenv("PRICE_TINY_ENSEMBLE_INSTABILITY_TARGET_SPEC", "instability_30s").strip()
DIRECTION_TARGET_SPEC = os.getenv("PRICE_TINY_ENSEMBLE_DIRECTION_TARGET_SPEC", "direction_30s").strip()
RETURN_TARGET_SPEC = os.getenv("PRICE_TINY_ENSEMBLE_RETURN_TARGET_SPEC", "return_bps_30s").strip()

MOVE_CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD",
        os.getenv("PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE", "0.70"),
    )
)
INSTABILITY_MAX_PROBABILITY = float(
    os.getenv(
        "PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD",
        os.getenv("PRICE_TINY_ENSEMBLE_INSTABILITY_MAX_PROBABILITY", "0.70"),
    )
)
ENABLE_DIRECTION_MODEL = os.getenv("PRICE_TINY_ENSEMBLE_ENABLE_DIRECTION", "true").strip().lower() in {"1", "true", "yes", "y"}
REQUIRE_DIRECTION_AGREEMENT = os.getenv("PRICE_TINY_ENSEMBLE_REQUIRE_DIRECTION_AGREEMENT", "false").strip().lower() in {"1", "true", "yes", "y"}
ENABLE_REGRESSION_MODEL = os.getenv("PRICE_TINY_ENSEMBLE_ENABLE_REGRESSION", "false").strip().lower() in {"1", "true", "yes", "y"}
REGRESSION_MIN_ABS_BPS = float(os.getenv("PRICE_TINY_ENSEMBLE_REGRESSION_MIN_ABS_BPS", "1"))
MAX_SNAPSHOT_AGE_SECONDS = float(
    os.getenv(
        "PRICE_TINY_ENSEMBLE_MAX_SNAPSHOT_AGE_SECONDS",
        os.getenv("PRICE_TINY_MAX_SNAPSHOT_AGE_SECONDS", "15"),
    )
)
ENSEMBLE_RULE_VERSION = os.getenv("PRICE_TINY_ENSEMBLE_RULE_VERSION", "tiny_price_ensemble_v1").strip()


def slugify(value):
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "unknown"


def generated_run_id():
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3)
    return (
        f"{timestamp}_{slugify(SYMBOL)}_{slugify(PRIMARY_VENUE)}_"
        f"move{MOVE_CONFIDENCE_THRESHOLD:.2f}_inst{INSTABILITY_MAX_PROBABILITY:.2f}_{suffix}"
    )


RUN_ID = os.getenv("PRICE_TINY_ENSEMBLE_RUN_ID", "").strip() or generated_run_id()

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
SNAPSHOT_PATH = VENUE_DIR / f"{SYMBOL}_10s_flow.csv"
OUTPUT_PATH = Path(
    os.getenv(
        "PRICE_TINY_ENSEMBLE_LIVE_PREDICTIONS_PATH",
        VENUE_DIR / f"{SYMBOL}_tiny_price_ensemble_live_predictions.csv",
    )
)
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH

CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "tiny_price" / PRIMARY_VENUE
SELECTED_ROOT = PROJECT_ROOT / "models" / "selected" / SYMBOL / "tiny_price" / PRIMARY_VENUE
SELECTED_MODEL_PATH = SELECTED_ROOT / "selected_model.json"
CANDIDATE_REGISTRY_PATH = SELECTED_ROOT / "candidate_registry.json"


TARGET_CONFIGS = {
    "move": {
        "target_spec": MOVE_TARGET_SPEC,
        "required": True,
        "env_paths": [
            "PRICE_TINY_ENSEMBLE_MOVE_MODEL_PATH",
            "PRICE_TINY_MOVE_MODEL_PATH",
            "PRICE_TINY_MOVE_BEFORE_ADVERSE_MODEL_PATH",
        ],
    },
    "instability": {
        "target_spec": INSTABILITY_TARGET_SPEC,
        "required": True,
        "env_paths": [
            "PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_PATH",
            "PRICE_TINY_INSTABILITY_MODEL_PATH",
        ],
    },
    "direction": {
        "target_spec": DIRECTION_TARGET_SPEC,
        "required": False,
        "env_paths": [
            "PRICE_TINY_ENSEMBLE_DIRECTION_MODEL_PATH",
            "PRICE_TINY_DIRECTION_MODEL_PATH",
        ],
    },
    "return": {
        "target_spec": RETURN_TARGET_SPEC,
        "required": False,
        "env_paths": [
            "PRICE_TINY_ENSEMBLE_RETURN_MODEL_PATH",
            "PRICE_TINY_RETURN_MODEL_PATH",
            "PRICE_TINY_REGRESSION_MODEL_PATH",
        ],
    },
}


def resolve_path(value):
    if not value:
        return None
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json_if_exists(path):
    path = resolve_path(path)
    if path is None or not path.exists():
        return None
    return load_json(path)


def atomic_append_prediction(row):
    existing = read_csv(OUTPUT_PATH)
    new_frame = pd.DataFrame([row])
    output = pd.concat([existing, new_frame], ignore_index=True) if len(existing) else new_frame
    output["timestamp"] = pd.to_numeric(output["timestamp"], errors="coerce")
    output = output.dropna(subset=["timestamp"])
    dedupe_columns = ["timestamp", "symbol", "primary_venue"]
    if "run_id" in output.columns:
        output["run_id"] = output["run_id"].fillna("").astype(str)
        dedupe_columns.append("run_id")
    output = output.drop_duplicates(dedupe_columns, keep="last")
    output = output.sort_values("timestamp")
    atomic_write_csv(output, OUTPUT_PATH)


def artifact_target_name(artifact):
    target_spec = artifact.get("target_spec", "")
    if isinstance(target_spec, dict):
        return str(target_spec.get("name", "")).strip()
    return str(target_spec or artifact.get("target_spec_name", "")).strip()


def artifact_matches(artifact, target_spec):
    if not artifact:
        return False, "empty_artifact"
    if str(artifact.get("symbol", "")).upper() != SYMBOL:
        return False, f"symbol_mismatch:{artifact.get('symbol')}"
    if str(artifact.get("primary_venue", "")).lower() != PRIMARY_VENUE:
        return False, f"venue_mismatch:{artifact.get('primary_venue')}"
    artifact_target = artifact_target_name(artifact)
    if artifact_target != target_spec:
        return False, f"target_spec_mismatch:{artifact_target}"
    if "feature_columns" not in artifact or "models" not in artifact or "selected_model_name" not in artifact:
        return False, "not_a_tiny_price_model_artifact"
    return True, "ok"


def explicit_model_path(config):
    for env_name in config["env_paths"]:
        path = resolve_path(os.getenv(env_name, "").strip())
        if path and path.exists():
            return path, env_name
    scoped_name = f"PRICE_TINY_ENSEMBLE_{config['target_spec'].upper().replace('-', '_').replace('/', '_')}_MODEL_PATH"
    scoped_name = scoped_name.replace(".", "_")
    path = resolve_path(os.getenv(scoped_name, "").strip())
    if path and path.exists():
        return path, scoped_name
    return None, ""


def selected_registry_paths():
    paths = []
    selected = load_json_if_exists(SELECTED_MODEL_PATH)
    if selected:
        selected_path = selected.get("champion_model_path") or selected.get("model_path")
        if selected_path:
            paths.append((resolve_path(selected_path), "selected_model"))
    registry = load_json_if_exists(CANDIDATE_REGISTRY_PATH)
    if registry:
        champion = resolve_path(registry.get("champion_model_path", ""))
        if champion:
            paths.append((champion, "candidate_registry_champion"))
        for index, item in enumerate(registry.get("challengers", []), start=1):
            if isinstance(item, dict) and item.get("retired"):
                continue
            raw_path = item.get("model_path") if isinstance(item, dict) else item
            path = resolve_path(raw_path)
            if path:
                paths.append((path, f"candidate_registry_challenger_{index}"))
    return paths


def candidate_paths_newest_first():
    if not CANDIDATE_ROOT.exists():
        return []
    return [(path, "candidate_archive") for path in sorted(CANDIDATE_ROOT.glob("*/model.json"), key=lambda value: value.parent.name, reverse=True)]


def resolve_model_for_target(key, config):
    target_spec = config["target_spec"]
    skipped = []
    explicit_path, explicit_source = explicit_model_path(config)
    search_paths = []
    if explicit_path:
        search_paths.append((explicit_path, f"explicit:{explicit_source}"))
    search_paths.extend(selected_registry_paths())
    search_paths.extend(candidate_paths_newest_first())
    seen = set()
    for path, source in search_paths:
        if path is None:
            continue
        path = resolve_path(path)
        path_key = str(path).lower()
        if path_key in seen:
            continue
        seen.add(path_key)
        if not path.exists():
            skipped.append({"path": str(path), "source": source, "reason": "missing_path"})
            continue
        try:
            artifact = load_json(path)
        except Exception as error:
            skipped.append({"path": str(path), "source": source, "reason": f"load_error:{error}"})
            continue
        ok, reason = artifact_matches(artifact, target_spec)
        if not ok:
            skipped.append({"path": str(path), "source": source, "reason": reason})
            continue
        return {
            "key": key,
            "target_spec": target_spec,
            "path": path,
            "source": source,
            "artifact": artifact,
            "skipped": skipped,
        }
    return {
        "key": key,
        "target_spec": target_spec,
        "path": None,
        "source": "",
        "artifact": None,
        "skipped": skipped,
    }


def latest_snapshot_diagnostics(snapshots):
    now_ms = int(time.time() * 1000)
    if len(snapshots) == 0 or "timestamp" not in snapshots.columns:
        return {
            "latest_snapshot_timestamp": np.nan,
            "now_timestamp": now_ms,
            "snapshot_age_seconds": np.inf,
            "snapshot_freshness": "stale",
            "snapshot_freshness_reason": "missing_snapshot_file_or_timestamp",
        }
    timestamps = pd.to_numeric(snapshots["timestamp"], errors="coerce").dropna()
    if len(timestamps) == 0:
        return {
            "latest_snapshot_timestamp": np.nan,
            "now_timestamp": now_ms,
            "snapshot_age_seconds": np.inf,
            "snapshot_freshness": "stale",
            "snapshot_freshness_reason": "no_numeric_snapshot_timestamp",
        }
    latest = int(timestamps.max())
    age = max(0.0, (now_ms - latest) / 1000.0)
    return {
        "latest_snapshot_timestamp": latest,
        "now_timestamp": now_ms,
        "snapshot_age_seconds": float(age),
        "snapshot_freshness": "fresh" if age <= MAX_SNAPSHOT_AGE_SECONDS else "stale",
        "snapshot_freshness_reason": "" if age <= MAX_SNAPSHOT_AGE_SECONDS else f"snapshot_age_seconds {age:.1f} > max {MAX_SNAPSHOT_AGE_SECONDS:.1f}",
    }


def predict_resolved_model(resolved, snapshots):
    artifact = resolved["artifact"]
    feature_row = build_current_features(snapshots, artifact)
    if len(feature_row) == 0:
        raise RuntimeError("no current feature row could be built")
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
    probs = np.asarray(probabilities[0], dtype=np.float64) if np.ndim(probabilities) == 2 else np.asarray([np.nan, np.nan, np.nan])
    return {
        "timestamp": int(feature_row["timestamp"].iloc[0]),
        "time": str(feature_row["time"].iloc[0]),
        "direction": int(pred_direction[0]),
        "confidence": float(confidence[0]),
        "prob_down": float(probs[0]) if len(probs) > 0 and np.isfinite(probs[0]) else np.nan,
        "prob_flat": float(probs[1]) if len(probs) > 1 and np.isfinite(probs[1]) else np.nan,
        "prob_up": float(probs[2]) if len(probs) > 2 and np.isfinite(probs[2]) else np.nan,
        "predicted_return_bps": float(pred_delta[0]),
        "predicted_log_return": float(pred_log[0]),
        "model_id": artifact.get("model_id", ""),
        "model_path": str(resolved["path"]),
        "model_source": resolved["source"],
        "selected_model_name": selected,
        "target_spec": artifact_target_name(artifact),
        "feature_schema_hash": artifact.get("feature_schema_hash", ""),
        "feature_spec_name": artifact.get("feature_spec", {}).get("name", artifact.get("feature_set_name", "")) if isinstance(artifact.get("feature_spec", {}), dict) else artifact.get("feature_set_name", ""),
        "horizon_seconds": int(artifact.get("horizon_seconds", 0) or 0),
        "confidence_type": artifact.get("confidence_type", "class_probability"),
    }


def instability_probability(prediction):
    if prediction is None:
        return np.nan
    if np.isfinite(prediction.get("prob_up", np.nan)):
        return float(prediction["prob_up"])
    if int(prediction.get("direction", 0)) > 0:
        return float(prediction.get("confidence", 0.0))
    return max(0.0, 1.0 - float(prediction.get("confidence", 0.0)))


def side_text(direction):
    if direction > 0:
        return "long"
    if direction < 0:
        return "short"
    return "no_trade"


def make_decision(predictions, resolutions, snapshot_diag):
    reasons = []
    move = predictions.get("move")
    instability = predictions.get("instability")
    direction = predictions.get("direction")
    regression = predictions.get("return")
    if snapshot_diag["snapshot_freshness"] != "fresh":
        reasons.append(f"snapshot_stale:{snapshot_diag['snapshot_freshness_reason']}")
    if move is None:
        reasons.append("missing_move_before_adverse_model")
    if instability is None:
        reasons.append("missing_instability_model")
    if reasons:
        return "no_trade", 0, ";".join(reasons)

    move_schema_hash = move.get("feature_schema_hash", "")
    instability_schema_hash = instability.get("feature_schema_hash", "")
    if not move_schema_hash or not instability_schema_hash or move_schema_hash != instability_schema_hash:
        reasons.append("required_schema_mismatch")

    move_direction = int(move["direction"])
    move_confidence = float(move["confidence"])
    instability_prob = instability_probability(instability)
    if move_direction == 0:
        reasons.append("move_before_adverse_direction_neutral")
    if move_confidence < MOVE_CONFIDENCE_THRESHOLD:
        reasons.append(f"move_confidence_below_{MOVE_CONFIDENCE_THRESHOLD:.2f}")
    if not np.isfinite(instability_prob):
        reasons.append("instability_probability_missing")
    elif instability_prob >= INSTABILITY_MAX_PROBABILITY:
        reasons.append(f"instability_probability_ge_{INSTABILITY_MAX_PROBABILITY:.2f}")

    if REQUIRE_DIRECTION_AGREEMENT:
        if direction is None:
            reasons.append("direction_model_required_but_missing")
        elif direction.get("feature_schema_hash", "") != move_schema_hash:
            reasons.append("optional_direction_schema_mismatch_used_in_gate")
        elif int(direction["direction"]) == 0:
            reasons.append("direction_model_neutral")
        elif int(direction["direction"]) != move_direction:
            reasons.append("direction_model_disagrees")

    if ENABLE_REGRESSION_MODEL:
        if regression is None:
            reasons.append("regression_model_required_but_missing")
        elif regression.get("feature_schema_hash", "") != move_schema_hash:
            reasons.append("optional_regression_schema_mismatch_used_in_gate")
        else:
            reg_return = float(regression["predicted_return_bps"])
            reg_direction = 1 if reg_return > 0 else (-1 if reg_return < 0 else 0)
            if abs(reg_return) < REGRESSION_MIN_ABS_BPS:
                reasons.append(f"regression_abs_return_lt_{REGRESSION_MIN_ABS_BPS:.2f}bps")
            if reg_direction == 0:
                reasons.append("regression_direction_neutral")
            elif reg_direction != move_direction:
                reasons.append("regression_sign_disagrees")

    if reasons:
        return "no_trade", 0, ";".join(reasons)
    return side_text(move_direction), move_direction, "passed_default_paper_rule"


def row_from_predictions(predictions, resolutions, snapshot_diag, final_signal, final_direction, reason):
    move = predictions.get("move") or {}
    instability = predictions.get("instability") or {}
    direction = predictions.get("direction") or {}
    regression = predictions.get("return") or {}
    schema_hashes = {
        key: prediction.get("feature_schema_hash", "")
        for key, prediction in predictions.items()
        if prediction is not None
    }
    move_schema_hash = move.get("feature_schema_hash", "")
    instability_schema_hash = instability.get("feature_schema_hash", "")
    direction_schema_hash = direction.get("feature_schema_hash", "")
    regression_schema_hash = regression.get("feature_schema_hash", "")
    required_schema_match = bool(move_schema_hash and instability_schema_hash and move_schema_hash == instability_schema_hash)
    optional_direction_schema_match = (
        True
        if not direction_schema_hash
        else bool(move_schema_hash and direction_schema_hash == move_schema_hash)
    )
    optional_regression_schema_match = (
        True
        if not regression_schema_hash
        else bool(move_schema_hash and regression_schema_hash == move_schema_hash)
    )
    optional_schema_match = optional_direction_schema_match and optional_regression_schema_match
    optional_direction_ignored_reason = ""
    if direction_schema_hash and not optional_direction_schema_match:
        optional_direction_ignored_reason = (
            "optional_direction_schema_mismatch_used_in_gate"
            if REQUIRE_DIRECTION_AGREEMENT
            else "optional_direction_schema_mismatch_diagnostic_only"
        )
    elif not REQUIRE_DIRECTION_AGREEMENT:
        optional_direction_ignored_reason = "optional_direction_diagnostic_only"
    optional_regression_ignored_reason = ""
    if regression_schema_hash and not optional_regression_schema_match:
        optional_regression_ignored_reason = (
            "optional_regression_schema_mismatch_used_in_gate"
            if ENABLE_REGRESSION_MODEL
            else "optional_regression_schema_mismatch_diagnostic_only"
        )
    elif not ENABLE_REGRESSION_MODEL:
        optional_regression_ignored_reason = "optional_regression_disabled"
    timestamp = int(move.get("timestamp") or instability.get("timestamp") or direction.get("timestamp") or regression.get("timestamp") or snapshot_diag["latest_snapshot_timestamp"])
    return {
        "timestamp": timestamp,
        "time": move.get("time") or instability.get("time") or direction.get("time") or regression.get("time") or "",
        "logged_at_timestamp": int(time.time() * 1000),
        "run_id": RUN_ID,
        "ensemble_rule_version": ENSEMBLE_RULE_VERSION,
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE,
        "venue": PRIMARY_VENUE,
        "active_move_confidence_threshold": MOVE_CONFIDENCE_THRESHOLD,
        "active_instability_threshold": INSTABILITY_MAX_PROBABILITY,
        "regression_enabled": ENABLE_REGRESSION_MODEL,
        "regression_min_abs_bps": REGRESSION_MIN_ABS_BPS,
        "move_model_id": move.get("model_id", ""),
        "move_model_path": move.get("model_path", ""),
        "move_schema_hash": move_schema_hash,
        "instability_model_id": instability.get("model_id", ""),
        "instability_model_path": instability.get("model_path", ""),
        "instability_schema_hash": instability_schema_hash,
        "direction_model_id": direction.get("model_id", ""),
        "direction_model_path": direction.get("model_path", ""),
        "direction_schema_hash": direction_schema_hash,
        "regression_model_id": regression.get("model_id", ""),
        "regression_model_path": regression.get("model_path", ""),
        "regression_schema_hash": regression_schema_hash,
        "required_schema_match": required_schema_match,
        "optional_schema_match": optional_schema_match,
        "optional_direction_schema_match": optional_direction_schema_match,
        "optional_regression_schema_match": optional_regression_schema_match,
        "optional_direction_ignored_reason": optional_direction_ignored_reason,
        "optional_regression_ignored_reason": optional_regression_ignored_reason,
        "optional_direction_used_in_gate": REQUIRE_DIRECTION_AGREEMENT,
        "optional_regression_used_in_gate": ENABLE_REGRESSION_MODEL,
        "move_before_adverse_direction": int(move.get("direction", 0) or 0),
        "move_before_adverse_confidence": float(move.get("confidence", np.nan)),
        "move_before_adverse_model_id": move.get("model_id", ""),
        "move_before_adverse_model_path": move.get("model_path", ""),
        "move_before_adverse_schema_hash": move_schema_hash,
        "instability_probability": float(instability_probability(instability)) if instability else np.nan,
        "instability_direction": int(instability.get("direction", 0) or 0),
        "instability_confidence": float(instability.get("confidence", np.nan)),
        "instability_model_id": instability.get("model_id", ""),
        "instability_model_path": instability.get("model_path", ""),
        "instability_schema_hash": instability_schema_hash,
        "direction_model_direction": int(direction.get("direction", 0) or 0),
        "direction_model_confidence": float(direction.get("confidence", np.nan)),
        "direction_model_available": bool(direction),
        "direction_model_id": direction.get("model_id", ""),
        "direction_model_path": direction.get("model_path", ""),
        "direction_model_schema_hash": direction_schema_hash,
        "regression_predicted_return_bps": float(regression.get("predicted_return_bps", np.nan)),
        "regression_direction": 1 if float(regression.get("predicted_return_bps", 0.0) or 0.0) > 0 else (-1 if float(regression.get("predicted_return_bps", 0.0) or 0.0) < 0 else 0),
        "regression_model_available": bool(regression),
        "regression_model_id": regression.get("model_id", ""),
        "regression_model_path": regression.get("model_path", ""),
        "regression_schema_hash": regression_schema_hash,
        "final_paper_signal": final_signal,
        "final_paper_signal_direction": int(final_direction),
        "no_trade_reason": "" if final_signal != "no_trade" else reason,
        "decision_reason": reason,
        "move_confidence_threshold": MOVE_CONFIDENCE_THRESHOLD,
        "instability_max_probability": INSTABILITY_MAX_PROBABILITY,
        "regression_enabled": ENABLE_REGRESSION_MODEL,
        "regression_min_abs_bps": REGRESSION_MIN_ABS_BPS,
        "direction_agreement_required": REQUIRE_DIRECTION_AGREEMENT,
        "snapshot_freshness": snapshot_diag["snapshot_freshness"],
        "snapshot_freshness_reason": snapshot_diag["snapshot_freshness_reason"],
        "latest_snapshot_timestamp": snapshot_diag["latest_snapshot_timestamp"],
        "now_timestamp": snapshot_diag["now_timestamp"],
        "snapshot_age_seconds": snapshot_diag["snapshot_age_seconds"],
        "max_snapshot_age_seconds": MAX_SNAPSHOT_AGE_SECONDS,
        "all_model_schema_hashes": json.dumps(schema_hashes, sort_keys=True),
        "schema_hashes_match": required_schema_match,
        "all_model_schema_hashes_match": len(set(value for value in schema_hashes.values() if value)) <= 1,
        "paper_only": True,
    }


def main():
    print("Tiny-price live paper ensemble prediction")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE}")
    print(f"RUN_ID: {RUN_ID}")
    print(f"ENSEMBLE_RULE_VERSION: {ENSEMBLE_RULE_VERSION}")
    print(f"Active move confidence threshold: {MOVE_CONFIDENCE_THRESHOLD:.2f}")
    print(f"Active instability threshold: {INSTABILITY_MAX_PROBABILITY:.2f}")
    print(f"Regression enabled: {ENABLE_REGRESSION_MODEL}")
    print(f"Snapshot path: {SNAPSHOT_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    snapshots = read_csv(SNAPSHOT_PATH)
    snapshot_diag = latest_snapshot_diagnostics(snapshots)
    print(
        "Snapshot freshness: "
        f"{snapshot_diag['snapshot_freshness']} age={snapshot_diag['snapshot_age_seconds']:.1f}s "
        f"max={MAX_SNAPSHOT_AGE_SECONDS:.1f}s"
    )

    resolutions = {}
    predictions = {}
    for key, config in TARGET_CONFIGS.items():
        if key == "direction" and not ENABLE_DIRECTION_MODEL and not REQUIRE_DIRECTION_AGREEMENT:
            resolutions[key] = {"artifact": None, "path": None, "skipped": [], "target_spec": config["target_spec"]}
            predictions[key] = None
            continue
        if key == "return" and not ENABLE_REGRESSION_MODEL:
            resolutions[key] = {"artifact": None, "path": None, "skipped": [], "target_spec": config["target_spec"]}
            predictions[key] = None
            continue
        resolved = resolve_model_for_target(key, config)
        resolutions[key] = resolved
        if resolved["artifact"] is None:
            predictions[key] = None
            status = "REQUIRED missing" if config["required"] else "optional missing"
            print(f"- {key} ({config['target_spec']}): {status}")
            for skipped in resolved.get("skipped", [])[:5]:
                print(f"  skipped {skipped['source']}: {skipped['path']} reason={skipped['reason']}")
            continue
        try:
            prediction = predict_resolved_model(resolved, snapshots)
            predictions[key] = prediction
            print(
                f"- {key} ({config['target_spec']}): loaded {resolved['source']} "
                f"path={resolved['path']} direction={prediction['direction']} "
                f"confidence={prediction['confidence']:.2%} schema={prediction['feature_schema_hash']}"
            )
        except Exception as error:
            predictions[key] = None
            print(f"- {key} ({config['target_spec']}): prediction failed: {error}")

    final_signal, final_direction, reason = make_decision(predictions, resolutions, snapshot_diag)
    row = row_from_predictions(predictions, resolutions, snapshot_diag, final_signal, final_direction, reason)
    atomic_append_prediction(row)

    print("Decision")
    print(f"- move_before_adverse direction/confidence: {row['move_before_adverse_direction']} / {row['move_before_adverse_confidence']:.2%}")
    print(f"- instability probability: {row['instability_probability']:.2%}")
    if row["direction_model_available"]:
        print(f"- direction model direction/confidence: {row['direction_model_direction']} / {row['direction_model_confidence']:.2%}")
    else:
        print("- direction model: unavailable/disabled")
    if row["regression_model_available"]:
        print(f"- regression predicted_return_bps: {row['regression_predicted_return_bps']:+.4f}")
    else:
        print("- regression model: unavailable/disabled")
    print(f"- final paper signal: {final_signal}")
    print(f"- decision reason: {reason}")
    if row.get("optional_direction_ignored_reason"):
        print(f"- optional direction note: {row['optional_direction_ignored_reason']}")
    if row.get("optional_regression_ignored_reason"):
        print(f"- optional regression note: {row['optional_regression_ignored_reason']}")
    print(f"- required schema match: {row['required_schema_match']}")
    print(f"- optional direction schema match: {row['optional_direction_schema_match']}")
    print(f"- optional regression schema match: {row['optional_regression_schema_match']}")
    print(f"- all model schema hashes: {row['all_model_schema_hashes']}")
    print(f"Live ensemble output: {OUTPUT_PATH}")
    print("Paper-only. No trades/orders/private API.")


if __name__ == "__main__":
    main()
