import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from show_tiny_price_prediction import (
    PROJECT_ROOT,
    SYMBOL,
    PRIMARY_VENUE,
    VENUE_DIR,
    SNAPSHOT_PATH,
    build_current_features,
    read_csv,
    atomic_write_csv,
    load_json,
)
from train_tiny_price_model import predict_model


DEFAULT_THRESHOLD = float(os.getenv("PRICE_TINY_THRESHOLD", "0.55"))
DEFAULT_REGIME_GATE = os.getenv("PRICE_TINY_REGIME_GATE", "no_gate").strip().lower()
MAX_SNAPSHOT_AGE_SECONDS = float(os.getenv("PRICE_TINY_MAX_SNAPSHOT_AGE_SECONDS", "15"))
MAX_ACTIVE_CHALLENGERS = int(os.getenv("PRICE_TINY_MAX_ACTIVE_CHALLENGERS", "3"))
ALLOWED_SHADOW_MODEL_TYPES = {
    value.strip()
    for value in os.getenv("PRICE_TINY_ALLOWED_SHADOW_MODEL_TYPES", "ridge_logistic,logistic_regression").split(",")
    if value.strip()
}
AUTO_REGISTER_CHALLENGERS = os.getenv("PRICE_TINY_AUTO_REGISTER_CHALLENGERS", "").strip().lower() in {
    "1",
    "true",
    "yes",
} or os.getenv("TRAIN_PRICE_TINY_MODEL", "").strip().lower() in {"1", "true", "yes"}

SELECTED_ROOT = PROJECT_ROOT / "models" / "selected" / SYMBOL / "tiny_price" / (PRIMARY_VENUE or "legacy")
SELECTED_MODEL_JSON = SELECTED_ROOT / "selected_model.json"
CANDIDATE_REGISTRY_PATH = SELECTED_ROOT / "candidate_registry.json"
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "tiny_price" / (PRIMARY_VENUE or "legacy")
SHADOW_OUTPUT_PATH = Path(
    os.getenv(
        "PRICE_TINY_SHADOW_OUTPUT_PATH",
        VENUE_DIR / f"{SYMBOL}_tiny_price_shadow_pool_predictions.csv",
    )
)
if not SHADOW_OUTPUT_PATH.is_absolute():
    SHADOW_OUTPUT_PATH = PROJECT_ROOT / SHADOW_OUTPUT_PATH


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_path(value):
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def selected_model_path_from_file():
    if not SELECTED_MODEL_JSON.exists():
        return None
    try:
        selected = load_json(SELECTED_MODEL_JSON)
    except Exception:
        return None
    return resolve_path(selected.get("champion_model_path") or selected.get("model_path"))


def model_path_from_env_or_selected():
    env_path = os.getenv("PRICE_TINY_MODEL_PATH", "").strip()
    if env_path:
        return resolve_path(env_path)
    return selected_model_path_from_file()


def default_policy_for_artifact(artifact, threshold=None, gate=None):
    return {
        "threshold": float(DEFAULT_THRESHOLD if threshold is None else threshold),
        "horizon": int(artifact.get("horizon_seconds", 1)),
        "feature_set": str(artifact.get("feature_set_name", "")),
        "lookback_profile": str(artifact.get("lookback_profile", "short")),
        "regime_gate": str(DEFAULT_REGIME_GATE if gate is None else gate).strip().lower(),
    }


def threshold_report_for_registration(artifact):
    metrics = artifact.get("forward_test_metrics", {})
    rows = metrics.get("confidence_threshold_directional_report", [])
    eligible = [row for row in rows if float(row.get("threshold", -1.0)) >= DEFAULT_THRESHOLD]
    if not eligible:
        return None
    return min(eligible, key=lambda row: float(row.get("threshold", float("inf"))))


def registration_gate_status(artifact):
    selected = str(artifact.get("selected_model_name", ""))
    if selected not in ALLOWED_SHADOW_MODEL_TYPES:
        return False, "disallowed_model_type"
    metrics = artifact.get("forward_test_metrics", {})
    if not bool(metrics.get("price_candidate_useful", False)) and not bool(metrics.get("direction_candidate_useful", False)):
        return False, "candidate_not_useful"
    threshold_row = threshold_report_for_registration(artifact)
    if threshold_row is None:
        return False, f"missing_threshold_report_at_or_above_{DEFAULT_THRESHOLD:.2f}"
    avg_return = float(threshold_row.get("avg_return_bps", threshold_row.get("avg_realized_return_bps", float("nan"))))
    if not np.isfinite(avg_return) or avg_return <= 0:
        return False, f"threshold_avg_return_not_positive:{avg_return}"
    if not bool(threshold_row.get("threshold_interesting", False)):
        return False, "threshold_report_not_interesting"
    if not bool(threshold_row.get("threshold_stable_candidate", False)):
        return False, "threshold_report_not_stable"
    return True, "passed_shadow_registration_gates"


def normalize_policy(entry, artifact):
    policy = dict(entry.get("policy", {}) if isinstance(entry, dict) else {})
    defaults = default_policy_for_artifact(artifact)
    return {
        "threshold": float(policy.get("threshold", defaults["threshold"])),
        "horizon": int(policy.get("horizon", defaults["horizon"])),
        "feature_set": str(policy.get("feature_set", defaults["feature_set"])),
        "lookback_profile": str(policy.get("lookback_profile", defaults["lookback_profile"])),
        "regime_gate": str(policy.get("regime_gate", defaults["regime_gate"])).strip().lower(),
    }


def load_or_create_registry():
    champion_path = model_path_from_env_or_selected()
    if CANDIDATE_REGISTRY_PATH.exists():
        registry = load_json(CANDIDATE_REGISTRY_PATH)
        if champion_path:
            if str(registry.get("champion_model_path", "")) != str(champion_path):
                registry["champion_model_path"] = str(champion_path)
                try:
                    artifact = load_json(champion_path)
                    registry["champion_policy"] = default_policy_for_artifact(artifact)
                except Exception:
                    pass
                registry["updated_at"] = now_iso()
                atomic_write_json(registry, CANDIDATE_REGISTRY_PATH)
        if not registry.get("champion_model_path"):
            raise RuntimeError(
                f"Candidate registry exists but has no champion_model_path. "
                f"Set PRICE_TINY_MODEL_PATH or create {SELECTED_MODEL_JSON}."
            )
        return registry

    if champion_path is None:
        raise RuntimeError(
            f"No tiny-price champion pinned. Set PRICE_TINY_MODEL_PATH or create {SELECTED_MODEL_JSON}."
        )
    if not champion_path.exists():
        raise RuntimeError(f"Pinned champion model does not exist: {champion_path}")
    artifact = load_json(champion_path)
    registry = {
        "paper_only": True,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE or "legacy",
        "champion_model_path": str(champion_path),
        "champion_policy": default_policy_for_artifact(artifact),
        "challengers": [],
        "max_active_challengers": MAX_ACTIVE_CHALLENGERS,
    }
    atomic_write_json(registry, CANDIDATE_REGISTRY_PATH)
    return registry


def candidate_model_paths_newest_first():
    if not CANDIDATE_ROOT.exists():
        return []
    return sorted(CANDIDATE_ROOT.glob("*/model.json"), key=lambda p: p.parent.name, reverse=True)


def auto_register_recent_challengers(registry):
    registry = retire_disallowed_challengers(registry)
    if not AUTO_REGISTER_CHALLENGERS:
        return registry, 0
    champion_path = resolve_path(registry.get("champion_model_path"))
    existing_paths = {
        str(resolve_path(item.get("model_path") if isinstance(item, dict) else item))
        for item in registry.get("challengers", [])
    }
    added = 0
    challengers = list(registry.get("challengers", []))
    for path in candidate_model_paths_newest_first():
        if champion_path and path.resolve() == champion_path.resolve():
            continue
        if str(path) in existing_paths:
            continue
        try:
            artifact = load_json(path)
        except Exception:
            continue
        if str(artifact.get("symbol", "")).upper() != SYMBOL:
            continue
        if str(artifact.get("primary_venue", "")).lower() != (PRIMARY_VENUE or "legacy"):
            continue
        passed, reason = registration_gate_status(artifact)
        if not passed:
            print(f"Auto-register skipped {path}: {reason}")
            continue
        challengers.append(
            {
                "model_path": str(path),
                "policy": default_policy_for_artifact(artifact),
                "added_at": now_iso(),
                "source": "auto_registered_from_candidate_root",
            }
        )
        added += 1
        if len(challengers) >= int(registry.get("max_active_challengers", MAX_ACTIVE_CHALLENGERS)):
            break
    if added:
        registry["challengers"] = challengers[: int(registry.get("max_active_challengers", MAX_ACTIVE_CHALLENGERS))]
        registry["updated_at"] = now_iso()
        atomic_write_json(registry, CANDIDATE_REGISTRY_PATH)
    return registry, added


def retire_disallowed_challengers(registry):
    challengers = []
    retired = list(registry.get("retired_challengers", []))
    changed = False
    for item in registry.get("challengers", []):
        if not isinstance(item, dict) or item.get("retired"):
            continue
        path = resolve_path(item.get("model_path"))
        artifact = None
        if path and path.exists():
            try:
                artifact = load_json(path)
            except Exception:
                artifact = None
        selected = str(artifact.get("selected_model_name", "")) if artifact else ""
        if artifact and selected not in ALLOWED_SHADOW_MODEL_TYPES:
            retired_item = dict(item)
            retired_item["retired"] = True
            retired_item["retired_at"] = now_iso()
            retired_item["retirement_reason"] = "disallowed_model_type"
            retired_item["selected_model_name"] = selected
            retired.append(retired_item)
            changed = True
        else:
            challengers.append(item)
    registry["challengers"] = challengers
    registry["retired_challengers"] = retired
    if changed:
        registry["updated_at"] = now_iso()
        atomic_write_json(registry, CANDIDATE_REGISTRY_PATH)
    return registry


def depth_from_log(row, column):
    value = float(row.get(column, 0.0))
    if not np.isfinite(value):
        return 0.0
    return max(0.0, float(np.expm1(value)))


def gate_status(feature_row, predicted_direction, gate_name):
    gate_name = str(gate_name or "no_gate").strip().lower()
    if gate_name in {"", "none", "no_gate"}:
        return True, "no_gate"
    row = feature_row.iloc[0] if isinstance(feature_row, pd.DataFrame) else feature_row
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
    if gate_name == "long_side_only":
        return predicted_direction > 0, "passes only long-side signals"
    if gate_name == "short_side_only":
        return predicted_direction < 0, "passes only short-side signals"
    if gate_name == "suppress_longs_when_bid_imbalance_high":
        blocked = predicted_direction > 0 and (imbalance10 >= imbalance10_high or imbalance25 >= imbalance25_high)
        return not blocked, f"imbalance10={imbalance10:.4f}, imbalance25={imbalance25:.4f}"
    if gate_name == "suppress_longs_when_low_volatility_and_bid_imbalance_high":
        high_imbalance = imbalance10 >= imbalance10_high or imbalance25 >= imbalance25_high
        low_vol = np.isfinite(volatility60) and volatility60 <= low_volatility60
        low_range = np.isfinite(range60) and range60 <= low_range60
        blocked = predicted_direction > 0 and high_imbalance and low_vol and low_range
        return (
            not blocked,
            f"imbalance10={imbalance10:.4f}, imbalance25={imbalance25:.4f}, "
            f"vol60={volatility60:.8f}, range60={range60:.8f}",
        )
    if gate_name == "suppress_longs_when_bid_depth_dominates_ask_depth":
        blocked = predicted_direction > 0 and (
            bid_depth_ratio10 >= depth_ratio10_high or bid_depth_ratio25 >= depth_ratio25_high
        )
        return not blocked, f"bid_depth_ratio10={bid_depth_ratio10:.4f}, bid_depth_ratio25={bid_depth_ratio25:.4f}"
    return True, f"unknown gate '{gate_name}' treated as pass"


def model_entries_from_registry(registry):
    champion_path = resolve_path(registry.get("champion_model_path"))
    if champion_path is None:
        return []
    entries = [
        {
            "model_role": "champion",
            "model_path": champion_path,
            "policy": registry.get("champion_policy", {}),
        }
    ]
    limit = int(registry.get("max_active_challengers", MAX_ACTIVE_CHALLENGERS))
    for index, challenger in enumerate(registry.get("challengers", [])[:limit], start=1):
        if isinstance(challenger, dict) and challenger.get("retired"):
            continue
        path_value = challenger.get("model_path") if isinstance(challenger, dict) else challenger
        path = resolve_path(path_value)
        if path is None:
            continue
        entries.append(
            {
                "model_role": f"challenger_{index}",
                "model_path": path,
                "policy": challenger.get("policy", {}) if isinstance(challenger, dict) else {},
            }
        )
    return entries


def latest_snapshot_diagnostics(snapshots):
    if len(snapshots) == 0 or "timestamp" not in snapshots.columns:
        return {
            "latest_snapshot_timestamp": np.nan,
            "now_timestamp": int(time.time() * 1000),
            "snapshot_age_seconds": np.inf,
            "max_snapshot_age_seconds": MAX_SNAPSHOT_AGE_SECONDS,
            "snapshot_freshness_status": "stale",
        }
    timestamps = pd.to_numeric(snapshots["timestamp"], errors="coerce").dropna()
    if len(timestamps) == 0:
        return {
            "latest_snapshot_timestamp": np.nan,
            "now_timestamp": int(time.time() * 1000),
            "snapshot_age_seconds": np.inf,
            "max_snapshot_age_seconds": MAX_SNAPSHOT_AGE_SECONDS,
            "snapshot_freshness_status": "stale",
        }
    latest_timestamp = int(timestamps.max())
    now_timestamp = int(time.time() * 1000)
    age_seconds = max(0.0, (now_timestamp - latest_timestamp) / 1000.0)
    return {
        "latest_snapshot_timestamp": latest_timestamp,
        "now_timestamp": now_timestamp,
        "snapshot_age_seconds": float(age_seconds),
        "max_snapshot_age_seconds": MAX_SNAPSHOT_AGE_SECONDS,
        "snapshot_freshness_status": "fresh" if age_seconds <= MAX_SNAPSHOT_AGE_SECONDS else "stale",
    }


def predict_entry(entry, snapshots, snapshot_diagnostics):
    model_path = entry["model_path"]
    artifact = load_json(model_path)
    if str(artifact.get("symbol", "")).upper() != SYMBOL:
        raise RuntimeError(f"model symbol mismatch: {artifact.get('symbol')} != {SYMBOL}")
    if str(artifact.get("primary_venue", "")).lower() != (PRIMARY_VENUE or "legacy"):
        raise RuntimeError(f"model venue mismatch: {artifact.get('primary_venue')} != {PRIMARY_VENUE or 'legacy'}")
    policy = normalize_policy(entry, artifact)
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
    timestamp = int(feature_row["timestamp"].iloc[0])
    age_seconds = float(snapshot_diagnostics["snapshot_age_seconds"])
    freshness_status = str(snapshot_diagnostics["snapshot_freshness_status"])
    raw_direction = int(pred_direction[0])
    threshold_passed = bool(float(confidence[0]) >= float(policy["threshold"]))
    gate_passed, gate_reason = gate_status(feature_row, raw_direction, policy["regime_gate"])
    if freshness_status != "fresh":
        threshold_passed = False
        gate_passed = False
        gate_reason = (
            f"stale snapshot: age={age_seconds:.1f}s "
            f"> max={MAX_SNAPSHOT_AGE_SECONDS:.1f}s"
        )
    paper_signal_direction = raw_direction if threshold_passed and gate_passed and freshness_status == "fresh" else 0
    model_horizon = int(artifact.get("horizon_seconds", 1))
    requested_horizon = int(policy.get("horizon", model_horizon))
    probs = np.asarray(probabilities[0], dtype=np.float64)
    target_spec = artifact.get("target_spec", {}) if isinstance(artifact.get("target_spec", {}), dict) else {}
    feature_spec = artifact.get("feature_spec", {}) if isinstance(artifact.get("feature_spec", {}), dict) else {}
    output_semantics = artifact.get("output_semantics", target_spec.get("output_semantics", {}))
    selected_target_columns = artifact.get("selected_target_columns", target_spec.get("selected_target_columns", []))
    if isinstance(selected_target_columns, str):
        selected_target_columns_text = selected_target_columns
    else:
        selected_target_columns_text = ",".join(str(value) for value in selected_target_columns)
    return {
        "timestamp": timestamp,
        "time": feature_row["time"].iloc[0],
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE or "legacy",
        "experiment_id": artifact.get("experiment_id", ""),
        "model_id": artifact.get("model_id", ""),
        "model_path": str(model_path),
        "model_role": entry["model_role"],
        "feature_spec": feature_spec.get("name", artifact.get("feature_set_name", "")),
        "enabled_feature_groups": ",".join(str(value) for value in feature_spec.get("enabled_feature_groups", [])),
        "target_spec": target_spec.get("name", f"direction_{artifact.get('horizon_seconds', 1)}s"),
        "model_spec": artifact.get("model_spec", {}).get("model_type", artifact.get("selected_model_name", "")),
        "feature_schema_hash": artifact.get("feature_schema_hash", ""),
        "available_target_columns": ",".join(str(value) for value in artifact.get("available_target_columns", artifact.get("target_columns", []))),
        "selected_target_columns": selected_target_columns_text,
        "selected_classification_target_column": artifact.get("selected_classification_target_column", target_spec.get("selected_classification_target_column", "")),
        "realized_return_column": artifact.get("realized_return_column", target_spec.get("realized_return_column", "")),
        "output_semantics": json.dumps(output_semantics, sort_keys=True) if isinstance(output_semantics, dict) else str(output_semantics or ""),
        "selected_model_name": selected,
        "threshold": float(policy["threshold"]),
        "gate": policy["regime_gate"],
        "horizon": requested_horizon,
        "model_horizon_seconds": model_horizon,
        "horizon_matches_model": bool(requested_horizon == model_horizon),
        "feature_set": artifact.get("feature_set_name", ""),
        "lookback_profile": artifact.get("lookback_profile", "short"),
        "confidence": float(confidence[0]),
        "prob_down": float(probs[0]) if len(probs) > 0 else np.nan,
        "prob_flat": float(probs[1]) if len(probs) > 1 else np.nan,
        "prob_up": float(probs[2]) if len(probs) > 2 else np.nan,
        "raw_direction": raw_direction,
        "predicted_direction": raw_direction,
        "predicted_return_bps": float(pred_delta[0]),
        "predicted_next_mid_delta_bps": float(pred_delta[0]),
        "predicted_next_mid_log_return": float(pred_log[0]),
        "confidence_type": artifact.get("confidence_type", "class_probability"),
        "threshold_passed": threshold_passed,
        "regime_gate_passed": gate_passed,
        "regime_gate_reason": gate_reason,
        "paper_signal_direction": int(paper_signal_direction),
        "freshness": freshness_status,
        "freshness_status": freshness_status,
        "snapshot_age_seconds": float(age_seconds),
        "latest_snapshot_timestamp": snapshot_diagnostics["latest_snapshot_timestamp"],
        "now_timestamp": snapshot_diagnostics["now_timestamp"],
        "max_snapshot_age_seconds": snapshot_diagnostics["max_snapshot_age_seconds"],
        "stale_reason": "" if freshness_status == "fresh" else gate_reason,
        "paper_only": True,
    }


def append_shadow_rows(rows):
    existing = read_csv(SHADOW_OUTPUT_PATH)
    output = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if len(existing) else pd.DataFrame(rows)
    output["timestamp"] = pd.to_numeric(output["timestamp"], errors="coerce")
    output = output.dropna(subset=["timestamp"])
    output = output.drop_duplicates(["timestamp", "model_id", "model_role", "model_path"], keep="last")
    output = output.sort_values(["timestamp", "model_role", "model_id"])
    atomic_write_csv(output, SHADOW_OUTPUT_PATH)


def main():
    print("Tiny price shadow pool prediction")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Snapshot path: {SNAPSHOT_PATH}")
    print(f"Registry path: {CANDIDATE_REGISTRY_PATH}")
    try:
        registry = load_or_create_registry()
        registry, added = auto_register_recent_challengers(registry)
    except Exception as exc:
        print(f"Blocked: {exc}")
        print("Paper-only. No trades/orders/private API.")
        return
    print(f"Max active challengers: {registry.get('max_active_challengers', MAX_ACTIVE_CHALLENGERS)}")
    print(f"Auto-registered challengers this run: {added if 'added' in locals() else 0}")
    snapshots = read_csv(SNAPSHOT_PATH)
    if len(snapshots) == 0:
        print(f"Blocked: missing or empty snapshot file: {SNAPSHOT_PATH}")
        print("Paper-only. No trades/orders/private API.")
        return
    snapshot_diagnostics = latest_snapshot_diagnostics(snapshots)
    print(f"latest_snapshot_timestamp: {snapshot_diagnostics['latest_snapshot_timestamp']}")
    print(f"now_timestamp: {snapshot_diagnostics['now_timestamp']}")
    print(f"snapshot_age_seconds: {snapshot_diagnostics['snapshot_age_seconds']:.1f}")
    print(f"max_snapshot_age_seconds: {snapshot_diagnostics['max_snapshot_age_seconds']:.1f}")
    if snapshot_diagnostics["snapshot_freshness_status"] != "fresh":
        print("WARNING: snapshot source is stale; writing no-signal stale diagnostics for all shadow models.")
    rows = []
    failures = []
    for entry in model_entries_from_registry(registry):
        try:
            rows.append(predict_entry(entry, snapshots, snapshot_diagnostics))
        except Exception as exc:
            failures.append((entry["model_role"], entry["model_path"], str(exc)))
    if rows:
        append_shadow_rows(rows)
    print(f"Predictions written this refresh: {len(rows)}")
    print(f"Shadow output: {SHADOW_OUTPUT_PATH}")
    for row in rows:
        print(
            f"- {row['model_role']} {row['model_id']} "
            f"dir={row['raw_direction']} signal={row['paper_signal_direction']} "
            f"conf={row['confidence']:.2%} threshold={row['threshold']:.2f} "
            f"gate={row['gate']} fresh={row['freshness_status']}"
        )
    for role, path, reason in failures:
        print(f"Skipped {role}: {path} reason={reason}")
    print("Champion remains pinned. Challengers are paper-only shadow models.")
    print("No trades/orders/private API.")


if __name__ == "__main__":
    main()
