import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR

SNAPSHOT_PATH = VENUE_DIR / f"{SYMBOL}_10s_flow.csv"
SHADOW_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_shadow_pool_predictions.csv"
EVALUATION_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_shadow_pool_evaluation.csv"
SELECTED_ROOT = PROJECT_ROOT / "models" / "selected" / SYMBOL / "tiny_price" / (PRIMARY_VENUE or "legacy")
SELECTED_MODEL_JSON = SELECTED_ROOT / "selected_model.json"
CANDIDATE_REGISTRY_PATH = SELECTED_ROOT / "candidate_registry.json"

MIN_SHARED_ROWS = int(os.getenv("PRICE_TINY_SHADOW_MIN_SHARED_ROWS", "300"))
MIN_ACTIVE_ROWS = int(os.getenv("PRICE_TINY_SHADOW_MIN_ACTIVE_ROWS", "300"))
ALLOW_PROMOTION = os.getenv("PRICE_TINY_SHADOW_ALLOW_PROMOTION", "false").strip().lower() in {"1", "true", "yes"}
MAX_FUTURE_GAP_MS = int(os.getenv("PRICE_TINY_SHADOW_MAX_FUTURE_GAP_MS", "2500"))
LEGACY_FEATURE_SPEC_NAME = "base_tiny_price_v1"
LEGACY_TARGET_SPEC_NAME = "direction_30s"
ARTIFACT_CACHE = {}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def resolve_path(value):
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def artifact_for_model_path(value):
    path = resolve_path(value)
    if path is None:
        return {}
    key = str(path)
    if key not in ARTIFACT_CACHE:
        try:
            ARTIFACT_CACHE[key] = load_json_if_exists(path) or {}
        except Exception:
            ARTIFACT_CACHE[key] = {}
    return ARTIFACT_CACHE[key]


def nonempty_text(*values, default=""):
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return default


def parse_list_value(value):
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    return [item.strip() for item in text.split(",") if item.strip()]


def target_horizon_seconds(target_spec_name):
    text = str(target_spec_name or LEGACY_TARGET_SPEC_NAME)
    for part in reversed(text.split("_")):
        if part.endswith("s") and part[:-1].isdigit():
            return int(part[:-1])
    return 30


def target_label_method(target_spec_name):
    text = str(target_spec_name or LEGACY_TARGET_SPEC_NAME).lower()
    if text.startswith("move_before_adverse"):
        return "move_before_adverse"
    if text.startswith("first_touch"):
        return "first_touch"
    if text.startswith("return"):
        return "return_bps"
    if text.startswith("chop") or "no_trade" in text:
        return "chop_no_trade"
    return "direction"


def default_selected_target_columns(target_spec_name):
    horizon = target_horizon_seconds(target_spec_name)
    method = target_label_method(target_spec_name)
    if method == "move_before_adverse":
        return [f"target_move_before_adverse_{horizon}s"]
    if method == "first_touch":
        return [f"target_first_touch_direction_{horizon}s"]
    if method == "chop_no_trade":
        return [f"target_chop_no_trade_{horizon}s"]
    if method == "return_bps":
        return [f"target_return_bps_{horizon}s"]
    return [f"target_direction_{horizon}s"]


def default_output_semantics(target_spec_name):
    horizon = target_horizon_seconds(target_spec_name)
    selected = default_selected_target_columns(target_spec_name)
    method = target_label_method(target_spec_name)
    return {
        "target_spec": target_spec_name,
        "selected_target_columns": selected,
        "selected_classification_target_column": selected[0] if selected else "",
        "selected_classification_target_method": method,
        "realized_return_column": f"target_return_bps_{horizon}s",
        "return_head_target_column": selected[0] if method == "return_bps" else f"target_next_mid_delta_bps_{horizon}s",
        "regression_metrics_key": "regression_return_metrics" if method == "return_bps" else "auxiliary_return_head_metrics",
        "paper_only": True,
    }


def canonical_json(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            value = {}
        else:
            try:
                value = json.loads(text)
            except Exception:
                return text
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def normalize_feature_spec_name(value):
    text = nonempty_text(value, default=LEGACY_FEATURE_SPEC_NAME)
    if text in {"tiny_price_v1", "base"}:
        return LEGACY_FEATURE_SPEC_NAME
    return text


def experiment_family_from_row(row):
    artifact = artifact_for_model_path(row.get("model_path", ""))
    feature_spec = artifact.get("feature_spec", {}) if isinstance(artifact.get("feature_spec", {}), dict) else {}
    target_spec = artifact.get("target_spec", {}) if isinstance(artifact.get("target_spec", {}), dict) else {}

    feature_spec_name = normalize_feature_spec_name(
        feature_spec.get("name", row.get("feature_spec", row.get("feature_set", "")))
    )
    enabled_feature_groups = parse_list_value(feature_spec.get("enabled_feature_groups"))
    if not enabled_feature_groups:
        enabled_feature_groups = parse_list_value(row.get("enabled_feature_groups", ""))
    if not enabled_feature_groups:
        enabled_feature_groups = [feature_spec_name or LEGACY_FEATURE_SPEC_NAME]
    enabled_feature_groups = sorted(set(enabled_feature_groups))

    target_spec_name = nonempty_text(
        target_spec.get("name", ""),
        row.get("target_spec", ""),
        default=LEGACY_TARGET_SPEC_NAME,
    )
    selected_target_columns = parse_list_value(
        artifact.get("selected_target_columns", target_spec.get("selected_target_columns", row.get("selected_target_columns", "")))
    )
    if not selected_target_columns:
        selected_target_columns = default_selected_target_columns(target_spec_name)

    realized_return_column = nonempty_text(
        artifact.get("realized_return_column", ""),
        target_spec.get("realized_return_column", ""),
        row.get("realized_return_column", ""),
        default=f"target_return_bps_{target_horizon_seconds(target_spec_name)}s",
    )
    output_semantics = artifact.get("output_semantics", target_spec.get("output_semantics", row.get("output_semantics", None)))
    if not output_semantics:
        output_semantics = default_output_semantics(target_spec_name)

    return {
        "symbol": nonempty_text(artifact.get("symbol", ""), row.get("symbol", ""), default=SYMBOL).upper(),
        "primary_venue": nonempty_text(
            artifact.get("primary_venue", ""),
            row.get("primary_venue", ""),
            default=PRIMARY_VENUE or "legacy",
        ).lower(),
        "feature_schema_hash": nonempty_text(
            artifact.get("feature_schema_hash", ""),
            row.get("feature_schema_hash", ""),
            default="",
        ),
        "feature_spec_name": feature_spec_name,
        "enabled_feature_groups": enabled_feature_groups,
        "target_spec_name": target_spec_name,
        "selected_target_columns": selected_target_columns,
        "realized_return_column": realized_return_column,
        "output_semantics": canonical_json(output_semantics),
    }


EXPERIMENT_FAMILY_FIELDS = [
    "symbol",
    "primary_venue",
    "feature_schema_hash",
    "feature_spec_name",
    "enabled_feature_groups",
    "target_spec_name",
    "selected_target_columns",
    "realized_return_column",
    "output_semantics",
]


def experiment_family_key(family):
    return canonical_json({field: family.get(field) for field in EXPERIMENT_FAMILY_FIELDS})


def compare_experiment_families(champion_family, challenger_family):
    mismatches = [
        field
        for field in EXPERIMENT_FAMILY_FIELDS
        if champion_family.get(field) != challenger_family.get(field)
    ]
    return len(mismatches) == 0, mismatches


def active_registry_model_paths():
    if not CANDIDATE_REGISTRY_PATH.exists():
        return None
    registry = load_json(CANDIDATE_REGISTRY_PATH)
    paths = []
    champion = resolve_path(registry.get("champion_model_path"))
    if champion:
        paths.append(str(champion))
    limit = int(registry.get("max_active_challengers", 3))
    for challenger in registry.get("challengers", [])[:limit]:
        if isinstance(challenger, dict) and challenger.get("retired"):
            continue
        value = challenger.get("model_path") if isinstance(challenger, dict) else challenger
        path = resolve_path(value)
        if path:
            paths.append(str(path))
    return set(paths)


def normalize_snapshots(frame):
    if len(frame) == 0:
        return frame
    frame = frame.copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["mid_price"] = pd.to_numeric(frame.get("mid_price"), errors="coerce")
    frame = frame.dropna(subset=["timestamp", "mid_price"]).sort_values("timestamp").drop_duplicates("timestamp")
    return frame.reset_index(drop=True)


def attach_actual_returns(predictions, snapshots):
    if len(predictions) == 0 or len(snapshots) == 0:
        return predictions
    predictions = predictions.copy()
    snapshots = normalize_snapshots(snapshots)
    snap_ts = snapshots["timestamp"].to_numpy(dtype=np.int64)
    snap_mid = snapshots["mid_price"].to_numpy(dtype=np.float64)
    actual_delta = []
    actual_log_return = []
    future_timestamp = []
    future_gap_ms = []
    outcome_status = []
    for _, row in predictions.iterrows():
        timestamp = int(row["timestamp"])
        horizon = int(row.get("horizon", row.get("model_horizon_seconds", 1)))
        target_ts = timestamp + horizon * 1000
        current_index = np.searchsorted(snap_ts, timestamp, side="left")
        future_index = np.searchsorted(snap_ts, target_ts, side="left")
        if current_index >= len(snap_ts):
            actual_delta.append(np.nan)
            actual_log_return.append(np.nan)
            future_timestamp.append(np.nan)
            future_gap_ms.append(np.nan)
            outcome_status.append("missing_current_snapshot")
            continue
        if future_index >= len(snap_ts):
            actual_delta.append(np.nan)
            actual_log_return.append(np.nan)
            future_timestamp.append(np.nan)
            future_gap_ms.append(np.nan)
            outcome_status.append("missing_future_snapshot")
            continue
        if abs(int(snap_ts[current_index]) - timestamp) > MAX_FUTURE_GAP_MS:
            actual_delta.append(np.nan)
            actual_log_return.append(np.nan)
            future_timestamp.append(np.nan)
            future_gap_ms.append(np.nan)
            outcome_status.append("current_snapshot_gap_too_large")
            continue
        gap = int(snap_ts[future_index]) - target_ts
        if gap < 0 or gap > MAX_FUTURE_GAP_MS:
            actual_delta.append(np.nan)
            actual_log_return.append(np.nan)
            future_timestamp.append(np.nan)
            future_gap_ms.append(float(gap))
            outcome_status.append("future_snapshot_gap_too_large")
            continue
        current_mid = float(snap_mid[current_index])
        future_mid = float(snap_mid[future_index])
        if current_mid <= 0 or future_mid <= 0:
            actual_delta.append(np.nan)
            actual_log_return.append(np.nan)
            future_timestamp.append(np.nan)
            future_gap_ms.append(float(gap))
            outcome_status.append("invalid_mid_price")
            continue
        log_return = float(np.log(future_mid / current_mid))
        actual_delta.append(float((future_mid / current_mid - 1.0) * 10000.0))
        actual_log_return.append(log_return)
        future_timestamp.append(int(snap_ts[future_index]))
        future_gap_ms.append(float(gap))
        outcome_status.append("ok")
    predictions["actual_next_mid_delta_bps"] = actual_delta
    predictions["actual_next_mid_log_return"] = actual_log_return
    predictions["actual_future_timestamp"] = future_timestamp
    predictions["actual_future_gap_ms"] = future_gap_ms
    predictions["actual_outcome_status"] = outcome_status
    predictions["actual_direction"] = np.where(
        predictions["actual_next_mid_delta_bps"] > 0,
        1,
        np.where(predictions["actual_next_mid_delta_bps"] < 0, -1, 0),
    )
    return predictions


def make_model_key(frame):
    return (
        frame["model_id"].astype(str)
        + "|"
        + frame["model_path"].astype(str)
    )


def prepare_predictions(predictions):
    predictions = predictions.copy()
    predictions["model_key"] = make_model_key(predictions)
    active_paths = active_registry_model_paths()
    if active_paths:
        predictions = predictions[predictions["model_path"].astype(str).isin(active_paths)].copy()
    predictions = predictions.drop_duplicates(["timestamp", "model_key"], keep="last")
    return predictions


def global_shared_scored_rows(predictions, model_keys):
    if not model_keys:
        return 0
    predictions = fresh_predictions_only(predictions)
    counts = predictions.groupby("timestamp")["model_key"].nunique()
    shared_timestamps = counts[counts == len(model_keys)].index
    shared = predictions[predictions["timestamp"].isin(shared_timestamps)].copy()
    shared = shared.dropna(subset=["actual_next_mid_delta_bps"])
    usable_timestamps = shared.groupby("timestamp")["model_key"].nunique()
    usable = usable_timestamps[usable_timestamps == len(model_keys)].index
    return int(len(shared[shared["timestamp"].isin(usable)]))


def identify_champion_key(predictions):
    champion_rows = predictions[predictions["model_role"].astype(str).eq("champion")].copy()
    if len(champion_rows) == 0:
        return None
    registry_champion = None
    if CANDIDATE_REGISTRY_PATH.exists():
        try:
            registry_champion = resolve_path(load_json(CANDIDATE_REGISTRY_PATH).get("champion_model_path"))
        except Exception:
            registry_champion = None
    if registry_champion is not None:
        matched = champion_rows[champion_rows["model_path"].astype(str).eq(str(registry_champion))]
        if len(matched):
            return matched["model_key"].value_counts().index[0]
    return champion_rows["model_key"].value_counts().index[0]


def model_key_diagnostics(predictions):
    rows = []
    for key, subset in predictions.groupby("model_key"):
        subset = subset.sort_values("timestamp")
        row0 = subset.iloc[-1]
        fresh = freshness_mask(subset)
        family = experiment_family_from_row(row0)
        rows.append(
            {
                "model_key": key,
                "symbol": row0.get("symbol", SYMBOL),
                "primary_venue": row0.get("primary_venue", PRIMARY_VENUE or "legacy"),
                "model_role": row0.get("model_role", ""),
                "model_id": row0.get("model_id", ""),
                "feature_spec": family["feature_spec_name"],
                "target_spec": family["target_spec_name"],
                "model_spec": row0.get("model_spec", row0.get("selected_model_name", "")),
                "experiment_family_key": experiment_family_key(family),
                "feature_schema_hash": family["feature_schema_hash"],
                "enabled_feature_groups": ",".join(family["enabled_feature_groups"]),
                "selected_target_columns": ",".join(family["selected_target_columns"]),
                "realized_return_column": family["realized_return_column"],
                "model_path": row0.get("model_path", ""),
                "rows": int(len(subset)),
                "first_timestamp": int(subset["timestamp"].iloc[0]),
                "last_timestamp": int(subset["timestamp"].iloc[-1]),
                "scored_rows": int((fresh & subset["actual_next_mid_delta_bps"].notna()).sum()),
                "unscored_rows": int((fresh & subset["actual_next_mid_delta_bps"].isna()).sum()),
                "stale_rows": int((~fresh).sum()),
                "fresh_rows": int(fresh.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_role", "last_timestamp", "model_id"])


def freshness_mask(frame):
    status = frame.get("freshness_status", frame.get("freshness", ""))
    if isinstance(status, str):
        return pd.Series(False, index=frame.index)
    return status.astype(str).str.lower().eq("fresh")


def fresh_predictions_only(frame):
    if len(frame) == 0:
        return frame.copy()
    return frame[freshness_mask(frame)].copy()


def pairwise_subset(predictions, champion_key, challenger_key):
    champion = predictions[predictions["model_key"] == champion_key].copy()
    challenger = predictions[predictions["model_key"] == challenger_key].copy()
    champion_timestamps = set(champion["timestamp"].tolist())
    challenger_timestamps = set(challenger["timestamp"].tolist())
    shared_timestamps = sorted(champion_timestamps & challenger_timestamps)
    pair = predictions[
        predictions["model_key"].isin([champion_key, challenger_key])
        & predictions["timestamp"].isin(shared_timestamps)
    ].copy()
    stale_excluded_rows = int((~freshness_mask(pair)).sum())
    fresh_pair = fresh_predictions_only(pair)
    realized = fresh_pair.dropna(subset=["actual_next_mid_delta_bps"]).copy()
    realized_counts = realized.groupby("timestamp")["model_key"].nunique()
    usable_timestamps = realized_counts[realized_counts == 2].index
    usable = realized[realized["timestamp"].isin(usable_timestamps)].copy()
    unscored = fresh_pair[~fresh_pair["timestamp"].isin(usable_timestamps)].copy()
    missing_future_snapshot = int(
        unscored.get("actual_outcome_status", pd.Series(dtype=str))
        .astype(str)
        .isin(["missing_future_snapshot", "future_snapshot_gap_too_large"])
        .sum()
    )
    return usable, {
        "pairwise_shared_timestamps": int(len(shared_timestamps)),
        "pairwise_shared_realized_timestamps": int(len(usable_timestamps)),
        "pairwise_shared_realized_rows": int(len(usable)),
        "pairwise_stale_excluded_rows": stale_excluded_rows,
        "unscored_rows_due_to_missing_future_outcome": int(len(unscored)),
        "unscored_rows_due_to_missing_snapshot_future_price": missing_future_snapshot,
    }


def bool_series(frame, column, default=False):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default)
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def evaluate_model(frame, model_key):
    subset = frame[frame["model_key"] == model_key].copy()
    signal = pd.to_numeric(subset.get("paper_signal_direction", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    actual_delta = pd.to_numeric(subset["actual_next_mid_delta_bps"], errors="coerce").to_numpy(dtype=np.float64)
    active = signal != 0
    returns = np.where(active, actual_delta * np.sign(signal), 0.0)
    active_returns = returns[active]
    fresh = subset.get("freshness_status", subset.get("freshness", "")).astype(str).str.lower().eq("fresh")
    horizon_ok = bool_series(subset, "horizon_matches_model", True)
    threshold_passed = bool_series(subset, "threshold_passed", False)
    gate_passed = bool_series(subset, "regime_gate_passed", False)
    long_mask = active & (signal > 0)
    short_mask = active & (signal < 0)
    row0 = subset.iloc[-1] if len(subset) else {}
    family = experiment_family_from_row(row0)
    return {
        "model_key": model_key,
        "symbol": row0.get("symbol", SYMBOL),
        "primary_venue": row0.get("primary_venue", PRIMARY_VENUE or "legacy"),
        "model_role": row0.get("model_role", ""),
        "model_id": row0.get("model_id", ""),
        "experiment_id": row0.get("experiment_id", ""),
        "feature_spec": family["feature_spec_name"],
        "target_spec": family["target_spec_name"],
        "model_spec": row0.get("model_spec", row0.get("selected_model_name", "")),
        "feature_schema_hash": family["feature_schema_hash"],
        "enabled_feature_groups": ",".join(family["enabled_feature_groups"]),
        "available_target_columns": row0.get("available_target_columns", ""),
        "selected_target_columns": ",".join(family["selected_target_columns"]),
        "selected_classification_target_column": row0.get("selected_classification_target_column", ""),
        "realized_return_column": family["realized_return_column"],
        "output_semantics": family["output_semantics"],
        "experiment_family_key": experiment_family_key(family),
        "model_path": row0.get("model_path", ""),
        "threshold": float(row0.get("threshold", np.nan)),
        "gate": row0.get("gate", ""),
        "horizon": int(float(row0.get("horizon", row0.get("model_horizon_seconds", 0)) or 0)),
        "feature_set": row0.get("feature_set", ""),
        "lookback_profile": row0.get("lookback_profile", ""),
        "shared_rows": int(len(subset)),
        "active_rows": int(active.sum()),
        "coverage": float(active.mean()) if len(active) else 0.0,
        "avg_return_bps": float(active_returns.mean()) if len(active_returns) else 0.0,
        "median_return_bps": float(np.median(active_returns)) if len(active_returns) else 0.0,
        "win_rate": float((active_returns > 0).mean()) if len(active_returns) else np.nan,
        "long_active_rows": int(long_mask.sum()),
        "long_avg_return_bps": float((actual_delta[long_mask]).mean()) if long_mask.any() else np.nan,
        "short_active_rows": int(short_mask.sum()),
        "short_avg_return_bps": float((-actual_delta[short_mask]).mean()) if short_mask.any() else np.nan,
        "threshold_passed_rows": int(threshold_passed.sum()),
        "gate_passed_rows": int(gate_passed.sum()),
        "freshness_issue_rows": int((~fresh).sum()),
        "horizon_issue_rows": int((~horizon_ok).sum()),
        "average_confidence": float(pd.to_numeric(subset.get("confidence", np.nan), errors="coerce").mean()),
        "paper_only": True,
    }


def pairwise_evaluation_rows(predictions, champion_key, challenger_keys):
    rows = []
    pair_diagnostics = []
    for challenger_key in challenger_keys:
        usable, diagnostics = pairwise_subset(predictions, champion_key, challenger_key)
        champion_model_rows = usable[usable["model_key"] == champion_key]
        challenger_model_rows = usable[usable["model_key"] == challenger_key]
        champion_meta = predictions[predictions["model_key"] == champion_key].iloc[-1]
        challenger_meta = predictions[predictions["model_key"] == challenger_key].iloc[-1]
        champion_family = experiment_family_from_row(champion_meta)
        challenger_family = experiment_family_from_row(challenger_meta)
        family_matches, family_mismatch_fields = compare_experiment_families(champion_family, challenger_family)
        diagnostics.update(
            {
                "champion_model_key": champion_key,
                "challenger_model_key": challenger_key,
                "champion_model_id": champion_meta.get("model_id", ""),
                "challenger_model_id": challenger_meta.get("model_id", ""),
                "challenger_role": challenger_meta.get("model_role", ""),
                "experiment_family_matches": bool(family_matches),
                "experiment_family_mismatch_fields": ",".join(family_mismatch_fields),
                "champion_experiment_family_key": experiment_family_key(champion_family),
                "challenger_experiment_family_key": experiment_family_key(challenger_family),
            }
        )
        pair_diagnostics.append(diagnostics)
        if len(champion_model_rows) == 0 or len(challenger_model_rows) == 0:
            continue
        champion_summary = evaluate_model(usable, champion_key)
        challenger_summary = evaluate_model(usable, challenger_key)
        comparison_id = f"{champion_summary['model_id']}__vs__{challenger_summary['model_id']}"
        for summary, opponent in [
            (champion_summary, challenger_summary),
            (challenger_summary, champion_summary),
        ]:
            summary.update(
                {
                    "comparison_id": comparison_id,
                    "comparison_role": "champion" if summary["model_key"] == champion_key else "challenger",
                    "opponent_model_key": opponent["model_key"],
                    "opponent_model_id": opponent["model_id"],
                    "opponent_avg_return_bps": opponent["avg_return_bps"],
                    "opponent_win_rate": opponent["win_rate"],
                    "opponent_active_rows": opponent["active_rows"],
                    "experiment_family_matches": bool(family_matches),
                    "experiment_family_mismatch_fields": ",".join(family_mismatch_fields),
                    "champion_experiment_family_key": experiment_family_key(champion_family),
                    "challenger_experiment_family_key": experiment_family_key(challenger_family),
                    "promotion_eligible": bool(family_matches),
                    "promotion_block_reason": "" if family_matches else "experiment_spec_mismatch",
                    "pairwise_shared_timestamps": diagnostics["pairwise_shared_timestamps"],
                    "pairwise_shared_realized_timestamps": diagnostics["pairwise_shared_realized_timestamps"],
                    "unscored_rows_due_to_missing_future_outcome": diagnostics[
                        "unscored_rows_due_to_missing_future_outcome"
                    ],
                    "unscored_rows_due_to_missing_snapshot_future_price": diagnostics[
                        "unscored_rows_due_to_missing_snapshot_future_price"
                    ],
                    "pairwise_stale_excluded_rows": diagnostics["pairwise_stale_excluded_rows"],
                }
            )
            rows.append(summary)
    return pd.DataFrame(rows), pd.DataFrame(pair_diagnostics)


def selected_payload_from_summary(summary_row):
    return {
        "paper_only": True,
        "updated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE or "legacy",
        "champion_model_path": summary_row["model_path"],
        "champion_policy": {
            "threshold": float(summary_row["threshold"]),
            "horizon": int(summary_row["horizon"]),
            "feature_set": summary_row["feature_set"],
            "lookback_profile": summary_row["lookback_profile"],
            "regime_gate": summary_row["gate"],
        },
        "selection_source": "tiny_price_shadow_pool_evaluator",
        "selection_note": "Paper-only champion pointer. No trades/orders/private API.",
    }


def maybe_promote(evaluation):
    challengers = evaluation[evaluation["comparison_role"].astype(str).eq("challenger")].copy()
    if len(challengers) == 0:
        return "No pairwise challenger rows available; no paper promotion."
    if "promotion_eligible" not in challengers.columns:
        challengers["promotion_eligible"] = True
    if "promotion_block_reason" not in challengers.columns:
        challengers["promotion_block_reason"] = ""
    challengers["beats_champion"] = (
        challengers["promotion_eligible"].astype(bool)
        &
        (challengers["active_rows"] >= MIN_ACTIVE_ROWS)
        & (challengers["pairwise_shared_realized_timestamps"] >= MIN_SHARED_ROWS)
        & (challengers["freshness_issue_rows"] == 0)
        & (challengers["pairwise_stale_excluded_rows"] == 0)
        & (challengers["horizon_issue_rows"] == 0)
        & (challengers["avg_return_bps"] > challengers["opponent_avg_return_bps"])
        & (
            challengers["opponent_win_rate"].isna()
            | challengers["win_rate"].isna()
            | (challengers["win_rate"] >= challengers["opponent_win_rate"])
        )
    )
    eligible = challengers[challengers["beats_champion"]].sort_values(
        ["avg_return_bps", "win_rate", "active_rows"], ascending=False
    )
    if len(eligible) == 0:
        mismatched = challengers[challengers["promotion_block_reason"].astype(str).eq("experiment_spec_mismatch")]
        if len(mismatched):
            return (
                "No challenger passed the paper champion replacement rule. "
                f"{len(mismatched)} challenger comparison row(s) were blocked by experiment_spec_mismatch."
            )
        return "No challenger passed the paper champion replacement rule."
    best = eligible.iloc[0]
    if not ALLOW_PROMOTION:
        return (
            "Best challenger passed the paper rule, but PRICE_TINY_SHADOW_ALLOW_PROMOTION=false; "
            f"champion unchanged. Best challenger: {best['model_id']}"
        )
    payload = selected_payload_from_summary(best)
    atomic_write_json(payload, SELECTED_MODEL_JSON)
    return f"Paper champion pointer updated to challenger: {best['model_id']}"


def main():
    print("Tiny price shadow pool evaluator")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Shadow predictions: {SHADOW_PATH}")
    print(f"Snapshot source: {SNAPSHOT_PATH}")
    predictions = read_csv(SHADOW_PATH)
    snapshots = read_csv(SNAPSHOT_PATH)
    if len(predictions) == 0:
        print("No shadow prediction rows found.")
        print("Paper-only. No trades/orders/private API.")
        return
    if len(snapshots) == 0:
        print("No snapshot rows found for realized outcomes.")
        print("Paper-only. No trades/orders/private API.")
        return
    predictions["timestamp"] = pd.to_numeric(predictions["timestamp"], errors="coerce")
    predictions = predictions.dropna(subset=["timestamp"]).copy()
    predictions["timestamp"] = predictions["timestamp"].astype(np.int64)
    scored = attach_actual_returns(predictions, snapshots)
    scored = prepare_predictions(scored)
    model_keys = sorted(scored["model_key"].dropna().unique().tolist())
    champion_key = identify_champion_key(scored)
    challenger_keys = [
        key
        for key in model_keys
        if key != champion_key
        and scored.loc[scored["model_key"] == key, "model_role"].astype(str).iloc[-1].startswith("challenger")
    ]
    diagnostics = model_key_diagnostics(scored) if len(scored) else pd.DataFrame()
    global_shared_rows = global_shared_scored_rows(scored, model_keys)
    print(f"Raw prediction rows: {len(predictions)}")
    print(f"Active prediction rows after registry filter/dedup: {len(scored)}")
    print(f"Model keys found: {len(model_keys)}")
    print(f"Global all-model shared scored rows: {global_shared_rows}")
    print("Rows by model key")
    for _, row in diagnostics.iterrows():
        print(
            f"- {row['model_role']} {row['model_id']} "
            f"feature_spec={row.get('feature_spec', '')} target_spec={row.get('target_spec', '')} "
            f"model_spec={row.get('model_spec', '')}: rows={int(row['rows'])} "
            f"fresh={int(row['fresh_rows'])} stale={int(row['stale_rows'])} "
            f"scored={int(row['scored_rows'])} unscored={int(row['unscored_rows'])} "
            f"first={int(row['first_timestamp'])} last={int(row['last_timestamp'])}"
        )
    if champion_key is None:
        print("Blocked: no champion model key found in shadow predictions.")
        print("Paper-only. No trades/orders/private API.")
        return
    if not challenger_keys:
        print("Blocked: no challenger model keys found in active shadow predictions.")
        print("Paper-only. No trades/orders/private API.")
        return
    print(f"Champion model key: {champion_key}")
    print(f"Pairwise challengers: {len(challenger_keys)}")
    evaluation, pair_diagnostics = pairwise_evaluation_rows(scored, champion_key, challenger_keys)
    print("Pairwise shared diagnostics")
    for _, row in pair_diagnostics.iterrows():
        print(
            f"- {row['challenger_role']} {row['challenger_model_id']}: "
            f"shared_ts={int(row['pairwise_shared_timestamps'])} "
            f"realized_ts={int(row['pairwise_shared_realized_timestamps'])} "
            f"realized_rows={int(row['pairwise_shared_realized_rows'])} "
            f"stale_excluded_rows={int(row['pairwise_stale_excluded_rows'])} "
            f"unscored_missing_outcome={int(row['unscored_rows_due_to_missing_future_outcome'])} "
            f"unscored_missing_future_snapshot={int(row['unscored_rows_due_to_missing_snapshot_future_price'])} "
            f"experiment_family_matches={bool(row.get('experiment_family_matches', False))} "
            f"mismatch_fields={row.get('experiment_family_mismatch_fields', '')}"
        )
    if len(evaluation) == 0:
        print("Blocked: no pairwise champion/challenger comparisons have realized outcomes yet.")
        print("Paper-only. No trades/orders/private API.")
        return
    evaluation["lift_vs_champion_avg_return_bps"] = np.where(
        evaluation["comparison_role"].eq("challenger"),
        evaluation["avg_return_bps"] - evaluation["opponent_avg_return_bps"],
        evaluation["opponent_avg_return_bps"] - evaluation["avg_return_bps"],
    )
    evaluation["lift_vs_champion_win_rate"] = np.where(
        evaluation["comparison_role"].eq("challenger"),
        evaluation["win_rate"] - evaluation["opponent_win_rate"],
        evaluation["opponent_win_rate"] - evaluation["win_rate"],
    )
    evaluation["min_active_rows_passed"] = evaluation["active_rows"] >= MIN_ACTIVE_ROWS
    evaluation["min_shared_rows_passed"] = evaluation["pairwise_shared_realized_timestamps"] >= MIN_SHARED_ROWS
    evaluation["no_freshness_or_horizon_issues"] = (
        (evaluation["freshness_issue_rows"] == 0)
        & (evaluation["pairwise_stale_excluded_rows"] == 0)
        & (evaluation["horizon_issue_rows"] == 0)
    )
    atomic_write_csv(evaluation, EVALUATION_PATH)
    print(f"Evaluation output: {EVALUATION_PATH}")
    print("Pairwise model comparison on champion/challenger shared timestamps")
    for _, row in evaluation.sort_values(["comparison_id", "comparison_role"]).iterrows():
        win_text = "n/a" if not np.isfinite(float(row["win_rate"])) else f"{row['win_rate']:.2%}"
        print(
            f"- {row['comparison_role']} {row['model_id']}: "
            f"feature_spec={row.get('feature_spec', '')} target_spec={row.get('target_spec', '')} "
            f"model_spec={row.get('model_spec', '')} "
            f"pair={row['comparison_id']} shared_ts={int(row['pairwise_shared_realized_timestamps'])} "
            f"active={int(row['active_rows'])} "
            f"coverage={row['coverage']:.2%} avg_return={row['avg_return_bps']:+.4f}bps "
            f"win_rate={win_text} freshness_issues={int(row['freshness_issue_rows'])} "
            f"stale_excluded={int(row['pairwise_stale_excluded_rows'])} "
            f"horizon_issues={int(row['horizon_issue_rows'])} "
            f"promotion_eligible={bool(row.get('promotion_eligible', False))} "
            f"promotion_block_reason={row.get('promotion_block_reason', '')}"
        )
    print(maybe_promote(evaluation))
    print("Promotion, if enabled, only updates selected_model.json. No trades/orders/private API.")


if __name__ == "__main__":
    main()
