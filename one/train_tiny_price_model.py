import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from tiny_price_feature_utils import (
    assert_numeric_feature_columns,
    feature_schema_hash,
    select_model_feature_columns,
    select_target_columns,
    slugify,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
FEATURE_SET = os.getenv("PRICE_TINY_FEATURE_SET", "tiny_price_v1").strip().lower()
HORIZON_SECONDS = int(os.getenv("PRICE_TINY_HORIZON_SECONDS", "1"))
LOOKBACK_PROFILE_ENV = os.getenv("PRICE_TINY_LOOKBACK_PROFILE", "short").strip().lower()
EPOCHS = int(os.getenv("PRICE_TINY_EPOCHS", "50"))
BATCH_SIZE = int(os.getenv("PRICE_TINY_BATCH_SIZE", "512"))
LEARNING_RATE = float(os.getenv("PRICE_TINY_LEARNING_RATE", "0.001"))
RIDGE_L2 = float(os.getenv("PRICE_TINY_RIDGE_L2", "0.001"))
MIN_ROWS = int(os.getenv("PRICE_TINY_MIN_ROWS", "500"))
MAX_TRAIN_ROWS = int(os.getenv("PRICE_TINY_MAX_TRAIN_ROWS", "50000"))
CONFIDENCE_THRESHOLD = float(os.getenv("PRICE_TINY_CONFIDENCE_THRESHOLD", "0.55"))
SHADOW_REGISTRATION_THRESHOLD = float(os.getenv("PRICE_TINY_THRESHOLD", str(CONFIDENCE_THRESHOLD)))
PRICE_TINY_FLAT_BPS = float(os.getenv("PRICE_TINY_FLAT_BPS", "0.10"))
SELECTION_OBJECTIVE = os.getenv("PRICE_TINY_SELECTION_OBJECTIVE", "mae").strip().lower()
MAX_RMSE_WORSENING_RATIO = float(os.getenv("PRICE_TINY_MAX_RMSE_WORSENING_RATIO", "0.10"))
MIN_PRED_DELTA_STD_BPS = float(os.getenv("PRICE_TINY_MIN_PRED_DELTA_STD_BPS", "0.001"))
CALIBRATION_INVERSION_MARGIN = float(os.getenv("PRICE_TINY_CALIBRATION_INVERSION_MARGIN", "0.02"))
CALIBRATION_MIN_BUCKET_ROWS = int(os.getenv("PRICE_TINY_CALIBRATION_MIN_BUCKET_ROWS", "30"))
MIN_THRESHOLD_ROWS_INTERESTING = int(os.getenv("PRICE_TINY_MIN_THRESHOLD_ROWS_INTERESTING", "100"))
MIN_THRESHOLD_ROWS_STABLE = int(os.getenv("PRICE_TINY_MIN_THRESHOLD_ROWS_STABLE", "300"))
CONFIDENCE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
REGRESSION_TARGET_CLIP_ENV = os.getenv("PRICE_TINY_REGRESSION_TARGET_CLIP_BPS", "").strip()
PRICE_TINY_REGRESSION_TARGET_CLIP_BPS = (
    float(REGRESSION_TARGET_CLIP_ENV) if REGRESSION_TARGET_CLIP_ENV else None
)
REGRESSION_MAE_WORSE_THAN_ZERO_LIMIT = float(os.getenv("PRICE_TINY_REGRESSION_MAE_WORSE_THAN_ZERO_LIMIT", "0.10"))
REGRESSION_ONE_SIDED_MIN_ACTIVE_PCT = float(os.getenv("PRICE_TINY_REGRESSION_ONE_SIDED_MIN_ACTIVE_PCT", "0.95"))
REGRESSION_UNCALIBRATED_THRESHOLD_BPS = float(os.getenv("PRICE_TINY_REGRESSION_UNCALIBRATED_THRESHOLD_BPS", "8"))
REGRESSION_UNCALIBRATED_KEEP_PCT = float(os.getenv("PRICE_TINY_REGRESSION_UNCALIBRATED_KEEP_PCT", "0.90"))
HIDDEN_SIZES = [
    int(value)
    for value in os.getenv("PRICE_TINY_HIDDEN_SIZES", "4,8,16,20").split(",")
    if value.strip()
]
PRICE_TINY_MODEL_SPECS = [
    value.strip()
    for value in os.getenv("PRICE_TINY_MODEL_SPECS", "").split(",")
    if value.strip()
]
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
TRAINING_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_training_rows.csv"
LATEST_TRAINING_METADATA_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_training_rows_latest.json"
TRAINING_PATH_ENV = os.getenv("PRICE_TINY_TRAINING_ROWS_PATH", "").strip()
FORWARD_TEST_PREDICTIONS_PATH = Path(
    os.getenv("PRICE_TINY_FORWARD_TEST_PREDICTIONS_PATH", VENUE_DIR / f"{SYMBOL}_tiny_price_forward_test_predictions.csv")
)
if not FORWARD_TEST_PREDICTIONS_PATH.is_absolute():
    FORWARD_TEST_PREDICTIONS_PATH = PROJECT_ROOT / FORWARD_TEST_PREDICTIONS_PATH
FORWARD_TEST_PREDICTION_ARCHIVE_DIR = Path(
    os.getenv("PRICE_TINY_FORWARD_TEST_PREDICTION_ARCHIVE_DIR", VENUE_DIR / "tiny_price_forward_test_predictions")
)
if not FORWARD_TEST_PREDICTION_ARCHIVE_DIR.is_absolute():
    FORWARD_TEST_PREDICTION_ARCHIVE_DIR = PROJECT_ROOT / FORWARD_TEST_PREDICTION_ARCHIVE_DIR
CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "tiny_price" / (PRIMARY_VENUE or "legacy")
SELECTED_ROOT = PROJECT_ROOT / "models" / "selected" / SYMBOL / "tiny_price" / (PRIMARY_VENUE or "legacy")
CANDIDATE_REGISTRY_PATH = SELECTED_ROOT / "candidate_registry.json"
MAX_ACTIVE_CHALLENGERS = int(os.getenv("PRICE_TINY_MAX_ACTIVE_CHALLENGERS", "3"))
AUTO_REGISTER_SHADOW_CHALLENGER = os.getenv("TRAIN_PRICE_TINY_MODEL", "").strip().lower() in {"1", "true", "yes"} or os.getenv(
    "PRICE_TINY_AUTO_REGISTER_CHALLENGERS", ""
).strip().lower() in {"1", "true", "yes"}
PRICE_TINY_ALLOWED_SHADOW_MODEL_TYPES = [
    value.strip()
    for value in os.getenv("PRICE_TINY_ALLOWED_SHADOW_MODEL_TYPES", "ridge_logistic,logistic_regression").split(",")
    if value.strip()
]
PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES = [
    value.strip()
    for value in os.getenv("PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES", "").split(",")
    if value.strip()
]
CLASS_NAMES = ["down", "flat", "up"]
REGRESSION_TARGET_METHODS = {"return_bps", "next_mid_delta_bps", "next_mid_log_return"}
SUPPORTED_EXPLICIT_MODEL_SPECS = {"ridge_logistic", "ridge_regression"}
TARGET_LOG_COLUMN = f"target_next_mid_log_return_{HORIZON_SECONDS}s"
TARGET_DELTA_COLUMN = f"target_next_mid_delta_bps_{HORIZON_SECONDS}s"
TARGET_DIRECTION_COLUMN = f"target_next_mid_direction_{HORIZON_SECONDS}s"
SELECTED_TARGET_COLUMNS = []
SELECTED_CLASSIFICATION_TARGET_COLUMN = TARGET_DIRECTION_COLUMN
REALIZED_RETURN_COLUMN = TARGET_DELTA_COLUMN
RETURN_HEAD_TARGET_COLUMN = TARGET_DELTA_COLUMN
TARGET_LABEL_METHOD = "direction"


def requested_model_specs():
    if PRICE_TINY_MODEL_SPECS:
        return PRICE_TINY_MODEL_SPECS
    if TARGET_LABEL_METHOD in REGRESSION_TARGET_METHODS:
        return ["ridge_regression"]
    return ["ridge_logistic", *[f"mlp_{hidden}" for hidden in HIDDEN_SIZES]]


def validate_model_specs(model_specs):
    invalid = []
    for spec in model_specs:
        if spec in SUPPORTED_EXPLICIT_MODEL_SPECS:
            continue
        if spec.startswith("mlp_") and hidden_layers_from_model_spec(spec):
            continue
        invalid.append(spec)
    if invalid:
        allowed = sorted(SUPPORTED_EXPLICIT_MODEL_SPECS) + ["mlp_<hidden>", "mlp_<hidden>_<hidden>"]
        raise RuntimeError(
            f"Unsupported PRICE_TINY_MODEL_SPECS: {invalid}. "
            f"Allowed specs: {allowed}"
        )


def is_regression_target_method(method=None):
    return (method or TARGET_LABEL_METHOD or "").strip().lower() in REGRESSION_TARGET_METHODS


def hidden_layers_from_model_spec(spec):
    if not spec.startswith("mlp_"):
        return []
    return [int(part) for part in spec.split("_")[1:] if part.isdigit()]


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


def atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_training_path():
    if TRAINING_PATH_ENV:
        path = Path(TRAINING_PATH_ENV)
        return path if path.is_absolute() else PROJECT_ROOT / path
    metadata = load_json_if_exists(LATEST_TRAINING_METADATA_PATH) or {}
    metadata_path = str(metadata.get("training_rows_path", "")).strip()
    if metadata_path:
        path = Path(metadata_path)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return TRAINING_PATH


def first_nonempty_string(frame, column, default=""):
    if column not in frame.columns or len(frame) == 0:
        return default
    values = frame[column].dropna()
    if len(values) == 0:
        return default
    text = str(values.iloc[0]).strip()
    return text or default


def first_numeric_value(frame, column, default=0.0):
    if column not in frame.columns or len(frame) == 0:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if len(values) == 0:
        return default
    return float(values.iloc[0])


def bool_from_frame(frame, column, default=False):
    if column not in frame.columns or len(frame) == 0:
        return default
    values = frame[column].dropna()
    if len(values) == 0:
        return default
    text = str(values.iloc[0]).strip().lower()
    return text in {"1", "true", "yes"}


def selected_target_columns_for_method(method, horizon):
    method = (method or "direction").strip().lower()
    if method == "move_before_adverse_net_aware":
        return [f"target_move_before_adverse_net_aware_{horizon}s"]
    if method == "move_before_adverse":
        return [f"target_move_before_adverse_{horizon}s"]
    if method == "first_touch":
        return [f"target_first_touch_direction_{horizon}s"]
    if method == "chop_no_trade":
        return [f"target_chop_no_trade_{horizon}s"]
    if method == "instability":
        return [f"target_instability_{horizon}s"]
    if method == "next_mid_delta_bps":
        return [f"target_next_mid_delta_bps_{horizon}s"]
    if method == "next_mid_log_return":
        return [f"target_next_mid_log_return_{horizon}s"]
    if method == "return_bps":
        return [f"target_return_bps_{horizon}s"]
    return [f"target_direction_{horizon}s"]


def output_semantics_payload(target_spec_name, target_label_method, selected_target_columns, realized_return_column, return_head_column):
    is_return_target = is_regression_target_method(target_label_method)
    is_instability_target = target_label_method == "instability"
    return {
        "target_spec": target_spec_name,
        "selected_target_columns": selected_target_columns,
        "selected_classification_target_column": selected_target_columns[0] if selected_target_columns else "",
        "selected_classification_target_method": target_label_method,
        "target_role": "risk_gate" if is_instability_target else "direction_or_return",
        "realized_return_column": realized_return_column,
        "return_head_target_column": return_head_column,
        "regression_metrics_key": "regression_return_metrics" if is_return_target else "auxiliary_return_head_metrics",
        "realized_return_metrics_note": (
            "Instability is a paper-only risk/gating target; realized return metrics are descriptive, not directional utility."
            if is_instability_target
            else "Paper utility metrics use realized future return even when the selected classification target is path/first-touch/chop semantics."
        ),
        "paper_only": True,
    }


def safe_metric_float(value):
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def compact_threshold_report_rows(rows):
    compact = []
    for row in rows or []:
        compact.append(
            {
                "threshold": safe_metric_float(row.get("threshold", np.nan)),
                "rows": int(row.get("rows_kept", row.get("rows", 0)) or 0),
                "avg_realized_return_bps": safe_metric_float(
                    row.get("avg_realized_return_bps", row.get("avg_return_bps", np.nan))
                ),
                "up_return_bps": safe_metric_float(row.get("predicted_up_return_bps", np.nan)),
                "down_return_bps": safe_metric_float(row.get("predicted_down_return_bps", np.nan)),
                "stable": bool(row.get("threshold_stable_candidate", row.get("stable", False))),
                "interesting": bool(row.get("threshold_interesting", row.get("interesting", False))),
            }
        )
    return compact


def compact_threshold_summary(row):
    if not isinstance(row, dict) or not row:
        return {}
    return compact_threshold_report_rows([row])[0]


def split_metric_artifact_fields(test_metrics, validation_metrics):
    threshold_reports = compact_threshold_report_rows(
        test_metrics.get("confidence_threshold_directional_report", [])
    )
    instability_test = test_metrics.get("instability_target_metrics", {})
    instability_validation = validation_metrics.get("instability_target_metrics", {}) if isinstance(validation_metrics, dict) else {}
    classification_metrics = {
        "selected_classification_target_column": SELECTED_CLASSIFICATION_TARGET_COLUMN,
        "sign_accuracy_excluding_flat": safe_metric_float(test_metrics.get("sign_accuracy_excluding_flat", np.nan)),
        "directional_win_rate": safe_metric_float(test_metrics.get("directional_win_rate", np.nan)),
        "validation_sign_acc": safe_metric_float(validation_metrics.get("sign_accuracy_excluding_flat", np.nan)),
        "instability_precision": safe_metric_float(instability_test.get("precision", np.nan)),
        "instability_recall": safe_metric_float(instability_test.get("recall", np.nan)),
        "instability_f1": safe_metric_float(instability_test.get("f1", np.nan)),
        "validation_instability_f1": safe_metric_float(instability_validation.get("f1", np.nan)),
    }
    realized_return_metrics = {
        "realized_return_column": REALIZED_RETURN_COLUMN,
        "threshold_reports": threshold_reports,
        "best_threshold_any_rows": compact_threshold_summary(test_metrics.get("best_threshold_any_rows", {})),
        "best_threshold_min_100_rows": compact_threshold_summary(test_metrics.get("best_threshold_min_100_rows", {})),
        "best_threshold_min_300_rows": compact_threshold_summary(test_metrics.get("best_threshold_min_300_rows", {})),
    }
    auxiliary_return_head_metrics = {
        "return_head_target_column": RETURN_HEAD_TARGET_COLUMN,
        "mae_bps": safe_metric_float(test_metrics.get("mae_bps", np.nan)),
        "rmse_bps": safe_metric_float(test_metrics.get("rmse_bps", np.nan)),
        "zero_return_baseline_mae_bps": safe_metric_float(
            test_metrics.get("zero_return_baseline_mae_bps", test_metrics.get("baselines", {}).get("zero_return_mae_bps", np.nan))
        ),
        "price_candidate_useful": bool(test_metrics.get("price_candidate_useful", False)),
    }
    return classification_metrics, realized_return_metrics, auxiliary_return_head_metrics


def model_type_allowed(model_name):
    return str(model_name) in set(PRICE_TINY_ALLOWED_SHADOW_MODEL_TYPES)


def threshold_report_for_registration(metrics):
    threshold_rows = metrics.get("confidence_threshold_directional_report", [])
    eligible = [
        row
        for row in threshold_rows
        if float(row.get("threshold", -1.0)) >= SHADOW_REGISTRATION_THRESHOLD
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda row: float(row.get("threshold", float("inf"))))


def registration_gate_status(artifact):
    target_spec = artifact.get("target_spec", {}) if isinstance(artifact, dict) else {}
    target_method = str(target_spec.get("label_construction_method", artifact.get("target_label_method", ""))).strip().lower()
    if target_method == "instability":
        return False, "instability_risk_gate_not_registerable", None
    selected_model = str(artifact.get("selected_model_name", ""))
    if not model_type_allowed(selected_model):
        return False, "disallowed_model_type", None
    metrics = artifact.get("forward_test_metrics", {})
    price_useful = bool(metrics.get("price_candidate_useful", False))
    direction_useful = bool(metrics.get("direction_candidate_useful", False))
    if not price_useful and not direction_useful:
        return False, "candidate_not_useful", None
    threshold_row = threshold_report_for_registration(metrics)
    if threshold_row is None:
        return False, f"missing_threshold_report_at_or_above_{SHADOW_REGISTRATION_THRESHOLD:.2f}", None
    avg_return = float(
        threshold_row.get(
            "avg_return_bps",
            threshold_row.get("avg_realized_return_bps", float("nan")),
        )
    )
    if not np.isfinite(avg_return) or avg_return <= 0:
        return False, f"threshold_avg_return_not_positive:{avg_return}", threshold_row
    if not bool(threshold_row.get("threshold_interesting", False)):
        return False, "threshold_report_not_interesting", threshold_row
    if not bool(threshold_row.get("threshold_stable_candidate", False)):
        return False, "threshold_report_not_stable", threshold_row
    return True, "passed_shadow_registration_gates", threshold_row


def retire_disallowed_challengers(registry):
    challengers = []
    retired = list(registry.get("retired_challengers", []))
    retired_count = 0
    for item in registry.get("challengers", []):
        if not isinstance(item, dict):
            challengers.append(item)
            continue
        path = item.get("model_path", "")
        artifact = load_json_if_exists(path) if path else None
        selected = str(artifact.get("selected_model_name", "")) if artifact else ""
        if artifact and not model_type_allowed(selected):
            retired_item = dict(item)
            retired_item["retired"] = True
            retired_item["retired_at"] = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            retired_item["retirement_reason"] = "disallowed_model_type"
            retired_item["selected_model_name"] = selected
            retired.append(retired_item)
            retired_count += 1
        else:
            challengers.append(item)
    registry["challengers"] = challengers
    registry["retired_challengers"] = retired
    return retired_count


def register_shadow_challenger(model_path, artifact):
    status = {
        "registered_challenger": False,
        "registration_block_reason": "",
        "registry_path": str(CANDIDATE_REGISTRY_PATH),
    }
    if not AUTO_REGISTER_SHADOW_CHALLENGER:
        status["registration_block_reason"] = "auto_registration_disabled"
        return status
    registry = load_json_if_exists(CANDIDATE_REGISTRY_PATH) or {
        "paper_only": True,
        "created_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE or "legacy",
        "champion_model_path": "",
        "champion_policy": {},
        "challengers": [],
        "max_active_challengers": MAX_ACTIVE_CHALLENGERS,
    }
    retired_count = retire_disallowed_challengers(registry)
    registry["updated_at"] = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    registry["max_active_challengers"] = int(registry.get("max_active_challengers", MAX_ACTIVE_CHALLENGERS))
    new_path = str(Path(model_path))
    champion_path = str(registry.get("champion_model_path", ""))
    challengers = [
        item
        for item in registry.get("challengers", [])
        if str(item.get("model_path") if isinstance(item, dict) else item) != new_path
    ]
    if champion_path and Path(champion_path).resolve() == Path(new_path).resolve():
        registry["challengers"] = challengers
        atomic_write_json(registry, CANDIDATE_REGISTRY_PATH)
        status["registration_block_reason"] = "candidate_matches_champion"
        status["retired_disallowed_challengers"] = retired_count
        return status
    passed, reason, threshold_row = registration_gate_status(artifact)
    if not passed:
        registry["challengers"] = challengers[: registry["max_active_challengers"]]
        atomic_write_json(registry, CANDIDATE_REGISTRY_PATH)
        status["registration_block_reason"] = reason
        status["retired_disallowed_challengers"] = retired_count
        status["selected_model_name"] = artifact.get("selected_model_name", "")
        if threshold_row:
            status["registration_threshold_report"] = threshold_row
        return status
    challengers.insert(
        0,
        {
            "model_path": new_path,
            "policy": {
                "threshold": float(SHADOW_REGISTRATION_THRESHOLD),
                "horizon": int(artifact.get("horizon_seconds", HORIZON_SECONDS)),
                "feature_set": str(artifact.get("feature_set_name", FEATURE_SET)),
                "lookback_profile": str(artifact.get("lookback_profile", LOOKBACK_PROFILE_ENV)),
                "regime_gate": os.getenv("PRICE_TINY_REGIME_GATE", "no_gate").strip().lower(),
            },
            "added_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "train_tiny_price_model",
        },
    )
    registry["challengers"] = challengers[: registry["max_active_challengers"]]
    atomic_write_json(registry, CANDIDATE_REGISTRY_PATH)
    status["registered_challenger"] = True
    status["registration_block_reason"] = ""
    status["retired_disallowed_challengers"] = retired_count
    status["selected_model_name"] = artifact.get("selected_model_name", "")
    if threshold_row:
        status["registration_threshold_report"] = threshold_row
    return status


def now_tag():
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def prediction_archive_paths(target_spec_name, selected_model_name, schema_hash, candidate_tag):
    target_slug = slugify(target_spec_name, f"target_{HORIZON_SECONDS}s")
    model_slug = slugify(selected_model_name, "model")
    schema_slug = slugify(schema_hash, "schema")
    base_name = f"{SYMBOL}_tiny_price_forward_test_predictions__{target_slug}__{model_slug}__{schema_slug}"
    stable_path = FORWARD_TEST_PREDICTION_ARCHIVE_DIR / f"{base_name}.csv"
    timestamped_path = FORWARD_TEST_PREDICTION_ARCHIVE_DIR / f"{base_name}__{candidate_tag}.csv"
    return stable_path, timestamped_path


def softmax(values):
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(np.clip(values, -40, 40))
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def standardize(train_x, validation_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (train_x - mean) / std, (validation_x - mean) / std, (test_x - mean) / std, mean, std


def split(frame):
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    train_end = int(len(frame) * 0.60)
    validation_end = int(len(frame) * 0.80)
    return frame.iloc[:train_end].copy(), frame.iloc[train_end:validation_end].copy(), frame.iloc[validation_end:].copy()


def direction_to_class(values):
    values = np.asarray(values, dtype=np.int64)
    return np.where(values < 0, 0, np.where(values > 0, 2, 1))


def class_to_direction(classes):
    classes = np.asarray(classes, dtype=np.int64)
    return np.where(classes == 0, -1, np.where(classes == 2, 1, 0))


def direction_from_delta(delta_bps):
    delta_bps = np.asarray(delta_bps, dtype=np.float64)
    return np.where(delta_bps > PRICE_TINY_FLAT_BPS, 1, np.where(delta_bps < -PRICE_TINY_FLAT_BPS, -1, 0))


def direction_from_regression_return(return_bps):
    return_bps = np.asarray(return_bps, dtype=np.float64)
    return np.where(return_bps > 0, 1, np.where(return_bps < 0, -1, 0))


def target_values_as_bps(frame, column, method=None):
    method = (method or TARGET_LABEL_METHOD or "").strip().lower()
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if method == "next_mid_log_return":
        return (np.exp(values) - 1.0) * 10000.0
    return values


def prediction_log_from_bps(predicted_bps):
    predicted_bps = np.asarray(predicted_bps, dtype=np.float64)
    return np.log1p(np.clip(predicted_bps / 10000.0, -0.99, 10.0))


def regression_correlation(predicted_bps, actual_bps):
    predicted_bps = np.asarray(predicted_bps, dtype=np.float64)
    actual_bps = np.asarray(actual_bps, dtype=np.float64)
    mask = np.isfinite(predicted_bps) & np.isfinite(actual_bps)
    if mask.sum() < 2:
        return np.nan
    if np.nanstd(predicted_bps[mask]) < 1e-12 or np.nanstd(actual_bps[mask]) < 1e-12:
        return np.nan
    return float(np.corrcoef(predicted_bps[mask], actual_bps[mask])[0, 1])


def distribution_stats(values, flat_threshold_bps=0.0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "rows": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p1": np.nan,
            "p5": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "positive_count": 0,
            "negative_count": 0,
            "flat_count": 0,
            "zero_count": 0,
        }
    quantiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    flat_threshold_bps = max(float(flat_threshold_bps), 0.0)
    return {
        "rows": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p1": float(quantiles[0]),
        "p5": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "positive_count": int((values > flat_threshold_bps).sum()),
        "negative_count": int((values < -flat_threshold_bps).sum()),
        "flat_count": int((np.abs(values) <= flat_threshold_bps).sum()),
        "zero_count": int((np.abs(values) <= 1e-12).sum()),
    }


def split_distribution_stats(train, validation, test, column, method=None):
    method = method or TARGET_LABEL_METHOD
    return {
        "train": distribution_stats(target_values_as_bps(train, column, method), PRICE_TINY_FLAT_BPS),
        "validation": distribution_stats(target_values_as_bps(validation, column, method), PRICE_TINY_FLAT_BPS),
        "test": distribution_stats(target_values_as_bps(test, column, method), PRICE_TINY_FLAT_BPS),
    }


def print_distribution_stats(title, stats_by_split):
    print(title)
    for split_name, stats in stats_by_split.items():
        print(
            f"- {split_name}: rows={stats['rows']} mean={stats['mean']:.4f} "
            f"median={stats['median']:.4f} std={stats['std']:.4f} "
            f"min={stats['min']:.4f} p1={stats['p1']:.4f} p5={stats['p5']:.4f} "
            f"p25={stats['p25']:.4f} p50={stats['p50']:.4f} p75={stats['p75']:.4f} "
            f"p95={stats['p95']:.4f} p99={stats['p99']:.4f} max={stats['max']:.4f} "
            f"pos={stats['positive_count']} neg={stats['negative_count']} flat={stats['flat_count']} zero={stats['zero_count']}"
        )


def target_clipping_report(unclipped_values, clipped_values, clip_bps):
    unclipped_values = np.asarray(unclipped_values, dtype=np.float64)
    clipped_values = np.asarray(clipped_values, dtype=np.float64)
    changed = np.isfinite(unclipped_values) & np.isfinite(clipped_values) & (np.abs(unclipped_values - clipped_values) > 1e-12)
    return {
        "enabled": clip_bps is not None,
        "clip_bps": float(clip_bps) if clip_bps is not None else np.nan,
        "clipped_count": int(changed.sum()),
        "clipped_pct": float(changed.mean()) if len(changed) else 0.0,
        "unclipped_distribution": distribution_stats(unclipped_values, PRICE_TINY_FLAT_BPS),
        "clipped_distribution": distribution_stats(clipped_values, PRICE_TINY_FLAT_BPS),
    }


def train_ridge(x, y):
    x_augmented = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(x_augmented.shape[1]) * RIDGE_L2
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(x_augmented.T @ x_augmented + penalty) @ x_augmented.T @ y
    return weights[1:], float(weights[0])


def train_softmax_logistic(x, classes):
    rng = np.random.default_rng(RANDOM_SEED)
    w = rng.normal(0, 0.01, (x.shape[1], 3))
    b = np.zeros(3)
    y = np.eye(3)[classes]
    counts = np.maximum(y.sum(axis=0), 1.0)
    class_weights = len(y) / (3.0 * counts)
    for _ in range(max(50, EPOCHS)):
        p = softmax(x @ w + b)
        weighted_error = (p - y) * class_weights.reshape(1, -1)
        w -= 0.05 * ((x.T @ weighted_error) / len(x) + RIDGE_L2 * w)
        b -= 0.05 * weighted_error.mean(axis=0)
    return w, b


def init_mlp(input_size, hidden_size, rng):
    return {
        "w1": rng.normal(0, np.sqrt(2 / max(1, input_size)), (input_size, hidden_size)),
        "b1": np.zeros(hidden_size),
        "w_delta": rng.normal(0, np.sqrt(2 / max(1, hidden_size)), (hidden_size, 1)),
        "b_delta": np.zeros(1),
        "w_class": rng.normal(0, np.sqrt(2 / max(1, hidden_size)), (hidden_size, 3)),
        "b_class": np.zeros(3),
    }


def init_mlp_layers(input_size, hidden_layers, rng):
    weights = []
    biases = []
    previous = input_size
    for hidden in hidden_layers:
        weights.append(rng.normal(0, np.sqrt(2 / max(1, previous)), (previous, hidden)))
        biases.append(np.zeros(hidden))
        previous = hidden
    return {
        "hidden_layers": list(hidden_layers),
        "hidden_weights": weights,
        "hidden_biases": biases,
        "w_delta": rng.normal(0, np.sqrt(2 / max(1, previous)), (previous, 1)),
        "b_delta": np.zeros(1),
        "w_class": rng.normal(0, np.sqrt(2 / max(1, previous)), (previous, 3)),
        "b_class": np.zeros(3),
    }


def mlp_forward(model, x):
    h_pre = x @ model["w1"] + model["b1"]
    h = np.maximum(h_pre, 0.0)
    delta = h @ model["w_delta"] + model["b_delta"]
    probs = softmax(h @ model["w_class"] + model["b_class"])
    return h_pre, h, delta.reshape(-1), probs


def mlp_layers_forward(model, x):
    hidden_weights = [np.asarray(value) for value in model["hidden_weights"]]
    hidden_biases = [np.asarray(value) for value in model["hidden_biases"]]
    pre_activations = []
    activations = [x]
    current = x
    for weight, bias in zip(hidden_weights, hidden_biases):
        pre = current @ weight + bias
        current = np.maximum(pre, 0.0)
        pre_activations.append(pre)
        activations.append(current)
    delta = current @ np.asarray(model["w_delta"]) + np.asarray(model["b_delta"])
    probs = softmax(current @ np.asarray(model["w_class"]) + np.asarray(model["b_class"]))
    return pre_activations, activations, delta.reshape(-1), probs


def train_mlp(x_train, y_delta_train, y_class_train, x_validation, y_delta_validation, hidden_size):
    rng = np.random.default_rng(RANDOM_SEED + hidden_size)
    model = init_mlp(x_train.shape[1], hidden_size, rng)
    y_class_one_hot = np.eye(3)[y_class_train]
    best_model = {key: value.copy() for key, value in model.items()}
    best_loss = float("inf")
    for _ in range(EPOCHS):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            xb = x_train[idx]
            yd = y_delta_train[idx]
            yc = y_class_one_hot[idx]
            h_pre, h, pred_delta, probs = mlp_forward(model, xb)
            d_delta = (2.0 * (pred_delta - yd) / max(1, len(xb))).reshape(-1, 1)
            d_class = (probs - yc) / max(1, len(xb))
            gradients = {
                "w_delta": h.T @ d_delta,
                "b_delta": d_delta.sum(axis=0),
                "w_class": h.T @ d_class,
                "b_class": d_class.sum(axis=0),
            }
            d_h = d_delta @ model["w_delta"].T + d_class @ model["w_class"].T
            d_h[h_pre <= 0] = 0.0
            gradients["w1"] = xb.T @ d_h
            gradients["b1"] = d_h.sum(axis=0)
            for key, gradient in gradients.items():
                model[key] -= LEARNING_RATE * gradient
        _, _, val_delta, val_probs = mlp_forward(model, x_validation)
        loss = float(np.mean((val_delta - y_delta_validation) ** 2) - 0.05 * np.mean(np.log(np.max(val_probs, axis=1) + 1e-8)))
        if loss < best_loss:
            best_loss = loss
            best_model = {key: value.copy() for key, value in model.items()}
    return best_model


def train_mlp_layers(x_train, y_delta_train, y_class_train, x_validation, y_delta_validation, hidden_layers):
    rng = np.random.default_rng(RANDOM_SEED + sum(hidden_layers) + len(hidden_layers))
    model = init_mlp_layers(x_train.shape[1], hidden_layers, rng)
    y_class_one_hot = np.eye(3)[y_class_train]
    best_model = {
        "hidden_layers": list(model["hidden_layers"]),
        "hidden_weights": [value.copy() for value in model["hidden_weights"]],
        "hidden_biases": [value.copy() for value in model["hidden_biases"]],
        "w_delta": model["w_delta"].copy(),
        "b_delta": model["b_delta"].copy(),
        "w_class": model["w_class"].copy(),
        "b_class": model["b_class"].copy(),
    }
    best_loss = float("inf")
    for _ in range(EPOCHS):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            xb = x_train[idx]
            yd = y_delta_train[idx]
            yc = y_class_one_hot[idx]
            pre_activations, activations, pred_delta, probs = mlp_layers_forward(model, xb)
            d_delta = (2.0 * (pred_delta - yd) / max(1, len(xb))).reshape(-1, 1)
            d_class = (probs - yc) / max(1, len(xb))
            grad_w_delta = activations[-1].T @ d_delta
            grad_b_delta = d_delta.sum(axis=0)
            grad_w_class = activations[-1].T @ d_class
            grad_b_class = d_class.sum(axis=0)
            d_h = d_delta @ model["w_delta"].T + d_class @ model["w_class"].T
            grad_hidden_weights = []
            grad_hidden_biases = []
            for layer_index in reversed(range(len(model["hidden_weights"]))):
                d_h = d_h.copy()
                d_h[pre_activations[layer_index] <= 0] = 0.0
                grad_hidden_weights.insert(0, activations[layer_index].T @ d_h)
                grad_hidden_biases.insert(0, d_h.sum(axis=0))
                d_h = d_h @ model["hidden_weights"][layer_index].T
            model["w_delta"] -= LEARNING_RATE * grad_w_delta
            model["b_delta"] -= LEARNING_RATE * grad_b_delta
            model["w_class"] -= LEARNING_RATE * grad_w_class
            model["b_class"] -= LEARNING_RATE * grad_b_class
            for layer_index in range(len(model["hidden_weights"])):
                model["hidden_weights"][layer_index] -= LEARNING_RATE * grad_hidden_weights[layer_index]
                model["hidden_biases"][layer_index] -= LEARNING_RATE * grad_hidden_biases[layer_index]
        _, _, val_delta, val_probs = mlp_layers_forward(model, x_validation)
        loss = float(np.mean((val_delta - y_delta_validation) ** 2) - 0.05 * np.mean(np.log(np.max(val_probs, axis=1) + 1e-8)))
        if loss < best_loss:
            best_loss = loss
            best_model = {
                "hidden_layers": list(model["hidden_layers"]),
                "hidden_weights": [value.copy() for value in model["hidden_weights"]],
                "hidden_biases": [value.copy() for value in model["hidden_biases"]],
                "w_delta": model["w_delta"].copy(),
                "b_delta": model["b_delta"].copy(),
                "w_class": model["w_class"].copy(),
                "b_class": model["b_class"].copy(),
            }
    return best_model


def predict_model(model_name, model, x, delta_mean, delta_std, log_model=None):
    if model_name == "ridge_regression":
        pred_delta = x @ np.asarray(model["target_weights"]) + float(model["target_bias"])
        pred_log = prediction_log_from_bps(pred_delta)
        pred_class_direction = direction_from_regression_return(pred_delta)
        confidence = confidence_from_delta(pred_delta)
        probs = np.full((len(x), 3), np.nan, dtype=np.float64)
        return pred_delta, pred_log, pred_class_direction, confidence, probs
    if model_name == "ridge_logistic":
        pred_delta = x @ np.asarray(model["delta_weights"]) + float(model["delta_bias"])
        pred_log = x @ np.asarray(model["log_weights"]) + float(model["log_bias"])
        probs = softmax(x @ np.asarray(model["class_weights"]) + np.asarray(model["class_bias"]))
    elif "hidden_weights" in model:
        converted = {
            "hidden_weights": [np.asarray(value) for value in model["hidden_weights"]],
            "hidden_biases": [np.asarray(value) for value in model["hidden_biases"]],
            "w_delta": np.asarray(model["w_delta"]),
            "b_delta": np.asarray(model["b_delta"]),
            "w_class": np.asarray(model["w_class"]),
            "b_class": np.asarray(model["b_class"]),
        }
        _, _, scaled_delta, probs = mlp_layers_forward(converted, x)
        pred_delta = scaled_delta * delta_std + delta_mean
        pred_log = np.log1p(np.clip(pred_delta / 10000.0, -0.99, 10.0))
    else:
        _, _, scaled_delta, probs = mlp_forward({key: np.asarray(value) for key, value in model.items()}, x)
        pred_delta = scaled_delta * delta_std + delta_mean
        pred_log = np.log1p(np.clip(pred_delta / 10000.0, -0.99, 10.0))
    pred_class = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)
    return pred_delta, pred_log, class_to_direction(pred_class), confidence, probs


def metrics(frame, pred_delta, pred_direction, confidence, classification_column=None, realized_return_column=None, return_head_column=None, target_label_method=None):
    classification_column = classification_column or SELECTED_CLASSIFICATION_TARGET_COLUMN
    realized_return_column = realized_return_column or REALIZED_RETURN_COLUMN
    return_head_column = return_head_column or RETURN_HEAD_TARGET_COLUMN
    target_label_method = target_label_method or TARGET_LABEL_METHOD
    actual_return = frame[realized_return_column].to_numpy(dtype=np.float64)
    if is_regression_target_method(target_label_method):
        actual_return_head = target_values_as_bps(frame, return_head_column, target_label_method)
        selected_target_bps = target_values_as_bps(frame, classification_column, target_label_method)
        actual_direction = direction_from_regression_return(selected_target_bps)
    else:
        actual_return_head = frame[return_head_column].to_numpy(dtype=np.float64)
        actual_direction = frame[classification_column].to_numpy(dtype=np.int64)
    if target_label_method == "instability":
        actual_unstable = actual_direction > 0
        predicted_unstable = np.asarray(pred_direction, dtype=np.int64) > 0
        true_positive = int((predicted_unstable & actual_unstable).sum())
        false_positive = int((predicted_unstable & ~actual_unstable).sum())
        false_negative = int((~predicted_unstable & actual_unstable).sum())
        true_negative = int((~predicted_unstable & ~actual_unstable).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        binary_accuracy = float((predicted_unstable == actual_unstable).mean()) if len(actual_unstable) else np.nan
        mae = float(np.mean(np.abs(pred_delta - actual_return_head))) if len(frame) else np.nan
        rmse = float(np.sqrt(np.mean((pred_delta - actual_return_head) ** 2))) if len(frame) else np.nan
        range_column = f"target_future_range_bps_{HORIZON_SECONDS}s"
        if range_column in frame.columns:
            actual_range = pd.to_numeric(frame[range_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            actual_range = np.abs(actual_return)
        stable_mask = ~actual_unstable
        unstable_mask = actual_unstable
        pred_delta_std = float(np.nanstd(pred_delta)) if len(pred_delta) else 0.0
        instability_metrics = {
            "target_column": classification_column,
            "percent_unstable": float(actual_unstable.mean()) if len(actual_unstable) else np.nan,
            "percent_predicted_unstable": float(predicted_unstable.mean()) if len(predicted_unstable) else np.nan,
            "accuracy": binary_accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
            "average_realized_return_bps_stable": float(actual_return[stable_mask].mean()) if stable_mask.any() else np.nan,
            "average_realized_return_bps_unstable": float(actual_return[unstable_mask].mean()) if unstable_mask.any() else np.nan,
            "average_abs_realized_return_bps_stable": float(np.abs(actual_return[stable_mask]).mean()) if stable_mask.any() else np.nan,
            "average_abs_realized_return_bps_unstable": float(np.abs(actual_return[unstable_mask]).mean()) if unstable_mask.any() else np.nan,
            "average_future_range_bps_stable": float(actual_range[stable_mask].mean()) if stable_mask.any() else np.nan,
            "average_future_range_bps_unstable": float(actual_range[unstable_mask].mean()) if unstable_mask.any() else np.nan,
            "note": "Instability is a paper-only risk/gating target, not a direction model.",
        }
        regression_metrics = {
            "target_column": return_head_column,
            "mae_bps": mae,
            "rmse_bps": rmse,
        }
        classification_metrics = {
            "target_column": classification_column,
            "target_label_method": target_label_method,
            "risk_gate_accuracy": binary_accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        realized_return_metrics = {
            "realized_return_column": realized_return_column,
            "forward_test_avg_return_bps": float(actual_return[predicted_unstable].mean()) if predicted_unstable.any() else 0.0,
            "average_realized_bps_when_pred_up": float(actual_return[predicted_unstable].mean()) if predicted_unstable.any() else np.nan,
            "average_realized_bps_when_pred_down": np.nan,
        }
        return {
            "rows": int(len(frame)),
            "target_label_method": target_label_method,
            "mae_bps": mae,
            "rmse_bps": rmse,
            "directional_accuracy": binary_accuracy,
            "sign_accuracy_excluding_flat": binary_accuracy,
            "directional_win_rate": np.nan,
            "coverage": float(predicted_unstable.mean()) if len(predicted_unstable) else 0.0,
            "forward_test_avg_return_bps": realized_return_metrics["forward_test_avg_return_bps"],
            "average_realized_bps_when_pred_up": realized_return_metrics["average_realized_bps_when_pred_up"],
            "average_realized_bps_when_pred_down": realized_return_metrics["average_realized_bps_when_pred_down"],
            "average_confidence": float(np.mean(confidence)) if len(confidence) else np.nan,
            "prediction_count_above_confidence_threshold": int((confidence >= CONFIDENCE_THRESHOLD).sum()) if len(confidence) else 0,
            "pred_delta_std_bps": pred_delta_std,
            "prediction_distribution_collapsed": bool(pred_delta_std < MIN_PRED_DELTA_STD_BPS),
            "predicted_direction_count": int(len(set(np.asarray(pred_direction, dtype=np.int64).tolist()))),
            "classification_target_metrics": classification_metrics,
            "realized_return_metrics": realized_return_metrics,
            "auxiliary_return_head_metrics": regression_metrics,
            "instability_target_metrics": instability_metrics,
        }
    nonflat = actual_direction != 0
    directional = pred_direction != 0
    active = directional
    mae = float(np.mean(np.abs(pred_delta - actual_return_head)))
    rmse = float(np.sqrt(np.mean((pred_delta - actual_return_head) ** 2)))
    pred_delta_std = float(np.nanstd(pred_delta)) if len(pred_delta) else 0.0
    correlation = regression_correlation(pred_delta, actual_return_head)
    predicted_distribution = distribution_stats(pred_delta, 0.0)
    predicted_one_sided = bool(
        is_regression_target_method(target_label_method)
        and len(pred_delta) > 0
        and (
            predicted_distribution["positive_count"] == 0
            or predicted_distribution["negative_count"] == 0
            or max(predicted_distribution["positive_count"], predicted_distribution["negative_count"]) / max(len(pred_delta), 1)
            >= REGRESSION_ONE_SIDED_MIN_ACTIVE_PCT
        )
    )
    direction_values = set(np.asarray(pred_direction, dtype=np.int64).tolist())
    sign_accuracy = float((pred_direction[nonflat] == actual_direction[nonflat]).mean()) if nonflat.any() else np.nan
    directional_win = float((np.sign(pred_direction[active]) == np.sign(actual_return[active])).mean()) if active.any() else np.nan
    strategy_return_bps = np.where(active, actual_return * np.sign(pred_direction), 0.0)
    regression_metrics = {
        "target_column": return_head_column,
        "mae_bps": mae,
        "rmse_bps": rmse,
        "correlation": correlation,
    }
    classification_metrics = {
        "target_column": classification_column,
        "target_label_method": target_label_method,
        "directional_accuracy": float((pred_direction == actual_direction).mean()) if len(actual_direction) else np.nan,
        "sign_accuracy_excluding_flat": sign_accuracy,
        "directional_win_rate": directional_win,
    }
    realized_return_metrics = {
        "realized_return_column": realized_return_column,
        "forward_test_avg_return_bps": float(strategy_return_bps.mean()) if len(strategy_return_bps) else 0.0,
        "average_realized_bps_when_pred_up": float(actual_return[pred_direction > 0].mean()) if (pred_direction > 0).any() else np.nan,
        "average_realized_bps_when_pred_down": float(actual_return[pred_direction < 0].mean()) if (pred_direction < 0).any() else np.nan,
    }
    output = {
        "rows": int(len(frame)),
        "target_label_method": target_label_method,
        "mae_bps": mae,
        "rmse_bps": rmse,
        "directional_accuracy": classification_metrics["directional_accuracy"],
        "sign_accuracy_excluding_flat": sign_accuracy,
        "directional_win_rate": directional_win,
        "coverage": float(active.mean()) if len(active) else 0.0,
        "forward_test_avg_return_bps": realized_return_metrics["forward_test_avg_return_bps"],
        "average_realized_bps_when_pred_up": realized_return_metrics["average_realized_bps_when_pred_up"],
        "average_realized_bps_when_pred_down": realized_return_metrics["average_realized_bps_when_pred_down"],
        "average_confidence": float(np.mean(confidence)) if len(confidence) else np.nan,
        "prediction_count_above_confidence_threshold": int((confidence >= CONFIDENCE_THRESHOLD).sum()) if len(confidence) else 0,
        "pred_delta_std_bps": pred_delta_std,
        "predicted_vs_actual_return_correlation": correlation,
        "prediction_return_distribution": predicted_distribution,
        "predicted_up_count": predicted_distribution["positive_count"],
        "predicted_down_count": predicted_distribution["negative_count"],
        "predicted_zero_count": predicted_distribution["zero_count"],
        "predicted_one_sided": predicted_one_sided,
        "prediction_distribution_collapsed": bool(pred_delta_std < MIN_PRED_DELTA_STD_BPS and len(direction_values) <= 1),
        "predicted_direction_count": int(len(direction_values)),
        "classification_target_metrics": classification_metrics,
        "realized_return_metrics": realized_return_metrics,
    }
    if is_regression_target_method(target_label_method):
        output["regression_return_metrics"] = regression_metrics
    else:
        output["auxiliary_return_head_metrics"] = regression_metrics
    return output


def baseline_metrics(train, test, classification_column=None, realized_return_column=None, target_label_method=None):
    classification_column = classification_column or SELECTED_CLASSIFICATION_TARGET_COLUMN
    realized_return_column = realized_return_column or REALIZED_RETURN_COLUMN
    target_label_method = target_label_method or TARGET_LABEL_METHOD
    actual_delta = test[realized_return_column].to_numpy(dtype=np.float64)
    if is_regression_target_method(target_label_method):
        actual_direction = direction_from_regression_return(target_values_as_bps(test, classification_column, target_label_method))
        train_direction = direction_from_regression_return(target_values_as_bps(train, classification_column, target_label_method))
    else:
        actual_direction = test[classification_column].to_numpy(dtype=np.int64)
        train_direction = train[classification_column].to_numpy(dtype=np.int64) if len(train) else np.asarray([0])
    previous_return = test["feature_mid_return_1s"].to_numpy(dtype=np.float64) * 10000.0 if "feature_mid_return_1s" in test.columns else np.zeros(len(test))
    majority_direction = int(pd.Series(train_direction).mode().iloc[0]) if len(train_direction) else 0
    pressure_direction = np.sign(test["feature_aggressive_flow_pressure"].to_numpy(dtype=np.float64)) if "feature_aggressive_flow_pressure" in test.columns else np.zeros(len(test))
    nonflat = actual_direction != 0
    baselines = {
        "zero_return_mae_bps": float(np.mean(np.abs(actual_delta))),
        "zero_return_rmse_bps": float(np.sqrt(np.mean(actual_delta ** 2))),
        "previous_return_persists_mae_bps": float(np.mean(np.abs(previous_return - actual_delta))),
        "previous_return_persists_rmse_bps": float(np.sqrt(np.mean((previous_return - actual_delta) ** 2))),
        "always_up_win_rate": float((actual_delta > 0).mean()),
        "always_down_win_rate": float((actual_delta < 0).mean()),
        "majority_direction_accuracy": float((np.full(len(test), majority_direction) == actual_direction).mean()),
        "majority_direction_sign_accuracy_excluding_flat": float((np.full(len(test), majority_direction)[nonflat] == actual_direction[nonflat]).mean()) if nonflat.any() else np.nan,
        "majority_direction_win_rate": float((np.sign(majority_direction) == np.sign(actual_delta)).mean()),
        "follow_1s_pressure_win_rate": float((np.sign(pressure_direction) == np.sign(actual_delta)).mean()),
    }
    return baselines


def calibration_buckets(confidence, pred_direction, actual_direction):
    rows = []
    for low, high in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]:
        mask = (confidence >= low) & (confidence < high)
        if not mask.any():
            rows.append({"bucket": f"{low:.1f}-{min(high, 1.0):.1f}", "rows": 0})
            continue
        rows.append(
            {
                "bucket": f"{low:.1f}-{min(high, 1.0):.1f}",
                "rows": int(mask.sum()),
                "accuracy": float((pred_direction[mask] == actual_direction[mask]).mean()),
                "avg_confidence": float(confidence[mask].mean()),
            }
        )
    return rows


def weighted_bucket_accuracy(buckets, low_threshold=None, high_threshold=None):
    total = 0
    weighted = 0.0
    for row in buckets:
        if row.get("rows", 0) < CALIBRATION_MIN_BUCKET_ROWS or "accuracy" not in row:
            continue
        bucket_low = float(row["bucket"].split("-", 1)[0])
        bucket_high = float(row["bucket"].split("-", 1)[1])
        if low_threshold is not None and bucket_high > low_threshold:
            continue
        if high_threshold is not None and bucket_low < high_threshold:
            continue
        rows = int(row["rows"])
        total += rows
        weighted += rows * float(row["accuracy"])
    if total == 0:
        return np.nan, 0
    return weighted / total, total


def calibration_is_inverted(buckets):
    low_accuracy, low_rows = weighted_bucket_accuracy(buckets, low_threshold=0.60)
    high_accuracy, high_rows = weighted_bucket_accuracy(buckets, high_threshold=0.70)
    if low_rows < CALIBRATION_MIN_BUCKET_ROWS or high_rows < CALIBRATION_MIN_BUCKET_ROWS:
        return False
    return bool(high_accuracy + CALIBRATION_INVERSION_MARGIN < low_accuracy)


def confidence_threshold_directional_report(frame, pred_direction, confidence, majority_accuracy, classification_column=None, realized_return_column=None, target_label_method=None):
    classification_column = classification_column or SELECTED_CLASSIFICATION_TARGET_COLUMN
    realized_return_column = realized_return_column or REALIZED_RETURN_COLUMN
    target_label_method = target_label_method or TARGET_LABEL_METHOD
    actual_delta = frame[realized_return_column].to_numpy(dtype=np.float64)
    if is_regression_target_method(target_label_method):
        actual_direction = direction_from_regression_return(target_values_as_bps(frame, classification_column, target_label_method))
    else:
        actual_direction = frame[classification_column].to_numpy(dtype=np.int64)
    rows = []
    for threshold in CONFIDENCE_THRESHOLDS:
        mask = (confidence >= threshold) & (pred_direction != 0)
        if not mask.any():
            rows.append(
                {
                    "threshold": threshold,
                    "rows_kept": 0,
                    "derived_from_regression_output": bool(is_regression_target_method(target_label_method)),
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
        strategy_return = actual_delta[mask] * np.sign(pred_direction[mask])
        up_mask = mask & (pred_direction > 0)
        down_mask = mask & (pred_direction < 0)
        accuracy = float((pred_direction[mask] == actual_direction[mask]).mean())
        rows_kept = int(mask.sum())
        avg_return = float(strategy_return.mean())
        lift = float(accuracy - majority_accuracy)
        rows.append(
            {
                "threshold": threshold,
                "rows_kept": rows_kept,
                "derived_from_regression_output": bool(is_regression_target_method(target_label_method)),
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


def absolute_predicted_return_threshold_report(frame, pred_delta, realized_return_column=None):
    realized_return_column = realized_return_column or REALIZED_RETURN_COLUMN
    actual_delta = frame[realized_return_column].to_numpy(dtype=np.float64)
    pred_delta = np.asarray(pred_delta, dtype=np.float64)
    pred_direction = direction_from_regression_return(pred_delta)
    rows = []
    for threshold_bps in [1.0, 2.0, 5.0, 8.0]:
        mask = np.abs(pred_delta) >= threshold_bps
        if not mask.any():
            rows.append(
                {
                    "abs_predicted_return_threshold_bps": threshold_bps,
                    "rows": 0,
                    "avg_realized_return_bps": np.nan,
                    "strategy_avg_return_bps": np.nan,
                    "sign_accuracy": np.nan,
                    "predicted_up_rows": 0,
                    "predicted_down_rows": 0,
                    "predicted_up_realized_return_bps": np.nan,
                    "predicted_down_realized_return_bps": np.nan,
                }
            )
            continue
        up_mask = mask & (pred_direction > 0)
        down_mask = mask & (pred_direction < 0)
        rows.append(
            {
                "abs_predicted_return_threshold_bps": threshold_bps,
                "rows": int(mask.sum()),
                "avg_realized_return_bps": float(actual_delta[mask].mean()),
                "strategy_avg_return_bps": float((actual_delta[mask] * np.sign(pred_direction[mask])).mean()),
                "sign_accuracy": float((np.sign(pred_direction[mask]) == np.sign(actual_delta[mask])).mean()),
                "predicted_up_rows": int(up_mask.sum()),
                "predicted_down_rows": int(down_mask.sum()),
                "predicted_up_realized_return_bps": float(actual_delta[up_mask].mean()) if up_mask.any() else np.nan,
                "predicted_down_realized_return_bps": float(actual_delta[down_mask].mean()) if down_mask.any() else np.nan,
            }
        )
    return rows


def best_threshold_summary(threshold_rows, min_rows):
    eligible = [
        row
        for row in threshold_rows
        if int(row.get("rows_kept", 0)) >= min_rows and np.isfinite(row.get("avg_realized_return_bps", np.nan))
    ]
    if not eligible:
        return {
            "threshold": np.nan,
            "rows_kept": 0,
            "directional_accuracy": np.nan,
            "avg_realized_return_bps": np.nan,
            "predicted_up_return_bps": np.nan,
            "predicted_down_return_bps": np.nan,
            "lift_vs_majority": np.nan,
            "threshold_interesting": False,
            "threshold_stable_candidate": False,
        }
    return max(
        eligible,
        key=lambda row: (
            float(row.get("avg_realized_return_bps", float("-inf"))),
            float(row.get("lift_vs_majority", float("-inf"))),
            int(row.get("rows_kept", 0)),
        ),
    )


def attach_usefulness_flags(report, baselines):
    if report.get("target_label_method") == "instability" or TARGET_LABEL_METHOD == "instability":
        report["price_candidate_useful"] = False
        report["direction_candidate_useful"] = False
        report["candidate_useful"] = False
        report["calibration_inverted"] = False
        report["best_threshold_any_rows"] = {}
        report["best_threshold_min_100_rows"] = {}
        report["best_threshold_min_300_rows"] = {}
        report["warnings"] = [
            "Instability target is a paper-only risk/gating label, not a direction model.",
            "No auto-registration/promotion is allowed for instability risk-gate artifacts.",
        ]
        return report
    warnings = []
    calibration_inverted = calibration_is_inverted(report.get("calibration_buckets", []))
    rmse_limit = baselines["zero_return_rmse_bps"] * (1.0 + MAX_RMSE_WORSENING_RATIO)
    price_useful = bool(
        report["mae_bps"] < baselines["zero_return_mae_bps"]
        and report["rmse_bps"] <= rmse_limit
        and not report.get("prediction_distribution_collapsed", False)
    )
    sign_baseline = baselines.get("majority_direction_sign_accuracy_excluding_flat", 0.0)
    if np.isnan(sign_baseline):
        sign_baseline = 0.0
    up_return = report.get("average_realized_bps_when_pred_up", np.nan)
    down_return = report.get("average_realized_bps_when_pred_down", np.nan)
    threshold_rows = report.get("confidence_threshold_directional_report", [])
    best_any = best_threshold_summary(threshold_rows, 1)
    best_min_interesting = best_threshold_summary(threshold_rows, MIN_THRESHOLD_ROWS_INTERESTING)
    best_min_stable = best_threshold_summary(threshold_rows, MIN_THRESHOLD_ROWS_STABLE)
    direction_useful = bool(
        report["directional_accuracy"] > baselines["majority_direction_accuracy"]
        and report["sign_accuracy_excluding_flat"] > sign_baseline
        and np.isfinite(up_return)
        and np.isfinite(down_return)
        and up_return > down_return
        and report["forward_test_avg_return_bps"] > 0
        and not calibration_inverted
    )
    stable_threshold_available = bool(best_min_stable.get("threshold_stable_candidate", False))
    if not direction_useful and stable_threshold_available and not calibration_inverted:
        direction_useful = True
    if direction_useful and not price_useful:
        warnings.append("Model may be directionally useful but not useful for exact price prediction.")
    if calibration_inverted:
        warnings.append("Model is overconfident/inverted.")
    if best_any.get("rows_kept", 0) > 0 and best_any.get("rows_kept", 0) < MIN_THRESHOLD_ROWS_INTERESTING:
        warnings.append("Best confidence-threshold result has too few rows to trust.")
    if is_regression_target_method(report.get("target_label_method", TARGET_LABEL_METHOD)):
        mae_limit = baselines["zero_return_mae_bps"] * (1.0 + REGRESSION_MAE_WORSE_THAN_ZERO_LIMIT)
        mae_worse_than_zero = bool(report["mae_bps"] > mae_limit)
        sign_below_random = bool(
            np.isfinite(report.get("sign_accuracy_excluding_flat", np.nan))
            and report.get("sign_accuracy_excluding_flat", 1.0) < 0.50
        )
        one_sided = bool(report.get("predicted_one_sided", False))
        threshold_rows = report.get("absolute_predicted_return_threshold_report", [])
        high_abs_row = next(
            (
                row
                for row in threshold_rows
                if float(row.get("abs_predicted_return_threshold_bps", -1.0)) >= REGRESSION_UNCALIBRATED_THRESHOLD_BPS
            ),
            None,
        )
        high_abs_keep_pct = float(high_abs_row.get("rows", 0)) / max(int(report.get("rows", 0)), 1) if high_abs_row else 0.0
        magnitudes_uncalibrated = bool(high_abs_keep_pct >= REGRESSION_UNCALIBRATED_KEEP_PCT)
        report["regression_guardrails"] = {
            "mae_worse_than_zero_by_limit": mae_worse_than_zero,
            "mae_limit_bps": float(mae_limit),
            "zero_return_baseline_mae_bps": float(baselines["zero_return_mae_bps"]),
            "sign_accuracy_below_50pct": sign_below_random,
            "predicted_one_sided": one_sided,
            "high_abs_threshold_bps": REGRESSION_UNCALIBRATED_THRESHOLD_BPS,
            "high_abs_threshold_keep_pct": high_abs_keep_pct,
            "magnitudes_uncalibrated": magnitudes_uncalibrated,
        }
        if mae_worse_than_zero:
            warnings.append(
                f"Regression MAE is worse than zero-return baseline by more than {REGRESSION_MAE_WORSE_THAN_ZERO_LIMIT:.0%}."
            )
        if sign_below_random:
            warnings.append("Regression sign accuracy is below 50%; do not mark direction useful.")
        if one_sided:
            warnings.append("Regression predicted direction is one-sided on this split.")
        if magnitudes_uncalibrated:
            warnings.append(
                f"Predicted magnitudes are likely uncalibrated: abs(pred)>={REGRESSION_UNCALIBRATED_THRESHOLD_BPS:.1f}bps keeps {high_abs_keep_pct:.2%} of rows."
            )
        if mae_worse_than_zero or sign_below_random or one_sided or magnitudes_uncalibrated:
            price_useful = False
            direction_useful = False
    report["price_candidate_useful"] = price_useful
    report["direction_candidate_useful"] = direction_useful
    report["candidate_useful"] = bool(price_useful or direction_useful)
    report["calibration_inverted"] = calibration_inverted
    report["best_threshold_any_rows"] = best_any
    report["best_threshold_min_100_rows"] = best_min_interesting
    report["best_threshold_min_300_rows"] = best_min_stable
    report["warnings"] = warnings
    return report


def confidence_from_delta(pred_delta):
    # Regression-only models do not produce class probabilities, so this gives
    # a simple confidence proxy based on how far the predicted move is from flat.
    scale = max(PRICE_TINY_FLAT_BPS * 4.0, 1e-6)
    return np.clip(np.abs(np.asarray(pred_delta, dtype=np.float64)) / scale, 0.0, 1.0)


def report_for_forward_model(model_type, hidden_nodes, frame, pred_delta, pred_direction, confidence, baselines, classification_column=None, realized_return_column=None, return_head_column=None, target_label_method=None):
    classification_column = classification_column or SELECTED_CLASSIFICATION_TARGET_COLUMN
    realized_return_column = realized_return_column or REALIZED_RETURN_COLUMN
    return_head_column = return_head_column or RETURN_HEAD_TARGET_COLUMN
    target_label_method = target_label_method or TARGET_LABEL_METHOD
    report = metrics(frame, pred_delta, pred_direction, confidence, classification_column, realized_return_column, return_head_column, target_label_method)
    report["model_type"] = model_type
    report["hidden_nodes"] = hidden_nodes
    report["mae_lift_vs_zero_return_baseline"] = float(baselines["zero_return_mae_bps"] - report["mae_bps"])
    report["calibration_buckets"] = calibration_buckets(
        confidence,
        pred_direction,
        (
            direction_from_regression_return(target_values_as_bps(frame, classification_column, target_label_method))
            if is_regression_target_method(target_label_method)
            else frame[classification_column].to_numpy(dtype=np.int64)
        ),
    )
    if target_label_method == "instability":
        report["confidence_threshold_directional_report"] = []
    else:
        report["confidence_threshold_directional_report"] = confidence_threshold_directional_report(
            frame,
            pred_direction,
            confidence,
            baselines["majority_direction_accuracy"],
            classification_column,
            realized_return_column,
            target_label_method,
        )
    if is_regression_target_method(target_label_method):
        report["absolute_predicted_return_threshold_report"] = absolute_predicted_return_threshold_report(
            frame,
            pred_delta,
            realized_return_column,
        )
    attach_usefulness_flags(report, baselines)
    if is_regression_target_method(target_label_method):
        warnings = list(report.get("warnings", []))
        if (
            np.isfinite(report.get("predicted_vs_actual_return_correlation", np.nan))
            and report.get("predicted_vs_actual_return_correlation", 0.0) < 0
        ) or (
            np.isfinite(report.get("sign_accuracy_excluding_flat", np.nan))
            and report.get("sign_accuracy_excluding_flat", 1.0) < 0.5
        ):
            warnings.append("Regression sign may be inverted.")
        report["warnings"] = warnings
    return report


def build_forward_test_reports(models, x_test, train, test, delta_mean, delta_std, classification_column=None, realized_return_column=None, return_head_column=None, target_label_method=None):
    classification_column = classification_column or SELECTED_CLASSIFICATION_TARGET_COLUMN
    realized_return_column = realized_return_column or REALIZED_RETURN_COLUMN
    return_head_column = return_head_column or RETURN_HEAD_TARGET_COLUMN
    target_label_method = target_label_method or TARGET_LABEL_METHOD
    baselines = baseline_metrics(train, test, classification_column, realized_return_column, target_label_method)
    actual_count = len(test)
    actual_delta = test[realized_return_column].to_numpy(dtype=np.float64)
    zero_delta = np.zeros(actual_count, dtype=np.float64)
    zero_direction = np.zeros(actual_count, dtype=np.int64)
    previous_delta = (
        test["feature_mid_return_1s"].to_numpy(dtype=np.float64) * 10000.0
        if "feature_mid_return_1s" in test.columns
        else np.zeros(actual_count, dtype=np.float64)
    )
    reports = {
        "zero_return_baseline": report_for_forward_model(
            "zero_return_baseline",
            None,
            test,
            zero_delta,
            zero_direction,
            np.zeros(actual_count, dtype=np.float64),
            baselines,
            classification_column,
            realized_return_column,
            return_head_column,
            target_label_method,
        ),
        "previous_return_baseline": report_for_forward_model(
            "previous_return_baseline",
            None,
            test,
            previous_delta,
            direction_from_delta(previous_delta),
            confidence_from_delta(previous_delta),
            baselines,
            classification_column,
            realized_return_column,
            return_head_column,
            target_label_method,
        ),
    }
    if "ridge_regression" in models:
        ridge_model = models["ridge_regression"]
        ridge_delta = x_test @ np.asarray(ridge_model["target_weights"]) + float(ridge_model["target_bias"])
        reports["ridge_regression"] = report_for_forward_model(
            "ridge_regression",
            None,
            test,
            ridge_delta,
            direction_from_regression_return(ridge_delta),
            confidence_from_delta(ridge_delta),
            baselines,
            classification_column,
            realized_return_column,
            return_head_column,
            target_label_method,
        )
    if "ridge_logistic" in models:
        ridge_model = models["ridge_logistic"]
        ridge_delta = x_test @ np.asarray(ridge_model["delta_weights"]) + float(ridge_model["delta_bias"])
        reports["ridge_regression"] = report_for_forward_model(
            "ridge_regression",
            None,
            test,
            ridge_delta,
            direction_from_delta(ridge_delta),
            confidence_from_delta(ridge_delta),
            baselines,
            classification_column,
            realized_return_column,
            return_head_column,
            target_label_method,
        )
        class_probs = softmax(x_test @ np.asarray(ridge_model["class_weights"]) + np.asarray(ridge_model["class_bias"]))
        logistic_direction = class_to_direction(np.argmax(class_probs, axis=1))
        reports["logistic_regression"] = report_for_forward_model(
            "logistic_regression",
            None,
            test,
            zero_delta.copy(),
            logistic_direction,
            class_probs.max(axis=1),
            baselines,
            classification_column,
            realized_return_column,
            return_head_column,
            target_label_method,
        )
    for model_name, model in models.items():
        if not model_name.startswith("mlp_"):
            continue
        hidden_layers = hidden_layers_from_model_spec(model_name)
        hidden_nodes = hidden_layers[0] if hidden_layers else None
        pred_delta, pred_log, pred_direction, confidence, probs = predict_model(
            model_name,
            model,
            x_test,
            delta_mean,
            delta_std,
        )
        reports[model_report_name(model_name)] = report_for_forward_model(
            model_report_name(model_name),
            hidden_nodes,
            test,
            pred_delta,
            pred_direction,
            confidence,
            baselines,
            classification_column,
            realized_return_column,
            return_head_column,
            target_label_method,
        )
    for report in reports.values():
        report["mae_lift_vs_previous_return_baseline"] = float(
            baselines["previous_return_persists_mae_bps"] - report["mae_bps"]
        )
        report["forward_test_rows"] = int(actual_count)
        report["zero_return_baseline_mae_bps"] = float(baselines["zero_return_mae_bps"])
        report["zero_return_baseline_rmse_bps"] = float(baselines["zero_return_rmse_bps"])
        report["majority_direction_baseline_accuracy"] = float(baselines["majority_direction_accuracy"])
        report["majority_direction_baseline_sign_accuracy_excluding_flat"] = float(
            baselines["majority_direction_sign_accuracy_excluding_flat"]
        )
        report["average_actual_delta_bps"] = float(np.mean(actual_delta)) if actual_count else np.nan
    return reports, baselines


def select_model_name(reports):
    if TARGET_LABEL_METHOD == "instability":
        return max(
            reports,
            key=lambda name: (
                reports[name].get("instability_target_metrics", {}).get("f1", float("-inf")),
                reports[name].get("instability_target_metrics", {}).get("precision", float("-inf")),
                reports[name].get("instability_target_metrics", {}).get("recall", float("-inf")),
                -reports[name].get("mae_bps", float("inf")),
            ),
        )
    objective = SELECTION_OBJECTIVE if SELECTION_OBJECTIVE in {"mae", "direction", "return"} else "mae"
    if objective == "direction":
        return max(
            reports,
            key=lambda name: (
                reports[name].get("directional_accuracy", float("-inf")),
                reports[name].get("sign_accuracy_excluding_flat", float("-inf")),
                -reports[name].get("mae_bps", float("inf")),
            ),
        )
    if objective == "return":
        return max(
            reports,
            key=lambda name: (
                reports[name].get("forward_test_avg_return_bps", float("-inf")),
                reports[name].get("directional_accuracy", float("-inf")),
                -reports[name].get("mae_bps", float("inf")),
            ),
        )
    return min(reports, key=lambda name: reports[name]["mae_bps"])


def model_report_name(model_name):
    if model_name == "ridge_logistic":
        if TARGET_LABEL_METHOD == "instability":
            return "logistic_regression"
        objective = SELECTION_OBJECTIVE if SELECTION_OBJECTIVE in {"mae", "direction", "return"} else "mae"
        return "logistic_regression" if objective in {"direction", "return"} else "ridge_regression"
    if model_name.startswith("mlp_"):
        return f"mlp_hidden_{model_name.split('_', 1)[1]}"
    return model_name


def allowed_training_model_names(models):
    if not PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES:
        return list(models.keys())
    allowed = set(PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES)
    names = []
    for name in models.keys():
        if name in allowed:
            names.append(name)
            continue
        report_name = model_report_name(name)
        if report_name in allowed:
            names.append(name)
    return names


def report_subset_for_models(forward_reports, model_names):
    subset = {}
    for name in model_names:
        report_name = model_report_name(name)
        if report_name in forward_reports:
            subset[name] = forward_reports[report_name]
    return subset


def prediction_frame(frame, pred_delta, pred_log, pred_direction, confidence, model_id, probabilities=None):
    output = frame[["timestamp", "time"]].copy()
    output["predicted_return_bps"] = pred_delta
    output["predicted_next_mid_delta_bps"] = pred_delta
    output["predicted_next_mid_log_return"] = pred_log
    output["predicted_direction"] = pred_direction
    output["confidence"] = confidence
    if probabilities is not None and np.ndim(probabilities) == 2 and probabilities.shape[1] >= 3:
        output["prob_down"] = probabilities[:, 0]
        output["prob_flat"] = probabilities[:, 1]
        output["prob_up"] = probabilities[:, 2]
    else:
        output["prob_down"] = np.nan
        output["prob_flat"] = np.nan
        output["prob_up"] = np.nan
    output["confidence_type"] = "absolute_predicted_return_magnitude" if SELECTED_TARGET_COLUMNS and is_regression_target_method(TARGET_LABEL_METHOD) else "class_probability"
    output["prediction_score_note"] = (
        "Regression confidence is a magnitude score derived from absolute predicted return, not a class probability."
        if is_regression_target_method(TARGET_LABEL_METHOD)
        else "Classification confidence is max class probability."
    )
    output["model_id"] = model_id
    output["feature_set_name"] = FEATURE_SET
    output["horizon_seconds"] = HORIZON_SECONDS
    output["lookback_profile"] = frame["lookback_profile"].iloc[0] if "lookback_profile" in frame.columns and len(frame) else LOOKBACK_PROFILE_ENV
    output["freshness_status"] = "historical_forward_test"
    output["selected_target_columns"] = ",".join(SELECTED_TARGET_COLUMNS)
    output["selected_classification_target_column"] = SELECTED_CLASSIFICATION_TARGET_COLUMN
    output["realized_return_column"] = REALIZED_RETURN_COLUMN
    output["output_semantics"] = TARGET_LABEL_METHOD
    output["actual_next_mid_delta_bps"] = frame[TARGET_DELTA_COLUMN].to_numpy(dtype=np.float64)
    output["actual_next_mid_log_return"] = frame[TARGET_LOG_COLUMN].to_numpy(dtype=np.float64)
    output["actual_selected_target_value"] = pd.to_numeric(frame[SELECTED_CLASSIFICATION_TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    if is_regression_target_method(TARGET_LABEL_METHOD):
        output["actual_direction"] = direction_from_regression_return(
            target_values_as_bps(frame, SELECTED_CLASSIFICATION_TARGET_COLUMN, TARGET_LABEL_METHOD)
        )
    else:
        output["actual_direction"] = frame[SELECTED_CLASSIFICATION_TARGET_COLUMN].to_numpy(dtype=np.int64)
    output["actual_realized_return_bps"] = frame[REALIZED_RETURN_COLUMN].to_numpy(dtype=np.float64)
    max_favorable_column = f"target_max_favorable_excursion_bps_{HORIZON_SECONDS}s"
    max_adverse_column = f"target_max_adverse_excursion_bps_{HORIZON_SECONDS}s"
    if max_favorable_column in frame.columns:
        output["actual_max_favorable_excursion_bps"] = pd.to_numeric(frame[max_favorable_column], errors="coerce").to_numpy(dtype=np.float64)
    if max_adverse_column in frame.columns:
        output["actual_max_adverse_excursion_bps"] = pd.to_numeric(frame[max_adverse_column], errors="coerce").to_numpy(dtype=np.float64)
    return output


def main():
    global HORIZON_SECONDS, TARGET_LOG_COLUMN, TARGET_DELTA_COLUMN, TARGET_DIRECTION_COLUMN
    global SELECTED_TARGET_COLUMNS, SELECTED_CLASSIFICATION_TARGET_COLUMN, REALIZED_RETURN_COLUMN, RETURN_HEAD_TARGET_COLUMN, TARGET_LABEL_METHOD
    training_rows_path = resolve_training_path()
    frame = read_csv(training_rows_path)
    if len(frame) == 0:
        raise FileNotFoundError(f"Missing tiny price training rows: {training_rows_path}. Run npm run tiny-price-build first.")
    if "target_horizon_seconds" in frame.columns and len(frame["target_horizon_seconds"].dropna()):
        inferred_horizon = int(float(frame["target_horizon_seconds"].dropna().iloc[0]))
    elif "horizon_seconds" in frame.columns and len(frame["horizon_seconds"].dropna()):
        inferred_horizon = int(float(frame["horizon_seconds"].dropna().iloc[0]))
    else:
        inferred_horizon = HORIZON_SECONDS
    HORIZON_SECONDS = inferred_horizon
    TARGET_LOG_COLUMN = f"target_next_mid_log_return_{HORIZON_SECONDS}s"
    TARGET_DELTA_COLUMN = f"target_next_mid_delta_bps_{HORIZON_SECONDS}s"
    TARGET_DIRECTION_COLUMN = f"target_next_mid_direction_{HORIZON_SECONDS}s"
    target_spec_name = first_nonempty_string(frame, "target_spec_name", f"direction_{HORIZON_SECONDS}s")
    TARGET_LABEL_METHOD = first_nonempty_string(frame, "target_label_method", "direction")
    SELECTED_TARGET_COLUMNS = selected_target_columns_for_method(TARGET_LABEL_METHOD, HORIZON_SECONDS)
    SELECTED_CLASSIFICATION_TARGET_COLUMN = SELECTED_TARGET_COLUMNS[0]
    REALIZED_RETURN_COLUMN = f"target_return_bps_{HORIZON_SECONDS}s" if f"target_return_bps_{HORIZON_SECONDS}s" in frame.columns else TARGET_DELTA_COLUMN
    RETURN_HEAD_TARGET_COLUMN = SELECTED_CLASSIFICATION_TARGET_COLUMN if is_regression_target_method(TARGET_LABEL_METHOD) else TARGET_DELTA_COLUMN
    if SELECTED_CLASSIFICATION_TARGET_COLUMN not in frame.columns:
        if TARGET_LABEL_METHOD == "direction" and TARGET_DIRECTION_COLUMN in frame.columns:
            SELECTED_TARGET_COLUMNS = [TARGET_DIRECTION_COLUMN]
            SELECTED_CLASSIFICATION_TARGET_COLUMN = TARGET_DIRECTION_COLUMN
        elif TARGET_LABEL_METHOD in {"first_touch", "move_before_adverse", "move_before_adverse_net_aware"} and TARGET_DIRECTION_COLUMN in frame.columns:
            SELECTED_TARGET_COLUMNS = [TARGET_DIRECTION_COLUMN]
            SELECTED_CLASSIFICATION_TARGET_COLUMN = TARGET_DIRECTION_COLUMN
        else:
            raise RuntimeError(
                f"Selected target column is missing: {SELECTED_CLASSIFICATION_TARGET_COLUMN}. "
                f"target_spec={target_spec_name} target_label_method={TARGET_LABEL_METHOD}"
            )
    if SELECTED_CLASSIFICATION_TARGET_COLUMN not in frame.columns:
        raise RuntimeError(
            f"Selected target column is missing: {SELECTED_CLASSIFICATION_TARGET_COLUMN}. "
            f"target_spec={target_spec_name} target_label_method={TARGET_LABEL_METHOD}"
        )
    feature_columns = select_model_feature_columns(frame)
    print("Tiny price canonical feature selection")
    print(f"feature_columns={feature_columns}")
    print(f"feature_count={len(feature_columns)}")
    assert_numeric_feature_columns(frame, feature_columns)
    for fallback, dynamic in [
        ("target_next_mid_delta_bps_1s", TARGET_DELTA_COLUMN),
        ("target_next_mid_log_return_1s", TARGET_LOG_COLUMN),
        ("target_next_mid_direction_1s", TARGET_DIRECTION_COLUMN),
    ]:
        if dynamic not in frame.columns and fallback in frame.columns:
            frame[dynamic] = frame[fallback]
    required = [
        "timestamp",
        TARGET_DELTA_COLUMN,
        TARGET_LOG_COLUMN,
        TARGET_DIRECTION_COLUMN,
        SELECTED_CLASSIFICATION_TARGET_COLUMN,
        REALIZED_RETURN_COLUMN,
        RETURN_HEAD_TARGET_COLUMN,
        *feature_columns,
    ]
    required = list(dict.fromkeys(required))
    frame[required] = frame[required].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)
    if len(frame) < MIN_ROWS:
        raise RuntimeError(f"Not enough rows for tiny price training: {len(frame)} < {MIN_ROWS}")
    lookback_profile = LOOKBACK_PROFILE_ENV
    if "lookback_profile" in frame.columns and len(frame["lookback_profile"].dropna()):
        lookback_profile = str(frame["lookback_profile"].dropna().iloc[0]).strip().lower() or LOOKBACK_PROFILE_ENV
    train, validation, test = split(frame)
    if MAX_TRAIN_ROWS > 0 and len(train) > MAX_TRAIN_ROWS:
        train = train.tail(MAX_TRAIN_ROWS).copy()
    x_train_raw = train[feature_columns].to_numpy(dtype=np.float64)
    x_validation_raw = validation[feature_columns].to_numpy(dtype=np.float64)
    x_test_raw = test[feature_columns].to_numpy(dtype=np.float64)
    x_train, x_validation, x_test, feature_mean, feature_std = standardize(x_train_raw, x_validation_raw, x_test_raw)
    if is_regression_target_method(TARGET_LABEL_METHOD):
        y_delta_train = target_values_as_bps(train, RETURN_HEAD_TARGET_COLUMN, TARGET_LABEL_METHOD)
        y_delta_validation = target_values_as_bps(validation, RETURN_HEAD_TARGET_COLUMN, TARGET_LABEL_METHOD)
    else:
        y_delta_train = train[TARGET_DELTA_COLUMN].to_numpy(dtype=np.float64)
        y_delta_validation = validation[TARGET_DELTA_COLUMN].to_numpy(dtype=np.float64)
    target_distribution_diagnostics = (
        split_distribution_stats(train, validation, test, RETURN_HEAD_TARGET_COLUMN, TARGET_LABEL_METHOD)
        if is_regression_target_method(TARGET_LABEL_METHOD)
        else {}
    )
    unclipped_y_delta_train = y_delta_train.copy()
    unclipped_y_delta_validation = y_delta_validation.copy()
    regression_target_clipping = {
        "enabled": False,
        "clip_bps": np.nan,
        "train": target_clipping_report(unclipped_y_delta_train, y_delta_train, None),
        "validation": target_clipping_report(unclipped_y_delta_validation, y_delta_validation, None),
        "note": "Target clipping is applied only to the fitted regression target. Validation/test metrics are evaluated against unclipped realized returns.",
    }
    if is_regression_target_method(TARGET_LABEL_METHOD) and PRICE_TINY_REGRESSION_TARGET_CLIP_BPS is not None:
        clip_bps = float(abs(PRICE_TINY_REGRESSION_TARGET_CLIP_BPS))
        y_delta_train = np.clip(y_delta_train, -clip_bps, clip_bps)
        y_delta_validation = np.clip(y_delta_validation, -clip_bps, clip_bps)
        regression_target_clipping = {
            "enabled": True,
            "clip_bps": clip_bps,
            "train": target_clipping_report(unclipped_y_delta_train, y_delta_train, clip_bps),
            "validation": target_clipping_report(unclipped_y_delta_validation, y_delta_validation, clip_bps),
            "note": "Target clipping is applied only to the fitted regression target. Validation/test metrics are evaluated against unclipped realized returns.",
        }
    delta_mean = float(y_delta_train.mean())
    delta_std = float(y_delta_train.std() if y_delta_train.std() > 1e-9 else 1.0)
    y_delta_train_scaled = (y_delta_train - delta_mean) / delta_std
    y_delta_validation_scaled = (y_delta_validation - delta_mean) / delta_std
    y_log_train = train[TARGET_LOG_COLUMN].to_numpy(dtype=np.float64)
    if is_regression_target_method(TARGET_LABEL_METHOD):
        class_train = direction_to_class(direction_from_regression_return(target_values_as_bps(train, SELECTED_CLASSIFICATION_TARGET_COLUMN, TARGET_LABEL_METHOD)))
    else:
        class_train = direction_to_class(train[SELECTED_CLASSIFICATION_TARGET_COLUMN].to_numpy(dtype=np.int64))

    model_specs = requested_model_specs()
    validate_model_specs(model_specs)
    if is_regression_target_method(TARGET_LABEL_METHOD):
        incompatible = [spec for spec in model_specs if spec == "ridge_logistic" or spec.startswith("mlp_")]
        if incompatible:
            raise RuntimeError(
                "Regression target specs must use true regression model specs. "
                f"Unsupported for {TARGET_LABEL_METHOD}: {incompatible}. "
                "Use PRICE_TINY_MODEL_SPECS=ridge_regression."
            )
    models = {}
    if "ridge_regression" in model_specs:
        target_weights_scaled, target_bias_scaled = train_ridge(x_train, y_delta_train_scaled)
        models["ridge_regression"] = {
            "model_type": "ridge_regression",
            "target_column": RETURN_HEAD_TARGET_COLUMN,
            "target_output_unit": "bps",
            "target_transform": "log_return_to_bps" if TARGET_LABEL_METHOD == "next_mid_log_return" else "identity_bps",
            "target_weights": (target_weights_scaled * delta_std).tolist(),
            "target_bias": float(target_bias_scaled * delta_std + delta_mean),
            "coefficients": (target_weights_scaled * delta_std).tolist(),
            "intercept": float(target_bias_scaled * delta_std + delta_mean),
            "confidence_type": "absolute_predicted_return_magnitude",
        }
    if "ridge_logistic" in model_specs:
        ridge_delta_weights, ridge_delta_bias_scaled = train_ridge(x_train, y_delta_train_scaled)
        ridge_log_weights, ridge_log_bias = train_ridge(x_train, y_log_train)
        class_weights, class_bias = train_softmax_logistic(x_train, class_train)
        ridge_model = {
            "delta_weights": (ridge_delta_weights * delta_std).tolist(),
            "delta_bias": float(ridge_delta_bias_scaled * delta_std + delta_mean),
            "log_weights": ridge_log_weights.tolist(),
            "log_bias": float(ridge_log_bias),
            "class_weights": class_weights.tolist(),
            "class_bias": class_bias.tolist(),
        }
        models["ridge_logistic"] = ridge_model
    reports = {}
    for name, model in models.items():
        pred_delta, pred_log, pred_direction, confidence, probs = predict_model(name, model, x_validation, delta_mean, delta_std)
        reports[name] = metrics(validation, pred_delta, pred_direction, confidence)
    for spec in model_specs:
        if not spec.startswith("mlp_"):
            continue
        hidden_layers = hidden_layers_from_model_spec(spec)
        if not hidden_layers:
            continue
        name = spec
        model = train_mlp_layers(x_train, y_delta_train_scaled, class_train, x_validation, y_delta_validation_scaled, hidden_layers)
        models[name] = {
            "hidden_layers": model["hidden_layers"],
            "hidden_weights": [value.tolist() for value in model["hidden_weights"]],
            "hidden_biases": [value.tolist() for value in model["hidden_biases"]],
            "w_delta": model["w_delta"].tolist(),
            "b_delta": model["b_delta"].tolist(),
            "w_class": model["w_class"].tolist(),
            "b_class": model["b_class"].tolist(),
        }
        pred_delta, pred_log, pred_direction, confidence, probs = predict_model(name, models[name], x_validation, delta_mean, delta_std)
        reports[name] = metrics(validation, pred_delta, pred_direction, confidence)
    if not models:
        raise RuntimeError(f"No tiny-price model specs were trainable: PRICE_TINY_MODEL_SPECS={model_specs}")
    forward_test_reports, baselines = build_forward_test_reports(
        models,
        x_test,
        train,
        test,
        delta_mean,
        delta_std,
        SELECTED_CLASSIFICATION_TARGET_COLUMN,
        REALIZED_RETURN_COLUMN,
        RETURN_HEAD_TARGET_COLUMN,
        TARGET_LABEL_METHOD,
    )
    raw_candidate_reports = report_subset_for_models(forward_test_reports, list(models.keys()))
    raw_selected_model_name = select_model_name(raw_candidate_reports)
    raw_selected_report_name = model_report_name(raw_selected_model_name)
    allowed_model_names = allowed_training_model_names(models)
    allowed_report_subset = report_subset_for_models(forward_test_reports, allowed_model_names)
    if not allowed_report_subset:
        raise RuntimeError(
            "No allowed tiny-price model can be trained/evaluated. "
            f"PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES={PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES or '(unset/all)'} "
            f"evaluated_models={list(models.keys())}"
        )
    selected_name = select_model_name(allowed_report_subset)
    selected_report_name = model_report_name(selected_name)
    selected_model_allowed = selected_name in allowed_model_names
    if PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES and not selected_model_allowed:
        raise RuntimeError(
            f"Internal selection error: selected disallowed model {selected_name} from allowed list {allowed_model_names}"
        )
    model_id = f"{SYMBOL}_{PRIMARY_VENUE or 'legacy'}_tiny_price_{now_tag()}_{FEATURE_SET}_{selected_name}"
    pred_delta_train_selected, _, _, confidence_train_selected, _ = predict_model(
        selected_name,
        models[selected_name],
        x_train,
        delta_mean,
        delta_std,
    )
    pred_delta_validation_selected, _, _, confidence_validation_selected, _ = predict_model(
        selected_name,
        models[selected_name],
        x_validation,
        delta_mean,
        delta_std,
    )
    pred_delta, pred_log, pred_direction, confidence, probs = predict_model(selected_name, models[selected_name], x_test, delta_mean, delta_std)
    prediction_distribution_diagnostics = (
        {
            "train": distribution_stats(pred_delta_train_selected, 0.0),
            "validation": distribution_stats(pred_delta_validation_selected, 0.0),
            "test": distribution_stats(pred_delta, 0.0),
            "confidence_mean_by_split": {
                "train": float(np.nanmean(confidence_train_selected)) if len(confidence_train_selected) else np.nan,
                "validation": float(np.nanmean(confidence_validation_selected)) if len(confidence_validation_selected) else np.nan,
                "test": float(np.nanmean(confidence)) if len(confidence) else np.nan,
            },
        }
        if is_regression_target_method(TARGET_LABEL_METHOD)
        else {}
    )
    test_metrics = dict(forward_test_reports[selected_report_name])
    test_metrics["selected_model_artifact_name"] = selected_name
    test_metrics["selected_model_report_name"] = selected_report_name
    test_metrics["baselines"] = baselines
    candidate_dir = CANDIDATE_ROOT / now_tag()
    schema_hash = feature_schema_hash(feature_columns)
    feature_groups = []
    if "feature_groups" in frame.columns and len(frame["feature_groups"].dropna()):
        feature_groups = [value.strip() for value in str(frame["feature_groups"].dropna().iloc[0]).split(",") if value.strip()]
    if not feature_groups:
        feature_groups = ["base_tiny_price_v1"]
    crossvenue_metadata = {
        "crossvenue_available_pct": first_numeric_value(frame, "crossvenue_available_pct", 0.0),
        "crossvenue_max_age_ms": first_numeric_value(frame, "crossvenue_max_age_ms", 0.0),
        "crossvenue_median_age_ms": first_numeric_value(frame, "crossvenue_median_age_ms", 0.0),
        "crossvenue_join_policy": first_nonempty_string(frame, "crossvenue_join_policy", ""),
        "crossvenue_missing_policy": first_nonempty_string(frame, "crossvenue_missing_policy", ""),
        "crossvenue_max_join_age_ms": first_numeric_value(frame, "crossvenue_max_join_age_ms", 0.0),
        "crossvenue_strict_required": bool_from_frame(frame, "crossvenue_strict_required", False),
    }
    regime_context_metadata = {
        "btc_context_symbol": first_nonempty_string(frame, "regime_context_btc_symbol", "BTCUSDT"),
        "btc_context_available_pct": first_numeric_value(frame, "regime_context_btc_available_pct", 0.0),
        "btc_context_max_age_ms": first_numeric_value(frame, "regime_context_btc_max_age_ms", 0.0),
        "btc_context_join_policy": first_nonempty_string(frame, "regime_context_btc_join_policy", ""),
    }
    feature_spec = {
        "name": str(frame["feature_spec_name"].dropna().iloc[0]) if "feature_spec_name" in frame.columns and len(frame["feature_spec_name"].dropna()) else "+".join(feature_groups),
        "enabled_feature_groups": feature_groups,
        "feature_columns": feature_columns,
        "feature_schema_hash": schema_hash,
        "missing_feature_policy": str(frame["missing_feature_policy"].dropna().iloc[0]) if "missing_feature_policy" in frame.columns and len(frame["missing_feature_policy"].dropna()) else "fill_zero",
        "crossvenue_metadata": crossvenue_metadata,
        "regime_context_metadata": regime_context_metadata,
    }
    target_columns = select_target_columns(frame)
    output_semantics = output_semantics_payload(
        target_spec_name,
        TARGET_LABEL_METHOD,
        SELECTED_TARGET_COLUMNS,
        REALIZED_RETURN_COLUMN,
        RETURN_HEAD_TARGET_COLUMN,
    )
    target_spec = {
        "name": target_spec_name,
        "horizon_seconds": HORIZON_SECONDS,
        "target_columns": target_columns,
        "available_target_columns": target_columns,
        "selected_target_columns": SELECTED_TARGET_COLUMNS,
        "selected_classification_target_column": SELECTED_CLASSIFICATION_TARGET_COLUMN,
        "realized_return_column": REALIZED_RETURN_COLUMN,
        "return_head_target_column": RETURN_HEAD_TARGET_COLUMN,
        "output_semantics": output_semantics,
        "label_construction_method": TARGET_LABEL_METHOD,
        "no_lookahead_validation": "features use rows with timestamp <= prediction timestamp; train/validation/test are chronological",
    }
    model_spec = {
        "model_type": selected_name,
        "hidden_layers": hidden_layers_from_model_spec(selected_name),
        "regularization": {"ridge_l2": RIDGE_L2},
        "training_objective": SELECTION_OBJECTIVE,
        "calibration_metadata": {
            "confidence_thresholds": CONFIDENCE_THRESHOLDS,
            "calibration_inverted": bool(test_metrics.get("calibration_inverted", False)),
        },
    }
    experiment_id = f"{SYMBOL}_{PRIMARY_VENUE or 'legacy'}_{feature_spec['name']}_{target_spec['name']}_{selected_name}_{schema_hash}"
    selected_validation_metrics = reports.get(selected_name, reports.get(selected_report_name, {}))
    (
        classification_target_metrics,
        realized_return_metrics,
        auxiliary_return_head_metrics,
    ) = split_metric_artifact_fields(test_metrics, selected_validation_metrics)
    selected_model_parameters = models[selected_name]
    artifact = {
        "model_type": selected_name if selected_name == "ridge_regression" else "paper_only_tiny_price_numpy",
        "artifact_type": "paper_only_tiny_price_numpy",
        "experiment_id": experiment_id,
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE or "legacy",
        "model_id": model_id,
        "created_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "trained_until_timestamp": int(train["timestamp"].max()),
        "feature_set_name": FEATURE_SET,
        "horizon_seconds": HORIZON_SECONDS,
        "lookback_profile": lookback_profile,
        "feature_spec": feature_spec,
        "target_spec": target_spec,
        "model_spec": model_spec,
        "training_rows_path": str(training_rows_path),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "feature_schema_hash": schema_hash,
        "crossvenue_available_pct": crossvenue_metadata["crossvenue_available_pct"],
        "crossvenue_max_age_ms": crossvenue_metadata["crossvenue_max_age_ms"],
        "crossvenue_join_policy": crossvenue_metadata["crossvenue_join_policy"],
        "crossvenue_missing_policy": crossvenue_metadata["crossvenue_missing_policy"],
        "crossvenue_metadata": crossvenue_metadata,
        "regime_context_metadata": regime_context_metadata,
        "target_columns": target_columns,
        "target_column": RETURN_HEAD_TARGET_COLUMN if selected_name == "ridge_regression" else SELECTED_CLASSIFICATION_TARGET_COLUMN,
        "available_target_columns": target_columns,
        "selected_target_columns": SELECTED_TARGET_COLUMNS,
        "selected_classification_target_column": SELECTED_CLASSIFICATION_TARGET_COLUMN,
        "realized_return_column": REALIZED_RETURN_COLUMN,
        "return_head_target_column": RETURN_HEAD_TARGET_COLUMN,
        "output_semantics": output_semantics,
        "classification_target_metrics": classification_target_metrics,
        "realized_return_metrics": realized_return_metrics,
        "regression_return_metrics": test_metrics.get("regression_return_metrics", {}),
        "auxiliary_return_head_metrics": auxiliary_return_head_metrics,
        "instability_target_metrics": test_metrics.get("instability_target_metrics", {}),
        "regression_target_distribution_diagnostics": target_distribution_diagnostics,
        "regression_prediction_distribution_diagnostics": prediction_distribution_diagnostics,
        "regression_target_clipping": regression_target_clipping,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "delta_target_mean": delta_mean,
        "delta_target_std": delta_std,
        "class_names": CLASS_NAMES,
        "selection_objective": SELECTION_OBJECTIVE,
        "selected_model_name": selected_name,
        "selected_model_type": selected_name,
        "selected_model_report_name": selected_report_name,
        "selected_model_allowed": selected_model_allowed,
        "regression_target_column": RETURN_HEAD_TARGET_COLUMN if selected_name == "ridge_regression" else "",
        "regression_target_output_unit": selected_model_parameters.get("target_output_unit", "") if isinstance(selected_model_parameters, dict) else "",
        "regression_target_transform": selected_model_parameters.get("target_transform", "") if isinstance(selected_model_parameters, dict) else "",
        "regression_target_clip_bps": regression_target_clipping.get("clip_bps", np.nan),
        "regression_target_clip_enabled": bool(regression_target_clipping.get("enabled", False)),
        "ridge_regression_coefficients": selected_model_parameters.get("coefficients", []) if selected_name == "ridge_regression" else [],
        "ridge_regression_intercept": selected_model_parameters.get("intercept", np.nan) if selected_name == "ridge_regression" else np.nan,
        "confidence_type": selected_model_parameters.get("confidence_type", "class_probability") if isinstance(selected_model_parameters, dict) else "class_probability",
        "allowed_training_model_types": PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES,
        "allowed_shadow_model_types": PRICE_TINY_ALLOWED_SHADOW_MODEL_TYPES,
        "best_raw_candidate_by_objective": raw_selected_report_name,
        "best_allowed_candidate_by_objective": selected_report_name,
        "models": models,
        "validation_reports": reports,
        "forward_test_metrics": test_metrics,
        "validation_metrics": selected_validation_metrics,
        "forward_test_threshold_metrics": test_metrics.get("confidence_threshold_directional_report", []),
        "forward_test_reports": forward_test_reports,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "promoted": False,
        "paper_only": True,
    }
    eligible, eligibility_reason, eligibility_threshold_row = registration_gate_status(artifact)
    artifact["registration_eligibility"] = {
        "eligible": bool(eligible),
        "reason": eligibility_reason,
        "threshold_report": eligibility_threshold_row,
        "allowed_shadow_model_types": PRICE_TINY_ALLOWED_SHADOW_MODEL_TYPES,
    }
    atomic_write_json(artifact, candidate_dir / "model.json")
    shadow_registry_note = register_shadow_challenger(candidate_dir / "model.json", artifact)
    preds = prediction_frame(test, pred_delta, pred_log, pred_direction, confidence, model_id, probs)
    prediction_stable_archive_path, prediction_timestamped_archive_path = prediction_archive_paths(
        target_spec_name,
        selected_name,
        schema_hash,
        candidate_dir.name,
    )
    preds["symbol"] = SYMBOL
    preds["primary_venue"] = PRIMARY_VENUE or "legacy"
    preds["target_spec"] = target_spec_name
    preds["target_label_method"] = TARGET_LABEL_METHOD
    preds["selected_model_name"] = selected_name
    preds["selected_model_type"] = selected_name
    preds["model_type"] = artifact["model_type"]
    preds["feature_schema_hash"] = schema_hash
    preds["feature_spec_name"] = feature_spec["name"]
    preds["enabled_feature_groups"] = ",".join(feature_spec["enabled_feature_groups"])
    preds["model_path"] = str(candidate_dir / "model.json")
    preds["candidate_tag"] = candidate_dir.name
    preds["prediction_archive_stable_path"] = str(prediction_stable_archive_path)
    preds["prediction_archive_timestamped_path"] = str(prediction_timestamped_archive_path)
    atomic_write_csv(preds, FORWARD_TEST_PREDICTIONS_PATH)
    atomic_write_csv(preds, prediction_stable_archive_path)
    atomic_write_csv(preds, prediction_timestamped_archive_path)
    print("Tiny price model trained")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Training rows path: {training_rows_path}")
    print(f"Rows: train={len(train)}, validation={len(validation)}, test={len(test)}")
    print(f"Feature set: {FEATURE_SET}")
    print(f"Lookback profile: {lookback_profile}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"feature_spec={feature_spec['name']}")
    print(f"feature_groups={','.join(feature_spec['enabled_feature_groups'])}")
    print(f"feature_schema_hash={schema_hash}")
    if "cross_venue_features" in feature_spec["enabled_feature_groups"]:
        print(f"crossvenue_available_pct={crossvenue_metadata['crossvenue_available_pct']:.4%}")
        print(f"crossvenue_max_age_ms={crossvenue_metadata['crossvenue_max_age_ms']:.2f}")
        print(f"crossvenue_join_policy={crossvenue_metadata['crossvenue_join_policy']}")
        print(f"crossvenue_missing_policy={crossvenue_metadata['crossvenue_missing_policy']}")
    if "regime_context_features" in feature_spec["enabled_feature_groups"]:
        print(f"regime_context_btc_symbol={regime_context_metadata['btc_context_symbol']}")
        print(f"regime_context_btc_available_pct={regime_context_metadata['btc_context_available_pct']:.4%}")
        print(f"regime_context_btc_max_age_ms={regime_context_metadata['btc_context_max_age_ms']:.2f}")
        print(f"regime_context_btc_join_policy={regime_context_metadata['btc_context_join_policy']}")
    print(f"target_spec={target_spec['name']}")
    print(f"target_label_method={target_spec['label_construction_method']}")
    print(f"available_target_columns={','.join(target_columns)}")
    print(f"selected_target_columns={','.join(SELECTED_TARGET_COLUMNS)}")
    print(f"selected_classification_target_column={SELECTED_CLASSIFICATION_TARGET_COLUMN}")
    print(f"realized_return_column={REALIZED_RETURN_COLUMN}")
    print(f"return_head_target_column={RETURN_HEAD_TARGET_COLUMN}")
    print(f"output_semantics_metrics_key={output_semantics['regression_metrics_key']}")
    if is_regression_target_method(TARGET_LABEL_METHOD):
        print_distribution_stats("Regression target distribution diagnostics (unclipped realized target, bps)", target_distribution_diagnostics)
        print("Regression target clipping")
        print(f"- enabled={regression_target_clipping.get('enabled', False)}")
        print(f"- clip_bps={regression_target_clipping.get('clip_bps', np.nan)}")
        for split_name in ["train", "validation"]:
            clip_stats = regression_target_clipping.get(split_name, {})
            print(
                f"- {split_name}: clipped_count={clip_stats.get('clipped_count', 0)} "
                f"clipped_pct={clip_stats.get('clipped_pct', 0.0):.4%}"
            )
        print_distribution_stats(
            "Regression prediction distribution diagnostics (selected model predicted_return_bps)",
            {key: prediction_distribution_diagnostics[key] for key in ["train", "validation", "test"]},
        )
        confidence_means = prediction_distribution_diagnostics.get("confidence_mean_by_split", {})
        print(
            "- regression magnitude score mean by split: "
            f"train={confidence_means.get('train', np.nan):.4f} "
            f"validation={confidence_means.get('validation', np.nan):.4f} "
            f"test={confidence_means.get('test', np.nan):.4f}"
        )
    if TARGET_LABEL_METHOD == "instability":
        class_distribution = test[SELECTED_CLASSIFICATION_TARGET_COLUMN].value_counts().sort_index().to_dict()
        print("Instability target diagnostics")
        print(f"- class_distribution_forward_test={class_distribution}")
        metrics_for_instability = test_metrics.get("instability_target_metrics", {})
        print(f"- percent_unstable={metrics_for_instability.get('percent_unstable', np.nan):.2%}")
        print(f"- precision={metrics_for_instability.get('precision', np.nan):.2%}")
        print(f"- recall={metrics_for_instability.get('recall', np.nan):.2%}")
        print(f"- f1={metrics_for_instability.get('f1', np.nan):.2%}")
        print(
            "- avg_realized_return_bps stable/unstable="
            f"{metrics_for_instability.get('average_realized_return_bps_stable', np.nan):.4f}/"
            f"{metrics_for_instability.get('average_realized_return_bps_unstable', np.nan):.4f}"
        )
        print(
            "- avg_future_range_bps stable/unstable="
            f"{metrics_for_instability.get('average_future_range_bps_stable', np.nan):.4f}/"
            f"{metrics_for_instability.get('average_future_range_bps_unstable', np.nan):.4f}"
        )
        print("- Instability is a paper-only risk/gating target, not a direction model.")
    print(f"model_specs={','.join(model_specs)}")
    print(f"Selection objective: {SELECTION_OBJECTIVE}")
    print(f"allowed_training_model_types={','.join(PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES) if PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES else '(all)'}")
    print(f"allowed_shadow_model_types={','.join(PRICE_TINY_ALLOWED_SHADOW_MODEL_TYPES)}")
    print(f"best_raw_candidate_by_objective={raw_selected_report_name}")
    print(f"best_allowed_candidate_by_objective={selected_report_name}")
    print(f"selected_model={selected_name}")
    print(f"selected_model_name={selected_name}")
    print(f"selected_model_type={selected_name}")
    print(f"selected_model_allowed={selected_model_allowed}")
    print(f"registration_eligible={artifact['registration_eligibility']['eligible']}")
    print(f"registration_eligibility_reason={artifact['registration_eligibility']['reason']}")
    print(f"Candidate model: {candidate_dir / 'model.json'}")
    print(f"candidate_model_path={candidate_dir / 'model.json'}")
    if shadow_registry_note:
        print(f"registered_challenger={shadow_registry_note.get('registered_challenger', False)}")
        print(f"registration_block_reason={shadow_registry_note.get('registration_block_reason', '')}")
        print(f"allowed_shadow_model_types={','.join(PRICE_TINY_ALLOWED_SHADOW_MODEL_TYPES)}")
        print(f"retired_disallowed_challengers={shadow_registry_note.get('retired_disallowed_challengers', 0)}")
        print(f"Shadow pool registry: {shadow_registry_note.get('registry_path', CANDIDATE_REGISTRY_PATH)}")
    print(f"Forward-test predictions: {FORWARD_TEST_PREDICTIONS_PATH}")
    print(f"Forward-test prediction archive latest: {prediction_stable_archive_path}")
    print(f"Forward-test prediction archive timestamped: {prediction_timestamped_archive_path}")
    print("Validation reports")
    for name, report in reports.items():
        print(f"- {name}: mae={report['mae_bps']:.4f}bps rmse={report['rmse_bps']:.4f}bps sign_acc={report['sign_accuracy_excluding_flat']:.2%}")
    print("Forward test")
    metrics_key = "regression_return_metrics" if is_regression_target_method(TARGET_LABEL_METHOD) else "auxiliary_return_head_metrics"
    print(f"- {metrics_key} MAE: {test_metrics['mae_bps']:.4f} bps")
    print(f"- {metrics_key} RMSE: {test_metrics['rmse_bps']:.4f} bps")
    if is_regression_target_method(TARGET_LABEL_METHOD):
        print(f"- regression correlation: {test_metrics.get('predicted_vs_actual_return_correlation', np.nan):.4f}")
        print("- predicted direction is derived from sign(predicted_return_bps).")
        print("- regression confidence is a magnitude score, not a class probability.")
    elif TARGET_LABEL_METHOD != "return_bps":
        print("- selected classification target is not a return target; MAE/RMSE are auxiliary return-head diagnostics.")
    print(f"- sign accuracy excluding flat: {test_metrics['sign_accuracy_excluding_flat']:.2%}")
    if np.isfinite(test_metrics.get("directional_win_rate", np.nan)):
        print(f"- directional win rate: {test_metrics['directional_win_rate']:.2%}")
    else:
        print("- directional win rate: n/a for this target")
    print(f"- zero return baseline MAE: {test_metrics['baselines']['zero_return_mae_bps']:.4f} bps")
    print(f"- price_candidate_useful: {test_metrics['price_candidate_useful']}")
    print(f"- direction_candidate_useful: {test_metrics['direction_candidate_useful']}")
    for warning in test_metrics.get("warnings", []):
        print(f"WARNING: {warning}")
    if TARGET_LABEL_METHOD == "instability":
        print("Confidence-threshold directional report: skipped for instability risk/gating target.")
        print("Best confidence thresholds: skipped for instability risk/gating target.")
    else:
        print("Confidence-threshold directional report")
        for row in test_metrics.get("confidence_threshold_directional_report", []):
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
        if is_regression_target_method(TARGET_LABEL_METHOD):
            print("Absolute predicted-return threshold report")
            for row in test_metrics.get("absolute_predicted_return_threshold_report", []):
                print(
                    f"- abs_pred_return>={row['abs_predicted_return_threshold_bps']:.1f}bps: "
                    f"rows={row['rows']} sign_acc={row['sign_accuracy']:.2%} "
                    f"strategy_avg_return={row['strategy_avg_return_bps']:.4f}bps "
                    f"avg_realized={row['avg_realized_return_bps']:.4f}bps "
                    f"up_rows={row['predicted_up_rows']} up_return={row['predicted_up_realized_return_bps']:.4f}bps "
                    f"down_rows={row['predicted_down_rows']} down_return={row['predicted_down_realized_return_bps']:.4f}bps"
                )
        print("Best confidence thresholds")
        for label in ["best_threshold_any_rows", "best_threshold_min_100_rows", "best_threshold_min_300_rows"]:
            row = test_metrics.get(label, {})
            if not row or int(row.get("rows_kept", 0)) <= 0 or not np.isfinite(row.get("threshold", np.nan)):
                print(f"- {label}: none")
                continue
            print(
                f"- {label}: threshold>={row['threshold']:.2f} rows={row['rows_kept']} "
                f"avg_return={row['avg_realized_return_bps']:.4f}bps "
                f"dir_acc={row['directional_accuracy']:.2%} "
                f"lift_vs_majority={row['lift_vs_majority']:.2%} "
                f"interesting={row['threshold_interesting']} "
                f"stable={row['threshold_stable_candidate']}"
            )
    print("No promotion. Paper-only. No trades/orders/private API.")


if __name__ == "__main__":
    main()
