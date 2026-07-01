import hashlib
import json
import re

import pandas as pd


FEATURE_METADATA_COLUMNS = {
    "experiment_id",
    "feature_set",
    "feature_group",
    "feature_groups",
    "feature_spec",
    "feature_spec_name",
    "feature_schema_hash",
    "target_spec",
    "target_spec_name",
    "target_label_method",
    "target_columns",
    "available_target_columns",
    "selected_target_columns",
    "output_semantics",
    "training_rows_path",
    "training_rows_latest_path",
    "horizon_seconds",
    "target_horizon_seconds",
    "model_spec",
    "model_spec_name",
    "selected_model_type",
    "selected_model_allowed",
    "registration_eligibility",
    "paper_only",
    "feature_set_name",
    "model_feature_columns",
    "simulation_run_id",
    "source_scenario",
    "source_seed",
    "hidden_scenario",
}

TARGET_METADATA_COLUMNS = {
    "target_columns",
    "available_target_columns",
    "selected_target_columns",
    "output_semantics",
    "target_spec",
    "target_spec_name",
    "target_label_method",
    "target_horizon_seconds",
}


def feature_schema_hash(columns):
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()[:16]


def slugify(value, default="experiment"):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or default


def parse_feature_columns_metadata(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    return [item.strip() for item in text.split(",") if item.strip()]


def metadata_feature_columns_from_frame(frame):
    if "model_feature_columns" not in frame.columns or len(frame) == 0:
        return []
    values = frame["model_feature_columns"].dropna()
    if len(values) == 0:
        return []
    return parse_feature_columns_metadata(values.iloc[0])


def legacy_candidate_feature_columns(frame):
    return sorted(
        column
        for column in frame.columns
        if column.startswith("feature_") and column not in FEATURE_METADATA_COLUMNS
    )


def select_model_feature_columns(frame):
    """Return the canonical ordered numeric input feature columns.

    New experiment-spec rows should carry `model_feature_columns`, which is the
    exact list written by the builder. Older rows fall back to strict legacy
    `feature_*` selection after removing known metadata columns.
    """
    metadata_columns = metadata_feature_columns_from_frame(frame)
    if metadata_columns:
        missing = [column for column in metadata_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"model_feature_columns references missing columns: {missing}")
        forbidden = [
            column
            for column in metadata_columns
            if column in FEATURE_METADATA_COLUMNS or column.startswith("hidden_")
        ]
        if forbidden:
            raise ValueError(f"model_feature_columns includes metadata columns: {forbidden}")
        return metadata_columns
    return legacy_candidate_feature_columns(frame)


def nonnumeric_feature_columns(frame, feature_columns, sample_rows=3):
    problems = []
    for column in feature_columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        original_non_null = frame[column].notna()
        bad_mask = original_non_null & converted.isna()
        if bad_mask.any():
            samples = frame.loc[bad_mask, column].head(sample_rows).astype(str).tolist()
            problems.append({"column": column, "sample_values": samples})
    return problems


def assert_numeric_feature_columns(frame, feature_columns):
    problems = nonnumeric_feature_columns(frame, feature_columns)
    if problems:
        details = "; ".join(
            f"{problem['column']} samples={problem['sample_values']}"
            for problem in problems
        )
        raise ValueError(f"Selected feature columns contain nonnumeric values: {details}")


def select_target_columns(frame):
    return sorted(
        column
        for column in frame.columns
        if column.startswith("target_") and column not in TARGET_METADATA_COLUMNS
    )
