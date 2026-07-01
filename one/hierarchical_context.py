import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


EPSILON = 1e-12
RAW_PRICE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "best_bid",
    "best_ask",
    "mid_price",
    "entry_price",
}

FLOW1S_CONTEXT_VALUE_COLUMNS = [
    "feature_context_flow_1s_prob_sell_dominant",
    "feature_context_flow_1s_prob_neutral",
    "feature_context_flow_1s_prob_buy_dominant",
    "feature_context_flow_1s_pred_market_buy_volume_1s",
    "feature_context_flow_1s_pred_market_sell_volume_1s",
    "feature_context_flow_1s_pred_market_pressure_1s",
    "feature_context_flow_1s_pred_pressure_magnitude_1s",
    "feature_context_flow_1s_pred_trade_count_1s",
    "feature_context_flow_1s_buy_burst_prob_1s",
    "feature_context_flow_1s_sell_burst_prob_1s",
]


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_timestamp_series(values):
    timestamps = pd.to_numeric(values, errors="coerce")
    return np.where(timestamps < 10_000_000_000, timestamps * 1000, timestamps)


def coerce_numeric_except(frame, skip_columns):
    frame = frame.copy()
    for column in frame.columns:
        if column not in skip_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def context_max_age_ms(layer):
    defaults = {
        "htf": 3 * 60 * 60 * 1000,
        "regime15": 2 * 60 * 60 * 1000,
        "regime30": 3 * 60 * 60 * 1000,
        "flow3m": 10 * 60 * 1000,
        "flow1s": 2 * 60 * 1000,
        "micro10s": 2 * 60 * 1000,
    }
    env_name = f"MAX_{layer.upper()}_CONTEXT_AGE_MS"
    alternate_env_names = {
        "flow1s": "MAX_FLOW_1S_CONTEXT_AGE_MS",
    }
    alternate_env_name = alternate_env_names.get(layer)
    if alternate_env_name and os.getenv(alternate_env_name) is not None:
        return int(os.getenv(alternate_env_name))
    return int(os.getenv(env_name, str(defaults[layer])))


def numeric_context_columns(frame, timestamp_column, include_prefix=None):
    columns = []
    skip = {
        timestamp_column,
        "timestamp",
        "close_timestamp",
        "time",
        "symbol",
        "regime",
        "model_id",
        "feature_schema_hash",
        "source_quality",
        "venues_present",
    }
    for column in frame.columns:
        if column in skip:
            continue
        if column in RAW_PRICE_COLUMNS:
            continue
        if column.startswith("actual_") or column.startswith("target_"):
            continue
        if include_prefix and not column.startswith(include_prefix):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def read_context_file(path, timestamp_candidates=("close_timestamp", "timestamp")):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), None
    try:
        frame = pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame(), None
    timestamp_column = next(
        (column for column in timestamp_candidates if column in frame.columns),
        None,
    )
    if timestamp_column is None:
        return pd.DataFrame(), None
    frame[timestamp_column] = normalize_timestamp_series(frame[timestamp_column])
    frame = frame.dropna(subset=[timestamp_column]).copy()
    frame[timestamp_column] = frame[timestamp_column].astype(np.int64)
    frame = coerce_numeric_except(
        frame,
        {"time", "symbol", "regime", "model_id", "feature_schema_hash", "source_quality"},
    )
    frame = frame.sort_values(timestamp_column).drop_duplicates(timestamp_column, keep="last")
    return frame.reset_index(drop=True), timestamp_column


def prepare_htf_context(project_root, symbol):
    path = Path(project_root) / "data" / f"{symbol}_htf_context_features.csv"
    frame, timestamp_column = read_context_file(path)
    if len(frame) == 0:
        return None, path
    columns = numeric_context_columns(frame, timestamp_column)
    prepared = frame[[timestamp_column, *columns]].copy()
    return (prepared, timestamp_column, "htf", context_max_age_ms("htf")), path


def prepare_regime_context(project_root, symbol, timeframe):
    layer = f"regime{timeframe}"
    path = Path(project_root) / "data" / f"{symbol}_{timeframe}m_regime_features.csv"
    frame, timestamp_column = read_context_file(path)
    if len(frame) == 0:
        return None, path

    prepared = pd.DataFrame({timestamp_column: frame[timestamp_column]})
    for column in numeric_context_columns(frame, timestamp_column):
        prepared[f"regime_{timeframe}m_{column}"] = frame[column]

    if "regime" in frame.columns:
        regime_names = frame["regime"].fillna("unknown").astype(str).str.lower()
        for name in ["bullish", "bearish", "chop", "high_volatility_chop"]:
            prepared[f"regime_{timeframe}m_is_{name}"] = (regime_names == name).astype(float)

    return (
        prepared,
        timestamp_column,
        f"regime_{timeframe}m",
        context_max_age_ms(layer),
    ), path


def prepare_flow3m_context(project_root, symbol):
    path = Path(project_root) / "data" / "live_predictions" / f"{symbol}_live_3m_predictions.csv"
    frame, timestamp_column = read_context_file(path, timestamp_candidates=("timestamp",))
    if len(frame) == 0:
        return None, path
    prepared = pd.DataFrame({timestamp_column: frame[timestamp_column]})
    wanted = [
        "prob_short",
        "prob_neutral",
        "prob_long",
        "pred_future_return_3",
        "pred_future_range_percent_3",
        "pred_future_market_pressure_3m",
        "pred_future_order_book_imbalance_10bps_3m",
        "pred_future_spread_percent_3m",
        "pred_future_breakout_pressure_3m",
        "pred_future_absorption_3m",
        "validation_score",
        "training_row_count",
    ]
    for column in wanted:
        if column in frame.columns:
            prepared[f"flow_3m_{column}"] = pd.to_numeric(frame[column], errors="coerce")
    return (prepared, timestamp_column, "flow_3m", context_max_age_ms("flow3m")), path


def prepare_micro10s_context(project_root, symbol):
    primary_venue = os.getenv("PRIMARY_VENUE", "").strip().lower()
    realtime_root = Path(project_root) / "data" / "realtime"
    path = (
        realtime_root / primary_venue / f"{symbol}_10s_microstructure_predictions.csv"
        if primary_venue
        else realtime_root / f"{symbol}_10s_microstructure_predictions.csv"
    )
    frame, timestamp_column = read_context_file(path, timestamp_candidates=("timestamp",))
    if len(frame) == 0:
        return None, path
    prepared = pd.DataFrame({timestamp_column: frame[timestamp_column]})
    for column in numeric_context_columns(frame, timestamp_column):
        if column.startswith("prob_") or column.startswith("pred_"):
            prepared[f"micro_10s_{column}"] = frame[column]
    return (prepared, timestamp_column, "micro_10s", context_max_age_ms("micro10s")), path


def prepare_flow1s_context(project_root, symbol):
    primary_venue = os.getenv("PRIMARY_VENUE", "").strip().lower()
    realtime_root = Path(project_root) / "data" / "realtime"
    path = (
        realtime_root / primary_venue / f"{symbol}_1s_order_flow_predictions.csv"
        if primary_venue
        else realtime_root / f"{symbol}_1s_order_flow_predictions.csv"
    )
    frame, timestamp_column = read_context_file(path, timestamp_candidates=("timestamp",))
    if len(frame) == 0:
        return None, path

    prepared = pd.DataFrame({timestamp_column: frame[timestamp_column]})
    wanted = {
        FLOW1S_CONTEXT_VALUE_COLUMNS[0]: [
            "prob_sell_dominant_1s",
            "prob_sell_dominant",
        ],
        FLOW1S_CONTEXT_VALUE_COLUMNS[1]: [
            "prob_neutral_1s",
            "prob_neutral",
        ],
        FLOW1S_CONTEXT_VALUE_COLUMNS[2]: [
            "prob_buy_dominant_1s",
            "prob_buy_dominant",
        ],
        FLOW1S_CONTEXT_VALUE_COLUMNS[3]: [
            "pred_market_buy_volume_1s",
            "pred_future_market_buy_volume_1s",
        ],
        FLOW1S_CONTEXT_VALUE_COLUMNS[4]: [
            "pred_market_sell_volume_1s",
            "pred_future_market_sell_volume_1s",
        ],
        FLOW1S_CONTEXT_VALUE_COLUMNS[5]: [
            "pred_market_pressure_1s",
            "pred_future_market_pressure_1s",
        ],
        FLOW1S_CONTEXT_VALUE_COLUMNS[6]: [
            "pred_pressure_magnitude_1s",
        ],
        FLOW1S_CONTEXT_VALUE_COLUMNS[7]: [
            "pred_trade_count_1s",
            "pred_future_trade_count_1s",
        ],
        FLOW1S_CONTEXT_VALUE_COLUMNS[8]: [
            "buy_burst_prob_1s",
            "prob_aggressive_buy_burst_1s",
        ],
        FLOW1S_CONTEXT_VALUE_COLUMNS[9]: [
            "sell_burst_prob_1s",
            "prob_aggressive_sell_burst_1s",
        ],
    }
    for output_column, source_candidates in wanted.items():
        for source_column in source_candidates:
            if source_column in frame.columns:
                prepared[output_column] = pd.to_numeric(frame[source_column], errors="coerce")
                break
        if output_column not in prepared.columns:
            prepared[output_column] = 0.0

    return (
        prepared,
        timestamp_column,
        "feature_context_flow_1s",
        context_max_age_ms("flow1s"),
    ), path


def layer_preparer(layer, project_root, symbol):
    if layer == "htf":
        return prepare_htf_context(project_root, symbol)
    if layer == "regime15":
        return prepare_regime_context(project_root, symbol, "15")
    if layer == "regime30":
        return prepare_regime_context(project_root, symbol, "30")
    if layer == "flow3m":
        return prepare_flow3m_context(project_root, symbol)
    if layer == "flow1s":
        return prepare_flow1s_context(project_root, symbol)
    if layer == "micro10s":
        return prepare_micro10s_context(project_root, symbol)
    raise ValueError(f"Unknown hierarchical context layer: {layer}")


def layer_context_prefix(layer):
    return {
        "htf": "htf",
        "regime15": "regime_15m",
        "regime30": "regime_30m",
        "flow3m": "flow_3m",
        "flow1s": "feature_context_flow_1s",
        "micro10s": "micro_10s",
    }.get(layer, layer)


def attach_single_context(rows, context_tuple):
    context, timestamp_column, prefix, max_age_ms = context_tuple
    rows = rows.sort_values("timestamp").copy()
    context = context.sort_values(timestamp_column).copy()
    context_timestamp_column = f"{prefix}_context_timestamp"
    context = context.rename(columns={timestamp_column: context_timestamp_column})
    value_columns = [
        column
        for column in context.columns
        if column != context_timestamp_column
    ]

    merged = pd.merge_asof(
        rows,
        context,
        left_on="timestamp",
        right_on=context_timestamp_column,
        direction="backward",
    )
    age_column = f"{prefix}_context_age_ms"
    available_column = f"{prefix}_context_available"
    merged[age_column] = merged["timestamp"] - merged[context_timestamp_column]
    available = (
        merged[context_timestamp_column].notna()
        & (merged[age_column] >= 0)
        & (merged[age_column] <= max_age_ms)
    )
    merged[available_column] = available.astype(float)
    for column in value_columns:
        merged.loc[~available, column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    # Optional context is allowed to be missing or stale. A blank age would
    # later become NaN and could accidentally make every training row look
    # invalid, so use -1.0 as the explicit "no usable context" sentinel.
    merged.loc[~available, age_column] = -1.0
    merged[age_column] = pd.to_numeric(merged[age_column], errors="coerce").fillna(-1.0)
    return merged.drop(columns=[context_timestamp_column], errors="ignore")


def fill_optional_context_defaults(frame):
    frame = frame.copy()
    for column in frame.columns:
        if "_context_available" in column:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        elif "_context_age_ms" in column:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(-1.0)
        elif column.startswith("feature_context_") or "_context_" in column:
            if column not in {"time", "symbol", "regime"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def prefix_context_as_model_features(frame, original_columns):
    rename_map = {}
    for column in frame.columns:
        if column in original_columns:
            continue
        if column in {"timestamp", "time", "feature_ready"}:
            continue
        if column.startswith("feature_context_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            rename_map[column] = f"feature_context_{column}"
    return frame.rename(columns=rename_map)


def attach_hierarchical_context(
    rows,
    project_root,
    symbol,
    layers=("htf", "regime15", "regime30"),
    as_model_features=False,
):
    if rows is None or len(rows) == 0:
        return rows, {}
    output = rows.copy()
    output["timestamp"] = normalize_timestamp_series(output["timestamp"]).astype(np.int64)
    original_columns = set(output.columns)
    diagnostics = {}

    for layer in layers:
        context_tuple, path = layer_preparer(layer, project_root, symbol)
        diagnostics[layer] = {
            "path": str(path),
            "rows": 0 if context_tuple is None else int(len(context_tuple[0])),
            "attached": False,
        }
        if context_tuple is None or len(context_tuple[0]) == 0:
            available_column = f"{layer_context_prefix(layer)}_context_available"
            output[available_column] = 0.0
            age_column = f"{layer_context_prefix(layer)}_context_age_ms"
            output[age_column] = -1.0
            if layer == "flow1s":
                for column in FLOW1S_CONTEXT_VALUE_COLUMNS:
                    output[column] = 0.0
            continue
        output = attach_single_context(output, context_tuple)
        diagnostics[layer]["attached"] = True

    output = fill_optional_context_defaults(output)

    if as_model_features:
        output = prefix_context_as_model_features(output, original_columns)
        output = fill_optional_context_defaults(output)

    return output, diagnostics


def print_context_diagnostics(diagnostics):
    print("Hierarchical context diagnostics:")
    if not diagnostics:
        print("- no context layers requested")
        return
    for layer, info in diagnostics.items():
        status = "attached" if info.get("attached") else "not attached"
        print(f"- {layer}: {status}, rows={info.get('rows', 0)}, path={info.get('path')}")


def context_age_column_for(available_column):
    if available_column.endswith("_context_available"):
        return available_column.replace("_context_available", "_context_age_ms")
    return None


def print_context_availability_summary(frame, title="Context availability summary"):
    print(f"\n{title}")
    if frame is None or len(frame) == 0:
        print("- no rows available")
        return

    availability_columns = [
        column
        for column in frame.columns
        if column.endswith("_context_available")
        or column.endswith("micro_10s_context_available")
    ]
    if not availability_columns:
        print("- no context availability columns found")
        return

    for column in sorted(availability_columns):
        availability = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        available_mask = availability > 0.0
        available_count = int(available_mask.sum())
        unavailable_count = int(len(frame) - available_count)
        mean_availability = float(availability.mean()) if len(frame) else 0.0
        age_column = context_age_column_for(column)

        print(f"- {column}")
        print(f"  mean availability: {mean_availability:.2%}")
        print(f"  available rows: {available_count}")
        print(f"  unavailable rows: {unavailable_count}")

        if age_column and age_column in frame.columns and available_count:
            ages = pd.to_numeric(frame.loc[available_mask, age_column], errors="coerce")
            ages = ages.replace([np.inf, -np.inf], np.nan).dropna()
            if len(ages):
                print(f"  median context age ms: {float(ages.median()):.0f}")
                print(f"  p90 context age ms: {float(ages.quantile(0.90)):.0f}")
                print(f"  max context age ms: {float(ages.max()):.0f}")
            else:
                print("  context age stats: unavailable")
        else:
            print("  context age stats: unavailable")
