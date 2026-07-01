import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
VENUE_TAG = PRIMARY_VENUE or "legacy"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
INPUT_PATH = VENUE_DIR / f"{SYMBOL}_hierarchy_forecast_log.csv"
OUTPUT_PATH = VENUE_DIR / f"{SYMBOL}_hierarchy_mediator_training_rows.csv"

RETURN_COLUMNS = [
    "realized_return_10s",
    "realized_return_30s",
    "realized_return_60s",
]


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def numeric(frame, column, default=0.0):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def text(frame, column):
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str)


def flag_from_text(frame, column, pattern):
    return text(frame, column).str.contains(pattern, case=False, regex=True).astype(float)


def direction_from_return(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values > 0, 1, np.where(values < 0, -1, 0))


def add_feature(output, name, values):
    output[f"feature_{name}"] = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)


def build_rows(log_frame):
    frame = log_frame.copy()
    frame["timestamp"] = pd.to_numeric(frame.get("timestamp"), errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    for column in RETURN_COLUMNS:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame = frame.dropna(subset=RETURN_COLUMNS)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    output = pd.DataFrame()
    output["timestamp"] = frame["timestamp"].astype(np.int64)
    output["time"] = frame.get("time", "")
    output["symbol"] = SYMBOL
    output["primary_venue"] = VENUE_TAG

    sell = numeric(frame, "flow_1s_prob_sell")
    neutral = numeric(frame, "flow_1s_prob_neutral")
    buy = numeric(frame, "flow_1s_prob_buy")
    pressure = numeric(frame, "flow_1s_pred_pressure")
    abs_pressure = pressure.abs()
    class_pressure_disagreement = numeric(frame, "flow_1s_class_pressure_disagreement")

    up_scare = numeric(frame, "micro_upside_scare_prob")
    down_scare = numeric(frame, "micro_downside_scare_prob")
    max_scare = pd.concat([up_scare, down_scare, numeric(frame, "micro_scare_score")], axis=1).max(axis=1)
    spread = numeric(frame, "micro_spread_expansion_prob")
    bid_drop = numeric(frame, "micro_bid_liquidity_drop_prob")
    ask_drop = numeric(frame, "micro_ask_liquidity_drop_prob")
    max_liquidity = pd.concat([bid_drop, ask_drop], axis=1).max(axis=1)

    both_agree_bullish = ((pressure > 0) & (up_scare > down_scare)).astype(float)
    both_agree_bearish = ((pressure < 0) & (down_scare > up_scare)).astype(float)
    flow_bull_micro_bear = ((pressure > 0) & (down_scare > up_scare)).astype(float)
    flow_bear_micro_bull = ((pressure < 0) & (up_scare > down_scare)).astype(float)

    add_feature(output, "flow_1s_prob_sell", sell)
    add_feature(output, "flow_1s_prob_neutral", neutral)
    add_feature(output, "flow_1s_prob_buy", buy)
    add_feature(output, "flow_1s_pred_pressure", pressure)
    add_feature(output, "abs_flow_1s_pred_pressure", abs_pressure)
    add_feature(output, "flow_1s_class_pressure_disagreement", class_pressure_disagreement)

    add_feature(output, "micro_upside_scare_prob", up_scare)
    add_feature(output, "micro_downside_scare_prob", down_scare)
    add_feature(output, "max_micro_scare_probability", max_scare)
    add_feature(output, "micro_spread_expansion_prob", spread)
    add_feature(output, "micro_bid_liquidity_drop_prob", bid_drop)
    add_feature(output, "micro_ask_liquidity_drop_prob", ask_drop)
    add_feature(output, "max_liquidity_drop_probability", max_liquidity)
    add_feature(output, "micro_10s_event_only", numeric(frame, "micro_10s_event_only"))
    for source_column in [
        "micro_continuation_30s_prob",
        "micro_continuation_60s_prob",
        "micro_direction_flip_10s_prob",
        "micro_reversal_after_upside_scare_30s_prob",
        "micro_reversal_after_downside_scare_30s_prob",
        "micro_reversal_after_upside_scare_60s_prob",
        "micro_reversal_after_downside_scare_60s_prob",
    ]:
        add_feature(output, source_column, numeric(frame, source_column))

    add_feature(output, "both_agree_bullish", both_agree_bullish)
    add_feature(output, "both_agree_bearish", both_agree_bearish)
    add_feature(output, "flow_bullish_micro_bearish_conflict", flow_bull_micro_bear)
    add_feature(output, "flow_bearish_micro_bullish_conflict", flow_bear_micro_bull)
    add_feature(output, "agreement_score", numeric(frame, "agreement_score"))
    add_feature(output, "contradiction_score", numeric(frame, "contradiction_score"))
    add_feature(output, "spread_expansion_risk_high", flag_from_text(frame, "abstain_reason", "spread_expansion_risk_high"))
    add_feature(output, "liquidity_risk_high", flag_from_text(frame, "abstain_reason", "liquidity(?:_drop)?_risk_high"))
    add_feature(output, "abstain_has_reason", text(frame, "abstain_reason").str.len().gt(0).astype(float))
    add_feature(output, "abstain_3m_unavailable", flag_from_text(frame, "abstain_reason", "3m_optional_unavailable_or_stale"))
    add_feature(output, "abstain_regime_unavailable", flag_from_text(frame, "abstain_reason", "(?:15m|30m)_optional_unavailable_or_stale"))
    add_feature(output, "abstain_htf_unavailable", flag_from_text(frame, "abstain_reason", "htf_optional_unavailable_or_stale"))

    add_feature(output, "regime_15m_available", text(frame, "regime_15m_status").eq("fresh").astype(float))
    add_feature(output, "regime_30m_available", text(frame, "regime_30m_status").eq("fresh").astype(float))
    add_feature(output, "htf_available", text(frame, "htf_status").eq("fresh").astype(float))
    add_feature(output, "path_3m_available", text(frame, "path_3m_status").eq("fresh").astype(float))
    add_feature(output, "path_3m_stale_or_unavailable", (~text(frame, "path_3m_status").eq("fresh")).astype(float))

    for horizon in ["10s", "30s", "60s"]:
        return_column = f"realized_return_{horizon}"
        output[f"target_realized_return_{horizon}"] = frame[return_column].astype(float)
        output[f"target_realized_direction_{horizon}"] = direction_from_return(frame[return_column].to_numpy(dtype=float))

    return output


def print_diagnostics(log_frame, rows):
    print("Hierarchy mediator training row builder")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {VENUE_TAG}")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Hierarchy log rows: {len(log_frame)}")
    print(f"Usable mediator rows: {len(rows)}")
    print(f"Feature count: {len([column for column in rows.columns if column.startswith('feature_')])}")
    if len(rows):
        print(f"First timestamp: {int(rows['timestamp'].min())}")
        print(f"Last timestamp: {int(rows['timestamp'].max())}")
        for horizon in ["10s", "30s", "60s"]:
            values = rows[f"target_realized_direction_{horizon}"]
            counts = values.value_counts().to_dict()
            print(f"Direction distribution {horizon}: {counts}")
    print("Paper-only. No trades/orders/private API.")


def main():
    log_frame = read_csv(INPUT_PATH)
    if len(log_frame) == 0:
        rows = pd.DataFrame()
        atomic_write_csv(rows, OUTPUT_PATH)
        print_diagnostics(log_frame, rows)
        return
    rows = build_rows(log_frame)
    atomic_write_csv(rows, OUTPUT_PATH)
    print_diagnostics(log_frame, rows)


if __name__ == "__main__":
    main()
