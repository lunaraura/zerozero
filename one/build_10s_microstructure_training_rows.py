import os
from pathlib import Path

import pandas as pd

from microstructure_model_utils import (
    DIAGNOSTIC_TARGET_COLUMNS,
    EVENT_TARGET_COLUMNS,
    REGRESSION_TARGET_COLUMNS,
    atomic_write_csv,
    build_training_rows,
    future_window,
    get_micro_feature_columns,
    load_snapshot_rows,
    percent,
    safe_ratio,
)
from hierarchical_context import (
    attach_hierarchical_context,
    print_context_availability_summary,
    print_context_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
MICRO_MOVE_THRESHOLD = float(os.getenv("MICRO_MOVE_THRESHOLD", "0.003"))
MICRO_MOVE_THRESHOLD_MODE = os.getenv("MICRO_MOVE_THRESHOLD_MODE", "volatility").strip().lower()
MIN_MICRO_MOVE_THRESHOLD = float(os.getenv("MIN_MICRO_MOVE_THRESHOLD", "0.0002"))
MICRO_MOVE_THRESHOLD_QUANTILE = float(os.getenv("MICRO_MOVE_THRESHOLD_QUANTILE", "0.90"))
MAX_MICRO_SNAPSHOTS = int(os.getenv("MAX_MICRO_SNAPSHOTS", "0"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_OUTPUT_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR
INPUT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_10s_flow.csv"
OUTPUT_PATH = VENUE_OUTPUT_DIR / f"{SYMBOL}_10s_microstructure_training_rows.csv"


def print_event_distributions(frame):
    print("\nEvent target distributions")
    for column in EVENT_TARGET_COLUMNS:
        positives = int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())
        total = len(frame) or 1
        print(f"- {column}: {positives}/{len(frame)} ({percent(positives / total)})")


def print_regression_summary(frame):
    print("\n10s regression target summary")
    for column in [*REGRESSION_TARGET_COLUMNS, *DIAGNOSTIC_TARGET_COLUMNS]:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        print(
            f"- {column}: "
            f"min={values.min():.8g}, mean={values.mean():.8g}, max={values.max():.8g}"
        )


def print_flow1s_context_diagnostics(frame, context_diagnostics):
    info = context_diagnostics.get("flow1s", {}) if context_diagnostics else {}
    available_column = "feature_context_flow_1s_context_available"
    age_column = "feature_context_flow_1s_context_age_ms"

    print("\n1s order-flow context diagnostics")
    print(f"- flow1s context file path: {info.get('path', 'not checked')}")
    print(f"- flow1s rows loaded: {info.get('rows', 0)}")
    if frame is None or len(frame) == 0 or available_column not in frame.columns:
        print("- flow1s availability percentage: 0.00%")
        print("- flow1s available row count: 0")
        print(f"- flow1s unavailable row count: {0 if frame is None else len(frame)}")
        print("- stale/missing reason counts: unavailable; context column was not created")
        return

    available = pd.to_numeric(frame[available_column], errors="coerce").fillna(0.0) > 0
    available_count = int(available.sum())
    unavailable_count = int(len(frame) - available_count)
    availability_pct = available_count / max(1, len(frame))
    print(f"- flow1s availability percentage: {availability_pct:.2%}")
    print(f"- flow1s available row count: {available_count}")
    print(f"- flow1s unavailable row count: {unavailable_count}")

    if available_count and age_column in frame.columns:
        ages = pd.to_numeric(frame.loc[available, age_column], errors="coerce").dropna()
        if len(ages):
            print(f"- flow1s context age median ms: {float(ages.median()):.0f}")
            print(f"- flow1s context age p90 ms: {float(ages.quantile(0.90)):.0f}")
            print(f"- flow1s context age max ms: {float(ages.max()):.0f}")
        else:
            print("- flow1s context age stats: unavailable")
    else:
        print("- flow1s context age stats: unavailable")

    stale_or_missing = int((~available).sum())
    print("- stale/missing reason counts:")
    if info.get("rows", 0) == 0:
        print(f"  - context file missing/empty: {stale_or_missing}")
    else:
        print(
            "  - no causal context row or context older than MAX_FLOW_1S_CONTEXT_AGE_MS: "
            f"{stale_or_missing}"
        )


def absolute_10s_future_returns(snapshots):
    values = []
    if len(snapshots) == 0:
        return pd.Series(dtype="float64")
    for index in range(len(snapshots)):
        entry_mid = pd.to_numeric(pd.Series([snapshots.loc[index, "mid_price"]]), errors="coerce").iloc[0]
        if pd.isna(entry_mid) or float(entry_mid) <= 0:
            continue
        future = future_window(snapshots, index, 10)
        if len(future) == 0 or "mid_price" not in future.columns:
            continue
        future_mid = pd.to_numeric(future["mid_price"], errors="coerce").dropna()
        if len(future_mid) == 0:
            continue
        future_return = safe_ratio(float(future_mid.iloc[-1]) - float(entry_mid), float(entry_mid))
        values.append(abs(future_return))
    return pd.Series(values, dtype="float64")


def select_micro_move_threshold(snapshots):
    if MICRO_MOVE_THRESHOLD_MODE == "fixed":
        return MICRO_MOVE_THRESHOLD, {"mode": "fixed", "sample_count": 0, "quantile_value": MICRO_MOVE_THRESHOLD}
    if MICRO_MOVE_THRESHOLD_MODE != "volatility":
        print(
            f"Unknown MICRO_MOVE_THRESHOLD_MODE={MICRO_MOVE_THRESHOLD_MODE!r}; "
            "falling back to volatility mode."
        )
    abs_returns = absolute_10s_future_returns(snapshots)
    quantile_value = (
        float(abs_returns.quantile(MICRO_MOVE_THRESHOLD_QUANTILE))
        if len(abs_returns)
        else MICRO_MOVE_THRESHOLD
    )
    threshold = max(MIN_MICRO_MOVE_THRESHOLD, quantile_value)
    return threshold, {
        "mode": "volatility",
        "sample_count": int(len(abs_returns)),
        "quantile": MICRO_MOVE_THRESHOLD_QUANTILE,
        "quantile_value": quantile_value,
        "minimum": MIN_MICRO_MOVE_THRESHOLD,
    }


def main():
    snapshots = load_snapshot_rows(INPUT_PATH)
    original_snapshot_count = len(snapshots)
    if MAX_MICRO_SNAPSHOTS > 0 and len(snapshots) > MAX_MICRO_SNAPSHOTS:
        snapshots = snapshots.tail(MAX_MICRO_SNAPSHOTS).reset_index(drop=True)
    selected_threshold, threshold_info = select_micro_move_threshold(snapshots)
    training_rows, skipped_reasons, snapshot_step_seconds = build_training_rows(
        snapshots,
        selected_threshold,
    )
    context_diagnostics = {}
    if len(training_rows):
        training_rows, context_diagnostics = attach_hierarchical_context(
            training_rows,
            PROJECT_ROOT,
            SYMBOL,
            layers=("htf", "regime15", "regime30", "flow3m", "flow1s"),
            as_model_features=True,
        )

    if len(training_rows):
        atomic_write_csv(training_rows, OUTPUT_PATH)

    print("10s microstructure training row builder")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Input snapshot path: {INPUT_PATH}")
    print(f"Output training path: {OUTPUT_PATH}")
    print(f"Input snapshot rows available: {original_snapshot_count}")
    print(
        "Input snapshot rows used: "
        f"{len(snapshots)}"
        + (f" (MAX_MICRO_SNAPSHOTS={MAX_MICRO_SNAPSHOTS})" if MAX_MICRO_SNAPSHOTS > 0 else "")
    )
    print(f"Inferred snapshot step seconds: {snapshot_step_seconds:.3g}")
    print(f"MICRO_MOVE_THRESHOLD_MODE: {threshold_info['mode']}")
    print(f"Selected MICRO_MOVE_THRESHOLD: {selected_threshold:.4%}")
    if threshold_info["mode"] == "volatility":
        print(f"MIN_MICRO_MOVE_THRESHOLD: {threshold_info['minimum']:.4%}")
        print(f"MICRO_MOVE_THRESHOLD_QUANTILE: {threshold_info['quantile']:.2f}")
        print(f"abs 10s future return samples: {threshold_info['sample_count']}")
        print(f"quantile threshold candidate: {threshold_info['quantile_value']:.4%}")
    else:
        print(f"Fixed MICRO_MOVE_THRESHOLD: {MICRO_MOVE_THRESHOLD:.4%}")
    print(f"Generated training rows: {len(training_rows)}")
    if len(training_rows):
        print(f"First training timestamp: {int(training_rows['timestamp'].min())}")
        print(f"Last training timestamp: {int(training_rows['timestamp'].max())}")
        print_context_diagnostics(context_diagnostics)
        print_context_availability_summary(training_rows)
        print_flow1s_context_diagnostics(training_rows, context_diagnostics)
    print("Skipped rows by reason:")
    if skipped_reasons:
        for reason, count in sorted(skipped_reasons.items()):
            print(f"- {reason}: {count}")
    else:
        print("- none")

    if len(training_rows):
        feature_columns = get_micro_feature_columns(training_rows)
        print(f"Feature count: {len(feature_columns)}")
        print_event_distributions(training_rows)
        print_regression_summary(training_rows)
    else:
        print("No rows were written because no complete feature+future-label rows were available.")

    print("No trades were placed.")


if __name__ == "__main__":
    main()
