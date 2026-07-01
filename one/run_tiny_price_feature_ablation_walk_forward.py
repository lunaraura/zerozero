import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from tiny_price_feature_utils import slugify


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
VENUE_DIR = OUTPUT_DIR / PRIMARY_VENUE if PRIMARY_VENUE else OUTPUT_DIR

TARGET_SPEC = os.getenv("PRICE_TINY_FEATURE_ABLATION_TARGET_SPEC", os.getenv("PRICE_TINY_TARGET_SPEC", "move_before_adverse_30s")).strip().lower()

DEFAULT_FEATURE_GROUP_SETS = [
    "base_tiny_price_v1",
    "base_tiny_price_v1,calendar_session_features",
    "base_tiny_price_v1,spread_volatility_features",
    "base_tiny_price_v1,pressure_change_features",
    "base_tiny_price_v1,depth_acceleration_features",
    "base_tiny_price_v1,calendar_session_features,spread_volatility_features",
    "base_tiny_price_v1,calendar_session_features,pressure_change_features",
    "base_tiny_price_v1,calendar_session_features,depth_acceleration_features",
]


def parse_feature_group_sets():
    text = os.getenv("PRICE_TINY_FEATURE_ABLATION_GROUP_SETS", "").strip()
    if not text:
        return DEFAULT_FEATURE_GROUP_SETS
    # Semicolon separates ablation experiments; commas stay inside each feature set.
    return [part.strip() for part in text.split(";") if part.strip()]


FEATURE_GROUP_SETS = parse_feature_group_sets()
OUTPUT_PATH = VENUE_DIR / f"{SYMBOL}_tiny_price_feature_ablation_walk_forward.csv"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def numeric_series(frame, column):
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def mean_value(frame, column):
    values = numeric_series(frame, column)
    return float(values.mean()) if values.notna().any() else np.nan


def median_value(frame, column):
    values = numeric_series(frame, column)
    return float(values.median()) if values.notna().any() else np.nan


def min_value(frame, column):
    values = numeric_series(frame, column)
    return float(values.min()) if values.notna().any() else np.nan


def sum_int(frame, column):
    values = numeric_series(frame, column).fillna(0)
    return int(values.sum()) if len(values) else 0


def bool_any(frame, column):
    if column not in frame.columns:
        return False
    values = frame[column].fillna(False).astype(str).str.lower()
    return bool(values.isin({"true", "1", "yes"}).any())


def unique_join(frame, column):
    if column not in frame.columns:
        return ""
    values = [str(value) for value in frame[column].dropna().unique().tolist()]
    return ",".join(values)


def threshold_by_window(frame):
    if "window_index" not in frame.columns or "selected_threshold" not in frame.columns:
        return ""
    pairs = []
    for _, row in frame[["window_index", "selected_threshold"]].drop_duplicates().sort_values("window_index").iterrows():
        pairs.append(f"{int(row['window_index'])}:{float(row['selected_threshold']):.2f}")
    return "|".join(pairs)


def target_output_path():
    return VENUE_DIR / f"{SYMBOL}_tiny_price_walk_forward_{slugify(TARGET_SPEC)}.csv"


def feature_group_slug(feature_groups):
    return slugify(feature_groups.replace(",", "__"), "feature_groups")


def ablation_output_path_for(feature_groups):
    return VENUE_DIR / f"{SYMBOL}_tiny_price_walk_forward_{slugify(TARGET_SPEC)}__ablation_{feature_group_slug(feature_groups)}.csv"


def run_walk_forward_for_feature_groups(feature_groups):
    env = os.environ.copy()
    env["SYMBOL"] = SYMBOL
    env["PRIMARY_VENUE"] = PRIMARY_VENUE
    env["PRICE_TINY_FEATURE_GROUPS"] = feature_groups
    env["PRICE_TINY_WALK_FORWARD_TARGET_SPECS"] = TARGET_SPEC
    env["PRICE_TINY_REQUIRE_EXACT_FEATURE_GROUPS"] = "true"
    env["PRICE_TINY_WALK_FORWARD_OUTPUT_SUFFIX"] = f"ablation_{feature_group_slug(feature_groups)}"
    env["PROMOTE_BEST"] = "false"
    env["PRICE_TINY_AUTO_REGISTER_CHALLENGERS"] = "false"
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_tiny_price_walk_forward.py")]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    return result


def summarize_run(feature_groups, return_code, error_text=""):
    detail_path = ablation_output_path_for(feature_groups)
    detail = read_csv(detail_path)
    if return_code != 0 or len(detail) == 0:
        return {
            "symbol": SYMBOL,
            "primary_venue": PRIMARY_VENUE,
            "target_spec": TARGET_SPEC,
            "feature_groups": feature_groups,
            "feature_schema_hash": "",
            "windows": 0,
            "total_active_rows": 0,
            "positive_windows": 0,
            "negative_windows": 0,
            "mean_gross_return_bps": np.nan,
            "mean_net_return_bps": np.nan,
            "worst_window_return_bps": np.nan,
            "median_window_return_bps": np.nan,
            "mean_sign_accuracy": np.nan,
            "total_long_count": 0,
            "total_short_count": 0,
            "long_short_balance": "",
            "long_share": np.nan,
            "one_sided_warning": True,
            "selected_thresholds_by_window": "",
            "run_return_code": int(return_code),
            "run_status": "failed" if return_code != 0 else "no_detail_rows",
            "failure_reason": error_text[:1000],
            "paper_only": True,
            "no_promotion": True,
        }

    gross = numeric_series(detail, "avg_strategy_return_bps")
    net = numeric_series(detail, "estimated_net_avg_strategy_return_bps")
    active_rows = numeric_series(detail, "active_rows").fillna(0)
    long_count = sum_int(detail, "active_long_count")
    short_count = sum_int(detail, "active_short_count")
    if long_count + short_count == 0:
        long_count = sum_int(detail, "predicted_up_count")
        short_count = sum_int(detail, "predicted_down_count")
    total_directional = long_count + short_count
    return {
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE,
        "target_spec": TARGET_SPEC,
        "feature_groups": feature_groups,
        "feature_schema_hash": unique_join(detail, "feature_schema_hash"),
        "windows": int(len(detail)),
        "total_active_rows": int(active_rows.sum()),
        "positive_windows": int((gross > 0).sum()) if gross.notna().any() else 0,
        "negative_windows": int((gross < 0).sum()) if gross.notna().any() else 0,
        "mean_gross_return_bps": mean_value(detail, "avg_strategy_return_bps"),
        "mean_net_return_bps": mean_value(detail, "estimated_net_avg_strategy_return_bps"),
        "worst_window_return_bps": min_value(detail, "avg_strategy_return_bps"),
        "median_window_return_bps": median_value(detail, "avg_strategy_return_bps"),
        "mean_sign_accuracy": mean_value(detail, "sign_accuracy"),
        "total_long_count": long_count,
        "total_short_count": short_count,
        "long_short_balance": f"{long_count}:{short_count}",
        "long_share": float(long_count / total_directional) if total_directional else np.nan,
        "one_sided_warning": bool_any(detail, "one_sided_prediction_warning") or long_count == 0 or short_count == 0,
        "selected_thresholds_by_window": threshold_by_window(detail),
        "run_return_code": int(return_code),
        "run_status": "ok",
        "failure_reason": unique_join(detail, "failure_reason"),
        "source_walk_forward_output": str(detail_path),
        "paper_only": True,
        "no_promotion": True,
    }


def main():
    print("Tiny-price feature-group ablation walk-forward")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print(f"Target: {TARGET_SPEC}")
    print("Feature group sets:")
    for index, feature_groups in enumerate(FEATURE_GROUP_SETS, start=1):
        print(f"  {index}. {feature_groups}")
    print("")

    rows = []
    for index, feature_groups in enumerate(FEATURE_GROUP_SETS, start=1):
        print(f"=== Ablation {index}/{len(FEATURE_GROUP_SETS)}: {feature_groups} ===")
        result = run_walk_forward_for_feature_groups(feature_groups)
        error_text = "\n".join(part for part in [result.stdout, result.stderr] if part)
        rows.append(summarize_run(feature_groups, result.returncode, error_text))

    write_csv(rows, OUTPUT_PATH)
    print("")
    print(f"Ablation output: {OUTPUT_PATH}")
    print("")
    print("Compact summary")
    for row in rows:
        print(
            f"- {row['feature_groups']}: windows={row['windows']} active={row['total_active_rows']} "
            f"mean_gross={row['mean_gross_return_bps']:.4f}bps "
            f"mean_net={row['mean_net_return_bps']:.4f}bps "
            f"worst={row['worst_window_return_bps']:.4f}bps "
            f"long_short={row['long_short_balance']} status={row['run_status']}"
        )
    print("Research only. No promotion, registry writes, live prediction writes, orders, or private API use.")


if __name__ == "__main__":
    main()
