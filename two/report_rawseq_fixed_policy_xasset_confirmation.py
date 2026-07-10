#!/usr/bin/env python3
"""Cross-symbol fixed-rule confirmation for frozen rawseq shadow outputs.

Applies one policy/threshold/cost to SOL, DOGE, BNB, and LINK without
per-symbol tuning. Read-only except for writing reports.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENUE = os.getenv("PRIMARY_VENUE", os.getenv("RAWSEQ_XASSET_VENUE", "kraken")).strip().lower()
SYMBOLS = [
    item.strip().upper()
    for item in os.getenv("RAWSEQ_XASSET_SYMBOLS", "SOLUSDT,DOGEUSDT,BNBUSDT,LINKUSDT").split(",")
    if item.strip()
]
POLICY = os.getenv("RAWSEQ_XASSET_POLICY", "inverse_gt").strip().lower()
THRESHOLD_BPS = float(os.getenv("RAWSEQ_XASSET_THRESHOLD_BPS", "0.3"))
COST_BPS = float(os.getenv("RAWSEQ_XASSET_COST_BPS", "0.10"))
TEST_FRAC = float(os.getenv("RAWSEQ_XASSET_TEST_FRAC", "0.20"))
MAX_RUN_FOLDERS_PER_SYMBOL = int(os.getenv("RAWSEQ_XASSET_MAX_RUN_FOLDERS_PER_SYMBOL", "10"))
INCLUDE_REALTIME = os.getenv("RAWSEQ_XASSET_INCLUDE_REALTIME", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
INCLUDE_CONFIRM_RUNS = os.getenv("RAWSEQ_XASSET_INCLUDE_CONFIRM_RUNS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
EXPECTED_FEATURE = os.getenv("RAWSEQ_XASSET_EXPECTED_FEATURE", "ma_distance").strip().lower()
EXPECTED_MA_WINDOW = os.getenv("RAWSEQ_XASSET_EXPECTED_MA_WINDOW", "60").strip()
EXPECTED_SEQ_LEN = os.getenv("RAWSEQ_XASSET_EXPECTED_SEQ_LEN", "60").strip()
EXPECTED_BUCKET_SECONDS = os.getenv("RAWSEQ_XASSET_EXPECTED_BUCKET_SECONDS", "10").strip()
SPREAD_PERCENT_TO_BPS = float(os.getenv("RAWSEQ_XASSET_SPREAD_PERCENT_TO_BPS", "100.0"))

REALTIME_DIR = PROJECT_ROOT / "data" / "realtime" / VENUE
RUN_ROOT = Path(os.getenv("RAWSEQ_XASSET_RUN_ROOT", PROJECT_ROOT / "data" / "rawseq_runs"))
if not RUN_ROOT.is_absolute():
    RUN_ROOT = PROJECT_ROOT / RUN_ROOT

OUTPUT_PATH = Path(
    os.getenv(
        "RAWSEQ_XASSET_OUTPUT_PATH",
        REALTIME_DIR / "rawseq_fixed_policy_xasset_confirmation.csv",
    )
)
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH
ROLLUP_PATH = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_rollup.csv")
TEXT_OUTPUT_PATH = OUTPUT_PATH.with_suffix(".txt")

PRED_COLUMN = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COLUMN = "rawseq_path_actual_horizon_return_bps"
BASE_COLUMNS = ["timestamp", "time", PRED_COLUMN, ACTUAL_COLUMN]
META_COLUMNS = [
    "rawseq_input_feature",
    "rawseq_ma_window",
    "rawseq_bucket_seconds",
    "rawseq_len",
    "target_horizon_seconds",
]
FLOW_COLUMNS = [
    "timestamp",
    "spread_percent",
    "bid_depth_10bps",
    "ask_depth_10bps",
    "bid_depth_25bps",
    "ask_depth_25bps",
    "total_trade_volume_10s",
    "trade_count_10s",
]
SEED_RE = re.compile(r"(?:^|_)seed[_-]?(\d+)(?:_|$)", re.IGNORECASE)


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except Exception:
        return math.nan


def max_dip_bps(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    cumulative = np.cumsum(values)
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def count_csv_data_rows(path: Path) -> int:
    with path.open("rb") as handle:
        line_count = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))
    return max(0, line_count - 1)


def infer_seed(path: Path) -> str:
    match = SEED_RE.search(path.parent.name)
    return match.group(1) if match else ""


def discover_sources(symbol: str) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if INCLUDE_REALTIME:
        realtime_path = REALTIME_DIR / f"{symbol}_tiny_price_prediction_evaluation_rows_with_rawseq_path_v1.csv"
        if realtime_path.exists():
            sources.append(
                {
                    "symbol": symbol,
                    "source_type": "realtime_latest",
                    "run_name": "realtime_latest",
                    "seed": "",
                    "path": realtime_path,
                }
            )

    if INCLUDE_CONFIRM_RUNS and RUN_ROOT.exists():
        pattern = f"xasset_confirm_{VENUE}_b10s_{symbol}_ma_distance_60_h2x2*_seed_*"
        run_dirs = sorted(
            [path for path in RUN_ROOT.glob(pattern) if path.is_dir()],
            key=lambda path: path.name,
        )
        for run_dir in run_dirs[:MAX_RUN_FOLDERS_PER_SYMBOL]:
            annotated_path = run_dir / "annotated.csv"
            if not annotated_path.exists():
                continue
            sources.append(
                {
                    "symbol": symbol,
                    "source_type": "confirm_run_folder",
                    "run_name": run_dir.name,
                    "seed": infer_seed(annotated_path),
                    "path": annotated_path,
                }
            )
    return sources


def load_test_slice(path: Path) -> pd.DataFrame:
    total_rows = count_csv_data_rows(path)
    if total_rows <= 0:
        raise ValueError("CSV has no data rows")
    split_at = int(total_rows * (1.0 - TEST_FRAC))
    usecols = lambda column: column in set(BASE_COLUMNS + META_COLUMNS)
    try:
        frame = pd.read_csv(
            path,
            usecols=usecols,
            skiprows=range(1, split_at + 1),
            low_memory=False,
        )
    except ValueError as exc:
        raise ValueError(f"missing required columns: {exc}") from exc
    missing = [column for column in BASE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame[PRED_COLUMN] = pd.to_numeric(frame[PRED_COLUMN], errors="coerce")
    frame[ACTUAL_COLUMN] = pd.to_numeric(frame[ACTUAL_COLUMN], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["timestamp", PRED_COLUMN, ACTUAL_COLUMN]
    )
    if frame.empty:
        raise ValueError("test split has no finite timestamp/prediction/actual rows")
    frame["timestamp"] = frame["timestamp"].astype("int64")
    return frame.sort_values("timestamp").reset_index(drop=True)


def policy_returns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred = frame[PRED_COLUMN].to_numpy(dtype="float64")
    actual = frame[ACTUAL_COLUMN].to_numpy(dtype="float64")
    if POLICY == "inverse_gt":
        mask = pred > THRESHOLD_BPS
        gross = -actual
    elif POLICY == "direct_gt":
        mask = pred > THRESHOLD_BPS
        gross = actual
    elif POLICY == "inverse_directional_abs_gt":
        mask = np.abs(pred) > THRESHOLD_BPS
        gross = -np.sign(pred) * actual
    else:
        raise SystemExit(
            "RAWSEQ_XASSET_POLICY must be one of: inverse_gt, direct_gt, inverse_directional_abs_gt"
        )
    mask = np.asarray(mask, dtype=bool) & np.isfinite(gross)
    return mask, np.asarray(gross, dtype="float64")


def first_nonempty(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    series = frame[column].dropna()
    if series.empty:
        return ""
    return str(series.iloc[0]).strip()


def load_flow(symbol: str) -> pd.DataFrame:
    path = REALTIME_DIR / f"{symbol}_10s_flow.csv"
    if not path.exists():
        return pd.DataFrame(columns=FLOW_COLUMNS)
    flow = pd.read_csv(path, usecols=lambda column: column in FLOW_COLUMNS, low_memory=False)
    if "timestamp" not in flow.columns:
        return pd.DataFrame(columns=FLOW_COLUMNS)
    flow["timestamp"] = pd.to_numeric(flow["timestamp"], errors="coerce")
    flow = flow.dropna(subset=["timestamp"]).copy()
    flow["timestamp"] = flow["timestamp"].astype("int64")
    for column in FLOW_COLUMNS:
        if column not in flow.columns:
            flow[column] = np.nan
    return flow.drop_duplicates("timestamp", keep="last")


def liquidity_metrics(symbol: str, selected: pd.DataFrame, flow_cache: dict[str, pd.DataFrame]) -> dict[str, Any]:
    if selected.empty:
        return {
            "flow_join_rate_selected": math.nan,
            "median_half_spread_bps_selected": math.nan,
            "median_min_depth_10bps_selected": math.nan,
            "median_min_depth_25bps_selected": math.nan,
            "median_trade_volume_10s_selected": math.nan,
            "median_trade_count_10s_selected": math.nan,
        }
    if symbol not in flow_cache:
        flow_cache[symbol] = load_flow(symbol)
    flow = flow_cache[symbol]
    if flow.empty:
        return {
            "flow_join_rate_selected": 0.0,
            "median_half_spread_bps_selected": math.nan,
            "median_min_depth_10bps_selected": math.nan,
            "median_min_depth_25bps_selected": math.nan,
            "median_trade_volume_10s_selected": math.nan,
            "median_trade_count_10s_selected": math.nan,
        }
    joined = selected[["timestamp"]].merge(flow, on="timestamp", how="left", indicator=True)
    spread = pd.to_numeric(joined["spread_percent"], errors="coerce").clip(lower=0.0)
    bid10 = pd.to_numeric(joined["bid_depth_10bps"], errors="coerce")
    ask10 = pd.to_numeric(joined["ask_depth_10bps"], errors="coerce")
    bid25 = pd.to_numeric(joined["bid_depth_25bps"], errors="coerce")
    ask25 = pd.to_numeric(joined["ask_depth_25bps"], errors="coerce")
    min10 = pd.concat([bid10, ask10], axis=1).min(axis=1)
    min25 = pd.concat([bid25, ask25], axis=1).min(axis=1)
    return {
        "flow_join_rate_selected": float(joined["_merge"].eq("both").mean()),
        "median_half_spread_bps_selected": finite_or_nan((spread * SPREAD_PERCENT_TO_BPS * 0.5).median()),
        "median_min_depth_10bps_selected": finite_or_nan(min10.median()),
        "median_min_depth_25bps_selected": finite_or_nan(min25.median()),
        "median_trade_volume_10s_selected": finite_or_nan(
            pd.to_numeric(joined["total_trade_volume_10s"], errors="coerce").median()
        ),
        "median_trade_count_10s_selected": finite_or_nan(
            pd.to_numeric(joined["trade_count_10s"], errors="coerce").median()
        ),
    }


def config_notes(frame: pd.DataFrame) -> tuple[str, str, str, str, list[str]]:
    feature = first_nonempty(frame, "rawseq_input_feature")
    ma_window = first_nonempty(frame, "rawseq_ma_window")
    bucket_seconds = first_nonempty(frame, "rawseq_bucket_seconds")
    seq_len = first_nonempty(frame, "rawseq_len")
    notes: list[str] = []
    if feature and feature.lower() != EXPECTED_FEATURE:
        notes.append(f"feature={feature} expected={EXPECTED_FEATURE}")
    if ma_window and str(ma_window) != EXPECTED_MA_WINDOW:
        notes.append(f"ma_window={ma_window} expected={EXPECTED_MA_WINDOW}")
    if bucket_seconds and str(bucket_seconds) != EXPECTED_BUCKET_SECONDS:
        notes.append(f"bucket_seconds={bucket_seconds} expected={EXPECTED_BUCKET_SECONDS}")
    if seq_len and str(seq_len) != EXPECTED_SEQ_LEN:
        notes.append(f"seq_len={seq_len} expected={EXPECTED_SEQ_LEN}")
    return feature, ma_window, bucket_seconds, seq_len, notes


def evaluate_source(source: dict[str, Any], flow_cache: dict[str, pd.DataFrame]) -> dict[str, Any]:
    path = Path(source["path"])
    symbol = str(source["symbol"])
    try:
        frame = load_test_slice(path)
        mask, gross_all = policy_returns(frame)
        selected = frame[mask].copy()
        gross = gross_all[mask]
        gross = gross[np.isfinite(gross)]
        net = gross - COST_BPS
        feature, ma_window, bucket_seconds, seq_len, config_issue_notes = config_notes(frame)
        liq = liquidity_metrics(symbol, selected, flow_cache)
        selected_rows = int(len(net))
        notes = list(config_issue_notes)
        avg_gross = float(np.mean(gross)) if selected_rows else math.nan
        avg_net = float(np.mean(net)) if selected_rows else math.nan
        cum_net = float(np.sum(net)) if selected_rows else 0.0
        max_dip = max_dip_bps(net)
        if selected_rows == 0:
            notes.append("no selected rows at fixed threshold")
        elif cum_net <= 0.0:
            notes.append("negative post-cost fixed-rule edge")
        if selected_rows < 100:
            notes.append("sparse fixed-rule selection")
        if math.isfinite(liq["median_half_spread_bps_selected"]) and math.isfinite(avg_gross):
            if liq["median_half_spread_bps_selected"] + COST_BPS >= max(avg_gross, 0.0):
                notes.append("liquidity/spread can erase gross edge")
        if math.isfinite(max_dip) and cum_net > 0 and abs(max_dip) > cum_net:
            notes.append("drawdown exceeds profit")

        return {
            **source,
            "path": str(path),
            "status": "ok",
            "issues": "",
            "policy": POLICY,
            "threshold_bps": THRESHOLD_BPS,
            "cost_bps": COST_BPS,
            "test_frac": TEST_FRAC,
            "test_rows": int(len(frame)),
            "selected_rows": selected_rows,
            "avg_gross_bps": avg_gross,
            "avg_net_bps": avg_net,
            "cum_net_bps": cum_net,
            "win_rate_net": float(np.mean(net > 0.0)) if selected_rows else math.nan,
            "max_dip_net_bps": max_dip,
            "first_time": str(frame["time"].iloc[0]) if "time" in frame.columns else "",
            "last_time": str(frame["time"].iloc[-1]) if "time" in frame.columns else "",
            "rawseq_input_feature": feature,
            "rawseq_ma_window": ma_window,
            "rawseq_bucket_seconds": bucket_seconds,
            "rawseq_len": seq_len,
            **liq,
            "explanation_notes": "; ".join(dict.fromkeys(notes)),
            "positive_post_cost": cum_net > 0.0,
        }
    except Exception as exc:
        return {
            **source,
            "path": str(path),
            "status": "error",
            "issues": str(exc),
            "policy": POLICY,
            "threshold_bps": THRESHOLD_BPS,
            "cost_bps": COST_BPS,
            "test_frac": TEST_FRAC,
            "test_rows": 0,
            "selected_rows": 0,
            "avg_gross_bps": math.nan,
            "avg_net_bps": math.nan,
            "cum_net_bps": 0.0,
            "win_rate_net": math.nan,
            "max_dip_net_bps": math.nan,
            "first_time": "",
            "last_time": "",
            "rawseq_input_feature": "",
            "rawseq_ma_window": "",
            "rawseq_bucket_seconds": "",
            "rawseq_len": "",
            "flow_join_rate_selected": math.nan,
            "median_half_spread_bps_selected": math.nan,
            "median_min_depth_10bps_selected": math.nan,
            "median_min_depth_25bps_selected": math.nan,
            "median_trade_volume_10s_selected": math.nan,
            "median_trade_count_10s_selected": math.nan,
            "explanation_notes": "source failed to evaluate",
            "positive_post_cost": False,
        }


def summarize_symbol(group: pd.DataFrame) -> dict[str, Any]:
    ok = group[group["status"].eq("ok")].copy()
    active = ok[ok["selected_rows"] > 0].copy()
    realtime = ok[ok["source_type"].eq("realtime_latest")]
    notes: list[str] = []
    if ok.empty:
        notes.append("no evaluable sources")
    if active.empty:
        notes.append("no active fixed-rule sources")
    if not active.empty and float((active["cum_net_bps"] > 0.0).mean()) < 0.5:
        notes.append("most fixed-rule sources are negative")
    if active["selected_rows"].median() < 100 if not active.empty else False:
        notes.append("fixed threshold is sparse")
    if (
        active["median_half_spread_bps_selected"].median() + COST_BPS
        >= max(active["avg_gross_bps"].median(), 0.0)
        if not active.empty and math.isfinite(finite_or_nan(active["median_half_spread_bps_selected"].median()))
        else False
    ):
        notes.append("median spread/cost pressure explains weak edge")
    if active["rawseq_input_feature"].astype(str).str.lower().ne(EXPECTED_FEATURE).any() if not active.empty else False:
        notes.append("model/config mismatch")
    if not active.empty and active["max_dip_net_bps"].min() < -abs(active["cum_net_bps"].median()):
        notes.append("regime/drawdown instability")

    return {
        "symbol": str(group["symbol"].iloc[0]),
        "sources": int(len(group)),
        "ok_sources": int(len(ok)),
        "active_sources": int(len(active)),
        "positive_sources": int((ok["cum_net_bps"] > 0.0).sum()) if not ok.empty else 0,
        "positive_fraction_ok": float((ok["cum_net_bps"] > 0.0).mean()) if not ok.empty else math.nan,
        "median_selected_rows": finite_or_nan(active["selected_rows"].median()) if not active.empty else 0.0,
        "median_avg_net_bps": finite_or_nan(active["avg_net_bps"].median()) if not active.empty else math.nan,
        "median_cum_net_bps": finite_or_nan(active["cum_net_bps"].median()) if not active.empty else math.nan,
        "best_cum_net_bps": finite_or_nan(ok["cum_net_bps"].max()) if not ok.empty else math.nan,
        "worst_cum_net_bps": finite_or_nan(ok["cum_net_bps"].min()) if not ok.empty else math.nan,
        "median_max_dip_net_bps": finite_or_nan(active["max_dip_net_bps"].median()) if not active.empty else math.nan,
        "realtime_cum_net_bps": finite_or_nan(realtime["cum_net_bps"].iloc[0]) if not realtime.empty else math.nan,
        "realtime_selected_rows": int(realtime["selected_rows"].iloc[0]) if not realtime.empty else 0,
        "median_half_spread_bps_selected": finite_or_nan(active["median_half_spread_bps_selected"].median())
        if not active.empty
        else math.nan,
        "median_min_depth_10bps_selected": finite_or_nan(active["median_min_depth_10bps_selected"].median())
        if not active.empty
        else math.nan,
        "explanation_notes": "; ".join(dict.fromkeys(notes)),
    }


def build_reports() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    sources: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        sources.extend(discover_sources(symbol))
    if not sources:
        raise SystemExit("No xasset sources discovered.")

    flow_cache: dict[str, pd.DataFrame] = {}
    all_rows = pd.DataFrame([evaluate_source(source, flow_cache) for source in sources])
    rollup = pd.DataFrame(
        [summarize_symbol(group.reset_index(drop=True)) for _, group in all_rows.groupby("symbol", sort=True)]
    )

    sol = rollup[rollup["symbol"].eq("SOLUSDT")]
    non_sol = rollup[~rollup["symbol"].eq("SOLUSDT")]
    overall_status = "FAIL"
    if not sol.empty and not non_sol.empty:
        sol_score = finite_or_nan(sol["realtime_cum_net_bps"].iloc[0])
        non_sol_best = finite_or_nan(non_sol["realtime_cum_net_bps"].max())
        explained = non_sol["explanation_notes"].astype(str).str.len().gt(0).all()
        if math.isfinite(sol_score) and sol_score > max(non_sol_best, 0.0) and explained:
            overall_status = "PASS"

    rollup["overall_status"] = overall_status
    return all_rows, rollup, overall_status


def render_text(all_rows: pd.DataFrame, rollup: pd.DataFrame, overall_status: str) -> str:
    lines = [
        "Rawseq Fixed Policy Cross-Symbol Confirmation",
        "",
        f"Status: {overall_status}",
        f"Venue: {VENUE}",
        f"Symbols: {', '.join(SYMBOLS)}",
        f"Policy: {POLICY}",
        f"Threshold bps: {THRESHOLD_BPS:g}",
        f"Cost bps: {COST_BPS:g}",
        f"Test frac: {TEST_FRAC:g}",
        "Threshold mode: fixed shared threshold; no per-symbol tuning",
        "",
        "Symbol Rollup",
        "  symbol   sources pos_frac realtime_cum median_cum median_avg best_cum notes",
        "  -------- ------- -------- ------------ ---------- ---------- -------- ------------------------------",
    ]
    for _, row in rollup.sort_values("symbol").iterrows():
        lines.append(
            "  "
            + " ".join(
                [
                    str(row["symbol"])[:8].ljust(8),
                    str(int(row["sources"])).rjust(7),
                    f"{finite_or_nan(row['positive_fraction_ok']):.4f}".rjust(8),
                    f"{finite_or_nan(row['realtime_cum_net_bps']):.2f}".rjust(12),
                    f"{finite_or_nan(row['median_cum_net_bps']):.2f}".rjust(10),
                    f"{finite_or_nan(row['median_avg_net_bps']):.4f}".rjust(10),
                    f"{finite_or_nan(row['best_cum_net_bps']):.2f}".rjust(8),
                    str(row["explanation_notes"])[:70],
                ]
            )
        )

    lines.extend(
        [
            "",
            "Top Source Rows",
            "  symbol   source_type          seed selected cum_net avg_net max_dip notes",
            "  -------- -------------------- ---- -------- ------- ------- ------- ------------------------------",
        ]
    )
    display = all_rows.sort_values(["symbol", "source_type", "cum_net_bps"], ascending=[True, True, False])
    for _, row in display.groupby("symbol", sort=True).head(4).iterrows():
        lines.append(
            "  "
            + " ".join(
                [
                    str(row["symbol"])[:8].ljust(8),
                    str(row["source_type"])[:20].ljust(20),
                    str(row["seed"])[:4].ljust(4),
                    str(int(row["selected_rows"])).rjust(8),
                    f"{finite_or_nan(row['cum_net_bps']):.2f}".rjust(7),
                    f"{finite_or_nan(row['avg_net_bps']):.4f}".rjust(7),
                    f"{finite_or_nan(row['max_dip_net_bps']):.2f}".rjust(7),
                    str(row["explanation_notes"])[:70],
                ]
            )
        )

    lines.extend(
        [
            "",
            "Retrain reproducibility harness: not created; gated until validation items 1-9 are acceptable.",
            f"All-source CSV: {OUTPUT_PATH}",
            f"Rollup CSV: {ROLLUP_PATH}",
            f"Text report: {TEXT_OUTPUT_PATH}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    all_rows, rollup, overall_status = build_reports()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(OUTPUT_PATH, index=False)
    rollup.to_csv(ROLLUP_PATH, index=False)
    text = render_text(all_rows, rollup, overall_status)
    TEXT_OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
