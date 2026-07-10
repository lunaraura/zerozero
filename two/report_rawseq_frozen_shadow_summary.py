#!/usr/bin/env python3
"""Combine frozen rawseq shadow reports into a threshold decision summary."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip()
REPORT_DIR = Path(
    os.getenv(
        "RAWSEQ_SHADOW_REPORT_DIR",
        PROJECT_ROOT / "data" / "realtime" / PRIMARY_VENUE,
    )
)
SUMMARY_COST_BPS = float(os.getenv("RAWSEQ_SHADOW_SUMMARY_COST_BPS", "0.1"))
MIN_SELECTED_ROWS = int(os.getenv("RAWSEQ_SHADOW_SUMMARY_MIN_SELECTED_ROWS", "500"))
MIN_ROLLING_POSITIVE_FRACTION = float(
    os.getenv("RAWSEQ_SHADOW_SUMMARY_MIN_ROLLING_POSITIVE_FRACTION", "0.60")
)

if not REPORT_DIR.is_absolute():
    REPORT_DIR = PROJECT_ROOT / REPORT_DIR

OUTPUT_TXT = REPORT_DIR / f"{SYMBOL}_rawseq_frozen_shadow_summary.txt"
OUTPUT_CSV = REPORT_DIR / f"{SYMBOL}_rawseq_frozen_shadow_summary.csv"
COMPARISON_CSV = REPORT_DIR / f"{SYMBOL}_rawseq_frozen_shadow_threshold_rolling_comparison.csv"

COST_SENSITIVITY_PATH = REPORT_DIR / f"{SYMBOL}_rawseq_frozen_shadow_cost_sensitivity.csv"
THRESHOLD_SUMMARY_PATH = REPORT_DIR / f"{SYMBOL}_rawseq_frozen_shadow_threshold_sweep_summary.csv"
THRESHOLD_GLOB = f"{SYMBOL}_rawseq_frozen_shadow_cost_threshold_*.csv"
ROLLING_GLOB = f"{SYMBOL}_rawseq_frozen_shadow_rolling_*.csv"
SMALL_EPSILON = 1e-9
ROLLING_WINDOWS_OF_INTEREST = [6.0, 12.0, 24.0]


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except Exception:
        return math.nan


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def cost_label(cost: float) -> str:
    return f"{cost:g}".replace(".", "p").replace("-", "m")


def metric_column(metric: str) -> str:
    return f"{metric}_cost_{cost_label(SUMMARY_COST_BPS)}"


def row_for_cost(frame: pd.DataFrame, cost: float) -> pd.Series | None:
    if frame.empty or "cost_bps" not in frame.columns:
        return None
    cost_values = pd.to_numeric(frame["cost_bps"], errors="coerce")
    matches = frame[np.isclose(cost_values, cost, atol=1e-9, rtol=0.0)]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_threshold_summary_from_files() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(REPORT_DIR.glob(THRESHOLD_GLOB)):
        frame = read_csv_optional(path)
        if frame.empty or "threshold_bps" not in frame.columns:
            continue
        selected_source = row_for_cost(frame, 0.0)
        if selected_source is None:
            selected_source = row_for_cost(frame, SUMMARY_COST_BPS)
        source = row_for_cost(frame, SUMMARY_COST_BPS)
        if selected_source is None or source is None:
            continue
        rows.append(
            {
                "threshold_bps": finite_or_nan(source.get("threshold_bps")),
                "selected_rows_at_cost_0": int(finite_or_nan(selected_source.get("selected_rows", 0)) or 0),
                metric_column("cum_net_bps"): finite_or_nan(source.get("cum_net_bps")),
                metric_column("avg_net_bps"): finite_or_nan(source.get("avg_net_bps")),
                metric_column("max_dip_net_bps"): finite_or_nan(source.get("max_dip_net_bps")),
                metric_column("win_rate_net"): finite_or_nan(source.get("win_rate_net")),
                "source_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_threshold_summary() -> pd.DataFrame:
    summary = read_csv_optional(THRESHOLD_SUMMARY_PATH)
    if summary.empty:
        return build_threshold_summary_from_files()
    if metric_column("cum_net_bps") in summary.columns:
        return summary
    return build_threshold_summary_from_files()


def load_rolling_reports() -> pd.DataFrame:
    frames = []
    for path in sorted(REPORT_DIR.glob(ROLLING_GLOB)):
        frame = read_csv_optional(path)
        if not frame.empty:
            frame["source_path"] = str(path)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    dedupe_columns = [
        column
        for column in ["threshold_bps", "window_hours", "window_start_time", "window_end_time"]
        if column in combined.columns
    ]
    if len(dedupe_columns) == 4:
        combined = combined.drop_duplicates(dedupe_columns, keep="first")
    return combined.reset_index(drop=True)


def count_positive_windows(frame: pd.DataFrame, window_hours: float) -> tuple[int, int]:
    if frame.empty or "window_hours" not in frame.columns:
        return 0, 0
    hours = pd.to_numeric(frame["window_hours"], errors="coerce")
    subset = frame[np.isclose(hours, window_hours, atol=1e-9, rtol=0.0)]
    if subset.empty:
        return 0, 0
    cum_net = pd.to_numeric(subset["cum_net_bps"], errors="coerce")
    return int((cum_net > 0.0).sum()), int(cum_net.notna().sum())


def rolling_metrics_for_threshold(rolling: pd.DataFrame, threshold_bps: float) -> dict[str, Any]:
    empty = {
        "rolling_windows": 0,
        "rolling_positive_fraction": math.nan,
        "rolling_positive_6h": 0,
        "rolling_total_6h": 0,
        "rolling_positive_12h": 0,
        "rolling_total_12h": 0,
        "rolling_positive_24h": 0,
        "rolling_total_24h": 0,
        "rolling_total_cum_net": math.nan,
        "rolling_worst_window_cum_net": math.nan,
        "rolling_worst_window_max_dip": math.nan,
        "rolling_worst_window": "",
    }
    if rolling.empty or "threshold_bps" not in rolling.columns:
        return empty

    frame = rolling.copy()
    frame["threshold_bps"] = pd.to_numeric(frame["threshold_bps"], errors="coerce")
    matched = frame[np.isclose(frame["threshold_bps"], threshold_bps, atol=1e-9, rtol=0.0)].copy()
    if matched.empty:
        return empty

    matched["cum_net_bps"] = pd.to_numeric(matched["cum_net_bps"], errors="coerce")
    matched["max_dip_net_bps"] = pd.to_numeric(matched["max_dip_net_bps"], errors="coerce")
    valid_cum = matched["cum_net_bps"].dropna()
    if valid_cum.empty:
        return empty

    result = dict(empty)
    result["rolling_windows"] = int(len(valid_cum))
    result["rolling_positive_fraction"] = float((valid_cum > 0.0).mean())
    for window_hours in ROLLING_WINDOWS_OF_INTEREST:
        positive, total = count_positive_windows(matched, window_hours)
        label = f"{int(window_hours)}h"
        result[f"rolling_positive_{label}"] = positive
        result[f"rolling_total_{label}"] = total
    result["rolling_total_cum_net"] = float(valid_cum.sum())
    result["rolling_worst_window_cum_net"] = float(valid_cum.min())

    worst = matched.sort_values(["cum_net_bps", "max_dip_net_bps"], ascending=[True, True]).iloc[0]
    result["rolling_worst_window_max_dip"] = finite_or_nan(worst.get("max_dip_net_bps"))
    result["rolling_worst_window"] = (
        f"{worst.get('window_hours', '')}h "
        f"{worst.get('window_start_time', '')} to {worst.get('window_end_time', '')}"
    )
    return result


def threshold_status(row: dict[str, Any]) -> str:
    cum_net = finite_or_nan(row.get("cum_net_0p10"))
    selected_rows = int(row.get("selected_rows", 0) or 0)
    rolling_positive_fraction = finite_or_nan(row.get("rolling_positive_fraction"))
    max_dip = finite_or_nan(row.get("max_dip_0p10"))
    worst_window = finite_or_nan(row.get("rolling_worst_window_cum_net"))

    if not math.isfinite(cum_net) or cum_net <= 0.0 or selected_rows < MIN_SELECTED_ROWS:
        return "reject"
    if not math.isfinite(rolling_positive_fraction) or rolling_positive_fraction < MIN_ROLLING_POSITIVE_FRACTION:
        return "research_only"
    if math.isfinite(max_dip) and abs(max_dip) > max(cum_net * 2.5, 250.0):
        return "research_only"
    if math.isfinite(worst_window) and abs(worst_window) > max(cum_net * 1.5, 150.0):
        return "research_only"
    return "keep_shadow"


def notes_for(row: dict[str, Any]) -> str:
    notes: list[str] = []
    cum_net = finite_or_nan(row.get("cum_net_0p10"))
    selected_rows = int(row.get("selected_rows", 0) or 0)
    rolling_positive_fraction = finite_or_nan(row.get("rolling_positive_fraction"))
    max_dip = finite_or_nan(row.get("max_dip_0p10"))
    worst_window = finite_or_nan(row.get("rolling_worst_window_cum_net"))

    if not math.isfinite(cum_net):
        notes.append("missing cost-threshold metric")
    elif cum_net <= 0:
        notes.append("non-positive 0.10bps cumulative net")
    if selected_rows < MIN_SELECTED_ROWS:
        notes.append("sparse selected rows")
    if not math.isfinite(rolling_positive_fraction):
        notes.append("no matching rolling coverage")
    elif rolling_positive_fraction < MIN_ROLLING_POSITIVE_FRACTION:
        notes.append("weak rolling positive fraction")
    if math.isfinite(max_dip) and math.isfinite(cum_net) and abs(max_dip) > max(cum_net * 2.5, 250.0):
        notes.append("drawdown too large vs profit")
    if math.isfinite(worst_window) and math.isfinite(cum_net) and abs(worst_window) > max(cum_net * 1.5, 150.0):
        notes.append("worst window too large vs profit")
    return "; ".join(notes)


def threshold_score(row: dict[str, Any]) -> float:
    cum_net = finite_or_nan(row.get("cum_net_0p10"))
    avg_net = finite_or_nan(row.get("avg_net_0p10"))
    selected_rows = int(row.get("selected_rows", 0) or 0)
    rolling_positive_fraction = finite_or_nan(row.get("rolling_positive_fraction"))
    drawdown_ratio = finite_or_nan(row.get("drawdown_to_profit_ratio"))
    worst_ratio = finite_or_nan(row.get("worst_window_to_profit_ratio"))
    rolling_windows = int(row.get("rolling_windows", 0) or 0)
    coverage = sum(1 for hours in [6, 12, 24] if int(row.get(f"rolling_total_{hours}h", 0) or 0) > 0)

    score = 0.0
    score += max(cum_net, -1_000.0) if math.isfinite(cum_net) else -1_000.0
    score += (avg_net * 1_000.0) if math.isfinite(avg_net) else 0.0
    score += min(selected_rows, MIN_SELECTED_ROWS) * 0.10
    if selected_rows >= MIN_SELECTED_ROWS:
        score += 500.0
    else:
        score -= (MIN_SELECTED_ROWS - selected_rows) * 5.0
    if math.isfinite(rolling_positive_fraction):
        score += rolling_positive_fraction * 1_000.0
        if rolling_positive_fraction >= MIN_ROLLING_POSITIVE_FRACTION:
            score += 750.0
    else:
        score -= 1_000.0
    score += coverage * 150.0
    score += min(rolling_windows, 12) * 10.0
    score -= (drawdown_ratio * 150.0) if math.isfinite(drawdown_ratio) else 750.0
    score -= (worst_ratio * 200.0) if math.isfinite(worst_ratio) else 750.0
    if finite_or_nan(row.get("cum_net_0p10")) <= 0.0:
        score -= 1_500.0
    return float(score)


def build_threshold_comparison(threshold_summary: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    if threshold_summary.empty:
        return pd.DataFrame()

    cum_col = metric_column("cum_net_bps")
    avg_col = metric_column("avg_net_bps")
    dip_col = metric_column("max_dip_net_bps")
    win_col = metric_column("win_rate_net")
    rows: list[dict[str, Any]] = []

    for _, source in threshold_summary.iterrows():
        threshold = finite_or_nan(source.get("threshold_bps"))
        if not math.isfinite(threshold):
            continue
        cum_net = finite_or_nan(source.get(cum_col))
        max_dip = finite_or_nan(source.get(dip_col))
        rolling_stats = rolling_metrics_for_threshold(rolling, threshold)
        worst_window = finite_or_nan(rolling_stats.get("rolling_worst_window_cum_net"))
        profit_denominator = max(cum_net, SMALL_EPSILON) if math.isfinite(cum_net) else SMALL_EPSILON
        row = {
            "threshold_bps": threshold,
            "selected_rows": int(finite_or_nan(source.get("selected_rows_at_cost_0", 0)) or 0),
            "cum_net_0p10": cum_net,
            "avg_net_0p10": finite_or_nan(source.get(avg_col)),
            "max_dip_0p10": max_dip,
            "win_rate_0p10": finite_or_nan(source.get(win_col)),
            **rolling_stats,
            "drawdown_to_profit_ratio": abs(max_dip) / profit_denominator if math.isfinite(max_dip) else math.nan,
            "worst_window_to_profit_ratio": (
                abs(worst_window) / profit_denominator if math.isfinite(worst_window) else math.nan
            ),
        }
        row["threshold_status"] = threshold_status(row)
        row["notes"] = notes_for(row)
        row["threshold_score"] = threshold_score(row)
        rows.append(row)

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    return comparison.sort_values(
        ["threshold_status", "threshold_score", "rolling_positive_fraction", "cum_net_0p10"],
        key=lambda series: series.map({"keep_shadow": 2, "research_only": 1, "reject": 0})
        if series.name == "threshold_status"
        else series,
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def best_by_cumulative(comparison: pd.DataFrame) -> pd.Series | None:
    if comparison.empty:
        return None
    frame = comparison.copy()
    frame["cum_net_0p10"] = pd.to_numeric(frame["cum_net_0p10"], errors="coerce")
    frame = frame.dropna(subset=["cum_net_0p10"])
    if frame.empty:
        return None
    return frame.sort_values(["cum_net_0p10", "selected_rows"], ascending=[False, False]).iloc[0]


def best_by_combined(comparison: pd.DataFrame) -> pd.Series | None:
    if comparison.empty:
        return None
    frame = comparison.copy()
    return frame.sort_values(
        ["threshold_score", "rolling_positive_fraction", "drawdown_to_profit_ratio", "cum_net_0p10"],
        ascending=[False, False, True, False],
    ).iloc[0]


def format_number(value: Any, digits: int = 2) -> str:
    number = finite_or_nan(value)
    return "nan" if not math.isfinite(number) else f"{number:.{digits}f}"


def row_summary(row: pd.Series | None) -> str:
    if row is None:
        return "none"
    return (
        f"threshold={format_number(row.get('threshold_bps'), 4)} "
        f"status={row.get('threshold_status', '')} "
        f"selected={int(row.get('selected_rows', 0) or 0)} "
        f"cum_net={format_number(row.get('cum_net_0p10'))} "
        f"max_dip={format_number(row.get('max_dip_0p10'))} "
        f"rolling_pos={format_number(row.get('rolling_positive_fraction'), 4)} "
        f"score={format_number(row.get('threshold_score'))}"
    )


def render_top_rows(comparison: pd.DataFrame, limit: int = 5) -> list[str]:
    if comparison.empty:
        return ["  no threshold comparison rows"]
    lines = []
    columns = [
        "threshold_bps",
        "threshold_status",
        "selected_rows",
        "cum_net_0p10",
        "max_dip_0p10",
        "rolling_positive_fraction",
        "rolling_worst_window_cum_net",
        "threshold_score",
    ]
    widths = {
        "threshold_bps": 10,
        "threshold_status": 14,
        "selected_rows": 8,
        "cum_net_0p10": 10,
        "max_dip_0p10": 10,
        "rolling_positive_fraction": 11,
        "rolling_worst_window_cum_net": 12,
        "threshold_score": 10,
    }
    lines.append("  " + " ".join(column[: widths[column]].ljust(widths[column]) for column in columns))
    lines.append("  " + " ".join("-" * widths[column] for column in columns))
    for _, row in comparison.head(limit).iterrows():
        values = {
            "threshold_bps": format_number(row.get("threshold_bps"), 4),
            "threshold_status": str(row.get("threshold_status", "")),
            "selected_rows": str(int(row.get("selected_rows", 0) or 0)),
            "cum_net_0p10": format_number(row.get("cum_net_0p10")),
            "max_dip_0p10": format_number(row.get("max_dip_0p10")),
            "rolling_positive_fraction": format_number(row.get("rolling_positive_fraction"), 4),
            "rolling_worst_window_cum_net": format_number(row.get("rolling_worst_window_cum_net")),
            "threshold_score": format_number(row.get("threshold_score")),
        }
        lines.append("  " + " ".join(values[column].ljust(widths[column]) for column in columns))
    return lines


def recommendation_reason(recommended: pd.Series | None, best_cum: pd.Series | None) -> str:
    if recommended is None:
        return "No threshold comparison could be built."
    parts = []
    if best_cum is not None and finite_or_nan(best_cum.get("threshold_bps")) != finite_or_nan(recommended.get("threshold_bps")):
        parts.append(
            "combined score chose a different threshold than raw cumulative net because rolling/drawdown penalties mattered"
        )
    if str(recommended.get("notes", "")).strip():
        parts.append(str(recommended.get("notes")))
    else:
        parts.append("threshold passed the configured decision rules")
    return "; ".join(parts)


def build_summary_metrics(comparison: pd.DataFrame) -> dict[str, Any]:
    best_cum = best_by_cumulative(comparison)
    recommended = best_by_combined(comparison)
    metrics: dict[str, Any] = {
        "symbol": SYMBOL,
        "venue": PRIMARY_VENUE,
        "report_dir": str(REPORT_DIR),
        "summary_cost_bps": SUMMARY_COST_BPS,
        "min_selected_rows": MIN_SELECTED_ROWS,
        "min_rolling_positive_fraction": MIN_ROLLING_POSITIVE_FRACTION,
        "threshold_files": len(list(REPORT_DIR.glob(THRESHOLD_GLOB))),
        "rolling_files": len(list(REPORT_DIR.glob(ROLLING_GLOB))),
        "comparison_path": str(COMPARISON_CSV),
    }
    if best_cum is not None:
        metrics.update(
            {
                "best_cumulative_threshold_bps": finite_or_nan(best_cum.get("threshold_bps")),
                "best_cumulative_cum_net_0p10": finite_or_nan(best_cum.get("cum_net_0p10")),
                "best_cumulative_status": best_cum.get("threshold_status", ""),
            }
        )
    if recommended is not None:
        metrics.update(
            {
                "recommended_threshold_bps": finite_or_nan(recommended.get("threshold_bps")),
                "recommendation": recommended.get("threshold_status", "reject"),
                "recommended_selected_rows": int(recommended.get("selected_rows", 0) or 0),
                "recommended_cum_net_0p10": finite_or_nan(recommended.get("cum_net_0p10")),
                "recommended_avg_net_0p10": finite_or_nan(recommended.get("avg_net_0p10")),
                "recommended_max_dip_0p10": finite_or_nan(recommended.get("max_dip_0p10")),
                "recommended_rolling_positive_fraction": finite_or_nan(
                    recommended.get("rolling_positive_fraction")
                ),
                "recommended_worst_window_cum_net": finite_or_nan(
                    recommended.get("rolling_worst_window_cum_net")
                ),
                "recommended_threshold_score": finite_or_nan(recommended.get("threshold_score")),
                "recommendation_reason": recommendation_reason(recommended, best_cum),
            }
        )
    else:
        metrics["recommendation"] = "reject"
        metrics["recommendation_reason"] = "No threshold comparison could be built."
    return metrics


def render_text(metrics: dict[str, Any], comparison: pd.DataFrame) -> str:
    best_cum = best_by_cumulative(comparison)
    best_combined = best_by_combined(comparison)
    lines = [
        "Rawseq Frozen Shadow Threshold Decision Summary",
        "",
        f"Symbol: {metrics['symbol']}",
        f"Venue: {metrics['venue']}",
        f"Report dir: {metrics['report_dir']}",
        f"Summary cost bps: {SUMMARY_COST_BPS:g}",
        "",
        "Best By Cumulative 0.10 bps Net",
        f"  {row_summary(best_cum)}",
        "",
        "Best By Combined Rolling Stability",
        f"  {row_summary(best_combined)}",
        "",
        f"Recommended threshold: {format_number(metrics.get('recommended_threshold_bps'), 4)}",
        f"Recommendation: {metrics.get('recommendation', 'reject')}",
        f"Why: {metrics.get('recommendation_reason', '')}",
        "",
        "Top 5 Threshold Comparison Rows",
        *render_top_rows(comparison, limit=5),
        "",
        "Notes:",
        "- Recommendation is report-only and does not promote models or place orders.",
        "- Combined score favors rolling support and drawdown control over raw cumulative net alone.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    threshold_summary = load_threshold_summary()
    rolling = load_rolling_reports()
    comparison = build_threshold_comparison(threshold_summary, rolling)
    metrics = build_summary_metrics(comparison)
    text = render_text(metrics, comparison)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_CSV, index=False)
    OUTPUT_TXT.write_text(text, encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(OUTPUT_CSV, index=False)
    print(text)
    print(f"Comparison CSV: {COMPARISON_CSV}")
    print(f"CSV summary: {OUTPUT_CSV}")
    print(f"Text summary: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
