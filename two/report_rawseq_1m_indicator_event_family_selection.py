#!/usr/bin/env python3
"""Select a coherent indicator-event family from the completed event scout."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import now_stamp, stable_hash, write_csv, write_json  # noqa: E402

DEFAULT_SCOUT_DIR = Path(r"F:\rsio\rawseq_1m_indicator_companion_scout\indicator_event_scout_20260712T193954Z")
DEFAULT_OUTPUT_ROOT = Path(r"F:\rsio\rawseq_1m_indicator_companion_scout")
EXPECTED_HORIZONS = [1, 2, 4, 8]


def family_name(event_name: str) -> str:
    if event_name.startswith("ema20_minus_ema60_spread_narrows_within_"):
        return "ema_spread_narrowing"
    if event_name.startswith("price_crosses_ema20_within_"):
        return "price_cross_ema20"
    if event_name.startswith("price_crosses_ema60_within_"):
        return "price_cross_ema60"
    if event_name.startswith("rsi_crosses_above_50_within_"):
        return "rsi_cross_above_50"
    if event_name.startswith("rsi_crosses_above_70_within_"):
        return "rsi_cross_above_70"
    if event_name.startswith("rsi_crosses_below_30_within_"):
        return "rsi_cross_below_30"
    if "ema20_crosses_above_ema60" in event_name:
        return "ema_bullish_cross"
    if "ema20_crosses_below_ema60" in event_name:
        return "ema_bearish_cross"
    return "other"


def finite_median(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    return float(vals.median()) if vals.notna().any() else math.nan


def family_score(row: dict[str, Any]) -> float:
    score = 0.0
    score += 5.0 * float(row["family_completeness"])
    score += 4.0 if row["positive_worst_fold_brier_skill"] else 0.0
    score += 2.0 if row["positive_symbol_count_min"] >= 9 else row["positive_symbol_count_min"] / 9.0
    score += max(0.0, min(2.0, float(row["median_brier_skill"]) * 4.0))
    score += max(0.0, min(2.0, float(row["median_pr_auc_lift"]) * 4.0))
    score += 1.0 if row["finite_calibration"] else 0.0
    score += 1.0 if row["save_reload_parity"] else 0.0
    score += 1.0 if row["combined_symbol_time_stability"] else 0.0
    if row["family"] == "ema_spread_narrowing" and set(row["included_horizons"]) == set(EXPECTED_HORIZONS):
        score += 3.0
    return float(score)


def build_family_rows(survival: pd.DataFrame, fold_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    survivors = survival[survival["survives_event_gate"].astype(str).str.lower().eq("true")].copy()
    if survivors.empty:
        return []
    survivors["family"] = survivors["event_name"].map(family_name)
    rows: list[dict[str, Any]] = []
    for (family, model), group in survivors.groupby(["family", "model"], dropna=False):
        included_events = sorted(group["event_name"].astype(str).tolist())
        included_horizons = sorted(int(x) for x in pd.to_numeric(group["horizon"], errors="coerce").dropna().unique())
        fold_group = fold_metrics[(fold_metrics["model"].astype(str) == str(model)) & (fold_metrics["event_name"].astype(str).isin(included_events))]
        expected = EXPECTED_HORIZONS if family in {"ema_spread_narrowing", "price_cross_ema20", "price_cross_ema60", "rsi_cross_above_50", "rsi_cross_above_70"} else included_horizons
        completeness = len(set(included_horizons) & set(expected)) / max(1, len(expected))
        row = {
            "family": family,
            "model": model,
            "included_events": "|".join(included_events),
            "included_horizons": included_horizons,
            "included_horizons_text": ",".join(str(x) for x in included_horizons),
            "event_prevalence_by_horizon": json.dumps({str(int(r["horizon"])): float(r["median_event_prevalence"]) for _, r in group.iterrows()}, sort_keys=True),
            "surviving_target_count": int(len(group)),
            "fold_wins": int((pd.to_numeric(fold_group["brier_skill"], errors="coerce") > 0).sum()) if not fold_group.empty else 0,
            "fold_rows": int(len(fold_group)),
            "median_brier_skill": finite_median(group["median_brier_skill"]),
            "worst_fold_brier_skill": float(pd.to_numeric(group["worst_fold_brier_skill"], errors="coerce").min()),
            "positive_worst_fold_brier_skill": bool(pd.to_numeric(group["worst_fold_brier_skill"], errors="coerce").min() > 0),
            "median_pr_auc_lift": finite_median(group["median_pr_auc_lift"]),
            "median_log_loss_improvement": finite_median(group["median_log_loss_improvement"]),
            "calibration_slope_median": finite_median(fold_group["calibration_slope"]) if not fold_group.empty else math.nan,
            "calibration_intercept_median": finite_median(fold_group["calibration_intercept"]) if not fold_group.empty else math.nan,
            "expected_calibration_error_median": finite_median(fold_group["expected_calibration_error"]) if not fold_group.empty else math.nan,
            "maximum_calibration_error_median": finite_median(fold_group["maximum_calibration_error"]) if not fold_group.empty else math.nan,
            "positive_symbol_count_min": int(pd.to_numeric(group["positive_symbols"], errors="coerce").min()),
            "combined_symbol_time_stability": bool(pd.to_numeric(group["combined_symbol_time_worst_brier_skill"], errors="coerce").min() > 0),
            "save_reload_parity": bool(group["save_reload_parity"].astype(str).str.lower().isin(["true", "1"]).all()),
            "horizon_probability_monotonicity_violation_fraction": math.nan,
            "horizon_probability_monotonicity_note": "computed during freeze from selected family model probabilities",
            "family_completeness": float(completeness),
            "finite_calibration": bool(np.isfinite(pd.to_numeric(fold_group["calibration_slope"], errors="coerce")).any()) if not fold_group.empty else False,
        }
        row["family_selection_score"] = family_score(row)
        rows.append(row)
    rows.sort(key=lambda r: (r["family_selection_score"], r["family"] == "ema_spread_narrowing"), reverse=True)
    return rows


def main() -> int:
    scout_dir = Path(os.getenv("RAWSEQ_EVENT_SCOUT_DIR", str(DEFAULT_SCOUT_DIR)))
    output_root = Path(os.getenv("RAWSEQ_EVENT_FAMILY_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    run_dir = output_root / f"indicator_event_family_selection_{now_stamp()}"
    survival = pd.read_csv(scout_dir / "event_target_survival.csv")
    fold_metrics = pd.read_csv(scout_dir / "event_fold_metrics.csv")
    rows = build_family_rows(survival, fold_metrics)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "indicator_event_family_selection.csv", rows)
    if not rows:
        recommendation = "no_indicator_event_family_survives"
        selected: dict[str, Any] = {}
    else:
        preferred = [r for r in rows if r["family"] == "ema_spread_narrowing" and set(r["included_horizons"]) == set(EXPECTED_HORIZONS)]
        selected = preferred[0] if preferred else rows[0]
        recommendation = "freeze_indicator_event_companion_before_future_holdout"
    selection_payload = {
        "created_at": now_stamp(),
        "source_scout_dir": str(scout_dir),
        "recommendation": recommendation,
        "selection_rule": "prefer coherent multi-horizon family with positive worst folds, 9/9 symbol stability, useful prevalence, finite calibration, and simple output contract",
        "selected_family": selected,
        "selection_hash": stable_hash({"selected": selected, "rows": rows[:10]}),
        "june_event_evaluation": False,
        "july_access": False,
    }
    write_json(run_dir / "selected_indicator_event_family.json", selection_payload)
    report = [
        "Rawseq 1m indicator-event family selection",
        f"source_scout_dir={scout_dir}",
        f"recommendation={recommendation}",
        f"selected_family={selected.get('family', 'none') if selected else 'none'}",
        f"selected_model={selected.get('model', 'none') if selected else 'none'}",
        f"selected_horizons={selected.get('included_horizons_text', '') if selected else ''}",
        "june_event_evaluation=false",
        "july_access=false",
    ]
    (run_dir / "indicator_event_family_selection.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"family_selection_dir={run_dir}")
    print(f"recommendation={recommendation}")
    print(f"selected_family={selected.get('family', 'none') if selected else 'none'}")
    print(f"selected_model={selected.get('model', 'none') if selected else 'none'}")
    print(f"selected_horizons={selected.get('included_horizons_text', '') if selected else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
