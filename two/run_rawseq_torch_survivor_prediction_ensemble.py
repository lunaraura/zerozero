#!/usr/bin/env python3
"""Average predictions from survivor rawseq torch sequence models.

This is the first ensemble stage after GPU survivors exist. It only combines
models that already passed the torch benchmark survivor gate, aligns by the
same exported sequence dataset contract, and reuses the same baseline and
policy metrics as the torch benchmark.

Safety: no training, no private API, no orders, no promotion, and no champion
mutation.
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_rawseq_torch_sequence_benchmark import (
    apply_baseline_status_to_summary,
    apply_policy_status_to_summary,
    build_baseline_reference,
    build_policy_metrics,
    compare_horizon_metrics_to_baselines,
    parse_float_list,
    safe_float,
    select_validation_policy_metrics,
    split_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SURVIVOR_STATUSES = {"single_horizon_holdout_baseline_survivor", "multi_horizon_holdout_baseline_survivor"}


def resolve_path(value: str | Path, default: Path | None = None) -> Path:
    path = Path(value or default or ".").expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def latest_torch_run() -> Path:
    roots = [
        Path("F:/rsio/rawseq_torch_sequence_benchmark_gpu_full_10epoch_5models_predictions"),
        Path("F:/rsio/rawseq_torch_sequence_benchmark_gpu_full_10epoch_5models"),
        PROJECT_ROOT / "data" / "research" / "rawseq_torch_sequence_benchmarks",
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("**/torch_sequence_model_summary.csv"))
    if not candidates:
        raise SystemExit("RAWSEQ_TORCH_ENSEMBLE_RUN_DIR is required; no torch_sequence_model_summary.csv found")
    return sorted([path.parent for path in candidates], key=lambda path: path.stat().st_mtime, reverse=True)[0]


def load_prediction(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "prediction": data["prediction"].astype(np.float64),
            "actual": data["actual"].astype(np.float64),
            "splits": data["splits"].astype(str),
            "target_columns": data["target_columns"].astype(str).tolist(),
            "horizon_buckets": data["horizon_buckets"].astype(int).tolist(),
        }


def load_dataset_timestamps(dataset_path: Path, rows: int) -> np.ndarray:
    if dataset_path.exists():
        with np.load(dataset_path, allow_pickle=False) as data:
            if "decision_timestamps" in data.files:
                return data["decision_timestamps"].astype(np.float64)
    return np.arange(rows, dtype=np.float64) * 10_000.0


def survivor_candidates(summary: pd.DataFrame, min_models: int) -> pd.DataFrame:
    required_cols = {"prediction_path", "dataset_path", "sequence_model_status", "feature_group", "seq_len", "model_kind"}
    if summary.empty or not required_cols.issubset(set(summary.columns)):
        return pd.DataFrame()
    frame = summary[
        summary["sequence_model_status"].astype(str).isin(SURVIVOR_STATUSES)
        & summary["prediction_path"].astype(str).ne("")
        & summary["prediction_path"].map(lambda path: Path(str(path)).exists())
    ].copy()
    if frame.empty:
        return frame
    group_cols = ["dataset_index", "feature_group", "seq_len"]
    counts = frame.groupby(group_cols, dropna=False)["model_kind"].transform("nunique")
    return frame[counts >= min_models].copy()


def evaluate_ensembles(
    candidates: pd.DataFrame,
    baseline_reference: pd.DataFrame,
    thresholds: list[float],
    costs: list[float],
    decision_cost_bps: float,
    policy: str,
    min_baseline_improvement: float,
    min_holdout_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    policy_frames: list[pd.DataFrame] = []
    group_cols = ["dataset_index", "feature_group", "seq_len"]
    for key, group in candidates.groupby(group_cols, dropna=False):
        group = group.sort_values("model_kind")
        loaded = [load_prediction(Path(str(path))) for path in group["prediction_path"]]
        shapes = {item["prediction"].shape for item in loaded}
        if len(shapes) != 1:
            continue
        prediction_stack = np.stack([item["prediction"] for item in loaded], axis=0)
        ensemble_pred = np.nanmean(prediction_stack, axis=0)
        actual = loaded[0]["actual"]
        splits = loaded[0]["splits"]
        horizon_buckets = loaded[0]["horizon_buckets"]
        first = group.iloc[0]
        timestamps = load_dataset_timestamps(Path(str(first.get("dataset_path", ""))), rows=ensemble_pred.shape[0])
        finite_timestamps = timestamps[np.isfinite(timestamps)]
        if len(finite_timestamps) >= 2:
            diffs = np.diff(np.sort(finite_timestamps))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            bucket_seconds = float(np.median(diffs) / 1000.0) if len(diffs) else 10.0
        else:
            bucket_seconds = 10.0
        dataset_meta = {
            "dataset_index": first.get("dataset_index"),
            "feature_group": first.get("feature_group"),
            "seq_len": int(safe_float(first.get("seq_len"), -1)),
            "feature_count": first.get("feature_count", math.nan),
            "horizon_count": ensemble_pred.shape[1],
            "bucket_seconds": bucket_seconds,
            "dataset_path": first.get("dataset_path", ""),
            "ensemble_model_count": int(group["model_kind"].nunique()),
            "ensemble_model_kinds": ",".join(group["model_kind"].astype(str).tolist()),
        }
        model_kind = "survivor_average"
        split_summary, split_horizon = split_metrics(actual, ensemble_pred, splits, horizon_buckets, dataset_meta, model_kind)
        by_split = {row["split"]: row for row in split_summary}
        summary_rows.append(
            {
                **dataset_meta,
                "model_kind": model_kind,
                "status": "ok",
                "sequence_model_status": "ensemble_pending_baseline_guard",
                "validation_combined_rmse": by_split.get("validation", {}).get("combined_rmse", math.nan),
                "holdout_combined_rmse": by_split.get("untouched_holdout", {}).get("combined_rmse", math.nan),
                "holdout_combined_correlation": by_split.get("untouched_holdout", {}).get("combined_correlation", math.nan),
            }
        )
        horizon_rows.extend(split_horizon)
        policy_frames.append(
            build_policy_metrics(
                actual,
                ensemble_pred,
                splits,
                timestamps,
                horizon_buckets,
                dataset_meta,
                model_kind,
                thresholds,
                costs,
                policy,
                bucket_seconds,
            )
        )
    summary_df = pd.DataFrame(summary_rows)
    horizon_df = pd.DataFrame(horizon_rows)
    policy_df = pd.concat(policy_frames, ignore_index=True) if policy_frames else pd.DataFrame()
    selected_policy_df = select_validation_policy_metrics(policy_df, decision_cost_bps)
    baseline_comparison_df = compare_horizon_metrics_to_baselines(
        horizon_df,
        baseline_reference,
        min_improvement_fraction=min_baseline_improvement,
        min_holdout_rows=min_holdout_rows,
    )
    summary_df = apply_baseline_status_to_summary(summary_df, baseline_comparison_df)
    summary_df = apply_policy_status_to_summary(summary_df, selected_policy_df)
    return summary_df, horizon_df, policy_df, selected_policy_df, baseline_comparison_df


def write_report(path: Path, contract: dict[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        "# Rawseq Torch Survivor Prediction Ensemble",
        "",
        f"Created at: {contract['created_at']}",
        f"Torch run: {contract['torch_run_dir']}",
        f"Candidate survivor rows: {contract['candidate_survivor_rows']}",
        f"Ensemble rows: {contract['ensemble_rows']}",
        "",
        "## Top Ensembles",
    ]
    if summary.empty:
        lines.append("No eligible survivor ensembles were produced.")
    else:
        cols = [
            col
            for col in [
                "feature_group",
                "seq_len",
                "ensemble_model_count",
                "ensemble_model_kinds",
                "sequence_model_status",
                "holdout_horizons_beating_best_baseline",
                "policy_horizons_holdout_positive",
                "best_holdout_position_cum_net_bps",
                "holdout_combined_rmse",
                "holdout_combined_correlation",
            ]
            if col in summary.columns
        ]
        lines.append(summary[cols].head(40).to_string(index=False))
    lines += [
        "",
        "## Safety",
        "- no_training=true",
        "- private_api=false",
        "- orders=false",
        "- promotion=false",
        "- champion_mutation=false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    run_env = os.getenv("RAWSEQ_TORCH_ENSEMBLE_RUN_DIR", "").strip()
    torch_run_dir = resolve_path(run_env) if run_env else latest_torch_run()
    output_root = resolve_path(
        os.getenv("RAWSEQ_TORCH_ENSEMBLE_OUTPUT_DIR", "F:/rsio/rawseq_torch_survivor_ensembles")
    )
    output_dir = output_root / f"torch_survivor_ensemble_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    min_models = int(float(os.getenv("RAWSEQ_TORCH_ENSEMBLE_MIN_MODELS", "2")))
    thresholds = parse_float_list("RAWSEQ_TORCH_ENSEMBLE_THRESHOLDS_BPS", "0,0.1,0.25,0.5,1,2")
    costs = parse_float_list("RAWSEQ_TORCH_ENSEMBLE_COSTS_BPS", "0.1,1,5")
    decision_cost_bps = float(os.getenv("RAWSEQ_TORCH_ENSEMBLE_DECISION_COST_BPS", str(costs[0] if costs else 0.1)))
    policy = os.getenv("RAWSEQ_TORCH_ENSEMBLE_POLICY", "direct_gt").strip().lower() or "direct_gt"
    min_baseline_improvement = float(os.getenv("RAWSEQ_TORCH_ENSEMBLE_MIN_BASELINE_IMPROVEMENT_FRACTION", "0.0"))
    min_holdout_rows = int(float(os.getenv("RAWSEQ_TORCH_ENSEMBLE_MIN_HOLDOUT_ROWS", "30")))

    summary = pd.read_csv(torch_run_dir / "torch_sequence_model_summary.csv")
    baseline_reference = pd.read_csv(torch_run_dir / "torch_sequence_baseline_reference.csv")
    candidates = survivor_candidates(summary, min_models=min_models)
    ensemble_summary, horizon, policy_metrics, selected_policy, baseline_comparison = evaluate_ensembles(
        candidates,
        baseline_reference,
        thresholds,
        costs,
        decision_cost_bps,
        policy,
        min_baseline_improvement,
        min_holdout_rows,
    )
    if not ensemble_summary.empty:
        ensemble_summary = ensemble_summary.sort_values(
            ["policy_horizons_holdout_positive", "holdout_horizons_beating_best_baseline", "best_holdout_position_cum_net_bps"],
            ascending=[False, False, False],
            na_position="last",
        )
    contract = {
        "created_at": datetime.now(UTC).isoformat(),
        "torch_run_dir": str(torch_run_dir),
        "output_dir": str(output_dir),
        "candidate_survivor_rows": int(len(candidates)),
        "ensemble_rows": int(len(ensemble_summary)),
        "min_models": min_models,
        "policy": policy,
        "thresholds_bps": thresholds,
        "costs_bps": costs,
        "decision_cost_bps": decision_cost_bps,
        "paper_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    ensemble_summary.to_csv(output_dir / "torch_survivor_ensemble_summary.csv", index=False)
    horizon.to_csv(output_dir / "torch_survivor_ensemble_per_horizon_metrics.csv", index=False)
    policy_metrics.to_csv(output_dir / "torch_survivor_ensemble_policy_metrics.csv", index=False)
    selected_policy.to_csv(output_dir / "torch_survivor_ensemble_selected_policy_metrics.csv", index=False)
    baseline_comparison.to_csv(output_dir / "torch_survivor_ensemble_baseline_comparison.csv", index=False)
    (output_dir / "torch_survivor_ensemble_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "torch_survivor_ensemble_report.txt", contract, ensemble_summary)
    print("Rawseq torch survivor ensemble complete")
    print(f"Output dir: {output_dir}")
    print(f"Candidate survivor rows: {len(candidates)}")
    print(f"Ensemble rows: {len(ensemble_summary)}")
    print("Safety: no_training=true private_api=false orders=false promotion=false champion_mutation=false")
    if not ensemble_summary.empty:
        cols = [
            col
            for col in [
                "feature_group",
                "seq_len",
                "ensemble_model_count",
                "ensemble_model_kinds",
                "sequence_model_status",
                "policy_horizons_holdout_positive",
                "best_holdout_position_cum_net_bps",
            ]
            if col in ensemble_summary.columns
        ]
        print(ensemble_summary[cols].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
