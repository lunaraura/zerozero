#!/usr/bin/env python3
"""Report whether GRU screen survivors repeat across seeds by contract."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRU_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_gru_locked_bundle_screen"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_gru_contract_survivors"
BASELINE_BASE_MODELS = {
    "zero_return_baseline",
    "training_mean_return_baseline",
    "rolling_mean_momentum_baseline",
    "mean_reversion_baseline",
    "ridge_multi_output",
    "elastic_net_multi_output",
    "logistic_direction_model",
    "small_tree_baseline",
    "boosted_tree_baseline",
}


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def latest_dir(root: Path, pattern: str) -> Path | None:
    paths = [path for path in root.glob(pattern) if path.is_dir()] if root.exists() else []
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def git_status_dirty() -> bool:
    try:
        return bool(subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=10).stdout.strip())
    except Exception:
        return True


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def bool_cell(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def infer_ablation_dir(benchmark_dir: Path) -> Path | None:
    contract_path = benchmark_dir / "torch_sequence_benchmark_contract.json"
    if not contract_path.exists():
        return None
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest_path = Path(str(contract.get("sequence_dataset_manifest", "")))
    dataset_contract = manifest_path.parent / "locked_bundle_sequence_dataset_contract.json"
    if not dataset_contract.exists():
        return None
    dataset_payload = json.loads(dataset_contract.read_text(encoding="utf-8"))
    ablation_dir = Path(str(dataset_payload.get("ablation_dir", "")))
    return ablation_dir if ablation_dir.exists() else None


def validation_baseline_reference(ablation_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(ablation_dir / "ablation_metrics.csv")
    frame = metrics[metrics["base_model"].astype(str).isin(BASELINE_BASE_MODELS)].copy()
    rows: list[dict[str, Any]] = []
    for horizon, group in frame.groupby("horizon_buckets", dropna=False):
        group = group.copy()
        group["_validation_rmse"] = pd.to_numeric(group["validation_rmse"], errors="coerce")
        finite = group[np.isfinite(group["_validation_rmse"])].copy()
        if finite.empty:
            rows.append(
                {
                    "horizon_buckets": int(float(horizon)),
                    "best_validation_baseline_model": "",
                    "best_validation_baseline_rmse": math.nan,
                    "validation_baseline_status": "missing_finite_validation_baseline",
                }
            )
            continue
        best = finite.sort_values(["_validation_rmse", "model"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "horizon_buckets": int(float(horizon)),
                "best_validation_baseline_model": best.get("model", ""),
                "best_validation_baseline_rmse": safe_float(best.get("validation_rmse")),
                "validation_baseline_status": "ok",
            }
        )
    return pd.DataFrame(rows)


def same_finite_sign(values: pd.Series) -> bool:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return False
    signs = np.sign(vals)
    signs = signs[signs != 0]
    return bool(len(signs) > 0 and (np.all(signs > 0) or np.all(signs < 0)))


def compact_list(values: pd.Series) -> str:
    out = []
    for value in values:
        text = str(value)
        if text and text.lower() != "nan" and text not in out:
            out.append(text)
    return ";".join(out)


def load_benchmark_frames(benchmark_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(benchmark_dir / "torch_sequence_model_summary.csv"),
        pd.read_csv(benchmark_dir / "torch_sequence_per_horizon_metrics.csv"),
        pd.read_csv(benchmark_dir / "torch_sequence_baseline_comparison.csv"),
        pd.read_csv(benchmark_dir / "torch_sequence_selected_policy_metrics.csv"),
    )


def build_detail_and_rollup(
    summary: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    holdout_baseline: pd.DataFrame,
    selected_policy: pd.DataFrame,
    validation_baseline: pd.DataFrame,
    min_seed_passes: int,
    min_mean_improvement: float,
    min_worst_improvement: float,
    min_validation_policy_trades: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = horizon_metrics[horizon_metrics["split"].astype(str).eq("validation")].copy()
    holdout = horizon_metrics[horizon_metrics["split"].astype(str).eq("untouched_holdout")].copy()
    key_cols = ["dataset_index", "feature_group", "seq_len", "model_kind", "seed", "horizon_buckets"]
    validation = validation.merge(validation_baseline, on="horizon_buckets", how="left")
    validation["validation_rmse_improvement_vs_best_baseline"] = (
        pd.to_numeric(validation["best_validation_baseline_rmse"], errors="coerce")
        - pd.to_numeric(validation["rmse"], errors="coerce")
    ) / pd.to_numeric(validation["best_validation_baseline_rmse"], errors="coerce")
    validation["validation_baseline_guard_pass"] = validation["validation_rmse_improvement_vs_best_baseline"] > 0.0
    holdout_cols = key_cols + [
        "rmse",
        "correlation",
        "directional_accuracy",
    ]
    holdout = holdout.reindex(columns=holdout_cols).rename(
        columns={
            "rmse": "holdout_rmse",
            "correlation": "holdout_correlation",
            "directional_accuracy": "holdout_directional_accuracy",
        }
    )
    validation = validation.rename(
        columns={
            "rmse": "validation_rmse",
            "correlation": "validation_correlation",
            "directional_accuracy": "validation_directional_accuracy",
        }
    )
    detail = validation.merge(holdout, on=key_cols, how="left")
    holdout_flags = holdout_baseline.reindex(columns=key_cols + ["baseline_guard_pass", "rmse_improvement_fraction_vs_best_baseline"]).rename(
        columns={
            "baseline_guard_pass": "holdout_baseline_guard_pass_reporting_only",
            "rmse_improvement_fraction_vs_best_baseline": "holdout_rmse_improvement_vs_best_baseline_reporting_only",
        }
    )
    detail = detail.merge(holdout_flags, on=key_cols, how="left")
    policy_cols = key_cols + [
        "validation_position_trade_count",
        "validation_position_cum_net_bps",
        "holdout_position_trade_count",
        "holdout_position_cum_net_bps",
        "selected_threshold_bps",
    ]
    detail = detail.merge(selected_policy.reindex(columns=policy_cols), on=key_cols, how="left")
    summary_cols = [
        "dataset_index",
        "feature_group",
        "seq_len",
        "model_kind",
        "seed",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_save_status",
        "checkpoint_roundtrip_status",
        "checkpoint_roundtrip_max_abs_diff",
        "model_constructor_config",
        "feature_columns_sha256",
        "target_columns_sha256",
        "feature_scaler_state_present",
        "target_scaler_state_present",
        "status",
        "failure",
    ]
    detail = detail.merge(summary.reindex(columns=summary_cols), on=["dataset_index", "feature_group", "seq_len", "model_kind", "seed"], how="left")
    detail["checkpoint_path_exists"] = detail["checkpoint_path"].apply(lambda value: Path(str(value)).exists() if str(value) and str(value).lower() != "nan" else False)
    detail["checkpoint_load_status"] = detail.get("checkpoint_roundtrip_status", "").astype(str)
    detail["schema_or_checkpoint_failure"] = ~(
        detail["status"].astype(str).eq("ok")
        & detail["checkpoint_save_status"].astype(str).eq("saved")
        & detail["checkpoint_roundtrip_status"].astype(str).eq("ok")
        & detail["checkpoint_path_exists"].astype(bool)
    )

    rows: list[dict[str, Any]] = []
    for (feature_group, seq_len, horizon), group in detail.groupby(["feature_group", "seq_len", "horizon_buckets"], dropna=False):
        seeds = sorted(pd.to_numeric(group["seed"], errors="coerce").dropna().astype(int).unique().tolist())
        val_pass = group["validation_baseline_guard_pass"].astype(bool)
        holdout_pass = group["holdout_baseline_guard_pass_reporting_only"].apply(bool_cell)
        val_improvement = pd.to_numeric(group["validation_rmse_improvement_vs_best_baseline"], errors="coerce")
        validation_policy_trades = pd.to_numeric(group["validation_position_trade_count"], errors="coerce").fillna(0)
        all_checkpoint_ok = bool((~group["schema_or_checkpoint_failure"].astype(bool)).all())
        validation_corr_same_sign = same_finite_sign(group["validation_correlation"])
        status = "reject"
        reasons = []
        if int(val_pass.sum()) < min_seed_passes:
            reasons.append(f"validation_seed_pass_count<{min_seed_passes}")
        if not (safe_float(val_improvement.mean()) > min_mean_improvement):
            reasons.append(f"mean_validation_improvement<={min_mean_improvement}")
        if not (safe_float(val_improvement.min()) >= min_worst_improvement):
            reasons.append(f"worst_seed_validation_improvement<{min_worst_improvement}")
        if not validation_corr_same_sign:
            reasons.append("validation_correlation_sign_not_stable")
        if int(validation_policy_trades.sum()) < min_validation_policy_trades:
            reasons.append(f"validation_policy_trade_count<{min_validation_policy_trades}")
        if not all_checkpoint_ok:
            reasons.append("checkpoint_or_schema_not_freezeable")
        predictive_reasons = [
            reason
            for reason in reasons
            if reason not in {"checkpoint_or_schema_not_freezeable"} and not reason.startswith("validation_policy_trade_count")
        ]
        if not predictive_reasons:
            status = "rerun_candidate"
        elif int(val_pass.sum()) >= min_seed_passes and safe_float(val_improvement.mean()) > min_mean_improvement:
            status = "research_watch"
        rows.append(
            {
                "feature_bundle": feature_group,
                "seq_len": int(float(seq_len)),
                "horizon_buckets": int(float(horizon)),
                "seeds_tested": len(seeds),
                "seeds": ",".join(str(seed) for seed in seeds),
                "validation_seed_pass_count": int(val_pass.sum()),
                "holdout_seed_pass_count_reporting_only": int(holdout_pass.sum()),
                "mean_validation_improvement": safe_float(val_improvement.mean()),
                "worst_seed_validation_improvement": safe_float(val_improvement.min()),
                "best_seed_validation_improvement": safe_float(val_improvement.max()),
                "validation_correlation_mean": safe_float(pd.to_numeric(group["validation_correlation"], errors="coerce").mean()),
                "validation_correlation_min": safe_float(pd.to_numeric(group["validation_correlation"], errors="coerce").min()),
                "validation_correlation_max": safe_float(pd.to_numeric(group["validation_correlation"], errors="coerce").max()),
                "validation_correlation_same_sign": validation_corr_same_sign,
                "holdout_correlation_mean_reporting_only": safe_float(pd.to_numeric(group["holdout_correlation"], errors="coerce").mean()),
                "validation_policy_trade_count": int(validation_policy_trades.sum()),
                "holdout_policy_trade_count_reporting_only": int(pd.to_numeric(group["holdout_position_trade_count"], errors="coerce").fillna(0).sum()),
                "checkpoint_paths": compact_list(group["checkpoint_path"]),
                "checkpoint_sha256s": compact_list(group["checkpoint_sha256"]),
                "checkpoint_load_status": compact_list(group["checkpoint_load_status"]),
                "checkpoint_roundtrip_max_abs_diff_max": safe_float(pd.to_numeric(group["checkpoint_roundtrip_max_abs_diff"], errors="coerce").max()),
                "all_checkpoints_roundtrip_ok": all_checkpoint_ok,
                "model_constructor_configs": compact_list(group["model_constructor_config"]),
                "feature_columns_sha256": compact_list(group["feature_columns_sha256"]),
                "target_columns_sha256": compact_list(group["target_columns_sha256"]),
                "selection_stage": "validation_selected_contract",
                "holdout_role": "reporting_only",
                "survivor_status": status,
                "rejection_reasons": ";".join(reasons),
            }
        )
    rollup = pd.DataFrame(rows)
    if not rollup.empty:
        status_priority = {"rerun_candidate": 0, "research_watch": 1, "reject": 2}
        rollup["_status_priority"] = rollup["survivor_status"].map(status_priority).fillna(9)
        rollup = rollup.sort_values(
            [
                "_status_priority",
                "validation_seed_pass_count",
                "mean_validation_improvement",
                "worst_seed_validation_improvement",
                "validation_policy_trade_count",
                "feature_bundle",
                "seq_len",
                "horizon_buckets",
            ],
            ascending=[True, False, False, False, False, True, True, True],
        ).drop(columns=["_status_priority"])
    return detail, rollup


def main() -> int:
    benchmark_env = os.getenv("RAWSEQ_GRU_SURVIVOR_BENCHMARK_DIR", "").strip()
    benchmark_dir = resolve_path(benchmark_env) if benchmark_env else latest_dir(DEFAULT_GRU_ROOT, "torch_sequence_benchmark_*")
    if benchmark_dir is None:
        raise SystemExit("Could not find GRU benchmark directory.")
    ablation_env = os.getenv("RAWSEQ_GRU_SURVIVOR_ABLATION_DIR", "").strip()
    ablation_dir = resolve_path(ablation_env) if ablation_env else infer_ablation_dir(benchmark_dir)
    if ablation_dir is None:
        raise SystemExit("Could not infer ablation directory.")
    output_root = resolve_path(os.getenv("RAWSEQ_GRU_SURVIVOR_OUTPUT_DIR", str(DEFAULT_OUTPUT_ROOT)))
    out_dir = output_root / f"rawseq_gru_contract_survivors_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    min_seed_passes = int(float(os.getenv("RAWSEQ_GRU_SURVIVOR_MIN_SEED_PASSES", "2")))
    min_mean_improvement = float(os.getenv("RAWSEQ_GRU_SURVIVOR_MIN_MEAN_VALIDATION_IMPROVEMENT", "0.0"))
    min_worst_improvement = float(os.getenv("RAWSEQ_GRU_SURVIVOR_MIN_WORST_VALIDATION_IMPROVEMENT", "-0.05"))
    min_validation_policy_trades = int(float(os.getenv("RAWSEQ_GRU_SURVIVOR_MIN_VALIDATION_POLICY_TRADES", "0")))

    summary, horizon_metrics, holdout_baseline, selected_policy = load_benchmark_frames(benchmark_dir)
    validation_baseline = validation_baseline_reference(ablation_dir)
    detail, rollup = build_detail_and_rollup(
        summary,
        horizon_metrics,
        holdout_baseline,
        selected_policy,
        validation_baseline,
        min_seed_passes,
        min_mean_improvement,
        min_worst_improvement,
        min_validation_policy_trades,
    )
    detail_path = out_dir / "gru_contract_seed_detail.csv"
    rollup_path = out_dir / "gru_contract_survivor_rollup.csv"
    selected_path = out_dir / "gru_contract_rerun_candidates.csv"
    detail.to_csv(detail_path, index=False)
    rollup.to_csv(rollup_path, index=False)
    rollup[rollup["survivor_status"].astype(str).eq("rerun_candidate")].to_csv(selected_path, index=False)
    contract = {
        "created_at": now_iso(),
        "generator_path": "scripts/tiny/report_rawseq_gru_contract_survivors.py",
        "git_head": git_head(),
        "git_status_dirty": git_status_dirty(),
        "benchmark_dir": str(benchmark_dir),
        "ablation_dir": str(ablation_dir),
        "output_dir": str(out_dir),
        "min_seed_passes": min_seed_passes,
        "min_mean_validation_improvement": min_mean_improvement,
        "min_worst_validation_improvement": min_worst_improvement,
        "min_validation_policy_trades": min_validation_policy_trades,
        "selection_stage": "validation_only_contract_selection",
        "holdout_role": "reporting_only",
        "paper_only": True,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    (out_dir / "gru_contract_survivor_report_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidates = rollup[rollup["survivor_status"].astype(str).eq("rerun_candidate")].copy()
    lines = [
        "# Rawseq GRU Contract Survivor Report",
        "",
        f"Created at: {contract['created_at']}",
        f"Benchmark: `{benchmark_dir}`",
        f"Ablation: `{ablation_dir}`",
        f"Output: `{out_dir}`",
        "",
        "Selection is validation-only. Holdout columns are reporting-only.",
        "",
        "## Counts",
        f"- seed-detail rows: {len(detail)}",
        f"- contract rows: {len(rollup)}",
        f"- rerun candidates: {len(candidates)}",
        "",
        "## Rerun Candidates",
    ]
    if candidates.empty:
        lines.append("No contract survived the validation seed-agreement filter.")
    else:
        cols = [
            "feature_bundle",
            "seq_len",
            "horizon_buckets",
            "seeds_tested",
            "validation_seed_pass_count",
            "mean_validation_improvement",
            "worst_seed_validation_improvement",
            "validation_correlation_same_sign",
            "validation_policy_trade_count",
            "all_checkpoints_roundtrip_ok",
            "holdout_seed_pass_count_reporting_only",
        ]
        lines.append(candidates[cols].head(20).to_string(index=False))
    lines += [
        "",
        "## Top Contracts",
    ]
    if rollup.empty:
        lines.append("No contracts were evaluated.")
    else:
        cols = [
            "feature_bundle",
            "seq_len",
            "horizon_buckets",
            "survivor_status",
            "validation_seed_pass_count",
            "mean_validation_improvement",
            "worst_seed_validation_improvement",
            "validation_correlation_same_sign",
            "all_checkpoints_roundtrip_ok",
            "rejection_reasons",
        ]
        lines.append(rollup[cols].head(30).to_string(index=False))
    lines += [
        "",
        "Safety: paper-only report; no training, no promotion, no champion mutation, no orders.",
    ]
    (out_dir / "gru_contract_survivor_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Rawseq GRU contract survivor report complete")
    print(f"Output: {out_dir}")
    print(f"Rollup: {rollup_path}")
    print(f"Rerun candidates: {len(candidates)}")
    if not candidates.empty:
        print(candidates[["feature_bundle", "seq_len", "horizon_buckets", "validation_seed_pass_count", "mean_validation_improvement", "worst_seed_validation_improvement"]].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
