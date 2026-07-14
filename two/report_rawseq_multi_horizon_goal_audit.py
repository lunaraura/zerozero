#!/usr/bin/env python3
"""Report-only audit for the rawseq multi-horizon indicator/sequence goal.

The audit reads a multi-horizon indicator run and an optional torch sequence
benchmark run, then classifies evidence against the current research plan:
causal indicator bank, boring baselines, sequence datasets, GPU sequence
models, per-horizon/policy metrics, ablations, and the "no ensemble before
survivors" gate.

Safety: report only; no training, no private API, no orders, no promotion, and
no champion mutation.
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_multi_horizon_goal_audits"
REQUIRED_WINDOWS = [3, 5, 10, 20, 30, 60, 120, 240]
REQUIRED_HORIZONS = [1, 3, 6, 12, 24, 48]
REQUIRED_SEQUENCE_LENS = [60, 120, 240]
REQUIRED_FEATURE_FAMILIES = ["raw", "trend", "momentum", "volatility", "breakout", "volume", "regime", "cross_market"]
REQUIRED_BASELINES = [
    "zero_return_baseline",
    "training_mean_return_baseline",
    "rolling_mean_momentum_baseline",
    "mean_reversion_baseline",
    "ridge_multi_output",
    "elastic_net_multi_output",
    "logistic_direction_model",
    "small_tree_baseline",
    "boosted_tree_baseline",
]
REQUIRED_TORCH_MODELS = ["torch_tcn_sequence_model", "torch_gru_sequence_model", "torch_lstm_sequence_model", "torch_transformer_sequence_model"]
REQUIRED_TORCH_BENCHMARK_MODELS = ["mlp", "tcn", "gru", "lstm", "transformer"]


def resolve_path(value: str | Path, default: Path | None = None) -> Path:
    path = Path(value or default or ".").expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def latest_dir_with(marker: str, roots: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend([path.parent for path in root.glob(f"**/{marker}")])
    if not candidates:
        return None
    return sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)[0]


def default_search_roots(kind: str) -> list[Path]:
    roots = []
    if kind == "indicator":
        roots.extend(
            [
                PROJECT_ROOT / "data" / "research" / "rawseq_multi_horizon_indicator_returns",
                Path("F:/rsio/rawseq_multi_horizon_indicator_seq_export_smoke"),
                Path("F:/rsio/rawseq_multi_horizon_indicator_default_seq_smoke"),
            ]
        )
    else:
        roots.extend(
            [
                PROJECT_ROOT / "data" / "research" / "rawseq_torch_sequence_benchmarks",
                Path("F:/rsio/rawseq_torch_sequence_benchmark_smoke"),
            ]
        )
    return roots


def infer_indicator_run_dir() -> Path | None:
    env = os.getenv("RAWSEQ_GOAL_AUDIT_INDICATOR_RUN_DIR", "").strip()
    if env:
        return resolve_path(env)
    return latest_dir_with("multi_horizon_indicator_report.txt", default_search_roots("indicator"))


def infer_torch_run_dir() -> Path | None:
    env = os.getenv("RAWSEQ_GOAL_AUDIT_TORCH_RUN_DIR", "").strip()
    if env:
        return resolve_path(env)
    return latest_dir_with("torch_sequence_benchmark_report.txt", default_search_roots("torch"))


def infer_ensemble_run_dir() -> Path | None:
    env = os.getenv("RAWSEQ_GOAL_AUDIT_ENSEMBLE_RUN_DIR", "").strip()
    if env:
        return resolve_path(env)
    return latest_dir_with(
        "torch_survivor_ensemble_report.txt",
        [
            Path("F:/rsio/rawseq_torch_survivor_ensembles"),
            PROJECT_ROOT / "data" / "research" / "rawseq_torch_survivor_ensembles",
        ],
    )


def requirement_row(
    category: str,
    requirement: str,
    status: str,
    evidence: str,
    action: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "recommended_action": action,
    }


def list_missing(required: list[Any], observed: list[Any]) -> list[Any]:
    observed_set = {str(item) for item in observed}
    return [item for item in required if str(item) not in observed_set]


def audit_feature_families(feature_family: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    counts = (
        feature_family["feature_family"].astype(str).value_counts().sort_index().to_dict()
        if not feature_family.empty and "feature_family" in feature_family.columns
        else {}
    )
    missing = [family for family in REQUIRED_FEATURE_FAMILIES if int(counts.get(family, 0)) <= 0]
    rows.append(
        requirement_row(
            "features",
            "causal feature bank includes required feature families",
            "pass" if not missing else "fail",
            f"family_counts={counts}; missing={missing}",
            "add missing causal feature families or run with cross-market sources" if missing else "",
        )
    )
    causal_ok = bool(
        not feature_family.empty
        and "causal_inputs_only" in feature_family.columns
        and feature_family["causal_inputs_only"].map(parse_bool).all()
    )
    rows.append(
        requirement_row(
            "features",
            "feature manifest marks features as causal/current-or-historical",
            "pass" if causal_ok else "fail",
            f"causal_inputs_only_all={causal_ok}",
            "regenerate feature manifest with causal_inputs_only=true and inspect rolling joins" if not causal_ok else "",
        )
    )
    return rows, {str(key): int(value) for key, value in counts.items()}


def audit_window_and_target_scope(feature_manifest: dict[str, Any], target_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    windows = [int(item) for item in feature_manifest.get("feature_windows", []) if str(item).strip()]
    missing_windows = list_missing(REQUIRED_WINDOWS, windows)
    rows.append(
        requirement_row(
            "scope",
            "feature windows cover 3,5,10,20,30,60,120,240",
            "pass" if not missing_windows else "partial",
            f"observed_windows={windows}; missing={missing_windows}",
            "run full-scope pipeline with RAWSEQ_MH_FEATURE_WINDOWS=3,5,10,20,30,60,120,240"
            if missing_windows
            else "",
        )
    )
    horizons = [int(item) for item in target_manifest.get("horizon_buckets", []) if str(item).strip()]
    missing_horizons = list_missing(REQUIRED_HORIZONS, horizons)
    rows.append(
        requirement_row(
            "scope",
            "return-vector horizons cover 1,3,6,12,24,48",
            "pass" if not missing_horizons else "partial",
            f"observed_horizons={horizons}; missing={missing_horizons}",
            "run full-scope pipeline with RAWSEQ_MH_HORIZON_BUCKETS=1,3,6,12,24,48"
            if missing_horizons
            else "",
        )
    )
    output_dim_ok = int(target_manifest.get("output_dim", 0) or 0) == len(horizons) and len(horizons) > 0
    rows.append(
        requirement_row(
            "targets",
            "target is multi-output future return vector",
            "pass" if output_dim_ok else "fail",
            f"target_family={target_manifest.get('target_family')}; output_dim={target_manifest.get('output_dim')}; horizons={horizons}",
            "regenerate target manifest with one output per horizon" if not output_dim_ok else "",
        )
    )
    return rows


def audit_splits(split_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    guards = split_manifest.get("guards", {}) if isinstance(split_manifest.get("guards"), dict) else {}
    guard_keys = [
        "feature_timestamp_max_lte_decision_timestamp",
        "label_end_timestamp_gt_decision_timestamp",
        "unique_symbol_venue_timestamp_keys",
        "sorted_chronological_order",
    ]
    failed = [key for key in guard_keys if not bool(guards.get(key))]
    rows = [
        requirement_row(
            "splits",
            "causality and chronological table guards pass",
            "pass" if not failed else "fail",
            f"failed_guards={failed}; guards={guards}",
            "fix source ordering, timestamps, or label construction" if failed else "",
        )
    ]
    has_holdout = str(split_manifest.get("holdout_usage", "")).lower().find("untouched") >= 0
    rows.append(
        requirement_row(
            "splits",
            "validation selects; untouched holdout evaluates",
            "pass" if has_holdout else "fail",
            f"model_selection_stage={split_manifest.get('model_selection_stage')}; holdout_usage={split_manifest.get('holdout_usage')}",
            "restore validation-selected/untouched-holdout split contract" if not has_holdout else "",
        )
    )
    return rows


def audit_models(model_manifest: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    base_models = set(model_manifest.get("base_model", pd.Series(dtype=str)).astype(str)) if not model_manifest.empty else set()
    missing_baselines = [model for model in REQUIRED_BASELINES if model not in base_models]
    rows.append(
        requirement_row(
            "baselines",
            "boring baselines are present before GPU models",
            "pass" if not missing_baselines else "fail",
            f"base_models={sorted(base_models)}; missing={missing_baselines}",
            "enable missing CPU baselines before treating GPU runs as meaningful" if missing_baselines else "",
        )
    )
    missing_torch = [model for model in REQUIRED_TORCH_MODELS if model not in base_models]
    torch_statuses = {}
    if not model_manifest.empty and "base_model" in model_manifest.columns and "status" in model_manifest.columns:
        torch_frame = model_manifest[model_manifest["base_model"].astype(str).isin(REQUIRED_TORCH_MODELS)]
        torch_statuses = torch_frame.groupby("base_model")["status"].apply(lambda x: sorted(set(x.astype(str)))).to_dict()
    rows.append(
        requirement_row(
            "gpu_models",
            "TCN/GRU/LSTM/Transformer model hooks are represented",
            "pass" if not missing_torch else "partial",
            f"torch_statuses={torch_statuses}; missing={missing_torch}",
            "request torch sequence models in the benchmark run" if missing_torch else "",
        )
    )
    return rows


def audit_ablation(ablation: pd.DataFrame) -> list[dict[str, Any]]:
    required = ["raw", "raw_trend", "raw_momentum", "raw_volatility", "raw_volume", "raw_cross_market", "all"]
    groups = sorted(set(ablation.get("feature_group", pd.Series(dtype=str)).astype(str))) if not ablation.empty else []
    missing = [group for group in required if group not in groups]
    return [
        requirement_row(
            "ablation",
            "indicator ablation groups are evaluated",
            "pass" if not missing else "partial",
            f"observed_groups={groups}; missing={missing}",
            "run with RAWSEQ_MH_ABLATION_GROUPS=raw,raw_trend,raw_momentum,raw_volatility,raw_volume,raw_cross_market,all"
            if missing
            else "",
        )
    ]


def audit_sequence_datasets(sequence_manifest: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if sequence_manifest.empty:
        return [
            requirement_row(
                "sequence_data",
                "sequence datasets exist for GPU handoff",
                "fail",
                "sequence_dataset_manifest.csv missing or empty",
                "rerun indicator pipeline with RAWSEQ_MH_WRITE_SEQUENCE_DATASETS=true",
            )
        ]
    seq_lens = sorted({int(safe_float(value, -1)) for value in sequence_manifest.get("seq_len", []) if safe_float(value, -1) > 0})
    missing = list_missing(REQUIRED_SEQUENCE_LENS, seq_lens)
    rows.append(
        requirement_row(
            "sequence_data",
            "sequence lengths include 60,120,240",
            "pass" if not missing else "partial",
            f"observed_seq_lens={seq_lens}; missing={missing}",
            "export full sequence datasets with RAWSEQ_MH_SEQUENCE_LENS=60,120,240 and RAWSEQ_MH_WRITE_SEQUENCE_DATASETS=true"
            if missing
            else "",
        )
    )
    arrays_written = bool(sequence_manifest.get("arrays_written", pd.Series(dtype=bool)).map(parse_bool).any())
    shape_ok = bool(
        not sequence_manifest.empty
        and sequence_manifest.get("sequence_input_shape", pd.Series(dtype=str)).astype(str).str.contains("batch, seq_len, feature_count", regex=False).any()
        and sequence_manifest.get("target_shape", pd.Series(dtype=str)).astype(str).str.contains("batch, horizon_count", regex=False).any()
    )
    rows.append(
        requirement_row(
            "sequence_data",
            "sequence tensors are exported as X=[batch,seq_len,feature_count], y=[batch,horizon_count]",
            "pass" if arrays_written and shape_ok else "partial",
            f"arrays_written_any={arrays_written}; shape_contract_ok={shape_ok}",
            "enable RAWSEQ_MH_WRITE_SEQUENCE_DATASETS=true for the full run" if not arrays_written else "",
        )
    )
    return rows


def audit_torch_run(torch_dir: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if torch_dir is None:
        return [
            requirement_row(
                "gpu_models",
                "torch benchmark run exists",
                "fail",
                "no torch_sequence_benchmark_report.txt found",
                "run scripts/tiny/run_rawseq_torch_sequence_benchmark.py on exported sequence datasets",
            )
        ], {}
    contract = read_json(torch_dir / "torch_sequence_benchmark_contract.json")
    summary = read_csv(torch_dir / "torch_sequence_model_summary.csv")
    horizon = read_csv(torch_dir / "torch_sequence_per_horizon_metrics.csv")
    policy = read_csv(torch_dir / "torch_sequence_policy_metrics.csv")
    selected_policy = read_csv(torch_dir / "torch_sequence_selected_policy_metrics.csv")
    baseline_comparison = read_csv(torch_dir / "torch_sequence_baseline_comparison.csv")
    torch_status = contract.get("torch_status", {})
    torch_available = bool(torch_status.get("torch_available"))
    cuda_available = bool(torch_status.get("cuda_available"))
    trained_ok = bool(not summary.empty and (summary.get("status", pd.Series(dtype=str)).astype(str) == "ok").any())
    trained_model_kinds = sorted(
        set(
            summary[
                summary.get("status", pd.Series(dtype=str)).astype(str).eq("ok")
            ].get("model_kind", pd.Series(dtype=str)).astype(str)
        )
    ) if not summary.empty and "model_kind" in summary.columns and "status" in summary.columns else []
    missing_benchmark_models = [model for model in REQUIRED_TORCH_BENCHMARK_MODELS if model not in trained_model_kinds]
    survivor_statuses = {
        "multi_horizon_holdout_baseline_survivor",
        "single_horizon_holdout_baseline_survivor",
    }
    survivor_rows = (
        summary[summary.get("sequence_model_status", pd.Series(dtype=str)).astype(str).isin(survivor_statuses)]
        if not summary.empty and "sequence_model_status" in summary.columns
        else pd.DataFrame()
    )
    policy_positive_rows = (
        summary[pd.to_numeric(summary.get("policy_horizons_holdout_positive", pd.Series(dtype=float)), errors="coerce") > 0]
        if not summary.empty
        else pd.DataFrame()
    )
    rows = [
        requirement_row(
            "gpu_models",
            "torch runtime is available for GPU/sequence training",
            "pass" if torch_available else "blocked_external_runtime",
            f"torch_status={torch_status}",
            "run the torch benchmark in a torch/CUDA-capable environment" if not torch_available else "",
        ),
        requirement_row(
            "gpu_models",
            "GPU sequence models have trained rows",
            "pass" if trained_ok else "blocked_external_runtime" if not torch_available else "fail",
            f"trained_ok={trained_ok}; summary_statuses={sorted(set(summary.get('status', pd.Series(dtype=str)).astype(str))) if not summary.empty else []}",
            "rerun on torch-capable environment; current local run only proves readiness" if not trained_ok else "",
        ),
        requirement_row(
            "gpu_models",
            "GPU benchmark includes MLP/TCN/GRU/LSTM/Transformer",
            "pass" if not missing_benchmark_models else "partial" if trained_ok else "fail",
            f"trained_model_kinds={trained_model_kinds}; missing={missing_benchmark_models}",
            "rerun RAWSEQ_TORCH_SEQUENCE_MODELS=mlp,tcn,gru,lstm,transformer" if missing_benchmark_models else "",
        ),
        requirement_row(
            "metrics",
            "GPU sequence run writes per-horizon RMSE/MAE/correlation/directional metrics",
            "pass" if not horizon.empty else "blocked_external_runtime" if not torch_available else "fail",
            f"horizon_metric_rows={len(horizon)}",
            "train at least one torch sequence model to populate per-horizon metrics" if horizon.empty else "",
        ),
        requirement_row(
            "metrics",
            "GPU sequence run writes policy, non-overlap, and position metrics",
            "pass" if not policy.empty and not selected_policy.empty else "blocked_external_runtime" if not torch_available else "fail",
            f"policy_rows={len(policy)}; selected_policy_rows={len(selected_policy)}",
            "train at least one torch sequence model to populate policy metrics" if policy.empty else "",
        ),
        requirement_row(
            "baselines",
            "GPU sequence model beats best boring baseline on untouched holdout",
            "pass" if not survivor_rows.empty else "not_yet",
            f"survivor_rows={len(survivor_rows)}; baseline_comparison_rows={len(baseline_comparison)}",
            "do not call any GPU sequence model a survivor until baseline guard passes" if survivor_rows.empty else "",
        ),
        requirement_row(
            "policy",
            "survivor also shows positive holdout policy metrics",
            "pass" if not policy_positive_rows.empty else "not_yet",
            f"policy_positive_summary_rows={len(policy_positive_rows)}",
            "require validation-selected threshold/cost policy to survive untouched holdout" if policy_positive_rows.empty else "",
        ),
    ]
    return rows, {
        "torch_dir": str(torch_dir),
        "torch_available": torch_available,
        "cuda_available": cuda_available,
        "trained_ok": trained_ok,
        "trained_model_kinds": trained_model_kinds,
        "missing_benchmark_models": missing_benchmark_models,
        "survivor_rows": int(len(survivor_rows)),
        "policy_positive_summary_rows": int(len(policy_positive_rows)),
    }


def audit_ensemble_run(ensemble_dir: Path | None, ensemble_allowed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if ensemble_dir is None:
        return [
            requirement_row(
                "ensemble",
                "survivor-only ensemble evaluated after survivor evidence",
                "not_yet" if ensemble_allowed else "blocked_until_survivors",
                "no torch_survivor_ensemble_report.txt found",
                "run survivor-only ensemble after the GPU audit reports survivor evidence" if ensemble_allowed else "",
            )
        ], {"ensemble_rows": 0, "candidate_survivor_rows": 0}
    contract = read_json(ensemble_dir / "torch_survivor_ensemble_contract.json")
    summary = read_csv(ensemble_dir / "torch_survivor_ensemble_summary.csv")
    ensemble_rows = int(len(summary))
    candidate_survivor_rows = int(contract.get("candidate_survivor_rows", 0) or 0)
    safety_ok = bool(
        contract.get("paper_only")
        and not contract.get("orders")
        and not contract.get("promotion")
        and not contract.get("champion_mutation")
    )
    positive_policy_rows = (
        summary[pd.to_numeric(summary.get("policy_horizons_holdout_positive", pd.Series(dtype=float)), errors="coerce") > 0]
        if not summary.empty
        else pd.DataFrame()
    )
    baseline_survivor_rows = (
        summary[summary.get("sequence_model_status", pd.Series(dtype=str)).astype(str).isin({"single_horizon_holdout_baseline_survivor", "multi_horizon_holdout_baseline_survivor"})]
        if not summary.empty
        else pd.DataFrame()
    )
    rows = [
        requirement_row(
            "ensemble",
            "survivor-only ensemble evaluated after survivor evidence",
            "pass" if ensemble_allowed and ensemble_rows > 0 and candidate_survivor_rows > 0 else "fail",
            f"ensemble_dir={ensemble_dir}; ensemble_rows={ensemble_rows}; candidate_survivor_rows={candidate_survivor_rows}",
            "only evaluate ensembles after survivor evidence exists" if not ensemble_allowed else "",
        ),
        requirement_row(
            "ensemble",
            "ensemble report keeps paper-only/no-promotion safety",
            "pass" if safety_ok else "fail",
            f"safety_ok={safety_ok}; contract={contract}",
            "fix ensemble contract safety flags" if not safety_ok else "",
        ),
        requirement_row(
            "ensemble",
            "ensemble has baseline and holdout policy survivor evidence",
            "pass" if len(baseline_survivor_rows) > 0 and len(positive_policy_rows) > 0 else "not_yet",
            f"baseline_survivor_rows={len(baseline_survivor_rows)}; positive_policy_rows={len(positive_policy_rows)}",
            "keep ensemble as research-only until baseline and holdout policy rows survive" if len(positive_policy_rows) == 0 else "",
        ),
    ]
    return rows, {
        "ensemble_dir": str(ensemble_dir),
        "ensemble_rows": ensemble_rows,
        "candidate_survivor_rows": candidate_survivor_rows,
        "positive_policy_rows": int(len(positive_policy_rows)),
        "baseline_survivor_rows": int(len(baseline_survivor_rows)),
    }


def classify_overall(rows: list[dict[str, Any]], torch_summary: dict[str, Any]) -> tuple[str, bool, str]:
    statuses = [row["status"] for row in rows]
    ensemble_allowed = bool(
        torch_summary.get("survivor_rows", 0) > 0
        and torch_summary.get("policy_positive_summary_rows", 0) > 0
    )
    if "fail" in statuses:
        return "incomplete_or_failed_contract", False, "One or more required contracts are missing or failed."
    if "blocked_external_runtime" in statuses:
        return "ready_for_gpu_runtime_but_unverified", False, "Local artifacts are ready, but torch/CUDA training has not been verified."
    if "partial" in statuses or "not_yet" in statuses:
        return "preliminary_smoke_or_no_survivor_yet", False, "Current artifacts are smoke/partial or have no survivor evidence."
    if ensemble_allowed:
        return "survivor_and_ensemble_research_ready", True, "Torch survivors and survivor-only ensemble evidence exist."
    return "audit_pass_no_ensemble_survivor", False, "Contracts pass, but no ensemble survivor gate is open."


def write_text_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Rawseq Multi-Horizon Goal Audit",
        "",
        f"Created at: {summary['created_at']}",
        f"Indicator run: {summary.get('indicator_run_dir', '')}",
        f"Torch run: {summary.get('torch_run_dir', '')}",
        f"Ensemble run: {summary.get('ensemble_run_dir', '')}",
        f"Overall status: {summary['overall_status']}",
        f"Ensemble allowed: {summary['ensemble_allowed']}",
        f"Reason: {summary['overall_reason']}",
        "",
        "## Requirement Status",
    ]
    for row in rows:
        lines.append(
            f"- [{row['status']}] {row['category']}: {row['requirement']} | {row['evidence']}"
        )
        if row.get("recommended_action"):
            lines.append(f"  action: {row['recommended_action']}")
    lines += [
        "",
        "## Safety",
        "- report_only=true",
        "- private_api=false",
        "- orders=false",
        "- promotion=false",
        "- champion_mutation=false",
        "- ensemble_search=false unless ensemble_allowed=true in a future audit",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    indicator_dir = infer_indicator_run_dir()
    torch_dir = infer_torch_run_dir()
    ensemble_dir = infer_ensemble_run_dir()
    output_root = resolve_path(os.getenv("RAWSEQ_GOAL_AUDIT_OUTPUT_DIR", ""), DEFAULT_OUTPUT_ROOT)
    output_dir = output_root / f"rawseq_multi_horizon_goal_audit_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    feature_counts: dict[str, int] = {}
    if indicator_dir is None:
        rows.append(
            requirement_row(
                "indicator_run",
                "multi-horizon indicator run exists",
                "fail",
                "no multi_horizon_indicator_report.txt found",
                "run scripts/tiny/run_rawseq_multi_horizon_indicator_pipeline.py",
            )
        )
    else:
        feature_manifest = read_json(indicator_dir / "feature_manifest.json")
        target_manifest = read_json(indicator_dir / "target_manifest.json")
        split_manifest = read_json(indicator_dir / "split_manifest.json")
        feature_family = read_csv(indicator_dir / "feature_family_manifest.csv")
        model_manifest = read_csv(indicator_dir / "model_manifest.csv")
        ablation = read_csv(indicator_dir / "feature_ablation_manifest.csv")
        sequence_dataset_manifest = read_csv(indicator_dir / "sequence_dataset_manifest.csv")
        feature_rows, feature_counts = audit_feature_families(feature_family)
        rows.extend(
            [
                requirement_row(
                    "indicator_run",
                    "multi-horizon indicator run exists",
                    "pass",
                    str(indicator_dir),
                    "",
                ),
                *feature_rows,
                *audit_window_and_target_scope(feature_manifest, target_manifest),
                *audit_splits(split_manifest),
                *audit_models(model_manifest),
                *audit_ablation(ablation),
                *audit_sequence_datasets(sequence_dataset_manifest),
            ]
        )

    torch_rows, torch_summary = audit_torch_run(torch_dir)
    rows.extend(torch_rows)
    preliminary_status, preliminary_ensemble_allowed, _reason = classify_overall(rows, torch_summary)
    ensemble_rows, ensemble_summary = audit_ensemble_run(ensemble_dir, preliminary_ensemble_allowed)
    rows.extend(ensemble_rows)
    overall_status, ensemble_allowed, reason = classify_overall(rows, torch_summary)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "indicator_run_dir": str(indicator_dir) if indicator_dir else "",
        "torch_run_dir": str(torch_dir) if torch_dir else "",
        "ensemble_run_dir": str(ensemble_dir) if ensemble_dir else "",
        "overall_status": overall_status,
        "overall_reason": reason,
        "ensemble_allowed": ensemble_allowed,
        "feature_family_counts": feature_counts,
        "torch_summary": torch_summary,
        "ensemble_summary": ensemble_summary,
        "paper_only": True,
        "private_api": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
        "ensemble_search": False,
    }

    rows_df = pd.DataFrame(rows)
    rows_path = output_dir / "rawseq_multi_horizon_goal_audit.csv"
    summary_path = output_dir / "rawseq_multi_horizon_goal_audit_summary.json"
    text_path = output_dir / "rawseq_multi_horizon_goal_audit.txt"
    rows_df.to_csv(rows_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_text_report(text_path, summary, rows)

    print("Rawseq multi-horizon goal audit complete")
    print(f"Output dir: {output_dir}")
    print(f"Overall status: {overall_status}")
    print(f"Ensemble allowed: {ensemble_allowed}")
    print(f"Indicator run: {indicator_dir}")
    print(f"Torch run: {torch_dir}")
    print(f"Ensemble run: {ensemble_dir}")
    print(f"CSV: {rows_path}")
    print(f"TXT: {text_path}")
    print("Safety: report_only=true private_api=false orders=false promotion=false champion_mutation=false ensemble_search=false")
    print(rows_df[["category", "requirement", "status", "evidence"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
