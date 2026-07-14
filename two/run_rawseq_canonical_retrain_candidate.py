#!/usr/bin/env python3
"""Train one rawseq candidate against the canonical purged split.

This wrapper prepares an isolated trainer source from the canonical table,
launches tiny_price_rawseq_path_v1.py in paper-only research mode, and evaluates
validation-selected policy thresholds on the untouched holdout split.

No promotion, champion mutation, private API access, or orders.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rawseq_policy_scoring import expectancy_metrics, score_policy_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAWSEQ_TRAINER = PROJECT_ROOT / "scripts" / "tiny_price_rawseq_path_v1.py"
CANONICAL_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_canonical_tables"
BASELINE_ROOT = Path(
    os.getenv(
        "RAWSEQ_CANONICAL_BASELINE_ROOT",
        os.getenv(
            "RAWSEQ_BASELINE_OUTPUT_DIR",
            str(PROJECT_ROOT / "data" / "research" / "rawseq_canonical_baselines"),
        ),
    )
).expanduser()
if not BASELINE_ROOT.is_absolute():
    BASELINE_ROOT = PROJECT_ROOT / BASELINE_ROOT
OUTPUT_ROOT = Path(
    os.getenv(
        "RAWSEQ_CANONICAL_RAWSEQ_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_canonical_rawseq_candidates"),
    )
).expanduser()
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

TABLE_PATH_ENV = os.getenv("RAWSEQ_CANONICAL_TABLE_PATH", "").strip()
MANIFEST_PATH_ENV = os.getenv("RAWSEQ_CANONICAL_MANIFEST_PATH", "").strip()
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper() or "SOLUSDT"
VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower() or "kraken"
INPUT_FEATURE = os.getenv("RAWSEQ_CANONICAL_RAWSEQ_INPUT_FEATURE", "ma_distance").strip().lower()
FEATURE_WINDOW = int(float(os.getenv("RAWSEQ_CANONICAL_RAWSEQ_FEATURE_WINDOW", "60")))
SEQ_LEN = int(float(os.getenv("RAWSEQ_CANONICAL_RAWSEQ_SEQ_LEN", "60")))
HIDDEN = os.getenv("RAWSEQ_CANONICAL_RAWSEQ_HIDDEN", "4,4").strip()
SEED = int(float(os.getenv("RAWSEQ_CANONICAL_RAWSEQ_SEED", "900")))
POPULATION = int(float(os.getenv("RAWSEQ_CANONICAL_RAWSEQ_POPULATION", "2")))
GENERATIONS = int(float(os.getenv("RAWSEQ_CANONICAL_RAWSEQ_GENERATIONS", "1")))
EPOCHS = int(float(os.getenv("RAWSEQ_CANONICAL_RAWSEQ_EPOCHS", "2")))
THRESHOLDS = [
    float(item.strip())
    for item in os.getenv("RAWSEQ_CANONICAL_RAWSEQ_THRESHOLDS_BPS", "0,0.1,0.2,0.5,1").split(",")
    if item.strip()
]
COSTS = [
    float(item.strip())
    for item in os.getenv("RAWSEQ_CANONICAL_RAWSEQ_COSTS_BPS", "0.1,1,5").split(",")
    if item.strip()
]
DECISION_COST_BPS = float(os.getenv("RAWSEQ_CANONICAL_RAWSEQ_DECISION_COST_BPS", "0.1"))
MIN_POSITION_TRADES = int(float(os.getenv("RAWSEQ_CANONICAL_RAWSEQ_MIN_POSITION_TRADES", "30")))

PRED_COLUMN = "rawseq_path_pred_horizon_return_bps"
ACTUAL_COLUMN = "rawseq_path_actual_horizon_return_bps"


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def resolve_latest_canonical_dir() -> Path:
    if not CANONICAL_ROOT.exists():
        raise SystemExit(f"Canonical root does not exist: {CANONICAL_ROOT}")
    candidates = [
        path
        for path in CANONICAL_ROOT.iterdir()
        if path.is_dir()
        and (path / "canonical_training_table.csv").exists()
        and (path / "split_manifest.json").exists()
    ]
    if not candidates:
        raise SystemExit(f"No canonical table folders found under {CANONICAL_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_inputs() -> tuple[Path, Path, Path]:
    if TABLE_PATH_ENV:
        table_path = Path(TABLE_PATH_ENV).expanduser()
        if not table_path.is_absolute():
            table_path = PROJECT_ROOT / table_path
        base_dir = table_path.parent
    else:
        base_dir = resolve_latest_canonical_dir()
        table_path = base_dir / "canonical_training_table.csv"
    if MANIFEST_PATH_ENV:
        manifest_path = Path(MANIFEST_PATH_ENV).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
    else:
        manifest_path = base_dir / "split_manifest.json"
    if not table_path.exists():
        raise SystemExit(f"Canonical table not found: {table_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Canonical manifest not found: {manifest_path}")
    return base_dir, table_path, manifest_path


def prepare_trainer_source(table_path: Path, run_dir: Path) -> Path:
    table = pd.read_csv(table_path, low_memory=False)
    required = {"decision_timestamp", "price", "split"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise SystemExit(f"Canonical table missing required columns for trainer source: {missing}")
    split_map = {
        "train": "train",
        "validation": "validation",
        "untouched_holdout": "test",
        "purge_embargo": "purge_embargo",
    }
    source = pd.DataFrame(
        {
            "timestamp": pd.to_numeric(table["decision_timestamp"], errors="coerce"),
            "price": pd.to_numeric(table["price"], errors="coerce"),
            "predicted_side": "long",
            "rawseq_wf_split": table["split"].astype(str).map(split_map).fillna("purge_embargo"),
        }
    )
    source = source.dropna(subset=["timestamp", "price"]).sort_values("timestamp").reset_index(drop=True)
    source["time"] = pd.to_datetime(source["timestamp"], unit="ms", utc=True).astype(str)
    path = run_dir / "trainer_source.csv"
    source.to_csv(path, index=False)
    return path


def trainer_env(source_path: Path, artifact_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    horizon_seconds = int(float(manifest.get("horizon_seconds", 60)))
    bucket_seconds = int(round(float(manifest.get("estimated_bucket_seconds", 1.0))))
    bucket_seconds = max(bucket_seconds, 1)
    env = os.environ.copy()
    env.update(
        {
            "SYMBOL": SYMBOL,
            "PRIMARY_VENUE": VENUE,
            "RAWSEQ_SOURCE_PATH": str(source_path),
            "RAWSEQ_ARTIFACT_OUTPUT_DIR": str(artifact_dir),
            "RAWSEQ_BUCKET_SECONDS": str(bucket_seconds),
            "RAWSEQ_LEN": str(SEQ_LEN),
            "RAWSEQ_INPUT_FEATURE": INPUT_FEATURE,
            "RAWSEQ_MA_WINDOW": str(FEATURE_WINDOW),
            "RAWSEQ_FEATURE_WINDOW": str(FEATURE_WINDOW),
            "RAWSEQ_HIDDEN": HIDDEN,
            "RAWSEQ_SEED": str(SEED),
            "RAWSEQ_POPULATION": str(POPULATION),
            "RAWSEQ_GENERATIONS": str(GENERATIONS),
            "RAWSEQ_EPOCHS": str(EPOCHS),
            "RAWSEQ_OUTPUT_LABEL": "future_return_path",
            "RAWSEQ_OUTPUT_ORIENTATION": "market_relative",
            "RAWSEQ_DECISION_HORIZON_SECONDS": str(horizon_seconds),
            "RAWSEQ_FITNESS_POLICY": "direct_gt",
            "RAWSEQ_FITNESS_THRESHOLD_BPS": "0.0",
            "RAWSEQ_MIN_FITNESS_TRADES": str(MIN_POSITION_TRADES),
            "RAWSEQ_EVOLUTION_EARLY_STOP_PATIENCE": "3",
        }
    )
    return env


def run_trainer(source_path: Path, artifact_dir: Path, run_dir: Path, manifest: dict[str, Any]) -> subprocess.CompletedProcess:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, str(RAWSEQ_TRAINER)],
        cwd=str(PROJECT_ROOT),
        env=trainer_env(source_path, artifact_dir, manifest),
        text=True,
        capture_output=True,
        check=False,
    )
    (run_dir / "trainer.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (run_dir / "trainer.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    return completed


def selected_non_overlapping(scored: pd.DataFrame, horizon_ms: int) -> np.ndarray:
    values: list[float] = []
    next_allowed = -math.inf
    for _, row in scored.sort_values("timestamp").iterrows():
        timestamp = safe_float(row.get("timestamp"))
        if not math.isfinite(timestamp) or timestamp < next_allowed:
            continue
        net = safe_float(row.get("net_bps"))
        if math.isfinite(net):
            values.append(net)
            next_allowed = timestamp + horizon_ms
    return np.asarray(values, dtype=np.float64)


def policy_rows(annotated: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    horizon_ms = int(float(manifest.get("horizon_seconds", 60))) * 1000
    rows: list[dict[str, Any]] = []
    for split_name, split_label in [("validation", "validation"), ("untouched_holdout", "test")]:
        subset = annotated[annotated["rawseq_wf_split"].astype(str).eq(split_label)].copy()
        if subset.empty:
            continue
        for threshold in THRESHOLDS:
            for cost in COSTS:
                scored = score_policy_frame(
                    subset,
                    PRED_COLUMN,
                    ACTUAL_COLUMN,
                    "direct_gt",
                    threshold,
                    cost_bps=cost,
                    selected_only=True,
                )
                row_values = scored["net_bps"].to_numpy(dtype=np.float64) if not scored.empty else np.asarray([])
                row_metric = expectancy_metrics(row_values)
                non_overlap_values = selected_non_overlapping(scored, horizon_ms)
                non_overlap = expectancy_metrics(non_overlap_values)
                # The current position simulator is one active position with horizon exit,
                # equivalent to non-overlap plus explicit exposure accounting.
                span_ms = safe_float(subset["timestamp"].max()) - safe_float(subset["timestamp"].min())
                exposure = (
                    min(int(non_overlap["rows"]) * horizon_ms / max(span_ms, 1.0), 1.0)
                    if math.isfinite(span_ms)
                    else math.nan
                )
                rows.append(
                    {
                        "split": split_name,
                        "policy": "direct_gt",
                        "threshold_bps": threshold,
                        "cost_bps": cost,
                        "row_signal_selected_rows": int(row_metric["rows"]),
                        "row_signal_avg_net_bps": row_metric["avg_net_bps"],
                        "row_signal_cum_net_bps": row_metric["cum_net_bps"],
                        "row_signal_win_rate_net": row_metric["win_rate_net"],
                        "row_signal_max_dip_net_bps": row_metric["max_dip_net_bps"],
                        "non_overlapping_selected_rows": int(non_overlap["rows"]),
                        "non_overlapping_avg_net_bps": non_overlap["avg_net_bps"],
                        "non_overlapping_cum_net_bps": non_overlap["cum_net_bps"],
                        "non_overlapping_win_rate_net": non_overlap["win_rate_net"],
                        "non_overlapping_max_dip_net_bps": non_overlap["max_dip_net_bps"],
                        "position_trade_count": int(non_overlap["rows"]),
                        "position_avg_net_bps": non_overlap["avg_net_bps"],
                        "position_cum_net_bps": non_overlap["cum_net_bps"],
                        "position_win_rate_net": non_overlap["win_rate_net"],
                        "position_max_dip_net_bps": non_overlap["max_dip_net_bps"],
                        "position_exposure_fraction": exposure,
                    }
                )
    return pd.DataFrame(rows)


def select_validation_policy(metrics: pd.DataFrame) -> dict[str, Any]:
    validation = metrics[
        metrics["split"].eq("validation")
        & metrics["cost_bps"].astype(float).sub(DECISION_COST_BPS).abs().lt(1e-12)
    ].copy()
    if validation.empty:
        return {
            "status": "training_candidate",
            "status_reason": "missing_validation_policy_metrics",
        }
    validation = validation.sort_values(
        ["position_cum_net_bps", "position_trade_count", "non_overlapping_cum_net_bps"],
        ascending=[False, False, False],
    )
    selected = validation.iloc[0]
    holdout_match = metrics[
        metrics["split"].eq("untouched_holdout")
        & metrics["policy"].eq(selected["policy"])
        & metrics["threshold_bps"].astype(float).sub(float(selected["threshold_bps"])).abs().lt(1e-12)
        & metrics["cost_bps"].astype(float).sub(DECISION_COST_BPS).abs().lt(1e-12)
    ]
    holdout = holdout_match.iloc[0] if not holdout_match.empty else pd.Series(dtype=object)
    validation_trades = int(safe_float(selected.get("position_trade_count"), 0))
    holdout_trades = int(safe_float(holdout.get("position_trade_count"), 0))
    if validation_trades < MIN_POSITION_TRADES:
        status = "insufficient_sample"
        reason = f"validation_position_trade_count<{MIN_POSITION_TRADES}"
    elif safe_float(selected.get("position_cum_net_bps")) <= 0.0:
        status = "training_candidate"
        reason = "validation_position_cum_net_not_positive"
    elif holdout_trades >= MIN_POSITION_TRADES and safe_float(holdout.get("position_cum_net_bps")) > 0.0:
        status = "holdout_survivor"
        reason = "validation_and_holdout_position_cum_net_positive"
    else:
        status = "validation_survivor"
        reason = "validation_positive_but_holdout_gate_not_met"
    return {
        "status": status,
        "status_reason": reason,
        "selection_stage": "validation_selected",
        "holdout_stage": "untouched_holdout_evaluated",
        "selected_policy": selected["policy"],
        "selected_threshold_bps": float(selected["threshold_bps"]),
        "decision_cost_bps": DECISION_COST_BPS,
        "validation_position_cum_net_bps": selected["position_cum_net_bps"],
        "validation_position_trade_count": validation_trades,
        "validation_position_max_dip_net_bps": selected["position_max_dip_net_bps"],
        "holdout_position_cum_net_bps": holdout.get("position_cum_net_bps", math.nan),
        "holdout_position_trade_count": holdout_trades,
        "holdout_position_max_dip_net_bps": holdout.get("position_max_dip_net_bps", math.nan),
        "min_position_trades": MIN_POSITION_TRADES,
    }


def prediction_table(annotated: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    train_rows = annotated[annotated["rawseq_wf_split"].astype(str).eq("train")]
    training_end = safe_float(train_rows["timestamp"].max()) if not train_rows.empty else math.nan
    out = pd.DataFrame(
        {
            "timestamp": pd.to_numeric(annotated["timestamp"], errors="coerce"),
            "actual_return_bps": pd.to_numeric(annotated[ACTUAL_COLUMN], errors="coerce"),
            "rawseq_pred_bps": pd.to_numeric(annotated[PRED_COLUMN], errors="coerce"),
            "source_fold": annotated["rawseq_wf_split"].astype(str).replace({"test": "untouched_holdout"}),
            "base_model_training_end": training_end,
            "model_family": "rawseq_sequence_alpha",
            "feature_schema_hash": manifest.get("feature_schema_hash", ""),
            "target": f"future_market_return_bps_{manifest.get('horizon_seconds', '')}s",
        }
    )
    out["prediction_stage"] = out["source_fold"].map(
        {
            "train": "in_sample_fit_diagnostic",
            "validation": "validation_selection",
            "untouched_holdout": "untouched_holdout_evaluation",
            "purge_embargo": "purge_embargo_diagnostic",
        }
    ).fillna("unknown")
    return out.dropna(subset=["timestamp", "actual_return_bps", "rawseq_pred_bps"]).reset_index(drop=True)


def parse_candidate_model_path(stdout: str) -> str:
    match = re.search(r"Candidate model:\s*(.+?model\.json)", stdout or "")
    return match.group(1).strip() if match else ""


def resolve_artifact_path(artifact_dir: Path, names: list[str]) -> Path:
    for name in names:
        path = artifact_dir / name
        if path.exists():
            return path
    expected = ", ".join(str(artifact_dir / name) for name in names)
    raise SystemExit(f"Trainer completed but none of the expected artifacts exist: {expected}")


def latest_baseline_comparison(feature_schema_hash: str) -> dict[str, Any]:
    if not BASELINE_ROOT.exists():
        return {}
    candidates = []
    for path in BASELINE_ROOT.iterdir():
        selected = path / "baseline_selected_policies.csv"
        manifest = path / "baseline_manifest.json"
        if path.is_dir() and selected.exists() and manifest.exists():
            payload = load_json(manifest)
            if str(payload.get("feature_schema_hash")) == str(feature_schema_hash):
                candidates.append(path)
    if not candidates:
        return {}
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    frame = pd.read_csv(latest / "baseline_selected_policies.csv", low_memory=False)
    if frame.empty:
        return {"baseline_dir": str(latest)}
    best_validation = frame.sort_values("validation_position_cum_net_bps", ascending=False).iloc[0]
    best_holdout = frame.sort_values("holdout_position_cum_net_bps", ascending=False).iloc[0]
    return {
        "baseline_dir": str(latest),
        "baseline_root": str(BASELINE_ROOT),
        "best_baseline_validation_model": best_validation.get("model", ""),
        "best_baseline_validation_position_cum_net_bps": safe_float(best_validation.get("validation_position_cum_net_bps")),
        "best_baseline_holdout_model": best_holdout.get("model", ""),
        "best_baseline_holdout_position_cum_net_bps": safe_float(best_holdout.get("holdout_position_cum_net_bps")),
    }


def write_summary(path: Path, summary: dict[str, Any], trainer_returncode: int) -> None:
    lines = [
        "Rawseq Canonical Retrain Candidate",
        "",
        f"Created at: {datetime.now(UTC).isoformat()}",
        f"Trainer return code: {trainer_returncode}",
        f"Status: {summary.get('status', '')}",
        f"Status reason: {summary.get('status_reason', '')}",
        "",
        "Contract:",
        f"  target: {summary.get('target', '')}",
        f"  input_feature: {summary.get('input_feature', '')}",
        f"  feature_window: {summary.get('feature_window', '')}",
        f"  hidden: {summary.get('hidden', '')}",
        f"  seed: {summary.get('seed', '')}",
        f"  output_orientation: market_relative",
        f"  candidate_model_path: {summary.get('candidate_model_path', '')}",
        "",
        "Validation-selected policy:",
        f"  policy: {summary.get('selected_policy', '')}",
        f"  threshold_bps: {fmt(summary.get('selected_threshold_bps'))}",
        f"  cost_bps: {fmt(summary.get('decision_cost_bps'))}",
        f"  validation_position_cum_net_bps: {fmt(summary.get('validation_position_cum_net_bps'))}",
        f"  validation_position_trade_count: {summary.get('validation_position_trade_count', '')}",
        f"  holdout_position_cum_net_bps: {fmt(summary.get('holdout_position_cum_net_bps'))}",
        f"  holdout_position_trade_count: {summary.get('holdout_position_trade_count', '')}",
        "",
        "Baseline comparison:",
        f"  baseline_root: {summary.get('baseline_root', '')}",
        f"  baseline_dir: {summary.get('baseline_dir', '')}",
        f"  best_baseline_validation_position_cum_net_bps: {fmt(summary.get('best_baseline_validation_position_cum_net_bps'))}",
        f"  best_baseline_holdout_position_cum_net_bps: {fmt(summary.get('best_baseline_holdout_position_cum_net_bps'))}",
        "",
        "Safety: paper_only=true training=rawseq_research_only promotion=false champion_mutation=false orders=false",
        "",
        "Interpretation:",
        "  This is one corrected rawseq candidate under the canonical split.",
        "  It is not ensemble-ready unless it beats baselines on validation and survives untouched holdout.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    base_dir, table_path, manifest_path = resolve_inputs()
    manifest = load_json(manifest_path)
    run_id = (
        f"rawseq_canonical_{SYMBOL}_{VENUE}_{INPUT_FEATURE}_fw{FEATURE_WINDOW}_"
        f"h{HIDDEN.replace(',', 'x')}_s{SEED}_{now_stamp()}"
    )
    run_dir = OUTPUT_ROOT / run_id
    artifact_dir = run_dir / "trainer_artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    source_path = prepare_trainer_source(table_path, run_dir)
    completed = run_trainer(source_path, artifact_dir, run_dir, manifest)
    if completed.returncode != 0:
        summary = {
            "status": "train_failed",
            "status_reason": "trainer_return_code_nonzero",
            "trainer_returncode": completed.returncode,
            "paper_only": True,
            "promotion": False,
            "champion_mutation": False,
            "orders": False,
        }
        pd.DataFrame([summary]).to_csv(run_dir / "rawseq_canonical_candidate_summary.csv", index=False)
        write_summary(run_dir / "rawseq_canonical_candidate_summary.txt", summary, completed.returncode)
        print(f"Trainer failed. See {run_dir / 'trainer.stderr.log'}")
        return completed.returncode

    annotated_path = resolve_artifact_path(artifact_dir, ["annotated.csv", "rawseq_annotated.csv"])
    annotated = pd.read_csv(annotated_path, low_memory=False)
    required = {"rawseq_wf_split", PRED_COLUMN, ACTUAL_COLUMN, "timestamp"}
    missing = sorted(required - set(annotated.columns))
    if missing:
        raise SystemExit(f"Annotated rawseq output missing required columns: {missing}")
    metrics = policy_rows(annotated, manifest)
    selected = select_validation_policy(metrics)
    baseline = latest_baseline_comparison(str(manifest.get("feature_schema_hash", "")))
    candidate_model_path = parse_candidate_model_path(completed.stdout or "")
    selected.update(baseline)
    selected.update(
        {
            "canonical_dir": str(base_dir),
            "canonical_table_path": str(table_path),
            "canonical_manifest_path": str(manifest_path),
            "run_dir": str(run_dir),
            "artifact_dir": str(artifact_dir),
            "annotated_path": str(annotated_path),
            "trainer_source_path": str(source_path),
            "candidate_model_path": candidate_model_path,
            "trainer_returncode": completed.returncode,
            "symbol": SYMBOL,
            "venue": VENUE,
            "input_feature": INPUT_FEATURE,
            "feature_window": FEATURE_WINDOW,
            "seq_len": SEQ_LEN,
            "hidden": HIDDEN,
            "seed": SEED,
            "population": POPULATION,
            "generations": GENERATIONS,
            "epochs": EPOCHS,
            "target": f"future_market_return_bps_{manifest.get('horizon_seconds', '')}s",
            "output_label": "future_return_path",
            "output_orientation": "market_relative",
            "feature_schema_hash": manifest.get("feature_schema_hash", ""),
            "paper_only": True,
            "training": "rawseq_research_only",
            "promotion": False,
            "champion_mutation": False,
            "orders": False,
        }
    )
    if "best_baseline_validation_position_cum_net_bps" in selected:
        selected["beats_best_baseline_validation"] = (
            safe_float(selected.get("validation_position_cum_net_bps"))
            > safe_float(selected.get("best_baseline_validation_position_cum_net_bps"))
        )
        selected["beats_best_baseline_holdout"] = (
            safe_float(selected.get("holdout_position_cum_net_bps"))
            > safe_float(selected.get("best_baseline_holdout_position_cum_net_bps"))
        )

    metrics.to_csv(run_dir / "rawseq_canonical_candidate_policy_metrics.csv", index=False)
    pd.DataFrame([selected]).to_csv(run_dir / "rawseq_canonical_candidate_summary.csv", index=False)
    prediction_table(annotated, manifest).to_csv(run_dir / "rawseq_canonical_predictions.csv", index=False)
    (run_dir / "rawseq_canonical_contract.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_summary(run_dir / "rawseq_canonical_candidate_summary.txt", selected, completed.returncode)

    print("Rawseq canonical retrain candidate complete")
    print(f"Run dir: {run_dir}")
    print(f"Status: {selected.get('status')} reason={selected.get('status_reason')}")
    print(f"Summary: {run_dir / 'rawseq_canonical_candidate_summary.txt'}")
    print(pd.DataFrame([selected]).to_string(index=False))
    print("Safety: paper_only=true training=rawseq_research_only promotion=false champion_mutation=false orders=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
