#!/usr/bin/env python3
"""Select walk-forward rawseq contract representatives worth full-source probing.

Read-only except for writing survivor report outputs.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WF_ROOT = PROJECT_ROOT / "data" / "rawseq_walkforward"
PROBE_ROOT = PROJECT_ROOT / "data" / "research" / "rawseq_candidate_shadow_probes"

WF_RUN_DIR_ENV = os.getenv("RAWSEQ_WF_RUN_DIR", "").strip()
MIN_ROWS = int(os.getenv("RAWSEQ_SURVIVOR_MIN_ROWS", "300"))
DECISION_COST_BPS = float(os.getenv("RAWSEQ_SURVIVOR_DECISION_COST_BPS", "0.1"))
MIN_POSITIVE_WINDOW_FRACTION = float(os.getenv("RAWSEQ_SURVIVOR_MIN_POSITIVE_WINDOW_FRACTION", "0.50"))
OUTPUT_PATH_ENV = os.getenv("RAWSEQ_SURVIVOR_OUTPUT_PATH", "").strip()

CONTRACT_COLS = ["input_feature", "ma_window", "hidden", "input_stride", "output_stride"]


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def norm_path(value: Any) -> str:
    text = safe_str(value)
    if not text:
        return ""
    try:
        return str(Path(text).resolve()).lower()
    except Exception:
        return str(Path(text)).lower()


def normalize_hidden(value: Any) -> str:
    text = safe_str(value).replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    return ",".join(part.strip() for part in text.split(",") if part.strip()).replace(" ", "")


def normalize_contract_fields(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in CONTRACT_COLS:
        if column not in out.columns:
            out[column] = ""
    out["input_feature"] = out["input_feature"].astype(str).str.strip().str.lower()
    out["ma_window"] = out["ma_window"].apply(lambda value: str(safe_int(value)) if safe_str(value) else "")
    out["hidden"] = out["hidden"].apply(normalize_hidden)
    out["input_stride"] = out["input_stride"].apply(lambda value: str(safe_int(value, 1) or 1))
    out["output_stride"] = out["output_stride"].apply(lambda value: str(safe_int(value, 1) or 1))
    return out


def contract_key(row: pd.Series | dict[str, Any]) -> str:
    return "|".join(safe_str(row.get(column)) for column in CONTRACT_COLS)


def latest_walkforward_run() -> Path:
    if WF_RUN_DIR_ENV:
        path = Path(WF_RUN_DIR_ENV)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise SystemExit(f"RAWSEQ_WF_RUN_DIR does not exist: {path}")
        return path.resolve()
    if not WF_ROOT.exists():
        raise SystemExit(f"Walk-forward root does not exist: {WF_ROOT}")
    candidates = [
        path
        for path in WF_ROOT.iterdir()
        if path.is_dir() and (path / "candidates.csv").exists() and (path / "contract_leaderboard.csv").exists()
    ]
    if not candidates:
        raise SystemExit(f"No walk-forward run folder found under {WF_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def output_paths(run_dir: Path) -> tuple[Path, Path]:
    if OUTPUT_PATH_ENV:
        output = Path(OUTPUT_PATH_ENV)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
    else:
        output = run_dir / "walkforward_contract_survivors.csv"
    return output, output.with_suffix(".txt")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc


def parse_threshold_from_strategy(value: Any) -> float:
    text = safe_str(value)
    match = re.search(r"_gt_([-+]?\d+(?:\.\d+)?)", text)
    if match:
        return safe_float(match.group(1))
    return math.nan


def load_probe_decisions() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not PROBE_ROOT.exists():
        return pd.DataFrame()
    for path in sorted(PROBE_ROOT.glob("*/decision_summary.csv")):
        frame = read_csv(path)
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        row["probe_decision_path"] = str(path.resolve())
        row["probe_dir"] = str(path.parent.resolve())
        row["probe_model_path_norm"] = norm_path(row.get("model_path"))
        row["probe_decision"] = safe_str(row.get("decision"))
        row["probe_cum_net_bps"] = safe_float(row.get("cum_net_bps"))
        row["probe_selected_rows"] = safe_int(row.get("selected_rows"))
        row["input_feature"] = safe_str(row.get("input_feature")).lower()
        row["hidden"] = normalize_hidden(row.get("hidden"))
        row["input_stride"] = str(safe_int(row.get("input_stride"), 1) or 1)
        row["output_stride"] = str(safe_int(row.get("output_stride"), 1) or 1)
        rows.append(row)
    return pd.DataFrame(rows)


def annotate_probe_matches(candidates: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["archived_model_path_norm"] = out["archived_model_path"].apply(norm_path) if "archived_model_path" in out.columns else ""
    out["model_path_norm"] = out["model_path"].apply(norm_path) if "model_path" in out.columns else ""
    out["matched_probe_decision"] = ""
    out["matched_probe_dir"] = ""
    out["matched_probe_cum_net_bps"] = math.nan
    out["matched_probe_selected_rows"] = 0
    if probes.empty:
        return out
    probe_by_model = {
        safe_str(row["probe_model_path_norm"]): row
        for _, row in probes.iterrows()
        if safe_str(row.get("probe_model_path_norm"))
    }
    for idx, row in out.iterrows():
        match = None
        for key in [row.get("archived_model_path_norm"), row.get("model_path_norm")]:
            if safe_str(key) in probe_by_model:
                match = probe_by_model[safe_str(key)]
                break
        if match is None:
            continue
        out.at[idx, "matched_probe_decision"] = safe_str(match.get("probe_decision"))
        out.at[idx, "matched_probe_dir"] = safe_str(match.get("probe_dir"))
        out.at[idx, "matched_probe_cum_net_bps"] = safe_float(match.get("probe_cum_net_bps"))
        out.at[idx, "matched_probe_selected_rows"] = safe_int(match.get("probe_selected_rows"))
    return out


def best_threshold_rows(test_scores: pd.DataFrame) -> pd.DataFrame:
    if test_scores.empty:
        return pd.DataFrame()
    scores = normalize_contract_fields(test_scores.copy())
    if "split" in scores.columns:
        scores = scores[scores["split"].astype(str).str.lower().str.contains("test", na=False)]
    for column in ["rows", "avg_return_bps", "cumulative_return_bps", "threshold_bps"]:
        if column in scores.columns:
            scores[column] = pd.to_numeric(scores[column], errors="coerce")
    if "threshold_bps" not in scores.columns:
        scores["threshold_bps"] = scores.get("strategy", "").apply(parse_threshold_from_strategy)
    scores = scores[scores["cumulative_return_bps"].notna()] if "cumulative_return_bps" in scores.columns else scores
    if scores.empty:
        return scores
    best_rows = []
    group_cols = CONTRACT_COLS + ["window_id", "seed"]
    for _, group in scores.groupby(group_cols, dropna=False, sort=False):
        ranked = group.sort_values(["cumulative_return_bps", "rows"], ascending=[False, False])
        best_rows.append(ranked.iloc[0].to_dict())
    return pd.DataFrame(best_rows)


def representative_candidate(group: pd.DataFrame, contract_ok: bool) -> pd.Series | None:
    eligible = group[group["best_test_rows_num"] >= MIN_ROWS].copy()
    if eligible.empty:
        return None
    eligible["threshold_preference"] = (eligible["best_test_threshold_bps_num"] >= 0.1).astype(int)
    eligible["probe_reject_penalty"] = eligible["matched_probe_decision"].astype(str).str.lower().eq("reject").astype(int)
    eligible["contract_stability_preference"] = 1 if contract_ok else 0
    ranked = eligible.sort_values(
        [
            "probe_reject_penalty",
            "threshold_preference",
            "contract_stability_preference",
            "best_test_cumulative_return_bps_num",
            "best_test_rows_num",
        ],
        ascending=[True, False, False, False, False],
    )
    return ranked.iloc[0]


def build_survivors(
    leaderboard: pd.DataFrame,
    candidates: pd.DataFrame,
    test_scores: pd.DataFrame,
    selected_by_window: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    candidates = normalize_contract_fields(candidates)
    leaderboard = normalize_contract_fields(leaderboard) if not leaderboard.empty else pd.DataFrame()
    selected_by_window = normalize_contract_fields(selected_by_window) if not selected_by_window.empty else pd.DataFrame()
    best_scores = best_threshold_rows(test_scores)
    selected_keys = set(selected_by_window.apply(contract_key, axis=1).tolist()) if not selected_by_window.empty else set()

    for column in ["best_test_rows", "best_test_cumulative_return_bps", "best_test_avg_return_bps", "best_test_threshold_bps"]:
        if column in candidates.columns:
            candidates[f"{column}_num"] = pd.to_numeric(candidates[column], errors="coerce")
        else:
            candidates[f"{column}_num"] = math.nan

    rows = []
    for key, group in candidates.groupby(CONTRACT_COLS, dropna=False, sort=True):
        contract = dict(zip(CONTRACT_COLS, key))
        group = group.copy()
        windows_tested = int(group["window_id"].nunique()) if "window_id" in group.columns else 0
        positive = group[group["best_test_cumulative_return_bps_num"] > 0.0]
        positive_windows = int(positive["window_id"].nunique()) if "window_id" in positive.columns else int(len(positive))
        positive_fraction = positive_windows / windows_tested if windows_tested else math.nan
        total_rows = int(group["best_test_rows_num"].fillna(0).sum())
        total_cum = float(group["best_test_cumulative_return_bps_num"].fillna(0.0).sum())
        mean_avg = float(group["best_test_avg_return_bps_num"].mean())
        worst_window = safe_float(group["best_test_cumulative_return_bps_num"].min())
        thresholds = group["best_test_threshold_bps_num"].dropna()
        threshold_0_best_count = int((thresholds.abs() < 1e-12).sum())
        threshold_0_1_or_higher_best_count = int((thresholds >= 0.1 - 1e-12).sum())
        sparse_high_threshold_count = int(
            ((group["best_test_threshold_bps_num"] >= 0.1 - 1e-12) & (group["best_test_rows_num"] < MIN_ROWS)).sum()
        )
        selected_windows = int(
            selected_by_window[selected_by_window.apply(contract_key, axis=1).eq("|".join(key))]["window_id"].nunique()
        ) if not selected_by_window.empty and "|".join(key) in selected_keys else 0

        probe_rejects = group[group["matched_probe_decision"].astype(str).str.lower().eq("reject")]
        probe_research = group[
            group["matched_probe_decision"].astype(str).str.lower().isin(["research_candidate", "clean_champion_candidate"])
        ]
        contract_stable = math.isfinite(positive_fraction) and positive_fraction >= MIN_POSITIVE_WINDOW_FRACTION
        rep = representative_candidate(group, contract_stable)

        reasons: list[str] = []
        survivor_status = "probe_candidate"
        if not probe_rejects.empty and probe_research.empty:
            survivor_status = "already_probe_rejected"
            reasons.append(f"full_source_probe_rejects={len(probe_rejects)}")
        elif threshold_0_best_count > 0 and threshold_0_1_or_higher_best_count == 0:
            survivor_status = "reject"
            reasons.append("threshold_0_only_best_behavior")
        elif rep is None:
            survivor_status = "reject"
            reasons.append(f"no_representative_rows_ge_{MIN_ROWS}")
        elif not contract_stable:
            survivor_status = "reject"
            reasons.append(f"positive_window_fraction_below_{MIN_POSITIVE_WINDOW_FRACTION:g}")
        elif threshold_0_best_count > threshold_0_1_or_higher_best_count:
            survivor_status = "reject"
            reasons.append("threshold_0_dominates_best_windows")
        elif not probe_research.empty:
            survivor_status = "research_candidate"
            reasons.append(f"full_source_probe_survivors={len(probe_research)}")
        else:
            survivor_status = "probe_candidate"

        if sparse_high_threshold_count:
            reasons.append(f"sparse_high_threshold_windows={sparse_high_threshold_count}")
        if worst_window < 0.0:
            reasons.append(f"worst_window_negative={worst_window:.6g}")
        if selected_windows <= 1:
            reasons.append(f"selected_windows={selected_windows}")

        if rep is None:
            rep_data: dict[str, Any] = {}
        else:
            rep_data = rep.to_dict()

        contract_score = (
            total_cum
            + 500.0 * safe_float(positive_fraction, 0.0)
            + 100.0 * threshold_0_1_or_higher_best_count
            - 100.0 * threshold_0_best_count
            - 250.0 * len(probe_rejects)
            - 150.0 * sparse_high_threshold_count
        )
        rows.append(
            {
                **contract,
                "windows_tested": windows_tested,
                "positive_windows": positive_windows,
                "positive_window_fraction": positive_fraction,
                "selected_windows": selected_windows,
                "total_rows": total_rows,
                "total_cumulative_return_bps": total_cum,
                "mean_avg_return_bps": mean_avg,
                "worst_window_cum_bps": worst_window,
                "threshold_0_best_count": threshold_0_best_count,
                "threshold_0_1_or_higher_best_count": threshold_0_1_or_higher_best_count,
                "threshold_distribution": ",".join(
                    f"{threshold:g}:{count}"
                    for threshold, count in thresholds.round(10).value_counts().sort_index().items()
                ),
                "sparse_high_threshold_count": sparse_high_threshold_count,
                "full_source_probe_reject_count": int(len(probe_rejects)),
                "full_source_probe_survivor_count": int(len(probe_research)),
                "full_source_probe_reject_dirs": ";".join(
                    sorted({safe_str(value) for value in probe_rejects["matched_probe_dir"] if safe_str(value)})
                ),
                "representative_archived_model_path": safe_str(rep_data.get("archived_model_path")),
                "representative_window_id": safe_str(rep_data.get("window_id")),
                "representative_seed": safe_str(rep_data.get("seed")),
                "representative_rows": safe_int(rep_data.get("best_test_rows")),
                "representative_cum_bps": safe_float(rep_data.get("best_test_cumulative_return_bps")),
                "representative_threshold_bps": safe_float(rep_data.get("best_test_threshold_bps")),
                "representative_probe_decision": safe_str(rep_data.get("matched_probe_decision")),
                "survivor_status": survivor_status,
                "contract_score": contract_score,
                "reasons": "; ".join(reasons),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    status_rank = {"research_candidate": 3, "probe_candidate": 2, "already_probe_rejected": 1, "reject": 0}
    out["_status_rank"] = out["survivor_status"].map(status_rank).fillna(0)
    out = out.sort_values(
        ["_status_rank", "contract_score", "positive_window_fraction", "total_cumulative_return_bps"],
        ascending=[False, False, False, False],
    ).drop(columns=["_status_rank"])
    return out


def render_contract(row: pd.Series) -> str:
    return (
        f"{row['input_feature']} ma={row['ma_window'] or 'NA'} hidden={row['hidden']} "
        f"stride={row['input_stride']}/{row['output_stride']}"
    )


def render_text(report: pd.DataFrame, run_dir: Path, output_csv: Path, output_txt: Path) -> str:
    lines = [
        "Rawseq Walk-Forward Contract Survivors",
        "",
        f"Run dir: {run_dir}",
        f"Minimum representative rows: {MIN_ROWS}",
        f"Decision cost bps: {DECISION_COST_BPS:g}",
        f"Minimum positive window fraction: {MIN_POSITIVE_WINDOW_FRACTION:g}",
        f"Contracts evaluated: {len(report)}",
        "",
        "1. Contracts Rejected Due To Full-Source Probe Failures",
    ]
    rejected_probe = report[report["survivor_status"].eq("already_probe_rejected")] if not report.empty else pd.DataFrame()
    if rejected_probe.empty:
        lines.append("  none")
    else:
        for _, row in rejected_probe.iterrows():
            lines.append(
                f"  {render_contract(row)} rejects={int(row['full_source_probe_reject_count'])} "
                f"pf={safe_float(row['positive_window_fraction']):.3f} reasons={row['reasons']}"
            )

    lines += ["", "2. Contracts Rejected Due To Threshold=0-Only Behavior"]
    threshold_zero = report[
        report["reasons"].astype(str).str.contains("threshold_0_only|threshold_0_dominates", na=False)
    ] if not report.empty else pd.DataFrame()
    if threshold_zero.empty:
        lines.append("  none")
    else:
        for _, row in threshold_zero.iterrows():
            lines.append(
                f"  {render_contract(row)} t0={int(row['threshold_0_best_count'])} "
                f"t01plus={int(row['threshold_0_1_or_higher_best_count'])} status={row['survivor_status']}"
            )

    lines += ["", "3. Contracts Rejected Due To Sparse Representatives"]
    sparse = report[
        report["survivor_status"].eq("reject")
        & report["reasons"].astype(str).str.contains("no_representative_rows", na=False)
    ] if not report.empty else pd.DataFrame()
    if sparse.empty:
        lines.append("  none")
    else:
        for _, row in sparse.head(30).iterrows():
            lines.append(
                f"  {render_contract(row)} rep_rows={safe_int(row['representative_rows'])} "
                f"sparse_high={safe_int(row['sparse_high_threshold_count'])} status={row['survivor_status']}"
            )

    penalized_sparse = report[
        report["reasons"].astype(str).str.contains("sparse_high_threshold", na=False)
        & ~report["survivor_status"].eq("reject")
    ] if not report.empty else pd.DataFrame()
    lines += ["", "3b. Sparse High-Threshold Penalties Applied"]
    if penalized_sparse.empty:
        lines.append("  none")
    else:
        for _, row in penalized_sparse.head(20).iterrows():
            lines.append(
                f"  {render_contract(row)} status={row['survivor_status']} "
                f"sparse_high={safe_int(row['sparse_high_threshold_count'])} "
                f"rep_rows={safe_int(row['representative_rows'])}"
            )

    lines += ["", "4. Best Remaining Representative Model Paths To Probe"]
    candidates = report[report["survivor_status"].isin(["probe_candidate", "research_candidate"])] if not report.empty else pd.DataFrame()
    if candidates.empty:
        lines.append("  none")
    else:
        for _, row in candidates.head(25).iterrows():
            lines.append(
                f"  {render_contract(row)} status={row['survivor_status']} "
                f"window={row['representative_window_id']} seed={row['representative_seed']} "
                f"rows={safe_int(row['representative_rows'])} cum={safe_float(row['representative_cum_bps']):.6g} "
                f"threshold={safe_float(row['representative_threshold_bps']):.6g} "
                f"path={row['representative_archived_model_path']}"
            )

    lines += [
        "",
        "5. Champion Warning",
        "  No champion should be created from walk-forward ranking alone. Full-source probe, decision, dynamic-cost, and stability reports still gate any clean champion candidate.",
        "",
        f"CSV: {output_csv}",
        f"TXT: {output_txt}",
        "Safety: read-only except reports. No training. No champion mutation. No promotion. No orders.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    run_dir = latest_walkforward_run()
    output_csv, output_txt = output_paths(run_dir)
    leaderboard = read_csv(run_dir / "contract_leaderboard.csv")
    candidates = read_csv(run_dir / "candidates.csv")
    selected_by_window = read_csv(run_dir / "selected_by_window.csv")
    test_scores = read_csv(run_dir / "test_scores.csv")
    probes = load_probe_decisions()
    candidates = annotate_probe_matches(candidates, probes)
    report = build_survivors(leaderboard, candidates, test_scores, selected_by_window)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
    text = render_text(report, run_dir, output_csv, output_txt)
    output_txt.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
