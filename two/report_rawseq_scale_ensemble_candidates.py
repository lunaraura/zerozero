#!/usr/bin/env python3
"""Build a temporal-scale leaderboard for rawseq probe candidates.

Read-only except for writing scale ensemble report outputs.
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
PROBE_ROOT = Path(
    os.getenv(
        "RAWSEQ_ENSEMBLE_PROBE_ROOT",
        str(PROJECT_ROOT / "data" / "research" / "rawseq_candidate_shadow_probes"),
    )
)
if not PROBE_ROOT.is_absolute():
    PROBE_ROOT = PROJECT_ROOT / PROBE_ROOT

MIN_STATUS = os.getenv("RAWSEQ_ENSEMBLE_MIN_STATUS", "research_candidate").strip().lower()
DEFAULT_OUTPUT_PATH = PROBE_ROOT / "scale_ensemble_candidates.csv"
OUTPUT_PATH = Path(os.getenv("RAWSEQ_ENSEMBLE_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH
TEXT_OUTPUT_PATH = OUTPUT_PATH.with_suffix(".txt")

STATUS_RANK = {
    "all": -1,
    "missing_decision": 0,
    "reject": 0,
    "research_candidate": 1,
    "clean_champion_candidate": 2,
}
GROUP_COLS = [
    "input_feature",
    "input_stride",
    "output_stride",
    "seq_len",
    "bucket_seconds",
    "hidden",
    "source_path_basename",
]
SHORT_INPUT_SPAN_SECONDS = 15 * 60
MEDIUM_INPUT_SPAN_SECONDS = 2 * 60 * 60


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def pass_fail(value: bool) -> str:
    return "pass" if value else "fail"


def compact_status(value: Any) -> str:
    text = safe_str(value)
    if text == "missing":
        return "miss"
    return text or "none"


def fmt_float(value: Any) -> str:
    number = safe_float(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6g}"


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, str(exc)
    if not isinstance(payload, dict):
        return {}, "root is not an object"
    return payload, ""


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def first_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def value_from(*values: Any, default: str = "") -> str:
    for value in values:
        text = safe_str(value)
        if text:
            return text
    return default


def int_text(value: Any, default: str = "") -> str:
    text = value_from(value, default=default)
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def status_rank(status: str) -> int:
    return STATUS_RANK.get(status.strip().lower(), -1)


def minimum_rank() -> int:
    if MIN_STATUS not in STATUS_RANK:
        raise SystemExit(
            "RAWSEQ_ENSEMBLE_MIN_STATUS must be one of: "
            + ", ".join(sorted(STATUS_RANK))
        )
    return STATUS_RANK[MIN_STATUS]


def discover_probe_dirs() -> list[Path]:
    if not PROBE_ROOT.exists():
        return []
    dirs = {path.parent for path in PROBE_ROOT.glob("*/model_contract.json")}
    return sorted(dirs, key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)


def nearest_row(frame: pd.DataFrame, threshold: float, cost: float) -> pd.Series | None:
    if frame.empty or "threshold_bps" not in frame.columns or "cost_bps" not in frame.columns:
        return None
    working = frame.copy()
    working["_threshold"] = pd.to_numeric(working["threshold_bps"], errors="coerce")
    working["_cost"] = pd.to_numeric(working["cost_bps"], errors="coerce")
    working["_distance"] = (working["_threshold"] - threshold).abs() + (working["_cost"] - cost).abs()
    working = working[working["_distance"].notna()]
    if working.empty:
        return None
    return working.sort_values("_distance").iloc[0]


def fixed_cost_cum_net(
    decision: dict[str, Any],
    cost_summary: pd.DataFrame,
    threshold: float,
    cost: float,
) -> float:
    label = f"cost_{str(cost).replace('.', 'p')}_cum_net_bps"
    if label in decision:
        value = safe_float(decision.get(label))
        if math.isfinite(value):
            return value
    row = nearest_row(cost_summary, threshold, cost)
    if row is None:
        return math.nan
    return safe_float(row.get("cum_net_bps"))


def rolling_positive_fraction(
    decision: dict[str, Any],
    rolling: pd.DataFrame,
    threshold: float,
    cost: float,
    hours: float,
) -> float:
    key = f"positive_{hours:g}h_window_fraction"
    if key in decision:
        value = safe_float(decision.get(key))
        if math.isfinite(value):
            return value
    if rolling.empty or "window_hours" not in rolling.columns:
        return math.nan
    working = rolling.copy()
    working["_window_hours"] = pd.to_numeric(working["window_hours"], errors="coerce")
    working["_threshold"] = pd.to_numeric(working.get("threshold_bps"), errors="coerce")
    working["_cost"] = pd.to_numeric(working.get("cost_bps"), errors="coerce")
    subset = working[
        (working["_window_hours"].sub(hours).abs() < 1e-9)
        & (working["_threshold"].sub(threshold).abs() < 1e-9)
        & (working["_cost"].sub(cost).abs() < 1e-9)
    ]
    if subset.empty:
        return math.nan
    cum = pd.to_numeric(subset.get("cum_net_bps"), errors="coerce")
    cum = cum[cum.notna()]
    if cum.empty:
        return math.nan
    return float((cum > 0.0).mean())


def dynamic_survival(dynamic: pd.DataFrame) -> dict[str, Any]:
    if dynamic.empty or "scenario" not in dynamic.columns:
        return {
            "dynamic_cost_survival": "missing",
            "dynamic_cost_survival_bool": False,
            "dynamic_base_scenario": "",
            "dynamic_base_cum_net_bps": math.nan,
            "dynamic_positive_scenarios": 0,
            "dynamic_scenarios": 0,
            "dynamic_worst_cum_net_bps": math.nan,
        }
    working = dynamic.copy()
    working["cum_net_bps"] = pd.to_numeric(working.get("cum_net_bps"), errors="coerce")
    scenarios = working["scenario"].astype(str)
    non_fixed = working[~scenarios.str.startswith("fixed_", na=False)].copy()
    if non_fixed.empty:
        return {
            "dynamic_cost_survival": "missing",
            "dynamic_cost_survival_bool": False,
            "dynamic_base_scenario": "",
            "dynamic_base_cum_net_bps": math.nan,
            "dynamic_positive_scenarios": 0,
            "dynamic_scenarios": 0,
            "dynamic_worst_cum_net_bps": math.nan,
        }
    preferred = non_fixed[non_fixed["scenario"].astype(str).eq("half_spread_plus_0_05_bps")]
    base = preferred.iloc[0] if not preferred.empty else non_fixed.iloc[0]
    base_cum = safe_float(base.get("cum_net_bps"))
    positive = int((non_fixed["cum_net_bps"] > 0.0).sum())
    total = int(len(non_fixed))
    status = "pass" if math.isfinite(base_cum) and base_cum > 0.0 else "fail"
    return {
        "dynamic_cost_survival": status,
        "dynamic_cost_survival_bool": status == "pass",
        "dynamic_base_scenario": safe_str(base.get("scenario")),
        "dynamic_base_cum_net_bps": base_cum,
        "dynamic_positive_scenarios": positive,
        "dynamic_scenarios": total,
        "dynamic_worst_cum_net_bps": safe_float(non_fixed["cum_net_bps"].min()),
    }


def scale_bucket(input_span_seconds: int) -> str:
    if input_span_seconds <= SHORT_INPUT_SPAN_SECONDS:
        return "short"
    if input_span_seconds <= MEDIUM_INPUT_SPAN_SECONDS:
        return "medium"
    return "long"


def ensemble_role(
    eligible: bool,
    decision: str,
    cum_net_bps: float,
    input_span_seconds: int,
    output_span_seconds: int,
    dynamic_status: str,
    cost_025_survival: bool,
) -> str:
    if not eligible or decision == "reject" or not math.isfinite(cum_net_bps) or cum_net_bps <= 0.0:
        return "reject"
    if (
        output_span_seconds <= 10 * 60
        and cost_025_survival
        and dynamic_status in {"pass", "missing"}
    ):
        return "execution_filter"
    bucket = scale_bucket(input_span_seconds)
    if bucket == "short":
        return "short_signal"
    if bucket == "medium":
        return "medium_context"
    return "long_regime_filter"


def ensemble_score(row: dict[str, Any]) -> float:
    score = safe_float(row.get("cum_net_bps"), 0.0)
    score += 250.0 * safe_float(row.get("positive_1h_rolling_fraction"), 0.0)
    score += 150.0 * safe_float(row.get("positive_3h_rolling_fraction"), 0.0)
    score -= 75.0 * max(0.0, safe_float(row.get("max_dip_to_cum_net_ratio"), 0.0) - 1.0)
    if row.get("cost_0_25_survival"):
        score += 125.0
    else:
        score -= 125.0
    if row.get("dynamic_cost_survival") == "pass":
        score += 125.0
    elif row.get("dynamic_cost_survival") == "fail":
        score -= 125.0
    if not row.get("eligible_min_status"):
        score -= 500.0
    if row.get("ensemble_role") == "reject":
        score -= 250.0
    return float(score)


def parse_probe(probe_dir: Path, min_rank: int) -> dict[str, Any]:
    contract, contract_issue = read_json(probe_dir / "model_contract.json")
    decision_frame = read_csv(probe_dir / "decision_summary.csv")
    decision_row = first_row(decision_frame)
    cost_summary = read_csv(probe_dir / "cost_threshold_summary.csv")
    rolling = read_csv(probe_dir / "rolling_summary.csv")
    dynamic = read_csv(probe_dir / "dynamic_cost_summary.csv")

    decision = value_from(decision_row.get("decision"), default="missing_decision")
    decision_threshold = safe_float(decision_row.get("decision_threshold_bps"), 0.1)
    decision_cost = safe_float(decision_row.get("decision_cost_bps"), 0.1)
    fixed_row = nearest_row(cost_summary, decision_threshold, decision_cost)

    symbol = value_from(decision_row.get("symbol"), contract.get("symbol"))
    venue = value_from(decision_row.get("venue"), contract.get("venue"))
    input_feature = value_from(decision_row.get("input_feature"), contract.get("input_feature"))
    source_basename = value_from(
        decision_row.get("source_path_basename"),
        contract.get("source_path_basename"),
        Path(safe_str(contract.get("source_path")).replace("\\", "/")).name,
    )
    bucket_seconds = int_text(decision_row.get("bucket_seconds") or contract.get("bucket_seconds"), "1")
    seq_len = int_text(decision_row.get("seq_len") or contract.get("seq_len"), "60")
    input_stride = int_text(decision_row.get("input_stride") or contract.get("input_stride"), "1")
    output_stride = int_text(decision_row.get("output_stride") or contract.get("output_stride"), "1")
    hidden = value_from(decision_row.get("hidden"), contract.get("hidden"))

    input_span_seconds = safe_int(
        contract.get("input_span_seconds"),
        safe_int(seq_len, 0) * safe_int(input_stride, 1) * safe_int(bucket_seconds, 1),
    )
    output_span_seconds = safe_int(
        contract.get("output_span_seconds"),
        safe_int(seq_len, 0) * safe_int(output_stride, 1) * safe_int(bucket_seconds, 1),
    )

    selected_rows = safe_int(
        decision_row.get("selected_rows"),
        safe_int(fixed_row.get("selected_rows") if fixed_row is not None else 0),
    )
    cum_net = safe_float(
        decision_row.get("cum_net_bps"),
        safe_float(fixed_row.get("cum_net_bps") if fixed_row is not None else math.nan),
    )
    max_dip_ratio = safe_float(decision_row.get("max_dip_to_cum_net_ratio"))
    positive_1h = rolling_positive_fraction(decision_row, rolling, decision_threshold, decision_cost, 1.0)
    positive_3h = rolling_positive_fraction(decision_row, rolling, decision_threshold, decision_cost, 3.0)
    cost_025_cum = fixed_cost_cum_net(decision_row, cost_summary, decision_threshold, 0.25)
    cost_025_survival = math.isfinite(cost_025_cum) and cost_025_cum > 0.0
    dyn = dynamic_survival(dynamic)

    eligible = status_rank(decision) >= min_rank
    role = ensemble_role(
        eligible,
        decision,
        cum_net,
        input_span_seconds,
        output_span_seconds,
        dyn["dynamic_cost_survival"],
        cost_025_survival,
    )
    issues = []
    if contract_issue:
        issues.append(f"contract={contract_issue}")
    if decision_frame.empty:
        issues.append("missing_decision_summary")
    if cost_summary.empty:
        issues.append("missing_cost_threshold_summary")
    if rolling.empty:
        issues.append("missing_rolling_summary")
    if dynamic.empty:
        issues.append("missing_dynamic_cost_summary")
    if not eligible:
        issues.append(f"below_min_status={MIN_STATUS}")
    if not math.isfinite(cum_net) or cum_net <= 0.0:
        issues.append("nonpositive_decision_cum_net")

    row = {
        "probe_dir": str(probe_dir.resolve()),
        "probe_folder": probe_dir.name,
        "model_path": value_from(decision_row.get("model_path"), contract.get("model_path")),
        "symbol": symbol,
        "venue": venue,
        "input_feature": input_feature,
        "input_stride": input_stride,
        "output_stride": output_stride,
        "seq_len": seq_len,
        "bucket_seconds": bucket_seconds,
        "hidden": hidden,
        "source_path_basename": source_basename,
        "input_span_seconds": input_span_seconds,
        "output_span_seconds": output_span_seconds,
        "scale_bucket": scale_bucket(input_span_seconds),
        "decision": decision,
        "eligible_min_status": eligible,
        "selected_rows": selected_rows,
        "decision_threshold_bps": decision_threshold,
        "decision_cost_bps": decision_cost,
        "avg_net_bps": safe_float(decision_row.get("avg_net_bps")),
        "cum_net_bps": cum_net,
        "max_dip_to_cum_net_ratio": max_dip_ratio,
        "positive_1h_rolling_fraction": positive_1h,
        "positive_3h_rolling_fraction": positive_3h,
        "cost_0_25_cum_net_bps": cost_025_cum,
        "cost_0_25_survival": cost_025_survival,
        "ensemble_role": role,
        "exclude_reason": "; ".join(issues),
        "report_created_at": now_iso(),
        **dyn,
    }
    row["ensemble_score"] = ensemble_score(row)
    row["contract_group"] = "|".join(safe_str(row[column]) for column in GROUP_COLS)
    return row


def add_group_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["contract_group_candidates"] = out.groupby(GROUP_COLS, dropna=False)["probe_dir"].transform("count")
    ranks = []
    for _, group in out.groupby(GROUP_COLS, dropna=False, sort=False):
        ranked = group.sort_values(
            ["eligible_min_status", "ensemble_score", "cum_net_bps", "selected_rows"],
            ascending=[False, False, False, False],
        )
        for rank, idx in enumerate(ranked.index, start=1):
            ranks.append((idx, rank))
    rank_series = pd.Series({idx: rank for idx, rank in ranks})
    out["contract_group_rank"] = out.index.map(rank_series).fillna(0).astype(int)
    out["best_in_contract_group"] = out["contract_group_rank"].eq(1)
    out = out.sort_values(
        ["eligible_min_status", "ensemble_role", "ensemble_score", "cum_net_bps"],
        ascending=[False, True, False, False],
    )
    return out


def best_by_scale(frame: pd.DataFrame, scale: str) -> pd.Series | None:
    subset = frame[
        frame["eligible_min_status"].astype(bool)
        & frame["scale_bucket"].eq(scale)
        & ~frame["ensemble_role"].eq("reject")
    ].copy()
    if subset.empty:
        return None
    return subset.sort_values(["ensemble_score", "cum_net_bps", "selected_rows"], ascending=[False, False, False]).iloc[0]


def row_label(row: pd.Series | None) -> str:
    if row is None:
        return "none"
    return (
        f"{row['probe_folder']} role={row['ensemble_role']} "
        f"feature={row['input_feature']} hidden={row['hidden']} "
        f"stride={row['input_stride']}/{row['output_stride']} "
        f"span={row['input_span_seconds']}s cum={fmt_float(row['cum_net_bps'])} "
        f"score={fmt_float(row['ensemble_score'])}"
    )


def render_text(frame: pd.DataFrame) -> str:
    eligible = frame[frame["eligible_min_status"].astype(bool)].copy() if not frame.empty else pd.DataFrame()
    usable = eligible[~eligible["ensemble_role"].eq("reject")].copy() if not eligible.empty else pd.DataFrame()
    best_short = best_by_scale(frame, "short")
    best_medium = best_by_scale(frame, "medium")
    best_long = best_by_scale(frame, "long")

    scale_counts = usable["scale_bucket"].value_counts().to_dict() if not usable.empty else {}
    has_short = best_short is not None
    has_context = best_medium is not None or best_long is not None
    viable = has_short and has_context
    if viable:
        viable_text = "yes: at least one short-scale signal and one medium/long context candidate are eligible"
    elif usable.empty:
        viable_text = "no: no eligible non-reject candidates"
    elif not has_short:
        viable_text = "no: missing an eligible short-scale signal"
    else:
        viable_text = "no: missing an eligible medium/long context candidate"

    lines = [
        "Rawseq Scale Ensemble Candidates",
        "",
        f"Probe root: {PROBE_ROOT}",
        f"Minimum status: {MIN_STATUS}",
        f"Probe candidates scanned: {len(frame)}",
        f"Contract groups: {frame['contract_group'].nunique() if not frame.empty else 0}",
        f"Eligible non-reject candidates: {len(usable)}",
        f"Usable scale counts: short={scale_counts.get('short', 0)} "
        f"medium={scale_counts.get('medium', 0)} long={scale_counts.get('long', 0)}",
        "",
        "Recommendations",
        f"  best short-scale candidate: {row_label(best_short)}",
        f"  best medium-scale candidate: {row_label(best_medium)}",
        f"  best long-scale candidate: {row_label(best_long)}",
        f"  ensemble currently viable: {viable_text}",
        "",
        "Leaderboard",
        "  rank role                 scale decision                 score cum_net rows pos1h pos3h c025 dyn folder",
        "  ---- -------------------- ----- ------------------------ ----- ------- ---- ----- ----- ---- --- ------",
    ]
    if usable.empty:
        lines.append("  none")
    else:
        for rank, (_, row) in enumerate(
            usable.sort_values(["ensemble_score", "cum_net_bps"], ascending=[False, False]).head(25).iterrows(),
            start=1,
        ):
            lines.append(
                "  "
                + " ".join(
                    [
                        str(rank).rjust(4),
                        safe_str(row["ensemble_role"])[:20].ljust(20),
                        safe_str(row["scale_bucket"])[:5].ljust(5),
                        safe_str(row["decision"])[:24].ljust(24),
                        fmt_float(row["ensemble_score"]).rjust(5),
                        fmt_float(row["cum_net_bps"]).rjust(7),
                        str(safe_int(row["selected_rows"])).rjust(4),
                        fmt_float(row["positive_1h_rolling_fraction"]).rjust(5),
                        fmt_float(row["positive_3h_rolling_fraction"]).rjust(5),
                        pass_fail(bool(row["cost_0_25_survival"])).rjust(4),
                        compact_status(row["dynamic_cost_survival"]).rjust(4),
                        safe_str(row["probe_folder"])[:80],
                    ]
                )
            )

    excluded = frame[frame["ensemble_role"].eq("reject") | ~frame["eligible_min_status"].astype(bool)].copy()
    lines += ["", "Candidates To Exclude"]
    if excluded.empty:
        lines.append("  none")
    else:
        excluded = excluded.sort_values(["eligible_min_status", "cum_net_bps"], ascending=[True, True])
        for _, row in excluded.head(30).iterrows():
            reason = safe_str(row.get("exclude_reason")) or "role=reject"
            lines.append(
                f"  {row['probe_folder']} decision={row['decision']} "
                f"cum={fmt_float(row['cum_net_bps'])} reason={reason[:160]}"
            )
        if len(excluded) > 30:
            lines.append(f"  ... {len(excluded) - 30} more")

    lines += [
        "",
        f"CSV: {OUTPUT_PATH}",
        f"TXT: {TEXT_OUTPUT_PATH}",
        "Safety: read-only except report outputs. No training. No champion mutation. No promotion. No orders.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    min_rank = minimum_rank()
    rows = [parse_probe(path, min_rank) for path in discover_probe_dirs()]
    frame = add_group_ranks(pd.DataFrame(rows))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    text = render_text(frame)
    TEXT_OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
