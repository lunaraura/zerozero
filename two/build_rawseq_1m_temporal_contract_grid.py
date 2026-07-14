#!/usr/bin/env python3
"""Build one-minute temporal contract grid for rawseq baseline scouting."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import DEFAULT_OUTPUT_ROOT, SAFETY_FLAGS, env_path, now_stamp, parse_int_list, stable_hash, write_csv, write_json

DEFAULT_FEATURES = [
    "signed_bucket_return_bps",
    "rolling_range_bps",
    "rolling_volatility_bps",
    "distance_to_recent_high_bps",
    "distance_to_recent_low_bps",
    "close_to_ema_bps",
    "ema_slope_bps",
    "candle_range_bps",
    "candle_body_bps",
    "upper_wick_bps",
    "lower_wick_bps",
]
DEFAULT_HIDDENS = ["2,2", "4,4", "8,8"]
OUTPUT_DIMS = {
    "future_return_path": lambda n: n,
    "future_low_from_now_bps_path": lambda n: n,
    "future_range_envelope_path": lambda n: n * 2,
    "downside_event_0p5vol": lambda n: 1,
}


def slug_token(value: str) -> str:
    return value.replace(",", "x").replace("_from_now_bps", "").replace("future_", "f").replace("_path", "")


def contract_row(**kwargs: Any) -> dict[str, Any]:
    base = {
        **kwargs,
        "input_cadence_seconds": 60,
        "support_status": "supported",
        "unsupported_reason": "",
        "frozen_candidate_weights_reused": False,
        "frozen_candidate_thresholds_reused": False,
        "frozen_candidate_calibration_reused": False,
        **SAFETY_FLAGS,
    }
    readable = "_".join(
        str(base.get(key, ""))
        for key in [
            "contract_family",
            "input_seq_len",
            "output_seq_len",
            "output_label",
            "input_feature",
            "feature_window_minutes",
            "hidden",
        ]
    )
    base["full_contract_slug"] = readable
    base["contract_hash"] = stable_hash(base)[:12]
    base["contract_slug"] = f"{base['contract_family']}_i{base['input_seq_len']}_o{base['output_seq_len']}_{slug_token(base['output_label'])}_fw{base['feature_window_minutes']}_h{slug_token(str(base['hidden']))}_{base['contract_hash']}"
    return base


def build_grid() -> list[dict[str, Any]]:
    symbol = os.getenv("RAWSEQ_1M_SYMBOL", "SOLUSDT").strip() or "SOLUSDT"
    venue = os.getenv("RAWSEQ_1M_VENUE", "binance_public").strip() or "unknown"
    feature_windows = parse_int_list(os.getenv("RAWSEQ_1M_FEATURE_WINDOWS", ""), [15, 30, 60, 240])
    hiddens = [x.strip() for x in os.getenv("RAWSEQ_1M_HIDDENS", "").split(";") if x.strip()] or DEFAULT_HIDDENS
    input_features = [x.strip() for x in os.getenv("RAWSEQ_1M_INPUT_FEATURES", "").split(",") if x.strip()] or DEFAULT_FEATURES
    rows: list[dict[str, Any]] = []

    for vol_window in feature_windows:
        for horizon in [1, 2, 4, 8]:
            rows.append(
                contract_row(
                    symbol=symbol,
                    venue=venue,
                    contract_family="A_elapsed_downside",
                    input_seq_len=vol_window,
                    input_window_minutes=vol_window,
                    output_seq_len=1,
                    output_window_minutes=horizon,
                    target_horizons_minutes=str(horizon),
                    input_feature="causal_feature_bank",
                    feature_window_minutes=vol_window,
                    output_label="downside_event_0p5vol",
                    output_dim=1,
                    hidden="cpu_baseline",
                    model_family="cpu_downside_risk",
                )
            )
    for output_label in ["future_return_path", "future_low_from_now_bps_path", "future_range_envelope_path"]:
        for feature in input_features:
            for feature_window in feature_windows:
                for hidden in hiddens:
                    output_len = 60
                    rows.append(
                        contract_row(
                            symbol=symbol,
                            venue=venue,
                            contract_family="B_same_shape_60x60",
                            input_seq_len=60,
                            input_window_minutes=60,
                            output_seq_len=output_len,
                            output_window_minutes=output_len,
                            target_horizons_minutes="1..60",
                            input_feature=feature,
                            feature_window_minutes=feature_window,
                            output_label=output_label,
                            output_dim=OUTPUT_DIMS[output_label](output_len),
                            hidden=hidden,
                            model_family="rawseq_new_weights",
                        )
                    )
    for input_len in [15, 30, 60]:
        for output_len in [8, 15, 30]:
            for output_label in ["future_low_from_now_bps_path", "future_range_envelope_path"]:
                for feature in input_features:
                    for feature_window in feature_windows:
                        for hidden in hiddens:
                            rows.append(
                                contract_row(
                                    symbol=symbol,
                                    venue=venue,
                                    contract_family="C_compact_elapsed_path",
                                    input_seq_len=input_len,
                                    input_window_minutes=input_len,
                                    output_seq_len=output_len,
                                    output_window_minutes=output_len,
                                    target_horizons_minutes=f"1..{output_len}",
                                    input_feature=feature,
                                    feature_window_minutes=feature_window,
                                    output_label=output_label,
                                    output_dim=OUTPUT_DIMS[output_label](output_len),
                                    hidden=hidden,
                                    model_family="rawseq_new_weights",
                                )
                            )
    return rows


def main() -> int:
    out_root = env_path("RAWSEQ_1M_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    out_dir = Path(os.getenv("RAWSEQ_1M_GRID_OUTPUT_DIR", "").strip() or out_root / f"rawseq_1m_contract_grid_{now_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=False)
    rows = build_grid()
    payload = {
        "created_at": now_stamp(),
        "row_count": len(rows),
        "families": sorted(set(row["contract_family"] for row in rows)),
        "safety": SAFETY_FLAGS,
        "contract_grid_sha256": stable_hash(rows),
    }
    write_csv(out_dir / "one_minute_contract_grid.csv", rows)
    write_json(out_dir / "one_minute_contract_grid.json", payload)
    lines = [
        "Rawseq 1m temporal contract grid",
        f"Output: {out_dir}",
        f"Rows: {len(rows)}",
        f"Families: {', '.join(payload['families'])}",
        "Status: supported contracts are research-only and require newly initialized weights.",
        "Safety: paper_only=true, orders=false, promotion=false, champion_mutation=false.",
    ]
    (out_dir / "one_minute_contract_grid.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
