from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken")
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()

CHAMPION_NAME = os.getenv(
    "RAWSEQ_SHADOW_CHAMPION_NAME",
    "rawseq_fade_ma_distance_60_h2x2_v1_seed906",
)

CHAMPION_DIR = ROOT / "data" / "paper_champions" / CHAMPION_NAME
MODEL_PATH = Path(os.getenv("RAWSEQ_LOAD_MODEL_PATH", CHAMPION_DIR / "model.json"))

SOURCE_PATH = os.getenv(
    "RAWSEQ_SOURCE_PATH",
    str(ROOT / "data" / "realtime" / PRIMARY_VENUE / f"{SYMBOL}_10s_flow.csv"),
)

BUCKET_SECONDS = os.getenv("RAWSEQ_BUCKET_SECONDS", "10")
SEQ_LEN = os.getenv("RAWSEQ_LEN", "60")
INPUT_FEATURE = os.getenv("RAWSEQ_INPUT_FEATURE", "ma_distance")
MA_WINDOW = os.getenv("RAWSEQ_MA_WINDOW", "60")
HIDDEN = os.getenv("RAWSEQ_HIDDEN", "2,2")

POLICY = os.getenv("RAWSEQ_SHADOW_POLICY", "inverse_gt_0.2")
THRESHOLD = float(os.getenv("RAWSEQ_SHADOW_THRESHOLD_BPS", "0.2"))

SCRIPT = ROOT / "scripts" / "tiny_price_rawseq_path_v1.py"

EVAL_PATH = ROOT / "data" / "realtime" / PRIMARY_VENUE / f"{SYMBOL}_tiny_price_rawseq_path_v1_shadow_evaluation.csv"
ANNOTATED_PATH = ROOT / "data" / "realtime" / PRIMARY_VENUE / f"{SYMBOL}_tiny_price_prediction_evaluation_rows_with_rawseq_path_v1.csv"
ROWS_PATH = ROOT / "data" / "realtime" / PRIMARY_VENUE / f"{SYMBOL}_tiny_price_rawseq_path_v1_rows.csv"

SHADOW_ROOT = CHAMPION_DIR / "frozen_shadow_runs"
SUMMARY_PATH = CHAMPION_DIR / "frozen_shadow_summary.csv"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def max_dip_bps(returns: pd.Series) -> float:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return math.nan
    c = returns.cumsum()
    return float((c - c.cummax()).min())


def evaluate_policy(annotated_path: Path) -> dict:
    df = pd.read_csv(annotated_path, low_memory=False)

    # Keep this consistent with the training harness evaluation convention.
    test = df.iloc[int(len(df) * 0.80):].copy()
    if "timestamp" in df.columns:
        ts_all = pd.to_numeric(df["timestamp"], errors="coerce")
        ts_test = pd.to_numeric(test["timestamp"], errors="coerce")

        source_timestamp_min = float(ts_all.min())
        source_timestamp_max = float(ts_all.max())
        test_timestamp_min = float(ts_test.min())
        test_timestamp_max = float(ts_test.max())
    else:
        source_timestamp_min = math.nan
        source_timestamp_max = math.nan
        test_timestamp_min = math.nan
        test_timestamp_max = math.nan
    pred = pd.to_numeric(test["rawseq_path_pred_horizon_return_bps"], errors="coerce")
    actual = pd.to_numeric(test["rawseq_path_actual_horizon_return_bps"], errors="coerce")

    if POLICY.startswith("inverse_gt_"):
        returns = -actual[pred > THRESHOLD]
    elif POLICY.startswith("direct_gt_"):
        returns = actual[pred > THRESHOLD]
    elif POLICY.startswith("inverse_directional_abs_gt_"):
        mask = pred.abs() > THRESHOLD
        returns = -np.sign(pred[mask]) * actual[mask]
    else:
        raise SystemExit(f"Unsupported RAWSEQ_SHADOW_POLICY={POLICY}")

    returns = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "champion_name": CHAMPION_NAME,
        "symbol": SYMBOL,
        "mode": "frozen_inference_only",
        "source_path": SOURCE_PATH,
        "model_path": str(MODEL_PATH),
        "primary_venue": PRIMARY_VENUE,
        "bucket_seconds": int(BUCKET_SECONDS),
        "seq_len": int(SEQ_LEN),
        "input_feature": INPUT_FEATURE,
        "ma_window": int(MA_WINDOW),
        "hidden": HIDDEN,
        "policy": POLICY,
        "threshold_bps": THRESHOLD,
        "rows": int(len(returns)),
        "avg_return_bps": float(returns.mean()) if len(returns) else math.nan,
        "cumulative_return_bps": float(returns.sum()) if len(returns) else math.nan,
        "win_rate": float((returns > 0).mean()) if len(returns) else math.nan,
        "max_dip_bps": max_dip_bps(returns),
        "paper_only": True,
        "promotion": False,
        "orders": False,
        "source_timestamp_min": source_timestamp_min,
        "source_timestamp_max": source_timestamp_max,
        "test_timestamp_min": test_timestamp_min,
        "test_timestamp_max": test_timestamp_max,
    }


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing champion model.json: {MODEL_PATH}")

    if not Path(SOURCE_PATH).exists():
        raise SystemExit(f"Missing source file: {SOURCE_PATH}")

    SHADOW_ROOT.mkdir(parents=True, exist_ok=True)

    stamp = now_stamp()
    run_dir = SHADOW_ROOT / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"

    env = os.environ.copy()
    env.update(
        {
            "SYMBOL": SYMBOL,
            "PRIMARY_VENUE": PRIMARY_VENUE,
            "RAWSEQ_SOURCE_PATH": SOURCE_PATH,
            "RAWSEQ_BUCKET_SECONDS": BUCKET_SECONDS,
            "RAWSEQ_LEN": SEQ_LEN,
            "RAWSEQ_INPUT_FEATURE": INPUT_FEATURE,
            "RAWSEQ_MA_WINDOW": MA_WINDOW,
            "RAWSEQ_HIDDEN": HIDDEN,
            "RAWSEQ_INCLUDE_WINDOW_GUIDE": "false",
            "RAWSEQ_INFERENCE_ONLY": "true",
            "RAWSEQ_LOAD_MODEL_PATH": str(MODEL_PATH),
        }
    )

    print(f"Running frozen shadow: {CHAMPION_NAME}")
    print(f"Symbol: {SYMBOL}")
    print(f"Source: {SOURCE_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Policy: {POLICY}, threshold={THRESHOLD}")

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(proc.stdout)
        log.write(proc.stdout)

    if proc.returncode != 0:
        raise SystemExit(f"Inference script failed with exit code {proc.returncode}")


    if SYMBOL not in ANNOTATED_PATH.name or SYMBOL not in EVAL_PATH.name or SYMBOL not in ROWS_PATH.name:
        raise SystemExit(
            f"Output path symbol mismatch. SYMBOL={SYMBOL}, "
            f"annotated={ANNOTATED_PATH}, eval={EVAL_PATH}, rows={ROWS_PATH}"
        )
    for src, name in [
        (EVAL_PATH, "evaluation.csv"),
        (ANNOTATED_PATH, "annotated.csv"),
        (ROWS_PATH, "rows.csv"),
    ]:
        if src.exists():
            shutil.copy2(src, run_dir / name)
        else:
            raise SystemExit(f"Expected output missing: {src}")

    result = evaluate_policy(run_dir / "annotated.csv")
    result["run_dir"] = str(run_dir)

    pd.DataFrame([result]).to_csv(run_dir / "result.csv", index=False)

    if SUMMARY_PATH.exists():
        old = pd.read_csv(SUMMARY_PATH, low_memory=False)
        summary = pd.concat([old, pd.DataFrame([result])], ignore_index=True, sort=False)
    else:
        summary = pd.DataFrame([result])

    summary.to_csv(SUMMARY_PATH, index=False)

    print()
    print("Frozen shadow result")
    print(f"rows: {result['rows']}")
    print(f"avg_return_bps: {result['avg_return_bps']}")
    print(f"cumulative_return_bps: {result['cumulative_return_bps']}")
    print(f"win_rate: {result['win_rate']}")
    print(f"max_dip_bps: {result['max_dip_bps']}")
    print(f"Archived: {run_dir}")
    print(f"Summary: {SUMMARY_PATH}")
    print("Safety: paper-only. No promotion. No orders.")


if __name__ == "__main__":
    main()