import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
INPUT_ROOT = Path(os.getenv("SIM_COMBINE_INPUT_ROOT", PROJECT_ROOT / "data" / "simulated_runs"))
OUTPUT_DIR = Path(os.getenv("SIM_COMBINE_OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime" / "simulated"))
RUN_GAP_SECONDS = int(os.getenv("SIM_COMBINE_RUN_GAP_SECONDS", "3600"))
START_TIMESTAMP_MS = int(os.getenv("SIM_COMBINE_START_TIMESTAMP_MS", "1767225600000"))

if not INPUT_ROOT.is_absolute():
    INPUT_ROOT = PROJECT_ROOT / INPUT_ROOT
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def ordered_columns(frame):
    hidden_columns = [column for column in frame.columns if str(column).startswith("hidden_")]
    grouping_columns = [
        column
        for column in ["simulation_run_id", "source_scenario", "source_seed"]
        if column in frame.columns
    ]
    visible_columns = [
        column
        for column in frame.columns
        if column not in grouping_columns and column not in hidden_columns
    ]
    return [*visible_columns, *grouping_columns, *hidden_columns]


def discover_runs():
    runs = []
    if not INPUT_ROOT.exists():
        return runs
    for scenario_dir in sorted(path for path in INPUT_ROOT.iterdir() if path.is_dir()):
        for seed_dir in sorted(path for path in scenario_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
            runs.append(
                {
                    "scenario": scenario_dir.name,
                    "seed": seed_dir.name.replace("seed_", ""),
                    "path": seed_dir,
                    "run_id": f"{scenario_dir.name}_{seed_dir.name}",
                }
            )
    return runs


def shift_run_timestamps(frame, start_timestamp_ms):
    frame = frame.copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if len(frame) == 0:
        return frame, start_timestamp_ms
    original_start = int(frame["timestamp"].iloc[0])
    shift = int(start_timestamp_ms) - original_start
    frame["timestamp"] = frame["timestamp"].astype("int64") + shift
    if "time" in frame.columns:
        frame["time"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    next_start = int(frame["timestamp"].iloc[-1]) + RUN_GAP_SECONDS * 1000
    return frame, next_start


def combine_kind(runs, suffix):
    combined = []
    next_start = START_TIMESTAMP_MS
    for run in runs:
        path = run["path"] / f"{SYMBOL}_{suffix}.csv"
        frame = read_csv(path)
        if len(frame) == 0:
            continue
        frame, next_start = shift_run_timestamps(frame, next_start)
        frame["simulation_run_id"] = run["run_id"]
        frame["source_scenario"] = run["scenario"]
        frame["source_seed"] = run["seed"]
        combined.append(frame)
    if not combined:
        return pd.DataFrame()
    output = pd.concat(combined, ignore_index=True)
    output = output.sort_values(["timestamp", "simulation_run_id"]).reset_index(drop=True)
    output = output[ordered_columns(output)]
    return output


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def main():
    runs = discover_runs()
    ten_second = combine_kind(runs, "10s_flow")
    one_minute = combine_kind(runs, "1m_flow")
    ten_second_path = OUTPUT_DIR / f"{SYMBOL}_10s_flow.csv"
    one_minute_path = OUTPUT_DIR / f"{SYMBOL}_1m_flow.csv"
    if len(ten_second):
        write_csv(ten_second, ten_second_path)
    if len(one_minute):
        write_csv(one_minute, one_minute_path)
    print("Combined simulated market dataset")
    print(f"SYMBOL: {SYMBOL}")
    print(f"Input root: {INPUT_ROOT}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Runs discovered: {len(runs)}")
    print(f"10s rows: {len(ten_second)} -> {ten_second_path}")
    print(f"1m rows: {len(one_minute)} -> {one_minute_path}")
    if len(ten_second):
        print(f"10s scenario counts: {ten_second['source_scenario'].value_counts().to_dict()}")
    if len(one_minute):
        print(f"1m scenario counts: {one_minute['source_scenario'].value_counts().to_dict()}")
    print("simulation_run_id/source_scenario columns are included.")
    print("Hidden diagnostic columns are written after visible and grouping columns.")
    print("Tiny-price target builders enforce simulation_run_id boundaries when this column exists.")
    print("Paper-only synthetic data utility. No trades/orders/private API.")


if __name__ == "__main__":
    main()
