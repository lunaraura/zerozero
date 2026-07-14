import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[3]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

from tiny.core import tiny_io, tiny_paths


PROJECT_ROOT = tiny_paths.ROOT
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SIMULATION_SCRIPTS_DIR = SCRIPTS_DIR / "tiny" / "simulation"


def parse_csv_env(name, default):
    value = os.getenv(name, default)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def env_int(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
SOURCE_KIND = "synthetic"
SCENARIOS = parse_csv_env(
    "SIM_ROLLING_SCENARIOS",
    "calm_chop,bullish_breakout,bearish_breakdown,liquidity_crisis,fakeout_reversal,news_shock_up,news_shock_down",
)
HOLDOUT_SCENARIOS = parse_csv_env("SIM_ROLLING_HOLDOUT_SCENARIOS", ",".join(SCENARIOS))
SIM_SECONDS = env_int("SIM_ROLLING_SIM_SECONDS", os.getenv("SIM_SECONDS", "3600"))
START_SEED = env_int("SIM_ROLLING_START_SEED", "1")
END_SEED_RAW = os.getenv("SIM_ROLLING_END_SEED", "").strip()
END_SEED = int(END_SEED_RAW) if END_SEED_RAW else None
MAX_CYCLES = env_int("SIM_ROLLING_MAX_CYCLES", "0")
CONTINUOUS = env_bool("SIM_ROLLING_CONTINUOUS", False)

ROLLING_MAX_RUNS = env_int("SIM_ROLLING_MAX_RUNS", "20")
ROLLING_MAX_10S_ROWS = env_int("SIM_ROLLING_MAX_10S_ROWS", "200000")
ROLLING_OUTPUT_VENUE = os.getenv("SIM_ROLLING_OUTPUT_VENUE", "simulated_rolling").strip().lower()
HOLDOUT_OUTPUT_VENUE = os.getenv("SIM_ROLLING_HOLDOUT_OUTPUT_VENUE", "sim_holdout").strip().lower()
TRAIN_EVERY_RUNS = max(1, env_int("SIM_ROLLING_TRAIN_EVERY_RUNS", "2"))
EVAL_EVERY_RUNS = max(1, env_int("SIM_ROLLING_EVAL_EVERY_RUNS", "10"))
HOLDOUT_SEED_OFFSET = env_int("SIM_ROLLING_HOLDOUT_SEED_OFFSET", "1000000")

FEATURE_GROUPS = os.getenv("PRICE_TINY_FEATURE_GROUPS", "base_tiny_price_v1").strip()
TARGET_SPEC = os.getenv("PRICE_TINY_TARGET_SPEC", "move_before_adverse_30s_net_aware").strip()
WALK_FORWARD_TARGET_SPECS = os.getenv(
    "PRICE_TINY_WALK_FORWARD_TARGET_SPECS",
    "move_before_adverse_30s_net_aware,instability_30s",
).strip()
MODEL_SPECS = os.getenv("PRICE_TINY_MODEL_SPECS", "ridge_logistic").strip()
TRAIN_CANDIDATE_ARTIFACT = env_bool("SIM_ROLLING_TRAIN_CANDIDATE_ARTIFACT", True)
TRAIN_CANDIDATE_TARGET_SPECS = parse_csv_env("SIM_ROLLING_CANDIDATE_TARGET_SPECS", TARGET_SPEC)

SIM_OUTPUT_DIR = resolve_path(os.getenv("SIM_OUTPUT_DIR", PROJECT_ROOT / "data" / "simulated"))
SIM_RUNS_ROOT = resolve_path(os.getenv("SIM_ROLLING_RUNS_ROOT", PROJECT_ROOT / "data" / "simulated_runs"))
HOLDOUT_RUNS_ROOT = resolve_path(
    os.getenv("SIM_ROLLING_HOLDOUT_RUNS_ROOT", PROJECT_ROOT / "data" / "simulated_holdout_runs")
)
REALTIME_ROOT = resolve_path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
ROLLING_VENUE_DIR = REALTIME_ROOT / ROLLING_OUTPUT_VENUE
HOLDOUT_VENUE_DIR = REALTIME_ROOT / HOLDOUT_OUTPUT_VENUE

RUN_GAP_SECONDS = env_int("SIM_COMBINE_RUN_GAP_SECONDS", "3600")
START_TIMESTAMP_MS = env_int("SIM_COMBINE_START_TIMESTAMP_MS", "1767225600000")

SUMMARY_PATH = ROLLING_VENUE_DIR / f"{SYMBOL}_sim_rolling_cycle_summary.csv"


def slugify(value, default="run"):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or default


def target_slug(target_spec):
    return slugify(target_spec, "target")


def read_csv(path, nrows=None):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, nrows=nrows)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def atomic_write_csv(frame, path):
    tiny_io.safe_write_csv_atomic(frame, path)


def ordered_columns(frame):
    hidden_columns = [column for column in frame.columns if str(column).startswith("hidden_")]
    grouping_columns = [
        column
        for column in ["simulation_run_id", "source_scenario", "source_seed", "source_kind", "symbol"]
        if column in frame.columns
    ]
    visible_columns = [
        column
        for column in frame.columns
        if column not in grouping_columns and column not in hidden_columns
    ]
    return [*visible_columns, *grouping_columns, *hidden_columns]


def count_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)


def run_python_script(script_name, env_updates, label):
    child_env = os.environ.copy()
    child_env.update({key: str(value) for key, value in env_updates.items()})
    child_env["PROMOTE_BEST"] = "false"
    child_env["PRICE_TINY_AUTO_REGISTER_CHALLENGERS"] = "false"
    child_env["TRAIN_PRICE_TINY_MODEL"] = "false"

    script_path = SIMULATION_SCRIPTS_DIR / script_name
    if not script_path.exists():
        script_path = SCRIPTS_DIR / script_name
    command = [sys.executable, str(script_path)]
    print(f"\n[{label}] running {script_name}")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=child_env,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())
    if completed.returncode != 0:
        print(f"[{label}] failed with exit code {completed.returncode}")
    return completed


def simulation_env(scenario, seed):
    return {
        "SYMBOL": SYMBOL,
        "SCENARIO": scenario,
        "SIM_SCENARIO": scenario,
        "RANDOM_SEED": seed,
        "SIM_SECONDS": SIM_SECONDS,
        "OUTPUT_DIR": SIM_OUTPUT_DIR,
        "PROMOTE_BEST": "false",
        "PRICE_TINY_AUTO_REGISTER_CHALLENGERS": "false",
    }


def run_simulation(scenario, seed, label):
    return run_python_script(
        "simulate_market_microstructure.py",
        simulation_env(scenario, seed),
        label,
    )


def add_run_metadata(frame, scenario, seed, run_id):
    frame = frame.copy()
    frame["simulation_run_id"] = run_id
    frame["source_scenario"] = scenario
    frame["source_seed"] = str(seed)
    frame["source_kind"] = SOURCE_KIND
    frame["symbol"] = SYMBOL
    return frame[ordered_columns(frame)]


def archive_simulation_outputs(scenario, seed, runs_root):
    scenario_slug = slugify(scenario, scenario)
    run_id = f"{scenario_slug}_seed_{seed}"
    source_dir = SIM_OUTPUT_DIR / scenario_slug
    target_dir = runs_root / scenario_slug / f"seed_{seed}"
    target_dir.mkdir(parents=True, exist_ok=True)

    archived = {}
    for suffix in ["10s_flow", "1m_flow"]:
        source_path = source_dir / f"{SYMBOL}_{suffix}.csv"
        target_path = target_dir / f"{SYMBOL}_{suffix}.csv"
        if not source_path.exists():
            archived[suffix] = {"path": target_path, "rows": 0, "error": f"missing {source_path}"}
            continue
        frame = read_csv(source_path)
        frame = add_run_metadata(frame, scenario_slug, seed, run_id)
        atomic_write_csv(frame, target_path)
        archived[suffix] = {"path": target_path, "rows": len(frame), "error": ""}

    return {
        "scenario": scenario_slug,
        "seed": str(seed),
        "path": target_dir,
        "run_id": run_id,
        "archived": archived,
        "mtime": time.time(),
    }


def discover_runs(input_root, allowed_scenarios):
    allowed = {slugify(item, item) for item in allowed_scenarios}
    runs = []
    if not input_root.exists():
        return runs
    for scenario_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        scenario = scenario_dir.name
        if allowed and scenario not in allowed:
            continue
        for seed_dir in sorted(path for path in scenario_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
            ten_second_path = seed_dir / f"{SYMBOL}_10s_flow.csv"
            one_minute_path = seed_dir / f"{SYMBOL}_1m_flow.csv"
            ten_second_rows = count_csv_rows(ten_second_path)
            if ten_second_rows <= 0:
                continue
            seed = seed_dir.name.replace("seed_", "")
            runs.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "path": seed_dir,
                    "run_id": f"{scenario}_seed_{seed}",
                    "mtime": max(
                        ten_second_path.stat().st_mtime if ten_second_path.exists() else 0,
                        one_minute_path.stat().st_mtime if one_minute_path.exists() else 0,
                    ),
                    "ten_second_rows": ten_second_rows,
                    "tail_10s_rows": 0,
                }
            )
    return runs


def select_recent_runs(runs):
    newest_first = sorted(runs, key=lambda item: (item["mtime"], item["scenario"], item["seed"]), reverse=True)
    selected = []
    total_rows = 0

    for run in newest_first:
        if ROLLING_MAX_RUNS > 0 and len(selected) >= ROLLING_MAX_RUNS:
            break
        rows = int(run.get("ten_second_rows", 0))
        if ROLLING_MAX_10S_ROWS > 0 and total_rows + rows > ROLLING_MAX_10S_ROWS:
            if not selected:
                run = dict(run)
                run["tail_10s_rows"] = ROLLING_MAX_10S_ROWS
                selected.append(run)
                total_rows += min(rows, ROLLING_MAX_10S_ROWS)
            break
        selected.append(dict(run))
        total_rows += rows

    selected = list(reversed(selected))
    return selected, total_rows


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


def read_run_frame(run, suffix):
    path = run["path"] / f"{SYMBOL}_{suffix}.csv"
    frame = read_csv(path)
    if len(frame) == 0:
        return frame
    tail_10s_rows = int(run.get("tail_10s_rows", 0) or 0)
    if tail_10s_rows > 0:
        if suffix == "10s_flow":
            frame = frame.tail(tail_10s_rows).copy()
        elif suffix == "1m_flow":
            one_minute_tail = max(1, int(tail_10s_rows / 6))
            frame = frame.tail(one_minute_tail).copy()
    frame["simulation_run_id"] = run["run_id"]
    frame["source_scenario"] = run["scenario"]
    frame["source_seed"] = str(run["seed"])
    frame["source_kind"] = SOURCE_KIND
    frame["symbol"] = SYMBOL
    return frame[ordered_columns(frame)]


def combine_kind(runs, suffix):
    combined = []
    next_start = START_TIMESTAMP_MS
    for run in runs:
        frame = read_run_frame(run, suffix)
        if len(frame) == 0:
            continue
        frame, next_start = shift_run_timestamps(frame, next_start)
        frame["simulation_run_id"] = run["run_id"]
        frame["source_scenario"] = run["scenario"]
        frame["source_seed"] = str(run["seed"])
        frame["source_kind"] = SOURCE_KIND
        frame["symbol"] = SYMBOL
        combined.append(frame)
    if not combined:
        return pd.DataFrame()
    output = pd.concat(combined, ignore_index=True)
    output = output.sort_values(["timestamp", "simulation_run_id"]).reset_index(drop=True)
    return output[ordered_columns(output)]


def write_combined_dataset(runs, output_venue_dir):
    output_venue_dir.mkdir(parents=True, exist_ok=True)
    ten_second = combine_kind(runs, "10s_flow")
    one_minute = combine_kind(runs, "1m_flow")
    ten_second_path = output_venue_dir / f"{SYMBOL}_10s_flow.csv"
    one_minute_path = output_venue_dir / f"{SYMBOL}_1m_flow.csv"
    if len(ten_second):
        atomic_write_csv(ten_second, ten_second_path)
    if len(one_minute):
        atomic_write_csv(one_minute, one_minute_path)
    return {
        "10s_path": ten_second_path,
        "1m_path": one_minute_path,
        "10s_rows": len(ten_second),
        "1m_rows": len(one_minute),
        "run_count": len(runs),
    }


def rolling_combine():
    runs = discover_runs(SIM_RUNS_ROOT, SCENARIOS)
    selected, selected_rows = select_recent_runs(runs)
    result = write_combined_dataset(selected, ROLLING_VENUE_DIR)
    result["eligible_run_count"] = len(runs)
    result["selected_10s_row_budget_count"] = selected_rows
    result["selected_runs"] = [run["run_id"] for run in selected]
    print("\n[rolling-combine] combined recent simulated runs")
    print(f"[rolling-combine] eligible runs: {len(runs)}")
    print(f"[rolling-combine] selected runs: {len(selected)}")
    print(f"[rolling-combine] 10s rows: {result['10s_rows']} -> {result['10s_path']}")
    print(f"[rolling-combine] 1m rows: {result['1m_rows']} -> {result['1m_path']}")
    return result


def tiny_price_child_env(primary_venue, extra=None):
    env = {
        "SYMBOL": SYMBOL,
        "PRIMARY_VENUE": primary_venue,
        "PRICE_TINY_FEATURE_GROUPS": FEATURE_GROUPS,
        "PRICE_TINY_TARGET_SPEC": TARGET_SPEC,
        "PRICE_TINY_WALK_FORWARD_TARGET_SPECS": WALK_FORWARD_TARGET_SPECS,
        "PRICE_TINY_MODEL_SPECS": MODEL_SPECS,
        "PROMOTE_BEST": "false",
        "PRICE_TINY_AUTO_REGISTER_CHALLENGERS": "false",
        "TRAIN_PRICE_TINY_MODEL": "false",
        "PRICE_TINY_WALK_FORWARD_TRAIN_ROWS": os.getenv("PRICE_TINY_WALK_FORWARD_TRAIN_ROWS", "30000"),
        "PRICE_TINY_WALK_FORWARD_VALIDATION_ROWS": os.getenv("PRICE_TINY_WALK_FORWARD_VALIDATION_ROWS", "10000"),
        "PRICE_TINY_WALK_FORWARD_TEST_ROWS": os.getenv("PRICE_TINY_WALK_FORWARD_TEST_ROWS", "5000"),
        "PRICE_TINY_WALK_FORWARD_STEP_ROWS": os.getenv("PRICE_TINY_WALK_FORWARD_STEP_ROWS", "10000"),
        "PRICE_TINY_WALK_FORWARD_MAX_WINDOWS": os.getenv("PRICE_TINY_WALK_FORWARD_MAX_WINDOWS", "20"),
        "PRICE_TINY_WALK_FORWARD_AUTO_BUILD": os.getenv("PRICE_TINY_WALK_FORWARD_AUTO_BUILD", "true"),
    }
    if extra:
        env.update(extra)
    return env


def build_and_walk_forward(primary_venue, label, output_suffix=""):
    env = tiny_price_child_env(primary_venue)
    build = run_python_script("build_tiny_price_training_rows.py", env, f"{label} build")
    if build.returncode != 0:
        return {"ok": False, "reason": "tiny_price_build_failed", "summary_path": ""}

    eval_env = dict(env)
    if output_suffix:
        eval_env["PRICE_TINY_WALK_FORWARD_OUTPUT_SUFFIX"] = output_suffix
    evaluation = run_python_script("evaluate_tiny_price_walk_forward.py", eval_env, f"{label} walk-forward")
    if evaluation.returncode != 0:
        return {"ok": False, "reason": "walk_forward_failed", "summary_path": ""}

    summary_path = walk_forward_summary_path(primary_venue, TARGET_SPEC, output_suffix)
    metrics = read_walk_forward_summary(summary_path)
    return {"ok": True, "reason": "", "summary_path": str(summary_path), **metrics}


def extract_candidate_model_path(output):
    matches = re.findall(r"candidate_model_path=(.+)", output or "")
    return matches[-1].strip() if matches else ""


def train_candidate_artifacts(primary_venue, label, target_specs=None):
    if not TRAIN_CANDIDATE_ARTIFACT:
        print(f"[{label}] candidate artifact training disabled by SIM_ROLLING_TRAIN_CANDIDATE_ARTIFACT=false.")
        return {"ok": True, "reason": "candidate_training_disabled", "candidate_model_paths": ""}

    target_specs = target_specs or TRAIN_CANDIDATE_TARGET_SPECS or [TARGET_SPEC]
    candidate_paths = []
    for target_spec in target_specs:
        env = tiny_price_child_env(
            primary_venue,
            {
                "PRICE_TINY_TARGET_SPEC": target_spec,
                "PRICE_TINY_MODEL_SPECS": MODEL_SPECS,
                "PROMOTE_BEST": "false",
                "PRICE_TINY_AUTO_REGISTER_CHALLENGERS": "false",
                "TRAIN_PRICE_TINY_MODEL": "false",
            },
        )
        build = run_python_script("build_tiny_price_training_rows.py", env, f"{label} candidate build {target_spec}")
        if build.returncode != 0:
            return {
                "ok": False,
                "reason": f"candidate_build_failed:{target_spec}",
                "candidate_model_paths": "|".join(candidate_paths),
            }
        train = run_python_script("train_tiny_price_model.py", env, f"{label} candidate train {target_spec}")
        if train.returncode != 0:
            return {
                "ok": False,
                "reason": f"candidate_train_failed:{target_spec}",
                "candidate_model_paths": "|".join(candidate_paths),
            }
        candidate_path = extract_candidate_model_path(train.stdout)
        if candidate_path:
            candidate_paths.append(candidate_path)
            print(f"[{label}] candidate artifact for {target_spec}: {candidate_path}")
        else:
            print(f"[{label}] warning: train completed for {target_spec}, but candidate_model_path was not found.")

    return {
        "ok": True,
        "reason": "",
        "candidate_model_paths": "|".join(candidate_paths),
    }


def walk_forward_summary_path(primary_venue, target_spec, output_suffix=""):
    venue_dir = REALTIME_ROOT / primary_venue
    suffix = f"__{slugify(output_suffix, '')}" if output_suffix else ""
    return venue_dir / f"{SYMBOL}_tiny_price_walk_forward_summary_{target_slug(target_spec)}{suffix}.csv"


def read_walk_forward_summary(path):
    frame = read_csv(path)
    if len(frame) == 0:
        return {
            "mean_sign_accuracy": "",
            "mean_avg_return_bps": "",
            "positive_window_count": "",
        }
    row = frame.iloc[0].to_dict()
    return {
        "mean_sign_accuracy": row.get("mean_sign_accuracy", ""),
        "mean_avg_return_bps": row.get("mean_avg_strategy_return_bps", ""),
        "positive_window_count": row.get("positive_return_windows", ""),
    }


def run_holdout_evaluation(cycle_number, scenario, seed):
    holdout_seed = int(seed) + HOLDOUT_SEED_OFFSET
    holdout_scenario = slugify(HOLDOUT_SCENARIOS[(cycle_number - 1) % len(HOLDOUT_SCENARIOS)], scenario)
    label = f"holdout cycle {cycle_number} {holdout_scenario} seed {holdout_seed}"
    simulation = run_simulation(holdout_scenario, holdout_seed, label)
    if simulation.returncode != 0:
        return {"ok": False, "reason": "holdout_simulation_failed", "summary_path": ""}

    archived = archive_simulation_outputs(holdout_scenario, holdout_seed, HOLDOUT_RUNS_ROOT)
    errors = [
        info["error"]
        for info in archived["archived"].values()
        if info.get("error")
    ]
    if errors:
        return {"ok": False, "reason": "; ".join(errors), "summary_path": ""}

    runs = discover_runs(HOLDOUT_RUNS_ROOT, HOLDOUT_SCENARIOS)
    selected, selected_rows = select_recent_runs(runs)
    combine = write_combined_dataset(selected, HOLDOUT_VENUE_DIR)
    print("\n[holdout-combine] combined recent holdout simulated runs")
    print(f"[holdout-combine] eligible runs: {len(runs)}")
    print(f"[holdout-combine] selected runs: {len(selected)}")
    print(f"[holdout-combine] selected 10s row budget count: {selected_rows}")
    print(f"[holdout-combine] selected run ids: {'|'.join(run['run_id'] for run in selected)}")
    print(f"[holdout-combine] 10s rows: {combine['10s_rows']} -> {combine['10s_path']}")
    print(f"[holdout-combine] 1m rows: {combine['1m_rows']} -> {combine['1m_path']}")
    result = build_and_walk_forward(
        HOLDOUT_OUTPUT_VENUE,
        f"holdout cycle {cycle_number}",
        output_suffix=f"holdout_cycle_{cycle_number}",
    )
    result["holdout_eligible_run_count"] = len(runs)
    result["holdout_selected_run_count"] = len(selected)
    result["holdout_selected_10s_row_budget_count"] = selected_rows
    result["holdout_selected_runs"] = "|".join(run["run_id"] for run in selected)
    result["holdout_10s_rows"] = combine.get("10s_rows", "")
    result["holdout_1m_rows"] = combine.get("1m_rows", "")
    return result


def append_cycle_summary(row):
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if SUMMARY_PATH.exists() and SUMMARY_PATH.stat().st_size > 0:
        with SUMMARY_PATH.open("r", newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
    existing_rows.append(row)
    fieldnames = []
    for item in existing_rows:
        for key in item.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    tmp_path = SUMMARY_PATH.with_suffix(SUMMARY_PATH.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
    tmp_path.replace(SUMMARY_PATH)


def cycle_iterator():
    if END_SEED is None and MAX_CYCLES <= 0 and not CONTINUOUS:
        print("SIM_ROLLING_END_SEED and SIM_ROLLING_MAX_CYCLES are unset; defaulting to one safe cycle.")
        print("Set SIM_ROLLING_CONTINUOUS=true for an open-ended research loop.")
        max_cycles = 1
    else:
        max_cycles = MAX_CYCLES

    cycle_number = 0
    seed = START_SEED
    while True:
        if END_SEED is not None and seed > END_SEED:
            break
        for scenario in SCENARIOS:
            cycle_number += 1
            if max_cycles > 0 and cycle_number > max_cycles:
                return
            yield cycle_number, slugify(scenario, scenario), seed
        seed += 1


def print_startup_plan():
    print("Rolling simulated tiny-price cycle")
    print(f"SYMBOL: {SYMBOL}")
    print(f"Scenarios: {', '.join(SCENARIOS)}")
    print(f"Seeds: start={START_SEED} end={END_SEED if END_SEED is not None else 'not set'} max_cycles={MAX_CYCLES or 'not set'}")
    print(f"Continuous mode: {CONTINUOUS}")
    print(f"SIM_SECONDS per run: {SIM_SECONDS}")
    print(f"Rolling venue: {ROLLING_OUTPUT_VENUE}")
    print(f"Holdout venue: {HOLDOUT_OUTPUT_VENUE}")
    print(f"Holdout scenarios: {', '.join(HOLDOUT_SCENARIOS)}")
    print(f"Rolling max runs: {ROLLING_MAX_RUNS}")
    print(f"Rolling max 10s rows: {ROLLING_MAX_10S_ROWS}")
    print(f"Train/evaluate every runs: {TRAIN_EVERY_RUNS}")
    print(f"Holdout evaluate every runs: {EVAL_EVERY_RUNS}")
    print(f"Feature groups: {FEATURE_GROUPS}")
    print(f"Target spec: {TARGET_SPEC}")
    print(f"Walk-forward targets: {WALK_FORWARD_TARGET_SPECS}")
    print(f"Model specs: {MODEL_SPECS}")
    print(f"Candidate artifact training enabled: {TRAIN_CANDIDATE_ARTIFACT}")
    print(f"Candidate artifact target specs: {', '.join(TRAIN_CANDIDATE_TARGET_SPECS)}")
    print(f"Summary output: {SUMMARY_PATH}")
    print("Research-only. No challenger registration, promotion, live predictions, orders, or private API.")


def main():
    print_startup_plan()
    last_walk_forward = {"ok": False, "reason": "not_run_yet", "summary_path": ""}
    last_holdout = {"ok": False, "reason": "not_run_yet", "summary_path": ""}
    last_candidate = {"ok": False, "reason": "not_run_yet", "candidate_model_paths": ""}

    for cycle_number, scenario, seed in cycle_iterator():
        failed_reason = ""
        print(f"\n=== Cycle {cycle_number}: scenario={scenario} seed={seed} ===")
        simulation = run_simulation(scenario, seed, f"cycle {cycle_number}")
        if simulation.returncode != 0:
            failed_reason = "simulation_failed"
            combine = {
                "run_count": "",
                "10s_rows": "",
                "selected_runs": [],
            }
        else:
            archived = archive_simulation_outputs(scenario, seed, SIM_RUNS_ROOT)
            archive_errors = [
                info["error"]
                for info in archived["archived"].values()
                if info.get("error")
            ]
            if archive_errors:
                failed_reason = "; ".join(archive_errors)
            combine = rolling_combine()

        if cycle_number % TRAIN_EVERY_RUNS == 0 and not failed_reason:
            last_walk_forward = build_and_walk_forward(ROLLING_OUTPUT_VENUE, f"cycle {cycle_number}")
            if not last_walk_forward.get("ok"):
                failed_reason = last_walk_forward.get("reason", "walk_forward_failed")
            else:
                last_candidate = train_candidate_artifacts(ROLLING_OUTPUT_VENUE, f"cycle {cycle_number}")
                if not last_candidate.get("ok"):
                    failed_reason = last_candidate.get("reason", "candidate_training_failed")
        else:
            print(f"[cycle {cycle_number}] tiny-price build/walk-forward skipped until every {TRAIN_EVERY_RUNS} runs.")

        if cycle_number % EVAL_EVERY_RUNS == 0 and not failed_reason:
            last_holdout = run_holdout_evaluation(cycle_number, scenario, seed)
            if not last_holdout.get("ok"):
                print(f"[cycle {cycle_number}] holdout evaluation skipped/failed: {last_holdout.get('reason')}")
        else:
            print(f"[cycle {cycle_number}] holdout evaluation skipped until every {EVAL_EVERY_RUNS} runs.")

        summary_row = {
            "written_at": utc_now_iso(),
            "symbol": SYMBOL,
            "source_kind": SOURCE_KIND,
            "cycle_number": cycle_number,
            "scenario": scenario,
            "seed": seed,
            "rolling_train_run_count": combine.get("run_count", ""),
            "rolling_eligible_run_count": combine.get("eligible_run_count", ""),
            "rolling_10s_row_count": combine.get("10s_rows", ""),
            "rolling_1m_row_count": combine.get("1m_rows", ""),
            "rolling_selected_runs": "|".join(combine.get("selected_runs", [])),
            "feature_groups": FEATURE_GROUPS,
            "target_spec": TARGET_SPEC,
            "walk_forward_summary_path": last_walk_forward.get("summary_path", ""),
            "mean_sign_accuracy": last_walk_forward.get("mean_sign_accuracy", ""),
            "mean_avg_return_bps": last_walk_forward.get("mean_avg_return_bps", ""),
            "positive_window_count": last_walk_forward.get("positive_window_count", ""),
            "candidate_model_paths": last_candidate.get("candidate_model_paths", ""),
            "candidate_training_reason": last_candidate.get("reason", ""),
            "holdout_summary_path": last_holdout.get("summary_path", ""),
            "holdout_mean_sign_accuracy": last_holdout.get("mean_sign_accuracy", ""),
            "holdout_mean_avg_return_bps": last_holdout.get("mean_avg_return_bps", ""),
            "holdout_positive_window_count": last_holdout.get("positive_window_count", ""),
            "holdout_eligible_run_count": last_holdout.get("holdout_eligible_run_count", ""),
            "holdout_selected_run_count": last_holdout.get("holdout_selected_run_count", ""),
            "holdout_10s_row_count": last_holdout.get("holdout_10s_rows", ""),
            "holdout_1m_row_count": last_holdout.get("holdout_1m_rows", ""),
            "holdout_selected_runs": last_holdout.get("holdout_selected_runs", ""),
            "failed_skipped_reason": failed_reason,
        }
        append_cycle_summary(summary_row)
        print(f"[cycle {cycle_number}] summary updated: {SUMMARY_PATH}")

    print("\nRolling simulated tiny-price cycle complete.")
    print(f"Summary: {SUMMARY_PATH}")
    print("Research-only. No promotion, no live prediction writes, no orders.")


if __name__ == "__main__":
    main()
