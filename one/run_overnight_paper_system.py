import csv
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IS_WINDOWS = os.name == "nt"
CREATE_NEW_PROCESS_GROUP = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0

SYMBOLS = [
    value.strip().upper()
    for value in os.getenv("SYMBOLS", os.getenv("SYMBOL", "SOLUSDT")).split(",")
    if value.strip()
]
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
VENUES = os.getenv("VENUES", PRIMARY_VENUE).strip()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

NIGHT_STATUS_SECONDS = int(os.getenv("NIGHT_STATUS_SECONDS", "60"))
NIGHT_MARKET_STACK_SECONDS = int(os.getenv("NIGHT_MARKET_STACK_SECONDS", "60"))
NIGHT_REGIME_UPDATE_SECONDS = int(os.getenv("NIGHT_REGIME_UPDATE_SECONDS", "900"))
NIGHT_RUN_ENABLE_MARKET_STACK = os.getenv("NIGHT_RUN_ENABLE_MARKET_STACK", "true").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
NIGHT_RUN_ENABLE_MICRO_BUILD = os.getenv("NIGHT_RUN_ENABLE_MICRO_BUILD", "false").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
NIGHT_RUN_ENABLE_MICRO_TRAIN = os.getenv("NIGHT_RUN_ENABLE_MICRO_TRAIN", "false").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
NIGHT_RUN_MICRO_TRAIN_EVERY_MINUTES = int(os.getenv("NIGHT_RUN_MICRO_TRAIN_EVERY_MINUTES", "120"))
HIERARCHY_LOG_INTERVAL_SECONDS = os.getenv("HIERARCHY_LOG_INTERVAL_SECONDS", "5")
STOP_TIMEOUT_SECONDS = int(os.getenv("NIGHT_RUN_STOP_TIMEOUT_SECONDS", "15"))


def env_bool(name, default="false"):
    return os.getenv(name, default).strip().lower() in {"true", "1", "yes", "y"}


ENABLE_RECORDERS = env_bool("ENABLE_RECORDERS", "true")
ENABLE_PRICE_TINY_MODEL = env_bool("ENABLE_PRICE_TINY_MODEL", "false")
ENABLE_FLOW_1S_MODEL = env_bool("ENABLE_FLOW_1S_MODEL", os.getenv("ENABLE_FLOW_1S_LOOPS", "true"))
ENABLE_MICRO_10S_MODEL = env_bool("ENABLE_MICRO_10S_MODEL", "false")
ENABLE_MEDIATOR_MODEL = env_bool("ENABLE_MEDIATOR_MODEL", "false")
ENABLE_SHADOW_POOL = env_bool("ENABLE_SHADOW_POOL", "false")
TRAIN_PRICE_TINY_MODEL = env_bool("TRAIN_PRICE_TINY_MODEL", "false")
TRAIN_FLOW_1S_MODEL = env_bool("TRAIN_FLOW_1S_MODEL", "false")
TRAIN_MICRO_10S_MODEL = env_bool("TRAIN_MICRO_10S_MODEL", str(NIGHT_RUN_ENABLE_MICRO_TRAIN).lower())
TRAIN_MEDIATOR_MODEL = env_bool("TRAIN_MEDIATOR_MODEL", "false")

stopping = threading.Event()
managed_processes = []


def npm_command():
    return "npm.cmd" if IS_WINDOWS else "npm"


def print_night(message):
    print(f"[night-run] {message}", flush=True)


def build_base_env(symbol=None):
    env = os.environ.copy()
    env["PRIMARY_VENUE"] = PRIMARY_VENUE
    env["VENUES"] = VENUES
    env["OUTPUT_DIR"] = str(OUTPUT_DIR)
    env["PROMOTE_BEST"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("FLOW_1S_MODEL_SELECTION", "latest_candidate")
    env.setdefault("MICRO_MODEL_SELECTION", "latest_candidate")
    if symbol:
        env["SYMBOL"] = symbol
        env["SYMBOLS"] = symbol
    else:
        env["SYMBOLS"] = ",".join(SYMBOLS)
        env["SYMBOL"] = SYMBOLS[0] if SYMBOLS else "SOLUSDT"
    return env


def live_system_env():
    env = build_base_env()
    env["SYMBOLS"] = ",".join(SYMBOLS)
    env["ENABLE_RECORDERS"] = str(ENABLE_RECORDERS).lower()
    env["ENABLE_PRICE_TINY_MODEL"] = str(ENABLE_PRICE_TINY_MODEL).lower()
    env["ENABLE_FLOW_1S_MODEL"] = str(ENABLE_FLOW_1S_MODEL).lower()
    env["ENABLE_FLOW_1S_LOOPS"] = str(ENABLE_FLOW_1S_MODEL).lower()
    env["ENABLE_MICRO_10S_MODEL"] = str(ENABLE_MICRO_10S_MODEL).lower()
    env["ENABLE_MEDIATOR_MODEL"] = str(ENABLE_MEDIATOR_MODEL).lower()
    env["ENABLE_SHADOW_POOL"] = str(ENABLE_SHADOW_POOL).lower()
    env["TRAIN_PRICE_TINY_MODEL"] = str(TRAIN_PRICE_TINY_MODEL).lower()
    env["TRAIN_FLOW_1S_MODEL"] = str(TRAIN_FLOW_1S_MODEL).lower()
    env["TRAIN_MICRO_10S_MODEL"] = str(TRAIN_MICRO_10S_MODEL).lower()
    env["TRAIN_MEDIATOR_MODEL"] = str(TRAIN_MEDIATOR_MODEL).lower()
    env["ENABLE_LIVE_LOOPS"] = os.getenv("ENABLE_LIVE_LOOPS", "false")
    env["ENABLE_MARKET_STACK"] = "false"
    env["ENABLE_REGIME_UPDATERS"] = "false"
    env["ENABLE_MICRO_BUILDERS"] = "true" if (NIGHT_RUN_ENABLE_MICRO_BUILD or ENABLE_MICRO_10S_MODEL or TRAIN_MICRO_10S_MODEL) else "false"
    env["ENABLE_HTF_UPDATERS"] = os.getenv("ENABLE_HTF_UPDATERS", "true")
    env["SNAPSHOT_INTERVAL_SECONDS"] = os.getenv("SNAPSHOT_INTERVAL_SECONDS", "1")
    env["ORDER_BOOK_DEPTH"] = os.getenv("ORDER_BOOK_DEPTH", "100")
    return env


@dataclass
class ManagedProcess:
    name: str
    command: list
    env: dict
    process: subprocess.Popen | None = None


def stream_output(managed):
    assert managed.process is not None
    try:
        for line in managed.process.stdout:
            print(f"[{managed.name}] {line.rstrip()}", flush=True)
    except Exception as error:
        if not stopping.is_set():
            print_night(f"output reader failed for {managed.name}: {error}")


def start_long_running(name, command, env):
    managed = ManagedProcess(name=name, command=command, env=env)
    managed_processes.append(managed)
    print_night(f"starting {name}: {' '.join(str(part) for part in command)}")
    managed.process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        creationflags=CREATE_NEW_PROCESS_GROUP,
    )
    threading.Thread(target=stream_output, args=(managed,), daemon=True).start()
    return managed


def run_one_shot(name, command, env):
    if stopping.is_set():
        return
    print_night(f"running {name}: {' '.join(str(part) for part in command)}")
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=None,
        )
        output = completed.stdout.strip()
        if output:
            for line in output.splitlines():
                print(f"[{name}] {line}", flush=True)
        print_night(f"{name} finished with code {completed.returncode}")
    except Exception as error:
        print_night(f"{name} failed: {error}")


def periodic_runner(name, command_factory, env_factory, interval_seconds):
    while not stopping.is_set():
        command = command_factory()
        env = env_factory()
        run_one_shot(name, command, env)
        wait_seconds(interval_seconds)


def wait_seconds(seconds):
    deadline = time.time() + max(0, int(seconds))
    while not stopping.is_set() and time.time() < deadline:
        time.sleep(0.25)


def normalize_timestamp(raw_value):
    try:
        value = float(raw_value)
    except Exception:
        return None
    if value < 10_000_000_000:
        value *= 1000.0
    return int(value)


def read_latest_csv_row(path):
    path = Path(path)
    if not path.exists():
        return None, 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            latest = None
            count = 0
            for row in reader:
                if not row:
                    continue
                count += 1
                latest = row
            return latest, count
    except Exception:
        return None, 0


def file_freshness(path):
    row, count = read_latest_csv_row(path)
    if row is None:
        return {"exists": Path(path).exists(), "rows": count, "timestamp": None, "age_seconds": None}
    timestamp = normalize_timestamp(row.get("timestamp"))
    age = None
    if timestamp is not None:
        age = max(0.0, time.time() - timestamp / 1000.0)
    return {"exists": True, "rows": count, "timestamp": timestamp, "age_seconds": age, "row": row}


def venue_path(symbol, filename):
    return OUTPUT_DIR / PRIMARY_VENUE / filename.format(SYMBOL=symbol)


def status_for_symbol(symbol):
    ten_second = file_freshness(venue_path(symbol, "{SYMBOL}_10s_flow.csv"))
    one_minute = file_freshness(venue_path(symbol, "{SYMBOL}_1m_flow.csv"))
    one_second_prediction = file_freshness(venue_path(symbol, "{SYMBOL}_1s_order_flow_predictions.csv"))
    tiny_price_prediction = file_freshness(venue_path(symbol, "{SYMBOL}_tiny_price_live_predictions.csv"))
    hierarchy = file_freshness(venue_path(symbol, "{SYMBOL}_hierarchy_forecast_log.csv"))
    latest_abstain = ""
    if hierarchy.get("row"):
        latest_abstain = str(hierarchy["row"].get("abstain_reason", ""))
    return {
        "10s_age": ten_second["age_seconds"],
        "1m_age": one_minute["age_seconds"],
        "1s_prediction_age": one_second_prediction["age_seconds"],
        "tiny_price_prediction_age": tiny_price_prediction["age_seconds"],
        "hierarchy_rows": hierarchy["rows"],
        "latest_abstain_reason": latest_abstain,
    }


def format_age(age):
    if age is None:
        return "missing"
    return f"{age:.1f}s"


def status_loop():
    while not stopping.is_set():
        for symbol in SYMBOLS:
            status = status_for_symbol(symbol)
            print_night(
                f"{symbol} status | "
                f"10s_flow_age={format_age(status['10s_age'])} | "
                f"1m_flow_age={format_age(status['1m_age'])} | "
                f"1s_prediction_age={format_age(status['1s_prediction_age'])} | "
                f"tiny_price_age={format_age(status['tiny_price_prediction_age'])} | "
                f"hierarchy_rows={status['hierarchy_rows']} | "
                f"latest_abstain_reason={status['latest_abstain_reason'] or 'none'}"
            )
        wait_seconds(NIGHT_STATUS_SECONDS)


def stop_long_running_processes():
    for managed in managed_processes:
        process = managed.process
        if process is None or process.poll() is not None:
            continue
        print_night(f"stopping {managed.name}")
        try:
            if IS_WINDOWS:
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
    deadline = time.time() + STOP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if all(managed.process is None or managed.process.poll() is not None for managed in managed_processes):
            return
        time.sleep(0.25)
    for managed in managed_processes:
        process = managed.process
        if process is not None and process.poll() is None:
            print_night(f"force stopping {managed.name}")
            process.kill()


def main():
    if not SYMBOLS:
        raise RuntimeError("SYMBOLS is empty.")
    print_night("overnight paper system starting")
    print_night(f"symbols: {', '.join(SYMBOLS)}")
    print_night(f"primary venue: {PRIMARY_VENUE}")
    print_night("startup component plan")
    print_night(f"- recorder enabled: {ENABLE_RECORDERS}")
    print_night(f"- price tiny enabled: {ENABLE_PRICE_TINY_MODEL} train={TRAIN_PRICE_TINY_MODEL}")
    print_night(f"- 1s flow enabled: {ENABLE_FLOW_1S_MODEL} train={TRAIN_FLOW_1S_MODEL}")
    print_night(f"- 10s micro enabled: {ENABLE_MICRO_10S_MODEL or NIGHT_RUN_ENABLE_MICRO_BUILD} train={TRAIN_MICRO_10S_MODEL}")
    print_night(f"- mediator enabled: {ENABLE_MEDIATOR_MODEL} train={TRAIN_MEDIATOR_MODEL}")
    print_night(f"- shadow pool enabled: {ENABLE_SHADOW_POOL}")
    print_night(f"micro build enabled: {NIGHT_RUN_ENABLE_MICRO_BUILD}")
    print_night(f"micro train enabled: {NIGHT_RUN_ENABLE_MICRO_TRAIN}")
    print_night("PROMOTE_BEST forced to false. No trades/orders/private API behavior.")
    if not ENABLE_RECORDERS and (
        ENABLE_PRICE_TINY_MODEL
        or ENABLE_FLOW_1S_MODEL
        or ENABLE_MICRO_10S_MODEL
        or ENABLE_MEDIATOR_MODEL
        or ENABLE_SHADOW_POOL
    ):
        raise RuntimeError(
            "Fail closed: ENABLE_RECORDERS=false while at least one live prediction "
            "component requires fresh snapshots. Disable those components or enable recorders."
        )

    start_long_running(
        "live-system",
        [npm_command(), "run", "live-system"],
        live_system_env(),
    )
    if ENABLE_MEDIATOR_MODEL:
        for symbol in SYMBOLS:
            env = build_base_env(symbol)
            env["HIERARCHY_LOG_LOOP"] = "true"
            env["HIERARCHY_LOG_INTERVAL_SECONDS"] = HIERARCHY_LOG_INTERVAL_SECONDS
            start_long_running(
                f"{symbol} hierarchy-log-loop",
                [npm_command(), "run", "hierarchy-log-loop"],
                env,
            )

    for symbol in SYMBOLS:
        threading.Thread(
            target=periodic_runner,
            args=(
                f"{symbol} regime-update",
                lambda: [npm_command(), "run", "regime-update"],
                lambda symbol=symbol: build_base_env(symbol),
                NIGHT_REGIME_UPDATE_SECONDS,
            ),
            daemon=True,
        ).start()

    if NIGHT_RUN_ENABLE_MARKET_STACK:
        threading.Thread(
            target=periodic_runner,
            args=(
                "market-stack",
                lambda: [npm_command(), "run", "market-stack"],
                lambda: build_base_env(),
                NIGHT_MARKET_STACK_SECONDS,
            ),
            daemon=True,
        ).start()

    if NIGHT_RUN_ENABLE_MICRO_TRAIN:
        for symbol in SYMBOLS:
            threading.Thread(
                target=periodic_runner,
                args=(
                    f"{symbol} micro-train",
                    lambda: [npm_command(), "run", "micro-train"],
                    lambda symbol=symbol: build_base_env(symbol),
                    NIGHT_RUN_MICRO_TRAIN_EVERY_MINUTES * 60,
                ),
                daemon=True,
            ).start()

    threading.Thread(target=status_loop, daemon=True).start()

    try:
        while any(
            managed.process is not None and managed.process.poll() is None
            for managed in managed_processes
        ):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print_night("Ctrl+C received. Stopping overnight paper system...")
    finally:
        stopping.set()
        stop_long_running_processes()
        print_night("overnight paper system stopped")


if __name__ == "__main__":
    main()
