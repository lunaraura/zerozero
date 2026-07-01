import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYMBOLS = [
    value.strip().upper()
    for value in os.getenv("SYMBOLS", "SOLUSDT,BTCUSDT,ETHUSDT").split(",")
    if value.strip()
]
VENUES = [
    value.strip().lower()
    for value in os.getenv("VENUES", os.getenv("VENUE", "binanceus")).split(",")
    if value.strip()
]
PRIMARY_VENUE = os.getenv(
    "PRIMARY_VENUE",
    VENUES[0] if VENUES else "binanceus",
).strip().lower()


def parse_symbol_list(value, default_values):
    raw = value if value is not None and str(value).strip() else ",".join(default_values)
    return [item.strip().upper() for item in str(raw).split(",") if item.strip()]


def parse_venue_list(value, default_values):
    raw = value if value is not None and str(value).strip() else ",".join(default_values)
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


PRICE_TINY_SYMBOLS = parse_symbol_list(
    os.getenv("PRICE_TINY_SYMBOLS"),
    SYMBOLS if SYMBOLS else [os.getenv("SYMBOL", "SOLUSDT")],
)
PRICE_TINY_PRIMARY_VENUES = parse_venue_list(
    os.getenv("PRICE_TINY_PRIMARY_VENUES"),
    [PRIMARY_VENUE] if PRIMARY_VENUE else ([VENUES[0]] if VENUES else ["binanceus"]),
)

SNAPSHOT_INTERVAL_SECONDS = os.getenv("SNAPSHOT_INTERVAL_SECONDS", "1")
ORDER_BOOK_DEPTH = os.getenv("ORDER_BOOK_DEPTH", "100")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/realtime")
LOOP_SECONDS = os.getenv("LOOP_SECONDS", "60")
TRAIN_EVERY_MINUTES = os.getenv("TRAIN_EVERY_MINUTES", "60")
FLOW_1S_MODEL_SELECTION = os.getenv("FLOW_1S_MODEL_SELECTION", "latest_candidate")
MICRO_MODEL_SELECTION = os.getenv("MICRO_MODEL_SELECTION", "latest_candidate")
PROMOTE_BEST = os.getenv("PROMOTE_BEST", "false")
MICRO_RUN_FLOW_1S_ABLATION = os.getenv("MICRO_RUN_FLOW_1S_ABLATION", "false")
MICRO_ALLOW_GROUP_DISAGREEMENT = os.getenv("MICRO_ALLOW_GROUP_DISAGREEMENT", "false")
RESTART_DELAY_SECONDS = int(os.getenv("RESTART_DELAY_SECONDS", "10"))
ENABLE_RECORDERS = os.getenv("ENABLE_RECORDERS", "true").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
ENABLE_LIVE_LOOPS = os.getenv("ENABLE_LIVE_LOOPS", "true").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
ENABLE_MICRO_BUILDERS = os.getenv("ENABLE_MICRO_BUILDERS", "false").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
ENABLE_MARKET_STACK = os.getenv("ENABLE_MARKET_STACK", "false").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
ENABLE_REGIME_UPDATERS = os.getenv("ENABLE_REGIME_UPDATERS", "true").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
ENABLE_HTF_UPDATERS = os.getenv("ENABLE_HTF_UPDATERS", "true").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
ENABLE_FLOW_1S_LOOPS = os.getenv("ENABLE_FLOW_1S_LOOPS", "true").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
}
MICRO_BUILD_SECONDS = os.getenv("MICRO_BUILD_SECONDS", "60")
REGIME_UPDATE_SECONDS = os.getenv("REGIME_UPDATE_SECONDS", "300")
HTF_UPDATE_SECONDS = os.getenv("HTF_UPDATE_SECONDS", "300")
FLOW_1S_LOOP_SECONDS = os.getenv("FLOW_1S_LOOP_SECONDS", "1")
MAX_MICRO_SNAPSHOTS = os.getenv("MAX_MICRO_SNAPSHOTS", "2000")
STACK_LOOP_SECONDS = os.getenv("STACK_LOOP_SECONDS", LOOP_SECONDS)
STOP_TIMEOUT_SECONDS = int(os.getenv("STOP_TIMEOUT_SECONDS", "15"))


def env_bool(name, default="false"):
    return os.getenv(name, default).strip().lower() in {"true", "1", "yes", "y"}


ENABLE_PRICE_TINY_MODEL = env_bool("ENABLE_PRICE_TINY_MODEL", "false")
ENABLE_FLOW_1S_MODEL = env_bool("ENABLE_FLOW_1S_MODEL", os.getenv("ENABLE_FLOW_1S_LOOPS", "true"))
ENABLE_MICRO_10S_MODEL = env_bool("ENABLE_MICRO_10S_MODEL", os.getenv("ENABLE_MICRO_BUILDERS", "false"))
ENABLE_MEDIATOR_MODEL = env_bool("ENABLE_MEDIATOR_MODEL", "false")
ENABLE_SHADOW_POOL = env_bool("ENABLE_SHADOW_POOL", "false")
ENABLE_PRICE_TINY_SHADOW_POOL = env_bool(
    "ENABLE_PRICE_TINY_SHADOW_POOL",
    os.getenv("ENABLE_SHADOW_POOL", "false"),
)
ENABLE_PRICE_TINY_ENSEMBLE_SHOW = env_bool(
    "ENABLE_PRICE_TINY_ENSEMBLE_SHOW",
    os.getenv("ENABLE_PRICE_TINY_SHADOW_POOL", "false"),
)
TRAIN_PRICE_TINY_MODEL = env_bool("TRAIN_PRICE_TINY_MODEL", "false")
TRAIN_FLOW_1S_MODEL = env_bool("TRAIN_FLOW_1S_MODEL", "false")
TRAIN_MICRO_10S_MODEL = env_bool("TRAIN_MICRO_10S_MODEL", "false")
TRAIN_MEDIATOR_MODEL = env_bool("TRAIN_MEDIATOR_MODEL", "false")
PRICE_TINY_BUILD_SECONDS = os.getenv("PRICE_TINY_BUILD_SECONDS", "30")
PRICE_TINY_SHOW_SECONDS = os.getenv("PRICE_TINY_SHOW_SECONDS", "5")
PRICE_TINY_TRAIN_SECONDS = os.getenv("PRICE_TINY_TRAIN_SECONDS", "300")
PRICE_TINY_SHADOW_SHOW_SECONDS = os.getenv("PRICE_TINY_SHADOW_SHOW_SECONDS", "5")
PRICE_TINY_SHADOW_EVALUATE_SECONDS = os.getenv("PRICE_TINY_SHADOW_EVALUATE_SECONDS", "300")
PRICE_TINY_ENSEMBLE_SHOW_SECONDS = os.getenv("PRICE_TINY_ENSEMBLE_SHOW_SECONDS", "15")
PRICE_TINY_SHADOW_ALLOW_PROMOTION = os.getenv("PRICE_TINY_SHADOW_ALLOW_PROMOTION", "false")
PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES = os.getenv("PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES", "")
FLOW_1S_TRAIN_SECONDS = os.getenv("FLOW_1S_TRAIN_SECONDS", "600")
MICRO_10S_TRAIN_SECONDS = os.getenv("MICRO_10S_TRAIN_SECONDS", "900")
MEDIATOR_BUILD_SECONDS = os.getenv("MEDIATOR_BUILD_SECONDS", "60")
MEDIATOR_TRAIN_SECONDS = os.getenv("MEDIATOR_TRAIN_SECONDS", "900")

IS_WINDOWS = os.name == "nt"
CREATE_NEW_PROCESS_GROUP = (
    subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
)

stopping = threading.Event()


@dataclass
class ManagedProcess:
    symbol: str
    process_type: str
    command: list
    env: dict
    venue: str | None = None
    process: subprocess.Popen | None = None
    monitor_thread: threading.Thread | None = None

    @property
    def prefix(self):
        if self.venue:
            return f"[{self.symbol} {self.venue} {self.process_type}]"
        return f"[{self.symbol} {self.process_type}]"


def print_supervisor(message):
    print(f"[supervisor] {message}", flush=True)


def build_child_env(symbol):
    child_env = os.environ.copy()
    child_env["SYMBOL"] = symbol
    # record_realtime_market_data.js can read SYMBOLS and start multiple
    # symbols by itself. The supervisor already creates one child per symbol,
    # so force each child to see only its assigned symbol and avoid duplicate
    # writers for the same CSVs.
    child_env["SYMBOLS"] = symbol
    child_env["PRIMARY_VENUE"] = PRIMARY_VENUE
    child_env["SNAPSHOT_INTERVAL_SECONDS"] = SNAPSHOT_INTERVAL_SECONDS
    child_env["ORDER_BOOK_DEPTH"] = ORDER_BOOK_DEPTH
    child_env["OUTPUT_DIR"] = OUTPUT_DIR
    child_env["LOOP_SECONDS"] = LOOP_SECONDS
    child_env["TRAIN_EVERY_MINUTES"] = TRAIN_EVERY_MINUTES
    child_env["FLOW_1S_MODEL_SELECTION"] = FLOW_1S_MODEL_SELECTION
    child_env["MICRO_MODEL_SELECTION"] = MICRO_MODEL_SELECTION
    child_env["PROMOTE_BEST"] = PROMOTE_BEST
    child_env["MICRO_RUN_FLOW_1S_ABLATION"] = MICRO_RUN_FLOW_1S_ABLATION
    child_env["MICRO_ALLOW_GROUP_DISAGREEMENT"] = MICRO_ALLOW_GROUP_DISAGREEMENT
    child_env["MICRO_BUILD_SECONDS"] = MICRO_BUILD_SECONDS
    child_env["REGIME_UPDATE_SECONDS"] = REGIME_UPDATE_SECONDS
    child_env["HTF_UPDATE_SECONDS"] = HTF_UPDATE_SECONDS
    child_env["FLOW_1S_LOOP_SECONDS"] = FLOW_1S_LOOP_SECONDS
    child_env["MAX_MICRO_SNAPSHOTS"] = MAX_MICRO_SNAPSHOTS
    child_env["ENABLE_PRICE_TINY_MODEL"] = str(ENABLE_PRICE_TINY_MODEL).lower()
    child_env["ENABLE_FLOW_1S_MODEL"] = str(ENABLE_FLOW_1S_MODEL).lower()
    child_env["ENABLE_MICRO_10S_MODEL"] = str(ENABLE_MICRO_10S_MODEL).lower()
    child_env["ENABLE_MEDIATOR_MODEL"] = str(ENABLE_MEDIATOR_MODEL).lower()
    child_env["ENABLE_SHADOW_POOL"] = str(ENABLE_SHADOW_POOL).lower()
    child_env["ENABLE_PRICE_TINY_SHADOW_POOL"] = str(ENABLE_PRICE_TINY_SHADOW_POOL).lower()
    child_env["ENABLE_PRICE_TINY_ENSEMBLE_SHOW"] = str(ENABLE_PRICE_TINY_ENSEMBLE_SHOW).lower()
    child_env["TRAIN_PRICE_TINY_MODEL"] = str(TRAIN_PRICE_TINY_MODEL).lower()
    child_env["TRAIN_FLOW_1S_MODEL"] = str(TRAIN_FLOW_1S_MODEL).lower()
    child_env["TRAIN_MICRO_10S_MODEL"] = str(TRAIN_MICRO_10S_MODEL).lower()
    child_env["TRAIN_MEDIATOR_MODEL"] = str(TRAIN_MEDIATOR_MODEL).lower()
    child_env["PRICE_TINY_SHADOW_SHOW_SECONDS"] = PRICE_TINY_SHADOW_SHOW_SECONDS
    child_env["PRICE_TINY_SHADOW_EVALUATE_SECONDS"] = PRICE_TINY_SHADOW_EVALUATE_SECONDS
    child_env["PRICE_TINY_ENSEMBLE_SHOW_SECONDS"] = PRICE_TINY_ENSEMBLE_SHOW_SECONDS
    child_env["PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES"] = PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES
    # Promotion stays disabled unless the user explicitly enables this env var.
    child_env["PRICE_TINY_SHADOW_ALLOW_PROMOTION"] = PRICE_TINY_SHADOW_ALLOW_PROMOTION
    child_env["PYTHONUNBUFFERED"] = "1"
    # The supervisor should run the continuous loop, not one-shot mode.
    child_env.pop("RUN_ONCE", None)
    return child_env


def scoped_price_tiny_model_env_name(symbol, venue):
    safe_symbol = "".join(ch if ch.isalnum() else "_" for ch in symbol.upper())
    safe_venue = "".join(ch if ch.isalnum() else "_" for ch in venue.upper())
    return f"PRICE_TINY_MODEL_PATH_{safe_symbol}_{safe_venue}"


def resolve_scoped_price_tiny_model_path(symbol, venue):
    scoped_name = scoped_price_tiny_model_env_name(symbol, venue)
    scoped_value = os.getenv(scoped_name, "").strip()
    if scoped_value:
        return scoped_name, scoped_value
    global_value = os.getenv("PRICE_TINY_MODEL_PATH", "").strip()
    if global_value and len(PRICE_TINY_SYMBOLS) == 1 and len(PRICE_TINY_PRIMARY_VENUES) == 1:
        return "PRICE_TINY_MODEL_PATH", global_value
    return None, None


def build_price_tiny_child_env(symbol, venue):
    child_env = build_child_env(symbol)
    child_env["SYMBOL"] = symbol
    child_env["SYMBOLS"] = symbol
    child_env["PRIMARY_VENUE"] = venue
    child_env["PRICE_TINY_SHADOW_ALLOW_PROMOTION"] = PRICE_TINY_SHADOW_ALLOW_PROMOTION
    source_name, model_path = resolve_scoped_price_tiny_model_path(symbol, venue)
    child_env.pop("PRICE_TINY_MODEL_PATH", None)
    if model_path:
        child_env["PRICE_TINY_MODEL_PATH"] = model_path
        child_env["PRICE_TINY_MODEL_PATH_SOURCE"] = source_name
    else:
        child_env["PRICE_TINY_MODEL_PATH_SOURCE"] = "selected_model_or_registry"
    return child_env


def recorder_command():
    # Keep this as the existing npm recorder command; the supervisor does not
    # duplicate websocket or file-writing logic.
    return ["npm.cmd" if IS_WINDOWS else "npm", "run", "record-realtime"]


def live_loop_command():
    # Keep this as the existing Python loop; the supervisor does not duplicate
    # prediction, labeling, training, or promotion logic.
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "run_live_3m_prediction_loop.py")]


def micro_builder_command():
    # This is a paper-data builder only. It does not train, promote, or place
    # trades. The supervisor restarts it after it exits, producing a simple
    # incremental/recent snapshot refresh loop.
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "build_10s_microstructure_training_rows.py")]


def regime_updater_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "update_live_regime_features.py")]


def htf_updater_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "update_live_htf_context_features.py")]


def flow_1s_loop_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "run_live_1s_order_flow_prediction_loop.py")]


def market_stack_command():
    # One dashboard process prints the combined paper-only stack for all
    # symbols. It reads existing model outputs and current snapshots.
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "show_live_market_stack.py")]


def price_tiny_build_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "build_tiny_price_training_rows.py")]


def price_tiny_train_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "train_tiny_price_model.py")]


def price_tiny_show_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "show_tiny_price_prediction.py")]


def price_tiny_shadow_show_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "show_tiny_price_shadow_pool_predictions.py")]


def price_tiny_shadow_evaluate_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_tiny_price_shadow_pool.py")]


def price_tiny_ensemble_show_command():
    return ["npm.cmd" if IS_WINDOWS else "npm", "run", "tiny-price-ensemble-show"]


def flow_1s_train_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "train_1s_order_flow_model.py")]


def micro_train_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "train_10s_microstructure_model.py")]


def hierarchy_log_loop_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "log_hierarchy_forecast_snapshot.py"), "--loop"]


def mediator_build_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "build_hierarchy_mediator_training_rows.py")]


def mediator_train_command():
    return [sys.executable, str(PROJECT_ROOT / "scripts" / "train_hierarchy_mediator_model.py")]


def stream_output(managed):
    assert managed.process is not None
    try:
        for line in managed.process.stdout:
            print(f"{managed.prefix} {line.rstrip()}", flush=True)
    except Exception as error:
        if not stopping.is_set():
            print_supervisor(f"output reader failed for {managed.prefix}: {error}")


def start_process(managed):
    print_supervisor(
        f"starting {managed.prefix}: {' '.join(str(part) for part in managed.command)}"
    )
    if managed.process_type == "recorder":
        print_supervisor(
            f"{managed.prefix} child env SYMBOL={managed.env.get('SYMBOL')} "
            f"SYMBOLS={managed.env.get('SYMBOLS')}"
        )
    managed.process = subprocess.Popen(
        managed.command,
        cwd=PROJECT_ROOT,
        env=managed.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        creationflags=CREATE_NEW_PROCESS_GROUP,
    )
    reader_thread = threading.Thread(
        target=stream_output,
        args=(managed,),
        daemon=True,
    )
    reader_thread.start()


def monitor_process(managed):
    while not stopping.is_set():
        try:
            start_process(managed)
        except FileNotFoundError as error:
            print_supervisor(
                f"could not start {managed.prefix}: {error}. "
                f"Retrying in {RESTART_DELAY_SECONDS}s."
            )
            wait_for_restart_delay()
            continue
        except Exception as error:
            print_supervisor(
                f"could not start {managed.prefix}: {error}. "
                f"Retrying in {RESTART_DELAY_SECONDS}s."
            )
            wait_for_restart_delay()
            continue

        return_code = managed.process.wait()
        if stopping.is_set():
            print_supervisor(f"{managed.prefix} stopped with code {return_code}")
            break

        if managed.process_type == "micro-build" and return_code == 0:
            print_supervisor(
                f"{managed.prefix} completed paper microstructure refresh. "
                f"Running again in {MICRO_BUILD_SECONDS}s."
            )
            wait_seconds(int(MICRO_BUILD_SECONDS))
            continue
        periodic_seconds = {
            "price-tiny-build": int(PRICE_TINY_BUILD_SECONDS),
            "price-tiny-show": int(PRICE_TINY_SHOW_SECONDS),
            "price-tiny-train": int(PRICE_TINY_TRAIN_SECONDS),
            "price-tiny-shadow-show": int(PRICE_TINY_SHADOW_SHOW_SECONDS),
            "price-tiny-shadow-evaluate": int(PRICE_TINY_SHADOW_EVALUATE_SECONDS),
            "price-tiny-ensemble-show": int(PRICE_TINY_ENSEMBLE_SHOW_SECONDS),
            "flow1s-train": int(FLOW_1S_TRAIN_SECONDS),
            "micro-train": int(MICRO_10S_TRAIN_SECONDS),
            "mediator-build": int(MEDIATOR_BUILD_SECONDS),
            "mediator-train": int(MEDIATOR_TRAIN_SECONDS),
        }
        if managed.process_type in periodic_seconds and return_code == 0:
            wait_for = periodic_seconds[managed.process_type]
            print_supervisor(
                f"{managed.prefix} completed paper refresh. Running again in {wait_for}s."
            )
            wait_seconds(wait_for)
            continue
        if managed.process_type == "regime-update" and return_code == 0:
            print_supervisor(
                f"{managed.prefix} completed live regime refresh. "
                f"Running again in {REGIME_UPDATE_SECONDS}s."
            )
            wait_seconds(int(REGIME_UPDATE_SECONDS))
            continue
        if managed.process_type == "htf-update" and return_code == 0:
            print_supervisor(
                f"{managed.prefix} completed live HTF context refresh. "
                f"Running again in {HTF_UPDATE_SECONDS}s."
            )
            wait_seconds(int(HTF_UPDATE_SECONDS))
            continue

        print_supervisor(
            f"{managed.prefix} exited unexpectedly with code {return_code}. "
            f"Restarting in {RESTART_DELAY_SECONDS}s."
        )
        wait_for_restart_delay()


def wait_for_restart_delay():
    wait_seconds(RESTART_DELAY_SECONDS)


def wait_seconds(seconds):
    seconds = max(0, int(seconds))
    deadline = time.time() + seconds
    while not stopping.is_set() and time.time() < deadline:
        time.sleep(0.25)


def stop_process(managed):
    process = managed.process
    if process is None or process.poll() is not None:
        return

    print_supervisor(f"stopping {managed.prefix}")
    try:
        if IS_WINDOWS:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
    except Exception as error:
        print_supervisor(f"graceful stop signal failed for {managed.prefix}: {error}")
        try:
            process.terminate()
        except Exception:
            pass


def kill_if_still_running(managed):
    process = managed.process
    if process is None or process.poll() is not None:
        return
    print_supervisor(f"force stopping {managed.prefix}")
    try:
        process.kill()
    except Exception as error:
        print_supervisor(f"force stop failed for {managed.prefix}: {error}")


def create_managed_processes():
    managed_processes = []
    for symbol in SYMBOLS:
        child_env = build_child_env(symbol)
        if ENABLE_RECORDERS:
            managed_processes.append(
                ManagedProcess(
                    symbol=symbol,
                    process_type="recorder",
                    command=recorder_command(),
                    env=child_env,
                )
            )
        if ENABLE_LIVE_LOOPS:
            managed_processes.append(
                ManagedProcess(
                    symbol=symbol,
                    process_type="loop",
                    command=live_loop_command(),
                    env=child_env,
                )
            )
        if ENABLE_MICRO_BUILDERS or ENABLE_MICRO_10S_MODEL or TRAIN_MICRO_10S_MODEL:
            managed_processes.append(
                ManagedProcess(
                    symbol=symbol,
                    process_type="micro-build",
                    command=micro_builder_command(),
                    env=child_env,
                )
            )
        if ENABLE_REGIME_UPDATERS:
            managed_processes.append(
                ManagedProcess(
                    symbol=symbol,
                    process_type="regime-update",
                    command=regime_updater_command(),
                    env=child_env,
                )
            )
        if ENABLE_HTF_UPDATERS:
            managed_processes.append(
                ManagedProcess(
                    symbol=symbol,
                    process_type="htf-update",
                    command=htf_updater_command(),
                    env=child_env,
                )
            )
        if ENABLE_FLOW_1S_MODEL:
            managed_processes.append(
                ManagedProcess(
                    symbol=symbol,
                    process_type="flow1s-loop",
                    command=flow_1s_loop_command(),
                    env=child_env,
                )
            )
        if TRAIN_FLOW_1S_MODEL:
            managed_processes.append(
                ManagedProcess(
                    symbol=symbol,
                    process_type="flow1s-train",
                    command=flow_1s_train_command(),
                    env=child_env,
                )
            )
        if TRAIN_MICRO_10S_MODEL:
            managed_processes.append(
                ManagedProcess(
                    symbol=symbol,
                    process_type="micro-train",
                    command=micro_train_command(),
                    env=child_env,
                )
            )
        if ENABLE_MEDIATOR_MODEL:
            managed_processes.append(
                ManagedProcess(
                    symbol=symbol,
                    process_type="hierarchy-log-loop",
                    command=hierarchy_log_loop_command(),
                    env=child_env,
                )
            )
            managed_processes.append(
                ManagedProcess(
                    symbol=symbol,
                    process_type="mediator-build",
                    command=mediator_build_command(),
                    env=child_env,
                )
            )
        if TRAIN_MEDIATOR_MODEL:
            managed_processes.append(
                ManagedProcess(
                    symbol=symbol,
                    process_type="mediator-train",
                    command=mediator_train_command(),
                    env=child_env,
                )
            )
    for tiny_symbol in PRICE_TINY_SYMBOLS:
        for tiny_venue in PRICE_TINY_PRIMARY_VENUES:
            tiny_env = build_price_tiny_child_env(tiny_symbol, tiny_venue)
            if ENABLE_PRICE_TINY_MODEL or TRAIN_PRICE_TINY_MODEL:
                managed_processes.append(
                    ManagedProcess(
                        symbol=tiny_symbol,
                        venue=tiny_venue,
                        process_type="price-tiny-build",
                        command=price_tiny_build_command(),
                        env=tiny_env,
                    )
                )
            if ENABLE_PRICE_TINY_MODEL:
                managed_processes.append(
                    ManagedProcess(
                        symbol=tiny_symbol,
                        venue=tiny_venue,
                        process_type="price-tiny-show",
                        command=price_tiny_show_command(),
                        env=tiny_env,
                    )
                )
            if TRAIN_PRICE_TINY_MODEL:
                managed_processes.append(
                    ManagedProcess(
                        symbol=tiny_symbol,
                        venue=tiny_venue,
                        process_type="price-tiny-train",
                        command=price_tiny_train_command(),
                        env=tiny_env,
                    )
                )
            if ENABLE_PRICE_TINY_SHADOW_POOL:
                managed_processes.append(
                    ManagedProcess(
                        symbol=tiny_symbol,
                        venue=tiny_venue,
                        process_type="price-tiny-shadow-show",
                        command=price_tiny_shadow_show_command(),
                        env=tiny_env,
                    )
                )
                managed_processes.append(
                    ManagedProcess(
                        symbol=tiny_symbol,
                        venue=tiny_venue,
                        process_type="price-tiny-shadow-evaluate",
                        command=price_tiny_shadow_evaluate_command(),
                        env=tiny_env,
                    )
                )
            if ENABLE_PRICE_TINY_ENSEMBLE_SHOW:
                managed_processes.append(
                    ManagedProcess(
                        symbol=tiny_symbol,
                        venue=tiny_venue,
                        process_type="price-tiny-ensemble-show",
                        command=price_tiny_ensemble_show_command(),
                        env=tiny_env,
                    )
                )
    if ENABLE_MARKET_STACK:
        stack_env = os.environ.copy()
        stack_env["SYMBOLS"] = ",".join(SYMBOLS)
        stack_env["OUTPUT_DIR"] = OUTPUT_DIR
        stack_env["PRIMARY_VENUE"] = PRIMARY_VENUE
        stack_env["FLOW_1S_MODEL_SELECTION"] = FLOW_1S_MODEL_SELECTION
        stack_env["MICRO_MODEL_SELECTION"] = MICRO_MODEL_SELECTION
        stack_env["PROMOTE_BEST"] = PROMOTE_BEST
        stack_env["MICRO_RUN_FLOW_1S_ABLATION"] = MICRO_RUN_FLOW_1S_ABLATION
        stack_env["MICRO_ALLOW_GROUP_DISAGREEMENT"] = MICRO_ALLOW_GROUP_DISAGREEMENT
        stack_env["ENABLE_PRICE_TINY_MODEL"] = str(ENABLE_PRICE_TINY_MODEL).lower()
        stack_env["ENABLE_FLOW_1S_MODEL"] = str(ENABLE_FLOW_1S_MODEL).lower()
        stack_env["ENABLE_MICRO_10S_MODEL"] = str(ENABLE_MICRO_10S_MODEL).lower()
        stack_env["ENABLE_MEDIATOR_MODEL"] = str(ENABLE_MEDIATOR_MODEL).lower()
        stack_env["ENABLE_SHADOW_POOL"] = str(ENABLE_SHADOW_POOL).lower()
        stack_env["ENABLE_PRICE_TINY_SHADOW_POOL"] = str(ENABLE_PRICE_TINY_SHADOW_POOL).lower()
        stack_env["ENABLE_PRICE_TINY_ENSEMBLE_SHOW"] = str(ENABLE_PRICE_TINY_ENSEMBLE_SHOW).lower()
        stack_env["PRICE_TINY_SHADOW_ALLOW_PROMOTION"] = PRICE_TINY_SHADOW_ALLOW_PROMOTION
        stack_env["STACK_LOOP_SECONDS"] = STACK_LOOP_SECONDS
        stack_env["STACK_RUN_ONCE"] = "false"
        stack_env["PYTHONUNBUFFERED"] = "1"
        managed_processes.append(
            ManagedProcess(
                symbol="ALL",
                process_type="market-stack",
                command=market_stack_command(),
                env=stack_env,
            )
        )
    return managed_processes


def print_config():
    print_supervisor("live system supervisor starting")
    print_supervisor(f"symbols: {', '.join(SYMBOLS) if SYMBOLS else '(none)'}")
    print_supervisor(f"venues: {', '.join(VENUES) if VENUES else '(none)'}")
    print_supervisor(f"PRIMARY_VENUE: {PRIMARY_VENUE or 'legacy'}")
    print_supervisor(f"PRICE_TINY_SYMBOLS resolved: {', '.join(PRICE_TINY_SYMBOLS) if PRICE_TINY_SYMBOLS else '(none)'}")
    print_supervisor(
        f"PRICE_TINY_PRIMARY_VENUES resolved: "
        f"{', '.join(PRICE_TINY_PRIMARY_VENUES) if PRICE_TINY_PRIMARY_VENUES else '(none)'}"
    )
    print_supervisor(f"ENABLE_RECORDERS: {ENABLE_RECORDERS}")
    print_supervisor(f"ENABLE_LIVE_LOOPS: {ENABLE_LIVE_LOOPS}")
    print_supervisor(f"ENABLE_MICRO_BUILDERS: {ENABLE_MICRO_BUILDERS}")
    print_supervisor(f"ENABLE_MARKET_STACK: {ENABLE_MARKET_STACK}")
    print_supervisor(f"ENABLE_REGIME_UPDATERS: {ENABLE_REGIME_UPDATERS}")
    print_supervisor(f"ENABLE_HTF_UPDATERS: {ENABLE_HTF_UPDATERS}")
    print_supervisor(f"ENABLE_FLOW_1S_LOOPS: {ENABLE_FLOW_1S_LOOPS}")
    print_supervisor("startup component plan")
    print_supervisor(f"- recorder enabled: {ENABLE_RECORDERS}")
    print_supervisor(f"- price tiny enabled: {ENABLE_PRICE_TINY_MODEL} train={TRAIN_PRICE_TINY_MODEL}")
    print_supervisor(f"- 1s flow enabled: {ENABLE_FLOW_1S_MODEL} train={TRAIN_FLOW_1S_MODEL}")
    print_supervisor(f"- 10s micro enabled: {ENABLE_MICRO_10S_MODEL or ENABLE_MICRO_BUILDERS} train={TRAIN_MICRO_10S_MODEL}")
    print_supervisor(f"- mediator enabled: {ENABLE_MEDIATOR_MODEL} train={TRAIN_MEDIATOR_MODEL}")
    print_supervisor(f"- shadow pool enabled: {ENABLE_SHADOW_POOL}")
    print_supervisor(f"- tiny-price shadow pool enabled: {ENABLE_PRICE_TINY_SHADOW_POOL}")
    print_supervisor(f"- tiny-price ensemble-show enabled: {ENABLE_PRICE_TINY_ENSEMBLE_SHOW}")
    print_supervisor(f"SNAPSHOT_INTERVAL_SECONDS: {SNAPSHOT_INTERVAL_SECONDS}")
    print_supervisor(f"ORDER_BOOK_DEPTH: {ORDER_BOOK_DEPTH}")
    print_supervisor(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print_supervisor(f"LOOP_SECONDS: {LOOP_SECONDS}")
    print_supervisor(f"TRAIN_EVERY_MINUTES: {TRAIN_EVERY_MINUTES}")
    print_supervisor(f"FLOW_1S_MODEL_SELECTION: {FLOW_1S_MODEL_SELECTION}")
    print_supervisor(f"MICRO_MODEL_SELECTION: {MICRO_MODEL_SELECTION}")
    print_supervisor(f"PROMOTE_BEST: {PROMOTE_BEST}")
    print_supervisor(f"MICRO_RUN_FLOW_1S_ABLATION: {MICRO_RUN_FLOW_1S_ABLATION}")
    print_supervisor(f"MICRO_ALLOW_GROUP_DISAGREEMENT: {MICRO_ALLOW_GROUP_DISAGREEMENT}")
    print_supervisor(f"MICRO_BUILD_SECONDS: {MICRO_BUILD_SECONDS}")
    print_supervisor(f"REGIME_UPDATE_SECONDS: {REGIME_UPDATE_SECONDS}")
    print_supervisor(f"HTF_UPDATE_SECONDS: {HTF_UPDATE_SECONDS}")
    print_supervisor(f"FLOW_1S_LOOP_SECONDS: {FLOW_1S_LOOP_SECONDS}")
    print_supervisor(f"PRICE_TINY_SHADOW_SHOW_SECONDS: {PRICE_TINY_SHADOW_SHOW_SECONDS}")
    print_supervisor(f"PRICE_TINY_SHADOW_EVALUATE_SECONDS: {PRICE_TINY_SHADOW_EVALUATE_SECONDS}")
    print_supervisor(f"PRICE_TINY_ENSEMBLE_SHOW_SECONDS: {PRICE_TINY_ENSEMBLE_SHOW_SECONDS}")
    print_supervisor(f"PRICE_TINY_SHADOW_ALLOW_PROMOTION: {PRICE_TINY_SHADOW_ALLOW_PROMOTION}")
    print_supervisor(
        f"PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES: "
        f"{PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES if PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES else '(unset/all)'}"
    )
    print_supervisor(f"PRICE_TINY_MODEL_PATH set: {bool(os.getenv('PRICE_TINY_MODEL_PATH', '').strip())}")
    global_model_path_set = bool(os.getenv("PRICE_TINY_MODEL_PATH", "").strip())
    multi_tiny_pairs = len(PRICE_TINY_SYMBOLS) * len(PRICE_TINY_PRIMARY_VENUES) > 1
    if global_model_path_set and multi_tiny_pairs:
        print_supervisor(
            "WARNING: global PRICE_TINY_MODEL_PATH is set but multiple tiny-price "
            "symbol/venue pairs are enabled. It will be ignored unless a scoped "
            "PRICE_TINY_MODEL_PATH_<SYMBOL>_<VENUE> env var exists."
        )
    print_supervisor("tiny-price scoped champion path resolution")
    for symbol in PRICE_TINY_SYMBOLS:
        for venue in PRICE_TINY_PRIMARY_VENUES:
            source_name, model_path = resolve_scoped_price_tiny_model_path(symbol, venue)
            print_supervisor(
                f"- [{symbol} {venue}] champion_path_source={source_name or 'selected_model_or_registry'} "
                f"path_set={bool(model_path)}"
            )
    tiny_build_train_count = 0
    tiny_shadow_count = 0
    tiny_ensemble_count = 0
    if ENABLE_PRICE_TINY_MODEL or TRAIN_PRICE_TINY_MODEL:
        tiny_build_train_count += len(PRICE_TINY_SYMBOLS) * len(PRICE_TINY_PRIMARY_VENUES)
    if TRAIN_PRICE_TINY_MODEL:
        tiny_build_train_count += len(PRICE_TINY_SYMBOLS) * len(PRICE_TINY_PRIMARY_VENUES)
    if ENABLE_PRICE_TINY_SHADOW_POOL:
        tiny_shadow_count += 2 * len(PRICE_TINY_SYMBOLS) * len(PRICE_TINY_PRIMARY_VENUES)
    if ENABLE_PRICE_TINY_ENSEMBLE_SHOW:
        tiny_ensemble_count += len(PRICE_TINY_SYMBOLS) * len(PRICE_TINY_PRIMARY_VENUES)
    print_supervisor(f"total tiny-price build/train process count: {tiny_build_train_count}")
    print_supervisor(f"total tiny-price shadow process count: {tiny_shadow_count}")
    print_supervisor(f"total tiny-price ensemble-show process count: {tiny_ensemble_count}")
    print_supervisor("tiny-price process labels")
    for symbol in PRICE_TINY_SYMBOLS:
        for venue in PRICE_TINY_PRIMARY_VENUES:
            if ENABLE_PRICE_TINY_MODEL or TRAIN_PRICE_TINY_MODEL:
                print_supervisor(f"- [{symbol} {venue} price-tiny-build]")
            if ENABLE_PRICE_TINY_MODEL:
                print_supervisor(f"- [{symbol} {venue} price-tiny-show]")
            if TRAIN_PRICE_TINY_MODEL:
                print_supervisor(f"- [{symbol} {venue} price-tiny-train]")
            if ENABLE_PRICE_TINY_SHADOW_POOL:
                print_supervisor(f"- [{symbol} {venue} price-tiny-shadow-show]")
                print_supervisor(f"- [{symbol} {venue} price-tiny-shadow-evaluate]")
            if ENABLE_PRICE_TINY_ENSEMBLE_SHOW:
                print_supervisor(f"- [{symbol} {venue} price-tiny-ensemble-show]")
    print_supervisor(f"MAX_MICRO_SNAPSHOTS: {MAX_MICRO_SNAPSHOTS}")
    print_supervisor(f"STACK_LOOP_SECONDS: {STACK_LOOP_SECONDS}")
    print_supervisor(f"RESTART_DELAY_SECONDS: {RESTART_DELAY_SECONDS}")
    print_supervisor("No trades are placed by this supervisor.")
    print_supervisor("Training gates remain inside the existing live training script.")


def main():
    print_config()
    if not SYMBOLS:
        raise RuntimeError("SYMBOLS is empty.")
    if (
        ENABLE_PRICE_TINY_MODEL
        or TRAIN_PRICE_TINY_MODEL
        or ENABLE_PRICE_TINY_SHADOW_POOL
        or ENABLE_PRICE_TINY_ENSEMBLE_SHOW
    ) and (
        not PRICE_TINY_SYMBOLS or not PRICE_TINY_PRIMARY_VENUES
    ):
        raise RuntimeError("PRICE_TINY_SYMBOLS or PRICE_TINY_PRIMARY_VENUES resolved to an empty list.")
    if (
        not ENABLE_RECORDERS
        and not ENABLE_LIVE_LOOPS
        and not ENABLE_MICRO_BUILDERS
        and not ENABLE_PRICE_TINY_MODEL
        and not ENABLE_FLOW_1S_MODEL
        and not ENABLE_MICRO_10S_MODEL
        and not ENABLE_MEDIATOR_MODEL
        and not ENABLE_PRICE_TINY_SHADOW_POOL
        and not ENABLE_PRICE_TINY_ENSEMBLE_SHOW
        and not ENABLE_REGIME_UPDATERS
        and not ENABLE_HTF_UPDATERS
        and not ENABLE_MARKET_STACK
    ):
        raise RuntimeError(
            "ENABLE_RECORDERS, ENABLE_LIVE_LOOPS, ENABLE_MICRO_BUILDERS, "
            "ENABLE_REGIME_UPDATERS, ENABLE_HTF_UPDATERS, ENABLE_FLOW_1S_LOOPS, "
            "ENABLE_PRICE_TINY_SHADOW_POOL, ENABLE_PRICE_TINY_ENSEMBLE_SHOW, "
            "and ENABLE_MARKET_STACK are all false."
        )
    if not ENABLE_RECORDERS and (
        ENABLE_PRICE_TINY_MODEL
        or ENABLE_FLOW_1S_MODEL
        or ENABLE_MICRO_10S_MODEL
        or ENABLE_MEDIATOR_MODEL
        or ENABLE_SHADOW_POOL
        or ENABLE_PRICE_TINY_SHADOW_POOL
        or ENABLE_PRICE_TINY_ENSEMBLE_SHOW
    ):
        raise RuntimeError(
            "Fail closed: ENABLE_RECORDERS=false while at least one live prediction "
            "component requires fresh snapshots. Disable the live component toggles or enable recorders."
        )

    managed_processes = create_managed_processes()
    for managed in managed_processes:
        thread = threading.Thread(
            target=monitor_process,
            args=(managed,),
            daemon=True,
        )
        managed.monitor_thread = thread
        thread.start()

    try:
        while any(thread.monitor_thread.is_alive() for thread in managed_processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print_supervisor("Ctrl+C received. Stopping child processes...")
    finally:
        stopping.set()
        for managed in managed_processes:
            stop_process(managed)

        deadline = time.time() + STOP_TIMEOUT_SECONDS
        while time.time() < deadline:
            if all(
                managed.process is None or managed.process.poll() is not None
                for managed in managed_processes
            ):
                break
            time.sleep(0.25)

        for managed in managed_processes:
            kill_if_still_running(managed)

        print_supervisor("live system supervisor stopped")


if __name__ == "__main__":
    main()
