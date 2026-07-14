import csv
import bisect
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST = os.getenv("CONTROL_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.getenv("CONTROL_DASHBOARD_PORT", "8770"))
IS_WINDOWS = os.name == "nt"
MAX_LOG_CHARS = int(os.getenv("CONTROL_DASHBOARD_MAX_LOG_CHARS", "80000"))

def env_default(name, default):
    return os.getenv(name, default)


DEFAULT_STATE = {
    "SYMBOL": env_default("SYMBOL", "SOLUSDT"),
    "SYMBOLS": env_default("SYMBOLS", "SOLUSDT"),
    "PRIMARY_VENUE": env_default("PRIMARY_VENUE", "kraken"),
    "VENUES": env_default("VENUES", "kraken,binanceus"),
    "FLOW_1S_MODEL_SELECTION": env_default("FLOW_1S_MODEL_SELECTION", "latest_candidate"),
    "MICRO_MODEL_SELECTION": env_default("MICRO_MODEL_SELECTION", "latest_candidate"),
    "PROMOTE_BEST": env_default("PROMOTE_BEST", "false"),
    "TRAIN_PRICE_TINY_MODEL": env_default("TRAIN_PRICE_TINY_MODEL", "false"),
    "PRICE_TINY_AUTO_REGISTER_CHALLENGERS": env_default("PRICE_TINY_AUTO_REGISTER_CHALLENGERS", "false"),
    "ENABLE_PRICE_TINY_SHADOW_POOL": env_default("ENABLE_PRICE_TINY_SHADOW_POOL", "false"),
    "ENABLE_PRICE_TINY_ENSEMBLE_SHOW": env_default("ENABLE_PRICE_TINY_ENSEMBLE_SHOW", "false"),
    "PRICE_TINY_ENSEMBLE_SHOW_SECONDS": env_default("PRICE_TINY_ENSEMBLE_SHOW_SECONDS", "15"),
    "PRICE_TINY_ENSEMBLE_RUN_ID": env_default("PRICE_TINY_ENSEMBLE_RUN_ID", ""),
    "PRICE_TINY_ENSEMBLE_MOVE_TARGET_SPEC": env_default("PRICE_TINY_ENSEMBLE_MOVE_TARGET_SPEC", "move_before_adverse_30s"),
    "PRICE_TINY_ENSEMBLE_INSTABILITY_TARGET_SPEC": env_default("PRICE_TINY_ENSEMBLE_INSTABILITY_TARGET_SPEC", "instability_30s"),
    "PRICE_TINY_ENSEMBLE_FEATURE_GROUPS": env_default("PRICE_TINY_ENSEMBLE_FEATURE_GROUPS", ""),
    "PRICE_TINY_ENSEMBLE_FEATURE_SCHEMA_HASH": env_default("PRICE_TINY_ENSEMBLE_FEATURE_SCHEMA_HASH", ""),
    "PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD": env_default("PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD", "0.60"),
    "PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD": env_default("PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD", "0.70"),
    "PRICE_TINY_ENSEMBLE_ALLOWED_SIDES": env_default("PRICE_TINY_ENSEMBLE_ALLOWED_SIDES", "both"),
    "PRICE_TINY_ENSEMBLE_MOVE_MODEL_PATH": env_default("PRICE_TINY_ENSEMBLE_MOVE_MODEL_PATH", ""),
    "PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_PATH": env_default("PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_PATH", ""),
    "PRICE_TINY_ENSEMBLE_MOVE_MODEL_ID": env_default("PRICE_TINY_ENSEMBLE_MOVE_MODEL_ID", ""),
    "PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_ID": env_default("PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_ID", ""),
    "PRICE_TINY_ENSEMBLE_REQUIRE_DIRECTION_AGREEMENT": env_default("PRICE_TINY_ENSEMBLE_REQUIRE_DIRECTION_AGREEMENT", "false"),
    "PRICE_TINY_ENSEMBLE_ENABLE_REGRESSION": env_default("PRICE_TINY_ENSEMBLE_ENABLE_REGRESSION", "false"),
    "FORECAST_CHART_ROWS": env_default("FORECAST_CHART_ROWS", "500"),
    "MICRO_RUN_FLOW_1S_ABLATION": env_default("MICRO_RUN_FLOW_1S_ABLATION", "false"),
    "MICRO_ALLOW_GROUP_DISAGREEMENT": env_default("MICRO_ALLOW_GROUP_DISAGREEMENT", "false"),
    "MAX_MICRO_SNAPSHOTS": env_default("MAX_MICRO_SNAPSHOTS", "2000"),
    "VENUE": env_default("VENUE", "kraken"),
    "TRADE_START_TIME": env_default("TRADE_START_TIME", ""),
    "TRADE_END_TIME": env_default("TRADE_END_TIME", ""),
    "SIM_SECONDS": env_default("SIM_SECONDS", "7200"),
    "BINANCE_START_MONTH": env_default("BINANCE_START_MONTH", ""),
    "BINANCE_END_MONTH": env_default("BINANCE_END_MONTH", ""),
}

STATE_LOCK = threading.Lock()
DASHBOARD_STATE = dict(DEFAULT_STATE)
PROCESS_LOCK = threading.Lock()
PROCESSES = {}


COMMANDS = [
    {"id": "live-system", "script": "live-system", "label": "live-system", "section": "Live orchestration", "long_running": True},
    {"id": "night-run", "script": "night-run", "label": "night-run", "section": "Live orchestration", "long_running": True},
    {"id": "realtime-kraken-multi-recorder", "script": "realtime-kraken-multi-recorder", "label": "realtime-kraken-multi-recorder", "section": "Rawseq 1m live", "long_running": True},
    {"id": "rawseq1m-live-paper-dashboard", "script": "rawseq1m-live-paper-dashboard", "label": "rawseq1m-live-paper-dashboard", "section": "Rawseq 1m live", "long_running": False},
    {"id": "rawseq1m-predict", "script": "rawseq1m-predict", "label": "rawseq1m-predict", "section": "Rawseq 1m live", "long_running": False},
    {"id": "rawseq1m-predict-all", "script": "rawseq1m-predict-all", "label": "rawseq1m-predict-all", "section": "Rawseq 1m live", "long_running": False},
    {"id": "market-stack", "script": "market-stack", "label": "market-stack", "section": "Live monitoring", "long_running": False},
    {"id": "forecast1s-gui", "script": "forecast1s-gui", "label": "forecast1s-gui", "section": "Live monitoring", "long_running": True},
    {"id": "paper-signal-log", "script": "paper-signal-log", "label": "paper-signal-log", "section": "Live monitoring", "long_running": False},
    {"id": "hierarchy-log", "script": "hierarchy-log", "label": "hierarchy-log", "section": "Live monitoring", "long_running": False},
    {"id": "hierarchy-log-loop", "script": "hierarchy-log-loop", "label": "hierarchy-log-loop", "section": "Live monitoring", "long_running": True},
    {"id": "hierarchy-evaluate", "script": "hierarchy-evaluate", "label": "hierarchy-evaluate", "section": "Live monitoring", "long_running": False},
    {"id": "tiny-price-ensemble-show", "script": "tiny-price-ensemble-show", "label": "tiny-price-ensemble-show", "section": "Tiny price ensemble", "long_running": False},
    {"id": "tiny-price-ensemble-live-evaluate", "script": "tiny-price-ensemble-live-evaluate", "label": "tiny-price-ensemble-live-evaluate", "section": "Tiny price ensemble", "long_running": False},
    {"id": "tiny-price-build", "script": "tiny-price-build", "label": "tiny-price-build", "section": "Tiny price research", "long_running": False},
    {"id": "tiny-price-train", "script": "tiny-price-train", "label": "tiny-price-train", "section": "Tiny price research", "long_running": False},
    {"id": "tiny-price-walk-forward-evaluate", "script": "tiny-price-walk-forward-evaluate", "label": "tiny-price-walk-forward-evaluate", "section": "Tiny price research", "long_running": False},
    {"id": "tiny-price-walk-forward-ensemble", "script": "tiny-price-walk-forward-ensemble", "label": "tiny-price-walk-forward-ensemble", "section": "Tiny price research", "long_running": False},
    {"id": "tiny-price-feature-ablation-walk-forward", "script": "tiny-price-feature-ablation-walk-forward", "label": "tiny-price-feature-ablation-walk-forward", "section": "Tiny price research", "long_running": False},
    {"id": "binance-trades-fetch", "script": "binance-trades-fetch", "label": "binance-trades-fetch", "section": "Historical replay", "long_running": False},
    {"id": "replay-market-from-trades", "script": "replay-market-from-trades", "label": "replay-market-from-trades", "section": "Historical replay", "long_running": False, "optional": True},
    {"id": "validate-replay", "script": "validate-replay", "label": "validate-replay", "section": "Historical replay", "long_running": False, "optional": True},
    {"id": "flow1s-show", "script": "flow1s-show", "label": "flow1s-show", "section": "1s order-flow", "long_running": False},
    {"id": "flow1s-evaluate", "script": "flow1s-evaluate", "label": "flow1s-evaluate", "section": "1s order-flow", "long_running": False},
    {"id": "micro-build", "script": "micro-build", "label": "micro-build", "section": "10s microstructure", "long_running": False},
    {"id": "micro-train", "script": "micro-train", "label": "micro-train", "section": "10s microstructure", "long_running": False},
    {"id": "micro-show", "script": "micro-show", "label": "micro-show", "section": "10s microstructure", "long_running": False},
    {"id": "regime-update", "script": "regime-update", "label": "regime-update", "section": "Higher timeframe context", "long_running": False, "optional": True},
    {"id": "live-env-check", "script": "live-env-check", "label": "live-env-check", "section": "Diagnostics", "long_running": False, "optional": True},
]

FILE_SPECS = [
    ("10s flow CSV", "venue", "{SYMBOL}_10s_flow.csv", 120),
    ("1m flow CSV", "venue", "{SYMBOL}_1m_flow.csv", 180),
    ("1s prediction CSV", "venue", "{SYMBOL}_1s_order_flow_predictions.csv", 120),
    ("1s loop diagnostics", "venue", "{SYMBOL}_1s_order_flow_loop_diagnostics.json", 120),
    ("10s micro training rows", "venue", "{SYMBOL}_10s_microstructure_training_rows.csv", 300),
    ("10s micro predictions", "venue", "{SYMBOL}_10s_microstructure_predictions.csv", 120),
    ("paper signal log", "venue", "{SYMBOL}_paper_signal_log.csv", 600),
    ("hierarchy forecast log", "venue", "{SYMBOL}_hierarchy_forecast_log.csv", 600),
    ("hierarchy forecast evaluation", "venue", "{SYMBOL}_hierarchy_forecast_evaluation.csv", 3600),
    ("tiny ensemble live predictions", "venue", "{SYMBOL}_tiny_price_ensemble_live_predictions.csv", 300),
    ("tiny ensemble evaluated rows", "venue", "{SYMBOL}_tiny_price_ensemble_live_evaluated_rows.csv", 600),
    ("tiny ensemble evaluation", "venue", "{SYMBOL}_tiny_price_ensemble_live_evaluation.csv", 3600),
    ("tiny ensemble evaluation by run", "venue", "{SYMBOL}_tiny_price_ensemble_live_evaluation_by_run.csv", 3600),
    ("tiny ensemble threshold sensitivity", "venue", "{SYMBOL}_tiny_price_ensemble_threshold_sensitivity.csv", 3600),
    ("tiny ensemble side/regime diagnostics", "venue", "{SYMBOL}_tiny_price_ensemble_side_regime_diagnostics.csv", 3600),
    ("tiny price walk-forward target eval", "venue", "{SYMBOL}_tiny_price_walk_forward_{PRICE_TINY_ENSEMBLE_MOVE_TARGET_SPEC}.csv", 3600),
    ("replay validation", "replayed_venue", "{SYMBOL}_replay_validation.csv", 3600),
    ("1s forecast evaluation CSV", "venue", "{SYMBOL}_1s_forecast_evaluation.csv", 3600),
    ("15m regime features", "data", "{SYMBOL}_15m_regime_features.csv", 3600),
    ("30m regime features", "data", "{SYMBOL}_30m_regime_features.csv", 3600),
    ("HTF context features", "data", "{SYMBOL}_htf_context_features.csv", 3 * 3600),
]

PRIVATE_ENV_MARKERS = (
    "API_KEY",
    "API_SECRET",
    "SECRET_KEY",
    "PRIVATE_KEY",
    "ACCESS_TOKEN",
    "REFRESH_TOKEN",
    "CLIENT_SECRET",
)


def npm_command():
    return "npm.cmd" if IS_WINDOWS else "npm"


def load_package_scripts():
    try:
        with (PROJECT_ROOT / "package.json").open("r", encoding="utf-8") as handle:
            return json.load(handle).get("scripts", {})
    except Exception:
        return {}


def available_commands():
    scripts = load_package_scripts()
    output = []
    for command in COMMANDS:
        if command.get("optional") and command["script"] not in scripts:
            continue
        item = dict(command)
        item["available"] = command["script"] in scripts
        output.append(item)
    return output


def command_by_id(command_id):
    for command in available_commands():
        if command["id"] == command_id:
            return command
    return None


def sanitize_state(payload):
    clean = {}
    for key, default in DEFAULT_STATE.items():
        value = str(payload.get(key, default)).strip()
        clean[key] = value if value else default
    clean["SYMBOL"] = clean["SYMBOL"].upper()
    clean["SYMBOLS"] = ",".join(
        symbol.strip().upper()
        for symbol in clean["SYMBOLS"].split(",")
        if symbol.strip()
    ) or clean["SYMBOL"]
    clean["PRIMARY_VENUE"] = clean["PRIMARY_VENUE"].lower()
    clean["VENUES"] = ",".join(
        venue.strip().lower()
        for venue in clean["VENUES"].split(",")
        if venue.strip()
    ) or clean["PRIMARY_VENUE"]
    for bool_key in [
        "PROMOTE_BEST",
        "TRAIN_PRICE_TINY_MODEL",
        "PRICE_TINY_AUTO_REGISTER_CHALLENGERS",
        "ENABLE_PRICE_TINY_SHADOW_POOL",
        "ENABLE_PRICE_TINY_ENSEMBLE_SHOW",
        "PRICE_TINY_ENSEMBLE_REQUIRE_DIRECTION_AGREEMENT",
        "PRICE_TINY_ENSEMBLE_ENABLE_REGRESSION",
        "MICRO_RUN_FLOW_1S_ABLATION",
        "MICRO_ALLOW_GROUP_DISAGREEMENT",
    ]:
        clean[bool_key] = "true" if clean[bool_key].lower() in {"true", "1", "yes", "y"} else "false"
    clean["PRICE_TINY_ENSEMBLE_ALLOWED_SIDES"] = clean["PRICE_TINY_ENSEMBLE_ALLOWED_SIDES"].lower()
    if clean["PRICE_TINY_ENSEMBLE_ALLOWED_SIDES"] not in {"long", "short", "both"}:
        clean["PRICE_TINY_ENSEMBLE_ALLOWED_SIDES"] = "both"
    clean["VENUE"] = clean["VENUE"].lower()
    return clean


def current_state():
    with STATE_LOCK:
        return dict(DASHBOARD_STATE)


def update_state(payload):
    clean = sanitize_state(payload)
    with STATE_LOCK:
        DASHBOARD_STATE.update(clean)
        return dict(DASHBOARD_STATE)


def build_child_env(state):
    env = os.environ.copy()
    # This dashboard is deliberately paper-only. Strip common private/account
    # credential variables from child processes so no button can accidentally
    # authorize account/trading behavior.
    for key in list(env.keys()):
        upper = key.upper()
        if any(marker in upper for marker in PRIVATE_ENV_MARKERS):
            env.pop(key, None)
    env.update(state)
    env["PYTHONUNBUFFERED"] = "1"
    # Dashboard-launched jobs are research/paper-only. These are hard-forced
    # even if the UI or parent shell contains different values.
    env["PROMOTE_BEST"] = "false"
    env["PRICE_TINY_AUTO_REGISTER_CHALLENGERS"] = "false"
    env.setdefault("TRAIN_PRICE_TINY_MODEL", "false")
    return env


class ManagedProcess:
    def __init__(self, command_info, env_state):
        self.id = uuid.uuid4().hex[:12]
        self.command_info = command_info
        self.env_state = dict(env_state)
        self.env_state["PROMOTE_BEST"] = "false"
        self.env_state["PRICE_TINY_AUTO_REGISTER_CHALLENGERS"] = "false"
        self.command = [npm_command(), "run", command_info["script"]]
        self.started_at = time.time()
        self.finished_at = None
        self.returncode = None
        self.status = "starting"
        self.output = deque()
        self.output_chars = 0
        self.process = None
        self.thread = None

    def append_output(self, text):
        if not text:
            return
        self.output.append(text)
        self.output_chars += len(text)
        while self.output_chars > MAX_LOG_CHARS and self.output:
            removed = self.output.popleft()
            self.output_chars -= len(removed)

    def output_text(self):
        return "".join(self.output)

    def to_dict(self):
        return {
            "id": self.id,
            "command_id": self.command_info["id"],
            "script": self.command_info["script"],
            "label": self.command_info["label"],
            "command": " ".join(self.command),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "running": self.process is not None and self.process.poll() is None,
            "output": self.output_text(),
            "env": self.env_state,
        }

    def start(self):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
        self.process = subprocess.Popen(
            self.command,
            cwd=PROJECT_ROOT,
            env=build_child_env(self.env_state),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
        self.status = "running"
        self.thread = threading.Thread(target=self._read_output, daemon=True)
        self.thread.start()

    def _read_output(self):
        self.append_output(
            f"$ {' '.join(self.command)}\n"
            f"SYMBOL={self.env_state.get('SYMBOL')} "
            f"PRIMARY_VENUE={self.env_state.get('PRIMARY_VENUE')} "
            f"PROMOTE_BEST={self.env_state.get('PROMOTE_BEST')}\n"
            "Paper-only dashboard: no trades/orders/private API keys.\n\n"
        )
        try:
            for line in self.process.stdout:
                self.append_output(line)
        except Exception as error:
            self.append_output(f"\n[dashboard] output reader error: {error}\n")
        self.returncode = self.process.wait()
        self.finished_at = time.time()
        self.status = "finished" if self.returncode == 0 else "failed"
        self.append_output(f"\n[dashboard] process finished with code {self.returncode}\n")

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return False
        self.status = "stopping"
        self.append_output("\n[dashboard] stop requested\n")
        try:
            if IS_WINDOWS:
                self.process.terminate()
            else:
                self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
        finally:
            self.returncode = self.process.poll()
            self.finished_at = time.time()
            self.status = "stopped"
            self.append_output("\n[dashboard] process stopped\n")
        return True


def running_duplicate(command_id):
    with PROCESS_LOCK:
        for process in PROCESSES.values():
            if process.command_info["id"] == command_id and process.process is not None and process.process.poll() is None:
                return process
    return None


def start_command(command_id, env_state, allow_duplicate=False):
    command = command_by_id(command_id)
    if command is None:
        return None, f"Unknown or unavailable command: {command_id}"
    if not command.get("available", False):
        return None, f"npm script is not available: {command['script']}"
    duplicate = running_duplicate(command_id)
    if duplicate is not None and not allow_duplicate:
        return None, f"{command['label']} is already running as process {duplicate.id}"
    process = ManagedProcess(command, env_state)
    with PROCESS_LOCK:
        PROCESSES[process.id] = process
    try:
        process.start()
    except Exception as error:
        process.status = "failed"
        process.returncode = -1
        process.finished_at = time.time()
        process.append_output(f"[dashboard] failed to start: {error}\n")
        return process, str(error)
    return process, None


def stop_process(process_id):
    with PROCESS_LOCK:
        process = PROCESSES.get(process_id)
    if process is None:
        return False, "process not found"
    stopped = process.stop()
    return stopped, None if stopped else "process is not running"


def normalize_timestamp_ms(value):
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if not timestamp or not (timestamp == timestamp):
        return None
    if timestamp < 10_000_000_000:
        timestamp *= 1000
    return int(timestamp)


def file_path_for_spec(state, location, pattern):
    symbol = state["SYMBOL"]
    primary_venue = state["PRIMARY_VENUE"]
    if location == "venue":
        base = PROJECT_ROOT / "data" / "realtime" / primary_venue if primary_venue else PROJECT_ROOT / "data" / "realtime"
    elif location == "replayed_venue":
        venue = state.get("VENUE", primary_venue) or primary_venue
        base = PROJECT_ROOT / "data" / "realtime" / "replayed" / venue
    else:
        base = PROJECT_ROOT / "data"
    fmt_state = dict(state)
    fmt_state["SYMBOL"] = symbol
    return base / pattern.format(**fmt_state)


def file_status(path, stale_seconds):
    path = Path(path)
    if not path.exists():
        return {
            "exists": False,
            "row_count": 0,
            "latest_timestamp": None,
            "latest_time": "",
            "age_seconds": None,
            "freshness": "missing",
        }
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            row_count = 0
            last_row = None
            for row in reader:
                if not row:
                    continue
                row_count += 1
                last_row = row
    except Exception as error:
        return {
            "exists": True,
            "row_count": 0,
            "latest_timestamp": None,
            "latest_time": "",
            "age_seconds": None,
            "freshness": f"read error: {error}",
        }
    timestamp = None
    latest_time = ""
    if last_row and header:
        timestamp_index = next(
            (header.index(column) for column in ["close_timestamp", "timestamp"] if column in header),
            None,
        )
        time_index = header.index("time") if "time" in header else None
        if timestamp_index is not None:
            timestamp = normalize_timestamp_ms(last_row[timestamp_index] if timestamp_index < len(last_row) else None)
        if time_index is not None and time_index < len(last_row):
            latest_time = last_row[time_index]
    age_seconds = None
    freshness = "no timestamp"
    if timestamp is not None:
        age_seconds = max(0.0, time.time() - timestamp / 1000.0)
        freshness = "fresh" if age_seconds <= stale_seconds else "stale"
    return {
        "exists": True,
        "row_count": row_count,
        "latest_timestamp": timestamp,
        "latest_time": latest_time,
        "age_seconds": age_seconds,
        "freshness": freshness,
    }


def status_payload():
    state = current_state()
    files = []
    for label, location, pattern, stale_seconds in FILE_SPECS:
        path = file_path_for_spec(state, location, pattern)
        info = file_status(path, stale_seconds)
        files.append(
            {
                "label": label,
                "path": str(path),
                "stale_seconds": stale_seconds,
                **info,
            }
        )
    diagnostics = one_second_loop_diagnostics(state)
    return {
        "state": state,
        "files": files,
        "flow_1s_loop_diagnostics": diagnostics,
        "commands": available_commands(),
        "safety": {
            "paper_only": True,
            "orders_enabled": False,
            "private_api_keys_passed_to_children": False,
        },
    }


def one_second_loop_diagnostics(state):
    path = file_path_for_spec(state, "venue", "{SYMBOL}_1s_order_flow_loop_diagnostics.json")
    if not path.exists():
        return {"exists": False, "path": str(path), "status": "missing diagnostics file"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"exists": True, "path": str(path), "status": f"read error: {error}"}
    payload["exists"] = True
    payload["path"] = str(path)
    return payload


def parse_float(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def parse_int(value, default=None):
    number = parse_float(value)
    if number is None:
        return default
    return int(number)


def read_csv_tail(path, max_rows=2000):
    path = Path(path)
    if not path.exists():
        return [], str(path), "missing"
    rows = deque(maxlen=max(1, int(max_rows)))
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row:
                    rows.append(row)
    except Exception as error:
        return [], str(path), f"read_error:{error}"
    return list(rows), str(path), "ok"


def truthy_text(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def direction_from_signal(value):
    text_value = str(value or "").strip().lower()
    if text_value in {"long", "buy", "up", "1"}:
        return 1
    if text_value in {"short", "sell", "down", "-1"}:
        return -1
    return 0


def load_recent_snapshots(state, max_rows):
    snapshot_path = file_path_for_spec(state, "venue", "{SYMBOL}_10s_flow.csv")
    rows, path, status = read_csv_tail(snapshot_path, max_rows=max_rows)
    points = []
    for row in rows:
        timestamp = normalize_timestamp_ms(row.get("timestamp"))
        mid_price = parse_float(row.get("mid_price"), None)
        if mid_price is None:
            mid_price = parse_float(row.get("close"), None)
        if timestamp is None or mid_price is None or mid_price <= 0:
            continue
        points.append(
            {
                "timestamp": timestamp,
                "time": row.get("time", ""),
                "mid_price": mid_price,
                "spread_percent": parse_float(row.get("spread_percent"), None),
                "best_bid": parse_float(row.get("best_bid"), None),
                "best_ask": parse_float(row.get("best_ask"), None),
            }
        )
    points.sort(key=lambda item: item["timestamp"])
    deduped = []
    for point in points:
        if deduped and deduped[-1]["timestamp"] == point["timestamp"]:
            deduped[-1] = point
        else:
            deduped.append(point)
    return deduped, {"path": path, "status": status, "rows_loaded": len(deduped)}


def find_point_at_or_before(points, timestamps, target_timestamp):
    index = bisect.bisect_right(timestamps, target_timestamp) - 1
    if index < 0:
        return None, None
    return points[index], index


def find_point_at_or_after(points, timestamps, target_timestamp, max_gap_ms):
    index = bisect.bisect_left(timestamps, target_timestamp)
    if index >= len(points):
        return None, None
    point = points[index]
    if point["timestamp"] - target_timestamp > max_gap_ms:
        return None, None
    return point, index


def latest_evaluation_row(state):
    summary_path = file_path_for_spec(state, "venue", "{SYMBOL}_tiny_price_ensemble_live_evaluation.csv")
    rows, path, status = read_csv_tail(summary_path, max_rows=1)
    return {
        "path": path,
        "status": status,
        "row": rows[-1] if rows else {},
    }


def sample_strength(signals):
    signals = parse_int(signals, 0) or 0
    if signals < 25:
        return "low", "<25 signals"
    if signals >= 100:
        return "strong", "100+ signals"
    if signals >= 50:
        return "useful", "50+ signals"
    return "warming", "25-49 signals"


def run_comparison_payload(state):
    by_run_path = file_path_for_spec(state, "venue", "{SYMBOL}_tiny_price_ensemble_live_evaluation_by_run.csv")
    eval_rows, eval_path_text, eval_status = read_csv_tail(by_run_path, max_rows=500)

    predictions_path = file_path_for_spec(state, "venue", "{SYMBOL}_tiny_price_ensemble_live_predictions.csv")
    prediction_rows, prediction_path_text, prediction_status = read_csv_tail(predictions_path, max_rows=5000)
    metadata_by_run = {}
    for row in prediction_rows:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        metadata_key = (
            run_id,
            str(row.get("move_model_id", "")).strip(),
            str(row.get("instability_model_id", "")).strip(),
        )
        metadata_by_run[metadata_key] = {
            "feature_groups": row.get("active_feature_groups", row.get("move_feature_groups", "")),
            "schema_hash": row.get("active_feature_schema_hash", row.get("move_schema_hash", "")),
            "move_threshold": row.get("active_move_confidence_threshold", row.get("move_confidence_threshold", "")),
            "instability_threshold": row.get("active_instability_threshold", row.get("instability_max_probability", "")),
            "allowed_sides": row.get("active_allowed_sides", row.get("allowed_sides", "")),
            "move_model_id": row.get("move_model_id", ""),
            "instability_model_id": row.get("instability_model_id", ""),
            "model_pinning_status": row.get("model_pinning_status", ""),
            "models_pinned": row.get("models_pinned", ""),
        }

    rows = []
    seen = set()
    for row in reversed(eval_rows):
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        move_model_id = str(row.get("move_model_id", "")).strip()
        instability_model_id = str(row.get("instability_model_id", "")).strip()
        group_identity = (run_id, move_model_id, instability_model_id)
        if group_identity in seen:
            continue
        seen.add(group_identity)
        meta = metadata_by_run.get(group_identity) or metadata_by_run.get((run_id, "", "")) or {}
        signal_count = parse_int(row.get("paper_signal_rows", row.get("signal_rows", "")), 0) or 0
        sample_class, sample_label = sample_strength(signal_count)
        rows.append(
            {
                "run_id": run_id,
                "feature_groups": meta.get("feature_groups", ""),
                "schema_hash": meta.get("schema_hash", ""),
                "move_model_id": move_model_id or meta.get("move_model_id", ""),
                "instability_model_id": instability_model_id or meta.get("instability_model_id", ""),
                "model_pinning_status": row.get("model_pinning_status", meta.get("model_pinning_status", "")),
                "models_pinned": row.get("models_pinned", meta.get("models_pinned", "")),
                "move_threshold": parse_float(row.get("active_move_confidence_threshold", meta.get("move_threshold")), None),
                "instability_threshold": parse_float(row.get("active_instability_threshold", meta.get("instability_threshold")), None),
                "allowed_sides": row.get("active_allowed_sides", meta.get("allowed_sides", "")),
                "rows": parse_int(row.get("evaluated_rows", row.get("total_rows", "")), 0),
                "signals": signal_count,
                "long": parse_int(row.get("long_count", row.get("long_rows", "")), 0),
                "short": parse_int(row.get("short_count", row.get("short_rows", "")), 0),
                "sign_acc": parse_float(row.get("sign_accuracy", row.get("gross_sign_accuracy", "")), None),
                "gross_bps": parse_float(row.get("gross_avg_strategy_return_bps", row.get("average_realized_30s_strategy_return_bps", "")), None),
                "net_bps": parse_float(row.get("net_avg_strategy_return_bps"), None),
                "net_positive_rate": parse_float(row.get("net_positive_rate"), None),
                "stale_rows": parse_int(row.get("stale_rows"), 0),
                "schema_mismatch_rows": parse_int(row.get("required_schema_mismatch_rows"), 0),
                "sample_class": sample_class,
                "sample_label": sample_label,
            }
        )
    rows.reverse()
    return {
        "path": eval_path_text,
        "status": eval_status,
        "prediction_path": prediction_path_text,
        "prediction_status": prediction_status,
        "rows": rows[-100:],
    }


def forecast_actual_payload(query):
    state = current_state()
    symbol = (query.get("symbol", [state["SYMBOL"]])[0] or state["SYMBOL"]).upper()
    venue = (query.get("venue", [state["PRIMARY_VENUE"]])[0] or state["PRIMARY_VENUE"]).lower()
    rows_requested = parse_int(query.get("rows", [state.get("FORECAST_CHART_ROWS", "500")])[0], 500)
    rows_requested = max(25, min(rows_requested or 500, 2000))
    horizon_seconds = parse_int(query.get("horizon_seconds", ["30"])[0], 30) or 30
    run_id_filter = str(query.get("run_id", [""])[0] or "").strip()

    local_state = dict(state)
    local_state["SYMBOL"] = symbol
    local_state["PRIMARY_VENUE"] = venue

    predictions_path = file_path_for_spec(local_state, "venue", "{SYMBOL}_tiny_price_ensemble_live_predictions.csv")
    prediction_rows, prediction_path_text, prediction_status = read_csv_tail(
        predictions_path,
        max_rows=max(2000, rows_requested * 8),
    )
    filtered_predictions = []
    for row in prediction_rows:
        row_symbol = str(row.get("symbol", symbol)).upper()
        row_venue = str(row.get("primary_venue", row.get("venue", venue))).lower()
        timestamp = normalize_timestamp_ms(row.get("timestamp"))
        if row_symbol != symbol or row_venue != venue or timestamp is None:
            continue
        if run_id_filter and str(row.get("run_id", "")) != run_id_filter:
            continue
        row = dict(row)
        row["_timestamp_ms"] = timestamp
        filtered_predictions.append(row)
    filtered_predictions.sort(key=lambda item: item["_timestamp_ms"])
    filtered_predictions = filtered_predictions[-rows_requested:]

    snapshot_rows_to_load = max(2000, rows_requested * max(6, int(horizon_seconds / 5) + 4))
    snapshots, snapshot_info = load_recent_snapshots(local_state, snapshot_rows_to_load)
    snapshot_timestamps = [point["timestamp"] for point in snapshots]
    max_future_gap_ms = max(60_000, horizon_seconds * 2000)

    chart_rows = []
    run_ids = []
    for row in filtered_predictions:
        timestamp = row["_timestamp_ms"]
        run_id = str(row.get("run_id", ""))
        if run_id and run_id not in run_ids:
            run_ids.append(run_id)
        current_point, current_index = find_point_at_or_before(snapshots, snapshot_timestamps, timestamp)
        future_point, future_index = find_point_at_or_after(
            snapshots,
            snapshot_timestamps,
            timestamp + horizon_seconds * 1000,
            max_future_gap_ms,
        )
        actual_return_bps = None
        mfe_bps = None
        mae_bps = None
        outcome_ready = bool(current_point and future_point)
        if outcome_ready:
            current_mid = current_point["mid_price"]
            future_mid = future_point["mid_price"]
            actual_return_bps = (future_mid / current_mid - 1.0) * 10_000.0
            future_slice = snapshots[current_index + 1 : future_index + 1]
            if future_slice:
                max_mid = max(point["mid_price"] for point in future_slice)
                min_mid = min(point["mid_price"] for point in future_slice)
                mfe_bps = (max_mid / current_mid - 1.0) * 10_000.0
                mae_bps = (min_mid / current_mid - 1.0) * 10_000.0

        final_signal = str(row.get("final_paper_signal", "no_trade") or "no_trade").lower()
        final_direction = parse_int(row.get("final_paper_signal_direction"), None)
        if final_direction is None:
            final_direction = direction_from_signal(final_signal)
        strategy_return_bps = None
        prediction_correct = None
        if outcome_ready and final_direction:
            strategy_return_bps = actual_return_bps * final_direction
            prediction_correct = strategy_return_bps > 0

        chart_rows.append(
            {
                "timestamp": timestamp,
                "time": row.get("time", ""),
                "run_id": run_id,
                "final_paper_signal": final_signal,
                "final_paper_signal_direction": final_direction,
                "move_direction": parse_int(row.get("move_before_adverse_direction"), 0) or 0,
                "move_confidence": parse_float(row.get("move_before_adverse_confidence"), None),
                "instability_probability": parse_float(row.get("instability_probability"), None),
                "decision_reason": row.get("decision_reason", row.get("no_trade_reason", "")),
                "no_trade_reason": row.get("no_trade_reason", ""),
                "snapshot_freshness": row.get("snapshot_freshness", ""),
                "snapshot_age_seconds": parse_float(row.get("snapshot_age_seconds"), None),
                "required_schema_match": truthy_text(row.get("required_schema_match", "")),
                "optional_direction_schema_match": truthy_text(row.get("optional_direction_schema_match", "true")),
                "optional_regression_schema_match": truthy_text(row.get("optional_regression_schema_match", "true")),
                "active_allowed_sides": row.get("active_allowed_sides", row.get("allowed_sides", "")),
                "active_feature_groups": row.get("active_feature_groups", ""),
                "active_feature_schema_hash": row.get("active_feature_schema_hash", ""),
                "model_pinning_status": row.get("model_pinning_status", "unknown"),
                "models_pinned": truthy_text(row.get("models_pinned", "")),
                "move_model_id": row.get("move_model_id", ""),
                "instability_model_id": row.get("instability_model_id", ""),
                "current_mid_price": current_point["mid_price"] if current_point else None,
                "future_timestamp": future_point["timestamp"] if future_point else None,
                "future_mid_price": future_point["mid_price"] if future_point else None,
                "outcome_ready": outcome_ready,
                "actual_return_bps": actual_return_bps,
                "strategy_return_bps": strategy_return_bps,
                "prediction_correct": prediction_correct,
                "mfe_bps": mfe_bps,
                "mae_bps": mae_bps,
            }
        )

    evaluated_signals = [row for row in chart_rows if row["outcome_ready"] and row["final_paper_signal_direction"]]
    evaluated_rows = [row for row in chart_rows if row["outcome_ready"]]
    long_rows = [row for row in evaluated_signals if row["final_paper_signal_direction"] > 0]
    short_rows = [row for row in evaluated_signals if row["final_paper_signal_direction"] < 0]
    correct_rows = [row for row in evaluated_signals if row["prediction_correct"]]
    stale_rows = [row for row in chart_rows if str(row.get("snapshot_freshness", "")).lower() == "stale"]
    schema_mismatch_rows = [row for row in chart_rows if not row.get("required_schema_match")]
    avg_strategy = (
        sum(row["strategy_return_bps"] for row in evaluated_signals if row["strategy_return_bps"] is not None) / len(evaluated_signals)
        if evaluated_signals
        else None
    )
    summary = {
        "prediction_rows": len(chart_rows),
        "evaluated_rows": len(evaluated_rows),
        "paper_signal_rows": len(evaluated_signals),
        "long_rows": len(long_rows),
        "short_rows": len(short_rows),
        "no_trade_rows": len([row for row in chart_rows if not row["final_paper_signal_direction"]]),
        "sign_accuracy": len(correct_rows) / len(evaluated_signals) if evaluated_signals else None,
        "avg_strategy_return_bps": avg_strategy,
        "stale_rows": len(stale_rows),
        "required_schema_mismatch_rows": len(schema_mismatch_rows),
        "latest_run_id": chart_rows[-1]["run_id"] if chart_rows else "",
        "run_ids": run_ids[-20:],
        "horizon_seconds": horizon_seconds,
    }

    return {
        "state": local_state,
        "summary": summary,
        "rows": chart_rows,
        "files": {
            "predictions": {"path": prediction_path_text, "status": prediction_status, "rows_loaded": len(filtered_predictions)},
            "snapshots": snapshot_info,
            "evaluation": latest_evaluation_row(local_state),
            "run_comparison": run_comparison_payload(local_state),
        },
        "safety": {"paper_only": True, "orders_enabled": False, "promotion_enabled": False},
    }


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Paper Live Control Dashboard</title>
  <style>
    :root {
      --bg:#08111f; --panel:#111b2e; --panel2:#17233b; --text:#edf3ff;
      --muted:#9badc8; --line:#273958; --green:#3ddc97; --red:#ff5f7a;
      --yellow:#ffd166; --blue:#68b7ff; --orange:#ff9f43;
    }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:Segoe UI, system-ui, sans-serif; }
    header { padding:18px 22px; border-bottom:1px solid var(--line); background:#0d1729; position:sticky; top:0; z-index:2; }
    h1 { margin:0; font-size:22px; }
    .sub { color:var(--muted); margin-top:4px; font-size:13px; }
    main { display:grid; grid-template-columns: 380px 1fr; gap:16px; padding:16px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px; box-shadow:0 8px 30px #0004; }
    h2 { margin:0 0 12px; font-size:17px; }
    h3 { margin:16px 0 8px; font-size:14px; color:#c9d8f2; }
    label { display:block; color:var(--muted); font-size:12px; margin:8px 0 3px; }
    input, select { width:100%; background:#0b1425; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:8px; }
    .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    button { cursor:pointer; background:var(--blue); color:#07101c; border:0; border-radius:9px; padding:9px 11px; font-weight:700; margin:4px 4px 4px 0; }
    button.secondary { background:#233553; color:var(--text); border:1px solid var(--line); }
    button.danger { background:var(--red); color:white; }
    button.warn { background:var(--yellow); color:#281c00; }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .warning { display:none; background:#3a2505; color:#ffdca8; border:1px solid #805a19; padding:10px; border-radius:10px; margin:10px 0; }
    .safety { color:var(--green); font-size:13px; margin-top:8px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { border-bottom:1px solid var(--line); padding:7px 6px; text-align:left; vertical-align:top; }
    th { color:#c9d8f2; background:var(--panel2); position:sticky; top:80px; }
    .fresh { color:var(--green); font-weight:700; }
    .stale, .missing { color:var(--red); font-weight:700; }
    .notimestamp { color:var(--yellow); }
    .path { color:var(--muted); font-family:Consolas, monospace; font-size:11px; word-break:break-all; }
    .commandGroup { margin-bottom:10px; }
    .process { border:1px solid var(--line); background:#0c1628; border-radius:12px; margin:10px 0; padding:10px; }
    .processHeader { display:flex; justify-content:space-between; gap:8px; align-items:center; }
    pre { white-space:pre-wrap; overflow:auto; max-height:360px; background:#050a13; border:1px solid #1d2b44; padding:10px; border-radius:10px; color:#d8e6ff; }
    .pill { display:inline-block; padding:2px 7px; border-radius:999px; background:#263a5a; color:#dce9ff; font-size:12px; }
    .metricGrid { display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:8px; margin:10px 0; }
    .metric { background:#0b1425; border:1px solid var(--line); border-radius:10px; padding:9px; }
    .metric .label { color:var(--muted); font-size:11px; margin:0 0 3px; }
    .metric .value { font-size:17px; font-weight:800; }
    .chartWrap { overflow-x:auto; border:1px solid var(--line); border-radius:12px; background:#060d19; padding:8px; }
    .chartAxis { stroke:#334868; stroke-width:1; }
    .chartLine { fill:none; stroke:var(--blue); stroke-width:2; opacity:.9; }
    .chartZero { stroke:#7085a8; stroke-width:1; stroke-dasharray:4 4; }
    .chartCorrect { fill:var(--green); stroke:#052916; stroke-width:1; }
    .chartWrong { fill:var(--red); stroke:#2b0710; stroke-width:1; }
    .chartNoSignal { fill:#6f7d94; opacity:.75; }
    .chartPending { fill:var(--yellow); opacity:.8; }
    .tinyTable { max-height:320px; overflow:auto; border:1px solid var(--line); border-radius:12px; }
    .sampleLow { color:var(--red); font-weight:800; }
    .sampleUseful { color:var(--yellow); font-weight:800; }
    .sampleStrong { color:var(--green); font-weight:800; }
    .sampleWarming { color:var(--orange); font-weight:800; }
    .running { color:var(--yellow); }
    .finished { color:var(--green); }
    .failed, .stopped { color:var(--red); }
    @media (max-width: 1000px) { main { grid-template-columns:1fr; } th { position:static; } }
  </style>
</head>
<body>
  <header>
    <h1>Paper Live Control Dashboard</h1>
    <div class="sub">Local only: http://127.0.0.1:8770/ · no trades · no orders · no private API keys</div>
  </header>
  <main>
    <div>
      <section>
        <h2>Environment</h2>
        <div class="grid2">
          <div><label>SYMBOL</label><input id="SYMBOL"></div>
          <div><label>SYMBOLS</label><input id="SYMBOLS"></div>
          <div><label>PRIMARY_VENUE</label><input id="PRIMARY_VENUE"></div>
          <div><label>VENUES</label><input id="VENUES"></div>
          <div><label>FLOW_1S_MODEL_SELECTION</label><select id="FLOW_1S_MODEL_SELECTION"><option>latest_candidate</option><option>active_only</option></select></div>
          <div><label>MICRO_MODEL_SELECTION</label><select id="MICRO_MODEL_SELECTION"><option>latest_candidate</option><option>active_only</option></select></div>
          <div><label>PROMOTE_BEST</label><select id="PROMOTE_BEST"><option>false</option><option>true</option></select></div>
          <div><label>MAX_MICRO_SNAPSHOTS</label><input id="MAX_MICRO_SNAPSHOTS"></div>
          <div><label>MICRO_RUN_FLOW_1S_ABLATION</label><select id="MICRO_RUN_FLOW_1S_ABLATION"><option>false</option><option>true</option></select></div>
          <div><label>MICRO_ALLOW_GROUP_DISAGREEMENT</label><select id="MICRO_ALLOW_GROUP_DISAGREEMENT"><option>false</option><option>true</option></select></div>
          <div><label>TRAIN_PRICE_TINY_MODEL</label><select id="TRAIN_PRICE_TINY_MODEL"><option>false</option><option>true</option></select></div>
          <div><label>PRICE_TINY_AUTO_REGISTER_CHALLENGERS</label><select id="PRICE_TINY_AUTO_REGISTER_CHALLENGERS"><option>false</option><option>true</option></select></div>
        </div>
        <h3>Tiny-price live ensemble</h3>
        <div class="grid2">
          <div><label>ENABLE_PRICE_TINY_SHADOW_POOL</label><select id="ENABLE_PRICE_TINY_SHADOW_POOL"><option>false</option><option>true</option></select></div>
          <div><label>ENABLE_PRICE_TINY_ENSEMBLE_SHOW</label><select id="ENABLE_PRICE_TINY_ENSEMBLE_SHOW"><option>false</option><option>true</option></select></div>
          <div><label>PRICE_TINY_ENSEMBLE_SHOW_SECONDS</label><input id="PRICE_TINY_ENSEMBLE_SHOW_SECONDS"></div>
          <div><label>PRICE_TINY_ENSEMBLE_RUN_ID</label><input id="PRICE_TINY_ENSEMBLE_RUN_ID" placeholder="blank = script-generated"></div>
          <div><label>MOVE_TARGET_SPEC</label><input id="PRICE_TINY_ENSEMBLE_MOVE_TARGET_SPEC"></div>
          <div><label>INSTABILITY_TARGET_SPEC</label><input id="PRICE_TINY_ENSEMBLE_INSTABILITY_TARGET_SPEC"></div>
          <div><label>FEATURE_GROUPS</label><input id="PRICE_TINY_ENSEMBLE_FEATURE_GROUPS" placeholder="base_tiny_price_v1,calendar_session_features"></div>
          <div><label>FEATURE_SCHEMA_HASH</label><input id="PRICE_TINY_ENSEMBLE_FEATURE_SCHEMA_HASH" placeholder="optional exact schema"></div>
          <div><label>MOVE_CONFIDENCE_THRESHOLD</label><input id="PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD"></div>
          <div><label>INSTABILITY_THRESHOLD</label><input id="PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD"></div>
          <div><label>ALLOWED_SIDES</label><select id="PRICE_TINY_ENSEMBLE_ALLOWED_SIDES"><option>both</option><option>long</option><option>short</option></select></div>
          <div><label>FORECAST_CHART_ROWS</label><input id="FORECAST_CHART_ROWS"></div>
          <div><label>MOVE_MODEL_PATH pin</label><input id="PRICE_TINY_ENSEMBLE_MOVE_MODEL_PATH" placeholder="optional exact model.json"></div>
          <div><label>INSTABILITY_MODEL_PATH pin</label><input id="PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_PATH" placeholder="optional exact model.json"></div>
          <div><label>MOVE_MODEL_ID pin</label><input id="PRICE_TINY_ENSEMBLE_MOVE_MODEL_ID" placeholder="optional exact model_id"></div>
          <div><label>INSTABILITY_MODEL_ID pin</label><input id="PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_ID" placeholder="optional exact model_id"></div>
          <div><label>REQUIRE_DIRECTION_AGREEMENT</label><select id="PRICE_TINY_ENSEMBLE_REQUIRE_DIRECTION_AGREEMENT"><option>false</option><option>true</option></select></div>
          <div><label>ENABLE_REGRESSION</label><select id="PRICE_TINY_ENSEMBLE_ENABLE_REGRESSION"><option>false</option><option>true</option></select></div>
        </div>
        <div style="margin-top:8px">
          <button class="secondary" onclick="applyPreset('baseline')">Preset: baseline both-side</button>
          <button class="secondary" onclick="applyPreset('calendarLong')">Preset: calendar/session long-only</button>
          <button class="secondary" onclick="applyPreset('baseStrictLong')">Preset: base strict long-only</button>
        </div>
        <h3>Replay / historical trade research</h3>
        <div class="grid2">
          <div><label>VENUE</label><input id="VENUE"></div>
          <div><label>SIM_SECONDS</label><input id="SIM_SECONDS"></div>
          <div><label>TRADE_START_TIME</label><input id="TRADE_START_TIME" placeholder="optional ISO/time"></div>
          <div><label>TRADE_END_TIME</label><input id="TRADE_END_TIME" placeholder="optional ISO/time"></div>
          <div><label>BINANCE_START_MONTH</label><input id="BINANCE_START_MONTH" placeholder="YYYY-MM"></div>
          <div><label>BINANCE_END_MONTH</label><input id="BINANCE_END_MONTH" placeholder="YYYY-MM"></div>
        </div>
        <div id="promoteWarning" class="warning">PROMOTE_BEST=true is enabled. Promotion gates still apply, but this dashboard never promotes by default.</div>
        <button onclick="saveState()">Save environment</button>
        <button class="secondary" onclick="loadState()">Reset from server</button>
        <div class="safety">Hard safety: paper-only commands, account/private API variables are stripped from child process environments.</div>
      </section>

      <section style="margin-top:16px">
        <h2>Commands</h2>
        <label><input style="width:auto" type="checkbox" id="allowDuplicate"> explicitly allow duplicate command</label>
        <div id="commands"></div>
      </section>
    </div>

    <div>
      <section>
        <h2>Forecast attempts vs actual 30s outcome</h2>
        <div class="sub" id="forecastMeta">Loading paper forecast chart...</div>
        <div class="metricGrid" id="forecastMetrics"></div>
        <div class="chartWrap">
          <svg id="forecastChart" width="980" height="300" role="img" aria-label="Tiny-price paper forecasts versus actual future return"></svg>
        </div>
        <div class="sub" style="margin-top:8px">
          Blue line = realized future mid-price return in bps. Green/red markers = correct/wrong paper signals. Gray = no-trade, yellow = pending outcome.
        </div>
        <div class="tinyTable" style="margin-top:10px">
          <table>
            <thead><tr><th>Time</th><th>Signal</th><th>Move conf</th><th>Instability</th><th>Actual 30s</th><th>Result</th><th>Reason</th></tr></thead>
            <tbody id="forecastRows"></tbody>
          </table>
        </div>
      </section>

      <section style="margin-top:16px">
        <h2>Live evaluation snapshot</h2>
        <div id="evaluationSnapshot" class="sub">Waiting for evaluation summary...</div>
      </section>

      <section style="margin-top:16px">
        <h2>Run comparison</h2>
        <div id="runComparisonMeta" class="sub">Waiting for run comparison CSV...</div>
        <div class="tinyTable" style="margin-top:10px">
          <table>
            <thead>
              <tr>
                <th>Run</th><th>Profile</th><th>Thresholds</th><th>Sides</th>
                <th>Pinning</th><th>Rows</th><th>Signals</th><th>L/S</th><th>Sign acc</th>
                <th>Gross</th><th>Net</th><th>Net +</th><th>Stale</th><th>Schema</th>
              </tr>
            </thead>
            <tbody id="runComparisonRows"></tbody>
          </table>
        </div>
      </section>

      <section>
        <h2>Status panel</h2>
        <div id="statusMeta" class="sub"></div>
        <div id="flow1sDiag" class="sub" style="margin:10px 0; padding:10px; border:1px solid var(--line); border-radius:10px; background:#0b1425"></div>
        <table>
          <thead><tr><th>File</th><th>Status</th><th>Rows</th><th>Latest</th><th>Age</th><th>Path</th></tr></thead>
          <tbody id="fileRows"></tbody>
        </table>
      </section>

      <section style="margin-top:16px">
        <h2>Command output</h2>
        <div id="processes"></div>
      </section>
    </div>
  </main>
  <script>
    const stateKeys = [
      "SYMBOL","SYMBOLS","PRIMARY_VENUE","VENUES","FLOW_1S_MODEL_SELECTION",
      "MICRO_MODEL_SELECTION","PROMOTE_BEST","MICRO_RUN_FLOW_1S_ABLATION",
      "MICRO_ALLOW_GROUP_DISAGREEMENT","MAX_MICRO_SNAPSHOTS",
      "TRAIN_PRICE_TINY_MODEL","PRICE_TINY_AUTO_REGISTER_CHALLENGERS",
      "ENABLE_PRICE_TINY_SHADOW_POOL","ENABLE_PRICE_TINY_ENSEMBLE_SHOW",
      "PRICE_TINY_ENSEMBLE_SHOW_SECONDS","PRICE_TINY_ENSEMBLE_RUN_ID",
      "PRICE_TINY_ENSEMBLE_MOVE_TARGET_SPEC","PRICE_TINY_ENSEMBLE_INSTABILITY_TARGET_SPEC",
      "PRICE_TINY_ENSEMBLE_FEATURE_GROUPS","PRICE_TINY_ENSEMBLE_FEATURE_SCHEMA_HASH",
      "PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD","PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD",
      "PRICE_TINY_ENSEMBLE_ALLOWED_SIDES","PRICE_TINY_ENSEMBLE_MOVE_MODEL_PATH",
      "PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_PATH","PRICE_TINY_ENSEMBLE_MOVE_MODEL_ID",
      "PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_ID","PRICE_TINY_ENSEMBLE_REQUIRE_DIRECTION_AGREEMENT",
      "PRICE_TINY_ENSEMBLE_ENABLE_REGRESSION","FORECAST_CHART_ROWS",
      "VENUE","TRADE_START_TIME","TRADE_END_TIME","SIM_SECONDS","BINANCE_START_MONTH","BINANCE_END_MONTH"
    ];
    function fieldState() {
      const state = {};
      for (const key of stateKeys) state[key] = document.getElementById(key).value;
      return state;
    }
    function setFields(state) {
      for (const key of stateKeys) {
        const el = document.getElementById(key);
        if (el && state[key] !== undefined) el.value = state[key];
      }
      document.getElementById('promoteWarning').style.display =
        String(document.getElementById('PROMOTE_BEST').value).toLowerCase() === 'true' ? 'block' : 'none';
    }
    function applyPreset(name) {
      const presets = {
        baseline: {
          PRICE_TINY_ENSEMBLE_MOVE_TARGET_SPEC: 'move_before_adverse_30s_net_aware',
          PRICE_TINY_ENSEMBLE_INSTABILITY_TARGET_SPEC: 'instability_30s',
          PRICE_TINY_ENSEMBLE_FEATURE_GROUPS: 'base_tiny_price_v1',
          PRICE_TINY_ENSEMBLE_FEATURE_SCHEMA_HASH: '543c07fec8e33baf',
          PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD: '0.60',
          PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD: '0.80',
          PRICE_TINY_ENSEMBLE_ALLOWED_SIDES: 'both',
          PRICE_TINY_ENSEMBLE_RUN_ID: 'sol_kraken_move060_inst080_days_001',
          PRICE_TINY_ENSEMBLE_MOVE_MODEL_PATH: '',
          PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_PATH: '',
          PRICE_TINY_ENSEMBLE_MOVE_MODEL_ID: '',
          PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_ID: ''
        },
        calendarLong: {
          PRICE_TINY_ENSEMBLE_MOVE_TARGET_SPEC: 'move_before_adverse_30s_net_aware',
          PRICE_TINY_ENSEMBLE_INSTABILITY_TARGET_SPEC: 'instability_30s',
          PRICE_TINY_ENSEMBLE_FEATURE_GROUPS: 'base_tiny_price_v1,calendar_session_features',
          PRICE_TINY_ENSEMBLE_FEATURE_SCHEMA_HASH: '6e16bd2e85a65ee6',
          PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD: '0.65',
          PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD: '0.70',
          PRICE_TINY_ENSEMBLE_ALLOWED_SIDES: 'long',
          PRICE_TINY_ENSEMBLE_RUN_ID: 'sol_kraken_cal_move065_inst070_longonly_challenger_001',
          PRICE_TINY_ENSEMBLE_MOVE_MODEL_PATH: '',
          PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_PATH: '',
          PRICE_TINY_ENSEMBLE_MOVE_MODEL_ID: '',
          PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_ID: ''
        },
        baseStrictLong: {
          PRICE_TINY_ENSEMBLE_MOVE_TARGET_SPEC: 'move_before_adverse_30s_net_aware',
          PRICE_TINY_ENSEMBLE_INSTABILITY_TARGET_SPEC: 'instability_30s',
          PRICE_TINY_ENSEMBLE_FEATURE_GROUPS: 'base_tiny_price_v1',
          PRICE_TINY_ENSEMBLE_FEATURE_SCHEMA_HASH: '543c07fec8e33baf',
          PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD: '0.70',
          PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD: '0.70',
          PRICE_TINY_ENSEMBLE_ALLOWED_SIDES: 'long',
          PRICE_TINY_ENSEMBLE_RUN_ID: 'sol_kraken_base_move070_inst070_longonly_challenger_001',
          PRICE_TINY_ENSEMBLE_MOVE_MODEL_PATH: '',
          PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_PATH: '',
          PRICE_TINY_ENSEMBLE_MOVE_MODEL_ID: '',
          PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_ID: ''
        }
      };
      const preset = presets[name];
      if (!preset) return;
      for (const [key, value] of Object.entries(preset)) {
        const el = document.getElementById(key);
        if (el) el.value = value;
      }
    }
    async function api(path, options={}) {
      const res = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
      return await res.json();
    }
    async function loadState() {
      const data = await api('/api/state');
      setFields(data.state);
      await refresh();
    }
    async function saveState() {
      const data = await api('/api/state', {method:'POST', body:JSON.stringify(fieldState())});
      setFields(data.state);
      await refresh();
    }
    async function runCommand(id) {
      await saveState();
      const allowDuplicate = document.getElementById('allowDuplicate').checked;
      const result = await api('/api/run', {method:'POST', body:JSON.stringify({command_id:id, allow_duplicate:allowDuplicate, env:fieldState()})});
      if (result.error) alert(result.error);
      await refreshProcesses();
    }
    async function stopProcess(id) {
      await api('/api/stop', {method:'POST', body:JSON.stringify({process_id:id})});
      await refreshProcesses();
    }
    function clsStatus(value) {
      const v = String(value || '').toLowerCase().replace(/\s+/g, '');
      if (v.includes('fresh')) return 'fresh';
      if (v.includes('missing')) return 'missing';
      if (v.includes('stale')) return 'stale';
      return 'notimestamp';
    }
    function ageText(value) {
      if (value === null || value === undefined) return 'n/a';
      return Number(value).toFixed(1) + 's';
    }
    async function refreshStatus() {
      const data = await api('/api/status');
      document.getElementById('statusMeta').textContent =
        `SYMBOL=${data.state.SYMBOL} · PRIMARY_VENUE=${data.state.PRIMARY_VENUE} · refreshed ${new Date().toLocaleTimeString()}`;
      const d = data.flow_1s_loop_diagnostics || {};
      document.getElementById('flow1sDiag').innerHTML = `
        <strong>1s live-loop diagnostics</strong><br>
        status=${d.status || 'missing'} · reason=${d.blocking_reason || 'none'}<br>
        latest_snapshot=${d.latest_snapshot_timestamp || 'n/a'} · latest_prediction=${d.latest_prediction_timestamp || 'n/a'} ·
        candidate_newer=${d.candidate_rows_newer_than_latest_prediction ?? 'n/a'} ·
        feature_not_ready=${d.feature_ready_skipped_rows ?? 'n/a'} ·
        non_finite=${d.non_finite_feature_skipped_rows ?? 'n/a'} ·
        invalid_book=${d.invalid_book_skipped_rows ?? 'n/a'} ·
        written=${d.newly_written_predictions ?? 'n/a'}<br>
        <span class="path">${d.path || ''}</span>`;
      const rows = data.files.map(file => `
        <tr>
          <td>${file.label}</td>
          <td class="${clsStatus(file.freshness)}">${file.freshness}</td>
          <td>${file.row_count}</td>
          <td>${file.latest_timestamp || 'n/a'}<br><span class="sub">${file.latest_time || ''}</span></td>
          <td>${ageText(file.age_seconds)}</td>
          <td class="path">${file.path}</td>
        </tr>`).join('');
      document.getElementById('fileRows').innerHTML = rows;
      renderCommands(data.commands);
    }
    function fmtNum(value, digits=2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
      return Number(value).toFixed(digits);
    }
    function pct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
      return (Number(value) * 100).toFixed(1) + '%';
    }
    async function refreshForecast() {
      const state = fieldState();
      const rows = Math.max(25, Math.min(Number(state.FORECAST_CHART_ROWS || 500), 2000));
      const runId = encodeURIComponent(state.PRICE_TINY_ENSEMBLE_RUN_ID || '');
      const url = `/api/forecast-actual?symbol=${encodeURIComponent(state.SYMBOL)}&venue=${encodeURIComponent(state.PRIMARY_VENUE)}&rows=${rows}&run_id=${runId}`;
      const data = await api(url);
      renderForecast(data);
    }
    function renderForecast(data) {
      const summary = data.summary || {};
      const rows = data.rows || [];
      document.getElementById('forecastMeta').textContent =
        `SYMBOL=${data.state?.SYMBOL || ''} · PRIMARY_VENUE=${data.state?.PRIMARY_VENUE || ''} · ` +
        `predictions=${data.files?.predictions?.status || 'n/a'} · snapshots=${data.files?.snapshots?.status || 'n/a'} · ` +
        `latest_run=${summary.latest_run_id || 'n/a'} · pinning=${rows.length ? (rows[rows.length - 1].model_pinning_status || 'unknown') : 'n/a'}`;
      document.getElementById('forecastMetrics').innerHTML = [
        ['Pred rows', summary.prediction_rows ?? 0],
        ['Evaluated', summary.evaluated_rows ?? 0],
        ['Signals', summary.paper_signal_rows ?? 0],
        ['Long / short', `${summary.long_rows ?? 0} / ${summary.short_rows ?? 0}`],
        ['Sign accuracy', pct(summary.sign_accuracy)],
        ['Avg signal return', `${fmtNum(summary.avg_strategy_return_bps)} bps`],
        ['Stale rows', summary.stale_rows ?? 0],
        ['Schema mismatch', summary.required_schema_mismatch_rows ?? 0],
      ].map(([label, value]) => `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`).join('');
      renderForecastChart(rows);
      renderForecastRows(rows);
      const evalRow = data.files?.evaluation?.row || {};
      const evalStatus = data.files?.evaluation?.status || 'missing';
      document.getElementById('evaluationSnapshot').innerHTML =
        `<strong>latest evaluation CSV:</strong> ${evalStatus}<br>` +
        `gross_avg=${escapeHtml(evalRow.gross_avg_strategy_return_bps ?? evalRow.avg_strategy_return_bps ?? 'n/a')} bps · ` +
        `net_avg=${escapeHtml(evalRow.net_avg_strategy_return_bps ?? 'n/a')} bps · ` +
        `signals=${escapeHtml(evalRow.paper_signal_rows ?? evalRow.signal_rows ?? 'n/a')} · ` +
        `path=<span class="path">${escapeHtml(data.files?.evaluation?.path || '')}</span>`;
      renderRunComparison(data.files?.run_comparison || {});
    }
    function sampleClass(value) {
      const v = String(value || '').toLowerCase();
      if (v === 'low') return 'sampleLow';
      if (v === 'useful') return 'sampleUseful';
      if (v === 'strong') return 'sampleStrong';
      return 'sampleWarming';
    }
    function renderRunComparison(runComparison) {
      const rows = runComparison.rows || [];
      document.getElementById('runComparisonMeta').innerHTML =
        `evaluation_by_run=${escapeHtml(runComparison.status || 'missing')} · ` +
        `rows=${rows.length} · ` +
        `path=<span class="path">${escapeHtml(runComparison.path || '')}</span>`;
      const html = rows.slice().reverse().map(row => {
        const profile = [
          row.feature_groups || 'n/a',
          row.schema_hash ? `schema=${row.schema_hash}` : '',
          row.move_model_id ? `move=${String(row.move_model_id).slice(0,18)}` : '',
          row.instability_model_id ? `inst=${String(row.instability_model_id).slice(0,18)}` : ''
        ].filter(Boolean).join('<br>');
        const thresholds = `move=${fmtNum(row.move_threshold, 2)}<br>inst=${fmtNum(row.instability_threshold, 2)}`;
        const pinned = String(row.models_pinned).toLowerCase() === 'true' || String(row.model_pinning_status).toLowerCase() === 'pinned';
        const pinClass = pinned ? 'fresh' : 'notimestamp';
        return `<tr>
          <td><span class="path">${escapeHtml(row.run_id || '')}</span></td>
          <td>${profile}</td>
          <td>${thresholds}</td>
          <td>${escapeHtml(row.allowed_sides || '')}</td>
          <td class="${pinClass}">${escapeHtml(row.model_pinning_status || (pinned ? 'pinned' : 'floating'))}</td>
          <td>${row.rows ?? 0}</td>
          <td class="${sampleClass(row.sample_class)}">${row.signals ?? 0}<br><span class="sub">${escapeHtml(row.sample_label || '')}</span></td>
          <td>${row.long ?? 0}/${row.short ?? 0}</td>
          <td>${pct(row.sign_acc)}</td>
          <td>${fmtNum(row.gross_bps)} bps</td>
          <td>${fmtNum(row.net_bps)} bps</td>
          <td>${pct(row.net_positive_rate)}</td>
          <td>${row.stale_rows ?? 0}</td>
          <td>${row.schema_mismatch_rows ?? 0}</td>
        </tr>`;
      }).join('');
      document.getElementById('runComparisonRows').innerHTML =
        html || '<tr><td colspan="14" class="sub">No run comparison rows found yet. Run tiny-price-ensemble-live-evaluate after collecting predictions.</td></tr>';
    }
    function renderForecastChart(rows) {
      const svg = document.getElementById('forecastChart');
      const width = Number(svg.getAttribute('width'));
      const height = Number(svg.getAttribute('height'));
      const pad = {left:46, right:18, top:18, bottom:34};
      const values = rows.filter(r => r.outcome_ready && r.actual_return_bps !== null).map(r => Number(r.actual_return_bps));
      const maxAbs = Math.max(1, ...values.map(v => Math.abs(v)));
      const yMax = maxAbs * 1.18;
      const usableW = width - pad.left - pad.right;
      const usableH = height - pad.top - pad.bottom;
      const x = i => pad.left + (rows.length <= 1 ? 0 : i * usableW / (rows.length - 1));
      const y = v => pad.top + (yMax - v) * usableH / (2 * yMax);
      const zeroY = y(0);
      const points = [];
      rows.forEach((row, i) => {
        if (row.outcome_ready && row.actual_return_bps !== null) points.push(`${x(i).toFixed(1)},${y(Number(row.actual_return_bps)).toFixed(1)}`);
      });
      let html = '';
      html += `<line class="chartAxis" x1="${pad.left}" y1="${height-pad.bottom}" x2="${width-pad.right}" y2="${height-pad.bottom}"></line>`;
      html += `<line class="chartAxis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height-pad.bottom}"></line>`;
      html += `<line class="chartZero" x1="${pad.left}" y1="${zeroY}" x2="${width-pad.right}" y2="${zeroY}"></line>`;
      html += `<text x="8" y="${pad.top + 10}" fill="#9badc8" font-size="11">+${fmtNum(yMax,1)}bps</text>`;
      html += `<text x="10" y="${zeroY - 4}" fill="#9badc8" font-size="11">0</text>`;
      html += `<text x="8" y="${height-pad.bottom}" fill="#9badc8" font-size="11">-${fmtNum(yMax,1)}bps</text>`;
      if (points.length > 1) html += `<polyline class="chartLine" points="${points.join(' ')}"></polyline>`;
      rows.forEach((row, i) => {
        const hasSignal = Number(row.final_paper_signal_direction || 0) !== 0;
        const cx = x(i);
        const cy = row.outcome_ready && row.actual_return_bps !== null ? y(Number(row.actual_return_bps)) : zeroY;
        let klass = 'chartNoSignal';
        if (!row.outcome_ready) klass = 'chartPending';
        else if (hasSignal) klass = row.prediction_correct ? 'chartCorrect' : 'chartWrong';
        html += `<circle class="${klass}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${hasSignal ? 4 : 2.5}"><title>${escapeHtml(row.time || row.timestamp)} · ${escapeHtml(row.final_paper_signal)} · actual=${fmtNum(row.actual_return_bps)}bps · reason=${escapeHtml(row.decision_reason || '')}</title></circle>`;
      });
      svg.innerHTML = html;
    }
    function renderForecastRows(rows) {
      const recent = rows.slice(-18).reverse();
      document.getElementById('forecastRows').innerHTML = recent.map(row => {
        const result = !row.outcome_ready ? 'pending' : (Number(row.final_paper_signal_direction || 0) === 0 ? 'no signal' : (row.prediction_correct ? 'correct' : 'wrong'));
        const resultClass = result === 'correct' ? 'fresh' : (result === 'wrong' ? 'stale' : 'notimestamp');
        return `<tr>
          <td>${escapeHtml(row.time || row.timestamp)}</td>
          <td>${escapeHtml(row.final_paper_signal || '')}</td>
          <td>${pct(row.move_confidence)}</td>
          <td>${pct(row.instability_probability)}</td>
          <td>${fmtNum(row.actual_return_bps)} bps</td>
          <td class="${resultClass}">${result}</td>
          <td>${escapeHtml(row.no_trade_reason || row.decision_reason || '')}</td>
        </tr>`;
      }).join('') || '<tr><td colspan="7" class="sub">No tiny-price ensemble paper predictions found yet. Run tiny-price-ensemble-show or live-system with ensemble enabled.</td></tr>';
    }
    function renderCommands(commands) {
      const bySection = {};
      for (const cmd of commands) {
        if (!cmd.available) continue;
        (bySection[cmd.section] ||= []).push(cmd);
      }
      let html = '';
      for (const section of Object.keys(bySection)) {
        html += `<div class="commandGroup"><h3>${section}</h3>`;
        for (const cmd of bySection[section]) {
          html += `<button onclick="runCommand('${cmd.id}')">${cmd.label}</button>`;
        }
        html += `</div>`;
      }
      document.getElementById('commands').innerHTML = html || '<div class="sub">No commands available.</div>';
    }
    async function refreshProcesses() {
      const data = await api('/api/processes');
      const html = data.processes.map(proc => `
        <div class="process">
          <div class="processHeader">
            <div><strong>${proc.label}</strong> <span class="pill ${proc.status}">${proc.status}</span><br><span class="sub">${proc.command}</span></div>
            <button class="danger" ${proc.running ? '' : 'disabled'} onclick="stopProcess('${proc.id}')">Stop</button>
          </div>
          <pre>${escapeHtml(proc.output || '')}</pre>
        </div>`).join('');
      document.getElementById('processes').innerHTML = html || '<div class="sub">No commands run yet.</div>';
    }
    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    async function refresh() { await refreshStatus(); await refreshForecast(); await refreshProcesses(); }
    for (const key of stateKeys) {
      window.addEventListener('load', () => {
        const el = document.getElementById(key);
        if (el) el.addEventListener('change', () => {
          document.getElementById('promoteWarning').style.display =
            String(document.getElementById('PROMOTE_BEST').value).toLowerCase() === 'true' ? 'block' : 'none';
        });
      });
    }
    loadState();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stdout.write(f"[control-dashboard] {self.address_string()} {fmt % args}\n")

    def send_json(self, payload, status=200):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in {"/", "/index.html"}:
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/state":
            self.send_json({"state": current_state()})
            return
        if path == "/api/status":
            self.send_json(status_payload())
            return
        if path == "/api/forecast-actual":
            self.send_json(forecast_actual_payload(query))
            return
        if path == "/api/commands":
            self.send_json({"commands": available_commands()})
            return
        if path == "/api/processes":
            with PROCESS_LOCK:
                processes = [process.to_dict() for process in sorted(PROCESSES.values(), key=lambda item: item.started_at, reverse=True)]
            self.send_json({"processes": processes})
            return
        self.send_json({"error": "not found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
        except Exception as error:
            self.send_json({"error": f"invalid JSON: {error}"}, status=400)
            return
        if path == "/api/state":
            self.send_json({"state": update_state(payload)})
            return
        if path == "/api/run":
            env_state = update_state(payload.get("env", current_state()))
            command_id = str(payload.get("command_id", "")).strip()
            allow_duplicate = bool(payload.get("allow_duplicate", False))
            process, error = start_command(command_id, env_state, allow_duplicate=allow_duplicate)
            if error and process is None:
                self.send_json({"error": error}, status=409)
                return
            self.send_json({"process": process.to_dict(), "error": error})
            return
        if path == "/api/stop":
            process_id = str(payload.get("process_id", "")).strip()
            stopped, error = stop_process(process_id)
            self.send_json({"stopped": stopped, "error": error})
            return
        self.send_json({"error": "not found"}, status=404)


def main():
    if HOST not in {"127.0.0.1", "localhost"}:
        print(
            "WARNING: CONTROL_DASHBOARD_HOST is not loopback. "
            "This dashboard is intended for local paper-only control."
        )
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print("Paper-only live prediction control dashboard")
    print(f"URL: http://{HOST}:{PORT}/")
    print(f"Project root: {PROJECT_ROOT}")
    print("No trades. No orders. No private API keys are passed to child commands.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard and child processes...")
    finally:
        with PROCESS_LOCK:
            processes = list(PROCESSES.values())
        for process in processes:
            process.stop()
        server.server_close()


if __name__ == "__main__":
    main()
