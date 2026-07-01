import csv
import json
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
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST = os.getenv("CONTROL_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.getenv("CONTROL_DASHBOARD_PORT", "8770"))
IS_WINDOWS = os.name == "nt"
MAX_LOG_CHARS = int(os.getenv("CONTROL_DASHBOARD_MAX_LOG_CHARS", "80000"))

DEFAULT_STATE = {
    "SYMBOL": "SOLUSDT",
    "SYMBOLS": "SOLUSDT",
    "PRIMARY_VENUE": "kraken",
    "VENUES": "kraken,binanceus",
    "FLOW_1S_MODEL_SELECTION": "latest_candidate",
    "MICRO_MODEL_SELECTION": "latest_candidate",
    "PROMOTE_BEST": "false",
    "MICRO_RUN_FLOW_1S_ABLATION": "false",
    "MICRO_ALLOW_GROUP_DISAGREEMENT": "false",
    "MAX_MICRO_SNAPSHOTS": "2000",
}

STATE_LOCK = threading.Lock()
DASHBOARD_STATE = dict(DEFAULT_STATE)
PROCESS_LOCK = threading.Lock()
PROCESSES = {}


COMMANDS = [
    {"id": "live-system", "script": "live-system", "label": "live-system", "section": "Live orchestration", "long_running": True},
    {"id": "night-run", "script": "night-run", "label": "night-run", "section": "Live orchestration", "long_running": True},
    {"id": "market-stack", "script": "market-stack", "label": "market-stack", "section": "Live monitoring", "long_running": False},
    {"id": "forecast1s-gui", "script": "forecast1s-gui", "label": "forecast1s-gui", "section": "Live monitoring", "long_running": True},
    {"id": "paper-signal-log", "script": "paper-signal-log", "label": "paper-signal-log", "section": "Live monitoring", "long_running": False},
    {"id": "hierarchy-log", "script": "hierarchy-log", "label": "hierarchy-log", "section": "Live monitoring", "long_running": False},
    {"id": "hierarchy-log-loop", "script": "hierarchy-log-loop", "label": "hierarchy-log-loop", "section": "Live monitoring", "long_running": True},
    {"id": "hierarchy-evaluate", "script": "hierarchy-evaluate", "label": "hierarchy-evaluate", "section": "Live monitoring", "long_running": False},
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
    for bool_key in ["PROMOTE_BEST", "MICRO_RUN_FLOW_1S_ABLATION", "MICRO_ALLOW_GROUP_DISAGREEMENT"]:
        clean[bool_key] = "true" if clean[bool_key].lower() in {"true", "1", "yes", "y"} else "false"
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
    env.setdefault("PROMOTE_BEST", "false")
    return env


class ManagedProcess:
    def __init__(self, command_info, env_state):
        self.id = uuid.uuid4().hex[:12]
        self.command_info = command_info
        self.env_state = dict(env_state)
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
    else:
        base = PROJECT_ROOT / "data"
    return base / pattern.format(SYMBOL=symbol)


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
      "MICRO_ALLOW_GROUP_DISAGREEMENT","MAX_MICRO_SNAPSHOTS"
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
    async function refresh() { await refreshStatus(); await refreshProcesses(); }
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
        path = urlparse(self.path).path
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
