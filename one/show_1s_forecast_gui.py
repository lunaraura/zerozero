import datetime as dt
import html
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
VENUE_TAG = PRIMARY_VENUE or "legacy"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
FORECAST_GUI_ROWS = int(os.getenv("FORECAST_GUI_ROWS", "500"))
FORECAST_GUI_REFRESH_SECONDS = float(os.getenv("FORECAST_GUI_REFRESH_SECONDS", "2"))
FORECAST_GUI_HOST = os.getenv("FORECAST_GUI_HOST", "127.0.0.1").strip()
FORECAST_GUI_PORT = int(os.getenv("FORECAST_GUI_PORT", "8768"))
FORECAST_GUI_RUN_ONCE = os.getenv("FORECAST_GUI_RUN_ONCE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
FORECAST_GUI_STALE_SECONDS = float(os.getenv("FORECAST_GUI_STALE_SECONDS", "120"))
FLOW_1S_MODEL_SELECTION = os.getenv("FLOW_1S_MODEL_SELECTION", "latest_candidate").strip()
FLOW_PRESSURE_THRESHOLD = float(os.getenv("FLOW_PRESSURE_THRESHOLD", "0.20"))
FUTURE_MAX_GAP_MS = int(os.getenv("FORECAST_GUI_FUTURE_MAX_GAP_MS", "2500"))
FORECAST_GUI_DEFAULT_TARGET_HORIZON_SECONDS = int(os.getenv("FORECAST_GUI_DEFAULT_TARGET_HORIZON_SECONDS", "1"))

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

VENUE_DIR = OUTPUT_DIR / VENUE_TAG if PRIMARY_VENUE else OUTPUT_DIR
PREDICTION_PATH = VENUE_DIR / f"{SYMBOL}_1s_order_flow_predictions.csv"
SNAPSHOT_PATH = VENUE_DIR / f"{SYMBOL}_10s_flow.csv"
SIGNAL_LOG_PATH = VENUE_DIR / f"{SYMBOL}_paper_signal_log.csv"
STATIC_OUTPUT_PATH = VENUE_DIR / f"{SYMBOL}_1s_forecast_dashboard.html"


def now_ms():
    return int(time.time() * 1000)


def iso_from_ms(timestamp):
    if timestamp is None or not np.isfinite(float(timestamp)):
        return ""
    return dt.datetime.fromtimestamp(float(timestamp) / 1000.0, tz=dt.UTC).isoformat().replace("+00:00", "Z")


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), f"missing: {path}"
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(), f"empty: {path}"
    except Exception as error:
        return pd.DataFrame(), f"read error for {path}: {error}"
    if len(frame) == 0:
        return pd.DataFrame(), f"empty: {path}"
    return frame, None


def numeric(series, default=np.nan):
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series(default, index=getattr(series, "index", None))


def as_float(value, default=np.nan):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def normalize_timestamps(frame):
    frame = frame.copy()
    if "timestamp" not in frame.columns:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    return frame.sort_values("timestamp").reset_index(drop=True)


def latest_file_warning(path, frame, label):
    if len(frame) == 0 or "timestamp" not in frame.columns:
        return None
    latest_timestamp = int(pd.to_numeric(frame["timestamp"], errors="coerce").max())
    age_seconds = max(0.0, (now_ms() - latest_timestamp) / 1000.0)
    if age_seconds > FORECAST_GUI_STALE_SECONDS:
        return f"{label} is stale: latest age {age_seconds:.1f}s > {FORECAST_GUI_STALE_SECONDS:.1f}s ({path})"
    return None


def mid_price_from_row(row):
    for column in ["mid_price", "close"]:
        value = as_float(row.get(column))
        if np.isfinite(value) and value > 0:
            return value
    bid = as_float(row.get("best_bid"))
    ask = as_float(row.get("best_ask"))
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return np.nan


def prediction_class(row):
    value = row.get("decoded_flow_class_1s", "")
    if isinstance(value, str) and value:
        return value
    probs = {
        "sell_dominant": as_float(row.get("prob_sell_dominant_1s"), 0.0),
        "neutral": as_float(row.get("prob_neutral_1s"), 0.0),
        "buy_dominant": as_float(row.get("prob_buy_dominant_1s"), 0.0),
    }
    return max(probs, key=probs.get)


def prediction_target_horizon(row):
    for column in ["model_target_horizon_seconds", "target_horizon_seconds"]:
        value = as_float(row.get(column))
        if np.isfinite(value) and int(value) in {1, 3, 5}:
            return int(value)
    return FORECAST_GUI_DEFAULT_TARGET_HORIZON_SECONDS


def realized_class(realized_pressure, realized_return):
    if np.isfinite(realized_pressure):
        if realized_pressure <= -FLOW_PRESSURE_THRESHOLD:
            return "sell_dominant"
        if realized_pressure >= FLOW_PRESSURE_THRESHOLD:
            return "buy_dominant"
        return "neutral"
    if np.isfinite(realized_return):
        if realized_return < 0:
            return "sell_dominant"
        if realized_return > 0:
            return "buy_dominant"
        return "neutral"
    return ""


def marker_status(predicted, actual, outcome_filled):
    if not outcome_filled:
        return "pending"
    if predicted == "neutral":
        return "neutral"
    if predicted == actual:
        return "correct"
    return "wrong"


def format_pct(value):
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def load_signal_tags():
    frame, _ = read_csv(SIGNAL_LOG_PATH)
    if len(frame) == 0 or "timestamp" not in frame.columns:
        return {}
    frame = normalize_timestamps(frame)
    if "paper_signal_tag" not in frame.columns:
        return {}
    return {
        int(row["timestamp"]): str(row.get("paper_signal_tag", "") or "")
        for _, row in frame.iterrows()
    }


def build_rows():
    warnings = []
    predictions, error = read_csv(PREDICTION_PATH)
    if error:
        warnings.append(error)
    snapshots, error = read_csv(SNAPSHOT_PATH)
    if error:
        warnings.append(error)

    predictions = normalize_timestamps(predictions)
    snapshots = normalize_timestamps(snapshots)

    for path, frame, label in [
        (PREDICTION_PATH, predictions, "1s prediction file"),
        (SNAPSHOT_PATH, snapshots, "primary venue snapshot file"),
    ]:
        warning = latest_file_warning(path, frame, label)
        if warning:
            warnings.append(warning)

    if len(predictions) == 0:
        return [], warnings, {}
    if len(snapshots) == 0:
        return [], warnings, {}

    predictions = predictions.tail(max(1, FORECAST_GUI_ROWS)).reset_index(drop=True)
    snapshot_timestamps = snapshots["timestamp"].to_numpy(dtype=np.int64)
    signal_tags = load_signal_tags()

    rows = []
    for _, prediction in predictions.iterrows():
        timestamp = int(prediction["timestamp"])
        current_index = int(np.searchsorted(snapshot_timestamps, timestamp, side="right") - 1)
        target_horizon_seconds = prediction_target_horizon(prediction)
        future_start_index = int(np.searchsorted(snapshot_timestamps, timestamp, side="right"))
        future_end_index = future_start_index + target_horizon_seconds - 1

        current_row = snapshots.iloc[current_index] if current_index >= 0 else None
        future_row = snapshots.iloc[future_end_index] if future_end_index < len(snapshots) else None
        current_mid = mid_price_from_row(current_row) if current_row is not None else np.nan

        realized_future_timestamp = None
        realized_mid = np.nan
        realized_pressure = np.nan
        realized_return = np.nan
        outcome_filled = False
        if future_row is not None:
            candidate_future_timestamp = int(future_row["timestamp"])
            if candidate_future_timestamp > timestamp and candidate_future_timestamp - timestamp <= max(FUTURE_MAX_GAP_MS, target_horizon_seconds * FUTURE_MAX_GAP_MS):
                realized_future_timestamp = candidate_future_timestamp
                realized_mid = mid_price_from_row(future_row)
                future_window = snapshots.iloc[future_start_index:future_end_index + 1]
                buy_volume = numeric(future_window.get("market_buy_volume_10s", pd.Series(dtype=float))).fillna(0.0).sum()
                sell_volume = numeric(future_window.get("market_sell_volume_10s", pd.Series(dtype=float))).fillna(0.0).sum()
                total = buy_volume + sell_volume
                realized_pressure = (buy_volume - sell_volume) / total if total > 0 else 0.0
                if np.isfinite(current_mid) and np.isfinite(realized_mid) and current_mid > 0:
                    realized_return = realized_mid / current_mid - 1.0
                outcome_filled = np.isfinite(realized_pressure) or np.isfinite(realized_return)

        predicted = prediction_class(prediction)
        actual = realized_class(realized_pressure, realized_return) if outcome_filled else ""
        confidence = max(
            as_float(prediction.get("prob_sell_dominant_1s"), 0.0),
            as_float(prediction.get("prob_neutral_1s"), 0.0),
            as_float(prediction.get("prob_buy_dominant_1s"), 0.0),
        )

        rows.append(
            {
                "timestamp": timestamp,
                "time": str(prediction.get("time", "")) or iso_from_ms(timestamp),
                "realized_future_timestamp": realized_future_timestamp,
                "realized_future_time": iso_from_ms(realized_future_timestamp) if realized_future_timestamp else "",
                "target_horizon_seconds": target_horizon_seconds,
                "outcome_filled": bool(outcome_filled),
                "mid_price": None if not np.isfinite(current_mid) else float(current_mid),
                "realized_mid_price": None if not np.isfinite(realized_mid) else float(realized_mid),
                "prob_sell": as_float(prediction.get("prob_sell_dominant_1s"), 0.0),
                "prob_neutral": as_float(prediction.get("prob_neutral_1s"), 0.0),
                "prob_buy": as_float(prediction.get("prob_buy_dominant_1s"), 0.0),
                "predicted_class": predicted,
                "confidence": confidence,
                "predicted_pressure": as_float(prediction.get("pred_market_pressure_1s")),
                "realized_pressure": None if not np.isfinite(realized_pressure) else float(realized_pressure),
                "realized_return_1s": None if not np.isfinite(realized_return) else float(realized_return),
                "actual_class": actual,
                "marker_status": marker_status(predicted, actual, outcome_filled),
                "paper_signal_tag": signal_tags.get(timestamp, ""),
                "model_id": str(prediction.get("model_id", "") or ""),
            }
        )

    diagnostics = compute_diagnostics(rows)
    return rows, warnings, diagnostics


def precision_recall(rows, class_name):
    evaluated = [row for row in rows if row["outcome_filled"] and row["actual_class"]]
    tp = sum(1 for row in evaluated if row["predicted_class"] == class_name and row["actual_class"] == class_name)
    fp = sum(1 for row in evaluated if row["predicted_class"] == class_name and row["actual_class"] != class_name)
    fn = sum(1 for row in evaluated if row["predicted_class"] != class_name and row["actual_class"] == class_name)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return precision, recall


def compute_diagnostics(rows):
    evaluated = [row for row in rows if row["outcome_filled"] and row["actual_class"]]
    total = len(evaluated)
    directional = [row for row in evaluated if row["predicted_class"] != "neutral"]
    returns = [
        row["realized_return_1s"]
        for row in evaluated
        if row["realized_return_1s"] is not None and np.isfinite(row["realized_return_1s"])
    ]
    confidences = [row["confidence"] for row in rows if np.isfinite(row["confidence"])]
    buy_precision, buy_recall = precision_recall(rows, "buy_dominant")
    sell_precision, sell_recall = precision_recall(rows, "sell_dominant")
    return {
        "visible_rows": len(rows),
        "total_evaluated_rows": total,
        "class_accuracy": sum(1 for row in evaluated if row["predicted_class"] == row["actual_class"]) / max(1, total),
        "directional_accuracy_excluding_neutral_predictions": (
            sum(1 for row in directional if row["predicted_class"] == row["actual_class"]) / max(1, len(directional))
        ),
        "buy_precision": buy_precision,
        "buy_recall": buy_recall,
        "sell_precision": sell_precision,
        "sell_recall": sell_recall,
        "predicted_neutral_rate": sum(1 for row in rows if row["predicted_class"] == "neutral") / max(1, len(rows)),
        "average_predicted_confidence": float(np.mean(confidences)) if confidences else np.nan,
        "average_realized_return": float(np.mean(returns)) if returns else np.nan,
    }


def metric_card(label, value):
    return f"<div class='metric'><div class='metric-label'>{html.escape(label)}</div><div class='metric-value'>{html.escape(value)}</div></div>"


def render_html():
    rows, warnings, diagnostics = build_rows()
    generated_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    rows_json = json.dumps(rows, allow_nan=False)
    diagnostics_json = json.dumps(diagnostics, allow_nan=False)
    warning_html = "".join(f"<li>{html.escape(warning)}</li>" for warning in warnings)
    if not warning_html:
        warning_html = "<li>No file warnings.</li>"

    metrics = [
        ("Evaluated", str(diagnostics.get("total_evaluated_rows", 0))),
        ("Class accuracy", format_pct(diagnostics.get("class_accuracy", np.nan))),
        ("Directional accuracy", format_pct(diagnostics.get("directional_accuracy_excluding_neutral_predictions", np.nan))),
        ("Buy precision / recall", f"{format_pct(diagnostics.get('buy_precision', np.nan))} / {format_pct(diagnostics.get('buy_recall', np.nan))}"),
        ("Sell precision / recall", f"{format_pct(diagnostics.get('sell_precision', np.nan))} / {format_pct(diagnostics.get('sell_recall', np.nan))}"),
        ("Predicted neutral rate", format_pct(diagnostics.get("predicted_neutral_rate", np.nan))),
        ("Avg confidence", format_pct(diagnostics.get("average_predicted_confidence", np.nan))),
        ("Avg realized return", format_pct(diagnostics.get("average_realized_return", np.nan))),
    ]
    metrics_html = "".join(metric_card(label, value) for label, value in metrics)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="refresh" content="{FORECAST_GUI_REFRESH_SECONDS}">
  <title>{html.escape(SYMBOL)} 1s Order-Flow Forecast</title>
  <style>
    :root {{
      --bg: #0b1020; --panel: #121a2d; --panel2: #172036; --text: #e8eefc;
      --muted: #95a3bd; --grid: #26334f; --green: #35d07f; --red: #ff5c7a;
      --yellow: #ffd166; --gray: #8b95aa; --blue: #5eb1ff; --orange: #ff9f43;
    }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Segoe UI, system-ui, sans-serif; }}
    header {{ padding: 18px 22px 12px; border-bottom: 1px solid #202b45; background: #0e1528; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    main {{ padding: 18px 22px 28px; }}
    .warnings {{ background: #291a20; border: 1px solid #593140; color: #ffd9df; border-radius: 10px; padding: 10px 14px; margin-bottom: 14px; }}
    .warnings ul {{ margin: 6px 0 0 18px; padding: 0; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 14px 0; }}
    .metric {{ background: var(--panel); border: 1px solid #23304b; border-radius: 10px; padding: 10px 12px; }}
    .metric-label {{ color: var(--muted); font-size: 12px; }}
    .metric-value {{ font-size: 20px; margin-top: 3px; }}
    .chart-card {{ background: var(--panel); border: 1px solid #23304b; border-radius: 12px; padding: 12px; margin-top: 14px; }}
    canvas {{ width: 100%; height: 650px; display: block; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .dot {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; margin-right: 5px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; background: var(--panel); border-radius: 12px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid #22304c; padding: 7px 8px; font-size: 12px; text-align: right; }}
    th {{ color: var(--muted); background: var(--panel2); position: sticky; top: 0; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .correct {{ color: var(--green); }} .wrong {{ color: var(--red); }} .neutral {{ color: var(--gray); }} .pending {{ color: var(--yellow); }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(SYMBOL)} 1s order-flow forecast dashboard</h1>
    <div class="sub">
      venue={html.escape(VENUE_TAG)} · rows={FORECAST_GUI_ROWS} · refresh={FORECAST_GUI_REFRESH_SECONDS}s ·
      model_selection={html.escape(FLOW_1S_MODEL_SELECTION)} · generated={html.escape(generated_at)}
    </div>
    <div class="sub">Paper-only visualization. No trades, no orders, no private/account API access.</div>
  </header>
  <main>
    <section class="warnings"><strong>Status / warnings</strong><ul>{warning_html}</ul></section>
    <section class="metrics">{metrics_html}</section>
    <section class="chart-card">
      <div class="legend">
        <span><span class="dot" style="background: var(--blue)"></span>mid price</span>
        <span><span class="dot" style="background: var(--red)"></span>sell probability</span>
        <span><span class="dot" style="background: var(--gray)"></span>neutral probability</span>
        <span><span class="dot" style="background: var(--green)"></span>buy probability</span>
        <span><span class="dot" style="background: var(--orange)"></span>predicted / realized pressure</span>
        <span><span class="dot" style="background: var(--green)"></span>correct</span>
        <span><span class="dot" style="background: var(--red)"></span>wrong</span>
        <span><span class="dot" style="background: var(--yellow)"></span>pending</span>
      </div>
      <canvas id="forecastChart" width="1400" height="650"></canvas>
    </section>
    <section id="tableSection"></section>
  </main>
  <script>
    const rows = {rows_json};
    const diagnostics = {diagnostics_json};
    const css = getComputedStyle(document.documentElement);
    const colors = {{
      grid: css.getPropertyValue('--grid').trim(),
      text: css.getPropertyValue('--text').trim(),
      muted: css.getPropertyValue('--muted').trim(),
      green: css.getPropertyValue('--green').trim(),
      red: css.getPropertyValue('--red').trim(),
      yellow: css.getPropertyValue('--yellow').trim(),
      gray: css.getPropertyValue('--gray').trim(),
      blue: css.getPropertyValue('--blue').trim(),
      orange: css.getPropertyValue('--orange').trim(),
    }};

    function finite(v) {{ return typeof v === 'number' && Number.isFinite(v); }}
    function pct(v) {{ return finite(v) ? (v * 100).toFixed(3) + '%' : 'n/a'; }}
    function num(v, d=4) {{ return finite(v) ? v.toFixed(d) : 'n/a'; }}
    function shortTime(iso) {{ return iso ? iso.slice(11, 19) : ''; }}
    function markerColor(status) {{
      if (status === 'correct') return colors.green;
      if (status === 'wrong') return colors.red;
      if (status === 'pending') return colors.yellow;
      return colors.gray;
    }}
    function scale(value, min, max, top, height) {{
      if (!finite(value)) return null;
      if (Math.abs(max - min) < 1e-12) return top + height / 2;
      return top + height - ((value - min) / (max - min)) * height;
    }}
    function drawLine(ctx, data, getter, min, max, top, height, color, width=2) {{
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = width;
      let started = false;
      data.forEach((row, i) => {{
        const v = getter(row); const y = scale(v, min, max, top, height);
        if (y === null) {{ started = false; return; }}
        const x = xFor(i);
        if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
      }});
      ctx.stroke();
    }}
    function drawPanel(ctx, label, top, height, min, max) {{
      ctx.strokeStyle = colors.grid; ctx.lineWidth = 1; ctx.fillStyle = colors.muted; ctx.font = '12px Segoe UI';
      ctx.fillText(label, 10, top + 15);
      for (let j = 0; j <= 4; j++) {{
        const y = top + (height * j / 4);
        ctx.beginPath(); ctx.moveTo(55, y); ctx.lineTo(ctx.canvas.width - 15, y); ctx.stroke();
        const value = max - (max - min) * j / 4;
        ctx.fillText(value.toFixed(label.includes('prob') || label.includes('pressure') ? 2 : 4), 8, y + 4);
      }}
    }}
    function xFor(i) {{
      const left = 60, right = 20;
      return left + (rows.length <= 1 ? 0 : i * ((canvas.width - left - right) / (rows.length - 1)));
    }}

    const canvas = document.getElementById('forecastChart');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (rows.length === 0) {{
      ctx.fillStyle = colors.muted; ctx.font = '18px Segoe UI';
      ctx.fillText('No rows available. Check status warnings above.', 30, 60);
    }} else {{
      const priceValues = rows.map(r => r.mid_price).filter(finite);
      const pMin = Math.min(...priceValues), pMax = Math.max(...priceValues);
      const pricePad = Math.max(0.0001, (pMax - pMin) * 0.15);
      const panels = [
        {{ label: 'mid price + correctness markers', top: 18, height: 185, min: pMin - pricePad, max: pMax + pricePad }},
        {{ label: 'probabilities', top: 230, height: 170, min: 0, max: 1 }},
        {{ label: 'predicted pressure vs realized target-horizon pressure', top: 430, height: 170, min: -1, max: 1 }},
      ];
      panels.forEach(p => drawPanel(ctx, p.label, p.top, p.height, p.min, p.max));
      drawLine(ctx, rows, r => r.mid_price, panels[0].min, panels[0].max, panels[0].top, panels[0].height, colors.blue, 2);
      rows.forEach((row, i) => {{
        const y = scale(row.mid_price, panels[0].min, panels[0].max, panels[0].top, panels[0].height);
        if (y === null) return;
        ctx.fillStyle = markerColor(row.marker_status);
        ctx.beginPath(); ctx.arc(xFor(i), y, row.predicted_class === 'neutral' ? 3 : 4.5, 0, Math.PI * 2); ctx.fill();
      }});
      drawLine(ctx, rows, r => r.prob_sell, 0, 1, panels[1].top, panels[1].height, colors.red, 1.6);
      drawLine(ctx, rows, r => r.prob_neutral, 0, 1, panels[1].top, panels[1].height, colors.gray, 1.4);
      drawLine(ctx, rows, r => r.prob_buy, 0, 1, panels[1].top, panels[1].height, colors.green, 1.6);
      rows.forEach((row, i) => {{
        ctx.fillStyle = row.predicted_class === 'buy_dominant' ? colors.green : (row.predicted_class === 'sell_dominant' ? colors.red : colors.gray);
        ctx.fillRect(xFor(i)-2, panels[1].top + panels[1].height + 5, 4, 8);
      }});
      drawLine(ctx, rows, r => r.predicted_pressure, -1, 1, panels[2].top, panels[2].height, colors.blue, 1.5);
      drawLine(ctx, rows, r => r.realized_pressure, -1, 1, panels[2].top, panels[2].height, colors.orange, 1.5);
      const labelStep = Math.max(1, Math.floor(rows.length / 8));
      ctx.fillStyle = colors.muted; ctx.font = '11px Segoe UI';
      rows.forEach((row, i) => {{
        if (i % labelStep !== 0 && i !== rows.length - 1) return;
        ctx.fillText(shortTime(row.time), xFor(i) - 20, canvas.height - 10);
      }});
    }}

    const recent = rows.slice(-40).reverse();
    const table = `<table><thead><tr>
      <th>prediction time</th><th>future time</th><th>filled</th><th>pred</th><th>actual</th>
      <th>horizon</th><th>confidence</th><th>return</th><th>pred pressure</th><th>realized pressure</th><th>marker</th><th>tag</th>
    </tr></thead><tbody>` + recent.map(row => `
      <tr>
        <td>${{row.time || ''}}</td>
        <td>${{row.realized_future_time || ''}}</td>
        <td>${{row.outcome_filled ? 'yes' : 'pending'}}</td>
        <td>${{row.predicted_class}}</td>
        <td>${{row.actual_class || ''}}</td>
        <td>${{row.target_horizon_seconds || 1}}s</td>
        <td>${{pct(row.confidence)}}</td>
        <td>${{pct(row.realized_return_1s)}}</td>
        <td>${{pct(row.predicted_pressure)}}</td>
        <td>${{pct(row.realized_pressure)}}</td>
        <td class="${{row.marker_status}}">${{row.marker_status}}</td>
        <td>${{row.paper_signal_tag || ''}}</td>
      </tr>`).join('') + '</tbody></table>';
    document.getElementById('tableSection').innerHTML = table;
  </script>
</body>
</html>"""


class ForecastHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in {"/", "/index.html"}:
            self.send_response(404)
            self.end_headers()
            return
        payload = render_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def write_static_once():
    STATIC_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATIC_OUTPUT_PATH.write_text(render_html(), encoding="utf-8")
    print(f"1s forecast dashboard written to: {STATIC_OUTPUT_PATH}")
    print("Paper-only visualization. No trades, no orders, no private API.")


def main():
    if FORECAST_GUI_RUN_ONCE:
        write_static_once()
        return
    server = ThreadingHTTPServer((FORECAST_GUI_HOST, FORECAST_GUI_PORT), ForecastHandler)
    url = f"http://{FORECAST_GUI_HOST}:{FORECAST_GUI_PORT}/"
    print("1s forecast GUI running.")
    print(f"Open: {url}")
    print(f"SYMBOL={SYMBOL} PRIMARY_VENUE={VENUE_TAG} rows={FORECAST_GUI_ROWS}")
    print(f"Prediction input: {PREDICTION_PATH}")
    print(f"Snapshot input: {SNAPSHOT_PATH}")
    print("Paper-only visualization. No trades, no orders, no private API.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n1s forecast GUI stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
