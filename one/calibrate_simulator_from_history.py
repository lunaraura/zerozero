import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
OUTPUT_PATH = Path(
    os.getenv(
        "SIM_CALIBRATION_OUTPUT_PATH",
        PROJECT_ROOT / "data" / "simulated" / "calibration" / f"{SYMBOL}_sim_calibration.json",
    )
)
SOURCE_PATH_ENV = os.getenv("SIM_CALIBRATION_SOURCE_PATH", "").strip()

if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH


BINANCE_KLINE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quoteVolume",
    "numberOfTrades",
    "takerBuyBaseVolume",
    "takerBuyQuoteVolume",
    "ignore",
]


def safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except EmptyDataError:
        return pd.DataFrame()


def discover_input_files():
    if SOURCE_PATH_ENV:
        path = Path(SOURCE_PATH_ENV)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return [path] if path.exists() else []

    candidates = []
    preferred = [
        PROJECT_ROOT / "data" / f"{SYMBOL}_5m_imported.csv",
        PROJECT_ROOT / "data" / f"{SYMBOL}_5m.csv",
        PROJECT_ROOT / "data" / f"{SYMBOL}_1m_imported.csv",
    ]
    candidates.extend(path for path in preferred if path.exists())

    binance_dir = PROJECT_ROOT / "data" / "binance_public_csv"
    if binance_dir.exists():
        candidates.extend(sorted(binance_dir.glob(f"{SYMBOL}-5m-*.csv")))

    realtime_root = PROJECT_ROOT / "data" / "realtime"
    if realtime_root.exists():
        candidates.extend(sorted(realtime_root.glob(f"*/{SYMBOL}_1m_flow.csv")))
        candidates.extend(sorted(realtime_root.glob(f"*/{SYMBOL}_10s_flow.csv")))
    return candidates


def normalize_candle_frame(frame, source_path):
    if frame is None or len(frame) == 0:
        return pd.DataFrame()

    original_columns = [str(column) for column in frame.columns]
    lower_columns = {str(column).strip().lower(): column for column in frame.columns}
    has_named_ohlcv = {"open", "high", "low", "close"}.issubset(lower_columns)

    if not has_named_ohlcv and len(frame.columns) >= 6:
        first_column = str(frame.columns[0])
        if first_column.replace(".", "", 1).isdigit():
            frame = read_csv(source_path, header=None, names=BINANCE_KLINE_COLUMNS)
            lower_columns = {str(column).strip().lower(): column for column in frame.columns}

    aliases = {
        "timestamp": ["timestamp", "open_time", "time_ms", "date"],
        "time": ["time", "datetime"],
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close", "mid_price"],
        "volume": ["volume", "total_trade_volume_10s"],
        "quoteVolume": ["quotevolume", "quote_volume"],
        "numberOfTrades": ["numberoftrades", "trade_count", "trade_count_10s"],
        "spread_percent": ["spread_percent"],
        "bid_depth_10bps": ["bid_depth_10bps"],
        "ask_depth_10bps": ["ask_depth_10bps"],
        "bid_depth_25bps": ["bid_depth_25bps"],
        "ask_depth_25bps": ["ask_depth_25bps"],
    }

    output = pd.DataFrame()
    for canonical, names in aliases.items():
        source_column = next((lower_columns[name] for name in names if name in lower_columns), None)
        if source_column is not None:
            output[canonical] = frame[source_column]

    if "timestamp" not in output.columns and "time" in output.columns:
        parsed = pd.to_datetime(output["time"], errors="coerce", utc=True)
        output["timestamp"] = (parsed.astype("int64") // 1_000_000).where(parsed.notna(), np.nan)
    if "timestamp" not in output.columns:
        return pd.DataFrame()

    if {"open", "high", "low", "close"}.issubset(output.columns):
        pass
    elif "close" in output.columns:
        output["open"] = output["close"]
        output["high"] = output["close"]
        output["low"] = output["close"]
    else:
        return pd.DataFrame()

    numeric_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quoteVolume",
        "numberOfTrades",
        "spread_percent",
        "bid_depth_10bps",
        "ask_depth_10bps",
        "bid_depth_25bps",
        "ask_depth_25bps",
    ]
    for column in numeric_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")

    output = output.dropna(subset=["timestamp", "open", "high", "low", "close"])
    output = output[(output["open"] > 0) & (output["high"] > 0) & (output["low"] > 0) & (output["close"] > 0)]
    output["timestamp"] = output["timestamp"].astype("int64")
    output["source_path"] = str(source_path)
    output["source_columns"] = ",".join(original_columns[:20])
    return output


def load_history():
    paths = discover_input_files()
    if not paths:
        return pd.DataFrame(), []

    frames = []
    used = []
    # Prefer one complete imported file. If it exists, avoid duplicating the same
    # Binance monthly rows. Otherwise combine compatible monthly/source files.
    imported = next((path for path in paths if path.name == f"{SYMBOL}_5m_imported.csv"), None)
    selected_paths = [imported] if imported else paths
    for path in selected_paths:
        frame = normalize_candle_frame(read_csv(path), path)
        if len(frame):
            frames.append(frame)
            used.append(path)
    if not frames:
        return pd.DataFrame(), []
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return combined, used


def quantiles(values, probs=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)):
    values = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return {str(prob): 0.0 for prob in probs}
    return {str(prob): float(values.quantile(prob)) for prob in probs}


def range_from_quantiles(values, low=0.10, high=0.90, default=(0.0, 0.0), clip=None):
    values = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return [float(default[0]), float(default[1])]
    a = float(values.quantile(low))
    b = float(values.quantile(high))
    if clip:
        a = float(np.clip(a, clip[0], clip[1]))
        b = float(np.clip(b, clip[0], clip[1]))
    return [a, b]


def run_lengths(labels):
    lengths = []
    previous = None
    count = 0
    for label in labels:
        if label == previous:
            count += 1
        else:
            if previous is not None:
                lengths.append((previous, count))
            previous = label
            count = 1
    if previous is not None:
        lengths.append((previous, count))
    return lengths


def max_drawdown_and_runup(close):
    values = np.asarray(close, dtype=np.float64)
    running_max = np.maximum.accumulate(values)
    running_min = np.minimum.accumulate(values)
    drawdown = values / np.maximum(running_max, 1e-12) - 1.0
    runup = values / np.maximum(running_min, 1e-12) - 1.0
    return float(np.min(drawdown)), float(np.max(runup)), drawdown, runup


def classify_regimes(close, returns, step_seconds):
    close_series = pd.Series(close)
    returns_series = pd.Series(returns)
    window = max(6, int(round(3600 / max(step_seconds, 1))))
    rolling_return = np.log(close_series / close_series.shift(window)).fillna(0.0)
    rolling_volatility = returns_series.rolling(window, min_periods=max(3, window // 4)).std().fillna(0.0)
    trend_threshold = np.maximum(
        rolling_volatility * math.sqrt(window) * 0.75,
        max(0.0015, float(np.nanmedian(np.abs(rolling_return))) * 0.8),
    )
    high_vol_threshold = float(rolling_volatility.quantile(0.75)) if len(rolling_volatility) else 0.0
    labels = []
    for ret, vol, threshold in zip(rolling_return, rolling_volatility, trend_threshold):
        if ret > threshold:
            labels.append("uptrend")
        elif ret < -threshold:
            labels.append("downtrend")
        elif vol > high_vol_threshold and high_vol_threshold > 0:
            labels.append("high_vol_chop")
        else:
            labels.append("chop")
    return labels, rolling_return, rolling_volatility


def build_low_frequency_wave(close):
    log_close = pd.Series(np.log(np.asarray(close, dtype=np.float64)))
    smooth_window = max(24, min(576, len(log_close) // 20))
    smooth = log_close.rolling(smooth_window, min_periods=max(3, smooth_window // 4)).mean()
    smooth = smooth.interpolate().bfill().ffill()
    detrended = smooth - smooth.rolling(max(smooth_window * 4, 48), min_periods=3).mean().interpolate().bfill().ffill()
    values = detrended.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    std = float(np.std(values))
    normalized = values / std if std > 1e-12 else values
    sample_count = min(512, max(64, len(normalized)))
    positions = np.linspace(0, len(normalized) - 1, sample_count)
    sampled = np.interp(positions, np.arange(len(normalized)), normalized)
    sampled = np.clip(sampled, -3.0, 3.0)
    drift = np.diff(sampled)
    return {
        "normalized_points": [float(value) for value in sampled],
        "point_count": int(sample_count),
        "source_smooth_window_rows": int(smooth_window),
        "wave_std_log_price": std,
        "wave_drift_scale_per_second": float(np.clip(np.std(drift) * 0.00008, 0.000005, 0.00008)),
    }


def calibration_payload(frame, used_paths):
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    log_close = np.log(close)
    returns = np.diff(log_close, prepend=log_close[0])
    step_seconds = int(np.nanmedian(np.diff(timestamps)) / 1000) if len(timestamps) > 1 else 300
    days = max((timestamps[-1] - timestamps[0]) / 86_400_000, 1e-9) if len(timestamps) > 1 else 1.0

    labels, rolling_return, rolling_volatility = classify_regimes(close, returns, step_seconds)
    label_counts = Counter(labels)
    lengths = run_lengths(labels)
    durations_seconds = [count * step_seconds for _, count in lengths]
    duration_by_label = {
        label: [count * step_seconds for item_label, count in lengths if item_label == label]
        for label in sorted(label_counts)
    }

    drawdown_min, runup_max, drawdown_series, runup_series = max_drawdown_and_runup(close)
    abs_returns = np.abs(returns)
    shock_threshold = max(float(pd.Series(abs_returns).quantile(0.99)), float(np.std(returns) * 3.0))
    shock_mask = abs_returns >= shock_threshold
    positive_shock_mask = returns >= shock_threshold
    negative_shock_mask = returns <= -shock_threshold

    breakout_window = max(12, int(round(4 * 3600 / max(step_seconds, 1))))
    future_window = max(3, int(round(30 * 60 / max(step_seconds, 1))))
    prev_high = pd.Series(high).rolling(breakout_window, min_periods=max(3, breakout_window // 4)).max().shift(1)
    prev_low = pd.Series(low).rolling(breakout_window, min_periods=max(3, breakout_window // 4)).min().shift(1)
    close_series = pd.Series(close)
    up_breakout = close_series > prev_high
    down_breakout = close_series < prev_low
    future_return = np.log(close_series.shift(-future_window) / close_series).replace([np.inf, -np.inf], np.nan)
    up_fakeout = up_breakout & (future_return < 0)
    down_fakeout = down_breakout & (future_return > 0)

    return_autocorr = float(pd.Series(returns).autocorr(lag=1) or 0.0)
    volatility_clustering = float(pd.Series(abs_returns).autocorr(lag=1) or 0.0)
    vol_per_second = pd.Series(rolling_volatility / math.sqrt(max(step_seconds, 1))).replace([np.inf, -np.inf], np.nan).dropna()
    per_second_returns = pd.Series(returns / max(step_seconds, 1)).replace([np.inf, -np.inf], np.nan).dropna()

    volume_summary = {}
    if "volume" in frame.columns:
        volume_summary["volume_quantiles"] = quantiles(frame["volume"])
        volume_summary["volume_regime_ranges"] = {
            "low": range_from_quantiles(frame["volume"], 0.05, 0.33),
            "normal": range_from_quantiles(frame["volume"], 0.33, 0.67),
            "high": range_from_quantiles(frame["volume"], 0.67, 0.95),
        }
    if "numberOfTrades" in frame.columns:
        volume_summary["trade_count_quantiles"] = quantiles(frame["numberOfTrades"])

    microstructure_summary = {}
    for column in ["spread_percent", "bid_depth_10bps", "ask_depth_10bps", "bid_depth_25bps", "ask_depth_25bps"]:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                microstructure_summary[column] = {
                    "mean": float(values.mean()),
                    "quantiles": quantiles(values),
                }

    regime_probabilities = {label: float(count / len(labels)) for label, count in label_counts.items()}
    for label in ["uptrend", "downtrend", "chop", "high_vol_chop"]:
        regime_probabilities.setdefault(label, 0.0)

    def duration_distribution(values):
        return {
            "count": int(len(values)),
            "quantiles_seconds": quantiles(values, probs=(0.1, 0.5, 0.9)),
            "mean_seconds": float(np.mean(values)) if len(values) else 0.0,
        }

    shock_abs = pd.Series(abs_returns[shock_mask]).replace([np.inf, -np.inf], np.nan).dropna()
    event_magnitude_range = [
        float(np.clip((shock_abs.quantile(0.25) if len(shock_abs) else shock_threshold) * 65.0, 0.12, 0.95)),
        float(np.clip((shock_abs.quantile(0.90) if len(shock_abs) else shock_threshold) * 90.0, 0.18, 1.20)),
    ]
    event_duration_range = [
        int(max(180, np.nanquantile(durations_seconds, 0.10) if durations_seconds else 300)),
        int(min(2400, max(300, np.nanquantile(durations_seconds, 0.75) if durations_seconds else 900))),
    ]

    event_frequency_distributions = {
        "positive_news": {
            "frequency_per_day": float(positive_shock_mask.sum() / days),
            "direction": 1,
            "magnitude_range": event_magnitude_range,
            "duration_seconds_range": event_duration_range,
        },
        "negative_news": {
            "frequency_per_day": float(negative_shock_mask.sum() / days),
            "direction": -1,
            "magnitude_range": event_magnitude_range,
            "duration_seconds_range": event_duration_range,
        },
        "fake_breakout": {
            "frequency_per_day": float((up_fakeout.sum() + down_fakeout.sum()) / days),
            "direction": 0,
            "magnitude_range": [max(0.10, event_magnitude_range[0] * 0.7), max(0.18, event_magnitude_range[1] * 0.8)],
            "duration_seconds_range": [180, max(360, event_duration_range[0])],
        },
        "short_squeeze": {
            "frequency_per_day": float(up_breakout.sum() / days * 0.20),
            "direction": 1,
            "magnitude_range": event_magnitude_range,
            "duration_seconds_range": event_duration_range,
        },
        "panic_selloff": {
            "frequency_per_day": float(down_breakout.sum() / days * 0.20),
            "direction": -1,
            "magnitude_range": event_magnitude_range,
            "duration_seconds_range": event_duration_range,
        },
        "liquidity_vacuum": {
            "frequency_per_day": float(shock_mask.sum() / days * 0.15),
            "direction": 0,
            "magnitude_range": [max(0.12, event_magnitude_range[0] * 0.6), max(0.25, event_magnitude_range[1] * 0.7)],
            "duration_seconds_range": [240, max(600, event_duration_range[1])],
        },
    }

    vol_low, vol_high = range_from_quantiles(vol_per_second, 0.10, 0.90, default=(0.00005, 0.00035), clip=(0.00002, 0.0012))
    high_vol_low, high_vol_high = range_from_quantiles(vol_per_second, 0.70, 0.98, default=(vol_low, vol_high), clip=(0.00005, 0.0012))
    trend_low, trend_high = range_from_quantiles(per_second_returns, 0.10, 0.90, default=(-0.00005, 0.00005), clip=(-0.00012, 0.00012))

    fakeout_rate = float((up_fakeout.sum() + down_fakeout.sum()) / max(up_breakout.sum() + down_breakout.sum(), 1))
    breakout_rate = float((up_breakout.sum() + down_breakout.sum()) / days)
    shock_rate = float(shock_mask.sum() / days)
    normalized_volume_std = 0.0
    if "volume" in frame.columns:
        volume_values = pd.to_numeric(frame["volume"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        normalized_volume_std = float(volume_values.std() / max(volume_values.mean(), 1e-12)) if len(volume_values) else 0.0

    demand_intensity_ranges = {
        "retail": [0.0, float(np.clip(0.05 + normalized_volume_std * 0.10, 0.04, 0.30))],
        "institutional": [0.0, float(np.clip(0.05 + abs(trend_high - trend_low) * 1500, 0.04, 0.35))],
        "momentum": [0.0, float(np.clip(0.04 + breakout_rate * 0.015 + max(return_autocorr, 0) * 0.20, 0.04, 0.35))],
        "mean_reversion": [0.0, float(np.clip(0.05 + fakeout_rate * 0.25 + max(-return_autocorr, 0) * 0.20, 0.04, 0.40))],
        "liquidity": [-float(np.clip(shock_rate * 0.015, 0.03, 0.35)), float(np.clip(0.05 + (1.0 - regime_probabilities["high_vol_chop"]) * 0.12, 0.05, 0.35))],
        "panic": [0.0, float(np.clip(0.04 + abs(drawdown_min) * 1.5 + shock_rate * 0.010, 0.04, 0.45))],
        "news": [0.0, float(np.clip(0.04 + shock_rate * 0.012, 0.04, 0.35))],
        "macro_risk": [-float(np.clip(abs(drawdown_min) * 0.8, 0.03, 0.30)), float(np.clip(runup_max * 0.6, 0.03, 0.30))],
    }

    payload = {
        "schema_version": 1,
        "symbol": SYMBOL,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_files": [str(path) for path in used_paths],
        "row_count": int(len(frame)),
        "first_timestamp": int(timestamps[0]) if len(timestamps) else None,
        "last_timestamp": int(timestamps[-1]) if len(timestamps) else None,
        "timeframe_seconds": int(step_seconds),
        "historical_distribution": {
            "log_return_quantiles": quantiles(returns),
            "abs_log_return_quantiles": quantiles(abs_returns),
            "rolling_volatility_quantiles": quantiles(rolling_volatility),
            "drawdown_min": drawdown_min,
            "runup_max": runup_max,
            "large_shock_threshold": float(shock_threshold),
            "large_shock_frequency_per_day": float(shock_mask.sum() / days),
            "positive_shock_frequency_per_day": float(positive_shock_mask.sum() / days),
            "negative_shock_frequency_per_day": float(negative_shock_mask.sum() / days),
            "return_autocorrelation_lag1": return_autocorr,
            "volatility_clustering_lag1": volatility_clustering,
        },
        "regime_probabilities": regime_probabilities,
        "regime_duration_distributions": {
            label: duration_distribution(values)
            for label, values in duration_by_label.items()
        },
        "event_frequency_distributions": {
            "breakout_frequency_per_day": float((up_breakout.sum() + down_breakout.sum()) / days),
            "up_breakout_frequency_per_day": float(up_breakout.sum() / days),
            "down_breakout_frequency_per_day": float(down_breakout.sum() / days),
            "fakeout_frequency_per_day": float((up_fakeout.sum() + down_fakeout.sum()) / days),
            "fakeout_after_breakout_rate": fakeout_rate,
            "large_shock_frequency_per_day": float(shock_mask.sum() / days),
        },
        "volume_trade_count_regimes": volume_summary,
        "spread_depth_regimes": microstructure_summary,
        "fair_value_waveform": build_low_frequency_wave(close),
        "simulator_parameters": {
            "regime_probabilities": regime_probabilities,
            "regime_duration_distributions": {
                label: duration_distribution(values)
                for label, values in duration_by_label.items()
            },
            "event_frequency_distributions": event_frequency_distributions,
            "volatility_ranges": {
                "overall_per_second": [vol_low, vol_high],
                "high_vol_per_second": [high_vol_low, high_vol_high],
            },
            "liquidity_ranges": {
                "thin": [0.25, 0.75],
                "normal": [0.75, 1.35],
                "deep": [1.10, 1.90],
            },
            "trend_bias_ranges": {
                "overall_per_second": [trend_low, trend_high],
                "uptrend_per_second": [max(0.0, trend_high * 0.25), max(0.00001, trend_high)],
                "downtrend_per_second": [min(-0.00001, trend_low), min(0.0, trend_low * 0.25)],
                "chop_per_second": [min(-0.00001, trend_low * 0.20), max(0.00001, trend_high * 0.20)],
            },
            "demand_intensity_ranges": demand_intensity_ranges,
        },
        "diagnostic_notes": [
            "Calibration contains aggregate simulator parameters only.",
            "The simulator samples regimes/events/demands from these ranges; it does not replay the historical path.",
            "No row-level future labels or model targets are stored.",
        ],
    }
    return payload


def atomic_write_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def main():
    history, used_paths = load_history()
    print("Historical simulator calibration")
    print(f"SYMBOL: {SYMBOL}")
    if len(history) == 0:
        print("No historical SOLUSDT/SYMBOL candle file found. Nothing was written.")
        print("Looked for imported candles, Binance public CSVs, and realtime flow CSVs under data/.")
        return
    payload = calibration_payload(history, used_paths)
    atomic_write_json(payload, OUTPUT_PATH)
    print(f"Rows used: {payload['row_count']}")
    print(f"First timestamp: {payload['first_timestamp']}")
    print(f"Last timestamp: {payload['last_timestamp']}")
    print(f"Timeframe seconds: {payload['timeframe_seconds']}")
    print(f"Source files used: {len(used_paths)}")
    for path in used_paths[:5]:
        print(f"- {path}")
    if len(used_paths) > 5:
        print(f"- ... {len(used_paths) - 5} more")
    print(f"Regime probabilities: {payload['regime_probabilities']}")
    print(f"Large shock frequency/day: {payload['historical_distribution']['large_shock_frequency_per_day']:.4f}")
    print(f"Return autocorrelation lag1: {payload['historical_distribution']['return_autocorrelation_lag1']:.4f}")
    print(f"Volatility clustering lag1: {payload['historical_distribution']['volatility_clustering_lag1']:.4f}")
    print(f"Calibration output: {OUTPUT_PATH}")
    print("Paper-only calibration. No trades/orders/private API.")


if __name__ == "__main__":
    main()
