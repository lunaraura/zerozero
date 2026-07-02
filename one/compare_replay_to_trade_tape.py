"""
Validate replayed market-flow output against the raw historical trade tape.

Inputs by default:
  data/historical_trades/<VENUE>/<SYMBOL>_trades.csv
  data/realtime/replayed/<VENUE>/<SYMBOL>_1m_flow.csv
  data/realtime/replayed/<VENUE>/<SYMBOL>_10s_flow.csv

Output:
  data/realtime/replayed/<VENUE>/<SYMBOL>_replay_validation.csv

Research-only. No model promotion, no live prediction writes, no orders.
"""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
VENUE = os.getenv("VENUE", os.getenv("PRIMARY_VENUE", "kraken")).strip().lower()
TRADE_SIZE_SCALE = float(os.getenv("SIM_TRADE_REPLAY_SIZE_SCALE", "1.0"))

MIN_SIGN_AGREEMENT = float(os.getenv("REPLAY_MIN_RETURN_SIGN_AGREEMENT", "0.55"))
MAX_MEAN_ABS_PRICE_ERROR_BPS = float(os.getenv("REPLAY_MAX_MEAN_ABS_PRICE_ERROR_BPS", "5.0"))
MAX_ABS_PRICE_ERROR_BPS = float(os.getenv("REPLAY_MAX_ABS_PRICE_ERROR_BPS", "50.0"))
MIN_SNAPSHOT_COVERAGE = float(os.getenv("REPLAY_MIN_SNAPSHOT_COVERAGE", "0.95"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def iso_time(timestamp_ms: float | int | None) -> str:
    if timestamp_ms is None or not math.isfinite(float(timestamp_ms)):
        return ""
    return datetime.fromtimestamp(float(timestamp_ms) / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise SystemExit(f"{label} is empty: {path}")
    return frame


def numeric_column(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def monotonic_status(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> tuple[bool, int]:
    timestamps = numeric_column(frame, timestamp_column).dropna().to_numpy(dtype=np.float64)
    if len(timestamps) <= 1:
        return True, 0
    diffs = np.diff(timestamps)
    duplicate_count = int((diffs == 0).sum())
    return bool((diffs >= 0).all()), duplicate_count


def quantiles(values: pd.Series, prefix: str, probs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {f"{prefix}_p{int(prob * 100):02d}": np.nan for prob in probs}
    result = {}
    for prob in probs:
        result[f"{prefix}_p{int(prob * 100):02d}"] = float(clean.quantile(prob))
    return result


def aggregate_trades_to_1m(trades: pd.DataFrame) -> pd.DataFrame:
    timestamp_column = "timestamp_ms" if "timestamp_ms" in trades.columns else "timestamp" if "timestamp" in trades.columns else ""
    required = {timestamp_column, "price", "size", "side"} if timestamp_column else {"timestamp_ms", "price", "size", "side"}
    missing = sorted(required - set(trades.columns))
    if missing:
        raise SystemExit(f"Trade tape is missing required columns or aliases: {missing}")

    frame = trades.copy()
    frame["timestamp_ms"] = pd.to_numeric(frame[timestamp_column], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["size"] = pd.to_numeric(frame["size"], errors="coerce")
    frame["side"] = frame["side"].astype(str).str.lower().str.strip()
    frame = frame.dropna(subset=["timestamp_ms", "price", "size"])
    frame = frame[(frame["price"] > 0) & (frame["size"] > 0) & frame["side"].isin(["buy", "sell"])]
    frame = frame.sort_values("timestamp_ms").reset_index(drop=True)
    frame["minute_start_ms"] = (frame["timestamp_ms"].astype("int64") // 60000) * 60000
    frame["minute_end_ms"] = frame["minute_start_ms"] + 59000

    grouped = frame.groupby("minute_end_ms", sort=True)
    ohlcv = grouped.agg(
        trade_open=("price", "first"),
        trade_high=("price", "max"),
        trade_low=("price", "min"),
        trade_close=("price", "last"),
        trade_volume=("size", "sum"),
        trade_count=("size", "size"),
        first_trade_timestamp_ms=("timestamp_ms", "first"),
        last_trade_timestamp_ms=("timestamp_ms", "last"),
    ).reset_index()
    ohlcv["trade_return_1m"] = ohlcv["trade_close"].pct_change()
    return ohlcv


def normalize_replay_1m(replayed_1m: pd.DataFrame) -> pd.DataFrame:
    frame = replayed_1m.copy()
    if "timestamp" not in frame.columns:
        raise SystemExit("Replayed 1m file is missing timestamp column.")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame["minute_end_ms"] = (frame["timestamp"].astype("int64") // 60000) * 60000 + 59000
    frame["replay_close"] = numeric_column(frame, "close")
    frame["replay_mid_price"] = numeric_column(frame, "mid_price")
    frame["replay_compare_price"] = frame["replay_close"].where(frame["replay_close"].notna(), frame["replay_mid_price"])
    frame["replay_volume"] = numeric_column(frame, "volume")
    frame["replay_trade_count"] = numeric_column(frame, "trade_count")
    frame["replay_return_1m"] = frame["replay_compare_price"].pct_change()
    return frame


def snapshot_expected_coverage(replayed_10s: pd.DataFrame) -> dict:
    timestamps = numeric_column(replayed_10s, "timestamp").dropna().sort_values().to_numpy(dtype=np.float64)
    if len(timestamps) <= 1:
        return {
            "replay_10s_rows": int(len(timestamps)),
            "expected_10s_rows": int(len(timestamps)),
            "snapshot_cadence_ms": np.nan,
            "snapshot_coverage": 1.0 if len(timestamps) else 0.0,
        }
    diffs = np.diff(timestamps)
    positive_diffs = diffs[diffs > 0]
    cadence = float(np.median(positive_diffs)) if len(positive_diffs) else np.nan
    if not math.isfinite(cadence) or cadence <= 0:
        expected = len(timestamps)
    else:
        expected = int(math.floor((timestamps[-1] - timestamps[0]) / cadence) + 1)
    coverage = len(timestamps) / expected if expected > 0 else 0.0
    return {
        "replay_10s_rows": int(len(timestamps)),
        "expected_10s_rows": int(expected),
        "snapshot_cadence_ms": cadence,
        "snapshot_coverage": float(coverage),
    }


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    valid = pd.to_numeric(a, errors="coerce").notna() & pd.to_numeric(b, errors="coerce").notna()
    if valid.sum() < 3:
        return np.nan
    left = pd.to_numeric(a[valid], errors="coerce")
    right = pd.to_numeric(b[valid], errors="coerce")
    if left.std(ddof=0) <= 1e-12 or right.std(ddof=0) <= 1e-12:
        return np.nan
    return float(left.corr(right))


def sign_agreement(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    left = pd.to_numeric(a, errors="coerce")
    right = pd.to_numeric(b, errors="coerce")
    valid = left.notna() & right.notna() & (left != 0) & (right != 0)
    if valid.sum() == 0:
        return np.nan, 0
    agreement = (np.sign(left[valid]) == np.sign(right[valid])).mean()
    return float(agreement), int(valid.sum())


def build_summary(
    trades: pd.DataFrame,
    replayed_1m: pd.DataFrame,
    replayed_10s: pd.DataFrame,
    trade_1m: pd.DataFrame,
    replay_1m: pd.DataFrame,
    merged: pd.DataFrame,
) -> dict:
    warnings: list[str] = []

    trade_timestamp_column = "timestamp_ms" if "timestamp_ms" in trades.columns else "timestamp" if "timestamp" in trades.columns else "timestamp_ms"
    trades_monotonic, _ = monotonic_status(trades.rename(columns={trade_timestamp_column: "timestamp"}))
    replay_1m_monotonic, replay_1m_duplicate_timestamps = monotonic_status(replayed_1m)
    replay_10s_monotonic, replay_10s_duplicate_timestamps = monotonic_status(replayed_10s)

    duplicate_trade_rows = int(
        trades.duplicated(subset=[trade_timestamp_column, "price", "size", "side"]).sum()
        if {trade_timestamp_column, "price", "size", "side"}.issubset(trades.columns)
        else 0
    )

    if not trades_monotonic:
        warnings.append("trade_timestamps_not_monotonic")
    if duplicate_trade_rows > 0:
        warnings.append("duplicate_trades_present")
    if not replay_1m_monotonic:
        warnings.append("replayed_1m_timestamps_not_monotonic")
    if not replay_10s_monotonic:
        warnings.append("replayed_10s_timestamps_not_monotonic")

    coverage = snapshot_expected_coverage(replayed_10s)
    if coverage["snapshot_coverage"] < MIN_SNAPSHOT_COVERAGE:
        warnings.append("replay_10s_rows_below_expected")

    if merged.empty:
        warnings.append("no_1m_overlap_between_trade_tape_and_replay")
        price_errors = pd.Series(dtype="float64")
        price_error_bps = pd.Series(dtype="float64")
    else:
        price_errors = (merged["replay_compare_price"] - merged["trade_close"]).abs()
        price_error_bps = price_errors / merged["trade_close"].abs().clip(lower=1e-12) * 10000.0

    mean_abs_price_error = float(price_errors.mean()) if len(price_errors) else np.nan
    max_abs_price_error = float(price_errors.max()) if len(price_errors) else np.nan
    mean_abs_price_error_bps = float(price_error_bps.mean()) if len(price_error_bps) else np.nan
    max_abs_price_error_bps = float(price_error_bps.max()) if len(price_error_bps) else np.nan

    if math.isfinite(mean_abs_price_error_bps) and mean_abs_price_error_bps > MAX_MEAN_ABS_PRICE_ERROR_BPS:
        warnings.append("mean_abs_price_error_material")
    if math.isfinite(max_abs_price_error_bps) and max_abs_price_error_bps > MAX_ABS_PRICE_ERROR_BPS:
        warnings.append("max_abs_price_error_material")

    return_corr = safe_corr(merged.get("replay_return_1m", pd.Series(dtype="float64")), merged.get("trade_return_1m", pd.Series(dtype="float64")))
    return_sign_agreement, sign_rows = sign_agreement(
        merged.get("replay_return_1m", pd.Series(dtype="float64")),
        merged.get("trade_return_1m", pd.Series(dtype="float64")),
    )
    if math.isfinite(return_sign_agreement) and return_sign_agreement < MIN_SIGN_AGREEMENT:
        warnings.append("low_replayed_1m_return_sign_agreement")

    matched_trade_count = float(merged["trade_count"].sum()) if "trade_count" in merged else 0.0
    matched_replay_trade_count = float(merged["replay_trade_count"].sum()) if "replay_trade_count" in merged else 0.0
    matched_trade_volume = float(merged["trade_volume"].sum()) if "trade_volume" in merged else 0.0
    matched_scaled_trade_volume = matched_trade_volume * TRADE_SIZE_SCALE
    matched_replay_volume = float(merged["replay_volume"].sum()) if "replay_volume" in merged else 0.0

    summary = {
        "symbol": SYMBOL,
        "venue": VENUE,
        "status": "warn" if warnings else "ok",
        "warnings": ";".join(warnings),
        "trade_rows": int(len(trades)),
        "trade_1m_rows": int(len(trade_1m)),
        "replay_1m_rows": int(len(replay_1m)),
        "matched_1m_rows": int(len(merged)),
        "first_trade_time": iso_time(pd.to_numeric(trades.get(trade_timestamp_column), errors="coerce").min()),
        "last_trade_time": iso_time(pd.to_numeric(trades.get(trade_timestamp_column), errors="coerce").max()),
        "first_replay_1m_time": iso_time(pd.to_numeric(replayed_1m.get("timestamp"), errors="coerce").min()),
        "last_replay_1m_time": iso_time(pd.to_numeric(replayed_1m.get("timestamp"), errors="coerce").max()),
        "trade_timestamps_monotonic": trades_monotonic,
        "replay_1m_timestamps_monotonic": replay_1m_monotonic,
        "replay_10s_timestamps_monotonic": replay_10s_monotonic,
        "duplicate_trade_rows": duplicate_trade_rows,
        "replay_1m_duplicate_timestamps": replay_1m_duplicate_timestamps,
        "replay_10s_duplicate_timestamps": replay_10s_duplicate_timestamps,
        "mean_abs_price_error": mean_abs_price_error,
        "max_abs_price_error": max_abs_price_error,
        "mean_abs_price_error_bps": mean_abs_price_error_bps,
        "max_abs_price_error_bps": max_abs_price_error_bps,
        "return_correlation_1m": return_corr,
        "return_sign_agreement_1m": return_sign_agreement,
        "return_sign_agreement_rows": sign_rows,
        "matched_trade_count": matched_trade_count,
        "matched_replay_trade_count": matched_replay_trade_count,
        "trade_count_preservation_ratio": matched_replay_trade_count / matched_trade_count if matched_trade_count else np.nan,
        "matched_trade_volume": matched_trade_volume,
        "matched_scaled_trade_volume": matched_scaled_trade_volume,
        "matched_replay_volume": matched_replay_volume,
        "volume_preservation_ratio_vs_scaled_trade_volume": (
            matched_replay_volume / matched_scaled_trade_volume if matched_scaled_trade_volume else np.nan
        ),
        **coverage,
    }

    for column in [
        "spread_percent",
        "bid_depth_10bps",
        "ask_depth_10bps",
        "bid_depth_25bps",
        "ask_depth_25bps",
        "order_book_imbalance_10bps",
        "order_book_imbalance_25bps",
    ]:
        if column in replayed_10s.columns:
            summary.update(quantiles(replayed_10s[column], f"replay_{column}"))

    return summary


def main() -> None:
    trade_path = resolve_path(
        os.getenv("TRADE_HISTORY_PATH", PROJECT_ROOT / "data" / "historical_trades" / VENUE / f"{SYMBOL}_trades.csv")
    )
    replay_root = resolve_path(os.getenv("REPLAY_OUTPUT_ROOT", PROJECT_ROOT / "data" / "realtime" / "replayed"))
    replay_1m_path = resolve_path(os.getenv("REPLAY_1M_PATH", replay_root / VENUE / f"{SYMBOL}_1m_flow.csv"))
    replay_10s_path = resolve_path(os.getenv("REPLAY_10S_PATH", replay_root / VENUE / f"{SYMBOL}_10s_flow.csv"))
    output_path = resolve_path(os.getenv("REPLAY_VALIDATION_OUTPUT_PATH", replay_root / VENUE / f"{SYMBOL}_replay_validation.csv"))

    print("Replay vs trade-tape validation")
    print(f"SYMBOL: {SYMBOL}")
    print(f"VENUE: {VENUE}")
    print(f"Trade tape: {trade_path}")
    print(f"Replayed 1m: {replay_1m_path}")
    print(f"Replayed 10s: {replay_10s_path}")

    trades = read_csv_required(trade_path, "trade tape")
    replayed_1m = read_csv_required(replay_1m_path, "replayed 1m flow")
    replayed_10s = read_csv_required(replay_10s_path, "replayed 10s flow")

    trade_1m = aggregate_trades_to_1m(trades)
    replay_1m = normalize_replay_1m(replayed_1m)
    merged = replay_1m.merge(trade_1m, on="minute_end_ms", how="inner")

    summary = build_summary(trades, replayed_1m, replayed_10s, trade_1m, replay_1m, merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_path, index=False)

    print(f"Status: {summary['status']}")
    if summary["warnings"]:
        print(f"Warnings: {summary['warnings']}")
    print(f"Matched 1m rows: {summary['matched_1m_rows']}")
    print(f"Mean abs price error: {summary['mean_abs_price_error']:.8g}")
    print(f"Max abs price error: {summary['max_abs_price_error']:.8g}")
    print(f"Mean abs price error bps: {summary['mean_abs_price_error_bps']:.4f}")
    print(f"Max abs price error bps: {summary['max_abs_price_error_bps']:.4f}")
    print(f"1m return correlation: {summary['return_correlation_1m']}")
    print(f"1m return sign agreement: {summary['return_sign_agreement_1m']}")
    print(f"Trade count preservation ratio: {summary['trade_count_preservation_ratio']}")
    print(f"Volume preservation ratio vs scaled trade volume: {summary['volume_preservation_ratio_vs_scaled_trade_volume']}")
    print(f"Replay snapshot coverage: {summary['snapshot_coverage']:.4f}")
    print(f"Wrote validation summary: {output_path}")
    print("Research-only. No model promotion, no live prediction writes, no orders.")


if __name__ == "__main__":
    main()
