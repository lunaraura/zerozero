import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
SOURCE_VENUE = os.getenv("SNAPSHOT_SOURCE_VENUE", os.getenv("PRIMARY_VENUE", "kraken")).strip().lower()
TRAIN_VENUE = os.getenv("SNAPSHOT_TRAIN_VENUE", f"{SOURCE_VENUE}_train_snapshot").strip().lower()
HOLDOUT_VENUE = os.getenv("SNAPSHOT_HOLDOUT_VENUE", f"{SOURCE_VENUE}_holdout_snapshot").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
HOLDOUT_HOURS_ENV = os.getenv("SNAPSHOT_HOLDOUT_HOURS", os.getenv("HOLDOUT_HOURS", "24")).strip()
HOLDOUT_DAYS_ENV = os.getenv("SNAPSHOT_HOLDOUT_DAYS", os.getenv("HOLDOUT_DAYS", "")).strip()
HOLDOUT_START_ENV = os.getenv("SNAPSHOT_HOLDOUT_START_TIME", os.getenv("HOLDOUT_START_TIME", "")).strip()
HOLDOUT_END_ENV = os.getenv("SNAPSHOT_HOLDOUT_END_TIME", os.getenv("HOLDOUT_END_TIME", "")).strip()

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR


def parse_time_ms(value):
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        numeric = float(text)
        if numeric > 1_000_000_000_000:
            return int(numeric)
        if numeric > 1_000_000_000:
            return int(numeric * 1000)
    except ValueError:
        pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def iso_time(timestamp_ms):
    if timestamp_ms is None:
        return ""
    return datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def normalize_timestamp_frame(frame):
    if len(frame) == 0:
        return frame
    frame = frame.copy()
    if "timestamp" not in frame.columns:
        raise SystemExit("Snapshot CSV is missing timestamp column.")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).copy()
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return frame


def split_bounds(source_10s):
    if len(source_10s) == 0:
        raise SystemExit("Cannot choose split bounds from an empty 10s snapshot file.")
    min_ts = int(source_10s["timestamp"].min())
    max_ts = int(source_10s["timestamp"].max())
    explicit_start = parse_time_ms(HOLDOUT_START_ENV)
    explicit_end = parse_time_ms(HOLDOUT_END_ENV)
    if explicit_start is not None:
        holdout_start = explicit_start
        holdout_end = explicit_end if explicit_end is not None else max_ts + 1
        mode = "explicit"
    else:
        if HOLDOUT_DAYS_ENV:
            holdout_ms = int(float(HOLDOUT_DAYS_ENV) * 24 * 3600 * 1000)
            mode = f"last_{HOLDOUT_DAYS_ENV}_days"
        else:
            holdout_ms = int(float(HOLDOUT_HOURS_ENV or "24") * 3600 * 1000)
            mode = f"last_{HOLDOUT_HOURS_ENV or '24'}_hours"
        holdout_end = explicit_end if explicit_end is not None else max_ts + 1
        holdout_start = holdout_end - holdout_ms
    holdout_start = max(min_ts, int(holdout_start))
    holdout_end = min(max_ts + 1, int(holdout_end))
    if holdout_start >= holdout_end:
        raise SystemExit(f"Invalid holdout bounds: start={holdout_start} end={holdout_end}")
    return holdout_start, holdout_end, mode


def split_frame(frame, holdout_start, holdout_end):
    if len(frame) == 0:
        return frame.copy(), frame.copy()
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
    train = frame[timestamps < holdout_start].copy()
    holdout = frame[(timestamps >= holdout_start) & (timestamps < holdout_end)].copy()
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


def main():
    source_dir = OUTPUT_DIR / SOURCE_VENUE
    train_dir = OUTPUT_DIR / TRAIN_VENUE
    holdout_dir = OUTPUT_DIR / HOLDOUT_VENUE
    source_10s_path = source_dir / f"{SYMBOL}_10s_flow.csv"
    source_1m_path = source_dir / f"{SYMBOL}_1m_flow.csv"
    source_10s = normalize_timestamp_frame(read_csv(source_10s_path))
    source_1m = normalize_timestamp_frame(read_csv(source_1m_path))
    if len(source_10s) == 0:
        raise SystemExit(f"Missing or empty source 10s snapshots: {source_10s_path}")

    holdout_start, holdout_end, mode = split_bounds(source_10s)
    train_10s, holdout_10s = split_frame(source_10s, holdout_start, holdout_end)
    train_1m, holdout_1m = split_frame(source_1m, holdout_start, holdout_end)

    train_10s_path = train_dir / f"{SYMBOL}_10s_flow.csv"
    holdout_10s_path = holdout_dir / f"{SYMBOL}_10s_flow.csv"
    train_1m_path = train_dir / f"{SYMBOL}_1m_flow.csv"
    holdout_1m_path = holdout_dir / f"{SYMBOL}_1m_flow.csv"
    atomic_write_csv(train_10s, train_10s_path)
    atomic_write_csv(holdout_10s, holdout_10s_path)
    if len(source_1m):
        atomic_write_csv(train_1m, train_1m_path)
        atomic_write_csv(holdout_1m, holdout_1m_path)

    payload = {
        "symbol": SYMBOL,
        "source_venue": SOURCE_VENUE,
        "train_venue": TRAIN_VENUE,
        "holdout_venue": HOLDOUT_VENUE,
        "mode": mode,
        "holdout_start_timestamp": holdout_start,
        "holdout_end_timestamp": holdout_end,
        "holdout_start_time": iso_time(holdout_start),
        "holdout_end_time": iso_time(holdout_end),
        "source_10s_rows": int(len(source_10s)),
        "train_10s_rows": int(len(train_10s)),
        "holdout_10s_rows": int(len(holdout_10s)),
        "source_1m_rows": int(len(source_1m)),
        "train_1m_rows": int(len(train_1m)),
        "holdout_1m_rows": int(len(holdout_1m)),
        "train_10s_path": str(train_10s_path),
        "holdout_10s_path": str(holdout_10s_path),
        "paper_only": True,
    }
    atomic_write_json(payload, OUTPUT_DIR / f"{SYMBOL}_snapshot_split_{TRAIN_VENUE}_to_{HOLDOUT_VENUE}.json")

    print("Chronological snapshot split complete.")
    print(f"SYMBOL: {SYMBOL}")
    print(f"SOURCE_VENUE: {SOURCE_VENUE}")
    print(f"TRAIN_VENUE: {TRAIN_VENUE}")
    print(f"HOLDOUT_VENUE: {HOLDOUT_VENUE}")
    print(f"Split mode: {mode}")
    print(f"Holdout: {iso_time(holdout_start)} -> {iso_time(holdout_end)}")
    print(f"10s rows: source={len(source_10s)} train={len(train_10s)} holdout={len(holdout_10s)}")
    print(f"1m rows: source={len(source_1m)} train={len(train_1m)} holdout={len(holdout_1m)}")
    print(f"Train 10s output: {train_10s_path}")
    print(f"Holdout 10s output: {holdout_10s_path}")
    print("Paper-only. No private API. No orders. No promotion.")


if __name__ == "__main__":
    main()
