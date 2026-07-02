"""
Replay historical trades through the synthetic market simulator's book.

This is a paper-only bridge between real trade prints and the existing
10s_flow / 1m_flow schema. It does not place orders, use private APIs, or
modify the procedural simulator.

Key replay behavior:
  - Synthetic market makers provide resting liquidity.
  - Real historical trades consume that book as market orders.
  - Fair value is re-synced to the book after real trades each second.
  - Procedural taker agents and procedural return caps are not used.
"""

from __future__ import annotations

import csv
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
SOURCE_VENUE = os.getenv("VENUE", os.getenv("PRIMARY_VENUE", "kraken")).strip().lower()
SCENARIO = os.getenv("SCENARIO", "calm_chop").strip().lower()
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))
SIZE_SCALE = float(os.getenv("SIM_TRADE_REPLAY_SIZE_SCALE", "1.0"))
ANCHOR_MODE = os.getenv("SIM_REPLAY_ANCHOR_MODE", "trade_price").strip().lower()
if ANCHOR_MODE not in {"trade_price", "hybrid", "book_mid"}:
    raise SystemExit("SIM_REPLAY_ANCHOR_MODE must be one of: trade_price, hybrid, book_mid")
NET_FAVORABLE_COST_BUFFER_BPS = float(os.getenv("SIM_REPLAY_NET_FAVORABLE_COST_BUFFER_BPS", "0.0"))

os.environ.setdefault("SIM_REPLAY_MODE", "true")


def parse_time_ms(value: str | None) -> int | None:
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


def iso_time(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_trade_path() -> Path:
    return PROJECT_ROOT / "data" / "historical_trades" / SOURCE_VENUE / f"{SYMBOL}_trades.csv"


def first_env_time_ms(*names: str) -> int | None:
    for name in names:
        parsed = parse_time_ms(os.getenv(name))
        if parsed is not None:
            return parsed
    return None


def trade_timestamp_from_row(row: dict) -> int:
    for column in ("timestamp_ms", "timestamp"):
        value = row.get(column)
        if value is None or str(value).strip() == "":
            continue
        return int(float(value))
    raise ValueError("missing timestamp_ms/timestamp")


def first_valid_trade_timestamp(path: Path) -> int | None:
    if not path.exists():
        raise SystemExit(f"Trade history file was not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                return trade_timestamp_from_row(row)
            except (TypeError, ValueError):
                continue
    return None


def read_trades(path: Path, start_ms: int | None = None, end_ms: int | None = None) -> tuple[list[dict], dict]:
    if not path.exists():
        raise SystemExit(f"Trade history file was not found: {path}")
    rows: list[dict] = []
    diagnostics = {
        "rows_scanned": 0,
        "rows_loaded": 0,
        "rows_skipped_before_range": 0,
        "rows_stopped_after_range": 0,
        "rows_malformed": 0,
        "timestamp_column": "",
    }
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        timestamp_column = "timestamp_ms" if "timestamp_ms" in fieldnames else "timestamp" if "timestamp" in fieldnames else ""
        diagnostics["timestamp_column"] = timestamp_column
        required = {timestamp_column, "price", "size", "side"} if timestamp_column else {"timestamp_ms", "price", "size", "side"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"Trade CSV is missing required columns or aliases: {missing}")
        previous_timestamp_ms = None
        non_monotonic_rows = 0
        for row in reader:
            diagnostics["rows_scanned"] += 1
            try:
                timestamp_ms = trade_timestamp_from_row(row)
                price = float(row["price"])
                size = float(row["size"])
            except (TypeError, ValueError):
                diagnostics["rows_malformed"] += 1
                continue
            if previous_timestamp_ms is not None and timestamp_ms < previous_timestamp_ms:
                non_monotonic_rows += 1
            previous_timestamp_ms = timestamp_ms
            if start_ms is not None and timestamp_ms < start_ms:
                diagnostics["rows_skipped_before_range"] += 1
                continue
            if end_ms is not None and timestamp_ms >= end_ms:
                diagnostics["rows_stopped_after_range"] += 1
                break
            side = str(row.get("side", "")).strip().lower()
            if side not in {"buy", "sell"} or price <= 0 or size <= 0:
                diagnostics["rows_malformed"] += 1
                continue
            rows.append({"timestamp_ms": timestamp_ms, "price": price, "size": size, "side": side})
            diagnostics["rows_loaded"] += 1
    diagnostics["non_monotonic_rows"] = non_monotonic_rows
    return rows, diagnostics


def set_sim_start_price_before_import(start_price: float) -> None:
    """The simulator module reads SIM_START_PRICE at import time."""
    if "SIM_START_PRICE" not in os.environ:
        os.environ["SIM_START_PRICE"] = f"{start_price:.12g}"
    if "SYMBOL" not in os.environ:
        os.environ["SYMBOL"] = SYMBOL
    if "SCENARIO" not in os.environ:
        os.environ["SCENARIO"] = SCENARIO


def import_simulator():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from simulate_market_microstructure import (  # pylint: disable=import-error,import-outside-toplevel
        BotSwarm,
        LatentState,
        LimitOrderBook,
        ONE_MINUTE_COLUMNS,
        SNAPSHOT_COLUMNS,
        finite_float,
        one_minute_row,
        round_price,
        snapshot_from_book,
        write_csv,
    )

    return {
        "BotSwarm": BotSwarm,
        "LatentState": LatentState,
        "LimitOrderBook": LimitOrderBook,
        "SNAPSHOT_COLUMNS": SNAPSHOT_COLUMNS,
        "ONE_MINUTE_COLUMNS": ONE_MINUTE_COLUMNS,
        "snapshot_from_book": snapshot_from_book,
        "one_minute_row": one_minute_row,
        "write_csv": write_csv,
        "round_price": round_price,
        "finite_float": finite_float,
    }


def clone_row_with_replay_venue(row: dict) -> dict:
    output = dict(row)
    output["venue"] = f"replayed_{SOURCE_VENUE}"
    output["source_quality"] = "trade_replay_synthetic_book"
    return output


def summarize_levels(levels_touched: list[int]) -> dict:
    if not levels_touched:
        return {
            "average": 0.0,
            "median": 0.0,
            "distribution": {},
        }
    return {
        "average": float(statistics.mean(levels_touched)),
        "median": float(statistics.median(levels_touched)),
        "distribution": dict(sorted(Counter(levels_touched).items())),
    }


def return_std_and_range_from_prices(prices: pd.Series) -> dict:
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    prices = prices[prices > 0]
    if len(prices) < 2:
        return {
            "rows": int(len(prices)),
            "return_std_bps": math.nan,
            "mid_range_bps": math.nan,
        }
    returns_bps = prices.pct_change().dropna() * 10000.0
    first = float(prices.iloc[0])
    mid_range_bps = (float(prices.max()) / first - float(prices.min()) / first) * 10000.0 if first > 0 else math.nan
    return {
        "rows": int(len(prices)),
        "return_std_bps": float(returns_bps.std(ddof=0)) if len(returns_bps) else math.nan,
        "mid_range_bps": float(mid_range_bps),
    }


def trade_tape_close_series(trades: list[dict], bucket_ms: int) -> pd.Series:
    if not trades:
        return pd.Series(dtype="float64")
    frame = pd.DataFrame(trades)
    frame["timestamp_ms"] = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["timestamp_ms", "price"])
    frame = frame[frame["price"] > 0].sort_values("timestamp_ms")
    if frame.empty:
        return pd.Series(dtype="float64")
    frame["bucket"] = (frame["timestamp_ms"].astype("int64") // bucket_ms) * bucket_ms
    return frame.groupby("bucket")["price"].last().sort_index()


def replay_mid_series(snapshots: list[dict], bucket_ms: int) -> pd.Series:
    if not snapshots:
        return pd.Series(dtype="float64")
    frame = pd.DataFrame(snapshots)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["mid_price"] = pd.to_numeric(frame["mid_price"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "mid_price"])
    frame = frame[frame["mid_price"] > 0].sort_values("timestamp")
    if frame.empty:
        return pd.Series(dtype="float64")
    frame["bucket"] = (frame["timestamp"].astype("int64") // bucket_ms) * bucket_ms
    return frame.groupby("bucket")["mid_price"].last().sort_index()


def spread_stats(snapshots: list[dict]) -> dict:
    if not snapshots:
        return {"mean_bps": math.nan, "min_bps": math.nan, "max_bps": math.nan}
    frame = pd.DataFrame(snapshots)
    if "spread_percent" not in frame.columns:
        return {"mean_bps": math.nan, "min_bps": math.nan, "max_bps": math.nan}
    spread_percent = pd.to_numeric(frame["spread_percent"], errors="coerce").dropna()
    if spread_percent.empty:
        return {"mean_bps": math.nan, "min_bps": math.nan, "max_bps": math.nan}
    # Recorder/simulator spread_percent is stored as a fraction, so
    # 0.00019 means roughly 1.9 bps.
    spread_bps = spread_percent * 10000.0
    return {
        "mean_bps": float(spread_bps.mean()),
        "min_bps": float(spread_bps.min()),
        "max_bps": float(spread_bps.max()),
    }


def approximate_label_rates(snapshots: list[dict], horizon_seconds: int = 30) -> dict:
    if len(snapshots) <= horizon_seconds:
        return {
            "net_favorable_rate": math.nan,
            "instability_rate": math.nan,
            "average_mfe_bps": math.nan,
            "average_mae_bps": math.nan,
        }
    frame = pd.DataFrame(snapshots)
    frame["mid_price"] = pd.to_numeric(frame["mid_price"], errors="coerce")
    frame["spread_percent"] = pd.to_numeric(frame.get("spread_percent", 0.0), errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["mid_price"])
    mids = frame["mid_price"].to_numpy(dtype="float64")
    spreads_bps = (frame["spread_percent"].to_numpy(dtype="float64") * 10000.0) + NET_FAVORABLE_COST_BUFFER_BPS
    favorable = []
    instability = []
    mfes = []
    maes = []
    for index in range(0, len(mids) - horizon_seconds):
        entry = mids[index]
        if entry <= 0 or not math.isfinite(entry):
            continue
        future = mids[index + 1 : index + horizon_seconds + 1]
        if len(future) < horizon_seconds:
            continue
        runup_bps = (float(future.max()) / entry - 1.0) * 10000.0
        drawdown_bps = (float(future.min()) / entry - 1.0) * 10000.0
        final_return_bps = (float(future[-1]) / entry - 1.0) * 10000.0
        cost_bps = max(0.0, spreads_bps[index])
        mfes.append(runup_bps)
        maes.append(drawdown_bps)
        favorable.append((runup_bps > cost_bps) or (abs(drawdown_bps) > cost_bps))
        instability.append(abs(final_return_bps) > cost_bps)
    return {
        "net_favorable_rate": float(sum(favorable) / len(favorable)) if favorable else math.nan,
        "instability_rate": float(sum(instability) / len(instability)) if instability else math.nan,
        "average_mfe_bps": float(statistics.mean(mfes)) if mfes else math.nan,
        "average_mae_bps": float(statistics.mean(maes)) if maes else math.nan,
    }


def print_replay_magnitude_diagnostics(trades: list[dict], snapshots: list[dict]) -> bool:
    raw_10s = return_std_and_range_from_prices(trade_tape_close_series(trades, 10_000))
    raw_1m = return_std_and_range_from_prices(trade_tape_close_series(trades, 60_000))
    replay_10s = return_std_and_range_from_prices(replay_mid_series(snapshots, 10_000))
    replay_1m = return_std_and_range_from_prices(replay_mid_series(snapshots, 60_000))
    spreads = spread_stats(snapshots)
    rates = approximate_label_rates(snapshots, horizon_seconds=30)

    print("Replay magnitude diagnostics")
    print(
        f"raw trade-tape 10s return std={raw_10s['return_std_bps']:.6g} bps; "
        f"replayed 10s return std={replay_10s['return_std_bps']:.6g} bps"
    )
    print(
        f"raw trade-tape 1m return std={raw_1m['return_std_bps']:.6g} bps; "
        f"replayed 1m return std={replay_1m['return_std_bps']:.6g} bps"
    )
    print(
        f"raw trade-tape mid range={raw_10s['mid_range_bps']:.6g} bps; "
        f"replayed mid range={replay_10s['mid_range_bps']:.6g} bps"
    )
    print(
        "replay spread bps: "
        f"mean={spreads['mean_bps']:.6g} min={spreads['min_bps']:.6g} max={spreads['max_bps']:.6g}"
    )
    print(
        "approx replay 30s label rates: "
        f"net_favorable_rate={rates['net_favorable_rate']:.4%} "
        f"instability_rate={rates['instability_rate']:.4%} "
        f"average_mfe_bps={rates['average_mfe_bps']:.6g} "
        f"average_mae_bps={rates['average_mae_bps']:.6g}"
    )

    warnings = []
    if math.isfinite(rates["net_favorable_rate"]) and rates["net_favorable_rate"] < 0.01:
        warnings.append("net_favorable_rate < 1%")
    if math.isfinite(rates["instability_rate"]) and rates["instability_rate"] < 0.01:
        warnings.append("instability_rate < 1%")
    if (
        math.isfinite(raw_10s["return_std_bps"])
        and math.isfinite(replay_10s["return_std_bps"])
        and raw_10s["return_std_bps"] > 0
        and replay_10s["return_std_bps"] < 0.5 * raw_10s["return_std_bps"]
    ):
        warnings.append("replayed 10s return std is less than 50% of raw trade-tape return std")
    if (
        math.isfinite(raw_10s["mid_range_bps"])
        and math.isfinite(replay_10s["mid_range_bps"])
        and raw_10s["mid_range_bps"] > 0
        and replay_10s["mid_range_bps"] < 0.5 * raw_10s["mid_range_bps"]
    ):
        warnings.append("replayed mid range is less than 50% of raw trade-tape mid range")

    validation_pass = len(warnings) == 0
    if warnings:
        print("REPLAY VALIDATION WARNING: replay output is too flat / not training-safe yet.")
        for warning in warnings:
            print(f"- {warning}")
        print("Do not use this replay output for model training summaries unless validation passes.")
    else:
        print("Replay validation magnitude check: PASS")
    return validation_pass


def compare_against_recorded(replayed_snapshots: list[dict], nested_output_path: Path) -> None:
    recorded_path = PROJECT_ROOT / "data" / "realtime" / SOURCE_VENUE / f"{SYMBOL}_10s_flow.csv"
    if not recorded_path.exists():
        print(f"No matching real recorded 10s flow found for comparison: {recorded_path}")
        return
    if recorded_path.resolve() == nested_output_path.resolve():
        return

    replayed_frame = pd.DataFrame(replayed_snapshots)
    real_frame = pd.read_csv(recorded_path, low_memory=False)
    if replayed_frame.empty or real_frame.empty or "timestamp" not in real_frame.columns:
        print("Recorded comparison skipped: empty file or missing timestamp column.")
        return

    replayed_frame["timestamp"] = pd.to_numeric(replayed_frame["timestamp"], errors="coerce")
    real_frame["timestamp"] = pd.to_numeric(real_frame["timestamp"], errors="coerce")
    merged = replayed_frame.merge(real_frame, on="timestamp", suffixes=("_replayed", "_recorded"))
    if merged.empty:
        print("Recorded comparison: no exact timestamp overlap.")
        return

    print("Recorded comparison on exact timestamp overlap")
    print(f"overlap rows: {len(merged)}")
    for column in [
        "spread_percent",
        "bid_depth_10bps",
        "ask_depth_10bps",
        "order_book_imbalance_10bps",
        "bid_depth_25bps",
        "ask_depth_25bps",
    ]:
        replayed_column = f"{column}_replayed"
        recorded_column = f"{column}_recorded"
        if replayed_column not in merged or recorded_column not in merged:
            continue
        replayed_values = pd.to_numeric(merged[replayed_column], errors="coerce")
        recorded_values = pd.to_numeric(merged[recorded_column], errors="coerce")
        valid = replayed_values.notna() & recorded_values.notna()
        if not valid.any():
            continue
        replayed_mean = float(replayed_values[valid].mean())
        recorded_mean = float(recorded_values[valid].mean())
        ratio = replayed_mean / recorded_mean if abs(recorded_mean) > 1e-12 else math.nan
        print(
            f"{column}: replayed_mean={replayed_mean:.8g} "
            f"recorded_mean={recorded_mean:.8g} ratio={ratio:.4g}"
        )


def write_compatibility_copy(write_csv, columns: list[str], rows: list[dict], compatibility_path: Path) -> None:
    """Write the legacy builder-friendly path: data/realtime/replayed/<SYMBOL>_*.csv."""
    write_csv(compatibility_path, columns, rows)


def write_replay_quality_marker(path: Path, validation_pass: bool) -> None:
    payload = {
        "symbol": SYMBOL,
        "source_venue": SOURCE_VENUE,
        "anchor_mode": ANCHOR_MODE,
        "validation_pass": bool(validation_pass),
        "paper_only": True,
        "message": (
            "Replay magnitude validation passed."
            if validation_pass
            else "Replay magnitude validation failed; do not use for tiny-price training summaries."
        ),
        "written_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(payload), encoding="utf-8")


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


def replay() -> tuple[list[dict], list[dict]]:
    trade_path = resolve_project_path(os.getenv("TRADE_HISTORY_PATH", default_trade_path()))
    requested_start_ms = (
        first_env_time_ms("REPLAY_START_TIME", "SIM_REPLAY_START_TIME", "TRADE_START_TIME", "START_TIME")
        or first_valid_trade_timestamp(trade_path)
    )
    explicit_end_ms = (
        first_env_time_ms("REPLAY_END_TIME", "SIM_REPLAY_END_TIME", "TRADE_END_TIME", "END_TIME")
    )
    if requested_start_ms is None:
        raise SystemExit(f"No valid trade timestamps found in {trade_path}")

    sim_seconds_env = os.getenv("SIM_SECONDS", "").strip()
    read_end_ms = explicit_end_ms
    if sim_seconds_env:
        sim_seconds = int(float(sim_seconds_env))
        read_end_ms = requested_start_ms + max(1, sim_seconds) * 1000
    else:
        sim_seconds = None

    trades, trade_read_diagnostics = read_trades(trade_path, requested_start_ms, read_end_ms)
    if not trades:
        raise SystemExit("No trades are inside the replay window.")

    replay_start_ms = requested_start_ms
    if sim_seconds_env:
        replay_end_ms = requested_start_ms + max(1, int(float(sim_seconds_env))) * 1000 - 1
    elif explicit_end_ms is not None:
        replay_end_ms = explicit_end_ms - 1
        sim_seconds = max(1, int((replay_end_ms // 1000) - (replay_start_ms // 1000) + 1))
    else:
        replay_end_ms = trades[-1]["timestamp_ms"]
        sim_seconds = max(1, int((replay_end_ms // 1000) - (replay_start_ms // 1000) + 1))

    usable_trades = trades
    if not usable_trades:
        raise SystemExit("No trades are inside the replay window.")

    start_price = float(os.getenv("SIM_START_PRICE", usable_trades[0]["price"]))
    set_sim_start_price_before_import(start_price)
    sim = import_simulator()

    rng = random.Random(RANDOM_SEED)
    state = sim["LatentState"](SCENARIO, rng)
    book = sim["LimitOrderBook"](start_price, rng)
    bots = sim["BotSwarm"](book, state, rng)

    trades_by_second: dict[int, list[dict]] = defaultdict(list)
    for trade in usable_trades:
        trades_by_second[trade["timestamp_ms"] // 1000].append(trade)

    snapshots: list[dict] = []
    one_minute_rows: list[dict] = []
    minute_bucket: list[dict] = []
    previous_snapshot = None
    previous_close = None
    trade_anchor_price = start_price
    levels_touched: list[int] = []
    trades_consumed = 0
    start_second = replay_start_ms // 1000
    end_second = replay_end_ms // 1000

    # Keep the latent state anchored to the first real print before quoting starts.
    state.fair_value = start_price
    state.active_event_type = "trade_replay"
    state.active_event_direction = 0.0
    state.active_event_magnitude = 0.0

    for second_ts in range(start_second, end_second + 1):
        # Do not call LatentState.update(); its independent noise process would
        # fight the historical trade-driven path in replay mode.
        state.active_event_type = "trade_replay"
        state.active_event_direction = 0.0
        state.active_event_magnitude = 0.0
        state.waveform_value = 0.0
        state.waveform_contribution = 0.0

        bots.market_maker()
        bots.mean_reversion_liquidity_provider()

        fills: list[dict] = []
        latest_trade_price_this_second = None
        for trade in trades_by_second.get(second_ts, []):
            scaled_size = max(0.0, trade["size"] * SIZE_SCALE)
            if scaled_size <= 0:
                continue
            fill = book.execute_market(trade["side"], scaled_size)
            fill["side"] = trade["side"]
            fill["agent"] = "historical_trade_replay"
            fill["historical_price"] = trade["price"]
            fill["historical_size"] = trade["size"]
            fills.append(fill)
            trades_consumed += 1
            levels_touched.append(int(fill.get("levels_touched", 0)))
            latest_trade_price_this_second = float(trade["price"])

        # Resync fair value after historical trades so maker quote skew follows
        # the real trade-driven path instead of procedural drift/noise.
        #
        # The important replay invariant: in trade_price/hybrid anchoring,
        # hidden_fair_value and the visible synthetic book mid follow the latest
        # raw trade-tape price. Synthetic depth/spread may vary, but the center
        # should not become a nearly fixed book-derived value.
        if latest_trade_price_this_second is not None:
            trade_anchor_price = latest_trade_price_this_second
        if ANCHOR_MODE in {"trade_price", "hybrid"}:
            state.fair_value = trade_anchor_price
            book.last_trade_price = trade_anchor_price
            book.shift_prices(trade_anchor_price)
        else:
            state.fair_value = book.mid_price()
        book.replenish_around(
            state.fair_value,
            state.liquidity_regime,
            getattr(state, "book_depth_multiplier", 1.0),
        )
        if ANCHOR_MODE in {"trade_price", "hybrid"}:
            book.shift_prices(trade_anchor_price)

        buy_volume = sum(fill["filled"] for fill in fills if fill.get("side") == "buy")
        sell_volume = sum(fill["filled"] for fill in fills if fill.get("side") == "sell")
        total_volume = buy_volume + sell_volume
        if getattr(bots, "recent_returns", None) is not None:
            if previous_snapshot:
                previous_mid = float(previous_snapshot["mid_price"])
                current_mid = book.mid_price()
                bots.recent_returns.append((current_mid - previous_mid) / previous_mid if previous_mid else 0.0)
            else:
                bots.recent_returns.append(0.0)
        if getattr(bots, "recent_pressures", None) is not None:
            bots.recent_pressures.append((buy_volume - sell_volume) / total_volume if total_volume else 0.0)
        bots.last_active_agent_types = ["market_maker", "mean_reversion_liquidity_provider"]
        if fills:
            bots.last_active_agent_types.append("historical_trade_replay")
        bots.last_agent_buy_intensity = buy_volume
        bots.last_agent_sell_intensity = sell_volume
        bots.last_whale_pressure = 0.0

        snapshot = sim["snapshot_from_book"](second_ts * 1000, book, state, fills, previous_snapshot)
        snapshot = clone_row_with_replay_venue(snapshot)
        if hasattr(state, "update_depth_feedback"):
            state.update_depth_feedback(snapshot["bid_depth_10bps"], snapshot["ask_depth_10bps"])
        snapshots.append(snapshot)
        minute_bucket.append(snapshot)
        previous_snapshot = snapshot

        if len(minute_bucket) == 60:
            row = sim["one_minute_row"](minute_bucket, previous_close)
            row = clone_row_with_replay_venue(row)
            one_minute_rows.append(row)
            previous_close = float(row["close"])
            minute_bucket = []

    output_root = resolve_project_path(os.getenv("REPLAY_OUTPUT_ROOT", PROJECT_ROOT / "data" / "realtime" / "replayed"))
    nested_dir = output_root / SOURCE_VENUE
    nested_snapshot_path = nested_dir / f"{SYMBOL}_10s_flow.csv"
    nested_one_minute_path = nested_dir / f"{SYMBOL}_1m_flow.csv"
    compatibility_snapshot_path = output_root / f"{SYMBOL}_10s_flow.csv"
    compatibility_one_minute_path = output_root / f"{SYMBOL}_1m_flow.csv"

    sim["write_csv"](nested_snapshot_path, sim["SNAPSHOT_COLUMNS"], snapshots)
    sim["write_csv"](nested_one_minute_path, sim["ONE_MINUTE_COLUMNS"], one_minute_rows)
    write_compatibility_copy(sim["write_csv"], sim["SNAPSHOT_COLUMNS"], snapshots, compatibility_snapshot_path)
    write_compatibility_copy(sim["write_csv"], sim["ONE_MINUTE_COLUMNS"], one_minute_rows, compatibility_one_minute_path)

    level_summary = summarize_levels(levels_touched)
    print("Historical trade replay complete.")
    print(f"SYMBOL: {SYMBOL}")
    print(f"SOURCE_VENUE: {SOURCE_VENUE}")
    print(f"SIM_REPLAY_MODE: {os.getenv('SIM_REPLAY_MODE')}")
    print(f"SIM_REPLAY_ANCHOR_MODE: {ANCHOR_MODE}")
    print(f"SIM_TRADE_REPLAY_SIZE_SCALE: {SIZE_SCALE}")
    print(f"Trade source: {trade_path}")
    print(f"Trade timestamp column: {trade_read_diagnostics.get('timestamp_column')}")
    print(f"Trade rows scanned: {trade_read_diagnostics.get('rows_scanned')}")
    print(f"Trade rows loaded: {trade_read_diagnostics.get('rows_loaded')}")
    print(f"Trade rows skipped before range: {trade_read_diagnostics.get('rows_skipped_before_range')}")
    print(f"Trade rows stopped after range: {trade_read_diagnostics.get('rows_stopped_after_range')}")
    print(f"Trade rows malformed/skipped: {trade_read_diagnostics.get('rows_malformed')}")
    print(f"Trade non-monotonic rows observed: {trade_read_diagnostics.get('non_monotonic_rows')}")
    print(f"Replay window: {iso_time(start_second * 1000)} -> {iso_time(end_second * 1000)}")
    print(f"10s-flow-shaped rows written: {len(snapshots)}")
    print(f"1m rows written: {len(one_minute_rows)}")
    print(f"Trades consumed: {trades_consumed}")
    print(
        "Trades skipped outside replay window: "
        f"{trade_read_diagnostics.get('rows_skipped_before_range', 0) + trade_read_diagnostics.get('rows_stopped_after_range', 0)}"
    )
    print(f"Average levels_touched: {level_summary['average']:.4f}")
    print(f"Median levels_touched: {level_summary['median']:.4f}")
    print(f"levels_touched distribution: {level_summary['distribution']}")
    print(f"Nested 10s output: {nested_snapshot_path}")
    print(f"Nested 1m output: {nested_one_minute_path}")
    print(f"Compatibility 10s output for PRIMARY_VENUE=replayed: {compatibility_snapshot_path}")
    print(f"Compatibility 1m output for PRIMARY_VENUE=replayed: {compatibility_one_minute_path}")
    validation_pass = print_replay_magnitude_diagnostics(usable_trades, snapshots)
    quality_marker_path = nested_dir / f"{SYMBOL}_replay_quality.json"
    compatibility_quality_marker_path = output_root / f"{SYMBOL}_replay_quality.json"
    write_replay_quality_marker(quality_marker_path, validation_pass)
    write_replay_quality_marker(compatibility_quality_marker_path, validation_pass)
    print(f"Replay quality marker: {quality_marker_path}")
    print(f"Compatibility replay quality marker: {compatibility_quality_marker_path}")
    compare_against_recorded(snapshots, nested_snapshot_path)
    print("Paper-only. No private API keys. No orders.")
    return snapshots, one_minute_rows


def main() -> None:
    replay()


if __name__ == "__main__":
    main()
