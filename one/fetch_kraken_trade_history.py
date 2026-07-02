"""
Fetch and cache public Kraken historical trades.

This script is paper/research-only. It uses Kraken's public REST Trades
endpoint, requires no API keys, and never places orders.

Output:
  data/historical_trades/<venue>/<SYMBOL>_trades.csv

Columns:
  timestamp_ms, price, size, side
"""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KRAKEN_TRADES_URL = "https://api.kraken.com/0/public/Trades"

SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
VENUE = os.getenv("VENUE", "kraken").strip().lower()
OUTPUT_ROOT = Path(os.getenv("HISTORICAL_TRADES_DIR", PROJECT_ROOT / "data" / "historical_trades"))
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

OUTPUT_PATH = OUTPUT_ROOT / VENUE / f"{SYMBOL}_trades.csv"

RATE_LIMIT_SECONDS = float(os.getenv("KRAKEN_RATE_LIMIT_SECONDS", "1.2"))
MAX_RETRIES = int(os.getenv("KRAKEN_MAX_RETRIES", "8"))
MAX_PAGES = int(os.getenv("KRAKEN_TRADES_MAX_PAGES", "100000"))


def parse_time_ms(value: str | None) -> int | None:
    """Parse an env time as ms, seconds, or ISO-8601 UTC-ish text."""
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


def kraken_pair_from_symbol(symbol: str) -> str:
    """Best-effort Kraken pair conversion. Use KRAKEN_PAIR to override."""
    override = os.getenv("KRAKEN_PAIR", "").strip()
    if override:
        return override

    cleaned = symbol.replace("/", "").replace("-", "").upper()
    quote_candidates = ["USDT", "USD", "EUR", "GBP", "BTC", "ETH"]
    for quote in quote_candidates:
        if cleaned.endswith(quote):
            base = cleaned[: -len(quote)]
            kraken_base = "XBT" if base == "BTC" else base
            kraken_quote = "XBT" if quote == "BTC" else quote
            return f"{kraken_base}{kraken_quote}"
    return cleaned.replace("BTC", "XBT")


def read_cached_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            try:
                timestamp_ms = int(float(row.get("timestamp_ms", "")))
                price = float(row.get("price", ""))
                size = float(row.get("size", ""))
            except (TypeError, ValueError):
                continue
            side = str(row.get("side", "")).strip().lower()
            if side not in {"buy", "sell"}:
                continue
            rows.append(
                {
                    "timestamp_ms": str(timestamp_ms),
                    "price": f"{price:.12g}",
                    "size": f"{size:.12g}",
                    "side": side,
                }
            )
    return sorted(dedupe_rows(rows), key=lambda item: int(item["timestamp_ms"]))


def dedupe_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str]] = set()
    output: list[dict[str, str]] = []
    for row in rows:
        key = (
            str(int(float(row["timestamp_ms"]))),
            f"{float(row['price']):.12g}",
            f"{float(row['size']):.12g}",
            str(row["side"]).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "timestamp_ms": key[0],
                "price": key[1],
                "size": key[2],
                "side": key[3],
            }
        )
    return output


def atomic_write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_ms", "price", "size", "side"])
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def kraken_request(pair: str, since_cursor: str | None) -> dict:
    params = {"pair": pair}
    if since_cursor:
        params["since"] = since_cursor
    url = f"{KRAKEN_TRADES_URL}?{urllib.parse.urlencode(params)}"

    backoff = RATE_LIMIT_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "paper-research-trade-fetcher/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            errors = payload.get("error") or []
            if errors:
                error_text = "; ".join(errors)
                if any("rate" in error.lower() or "limit" in error.lower() for error in errors):
                    print(f"Kraken rate-limit response; backing off {backoff:.1f}s: {error_text}")
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 60.0)
                    continue
                raise RuntimeError(f"Kraken returned error: {error_text}")
            return payload["result"]
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else backoff
                print(f"HTTP 429 from Kraken; backing off {delay:.1f}s.")
                time.sleep(delay)
                backoff = min(backoff * 2.0, 60.0)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= MAX_RETRIES:
                raise
            print(f"Network issue fetching Kraken trades; retry {attempt}/{MAX_RETRIES}: {exc}")
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)

    raise RuntimeError("Kraken request failed after retries.")


def parse_kraken_trade_rows(result: dict, pair: str) -> tuple[list[dict[str, str]], str | None]:
    pair_keys = [key for key in result.keys() if key != "last"]
    if not pair_keys:
        return [], result.get("last")
    pair_key = pair if pair in pair_keys else pair_keys[0]
    parsed: list[dict[str, str]] = []
    for raw in result.get(pair_key, []):
        # Kraken trade format: price, volume, time, side, ordertype, misc, ...
        try:
            price = float(raw[0])
            size = float(raw[1])
            timestamp_ms = int(float(raw[2]) * 1000)
            side = "buy" if str(raw[3]).lower().startswith("b") else "sell"
        except (IndexError, TypeError, ValueError):
            continue
        if price <= 0 or size <= 0:
            continue
        parsed.append(
            {
                "timestamp_ms": str(timestamp_ms),
                "price": f"{price:.12g}",
                "size": f"{size:.12g}",
                "side": side,
            }
        )
    return parsed, result.get("last")


def main() -> None:
    pair = kraken_pair_from_symbol(SYMBOL)

    default_start = datetime.now(UTC) - timedelta(hours=1)
    default_end = datetime.now(UTC)
    start_ms = (
        parse_time_ms(os.getenv("TRADE_START_TIME"))
        or parse_time_ms(os.getenv("START_TIME"))
        or parse_time_ms(os.getenv("KRAKEN_START_TIME"))
        or int(default_start.timestamp() * 1000)
    )
    end_ms = (
        parse_time_ms(os.getenv("TRADE_END_TIME"))
        or parse_time_ms(os.getenv("END_TIME"))
        or parse_time_ms(os.getenv("KRAKEN_END_TIME"))
        or int(default_end.timestamp() * 1000)
    )
    if end_ms <= start_ms:
        raise SystemExit("End time must be after start time.")

    cached_rows = read_cached_rows(OUTPUT_PATH)
    cache_last_ms = max((int(row["timestamp_ms"]) for row in cached_rows), default=None)
    fetch_start_ms = start_ms
    if cache_last_ms is not None:
        fetch_start_ms = max(fetch_start_ms, cache_last_ms + 1)

    print("Kraken historical trade fetcher")
    print(f"SYMBOL: {SYMBOL}")
    print(f"Kraken pair: {pair}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Requested range: {iso_time(start_ms)} -> {iso_time(end_ms)}")
    print(f"Cached rows: {len(cached_rows)}")
    if cache_last_ms is not None:
        print(f"Last cached timestamp: {iso_time(cache_last_ms)}")

    if cache_last_ms is not None and cache_last_ms >= end_ms:
        filtered = [row for row in cached_rows if start_ms <= int(row["timestamp_ms"]) <= end_ms]
        atomic_write_rows(OUTPUT_PATH, sorted(dedupe_rows(cached_rows), key=lambda item: int(item["timestamp_ms"])))
        print(f"Cache already covers requested end time. Rows in requested range: {len(filtered)}")
        print("Paper-only. No private API keys. No orders.")
        return

    since_cursor = str(fetch_start_ms * 1_000_000)
    fetched_rows: list[dict[str, str]] = []
    for page in range(1, MAX_PAGES + 1):
        current_cursor = since_cursor
        result = kraken_request(pair, since_cursor)
        page_rows, last_cursor = parse_kraken_trade_rows(result, pair)
        new_page_rows = [
            row for row in page_rows if fetch_start_ms <= int(row["timestamp_ms"]) <= end_ms
        ]
        fetched_rows.extend(new_page_rows)
        max_page_ms = max((int(row["timestamp_ms"]) for row in page_rows), default=None)
        print(
            f"page={page} raw_rows={len(page_rows)} kept_rows={len(new_page_rows)} "
            f"max_page_time={iso_time(max_page_ms) if max_page_ms else 'none'}"
        )

        if max_page_ms is not None and max_page_ms >= end_ms:
            break
        if not last_cursor or str(last_cursor) == str(current_cursor):
            print("Kraken cursor did not advance; stopping.")
            break
        since_cursor = str(last_cursor)
        time.sleep(RATE_LIMIT_SECONDS)

    combined_rows = sorted(dedupe_rows([*cached_rows, *fetched_rows]), key=lambda item: int(item["timestamp_ms"]))
    atomic_write_rows(OUTPUT_PATH, combined_rows)

    requested_rows = [row for row in combined_rows if start_ms <= int(row["timestamp_ms"]) <= end_ms]
    print("Fetch complete.")
    print(f"Fetched new rows kept: {len(fetched_rows)}")
    print(f"Total cached rows: {len(combined_rows)}")
    print(f"Rows in requested range: {len(requested_rows)}")
    if requested_rows:
        print(f"First requested trade: {iso_time(int(requested_rows[0]['timestamp_ms']))}")
        print(f"Last requested trade: {iso_time(int(requested_rows[-1]['timestamp_ms']))}")
    print("Paper-only. Public REST only. No private API keys. No orders.")


if __name__ == "__main__":
    main()
