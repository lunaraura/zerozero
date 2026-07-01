import csv
import json
import math
import os
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
SCENARIO = os.getenv("SCENARIO", "calm_chop").strip().lower()
SIM_SECONDS = int(os.getenv("SIM_SECONDS", "7200"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "123"))
START_PRICE = float(os.getenv("SIM_START_PRICE", "70.0"))
PRICE_TICK = float(os.getenv("SIM_PRICE_TICK", "0.01"))
LEVEL_COUNT = int(os.getenv("SIM_LEVEL_COUNT", "80"))
BASE_LEVEL_SIZE = float(os.getenv("SIM_BASE_LEVEL_SIZE", "120.0"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "simulated"))
SIM_MAX_ABS_SECOND_RETURN = float(os.getenv("SIM_MAX_ABS_SECOND_RETURN", "0.0025"))
SIM_MAX_ABS_MINUTE_RETURN = float(os.getenv("SIM_MAX_ABS_MINUTE_RETURN", "0.012"))
SIM_MAX_CLOSE_TO_CLOSE_RETURN = float(os.getenv("SIM_MAX_CLOSE_TO_CLOSE_RETURN", "0.08"))
SIM_PRICE_IMPACT_SCALE = float(os.getenv("SIM_PRICE_IMPACT_SCALE", "0.45"))
SIM_EVENT_MAGNITUDE_SCALE = float(os.getenv("SIM_EVENT_MAGNITUDE_SCALE", "0.70"))
SIM_WHALE_INTENSITY = float(os.getenv("SIM_WHALE_INTENSITY", "1.0"))
SIM_RETAIL_INTENSITY = float(os.getenv("SIM_RETAIL_INTENSITY", "1.0"))
SIM_INSTITUTIONAL_INTENSITY = float(os.getenv("SIM_INSTITUTIONAL_INTENSITY", "1.0"))
SIM_ENABLE_RICH_AGENTS = os.getenv("SIM_ENABLE_RICH_AGENTS", "true").strip().lower() in {"1", "true", "yes"}
SIM_CALIBRATION_PATH_ENV = os.getenv("SIM_CALIBRATION_PATH", "").strip()
SIM_FAIR_VALUE_MODE = os.getenv("SIM_FAIR_VALUE_MODE", "stochastic").strip().lower()
if SIM_FAIR_VALUE_MODE not in {"stochastic", "historical_wave_calibrated"}:
    print(f"Unknown SIM_FAIR_VALUE_MODE={SIM_FAIR_VALUE_MODE}; using stochastic.")
    SIM_FAIR_VALUE_MODE = "stochastic"
START_TIMESTAMP_MS = int(
    os.getenv(
        "SIM_START_TIMESTAMP_MS",
        str(int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)),
    )
)

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

SCENARIO_DIR = OUTPUT_DIR / SCENARIO
SNAPSHOT_PATH = SCENARIO_DIR / f"{SYMBOL}_10s_flow.csv"
ONE_MINUTE_PATH = SCENARIO_DIR / f"{SYMBOL}_1m_flow.csv"


def resolve_project_path(value):
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_sim_calibration():
    path = resolve_project_path(SIM_CALIBRATION_PATH_ENV)
    if path is None:
        return None, None
    if not path.exists():
        print(f"SIM_CALIBRATION_PATH was set but file was not found: {path}")
        return None, path
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload, path
    except Exception as exc:
        print(f"Failed to load SIM_CALIBRATION_PATH={path}: {exc}")
        return None, path


SIM_CALIBRATION, SIM_CALIBRATION_PATH = load_sim_calibration()


SNAPSHOT_COLUMNS = [
    "venue",
    "source_quality",
    "timestamp",
    "time",
    "mid_price",
    "best_bid",
    "best_ask",
    "spread_percent",
    "market_buy_volume_10s",
    "market_sell_volume_10s",
    "total_trade_volume_10s",
    "trade_count_10s",
    "market_pressure_10s",
    "bid_depth_10bps",
    "ask_depth_10bps",
    "bid_depth_25bps",
    "ask_depth_25bps",
    "order_book_imbalance_10bps",
    "order_book_imbalance_25bps",
    "bid_depth_change_10bps",
    "ask_depth_change_10bps",
    "imbalance_change_10bps",
    "large_bid_wall_distance",
    "large_ask_wall_distance",
    "large_bid_wall_size",
    "large_ask_wall_size",
    "hidden_fair_value",
    "hidden_trend_bias",
    "hidden_volatility_regime",
    "hidden_liquidity_regime",
    "hidden_news_shock",
    "hidden_risk_aversion",
    "hidden_scenario",
    "hidden_retail_demand",
    "hidden_institutional_demand",
    "hidden_momentum_demand",
    "hidden_mean_reversion_demand",
    "hidden_liquidity_demand",
    "hidden_panic_demand",
    "hidden_news_demand",
    "hidden_macro_risk_demand",
    "hidden_active_event_type",
    "hidden_active_event_direction",
    "hidden_active_event_magnitude",
    "hidden_whale_pressure",
    "hidden_agent_buy_intensity",
    "hidden_agent_sell_intensity",
    "hidden_active_agent_types",
    "hidden_calibration_regime",
    "hidden_fair_value_waveform_value",
    "hidden_fair_value_waveform_contribution",
]

SNAPSHOT_SUMMARY_METRICS = [
    ("spread_percent", "snapshot_spread_percent"),
    ("order_book_imbalance_10bps", "snapshot_imbalance_10bps"),
    ("order_book_imbalance_25bps", "snapshot_imbalance_25bps"),
    ("bid_depth_10bps", "snapshot_bid_depth_10bps"),
    ("ask_depth_10bps", "snapshot_ask_depth_10bps"),
    ("bid_depth_25bps", "snapshot_bid_depth_25bps"),
    ("ask_depth_25bps", "snapshot_ask_depth_25bps"),
    ("market_pressure_10s", "snapshot_market_pressure_10s"),
]

ONE_MINUTE_COLUMNS = [
    "venue",
    "source_quality",
    "timestamp",
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_sell_volume",
    "taker_buy_ratio",
    "return_1",
    "range_percent",
    "upper_wick_percent",
    "lower_wick_percent",
    "close_position_in_range",
    "best_bid",
    "best_ask",
    "spread_percent",
    "mid_price",
    "bid_depth_10bps",
    "ask_depth_10bps",
    "bid_depth_25bps",
    "ask_depth_25bps",
    "order_book_imbalance_10bps",
    "order_book_imbalance_25bps",
    "bid_depth_change_10bps",
    "ask_depth_change_10bps",
    "large_bid_wall_distance",
    "large_ask_wall_distance",
    "large_bid_wall_size",
    "large_ask_wall_size",
]

for _, prefix in SNAPSHOT_SUMMARY_METRICS:
    ONE_MINUTE_COLUMNS.extend(
        [
            f"{prefix}_mean",
            f"{prefix}_min",
            f"{prefix}_max",
            f"{prefix}_last",
            f"{prefix}_change",
        ]
    )

ONE_MINUTE_COLUMNS.extend(
    [
        "hidden_fair_value",
        "hidden_trend_bias",
        "hidden_volatility_regime",
        "hidden_liquidity_regime",
        "hidden_news_shock",
        "hidden_risk_aversion",
        "hidden_scenario",
        "hidden_retail_demand",
        "hidden_institutional_demand",
        "hidden_momentum_demand",
        "hidden_mean_reversion_demand",
        "hidden_liquidity_demand",
        "hidden_panic_demand",
        "hidden_news_demand",
        "hidden_macro_risk_demand",
        "hidden_active_event_type",
        "hidden_active_event_direction",
        "hidden_active_event_magnitude",
        "hidden_whale_pressure",
        "hidden_agent_buy_intensity",
        "hidden_agent_sell_intensity",
        "hidden_active_agent_types",
        "hidden_calibration_regime",
        "hidden_fair_value_waveform_value",
        "hidden_fair_value_waveform_contribution",
    ]
)


SCENARIOS = {
    "calm_chop": {
        "trend_bias": 0.0,
        "volatility": 0.00008,
        "liquidity": 1.3,
        "risk_aversion": 0.25,
        "shock_direction": 0.0,
        "demand_bias": {"retail": 0.02, "mean_reversion": 0.10, "liquidity": 0.16},
        "events": [],
    },
    "bullish_breakout": {
        "trend_bias": 0.000055,
        "volatility": 0.00018,
        "liquidity": 1.05,
        "risk_aversion": 0.35,
        "shock_direction": 0.2,
        "demand_bias": {"retail": 0.08, "institutional": 0.10, "momentum": 0.12},
        "events": [("positive_news", 0.28, 1, 0.42, 540), ("short_squeeze", 0.55, 1, 0.36, 420)],
    },
    "bearish_breakdown": {
        "trend_bias": -0.000055,
        "volatility": 0.00018,
        "liquidity": 1.05,
        "risk_aversion": 0.35,
        "shock_direction": -0.2,
        "demand_bias": {"retail": -0.08, "institutional": -0.06, "momentum": -0.12},
        "events": [("negative_news", 0.28, -1, 0.42, 540), ("panic_selloff", 0.55, -1, 0.36, 420)],
    },
    "liquidity_crisis": {
        "trend_bias": -0.000015,
        "volatility": 0.00042,
        "liquidity": 0.32,
        "risk_aversion": 0.9,
        "shock_direction": -0.25,
        "demand_bias": {"liquidity": -0.50, "panic": -0.16, "macro_risk": -0.12},
        "events": [("exchange_outage", 0.20, -1, 0.38, 900), ("liquidity_vacuum", 0.42, -1, 0.55, 1200)],
    },
    "fakeout_reversal": {
        "trend_bias": 0.000045,
        "volatility": 0.00024,
        "liquidity": 0.85,
        "risk_aversion": 0.55,
        "shock_direction": 0.45,
        "demand_bias": {"momentum": 0.08, "mean_reversion": -0.04},
        "events": [("fake_breakout", 0.30, 1, 0.52, 420), ("mean_reversion_snapback", 0.48, -1, 0.50, 540)],
    },
    "news_shock_up": {
        "trend_bias": 0.00002,
        "volatility": 0.00032,
        "liquidity": 0.75,
        "risk_aversion": 0.7,
        "shock_direction": 1.0,
        "demand_bias": {"news": 0.04, "retail": 0.05},
        "events": [("positive_news", 0.35, 1, 0.85, 780)],
    },
    "news_shock_down": {
        "trend_bias": -0.00002,
        "volatility": 0.00032,
        "liquidity": 0.75,
        "risk_aversion": 0.7,
        "shock_direction": -1.0,
        "demand_bias": {"news": -0.04, "panic": -0.06},
        "events": [("negative_news", 0.35, -1, 0.85, 780)],
    },
    "accumulation_grind": {
        "trend_bias": 0.000018,
        "volatility": 0.00010,
        "liquidity": 1.15,
        "risk_aversion": 0.30,
        "shock_direction": 0.15,
        "demand_bias": {"institutional": 0.14, "liquidity": 0.06, "mean_reversion": 0.04},
        "events": [("whale_accumulation", 0.18, 1, 0.35, 1800)],
    },
    "distribution_grind": {
        "trend_bias": -0.000018,
        "volatility": 0.00010,
        "liquidity": 1.12,
        "risk_aversion": 0.36,
        "shock_direction": -0.15,
        "demand_bias": {"institutional": -0.14, "liquidity": 0.04, "mean_reversion": -0.03},
        "events": [("whale_distribution", 0.18, -1, 0.35, 1800)],
    },
    "high_vol_chop": {
        "trend_bias": 0.0,
        "volatility": 0.00034,
        "liquidity": 0.75,
        "risk_aversion": 0.62,
        "shock_direction": 0.0,
        "demand_bias": {"retail": 0.06, "mean_reversion": 0.18, "liquidity": -0.12},
        "events": [("positive_news", 0.25, 1, 0.25, 300), ("negative_news", 0.55, -1, 0.30, 360)],
    },
    "thin_book_fakeout": {
        "trend_bias": 0.000015,
        "volatility": 0.00030,
        "liquidity": 0.42,
        "risk_aversion": 0.75,
        "shock_direction": 0.3,
        "demand_bias": {"liquidity": -0.34, "momentum": 0.08},
        "events": [("liquidity_vacuum", 0.25, -1, 0.36, 700), ("fake_breakout", 0.38, 1, 0.55, 420)],
    },
    "whale_pump_and_dump": {
        "trend_bias": 0.00002,
        "volatility": 0.00030,
        "liquidity": 0.70,
        "risk_aversion": 0.62,
        "shock_direction": 0.4,
        "demand_bias": {"retail": 0.10, "momentum": 0.10},
        "events": [("whale_accumulation", 0.20, 1, 0.65, 800), ("whale_distribution", 0.55, -1, 0.75, 900)],
    },
    "macro_risk_off": {
        "trend_bias": -0.000030,
        "volatility": 0.00024,
        "liquidity": 0.85,
        "risk_aversion": 0.78,
        "shock_direction": -0.25,
        "demand_bias": {"macro_risk": -0.20, "panic": -0.08, "institutional": -0.08},
        "events": [("negative_news", 0.22, -1, 0.38, 1100), ("panic_selloff", 0.52, -1, 0.32, 700)],
    },
    "macro_risk_on": {
        "trend_bias": 0.000030,
        "volatility": 0.00020,
        "liquidity": 1.00,
        "risk_aversion": 0.38,
        "shock_direction": 0.25,
        "demand_bias": {"macro_risk": 0.16, "institutional": 0.09, "retail": 0.06},
        "events": [("positive_news", 0.22, 1, 0.38, 1100), ("short_squeeze", 0.52, 1, 0.32, 700)],
    },
}


def iso_time(timestamp_ms):
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def round_price(price):
    return round(max(PRICE_TICK, price) / PRICE_TICK) * PRICE_TICK


def safe_ratio(numerator, denominator):
    return numerator / denominator if abs(denominator) > 1e-12 else 0.0


def finite_float(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def sample_range(rng, values, default):
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return float(default)
    low = finite_float(values[0], default)
    high = finite_float(values[1], default)
    if high < low:
        low, high = high, low
    return rng.uniform(low, high)


def weighted_choice(rng, weights, default):
    if not isinstance(weights, dict) or not weights:
        return default
    total = sum(max(0.0, finite_float(value, 0.0)) for value in weights.values())
    if total <= 0:
        return default
    draw = rng.random() * total
    cumulative = 0.0
    for key, value in weights.items():
        cumulative += max(0.0, finite_float(value, 0.0))
        if draw <= cumulative:
            return key
    return default


def poisson_sample(rng, expected_count):
    expected_count = max(0.0, float(expected_count))
    if expected_count <= 0:
        return 0
    if expected_count > 25:
        return max(0, int(rng.gauss(expected_count, math.sqrt(expected_count))))
    threshold = math.exp(-expected_count)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return max(0, count - 1)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return f"{float(value):.10g}"


class LimitOrderBook:
    def __init__(self, mid_price, rng):
        self.rng = rng
        self.bids = defaultdict(float)
        self.asks = defaultdict(float)
        self.last_trade_price = mid_price
        self.seed_book(mid_price, liquidity=1.0)

    def seed_book(self, mid_price, liquidity):
        center = round_price(mid_price)
        for level in range(1, LEVEL_COUNT + 1):
            decay = math.exp(-level / 36.0)
            bid_price = round_price(center - level * PRICE_TICK)
            ask_price = round_price(center + level * PRICE_TICK)
            noise_bid = self.rng.uniform(0.65, 1.35)
            noise_ask = self.rng.uniform(0.65, 1.35)
            size = BASE_LEVEL_SIZE * liquidity * decay
            self.bids[bid_price] += max(0.01, size * noise_bid)
            self.asks[ask_price] += max(0.01, size * noise_ask)
        self.clean()

    def clean(self):
        for book in [self.bids, self.asks]:
            empty = [price for price, size in book.items() if size <= 1e-9]
            for price in empty:
                del book[price]
        self.uncross()

    def uncross(self):
        # Synthetic liquidity providers can occasionally quote through each other
        # after a fast fair-value move. A real matching engine would immediately
        # match those resting orders, so we remove crossed top levels here.
        while self.bids and self.asks:
            bid = max(self.bids)
            ask = min(self.asks)
            if bid < ask:
                return
            matched = min(self.bids[bid], self.asks[ask])
            self.bids[bid] -= matched
            self.asks[ask] -= matched
            self.last_trade_price = (bid + ask) / 2.0
            if self.bids[bid] <= 1e-9:
                del self.bids[bid]
            if self.asks[ask] <= 1e-9:
                del self.asks[ask]

    def best_bid(self):
        return max(self.bids) if self.bids else round_price(self.last_trade_price - PRICE_TICK)

    def best_ask(self):
        return min(self.asks) if self.asks else round_price(self.last_trade_price + PRICE_TICK)

    def mid_price(self):
        return (self.best_bid() + self.best_ask()) / 2.0

    def place_limit(self, side, price, size):
        price = round_price(price)
        if size <= 0:
            return
        if side == "buy":
            if price >= self.best_ask():
                self.execute_market("buy", size)
            else:
                self.bids[price] += size
        else:
            if price <= self.best_bid():
                self.execute_market("sell", size)
            else:
                self.asks[price] += size
        self.clean()

    def cancel_random(self, side, probability, max_fraction):
        book = self.bids if side == "buy" else self.asks
        if not book:
            return
        for price in list(book.keys()):
            if self.rng.random() < probability:
                book[price] *= max(0.0, 1.0 - self.rng.uniform(0.05, max_fraction))
        self.clean()

    def execute_market(self, side, size):
        remaining = max(0.0, size)
        filled = 0.0
        notional = 0.0
        levels_touched = 0
        book = self.asks if side == "buy" else self.bids
        while remaining > 1e-9 and book:
            price = min(book) if side == "buy" else max(book)
            available = book[price]
            take = min(remaining, available)
            book[price] -= take
            remaining -= take
            filled += take
            notional += take * price
            levels_touched += 1
            self.last_trade_price = price
            if book[price] <= 1e-9:
                del book[price]
        self.clean()
        return {
            "filled": filled,
            "notional": notional,
            "average_price": notional / filled if filled > 0 else self.mid_price(),
            "levels_touched": levels_touched,
        }

    def replenish_around(self, fair_value, liquidity):
        mid = self.mid_price()
        anchor = 0.70 * mid + 0.30 * fair_value
        for level in range(1, LEVEL_COUNT + 1):
            decay = math.exp(-level / 42.0)
            size = BASE_LEVEL_SIZE * liquidity * decay * self.rng.uniform(0.15, 0.55)
            self.bids[round_price(anchor - level * PRICE_TICK)] += size
            self.asks[round_price(anchor + level * PRICE_TICK)] += size
        self.clean()

    def shift_prices(self, target_mid):
        current_mid = self.mid_price()
        if current_mid <= 0 or not math.isfinite(current_mid) or not math.isfinite(target_mid):
            return
        shift = target_mid - current_mid
        if abs(shift) < PRICE_TICK * 0.25:
            return
        new_bids = defaultdict(float)
        new_asks = defaultdict(float)
        for price, size in self.bids.items():
            new_bids[round_price(price + shift)] += size
        for price, size in self.asks.items():
            new_asks[round_price(price + shift)] += size
        self.bids = new_bids
        self.asks = new_asks
        self.last_trade_price = round_price(self.last_trade_price + shift)
        self.clean()

    def depth_band(self, side, bps):
        mid = self.mid_price()
        if side == "bid":
            floor = mid * (1.0 - bps / 10000.0)
            return sum(size for price, size in self.bids.items() if price >= floor)
        ceiling = mid * (1.0 + bps / 10000.0)
        return sum(size for price, size in self.asks.items() if price <= ceiling)

    def imbalance(self, bps):
        bid = self.depth_band("bid", bps)
        ask = self.depth_band("ask", bps)
        return safe_ratio(bid - ask, bid + ask)

    def large_wall(self, side, bps_limit=75):
        mid = self.mid_price()
        book = self.bids if side == "bid" else self.asks
        if not book:
            return 0.0, 0.0
        if side == "bid":
            candidates = [
                (price, size)
                for price, size in book.items()
                if 0 <= (mid - price) / mid * 10000.0 <= bps_limit
            ]
        else:
            candidates = [
                (price, size)
                for price, size in book.items()
                if 0 <= (price - mid) / mid * 10000.0 <= bps_limit
            ]
        if not candidates:
            return 0.0, 0.0
        price, size = max(candidates, key=lambda item: item[1])
        distance = abs(price - mid) / mid
        return distance, size


@dataclass
class SyntheticEvent:
    event_type: str
    direction: float
    magnitude: float
    start_second: int
    duration_seconds: int
    decay: float = 2.8

    def intensity_at(self, second):
        if second < self.start_second or second >= self.start_second + self.duration_seconds:
            return 0.0
        progress = (second - self.start_second) / max(1, self.duration_seconds)
        return self.magnitude * math.exp(-self.decay * progress)


DEMAND_KEYS = [
    "retail",
    "institutional",
    "momentum",
    "mean_reversion",
    "liquidity",
    "panic",
    "news",
    "macro_risk",
]


EVENT_DEMAND_MAP = {
    "positive_news": {"news": 1.0, "retail": 0.45, "momentum": 0.35, "liquidity": -0.10},
    "negative_news": {"news": 1.0, "panic": 0.55, "retail": 0.25, "liquidity": -0.12},
    "exchange_outage": {"panic": 0.70, "liquidity": -0.80, "news": 0.35},
    "liquidity_vacuum": {"liquidity": -1.00, "panic": 0.25, "momentum": 0.15},
    "whale_accumulation": {"institutional": 0.85, "liquidity": -0.18, "momentum": 0.20},
    "whale_distribution": {"institutional": 0.85, "liquidity": -0.18, "panic": 0.12},
    "fake_breakout": {"momentum": 0.75, "retail": 0.30, "liquidity": -0.16},
    "panic_selloff": {"panic": 1.0, "liquidity": -0.35, "momentum": 0.25},
    "short_squeeze": {"momentum": 0.70, "panic": -0.20, "liquidity": -0.22},
    "mean_reversion_snapback": {"mean_reversion": 0.90, "momentum": -0.30, "liquidity": 0.08},
}


class LatentState:
    def __init__(self, scenario, rng):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown SCENARIO={scenario}. Options: {', '.join(sorted(SCENARIOS))}")
        preset = SCENARIOS[scenario]
        self.scenario = scenario
        self.rng = rng
        self.preset = preset
        self.fair_value = START_PRICE
        self.trend_bias = preset["trend_bias"]
        self.volatility_regime = preset["volatility"]
        self.liquidity_regime = preset["liquidity"]
        self.news_shock = 0.0
        self.risk_aversion = preset["risk_aversion"]
        self.shock_direction = preset["shock_direction"]
        self.base_liquidity = preset["liquidity"]
        self.base_risk_aversion = preset["risk_aversion"]
        self.base_volatility = preset["volatility"]
        self.demands = {key: float(preset.get("demand_bias", {}).get(key, 0.0)) for key in DEMAND_KEYS}
        self.demand_persistence = {
            "retail": 0.985,
            "institutional": 0.998,
            "momentum": 0.965,
            "mean_reversion": 0.970,
            "liquidity": 0.992,
            "panic": 0.955,
            "news": 0.930,
            "macro_risk": 0.997,
        }
        self.events = [
            SyntheticEvent(
                event_type=event_type,
                start_second=int(SIM_SECONDS * start_fraction),
                direction=float(direction),
                magnitude=float(magnitude) * SIM_EVENT_MAGNITUDE_SCALE,
                duration_seconds=max(1, int(duration)),
            )
            for event_type, start_fraction, direction, magnitude, duration in preset.get("events", [])
        ]
        self.calibration = SIM_CALIBRATION
        self.calibration_regime = "none"
        self.waveform_points = []
        self.waveform_offset = 0.0
        self.waveform_scale = 0.0
        self.waveform_value = 0.0
        self.waveform_contribution = 0.0
        if self.calibration:
            self.apply_calibration()
        self.generated_events = list(self.events)
        self.active_events = []
        self.active_event_type = "none"
        self.active_event_direction = 0.0
        self.active_event_magnitude = 0.0
        self.event_counts = Counter()

    def simulator_parameters(self):
        return self.calibration.get("simulator_parameters", {}) if isinstance(self.calibration, dict) else {}

    def apply_calibration(self):
        params = self.simulator_parameters()
        self.calibration_regime = weighted_choice(
            self.rng,
            params.get("regime_probabilities", self.calibration.get("regime_probabilities", {})),
            "chop",
        )
        volatility_ranges = params.get("volatility_ranges", {})
        trend_ranges = params.get("trend_bias_ranges", {})
        liquidity_ranges = params.get("liquidity_ranges", {})
        demand_ranges = params.get("demand_intensity_ranges", {})

        if self.calibration_regime == "high_vol_chop":
            self.base_volatility = sample_range(
                self.rng,
                volatility_ranges.get("high_vol_per_second"),
                self.base_volatility,
            )
            self.base_liquidity = sample_range(self.rng, liquidity_ranges.get("thin"), self.base_liquidity)
            self.base_risk_aversion = max(self.base_risk_aversion, self.rng.uniform(0.55, 0.85))
            self.trend_bias = sample_range(self.rng, trend_ranges.get("chop_per_second"), 0.0)
        elif self.calibration_regime == "uptrend":
            self.base_volatility = sample_range(
                self.rng,
                volatility_ranges.get("overall_per_second"),
                self.base_volatility,
            )
            self.base_liquidity = sample_range(self.rng, liquidity_ranges.get("normal"), self.base_liquidity)
            self.trend_bias = abs(sample_range(self.rng, trend_ranges.get("uptrend_per_second"), abs(self.trend_bias)))
        elif self.calibration_regime == "downtrend":
            self.base_volatility = sample_range(
                self.rng,
                volatility_ranges.get("overall_per_second"),
                self.base_volatility,
            )
            self.base_liquidity = sample_range(self.rng, liquidity_ranges.get("normal"), self.base_liquidity)
            self.trend_bias = -abs(sample_range(self.rng, trend_ranges.get("downtrend_per_second"), abs(self.trend_bias)))
        else:
            self.base_volatility = sample_range(
                self.rng,
                volatility_ranges.get("overall_per_second"),
                self.base_volatility,
            )
            self.base_liquidity = sample_range(self.rng, liquidity_ranges.get("deep"), self.base_liquidity)
            self.trend_bias = sample_range(self.rng, trend_ranges.get("chop_per_second"), self.trend_bias * 0.25)

        self.volatility_regime = self.base_volatility
        self.liquidity_regime = self.base_liquidity
        self.risk_aversion = self.base_risk_aversion
        for key in DEMAND_KEYS:
            if key not in demand_ranges:
                continue
            sampled = sample_range(self.rng, demand_ranges[key], self.demands.get(key, 0.0))
            if self.calibration_regime == "uptrend" and key in {"momentum", "institutional", "retail"}:
                sampled = abs(sampled)
            elif self.calibration_regime == "downtrend" and key in {"momentum", "institutional", "retail", "macro_risk"}:
                sampled = -abs(sampled)
            elif self.calibration_regime == "chop" and key == "mean_reversion":
                sampled = abs(sampled)
            self.demands[key] = float(np.clip(sampled, -1.5, 1.5))

        self.append_calibrated_events(params.get("event_frequency_distributions", {}))
        waveform = self.calibration.get("fair_value_waveform", {}) if isinstance(self.calibration, dict) else {}
        points = waveform.get("normalized_points", [])
        if isinstance(points, list) and len(points) >= 4:
            self.waveform_points = [finite_float(value, 0.0) for value in points]
            self.waveform_offset = self.rng.uniform(0, len(self.waveform_points))
            self.waveform_scale = finite_float(waveform.get("wave_drift_scale_per_second"), 0.0)

    def append_calibrated_events(self, event_params):
        if not isinstance(event_params, dict):
            return
        sim_days = SIM_SECONDS / 86400.0
        added = 0
        for event_type, info in event_params.items():
            if not isinstance(info, dict) or "frequency_per_day" not in info:
                continue
            expected = finite_float(info.get("frequency_per_day"), 0.0) * sim_days
            count = min(6, poisson_sample(self.rng, expected))
            for _ in range(count):
                direction = int(finite_float(info.get("direction"), 0.0))
                if direction == 0:
                    direction = 1 if self.rng.random() < 0.5 else -1
                magnitude = sample_range(self.rng, info.get("magnitude_range"), 0.25) * SIM_EVENT_MAGNITUDE_SCALE
                duration = int(sample_range(self.rng, info.get("duration_seconds_range"), 600))
                latest_start = max(1, SIM_SECONDS - max(1, duration))
                start_second = self.rng.randint(0, latest_start)
                self.events.append(
                    SyntheticEvent(
                        event_type=event_type,
                        direction=direction,
                        magnitude=magnitude,
                        start_second=start_second,
                        duration_seconds=max(1, duration),
                    )
                )
                added += 1
        if added and self.events:
            self.events = sorted(self.events, key=lambda event: event.start_second)

    def waveform_at(self, second):
        if not self.waveform_points:
            return 0.0
        position = (second / max(1, SIM_SECONDS)) * len(self.waveform_points) + self.waveform_offset
        low = int(math.floor(position)) % len(self.waveform_points)
        high = (low + 1) % len(self.waveform_points)
        fraction = position - math.floor(position)
        return (1.0 - fraction) * self.waveform_points[low] + fraction * self.waveform_points[high]

    def demand(self, key):
        return float(self.demands.get(key, 0.0))

    def directional_demand(self):
        return (
            self.demand("retail") * 0.24
            + self.demand("institutional") * 0.30
            + self.demand("momentum") * 0.22
            - self.demand("mean_reversion") * 0.12
            + self.demand("news") * 0.32
            + self.demand("panic") * 0.26
            + self.demand("macro_risk") * 0.20
        )

    def update_event_demands(self, second):
        self.active_events = []
        event_contributions = {key: 0.0 for key in DEMAND_KEYS}
        active = []
        for event in self.events:
            intensity = event.intensity_at(second)
            if intensity <= 0:
                continue
            signed = event.direction * intensity
            active.append((event, signed))
            for demand_key, weight in EVENT_DEMAND_MAP.get(event.event_type, {}).items():
                if demand_key == "liquidity":
                    event_contributions[demand_key] += abs(intensity) * weight
                elif event.event_type in {"exchange_outage", "liquidity_vacuum"} and demand_key == "panic":
                    event_contributions[demand_key] += -abs(intensity) * weight
                else:
                    event_contributions[demand_key] += signed * weight
        if active:
            strongest, strongest_signed = max(active, key=lambda item: abs(item[1]))
            self.active_event_type = strongest.event_type
            self.active_event_direction = float(np.sign(strongest_signed))
            self.active_event_magnitude = float(abs(strongest_signed))
            self.event_counts[strongest.event_type] += 1
            self.active_events = [event for event, _ in active]
        else:
            self.active_event_type = "none"
            self.active_event_direction = 0.0
            self.active_event_magnitude = 0.0
        return event_contributions

    def update(self, second):
        progress = second / max(1, SIM_SECONDS)
        event_contributions = self.update_event_demands(second)
        for key in DEMAND_KEYS:
            base = float(self.preset.get("demand_bias", {}).get(key, 0.0))
            noise = self.rng.gauss(0.0, 0.0018)
            self.demands[key] = (
                self.demand_persistence[key] * self.demands[key]
                + (1.0 - self.demand_persistence[key]) * base
                + event_contributions.get(key, 0.0) * 0.055
                + noise
            )
            self.demands[key] = float(np.clip(self.demands[key], -1.5, 1.5))

        if self.scenario == "liquidity_crisis" and 0.35 < progress < 0.75:
            self.demands["liquidity"] = max(-1.2, self.demands["liquidity"] - 0.0025)
            self.demands["panic"] = min(1.2, self.demands["panic"] - 0.0015)
        if self.scenario == "fakeout_reversal":
            self.trend_bias = 0.000045 if progress < 0.45 else -0.000050

        directional = self.directional_demand()
        self.news_shock = self.demand("news") * 0.006
        self.risk_aversion = float(np.clip(self.base_risk_aversion + abs(self.demand("panic")) * 0.32 + max(-self.demand("liquidity"), 0) * 0.18, 0.05, 1.0))
        self.liquidity_regime = float(np.clip(self.base_liquidity * (1.0 + self.demand("liquidity") * 0.55 - self.risk_aversion * 0.18), 0.12, 2.2))
        self.volatility_regime = float(np.clip(self.base_volatility * (1.0 + abs(directional) * 2.2 + self.risk_aversion * 0.8), 0.000025, 0.0012))
        noise = self.rng.gauss(0.0, self.volatility_regime)
        self.waveform_value = 0.0
        self.waveform_contribution = 0.0
        if SIM_FAIR_VALUE_MODE == "historical_wave_calibrated" and self.waveform_points:
            self.waveform_value = self.waveform_at(second)
            previous_wave = self.waveform_at(max(0, second - 1))
            self.waveform_contribution = float(
                np.clip((self.waveform_value - previous_wave) * self.waveform_scale, -0.00008, 0.00008)
            )
        drift = self.trend_bias + directional * 0.00018
        drift += self.waveform_contribution
        second_return = float(np.clip(drift + noise, -SIM_MAX_ABS_SECOND_RETURN, SIM_MAX_ABS_SECOND_RETURN))
        self.fair_value = max(PRICE_TICK, self.fair_value * (1.0 + second_return))
        if SIM_MAX_CLOSE_TO_CLOSE_RETURN > 0:
            lower = START_PRICE * max(0.05, 1.0 - SIM_MAX_CLOSE_TO_CLOSE_RETURN)
            upper = START_PRICE * (1.0 + SIM_MAX_CLOSE_TO_CLOSE_RETURN)
            self.fair_value = float(np.clip(self.fair_value, lower, upper))


class BotSwarm:
    AGENT_TYPES = [
        "market_maker",
        "mean_reversion_liquidity_provider",
        "momentum_taker",
        "noise_trader",
        "casual_trader",
        "day_trader",
        "whale",
        "long_term_investor",
        "news_reactor",
    ]

    def __init__(self, book, state, rng):
        self.book = book
        self.state = state
        self.rng = rng
        self.recent_returns = deque(maxlen=90)
        self.recent_pressures = deque(maxlen=90)
        self.whale_queue = deque()
        self.liquidity_withdrawal_seconds = 0
        self.last_active_agent_types = []
        self.last_whale_pressure = 0.0
        self.last_agent_buy_intensity = 0.0
        self.last_agent_sell_intensity = 0.0

    def action(self, agent, side, size):
        size = max(0.0, float(size)) * SIM_PRICE_IMPACT_SCALE
        if size <= 1e-9:
            return None
        return {"agent": agent, "side": side, "size": size}

    def market_maker(self):
        mid = self.book.mid_price()
        fair = self.state.fair_value
        skew = safe_ratio(fair - mid, mid)
        inventory_skew = safe_ratio(-self.last_whale_pressure, BASE_LEVEL_SIZE * 20.0)
        risk_multiplier = max(0.15, 1.0 - self.state.risk_aversion * 0.55 - self.state.volatility_regime * 400.0)
        withdrawal_multiplier = 0.35 if self.liquidity_withdrawal_seconds > 0 else 1.0
        base_size = BASE_LEVEL_SIZE * self.state.liquidity_regime * risk_multiplier * withdrawal_multiplier
        for level in range(1, 7):
            size = base_size * math.exp(-level / 5.0) * self.rng.uniform(0.35, 0.95)
            buy_skew = 1.0 + max((skew + inventory_skew) * 70, 0)
            sell_skew = 1.0 + max((-skew - inventory_skew) * 70, 0)
            self.book.place_limit("buy", mid - level * PRICE_TICK, size * buy_skew)
            self.book.place_limit("sell", mid + level * PRICE_TICK, size * sell_skew)
        cancel_probability = 0.015 + 0.08 * self.state.risk_aversion + max(-self.state.demand("liquidity"), 0.0) * 0.04
        self.book.cancel_random("buy", cancel_probability, 0.55)
        self.book.cancel_random("sell", cancel_probability, 0.55)
        if self.liquidity_withdrawal_seconds > 0:
            self.liquidity_withdrawal_seconds -= 1

    def mean_reversion_liquidity_provider(self):
        mid = self.book.mid_price()
        deviation = safe_ratio(mid - self.state.fair_value, self.state.fair_value)
        size = BASE_LEVEL_SIZE * self.state.liquidity_regime * self.rng.uniform(0.2, 0.8)
        if deviation > 0:
            self.book.place_limit("sell", mid + self.rng.randint(2, 10) * PRICE_TICK, size * (1 + abs(deviation) * 200))
            self.book.place_limit("buy", mid - self.rng.randint(8, 18) * PRICE_TICK, size * 0.4)
        else:
            self.book.place_limit("buy", mid - self.rng.randint(2, 10) * PRICE_TICK, size * (1 + abs(deviation) * 200))
            self.book.place_limit("sell", mid + self.rng.randint(8, 18) * PRICE_TICK, size * 0.4)

    def momentum_taker(self):
        recent = sum(self.recent_returns) if self.recent_returns else 0.0
        signal = recent * 18.0 + self.state.demand("momentum") * 0.9 + self.state.demand("news") * 0.5 + self.state.trend_bias * 2000.0
        probability = min(0.85, 0.06 + abs(signal) * 0.05 + self.state.risk_aversion * 0.06)
        if self.rng.random() > probability:
            return None
        side = "buy" if signal >= 0 else "sell"
        size = BASE_LEVEL_SIZE * self.rng.lognormvariate(-0.2, 0.85) * (1.0 + self.state.risk_aversion)
        return self.action("momentum_taker", side, size)

    def noise_trader(self):
        if self.rng.random() < 0.35:
            side = "buy" if self.rng.random() < 0.5 else "sell"
            size = BASE_LEVEL_SIZE * self.rng.lognormvariate(-1.0, 0.75)
            return self.action("noise_trader", side, size)
        side = "buy" if self.rng.random() < 0.5 else "sell"
        mid = self.book.mid_price()
        offset = self.rng.randint(1, 25) * PRICE_TICK
        price = mid - offset if side == "buy" else mid + offset
        self.book.place_limit(side, price, BASE_LEVEL_SIZE * self.rng.uniform(0.05, 0.45))
        return None

    def casual_trader(self):
        sentiment = self.state.demand("retail") + 0.35 * self.state.demand("news")
        probability = min(0.45, 0.04 * SIM_RETAIL_INTENSITY + abs(sentiment) * 0.035)
        if self.rng.random() > probability:
            return None
        side_probability = 0.5 + np.clip(sentiment, -1.0, 1.0) * 0.22
        side = "buy" if self.rng.random() < side_probability else "sell"
        size = BASE_LEVEL_SIZE * self.rng.lognormvariate(-1.65, 0.55) * SIM_RETAIL_INTENSITY
        return self.action("casual_trader", side, size)

    def day_trader(self):
        recent_return = sum(list(self.recent_returns)[-12:]) if self.recent_returns else 0.0
        recent_pressure = sum(list(self.recent_pressures)[-12:]) / max(1, min(12, len(self.recent_pressures))) if self.recent_pressures else 0.0
        spread = safe_ratio(self.book.best_ask() - self.book.best_bid(), self.book.mid_price())
        signal = recent_return * 22.0 + recent_pressure * 0.45 + self.state.demand("momentum") * 0.55 - spread * 15.0 * np.sign(recent_return)
        probability = min(0.55, 0.05 + abs(signal) * 0.06 + self.state.volatility_regime * 120.0)
        if self.rng.random() > probability:
            return None
        side = "buy" if signal >= 0 else "sell"
        size = BASE_LEVEL_SIZE * self.rng.lognormvariate(-0.75, 0.65)
        return self.action("day_trader", side, size)

    def maybe_start_whale(self):
        event_bias = 0.0
        if self.state.active_event_type in {"whale_accumulation", "short_squeeze"}:
            event_bias = self.state.active_event_direction * self.state.active_event_magnitude
        elif self.state.active_event_type in {"whale_distribution", "panic_selloff"}:
            event_bias = self.state.active_event_direction * self.state.active_event_magnitude
        signal = self.state.demand("institutional") + event_bias
        probability = min(0.08, SIM_WHALE_INTENSITY * (0.0015 + abs(signal) * 0.018))
        if self.rng.random() > probability:
            return
        side = "buy" if signal >= 0 else "sell"
        parent_size = BASE_LEVEL_SIZE * self.rng.lognormvariate(1.55, 0.75) * SIM_WHALE_INTENSITY
        child_count = self.rng.randint(4, 18)
        child_size = parent_size / child_count
        for _ in range(child_count):
            self.whale_queue.append({"side": side, "size": child_size * self.rng.uniform(0.55, 1.45)})
        if self.rng.random() < 0.55:
            self.liquidity_withdrawal_seconds = max(self.liquidity_withdrawal_seconds, self.rng.randint(8, 45))

    def whale(self):
        if not self.whale_queue:
            self.maybe_start_whale()
        if not self.whale_queue:
            return None
        child = self.whale_queue.popleft()
        if self.rng.random() < 0.35 and child["size"] > BASE_LEVEL_SIZE * 0.4:
            # Iceberg behavior: a visible child order plus a delayed remainder.
            remainder = child["size"] * self.rng.uniform(0.25, 0.55)
            child["size"] -= remainder
            self.whale_queue.append({"side": child["side"], "size": remainder})
        return self.action("whale", child["side"], child["size"])

    def long_term_investor(self):
        gap = safe_ratio(self.state.fair_value - self.book.mid_price(), self.book.mid_price())
        signal = gap * 50.0 + self.state.demand("institutional") * 0.35 + self.state.demand("macro_risk") * 0.25
        probability = min(0.18, 0.010 * SIM_INSTITUTIONAL_INTENSITY + abs(signal) * 0.025)
        if self.rng.random() > probability:
            return None
        side = "buy" if signal >= 0 else "sell"
        if self.rng.random() < 0.65:
            mid = self.book.mid_price()
            offset = self.rng.randint(3, 20) * PRICE_TICK
            price = mid - offset if side == "buy" else mid + offset
            self.book.place_limit(side, price, BASE_LEVEL_SIZE * self.rng.uniform(0.4, 1.8) * SIM_INSTITUTIONAL_INTENSITY)
            return None
        size = BASE_LEVEL_SIZE * self.rng.lognormvariate(-0.2, 0.7) * SIM_INSTITUTIONAL_INTENSITY
        return self.action("long_term_investor", side, size)

    def news_reactor(self):
        if self.state.active_event_type == "none":
            return None
        signal = self.state.active_event_direction * self.state.active_event_magnitude + self.state.demand("news") * 0.65
        probability = min(0.75, abs(signal) * 0.22)
        if self.rng.random() > probability:
            return None
        # Occasionally fade an overreaction near the end of an event.
        side = "buy" if signal >= 0 else "sell"
        if self.state.active_event_type in {"fake_breakout", "mean_reversion_snapback"} and self.rng.random() < 0.35:
            side = "sell" if side == "buy" else "buy"
        size = BASE_LEVEL_SIZE * self.rng.lognormvariate(-0.15, 0.95) * (1.0 + abs(signal))
        return self.action("news_reactor", side, size)

    def step(self):
        before_mid = self.book.mid_price()
        self.market_maker()
        self.mean_reversion_liquidity_provider()
        fills = []
        actions = [self.momentum_taker(), self.noise_trader()]
        if SIM_ENABLE_RICH_AGENTS:
            actions.extend(
                [
                    self.casual_trader(),
                    self.day_trader(),
                    self.whale(),
                    self.long_term_investor(),
                    self.news_reactor(),
                ]
            )
        self.last_active_agent_types = ["market_maker", "mean_reversion_liquidity_provider"]
        for action in actions:
            if action is None:
                continue
            fill = self.book.execute_market(action["side"], action["size"])
            fill["side"] = action["side"]
            fill["agent"] = action["agent"]
            fills.append(fill)
            if fill["filled"] > 0:
                self.last_active_agent_types.append(action["agent"])
        self.book.replenish_around(self.state.fair_value, self.state.liquidity_regime)
        after_mid = self.book.mid_price()
        raw_return = safe_ratio(after_mid - before_mid, before_mid)
        if abs(raw_return) > SIM_MAX_ABS_SECOND_RETURN:
            capped_mid = before_mid * (1.0 + np.sign(raw_return) * SIM_MAX_ABS_SECOND_RETURN)
            self.book.shift_prices(capped_mid)
            after_mid = self.book.mid_price()
            raw_return = safe_ratio(after_mid - before_mid, before_mid)
        buy_volume = sum(fill["filled"] for fill in fills if fill["side"] == "buy")
        sell_volume = sum(fill["filled"] for fill in fills if fill["side"] == "sell")
        total_volume = buy_volume + sell_volume
        self.recent_returns.append(raw_return)
        self.recent_pressures.append(safe_ratio(buy_volume - sell_volume, total_volume))
        self.last_whale_pressure = sum(
            fill["filled"] * (1 if fill["side"] == "buy" else -1)
            for fill in fills
            if fill.get("agent") == "whale"
        )
        self.last_agent_buy_intensity = buy_volume
        self.last_agent_sell_intensity = sell_volume
        return fills


def snapshot_from_book(timestamp_ms, book, state, fills, previous_snapshot):
    best_bid = book.best_bid()
    best_ask = book.best_ask()
    mid = book.mid_price()
    buy_volume = sum(fill["filled"] for fill in fills if fill["side"] == "buy")
    sell_volume = sum(fill["filled"] for fill in fills if fill["side"] == "sell")
    trade_count = len([fill for fill in fills if fill["filled"] > 0])
    total_volume = buy_volume + sell_volume
    bid_10 = book.depth_band("bid", 10)
    ask_10 = book.depth_band("ask", 10)
    bid_25 = book.depth_band("bid", 25)
    ask_25 = book.depth_band("ask", 25)
    imbalance_10 = safe_ratio(bid_10 - ask_10, bid_10 + ask_10)
    imbalance_25 = safe_ratio(bid_25 - ask_25, bid_25 + ask_25)
    prev_bid_10 = previous_snapshot.get("bid_depth_10bps", bid_10) if previous_snapshot else bid_10
    prev_ask_10 = previous_snapshot.get("ask_depth_10bps", ask_10) if previous_snapshot else ask_10
    prev_imbalance = previous_snapshot.get("order_book_imbalance_10bps", imbalance_10) if previous_snapshot else imbalance_10
    large_bid_distance, large_bid_size = book.large_wall("bid")
    large_ask_distance, large_ask_size = book.large_wall("ask")
    whale_pressure = sum(fill["filled"] * (1 if fill.get("side") == "buy" else -1) for fill in fills if fill.get("agent") == "whale")
    active_agent_types = sorted({fill.get("agent", "") for fill in fills if fill.get("agent")})
    return {
        "venue": "simulated",
        "source_quality": "simulated_clob",
        "timestamp": timestamp_ms,
        "time": iso_time(timestamp_ms),
        "mid_price": mid,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_percent": safe_ratio(best_ask - best_bid, mid),
        "market_buy_volume_10s": buy_volume,
        "market_sell_volume_10s": sell_volume,
        "total_trade_volume_10s": total_volume,
        "trade_count_10s": trade_count,
        "market_pressure_10s": safe_ratio(buy_volume - sell_volume, total_volume),
        "bid_depth_10bps": bid_10,
        "ask_depth_10bps": ask_10,
        "bid_depth_25bps": bid_25,
        "ask_depth_25bps": ask_25,
        "order_book_imbalance_10bps": imbalance_10,
        "order_book_imbalance_25bps": imbalance_25,
        "bid_depth_change_10bps": safe_ratio(bid_10 - prev_bid_10, prev_bid_10),
        "ask_depth_change_10bps": safe_ratio(ask_10 - prev_ask_10, prev_ask_10),
        "imbalance_change_10bps": imbalance_10 - prev_imbalance,
        "large_bid_wall_distance": large_bid_distance,
        "large_ask_wall_distance": large_ask_distance,
        "large_bid_wall_size": large_bid_size,
        "large_ask_wall_size": large_ask_size,
        "hidden_fair_value": state.fair_value,
        "hidden_trend_bias": state.trend_bias,
        "hidden_volatility_regime": state.volatility_regime,
        "hidden_liquidity_regime": state.liquidity_regime,
        "hidden_news_shock": state.news_shock,
        "hidden_risk_aversion": state.risk_aversion,
        "hidden_scenario": state.scenario,
        "hidden_retail_demand": state.demand("retail"),
        "hidden_institutional_demand": state.demand("institutional"),
        "hidden_momentum_demand": state.demand("momentum"),
        "hidden_mean_reversion_demand": state.demand("mean_reversion"),
        "hidden_liquidity_demand": state.demand("liquidity"),
        "hidden_panic_demand": state.demand("panic"),
        "hidden_news_demand": state.demand("news"),
        "hidden_macro_risk_demand": state.demand("macro_risk"),
        "hidden_active_event_type": state.active_event_type,
        "hidden_active_event_direction": state.active_event_direction,
        "hidden_active_event_magnitude": state.active_event_magnitude,
        "hidden_whale_pressure": whale_pressure,
        "hidden_agent_buy_intensity": buy_volume,
        "hidden_agent_sell_intensity": sell_volume,
        "hidden_active_agent_types": ",".join(active_agent_types) if active_agent_types else "none",
        "hidden_calibration_regime": state.calibration_regime,
        "hidden_fair_value_waveform_value": state.waveform_value,
        "hidden_fair_value_waveform_contribution": state.waveform_contribution,
    }


def summary_stats(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(values[-1]),
        float(values[-1] - values[0]),
    )


def one_minute_row(snapshot_rows, previous_close):
    open_price = float(snapshot_rows[0]["mid_price"])
    close_price = float(snapshot_rows[-1]["mid_price"])
    high_price = max(float(row["mid_price"]) for row in snapshot_rows)
    low_price = min(float(row["mid_price"]) for row in snapshot_rows)
    volume = sum(float(row["total_trade_volume_10s"]) for row in snapshot_rows)
    buy_volume = sum(float(row["market_buy_volume_10s"]) for row in snapshot_rows)
    sell_volume = sum(float(row["market_sell_volume_10s"]) for row in snapshot_rows)
    trade_count = sum(int(row["trade_count_10s"]) for row in snapshot_rows)
    last = snapshot_rows[-1]
    candle_range = max(high_price - low_price, 1e-12)
    row = {
        "venue": "simulated",
        "source_quality": "simulated_clob",
        "timestamp": int(last["timestamp"]),
        "time": last["time"],
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
        "quote_volume": volume * close_price,
        "trade_count": trade_count,
        "taker_buy_volume": buy_volume,
        "taker_sell_volume": sell_volume,
        "taker_buy_ratio": safe_ratio(buy_volume, volume),
        "return_1": safe_ratio(close_price - previous_close, previous_close) if previous_close else 0.0,
        "range_percent": safe_ratio(high_price - low_price, open_price),
        "upper_wick_percent": safe_ratio(high_price - max(open_price, close_price), open_price),
        "lower_wick_percent": safe_ratio(min(open_price, close_price) - low_price, open_price),
        "close_position_in_range": safe_ratio(close_price - low_price, candle_range),
        "best_bid": last["best_bid"],
        "best_ask": last["best_ask"],
        "spread_percent": last["spread_percent"],
        "mid_price": last["mid_price"],
        "bid_depth_10bps": last["bid_depth_10bps"],
        "ask_depth_10bps": last["ask_depth_10bps"],
        "bid_depth_25bps": last["bid_depth_25bps"],
        "ask_depth_25bps": last["ask_depth_25bps"],
        "order_book_imbalance_10bps": last["order_book_imbalance_10bps"],
        "order_book_imbalance_25bps": last["order_book_imbalance_25bps"],
        "bid_depth_change_10bps": last["bid_depth_change_10bps"],
        "ask_depth_change_10bps": last["ask_depth_change_10bps"],
        "large_bid_wall_distance": last["large_bid_wall_distance"],
        "large_ask_wall_distance": last["large_ask_wall_distance"],
        "large_bid_wall_size": last["large_bid_wall_size"],
        "large_ask_wall_size": last["large_ask_wall_size"],
        "hidden_fair_value": last["hidden_fair_value"],
        "hidden_trend_bias": last["hidden_trend_bias"],
        "hidden_volatility_regime": last["hidden_volatility_regime"],
        "hidden_liquidity_regime": last["hidden_liquidity_regime"],
        "hidden_news_shock": last["hidden_news_shock"],
        "hidden_risk_aversion": last["hidden_risk_aversion"],
        "hidden_scenario": last["hidden_scenario"],
        "hidden_retail_demand": last["hidden_retail_demand"],
        "hidden_institutional_demand": last["hidden_institutional_demand"],
        "hidden_momentum_demand": last["hidden_momentum_demand"],
        "hidden_mean_reversion_demand": last["hidden_mean_reversion_demand"],
        "hidden_liquidity_demand": last["hidden_liquidity_demand"],
        "hidden_panic_demand": last["hidden_panic_demand"],
        "hidden_news_demand": last["hidden_news_demand"],
        "hidden_macro_risk_demand": last["hidden_macro_risk_demand"],
        "hidden_active_event_type": last["hidden_active_event_type"],
        "hidden_active_event_direction": last["hidden_active_event_direction"],
        "hidden_active_event_magnitude": last["hidden_active_event_magnitude"],
        "hidden_whale_pressure": sum(float(row.get("hidden_whale_pressure", 0.0)) for row in snapshot_rows),
        "hidden_agent_buy_intensity": buy_volume,
        "hidden_agent_sell_intensity": sell_volume,
        "hidden_active_agent_types": ",".join(
            sorted(
                {
                    agent
                    for snapshot in snapshot_rows
                    for agent in str(snapshot.get("hidden_active_agent_types", "none")).split(",")
                    if agent and agent != "none"
                }
            )
        )
        or "none",
        "hidden_calibration_regime": last["hidden_calibration_regime"],
        "hidden_fair_value_waveform_value": last["hidden_fair_value_waveform_value"],
        "hidden_fair_value_waveform_contribution": sum(
            float(row.get("hidden_fair_value_waveform_contribution", 0.0)) for row in snapshot_rows
        ),
    }
    for key, prefix in SNAPSHOT_SUMMARY_METRICS:
        mean, min_value, max_value, last_value, change = summary_stats(snapshot_rows, key)
        row[f"{prefix}_mean"] = mean
        row[f"{prefix}_min"] = min_value
        row[f"{prefix}_max"] = max_value
        row[f"{prefix}_last"] = last_value
        row[f"{prefix}_change"] = change
    return row


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt(row.get(column, "")) for column in columns})


def apply_book_price_caps(book, minute_open_mid):
    """Keep synthetic prices inside configured pretraining realism bounds."""
    mid = book.mid_price()
    if SIM_MAX_ABS_MINUTE_RETURN > 0 and minute_open_mid and minute_open_mid > 0:
        minute_return = safe_ratio(mid - minute_open_mid, minute_open_mid)
        if abs(minute_return) > SIM_MAX_ABS_MINUTE_RETURN:
            capped_mid = minute_open_mid * (1.0 + np.sign(minute_return) * SIM_MAX_ABS_MINUTE_RETURN)
            book.shift_prices(capped_mid)
            mid = book.mid_price()

    if SIM_MAX_CLOSE_TO_CLOSE_RETURN > 0 and START_PRICE > 0:
        full_return = safe_ratio(mid - START_PRICE, START_PRICE)
        if abs(full_return) > SIM_MAX_CLOSE_TO_CLOSE_RETURN:
            capped_mid = START_PRICE * (1.0 + np.sign(full_return) * SIM_MAX_CLOSE_TO_CLOSE_RETURN)
            book.shift_prices(capped_mid)


def run_simulation():
    rng = random.Random(RANDOM_SEED)
    state = LatentState(SCENARIO, rng)
    book = LimitOrderBook(START_PRICE, rng)
    bots = BotSwarm(book, state, rng)
    snapshots = []
    one_minute_rows = []
    minute_bucket = []
    previous_snapshot = None
    previous_close = None
    minute_open_mid = book.mid_price()

    for second in range(SIM_SECONDS):
        if second % 60 == 0:
            minute_open_mid = book.mid_price()
        state.update(second)
        fills = bots.step()
        apply_book_price_caps(book, minute_open_mid)
        timestamp_ms = START_TIMESTAMP_MS + second * 1000
        snapshot = snapshot_from_book(timestamp_ms, book, state, fills, previous_snapshot)
        snapshots.append(snapshot)
        minute_bucket.append(snapshot)
        previous_snapshot = snapshot

        if len(minute_bucket) == 60:
            row = one_minute_row(minute_bucket, previous_close)
            one_minute_rows.append(row)
            previous_close = float(row["close"])
            minute_bucket = []

    write_csv(SNAPSHOT_PATH, SNAPSHOT_COLUMNS, snapshots)
    write_csv(ONE_MINUTE_PATH, ONE_MINUTE_COLUMNS, one_minute_rows)
    return snapshots, one_minute_rows, state


def quantile_distance(values, historical_quantiles):
    if not historical_quantiles:
        return None
    values = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return None
    diffs = []
    for key, historical_value in historical_quantiles.items():
        try:
            prob = float(key)
            diffs.append(abs(float(values.quantile(prob)) - finite_float(historical_value)))
        except Exception:
            continue
    return float(np.mean(diffs)) if diffs else None


def max_drawdown_and_runup(close_values):
    closes = np.asarray(close_values, dtype=np.float64)
    if len(closes) == 0:
        return 0.0, 0.0, np.asarray([]), np.asarray([])
    running_max = np.maximum.accumulate(closes)
    running_min = np.minimum.accumulate(closes)
    drawdown = closes / np.maximum(running_max, 1e-12) - 1.0
    runup = closes / np.maximum(running_min, 1e-12) - 1.0
    return float(np.min(drawdown)), float(np.max(runup)), drawdown, runup


def trend_chop_duration_summary(close_values):
    close = np.asarray(close_values, dtype=np.float64)
    if len(close) < 4:
        return {"mean_seconds": 0.0, "segments": 0}
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    window = min(30, max(3, len(close) // 10))
    rolling_return = pd.Series(np.log(close)).diff(window).fillna(0.0)
    rolling_vol = pd.Series(returns).rolling(window, min_periods=2).std().fillna(0.0)
    threshold = np.maximum(rolling_vol * math.sqrt(window) * 0.75, 0.0015)
    labels = np.where(rolling_return > threshold, "trend", np.where(rolling_return < -threshold, "trend", "chop"))
    lengths = []
    previous = None
    count = 0
    for label in labels:
        if label == previous:
            count += 1
        else:
            if previous is not None:
                lengths.append(count)
            previous = label
            count = 1
    if count:
        lengths.append(count)
    return {
        "mean_seconds": float(np.mean(lengths) * 60.0) if lengths else 0.0,
        "segments": int(len(lengths)),
    }


def print_calibration_comparison(frame, minute_frame, state):
    calibration = state.calibration
    if not calibration or len(minute_frame) < 3:
        return
    historical = calibration.get("historical_distribution", {})
    closes = pd.to_numeric(minute_frame["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(closes) < 3:
        return
    synthetic_returns = np.diff(np.log(closes.to_numpy(dtype=np.float64)), prepend=np.log(closes.iloc[0]))
    synthetic_abs_returns = np.abs(synthetic_returns)
    synthetic_vol = pd.Series(synthetic_returns).rolling(12, min_periods=2).std().dropna()
    drawdown, runup, _, _ = max_drawdown_and_runup(closes)
    shock_threshold = finite_float(historical.get("large_shock_threshold"), float(np.std(synthetic_returns) * 3.0))
    shock_frequency = float((synthetic_abs_returns >= shock_threshold).sum() / max(len(minute_frame) / 1440.0, 1e-9))
    return_distance = quantile_distance(synthetic_returns, historical.get("log_return_quantiles", {}))
    abs_return_distance = quantile_distance(synthetic_abs_returns, historical.get("abs_log_return_quantiles", {}))
    vol_distance = quantile_distance(synthetic_vol, historical.get("rolling_volatility_quantiles", {}))
    duration_summary = trend_chop_duration_summary(closes)
    print("Historical calibration comparison")
    print(f"- calibration path: {SIM_CALIBRATION_PATH}")
    print(f"- calibration regime sampled: {state.calibration_regime}")
    print(f"- fair value mode: {SIM_FAIR_VALUE_MODE}")
    if return_distance is not None:
        print(f"- return distribution distance: {return_distance:.8g}")
    if abs_return_distance is not None:
        print(f"- abs-return distribution distance: {abs_return_distance:.8g}")
    if vol_distance is not None:
        print(f"- rolling volatility distribution distance: {vol_distance:.8g}")
    print(
        "- synthetic drawdown/runup vs historical range: "
        f"{drawdown:.2%}/{runup:.2%} vs "
        f"{finite_float(historical.get('drawdown_min')):.2%}/{finite_float(historical.get('runup_max')):.2%}"
    )
    print(
        "- shock frequency/day synthetic vs historical: "
        f"{shock_frequency:.4f} vs {finite_float(historical.get('large_shock_frequency_per_day')):.4f}"
    )
    historical_durations = calibration.get("regime_duration_distributions", {})
    historical_mean_duration = np.mean(
        [
            finite_float(info.get("mean_seconds"))
            for info in historical_durations.values()
            if isinstance(info, dict)
        ]
        or [0.0]
    )
    print(
        "- trend/chop mean duration seconds synthetic vs historical: "
        f"{duration_summary['mean_seconds']:.2f} vs {historical_mean_duration:.2f}"
    )
    spread_depth = calibration.get("spread_depth_regimes", {})
    if spread_depth and len(frame):
        if "spread_percent" in spread_depth:
            historical_spread = finite_float(spread_depth["spread_percent"].get("mean"))
            print(
                "- average spread synthetic vs historical: "
                f"{pd.to_numeric(frame['spread_percent'], errors='coerce').mean():.6g} vs {historical_spread:.6g}"
            )
        for column in ["bid_depth_10bps", "ask_depth_10bps"]:
            if column in spread_depth and column in frame.columns:
                historical_depth = finite_float(spread_depth[column].get("mean"))
                synthetic_depth = pd.to_numeric(frame[column], errors="coerce").mean()
                print(f"- average {column} synthetic vs historical: {synthetic_depth:.6g} vs {historical_depth:.6g}")


def diagnostics(snapshots, one_minute_rows, state):
    frame = pd.DataFrame(snapshots)
    minute_frame = pd.DataFrame(one_minute_rows)
    print("Synthetic market microstructure simulator")
    print(f"SYMBOL: {SYMBOL}")
    print(f"SCENARIO: {SCENARIO}")
    print(f"SIM_SECONDS: {SIM_SECONDS}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")
    print(f"SIM_CALIBRATION_PATH: {SIM_CALIBRATION_PATH if SIM_CALIBRATION else 'not loaded'}")
    print(f"SIM_FAIR_VALUE_MODE: {SIM_FAIR_VALUE_MODE}")
    print(f"Active agent types: {', '.join(BotSwarm.AGENT_TYPES if SIM_ENABLE_RICH_AGENTS else BotSwarm.AGENT_TYPES[:4])}")
    print(f"Generated events: {len(state.generated_events)}")
    event_type_counts = Counter(event.event_type for event in state.generated_events)
    print(f"Event type counts: {dict(event_type_counts)}")
    print(f"Snapshot output: {SNAPSHOT_PATH}")
    print(f"1m output: {ONE_MINUTE_PATH}")
    print(f"Snapshot rows: {len(frame)}")
    print(f"1m rows: {len(minute_frame)}")
    if len(frame):
        print(f"First timestamp: {int(frame['timestamp'].iloc[0])} {frame['time'].iloc[0]}")
        print(f"Last timestamp: {int(frame['timestamp'].iloc[-1])} {frame['time'].iloc[-1]}")
        print(f"Mid price min/mean/max: {frame['mid_price'].min():.6g} / {frame['mid_price'].mean():.6g} / {frame['mid_price'].max():.6g}")
        print(f"Average spread: {frame['spread_percent'].mean():.4%}")
        print(f"Average total flow per second: {frame['total_trade_volume_10s'].mean():.6g}")
        print(f"Average trade count per second: {frame['trade_count_10s'].mean():.3f}")
        print(f"Average 10bps imbalance: {frame['order_book_imbalance_10bps'].mean():.4f}")
        print(f"Average bid/ask depth 10bps: {frame['bid_depth_10bps'].mean():.6g} / {frame['ask_depth_10bps'].mean():.6g}")
        whale_active_pct = (pd.to_numeric(frame["hidden_whale_pressure"], errors="coerce").fillna(0.0).abs() > 0).mean()
        print(f"Seconds with whale activity: {whale_active_pct:.2%}")
        agent_counts = Counter()
        for value in frame["hidden_active_agent_types"].fillna("none"):
            for agent in str(value).split(","):
                if agent and agent != "none":
                    agent_counts[agent] += 1
        print(f"Active agent observation counts: {dict(agent_counts)}")
        active_event_counts = frame["hidden_active_event_type"].fillna("none").value_counts().to_dict()
        print(f"Active event observation counts: {active_event_counts}")
        for column in [
            "hidden_retail_demand",
            "hidden_institutional_demand",
            "hidden_momentum_demand",
            "hidden_mean_reversion_demand",
            "hidden_liquidity_demand",
            "hidden_panic_demand",
            "hidden_news_demand",
            "hidden_macro_risk_demand",
        ]:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(values):
                print(f"{column} min/mean/max: {values.min():+.4f} / {values.mean():+.4f} / {values.max():+.4f}")
        for column in [
            "hidden_fair_value_waveform_value",
            "hidden_fair_value_waveform_contribution",
        ]:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(values):
                print(f"{column} min/mean/max: {values.min():+.6g} / {values.mean():+.6g} / {values.max():+.6g}")
    if len(minute_frame):
        final_return = minute_frame["close"].iloc[-1] / minute_frame["open"].iloc[0] - 1.0
        closes = pd.to_numeric(minute_frame["close"], errors="coerce").to_numpy(dtype=np.float64)
        running_max = np.maximum.accumulate(closes)
        running_min = np.minimum.accumulate(closes)
        max_drawdown = float(np.min(closes / np.maximum(running_max, 1e-12) - 1.0))
        max_runup = float(np.max(closes / np.maximum(running_min, 1e-12) - 1.0))
        print(f"Simulated close-to-close return: {final_return:.2%}")
        print(f"Max drawdown: {max_drawdown:.2%}")
        print(f"Max runup: {max_runup:.2%}")
        minute_returns = pd.to_numeric(minute_frame["return_1"], errors="coerce").abs()
        if abs(final_return) > SIM_MAX_CLOSE_TO_CLOSE_RETURN * 1.05:
            print("WARNING: close-to-close return exceeded configured pretraining cap.")
        if len(minute_returns) and minute_returns.max() > SIM_MAX_ABS_MINUTE_RETURN:
            print("WARNING: at least one 1m return is larger than SIM_MAX_ABS_MINUTE_RETURN.")
    print_calibration_comparison(frame, minute_frame, state)
    print("Hidden diagnostic columns are prefixed with hidden_ and are ignored by model feature selectors.")
    print("No trades/orders/private API behavior. Simulation only.")


def main():
    snapshots, one_minute_rows, state = run_simulation()
    diagnostics(snapshots, one_minute_rows, state)


if __name__ == "__main__":
    main()
