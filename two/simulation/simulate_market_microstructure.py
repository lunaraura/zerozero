import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[3]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

from tiny.core import tiny_io, tiny_paths


PROJECT_ROOT = tiny_paths.ROOT
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
SOURCE_KIND = "synthetic"
SCENARIO = os.getenv("SCENARIO", "calm_chop").strip().lower()
SIM_MIXED_SCENARIOS = [
    item.strip().lower()
    for item in os.getenv(
        "SIM_MIXED_SCENARIOS",
        "calm_chop,high_vol_chop,bullish_breakout,bearish_breakdown,"
        "fakeout_reversal,news_shock_up,news_shock_down,liquidity_crisis,"
        "thin_book_fakeout,whale_pump_and_dump,macro_risk_on,macro_risk_off",
    ).split(",")
    if item.strip()
]
SIM_MIXED_MIN_SECONDS = int(os.getenv("SIM_MIXED_MIN_SECONDS", "300"))
SIM_MIXED_MAX_SECONDS = int(os.getenv("SIM_MIXED_MAX_SECONDS", "1800"))
SIM_MIXED_RUNTIME = SCENARIO == "mixed_runtime"
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
SIM_RETURN_INNOVATION_MODE = os.getenv("SIM_RETURN_INNOVATION_MODE", "student_t").strip().lower()
if SIM_RETURN_INNOVATION_MODE not in {"gaussian", "student_t", "mixture"}:
    print(f"Unknown SIM_RETURN_INNOVATION_MODE={SIM_RETURN_INNOVATION_MODE}; using student_t.")
    SIM_RETURN_INNOVATION_MODE = "student_t"
SIM_STUDENT_T_DF = max(2.1, float(os.getenv("SIM_STUDENT_T_DF", "4.0")))
SIM_JUMP_MIXTURE_PROBABILITY = float(os.getenv("SIM_JUMP_MIXTURE_PROBABILITY", "0.03"))
SIM_JUMP_MIXTURE_SCALE = float(os.getenv("SIM_JUMP_MIXTURE_SCALE", "4.0"))
SIM_VOL_PERSISTENCE = float(np.clip(float(os.getenv("SIM_VOL_PERSISTENCE", "0.95")), 0.0, 0.999))
SIM_USE_SOFT_RETURN_LIMITS = os.getenv("SIM_USE_SOFT_RETURN_LIMITS", "true").strip().lower() in {"1", "true", "yes", "y"}
SIM_SCENARIO_CAP_MULTIPLIER_ENV = os.getenv("SIM_SCENARIO_CAP_MULTIPLIER", "").strip()
SIM_SCENARIO_CAP_MULTIPLIER_OVERRIDE = (
    float(SIM_SCENARIO_CAP_MULTIPLIER_ENV) if SIM_SCENARIO_CAP_MULTIPLIER_ENV else None
)
SIM_EVENT_FREQUENCY_SCALE = float(os.getenv("SIM_EVENT_FREQUENCY_SCALE", "1.0"))
SIM_REGIME_PERSISTENCE_SCALE_ENV = os.getenv("SIM_REGIME_PERSISTENCE_SCALE", "").strip()
SIM_REGIME_PERSISTENCE_SCALE_OVERRIDE = (
    float(SIM_REGIME_PERSISTENCE_SCALE_ENV) if SIM_REGIME_PERSISTENCE_SCALE_ENV else None
)
SIM_REGIME_TARGET_DURATION_SECONDS_ENV = os.getenv("SIM_REGIME_TARGET_DURATION_SECONDS", "").strip()
SIM_REGIME_TARGET_DURATION_SECONDS_OVERRIDE = (
    float(SIM_REGIME_TARGET_DURATION_SECONDS_ENV) if SIM_REGIME_TARGET_DURATION_SECONDS_ENV else None
)
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
    "source_kind",
    "symbol",
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
    "hidden_runtime_scenario",
    "hidden_regime_segment_id",
    "hidden_regime_age_seconds",
    "hidden_regime_remaining_seconds",
    "hidden_regime_transition",
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
    "source_kind",
    "symbol",
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
        "hidden_runtime_scenario",
        "hidden_regime_segment_id",
        "hidden_regime_age_seconds",
        "hidden_regime_remaining_seconds",
        "hidden_regime_transition",
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
class MixedRuntimeScheduler:
    def __init__(self, rng, scenario_names, min_seconds, max_seconds):
        valid = [name for name in scenario_names if name in SCENARIOS and name != "mixed_runtime"]
        if not valid:
            valid = ["calm_chop"]

        self.rng = rng
        self.scenario_names = valid
        self.min_seconds = max(10, int(min_seconds))
        self.max_seconds = max(self.min_seconds, int(max_seconds))
        self.current_scenario = None
        self.segment_id = -1
        self.segment_start_second = 0
        self.segment_end_second = 0
        self.transition_now = False

    def choose_next(self, second):
        previous = self.current_scenario
        choices = [name for name in self.scenario_names if name != previous] or self.scenario_names

        self.current_scenario = self.rng.choice(choices)
        self.segment_id += 1
        self.segment_start_second = int(second)

        duration = self.rng.randint(self.min_seconds, self.max_seconds)
        self.segment_end_second = int(second + duration)
        self.transition_now = True

    def update(self, second):
        self.transition_now = False

        if self.current_scenario is None or second >= self.segment_end_second:
            self.choose_next(second)

        return {
            "scenario": self.current_scenario,
            "segment_id": self.segment_id,
            "age_seconds": int(second - self.segment_start_second),
            "remaining_seconds": int(max(0, self.segment_end_second - second)),
            "transition": bool(self.transition_now),
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


BASE_LEVEL_SIZE_RAW = BASE_LEVEL_SIZE
SIM_DEPTH_CALIBRATION_SCALE = 1.0
SIM_DEPTH_TARGET_10BPS = 0.0
SIM_DEPTH_HISTORICAL_10BPS = 0.0


def scenario_depth_target_range(scenario):
    scenario = str(scenario or "").lower()
    if scenario == "calm_chop":
        return 0.80, 1.30, 1.00
    if "high_vol_chop" in scenario:
        return 0.50, 1.00, 0.75
    if any(token in scenario for token in ["bullish_breakout", "bearish_breakdown", "fakeout_reversal"]):
        return 0.40, 0.90, 0.65
    if "liquidity_crisis" in scenario:
        return 0.10, 0.35, 0.22
    if any(token in scenario for token in ["news_shock", "whale", "thin_book"]):
        return 0.25, 0.75, 0.50
    return 0.60, 1.20, 0.90


def calibrated_base_level_size(raw_base_size):
    if not isinstance(SIM_CALIBRATION, dict):
        return raw_base_size, 1.0, 0.0, 0.0
    spread_depth = SIM_CALIBRATION.get("spread_depth_regimes", {})
    bid_info = spread_depth.get("bid_depth_10bps", {}) if isinstance(spread_depth, dict) else {}
    ask_info = spread_depth.get("ask_depth_10bps", {}) if isinstance(spread_depth, dict) else {}
    bid_mean = finite_float(bid_info.get("mean") if isinstance(bid_info, dict) else 0.0, 0.0)
    ask_mean = finite_float(ask_info.get("mean") if isinstance(ask_info, dict) else 0.0, 0.0)
    means = [value for value in [bid_mean, ask_mean] if value > 0]
    if not means:
        return raw_base_size, 1.0, 0.0, 0.0
    historical_target = float(np.mean(means))
    # Empirical depth proxy: with the default 120 base size, calm synthetic
    # books have tended to sit around ~27k units inside 10 bps. Use that
    # relationship to pull calibrated runs toward historical depth without
    # changing live/training code or requiring an expensive dry run.
    default_depth_proxy = max(raw_base_size * 225.0, 1e-9)
    _, _, scenario_multiplier = scenario_depth_target_range(SCENARIO)
    target_depth = historical_target * scenario_multiplier
    scale = float(np.clip(target_depth / default_depth_proxy, 0.05, 1.50))
    return raw_base_size * scale, scale, target_depth, historical_target


(
    BASE_LEVEL_SIZE,
    SIM_DEPTH_CALIBRATION_SCALE,
    SIM_DEPTH_TARGET_10BPS,
    SIM_DEPTH_HISTORICAL_10BPS,
) = calibrated_base_level_size(BASE_LEVEL_SIZE_RAW)


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


def scenario_cap_multiplier(scenario):
    if SIM_SCENARIO_CAP_MULTIPLIER_OVERRIDE is not None:
        return max(0.05, SIM_SCENARIO_CAP_MULTIPLIER_OVERRIDE)
    scenario = str(scenario or "").lower()
    if scenario == "calm_chop":
        return 0.75
    if any(token in scenario for token in ["liquidity_crisis", "thin_book", "whale"]):
        return 2.25
    if any(token in scenario for token in ["news", "fakeout", "high_vol"]):
        return 1.85
    if any(token in scenario for token in ["breakout", "breakdown", "macro"]):
        return 1.45
    if any(token in scenario for token in ["accumulation", "distribution"]):
        return 1.20
    return 1.0


def scenario_event_frequency_default(scenario):
    scenario = str(scenario or "").lower()
    if scenario == "calm_chop":
        return 0.10
    if "high_vol_chop" in scenario:
        return 0.35
    if any(token in scenario for token in ["bullish_breakout", "bearish_breakdown", "fakeout_reversal"]):
        return 0.50
    if "liquidity_crisis" in scenario:
        return 0.85
    if "whale_pump_and_dump" in scenario:
        return 1.00
    if "news_shock" in scenario:
        return 0.85
    return 0.60


def scenario_event_policy(scenario):
    scenario = str(scenario or "").lower()
    if scenario == "calm_chop":
        return {
            "allowed": {"fake_breakout", "mean_reversion_snapback"},
            "per_event_scale": {"fake_breakout": 0.25, "mean_reversion_snapback": 0.20},
            "magnitude_scale": 0.35,
            "duration_scale": 0.50,
        }
    if "high_vol_chop" in scenario:
        return {
            "allowed": {"positive_news", "negative_news", "fake_breakout", "liquidity_vacuum"},
            "per_event_scale": {"positive_news": 0.45, "negative_news": 0.45, "liquidity_vacuum": 0.35},
            "magnitude_scale": 0.65,
            "duration_scale": 0.75,
        }
    return {
        "allowed": None,
        "per_event_scale": {},
        "magnitude_scale": 1.0,
        "duration_scale": 1.0,
    }


def effective_event_frequency_scale(scenario):
    return max(0.0, SIM_EVENT_FREQUENCY_SCALE * scenario_event_frequency_default(scenario))


def scenario_directional_drift_scale(scenario):
    scenario = str(scenario or "").lower()
    if scenario == "calm_chop":
        return 0.000045
    if "high_vol_chop" in scenario:
        return 0.00010
    if any(token in scenario for token in ["liquidity_crisis", "news_shock", "whale"]):
        return 0.00016
    return 0.00014


def scenario_event_demand_scale(scenario):
    scenario = str(scenario or "").lower()
    if scenario == "calm_chop":
        return 0.018
    if "high_vol_chop" in scenario:
        return 0.032
    if any(token in scenario for token in ["bullish_breakout", "bearish_breakdown", "fakeout_reversal"]):
        return 0.042
    return 0.055


def scenario_trend_bias_scale(scenario):
    scenario = str(scenario or "").lower()
    if scenario == "calm_chop":
        return 0.25
    if "high_vol_chop" in scenario:
        return 0.50
    if "fakeout_reversal" in scenario:
        return 0.80
    return 1.0


def scenario_regime_persistence_scale(scenario, calibration=None):
    if SIM_REGIME_PERSISTENCE_SCALE_OVERRIDE is not None:
        return max(0.10, SIM_REGIME_PERSISTENCE_SCALE_OVERRIDE)
    scenario = str(scenario or "").lower()
    if scenario == "calm_chop":
        scale = 0.45
    elif "high_vol_chop" in scenario:
        scale = 0.55
    elif "fakeout_reversal" in scenario:
        scale = 0.65
    elif any(token in scenario for token in ["news_shock", "liquidity_crisis", "whale"]):
        scale = 0.75
    else:
        scale = 0.85
    if isinstance(calibration, dict):
        durations = calibration.get("regime_duration_distributions", {})
        means = [
            finite_float(info.get("mean_seconds"), 0.0)
            for info in durations.values()
            if isinstance(info, dict) and finite_float(info.get("mean_seconds"), 0.0) > 0
        ]
        if means:
            historical_mean = float(np.mean(means))
            scale = min(scale, float(np.clip(historical_mean / 600.0, 0.20, 1.0)))
    return max(0.10, scale)


def calibrated_regime_duration_seconds(calibration):
    if not isinstance(calibration, dict):
        return 0.0
    durations = calibration.get("regime_duration_distributions", {})
    candidates = []
    for info in durations.values():
        if not isinstance(info, dict):
            continue
        quantiles = info.get("quantiles_seconds", {})
        median = finite_float(quantiles.get("0.5") if isinstance(quantiles, dict) else None, 0.0)
        mean = finite_float(info.get("mean_seconds"), 0.0)
        if mean > 0:
            candidates.append(mean)
        elif median > 0:
            candidates.append(median)
    return float(np.median(candidates)) if candidates else 0.0


def scenario_regime_duration_multiplier(scenario):
    scenario = str(scenario or "").lower()
    if scenario == "calm_chop":
        return 1.35
    if "high_vol_chop" in scenario:
        return 1.00
    if any(token in scenario for token in ["bullish_breakout", "bearish_breakdown"]):
        return 3.00
    if "fakeout_reversal" in scenario:
        return 2.25
    if "liquidity_crisis" in scenario:
        return 3.50
    return 2.00


def scenario_regime_target_duration_seconds(scenario, calibration=None):
    if SIM_REGIME_TARGET_DURATION_SECONDS_OVERRIDE is not None:
        return max(10.0, SIM_REGIME_TARGET_DURATION_SECONDS_OVERRIDE)
    historical = calibrated_regime_duration_seconds(calibration)
    if historical <= 0:
        return 0.0
    target = historical * scenario_regime_duration_multiplier(scenario)
    return float(np.clip(target, 30.0, 900.0))


def apply_persistence_scale(persistence, scale):
    scale = max(0.10, finite_float(scale, 1.0))
    return float(np.clip(1.0 - (1.0 - persistence) / scale, 0.50, 0.9995))


def effective_return_caps(scenario):
    multiplier = scenario_cap_multiplier(scenario)
    return {
        "scenario_cap_multiplier": multiplier,
        "second": max(0.0, SIM_MAX_ABS_SECOND_RETURN * multiplier),
        "minute": max(0.0, SIM_MAX_ABS_MINUTE_RETURN * multiplier),
        "close_to_close": max(0.0, SIM_MAX_CLOSE_TO_CLOSE_RETURN * multiplier),
    }


def soft_limit_return(value, limit):
    """Compress extreme returns instead of deleting them with a hard cap."""
    value = finite_float(value, 0.0)
    limit = abs(finite_float(limit, 0.0))
    if limit <= 0.0:
        return value
    magnitude = abs(value)
    if magnitude <= limit:
        return value
    excess = magnitude - limit
    compressed = limit + math.log1p(excess / max(limit, 1e-12)) * limit * 0.35
    return math.copysign(compressed, value)


def bounded_return(value, limit, use_soft_limits=True):
    value = finite_float(value, 0.0)
    limit = abs(finite_float(limit, 0.0))
    if limit <= 0.0:
        return value
    if use_soft_limits:
        return soft_limit_return(value, limit)
    return float(np.clip(value, -limit, limit))


def absurdity_guard_price(price):
    price = finite_float(price, START_PRICE)
    if price <= 0:
        return PRICE_TICK
    lower = max(PRICE_TICK, START_PRICE * 0.02)
    upper = max(lower * 1.01, START_PRICE * 50.0)
    return float(np.clip(price, lower, upper))


def sample_laplace(rng, scale):
    scale = max(float(scale), 1e-12)
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return sign * rng.expovariate(1.0 / scale)


def sample_student_t_unit_variance(rng, df):
    df = max(2.1, float(df))
    z = rng.gauss(0.0, 1.0)
    chi_square = rng.gammavariate(df / 2.0, 2.0)
    t_value = z / math.sqrt(max(chi_square / df, 1e-12))
    variance_normalizer = math.sqrt((df - 2.0) / df)
    return t_value * variance_normalizer


def sample_return_innovation(rng, volatility_regime):
    volatility = max(0.0, finite_float(volatility_regime, 0.0))
    if volatility <= 0.0:
        return 0.0
    if SIM_RETURN_INNOVATION_MODE == "gaussian":
        return rng.gauss(0.0, volatility)
    if SIM_RETURN_INNOVATION_MODE == "student_t":
        return sample_student_t_unit_variance(rng, SIM_STUDENT_T_DF) * volatility
    if rng.random() < max(0.0, min(1.0, SIM_JUMP_MIXTURE_PROBABILITY)):
        jump_scale = volatility * max(1.0, SIM_JUMP_MIXTURE_SCALE)
        if rng.random() < 0.5:
            return rng.gauss(0.0, jump_scale)
        return sample_laplace(rng, jump_scale / math.sqrt(2.0))
    return rng.gauss(0.0, volatility)


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

    def replenish_around(self, fair_value, liquidity, depth_multiplier=1.0):
        mid = self.mid_price()
        anchor = 0.70 * mid + 0.30 * fair_value
        for level in range(1, LEVEL_COUNT + 1):
            decay = math.exp(-level / 42.0)
            size = BASE_LEVEL_SIZE * liquidity * depth_multiplier * decay * self.rng.uniform(0.15, 0.55)
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
        self.trend_bias = preset["trend_bias"] * scenario_trend_bias_scale(scenario)
        self.volatility_regime = preset["volatility"]
        self.liquidity_regime = preset["liquidity"]
        self.news_shock = 0.0
        self.risk_aversion = preset["risk_aversion"]
        self.shock_direction = preset["shock_direction"]
        self.base_liquidity = preset["liquidity"]
        self.base_risk_aversion = preset["risk_aversion"]
        self.base_volatility = preset["volatility"]
        self.return_caps = effective_return_caps(scenario)
        self.event_frequency_scale = effective_event_frequency_scale(scenario)
        self.event_policy = scenario_event_policy(scenario)
        self.event_demand_scale = scenario_event_demand_scale(scenario)
        self.directional_drift_scale = scenario_directional_drift_scale(scenario)
        self.regime_persistence_scale = scenario_regime_persistence_scale(scenario, SIM_CALIBRATION)
        self.regime_target_duration_seconds = scenario_regime_target_duration_seconds(scenario, SIM_CALIBRATION)
        self.next_regime_flip_second = 0
        self.latent_regime_label = "chop"
        self.book_depth_multiplier = 1.0
        self.depth_ema_10bps = 0.0
        self.depth_feedback_updates = 0
        self.depth_target_10bps = SIM_DEPTH_TARGET_10BPS
        self.depth_historical_10bps = SIM_DEPTH_HISTORICAL_10BPS
        depth_min_multiplier, depth_max_multiplier, _ = scenario_depth_target_range(scenario)
        self.depth_target_min_10bps = self.depth_historical_10bps * depth_min_multiplier
        self.depth_target_max_10bps = self.depth_historical_10bps * depth_max_multiplier
        self.demands = {key: float(preset.get("demand_bias", {}).get(key, 0.0)) for key in DEMAND_KEYS}
        base_demand_persistence = {
            "retail": 0.985,
            "institutional": 0.998,
            "momentum": 0.965,
            "mean_reversion": 0.970,
            "liquidity": 0.992,
            "panic": 0.955,
            "news": 0.930,
            "macro_risk": 0.997,
        }
        self.demand_persistence = {
            key: apply_persistence_scale(value, self.regime_persistence_scale)
            for key, value in base_demand_persistence.items()
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
        self.waveform_bands = []
        self.waveform_value = 0.0
        self.waveform_contribution = 0.0
        if self.calibration:
            self.apply_calibration()
        self.latent_regime_label = self.calibration_regime if self.calibration_regime != "none" else "chop"
        self.schedule_next_regime_flip(0)
        self.generated_events = list(self.events)
        self.active_events = []
        self.active_event_type = "none"
        self.active_event_direction = 0.0
        self.active_event_magnitude = 0.0
        self.event_counts = Counter()

    def simulator_parameters(self):
        return self.calibration.get("simulator_parameters", {}) if isinstance(self.calibration, dict) else {}

    def schedule_next_regime_flip(self, current_second=0):
        if self.regime_target_duration_seconds <= 0:
            self.next_regime_flip_second = 0
            return
        sampled = self.rng.expovariate(1.0 / max(self.regime_target_duration_seconds, 1.0))
        sampled = float(np.clip(sampled, self.regime_target_duration_seconds * 0.35, self.regime_target_duration_seconds * 2.50))
        self.next_regime_flip_second = int(current_second + max(15.0, sampled))

    def choose_next_latent_regime(self):
        params = self.simulator_parameters()
        probabilities = params.get("regime_probabilities", {}) if isinstance(params, dict) else {}
        next_label = weighted_choice(self.rng, probabilities, "chop")
        if next_label == self.latent_regime_label and self.rng.random() < 0.65:
            alternatives = [label for label in ["chop", "uptrend", "downtrend", "high_vol_chop"] if label != next_label]
            next_label = self.rng.choice(alternatives)
        return next_label

    def apply_latent_regime(self, label):
        params = self.simulator_parameters()
        trend_ranges = params.get("trend_bias_ranges", {}) if isinstance(params, dict) else {}
        volatility_ranges = params.get("volatility_ranges", {}) if isinstance(params, dict) else {}
        self.latent_regime_label = label
        if label == "uptrend":
            self.trend_bias = abs(sample_range(self.rng, trend_ranges.get("uptrend_per_second"), max(abs(self.trend_bias), 0.000012)))
            self.demands["momentum"] = abs(self.demands.get("momentum", 0.0)) + self.rng.uniform(0.015, 0.060)
            self.demands["mean_reversion"] *= 0.65
        elif label == "downtrend":
            self.trend_bias = -abs(sample_range(self.rng, trend_ranges.get("downtrend_per_second"), max(abs(self.trend_bias), 0.000012)))
            self.demands["momentum"] = -abs(self.demands.get("momentum", 0.0)) - self.rng.uniform(0.015, 0.060)
            self.demands["panic"] = max(self.demands.get("panic", 0.0), self.rng.uniform(0.010, 0.050))
        elif label == "high_vol_chop":
            self.trend_bias = sample_range(self.rng, trend_ranges.get("chop_per_second"), 0.0)
            self.base_volatility = sample_range(self.rng, volatility_ranges.get("high_vol_per_second"), self.base_volatility)
            self.demands["mean_reversion"] = abs(self.demands.get("mean_reversion", 0.0)) + self.rng.uniform(0.020, 0.070)
            self.demands["liquidity"] -= self.rng.uniform(0.020, 0.080)
        else:
            self.trend_bias = sample_range(self.rng, trend_ranges.get("chop_per_second"), 0.0)
            self.demands["mean_reversion"] = abs(self.demands.get("mean_reversion", 0.0)) + self.rng.uniform(0.015, 0.060)
            self.demands["momentum"] *= 0.55
        self.trend_bias *= scenario_trend_bias_scale(self.scenario)
        for key in DEMAND_KEYS:
            self.demands[key] = float(np.clip(self.demands.get(key, 0.0), -1.5, 1.5))

    def maybe_flip_latent_regime(self, second):
        if self.regime_target_duration_seconds <= 0:
            return
        if self.next_regime_flip_second <= 0:
            self.schedule_next_regime_flip(second)
            return
        if second < self.next_regime_flip_second:
            return
        self.apply_latent_regime(self.choose_next_latent_regime())
        self.schedule_next_regime_flip(second)


    def switch_runtime_scenario(self, scenario, second=0):
        if scenario not in SCENARIOS:
            return

        if scenario == self.scenario:
            return

        preset = SCENARIOS[scenario]
        self.scenario = scenario
        self.preset = preset

        self.base_liquidity = preset["liquidity"]
        self.base_risk_aversion = preset["risk_aversion"]
        self.base_volatility = preset["volatility"]

        self.trend_bias = preset["trend_bias"] * scenario_trend_bias_scale(scenario)
        self.volatility_regime = 0.70 * self.volatility_regime + 0.30 * preset["volatility"]
        self.liquidity_regime = 0.70 * self.liquidity_regime + 0.30 * preset["liquidity"]
        self.risk_aversion = 0.70 * self.risk_aversion + 0.30 * preset["risk_aversion"]
        self.shock_direction = preset["shock_direction"]

        self.return_caps = effective_return_caps(scenario)
        self.event_frequency_scale = effective_event_frequency_scale(scenario)
        self.event_policy = scenario_event_policy(scenario)
        self.event_demand_scale = scenario_event_demand_scale(scenario)
        self.directional_drift_scale = scenario_directional_drift_scale(scenario)
        self.regime_persistence_scale = scenario_regime_persistence_scale(scenario, SIM_CALIBRATION)
        self.regime_target_duration_seconds = scenario_regime_target_duration_seconds(scenario, SIM_CALIBRATION)

        for key in DEMAND_KEYS:
            target = float(preset.get("demand_bias", {}).get(key, 0.0))
            current = float(self.demands.get(key, 0.0))
            self.demands[key] = float(np.clip(0.70 * current + 0.30 * target, -1.5, 1.5))

        for event_type, start_fraction, direction, magnitude, duration in preset.get("events", []):
            delay = int(max(5, duration * float(start_fraction) * 0.25))

            event = SyntheticEvent(
                event_type=event_type,
                start_second=int(second + delay),
                direction=float(direction),
                magnitude=float(magnitude) * SIM_EVENT_MAGNITUDE_SCALE,
                duration_seconds=max(1, int(duration)),
            )

            self.events.append(event)
            self.generated_events.append(event)

        self.events = sorted(self.events, key=lambda event: event.start_second)
        self.generated_events = sorted(self.generated_events, key=lambda event: event.start_second)


    def update_depth_feedback(self, bid_depth_10bps, ask_depth_10bps):
        if self.depth_target_10bps <= 0:
            return
        realized = (finite_float(bid_depth_10bps, 0.0) + finite_float(ask_depth_10bps, 0.0)) / 2.0
        if realized <= 0:
            return
        alpha = 0.04
        if self.depth_feedback_updates == 0:
            self.depth_ema_10bps = realized
        else:
            self.depth_ema_10bps = (1.0 - alpha) * self.depth_ema_10bps + alpha * realized
        self.depth_feedback_updates += 1
        target = self.depth_target_10bps
        severe_liquidity_event = self.active_event_type in {"exchange_outage", "liquidity_vacuum"}
        lower_guard = self.depth_target_min_10bps if not severe_liquidity_event else self.depth_historical_10bps * 0.03
        upper_guard = self.depth_target_max_10bps
        ratio = target / max(self.depth_ema_10bps, 1e-9)
        exponent = 0.035
        if lower_guard > 0 and self.depth_ema_10bps < lower_guard:
            exponent = 0.075
        elif upper_guard > 0 and self.depth_ema_10bps > upper_guard:
            exponent = 0.075
        adjustment = float(np.clip(ratio, 0.70, 1.35) ** exponent)
        self.book_depth_multiplier = float(np.clip(self.book_depth_multiplier * adjustment, 0.08, 6.0))

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

        self.trend_bias *= scenario_trend_bias_scale(self.scenario)
        self.append_calibrated_events(params.get("event_frequency_distributions", {}))
        self.load_waveform_bands()

    def load_one_waveform_band(self, name, waveform):
        if not isinstance(waveform, dict):
            return
        points = waveform.get("normalized_points", [])
        if not isinstance(points, list) or len(points) < 4:
            return
        scale = finite_float(waveform.get("wave_drift_scale_per_second"), 0.0)
        if scale <= 0:
            return
        self.waveform_bands.append(
            {
                "name": name,
                "points": [finite_float(value, 0.0) for value in points],
                "offset": self.rng.uniform(0, len(points)),
                "scale": scale,
                "dominant_period_seconds": finite_float(waveform.get("dominant_period_seconds"), 0.0),
                "band_energy_fraction": finite_float(waveform.get("band_energy_fraction"), 0.0),
            }
        )

    def load_waveform_bands(self):
        if not isinstance(self.calibration, dict):
            return
        waveforms = self.calibration.get("fair_value_waveforms", {})
        if isinstance(waveforms, dict):
            for name in ["session", "low", "mid", "high"]:
                self.load_one_waveform_band(name, waveforms.get(name, {}))
        if self.waveform_bands:
            return
        waveform = self.calibration.get("fair_value_waveform", {})
        points = waveform.get("normalized_points", []) if isinstance(waveform, dict) else []
        if isinstance(points, list) and len(points) >= 4:
            self.waveform_points = [finite_float(value, 0.0) for value in points]
            self.waveform_offset = self.rng.uniform(0, len(self.waveform_points))
            self.waveform_scale = finite_float(waveform.get("wave_drift_scale_per_second"), 0.0)

    def append_calibrated_events(self, event_params):
        if not isinstance(event_params, dict):
            return
        sim_days = SIM_SECONDS / 86400.0
        added = 0
        policy = self.event_policy
        allowed = policy.get("allowed")
        per_event_scale = policy.get("per_event_scale", {})
        magnitude_scale = finite_float(policy.get("magnitude_scale"), 1.0)
        duration_scale = finite_float(policy.get("duration_scale"), 1.0)
        for event_type, info in event_params.items():
            if not isinstance(info, dict) or "frequency_per_day" not in info:
                continue
            if allowed is not None and event_type not in allowed:
                continue
            event_scale = self.event_frequency_scale * finite_float(per_event_scale.get(event_type), 1.0)
            expected = finite_float(info.get("frequency_per_day"), 0.0) * sim_days * event_scale
            count = min(6, poisson_sample(self.rng, expected))
            for _ in range(count):
                direction = int(finite_float(info.get("direction"), 0.0))
                if direction == 0:
                    direction = 1 if self.rng.random() < 0.5 else -1
                magnitude = (
                    sample_range(self.rng, info.get("magnitude_range"), 0.25)
                    * SIM_EVENT_MAGNITUDE_SCALE
                    * magnitude_scale
                )
                duration = int(sample_range(self.rng, info.get("duration_seconds_range"), 600) * duration_scale)
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
            self.generated_events = sorted(self.generated_events, key=lambda event: event.start_second)

    def waveform_at(self, second):
        if not self.waveform_points:
            return 0.0
        position = (second / max(1, SIM_SECONDS)) * len(self.waveform_points) + self.waveform_offset
        low = int(math.floor(position)) % len(self.waveform_points)
        high = (low + 1) % len(self.waveform_points)
        fraction = position - math.floor(position)
        return (1.0 - fraction) * self.waveform_points[low] + fraction * self.waveform_points[high]

    def waveform_band_at(self, band, second):
        points = band.get("points", [])
        if not points:
            return 0.0
        position = (second / max(1, SIM_SECONDS)) * len(points) + finite_float(band.get("offset"), 0.0)
        low = int(math.floor(position)) % len(points)
        high = (low + 1) % len(points)
        fraction = position - math.floor(position)
        return (1.0 - fraction) * points[low] + fraction * points[high]

    def multiband_waveform_contribution_at(self, second):
        if not self.waveform_bands:
            return 0.0, 0.0
        total_value = 0.0
        total_contribution = 0.0
        for band in self.waveform_bands:
            value = self.waveform_band_at(band, second)
            previous = self.waveform_band_at(band, max(0, second - 1))
            contribution = (value - previous) * finite_float(band.get("scale"), 0.0)
            total_value += value
            total_contribution += contribution
        return total_value, float(np.clip(total_contribution, -0.00012, 0.00012))

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
        self.maybe_flip_latent_regime(second)
        event_contributions = self.update_event_demands(second)
        for key in DEMAND_KEYS:
            base = float(self.preset.get("demand_bias", {}).get(key, 0.0))
            noise = self.rng.gauss(0.0, 0.0018)
            self.demands[key] = (
                self.demand_persistence[key] * self.demands[key]
                + (1.0 - self.demand_persistence[key]) * base
                + event_contributions.get(key, 0.0) * self.event_demand_scale
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
        target_volatility = float(np.clip(self.base_volatility * (1.0 + abs(directional) * 2.2 + self.risk_aversion * 0.8), 0.000025, 0.0012))
        self.volatility_regime = float(
            np.clip(
                SIM_VOL_PERSISTENCE * self.volatility_regime
                + (1.0 - SIM_VOL_PERSISTENCE) * target_volatility,
                0.000025,
                0.0012,
            )
        )
        noise = sample_return_innovation(self.rng, self.volatility_regime)
        self.waveform_value = 0.0
        self.waveform_contribution = 0.0
        if SIM_FAIR_VALUE_MODE == "historical_wave_calibrated":
            if self.waveform_bands:
                self.waveform_value, self.waveform_contribution = self.multiband_waveform_contribution_at(second)
            elif self.waveform_points:
                self.waveform_value = self.waveform_at(second)
                previous_wave = self.waveform_at(max(0, second - 1))
                self.waveform_contribution = float(
                    np.clip((self.waveform_value - previous_wave) * self.waveform_scale, -0.00008, 0.00008)
                )
        drift = self.trend_bias + directional * self.directional_drift_scale
        drift += self.waveform_contribution
        second_return = bounded_return(
            drift + noise,
            self.return_caps["second"],
            SIM_USE_SOFT_RETURN_LIMITS,
        )
        self.fair_value = absurdity_guard_price(self.fair_value * (1.0 + second_return))
        close_to_close_return = safe_ratio(self.fair_value - START_PRICE, START_PRICE)
        capped_close_return = bounded_return(
            close_to_close_return,
            self.return_caps["close_to_close"],
            SIM_USE_SOFT_RETURN_LIMITS,
        )
        self.fair_value = absurdity_guard_price(START_PRICE * (1.0 + capped_close_return))


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
        base_size = BASE_LEVEL_SIZE * self.state.liquidity_regime * self.state.book_depth_multiplier * risk_multiplier * withdrawal_multiplier
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
        size = BASE_LEVEL_SIZE * self.state.liquidity_regime * self.state.book_depth_multiplier * self.rng.uniform(0.2, 0.8)
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
        self.book.place_limit(side, price, BASE_LEVEL_SIZE * self.state.book_depth_multiplier * self.rng.uniform(0.05, 0.45))
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
            self.book.place_limit(
                side,
                price,
                BASE_LEVEL_SIZE * self.state.book_depth_multiplier * self.rng.uniform(0.4, 1.8) * SIM_INSTITUTIONAL_INTENSITY,
            )
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
        self.book.replenish_around(
            self.state.fair_value,
            self.state.liquidity_regime,
            self.state.book_depth_multiplier,
        )
        after_mid = self.book.mid_price()
        raw_return = safe_ratio(after_mid - before_mid, before_mid)
        second_cap = effective_return_caps(self.state.scenario)["second"]
        limited_return = bounded_return(raw_return, second_cap, SIM_USE_SOFT_RETURN_LIMITS)
        if abs(limited_return - raw_return) > 1e-12:
            capped_mid = before_mid * (1.0 + limited_return)
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
        "source_kind": SOURCE_KIND,
        "symbol": SYMBOL,
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


def ten_second_row(second_rows, previous_ten_second_snapshot):
    """Aggregate per-second simulator observations into one true 10s flow row."""
    first = second_rows[0]
    last = second_rows[-1]
    buy_volume = sum(float(row.get("market_buy_volume_10s", 0.0)) for row in second_rows)
    sell_volume = sum(float(row.get("market_sell_volume_10s", 0.0)) for row in second_rows)
    total_volume = buy_volume + sell_volume
    trade_count = sum(int(row.get("trade_count_10s", 0) or 0) for row in second_rows)
    bid_10 = float(last["bid_depth_10bps"])
    ask_10 = float(last["ask_depth_10bps"])
    imbalance_10 = float(last["order_book_imbalance_10bps"])
    prev_bid_10 = (
        previous_ten_second_snapshot.get("bid_depth_10bps", bid_10)
        if previous_ten_second_snapshot
        else float(first.get("bid_depth_10bps", bid_10))
    )
    prev_ask_10 = (
        previous_ten_second_snapshot.get("ask_depth_10bps", ask_10)
        if previous_ten_second_snapshot
        else float(first.get("ask_depth_10bps", ask_10))
    )
    prev_imbalance = (
        previous_ten_second_snapshot.get("order_book_imbalance_10bps", imbalance_10)
        if previous_ten_second_snapshot
        else float(first.get("order_book_imbalance_10bps", imbalance_10))
    )
    active_agent_types = sorted(
        {
            agent
            for row in second_rows
            for agent in str(row.get("hidden_active_agent_types", "none")).split(",")
            if agent and agent != "none"
        }
    )
    row = dict(last)
    row.update(
        {
            "source_kind": SOURCE_KIND,
            "symbol": SYMBOL,
            "market_buy_volume_10s": buy_volume,
            "market_sell_volume_10s": sell_volume,
            "total_trade_volume_10s": total_volume,
            "trade_count_10s": trade_count,
            "market_pressure_10s": safe_ratio(buy_volume - sell_volume, total_volume),
            "bid_depth_change_10bps": safe_ratio(bid_10 - prev_bid_10, prev_bid_10),
            "ask_depth_change_10bps": safe_ratio(ask_10 - prev_ask_10, prev_ask_10),
            "imbalance_change_10bps": imbalance_10 - float(prev_imbalance),
            "hidden_whale_pressure": sum(float(item.get("hidden_whale_pressure", 0.0)) for item in second_rows),
            "hidden_agent_buy_intensity": buy_volume,
            "hidden_agent_sell_intensity": sell_volume,
            "hidden_active_agent_types": ",".join(active_agent_types) if active_agent_types else "none",
        }
    )
    return row


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
        "source_kind": SOURCE_KIND,
        "symbol": SYMBOL,
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
        "hidden_runtime_scenario": last.get("hidden_runtime_scenario", last["hidden_scenario"]),
        "hidden_regime_segment_id": last.get("hidden_regime_segment_id", 0),
        "hidden_regime_age_seconds": last.get("hidden_regime_age_seconds", 0),
        "hidden_regime_remaining_seconds": last.get("hidden_regime_remaining_seconds", 0),
        "hidden_regime_transition": last.get("hidden_regime_transition", False),
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
    frame = pd.DataFrame(
        [{column: fmt(row.get(column, "")) for column in columns} for row in rows],
        columns=columns,
    )
    tiny_io.safe_write_csv_atomic(frame, path)


def apply_book_price_caps(book, minute_open_mid, state):
    """Keep synthetic prices inside configured pretraining realism bounds."""
    caps = getattr(state, "return_caps", effective_return_caps(SCENARIO))
    mid = book.mid_price()

    if caps["minute"] > 0 and minute_open_mid and minute_open_mid > 0:
        minute_return = safe_ratio(mid - minute_open_mid, minute_open_mid)
        limited_minute_return = bounded_return(minute_return, caps["minute"], SIM_USE_SOFT_RETURN_LIMITS)
        if abs(limited_minute_return - minute_return) > 1e-12:
            capped_mid = absurdity_guard_price(minute_open_mid * (1.0 + limited_minute_return))
            book.shift_prices(capped_mid)
            mid = book.mid_price()

    if caps["close_to_close"] > 0 and START_PRICE > 0:
        full_return = safe_ratio(mid - START_PRICE, START_PRICE)
        limited_full_return = bounded_return(full_return, caps["close_to_close"], SIM_USE_SOFT_RETURN_LIMITS)
        if abs(limited_full_return - full_return) > 1e-12:
            capped_mid = absurdity_guard_price(START_PRICE * (1.0 + limited_full_return))
            book.shift_prices(capped_mid)

def run_simulation():
    rng = random.Random(RANDOM_SEED)

    scheduler = None
    initial_scenario = SCENARIO

    if SIM_MIXED_RUNTIME:
        scheduler = MixedRuntimeScheduler(
            rng,
            SIM_MIXED_SCENARIOS,
            SIM_MIXED_MIN_SECONDS,
            SIM_MIXED_MAX_SECONDS,
        )
        first = scheduler.update(0)
        initial_scenario = first["scenario"]

    state = LatentState(initial_scenario, rng)
    book = LimitOrderBook(START_PRICE, rng)
    bots = BotSwarm(book, state, rng)
    snapshots = []
    one_minute_rows = []
    ten_second_bucket = []
    minute_bucket = []
    previous_second_snapshot = None
    previous_ten_second_snapshot = None
    previous_close = None
    minute_open_mid = book.mid_price()

    for second in range(SIM_SECONDS):
        if second % 60 == 0:
            minute_open_mid = book.mid_price()

        runtime_info = {
            "scenario": state.scenario,
            "segment_id": 0,
            "age_seconds": second,
            "remaining_seconds": max(0, SIM_SECONDS - second),
            "transition": False,
        }

        if scheduler is not None:
            runtime_info = scheduler.update(second)
            if runtime_info["scenario"] != state.scenario:
                state.switch_runtime_scenario(runtime_info["scenario"], second)

        state.update(second)
        fills = bots.step()
        apply_book_price_caps(book, minute_open_mid, state)

        timestamp_ms = START_TIMESTAMP_MS + second * 1000
        second_snapshot = snapshot_from_book(timestamp_ms, book, state, fills, previous_second_snapshot)

        second_snapshot["hidden_runtime_scenario"] = runtime_info["scenario"]
        second_snapshot["hidden_regime_segment_id"] = runtime_info["segment_id"]
        second_snapshot["hidden_regime_age_seconds"] = runtime_info["age_seconds"]
        second_snapshot["hidden_regime_remaining_seconds"] = runtime_info["remaining_seconds"]
        second_snapshot["hidden_regime_transition"] = runtime_info["transition"]

        state.update_depth_feedback(second_snapshot["bid_depth_10bps"], second_snapshot["ask_depth_10bps"])
        ten_second_bucket.append(second_snapshot)
        previous_second_snapshot = second_snapshot

        if len(ten_second_bucket) == 10:
            snapshot = ten_second_row(ten_second_bucket, previous_ten_second_snapshot)
            snapshots.append(snapshot)
            minute_bucket.append(snapshot)
            previous_ten_second_snapshot = snapshot
            ten_second_bucket = []

        if len(minute_bucket) == 6:
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


def trend_chop_duration_summary(close_values, step_seconds=60):
    close = np.asarray(close_values, dtype=np.float64)
    if len(close) < 4:
        return {"mean_seconds": 0.0, "median_seconds": 0.0, "max_seconds": 0.0, "segments": 0}
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    window = min(30, max(3, len(close) // 10))
    rolling_return = pd.Series(np.log(close)).diff(window).fillna(0.0)
    rolling_vol = pd.Series(returns).rolling(window, min_periods=2).std().fillna(0.0)
    threshold = np.maximum(rolling_vol * math.sqrt(window) * 0.75, 0.0015)
    high_vol_threshold = float(rolling_vol.quantile(0.75)) if len(rolling_vol) else 0.0
    labels = np.where(
        rolling_return > threshold,
        "trend",
        np.where(
            rolling_return < -threshold,
            "trend",
            np.where((rolling_vol > high_vol_threshold) & (high_vol_threshold > 0), "high_vol_chop", "chop"),
        ),
    )
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
    label_counts = {str(label): int(count) for label, count in Counter(labels).items()}
    return {
        "mean_seconds": float(np.mean(lengths) * step_seconds) if lengths else 0.0,
        "median_seconds": float(np.median(lengths) * step_seconds) if lengths else 0.0,
        "max_seconds": float(np.max(lengths) * step_seconds) if lengths else 0.0,
        "segments": int(len(lengths)),
        "label_counts": dict(label_counts),
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
    duration_summary = trend_chop_duration_summary(closes, step_seconds=60)
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
    historical_mean_values = []
    historical_median_values = []
    for info in historical_durations.values():
        if not isinstance(info, dict):
            continue
        mean_value = finite_float(info.get("mean_seconds"), 0.0)
        quantiles = info.get("quantiles_seconds", {})
        median_value = finite_float(quantiles.get("0.5") if isinstance(quantiles, dict) else None, 0.0)
        if mean_value > 0:
            historical_mean_values.append(mean_value)
        if median_value > 0:
            historical_median_values.append(median_value)
    historical_mean_duration = float(np.mean(historical_mean_values)) if historical_mean_values else 0.0
    historical_median_duration = float(np.median(historical_median_values)) if historical_median_values else 0.0
    print(
        "- trend/chop duration seconds synthetic: "
        f"segments={duration_summary['segments']} "
        f"mean={duration_summary['mean_seconds']:.2f} "
        f"median={duration_summary['median_seconds']:.2f} "
        f"max={duration_summary['max_seconds']:.2f} "
        f"labels={duration_summary.get('label_counts', {})}"
    )
    print(
        "- trend/chop mean/median duration seconds historical: "
        f"mean={historical_mean_duration:.2f} median={historical_median_duration:.2f}"
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


def saturation_ratio(value, cap):
    cap = abs(finite_float(cap, 0.0))
    if cap <= 0:
        return 0.0
    return abs(finite_float(value, 0.0)) / cap

class RegimeScheduler:
    def __init__(self, rng, scenario_names, min_seconds=300, max_seconds=1800):
        self.rng = rng
        self.scenario_names = list(scenario_names)
        self.min_seconds = int(min_seconds)
        self.max_seconds = int(max_seconds)
        self.segment_id = -1
        self.current_scenario = None
        self.segment_start_second = 0
        self.segment_end_second = 0
        self.transition_now = False

    def choose_next(self, now_second):
        previous = self.current_scenario
        choices = [s for s in self.scenario_names if s != previous] or self.scenario_names

        self.current_scenario = self.rng.choice(choices)
        self.segment_id += 1
        self.segment_start_second = int(now_second)
        duration = self.rng.randint(self.min_seconds, self.max_seconds)
        self.segment_end_second = int(now_second + duration)
        self.transition_now = True

    def update(self, now_second):
        self.transition_now = False
        if self.current_scenario is None or now_second >= self.segment_end_second:
            self.choose_next(now_second)

        return {
            "scenario": self.current_scenario,
            "segment_id": self.segment_id,
            "age_seconds": int(now_second - self.segment_start_second),
            "remaining_seconds": int(max(0, self.segment_end_second - now_second)),
            "transition": self.transition_now,
        }


def diagnostics(snapshots, one_minute_rows, state):
    frame = pd.DataFrame(snapshots)
    minute_frame = pd.DataFrame(one_minute_rows)
    caps = effective_return_caps(SCENARIO)
    print("Synthetic market microstructure simulator")
    print(f"SYMBOL: {SYMBOL}")
    print(f"SCENARIO: {SCENARIO}")
    print(f"SIM_SECONDS: {SIM_SECONDS}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")
    print(f"SIM_CALIBRATION_PATH: {SIM_CALIBRATION_PATH if SIM_CALIBRATION else 'not loaded'}")
    print(f"SIM_FAIR_VALUE_MODE: {SIM_FAIR_VALUE_MODE}")
    print(
        "Return innovation: "
        f"mode={SIM_RETURN_INNOVATION_MODE} "
        f"student_t_df={SIM_STUDENT_T_DF:g} "
        f"jump_probability={SIM_JUMP_MIXTURE_PROBABILITY:g} "
        f"jump_scale={SIM_JUMP_MIXTURE_SCALE:g}"
    )
    print(f"Volatility persistence: {SIM_VOL_PERSISTENCE:g}")
    print(f"Soft return limits enabled: {SIM_USE_SOFT_RETURN_LIMITS}")
    print(
        "Effective scenario caps: "
        f"multiplier={caps['scenario_cap_multiplier']:.3g} "
        f"second={caps['second']:.4%} "
        f"minute={caps['minute']:.4%} "
        f"close_to_close={caps['close_to_close']:.2%}"
    )
    print(
        "Scenario event controls: "
        f"global_scale={SIM_EVENT_FREQUENCY_SCALE:g} "
        f"effective_scale={state.event_frequency_scale:g} "
        f"event_demand_scale={state.event_demand_scale:g} "
        f"directional_drift_scale={state.directional_drift_scale:g}"
    )
    print(f"Regime persistence scale: {state.regime_persistence_scale:g}")
    print(f"Regime target duration seconds: {state.regime_target_duration_seconds:g}")
    print(
        "Book depth calibration: "
        f"raw_base_level_size={BASE_LEVEL_SIZE_RAW:g} "
        f"effective_base_level_size={BASE_LEVEL_SIZE:g} "
        f"depth_scale={SIM_DEPTH_CALIBRATION_SCALE:g} "
        f"historical_10bps_depth={SIM_DEPTH_HISTORICAL_10BPS:g} "
        f"target_10bps_depth={SIM_DEPTH_TARGET_10BPS:g}"
    )
    if state.waveform_bands:
        band_summary = ", ".join(
            f"{band['name']}(period={band.get('dominant_period_seconds', 0.0):.0f}s, energy={band.get('band_energy_fraction', 0.0):.3f})"
            for band in state.waveform_bands
        )
        print(f"Fair-value waveform bands loaded: {band_summary}")
    elif state.waveform_points:
        print("Fair-value waveform bands loaded: legacy_single_wave")
    else:
        print("Fair-value waveform bands loaded: none")
    print("Recommended calibrated synthetic-training mode:")
    print('  $env:SIM_CALIBRATION_PATH="data/simulated/calibration/SOLUSDT_kraken_sim_calibration.json"')
    print('  $env:SIM_FAIR_VALUE_MODE="historical_wave_calibrated"')
    print(f"Active agent types: {', '.join(BotSwarm.AGENT_TYPES if SIM_ENABLE_RICH_AGENTS else BotSwarm.AGENT_TYPES[:4])}")
    print(f"Generated events: {len(state.generated_events)}")
    event_type_counts = Counter(event.event_type for event in state.generated_events)
    print(f"Event type counts: {dict(event_type_counts)}")
    print(f"10s flow output: {SNAPSHOT_PATH}")
    print(f"1m output: {ONE_MINUTE_PATH}")
    print(f"10s rows: {len(frame)}")
    print(f"1m rows: {len(minute_frame)}")
    if len(frame):
        print(f"First timestamp: {int(frame['timestamp'].iloc[0])} {frame['time'].iloc[0]}")
        print(f"Last timestamp: {int(frame['timestamp'].iloc[-1])} {frame['time'].iloc[-1]}")
        print(f"Mid price min/mean/max: {frame['mid_price'].min():.6g} / {frame['mid_price'].mean():.6g} / {frame['mid_price'].max():.6g}")
        print(f"Average spread: {frame['spread_percent'].mean():.4%}")
        print(f"Average total flow per 10s: {frame['total_trade_volume_10s'].mean():.6g}")
        print(f"Average trade count per 10s: {frame['trade_count_10s'].mean():.3f}")
        print(f"Average 10bps imbalance: {frame['order_book_imbalance_10bps'].mean():.4f}")
        avg_bid_10 = float(pd.to_numeric(frame["bid_depth_10bps"], errors="coerce").mean())
        avg_ask_10 = float(pd.to_numeric(frame["ask_depth_10bps"], errors="coerce").mean())
        avg_depth_10 = (avg_bid_10 + avg_ask_10) / 2.0
        print(f"Average bid/ask depth 10bps: {avg_bid_10:.6g} / {avg_ask_10:.6g}")
        if state.depth_historical_10bps > 0:
            print(
                "Realized depth calibration: "
                f"historical_10bps={state.depth_historical_10bps:.6g} "
                f"target_10bps={state.depth_target_10bps:.6g} "
                f"realized_avg_10bps={avg_depth_10:.6g} "
                f"realized/historical={safe_ratio(avg_depth_10, state.depth_historical_10bps):.3f} "
                f"final_feedback_multiplier={state.book_depth_multiplier:.3f}"
            )
            if SCENARIO == "calm_chop" and avg_depth_10 > state.depth_historical_10bps * 2.0:
                print("WARNING: calm_chop realized 10bps depth is above 2x historical.")
            if SCENARIO == "calm_chop" and avg_depth_10 < state.depth_historical_10bps * 0.5:
                print("WARNING: calm_chop realized 10bps depth is below 0.5x historical.")
            if SCENARIO == "liquidity_crisis" and avg_depth_10 < state.depth_historical_10bps * 0.03:
                print("WARNING: liquidity_crisis realized 10bps depth stayed below 0.03x historical for the full run.")
        mid_prices = pd.to_numeric(frame["mid_price"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        max_abs_1s_return = 0.0
        if len(mid_prices) > 1:
            max_abs_1s_return = float(mid_prices.pct_change().abs().max() or 0.0)
        print(
            "Cap saturation diagnostics: "
            f"max_abs_1s_return={max_abs_1s_return:.4%} "
            f"ratio={saturation_ratio(max_abs_1s_return, caps['second']):.3f}"
        )
        whale_active_pct = (pd.to_numeric(frame["hidden_whale_pressure"], errors="coerce").fillna(0.0).abs() > 0).mean()
        print(f"10s rows with whale activity: {whale_active_pct:.2%}")
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
        duration_summary = trend_chop_duration_summary(closes, step_seconds=60)
        print(
            "Trend/chop segment diagnostics: "
            f"segments={duration_summary['segments']} "
            f"mean_seconds={duration_summary['mean_seconds']:.2f} "
            f"median_seconds={duration_summary['median_seconds']:.2f} "
            f"max_seconds={duration_summary['max_seconds']:.2f} "
            f"labels={duration_summary.get('label_counts', {})}"
        )
        minute_returns = pd.to_numeric(minute_frame["return_1"], errors="coerce").abs()
        max_abs_1m_return = float(minute_returns.max()) if len(minute_returns) else 0.0
        close_saturation = saturation_ratio(final_return, caps["close_to_close"])
        print(
            "Cap saturation diagnostics: "
            f"abs_close_to_close/effective_cap={close_saturation:.3f} "
            f"max_abs_1m_return/effective_cap={saturation_ratio(max_abs_1m_return, caps['minute']):.3f}"
        )
        if close_saturation > 0.90:
            print("WARNING: close-to-close saturation ratio is above 0.90.")
        if caps["close_to_close"] > 0 and abs(final_return) > caps["close_to_close"] * 1.25:
            print("WARNING: close-to-close return materially exceeded effective scenario cap.")
        if len(minute_returns) and caps["minute"] > 0 and max_abs_1m_return > caps["minute"] * 1.25:
            print("WARNING: at least one 1m return is materially larger than effective scenario minute cap.")
    print_calibration_comparison(frame, minute_frame, state)
    print("Hidden diagnostic columns are prefixed with hidden_ and are ignored by model feature selectors.")
    print("No trades/orders/private API behavior. Simulation only.")


def main():
    snapshots, one_minute_rows, state = run_simulation()
    diagnostics(snapshots, one_minute_rows, state)


if __name__ == "__main__":
    main()
