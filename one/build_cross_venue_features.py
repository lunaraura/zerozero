import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
SYMBOLS = [
    value.strip().upper()
    for value in os.getenv("SYMBOLS", SYMBOL).split(",")
    if value.strip()
]
VENUES = [
    value.strip().lower()
    for value in os.getenv("VENUES", "binanceus,kraken").split(",")
    if value.strip()
]
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data" / "realtime"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
EPSILON = 1e-12
STALE_MS_THRESHOLD = int(os.getenv("CROSS_VENUE_STALE_MS", "1500"))


def venue_snapshot_path(venue, symbol):
    return OUTPUT_DIR / venue / f"{symbol}_10s_flow.csv"


def output_path(symbol):
    return OUTPUT_DIR / f"{symbol}_cross_venue_features.csv"


def load_venue_snapshots(venue, symbol):
    path = venue_snapshot_path(venue, symbol)
    if not path.exists():
        return pd.DataFrame(), path

    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        return pd.DataFrame(), path

    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).copy()
    frame["timestamp"] = frame["timestamp"].astype("int64")
    # Align snapshots to the nearest whole second so 1s/10s venue snapshots
    # from different websocket clocks can be compared without peeking forward.
    frame["aligned_second"] = np.rint(frame["timestamp"] / 1000).astype("int64")

    for column in frame.columns:
        if column not in {"time", "venue", "source_quality"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    keep_columns = [
        "aligned_second",
        "timestamp",
        "time",
        "source_quality",
        "mid_price",
        "best_bid",
        "best_ask",
        "spread_percent",
        "market_pressure_10s",
        "bid_depth_10bps",
        "ask_depth_10bps",
        "order_book_imbalance_10bps",
    ]
    for column in keep_columns:
        if column not in frame.columns:
            frame[column] = np.nan

    frame = frame.sort_values("timestamp").drop_duplicates("aligned_second", keep="last")
    frame["venue"] = venue
    return frame[keep_columns + ["venue"]].reset_index(drop=True), path


def safe_ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return np.nan
    return float(numerator / denominator)


def basis_points_fraction(value):
    if not np.isfinite(value):
        return np.nan
    return float(value * 10_000.0)


def signed_return(series, offset_rows):
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean.where(clean > 0, np.nan)
    return clean.pct_change(periods=offset_rows, fill_method=None).replace([np.inf, -np.inf], np.nan)


def enrich_returns(frame):
    frame = frame.sort_values("aligned_second").copy()
    frame["return_1s"] = signed_return(frame["mid_price"], 1)
    frame["return_3s"] = signed_return(frame["mid_price"], 3)
    frame["return_10s"] = signed_return(frame["mid_price"], 10)
    return frame


def pressure_agreement_and_divergence(values):
    if len(values) < 2:
        return np.nan, np.nan

    finite = [float(value) for value in values if np.isfinite(value)]
    if len(finite) < 2:
        return 0.0, 0.0

    nonzero = [value for value in finite if abs(value) > EPSILON]
    if len(nonzero) < 2:
        return 0.0, 0.0

    signs = [np.sign(value) for value in nonzero]
    same_sign = all(sign == signs[0] for sign in signs)
    return float(same_sign), float(not same_sign)


def clipped_log_ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    return float(np.clip(np.log((numerator + EPSILON) / (denominator + EPSILON)), -5.0, 5.0))


def strongest_abs_return(values_by_venue):
    best_venue = ""
    best_value = np.nan
    best_abs = -1.0
    for venue, value in values_by_venue.items():
        if np.isfinite(value) and abs(value) > best_abs:
            best_venue = venue
            best_value = value
            best_abs = abs(value)
    return best_venue, best_value


def build_symbol_features(symbol):
    frames = {}
    paths = {}
    for venue in VENUES:
        frame, path = load_venue_snapshots(venue, symbol)
        paths[venue] = path
        if len(frame) > 0:
            frames[venue] = enrich_returns(frame)

    if len(frames) == 0:
        return pd.DataFrame(), paths

    all_seconds = sorted(set().union(*[set(frame["aligned_second"]) for frame in frames.values()]))
    indexed = {
        venue: frame.set_index("aligned_second", drop=False)
        for venue, frame in frames.items()
    }

    rows = []
    for aligned_second in all_seconds:
        venue_rows = {}
        for venue, frame in indexed.items():
            if aligned_second in frame.index:
                venue_rows[venue] = frame.loc[aligned_second]

        mid_values = [float(row["mid_price"]) for row in venue_rows.values()]
        spread_values = [float(row["spread_percent"]) for row in venue_rows.values()]
        bid_depth_values = [float(row["bid_depth_10bps"]) for row in venue_rows.values()]
        ask_depth_values = [float(row["ask_depth_10bps"]) for row in venue_rows.values()]
        imbalance_values = [float(row["order_book_imbalance_10bps"]) for row in venue_rows.values()]
        pressure_values = [float(row["market_pressure_10s"]) for row in venue_rows.values()]

        finite_mids = [value for value in mid_values if np.isfinite(value) and value > 0]
        mean_mid = float(np.mean(finite_mids)) if finite_mids else np.nan
        venue_mid_diff_bps = (
            basis_points_fraction((max(finite_mids) - min(finite_mids)) / mean_mid)
            if len(finite_mids) >= 2 and mean_mid > 0
            else np.nan
        )

        finite_spreads = [value for value in spread_values if np.isfinite(value)]
        venue_spread_diff_bps = (
            basis_points_fraction(max(finite_spreads) - min(finite_spreads))
            if len(finite_spreads) >= 2
            else np.nan
        )

        finite_bid_depth = [value for value in bid_depth_values if np.isfinite(value) and value > 0]
        finite_ask_depth = [value for value in ask_depth_values if np.isfinite(value) and value > 0]
        finite_imbalances = [value for value in imbalance_values if np.isfinite(value)]

        returns_1s = {venue: float(row["return_1s"]) for venue, row in venue_rows.items()}
        returns_3s = {venue: float(row["return_3s"]) for venue, row in venue_rows.items()}
        returns_10s = {venue: float(row["return_10s"]) for venue, row in venue_rows.items()}
        lead_1s_venue, lead_1s_return = strongest_abs_return(returns_1s)
        lead_3s_venue, lead_3s_return = strongest_abs_return(returns_3s)
        lead_10s_venue, lead_10s_return = strongest_abs_return(returns_10s)
        agreement, divergence = pressure_agreement_and_divergence(pressure_values)

        binance_bid_depth = (
            float(venue_rows["binanceus"]["bid_depth_10bps"])
            if "binanceus" in venue_rows
            else np.nan
        )
        kraken_bid_depth = (
            float(venue_rows["kraken"]["bid_depth_10bps"])
            if "kraken" in venue_rows
            else np.nan
        )
        binance_ask_depth = (
            float(venue_rows["binanceus"]["ask_depth_10bps"])
            if "binanceus" in venue_rows
            else np.nan
        )
        kraken_ask_depth = (
            float(venue_rows["kraken"]["ask_depth_10bps"])
            if "kraken" in venue_rows
            else np.nan
        )
        source_quality = (
            "ok"
            if len(venue_rows) >= 2
            else "insufficient_venues"
        )
        timestamps = [
            int(row["timestamp"])
            for row in venue_rows.values()
            if np.isfinite(float(row["timestamp"]))
        ]
        timestamp_diff_ms = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else np.nan
        if np.isfinite(timestamp_diff_ms) and timestamp_diff_ms > STALE_MS_THRESHOLD:
            source_quality = "stale_venue_alignment"

        row = {
            "timestamp": int(aligned_second * 1000),
            "time": pd.to_datetime(aligned_second, unit="s", utc=True).isoformat(),
            "symbol": symbol,
            "venues_present": ",".join(sorted(venue_rows.keys())),
            "venue_count": len(venue_rows),
            "cross_venue_source_quality": source_quality,
            "venue_timestamp_diff_ms": timestamp_diff_ms,
            "venue_mid_diff_bps": venue_mid_diff_bps,
            "venue_spread_diff_bps": venue_spread_diff_bps,
            "venue_bid_depth_ratio_10bps": safe_ratio(max(finite_bid_depth), min(finite_bid_depth))
            if len(finite_bid_depth) >= 2
            else np.nan,
            "venue_ask_depth_ratio_10bps": safe_ratio(max(finite_ask_depth), min(finite_ask_depth))
            if len(finite_ask_depth) >= 2
            else np.nan,
            "log_bid_depth_ratio_10bps": clipped_log_ratio(binance_bid_depth, kraken_bid_depth),
            "log_ask_depth_ratio_10bps": clipped_log_ratio(binance_ask_depth, kraken_ask_depth),
            "venue_imbalance_diff_10bps": float(max(finite_imbalances) - min(finite_imbalances))
            if len(finite_imbalances) >= 2
            else np.nan,
            "leading_venue_1s": lead_1s_venue,
            "leading_venue_return_1s": lead_1s_return,
            "leading_venue_3s": lead_3s_venue,
            "leading_venue_return_3s": lead_3s_return,
            "leading_venue_10s": lead_10s_venue,
            "leading_venue_return_10s": lead_10s_return,
            "cross_venue_pressure_agreement": agreement,
            "cross_venue_pressure_divergence": divergence,
        }

        for venue, venue_row in venue_rows.items():
            row[f"{venue}_mid_price"] = venue_row["mid_price"]
            row[f"{venue}_spread_percent"] = venue_row["spread_percent"]
            row[f"{venue}_market_pressure_10s"] = venue_row["market_pressure_10s"]
            row[f"{venue}_order_book_imbalance_10bps"] = venue_row["order_book_imbalance_10bps"]
            row[f"{venue}_source_quality"] = venue_row.get("source_quality", "")

        rows.append(row)

    return pd.DataFrame(rows), paths


def summarize_numeric(series, label):
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        print(f"- {label}: no finite values")
        return
    print(
        f"- {label}: median={values.median():.6g}, "
        f"p90={values.quantile(0.90):.6g}, max={values.max():.6g}"
    )


def print_health_report(features):
    print("\nCross-venue health report")
    print(f"- row count: {len(features)}")
    if len(features) == 0:
        return

    print("- venue_count distribution:")
    for value, count in features["venue_count"].value_counts(dropna=False).sort_index().items():
        print(f"  {value}: {count}")

    print("- null counts:")
    null_counts = features.isna().sum()
    for column, count in null_counts[null_counts > 0].sort_values(ascending=False).items():
        print(f"  {column}: {int(count)}")

    if "cross_venue_source_quality" in features.columns:
        print("- cross_venue_source_quality distribution:")
        for value, count in features["cross_venue_source_quality"].fillna("blank").value_counts().items():
            print(f"  {value}: {count}")

    print("- source_quality distribution by venue:")
    for venue in VENUES:
        column = f"{venue}_source_quality"
        if column not in features.columns:
            print(f"  {venue}: column missing")
            continue
        counts = features[column].fillna("missing").replace("", "blank").value_counts()
        for value, count in counts.items():
            print(f"  {venue} {value}: {count}")

    abs_mid_diff = pd.to_numeric(
        features.get("venue_mid_diff_bps", pd.Series(dtype=float)),
        errors="coerce",
    ).abs()
    summarize_numeric(abs_mid_diff, "abs venue_mid_diff_bps")
    summarize_numeric(features.get("venue_bid_depth_ratio_10bps", pd.Series(dtype=float)), "venue_bid_depth_ratio_10bps")
    summarize_numeric(features.get("venue_ask_depth_ratio_10bps", pd.Series(dtype=float)), "venue_ask_depth_ratio_10bps")
    summarize_numeric(features.get("log_bid_depth_ratio_10bps", pd.Series(dtype=float)).abs(), "abs log_bid_depth_ratio_10bps")
    summarize_numeric(features.get("log_ask_depth_ratio_10bps", pd.Series(dtype=float)).abs(), "abs log_ask_depth_ratio_10bps")

    agreement = pd.to_numeric(
        features.get("cross_venue_pressure_agreement", pd.Series(dtype=float)),
        errors="coerce",
    )
    divergence = pd.to_numeric(
        features.get("cross_venue_pressure_divergence", pd.Series(dtype=float)),
        errors="coerce",
    )
    print(f"- pressure agreement count: {int((agreement == 1).sum())}")
    print(f"- pressure divergence count: {int((divergence == 1).sum())}")
    print(f"- pressure neutral/zero count: {int(((agreement == 0) & (divergence == 0)).sum())}")

    timestamp_diff = pd.to_numeric(
        features.get("venue_timestamp_diff_ms", pd.Series(dtype=float)),
        errors="coerce",
    )
    stale_count = int((timestamp_diff > STALE_MS_THRESHOLD).sum())
    print(f"- stale row count > {STALE_MS_THRESHOLD}ms: {stale_count}")


def main():
    print("Cross-venue realtime feature builder")
    print(f"SYMBOLS: {', '.join(SYMBOLS)}")
    print(f"VENUES: {', '.join(VENUES)}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print("No trades are placed.")

    for symbol in SYMBOLS:
        features, paths = build_symbol_features(symbol)
        print(f"\nSymbol: {symbol}")
        for venue, path in paths.items():
            print(f"- {venue}: {path} {'found' if path.exists() else 'missing'}")

        output = output_path(symbol)
        output.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output, index=False)
        print(f"Rows written: {len(features)}")
        print(f"Output: {output}")
        if len(features) > 0:
            print(f"Date range: {features.iloc[0]['time']} -> {features.iloc[-1]['time']}")
        print_health_report(features)


if __name__ == "__main__":
    main()
