#!/usr/bin/env python3
"""Emit rawseq input/output contract grids for batch discovery."""

from __future__ import annotations

import csv
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYMBOL = os.getenv("RAWSEQ_IO_SYMBOL", "SOLUSDT").strip().upper() or "SOLUSDT"
VENUE = os.getenv("RAWSEQ_IO_VENUE", "kraken").strip().lower() or "kraken"
SOURCE_PATH_TEXT = os.getenv(
    "RAWSEQ_IO_SOURCE_PATH",
    f"data/realtime/{VENUE}/{SYMBOL}_10s_flow.csv",
).strip()
BUCKET_SECONDS = int(float(os.getenv("RAWSEQ_IO_BUCKET_SECONDS", "10")))
SEQ_LENS = os.getenv("RAWSEQ_IO_SEQ_LENS", "60")
INPUT_STRIDES = os.getenv("RAWSEQ_IO_INPUT_STRIDES", "1,3,6")
OUTPUT_STRIDES = os.getenv("RAWSEQ_IO_OUTPUT_STRIDES", "1,3,6")
INPUT_FEATURES = os.getenv("RAWSEQ_IO_INPUT_FEATURES", "return,ma_distance")
MA_WINDOWS = os.getenv("RAWSEQ_IO_MA_WINDOWS", "60,150")
HIDDENS = os.getenv("RAWSEQ_IO_HIDDENS", "2,2;3,3;4,4")
OUTPUT_LABELS = os.getenv("RAWSEQ_IO_OUTPUT_LABELS", "future_return_path")
OUTPUT_DIR_TEXT = os.getenv(
    "RAWSEQ_IO_OUTPUT_DIR",
    str(PROJECT_ROOT / "data" / "research" / "rawseq_io_contract_grids"),
).strip()

BASE_SOURCE_COLUMNS = ["timestamp", "price"]
OPTIONAL_SOURCE_COLUMNS = ["time", "predicted_side"]
SUPPORTED_INPUT_FEATURES = {"return", "ma_distance", "ma_slope"}
SUPPORTED_OUTPUT_LABELS = {"future_return_path"}


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_csv_ints(text: str, label: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(float(item)))
        except ValueError as exc:
            raise SystemExit(f"{label} contains a non-integer value: {item}") from exc
    if not values:
        raise SystemExit(f"{label} produced no values.")
    return values


def parse_csv_strings(text: str, sep: str = ",") -> list[str]:
    return [item.strip() for item in text.split(sep) if item.strip()]


def parse_hiddens(text: str) -> list[str]:
    values = []
    for item in parse_csv_strings(text, sep=";"):
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if len(parts) != 2:
            raise SystemExit(f"Hidden spec must look like h1,h2: {item}")
        values.append(",".join(str(int(float(part))) for part in parts))
    if not values:
        raise SystemExit("RAWSEQ_IO_HIDDENS produced no values.")
    return values


def safe_slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "rawseq_contract"


def required_columns(input_feature: str) -> list[str]:
    columns = list(BASE_SOURCE_COLUMNS)
    if input_feature in {"return", "ma_distance", "ma_slope"}:
        return columns
    return columns


def support_status(input_feature: str, ma_window: str, output_label: str, hidden: str, seq_len: int) -> tuple[str, str]:
    reasons: list[str] = []
    if input_feature not in SUPPORTED_INPUT_FEATURES:
        reasons.append(f"unsupported_input_feature={input_feature}")
    if output_label not in SUPPORTED_OUTPUT_LABELS:
        reasons.append(f"unsupported_output_label={output_label}")
    if input_feature in {"ma_distance", "ma_slope"} and not ma_window:
        reasons.append("ma_window_required")
    if input_feature == "return" and ma_window:
        reasons.append("ma_window_ignored_for_return")
    h_parts = hidden.split(",")
    if len(h_parts) != 2 or any(int(part) <= 0 for part in h_parts):
        reasons.append(f"unsupported_hidden={hidden}")
    if seq_len <= 0:
        reasons.append(f"unsupported_seq_len={seq_len}")
    unsupported = [reason for reason in reasons if reason != "ma_window_ignored_for_return"]
    return ("unsupported" if unsupported else "supported"), ";".join(reasons)


def contract_slug(
    symbol: str,
    venue: str,
    input_feature: str,
    ma_window: str,
    hidden: str,
    seq_len: int,
    bucket_seconds: int,
    input_stride: int,
    output_stride: int,
    output_label: str,
) -> str:
    ma_part = f"ma{ma_window}" if ma_window else "maNA"
    hidden_part = "h" + hidden.replace(",", "x")
    return safe_slug(
        "_".join(
            [
                symbol,
                venue,
                input_feature,
                ma_part,
                hidden_part,
                f"seq{seq_len}",
                f"b{bucket_seconds}",
                f"is{input_stride}",
                f"os{output_stride}",
                output_label,
            ]
        )
    )


def build_rows() -> list[dict[str, Any]]:
    source_path = resolve_path(SOURCE_PATH_TEXT)
    seq_lens = parse_csv_ints(SEQ_LENS, "RAWSEQ_IO_SEQ_LENS")
    input_strides = parse_csv_ints(INPUT_STRIDES, "RAWSEQ_IO_INPUT_STRIDES")
    output_strides = parse_csv_ints(OUTPUT_STRIDES, "RAWSEQ_IO_OUTPUT_STRIDES")
    input_features = parse_csv_strings(INPUT_FEATURES)
    ma_windows = parse_csv_ints(MA_WINDOWS, "RAWSEQ_IO_MA_WINDOWS")
    hiddens = parse_hiddens(HIDDENS)
    output_labels = parse_csv_strings(OUTPUT_LABELS)

    rows: list[dict[str, Any]] = []
    for seq_len in seq_lens:
        for input_stride in input_strides:
            for output_stride in output_strides:
                for input_feature in input_features:
                    feature_ma_windows: list[str]
                    if input_feature in {"ma_distance", "ma_slope"}:
                        feature_ma_windows = [str(value) for value in ma_windows]
                    else:
                        feature_ma_windows = [""]
                    for ma_window in feature_ma_windows:
                        for hidden in hiddens:
                            for output_label in output_labels:
                                status, reason = support_status(input_feature, ma_window, output_label, hidden, seq_len)
                                rows.append(
                                    {
                                        "contract_slug": contract_slug(
                                            SYMBOL,
                                            VENUE,
                                            input_feature,
                                            ma_window,
                                            hidden,
                                            seq_len,
                                            BUCKET_SECONDS,
                                            input_stride,
                                            output_stride,
                                            output_label,
                                        ),
                                        "symbol": SYMBOL,
                                        "venue": VENUE,
                                        "source_path": str(source_path),
                                        "source_path_basename": source_path.name,
                                        "bucket_seconds": BUCKET_SECONDS,
                                        "seq_len": seq_len,
                                        "input_stride": input_stride,
                                        "output_stride": output_stride,
                                        "input_feature": input_feature,
                                        "ma_window": ma_window,
                                        "hidden": hidden,
                                        "output_label": output_label,
                                        "input_window_seconds": BUCKET_SECONDS * seq_len * input_stride,
                                        "output_window_seconds": BUCKET_SECONDS * seq_len * output_stride,
                                        "required_source_columns": ";".join(required_columns(input_feature)),
                                        "optional_source_columns": ";".join(OPTIONAL_SOURCE_COLUMNS),
                                        "support_status": status,
                                        "unsupported_reason": reason,
                                        "paper_only": True,
                                        "training": False,
                                        "promotion": False,
                                        "champion_mutation": False,
                                        "orders": False,
                                    }
                                )
    return rows


def write_text(path: Path, rows: list[dict[str, Any]]) -> None:
    supported = [row for row in rows if row["support_status"] == "supported"]
    unsupported = [row for row in rows if row["support_status"] != "supported"]
    by_feature: dict[str, int] = {}
    by_scale: dict[str, int] = {}
    for row in rows:
        by_feature[row["input_feature"]] = by_feature.get(row["input_feature"], 0) + 1
        key = f"is{row['input_stride']}_os{row['output_stride']}"
        by_scale[key] = by_scale.get(key, 0) + 1

    lines = [
        "Rawseq I/O Contract Grid",
        "",
        f"Created at: {now_stamp()}",
        f"Rows: {len(rows)}",
        f"Supported rows: {len(supported)}",
        f"Unsupported rows: {len(unsupported)}",
        f"Symbol: {SYMBOL}",
        f"Venue: {VENUE}",
        f"Source path: {resolve_path(SOURCE_PATH_TEXT)}",
        "",
        "Safety:",
        "  paper_only=true",
        "  training=false",
        "  promotion=false",
        "  champion_mutation=false",
        "  orders=false",
        "",
        "Rows by input feature:",
    ]
    for key in sorted(by_feature):
        lines.append(f"  {key}: {by_feature[key]}")
    lines.append("")
    lines.append("Rows by stride pair:")
    for key in sorted(by_scale):
        lines.append(f"  {key}: {by_scale[key]}")
    if unsupported:
        lines.append("")
        lines.append("Unsupported combinations:")
        for row in unsupported[:50]:
            lines.append(f"  {row['contract_slug']}: {row['unsupported_reason']}")
        if len(unsupported) > 50:
            lines.append(f"  ... {len(unsupported) - 50} more")
    lines.append("")
    lines.append("Warning: this manifest does not train, promote, mutate champions, or place orders.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def output_dir() -> Path:
    root = resolve_path(OUTPUT_DIR_TEXT)
    path = root / f"rawseq_io_contract_grid_{now_stamp()}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def main() -> int:
    rows = build_rows()
    out_dir = output_dir()
    csv_path = out_dir / "rawseq_io_contract_grid.csv"
    txt_path = out_dir / "rawseq_io_contract_grid.txt"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_text(txt_path, rows)
    print("Rawseq I/O contract grid complete")
    print(f"Rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"TXT: {txt_path}")
    print("Safety: no training. No promotion. No champion mutation. No orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
