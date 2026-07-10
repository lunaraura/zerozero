#!/usr/bin/env python
"""Declarative registry for rawseq input features and output labels."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.getenv("RAWSEQ_REGISTRY_OUTPUT_DIR", PROJECT_ROOT / "data" / "research" / "rawseq_feature_label_registry"))
DEFAULT_SEQ_LEN = int(float(os.getenv("RAWSEQ_REGISTRY_SEQ_LEN", "60")))
DEFAULT_BARRIER_LEVELS_BPS = [5, 10, 20, 40, 80]
DEFAULT_RUNG_ENTRY_LEVELS_BPS = [10, 20, 40, 80]
DEFAULT_TAKE_PROFIT_LEVELS_BPS = [10, 20, 40]
DEFAULT_STOP_LOSS_LEVELS_BPS = [40, 80, 120]
DEFAULT_TIMEOUT_BUCKETS = [60, 180, 360]


@dataclass(frozen=True)
class InputFeatureDefinition:
    feature_name: str
    task_role: str
    required_source_columns: list[str]
    default_windows: list[int] = field(default_factory=list)
    default_thresholds: list[float] = field(default_factory=list)
    leakage_warning: str = ""
    compatible_labels: list[str] = field(default_factory=list)
    description: str = ""

    def row(self) -> dict[str, Any]:
        return {
            "registry_type": "input_feature",
            "name": self.feature_name,
            "label_name": "",
            "feature_name": self.feature_name,
            "task_type": "",
            "task_role": self.task_role,
            "output_dim": "",
            "required_source_columns": ",".join(self.required_source_columns),
            "required_horizon_buckets": "",
            "default_windows": ",".join(str(x) for x in self.default_windows),
            "default_thresholds": ",".join(str(x) for x in self.default_thresholds),
            "compatible_policies": "",
            "compatible_labels": ",".join(self.compatible_labels),
            "leakage_warning": self.leakage_warning,
            "description": self.description,
        }


@dataclass(frozen=True)
class OutputLabelDefinition:
    label_name: str
    task_type: str
    output_dim: int
    required_source_columns: list[str]
    required_horizon_buckets: int
    leakage_warning: str
    compatible_policies: list[str]
    default_thresholds: list[float | str]
    description: str
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        return {
            "registry_type": "output_label",
            "name": self.label_name,
            "label_name": self.label_name,
            "feature_name": "",
            "task_type": self.task_type,
            "task_role": "",
            "output_dim": self.output_dim,
            "required_source_columns": ",".join(self.required_source_columns),
            "required_horizon_buckets": self.required_horizon_buckets,
            "default_windows": "",
            "default_thresholds": ",".join(str(x) for x in self.default_thresholds),
            "compatible_policies": ",".join(self.compatible_policies),
            "compatible_labels": "",
            "leakage_warning": self.leakage_warning,
            "description": self.description,
            **{f"extra_{key}": json.dumps(value, sort_keys=True) for key, value in self.extra_fields.items()},
        }


def input_features() -> list[InputFeatureDefinition]:
    common_labels = [
        "future_return_path",
        "future_high_from_now_bps_path",
        "future_low_from_now_bps_path",
        "future_range_envelope_path",
        "barrier_hit_levels",
        "tp_before_stop_by_rung",
    ]
    return [
        InputFeatureDefinition(
            feature_name="return",
            task_role="price_momentum",
            required_source_columns=["price"],
            default_thresholds=[0.0, 0.1, 0.2, 0.3, 0.5],
            compatible_labels=common_labels,
            leakage_warning="Uses only current and prior bucket prices when built causally.",
            description="Unsigned raw bucket log return in bps from the prior bucket to the current bucket.",
        ),
        InputFeatureDefinition(
            feature_name="signed_bucket_return_bps",
            task_role="side_conditioned_price_momentum",
            required_source_columns=["price", "predicted_side"],
            default_thresholds=[0.0, 0.1, 0.2, 0.3, 0.5],
            compatible_labels=common_labels,
            leakage_warning="Side must be known before the bucket being evaluated; do not derive side from future labels.",
            description="Bucket log return in bps multiplied by the predeclared trade side sign.",
        ),
        InputFeatureDefinition(
            feature_name="ma_distance",
            task_role="mean_reversion_context",
            required_source_columns=["price"],
            default_windows=[60, 150, 300],
            default_thresholds=[0.0, 0.1, 0.2, 0.3, 0.5],
            compatible_labels=common_labels,
            leakage_warning="Moving average window must be trailing only.",
            description="Log distance in bps from current price to trailing moving average.",
        ),
        InputFeatureDefinition(
            feature_name="ma_slope",
            task_role="trend_context",
            required_source_columns=["price"],
            default_windows=[60, 150, 300],
            default_thresholds=[0.0, 0.1, 0.2, 0.3, 0.5],
            compatible_labels=common_labels,
            leakage_warning="Moving average slope must be trailing only.",
            description="Log return in bps of the trailing moving average from the prior bucket to current bucket.",
        ),
        InputFeatureDefinition(
            feature_name="rolling_volatility_bps",
            task_role="risk_scale_context",
            required_source_columns=["price"],
            default_windows=[60, 150, 300],
            compatible_labels=common_labels,
            leakage_warning="Rolling volatility must use prior/current bucket returns only.",
            description="Trailing realized volatility of bucket log returns, expressed in bps.",
        ),
        InputFeatureDefinition(
            feature_name="rolling_range_bps",
            task_role="range_context",
            required_source_columns=["price"],
            default_windows=[60, 150, 300],
            compatible_labels=common_labels,
            leakage_warning="Rolling high/low range must be computed from trailing prices only.",
            description="Trailing rolling price range in log-return bps from window low to high.",
        ),
        InputFeatureDefinition(
            feature_name="distance_to_recent_high_bps",
            task_role="breakout_or_pullback_context",
            required_source_columns=["price"],
            default_windows=[60, 150, 300],
            compatible_labels=common_labels,
            leakage_warning="Recent high must be trailing only and exclude future prices.",
            description="Current price log distance in bps to the trailing rolling high.",
        ),
        InputFeatureDefinition(
            feature_name="distance_to_recent_low_bps",
            task_role="breakout_or_pullback_context",
            required_source_columns=["price"],
            default_windows=[60, 150, 300],
            compatible_labels=common_labels,
            leakage_warning="Recent low must be trailing only and exclude future prices.",
            description="Current price log distance in bps to the trailing rolling low.",
        ),
    ]


def output_labels(seq_len: int = DEFAULT_SEQ_LEN) -> list[OutputLabelDefinition]:
    return [
        OutputLabelDefinition(
            label_name="future_return_path",
            task_type="regression",
            output_dim=seq_len,
            required_source_columns=["price"],
            required_horizon_buckets=seq_len,
            leakage_warning="Labels must be built only after train/test split boundaries are fixed; future prices are labels only.",
            compatible_policies=["direct_gt", "inverse_gt", "direct_lt", "fixed_threshold", "path_final"],
            default_thresholds=[0.0, 0.1, 0.2, 0.3, 0.5],
            description="Existing path label: y[k] is signed/log future return bps from current price to future step k.",
        ),
        OutputLabelDefinition(
            label_name="future_high_from_now_bps_path",
            task_type="regression",
            output_dim=seq_len,
            required_source_columns=["price"],
            required_horizon_buckets=seq_len,
            leakage_warning="Upper envelope uses future highs and must never be mixed into input features.",
            compatible_policies=["max_up_gt", "tp_feasibility", "path_envelope_gate"],
            default_thresholds=[5, 10, 20, 40, 80],
            description="Future upper envelope: y[k] is max log-return bps from current price over future steps 1..k.",
        ),
        OutputLabelDefinition(
            label_name="future_low_from_now_bps_path",
            task_type="regression",
            output_dim=seq_len,
            required_source_columns=["price"],
            required_horizon_buckets=seq_len,
            leakage_warning="Lower envelope uses future lows and must never be mixed into input features.",
            compatible_policies=["max_down_gt", "stop_risk_filter", "path_envelope_gate"],
            default_thresholds=[-5, -10, -20, -40, -80],
            description="Future lower envelope: y[k] is min log-return bps from current price over future steps 1..k.",
        ),
        OutputLabelDefinition(
            label_name="future_range_envelope_path",
            task_type="regression",
            output_dim=2 * seq_len,
            required_source_columns=["price"],
            required_horizon_buckets=seq_len,
            leakage_warning="Concatenated high/low envelopes are future labels only; split before fitting scalers.",
            compatible_policies=["path_envelope_gate", "tp_stop_feasibility", "ladder_gate", "risk_filter"],
            default_thresholds=["up:5", "up:10", "down:-20", "down:-40"],
            description="Multi-output envelope label that concatenates future_high_from_now_bps_path and future_low_from_now_bps_path.",
        ),
        OutputLabelDefinition(
            label_name="barrier_hit_levels",
            task_type="multi_binary",
            output_dim=2 * len(DEFAULT_BARRIER_LEVELS_BPS),
            required_source_columns=["price"],
            required_horizon_buckets=seq_len,
            leakage_warning="Barrier outcomes use the full future horizon and must be labels only.",
            compatible_policies=["hit_up_probability", "hit_down_filter", "barrier_odds", "execution_filter"],
            default_thresholds=[0.5, 0.6, 0.7],
            description=(
                "Classification-style label with hit_up/down fields for default levels "
                "5,10,20,40,80 bps touched within the horizon."
            ),
            extra_fields={
                "barrier_levels_bps": DEFAULT_BARRIER_LEVELS_BPS,
                "fields": [
                    *[f"hit_up_{level}_bps" for level in DEFAULT_BARRIER_LEVELS_BPS],
                    *[f"hit_down_{level}_bps" for level in DEFAULT_BARRIER_LEVELS_BPS],
                ],
            },
        ),
        OutputLabelDefinition(
            label_name="tp_before_stop_by_rung",
            task_type="binary",
            output_dim=(
                len(DEFAULT_RUNG_ENTRY_LEVELS_BPS)
                * len(DEFAULT_TAKE_PROFIT_LEVELS_BPS)
                * len(DEFAULT_STOP_LOSS_LEVELS_BPS)
                * len(DEFAULT_TIMEOUT_BUCKETS)
                * 2
            ),
            required_source_columns=["price"],
            required_horizon_buckets=max(DEFAULT_TIMEOUT_BUCKETS),
            leakage_warning=(
                "Hypothetical entry, TP, stop, and timeout paths are future labels only. "
                "Do not allow ladder simulation state from validation/test to influence training features."
            ),
            compatible_policies=["ladder_tp_odds", "ladder_entry_filter", "tp_before_stop_gate", "execution_filter"],
            default_thresholds=[0.5, 0.6, 0.7],
            description=(
                "Ladder-specific label. For each rung entry, take-profit, stop-loss, and timeout combination, "
                "records entry_filled and whether TP occurs before stop or timeout."
            ),
            extra_fields={
                "rung_entry_levels_bps": DEFAULT_RUNG_ENTRY_LEVELS_BPS,
                "take_profit_levels_bps": DEFAULT_TAKE_PROFIT_LEVELS_BPS,
                "stop_loss_levels_bps": DEFAULT_STOP_LOSS_LEVELS_BPS,
                "timeout_buckets": DEFAULT_TIMEOUT_BUCKETS,
                "field_pattern": "entry_{entry}_tp_{tp}_stop_{stop}_timeout_{timeout}_{entry_filled|tp_before_stop}",
            },
        ),
    ]


def registry_rows(seq_len: int = DEFAULT_SEQ_LEN) -> list[dict[str, Any]]:
    rows = [definition.row() for definition in input_features()]
    rows.extend(definition.row() for definition in output_labels(seq_len))
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return [{key: row.get(key, "") for key in fieldnames} for row in rows]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, features: list[InputFeatureDefinition], labels: list[OutputLabelDefinition]) -> None:
    lines = [
        "Rawseq feature and label registry",
        "",
        "Safety:",
        "  report_only=true",
        "  training=false",
        "  promotion=false",
        "  champion_mutation=false",
        "  orders=false",
        "",
        "Input features:",
    ]
    for item in features:
        windows = ",".join(str(x) for x in item.default_windows) if item.default_windows else "none"
        lines.append(f"  {item.feature_name}: role={item.task_role} windows={windows} columns={','.join(item.required_source_columns)}")
        lines.append(f"    {item.description}")
        lines.append(f"    leakage: {item.leakage_warning}")
    lines.extend(["", "Output labels:"])
    for item in labels:
        thresholds = ",".join(str(x) for x in item.default_thresholds)
        lines.append(
            f"  {item.label_name}: type={item.task_type} output_dim={item.output_dim} "
            f"horizon={item.required_horizon_buckets} thresholds={thresholds}"
        )
        lines.append(f"    policies={','.join(item.compatible_policies)}")
        lines.append(f"    {item.description}")
        lines.append(f"    leakage: {item.leakage_warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    features = input_features()
    labels = output_labels(DEFAULT_SEQ_LEN)
    rows = registry_rows(DEFAULT_SEQ_LEN)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "rawseq_feature_label_registry.csv"
    txt_path = OUTPUT_DIR / "rawseq_feature_label_registry.txt"
    write_csv(csv_path, rows)
    write_text(txt_path, features, labels)

    print("Input features:")
    for item in features:
        print(f"  {item.feature_name}: {item.description}")
    print("")
    print("Output labels:")
    for item in labels:
        print(f"  {item.label_name}: type={item.task_type} output_dim={item.output_dim}")
    print("")
    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
