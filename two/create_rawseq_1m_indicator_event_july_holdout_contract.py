#!/usr/bin/env python3
"""Create July holdout contract for the frozen indicator-event companion."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny.rawseq_1m_baseline_utils import now_stamp, stable_hash, write_json  # noqa: E402

DEFAULT_ROOT = Path(r"F:\rsio\rawseq_1m_indicator_companion_scout")
SYMBOLS = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def latest_freeze(root: Path) -> Path:
    dirs = sorted(root.glob("frozen_indicator_event_companion_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise SystemExit("No frozen indicator-event companion packet found")
    return dirs[0]


def main() -> int:
    root = Path(os.getenv("RAWSEQ_EVENT_FREEZE_ROOT", str(DEFAULT_ROOT)))
    freeze_dir = Path(os.getenv("RAWSEQ_EVENT_FREEZE_DIR", "") or latest_freeze(root))
    contract = read_json(freeze_dir / "indicator_event_companion_contract.json")
    run_dir = root / f"indicator_event_july_holdout_contract_{now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    holdout_contract = {
        "contract_kind": "indicator_event_companion_july_2026_holdout",
        "created_at": now_stamp(),
        "frozen_event_companion_dir": str(freeze_dir),
        "event_companion_contract_hash": contract["companion_contract_hash"],
        "frozen_downside_candidate_hash": contract["frozen_downside_candidate_hash"],
        "holdout_start": "2026-07-01T00:00:00Z",
        "holdout_end": "2026-07-31T23:59:00Z",
        "symbols": SYMBOLS,
        "cadence": "1m",
        "source_pattern": r"F:\AITicker\Misc\data\binance_public_zips\{symbol}-1m-2026-07.zip",
        "july_files_opened": False,
        "july_timestamps_enumerated": False,
        "july_labels_computed": False,
        "july_prevalence_computed": False,
        "july_features_computed": False,
        "july_predictions_computed": False,
        "july_metrics_computed": False,
        "orders": False,
        "promotion": False,
        "champion_mutation": False,
    }
    holdout_contract["holdout_contract_hash"] = stable_hash({k: v for k, v in holdout_contract.items() if k != "holdout_contract_hash"})
    acceptance_rule = {
        "rule_kind": "indicator_event_july_acceptance_rule_v1",
        "created_at": now_stamp(),
        "event_companion_contract_hash": contract["companion_contract_hash"],
        "included_horizons": contract["horizons_minutes"],
        "included_events": contract["event_definitions"],
        "positive_equal_symbol_weighted_brier_skill": "every_included_horizon_or_frozen_family_level_rule",
        "positive_log_loss_improvement": True,
        "positive_pr_auc_lift": True,
        "minimum_positive_symbols": 7,
        "positive_worst_horizon_or_frozen_floor": True,
        "finite_calibration": True,
        "monotonic_output_contract_passes": True,
        "save_reload_parity": True,
        "feature_and_target_contract_parity": True,
        "no_rule_adjustment_after_july_access": True,
        "july_access_before_rule_creation": False,
    }
    acceptance_rule["acceptance_rule_hash"] = stable_hash({k: v for k, v in acceptance_rule.items() if k != "acceptance_rule_hash"})
    ledger = {
        "created_at": now_stamp(),
        "event_companion_contract_hash": contract["companion_contract_hash"],
        "july_files_opened": False,
        "july_timestamps_enumerated": False,
        "july_labels_computed": False,
        "july_prevalence_computed": False,
        "july_features_computed": False,
        "july_predictions_computed": False,
        "july_metrics_computed": False,
    }
    source_expectation = {
        "symbols": SYMBOLS,
        "expected_month": "2026-07",
        "expected_first_timestamp": "2026-07-01T00:00:00Z",
        "expected_last_timestamp": "2026-07-31T23:59:00Z",
        "expected_rows_per_symbol": 44640,
        "source_pattern": holdout_contract["source_pattern"],
        "files_opened_to_create_this_expectation": False,
    }
    write_json(run_dir / "indicator_event_july_holdout_contract.json", holdout_contract)
    write_json(run_dir / "indicator_event_july_acceptance_rule.json", acceptance_rule)
    write_json(run_dir / "indicator_event_july_access_ledger.json", ledger)
    write_json(run_dir / "indicator_event_july_source_expectation.json", source_expectation)
    print(f"july_contract_dir={run_dir}")
    print(f"event_companion_contract_hash={contract['companion_contract_hash']}")
    print(f"july_acceptance_rule_hash={acceptance_rule['acceptance_rule_hash']}")
    print("july_files_opened=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
