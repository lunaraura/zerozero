import json
import os
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower() or "legacy"
PRICE_TINY_THRESHOLD = float(
    os.getenv("PRICE_TINY_THRESHOLD", os.getenv("PRICE_TINY_CONFIDENCE_THRESHOLD", "0.55"))
)
ALLOWED_MODEL_TYPES = {
    value.strip()
    for value in os.getenv("PRICE_TINY_ALLOWED_SHADOW_MODEL_TYPES", "ridge_logistic,logistic_regression").split(",")
    if value.strip()
}
BOOTSTRAP_OVERWRITE = os.getenv("PRICE_TINY_BOOTSTRAP_OVERWRITE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
BOOTSTRAP_OBJECTIVE = os.getenv("PRICE_TINY_BOOTSTRAP_OBJECTIVE", "newest").strip().lower()

CANDIDATE_ROOT = PROJECT_ROOT / "models" / "candidates" / SYMBOL / "tiny_price" / PRIMARY_VENUE
SELECTED_ROOT = PROJECT_ROOT / "models" / "selected" / SYMBOL / "tiny_price" / PRIMARY_VENUE
SELECTED_MODEL_PATH = SELECTED_ROOT / "selected_model.json"
CANDIDATE_REGISTRY_PATH = SELECTED_ROOT / "candidate_registry.json"


def now_iso():
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    return load_json(path)


def atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def candidate_paths_newest_first():
    if not CANDIDATE_ROOT.exists():
        return []
    return sorted(CANDIDATE_ROOT.glob("*/model.json"), key=lambda path: path.parent.name, reverse=True)


def threshold_report_for_registration(artifact):
    metrics = artifact.get("forward_test_metrics", {})
    rows = metrics.get("confidence_threshold_directional_report", [])
    eligible = [
        row
        for row in rows
        if float(row.get("threshold", -1.0)) >= PRICE_TINY_THRESHOLD
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda row: float(row.get("threshold", float("inf"))))


def gate_status(artifact):
    if str(artifact.get("symbol", "")).upper() != SYMBOL:
        return False, "symbol_mismatch", None
    if str(artifact.get("primary_venue", "")).lower() != PRIMARY_VENUE:
        return False, "primary_venue_mismatch", None
    selected_model = str(artifact.get("selected_model_name", ""))
    if selected_model not in ALLOWED_MODEL_TYPES:
        return False, "disallowed_model_type", None
    metrics = artifact.get("forward_test_metrics", {})
    price_useful = bool(metrics.get("price_candidate_useful", False))
    direction_useful = bool(metrics.get("direction_candidate_useful", False))
    if not price_useful and not direction_useful:
        return False, "candidate_not_useful", None
    threshold_row = threshold_report_for_registration(artifact)
    if threshold_row is None:
        return False, f"missing_threshold_report_at_or_above_{PRICE_TINY_THRESHOLD:.2f}", None
    avg_return = float(
        threshold_row.get(
            "avg_return_bps",
            threshold_row.get("avg_realized_return_bps", float("nan")),
        )
    )
    if avg_return <= 0:
        return False, f"threshold_avg_return_not_positive:{avg_return}", threshold_row
    if not bool(threshold_row.get("threshold_stable_candidate", False)):
        return False, "threshold_report_not_stable", threshold_row
    if not bool(threshold_row.get("threshold_interesting", False)):
        return False, "threshold_report_not_interesting", threshold_row
    return True, "passed_bootstrap_gates", threshold_row


def candidate_score(path, artifact, threshold_row):
    if BOOTSTRAP_OBJECTIVE == "return":
        return (
            float(threshold_row.get("avg_return_bps", threshold_row.get("avg_realized_return_bps", 0.0))),
            int(threshold_row.get("rows_kept", 0)),
            str(path.parent.name),
        )
    if BOOTSTRAP_OBJECTIVE == "rows":
        return (
            int(threshold_row.get("rows_kept", 0)),
            float(threshold_row.get("avg_return_bps", threshold_row.get("avg_realized_return_bps", 0.0))),
            str(path.parent.name),
        )
    return (str(path.parent.name),)


def find_candidate():
    passing = []
    skipped = []
    for path in candidate_paths_newest_first():
        try:
            artifact = load_json(path)
        except Exception as error:
            skipped.append((str(path), f"load_error:{error}"))
            continue
        passed, reason, threshold_row = gate_status(artifact)
        if not passed:
            skipped.append((str(path), reason))
            continue
        passing.append((path, artifact, threshold_row, candidate_score(path, artifact, threshold_row)))
    if not passing:
        return None, None, None, skipped
    if BOOTSTRAP_OBJECTIVE in {"return", "rows"}:
        passing = sorted(passing, key=lambda item: item[3], reverse=True)
    # For default "newest", candidate_paths_newest_first already gave us the
    # desired order, so this keeps the first passing candidate.
    path, artifact, threshold_row, _score = passing[0]
    return path, artifact, threshold_row, skipped


def existing_champion_path():
    selected = load_json_if_exists(SELECTED_MODEL_PATH)
    if selected and (selected.get("champion_model_path") or selected.get("model_path")):
        return selected.get("champion_model_path") or selected.get("model_path")
    registry = load_json_if_exists(CANDIDATE_REGISTRY_PATH)
    if registry and registry.get("champion_model_path"):
        return registry.get("champion_model_path")
    return ""


def registry_payload(path, artifact, threshold_row):
    registry = load_json_if_exists(CANDIDATE_REGISTRY_PATH) or {
        "paper_only": True,
        "created_at": now_iso(),
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE,
        "challengers": [],
        "retired_challengers": [],
        "max_active_challengers": int(os.getenv("PRICE_TINY_MAX_ACTIVE_CHALLENGERS", "3")),
    }
    registry["updated_at"] = now_iso()
    registry["champion_model_path"] = str(path)
    registry["champion_model_id"] = artifact.get("model_id", "")
    registry["champion_policy"] = {
        "threshold": PRICE_TINY_THRESHOLD,
        "horizon": int(artifact.get("horizon_seconds", 1)),
        "feature_set": artifact.get("feature_set_name", ""),
        "lookback_profile": artifact.get("lookback_profile", "short"),
        "regime_gate": os.getenv("PRICE_TINY_REGIME_GATE", "no_gate").strip().lower(),
    }
    registry["champion_bootstrap_threshold_report"] = threshold_row
    return registry


def selected_payload(path, artifact, threshold_row):
    return {
        "paper_only": True,
        "symbol": SYMBOL,
        "primary_venue": PRIMARY_VENUE,
        "model_path": str(path),
        "champion_model_path": str(path),
        "model_id": artifact.get("model_id", ""),
        "champion_model_id": artifact.get("model_id", ""),
        "selected_at": now_iso(),
        "bootstrap_reason": "newest passing candidate" if BOOTSTRAP_OBJECTIVE == "newest" else f"best {BOOTSTRAP_OBJECTIVE} passing candidate",
        "selected_model_name": artifact.get("selected_model_name", ""),
        "threshold": PRICE_TINY_THRESHOLD,
        "horizon": int(artifact.get("horizon_seconds", 1)),
        "feature_set": artifact.get("feature_set_name", ""),
        "lookback_profile": artifact.get("lookback_profile", "short"),
        "regime_gate": os.getenv("PRICE_TINY_REGIME_GATE", "no_gate").strip().lower(),
        "threshold_report": threshold_row,
        "selection_source": "bootstrap_tiny_price_champion",
        "selection_note": "Paper-only champion bootstrap. No trades/orders/private API.",
    }


def main():
    print("Tiny price champion bootstrap")
    print(f"SYMBOL: {SYMBOL}")
    print(f"PRIMARY_VENUE: {PRIMARY_VENUE}")
    print(f"Candidate root: {CANDIDATE_ROOT}")
    print(f"Selected model path: {SELECTED_MODEL_PATH}")
    print(f"Registry path: {CANDIDATE_REGISTRY_PATH}")
    print(f"Allowed model types: {', '.join(sorted(ALLOWED_MODEL_TYPES))}")
    print(f"PRICE_TINY_THRESHOLD: {PRICE_TINY_THRESHOLD}")
    existing_champion = existing_champion_path()
    if existing_champion and not BOOTSTRAP_OVERWRITE:
        print("bootstrap_champion=false")
        print("bootstrap_model_path=")
        print("bootstrap_block_reason=champion_already_exists")
        print(f"existing_champion_model_path={existing_champion}")
        print("Set PRICE_TINY_BOOTSTRAP_OVERWRITE=true to replace the selected champion pointer.")
        print("Paper-only. No trades/orders/private API.")
        return
    path, artifact, threshold_row, skipped = find_candidate()
    if path is None:
        print("bootstrap_champion=false")
        print("bootstrap_model_path=")
        reason = "no_passing_candidate"
        if not CANDIDATE_ROOT.exists():
            reason = "candidate_root_missing"
        print(f"bootstrap_block_reason={reason}")
        print("Recent skipped candidates")
        for skipped_path, skipped_reason in skipped[:10]:
            print(f"- {skipped_path}: {skipped_reason}")
        print("Paper-only. No trades/orders/private API.")
        return
    atomic_write_json(selected_payload(path, artifact, threshold_row), SELECTED_MODEL_PATH)
    atomic_write_json(registry_payload(path, artifact, threshold_row), CANDIDATE_REGISTRY_PATH)
    print("bootstrap_champion=true")
    print(f"bootstrap_model_path={path}")
    print("bootstrap_block_reason=")
    print(f"bootstrap_model_id={artifact.get('model_id', '')}")
    print(f"selected_model_name={artifact.get('selected_model_name', '')}")
    print(f"selected_model_json={SELECTED_MODEL_PATH}")
    print(f"candidate_registry_json={CANDIDATE_REGISTRY_PATH}")
    print("Paper-only. No trades/orders/private API.")


if __name__ == "__main__":
    main()
