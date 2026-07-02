import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IS_WINDOWS = os.name == "nt"

SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
SNAPSHOT_VENUE = os.getenv("PRICE_TINY_PINNED_SNAPSHOT_VENUE", "kraken_snapshot_base").strip().lower()
LIVE_VENUE = os.getenv("PRICE_TINY_PINNED_LIVE_VENUE", os.getenv("PRIMARY_VENUE", "kraken")).strip().lower()
FEATURE_GROUPS = os.getenv("PRICE_TINY_FEATURE_GROUPS", "base_tiny_price_v1").strip()
EXPECTED_SCHEMA_HASH = os.getenv("PRICE_TINY_EXPECTED_FEATURE_SCHEMA_HASH", "543c07fec8e33baf").strip()
EXPECTED_FEATURE_COUNT = int(os.getenv("PRICE_TINY_EXPECTED_FEATURE_COUNT", "22"))
MOVE_TARGET_SPEC = os.getenv("PRICE_TINY_PINNED_MOVE_TARGET_SPEC", "move_before_adverse_30s_net_aware").strip()
INSTABILITY_TARGET_SPEC = os.getenv("PRICE_TINY_PINNED_INSTABILITY_TARGET_SPEC", "instability_30s").strip()
MODEL_SPECS = os.getenv("PRICE_TINY_MODEL_SPECS", "ridge_logistic").strip()
OUTPUT_PATH = Path(
    os.getenv(
        "PRICE_TINY_PINNED_OUTPUT_PATH",
        PROJECT_ROOT / "models" / "selected" / SYMBOL / "tiny_price" / LIVE_VENUE / "pinned_snapshot_challenger.json",
    )
)
if not OUTPUT_PATH.is_absolute():
    OUTPUT_PATH = PROJECT_ROOT / OUTPUT_PATH
FORCE_REBUILD_ROWS = os.getenv("PRICE_TINY_PINNED_FORCE_REBUILD_ROWS", "false").strip().lower() in {"1", "true", "yes", "y"}


def npm_command():
    return "npm.cmd" if IS_WINDOWS else "npm"


def run_command(command, env, label):
    print("")
    print(f"=== {label} ===")
    print(" ".join(command))
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(process.stdout)
    if process.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {process.returncode}")
    return process.stdout


def base_env():
    env = os.environ.copy()
    env.update(
        {
            "SYMBOL": SYMBOL,
            "PRICE_TINY_FEATURE_GROUPS": FEATURE_GROUPS,
            "PRICE_TINY_MODEL_SPECS": MODEL_SPECS,
            "PRICE_TINY_TRAIN_ALLOWED_MODEL_TYPES": MODEL_SPECS,
            "PRICE_TINY_ALLOWED_SHADOW_MODEL_TYPES": MODEL_SPECS,
            "PRICE_TINY_AUTO_REGISTER_CHALLENGERS": "false",
            "TRAIN_PRICE_TINY_MODEL": "false",
            "PROMOTE_BEST": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # This final candidate is intentionally trained from the full frozen
    # snapshot unless the caller explicitly caps it.
    env.setdefault("PRICE_TINY_MAX_TRAIN_ROWS", "0")
    env.setdefault("PRICE_TINY_SELECTION_OBJECTIVE", "direction")
    return env


def latest_training_rows_path(metadata_path):
    if not metadata_path.exists():
        raise SystemExit(f"Missing tiny-price latest metadata after build: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    path = Path(payload.get("training_rows_path", ""))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise SystemExit(f"Training rows path from metadata does not exist: {path}")
    return path, payload


def expected_snapshot_rows_path(target_spec):
    slug = "_".join(part for part in FEATURE_GROUPS.split(",") if part).replace("+", "_")
    return (
        PROJECT_ROOT
        / "data"
        / "realtime"
        / SNAPSHOT_VENUE
        / f"{SYMBOL}_tiny_price_training_rows__{slug}__{target_spec}__30s__{EXPECTED_SCHEMA_HASH}.csv"
    )


def parse_candidate_model_path(stdout):
    matches = re.findall(r"candidate_model_path=(.+)", stdout)
    if not matches:
        raise SystemExit("Could not find candidate_model_path=... in tiny-price train output.")
    path = Path(matches[-1].strip())
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise SystemExit(f"Candidate model path was printed but does not exist: {path}")
    return path


def artifact_target_name(artifact):
    target_spec = artifact.get("target_spec", "")
    if isinstance(target_spec, dict):
        return str(target_spec.get("name", ""))
    return str(target_spec)


def artifact_feature_groups(artifact):
    feature_spec = artifact.get("feature_spec", {})
    groups = []
    if isinstance(feature_spec, dict):
        groups = feature_spec.get("enabled_feature_groups", [])
    if not groups:
        raw = artifact.get("feature_groups", artifact.get("enabled_feature_groups", ""))
        if isinstance(raw, str):
            groups = [part.strip() for part in raw.split(",") if part.strip()]
        elif isinstance(raw, list):
            groups = raw
    return [str(group).strip() for group in groups if str(group).strip()]


def load_artifact(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def smoke_check_model(path, expected_target):
    artifact = load_artifact(path)
    errors = []
    if str(artifact.get("symbol", "")).upper() != SYMBOL:
        errors.append(f"symbol mismatch: {artifact.get('symbol')}")
    if str(artifact.get("primary_venue", "")).lower() != LIVE_VENUE:
        errors.append(f"primary_venue mismatch: {artifact.get('primary_venue')} expected {LIVE_VENUE}")
    if artifact_target_name(artifact) != expected_target:
        errors.append(f"target_spec mismatch: {artifact_target_name(artifact)} expected {expected_target}")
    if str(artifact.get("feature_schema_hash", "")) != EXPECTED_SCHEMA_HASH:
        errors.append(f"schema hash mismatch: {artifact.get('feature_schema_hash')} expected {EXPECTED_SCHEMA_HASH}")
    groups = artifact_feature_groups(artifact)
    expected_groups = [part.strip() for part in FEATURE_GROUPS.split(",") if part.strip()]
    if groups != expected_groups:
        errors.append(f"feature groups mismatch: {groups} expected {expected_groups}")
    feature_columns = artifact.get("feature_columns", [])
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        errors.append(f"feature count mismatch: {len(feature_columns)} expected {EXPECTED_FEATURE_COUNT}")
    if errors:
        raise SystemExit("Artifact smoke check failed:\n- " + "\n- ".join(errors))
    return artifact


def smoke_check_live_features(path):
    # Import after setting env so helper modules resolve the live venue paths.
    os.environ["SYMBOL"] = SYMBOL
    os.environ["PRIMARY_VENUE"] = LIVE_VENUE
    from show_tiny_price_prediction import build_current_features, read_csv  # pylint: disable=import-outside-toplevel

    artifact = load_artifact(path)
    snapshot_path = PROJECT_ROOT / "data" / "realtime" / LIVE_VENUE / f"{SYMBOL}_10s_flow.csv"
    snapshots = read_csv(snapshot_path)
    if len(snapshots) == 0:
        raise SystemExit(f"Live feature smoke test needs at least one snapshot row: {snapshot_path}")
    feature_row = build_current_features(snapshots, artifact)
    if len(feature_row) == 0:
        raise SystemExit(f"Could not build a live feature row for {path}")
    missing = list(feature_row.attrs.get("missing_model_feature_columns_before_fill", []))
    if missing:
        raise SystemExit(f"Live feature smoke check failed; missing_before_fill={len(missing)} columns={missing[:20]}")
    values = feature_row[artifact["feature_columns"]].replace([np.inf, -np.inf], np.nan)
    nonfinite = [column for column in artifact["feature_columns"] if pd.to_numeric(values[column], errors="coerce").isna().any()]
    if nonfinite:
        raise SystemExit(f"Live feature smoke check failed; non-finite columns={nonfinite[:20]}")
    return {
        "snapshot_path": str(snapshot_path),
        "timestamp": int(feature_row["timestamp"].iloc[0]),
        "feature_count": len(artifact["feature_columns"]),
        "missing_before_fill": 0,
    }


def build_rows_for_target(target_spec):
    expected_path = expected_snapshot_rows_path(target_spec)
    if expected_path.exists() and not FORCE_REBUILD_ROWS:
        print("")
        print(f"=== reuse frozen rows for {target_spec} ===")
        print(f"Training rows already exist: {expected_path}")
        metadata = {
            "symbol": SYMBOL,
            "primary_venue": SNAPSHOT_VENUE,
            "training_rows_path": str(expected_path),
            "feature_groups": FEATURE_GROUPS,
            "target_spec": target_spec,
            "feature_schema_hash": EXPECTED_SCHEMA_HASH,
            "reused_existing_rows": True,
        }
        return expected_path, metadata
    env = base_env()
    env["PRIMARY_VENUE"] = SNAPSHOT_VENUE
    env["PRICE_TINY_TARGET_SPEC"] = target_spec
    run_command([npm_command(), "run", "tiny-price-build"], env, f"build frozen rows for {target_spec}")
    metadata_path = PROJECT_ROOT / "data" / "realtime" / SNAPSHOT_VENUE / f"{SYMBOL}_tiny_price_training_rows_latest.json"
    return latest_training_rows_path(metadata_path)


def train_target_from_rows(target_spec, training_rows_path):
    env = base_env()
    env["PRIMARY_VENUE"] = LIVE_VENUE
    env["PRICE_TINY_TARGET_SPEC"] = target_spec
    env["PRICE_TINY_TRAINING_ROWS_PATH"] = str(training_rows_path)
    pinned_output_dir = PROJECT_ROOT / "ptp" / SYMBOL / LIVE_VENUE
    pinned_output_dir.mkdir(parents=True, exist_ok=True)
    env["PRICE_TINY_FORWARD_TEST_PREDICTIONS_PATH"] = str(
        pinned_output_dir / f"{SYMBOL}_{target_spec}_forward_test_predictions.csv"
    )
    archive_dir = pinned_output_dir / "forward_test_prediction_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    env["PRICE_TINY_FORWARD_TEST_PREDICTION_ARCHIVE_DIR"] = str(archive_dir)
    stdout = run_command([npm_command(), "run", "tiny-price-train"], env, f"train live-pinned artifact for {target_spec}")
    return parse_candidate_model_path(stdout)


def main():
    snapshot_path = PROJECT_ROOT / "data" / "realtime" / SNAPSHOT_VENUE / f"{SYMBOL}_10s_flow.csv"
    if not snapshot_path.exists():
        raise SystemExit(f"Frozen snapshot source is missing: {snapshot_path}")

    print("Pinned tiny-price challenger trainer")
    print(f"SYMBOL={SYMBOL}")
    print(f"SNAPSHOT_VENUE={SNAPSHOT_VENUE}")
    print(f"LIVE_VENUE={LIVE_VENUE}")
    print(f"FEATURE_GROUPS={FEATURE_GROUPS}")
    print(f"EXPECTED_SCHEMA_HASH={EXPECTED_SCHEMA_HASH}")
    print(f"MODEL_SPECS={MODEL_SPECS}")
    print("Paper-only. No promotion. No private API. No orders.")

    move_rows_path, move_metadata = build_rows_for_target(MOVE_TARGET_SPEC)
    move_model_path = train_target_from_rows(MOVE_TARGET_SPEC, move_rows_path)
    move_artifact = smoke_check_model(move_model_path, MOVE_TARGET_SPEC)
    move_live_smoke = smoke_check_live_features(move_model_path)

    instability_rows_path, instability_metadata = build_rows_for_target(INSTABILITY_TARGET_SPEC)
    instability_model_path = train_target_from_rows(INSTABILITY_TARGET_SPEC, instability_rows_path)
    instability_artifact = smoke_check_model(instability_model_path, INSTABILITY_TARGET_SPEC)
    instability_live_smoke = smoke_check_live_features(instability_model_path)

    payload = {
        "symbol": SYMBOL,
        "snapshot_venue": SNAPSHOT_VENUE,
        "live_venue": LIVE_VENUE,
        "feature_groups": FEATURE_GROUPS,
        "feature_schema_hash": EXPECTED_SCHEMA_HASH,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "move_target_spec": MOVE_TARGET_SPEC,
        "instability_target_spec": INSTABILITY_TARGET_SPEC,
        "move_model_path": str(move_model_path),
        "move_model_id": move_artifact.get("model_id", ""),
        "instability_model_path": str(instability_model_path),
        "instability_model_id": instability_artifact.get("model_id", ""),
        "move_training_rows_path": str(move_rows_path),
        "instability_training_rows_path": str(instability_rows_path),
        "move_training_metadata": move_metadata,
        "instability_training_metadata": instability_metadata,
        "move_live_smoke": move_live_smoke,
        "instability_live_smoke": instability_live_smoke,
        "intended_live_config": {
            "PRICE_TINY_ENSEMBLE_RUN_ID": "sol_kraken_base_pinned_070_070_long_001",
            "PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD": "0.70",
            "PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD": "0.70",
            "PRICE_TINY_ENSEMBLE_ALLOWED_SIDES": "long",
            "PRICE_TINY_ENSEMBLE_MOVE_MODEL_PATH": str(move_model_path),
            "PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_PATH": str(instability_model_path),
            "PRICE_TINY_ENSEMBLE_FEATURE_GROUPS": FEATURE_GROUPS,
            "PRICE_TINY_ENSEMBLE_FEATURE_SCHEMA_HASH": EXPECTED_SCHEMA_HASH,
        },
        "paper_only": True,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("")
    print("Pinned challenger ready.")
    print(f"move_model_path={move_model_path}")
    print(f"instability_model_path={instability_model_path}")
    print(f"move_model_id={payload['move_model_id']}")
    print(f"instability_model_id={payload['instability_model_id']}")
    print(f"smoke_feature_schema_hash={EXPECTED_SCHEMA_HASH}")
    print(f"smoke_feature_groups={FEATURE_GROUPS}")
    print(f"smoke_feature_count={EXPECTED_FEATURE_COUNT}")
    print("smoke_missing_before_fill=0")
    print(f"pinned_config_path={OUTPUT_PATH}")
    print("")
    print("Live paper env:")
    print(f'$env:PRICE_TINY_ENSEMBLE_MOVE_MODEL_PATH="{move_model_path}"')
    print(f'$env:PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_PATH="{instability_model_path}"')
    print('$env:PRICE_TINY_ENSEMBLE_RUN_ID="sol_kraken_base_pinned_070_070_long_001"')
    print('$env:PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD="0.70"')
    print('$env:PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD="0.70"')
    print('$env:PRICE_TINY_ENSEMBLE_ALLOWED_SIDES="long"')
    print("Paper-only. No private API. No orders. No promotion.")


if __name__ == "__main__":
    main()
