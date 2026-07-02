import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IS_WINDOWS = os.name == "nt"

SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
PRIMARY_VENUE = os.getenv("PRIMARY_VENUE", "kraken").strip().lower()
CONFIG_PATH = Path(
    os.getenv(
        "PRICE_TINY_PINNED_CONFIG_PATH",
        PROJECT_ROOT / "models" / "selected" / SYMBOL / "tiny_price" / PRIMARY_VENUE / "pinned_snapshot_challenger.json",
    )
)
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = PROJECT_ROOT / CONFIG_PATH


def npm_command():
    return "npm.cmd" if IS_WINDOWS else "npm"


def load_config():
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Pinned challenger config not found: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise SystemExit(f"Smoke check failed: {message}")


def main():
    config = load_config()
    move_path = Path(config["move_model_path"])
    instability_path = Path(config["instability_model_path"])
    expected_schema = config.get("feature_schema_hash", "543c07fec8e33baf")
    expected_groups = config.get("feature_groups", "base_tiny_price_v1")
    expected_feature_count = int(config.get("feature_count", 22))
    intended = config.get("intended_live_config", {})

    require(move_path.exists(), f"move model path missing: {move_path}")
    require(instability_path.exists(), f"instability model path missing: {instability_path}")

    smoke_output = PROJECT_ROOT / "ptp" / SYMBOL / PRIMARY_VENUE / "pinned_challenger_live_smoke.csv"
    smoke_output.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "SYMBOL": SYMBOL,
            "PRIMARY_VENUE": PRIMARY_VENUE,
            "PRICE_TINY_ENSEMBLE_MOVE_MODEL_PATH": str(move_path),
            "PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_PATH": str(instability_path),
            "PRICE_TINY_ENSEMBLE_MOVE_TARGET_SPEC": config.get("move_target_spec", "move_before_adverse_30s_net_aware"),
            "PRICE_TINY_ENSEMBLE_INSTABILITY_TARGET_SPEC": config.get("instability_target_spec", "instability_30s"),
            "PRICE_TINY_ENSEMBLE_FEATURE_GROUPS": expected_groups,
            "PRICE_TINY_ENSEMBLE_FEATURE_SCHEMA_HASH": expected_schema,
            "PRICE_TINY_ENSEMBLE_RUN_ID": intended.get("PRICE_TINY_ENSEMBLE_RUN_ID", "sol_kraken_base_pinned_070_070_long_001"),
            "PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD": intended.get("PRICE_TINY_ENSEMBLE_MOVE_CONFIDENCE_THRESHOLD", "0.70"),
            "PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD": intended.get("PRICE_TINY_ENSEMBLE_INSTABILITY_THRESHOLD", "0.70"),
            "PRICE_TINY_ENSEMBLE_ALLOWED_SIDES": intended.get("PRICE_TINY_ENSEMBLE_ALLOWED_SIDES", "long"),
            "PRICE_TINY_ENSEMBLE_ENABLE_DIRECTION": "false",
            "PRICE_TINY_ENSEMBLE_ENABLE_REGRESSION": "false",
            "PRICE_TINY_ENSEMBLE_LIVE_PREDICTIONS_PATH": str(smoke_output),
            "PROMOTE_BEST": "false",
            "PRICE_TINY_AUTO_REGISTER_CHALLENGERS": "false",
            "TRAIN_PRICE_TINY_MODEL": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )

    process = subprocess.run(
        [npm_command(), "run", "tiny-price-ensemble-show"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(process.stdout)
    require(process.returncode == 0, f"tiny-price-ensemble-show exited with {process.returncode}")

    output = process.stdout
    require(f"Move model path pin: {move_path}" in output, "move model pin path was not printed/used")
    require(f"Instability model path pin: {instability_path}" in output, "instability model pin path was not printed/used")
    require("loaded explicit:PRICE_TINY_ENSEMBLE_MOVE_MODEL_PATH" in output, "move model did not load from explicit pinned path")
    require(
        "loaded explicit:PRICE_TINY_ENSEMBLE_INSTABILITY_MODEL_PATH" in output,
        "instability model did not load from explicit pinned path",
    )
    require(f"schema={expected_schema}" in output or f"/ {expected_schema} /" in output, "expected schema hash was not shown")
    require(f"feature_groups={expected_groups}" in output, "expected feature groups were not shown")
    require(f"feature_count={expected_feature_count}" in output, "expected feature count was not shown")
    require("missing_before_fill=0" in output, "missing_before_fill was not zero")
    require("required schema match: True" in output, "required schema match was not true")
    require("model pinning status: pinned" in output, "ensemble did not report pinned model status")

    print("")
    print("Pinned tiny-price challenger smoke check passed.")
    print(f"move_model_path={move_path}")
    print(f"instability_model_path={instability_path}")
    print(f"schema_hash={expected_schema}")
    print(f"feature_groups={expected_groups}")
    print(f"feature_count={expected_feature_count}")
    print("missing_before_fill=0")
    print(f"smoke_output={smoke_output}")
    print("Paper-only. No private API. No orders. No promotion.")


if __name__ == "__main__":
    main()
