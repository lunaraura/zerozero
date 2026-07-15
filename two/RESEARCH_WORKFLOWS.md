# Research Workflows

This document gives copy-paste PowerShell workflows for common research loops. Keep them bounded first; scale rows, windows, seeds, and configs only after the smoke path writes coherent artifacts.

For choosing the right script family, read [`PROJECT_SCRIPT_GENERATIONS.md`](PROJECT_SCRIPT_GENERATIONS.md) first. It separates current rawseq1m/future-shadow work from older rawseq 10s, GPU, ladder, and legacy tiny-price workflows.

## Safety Defaults

These workflows are intended for public/recorded data and paper-only reports. They should not place orders, mutate paper champions, or promote models.

Before a long run:

```powershell
git status --short
python -m py_compile scripts/tiny_price_rawseq_path_v1.py
Get-ChildItem scripts/tiny/*.py | ForEach-Object { python -m py_compile $_.FullName }
```

## 1. Generate a Rawseq I/O Contract Grid

```powershell
$env:RAWSEQ_IO_SYMBOL = "SOLUSDT"
$env:RAWSEQ_IO_VENUE = "kraken"
$env:RAWSEQ_IO_SOURCE_PATH = "data/realtime/kraken/SOLUSDT_10s_flow.csv"
$env:RAWSEQ_IO_SEQ_LENS = "60"
$env:RAWSEQ_IO_INPUT_STRIDES = "1"
$env:RAWSEQ_IO_OUTPUT_STRIDES = "1"
$env:RAWSEQ_IO_INPUT_FEATURES = "return,ma_distance"
$env:RAWSEQ_IO_MA_WINDOWS = "60,150"
$env:RAWSEQ_IO_HIDDENS = "4,4"
$env:RAWSEQ_IO_OUTPUT_LABELS = "future_return_path,future_range_envelope_path"
python scripts/tiny/rawseq_io_contract_grid.py
```

Expected outputs are under `data/research/rawseq_io_contract_grids` unless `RAWSEQ_IO_OUTPUT_DIR` overrides it.

## 2. Run a Bounded Rawseq Discovery Smoke

```powershell
$env:RAWSEQ_IO_DISCOVERY_GRID_PATH = "data/research/rawseq_io_contract_grids/rawseq_io_contract_grid.csv"
$env:RAWSEQ_IO_DISCOVERY_MAX_CONTRACTS = "1"
$env:RAWSEQ_WF_MAX_WINDOWS = "1"
$env:RAWSEQ_IO_DISCOVERY_POPULATION = "2"
$env:RAWSEQ_IO_DISCOVERY_GENERATIONS = "1"
$env:RAWSEQ_IO_DISCOVERY_EPOCHS = "2"
$env:RAWSEQ_IO_DISCOVERY_SEEDS = "900"
python scripts/tiny/run_rawseq_io_contract_discovery_batch.py
```

Review `attempts.csv`, `successful_candidates.csv`, `failed_attempts.csv`, `run_state.json`, and per-contract walk-forward outputs before increasing bounds.

## 3. Run Until One Registered Candidate Exists

```powershell
$env:RAWSEQ_DISCOVERY_RUN_UNTIL_SUCCESS = "true"
$env:RAWSEQ_DISCOVERY_SUCCESS_MODE = "registered"
$env:RAWSEQ_DISCOVERY_MIN_OK_CANDIDATES = "1"
$env:RAWSEQ_DISCOVERY_MAX_ATTEMPTS = "20"
$env:RAWSEQ_DISCOVERY_MAX_RUNTIME_SECONDS = "1800"
$env:RAWSEQ_DISCOVERY_RANDOMIZE_SEED = "true"
$env:RAWSEQ_DISCOVERY_START_SEED = "900"
python scripts/tiny/run_rawseq_io_contract_discovery_batch.py
```

Use `quality_gate` only after label metrics and baseline guards are verified.

## 4. Probe a Candidate Model Truthfully

```powershell
$env:RAWSEQ_PROBE_MODEL_PATH = "models/candidates/SOLUSDT/tiny_price_rawseq_path_v1/kraken/<run>/model.json"
$env:SYMBOL = "SOLUSDT"
$env:PRIMARY_VENUE = "kraken"
$env:RAWSEQ_PROBE_THRESHOLD_BPS_LIST = "0,0.1,0.2,0.3,0.5"
$env:RAWSEQ_PROBE_COST_BPS_LIST = "0,0.05,0.1,0.25"
python scripts/tiny/run_rawseq_candidate_shadow_probe.py
```

The probe must use the contract declared in `model.json`, not contaminated champion metadata or frozen-shadow defaults.

## 5. Decide and Register Probe Thresholds

```powershell
$env:RAWSEQ_PROBE_DIR = "data/research/rawseq_candidate_shadow_probes/<probe-folder>"
$env:RAWSEQ_DECISION_THRESHOLD_BPS = "0.1"
$env:RAWSEQ_DECISION_COST_BPS = "0.1"
python scripts/tiny/report_rawseq_candidate_probe_decision.py

python scripts/tiny/evaluate_rawseq_candidate_probe_dynamic_costs.py
python scripts/tiny/report_rawseq_probe_threshold_registry.py
```

Use the threshold registry for candidate selection. A probe can be rejected at one threshold and useful at another.

## 6. Freeze a Research Shadow Candidate

```powershell
$env:RAWSEQ_REGISTRY_DIR = "data/research/rawseq_probe_registry/<latest>"
$env:RAWSEQ_SHADOW_SOURCE_PROBE_DIR = "data/research/rawseq_candidate_shadow_probes/<probe-folder>"
python scripts/tiny/freeze_shadow_candidate.py
```

This creates a research shadow folder, not a `data/paper_champions` folder.

## 7. Forward-Paper a Frozen Shadow Candidate

```powershell
$env:RAWSEQ_SHADOW_DIR = "data/research/rawseq_shadow_candidates/<candidate>"
$env:RAWSEQ_FORWARD_SOURCE_PATH = "data/realtime/kraken/SOLUSDT_10s_flow.csv"
$env:RAWSEQ_FORWARD_COST_BPS = "0.1"
$env:RAWSEQ_FORWARD_REPLAY_MODE = "replay_window"
$env:RAWSEQ_FORWARD_LOOKBACK_ROWS = "200000"
python scripts/tiny/run_frozen_shadow_candidate_forward_paper.py
python scripts/tiny/report_shadow_candidate_forward_comparison.py
```

Use `incremental` for real forward accumulation; use `replay_window` for repeatable report windows.

## 8. Ladder Baseline Smoke and Walk-Forward

```powershell
$env:LADDER_MAX_ROWS = "50000"
$env:LADDER_MAX_CONFIGS = "25"
python scripts/tiny/simulate_path_aware_ladder_baseline.py

python scripts/tiny/validate_ladder_baseline_walkforward.py

$env:LADDER_SWEEP_SMOKE_MODE = "true"
$env:LADDER_MAX_CONFIGS = "25"
python scripts/tiny/sweep_ladder_risk_walkforward.py
```

Read `ladder_grid_results.txt`, `ladder_diagnostics.txt`, and walk-forward classifications before treating a single-slice winner as meaningful.

## 9. Compare Rawseq and Ladder Policies on the Same Slices

```powershell
$env:POLICY_COMPARE_SMOKE_MODE = "true"
$env:POLICY_COMPARE_MAX_SLICES = "2"
$env:POLICY_COMPARE_LADDER_CONTRACT_PATH = "data/research/ladder_risk_walkforward_sweeps/<run>/best_ladder_walkforward_contract.json"
$env:POLICY_COMPARE_RAWSEQ_SHADOW_DIRS = "data/research/rawseq_shadow_candidates/<candidate>"
python scripts/tiny/compare_rawseq_vs_ladder_policies.py
```

Watch for `insufficient_sample`, especially rawseq-gated ladder policies with very few trades.

## 10. Build Behavior Archives and I/O Contract Leaderboards

```powershell
python scripts/tiny/rawseq_candidate_behavior_archive.py
python scripts/tiny/report_rawseq_io_contract_leaderboard.py
```

These reports help avoid retesting known failure modes and distinguish contracts that need freezing, more forward paper, or deprioritization.

## Review Checklist Before Scaling

- `model_contract.json` agrees with `model.json` weights and inferred shapes.
- Label output dimensions match scaler/model output dimensions.
- Envelope high paths are nonnegative and low paths are nonpositive.
- Baseline guard passes for high/low/envelope quality candidates.
- No output path crosses the Windows path guard.
- Reports say paper-only/no-orders/no-promotion/no-champion-mutation.
- Candidate rows survive dynamic cost and rolling stability checks.
