# Project Script Generations

_Manual update: 2026-07-15. Documentation-only map of the de-facto project versions, current script roles, and which lanes are active versus historical. No training, tuning, freeze, promotion, champion mutation, private API use, or orders were run for this update._

## How To Read This

This repository has grown by research lineage, not by a single product rewrite. The safest way to navigate it is by generation:

- **Current operational lane**: frozen 1m downside-risk paper/shadow and dashboard tooling.
- **Current research lane**: rawseq1m board-member feature-family evolution and confirmation tooling.
- **Historical research lanes**: older 10s rawseq, GPU return/envelope screens, ladder/policy experiments, and legacy tiny price systems.

Scripts marked **current** are the normal entry points for new work. Scripts marked **historical** are still useful for reports, audits, or reproducing old artifacts, but should not be used as the basis for new promotion/freeze work without a fresh audit.

## Safety Defaults

Unless a separate audited freeze/evaluation packet says otherwise, assume every script here is:

- `paper_only=true`
- `private_api=false`
- `orders=false`
- `promotion=false`
- `champion_mutation=false`

The active 1m downside-risk model and active future-shadow candidate must remain untouched by challenger research. New feature-evolution or target-tournament work must not consume accumulating future-shadow labels for selection.

## Generation Map

| Generation | De-facto project version | Primary question | Status | Main output area |
| --- | --- | --- | --- | --- |
| G0 | Legacy Node/browser and early candle predictors | Can we wire basic market data, dashboards, and simple prediction surfaces? | Historical / support | `src/`, `scripts/control_dashboard.py`, `docs/` |
| G1 | Tiny price and microstructure systems | Can small classifiers/regressors predict short-term movement from Kraken flow? | Historical | `data/realtime`, `models/`, `ptp/` |
| G2 | Rawseq 10s SOL path models | Can scalar rawseq models find a short-horizon edge? | Historical, mostly superseded | `data/rawseq_runs`, `data/research/rawseq_candidate_shadow_probes` |
| G3 | Rawseq schema, target lanes, GPU screens | Can sequence targets and feature contracts be made auditable? | Historical tooling plus schema support | `configs/rawseq`, `data/research/rawseq_*` |
| G4 | 1m multisymbol downside-risk classifier | Can pooled CPU models predict normalized downside excursion on public 1m candles? | Current frozen/replay/paper lane | `F:\rsio\rawseq_1m_cross_asset_scout`, `data/realtime/binance_1m_candles_multi` |
| G5 | 1m live-paper dashboard and future-shadow ops | Can frozen 1m models be evaluated prospectively without backfill contamination? | Current operations lane | `data/research/rawseq_downside_risk_*`, `docs/rawseq_1m_risk_dashboard.html` |
| G6 | 1m board-member target and feature-family evolution | Which new board-member outputs/features survive strict validation before confirmation? | Current research lane | `H:\rsio_rawseq\feature_evolution` or configured output root |
| G7 | Ladder, policy, and overlay experiments | Can model scores help risk overlays, ladders, or abstention systems? | Research/historical | `data/research/ladder_*`, `data/research/policy_comparisons` |

## G6 Current Research: Rawseq1m Feature-Family Evolution

Use this lane for unattended development-data research over board-member targets and feature families. It is CPU-only and should not freeze or promote anything.

| Script / doc | Role | Status |
| --- | --- | --- |
| `scripts/tiny/run_rawseq_1m_feature_family_evolution.py` | Main staged feature-family evolution runner. Supports stage preparation, parallel candidates, sharded stage cache, disk preflight, and resume. | Current |
| `scripts/tiny/rawseq_1m_feature_evolution_runtime.py` | Shared runtime: disk checks, sharded cache, telemetry, source readability, atomic write hardening. | Current support |
| `scripts/tiny/rawseq_1m_feature_evolution_parallel.py` | Parallel stage/candidate worker helpers. | Current support |
| `docs/rawseq_1m_feature_evolution_parallel_runbook.ps1` | Inert PowerShell runbook for preflight, stage-prep smoke, launch, resume, watch, and Windows failure-event capture. | Current operator entry |
| `scripts/tiny/run_rawseq_1m_feature_evolution_parallel_benchmark.py` | Bounded benchmark harness for parallel implementation checks. | Diagnostic only |
| `tests/test_rawseq_1m_feature_evolution_pre_candidate.py` | Pre-candidate hardening tests: disk guard, sharded cache parity, source readability, telemetry. | Current tests |
| `tests/test_rawseq_1m_feature_family_evolution.py` | Feature-evolution runner and ranking tests. | Current tests |
| `tests/test_rawseq_1m_feature_evolution_parallel.py` | Parallel helper tests. | Current tests |

Current recommended storage for this lane is an external/high-capacity root, for example:

```powershell
$env:RAWSEQ_EVOLVE_OUTPUT_ROOT = "H:\rsio_rawseq\feature_evolution"
$env:RAWSEQ_EVOLVE_STAGE_CACHE_ROOT = "H:\rsio_rawseq\stage_cache"
$env:RAWSEQ_EVOLVE_CHECKPOINT_ROOT = "H:\rsio_rawseq\checkpoints"
$env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_GB = "50"
```

The normal sequence is:

```powershell
Set-Location "F:\AITicker\Misc"
. .\docs\rawseq_1m_feature_evolution_parallel_runbook.ps1
Invoke-RawseqMediumPreflight -AllowDirtyRelevant
Invoke-RawseqFullSourceStagePreparationSmoke -AllowDirtyRelevant
Watch-RawseqParallelEvolution
```

Only after stage preparation passes should a full feature-evolution run be launched.

## G5 Current Operations: Frozen Downside-Risk Future Shadow

Use this lane to accumulate true-forward evidence for the frozen downside-risk candidate. Do not use it for challenger tuning.

| Script | Role | Status |
| --- | --- | --- |
| `scripts/tiny/run_rawseq_downside_risk_future_shadow_if_due.py` | Due-checked hourly-ish wrapper. Normal operational entry point. | Current operations |
| `scripts/tiny/run_rawseq_downside_risk_future_shadow_advance.py` | Advances true-forward decisions and matured labels when due. | Current operations |
| `scripts/tiny/run_rawseq_downside_risk_future_shadow_cycle.py` | Cycle wrapper around advance/status/health reports. | Current operations |
| `scripts/tiny/run_rawseq_downside_risk_future_paper_shadow.py` | Frozen paper-shadow evaluator. | Current operations |
| `scripts/tiny/report_rawseq_downside_risk_future_shadow_status.py` | Acceptance/status report. | Current operations |
| `scripts/tiny/report_rawseq_downside_risk_future_shadow_operational_health.py` | Source freshness, ledger, parity, and disk health. | Current operations |
| `scripts/tiny/report_rawseq_downside_risk_shadow_contract_parity.py` | Frozen contract parity checks. | Current operations |
| `scripts/tiny/report_rawseq_downside_risk_shadow_source_freshness.py` | Source freshness report. | Current operations |
| `scripts/tiny/report_rawseq_downside_risk_future_shadow_data_freshness.py` | Data freshness report. | Current operations |
| `scripts/tiny/rawseq_future_shadow_lock.py` | Implementation-lock and hashing helper. | Current support |
| `scripts/tiny/refresh_rawseq_downside_risk_shadow_feature_table.py` | Refreshes the frozen candidate feature table, when explicitly needed. | Current support |

Normal command:

```powershell
Set-Location "F:\AITicker\Misc"
python scripts\tiny\run_rawseq_downside_risk_future_shadow_if_due.py
```

Acceptance must wait for both predeclared hard gates: at least 30 calendar days and at least 5,000 non-overlapping h480 labeled outcomes.

## G4 Current Frozen/Replay Lane: 1m Multisymbol Downside-Risk Board

Use this lane for the frozen June holdout packet, console predictor, dashboard, and board-member confirmation reports. Do not retrain or recalibrate frozen candidates from these scripts.

| Script | Role | Status |
| --- | --- | --- |
| `scripts/tiny/evaluate_rawseq_1m_multisymbol_future_holdout.py` | Official frozen holdout evaluator for multisymbol 1m candidate packets. | Current/replay |
| `scripts/tiny/run_rawseq_1m_pooled_candidate_pipeline.py` | Fixed pooled-candidate confirmation wrapper; no new search. | Current/replay |
| `scripts/tiny/freeze_rawseq_1m_cpu_challenger.py` | Freezes a CPU challenger packet after explicit confirmation. | Audited freeze path only |
| `scripts/tiny/predict_rawseq_1m_console.py` | Console predictor for latest/replay 1m downside and companion outputs. | Current operator UX |
| `scripts/tiny/run_rawseq_1m_live_paper_dashboard.py` | Live-paper/replay dashboard state builder. | Current operator UX |
| `scripts/tiny/build_rawseq_1m_risk_dashboard_data.py` | Historical replay dashboard data builder. | Support |
| `scripts/tiny/report_rawseq_1m_calibration_audit.py` | Calibration diagnostics for 1m candidates. | Support |
| `scripts/tiny/report_rawseq_1m_cross_asset_decision.py` | Cross-asset decision report. | Support |
| `scripts/tiny/report_rawseq_1m_cross_asset_fixed_transfer.py` | Fixed transfer report. | Support |
| `scripts/tiny/report_rawseq_1m_cross_asset_panel_scout.py` | Panel scout report. | Historical/support |
| `scripts/tiny/audit_rawseq_1m_holdout_integrity.py` | Holdout integrity audit. | Support |
| `scripts/tiny/audit_rawseq_historical_1m_candles.py` | Historical 1m candle audit. | Support |
| `scripts/tiny/audit_rawseq_historical_1m_multisymbol_inventory.py` | Multisymbol inventory audit. | Support |

NPM aliases:

```powershell
npm run rawseq1m-predict
npm run rawseq1m-predict-all
npm run rawseq1m-live-paper-dashboard
```

## G6 Target Confirmation: New Board-Member Families

Use these after broad tournaments identify a target family worth fixed confirmation. These are not broad search scripts.

| Script | Role | Status |
| --- | --- | --- |
| `scripts/tiny/run_rawseq_1m_board_member_target_feature_tournament.py` | Broad target/feature tournament across board-member output families. | Current research, loose gate |
| `scripts/tiny/report_rawseq_1m_board_tournament_triage.py` | Triage report for board-member tournament outputs. | Current research |
| `scripts/tiny/report_rawseq_1m_methodology_supersession.py` | Documents methodology supersession and branch closure decisions. | Current research |
| `scripts/tiny/run_rawseq_1m_downside_severity_family_confirmation.py` | Fixed downside-severity family confirmation. | Current confirmation |
| `scripts/tiny/run_rawseq_1m_multihorizon_downside_calibrated_confirmation.py` | Fixed multi-horizon downside confirmation. | Current confirmation |
| `scripts/tiny/run_rawseq_1m_upside_excursion_calibrated_confirmation.py` | Fixed upside-excursion confirmation. | Current confirmation |
| `scripts/tiny/run_rawseq_1m_volatility_family_confirmation.py` | Fixed volatility-family confirmation. | Current confirmation |
| `scripts/tiny/run_rawseq_1m_indicator_event_scout.py` | Event-only indicator scout. | Bounded final pass / historical |
| `scripts/tiny/report_rawseq_1m_indicator_event_family_selection.py` | Indicator event-family selection report. | Historical/support |
| `scripts/tiny/run_rawseq_1m_indicator_residual_scout.py` | Indicator residual scout. | Closed historical lane |
| `scripts/tiny/run_rawseq_1m_dual_timescale_indicator_scout.py` | Dual-timescale indicator scout. | Historical/closed unless reopened |
| `scripts/tiny/build_rawseq_1m_indicator_companion_dataset.py` | Indicator companion dataset builder. | Historical/support |
| `scripts/tiny/build_rawseq_1m_indicator_residual_event_dataset.py` | Residual event dataset builder. | Historical/support |
| `scripts/tiny/create_rawseq_1m_indicator_event_july_holdout_contract.py` | Creates explicit July event holdout contract. | Support; do not evaluate unless authorized |
| `scripts/tiny/freeze_rawseq_1m_indicator_event_companion.py` | Freezes indicator event companion after explicit pass. | Audited freeze path only |

## G3 Schema, Target-Lane, and GPU Sequence Research

These scripts built the rawseq schema, feature registry, target-lane tournament, locked sequence datasets, and GPU screens. They are still valuable for reports and new carefully scoped target-lane research, but the old return/range/GRU artifacts should be treated as historical unless revalidated.

| Script | Role | Status |
| --- | --- | --- |
| `scripts/tiny/rawseq_feature_label_registry.py` | Central feature/label registry compatibility layer. | Current support |
| `scripts/tiny/report_rawseq_source_column_inventory.py` | Source-column inventory. | Current support |
| `scripts/tiny/report_rawseq_schema_contracts.py` | Full schema/tensor/lineage contract report. | Current support |
| `scripts/tiny/report_rawseq_feature_diagnostic_registry.py` | Per-column feature diagnostic registry. | Current support |
| `scripts/tiny/report_rawseq_feature_diagnostics.py` | Earlier feature diagnostics. | Historical/support |
| `scripts/tiny/run_rawseq_multi_horizon_indicator_pipeline.py` | Multi-horizon indicator artifact pipeline. | Historical/support |
| `scripts/tiny/report_rawseq_target_lane_baseline_tournament.py` | Target-lane baseline tournament. | Historical/support |
| `scripts/tiny/build_rawseq_locked_bundle_sequence_datasets.py` | Builds locked NPZ sequence datasets from selected manifests. | Historical/support |
| `scripts/tiny/run_rawseq_torch_sequence_benchmark.py` | Torch sequence benchmark across model families. | Historical GPU support |
| `scripts/tiny/run_rawseq_torch_survivor_prediction_ensemble.py` | Survivor-only ensemble from old GPU screens. | Historical; do not use for new selection |
| `scripts/tiny/prepare_rawseq_gpu_sequence_handoff.py` | GPU handoff command generator. | Historical; audit before use |
| `scripts/tiny/report_rawseq_validation_failure_attribution.py` | Explains old GRU baseline-guard failures. | Historical diagnostic |
| `scripts/tiny/report_rawseq_gru_contract_survivors.py` | Old GRU survivor rollup. | Historical diagnostic |
| `scripts/tiny/report_rawseq_low_path_bounded_cpu_reconciliation.py` | Low-path CPU parity/reconciliation report. | Historical diagnostic |
| `scripts/tiny/run_rawseq_low_path_residual_gru_screen.py` | Residual GRU screen. | Closed historical lane |
| `scripts/tiny/freeze_rawseq_low_path_ridge_research_candidate.py` | Low-path ridge freeze helper. | Historical; do not use without fresh review |
| `scripts/tiny/report_rawseq_downside_target_redesign.py` | Downside target redesign report. | Historical/support |
| `scripts/tiny/run_rawseq_canonical_retrain_candidate.py` | Canonical retrain candidate runner. | Historical/support |
| `scripts/tiny/train_rawseq_canonical_baselines.py` | Canonical baseline trainer. | Historical/support |
| `scripts/tiny/build_rawseq_canonical_training_table.py` | Canonical training table builder. | Historical/support |
| `scripts/tiny/generate_rawseq_canonical_oof_predictions.py` | Out-of-fold prediction generator. | Historical/support |

## G2 Rawseq 10s SOL Candidate/Probe/Registry Work

This lane produced many useful guardrails: contract audits, truthful probes, threshold registries, dynamic costs, shadow freeze packets, and forward paper mechanics. It is largely superseded by the 1m downside-risk board for current model work, but many scripts remain good report templates.

| Script | Role | Status |
| --- | --- | --- |
| `scripts/tiny_price_rawseq_path_v1.py` | Original tiny rawseq path trainer/inference. | Historical core |
| `scripts/tiny/run_rawseq_recorded_walkforward_evolution.py` | Recorded-live walk-forward evolution. | Historical/support |
| `scripts/tiny/run_rawseq_io_contract_discovery_batch.py` | Rawseq I/O contract discovery batch. | Historical/support |
| `scripts/tiny/rawseq_io_contract_grid.py` | I/O contract grid generator. | Historical/support |
| `scripts/tiny/run_rawseq_candidate_shadow_probe.py` | Truthful candidate model probe. | Historical/support |
| `scripts/tiny/report_rawseq_candidate_probe_decision.py` | Probe decision report. | Historical/support |
| `scripts/tiny/report_rawseq_probe_threshold_registry.py` | Threshold-aware probe registry. | Historical/support |
| `scripts/tiny/evaluate_rawseq_candidate_probe_dynamic_costs.py` | Dynamic-cost probe evaluation. | Historical/support |
| `scripts/tiny/freeze_shadow_candidate.py` | Research shadow candidate freeze helper. | Historical/support |
| `scripts/tiny/run_frozen_shadow_candidate_forward_paper.py` | Frozen shadow forward paper. | Historical/support |
| `scripts/tiny/report_shadow_candidate_forward_comparison.py` | Forward comparison report. | Historical/support |
| `scripts/tiny/report_rawseq_io_contract_leaderboard.py` | I/O contract leaderboard. | Historical/support |
| `scripts/tiny/report_rawseq_walkforward_contract_survivors.py` | Walk-forward survivor report. | Historical/support |
| `scripts/tiny/rawseq_candidate_behavior_archive.py` | Good/near-miss/bad behavior archive. | Historical/support |
| `scripts/tiny/audit_rawseq_frozen_champion_contract.py` | Frozen champion contract audit. | Historical/support |
| `scripts/tiny/find_rawseq_champion_candidate_artifact.py` | Candidate artifact recovery report. | Historical/support |
| `scripts/tiny/report_rawseq_candidate_contract_inventory.py` | Candidate contract inventory. | Historical/support |
| `scripts/tiny/report_rawseq_overnight_grid_triage.py` | Overnight grid triage. | Historical/support |
| `scripts/tiny/report_rawseq_fixed_policy_block_stability.py` | Fixed policy block stability. | Historical/support |
| `scripts/tiny/report_rawseq_fixed_policy_xasset_confirmation.py` | Cross-symbol fixed-policy confirmation. | Historical/support |
| `scripts/tiny/evaluate_rawseq_probe_prediction_ensemble.py` | Probe prediction averaging. | Historical; no new ensemble search by default |
| `scripts/tiny/rank_rawseq_probe_ensemble_weights.py` | Probe ensemble weight ranking. | Historical |
| `scripts/tiny/rank_rawseq_probe_ensemble_gates.py` | Gated probe ensemble ranking. | Historical |
| `scripts/tiny/run_rawseq_registry_ensemble_batch.py` | Registry-driven ensemble batch. | Historical |
| `scripts/tiny/report_rawseq_scale_ensemble_candidates.py` | Scale ensemble candidate report. | Historical |
| `scripts/tiny/report_rawseq_champion_lineage.py` | Champion lineage report. | Historical/support |
| `scripts/tiny/report_rawseq_run_health.py` | Rawseq run health report. | Historical/support |
| `scripts/tiny/report_rawseq_frozen_shadow_summary.py` | Frozen shadow summary. | Historical/support |
| `scripts/tiny/summarize_rawseq_frozen_threshold_sweep.py` | Threshold sweep summary. | Historical/support |
| `scripts/tiny/sweep_rawseq_policies_cost_aware.py` | Cost-aware policy sweep. | Historical/support |

## G7 Ladder, Policy, and Overlay Experiments

These scripts are useful for risk overlay ideas, but no ladder or policy candidate is currently the primary model lineage.

| Script | Role | Status |
| --- | --- | --- |
| `scripts/tiny/simulate_path_aware_ladder_baseline.py` | Risk-controlled ladder/grid baseline simulator. | Historical/research |
| `scripts/tiny/validate_ladder_baseline_walkforward.py` | Ladder walk-forward validation. | Historical/research |
| `scripts/tiny/report_ladder_walkforward_failure_attribution.py` | Ladder failure attribution. | Historical/research |
| `scripts/tiny/sweep_ladder_risk_walkforward.py` | Ladder risk walk-forward sweep. | Historical/research |
| `scripts/tiny/compare_rawseq_vs_ladder_policies.py` | Rawseq vs ladder policy comparison. | Historical/research |

## Trade-Flow and Source-Data Exploration

This lane explores whether Binance/Kraken raw trades or aggTrades can add features or resolve barrier ordering. It is not yet the main feature-evolution input unless explicitly integrated.

| Script | Role | Status |
| --- | --- | --- |
| `scripts/tiny/rawseq_1m_trade_flow_features.py` | Trade-flow feature helpers. | Research support |
| `scripts/tiny/rawseq_1m_trade_flow_evolution.py` | Trade-flow evolution scaffolding. | Research support |
| `scripts/tiny/report_rawseq_1m_trade_flow_source_audit.py` | Trade source audit. | Research support |
| `scripts/tiny/report_rawseq_1m_trade_flow_coverage_audit.py` | Coverage audit. | Research support |
| `scripts/tiny/report_rawseq_1m_trade_flow_evolution_integration.py` | Integration report. | Research support |
| `scripts/tiny/run_rawseq_1m_sol_trade_flow_feasibility.py` | SOL trade-flow feasibility run. | Research support |

## Legacy Tiny / Node / Live System Scripts

These scripts predate the current 1m board-member methodology. Use them only for legacy operation, data recording, or reproducing old reports.

| Area | Representative scripts | Status |
| --- | --- | --- |
| Node/browser app | `main.js`, `src/predict.js`, `scripts/predict_realtime_flow.js` | Legacy/support |
| Legacy dashboard | `scripts/control_dashboard.py` | Legacy/support |
| Live system orchestration | `scripts/run_live_system.py`, `scripts/run_overnight_paper_system.py` | Legacy; audit before use |
| 1s/order-flow loops | `scripts/run_live_1s_order_flow_prediction_loop.py`, `scripts/show_latest_1s_order_flow_prediction.py` | Legacy/research |
| Tiny price training/evaluation | `scripts/build_tiny_price_training_rows.py`, `scripts/evaluate_tiny_price_walk_forward.py`, `scripts/show_tiny_price_prediction.py` | Historical |
| 5m/cross-asset context | `scripts/build_5m_cross_asset_context_training_rows.py`, `scripts/predict_5m_cross_asset_context.py`, `scripts/evaluate_5m_cross_asset_context.py` | Historical/support |
| Technical/regime policies | `scripts/build_tiny_price_technical_features.py`, `scripts/evaluate_tiny_regime_gate_policy.py`, `scripts/evaluate_tiny_technical_policy_f.py` | Historical |

## Simulation Scripts

The simulation folder is separate from the public recorded-data lineages and should not be mixed into freezeable evidence without a clear sim-to-real contract.

| Script | Role | Status |
| --- | --- | --- |
| `scripts/tiny/simulation/simulate_market_microstructure.py` | Synthetic market microstructure simulator. | Historical/research |
| `scripts/tiny/simulation/build_combined_simulated_market_dataset.py` | Simulated dataset builder. | Historical/research |
| `scripts/tiny/simulation/calibrate_simulator_from_history.py` | Simulator calibration from history. | Historical/research |
| `scripts/tiny/simulation/run_simulated_tiny_price_cycle.py` | Simulated tiny-price cycle. | Historical/research |
| `scripts/tiny/simulation/sim2real_distribution_gap_report.py` | Sim-to-real gap report. | Historical/research |
| `scripts/tiny/simulation/synthetic_holdout_report.py` | Synthetic holdout report. | Historical/research |

## What To Run Now

For current project progress, the main command families are:

```powershell
# Future-shadow operation, when due.
python scripts\tiny\run_rawseq_downside_risk_future_shadow_if_due.py

# 1m console/dashboard UX.
npm run rawseq1m-predict
npm run rawseq1m-predict-all
npm run rawseq1m-live-paper-dashboard

# Feature-evolution preflight and stage-prep smoke.
. .\docs\rawseq_1m_feature_evolution_parallel_runbook.ps1
Invoke-RawseqMediumPreflight -AllowDirtyRelevant
Invoke-RawseqFullSourceStagePreparationSmoke -AllowDirtyRelevant
Watch-RawseqParallelEvolution
```

Avoid broad ensemble search, GPU expansion, or freeze work unless the current stage outputs explicitly justify the next fixed confirmation.

