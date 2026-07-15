# Rawseq1m Full-Development Preflight

This sequence hardens and verifies stage preparation without fitting candidates.
The recommended runtime contract is four candidate workers, one stage worker,
and one BLAS thread per worker.

## 1. Load the inert runbook

```powershell
Set-Location "F:\AITicker\Misc"
. .\docs\rawseq_1m_feature_evolution_parallel_runbook.ps1
```

Loading the file defines functions only. It does not open source data or start
Python.

## 2. Run focused preflight checks

```powershell
Invoke-RawseqMediumPreflight
```

This compiles the hardened runtime and runs focused synthetic tests. It does
not launch a recorded-history benchmark.

## 3. Run the bounded all-source stage smoke

```powershell
Invoke-RawseqFullSourceStagePreparationSmoke
```

Type `YES` when prompted. The smoke resolves and readability-checks every
monthly source file, materializes at most 50,000 final rows per symbol, writes
per-symbol checkpoints and a sharded stage cache, and exits before candidate
grid construction or fitting.

## 4. Monitor and inspect

```powershell
Watch-RawseqParallelEvolution
$State = Get-RawseqParallelState

Get-Content "$($State.output_path)\pre_candidate_preflight.json"
Get-Content "$($State.output_path)\stage_preparation_only_summary.json"
Get-Content "$($State.output_path)\process_exit_telemetry.json"
Get-ChildItem "$($State.output_path)\stage_preparation_checkpoints\manifest"
```

Required smoke results:

- `status=STAGE_PREPARATION_OK`
- `prepared_symbol_count=9`
- `candidate_fitting_started=false`
- source readability passes for every resolved file
- cache/output/checkpoint disk guards pass
- stage cache schema is `rawseq_1m_stage_preparation_cache_v2_sharded`
- process exit is `OK` with exit code `0`

## 5. Diagnose a traceback-free Windows exit

```powershell
Get-RawseqWindowsFailureEvents
```

This reads relevant Application events (1000, 1001, 1026) and System events
(41, 6008, 2004, 2019) around the saved run start. It does not launch Python.
Correlate those events with `process_exit_telemetry.json`; a file left in
`RUNNING` indicates termination outside Python before a final handler ran.

## 6. Full-development launch, only after smoke review

Do not run this during hardening verification. After reviewing the smoke and
free disk space, the explicit command is:

```powershell
Start-RawseqParallelEvolution `
    -StagePreset full_dev `
    -CandidateWorkers 4 `
    -StageWorkers 1 `
    -MemoryBudgetGB 24
```

The function prints the complete contract and requires `YES`. Resume with:

```powershell
Resume-RawseqParallelEvolution
```

All paths remain paper/research only. Private API, orders, promotion, champion
mutation, dashboard mutation, and active future-shadow mutation are disabled.
