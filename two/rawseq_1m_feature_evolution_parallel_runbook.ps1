# Inert handoff runbook for bounded rawseq1m feature-evolution parallelism.
# Dot-sourcing this file defines functions only. It never starts Python.

function Get-RawseqParallelRepositoryRoot {
    return "F:\AITicker\Misc"
}

function Get-RawseqParallelStatePath {
    return "F:\rsio\rawseq_1m_feature_evolution_parallelism\parallel_runbook_state.json"
}

function Get-RawseqParallelRelevantPaths {
    return @(
        "scripts/tiny/run_rawseq_1m_feature_family_evolution.py",
        "scripts/tiny/rawseq_1m_feature_evolution_parallel.py",
        "scripts/tiny/rawseq_1m_feature_evolution_runtime.py",
        "scripts/tiny/run_rawseq_1m_feature_evolution_parallel_benchmark.py",
        "tests/test_rawseq_1m_feature_family_evolution.py",
        "tests/test_rawseq_1m_feature_evolution_parallel.py",
        "tests/test_rawseq_1m_feature_evolution_pre_candidate.py",
        "docs/rawseq_1m_feature_evolution_parallel_runbook.ps1"
    )
}

function Get-RawseqParallelGitIdentity {
    $repo = Get-RawseqParallelRepositoryRoot
    $branch = (& git -C $repo branch --show-current).Trim()
    $commit = (& git -C $repo rev-parse HEAD).Trim()
    return @{ branch = $branch; commit = $commit }
}

function Test-RawseqParallelRelevantDirty {
    $repo = Get-RawseqParallelRepositoryRoot
    $relevant = Get-RawseqParallelRelevantPaths
    $dirty = @()
    foreach ($line in (& git -C $repo status --short)) {
        if ($line.Length -lt 4) { continue }
        $path = $line.Substring(3).Trim().Replace("\", "/")
        if ($path.Contains(" -> ")) { $path = $path.Split(" -> ")[-1].Trim() }
        if ($relevant -contains $path) { $dirty += $path }
    }
    return @($dirty | Sort-Object -Unique)
}

function Confirm-RawseqParallelLaunch {
    param([switch]$Force, [string]$Prompt = "Start bounded rawseq parallel work?")
    if ($Force) { return $true }
    $answer = Read-Host "$Prompt Type YES to continue"
    return $answer -ceq "YES"
}

function Save-RawseqParallelState {
    param([hashtable]$State)
    $statePath = Get-RawseqParallelStatePath
    $parent = Split-Path -Parent $statePath
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
}

function Get-RawseqParallelState {
    $statePath = Get-RawseqParallelStatePath
    if (-not (Test-Path -LiteralPath $statePath)) {
        Write-Host "No parallel runbook state exists at $statePath"
        return $null
    }
    return Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
}

function Get-RawseqWindowsFailureEvents {
    param(
        [datetime]$StartTime,
        [int]$HoursBefore = 1,
        [int]$HoursAfter = 1
    )
    if (-not $StartTime) {
        $state = Get-RawseqParallelState
        if ($null -eq $state) { throw "Provide -StartTime or create a runbook state first." }
        $StartTime = [datetime]::Parse([string]$state.started_at_utc).ToLocalTime()
    }
    $from = $StartTime.AddHours(-[math]::Abs($HoursBefore))
    $to = $StartTime.AddHours([math]::Abs($HoursAfter))
    Write-Host "Reading Windows Application/System events from $from through $to. This does not launch Python."
    $application = Get-WinEvent -FilterHashtable @{ LogName = "Application"; StartTime = $from; EndTime = $to } -ErrorAction SilentlyContinue |
        Where-Object { $_.Id -in 1000,1001,1026 -or $_.ProviderName -match "Python|Application Error|Windows Error Reporting" }
    $system = Get-WinEvent -FilterHashtable @{ LogName = "System"; StartTime = $from; EndTime = $to } -ErrorAction SilentlyContinue |
        Where-Object { $_.Id -in 41,6008,2004,2019 }
    return @($application + $system) | Sort-Object TimeCreated | Select-Object TimeCreated,LogName,ProviderName,Id,LevelDisplayName,Message
}

function Assert-RawseqParallelLaunchReady {
    param([switch]$AllowDirtyRelevant)
    $dirty = @(Test-RawseqParallelRelevantDirty)
    if ($dirty.Count -gt 0 -and -not $AllowDirtyRelevant) {
        throw "Relevant runner/runtime/test files are dirty: $($dirty -join ', '). Re-run with -AllowDirtyRelevant only after reviewing the exact diff."
    }
    return $dirty
}

function Assert-RawseqPathRootAvailable {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Name
    )
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "$Name is empty. Set the corresponding RAWSEQ_EVOLVE_* path variable or clear the stale one."
    }
    $root = [System.IO.Path]::GetPathRoot($Path)
    if ([string]::IsNullOrWhiteSpace($root)) {
        return
    }
    if (-not (Test-Path -LiteralPath $root)) {
        throw "$Name uses unavailable path root '$root' from '$Path'. Fix the drive letter or clear the environment variable before launching."
    }
}

function Resolve-RawseqOptionalRoot {
    param(
        [string]$Value,
        [string]$Fallback,
        [string]$Name
    )
    $resolved = if ([string]::IsNullOrWhiteSpace($Value)) { $Fallback } else { $Value }
    Assert-RawseqPathRootAvailable -Path $resolved -Name $Name
    return $resolved
}

function Invoke-RawseqMediumPreflight {
    param([switch]$AllowDirtyRelevant)
    $repo = Get-RawseqParallelRepositoryRoot
    $dirty = Assert-RawseqParallelLaunchReady -AllowDirtyRelevant:$AllowDirtyRelevant
    Push-Location $repo
    try {
        Write-Host "Running compile checks and focused synthetic tests only."
        python -m py_compile `
            scripts/tiny/rawseq_1m_feature_evolution_parallel.py `
            scripts/tiny/run_rawseq_1m_feature_family_evolution.py `
            scripts/tiny/run_rawseq_1m_feature_evolution_parallel_benchmark.py `
            tests/test_rawseq_1m_feature_evolution_parallel.py `
            tests/test_rawseq_1m_feature_evolution_pre_candidate.py
        if ($LASTEXITCODE -ne 0) { throw "py_compile failed with exit code $LASTEXITCODE" }
        python -m unittest `
            tests.test_rawseq_1m_feature_evolution_parallel `
            tests.test_rawseq_1m_feature_family_evolution `
            tests.test_rawseq_1m_feature_evolution_pre_candidate
        if ($LASTEXITCODE -ne 0) { throw "focused tests failed with exit code $LASTEXITCODE" }
        git diff --check
        if ($LASTEXITCODE -ne 0) { throw "git diff --check failed with exit code $LASTEXITCODE" }
        [pscustomobject]@{ status = "PASS"; relevant_dirty_paths = $dirty; recorded_history_benchmark_started = $false }
    }
    finally {
        Pop-Location
    }
}

function Start-RawseqParallelProcess {
    param(
        [string]$Mode,
        [string]$OutputPath,
        [string]$CheckpointPath,
        [string]$CachePath,
        [string]$LogPath,
        [string[]]$Arguments,
        [hashtable]$Environment,
        [switch]$Force,
        [switch]$AllowDirtyRelevant
    )
    $repo = Get-RawseqParallelRepositoryRoot
    $dirty = Assert-RawseqParallelLaunchReady -AllowDirtyRelevant:$AllowDirtyRelevant
    $identity = Get-RawseqParallelGitIdentity
    $commandText = "python " + ($Arguments -join " ")
    $contract = [ordered]@{
        mode = $Mode
        command = $commandText
        output_path = $OutputPath
        checkpoint_path = $CheckpointPath
        cache_path = $CachePath
        log_path = $LogPath
        branch = $identity.branch
        commit = $identity.commit
        candidate_workers = $Environment.RAWSEQ_EVOLVE_CANDIDATE_WORKERS
        stage_workers = $Environment.RAWSEQ_EVOLVE_STAGE_WORKERS
        blas_threads_per_worker = $Environment.RAWSEQ_EVOLVE_BLAS_THREADS_PER_WORKER
        parallel_memory_budget_gb = $Environment.RAWSEQ_EVOLVE_PARALLEL_MEMORY_BUDGET_GB
        max_worker_private_gb = $Environment.RAWSEQ_EVOLVE_MAX_WORKER_PRIVATE_GB
        parallel_prefetch = $Environment.RAWSEQ_EVOLVE_PARALLEL_PREFETCH
        stage_preset = $Environment.RAWSEQ_EVOLVE_STAGE_PRESET
        private_api = $false
        orders = $false
        promotion = $false
        freeze = $false
        champion_mutation = $false
        dashboard_mutation = $false
        future_shadow_mutation = $false
    }
    Write-Host "Resolved command: $commandText"
    $contract | Format-List | Out-Host
    if (-not (Confirm-RawseqParallelLaunch -Force:$Force -Prompt "Launch $Mode?")) {
        Write-Host "Launch cancelled."
        return
    }
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
    New-Item -ItemType Directory -Path (Split-Path -Parent $LogPath) -Force | Out-Null
    $previous = @{}
    foreach ($name in $Environment.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
        [Environment]::SetEnvironmentVariable($name, [string]$Environment[$name], "Process")
    }
    try {
        $process = Start-Process -FilePath "python" -ArgumentList $Arguments -WorkingDirectory $repo -RedirectStandardOutput $LogPath -RedirectStandardError "$LogPath.stderr" -WindowStyle Hidden -PassThru
    }
    finally {
        foreach ($name in $Environment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process")
        }
    }
    $state = @{
        status = "running"
        mode = $Mode
        process_id = $process.Id
        output_path = $OutputPath
        checkpoint_path = $CheckpointPath
        cache_path = $CachePath
        log_path = $LogPath
        stderr_path = "$LogPath.stderr"
        branch = $identity.branch
        commit = $identity.commit
        command = $commandText
        contract = $contract
        relevant_dirty_override = [bool]$AllowDirtyRelevant
        relevant_dirty_paths = $dirty
        started_at_utc = [DateTime]::UtcNow.ToString("o")
    }
    Save-RawseqParallelState -State $state
    return [pscustomobject]$state
}

function Start-RawseqParallelBenchmark {
    param([switch]$Force, [switch]$AllowDirtyRelevant, [switch]$SkipFourWorker)
    $root = "F:\rsio\rawseq_1m_feature_evolution_parallelism"
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $output = Join-Path $root "runbook_benchmark_$stamp"
    $cache = "F:\rsio\rawseq_1m_feature_evolution_manual_medium_runs\_stage_prep_cache"
    $log = Join-Path $output "parallel_benchmark_wrapper.log"
    $arguments = @("scripts/tiny/run_rawseq_1m_feature_evolution_parallel_benchmark.py", "--output-root", $output, "--stage-cache", $cache)
    if ($SkipFourWorker) { $arguments += "--skip-four-worker" }
    $environment = @{
        RAWSEQ_EVOLVE_CANDIDATE_WORKERS = "2"
        RAWSEQ_EVOLVE_STAGE_WORKERS = "1"
        RAWSEQ_EVOLVE_BLAS_THREADS_PER_WORKER = "1"
        RAWSEQ_EVOLVE_PARALLEL_MEMORY_BUDGET_GB = "24"
        RAWSEQ_EVOLVE_MAX_WORKER_PRIVATE_GB = "4"
        RAWSEQ_EVOLVE_PARALLEL_PREFETCH = "1"
    }
    Start-RawseqParallelProcess -Mode "bounded_parallel_benchmark" -OutputPath $output -CheckpointPath (Join-Path $output "checkpoints") -CachePath $cache -LogPath $log -Arguments $arguments -Environment $environment -Force:$Force -AllowDirtyRelevant:$AllowDirtyRelevant
}

function Start-RawseqParallelEvolution {
    param(
        [int]$CandidateWorkers = 4,
        [int]$StageWorkers = 1,
        [double]$MemoryBudgetGB = 24,
        [ValidateSet("smoke", "overnight", "full_dev")][string]$StagePreset = "overnight",
        [switch]$Force,
        [switch]$AllowDirtyRelevant
    )
    $root = Resolve-RawseqOptionalRoot -Value $env:RAWSEQ_EVOLVE_OUTPUT_ROOT -Fallback "F:\rsio\rawseq_1m_feature_family_evolution" -Name "RAWSEQ_EVOLVE_OUTPUT_ROOT"
    $checkpointRoot = Resolve-RawseqOptionalRoot -Value $env:RAWSEQ_EVOLVE_CHECKPOINT_ROOT -Fallback $root -Name "RAWSEQ_EVOLVE_CHECKPOINT_ROOT"
    $cacheRootValue = if ($env:RAWSEQ_EVOLVE_STAGE_CACHE_ROOT) { $env:RAWSEQ_EVOLVE_STAGE_CACHE_ROOT } elseif ($env:RAWSEQ_EVOLVE_STAGE_PREP_CACHE_DIR) { $env:RAWSEQ_EVOLVE_STAGE_PREP_CACHE_DIR } else { "F:\rsio\rawseq_1m_feature_evolution_manual_medium_runs\_stage_prep_cache" }
    $cacheRoot = Resolve-RawseqOptionalRoot -Value $cacheRootValue -Fallback "F:\rsio\rawseq_1m_feature_evolution_manual_medium_runs\_stage_prep_cache" -Name "RAWSEQ_EVOLVE_STAGE_CACHE_ROOT"
    $diskSafetyMarginGB = if ($env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_GB) { $env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_GB } else { "10" }
    $diskSafetyMarginFraction = if ($env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_FRACTION) { $env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_FRACTION } else { "0.01" }
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $output = Join-Path $root "parallel_evolution_$stamp"
    $checkpoint = Join-Path (Join-Path $checkpointRoot "parallel_evolution_$stamp") "checkpoints"
    $cache = $cacheRoot
    $log = Join-Path $output "evolution.log"
    $arguments = @("scripts/tiny/run_rawseq_1m_feature_family_evolution.py", "--output-dir", $output, "--checkpoint-dir", $checkpoint)
    $environment = @{
        RAWSEQ_EVOLVE_CANDIDATE_WORKERS = [string]$CandidateWorkers
        RAWSEQ_EVOLVE_STAGE_WORKERS = [string]$StageWorkers
        RAWSEQ_EVOLVE_BLAS_THREADS_PER_WORKER = "1"
        RAWSEQ_EVOLVE_PARALLEL_MEMORY_BUDGET_GB = [string]$MemoryBudgetGB
        RAWSEQ_EVOLVE_MAX_WORKER_PRIVATE_GB = "4"
        RAWSEQ_EVOLVE_PARALLEL_PREFETCH = "1"
        RAWSEQ_EVOLVE_PARALLEL_MEMORY_POLICY = "drain_and_pause"
        RAWSEQ_EVOLVE_RETRY_FAILED_SERIAL = "false"
        RAWSEQ_EVOLVE_STAGE_PREP_CACHE_DIR = $cache
        RAWSEQ_EVOLVE_STAGE_PRESET = $StagePreset
        RAWSEQ_EVOLVE_STAGE_PREP_CACHE_SHARDED = "true"
        RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_GB = $diskSafetyMarginGB
        RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_FRACTION = $diskSafetyMarginFraction
    }
    Start-RawseqParallelProcess -Mode "feature_evolution" -OutputPath $output -CheckpointPath $checkpoint -CachePath $cache -LogPath $log -Arguments $arguments -Environment $environment -Force:$Force -AllowDirtyRelevant:$AllowDirtyRelevant
}

function Invoke-RawseqFullSourceStagePreparationSmoke {
    param([switch]$Force, [switch]$AllowDirtyRelevant)
    $baseRoot = Resolve-RawseqOptionalRoot -Value $env:RAWSEQ_EVOLVE_OUTPUT_ROOT -Fallback "F:\rsio\rawseq_1m_feature_evolution_pre_candidate_hardening" -Name "RAWSEQ_EVOLVE_OUTPUT_ROOT"
    $root = if ($env:RAWSEQ_EVOLVE_OUTPUT_ROOT) { Join-Path $baseRoot "pre_candidate_hardening" } else { $baseRoot }
    $baseCheckpointRoot = Resolve-RawseqOptionalRoot -Value $env:RAWSEQ_EVOLVE_CHECKPOINT_ROOT -Fallback $root -Name "RAWSEQ_EVOLVE_CHECKPOINT_ROOT"
    $checkpointRoot = if ($env:RAWSEQ_EVOLVE_CHECKPOINT_ROOT) { Join-Path $baseCheckpointRoot "pre_candidate_hardening" } else { $baseCheckpointRoot }
    $cacheRootValue = if ($env:RAWSEQ_EVOLVE_STAGE_CACHE_ROOT) { $env:RAWSEQ_EVOLVE_STAGE_CACHE_ROOT } elseif ($env:RAWSEQ_EVOLVE_STAGE_PREP_CACHE_DIR) { $env:RAWSEQ_EVOLVE_STAGE_PREP_CACHE_DIR } else { Join-Path $root "_stage_prep_cache" }
    $cacheRoot = Resolve-RawseqOptionalRoot -Value $cacheRootValue -Fallback (Join-Path $root "_stage_prep_cache") -Name "RAWSEQ_EVOLVE_STAGE_CACHE_ROOT"
    $diskSafetyMarginGB = if ($env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_GB) { $env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_GB } else { "10" }
    $diskSafetyMarginFraction = if ($env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_FRACTION) { $env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_FRACTION } else { "0.01" }
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $output = Join-Path $root "full_source_stage_preparation_$stamp"
    $checkpoint = Join-Path (Join-Path $checkpointRoot "full_source_stage_preparation_$stamp") "checkpoints"
    $cache = $cacheRoot
    $log = Join-Path $output "stage_preparation.log"
    $arguments = @(
        "scripts/tiny/run_rawseq_1m_feature_family_evolution.py",
        "--stage-preparation-only",
        "--output-dir", $output,
        "--checkpoint-dir", $checkpoint,
        "--candidate-workers", "4",
        "--stage-workers", "1",
        "--blas-threads-per-worker", "1"
    )
    $environment = @{
        RAWSEQ_EVOLVE_SOURCE_PATH = "F:\AITicker\Misc\data\binance_public_zips"
        RAWSEQ_EVOLVE_STAGE_PREPARATION_SOURCE_ROWS = "50000"
        RAWSEQ_EVOLVE_STAGE_PREP_CACHE_DIR = $cache
        RAWSEQ_EVOLVE_STAGE_PREP_CACHE_SHARDED = "true"
        RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_GB = $diskSafetyMarginGB
        RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_FRACTION = $diskSafetyMarginFraction
        RAWSEQ_EVOLVE_CANDIDATE_WORKERS = "4"
        RAWSEQ_EVOLVE_STAGE_WORKERS = "1"
        RAWSEQ_EVOLVE_BLAS_THREADS_PER_WORKER = "1"
    }
    Start-RawseqParallelProcess -Mode "full_source_stage_preparation_only" -OutputPath $output -CheckpointPath $checkpoint -CachePath $cache -LogPath $log -Arguments $arguments -Environment $environment -Force:$Force -AllowDirtyRelevant:$AllowDirtyRelevant
}

function Resume-RawseqParallelEvolution {
    param([switch]$Force, [switch]$AllowDirtyRelevant)
    $state = Get-RawseqParallelState
    if ($null -eq $state) { throw "No state is available to resume." }
    if ($state.mode -ne "feature_evolution") { throw "Latest state is not a feature-evolution run." }
    $diskSafetyMarginGB = if ($env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_GB) { $env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_GB } else { "10" }
    $diskSafetyMarginFraction = if ($env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_FRACTION) { $env:RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_FRACTION } else { "0.01" }
    $arguments = @("scripts/tiny/run_rawseq_1m_feature_family_evolution.py", "--resume", "--output-dir", [string]$state.output_path, "--checkpoint-dir", [string]$state.checkpoint_path)
    $environment = @{
        RAWSEQ_EVOLVE_CANDIDATE_WORKERS = [string]$state.contract.candidate_workers
        RAWSEQ_EVOLVE_STAGE_WORKERS = [string]$state.contract.stage_workers
        RAWSEQ_EVOLVE_BLAS_THREADS_PER_WORKER = [string]$state.contract.blas_threads_per_worker
        RAWSEQ_EVOLVE_PARALLEL_MEMORY_BUDGET_GB = [string]$state.contract.parallel_memory_budget_gb
        RAWSEQ_EVOLVE_MAX_WORKER_PRIVATE_GB = [string]$state.contract.max_worker_private_gb
        RAWSEQ_EVOLVE_PARALLEL_PREFETCH = [string]$state.contract.parallel_prefetch
        RAWSEQ_EVOLVE_PARALLEL_MEMORY_POLICY = "drain_and_pause"
        RAWSEQ_EVOLVE_RETRY_FAILED_SERIAL = "false"
        RAWSEQ_EVOLVE_STAGE_PREP_CACHE_DIR = [string]$state.cache_path
        RAWSEQ_EVOLVE_STAGE_PRESET = [string]$state.contract.stage_preset
        RAWSEQ_EVOLVE_STAGE_PREP_CACHE_SHARDED = "true"
        RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_GB = $diskSafetyMarginGB
        RAWSEQ_EVOLVE_DISK_SAFETY_MARGIN_FRACTION = $diskSafetyMarginFraction
    }
    Start-RawseqParallelProcess -Mode "feature_evolution" -OutputPath ([string]$state.output_path) -CheckpointPath ([string]$state.checkpoint_path) -CachePath ([string]$state.cache_path) -LogPath ([string]$state.log_path) -Arguments $arguments -Environment $environment -Force:$Force -AllowDirtyRelevant:$AllowDirtyRelevant
}

function Watch-RawseqParallelEvolution {
    param([int]$Tail = 30)
    $state = Get-RawseqParallelState
    if ($null -eq $state) { return }
    $process = Get-Process -Id $state.process_id -ErrorAction SilentlyContinue
    $telemetryPath = Join-Path ([string]$state.output_path) "process_exit_telemetry.json"
    $telemetry = $null
    if (Test-Path -LiteralPath $telemetryPath) {
        $telemetry = Get-Content -LiteralPath $telemetryPath -Raw | ConvertFrom-Json
    }
    $effectiveStatus = if ($null -ne $process) { "running" } elseif ($null -ne $telemetry) { [string]$telemetry.status } else { "not_running" }
    [pscustomobject]@{
        status = $effectiveStatus
        process_id = $state.process_id
        output_path = $state.output_path
        checkpoint_path = $state.checkpoint_path
        log_path = $state.log_path
        telemetry_path = $telemetryPath
        exit_code = if ($null -eq $telemetry) { $null } else { $telemetry.exit_code }
        last_phase = if ($null -eq $telemetry) { "" } else { $telemetry.last_phase }
    }
    if (Test-Path -LiteralPath $state.log_path) { Get-Content -LiteralPath $state.log_path -Tail $Tail }
}

function Compare-RawseqSerialParallelResults {
    param([string]$PacketPath)
    if (-not $PacketPath) {
        $PacketPath = Get-ChildItem "F:\rsio\rawseq_1m_feature_evolution_parallelism" -Directory -Filter "rawseq_1m_feature_evolution_parallelism_*" |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $PacketPath) { throw "No parallelism packet was found." }
    $benchmark = Join-Path $PacketPath "parallel_benchmark_results.csv"
    $parity = Join-Path $PacketPath "serial_parallel_parity.json"
    if (-not (Test-Path $benchmark) -or -not (Test-Path $parity)) { throw "Packet is missing benchmark or parity artifacts." }
    Import-Csv $benchmark | Sort-Object {[double]$_.wall_clock_seconds} | Format-Table configuration,candidate_workers,stage_workers,wall_clock_seconds,result_parity,peak_combined_private_bytes -AutoSize
    Get-Content $parity -Raw | ConvertFrom-Json
}

Write-Host "Rawseq parallel runbook loaded. Available functions include Invoke-RawseqMediumPreflight, Invoke-RawseqFullSourceStagePreparationSmoke, Start/Resume/Watch-RawseqParallelEvolution, Get-RawseqWindowsFailureEvents."
