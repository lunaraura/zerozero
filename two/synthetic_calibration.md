# Synthetic simulator calibration

For synthetic tiny-price pretraining, prefer calibration-loaded simulation when a historical calibration file is available. This keeps the simulator stochastic, but samples regime probabilities, event frequencies, volatility ranges, liquidity ranges, and fair-value drift/cycle priors from historical SOL behavior.

PowerShell example:

```powershell
$env:SYMBOL="SOLUSDT"
$env:SIM_CALIBRATION_OUTPUT_PATH="data/simulated/calibration/SOLUSDT_kraken_sim_calibration.json"
python scripts/tiny/simulation/calibrate_simulator_from_history.py

$env:SCENARIO="news_shock_up"
$env:SIM_CALIBRATION_PATH="data/simulated/calibration/SOLUSDT_kraken_sim_calibration.json"
$env:SIM_FAIR_VALUE_MODE="historical_wave_calibrated"
$env:SIM_RETURN_INNOVATION_MODE="student_t"
$env:SIM_VOL_PERSISTENCE="0.95"
$env:SIM_EVENT_FREQUENCY_SCALE="1.0"
$env:SIM_REGIME_PERSISTENCE_SCALE="0.5" # optional override
$env:SIM_REGIME_TARGET_DURATION_SECONDS="180" # optional override

npm run sim-market
```

Notes:

- The simulator does not replay historical SOL prices exactly.
- Historical waveform mode only nudges latent fair value; visible trade prices still emerge from the agent/order-book simulation.
- New calibration files store FFT reconstructed fair-value bands separately: `fair_value_waveform_session`, `fair_value_waveform_low`, `fair_value_waveform_mid`, and `fair_value_waveform_high`, plus the grouped `fair_value_waveforms` object. Older calibration files without these keys still load through the legacy single-wave fallback.
- Calm synthetic scenarios intentionally downweight major calibrated events. They should mostly be no-event paths with small fakeout/liquidity variation.
- When calibration includes `spread_depth_regimes`, synthetic book depth is scaled toward historical 10bps depth instead of using the uncalibrated default.
- The simulator now applies realized-depth feedback during the run, so printed `target_10bps_depth` can be compared with realized average bid/ask depth.
- Calibrated runs derive a regime flip interval from historical trend/chop duration. Use `SIM_REGIME_TARGET_DURATION_SECONDS` only when you intentionally want to override that target.
- Each run prints cap-saturation ratios so you can see when the path is pressing against scenario safety bounds.
- Calibration and waveform diagnostics stay in `hidden_*` columns, which tiny-price feature selectors should ignore.
- This is paper-only synthetic data generation. No orders, promotion, or private API behavior.
