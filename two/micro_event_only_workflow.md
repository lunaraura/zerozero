# Paper-only 10s microstructure event-only workflow

Use this when full 10s microstructure candidates fail live-input sanity because regression heads are unstable, but event heads are still useful for scare, spread, liquidity, burst, continuation, and reversal context.

PowerShell:

```powershell
$env:MICRO_TRAIN_EVENT_ONLY="true"
npm run micro-build
npm run micro-train
npm run micro-show
npm run market-stack
```

This trains only event heads. It does not train regression targets, does not save `regression_target_scalers`, and does not place trades, send orders, use private APIs, or promote models automatically.

