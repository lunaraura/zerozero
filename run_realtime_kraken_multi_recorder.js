const { spawn } = require("child_process");
const path = require("path");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const DEFAULT_MULTI_RECORDER_SYMBOLS =
  "SOLUSDT,BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT";

function parseSymbols(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean);
}

function main() {
  const explicitMultiSymbols =
    process.env.KRAKEN_MULTI_SYMBOLS ||
    process.env.MULTI_RECORDER_SYMBOLS ||
    process.env.CROSS_ASSET_SYMBOLS ||
    "";
  const respectInheritedSymbols = ["true", "1", "yes", "y"].includes(
    String(process.env.KRAKEN_MULTI_RESPECT_SYMBOLS || "false").trim().toLowerCase()
  );
  const symbols = parseSymbols(
    explicitMultiSymbols ||
      (respectInheritedSymbols ? process.env.SYMBOLS : "") ||
      DEFAULT_MULTI_RECORDER_SYMBOLS
  );

  if (symbols.length === 0) {
    console.error("No symbols configured for Kraken multi recorder.");
    process.exitCode = 1;
    return;
  }

  const env = {
    ...process.env,
    SYMBOL: symbols[0],
    SYMBOLS: symbols.join(","),
    VENUE: "kraken",
    VENUES: "kraken",
    SNAPSHOT_INTERVAL_SECONDS: process.env.SNAPSHOT_INTERVAL_SECONDS || "1",
    RECORDER_COMPACT_CONSOLE: process.env.RECORDER_COMPACT_CONSOLE || "1"
  };

  console.log("Realtime Kraken multi-symbol public recorder");
  console.log(`Symbols: ${symbols.join(", ")}`);
  console.log(`Default symbol set: ${DEFAULT_MULTI_RECORDER_SYMBOLS}`);
  if (process.env.SYMBOLS && !explicitMultiSymbols && !respectInheritedSymbols) {
    console.log(
      `Ignoring inherited SYMBOLS=${process.env.SYMBOLS}; ` +
        "set KRAKEN_MULTI_SYMBOLS or KRAKEN_MULTI_RESPECT_SYMBOLS=true to override."
    );
  }
  console.log("Venue: kraken");
  console.log(`Snapshot interval seconds: ${env.SNAPSHOT_INTERVAL_SECONDS}`);
  console.log(`Compact console: ${env.RECORDER_COMPACT_CONSOLE}`);
  console.log("Output: data/realtime/kraken/<SYMBOL>_10s_flow.csv and <SYMBOL>_1m_flow.csv");
  console.log("Public market data only. Paper-only. No private API. No orders.");

  const child = spawn(process.execPath, [path.join(__dirname, "record_realtime_market_data.js")], {
    cwd: PROJECT_ROOT,
    env,
    stdio: "inherit"
  });

  const stop = () => {
    if (!child.killed) {
      child.kill("SIGINT");
    }
  };

  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exitCode = code ?? 0;
  });
}

main();
