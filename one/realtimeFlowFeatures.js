const fs = require("fs");
const path = require("path");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const CLASS_BEARISH = 0;
const CLASS_NEUTRAL = 1;
const CLASS_BULLISH = 2;
const CLASS_NAMES = ["bearish", "neutral", "bullish"];

const BASE_FEATURE_NAMES = [
  "return_1m",
  "volume",
  "quote_volume",
  "taker_buy_volume",
  "taker_sell_volume",
  "buy_sell_volume_ratio",
  "agg_trade_count",
  "buy_trade_count",
  "sell_trade_count",
  "trade_imbalance",
  "best_bid",
  "best_ask",
  "mid_price",
  "spread",
  "spread_percent",
  "bid_depth_10bps",
  "ask_depth_10bps",
  "order_book_imbalance_10bps",
  "bid_depth_25bps",
  "ask_depth_25bps",
  "order_book_imbalance_25bps"
];

const REQUIRED_NUMERIC_COLUMNS = [
  "timestamp",
  "open",
  "high",
  "low",
  "close",
  "volume",
  "quote_volume",
  "trade_count",
  "taker_buy_volume",
  "taker_sell_volume",
  "taker_buy_ratio",
  "return_1",
  "best_bid",
  "best_ask",
  "mid_price",
  "spread_percent",
  "bid_depth_10bps",
  "ask_depth_10bps",
  "order_book_imbalance_10bps",
  "bid_depth_25bps",
  "ask_depth_25bps",
  "order_book_imbalance_25bps"
];

function getSymbol() {
  return (process.env.SYMBOL || "SOLUSDT").trim().toUpperCase();
}

function getRealtimeCsvPath(symbol = getSymbol()) {
  const outputDir = path.resolve(
    PROJECT_ROOT,
    process.env.OUTPUT_DIR || path.join("data", "realtime")
  );

  return path.join(outputDir, `${symbol}_1m_flow.csv`);
}

function getModelDirectory(symbol = getSymbol()) {
  return path.join(PROJECT_ROOT, "models", "realtime_flow", symbol);
}

function getMetadataPath(modelDirectory) {
  return path.join(modelDirectory, "normalization.json");
}

function parseCsvLine(line) {
  const values = [];
  let value = "";
  let insideQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    const nextCharacter = line[index + 1];

    if (character === '"' && insideQuotes && nextCharacter === '"') {
      value += '"';
      index += 1;
      continue;
    }

    if (character === '"') {
      insideQuotes = !insideQuotes;
      continue;
    }

    if (character === "," && !insideQuotes) {
      values.push(value.trim());
      value = "";
      continue;
    }

    value += character;
  }

  values.push(value.trim());
  return values;
}

function readCsvRows(csvPath) {
  if (!fs.existsSync(csvPath)) {
    throw new Error(
      `Realtime flow CSV not found: ${csvPath}. Run npm run record-realtime first.`
    );
  }

  const lines = fs
    .readFileSync(csvPath, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);

  if (lines.length < 2) {
    throw new Error(`Realtime flow CSV has no data rows: ${csvPath}`);
  }

  const headers = parseCsvLine(lines[0]);

  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    const row = {};

    headers.forEach((header, index) => {
      row[header] = values[index] === undefined ? "" : values[index];
    });

    return row;
  });
}

function toNumber(value) {
  const parsedValue = Number(value);
  return Number.isFinite(parsedValue) ? parsedValue : NaN;
}

function safeRatio(numerator, denominator) {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator)) {
    return 0;
  }

  return Math.abs(denominator) < 1e-12 ? 0 : numerator / denominator;
}

function normalizeRawRow(row) {
  const normalized = {
    time: row.time || "",
    timestamp: toNumber(row.timestamp)
  };

  for (const column of REQUIRED_NUMERIC_COLUMNS) {
    normalized[column] = toNumber(row[column]);
  }

  return normalized;
}

function isValidRealtimeRow(row) {
  if (row.best_bid <= 0 || row.best_ask <= 0 || row.mid_price <= 0) {
    return false;
  }

  return REQUIRED_NUMERIC_COLUMNS.every((column) =>
    Number.isFinite(row[column])
  );
}

function loadValidRealtimeRows(csvPath) {
  const rawRows = readCsvRows(csvPath);
  const normalizedRows = rawRows.map(normalizeRawRow);
  const validRows = normalizedRows
    .filter(isValidRealtimeRow)
    .sort((left, right) => left.timestamp - right.timestamp);

  return {
    rawRowCount: rawRows.length,
    invalidRowCount: rawRows.length - validRows.length,
    rows: validRows
  };
}

function rowToFeatureVector(row) {
  const buySellVolumeRatio = safeRatio(
    row.taker_buy_volume,
    row.taker_sell_volume
  );
  const buyTradeCount = row.trade_count * row.taker_buy_ratio;
  const sellTradeCount = row.trade_count - buyTradeCount;
  const spread = row.best_ask - row.best_bid;

  // These features describe only the current completed 1m row. Labels are
  // calculated from future rows later, so future data never enters this vector.
  return [
    row.return_1,
    row.volume,
    row.quote_volume,
    row.taker_buy_volume,
    row.taker_sell_volume,
    buySellVolumeRatio,
    row.trade_count,
    buyTradeCount,
    sellTradeCount,
    safeRatio(row.taker_buy_volume - row.taker_sell_volume, row.volume),
    row.best_bid,
    row.best_ask,
    row.mid_price,
    spread,
    row.spread_percent,
    row.bid_depth_10bps,
    row.ask_depth_10bps,
    row.order_book_imbalance_10bps,
    row.bid_depth_25bps,
    row.ask_depth_25bps,
    row.order_book_imbalance_25bps
  ];
}

function futureReturnForRow(rows, index, horizon, labelPriceField) {
  const currentPrice = rows[index][labelPriceField];
  const futurePrice = rows[index + horizon][labelPriceField];

  return safeRatio(futurePrice - currentPrice, currentPrice);
}

function futureReturnToClass(futureReturn, threshold) {
  if (futureReturn > threshold) {
    return CLASS_BULLISH;
  }

  if (futureReturn < -threshold) {
    return CLASS_BEARISH;
  }

  return CLASS_NEUTRAL;
}

function buildLabeledExamples(rows, { horizon, threshold, labelPriceField }) {
  const inputs = [];
  const labels = [];
  const futureReturns = [];
  const timestamps = [];
  const times = [];

  for (let index = 0; index + horizon < rows.length; index += 1) {
    const futureReturn = futureReturnForRow(
      rows,
      index,
      horizon,
      labelPriceField
    );

    inputs.push(rowToFeatureVector(rows[index]));
    labels.push(futureReturnToClass(futureReturn, threshold));
    futureReturns.push(futureReturn);
    timestamps.push(rows[index].timestamp);
    times.push(rows[index].time);
  }

  return {
    inputs,
    labels,
    futureReturns,
    timestamps,
    times,
    featureNames: BASE_FEATURE_NAMES
  };
}

function calculateNormalization(inputs) {
  const featureCount = inputs[0].length;
  const means = Array(featureCount).fill(0);
  const stds = Array(featureCount).fill(0);

  for (const row of inputs) {
    row.forEach((value, index) => {
      means[index] += value;
    });
  }

  for (let index = 0; index < featureCount; index += 1) {
    means[index] /= inputs.length;
  }

  for (const row of inputs) {
    row.forEach((value, index) => {
      stds[index] += (value - means[index]) ** 2;
    });
  }

  for (let index = 0; index < featureCount; index += 1) {
    stds[index] = Math.sqrt(stds[index] / inputs.length);
    if (stds[index] < 1e-12 || !Number.isFinite(stds[index])) {
      stds[index] = 1;
    }
  }

  return { means, stds };
}

function applyNormalization(inputs, normalization) {
  return inputs.map((row) =>
    row.map((value, index) => (value - normalization.means[index]) / normalization.stds[index])
  );
}

function oneHot(labels, classCount = 3) {
  return labels.map((label) => {
    const row = Array(classCount).fill(0);
    row[label] = 1;
    return row;
  });
}

function classCounts(labels) {
  const counts = [0, 0, 0];

  for (const label of labels) {
    counts[label] += 1;
  }

  return counts;
}

module.exports = {
  CLASS_BEARISH,
  CLASS_NEUTRAL,
  CLASS_BULLISH,
  CLASS_NAMES,
  BASE_FEATURE_NAMES,
  getSymbol,
  getRealtimeCsvPath,
  getModelDirectory,
  getMetadataPath,
  loadValidRealtimeRows,
  buildLabeledExamples,
  calculateNormalization,
  applyNormalization,
  oneHot,
  classCounts,
  rowToFeatureVector
};
