const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const WebSocket = globalThis.WebSocket ?? require("ws");
const PROJECT_ROOT = path.resolve(__dirname, "..");
const SYMBOL = (process.env.SYMBOL || "SOLUSDT").trim().toUpperCase();
const SYMBOLS = (process.env.SYMBOLS || SYMBOL)
  .split(",")
  .map((value) => value.trim().toUpperCase())
  .filter(Boolean);
const VENUES = (process.env.VENUES || process.env.VENUE || "binanceus")
  .split(",")
  .map((value) => value.trim().toLowerCase())
  .filter(Boolean);
const VENUE = (process.env.VENUE || VENUES[0] || "binanceus").trim().toLowerCase();
const LEGACY_BINANCEUS_OUTPUT = ["true", "1", "yes", "y"].includes(
  String(process.env.LEGACY_BINANCEUS_OUTPUT || "false").trim().toLowerCase()
);
const ORDER_BOOK_DEPTH = Number.parseInt(process.env.ORDER_BOOK_DEPTH || "100", 10);
const SNAPSHOT_INTERVAL_SECONDS = Number.parseInt(
  process.env.SNAPSHOT_INTERVAL_SECONDS || "10",
  10
);
const OUTPUT_DIR = path.resolve(
  PROJECT_ROOT,
  process.env.OUTPUT_DIR || path.join("data", "realtime")
);
const VENUE_OUTPUT_DIR =
  VENUE === "binanceus" && LEGACY_BINANCEUS_OUTPUT
    ? OUTPUT_DIR
    : path.join(OUTPUT_DIR, VENUE);
const OUTPUT_PATH = path.join(VENUE_OUTPUT_DIR, `${SYMBOL}_1m_flow.csv`);
const SNAPSHOT_OUTPUT_PATH = path.join(VENUE_OUTPUT_DIR, `${SYMBOL}_10s_flow.csv`);
const STREAM_SYMBOL = SYMBOL.toLowerCase();
const BINANCE_REST_BASE_URL = "https://api.binance.us";
const BINANCE_WS_URL =
  `wss://stream.binance.us:9443/stream?streams=` +
  `${STREAM_SYMBOL}@kline_1m/` +
  `${STREAM_SYMBOL}@aggTrade/` +
  `${STREAM_SYMBOL}@depth@100ms`;

const OUTPUT_COLUMNS = [
  "venue",
  "source_quality",
  "timestamp",
  "time",
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
  "range_percent",
  "upper_wick_percent",
  "lower_wick_percent",
  "close_position_in_range",
  "best_bid",
  "best_ask",
  "spread_percent",
  "mid_price",
  "bid_depth_10bps",
  "ask_depth_10bps",
  "bid_depth_25bps",
  "ask_depth_25bps",
  "order_book_imbalance_10bps",
  "order_book_imbalance_25bps",
  "bid_depth_change_10bps",
  "ask_depth_change_10bps",
  "large_bid_wall_distance",
  "large_ask_wall_distance",
  "large_bid_wall_size",
  "large_ask_wall_size"
];

const SNAPSHOT_SUMMARY_METRICS = [
  {
    key: "spread_percent",
    columnPrefix: "snapshot_spread_percent"
  },
  {
    key: "order_book_imbalance_10bps",
    columnPrefix: "snapshot_imbalance_10bps"
  },
  {
    key: "order_book_imbalance_25bps",
    columnPrefix: "snapshot_imbalance_25bps"
  },
  {
    key: "bid_depth_10bps",
    columnPrefix: "snapshot_bid_depth_10bps"
  },
  {
    key: "ask_depth_10bps",
    columnPrefix: "snapshot_ask_depth_10bps"
  },
  {
    key: "bid_depth_25bps",
    columnPrefix: "snapshot_bid_depth_25bps"
  },
  {
    key: "ask_depth_25bps",
    columnPrefix: "snapshot_ask_depth_25bps"
  },
  {
    key: "market_pressure_10s",
    columnPrefix: "snapshot_market_pressure_10s"
  }
];

for (const metric of SNAPSHOT_SUMMARY_METRICS) {
  OUTPUT_COLUMNS.push(
    `${metric.columnPrefix}_mean`,
    `${metric.columnPrefix}_min`,
    `${metric.columnPrefix}_max`,
    `${metric.columnPrefix}_last`,
    `${metric.columnPrefix}_change`
  );
}

const SNAPSHOT_COLUMNS = [
  "venue",
  "source_quality",
  "timestamp",
  "time",
  "mid_price",
  "best_bid",
  "best_ask",
  "spread_percent",
  "market_buy_volume_10s",
  "market_sell_volume_10s",
  "total_trade_volume_10s",
  "trade_count_10s",
  "market_pressure_10s",
  "bid_depth_10bps",
  "ask_depth_10bps",
  "bid_depth_25bps",
  "ask_depth_25bps",
  "order_book_imbalance_10bps",
  "order_book_imbalance_25bps",
  "bid_depth_change_10bps",
  "ask_depth_change_10bps",
  "imbalance_change_10bps",
  "large_bid_wall_distance",
  "large_ask_wall_distance",
  "large_bid_wall_size",
  "large_ask_wall_size"
];

let lastWrittenTimestamp = 0;
let lastSnapshotWrittenTimestamp = 0;

function migrateCsvHeaderIfNeeded(filePath, columns) {
  const text = fs.readFileSync(filePath, "utf8");
  const lines = text.split(/\r?\n/);
  const existingHeader = lines[0] || "";
  const existingColumns = existingHeader.split(",");

  if (columns.every((column) => existingColumns.includes(column))) {
    return;
  }

  const dataLines = lines.slice(1).filter((line) => line.trim().length > 0);
  const expandedLines = dataLines.map((line) => {
    const cells = line.split(",");
    const rowByColumn = new Map();

    existingColumns.forEach((column, index) => {
      rowByColumn.set(column, cells[index] === undefined ? "" : cells[index]);
    });

    return columns.map((column) => rowByColumn.get(column) || "").join(",");
  });

  fs.writeFileSync(filePath, [columns.join(","), ...expandedLines].join("\n") + "\n");
}

function readLastTimestampFromCsv(filePath) {
  const lines = fs
    .readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);
  const header = lines[0] || "";
  const columns = header.split(",");
  const timestampIndex = columns.indexOf("timestamp");
  const lastLine = lines.length > 1 ? lines[lines.length - 1] : null;

  if (!lastLine || timestampIndex < 0) {
    return 0;
  }

  return Number.parseInt(lastLine.split(",")[timestampIndex], 10) || 0;
}

function ensureCsvFile(filePath, columns) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });

  if (!fs.existsSync(filePath) || fs.statSync(filePath).size === 0) {
    fs.writeFileSync(filePath, `${columns.join(",")}\n`);
    return 0;
  }

  migrateCsvHeaderIfNeeded(filePath, columns);
  return readLastTimestampFromCsv(filePath);
}

function ensureOutputFiles() {
  lastWrittenTimestamp = ensureCsvFile(OUTPUT_PATH, OUTPUT_COLUMNS);
  lastSnapshotWrittenTimestamp = ensureCsvFile(
    SNAPSHOT_OUTPUT_PATH,
    SNAPSHOT_COLUMNS
  );
}

function csvValue(value) {
  if (value === null || value === undefined) {
    return "";
  }

  const text = String(value);

  if (text.includes(",") || text.includes('"') || text.includes("\n")) {
    return `"${text.replace(/"/g, '""')}"`;
  }

  return text;
}

function appendCsvRow(filePath, columns, row, lastTimestamp) {
  if (Number(row.timestamp) <= lastTimestamp) {
    console.log(
      `Skipping duplicate or old row ${row.time}; latest saved timestamp is ${lastTimestamp}.`
    );
    return {
      wrote: false,
      lastTimestamp
    };
  }

  const line = columns.map((column) => csvValue(row[column])).join(",");
  fs.appendFileSync(filePath, `${line}\n`);
  return {
    wrote: true,
    lastTimestamp: Number(row.timestamp)
  };
}

function appendOneMinuteRow(row) {
  const result = appendCsvRow(
    OUTPUT_PATH,
    OUTPUT_COLUMNS,
    row,
    lastWrittenTimestamp
  );

  lastWrittenTimestamp = result.lastTimestamp;
  return result.wrote;
}

function appendSnapshotRow(row) {
  const result = appendCsvRow(
    SNAPSHOT_OUTPUT_PATH,
    SNAPSHOT_COLUMNS,
    row,
    lastSnapshotWrittenTimestamp
  );

  lastSnapshotWrittenTimestamp = result.lastTimestamp;
  return result.wrote;
}

function formatTimestampForLog(timestamp) {
  return timestamp > 0 ? `${timestamp} (${new Date(timestamp).toISOString()})` : "none";
}

function logRecorderFreshness(label, snapshotTimestamp, oneMinuteTimestamp, oneMinutePath) {
  const now = Date.now();
  const currentMinuteStart = Math.floor(now / 60000) * 60000;
  const latestCompletedMinuteStart = currentMinuteStart - 60000;
  const oneMinuteAgeSeconds =
    oneMinuteTimestamp > 0 ? ((now - oneMinuteTimestamp) / 1000).toFixed(1) : "n/a";
  const secondsSinceLatestCompleted =
    latestCompletedMinuteStart > 0
      ? ((now - latestCompletedMinuteStart) / 1000).toFixed(1)
      : "n/a";
  const waitingForCompletedMinute =
    oneMinuteTimestamp <= 0 || oneMinuteTimestamp < latestCompletedMinuteStart;

  console.log(
    `Recorder freshness [${label}]: ` +
      `latest_10s_timestamp=${formatTimestampForLog(snapshotTimestamp)} ` +
      `latest_1m_timestamp=${formatTimestampForLog(oneMinuteTimestamp)} ` +
      `seconds_since_latest_1m=${oneMinuteAgeSeconds} ` +
      `seconds_since_latest_completed_1m=${secondsSinceLatestCompleted} ` +
      `waiting_for_completed_minute=${waitingForCompletedMinute} ` +
      `1m_path=${oneMinutePath}`
  );
}

function toNumber(value, defaultValue = 0) {
  const parsedValue = Number(value);
  return Number.isFinite(parsedValue) ? parsedValue : defaultValue;
}

function safeRatio(numerator, denominator) {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator)) {
    return 0;
  }

  return denominator === 0 ? 0 : numerator / denominator;
}

function formatNumber(value, decimals = 10) {
  if (!Number.isFinite(value)) {
    return "";
  }

  return Number(value.toFixed(decimals));
}

function isFinitePositive(value) {
  return Number.isFinite(value) && value > 0;
}

function validateBookMetrics(metrics) {
  const positiveColumns = [
    ["best_bid", metrics.bestBid],
    ["best_ask", metrics.bestAsk],
    ["mid_price", metrics.midPrice],
    ["bid_depth_10bps", metrics.bidDepth10],
    ["ask_depth_10bps", metrics.askDepth10],
    ["bid_depth_25bps", metrics.bidDepth25],
    ["ask_depth_25bps", metrics.askDepth25]
  ];
  const finiteColumns = [
    ["spread_percent", metrics.spreadPercent],
    ["order_book_imbalance_10bps", metrics.imbalance10],
    ["order_book_imbalance_25bps", metrics.imbalance25]
  ];
  const invalid = [];

  for (const [column, value] of positiveColumns) {
    if (!isFinitePositive(value)) {
      invalid.push(column);
    }
  }

  for (const [column, value] of finiteColumns) {
    if (!Number.isFinite(value)) {
      invalid.push(column);
    }
  }

  if (metrics.bestBid > 0 && metrics.bestAsk > 0 && metrics.bestAsk <= metrics.bestBid) {
    invalid.push("best_ask<=best_bid");
  }

  return {
    valid: invalid.length === 0,
    invalid
  };
}

function formatBookDiagnostics(metrics) {
  return (
    `best_bid=${formatNumber(metrics.bestBid)} ` +
    `best_ask=${formatNumber(metrics.bestAsk)} ` +
    `bid_depth_10bps=${formatNumber(metrics.bidDepth10)} ` +
    `ask_depth_10bps=${formatNumber(metrics.askDepth10)} ` +
    `imbalance_10bps=${formatNumber(metrics.imbalance10)}`
  );
}

function validateSettings() {
  if (!/^[A-Z0-9]+$/.test(SYMBOL)) {
    throw new Error(`SYMBOL must look like SOLUSDT. Received: ${SYMBOL}`);
  }

  if (!Number.isInteger(ORDER_BOOK_DEPTH) || ORDER_BOOK_DEPTH <= 0) {
    throw new Error("ORDER_BOOK_DEPTH must be a positive whole number.");
  }

  if (
    !Number.isInteger(SNAPSHOT_INTERVAL_SECONDS) ||
    SNAPSHOT_INTERVAL_SECONDS <= 0 ||
    SNAPSHOT_INTERVAL_SECONDS > 60
  ) {
    throw new Error(
      "SNAPSHOT_INTERVAL_SECONDS must be a whole number between 1 and 60."
    );
  }

  if (typeof WebSocket === "undefined") {
    throw new Error(
      "This script needs a modern Node.js runtime with built-in WebSocket support."
    );
  }

  if (VENUES.length === 0) {
    throw new Error("VENUES must include at least one venue.");
  }
}

class OrderBook {
  constructor(depthLimit) {
    this.depthLimit = depthLimit;
    this.bids = new Map();
    this.asks = new Map();
    this.lastUpdateId = null;
    this.ready = false;
  }

  reset() {
    this.bids.clear();
    this.asks.clear();
    this.lastUpdateId = null;
    this.ready = false;
  }

  loadSnapshot(snapshot) {
    this.reset();
    this.lastUpdateId = Number(snapshot.lastUpdateId);

    for (const [price, quantity] of snapshot.bids || []) {
      this.setLevel(this.bids, price, quantity);
    }

    for (const [price, quantity] of snapshot.asks || []) {
      this.setLevel(this.asks, price, quantity);
    }

    this.prune();
    this.ready = true;
  }

  setLevel(side, priceText, quantityText) {
    const price = toNumber(priceText, NaN);
    const quantity = toNumber(quantityText, NaN);

    if (!Number.isFinite(price) || !Number.isFinite(quantity)) {
      return;
    }

    if (quantity === 0) {
      side.delete(price);
    } else {
      side.set(price, quantity);
    }
  }

  applyDepthUpdate(update) {
    if (!this.ready) {
      return "not_ready";
    }

    const firstUpdateId = Number(update.U);
    const finalUpdateId = Number(update.u);

    if (finalUpdateId <= this.lastUpdateId) {
      return "old_update";
    }

    if (firstUpdateId > this.lastUpdateId + 1) {
      return "gap";
    }

    for (const [price, quantity] of update.b || []) {
      this.setLevel(this.bids, price, quantity);
    }

    for (const [price, quantity] of update.a || []) {
      this.setLevel(this.asks, price, quantity);
    }

    this.lastUpdateId = finalUpdateId;
    this.prune();
    return "applied";
  }

  prune() {
    this.bids = new Map(
      [...this.bids.entries()]
        .sort((left, right) => right[0] - left[0])
        .slice(0, this.depthLimit)
    );
    this.asks = new Map(
      [...this.asks.entries()]
        .sort((left, right) => left[0] - right[0])
        .slice(0, this.depthLimit)
    );
  }

  getBestBid() {
    const firstBid = [...this.bids.keys()].sort((left, right) => right - left)[0];
    return Number.isFinite(firstBid) ? firstBid : 0;
  }

  getBestAsk() {
    const firstAsk = [...this.asks.keys()].sort((left, right) => left - right)[0];
    return Number.isFinite(firstAsk) ? firstAsk : 0;
  }

  sumDepthWithinBps(side, midPrice, bps) {
    if (!Number.isFinite(midPrice) || midPrice <= 0) {
      return 0;
    }

    const distance = bps / 10000;
    let total = 0;

    if (side === "bid") {
      const minimumBid = midPrice * (1 - distance);

      for (const [price, quantity] of this.bids.entries()) {
        if (price >= minimumBid) {
          total += quantity;
        }
      }
    } else {
      const maximumAsk = midPrice * (1 + distance);

      for (const [price, quantity] of this.asks.entries()) {
        if (price <= maximumAsk) {
          total += quantity;
        }
      }
    }

    return total;
  }

  findLargestWall(side, midPrice) {
    const levels = side === "bid" ? this.bids.entries() : this.asks.entries();
    let wallPrice = 0;
    let wallSize = 0;

    for (const [price, quantity] of levels) {
      if (quantity > wallSize) {
        wallPrice = price;
        wallSize = quantity;
      }
    }

    const distance = midPrice > 0 ? Math.abs(wallPrice - midPrice) / midPrice : 0;

    return {
      distance,
      size: wallSize
    };
  }

  createMetrics(previousMetrics) {
    const bestBid = this.getBestBid();
    const bestAsk = this.getBestAsk();
    const midPrice =
      bestBid > 0 && bestAsk > 0
        ? (bestBid + bestAsk) / 2
        : bestBid > 0
          ? bestBid
          : bestAsk;
    const bidDepth10 = this.sumDepthWithinBps("bid", midPrice, 10);
    const askDepth10 = this.sumDepthWithinBps("ask", midPrice, 10);
    const bidDepth25 = this.sumDepthWithinBps("bid", midPrice, 25);
    const askDepth25 = this.sumDepthWithinBps("ask", midPrice, 25);
    const largeBidWall = this.findLargestWall("bid", midPrice);
    const largeAskWall = this.findLargestWall("ask", midPrice);

    return {
      bestBid,
      bestAsk,
      spreadPercent: safeRatio(bestAsk - bestBid, midPrice),
      midPrice,
      bidDepth10,
      askDepth10,
      bidDepth25,
      askDepth25,
      imbalance10: safeRatio(bidDepth10 - askDepth10, bidDepth10 + askDepth10),
      imbalance25: safeRatio(bidDepth25 - askDepth25, bidDepth25 + askDepth25),
      bidDepthChange10: previousMetrics ? bidDepth10 - previousMetrics.bidDepth10 : 0,
      askDepthChange10: previousMetrics ? askDepth10 - previousMetrics.askDepth10 : 0,
      largeBidWallDistance: largeBidWall.distance,
      largeAskWallDistance: largeAskWall.distance,
      largeBidWallSize: largeBidWall.size,
      largeAskWallSize: largeAskWall.size
    };
  }
}

const orderBook = new OrderBook(ORDER_BOOK_DEPTH);
let depthBuffer = [];
let loadingSnapshot = false;
let previousClose = null;
let previousBookMetrics = null;
let previousSnapshotMetrics = null;
const aggregateTradeBuckets = new Map();
let snapshotTradeBucket = {
  marketBuyVolume: 0,
  marketSellVolume: 0,
  totalTradeVolume: 0,
  tradeCount: 0
};
const snapshotRowsByMinute = new Map();
let snapshotTimer = null;

async function loadOrderBookSnapshot() {
  if (loadingSnapshot) {
    return;
  }

  loadingSnapshot = true;

  try {
    const url =
      `${BINANCE_REST_BASE_URL}/api/v3/depth?` +
      `symbol=${encodeURIComponent(SYMBOL)}&limit=${ORDER_BOOK_DEPTH}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const snapshot = await response.json();
    orderBook.loadSnapshot(snapshot);
    applyBufferedDepthUpdates();
    console.log(
      `Loaded ${SYMBOL} order book snapshot at update ${orderBook.lastUpdateId}.`
    );
  } catch (error) {
    console.error(`Order book snapshot failed: ${error.message}`);
    orderBook.ready = false;
    setTimeout(loadOrderBookSnapshot, 5000);
  } finally {
    loadingSnapshot = false;
  }
}

function applyBufferedDepthUpdates() {
  if (!orderBook.ready) {
    return;
  }

  const bufferedUpdates = depthBuffer;
  depthBuffer = [];

  for (const update of bufferedUpdates) {
    const result = orderBook.applyDepthUpdate(update);

    if (result === "gap") {
      console.warn("Depth stream gap while applying buffer; reloading snapshot.");
      orderBook.ready = false;
      depthBuffer = [];
      loadOrderBookSnapshot();
      return;
    }
  }
}

function handleDepthUpdate(update) {
  if (!orderBook.ready) {
    depthBuffer.push(update);
    return;
  }

  const result = orderBook.applyDepthUpdate(update);

  if (result === "gap") {
    console.warn("Depth stream gap detected; reloading order book snapshot.");
    orderBook.ready = false;
    depthBuffer = [update];
    loadOrderBookSnapshot();
  }
}

function getMinuteTimestamp(timestamp) {
  return Math.floor(timestamp / 60000) * 60000;
}

function handleAggregateTrade(trade) {
  const minuteTimestamp = getMinuteTimestamp(Number(trade.T || trade.E));
  const quantity = toNumber(trade.q);
  const quoteQuantity = quantity * toNumber(trade.p);
  const bucket = aggregateTradeBuckets.get(minuteTimestamp) || {
    tradeCount: 0,
    takerBuyVolume: 0,
    takerSellVolume: 0,
    quoteVolume: 0
  };

  bucket.tradeCount += 1;
  bucket.quoteVolume += quoteQuantity;

  // In Binance aggregate trades, m=true means the buyer was the maker, so the
  // aggressive/taker side was sell. m=false means taker buy.
  if (trade.m === true) {
    bucket.takerSellVolume += quantity;
    snapshotTradeBucket.marketSellVolume += quantity;
  } else {
    bucket.takerBuyVolume += quantity;
    snapshotTradeBucket.marketBuyVolume += quantity;
  }

  snapshotTradeBucket.totalTradeVolume += quantity;
  snapshotTradeBucket.tradeCount += 1;
  aggregateTradeBuckets.set(minuteTimestamp, bucket);
}

function cleanOldTradeBuckets(latestTimestamp) {
  for (const minuteTimestamp of aggregateTradeBuckets.keys()) {
    if (minuteTimestamp < latestTimestamp - 10 * 60 * 1000) {
      aggregateTradeBuckets.delete(minuteTimestamp);
    }
  }
}

function getSnapshotMinuteTimestamp(timestamp) {
  return Math.floor(timestamp / 60000) * 60000;
}

function rememberSnapshotForMinute(row) {
  const minuteTimestamp = getSnapshotMinuteTimestamp(row.timestamp);
  const rows = snapshotRowsByMinute.get(minuteTimestamp) || [];

  rows.push(row);
  snapshotRowsByMinute.set(minuteTimestamp, rows);

  for (const storedMinuteTimestamp of snapshotRowsByMinute.keys()) {
    if (storedMinuteTimestamp < minuteTimestamp - 10 * 60 * 1000) {
      snapshotRowsByMinute.delete(storedMinuteTimestamp);
    }
  }
}

function createSnapshotRow(timestamp) {
  const metrics = orderBook.createMetrics(previousSnapshotMetrics);
  const marketPressure = safeRatio(
    snapshotTradeBucket.marketBuyVolume - snapshotTradeBucket.marketSellVolume,
    snapshotTradeBucket.totalTradeVolume
  );
  const imbalanceChange10 = previousSnapshotMetrics
    ? metrics.imbalance10 - previousSnapshotMetrics.imbalance10
    : 0;

  const row = {
    venue: VENUE,
    source_quality: "binanceus_book_depth_trade_snapshot",
    timestamp,
    time: new Date(timestamp).toISOString(),
    mid_price: formatNumber(metrics.midPrice),
    best_bid: formatNumber(metrics.bestBid),
    best_ask: formatNumber(metrics.bestAsk),
    spread_percent: formatNumber(metrics.spreadPercent),
    market_buy_volume_10s: formatNumber(snapshotTradeBucket.marketBuyVolume),
    market_sell_volume_10s: formatNumber(snapshotTradeBucket.marketSellVolume),
    total_trade_volume_10s: formatNumber(snapshotTradeBucket.totalTradeVolume),
    trade_count_10s: snapshotTradeBucket.tradeCount,
    market_pressure_10s: formatNumber(marketPressure),
    bid_depth_10bps: formatNumber(metrics.bidDepth10),
    ask_depth_10bps: formatNumber(metrics.askDepth10),
    bid_depth_25bps: formatNumber(metrics.bidDepth25),
    ask_depth_25bps: formatNumber(metrics.askDepth25),
    order_book_imbalance_10bps: formatNumber(metrics.imbalance10),
    order_book_imbalance_25bps: formatNumber(metrics.imbalance25),
    bid_depth_change_10bps: formatNumber(metrics.bidDepthChange10),
    ask_depth_change_10bps: formatNumber(metrics.askDepthChange10),
    imbalance_change_10bps: formatNumber(imbalanceChange10),
    large_bid_wall_distance: formatNumber(metrics.largeBidWallDistance),
    large_ask_wall_distance: formatNumber(metrics.largeAskWallDistance),
    large_bid_wall_size: formatNumber(metrics.largeBidWallSize),
    large_ask_wall_size: formatNumber(metrics.largeAskWallSize)
  };

  previousSnapshotMetrics = metrics;
  snapshotTradeBucket = {
    marketBuyVolume: 0,
    marketSellVolume: 0,
    totalTradeVolume: 0,
    tradeCount: 0
  };

  return row;
}

function writeSnapshot(timestamp = Date.now()) {
  if (!orderBook.ready) {
    return;
  }

  const intervalMilliseconds = SNAPSHOT_INTERVAL_SECONDS * 1000;
  const snapshotTimestamp =
    Math.floor(timestamp / intervalMilliseconds) * intervalMilliseconds;

  if (snapshotTimestamp <= lastSnapshotWrittenTimestamp) {
    return;
  }

  const row = createSnapshotRow(snapshotTimestamp);
  const wroteRow = appendSnapshotRow(row);

  if (wroteRow) {
    rememberSnapshotForMinute(row);
    console.log(
      `${row.time} ${SYMBOL} 10s pressure=${row.market_pressure_10s} ` +
        `imbalance10=${row.order_book_imbalance_10bps} wrote ${SNAPSHOT_OUTPUT_PATH}`
    );
    logRecorderFreshness(
      `${VENUE} ${SYMBOL}`,
      lastSnapshotWrittenTimestamp,
      lastWrittenTimestamp,
      OUTPUT_PATH
    );
  }
}

function startSnapshotTimer() {
  if (snapshotTimer !== null) {
    clearTimeout(snapshotTimer);
    snapshotTimer = null;
  }

  const intervalMilliseconds = SNAPSHOT_INTERVAL_SECONDS * 1000;
  const scheduleNext = () => {
    const now = Date.now();
    const delay = intervalMilliseconds - (now % intervalMilliseconds);

    snapshotTimer = setTimeout(() => {
      writeSnapshot(Date.now());
      scheduleNext();
    }, delay);
  };

  scheduleNext();
}

function aggregateValues(values) {
  if (values.length === 0) {
    return {
      mean: 0,
      min: 0,
      max: 0,
      last: 0,
      change: 0
    };
  }

  const first = values[0];
  const last = values[values.length - 1];

  return {
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    min: Math.min(...values),
    max: Math.max(...values),
    last,
    change: last - first
  };
}

function aggregateSnapshotFeaturesForMinute(minuteTimestamp) {
  const rows = snapshotRowsByMinute.get(minuteTimestamp) || [];
  const features = {};

  for (const metric of SNAPSHOT_SUMMARY_METRICS) {
    const values = rows
      .map((row) => Number(row[metric.key]))
      .filter((value) => Number.isFinite(value));
    const summary = aggregateValues(values);

    features[`${metric.columnPrefix}_mean`] = formatNumber(summary.mean);
    features[`${metric.columnPrefix}_min`] = formatNumber(summary.min);
    features[`${metric.columnPrefix}_max`] = formatNumber(summary.max);
    features[`${metric.columnPrefix}_last`] = formatNumber(summary.last);
    features[`${metric.columnPrefix}_change`] = formatNumber(summary.change);
  }

  snapshotRowsByMinute.delete(minuteTimestamp);
  return features;
}

function createCandleFlowRow(kline) {
  const timestamp = Number(kline.t);
  const open = toNumber(kline.o);
  const high = toNumber(kline.h);
  const low = toNumber(kline.l);
  const close = toNumber(kline.c);
  const volume = toNumber(kline.v);
  const quoteVolume = toNumber(kline.q);
  const tradeCount = Number.parseInt(kline.n, 10) || 0;
  const klineTakerBuyVolume = toNumber(kline.V, NaN);
  const tradeBucket = aggregateTradeBuckets.get(timestamp);
  const takerBuyVolume = Number.isFinite(klineTakerBuyVolume)
    ? klineTakerBuyVolume
    : tradeBucket
      ? tradeBucket.takerBuyVolume
      : 0;
  const takerSellVolume = Math.max(0, volume - takerBuyVolume);
  const candleRange = high - low;
  const bookMetrics = orderBook.createMetrics(previousBookMetrics);
  const snapshotSummaryFeatures = aggregateSnapshotFeaturesForMinute(timestamp);

  const row = {
    venue: VENUE,
    source_quality: "binanceus_exchange_kline_book_depth",
    timestamp,
    time: new Date(timestamp).toISOString(),
    open: formatNumber(open),
    high: formatNumber(high),
    low: formatNumber(low),
    close: formatNumber(close),
    volume: formatNumber(volume),
    quote_volume: formatNumber(quoteVolume),
    trade_count: tradeCount,
    taker_buy_volume: formatNumber(takerBuyVolume),
    taker_sell_volume: formatNumber(takerSellVolume),
    taker_buy_ratio: formatNumber(safeRatio(takerBuyVolume, volume)),
    return_1: formatNumber(
      previousClose === null ? 0 : safeRatio(close - previousClose, previousClose)
    ),
    range_percent: formatNumber(safeRatio(candleRange, close)),
    upper_wick_percent: formatNumber(
      safeRatio(high - Math.max(open, close), close)
    ),
    lower_wick_percent: formatNumber(
      safeRatio(Math.min(open, close) - low, close)
    ),
    close_position_in_range: formatNumber(
      candleRange > 0 ? (close - low) / candleRange : 0.5
    ),
    best_bid: formatNumber(bookMetrics.bestBid),
    best_ask: formatNumber(bookMetrics.bestAsk),
    spread_percent: formatNumber(bookMetrics.spreadPercent),
    mid_price: formatNumber(bookMetrics.midPrice),
    bid_depth_10bps: formatNumber(bookMetrics.bidDepth10),
    ask_depth_10bps: formatNumber(bookMetrics.askDepth10),
    bid_depth_25bps: formatNumber(bookMetrics.bidDepth25),
    ask_depth_25bps: formatNumber(bookMetrics.askDepth25),
    order_book_imbalance_10bps: formatNumber(bookMetrics.imbalance10),
    order_book_imbalance_25bps: formatNumber(bookMetrics.imbalance25),
    bid_depth_change_10bps: formatNumber(bookMetrics.bidDepthChange10),
    ask_depth_change_10bps: formatNumber(bookMetrics.askDepthChange10),
    large_bid_wall_distance: formatNumber(bookMetrics.largeBidWallDistance),
    large_ask_wall_distance: formatNumber(bookMetrics.largeAskWallDistance),
    large_bid_wall_size: formatNumber(bookMetrics.largeBidWallSize),
    large_ask_wall_size: formatNumber(bookMetrics.largeAskWallSize),
    ...snapshotSummaryFeatures
  };

  previousClose = close;
  previousBookMetrics = bookMetrics;
  aggregateTradeBuckets.delete(timestamp);
  cleanOldTradeBuckets(timestamp);

  return row;
}

function handleClosedKline(kline) {
  const row = createCandleFlowRow(kline);
  const wroteRow = appendOneMinuteRow(row);

  if (wroteRow) {
    console.log(
      `${row.time} ${SYMBOL} close=${row.close} return_1=${row.return_1} ` +
        `spread=${row.spread_percent} imbalance10=${row.order_book_imbalance_10bps} ` +
        `wrote ${OUTPUT_PATH}`
    );
    logRecorderFreshness(
      `${VENUE} ${SYMBOL}`,
      lastSnapshotWrittenTimestamp,
      lastWrittenTimestamp,
      OUTPUT_PATH
    );
  }
}

function handleMessage(rawMessage) {
  const parsed = JSON.parse(rawMessage);
  const event = parsed.data || parsed;

  if (event.e === "depthUpdate") {
    handleDepthUpdate(event);
    return;
  }

  if (event.e === "aggTrade") {
    handleAggregateTrade(event);
    return;
  }

  if (event.e === "kline" && event.k && event.k.x === true) {
    handleClosedKline(event.k);
  }
}

function connectWebSocket() {
  const websocket = new WebSocket(BINANCE_WS_URL);

  websocket.addEventListener("open", () => {
    console.log(`Connected to Binance streams for ${SYMBOL}.`);
    orderBook.reset();
    depthBuffer = [];
    loadOrderBookSnapshot();
  });

  websocket.addEventListener("message", (message) => {
    try {
      handleMessage(message.data);
    } catch (error) {
      console.error(`Message handling failed: ${error.message}`);
    }
  });

  websocket.addEventListener("error", (error) => {
    console.error(`WebSocket error: ${error.message || "unknown error"}`);
  });

  websocket.addEventListener("close", () => {
    console.warn("WebSocket closed; reconnecting in 5 seconds.");
    orderBook.ready = false;
    setTimeout(connectWebSocket, 5000);
  });

  return websocket;
}

function symbolToVenuePair(symbol, venue) {
  const upper = symbol.toUpperCase();
  const base = upper.endsWith("USDT")
    ? upper.slice(0, -4)
    : upper.endsWith("USD")
      ? upper.slice(0, -3)
      : upper;

  if (venue === "kraken") {
    return `${base}/USD`;
  }

  return upper;
}

function venueOutputPaths(venue, symbol, config) {
  const venueOutputDir =
    venue === "binanceus" && config.legacyBinanceUsOutput
      ? config.outputDir
      : path.join(config.outputDir, venue);

  return {
    oneMinutePath: path.join(venueOutputDir, `${symbol}_1m_flow.csv`),
    snapshotPath: path.join(venueOutputDir, `${symbol}_10s_flow.csv`)
  };
}

class KrakenRecorder {
  constructor(symbol, config) {
    this.venue = "kraken";
    this.symbol = symbol;
    this.krakenSymbol = symbolToVenuePair(symbol, this.venue);
    this.config = config;
    this.paths = venueOutputPaths(this.venue, symbol, config);
    this.orderBook = new OrderBook(config.orderBookDepth);
    this.orderBook.ready = false;
    this.previousBookMetrics = null;
    this.previousSnapshotMetrics = null;
    this.previousClose = null;
    this.snapshotTimer = null;
    this.websocket = null;
    this.lastWrittenTimestamp = ensureCsvFile(this.paths.oneMinutePath, OUTPUT_COLUMNS);
    this.lastSnapshotWrittenTimestamp = ensureCsvFile(this.paths.snapshotPath, SNAPSHOT_COLUMNS);
    this.snapshotTradeBucket = this.emptyTradeBucket();
    this.candleBuckets = new Map();
  }

  emptyTradeBucket() {
    return {
      marketBuyVolume: 0,
      marketSellVolume: 0,
      totalTradeVolume: 0,
      quoteVolume: 0,
      tradeCount: 0
    };
  }

  start() {
    console.log(
      `Starting Kraken public recorder for ${this.symbol} as ${this.krakenSymbol}.`
    );
    console.log(`Kraken 1m output file: ${this.paths.oneMinutePath}`);
    console.log(`Kraken ${SNAPSHOT_INTERVAL_SECONDS}s output file: ${this.paths.snapshotPath}`);
    this.connect();
    this.startSnapshotTimer();
    return {
      stop: () => this.stop()
    };
  }

  stop() {
    if (this.snapshotTimer !== null) {
      clearTimeout(this.snapshotTimer);
      this.snapshotTimer = null;
    }
    if (this.websocket) {
      this.websocket.close();
    }
  }

  connect() {
    this.websocket = new WebSocket("wss://ws.kraken.com/v2");

    this.websocket.addEventListener("open", () => {
      console.log(`Connected to Kraken public websocket for ${this.krakenSymbol}.`);
      const subscriptions = [
        {
          method: "subscribe",
          params: {
            channel: "book",
            symbol: [this.krakenSymbol],
            depth: Math.min(this.config.orderBookDepth, 100)
          }
        },
        {
          method: "subscribe",
          params: {
            channel: "trade",
            symbol: [this.krakenSymbol]
          }
        }
      ];
      for (const message of subscriptions) {
        this.websocket.send(JSON.stringify(message));
      }
    });

    this.websocket.addEventListener("message", (message) => {
      try {
        this.handleMessage(JSON.parse(message.data));
      } catch (error) {
        console.error(`Kraken message handling failed for ${this.symbol}: ${error.message}`);
      }
    });

    this.websocket.addEventListener("error", (error) => {
      console.error(`Kraken websocket error for ${this.symbol}: ${error.message || "unknown error"}`);
    });

    this.websocket.addEventListener("close", () => {
      console.warn(`Kraken websocket closed for ${this.symbol}; reconnecting in 5 seconds.`);
      this.orderBook.ready = false;
      setTimeout(() => this.connect(), 5000);
    });
  }

  handleMessage(message) {
    if (!message || !message.channel) {
      return;
    }
    if (message.channel === "book") {
      this.handleBookMessage(message);
    } else if (message.channel === "trade") {
      this.handleTradeMessage(message);
    }
  }

  handleBookMessage(message) {
    const rows = Array.isArray(message.data) ? message.data : [];
    for (const row of rows) {
      if (Array.isArray(row.bids) || Array.isArray(row.asks)) {
        if (message.type === "snapshot") {
          this.orderBook.reset();
        }
        for (const level of row.bids || []) {
          this.orderBook.setLevel(this.orderBook.bids, level.price ?? level[0], level.qty ?? level[1]);
        }
        for (const level of row.asks || []) {
          this.orderBook.setLevel(this.orderBook.asks, level.price ?? level[0], level.qty ?? level[1]);
        }
        this.orderBook.prune();
        this.orderBook.ready = this.orderBook.bids.size > 0 && this.orderBook.asks.size > 0;
      }
    }
  }

  handleTradeMessage(message) {
    const rows = Array.isArray(message.data) ? message.data : [];
    for (const row of rows) {
      const price = toNumber(row.price, NaN);
      const quantity = toNumber(row.qty ?? row.quantity, NaN);
      const timestamp = Date.parse(row.timestamp || row.time || new Date().toISOString());
      if (!Number.isFinite(price) || !Number.isFinite(quantity) || !Number.isFinite(timestamp)) {
        continue;
      }

      this.rememberTrade(timestamp, price, quantity, String(row.side || "").toLowerCase());
      this.flushCompletedCandles(Math.floor(timestamp / 60000) * 60000);
    }
  }

  rememberTrade(timestamp, price, quantity, side) {
    const minuteTimestamp = Math.floor(timestamp / 60000) * 60000;
    const bucket = this.candleBuckets.get(minuteTimestamp) || {
      timestamp: minuteTimestamp,
      open: price,
      high: price,
      low: price,
      close: price,
      volume: 0,
      quoteVolume: 0,
      tradeCount: 0,
      takerBuyVolume: 0,
      takerSellVolume: 0
    };

    bucket.high = Math.max(bucket.high, price);
    bucket.low = Math.min(bucket.low, price);
    bucket.close = price;
    bucket.volume += quantity;
    bucket.quoteVolume += quantity * price;
    bucket.tradeCount += 1;

    if (side === "buy") {
      bucket.takerBuyVolume += quantity;
      this.snapshotTradeBucket.marketBuyVolume += quantity;
    } else if (side === "sell") {
      bucket.takerSellVolume += quantity;
      this.snapshotTradeBucket.marketSellVolume += quantity;
    }

    this.snapshotTradeBucket.totalTradeVolume += quantity;
    this.snapshotTradeBucket.quoteVolume += quantity * price;
    this.snapshotTradeBucket.tradeCount += 1;
    this.candleBuckets.set(minuteTimestamp, bucket);
  }

  flushCompletedCandles(currentMinuteTimestamp) {
    for (const [minuteTimestamp, bucket] of [...this.candleBuckets.entries()]) {
      if (minuteTimestamp < currentMinuteTimestamp) {
        this.writeOneMinuteBucket(bucket);
        this.candleBuckets.delete(minuteTimestamp);
      }
    }
  }

  createSnapshotRow(timestamp, metrics) {
    const marketPressure = safeRatio(
      this.snapshotTradeBucket.marketBuyVolume - this.snapshotTradeBucket.marketSellVolume,
      this.snapshotTradeBucket.totalTradeVolume
    );
    const imbalanceChange10 = this.previousSnapshotMetrics
      ? metrics.imbalance10 - this.previousSnapshotMetrics.imbalance10
      : 0;

    const row = {
      venue: this.venue,
      source_quality: this.orderBook.ready
        ? "kraken_trade_stream_book_depth_approx"
        : "kraken_trade_stream_book_missing",
      timestamp,
      time: new Date(timestamp).toISOString(),
      mid_price: formatNumber(metrics.midPrice),
      best_bid: formatNumber(metrics.bestBid),
      best_ask: formatNumber(metrics.bestAsk),
      spread_percent: formatNumber(metrics.spreadPercent),
      market_buy_volume_10s: formatNumber(this.snapshotTradeBucket.marketBuyVolume),
      market_sell_volume_10s: formatNumber(this.snapshotTradeBucket.marketSellVolume),
      total_trade_volume_10s: formatNumber(this.snapshotTradeBucket.totalTradeVolume),
      trade_count_10s: this.snapshotTradeBucket.tradeCount,
      market_pressure_10s: formatNumber(marketPressure),
      bid_depth_10bps: this.orderBook.ready ? formatNumber(metrics.bidDepth10) : "",
      ask_depth_10bps: this.orderBook.ready ? formatNumber(metrics.askDepth10) : "",
      bid_depth_25bps: this.orderBook.ready ? formatNumber(metrics.bidDepth25) : "",
      ask_depth_25bps: this.orderBook.ready ? formatNumber(metrics.askDepth25) : "",
      order_book_imbalance_10bps: this.orderBook.ready ? formatNumber(metrics.imbalance10) : "",
      order_book_imbalance_25bps: this.orderBook.ready ? formatNumber(metrics.imbalance25) : "",
      bid_depth_change_10bps: this.orderBook.ready ? formatNumber(metrics.bidDepthChange10) : "",
      ask_depth_change_10bps: this.orderBook.ready ? formatNumber(metrics.askDepthChange10) : "",
      imbalance_change_10bps: this.orderBook.ready ? formatNumber(imbalanceChange10) : "",
      large_bid_wall_distance: this.orderBook.ready ? formatNumber(metrics.largeBidWallDistance) : "",
      large_ask_wall_distance: this.orderBook.ready ? formatNumber(metrics.largeAskWallDistance) : "",
      large_bid_wall_size: this.orderBook.ready ? formatNumber(metrics.largeBidWallSize) : "",
      large_ask_wall_size: this.orderBook.ready ? formatNumber(metrics.largeAskWallSize) : ""
    };

    this.previousSnapshotMetrics = metrics;
    this.snapshotTradeBucket = this.emptyTradeBucket();
    return row;
  }

  writeSnapshot(timestamp = Date.now()) {
    const intervalMilliseconds = SNAPSHOT_INTERVAL_SECONDS * 1000;
    const snapshotTimestamp =
      Math.floor(timestamp / intervalMilliseconds) * intervalMilliseconds;
    if (snapshotTimestamp <= this.lastSnapshotWrittenTimestamp) {
      return;
    }
    const metrics = this.orderBook.createMetrics(this.previousSnapshotMetrics);
    const validation = validateBookMetrics(metrics);
    if (!this.orderBook.ready || !validation.valid) {
      console.warn(
        `${new Date(snapshotTimestamp).toISOString()} kraken ${this.symbol} snapshot skipped ` +
          `book_initialized=${this.orderBook.ready} ${formatBookDiagnostics(metrics)} ` +
          `reason=${this.orderBook.ready ? `invalid_book_fields:${validation.invalid.join("|")}` : "book_not_initialized"}`
      );
      return;
    }

    const row = this.createSnapshotRow(snapshotTimestamp, metrics);
    const result = appendCsvRow(
      this.paths.snapshotPath,
      SNAPSHOT_COLUMNS,
      row,
      this.lastSnapshotWrittenTimestamp
    );
    this.lastSnapshotWrittenTimestamp = result.lastTimestamp;
    if (result.wrote) {
      console.log(
        `${row.time} kraken ${this.symbol} snapshot pressure=${row.market_pressure_10s} ` +
          `book_initialized=${this.orderBook.ready} best_bid=${row.best_bid} best_ask=${row.best_ask} ` +
          `bid_depth_10bps=${row.bid_depth_10bps} ask_depth_10bps=${row.ask_depth_10bps} ` +
          `imbalance10=${row.order_book_imbalance_10bps} wrote ${this.paths.snapshotPath}`
      );
      logRecorderFreshness(
        `${this.venue} ${this.symbol}`,
        this.lastSnapshotWrittenTimestamp,
        this.lastWrittenTimestamp,
        this.paths.oneMinutePath
      );
    }
  }

  startSnapshotTimer() {
    const intervalMilliseconds = SNAPSHOT_INTERVAL_SECONDS * 1000;
    const scheduleNext = () => {
      const now = Date.now();
      const delay = intervalMilliseconds - (now % intervalMilliseconds);
      this.snapshotTimer = setTimeout(() => {
        this.writeSnapshot(Date.now());
        this.flushCompletedCandles(Math.floor(Date.now() / 60000) * 60000);
        scheduleNext();
      }, delay);
    };
    scheduleNext();
  }

  writeOneMinuteBucket(bucket) {
    const candleRange = bucket.high - bucket.low;
    const bookMetrics = this.orderBook.createMetrics(this.previousBookMetrics);
    const row = {
      venue: this.venue,
      source_quality: "kraken_trade_aggregated_1m_book_depth_approx",
      timestamp: bucket.timestamp,
      time: new Date(bucket.timestamp).toISOString(),
      open: formatNumber(bucket.open),
      high: formatNumber(bucket.high),
      low: formatNumber(bucket.low),
      close: formatNumber(bucket.close),
      volume: formatNumber(bucket.volume),
      quote_volume: formatNumber(bucket.quoteVolume),
      trade_count: bucket.tradeCount,
      taker_buy_volume: formatNumber(bucket.takerBuyVolume),
      taker_sell_volume: formatNumber(bucket.takerSellVolume),
      taker_buy_ratio: formatNumber(safeRatio(bucket.takerBuyVolume, bucket.volume)),
      return_1: formatNumber(
        this.previousClose === null
          ? 0
          : safeRatio(bucket.close - this.previousClose, this.previousClose)
      ),
      range_percent: formatNumber(safeRatio(candleRange, bucket.close)),
      upper_wick_percent: formatNumber(
        safeRatio(bucket.high - Math.max(bucket.open, bucket.close), bucket.close)
      ),
      lower_wick_percent: formatNumber(
        safeRatio(Math.min(bucket.open, bucket.close) - bucket.low, bucket.close)
      ),
      close_position_in_range: formatNumber(
        candleRange > 0 ? (bucket.close - bucket.low) / candleRange : 0.5
      ),
      best_bid: this.orderBook.ready ? formatNumber(bookMetrics.bestBid) : "",
      best_ask: this.orderBook.ready ? formatNumber(bookMetrics.bestAsk) : "",
      spread_percent: this.orderBook.ready ? formatNumber(bookMetrics.spreadPercent) : "",
      mid_price: this.orderBook.ready ? formatNumber(bookMetrics.midPrice) : "",
      bid_depth_10bps: this.orderBook.ready ? formatNumber(bookMetrics.bidDepth10) : "",
      ask_depth_10bps: this.orderBook.ready ? formatNumber(bookMetrics.askDepth10) : "",
      bid_depth_25bps: this.orderBook.ready ? formatNumber(bookMetrics.bidDepth25) : "",
      ask_depth_25bps: this.orderBook.ready ? formatNumber(bookMetrics.askDepth25) : "",
      order_book_imbalance_10bps: this.orderBook.ready ? formatNumber(bookMetrics.imbalance10) : "",
      order_book_imbalance_25bps: this.orderBook.ready ? formatNumber(bookMetrics.imbalance25) : "",
      bid_depth_change_10bps: this.orderBook.ready ? formatNumber(bookMetrics.bidDepthChange10) : "",
      ask_depth_change_10bps: this.orderBook.ready ? formatNumber(bookMetrics.askDepthChange10) : "",
      large_bid_wall_distance: this.orderBook.ready ? formatNumber(bookMetrics.largeBidWallDistance) : "",
      large_ask_wall_distance: this.orderBook.ready ? formatNumber(bookMetrics.largeAskWallDistance) : "",
      large_bid_wall_size: this.orderBook.ready ? formatNumber(bookMetrics.largeBidWallSize) : "",
      large_ask_wall_size: this.orderBook.ready ? formatNumber(bookMetrics.largeAskWallSize) : ""
    };
    for (const metric of SNAPSHOT_SUMMARY_METRICS) {
      row[`${metric.columnPrefix}_mean`] = "";
      row[`${metric.columnPrefix}_min`] = "";
      row[`${metric.columnPrefix}_max`] = "";
      row[`${metric.columnPrefix}_last`] = "";
      row[`${metric.columnPrefix}_change`] = "";
    }

    const result = appendCsvRow(
      this.paths.oneMinutePath,
      OUTPUT_COLUMNS,
      row,
      this.lastWrittenTimestamp
    );
    this.lastWrittenTimestamp = result.lastTimestamp;
    this.previousClose = bucket.close;
    this.previousBookMetrics = bookMetrics;
    if (result.wrote) {
      console.log(
        `${row.time} kraken ${this.symbol} close=${row.close} ` +
          `return_1=${row.return_1} wrote ${this.paths.oneMinutePath}`
      );
      logRecorderFreshness(
        `${this.venue} ${this.symbol}`,
        this.lastSnapshotWrittenTimestamp,
        this.lastWrittenTimestamp,
        this.paths.oneMinutePath
      );
    }
  }
}

function startBinanceUsRecorder(symbol, config) {
  if (symbol !== SYMBOL) {
    throw new Error("Binance US recorder currently runs one symbol per process.");
  }
  validateSettings();
  ensureOutputFiles();

  console.log("Realtime Binance US 1m market data recorder");
  console.log(`Venue: ${VENUE}`);
  console.log(`Symbol: ${SYMBOL}`);
  console.log(`Order book depth: ${ORDER_BOOK_DEPTH}`);
  console.log(`Snapshot interval seconds: ${SNAPSHOT_INTERVAL_SECONDS}`);
  console.log(`Legacy Binance US output enabled: ${LEGACY_BINANCEUS_OUTPUT}`);
  console.log(`1m output file: ${OUTPUT_PATH}`);
  console.log(`10s output file: ${SNAPSHOT_OUTPUT_PATH}`);
  console.log("Streams: 1m kline, aggregate trades, depth updates");
  console.log("This recorder does not place trades and does not train models.");

  const websocket = connectWebSocket();
  startSnapshotTimer();
  return {
    stop: () => {
      if (snapshotTimer !== null) {
        clearTimeout(snapshotTimer);
      }
      websocket.close();
    }
  };
}

function startVenueRecorder(venue, symbol, config) {
  const normalizedVenue = String(venue || "").trim().toLowerCase();
  if (normalizedVenue === "binanceus") {
    return startBinanceUsRecorder(symbol, config);
  }
  if (normalizedVenue === "kraken") {
    return new KrakenRecorder(symbol, config).start();
  }
  throw new Error(`Unsupported venue: ${venue}. Supported venues: binanceus, kraken.`);
}

function runSupervisorProcesses() {
  const combinations = [];
  for (const symbol of SYMBOLS) {
    for (const venue of VENUES) {
      combinations.push({ symbol, venue });
    }
  }

  if (combinations.length <= 1) {
    return false;
  }

  console.log("Starting multi-venue realtime recorder supervisor");
  console.log(`Symbols: ${SYMBOLS.join(", ")}`);
  console.log(`Venues: ${VENUES.join(", ")}`);
  console.log("No trades are placed. No private/account API keys are used.");

  const children = [];
  for (const { symbol, venue } of combinations) {
    const childEnv = {
      ...process.env,
      RECORDER_CHILD: "1",
      SYMBOL: symbol,
      SYMBOLS: symbol,
      VENUE: venue,
      VENUES: venue
    };
    const child = spawn(process.execPath, [__filename], {
      cwd: PROJECT_ROOT,
      env: childEnv,
      stdio: ["ignore", "pipe", "pipe"]
    });
    children.push(child);
    const prefix = `[${venue} ${symbol}]`;
    child.stdout.on("data", (chunk) => {
      process.stdout.write(
        String(chunk)
          .split(/\r?\n/)
          .filter(Boolean)
          .map((line) => `${prefix} ${line}`)
          .join("\n") + "\n"
      );
    });
    child.stderr.on("data", (chunk) => {
      process.stderr.write(
        String(chunk)
          .split(/\r?\n/)
          .filter(Boolean)
          .map((line) => `${prefix} ${line}`)
          .join("\n") + "\n"
      );
    });
    child.on("exit", (code) => {
      console.warn(`${prefix} recorder exited with code ${code}`);
    });
  }

  process.on("SIGINT", () => {
    console.log("\nStopping multi-venue recorder supervisor.");
    for (const child of children) {
      child.kill("SIGINT");
    }
    process.exit(0);
  });

  return true;
}

function main() {
  if (process.env.RECORDER_CHILD !== "1" && runSupervisorProcesses()) {
    return;
  }

  const config = {
    outputDir: OUTPUT_DIR,
    orderBookDepth: ORDER_BOOK_DEPTH,
    snapshotIntervalSeconds: SNAPSHOT_INTERVAL_SECONDS,
    legacyBinanceUsOutput: LEGACY_BINANCEUS_OUTPUT
  };

  const recorder = startVenueRecorder(VENUE, SYMBOL, config);

  process.on("SIGINT", () => {
    console.log("\nStopping recorder.");
    recorder.stop();
    process.exit(0);
  });
}

/*
 * Legacy main logic is intentionally replaced by startVenueRecorder above.
 * Binance US still uses the original stream handlers and state; Kraken uses
 * official public websocket trade/book streams and fills unavailable fields
 * with blanks plus source_quality flags.
 */
function legacyMain() {
  validateSettings();
  ensureOutputFiles();

  console.log("Realtime Binance 1m market data recorder");
  console.log(`Symbol: ${SYMBOL}`);
  console.log(`Order book depth: ${ORDER_BOOK_DEPTH}`);
  console.log(`Snapshot interval seconds: ${SNAPSHOT_INTERVAL_SECONDS}`);
  console.log(`1m output file: ${OUTPUT_PATH}`);
  console.log(`10s output file: ${SNAPSHOT_OUTPUT_PATH}`);
  console.log("Streams: 1m kline, aggregate trades, depth updates");
  console.log("This recorder does not place trades and does not train models.");

  const websocket = connectWebSocket();
  startSnapshotTimer();

  process.on("SIGINT", () => {
    console.log("\nStopping recorder.");
    if (snapshotTimer !== null) {
      clearTimeout(snapshotTimer);
    }
    websocket.close();
    process.exit(0);
  });
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    console.error("Realtime recorder failed:", error.message);
    process.exitCode = 1;
  }
}

module.exports = {
  OrderBook,
  createCandleFlowRow,
  handleAggregateTrade,
  writeSnapshot,
  safeRatio
};
