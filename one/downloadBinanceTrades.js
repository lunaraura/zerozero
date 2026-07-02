/*
 * Download Binance public historical market-trade archives.
 *
 * Streams ZIP CSV entries directly into the canonical trade-tape output:
 *
 *   timestamp_ms,price,size,side
 *
 * No API keys, no account/private endpoints, no orders, no live prediction
 * writes. Public market-data research only.
 */

const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const readline = require("readline");
const { once } = require("events");
const { Readable } = require("stream");
const { pipeline } = require("stream/promises");

const PROJECT_ROOT = path.resolve(__dirname, "..");

function readString(name, defaultValue) {
  const value = process.env[name];
  return value === undefined || value.trim() === "" ? defaultValue : value.trim();
}

function readNumber(name, defaultValue) {
  const value = process.env[name];
  if (value === undefined || value.trim() === "") return defaultValue;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${name} must be numeric. Received: ${value}`);
  return parsed;
}

function resolvePath(value) {
  return path.isAbsolute(value) ? value : path.join(PROJECT_ROOT, value);
}

function ensureDirectory(directoryPath) {
  fs.mkdirSync(directoryPath, { recursive: true });
}

function validateMonth(month) {
  if (!/^\d{4}-\d{2}$/.test(month)) {
    throw new Error(`Month must use YYYY-MM format. Received: ${month}`);
  }
  const monthNumber = Number(month.slice(5, 7));
  if (monthNumber < 1 || monthNumber > 12) {
    throw new Error(`Month must be between 01 and 12. Received: ${month}`);
  }
}

function addOneMonth(month) {
  const year = Number(month.slice(0, 4));
  const monthNumber = Number(month.slice(5, 7));
  const nextMonth = monthNumber === 12 ? 1 : monthNumber + 1;
  const nextYear = monthNumber === 12 ? year + 1 : year;
  return `${nextYear}-${String(nextMonth).padStart(2, "0")}`;
}

function listMonths(startMonth, endMonth) {
  validateMonth(startMonth);
  validateMonth(endMonth);
  if (startMonth > endMonth) throw new Error("BINANCE_START_MONTH must be <= BINANCE_END_MONTH.");

  const months = [];
  let month = startMonth;
  while (month <= endMonth) {
    months.push(month);
    month = addOneMonth(month);
  }
  return months;
}

function parseTimeMs(value) {
  if (value === undefined || String(value).trim() === "") return null;
  const text = String(value).trim();
  const numeric = Number(text);
  if (Number.isFinite(numeric)) {
    if (numeric > 1e12) return Math.floor(numeric);
    if (numeric > 1e9) return Math.floor(numeric * 1000);
  }
  const parsed = Date.parse(text);
  if (!Number.isFinite(parsed)) throw new Error(`Could not parse time: ${value}`);
  return parsed;
}

function zipFileName(symbol, dataType, month) {
  return `${symbol}-${dataType}-${month}.zip`;
}

function downloadUrl(baseUrl, symbol, dataType, month) {
  return [
    baseUrl.replace(/\/+$/, ""),
    dataType,
    symbol,
    zipFileName(symbol, dataType, month),
  ].join("/");
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function downloadZip(url, destinationPath, maxRetries) {
  if (fs.existsSync(destinationPath)) return "cached";

  let delayMs = 1000;
  for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
    const response = await fetch(url);
    if (response.status === 404) return "missing";
    if (response.status === 429 || response.status >= 500) {
      if (attempt === maxRetries) {
        throw new Error(`Download failed after retries with HTTP ${response.status}: ${url}`);
      }
      console.log(`HTTP ${response.status}; retrying in ${delayMs}ms: ${url}`);
      await sleep(delayMs);
      delayMs = Math.min(delayMs * 2, 30000);
      continue;
    }
    if (!response.ok) throw new Error(`Download failed with HTTP ${response.status}: ${url}`);
    if (!response.body) throw new Error(`Download response has no body: ${url}`);

    const tempPath = `${destinationPath}.tmp`;
    await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(tempPath));
    fs.renameSync(tempPath, destinationPath);
    return "downloaded";
  }

  throw new Error(`Download failed: ${url}`);
}

function readRange(filePath, position, length) {
  const fd = fs.openSync(filePath, "r");
  try {
    const buffer = Buffer.alloc(length);
    const bytesRead = fs.readSync(fd, buffer, 0, length, position);
    return bytesRead === length ? buffer : buffer.subarray(0, bytesRead);
  } finally {
    fs.closeSync(fd);
  }
}

function uint64ToNumber(buffer, offset) {
  const value = buffer.readBigUInt64LE(offset);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error("ZIP entry is larger than JavaScript can safely address.");
  }
  return Number(value);
}

function findEndOfCentralDirectory(filePath) {
  const stat = fs.statSync(filePath);
  const tailLength = Math.min(stat.size, 65557);
  const tailStart = stat.size - tailLength;
  const tail = readRange(filePath, tailStart, tailLength);

  for (let index = tail.length - 22; index >= 0; index -= 1) {
    if (tail.readUInt32LE(index) !== 0x06054b50) continue;

    let entryCount = tail.readUInt16LE(index + 10);
    let centralDirectorySize = tail.readUInt32LE(index + 12);
    let centralDirectoryOffset = tail.readUInt32LE(index + 16);

    if (
      entryCount === 0xffff ||
      centralDirectorySize === 0xffffffff ||
      centralDirectoryOffset === 0xffffffff
    ) {
      const locatorIndex = index - 20;
      if (locatorIndex < 0 || tail.readUInt32LE(locatorIndex) !== 0x07064b50) {
        throw new Error("ZIP64 archive found, but ZIP64 locator was not present in the readable tail.");
      }
      const zip64EocdOffset = uint64ToNumber(tail, locatorIndex + 8);
      const zip64Header = readRange(filePath, zip64EocdOffset, 56);
      if (zip64Header.readUInt32LE(0) !== 0x06064b50) {
        throw new Error("Invalid ZIP64 end of central directory record.");
      }
      entryCount = uint64ToNumber(zip64Header, 32);
      centralDirectorySize = uint64ToNumber(zip64Header, 40);
      centralDirectoryOffset = uint64ToNumber(zip64Header, 48);
    }

    return { entryCount, centralDirectorySize, centralDirectoryOffset };
  }

  throw new Error("Could not find ZIP central directory.");
}

function parseZip64Extra(extraBuffer, needs) {
  const values = {};
  let offset = 0;
  while (offset + 4 <= extraBuffer.length) {
    const headerId = extraBuffer.readUInt16LE(offset);
    const dataSize = extraBuffer.readUInt16LE(offset + 2);
    const dataStart = offset + 4;
    const dataEnd = dataStart + dataSize;
    if (dataEnd > extraBuffer.length) break;

    if (headerId === 0x0001) {
      let dataOffset = dataStart;
      if (needs.uncompressedSize && dataOffset + 8 <= dataEnd) {
        values.uncompressedSize = uint64ToNumber(extraBuffer, dataOffset);
        dataOffset += 8;
      }
      if (needs.compressedSize && dataOffset + 8 <= dataEnd) {
        values.compressedSize = uint64ToNumber(extraBuffer, dataOffset);
        dataOffset += 8;
      }
      if (needs.localHeaderOffset && dataOffset + 8 <= dataEnd) {
        values.localHeaderOffset = uint64ToNumber(extraBuffer, dataOffset);
      }
      return values;
    }
    offset = dataEnd;
  }
  return values;
}

function readZipEntries(filePath) {
  const { entryCount, centralDirectorySize, centralDirectoryOffset } = findEndOfCentralDirectory(filePath);
  const central = readRange(filePath, centralDirectoryOffset, centralDirectorySize);
  const entries = [];
  let offset = 0;

  for (let index = 0; index < entryCount; index += 1) {
    if (central.readUInt32LE(offset) !== 0x02014b50) {
      throw new Error("Invalid ZIP central directory entry.");
    }

    const compressionMethod = central.readUInt16LE(offset + 10);
    let compressedSize = central.readUInt32LE(offset + 20);
    let uncompressedSize = central.readUInt32LE(offset + 24);
    const fileNameLength = central.readUInt16LE(offset + 28);
    const extraLength = central.readUInt16LE(offset + 30);
    const commentLength = central.readUInt16LE(offset + 32);
    let localHeaderOffset = central.readUInt32LE(offset + 42);
    const fileName = central.subarray(offset + 46, offset + 46 + fileNameLength).toString("utf8");
    const extra = central.subarray(offset + 46 + fileNameLength, offset + 46 + fileNameLength + extraLength);

    const zip64 = parseZip64Extra(extra, {
      uncompressedSize: uncompressedSize === 0xffffffff,
      compressedSize: compressedSize === 0xffffffff,
      localHeaderOffset: localHeaderOffset === 0xffffffff,
    });
    compressedSize = zip64.compressedSize ?? compressedSize;
    uncompressedSize = zip64.uncompressedSize ?? uncompressedSize;
    localHeaderOffset = zip64.localHeaderOffset ?? localHeaderOffset;

    entries.push({ fileName, compressionMethod, compressedSize, uncompressedSize, localHeaderOffset });
    offset += 46 + fileNameLength + extraLength + commentLength;
  }

  return entries;
}

function openZipEntryStream(filePath, entry) {
  const localHeader = readRange(filePath, entry.localHeaderOffset, 30);
  if (localHeader.readUInt32LE(0) !== 0x04034b50) {
    throw new Error(`Invalid ZIP local header for ${entry.fileName}`);
  }
  const fileNameLength = localHeader.readUInt16LE(26);
  const extraLength = localHeader.readUInt16LE(28);
  const dataStart = entry.localHeaderOffset + 30 + fileNameLength + extraLength;
  const dataEnd = dataStart + entry.compressedSize - 1;
  const compressedStream = fs.createReadStream(filePath, { start: dataStart, end: dataEnd });

  if (entry.compressionMethod === 0) return compressedStream;
  if (entry.compressionMethod === 8) return compressedStream.pipe(zlib.createInflateRaw());
  throw new Error(`Unsupported ZIP compression method ${entry.compressionMethod} in ${entry.fileName}`);
}

function parseBoolean(value) {
  const text = String(value).trim().toLowerCase();
  if (text === "true" || text === "1") return true;
  if (text === "false" || text === "0") return false;
  return null;
}

function parseBinanceTimestamp(value) {
  const timestamp = Number(value);
  if (!Number.isFinite(timestamp)) return NaN;
  if (timestamp > 1e14) return Math.floor(timestamp / 1000);
  return Math.floor(timestamp);
}

function parseTradeLine(line, dataType) {
  const trimmed = line.trim();
  if (!trimmed) return null;
  const columns = trimmed.split(",").map((value) => value.trim());
  if (!Number.isFinite(Number(columns[0]))) return null;

  if (dataType === "trades") {
    if (columns.length < 6) return null;
    const hasQuoteQty = columns.length >= 7;
    const price = Number(columns[1]);
    const size = Number(columns[2]);
    const timestampMs = parseBinanceTimestamp(hasQuoteQty ? columns[4] : columns[3]);
    const isBuyerMaker = parseBoolean(hasQuoteQty ? columns[5] : columns[4]);

    if (!Number.isFinite(timestampMs) || !Number.isFinite(price) || !Number.isFinite(size)) return null;
    if (price <= 0 || size <= 0 || isBuyerMaker === null) return null;
    return {
      timestamp_ms: timestampMs,
      price,
      size,
      side: isBuyerMaker ? "sell" : "buy",
      dedupe_id: columns[0],
    };
  }

  if (dataType === "aggTrades") {
    if (columns.length < 8) return null;
    const price = Number(columns[1]);
    const size = Number(columns[2]);
    const timestampMs = parseBinanceTimestamp(columns[5]);
    const isBuyerMaker = parseBoolean(columns[6]);

    if (!Number.isFinite(timestampMs) || !Number.isFinite(price) || !Number.isFinite(size)) return null;
    if (price <= 0 || size <= 0 || isBuyerMaker === null) return null;
    return {
      timestamp_ms: timestampMs,
      price,
      size,
      side: isBuyerMaker ? "sell" : "buy",
      dedupe_id: `${columns[0]}:${columns[3]}-${columns[4]}`,
    };
  }

  throw new Error(`Unsupported BINANCE_DATA_TYPE=${dataType}. Use trades or aggTrades.`);
}

function canonicalCsvRow(trade) {
  return [
    trade.timestamp_ms,
    Number(trade.price).toPrecision(12),
    Number(trade.size).toPrecision(12),
    trade.side,
  ].join(",");
}

async function writeLine(outputStream, line) {
  if (!outputStream.write(line)) {
    await once(outputStream, "drain");
  }
}

async function appendParsedTradesFromZipEntry({
  zipPath,
  entry,
  dataType,
  outputStream,
  startMs,
  endMs,
  state,
}) {
  const input = openZipEntryStream(zipPath, entry);
  const reader = readline.createInterface({ input, crlfDelay: Infinity });

  let parsedRows = 0;
  let writtenRows = 0;
  let skippedRows = 0;
  let skippedBeforeRange = 0;
  let stoppedAfterRange = 0;
  let adjacentDuplicateRows = 0;

  for await (const line of reader) {
    const trade = parseTradeLine(line, dataType);
    if (trade === null) {
      skippedRows += 1;
      continue;
    }

    parsedRows += 1;
    if (startMs !== null && trade.timestamp_ms < startMs) {
      skippedBeforeRange += 1;
      continue;
    }
    if (endMs !== null && trade.timestamp_ms >= endMs) {
      stoppedAfterRange += 1;
      state.stoppedAfterRange = true;
      break;
    }

    const key = `${trade.timestamp_ms}|${trade.dedupe_id}|${trade.price}|${trade.size}|${trade.side}`;
    if (state.lastDedupeKey === key) {
      adjacentDuplicateRows += 1;
      continue;
    }
    state.lastDedupeKey = key;

    if (state.lastTimestamp !== null && trade.timestamp_ms < state.lastTimestamp) {
      state.nonMonotonicRows += 1;
    }
    state.lastTimestamp = trade.timestamp_ms;
    state.firstTimestamp = state.firstTimestamp === null ? trade.timestamp_ms : Math.min(state.firstTimestamp, trade.timestamp_ms);
    state.lastWrittenTimestamp = state.lastWrittenTimestamp === null ? trade.timestamp_ms : Math.max(state.lastWrittenTimestamp, trade.timestamp_ms);
    await writeLine(outputStream, `${canonicalCsvRow(trade)}\n`);
    writtenRows += 1;
  }

  return {
    source: `${path.basename(zipPath)}:${entry.fileName}`,
    parsedRows,
    writtenRows,
    skippedRows,
    skippedBeforeRange,
    stoppedAfterRange,
    adjacentDuplicateRows,
  };
}

async function main() {
  const symbol = readString("BINANCE_SYMBOL", readString("SYMBOL", "SOLUSDT")).toUpperCase();
  const dataType = readString("BINANCE_DATA_TYPE", "trades");
  if (!["trades", "aggTrades"].includes(dataType)) {
    throw new Error('BINANCE_DATA_TYPE must be "trades" or "aggTrades".');
  }

  const currentMonth = new Date().toISOString().slice(0, 7);
  const startMonth = readString("BINANCE_START_MONTH", "2024-01");
  const endMonth = readString("BINANCE_END_MONTH", currentMonth);
  const baseUrl = readString("BINANCE_BASE_URL", "https://data.binance.vision/data/spot/monthly");
  const outputPath = resolvePath(
    readString("BINANCE_OUTPUT_PATH", path.join("data", "historical_trades", "binance", `${symbol}_trades.csv`))
  );
  const downloadDirectory = resolvePath(
    readString("BINANCE_DOWNLOAD_DIR", path.join("data", "downloads", "binance", dataType, symbol))
  );
  const startMs = parseTimeMs(process.env.BINANCE_START_TIME || process.env.TRADE_START_TIME || process.env.START_TIME);
  const endMs = parseTimeMs(process.env.BINANCE_END_TIME || process.env.TRADE_END_TIME || process.env.END_TIME);
  const maxRetries = Math.max(1, Math.floor(readNumber("BINANCE_DOWNLOAD_MAX_RETRIES", 5)));

  const months = listMonths(startMonth, endMonth);
  ensureDirectory(downloadDirectory);
  ensureDirectory(path.dirname(outputPath));

  console.log("Binance historical market-trade downloader");
  console.log(`SYMBOL: ${symbol}`);
  console.log(`BINANCE_DATA_TYPE: ${dataType}`);
  console.log(`Months: ${startMonth} -> ${endMonth}`);
  console.log(`Time filter: ${startMs === null ? "none" : new Date(startMs).toISOString()} -> ${endMs === null ? "none" : new Date(endMs).toISOString()}`);
  console.log(`Output: ${outputPath}`);
  console.log("Normalized columns: timestamp_ms,price,size,side");
  console.log("Streaming ZIP CSV entries directly; no decompressed CSV files or all-row arrays are held in memory.");
  console.log("Taker side inference: isBuyerMaker=true => sell; false => buy.");

  const outputStream = fs.createWriteStream(outputPath, { flags: "w", encoding: "utf8" });
  await writeLine(outputStream, "timestamp_ms,price,size,side\n");

  const downloadedFiles = [];
  const cachedFiles = [];
  const missingFiles = [];
  const state = {
    firstTimestamp: null,
    lastTimestamp: null,
    lastWrittenTimestamp: null,
    lastDedupeKey: null,
    nonMonotonicRows: 0,
    stoppedAfterRange: false,
  };

  let totalZipCsvEntries = 0;
  let totalParsedRows = 0;
  let totalWrittenRows = 0;
  let totalSkippedRows = 0;
  let totalSkippedBeforeRange = 0;
  let totalStoppedAfterRange = 0;
  let totalAdjacentDuplicateRows = 0;

  try {
    for (const month of months) {
      if (state.stoppedAfterRange) break;

      const name = zipFileName(symbol, dataType, month);
      const url = downloadUrl(baseUrl, symbol, dataType, month);
      const zipPath = path.join(downloadDirectory, name);
      const status = await downloadZip(url, zipPath, maxRetries);

      if (status === "missing") {
        missingFiles.push(name);
        console.log(`Missing: ${name}`);
        continue;
      }
      if (status === "cached") {
        cachedFiles.push(name);
        console.log(`Cached: ${name}`);
      } else {
        downloadedFiles.push(name);
        console.log(`Downloaded: ${name}`);
      }

      const entries = readZipEntries(zipPath).filter((entry) => entry.fileName.toLowerCase().endsWith(".csv"));
      totalZipCsvEntries += entries.length;

      for (const entry of entries) {
        if (state.stoppedAfterRange) break;
        const summary = await appendParsedTradesFromZipEntry({
          zipPath,
          entry,
          dataType,
          outputStream,
          startMs,
          endMs,
          state,
        });
        totalParsedRows += summary.parsedRows;
        totalWrittenRows += summary.writtenRows;
        totalSkippedRows += summary.skippedRows;
        totalSkippedBeforeRange += summary.skippedBeforeRange;
        totalStoppedAfterRange += summary.stoppedAfterRange;
        totalAdjacentDuplicateRows += summary.adjacentDuplicateRows;
        console.log(
          `Streamed ${summary.source}: parsed=${summary.parsedRows} written=${summary.writtenRows} ` +
          `skipped=${summary.skippedRows} skipped_before_range=${summary.skippedBeforeRange} ` +
          `stopped_after_range=${summary.stoppedAfterRange} adjacent_duplicates=${summary.adjacentDuplicateRows}`
        );
      }
    }
  } finally {
    await new Promise((resolve) => outputStream.end(resolve));
  }

  console.log("\nBinance historical trade diagnostics");
  console.log(`Downloaded ZIP files: ${downloadedFiles.length}`);
  console.log(`Cached ZIP files: ${cachedFiles.length}`);
  console.log(`Missing ZIP files: ${missingFiles.length}`);
  console.log(`Streamed ZIP CSV entries: ${totalZipCsvEntries}`);
  console.log(`Parsed rows: ${totalParsedRows}`);
  console.log(`Written normalized rows: ${totalWrittenRows}`);
  console.log(`Skipped malformed/header rows: ${totalSkippedRows}`);
  console.log(`Skipped before time range: ${totalSkippedBeforeRange}`);
  console.log(`Stopped after time range rows observed: ${totalStoppedAfterRange}`);
  console.log(`Adjacent duplicate rows skipped: ${totalAdjacentDuplicateRows}`);
  console.log(`Non-monotonic rows observed while streaming: ${state.nonMonotonicRows}`);
  console.log(`First written timestamp: ${state.firstTimestamp === null ? "n/a" : new Date(state.firstTimestamp).toISOString()}`);
  console.log(`Last written timestamp: ${state.lastWrittenTimestamp === null ? "n/a" : new Date(state.lastWrittenTimestamp).toISOString()}`);
  console.log(`Saved normalized trade tape to: ${outputPath}`);
  console.log("Paper-only public market data. No private API keys. No orders.");
}

if (require.main === module) {
  main().catch((error) => {
    console.error("Binance trade download failed:", error.message);
    process.exitCode = 1;
  });
}

module.exports = {
  listMonths,
  parseTradeLine,
  parseBinanceTimestamp,
  readZipEntries,
  openZipEntryStream,
};
