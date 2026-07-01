const fs = require("fs");
const tf = require("@tensorflow/tfjs");
const {
  loadModelFromDirectory
} = require("../src/modelStorage");
const {
  CLASS_NAMES,
  getSymbol,
  getRealtimeCsvPath,
  getModelDirectory,
  getMetadataPath,
  loadValidRealtimeRows,
  rowToFeatureVector,
  applyNormalization
} = require("./realtimeFlowFeatures");

function argMax(values) {
  let bestIndex = 0;
  let bestValue = values[0];

  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > bestValue) {
      bestIndex = index;
      bestValue = values[index];
    }
  }

  return bestIndex;
}

function percent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

async function predictRealtimeFlow() {
  const symbol = getSymbol();
  const csvPath = getRealtimeCsvPath(symbol);
  const modelDirectory = getModelDirectory(symbol);
  const metadataPath = getMetadataPath(modelDirectory);

  if (!fs.existsSync(metadataPath)) {
    throw new Error(
      `Normalization metadata not found: ${metadataPath}. Run npm run train-flow first.`
    );
  }

  const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
  const { rows, rawRowCount, invalidRowCount } = loadValidRealtimeRows(csvPath);

  if (rows.length === 0) {
    throw new Error("No valid realtime flow rows are available for prediction.");
  }

  const latestRow = rows[rows.length - 1];
  const rawFeatures = rowToFeatureVector(latestRow);
  const normalizedFeatures = applyNormalization(
    [rawFeatures],
    metadata.normalization
  );
  const model = await loadModelFromDirectory(tf, modelDirectory);
  const inputTensor = tf.tensor2d(
    normalizedFeatures,
    [1, metadata.featureNames.length]
  );
  const outputTensor = model.predict(inputTensor);
  const probabilities = Array.from(await outputTensor.data());
  const predictedClass = argMax(probabilities);

  console.log("Realtime flow prediction");
  console.log(`Symbol: ${symbol}`);
  console.log(`Realtime CSV: ${csvPath}`);
  console.log(`Raw rows: ${rawRowCount}`);
  console.log(`Invalid rows filtered: ${invalidRowCount}`);
  console.log(`Latest valid row: ${latestRow.time}`);
  console.log(`Model directory: ${modelDirectory}`);
  console.log(
    `Purpose: entry-timing / confirmation only, not a replacement for the historical model.`
  );
  console.log(
    `Training horizon: ${metadata.horizon} completed 1m candles; ` +
      `threshold ${(metadata.threshold * 100).toFixed(4)}%; ` +
      `label price ${metadata.labelPriceField}`
  );
  console.log(`Bearish probability: ${percent(probabilities[0])}`);
  console.log(`Neutral probability: ${percent(probabilities[1])}`);
  console.log(`Bullish probability: ${percent(probabilities[2])}`);
  console.log(`Predicted class: ${predictedClass} ${CLASS_NAMES[predictedClass]}`);

  inputTensor.dispose();
  outputTensor.dispose();
  model.dispose();

  return {
    probabilities,
    predictedClass
  };
}

if (require.main === module) {
  predictRealtimeFlow().catch((error) => {
    console.error("Realtime flow prediction failed:", error.message);
    process.exitCode = 1;
  });
}

module.exports = {
  predictRealtimeFlow
};
