const fs = require("fs");
const tf = require("@tensorflow/tfjs");
const {
  saveModelToDirectory
} = require("../src/modelStorage");
const {
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
  classCounts
} = require("./realtimeFlowFeatures");

function readNumberSetting(name, defaultValue) {
  const rawValue = process.env[name];

  if (rawValue === undefined || rawValue.trim() === "") {
    return defaultValue;
  }

  const parsedValue = Number.parseFloat(rawValue);

  if (!Number.isFinite(parsedValue)) {
    throw new Error(`${name} must be a number.`);
  }

  return parsedValue;
}

function readIntegerSetting(name, defaultValue) {
  const rawValue = process.env[name];

  if (rawValue === undefined || rawValue.trim() === "") {
    return defaultValue;
  }

  const parsedValue = Number.parseInt(rawValue, 10);

  if (!Number.isInteger(parsedValue)) {
    throw new Error(`${name} must be a whole number.`);
  }

  return parsedValue;
}

function readStringSetting(name, defaultValue) {
  const rawValue = process.env[name];

  if (rawValue === undefined || rawValue.trim() === "") {
    return defaultValue;
  }

  return rawValue.trim();
}

function splitChronologically(values) {
  const trainEnd = Math.floor(values.length * 0.7);
  const validationEnd = Math.floor(values.length * 0.85);

  return {
    train: values.slice(0, trainEnd),
    validation: values.slice(trainEnd, validationEnd),
    test: values.slice(validationEnd)
  };
}

function buildModel(inputSize) {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
      inputShape: [inputSize]
    })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.1 }));
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(readNumberSetting("FLOW_LEARNING_RATE", 0.001)),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  return model;
}

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

function createConfusionMatrix(actualLabels, predictedLabels) {
  const matrix = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];

  actualLabels.forEach((actual, index) => {
    matrix[actual][predictedLabels[index]] += 1;
  });

  return matrix;
}

function evaluatePredictions(actualLabels, predictedLabels) {
  const confusionMatrix = createConfusionMatrix(actualLabels, predictedLabels);
  const correct = actualLabels.filter(
    (actual, index) => actual === predictedLabels[index]
  ).length;
  const accuracy = actualLabels.length > 0 ? correct / actualLabels.length : 0;
  const perClass = CLASS_NAMES.map((name, classIndex) => {
    const truePositive = confusionMatrix[classIndex][classIndex];
    let predictedAsClass = 0;
    let actualClassCount = 0;

    for (let index = 0; index < 3; index += 1) {
      predictedAsClass += confusionMatrix[index][classIndex];
      actualClassCount += confusionMatrix[classIndex][index];
    }

    return {
      name,
      precision:
        predictedAsClass > 0 ? truePositive / predictedAsClass : 0,
      recall:
        actualClassCount > 0 ? truePositive / actualClassCount : 0,
      support: actualClassCount
    };
  });

  return {
    accuracy,
    confusionMatrix,
    perClass
  };
}

function printClassCounts(label, labels) {
  const counts = classCounts(labels);
  const total = labels.length || 1;

  console.log(`${label} class distribution:`);
  CLASS_NAMES.forEach((name, index) => {
    console.log(
      `- ${index} ${name}: ${counts[index]} ` +
        `(${((counts[index] / total) * 100).toFixed(2)}%)`
    );
  });
}

function printEvaluation(title, evaluation) {
  console.log(`\n${title}`);
  console.log(`Accuracy: ${(evaluation.accuracy * 100).toFixed(2)}%`);
  console.log("Confusion matrix rows=actual, columns=predicted:");
  console.log("          bearish neutral bullish");
  evaluation.confusionMatrix.forEach((row, index) => {
    console.log(
      `${CLASS_NAMES[index].padStart(7)} ${row
        .map((value) => String(value).padStart(7))
        .join(" ")}`
    );
  });
  console.log("Per-class precision/recall:");
  evaluation.perClass.forEach((item) => {
    console.log(
      `- ${item.name}: precision ${(item.precision * 100).toFixed(2)}%, ` +
        `recall ${(item.recall * 100).toFixed(2)}%, support ${item.support}`
    );
  });
}

function getMajorityClass(labels) {
  const counts = classCounts(labels);
  return counts.indexOf(Math.max(...counts));
}

async function trainRealtimeFlowModel() {
  const symbol = getSymbol();
  const csvPath = getRealtimeCsvPath(symbol);
  const modelDirectory = getModelDirectory(symbol);
  const metadataPath = getMetadataPath(modelDirectory);
  const horizon = readIntegerSetting("FLOW_HORIZON", 3);
  const threshold = readNumberSetting("FLOW_TARGET_THRESHOLD", 0.0005);
  const labelPriceField = readStringSetting("FLOW_LABEL_PRICE", "mid_price");
  const epochs = readIntegerSetting("FLOW_EPOCHS", 30);
  const batchSize = readIntegerSetting("FLOW_BATCH_SIZE", 64);

  if (!["close", "mid_price"].includes(labelPriceField)) {
    throw new Error('FLOW_LABEL_PRICE must be "close" or "mid_price".');
  }

  const { rows, rawRowCount, invalidRowCount } = loadValidRealtimeRows(csvPath);
  const examples = buildLabeledExamples(rows, {
    horizon,
    threshold,
    labelPriceField
  });

  if (examples.inputs.length < 100) {
    throw new Error(
      `Need more realtime rows before training. Usable examples: ${examples.inputs.length}.`
    );
  }

  const inputSplits = splitChronologically(examples.inputs);
  const labelSplits = splitChronologically(examples.labels);
  const normalization = calculateNormalization(inputSplits.train);
  const xTrainValues = applyNormalization(inputSplits.train, normalization);
  const xValidationValues = applyNormalization(inputSplits.validation, normalization);
  const xTestValues = applyNormalization(inputSplits.test, normalization);
  const inputSize = BASE_FEATURE_NAMES.length;

  const xTrain = tf.tensor2d(xTrainValues, [xTrainValues.length, inputSize]);
  const yTrain = tf.tensor2d(oneHot(labelSplits.train), [labelSplits.train.length, 3]);
  const xValidation = tf.tensor2d(xValidationValues, [
    xValidationValues.length,
    inputSize
  ]);
  const yValidation = tf.tensor2d(oneHot(labelSplits.validation), [
    labelSplits.validation.length,
    3
  ]);
  const xTest = tf.tensor2d(xTestValues, [xTestValues.length, inputSize]);

  const model = buildModel(inputSize);

  console.log("Realtime flow model training");
  console.log(`Symbol: ${symbol}`);
  console.log(`Realtime CSV: ${csvPath}`);
  console.log(`Raw rows: ${rawRowCount}`);
  console.log(`Invalid rows filtered: ${invalidRowCount}`);
  console.log(`Usable labeled examples: ${examples.inputs.length}`);
  console.log(`Horizon: ${horizon} completed 1m candles`);
  console.log(`Target threshold: ${(threshold * 100).toFixed(4)}%`);
  console.log(`Label price field: ${labelPriceField}`);
  console.log(`Feature count: ${inputSize}`);
  console.log(`Features: ${BASE_FEATURE_NAMES.join(", ")}`);
  console.log(
    `Chronological split: train ${xTrainValues.length}, ` +
      `validation ${xValidationValues.length}, test ${xTestValues.length}`
  );
  console.log(
    "This is an entry-timing / confirmation model, not a replacement for " +
      "the historical candle model."
  );
  printClassCounts("Train", labelSplits.train);
  printClassCounts("Validation", labelSplits.validation);
  printClassCounts("Test", labelSplits.test);

  await model.fit(xTrain, yTrain, {
    epochs,
    batchSize,
    shuffle: false,
    validationData: [xValidation, yValidation],
    callbacks: [
      tf.callbacks.earlyStopping({
        monitor: "val_loss",
        patience: 5
      })
    ]
  });

  const testProbabilities = await model.predict(xTest).array();
  const predictedLabels = testProbabilities.map(argMax);
  const testEvaluation = evaluatePredictions(labelSplits.test, predictedLabels);
  const majorityClass = getMajorityClass(labelSplits.train);
  const baselinePredictions = Array(labelSplits.test.length).fill(majorityClass);
  const baselineEvaluation = evaluatePredictions(
    labelSplits.test,
    baselinePredictions
  );

  printEvaluation("Test evaluation", testEvaluation);
  console.log(
    `\nMajority-class baseline predicts: ${majorityClass} ` +
      `${CLASS_NAMES[majorityClass]}`
  );
  printEvaluation("Baseline evaluation", baselineEvaluation);

  await saveModelToDirectory(model, modelDirectory);
  fs.writeFileSync(
    metadataPath,
    JSON.stringify(
      {
        symbol,
        createdAt: new Date().toISOString(),
        modelType: "realtime_flow_dense_softmax",
        purpose:
          "Entry-timing / confirmation model; not a replacement for the historical candle model.",
        horizon,
        threshold,
        labelPriceField,
        featureNames: BASE_FEATURE_NAMES,
        normalization,
        trainRows: xTrainValues.length,
        validationRows: xValidationValues.length,
        testRows: xTestValues.length,
        classNames: CLASS_NAMES
      },
      null,
      2
    )
  );

  console.log(`\nSaved realtime flow model to: ${modelDirectory}`);
  console.log(`Saved normalization metadata to: ${metadataPath}`);

  xTrain.dispose();
  yTrain.dispose();
  xValidation.dispose();
  yValidation.dispose();
  xTest.dispose();
  model.dispose();
}

if (require.main === module) {
  trainRealtimeFlowModel().catch((error) => {
    console.error("Realtime flow training failed:", error.message);
    process.exitCode = 1;
  });
}

module.exports = {
  trainRealtimeFlowModel,
  evaluatePredictions,
  buildModel
};
