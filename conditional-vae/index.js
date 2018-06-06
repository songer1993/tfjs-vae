const ORIGINAL_DIM = 784;
const INTERMEDIATE_DIM = 512;
const LATENT_DIM = 2;
const LABEL_DIM = 10;

const BATCH_SIZE = 128;
const NUM_BATCH = 50;
const TEST_BATCH_SIZE = 1000;

let data;
let model;
let ui = new UserInterface();

async function load() {
  ui.setStatus('Loading data...\n');
  data = new MnistData();
  await data.load();
  ui.setStatus('Data loaded.\n');
}

async function train() {
  if (data.isLoaded){
    ui.clear();
    ui.setStatus('Training...\n');
    model = new ConditionalVAE({
      modelConfig:{
        originalDim: ORIGINAL_DIM,
        intermediateDim: INTERMEDIATE_DIM,
        latentDim: LATENT_DIM,
        labelDim: LABEL_DIM
      },
      trainConfig:{
        batchSize: ui.getSamples(),
        numBatch: ui.getBatches(),
        testBatchSize: TEST_BATCH_SIZE,
        epochs: ui.getEpochs(),
        optimizer: tf.train.adam(),
        logMessage: ui.logMessage,
        plotTrainLoss: ui.plotTrainLoss,
        plotValLoss: ui.plotValLoss,
        updateProgressBar: ui.updateProgressBar
      }
    });
    await model.train(data);
    ui.setStatus('Model Trained.\n');
  }
}

async function test() {
  if (model.isTrained) {
    const testSampleSize = ui.getTestSampleSize();
    const label = ui.getLabel();
    const zs = tf.randomNormal([testSampleSize, 2]);
    const ys = tf.oneHot(tf.ones([testSampleSize], 'int32').mul(tf.scalar(label, 'int32')), LABEL_DIM);
    const outputs = model.decoder.apply([zs, ys]);
    await ui.showTestResults([zs, label], outputs);
  }
}

async function main() {
  await load();
  ui.setRetrainFunction(train);
  ui.setTestFunction(test);
}

main();
