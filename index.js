class sampleLayer extends tf.layers.Layer {
  constructor(args) {
    super({});
  }

  computeOutputShape(inputShape) {
    return inputShape[0];
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      const [z_mean, z_log_var] = inputs;
      const batch = z_mean.shape[0];
      const dim = z_mean.shape[1];
      const epsilon = tf.randomNormal([batch, dim]);
      const half = tf.scalar(0.5);
      const temp = z_log_var.mul(half).exp().mul(epsilon);
      const sample = z_mean.add(temp);
      return sample;
    });
  }

  getClassName() {
    return 'sampleLayer';
  }
}

const SPACE = `&nbsp;`;
const TAB = `&emsp;`;
const LF = `<br>`;

async function train(config) {
  // # network parameters
  const input_shape = config.input_shape;
  const original_dim = config.original_dim;
  const intermediate_dim = config.intermediate_dim;
  const batch_size = config.batch_size;
  const num_batch = config.num_batch;
  const latent_dim = config.latent_dim;
  const epochs = config.epochs;

  // # VAE model = encoder + decoder
  // # build encoder model
  const encoder_inputs = tf.input({shape: [original_dim]});
  const x_encoder = tf.layers.dense({units: intermediate_dim, activation: 'relu'}).apply(encoder_inputs);
  const z_mean = tf.layers.dense({units: latent_dim, name: 'z_mean'}).apply(x_encoder);
  const z_log_var = tf.layers.dense({units: latent_dim, name: 'z_log_var'}).apply(x_encoder);

  // # use reparameterization trick to push the sampling out as input
  // # note that "output_shape" isn't necessary with the TensorFlow backend
  const z = new sampleLayer().apply([z_mean, z_log_var]);

  // # build decoder model
  const decoder_inputs = tf.input({shape: [latent_dim]});
  const x_decoder = tf.layers.dense({units: intermediate_dim, activation: 'relu'}).apply(decoder_inputs);
  const decoder_outputs = tf.layers.dense({units: original_dim, activation: 'sigmoid'}).apply(x_decoder);

  //  # instantiate VAE model
  // const model = tf.model({
  //   inputs: inputs,
  //   outputs: [z_mean, z_log_var, decoder_outputs],
  //   name: 'vae_mlp'
  // });
  const encoder = tf.model({
    inputs: encoder_inputs,
    outputs: [
      z_mean, z_log_var, z
    ],
    name: "encoder"
  })

  const decoder = tf.model({inputs: decoder_inputs, outputs: decoder_outputs, name: "decoder"})

  const vae = (inputs) => {
    return tf.tidy(() => {
      const [z_mean, z_log_var, z] = encoder.apply(inputs);
      const outputs = decoder.apply(z);
      return [z_mean, z_log_var, outputs];
    })
  }

  const optimizer = tf.train.adam();

  const reconstructionLoss = (yTrue, yPred) => {
    return tf.tidy(() => {
      let reconstruction_loss;
      reconstruction_loss = tf.metrics.binaryCrossentropy(yTrue, yPred)
      reconstruction_loss = reconstruction_loss.mul(tf.scalar(yPred.shape[1]));
      return reconstruction_loss;
    });
  }

  const klLoss = (z_mean, z_log_var) => {
    return tf.tidy(() => {
      let kl_loss;
      kl_loss = tf.scalar(1).add(z_log_var).sub(z_mean.square()).sub(z_log_var.exp());
      kl_loss = tf.sum(kl_loss, -1);
      kl_loss = kl_loss.mul(tf.scalar(-0.5));
      return kl_loss;
    });
  }

  const vaeLoss = (yTrue, yPred) => {
    return tf.tidy(() => {
      // K.max(y_pred,0)-y_pred * y_true + K.log(1+K.exp((-1)*K.abs(y_pred)))
      const [z_mean, z_log_var, y] = yPred;
      // const reconstruction_loss = binaryCrossentropy(yTrue, y);
      const reconstruction_loss = reconstructionLoss(yTrue, y);
      const kl_loss = klLoss(z_mean, z_log_var);
      const total_loss = tf.mean(reconstruction_loss.add(kl_loss));
      return total_loss;
    });
  }

  for (let i = 0; i < epochs; i++) {
    let batch_input;
    let test_batch_input;
    let loss;
    let validation_loss;
    let test_batch_result;
    let epoch_loss;

    logMessage(`[Epoch ${i + 1}]\n`);
    epoch_loss = 0;
    for (let j = 0; j < num_batch; j++) {
      batch_input = data.nextTrainBatch(batch_size).xs.reshape([batch_size, original_dim]);
      loss = await optimizer.minimize(() => vaeLoss(batch_input, vae(batch_input)), true).data();
      loss = Number(loss);
      epoch_loss = epoch_loss + loss;
      logMessage(`\t[Batch ${j + 1}] Training Loss: ${loss}.\n`);
      plotLosses(loss);
      await tf.nextFrame();
    }
    epoch_loss = epoch_loss / num_batch;
    logMessage(`\t[Average] Training Loss: ${epoch_loss}.\n`);
    updateProgressBar(i, epochs);
    await tf.nextFrame();
  }

  return [encoder, decoder];
}

let isTrained = false;
let data;
async function load() {
  data = new MnistData();
  await data.load();
}

let encoder,
  decoder;
async function loadAndTrain() {
  setStatus('Loading data...\n');
  await load();

  setStatus('Training...\n');
  [encoder, decoder] = await train({
    input_shape: 784,
    original_dim: 784,
    intermediate_dim: 512,
    batch_size: 128,
    num_batch: 50,
    latent_dim: 2,
    epochs: getEpochs()
  });
  setStatus('Model Trained.\n');
  isTrained = true;
}

async function test() {
  if (isTrained) {
    const testSampleSize = getTestSampleSize();
    const zs = tf.randomNormal([testSampleSize, 2]);
    const outputs = decoder.apply(zs);
    await showTestResults(zs, outputs);
  }
}

function main() {
  setRetrainFunction(loadAndTrain);
  setTestFunction(test);
}

main();
