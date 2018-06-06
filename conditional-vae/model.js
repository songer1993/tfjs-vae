class sampleLayer extends tf.layers.Layer {
  constructor(args) {
    super({});
  }

  computeOutputShape(inputShape) {
    return inputShape[0];
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      const [zMean, zLogVar] = inputs;
      const batch = zMean.shape[0];
      const dim = zMean.shape[1];
      const epsilon = tf.randomNormal([batch, dim]);
      const half = tf.scalar(0.5);
      const temp = zLogVar.mul(half).exp().mul(epsilon);
      const sample = zMean.add(temp);
      return sample;
    });
  }

  getClassName() {
    return 'sampleLayer';
  }
}


class ConditionalVAE {
  constructor(config) {
    this.modelConfig = config.modelConfig;
    this.trainConfig = config.trainConfig;
    [this.encoder, this.decoder, this.apply] = this.build();
    this.isTrained = false;
  }


  build(modelConfig) {
    if (modelConfig != undefined){
      this.modelConfig = modelConfig;
    }
    const config = this.modelConfig;

    const originalDim = config.originalDim;
    const intermediateDim = config.intermediateDim;
    const latentDim = config.latentDim;
    const labelDim = config.labelDim;

    // VAE model = encoder + decoder
    // build encoder model
    const encoderX = tf.input({shape: [originalDim]});
    const encoderY = tf.input({shape: [labelDim]});
    const encoderInputs = tf.layers.concatenate({axis: 1}).apply([encoderX, encoderY]);
    const x1Linear = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(encoderInputs);
    const x1Normalised = tf.layers.batchNormalization({axis: 1}).apply(x1Linear);
    const x1 = tf.layers.leakyReLU().apply(x1Normalised);
    const zMean = tf.layers.dense({units: latentDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x1);
    const zLogVar = tf.layers.dense({units: latentDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x1);
    const z = new sampleLayer().apply([zMean, zLogVar]);
    const encoderOutputs = [zMean, zLogVar, z];
    const encoder = tf.model({inputs: [encoderX, encoderY], outputs: encoderOutputs, name: "encoder"})

    // build decoder model
    const decoderZ = tf.input({shape: [latentDim]});
    const decoderY = tf.input({shape: [labelDim]});
    const decoderInputs = tf.layers.concatenate({axis: 1}).apply([decoderZ, decoderY]);
    const x2Linear = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(decoderInputs);
    const x2Normalised = tf.layers.batchNormalization({axis: 1}).apply(x2Linear);
    const x2 = tf.layers.leakyReLU().apply(x2Normalised);
    const decoderOutputs = tf.layers.dense({units: originalDim, activation: 'sigmoid'}).apply(x2);
    const decoder = tf.model({inputs: [decoderZ, decoderY], outputs: decoderOutputs, name: "decoder"})

    // build VAE model
    const vae = (inputs) => {
      return tf.tidy(() => {
        const [X, Y] = inputs;
        const [zMean, zLogVar, z] = this.encoder.apply([X, Y]);
        const outputs = this.decoder.apply([z, Y]);
        return [zMean, zLogVar, outputs];
      });
    }

    return [encoder, decoder, vae];
  }


  reconstructionLoss(yTrue, yPred) {
    return tf.tidy(() => {
      let reconstruction_loss;
      reconstruction_loss = tf.metrics.binaryCrossentropy(yTrue, yPred)
      reconstruction_loss = reconstruction_loss.mul(tf.scalar(yPred.shape[1]));
      return reconstruction_loss;
    });
  }

  klLoss(z_mean, z_log_var) {
    return tf.tidy(() => {
      let kl_loss;
      kl_loss = tf.scalar(1).add(z_log_var).sub(z_mean.square()).sub(z_log_var.exp());
      kl_loss = tf.sum(kl_loss, -1);
      kl_loss = kl_loss.mul(tf.scalar(-0.5));
      return kl_loss;
    });
  }

  vaeLoss(yTrue, yPred) {
    return tf.tidy(() => {
      const [z_mean, z_log_var, y] = yPred;
      const reconstruction_loss = this.reconstructionLoss(yTrue, y);
      const kl_loss = this.klLoss(z_mean, z_log_var);
      const total_loss = tf.mean(reconstruction_loss.add(kl_loss));
      return total_loss;
    });
  }

  async train(data, trainConfig) {
    this.isTrained = false;
    if (trainConfig != undefined){
      this.trainConfig = trainConfig;
    }
    const config = this.trainConfig;

    const batchSize = config.batchSize;
    const numBatch = config.numBatch;
    const epochs = config.epochs;
    const testBatchSize = config.testBatchSize;
    const optimizer = config.optimizer;
    const logMessage = config.logMessage;
    const plotTrainLoss = config.plotTrainLoss;
    const plotValLoss = config.plotValLoss;
    const updateProgressBar = config.updateProgressBar;

    const originalDim = this.modelConfig.originalDim;
    const labelDim = this.modelConfig.labelDim;

    for (let i = 0; i < epochs; i++) {
      let batch;
      let batchInput;
      let testBatch;
      let testBatchInput;
      let trainLoss;
      let valLoss;
      let testBatchResult;
      let epochLoss;

      logMessage(`[Epoch ${i + 1}]\n`);
      epochLoss = 0;
      for (let j = 0; j < numBatch; j++) {
        batch = data.nextTrainBatch(batchSize);
        batchInput = [batch.xs.reshape([batchSize, originalDim]), batch.labels.reshape([batchSize, labelDim])]
        trainLoss = await optimizer.minimize(() => this.vaeLoss(batchInput[0], this.apply(batchInput)), true);
        trainLoss = Number(trainLoss.dataSync());
        epochLoss = epochLoss + trainLoss;
        logMessage(`\t[Batch ${j + 1}] Training Loss: ${trainLoss}.\n`);
        plotTrainLoss(trainLoss);
        await tf.nextFrame();
      }
      epochLoss = epochLoss / numBatch;
      logMessage(`\t[Average] Training Loss: ${epochLoss}.\n`);
      updateProgressBar(i, epochs);
      testBatch = data.nextTrainBatch(testBatchSize)
      testBatchInput = [testBatch.xs.reshape([testBatchSize, originalDim]), testBatch.labels.reshape([testBatchSize, labelDim])];
      testBatchResult = this.apply(testBatchInput);
      valLoss = this.vaeLoss(testBatchInput[0], testBatchResult);
      valLoss = Number(valLoss.dataSync());
      plotValLoss(valLoss);
      await tf.nextFrame();
    }
    this.isTrained = true;
  }

}
