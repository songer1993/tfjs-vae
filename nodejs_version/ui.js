// import * as tf from '@tensorflow/tfjs';
import * as Plotly from 'plotly.js';

const statusElement = document.getElementById('status');
const loggingElement = document.getElementById('logging-message');
const trainingElement = document.getElementById('training');
const testingElement = document.getElementById('testing');
const trainLossCanvasElement = document.getElementById('trainLossCanvas');
const valLossCanvasElement = document.getElementById('valLossCanvas');

export function getEpochs() {
  return Number.parseInt(document.getElementById('epochs').value);
}

export function getTestSampleSize() {
  return Number.parseInt(document.getElementById('test-sample-size').value);
}

export function setRetrainFunction(retrain) {
  const retrainButton = document.getElementById('retrain');
  retrainButton.addEventListener('click', async () => retrain());
}

export function setTestFunction(test) {
  const retrainButton = document.getElementById('test');
  retrainButton.addEventListener('click', async () => test());
}

export function setStatus(status) {
  statusElement.innerText = status;
}

export function logMessage(message) {
  loggingElement.innerText += message;
}

export function plotTrainLoss(loss){
  plotLoss(trainLossCanvasElement, loss, 'blue', 'Train Loss', 'batch', 'loss');
}

export function plotValLoss(loss){
  plotLoss(valLossCanvasElement, loss, 'red', 'Validation Loss', 'epoch', 'loss');
}

export function plotLoss(plotDiv, loss, color, title, xaxis, yaxis) {
  if (!plotDiv.hasChildNodes()) {
    const trace = {
      y: [loss],
      type: "scatter",
      mode: "lines",
      marker: {
        color: color
      }
    };
    const data = [trace];
    const layout = {
      title: title,
      xaxis: {
        title: xaxis
      },
      yaxis: {
        title: yaxis
      }
    };
    Plotly.plot(plotDiv, data, layout);
  } else {
    Plotly.extendTraces(plotDiv, {
      y: [[loss]]
    }, [0])
  }
}

export function updateProgressBar(epoch, epochs) {
  const trainProg = document.getElementById('trainProg');
  trainProg.value = (epoch + 1) / epochs * 100;
}

export function draw(image, canvas) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

export async function showTestResults(zs, outputs) {
  testingElement.style.display = "block";
  const testExamples = zs.shape[0];
  for (let i = 0; i < testExamples; i++) {
    const image = outputs.slice([
      i, 0
    ], [
      1, outputs.shape[1]
    ]);

    const div = document.createElement('div');
    div.className = 'result-container';

    const canvas = document.createElement('canvas');
    canvas.className = 'result-canvas';
    draw(image.flatten(), canvas);

    const z = Object.values(await zs.slice([i], [1]).dataSync()).map((el) => {
      return Number(el.toFixed(2));
    });
    const latent = document.createElement('div');
    latent.className = 'latent-label';
    latent.innerText = `z: ${z}`;

    div.appendChild(latent);
    div.appendChild(canvas);

    testingElement.appendChild(div);
  }
}
