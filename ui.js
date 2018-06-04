const statusElement = document.getElementById('status');
const loggingElement = document.getElementById('logging-message');
const trainingElement = document.getElementById('training');
const testingElement = document.getElementById('testing');
const lossCanvasElement = document.getElementById('lossCanvas');


function getEpochs() {
  return Number.parseInt(document.getElementById('epochs').value);
}

function getTestSampleSize() {
  return Number.parseInt(document.getElementById('test-sample-size').value);
}

function setRetrainFunction(retrain) {
  const retrainButton = document.getElementById('retrain');
  retrainButton.addEventListener('click', async () => retrain());
}

function setTestFunction(test) {
  const retrainButton = document.getElementById('test');
  retrainButton.addEventListener('click', async () => test());
}

function setStatus(status) {
  statusElement.innerText = status;
}

function logMessage(message) {
  loggingElement.innerText += message;
}

function plotLosses(loss) {
  if (!lossCanvasElement.hasChildNodes()) {
    var trace = {
      y: [loss],
      type: "scatter",
      mode: 'lines'
    };
    var data = [trace];
    var layout = {
      title: "Training Loss",
      xaxis: {title: "batch"},
      yaxis: {title: "loss"}
    };
    Plotly.newPlot(lossCanvasElement, data, layout);
  }
  else {
    Plotly.extendTraces(lossCanvasElement, {y: [[loss]]}, [0])
  }
}

function updateProgressBar(epoch, epochs) {
  const trainProg = document.getElementById('trainProg');
  trainProg.value = (epoch + 1) / epochs * 100;
}

function draw(image, canvas) {
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

async function showTestResults(zs, outputs) {
  testingElement.style.display = "block";
  const testExamples = zs.shape[0];
  for (let i = 0; i < testExamples; i++) {
    const image = outputs.slice([i, 0], [1, outputs.shape[1]]);

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
