function resetNode(target){
  while (target.firstChild) {
      target.removeChild(target.firstChild);
  }
}

function plotLoss(plotDiv, loss, color, title, xaxis, yaxis) {
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
      y: [
        [loss]
      ]
    }, [0])
  }
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

class UserInterface {
  constructor(args) {
  }

  getSamples() {
    return Number.parseInt(document.getElementById('samples').value);
  }

  getBatches() {
    return Number.parseInt(document.getElementById('batches').value);
  }

  getEpochs() {
    return Number.parseInt(document.getElementById('epochs').value);
  }

  getTestSampleSize() {
    return Number.parseInt(document.getElementById('test-sample-size').value);
  }

  getLabel() {
    return Number.parseInt(document.getElementById('label').value);
  }

  setRetrainFunction(retrain) {
    const retrainButton = document.getElementById('retrain');
    retrainButton.addEventListener('click', async () => retrain());
  }

  setTestFunction(test) {
    const retrainButton = document.getElementById('test');
    retrainButton.addEventListener('click', async () => test());
  }

  setStatus(status) {
    const statusElement = document.getElementById('status');
    statusElement.innerText = status;
  }

  logMessage(message) {
    const loggingElement = document.getElementById('logging-message');
    loggingElement.innerText += message;
  }


  plotTrainLoss(loss) {
    const trainLossCanvasElement = document.getElementById('trainLossCanvas');
    plotLoss(trainLossCanvasElement, loss, 'blue', 'Train Loss', 'batch', 'loss');
  }

  plotValLoss(loss) {
    const valLossCanvasElement = document.getElementById('valLossCanvas');
    plotLoss(valLossCanvasElement, loss, 'red', 'Validation Loss', 'epoch', 'loss');
  }

  updateProgressBar(epoch, epochs) {
    const trainProg = document.getElementById('trainProg');
    trainProg.value = (epoch + 1) / epochs * 100;
  }

  async showTestResults(inputs, outputs) {
    const [zs, label] = inputs;
    const testingElement = document.getElementById('testing');
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

      const z = Object.values(zs.slice([i], [1]).dataSync()).map((el) => {
        return Number(el.toFixed(2));
      });
      const latent = document.createElement('div');
      latent.className = 'latent-label';
      latent.innerText = `${label}, z: ${z}`;

      div.appendChild(latent);
      div.appendChild(canvas);

      testingElement.appendChild(div);
    }
  }

  clear(){
    resetNode(document.getElementById('trainLossCanvas'));
    resetNode(document.getElementById('valLossCanvas'));
    resetNode(document.getElementById('logging-message'));    
  }
}
