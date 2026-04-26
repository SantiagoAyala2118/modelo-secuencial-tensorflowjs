const EPOCHS = 700;
let trainedModel = null;

const modeloSecuencial = async () => {
  const trainBtn = document.getElementById("train-btn");
  const statusText = document.getElementById("status-text");
  const progress = document.getElementById("progress-fill");
  const badge = document.getElementById("model-badge");

  trainBtn.disabled = true;
  badge.className = "badge badge-running";
  badge.textContent = "Entrenando...";

  //* Se inicializa el modelo
  const model = tf.sequential();

  //* Se define cuantas capas y neuronas va a poseer el modelo
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  //* Se prepara el modelo para el entrenamiento
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  //* Valores de entrada y salida para y = 2x + 6
  const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
  const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1]);

  //* Entrenamiento con callbacks para actualizar la UI
  await model.fit(xs, ys, {
    epochs: EPOCHS,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const pct = (((epoch + 1) / EPOCHS) * 100).toFixed(1);
        progress.style.width = pct + "%";
        statusText.textContent = `Entrenando modelo... Época ${epoch + 1}/${EPOCHS} - Loss: ${logs.loss.toFixed(4)}`;
      },
    },
  });

  //* Entrenamiento terminado
  trainedModel = model;
  statusText.textContent = `✓ Entrenamiento completado — modelo listo para usar`;
  trainBtn.textContent = "✓ Modelo entrenado";
  badge.className = "badge badge-done";
  badge.textContent = "Listo ✓";
  document.getElementById("x-input").disabled = false;
  document.getElementById("predict-btn").disabled = false;
};

const predict = () => {
  if (!trainedModel) return;
  const xVal = parseFloat(document.getElementById("x-input").value);
  if (isNaN(xVal)) {
    document.getElementById("result-value").textContent =
      "Ingresá un número válido";
    return;
  }
  const result = trainedModel
    .predict(tf.tensor2d([xVal], [1, 1]))
    .dataSync()[0];
  document.getElementById("result-value").textContent = result.toFixed(4);
  document.getElementById("result-hint").textContent =
    `y = f(${xVal}) ≈ ${result.toFixed(4)}`;
};

document
  .getElementById("train-btn")
  .addEventListener("click", modeloSecuencial);
document.getElementById("predict-btn").addEventListener("click", predict);
document.getElementById("x-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") predict();
});
