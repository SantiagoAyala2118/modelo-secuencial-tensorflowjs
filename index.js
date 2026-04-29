const EPOCHS = 700;
let trainedModel = null;
let lossChart; // Variable para guardar la instancia del gráfico

// Función para inicializar el gráfico (con tus colores del CSS)
const initChart = () => {
  const ctx = document.getElementById("lossChart").getContext("2d");

  // Si ya existe un gráfico (re-entrenamiento), lo destruimos para empezar de cero
  if (lossChart) lossChart.destroy();

  lossChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [], // Épocas
      datasets: [
        {
          label: "Pérdida",
          data: [], // Valores de loss
          borderColor: "#3cb46a",
          backgroundColor: "rgba(60, 180, 106, 0.1)",
          borderWidth: 2,
          pointRadius: 0, // No mostramos puntos para que sea una línea suave
          fill: true,
          tension: 0.4, // Curva suave
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: "Época",
            color: "#3b6d11",
            font: { family: "JetBrains Mono" },
          },
          grid: { color: "#1a2e1a" },
          ticks: { color: "#3b6d11" },
        },
        y: {
          display: true,
          title: {
            display: true,
            text: "Loss",
            color: "#3b6d11",
            font: { family: "JetBrains Mono" },
          },
          grid: { color: "#1a2e1a" },
          ticks: { color: "#3b6d11" },
        },
      },
    },
  });
};

const modeloSecuencial = async () => {
  const trainBtn = document.getElementById("train-btn");
  const statusText = document.getElementById("status-text");
  const progress = document.getElementById("progress-fill");
  const badge = document.getElementById("model-badge");

  trainBtn.disabled = true;
  badge.className = "badge badge-running";
  badge.textContent = "Entrenando...";

  // Inicializamos el gráfico vacío
  initChart();

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
  const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1]);

  await model.fit(xs, ys, {
    epochs: EPOCHS,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const pct = (((epoch + 1) / EPOCHS) * 100).toFixed(1);
        progress.style.width = pct + "%";
        statusText.textContent = `Época ${epoch + 1}/${EPOCHS} - Loss: ${logs.loss.toFixed(4)}`;

        // --- ACTUALIZACIÓN DEL GRÁFICO EN TIEMPO REAL ---
        // Para no sobrecargar el navegador, actualizamos cada 10 épocas
        if (epoch % 10 === 0 || epoch === EPOCHS - 1) {
          lossChart.data.labels.push(epoch);
          lossChart.data.datasets[0].data.push(logs.loss);
          lossChart.update("none"); // 'none' para que no haga animaciones lentas
        }
      },
    },
  });

  trainedModel = model;
  statusText.textContent = `✓ Entrenamiento completado`;
  trainBtn.textContent = "✓ Modelo entrenado";
  badge.className = "badge badge-done";
  badge.textContent = "Listo ✓";
  document.getElementById("x-input").disabled = false;
  document.getElementById("predict-btn").disabled = false;
};

// ... El resto de tu función predict() y listeners se mantienen igual

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
