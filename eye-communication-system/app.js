const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const commands = ["SPACE", "DELETE", "SPEAK", "CLEAR"];
const quickPhrases = ["Yes", "No", "Water", "Pain", "Help", "Food", "Rest", "Thank you"];

const board = document.querySelector("#board");
const quickActions = document.querySelector("#quickActions");
const message = document.querySelector("#message");
const statusText = document.querySelector("#status");
const gazeDot = document.querySelector("#gazeDot");
const calibrationTarget = document.querySelector("#calibrationTarget");
const calibrationCount = document.querySelector("#calibrationCount");
const progressBar = document.querySelector("#progressBar");
const dwellTimeInput = document.querySelector("#dwellTime");
const dwellLabel = document.querySelector("#dwellLabel");

let gazeSamples = JSON.parse(localStorage.getItem("eyeTalkSamples") || "[]");
let activeTile = null;
let activeStartedAt = 0;
let dwellTime = Number(dwellTimeInput.value);

function makeTile(label, className = "") {
  const tile = document.createElement("button");
  tile.className = `tile ${className}`.trim();
  tile.type = "button";
  tile.dataset.value = label;
  tile.innerHTML = `<span>${label}</span><span class="fill"></span>`;
  tile.addEventListener("click", () => selectValue(label));
  return tile;
}

function buildBoard() {
  quickPhrases.forEach((phrase) => quickActions.appendChild(makeTile(phrase, "quick")));
  [...letters, ...commands].forEach((item) => board.appendChild(makeTile(item)));
}

function selectValue(value) {
  if (value === "SPACE") {
    message.value += " ";
  } else if (value === "DELETE") {
    message.value = message.value.slice(0, -1);
  } else if (value === "SPEAK") {
    speakMessage();
  } else if (value === "CLEAR") {
    message.value = "";
  } else if (quickPhrases.includes(value)) {
    message.value += `${value} `;
  } else {
    message.value += value;
  }
}

function speakMessage() {
  const text = message.value.trim();
  if (!text) return;
  speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

function updateDwellLabel() {
  dwellTime = Number(dwellTimeInput.value);
  dwellLabel.textContent = `${(dwellTime / 1000).toFixed(1)}s`;
}

function tileAtPoint(x, y) {
  const element = document.elementFromPoint(x, y);
  return element?.closest?.(".tile") || null;
}

function resetTile(tile) {
  if (!tile) return;
  tile.classList.remove("active");
  const fill = tile.querySelector(".fill");
  if (fill) fill.style.height = "0";
}

function processGaze(data) {
  if (!data) return;
  const x = Math.max(0, Math.min(window.innerWidth, data.x));
  const y = Math.max(0, Math.min(window.innerHeight, data.y));
  gazeDot.style.transform = `translate(${x}px, ${y}px)`;

  const tile = tileAtPoint(x, y);
  if (tile !== activeTile) {
    resetTile(activeTile);
    activeTile = tile;
    activeStartedAt = Date.now();
  }

  if (!activeTile) return;

  activeTile.classList.add("active");
  const elapsed = Date.now() - activeStartedAt;
  const fill = activeTile.querySelector(".fill");
  if (fill) fill.style.height = `${Math.min(100, (elapsed / dwellTime) * 100)}%`;

  if (elapsed >= dwellTime) {
    selectValue(activeTile.dataset.value);
    resetTile(activeTile);
    activeTile = null;
    activeStartedAt = Date.now();
  }
}

async function startWebGazer() {
  if (!window.webgazer) {
    statusText.textContent = "WebGazer not loaded";
    return;
  }

  webgazer
    .setRegression("ridge")
    .setTracker("clmtrackr")
    .showVideoPreview(true)
    .showPredictionPoints(false)
    .setGazeListener((data) => processGaze(data));

  await webgazer.begin();
  statusText.textContent = "Look at a tile";
}

function calibrationPoints() {
  const padX = window.innerWidth * 0.12;
  const padY = window.innerHeight * 0.12;
  const centerX = window.innerWidth / 2;
  const centerY = window.innerHeight / 2;
  return [
    [padX, padY],
    [centerX, padY],
    [window.innerWidth - padX, padY],
    [padX, centerY],
    [centerX, centerY],
    [window.innerWidth - padX, centerY],
    [padX, window.innerHeight - padY],
    [centerX, window.innerHeight - padY],
    [window.innerWidth - padX, window.innerHeight - padY],
  ];
}

async function autoCalibrate() {
  const points = calibrationPoints();
  statusText.textContent = "Calibrating...";
  calibrationTarget.classList.remove("hidden");

  for (let index = 0; index < points.length; index += 1) {
    const [x, y] = points[index];
    calibrationTarget.style.left = `${x}px`;
    calibrationTarget.style.top = `${y}px`;
    calibrationCount.textContent = `${index + 1} / ${points.length}`;
    progressBar.style.width = `${((index + 1) / points.length) * 100}%`;

    for (let sample = 0; sample < 6; sample += 1) {
      webgazer.recordScreenPosition(x, y, "click");
      gazeSamples.push({ x, y, createdAt: new Date().toISOString() });
      await new Promise((resolve) => setTimeout(resolve, 120));
    }
  }

  localStorage.setItem("eyeTalkSamples", JSON.stringify(gazeSamples.slice(-1000)));
  calibrationTarget.classList.add("hidden");
  statusText.textContent = "Calibration ready";
}

function exportData() {
  const header = "x,y,createdAt\n";
  const rows = gazeSamples.map((sample) => `${sample.x},${sample.y},${sample.createdAt}`).join("\n");
  const blob = new Blob([header + rows], { type: "text/csv" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "eye-gaze-training-data.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

buildBoard();
updateDwellLabel();
dwellTimeInput.addEventListener("input", updateDwellLabel);
document.querySelector("#startCalibration").addEventListener("click", autoCalibrate);
document.querySelector("#speak").addEventListener("click", speakMessage);
document.querySelector("#clear").addEventListener("click", () => {
  message.value = "";
});
document.querySelector("#exportData").addEventListener("click", exportData);

window.addEventListener("load", startWebGazer);
