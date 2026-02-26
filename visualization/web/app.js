const state = {
  summary: null,
  history: [],
  ids: [],
  currentIdx: 0,
  timer: null,
  playing: false,
  labelCache: new Map(),
  colors: new Map(),
};

const healthBadge = document.getElementById("healthBadge");
const historyBadge = document.getElementById("historyBadge");
const historyPath = document.getElementById("historyPath");
const metricCards = document.getElementById("metricCards");
const latestTableBody = document.querySelector("#latestTable tbody");
const datasetGrid = document.getElementById("datasetGrid");
const classBars = document.getElementById("classBars");
const splitSelect = document.getElementById("splitSelect");
const fpsInput = document.getElementById("fpsInput");
const stepInput = document.getElementById("stepInput");
const loadBtn = document.getElementById("loadBtn");
const playBtn = document.getElementById("playBtn");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const frameSlider = document.getElementById("frameSlider");
const currentInfo = document.getElementById("currentInfo");
const legend = document.getElementById("legend");
const canvas = document.getElementById("viewerCanvas");
const ctx = canvas.getContext("2d");
const f1Chart = document.getElementById("f1Chart");
const chartCtx = f1Chart.getContext("2d");

function colorForClass(name) {
  if (state.colors.has(name)) return state.colors.get(name);
  const palette = [
    "#ff715b", "#2a9d8f", "#264653", "#e76f51", "#8ab17d", "#118ab2", "#ef476f",
    "#3a86ff", "#fb5607", "#06d6a0", "#8338ec", "#ffbe0b",
  ];
  const c = palette[state.colors.size % palette.length];
  state.colors.set(name, c);
  return c;
}

function setBadge(el, text, ok) {
  el.textContent = text;
  el.classList.remove("badge-ok", "badge-warn");
  el.classList.add(ok ? "badge-ok" : "badge-warn");
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function num(v) {
  if (typeof v !== "number") return String(v);
  return Number.isInteger(v) ? String(v) : v.toFixed(4);
}

function renderMetricCards(latest, best) {
  metricCards.innerHTML = "";
  const cards = [
    ["Best mean_f1", best?.mean_f1 ?? 0],
    ["Latest mean_f1", latest?.mean_f1 ?? 0],
    ["Latest train_loss", latest?.loss ?? 0],
    ["Latest val_loss", latest?.val_loss ?? 0],
  ];
  for (const [k, v] of cards) {
    const div = document.createElement("div");
    div.className = "card";
    div.innerHTML = `<div class="card-title">${k}</div><div class="card-value">${num(v)}</div>`;
    metricCards.appendChild(div);
  }
}

function renderLatestTable(latest) {
  latestTableBody.innerHTML = "";
  const keys = [
    "epoch", "loss", "val_loss",
    "Car_precision", "Car_recall", "Car_f1",
    "Pedestrian_precision", "Pedestrian_recall", "Pedestrian_f1",
    "Cyclist_precision", "Cyclist_recall", "Cyclist_f1",
    "mean_f1",
  ];
  for (const key of keys) {
    if (!(key in latest)) continue;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${key}</td><td>${num(latest[key])}</td>`;
    latestTableBody.appendChild(tr);
  }
}

function renderF1Chart(history) {
  const w = f1Chart.width;
  const h = f1Chart.height;
  chartCtx.clearRect(0, 0, w, h);

  chartCtx.fillStyle = "#fff";
  chartCtx.fillRect(0, 0, w, h);
  chartCtx.strokeStyle = "#d8e3d6";
  chartCtx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = 20 + (h - 40) * (i / 5);
    chartCtx.beginPath();
    chartCtx.moveTo(40, y);
    chartCtx.lineTo(w - 20, y);
    chartCtx.stroke();
  }
  chartCtx.fillStyle = "#5b6666";
  chartCtx.font = "12px IBM Plex Mono, Consolas, monospace";
  chartCtx.fillText("mean_f1", 42, 16);

  if (!history.length) return;
  const values = history.map(x => Number(x.mean_f1 ?? 0));
  const maxY = Math.max(0.01, ...values) * 1.05;
  const minY = 0.0;
  const n = values.length;

  chartCtx.strokeStyle = "#0f8b8d";
  chartCtx.lineWidth = 2.5;
  chartCtx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = 40 + (w - 60) * (i / Math.max(1, n - 1));
    const y = 20 + (h - 40) * (1 - (values[i] - minY) / Math.max(1e-6, maxY - minY));
    if (i === 0) chartCtx.moveTo(x, y);
    else chartCtx.lineTo(x, y);
  }
  chartCtx.stroke();

  chartCtx.fillStyle = "#1d2a2a";
  chartCtx.fillText(`max=${maxY.toFixed(3)}`, w - 110, 16);
}

function renderDatasetSummary(summary) {
  datasetGrid.innerHTML = "";
  const c = summary.counts || {};
  const items = [
    ["images", c.images ?? 0],
    ["radar_bins", c.radar_bins ?? 0],
    ["labels", c.labels ?? 0],
    ["calib", c.calib ?? 0],
    ["split_train", c.split_train ?? 0],
    ["split_val", c.split_val ?? 0],
    ["split_train_val", c.split_train_val ?? 0],
  ];
  for (const [k, v] of items) {
    const div = document.createElement("div");
    div.className = "kv";
    div.innerHTML = `<div class="k">${k}</div><div class="v">${num(v)}</div>`;
    datasetGrid.appendChild(div);
  }
}

function renderClassBars(hist) {
  classBars.innerHTML = "";
  const entries = Object.entries(hist || {});
  if (!entries.length) return;
  const maxVal = Math.max(...entries.map(x => x[1]));
  for (const [cls, val] of entries.slice(0, 20)) {
    const ratio = (val / maxVal) * 100;
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div>${cls}</div>
      <div class="bar"><span style="width:${ratio.toFixed(2)}%"></span></div>
      <div>${val}</div>
    `;
    classBars.appendChild(row);
  }
}

async function loadHistory() {
  const data = await fetchJson("/api/history");
  state.history = data.history || [];
  setBadge(historyBadge, `History: ${data.exists ? "Loaded" : "Missing"}`, data.exists);
  historyPath.textContent = data.path || "";
  renderMetricCards(data.latest || {}, data.best_mean_f1 || {});
  renderLatestTable(data.latest || {});
  renderF1Chart(state.history);
}

async function loadSummary() {
  const data = await fetchJson("/api/summary");
  state.summary = data;
  renderDatasetSummary(data);
  renderClassBars(data.class_hist_train_val || {});
}

async function loadSampleIds() {
  const split = splitSelect.value;
  const data = await fetchJson(`/api/samples?split=${encodeURIComponent(split)}&limit=5000&offset=0`);
  state.ids = data.ids || [];
  state.currentIdx = 0;
  frameSlider.min = "0";
  frameSlider.max = String(Math.max(0, state.ids.length - 1));
  frameSlider.value = "0";
  await renderCurrentFrame();
}

async function getLabels(sampleId) {
  if (state.labelCache.has(sampleId)) return state.labelCache.get(sampleId);
  const data = await fetchJson(`/api/labels/${encodeURIComponent(sampleId)}`);
  const labels = data.labels || [];
  state.labelCache.set(sampleId, labels);
  return labels;
}

function drawLegend(labels) {
  const names = [...new Set(labels.map(x => x.name))].sort();
  legend.innerHTML = "";
  for (const name of names) {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.style.borderColor = colorForClass(name);
    item.textContent = name;
    legend.appendChild(item);
  }
}

function drawFrame(image, labels, sampleId) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const imgW = image.naturalWidth;
  const imgH = image.naturalHeight;
  const scale = Math.min(canvas.width / imgW, canvas.height / imgH);
  const drawW = imgW * scale;
  const drawH = imgH * scale;
  const dx = (canvas.width - drawW) / 2;
  const dy = (canvas.height - drawH) / 2;

  ctx.drawImage(image, dx, dy, drawW, drawH);

  for (const item of labels) {
    const [x1, y1, x2, y2] = item.bbox;
    const rx = dx + x1 * scale;
    const ry = dy + y1 * scale;
    const rw = (x2 - x1) * scale;
    const rh = (y2 - y1) * scale;
    const color = colorForClass(item.name);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.fillStyle = color;
    const text = item.name;
    ctx.font = "13px IBM Plex Mono, Consolas, monospace";
    const tw = ctx.measureText(text).width + 8;
    ctx.fillRect(rx, Math.max(0, ry - 18), tw, 16);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, rx + 4, Math.max(12, ry - 6));
  }

  currentInfo.textContent = `id=${sampleId} | frame=${state.currentIdx + 1}/${state.ids.length} | labels=${labels.length}`;
  drawLegend(labels);
}

async function renderCurrentFrame() {
  if (!state.ids.length) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    currentInfo.textContent = "No samples loaded";
    return;
  }
  const sampleId = state.ids[state.currentIdx];
  const labels = await getLabels(sampleId);

  const img = new Image();
  img.onload = () => drawFrame(img, labels, sampleId);
  img.onerror = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    currentInfo.textContent = `Image load failed: ${sampleId}`;
  };
  img.src = `/data/image/${sampleId}.jpg`;
}

function stepFrame(delta) {
  if (!state.ids.length) return;
  const n = state.ids.length;
  state.currentIdx = (state.currentIdx + delta + n) % n;
  frameSlider.value = String(state.currentIdx);
  renderCurrentFrame();
}

function startPlay() {
  if (state.playing) return;
  const fps = Math.max(1, Math.min(20, Number(fpsInput.value) || 6));
  const step = Math.max(1, Math.min(20, Number(stepInput.value) || 1));
  state.playing = true;
  playBtn.textContent = "Pause";
  state.timer = setInterval(() => stepFrame(step), Math.floor(1000 / fps));
}

function stopPlay() {
  if (!state.playing) return;
  clearInterval(state.timer);
  state.timer = null;
  state.playing = false;
  playBtn.textContent = "Play";
}

function bindEvents() {
  loadBtn.addEventListener("click", async () => {
    stopPlay();
    await loadSampleIds();
  });
  playBtn.addEventListener("click", () => {
    if (state.playing) stopPlay();
    else startPlay();
  });
  prevBtn.addEventListener("click", () => stepFrame(-1));
  nextBtn.addEventListener("click", () => stepFrame(1));
  frameSlider.addEventListener("input", () => {
    state.currentIdx = Number(frameSlider.value);
    renderCurrentFrame();
  });
}

async function bootstrap() {
  try {
    await fetchJson("/api/health");
    setBadge(healthBadge, "Server: Online", true);
  } catch (e) {
    setBadge(healthBadge, "Server: Offline", false);
    throw e;
  }
  await loadHistory();
  await loadSummary();
  await loadSampleIds();
}

bindEvents();
bootstrap().catch((err) => {
  console.error(err);
  currentInfo.textContent = String(err);
});

