const state = {
  summary: null,
  history: [],
  labelCache: new Map(),
  colors: new Map(),
};

const healthBadge = document.getElementById("healthBadge");
const historyBadge = document.getElementById("historyBadge");
const historyPath = document.getElementById("historyPath");
const metricCards = document.getElementById("metricCards");
const latestTableBody = document.querySelector("#latestTable tbody");
const datasetGrid = document.getElementById("datasetGrid");
const f1Chart = document.getElementById("f1Chart");
const chartCtx = f1Chart.getContext("2d");

function resizeCanvas(canvas, targetHeight = 280) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = Math.max(320, Math.floor(canvas.clientWidth || targetHeight * 2));
  const cssH = targetHeight;
  const pxW = Math.floor(cssW * dpr);
  const pxH = Math.floor(cssH * dpr);
  if (canvas.width !== pxW || canvas.height !== pxH) {
    canvas.width = pxW;
    canvas.height = pxH;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w: cssW, h: cssH };
}

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
  const { w, h } = resizeCanvas(f1Chart, 280);
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
  const values = history.map((x) => Number(x.mean_f1 ?? 0));
  const maxY = Math.max(0.01, ...values) * 1.05;
  const n = values.length;

  chartCtx.strokeStyle = "#0f8b8d";
  chartCtx.lineWidth = 2.5;
  chartCtx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = 40 + (w - 60) * (i / Math.max(1, n - 1));
    const y = 20 + (h - 40) * (1 - values[i] / Math.max(1e-6, maxY));
    if (i === 0) chartCtx.moveTo(x, y);
    else chartCtx.lineTo(x, y);
  }
  chartCtx.stroke();
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
    ["split_test", c.split_test ?? 0],
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

function renderClassBars(containerId, hist) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const entries = Object.entries(hist || {});
  if (!entries.length) return;
  const maxVal = Math.max(...entries.map((x) => x[1]));
  for (const [cls, val] of entries.slice(0, 20)) {
    const ratio = (val / maxVal) * 100;
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div>${cls}</div>
      <div class="bar"><span style="width:${ratio.toFixed(2)}%"></span></div>
      <div>${val}</div>
    `;
    container.appendChild(row);
  }
}

function renderSplitStats(containerId, splitName, frameCount, hist) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const labelCount = Object.values(hist || {}).reduce((a, b) => a + Number(b || 0), 0);
  const items = [
    ["split", splitName],
    ["frames", frameCount],
    ["labels", labelCount],
    ["classes", Object.keys(hist || {}).length],
  ];
  for (const [k, v] of items) {
    const div = document.createElement("div");
    div.className = "kv";
    div.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    container.appendChild(div);
  }
}

async function loadHistory() {
  const data = await fetchJson("/api/history?exp=baseline");
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

  const counts = data.counts || {};
  const hist = data.class_hist || {};
  renderSplitStats("trainStats", "train", counts.split_train || 0, hist.train || {});
  renderSplitStats("testStats", "test", counts.split_test || 0, hist.test || {});
  renderClassBars("trainClassBars", hist.train || {});
  renderClassBars("testClassBars", hist.test || {});
}

async function getLabels(sampleId) {
  if (state.labelCache.has(sampleId)) return state.labelCache.get(sampleId);
  const data = await fetchJson(`/api/labels/${encodeURIComponent(sampleId)}`);
  const labels = data.labels || [];
  state.labelCache.set(sampleId, labels);
  return labels;
}

function createPlayer(prefix, split) {
  const player = {
    split,
    ids: [],
    currentIdx: 0,
    timer: null,
    playing: false,
    currentInfo: document.getElementById(`${prefix}CurrentInfo`),
    fpsInput: document.getElementById(`${prefix}FpsInput`),
    stepInput: document.getElementById(`${prefix}StepInput`),
    loadBtn: document.getElementById(`${prefix}LoadBtn`),
    playBtn: document.getElementById(`${prefix}PlayBtn`),
    prevBtn: document.getElementById(`${prefix}PrevBtn`),
    nextBtn: document.getElementById(`${prefix}NextBtn`),
    frameSlider: document.getElementById(`${prefix}FrameSlider`),
    legend: document.getElementById(`${prefix}Legend`),
    canvas: document.getElementById(`${prefix}ViewerCanvas`),
  };
  player.ctx = player.canvas.getContext("2d");
  return player;
}

function drawLegend(player, labels) {
  const names = [...new Set(labels.map((x) => x.name))].sort();
  player.legend.innerHTML = "";
  for (const name of names) {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.style.borderColor = colorForClass(name);
    item.textContent = name;
    player.legend.appendChild(item);
  }
}

function drawFrame(player, image, labels, sampleId) {
  const { ctx, canvas } = player;
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

  player.currentInfo.textContent = `split=${player.split} | id=${sampleId} | frame=${player.currentIdx + 1}/${player.ids.length} | labels=${labels.length}`;
  drawLegend(player, labels);
}

async function renderCurrentFrame(player) {
  if (!player.ids.length) {
    player.ctx.clearRect(0, 0, player.canvas.width, player.canvas.height);
    player.currentInfo.textContent = `split=${player.split} | No samples loaded`;
    return;
  }
  const sampleId = player.ids[player.currentIdx];
  const labels = await getLabels(sampleId);

  const img = new Image();
  img.onload = () => drawFrame(player, img, labels, sampleId);
  img.onerror = () => {
    player.ctx.clearRect(0, 0, player.canvas.width, player.canvas.height);
    player.currentInfo.textContent = `Image load failed: ${sampleId}`;
  };
  img.src = `/data/image/${sampleId}.jpg`;
}

function stepFrame(player, delta) {
  if (!player.ids.length) return;
  const n = player.ids.length;
  player.currentIdx = (player.currentIdx + delta + n) % n;
  player.frameSlider.value = String(player.currentIdx);
  renderCurrentFrame(player);
}

function startPlay(player) {
  if (player.playing) return;
  const fps = Math.max(1, Math.min(20, Number(player.fpsInput.value) || 6));
  const step = Math.max(1, Math.min(20, Number(player.stepInput.value) || 1));
  player.playing = true;
  player.playBtn.textContent = "Pause";
  player.timer = setInterval(() => stepFrame(player, step), Math.floor(1000 / fps));
}

function stopPlay(player) {
  if (!player.playing) return;
  clearInterval(player.timer);
  player.timer = null;
  player.playing = false;
  player.playBtn.textContent = "Play";
}

async function loadSampleIds(player) {
  const data = await fetchJson(`/api/samples?split=${encodeURIComponent(player.split)}&limit=5000&offset=0`);
  player.ids = data.ids || [];
  player.currentIdx = 0;
  player.frameSlider.min = "0";
  player.frameSlider.max = String(Math.max(0, player.ids.length - 1));
  player.frameSlider.value = "0";
  await renderCurrentFrame(player);
}

function bindPlayerEvents(player) {
  player.loadBtn.addEventListener("click", async () => {
    stopPlay(player);
    await loadSampleIds(player);
  });
  player.playBtn.addEventListener("click", () => {
    if (player.playing) stopPlay(player);
    else startPlay(player);
  });
  player.prevBtn.addEventListener("click", () => stepFrame(player, -1));
  player.nextBtn.addEventListener("click", () => stepFrame(player, 1));
  player.frameSlider.addEventListener("input", () => {
    player.currentIdx = Number(player.frameSlider.value);
    renderCurrentFrame(player);
  });
}

const trainPlayer = createPlayer("train", "train");
const testPlayer = createPlayer("test", "test");
bindPlayerEvents(trainPlayer);
bindPlayerEvents(testPlayer);

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
  await loadSampleIds(trainPlayer);
  await loadSampleIds(testPlayer);
}

bootstrap().catch((err) => {
  console.error(err);
  trainPlayer.currentInfo.textContent = String(err);
  testPlayer.currentInfo.textContent = String(err);
});

window.addEventListener("resize", () => {
  renderF1Chart(state.history || []);
});
