const state = {
  experiments: [],
  baselineLatest: null,
};

const serverBadge = document.getElementById("serverBadge");
const expBadge = document.getElementById("expBadge");
const expSelect = document.getElementById("expSelect");
const reloadBtn = document.getElementById("reloadBtn");
const historyPath = document.getElementById("historyPath");
const historyMeta = document.getElementById("historyMeta");
const metricCards = document.getElementById("metricCards");
const latestTableBody = document.querySelector("#latestTable tbody");
const compareBox = document.getElementById("compareBox");

const f1Chart = document.getElementById("f1Chart");
const lossChart = document.getElementById("lossChart");
const f1Ctx = f1Chart.getContext("2d");
const lossCtx = lossChart.getContext("2d");

function resizeCanvas(canvas, targetHeight = 260) {
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

function setBadge(el, text, ok) {
  el.textContent = text;
  el.classList.remove("badge-ok", "badge-warn");
  el.classList.add(ok ? "badge-ok" : "badge-warn");
}

function num(v) {
  if (typeof v !== "number") return String(v ?? "-");
  return Number.isInteger(v) ? String(v) : v.toFixed(4);
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function drawLineChart(ctx, canvas, values, color, title) {
  const { w, h } = resizeCanvas(canvas, 260);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#d8e3d6";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = 20 + (h - 40) * (i / 5);
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(w - 20, y);
    ctx.stroke();
  }
  ctx.fillStyle = "#5b6666";
  ctx.font = "12px IBM Plex Mono, Consolas, monospace";
  ctx.fillText(title, 42, 16);
  if (!values.length) return;
  const maxY = Math.max(1e-6, ...values) * 1.05;
  const minY = Math.min(0, ...values);
  const range = Math.max(1e-6, maxY - minY);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < values.length; i++) {
    const x = 40 + (w - 60) * (i / Math.max(1, values.length - 1));
    const y = 20 + (h - 40) * (1 - (values[i] - minY) / range);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawLossChart(history) {
  const { w, h } = resizeCanvas(lossChart, 260);
  lossCtx.clearRect(0, 0, w, h);
  lossCtx.fillStyle = "#fff";
  lossCtx.fillRect(0, 0, w, h);
  lossCtx.strokeStyle = "#d8e3d6";
  lossCtx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = 20 + (h - 40) * (i / 5);
    lossCtx.beginPath();
    lossCtx.moveTo(40, y);
    lossCtx.lineTo(w - 20, y);
    lossCtx.stroke();
  }
  lossCtx.fillStyle = "#5b6666";
  lossCtx.font = "12px IBM Plex Mono, Consolas, monospace";
  lossCtx.fillText("train/val loss", 42, 16);

  if (!history.length) return;
  const trainLoss = history.map((x) => Number(x.loss_total ?? x.loss ?? 0));
  const valLoss = history.map((x) => Number(x.val_loss_total ?? x.val_loss ?? 0));
  const all = [...trainLoss, ...valLoss];
  const maxY = Math.max(1e-6, ...all) * 1.05;
  const minY = Math.min(0, ...all);
  const range = Math.max(1e-6, maxY - minY);

  function drawOne(values, color) {
    lossCtx.strokeStyle = color;
    lossCtx.lineWidth = 2.1;
    lossCtx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = 40 + (w - 60) * (i / Math.max(1, values.length - 1));
      const y = 20 + (h - 40) * (1 - (values[i] - minY) / range);
      if (i === 0) lossCtx.moveTo(x, y);
      else lossCtx.lineTo(x, y);
    }
    lossCtx.stroke();
  }

  drawOne(trainLoss, "#0f8b8d");
  drawOne(valLoss, "#ff715b");
}

function renderMetricCards(latest, best) {
  metricCards.innerHTML = "";
  const cards = [
    ["Best mean_f1", best?.mean_f1 ?? 0],
    ["Latest mean_f1", latest?.mean_f1 ?? 0],
    ["Latest train_loss", latest?.loss_total ?? latest?.loss ?? 0],
    ["Latest val_loss", latest?.val_loss_total ?? latest?.val_loss ?? 0],
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
    "epoch",
    "loss_total", "loss_hm", "loss_box", "loss_yaw", "loss_vel",
    "val_loss_total", "val_loss_hm", "val_loss_box", "val_loss_yaw", "val_loss_vel",
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

function renderCompare(latest) {
  compareBox.innerHTML = "";
  const base = state.baselineLatest;
  const cur = Number(latest?.mean_f1 ?? 0);
  const baseF1 = Number(base?.mean_f1 ?? 0);
  const delta = cur - baseF1;
  const ratio = baseF1 > 0 ? cur / baseF1 : 0;

  const items = [
    ["baseline mean_f1", baseF1],
    ["current mean_f1", cur],
    ["delta", delta],
    ["ratio", ratio],
  ];
  for (const [k, v] of items) {
    const div = document.createElement("div");
    div.className = "kv";
    div.innerHTML = `<div class="k">${k}</div><div class="v">${num(v)}</div>`;
    compareBox.appendChild(div);
  }
}

function renderExperiments(exps) {
  expSelect.innerHTML = "";
  for (const e of exps) {
    const opt = document.createElement("option");
    opt.value = e.name;
    const score = e.latest_mean_f1 == null ? "-" : Number(e.latest_mean_f1).toFixed(4);
    opt.textContent = `${e.name} | epochs=${e.epochs} | mean_f1=${score}`;
    expSelect.appendChild(opt);
  }
  const preferred = exps.find((x) => x.name === "centerpoint_v1" && x.exists) ? "centerpoint_v1" : "baseline";
  expSelect.value = preferred;
}

async function loadExperiments() {
  const data = await fetchJson("/api/experiments");
  state.experiments = data.experiments || [];
  renderExperiments(state.experiments);
}

async function loadBaselineRef() {
  const data = await fetchJson("/api/history?exp=baseline");
  state.baselineLatest = data.latest || null;
}

async function loadSelectedHistory() {
  const exp = expSelect.value;
  const data = await fetchJson(`/api/history?exp=${encodeURIComponent(exp)}`);
  setBadge(expBadge, `Experiment: ${exp}`, data.exists);
  historyPath.textContent = data.path || "";
  historyMeta.textContent = `exists=${data.exists} | epochs=${(data.history || []).length}`;
  const history = data.history || [];
  const latest = data.latest || {};
  const best = data.best_mean_f1 || {};
  renderMetricCards(latest, best);
  renderLatestTable(latest);
  renderCompare(latest);
  drawLineChart(f1Ctx, f1Chart, history.map((x) => Number(x.mean_f1 ?? 0)), "#0f8b8d", "mean_f1");
  drawLossChart(history);
}

async function bootstrap() {
  try {
    await fetchJson("/api/health");
    setBadge(serverBadge, "Server: Online", true);
  } catch (err) {
    setBadge(serverBadge, "Server: Offline", false);
    throw err;
  }
  await Promise.all([loadExperiments(), loadBaselineRef()]);
  await loadSelectedHistory();
}

reloadBtn.addEventListener("click", () => {
  loadSelectedHistory().catch((err) => console.error(err));
});
expSelect.addEventListener("change", () => {
  loadSelectedHistory().catch((err) => console.error(err));
});

window.addEventListener("resize", () => {
  loadSelectedHistory().catch((err) => console.error(err));
});

bootstrap().catch((err) => {
  console.error(err);
  historyMeta.textContent = String(err);
});
