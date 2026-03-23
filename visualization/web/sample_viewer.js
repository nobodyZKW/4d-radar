const BEV_RANGE = { xMin: 0, xMax: 60, yMin: -30, yMax: 30 };
const MODEL_COLORS = ["#22c55e", "#3b82f6", "#f97316", "#e11d48", "#a855f7", "#14b8a6", "#eab308"];

let sampleIds = [];
let currentSampleIdx = 0;
let currentSample = null;
let currentPred = [];

let isPlaying = false;
let playTimer = null;
let isFrameLoading = false;

function colorForExp(expId, index) {
  if (expId === "GT") return "#ffffff";
  return MODEL_COLORS[index % MODEL_COLORS.length];
}

function toCanvasXY(x, y, width, height) {
  const u = ((x - BEV_RANGE.xMin) / (BEV_RANGE.xMax - BEV_RANGE.xMin)) * width;
  const v = height - ((y - BEV_RANGE.yMin) / (BEV_RANGE.yMax - BEV_RANGE.yMin)) * height;
  return [u, v];
}

function drawRadarPoints(ctx, points, width, height) {
  ctx.fillStyle = "rgba(148, 163, 184, 0.45)";
  points.forEach((p) => {
    const [u, v] = toCanvasXY(Number(p[0]), Number(p[1]), width, height);
    ctx.fillRect(u, v, 2, 2);
  });
}

function drawBox(ctx, box, color, width, height, textLabel = "") {
  if (!Array.isArray(box) || box.length < 7) return;
  const [x, y, _z, l, w, _h, yaw] = box.map(Number);
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const corners = [
    [l / 2, w / 2],
    [l / 2, -w / 2],
    [-l / 2, -w / 2],
    [-l / 2, w / 2],
  ].map(([dx, dy]) => [x + c * dx - s * dy, y + s * dx + c * dy]);
  const pts = corners.map(([cx, cy]) => toCanvasXY(cx, cy, width, height));

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  ctx.stroke();

  const front = toCanvasXY(x + c * l * 0.5, y + s * l * 0.5, width, height);
  const center = toCanvasXY(x, y, width, height);
  ctx.beginPath();
  ctx.moveTo(center[0], center[1]);
  ctx.lineTo(front[0], front[1]);
  ctx.stroke();

  if (textLabel) {
    ctx.fillStyle = color;
    ctx.font = "12px sans-serif";
    ctx.fillText(textLabel, pts[0][0] + 2, pts[0][1] - 4);
  }
}

function selectedExperiments() {
  return Array.from(document.querySelectorAll("input[name='exp_select']:checked")).map((x) => x.value);
}

function filteredPredBoxes(predRows, classFilter, scoreThresh) {
  return (predRows || []).filter((b) => {
    const score = Number(b.score ?? 0);
    const label = String(b.label || "");
    if (score < scoreThresh) return false;
    if (classFilter && label !== classFilter) return false;
    return true;
  });
}

function filteredGtBoxes(gtRows, classFilter) {
  return (gtRows || []).filter((b) => {
    const label = String(b.label || "");
    return !classFilter || label === classFilter;
  });
}

function drawSingleCanvas() {
  const canvas = document.getElementById("bevCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, w, h);

  if (!currentSample) return;
  drawRadarPoints(ctx, currentSample.radar_points || [], w, h);

  const classFilter = document.getElementById("classFilter").value;
  const scoreThresh = Number(document.getElementById("scoreThresh").value);
  const showGt = document.getElementById("showGt").checked;
  const gtBoxes = showGt ? filteredGtBoxes(currentSample.gt_boxes || [], classFilter) : [];
  gtBoxes.forEach((g) => drawBox(ctx, g.box_lidar, "#ffffff", w, h, `GT:${g.label}`));

  const selected = selectedExperiments();
  const legend = [];
  let colorIdx = 0;
  selected.forEach((expId) => {
    const pred = (currentPred || []).find((x) => x.exp_id === expId);
    const color = colorForExp(expId, colorIdx++);
    const boxes = filteredPredBoxes(pred ? pred.pred_boxes : [], classFilter, scoreThresh);
    boxes.forEach((b) => drawBox(ctx, b.box_lidar, color, w, h, `${b.label}:${f4(b.score, 2)}`));
    legend.push(`<span style="border-color:${color};color:${color}">${expId} (${boxes.length})</span>`);
  });
  document.getElementById("bevLegend").innerHTML = [
    `<span style="border-color:#fff;color:#111;background:#fff">GT (${gtBoxes.length})</span>`,
    ...legend,
  ].join("");
}

function drawGridCanvas() {
  const grid = document.getElementById("bevGrid");
  grid.innerHTML = "";
  if (!currentSample) return;
  const classFilter = document.getElementById("classFilter").value;
  const scoreThresh = Number(document.getElementById("scoreThresh").value);
  const showGt = document.getElementById("showGt").checked;
  const selected = selectedExperiments();
  selected.forEach((expId, idx) => {
    const wrap = document.createElement("div");
    wrap.className = "card";
    wrap.innerHTML = `<div class="small">${expId}</div><canvas width="600" height="380"></canvas>`;
    const canvas = wrap.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, w, h);
    drawRadarPoints(ctx, currentSample.radar_points || [], w, h);

    const gtBoxes = showGt ? filteredGtBoxes(currentSample.gt_boxes || [], classFilter) : [];
    gtBoxes.forEach((g) => drawBox(ctx, g.box_lidar, "#ffffff", w, h, ""));

    const pred = (currentPred || []).find((x) => x.exp_id === expId);
    const color = colorForExp(expId, idx);
    const boxes = filteredPredBoxes(pred ? pred.pred_boxes : [], classFilter, scoreThresh);
    boxes.forEach((b) => drawBox(ctx, b.box_lidar, color, w, h, ""));
    grid.appendChild(wrap);
  });
  document.getElementById("bevLegend").innerHTML = `<span>grid 模式：每个卡片一个模型</span>`;
}

function renderBev() {
  const mode = document.getElementById("viewMode").value;
  const single = document.getElementById("bevSingle");
  const grid = document.getElementById("bevGrid");
  if (mode === "grid") {
    single.style.display = "none";
    grid.style.display = "grid";
    drawGridCanvas();
  } else {
    single.style.display = "block";
    grid.style.display = "none";
    drawSingleCanvas();
  }
}

function updatePlayUi() {
  const playBtn = document.getElementById("playBtn");
  const status = document.getElementById("playStatus");
  playBtn.textContent = isPlaying ? "暂停" : "播放";
  status.textContent = isPlaying ? "播放中" : "已暂停";
}

function stopPlay() {
  isPlaying = false;
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
  updatePlayUi();
}

function startPlay() {
  if (sampleIds.length <= 1) return;
  if (isPlaying) return;
  const fps = Math.max(1, Number(document.getElementById("playFps").value || 2));
  const interval = Math.floor(1000 / fps);
  isPlaying = true;
  updatePlayUi();
  playTimer = setInterval(() => {
    if (isFrameLoading) return;
    stepNext(true).catch(console.error);
  }, interval);
}

function togglePlay() {
  if (isPlaying) {
    stopPlay();
  } else {
    startPlay();
  }
}

async function loadSamplePred(sampleId) {
  const ids = selectedExperiments();
  const q = new URLSearchParams();
  if (ids.length > 0) q.set("exp", ids.join(","));
  const predData = await fetchJson(`/api/samples/${encodeURIComponent(sampleId)}/predictions?${q.toString()}`);
  currentPred = predData.predictions || [];
}

function updateSampleMeta() {
  if (!currentSample) {
    text(document.getElementById("sampleMeta"), "无样本");
    return;
  }
  const total = sampleIds.length;
  const idx = currentSampleIdx + 1;
  text(
    document.getElementById("sampleMeta"),
    `sample=${currentSample.sample_id} | frame=${idx}/${total} | gt=${(currentSample.gt_boxes || []).length} | radar_points=${(currentSample.radar_points || []).length}`
  );
}

async function gotoSampleIndex(index) {
  if (sampleIds.length === 0) return;
  if (index < 0 || index >= sampleIds.length) return;
  const sid = sampleIds[index];
  isFrameLoading = true;
  try {
    const data = await fetchJson(`/api/samples/${encodeURIComponent(sid)}?max_points=12000`);
    currentSample = data.sample;
    currentSampleIdx = index;
    document.getElementById("sampleImage").src = currentSample.image_url || "";
    document.getElementById("sampleSelect").value = sid;
    await loadSamplePred(sid);
    updateSampleMeta();
    renderBev();
  } finally {
    isFrameLoading = false;
  }
}

async function stepNext(fromAuto = false) {
  if (sampleIds.length === 0) return;
  const loop = document.getElementById("loopPlay").checked;
  let nextIdx = currentSampleIdx + 1;
  if (nextIdx >= sampleIds.length) {
    if (!loop) {
      if (fromAuto) stopPlay();
      return;
    }
    nextIdx = 0;
  }
  await gotoSampleIndex(nextIdx);
}

async function stepPrev() {
  if (sampleIds.length === 0) return;
  const loop = document.getElementById("loopPlay").checked;
  let prevIdx = currentSampleIdx - 1;
  if (prevIdx < 0) {
    prevIdx = loop ? sampleIds.length - 1 : 0;
  }
  await gotoSampleIndex(prevIdx);
}

async function loadSplitSamples() {
  stopPlay();
  const split = document.getElementById("split").value;
  const data = await fetchJson(`/api/samples?split=${encodeURIComponent(split)}&limit=400&offset=0`);
  sampleIds = data.ids || [];
  currentSampleIdx = 0;
  const select = document.getElementById("sampleSelect");
  select.innerHTML = "";
  sampleIds.forEach((id) => {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id;
    select.appendChild(opt);
  });
  if (sampleIds.length > 0) {
    await gotoSampleIndex(0);
  } else {
    currentSample = null;
    currentPred = [];
    updateSampleMeta();
    renderBev();
  }
}

function renderExpSelector(rows, defaults) {
  const container = document.getElementById("expSelector");
  container.innerHTML = "";
  rows.forEach((row, idx) => {
    const checked = defaults.includes(row.id) ? "checked" : "";
    const color = colorForExp(row.id, idx);
    const card = document.createElement("label");
    card.className = "card";
    card.innerHTML = `
      <div class="row">
        <input type="checkbox" name="exp_select" value="${row.id}" ${checked} />
        <strong>${row.display_name}</strong>
      </div>
      <div class="small">${row.family} | ${row.status}</div>
      <div class="small">mean_f1=${f4((row.quick_summary || {}).mean_f1)}</div>
      <div class="small" style="color:${color}">颜色: ${color}</div>
    `;
    container.appendChild(card);
  });

  container.querySelectorAll("input[name='exp_select']").forEach((el) => {
    el.addEventListener("change", async () => {
      if (!currentSample) return;
      await loadSamplePred(currentSample.sample_id);
      renderBev();
    });
  });
}

function parseDefaultExps() {
  const q = new URLSearchParams(window.location.search);
  const exp = q.get("exp");
  if (exp) return exp.split(",").map((x) => x.trim()).filter(Boolean);
  const raw = localStorage.getItem("compare_selection") || "";
  if (raw) return raw.split(",").map((x) => x.trim()).filter(Boolean);
  return ["baseline_main", "improve_v1_main", "improve_v1_5_main"];
}

async function bootstrap() {
  setActiveNav();
  const expData = await fetchJson("/api/experiments?sort=sort_order");
  const rows = (expData.experiments || []).filter((x) => x.status !== "archived");
  renderExpSelector(rows, parseDefaultExps());

  await loadSplitSamples();

  document.getElementById("split").addEventListener("change", () => {
    loadSplitSamples().catch(console.error);
  });
  document.getElementById("sampleSelect").addEventListener("change", (e) => {
    const sid = e.target.value;
    const idx = sampleIds.indexOf(sid);
    if (idx >= 0) gotoSampleIndex(idx).catch(console.error);
  });
  document.getElementById("prevBtn").addEventListener("click", () => {
    stepPrev().catch(console.error);
  });
  document.getElementById("nextBtn").addEventListener("click", () => {
    stepNext(false).catch(console.error);
  });

  document.getElementById("playBtn").addEventListener("click", togglePlay);
  document.getElementById("playFps").addEventListener("change", () => {
    if (isPlaying) {
      stopPlay();
      startPlay();
    }
  });

  document.getElementById("scoreThresh").addEventListener("input", () => {
    const v = Number(document.getElementById("scoreThresh").value);
    text(document.getElementById("scoreValue"), v.toFixed(2));
    renderBev();
  });
  document.getElementById("classFilter").addEventListener("change", renderBev);
  document.getElementById("viewMode").addEventListener("change", renderBev);
  document.getElementById("showGt").addEventListener("change", renderBev);

  updatePlayUi();
}

bootstrap().catch((err) => {
  text(document.getElementById("sampleMeta"), `加载失败: ${String(err)}`);
});
