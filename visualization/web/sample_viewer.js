const BEV_RANGE = { xMin: 0, xMax: 60, yMin: -30, yMax: 30 };
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MODEL_COLORS = ["#22c55e", "#3b82f6", "#f97316", "#e11d48", "#a855f7", "#14b8a6", "#eab308"];

let sampleIds = [];
let currentSampleIdx = 0;
let currentSample = null;
let currentPred = [];

let isPlaying = false;
let playTimer = null;
let isFrameLoading = false;

const point3d = {
  initialized: false,
  mount: null,
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  points: null,
  boxGroup: null,
  frameId: 0,
};

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function colorForExp(expId, index) {
  if (expId === "GT") return "#ffffff";
  return MODEL_COLORS[index % MODEL_COLORS.length];
}

function initPointCloud3D() {
  if (point3d.initialized) return;
  const mount = document.getElementById("point3dMount");
  if (!mount) return;

  text(document.getElementById("point3dStatus"), "加载中");
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#05070d");

  const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 500);
  let renderer = null;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  } catch (err) {
    text(document.getElementById("point3dStatus"), `WebGL 初始化失败: ${String(err.message || err)}`);
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x05070d, 1);
  mount.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 4;
  controls.maxDistance = 150;
  controls.target.set(30, 0, 0);

  const grid = new THREE.GridHelper(60, 12, 0x64748b, 0x1f2937);
  grid.position.set(30, 0, 0);
  scene.add(grid);

  const axes = new THREE.AxesHelper(8);
  axes.position.set(0, 0.02, 0);
  scene.add(axes);

  const boxGroup = new THREE.Group();
  scene.add(boxGroup);

  point3d.initialized = true;
  point3d.mount = mount;
  point3d.scene = scene;
  point3d.camera = camera;
  point3d.renderer = renderer;
  point3d.controls = controls;
  point3d.boxGroup = boxGroup;

  new ResizeObserver(resizePointCloud3D).observe(mount);
  resizePointCloud3D();
  applyCameraPreset();
  animatePointCloud3D();
}

function resizePointCloud3D() {
  if (!point3d.initialized) return;
  const width = Math.max(1, point3d.mount.clientWidth);
  const height = Math.max(1, point3d.mount.clientHeight);
  point3d.renderer.setSize(width, height, false);
  point3d.camera.aspect = width / height;
  point3d.camera.updateProjectionMatrix();
}

function animatePointCloud3D() {
  if (!point3d.initialized) return;
  point3d.frameId = window.requestAnimationFrame(animatePointCloud3D);
  point3d.controls.update();
  point3d.renderer.render(point3d.scene, point3d.camera);
}

function applyCameraPreset() {
  if (!point3d.initialized) return;
  const preset = document.getElementById("cameraPreset")?.value || "perspective";
  const camera = point3d.camera;
  camera.up.set(0, 1, 0);
  point3d.controls.target.set(30, 0, 0);

  if (preset === "top") {
    camera.up.set(0, 0, -1);
    camera.position.set(30, 88, 0.01);
  } else if (preset === "side") {
    camera.position.set(30, 14, 78);
  } else {
    camera.position.set(30, 66, 42);
  }
  camera.lookAt(point3d.controls.target);
  point3d.controls.update();
}

function disposeObject(obj) {
  if (!obj) return;
  if (obj.geometry) obj.geometry.dispose();
  if (obj.material) {
    if (Array.isArray(obj.material)) {
      obj.material.forEach((m) => m.dispose());
    } else {
      obj.material.dispose();
    }
  }
}

function clearPointCloud3D() {
  if (!point3d.initialized) return;
  if (point3d.points) {
    point3d.scene.remove(point3d.points);
    disposeObject(point3d.points);
    point3d.points = null;
  }
  while (point3d.boxGroup.children.length > 0) {
    const child = point3d.boxGroup.children[0];
    point3d.boxGroup.remove(child);
    disposeObject(child);
  }
}

function pointColor(point, mode) {
  const color = new THREE.Color();
  if (mode === "doppler") {
    const vr = Number(point[3] ?? 0);
    const t = clamp01((vr + 8) / 16);
    color.lerpColors(new THREE.Color("#2563eb"), new THREE.Color("#ef4444"), t);
    return color;
  }

  const z = Number(point[2] ?? 0);
  const t = clamp01((z + 3) / 5);
  color.setHSL(0.62 - 0.48 * t, 0.82, 0.56);
  return color;
}

function addRadarPointCloud(points) {
  const mode = document.getElementById("pointColorMode")?.value || "height";
  const pointSize = Number(document.getElementById("pointSize")?.value || 0.08);
  const rows = points || [];
  const positions = new Float32Array(rows.length * 3);
  const colors = new Float32Array(rows.length * 3);

  rows.forEach((p, i) => {
    positions[i * 3 + 0] = Number(p[0] ?? 0);
    positions[i * 3 + 1] = Number(p[2] ?? 0);
    positions[i * 3 + 2] = Number(p[1] ?? 0);
    const c = pointColor(p, mode);
    colors[i * 3 + 0] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({
    size: pointSize,
    sizeAttenuation: true,
    vertexColors: true,
  });
  point3d.points = new THREE.Points(geometry, material);
  point3d.scene.add(point3d.points);
}

function boxCorners3D(box) {
  if (!Array.isArray(box) || box.length < 7) return [];
  const [x, y, z, l, w, h, yaw] = box.map(Number);
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const base = [
    [l / 2, w / 2],
    [l / 2, -w / 2],
    [-l / 2, -w / 2],
    [-l / 2, w / 2],
  ].map(([dx, dy]) => [x + c * dx - s * dy, y + s * dx + c * dy]);
  const zBottom = z - h / 2;
  const zTop = z + h / 2;
  return [
    ...base.map(([bx, by]) => [bx, by, zBottom]),
    ...base.map(([bx, by]) => [bx, by, zTop]),
  ];
}

function toScenePoint([x, y, z]) {
  return [x, z, y];
}

function addBox3D(box, color) {
  const corners = boxCorners3D(box);
  if (corners.length !== 8) return;
  const edges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
  ];
  const positions = new Float32Array(edges.length * 2 * 3);
  edges.forEach(([a, b], i) => {
    const pa = toScenePoint(corners[a]);
    const pb = toScenePoint(corners[b]);
    positions.set(pa, i * 6);
    positions.set(pb, i * 6 + 3);
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color });
  point3d.boxGroup.add(new THREE.LineSegments(geometry, material));
}

function renderPointCloud3D() {
  initPointCloud3D();
  if (!point3d.initialized) return;
  clearPointCloud3D();

  if (!currentSample) {
    text(document.getElementById("point3dStatus"), "无样本");
    document.getElementById("point3dLegend").innerHTML = "";
    return;
  }

  const points = currentSample.radar_points || [];
  addRadarPointCloud(points);

  const classFilter = document.getElementById("classFilter").value;
  const scoreThresh = Number(document.getElementById("scoreThresh").value);
  const showGt = document.getElementById("show3dGt")?.checked ?? true;
  const showPred = document.getElementById("show3dPred")?.checked ?? true;
  const legend = [];
  let gtCount = 0;
  let predCount = 0;

  if (showGt) {
    const gtBoxes = filteredGtBoxes(currentSample.gt_boxes || [], classFilter);
    gtCount = gtBoxes.length;
    gtBoxes.forEach((g) => addBox3D(g.box_lidar, "#ffffff"));
    legend.push(`<span style="border-color:#fff;color:#111;background:#fff">GT (${gtCount})</span>`);
  }

  if (showPred) {
    let colorIdx = 0;
    selectedExperiments().forEach((expId) => {
      const pred = (currentPred || []).find((x) => x.exp_id === expId);
      const color = colorForExp(expId, colorIdx++);
      const boxes = filteredPredBoxes(pred ? pred.pred_boxes : [], classFilter, scoreThresh);
      predCount += boxes.length;
      boxes.forEach((b) => addBox3D(b.box_lidar, color));
      legend.push(`<span style="border-color:${color};color:${color}">${expId} (${boxes.length})</span>`);
    });
  }

  text(document.getElementById("point3dStatus"), `points=${points.length} | boxes=${gtCount + predCount}`);
  document.getElementById("point3dLegend").innerHTML = legend.join("");
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
  renderPointCloud3D();
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
  document.getElementById("pointColorMode").addEventListener("change", renderPointCloud3D);
  document.getElementById("pointSize").addEventListener("input", renderPointCloud3D);
  document.getElementById("cameraPreset").addEventListener("change", applyCameraPreset);
  document.getElementById("show3dGt").addEventListener("change", renderPointCloud3D);
  document.getElementById("show3dPred").addEventListener("change", renderPointCloud3D);

  updatePlayUi();
}

bootstrap().catch((err) => {
  text(document.getElementById("sampleMeta"), `加载失败: ${String(err)}`);
});
