const globalState = {
  colors: new Map(),
  labelCache: new Map(),
  radarCache: new Map(),
  imageCache: new Map(),
};

const serverBadge = document.getElementById("serverBadge");

function colorForClass(name) {
  if (globalState.colors.has(name)) return globalState.colors.get(name);
  const palette = [
    "#ff715b", "#2a9d8f", "#264653", "#e76f51", "#8ab17d", "#118ab2",
    "#ef476f", "#3a86ff", "#fb5607", "#06d6a0", "#8338ec", "#ffbe0b",
  ];
  const c = palette[globalState.colors.size % palette.length];
  globalState.colors.set(name, c);
  return c;
}

function setBadge(text, ok) {
  serverBadge.textContent = text;
  serverBadge.classList.remove("badge-ok", "badge-warn");
  serverBadge.classList.add(ok ? "badge-ok" : "badge-warn");
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function clampInt(v, minV, maxV, fallback) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(minV, Math.min(maxV, Math.round(n)));
}

function bevColor(v, vMin, vMax) {
  const denom = Math.max(1e-6, vMax - vMin);
  const t = Math.max(0, Math.min(1, (v - vMin) / denom));
  const r = Math.round(255 * t);
  const g = Math.round(80 + 120 * (1 - Math.abs(t - 0.5) * 2));
  const b = Math.round(255 * (1 - t));
  return `rgb(${r},${g},${b})`;
}

function createPlayer(prefix, split) {
  const player = {
    prefix,
    split,
    ids: [],
    currentIdx: 0,
    timer: null,
    playing: false,
    renderToken: 0,
    fpsInput: document.getElementById(`${prefix}FpsInput`),
    stepInput: document.getElementById(`${prefix}StepInput`),
    pointInput: document.getElementById(`${prefix}PointInput`),
    loadBtn: document.getElementById(`${prefix}LoadBtn`),
    playBtn: document.getElementById(`${prefix}PlayBtn`),
    prevBtn: document.getElementById(`${prefix}PrevBtn`),
    nextBtn: document.getElementById(`${prefix}NextBtn`),
    frameSlider: document.getElementById(`${prefix}FrameSlider`),
    frameInfo: document.getElementById(`${prefix}FrameInfo`),
    cacheInfo: document.getElementById(`${prefix}CacheInfo`),
    classLegend: document.getElementById(`${prefix}ClassLegend`),
    canvas: document.getElementById(`${prefix}ViewerCanvas`),
  };
  player.ctx = player.canvas.getContext("2d");
  return player;
}

function drawClassLegend(player, names) {
  const sortedNames = [...new Set(names || [])].sort();
  player.classLegend.innerHTML = "";
  for (const name of sortedNames) {
    const span = document.createElement("span");
    span.className = "legend-item";
    span.style.borderColor = colorForClass(name);
    span.textContent = name;
    player.classLegend.appendChild(span);
  }
}

function loadImage(sampleId) {
  if (globalState.imageCache.has(sampleId)) return globalState.imageCache.get(sampleId);
  const p = new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`image load failed: ${sampleId}`));
    img.src = `/data/image/${sampleId}.jpg`;
  });
  globalState.imageCache.set(sampleId, p);
  return p;
}

async function getLabels(sampleId) {
  if (globalState.labelCache.has(sampleId)) return globalState.labelCache.get(sampleId);
  const data = await fetchJson(`/api/labels/${encodeURIComponent(sampleId)}`);
  const labels = data.labels || [];
  globalState.labelCache.set(sampleId, labels);
  return labels;
}

async function getRadar(sampleId, maxPoints) {
  const key = `${sampleId}:${maxPoints}:label`;
  if (globalState.radarCache.has(key)) return globalState.radarCache.get(key);
  const url =
    `/api/radar/${encodeURIComponent(sampleId)}?max_points=${maxPoints}&x_min=0&x_max=60&y_min=-30&y_max=30&color_by=label`;
  const data = await fetchJson(url);
  const payload = {
    mode: data.mode || "velocity",
    class_names: data.class_names || ["background"],
    label_hist: data.label_hist || {},
    points: data.points || [],
  };
  globalState.radarCache.set(key, payload);
  return payload;
}

function drawImagePane(ctx, img, labels, pane) {
  ctx.fillStyle = "#101316";
  ctx.fillRect(pane.x, pane.y, pane.w, pane.h);

  const imgW = img.naturalWidth;
  const imgH = img.naturalHeight;
  const scale = Math.min(pane.w / imgW, pane.h / imgH);
  const drawW = imgW * scale;
  const drawH = imgH * scale;
  const dx = pane.x + (pane.w - drawW) * 0.5;
  const dy = pane.y + (pane.h - drawH) * 0.5;
  ctx.drawImage(img, dx, dy, drawW, drawH);

  for (const item of labels) {
    const [x1, y1, x2, y2] = item.bbox;
    const rx = dx + x1 * scale;
    const ry = dy + y1 * scale;
    const rw = Math.max(1, (x2 - x1) * scale);
    const rh = Math.max(1, (y2 - y1) * scale);
    const color = colorForClass(item.name);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.fillStyle = color;
    ctx.font = "12px IBM Plex Mono, Consolas, monospace";
    const text = item.name;
    const tw = ctx.measureText(text).width + 8;
    const ty = Math.max(0, ry - 16);
    ctx.fillRect(rx, ty, tw, 14);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, rx + 4, ty + 11);
  }

  ctx.fillStyle = "#e8f3ee";
  ctx.font = "bold 14px Space Grotesk, Noto Sans SC, sans-serif";
  ctx.fillText("Image + Labels", pane.x + 10, pane.y + 20);
}

function drawBevPane(ctx, radarData, pane) {
  const points = radarData?.points || [];
  const mode = radarData?.mode || "velocity";
  const classNames = radarData?.class_names || ["background"];

  ctx.fillStyle = "#0f1316";
  ctx.fillRect(pane.x, pane.y, pane.w, pane.h);

  const padL = 36;
  const padR = 12;
  const padT = 18;
  const padB = 26;
  const plotX = pane.x + padL;
  const plotY = pane.y + padT;
  const plotW = pane.w - padL - padR;
  const plotH = pane.h - padT - padB;

  ctx.strokeStyle = "#22303a";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 6; i++) {
    const gx = plotX + (plotW * i) / 6;
    ctx.beginPath();
    ctx.moveTo(gx, plotY);
    ctx.lineTo(gx, plotY + plotH);
    ctx.stroke();
  }
  for (let i = 0; i <= 6; i++) {
    const gy = plotY + (plotH * i) / 6;
    ctx.beginPath();
    ctx.moveTo(plotX, gy);
    ctx.lineTo(plotX + plotW, gy);
    ctx.stroke();
  }

  const xMin = 0.0;
  const xMax = 60.0;
  const yMin = -30.0;
  const yMax = 30.0;
  let vMin = -10.0;
  let vMax = 10.0;
  if (points.length) {
    vMin = points[0][2];
    vMax = points[0][2];
    for (const p of points) {
      if (p[2] < vMin) vMin = p[2];
      if (p[2] > vMax) vMax = p[2];
    }
    if (Math.abs(vMax - vMin) < 1e-6) {
      vMin -= 1.0;
      vMax += 1.0;
    }
  }

  for (const p of points) {
    const x = p[0];
    const y = p[1];
    const v = p[2];
    const tx = (x - xMin) / (xMax - xMin);
    const ty = (y - yMin) / (yMax - yMin);
    if (tx < 0 || tx > 1 || ty < 0 || ty > 1) continue;
    const px = plotX + tx * plotW;
    const py = plotY + (1.0 - ty) * plotH;
    if (mode === "label" && p.length > 3) {
      const clsId = Math.max(0, Math.floor(Number(p[3]) || 0));
      const clsName = classNames[clsId] || `class_${clsId}`;
      ctx.fillStyle = clsId === 0 ? "#6a7783" : colorForClass(clsName);
    } else {
      ctx.fillStyle = bevColor(v, vMin, vMax);
    }
    ctx.fillRect(px, py, 2, 2);
  }

  ctx.strokeStyle = "#d8ecff";
  ctx.lineWidth = 1.2;
  ctx.strokeRect(plotX, plotY, plotW, plotH);
  ctx.fillStyle = "#e8f3ee";
  ctx.font = "bold 14px Space Grotesk, Noto Sans SC, sans-serif";
  ctx.fillText(mode === "label" ? "Radar BEV (Label Color)" : "Radar BEV (Velocity Color)", pane.x + 10, pane.y + 20);
  ctx.font = "12px IBM Plex Mono, Consolas, monospace";
  if (mode === "label") {
    const nonBg = Object.entries(radarData?.label_hist || {})
      .filter(([k]) => k !== "background")
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([k, v]) => `${k}:${v}`)
      .join(" | ");
    ctx.fillText(`points=${points.length} | ${nonBg || "no foreground points"}`, pane.x + 10, pane.y + pane.h - 8);
  } else {
    ctx.fillText(`points=${points.length} | vel=[${vMin.toFixed(2)}, ${vMax.toFixed(2)}]`, pane.x + 10, pane.y + pane.h - 8);
  }
}

function drawFrame(player, img, labels, radarData, sampleId) {
  const ctx = player.ctx;
  const w = player.canvas.width;
  const h = player.canvas.height;
  const half = Math.floor(w / 2);
  ctx.clearRect(0, 0, w, h);
  drawImagePane(ctx, img, labels, { x: 0, y: 0, w: half, h });
  drawBevPane(ctx, radarData, { x: half, y: 0, w: w - half, h });
  ctx.strokeStyle = "#ffffff33";
  ctx.beginPath();
  ctx.moveTo(half + 0.5, 0);
  ctx.lineTo(half + 0.5, h);
  ctx.stroke();

  const radarPoints = radarData?.points || [];
  player.frameInfo.textContent =
    `split=${player.split} | id=${sampleId} | frame=${player.currentIdx + 1}/${player.ids.length} | labels=${labels.length} | bev_points=${radarPoints.length}`;
  player.cacheInfo.textContent =
    `cache labels=${globalState.labelCache.size}, radar=${globalState.radarCache.size}, images=${globalState.imageCache.size}`;
  const names = labels.map((x) => x.name);
  for (const n of radarData?.class_names || []) {
    if (n !== "background") names.push(n);
  }
  drawClassLegend(player, names);
}

async function renderCurrentFrame(player) {
  if (!player.ids.length) {
    player.ctx.clearRect(0, 0, player.canvas.width, player.canvas.height);
    player.frameInfo.textContent = `split=${player.split} | No samples loaded`;
    return;
  }

  const token = ++player.renderToken;
  const sampleId = player.ids[player.currentIdx];
  const maxPoints = clampInt(player.pointInput.value, 500, 30000, 8000);

  try {
    const [img, labels, radarData] = await Promise.all([
      loadImage(sampleId),
      getLabels(sampleId),
      getRadar(sampleId, maxPoints),
    ]);
    if (token !== player.renderToken) return;
    drawFrame(player, img, labels, radarData, sampleId);
  } catch (err) {
    if (token !== player.renderToken) return;
    player.ctx.clearRect(0, 0, player.canvas.width, player.canvas.height);
    player.frameInfo.textContent = String(err);
  }
}

function stepFrame(player, delta) {
  if (!player.ids.length) return;
  const n = player.ids.length;
  player.currentIdx = (player.currentIdx + delta + n) % n;
  player.frameSlider.value = String(player.currentIdx);
  renderCurrentFrame(player);
}

function stopPlay(player) {
  if (!player.playing) return;
  clearInterval(player.timer);
  player.timer = null;
  player.playing = false;
  player.playBtn.textContent = "Play";
}

function startPlay(player) {
  if (player.playing) return;
  const fps = clampInt(player.fpsInput.value, 1, 30, 8);
  const step = clampInt(player.stepInput.value, 1, 30, 1);
  player.playing = true;
  player.playBtn.textContent = "Pause";
  player.timer = setInterval(() => stepFrame(player, step), Math.floor(1000 / fps));
}

async function loadSplit(player) {
  stopPlay(player);
  player.currentIdx = 0;
  const data = await fetchJson(`/api/samples?split=${encodeURIComponent(player.split)}&limit=10000&offset=0`);
  player.ids = data.ids || [];
  player.frameSlider.min = "0";
  player.frameSlider.max = String(Math.max(0, player.ids.length - 1));
  player.frameSlider.value = "0";
  await renderCurrentFrame(player);
}

function bindPlayer(player) {
  player.loadBtn.addEventListener("click", () => loadSplit(player).catch((e) => (player.frameInfo.textContent = String(e))));
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
  player.pointInput.addEventListener("change", () => renderCurrentFrame(player));
}

const trainPlayer = createPlayer("train", "train");
const testPlayer = createPlayer("test", "test");
bindPlayer(trainPlayer);
bindPlayer(testPlayer);

async function bootstrap() {
  try {
    await fetchJson("/api/health");
    setBadge("Server: Online", true);
  } catch (err) {
    setBadge("Server: Offline", false);
    throw err;
  }
  await loadSplit(trainPlayer);
  await loadSplit(testPlayer);
}

bootstrap().catch((err) => {
  console.error(err);
  trainPlayer.frameInfo.textContent = String(err);
  testPlayer.frameInfo.textContent = String(err);
});
