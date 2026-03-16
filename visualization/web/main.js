const healthBadge = document.getElementById("healthBadge");
const baselineMeta = document.getElementById("baselineMeta");
const v1Meta = document.getElementById("v1Meta");
const datasetGrid = document.getElementById("datasetGrid");

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

function renderDataset(summary) {
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
  datasetGrid.innerHTML = "";
  for (const [k, v] of items) {
    const div = document.createElement("div");
    div.className = "kv";
    div.innerHTML = `<div class="k">${k}</div><div class="v">${num(v)}</div>`;
    datasetGrid.appendChild(div);
  }
}

function pickExp(exps, name) {
  return (exps || []).find((x) => x.name === name) || null;
}

async function bootstrap() {
  await fetchJson("/api/health");
  setBadge(healthBadge, "Server: Online", true);

  const [summary, expsData] = await Promise.all([fetchJson("/api/summary"), fetchJson("/api/experiments")]);
  renderDataset(summary);

  const exps = expsData.experiments || [];
  const base = pickExp(exps, "baseline");
  const v1 = pickExp(exps, "centerpoint_v1");
  baselineMeta.textContent = base
    ? `epochs=${base.epochs} | latest mean_f1=${base.latest_mean_f1 == null ? "-" : num(base.latest_mean_f1)}`
    : "baseline history missing";
  v1Meta.textContent = v1
    ? `epochs=${v1.epochs} | latest mean_f1=${v1.latest_mean_f1 == null ? "-" : num(v1.latest_mean_f1)}`
    : "v1 history missing";
}

bootstrap().catch((err) => {
  console.error(err);
  setBadge(healthBadge, "Server: Offline", false);
  baselineMeta.textContent = String(err);
  v1Meta.textContent = String(err);
});

