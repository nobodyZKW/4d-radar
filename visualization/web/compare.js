let allExperiments = [];

function getSelectedIds() {
  return Array.from(document.querySelectorAll("input[name='exp_select']:checked")).map((x) => x.value);
}

function setSelected(ids) {
  const set = new Set(ids);
  document.querySelectorAll("input[name='exp_select']").forEach((el) => {
    el.checked = set.has(el.value);
  });
}

function parseIdsFromUrl() {
  const q = new URLSearchParams(window.location.search);
  const ids = q.get("ids");
  if (ids) return ids.split(",").map((x) => x.trim()).filter(Boolean);
  const local = localStorage.getItem("compare_selection") || "";
  return local ? local.split(",").map((x) => x.trim()).filter(Boolean) : [];
}

function renderSelector(rows) {
  const container = document.getElementById("expSelector");
  container.innerHTML = "";
  rows.forEach((row) => {
    const card = document.createElement("label");
    card.className = "card";
    card.innerHTML = `
      <div class="row">
        <input type="checkbox" name="exp_select" value="${row.id}" />
        <strong>${row.display_name}</strong>
      </div>
      <div class="small">${row.family} | ${row.status}</div>
      <div class="small">quick mean_f1=${f4((row.quick_summary || {}).mean_f1)}</div>
      <div class="small">official mean_3d=${f4((row.official_summary || {}).mean_3d_primary)}</div>
    `;
    container.appendChild(card);
  });
}

function renderCompare(data) {
  renderTable(
    document.getElementById("quickTable"),
    [
      { key: "display_name", title: "实验" },
      { key: "family", title: "family" },
      { key: "mean_f1", title: "mean_f1", formatter: (v) => f4(v) },
      { key: "Car_f1", title: "Car_f1", formatter: (v) => f4(v) },
      { key: "Pedestrian_f1", title: "Pedestrian_f1", formatter: (v) => f4(v) },
      { key: "Cyclist_f1", title: "Cyclist_f1", formatter: (v) => f4(v) },
      { key: "loss", title: "loss", formatter: (v) => f4(v) },
      { key: "val_loss", title: "val_loss", formatter: (v) => f4(v) },
    ],
    data.quick_table || []
  );

  renderTable(
    document.getElementById("officialTable"),
    [
      { key: "display_name", title: "实验" },
      { key: "official_available", title: "official", formatter: (v) => (v ? "yes" : "missing") },
      { key: "mean_3d_primary", title: "mean_3d", formatter: (v) => f4(v) },
      { key: "Car_3d_primary", title: "Car_3d", formatter: (v) => f4(v) },
      { key: "Pedestrian_3d_primary", title: "Ped_3d", formatter: (v) => f4(v) },
      { key: "Cyclist_3d_primary", title: "Cyc_3d", formatter: (v) => f4(v) },
    ],
    data.official_table || []
  );

  renderTable(
    document.getElementById("configDiff"),
    [
      { key: "key", title: "字段" },
      {
        key: "values",
        title: "值",
        formatter: (v) => Object.entries(v || {}).map(([k, val]) => `${k}: ${val}`).join("<br/>"),
      },
      { key: "different", title: "是否不同", formatter: (v) => (v ? "yes" : "no") },
    ],
    (data.config_diff || {}).rows || []
  );

  const curves = data.curves || {};
  const palette = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#d97706", "#0891b2"];
  const series = [];
  Object.entries(curves).forEach(([id, c], idx) => {
    const v = c.loss_avg || c.loss || [];
    if (Array.isArray(v) && v.length > 1) {
      series.push({ name: id, values: v, color: palette[idx % palette.length] });
    }
  });
  drawLineChart(document.getElementById("curveChart"), series, "Loss 曲线对比");
}

async function runCompare() {
  const ids = getSelectedIds();
  localStorage.setItem("compare_selection", ids.join(","));
  const viewerLink = document.getElementById("viewerLink");
  viewerLink.href = `/sample-viewer?exp=${encodeURIComponent(ids.join(","))}`;
  const q = new URLSearchParams();
  q.set("ids", ids.join(","));
  window.history.replaceState({}, "", `/compare?${q.toString()}`);
  if (ids.length === 0) {
    document.getElementById("quickTable").innerHTML = `<div class="empty">请先选择实验。</div>`;
    document.getElementById("officialTable").innerHTML = `<div class="empty">请先选择实验。</div>`;
    document.getElementById("configDiff").innerHTML = `<div class="empty">请先选择实验。</div>`;
    return;
  }
  const data = await fetchJson(`/api/compare?ids=${encodeURIComponent(ids.join(","))}`);
  renderCompare(data);
}

async function bootstrap() {
  setActiveNav();
  const expData = await fetchJson("/api/experiments?sort=sort_order");
  allExperiments = expData.experiments || [];
  renderSelector(allExperiments);
  setSelected(parseIdsFromUrl());

  document.getElementById("applyPresetBtn").addEventListener("click", () => {
    const p = document.getElementById("preset").value;
    if (!p) return;
    setSelected(p.split(",").map((x) => x.trim()));
  });

  document.getElementById("runCompareBtn").addEventListener("click", () => {
    runCompare().catch((err) => {
      document.getElementById("quickTable").innerHTML = `<div class="empty">对比失败: ${String(err)}</div>`;
    });
  });

  if (parseIdsFromUrl().length > 0) {
    await runCompare();
  }
}

bootstrap().catch((err) => {
  document.getElementById("quickTable").innerHTML = `<div class="empty">加载失败: ${String(err)}</div>`;
});
