function readQueryFilters() {
  const q = new URLSearchParams(window.location.search);
  return {
    family: q.get("family") || "",
    status: q.get("status") || "",
    tags: q.get("tags") || "",
    sort: q.get("sort") || "sort_order",
  };
}

function syncQueryFromInputs() {
  const family = document.getElementById("family").value.trim();
  const status = document.getElementById("status").value.trim();
  const tags = document.getElementById("tags").value.trim();
  const sort = document.getElementById("sort").value.trim();
  const q = new URLSearchParams();
  if (family) q.set("family", family);
  if (status) q.set("status", status);
  if (tags) q.set("tags", tags);
  if (sort) q.set("sort", sort);
  const next = `/experiments${q.toString() ? "?" + q.toString() : ""}`;
  window.history.replaceState({}, "", next);
  return { family, status, tags, sort };
}

function renderCards(rows) {
  const container = document.getElementById("resultCards");
  container.innerHTML = "";
  if (!rows || rows.length === 0) {
    container.innerHTML = `<div class="empty">没有匹配实验。</div>`;
    return;
  }
  rows.forEach((row) => {
    const tags = (row.tags || []).map((t) => `<span class="pill">${t}</span>`).join("");
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `
      <h3>${row.display_name}</h3>
      <p>${row.description || ""}</p>
      <div class="row">
        <span class="pill">${row.family}</span>
        <span class="pill">${row.status}</span>
        ${tags}
      </div>
      <div class="metric-grid" style="margin-top:8px">
        <div class="metric"><div class="k">quick mean_f1</div><div class="v">${f4((row.quick_summary || {}).mean_f1)}</div></div>
        <div class="metric"><div class="k">official mean_3d</div><div class="v">${f4((row.official_summary || {}).mean_3d_primary)}</div></div>
      </div>
      <div class="small" style="margin-top:8px">updated: ${row.updated_at || "-"}</div>
      <div class="row" style="margin-top:8px">
        <a href="/experiments/${encodeURIComponent(row.id)}">详情</a>
        <a href="/compare?ids=${encodeURIComponent(row.id)}">加入对比</a>
      </div>
    `;
    container.appendChild(card);
  });
}

async function loadFamilies(selectedFamily) {
  const data = await fetchJson("/api/experiments");
  const families = data.families || [];
  const select = document.getElementById("family");
  select.innerHTML = `<option value="">all</option>`;
  families.forEach((f) => {
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = f;
    if (f === selectedFamily) opt.selected = true;
    select.appendChild(opt);
  });
}

async function loadData(filters) {
  const q = new URLSearchParams();
  if (filters.family) q.set("family", filters.family);
  if (filters.status) q.set("status", filters.status);
  if (filters.tags) q.set("tags", filters.tags);
  if (filters.sort) q.set("sort", filters.sort);
  const data = await fetchJson(`/api/experiments?${q.toString()}`);
  renderCards(data.experiments || []);
}

async function bootstrap() {
  setActiveNav();
  const qf = readQueryFilters();
  await loadFamilies(qf.family);
  document.getElementById("status").value = qf.status;
  document.getElementById("tags").value = qf.tags;
  document.getElementById("sort").value = qf.sort;
  await loadData(qf);

  document.getElementById("applyBtn").addEventListener("click", async () => {
    const filters = syncQueryFromInputs();
    await loadData(filters);
  });
}

bootstrap().catch((err) => {
  const container = document.getElementById("resultCards");
  if (container) container.innerHTML = `<div class="empty">加载失败: ${String(err)}</div>`;
});
