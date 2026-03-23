async function bootstrap() {
  setActiveNav();
  const app = await fetchJson("/api/app-info");
  const family = await fetchJson("/api/family-overview");
  const expData = await fetchJson("/api/experiments?sort=time");

  text(
    document.getElementById("appInfo"),
    `${app.app.title} | project=${app.app.project_root} | data=${app.app.data_root}`
  );

  const cards = document.getElementById("familyCards");
  cards.innerHTML = "";
  (family.families || []).forEach((item) => {
    const main = item.main || {};
    const quick = main.quick_summary || {};
    const official = main.official_summary || {};
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `
      <h3>${item.family}</h3>
      <p>${main.description || "暂无描述"}</p>
      <div class="row">
        <span class="pill">实验数: ${item.count}</span>
        <span class="pill">主实验: ${main.display_name || "-"}</span>
      </div>
      <div class="metric-grid" style="margin-top:8px">
        <div class="metric"><div class="k">quick mean_f1</div><div class="v">${f4(quick.mean_f1)}</div></div>
        <div class="metric"><div class="k">official mean_3d</div><div class="v">${f4(official.mean_3d_primary)}</div></div>
      </div>
      <div class="row" style="margin-top:8px">
        <a href="/experiments?family=${encodeURIComponent(item.family)}">查看该家族实验</a>
        ${main.id ? `<a href="/experiments/${encodeURIComponent(main.id)}">主实验详情</a>` : ""}
      </div>
    `;
    cards.appendChild(card);
  });

  const recentList = document.getElementById("recentList");
  const recentRows = (expData.experiments || []).slice(0, 10);
  renderTable(
    recentList,
    [
      { key: "display_name", title: "实验" },
      { key: "family", title: "家族" },
      { key: "status", title: "状态" },
      { key: "updated_at", title: "更新时间" },
      { key: "quick_summary", title: "quick mean_f1", formatter: (v) => f4((v || {}).mean_f1) },
      { key: "official_summary", title: "official mean_3d", formatter: (v) => f4((v || {}).mean_3d_primary) },
      {
        key: "id",
        title: "详情",
        formatter: (v) => `<a href="/experiments/${encodeURIComponent(v)}">打开</a>`,
      },
    ],
    recentRows
  );
}

bootstrap().catch((err) => {
  const recentList = document.getElementById("recentList");
  if (recentList) recentList.innerHTML = `<div class="empty">加载失败: ${String(err)}</div>`;
});
