async function run() {
  const family = document.getElementById("family").value.trim();
  const data = await fetchJson(`/api/ablations?family=${encodeURIComponent(family)}`);
  const rows = [];
  (data.rows || []).forEach((r) => {
    ((r.ablation || {}).rows || []).forEach((x) => rows.push(x));
  });
  renderTable(
    document.getElementById("table"),
    [
      { key: "id", title: "id" },
      { key: "desc", title: "desc" },
      { key: "quick/mean_f1", title: "quick mean_f1", formatter: (v) => f4(v) },
      { key: "official/mean_3d", title: "official mean_3d", formatter: (v) => f4(v) },
    ],
    rows
  );
}

setActiveNav();
document.getElementById("runBtn").addEventListener("click", () => {
  run().catch((err) => {
    document.getElementById("table").innerHTML = `<div class="empty">查询失败: ${String(err)}</div>`;
  });
});
run().catch((err) => {
  document.getElementById("table").innerHTML = `<div class="empty">加载失败: ${String(err)}</div>`;
});
