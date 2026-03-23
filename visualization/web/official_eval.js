async function run() {
  const expId = document.getElementById("expId").value.trim();
  const data = await fetchJson(`/api/experiments/${encodeURIComponent(expId)}`);
  const official = (data.experiment || {}).official_metrics || {};
  renderMetricGrid(document.getElementById("summary"), official.summary || {});
  renderTable(
    document.getElementById("table"),
    [
      { key: "class", title: "class" },
      { key: "protocol", title: "protocol" },
      { key: "metric", title: "metric" },
      { key: "value", title: "value", formatter: (v) => f4(v) },
    ],
    official.rows || []
  );
}

setActiveNav();
document.getElementById("runBtn").addEventListener("click", () => {
  run().catch((err) => {
    document.getElementById("table").innerHTML = `<div class="empty">加载失败: ${String(err)}</div>`;
  });
});
run().catch((err) => {
  document.getElementById("table").innerHTML = `<div class="empty">加载失败: ${String(err)}</div>`;
});
