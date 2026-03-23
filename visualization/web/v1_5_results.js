function splitAblationRows(rows) {
  const groups = { A0_A4: [], feature: [], velocity: [], other: [] };
  (rows || []).forEach((r) => {
    const id = String(r.id || "");
    if (/^A[0-9]+$/.test(id)) groups.A0_A4.push(r);
    else if (id.startsWith("feat_")) groups.feature.push(r);
    else if (id.startsWith("vel_")) groups.velocity.push(r);
    else groups.other.push(r);
  });
  return groups;
}

async function bootstrap() {
  setActiveNav();
  const main = await fetchJson("/api/experiments/improve_v1_5_main");
  const mainExp = main.experiment || {};
  renderMetricGrid(document.getElementById("mainQuick"), (mainExp.quick_metrics || {}).summary || {});
  renderMetricGrid(document.getElementById("mainOfficial"), (mainExp.official_metrics || {}).summary || {});

  const ablationResp = await fetchJson("/api/ablations?family=improve-v1.5");
  const allRows = [];
  (ablationResp.rows || []).forEach((item) => {
    const rows = (item.ablation || {}).rows || [];
    rows.forEach((r) => allRows.push(r));
  });
  const grouped = splitAblationRows(allRows);

  const commonCols = [
    { key: "id", title: "id" },
    { key: "desc", title: "desc" },
    { key: "quick/mean_f1", title: "quick mean_f1", formatter: (v) => f4(v) },
    { key: "official/mean_3d", title: "official mean_3d", formatter: (v) => f4(v) },
  ];
  renderTable(document.getElementById("a0a4Table"), commonCols, grouped.A0_A4);
  renderTable(document.getElementById("featureTable"), commonCols, grouped.feature);
  renderTable(document.getElementById("velTable"), commonCols, grouped.velocity);

  const decodeResp = await fetchJson("/api/decode-tuning?exp=improve_v1_5_main");
  renderTable(
    document.getElementById("decodeTable"),
    [
      { key: "score_thresh", title: "score_thresh" },
      { key: "nms_thresh", title: "nms_thresh" },
      { key: "post_maxsize", title: "post_maxsize" },
      { key: "quick/mean_f1", title: "quick mean_f1", formatter: (v) => f4(v) },
      { key: "official/mean_3d", title: "official mean_3d", formatter: (v) => f4(v) },
    ],
    (decodeResp.decode_tuning || {}).rows || []
  );
}

bootstrap().catch((err) => {
  document.getElementById("a0a4Table").innerHTML = `<div class="empty">加载失败: ${String(err)}</div>`;
});
