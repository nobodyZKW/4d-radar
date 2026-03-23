function getExpId() {
  const pathId = parseExpIdFromPath();
  if (pathId) return pathId;
  const p = window.location.pathname;
  if (p === "/baseline-results") return "baseline_main";
  if (p === "/v1-results") return "improve_v1_main";
  return "improve_v1_5_main";
}

function saveCompareSelection(expId) {
  const raw = localStorage.getItem("compare_selection") || "";
  const ids = raw ? raw.split(",").map((x) => x.trim()).filter(Boolean) : [];
  if (!ids.includes(expId)) ids.push(expId);
  localStorage.setItem("compare_selection", ids.join(","));
}

function renderKv(container, obj) {
  const keys = Object.keys(obj || {});
  if (keys.length === 0) {
    container.innerHTML = `<div class="empty">暂无</div>`;
    return;
  }
  container.innerHTML = keys
    .map((k) => `<div class="kv"><div>${k}</div><div class="mono">${String(obj[k])}</div></div>`)
    .join("");
}

async function bootstrap() {
  setActiveNav();
  const expId = getExpId();
  const data = await fetchJson(`/api/experiments/${encodeURIComponent(expId)}`);
  const exp = data.experiment || {};
  text(document.getElementById("title"), exp.display_name || expId);
  text(
    document.getElementById("subtitle"),
    `${exp.family || ""} | status=${exp.status || ""} | updated=${exp.updated_at || "-"}`
  );

  const sampleLink = document.getElementById("sampleLink");
  sampleLink.href = `/sample-viewer?exp=${encodeURIComponent(expId)}`;

  document.getElementById("toCompareBtn").addEventListener("click", () => {
    saveCompareSelection(expId);
    window.location.href = "/compare";
  });

  renderKv(document.getElementById("basicInfo"), {
    id: exp.id,
    family: exp.family,
    status: exp.status,
    description: exp.description || "",
    tags: (exp.tags || []).join(", "),
    quick_metrics_type: (exp.quick_metrics || {}).type || "",
    official_metrics_type: (exp.official_metrics || {}).type || "",
  });

  renderMetricGrid(document.getElementById("quickMetrics"), (exp.quick_metrics || {}).summary || {});
  const quickCurves = (exp.quick_metrics || {}).curves || {};
  if (quickCurves.loss_avg && quickCurves.loss_avg.length > 1) {
    drawLineChart(
      document.getElementById("curveLoss"),
      [{ name: "loss_avg", values: quickCurves.loss_avg, color: "#2563eb" }],
      "训练损失曲线"
    );
  } else if (quickCurves.loss && quickCurves.loss.length > 1) {
    const series = [
      { name: "train_loss", values: quickCurves.loss, color: "#2563eb" },
      { name: "val_loss", values: quickCurves.val_loss || [], color: "#dc2626" },
    ].filter((x) => (x.values || []).length > 1);
    drawLineChart(document.getElementById("curveLoss"), series, "Loss 曲线");
  } else {
    document.getElementById("curveLoss").innerHTML = `<div class="empty">暂无可视化曲线</div>`;
  }

  if (quickCurves.lr && quickCurves.lr.length > 1) {
    drawLineChart(
      document.getElementById("curveLR"),
      [{ name: "lr", values: quickCurves.lr, color: "#16a34a" }],
      "Learning Rate 曲线"
    );
  } else {
    document.getElementById("curveLR").innerHTML = `<div class="empty">暂无 LR 曲线</div>`;
  }

  renderMetricGrid(document.getElementById("officialSummary"), (exp.official_metrics || {}).summary || {});
  const officialRows = (exp.official_metrics || {}).rows || [];
  renderTable(
    document.getElementById("officialRows"),
    [
      { key: "class", title: "Class" },
      { key: "protocol", title: "Protocol" },
      { key: "metric", title: "Metric" },
      { key: "value", title: "Value", formatter: (v) => f4(v) },
    ],
    officialRows
  );

  renderKv(document.getElementById("configSummary"), exp.config_summary || {});

  const abl = exp.ablation || {};
  const ablRows = (exp.ablation_focus && exp.ablation_focus.rows) || abl.rows || [];
  renderTable(
    document.getElementById("ablationContent"),
    [
      { key: "id", title: "id" },
      { key: "desc", title: "desc" },
      { key: "quick/mean_f1", title: "quick mean_f1", formatter: (v) => f4(v) },
      { key: "official/mean_3d", title: "official mean_3d", formatter: (v) => f4(v) },
    ],
    ablRows
  );

  const decode = exp.decode_tuning || {};
  renderTable(
    document.getElementById("decodeContent"),
    [
      { key: "score_thresh", title: "score_thresh" },
      { key: "nms_thresh", title: "nms_thresh" },
      { key: "post_maxsize", title: "post_maxsize" },
      { key: "quick/mean_f1", title: "quick mean_f1", formatter: (v) => f4(v) },
      { key: "official/mean_3d", title: "official mean_3d", formatter: (v) => f4(v) },
    ],
    decode.rows || []
  );

  renderKv(document.getElementById("artifacts"), exp.artifacts || {});
}

bootstrap().catch((err) => {
  text(document.getElementById("subtitle"), `加载失败: ${String(err)}`);
});
