async function fetchJson(url) {
  const resp = await fetch(url);
  let payload = {};
  try {
    payload = await resp.json();
  } catch (err) {
    payload = { ok: false, message: String(err) };
  }
  if (!resp.ok) {
    const msg = payload && payload.message ? payload.message : `${resp.status}`;
    throw new Error(msg);
  }
  return payload;
}

function f4(v, digits = 4) {
  if (v === null || v === undefined || v === "") return "-";
  const n = Number(v);
  if (Number.isNaN(n)) return String(v);
  return n.toFixed(digits);
}

function text(el, value) {
  if (!el) return;
  el.textContent = value ?? "";
}

function parseExpIdFromPath() {
  const p = window.location.pathname;
  const seg = p.split("/").filter(Boolean);
  if (seg.length === 2 && seg[0] === "experiments") {
    return decodeURIComponent(seg[1]);
  }
  const q = new URLSearchParams(window.location.search);
  return q.get("id") || "";
}

function setActiveNav() {
  const p = window.location.pathname;
  document.querySelectorAll(".nav a").forEach((a) => {
    const href = a.getAttribute("href") || "";
    if (href !== "/" && p.startsWith(href)) {
      a.classList.add("active");
    } else if (href === "/" && p === "/") {
      a.classList.add("active");
    }
  });
}

function renderMetricGrid(container, metrics) {
  if (!container) return;
  container.innerHTML = "";
  const keys = Object.keys(metrics || {});
  if (keys.length === 0) {
    container.innerHTML = `<div class="empty">暂无指标</div>`;
    return;
  }
  keys.forEach((k) => {
    const card = document.createElement("div");
    card.className = "metric";
    card.innerHTML = `<div class="k">${k}</div><div class="v">${f4(metrics[k])}</div>`;
    container.appendChild(card);
  });
}

function renderTable(container, columns, rows) {
  if (!container) return;
  if (!rows || rows.length === 0) {
    container.innerHTML = `<div class="empty">暂无数据</div>`;
    return;
  }
  let html = "<div class='table-wrap'><table><thead><tr>";
  html += columns.map((c) => `<th>${c.title}</th>`).join("");
  html += "</tr></thead><tbody>";
  rows.forEach((r) => {
    html += "<tr>";
    columns.forEach((c) => {
      const raw = c.key in r ? r[c.key] : "";
      const v = c.formatter ? c.formatter(raw, r) : raw;
      html += `<td>${v ?? "-"}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table></div>";
  container.innerHTML = html;
}

function drawLineChart(container, seriesList, title = "") {
  if (!container) return;
  const valid = (seriesList || []).filter((s) => s && Array.isArray(s.values) && s.values.length > 1);
  if (valid.length === 0) {
    container.innerHTML = `<div class="empty">${title || "暂无曲线"}</div>`;
    return;
  }

  const width = 900;
  const height = 260;
  const pad = 28;
  const allValues = valid.flatMap((s) => s.values.map((x) => Number(x)).filter((x) => Number.isFinite(x)));
  const minY = Math.min(...allValues);
  const maxY = Math.max(...allValues);
  const span = Math.max(1e-6, maxY - minY);

  const linePaths = valid
    .map((s) => {
      const n = s.values.length;
      const points = s.values
        .map((v, i) => {
          const x = pad + ((width - 2 * pad) * i) / Math.max(1, n - 1);
          const y = height - pad - ((height - 2 * pad) * (Number(v) - minY)) / span;
          return `${x.toFixed(2)},${y.toFixed(2)}`;
        })
        .join(" ");
      return `<polyline fill="none" stroke="${s.color || "#2563eb"}" stroke-width="2" points="${points}" />`;
    })
    .join("");

  const legend = valid
    .map((s) => `<span style="border-color:${s.color || "#2563eb"}">${s.name}</span>`)
    .join("");

  container.innerHTML = `
    <div class="small">${title}</div>
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d1d5db"></line>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#d1d5db"></line>
      ${linePaths}
    </svg>
    <div class="legend">${legend}</div>
  `;
}
