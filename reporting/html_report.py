"""Self-contained interactive HTML report generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reporting.console import step

# ---------------------------------------------------------------------------
# Data serialization
# ---------------------------------------------------------------------------

_DISPLAY_COLUMNS = [
    "smiles",
    "p_active",
    "cluster_id",
    "cluster_size",
    "max_tanimoto_similarity",
    "similar_training_smiles",
    "ad_score",
    "p_cytotox_3T3",
    "p_cytotox_HEK",
    "retro_solved",
    "retro_score",
    "n_steps",
    "first_gen",
    "last_gen",
    "n_evals",
]

_COLUMN_LABELS = {
    "smiles": "SMILES",
    "p_active": "P(active)",
    "cluster_id": "Cluster",
    "cluster_size": "Cluster size",
    "max_tanimoto_similarity": "Max Tanimoto",
    "similar_training_smiles": "Most similar (training)",
    "ad_score": "AD score",
    "p_cytotox_3T3": "P(cytotox 3T3)",
    "p_cytotox_HEK": "P(cytotox HEK)",
    "retro_solved": "Retro solved",
    "retro_score": "Retro score",
    "n_steps": "Retro steps",
    "first_gen": "First gen",
    "last_gen": "Last gen",
    "n_evals": "# evals",
    "rank_composite": "Rank",
}

_NUMERIC_COLUMNS = {
    "p_active",
    "max_tanimoto_similarity",
    "ad_score",
    "p_cytotox_3T3",
    "p_cytotox_HEK",
    "retro_score",
    "n_steps",
    "cluster_id",
    "cluster_size",
    "first_gen",
    "last_gen",
    "n_evals",
    "rank_composite",
}


def _serialize_dataframe(df: pd.DataFrame) -> str:
    """Convert candidates DataFrame to JSON for embedding in HTML."""
    available = [c for c in _DISPLAY_COLUMNS if c in df.columns]
    subset = df[available].copy()

    for col in subset.select_dtypes(include=[np.floating]).columns:
        subset[col] = subset[col].round(4)

    subset = subset.where(subset.notna(), None)
    return subset.to_json(orient="records")


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EVA Report — {{ run_name }}</title>
<style>
:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --surface2: #232733;
  --border: #2e3345;
  --text: #e4e6ed;
  --text-dim: #8b8fa3;
  --accent: #7c6aef;
  --accent2: #5b4bc9;
  --green: #22c55e;
  --red: #ef4444;
  --yellow: #eab308;
  --blue: #3b82f6;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
header {
  padding: 12px 20px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 24px;
  flex-shrink: 0;
}
header h1 {
  font-size: 14px;
  font-weight: 600;
  color: var(--accent);
}
header .stats {
  font-size: 11px;
  color: var(--text-dim);
  display: flex;
  gap: 16px;
}
header .stats span { white-space: nowrap; }
header .stats b { color: var(--text); }
.main {
  display: flex;
  flex: 1;
  overflow: hidden;
}
/* Sidebar */
.sidebar {
  width: 260px;
  min-width: 260px;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.sidebar h2 {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-dim);
  padding: 12px 14px 8px;
}
.sidebar .filters {
  padding: 0 14px 10px;
  border-bottom: 1px solid var(--border);
}
.sidebar .filters label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-dim);
  cursor: pointer;
  padding: 3px 0;
}
.sidebar .filters input[type="checkbox"] {
  accent-color: var(--accent);
}
.sidebar .filters input[type="range"] {
  width: 100%;
  accent-color: var(--accent);
  margin-top: 4px;
}
.sidebar .filters .range-label {
  font-size: 10px;
  color: var(--accent);
  float: right;
}
.cluster-list {
  flex: 1;
  overflow-y: auto;
  padding: 4px 0;
}
.cluster-item {
  padding: 6px 14px;
  font-size: 11px;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  transition: background 0.1s;
}
.cluster-item:hover { background: var(--surface2); }
.cluster-item.active { background: var(--accent2); color: white; }
.cluster-item .count {
  color: var(--text-dim);
  font-size: 10px;
}
.cluster-item.active .count { color: rgba(255,255,255,0.7); }
.cluster-item.all-clusters {
  font-weight: 600;
  border-bottom: 1px solid var(--border);
  margin-bottom: 4px;
}
/* Content area */
.content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
/* Table */
.table-wrap {
  flex: 1;
  overflow: auto;
  position: relative;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}
thead {
  position: sticky;
  top: 0;
  z-index: 10;
}
th {
  background: var(--surface2);
  padding: 8px 10px;
  text-align: left;
  font-weight: 600;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-dim);
  cursor: pointer;
  white-space: nowrap;
  border-bottom: 2px solid var(--border);
  user-select: none;
}
th:hover { color: var(--text); }
th .sort-arrow { margin-left: 4px; font-size: 8px; }
td {
  padding: 6px 10px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
}
tr { cursor: pointer; transition: background 0.1s; }
tr:hover { background: var(--surface2); }
tr.selected { background: var(--accent2); }
td.smiles-cell {
  max-width: 200px;
  font-size: 10px;
  color: var(--text-dim);
}
/* Score colors */
.score-high { color: var(--green); }
.score-mid { color: var(--yellow); }
.score-low { color: var(--red); }
.retro-yes { color: var(--green); font-weight: 600; }
.retro-no { color: var(--text-dim); }
/* Detail panel */
.detail-panel {
  height: 0;
  overflow: hidden;
  background: var(--surface);
  border-top: 1px solid var(--border);
  transition: height 0.2s;
}
.detail-panel.open {
  height: 320px;
}
.detail-inner {
  display: grid;
  grid-template-columns: 220px 1fr 1fr;
  height: 100%;
  overflow: hidden;
}
.detail-structure {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 12px;
  border-right: 1px solid var(--border);
}
.detail-structure canvas { max-width: 200px; max-height: 180px; }
.detail-structure svg { max-width: 200px; max-height: 180px; }
.detail-structure .smiles-label {
  font-size: 9px;
  color: var(--text-dim);
  word-break: break-all;
  margin-top: 8px;
  text-align: center;
}
.detail-scores {
  padding: 14px 18px;
  overflow-y: auto;
}
.detail-scores h3 {
  font-size: 11px;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 8px;
}
.score-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}
.score-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 8px;
  background: var(--surface2);
  border-radius: 4px;
  font-size: 11px;
}
.score-item .label { color: var(--text-dim); }
.score-item .value { font-weight: 600; }
.detail-extra {
  padding: 14px 18px;
  overflow-y: auto;
  border-left: 1px solid var(--border);
}
.detail-extra h3 {
  font-size: 11px;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 8px;
}
.detail-extra .route-img {
  max-width: 100%;
  border-radius: 4px;
  margin-top: 8px;
}
.detail-extra .similar-smi {
  font-size: 10px;
  color: var(--text-dim);
  word-break: break-all;
  margin-top: 6px;
}
.detail-extra .close-btn {
  position: absolute;
  top: 8px;
  right: 12px;
  background: none;
  border: none;
  color: var(--text-dim);
  cursor: pointer;
  font-size: 16px;
}
.detail-extra .close-btn:hover { color: var(--text); }
.detail-panel { position: relative; }
/* Footer */
.footer {
  padding: 6px 20px;
  font-size: 10px;
  color: var(--text-dim);
  background: var(--surface);
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
}
/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
</style>
</head>
<body>

<header>
  <h1>EVA — {{ run_name }}</h1>
  <div class="stats" id="stats"></div>
</header>

<div class="main">
  <div class="sidebar">
    <h2>Filters</h2>
    <div class="filters">
      <label>
        <input type="checkbox" id="filter-retro-tested">
        Retro tested only
      </label>
      <label>
        <input type="checkbox" id="filter-retro-solved">
        Retro solved only
      </label>
      <label>
        P(active) &gt;
        <span class="range-label" id="range-val">0.0</span>
      </label>
      <input type="range" id="filter-p-active" min="0" max="1" step="0.05" value="0">
    </div>
    <h2>Clusters</h2>
    <div class="cluster-list" id="cluster-list"></div>
  </div>

  <div class="content">
    <div class="table-wrap">
      <table>
        <thead><tr id="table-head"></tr></thead>
        <tbody id="table-body"></tbody>
      </table>
    </div>
    <div class="detail-panel" id="detail-panel">
      <div class="detail-inner">
        <div class="detail-structure" id="detail-structure"></div>
        <div class="detail-scores" id="detail-scores"></div>
        <div class="detail-extra" id="detail-extra">
          <button class="close-btn" onclick="closeDetail()">&times;</button>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="footer">
  <span id="footer-left"></span>
  <span>Generated by EVA</span>
</div>

<script>
// ── Data ──────────────────────────────────────────────────────────────────
const RAW_DATA = {{ data_json }};
const RETRO_IMAGES = {{ retro_images_json }};
const MOL_IMAGES = {{ mol_images_json }};
const COLUMNS = {{ columns_json }};
const COL_LABELS = {{ col_labels_json }};
const NUMERIC_COLS = {{ numeric_cols_json }};

// ── State ─────────────────────────────────────────────────────────────────
let data = RAW_DATA.map((d, i) => ({...d, _idx: i}));
let filtered = [...data];
let sortCol = "rank_composite" in COL_LABELS ? "rank_composite" : "p_active";
let sortAsc = true;
let selectedCluster = null;
let selectedIdx = null;

// ── Init ──────────────────────────────────────────────────────────────────
async function init() {
  buildClusterList();
  buildTableHead();
  bindFilters();
  applyFilters();
  updateStats();
}

// ── Stats ─────────────────────────────────────────────────────────────────
function updateStats() {
  const el = document.getElementById("stats");
  const n = filtered.length;
  const clusters = new Set(filtered.map(d => d.cluster_id).filter(v => v != null));
  const retroTested = filtered.filter(d => d.retro_solved != null).length;
  const retroSolved = filtered.filter(d => d.retro_solved === true).length;
  const maxPA = filtered.reduce((m, d) => Math.max(m, d.p_active || 0), 0);
  el.innerHTML = `
    <span><b>${n.toLocaleString()}</b> candidates</span>
    <span><b>${clusters.size}</b> clusters</span>
    <span>Best P(active): <b>${maxPA.toFixed(3)}</b></span>
    <span>Retro: <b>${retroSolved}</b> solved / ${retroTested} tested</span>
  `;
}

// ── Cluster sidebar ───────────────────────────────────────────────────────
function buildClusterList() {
  const el = document.getElementById("cluster-list");
  const clusterMap = {};
  data.forEach(d => {
    const cid = d.cluster_id;
    if (cid == null) return;
    if (!clusterMap[cid]) clusterMap[cid] = 0;
    clusterMap[cid]++;
  });
  const clusters = Object.entries(clusterMap)
    .map(([id, count]) => ({id: parseInt(id), count}))
    .sort((a, b) => a.id - b.id);

  let html = `<div class="cluster-item all-clusters active" data-cluster="all"
    onclick="selectCluster(null, this)">
    All <span class="count">${data.length}</span></div>`;
  clusters.forEach(c => {
    html += `<div class="cluster-item" data-cluster="${c.id}"
      onclick="selectCluster(${c.id}, this)">
      Cluster ${c.id} <span class="count">${c.count}</span></div>`;
  });
  el.innerHTML = html;
}

function selectCluster(clusterId, el) {
  selectedCluster = clusterId;
  document.querySelectorAll(".cluster-item").forEach(e => e.classList.remove("active"));
  if (el) el.classList.add("active");
  applyFilters();
}

// ── Table head ────────────────────────────────────────────────────────────
function buildTableHead() {
  const tr = document.getElementById("table-head");
  tr.innerHTML = "";
  const displayCols = ["smiles", "p_active", "cluster_id", "p_cytotox_3T3", "p_cytotox_HEK",
    "max_tanimoto_similarity", "ad_score", "retro_solved", "retro_score"];
  displayCols.forEach(col => {
    if (!(col in COL_LABELS)) return;
    const th = document.createElement("th");
    th.dataset.col = col;
    th.innerHTML = COL_LABELS[col] + `<span class="sort-arrow"></span>`;
    th.onclick = () => toggleSort(col);
    tr.appendChild(th);
  });
}

function toggleSort(col) {
  if (sortCol === col) {
    sortAsc = !sortAsc;
  } else {
    sortCol = col;
    sortAsc = NUMERIC_COLS.has(col) ? false : true;
  }
  applyFilters();
}

function renderSortArrows() {
  document.querySelectorAll("th").forEach(th => {
    const arrow = th.querySelector(".sort-arrow");
    if (!arrow) return;
    if (th.dataset.col === sortCol) {
      arrow.textContent = sortAsc ? " ▲" : " ▼";
    } else {
      arrow.textContent = "";
    }
  });
}

// ── Filters ───────────────────────────────────────────────────────────────
function bindFilters() {
  document.getElementById("filter-retro-tested").onchange = applyFilters;
  document.getElementById("filter-retro-solved").onchange = applyFilters;
  const slider = document.getElementById("filter-p-active");
  const label = document.getElementById("range-val");
  slider.oninput = () => { label.textContent = parseFloat(slider.value).toFixed(2); };
  slider.onchange = applyFilters;
}

function applyFilters() {
  const retroTested = document.getElementById("filter-retro-tested").checked;
  const retroSolved = document.getElementById("filter-retro-solved").checked;
  const pMin = parseFloat(document.getElementById("filter-p-active").value);

  filtered = data.filter(d => {
    if (selectedCluster != null && d.cluster_id !== selectedCluster) return false;
    if (retroTested && d.retro_solved == null) return false;
    if (retroSolved && d.retro_solved !== true) return false;
    if (d.p_active != null && d.p_active < pMin) return false;
    return true;
  });

  filtered.sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (va == null) va = sortAsc ? Infinity : -Infinity;
    if (vb == null) vb = sortAsc ? Infinity : -Infinity;
    if (typeof va === "string") return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return sortAsc ? va - vb : vb - va;
  });

  renderTable();
  renderSortArrows();
  updateStats();
}

// ── Table body ────────────────────────────────────────────────────────────
function renderTable() {
  const tbody = document.getElementById("table-body");
  const rows = filtered.slice(0, 5000).map(d => {
    const cls = d._idx === selectedIdx ? " selected" : "";
    return `<tr class="${cls}" onclick="selectMolecule(${d._idx})">
      <td class="smiles-cell" title="${esc(d.smiles)}">${esc(d.smiles)}</td>
      <td class="${scoreClass(d.p_active, 0.6, 0.8)}">${fmt(d.p_active)}</td>
      <td>${d.cluster_id != null ? d.cluster_id : "—"}</td>
      <td class="${scoreClassInv(d.p_cytotox_3T3, 0.5, 0.3)}">${fmt(d.p_cytotox_3T3)}</td>
      <td class="${scoreClassInv(d.p_cytotox_HEK, 0.5, 0.3)}">${fmt(d.p_cytotox_HEK)}</td>
      <td>${fmt(d.max_tanimoto_similarity)}</td>
      <td>${fmt(d.ad_score)}</td>
      <td class="${d.retro_solved === true ? 'retro-yes' : 'retro-no'}">${retroLabel(d.retro_solved)}</td>
      <td>${fmt(d.retro_score)}</td>
    </tr>`;
  });
  tbody.innerHTML = rows.join("");
  document.getElementById("footer-left").textContent =
    `Showing ${Math.min(filtered.length, 5000)} of ${filtered.length} candidates`;
}

function esc(s) { return s ? s.replace(/"/g, "&quot;").replace(/</g, "&lt;") : ""; }
function fmt(v) { return v != null ? (typeof v === "number" ? v.toFixed(3) : v) : "—"; }
function retroLabel(v) {
  if (v === true) return "✓";
  if (v === false) return "✗";
  return "—";
}
function scoreClass(v, mid, high) {
  if (v == null) return "";
  if (v >= high) return "score-high";
  if (v >= mid) return "score-mid";
  return "score-low";
}
function scoreClassInv(v, low, mid) {
  if (v == null) return "";
  if (v <= low) return "score-high";
  if (v <= mid) return "score-mid";
  return "score-low";
}

// ── Molecule detail ───────────────────────────────────────────────────────
function selectMolecule(idx) {
  selectedIdx = idx;
  const mol = data[idx];
  const panel = document.getElementById("detail-panel");
  panel.classList.add("open");

  // Structure (pre-rendered SVG)
  const structEl = document.getElementById("detail-structure");
  if (mol.smiles && MOL_IMAGES[mol.smiles]) {
    structEl.innerHTML = MOL_IMAGES[mol.smiles] + `<div class="smiles-label">${esc(mol.smiles)}</div>`;
  } else {
    structEl.innerHTML = `<div class="smiles-label">${esc(mol.smiles)}</div>`;
  }

  // Scores
  const scoresEl = document.getElementById("detail-scores");
  const scoreRows = [
    ["P(active)", mol.p_active, scoreClass(mol.p_active, 0.6, 0.8)],
    ["P(cytotox 3T3)", mol.p_cytotox_3T3, scoreClassInv(mol.p_cytotox_3T3, 0.5, 0.3)],
    ["P(cytotox HEK)", mol.p_cytotox_HEK, scoreClassInv(mol.p_cytotox_HEK, 0.5, 0.3)],
    ["AD score", mol.ad_score, ""],
    ["Max Tanimoto (train)", mol.max_tanimoto_similarity, ""],
    ["Cluster", `${mol.cluster_id} (${mol.cluster_size} members)`, ""],
    ["First gen / Last gen", `${mol.first_gen ?? "—"} / ${mol.last_gen ?? "—"} `, ""],
    ["# evaluations", mol.n_evals, ""],
    ["Retro solved", retroLabel(mol.retro_solved), mol.retro_solved === true ? "retro-yes" : ""],
    ["Retro score", fmt(mol.retro_score), ""],
    ["Retro steps", mol.n_steps != null ? mol.n_steps : "—", ""],
  ];
  scoresEl.innerHTML = `<h3>Scores</h3><div class="score-grid">${
    scoreRows.map(([label, value, cls]) =>
      `<div class="score-item"><span class="label">${label}</span><span class="value ${cls}">${value != null && typeof value === "number" ? value.toFixed(4) : value}</span></div>`
    ).join("")
  }</div>`;

  // Extra (similar smiles, route image)
  const extraEl = document.getElementById("detail-extra");
  extraEl.innerHTML = `<button class="close-btn" onclick="closeDetail()">&times;</button>`;
  extraEl.innerHTML += `<h3>Training similarity</h3>`;
  if (mol.similar_training_smiles) {
    extraEl.innerHTML += `<div class="similar-smi">
      Most similar: <b>${esc(mol.similar_training_smiles)}</b><br>
      Tanimoto: ${mol.max_tanimoto_similarity != null ? mol.max_tanimoto_similarity.toFixed(4) : "—"}
    </div>`;
    // Render similar molecule structure
    if (MOL_IMAGES[mol.similar_training_smiles]) {
      extraEl.innerHTML += MOL_IMAGES[mol.similar_training_smiles];
    }
  } else {
    extraEl.innerHTML += `<div class="similar-smi">No similar training molecule found.</div>`;
  }

  if (RETRO_IMAGES[mol.smiles]) {
    extraEl.innerHTML += `<h3 style="margin-top:12px">Retrosynthetic Route</h3>
      <img class="route-img" src="${RETRO_IMAGES[mol.smiles]}" alt="retro route">`;
  }

  // Re-render table to highlight selection
  renderTable();
}

function closeDetail() {
  document.getElementById("detail-panel").classList.remove("open");
  selectedIdx = null;
  renderTable();
}

// ── Start ─────────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_html_report(
    candidates_df: pd.DataFrame,
    run_name: str = "run",
    output_path: Path | str = "report.html",
    retro_images: dict[str, str] | None = None,
    mol_images: dict[str, str] | None = None,
) -> Path:
    """Generate a self-contained interactive HTML report.

    Parameters
    ----------
    candidates_df : DataFrame
        Full candidates table from the pipeline.
    run_name : str
        Name displayed in the report header.
    output_path : Path or str
        Where to write the HTML file.
    retro_images : dict or None
        ``{smiles: data_url}`` mapping for pre-rendered route images.
    mol_images : dict or None
        ``{smiles: svg_string}`` mapping for pre-rendered molecule SVGs.

    Returns
    -------
    Path
        Path to the written HTML file.
    """
    from jinja2 import Template

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    step(f"Generating HTML report ({len(candidates_df):,} candidates)")

    data_json = _serialize_dataframe(candidates_df)
    retro_images_json = json.dumps(retro_images or {})
    mol_images_json = json.dumps(mol_images or {})

    available_cols = [c for c in _DISPLAY_COLUMNS if c in candidates_df.columns]
    col_labels = {c: _COLUMN_LABELS.get(c, c) for c in available_cols}
    numeric = sorted(_NUMERIC_COLUMNS & set(available_cols))

    template = Template(_HTML_TEMPLATE)
    html = template.render(
        run_name=run_name,
        data_json=data_json,
        retro_images_json=retro_images_json,
        mol_images_json=mol_images_json,
        columns_json=json.dumps(available_cols),
        col_labels_json=json.dumps(col_labels),
        numeric_cols_json=json.dumps(numeric),
    )

    output_path.write_text(html, encoding="utf-8")
    step(f"  Report saved → {output_path}")
    return output_path
