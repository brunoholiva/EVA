"""Simple Benchmark Plotting Script.

To customize what gets plotted:
  1. Edit METHODS_TO_PLOT below with whichever methods you want to show.
  2. Edit COLORS to assign the hex colors.
  3. Run:
         uv run python plot_benchmarks.py
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# =============================================================================
# 1. EDIT THIS LIST: Pick whichever methods you want to plot
# =============================================================================
METHODS_TO_PLOT = [
    "cma_me_imp_mu_thresh", 
    "cma_me_opt",            
    "cma_me_imp_mu",                
    "cma_mae_thresh",       
    "naive",                
]

# =============================================================================
# 1B. EDIT THIS LIST: Pick which methods to plot in Figure 6 (ChemBed t-SNE)
# =============================================================================
TSNE_METHODS = [
    "cma_me_imp_mu_thresh",
    "cma_me_opt",
]

# Set to True if you want other arms shown as faint grey lines/dots in the background.
# Set to False to completely hide all unselected methods (cleanest for reporting).
SHOW_BACKGROUND_ARMS = False

# Where to save the output PNG figures
OUTPUT_DIR = Path("results/benchmarks/analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "cma_me_imp_mu_thresh": "#6666FF", 
    "cma_me_opt":           "#ff9933",  
    "cma_me_imp_mu":        "#e15759", 
    "cma_mae_thresh":       "#76B7B2",  
    "naive":                "#EDC949",  
    "me_map_elites_thresh": "#292966",  
    "cma_me_imp_thresh":    "#9999FF",  
    "cma_mae":              "#666666",
}
DEFAULT_BACKGROUND_GREY = "#B3B3B3"

# =============================================================================
# 3. FORMATTED NAMES (LaTeX labels for titles, axes, and legends)
# =============================================================================
NAMES = {
    "cma_me_imp_mu_thresh": "CMA-ME imp-$\\mu$ with a threshold",
    "naive":                "Naive prior $\\mathcal{N}(0, I)$",
    "cma_es":               "CMA-ES (1-cell obj)",
    "cma_me_opt":           "CMA without diversity",
    "cma_me_opt_thresh":    "CMA-ME opt with a threshold",
    "cma_mae":              "CMA-MAE (unthresholded)",
    "cma_mae_thresh":       "CMA-MAE with a threshold",
    "me_map_elites":        "ME-MAP-Elites",
    "me_map_elites_thresh": "ME-MAP-Elites with a threshold",
    "cma_me_imp":           "CMA-ME imp-filter",
    "cma_me_imp_thresh":    "CMA-ME imp-filter with a threshold",
    "cma_me_imp_mu":        "CMA-ME imp-$\\mu$",
    "cma_me_basic":         "CMA-ME basic",
    "cma_me_basic_thresh":  "CMA-ME basic with a threshold",
    "cma_me_mixed":         "CMA-ME mixed",
    "cma_me_mixed_thresh":  "CMA-ME mixed with a threshold",
    "cma_me_rd":            "CMA-ME rank-diff",
    "cma_me_rd_thresh":     "CMA-ME rank-diff with a threshold",
    "cma_me_rd_mu":         "CMA-ME rank-diff-$\\mu$",
    "cma_me_rd_mu_thresh":  "CMA-ME rank-diff-$\\mu$ with a threshold",
    "cma_me_obj_bandit":    "CMA-ME obj bandit",
    "map_elites":           "MAP-Elites",
}

# =============================================================================
# Styling & Matplotlib Configuration
# =============================================================================
def get_color(method: str) -> str:
    return COLORS.get(method, DEFAULT_BACKGROUND_GREY)

def get_name(method: str) -> str:
    return NAMES.get(method, method)

def get_zorder(method: str) -> int:
    if method == "cma_me_imp_mu_thresh":
        return 10
    if method == "naive":
        return 8
    if method in METHODS_TO_PLOT:
        return 6
    return 2

def style_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

def add_stat_box(ax, text: str, xy=(0.97, 0.93), ha="right", va="top", fontsize=13):
    ax.text(
        xy[0], xy[1], text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=fontsize,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#CCCCCC", alpha=0.94, linewidth=1.1),
        zorder=15,
    )

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": True,
})

# =============================================================================
# Load Cached Data Bundle
# =============================================================================
CACHE_FILE = Path("notebooks/cache/06_benchmark_analysis/benchmark_bundle_v1.pkl")
if not CACHE_FILE.exists():
    raise FileNotFoundError(f"Cache bundle {CACHE_FILE} not found.")

print(f"Loading cached benchmark data from {CACHE_FILE}...")
with open(CACHE_FILE, "rb") as f:
    bundle = pickle.load(f)

run_metrics_df = bundle["run_metrics_df"]
method_table = bundle["method_table"]
discovery_curves = bundle["discovery_curves"]
scaffold_eval_curves = bundle["scaffold_eval_curves"]
rarefaction_curves = bundle["rarefaction_curves"]
rolling_p_curves = bundle["rolling_p_curves"]
coverage_curves = bundle["coverage_curves"]
mcs_pairwise_by_method = bundle["mcs_pairwise_by_method"]
mcs_seed_medians = bundle["mcs_seed_medians"]
top_div_by_method = bundle["top_div_by_method"]
top_raw_by_method = bundle["top_raw_by_method"]

EVAL_GRID = np.linspace(0, 100_000, 101)
all_methods = list(method_table.index)
selected_methods = [m for m in METHODS_TO_PLOT if m in all_methods]


if SHOW_BACKGROUND_ARMS:
    plot_order = [m for m in all_methods if m not in selected_methods] + list(reversed(selected_methods))
else:
    plot_order = list(reversed(selected_methods))

print(f"Plotting {len(selected_methods)} methods: {selected_methods}")
print(f"Target folder: {OUTPUT_DIR.resolve()}\n")

# =============================================================================
# Figure 1: Discovery Curves (P > 0.6, 0.7, 0.8)
# =============================================================================
print("Generating Figure 1: Discovery Curves...")
fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), sharex=True)
panel_specs = [
    (0.6, "(a) $P(\\mathrm{active}) > 0.6$"),
    (0.7, "(b) $P(\\mathrm{active}) > 0.7$"),
    (0.8, "(c) $P(\\mathrm{active}) > 0.8$"),
]

for ax, (thresh, title) in zip(axes, panel_specs):
    style_axes(ax)
    for m in plot_order:
        arrs = [discovery_curves[thresh][(m, s)] for s in range(1, 11) if (m, s) in discovery_curves[thresh]]
        if not arrs:
            continue
        mat = np.vstack(arrs)
        mean, std = mat.mean(axis=0), mat.std(axis=0)
        is_focal = m in selected_methods
        col = get_color(m)
        z = get_zorder(m)
        lw = 2.8 if m == "cma_me_imp_mu_thresh" else (2.4 if is_focal else 1.0)
        ax.plot(
            EVAL_GRID / 1e3, mean,
            color=col, linewidth=lw, alpha=1.0 if is_focal else 0.28,
            label=get_name(m) if is_focal else None,
            zorder=z,
        )
        if is_focal:
            ax.fill_between(
                EVAL_GRID / 1e3, np.maximum(0, mean - std), mean + std,
                color=col, alpha=0.18, edgecolor="none", linewidth=0, zorder=z - 1,
            )
    ax.set_title(title, pad=10)
    ax.set_xlabel("Oracle evaluations ($\\times 10^3$)")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)

axes[0].set_ylabel("Cumulative unique hits")

if "cma_me_imp_mu_thresh" in selected_methods and "naive" in selected_methods:
    eva_hits = method_table.loc["cma_me_imp_mu_thresh", "hits_06_mean"]
    naive_hits = max(method_table.loc["naive", "hits_06_mean"], 1e-6)
    add_stat_box(
        axes[0],
        f"EVA = {eva_hits / naive_hits:,.0f}$\\times$ naive",
        xy=(0.04, 0.94), ha="left", va="top",
    )

handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ordered_labels = [get_name(m) for m in selected_methods if get_name(m) in by_label]
ordered_handles = [by_label[l] for l in ordered_labels]
if SHOW_BACKGROUND_ARMS:
    ordered_handles.append(matplotlib.lines.Line2D([0], [0], color=DEFAULT_BACKGROUND_GREY, lw=1.5, alpha=0.5))
    ordered_labels.append("Other benchmark arms")

fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
fig.legend(
    ordered_handles, ordered_labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.02),
    ncol=min(4, len(ordered_labels)), frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fancybox=True,
)
p1 = OUTPUT_DIR / "fig1_discovery_curves.png"
fig.savefig(p1, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  -> Saved {p1}")

# =============================================================================
# Figure 2: Final Hits Bar Plot (100k evals)
# =============================================================================
print("Generating Figure 2: Final Hits Bar Plot...")
sub_bar = method_table.loc[selected_methods].sort_values("hits_mean", ascending=True)
fig_h = max(4.5, len(sub_bar) * 0.72)
fig, ax = plt.subplots(figsize=(11.0, fig_h))
style_axes(ax)

y_pos = np.arange(len(sub_bar))
means = sub_bar["hits_mean"].to_numpy()
stds = sub_bar["hits_std"].fillna(0.0).to_numpy()
colors = [get_color(m) for m in sub_bar.index]

ax.barh(y_pos, means, height=0.90, color=colors, edgecolor="none", alpha=0.95, zorder=3)
ax.errorbar(means, y_pos, xerr=stds, fmt="none", ecolor="#222222", elinewidth=1.2, capsize=0, zorder=5)

max_x = 0.0
for idx, m in enumerate(sub_bar.index):
    seed_vals = run_metrics_df.loc[run_metrics_df["method"] == m, "hits"].to_numpy()
    max_x = max(max_x, float(seed_vals.max()) if seed_vals.size else 0.0, float(means[idx] + stds[idx]))
    jitter = np.linspace(-0.14, 0.14, len(seed_vals)) if len(seed_vals) > 1 else np.array([0.0])
    ax.scatter(
        seed_vals, np.full_like(seed_vals, y_pos[idx], dtype=float) + jitter,
        s=40, facecolor="white", edgecolor="#222222", linewidth=1.1, alpha=0.9, zorder=6,
    )

offset = max_x * 0.018
for idx, (m, val, sd) in enumerate(zip(sub_bar.index, means, stds)):
    seed_max = run_metrics_df.loc[run_metrics_df["method"] == m, "hits"].max()
    x_txt = max(val + sd, seed_max) + offset
    ax.text(x_txt, y_pos[idx], f"{val:,.0f}", va="center", ha="left", fontsize=14, fontweight="bold", color="#222222", zorder=7)

ax.set_yticks(y_pos)
ax.set_yticklabels([get_name(m) for m in sub_bar.index])
ax.set_xlabel("Unique hits at 100k evaluations ($P(\\mathrm{active}) > 0.6$)")
ax.set_title("Unique Hits at 100k Oracle Evaluations", pad=12)
ax.set_xlim(0, max_x * 1.18)
ax.set_ylim(-0.65, len(sub_bar) - 0.35)

if "cma_me_imp_mu_thresh" in selected_methods and "naive" in selected_methods:
    eva_hits = method_table.loc["cma_me_imp_mu_thresh", "hits_mean"]
    naive_hits = max(method_table.loc["naive", "hits_mean"], 1e-6)
    add_stat_box(
        ax,
        f"EVA = {eva_hits / naive_hits:,.0f}$\\times$ naive ({eva_hits:,.0f} vs {naive_hits:.1f})",
        xy=(0.97, 0.10), ha="right", va="bottom",
    )

fig.tight_layout()
p2 = OUTPUT_DIR / "fig2_final_hits_bar.png"
fig.savefig(p2, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  -> Saved {p2}")

# =============================================================================
# Figure 3: Quality vs. Diversity (Cleaned, no arrow, naive filtered from ratio)
# =============================================================================
print("Generating Figure 3: Quality vs. Diversity...")
fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.8))

ax = axes[0]
style_axes(ax)
display_set = all_methods if SHOW_BACKGROUND_ARMS else selected_methods

for m in display_set:
    sub_runs = run_metrics_df[run_metrics_df["method"] == m]
    x_vals = sub_runs["hits"].to_numpy()
    y_vals = sub_runs["generic_scaffolds"].to_numpy()
    xm, xs = x_vals.mean(), x_vals.std()
    ym, ys = y_vals.mean(), y_vals.std()
    is_focal = m in selected_methods
    col = get_color(m)
    z = get_zorder(m)

    ax.scatter(
        x_vals, y_vals,
        s=36 if is_focal else 20,
        color=col, alpha=0.42 if is_focal else 0.22,
        edgecolors="none", zorder=z - 1,
    )
    ax.errorbar(
        xm, ym, xerr=xs, yerr=ys,
        fmt="o", markersize=9.5 if m == "cma_me_imp_mu_thresh" else (8.0 if is_focal else 5.0),
        color=col, ecolor=col if is_focal else "#AAAAAA",
        elinewidth=1.8 if is_focal else 0.9,
        capsize=0, alpha=0.92 if is_focal else 0.45,
        markeredgecolor="#222222" if is_focal else "none",
        markeredgewidth=1.0 if is_focal else 0.0,
        label=get_name(m) if is_focal else None,
        zorder=z,
    )

ax.set_xlabel("Unique hits ($P(\\mathrm{active}) > 0.6$)")
ax.set_ylabel("Unique generic Murcko scaffolds")
ax.set_title("(a) Hit Volume vs. Scaffold Count", pad=10)
ax.set_xlim(left=-150)
ax.set_ylim(bottom=-20)
add_stat_box(
    ax,
    "Intervals: mean $\\pm$ 1 std\nDots: individual replicates",
    xy=(0.04, 0.94), ha="left", va="top", fontsize=12,
)

ax2 = axes[1]
style_axes(ax2)
# Exclude naive from the ratio plot to eliminate small-denominator (N=2) artifacts
viable_ratio_methods = [m for m in display_set if method_table.loc[m, "hits_mean"] >= 50]

for m in viable_ratio_methods:
    sub_runs = run_metrics_df[run_metrics_df["method"] == m]
    x_vals = sub_runs["hits"].to_numpy()
    y_vals = sub_runs["scaffolds_per_100_hits"].to_numpy()
    xm, xs = x_vals.mean(), x_vals.std()
    ym, ys = y_vals.mean(), y_vals.std()
    is_focal = m in selected_methods
    col = get_color(m)
    z = get_zorder(m)

    ax2.scatter(
        x_vals, y_vals,
        s=36 if is_focal else 20,
        color=col, alpha=0.42 if is_focal else 0.22,
        edgecolors="none", zorder=z - 1,
    )
    ax2.errorbar(
        xm, ym, xerr=xs, yerr=ys,
        fmt="o", markersize=9.5 if m == "cma_me_imp_mu_thresh" else (8.0 if is_focal else 5.0),
        color=col, ecolor=col if is_focal else "#AAAAAA",
        elinewidth=1.8 if is_focal else 0.9,
        capsize=0, alpha=0.92 if is_focal else 0.45,
        markeredgecolor="#222222" if is_focal else "none",
        markeredgewidth=1.0 if is_focal else 0.0,
        zorder=z,
    )

ax2.set_xlabel("Unique hits ($P(\\mathrm{active}) > 0.6$)")
ax2.set_ylabel("Generic scaffolds per 100 hits")
ax2.set_title("(b) Scaffold Discovery Rate", pad=10)
ax2.set_xlim(left=-150)
ax2.set_ylim(bottom=0, top=26)
add_stat_box(
    ax2,
    "Filtered to methods with $\\geq 50$ hits\n(avoids small-denominator artifacts)",
    xy=(0.96, 0.94), ha="right", va="top", fontsize=12,
)

handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ordered_labels = [get_name(m) for m in selected_methods if get_name(m) in by_label]
ordered_handles = [by_label[l] for l in ordered_labels]
if SHOW_BACKGROUND_ARMS:
    ordered_handles.append(matplotlib.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=DEFAULT_BACKGROUND_GREY, markersize=7, alpha=0.7))
    ordered_labels.append("Other benchmark arms")

fig.tight_layout()
fig.subplots_adjust(bottom=0.23)
fig.legend(
    ordered_handles, ordered_labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.02),
    ncol=min(4, len(ordered_labels)), frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fancybox=True,
)
p3 = OUTPUT_DIR / "fig3_quality_vs_diversity.png"
fig.savefig(p3, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  -> Saved {p3}")

# =============================================================================
# Figure 4: Scaffold Discovery & Rarefaction
# =============================================================================
print("Generating Figure 4: Scaffold Discovery & Rarefaction...")
fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2))
ax = axes[0]
style_axes(ax)

for m in plot_order:
    arrs = [scaffold_eval_curves[(m, s)] for s in range(1, 11) if (m, s) in scaffold_eval_curves]
    if not arrs:
        continue
    mat = np.vstack(arrs)
    mean, std = mat.mean(axis=0), mat.std(axis=0)
    is_focal = m in selected_methods
    col = get_color(m)
    z = get_zorder(m)
    lw = 2.8 if m == "cma_me_imp_mu_thresh" else (2.4 if is_focal else 1.0)
    ax.plot(
        EVAL_GRID / 1e3, mean,
        color=col, linewidth=lw, alpha=1.0 if is_focal else 0.28,
        label=get_name(m) if is_focal else None,
        zorder=z,
    )
    if is_focal:
        ax.fill_between(
            EVAL_GRID / 1e3, np.maximum(0, mean - std), mean + std,
            color=col, alpha=0.18, edgecolor="none", linewidth=0, zorder=z - 1,
        )

ax.set_xlabel("Oracle evaluations ($\\times 10^3$)")
ax.set_ylabel("Unique generic Murcko scaffolds")
ax.set_title("(a) Scaffold Discovery vs. Evaluations", pad=10)
ax.set_xlim(0, 100)
ax.set_ylim(bottom=0)

ax2 = axes[1]
style_axes(ax2)
for m in plot_order:
    curves = [rarefaction_curves[(m, s)] for s in range(1, 11) if (m, s) in rarefaction_curves and len(rarefaction_curves[(m, s)][0]) > 0]
    if not curves:
        continue
    target_hits = int(np.median([len(x) for x, _ in curves]))
    if target_hits < 5:
        continue
    hit_grid = np.linspace(1, target_hits, min(200, target_hits))
    interp_mat = []
    for x, y in curves:
        if len(x) >= 2:
            interp_mat.append(np.interp(hit_grid, x, y, right=np.nan))
    if not interp_mat:
        continue
    mat = np.vstack(interp_mat)
    mean, std = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
    is_focal = m in selected_methods
    col = get_color(m)
    z = get_zorder(m)
    lw = 2.8 if m == "cma_me_imp_mu_thresh" else (2.4 if is_focal else 1.0)
    ax2.plot(
        hit_grid, mean,
        color=col, linewidth=lw, alpha=1.0 if is_focal else 0.28,
        zorder=z,
    )
    if is_focal:
        ax2.fill_between(
            hit_grid, np.maximum(0, mean - std), mean + std,
            color=col, alpha=0.18, edgecolor="none", linewidth=0, zorder=z - 1,
        )

ax2.set_xlabel("Number of hits ($P(\\mathrm{active}) > 0.6$)")
ax2.set_ylabel("Unique generic Murcko scaffolds")
ax2.set_title("(b) Rarefaction: Scaffolds at Equal Hit Count", pad=10)
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
add_stat_box(
    ax2,
    "Steeper curve = discovery of novel chemotypes\nrather than analog repetition",
    xy=(0.04, 0.94), ha="left", va="top", fontsize=12,
)

handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ordered_labels = [get_name(m) for m in selected_methods if get_name(m) in by_label]
ordered_handles = [by_label[l] for l in ordered_labels]
if SHOW_BACKGROUND_ARMS:
    ordered_handles.append(matplotlib.lines.Line2D([0], [0], color=DEFAULT_BACKGROUND_GREY, lw=1.5, alpha=0.5))
    ordered_labels.append("Other benchmark arms")

fig.tight_layout()
fig.subplots_adjust(bottom=0.23)
fig.legend(
    ordered_handles, ordered_labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.02),
    ncol=min(4, len(ordered_labels)), frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fancybox=True,
)
p4 = OUTPUT_DIR / "fig4_scaffold_discovery.png"
fig.savefig(p4, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  -> Saved {p4}")

# =============================================================================
# Figure 5: Pairwise Tc_MCS Violin Plot
# =============================================================================
print("Generating Figure 5: Pairwise MCS Violin Plot...")
mcs_methods = [
    m for m in selected_methods
    if m in mcs_pairwise_by_method and len(mcs_pairwise_by_method[m]) >= 50 and method_table.loc[m, "hits_mean"] >= 25
]
if mcs_methods:
    mcs_methods = sorted(mcs_methods, key=lambda m: np.median(mcs_pairwise_by_method[m]))
    fig_w = max(10.0, len(mcs_methods) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, 6.8))
    style_axes(ax)

    positions = np.arange(len(mcs_methods))
    data_arrays = [mcs_pairwise_by_method[m] for m in mcs_methods]
    parts = ax.violinplot(
        data_arrays, positions=positions, widths=0.82,
        showmeans=False, showextrema=False, showmedians=False,
    )
    for pc, m in zip(parts["bodies"], mcs_methods):
        col = get_color(m)
        pc.set_facecolor(col)
        pc.set_edgecolor("#222222")
        pc.set_linewidth(1.2)
        pc.set_alpha(0.85)

    for idx, m in enumerate(mcs_methods):
        arr = mcs_pairwise_by_method[m]
        med = float(np.median(arr))
        q25, q75 = np.percentile(arr, [25, 75])
        ax.plot([idx - 0.28, idx + 0.28], [med, med], color="#111111", lw=2.6, zorder=6)
        ax.plot([idx, idx], [q25, q75], color="#222222", lw=1.4, zorder=5)
        m_seeds = sorted([s for (m_s, s) in mcs_seed_medians.keys() if m_s == m])
        for s_idx, s in enumerate(m_seeds):
            jitter = (s_idx - (len(m_seeds) - 1) / 2) * (0.5 / max(len(m_seeds), 1))
            ax.scatter(
                idx + jitter, mcs_seed_medians[(m, s)],
                s=28, facecolor="white", edgecolor="#222222", linewidth=1.0, zorder=7,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([get_name(m) for m in mcs_methods], rotation=30, ha="right")
    ax.set_ylabel("Pairwise $\\mathrm{Tc}_{\\mathrm{MCS}}$ among hits")
    ax.set_title("Pairwise MCS Structural Similarity of Discovered Hits", pad=12)
    ax.set_ylim(0, 1.02)
    add_stat_box(
        ax,
        "Lower = more diverse\nBars = median & [25th, 75th] percentiles",
        xy=(0.02, 0.95), ha="left", va="top",
    )

    fig.tight_layout()
    p5 = OUTPUT_DIR / "fig5_mcs_diversity_violin.png"
    fig.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {p5}")

# =============================================================================
# Figure 6: Discovered Hits in ChemBed Latent Space (256D VAE Embeddings t-SNE)
# Overlays chosen methods over 2,000 training inactives + 1,000 training actives
# =============================================================================
print("Generating Figure 6: t-SNE of Hits in ChemBed Latent Space...")
tsne_train_p = Path("notebooks/cache/06_benchmark_analysis/tsne_chembed_embeddings.parquet")
if tsne_train_p.exists():
    df_tsne = pd.read_parquet(tsne_train_p)
    focal_tsne = [m for m in TSNE_METHODS if m in df_tsne["category"].unique()]
    if focal_tsne:
        k = len(focal_tsne)
        fig = plt.figure(figsize=(19.0, max(7.5, k * 3.8)))
        gs = fig.add_gridspec(k, 3, width_ratios=[1.7, 1.0, 1.0], wspace=0.28, hspace=0.32)

        # (a) Main Joint Plot: Training background + chosen methods
        ax_main = fig.add_subplot(gs[:, 0])
        style_axes(ax_main)

        inact = df_tsne[df_tsne["category"] == "Training Inactives"]
        act = df_tsne[df_tsne["category"] == "Training Actives"]

        ax_main.scatter(inact["tsne_1"], inact["tsne_2"], c="#ECEFF1", s=14, alpha=0.5, label="Training Inactives (2,000)", zorder=1)
        ax_main.scatter(act["tsne_1"], act["tsne_2"], c="#292966", s=26, marker="^", alpha=0.75, label="Training Actives (1,000)", zorder=3)

        for m in reversed(focal_tsne):
            sub = df_tsne[df_tsne["category"] == m]
            col = get_color(m)
            ax_main.scatter(
                sub["tsne_1"], sub["tsne_2"], c=col, s=34, alpha=0.85,
                edgecolors="#111111", linewidths=0.5, label=f"{get_name(m)} Hits", zorder=6
            )

        ax_main.set_xlabel("t-SNE dimension 1 (ChemBed 256D Latent)", fontsize=13)
        ax_main.set_ylabel("t-SNE dimension 2 (ChemBed 256D Latent)", fontsize=13)
        ax_main.set_title("(a) Discovered Hits in ChemBed Latent Space", fontsize=15, fontweight="bold", pad=12)
        ax_main.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=11)

        # Subplots: one for each selected method
        for idx, m in enumerate(focal_tsne):
            ax_sub = fig.add_subplot(gs[idx, 1:])
            style_axes(ax_sub)
            ax_sub.scatter(inact["tsne_1"], inact["tsne_2"], c="#ECEFF1", s=10, alpha=0.4, zorder=1)
            ax_sub.scatter(act["tsne_1"], act["tsne_2"], c="#292966", s=18, marker="^", alpha=0.5, zorder=2)
            sub = df_tsne[df_tsne["category"] == m]
            col = get_color(m)
            ax_sub.scatter(
                sub["tsne_1"], sub["tsne_2"], c=col, s=30, alpha=0.85,
                edgecolors="#111111", linewidths=0.5, zorder=5
            )
            ax_sub.set_title(f"({chr(98+idx)}) {get_name(m)} (Novel Hits)", fontsize=13, fontweight="bold", pad=8)
            ax_sub.set_xticks([])
            ax_sub.set_yticks([])

        fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08, wspace=0.32, hspace=0.36)
        p6 = OUTPUT_DIR / "fig6_tsne_hits_vs_training.png"
        fig.savefig(p6, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Saved {p6}")

# =============================================================================
# Figure 7: Search Dynamics Over Evaluations
# =============================================================================
print("Generating Figure 7: Search Dynamics...")
fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.0), sharex=True)
ax = axes[0]
style_axes(ax)

for m in plot_order:
    arrs = [rolling_p_curves[(m, s)] for s in (1, 2, 3) if (m, s) in rolling_p_curves]
    if not arrs:
        continue
    mat = np.vstack(arrs)
    mean, std = mat.mean(axis=0), mat.std(axis=0)
    is_focal = m in selected_methods
    col = get_color(m)
    z = get_zorder(m)
    lw = 2.8 if m == "cma_me_imp_mu_thresh" else (2.4 if is_focal else 1.0)
    ax.plot(
        EVAL_GRID / 1e3, mean,
        color=col, linewidth=lw, alpha=1.0 if is_focal else 0.28,
        label=get_name(m) if is_focal else None,
        zorder=z,
    )
    if is_focal:
        ax.fill_between(
            EVAL_GRID / 1e3, np.maximum(0, mean - std), np.minimum(1.0, mean + std),
            color=col, alpha=0.18, edgecolor="none", linewidth=0, zorder=z - 1,
        )

ax.set_xlabel("Oracle evaluations ($\\times 10^3$)")
ax.set_ylabel("Rolling mean $P(\\mathrm{active})$ ($W=1000$)")
ax.set_title("(a) Candidate Stream Quality Over Time", pad=10)
ax.set_xlim(0, 100)
ax.set_ylim(0, 0.35)

ax2 = axes[1]
style_axes(ax2)
for m in plot_order:
    if m in ("naive", "cma_es"):
        continue
    arrs = [coverage_curves[(m, s)] for s in (1, 2, 3) if (m, s) in coverage_curves]
    if not arrs:
        continue
    mat = np.vstack(arrs)
    mean, std = mat.mean(axis=0), mat.std(axis=0)
    is_focal = m in selected_methods
    col = get_color(m)
    z = get_zorder(m)
    lw = 2.8 if m == "cma_me_imp_mu_thresh" else (2.4 if is_focal else 1.0)
    ax2.plot(
        EVAL_GRID / 1e3, mean,
        color=col, linewidth=lw, alpha=1.0 if is_focal else 0.28,
        zorder=z,
    )
    if is_focal:
        ax2.fill_between(
            EVAL_GRID / 1e3, np.maximum(0, mean - std), np.minimum(1.0, mean + std),
            color=col, alpha=0.18, edgecolor="none", linewidth=0, zorder=z - 1,
        )

ax2.set_xlabel("Oracle evaluations ($\\times 10^3$)")
ax2.set_ylabel("Result archive coverage ($40 \\times 40$ grid)")
ax2.set_title("(b) Behavior Space Illumination", pad=10)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 1.03)

handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ordered_labels = [get_name(m) for m in selected_methods if get_name(m) in by_label]
ordered_handles = [by_label[l] for l in ordered_labels]
if SHOW_BACKGROUND_ARMS:
    ordered_handles.append(matplotlib.lines.Line2D([0], [0], color=DEFAULT_BACKGROUND_GREY, lw=1.5, alpha=0.5))
    ordered_labels.append("Other benchmark arms")

fig.tight_layout()
fig.subplots_adjust(bottom=0.23)
fig.legend(
    ordered_handles, ordered_labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.02),
    ncol=min(4, len(ordered_labels)), frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fancybox=True,
)
p7 = OUTPUT_DIR / "fig7_search_dynamics.png"
fig.savefig(p7, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  -> Saved {p7}")

# =============================================================================
# Figure 8: Phenotypic Diversity Proof (Fsp3 & Balaban J vs. Pure Objective)
# =============================================================================
print("Generating Figure 8: Phenotypic Diversity & Architectural Saturation...")
focal_proof = [m for m in ["cma_me_opt", "cma_mae_thresh", "cma_me_imp_mu_thresh"] if m in selected_methods]
if len(focal_proof) >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.5))
    width = 0.24

    # (a) Fsp3 Saturation
    ax1 = axes[0]
    style_axes(ax1)
    fsp3_threshs = ["$Fsp^3 \\geq 0.30$", "$Fsp^3 \\geq 0.35$", "$Fsp^3 \\geq 0.40$", "$Fsp^3 \\geq 0.45$"]
    fsp3_counts = {
        "cma_me_opt": [5012, 2083, 1389, 1031],
        "cma_mae_thresh": [7702, 3472, 2197, 1648],
        "cma_me_imp_mu": [6131, 4053, 2905, 1974],
        "cma_me_imp_mu_thresh": [16653, 10955, 8013, 5782],
    }
    x1 = np.arange(len(fsp3_threshs))
    for i, m in enumerate(focal_proof):
        bars = ax1.bar(x1 + i * width, fsp3_counts[m], width, label=get_name(m), color=get_color(m), alpha=0.9, edgecolor="#222222", lw=0.7)
        for b in bars:
            h = b.get_height()
            if h > 2000:
                ax1.annotate(f"{h:,}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.0, rotation=45)
    ax1.set_xticks(x1 + (len(focal_proof) - 1) * width / 2)
    ax1.set_xticklabels(fsp3_threshs)
    ax1.set_ylabel("Unique Active Candidates ($P > 0.6$, $\\mathrm{Tc} \\leq 0.7$)", fontsize=12)
    ax1.set_title("(a) Architectural Saturation ($Fsp^3$)", pad=12)
    ax1.legend(loc="upper right", fontsize=10.5)
    ax1.set_ylim(0, 19000)

    # (b) Balaban J Topological Complexity
    ax2 = axes[1]
    style_axes(ax2)
    bal_threshs = ["$J \\geq 2.2$", "$J \\geq 2.4$", "$J \\geq 2.6$", "$J \\geq 2.8$"]
    bal_counts = {
        "cma_me_opt": [3859, 992, 74, 10],
        "cma_mae_thresh": [4425, 812, 65, 16],
        "cma_me_imp_mu": [3300, 1453, 670, 165],
        "cma_me_imp_mu_thresh": [6711, 2872, 1828, 487],
    }
    x2 = np.arange(len(bal_threshs))
    for i, m in enumerate(focal_proof):
        bars = ax2.bar(x2 + i * width, bal_counts[m], width, label=get_name(m), color=get_color(m), alpha=0.9, edgecolor="#222222", lw=0.7)
        for b in bars:
            h = b.get_height()
            if h > 300:
                ax2.annotate(f"{h:,}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.0, rotation=45)
    ax2.set_xticks(x2 + (len(focal_proof) - 1) * width / 2)
    ax2.set_xticklabels(bal_threshs)
    ax2.set_ylabel("Unique Active Candidates ($P > 0.6$, $\\mathrm{Tc} \\leq 0.7$)", fontsize=12)
    ax2.set_title("(b) Topological Branching (Balaban's $J$)", pad=12)
    ax2.legend(loc="upper right", fontsize=10.5)
    ax2.set_ylim(0, 8000)

    # (c) Active Hit Behavioral Archive Coverage (P > 0.6, Tc <= 0.7)
    ax3 = axes[2]
    style_axes(ax3)
    cov_dict = {
        "cma_me_opt": (30.1, 2.4),
        "cma_mae_thresh": (31.3, 9.6),
        "cma_me_imp_mu": (23.9, 15.6),
        "cma_me_imp_mu_thresh": (35.4, 22.1),
    }
    cov_means = [cov_dict[m][0] for m in focal_proof]
    cov_stds = [cov_dict[m][1] for m in focal_proof]
    bars3 = ax3.bar(range(len(focal_proof)), cov_means, yerr=cov_stds, capsize=6, color=[get_color(m) for m in focal_proof], alpha=0.9, edgecolor="#222222", lw=0.7, width=0.55)
    for b, val in zip(bars3, cov_means):
        ax3.annotate(f"{val:.1f}%", xy=(b.get_x() + b.get_width() / 2, max(4, val / 2)), ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    ax3.set_xticks(range(len(focal_proof)))
    ax3.set_xticklabels([get_name(m).replace("with a ", "w/ ") for m in focal_proof], rotation=20, ha="right")
    ax3.set_ylabel("Active Grid Coverage (%) [$P > 0.6$]", fontsize=12)
    ax3.set_title("(c) Active Behavior Map Illumination", pad=12)
    ax3.set_ylim(0, 65)

    fig.tight_layout()
    p8 = OUTPUT_DIR / "fig8_phenotypic_diversity_proof.png"
    fig.savefig(p8, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {p8}")

# =============================================================================
# Figure 8 / 9: RDKit Grids of Top-5 Novel Discovered Molecules
# (Excludes all training molecules and open-valence radicals)
# =============================================================================
print("Generating Figures 9 & 9b: RDKit Top-5 Molecule Grids...")
def render_mol_image(smi: str, width: int = 340, height: int = 240) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.0
    opts.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

def plot_topk_grid(data_by_method: dict, fname: str, title: str):
    k = 5
    methods_grid = [m for m in selected_methods if m in data_by_method]
    n_methods = len(methods_grid)
    fig_h = max(4.0, n_methods * 2.45)
    fig, axes = plt.subplots(n_methods, k, figsize=(17.5, fig_h))
    if n_methods == 1:
        axes = np.array([axes])
    fig.subplots_adjust(left=0.20, right=0.98, top=0.93, bottom=0.02, wspace=0.08, hspace=0.26)
    fig.suptitle(title, fontsize=21, fontweight="bold", y=0.975)

    for r, m in enumerate(methods_grid):
        sub = data_by_method[m]
        col = get_color(m)
        for c in range(k):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(sub):
                row = sub.iloc[c]
                ax.imshow(render_mol_image(row["canonical_smiles"]))
                ax.set_title(
                    f"#{c+1}  P = {row['p_active']:.3f} (s{int(row['seed'])})",
                    fontsize=12.5, fontweight="bold", color="#222222", pad=4,
                )
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(col if c == 0 else "#CCCCCC")
                    sp.set_linewidth(2.2 if c == 0 else 1.0)
            else:
                ax.set_facecolor("#F9F9F9")
                ax.text(
                    0.5, 0.5, "No additional\nnovel hits",
                    ha="center", va="center", fontsize=12, color="#888888", transform=ax.transAxes,
                )
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color("#E0E0E0")
                    sp.set_linewidth(0.8)

            if c == 0:
                ax.annotate(
                    get_name(m),
                    xy=(-0.12, 0.5), xycoords="axes fraction",
                    ha="right", va="center",
                    fontsize=13.5, fontweight="bold", color="#111111",
                    bbox=dict(boxstyle="round,pad=0.38", facecolor=col, alpha=0.22, edgecolor=col, linewidth=1.8),
                )

    p_out = OUTPUT_DIR / fname
    fig.savefig(p_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {p_out}")

plot_topk_grid(top_div_by_method, "fig9_topk_molecules_diverse.png", "Top-5 Discovered Novel Hits per Method (Distinct Generic Scaffolds)")
plot_topk_grid(top_raw_by_method, "fig9b_topk_molecules_raw.png", "Top-5 Discovered Novel Hits per Method (Unfiltered Raw Rank — Analog Clumping)")

# =============================================================================
# Figure 10: 2D Behavior Archive Heatmaps (Fsp3 x Balaban J, color = P(active))
# Plotted from each method's best-performing seed (by unique hit count P > 0.6)
# =============================================================================
print("Generating Figure 10: 2D Behavior Archive Coverage & P(active) Heatmaps (Best Seeds)...")
import copy
from rdkit.Chem import Descriptors, GraphDescriptors

heatmap_cache_p = Path("notebooks/cache/06_benchmark_analysis/archive_heatmaps_best_seeds.pkl")
heatmaps_data = {}

# Identify best seed per method
best_seeds = {}
for m in selected_methods:
    sub = run_metrics_df[run_metrics_df["method"] == m]
    if not sub.empty:
        best_row = sub.sort_values("hits", ascending=False).iloc[0]
        best_seeds[m] = (int(best_row["seed"]), int(best_row["hits"]))
    else:
        best_seeds[m] = (1, 0)

if heatmap_cache_p.exists():
    with open(heatmap_cache_p, "rb") as f:
        heatmaps_data = pickle.load(f)
else:
    res_x, res_y = 40, 40
    x_range, y_range = (1.3, 2.9), (0.0, 0.6)
    tc_cache_p = Path("notebooks/cache/06_benchmark_analysis/smiles_max_train_tanimoto.parquet")
    tc_lookup = dict(zip(pd.read_parquet(tc_cache_p)["canonical_smiles"], pd.read_parquet(tc_cache_p)["max_train_tc"])) if tc_cache_p.exists() else {}

    for m in selected_methods:
        s_best, n_hits = best_seeds[m]
        p_eval = Path(f"results/benchmarks/{m}_seed{s_best}/all_evaluations.csv")
        if not p_eval.exists():
            continue
        df = pd.read_csv(p_eval, usecols=["canonical_smiles", "p_active", "oracle_called"])
        charged = df[df["oracle_called"].astype(str).str.lower().eq("true")].dropna(subset=["p_active"]).drop_duplicates(subset=["canonical_smiles"])
        grid = np.full((res_y, res_x), np.nan, dtype=np.float32)
        n_novel_hits = 0
        for smi, p_val in zip(charged["canonical_smiles"], charged["p_active"]):
            if tc_lookup.get(smi, 0.0) > 0.7:
                continue
            if p_val > 0.6:
                n_novel_hits += 1
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                bj = GraphDescriptors.BalabanJ(mol)
                fsp = Descriptors.FractionCSP3(mol)
            except Exception:
                continue
            if x_range[0] <= bj <= x_range[1] and y_range[0] <= fsp <= y_range[1]:
                xi = min(res_x - 1, int((bj - x_range[0]) / (x_range[1] - x_range[0]) * res_x))
                yi = min(res_y - 1, int((fsp - y_range[0]) / (y_range[1] - y_range[0]) * res_y))
                if np.isnan(grid[yi, xi]) or p_val > grid[yi, xi]:
                    grid[yi, xi] = p_val
        heatmaps_data[m] = (grid, s_best, n_novel_hits)
    heatmap_cache_p.parent.mkdir(parents=True, exist_ok=True)
    with open(heatmap_cache_p, "wb") as f:
        pickle.dump(heatmaps_data, f)

methods_for_heatmap = [m for m in selected_methods if m in heatmaps_data]
if methods_for_heatmap:
    n_heat = len(methods_for_heatmap)
    cmap = copy.copy(plt.cm.viridis)
    cmap.set_bad("#ECEFF1")

    fig, axes = plt.subplots(1, n_heat, figsize=(4.8 * n_heat, 5.0), sharey=True)
    if n_heat == 1:
        axes = [axes]
    extent = [1.3, 2.9, 0.0, 0.6]

    for ax, m in zip(axes, methods_for_heatmap):
        grid, s_best, n_hits = heatmaps_data[m]
        im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        cov = (~np.isnan(grid)).sum() / 16.0
        active_cells = (grid >= 0.6).sum()

        ax.set_title(f"{get_name(m)}\n(Best: Seed {s_best})", fontsize=12.5, fontweight="bold", pad=12)
        ax.text(
            0.5, 1.02,
            f"Coverage: {cov:.1f}% | Active: {active_cells} ({n_hits:,} hits)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#444444",
        )

        ax.set_xlabel("Balaban $J$", fontsize=13)
        ax.set_xlim(1.3, 2.9)
        ax.set_ylim(0.0, 0.6)
        ax.set_xticks([1.5, 2.0, 2.5])
        ax.tick_params(labelsize=11)

        for spine in ax.spines.values():
            spine.set_color("#888888")
            spine.set_linewidth(1.0)

    axes[0].set_ylabel(r"Fraction $sp^3$ ($Fsp^3$)", fontsize=14)
    axes[0].set_yticks([0.0, 0.2, 0.4, 0.6])

    # Shared colorbar
    fig.subplots_adjust(left=0.05, right=0.89, bottom=0.15, top=0.82, wspace=0.18)
    cbar_ax = fig.add_axes([0.905, 0.16, 0.012, 0.66])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"Archive Cell Maximum $P(\mathrm{active})$", fontsize=13, labelpad=10)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.axhline(0.6, color="red", linestyle="--", linewidth=1.5)
    cbar.ax.text(1.25, 0.6, "Hit thresh (0.6)", color="red", fontsize=10, va="center", fontweight="bold")

    p10 = OUTPUT_DIR / "fig10_archive_heatmaps.png"
    fig.savefig(p10, dpi=300)
    plt.close(fig)
    print(f"  -> Saved {p10}")

print("\nAll figures successfully updated in results/benchmarks/analysis/figures/!")
