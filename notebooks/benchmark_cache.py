"""Shared styling, palette, and disk-caching utilities for the 06 benchmark notebooks."""

from __future__ import annotations

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, GraphDescriptors, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.manifold import TSNE

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemistry.scaffolds import generic_scaffold_hash

RUNS = ROOT / "results" / "benchmarks"
OUT = RUNS / "analysis"
FIGS = OUT / "figures"
CACHE = ROOT / "notebooks" / "cache" / "06_benchmark_analysis"
for _p in (OUT, FIGS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

P_CUT = 0.6
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
MAX_TRAIN_TC = 0.7  # Maximum allowed Tanimoto similarity to training actives (filters analogs > 0.7)
SAMPLE_N = 200  # up to 200 hits sampled per seed (19,900 pairs/seed) for MCS
MCS_TIMEOUT = 1  # seconds per FindMCS call
SAMPLE_SEED = 42
EVAL_GRID = np.linspace(0, 100_000, 101)  # 1,000-step grid for curves

# Complete human-readable labels for all 29 benchmark algorithms
ALL_LABELS = {
    "cma_me_imp_mu_thresh": "EVA (CMA-ME imp-$\\mu$ thresh)",
    "naive": "Naive prior $\\mathcal{N}(0, I)$",
    "cma_es": "CMA-ES (1-cell obj)",
    "cma_me_opt": "CMA-ME opt (pure obj)",
    "cma_me_opt_thresh": "CMA-ME opt (thresh)",
    "cma_mae": "CMA-MAE (unthresholded)",
    "cma_mae_thresh": "CMA-MAE (thresh)",
    "me_map_elites": "ME-MAP-Elites",
    "me_map_elites_thresh": "ME-MAP-Elites (thresh)",
    "cma_me_imp": "CMA-ME imp-filter",
    "cma_me_imp_thresh": "CMA-ME imp-filter (thresh)",
    "cma_me_imp_mu": "CMA-ME imp-$\\mu$",
    "cma_me_basic": "CMA-ME basic",
    "cma_me_basic_thresh": "CMA-ME basic (thresh)",
    "cma_me_mixed": "CMA-ME mixed",
    "cma_me_mixed_thresh": "CMA-ME mixed (thresh)",
    "cma_me_rd": "CMA-ME rank-diff",
    "cma_me_rd_thresh": "CMA-ME rank-diff (thresh)",
    "cma_me_rd_mu": "CMA-ME rank-diff-$\\mu$",
    "cma_me_rd_mu_thresh": "CMA-ME rank-diff-$\\mu$ (thresh)",
    "cma_me_obj_bandit": "CMA-ME obj bandit",
    "cma_me_obj_bandit_thresh": "CMA-ME obj bandit (thresh)",
    "map_elites": "MAP-Elites",
    "line_map_elites": "Line-MAP-Elites",
    "cvt_map_elites": "CVT-MAP-Elites",
    "line_cvt_map_elites": "Line-CVT-MAP-Elites",
    "ns_cma": "NS-CMA",
    "nslc_cma_imp": "NSLC-CMA imp",
    "dds_kde": "DDS-KDE",
}

ALL_SHORT_TITLES = {
    "cma_me_imp_mu_thresh": "CMA-ME imp-$\\mu$)",
    "naive": "Naive Sampling $\\mathcal{N}(0, I)$",
    "cma_es": "CMA-ES (No Diversity)",
    "cma_me_opt": "CMA-ME opt (pure obj)",
    "cma_me_opt_thresh": "CMA-ME opt (thresh)",
    "cma_mae": "CMA-MAE (unthresh)",
    "cma_mae_thresh": "CMA-MAE (thresh)",
    "me_map_elites": "ME-MAP-Elites",
    "me_map_elites_thresh": "ME-MAP-Elites (thresh)",
    "cma_me_imp": "CMA-ME imp-filter",
    "cma_me_imp_thresh": "CMA-ME imp-filter (thresh)",
    "cma_me_imp_mu": "CMA-ME imp-$\\mu$",
    "cma_me_basic": "CMA-ME basic",
    "cma_me_basic_thresh": "CMA-ME basic (thresh)",
    "cma_me_mixed": "CMA-ME mixed",
    "cma_me_mixed_thresh": "CMA-ME mixed (thresh)",
    "cma_me_rd": "CMA-ME rank-diff",
    "cma_me_rd_thresh": "CMA-ME rank-diff (thresh)",
    "cma_me_rd_mu": "CMA-ME rank-diff-$\\mu$",
    "cma_me_rd_mu_thresh": "CMA-ME rank-diff-$\\mu$ (thresh)",
    "cma_me_obj_bandit": "CMA-ME obj bandit",
    "cma_me_obj_bandit_thresh": "CMA-ME obj bandit (thresh)",
    "map_elites": "MAP-Elites",
    "line_map_elites": "Line-MAP-Elites",
    "cvt_map_elites": "CVT-MAP-Elites",
    "line_cvt_map_elites": "Line-CVT-MAP-Elites",
    "ns_cma": "NS-CMA",
    "nslc_cma_imp": "NSLC-CMA imp",
    "dds_kde": "DDS-KDE",
}

# Pre-defined scientific selection presets
METHOD_PRESETS = {
    # 5 core benchmark methods
    "paper": [
        "cma_me_imp_mu_thresh",
        "cma_me_opt",
        "cma_me_imp_mu",
        "cma_mae_thresh",
        "naive",
    ],
    "core": [
        "cma_me_imp_mu_thresh",
        "cma_me_opt",
        "cma_me_imp_mu",
        "cma_mae_thresh",
        "naive",
    ],
    # 8 focal methods currently reported in the default figures
    "focal": [
        "cma_me_imp_mu_thresh",
        "cma_me_opt",
        "cma_me_imp_mu",
        "cma_mae_thresh",
        "naive",
        "me_map_elites_thresh",
        "cma_me_imp_thresh",
        "cma_es",
        "cma_mae",
    ],
    # Comparison of pure objective optimization vs. quality-diversity
    "opt_vs_qd": [
        "cma_me_imp_mu_thresh",
        "cma_me_opt",
        "cma_es",
        "cma_mae_thresh",
        "naive",
    ],
    # Ablation showing effect of threshold_min = 0.6 across 3 algorithm families
    "threshold_ablation": [
        "cma_me_imp_mu_thresh",
        "cma_me_imp_mu",
        "cma_mae_thresh",
        "cma_mae",
        "me_map_elites_thresh",
        "me_map_elites",
    ],
    # All 29 algorithms
    "all": list(ALL_LABELS.keys()),
}

# Strict priority palette sequence (never red, green, or brown)
PALETTE_SLOTS = [
    "#292966",  # Slot 1: deep navy
    "#9999FF",  # Slot 2: light blue tint
    "#995C1F",  # Slot 3: dark amber
    "#FFCC99",  # Slot 4: peach tint
    "#2BA59A",  # Slot 5: teal
    "#666666",  # Slot 6: dark grey
]
DEFAULT_GREY = "#B3B3B3"  # Background / unselected methods
FOCUS_BLUE = "#6666FF"    # Focus method (EVA)
NAIVE_ORANGE = "#FF9933"  # Naive baseline


class SelectionConfig:
    """Active method selection, color palette, and display configuration."""

    def __init__(
        self,
        methods: list[str],
        focus_method: str = "cma_me_imp_mu_thresh",
        naive_method: str = "naive",
        show_background: bool = True,
    ):
        self.methods = list(methods)
        self.focus_method = focus_method
        self.naive_method = naive_method
        self.show_background = show_background

        # Build dynamic palette following strict rules
        palette: dict[str, str] = {}
        slot_idx = 0
        for m in self.methods:
            if m == self.focus_method:
                palette[m] = FOCUS_BLUE
            elif m == self.naive_method:
                palette[m] = NAIVE_ORANGE
            else:
                if slot_idx < len(PALETTE_SLOTS):
                    palette[m] = PALETTE_SLOTS[slot_idx]
                    slot_idx += 1
                else:
                    palette[m] = DEFAULT_GREY

        self.palette = palette
        self.labels = {m: ALL_LABELS.get(m, m) for m in self.methods}
        self.short_titles = {m: ALL_SHORT_TITLES.get(m, m) for m in self.methods}

    def color(self, m: str) -> str:
        return self.palette.get(m, DEFAULT_GREY)

    def label(self, m: str) -> str:
        return ALL_LABELS.get(m, m)

    def short_title(self, m: str) -> str:
        return ALL_SHORT_TITLES.get(m, m)

    def zorder(self, m: str) -> int:
        if m == self.focus_method:
            return 10
        if m == self.naive_method:
            return 8
        if m in self.palette:
            return 6
        return 2

    def plot_order(self, all_available: list[str]) -> list[str]:
        """Return plotting sequence (background arms first, selected arms on top)."""
        if self.show_background:
            bg = [m for m in all_available if m not in self.methods]
            return bg + [m for m in reversed(self.methods) if m in all_available]
        return [m for m in reversed(self.methods) if m in all_available]


# Module-level active configuration (defaults to 'focal' preset)
_ACTIVE_SELECTION = SelectionConfig(METHOD_PRESETS["focal"])

FOCUS_METHOD = _ACTIVE_SELECTION.focus_method
NAIVE_METHOD = _ACTIVE_SELECTION.naive_method
PALETTE = _ACTIVE_SELECTION.palette
FOCAL_METHODS = _ACTIVE_SELECTION.methods
LABELS = ALL_LABELS
SHORT_TITLES = ALL_SHORT_TITLES


def configure_selection(
    methods: list[str] | str = "focal",
    focus_method: str = "cma_me_imp_mu_thresh",
    naive_method: str = "naive",
    show_background: bool = True,
) -> SelectionConfig:
    """Select and configure which methods are highlighted and plotted.

    Parameters
    ----------
    methods : list[str] or str
        Either a preset name ('paper', 'core', 'focal', 'opt_vs_qd', 'threshold_ablation', 'all')
        or an explicit list of algorithm names from the 29 benchmark arms.
    focus_method : str
        Method to highlight with the primary blue palette color (#6666FF). Default: 'cma_me_imp_mu_thresh'.
    naive_method : str
        Method to highlight with the orange baseline color (#FF9933). Default: 'naive'.
    show_background : bool
        If True, unselected methods are plotted as thin background grey lines/dots (#B3B3B3).
        If False, ONLY the selected methods are plotted, completely hiding other arms.

    Returns
    -------
    SelectionConfig
        Config object with .methods, .palette, .labels, and .plot_order().
    """
    global _ACTIVE_SELECTION, FOCUS_METHOD, NAIVE_METHOD, PALETTE, FOCAL_METHODS
    if isinstance(methods, str):
        if methods not in METHOD_PRESETS:
            raise ValueError(f"Unknown preset {methods!r}. Choose from: {list(METHOD_PRESETS.keys())}")
        chosen = METHOD_PRESETS[methods]
    else:
        chosen = list(methods)

    _ACTIVE_SELECTION = SelectionConfig(
        chosen,
        focus_method=focus_method,
        naive_method=naive_method,
        show_background=show_background,
    )
    FOCUS_METHOD = _ACTIVE_SELECTION.focus_method
    NAIVE_METHOD = _ACTIVE_SELECTION.naive_method
    PALETTE = _ACTIVE_SELECTION.palette
    FOCAL_METHODS = _ACTIVE_SELECTION.methods
    return _ACTIVE_SELECTION


def apply_strict_style() -> None:
    """Configure matplotlib rcParams to follow the strict publication style."""
    plt.rcParams.update(
        {
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
            "figure.dpi": 100,  # Keep inline notebook display lightweight; fig.savefig uses dpi=300
        }
    )


def method_color(m: str) -> str:
    return PALETTE.get(m, DEFAULT_GREY)


def method_label(m: str) -> str:
    return LABELS.get(m, m)


def method_zorder(m: str) -> int:
    if m == FOCUS_METHOD:
        return 10
    if m == NAIVE_METHOD:
        return 8
    if m in PALETTE:
        return 6
    return 2


def style_axes(ax, is_heatmap: bool = False) -> None:
    ax.grid(False)
    if is_heatmap:
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("#444444")
            s.set_linewidth(1.0)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#222222")
        ax.spines["bottom"].set_color("#222222")
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_linewidth(1.1)


def add_stat_box(ax, text: str, xy=(0.97, 0.93), ha="right", va="top", fontsize=13) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color="#222222",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="#CCCCCC",
            alpha=0.94,
            linewidth=1.1,
        ),
        zorder=15,
    )


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return "invalid"
    if not scaff or not scaff.GetRingInfo().NumRings():
        return "acyclic"
    return Chem.MolToSmiles(scaff)


def has_radical(smi: str) -> bool:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return True
    return any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms())


def _mcs_worker(payload):
    smis, pairs = payload
    mols = [Chem.MolFromSmiles(s) for s in smis]
    counts = np.zeros(len(pairs), dtype=np.int16)
    for k, (i, j) in enumerate(pairs):
        res = rdFMCS.FindMCS(
            [mols[i], mols[j]],
            timeout=MCS_TIMEOUT,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
        counts[k] = res.numBonds
    return pairs, counts


def compute_mcs_tc_values(smiles_list: list[str], cache_path: Path) -> np.ndarray:
    n = len(smiles_list)
    if n < 2:
        return np.array([], dtype=np.float32)
    if cache_path.exists():
        mb = np.load(cache_path)
        if mb.shape == (n, n):
            ii, jj = np.triu_indices(n, 1)
            diag = np.diag(mb).astype(np.float32)
            c = mb[ii, jj].astype(np.float32)
            denom = diag[ii] + diag[jj] - c
            return np.where(denom > 0, c / denom, 0.0)

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    bonds = np.array([m.GetNumBonds() if m is not None else 1 for m in mols], dtype=np.float32)
    ii, jj = np.triu_indices(n, 1)
    pairs = list(zip(ii.tolist(), jj.tolist()))
    n_workers = min(12, os.cpu_count() or 4)
    n_chunks = max(1, min(len(pairs), n_workers * 2))
    chunks = [pairs[k::n_chunks] for k in range(n_chunks)]
    mb = np.zeros((n, n), dtype=np.int16)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for p_chunk, counts in pool.map(_mcs_worker, [(smiles_list, c) for c in chunks]):
            for (i, j), val in zip(p_chunk, counts):
                mb[i, j] = val
                mb[j, i] = val
    for i in range(n):
        mb[i, i] = int(bonds[i])
    np.save(cache_path, mb)
    c = mb[ii, jj].astype(np.float32)
    denom = bonds[ii] + bonds[jj] - c
    return np.where(denom > 0, c / denom, 0.0)


def load_or_compute_benchmark_cache(
    force_recompute: bool = False,
    max_train_tc: float | None = MAX_TRAIN_TC,
) -> dict[str, Any]:
    """Load all precomputed benchmark metrics, trajectories, MCS distributions, t-SNE, and top-k hits.

    Saves to ``notebooks/cache/06_benchmark_analysis/benchmark_bundle_v1.pkl`` so subsequent
    notebook runs complete in < 0.5s without re-reading the 87 ``all_evaluations.csv`` files.
    """
    bundle_path = CACHE / "benchmark_bundle_v1.pkl"
    if bundle_path.exists() and not force_recompute:
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
        print(
            f"Loaded cached benchmark bundle from {bundle_path.relative_to(ROOT)} "
            f"({len(bundle['run_metrics_df'])} runs, {len(bundle['method_table'])} methods)."
        )
        return bundle

    print(f"Computing benchmark cache across runs (filtering molecules with training Tc > {max_train_tc})...")
    actives_df = pd.read_parquet(ROOT / "notebooks" / "cache" / "01_chembed" / "reference_actives_minimol.parquet")
    actives_canon = {
        Chem.MolToSmiles(Chem.MolFromSmiles(s))
        for s in actives_df["smiles"]
        if Chem.MolFromSmiles(s) is not None
    }

    tc_lookup: dict[str, float] | None = None
    if max_train_tc is not None:
        tc_cache_p = CACHE / "smiles_max_train_tanimoto.parquet"
        if tc_cache_p.exists():
            tc_df = pd.read_parquet(tc_cache_p)
            tc_lookup = dict(zip(tc_df["canonical_smiles"], tc_df["max_train_tc"]))

    run_dirs = sorted(
        d
        for d in RUNS.iterdir()
        if d.is_dir() and (d / "summary.json").exists() and (d / "all_evaluations.csv").exists()
    )

    summary_rows = []
    discovery_curves: dict[float, dict[tuple[str, int], np.ndarray]] = {}
    scaffold_eval_curves: dict[tuple[str, int], np.ndarray] = {}
    rarefaction_curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    rolling_p_curves: dict[tuple[str, int], np.ndarray] = {}
    coverage_curves: dict[tuple[str, int], np.ndarray] = {}
    dup_rate_curves: dict[tuple[str, int], np.ndarray] = {}
    sampled_hits_by_run: dict[tuple[str, int], list[str]] = {}
    novel_hits_by_focal: dict[str, list[pd.DataFrame]] = {m: [] for m in FOCAL_METHODS}

    for d in run_dirs:
        method, _, seed_txt = d.name.rpartition("_seed")
        seed = int(seed_txt)
        summary = json.loads((d / "summary.json").read_text())
        curve_df = pd.read_csv(d / "curve.csv")
        last = curve_df.iloc[-1]

        df = pd.read_csv(
            d / "all_evaluations.csv",
            usecols=["canonical_smiles", "oracle_called", "p_active", "duplicate"],
        )
        charged = df[df["oracle_called"].astype(str).str.lower().eq("true")].copy()
        charged["canonical_smiles"] = charged["canonical_smiles"].astype(str)
        charged = charged.dropna(subset=["p_active"]).reset_index(drop=True)
        charged = charged.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)
        n_scored = len(charged)
        p_vals = charged["p_active"].to_numpy(dtype=np.float64)
        eval_idx = np.arange(1, n_scored + 1, dtype=np.float64)

        if max_train_tc is not None and tc_lookup is not None:
            too_similar = np.array([tc_lookup.get(s, 0.0) > max_train_tc for s in charged["canonical_smiles"]])
            p_vals = p_vals.copy()
            p_vals[too_similar] = 0.0

        thresh_counts = {}
        for t in THRESHOLDS:
            cum_hits = np.cumsum(p_vals > t)
            thresh_counts[t] = int(cum_hits[-1]) if n_scored else 0
            if t in (0.6, 0.7, 0.8):
                interp_hits = np.interp(
                    EVAL_GRID,
                    np.concatenate([[0.0], eval_idx]),
                    np.concatenate([[0.0], cum_hits]),
                )
                discovery_curves.setdefault(t, {})[(method, seed)] = interp_hits

        hit_mask = p_vals > P_CUT
        hits_df = charged[hit_mask].reset_index(drop=True)
        n_hits = len(hits_df)
        is_novel = ~hits_df["canonical_smiles"].isin(actives_canon)
        novel_hits = int(is_novel.sum())

        if method in FOCAL_METHODS and novel_hits > 0:
            nov_df = hits_df[is_novel][["canonical_smiles", "p_active"]].copy()
            nov_df["seed"] = seed
            novel_hits_by_focal[method].append(nov_df)

        if n_hits >= 2:
            hits_sorted = hits_df.sort_values("canonical_smiles", kind="stable")
            rng = np.random.default_rng(SAMPLE_SEED + seed)
            take = rng.choice(n_hits, size=min(SAMPLE_N, n_hits), replace=False)
            sampled_hits_by_run[(method, seed)] = (
                hits_sorted.iloc[np.sort(take)]["canonical_smiles"].astype(str).tolist()
            )
        else:
            sampled_hits_by_run[(method, seed)] = []

        hit_gen_scaffolds = [generic_scaffold_hash(s) for s in hits_df["canonical_smiles"]]
        hit_mur_scaffolds = [murcko_scaffold(s) for s in hits_df["canonical_smiles"]]

        seen_gen = set()
        cum_gen_at_hit = np.zeros(n_hits, dtype=np.float64)
        for k, sc in enumerate(hit_gen_scaffolds):
            seen_gen.add(sc)
            cum_gen_at_hit[k] = len(seen_gen)

        hit_eval_positions = eval_idx[hit_mask]
        if n_hits > 0:
            scaff_on_grid = np.interp(
                EVAL_GRID,
                np.concatenate([[0.0], hit_eval_positions, [float(n_scored)]]),
                np.concatenate([[0.0], cum_gen_at_hit, [cum_gen_at_hit[-1]]]),
            )
        else:
            scaff_on_grid = np.zeros_like(EVAL_GRID)
        scaffold_eval_curves[(method, seed)] = scaff_on_grid
        rarefaction_curves[(method, seed)] = (np.arange(1, n_hits + 1), cum_gen_at_hit)

        w = 1000
        if n_scored >= w:
            csum = np.concatenate([[0.0], np.cumsum(p_vals)])
            roll_mean = (csum[w:] - csum[:-w]) / float(w)
            roll_x = eval_idx[w - 1 :]
            roll_interp = np.interp(EVAL_GRID, roll_x, roll_mean)
        else:
            roll_interp = np.full_like(EVAL_GRID, np.mean(p_vals) if n_scored else 0.0)
        rolling_p_curves[(method, seed)] = roll_interp

        cov_x = curve_df["oracle_calls"].to_numpy(dtype=np.float64)
        cov_y = curve_df["result_coverage"].to_numpy(dtype=np.float64)
        coverage_curves[(method, seed)] = np.interp(
            EVAL_GRID, np.concatenate([[0.0], cov_x]), np.concatenate([[0.0], cov_y])
        )

        dup_frac = curve_df["duplicate_attempts"].to_numpy(dtype=np.float64) / np.maximum(
            1.0,
            (curve_df["oracle_calls"] + curve_df["duplicate_attempts"] + curve_df["invalid_attempts"]).to_numpy(
                dtype=np.float64
            ),
        )
        dup_rate_curves[(method, seed)] = np.interp(
            EVAL_GRID, np.concatenate([[0.0], cov_x]), np.concatenate([[0.0], dup_frac])
        )

        gen_counts = pd.Series(hit_gen_scaffolds).value_counts() if n_hits else pd.Series(dtype=int)
        mur_counts = pd.Series(hit_mur_scaffolds).value_counts() if n_hits else pd.Series(dtype=int)

        # Compute active cell coverage and Fsp3 descriptors
        active_cells = set()
        hit_fsp3 = []
        hit_bj = []
        res_x, res_y = 40, 40
        x_range, y_range = (1.3, 2.9), (0.0, 0.6)
        if n_hits > 0:
            for smi in hits_df["canonical_smiles"]:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                try:
                    bj = GraphDescriptors.BalabanJ(mol)
                    fsp = Descriptors.FractionCSP3(mol)
                    hit_bj.append(bj)
                    hit_fsp3.append(fsp)
                    if x_range[0] <= bj <= x_range[1] and y_range[0] <= fsp <= y_range[1]:
                        xi = min(res_x - 1, int((bj - x_range[0]) / (x_range[1] - x_range[0]) * res_x))
                        yi = min(res_y - 1, int((fsp - y_range[0]) / (y_range[1] - y_range[0]) * res_y))
                        active_cells.add((xi, yi))
                except Exception:
                    continue

        summary_rows.append(
            {
                "method": method,
                "seed": seed,
                "unique_scored": n_scored,
                "attempts": int(summary["attempts"]),
                "duplicate_attempts": int(summary["duplicate_attempts"]),
                "validity": float(summary["validity"]),
                "best_p_active": float(summary["best_p_active"]),
                "result_coverage": float(last["result_coverage"]),
                "result_qd_score": float(last["result_qd_score"]),
                "active_cells": len(active_cells),
                "active_coverage": float(len(active_cells)) / 1600.0,
                "fsp3_mean": float(np.mean(hit_fsp3)) if hit_fsp3 else np.nan,
                "fsp3_ge_035": int(sum(x >= 0.35 for x in hit_fsp3)),
                "bj_mean": float(np.mean(hit_bj)) if hit_bj else np.nan,
                "hits_05": thresh_counts[0.5],
                "hits": thresh_counts[0.6],
                "hits_07": thresh_counts[0.7],
                "hits_08": thresh_counts[0.8],
                "hits_09": thresh_counts[0.9],
                "novel_hits": novel_hits,
                "novel_hits_per_10k": 1e4 * novel_hits / max(n_scored, 1),
                "generic_scaffolds": int(gen_counts.size),
                "murcko_scaffolds": int(mur_counts.size),
                "scaffolds_per_100_hits": 100.0 * int(gen_counts.size) / n_hits if n_hits else 0.0,
                "top1_generic_share": float(gen_counts.iloc[0] / gen_counts.sum()) if n_hits else np.nan,
            }
        )

    run_metrics_df = pd.DataFrame(summary_rows)
    run_metrics_df.to_csv(OUT / "run_metrics_all87.csv", index=False)

    # Pairwise Tc_MCS from cached .npy matrices (or compute if missing)
    mcs_pairwise_by_method: dict[str, np.ndarray] = {}
    mcs_seed_medians: dict[tuple[str, int], float] = {}
    mcs_medians: dict[str, float] = {}
    mcs_means: dict[str, float] = {}

    for method in sorted(run_metrics_df["method"].unique()):
        pooled_tc = []
        avail_seeds = sorted(run_metrics_df[run_metrics_df["method"] == method]["seed"].unique())
        for seed in avail_seeds:
            sample_smis = sampled_hits_by_run.get((method, seed), [])
            cache_f = CACHE / f"mcs_n{SAMPLE_N}_{method}_seed{seed}.npy"
            if len(sample_smis) < 2 and not cache_f.exists():
                continue
            tc_vals = compute_mcs_tc_values(sample_smis, cache_f)
            if tc_vals.size:
                pooled_tc.append(tc_vals)
                mcs_seed_medians[(method, seed)] = float(np.median(tc_vals))
        if pooled_tc:
            cat = np.concatenate(pooled_tc)
            mcs_pairwise_by_method[method] = cat
            mcs_medians[method] = float(np.median(cat))
            mcs_means[method] = float(np.mean(cat))
        else:
            mcs_medians[method] = np.nan
            mcs_means[method] = np.nan

    agg_mean = run_metrics_df.groupby("method").mean(numeric_only=True)
    agg_std = run_metrics_df.groupby("method").std(numeric_only=True)
    agg_med = run_metrics_df.groupby("method").median(numeric_only=True)

    method_table = pd.DataFrame(
        {
            "hits_mean": agg_mean["hits"],
            "hits_std": agg_std["hits"],
            "hits_median": agg_med["hits"],
            "novel_hits_mean": agg_mean["novel_hits"],
            "generic_scaffolds_mean": agg_mean["generic_scaffolds"],
            "generic_scaffolds_std": agg_std["generic_scaffolds"],
            "generic_scaffolds_median": agg_med["generic_scaffolds"],
            "murcko_scaffolds_mean": agg_mean["murcko_scaffolds"],
            "murcko_scaffolds_median": agg_med["murcko_scaffolds"],
            "scaffolds_per_100_hits": agg_mean["scaffolds_per_100_hits"],
            "mcs_tc_median": pd.Series(mcs_medians),
            "mcs_tc_mean": pd.Series(mcs_means),
            "coverage_mean": agg_mean["result_coverage"],
            "active_coverage_mean": agg_mean["active_coverage"],
            "fsp3_mean": agg_mean["fsp3_mean"],
            "fsp3_ge_035_mean": agg_mean["fsp3_ge_035"],
            "qd_score_mean": agg_mean["result_qd_score"],
            "duplicates_mean": agg_mean["duplicate_attempts"],
            "hits_05_mean": agg_mean["hits_05"],
            "hits_06_mean": agg_mean["hits"],
            "hits_07_mean": agg_mean["hits_07"],
            "hits_08_mean": agg_mean["hits_08"],
            "hits_09_mean": agg_mean["hits_09"],
        }
    ).sort_values("hits_mean", ascending=False)
    method_table.to_csv(OUT / "method_summary_all29.csv")

    # Top-5 novel molecules per focal method (both diverse scaffolds and raw rank)
    top_raw_by_method: dict[str, pd.DataFrame] = {}
    top_div_by_method: dict[str, pd.DataFrame] = {}
    for m in FOCAL_METHODS:
        dfs = novel_hits_by_focal.get(m, [])
        if dfs:
            pooled = (
                pd.concat(dfs, ignore_index=True)
                .sort_values("p_active", ascending=False)
                .drop_duplicates("canonical_smiles")
                .reset_index(drop=True)
            )
            pooled = pooled[~pooled["canonical_smiles"].map(has_radical)].reset_index(drop=True)
            pooled["generic"] = [generic_scaffold_hash(s) for s in pooled["canonical_smiles"]]
            top_raw_by_method[m] = pooled.head(5).reset_index(drop=True)
            top_div_by_method[m] = pooled.drop_duplicates("generic").head(5).reset_index(drop=True)
        else:
            top_raw_by_method[m] = pd.DataFrame(columns=["canonical_smiles", "p_active", "seed", "generic"])
            top_div_by_method[m] = pd.DataFrame(columns=["canonical_smiles", "p_active", "seed", "generic"])

    # t-SNE coordinates (load from parquet cache if available)
    tsne_cache = CACHE / "tsne_focal_hits_ecfp4.parquet"
    if tsne_cache.exists():
        tsne_df = pd.read_parquet(tsne_cache)
    else:
        rows = []
        for m in FOCAL_METHODS:
            avail_seeds = sorted(run_metrics_df[run_metrics_df["method"] == m]["seed"].unique())
            for s in avail_seeds:
                for smi in sampled_hits_by_run.get((m, s), []):
                    rows.append({"method": m, "seed": s, "smiles": str(smi)})
        tsne_df = pd.DataFrame(rows)
        fps = []
        for smi in tsne_df["smiles"]:
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros((2048,), dtype=bool)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        X_fp = np.vstack(fps)
        coords = TSNE(
            n_components=2,
            metric="jaccard",
            perplexity=30,
            init="random",
            learning_rate="auto",
            random_state=SAMPLE_SEED,
        ).fit_transform(X_fp)
        tsne_df["tsne_1"] = coords[:, 0]
        tsne_df["tsne_2"] = coords[:, 1]
        tsne_df.to_parquet(tsne_cache, index=False)

    bundle = {
        "run_metrics_df": run_metrics_df,
        "method_table": method_table,
        "discovery_curves": discovery_curves,
        "scaffold_eval_curves": scaffold_eval_curves,
        "rarefaction_curves": rarefaction_curves,
        "rolling_p_curves": rolling_p_curves,
        "coverage_curves": coverage_curves,
        "dup_rate_curves": dup_rate_curves,
        "mcs_pairwise_by_method": mcs_pairwise_by_method,
        "mcs_seed_medians": mcs_seed_medians,
        "top_raw_by_method": top_raw_by_method,
        "top_div_by_method": top_div_by_method,
        "tsne_df": tsne_df,
    }
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved complete benchmark cache bundle to {bundle_path.relative_to(ROOT)}")
    return bundle
