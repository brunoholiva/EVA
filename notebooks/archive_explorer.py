"""Archive explorer: load, cluster, score, and visualize EVA result archives.

Shared chemistry utilities (scaffold, Tanimoto, scoring) live in eva_utils.py.
Production code is used directly where available:
  - reporting.multi_rep_archive.cluster_smiles for Butina clustering
  - chemistry.fingerprint for Morgan fingerprints
  - evaluation.activity for TabPFN prediction
  - reporting.plotting for archive figure creation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import SVG, display
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from notebooks.eva_utils import compute_scaffold, score_smiles

RDLogger.DisableLog("rdApp.*")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
AIZYNTH_CONFIG = (
    Path(__file__).resolve().parent.parent / "data" / "aizynth" / "config.yml"
)


# ---------------------------------------------------------------------------
# Archive discovery
# ---------------------------------------------------------------------------


def list_archives(results_dir: Path | str = RESULTS_DIR) -> dict[str, Path]:
    """Find all result_archive.csv files under results/.

    Returns
    -------
    dict mapping experiment name -> path to result_archive.csv
    """
    results_dir = Path(results_dir)
    archives = {}
    for exp_dir in sorted(results_dir.iterdir()):
        csv = exp_dir / "result_archive.csv"
        if csv.exists():
            archives[exp_dir.name] = csv
    return archives


def load_archive(path: Path | str) -> pd.DataFrame:
    """Load an archive CSV.

    Parameters
    ----------
    path : path to result_archive.csv

    Returns
    -------
    DataFrame with rank, smiles, p_active, and any extra columns present.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["smiles"])
    df["rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Molecule helpers
# ---------------------------------------------------------------------------


def smiles_to_mol(smi: str) -> Chem.Mol | None:
    """Convert SMILES to RDKit Mol, returning None on failure."""
    if not smi or not isinstance(smi, str):
        return None
    return Chem.MolFromSmiles(smi)


def load_training_actives(
    data_path: Path | str,
    smiles_col: str = "SMILES",
    target_col: str = "target",
) -> set[str]:
    """Load active SMILES from training CSV.

    Returns
    -------
    set of canonical SMILES strings (actives only)
    """
    df = pd.read_csv(data_path)
    df = df[df[target_col] == 1]
    smis = set()
    for s in df[smiles_col].dropna():
        mol = Chem.MolFromSmiles(s)
        if mol:
            smis.add(Chem.MolToSmiles(mol))
    return smis


def remove_training_molecules(
    df: pd.DataFrame, training_smiles: set[str]
) -> pd.DataFrame:
    """Remove molecules from archive that are in the training set."""
    mask = []
    for s in df["smiles"]:
        mol = Chem.MolFromSmiles(s)
        canonical = Chem.MolToSmiles(mol) if mol else s
        mask.append(canonical not in training_smiles)
    return df[mask].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Clustering (uses reporting.multi_rep_archive.cluster_smiles)
# ---------------------------------------------------------------------------


def tag_with_clusters(
    df: pd.DataFrame,
    threshold: float = 0.4,
    smiles_col: str = "smiles",
    score_col: str = "p_active",
) -> pd.DataFrame:
    """Cluster molecules and tag ALL rows with cluster_id and cluster_size.

    Uses Butina clustering from reporting.multi_rep_archive.cluster_smiles.

    Parameters
    ----------
    df : archive DataFrame
    threshold : Butina distance cutoff (1 - Tanimoto similarity)
    smiles_col : column with SMILES
    score_col : column to rank by (highest = best, used for cluster ordering)

    Returns
    -------
    DataFrame with added 'cluster_id' and 'cluster_size' columns.
    Cluster IDs are ordered by the best score in each cluster (descending).
    """
    from reporting.multi_rep_archive import cluster_smiles

    result = df.copy()
    clusters = cluster_smiles(result[smiles_col].tolist(), threshold=threshold)

    cluster_ids = np.full(len(result), -1, dtype=int)
    cluster_sizes = np.zeros(len(result), dtype=int)

    for cluster_idx, member_indices in enumerate(clusters):
        for idx in member_indices:
            cluster_ids[idx] = cluster_idx
            cluster_sizes[idx] = len(member_indices)

    result["cluster_id"] = cluster_ids
    result["cluster_size"] = cluster_sizes

    best_scores = result.groupby("cluster_id")[score_col].max()
    ordered_cluster_ids = best_scores.sort_values(ascending=False).index.tolist()
    remap = {old: new for new, old in enumerate(ordered_cluster_ids)}
    result["cluster_id"] = result["cluster_id"].map(remap)

    return result


# ---------------------------------------------------------------------------
# Archive visualization (heatmaps + 3D landscape)
# ---------------------------------------------------------------------------


def display_archive_heatmap(
    run_dir: Path | str,
    dimension_names: list[str] | None = None,
    suffix: str = "result_archive",
) -> None:
    """Load a saved archive and display its 2D heatmap.

    Parameters
    ----------
    run_dir : path to the run output directory
    dimension_names : names of the archive behavior dimensions
    suffix : archive suffix (default "result_archive")
    """
    import joblib
    from reporting.plotting import create_archive_figure

    run_dir = Path(run_dir)
    archive_path = run_dir / f"{suffix}.joblib"
    if not archive_path.exists():
        print(f"Archive not found: {archive_path}")
        return

    archive = joblib.load(archive_path)
    if dimension_names is None:
        dimension_names = [f"dim_{i}" for i in range(archive.measure_dim)]

    fig = create_archive_figure(archive, dimension_names)
    if fig is not None:
        display(fig)
    else:
        print("Could not create archive figure (need >= 2 dimensions).")


def plot_archive_landscape(
    gen0_csv: Path | str,
    final_csv: Path | str,
    dim_names: list[str],
    z_col: str = "p_active",
    resolution: int = 30,
    smooth_sigma: float = 1.2,
) -> plt.Figure:
    """Create a side-by-side 3D landscape plot of the archive at gen 0 vs final.

    Parameters
    ----------
    gen0_csv : path to archive_gen0.csv
    final_csv : path to archive.csv or result_archive.csv
    dim_names : names of the two behavior dimensions
    z_col : column to use as height (default "p_active")
    resolution : grid resolution for interpolation
    smooth_sigma : gaussian smoothing sigma

    Returns
    -------
    plt.Figure
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={"projection": "3d"})

    for ax, (csv_path, title) in zip(
        axes,
        [(gen0_csv, "Generation 0"), (final_csv, "Final")],
    ):
        df = pd.read_csv(csv_path)
        x = df[dim_names[0]].values
        y = df[dim_names[1]].values
        z = df[z_col].values

        xi = np.linspace(x.min(), x.max(), resolution)
        yi = np.linspace(y.min(), y.max(), resolution)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((x, y), z, (X, Y), method="linear")
        Z = np.nan_to_num(Z)
        Z = gaussian_filter(Z, sigma=smooth_sigma)

        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, edgecolor="none")
        ax.set_xlabel(dim_names[0])
        ax.set_ylabel(dim_names[1])
        ax.set_zlabel(z_col)
        ax.set_title(title)

    plt.suptitle("Archive Landscape: Beginning vs End", y=0.98)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_molecule(smi: str, size: tuple[int, int] = (350, 250)) -> SVG:
    """Draw a single molecule and return SVG."""
    mol = smiles_to_mol(smi)
    if mol is None:
        return SVG("<text>Invalid SMILES</text>")
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return SVG(drawer.GetDrawingText())


def draw_grid(
    smiles_list: list[str],
    mols_per_row: int = 4,
    subfig_size: tuple[int, int] = (300, 250),
    titles: list[str] | None = None,
) -> Draw:
    """Draw a grid of molecules.

    Parameters
    ----------
    smiles_list : list of SMILES strings
    mols_per_row : molecules per row
    subfig_size : (width, height) per molecule
    titles : optional list of titles (one per molecule)

    Returns
    -------
    IPython image object
    """
    mols = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is not None:
            mols.append(mol)
            valid_idx.append(i)

    if not mols:
        print("No valid molecules to draw.")
        return None

    kwargs: dict = {"subImgSize": subfig_size}
    if titles:
        valid_titles = [titles[i] for i in valid_idx]
        kwargs["legends"] = [t[:40] + "..." if len(t) > 40 else t for t in valid_titles]

    return Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, **kwargs)


def draw_molecules_interactive(
    df: pd.DataFrame,
    n_cols: int = 4,
    start: int = 0,
    end: int = 20,
    sort_by: str | None = "p_active",
    highlight_substructure: str | None = None,
):
    """Draw molecules from a DataFrame with sorting and substructure highlighting."""
    work = df.copy()
    if sort_by and sort_by in work.columns:
        work = work.sort_values(sort_by, ascending=False)

    work = work.iloc[start:end]

    mols = []
    titles = []
    hl_atoms = []
    for _, row in work.iterrows():
        mol = smiles_to_mol(row["smiles"])
        if mol is not None:
            mols.append(mol)
            title_parts = [f"#{int(row['rank'])}"]
            if "p_active" in row and pd.notna(row["p_active"]):
                title_parts.append(f"P={row['p_active']:.3f}")
            if "cluster_id" in row and pd.notna(row["cluster_id"]):
                title_parts.append(f"C{int(row['cluster_id'])}")
            titles.append(" | ".join(title_parts))

            if highlight_substructure:
                pattern = Chem.MolFromSmarts(highlight_substructure)
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    hl_atoms.append([atom for match in matches for atom in match])
                else:
                    hl_atoms.append([])
            else:
                hl_atoms.append([])

    if not mols:
        print("No valid molecules in range.")
        return

    short_titles = [t[:50] + "..." if len(t) > 50 else t for t in titles]
    kwargs: dict = {"subImgSize": (300, 250), "legends": short_titles}
    if any(hl_atoms):
        kwargs["highlightAtomLists"] = hl_atoms

    display(Draw.MolsToGridImage(mols, molsPerRow=n_cols, **kwargs))


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


def archive_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for an archive."""
    print(f"Total molecules: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if "p_active" in df.columns:
        pa = df["p_active"]
        print(
            f"P(active): min={pa.min():.4f}, median={pa.median():.4f}, max={pa.max():.4f}"
        )
        print(
            f"P>0.5: {(pa > 0.5).sum()}, P>0.6: {(pa > 0.6).sum()}, "
            f"P>0.7: {(pa > 0.7).sum()}, P>0.9: {(pa > 0.9).sum()}"
        )

    scaffolds = df["smiles"].apply(compute_scaffold)
    unique_scaffolds = scaffolds.dropna().nunique()
    print(f"Unique Murcko scaffolds: {unique_scaffolds}")

    if "cluster_id" in df.columns:
        n_clusters = df["cluster_id"].nunique()
        print(f"Clusters: {n_clusters}")
        if "cluster_size" in df.columns:
            sizes = df.groupby("cluster_id")["cluster_size"].first()
            print(
                f"Cluster sizes: min={sizes.min()}, median={sizes.median():.0f}, max={sizes.max()}"
            )

    if "max_sim" in df.columns:
        ms = df["max_sim"]
        print(
            f"Max sim to training: min={ms.min():.3f}, median={ms.median():.3f}, max={ms.max():.3f}"
        )


def pharmacophore_breakdown(df: pd.DataFrame, top_n: int = 100) -> None:
    """Show the dominant scaffolds in the top N molecules."""
    work = df.copy()
    if "p_active" in work.columns:
        work = work.sort_values("p_active", ascending=False)
    work = work.head(top_n)

    scaffolds = work["smiles"].apply(compute_scaffold)
    counts = scaffolds.value_counts()
    print(f"Top {top_n} molecules - unique Murcko scaffolds: {counts.shape[0]}")
    print("\nTop scaffolds:")
    for scaff, count in counts.head(15).items():
        print(f"  {count:3d}x  {scaff}")


# ---------------------------------------------------------------------------
# Interactive explorer widget
# ---------------------------------------------------------------------------


def explore_archives():
    """Launch an interactive widget to browse all archives."""
    archives = list_archives()
    names = list(archives.keys())

    dropdown = widgets.Dropdown(options=names, description="Experiment:")
    sort_col = widgets.Dropdown(
        options=["p_active", "rank", "max_sim", "ad"], description="Sort by:"
    )
    slice_start = widgets.IntSlider(value=0, min=0, max=500, description="Start:")
    slice_end = widgets.IntSlider(value=20, min=1, max=500, description="End:")
    highlight = widgets.Text(
        value="", placeholder="SMARTS pattern", description="Highlight:"
    )
    output = widgets.Output()

    def on_change(change=None):
        output.clear_output()
        with output:
            path = archives[dropdown.value]
            df = load_archive(path)
            archive_summary(df)
            print()
            pharmacophore_breakdown(df, top_n=slice_end.value)
            print()
            hl = highlight.value if highlight.value else None
            draw_molecules_interactive(
                df,
                n_cols=4,
                start=slice_start.value,
                end=slice_end.value,
                sort_by=sort_col.value,
                highlight_substructure=hl,
            )

    dropdown.observe(on_change, names="value")
    sort_col.observe(on_change, names="value")
    slice_start.observe(on_change, names="value")
    slice_end.observe(on_change, names="value")
    highlight.observe(on_change, names="value")

    ui = widgets.VBox(
        [
            widgets.HBox([dropdown, sort_col]),
            widgets.HBox([slice_start, slice_end]),
            highlight,
            output,
        ]
    )
    display(ui)
    on_change()


# ---------------------------------------------------------------------------
# Scoring (uses eva_utils.score_smiles)
# ---------------------------------------------------------------------------


def add_cytotox_scores(
    df: pd.DataFrame,
    model_path: Path | str,
    smiles_col: str = "smiles",
    device: str = "cuda",
    column_name: str = "p_cytotox",
) -> pd.DataFrame:
    """Add cytotoxicity scores to a DataFrame.

    Parameters
    ----------
    df : DataFrame with SMILES column
    model_path : path to cytotox TabPFN model (.joblib)
    smiles_col : column with SMILES
    device : device for inference
    column_name : name for the new column

    Returns
    -------
    DataFrame with added cytotoxicity column (lower = better).
    """
    import joblib
    from evaluation.activity import move_model_to_device

    model = joblib.load(model_path)
    model.inference_precision = "autocast"
    if device != "cpu":
        move_model_to_device(model, device)

    result = df.copy()
    valid_mask = result[smiles_col].notna()
    valid_smiles = result.loc[valid_mask, smiles_col].tolist()

    scores = np.full(len(result), np.nan)
    if valid_smiles:
        valid_scores = score_smiles(valid_smiles, model)
        scores[valid_mask.values] = valid_scores

    result[column_name] = scores
    return result


# ---------------------------------------------------------------------------
# AiZynthFinder retrosynthetic analysis
# ---------------------------------------------------------------------------


def _init_retro_worker(config_path: str) -> None:
    """Initialize a single AiZynthFinder instance per worker process."""
    global _worker_finder
    from aizynthfinder.aizynthfinder import AiZynthFinder

    _worker_finder = AiZynthFinder(configfile=config_path)
    _worker_finder.stock.select("zinc")
    _worker_finder.expansion_policy.select("uspto")
    _worker_finder.filter_policy.select("uspto")


def _retro_worker_search(
    smiles: str,
    time_limit: int,
    iteration_limit: int,
    max_transforms: int,
) -> dict:
    """Worker function that reuses the global finder instance."""
    global _worker_finder

    row = {"smiles": smiles}
    try:
        _worker_finder.config.search.time_limit = time_limit
        _worker_finder.config.search.iteration_limit = iteration_limit
        _worker_finder.config.search.max_transforms = max_transforms

        _worker_finder.target_smiles = smiles
        _worker_finder.tree_search()
        _worker_finder.build_routes()
        stats = _worker_finder.extract_statistics()

        row["is_solved"] = stats.get("is_solved", False)
        row["n_routes"] = stats.get("number_of_routes", 0)
        row["n_steps"] = stats.get("number_of_steps", 0)
        row["n_precursors"] = stats.get("number_of_precursors", 0)
        row["top_score"] = stats.get("top_score", 0.0)
        row["search_time"] = stats.get("search_time", 0.0)
    except Exception as e:
        row["is_solved"] = False
        row["n_routes"] = 0
        row["n_steps"] = 0
        row["n_precursors"] = 0
        row["top_score"] = 0.0
        row["search_time"] = 0.0
        row["error"] = str(e)

    return row


def _retro_worker_search_wrapper(task: tuple) -> dict:
    """Wrapper for imap_unordered (single-argument function)."""
    smiles, time_limit, iteration_limit, max_transforms = task
    return _retro_worker_search(smiles, time_limit, iteration_limit, max_transforms)


def aizynth_score(
    smiles_list: list[str],
    config_path: Path | str = AIZYNTH_CONFIG,
    time_limit: int = 120,
    iteration_limit: int = 100,
    max_transforms: int = 6,
    n_jobs: int = 4,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run AiZynthFinder tree search and return scores (parallelized).

    Uses multiprocessing with one finder instance per worker (~2 GB each).
    Each worker reuses its finder across all assigned molecules.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings to analyze.
    config_path : path to AiZynthFinder config.yml
    time_limit : max seconds per molecule
    iteration_limit : max MCTS iterations per molecule
    max_transforms : max depth of search tree
    n_jobs : number of parallel workers (default 4, ~8 GB memory)
    show_progress : show progress bar with ETA (default True)

    Returns
    -------
    DataFrame with columns: smiles, is_solved, n_routes, n_steps,
    n_precursors, top_score, search_time.
    """
    from multiprocessing import Pool

    tasks = [(smi, time_limit, iteration_limit, max_transforms) for smi in smiles_list]

    if n_jobs <= 1:
        _init_retro_worker(str(config_path))
        if show_progress:
            from rich.progress import track

            results = list(
                track(
                    (_retro_worker_search_wrapper(task) for task in tasks),
                    total=len(tasks),
                    description="Retrosynthesis",
                )
            )
        else:
            results = [_retro_worker_search_wrapper(task) for task in tasks]
    else:
        with Pool(
            n_jobs,
            initializer=_init_retro_worker,
            initargs=(str(config_path),),
        ) as pool:
            if show_progress:
                from rich.progress import track

                results = list(
                    track(
                        pool.imap_unordered(_retro_worker_search_wrapper, tasks),
                        total=len(tasks),
                        description="Retrosynthesis",
                    )
                )
            else:
                results = list(pool.imap_unordered(_retro_worker_search_wrapper, tasks))

    return pd.DataFrame(results)


def aizynth_find_routes(
    smiles: str,
    config_path: Path | str = AIZYNTH_CONFIG,
    time_limit: int = 120,
    iteration_limit: int = 100,
    max_transforms: int = 6,
):
    """Run AiZynthFinder and return the finder object with routes.

    Parameters
    ----------
    smiles : str
        Target SMILES string.
    config_path : path to AiZynthFinder config.yml
    time_limit : max seconds for search
    iteration_limit : max MCTS iterations
    max_transforms : max depth of search tree

    Returns
    -------
    AiZynthFinder instance with populated routes, or None if search failed.
    """
    from aizynthfinder.aizynthfinder import AiZynthFinder

    try:
        finder = AiZynthFinder(configfile=str(config_path))
        finder.stock.select("zinc")
        finder.expansion_policy.select("uspto")
        finder.filter_policy.select("uspto")

        finder.config.search.time_limit = time_limit
        finder.config.search.iteration_limit = iteration_limit
        finder.config.search.max_transforms = max_transforms

        finder.target_smiles = smiles
        finder.tree_search()
        finder.build_routes()
        return finder
    except Exception as e:
        print(f"Error running AiZynthFinder: {e}")
        return None


def draw_retro_route(
    finder,
    route_idx: int = 0,
    show_all: bool = True,
) -> Image | None:
    """Draw a retrosynthetic route from AiZynthFinder."""
    if (
        finder is None
        or not hasattr(finder, "routes")
        or len(finder.routes.reaction_trees) == 0
    ):
        print("No routes available to draw.")
        return None

    if route_idx >= len(finder.routes.reaction_trees):
        print(
            f"Route index {route_idx} out of range. Only {len(finder.routes.reaction_trees)} routes available."
        )
        return None

    reaction_tree = finder.routes.reaction_trees[route_idx]
    return reaction_tree.to_image(show_all=show_all)


def draw_retro_routes_grid(
    finder,
    n_routes: int = 4,
    max_cols: int = 2,
) -> Image | None:
    """Draw multiple retrosynthetic routes in a grid."""
    if (
        finder is None
        or not hasattr(finder, "routes")
        or len(finder.routes.reaction_trees) == 0
    ):
        print("No routes available to draw.")
        return None

    n_available = len(finder.routes.reaction_trees)
    n_routes = min(n_routes, n_available)

    images = []
    for i in range(n_routes):
        img = finder.routes.reaction_trees[i].to_image(show_all=True)
        images.append(img)

    n_cols = min(max_cols, n_routes)
    n_rows = (n_routes + n_cols - 1) // n_cols

    img_width, img_height = images[0].size
    grid_width = img_width * n_cols
    grid_height = img_height * n_rows

    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))

    return grid


# ---------------------------------------------------------------------------
# Composite ranking
# ---------------------------------------------------------------------------


def rank_composite(
    df: pd.DataFrame,
    higher_better: list[str] | None = None,
    lower_better: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank molecules by weighted composite score.

    Parameters
    ----------
    df : DataFrame with scoring columns
    higher_better : list of columns where higher values are better
    lower_better : list of columns where lower values are better
    weights : dict mapping column names to weights (default 1.0 for all)

    Returns
    -------
    DataFrame with added 'rank_composite' column (lower = better).
    """
    result = df.copy()
    rank_cols = []
    weights = weights or {}

    if higher_better:
        for col in higher_better:
            if col in result.columns:
                result[f"rank_{col}"] = result[col].rank(
                    ascending=False, na_option="bottom"
                )
                rank_cols.append(f"rank_{col}")

    if lower_better:
        for col in lower_better:
            if col in result.columns:
                result[f"rank_{col}"] = result[col].rank(
                    ascending=True, na_option="top"
                )
                rank_cols.append(f"rank_{col}")

    if rank_cols:
        weighted_ranks = []
        for rank_col in rank_cols:
            col_name = rank_col.replace("rank_", "", 1)
            weight = weights.get(col_name, 1.0)
            weighted_ranks.append(result[rank_col] * weight)

        result["rank_composite"] = sum(weighted_ranks)
        result = result.sort_values("rank_composite").reset_index(drop=True)

    return result


def best_per_cluster(
    df: pd.DataFrame,
    score_col: str = "rank_composite",
    ascending: bool = True,
) -> pd.DataFrame:
    """Select the best molecule from each cluster by a scoring column.

    Parameters
    ----------
    df : DataFrame with 'cluster_id' column
    score_col : column to rank by
    ascending : if True, lower is better; if False, higher is better

    Returns
    -------
    DataFrame with one row per cluster.
    """
    if "cluster_id" not in df.columns:
        return df

    representatives = []
    for cluster_id, group in df.groupby("cluster_id"):
        if ascending:
            best_idx = group[score_col].idxmin()
        else:
            best_idx = group[score_col].idxmax()
        rep = df.loc[best_idx].copy()
        representatives.append(rep)

    result = pd.DataFrame(representatives)
    result = result.sort_values(score_col, ascending=ascending).reset_index(drop=True)
    return result
