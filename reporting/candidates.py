"""Post-run candidate pipeline: dedup, score, cluster, rank, export."""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from chemistry.fingerprint import compute_morgan
from chemistry.smiles import canonicalize_smiles
from reporting.console import section, step
from reporting.suppress import suppress_rdkit_logs

suppress_rdkit_logs()

_RADIUS = 2
_N_BITS = 2048
_CHUNK_SIZE = 4096

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def smiles_to_svg(smiles: str, width: int = 200, height: int = 160) -> str | None:
    """Generate an SVG string for a SMILES string using Python RDKit.

    Parameters
    ----------
    smiles : str
        SMILES string.
    width : int
        SVG width in pixels.
    height : int
        SVG height in pixels.

    Returns
    -------
    str or None
        SVG markup string, or None if the SMILES is invalid.
    """
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def dedup_evaluations(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize SMILES and deduplicate, keeping the highest P(active).

    Parameters
    ----------
    df : DataFrame
        Raw ``all_evaluations.csv`` with columns
        ``eval, smiles, p_active, valid, gen``.

    Returns
    -------
    DataFrame
        Unique candidates with columns ``smiles, p_active, first_gen,
        last_gen, n_evals``.
    """
    step("Deduplicating evaluations")
    valid = df[df["valid"] == True].copy()  # noqa: E712
    valid["canonical"] = valid["smiles"].map(canonicalize_smiles)
    valid = valid[valid["canonical"].notna()]

    grouped = valid.groupby("canonical", sort=False).agg(
        p_active=("p_active", "max"),
        first_gen=("gen", "min"),
        last_gen=("gen", "max"),
        n_evals=("eval", "count"),
    )
    result = grouped.reset_index().rename(columns={"canonical": "smiles"})
    step(f"  {len(result):,} unique valid molecules from {len(df):,} rows")
    return result


def filter_by_activity(df: pd.DataFrame, p_active_min: float = 0.6) -> pd.DataFrame:
    """Keep only molecules above the P(active) threshold.

    Parameters
    ----------
    df : DataFrame
        Candidates with a ``p_active`` column.
    p_active_min : float
        Minimum P(active) threshold.

    Returns
    -------
    DataFrame
        Filtered candidates.
    """
    before = len(df)
    result = df[df["p_active"] > p_active_min].copy().reset_index(drop=True)
    step(
        f"  P(active) > {p_active_min}: {len(result):,} / {before:,} "
        f"({before - len(result):,} removed)"
    )
    return result


def load_training_actives(path: str | Path) -> set[str]:
    """Load canonical SMILES of training actives (target==1).

    Parameters
    ----------
    path : str or Path
        Path to ``predictor_training_data.csv``.

    Returns
    -------
    set of str
        Canonical SMILES strings of active training molecules.
    """
    df = pd.read_csv(path)
    actives = df[df["target"] == 1]["SMILES"]
    return {c for s in actives if (c := canonicalize_smiles(s))}


def remove_training_actives(
    df: pd.DataFrame, training_smiles: set[str]
) -> pd.DataFrame:
    """Remove candidates whose SMILES appear in the training active set.

    Parameters
    ----------
    df : DataFrame
        Candidates with a ``smiles`` column.
    training_smiles : set of str
        Canonical SMILES of training actives.

    Returns
    -------
    DataFrame
        Novel candidates not in training.
    """
    before = len(df)
    mask = ~df["smiles"].isin(training_smiles)
    result = df[mask].copy().reset_index(drop=True)
    step(
        f"  Remove training actives: {len(result):,} / {before:,} "
        f"({before - len(result):,} removed)"
    )
    return result


# ---------------------------------------------------------------------------
# Similarity to training
# ---------------------------------------------------------------------------


def compute_max_tanimoto_to_training(
    candidates: pd.DataFrame,
    training_smiles: list[str],
    training_fps: np.ndarray | None = None,
    training_sums: np.ndarray | None = None,
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    """Compute max Tanimoto similarity to the training active set.

    Uses chunked matrix multiplication for memory efficiency.

    Parameters
    ----------
    candidates : DataFrame
        Candidates with a SMILES column.
    training_smiles : list of str
        Canonical SMILES of training actives.
    training_fps : np.ndarray or None
        Pre-computed Morgan FPs for training set. If None, computed here.
    training_sums : np.ndarray or None
        Pre-computed row sums. If None, computed here.
    smiles_col : str
        Name of the SMILES column.

    Returns
    -------
    DataFrame
        Candidates with added ``max_tanimoto_similarity`` and
        ``similar_training_smiles`` columns.
    """
    step("Computing max Tanimoto to training actives")

    if training_fps is None:
        training_fps = _compute_morgan_matrix(training_smiles)
    if training_sums is None:
        training_sums = training_fps.sum(axis=1)

    cand_smiles = candidates[smiles_col].tolist()
    cand_fps = _compute_morgan_matrix(cand_smiles)
    cand_sums = cand_fps.sum(axis=1)

    n_cand = len(cand_smiles)
    max_sims = np.zeros(n_cand, dtype=np.float32)
    best_idx = np.zeros(n_cand, dtype=int)

    for start in range(0, n_cand, _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, n_cand)
        q = cand_fps[start:end]
        q_sums = cand_sums[start:end]

        inter = q @ training_fps.T
        union = q_sums[:, None] + training_sums[None, :] - inter
        sim = inter / (union + 1e-8)

        max_sims[start:end] = sim.max(axis=1)
        best_idx[start:end] = sim.argmax(axis=1)

    result = candidates.copy()
    result["max_tanimoto_similarity"] = max_sims
    result["similar_training_smiles"] = [
        training_smiles[i] if max_sims[k] > 0 else "" for k, i in enumerate(best_idx)
    ]
    step(f"  Max tanimoto range: {max_sims.min():.3f} – {max_sims.max():.3f}")
    return result


def _compute_morgan_matrix(smiles: list[str]) -> np.ndarray:
    """Compute Morgan fingerprint matrix (float32) from SMILES."""
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            rows.append(compute_morgan(mol, radius=_RADIUS, fp_size=_N_BITS))
        else:
            rows.append(np.zeros(_N_BITS, dtype=np.float32))
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def tag_with_clusters(
    df: pd.DataFrame,
    threshold: float = 0.65,
    smiles_col: str = "smiles",
    score_col: str = "p_active",
) -> pd.DataFrame:
    """Cluster molecules by Butina and tag with cluster_id / cluster_size.

    Cluster IDs are reordered so that cluster 0 contains the highest-scoring
    molecule, cluster 1 the second-highest, etc.

    Parameters
    ----------
    df : DataFrame
        Candidates.
    threshold : float
        Tanimoto distance cutoff (1 - similarity).
    smiles_col : str
        Column with SMILES.
    score_col : str
        Column to order clusters by (descending).

    Returns
    -------
    DataFrame
        Candidates with ``cluster_id`` and ``cluster_size`` columns.
    """
    from reporting.multi_rep_archive import cluster_smiles

    step(f"Butina clustering (threshold={threshold})")
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
    ordered = best_scores.sort_values(ascending=False).index.tolist()
    remap = {old: new for new, old in enumerate(ordered)}
    result["cluster_id"] = result["cluster_id"].map(remap)

    n_clusters = len(ordered)
    step(f"  {n_clusters} clusters, sizes {cluster_sizes.min()}–{cluster_sizes.max()}")
    return result


# ---------------------------------------------------------------------------
# Cytotoxicity scoring
# ---------------------------------------------------------------------------


def score_cytotox(
    df: pd.DataFrame,
    models: dict[str, Path | str],
    smiles_col: str = "smiles",
    device: str = "cuda",
) -> pd.DataFrame:
    """Score cytotoxicity for multiple cell-line models.

    Parameters
    ----------
    df : DataFrame
        Candidates.
    models : dict
        ``{column_suffix: model_path}`` mapping, e.g.
        ``{"3T3": "data/cytotox/tabpfn_3T3.joblib"}``.
    smiles_col : str
        Column with SMILES.
    device : str
        Device for TabPFN inference.

    Returns
    -------
    DataFrame
        Candidates with added ``p_cytotox_{suffix}`` columns.
    """
    import joblib

    from chemistry.features import MoleculeFeaturizer
    from evaluation.activity import move_model_to_device, predict_from_features

    result = df.copy()
    featurizer = MoleculeFeaturizer()

    for suffix, model_path in models.items():
        col_name = f"p_cytotox_{suffix}"
        step(f"Scoring cytotoxicity ({suffix})")

        model = joblib.load(model_path)
        model.inference_precision = "autocast"
        if device != "cpu":
            move_model_to_device(model, device)

        valid_mask = result[smiles_col].notna() & (result[smiles_col] != "")
        valid_smiles = result.loc[valid_mask, smiles_col].tolist()

        scores = np.full(len(result), np.nan)
        if valid_smiles:
            X = featurizer.transform(valid_smiles)
            _, probs = predict_from_features(X, model)
            scores[valid_mask.values] = probs[:, 1]

        result[col_name] = scores
        step(f"  {col_name}: {np.nanmean(scores):.3f} mean")

    return result


# ---------------------------------------------------------------------------
# AD scoring
# ---------------------------------------------------------------------------


def score_ad(
    df: pd.DataFrame,
    ad_model_path: Path | str,
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    """Compute applicability domain scores.

    Parameters
    ----------
    df : DataFrame
        Candidates.
    ad_model_path : Path
        Path to the kNN AD model joblib artifact.
    smiles_col : str
        Column with SMILES.

    Returns
    -------
    DataFrame
        Candidates with added ``ad_score`` column.
    """
    from chemistry.fingerprint import smiles_to_morgan
    from evaluation.applicability import ADScorer

    step("Scoring applicability domain")
    scorer = ADScorer(ad_model_path)

    fps, _valid_idx = smiles_to_morgan(
        df[smiles_col].tolist(),
        radius=scorer.radius,
        fp_size=scorer.n_bits,
    )

    valid_mask = fps.any(axis=1)
    ad_scores = np.full(len(df), 1.0)
    if valid_mask.any():
        ad_scores[valid_mask] = scorer.compute_from_fps(fps[valid_mask])

    result = df.copy()
    result["ad_score"] = ad_scores
    step(f"  AD score mean: {np.mean(ad_scores):.3f}")
    return result


# ---------------------------------------------------------------------------
# Retrosynthesis
# ---------------------------------------------------------------------------

_AIZYNTH_CONFIG = "data/aizynth/config.yml"


def _init_retro_worker(config_path: str) -> None:
    """Initialize one AiZynthFinder instance per worker process."""
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
    """Run tree search for a single molecule."""
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

        row["retro_solved"] = stats.get("is_solved", False)
        row["n_routes"] = stats.get("number_of_routes", 0)
        row["n_steps"] = stats.get("number_of_steps", 0)
        row["n_precursors"] = stats.get("number_of_precursors", 0)
        row["retro_score"] = stats.get("top_score", 0.0)
        row["search_time"] = stats.get("search_time", 0.0)
    except Exception:
        row["retro_solved"] = False
        row["n_routes"] = 0
        row["n_steps"] = 0
        row["n_precursors"] = 0
        row["retro_score"] = 0.0
        row["search_time"] = 0.0

    return row


def _retro_worker_wrapper(task: tuple) -> dict:
    """Unpack a task tuple for imap_unordered."""
    smiles, time_limit, iteration_limit, max_transforms = task
    return _retro_worker_search(smiles, time_limit, iteration_limit, max_transforms)


def aizynth_score(
    smiles_list: list[str],
    config_path: Path | str = _AIZYNTH_CONFIG,
    time_limit: int = 120,
    iteration_limit: int = 100,
    max_transforms: int = 6,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """Run AiZynthFinder tree search in parallel.

    Parameters
    ----------
    smiles_list : list of str
        SMILES to analyze.
    config_path : Path or str
        Path to AiZynthFinder ``config.yml``.
    time_limit : int
        Max seconds per molecule.
    iteration_limit : int
        Max MCTS iterations.
    max_transforms : int
        Max search tree depth.
    n_jobs : int
        Parallel workers (~2 GB each).

    Returns
    -------
    DataFrame
        Columns: ``smiles, retro_solved, n_routes, n_steps, n_precursors,
        retro_score, search_time``.
    """
    tasks = [(smi, time_limit, iteration_limit, max_transforms) for smi in smiles_list]

    if n_jobs <= 1:
        _init_retro_worker(str(config_path))
        results = [_retro_worker_wrapper(t) for t in tasks]
    else:
        with Pool(
            n_jobs,
            initializer=_init_retro_worker,
            initargs=(str(config_path),),
        ) as pool:
            results = list(pool.imap_unordered(_retro_worker_wrapper, tasks))

    return pd.DataFrame(results)


def aizynth_find_routes(
    smiles: str,
    config_path: Path | str = _AIZYNTH_CONFIG,
    time_limit: int = 120,
    iteration_limit: int = 100,
    max_transforms: int = 6,
):
    """Run AiZynthFinder and return the finder with populated routes.

    Parameters
    ----------
    smiles : str
        Target SMILES.
    config_path : Path or str
        AiZynthFinder config path.
    time_limit : int
        Max seconds.
    iteration_limit : int
        Max MCTS iterations.
    max_transforms : int
        Max tree depth.

    Returns
    -------
    AiZynthFinder or None
        Finder with routes, or None on failure.
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
    except Exception:
        return None


def draw_retro_route(finder, route_idx: int = 0):
    """Draw a single retrosynthetic route as a PIL Image.

    Parameters
    ----------
    finder : AiZynthFinder
        Finder with populated routes.
    route_idx : int
        Index of the route to draw.

    Returns
    -------
    PIL.Image or None
    """
    if (
        finder is None
        or not hasattr(finder, "routes")
        or len(finder.routes.reaction_trees) == 0
    ):
        return None

    if route_idx >= len(finder.routes.reaction_trees):
        return None

    return finder.routes.reaction_trees[route_idx].to_image(show_all=True)


def render_retro_routes(
    smiles_list: list[str],
    config_path: Path | str = _AIZYNTH_CONFIG,
    time_limit: int = 120,
    iteration_limit: int = 100,
    max_transforms: int = 6,
) -> dict[str, str]:
    """Render retrosynthetic route images and return base64-encoded PNGs.

    Parameters
    ----------
    smiles_list : list of str
        SMILES to find routes for.
    config_path : Path or str
        AiZynthFinder config path.
    time_limit : int
        Max seconds per molecule.
    iteration_limit : int
        Max MCTS iterations.
    max_transforms : int
        Max tree depth.

    Returns
    -------
    dict
        ``{smiles: base64_png_data_url}`` for solved molecules.
    """
    import base64
    from io import BytesIO

    images = {}
    for smi in smiles_list:
        finder = aizynth_find_routes(
            smi, config_path, time_limit, iteration_limit, max_transforms
        )
        img = draw_retro_route(finder)
        if img is not None:
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            images[smi] = f"data:image/png;base64,{b64}"
    return images


# ---------------------------------------------------------------------------
# Composite ranking
# ---------------------------------------------------------------------------


def rank_candidates(
    df: pd.DataFrame,
    higher_better: list[str] | None = None,
    lower_better: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank molecules by weighted composite score.

    Parameters
    ----------
    df : DataFrame
        Candidates with scoring columns.
    higher_better : list of str or None
        Columns where higher is better.
    lower_better : list of str or None
        Columns where lower is better.
    weights : dict or None
        Per-column weights (default 1.0).

    Returns
    -------
    DataFrame
        Candidates with rank columns and ``rank_composite``, sorted
        ascending (best first).
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
        weighted = []
        for rc in rank_cols:
            col_name = rc.replace("rank_", "", 1)
            w = weights.get(col_name, 1.0)
            weighted.append(result[rc] * w)
        result["rank_composite"] = sum(weighted)
        result = result.sort_values("rank_composite").reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Retrosynthesis integration
# ---------------------------------------------------------------------------


def run_retrosynthesis(
    df: pd.DataFrame,
    config_path: Path | str = _AIZYNTH_CONFIG,
    top_n_per_cluster: int = 3,
    retro_cap: int = 100,
    time_limit: int = 120,
    iteration_limit: int = 100,
    max_transforms: int = 6,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """Run retrosynthesis on top-N molecules per cluster.

    Parameters
    ----------
    df : DataFrame
        Candidates with ``cluster_id`` and ``rank_composite`` columns.
    config_path : Path or str
        AiZynthFinder config.
    top_n_per_cluster : int
        Max molecules per cluster to score.
    retro_cap : int
        Hard cap on total molecules to score.
    time_limit : int
        Seconds per molecule.
    iteration_limit : int
        MCTS iterations.
    max_transforms : int
        Tree depth.
    n_jobs : int
        Parallel workers.

    Returns
    -------
    DataFrame
        Candidates with ``retro_score, retro_solved, n_steps`` columns
        (NaN for non-scored molecules).
    """
    step(f"Selecting top-{top_n_per_cluster}/cluster for retrosynthesis")
    selected = (
        df.sort_values("rank_composite").groupby("cluster_id").head(top_n_per_cluster)
    )
    if len(selected) > retro_cap:
        selected = selected.head(retro_cap)
    step(f"  {len(selected)} molecules for retrosynthesis")

    retro_df = aizynth_score(
        selected["smiles"].tolist(),
        config_path=config_path,
        time_limit=time_limit,
        iteration_limit=iteration_limit,
        max_transforms=max_transforms,
        n_jobs=n_jobs,
    )

    result = df.copy()
    retro_cols = ["smiles", "retro_solved", "retro_score", "n_steps"]
    result = result.merge(retro_df[retro_cols], on="smiles", how="left")
    n_solved = retro_df["retro_solved"].sum()
    step(f"  {n_solved}/{len(retro_df)} retrosynthesis solved")
    return result


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_candidates_csv(df: pd.DataFrame, path: Path | str) -> Path:
    """Save candidates DataFrame to CSV.

    Parameters
    ----------
    df : DataFrame
        Full candidates table.
    path : Path or str
        Output CSV path.

    Returns
    -------
    Path
        Path to the written CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    step(f"  Saved {len(df):,} candidates → {path}")
    return path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    all_evaluations_path: Path | str,
    training_data_path: Path | str = "data/predictor/predictor_training_data.csv",
    ad_model_path: Path | str = "data/predictor/ad_model.joblib",
    cytotox_models: dict[str, Path | str] | None = None,
    p_active_min: float = 0.6,
    cluster_threshold: float = 0.65,
    device: str = "cuda",
) -> pd.DataFrame:
    """Run the full post-run candidate pipeline.

    Steps: dedup → activity filter → remove training → similarity →
    cluster → cytotox → AD.

    Parameters
    ----------
    all_evaluations_path : Path or str
        Path to ``all_evaluations.csv``.
    training_data_path : Path or str
        Training data CSV with SMILES + target columns.
    ad_model_path : Path or str
        kNN AD model joblib path.
    cytotox_models : dict or None
        ``{suffix: model_path}`` for cytotox models.
    p_active_min : float
        P(active) threshold.
    cluster_threshold : float
        Butina distance cutoff.
    device : str
        Device for TabPFN inference.

    Returns
    -------
    DataFrame
        Fully scored and ranked candidates.
    """
    section("Candidate Pipeline")

    # 1. Load and dedup
    raw = pd.read_csv(all_evaluations_path)
    step(f"Loaded {len(raw):,} rows from {all_evaluations_path}")
    candidates = dedup_evaluations(raw)

    # 2. Activity filter
    candidates = filter_by_activity(candidates, p_active_min)

    # 3. Remove training actives
    training_smiles_list = sorted(load_training_actives(training_data_path))
    candidates = remove_training_actives(candidates, set(training_smiles_list))

    # 4. Similarity to training
    candidates = compute_max_tanimoto_to_training(candidates, training_smiles_list)

    # 5. Cluster
    candidates = tag_with_clusters(candidates, threshold=cluster_threshold)

    # 6. Cytotox
    if cytotox_models:
        candidates = score_cytotox(candidates, cytotox_models, device=device)

    # 7. AD
    candidates = score_ad(candidates, ad_model_path)

    # 8. Pre-retro composite rank
    higher = ["p_active"]
    lower = [c for c in candidates.columns if c.startswith("p_cytotox_")]
    weights = {"p_active": 4.0}
    for c in lower:
        weights[c] = 1.0
    candidates = rank_candidates(
        candidates, higher_better=higher, lower_better=lower, weights=weights
    )

    step(f"Pipeline complete: {len(candidates):,} candidates")
    return candidates


def run_full_report(
    all_evaluations_path: Path | str,
    output_dir: Path | str,
    run_name: str = "run",
    training_data_path: Path | str = "data/predictor/predictor_training_data.csv",
    ad_model_path: Path | str = "data/predictor/ad_model.joblib",
    cytotox_models: dict[str, Path | str] | None = None,
    p_active_min: float = 0.6,
    cluster_threshold: float = 0.65,
    retro_enabled: bool = False,
    retro_top_n: int = 3,
    retro_cap: int = 100,
    retro_time_limit: int = 120,
    retro_iteration_limit: int = 100,
    retro_max_transforms: int = 6,
    retro_n_jobs: int = 8,
    device: str = "cuda",
) -> tuple[pd.DataFrame, Path]:
    """Run pipeline + optional retrosynthesis + export CSV + HTML report.

    Parameters
    ----------
    all_evaluations_path : Path or str
        Path to ``all_evaluations.csv``.
    output_dir : Path or str
        Directory for outputs.
    run_name : str
        Name of the run (used in report title).
    training_data_path : Path or str
        Training data CSV.
    ad_model_path : Path or str
        AD model path.
    cytotox_models : dict or None
        Cytotox model paths.
    p_active_min : float
        P(active) threshold.
    cluster_threshold : float
        Butina threshold.
    retro_enabled : bool
        Whether to run retrosynthesis.
    retro_top_n : int
        Top-N per cluster for retro.
    retro_cap : int
        Max molecules for retro.
    retro_time_limit : int
        Seconds per molecule.
    retro_iteration_limit : int
        MCTS iterations.
    retro_max_transforms : int
        Tree depth.
    retro_n_jobs : int
        Parallel workers.
    device : str
        TabPFN device.

    Returns
    -------
    tuple of (DataFrame, Path)
        Candidates DataFrame and path to CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run core pipeline
    candidates = run_pipeline(
        all_evaluations_path,
        training_data_path=training_data_path,
        ad_model_path=ad_model_path,
        cytotox_models=cytotox_models,
        p_active_min=p_active_min,
        cluster_threshold=cluster_threshold,
        device=device,
    )

    # Optional retrosynthesis
    retro_images = {}
    if retro_enabled:
        candidates = run_retrosynthesis(
            candidates,
            top_n_per_cluster=retro_top_n,
            retro_cap=retro_cap,
            time_limit=retro_time_limit,
            iteration_limit=retro_iteration_limit,
            max_transforms=retro_max_transforms,
            n_jobs=retro_n_jobs,
        )

        # Re-rank with retro score
        higher = ["p_active"]
        if "retro_score" in candidates.columns:
            higher.append("retro_score")
        lower = [c for c in candidates.columns if c.startswith("p_cytotox_")]
        weights = {"p_active": 4.0, "retro_score": 3.0}
        for c in lower:
            weights[c] = 1.0
        candidates = rank_candidates(
            candidates, higher_better=higher, lower_better=lower, weights=weights
        )

        # Render route images for solved molecules
        solved = candidates.loc[
            candidates["retro_solved"] == True, "smiles"  # noqa: E712
        ].tolist()
        if solved:
            from reporting.console import console

            console.print(
                f"  Rendering route images for {len(solved)} solved molecules..."
            )
            retro_images = render_retro_routes(
                solved,
                time_limit=retro_time_limit,
                iteration_limit=retro_iteration_limit,
                max_transforms=retro_max_transforms,
            )

    # Export CSV
    csv_path = export_candidates_csv(candidates, output_dir / "candidates.csv")

    # Generate molecule SVGs for HTML report
    step("Generating molecule SVGs for report")
    all_smiles = set(candidates["smiles"].tolist())
    if "similar_training_smiles" in candidates.columns:
        all_smiles.update(
            s for s in candidates["similar_training_smiles"].dropna().tolist() if s
        )
    mol_images: dict[str, str] = {}
    for smi in all_smiles:
        svg = smiles_to_svg(smi)
        if svg is not None:
            mol_images[smi] = svg
    step(f"  Generated {len(mol_images)} molecule SVGs")

    # Export HTML report
    from reporting.html_report import generate_html_report

    generate_html_report(
        candidates,
        run_name=run_name,
        output_path=output_dir / "report.html",
        retro_images=retro_images,
        mol_images=mol_images,
    )

    return candidates, csv_path
