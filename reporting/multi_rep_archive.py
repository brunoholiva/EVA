"""Build a multi-representative archive from a standard archive CSV.

Map-Elites keeps one elite per cell. When a cell contains multiple chemotypes,
this module selects up to k diverse, high-scoring representatives per cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ribs.archives import GridArchive

from chemistry.csk import cluster_by_csk


def compute_cell_indices(
    measures: np.ndarray,
    ranges: list[list[float]],
    resolutions: list[int],
    seed: int = 0,
) -> np.ndarray:
    """Compute flat GridArchive cell indices for measure vectors.

    Parameters
    ----------
    measures : np.ndarray of shape (n, d)
        Measure vectors.
    ranges : list of [min, max] pairs
        Value range for each measure dimension.
    resolutions : list of int
        Number of cells per dimension.
    seed : int
        Seed for the temporary GridArchive.

    Returns
    -------
    np.ndarray of int
        Flat cell index for each measure vector.
    """
    archive = GridArchive(
        solution_dim=1,
        dims=resolutions,
        ranges=ranges,
        learning_rate=1.0,
        threshold_min=0.0,
        seed=seed,
    )
    return archive.index_of(measures)


def cluster_smiles(
    smiles_list: list[str],
    **_kwargs,
) -> list[set[int]]:
    """Cluster SMILES by CSK structural hash.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings.

    Returns
    -------
    list of set[int]
        Each set contains the indices of SMILES sharing the same
        cyclic skeleton (CSK) hash.
    """
    return cluster_by_csk(smiles_list)


def select_top_k_per_cell(
    df: pd.DataFrame,
    dim_cols: list[str],
    ranges: list[list[float]],
    resolutions: list[int],
    k: int = 5,
    score_col: str = "p_active_real",
    **_kwargs,
) -> pd.DataFrame:
    """Select up to k diverse, high-scoring representatives per archive cell.

    Parameters
    ----------
    df : pd.DataFrame
        Archive DataFrame with SMILES, scores, and dimension columns.
    dim_cols : list of str
        Archive dimension column names.
    ranges : list of [min, max]
        Value range for each dimension.
    resolutions : list of int
        Number of cells per dimension.
    k : int
        Maximum representatives per cell.
    score_col : str
        Column used to rank representatives. Falls back to 'p_active' if missing.

    Returns
    -------
    pd.DataFrame
        Expanded archive with up to k rows per cell.
    """
    if score_col not in df.columns:
        score_col = "p_active"

    working = df.copy()
    measures = working[dim_cols].to_numpy(dtype=float)
    working["_cell_index"] = compute_cell_indices(measures, ranges, resolutions)

    representative_indices: list[int] = []
    for _cell_index, group in working.groupby("_cell_index"):
        smiles = group["smiles"].tolist()
        clusters = cluster_smiles(smiles)

        cluster_best: list[int] = []
        for cluster in clusters:
            cluster_df_indices = [group.index[i] for i in cluster]
            cluster_scores = working.loc[cluster_df_indices, score_col]
            best_idx = cluster_scores.idxmax()
            cluster_best.append(best_idx)

        best_scores = working.loc[cluster_best, score_col]
        top_k = best_scores.sort_values(ascending=False).head(k).index.tolist()
        representative_indices.extend(top_k)

    return working.loc[representative_indices].drop(columns=["_cell_index"])
