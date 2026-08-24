"""Build a multi-representative archive from a standard archive CSV.

Map-Elites keeps one elite per cell. When a cell contains multiple chemotypes,
this module selects up to k diverse, high-scoring representatives per cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Cluster import Butina
from ribs.archives import GridArchive


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
    threshold: float = 0.75,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[set[int]]:
    """Cluster SMILES by Tanimoto distance.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings.
    threshold : float
        Tanimoto distance cutoff.
    radius : int
        Morgan fingerprint radius.
    n_bits : int
        Morgan fingerprint bit length.

    Returns
    -------
    list of set[int]
        Each set contains the indices of SMILES in one cluster.
    """
    if not smiles_list:
        return []

    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(gen.GetFingerprint(mol) if mol else None)

    dists = []
    n = len(fps)
    for i in range(1, n):
        for j in range(i):
            if fps[i] is not None and fps[j] is not None:
                dists.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            else:
                dists.append(1.0)

    clusters = Butina.ClusterData(
        dists,
        nPts=n,
        distThresh=threshold,
        isDistData=True,
    )
    return [set(c) for c in clusters]


def select_top_k_per_cell(
    df: pd.DataFrame,
    dim_cols: list[str],
    ranges: list[list[float]],
    resolutions: list[int],
    k: int = 5,
    score_col: str = "p_active_real",
    cluster_threshold: float = 0.75,
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
    cluster_threshold : float
        Tanimoto distance cutoff for within-cell clustering.

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
        clusters = cluster_smiles(smiles, threshold=cluster_threshold)

        # Pick highest-scoring member of each cluster.
        cluster_best: list[int] = []
        for cluster in clusters:
            cluster_df_indices = [group.index[i] for i in cluster]
            cluster_scores = working.loc[cluster_df_indices, score_col]
            best_idx = cluster_scores.idxmax()
            cluster_best.append(best_idx)

        # Keep top-k by score.
        best_scores = working.loc[cluster_best, score_col]
        top_k = best_scores.sort_values(ascending=False).head(k).index.tolist()
        representative_indices.extend(top_k)

    return working.loc[representative_indices].drop(columns=["_cell_index"])
