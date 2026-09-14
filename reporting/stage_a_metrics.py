"""Metrics for Stage A comparison.

Focused on quality and structural diversity of high-scoring molecules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def _compute_scaffold(smi: str) -> str | None:
    """Return Murcko scaffold SMILES, or None on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def _compute_morgan_fps(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> list:
    """Compute Morgan fingerprints for valid SMILES."""
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(gen.GetFingerprint(mol) if mol else None)
    return fps


def _pairwise_tanimoto_matrix(fps: list) -> np.ndarray:
    """Compute pairwise Tanimoto similarity matrix from fingerprints."""
    n = len(fps)
    valid = [fp is not None for fp in fps]
    sim_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if valid[i] and valid[j]:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            else:
                sim = 0.0
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    return sim_matrix


def compute_dhq(
    smiles: list[str],
    scores: np.ndarray,
    score_threshold: float = 0.6,
    butina_threshold: float = 0.75,
    radius: int = 2,
    n_bits: int = 2048,
) -> int:
    """Compute diverse high-quality hit count.

    Counts distinct Butina clusters among molecules with score > threshold.

    Parameters
    ----------
    smiles : list of str
        SMILES strings.
    scores : np.ndarray
        Score for each molecule (e.g., P(active)).
    score_threshold : float
        Minimum score to include a molecule.
    butina_threshold : float
        Tanimoto distance cutoff for Butina clustering.
    radius : int
        Morgan fingerprint radius.
    n_bits : int
        Morgan fingerprint bit length.

    Returns
    -------
    int
        Number of distinct chemotype clusters.
    """
    from reporting.multi_rep_archive import cluster_smiles

    mask = scores > score_threshold
    selected_smiles = [s for s, m in zip(smiles, mask) if m]
    if not selected_smiles:
        return 0

    clusters = cluster_smiles(
        selected_smiles, threshold=butina_threshold, radius=radius, n_bits=n_bits
    )
    return len(clusters)


def compute_metrics(
    df: pd.DataFrame,
    score_col: str = "p_active_real",
    score_threshold: float = 0.6,
    butina_threshold: float = 0.75,
) -> dict:
    """Compute Stage A comparison metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Archive DataFrame with 'smiles' and a score column.
    score_col : str
        Column to use for scores. Falls back to 'p_active' if missing.
    score_threshold : float
        Threshold for high-quality molecules.
    butina_threshold : float
        Tanimoto distance cutoff for diversity clustering.

    Returns
    -------
    dict
        Dictionary of metrics.
    """
    if score_col not in df.columns:
        score_col = "p_active"

    smiles = df["smiles"].tolist()
    scores = df[score_col].to_numpy(dtype=float)

    high_mask = scores > score_threshold
    high_smiles = [s for s, m in zip(smiles, high_mask) if m]

    dhq = compute_dhq(
        smiles,
        scores,
        score_threshold=score_threshold,
        butina_threshold=butina_threshold,
    )

    # Scaffold diversity.
    scaffolds = [_compute_scaffold(s) for s in high_smiles]
    scaffolds = [s for s in scaffolds if s is not None]
    if scaffolds:
        scaffold_counts = pd.Series(scaffolds).value_counts()
        unique_scaffolds = int(scaffold_counts.shape[0])
        largest_scaffold_fraction = float(scaffold_counts.iloc[0] / len(scaffolds))
    else:
        unique_scaffolds = 0
        largest_scaffold_fraction = 0.0

    # Nearest-neighbor Tanimoto among high-scoring molecules.
    if len(high_smiles) > 1:
        fps = _compute_morgan_fps(high_smiles)
        sim_matrix = _pairwise_tanimoto_matrix(fps)
        np.fill_diagonal(sim_matrix, 0.0)
        nn_sims = sim_matrix.max(axis=1)
        mean_nn = float(nn_sims.mean())
    else:
        mean_nn = 0.0

    return {
        "n_total": len(df),
        "n_gt_threshold": int(high_mask.sum()),
        "max_p_active": float(scores.max()),
        "mean_p_active_top100": (
            float(np.sort(scores)[-100:].mean())
            if len(scores) >= 100
            else float(scores.mean())
        ),
        "dhq": dhq,
        "unique_scaffolds_gt_threshold": unique_scaffolds,
        "largest_scaffold_fraction": largest_scaffold_fraction,
        "mean_nn_tanimoto_gt_threshold": mean_nn,
    }
