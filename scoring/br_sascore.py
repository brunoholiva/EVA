"""BR-SAScore: retrosynthetic accessibility scoring."""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem

_scorer = None


def _get_scorer():
    """Return a cached SAScorer instance."""
    global _scorer
    if _scorer is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            from BRSAScore import SAScorer

            _scorer = SAScorer()
    return _scorer


def compute_br_sascore(smiles: str) -> float:
    """Compute the BR-SAScore for a given molecule.

    BR-SAScore is a retrosynthetic accessibility score based on reaction
    and building-block data.  Lower scores indicate easier synthesis.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.

    Returns
    -------
    float
        BR-SAScore.  Returns ``nan`` if the SMILES cannot be parsed.
    """
    if not smiles or not smiles.strip():
        return float("nan")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")

    try:
        score, _contribution = _get_scorer().calculateScore(smiles)
        return score
    except Exception:
        return float("nan")


def compute_br_sascore_batch(smiles: list[str], n_jobs: int = -1) -> np.ndarray:
    """Compute BR-SAScore for a batch of SMILES in parallel.

    Parameters
    ----------
    smiles : list of str
        SMILES strings to score.
    n_jobs : int, default=-1
        Number of parallel workers. -1 uses all CPUs.

    Returns
    -------
    np.ndarray of shape ``(len(smiles),)``
        BR-SAScore for each molecule. Invalid SMILES receive NaN.
    """
    scores = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(compute_br_sascore)(s) for s in smiles
    )
    return np.array(scores, dtype=np.float64)
