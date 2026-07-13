"""Scoring functions for molecular properties: BR-SAScore and novelty."""

from __future__ import annotations

import warnings

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
