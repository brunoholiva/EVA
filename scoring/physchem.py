"""Physicochemical property scoring (LogP, TPSA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Descriptors


@dataclass
class PhysChemResult:
    """Container for physicochemical property arrays."""

    logp: np.ndarray
    tpsa: np.ndarray


def _compute_single(smi: str) -> tuple[float, float]:
    """Compute LogP and TPSA for a single SMILES. Returns (logp, tpsa)."""
    if not smi or not smi.strip():
        return float("nan"), float("nan")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return float("nan"), float("nan")
    return Descriptors.MolLogP(mol), Descriptors.TPSA(mol)


class PhysChemScorer:
    """Compute LogP and TPSA for batches of SMILES.

    Parameters
    ----------
    logp_range : tuple[float, float]
        Expected (min, max) for LogP values.
    tpsa_range : tuple[float, float]
        Expected (min, max) for TPSA values.
    """

    def __init__(
        self,
        logp_range: tuple[float, float] = (-2.0, 8.0),
        tpsa_range: tuple[float, float] = (0.0, 250.0),
    ) -> None:
        self._logp_range = logp_range
        self._tpsa_range = tpsa_range

    def __call__(
        self, smiles: list[str], n_jobs: int = -1
    ) -> PhysChemResult:
        """Compute physicochemical properties for a list of SMILES.

        Parameters
        ----------
        smiles : list of str
            SMILES strings to score.
        n_jobs : int, default=-1
            Number of parallel workers. -1 uses all CPUs.

        Returns
        -------
        PhysChemResult
            Arrays of LogP and TPSA values. Invalid SMILES receive NaN.
        """
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_compute_single)(smi) for smi in smiles
        )
        logp = np.array([r[0] for r in results], dtype=np.float64)
        tpsa = np.array([r[1] for r in results], dtype=np.float64)
        return PhysChemResult(logp=logp, tpsa=tpsa)
