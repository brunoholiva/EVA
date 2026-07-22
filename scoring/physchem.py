"""Physicochemical property scoring (LogP, TPSA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


@dataclass
class PhysChemResult:
    """Container for physicochemical property arrays."""

    logp: np.ndarray
    tpsa: np.ndarray


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

    def __call__(self, smiles: list[str]) -> PhysChemResult:
        """Compute physicochemical properties for a list of SMILES.

        Parameters
        ----------
        smiles : list of str
            SMILES strings to score.

        Returns
        -------
        PhysChemResult
            Arrays of LogP and TPSA values. Invalid SMILES receive NaN.
        """
        n = len(smiles)
        logp = np.full(n, np.nan, dtype=np.float64)
        tpsa = np.full(n, np.nan, dtype=np.float64)

        for i, smi in enumerate(smiles):
            if not smi or not smi.strip():
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            logp[i] = Descriptors.MolLogP(mol)
            tpsa[i] = Descriptors.TPSA(mol)

        return PhysChemResult(logp=logp, tpsa=tpsa)
