"""BCUT2D_LOGPLOW dimension."""

from __future__ import annotations

import numpy as np

from evaluation.molecules import ParsedMolecules
from rdkit.Chem import Descriptors

class BCUT2D_LOGPLOWDimension:
    """BCUT2D_LOGPLOW dimension."""
    
    name = "bcut2d_logplow"
    resolution = 15
    range = (-3.2, -1.7)
    
    def compute(self, parsed: ParsedMolecules) -> np.ndarray:
        """Compute BCUT2D_LOGPLOW for parsed molecules."""
        return np.array([
            Descriptors.BCUT2D_LOGPLOW(mol) if mol is not None else float("nan")
            for mol in parsed.mols
        ])
