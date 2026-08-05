"""TPSA dimension."""

from __future__ import annotations

import numpy as np
from rdkit.Chem import Descriptors

from evaluation.molecules import ParsedMolecules


class TPSADimension:
    """Topological Polar Surface Area."""
    
    name = "tpsa"
    resolution = 10
    range = (0.0, 200.0)
    
    def compute(self, parsed: ParsedMolecules) -> np.ndarray:
        """Compute TPSA for parsed molecules."""
        return np.array([
            Descriptors.TPSA(mol) if mol is not None else float("nan")
            for mol in parsed.mols
        ])
