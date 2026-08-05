"""NumRotB dimension."""

from __future__ import annotations

import numpy as np
from rdkit.Chem import Descriptors

from evaluation.molecules import ParsedMolecules


class NumRotBDimension:
    """Number of rotatable bonds."""
    
    name = "num_rotb"
    resolution = 20
    range = (0.0, 20.0)
    
    def compute(self, parsed: ParsedMolecules) -> np.ndarray:
        """Compute NumRotB for parsed molecules."""
        return np.array([
            Descriptors.NumRotatableBonds(mol) if mol is not None else float("nan")
            for mol in parsed.mols
        ])
