"""NumRing dimension."""

from __future__ import annotations

import numpy as np
from rdkit.Chem import Lipinski

from evaluation.molecules import ParsedMolecules


class NumRingDimension:
    """Number of Rings."""
    
    name = "num_rings"
    resolution = 30
    range = (0.0, 10.0)
    
    def compute(self, parsed: ParsedMolecules) -> np.ndarray:
        """Compute number of rings for parsed molecules."""
        return np.array([
            Lipinski.RingCount(mol) if mol is not None else float("nan")
            for mol in parsed.mols
        ])
