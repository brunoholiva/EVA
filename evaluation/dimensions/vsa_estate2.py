"""VSA EState 2 dimension."""

from __future__ import annotations

import numpy as np

from evaluation.molecules import ParsedMolecules
from rdkit.Chem import Descriptors

class VSAEState2Dimension:
    """VSA EState 2 dimension."""
    
    name = "vsa_estate2"
    resolution = 15
    range = (0.0, 60.0)
    
    def compute(self, parsed: ParsedMolecules) -> np.ndarray:
        """Compute VSA EState 2 for parsed molecules."""
        return np.array([
            Descriptors.VSA_EState2(mol) if mol is not None else float("nan")
            for mol in parsed.mols
        ])
