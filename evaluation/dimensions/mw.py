"""MW dimension."""

from __future__ import annotations

import numpy as np
from rdkit.Chem import Descriptors

from evaluation.molecules import ParsedMolecules


class Mordimension:
    """Molecular Weight."""
    
    name = "mw"
    resolution = 30
    range = (50.0, 600.0)
    
    def compute(self, parsed: ParsedMolecules) -> np.ndarray:
        """Compute MW for parsed molecules."""
        return np.array([
            Descriptors.MolWt(mol) if mol is not None else float("nan")
            for mol in parsed.mols
        ])
