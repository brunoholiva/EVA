"""Fsp3 dimension."""

from __future__ import annotations

import numpy as np
from rdkit.Chem import GraphDescriptors

from evaluation.molecules import ParsedMolecules


class BalabanJDimension:
    """Balaban J index dimension."""
    
    name = "balabanj"
    resolution = 15
    range = (1.0, 7.0)
    
    def compute(self, parsed: ParsedMolecules) -> np.ndarray:
        """Compute Balaban J index for parsed molecules."""
        return np.array([
            GraphDescriptors.BalabanJ(mol) if mol is not None else float("nan")
            for mol in parsed.mols
        ])
