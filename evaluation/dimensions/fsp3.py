"""Fsp3 dimension."""

from __future__ import annotations

import numpy as np
from rdkit.Chem import Descriptors

from evaluation.molecules import ParsedMolecules


class Fsp3Dimension:
    """Fraction of sp3 carbons."""
    
    name = "fsp3"
    resolution = 30
    range = (0.0, 1.0)
    
    def compute(self, parsed: ParsedMolecules) -> np.ndarray:
        """Compute Fsp3 for parsed molecules."""
        return np.array([
            Descriptors.FractionCSP3(mol) if mol is not None else float("nan")
            for mol in parsed.mols
        ])
