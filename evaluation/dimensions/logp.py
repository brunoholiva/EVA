"""LogP dimension."""

from __future__ import annotations

import numpy as np
from rdkit.Chem import Descriptors

from evaluation.molecules import ParsedMolecules


class LogPDimension:
    """Lipophilicity (Wildman-Crippen LogP)."""
    
    name = "logp"
    resolution = 20
    range = (-1.0, 6.0)
    
    def compute(self, parsed: ParsedMolecules) -> np.ndarray:
        """Compute LogP for parsed molecules."""
        return np.array([
            Descriptors.MolLogP(mol) if mol is not None else float("nan")
            for mol in parsed.mols
        ])
