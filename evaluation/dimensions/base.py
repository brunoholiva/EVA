"""Base protocol for archive dimensions."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from evaluation.molecules import ParsedMolecules


class Dimension(Protocol):
    """Protocol for archive dimensions.
    
    Each dimension defines a behavior space axis for the CMA-MAE archive.
    Dimensions compute their values from parsed molecules.
    """
    
    name: str
    resolution: int
    range: tuple[float, float]
    
    def compute(self, parsed: ParsedMolecules) -> np.ndarray:
        """Compute this dimension for parsed molecules.
        
        Parameters
        ----------
        parsed : ParsedMolecules
            Parsed molecules with shared Mol objects and lazy fingerprints.
        
        Returns
        -------
        np.ndarray of shape ``(len(parsed.smiles),)``
            Dimension values. Invalid molecules receive NaN.
        """
        ...
