"""AD dimension."""

from __future__ import annotations

import numpy as np

from evaluation.applicability import ADScorer
from evaluation.molecules import ParsedMolecules


class ADDimension:
    """Applicability Domain distance.
    
    Computes mean Tanimoto distance to k-nearest neighbors in the
    training set. Requires fingerprints, which are lazily computed.
    """
    
    name = "ad"
    resolution = 15
    range = (0.0, 0.8)
    
    def __init__(self, ad_scorer: ADScorer):
        """Initialize with AD scorer.
        
        Parameters
        ----------
        ad_scorer : ADScorer
            Applicability domain scorer with pre-loaded training data.
        """
        self._ad_scorer = ad_scorer
    
    def compute(self, parsed: ParsedMolecules) -> np.ndarray:
        """Compute AD distance for parsed molecules."""
        fps, valid_mask = parsed.fingerprints
        
        ad = np.ones(len(parsed.smiles), dtype=np.float32)
        
        if fps is not None and len(fps) > 0:
            ad_dist = self._ad_scorer.compute_from_fps(fps)
            ad[valid_mask] = ad_dist
        
        return ad
