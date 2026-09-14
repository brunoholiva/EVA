"""Data-driven archive dimensions.

Each non-AD dimension is an RDKit descriptor function applied to each
parsed molecule. ``resolution``/``range`` live in ``config.toml``; the
registry here only maps names to compute functions.
"""

from __future__ import annotations

import numpy as np
from rdkit.Chem import Descriptors, GraphDescriptors

from evaluation.applicability import ADScorer
from evaluation.molecules import ParsedMolecules


def _descriptor_values(parsed: ParsedMolecules, fn) -> np.ndarray:
    """Apply an RDKit descriptor function to every parsed molecule."""
    return np.array(
        [fn(mol) if mol is not None else float("nan") for mol in parsed.mols]
    )


_DESCRIPTORS = {
    "logp": Descriptors.MolLogP,
    "tpsa": Descriptors.TPSA,
    "mw": Descriptors.MolWt,
    "fsp3": Descriptors.FractionCSP3,
    "num_rotb": Descriptors.NumRotatableBonds,
    "num_rings": Descriptors.RingCount,
    "balabanj": GraphDescriptors.BalabanJ,
    "vsa_estate2": Descriptors.VSA_EState2,
    "bcut2d_logplow": Descriptors.BCUT2D_LOGPLOW,
}


def compute_dimension(
    name: str,
    parsed: ParsedMolecules,
    ad_scorer: ADScorer | None = None,
) -> np.ndarray:
    """Compute a dimension's values for parsed molecules.

    Parameters
    ----------
    name : str
        Dimension name (e.g. ``"logp"``, ``"ad"``).
    parsed : ParsedMolecules
        Parsed molecules with shared Mol objects and lazy fingerprints.
    ad_scorer : ADScorer or None
        Required for the ``"ad"`` dimension.

    Returns
    -------
    np.ndarray of shape ``(len(parsed.smiles),)``
        Dimension values. Invalid molecules receive NaN (or 1.0 for AD).
    """
    if name == "ad":
        if ad_scorer is None:
            raise ValueError("AD dimension requires an 'ad_scorer' argument")
        fps, valid_mask = parsed.fingerprints
        ad = np.ones(len(parsed.smiles), dtype=np.float32)
        if fps is not None and len(fps) > 0:
            ad[valid_mask] = ad_scorer.compute_from_fps(fps)
        return ad
    if name not in _DESCRIPTORS:
        raise ValueError(
            f"Unknown dimension '{name}'. Available: {sorted(_DESCRIPTORS) + ['ad']}"
        )
    return _descriptor_values(parsed, _DESCRIPTORS[name])
