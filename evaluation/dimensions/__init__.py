"""Data-driven archive dimensions.

Each dimension is a name mapped to an RDKit descriptor function that returns
one scalar per molecule. Invalid molecules receive NaN.
"""

from __future__ import annotations

import numpy as np
from rdkit.Chem import Descriptors, GraphDescriptors

from evaluation.molecules import ParsedMolecules


def _descriptor_values(parsed: ParsedMolecules, fn) -> np.ndarray:
    """Apply an RDKit descriptor function to every parsed molecule."""
    return np.array(
        [fn(mol) if mol is not None else float("nan") for mol in parsed.mols]
    )


DIMENSIONS = {
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


def compute_dimension(name: str, parsed: ParsedMolecules) -> np.ndarray:
    """Compute a dimension's values for parsed molecules.

    Parameters
    ----------
    name : str
        Dimension name (e.g. ``"logp"``, ``"tpsa"``).
    parsed : ParsedMolecules
        Parsed molecules with shared Mol objects.

    Returns
    -------
    np.ndarray of shape ``(len(parsed.smiles),)``
        Dimension values. Invalid molecules receive NaN.

    Raises
    ------
    ValueError
        If *name* is not a known dimension.
    """
    if name not in DIMENSIONS:
        raise ValueError(
            f"Unknown dimension '{name}'. "
            f"Available dimensions: {sorted(DIMENSIONS)}"
        )
    return _descriptor_values(parsed, DIMENSIONS[name])
