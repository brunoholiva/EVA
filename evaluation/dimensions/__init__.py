"""Data-driven archive dimensions.

Each dimension is an RDKit descriptor function applied to each parsed
molecule. ``resolution``/``range`` live in ``config.toml``; the registry
here only maps names to compute functions.

Special dimensions computed elsewhere:

* ``max_tanimoto`` — max Tanimoto similarity to training actives,
  computed in the evaluator via :class:`evaluation.applicability.ADScorer`.
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


DESCRIPTORS = {
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

SPECIAL_DIMENSIONS = {"max_tanimoto"}


def compute_dimension(
    name: str,
    parsed: ParsedMolecules,
) -> np.ndarray:
    """Compute a dimension's values for parsed molecules.

    Only handles RDKit descriptor dimensions. Special dimensions
    (``max_tanimoto``) is computed in the evaluator.

    Parameters
    ----------
    name : str
        Dimension name (e.g. ``"logp"``, ``"tpsa"``).
    parsed : ParsedMolecules
        Parsed molecules with shared Mol objects and lazy fingerprints.

    Returns
    -------
    np.ndarray of shape ``(len(parsed.smiles),)``
        Dimension values. Invalid molecules receive NaN.

    Raises
    ------
    ValueError
        If *name* is not a known RDKit descriptor.
    """
    if name not in DESCRIPTORS:
        raise ValueError(
            f"Unknown dimension '{name}'. "
            f"RDKit descriptors: {sorted(DESCRIPTORS)}. "
            f"Special (evaluator): {sorted(SPECIAL_DIMENSIONS)}"
        )
    return _descriptor_values(parsed, DESCRIPTORS[name])
