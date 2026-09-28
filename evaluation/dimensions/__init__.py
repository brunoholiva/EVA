"""Data-driven archive dimensions.

Each dimension is registered here. RDKit descriptor dimensions operate on
parsed molecules; ``max_tanimoto`` additionally uses the optional reference
scorer and lazily requests fingerprints from the parsed molecules.
"""

from __future__ import annotations

import numpy as np
from rdkit.Chem import Descriptors, GraphDescriptors

from evaluation.molecules import ParsedMolecules
from evaluation.max_tanimoto import MaxTanimotoScorer


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


def _descriptor_dimension(fn):
    return lambda parsed, _scorer: _descriptor_values(parsed, fn)


def _max_tanimoto_dimension(
    parsed: ParsedMolecules, scorer: MaxTanimotoScorer | None
) -> np.ndarray:
    if scorer is None:
        raise ValueError("max_tanimoto requires a [max_tanimoto] reference model")

    fps, valid_mask = parsed.fingerprints
    values = np.ones(len(parsed.smiles), dtype=np.float32)
    if fps is not None and len(fps) > 0:
        values[valid_mask] = scorer.compute(fps)
    return values


DIMENSIONS = {
    **{name: _descriptor_dimension(fn) for name, fn in _DESCRIPTORS.items()},
    "max_tanimoto": _max_tanimoto_dimension,
}


def compute_dimension(
    name: str,
    parsed: ParsedMolecules,
    scorer: MaxTanimotoScorer | None = None,
) -> np.ndarray:
    """Compute a dimension's values for parsed molecules.

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
    if name not in DIMENSIONS:
        raise ValueError(
            f"Unknown dimension '{name}'. "
            f"Available dimensions: {sorted(DIMENSIONS)}"
        )
    return DIMENSIONS[name](parsed, scorer)
