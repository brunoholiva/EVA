"""Data-driven archive dimensions.

Each dimension is registered here. RDKit descriptor dimensions operate on
parsed molecules; ``max_tanimoto`` additionally uses the optional reference
scorer and lazily requests fingerprints from the parsed molecules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rdkit.Chem import Descriptors, GraphDescriptors

from evaluation.molecules import ParsedMolecules
from evaluation.max_tanimoto import MaxTanimotoScorer

if TYPE_CHECKING:
    from config import MaxTanimotoConfig


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


@dataclass(frozen=True)
class DimensionResources:
    """Resources loaded by dimensions that need external reference data."""

    max_tanimoto: MaxTanimotoScorer | None = None


def load_dimension_resources(
    dimension_names: list[str],
    max_tanimoto_config: MaxTanimotoConfig | None = None,
) -> DimensionResources:
    """Load resources required by the configured archive dimensions.

    Parameters
    ----------
    dimension_names : list[str]
        Names of the configured archive dimensions.
    max_tanimoto_config : MaxTanimotoConfig or None
        Configuration for the max-Tanimoto reference model.

    Returns
    -------
    DimensionResources
        Loaded resources for the configured dimensions.

    Raises
    ------
    ValueError
        If max-Tanimoto is configured without its reference model settings.
    """
    if "max_tanimoto" not in dimension_names:
        return DimensionResources()
    if max_tanimoto_config is None:
        raise ValueError(
            "max_tanimoto is configured as an archive dimension, "
            "but [max_tanimoto] is missing"
        )
    return DimensionResources(
        max_tanimoto=MaxTanimotoScorer(max_tanimoto_config.model_path)
    )


def _descriptor_dimension(fn):
    return lambda parsed, _scorer: _descriptor_values(parsed, fn)


def _max_tanimoto_dimension(
    parsed: ParsedMolecules, resources: DimensionResources | None
) -> np.ndarray:
    if resources is None or resources.max_tanimoto is None:
        raise ValueError("max_tanimoto requires a [max_tanimoto] reference model")

    fps, valid_mask = parsed.fingerprints
    values = np.ones(len(parsed.smiles), dtype=np.float32)
    if fps is not None and len(fps) > 0:
        values[valid_mask] = resources.max_tanimoto.compute(fps)
    return values


DIMENSIONS = {
    **{name: _descriptor_dimension(fn) for name, fn in _DESCRIPTORS.items()},
    "max_tanimoto": _max_tanimoto_dimension,
}


def compute_dimension(
    name: str,
    parsed: ParsedMolecules,
    resources: DimensionResources | None = None,
) -> np.ndarray:
    """Compute a dimension's values for parsed molecules.

    Parameters
    ----------
    name : str
        Dimension name (e.g. ``"logp"``, ``"tpsa"``).
    parsed : ParsedMolecules
        Parsed molecules with shared Mol objects and lazy fingerprints.
    resources : DimensionResources or None
        Resources required by dimensions with external reference data.

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
    return DIMENSIONS[name](parsed, resources)
