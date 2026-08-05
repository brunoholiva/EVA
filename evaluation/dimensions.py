"""Dimension registry for plug-and-play archive dimensions.

Each dimension is defined with a compute function that takes valid SMILES
and returns an array of values. Adding a new dimension is as simple as
registering a new entry here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from evaluation.applicability import ADScorer


@dataclass
class DimensionSpec:
    """Specification for a single archive dimension.

    Attributes
    ----------
    name : str
        Dimension name (must match config.toml).
    resolution : int
        Grid resolution for this dimension.
    range : tuple[float, float]
        Value range for this dimension.
    compute_fn : callable
        Function ``(valid_smiles, context) -> np.ndarray`` that computes
        dimension values for valid SMILES.
    """

    name: str
    resolution: int
    range: tuple[float, float]
    compute_fn: Callable[[list[str], "DimensionContext"], np.ndarray]


@dataclass
class DimensionContext:
    """Context passed to dimension compute functions.

    Contains shared resources needed by dimension computations.

    Attributes
    ----------
    ad_scorer : ADScorer or None
        Applicability domain scorer (for 'ad' dimension).
    """

    ad_scorer: ADScorer | None = None


def _compute_ad(valid_smiles: list[str], ctx: DimensionContext) -> np.ndarray:
    """Compute applicability domain distance."""
    from evaluation.behavior import batch_molecule_behaviors

    _, _, _, _, fps, valid_fp_mask = batch_molecule_behaviors(valid_smiles)
    ad = np.ones(len(valid_smiles), dtype=np.float32)
    if fps is not None and len(fps) > 0 and ctx.ad_scorer is not None:
        ad_dist = ctx.ad_scorer.compute_from_fps(fps)
        ad[valid_fp_mask] = ad_dist
    return ad


def _compute_logp(valid_smiles: list[str], ctx: DimensionContext) -> np.ndarray:
    """Compute LogP."""
    from evaluation.behavior import batch_molecule_behaviors

    logp, _, _, _, _, _ = batch_molecule_behaviors(valid_smiles)
    return logp


def _compute_tpsa(valid_smiles: list[str], ctx: DimensionContext) -> np.ndarray:
    """Compute TPSA."""
    from evaluation.behavior import batch_molecule_behaviors

    _, tpsa, _, _, _, _ = batch_molecule_behaviors(valid_smiles)
    return tpsa


def _compute_mw(valid_smiles: list[str], ctx: DimensionContext) -> np.ndarray:
    """Compute molecular weight."""
    from evaluation.behavior import batch_molecule_behaviors

    _, _, mw, _, _, _ = batch_molecule_behaviors(valid_smiles)
    return mw


def _compute_fsp3(valid_smiles: list[str], ctx: DimensionContext) -> np.ndarray:
    """Compute fraction of sp3 carbons."""
    from evaluation.behavior import batch_molecule_behaviors

    _, _, _, fsp3, _, _ = batch_molecule_behaviors(valid_smiles)
    return fsp3


DIMENSION_REGISTRY: dict[str, DimensionSpec] = {
    "ad": DimensionSpec(
        name="ad",
        resolution=15,
        range=(0.0, 0.8),
        compute_fn=_compute_ad,
    ),
    "logp": DimensionSpec(
        name="logp",
        resolution=20,
        range=(-1.0, 6.0),
        compute_fn=_compute_logp,
    ),
    "tpsa": DimensionSpec(
        name="tpsa",
        resolution=10,
        range=(0.0, 200.0),
        compute_fn=_compute_tpsa,
    ),
    "mw": DimensionSpec(
        name="mw",
        resolution=30,
        range=(50.0, 600.0),
        compute_fn=_compute_mw,
    ),
    "fsp3": DimensionSpec(
        name="fsp3",
        resolution=30,
        range=(0.0, 1.0),
        compute_fn=_compute_fsp3,
    ),
}


def get_dimension_spec(name: str) -> DimensionSpec:
    """Get dimension specification by name.

    Parameters
    ----------
    name : str
        Dimension name.

    Returns
    -------
    DimensionSpec
        Dimension specification.

    Raises
    ------
    ValueError
        If dimension name is not registered.
    """
    if name not in DIMENSION_REGISTRY:
        raise ValueError(
            f"Unknown dimension '{name}'. "
            f"Available: {list(DIMENSION_REGISTRY.keys())}"
        )
    return DIMENSION_REGISTRY[name]


def compute_dimensions(
    valid_smiles: list[str],
    dimension_names: list[str],
    ctx: DimensionContext,
) -> dict[str, np.ndarray]:
    """Compute all requested dimensions for valid SMILES.

    Parameters
    ----------
    valid_smiles : list of str
        Chemically valid SMILES strings.
    dimension_names : list of str
        Names of dimensions to compute.
    ctx : DimensionContext
        Context with shared resources.

    Returns
    -------
    dict of str to np.ndarray
        Dictionary mapping dimension names to computed values.
    """
    results = {}
    for name in dimension_names:
        spec = get_dimension_spec(name)
        results[name] = spec.compute_fn(valid_smiles, ctx)
    return results
