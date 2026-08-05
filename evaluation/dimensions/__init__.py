"""Dimension registry for plug-and-play archive dimensions."""

from __future__ import annotations

from evaluation.dimensions.ad import ADDimension
from evaluation.dimensions.base import Dimension
from evaluation.dimensions.fsp3 import Fsp3Dimension
from evaluation.dimensions.logp import LogPDimension
from evaluation.dimensions.mw import Mordimension
from evaluation.dimensions.num_rotb import NumRotBDimension
from evaluation.dimensions.tpsa import TPSADimension
from evaluation.dimensions.num_rings import NumRingDimension


__all__ = [
    "Dimension",
    "ADDimension",
    "LogPDimension",
    "TPSADimension",
    "Mordimension",
    "Fsp3Dimension",
    "NumRotBDimension",
    "NumRingDimension",
    "DIMENSION_REGISTRY",
]

# Registry of dimension classes (without dependencies)
DIMENSION_REGISTRY = {
    "logp": LogPDimension(),
    "tpsa": TPSADimension(),
    "mw": Mordimension(),
    "fsp3": Fsp3Dimension(),
    "num_rotb": NumRotBDimension(),
    "num_rings": NumRingDimension(),
}


def create_dimension(name: str, **kwargs) -> Dimension:
    """Create a dimension instance by name.
    
    Parameters
    ----------
    name : str
        Dimension name (e.g., "logp", "ad").
    **kwargs
        Additional arguments for dimension initialization (e.g., ad_scorer).
    
    Returns
    -------
    Dimension
        Dimension instance.
    
    Raises
    ------
    ValueError
        If dimension name is not registered or required arguments are missing.
    """
    if name == "ad":
        if "ad_scorer" not in kwargs:
            raise ValueError(
                "AD dimension requires 'ad_scorer' argument. "
                "Example: create_dimension('ad', ad_scorer=scorer)"
            )
        return ADDimension(**kwargs)
    if name not in DIMENSION_REGISTRY:
        raise ValueError(
            f"Unknown dimension '{name}'. "
            f"Available: {list(DIMENSION_REGISTRY.keys()) + ['ad']}"
        )
    return DIMENSION_REGISTRY[name]
