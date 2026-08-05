"""Evaluation domain: scoring pipeline for molecules."""

from __future__ import annotations

from evaluation.activity import load_model, predict_from_features
from evaluation.applicability import ADScorer
from evaluation.behavior import MolBehavior, batch_molecule_behaviors
from evaluation.dimensions import (
    DIMENSION_REGISTRY,
    DimensionContext,
    DimensionSpec,
    compute_dimensions,
    get_dimension_spec,
)

__all__ = [
    "ADScorer",
    "DIMENSION_REGISTRY",
    "DimensionContext",
    "DimensionSpec",
    "MolBehavior",
    "batch_molecule_behaviors",
    "compute_dimensions",
    "get_dimension_spec",
    "load_model",
    "predict_from_features",
]
