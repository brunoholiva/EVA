"""Tests for representation-specific activity features."""

from __future__ import annotations

import numpy as np
import pytest

from chemistry.featurizers import RDKit2DFeaturizer, make_activity_featurizer


def test_rdkit2d_shape_and_values() -> None:
    features = RDKit2DFeaturizer().transform(["CCO", "c1ccccc1"])

    assert features.shape == (2, 200)
    assert np.isfinite(features).all()


def test_rdkit2d_rejects_invalid_smiles() -> None:
    with pytest.raises(ValueError, match="position 0"):
        RDKit2DFeaturizer().transform(["CCO", "not-smiles"])


def test_activity_featurizer_registry() -> None:
    assert isinstance(make_activity_featurizer("rdkit2d"), RDKit2DFeaturizer)
    with pytest.raises(ValueError, match="chemeleon"):
        make_activity_featurizer("unknown")
