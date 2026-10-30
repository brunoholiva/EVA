"""Tests for MiniMol activity featurization."""

from __future__ import annotations

import numpy as np
import pytest

from chemistry.featurizers import MinimolFeaturizer, make_activity_featurizer


def test_activity_featurizer_registry() -> None:
    assert isinstance(make_activity_featurizer("minimol"), MinimolFeaturizer)
    assert isinstance(make_activity_featurizer(), MinimolFeaturizer)
    with pytest.raises(ValueError, match="minimol"):
        make_activity_featurizer("unknown")


def test_minimol_empty_batch_does_not_load_model() -> None:
    featurizer = MinimolFeaturizer()

    assert featurizer.transform([]).shape == (0, 512)
    assert featurizer._model is None


def test_minimol_shape_and_dtype() -> None:
    featurizer = MinimolFeaturizer()

    features = featurizer.transform(["CCO", "c1ccccc1"])

    assert features.shape == (2, 512)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
