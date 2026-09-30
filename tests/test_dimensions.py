"""Tests for archive dimension dispatch."""

from __future__ import annotations

import pytest
from rdkit import Chem

from chemistry.molecules import ParsedMolecules
from evaluation.dimensions import compute_dimension


def test_descriptor_dimension_returns_one_value_per_molecule() -> None:
    parsed = ParsedMolecules(["CCO", "c1ccccc1"])
    values = compute_dimension("tpsa", parsed)
    assert values.shape == (2,)
    assert values[0] == pytest.approx(
        Chem.MolFromSmiles("CCO").GetSubstructMatches and 20.23, abs=0.5
    )


def test_invalid_molecule_gets_nan() -> None:
    import numpy as np

    parsed = ParsedMolecules(["CCO", "not-a-molecule"])
    values = compute_dimension("tpsa", parsed)
    assert values[0] == pytest.approx(20.23, abs=0.5)
    assert np.isnan(values[1])


def test_unknown_dimension_raises() -> None:
    parsed = ParsedMolecules(["CCO"])
    with pytest.raises(ValueError, match="Unknown dimension"):
        compute_dimension("nope", parsed)
