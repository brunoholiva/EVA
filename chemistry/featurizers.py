"""Molecular featurization for activity models."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

N_MINIMOL_FEATURES: int = 512


def _import_minimol_onnx() -> None:
    """Make ``minimol_onnx`` importable under rdkit 2023.9.6 + pandas 2.3.

    rdkit 2023.9.6's ``Chem.PandasTools`` calls
    ``pandas.io.formats.format.get_adjustment`` (removed in pandas 2.3), and
    datamol imports ``PandasTools`` at module scope even though MiniMol never
    converts DataFrames. A stub satisfies that import; the real submodule is
    unused by this project.

    Kept lazy so the stub only exists when the minimol representation is used.
    """
    import sys
    import types

    if "rdkit.Chem.PandasTools" in sys.modules:
        return
    stub = types.ModuleType("rdkit.Chem.PandasTools")
    sys.modules["rdkit.Chem.PandasTools"] = stub
    Chem.PandasTools = stub


class MinimolFeaturizer:
    """Generate MiniMol 512-dim fingerprints via ``minimol-onnx``.

    ``minimol-onnx`` freezes the MiniMol graph into ONNX and vendors the
    graphium featurizer as pure Python, so the whole path runs inside EVA's
    environment -- no sidecar, no compiled PyG extensions. The ONNX session
    is constructed lazily on the first call. Fingerprints are deterministic
    given this environment; eigenvector-sign conventions are LAPACK-specific
    (upstream documents a cosine floor of 0.99 across platforms).
    """

    n_features = N_MINIMOL_FEATURES

    def __init__(self, batch_size: int = 512) -> None:
        self._model = None
        self.batch_size = batch_size

    def _ensure_model(self):
        """Return the loaded MiniMol ONNX model, loading it on first use."""
        if self._model is None:
            _import_minimol_onnx()
            from minimol_onnx import MinimolONNX

            self._model = MinimolONNX(batch_size=self.batch_size)
        return self._model

    def transform(self, smiles: list[str]) -> np.ndarray:
        """Return MiniMol fingerprints for *smiles*.

        Parameters
        ----------
        smiles : list of str
            Chemically valid SMILES strings.

        Returns
        -------
        np.ndarray of shape ``(n, 512)``
            Float32 MiniMol fingerprints.
        """
        if not smiles:
            return np.empty((0, self.n_features), dtype=np.float32)
        return self._ensure_model()(smiles)


def make_activity_featurizer(representation: str = "minimol") -> MinimolFeaturizer:
    """Create the featurizer matching an activity model representation.

    Parameters
    ----------
    representation : str, default="minimol"
        Feature representation name. Currently only ``"minimol"`` is supported.

    Returns
    -------
    MinimolFeaturizer
        Configured featurizer instance.

    Raises
    ------
    ValueError
        If *representation* is not ``"minimol"``.
    """
    if representation == "minimol":
        return MinimolFeaturizer()
    raise ValueError(
        f"Unknown activity representation {representation!r}; expected 'minimol'"
    )
