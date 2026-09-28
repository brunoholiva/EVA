"""Max-Tanimoto scoring against a reference molecule set."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from reporting.suppress import suppress_rdkit_logs

suppress_rdkit_logs()


class MaxTanimotoScorer:
    """Compute maximum Tanimoto similarity to reference molecules."""

    def __init__(self, model_path: Path | str) -> None:
        obj = joblib.load(model_path)
        self._train_fps: np.ndarray = obj["fps"].astype(np.float32)
        self._train_sum: np.ndarray = self._train_fps.sum(axis=1)

    def compute(self, fps: np.ndarray) -> np.ndarray:
        """Compute maximum Tanimoto similarity for query fingerprints."""
        batch_sum = fps.sum(axis=1)
        intersection = fps @ self._train_fps.T
        union = batch_sum[:, None] + self._train_sum[None, :] - intersection
        tanimoto = intersection / (union + 1e-8)
        return tanimoto.max(axis=1).astype(np.float32)
