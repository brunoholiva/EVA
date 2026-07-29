"""Applicability domain scoring: distance to training set."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from rdkit import RDLogger

from featurization.morgan import smiles_to_morgan
from featurization.tanimoto import batch_tanimoto_topk

N_NEIGHBORS_DEFAULT: int = 5


class ADScorer:
    """Compute applicability domain score for candidate molecules.

    The score is the mean Tanimoto distance (1 - similarity) to the *k* nearest
    neighbors in the antibiotic training set, computed via batch matrix
    multiplication on cached training fingerprints.

    *  Score ≈ 0.0 → molecule is very similar to known antibiotics
    *  Score ≈ 1.0 → molecule is far from the training manifold

    Parameters
    ----------
    model_path : Path
        Path to the joblib artifact built by ``build_ad_model.py``.
    n_neighbors : int, default=5
        Number of nearest neighbors to average over.
    """

    def __init__(
        self, model_path: Path, n_neighbors: int = N_NEIGHBORS_DEFAULT
    ) -> None:
        obj = joblib.load(model_path)
        self._train_fps: np.ndarray = obj["fps"].astype(np.float32)
        self._train_sum: np.ndarray = self._train_fps.sum(axis=1)
        self._n_bits: int = obj.get("n_bits", 2048)
        self._radius: int = obj.get("radius", 2)
        self._n_neighbors: int = n_neighbors
        RDLogger.DisableLog("rdApp.*")

    @property
    def radius(self) -> int:
        """Morgan fingerprint radius used by this scorer."""
        return self._radius

    @property
    def n_bits(self) -> int:
        """Number of bits in the Morgan fingerprint."""
        return self._n_bits

    @property
    def n_neighbors(self) -> int:
        """Number of nearest neighbors used for AD scoring."""
        return self._n_neighbors

    @property
    def train_fps(self) -> np.ndarray:
        """Training set fingerprints (``(n_train, n_bits)`` float32)."""
        return self._train_fps

    @property
    def train_sum(self) -> np.ndarray:
        """Precomputed row sums of *train_fps* (``(n_train,)`` float32)."""
        return self._train_sum

    @property
    def n_features(self) -> int:
        """Number of bits in the Morgan fingerprint."""
        return self._n_bits

    def compute_from_fps(self, fps: np.ndarray) -> np.ndarray:
        """Compute AD scores from pre-computed Morgan fingerprints.

        Parameters
        ----------
        fps : np.ndarray of shape ``(n, n_bits)``
            Morgan fingerprints for valid query molecules (float32).

        Returns
        -------
        np.ndarray of shape ``(n,)``
            Mean Tanimoto distance to k nearest training neighbors.
        """
        return batch_tanimoto_topk(
            fps, self._train_fps, self._train_sum, self._n_neighbors
        )

    def __call__(self, smiles: list[str]) -> np.ndarray:
        """Score a batch of SMILES strings.

        Parameters
        ----------
        smiles : list of str
            SMILES strings to score.

        Returns
        -------
        np.ndarray of shape ``(len(smiles),)``
            Mean Tanimoto distance to k nearest training neighbors.
            Invalid SMILES get a default distance of 1.0.
        """
        fps, valid_idx = smiles_to_morgan(
            smiles, radius=self._radius, fp_size=self._n_bits
        )

        scores = np.ones(len(smiles), dtype=np.float32)
        if len(fps) == 0:
            return scores

        dist = self.compute_from_fps(fps)
        for i, d in zip(valid_idx, dist):
            scores[i] = d
        return scores
