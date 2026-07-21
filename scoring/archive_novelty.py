"""Archive novelty scoring: structural diversity via Tanimoto distance."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
from rdkit import RDLogger

from featurization.morgan import smiles_to_morgan

RDLogger.DisableLog("rdApp.*")


class ArchiveNoveltyScorer:
    """Compute structural novelty relative to existing archive members.

    The score is the mean Tanimoto distance (1 - similarity) between a candidate
    molecule and its *k* nearest neighbors among archive members, based on
    Morgan fingerprints.

    *  Score ≈ 0.0 → molecule is very similar to archive contents
    *  Score ≈ 1.0 → molecule is structurally unique vs. archive

    The fingerprint cache is capped at *max_cache_size* entries (FIFO eviction)
    and only contains fingerprints for molecules actually in the archive.
    Call :meth:`update` with newly inserted archive members after each generation,
    or :meth:`rebuild` to replace the full cache (e.g. on resume).

    Parameters
    ----------
    n_bits : int, default=2048
        Morgan fingerprint bit length.
    radius : int, default=2
        Morgan fingerprint radius.
    max_cache_size : int, default=5000
        Maximum number of fingerprints to keep. Oldest entries are evicted
        first when the cap is exceeded.
    n_neighbors : int, default=5
        Number of nearest neighbors to average over.
    """

    def __init__(
        self,
        n_bits: int = 2048,
        radius: int = 2,
        max_cache_size: int = 5000,
        n_neighbors: int = 5,
    ) -> None:
        self._n_bits = n_bits
        self._radius = radius
        self._max_cache_size = max_cache_size
        self._n_neighbors = n_neighbors
        self._fingerprints: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache: np.ndarray = np.empty((0, n_bits), dtype=np.float32)
        self._cache_sums: np.ndarray = np.empty(0, dtype=np.float32)

    @property
    def cache_size(self) -> int:
        """Number of fingerprints in the cache."""
        return len(self._cache)

    @property
    def max_cache_size(self) -> int:
        """Maximum number of fingerprints allowed."""
        return self._max_cache_size

    def _rebuild_arrays(self) -> None:
        """Rebuild dense arrays from the OrderedDict cache."""
        if not self._fingerprints:
            self._cache = np.empty((0, self._n_bits), dtype=np.float32)
            self._cache_sums = np.empty(0, dtype=np.float32)
        else:
            self._cache = np.vstack(list(self._fingerprints.values()))
            self._cache_sums = self._cache.sum(axis=1)

    def _evict(self) -> None:
        """Remove oldest entries until cache is within max_cache_size."""
        while len(self._fingerprints) > self._max_cache_size:
            self._fingerprints.popitem(last=False)
        self._rebuild_arrays()

    def update(self, cell_indices: np.ndarray, smiles: list[str]) -> None:
        """Add fingerprints for newly inserted archive members.

        Parameters
        ----------
        cell_indices : np.ndarray of int
            Flat grid indices of the cells that were inserted or updated.
        smiles : list of str
            SMILES strings corresponding to *cell_indices*.
        """
        if not smiles:
            return
        fps, valid_idx = smiles_to_morgan(
            smiles, radius=self._radius, fp_size=self._n_bits
        )
        if len(fps) == 0:
            return
        for local_i, global_i in enumerate(valid_idx):
            cell = int(cell_indices[global_i])
            self._fingerprints[cell] = fps[local_i : local_i + 1].astype(np.float32)
            self._fingerprints.move_to_end(cell)
        self._evict()

    def rebuild(self, smiles: list[str]) -> None:
        """Replace the entire cache with fingerprints for the given SMILES.

        Use this on startup or resume to initialize the cache from the full
        archive contents.

        Parameters
        ----------
        smiles : list of str
            SMILES strings of all molecules currently in the archive.
        """
        self._fingerprints.clear()
        if not smiles:
            self._rebuild_arrays()
            return
        fps, _valid_idx = smiles_to_morgan(
            smiles, radius=self._radius, fp_size=self._n_bits
        )
        if len(fps) == 0:
            self._rebuild_arrays()
            return
        for i, fp in enumerate(fps):
            self._fingerprints[i] = fp.astype(np.float32)
        self._evict()

    def save(self, path: str | Path) -> None:
        """Persist the cache to disk via joblib.

        Parameters
        ----------
        path : str or Path
            File path to write (e.g. ``archive_novelty_cache.joblib``).
        """
        joblib.dump(
            {
                "n_bits": self._n_bits,
                "radius": self._radius,
                "max_cache_size": self._max_cache_size,
                "n_neighbors": self._n_neighbors,
                "fingerprints": self._fingerprints,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> ArchiveNoveltyScorer:
        """Restore a cached scorer from disk.

        Parameters
        ----------
        path : str or Path
            File path previously written by :meth:`save`.

        Returns
        -------
        ArchiveNoveltyScorer
            Scorer with restored fingerprint cache.
        """
        data = joblib.load(path)
        scorer = cls(
            n_bits=data["n_bits"],
            radius=data["radius"],
            max_cache_size=data["max_cache_size"],
            n_neighbors=data.get("n_neighbors", 5),
        )
        scorer._fingerprints = data["fingerprints"]
        scorer._rebuild_arrays()
        return scorer

    def __call__(self, smiles: list[str]) -> np.ndarray:
        """Score a batch of SMILES strings.

        Parameters
        ----------
        smiles : list of str
            SMILES strings to score.

        Returns
        -------
        np.ndarray of shape ``(len(smiles),)``
            Mean Tanimoto distance to k nearest archive neighbors.
            Invalid SMILES or empty cache get a default value of 1.0.
        """
        scores = np.ones(len(smiles), dtype=np.float32)

        if len(self._cache) == 0:
            return scores

        fps, valid_idx = smiles_to_morgan(
            smiles, radius=self._radius, fp_size=self._n_bits
        )
        if len(fps) == 0:
            return scores

        batch_sum = fps.sum(axis=1)

        intersection = fps @ self._cache.T
        union = batch_sum[:, None] + self._cache_sums[None, :] - intersection
        tanimoto = intersection / (union + 1e-8)

        k = min(self._n_neighbors, tanimoto.shape[1])
        if k >= tanimoto.shape[1]:
            dist = (1.0 - tanimoto).mean(axis=1).astype(np.float32)
        else:
            topk_idx = np.argpartition(-tanimoto, k, axis=1)[:, :k]
            topk_sim = np.take_along_axis(tanimoto, topk_idx, axis=1)
            dist = (1.0 - topk_sim).mean(axis=1).astype(np.float32)

        for i, d in zip(valid_idx, dist):
            scores[i] = d
        return scores
