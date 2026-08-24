"""Batch Tanimoto distance computation on Morgan fingerprints."""

from __future__ import annotations

import numpy as np


def batch_tanimoto_topk(
    query_fps: np.ndarray,
    cache_fps: np.ndarray,
    cache_sums: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Compute mean Tanimoto distance to the k nearest cache neighbors.

    Uses batch matrix multiplication for the full Tanimoto similarity
    matrix and ``np.argpartition`` for efficient top-k selection.

    Parameters
    ----------
    query_fps : np.ndarray of shape ``(n_valid, fp_size)``
        Fingerprint matrix for valid query molecules (float32).
    cache_fps : np.ndarray of shape ``(n_cache, fp_size)``
        Fingerprint matrix for cached reference molecules (float32).
    cache_sums : np.ndarray of shape ``(n_cache,)``
        Row sums of *cache_fps* (precomputed).
    n_neighbors : int
        Number of nearest neighbors to average over.

    Returns
    -------
    np.ndarray of shape ``(n_valid,)``
        Mean Tanimoto distance to the k nearest cache neighbors for each
        query. Values are in ``[0.0, 1.0]``.
    """
    batch_sum = query_fps.sum(axis=1)

    intersection = query_fps @ cache_fps.T
    union = batch_sum[:, None] + cache_sums[None, :] - intersection
    tanimoto = intersection / (union + 1e-8)

    k = min(n_neighbors, tanimoto.shape[1])
    if k >= tanimoto.shape[1]:
        return (1.0 - tanimoto).mean(axis=1).astype(np.float32)

    topk_idx = np.argpartition(-tanimoto, k, axis=1)[:, :k]
    topk_sim = np.take_along_axis(tanimoto, topk_idx, axis=1)
    return (1.0 - topk_sim).mean(axis=1).astype(np.float32)
