"""Structural archive dimensions via PCA on active-molecule fingerprints.

The archive's behaviour coordinates are what force CMA-MAE to fill
structurally distinct cells.  ``ad`` (distance to the training manifold)
is a *magnitude*: it keeps candidates in known space but is blind to
chemotype.  These projections add *direction*: a molecule's 2048-bit
Morgan fingerprint is projected onto the top principal components of the
active training molecules' fingerprints, so distinct chemotypes land in
different archive cells and the archive is forced to retain diverse
structures instead of letting one family sprawl across the grid.

The projector is deterministic and stored on disk; it never depends on
the state of a running optimization, so the archive cells are stable for
the whole experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.decomposition import PCA

from featurization.morgan import smiles_to_morgan

FP_RADIUS_DEFAULT = 2
FP_BITS_DEFAULT = 2048


@dataclass
class FPProjector:
    """Deterministic fingerprint -> structural-coordinate projection.

    Attributes
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA (``n_components_`` = number of structural axes).
    radius : int
        Morgan fingerprint radius the PCA was fit on.
    n_bits : int
        Morgan fingerprint bit length the PCA was fit on.
    bounds : np.ndarray of shape ``(k, 2)`` or None
        Per-axis ``[lo, hi]`` grid bounds (e.g. training percentiles).
    """

    pca: PCA
    radius: int = FP_RADIUS_DEFAULT
    n_bits: int = FP_BITS_DEFAULT
    bounds: np.ndarray | None = None

    @property
    def k(self) -> int:
        """Number of structural axes (PCA components)."""
        return self.pca.n_components_

    @property
    def ranges(self) -> list[list[float]]:
        """Grid ranges per axis, from ``bounds`` (fails if unset)."""
        if self.bounds is None:
            raise ValueError("FPProjector has no bounds; fit it with bounds_q")
        return [list(row) for row in np.asarray(self.bounds, dtype=float)]

    def transform(self, fps: np.ndarray) -> np.ndarray:
        """Project an ``(n, n_bits)`` fingerprint matrix to ``(n, k)``.

        Parameters
        ----------
        fps : np.ndarray of shape ``(n, n_bits)``
            Morgan fingerprints (the same bit length as fit).

        Returns
        -------
        np.ndarray of shape ``(n, k)``
            Structural coordinates (one value per PCA axis).
        """
        fps = np.asarray(fps, dtype=np.float32)
        if fps.ndim != 2 or fps.shape[1] != self.n_bits:
            raise ValueError(
                f"expected fingerprints of shape (n, {self.n_bits}), got {fps.shape}"
            )
        return self.pca.transform(fps).astype(np.float32)


def fit_fp_pca(
    smiles: list[str],
    radius: int = FP_RADIUS_DEFAULT,
    n_bits: int = FP_BITS_DEFAULT,
    n_components: int = 2,
    bounds_q: tuple[float, float] = (0.005, 0.995),
    out_path: str | None = None,
) -> FPProjector:
    """Fit an ``FPProjector`` on a set of SMILES and optionally save it.

    The PCA is fit on the molecules' Morgan fingerprints; per-axis grid
    bounds are taken from the *bounds_q* percentiles of the training
    projections (so most training molecules fall in-range).

    Parameters
    ----------
    smiles : list of str
        Molecules to fit on (e.g. the activity predictor's actives).
    radius : int, default=2
        Morgan fingerprint radius.
    n_bits : int, default=2048
        Morgan fingerprint bit length.
    n_components : int, default=2
        Number of structural axes to keep.
    bounds_q : tuple of float, default=(0.005, 0.995)
        Percentiles (as fractions) used for the grid bounds.
    out_path : str or None
        If given, the projector is dumped here via ``joblib``.

    Returns
    -------
    FPProjector
        Fitted projector with ``bounds`` set.
    """
    fps, _ = smiles_to_morgan(smiles, radius=radius, fp_size=n_bits)
    if len(fps) < n_components:
        raise ValueError("not enough valid fingerprints to fit PCA")

    pca = PCA(n_components=n_components, random_state=0).fit(fps)
    proj = pca.transform(fps)
    lo, hi = np.percentile(proj, [100.0 * bounds_q[0], 100.0 * bounds_q[1]], axis=0)
    bounds = np.column_stack([lo, hi])

    projector = FPProjector(pca=pca, radius=radius, n_bits=n_bits, bounds=bounds)
    if out_path is not None:
        joblib.dump(projector, out_path)
    return projector


def load_fp_pca(path: str) -> FPProjector:
    """Load an ``FPProjector`` saved by :func:`fit_fp_pca`."""
    return joblib.load(path)
