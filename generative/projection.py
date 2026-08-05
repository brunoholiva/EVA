"""PCA projection wrapper for VAE latent space."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA

from generative.vae import ChemBedVAE


class ProjectedVAE:
    """VAE wrapper with PCA projection + whitening.

    Transparently maps between a low-dimensional isotropic working space
    (used by CMA-ES) and the VAE's native 256-dim latent space.

    Parameters
    ----------
    vae : ChemBedVAE
        The underlying VAE.
    pca_path : str or Path
        Path to a joblib-saved ``sklearn.decomposition.PCA`` instance
        fitted with ``whiten=True``.
    """

    def __init__(self, vae: ChemBedVAE, pca_path: str | Path) -> None:
        self._vae = vae
        self._pca: PCA = joblib.load(pca_path)
        self._k: int = self._pca.n_components_

    @property
    def vae(self) -> ChemBedVAE:
        """The underlying VAE."""
        return self._vae

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the projected working space (k)."""
        return self._k

    @property
    def device(self) -> torch.device:
        """Device the underlying VAE is loaded on."""
        return self._vae.device

    def encode(self, smiles: list[str], **kwargs) -> np.ndarray:
        """Encode SMILES to k-dim whitened PCA space.

        Returns
        -------
        np.ndarray of shape ``(len(smiles), k)``.
        """
        z_256 = self._vae.encode(smiles, **kwargs)
        return self._pca.transform(z_256).astype(np.float32)

    def decode(self, z_k: np.ndarray, **kwargs) -> list[str]:
        """Decode k-dim PCA vectors to SMILES.

        Accepts k-dim vectors, inverse-transforms to 256-dim,
        and passes them to the VAE decoder.
        """
        z_256 = self._pca.inverse_transform(z_k).astype(np.float32)
        return self._vae.decode(z_256, **kwargs)
