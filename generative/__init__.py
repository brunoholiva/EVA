"""Molecule generation via pre-trained ChemBed Transformer VAE."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import selfies as sf
import torch
from chembed import checkpoint_utils
from chembed import decode as dec
from chembed import encode as enc
from rdkit import Chem, RDLogger
from sklearn.decomposition import PCA

RDLogger.DisableLog("rdApp.*")


class ChemBedVAE:
    """Wrapper around the pre-trained ChemBed SELFIES Transformer VAE.

    Provides encode/decode between SMILES and a 256-dim continuous
    latent space, plus random sampling for CMA-MAE seeding.

    Parameters
    ----------
    model_repo_id : str
        HuggingFace repo ID (default ``"3BioCompBio/chembed-default"``).
    device : str
        ``"cuda"`` or ``"cpu"``.
    latent_dim : int, default=256
        Dimensionality of the latent space.
    """

    def __init__(
        self,
        model_repo_id: str = "3BioCompBio/chembed-default",
        device: str = "cuda",
        latent_dim: int = 256,
    ) -> None:
        self._device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self._latent_dim = latent_dim
        self._vae = checkpoint_utils.load_vae_from_hub(
            device=self._device, repo_id=model_repo_id
        )
        self._vae.eval()
        torch.set_float32_matmul_precision("high")

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent space."""
        return self._latent_dim

    @property
    def device(self) -> torch.device:
        """Device the model is loaded on."""
        return self._device

    def encode(self, smiles: list[str], batch_size: int = 256) -> np.ndarray:
        """Encode SMILES strings to latent vectors.

        Parameters
        ----------
        smiles : list of str
            SMILES strings to encode.
        batch_size : int, default=256
            Batch size for the encoding pass.

        Returns
        -------
        np.ndarray of shape ``(len(smiles), latent_dim)``
            Latent vectors. Molecules that fail encoding (invalid SMILES
            or unknown SELFIES tokens) are assigned zero vectors.
        """
        if not smiles:
            return np.empty((0, self._latent_dim), dtype=np.float32)

        valid_idx: list[int] = []
        valid_smiles: list[str] = []
        for i, s in enumerate(smiles):
            if s and Chem.MolFromSmiles(s) is not None:
                valid_idx.append(i)
                valid_smiles.append(s)

        result = np.zeros((len(smiles), self._latent_dim), dtype=np.float32)
        if not valid_smiles:
            return result

        zs = enc.encode_multiple_smiles(valid_smiles, self._vae, batch_size=batch_size)
        arr = zs.squeeze(1).cpu().numpy().astype(np.float32)
        for i, idx in enumerate(valid_idx):
            result[idx] = arr[i]
        return result

    def decode(self, z: np.ndarray, batch_size: int = 256) -> list[str]:
        """Decode latent vectors to SMILES.

        Parameters
        ----------
        z : np.ndarray of shape ``(n, latent_dim)``
            Latent vectors.
        batch_size : int, default=256
            Batch size for the decoding pass.

        Returns
        -------
        list of str
            SMILES strings.  Molecules that fail RDKit parsing are
            returned as empty strings.
        """
        if len(z) == 0:
            return []

        z_tensor = torch.as_tensor(z, dtype=torch.float32).unsqueeze(1)
        device_type = "cuda" if self._device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            selfies_list = dec.decode_zs_to_selfies(
                z_tensor, self._vae, batch_size=batch_size
            )

        result: list[str] = []
        for s in selfies_list:
            try:
                smi = sf.decoder(s)
                mol = Chem.MolFromSmiles(smi)
                result.append(smi if mol is not None else "")
            except Exception:
                result.append("")
        return result

    def generate_random(
        self, n: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Sample random latent vectors from N(0, I).

        Parameters
        ----------
        n : int
            Number of samples.
        rng : np.random.Generator or None
            NumPy random generator for reproducibility.

        Returns
        -------
        np.ndarray of shape ``(n, latent_dim)``
            Random latent vectors.
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.standard_normal((n, self._latent_dim)).astype(np.float32)


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

    def generate_random(
        self, n: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Sample random latent vectors in the isotropic k-dim space.

        Returns
        -------
        np.ndarray of shape ``(n, k)``, sampled from N(0, I).
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.standard_normal((n, self._k)).astype(np.float32)
