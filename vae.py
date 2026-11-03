"""ChemBed SELFIES Transformer VAE wrapper (256D continuous latent space)."""

from __future__ import annotations

from math import prod
from pathlib import Path

import joblib
import numpy as np
import selfies as sf
import torch
from chembed import checkpoint_utils
from chembed import decode as dec
from chembed import encode as enc
from rdkit import Chem
from rdkit.rdBase import DisableLog

DisableLog("rdApp.*")

MAX_SELFIES_TOKENS = 200
"""Reject decoded SELFIES longer than 200 tokens."""


def _selfies_to_canonical(selfies_str: str) -> str | None:
    """Convert one SELFIES string to canonical SMILES, or ``None`` if invalid."""
    if not selfies_str or selfies_str.count("[") > MAX_SELFIES_TOKENS:
        return None
    try:
        smi = sf.decoder(selfies_str)
        if not smi:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol)
        if not canonical or Chem.MolFromSmiles(canonical) is None:
            return None
        return canonical
    except Exception:
        return None


class ChemBedVAE:
    """Encode and decode molecules in ChemBed's 256D (or PCA-whitened) latent space.

    Parameters
    ----------
    repo_id : str, default="3BioCompBio/chembed-default"
        HuggingFace model repository ID.
    device : str, default="cuda"
        Target device (``"cuda"`` or ``"cpu"``).
    pca_path : str, Path, or None, default=None
        Optional joblib-persisted ``sklearn.decomposition.PCA`` transform
        (whitened active subspace). When set, :attr:`latent_dim` becomes
        ``pca.n_components_`` and :meth:`decode` / :meth:`encode` map
        transparently through ``inverse_transform`` / ``transform``.
    """

    def __init__(
        self,
        repo_id: str = "3BioCompBio/chembed-default",
        device: str = "cuda",
        pca_path: str | Path | None = None,
    ) -> None:
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self._vae = checkpoint_utils.load_vae_from_hub(
            device=self.device, repo_id=repo_id
        )
        self._vae.eval()
        self.raw_latent_dim = int(prod(self._vae.latent_dimension))
        self._pca = joblib.load(pca_path) if pca_path else None
        if self._pca is not None and self._pca.n_features_in_ != self.raw_latent_dim:
            raise ValueError(
                f"PCA expects {self._pca.n_features_in_} features, "
                f"VAE has {self.raw_latent_dim}"
            )
        self.latent_dim = (
            int(self._pca.n_components_)
            if self._pca is not None
            else self.raw_latent_dim
        )
        torch.set_float32_matmul_precision("high")

    def decode(self, z: np.ndarray, batch_size: int = 256) -> list[str | None]:
        """Decode latent vectors ``(n, latent_dim)`` directly to canonical SMILES.

        Invalid or over-length decodes return ``None``.
        """
        if len(z) == 0:
            return []

        if self._pca is not None:
            z = self._pca.inverse_transform(z)

        z_tensor = torch.as_tensor(z, dtype=torch.float32).unsqueeze(1)
        with (
            torch.inference_mode(),
            torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16),
        ):
            selfies_list = dec.decode_zs_to_selfies(
                z_tensor, self._vae, batch_size=batch_size
            )
        return [_selfies_to_canonical(s) for s in selfies_list]

    def encode(self, smiles: list[str], batch_size: int = 256) -> np.ndarray:
        """Encode SMILES strings into ``(len(smiles), latent_dim)`` latent vectors."""
        if not smiles:
            return np.empty((0, self.latent_dim), dtype=np.float32)

        valid_idx: list[int] = []
        valid_smiles: list[str] = []
        for i, s in enumerate(smiles):
            if s and Chem.MolFromSmiles(s) is not None:
                valid_idx.append(i)
                valid_smiles.append(s)

        out = np.zeros((len(smiles), self.raw_latent_dim), dtype=np.float32)
        if valid_smiles:
            zs = enc.encode_multiple_smiles(
                valid_smiles, self._vae, batch_size=batch_size
            )
            arr = zs.squeeze(1).cpu().numpy().astype(np.float32)
            for i, idx in enumerate(valid_idx):
                out[idx] = arr[i]
        if self._pca is not None:
            return self._pca.transform(out).astype(np.float32)
        return out
