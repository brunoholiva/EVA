"""Shared helpers for ChemBed VAE diagnostic notebooks."""

from __future__ import annotations

import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
from rdkit.DataStructs import BulkTanimotoSimilarity
from sklearn.decomposition import PCA

RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_vae(device: str = "cuda"):
    """Load native 256d VAE and PCA projection. Returns (vae_native, pca_transform)."""
    from generative.vae import ChemBedVAE

    vae = ChemBedVAE(device=device)
    pca: PCA = joblib.load(DATA / "pca_latent.joblib")
    return vae, pca


def decode_and_classify(vae, z: np.ndarray, batch_size: int = 512) -> pd.DataFrame:
    """Decode latent vectors, return DataFrame with smiles/valid/canonical."""
    raw = vae.decode(z, batch_size=batch_size)
    rows = []
    for s in raw:
        mol = Chem.MolFromSmiles(s) if s else None
        rows.append(
            {
                "smiles": s,
                "canonical": Chem.MolToSmiles(mol) if mol else None,
                "valid": mol is not None,
            }
        )
    return pd.DataFrame(rows)


def sample_latent(n: int, dim: int, scale: float = 1.0, seed: int = 42) -> np.ndarray:
    """Sample scaled standard normal latent vectors."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, dim), dtype=np.float32) * scale).astype(np.float32)


def pairwise_tanimoto(smiles: list[str], n: int = 500, seed: int = 42) -> list[float]:
    """Sample pairwise Tanimoto distances from a list of valid SMILES."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(smiles), size=min(n, len(smiles)), replace=False)
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smiles[i]), radius=2, nBits=2048
        )
        for i in idx
    ]
    dists = []
    for i in range(len(fps)):
        sims = BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        dists.extend(1 - s for s in sims)
    return dists


def pca_project(pca, z_256: np.ndarray) -> np.ndarray:
    """Project native 256d vectors to whitened PCA space."""
    return pca.transform(z_256).astype(np.float32)


def clear_gpu():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cacher(name: str, cache_dir: Path | None = None):
    """Return (path, load, save) triple for a parquet cache file."""
    cache_dir = cache_dir or (ROOT / "notebooks" / "cache" / "01_chembed")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.parquet"

    def load():
        return pd.read_parquet(path) if path.exists() else None

    def save(df: pd.DataFrame):
        df.to_parquet(path, index=False)

    return path, load, save


def mol_grid(
    smiles: list[str], legends: list[str] | None = None, mols_per_row: int = 4
):
    """Display a grid of molecules from SMILES."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(300, 200),
        legends=legends,
    )


def load_reference_actives(force: bool = False) -> tuple[list[str], np.ndarray]:
    """Return canonical training actives with their oracle P(active) scores.

    Selects ``target == 1`` rows from ``data/predictor/predictor_training_data.csv``
    in file order, canonicalizes them, and scores them with the oracle configured
    in ``config.toml``. Results are cached in
    ``notebooks/cache/01_chembed/reference_actives_<representation>.parquet``.

    The oracle model (a ~9.4 GB joblib) is only loaded on a cache miss, so callers
    can reuse the cached scores without paying the load.

    Parameters
    ----------
    force : bool, default=False
        Recompute and overwrite the cache even if it exists.

    Returns
    -------
    smiles : list of str
        Canonical active SMILES.
    scores : np.ndarray of shape ``(len(smiles),)``
        P(active) from the configured oracle.
    """
    from config import ExperimentConfig
    from EVA import load_activity_oracle
    from evaluation.activity import predict_from_features

    cfg = ExperimentConfig.from_toml(ROOT / "config.toml")
    model_path = Path(cfg.activity.model_path)
    if not model_path.is_absolute():
        cfg.activity.model_path = str(ROOT / model_path)

    _, load, save = cacher(f"reference_actives_{cfg.activity.representation}")
    cached = None if force else load()
    if cached is None:
        from chemistry.smiles import parse_and_canonical

        frame = pd.read_csv(DATA / "predictor" / "predictor_training_data.csv")
        raw = frame.loc[frame["target"] == 1, "SMILES"].dropna().tolist()
        smiles = [
            canonical
            for valid, canonical in (parse_and_canonical(s) for s in raw)
            if valid
        ]
        tabpfn, featurizer = load_activity_oracle(cfg)
        _, probs = predict_from_features(featurizer.transform(smiles), tabpfn)
        cached = pd.DataFrame({"smiles": smiles, "p_active": probs[:, 1]})
        save(cached)
    return cached["smiles"].tolist(), cached["p_active"].to_numpy()
