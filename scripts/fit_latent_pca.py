"""Fit PCA on VAE latent space for dimensionality reduction + whitening.

Encodes the predictor training set through the ChemBed VAE, fits a PCA
with whitening at a configurable variance threshold, and saves the
transform for use by ``ProjectedVAE`` in the main pipeline.

Usage
-----
    micromamba run -n eva python scripts/fit_latent_pca.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.progress import Progress
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generative import ChemBedVAE
from chemistry.smiles import canonicalize_batch
from reporting.console import console, detail, saved, section, step

TRAINING_DATA = "data/predictor/predictor_training_data.csv"
OUTPUT_PATH = "data/pca_latent.joblib"
VARIANCE_THRESHOLD = 0.99
N_ENCODE_SAMPLE = 100000
ENCODE_BATCH_SIZE = 256
RANDOM_SEED = 42


def main() -> None:
    section("Fitting PCA on VAE latent space")

    vae = ChemBedVAE(device="cuda")
    detail(f"VAE latent dim: {vae.latent_dim}")

    step(f"Loading training data from {TRAINING_DATA}")
    df = pd.read_csv(TRAINING_DATA)
    all_smiles = df["SMILES"].dropna().tolist()
    all_smiles = list(dict.fromkeys(canonicalize_batch(all_smiles)))
    detail(f"Available SMILES: {len(all_smiles)}")

    rng = np.random.default_rng(RANDOM_SEED)
    n_sample = min(N_ENCODE_SAMPLE, len(all_smiles))
    sample_smiles = rng.choice(all_smiles, size=n_sample, replace=False).tolist()
    step(f"Encoding {n_sample} molecules through the VAE")

    with Progress(console=console) as progress:
        task = progress.add_task("Encoding molecules through the VAE", total=n_sample)
        z_list: list[np.ndarray] = []
        for i in range(0, n_sample, ENCODE_BATCH_SIZE):
            batch = sample_smiles[i : i + ENCODE_BATCH_SIZE]
            z_batch = vae.encode(batch, batch_size=ENCODE_BATCH_SIZE)
            z_list.append(z_batch)
            progress.update(task, advance=len(batch))
        z = np.concatenate(z_list, axis=0)
        progress.update(task, completed=n_sample)

    detail(f"Encoded shape: {z.shape}")
    detail(f"mean={z.mean():.4f}  std={z.std():.4f}")

    pca = PCA(n_components=VARIANCE_THRESHOLD, whiten=True)
    step("Fitting whitened PCA")
    pca.fit(z)

    k = pca.n_components_
    var_explained = pca.explained_variance_ratio_.sum()
    detail(f"PCA components: {k} / {vae.latent_dim}")
    detail(f"Variance explained: {var_explained:.4%}")

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, output_path)
    saved("PCA transform", output_path)


if __name__ == "__main__":
    main()
