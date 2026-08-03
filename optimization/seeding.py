"""Seed CMA-MAE from the activity predictor's training set.

The VAE prior (standard normal) decodes almost entirely into molecules far
from the predictor's training distribution (AD > 0.6, P(active) < 0.2), so
random seeding starts the search in an objective desert.  Encoding training
molecules through the VAE instead lands directly in the low-AD, high-activity
region, and Butina-clustering the actives lets each CMA-ES emitter begin at a
different chemotype family.

Two artifacts are produced:

* ``z_seeds`` — latent working-space points for :meth:`CMAMAELoop.seed_archive`
  (one per diverse active cluster, encoded through the VAE).
* ``x0s`` — per-emitter starting points: ``n_cluster_emitters`` are encoded
  cluster medoids (one active family each); the remainder are random
  ``N(0, I)`` draws to preserve global exploration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Cluster import Butina
from rich.console import Console

from config import SeedingConfig

if TYPE_CHECKING:
    from generative import ProjectedVAE

RDLogger.DisableLog("rdApp.*")

FP_RADIUS: int = 2
FP_SIZE: int = 2048

console = Console()

_GEN_CACHE: dict[tuple[int, int], object] = {}


def _generator(radius: int = FP_RADIUS, fp_size: int = FP_SIZE) -> object:
    """Return a cached Morgan generator for the given parameters."""
    key = (radius, fp_size)
    if key not in _GEN_CACHE:
        _GEN_CACHE[key] = GetMorganGenerator(radius=radius, fpSize=fp_size)
    return _GEN_CACHE[key]


def _fp(smi: str):
    """Return the Morgan fingerprint of *smi*, or ``None`` if invalid."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return _generator().GetFingerprint(mol)


def load_training_smiles(cfg: SeedingConfig) -> list[str]:
    """Load SMILES from the predictor training CSV, optionally actives only.

    Parameters
    ----------
    cfg : SeedingConfig
        Seeding configuration (data path, columns, active filter).

    Returns
    -------
    list of str
        Unique, RDKit-parseable SMILES strings.
    """
    df = pd.read_csv(cfg.data_path)
    if cfg.use_actives_only:
        df = df[df[cfg.target_col] == 1]
    smiles = df[cfg.smiles_col].dropna().unique().tolist()
    return [s for s in smiles if _fp(s) is not None]


def cluster_smiles(smiles: list[str], threshold: float = 0.5) -> list[list[int]]:
    """Butina-cluster SMILES by Morgan Tanimoto distance.

    Parameters
    ----------
    smiles : list of str
        SMILES strings to cluster.
    threshold : float, default=0.5
        Tanimoto distance threshold passed to Butina clustering.

    Returns
    -------
    list of list of int
        Clusters of indices into *smiles*, sorted by size descending.
    """
    fps = [_fp(s) for s in smiles]

    distances: list[float] = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend([1.0 - float(s) for s in sims])

    clusters = Butina.ClusterData(distances, len(fps), threshold, isDistData=True)
    return sorted(clusters, key=len, reverse=True)


def _medoid_index(smiles: list[str], cluster: list[int]) -> int:
    """Return the fingerprint-medoid index of *cluster* within *smiles*.

    Parameters
    ----------
    smiles : list of str
        Full SMILES list the cluster indices refer to.
    cluster : list of int
        Cluster member indices.

    Returns
    -------
    int
        Index (into *smiles*) of the member with the highest mean Tanimoto
        similarity to the rest of the cluster.
    """
    fps = [_fp(smiles[i]) for i in cluster]
    best_i, best_score = cluster[0], -1.0
    for idx, fp in zip(cluster, fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        score = float(np.mean(sims))
        if score > best_score:
            best_score, best_i = score, idx
    return best_i


def make_seed_and_emitter_points(
    vae: ProjectedVAE,
    cfg: SeedingConfig,
    n_emitters: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build archive seeds and per-emitter starting points from actives.

    Actives are Butina-clustered (when ``cfg.dedupe_clusters`` is set) and
    one medoid per cluster is encoded through the VAE.  The first
    ``cfg.n_cluster_emitters`` emitters start at distinct cluster medoids;
    the rest start at random ``N(0, I)`` points.

    Parameters
    ----------
    vae : ProjectedVAE
        VAE wrapper whose ``encode`` maps SMILES into the k-dim working space.
    cfg : SeedingConfig
        Seeding configuration.
    n_emitters : int
        Total number of emitters (must be >= number of cluster emitters).
    rng : np.random.Generator or None
        Random generator for the random emitter draws.

    Returns
    -------
    z_seeds : np.ndarray of shape ``(n_seeds, k)``
        Latent working-space points for seeding the archive.
    x0s : list of np.ndarray
        One k-dim starting point per emitter.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    console.print("[bold green]Preparing training-set seeds ...")
    actives = load_training_smiles(cfg)
    console.print(f"  Loaded {len(actives):,} active training molecules")

    clusters: list[list[int]] = []
    medoid_smiles: list[str] = []
    if cfg.dedupe_clusters and actives:
        clusters = cluster_smiles(actives, cfg.cluster_threshold)
        medoid_smiles = [actives[_medoid_index(actives, c)] for c in clusters]
        console.print(
            f"  Butina clustering ({cfg.cluster_threshold}): "
            f"{len(clusters):,} clusters (largest {len(clusters[0])})"
        )
        seed_smiles = medoid_smiles[: cfg.max_seeds]
    else:
        seed_smiles = actives[: cfg.max_seeds]

    z_seeds = vae.encode(seed_smiles)
    console.print(f"  Seeding archive with {len(z_seeds):,} encoded actives")

    x0s: list[np.ndarray] = []
    n_cluster_used = 0
    n_cluster = min(cfg.n_cluster_emitters, n_emitters)
    if n_cluster > 0 and medoid_smiles:
        n_cluster_used = min(n_cluster, len(medoid_smiles))
        cluster_x0s = vae.encode(medoid_smiles[:n_cluster_used])
        x0s.extend(z.astype(np.float64) for z in cluster_x0s)

    for _ in range(len(x0s), n_emitters):
        x0s.append(rng.standard_normal(vae.latent_dim).astype(np.float64))

    console.print(
        f"  Emitter starts: {n_cluster_used} from cluster medoids, "
        f"{len(x0s) - n_cluster_used} random"
    )
    return z_seeds, x0s
