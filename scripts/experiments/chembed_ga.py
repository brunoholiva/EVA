"""ChemBed genetic algorithm search.

Implements the GA from the ChemBed paper, adapted for EVA:
- M=16 seeds from warm-start representatives (same as CMA-MAE)
- σ=0.5 mutation noise (same as CMA-MAE sigma0)
- ε=0.01 selection tolerance
- Npop=500 unique molecules per generation
- Ngen=1000 generations (scaled from paper's 10 to match CMA-MAE budget)

Total TabPFN evaluations: 500 × 1000 = 500,000.

Output: results/chembed_ga/chembed_ga.csv
Columns: eval, smiles, p_active, valid, gen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from rdkit import Chem

from config import ExperimentConfig
from chemistry.features import MoleculeFeaturizer
from evaluation.activity import load_model, predict_from_features
from generative import ChemBedVAE, ProjectedVAE
from reporting.console import detail, make_progress_bar, section, step
from reporting.suppress import suppress_joblib_warnings, suppress_rdkit_logs

suppress_rdkit_logs()
suppress_joblib_warnings()

M_SEEDS = 16
SIGMA = 0.5
EPSILON = 0.01
NPOP = 500
NGEN = 1000
MAX_MUTATION_ROUNDS = 20
SCORE_BATCH = 500
CHECKPOINT_EVERY = 50
SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChemBed GA search: crossover + mutation + selection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to experiment config TOML (default: config.toml)",
    )
    parser.add_argument(
        "--ngen",
        type=int,
        default=NGEN,
        help=f"Number of generations (default {NGEN})",
    )
    parser.add_argument(
        "--npop",
        type=int,
        default=NPOP,
        help=f"Target unique molecules per generation (default {NPOP})",
    )
    parser.add_argument(
        "--m-seeds",
        type=int,
        default=M_SEEDS,
        help=f"Max seeds per generation (default {M_SEEDS})",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA,
        help=f"Mutation noise std (default {SIGMA})",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=EPSILON,
        help=f"Relative selection tolerance (default {EPSILON})",
    )
    parser.add_argument(
        "--representatives-path",
        type=str,
        default=None,
        help="Path to warm-start CSV (default: from config.toml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/chembed_ga",
        help="Output directory (default: results/chembed_ga)",
    )
    return parser.parse_args()


def _canonical(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(mol) if mol else None


def _crossover(z: np.ndarray) -> np.ndarray:
    """Compute all pairwise interpolations: zij = (zi + zj) / 2 for i < j.

    Parameters
    ----------
    z : np.ndarray of shape (M, latent_dim)
        Seed latent vectors.

    Returns
    -------
    np.ndarray of shape (M*(M-1)/2, latent_dim)
        Mixture vectors.
    """
    m = len(z)
    upper_i, upper_j = np.triu_indices(m, k=1)
    return (z[upper_i] + z[upper_j]) / 2.0


def _mutation_round(
    base: np.ndarray,
    sigma: float,
    vae,
    seen: set[str],
) -> list[str]:
    """One round of mutation + decode + dedup.

    Parameters
    ----------
    base : np.ndarray of shape (n_base, latent_dim)
        Base vectors (seeds + mixtures).
    sigma : float
        Mutation noise std.
    vae : ChemBedVAE or ProjectedVAE
    seen : set of str
        Canonical SMILES already collected (not returned again).

    Returns
    -------
    list of str
        New unique canonical SMILES from this round.
    """
    noise = np.random.standard_normal(base.shape).astype(np.float32) * sigma
    candidates = (base + noise).astype(np.float32)
    smiles_batch = vae.decode(candidates, batch_size=len(candidates))

    new_unique: list[str] = []
    for smi in smiles_batch:
        canonical = _canonical(smi)
        if canonical is not None and canonical not in seen:
            seen.add(canonical)
            new_unique.append(canonical)
    return new_unique


def _score_batch(
    smiles: list[str],
    featurizer,
    model,
) -> np.ndarray:
    """Featurize and score a batch of SMILES with TabPFN."""
    X = featurizer.transform(smiles)
    _, probs = predict_from_features(X, model)
    return probs[:, 1]


def _select(
    smiles: list[str],
    p_active: np.ndarray,
    epsilon: float,
    max_seeds: int,
    rng: np.random.Generator,
) -> list[str]:
    """Select survivors: all with f >= (1-ε)*max(f), capped at max_seeds.

    Parameters
    ----------
    smiles : list of str
        Scored SMILES.
    p_active : np.ndarray
        Fitness values.
    epsilon : float
        Relative tolerance.
    max_seeds : int
        Cap on number of survivors.
    rng : np.random.Generator
        For random sampling when capping.

    Returns
    -------
    list of str
        Selected SMILES (length <= max_seeds).
    """
    threshold = (1 - epsilon) * p_active.max()
    selected_idx = np.where(p_active >= threshold)[0]

    if len(selected_idx) > max_seeds:
        selected_idx = rng.choice(selected_idx, size=max_seeds, replace=False)

    return [smiles[i] for i in selected_idx]


def main() -> None:
    args = _parse_args()
    cfg = ExperimentConfig.from_toml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rep_path = args.representatives_path
    if rep_path is None:
        rep_path = "data/warmstart_representatives/representatives_all.csv"
    if not Path(rep_path).exists():
        raise ValueError(
            f"Representatives file not found: {rep_path}. "
            "Pass --representatives-path."
        )

    section("ChemBed GA")
    step(f"M={args.m_seeds}, σ={args.sigma}, ε={args.epsilon}")
    step(f"Ngen={args.ngen}, Npop={args.npop}")
    detail(f"Total evaluations: {args.ngen * args.npop:,}")

    step("Loading VAE")
    vae = ChemBedVAE(
        model_repo_id=cfg.generative.model_repo_id,
        device=cfg.generative.device,
        latent_dim=cfg.generative.latent_dim,
    )
    if cfg.pca.enabled:
        vae = ProjectedVAE(vae, cfg.pca.path)
        detail(f"PCA projected: {vae.latent_dim} dim")

    step("Loading TabPFN + featurizer")
    model = load_model(
        path=cfg.activity.model_path,
        device=cfg.activity.device,
        softmax_temperature=cfg.activity.softmax_temperature,
    )
    featurizer = MoleculeFeaturizer()

    step(f"Loading seeds from {rep_path}")
    rep_df = pd.read_csv(rep_path)
    seed_smiles = rep_df["SMILES"].dropna().tolist()[: args.m_seeds]
    detail(f"Loaded {len(seed_smiles)} seeds")

    rng = np.random.default_rng(SEED)

    progress, task_id = make_progress_bar("GA", args.ngen)

    records: list[tuple[int, str, float, bool, int]] = []
    eval_counter = 0
    n_total_decoded = 0

    with progress:
        z = vae.encode(seed_smiles, batch_size=len(seed_smiles)).astype(np.float32)

        for gen in range(args.ngen):
            base = np.vstack([z, _crossover(z)])

            gen_smiles: list[str] = []
            seen: set[str] = set()

            for _round in range(MAX_MUTATION_ROUNDS):
                new = _mutation_round(base, args.sigma, vae, seen)
                gen_smiles.extend(new)
                n_total_decoded += len(base)
                if len(gen_smiles) >= args.npop:
                    break

            gen_smiles = gen_smiles[: args.npop]

            if not gen_smiles:
                detail(f"Gen {gen}: no valid molecules, reusing previous seeds")
                continue

            p_active = _score_batch(gen_smiles, featurizer, model)

            for smi, pa in zip(gen_smiles, p_active):
                eval_counter += 1
                records.append((eval_counter, smi, float(pa), True, gen))

            survivors = _select(gen_smiles, p_active, args.epsilon, args.m_seeds, rng)

            if survivors:
                z = vae.encode(survivors, batch_size=len(survivors)).astype(np.float32)

            progress.update(task_id, completed=gen + 1)

            if (gen + 1) % CHECKPOINT_EVERY == 0 and records:
                detail(
                    f"Gen {gen + 1}/{args.ngen}: evals={eval_counter:,}, "
                    f"survivors={len(survivors)}, decoded={n_total_decoded:,}"
                )
                _save_checkpoint(records, output_dir)

    step("Saving results")
    df = pd.DataFrame(records, columns=["eval", "smiles", "p_active", "valid", "gen"])
    csv_path = output_dir / "chembed_ga.csv"
    df.to_csv(csv_path, index=False)
    step(f"Saved {csv_path} ({len(df):,} rows)")

    checkpoint = output_dir / "chembed_ga_checkpoint.csv"
    if checkpoint.exists():
        checkpoint.unlink()

    section("Summary")
    detail(f"Generations: {args.ngen}")
    detail(f"Total decoded: {n_total_decoded:,}")
    detail(f"Total evaluated: {eval_counter:,}")
    detail(f"P(active) max: {df['p_active'].max():.4f}")
    detail(f"P(active) mean: {df['p_active'].mean():.4f}")
    detail(f"P > 0.5: {(df['p_active'] > 0.5).sum():,}")
    detail(f"P > 0.6: {(df['p_active'] > 0.6).sum():,}")
    detail(f"P > 0.7: {(df['p_active'] > 0.7).sum():,}")

    first_above_06 = df[df["p_active"] > 0.6]
    if len(first_above_06) > 0:
        detail(
            f"First P>0.6 at eval #{first_above_06['eval'].iloc[0]:,} "
            f"(gen {first_above_06['gen'].iloc[0]})"
        )
    else:
        detail("No molecules above 0.6")

    n_unique = df["smiles"].nunique()
    detail(f"Unique molecules: {n_unique:,}")

    gen_best = df.groupby("gen")["p_active"].max()
    detail(
        f"Best P(active) by gen: gen0={gen_best.iloc[0]:.4f}, "
        f"final={gen_best.iloc[-1]:.4f}"
    )


def _save_checkpoint(
    records: list[tuple[int, str, float, bool, int]],
    output_dir: Path,
) -> None:
    df = pd.DataFrame(records, columns=["eval", "smiles", "p_active", "valid", "gen"])
    df.to_csv(output_dir / "chembed_ga_checkpoint.csv", index=False)


if __name__ == "__main__":
    main()
