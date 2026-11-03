r"""Fit an oracle-free whitened PCA on ChemBed's intrinsic active manifold.

Samples ``z ~ N(0, prior_std^2 I)`` from ChemBed's nominal prior, decodes to
unique valid canonical SMILES, re-encodes ``mu(x) = Enc(Dec(z))`` back into
256D, and fits ``sklearn.decomposition.PCA(whiten=True)`` to remove posterior-
collapsed null dimensions and equalize active principal-axis scales.

Usage::

    uv run python scripts/fit_latent_pca.py \
        --n-samples 150000 \
        --variance 0.99 \
        --out data/pca_prior.joblib
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import selfies as sf
from chembed import encode as enc
from chembed import mol_utils
from sklearn.decomposition import PCA
from tqdm import tqdm

from vae import MAX_SELFIES_TOKENS, ChemBedVAE


def _to_clean_selfies(smiles: str, vocab_set: set[str]) -> str | None:
    """Standardize *smiles* as ChemBed expects and verify all SELFIES tokens."""
    try:
        std_smi = mol_utils.standardize_smiles(smiles)
        if not std_smi:
            return None
        s = sf.encoder(std_smi)
        if not s or s.count("[") > MAX_SELFIES_TOKENS:
            return None
        if not all(tok in vocab_set for tok in sf.split_selfies(s)):
            return None
        return s
    except Exception:
        return None


def sample_and_reencode(
    vae: ChemBedVAE,
    *,
    n_samples: int,
    prior_std: float,
    batch_size: int,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Generate *n_samples* unique prior molecules and return ``(n_samples, 256)``."""
    rng = np.random.default_rng(seed)
    vocab_set = set(vae._vae.vocab)
    seen: set[str] = set()
    embeddings = np.empty((n_samples, vae.raw_latent_dim), dtype=np.float32)

    counters = {"drawn": 0, "invalid": 0, "duplicate": 0, "skipped_vocab": 0}
    filled = 0

    with tqdm(
        total=n_samples,
        desc="Prior decode -> re-encode",
        unit="mol",
        dynamic_ncols=True,
    ) as pbar:
        while filled < n_samples:
            need = min(chunk_size, n_samples - filled)
            # Oversample slightly to absorb rare duplicates/skips in one pass
            draw = max(batch_size, int(np.ceil(need * 1.02)))
            z_prior = rng.normal(
                0.0, prior_std, size=(draw, vae.raw_latent_dim)
            ).astype(np.float32)
            counters["drawn"] += draw

            decoded = vae.decode(z_prior, batch_size=batch_size)
            clean_selfies: list[str] = []
            for smi in decoded:
                if len(clean_selfies) >= need:
                    break
                if smi is None:
                    counters["invalid"] += 1
                    continue
                if smi in seen:
                    counters["duplicate"] += 1
                    continue
                s = _to_clean_selfies(smi, vocab_set)
                if s is None:
                    counters["skipped_vocab"] += 1
                    continue
                seen.add(smi)
                clean_selfies.append(s)

            if not clean_selfies:
                continue

            zs = enc.encode_multiple_selfies(
                clean_selfies, vae._vae, batch_size=batch_size
            )
            arr = zs.squeeze(1).cpu().numpy().astype(np.float32)
            m = len(arr)
            embeddings[filled : filled + m] = arr
            filled += m
            pbar.update(m)
            pbar.set_postfix(
                dup=counters["duplicate"],
                inv=counters["invalid"],
                skip=counters["skipped_vocab"],
            )

    return embeddings, counters


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-samples",
        type=int,
        default=150_000,
        help="Number of unique valid prior-generated molecules to re-encode",
    )
    p.add_argument(
        "--variance",
        type=float,
        default=0.99,
        help="Target cumulative explained variance in (0, 1) (or integer >= 1)",
    )
    p.add_argument(
        "--prior-std",
        type=float,
        default=1.0,
        help="Standard deviation for prior sampling z ~ N(0, prior_std^2 I)",
    )
    p.add_argument("--batch-size", type=int, default=512, help="VAE GPU batch size")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Outer generation/encoding chunk size",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--vae-repo",
        type=str,
        default="3BioCompBio/chembed-default",
        help="ChemBed HuggingFace repo ID",
    )
    p.add_argument("--device", type=str, default="cuda", help="Torch device")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/pca_prior.joblib"),
        help="Output joblib path for fitted whitened PCA",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Sample prior molecules, re-encode, fit whitened PCA, and save joblib."""
    args = _parse_args(argv)
    n_comp: float | int = (
        int(args.variance) if args.variance >= 1.0 else float(args.variance)
    )

    t0 = time.perf_counter()
    print(f"Loading ChemBed VAE ({args.vae_repo}) on {args.device}...")
    vae = ChemBedVAE(repo_id=args.vae_repo, device=args.device, pca_path=None)

    embeddings, counters = sample_and_reencode(
        vae,
        n_samples=args.n_samples,
        prior_std=args.prior_std,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    print(
        f"Collected {len(embeddings):,} embeddings in {time.perf_counter() - t0:.1f}s "
        f"(drawn={counters['drawn']:,}, dup={counters['duplicate']:,}, "
        f"inv={counters['invalid']:,}, skipped_vocab={counters['skipped_vocab']:,})"
    )

    print(f"Fitting whitened PCA (n_components={n_comp})...")
    pca = PCA(n_components=n_comp, whiten=True, random_state=args.seed)
    pca.fit(embeddings)

    explained = float(pca.explained_variance_ratio_.sum())
    stds = np.sqrt(pca.explained_variance_)
    dead_std = float(np.sqrt(max(pca.noise_variance_, 0.0)))
    print(
        f"PCA components: {pca.n_components_} / {vae.raw_latent_dim} "
        f"({explained:.2%} variance explained)"
    )
    print(
        f"Active std range: [{stds[-1]:.4f}, {stds[0]:.4f}] | "
        f"dropped dims rms std: {dead_std:.5f} | "
        f"||mean||: {np.linalg.norm(pca.mean_):.4f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, args.out)
    print(f"Saved whitened PCA to {args.out}")


if __name__ == "__main__":
    main()
