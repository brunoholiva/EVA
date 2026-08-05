"""Build the applicability domain kNN model from training SMILES.

Fits a sklearn NearestNeighbors (Jaccard distance) on Morgan fingerprints
and saves the model + raw fingerprints as a joblib artifact.

Usage::

    python featurization/build_ad_model.py \\
        --csv data/predictor/predictor_training_data.csv \\
        --out data/predictor/ad_model.joblib
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from chemistry.fingerprint import compute_morgan
from reporting.console import console, detail, saved, step
from reporting.suppress import suppress_rdkit_logs

suppress_rdkit_logs()

N_BITS_DEFAULT: int = 2048
RADIUS_DEFAULT: int = 2
N_NEIGHBORS_DEFAULT: int = 5


def _smiles_to_fp(
    smiles: str, radius: int = RADIUS_DEFAULT, n_bits: int = N_BITS_DEFAULT
) -> np.ndarray | None:
    """Convert a SMILES string to a Morgan fingerprint numpy array.

    Parameters
    ----------
    smiles : str
        SMILES string.
    radius : int
        Morgan fingerprint radius.
    n_bits : int
        Fingerprint bit length.

    Returns
    -------
    np.ndarray of shape ``(n_bits,)`` or None
        Binary fingerprint as uint8, or None if SMILES is invalid.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return compute_morgan(mol, radius=radius, fp_size=n_bits).astype(np.uint8)


def build_model(
    csv_path: Path,
    smiles_col: str,
    out_path: Path,
    radius: int,
    n_bits: int,
    n_neighbors: int,
) -> None:
    """Build and save the kNN applicability domain model.

    Parameters
    ----------
    csv_path : Path
        Path to CSV with a SMILES column.
    smiles_col : str
        Name of the SMILES column in the CSV.
    out_path : Path
        Output path for the joblib artifact.
    radius : int
        Morgan fingerprint radius.
    n_bits : int
        Fingerprint bit length.
    n_neighbors : int
        Number of nearest neighbors for the kNN model.
    """
    df = pd.read_csv(csv_path)
    smiles_list = df[smiles_col].dropna().tolist()
    step(f"Loaded {len(smiles_list):,} SMILES from {csv_path}")

    columns = [
        TextColumn("  "),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="white", complete_style="magenta"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ]

    fps: list[np.ndarray] = []
    n_invalid = 0

    with Progress(*columns, console=console) as progress:
        task = progress.add_task(
            "[magenta]Computing fingerprints", total=len(smiles_list)
        )
        for smi in smiles_list:
            fp = _smiles_to_fp(smi, radius=radius, n_bits=n_bits)
            if fp is not None:
                fps.append(fp)
            else:
                n_invalid += 1
            progress.advance(task)

    fps_arr = np.array(fps, dtype=np.uint8)
    detail(
        f"Valid fingerprints: {len(fps):,}  |  Invalid SMILES skipped: {n_invalid:,}"
    )

    from sklearn.neighbors import NearestNeighbors

    t0 = time.perf_counter()
    nn = NearestNeighbors(metric="jaccard", n_neighbors=n_neighbors)
    nn.fit(fps_arr.astype(bool))
    elapsed = time.perf_counter() - t0
    detail(f"kNN fit ({n_neighbors} neighbors, Jaccard): {elapsed:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "nn": nn,
            "fps": fps_arr,
            "radius": radius,
            "n_bits": n_bits,
            "n_neighbors": n_neighbors,
        },
        out_path,
    )
    saved("AD model", out_path, f"{out_path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to training CSV")
    parser.add_argument("--smiles_col", default="SMILES", help="Column name for SMILES")
    parser.add_argument("--out", required=True, help="Output joblib path")
    parser.add_argument("--radius", type=int, default=RADIUS_DEFAULT)
    parser.add_argument("--n_bits", type=int, default=N_BITS_DEFAULT)
    parser.add_argument("--n_neighbors", type=int, default=N_NEIGHBORS_DEFAULT)
    args = parser.parse_args()

    build_model(
        csv_path=Path(args.csv),
        smiles_col=args.smiles_col,
        out_path=Path(args.out),
        radius=args.radius,
        n_bits=args.n_bits,
        n_neighbors=args.n_neighbors,
    )


if __name__ == "__main__":
    main()
