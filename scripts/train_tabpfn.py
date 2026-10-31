r"""Train a TabPFN v3.5 classifier on MiniMol molecular fingerprints.

Usage::

    uv run python scripts/train_tabpfn.py \
        --csv data/predictor/predictor_training_data.csv \
        --out data/predictor/tabpfn_minimol.joblib
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from oracle import MinimolFeaturizer, move_model_to_device


def load_data(csv_path: Path) -> tuple[list[str], np.ndarray]:
    """Load SMILES and binary activity labels from *csv_path*."""
    df = pd.read_csv(csv_path)
    smiles_col = next(
        (c for c in ("SMILES", "standardized_smiles") if c in df.columns), None
    )
    if smiles_col is None:
        raise ValueError(f"No SMILES column in {csv_path}")

    target_col = next(
        (c for c in ("target", "antimicrobial_activity") if c in df.columns), None
    )
    if target_col is None:
        raise ValueError(f"No target column in {csv_path}")

    data = df[[smiles_col, target_col]].dropna()
    return data[smiles_col].tolist(), data[target_col].to_numpy()


def main() -> None:
    """Featurize training molecules (or load cached arrays) and fit TabPFN v3.5."""
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="Training CSV with SMILES and target")
    src.add_argument(
        "--features-dir", type=Path, help="Dir with X_train.npy/y_train.npy"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output joblib path")
    args = parser.parse_args()

    if args.features_dir is not None:
        X = np.load(args.features_dir / "X_train.npy")
        y = np.load(args.features_dir / "y_train.npy")
    else:
        smiles, y = load_data(args.csv)
        print(f"Computing MiniMol fingerprints for {len(smiles):,} molecules...")
        X = MinimolFeaturizer().transform(smiles)

    n_pos = int(y.sum())
    print(
        f"Dataset: {len(y):,} molecules ({n_pos:,} pos, {len(y) - n_pos:,} neg), X={X.shape}"
    )

    t0 = time.perf_counter()
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5,
        device="cuda",
        random_state=42,
        fit_mode="fit_with_cache",
        show_progress_bar=True,
    )
    model.fit(X, y)
    print(
        f"Fit completed in {time.perf_counter() - t0:.1f}s; moving KV-cache to CPU..."
    )

    move_model_to_device(model, "cpu")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Saved model to {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
