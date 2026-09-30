"""Train a TabPFN v3.5 classifier on a binary classification dataset.

The representation must match the model used by EVA at inference time.

Usage::

    micromamba run -n eva2 python scripts/train_tabpfn.py \\
        --features-dir /path/to/brood/data/features/GNEtolC/rdkit2d \\
        --out data/predictor/tabpfn_rdkit2d.joblib \\
        --representation rdkit2d
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

from chemistry.featurizers import make_activity_featurizer
from evaluation.activity import move_model_to_device
from logs import suppress_rdkit_logs
from reporting.console import detail, saved, section, step

suppress_rdkit_logs()

SOFTMAX_TEMPERATURE: float = 2.0


def load_data(csv_path: Path) -> tuple[list[str], np.ndarray]:
    """Load SMILES and binary targets from a CSV file.

    Auto-detects column names: accepts ``SMILES`` or ``standardized_smiles``
    for the SMILES column, and ``target`` or ``antimicrobial_activity`` for
    the binary label.

    Parameters
    ----------
    csv_path : Path
        Path to CSV with SMILES and binary target columns.

    Returns
    -------
    smiles : list of str
        SMILES strings.
    y : np.ndarray
        Binary target array.
    """
    df = pd.read_csv(csv_path)

    smiles_col = next(
        (c for c in ("SMILES", "standardized_smiles") if c in df.columns), None
    )
    if smiles_col is None:
        raise ValueError(
            f"No SMILES column found in {csv_path}. "
            "Expected 'SMILES' or 'standardized_smiles'."
        )

    target_col = next(
        (c for c in ("target", "antimicrobial_activity") if c in df.columns), None
    )
    if target_col is None:
        raise ValueError(
            f"No target column found in {csv_path}. "
            "Expected 'target' or 'antimicrobial_activity'."
        )

    data = df[[smiles_col, target_col]].dropna()
    smiles = data[smiles_col].tolist()
    y = data[target_col].to_numpy()
    return smiles, y


def train_model(X: np.ndarray, y: np.ndarray) -> TabPFNClassifier:
    """Fit a TabPFN classifier with KV-cache.

    Parameters
    ----------
    X : np.ndarray of shape (n, n_features)
        Feature matrix.
    y : np.ndarray of shape (n,)
        Binary target array.

    Returns
    -------
    TabPFNClassifier
        Fitted classifier with cached KV representations.
    """
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5,
        softmax_temperature=SOFTMAX_TEMPERATURE,
        device="cuda",
        random_state=42,
        fit_mode="fit_with_cache",
        show_progress_bar=True,
    )
    model.fit(X, y)
    return model


def train_one(
    out_path: Path,
    representation: str,
    csv_path: Path | None = None,
    features_dir: Path | None = None,
    n_jobs: int = -1,
) -> None:
    """Train a TabPFN model on a single dataset and save it.

    Parameters
    ----------
    out_path : Path
        Output path for the joblib model artifact.
    csv_path : Path or None
        Path to training CSV (``SMILES``, ``target`` columns).
    features_dir : Path or None
        Brood feature cache containing ``X_train.npy`` and ``y_train.npy``.
    n_jobs : int, default=-1
        Parallel workers for featurisation.
    """
    if features_dir is not None:
        section(f"Training TabPFN on cached {representation} features")
        step(f"Loading {features_dir} ...")
        X = np.load(features_dir / "X_train.npy")
        y = np.load(features_dir / "y_train.npy")
        n_samples = len(y)
    else:
        if csv_path is None:
            raise ValueError("Provide csv_path or features_dir")
        section(f"Training TabPFN on {csv_path.name}")
        step(f"Loading {csv_path} ...")
        smiles, y = load_data(csv_path)
        featurizer = make_activity_featurizer(
            representation, n_jobs=n_jobs, device="cuda"
        )
        step(f"Computing {representation} features ...")
        X = featurizer.transform(smiles)
        n_samples = len(smiles)

    n_pos = int(y.sum())
    detail(
        f"Loaded {n_samples:,} molecules  "
        f"({n_pos:,} positive, {len(y) - n_pos:,} negative)"
    )
    detail(f"Feature matrix shape: {X.shape}")

    step(
        f"Training TabPFN v3.5 ({representation}, "
        f"softmax_temperature={SOFTMAX_TEMPERATURE}, "
        f"fit_mode=fit_with_cache) ..."
    )
    t0 = time.perf_counter()
    model = train_model(X, y)
    elapsed = time.perf_counter() - t0
    detail(f"Training completed in {elapsed:.1f}s")

    step("Moving model to CPU (including KV-cache) ...")
    move_model_to_device(model, "cpu")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    saved("TabPFN model", out_path, f"{out_path.stat().st_size / 1e9:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Training CSV")
    source.add_argument(
        "--features-dir",
        type=Path,
        help="Brood feature cache containing X_train.npy and y_train.npy",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output joblib path")
    parser.add_argument(
        "--representation",
        choices=("rdkit2d", "chemeleon"),
        required=True,
        help="Feature representation used for training",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1, help="Parallel workers for featurisation"
    )
    args = parser.parse_args()

    train_one(
        args.out,
        args.representation,
        csv_path=args.csv,
        features_dir=args.features_dir,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
