"""Shared Morgan fingerprint computation.


Generators are cached by ``(radius, fp_size)`` to avoid re-instantiation.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

_GEN_CACHE: dict[tuple[int, int], object] = {}


def _get_generator(radius: int, fp_size: int) -> object:
    """Return a cached MorganGenerator for the given parameters."""
    key = (radius, fp_size)
    if key not in _GEN_CACHE:
        _GEN_CACHE[key] = GetMorganGenerator(radius=radius, fpSize=fp_size)
    return _GEN_CACHE[key]


def compute_morgan(
    mol: Chem.Mol,
    radius: int = 2,
    fp_size: int = 2048,
) -> np.ndarray:
    """Compute a single Morgan fingerprint as a float32 array.

    Parameters
    ----------
    mol : Chem.RWMol
        RDKit molecule object.
    radius : int, default=2
        Morgan fingerprint radius.
    fp_size : int, default=2048
        Fingerprint bit length.

    Returns
    -------
    np.ndarray of shape ``(fp_size,)``
        Binary fingerprint stored as float32.
    """
    gen = _get_generator(radius, fp_size)
    return np.array(gen.GetFingerprint(mol), dtype=np.float32)


def _compute_single_fp(
    smi: str, radius: int, fp_size: int
) -> tuple[np.ndarray | None, int]:
    """Compute Morgan FP for one SMILES. Returns (fp or None, original_index)."""
    if not smi or not smi.strip():
        return None, -1
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, -1
    return compute_morgan(mol, radius=radius, fp_size=fp_size), -1


def smiles_to_morgan(
    smiles: list[str],
    radius: int = 2,
    fp_size: int = 2048,
    n_jobs: int = -1,
) -> tuple[np.ndarray, list[int]]:
    """Convert SMILES strings to Morgan fingerprints in batch.

    Invalid or empty SMILES are skipped. The returned array contains only
    valid fingerprints; *valid_idx* maps each row back to its original
    position in the input list.

    Parameters
    ----------
    smiles : list of str
        SMILES strings to featurize.
    radius : int, default=2
        Morgan fingerprint radius.
    fp_size : int, default=2048
        Fingerprint bit length.
    n_jobs : int, default=-1
        Number of parallel workers. -1 uses all CPUs.

    Returns
    -------
    fps : np.ndarray of shape ``(n_valid, fp_size)`` and dtype float32
        Fingerprints for valid molecules.
    valid_idx : list of int
        Original indices of valid molecules.
    """
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_compute_single_fp)(smi, radius, fp_size) for smi in smiles
    )
    fps: list[np.ndarray] = []
    valid_idx: list[int] = []
    for i, (fp, _) in enumerate(results):
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    if not fps:
        return np.empty((0, fp_size), dtype=np.float32), valid_idx
    return np.array(fps, dtype=np.float32), valid_idx
