"""Per-molecule behavior computation: BR-SAScore, LogP, TPSA, MW, Morgan FP.

Parses each SMILES once and reuses the RDKit Mol across all properties,
avoiding redundant parsing when multiple scorers need Mol-derived values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Descriptors

from featurization.morgan import compute_morgan
from scoring.br_sascore import _get_scorer


@dataclass
class MolBehavior:
    """Per-molecule properties computed from a single Mol parse.

    Attributes
    ----------
    br_sascore : float
        BR-SAScore (retrosynthetic accessibility). ``nan`` if scoring failed.
    logp : float
        Wildman-Crippen LogP.
    tpsa : float
        Topological Polar Surface Area.
    mw : float
        Molecular weight.
    morgan_fp : np.ndarray or None
        Morgan fingerprint as float32 array, or None if parsing failed.
    """

    br_sascore: float
    logp: float
    tpsa: float
    mw: float
    morgan_fp: np.ndarray | None


def compute_molecule_behavior(
    smi: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> MolBehavior:
    """Compute per-molecule properties from a SMILES string.

    Parameters
    ----------
    smi : str
        SMILES string of the molecule.
    radius : int, default=2
        Morgan fingerprint radius.
    n_bits : int, default=2048
        Morgan fingerprint bit length.

    Returns
    -------
    MolBehavior
        Properties for the molecule. Invalid or empty SMILES produce NaN
        values and a ``None`` fingerprint.
    """
    if not smi or not smi.strip():
        return MolBehavior(float("nan"), float("nan"), float("nan"), float("nan"), None)

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return MolBehavior(float("nan"), float("nan"), float("nan"), float("nan"), None)

    try:
        br, _ = _get_scorer().calculateScore(smi)
    except Exception:
        br = float("nan")

    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    fp = compute_morgan(mol, radius=radius, fp_size=n_bits)

    return MolBehavior(br_sascore=br, logp=logp, tpsa=tpsa, mw=mw, morgan_fp=fp)


def batch_molecule_behaviors(
    smiles: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Compute MolBehavior for a batch of SMILES in parallel.

    Parameters
    ----------
    smiles : list of str
        SMILES strings to score.
    radius : int, default=2
        Morgan fingerprint radius.
    n_bits : int, default=2048
        Morgan fingerprint bit length.
    n_jobs : int, default=-1
        Number of parallel workers. -1 uses all CPUs.

    Returns
    -------
    br : np.ndarray of shape ``(len(smiles),)``
        BR-SAScore values.  Invalid SMILES receive NaN.
    logp : np.ndarray of shape ``(len(smiles),)``
        LogP values.  Invalid SMILES receive NaN.
    tpsa : np.ndarray of shape ``(len(smiles),)``
        TPSA values.  Invalid SMILES receive NaN.
    mw : np.ndarray of shape ``(len(smiles),)``
        Molecular weight values.  Invalid SMILES receive NaN.
    fps : np.ndarray of shape ``(n_valid_fps, n_bits)`` or None
        Morgan fingerprints for molecules where parsing succeeded.
    valid_fp_mask : np.ndarray of bool, shape ``(len(smiles),)``
        Boolean mask indicating which input positions produced a valid
        fingerprint (used for mapping Tanimoto results back).
    """
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(compute_molecule_behavior)(smi, radius, n_bits) for smi in smiles
    )

    n = len(smiles)
    br = np.full(n, float("nan"), dtype=np.float64)
    logp = np.full(n, float("nan"), dtype=np.float64)
    tpsa = np.full(n, float("nan"), dtype=np.float64)
    mw = np.full(n, float("nan"), dtype=np.float64)
    fps_list: list[np.ndarray] = []
    valid_fp_indices: list[int] = []

    for i, mb in enumerate(results):
        br[i] = mb.br_sascore
        logp[i] = mb.logp
        tpsa[i] = mb.tpsa
        mw[i] = mb.mw
        if mb.morgan_fp is not None:
            fps_list.append(mb.morgan_fp)
            valid_fp_indices.append(i)

    valid_fp_mask = np.zeros(n, dtype=bool)
    valid_fp_mask[valid_fp_indices] = True

    if not fps_list:
        return br, logp, tpsa, mw, None, valid_fp_mask

    fps = np.array(fps_list, dtype=np.float32)
    return br, logp, tpsa, mw, fps, valid_fp_mask
