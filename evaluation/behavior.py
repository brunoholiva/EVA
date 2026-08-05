"""Per-molecule behavior computation: LogP, TPSA, MW, Morgan FP.

Parses each SMILES once and reuses the RDKit Mol across all properties,
avoiding redundant parsing when multiple scorers need Mol-derived values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Descriptors

from chemistry.fingerprint import compute_morgan
from reporting.suppress import suppress_joblib_warnings, suppress_rdkit_logs

suppress_rdkit_logs()
suppress_joblib_warnings()


@dataclass
class MolBehavior:
    """Per-molecule properties computed from a single Mol parse.

    Attributes
    ----------
    logp : float
        Wildman-Crippen LogP.
    tpsa : float
        Topological Polar Surface Area.
    mw : float
        Molecular weight.
    fsp3 : float
        Fraction of sp3-hybridized carbons (saturation / 3D-ness).
    morgan_fp : np.ndarray or None
        Morgan fingerprint as float32 array, or None if parsing failed.
    """

    logp: float
    tpsa: float
    mw: float
    fsp3: float
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

    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    fsp3 = Descriptors.FractionCSP3(mol)
    fp = compute_morgan(mol, radius=radius, fp_size=n_bits)

    return MolBehavior(logp=logp, tpsa=tpsa, mw=mw, fsp3=fsp3, morgan_fp=fp)


def batch_molecule_behaviors(
    smiles: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    n_jobs: int = -1,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
]:
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
    logp : np.ndarray of shape ``(len(smiles),)``
        LogP values.  Invalid SMILES receive NaN.
    tpsa : np.ndarray of shape ``(len(smiles),)``
        TPSA values.  Invalid SMILES receive NaN.
    mw : np.ndarray of shape ``(len(smiles),)``
        Molecular weight values.  Invalid SMILES receive NaN.
    fsp3 : np.ndarray of shape ``(len(smiles),)``
        Fraction of sp3 carbons.  Invalid SMILES receive NaN.
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
    logp = np.full(n, float("nan"), dtype=np.float64)
    tpsa = np.full(n, float("nan"), dtype=np.float64)
    mw = np.full(n, float("nan"), dtype=np.float64)
    fsp3 = np.full(n, float("nan"), dtype=np.float64)
    fps_list: list[np.ndarray] = []
    valid_fp_indices: list[int] = []

    for i, mb in enumerate(results):
        logp[i] = mb.logp
        tpsa[i] = mb.tpsa
        mw[i] = mb.mw
        fsp3[i] = mb.fsp3
        if mb.morgan_fp is not None:
            fps_list.append(mb.morgan_fp)
            valid_fp_indices.append(i)

    valid_fp_mask = np.zeros(n, dtype=bool)
    valid_fp_mask[valid_fp_indices] = True

    if not fps_list:
        return logp, tpsa, mw, fsp3, None, valid_fp_mask

    fps = np.array(fps_list, dtype=np.float32)
    return logp, tpsa, mw, fsp3, fps, valid_fp_mask
