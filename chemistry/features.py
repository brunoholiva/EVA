"""Molecular featurization: Morgan fingerprints + RDKit descriptors + MACCS."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from chemistry.fingerprint import compute_morgan
from reporting.console import console
from reporting.suppress import suppress_joblib_warnings

suppress_joblib_warnings()


N_RDKIT_DESCRIPTORS: int = 200
N_MACCS: int = 167


def _chunk_list(lst: list, chunks: int) -> list[list]:
    """Split a list into roughly equal parts."""
    k, m = divmod(len(lst), chunks)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(chunks)]


def _compute_rdkit_descriptors(mol: Chem.Mol, smiles: str, generator) -> np.ndarray:
    """Return 200 RDKit 2D normalized descriptors as a float32 array."""
    try:
        res = generator.processMol(mol, smiles, internalParsing=True)
        if res[0]:
            return np.array(res[1:], dtype=np.float32)
    except Exception:
        pass
    return np.zeros(N_RDKIT_DESCRIPTORS, dtype=np.float32)


def _compute_maccs(mol: Chem.Mol) -> np.ndarray:
    """Return 167-bit MACCS keys as a float32 array."""
    maccs_vec = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros(N_MACCS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(maccs_vec, maccs_arr)
    return maccs_arr


def _compute_features_batch(
    smiles_batch: list[str], n_bits: int = 512, radius: int = 2
) -> np.ndarray:
    """Compute Morgan + RDKit + MACCS features for a batch of SMILES.

    Parses each SMILES once and reuses the RDKit Mol object for the
    Morgan fingerprint, the 200 RDKit normalised descriptors, and the
    167-bit MACCS structural keys.  Molecules that fail or return a bad
    status get a zero vector.

    Creates one ``RDKit2DNormalized`` instance per process
    to avoid pickling issues with joblib.
    """
    from descriptastorus.descriptors import rdNormalizedDescriptors

    rdkit_gen = rdNormalizedDescriptors.RDKit2DNormalized()
    n_features = n_bits + N_RDKIT_DESCRIPTORS + N_MACCS

    out = np.empty((len(smiles_batch), n_features), dtype=np.float32)
    for i, smi in enumerate(smiles_batch):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out[i] = 0.0
        else:
            out[i, :n_bits] = compute_morgan(mol, radius=radius, fp_size=n_bits)
            out[i, n_bits : n_bits + N_RDKIT_DESCRIPTORS] = _compute_rdkit_descriptors(
                mol, smi, rdkit_gen
            )
            out[i, n_bits + N_RDKIT_DESCRIPTORS :] = _compute_maccs(mol)
    return out


class MoleculeFeaturizer:
    """Convert SMILES to a concatenated feature matrix.

    The matrix is built by concatenating a **Morgan fingerprint** (512-bit,
    radius 2) with 200 **RDKit normalised descriptors** (computed via
    ``descriptastorus``) and 167-bit **MACCS structural keys**.

    Parameters
    ----------
    n_bits : int, default=512
        Number of bits for the Morgan fingerprint.
    radius : int, default=2
        Morgan fingerprint radius.
    n_jobs : int, default=-1
        Number of parallel workers for the combined Morgan + RDKit descriptor
        computation.  ``-1`` uses all available CPUs.
    """

    def __init__(self, n_bits: int = 512, radius: int = 2, n_jobs: int = -1):
        self.n_bits = n_bits
        self.radius = radius
        self.n_jobs = n_jobs

    @property
    def n_features(self) -> int:
        return self.n_bits + N_RDKIT_DESCRIPTORS + N_MACCS

    def transform(
        self,
        X: list[str],
        progress: Progress | None = None,
        task_id: int | None = None,
    ) -> np.ndarray:
        """Featurise a list of SMILES into a ``(n, n_features)`` feature matrix.

        Parameters
        ----------
        X : list of str
            SMILES strings.
        progress : Progress or None, default=None
            Optional external progress bar to update. If None, creates a local bar.
        task_id : int or None, default=None
            Task ID in the external progress bar to update.

        Returns
        -------
        np.ndarray of shape ``(len(X), n_features)``
            Concatenated Morgan (``n_bits``) + RDKit (``N_RDKIT_DESCRIPTORS``)
            + MACCS (``N_MACCS``) features.
        """
        n = len(X)
        if n == 0:
            return np.empty((0, self.n_features), dtype=np.float32)

        njobs = effective_n_jobs(self.n_jobs)
        batches = _chunk_list(X, njobs * 2)

        # Use external progress bar if provided
        own_progress = progress is None
        if own_progress:
            columns = [
                TextColumn("  "),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None, style="white", complete_style="magenta"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ]
            progress = Progress(*columns, console=console)
            task_id = progress.add_task("[magenta]Featurizing molecules", total=n)
            progress.start()

        try:
            results = Parallel(n_jobs=njobs, return_as="generator")(
                delayed(_compute_features_batch)(batch, self.n_bits, self.radius)
                for batch in batches
            )
            parts = []
            for r in results:
                parts.append(r)
                progress.advance(task_id, advance=len(r))
            return np.vstack(parts)
        finally:
            if own_progress:
                progress.stop()
