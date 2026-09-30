"""SMILES canonicalization utilities."""

from __future__ import annotations

from rdkit import Chem

from logs import suppress_rdkit_logs
import selfies as sf
import numpy as np
from joblib import Parallel, delayed

suppress_rdkit_logs()


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES or None if invalid.

    Parameters
    ----------
    smiles : str
        Input SMILES string.

    Returns
    -------
    str or None
        Canonical SMILES, or None if parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def canonicalize_batch(smiles_list: list[str]) -> list[str]:
    """Canonicalize a list of SMILES, dropping invalid ones.

    Parameters
    ----------
    smiles_list : list of str
        Input SMILES strings.

    Returns
    -------
    list of str
        Canonical SMILES (invalid entries are omitted).
    """
    result = []
    for smi in smiles_list:
        canon = canonicalize_smiles(smi)
        if canon is not None:
            result.append(canon)
    return result


MAX_SELFIES_TOKENS = 200
"""Reject SELFIES longer than this many tokens — they never decode to drugs."""


def parse_and_canonical(smiles: str) -> tuple[bool, str | None]:
    """Parse a SMILES string and return ``(valid, canonical)``.

    Returns ``(False, None)`` for empty strings, failed RDKit parsing, SELFIES
    token counts exceeding *MAX_SELFIES_TOKENS*, or SMILES that RDKit parses
    but fails to canonicalize (``MolToSmiles`` invariant violations on
    degenerate molecules).
    """
    if not smiles:
        return False, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, None
    try:
        selfies = sf.encoder(smiles)
        tokens = list(sf.split_selfies(selfies))
        if len(tokens) > MAX_SELFIES_TOKENS:
            return False, None
    except Exception:
        return False, None
    try:
        canonical = Chem.MolToSmiles(mol)
    except Exception:
        return False, None
    return True, canonical


def parse_and_canonical_batch(
    smiles: list[str], n_jobs: int = -1
) -> tuple[np.ndarray, list[str | None]]:
    """Return ``(valid_mask, canonical)`` for a batch of SMILES strings.

    The validity check (including canonicalization) runs in parallel.
    Deduplication is *not* done here — it is handled by the caller via a
    ``seen`` set.
    """
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(parse_and_canonical)(s) for s in smiles
    )
    valid = np.array([r[0] for r in results], dtype=bool)
    canonical = [r[1] for r in results]
    return valid, canonical
