"""SMILES canonicalization utilities."""

from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


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
    return Chem.MolToSmiles(mol)


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
