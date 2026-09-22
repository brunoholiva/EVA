"""Generic Murcko scaffold utilities for molecular diversity analysis."""

from __future__ import annotations

from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric


def generic_scaffold_hash(smiles: str) -> str:
    """Return canonical SMILES of the generic Murcko scaffold.

    Uses ``GetScaffoldForMol`` then ``MakeScaffoldGeneric`` to collapse
    functional-group differences.
    Acyclic molecules return ``"acyclic"``; parse failures return ``"invalid"``.

    Parameters
    ----------
    smiles : str
        SMILES string.

    Returns
    -------
    str
        Canonical SMILES of the generic scaffold, or a fallback tag.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    scaffold = GetScaffoldForMol(mol)
    if not scaffold or not scaffold.GetRingInfo().NumRings():
        return "acyclic"
    generic = MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(generic)


def cluster_by_generic_scaffold(smiles_list: list[str]) -> list[set[int]]:
    """Cluster SMILES by generic Murcko scaffold hash.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings.

    Returns
    -------
    list of set[int]
        Each set contains the indices of SMILES sharing the same
        generic scaffold hash.
    """
    buckets: dict[str, set[int]] = defaultdict(set)
    for i, smi in enumerate(smiles_list):
        buckets[generic_scaffold_hash(smi)].add(i)
    return [group for group in buckets.values() if len(group) > 0]
