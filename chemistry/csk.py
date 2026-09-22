from __future__ import annotations

from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdMolHash
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol


def csk_hash(smiles: str) -> str:
    """Return the cyclic skeleton hash for a SMILES string.

    Uses the Murcko scaffold's anonymous graph topology, which
    strips element types and keeps only ring connectivity.
    Acyclic molecules return ``"acyclic"``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    scaffold = GetScaffoldForMol(mol)
    if not scaffold or not scaffold.GetRingInfo().NumRings():
        return "acyclic"
    return rdMolHash.MolHash(scaffold, rdMolHash.HashFunction.AnonymousGraph)


def cluster_by_csk(smiles_list: list[str]) -> list[set[int]]:
    """Cluster SMILES by cyclic skeleton (CSK) hash.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings.

    Returns
    -------
    list of set[int]
        Each set contains the indices of SMILES sharing the same
        CSK hash.
    """
    buckets: dict[str, set[int]] = defaultdict(set)
    for i, smi in enumerate(smiles_list):
        buckets[csk_hash(smi)].add(i)
    return [group for group in buckets.values() if len(group) > 0]
