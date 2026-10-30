"""Shared chemistry utilities for EVA diagnostic notebooks."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.DisableLog("rdApp.*")


def compute_scaffold(smiles: str) -> str | None:
    """Return Murcko scaffold SMILES, or None if invalid/acyclic."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold)


def score_smiles(smiles_list: list[str], model) -> np.ndarray:
    """Featurize SMILES and return P(active) from a fitted TabPFN model."""
    from chemistry.featurizers import MoleculeFeaturizer
    from evaluation.activity import predict_from_features

    featurizer = MoleculeFeaturizer()
    X = featurizer.transform(smiles_list)
    _, probs = predict_from_features(X, model)
    return probs[:, 1]


BENZOIC_ACID = Chem.MolFromSmarts("[c]1ccccc1C(=O)O")
HALOAROMATIC = Chem.MolFromSmarts("[c][F,Cl,Br,I]")


def check_halogenated_benzoic_acid(smiles: str) -> bool:
    """Return True if molecule contains a halogenated benzoic acid."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return False

    acid_matches = mol.GetSubstructMatches(BENZOIC_ACID)

    for acid_match in acid_matches:
        # atoms belonging to the benzene ring
        ring_atoms = set(acid_match[:6])

        # Check whether any of those same ring atoms has a halogen attached
        for atom_idx in ring_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)

            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() in {9, 17, 35, 53}:  # F, Cl, Br, I
                    return True

    return False


def compute_morgan_fps(
    smiles_list: list[str], radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """Compute Morgan fingerprints for a list of SMILES.

    Returns
    -------
    np.ndarray of shape ``(n_valid, n_bits)``
        Only valid molecules are included.
    """
    from chemistry.fingerprint import smiles_to_morgan

    fps, valid_idx = smiles_to_morgan(smiles_list, radius=radius, fp_size=n_bits)
    return fps


def max_tanimoto_to_set(
    smiles_list: list[str], reference_smiles: set[str] | list[str]
) -> tuple[np.ndarray, list[str]]:
    """For each SMILES, find the max Tanimoto similarity to the reference set.

    Parameters
    ----------
    smiles_list : list of str
        Query SMILES strings.
    reference_smiles : set or list of str
        Reference SMILES strings.

    Returns
    -------
    max_sims : np.ndarray of shape ``(len(smiles_list),)``
        Max Tanimoto similarity for each query.
    most_similar : list of str
        The reference SMILES with highest similarity for each query.
    """
    from rdkit.Chem import AllChem

    ref_mols = []
    ref_smi_list = list(reference_smiles)
    for s in ref_smi_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            ref_mols.append((mol, s))

    ref_fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        for m, _ in ref_mols
    ]

    max_sims = np.zeros(len(smiles_list), dtype=np.float32)
    most_similar = [""] * len(smiles_list)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        sims = BulkTanimotoSimilarity(fp, ref_fps)
        best = int(np.argmax(sims))
        max_sims[i] = sims[best]
        most_similar[i] = ref_mols[best][1]

    return max_sims, most_similar
