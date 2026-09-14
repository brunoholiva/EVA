"""Parsed molecules with lazy fingerprint computation."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from chemistry.fingerprint import mols_to_morgan
from reporting.suppress import suppress_rdkit_logs

suppress_rdkit_logs()


class ParsedMolecules:
    """Shared parsed molecules with lazy fingerprint computation.

    Parses SMILES once and caches the resulting Mol objects. Fingerprints
    are computed lazily on first access to avoid unnecessary computation.

    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to parse.
    """

    def __init__(self, smiles: list[str]):
        self.smiles = smiles
        self.mols = [Chem.MolFromSmiles(s) if s else None for s in smiles]
        self._fingerprints: np.ndarray | None = None
        self._valid_fp_mask: np.ndarray | None = None
        self._scaffold_smiles: list[str | None] | None = None

    @property
    def fingerprints(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Lazy-computed Morgan fingerprints.

        Returns
        -------
        tuple[np.ndarray | None, np.ndarray | None]
            Tuple of (fingerprints, valid_mask). Fingerprints is None if
            no valid molecules. Valid mask indicates which SMILES produced
            valid fingerprints.
        """
        if self._fingerprints is None:
            self._fingerprints, self._valid_fp_mask = mols_to_morgan(self.mols)

        return self._fingerprints, self._valid_fp_mask

    @property
    def scaffold_smiles(self) -> list[str | None]:
        """Lazy-computed Murcko scaffold SMILES.

        Returns
        -------
        list[str | None]
            Murcko scaffold SMILES for each molecule. None for invalid
            molecules or those without a scaffold (acyclic).
        """
        if self._scaffold_smiles is None:
            self._scaffold_smiles = []
            for mol in self.mols:
                if mol is None:
                    self._scaffold_smiles.append(None)
                else:
                    try:
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
                            self._scaffold_smiles.append(None)
                        else:
                            self._scaffold_smiles.append(Chem.MolToSmiles(scaffold))
                    except Exception:
                        self._scaffold_smiles.append(None)

        return self._scaffold_smiles
