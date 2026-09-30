"""Parsed molecules shared across the scoring pipeline."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from reporting.suppress import suppress_rdkit_logs

suppress_rdkit_logs()


def _scaffold_smiles(mol: Chem.Mol | None) -> str | None:
    """Return the Murcko scaffold SMILES for *mol*, or None."""
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
            return None
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


class ParsedMolecules:
    """SMILES parsed once, with lazily computed Murcko scaffolds.

    Parameters
    ----------
    smiles : list[str]
        List of SMILES strings to parse.
    """

    def __init__(self, smiles: list[str]):
        self.smiles = smiles
        self.mols = [Chem.MolFromSmiles(s) if s else None for s in smiles]
        self._scaffold_smiles: list[str | None] | None = None

    @property
    def scaffold_smiles(self) -> list[str | None]:
        """Lazy-computed Murcko scaffold SMILES.

        Returns
        -------
        list[str | None]
            Murcko scaffold SMILES per molecule. None for invalid molecules
            or those without a scaffold (acyclic).
        """
        if self._scaffold_smiles is None:
            self._scaffold_smiles = [_scaffold_smiles(mol) for mol in self.mols]
        return self._scaffold_smiles
