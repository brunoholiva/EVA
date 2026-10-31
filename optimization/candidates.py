"""Flat candidate hit store and generic Murcko scaffold tracking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from chemistry.scaffolds import generic_scaffold_hash

CANDIDATE_COLUMNS = [
    "rank",
    "canonical_smiles",
    "smiles",
    "p_active",
    "generic_scaffold",
    "walker_id",
    "step",
    "oracle_call",
]


@dataclass(frozen=True)
class CandidateRecord:
    """Metadata for a unique valid candidate exceeding ``hit_threshold``."""

    canonical_smiles: str
    smiles: str
    p_active: float
    generic_scaffold: str
    walker_id: int
    step: int
    oracle_call: int


class CandidateStore:
    """Stores unique active hits and maintains O(1) scaffold/threshold counters.

    Parameters
    ----------
    hit_threshold : float, default=0.5
        Minimum ``P(active)`` required to store a full :class:`CandidateRecord`
        for ``candidates.csv``. Cumulative counters at ``0.5``, ``0.6``, and
        ``0.8`` are always tracked regardless of *hit_threshold*.
    """

    def __init__(self, hit_threshold: float = 0.5) -> None:
        self.hit_threshold = hit_threshold
        self._records: dict[str, CandidateRecord] = {}
        self.best_p_active: float = 0.0
        self.n_hits_05: int = 0
        self.n_hits_06: int = 0
        self.n_hits_08: int = 0
        self._scaffold_counts_06: Counter[str] = Counter()

    def __len__(self) -> int:
        """Return number of stored candidate records."""
        return len(self._records)

    @property
    def unique_scaffolds_06(self) -> int:
        """Number of unique generic Murcko scaffolds among ``P >= 0.6`` hits."""
        return len(self._scaffold_counts_06)

    @property
    def largest_scaffold_frac_06(self) -> float:
        """Fraction of ``P >= 0.6`` hits belonging to the largest scaffold cluster."""
        if self.n_hits_06 == 0 or not self._scaffold_counts_06:
            return 0.0
        largest = self._scaffold_counts_06.most_common(1)[0][1]
        return float(largest / self.n_hits_06)

    def record_scored(
        self,
        *,
        canonical_smiles: str,
        smiles: str,
        p_active: float,
        walker_id: int,
        step: int,
        oracle_call: int,
    ) -> None:
        """Ingest one first-seen valid oracle-evaluated molecule.

        Parameters
        ----------
        canonical_smiles : str
            Canonical SMILES string (unique key).
        smiles : str
            Decoded SMILES string.
        p_active : float
            Predicted ``P(active)`` in ``[0, 1]``.
        walker_id : int
            Walker that proposed the candidate (``-1`` for stateless solvers).
        step : int
            Zero-based generation index.
        oracle_call : int
            1-based cumulative oracle call index.
        """
        if canonical_smiles in self._records:
            return

        if p_active > self.best_p_active:
            self.best_p_active = float(p_active)
        if p_active >= 0.5:
            self.n_hits_05 += 1
        if p_active >= 0.8:
            self.n_hits_08 += 1

        need_record = p_active >= self.hit_threshold
        need_scaffold_06 = p_active >= 0.6
        if not need_record and not need_scaffold_06:
            return

        scaffold = generic_scaffold_hash(canonical_smiles)
        if need_scaffold_06:
            self.n_hits_06 += 1
            self._scaffold_counts_06[scaffold] += 1

        if need_record:
            self._records[canonical_smiles] = CandidateRecord(
                canonical_smiles=canonical_smiles,
                smiles=smiles,
                p_active=float(p_active),
                generic_scaffold=scaffold,
                walker_id=int(walker_id),
                step=int(step),
                oracle_call=int(oracle_call),
            )

    def ranked_records(self, top_n: int | None = None) -> list[CandidateRecord]:
        """Return stored records sorted by descending ``p_active``."""
        ordered = sorted(
            self._records.values(),
            key=lambda r: (-r.p_active, r.oracle_call),
        )
        return ordered[:top_n] if top_n is not None else ordered

    def to_dataframe(self) -> pd.DataFrame:
        """Convert ranked candidate records to a DataFrame for CSV export."""
        records = self.ranked_records()
        if not records:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)
        rows = [
            {
                "rank": rank,
                "canonical_smiles": r.canonical_smiles,
                "smiles": r.smiles,
                "p_active": r.p_active,
                "generic_scaffold": r.generic_scaffold,
                "walker_id": r.walker_id,
                "step": r.step,
                "oracle_call": r.oracle_call,
            }
            for rank, r in enumerate(records, start=1)
        ]
        return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
