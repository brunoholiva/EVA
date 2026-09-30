"""Diversity and quality metrics for a finished archive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ribs.archives import GridArchive

from chemistry.scaffolds import cluster_by_generic_scaffold
from optimization.tracking import concentration_gini


@dataclass(frozen=True)
class ArchiveMetrics:
    """Diversity and quality summary for a finished archive."""

    scaffold_n_unique: int
    scaffold_largest_frac: float
    emitter_gini: float
    fraction_above_0_6: float
    best_objective: float


def final_archive_metrics(
    archive: GridArchive,
    decode_fn,
    real_objectives: dict[int, float],
    emitter_insertion_totals: list[int],
) -> ArchiveMetrics:
    """Summarise a finished archive.

    Parameters
    ----------
    archive : GridArchive
        The result archive to summarise.
    decode_fn : callable
        Latent vectors -> SMILES decoder.
    real_objectives : dict of int to float
        Cell index -> uncapped P(active) for archive solutions.
    emitter_insertion_totals : list of int
        Cumulative insertions per emitter.

    Returns
    -------
    ArchiveMetrics
    """
    gini = concentration_gini(emitter_insertion_totals)

    if len(archive) == 0 or not real_objectives:
        return ArchiveMetrics(0, 0.0, gini, 0.0, 0.0)

    smiles = decode_fn(archive.data()["solution"])
    clusters = cluster_by_generic_scaffold(smiles)
    largest_frac = max(len(c) for c in clusters) / len(smiles) if smiles else 0.0

    values = np.array(list(real_objectives.values()))
    return ArchiveMetrics(
        scaffold_n_unique=len(clusters),
        scaffold_largest_frac=largest_frac,
        emitter_gini=gini,
        fraction_above_0_6=float((values > 0.6).mean()),
        best_objective=float(values.max()),
    )
