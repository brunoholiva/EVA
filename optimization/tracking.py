"""Real-objective tracking and insertion statistics for CMA-MAE."""

from __future__ import annotations

import numpy as np
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler

from optimization.constants import INVALID_MOLECULE_OBJECTIVE
from optimization.evaluator import EvalResult


def occupied_cells(archive: GridArchive) -> dict[int, float]:
    """Return ``{cell_index: objective}`` for all occupied cells.

    Parameters
    ----------
    archive : GridArchive
        The archive to snapshot.

    Returns
    -------
    dict of int to float
        Mapping from cell index to stored objective for every occupied cell.
        Empty dict if the archive has no elites.
    """
    cells: dict[int, float] = {}
    if len(archive) > 0:
        data = archive.data()
        for idx, obj in zip(data["index"], data["objective"]):
            cells[int(idx)] = float(obj)
    return cells


def compute_insertion_stats(
    result: EvalResult,
    objectives: np.ndarray,
    old_occupied: dict[int, float],
    archive: GridArchive,
) -> dict[str, int]:
    """Categorise each candidate's archive insertion outcome.

    Parameters
    ----------
    result : EvalResult
        Scoring results for this generation.
    objectives : np.ndarray
        Thresholded/capped objectives passed to ``scheduler.tell``.
    old_occupied : dict of int to float
        Cell-index → objective snapshot taken *before* ``tell``.
    archive : GridArchive
        The archive insertion was performed into.

    Returns
    -------
    dict of str to int
        Counts keyed ``"inserted_new"``, ``"improved_existing"``,
        ``"rejected"``.
    """
    acceptable = _acceptable_mask(result.measures, objectives, archive)
    all_indices = archive.index_of(result.measures)

    inserted_new = 0
    improved_existing = 0
    rejected = 0

    for i in range(len(objectives)):
        if not acceptable[i]:
            rejected += 1
            continue
        cell_idx = int(all_indices[i])
        if cell_idx not in old_occupied:
            inserted_new += 1
        elif objectives[i] > old_occupied[cell_idx]:
            improved_existing += 1
        else:
            rejected += 1

    return {
        "inserted_new": inserted_new,
        "improved_existing": improved_existing,
        "rejected": rejected,
    }


def compute_emitter_stats(scheduler: Scheduler) -> list[dict]:
    """Return per-emitter distance from origin and restart count.

    Parameters
    ----------
    scheduler : Scheduler
        The pyribs scheduler whose emitters to inspect.

    Returns
    -------
    list of dict
        Each dict has keys ``"id"``, ``"distance"``, ``"restarts"``.
    """
    stats = []
    for i, emitter in enumerate(scheduler._emitters):
        if hasattr(emitter, "_opt"):
            dist = np.linalg.norm(emitter._opt.mean - emitter.x0)
        else:
            dist = 0.0
        stats.append(
            {
                "id": i,
                "distance": float(dist),
                "restarts": getattr(emitter, "restarts", 0),
            }
        )
    return stats


def compute_emitter_spread(scheduler: Scheduler) -> float:
    """Return the standard deviation of emitter means across all emitters.

    A low value indicates all emitters are clustered in the same basin.

    Parameters
    ----------
    scheduler : Scheduler
        The pyribs scheduler whose emitters to inspect.

    Returns
    -------
    float
        Mean std of emitter means across latent dimensions.
    """
    means = np.array([e._opt.mean for e in scheduler._emitters if hasattr(e, "_opt")])
    if len(means) == 0:
        return 0.0
    return float(np.mean(np.std(means, axis=0)))


def compute_emitter_insertions(
    result: EvalResult,
    objectives: np.ndarray,
    old_occupied: dict[int, float],
    archive: GridArchive,
    scheduler: Scheduler,
) -> list[dict]:
    """Attribute archive insertions to the emitters that produced them.

    Uses ``scheduler._num_emitted`` to determine which emitter produced
    each solution in the batch (solutions are contiguous per emitter).

    Parameters
    ----------
    result : EvalResult
        Scoring results for this generation.
    objectives : np.ndarray
        Thresholded/capped objectives passed to ``scheduler.tell``.
    old_occupied : dict of int to float
        Cell-index to objective snapshot taken *before* ``tell``.
    archive : GridArchive
        The archive insertion was performed into.
    scheduler : Scheduler
        The pyribs scheduler (provides emitter boundaries via
        ``_num_emitted``).

    Returns
    -------
    list of dict
        Each dict has keys ``"id"``, ``"inserted_new"``,
        ``"improved_existing"``, ``"rejected"``.
    """
    acceptable = _acceptable_mask(result.measures, objectives, archive)
    all_indices = archive.index_of(result.measures)

    boundaries: list[tuple[int, int]] = []
    pos = 0
    for n in scheduler._num_emitted:
        boundaries.append((pos, pos + n))
        pos += n

    emitter_stats: list[dict] = []
    for em_id, (start, end) in enumerate(boundaries):
        inserted_new = 0
        improved_existing = 0
        rejected = 0
        for i in range(start, min(end, len(objectives))):
            if not acceptable[i]:
                rejected += 1
                continue
            cell_idx = int(all_indices[i])
            if cell_idx not in old_occupied:
                inserted_new += 1
            elif objectives[i] > old_occupied[cell_idx]:
                improved_existing += 1
            else:
                rejected += 1
        emitter_stats.append(
            {
                "id": em_id,
                "inserted_new": inserted_new,
                "improved_existing": improved_existing,
                "rejected": rejected,
            }
        )
    return emitter_stats


class RealObjectiveTracker:
    """Track uncapped P(active) values for solutions accepted into archives.

    The CMA-MAE loop caps objectives before archive insertion to prevent
    exploitation. This tracker records the real (uncapped) P(active) of
    solutions that were actually accepted (new or strictly improved) into
    the archive and result archive, keyed by cell index.
    """

    def __init__(self) -> None:
        self._primary: dict[int, float] = {}
        self._result: dict[int, float] = {}

    @property
    def primary(self) -> dict[int, float]:
        """Real objectives for the primary archive, keyed by cell index."""
        return self._primary

    @property
    def result(self) -> dict[int, float]:
        """Real objectives for the result archive, keyed by cell index."""
        return self._result

    def update(
        self,
        archive: GridArchive,
        result_archive: GridArchive | None,
        z: np.ndarray,
        result: EvalResult,
        objectives: np.ndarray,
        old_primary_cells: dict[int, float],
        old_result_cells: dict[int, float],
    ) -> None:
        """Update real objectives for both archives after a ``tell()`` call.

        Parameters
        ----------
        archive : GridArchive
            Primary archive (post-tell state).
        result_archive : GridArchive or None
            Result archive (post-tell state), or ``None`` if not used.
        z : np.ndarray
            Latent vectors submitted to ``ask``.
        result : EvalResult
            Scoring results for this generation.
        objectives : np.ndarray
            Thresholded/capped objectives passed to ``tell``.
        old_primary_cells : dict of int to float
            Cell snapshot of *archive* taken before ``tell``.
        old_result_cells : dict of int to float
            Cell snapshot of *result_archive* taken before ``tell``.
        """
        self._update_one(
            archive,
            z,
            result,
            objectives,
            result.p_active,
            self._primary,
            old_primary_cells,
        )
        if result_archive is not None:
            self._update_one(
                result_archive,
                z,
                result,
                objectives,
                result.p_active,
                self._result,
                old_result_cells,
            )

    @staticmethod
    def _update_one(
        archive: GridArchive,
        z: np.ndarray,
        result: EvalResult,
        objectives: np.ndarray,
        real_array: np.ndarray,
        target: dict[int, float],
        old_cells: dict[int, float],
    ) -> None:
        """Update a single archive's real objective dict."""
        updated_cells = _find_updated_cells(archive, old_cells)
        if not updated_cells:
            return
        _assign_best_real_per_cell(
            archive,
            z,
            result,
            objectives,
            real_array,
            target,
            updated_cells,
        )


def _acceptable_mask(
    measures: np.ndarray,
    objectives: np.ndarray,
    archive: GridArchive,
) -> np.ndarray:
    """Return mask of candidates that are valid and within archive bounds."""
    valid = objectives != INVALID_MOLECULE_OBJECTIVE
    in_bounds = _in_bounds_mask(measures, archive)
    return valid & in_bounds


def _in_bounds_mask(measures: np.ndarray, archive: GridArchive) -> np.ndarray:
    """Return boolean mask of rows whose measures are within archive bounds."""
    lb = archive.lower_bounds
    ub = archive.upper_bounds
    return ((measures >= lb) & (measures <= ub)).all(axis=1)


def _find_updated_cells(
    archive: GridArchive,
    old_cells: dict[int, float],
) -> set[int]:
    """Find cells that were newly occupied or strictly improved."""
    if len(archive) == 0:
        return set()
    data = archive.data()
    updated: set[int] = set()
    for idx, obj in zip(data["index"], data["objective"]):
        cell_idx = int(idx)
        old_obj = old_cells.get(cell_idx)
        if old_obj is None or float(obj) > old_obj + 1e-9:
            updated.add(cell_idx)
    return updated


def _assign_best_real_per_cell(
    archive: GridArchive,
    z: np.ndarray,
    result: EvalResult,
    objectives: np.ndarray,
    real_array: np.ndarray,
    target: dict[int, float],
    updated_cells: set[int],
) -> None:
    """For each updated cell, record the best batch solution's real objective."""
    acceptable = _acceptable_mask(result.measures, objectives, archive)
    real_valid = real_array != INVALID_MOLECULE_OBJECTIVE
    eligible = acceptable & real_valid

    solutions_by_cell: dict[int, list[int]] = {}
    for i in range(len(z)):
        if not eligible[i]:
            continue
        cell_idx = int(archive.index_of_single(result.measures[i]))
        if cell_idx < 0:
            continue
        if cell_idx in updated_cells:
            solutions_by_cell.setdefault(cell_idx, []).append(i)

    for cell_idx, batch_indices in solutions_by_cell.items():
        best_idx = max(batch_indices, key=lambda i: objectives[i])
        target[cell_idx] = float(real_array[best_idx])
