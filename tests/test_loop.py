"""Tests for the CMA-MAE optimization loop."""

from __future__ import annotations

import numpy as np
import pytest
from ribs.archives import GridArchive

from evaluation.evaluator import INVALID_MOLECULE_OBJECTIVE, EvalResult
from optimization.loop import CMAMAELoop

SOLUTION_DIM = 2
MEASURES = np.array([[0.1], [0.5], [0.9]])
P_ACTIVE = np.array([0.9, 0.04, INVALID_MOLECULE_OBJECTIVE])


class _StubScheduler:
    """Minimal scheduler that records the archive objectives and ranking info."""

    def __init__(self):
        self.ask_rows = len(MEASURES)
        # 10 bins so the measure rows land in distinct cells.
        self.archive = GridArchive(
            solution_dim=SOLUTION_DIM,
            dims=[10],
            ranges=[[0.0, 1.0]],
            learning_rate=0.01,
            threshold_min=0.1,
        )
        self.result_archive = GridArchive(
            solution_dim=SOLUTION_DIM,
            dims=[1],
            ranges=[[0.0, 1.0]],
            learning_rate=1.0,
            threshold_min=0.0,
        )
        self.told: list[np.ndarray] = []
        self.add_info: dict = {}
        self._emitters: list = []
        self._num_emitted = [len(MEASURES)]

    def ask(self):
        """Return one latent vector per measure row."""
        return np.zeros((self.ask_rows, SOLUTION_DIM))

    def tell(self, objectives, measures):
        """Record the objectives received and the archive's ranking values."""
        objectives = np.asarray(objectives)
        self.told.append(np.array(objectives, dtype=np.float64))
        self.add_info = self.archive.add(
            solution=np.zeros((len(objectives), SOLUTION_DIM)),
            objective=objectives,
            measures=measures,
        )


def _run_loop(
    objectives: np.ndarray,
    threshold_min: float = 0.1,
    scheduler: _StubScheduler | None = None,
) -> tuple[_StubScheduler, CMAMAELoop]:
    scheduler = scheduler or _StubScheduler()
    result = EvalResult(
        smiles=["a", "b", "c"],
        objectives=objectives.copy(),
        measures=MEASURES.copy(),
        p_active=objectives.copy(),
        n_valid=3,
        gen_time=0.0,
    )
    loop = CMAMAELoop(scheduler, evaluate=lambda z: result, threshold_min=threshold_min)
    loop.run(n_generations=1)
    return scheduler, loop


def test_scheduler_receives_masked_objectives():
    """Sub-threshold scores must reach the archive and emitters as invalid.

    Masking is deliberate: ``ImprovementRanker`` ranks by
    ``objective - cell_threshold``, so masked candidates in unexplored cells
    outrank masked candidates in saturated ones. That steers CMA away from
    exhausted regions of the behaviour map.
    """
    scheduler, _ = _run_loop(P_ACTIVE)
    told = scheduler.told[0]

    assert told[0] == 0.9
    assert told[1] == INVALID_MOLECULE_OBJECTIVE
    assert told[2] == INVALID_MOLECULE_OBJECTIVE


def test_masked_rejects_rank_by_lowest_cell_threshold():
    """Equally-invalid candidates rank by cell threshold, not by activity.

    Guards the measured behaviour: feeding raw objectives instead cost a 5.4x
    drop in ``p>0.7`` output over 350 generations (``test2_before_fix`` vs
    ``test1_after_fix``, 147 -> 27 hits).
    """
    scheduler = _StubScheduler()
    scheduler.ask_rows = 2
    # Saturate the high cell so its threshold rises above threshold_min.
    scheduler.archive.add(
        solution=np.zeros((8, SOLUTION_DIM)),
        objective=np.linspace(0.6, 0.9, 8),
        measures=np.full((8, 1), 0.9),
    )
    saturated_threshold = scheduler.archive.data()["threshold"].max()

    masked = np.full(2, INVALID_MOLECULE_OBJECTIVE)
    result = EvalResult(
        smiles=["f", "g"],
        objectives=masked.copy(),
        measures=np.array([[0.1], [0.9]]),
        p_active=masked.copy(),
        n_valid=2,
        gen_time=0.0,
    )
    loop = CMAMAELoop(scheduler, evaluate=lambda z: result, threshold_min=0.1)
    loop.run(n_generations=1)

    value = scheduler.add_info["value"]
    assert saturated_threshold > 0.1
    # The reject in the unexplored cell (index 0) outranks the one in the
    # saturated cell (index 1), even though both objectives are identical.
    assert value[0] > value[1]
    assert value[0] == pytest.approx(INVALID_MOLECULE_OBJECTIVE - 0.1)


def test_insertion_stats_still_count_the_accepted_molecule():
    """Masking must not confuse the tracker about what the archive accepted."""
    _, loop = _run_loop(P_ACTIVE)

    assert loop.last_insertion_stats == {
        "inserted_new": 1,
        "improved_existing": 0,
        "rejected": 2,
    }
    assert len(loop.archive) == 1
    assert loop.archive.data()["objective"].max() == 0.9


def test_no_masking_when_threshold_is_zero():
    """With ``threshold_min=0`` the raw objectives must pass through untouched."""
    scheduler, _ = _run_loop(np.array([0.9, 0.04, 0.01]), threshold_min=0.0)

    assert scheduler.told[0].tolist() == [0.9, 0.04, 0.01]
