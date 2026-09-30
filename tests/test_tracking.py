"""Tests for optimization tracking utilities."""

from __future__ import annotations

import numpy as np
import pytest
from ribs.archives import GridArchive

from evaluation.evaluator import INVALID_MOLECULE_OBJECTIVE, EvalResult
from optimization.tracking import (
    RealObjectiveTracker,
    compute_insertion_stats,
    concentration_gini,
    occupied_cells,
)


def _make_archive(dims=10, ranges=None, seed=0):
    if ranges is None:
        ranges = [[0.0, 1.0]]
    return GridArchive(
        solution_dim=2,
        dims=[dims],
        ranges=ranges,
        learning_rate=1.0,
        threshold_min=0.0,
        seed=seed,
    )


def _make_result(measures, p_active, objectives=None):
    n = len(measures)
    if objectives is None:
        objectives = np.array(p_active, dtype=np.float64)
    return EvalResult(
        smiles=[""] * n,
        objectives=objectives,
        measures=np.array(measures, dtype=np.float64),
        p_active=np.array(p_active, dtype=np.float64),
        n_valid=n,
        gen_time=0.0,
    )


class TestOccupiedCells:
    """Tests for occupied_cells."""

    def test_empty_archive_returns_empty_dict(self) -> None:
        archive = _make_archive()
        assert occupied_cells(archive) == {}

    def test_returns_cell_index_to_objective(self) -> None:
        archive = _make_archive()
        archive.add(
            solution=np.array([[1.0, 2.0]]),
            objective=np.array([0.5]),
            measures=np.array([[0.3]]),
        )
        cells = occupied_cells(archive)
        assert len(cells) == 1
        cell_idx = list(cells.keys())[0]
        assert cells[cell_idx] == pytest.approx(0.5)


class TestComputeInsertionStats:
    """Tests for compute_insertion_stats."""

    def test_invalid_molecule_is_rejected(self) -> None:
        archive = _make_archive()
        result = _make_result(
            measures=[[0.5]],
            p_active=[0.0],
            objectives=np.array([INVALID_MOLECULE_OBJECTIVE]),
        )
        stats = compute_insertion_stats(result, result.objectives, {}, archive)
        assert stats == {"inserted_new": 0, "improved_existing": 0, "rejected": 1}

    def test_new_cell_is_inserted(self) -> None:
        archive = _make_archive()
        result = _make_result(measures=[[0.5]], p_active=[0.7])
        stats = compute_insertion_stats(result, result.objectives, {}, archive)
        assert stats["inserted_new"] == 1
        assert stats["rejected"] == 0

    def test_higher_objective_improves_existing(self) -> None:
        archive = _make_archive()
        archive.add(
            solution=np.array([[1.0, 2.0]]),
            objective=np.array([0.5]),
            measures=np.array([[0.5]]),
        )
        old = occupied_cells(archive)
        result = _make_result(measures=[[0.5]], p_active=[0.9])
        stats = compute_insertion_stats(result, result.objectives, old, archive)
        assert stats["improved_existing"] == 1

    def test_lower_objective_is_rejected(self) -> None:
        archive = _make_archive()
        archive.add(
            solution=np.array([[1.0, 2.0]]),
            objective=np.array([0.9]),
            measures=np.array([[0.5]]),
        )
        old = occupied_cells(archive)
        result = _make_result(measures=[[0.5]], p_active=[0.3])
        stats = compute_insertion_stats(result, result.objectives, old, archive)
        assert stats["rejected"] == 1

    def test_out_of_bounds_is_rejected(self) -> None:
        archive = _make_archive()
        result = _make_result(measures=[[1.5]], p_active=[0.7])
        stats = compute_insertion_stats(result, result.objectives, {}, archive)
        assert stats["rejected"] == 1
        assert stats["inserted_new"] == 0


class TestRealObjectiveTracker:
    """Tests for RealObjectiveTracker."""

    def test_tracks_real_objective_for_new_cell(self) -> None:
        archive = _make_archive()
        old_cells = occupied_cells(archive)
        z = np.array([[1.0, 2.0]])
        result = _make_result(measures=[[0.5]], p_active=[0.7])
        objectives = np.array([0.7])

        archive.add(z, objectives, np.array([[0.5]]))

        tracker = RealObjectiveTracker()
        tracker.update(archive, None, z, result, objectives, old_cells, {})
        assert len(tracker.primary) == 1
        assert list(tracker.primary.values())[0] == pytest.approx(0.7)
        assert tracker.result == {}

    def test_does_not_track_invalid_solutions(self) -> None:
        archive = _make_archive()
        old_cells = occupied_cells(archive)
        z = np.array([[1.0, 2.0]])
        result = _make_result(
            measures=[[0.5]],
            p_active=[0.7],
            objectives=np.array([INVALID_MOLECULE_OBJECTIVE]),
        )

        tracker = RealObjectiveTracker()
        tracker.update(archive, None, z, result, result.objectives, old_cells, {})
        assert tracker.primary == {}

    def test_updates_both_archares(self) -> None:
        archive = _make_archive(seed=0)
        result_archive = _make_archive(seed=1)
        old_primary = occupied_cells(archive)
        old_result = occupied_cells(result_archive)

        z = np.array([[1.0, 2.0]])
        result = _make_result(measures=[[0.5]], p_active=[0.7])
        objectives = np.array([0.7])

        archive.add(z, objectives, np.array([[0.5]]))
        result_archive.add(z, objectives, np.array([[0.5]]))

        tracker = RealObjectiveTracker()
        tracker.update(
            archive,
            result_archive,
            z,
            result,
            objectives,
            old_primary,
            old_result,
        )
        assert len(tracker.primary) == 1
        assert len(tracker.result) == 1
        assert list(tracker.primary.values())[0] == pytest.approx(0.7)
        assert list(tracker.result.values())[0] == pytest.approx(0.7)


class TestConcentrationGini:
    """Tests for concentration_gini."""

    def test_equal_counts_returns_zero(self) -> None:
        assert concentration_gini([10, 10, 10, 10]) == pytest.approx(0.0)

    def test_single_emitter_returns_zero(self) -> None:
        assert concentration_gini([100]) == pytest.approx(0.0)

    def test_empty_returns_zero(self) -> None:
        assert concentration_gini([]) == pytest.approx(0.0)

    def test_all_zeros_returns_zero(self) -> None:
        assert concentration_gini([0, 0, 0]) == pytest.approx(0.0)

    def test_single_nonzero_rest_zero(self) -> None:
        result = concentration_gini([100, 0, 0, 0])
        assert result == pytest.approx(0.75)

    def test_two_equal_rest_zero(self) -> None:
        result = concentration_gini([50, 50, 0, 0])
        assert 0.0 < result < 1.0
