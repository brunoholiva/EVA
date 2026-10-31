"""Tests for pycma walkers, MultiCMASolver, and NaiveSolver."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest

from config import CMAConfig
from optimization.cma import CMAWalker, MultiCMASolver, NaiveSolver, build_solver


def test_walker_maximizes_objective() -> None:
    target = np.full(8, 1.25)
    cfg = CMAConfig(
        n_populations=1,
        popsize=16,
        sigma0=0.5,
        init_std=0.0,
        bounds=3.0,
    )
    walker = CMAWalker(walker_id=0, solution_dim=8, cfg=cfg, init_seed=1, es_seed=2)
    initial_dist = float(np.linalg.norm(walker.mean - target))

    for _ in range(25):
        z = walker.ask()
        # Maximize negative squared distance in [0, 1]-like range
        objectives = -np.sum((z - target) ** 2, axis=1)
        walker.tell(z, objectives)

    final_dist = float(np.linalg.norm(walker.mean - target))
    assert final_dist < 0.25 * initial_dist


def test_walker_respects_bounds_and_init_std() -> None:
    cfg = CMAConfig(
        n_populations=1,
        popsize=12,
        sigma0=0.8,
        init_std=5.0,
        bounds=2.0,
    )
    walker = CMAWalker(walker_id=0, solution_dim=16, cfg=cfg, init_seed=7, es_seed=8)
    assert np.all(np.abs(walker.mean) <= 2.0)
    z = walker.ask()
    assert z.shape == (12, 16)
    assert np.all(z >= -2.0) and np.all(z <= 2.0)


def test_walker_max_gens_per_run_triggers_restart() -> None:
    cfg = CMAConfig(
        n_populations=1,
        popsize=8,
        sigma0=0.5,
        init_std=1.0,
        bounds=3.0,
        max_gens_per_run=3,
    )
    walker = CMAWalker(walker_id=0, solution_dim=6, cfg=cfg, init_seed=10, es_seed=11)
    restarted = []
    for _ in range(3):
        z = walker.ask()
        restarted.append(walker.tell(z, np.linspace(0.0, 1.0, len(z))))
    assert restarted == [False, False, True]
    assert walker.restarts == 1
    assert walker.gens_in_run == 0
    assert walker.stats().last_stop_reasons == {"max_gens_per_run": 3}


def test_walker_opts_passthrough_diagonal_and_minstd() -> None:
    cfg = CMAConfig(
        n_populations=1,
        popsize=10,
        sigma0=0.5,
        bounds=3.0,
        opts={"CMA_diagonal": True, "minstd": 0.3},
    )
    walker = CMAWalker(walker_id=0, solution_dim=12, cfg=cfg, init_seed=3, es_seed=4)
    for _ in range(10):
        z = walker.ask()
        walker.tell(z, -np.sum(z**2, axis=1))
    assert walker.axis_ratio >= 1.0
    assert np.all(walker._es.stds >= 0.3 - 1e-9)


def test_reserved_pycma_opts_rejected() -> None:
    cfg = CMAConfig(opts={"popsize": 20})
    with pytest.raises(ValueError, match="cma.opts"):
        CMAWalker(walker_id=0, solution_dim=4, cfg=cfg, init_seed=1, es_seed=2)


def test_multi_cma_solver_joblib_roundtrip_is_bit_identical() -> None:
    cfg = CMAConfig(n_populations=3, popsize=6, sigma0=0.5, init_std=1.0, bounds=3.0)
    solver = MultiCMASolver(solution_dim=10, cfg=cfg, seed=42)
    z0, wids = solver.ask()
    assert z0.shape == (18, 10)
    assert wids.tolist() == [0] * 6 + [1] * 6 + [2] * 6
    solver.tell(z0, np.linspace(-1.0, 0.9, 18))
    assert solver.walker_spread() > 0.0

    buf = io.BytesIO()
    joblib.dump(solver, buf)
    buf.seek(0)
    restored: MultiCMASolver = joblib.load(buf)

    z_orig, _ = solver.ask()
    z_restored, _ = restored.ask()
    assert np.allclose(z_orig, z_restored)


def test_naive_solver_and_factory() -> None:
    cfg = CMAConfig(n_populations=2, popsize=5)
    solver = build_solver("naive", solution_dim=8, cfg=cfg, seed=99)
    assert isinstance(solver, NaiveSolver)
    z, wids = solver.ask()
    assert z.shape == (10, 8)
    assert np.all(wids == -1)
    solver.tell(z, np.zeros(10))
    assert solver.walker_stats() == []
    assert solver.walker_spread() == 0.0
    assert solver.total_restarts() == 0
