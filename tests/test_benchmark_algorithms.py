"""Validate every algorithm spec against the sphere domain before GPU time.

This is the "sphere first" gate: each CONFIG entry must construct a scheduler
over shared geometry and survive a few ask/oracle/tell iterations. Rib-specific
API drift (renamed kwargs, missing rankers, bandit pools) surfaces here in
seconds instead of as a broken 24-minute molecule run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ribs.schedulers import BanditScheduler

from benchmarks.algorithms import (
    ALGORITHMS,
    CONFIG,
    FEASIBLE_BOUNDS_RATIO,
    THRESHOLD_MIN,
    create_scheduler,
    scheduler_batch_size,
)
from benchmarks.domains import ARCHIVE_RESOLUTION, molecule_domain, sphere_domain
from config import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def sphere(solutions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sphere objectives in [0, 100] and its two half-space measures.

    Reproduces Section 4 of Fontaine 2020: the optimum sits at ``0.4 * 5.12``,
    coordinates are clipped to ``[-5.12, 5.12]``, and each measure sums half
    of the clipped coordinates.
    """
    dim = solutions.shape[1]
    sphere_shift = 5.12 * 0.4
    best_obj = 0.0
    worst_obj = (-5.12 - sphere_shift) ** 2 * dim
    raw_obj = np.sum(np.square(solutions - sphere_shift), axis=1)
    objectives = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    clipped = solutions.copy()
    mask = (clipped < -5.12) | (clipped > 5.12)
    clipped[mask] = 5.12 / clipped[mask]
    measures = np.concatenate(
        (
            np.sum(clipped[:, : dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2 :], axis=1, keepdims=True),
        ),
        axis=1,
    )
    return objectives, measures


def test_registry_has_all_algorithms():
    """19 unthresholded sphere/bandit algorithms plus 10 thresholded ES variants."""
    assert len(ALGORITHMS) == 29
    assert set(ALGORITHMS) == set(CONFIG)
    thresh_names = [a for a in ALGORITHMS if a.endswith("_thresh")]
    assert len(thresh_names) == 10


@pytest.mark.parametrize("name", [a for a in ALGORITHMS if a.endswith("_thresh")])
def test_thresh_specs_survive_sub_threshold_warmup(name):
    """Sub-0.6 warmup batches must not crash or freeze before the first hit."""
    cfg = ExperimentConfig.from_toml(str(REPO_ROOT / "config.toml"))
    domain = molecule_domain(cfg)
    scheduler = create_scheduler(CONFIG[name], domain, seed=0)
    rng = np.random.default_rng(0)

    # Three warmup steps where all objectives are below THRESHOLD_MIN (0.6).
    for step in range(3):
        z = scheduler.ask()
        objectives = rng.uniform(0.01, THRESHOLD_MIN - 0.05, size=len(z))
        measures = np.column_stack(
            [rng.uniform(low, high, size=len(z)) for low, high in domain.measure_ranges]
        )
        scheduler.tell(objectives, measures)
        assert scheduler.archive.empty

    # Fourth step crosses THRESHOLD_MIN -> primary archive accepts elites.
    z = scheduler.ask()
    objectives = rng.uniform(THRESHOLD_MIN + 0.05, 0.95, size=len(z))
    measures = np.column_stack(
        [rng.uniform(low, high, size=len(z)) for low, high in domain.measure_ranges]
    )
    scheduler.tell(objectives, measures)
    assert not scheduler.archive.empty
    assert int(scheduler.result_archive.stats.num_elites) > 0


@pytest.mark.parametrize("name", ALGORITHMS)
def test_spec_builds_and_runs_on_sphere(name):
    domain = sphere_domain()
    scheduler = create_scheduler(CONFIG[name], domain, seed=0)

    # Must be known before the first ask: the runner derives max_steps from it.
    assert scheduler_batch_size(scheduler) > 0

    batches = []
    for _ in range(3):
        z = scheduler.ask()
        objectives, measures = sphere(z)
        assert objectives.shape == (scheduler_batch_size(scheduler),)
        assert measures.shape == (scheduler_batch_size(scheduler), domain.measure_dim)
        scheduler.tell(objectives, measures)
        batches.append(z.shape[0])

    assert batches[0] > 0
    stats = scheduler.result_archive.stats
    assert int(stats.num_elites) > 0
    assert np.isfinite(stats.qd_score)
    assert 0.0 < float(stats.coverage) <= 1.0


@pytest.mark.parametrize("name", ALGORITHMS)
def test_spec_builds_on_molecule_domain(name):
    """Every spec lays over the molecule map without touching the oracle."""
    cfg = ExperimentConfig.from_toml(str(REPO_ROOT / "config.toml"))
    domain = molecule_domain(cfg)

    scheduler = create_scheduler(CONFIG[name], domain, seed=0)

    assert scheduler.result_archive is not None
    if hasattr(scheduler.archive, "dims"):
        assert tuple(scheduler.archive.dims) == domain.archive_dims


def test_molecule_domain_uses_forty_by_forty_and_scaled_scales():
    """40x40 cells, ranges from config, novelty/KDE scales from box width."""
    cfg = ExperimentConfig.from_toml(str(REPO_ROOT / "config.toml"))
    domain = molecule_domain(cfg)

    widths = [high - low for low, high in domain.measure_ranges]
    mean_width = sum(widths) / len(widths)

    assert domain.archive_dims == (ARCHIVE_RESOLUTION,) * domain.measure_dim
    assert domain.n_cells == ARCHIVE_RESOLUTION * ARCHIVE_RESOLUTION
    assert domain.emitter_bounds == (
        -float(cfg.emitter.bounds),
        float(cfg.emitter.bounds),
    )
    assert domain.novelty_threshold == pytest.approx(0.037 * mean_width)
    assert domain.kde_bandwidth == pytest.approx(0.05 * mean_width)
    # A descriptor box under 1 unit wide must not inherit sphere's threshold.
    assert domain.novelty_threshold < 0.1


def test_sphere_domain_reproduces_paper_constants():
    """The scaling rule must land on the paper's 18.94 / 25.6 on sphere."""
    domain = sphere_domain()
    assert domain.solution_dim == 100
    assert domain.emitter_bounds is None
    assert domain.n_cells == 10_000
    assert domain.novelty_threshold == pytest.approx(0.037 * 512)
    assert domain.kde_bandwidth == pytest.approx(25.6)


def test_cvt_and_grid_share_cell_count():
    """Coverage and QD Score are only comparable if cell budgets match."""
    domain = sphere_domain()
    grid = create_scheduler(CONFIG["map_elites"], domain, seed=0)
    cvt = create_scheduler(CONFIG["cvt_map_elites"], domain, seed=0)

    assert tuple(grid.archive.dims) == (100, 100)
    assert cvt.archive.cells == 10_000


def test_multi_group_emitters_draw_distinct_seeds():
    """Every emitter owns a distinct spawn key, across all spec groups.

    Per-group ``SeedSequence(seed).spawn()`` restarts child keys at 0, so
    ``cma_me_mixed``'s 7+8 groups produced 7 pairs of identical emitters —
    252 of its first 540 proposals were intra-batch duplicates (pilot
    ``benchmark_results/cma_me_mixed_seed1``: unique_scored 288 at step 0).
    """
    domain = sphere_domain()

    def spawn_key(emitter):
        # ES holds its RNG on the CMA optimizer; IsoLine/Gaussian hold it directly.
        rng = getattr(emitter, "_rng", None) or emitter._opt._rng
        return tuple(rng.bit_generator.seed_seq.spawn_key)

    for name in ("cma_me_mixed", "me_map_elites"):
        scheduler = create_scheduler(CONFIG[name], domain, seed=0)
        pool = getattr(scheduler, "emitter_pool", None)
        emitters = scheduler.emitters if pool is None else pool
        keys = [spawn_key(e) for e in emitters]
        assert len(set(keys)) == len(keys), name


def test_obj_bandit_keeps_pool_wide_and_ranks_by_objective():
    """The hybrid: 60 emitters in the pool, 15 active, all ranking by objective.

    The pool is what gives the map its say (the bandit rotates emitters over
    the map), while ``ranker="obj"`` keeps the cell threshold out of the CMA's
    fitness -- the failure mode of every imp/2imp arm.
    """
    scheduler = create_scheduler(CONFIG["cma_me_obj_bandit"], sphere_domain(), seed=0)

    assert isinstance(scheduler, BanditScheduler)
    assert len(scheduler.emitter_pool) == 60
    assert scheduler_batch_size(scheduler) == 15 * 36

    z = scheduler.ask()
    assert z.shape == (540, 100)
    assert int(np.asarray(scheduler.active).sum()) == 15

    rankers = {type(emitter._ranker).__name__ for emitter in scheduler.emitter_pool}
    assert rankers == {"ObjectiveRanker"}


def test_cma_me_mixed_first_batch_is_fully_distinct():
    """540 proposals in, 540 distinct out — no duplicate mass at step 0."""
    scheduler = create_scheduler(CONFIG["cma_me_mixed"], sphere_domain(), seed=0)
    z = scheduler.ask()
    assert z.shape[0] == 540
    assert np.unique(z, axis=0).shape[0] == 540


def test_dds_sigma0_is_capped_to_fit_the_latent_box():
    """Uncapped, DDS's sigma0=1.5 never returns from ask() under a +-3 bound.

    pyribs resamples out-of-bounds proposals until every coordinate fits, so a
    box only ~2 sigma wide loops forever — this pin is what keeps the molecule
    sweep from hanging on ``dds_kde``.
    """
    cfg = ExperimentConfig.from_toml(str(REPO_ROOT / "config.toml"))
    domain = molecule_domain(cfg)
    cap = domain.emitter_bounds[1] / FEASIBLE_BOUNDS_RATIO

    scheduler = create_scheduler(CONFIG["dds_kde"], domain, seed=0)
    sigma0s = [e._sigma0 for e in scheduler.emitters if hasattr(e, "_sigma0")]

    assert sigma0s and all(s <= cap + 1e-12 for s in sigma0s)
    assert cap < 1.5  # the cap is what changes for DDS

    # Unbounded sphere keeps the published step size untouched.
    sphere_scheduler = create_scheduler(CONFIG["dds_kde"], sphere_domain(), seed=0)
    sphere_sigma0s = [
        e._sigma0 for e in sphere_scheduler.emitters if hasattr(e, "_sigma0")
    ]
    assert all(s == 1.5 for s in sphere_sigma0s)
