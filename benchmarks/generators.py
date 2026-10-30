"""Baseline generators for the oracle-budget benchmark.

Each generator proposes one batch per :meth:`step` and scores it through the
shared :class:`~evaluation.evaluator.Evaluator`. Generators differ only in how
the latent batch is proposed and whether feedback drives the next proposal:

- ``SchedulerGenerator`` wraps *any* pyribs algorithm spec from
  :mod:`benchmarks.algorithms` — this is what the multi-algorithm sweep runs.
- ``naive``     : ``N(0, I)`` draws in the projected working space, no feedback.
- ``cma_es``    : 15 independent single-objective CMA-ES populations sharing a
                  one-cell archive. One behavior-map knob off versus ``cma_mae``.
- ``cma_mae``   : the real :class:`~optimization.loop.CMAMAELoop` with the
                  configured multi-cell behavior map.

Every generator also reports its archive state via :meth:`Generator.archive_stats`
so the runner can record coverage and QD Score per step; generators without an
archive return an empty mapping.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import numpy as np
from ribs.schedulers import Scheduler

from benchmarks.algorithms import CONFIG, create_scheduler, scheduler_batch_size
from benchmarks.domains import Domain
from config import ArchiveConfig, DimensionConfig, ExperimentConfig
from evaluation.evaluator import EvalResult, Evaluator
from optimization.loop import CMAMAELoop, build_scheduler

EVA_GENERATORS = ("naive", "cma_es", "cma_mae")
"""The three arms EVA already ran; kept out of the multi-algorithm sweep."""

ARCHIVE_STAT_COLUMNS = [
    "result_cells",
    "result_coverage",
    "result_qd_score",
    "result_obj_max",
]
"""Per-step archive metrics written to ``curve.csv``."""


def _archive_stats(scheduler: Scheduler) -> dict[str, float]:
    """Read coverage/QD/objective max off the scheduler's result archive.

    ``Scheduler.result_archive`` falls back to the primary archive when none
    was constructed, which is what MAP-Elites-style algorithms want.
    """
    stats = scheduler.result_archive.stats
    return {
        "result_cells": int(stats.num_elites),
        "result_coverage": float(stats.coverage),
        "result_qd_score": float(stats.qd_score),
        "result_obj_max": (
            float(stats.obj_max) if stats.obj_max is not None else float("nan")
        ),
    }


class Generator(Protocol):
    """Proposes and scores one batch per step."""

    batch_size: int
    """Expected number of candidates proposed per step."""

    def step(self) -> EvalResult:
        """Propose one batch and return its scoring result."""
        ...

    def archive_stats(self) -> dict[str, float]:
        """Report current coverage/QD metrics, or ``{}`` when there is no archive."""
        ...


class SchedulerGenerator:
    """Generic ``ask -> evaluate -> tell`` wrapper for any pyribs algorithm.

    Objectives are passed through unmodified: the sub-threshold masking that
    :class:`~optimization.loop.CMAMAELoop` applies is specific to EVA's own
    runs and would be wrong to impose on algorithms being compared.
    """

    def __init__(self, scheduler: Scheduler, evaluator: Evaluator) -> None:
        self._scheduler = scheduler
        self._evaluator = evaluator
        self.batch_size = scheduler_batch_size(scheduler)

    def step(self) -> EvalResult:
        """Propose one batch, score it, and return the archive's verdict."""
        z = self._scheduler.ask()
        self.batch_size = len(z)
        result = self._evaluator(z)
        self._scheduler.tell(result.objectives, result.measures)
        return result

    def archive_stats(self) -> dict[str, float]:
        """Coverage/QD metrics from the scheduler's result archive."""
        return _archive_stats(self._scheduler)


class NaiveGenerator:
    """Unconditional ``N(0, I)`` sampling in the projected working space."""

    def __init__(
        self, evaluator: Evaluator, latent_dim: int, batch_size: int, seed: int
    ) -> None:
        self._evaluator = evaluator
        self._latent_dim = latent_dim
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)

    def step(self) -> EvalResult:
        """Draw a standard-normal latent batch and score it."""
        z = self._rng.standard_normal((self.batch_size, self._latent_dim))
        return self._evaluator(z)

    def archive_stats(self) -> dict[str, float]:
        """No archive to report on."""
        return {}


class CMASingleGenerator:
    """Single-objective CMA-ES on a one-cell archive (no behavior map).

    Every candidate lands in the archive's single cell, so the improvement
    ranker reduces to ordering by raw objective — this is the zero-diversity
    control for ``cma_mae``. With the same emitters as ``cma_mae`` it is 15
    independent single-objective CMA-ES populations.
    """

    def __init__(self, scheduler: Scheduler, evaluator: Evaluator) -> None:
        self._scheduler = scheduler
        self._evaluator = evaluator
        self.batch_size = sum(e.batch_size for e in scheduler.emitters)

    def step(self) -> EvalResult:
        """Ask the scheduler, score, and tell with a constant measure column."""
        z = self._scheduler.ask()
        result = self._evaluator(z)
        measures = np.zeros((len(z), 1), dtype=np.float64)
        self._scheduler.tell(result.objectives, measures)
        return result

    def archive_stats(self) -> dict[str, float]:
        """Coverage/QD metrics from the one-cell archive."""
        return _archive_stats(self._scheduler)


class CMAMAEGenerator:
    """The real CMA-MAE loop with the configured behavior map."""

    def __init__(self, loop: CMAMAELoop) -> None:
        self._loop = loop
        self.batch_size = sum(e.batch_size for e in loop.scheduler.emitters)

    def step(self) -> EvalResult:
        """Run one CMA-MAE generation."""
        return self._loop.step()

    def archive_stats(self) -> dict[str, float]:
        """Coverage/QD metrics from the loop's result archive."""
        return _archive_stats(self._loop.scheduler)


def _one_cell_archive(cfg: ArchiveConfig) -> ArchiveConfig:
    """Collapse *cfg* to a single-cell archive (all candidates share one cell)."""
    return replace(
        cfg,
        dimensions=[DimensionConfig(name="cell", resolution=1, range=(0.0, 1.0))],
        threshold_min=0.0,
    )


def build_generator(
    name: str,
    cfg: ExperimentConfig,
    make_evaluator,
    seed: int,
    domain: Domain | None = None,
) -> Generator:
    """Construct a benchmark generator by name.

    Dispatches across both registries: the :data:`benchmarks.algorithms.CONFIG`
    sweep entries when *name* matches one (in which case *domain* is required),
    and EVA's own three arms otherwise. A name in both (``cma_mae``) resolves to
    the sweep entry.

    Parameters
    ----------
    name : str
        One of ``"naive"``, ``"cma_es"``, ``"cma_mae"``, or any key of
        :data:`benchmarks.algorithms.CONFIG`.
    cfg : ExperimentConfig
        Experiment config. ``cfg.archive.solution_dim`` must already reflect the
        projected working space (``load_vae`` mutates it).
    make_evaluator : callable
        ``fn(archive_cfg) -> Evaluator`` building an evaluator for the given
        behavior dimensions. Evaluator dedup state is per instance, so each
        generator gets a fresh evaluator.
    seed : int
        Seed for the generator's proposal randomness.
    domain : Domain or None, default=None
        Required for entries from :data:`CONFIG`.

    Returns
    -------
    Generator

    Raises
    ------
    ValueError
        If *name* is in neither registry, or names a sweep entry without *domain*.
    """
    if name in CONFIG:
        if domain is None:
            raise ValueError(f"{name!r} is a sweep algorithm and needs a domain")
        scheduler = create_scheduler(CONFIG[name], domain, seed)
        return SchedulerGenerator(scheduler, make_evaluator(cfg.archive))

    latent_dim = cfg.archive.solution_dim
    batch = cfg.emitter.batch_size * cfg.emitter.n_emitters

    if name == "naive":
        evaluator = make_evaluator(replace(cfg.archive, dimensions=[]))
        return NaiveGenerator(evaluator, latent_dim, batch, seed)

    if name == "cma_es":
        evaluator = make_evaluator(replace(cfg.archive, dimensions=[]))
        scheduler = build_scheduler(_one_cell_archive(cfg.archive), cfg.emitter, seed)
        return CMASingleGenerator(scheduler, evaluator)

    if name == "cma_mae":
        evaluator = make_evaluator(cfg.archive)
        scheduler = build_scheduler(cfg.archive, cfg.emitter, seed)
        loop = CMAMAELoop(scheduler, evaluator, threshold_min=cfg.archive.threshold_min)
        return CMAMAEGenerator(loop)

    raise ValueError(
        f"Unknown algorithm {name!r}. Valid: {sorted(CONFIG)} or EVA's "
        f"{list(EVA_GENERATORS)}"
    )
