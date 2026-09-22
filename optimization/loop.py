"""CMA-MAE optimization loop via pyribs."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from config import ArchiveConfig, EmitterConfig
from optimization.constants import INVALID_MOLECULE_OBJECTIVE
from optimization.evaluator import EvalResult
from optimization.tracking import (
    RealObjectiveTracker,
    compute_emitter_insertions,
    compute_emitter_spread,
    compute_emitter_stats,
    compute_insertion_stats,
    occupied_cells,
)

GenerationCallback = Callable[[int, EvalResult, GridArchive], None]
StepCallback = Callable[[int, EvalResult, GridArchive], None]


class _SafeEvolutionStrategyEmitter(EvolutionStrategyEmitter):
    """EvolutionStrategyEmitter with hybrid restart and empty-archive guard.

    Replaces pyribs' default restart path with a configurable strategy:
    - ``"elite"``: restart from archive sample_elites (pyribs default)
    - ``"random"``: restart from fresh Gaussian N(0, 1)
    - ``"hybrid"``: with probability ``random_restart_prob``, random; else elite

    Also guards against empty-archive ``IndexError``.
    """

    def __init__(self, *args, restart_mode="hybrid", random_restart_prob=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self._restart_mode = restart_mode
        self._random_restart_prob = random_restart_prob
        self._restart_rng = np.random.default_rng(
            np.random.SeedSequence(kwargs.get("seed", 0)).spawn(1)[0]
        )

    def _sample_restart_x0(self):
        """Choose a new x0 based on restart mode."""
        if len(self.archive) == 0:
            return self._restart_rng.standard_normal(self._solution_dim)

        if self._restart_mode == "random":
            return self._restart_rng.standard_normal(self._solution_dim)
        if self._restart_mode == "elite":
            return self.archive.sample_elites(1)["solution"][0]
        if self._restart_rng.random() < self._random_restart_prob:
            return self._restart_rng.standard_normal(self._solution_dim)
        return self.archive.sample_elites(1)["solution"][0]

    def tell(self, solution, objective, measures, add_info=(), **fields):
        try:
            super().tell(solution, objective, measures, add_info=add_info, **fields)
        except IndexError:
            new_x0 = self._sample_restart_x0()
            self._opt.reset(new_x0)
            self._ranker.reset(self, self.archive)
            self._restarts += 1


def build_scheduler(
    archive_cfg: ArchiveConfig,
    emitter_cfg: EmitterConfig,
    seed: int,
) -> Scheduler:
    """Construct the pyribs scheduler from config.

    Parameters
    ----------
    archive_cfg : ArchiveConfig
        Archive dimensions, ranges, and learning rate.
    emitter_cfg : EmitterConfig
        Emitter sigma, batch size, and count.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Scheduler
        Configured scheduler with ``n_emitters`` emitters.
    """
    active_dims = archive_cfg.active_dims()
    active_ranges = archive_cfg.active_ranges()

    archive = GridArchive(
        solution_dim=archive_cfg.solution_dim,
        dims=active_dims,
        ranges=active_ranges,
        learning_rate=archive_cfg.learning_rate,
        threshold_min=archive_cfg.threshold_min,
        seed=seed,
    )

    result_archive = GridArchive(
        solution_dim=archive_cfg.solution_dim,
        dims=active_dims,
        ranges=active_ranges,
        learning_rate=1.0,
        threshold_min=0.0,
    )

    emitters = []
    n_emitters = emitter_cfg.n_emitters

    for i in range(n_emitters):
        rng = np.random.default_rng(seed + i)
        x0 = rng.standard_normal(archive_cfg.solution_dim).astype(np.float64)

        emitter = _SafeEvolutionStrategyEmitter(
            archive=archive,
            ranker="imp",
            es="cma_es",
            selection_rule="mu",
            restart_rule="no_improvement",
            x0=x0,
            sigma0=emitter_cfg.sigma0,
            bounds=None,
            batch_size=emitter_cfg.batch_size,
            seed=seed + i,
            restart_mode=emitter_cfg.restart_mode,
            random_restart_prob=emitter_cfg.random_restart_prob,
        )
        emitters.append(emitter)

    return Scheduler(archive, emitters, result_archive)


class CMAMAELoop:
    """CMA-MAE optimization in ChemBed latent space.

    Parameters
    ----------
    scheduler : Scheduler
        Configured pyribs scheduler.
    evaluate : callable
        Function ``(z) -> EvalResult`` that scores latent vectors.
    threshold_min : float
        Minimum objective for archive insertion.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        evaluate,
        threshold_min: float = 0.0,
    ) -> None:
        self._scheduler = scheduler
        self._evaluate = evaluate
        self._archive: GridArchive = scheduler.archive
        self._result_archive: GridArchive | None = scheduler.result_archive
        self._tracker = RealObjectiveTracker()
        self.last_insertion_stats: dict[str, int] | None = None
        self.last_emitter_stats: list[dict] | None = None
        self.last_emitter_insertions: list[dict] | None = None
        self.last_emitter_spread: float | None = None
        self._threshold_min = threshold_min
        self._emitter_totals: list[int] | None = None

    @property
    def archive(self) -> GridArchive:
        """The primary (CMA-MAE) archive."""
        return self._archive

    @property
    def result_archive(self) -> GridArchive | None:
        """Best-so-far archive (tracks elite per cell)."""
        return self._result_archive

    @property
    def scheduler(self) -> Scheduler:
        """The underlying scheduler."""
        return self._scheduler

    @property
    def real_objectives(self) -> dict[int, float]:
        """Real P(active) values for solutions in the archive (uncapped)."""
        return self._tracker.primary

    @property
    def result_real_objectives(self) -> dict[int, float]:
        """Real P(active) values for solutions in the result archive (uncapped)."""
        return self._tracker.result

    @property
    def emitter_insertion_totals(self) -> list[int]:
        """Cumulative per-emitter insertions (new + improved) for the run."""
        return list(self._emitter_totals) if self._emitter_totals else []

    def _accumulate_emitter_insertions(self, insertions: list[dict] | None) -> None:
        """Accumulate per-emitter insertion totals across generations."""
        if insertions is None:
            return
        if self._emitter_totals is None:
            self._emitter_totals = [0] * len(insertions)
        for entry in insertions:
            em_id = entry["id"]
            if em_id < len(self._emitter_totals):
                self._emitter_totals[em_id] += (
                    entry["inserted_new"] + entry["improved_existing"]
                )

    def run(
        self,
        n_generations: int,
        eval_every: int = 1,
        on_generation: GenerationCallback | None = None,
        on_step: StepCallback | None = None,
        start_gen: int = 0,
    ) -> GridArchive:
        """Execute the CMA-MAE loop.

        Parameters
        ----------
        n_generations : int
            Number of generations to run.
        eval_every : int
            How often to call the generation callback.
        on_generation : callable or None
            ``fn(gen, result, archive)`` called every *eval_every*
            generations and at the final generation.
        on_step : callable or None
            ``fn(gen, result, archive)`` called every generation (for
            progress bar updates).
        start_gen : int, default=0
            Generation to start from (for resuming a previous run).

        Returns
        -------
        GridArchive
            The final archive of scored candidates.
        """
        for gen in range(start_gen, n_generations):
            t_archive_start = time.time()
            old_primary_cells = occupied_cells(self._archive)
            old_result_cells = (
                occupied_cells(self._result_archive)
                if self._result_archive is not None
                else {}
            )

            z = self._scheduler.ask()
            result = self._evaluate(z)

            objectives = self._apply_threshold_and_cap(result.objectives, gen)
            self._scheduler.tell(objectives, result.measures)

            self._tracker.update(
                self._archive,
                self._result_archive,
                z,
                result,
                objectives,
                old_primary_cells,
                old_result_cells,
            )

            self.last_insertion_stats = compute_insertion_stats(
                result,
                objectives,
                old_primary_cells,
                self._archive,
            )
            self.last_emitter_stats = compute_emitter_stats(self._scheduler)
            self.last_emitter_insertions = compute_emitter_insertions(
                result,
                objectives,
                old_primary_cells,
                self._archive,
                self._scheduler,
            )
            self.last_emitter_spread = compute_emitter_spread(self._scheduler)
            self._accumulate_emitter_insertions(self.last_emitter_insertions)

            if result.timings is not None:
                result.timings.archive_ops = time.time() - t_archive_start

            if on_step:
                on_step(gen, result, self._archive)

            if on_generation and (gen % eval_every == 0 or gen == n_generations - 1):
                on_generation(gen, result, self._archive)

        return self._archive

    def _apply_threshold_and_cap(self, objectives: np.ndarray, gen: int) -> np.ndarray:
        """Apply threshold to objectives.

        Invalid molecules (INVALID_MOLECULE_OBJECTIVE) are preserved.
        Molecules below threshold_min are marked invalid.

        Parameters
        ----------
        objectives : np.ndarray
            Raw objectives from evaluator.
        gen : int
            Current generation number.

        Returns
        -------
        np.ndarray
            Thresholded objectives.
        """
        thresholded = objectives.copy()

        if self._threshold_min > 0:
            valid_mask = thresholded != INVALID_MOLECULE_OBJECTIVE
            below_threshold = thresholded < self._threshold_min
            thresholded[valid_mask & below_threshold] = INVALID_MOLECULE_OBJECTIVE

        return thresholded
