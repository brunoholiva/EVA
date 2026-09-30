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


class _MixedRestartEmitter(EvolutionStrategyEmitter):
    """Evolution-strategy emitter with random and local-best restarts."""

    def __init__(self, *args, random_restart: bool, restart_seed: int, **kwargs):
        self._random_restart = random_restart
        self._restart_rng = np.random.default_rng(restart_seed)
        self._best_solution = np.asarray(kwargs["x0"], dtype=np.float64).copy()
        self._best_objective = -np.inf
        self._last_restart_kind = "none"
        self._last_restart_distance = 0.0
        super().__init__(*args, **kwargs)

    def tell(self, solution, objective, measures, add_info, **fields):
        """Update CMA-ES and replace archive restarts with mixed restarts."""
        objective = np.asarray(objective)
        best_idx = int(np.argmax(objective))
        if objective[best_idx] > self._best_objective:
            self._best_objective = float(objective[best_idx])
            self._best_solution = np.asarray(solution[best_idx]).copy()

        previous_restarts = self.restarts
        super().tell(solution, objective, measures, add_info, **fields)
        if self.restarts == previous_restarts:
            return

        old_mean = self._opt.mean.copy()
        if self._random_restart:
            restart = self._restart_rng.standard_normal(self.solution_dim)
            self._last_restart_kind = "random"
        else:
            restart = self._best_solution
            self._last_restart_kind = "local_best"
        self._opt.reset(restart)
        self._last_restart_distance = float(np.linalg.norm(restart - old_mean))


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
        Emitter sigma, batch size, count, and restart rule.
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
    n_random = emitter_cfg.n_emitters - emitter_cfg.n_keeper_emitters
    if emitter_cfg.n_keeper_emitters < 0 or n_random < 0:
        raise ValueError("n_keeper_emitters must not exceed n_emitters")

    for i in range(emitter_cfg.n_emitters):
        rng = np.random.default_rng(seed + i)
        x0 = rng.standard_normal(archive_cfg.solution_dim).astype(np.float64)

        emitter = _MixedRestartEmitter(
            archive=archive,
            ranker="imp",
            es="cma_es",
            selection_rule="mu",
            restart_rule=emitter_cfg.restart_every,
            x0=x0,
            sigma0=emitter_cfg.sigma0,
            bounds=[(-emitter_cfg.bounds, emitter_cfg.bounds)]
            * archive_cfg.solution_dim,
            batch_size=emitter_cfg.batch_size,
            seed=seed + i,
            random_restart=i < n_random,
            restart_seed=seed + 10_000 + i,
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
    def report_archive(self) -> GridArchive:
        """The archive used for reporting: result archive if present, else primary."""
        return (
            self._result_archive if self._result_archive is not None else self._archive
        )

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
