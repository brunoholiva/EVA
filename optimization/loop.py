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

GenerationCallback = Callable[[int, EvalResult, GridArchive], None]
StepCallback = Callable[[int, EvalResult, GridArchive], None]


class _SafeEvolutionStrategyEmitter(EvolutionStrategyEmitter):
    """EvolutionStrategyEmitter with empty-archive restart guard.

    Pyribs' default restart path calls ``archive.sample_elites(1)`` without
    checking whether the archive is empty.  With a high ``threshold_min``,
    the archive may be empty for several generations, causing an
    ``IndexError`` on the first restart.  This subclass catches that error
    and falls back to a random Gaussian draw instead.
    """

    def tell(self, solution, objective, measures, add_info=(), **fields):
        try:
            super().tell(
                solution, objective, measures, add_info=add_info, **fields
            )
        except IndexError:
            rng = np.random.default_rng(self._restarts)
            self._opt.reset(rng.standard_normal(self._solution_dim))
            self._ranker.reset(self, self.archive)
            self._restarts += 1


def build_scheduler(
    archive_cfg: ArchiveConfig,
    emitter_cfg: EmitterConfig,
    seed: int,
    warm_start_latents: np.ndarray | None = None,
    warm_start_enabled: bool = False,
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
    warm_start_latents : np.ndarray or None
        Warm-start latent vectors (top-K by P(active) from random sampling).
        Shape ``(n_top, latent_dim)``. If provided, these are used as x0 for
        emitters instead of random initialization.
    warm_start_enabled : bool
        Whether warm-start is enabled. If True, the archive is created with
        threshold_min=0.0 to allow all solutions during warm-start phase.
        Threshold filtering is handled by CMAMAELoop.

    Returns
    -------
    Scheduler
        Configured scheduler with ``n_emitters`` emitters.
    """
    active_dims = archive_cfg.active_dims()
    active_ranges = archive_cfg.active_ranges()

    # If warm-start is enabled, create archive with threshold_min=0.0
    # Threshold filtering will be handled by CMAMAELoop
    archive_threshold = 0.0 if warm_start_enabled else archive_cfg.threshold_min

    archive = GridArchive(
        solution_dim=archive_cfg.solution_dim,
        dims=active_dims,
        ranges=active_ranges,
        learning_rate=archive_cfg.learning_rate,
        threshold_min=archive_threshold,
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
    n_warm = (
        min(len(warm_start_latents), emitter_cfg.n_emitters)
        if warm_start_latents is not None
        else 0
    )

    for i in range(emitter_cfg.n_emitters):
        rng = np.random.default_rng(seed + i)

        if i < n_warm and warm_start_latents is not None:
            x0 = warm_start_latents[i % len(warm_start_latents)].astype(np.float64)
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
            )
        else:
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
    objective_cap : float or None
        If set, objectives are capped at this value before archive insertion
        to mitigate exploitation. Invalid molecules remain at INVALID_MOLECULE_OBJECTIVE.
    warm_start_n_generations : int
        Number of generations for warm-start phase (low threshold).
    warm_start_threshold_min : float
        Minimum objective for archive insertion during warm-start phase.
    threshold_min : float
        Minimum objective for archive insertion after warm-start phase.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        evaluate,
        objective_cap: float | None = None,
        warm_start_n_generations: int = 0,
        warm_start_threshold_min: float = 0.0,
        threshold_min: float = 0.0,
    ) -> None:
        self._scheduler = scheduler
        self._evaluate = evaluate
        self._objective_cap = objective_cap
        self._archive: GridArchive = scheduler.archive
        self._result_archive: GridArchive | None = scheduler.result_archive
        self.last_insertion_stats: dict[str, int] | None = None
        self.last_emitter_stats: list[dict] | None = None
        self._real_objectives: dict[int, float] = {}
        self._result_real_objectives: dict[int, float] = {}

        self._warm_start_n_generations = warm_start_n_generations
        self._warm_start_threshold_min = warm_start_threshold_min
        self._threshold_min = threshold_min

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
        return self._real_objectives

    @property
    def result_real_objectives(self) -> dict[int, float]:
        """Real P(active) values for solutions in the result archive (uncapped)."""
        return self._result_real_objectives

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
            old_occupied = self._get_occupied_cells(self._archive)
            old_archive_objectives = dict(old_occupied)
            old_result_objectives = (
                self._get_occupied_cells(self._result_archive)
                if self._result_archive is not None
                else {}
            )

            z = self._scheduler.ask()
            result = self._evaluate(z)

            objectives = self._apply_threshold_and_cap(result.objectives, gen)
            self._scheduler.tell(objectives, result.measures)

            # Track real P(active) for solutions added to archive
            _track_real_objectives(
                self._archive,
                z,
                result,
                objectives,
                result.p_active,
                self._real_objectives,
                old_archive_objectives,
            )

            # Track real P(active) for solutions added to result_archive
            if self._result_archive is not None:
                _track_real_objectives(
                    self._result_archive,
                    z,
                    result,
                    objectives,
                    result.p_active,
                    self._result_real_objectives,
                    old_result_objectives,
                )

            self.last_insertion_stats = self._compute_insertion_stats(
                result, objectives, len(z), old_occupied
            )
            self.last_emitter_stats = self._compute_emitter_stats()
            
            if result.timings is not None:
                result.timings.archive_ops = time.time() - t_archive_start

            # Update main bar with archive stats
            if on_step:
                on_step(gen, result, self._archive)

            if on_generation and (gen % eval_every == 0 or gen == n_generations - 1):
                on_generation(gen, result, self._archive)

        return self._archive

    def _get_occupied_cells(self, archive: GridArchive) -> dict[int, float]:
        """Return ``{cell_index: objective}`` for all occupied cells."""
        occupied: dict[int, float] = {}
        if len(archive) > 0:
            data = archive.data()
            for idx, obj in zip(data["index"], data["objective"]):
                occupied[int(idx)] = float(obj)
        return occupied

    def _compute_insertion_stats(
        self,
        result: EvalResult,
        objectives: np.ndarray,
        n_total: int,
        old_occupied: dict[int, float],
    ) -> dict[str, int]:
        """Categorise each candidate's archive insertion outcome."""
        inserted_new = 0
        improved_existing = 0
        rejected = 0

        lb = self._archive.lower_bounds
        ub = self._archive.upper_bounds
        n_dims = self._archive.measure_dim
        all_indices = self._archive.index_of(result.measures)

        for i in range(n_total):
            if objectives[i] == INVALID_MOLECULE_OBJECTIVE:
                rejected += 1
                continue

            meas = result.measures[i]
            in_bounds = True
            for d in range(n_dims):
                if meas[d] < lb[d] or meas[d] > ub[d]:
                    in_bounds = False
                    break
            if not in_bounds:
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

    def _compute_emitter_stats(self) -> list[dict]:
        """Return per-emitter distance and restart count."""
        stats = []
        for i, emitter in enumerate(self._scheduler._emitters):
            dist = np.linalg.norm(emitter._opt.mean - emitter.x0)
            stats.append(
                {
                    "id": i,
                    "distance": float(dist),
                    "restarts": emitter.restarts,
                }
            )
        return stats

    def _apply_threshold_and_cap(self, objectives: np.ndarray, gen: int) -> np.ndarray:
        """Apply threshold and cap to objectives based on current generation.

        During warm-start phase (gen < warm_start_n_generations), uses
        warm_start_threshold_min. After warm-start, uses threshold_min.
        Invalid molecules (INVALID_MOLECULE_OBJECTIVE) are preserved.

        Parameters
        ----------
        objectives : np.ndarray
            Raw objectives from evaluator.
        gen : int
            Current generation number.

        Returns
        -------
        np.ndarray
            Filtered and capped objectives.
        """
        if self._objective_cap is None:
            thresholded = objectives.copy()
        else:
            thresholded = objectives.copy()
            valid_mask = thresholded != INVALID_MOLECULE_OBJECTIVE
            thresholded[valid_mask] = np.minimum(
                thresholded[valid_mask], self._objective_cap
            )

        # Apply threshold based on warm-start phase
        if gen < self._warm_start_n_generations:
            current_threshold = self._warm_start_threshold_min
        else:
            current_threshold = self._threshold_min

        if current_threshold > 0:
            valid_mask = thresholded != INVALID_MOLECULE_OBJECTIVE
            below_threshold = thresholded < current_threshold
            thresholded[valid_mask & below_threshold] = INVALID_MOLECULE_OBJECTIVE

        return thresholded

    def _cap_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """Cap objectives at the configured threshold.

        Invalid molecules (INVALID_MOLECULE_OBJECTIVE) are preserved.
        Returns a new array; the input is not modified.

        .. deprecated::
            Use :meth:`_apply_threshold_and_cap` instead.
        """
        if self._objective_cap is None:
            return objectives
        capped = objectives.copy()
        valid_mask = capped != INVALID_MOLECULE_OBJECTIVE
        capped[valid_mask] = np.minimum(capped[valid_mask], self._objective_cap)
        return capped


def _track_real_objectives(
    archive: GridArchive,
    z: np.ndarray,
    result: EvalResult,
    objectives: np.ndarray,
    real_objectives_array: np.ndarray,
    real_objectives_dict: dict[int, float],
    old_archive_objectives: dict[int, float],
) -> None:
    """Track real P(active) values for solutions actually accepted into archive.

    Compares archive state before and after tell() to identify cells that were
    updated (new or strictly improved). For each updated cell, records the real
    objective of the best batch solution mapping to that cell.

    The dict is keyed by archive cell index (not batch index) so it can be
    looked up correctly when exporting the archive.
    """
    lb = archive.lower_bounds
    ub = archive.upper_bounds
    n_dims = archive.measure_dim

    # Get new archive state after tell()
    new_archive_objectives: dict[int, float] = {}
    if len(archive) > 0:
        data = archive.data()
        for idx, obj in zip(data["index"], data["objective"]):
            new_archive_objectives[int(idx)] = float(obj)

    # Find cells that were updated (new or strictly improved)
    updated_cells: set[int] = set()
    for cell_idx, new_obj in new_archive_objectives.items():
        old_obj = old_archive_objectives.get(cell_idx)
        if old_obj is None or new_obj > old_obj + 1e-9:
            updated_cells.add(cell_idx)

    if not updated_cells:
        return

    # Group batch solutions by cell index
    solutions_by_cell: dict[int, list[int]] = {}
    for i in range(len(z)):
        if objectives[i] == INVALID_MOLECULE_OBJECTIVE:
            continue

        meas = result.measures[i]
        in_bounds = True
        for d in range(n_dims):
            if meas[d] < lb[d] or meas[d] > ub[d]:
                in_bounds = False
                break

        if not (in_bounds and real_objectives_array[i] != INVALID_MOLECULE_OBJECTIVE):
            continue

        cell_idx = archive.index_of_single(meas)
        if cell_idx < 0:
            continue

        if cell_idx in updated_cells:
            if cell_idx not in solutions_by_cell:
                solutions_by_cell[cell_idx] = []
            solutions_by_cell[cell_idx].append(i)

    # For each updated cell, find the best solution (highest capped objective)
    for cell_idx, batch_indices in solutions_by_cell.items():
        best_idx = max(batch_indices, key=lambda i: objectives[i])
        real_objectives_dict[cell_idx] = float(real_objectives_array[best_idx])
