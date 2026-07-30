"""CMA-MAE optimization loop via pyribs."""

from __future__ import annotations

from typing import Callable

import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from config import ArchiveConfig, EmitterConfig
from optimization.constants import INVALID_MOLECULE_OBJECTIVE
from optimization.evaluator import EvalResult

GenerationCallback = Callable[[int, EvalResult, GridArchive], None]
StepCallback = Callable[[int, EvalResult, GridArchive], None]


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
        Configured scheduler with ``n_emitters`` EvolutionStrategyEmitters.
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
    )

    emitters = []
    for i in range(emitter_cfg.n_emitters):
        rng = np.random.default_rng(seed + i)
        x0 = rng.standard_normal(archive_cfg.solution_dim).astype(np.float64)
        emitter = EvolutionStrategyEmitter(
            archive=archive,
            ranker="imp",
            es="cma_es",
            selection_rule="mu",
            restart_rule="basic",
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
    """

    def __init__(self, scheduler: Scheduler, evaluate) -> None:
        self._scheduler = scheduler
        self._evaluate = evaluate
        self._archive: GridArchive = scheduler.archive
        self._result_archive: GridArchive | None = scheduler.result_archive
        self.last_insertion_stats: dict[str, int] | None = None
        self.last_emitter_stats: list[dict] | None = None

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

    def seed_archive(self, z_seeds: np.ndarray) -> EvalResult:
        """Evaluate initial seeds and add them to both archives.

        Parameters
        ----------
        z_seeds : np.ndarray of shape ``(n, latent_dim)``
            Latent vectors for the initial population.

        Returns
        -------
        EvalResult
            Scoring results for the seed population.
        """
        result = self._evaluate(z_seeds)
        _add_to_archive(self._archive, z_seeds, result)
        if self._result_archive is not None:
            _add_to_archive(self._result_archive, z_seeds, result)
        return result

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
        archive_novelty = self._evaluate.archive_novelty
        decode_fn = self._evaluate.decode_fn

        for gen in range(start_gen, n_generations):
            old_primary, old_result = self._snapshot_archive()

            z = self._scheduler.ask()
            result = self._evaluate(z)

            old_occupied = self._get_occupied_cells(self._archive)
            self._scheduler.tell(result.objectives, result.measures)

            self.last_insertion_stats = self._compute_insertion_stats(
                result, len(z), old_occupied
            )
            self.last_emitter_stats = self._compute_emitter_stats()

            if archive_novelty is not None:
                self._sync_novelty_cache(
                    old_primary, old_result, archive_novelty, decode_fn
                )

            if on_step:
                on_step(gen, result, self._archive)

            if on_generation and (gen % eval_every == 0 or gen == n_generations - 1):
                on_generation(gen, result, self._archive)

        return self._archive

    def _snapshot_archive(self) -> tuple[set[int], set[int]]:
        """Return occupied cell indices for both archives."""
        if len(self._archive) == 0:
            primary = set()
        else:
            primary = set(self._archive.data()["index"].tolist())

        if self._result_archive is None or len(self._result_archive) == 0:
            result = set()
        else:
            result = set(self._result_archive.data()["index"].tolist())

        return primary, result

    def _get_occupied_cells(self, archive: GridArchive) -> dict[int, float]:
        """Return ``{cell_index: objective}`` for all occupied cells."""
        occupied: dict[int, float] = {}
        if len(archive) > 0:
            data = archive.data()
            for idx, obj in zip(data["index"], data["objective"]):
                occupied[int(idx)] = float(obj)
        return occupied

    def _compute_insertion_stats(
        self, result: EvalResult, n_total: int, old_occupied: dict[int, float]
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
            if result.objectives[i] == INVALID_MOLECULE_OBJECTIVE:
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
            elif result.objectives[i] > old_occupied[cell_idx]:
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

    def _sync_novelty_cache(
        self,
        old_primary: set[int],
        old_result: set[int],
        archive_novelty,
        decode_fn,
    ) -> None:
        """Update the archive novelty cache with newly inserted entries."""
        for archive, old_indices in [
            (self._archive, old_primary),
            (self._result_archive, old_result),
        ]:
            if archive is None or len(archive) == 0:
                continue
            new_data = archive.data()
            new_indices = set(new_data["index"].tolist())
            added = new_indices - old_indices
            if not added:
                continue
            added_mask = np.isin(new_data["index"], list(added))
            added_solutions = new_data["solution"][added_mask]
            smiles = decode_fn(added_solutions)
            valid_smiles = [s for s in smiles if s != ""]
            if valid_smiles:
                cell_indices = new_data["index"][added_mask]
                archive_novelty.update(cell_indices, valid_smiles)


def _add_to_archive(archive: GridArchive, z: np.ndarray, result: EvalResult) -> None:
    """Add valid candidates from *result* to *archive* one by one."""
    for i in range(len(z)):
        if result.objectives[i] != INVALID_MOLECULE_OBJECTIVE:
            archive.add(
                solution=z[i : i + 1],
                objective=result.objectives[i : i + 1],
                measures=result.measures[i : i + 1],
            )
