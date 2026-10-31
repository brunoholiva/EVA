"""Multi-population CMA-ES via ``pycma`` and random-prior baseline solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import cma
import numpy as np

from config import CMAConfig

_RESERVED_PYCMA_OPTS = frozenset({"popsize", "bounds", "randn", "seed"})


class _Randn:
    """Picklable standard-normal sampler isolating a walker from global RNG."""

    def __init__(self, seed: int | np.random.SeedSequence) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, lam: int, dim: int) -> np.ndarray:
        """Draw a ``(lam, dim)`` standard-normal array."""
        return self.rng.standard_normal((lam, dim))


@dataclass(frozen=True)
class WalkerStats:
    """Diagnostic snapshot of a single CMA-ES walker."""

    walker_id: int
    sigma: float
    mean_norm: float
    axis_ratio: float
    restarts: int
    gens_in_run: int
    last_stop_reasons: dict[str, Any]


class Solver(Protocol):
    """Protocol implemented by latent-space proposal solvers."""

    batch_size: int

    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """Propose ``(z, walker_ids)`` for one batch step."""
        ...

    def tell(self, z: np.ndarray, objectives: np.ndarray) -> None:
        """Report batch maximization objectives back to the solver."""
        ...

    def walker_stats(self) -> list[WalkerStats]:
        """Return per-walker diagnostic snapshots (empty if stateless)."""
        ...

    def walker_spread(self) -> float:
        """Return mean coordinate-wise std across walker means."""
        ...

    def total_restarts(self) -> int:
        """Return cumulative restarts across all walkers."""
        ...


class CMAWalker:
    """Single independent ``pycma`` population with isolated RNG and restarts.

    Parameters
    ----------
    walker_id : int
        Zero-based identifier for attribution in logs and ``candidates.csv``.
    solution_dim : int
        Dimensionality of the latent search space.
    cfg : CMAConfig
        Hyperparameters and ``pycma`` options passthrough.
    init_seed : int or SeedSequence
        Seed for sampling initial/restart means ``x0``.
    es_seed : int or SeedSequence
        Seed for ``CMAEvolutionStrategy`` internal proposals.
    """

    def __init__(
        self,
        walker_id: int,
        solution_dim: int,
        cfg: CMAConfig,
        init_seed: int | np.random.SeedSequence,
        es_seed: int | np.random.SeedSequence,
    ) -> None:
        overlap = _RESERVED_PYCMA_OPTS.intersection(cfg.opts)
        if overlap:
            raise ValueError(
                f"cma.opts must not set {sorted(overlap)}; configure those on [cma]"
            )

        self.walker_id = walker_id
        self.solution_dim = solution_dim
        self._cfg = cfg
        self._init_rng = np.random.default_rng(init_seed)
        self._randn = _Randn(es_seed)
        self._restarts = 0
        self._gens_in_run = 0
        self._last_stop_reasons: dict[str, Any] = {}
        self._es = self._make_es(self._sample_x0())

    def _sample_x0(self) -> np.ndarray:
        """Draw a starting mean ``x0`` in latent space."""
        if self._cfg.init_std == 0.0:
            x0 = np.zeros(self.solution_dim, dtype=np.float64)
        else:
            x0 = self._init_rng.normal(
                0.0, self._cfg.init_std, size=self.solution_dim
            ).astype(np.float64)

        if self._cfg.bounds is not None:
            limit = 0.999 * self._cfg.bounds
            x0 = np.clip(x0, -limit, limit)
        return x0

    def _make_es(self, x0: np.ndarray) -> cma.CMAEvolutionStrategy:
        """Instantiate ``cma.CMAEvolutionStrategy`` at *x0*."""
        opts: dict[str, Any] = dict(self._cfg.opts)
        opts.setdefault("verbose", -9)
        opts["popsize"] = self._cfg.popsize
        opts["randn"] = self._randn
        opts["seed"] = np.nan
        if self._cfg.bounds is not None:
            opts["bounds"] = [-self._cfg.bounds, self._cfg.bounds]
        return cma.CMAEvolutionStrategy(x0, self._cfg.sigma0, opts)

    @property
    def popsize(self) -> int:
        """Number of candidate solutions sampled per step."""
        return int(self._es.popsize)

    @property
    def mean(self) -> np.ndarray:
        """Current distribution mean in phenotype (latent) coordinates."""
        mean = np.asarray(self._es.mean, dtype=np.float64)
        gp = getattr(self._es, "gp", None)
        if gp is not None and hasattr(gp, "pheno"):
            mean = np.asarray(gp.pheno(mean), dtype=np.float64)
        bh = getattr(self._es, "boundary_handler", None)
        if bh is not None and hasattr(bh, "repair"):
            mean = np.asarray(bh.repair(mean), dtype=np.float64)
        return mean

    @property
    def sigma(self) -> float:
        """Current overall step size."""
        return float(self._es.sigma)

    @property
    def axis_ratio(self) -> float:
        """Principal-axis standard-deviation ratio ``max(D) / min(D)``."""
        spectrum = getattr(self._es.sm, "eigenspectrum", None)
        if spectrum is None or len(spectrum) == 0:
            return 1.0
        d = np.asarray(spectrum, dtype=np.float64)
        return float(np.max(d) / max(float(np.min(d)), 1e-12))

    @property
    def restarts(self) -> int:
        """Cumulative restarts triggered for this walker."""
        return self._restarts

    @property
    def gens_in_run(self) -> int:
        """Steps elapsed since initialization or the most recent restart."""
        return self._gens_in_run

    def stats(self) -> WalkerStats:
        """Return a diagnostic snapshot of this walker."""
        return WalkerStats(
            walker_id=self.walker_id,
            sigma=self.sigma,
            mean_norm=float(np.linalg.norm(self.mean)),
            axis_ratio=self.axis_ratio,
            restarts=self._restarts,
            gens_in_run=self._gens_in_run,
            last_stop_reasons=dict(self._last_stop_reasons),
        )

    def ask(self) -> np.ndarray:
        """Sample ``(popsize, solution_dim)`` candidate vectors."""
        return np.asarray(self._es.ask(), dtype=np.float64)

    def tell(self, z: np.ndarray, objectives: np.ndarray) -> bool:
        """Update CMA-ES from maximization *objectives* and restart if stopped.

        ``pycma`` minimizes losses, so ``objectives`` are negated (mapping
        ``P(active) in [0, 1]`` to ``[-1, 0]`` and invalid/duplicate ``-1.0``
        to ``+1.0``).

        Parameters
        ----------
        z : np.ndarray of shape ``(popsize, solution_dim)``
            Candidate vectors returned by the preceding :meth:`ask`.
        objectives : np.ndarray of shape ``(popsize,)``
            Maximization objectives from the evaluator.

        Returns
        -------
        bool
            ``True`` if the walker restarted at the end of this step.
        """
        losses = -np.asarray(objectives, dtype=np.float64)
        self._es.tell(z, losses)
        self._gens_in_run += 1

        stop = dict(self._es.stop())
        hit_gen_cap = (
            self._cfg.max_gens_per_run > 0
            and self._gens_in_run >= self._cfg.max_gens_per_run
        )
        if stop or hit_gen_cap:
            if not stop:
                stop = {"max_gens_per_run": self._cfg.max_gens_per_run}
            self._last_stop_reasons = stop
            self._restarts += 1
            self._gens_in_run = 0
            self._es = self._make_es(self._sample_x0())
            return True
        return False


class MultiCMASolver:
    """Coordinates *K* independent :class:`CMAWalker` populations in batch.

    Parameters
    ----------
    solution_dim : int
        Dimensionality of the latent search space.
    cfg : CMAConfig
        CMA hyperparameters and options.
    seed : int
        Master random seed used to spawn independent per-walker RNG streams.
    """

    def __init__(self, solution_dim: int, cfg: CMAConfig, seed: int) -> None:
        streams = np.random.SeedSequence(seed).spawn(2 * cfg.n_populations)
        self.walkers: list[CMAWalker] = [
            CMAWalker(
                walker_id=i,
                solution_dim=solution_dim,
                cfg=cfg,
                init_seed=streams[2 * i],
                es_seed=streams[2 * i + 1],
            )
            for i in range(cfg.n_populations)
        ]
        self.batch_size = sum(w.popsize for w in self.walkers)
        self._last_slices: list[tuple[int, int]] = []

    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """Ask every walker and concatenate into one GPU-friendly batch."""
        batches: list[np.ndarray] = []
        walker_ids: list[np.ndarray] = []
        slices: list[tuple[int, int]] = []
        cursor = 0
        for walker in self.walkers:
            x = walker.ask()
            n = len(x)
            batches.append(x)
            walker_ids.append(np.full(n, walker.walker_id, dtype=np.int64))
            slices.append((cursor, cursor + n))
            cursor += n
        self._last_slices = slices
        self.batch_size = cursor
        return np.concatenate(batches, axis=0), np.concatenate(walker_ids, axis=0)

    def tell(self, z: np.ndarray, objectives: np.ndarray) -> None:
        """Dispatch slices of *z* and *objectives* back to each walker."""
        if len(self._last_slices) != len(self.walkers):
            raise RuntimeError("tell() called before ask()")
        for walker, (start, end) in zip(self.walkers, self._last_slices):
            walker.tell(z[start:end], objectives[start:end])

    def walker_stats(self) -> list[WalkerStats]:
        """Return diagnostic snapshots for all walkers."""
        return [w.stats() for w in self.walkers]

    def walker_spread(self) -> float:
        """Return mean coordinate-wise std across walker means."""
        if len(self.walkers) <= 1:
            return 0.0
        means = np.stack([w.mean for w in self.walkers], axis=0)
        return float(np.mean(np.std(means, axis=0)))

    def total_restarts(self) -> int:
        """Return cumulative restarts across all walkers."""
        return sum(w.restarts for w in self.walkers)


class NaiveSolver:
    """Stateless ``N(0, I)`` random-prior baseline in latent space.

    Parameters
    ----------
    solution_dim : int
        Dimensionality of the latent search space.
    batch_size : int
        Number of random vectors proposed per step.
    seed : int
        Random seed.
    """

    def __init__(self, solution_dim: int, batch_size: int, seed: int) -> None:
        self.solution_dim = solution_dim
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)

    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``(batch_size, solution_dim)`` standard-normal vectors."""
        z = self._rng.standard_normal((self.batch_size, self.solution_dim))
        walker_ids = np.full(self.batch_size, -1, dtype=np.int64)
        return z, walker_ids

    def tell(self, z: np.ndarray, objectives: np.ndarray) -> None:
        """No-op for stateless random sampling."""

    def walker_stats(self) -> list[WalkerStats]:
        """Return an empty list (no walkers)."""
        return []

    def walker_spread(self) -> float:
        """Return 0.0 (no walkers)."""
        return 0.0

    def total_restarts(self) -> int:
        """Return 0 (no walkers)."""
        return 0


def build_solver(
    solver_name: str,
    solution_dim: int,
    cfg: CMAConfig,
    seed: int,
) -> MultiCMASolver | NaiveSolver:
    """Build a latent-space solver from configuration.

    Parameters
    ----------
    solver_name : str
        ``"pycma"`` or ``"naive"``.
    solution_dim : int
        Working latent dimensionality.
    cfg : CMAConfig
        CMA hyperparameters (``total_batch_size`` also sizes ``naive`` batches).
    seed : int
        Random seed.

    Returns
    -------
    MultiCMASolver or NaiveSolver
    """
    if solver_name == "pycma":
        return MultiCMASolver(solution_dim=solution_dim, cfg=cfg, seed=seed)
    if solver_name == "naive":
        return NaiveSolver(
            solution_dim=solution_dim, batch_size=cfg.total_batch_size, seed=seed
        )
    raise ValueError(f"Unknown solver {solver_name!r}; expected 'pycma' or 'naive'")
