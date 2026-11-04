"""Latent-space search solvers: multi-population ``pycma`` and random baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import cma
import numpy as np

from config import CMAConfig

_RESERVED_OPTS = frozenset({"popsize", "randn", "seed"})


class _Randn:
    """Picklable standard-normal callable isolating a walker from global RNG."""

    def __init__(self, seed: int | np.random.SeedSequence) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, lam: int, dim: int) -> np.ndarray:
        """Draw a ``(lam, dim)`` standard-normal matrix."""
        return self.rng.standard_normal((lam, dim))


@dataclass(frozen=True)
class SolverStats:
    """Compact per-step diagnostics logged into ``curve.csv``."""

    walker_spread: float = 0.0
    mean_norm_mean: float = 0.0
    sigma_mean: float = 0.0
    sigma_min: float = 0.0
    axis_ratio_max: float = 0.0
    restarts: int = 0


@dataclass(frozen=True)
class WalkerState:
    """Per-walker state snapshot for ``walkers.csv``."""

    walker_id: int
    popsize: int = 0
    steps: int = 0
    mean_norm: float = 0.0
    sigma: float = 0.0
    axis_ratio: float = 0.0
    restarts: int = 0
    steps_since_restart: int = 0


class Solver(Protocol):
    """Uniform ``ask`` / ``tell`` interface for latent-space search."""

    batch_size: int

    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(z, walker_ids)`` for one batch step."""
        ...

    def tell(self, z: np.ndarray, objectives: np.ndarray) -> None:
        """Update solver state from maximization *objectives*."""
        ...

    def stats(self) -> SolverStats:
        """Return current diagnostic summary."""
        ...

    def walker_states(self) -> list[WalkerState]:
        """Return per-walker state snapshots (empty for stateless baselines)."""
        ...


class CMAWalker:
    """One independent ``pycma`` CMA-ES population in unbounded latent space."""

    def __init__(
        self,
        walker_id: int,
        dim: int,
        cfg: CMAConfig,
        init_seed: int | np.random.SeedSequence,
        es_seed: int | np.random.SeedSequence,
    ) -> None:
        overlap = _RESERVED_OPTS.intersection(cfg.opts)
        if overlap:
            raise ValueError(
                f"cma.opts must not set {sorted(overlap)}; configure those on [cma]"
            )
        self.walker_id = walker_id
        self.dim = dim
        self._cfg = cfg
        self._init_rng = np.random.default_rng(init_seed)
        self._randn = _Randn(es_seed)
        self._current_popsize = cfg.popsize
        self.steps = 0
        self.restarts = 0
        self.steps_since_restart = 0
        self._cycle_best = -np.inf
        self._cycle_best_z: np.ndarray | None = None
        self._stagnant_steps = 0
        self._es = self._new_es()

    def _new_es(
        self, x0: np.ndarray | None = None, sigma: float | None = None
    ) -> cma.CMAEvolutionStrategy:
        """Instantiate ``CMAEvolutionStrategy`` at ``x0`` and ``sigma``."""
        if x0 is None:
            if self._cfg.init_std == 0.0:
                x0 = np.zeros(self.dim, dtype=np.float64)
            else:
                x0 = self._init_rng.normal(0.0, self._cfg.init_std, size=self.dim)
        if sigma is None:
            sigma = self._cfg.sigma0

        opts: dict[str, Any] = dict(self._cfg.opts)
        opts.setdefault("verbose", -9)
        opts["popsize"] = self._current_popsize
        opts["randn"] = self._randn
        opts["seed"] = np.nan
        return cma.CMAEvolutionStrategy(x0, sigma, opts)

    @property
    def popsize(self) -> int:
        """Population size per step."""
        return int(self._es.popsize)

    @property
    def mean(self) -> np.ndarray:
        """Current distribution mean (repaired if bounds are set in ``opts``)."""
        m = np.asarray(self._es.mean, dtype=np.float64)
        bh = getattr(self._es, "boundary_handler", None)
        if bh is not None and hasattr(bh, "repair"):
            m = np.asarray(bh.repair(m), dtype=np.float64)
        return m

    @property
    def mean_norm(self) -> float:
        """Euclidean distance ``||mean||_2`` from the latent origin."""
        return float(np.linalg.norm(self.mean))

    @property
    def sigma(self) -> float:
        """Current step-size ``sigma``."""
        return float(self._es.sigma)

    @property
    def axis_ratio(self) -> float:
        """Ratio of largest to smallest principal-axis std ``max(D) / min(D)``."""
        spectrum = getattr(self._es.sm, "eigenspectrum", None)
        if spectrum is None or len(spectrum) == 0:
            return 1.0
        d = np.asarray(spectrum, dtype=np.float64)
        sigma_vec = getattr(self._es, "sigma_vec", None)
        if sigma_vec is not None and hasattr(sigma_vec, "scaling"):
            scaling = getattr(sigma_vec, "scaling", None)
            if scaling is not None:
                d = d * np.asarray(scaling, dtype=np.float64)
        return float(np.max(d) / max(float(np.min(d)), 1e-12))

    def ask(self, number: int | None = None) -> np.ndarray:
        """Propose ``(popsize, dim)`` or ``(number, dim)`` candidate vectors."""
        if number is not None:
            return np.asarray(self._es.ask(number), dtype=np.float64)
        return np.asarray(self._es.ask(), dtype=np.float64)

    def resample_indices(self, z_batch: np.ndarray, indices: list[int]) -> np.ndarray:
        """Replace candidate vectors at *indices* with fresh samples."""
        if not indices:
            return z_batch
        fresh = np.asarray(self._es.ask(len(indices)), dtype=np.float64)
        for idx, new_x in zip(indices, fresh):
            z_batch[idx] = new_x
        return z_batch

    def tell(self, z: np.ndarray, objectives: np.ndarray) -> bool:
        """Feed maximization *objectives* (negated to losses) and restart if stopped."""
        objs = np.asarray(objectives, dtype=np.float64)
        self._es.tell(z, -objs)
        self.steps += 1

        step_best = float(np.max(objs))
        if step_best >= 0.0 and step_best > self._cycle_best + 1e-12:
            self._cycle_best = step_best
            best_idx = int(np.argmax(objs))
            self._cycle_best_z = np.copy(z[best_idx])
            self._stagnant_steps = 0
        else:
            self._stagnant_steps += 1

        stagnated = (
            self._cfg.stagnation_steps > 0
            and self._stagnant_steps >= self._cfg.stagnation_steps
        )
        if self._es.stop() or stagnated:
            self.restarts += 1
            self.steps_since_restart = 0

            if (
                self._cfg.soft_restart
                and self._cycle_best >= 0.5
                and self._cycle_best_z is not None
            ):
                restart_x0 = np.copy(self._cycle_best_z)
                restart_sigma = self._cfg.sigma_restart
            else:
                restart_x0 = None
                restart_sigma = None

            self._cycle_best = -np.inf
            self._cycle_best_z = None
            self._stagnant_steps = 0
            next_pop = int(round(self._current_popsize * self._cfg.ipop_factor))
            if self._cfg.max_popsize > 0:
                next_pop = min(next_pop, self._cfg.max_popsize)
            self._current_popsize = max(self._cfg.popsize, next_pop)
            self._es = self._new_es(x0=restart_x0, sigma=restart_sigma)
            return True
        self.steps_since_restart += 1
        return False

    def state(self) -> WalkerState:
        """Snapshot current walker state."""
        return WalkerState(
            walker_id=self.walker_id,
            popsize=self.popsize,
            steps=self.steps,
            mean_norm=self.mean_norm,
            sigma=self.sigma,
            axis_ratio=self.axis_ratio,
            restarts=self.restarts,
            steps_since_restart=self.steps_since_restart,
        )


class MultiCMA:
    """Runs *K* independent :class:`CMAWalker` populations in a single batch."""

    def __init__(self, dim: int, cfg: CMAConfig, seed: int) -> None:
        streams = np.random.SeedSequence(seed).spawn(2 * cfg.n_populations)
        self.walkers = [
            CMAWalker(i, dim, cfg, streams[2 * i], streams[2 * i + 1])
            for i in range(cfg.n_populations)
        ]
        self.batch_size = sum(w.popsize for w in self.walkers)
        self._slices: list[tuple[int, int]] = []

    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """Ask all walkers and concatenate into ``(batch_size, dim)``."""
        xs: list[np.ndarray] = []
        wids: list[np.ndarray] = []
        slices: list[tuple[int, int]] = []
        pos = 0
        for w in self.walkers:
            x = w.ask()
            n = len(x)
            xs.append(x)
            wids.append(np.full(n, w.walker_id, dtype=np.int64))
            slices.append((pos, pos + n))
            pos += n
        self._slices = slices
        self.batch_size = pos
        return np.concatenate(xs, axis=0), np.concatenate(wids, axis=0)

    def resample_batch(
        self, z_batch: np.ndarray, duplicate_indices: list[int]
    ) -> np.ndarray:
        """Resample candidate vectors at *duplicate_indices* using respective walkers."""
        if not duplicate_indices or not self._slices:
            return z_batch
        idx_set = set(duplicate_indices)
        for w, (s, e) in zip(self.walkers, self._slices):
            w_dups = [i - s for i in range(s, e) if i in idx_set]
            if w_dups:
                w.resample_indices(z_batch[s:e], w_dups)
        return z_batch

    def tell(self, z: np.ndarray, objectives: np.ndarray) -> None:
        """Slice batch objectives back to each walker."""
        if len(self._slices) != len(self.walkers):
            raise RuntimeError("tell() called before ask()")
        for w, (s, e) in zip(self.walkers, self._slices):
            w.tell(z[s:e], objectives[s:e])

    def stats(self) -> SolverStats:
        """Aggregate walker spread, mean norms, sigmas, axis ratios, and restarts."""
        means = np.stack([w.mean for w in self.walkers], axis=0)
        mean_norms = np.linalg.norm(means, axis=1)
        sigmas = [w.sigma for w in self.walkers]
        ratios = [w.axis_ratio for w in self.walkers]
        spread = float(np.mean(np.std(means, axis=0))) if len(self.walkers) > 1 else 0.0
        return SolverStats(
            walker_spread=spread,
            mean_norm_mean=float(np.mean(mean_norms)),
            sigma_mean=float(np.mean(sigmas)),
            sigma_min=float(np.min(sigmas)),
            axis_ratio_max=float(np.max(ratios)),
            restarts=sum(w.restarts for w in self.walkers),
        )

    def walker_states(self) -> list[WalkerState]:
        """Return per-walker state snapshots in ``walker_id`` order."""
        return [w.state() for w in self.walkers]


class NaiveSolver:
    """Unconditional ``N(0, I)`` random-prior baseline."""

    def __init__(self, dim: int, batch_size: int, seed: int) -> None:
        self.dim = dim
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)

    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``(batch_size, dim)`` from ``N(0, I)``."""
        z = self._rng.standard_normal((self.batch_size, self.dim))
        return z, np.full(self.batch_size, -1, dtype=np.int64)

    def tell(self, z: np.ndarray, objectives: np.ndarray) -> None:
        """No-op for stateless sampling."""

    def stats(self) -> SolverStats:
        """Return zeroed diagnostics."""
        return SolverStats()

    def walker_states(self) -> list[WalkerState]:
        """Stateless baseline has no CMA walkers."""
        return []


def build_solver(
    name: str, dim: int, cfg: CMAConfig, seed: int
) -> MultiCMA | NaiveSolver:
    """Construct ``MultiCMA`` (``'pycma'``) or ``NaiveSolver`` (``'naive'``)."""
    if name == "pycma":
        return MultiCMA(dim=dim, cfg=cfg, seed=seed)
    if name == "naive":
        return NaiveSolver(dim=dim, batch_size=cfg.batch_size, seed=seed)
    raise ValueError(f"Unknown solver {name!r}; expected 'pycma' or 'naive'")
