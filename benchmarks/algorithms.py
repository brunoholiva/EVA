"""Algorithm registry for the QD sweep: one spec dict per algorithm.

Mirrors pyribs' own ``sphere.py`` benchmark — a ``CONFIG`` of algorithm specs
plus a ``create_scheduler`` that lays every spec over the same
:class:`~benchmarks.domains.Domain`. Nothing about the behaviour map is
hard-coded here; the domain injects solution dimension, descriptor ranges, grid
resolution and the latent bound.

A spec holds only what distinguishes algorithms:

- ``archive`` / ``result_archive``: archive class plus its own hyperparameters
  (learning rate, threshold, buffer size, neighbour count).
- ``emitters``: one or more groups of ``(class, kwargs, num_emitters)``.
- ``scheduler``: ``Scheduler`` or ``BanditScheduler``.
- ``pass_result_archive_to_emitters``: density models store no solutions, so
  their emitters must rank against the result archive.

Unavailable in ribs 0.8.3 (see tests/test_benchmark_algorithms.py): the five
gradient-based DQD algorithms (TabPFN is not differentiable), ``dms``
(``DiscountArchive`` is absent), ``nslc`` (no ``nslc`` ranker), ``dds_cnf``
(``DensityArchive`` here supports only ``kde`` and ``kde_sklearn``) and
``dds_kde_sklearn`` — a second DDS variant whose sklearn Gaussian kernel
(``KernelDensity.score_samples``) is log density, i.e. the same ranking as
``kde`` under the ``density`` ranker, so it ran bit-identical to ``dds_kde``.
"""

from __future__ import annotations

import numpy as np
from ribs.archives import CVTArchive, DensityArchive, GridArchive, ProximityArchive
from ribs.emitters import EvolutionStrategyEmitter, GaussianEmitter, IsoLineEmitter
from ribs.schedulers import BanditScheduler, Scheduler

from benchmarks.domains import Domain


class _SafeES(EvolutionStrategyEmitter):
    """EvolutionStrategyEmitter that climbs objective while the archive is empty.

    When ``threshold_min = 0.6``, the initial steps around ``x0 = 0`` have
    ``P(active) < 0.6``, so the archive stays empty for ~3-6 steps until the
    first hit is discovered. During those warmup steps:
    - ``filter`` selection would otherwise set ``num_parents = 0`` (no update),
    - ``no_improvement`` would call ``sample_elites(1)`` on an empty archive
      and raise ``IndexError: No elements in archive``, and
    - ``2rd`` would rank by random measure directions instead of climbing toward
      the threshold.
    Falling back to objective-ranked ``mu`` selection without restarting while
    ``self.archive.empty`` is True lets every ES variant climb onto the ``>0.6``
    manifold; once the first elite enters the archive, standard pyribs behaviour
    runs unchanged.
    """

    def tell(self, solution, objective, measures, add_info, **fields):
        if self.archive.empty:
            obj = np.asarray(objective)
            self._itrs += 1
            indices = np.flip(np.argsort(obj))
            self._opt.tell(indices, obj, self._batch_size // 2)
            if self._opt.check_stop(obj[indices]):
                self._opt.reset(self.x0)
                self._ranker.reset(self, self.archive)
                self._restarts += 1
            return
        super().tell(solution, objective, measures, add_info, **fields)


class _SafeIsoLine(IsoLineEmitter):
    """IsoLineEmitter that only signals a restart after an active non-empty tell."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._restarts = 0

    @property
    def restarts(self) -> int:
        return self._restarts

    def tell(self, solution, objective, measures, add_info, **fields):
        if not self.archive.empty:
            self._restarts += 1


class _SafeBanditScheduler(BanditScheduler):
    """BanditScheduler with robust UCB1 warmup and all-emitter reselection.

    Fixes two upstream pyribs 0.8.3 bugs when ``reselect='all'``:
    1. Reselecting 'all' when ``sum(success) == 0`` computes ``log(0) = -inf`` and
       ``sqrt(-inf) = nan``. In numpy, ``argsort`` places ``nan`` at the end, so
       reversing the order puts ``nan`` at the front, starving untried emitters.
       During the zero-success warmup, we select least-evaluated emitters to ensure
       balanced exploration across the entire pool before active hits appear.
    2. Once hits appear, UCB1 balances empirical success rate (archive cell
       discoveries/improvements) against the exploration bonus zeta.
    """

    def ask(self):
        if self._last_called == "ask":
            raise RuntimeError(
                "ask cannot be called immediately after " + self._last_called
            )
        self._last_called = "ask"

        if self._reselect == "terminated":
            emitter_restarts = np.array(
                [
                    emitter.restarts if hasattr(emitter, "restarts") else -1
                    for emitter in self._emitter_pool
                ]
            )
            reselect = emitter_restarts > self._restarts
            reselect[emitter_restarts < 0] = True
            self._restarts = emitter_restarts
        else:
            reselect = self._active_arr.copy()

        num_needed = self._num_active - self._active_arr.sum()
        i = 0
        while num_needed > 0:
            reselect[i] = False
            if not self._active_arr[i]:
                self._active_arr[i] = True
                num_needed -= 1
            i += 1

        self._active_arr[reselect] = False

        if reselect.any():
            ucb1 = np.full_like(self._emitter_pool, np.inf, dtype=float)
            update_ucb = self._selection != 0
            if update_ucb.any():
                sum_succ = self._success.sum()
                if sum_succ > 0:
                    expl = self._zeta * np.sqrt(
                        np.log(max(sum_succ, 1.0)) / self._selection[update_ucb]
                    )
                    ucb1[update_ucb] = (
                        self._success[update_ucb] / self._selection[update_ucb]
                    ) + expl
                else:
                    ucb1[update_ucb] = -self._selection[update_ucb]

            activate = np.argsort(ucb1)[::-1]
            cur_active = self._active_arr.sum()
            for idx in activate:
                if cur_active >= self._num_active:
                    break
                if not self._active_arr[idx]:
                    self._active_arr[idx] = True
                    cur_active += 1

        self._cur_solutions = []
        for idx in np.where(self._active_arr)[0]:
            emitter = self._emitter_pool[idx]
            sols = emitter.ask()
            self._cur_solutions.append(sols)
            self._num_emitted[idx] = len(sols)

        self._cur_solutions = (
            np.concatenate(self._cur_solutions, axis=0)
            if self._cur_solutions
            else np.empty((0, self._solution_dim))
        )
        return self._cur_solutions


class _AntiCollapseES(_SafeES):
    """CMA-ES emitter that enforces a floor on step size sigma to prevent collapse."""

    def __init__(self, *args, min_sigma: float = 0.30, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_sigma = min_sigma

    def tell(self, solution, objective, measures, add_info, **fields):
        super().tell(solution, objective, measures, add_info, **fields)
        if hasattr(self, "_opt") and hasattr(self._opt, "sigma"):
            if self._opt.sigma < self.min_sigma:
                self._opt.sigma = self.min_sigma


class _ClearingES(_SafeES):
    """CMA-ES emitter with repulsive basin clearing to prevent chemotype over-exploitation.

    Tracks active solutions (P >= 0.6). When an emitter harvests hit_cap hits in the current
    local basin, it registers the basin centroid in the shared archive peak registry and
    executes a repulsive restart into an unvisited region of the 95D latent space.
    """

    def __init__(self, *args, hit_cap: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.hit_cap = hit_cap
        self.current_basin_hits = 0
        self.recent_active_sols: list[np.ndarray] = []
        peaks = getattr(self.archive, "_archived_peaks", None)
        if peaks is None:
            self.archive._archived_peaks = []

    @property
    def archived_peaks(self) -> list[np.ndarray]:
        return self.archive._archived_peaks

    def _repulsive_restart(self) -> None:
        dim = self.solution_dim
        low = np.where(np.isneginf(self.lower_bounds), -3.0, self.lower_bounds)
        high = np.where(np.isposinf(self.upper_bounds), 3.0, self.upper_bounds)
        cands = np.clip(np.random.normal(0, 1.0, size=(100, dim)), low, high)
        if self.archived_peaks:
            peaks = np.array(self.archived_peaks)
            dists = np.min(
                np.linalg.norm(cands[:, None, :] - peaks[None, :, :], axis=2), axis=1
            )
            best_cand = cands[np.argmax(dists)]
        else:
            best_cand = cands[0]

        self._opt.reset(best_cand)
        self._ranker.reset(self, self.archive)
        self._restarts += 1
        self.current_basin_hits = 0
        self.recent_active_sols = []

    def tell(self, solution, objective, measures, add_info, **fields):
        obj = np.asarray(objective)
        sols = np.asarray(solution)
        thresh = 0.6 if np.max(obj) <= 1.0 else 80.0
        active_mask = obj >= thresh
        if active_mask.any():
            self.current_basin_hits += int(active_mask.sum())
            self.recent_active_sols.extend(sols[active_mask])
            if len(self.recent_active_sols) > 200:
                self.recent_active_sols = self.recent_active_sols[-200:]

        super().tell(solution, objective, measures, add_info, **fields)

        should_eject = self.current_basin_hits >= self.hit_cap
        if should_eject:
            if self.recent_active_sols:
                centroid = np.mean(self.recent_active_sols, axis=0)
                self.archived_peaks.append(centroid)
            self._repulsive_restart()


ES = _SafeES

THRESHOLD_MIN = 0.6
"""Minimum P(active) threshold for ``*_thresh`` archive variants."""

_THRESH_ARCHIVE = {
    "class": GridArchive,
    "kwargs": {"threshold_min": THRESHOLD_MIN, "learning_rate": 1.0},
}
_RESULT_ARCHIVE = {"class": GridArchive, "kwargs": {}}

FEASIBLE_BOUNDS_RATIO = 5.12 / 1.5
"""Reference box-halfwidth / sigma0 ratio (sphere: ``+-5.12`` with ``sigma0 1.5``).

pyribs' bounds handling resamples out-of-bounds proposals until *every*
coordinate fits, and that loop never ends once the box is only ~2 sigma wide —
DDS's ``sigma0 = 1.5`` under EVA's ``+-3`` latent bound stalls the first
``ask()`` in 95-D. ``sigma0`` is therefore capped at
``box_halfwidth / FEASIBLE_BOUNDS_RATIO``, which leaves every other spec (0.5)
untouched and reproduces the reference configuration's feasibility.
"""

CONFIG: dict[str, dict] = {
    "map_elites": {
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": GaussianEmitter,
                "kwargs": {"sigma": 0.5, "batch_size": 540},
                "num_emitters": 1,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "line_map_elites": {
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": IsoLineEmitter,
                "kwargs": {"iso_sigma": 0.5, "line_sigma": 0.2, "batch_size": 540},
                "num_emitters": 1,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cvt_map_elites": {
        "archive": {"class": CVTArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": GaussianEmitter,
                "kwargs": {"sigma": 0.5, "batch_size": 540},
                "num_emitters": 1,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "line_cvt_map_elites": {
        "archive": {"class": CVTArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": IsoLineEmitter,
                "kwargs": {"iso_sigma": 0.5, "line_sigma": 0.2, "batch_size": 540},
                "num_emitters": 1,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "me_map_elites": {
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": ES,
                "kwargs": {"sigma0": 0.5, "ranker": "obj", "batch_size": 36},
                "num_emitters": 15,
            },
            {
                "class": ES,
                "kwargs": {"sigma0": 0.5, "ranker": "2rd", "batch_size": 36},
                "num_emitters": 15,
            },
            {
                "class": ES,
                "kwargs": {"sigma0": 0.5, "ranker": "2imp", "batch_size": 36},
                "num_emitters": 15,
            },
            {
                "class": _SafeIsoLine,
                "kwargs": {"iso_sigma": 0.01, "line_sigma": 0.1, "batch_size": 36},
                "num_emitters": 15,
            },
        ],
        "scheduler": {
            "class": BanditScheduler,
            "kwargs": {"num_active": 15, "reselect": "terminated"},
        },
    },
    "cma_me_imp": {
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "selection_rule": "filter",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_me_imp_mu": {
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "selection_rule": "mu",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_me_basic": {
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_me_rd": {
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2rd",
                    "selection_rule": "filter",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_me_rd_mu": {
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2rd",
                    "selection_rule": "mu",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_me_opt": {
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "obj",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_opt_clearing": {
        # Chemotype-Clearing Restarts (Option 1): Once an emitter harvests 100 hits
        # in a local basin, the basin centroid is archived and the emitter executes
        # a repulsive restart into an unvisited region of the 95D latent space.
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": _ClearingES,
                "kwargs": {
                    "sigma0": 0.5,
                    "hit_cap": 100,
                    "ranker": "obj",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_opt_anticollapse": {
        # Anti-Collapse CMA-ES (Option 2): Enforces a floor on step size (sigma >= 0.30)
        # so the covariance ellipsoid cannot collapse into a single point.
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": _AntiCollapseES,
                "kwargs": {
                    "sigma0": 0.5,
                    "min_sigma": 0.30,
                    "ranker": "obj",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_me_obj_bandit": {
        # The map decides *where* to look, the objective decides *what is good*:
        # 60 CMA emitters in the pool, 15 active per step, rotated by UCB1 over
        # "how many of my solutions did the archive accept". Every emitter ranks
        # its candidates by raw objective, so the cell threshold never enters
        # the CMA's fitness (the failure mode of the imp/2imp arms).
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": {"class": GridArchive, "kwargs": {}},
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "obj",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 60,
            }
        ],
        "scheduler": {
            "class": _SafeBanditScheduler,
            "kwargs": {"num_active": 15, "reselect": "all", "zeta": 0.1},
        },
    },
    "cma_me_mixed": {
        "archive": {"class": GridArchive, "kwargs": {}},
        "result_archive": None,
        "emitters": [
            {
                "class": ES,
                "kwargs": {"sigma0": 0.5, "ranker": "2rd", "batch_size": 36},
                "num_emitters": 7,
            },
            {
                "class": ES,
                "kwargs": {"sigma0": 0.5, "ranker": "2imp", "batch_size": 36},
                "num_emitters": 8,
            },
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_mae": {
        "archive": {
            "class": GridArchive,
            "kwargs": {"threshold_min": 0, "learning_rate": 0.01},
        },
        "result_archive": _RESULT_ARCHIVE,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "imp",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_mae_thresh": {
        "archive": {
            "class": GridArchive,
            "kwargs": {"threshold_min": THRESHOLD_MIN, "learning_rate": 0.01},
        },
        "result_archive": _RESULT_ARCHIVE,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "imp",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "me_map_elites_thresh": {
        "archive": _THRESH_ARCHIVE,
        "result_archive": _RESULT_ARCHIVE,
        "emitters": [
            {
                "class": ES,
                "kwargs": {"sigma0": 0.5, "ranker": "obj", "batch_size": 36},
                "num_emitters": 15,
            },
            {
                "class": ES,
                "kwargs": {"sigma0": 0.5, "ranker": "2rd", "batch_size": 36},
                "num_emitters": 15,
            },
            {
                "class": ES,
                "kwargs": {"sigma0": 0.5, "ranker": "2imp", "batch_size": 36},
                "num_emitters": 15,
            },
            {
                "class": _SafeIsoLine,
                "kwargs": {"iso_sigma": 0.01, "line_sigma": 0.1, "batch_size": 36},
                "num_emitters": 15,
            },
        ],
        "scheduler": {
            "class": BanditScheduler,
            "kwargs": {"num_active": 15, "reselect": "terminated"},
        },
    },
    "cma_me_imp_thresh": {
        "archive": _THRESH_ARCHIVE,
        "result_archive": _RESULT_ARCHIVE,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "selection_rule": "filter",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_me_imp_mu_thresh": {
        "archive": _THRESH_ARCHIVE,
        "result_archive": _RESULT_ARCHIVE,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "selection_rule": "mu",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_me_basic_thresh": {
        "archive": _THRESH_ARCHIVE,
        "result_archive": _RESULT_ARCHIVE,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_me_rd_thresh": {
        "archive": _THRESH_ARCHIVE,
        "result_archive": _RESULT_ARCHIVE,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2rd",
                    "selection_rule": "filter",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_me_rd_mu_thresh": {
        "archive": _THRESH_ARCHIVE,
        "result_archive": _RESULT_ARCHIVE,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2rd",
                    "selection_rule": "mu",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_me_opt_thresh": {
        "archive": _THRESH_ARCHIVE,
        "result_archive": _RESULT_ARCHIVE,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "obj",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "cma_me_obj_bandit_thresh": {
        "archive": _THRESH_ARCHIVE,
        "result_archive": _RESULT_ARCHIVE,
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "obj",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 60,
            }
        ],
        "scheduler": {
            "class": _SafeBanditScheduler,
            "kwargs": {"num_active": 15, "reselect": "all", "zeta": 0.1},
        },
    },
    "cma_me_mixed_thresh": {
        "archive": _THRESH_ARCHIVE,
        "result_archive": _RESULT_ARCHIVE,
        "emitters": [
            {
                "class": ES,
                "kwargs": {"sigma0": 0.5, "ranker": "2rd", "batch_size": 36},
                "num_emitters": 7,
            },
            {
                "class": ES,
                "kwargs": {"sigma0": 0.5, "ranker": "2imp", "batch_size": 36},
                "num_emitters": 8,
            },
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "ns_cma": {
        "archive": {"class": ProximityArchive, "kwargs": {"k_neighbors": 15}},
        "result_archive": {"class": GridArchive, "kwargs": {}},
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "nov",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "nslc_cma_imp": {
        "archive": {
            "class": ProximityArchive,
            "kwargs": {"k_neighbors": 15, "local_competition": True},
        },
        "result_archive": {"class": GridArchive, "kwargs": {}},
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "selection_rule": "filter",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
    "dds_kde": {
        "pass_result_archive_to_emitters": True,
        "archive": {
            "class": DensityArchive,
            "kwargs": {"buffer_size": 10000, "density_method": "kde"},
        },
        "result_archive": {"class": GridArchive, "kwargs": {}},
        "emitters": [
            {
                "class": ES,
                "kwargs": {
                    "sigma0": 1.5,
                    "ranker": "density",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {"class": Scheduler, "kwargs": {}},
    },
}

ALGORITHMS: list[str] = sorted(CONFIG)


def _build_archive(entry: dict | None, domain: Domain, seed: int | None):
    """Instantiate an archive entry, injecting the domain's geometry."""
    if entry is None:
        return None

    cls = entry["class"]
    kwargs = dict(entry["kwargs"])
    common = {"seed": seed}

    if cls is CVTArchive:
        return cls(
            solution_dim=domain.solution_dim,
            cells=domain.n_cells,
            ranges=domain.measure_ranges,
            **common,
            **kwargs,
        )
    if cls is ProximityArchive:
        kwargs.setdefault("novelty_threshold", domain.novelty_threshold)
        return cls(
            solution_dim=domain.solution_dim,
            measure_dim=domain.measure_dim,
            **common,
            **kwargs,
        )
    if cls is DensityArchive:
        kwargs.setdefault("bandwidth", domain.kde_bandwidth)
        return cls(measure_dim=domain.measure_dim, **common, **kwargs)
    return cls(
        solution_dim=domain.solution_dim,
        dims=domain.archive_dims,
        ranges=domain.measure_ranges,
        **common,
        **kwargs,
    )


def create_scheduler(spec: dict, domain: Domain, seed: int | None = None) -> Scheduler:
    """Build a scheduler for *spec* laid over *domain*.

    Parameters
    ----------
    spec : dict
        One entry of :data:`CONFIG`.
    domain : Domain
        Supplies solution dimension, descriptor ranges, grid resolution, and
        the latent bound emitters may propose in.
    seed : int or None, default=None
        Seed for the archive and for spawning per-emitter seeds.

    Returns
    -------
    ribs.schedulers.Scheduler
    """
    result_archive = _build_archive(spec["result_archive"], domain, seed)
    archive = _build_archive(spec["archive"], domain, seed)

    # Density models buffer no solutions, so emitters must rank against the
    # result archive instead (pyribs sphere.py does the same).
    archive_for_emitter = (
        result_archive if spec.get("pass_result_archive_to_emitters") else archive
    )

    emitter_kwargs: dict = {"x0": np.zeros(domain.solution_dim)}
    max_sigma0 = None
    if domain.emitter_bounds is not None:
        low, high = domain.emitter_bounds
        emitter_kwargs["bounds"] = [(low, high)] * domain.solution_dim
        max_sigma0 = (high - low) / 2.0 / FEASIBLE_BOUNDS_RATIO

    emitters = []
    # One spawn sequence for the whole spec: per-group spawns restart the
    # child keys at 0, so every group's emitter i would draw the same seed —
    # cma_me_mixed's 7+8 split then proposes 7 identical 36-point blocks.
    seeds = iter(
        np.random.SeedSequence(seed).spawn(
            sum(group["num_emitters"] for group in spec["emitters"])
        )
    )
    for group in spec["emitters"]:
        group_kwargs = dict(group["kwargs"])
        if max_sigma0 is not None and "sigma0" in group_kwargs:
            group_kwargs["sigma0"] = min(group_kwargs["sigma0"], max_sigma0)
        emitters += [
            group["class"](
                archive_for_emitter, **emitter_kwargs, **group_kwargs, seed=next(seeds)
            )
            for _ in range(group["num_emitters"])
        ]

    scheduler_cls = spec["scheduler"]["class"]
    return scheduler_cls(
        archive, emitters, result_archive, **spec["scheduler"]["kwargs"]
    )


def scheduler_batch_size(scheduler: Scheduler) -> int:
    """Return the number of candidates one :meth:`~ribs.schedulers.Scheduler.ask` returns.

    ``BanditScheduler`` keeps a larger emitter pool than it activates, so only
    the active members contribute. Its active mask is all-False until the first
    ``ask()`` (ribs 0.8.3 activates the first ``num_active`` pool members
    there), so before that the estimate uses those same first members — pyribs
    exposes no public accessor for ``num_active``.
    """
    pool = getattr(scheduler, "emitter_pool", None)
    if pool is None:
        return int(sum(e.batch_size for e in scheduler.emitters))
    active = np.asarray(scheduler.active, dtype=bool)
    if not active.any():
        n_active = int(getattr(scheduler, "_num_active", len(pool)))
        active = np.arange(len(pool)) < n_active
    return int(sum(e.batch_size for e, on in zip(pool, active) if on))
