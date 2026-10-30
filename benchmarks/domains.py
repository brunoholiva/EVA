"""Benchmark domains: the geometry every algorithm is compared over.

A domain is the *space*, not the oracle — it supplies solution dimension, the
latent box emitters may propose in, the descriptor ranges and grid resolution
the archive is laid over, plus the two density/novelty scales that spheres.py
hard-codes. The oracle stays with the caller: molecules use
:class:`~evaluation.evaluator.Evaluator`, the validation test uses sphere.

Scaling rule: both ``novelty_threshold`` and ``kde_bandwidth`` are fractions of
the mean measure-box width (0.037 and 0.05 respectively), which reproduces the
paper's sphere values exactly (512 wide -> 18.94 and 25.6) while giving a sane
scale in a descriptor box less than 1 unit wide.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from config import ExperimentConfig

ARCHIVE_RESOLUTION = 40
"""Cells per behaviour dimension for every algorithm in a sweep."""

_NOVELTY_FRACTION = 0.037
"""Novelty threshold as a fraction of mean measure-box width (paper's sphere)."""

_BANDWIDTH_FRACTION = 0.05
"""KDE bandwidth as a fraction of mean measure-box width (25.6 / 512)."""


@dataclass(frozen=True)
class Domain:
    """Geometry and derived scales shared by every algorithm in a sweep."""

    name: str
    solution_dim: int
    emitter_bounds: tuple[float, float] | None
    measure_ranges: tuple[tuple[float, float], ...]
    archive_dims: tuple[int, ...]
    novelty_threshold: float
    kde_bandwidth: float

    @property
    def n_cells(self) -> int:
        """Total cells in the grid archive (and CVT centroid count)."""
        return prod(self.archive_dims)

    @property
    def measure_dim(self) -> int:
        """Number of behaviour descriptors."""
        return len(self.measure_ranges)


def _box_scales(measure_ranges: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    """Return ``(novelty_threshold, kde_bandwidth)`` from the measure box."""
    widths = [high - low for low, high in measure_ranges]
    mean_width = sum(widths) / len(widths)
    return _NOVELTY_FRACTION * mean_width, _BANDWIDTH_FRACTION * mean_width


def sphere_domain() -> Domain:
    """Return the 100-D sphere benchmark's geometry (reference for validation).

    Two measures, each the sum of 50 clipped coordinates in ``[-256, 256]``,
    over a 100x100 grid — matching the setup in Fontaine 2020/2023.
    """
    measure_ranges = ((-256.0, 256.0), (-256.0, 256.0))
    novelty, bandwidth = _box_scales(measure_ranges)
    return Domain(
        name="sphere",
        solution_dim=100,
        emitter_bounds=None,
        measure_ranges=measure_ranges,
        archive_dims=(100, 100),
        novelty_threshold=novelty,
        kde_bandwidth=bandwidth,
    )


def molecule_domain(cfg: ExperimentConfig) -> Domain:
    """EVA's behaviour space, read from *cfg* with a fixed 40x40 grid.

    Must be called after :func:`EVA.load_vae`, which shrinks
    ``cfg.archive.solution_dim`` to the projected working space.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment config supplying descriptor ranges and the latent bound.

    Returns
    -------
    Domain
    """
    ranges = tuple(
        (float(d.range[0]), float(d.range[1])) for d in cfg.archive.dimensions
    )
    novelty, bandwidth = _box_scales(ranges)
    bound = float(cfg.emitter.bounds)
    return Domain(
        name="molecules",
        solution_dim=int(cfg.archive.solution_dim),
        emitter_bounds=(-bound, bound),
        measure_ranges=ranges,
        archive_dims=(ARCHIVE_RESOLUTION,) * len(ranges),
        novelty_threshold=novelty,
        kde_bandwidth=bandwidth,
    )
