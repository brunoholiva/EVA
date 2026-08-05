"""CMA-MAE optimization loop via pyribs."""

from __future__ import annotations

from optimization.constants import INVALID_MOLECULE_OBJECTIVE
from optimization.evaluator import EvalResult, Evaluator
from optimization.loop import CMAMAELoop, build_scheduler
from optimization.tensorboard import TensorBoardLogger
from reporting.persistence import (
    load_archive,
    load_scheduler,
    save_archive,
    save_scheduler,
)
from reporting.plotting import create_archive_figure, create_parallel_axes_figure
from reporting.reporting import print_generation, print_results
from reporting.visualization import visualize_archive

__all__ = [
    "INVALID_MOLECULE_OBJECTIVE",
    "CMAMAELoop",
    "EvalResult",
    "Evaluator",
    "TensorBoardLogger",
    "build_scheduler",
    "create_archive_figure",
    "create_parallel_axes_figure",
    "load_archive",
    "load_scheduler",
    "print_generation",
    "print_results",
    "save_archive",
    "save_scheduler",
    "visualize_archive",
]
