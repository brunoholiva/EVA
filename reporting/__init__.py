"""Reporting infrastructure: I/O, persistence, and visualization."""

from __future__ import annotations

from reporting.console import (
    console,
    detail,
    loaded,
    make_progress_bar,
    make_table,
    saved,
    section,
    skipped,
    step,
)
from reporting.persistence import (
    load_archive,
    load_scheduler,
    save_archive,
    save_scheduler,
)
from reporting.reporting import print_generation, print_results
from reporting.visualization import visualize_archive

__all__ = [
    "console",
    "detail",
    "loaded",
    "load_archive",
    "load_scheduler",
    "make_progress_bar",
    "make_table",
    "print_generation",
    "print_results",
    "save_archive",
    "save_scheduler",
    "saved",
    "section",
    "skipped",
    "step",
    "visualize_archive",
]
