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

# Candidate pipeline and HTML report are imported lazily (heavy deps).
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


# Lazy re-exports for candidates.html_report (avoids import-time overhead)
def __getattr__(name: str):
    if name == "run_pipeline":
        from reporting.candidates import run_pipeline

        return run_pipeline
    if name == "run_full_report":
        from reporting.candidates import run_full_report

        return run_full_report
    if name == "generate_html_report":
        from reporting.html_report import generate_html_report

        return generate_html_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
