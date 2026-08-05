"""Reporting infrastructure: I/O, persistence, and visualization."""

from __future__ import annotations

# Lazy imports to avoid circular dependencies
__all__ = [
    "console",
    "detail",
    "load_archive",
    "load_scheduler",
    "loaded",
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


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in {"console", "detail", "loaded", "make_progress_bar", "make_table", "saved", "section", "skipped", "step"}:
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
        return locals()[name]
    elif name in {"load_archive", "load_scheduler", "save_archive", "save_scheduler"}:
        from reporting.persistence import (
            load_archive,
            load_scheduler,
            save_archive,
            save_scheduler,
        )
        return locals()[name]
    elif name in {"print_generation", "print_results"}:
        from reporting.reporting import print_generation, print_results
        return locals()[name]
    elif name == "visualize_archive":
        from reporting.visualization import visualize_archive
        return visualize_archive
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
