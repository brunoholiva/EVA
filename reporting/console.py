"""Shared console helpers for consistent terminal output.

Every module that writes to the terminal imports ``console`` and the
helper functions from here so wording and formatting stay uniform:

* Steps are present-tense, declarative sentences naming a specific object
  (``"Featurizing molecules for predictor inference"``) - no trailing
  ellipses and no emoji.
* Details are indented two spaces under the step that produced them.
* Saved and loaded artifacts use plain words (``"Saved archive to path"``).
* Section headers are rendered as ``rich`` rules.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


def section(title: str) -> None:
    """Print a full-width rule with *title* to delimit a phase.

    Parameters
    ----------
    title : str
        Heading text for the rule.
    """
    console.rule(f"[bold]{title}")


def step(message: str) -> None:
    """Print a step message (declarative, no trailing ellipsis).

    Parameters
    ----------
    message : str
        Step description, e.g. ``"Loading ChemBed VAE"``.
    """
    console.print(message)


def detail(message: str) -> None:
    """Print an indented detail line under a step.

    Parameters
    ----------
    message : str
        Detail description, e.g. ``"Working space: 32 dim (was 256)"``.
    """
    console.print(f"  {message}")


def saved(artifact: str, path: str | Path, note: str = "") -> None:
    """Report that *artifact* was written to *path*.

    Parameters
    ----------
    artifact : str
        Name of the written artifact, e.g. ``"archive"``.
    path : str or Path
        Where the artifact was written.
    note : str, default=""
        Optional parenthetical detail, e.g. a file size.
    """
    text = f"Saved {artifact} to {path}"
    if note:
        text += f" ({note})"
    console.print(text)


def loaded(artifact: str, path: str | Path, note: str = "") -> None:
    """Report that *artifact* was read from *path*.

    Parameters
    ----------
    artifact : str
        Name of the loaded artifact, e.g. ``"scheduler"``.
    path : str or Path
        Where the artifact was read from.
    note : str, default=""
        Optional parenthetical detail, e.g. a cell count.
    """
    text = f"Loaded {artifact} from {path}"
    if note:
        text += f" ({note})"
    console.print(text)


def skipped(path: str | Path, reason: str) -> None:
    """Report that no file was written to *path*, with *reason*.

    Parameters
    ----------
    path : str or Path
        Path where the file would have been written.
    reason : str
        Why nothing was written, e.g. ``"elites to visualize"``.
    """
    console.print(f"Skipped {path}: no {reason}")


def make_progress_bar(description: str, total: int) -> tuple[Progress, int]:
    """Create a styled progress bar with consistent formatting.

    Parameters
    ----------
    description : str
        Task description shown in the progress bar.
    total : int
        Total number of steps.

    Returns
    -------
    tuple[Progress, int]
        A configured Progress instance and the task ID.
    """
    columns = [
        TextColumn("  "),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="white", complete_style="magenta"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ]
    progress = Progress(*columns, console=console)
    task_id = progress.add_task(description, total=total)
    return progress, task_id


def make_eva_progress(
    n_generations: int,
    batch_size: int,
) -> tuple[Progress, int, int, int]:
    """Create the 3-bar progress layout for EVA.

    Creates 3 persistent progress bars ordered fastest to slowest:
    1. Solutions (fastest) - tracks solutions evaluated within each generation
    2. Featurization (medium) - tracks molecules featurized for predictor
    3. CMA-MAE (slowest) - tracks generations

    Parameters
    ----------
    n_generations : int
        Total number of generations to run.
    batch_size : int
        Total number of solutions per generation (batch_size * n_emitters).

    Returns
    -------
    tuple[Progress, int, int, int]
        Progress instance and task IDs for (solutions, featurization, main).
    """
    columns = [
        TextColumn("  "),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="white", complete_style="green"),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("•"),
        TextColumn("{task.fields[status]}"),
        TimeRemainingColumn(),
    ]

    progress = Progress(*columns, console=console)

    solutions_task = progress.add_task(
        "Solutions",
        total=batch_size,
        completed=0,
        status="0 valid",
    )

    featurization_task = progress.add_task(
        "Featurization",
        total=100,
        completed=0,
        status="0 molecules",
    )

    main_task = progress.add_task(
        "CMA-MAE",
        total=n_generations,
        completed=0,
        status="0 cells",
    )

    return progress, solutions_task, featurization_task, main_task


def make_table(title: str, columns: list[tuple[str, str]]) -> Table:
    """Create a styled table with consistent formatting.

    Parameters
    ----------
    title : str
        Table title.
    columns : list of (name, style) tuples
        Column names and their styles.

    Returns
    -------
    Table
        A configured Table instance.
    """
    table = Table(title=title, show_header=True)
    for name, style in columns:
        table.add_column(name, style=style)
    return table
