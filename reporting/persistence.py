"""State persistence: save and load scheduler and archive."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler

from reporting.console import loaded, saved
from reporting.reporting import _rank_archive


def save_archive(
    archive: GridArchive,
    decode_fn,
    output_dir: str | Path,
    suffix: str = "archive",
    dimension_names: list[str] | None = None,
    real_objectives: dict[int, float] | None = None,
) -> Path:
    """Save the archive as joblib (full state) and CSV (tabular data).

    Parameters
    ----------
    archive : GridArchive
        The archive to persist.
    decode_fn : callable
        Latent vectors → SMILES decoder.
    output_dir : str or Path
        Directory to write ``<suffix>.joblib`` and ``<suffix>.csv``.
    suffix : str
        Base filename for the saved files (default ``"archive"``).
    dimension_names : list of str or None
        Names of enabled archive dimensions.
    real_objectives : dict or None
        Real P(active) values (uncapped) keyed by solution index.

    Returns
    -------
    Path
        Directory where files were written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = output_dir / f"{suffix}.joblib"
    joblib.dump(archive, joblib_path)
    saved("archive", joblib_path)

    csv_path = output_dir / f"{suffix}.csv"
    _export_archive_csv(archive, decode_fn, csv_path, dimension_names, real_objectives)
    saved("CSV", csv_path)

    return output_dir


def _export_archive_csv(
    archive: GridArchive,
    decode_fn,
    csv_path: Path,
    dimension_names: list[str] | None = None,
    real_objectives: dict[int, float] | None = None,
) -> None:
    """Write archive contents to a CSV file."""
    if dimension_names is None:
        dimension_names = [f"dim_{i}" for i in range(archive.measure_dim)]

    if len(archive) == 0:
        columns = ["rank", "smiles", "p_active", "p_active_real"] + dimension_names
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
        return

    order, _, smiles = _rank_archive(archive, decode_fn, len(archive))
    arch_data = archive.data()

    data = {
        "rank": np.arange(1, len(order) + 1),
        "smiles": smiles,
        "p_active": arch_data["objective"][order],
    }

    # Add real P(active) if available
    if real_objectives is not None:
        cell_indices = arch_data["index"][order]
        data["p_active_real"] = [
            real_objectives.get(idx, np.nan) for idx in cell_indices
        ]
    else:
        data["p_active_real"] = [np.nan] * len(order)

    for d, name in enumerate(dimension_names):
        data[name] = arch_data["measures"][order, d]

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)


def save_scheduler(scheduler: Scheduler, output_dir: str | Path) -> Path:
    """Save full scheduler state (archive + emitter + CMA-ES) as joblib.

    Parameters
    ----------
    scheduler : Scheduler
        The scheduler to persist.
    output_dir : str or Path
        Directory to write ``scheduler.joblib``.

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scheduler.joblib"
    joblib.dump(scheduler, path)
    saved("scheduler", path)
    return path


def load_scheduler(path: str | Path) -> Scheduler:
    """Load a saved scheduler, ready to continue ask/tell.

    Parameters
    ----------
    path : str or Path
        Path to ``scheduler.joblib``.

    Returns
    -------
    Scheduler
        The restored scheduler with full CMA-ES state.
    """
    scheduler: Scheduler = joblib.load(path)
    loaded("scheduler", path, f"{len(scheduler.archive)} cells")
    return scheduler
