"""State persistence: save and load scheduler, archive, and archive novelty cache."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler

from optimization.reporting import _rank_archive
from scoring.archive_novelty import ArchiveNoveltyScorer

console = Console()


def save_archive(
    archive: GridArchive,
    decode_fn,
    output_dir: str | Path,
    suffix: str = "archive",
    dimension_names: list[str] | None = None,
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

    Returns
    -------
    Path
        Directory where files were written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = output_dir / f"{suffix}.joblib"
    joblib.dump(archive, joblib_path)
    console.print(f"Saved archive → {joblib_path}")

    csv_path = output_dir / f"{suffix}.csv"
    _export_archive_csv(archive, decode_fn, csv_path, dimension_names)
    console.print(f"Saved CSV     → {csv_path}")

    return output_dir


def load_archive(path: str | Path) -> GridArchive:
    """Load a GridArchive from a joblib file.

    Parameters
    ----------
    path : str or Path
        Path to ``archive.joblib``.

    Returns
    -------
    GridArchive
        The restored archive.
    """
    archive: GridArchive = joblib.load(path)
    console.print(f"Loaded archive from {path} ({len(archive)} cells)")
    return archive


def _export_archive_csv(
    archive: GridArchive,
    decode_fn,
    csv_path: Path,
    dimension_names: list[str] | None = None,
) -> None:
    """Write archive contents to a CSV file."""
    if dimension_names is None:
        dimension_names = [f"dim_{i}" for i in range(archive.measure_dim)]

    if len(archive) == 0:
        columns = ["rank", "smiles", "p_active"] + dimension_names
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
        return

    order, _, smiles = _rank_archive(archive, decode_fn, len(archive))
    arch_data = archive.data()

    data = {
        "rank": np.arange(1, len(order) + 1),
        "smiles": smiles,
        "p_active": arch_data["objective"][order],
    }
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
    console.print(f"Saved scheduler → {path}")
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
    console.print(f"Loaded scheduler from {path} ({len(scheduler.archive)} cells)")
    return scheduler


def save_archive_novelty_cache(
    scorer: ArchiveNoveltyScorer,
    output_dir: str | Path,
) -> Path:
    """Persist the archive novelty scorer's fingerprint cache.

    Parameters
    ----------
    scorer : ArchiveNoveltyScorer
        Scorer whose cache to save.
    output_dir : str or Path
        Directory to write ``archive_novelty_cache.joblib``.

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "archive_novelty_cache.joblib"
    scorer.save(path)
    console.print(f"Saved archive novelty cache → {path}")
    return path


def load_archive_novelty_cache(
    path: str | Path,
    n_bits: int = 2048,
    radius: int = 2,
    max_cache_size: int = 5000,
    n_neighbors: int = 5,
) -> ArchiveNoveltyScorer:
    """Load an archive novelty scorer from disk, or create an empty one.

    Parameters
    ----------
    path : str or Path
        Path to ``archive_novelty_cache.joblib``.
    n_bits : int
        Fallback fingerprint size if file doesn't exist.
    radius : int
        Fallback radius if file doesn't exist.
    max_cache_size : int
        Fallback max cache size if file doesn't exist.
    n_neighbors : int
        Fallback number of nearest neighbors if file doesn't exist.

    Returns
    -------
    ArchiveNoveltyScorer
        Scorer with restored (or empty) cache.
    """
    p = Path(path)
    if p.exists():
        scorer = ArchiveNoveltyScorer.load(p)
        console.print(
            f"Loaded archive novelty cache from {p} ({scorer.cache_size} fingerprints)"
        )
        return scorer
    console.print("No archive novelty cache found — starting with empty cache")
    return ArchiveNoveltyScorer(
        n_bits=n_bits,
        radius=radius,
        max_cache_size=max_cache_size,
        n_neighbors=n_neighbors,
    )
