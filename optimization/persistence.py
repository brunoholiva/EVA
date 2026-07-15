"""Archive persistence: save and load GridArchive state."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from ribs.archives import GridArchive

from optimization.reporting import _rank_archive

console = Console()


def save_archive(
    archive: GridArchive,
    decode_fn,
    output_dir: str | Path,
) -> Path:
    """Save the archive as joblib (full state) and CSV (tabular data).

    Parameters
    ----------
    archive : GridArchive
        The archive to persist.
    decode_fn : callable
        Latent vectors → SMILES decoder.
    output_dir : str or Path
        Directory to write ``archive.joblib`` and ``archive.csv``.

    Returns
    -------
    Path
        Directory where files were written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = output_dir / "archive.joblib"
    joblib.dump(archive, joblib_path)
    console.print(f"Saved archive → {joblib_path}")

    csv_path = output_dir / "archive.csv"
    _export_archive_csv(archive, decode_fn, csv_path)
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
) -> None:
    """Write archive contents to a CSV file."""
    if len(archive) == 0:
        pd.DataFrame(
            columns=["rank", "smiles", "p_active", "br_sascore", "novelty"]
        ).to_csv(csv_path, index=False)
        return

    order, _, smiles = _rank_archive(archive, decode_fn, len(archive))
    arch_data = archive.data()

    df = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "smiles": smiles,
            "p_active": arch_data["objective"][order],
            "br_sascore": arch_data["measures"][order, 0],
            "novelty": arch_data["measures"][order, 1],
        }
    )
    df.to_csv(csv_path, index=False)
