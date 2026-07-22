"""Archive visualization via pyribs."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from ribs.archives import GridArchive

console = Console()

AXIS_LABELS = ["BR-SAScore", "Novelty", "Proximity"]


def visualize_archive(
    archive: GridArchive,
    output_dir: str | Path,
    filename: str = "parallel_axes.png",
) -> Path:
    """Render a parallel axes plot and save it as PNG.

    Parameters
    ----------
    archive : GridArchive
        The archive to visualize.
    output_dir : str or Path
        Directory to write the image.
    filename : str
        Name of the output image file (default ``"parallel_axes.png"``).

    Returns
    -------
    Path
        Path to the saved image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ribs.visualize import parallel_axes_plot

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir / filename

    fig, ax = plt.subplots(figsize=(12, 6))
    measure_order = [
        (i, label) for i, label in enumerate(AXIS_LABELS[: archive.measure_dim])
    ]
    parallel_axes_plot(
        archive,
        ax=ax,
        measure_order=measure_order,
        cmap="magma",
    )
    ax.set_title("Archive — Parallel Axes (color = P(active))")
    fig.tight_layout()
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved parallel axes → {img_path}")
    return img_path
