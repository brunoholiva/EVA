"""Archive visualization via pyribs."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from ribs.archives import GridArchive

from optimization.plotting import create_parallel_axes_figure

console = Console()


def visualize_archive(
    archive: GridArchive,
    output_dir: str | Path,
    filename: str = "parallel_axes.png",
    dimension_names: list[str] | None = None,
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
    dimension_names : list of str or None
        Names of enabled archive dimensions.

    Returns
    -------
    Path
        Path to the saved image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir / filename

    if dimension_names is None:
        dimension_names = [f"dim_{i}" for i in range(archive.measure_dim)]

    fig = create_parallel_axes_figure(archive, dimension_names)
    if fig is None:
        console.print(f"No elites to visualize — skipped {img_path}")
        return img_path

    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved parallel axes → {img_path}")
    return img_path
