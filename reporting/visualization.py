"""Archive visualization via pyribs."""

from __future__ import annotations

from pathlib import Path

from ribs.archives import GridArchive

from reporting.console import saved, skipped
from reporting.plotting import (
    configure_matplotlib_agg,
    create_archive_figure,
)


def visualize_archive(
    archive: GridArchive,
    output_dir: str | Path,
    filename: str | None = None,
    dimension_names: list[str] | None = None,
) -> Path:
    """Render an archive figure and save it as PNG.

    Archives with 1–2 measure dimensions are rendered as a grid heatmap;
    archives with 3+ dimensions use a parallel axes plot.

    Parameters
    ----------
    archive : GridArchive
        The archive to visualize.
    output_dir : str or Path
        Directory to write the image.
    filename : str or None
        Name of the output image file. If ``None``, uses ``"heatmap.png"``
        for 1–2 dimensional archives and ``"parallel_axes.png"`` otherwise.
    dimension_names : list of str or None
        Names of enabled archive dimensions.

    Returns
    -------
    Path
        Path to the saved image.
    """
    configure_matplotlib_agg()
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = "heatmap.png" if archive.measure_dim <= 2 else "parallel_axes.png"
    img_path = output_dir / filename

    if dimension_names is None:
        dimension_names = [f"dim_{i}" for i in range(archive.measure_dim)]

    fig = create_archive_figure(archive, dimension_names)
    if fig is None:
        skipped(img_path, "elites to visualize")
        return img_path

    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved("archive plot", img_path)
    return img_path
