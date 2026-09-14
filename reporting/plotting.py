"""Archive figure rendering (heatmap / parallel axes) and PNG export."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.figure import Figure
from ribs.archives import GridArchive

from reporting.console import saved, skipped

matplotlib.use("Agg")


def create_archive_figure(
    archive: GridArchive,
    dimension_names: list[str],
) -> Figure | None:
    """Create a figure for an archive, dispatching on dimensionality.

    Archives with 1–2 measure dimensions are rendered as a grid heatmap
    (colored by objective); archives with 3+ dimensions use a parallel
    axes plot.

    Parameters
    ----------
    archive : GridArchive
        The archive to plot.
    dimension_names : list of str
        Names of enabled archive dimensions.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or ``None`` if the archive is empty.
    """
    if len(archive) == 0:
        return None
    if archive.measure_dim <= 2:
        return create_grid_heatmap_figure(archive, dimension_names)
    return create_parallel_axes_figure(archive, dimension_names)


def create_grid_heatmap_figure(
    archive: GridArchive,
    dimension_names: list[str],
) -> Figure:
    """Create a grid heatmap figure from a 1D/2D archive.

    Parameters
    ----------
    archive : GridArchive
        The archive to plot (1 or 2 measure dimensions).
    dimension_names : list of str
        Names of enabled archive dimensions.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt
    from ribs.visualize import grid_archive_heatmap

    fig, ax = plt.subplots(figsize=(8, 6))
    grid_archive_heatmap(
        archive,
        ax=ax,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel(dimension_names[0])
    if archive.measure_dim > 1:
        ax.set_ylabel(dimension_names[1])
    ax.set_title("Archive — Heatmap (color = P(active))")
    fig.tight_layout()
    return fig


def create_parallel_axes_figure(
    archive: GridArchive,
    dimension_names: list[str],
) -> Figure:
    """Create a parallel axes figure from an archive.

    Parameters
    ----------
    archive : GridArchive
        The archive to plot.
    dimension_names : list of str
        Names of enabled archive dimensions.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt
    from ribs.visualize import parallel_axes_plot

    fig, ax = plt.subplots(figsize=(12, 6))
    measure_order = [
        (i, label) for i, label in enumerate(dimension_names[: archive.measure_dim])
    ]
    parallel_axes_plot(
        archive,
        ax=ax,
        measure_order=measure_order,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("Archive — Parallel Axes (color = P(active))")
    fig.tight_layout()
    return fig


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
