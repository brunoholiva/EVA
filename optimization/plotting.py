"""Shared plotting helpers for archive visualization."""

from __future__ import annotations

from matplotlib.figure import Figure
from ribs.archives import GridArchive


def configure_matplotlib_agg() -> None:
    """Force matplotlib onto the headless Agg backend.

    Must run before ``pyplot`` is imported. Headless hosts (e.g. Slurm)
    have no display, and the default backend raises a ``RuntimeError``
    during figure creation otherwise.
    """
    import matplotlib

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
) -> Figure | None:
    """Create a grid heatmap figure from a 1D/2D archive.

    Parameters
    ----------
    archive : GridArchive
        The archive to plot (1 or 2 measure dimensions).
    dimension_names : list of str
        Names of enabled archive dimensions.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or ``None`` if the archive is empty.
    """
    configure_matplotlib_agg()
    import matplotlib.pyplot as plt
    from ribs.visualize import grid_archive_heatmap

    if len(archive) == 0:
        return None

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
) -> Figure | None:
    """Create a parallel axes figure from an archive.

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
    configure_matplotlib_agg()
    import matplotlib.pyplot as plt
    from ribs.visualize import parallel_axes_plot

    if len(archive) == 0:
        return None

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
