"""Shared plotting helpers for archive visualization."""

from __future__ import annotations

from matplotlib.figure import Figure
from ribs.archives import GridArchive


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
    import matplotlib

    matplotlib.use("Agg")
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
    )
    ax.set_title("Archive — Parallel Axes (color = P(active))")
    fig.tight_layout()
    return fig
