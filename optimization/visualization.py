"""Archive visualization via pyribs heatmaps."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from ribs.archives import GridArchive

console = Console()


def visualize_archive(
    archive: GridArchive,
    output_dir: str | Path,
    filename: str = "heatmap.png",
) -> Path:
    """Render a pyribs grid archive heatmap and save it as PNG.

    Parameters
    ----------
    archive : GridArchive
        The archive to visualize.
    output_dir : str or Path
        Directory to write the image.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Path to the saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ribs.visualize import grid_archive_heatmap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir / filename

    fig, ax = plt.subplots(figsize=(10, 8))
    grid_archive_heatmap(archive, ax=ax, cmap="magma")
    ax.set_title("Archive Heatmap — BR-SAScore vs Novelty")
    fig.tight_layout()
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Saved heatmap → {img_path}")
    return img_path
