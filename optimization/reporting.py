"""Console reporting for the CMA-MAE evolution loop."""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.table import Table
from ribs.archives import GridArchive

from optimization.evaluator import EvalResult

console = Console()


def print_generation(
    gen: int,
    result: EvalResult,
    archive: GridArchive,
    result_archive: GridArchive | None = None,
) -> None:
    """Print a summary table for the current generation.

    Parameters
    ----------
    gen : int
        Generation index.
    result : EvalResult
        Scoring results for this generation.
    archive : GridArchive
        Primary archive state.
    result_archive : GridArchive or None
        Best-so-far archive; used for reporting if provided.
    """
    report = result_archive if result_archive is not None else archive

    table = Table(title=f"Generation {gen}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Valid candidates", str(result.n_valid))
    table.add_row("Archive size", str(len(report)))
    table.add_row("Best P(active)", _format_best_pa(result))
    table.add_row("Mean P(active)", _format_mean_pa(result))

    if len(report) > 0:
        arch_data = report.data()
        best_idx = np.argmax(arch_data["objective"])
        table.add_row("Archive best obj", f"{arch_data['objective'][best_idx]:.4f}")

    table.add_row("Gen time", f"{result.gen_time:.1f}s")
    console.print(table)


def print_results(
    archive: GridArchive,
    decode_fn,
    top_n: int = 10,
    result_archive: GridArchive | None = None,
) -> None:
    """Print the final archive summary and top candidates.

    Parameters
    ----------
    archive : GridArchive
        Primary archive.
    decode_fn : callable
        Latent vectors → SMILES decoder.
    top_n : int
        Number of top candidates to display.
    result_archive : GridArchive or None
        Best-so-far archive; used for reporting if provided.
    """
    report = result_archive if result_archive is not None else archive

    console.rule("[bold green]Results")
    console.print(f"Archive size: {len(report)}")

    if len(report) == 0:
        return

    arch_data = report.data()
    best_idx = np.argmax(arch_data["objective"])
    console.print(f"Best P(active): {arch_data['objective'][best_idx]:.4f}")

    order, solutions, smiles = _rank_archive(report, decode_fn, top_n)

    table = Table(title="Top Candidates", show_header=True)
    table.add_column("Rank", style="cyan")
    table.add_column("SMILES", style="white")
    table.add_column("P(active)", style="magenta")
    table.add_column("BR-SAScore", style="yellow")
    table.add_column("Novelty", style="green")
    table.add_column("Proximity", style="blue")

    for rank, idx in enumerate(order, 1):
        smi = smiles[rank - 1] if rank - 1 < len(smiles) else ""
        table.add_row(
            str(rank),
            smi,
            f"{arch_data['objective'][idx]:.4f}",
            f"{arch_data['measures'][idx][0]:.2f}",
            f"{arch_data['measures'][idx][1]:.3f}",
            f"{arch_data['measures'][idx][2]:.3f}",
        )
    console.print(table)


def _rank_archive(
    archive: GridArchive,
    decode_fn,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sort archive by objective and decode the top solutions.

    Parameters
    ----------
    archive : GridArchive
        The archive to rank.
    decode_fn : callable
        Latent vectors → SMILES decoder.
    top_n : int
        Number of top candidates to return.

    Returns
    -------
    order : np.ndarray
        Indices into archive.data(), sorted by descending objective.
    solutions : np.ndarray
        Latent vectors for the top candidates.
    smiles : list of str
        Decoded SMILES for the top candidates.
    """
    arch_data = archive.data()
    order = np.argsort(-arch_data["objective"])[:top_n]
    solutions = np.array([arch_data["solution"][i] for i in order])
    smiles = decode_fn(solutions)
    return order, solutions, smiles


def _format_best_pa(result: EvalResult) -> str:
    """Format the best P(active) from a generation."""
    if result.n_valid == 0:
        return "N/A"
    active = result.p_active[result.p_active > 0]
    return f"{max(active):.4f}" if len(active) > 0 else "N/A"


def _format_mean_pa(result: EvalResult) -> str:
    """Format the mean P(active) from valid candidates."""
    if result.n_valid == 0:
        return "N/A"
    active = result.p_active[result.p_active > 0]
    return f"{active.mean():.4f}" if len(active) > 0 else "N/A"
