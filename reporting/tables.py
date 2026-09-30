"""Console reporting for the CMA-MAE evolution loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ribs.archives import GridArchive

from reporting.console import console, make_table, section

if TYPE_CHECKING:
    from optimization.evaluator import EvalResult


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

    table = make_table(
        f"Generation {gen}",
        [("Metric", "cyan"), ("Value", "magenta")],
    )

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
    dimension_names: list[str] | None = None,
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
    dimension_names : list of str or None
        Names of enabled archive dimensions.
    """
    report = result_archive if result_archive is not None else archive

    section("Results")
    console.print(f"Archive size: {len(report)}")

    if len(report) == 0:
        return

    arch_data = report.data()
    best_idx = np.argmax(arch_data["objective"])
    console.print(f"Best P(active): {arch_data['objective'][best_idx]:.4f}")

    order, _, smiles = _rank_archive(report, decode_fn, top_n)

    if dimension_names is None:
        dimension_names = [f"dim_{i}" for i in range(report.measure_dim)]

    columns = [
        ("Rank", "cyan"),
        ("SMILES", "white"),
        ("P(active)", "magenta"),
    ]
    for name in dimension_names:
        columns.append((name, "yellow"))

    table = make_table("Top Candidates", columns)

    for rank, idx in enumerate(order, 1):
        smi = smiles[rank - 1] if rank - 1 < len(smiles) else ""
        row = [
            str(rank),
            smi,
            f"{arch_data['objective'][idx]:.4f}",
        ]
        for d in range(report.measure_dim):
            row.append(f"{arch_data['measures'][idx][d]:.3f}")
        table.add_row(*row)
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
