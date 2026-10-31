"""Console tables for EVA search progress and final candidate summaries."""

from __future__ import annotations

from typing import Any

from evaluation.evaluator import EvalResult
from optimization.candidates import CandidateStore
from optimization.loop import StepSnapshot
from reporting.console import console, make_table, section


def print_step(snapshot: StepSnapshot, oracle_budget: int = 0) -> None:
    """Print a compact summary table for a completed step."""
    table = make_table(
        f"Step {snapshot.step}",
        [("Metric", "cyan"), ("Value", "magenta")],
    )

    oracle_str = (
        f"{snapshot.oracle_calls:,} / {oracle_budget:,}"
        if oracle_budget > 0
        else f"{snapshot.oracle_calls:,}"
    )
    table.add_row("Oracle calls", oracle_str)
    table.add_row(
        "Batch valid / dup",
        f"{snapshot.result.n_valid} / {snapshot.result.n_duplicates}",
    )
    table.add_row("Batch best P(active)", _format_best_pa(snapshot.result))
    table.add_row("Batch mean P(active)", _format_mean_pa(snapshot.result))
    table.add_row("Best P(active)", f"{snapshot.best_p_active:.4f}")
    table.add_row(
        "Hits (>=0.5 / >=0.6 / >=0.8)",
        f"{snapshot.n_hits_05:,} / {snapshot.n_hits_06:,} / {snapshot.n_hits_08:,}",
    )
    table.add_row("Unique scaffolds (>=0.6)", f"{snapshot.unique_scaffolds_06:,}")
    if snapshot.walker_stats:
        table.add_row("Walker spread", f"{snapshot.walker_spread:.3f}")
        table.add_row("Total restarts", str(snapshot.total_restarts))
    table.add_row("Step time", f"{snapshot.result.gen_time:.1f}s")
    console.print(table)


def print_results(
    candidates: CandidateStore,
    summary: dict[str, Any],
    top_n: int = 10,
) -> None:
    """Print final run summary and the top-ranked candidate molecules."""
    section("Results")
    console.print(
        f"Oracle calls: {summary['oracle_calls']:,} "
        f"({summary['attempts']:,} attempts, "
        f"validity {summary['validity']:.1%}, "
        f"stopped_on={summary['stopped_on']})"
    )
    console.print(
        f"Hits >=0.5: {summary['n_hits_05']:,} | "
        f"Hits >=0.6: {summary['n_hits_06']:,} | "
        f"Hits >=0.8: {summary['n_hits_08']:,} | "
        f"Unique scaffolds (>=0.6): {summary['unique_scaffolds_06']:,} "
        f"(largest {summary['largest_scaffold_frac_06']:.1%})"
    )
    console.print(f"Best P(active): {summary['best_p_active']:.4f}")

    top = candidates.ranked_records(top_n=top_n)
    if not top:
        return

    table = make_table(
        "Top Candidates",
        [
            ("Rank", "cyan"),
            ("Canonical SMILES", "white"),
            ("P(active)", "magenta"),
            ("Generic Scaffold", "yellow"),
            ("Walker", "green"),
            ("Oracle #", "cyan"),
        ],
    )
    for rank, rec in enumerate(top, start=1):
        table.add_row(
            str(rank),
            rec.canonical_smiles,
            f"{rec.p_active:.4f}",
            rec.generic_scaffold,
            str(rec.walker_id),
            f"{rec.oracle_call:,}",
        )
    console.print(table)


def _format_best_pa(result: EvalResult) -> str:
    """Format the highest first-seen ``P(active)`` in the batch."""
    scored = result.p_active[result.p_active > 0]
    return f"{float(scored.max()):.4f}" if len(scored) > 0 else "N/A"


def _format_mean_pa(result: EvalResult) -> str:
    """Format the mean first-seen ``P(active)`` in the batch."""
    scored = result.p_active[result.p_active > 0]
    return f"{float(scored.mean()):.4f}" if len(scored) > 0 else "N/A"
