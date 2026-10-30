"""Oracle-budget benchmark loop and run artifacts.

The loop spends a fixed number of **oracle calls** — first-seen chemically
valid canonical SMILES charged to the activity predictor. Invalid and duplicate
molecules are free, so the number of attempts needed to reach the budget is a
first-class result: it is the proposal effort.

Every attempt (including invalid and duplicate molecules) is streamed to
``all_evaluations.csv`` so any downstream metric can be recomputed. Per-step
cumulative counters go to ``curve.csv``; the final state to ``summary.json``.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.generators import ARCHIVE_STAT_COLUMNS, Generator
from evaluation.evaluator import EvalResult

ATTEMPT_COLUMNS = [
    "method",
    "seed",
    "step",
    "attempt",
    "smiles",
    "canonical_smiles",
    "valid",
    "duplicate",
    "oracle_called",
    "p_active",
]

CURVE_COLUMNS = [
    "method",
    "seed",
    "step",
    "elapsed_s",
    "oracle_calls",
    "unique_scored",
    "duplicate_attempts",
    "invalid_attempts",
    "validity",
    "best_p_active",
    *ARCHIVE_STAT_COLUMNS,
]

_ARCHIVE_STAT_DEFAULTS = {
    "result_cells": 0,
    "result_coverage": 0.0,
    "result_qd_score": 0.0,
    "result_obj_max": float("nan"),
}
"""Written for generators that have no archive (e.g. ``naive``)."""


_MAX_STEP_MULTIPLIER = 20
"""A step may score nothing new (all duplicates); cap steps at 20x the minimum."""


def _classify(
    result: EvalResult, seen: set[str]
) -> list[tuple[bool, bool, str | None, bool]]:
    """Classify each attempt as ``(valid, duplicate, canonical, charged)``.

    ``charged`` means first-seen valid: the molecule is billed to the predictor.
    The per-attempt counts are asserted against ``EvalResult``'s own counters by
    the caller, so this reconstruction cannot silently drift from the evaluator.
    """
    rows = []
    for i in range(len(result.smiles)):
        canonical = result.canonical[i] if result.canonical is not None else None
        valid = canonical is not None
        duplicate = valid and canonical in seen
        charged = valid and not duplicate
        if charged:
            seen.add(canonical)
        rows.append((valid, duplicate, canonical, charged))
    return rows


def run_benchmark(
    generator: Generator,
    *,
    method: str,
    seed: int,
    budget: int,
    output_dir: Path,
    max_steps: int | None = None,
    log_every: int = 20,
) -> dict:
    """Run *generator* until *budget* oracle calls are spent.

    Parameters
    ----------
    generator : Generator
        Proposes and scores one batch per :meth:`step`.
    method : str
        Method label written into every CSV row.
    seed : int
        Seed label written into every CSV row.
    budget : int
        Target number of first-seen valid molecules (oracle calls). The final
        step may overshoot by up to one batch.
    output_dir : Path
        Directory for ``all_evaluations.csv``, ``curve.csv`` and ``summary.json``.
    max_steps : int or None, default=None
        Safety cap so a saturated generator cannot spin. Defaults to
        ``max(1000, 20 * ceil(budget / generator.batch_size))``.
    log_every : int, default=20
        Print a progress line every this many steps.

    Returns
    -------
    dict
        The summary written to ``summary.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_steps is None:
        # Generators report 0 before their first ask (bandit pools), so guard
        # the divisor rather than crash a run that would have worked.
        min_steps = max(1, int(np.ceil(budget / max(1, generator.batch_size))))
        max_steps = max(1000, _MAX_STEP_MULTIPLIER * min_steps)

    seen: set[str] = set()
    scored: dict[str, float] = {}
    attempt_idx = 0
    duplicate_attempts = 0
    invalid_attempts = 0
    oracle_calls = 0
    best_p_active = 0.0
    step = 0
    t0 = time.time()

    attempts_path = output_dir / "all_evaluations.csv"
    curve_path = output_dir / "curve.csv"

    with (
        attempts_path.open("w", newline="") as f_attempts,
        curve_path.open("w", newline="") as f_curve,
    ):
        writer_attempts = csv.writer(f_attempts)
        writer_curve = csv.writer(f_curve)
        writer_attempts.writerow(ATTEMPT_COLUMNS)
        writer_curve.writerow(CURVE_COLUMNS)

        while oracle_calls < budget and step < max_steps:
            result = generator.step()
            classified = _classify(result, seen)

            step_duplicate = step_invalid = step_scored = 0
            for i, (valid, duplicate, canonical, charged) in enumerate(classified):
                p_active = float(result.p_active[i]) if charged else float("nan")
                writer_attempts.writerow(
                    [
                        method,
                        seed,
                        step,
                        attempt_idx,
                        result.smiles[i],
                        canonical,
                        valid,
                        duplicate,
                        charged,
                        p_active,
                    ]
                )
                attempt_idx += 1
                if not valid:
                    step_invalid += 1
                elif duplicate:
                    step_duplicate += 1
                else:
                    step_scored += 1
                    scored[canonical] = p_active
                    best_p_active = max(best_p_active, p_active)

            _assert_accounting(result, step_scored, step_duplicate)

            duplicate_attempts += step_duplicate
            invalid_attempts += step_invalid
            oracle_calls += step_scored

            validity = (
                (attempt_idx - invalid_attempts) / attempt_idx if attempt_idx else 0.0
            )
            archive_stats = generator.archive_stats()
            writer_curve.writerow(
                [
                    method,
                    seed,
                    step,
                    time.time() - t0,
                    oracle_calls,
                    len(scored),
                    duplicate_attempts,
                    invalid_attempts,
                    validity,
                    best_p_active,
                    *[
                        archive_stats.get(column, _ARCHIVE_STAT_DEFAULTS[column])
                        for column in ARCHIVE_STAT_COLUMNS
                    ],
                ]
            )
            f_attempts.flush()
            f_curve.flush()

            if log_every and step % log_every == 0:
                print(
                    f"[{method} seed {seed}] step {step} | "
                    f"oracle {oracle_calls}/{budget} | "
                    f"unique {len(scored)} | dup {duplicate_attempts} | "
                    f"invalid {invalid_attempts} | best {best_p_active:.4f}"
                )
            step += 1

    summary = {
        "method": method,
        "seed": seed,
        "budget": budget,
        "steps": step,
        "attempts": attempt_idx,
        "oracle_calls": oracle_calls,
        "unique_scored": len(scored),
        "duplicate_attempts": duplicate_attempts,
        "invalid_attempts": invalid_attempts,
        "validity": (
            (attempt_idx - invalid_attempts) / attempt_idx if attempt_idx else 0.0
        ),
        "best_p_active": best_p_active,
        "stopped_on": "budget" if oracle_calls >= budget else "max_steps",
        "elapsed_s": time.time() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _assert_accounting(
    result: EvalResult, step_scored: int, step_duplicate: int
) -> None:
    """Assert the CSV classification matches the evaluator's own counters.

    A mismatch means the reporting path and the evaluator disagree about what
    was charged; fail loudly rather than write an inconsistent CSV.
    """
    assert (
        step_scored == result.n_oracle_calls
    ), f"scored={step_scored} != n_oracle_calls={result.n_oracle_calls}"
    assert (
        step_duplicate == result.n_duplicates
    ), f"duplicate={step_duplicate} != n_duplicates={result.n_duplicates}"
    assert (
        step_scored + step_duplicate == result.n_valid
    ), f"valid={step_scored + step_duplicate} != n_valid={result.n_valid}"
