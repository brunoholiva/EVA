"""Unified oracle-budget and generation-budget search loop."""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.evaluator import EvalResult
from optimization.candidates import CandidateStore
from optimization.cma import Solver, WalkerStats

ATTEMPT_COLUMNS = [
    "solver",
    "seed",
    "step",
    "attempt",
    "walker_id",
    "smiles",
    "canonical_smiles",
    "valid",
    "duplicate",
    "oracle_called",
    "p_active",
]

CURVE_COLUMNS = [
    "solver",
    "seed",
    "step",
    "elapsed_s",
    "attempts",
    "oracle_calls",
    "duplicate_attempts",
    "invalid_attempts",
    "validity",
    "best_p_active",
    "n_hits_05",
    "n_hits_06",
    "n_hits_08",
    "unique_scaffolds_06",
    "walker_spread",
    "total_restarts",
]

_MAX_STEP_MULTIPLIER = 20
"""Safety cap factor when running under an oracle budget without explicit max gens."""


@dataclass
class StepSnapshot:
    """State and metrics emitted at the end of a single search step."""

    step: int
    elapsed_s: float
    attempts: int
    oracle_calls: int
    duplicate_attempts: int
    invalid_attempts: int
    validity: float
    best_p_active: float
    n_hits_05: int
    n_hits_06: int
    n_hits_08: int
    unique_scaffolds_06: int
    walker_spread: float
    total_restarts: int
    result: EvalResult
    walker_ids: np.ndarray
    walker_stats: list[WalkerStats]


StepCallback = Callable[[StepSnapshot], None]


def resolve_max_generations(
    oracle_budget: int,
    max_generations: int,
    batch_size: int,
) -> int:
    """Compute the effective step ceiling for a run.

    Parameters
    ----------
    oracle_budget : int
        Target oracle calls (``<= 0`` if disabled).
    max_generations : int
        Explicit generation cap (``<= 0`` to derive a safety ceiling from
        *oracle_budget*).
    batch_size : int
        Expected proposals per step.

    Returns
    -------
    int
        Maximum number of steps permitted.
    """
    if max_generations > 0:
        return max_generations
    min_steps = max(1, int(np.ceil(oracle_budget / max(1, batch_size))))
    return max(1000, _MAX_STEP_MULTIPLIER * min_steps)


class SearchLoop:
    """Executes ``ask -> evaluate -> tell`` with CSV streaming and checkpoints.

    Parameters
    ----------
    solver : Solver
        Latent-space proposal engine (``MultiCMASolver`` or ``NaiveSolver``).
    evaluate : callable
        ``Evaluator`` scoring a ``(batch, dim)`` latent array.
    candidates : CandidateStore
        Hit store accumulating unique active molecules.
    solver_name : str
        Solver label written into CSV and summary outputs.
    seed : int
        Run seed label written into CSV and summary outputs.
    step : int, default=0
        Initial step counter (non-zero when resuming).
    attempts : int, default=0
        Initial total attempt counter.
    oracle_calls : int, default=0
        Initial oracle-call counter.
    duplicate_attempts : int, default=0
        Initial duplicate-attempt counter.
    invalid_attempts : int, default=0
        Initial invalid-attempt counter.
    elapsed_offset_s : float, default=0.0
        Prior wall-clock seconds accumulated before resuming.
    """

    def __init__(
        self,
        solver: Solver,
        evaluate,
        candidates: CandidateStore,
        *,
        solver_name: str,
        seed: int,
        step: int = 0,
        attempts: int = 0,
        oracle_calls: int = 0,
        duplicate_attempts: int = 0,
        invalid_attempts: int = 0,
        elapsed_offset_s: float = 0.0,
    ) -> None:
        self.solver = solver
        self.evaluate = evaluate
        self.candidates = candidates
        self.solver_name = solver_name
        self.seed = seed
        self.step_count = step
        self.attempts = attempts
        self.oracle_calls = oracle_calls
        self.duplicate_attempts = duplicate_attempts
        self.invalid_attempts = invalid_attempts
        self._elapsed_offset_s = elapsed_offset_s
        self._t_start: float | None = None

    @property
    def elapsed_s(self) -> float:
        """Total wall-clock seconds elapsed across original and resumed segments."""
        if self._t_start is None:
            return self._elapsed_offset_s
        return self._elapsed_offset_s + (time.time() - self._t_start)

    def step(
        self,
        attempts_writer=None,
    ) -> StepSnapshot:
        """Execute one ``ask -> evaluate -> tell`` step.

        Parameters
        ----------
        attempts_writer : csv.writer or None, default=None
            Optional open CSV writer for ``all_evaluations.csv``.

        Returns
        -------
        StepSnapshot
            Completed step metrics and evaluation output.
        """
        if self._t_start is None:
            self._t_start = time.time()

        cur_step = self.step_count
        z, walker_ids = self.solver.ask()
        result: EvalResult = self.evaluate(z)

        t_solver = time.time()
        self.solver.tell(z, result.objectives)
        if result.timings is not None:
            result.timings.solver_ops = time.time() - t_solver

        canonical_list = (
            result.canonical
            if result.canonical is not None
            else [None] * len(result.smiles)
        )
        oracle_mask = (
            result.oracle_mask
            if result.oracle_mask is not None
            else np.zeros(len(result.smiles), dtype=bool)
        )

        step_scored = 0
        step_dup = 0
        step_inv = 0

        for i, smi in enumerate(result.smiles):
            canon = canonical_list[i]
            valid = canon is not None
            charged = bool(oracle_mask[i])
            duplicate = valid and not charged
            pa = float(result.p_active[i]) if charged else float("nan")
            wid = int(walker_ids[i])

            if attempts_writer is not None:
                attempts_writer.writerow(
                    [
                        self.solver_name,
                        self.seed,
                        cur_step,
                        self.attempts,
                        wid,
                        smi,
                        canon,
                        valid,
                        duplicate,
                        charged,
                        pa,
                    ]
                )

            self.attempts += 1
            if not valid:
                step_inv += 1
            elif duplicate:
                step_dup += 1
            else:
                step_scored += 1
                self.oracle_calls += 1
                self.candidates.record_scored(
                    canonical_smiles=canon,
                    smiles=smi,
                    p_active=pa,
                    walker_id=wid,
                    step=cur_step,
                    oracle_call=self.oracle_calls,
                )

        assert (
            step_scored == result.n_oracle_calls
        ), f"scored={step_scored} != n_oracle_calls={result.n_oracle_calls}"
        assert (
            step_dup == result.n_duplicates
        ), f"dup={step_dup} != n_duplicates={result.n_duplicates}"
        assert (
            step_scored + step_dup == result.n_valid
        ), f"valid={step_scored + step_dup} != n_valid={result.n_valid}"

        self.duplicate_attempts += step_dup
        self.invalid_attempts += step_inv
        self.step_count += 1

        validity = (
            (self.attempts - self.invalid_attempts) / self.attempts
            if self.attempts
            else 0.0
        )

        return StepSnapshot(
            step=cur_step,
            elapsed_s=self.elapsed_s,
            attempts=self.attempts,
            oracle_calls=self.oracle_calls,
            duplicate_attempts=self.duplicate_attempts,
            invalid_attempts=self.invalid_attempts,
            validity=validity,
            best_p_active=self.candidates.best_p_active,
            n_hits_05=self.candidates.n_hits_05,
            n_hits_06=self.candidates.n_hits_06,
            n_hits_08=self.candidates.n_hits_08,
            unique_scaffolds_06=self.candidates.unique_scaffolds_06,
            walker_spread=self.solver.walker_spread(),
            total_restarts=self.solver.total_restarts(),
            result=result,
            walker_ids=walker_ids,
            walker_stats=self.solver.walker_stats(),
        )

    def run(
        self,
        *,
        output_dir: Path,
        oracle_budget: int,
        max_generations: int,
        save_all_evaluations: bool = True,
        log_every: int = 20,
        resuming: bool = False,
        on_step: StepCallback | None = None,
        on_log: StepCallback | None = None,
    ) -> dict[str, Any]:
        """Run the search loop until the oracle or generation budget is met.

        Parameters
        ----------
        output_dir : Path
            Directory where ``curve.csv`` and optional ``all_evaluations.csv``
            are streamed.
        oracle_budget : int
            Target oracle calls (``<= 0`` to ignore oracle budget).
        max_generations : int
            Explicit max generations (``<= 0`` for auto safety ceiling).
        save_all_evaluations : bool, default=True
            Whether to stream every attempt to ``all_evaluations.csv``.
        log_every : int, default=20
            Interval in steps for invoking *on_log*.
        resuming : bool, default=False
            Append to existing CSV streams if True; overwrite with headers if False.
        on_step : callable or None
            Invoked after every step with :class:`StepSnapshot`.
        on_log : callable or None
            Invoked every *log_every* steps and on the final step.

        Returns
        -------
        dict
            Run summary dictionary suitable for ``summary.json``.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        effective_max_gens = resolve_max_generations(
            oracle_budget, max_generations, self.solver.batch_size
        )
        self._t_start = time.time()

        curve_path = output_dir / "curve.csv"
        curve_exists = resuming and curve_path.exists()
        curve_mode = "a" if curve_exists else "w"

        attempts_path = output_dir / "all_evaluations.csv"
        attempts_exists = resuming and attempts_path.exists()
        attempts_mode = "a" if attempts_exists else "w"

        f_attempts = (
            attempts_path.open(attempts_mode, newline="")
            if save_all_evaluations
            else None
        )
        try:
            with curve_path.open(curve_mode, newline="") as f_curve:
                writer_curve = csv.writer(f_curve)
                if not curve_exists:
                    writer_curve.writerow(CURVE_COLUMNS)

                writer_attempts = None
                if f_attempts is not None:
                    writer_attempts = csv.writer(f_attempts)
                    if not attempts_exists:
                        writer_attempts.writerow(ATTEMPT_COLUMNS)

                last_snapshot: StepSnapshot | None = None
                while self._should_continue(oracle_budget, effective_max_gens):
                    snap = self.step(attempts_writer=writer_attempts)
                    last_snapshot = snap

                    writer_curve.writerow(
                        [
                            self.solver_name,
                            self.seed,
                            snap.step,
                            snap.elapsed_s,
                            snap.attempts,
                            snap.oracle_calls,
                            snap.duplicate_attempts,
                            snap.invalid_attempts,
                            snap.validity,
                            snap.best_p_active,
                            snap.n_hits_05,
                            snap.n_hits_06,
                            snap.n_hits_08,
                            snap.unique_scaffolds_06,
                            snap.walker_spread,
                            snap.total_restarts,
                        ]
                    )
                    f_curve.flush()
                    if f_attempts is not None:
                        f_attempts.flush()

                    if on_step is not None:
                        on_step(snap)

                    is_done = not self._should_continue(
                        oracle_budget, effective_max_gens
                    )
                    if on_log is not None and (
                        (log_every > 0 and snap.step % log_every == 0) or is_done
                    ):
                        on_log(snap)
        finally:
            if f_attempts is not None:
                f_attempts.close()

        return self.build_summary(
            oracle_budget=oracle_budget,
            effective_max_gens=effective_max_gens,
            last_snapshot=last_snapshot,
        )

    def _should_continue(self, oracle_budget: int, effective_max_gens: int) -> bool:
        """Return True while neither stopping criterion has been reached."""
        if oracle_budget > 0 and self.oracle_calls >= oracle_budget:
            return False
        if self.step_count >= effective_max_gens:
            return False
        return True

    def build_summary(
        self,
        *,
        oracle_budget: int,
        effective_max_gens: int,
        last_snapshot: StepSnapshot | None = None,
    ) -> dict[str, Any]:
        """Construct the final summary dictionary."""
        validity = (
            (self.attempts - self.invalid_attempts) / self.attempts
            if self.attempts
            else 0.0
        )
        if oracle_budget > 0 and self.oracle_calls >= oracle_budget:
            stopped_on = "oracle_budget"
        elif self.step_count >= effective_max_gens:
            stopped_on = "max_generations"
        else:
            stopped_on = "incomplete"

        spread = (
            last_snapshot.walker_spread
            if last_snapshot is not None
            else self.solver.walker_spread()
        )
        return {
            "solver": self.solver_name,
            "seed": self.seed,
            "oracle_budget": oracle_budget,
            "max_generations": effective_max_gens,
            "steps": self.step_count,
            "attempts": self.attempts,
            "oracle_calls": self.oracle_calls,
            "duplicate_attempts": self.duplicate_attempts,
            "invalid_attempts": self.invalid_attempts,
            "validity": validity,
            "best_p_active": self.candidates.best_p_active,
            "n_candidates_saved": len(self.candidates),
            "n_hits_05": self.candidates.n_hits_05,
            "n_hits_06": self.candidates.n_hits_06,
            "n_hits_08": self.candidates.n_hits_08,
            "unique_scaffolds_06": self.candidates.unique_scaffolds_06,
            "largest_scaffold_frac_06": self.candidates.largest_scaffold_frac_06,
            "walker_spread": spread,
            "total_restarts": self.solver.total_restarts(),
            "stopped_on": stopped_on,
            "elapsed_s": self.elapsed_s,
        }
