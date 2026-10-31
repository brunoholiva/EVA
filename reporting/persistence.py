"""State persistence: checkpoints, candidate CSVs, and JSON summaries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib

from config import ExperimentConfig
from optimization.candidates import CandidateStore
from optimization.cma import Solver
from optimization.loop import SearchLoop
from reporting.console import loaded, saved


@dataclass
class CheckpointState:
    """Serializable snapshot of a running :class:`~optimization.loop.SearchLoop`."""

    solver: Solver
    seen_smiles: set[str]
    candidates: CandidateStore
    step: int
    attempts: int
    oracle_calls: int
    duplicate_attempts: int
    invalid_attempts: int
    elapsed_s: float
    solver_name: str
    seed: int


def save_checkpoint(
    loop: SearchLoop,
    seen_smiles: set[str],
    output_dir: str | Path,
    *,
    quiet: bool = False,
) -> Path:
    """Persist full solver, deduplication, and candidate state to ``checkpoint.joblib``.

    Parameters
    ----------
    loop : SearchLoop
        Active search loop.
    seen_smiles : set of str
        Canonical SMILES already evaluated by the oracle.
    output_dir : str or Path
        Run output directory.
    quiet : bool, default=False
        Suppress terminal print when True (used on frequent background saves).

    Returns
    -------
    Path
        Path to ``checkpoint.joblib``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "checkpoint.joblib"
    state = CheckpointState(
        solver=loop.solver,
        seen_smiles=set(seen_smiles),
        candidates=loop.candidates,
        step=loop.step_count,
        attempts=loop.attempts,
        oracle_calls=loop.oracle_calls,
        duplicate_attempts=loop.duplicate_attempts,
        invalid_attempts=loop.invalid_attempts,
        elapsed_s=loop.elapsed_s,
        solver_name=loop.solver_name,
        seed=loop.seed,
    )
    joblib.dump(state, path)
    if not quiet:
        saved("checkpoint", path, f"step {state.step}, {state.oracle_calls:,} calls")
    return path


def load_checkpoint(path: str | Path) -> CheckpointState:
    """Load a :class:`CheckpointState` from ``checkpoint.joblib``."""
    state: CheckpointState = joblib.load(path)
    loaded(
        "checkpoint",
        path,
        f"step {state.step}, {state.oracle_calls:,} oracle calls",
    )
    return state


def save_candidates(
    candidates: CandidateStore,
    output_dir: str | Path,
    *,
    quiet: bool = False,
) -> Path:
    """Write ranked active hits to ``candidates.csv``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "candidates.csv"
    df = candidates.to_dataframe()
    df.to_csv(csv_path, index=False)
    if not quiet:
        saved("candidates", csv_path, f"{len(df):,} molecules")
    return csv_path


def save_summary(summary: dict[str, Any], output_dir: str | Path) -> Path:
    """Write ``summary.json`` to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    saved("summary", path)
    return path


def save_resolved_config(cfg: ExperimentConfig, output_dir: str | Path) -> Path:
    """Write resolved ``ExperimentConfig`` as ``config.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    path.write_text(json.dumps(asdict(cfg), indent=2, default=str))
    saved("config", path)
    return path
