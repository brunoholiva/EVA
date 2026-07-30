"""Compare experiment results from a batch directory.

Usage:
    python scripts/experiments/compare_results.py results/batch1

Scans each subdirectory for result_archive.csv and prints a table with:
    run_name  cells  coverage  best_P(active)  mean_P(active)  qd_score

For a quick comparison after experiments are done.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _try_load_tb_scalar(tb_dir: Path, tag: str) -> float | None:
    """Read the last value of a TensorBoard scalar event.

    Falls back to None if tensorboard isn't installed or the tag
    doesn't exist. Not the primary method — we prefer CSV data.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return None

    if not tb_dir.is_dir():
        return None

    try:
        ea = EventAccumulator(str(tb_dir))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if tag not in tags:
            return None
        events = ea.Scalars(tag)
        return events[-1].value if events else None
    except Exception:
        return None


def analyze_run(run_dir: Path) -> dict:
    """Extract summary stats from a single experiment run."""
    result = {
        "name": run_dir.name,
        "cells": None,
        "best_pa": None,
        "mean_pa": None,
        "qd_score": None,
        "n_dims": None,
        "config_path": None,
    }

    # Check for config
    config_path = run_dir / "config.toml"
    if config_path.exists():
        result["config_path"] = str(config_path)

    # Read result_archive.csv
    csv_path = run_dir / "result_archive.csv"
    if not csv_path.exists():
        csv_path = run_dir / "archive.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        n_cells = len(df)
        result["cells"] = n_cells

        if "p_active" in df.columns:
            pa = df["p_active"].values
            valid = pa > -1
            if valid.any():
                result["best_pa"] = float(pa[valid].max())
                result["mean_pa"] = float(pa[valid].mean())

    # Try to get QD score and coverage from TensorBoard
    tb_dir = run_dir / "tensorboard"
    if tb_dir.exists():
        coverage = _try_load_tb_scalar(tb_dir, "result_archive/coverage")
        if coverage is not None:
            result["coverage"] = coverage
        qd = _try_load_tb_scalar(tb_dir, "result_archive/qd_score")
        if qd is not None:
            result["qd_score"] = qd

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EVA experiment results")
    parser.add_argument("batch_dir", type=str, help="Path to batch results directory")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_dir():
        print(f"Error: not a directory: {batch_dir}", file=sys.stderr)
        sys.exit(1)

    # Find run directories (subdirs with a config or CSV)
    run_dirs = sorted(
        d for d in batch_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and (
            (d / "result_archive.csv").exists()
            or (d / "archive.csv").exists()
            or (d / "config.toml").exists()
        )
    )

    if not run_dirs:
        print(f"No experiment directories found in {batch_dir}")
        sys.exit(1)

    results = [analyze_run(d) for d in run_dirs]

    print(f"\n{'Run':20s}  {'Cells':>6s}  {'Coverage':>8s}  "
          f"{'Best P(act)':>11s}  {'Mean P(act)':>11s}  {'QD Score':>9s}")
    print("-" * 75)

    for r in results:
        cells = str(r["cells"]) if r["cells"] is not None else "—"
        cov = f'{r["coverage"]:.3f}' if r.get("coverage") is not None else "—"
        best = f'{r["best_pa"]:.4f}' if r["best_pa"] is not None else "—"
        mean_ = f'{r["mean_pa"]:.4f}' if r["mean_pa"] is not None else "—"
        qd = f'{r["qd_score"]:.2f}' if r.get("qd_score") is not None else "—"
        print(f'{r["name"]:20s}  {cells:>6s}  {cov:>8s}  {best:>11s}  {mean_:>11s}  {qd:>9s}')

    print()


if __name__ == "__main__":
    main()
