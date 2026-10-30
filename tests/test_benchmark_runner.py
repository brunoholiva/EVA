"""Tests for the benchmark runner budget loop and accounting.

The runner never trusts the evaluator's counters blindly: it reconstructs each
attempt's outcome from the canonical SMILES and asserts agreement. These tests
pin that invariant and the budget/stop behaviour with stubs — no GPU, no oracle.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from benchmarks.runner import ATTEMPT_COLUMNS, CURVE_COLUMNS, run_benchmark
from evaluation.evaluator import EvalResult


class _StubGenerator:
    """Replays fixed per-step canonical batches, modelling dedup like the oracle."""

    def __init__(self, step_specs: list[list[str | None]]) -> None:
        self._specs = list(step_specs)
        self._i = 0
        self.batch_size = max(len(s) for s in step_specs)
        self._seen: set[str] = set()

    def step(self) -> EvalResult:
        spec = self._specs[self._i % len(self._specs)]
        self._i += 1
        canonical = list(spec)
        valid = [c is not None for c in canonical]
        charged = []
        for c in canonical:
            is_new = c is not None and c not in self._seen
            if is_new:
                self._seen.add(c)
            charged.append(is_new)
        objectives = np.array([0.7 if n else -1.0 for n in charged])
        p_active = np.array([0.7 if n else 0.0 for n in charged])
        return EvalResult(
            smiles=[c or "" for c in canonical],
            objectives=objectives,
            measures=np.zeros((len(spec), 0)),
            p_active=p_active,
            n_valid=sum(valid),
            gen_time=0.0,
            n_duplicates=sum(v and not n for v, n in zip(valid, charged)),
            n_oracle_calls=sum(charged),
            canonical=canonical,
        )

    def archive_stats(self) -> dict[str, float]:
        """No archive in this stub."""
        return {}


def _read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_stops_at_oracle_budget(tmp_path):
    specs = [[f"A{i}", f"B{i}", None] for i in range(6)]
    gen = _StubGenerator(specs)

    summary = run_benchmark(
        gen, method="stub", seed=1, budget=5, output_dir=tmp_path, log_every=0
    )

    assert summary["stopped_on"] == "budget"
    assert summary["oracle_calls"] == 6
    assert summary["steps"] == 3
    assert summary["attempts"] == 9
    assert summary["invalid_attempts"] == 3
    assert summary["duplicate_attempts"] == 0
    assert summary["unique_scored"] == 6
    assert summary["best_p_active"] == pytest.approx(0.7)


def test_oracle_calls_equal_distinct_scored(tmp_path):
    specs = [[f"A{i}", f"B{i}", None] for i in range(6)]
    gen = _StubGenerator(specs)
    summary = run_benchmark(
        gen, method="stub", seed=1, budget=6, output_dir=tmp_path, log_every=0
    )

    rows = _read_csv(tmp_path / "all_evaluations.csv")
    charged = [r for r in rows if r["oracle_called"] == "True"]
    canonical = {r["canonical_smiles"] for r in charged}

    assert summary["oracle_calls"] == len(charged) == len(canonical)
    assert [r["attempt"] for r in rows] == [str(i) for i in range(len(rows))]


def test_duplicate_classification_and_streamed_csv(tmp_path):
    specs = [["A", "B"], ["A", "C"]]
    gen = _StubGenerator(specs)
    summary = run_benchmark(
        gen, method="stub", seed=1, budget=3, output_dir=tmp_path, log_every=0
    )

    assert summary["duplicate_attempts"] == 1
    assert summary["oracle_calls"] == 3

    rows = _read_csv(tmp_path / "all_evaluations.csv")
    assert list(rows[0].keys()) == ATTEMPT_COLUMNS
    by_canonical = {r["canonical_smiles"]: r for r in rows}
    assert by_canonical["A"]["duplicate"] == "True"
    assert by_canonical["A"]["oracle_called"] == "False"
    assert by_canonical["A"]["p_active"] == "nan"

    curve = _read_csv(tmp_path / "curve.csv")
    assert list(curve[0].keys()) == CURVE_COLUMNS
    assert [int(r["oracle_calls"]) for r in curve] == [2, 3]


def test_accounting_mismatch_fails_loudly(tmp_path):
    class _Bad(_StubGenerator):
        def step(self) -> EvalResult:
            result = super().step()
            result.n_oracle_calls = 99
            return result

    gen = _Bad([["A", "B"]])
    with pytest.raises(AssertionError):
        run_benchmark(
            gen, method="stub", seed=1, budget=2, output_dir=tmp_path, log_every=0
        )


def test_max_steps_guard(tmp_path):
    specs = [["A"]]  # every later step is a duplicate -> no new oracle calls
    gen = _StubGenerator(specs)
    summary = run_benchmark(
        gen,
        method="stub",
        seed=1,
        budget=10_000,
        output_dir=tmp_path,
        max_steps=3,
        log_every=0,
    )

    assert summary["stopped_on"] == "max_steps"
    assert summary["steps"] == 3
    assert summary["oracle_calls"] == 1
    assert (tmp_path / "summary.json").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["method"] == "stub"
