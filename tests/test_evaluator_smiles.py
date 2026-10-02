"""Tests for Evaluator.evaluate_smiles and oracle-call accounting."""

from __future__ import annotations

import numpy as np

from config import ArchiveConfig
from evaluation.evaluator import Evaluator


class _Featurizer:
    """Identity featurizer: one feature per molecule, so shapes stay simple."""

    def transform(self, smiles):
        return np.arange(len(smiles), dtype=np.float64).reshape(-1, 1)


class _Activity:
    """Deterministic predictor stub keyed off the feature value."""

    def __init__(self):
        self.calls = 0

    def __call__(self, X, model):
        self.calls += 1
        p = np.clip(0.1 + 0.05 * X[:, 0], 0.0, 1.0)
        return (p > 0.5).astype(int), np.stack([1 - p, p], axis=1)


def _evaluator(seed_decode=None, dimensions=None):
    cfg = ArchiveConfig(
        solution_dim=2,
        dimensions=dimensions or [],
        learning_rate=0.01,
        threshold_min=0.1,
    )
    activity = _Activity()
    ev = Evaluator(
        decode=seed_decode or (lambda z: []),
        activity=activity,
        activity_model=None,
        featurizer=_Featurizer(),
        archive_cfg=cfg,
    )
    return ev, activity


class TestOracleAccounting:
    def test_counts_only_first_seen_valid(self):
        ev, activity = _evaluator()
        r = ev.evaluate_smiles(["CCO", "CCO", "INVALID", "CCN"])
        assert r.n_oracle_calls == 2
        assert activity.calls == 1

    def test_duplicates_are_valid_but_not_scored(self):
        ev, _ = _evaluator()
        r = ev.evaluate_smiles(["CCO", "CCO"])
        assert r.n_valid == 2
        assert r.n_duplicates == 1
        assert r.n_oracle_calls == 1

    def test_invalid_is_not_valid_and_not_scored(self):
        ev, _ = _evaluator()
        r = ev.evaluate_smiles(["not-a-molecule"])
        assert r.n_valid == 0
        assert r.n_duplicates == 0
        assert r.n_oracle_calls == 0

    def test_seen_set_persists_across_calls(self):
        ev, _ = _evaluator()
        first = ev.evaluate_smiles(["CCO"])
        second = ev.evaluate_smiles(["CCO"])
        assert first.n_oracle_calls == 1
        assert second.n_oracle_calls == 0
        assert second.n_duplicates == 1

    def test_oracle_calls_never_exceed_attempts(self):
        ev, _ = _evaluator()
        r = ev.evaluate_smiles(["CCO", "CCN", "CCC", "INVALID", "CCO"])
        assert r.n_oracle_calls <= len(r.smiles)


class TestCanonicalExposure:
    def test_canonical_is_populated(self):
        ev, _ = _evaluator()
        r = ev.evaluate_smiles(["OCC"])
        assert r.canonical == ["CCO"]

    def test_invalid_canonical_is_none(self):
        ev, _ = _evaluator()
        r = ev.evaluate_smiles(["INVALID"])
        assert r.canonical == [None]

    def test_canonical_covers_every_attempt(self):
        ev, _ = _evaluator()
        r = ev.evaluate_smiles(["CCO", "INVALID", "CCN"])
        assert len(r.canonical) == 3

    def test_canonical_is_a_copy_not_shared_state(self):
        ev, _ = _evaluator()
        r = ev.evaluate_smiles(["CCO"])
        r.canonical[0] = None
        r2 = ev.evaluate_smiles(["CCN", "CCO"])
        assert r2.canonical == ["CCN", "CCO"]


class TestScorePlacement:
    def test_scored_positions_get_p_active(self):
        ev, _ = _evaluator()
        r = ev.evaluate_smiles(["CCO", "CCN"])
        assert r.p_active.shape == (2,)
        assert np.all(r.p_active > 0)

    def test_unscored_positions_are_zero(self):
        ev, _ = _evaluator()
        r = ev.evaluate_smiles(["CCO", "INVALID"])
        assert r.p_active[1] == 0.0

    def test_objectives_aligned_to_attempts(self):
        ev, _ = _evaluator()
        r = ev.evaluate_smiles(["CCO", "INVALID", "CCN"])
        assert r.objectives.shape == (3,)
        assert r.objectives[1] == -1.0


class TestLatentWrapper:
    def test_call_decodes_then_scores(self):
        ev, _ = _evaluator(seed_decode=lambda z: ["CCO"] * len(z))
        r = ev(z=np.zeros((3, 2)))
        assert r.n_oracle_calls == 1
        assert r.n_duplicates == 2

    def test_gen_time_includes_decode(self):
        def slow_decode(z):
            import time

            time.sleep(0.05)
            return ["CCO"] * len(z)

        ev, _ = _evaluator(seed_decode=slow_decode)
        r = ev(z=np.zeros((1, 2)))
        assert r.gen_time >= 0.05
        assert r.timings.decode >= 0.05

    def test_mixing_call_and_evaluate_shares_dedup(self):
        ev, _ = _evaluator(seed_decode=lambda z: ["CCO"] * len(z))
        first = ev(z=np.zeros((1, 2)))
        second = ev.evaluate_smiles(["CCO"])
        assert first.n_oracle_calls == 1
        assert second.n_oracle_calls == 0
