"""Tests for evaluator deduplication helpers."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from chemistry.smiles import parse_and_canonical
from evaluation.evaluator import _dedup_mask


class TestParseAndCanonical:
    def test_valid_smiles(self):
        valid, canon = parse_and_canonical("CCO")
        assert valid is True
        assert canon == "CCO"

    def test_invalid_smiles(self):
        valid, canon = parse_and_canonical("INVALID")
        assert valid is False
        assert canon is None

    def test_empty_string(self):
        valid, canon = parse_and_canonical("")
        assert valid is False
        assert canon is None

    def test_canonicalization(self):
        valid, canon = parse_and_canonical("C(O)C")
        assert valid is True
        assert canon == "CCO"

    def test_long_selfies_rejected(self):
        # SELFIES chain longer than MAX_SELFIES_TOKENS (200)
        long_smiles = "C" * 250
        valid, canon = parse_and_canonical(long_smiles)
        assert valid is False
        assert canon is None

    def test_canonicalization_invariant_violation_rejected(self, monkeypatch):
        # RDKit MolToSmiles can throw an C++ invariant violation on
        # degenerate molecules that MolFromSmiles still accepts. It must be
        # treated as invalid, never propagated out of the worker.
        def _raise(*args, **kwargs):
            raise RuntimeError("Invariant Violation: could not find atom2")

        monkeypatch.setattr(Chem, "MolToSmiles", _raise)
        valid, canon = parse_and_canonical("CCO")
        assert valid is False
        assert canon is None


class TestCanonicalizeSmilesGuard:
    def test_invariant_violation_returns_none(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise RuntimeError("Invariant Violation: could not find atom2")

        monkeypatch.setattr(Chem, "MolToSmiles", _raise)
        from chemistry.smiles import canonicalize_smiles

        assert canonicalize_smiles("CCO") is None

    def test_valid_still_canonicalizes(self):
        from chemistry.smiles import canonicalize_smiles

        assert canonicalize_smiles("C(O)C") == "CCO"


class TestDedupMask:
    def test_first_occurrence_true(self):
        canonical = ["CCO", "CCN"]
        valid_mask = np.array([True, True])
        seen: set[str] = set()
        mask = _dedup_mask(canonical, valid_mask, seen)
        assert mask.tolist() == [True, True]
        assert seen == {"CCO", "CCN"}

    def test_duplicate_false(self):
        canonical = ["CCO", "CCO"]
        valid_mask = np.array([True, True])
        seen: set[str] = set()
        mask = _dedup_mask(canonical, valid_mask, seen)
        assert mask.tolist() == [True, False]

    def test_invalid_always_false(self):
        canonical = ["CCO", None, "CCN"]
        valid_mask = np.array([True, True, True])
        seen: set[str] = set()
        mask = _dedup_mask(canonical, valid_mask, seen)
        assert mask.tolist() == [True, False, True]

    def test_invalid_not_added_to_seen(self):
        canonical = [None, "CCO"]
        valid_mask = np.array([True, True])
        seen: set[str] = set()
        _dedup_mask(canonical, valid_mask, seen)
        assert seen == {"CCO"}

    def test_seen_set_persists_across_calls(self):
        canonical = ["CCO", "CCN", "CCO"]
        valid_mask = np.array([True, True, True])
        seen: set[str] = set()
        mask = _dedup_mask(canonical, valid_mask, seen)
        assert mask.tolist() == [True, True, False]
        assert seen == {"CCO", "CCN"}

    def test_empty_input(self):
        mask = _dedup_mask([], np.array([], dtype=bool), set())
        assert len(mask) == 0
