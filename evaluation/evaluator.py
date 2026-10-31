"""Scoring orchestration: decode latent vectors and evaluate candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from chemistry.smiles import parse_and_canonical_batch
from logs import suppress_joblib_warnings

suppress_joblib_warnings()

INVALID_MOLECULE_OBJECTIVE: float = -1.0
"""Objective value assigned to invalid or duplicate candidates.

Valid first-seen molecules receive ``P(active)`` in ``[0, 1]``. Assigning
``-1.0`` to chemically invalid or already-seen canonical SMILES ranks them
strictly below all novel valid molecules without consuming oracle calls.
"""


@dataclass
class EvalTimings:
    """Per-step timing breakdown in seconds."""

    decode: float = 0.0
    validity: float = 0.0
    featurize_predict: float = 0.0
    assemble: float = 0.0
    solver_ops: float = 0.0


@dataclass
class EvalResult:
    """Result of scoring a single batch of candidate molecules."""

    smiles: list[str]
    objectives: np.ndarray
    p_active: np.ndarray
    n_valid: int
    gen_time: float
    timings: EvalTimings = field(default_factory=EvalTimings)
    n_duplicates: int = 0
    n_oracle_calls: int = 0
    """First-seen chemically valid canonical SMILES sent to TabPFN."""
    canonical: list[str | None] | None = None
    """Canonical SMILES per attempt; ``None`` for chemically invalid ones."""
    oracle_mask: np.ndarray | None = None
    """Boolean mask of attempts that were first-seen valid and sent to the oracle."""


class Evaluator:
    """Decode latent vectors and score novel valid SMILES with TabPFN.

    Parameters
    ----------
    decode : callable
        Function ``(n, latent_dim) -> list[str]`` that decodes latent
        vectors to SMILES.
    activity : callable
        Function ``(X, model) -> (preds, probs)`` that predicts from
        pre-featurized features (``predict_from_features``).
    activity_model : fitted TabPFNClassifier
        Activity predictor.
    featurizer : MinimolFeaturizer
        Feature transform matching the TabPFN training config.
    seen_smiles : set of str or None, default=None
        Optional initial deduplication set (e.g. restored from a checkpoint).
    """

    def __init__(
        self,
        decode,
        activity,
        activity_model,
        featurizer,
        seen_smiles: set[str] | None = None,
    ) -> None:
        self._decode_fn = decode
        self._activity_fn = activity
        self._activity_model = activity_model
        self._featurizer = featurizer
        self._seen_smiles: set[str] = set(seen_smiles) if seen_smiles else set()

    @property
    def decode_fn(self):
        """Return the latent-vector decoder used by this evaluator."""
        return self._decode_fn

    @property
    def seen_smiles(self) -> set[str]:
        """Return the set of canonical SMILES already scored by the oracle."""
        return self._seen_smiles

    def __call__(self, z: np.ndarray) -> EvalResult:
        """Decode latent batch *z* to SMILES and evaluate.

        Parameters
        ----------
        z : np.ndarray of shape ``(n, latent_dim)``
            Latent vectors to evaluate.

        Returns
        -------
        EvalResult
            Objectives, ``P(active)`` scores, canonical SMILES, and counters.
        """
        t1 = time.time()
        smiles_list = self._decode_fn(z)
        decode_time = time.time() - t1

        result = self.evaluate_smiles(smiles_list)
        result.timings.decode = decode_time
        result.gen_time += decode_time
        return result

    def evaluate_smiles(self, smiles_list: list[str]) -> EvalResult:
        """Score a batch of SMILES strings.

        Duplicate molecules (same canonical SMILES seen before) are skipped —
        they receive ``INVALID_MOLECULE_OBJECTIVE`` (``-1.0``) and zero
        ``p_active``, avoiding redundant featurization and oracle calls.

        Parameters
        ----------
        smiles_list : list of str
            Candidate SMILES from any generator.

        Returns
        -------
        EvalResult
            Objectives, ``P(active)`` scores, canonical SMILES, and counters.
        """
        timings = EvalTimings()
        t0 = time.time()
        n = len(smiles_list)

        t1 = time.time()
        valid_mask, canonical = parse_and_canonical_batch(smiles_list)
        timings.validity = time.time() - t1

        new_mask = _dedup_mask(canonical, valid_mask, self._seen_smiles)
        scored_positions = np.flatnonzero(new_mask)
        valid_smiles = [smiles_list[i] for i in scored_positions]

        pa = self._score_valid(valid_smiles, timings)

        t1 = time.time()
        result = self._assemble(n, valid_mask, new_mask, smiles_list, canonical, pa)
        timings.assemble = time.time() - t1

        result.gen_time = time.time() - t0
        result.timings = timings
        return result

    def _score_valid(
        self, valid_smiles: list[str], timings: EvalTimings
    ) -> np.ndarray | None:
        """Featurize and score first-seen valid SMILES."""
        if not valid_smiles:
            return None

        t1 = time.time()
        X = self._featurizer.transform(valid_smiles)
        _, probs = self._activity_fn(X, self._activity_model)
        pa = probs[:, 1].astype(np.float64)
        timings.featurize_predict = time.time() - t1
        return pa

    def _assemble(
        self,
        n: int,
        valid_mask: np.ndarray,
        new_mask: np.ndarray,
        smiles_list: list[str],
        canonical: list[str | None],
        pa: np.ndarray | None,
    ) -> EvalResult:
        """Map per-molecule scores back to the full candidate array."""
        objectives = np.full(n, INVALID_MOLECULE_OBJECTIVE, dtype=np.float64)
        p_active = np.zeros(n, dtype=np.float64)

        scored_positions = np.flatnonzero(new_mask)
        if pa is not None and len(scored_positions) > 0:
            finite = np.isfinite(pa)
            good_idx = scored_positions[finite]
            objectives[good_idx] = pa[finite]
            p_active[good_idx] = pa[finite]

        n_duplicates = int((valid_mask & ~new_mask).sum())

        return EvalResult(
            smiles=smiles_list,
            objectives=objectives,
            p_active=p_active,
            n_valid=int(valid_mask.sum()),
            gen_time=0.0,
            n_duplicates=n_duplicates,
            n_oracle_calls=int(new_mask.sum()),
            canonical=list(canonical),
            oracle_mask=new_mask.copy(),
        )


def _dedup_mask(
    canonical: list[str | None],
    valid_mask: np.ndarray,
    seen: set[str],
) -> np.ndarray:
    """Return a boolean mask of first-seen valid canonical SMILES.

    Invalid molecules (``canonical[i] is None``) always return False.
    First occurrences of a valid canonical SMILES are True and added
    to *seen*; later occurrences return False.
    """
    n = len(canonical)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if valid_mask[i] and canonical[i] is not None:
            if canonical[i] not in seen:
                mask[i] = True
                seen.add(canonical[i])
    return mask
