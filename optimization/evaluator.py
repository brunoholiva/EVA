"""Scoring orchestration: decode latent vectors and evaluate candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from optimization.constants import INVALID_MOLECULE_OBJECTIVE
from scoring.br_sascore import compute_br_sascore

if TYPE_CHECKING:
    from featurization.features import MoleculeFeaturizer


class MolecularScorer(Protocol):
    """Anything that scores a list of SMILES and returns a numeric array."""

    def __call__(self, smiles: list[str]) -> np.ndarray: ...


@dataclass
class EvalResult:
    """Result of scoring a single generation of candidates."""

    smiles: list[str]
    objectives: np.ndarray
    measures: np.ndarray
    p_active: np.ndarray
    br_sascore: np.ndarray
    novelty: np.ndarray
    n_valid: int
    gen_time: float


class Evaluator:
    """Decode latent vectors and score them on all objectives.

    Parameters
    ----------
    decode : callable
        Function ``(n, latent_dim) -> list[str]`` that decodes latent
        vectors to SMILES.
    novelty : MolecularScorer
        Applicability-domain scorer.
    activity : callable
        Function ``(smiles, model, featurizer) -> (preds, probs)``.
    activity_model : fitted TabPFNClassifier
        Activity predictor.
    featurizer : MoleculeFeaturizer
        Feature transform matching the TabPFN training config.
    """

    def __init__(
        self,
        decode,
        novelty: MolecularScorer,
        activity,
        activity_model,
        featurizer: MoleculeFeaturizer,
    ) -> None:
        self._decode_fn = decode
        self._novelty = novelty
        self._activity_fn = activity
        self._activity_model = activity_model
        self._featurizer = featurizer

    def __call__(self, z: np.ndarray) -> EvalResult:
        """Score a batch of latent vectors.

        Parameters
        ----------
        z : np.ndarray of shape ``(n, latent_dim)``
            Latent vectors to evaluate.

        Returns
        -------
        EvalResult
            Objectives (P(active)), behavior coords
            (BR-SAScore, novelty), and metadata.
        """
        t0 = time.time()
        n = len(z)

        smiles_list = self._decode_fn(z)
        valid_mask = _validity_mask(smiles_list)
        valid_smiles = [s for s, v in zip(smiles_list, valid_mask) if v]

        scores = self._score_valid(valid_smiles)
        result = self._assemble(n, valid_mask, smiles_list, scores)

        result.gen_time = time.time() - t0
        return result

    def _score_valid(self, valid_smiles: list[str]) -> _ScoreBundle | None:
        """Score valid SMILES with all three scorers."""
        if not valid_smiles:
            return None

        br = np.array([compute_br_sascore(s) for s in valid_smiles], dtype=np.float64)
        nov = self._novelty(valid_smiles)
        _, probs = self._activity_fn(
            valid_smiles, self._activity_model, self._featurizer
        )
        pa = probs[:, 1]

        return _ScoreBundle(br=br, nov=nov, pa=pa)

    def _assemble(
        self,
        n: int,
        valid_mask: np.ndarray,
        smiles_list: list[str],
        scores: _ScoreBundle | None,
    ) -> EvalResult:
        """Map per-molecule scores back to the full candidate array."""
        objectives = np.full(n, INVALID_MOLECULE_OBJECTIVE, dtype=np.float64)
        measures = np.zeros((n, 2), dtype=np.float64)
        p_active = np.zeros(n, dtype=np.float64)
        br_arr = np.full(n, np.nan, dtype=np.float64)
        nov_arr = np.zeros(n, dtype=np.float64)

        if scores is None:
            return EvalResult(
                smiles=smiles_list,
                objectives=objectives,
                measures=measures,
                p_active=p_active,
                br_sascore=br_arr,
                novelty=nov_arr,
                n_valid=int(valid_mask.sum()),
                gen_time=0.0,
            )

        j = 0
        for i in range(n):
            if not valid_mask[i]:
                continue
            if np.isnan(scores.br[j]):
                j += 1
                continue

            objectives[i] = scores.pa[j]
            measures[i] = [scores.br[j], scores.nov[j]]
            p_active[i] = scores.pa[j]
            br_arr[i] = scores.br[j]
            nov_arr[i] = scores.nov[j]
            j += 1

        return EvalResult(
            smiles=smiles_list,
            objectives=objectives,
            measures=measures,
            p_active=p_active,
            br_sascore=br_arr,
            novelty=nov_arr,
            n_valid=int(valid_mask.sum()),
            gen_time=0.0,
        )


@dataclass
class _ScoreBundle:
    """Container for per-molecule scores from all three scorers."""

    br: np.ndarray
    nov: np.ndarray
    pa: np.ndarray


def _validity_mask(smiles: list[str]) -> np.ndarray:
    """Return a boolean mask of which SMILES are non-empty strings."""
    return np.array([s != "" for s in smiles], dtype=bool)
