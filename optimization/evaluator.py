"""Scoring orchestration: decode latent vectors and evaluate candidates."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import selfies as sf
from joblib import Parallel, delayed
from rdkit import Chem

from optimization.constants import INVALID_MOLECULE_OBJECTIVE
from scoring.br_sascore import compute_br_sascore_batch
from scoring.ad_scorer import ADScorer
from scoring.archive_novelty import ArchiveNoveltyScorer
from scoring.physchem import PhysChemScorer

MAX_SELFIES_TOKENS = 200

if TYPE_CHECKING:
    from config import ArchiveConfig
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
    ad: np.ndarray
    archive_novelty: np.ndarray
    logp: np.ndarray
    tpsa: np.ndarray
    n_valid: int
    gen_time: float


class Evaluator:
    """Decode latent vectors and score them on all objectives.

    Parameters
    ----------
    decode : callable
        Function ``(n, latent_dim) -> list[str]`` that decodes latent
        vectors to SMILES.
    ad : ADScorer
        Applicability-domain scorer (distance to training set).
    archive_novelty : ArchiveNoveltyScorer
        Structural diversity scorer (distance to archive members).
    physchem : PhysChemScorer
        Physicochemical property scorer (LogP, TPSA).
    activity : callable
        Function ``(smiles, model, featurizer) -> (preds, probs)``.
    activity_model : fitted TabPFNClassifier
        Activity predictor.
    featurizer : MoleculeFeaturizer
        Feature transform matching the TabPFN training config.
    archive_cfg : ArchiveConfig
        Archive dimension configuration (names, enabled flags).
    """

    def __init__(
        self,
        decode,
        ad: ADScorer,
        archive_novelty: ArchiveNoveltyScorer,
        physchem: PhysChemScorer,
        activity,
        activity_model,
        featurizer,
        archive_cfg: ArchiveConfig,
    ) -> None:
        self._decode_fn = decode
        self._ad = ad
        self._archive_novelty = archive_novelty
        self._physchem = physchem
        self._activity_fn = activity
        self._activity_model = activity_model
        self._featurizer = featurizer
        self._archive_cfg = archive_cfg
        self._enabled_indices = [
            i for i, e in enumerate(archive_cfg.dimension_enabled) if e
        ]

    @property
    def decode_fn(self):
        """Return the latent-vector decoder used by this evaluator."""
        return self._decode_fn

    @property
    def archive_novelty(self) -> ArchiveNoveltyScorer:
        """Return the novelty scorer used by this evaluator."""
        return self._archive_novelty

    def __call__(self, z: np.ndarray) -> EvalResult:
        """Score a batch of latent vectors.

        Parameters
        ----------
        z : np.ndarray of shape ``(n, latent_dim)``
            Latent vectors to evaluate.

        Returns
        -------
        EvalResult
            Objectives (P(active)), behavior coords, and metadata.
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
        """Score valid SMILES with all scorers.

        CPU-bound scorers (BR-SAScore, AD, PhysChem) run concurrently
        with GPU-bound TabPFN via ThreadPoolExecutor.
        """
        if not valid_smiles:
            return None

        with ThreadPoolExecutor(max_workers=3) as pool:
            br_future = pool.submit(compute_br_sascore_batch, valid_smiles)
            ad_future = pool.submit(self._ad, valid_smiles)
            nov_future = pool.submit(self._archive_novelty, valid_smiles)
            phys_future = pool.submit(self._physchem, valid_smiles)
            _, probs = self._activity_fn(
                valid_smiles, self._activity_model, self._featurizer
            )

            br = br_future.result()
            ad = ad_future.result()
            nov = nov_future.result()
            phys = phys_future.result()

        pa = probs[:, 1]

        return _ScoreBundle(
            br=br, ad=ad, nov=nov, logp=phys.logp, tpsa=phys.tpsa, pa=pa
        )

    def _assemble(
        self,
        n: int,
        valid_mask: np.ndarray,
        smiles_list: list[str],
        scores: _ScoreBundle | None,
    ) -> EvalResult:
        """Map per-molecule scores back to the full candidate array."""
        objectives = np.full(n, INVALID_MOLECULE_OBJECTIVE, dtype=np.float64)
        n_active = len(self._enabled_indices)
        measures = np.zeros((n, n_active), dtype=np.float64)
        p_active = np.zeros(n, dtype=np.float64)
        br_arr = np.full(n, np.nan, dtype=np.float64)
        ad_arr = np.zeros(n, dtype=np.float64)
        nov_arr = np.ones(n, dtype=np.float64)
        logp_arr = np.full(n, np.nan, dtype=np.float64)
        tpsa_arr = np.full(n, np.nan, dtype=np.float64)

        if scores is None:
            return EvalResult(
                smiles=smiles_list,
                objectives=objectives,
                measures=measures,
                p_active=p_active,
                br_sascore=br_arr,
                ad=ad_arr,
                archive_novelty=nov_arr,
                logp=logp_arr,
                tpsa=tpsa_arr,
                n_valid=int(valid_mask.sum()),
                gen_time=0.0,
            )

        all_scores = {
            "br_sascore": scores.br,
            "ad": scores.ad,
            "novelty": scores.nov,
            "logp": scores.logp,
            "tpsa": scores.tpsa,
        }

        j = 0
        for i in range(n):
            if not valid_mask[i]:
                continue
            if np.isnan(scores.br[j]):
                j += 1
                continue

            objectives[i] = scores.pa[j]
            for k, dim_idx in enumerate(self._enabled_indices):
                dim_name = self._archive_cfg.dimension_names[dim_idx]
                measures[i, k] = all_scores[dim_name][j]
            p_active[i] = scores.pa[j]
            br_arr[i] = scores.br[j]
            ad_arr[i] = scores.ad[j]
            nov_arr[i] = scores.nov[j]
            logp_arr[i] = scores.logp[j]
            tpsa_arr[i] = scores.tpsa[j]
            j += 1

        return EvalResult(
            smiles=smiles_list,
            objectives=objectives,
            measures=measures,
            p_active=p_active,
            br_sascore=br_arr,
            ad=ad_arr,
            archive_novelty=nov_arr,
            logp=logp_arr,
            tpsa=tpsa_arr,
            n_valid=int(valid_mask.sum()),
            gen_time=0.0,
        )


@dataclass
class _ScoreBundle:
    """Container for per-molecule scores from all scorers."""

    br: np.ndarray
    ad: np.ndarray
    nov: np.ndarray
    logp: np.ndarray
    tpsa: np.ndarray
    pa: np.ndarray


def _check_valid(smi: str) -> bool:
    """Check if a single SMILES string is chemically valid.

    Rejects empty strings, molecules that fail RDKit parsing, and
    molecules whose SELFIES token count exceeds *MAX_SELFIES_TOKENS*
    (catches pathological repeating chains from extreme latent vectors).
    """
    if not smi:
        return False
    if Chem.MolFromSmiles(smi) is None:
        return False
    try:
        selfies = sf.encoder(smi)
        tokens = list(sf.split_selfies(selfies))
        if len(tokens) > MAX_SELFIES_TOKENS:
            return False
    except Exception:
        return False
    return True


def _validity_mask(smiles: list[str], n_jobs: int = -1) -> np.ndarray:
    """Return a boolean mask of chemically valid SMILES strings."""
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_check_valid)(s) for s in smiles
    )
    return np.array(results, dtype=bool)
