"""Scoring orchestration: decode latent vectors and evaluate candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import selfies as sf
from joblib import Parallel, delayed
from rdkit import Chem

from optimization.constants import INVALID_MOLECULE_OBJECTIVE
from reporting.suppress import suppress_joblib_warnings

suppress_joblib_warnings()

MAX_SELFIES_TOKENS = 200

if TYPE_CHECKING:
    from config import ArchiveConfig
    from evaluation.applicability import ADScorer


@dataclass
class EvalTimings:
    """Per-generation timing breakdown."""

    decode: float = 0.0
    validity: float = 0.0
    featurize_predict: float = 0.0
    cpu_scorers: float = 0.0
    assemble: float = 0.0
    total: float = 0.0
    archive_ops: float = 0.0


@dataclass
class EvalResult:
    """Result of scoring a single generation of candidates."""

    smiles: list[str]
    objectives: np.ndarray
    measures: np.ndarray
    p_active: np.ndarray
    n_valid: int
    gen_time: float
    timings: EvalTimings = field(default_factory=EvalTimings)


class Evaluator:
    """Decode latent vectors and score them on all objectives.

    Parameters
    ----------
    decode : callable
        Function ``(n, latent_dim) -> list[str]`` that decodes latent
        vectors to SMILES.
    ad : ADScorer
        Applicability-domain scorer (distance to training set).
    activity : callable
        Function ``(X, model) -> (preds, probs)`` that predicts from
        pre-featurized features (``predict_from_features``).
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
        activity=None,
        activity_model=None,
        featurizer=None,
        archive_cfg: ArchiveConfig | None = None,
    ) -> None:
        self._decode_fn = decode
        self._ad = ad
        self._activity_fn = activity
        self._activity_model = activity_model
        self._featurizer = featurizer
        self._archive_cfg = archive_cfg
        self._enabled_indices = list(range(len(archive_cfg.dimensions)))
        enabled_names = archive_cfg.active_dimension_names()
        self._ad_enabled = "ad" in enabled_names

    @property
    def decode_fn(self):
        """Return the latent-vector decoder used by this evaluator."""
        return self._decode_fn

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
        timings = EvalTimings()
        t0 = time.time()
        n = len(z)

        t1 = time.time()
        smiles_list = self._decode_fn(z)
        timings.decode = time.time() - t1

        t1 = time.time()
        valid_mask = _validity_mask(smiles_list)
        timings.validity = time.time() - t1
        valid_smiles = [s for s, v in zip(smiles_list, valid_mask) if v]

        scores = self._score_valid(valid_smiles, timings)

        t1 = time.time()
        result = self._assemble(n, valid_mask, smiles_list, scores)
        timings.assemble = time.time() - t1

        result.gen_time = time.time() - t0
        result.timings = timings
        return result

    def _score_valid(
        self, valid_smiles: list[str], timings: EvalTimings
    ) -> _ScoreBundle | None:
        """Score valid SMILES with all scorers.

        Phase 1: Featurization (CPU-heavy, all cores).
        Phase 2: CPU scorers (MolBehavior + Tanimoto distances) run
            concurrently with GPU-bound TabPFN prediction.
        """
        if not valid_smiles:
            return None

        t1 = time.time()
        X = self._featurizer.transform(valid_smiles)
        timings.featurize_predict = time.time() - t1

        t1 = time.time()
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            cpu_future = pool.submit(self._compute_cpu_scores, valid_smiles)
            _, probs = self._activity_fn(X, self._activity_model)
            dim_scores = cpu_future.result()

        timings.cpu_scorers = time.time() - t1

        pa = probs[:, 1]
        return _ScoreBundle(dim_scores=dim_scores, pa=pa)

    def _compute_cpu_scores(self, valid_smiles: list[str]) -> dict[str, np.ndarray]:
        """Compute CPU-bound molecular scores for valid SMILES.

        Parses molecules once, then computes only the enabled dimensions.
        Fingerprints are lazily computed only if needed (e.g., for AD dimension).
        Scaffold-based dimensions use Murcko scaffold SMILES instead.

        Parameters
        ----------
        valid_smiles : list of str
            SMILES strings known to be chemically valid.

        Returns
        -------
        dict of str to np.ndarray
            Dictionary mapping dimension names to computed values.
        """
        from evaluation.dimensions import create_dimension
        from evaluation.molecules import ParsedMolecules

        parsed = ParsedMolecules(valid_smiles)
        dim_configs = {d.name: d for d in self._archive_cfg.dimensions}

        dim_scores = {}
        for dim_name in self._archive_cfg.active_dimension_names():
            use_scaffold = dim_configs[dim_name].scaffold

            if use_scaffold:
                scaffold_smiles = [
                    s if s is not None else smi
                    for s, smi in zip(parsed.scaffold_smiles, valid_smiles)
                ]
                compute_parsed = ParsedMolecules(scaffold_smiles)
            else:
                compute_parsed = parsed

            if dim_name == "ad":
                dimension = create_dimension(dim_name, ad_scorer=self._ad)
            else:
                dimension = create_dimension(dim_name)
            dim_scores[dim_name] = dimension.compute(compute_parsed)

        return dim_scores

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

        kept = np.empty(0, dtype=np.int64)
        if scores is not None:
            accept = np.ones(len(scores.pa), dtype=bool)
            kept = np.flatnonzero(valid_mask)[accept]

            objectives[kept] = scores.pa[accept]
            p_active[kept] = scores.pa[accept]

            for k, dim_idx in enumerate(self._enabled_indices):
                dim_name = self._archive_cfg.dimensions[dim_idx].name
                measures[kept, k] = scores.dim_scores[dim_name][accept]

            # pyribs requires finite measures. Some valid molecules (e.g. those
            # containing metals like Na/Li/Mg) make certain descriptors return
            # NaN (BCUT2D_* raise on Gasteiger charge failure). Treat them like
            # invalid molecules: rejected by the objective, finite sentinel so
            # the whole-batch validation in GridArchive.add() passes.
            non_finite = ~np.isfinite(measures[kept]).all(axis=1)
            if non_finite.any():
                bad_idx = kept[non_finite]
                objectives[bad_idx] = INVALID_MOLECULE_OBJECTIVE
                p_active[bad_idx] = 0.0
                measures[bad_idx] = 0.0

        return EvalResult(
            smiles=smiles_list,
            objectives=objectives,
            measures=measures,
            p_active=p_active,
            n_valid=int(valid_mask.sum()),
            gen_time=0.0,
        )


@dataclass
class _ScoreBundle:
    """Container for per-molecule scores from all scorers."""

    dim_scores: dict[str, np.ndarray]
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
