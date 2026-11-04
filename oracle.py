"""MiniMol featurization, TabPFN activity scoring, deduplication, and scaffolds."""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Callable, Iterable
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric
from rdkit.rdBase import DisableLog

if TYPE_CHECKING:
    from tabpfn import TabPFNClassifier

DisableLog("rdApp.*")

INVALID_OBJECTIVE: float = -1.0
"""Maximization objective assigned to invalid or duplicate molecules."""

N_MINIMOL_FEATURES: int = 512

CIRCLE_DISTANCE_THRESHOLD: float = 0.75
"""Default Tanimoto distance cutoff ``t`` for ``#Circles`` (Xie et al., ICLR 2023)."""

DIVERSE_TOP_K: int = 10
"""Default circle count ``K`` for ``Diverse AUC Top-K``."""

_MORGAN_GEN = None


def _morgan_generator():
    """Lazily initialize ECFP4 (radius=2, 2048-bit) Morgan generator."""
    global _MORGAN_GEN
    if _MORGAN_GEN is None:
        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return _MORGAN_GEN


@dataclass(frozen=True)
class CircleMetrics:
    """Diversity and sample-efficiency metrics under ``#Circles`` sphere exclusion."""

    n_circles: int = 0
    diverse_top_k: float = 0.0
    diverse_auc_top_k: float = 0.0


def _greedy_circles(
    ordered: Iterable[tuple[float, int, Any]],
    max_sim: float,
    limit: int | None = None,
) -> list[float]:
    """Return scores of greedily selected circle centers from score-sorted *ordered*."""
    centers: list[Any] = []
    scores: list[float] = []
    for pa, _, fp in ordered:
        if (
            not centers
            or max(DataStructs.BulkTanimotoSimilarity(fp, centers)) < max_sim
        ):
            centers.append(fp)
            scores.append(pa)
            if limit is not None and len(centers) >= limit:
                break
    return scores


def circle_metrics(
    hits: Iterable[tuple[str, float, int]],
    oracle_budget: int,
    distance_threshold: float = CIRCLE_DISTANCE_THRESHOLD,
    top_k: int = DIVERSE_TOP_K,
    n_checkpoints: int = 100,
) -> CircleMetrics:
    """Compute ``#Circles(t)`` and ``Diverse AUC Top-K(t)`` up to *oracle_budget*.

    Parameters
    ----------
    hits : iterable of ``(smiles, p_active, oracle_call)``
        Active molecules (e.g. ``P(active) >= 0.6``) and their discovery call index.
    oracle_budget : int
        Total oracle-call budget ``B``; hits with ``oracle_call > B`` are excluded.
    distance_threshold : float
        Minimum pairwise Tanimoto distance ``d = 1 - sim > t`` between circle centers.
    top_k : int
        Number of top circle centers averaged for ``diverse_top_k`` and its AUC.
    n_checkpoints : int
        Number of evenly spaced budget intervals for trapezoidal AUC integration.
    """
    if oracle_budget <= 0 or top_k <= 0:
        return CircleMetrics()

    max_sim = 1.0 - float(distance_threshold)
    gen = _morgan_generator()
    items: list[tuple[float, int, Any]] = []
    for smi, pa, call in hits:
        if int(call) > oracle_budget:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        items.append((float(pa), int(call), gen.GetFingerprint(mol)))

    if not items:
        return CircleMetrics()

    items.sort(key=lambda x: (-x[0], x[1]))
    n_circles = len(_greedy_circles(items, max_sim))
    final_scores = _greedy_circles(items, max_sim, limit=top_k)
    diverse_top_k = float(sum(final_scores) / top_k)

    grid = np.linspace(0.0, float(oracle_budget), n_checkpoints + 1)
    curve = np.empty(len(grid), dtype=np.float64)
    for idx, c in enumerate(grid):
        sub_scores = _greedy_circles(
            (it for it in items if it[1] <= c), max_sim, limit=top_k
        )
        curve[idx] = sum(sub_scores) / top_k
    auc = float(np.trapz(curve, grid) / oracle_budget)

    return CircleMetrics(
        n_circles=n_circles,
        diverse_top_k=diverse_top_k,
        diverse_auc_top_k=auc,
    )


def canonicalize_smiles(smiles: str) -> str | None:
    """Return round-trip-valid canonical SMILES for *smiles*, or ``None``."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        canonical = Chem.MolToSmiles(mol)
        if not canonical or Chem.MolFromSmiles(canonical) is None:
            return None
        return canonical
    except Exception:
        return None


def generic_scaffold(smiles: str) -> str:
    """Return canonical SMILES of the generic Murcko scaffold.

    Acyclic molecules return ``"acyclic"``; parse or valence failures return
    ``"invalid"``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    try:
        scaffold = GetScaffoldForMol(mol)
    except Exception:
        return "invalid"
    if not scaffold or not scaffold.GetRingInfo().NumRings():
        return "acyclic"
    generic = Chem.MolToSmiles(scaffold)
    try:
        generic = Chem.MolToSmiles(MakeScaffoldGeneric(scaffold))
    except Exception:
        pass
    return generic


def _ensure_pandas_tools_stub() -> None:
    """Stub ``rdkit.Chem.PandasTools`` for rdkit 2023.9.6 + pandas 2.3 compatibility."""
    if "rdkit.Chem.PandasTools" in sys.modules:
        return
    stub = types.ModuleType("rdkit.Chem.PandasTools")
    sys.modules["rdkit.Chem.PandasTools"] = stub
    Chem.PandasTools = stub


class MinimolFeaturizer:
    """Compute 512D MiniMol fingerprints lazily via ``minimol-onnx``."""

    n_features = N_MINIMOL_FEATURES

    def __init__(self, batch_size: int = 512) -> None:
        self._model = None
        self.batch_size = batch_size

    def transform(self, smiles: list[str]) -> np.ndarray:
        """Featurize *smiles* into ``(n, 512)`` float32 features.

        Any rare molecule that fails graph construction receives ``NaN``s so a
        single degenerate decode cannot abort an entire search run.
        """
        if not smiles:
            return np.empty((0, self.n_features), dtype=np.float32)
        if self._model is None:
            _ensure_pandas_tools_stub()
            from minimol_onnx import MinimolONNX

            self._model = MinimolONNX(batch_size=self.batch_size)
        try:
            return self._model(smiles)
        except Exception:
            return self._transform_fallback(smiles)

    def _transform_fallback(self, smiles: list[str]) -> np.ndarray:
        """Featurize molecule-by-molecule when a batch contains a broken graph."""
        assert self._model is not None
        out = np.full((len(smiles), self.n_features), np.nan, dtype=np.float32)
        good_idx: list[int] = []
        datas = []
        for i, s in enumerate(smiles):
            try:
                datas.append(self._model.featurize_one(s))
                good_idx.append(i)
            except Exception:
                continue
        for start in range(0, len(datas), self.batch_size):
            sub_idx = good_idx[start : start + self.batch_size]
            feeds = self._model.collate(datas[start : start + self.batch_size])
            out[sub_idx] = self._model.session.run(None, feeds)[0]
        return out


def move_model_to_device(model: TabPFNClassifier, device: str) -> TabPFNClassifier:
    """Transfer TabPFN parameters and KV-caches to *device*."""
    model.to(device)
    if hasattr(model, "executor_") and hasattr(model.executor_, "kv_caches"):
        model.executor_.kv_caches = [
            cache.to(device) for cache in model.executor_.kv_caches
        ]
    return model


def load_tabpfn(path: str | Path, device: str = "cuda") -> TabPFNClassifier:
    """Load a joblib-persisted TabPFNClassifier onto *device*."""
    model: TabPFNClassifier = joblib.load(path)
    model.inference_precision = "autocast"
    if device != "cpu":
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in conf:
            extra = "expandable_segments:True"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"{conf},{extra}" if conf else extra
        move_model_to_device(model, device)
    return model


@contextmanager
def _quiet_stderr():
    """Silence TabPFN's internal tqdm output on stderr."""
    with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
        yield


def predict_p_active(X: np.ndarray, model: TabPFNClassifier) -> np.ndarray:
    """Return ``P(active)`` in ``[0, 1]`` for feature matrix *X*."""
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise ValueError(
            f"Feature dimension mismatch: got {X.shape[1]}, model expects {expected}"
        )
    with _quiet_stderr():
        probs = model.predict_proba(X)
    return np.asarray(probs[:, 1], dtype=np.float64)


@dataclass
class BatchScore:
    """Evaluation output for one batch of decoded canonical SMILES."""

    canonical: list[str | None]
    objectives: np.ndarray
    p_active: np.ndarray
    oracle_mask: np.ndarray
    n_valid: int
    n_duplicates: int
    n_oracle_calls: int


class Oracle:
    """Deduplicating molecular scorer backed by MiniMol + TabPFN.

    First-seen valid canonical SMILES are featurized and scored once. Invalid
    (``None``) and duplicate canonical SMILES receive ``INVALID_OBJECTIVE``
    (``-1.0``) and consume zero oracle calls.

    Parameters
    ----------
    featurizer : MinimolFeaturizer or similar
        Transforms ``list[str]`` into a 2D feature array.
    predict_fn : callable
        Maps feature matrix ``X`` to 1D ``P(active)`` probabilities.
    """

    def __init__(
        self,
        featurizer,
        predict_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self._featurizer = featurizer
        self._predict_fn = predict_fn
        self.seen: set[str] = set()

    @classmethod
    def from_pretrained(
        cls, predictor_path: str | Path, device: str = "cuda"
    ) -> Oracle:
        """Load ``MinimolFeaturizer`` and ``TabPFN`` from *predictor_path*."""
        model = load_tabpfn(predictor_path, device=device)
        featurizer = MinimolFeaturizer()
        return cls(featurizer, lambda X: predict_p_active(X, model))

    def score(self, canonical: list[str | None]) -> BatchScore:
        """Score a batch of canonical SMILES (``None`` for invalid decodes)."""
        out_canonical = list(canonical)
        n = len(out_canonical)
        valid_mask = np.zeros(n, dtype=bool)
        oracle_mask = np.zeros(n, dtype=bool)

        for i, smi in enumerate(out_canonical):
            if smi is not None:
                valid_mask[i] = True
                if smi not in self.seen:
                    self.seen.add(smi)
                    oracle_mask[i] = True

        objectives = np.full(n, INVALID_OBJECTIVE, dtype=np.float64)
        p_active = np.zeros(n, dtype=np.float64)

        idx = np.flatnonzero(oracle_mask)
        if len(idx) > 0:
            novel_smiles = [out_canonical[i] for i in idx]  # type: ignore[misc]
            X = self._featurizer.transform(novel_smiles)
            feat_ok = np.isfinite(X).all(axis=1)
            if not feat_ok.all():
                bad_smiles = {out_canonical[i] for i in idx[~feat_ok]}
                for s in bad_smiles:
                    if s is not None:
                        self.seen.discard(s)
                for j, smi in enumerate(out_canonical):
                    if smi in bad_smiles:
                        out_canonical[j] = None
                        valid_mask[j] = False
                        oracle_mask[j] = False
                idx = idx[feat_ok]
                X = X[feat_ok]

            if len(idx) > 0:
                pa = np.asarray(self._predict_fn(X), dtype=np.float64)
                finite = np.isfinite(pa)
                good = idx[finite]
                objectives[good] = pa[finite]
                p_active[good] = pa[finite]

        n_valid = int(valid_mask.sum())
        n_oracle = int(oracle_mask.sum())
        return BatchScore(
            canonical=out_canonical,
            objectives=objectives,
            p_active=p_active,
            oracle_mask=oracle_mask,
            n_valid=n_valid,
            n_duplicates=n_valid - n_oracle,
            n_oracle_calls=n_oracle,
        )
