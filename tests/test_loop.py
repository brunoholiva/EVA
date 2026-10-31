"""Tests for SearchLoop, checkpoint resume, and EVA single/multi-seed orchestration."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from config import (
    CMAConfig,
    ExperimentConfig,
    OutputConfig,
    RunConfig,
    TensorBoardConfig,
)
from EVA import run_single_seed
from evaluation.evaluator import Evaluator
from optimization.candidates import CandidateStore
from optimization.cma import build_solver
from optimization.loop import ATTEMPT_COLUMNS, CURVE_COLUMNS, SearchLoop


class _StubVAE:
    """Deterministic decoder emitting unique alkane/ring SMILES + occasional duplicates/invalids."""

    latent_dim = 4

    def __init__(self) -> None:
        self._counter = 0

    def decode(self, z: np.ndarray) -> list[str]:
        out: list[str] = []
        for _ in range(len(z)):
            idx = self._counter
            self._counter += 1
            if idx % 7 == 0:
                out.append("INVALID_SMILES")
            elif idx % 5 == 0:
                out.append("c1ccccc1")  # repeated duplicate after first time
            elif idx % 2 == 0:
                n_c = (idx % 12) + 1
                out.append("C" * n_c + "c1ccccc1")
            else:
                n_c = (idx % 15) + 1
                out.append("C" * n_c + "O")
        return out


class _StubFeaturizer:
    def transform(self, smiles: list[str]) -> np.ndarray:
        return np.array([[len(s)] for s in smiles], dtype=np.float32)


def _stub_predict(X: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    # Deterministic P(active) in [0.2, 0.9] based on SMILES length
    pa = np.clip(0.2 + 0.04 * X[:, 0], 0.0, 0.92)
    return (pa >= 0.5).astype(int), np.stack([1.0 - pa, pa], axis=1)


def test_search_loop_stops_on_oracle_budget_and_writes_csvs(tmp_path) -> None:
    cfg = CMAConfig(n_populations=2, popsize=4, sigma0=0.5, bounds=3.0)
    solver = build_solver("pycma", solution_dim=4, cfg=cfg, seed=1)
    vae = _StubVAE()
    ev = Evaluator(
        decode=vae.decode,
        activity=_stub_predict,
        activity_model=None,
        featurizer=_StubFeaturizer(),
    )
    candidates = CandidateStore(hit_threshold=0.5)
    loop = SearchLoop(solver, ev, candidates, solver_name="pycma", seed=1)

    summary = loop.run(
        output_dir=tmp_path,
        oracle_budget=15,
        max_generations=0,
        save_all_evaluations=True,
        log_every=2,
    )

    assert summary["stopped_on"] == "oracle_budget"
    assert summary["oracle_calls"] >= 15
    assert (
        summary["oracle_calls"]
        + summary["duplicate_attempts"]
        + summary["invalid_attempts"]
        == summary["attempts"]
    )

    curve = pd.read_csv(tmp_path / "curve.csv")
    assert list(curve.columns) == CURVE_COLUMNS
    assert int(curve["oracle_calls"].iloc[-1]) == summary["oracle_calls"]

    attempts = pd.read_csv(tmp_path / "all_evaluations.csv")
    assert list(attempts.columns) == ATTEMPT_COLUMNS
    assert len(attempts) == summary["attempts"]
    assert int(attempts["oracle_called"].sum()) == summary["oracle_calls"]


def test_checkpoint_and_resume_appends_cleanly(tmp_path, monkeypatch) -> None:
    import EVA

    monkeypatch.setattr(EVA, "predict_from_features", _stub_predict)

    vae = _StubVAE()
    base_cfg = ExperimentConfig(
        cma=CMAConfig(n_populations=2, popsize=4, sigma0=0.5, bounds=3.0),
        run=RunConfig(
            solver="pycma",
            oracle_budget=0,
            max_generations=2,
            hit_threshold=0.5,
            save_all_evaluations=True,
            log_every=1,
            seed=7,
        ),
        output=OutputConfig(output_dir=str(tmp_path), run_name="run1"),
        tensorboard=TensorBoardConfig(enabled=False),
        solution_dim=4,
    )

    out_dir = tmp_path / "run1"
    s1 = run_single_seed(
        base_cfg,
        vae=vae,
        tabpfn=None,
        featurizer=_StubFeaturizer(),
        output_dir=out_dir,
    )
    assert s1["steps"] == 2
    assert (out_dir / "checkpoint.joblib").exists()
    assert (out_dir / "candidates.csv").exists()

    resumed_cfg = replace(
        base_cfg,
        run=replace(base_cfg.run, max_generations=4),
    )
    s2 = run_single_seed(
        resumed_cfg,
        vae=vae,
        tabpfn=None,
        featurizer=_StubFeaturizer(),
        output_dir=out_dir,
        resume_from=str(out_dir / "checkpoint.joblib"),
    )
    assert s2["steps"] == 4
    assert s2["attempts"] == 32
    assert s2["oracle_calls"] >= s1["oracle_calls"]

    curve = pd.read_csv(out_dir / "curve.csv")
    assert curve["step"].tolist() == [0, 1, 2, 3]
    attempts = pd.read_csv(out_dir / "all_evaluations.csv")
    assert len(attempts) == 32

    saved_summary = json.loads((out_dir / "summary.json").read_text())
    assert saved_summary["steps"] == 4
