"""Unit and integration tests for barebones EVA."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import selfies as sf

from config import CMAConfig, Config, RunConfig
from EVA import (
    ATTEMPT_COLUMNS,
    CANDIDATE_COLUMNS,
    CURVE_COLUMNS,
    CandidateStore,
    run_search,
)
from oracle import (
    INVALID_OBJECTIVE,
    MinimolFeaturizer,
    Oracle,
    canonicalize_smiles,
    generic_scaffold,
)
from solvers import CMAWalker, MultiCMA, NaiveSolver, build_solver
from vae import _selfies_to_canonical


class _LengthFeaturizer:
    def transform(self, smiles: list[str]) -> np.ndarray:
        return np.array([[len(s)] for s in smiles], dtype=np.float32)


def test_config_loads_repo_toml() -> None:
    cfg = Config.from_toml("config.toml")
    assert cfg.cma.n_populations == 15
    assert cfg.cma.popsize == 36
    assert cfg.cma.batch_size == 540
    assert cfg.run.solver == "pycma"
    assert cfg.run.oracle_budget == 100_000
    assert cfg.run.save_all is False


def test_config_opts_and_validation(tmp_path) -> None:
    p = tmp_path / "exp.toml"
    p.write_text(
        "[cma]\npopsize = 20\n[cma.opts]\nCMA_diagonal = true\nminstd = 0.2\n"
        "[run]\noracle_budget = 500\n"
    )
    cfg = Config.from_toml(p)
    assert cfg.cma.opts == {"CMA_diagonal": True, "minstd": 0.2}

    with pytest.raises(ValueError, match="solver"):
        RunConfig(solver="grid")
    with pytest.raises(ValueError, match="oracle_budget"):
        RunConfig(oracle_budget=0)


def test_selfies_to_canonical_and_token_limit() -> None:
    assert _selfies_to_canonical(sf.encoder("OCC")) == "CCO"
    assert _selfies_to_canonical("") is None
    assert _selfies_to_canonical("[C]" * 205) is None
    assert canonicalize_smiles("C(O)C") == "CCO"
    assert canonicalize_smiles("not-a-smiles") is None


def test_generic_scaffold() -> None:
    assert generic_scaffold("c1ccccc1") == generic_scaffold("Cc1ccccc1")
    assert generic_scaffold("CCO") == "acyclic"
    assert generic_scaffold("invalid_smiles") == "invalid"
    hypervalent = (
        "COC1CNC(=O)SC(=O)/C(=C(/O)C=O)OC=CS12(=O)NC=C(OC=S)N=C(Oc1ccc(F)cc1-c1cccs1)O2"
    )
    assert generic_scaffold(hypervalent) not in ("invalid", "acyclic", "")


def test_minimol_featurizer() -> None:
    feat = MinimolFeaturizer()
    assert feat.transform([]).shape == (0, 512)
    assert feat._model is None
    out = feat.transform(["CCO", "c1ccccc1"])
    assert out.shape == (2, 512)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_oracle_deduplication_and_accounting() -> None:
    calls = [0]

    def predict(X: np.ndarray) -> np.ndarray:
        calls[0] += 1
        return np.clip(0.1 * X[:, 0], 0.0, 0.95)

    oracle = Oracle(_LengthFeaturizer(), predict)
    b1 = oracle.score(["CCO", "CCO", None, "c1ccccc1"])
    assert calls[0] == 1
    assert b1.n_valid == 3
    assert b1.n_duplicates == 1
    assert b1.n_oracle_calls == 2
    assert b1.objectives[0] > 0.0
    assert b1.objectives[1] == INVALID_OBJECTIVE
    assert b1.objectives[2] == INVALID_OBJECTIVE
    assert b1.objectives[3] > b1.objectives[0]

    b2 = oracle.score(["CCO"])
    assert calls[0] == 1
    assert b2.n_oracle_calls == 0
    assert b2.n_duplicates == 1
    assert b2.objectives[0] == INVALID_OBJECTIVE


def test_cma_walker_and_solvers() -> None:
    target = np.full(6, 1.5)
    cfg = CMAConfig(n_populations=1, popsize=14, sigma0=0.5, init_std=0.0)
    w = CMAWalker(0, 6, cfg, init_seed=1, es_seed=2)
    d0 = float(np.linalg.norm(w.mean - target))
    for _ in range(25):
        z = w.ask()
        w.tell(z, -np.sum((z - target) ** 2, axis=1))
    assert float(np.linalg.norm(w.mean - target)) < 0.25 * d0

    with pytest.raises(ValueError, match="cma.opts"):
        CMAWalker(0, 4, CMAConfig(opts={"popsize": 10}), 1, 2)

    mc = MultiCMA(dim=8, cfg=CMAConfig(n_populations=3, popsize=6), seed=42)
    z, wids = mc.ask()
    assert z.shape == (18, 8)
    assert wids.tolist() == [0] * 6 + [1] * 6 + [2] * 6
    mc.tell(z, np.linspace(-1.0, 0.9, 18))
    st = mc.stats()
    assert st.walker_spread > 0.0
    assert st.sigma_mean > 0.0

    nv = build_solver("naive", dim=8, cfg=CMAConfig(n_populations=2, popsize=5), seed=1)
    assert isinstance(nv, NaiveSolver)
    zn, wn = nv.ask()
    assert zn.shape == (10, 8)
    assert np.all(wn == -1)


def test_candidate_store_and_run_search(tmp_path) -> None:
    counter = [0]

    def decode_stub(z: np.ndarray) -> list[str | None]:
        out: list[str | None] = []
        for _ in range(len(z)):
            idx = counter[0]
            counter[0] += 1
            if idx % 6 == 0:
                out.append(None)
            elif idx % 4 == 0:
                out.append("c1ccccc1")
            elif idx % 2 == 0:
                out.append("C" * (idx + 1) + "c1ccccc1")
            else:
                out.append("C" * (idx + 1) + "c1ccc(-c2ccccc2)cc1")
        return out

    oracle = Oracle(
        _LengthFeaturizer(),
        lambda X: np.clip(0.2 + 0.03 * X[:, 0], 0.0, 0.92),
    )
    solver = build_solver(
        "pycma", dim=4, cfg=CMAConfig(n_populations=2, popsize=4), seed=7
    )

    out_off = tmp_path / "run_light"
    summary, store = run_search(
        solver=solver,
        decode_fn=decode_stub,
        oracle=oracle,
        output_dir=out_off,
        solver_name="pycma",
        seed=7,
        oracle_budget=14,
        hit_threshold=0.5,
        save_all=False,
        show_progress=False,
    )
    assert summary["stopped_on"] == "oracle_budget"
    assert summary["oracle_calls"] >= 14
    assert not (out_off / "all_evaluations.csv").exists()
    assert isinstance(store, CandidateStore)

    curve = pd.read_csv(out_off / "curve.csv")
    assert list(curve.columns) == CURVE_COLUMNS
    cands = pd.read_csv(out_off / "candidates.csv")
    assert list(cands.columns) == CANDIDATE_COLUMNS
    saved = json.loads((out_off / "summary.json").read_text())
    assert saved["oracle_calls"] == summary["oracle_calls"]

    # Verify save_all=True writes all_evaluations.csv
    out_all = tmp_path / "run_all"
    solver2 = build_solver(
        "naive", dim=4, cfg=CMAConfig(n_populations=2, popsize=4), seed=8
    )
    oracle2 = Oracle(
        _LengthFeaturizer(),
        lambda X: np.clip(0.2 + 0.03 * X[:, 0], 0.0, 0.92),
    )
    s2, _ = run_search(
        solver=solver2,
        decode_fn=decode_stub,
        oracle=oracle2,
        output_dir=out_all,
        solver_name="naive",
        seed=8,
        oracle_budget=10,
        save_all=True,
        show_progress=False,
    )
    evals = pd.read_csv(out_all / "all_evaluations.csv")
    assert list(evals.columns) == ATTEMPT_COLUMNS
    assert len(evals) == s2["attempts"]
