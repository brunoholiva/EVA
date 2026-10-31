"""Tests for ExperimentConfig and TOML parsing."""

from __future__ import annotations

import pytest

from config import CMAConfig, ExperimentConfig, RunConfig


def test_load_repo_config_toml() -> None:
    cfg = ExperimentConfig.from_toml("config.toml")
    assert cfg.cma.n_populations == 15
    assert cfg.cma.popsize == 36
    assert cfg.cma.total_batch_size == 540
    assert cfg.pca.enabled is False
    assert cfg.solution_dim == 256
    assert cfg.activity.representation == "minimol"
    assert cfg.run.solver == "pycma"
    assert cfg.run.oracle_budget == 100_000


def test_cma_opts_passthrough_from_toml(tmp_path) -> None:
    toml_file = tmp_path / "exp.toml"
    toml_file.write_text("""
[cma]
n_populations = 4
popsize = 10
sigma0 = 0.4
bounds = 0.0

[cma.opts]
minstd = 0.25
CMA_diagonal = true

[run]
solver = "pycma"
oracle_budget = 500
""")
    cfg = ExperimentConfig.from_toml(str(toml_file))
    assert cfg.cma.bounds is None
    assert cfg.cma.opts == {"minstd": 0.25, "CMA_diagonal": True}


def test_invalid_solver_rejected() -> None:
    with pytest.raises(ValueError, match="solver"):
        RunConfig(solver="map_elites", oracle_budget=100)


def test_missing_budget_and_generations_rejected() -> None:
    with pytest.raises(ValueError, match="oracle_budget"):
        RunConfig(oracle_budget=0, max_generations=0)


def test_cma_validation() -> None:
    with pytest.raises(ValueError, match="popsize"):
        CMAConfig(popsize=1)
    with pytest.raises(ValueError, match="sigma0"):
        CMAConfig(sigma0=0.0)
