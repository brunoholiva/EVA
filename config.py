"""Barebones TOML configuration for EVA."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_SOLVERS = ("pycma", "naive")


@dataclass
class CMAConfig:
    """Hyperparameters for independent ``pycma`` walkers."""

    n_populations: int = 15
    popsize: int = 36
    sigma0: float = 0.5
    init_std: float = 1.0
    stagnation_steps: int = 0
    ipop_factor: float = 1.0
    max_popsize: int = 0
    resample_duplicates: int = 0
    soft_restart: bool = True
    sigma_restart: float = 0.8
    opts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate CMA parameters."""
        if self.n_populations < 1:
            raise ValueError("cma.n_populations must be >= 1")
        if self.popsize < 2:
            raise ValueError("cma.popsize must be >= 2")
        if self.sigma0 <= 0.0:
            raise ValueError("cma.sigma0 must be > 0")
        if self.init_std < 0.0:
            raise ValueError("cma.init_std must be >= 0")
        if self.stagnation_steps < 0:
            raise ValueError("cma.stagnation_steps must be >= 0")
        if self.ipop_factor < 1.0:
            raise ValueError("cma.ipop_factor must be >= 1.0")
        if self.max_popsize < 0 or (0 < self.max_popsize < self.popsize):
            raise ValueError("cma.max_popsize must be 0 or >= cma.popsize")
        if self.resample_duplicates < 0:
            raise ValueError("cma.resample_duplicates must be >= 0")
        if self.sigma_restart <= 0.0:
            raise ValueError("cma.sigma_restart must be > 0")

    @property
    def batch_size(self) -> int:
        """Total candidates proposed per step across all populations."""
        return self.n_populations * self.popsize


@dataclass
class ModelsConfig:
    """Model paths and compute device."""

    vae_repo: str = "3BioCompBio/chembed-default"
    pca_path: str = ""
    predictor_path: str = "data/predictor/tabpfn_minimol.joblib"
    device: str = "cuda"


@dataclass
class RunConfig:
    """Execution, budget, and output parameters."""

    solver: str = "pycma"
    oracle_budget: int = 100_000
    hit_threshold: float = 0.5
    save_all: bool = False
    seed: int = 42
    output_dir: str = "results"
    run_name: str = "01_pycma_baseline"

    def __post_init__(self) -> None:
        """Validate run settings."""
        if self.solver not in VALID_SOLVERS:
            raise ValueError(
                f"Unknown run.solver {self.solver!r}; expected {VALID_SOLVERS}"
            )
        if self.oracle_budget < 1:
            raise ValueError("run.oracle_budget must be >= 1")
        if not (0.0 <= self.hit_threshold <= 1.0):
            raise ValueError("run.hit_threshold must be in [0.0, 1.0]")


@dataclass
class Config:
    """Top-level EVA configuration."""

    cma: CMAConfig = field(default_factory=CMAConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load and validate a :class:`Config` from a TOML file."""
        with Path(path).open("rb") as f:
            raw = tomllib.load(f)

        cma_raw = dict(raw.get("cma", {}))
        opts = dict(cma_raw.pop("opts", {}))
        return cls(
            cma=CMAConfig(**cma_raw, opts=opts),
            models=ModelsConfig(**raw.get("models", {})),
            run=RunConfig(**raw.get("run", {})),
        )
