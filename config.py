"""Configuration dataclasses for EVA parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import toml

VALID_SOLVERS = ("pycma", "naive")


@dataclass
class CMAConfig:
    """Configuration for multi-population ``pycma`` CMA-ES walkers."""

    n_populations: int = 15
    popsize: int = 36
    sigma0: float = 0.5
    init_std: float = 1.0
    bounds: float | None = 3.0
    max_gens_per_run: int = 0
    opts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate CMA hyperparameters and normalize unbounded settings."""
        if self.n_populations < 1:
            raise ValueError("cma.n_populations must be >= 1")
        if self.popsize < 2:
            raise ValueError("cma.popsize must be >= 2")
        if self.sigma0 <= 0.0:
            raise ValueError("cma.sigma0 must be > 0")
        if self.init_std < 0.0:
            raise ValueError("cma.init_std must be >= 0")
        if self.bounds is not None and self.bounds <= 0.0:
            self.bounds = None
        if self.max_gens_per_run < 0:
            raise ValueError("cma.max_gens_per_run must be >= 0")

    @property
    def total_batch_size(self) -> int:
        """Total candidates proposed across all populations per step."""
        return self.n_populations * self.popsize


@dataclass
class GenerativeConfig:
    """Configuration for the ChemBed VAE generative model."""

    model_repo_id: str = "3BioCompBio/chembed-default"
    device: str = "cuda"
    latent_dim: int = 256
    seed: int = 42


@dataclass
class PCALatentConfig:
    """Configuration for optional whitened PCA latent reduction."""

    path: str = "data/pca_latent.joblib"
    enabled: bool = False


@dataclass
class ActivityConfig:
    """Configuration for the TabPFN activity oracle."""

    model_path: str = "data/predictor/tabpfn_minimol.joblib"
    device: str = "cuda"
    representation: str = "minimol"


@dataclass
class RunConfig:
    """Configuration for stopping budgets, logging, and reproducibility."""

    solver: str = "pycma"
    oracle_budget: int = 100_000
    max_generations: int = 0
    hit_threshold: float = 0.5
    save_all_evaluations: bool = True
    log_every: int = 20
    seed: int = 42
    resume_from: str = ""
    hypothesis: str | None = None

    def __post_init__(self) -> None:
        """Validate solver choice and stopping criteria."""
        if self.solver not in VALID_SOLVERS:
            raise ValueError(
                f"Unknown run.solver {self.solver!r}; expected one of {VALID_SOLVERS}"
            )
        if self.oracle_budget <= 0 and self.max_generations <= 0:
            raise ValueError(
                "At least one of run.oracle_budget or run.max_generations must be > 0"
            )
        if not (0.0 <= self.hit_threshold <= 1.0):
            raise ValueError("run.hit_threshold must lie in [0.0, 1.0]")


@dataclass
class OutputConfig:
    """Configuration for output paths."""

    output_dir: str = "results"
    run_name: str = "default"


@dataclass
class TensorBoardConfig:
    """Configuration for TensorBoard logging."""

    enabled: bool = False
    log_dir: str = "tensorboard"
    scalar_every: int = 1
    molecule_every: int = 10


@dataclass
class ExperimentConfig:
    """Top-level configuration for an EVA experiment."""

    cma: CMAConfig = field(default_factory=CMAConfig)
    generative: GenerativeConfig = field(default_factory=GenerativeConfig)
    activity: ActivityConfig = field(default_factory=ActivityConfig)
    run: RunConfig = field(default_factory=RunConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    pca: PCALatentConfig = field(default_factory=PCALatentConfig)
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    solution_dim: int = 256

    def __post_init__(self) -> None:
        """Sync default working dimensionality from generative config."""
        if self.solution_dim <= 0:
            self.solution_dim = self.generative.latent_dim

    @classmethod
    def from_toml(cls, file_path: str) -> ExperimentConfig:
        """Load an :class:`ExperimentConfig` from a TOML file.

        Parameters
        ----------
        file_path : str
            Path to the TOML configuration file.

        Returns
        -------
        ExperimentConfig
            Parsed and validated experiment configuration.
        """
        with open(file_path, "r") as f:
            raw = toml.load(f)

        cma_raw = dict(raw.get("cma", {}))
        cma_opts = dict(cma_raw.pop("opts", {}))
        cma_cfg = CMAConfig(**cma_raw, opts=cma_opts)

        generative_cfg = GenerativeConfig(**raw.get("generative", {}))
        return cls(
            cma=cma_cfg,
            generative=generative_cfg,
            activity=ActivityConfig(**raw.get("activity", {})),
            run=RunConfig(**raw.get("run", {})),
            output=OutputConfig(**raw.get("output", {})),
            pca=PCALatentConfig(**raw.get("pca", {})),
            tensorboard=TensorBoardConfig(**raw.get("tensorboard", {})),
            solution_dim=generative_cfg.latent_dim,
        )
