from __future__ import annotations

from dataclasses import dataclass, field

import toml


@dataclass
class ArchiveConfig:
    solution_dim: int
    dims: list[int]
    ranges: list[list[float]]
    dimension_names: list[str]
    dimension_enabled: list[bool]
    learning_rate: float
    threshold_min: float

    def validate(self) -> None:
        """Validate archive dimension configuration."""
        lengths = {
            "dims": len(self.dims),
            "ranges": len(self.ranges),
            "dimension_names": len(self.dimension_names),
            "dimension_enabled": len(self.dimension_enabled),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "archive dims, ranges, dimension_names, and dimension_enabled "
                "must have the same length"
            )
        if not any(self.dimension_enabled):
            raise ValueError("archive must enable at least one dimension")

        valid_names = {"br_sascore", "ad", "logp", "tpsa", "mw"}
        invalid_names = [
            name for name in self.dimension_names if name not in valid_names
        ]
        if invalid_names:
            raise ValueError(
                "archive dimension_names contain unsupported metrics: "
                + ", ".join(invalid_names)
            )

    def active_dims(self) -> list[int]:
        """Return grid resolution for enabled dimensions only."""
        return [d for d, e in zip(self.dims, self.dimension_enabled) if e]

    def active_ranges(self) -> list[list[float]]:
        """Return value ranges for enabled dimensions only."""
        return [r for r, e in zip(self.ranges, self.dimension_enabled) if e]

    def active_dimension_names(self) -> list[str]:
        """Return names of enabled dimensions."""
        return [n for n, e in zip(self.dimension_names, self.dimension_enabled) if e]


@dataclass
class EmitterConfig:
    sigma0: float
    batch_size: int
    n_emitters: int


@dataclass
class GenerativeConfig:
    model_repo_id: str
    device: str
    latent_dim: int
    n_seeds: int
    seed: int


@dataclass
class ADConfig:
    ad_model_path: str
    n_neighbors: int
    n_bits: int
    radius: int
    max_cache_size: int = 5000


@dataclass
class ActivityConfig:
    model_path: str
    device: str
    softmax_temperature: float


@dataclass
class RunConfig:
    n_generations: int
    eval_every: int
    seed: int
    resume_from: str


@dataclass
class OutputConfig:
    output_dir: str
    run_name: str


@dataclass
class PCALatentConfig:
    path: str = "data/pca_latent.joblib"
    enabled: bool = True
    variance_threshold: float = 0.99


@dataclass
class TensorBoardConfig:
    enabled: bool = False
    log_dir: str = "tensorboard"
    scalar_every: int = 1
    histogram_every: int = 10
    figure_every: int = 25


@dataclass
class ExperimentConfig:
    archive: ArchiveConfig
    emitter: EmitterConfig
    generative: GenerativeConfig
    ad: ADConfig
    activity: ActivityConfig
    run: RunConfig
    output: OutputConfig
    pca: PCALatentConfig = field(default_factory=PCALatentConfig)
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)

    @classmethod
    def from_toml(cls, file_path: str) -> ExperimentConfig:
        """Read a TOML file and convert it into typed dataclasses."""
        with open(file_path, "r") as f:
            raw = toml.load(f)

        output_defaults = {"output_dir": "results", "run_name": "default"}
        config = cls(
            archive=ArchiveConfig(**raw.get("archive", {})),
            emitter=EmitterConfig(**raw.get("emitter", {})),
            generative=GenerativeConfig(**raw.get("generative", {})),
            ad=ADConfig(**raw.get("ad", {})),
            activity=ActivityConfig(**raw.get("activity", {})),
            run=RunConfig(**raw.get("run", {})),
            output=OutputConfig(**raw.get("output", output_defaults)),
            pca=PCALatentConfig(**raw.get("pca", {})),
            tensorboard=TensorBoardConfig(**raw.get("tensorboard", {})),
        )
        config.archive.validate()
        return config
