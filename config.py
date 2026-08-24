from __future__ import annotations

from dataclasses import dataclass, field

import toml


@dataclass
class DimensionConfig:
    """Configuration for a single archive dimension."""

    name: str
    resolution: int
    range: tuple[float, float]
    scaffold: bool = False


@dataclass
class ArchiveConfig:
    solution_dim: int
    dimensions: list[DimensionConfig]
    learning_rate: float
    threshold_min: float
    objective_cap: float | None = None

    def validate(self) -> None:
        """Validate archive dimension configuration."""
        if not self.dimensions:
            raise ValueError("archive must have at least one dimension")

        valid_names = {
            "ad",
            "logp",
            "tpsa",
            "mw",
            "fsp3",
            "num_rotb",
            "num_rings",
            "balabanj",
            "vsa_estate2",
            "bcut2d_logplow",
        }
        invalid_names = [d.name for d in self.dimensions if d.name not in valid_names]
        if invalid_names:
            raise ValueError(
                "archive dimensions contain unsupported metrics: "
                + ", ".join(invalid_names)
            )

    def active_dims(self) -> list[int]:
        """Return grid resolution for all dimensions."""
        return [d.resolution for d in self.dimensions]

    def active_ranges(self) -> list[list[float]]:
        """Return value ranges for all dimensions."""
        return [list(d.range) for d in self.dimensions]

    def active_dimension_names(self) -> list[str]:
        """Return names of all dimensions."""
        return [d.name for d in self.dimensions]


@dataclass
class EmitterConfig:
    sigma0: float
    batch_size: int
    n_emitters: int
    restart_mode: str = "hybrid"
    random_restart_prob: float = 0.3


@dataclass
class GenerativeConfig:
    model_repo_id: str
    device: str
    latent_dim: int
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
class WarmStartConfig:
    enabled: bool = False
    n_top: int = 16
    n_generations: int = 50
    threshold_min: float = 0.0
    representatives_path: str | None = None


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
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)

    @classmethod
    def from_toml(cls, file_path: str) -> ExperimentConfig:
        """Read a TOML file and convert it into typed dataclasses."""
        with open(file_path, "r") as f:
            raw = toml.load(f)

        output_defaults = {"output_dir": "results", "run_name": "default"}

        archive_raw = raw.get("archive", {})
        dimensions_raw = archive_raw.pop("dimensions", [])
        dimensions = [
            DimensionConfig(
                name=d["name"],
                resolution=d["resolution"],
                range=tuple(d["range"]),
                scaffold=d.get("scaffold", False),
            )
            for d in dimensions_raw
        ]
        archive = ArchiveConfig(
            solution_dim=archive_raw["solution_dim"],
            dimensions=dimensions,
            learning_rate=archive_raw["learning_rate"],
            threshold_min=archive_raw["threshold_min"],
            objective_cap=archive_raw.get("objective_cap"),
        )

        config = cls(
            archive=archive,
            emitter=EmitterConfig(**raw.get("emitter", {})),
            generative=GenerativeConfig(**raw.get("generative", {})),
            ad=ADConfig(**raw.get("ad", {})),
            activity=ActivityConfig(**raw.get("activity", {})),
            run=RunConfig(**raw.get("run", {})),
            output=OutputConfig(**raw.get("output", output_defaults)),
            pca=PCALatentConfig(**raw.get("pca", {})),
            warm_start=WarmStartConfig(**raw.get("warm_start", {})),
            tensorboard=TensorBoardConfig(**raw.get("tensorboard", {})),
        )
        config.archive.validate()
        return config
