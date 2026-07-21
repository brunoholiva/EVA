from __future__ import annotations

from dataclasses import dataclass

import toml


@dataclass
class ArchiveConfig:
    solution_dim: int
    dims: list[int]
    ranges: list[list[float]]
    learning_rate: float
    threshold_min: float


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
class NoveltyConfig:
    n_neighbors: int = 5
    n_bits: int = 2048
    radius: int = 2
    max_cache_size: int = 5000


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
class ExperimentConfig:
    archive: ArchiveConfig
    emitter: EmitterConfig
    generative: GenerativeConfig
    ad: ADConfig
    activity: ActivityConfig
    novelty: NoveltyConfig
    run: RunConfig
    output: OutputConfig

    @classmethod
    def from_toml(cls, file_path: str) -> ExperimentConfig:
        """Read a TOML file and convert it into typed dataclasses."""
        with open(file_path, "r") as f:
            raw = toml.load(f)

        output_defaults = {"output_dir": "results", "run_name": "default"}
        return cls(
            archive=ArchiveConfig(**raw.get("archive", {})),
            emitter=EmitterConfig(**raw.get("emitter", {})),
            generative=GenerativeConfig(**raw.get("generative", {})),
            ad=ADConfig(**raw.get("ad", {})),
            activity=ActivityConfig(**raw.get("activity", {})),
            novelty=NoveltyConfig(**raw.get("novelty", {})),
            run=RunConfig(**raw.get("run", {})),
            output=OutputConfig(**raw.get("output", output_defaults)),
        )
