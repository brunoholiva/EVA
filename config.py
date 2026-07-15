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
class NoveltyConfig:
    ad_model_path: str
    n_neighbors: int
    n_bits: int
    radius: int


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


@dataclass
class OutputConfig:
    output_dir: str
    run_name: str


@dataclass
class ExperimentConfig:
    archive: ArchiveConfig
    emitter: EmitterConfig
    generative: GenerativeConfig
    novelty: NoveltyConfig
    activity: ActivityConfig
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
            novelty=NoveltyConfig(**raw.get("novelty", {})),
            activity=ActivityConfig(**raw.get("activity", {})),
            run=RunConfig(**raw.get("run", {})),
            output=OutputConfig(**raw.get("output", output_defaults)),
        )
