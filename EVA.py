"""Main CMA-MAE evolution loop for EVA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from config import ExperimentConfig
from featurization.features import MoleculeFeaturizer
from generative import ChemBedVAE
from optimization import (
    Evaluator,
    build_scheduler,
    print_generation,
    print_results,
    save_archive,
    visualize_archive,
)
from optimization.loop import CMAMAELoop
from prediction.activity import load_model as load_tabpfn, predict
from scoring.novelty import NoveltyScorer

console = Console()


def _parse_args(argv: list[str] | None) -> str:
    """Return the config path from CLI arguments."""
    parser = argparse.ArgumentParser(description="EVA: Evolution of Viable Antibiotics")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to experiment config TOML (default: config.toml)",
    )
    return parser.parse_args(argv).config


def _load_vae(cfg: ExperimentConfig) -> ChemBedVAE:
    """Load the ChemBed VAE from HuggingFace."""
    console.print("Loading ChemBed VAE ...")
    return ChemBedVAE(
        model_repo_id=cfg.generative.model_repo_id,
        device=cfg.generative.device,
        latent_dim=cfg.generative.latent_dim,
    )


def _load_scorers(cfg: ExperimentConfig):
    """Load novelty scorer, TabPFN, and featurizer."""
    console.print("Loading scorers ...")
    novelty = NoveltyScorer(
        model_path=cfg.novelty.ad_model_path,
        n_neighbors=cfg.novelty.n_neighbors,
    )
    tabpfn = load_tabpfn(
        path=cfg.activity.model_path,
        device=cfg.activity.device,
        softmax_temperature=cfg.activity.softmax_temperature,
    )
    featurizer = MoleculeFeaturizer()
    return novelty, tabpfn, featurizer


def _seed_and_run(
    loop: CMAMAELoop,
    cfg: ExperimentConfig,
    vae: ChemBedVAE,
) -> None:
    """Seed the archive and run the optimization loop with a progress bar."""
    console.print(f"Generating {cfg.generative.n_seeds} random seed molecules ...")
    rng = np.random.default_rng(cfg.generative.seed)
    z_seeds = vae.generate_random(cfg.generative.n_seeds, rng=rng)
    result = loop.seed_archive(z_seeds)
    console.print(
        f"  Seeded archive with {result.n_valid}/{len(z_seeds)} valid molecules"
    )

    eval_every = cfg.run.eval_every
    n_gen = cfg.run.n_generations

    console.rule("[bold green]Running CMA-MAE")

    columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="white", complete_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.fields[archive_size]} cells"),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]

    with Progress(*columns, console=console) as progress:
        task = progress.add_task("CMA-MAE", total=n_gen, archive_size=0)

        def _on_progress(gen, result, archive):
            """Print generation table at intervals and update progress bar."""
            if gen % eval_every == 0 or gen == n_gen - 1:
                print_generation(gen, result, archive)
            progress.update(task, completed=gen + 1, archive_size=len(archive))

        loop.run(
            n_generations=n_gen,
            eval_every=eval_every,
            on_generation=_on_progress,
        )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the EVA evolution loop."""
    config_path = _parse_args(argv)

    console.rule("[bold green]EVA")
    console.print(f"Loading config from {config_path}")
    cfg = ExperimentConfig.from_toml(config_path)

    vae = _load_vae(cfg)
    novelty, tabpfn, featurizer = _load_scorers(cfg)

    evaluator = Evaluator(
        decode=vae.decode,
        novelty=novelty,
        activity=predict,
        activity_model=tabpfn,
        featurizer=featurizer,
    )
    scheduler = build_scheduler(cfg.archive, cfg.emitter, cfg.run.seed)
    loop = CMAMAELoop(scheduler, evaluator)

    _seed_and_run(loop, cfg, vae)
    print_results(loop.archive, vae.decode)

    output_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    save_archive(loop.archive, vae.decode, output_dir)
    visualize_archive(loop.archive, output_dir)


if __name__ == "__main__":
    main()
