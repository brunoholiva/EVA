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
    load_archive_novelty_cache,
    load_scheduler,
    print_generation,
    print_results,
    save_archive,
    save_archive_novelty_cache,
    save_scheduler,
    visualize_archive,
)
from optimization.loop import CMAMAELoop
from prediction.activity import load_model as load_tabpfn, predict
from scoring.ad_scorer import ADScorer
from scoring.archive_novelty import ArchiveNoveltyScorer

console = Console()


def _parse_args(argv: list[str] | None) -> tuple[str, str | None]:
    """Return the config path and optional resume path from CLI arguments."""
    parser = argparse.ArgumentParser(description="EVA: Evolution of Viable Antibiotics")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to experiment config TOML (default: config.toml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to scheduler.joblib to resume from (overrides config)",
    )
    args = parser.parse_args(argv)
    return args.config, args.resume


def _load_vae(cfg: ExperimentConfig) -> ChemBedVAE:
    """Load the ChemBed VAE from HuggingFace."""
    console.print("Loading ChemBed VAE ...")
    return ChemBedVAE(
        model_repo_id=cfg.generative.model_repo_id,
        device=cfg.generative.device,
        latent_dim=cfg.generative.latent_dim,
    )


def _load_scorers(cfg: ExperimentConfig):
    """Load AD scorer, archive novelty scorer, TabPFN, and featurizer."""
    console.print("Loading scorers ...")
    ad = ADScorer(
        model_path=cfg.ad.ad_model_path,
        n_neighbors=cfg.ad.n_neighbors,
    )
    archive_novelty = ArchiveNoveltyScorer(
        n_bits=cfg.ad.n_bits,
        radius=cfg.ad.radius,
        max_cache_size=cfg.ad.max_cache_size,
    )
    tabpfn = load_tabpfn(
        path=cfg.activity.model_path,
        device=cfg.activity.device,
        softmax_temperature=cfg.activity.softmax_temperature,
    )
    featurizer = MoleculeFeaturizer()
    return ad, archive_novelty, tabpfn, featurizer


def _run_loop(
    loop: CMAMAELoop,
    cfg: ExperimentConfig,
    output_dir: Path,
    archive_novelty: ArchiveNoveltyScorer,
    start_gen: int = 0,
) -> None:
    """Run the CMA-MAE loop with a progress bar and periodic saves."""
    eval_every = cfg.run.eval_every
    n_gen = cfg.run.n_generations

    columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="white", complete_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.fields[archive_size]} cells"),
        TextColumn("•"),
        TextColumn("{task.fields[cache_size]} cache"),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]

    with Progress(*columns, console=console) as progress:
        task = progress.add_task(
            "CMA-MAE",
            total=n_gen,
            completed=start_gen,
            archive_size=len(loop.archive),
            cache_size=archive_novelty.cache_size,
        )

        def _on_step(gen, result, archive):
            """Update progress bar every generation."""
            progress.update(
                task,
                completed=gen + 1,
                archive_size=len(archive),
                cache_size=archive_novelty.cache_size,
            )

        def _on_progress(gen, result, archive):
            """Print generation table and save state at intervals."""
            if gen % eval_every == 0 or gen == n_gen - 1:
                print_generation(gen, result, archive)
                save_scheduler(loop.scheduler, output_dir)
                save_archive_novelty_cache(archive_novelty, output_dir)

        loop.run(
            n_generations=n_gen,
            eval_every=eval_every,
            on_generation=_on_progress,
            on_step=_on_step,
            start_gen=start_gen,
        )


def main(argv: list[str | None] | None = None) -> None:
    """Entry point for the EVA evolution loop."""
    config_path, resume_path = _parse_args(argv)

    console.rule("[bold green]EVA")
    console.print(f"Loading config from {config_path}")
    cfg = ExperimentConfig.from_toml(config_path)

    output_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    resume_from = resume_path or cfg.run.resume_from or None

    vae = _load_vae(cfg)
    ad, archive_novelty, tabpfn, featurizer = _load_scorers(cfg)

    evaluator = Evaluator(
        decode=vae.decode,
        ad=ad,
        archive_novelty=archive_novelty,
        activity=predict,
        activity_model=tabpfn,
        featurizer=featurizer,
    )

    if resume_from:
        console.rule("[bold green]Resuming CMA-MAE")
        scheduler = load_scheduler(resume_from)
        start_gen = scheduler.emitters[0]._itrs
        console.print(f"  Resuming from generation {start_gen}")

        cache_path = Path(resume_from).parent / "archive_novelty_cache.joblib"
        restored = load_archive_novelty_cache(
            cache_path,
            n_bits=cfg.ad.n_bits,
            radius=cfg.ad.radius,
            max_cache_size=cfg.ad.max_cache_size,
        )
        if restored.cache_size == 0 and len(scheduler.archive) > 0:
            console.print("  Rebuilding archive novelty cache from archive ...")
            _rebuild_novelty_cache(scheduler.archive, vae.decode, archive_novelty)
        else:
            archive_novelty._fingerprints = restored._fingerprints
            archive_novelty._rebuild_arrays()
    else:
        scheduler = build_scheduler(cfg.archive, cfg.emitter, cfg.run.seed)
        start_gen = 0

    loop = CMAMAELoop(scheduler, evaluator)

    if not resume_from:
        console.print(f"Generating {cfg.generative.n_seeds} random seed molecules ...")
        rng = np.random.default_rng(cfg.generative.seed)
        z_seeds = vae.generate_random(cfg.generative.n_seeds, rng=rng)
        result = loop.seed_archive(z_seeds)
        console.print(
            f"  Seeded archive with {result.n_valid}/{len(z_seeds)} valid molecules"
        )
        _rebuild_novelty_cache(loop.archive, vae.decode, archive_novelty)

    console.rule("[bold green]Running CMA-MAE")
    _run_loop(loop, cfg, output_dir, archive_novelty, start_gen=start_gen)

    print_results(loop.archive, vae.decode)
    save_archive(loop.archive, vae.decode, output_dir)
    save_scheduler(loop.scheduler, output_dir)
    save_archive_novelty_cache(archive_novelty, output_dir)
    visualize_archive(loop.archive, output_dir)


def _rebuild_novelty_cache(archive, decode_fn, archive_novelty: ArchiveNoveltyScorer) -> None:
    """Decode all archive solutions and rebuild the archive novelty cache."""
    if len(archive) == 0:
        return
    data = archive.data()
    solutions = data["solution"]
    smiles = decode_fn(solutions)
    valid_smiles = [s for s in smiles if s != ""]
    archive_novelty.rebuild(valid_smiles)
    console.print(
        f"  Archive novelty cache: {archive_novelty.cache_size} fingerprints "
        f"(max {archive_novelty.max_cache_size})"
    )


if __name__ == "__main__":
    main()
