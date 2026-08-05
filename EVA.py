"""Main CMA-MAE evolution loop for EVA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import ExperimentConfig
from chemistry.features import MoleculeFeaturizer
from generative import ChemBedVAE, ProjectedVAE
from optimization import (
    Evaluator,
    TensorBoardLogger,
    build_scheduler,
    load_scheduler,
    print_generation,
    print_results,
    save_archive,
    save_scheduler,
    visualize_archive,
)
from reporting.console import console, detail, make_eva_progress, section, step
from optimization.loop import CMAMAELoop
from evaluation.activity import load_model as load_tabpfn
from evaluation.activity import predict_from_features
from evaluation.applicability import ADScorer


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


def _load_vae(cfg: ExperimentConfig) -> ChemBedVAE | ProjectedVAE:
    """Load the ChemBed VAE, optionally wrapped with PCA projection."""
    step("Loading ChemBed VAE")
    vae = ChemBedVAE(
        model_repo_id=cfg.generative.model_repo_id,
        device=cfg.generative.device,
        latent_dim=cfg.generative.latent_dim,
    )
    if cfg.pca.enabled:
        step(f"Applying PCA projection ({cfg.pca.path})")
        vae = ProjectedVAE(vae, cfg.pca.path)
        cfg.archive.solution_dim = vae.latent_dim
        detail(f"Working space: {vae.latent_dim} dim (was {cfg.generative.latent_dim})")
    return vae


def _load_scorers(cfg: ExperimentConfig):
    """Load AD scorer, TabPFN, and featurizer."""
    step("Loading scorers")
    ad = ADScorer(
        model_path=cfg.ad.ad_model_path,
        n_neighbors=cfg.ad.n_neighbors,
    )
    tabpfn = load_tabpfn(
        path=cfg.activity.model_path,
        device=cfg.activity.device,
        softmax_temperature=cfg.activity.softmax_temperature,
    )
    featurizer = MoleculeFeaturizer()
    return ad, tabpfn, featurizer


def _run_loop(
    loop: CMAMAELoop,
    cfg: ExperimentConfig,
    output_dir: Path,
    tb_logger: TensorBoardLogger,
    evaluator: Evaluator,
    start_gen: int = 0,
) -> None:
    """Run the CMA-MAE loop with a progress bar and periodic saves."""
    eval_every = cfg.run.eval_every
    n_gen = cfg.run.n_generations
    batch_size = cfg.emitter.batch_size * cfg.emitter.n_emitters

    # Create 3-bar layout (fastest at top)
    progress, solutions_task, featurization_task, main_task = make_eva_progress(
        n_gen, batch_size
    )

    # Wire progress bars to evaluator
    evaluator._progress = progress
    evaluator._solutions_task = solutions_task
    evaluator._featurization_task = featurization_task

    with progress:
        def _on_step(gen, result, archive):
            """Update progress bar every generation."""
            # Update main bar
            progress.update(
                main_task,
                completed=gen + 1,
                status=f"{len(archive)} cells",
            )
            # Update featurization bar with timing info
            if result.timings.featurize_predict > 0:
                progress.update(
                    featurization_task,
                    completed=0,
                    total=1,
                    status=f"{result.timings.featurize_predict:.1f}s",
                )
            # Log to TensorBoard
            tb_logger.log_generation(
                step=gen,
                result=result,
                archive=archive,
                result_archive=loop.result_archive,
                dimension_names=cfg.archive.active_dimension_names(),
                insertion_stats=loop.last_insertion_stats,
                emitter_stats=loop.last_emitter_stats,
                real_objectives=loop.real_objectives,
                objective_cap=cfg.archive.objective_cap,
            )

        def _on_progress(gen, result, archive):
            """Print generation table and save state at intervals."""
            if gen % eval_every == 0 or gen == n_gen - 1:
                print_generation(gen, result, archive, loop.result_archive)
                if gen > 0 and result.timings.decode > 0:
                    t = result.timings
                    detail(
                        f"decode={t.decode:.1f}s valid={t.validity:.1f}s "
                        f"feat={t.featurize_predict:.1f}s cpu={t.cpu_scorers:.1f}s"
                    )
                save_scheduler(loop.scheduler, output_dir)

        loop.run(
            n_generations=n_gen,
            eval_every=eval_every,
            on_generation=_on_progress,
            on_step=_on_step,
            start_gen=start_gen,
            progress=progress,
            solutions_task=solutions_task,
            featurization_task=featurization_task,
        )


def _save_results(
    loop: CMAMAELoop,
    vae: ChemBedVAE | ProjectedVAE,
    output_dir: Path,
    dimension_names: list[str],
) -> None:
    """Save archives, scheduler, and visualizations."""
    save_archive(
        loop.archive,
        vae.decode,
        output_dir,
        dimension_names=dimension_names,
        real_objectives=loop.real_objectives,
    )
    if loop.result_archive is not None:
        save_archive(
            loop.result_archive,
            vae.decode,
            output_dir,
            suffix="result_archive",
            dimension_names=dimension_names,
            real_objectives=loop.real_objectives,
        )
    save_scheduler(loop.scheduler, output_dir)
    visualize_archive(
        loop.result_archive if loop.result_archive is not None else loop.archive,
        output_dir,
        dimension_names=dimension_names,
    )


def main(argv: list[str | None] | None = None) -> None:
    """Entry point for the EVA evolution loop."""
    config_path, resume_path = _parse_args(argv)

    section("EVA")
    step(f"Loading config from {config_path}")
    cfg = ExperimentConfig.from_toml(config_path)

    output_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    resume_from = resume_path or cfg.run.resume_from or None
    tb_logger = TensorBoardLogger(cfg.tensorboard, output_dir)
    tb_logger.log_config(cfg)

    vae = _load_vae(cfg)
    ad, tabpfn, featurizer = _load_scorers(cfg)

    evaluator = Evaluator(
        decode=vae.decode,
        ad=ad,
        activity=predict_from_features,
        activity_model=tabpfn,
        featurizer=featurizer,
        archive_cfg=cfg.archive,
    )

    if resume_from:
        section("Resuming CMA-MAE")
        scheduler = load_scheduler(resume_from)
        start_gen = scheduler.emitters[0]._itrs
        detail(f"Resuming from generation {start_gen}")
    else:
        scheduler = build_scheduler(cfg.archive, cfg.emitter, cfg.run.seed)
        start_gen = 0

    loop = CMAMAELoop(scheduler, evaluator, objective_cap=cfg.archive.objective_cap)

    try:
        section("Running CMA-MAE")
        _run_loop(
            loop,
            cfg,
            output_dir,
            tb_logger,
            evaluator,
            start_gen=start_gen,
        )

        print_results(
            loop.archive,
            vae.decode,
            result_archive=loop.result_archive,
            dimension_names=cfg.archive.active_dimension_names(),
        )
        _save_results(
            loop,
            vae,
            output_dir,
            cfg.archive.active_dimension_names(),
        )
    finally:
        tb_logger.close()


if __name__ == "__main__":
    main()
