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
from reporting.console import console, detail, make_progress_bar, section, step
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
    vae: ChemBedVAE | ProjectedVAE,
    start_gen: int = 0,
) -> None:
    """Run the CMA-MAE loop with a progress bar and periodic saves."""
    eval_every = cfg.run.eval_every
    n_gen = cfg.run.n_generations

    # Simple progress bar for generations only
    progress, task_id = make_progress_bar("CMA-MAE", n_gen)

    with progress:

        def _on_step(gen, result, archive):
            """Update progress bar every generation."""
            progress.update(task_id, completed=gen + 1)
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
                if gen == 0:
                    save_archive(
                        loop.archive,
                        vae.decode,
                        output_dir,
                        suffix="archive_gen0",
                        dimension_names=cfg.archive.active_dimension_names(),
                        real_objectives=loop.real_objectives,
                    )

        loop.run(
            n_generations=n_gen,
            eval_every=eval_every,
            on_generation=_on_progress,
            on_step=_on_step,
            start_gen=start_gen,
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
            real_objectives=loop.result_real_objectives,
        )
    save_scheduler(loop.scheduler, output_dir)
    visualize_archive(
        loop.result_archive if loop.result_archive is not None else loop.archive,
        output_dir,
        dimension_names=dimension_names,
    )


def _warm_start(
    vae: ChemBedVAE | ProjectedVAE,
    evaluator: Evaluator,
    cfg: ExperimentConfig,
) -> np.ndarray | None:
    """Sample random latent vectors and return top-K by P(active) as warm-start.

    Parameters
    ----------
    vae : ChemBedVAE or ProjectedVAE
        The generative model.
    evaluator : Evaluator
        The scoring pipeline.
    cfg : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    np.ndarray or None
        Top-K latent vectors by P(active), shape (n_top, latent_dim).
        Returns None if warm-start is disabled or no valid molecules found.
    """
    if not cfg.warm_start.enabled:
        return None

    section("Warm-Start Initialization")
    n_samples = cfg.warm_start.n_samples
    n_top = cfg.warm_start.n_top
    latent_dim = cfg.archive.solution_dim

    step(f"Sampling {n_samples} random latent vectors")
    rng = np.random.default_rng(cfg.run.seed)
    z_candidates = rng.standard_normal((n_samples, latent_dim))

    step("Evaluating candidates")
    result = evaluator(z_candidates)

    valid_mask = result.p_active > 0
    n_valid = int(valid_mask.sum())
    detail(f"Valid molecules: {n_valid}/{n_samples} ({100 * n_valid / n_samples:.1f}%)")

    if n_valid == 0:
        detail("No valid molecules found, skipping warm-start")
        return None

    valid_indices = np.where(valid_mask)[0]
    valid_p_active = result.p_active[valid_mask]

    top_k = min(n_top, n_valid)
    top_indices = valid_indices[np.argsort(valid_p_active)[-top_k:][::-1]]
    top_latents = z_candidates[top_indices]

    detail(f"Top-{top_k} P(active): [{result.p_active[top_indices].min():.4f}, "
           f"{result.p_active[top_indices].max():.4f}]")

    return top_latents


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
        warm_start_latents = None
    else:
        warm_start_latents = _warm_start(vae, evaluator, cfg)

        scheduler = build_scheduler(
            cfg.archive, cfg.emitter, cfg.run.seed,
            warm_start_latents=warm_start_latents,
            warm_start_enabled=cfg.warm_start.enabled,
        )
        start_gen = 0

    loop = CMAMAELoop(
        scheduler,
        evaluator,
        objective_cap=cfg.archive.objective_cap,
        warm_start_n_generations=cfg.warm_start.n_generations if cfg.warm_start.enabled else 0,
        warm_start_threshold_min=cfg.warm_start.threshold_min if cfg.warm_start.enabled else cfg.archive.threshold_min,
        threshold_min=cfg.archive.threshold_min,
    )

    try:
        section("Running CMA-MAE")
        _run_loop(
            loop,
            cfg,
            output_dir,
            tb_logger,
            evaluator,
            vae,
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
