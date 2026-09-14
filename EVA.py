"""Main CMA-MAE evolution loop for EVA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig
from chemistry.features import MoleculeFeaturizer
from generative import ChemBedVAE, ProjectedVAE
from optimization import Evaluator, TensorBoardLogger, build_scheduler
from optimization.constants import INVALID_MOLECULE_OBJECTIVE
from optimization.loop import CMAMAELoop
from reporting.console import detail, make_progress_bar, section, step
from reporting.persistence import load_scheduler, save_archive, save_scheduler
from reporting.reporting import print_generation, print_results
from reporting.plotting import visualize_archive
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
    """Run the CMA-MAE loop with a progress bar and periodic saves.

    Collects every evaluated molecule into ``all_evaluations.csv`` with
    columns ``eval, smiles, p_active, valid, gen`` so that the run can
    be compared against naive and GA baselines on the same plots.
    """
    eval_every = cfg.run.eval_every
    n_gen = cfg.run.n_generations

    progress, task_id = make_progress_bar("CMA-MAE", n_gen)

    all_evals: list[tuple[int, str, float, bool, int]] = []
    eval_counter = [start_gen * cfg.emitter.batch_size * cfg.emitter.n_emitters]

    if start_gen > 0:
        existing_csv = output_dir / "all_evaluations.csv"
        if existing_csv.exists():
            existing = pd.read_csv(existing_csv)
            all_evals = [
                (int(r.eval), r.smiles, float(r.p_active), bool(r.valid), int(r.gen))
                for r in existing.itertuples(index=False)
            ]
            eval_counter = [int(existing["eval"].max())]
            detail(
                f"Loaded {len(all_evals):,} existing evaluations from {existing_csv}"
            )

    with progress:

        def _on_step(gen, result, archive):
            """Update progress bar, log to TensorBoard, collect evaluations."""
            progress.update(task_id, completed=gen + 1)
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
            for i in range(len(result.smiles)):
                eval_num = eval_counter[0] + i + 1
                valid = result.objectives[i] != INVALID_MOLECULE_OBJECTIVE
                if valid:
                    all_evals.append(
                        (
                            eval_num,
                            result.smiles[i],
                            float(result.p_active[i]),
                            True,
                            gen,
                        )
                    )
                else:
                    all_evals.append((eval_num, "", float("nan"), False, gen))
            eval_counter[0] += len(result.smiles)

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
                _save_all_evals(all_evals, output_dir)
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

    _save_all_evals(all_evals, output_dir)
    detail(f"Saved {len(all_evals):,} evaluations to all_evaluations.csv")


def _save_all_evals(
    evals: list[tuple[int, str, float, bool, int]],
    output_dir: Path,
) -> None:
    """Save collected evaluations to all_evaluations.csv."""
    df = pd.DataFrame(evals, columns=["eval", "smiles", "p_active", "valid", "gen"])
    df.to_csv(output_dir / "all_evaluations.csv", index=False)


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
    cfg: ExperimentConfig,
) -> np.ndarray | None:
    """Return latent vectors from predefined representatives to seed CMA-ES.

    Loads representative SMILES from a CSV file (column ``"SMILES"``)
    and encodes them through the VAE to obtain latent vectors used as
    initial emitter means.

    Parameters
    ----------
    vae : ChemBedVAE or ProjectedVAE
        The generative model (handles PCA projection transparently).
    cfg : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    np.ndarray or None
        Latent vectors, shape ``(n_top, latent_dim)``. Returns None if
        warm-start is disabled.
    """
    if not cfg.warm_start.enabled:
        return None

    section("Warm-Start Initialization")

    rep_path = Path(cfg.warm_start.representatives_path)
    step(f"Loading warm-start representatives from {rep_path}")
    rep_df = pd.read_csv(rep_path)
    smiles = rep_df["SMILES"].dropna().tolist()
    n_top = min(cfg.warm_start.n_top, len(smiles))
    smiles = smiles[:n_top]
    detail(f"Encoding {n_top} representative SMILES")
    latents = vae.encode(smiles, batch_size=16)
    detail(f"Latents shape: {latents.shape}")
    return latents


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
        warm_start_latents = _warm_start(vae, cfg)

        scheduler = build_scheduler(
            cfg.archive,
            cfg.emitter,
            cfg.run.seed,
            warm_start_latents=warm_start_latents,
            warm_start_enabled=cfg.warm_start.enabled,
        )
        start_gen = 0

    loop = CMAMAELoop(
        scheduler,
        evaluator,
        objective_cap=cfg.archive.objective_cap,
        warm_start_n_generations=(
            cfg.warm_start.n_generations if cfg.warm_start.enabled else 0
        ),
        warm_start_threshold_min=(
            cfg.warm_start.threshold_min
            if cfg.warm_start.enabled
            else cfg.archive.threshold_min
        ),
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
