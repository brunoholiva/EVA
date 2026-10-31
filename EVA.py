"""Main multi-population CMA-ES search entrypoint for EVA."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from chemistry.featurizers import make_activity_featurizer
from config import VALID_SOLVERS, ExperimentConfig
from evaluation import Evaluator
from evaluation.activity import load_model as load_tabpfn
from evaluation.activity import predict_from_features
from generative import ChemBedVAE, ProjectedVAE
from optimization import CandidateStore, SearchLoop, TensorBoardLogger, build_solver
from optimization.loop import resolve_max_generations
from reporting.console import detail, make_progress_bar, section, step
from reporting.persistence import (
    load_checkpoint,
    save_candidates,
    save_checkpoint,
    save_resolved_config,
    save_summary,
)
from reporting.tables import print_results, print_step


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="EVA: Evolution of Viable Antibiotics (pycma)"
    )
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
        help="Path to checkpoint.joblib to resume a single run from",
    )
    parser.add_argument(
        "--solver",
        choices=VALID_SOLVERS,
        default=None,
        help="Override run.solver ('pycma' or 'naive')",
    )
    parser.add_argument(
        "--oracle-budget",
        type=int,
        default=None,
        help="Override run.oracle_budget (unique valid molecules scored)",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Override run.max_generations",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for multi-seed runs under <run_name>/seed_<k>/",
    )
    return parser.parse_args(argv)


def load_vae(cfg: ExperimentConfig) -> ChemBedVAE | ProjectedVAE:
    """Load the ChemBed VAE, optionally wrapped with whitened PCA projection."""
    step("Loading ChemBed VAE")
    vae = ChemBedVAE(
        model_repo_id=cfg.generative.model_repo_id,
        device=cfg.generative.device,
        latent_dim=cfg.generative.latent_dim,
    )
    if cfg.pca.enabled:
        step(f"Applying PCA projection ({cfg.pca.path})")
        vae = ProjectedVAE(vae, cfg.pca.path)
        cfg.solution_dim = vae.latent_dim
        detail(f"Working space: {vae.latent_dim} dim (was {cfg.generative.latent_dim})")
    else:
        cfg.solution_dim = vae.latent_dim
        detail(f"Working space: {vae.latent_dim} dim (raw ChemBed latent space)")
    return vae


def load_activity_oracle(cfg: ExperimentConfig):
    """Load the TabPFN activity model and MiniMol featurizer."""
    step("Loading activity oracle")
    tabpfn = load_tabpfn(
        path=Path(cfg.activity.model_path),
        device=cfg.activity.device,
    )
    featurizer = make_activity_featurizer(cfg.activity.representation)
    return tabpfn, featurizer


def run_single_seed(
    cfg: ExperimentConfig,
    *,
    vae: ChemBedVAE | ProjectedVAE,
    tabpfn,
    featurizer,
    output_dir: Path,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """Execute one search run (fresh or resumed) and persist all artifacts."""
    tb_logger = TensorBoardLogger(cfg.tensorboard, output_dir)
    tb_logger.log_config(cfg)

    resuming = bool(resume_from)
    if resume_from:
        section(f"Resuming {cfg.run.solver} (seed {cfg.run.seed})")
        ckpt = load_checkpoint(resume_from)
        evaluator = Evaluator(
            decode=vae.decode,
            activity=predict_from_features,
            activity_model=tabpfn,
            featurizer=featurizer,
            seen_smiles=ckpt.seen_smiles,
        )
        loop = SearchLoop(
            ckpt.solver,
            evaluator,
            ckpt.candidates,
            solver_name=ckpt.solver_name,
            seed=ckpt.seed,
            step=ckpt.step,
            attempts=ckpt.attempts,
            oracle_calls=ckpt.oracle_calls,
            duplicate_attempts=ckpt.duplicate_attempts,
            invalid_attempts=ckpt.invalid_attempts,
            elapsed_offset_s=ckpt.elapsed_s,
        )
    else:
        section(f"Running {cfg.run.solver} (seed {cfg.run.seed})")
        solver = build_solver(
            solver_name=cfg.run.solver,
            solution_dim=cfg.solution_dim,
            cfg=cfg.cma,
            seed=cfg.run.seed,
        )
        evaluator = Evaluator(
            decode=vae.decode,
            activity=predict_from_features,
            activity_model=tabpfn,
            featurizer=featurizer,
        )
        candidates = CandidateStore(hit_threshold=cfg.run.hit_threshold)
        loop = SearchLoop(
            solver,
            evaluator,
            candidates,
            solver_name=cfg.run.solver,
            seed=cfg.run.seed,
        )

    oracle_budget = cfg.run.oracle_budget
    effective_max_gens = resolve_max_generations(
        oracle_budget, cfg.run.max_generations, loop.solver.batch_size
    )
    use_oracle_bar = oracle_budget > 0
    bar_label = (
        f"{cfg.run.solver} (oracle calls)"
        if use_oracle_bar
        else f"{cfg.run.solver} (steps)"
    )
    bar_total = oracle_budget if use_oracle_bar else effective_max_gens
    progress, task_id = make_progress_bar(bar_label, bar_total)

    try:
        with progress:

            def _on_step(snap):
                completed = (
                    min(snap.oracle_calls, oracle_budget)
                    if use_oracle_bar
                    else snap.step + 1
                )
                progress.update(task_id, completed=completed)
                tb_logger.log_step(snap)

            def _on_log(snap):
                print_step(snap, oracle_budget=oracle_budget)
                if snap.step > 0 and snap.result.timings.decode > 0:
                    t = snap.result.timings
                    detail(
                        f"decode={t.decode:.1f}s valid={t.validity:.1f}s "
                        f"feat+pred={t.featurize_predict:.1f}s "
                        f"solver={t.solver_ops:.2f}s"
                    )
                save_checkpoint(loop, evaluator.seen_smiles, output_dir, quiet=True)
                save_candidates(loop.candidates, output_dir, quiet=True)

            summary = loop.run(
                output_dir=output_dir,
                oracle_budget=oracle_budget,
                max_generations=cfg.run.max_generations,
                save_all_evaluations=cfg.run.save_all_evaluations,
                log_every=cfg.run.log_every,
                resuming=resuming,
                on_step=_on_step,
                on_log=_on_log,
            )

        print_results(loop.candidates, summary)
        save_checkpoint(loop, evaluator.seen_smiles, output_dir)
        save_candidates(loop.candidates, output_dir)
        save_summary(summary, output_dir)
        save_resolved_config(cfg, output_dir)
        tb_logger.log_summary(summary)
        return summary
    finally:
        tb_logger.close()


def _apply_cli_overrides(
    cfg: ExperimentConfig, args: argparse.Namespace
) -> ExperimentConfig:
    """Return a copy of *cfg* with CLI overrides applied."""
    run_cfg = cfg.run
    if args.solver is not None:
        run_cfg = replace(run_cfg, solver=args.solver)
    if args.oracle_budget is not None:
        run_cfg = replace(run_cfg, oracle_budget=args.oracle_budget)
    if args.max_generations is not None:
        run_cfg = replace(run_cfg, max_generations=args.max_generations)
    return replace(cfg, run=run_cfg)


def main(argv: list[str] | None = None) -> None:
    """Entry point for EVA."""
    args = _parse_args(argv)

    section("EVA")
    step(f"Loading config from {args.config}")
    cfg = _apply_cli_overrides(ExperimentConfig.from_toml(args.config), args)

    seeds: list[int] | None = None
    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.resume:
            raise ValueError("--resume cannot be combined with --seeds")

    vae = load_vae(cfg)
    tabpfn, featurizer = load_activity_oracle(cfg)

    base_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    if seeds is None:
        resume_from = args.resume or cfg.run.resume_from or None
        run_single_seed(
            cfg,
            vae=vae,
            tabpfn=tabpfn,
            featurizer=featurizer,
            output_dir=base_dir,
            resume_from=resume_from,
        )
        return

    for seed in seeds:
        seed_dir = base_dir / f"seed_{seed}"
        if (seed_dir / "summary.json").exists():
            detail(f"Skipping seed {seed}: {seed_dir / 'summary.json'} already exists")
            continue
        seed_cfg = replace(cfg, run=replace(cfg.run, seed=seed, resume_from=""))
        run_single_seed(
            seed_cfg,
            vae=vae,
            tabpfn=tabpfn,
            featurizer=featurizer,
            output_dir=seed_dir,
        )


if __name__ == "__main__":
    main()
