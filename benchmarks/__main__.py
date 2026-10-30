"""CLI: run algorithms at a fixed oracle budget over EVA's TabPFN predictor.

Usage::

    uv run python -m benchmarks --algorithms all --budget 100000 --seeds 1,2,3
    uv run python -m benchmarks --algorithms map_elites,cma_mae --budget 100000

``--algorithms`` is required: a bare invocation would otherwise mean hours of
GPU time. ``all`` expands to every entry in :data:`benchmarks.algorithms.CONFIG`;
``naive`` and ``cma_es`` (EVA's own arms) are accepted by name too. A name that
appears in both registries resolves to the sweep spec, so ``cma_mae`` is pyribs'
CMA-MAE, not :class:`~optimization.loop.CMAMAELoop` — point ``--out`` somewhere
new when sweeping next to the finished three-arm runs.

Runs are resumable: an ``<algorithm>_seed<seed>`` directory that already holds a
``summary.json`` is skipped, so a partial sweep can be continued cheaply.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.algorithms import ALGORITHMS
from benchmarks.domains import molecule_domain
from benchmarks.generators import EVA_GENERATORS, build_generator
from benchmarks.runner import run_benchmark
from config import ExperimentConfig
from EVA import load_activity_oracle, load_vae
from evaluation.activity import predict_from_features
from evaluation.evaluator import Evaluator

VALID_ALGORITHMS = set(ALGORITHMS) | set(EVA_GENERATORS)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default="config.toml", help="Experiment config TOML"
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        required=True,
        help=(
            "Comma-separated algorithm names, or 'all' for the "
            f"{len(ALGORITHMS)}-algorithm sweep. EVA arms: "
            f"{', '.join(EVA_GENERATORS)}"
        ),
    )
    parser.add_argument(
        "--seeds", type=str, default="1,2,3", help="Comma-separated integer seeds"
    )
    parser.add_argument(
        "--budget", type=int, default=100_000, help="Oracle-call budget per run"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/benchmarks"),
        help="Output root; each run lands in <out>/<algorithm>_seed<seed>/",
    )
    parser.add_argument(
        "--log-every", type=int, default=20, help="Progress print interval (steps)"
    )
    return parser.parse_args(argv)


def _absolutize(cfg: ExperimentConfig) -> ExperimentConfig:
    """Resolve config-relative paths against the current working directory."""
    root = Path.cwd()
    for attr in ("pca", "activity"):
        section = getattr(cfg, attr)
        path = Path(section.path if attr == "pca" else section.model_path)
        if not path.is_absolute():
            resolved = root / path
            if attr == "pca":
                section.path = str(resolved)
            else:
                section.model_path = str(resolved)
    return cfg


def _resolve_algorithms(spec: str) -> list[str]:
    """Expand ``all`` and fail fast on unknown names, before any model loads."""
    names = [a.strip() for a in spec.split(",") if a.strip()]
    expanded: list[str] = []
    for a in names:
        if a == "all":
            expanded.extend(ALGORITHMS)
        else:
            expanded.append(a)
    unknown = [a for a in expanded if a not in VALID_ALGORITHMS]
    if unknown:
        raise SystemExit(
            f"Unknown algorithm(s): {unknown}. Valid: {sorted(VALID_ALGORITHMS)}"
        )
    return list(dict.fromkeys(expanded))


def main(argv: list[str] | None = None) -> None:
    """Run the requested algorithms and seeds."""
    args = _parse_args(argv)
    algorithms = _resolve_algorithms(args.algorithms)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cfg = _absolutize(ExperimentConfig.from_toml(args.config))
    vae = load_vae(cfg)
    domain = molecule_domain(cfg)
    tabpfn, featurizer = load_activity_oracle(cfg)

    def make_evaluator(archive_cfg):
        """Fresh evaluator per generator: dedup state is per-instance."""
        return Evaluator(
            decode=vae.decode,
            activity=predict_from_features,
            activity_model=tabpfn,
            featurizer=featurizer,
            archive_cfg=archive_cfg,
        )

    for algorithm in algorithms:
        for seed in seeds:
            output_dir = args.out / f"{algorithm}_seed{seed}"
            if (output_dir / "summary.json").exists():
                print(f"{algorithm} seed {seed}: done, skipping")
                continue

            generator = build_generator(
                algorithm, cfg, make_evaluator, seed, domain=domain
            )
            summary = run_benchmark(
                generator,
                method=algorithm,
                seed=seed,
                budget=args.budget,
                output_dir=output_dir,
                log_every=args.log_every,
            )
            (output_dir / "config.json").write_text(
                json.dumps(asdict(cfg), indent=2, default=str)
            )
            print(
                f"{algorithm} seed {seed}: oracle {summary['oracle_calls']}, "
                f"best {summary['best_p_active']:.4f}, "
                f"{summary['attempts']} attempts, "
                f"{summary['elapsed_s']:.0f}s, "
                f"stopped_on={summary['stopped_on']}"
            )


if __name__ == "__main__":
    main()
