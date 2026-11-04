"""EVA: Barebones multi-population CMA-ES search for viable antibiotics."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import VALID_SOLVERS, Config
from oracle import Oracle, circle_metrics, generic_scaffold
from solvers import Solver, build_solver
from vae import ChemBedVAE

CANDIDATE_COLUMNS = [
    "rank",
    "canonical_smiles",
    "p_active",
    "generic_scaffold",
    "walker_id",
    "step",
    "oracle_call",
]

CURVE_COLUMNS = [
    "solver",
    "seed",
    "step",
    "elapsed_s",
    "attempts",
    "oracle_calls",
    "duplicate_attempts",
    "invalid_attempts",
    "validity",
    "best_p_active",
    "n_hits_05",
    "n_hits_06",
    "n_hits_08",
    "unique_scaffolds_06",
    "walker_spread",
    "mean_norm_mean",
    "sigma_mean",
    "sigma_min",
    "axis_ratio_max",
    "restarts",
]

WALKER_COLUMNS = [
    "walker_id",
    "popsize",
    "steps",
    "attempts",
    "oracle_calls",
    "duplicate_attempts",
    "invalid_attempts",
    "duplicate_rate",
    "best_p_active",
    "n_hits_05",
    "n_hits_06",
    "n_hits_08",
    "hit_share_06",
    "unique_scaffolds_06",
    "mean_norm",
    "sigma",
    "axis_ratio",
    "restarts",
    "steps_since_restart",
]

ATTEMPT_COLUMNS = [
    "solver",
    "seed",
    "step",
    "attempt",
    "walker_id",
    "canonical_smiles",
    "valid",
    "duplicate",
    "oracle_called",
    "p_active",
]

_MAX_STEP_MULTIPLIER = 20


@dataclass
class _WalkerStats:
    """Per-walker evaluation counters accumulated during ``run_search``."""

    attempts: int = 0
    oracle_calls: int = 0
    duplicate_attempts: int = 0
    invalid_attempts: int = 0
    best_p_active: float = 0.0
    n_hits_05: int = 0
    n_hits_06: int = 0
    n_hits_08: int = 0
    scaffolds_06: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Candidate:
    """Unique hit stored when ``P(active) >= hit_threshold``."""

    canonical_smiles: str
    p_active: float
    generic_scaffold: str
    walker_id: int
    step: int
    oracle_call: int


class CandidateStore:
    """Accumulates unique hits and tracks generic Murcko scaffold diversity."""

    def __init__(self, hit_threshold: float = 0.5) -> None:
        self.hit_threshold = hit_threshold
        self._hits: dict[str, Candidate] = {}
        self.hits_06: list[tuple[str, float, int]] = []
        self._seen_06: set[str] = set()
        self.best_p_active: float = 0.0
        self.n_hits_05: int = 0
        self.n_hits_06: int = 0
        self.n_hits_08: int = 0
        self._scaffolds_06: Counter[str] = Counter()

    def __len__(self) -> int:
        """Return the number of saved candidates."""
        return len(self._hits)

    @property
    def unique_scaffolds_06(self) -> int:
        """Unique generic scaffolds among ``P >= 0.6`` molecules."""
        return len(self._scaffolds_06)

    @property
    def largest_scaffold_frac_06(self) -> float:
        """Share of ``P >= 0.6`` hits in the single largest scaffold cluster."""
        if self.n_hits_06 == 0 or not self._scaffolds_06:
            return 0.0
        return float(self._scaffolds_06.most_common(1)[0][1] / self.n_hits_06)

    def add(
        self,
        smiles: str,
        p_active: float,
        walker_id: int,
        step: int,
        oracle_call: int,
    ) -> str | None:
        """Record one first-seen valid oracle-scored molecule; return scaffold if computed."""
        if smiles in self._hits or smiles in self._seen_06:
            return None
        if p_active > self.best_p_active:
            self.best_p_active = float(p_active)
        if p_active >= 0.5:
            self.n_hits_05 += 1
        if p_active >= 0.8:
            self.n_hits_08 += 1

        keep = p_active >= self.hit_threshold
        in_06 = p_active >= 0.6
        if not keep and not in_06:
            return None

        scaff = generic_scaffold(smiles)
        if in_06:
            self._seen_06.add(smiles)
            self.hits_06.append((smiles, float(p_active), int(oracle_call)))
            self.n_hits_06 += 1
            self._scaffolds_06[scaff] += 1
        if keep:
            self._hits[smiles] = Candidate(
                canonical_smiles=smiles,
                p_active=float(p_active),
                generic_scaffold=scaff,
                walker_id=int(walker_id),
                step=int(step),
                oracle_call=int(oracle_call),
            )
        return scaff

    def ranked(self, top_n: int | None = None) -> list[Candidate]:
        """Return hits sorted descending by ``p_active``."""
        ordered = sorted(
            self._hits.values(), key=lambda c: (-c.p_active, c.oracle_call)
        )
        return ordered[:top_n] if top_n is not None else ordered

    def to_dataframe(self) -> pd.DataFrame:
        """Export ranked hits as a DataFrame."""
        items = self.ranked()
        if not items:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)
        rows = [
            {
                "rank": i,
                "canonical_smiles": c.canonical_smiles,
                "p_active": c.p_active,
                "generic_scaffold": c.generic_scaffold,
                "walker_id": c.walker_id,
                "step": c.step,
                "oracle_call": c.oracle_call,
            }
            for i, c in enumerate(items, start=1)
        ]
        return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def run_search(
    *,
    solver: Solver,
    decode_fn,
    oracle: Oracle,
    output_dir: Path,
    solver_name: str,
    seed: int,
    oracle_budget: int,
    hit_threshold: float = 0.5,
    resample_duplicates: int = 0,
    save_all: bool = False,
    show_progress: bool = True,
) -> tuple[dict[str, Any], CandidateStore]:
    """Execute ``ask -> decode -> score -> tell`` until *oracle_budget* is spent."""
    output_dir.mkdir(parents=True, exist_ok=True)
    min_steps = max(1, int(np.ceil(oracle_budget / max(1, solver.batch_size))))
    max_steps = max(1000, _MAX_STEP_MULTIPLIER * min_steps)

    store = CandidateStore(hit_threshold=hit_threshold)
    walker_stats: dict[int, _WalkerStats] = {
        ws.walker_id: _WalkerStats() for ws in solver.walker_states()
    }
    attempts = 0
    oracle_calls = 0
    dup_attempts = 0
    inv_attempts = 0
    step = 0
    t0 = time.time()

    f_all = (
        (output_dir / "all_evaluations.csv").open("w", newline="") if save_all else None
    )
    pbar = tqdm(
        total=oracle_budget,
        desc=f"{solver_name} (seed {seed})",
        unit="mol",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    try:
        with (output_dir / "curve.csv").open("w", newline="") as f_curve:
            w_curve = csv.writer(f_curve)
            w_curve.writerow(CURVE_COLUMNS)

            w_all = None
            if f_all is not None:
                w_all = csv.writer(f_all)
                w_all.writerow(ATTEMPT_COLUMNS)

            while oracle_calls < oracle_budget and step < max_steps:
                z, walker_ids = solver.ask()
                canonical = decode_fn(z)

                if resample_duplicates > 0 and hasattr(solver, "resample_batch"):
                    for _ in range(resample_duplicates):
                        batch_seen: set[str] = set()
                        dup_indices: list[int] = []
                        for i, smi in enumerate(canonical):
                            if smi is not None:
                                if smi in oracle.seen or smi in batch_seen:
                                    dup_indices.append(i)
                                else:
                                    batch_seen.add(smi)
                        if not dup_indices:
                            break
                        solver.resample_batch(z, dup_indices)
                        new_sub = decode_fn(z[dup_indices])
                        for orig_i, new_smi in zip(dup_indices, new_sub):
                            canonical[orig_i] = new_smi

                batch = oracle.score(canonical)
                solver.tell(z, batch.objectives)

                prev_calls = oracle_calls
                step_scored = 0
                step_dup = 0
                for i, smi in enumerate(batch.canonical):
                    valid = smi is not None
                    charged = bool(batch.oracle_mask[i])
                    dup = valid and not charged
                    pa = float(batch.p_active[i]) if charged else float("nan")
                    wid = int(walker_ids[i])

                    if w_all is not None:
                        w_all.writerow(
                            [
                                solver_name,
                                seed,
                                step,
                                attempts,
                                wid,
                                smi,
                                valid,
                                dup,
                                charged,
                                pa,
                            ]
                        )

                    ws = walker_stats.get(wid)
                    attempts += 1
                    if ws is not None:
                        ws.attempts += 1
                    if not valid:
                        inv_attempts += 1
                        if ws is not None:
                            ws.invalid_attempts += 1
                    elif dup:
                        step_dup += 1
                        dup_attempts += 1
                        if ws is not None:
                            ws.duplicate_attempts += 1
                    else:
                        step_scored += 1
                        oracle_calls += 1
                        assert smi is not None
                        scaff = store.add(smi, pa, wid, step, oracle_calls)
                        if ws is not None:
                            ws.oracle_calls += 1
                            if pa > ws.best_p_active:
                                ws.best_p_active = pa
                            if pa >= 0.5:
                                ws.n_hits_05 += 1
                            if pa >= 0.6:
                                ws.n_hits_06 += 1
                                if scaff is not None:
                                    ws.scaffolds_06.add(scaff)
                            if pa >= 0.8:
                                ws.n_hits_08 += 1

                assert step_scored == batch.n_oracle_calls
                assert step_dup == batch.n_duplicates

                validity = (attempts - inv_attempts) / attempts if attempts else 0.0
                st = solver.stats()
                elapsed = time.time() - t0

                w_curve.writerow(
                    [
                        solver_name,
                        seed,
                        step,
                        elapsed,
                        attempts,
                        oracle_calls,
                        dup_attempts,
                        inv_attempts,
                        validity,
                        store.best_p_active,
                        store.n_hits_05,
                        store.n_hits_06,
                        store.n_hits_08,
                        store.unique_scaffolds_06,
                        st.walker_spread,
                        st.mean_norm_mean,
                        st.sigma_mean,
                        st.sigma_min,
                        st.axis_ratio_max,
                        st.restarts,
                    ]
                )
                f_curve.flush()
                if f_all is not None:
                    f_all.flush()

                delta = min(oracle_calls, oracle_budget) - min(
                    prev_calls, oracle_budget
                )
                if delta > 0:
                    pbar.update(delta)
                pbar.set_postfix(
                    best=f"{store.best_p_active:.3f}",
                    h60=store.n_hits_06,
                    scaff60=store.unique_scaffolds_06,
                    val=f"{validity:.0%}",
                    sig=f"{st.sigma_mean:.2f}",
                    rst=st.restarts,
                )
                step += 1
    finally:
        pbar.close()
        if f_all is not None:
            f_all.close()

    store.to_dataframe().to_csv(output_dir / "candidates.csv", index=False)

    walker_rows = []
    for wst in solver.walker_states():
        cnt = walker_stats.get(wst.walker_id, _WalkerStats())
        dup_rate = cnt.duplicate_attempts / cnt.attempts if cnt.attempts else 0.0
        share_06 = cnt.n_hits_06 / store.n_hits_06 if store.n_hits_06 else 0.0
        walker_rows.append(
            {
                "walker_id": wst.walker_id,
                "popsize": wst.popsize,
                "steps": wst.steps,
                "attempts": cnt.attempts,
                "oracle_calls": cnt.oracle_calls,
                "duplicate_attempts": cnt.duplicate_attempts,
                "invalid_attempts": cnt.invalid_attempts,
                "duplicate_rate": dup_rate,
                "best_p_active": cnt.best_p_active,
                "n_hits_05": cnt.n_hits_05,
                "n_hits_06": cnt.n_hits_06,
                "n_hits_08": cnt.n_hits_08,
                "hit_share_06": share_06,
                "unique_scaffolds_06": len(cnt.scaffolds_06),
                "mean_norm": wst.mean_norm,
                "sigma": wst.sigma,
                "axis_ratio": wst.axis_ratio,
                "restarts": wst.restarts,
                "steps_since_restart": wst.steps_since_restart,
            }
        )
    pd.DataFrame(walker_rows, columns=WALKER_COLUMNS).to_csv(
        output_dir / "walkers.csv", index=False
    )

    n_walkers = len(walker_rows)
    active_walkers_06 = sum(1 for r in walker_rows if r["n_hits_06"] > 0)
    top_walker_share_06 = max((r["hit_share_06"] for r in walker_rows), default=0.0)
    walkers_restarted = sum(1 for r in walker_rows if r["restarts"] > 0)

    cm = circle_metrics(store.hits_06, oracle_budget=oracle_budget)
    st = solver.stats()
    validity = (attempts - inv_attempts) / attempts if attempts else 0.0
    summary = {
        "solver": solver_name,
        "seed": seed,
        "oracle_budget": oracle_budget,
        "steps": step,
        "attempts": attempts,
        "oracle_calls": oracle_calls,
        "duplicate_attempts": dup_attempts,
        "invalid_attempts": inv_attempts,
        "validity": validity,
        "best_p_active": store.best_p_active,
        "n_candidates_saved": len(store),
        "n_hits_05": store.n_hits_05,
        "n_hits_06": store.n_hits_06,
        "n_hits_08": store.n_hits_08,
        "unique_scaffolds_06": store.unique_scaffolds_06,
        "largest_scaffold_frac_06": store.largest_scaffold_frac_06,
        "n_circles_06": cm.n_circles,
        "diverse_top10_06": cm.diverse_top_k,
        "diverse_auc_top10_06": cm.diverse_auc_top_k,
        "n_walkers": n_walkers,
        "active_walkers_06": active_walkers_06,
        "top_walker_share_06": top_walker_share_06,
        "walkers_restarted": walkers_restarted,
        "walker_spread": st.walker_spread,
        "mean_norm_mean": st.mean_norm_mean,
        "sigma_mean": st.sigma_mean,
        "sigma_min": st.sigma_min,
        "axis_ratio_max": st.axis_ratio_max,
        "restarts": st.restarts,
        "stopped_on": "oracle_budget" if oracle_calls >= oracle_budget else "max_steps",
        "elapsed_s": time.time() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary, store


def _print_summary(
    summary: dict[str, Any], store: CandidateStore, out_dir: Path
) -> None:
    """Print a concise plain-text summary after a run completes."""
    print(
        f"\n[{summary['solver']} seed={summary['seed']}] "
        f"oracle={summary['oracle_calls']:,} ({summary['attempts']:,} attempts, "
        f"valid={summary['validity']:.1%}, {summary['elapsed_s']:.1f}s) -> {out_dir}"
    )
    print(
        f"  best={summary['best_p_active']:.4f} | "
        f"hits>=0.5: {summary['n_hits_05']:,} | "
        f"hits>=0.6: {summary['n_hits_06']:,} | "
        f"hits>=0.8: {summary['n_hits_08']:,} | "
        f"scaffolds>=0.6: {summary['unique_scaffolds_06']:,} "
        f"(max cluster {summary['largest_scaffold_frac_06']:.1%})"
    )
    print(
        f"  diversity (P>=0.6, d>0.75): circles={summary['n_circles_06']:,} | "
        f"div_top10={summary['diverse_top10_06']:.4f} | "
        f"div_auc_top10={summary['diverse_auc_top10_06']:.4f}"
    )
    if summary["n_walkers"] > 0:
        print(
            f"  walkers: {summary['active_walkers_06']}/{summary['n_walkers']} "
            f"active (hits>=0.6), top share={summary['top_walker_share_06']:.1%}, "
            f"restarts={summary['restarts']} "
            f"({summary['walkers_restarted']}/{summary['n_walkers']} walkers), "
            f"||mean||={summary['mean_norm_mean']:.2f}"
        )
    for rank, c in enumerate(store.ranked(top_n=5), start=1):
        print(
            f"  #{rank} P={c.p_active:.4f} [w{c.walker_id} @{c.oracle_call:,}] "
            f"{c.canonical_smiles}  (scaffold: {c.generic_scaffold})"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags."""
    p = argparse.ArgumentParser(description="EVA: Barebones CMA-ES Molecule Search")
    p.add_argument("--config", default="config.toml", help="Path to TOML config")
    p.add_argument("--solver", choices=VALID_SOLVERS, default=None)
    p.add_argument("--oracle-budget", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds")
    p.add_argument(
        "--save-all",
        action="store_true",
        help="Also stream per-attempt all_evaluations.csv",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load models once and execute single-seed or multi-seed search."""
    args = _parse_args(argv)
    cfg = Config.from_toml(args.config)

    run_cfg = cfg.run
    if args.solver is not None:
        run_cfg = replace(run_cfg, solver=args.solver)
    if args.oracle_budget is not None:
        run_cfg = replace(run_cfg, oracle_budget=args.oracle_budget)
    if args.run_name is not None:
        run_cfg = replace(run_cfg, run_name=args.run_name)
    if args.save_all:
        run_cfg = replace(run_cfg, save_all=True)
    cfg = replace(cfg, run=run_cfg)

    seeds = (
        [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else None
    )

    print(f"Loading ChemBed VAE ({cfg.models.vae_repo}) on {cfg.models.device}")
    vae = ChemBedVAE(
        repo_id=cfg.models.vae_repo,
        device=cfg.models.device,
        pca_path=cfg.models.pca_path or None,
    )
    space_note = f"PCA: {cfg.models.pca_path}" if cfg.models.pca_path else "raw VAE"
    print(f"Search space: {vae.latent_dim}D ({space_note})")
    print(f"Loading MiniMol + TabPFN oracle ({cfg.models.predictor_path})")
    tabpfn_model_oracle = Oracle.from_pretrained(
        cfg.models.predictor_path, device=cfg.models.device
    )
    featurizer = tabpfn_model_oracle._featurizer
    predict_fn = tabpfn_model_oracle._predict_fn

    root = Path(cfg.run.output_dir) / cfg.run.run_name
    targets = (
        [(cfg.run.seed, root)]
        if seeds is None
        else [(s, root / f"seed_{s}") for s in seeds]
    )

    for seed, out_dir in targets:
        if seeds is not None and (out_dir / "summary.json").exists():
            print(f"Skipping seed {seed}: {out_dir / 'summary.json'} exists")
            continue

        seed_cfg = replace(cfg, run=replace(cfg.run, seed=seed))
        solver = build_solver(seed_cfg.run.solver, vae.latent_dim, seed_cfg.cma, seed)
        oracle = Oracle(featurizer, predict_fn)

        summary, store = run_search(
            solver=solver,
            decode_fn=vae.decode,
            oracle=oracle,
            output_dir=out_dir,
            solver_name=seed_cfg.run.solver,
            seed=seed,
            oracle_budget=seed_cfg.run.oracle_budget,
            hit_threshold=seed_cfg.run.hit_threshold,
            resample_duplicates=seed_cfg.cma.resample_duplicates,
            save_all=seed_cfg.run.save_all,
        )
        (out_dir / "config.json").write_text(json.dumps(asdict(seed_cfg), indent=2))
        _print_summary(summary, store, out_dir)


if __name__ == "__main__":
    main()
