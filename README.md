# EVA — Evolution of Viable Antibiotics

Antibiotics span structurally distant chemical families, and single-population
optimization frequently collapses onto one chemotype or exploits predictor biases.
EVA searches the continuous learned latent space of the
[ChemBed](https://link.springer.com/article/10.1186/s13321-026-01243-0) SELFIES
Transformer VAE using multiple independent **CMA-ES** populations (`pycma`),
scored by a **TabPFN v3.5** antimicrobial activity classifier on **MiniMol**
molecular fingerprints.

*(Historical Quality-Diversity / `pyribs` CMA-MAE and MAP-Elites experiments are
archived on the `legacy/qd-pyribs` branch at tag `v0.1-qd-final`.)*

- **Search (`optimization/cma.py`):** `n_populations` independent
  `cma.CMAEvolutionStrategy` walkers explore ChemBed's 256D latent space in
  parallel (optionally projected through a whitened 95D PCA via `[pca]`). All
  populations are batched together on every step for single-pass GPU decoding
  and oracle inference. Any native `pycma` option (`minstd`, `CMA_diagonal`,
  `CSA_dampfac`, `tolflatfitness`, etc.) can be passed directly under
  `[cma.opts]` in `config.toml`, and `"naive"` (`N(0, I)` random-prior sampling)
  is built in as a zero-feedback baseline.
- **Scoring (`evaluation/` + `chemistry/`):** Candidate latent vectors decode to
  SMILES, pass RDKit validity and canonicalization checks, and are globally
  deduplicated so duplicate molecules cost **zero** oracle calls (and receive
  `-1.0` objective penalty to push walkers off discrete VAE plateaus). First-seen
  valid molecules are featurized with `minimol-onnx` (512D) and scored by
  `TabPFNClassifier.predict_proba()` for `P(active)`.
- **Candidate & Scaffold Tracking (`optimization/candidates.py`):** Unique hits
  exceeding `run.hit_threshold` are stored and clustered by generic Murcko
  scaffolds (`chemistry/scaffolds.py`) to monitor structural diversity across
  oracle calls without adding any diversity or similarity term to the objective.

## Environment Setup

Dependencies are managed exclusively with [uv](https://docs.astral.sh/uv/) and
pinned in `uv.lock`:

```bash
uv python pin 3.12
uv sync
```

MiniMol runs inside EVA's environment via `minimol-onnx` (frozen ONNX graph +
pure-Python featurizer). Note that `rdkit==2023.9.6`, `numpy==1.26.4`, and
`datamol==0.12.5` are deliberately pinned for numerical reproducibility of the
trained `TabPFN` model (`data/predictor/tabpfn_minimol.joblib`). If the
environment changes, retrain the classifier:

```bash
uv run python scripts/train_tabpfn.py \
  --csv data/predictor/predictor_training_data.csv \
  --out data/predictor/tabpfn_minimol.joblib \
  --representation minimol
```

If enabling `[pca]` (`enabled = true`), fit the whitened latent PCA transform:

```bash
uv run python scripts/fit_latent_pca.py
```

## Running EVA

Single run (reads `config.toml`, stops at `run.oracle_budget` and/or
`run.max_generations`):

```bash
uv run python EVA.py --config config.toml
```

CLI overrides for multi-seed oracle-budget comparisons or baselines:

```bash
# Multi-seed pycma sweep (writes under results/<run_name>/seed_<k>/)
uv run python EVA.py --config config.toml --oracle-budget 100000 --seeds 1,2,3

# Unconditional N(0, I) random-prior baseline at the same oracle budget
uv run python EVA.py --config config.toml --solver naive --oracle-budget 100000 --seeds 1,2,3
```

Resume an interrupted single run from its checkpoint:

```bash
uv run python EVA.py --config config.toml --resume results/<run_name>/checkpoint.joblib
```

### Outputs (`results/<run_name>/`)

- `candidates.csv` — unique valid molecules with `P(active) >= hit_threshold`,
  ranked descending with `canonical_smiles`, `p_active`, `generic_scaffold`,
  `walker_id`, `step`, and `oracle_call`.
- `curve.csv` — per-step cumulative trajectory (`oracle_calls`, `validity`,
  `best_p_active`, `n_hits_05`, `n_hits_06`, `n_hits_08`,
  `unique_scaffolds_06`, `walker_spread`, `total_restarts`).
- `summary.json` & `config.json` — final metrics, generic scaffold diversity
  summary, stopping reason, and resolved configuration.
- `checkpoint.joblib` — full solver (`CMAWalker`s + isolated RNG states),
  deduplication set, and candidate store for exact `--resume`.
- `all_evaluations.csv` — per-attempt log (when `run.save_all_evaluations = true`).
- `tensorboard/` — step-indexed CMA diagnostics, oracle-call-indexed hit/scaffold
  curves, and rendered top-molecule grids.

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run black .
```
