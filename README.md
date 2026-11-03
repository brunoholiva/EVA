# EVA: Evolution of Viable Antibiotics

Barebones multi-population **CMA-ES** (`pycma`) search in the 256D continuous
latent space of the [ChemBed](https://link.springer.com/article/10.1186/s13321-026-01243-0)
SELFIES Transformer VAE, scored by a **TabPFN v3.5** antimicrobial classifier on
**MiniMol** 512D fingerprints.

*(Earlier `pyribs` CMA-MAE / MAP-Elites experiments are preserved on branch
`legacy/qd-pyribs`, tag `v0.1-qd-final`.)*

## Layout

| File | Role |
| :--- | :--- |
| `vae.py` | `ChemBedVAE`: 256D (or whitened-PCA) latent $\leftrightarrow$ canonical SMILES in a single pass |
| `oracle.py` | `MinimolFeaturizer`, `TabPFN` inference, SMILES deduplication (`-1.0` penalty, `0` oracle calls on repeats), and generic Murcko scaffolds |
| `solvers.py` | `MultiCMA` (`K` independent `pycma` walkers with `[cma.opts]` passthrough) and `NaiveSolver` (`N(0, I)` baseline) |
| `EVA.py` | Oracle-budget loop (`tqdm`), `CandidateStore`, CSV/JSON outputs, and CLI |
| `config.py` / `config.toml` | Minimal typed TOML configuration (`stdlib` `tomllib`) |
| `scripts/fit_latent_pca.py` | Fits oracle-free whitened PCA (`data/pca_prior.joblib`) from 150k prior-generated ChemBed molecules |
| `scripts/train_tabpfn.py` | Fits `data/predictor/tabpfn_minimol.joblib` from training SMILES |

## Setup

```bash
uv python pin 3.12
uv sync
```

## Usage

```bash
# Run default config (15 independent pycma walkers, 100k oracle-call budget)
uv run python EVA.py --config config.toml

# Multi-seed comparison: pycma vs N(0, I) naive baseline
uv run python EVA.py --config config.toml --run-name pycma_base --seeds 1,2,3
uv run python EVA.py --config config.toml --solver naive --run-name naive_base --seeds 1,2,3

# Also stream per-attempt all_evaluations.csv (~40 MB per 100k budget)
uv run python EVA.py --config config.toml --save-all
```

### Outputs (`results/<run_name>/`)

- `candidates.csv` — unique valid molecules with `P(active) >= hit_threshold`,
  ranked descending (`rank, canonical_smiles, p_active, generic_scaffold, walker_id, step, oracle_call`).
- `curve.csv` — per-step trajectory (`oracle_calls`, `validity`, `best_p_active`,
  `n_hits_05/06/08`, `unique_scaffolds_06`, `walker_spread`, `mean_norm_mean`,
  `sigma_mean`, `sigma_min`, `axis_ratio_max`, `restarts`).
- `walkers.csv` — per-walker diagnostics (`attempts`, `oracle_calls`,
  `duplicate_rate`, `best_p_active`, `n_hits_05/06/08`, `hit_share_06`,
  `unique_scaffolds_06`, `mean_norm`, `sigma`, `axis_ratio`, `restarts`,
  `steps_since_restart`).
- `summary.json` & `config.json` — final metrics, walker contribution share, scaffold cluster share, and resolved config.
- `all_evaluations.csv` — written only when `save_all = true` or `--save-all` is passed.

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run black .
```
