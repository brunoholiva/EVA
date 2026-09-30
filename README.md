# EVA - Evolution of Viable Antibiotics

Antibiotics are extremely diverse, and conventional optimization frequently
exploits data and predictor-model biases. EVA uses quality-diversity search:
molecules compete only within the same niche of a configurable behavior space,
instead of chasing one optimal solution.  While diverse, in a real chemical space, antibiotics tend to cluster far apart from each other, so we need to balance exploration vs exploitation very carefully.

- **Search:** The search uses PyRibs emitters to explore/exploit the latent space of the [ChemBed](https://link.springer.com/article/10.1186/s13321-026-01243-0) VAE, so evolution operates on a continuous learned molecular representation. Specifically, we are employing CMA-MAE, where each of the 16 emitters evolves its own CMA-ES population of latent vectors, which the VAE decodes into SMILES.

- **Scoring:** a **TabPFN v3.5** classifier trained on the public
  [GNEtolC](https://github.com/Genentech/gneprop) dataset predicts P(active). This functions as the reward for our  search, therefore, molecules are only as good as the model and data that were used to train.
  The activity representation is selected in `config.toml` with
  `activity.representation`: `rdkit2d` or `chemeleon`. Model performance
  metrics and evaluation are private for now, and will be added in future updates.
- **Behavior space:** These will be the primary drivers for the notion of diversity in the search. They are not guaranteed to accurately reflect _structural_ diversity, as diverisity in this case are in the terms of the dimension itself. For now, any combination of the dimensions are available:
  `logp`, `tpsa`, `mw`, `fsp3`, `num_rotb`, `num_rings`, `balabanj`,
  `vsa_estate2`, `bcut2d_logplow`. 
  For now, the ranges are set from the observed support of the training actives, so no resolution is spent on empty space, and a cell is only replaced by a better-scoring molecule rather than by a different one.

Most generation parameters can be easily changed in `config.toml`

## Requirements

The environment is managed with [uv](https://docs.astral.sh/uv/), which
resolves the full dependency tree and pins it in `uv.lock`:

```bash
uv python pin 3.12
uv sync
```

The conda environment is kept as a fallback. `environment.yaml` pins the same
versions, but there `chembed` has to be installed by hand:

```bash
micromamba env create -f environment.yaml
micromamba activate eva
pip install --no-deps chembed
```

The TabPFN model, and PCA projection must be trained/fitted first. The GNEtolC
features are not part of this repo, so point `--features-dir` at your local copy:

```bash
python scripts/train_tabpfn.py \
  --features-dir /path/to/GNEtolC/rdkit2d \
  --out data/predictor/tabpfn_rdkit2d.joblib \
  --representation rdkit2d
python scripts/train_tabpfn.py \
  --features-dir /path/to/GNEtolC/chemeleon \
  --out data/predictor/tabpfn_chemeleon.joblib \
  --representation chemeleon
python scripts/fit_latent_pca.py
```

The trained GNEtolC models are `data/predictor/tabpfn_rdkit2d.joblib` and
`data/predictor/tabpfn_chemeleon.joblib`. CheMeleon inference requires the
`chemprop` dependency and its checkpoint, downloaded on first use.

## Run

```bash
uv run python EVA.py --config config.toml
```

- All experiment parameters live in `config.toml` (see `config.py`). Set
  `output.run_name` before each experiment, as a run overwrites the results
  directory of the same name.
- Resume an interrupted run: `uv run python EVA.py --config config.toml --resume results/<run>/scheduler.joblib`
- Outputs land in `results/<run_name>/`: `all_evaluations.csv` with every
  molecule that was scored, `result_archive.csv` with the best molecule per
  cell, `scheduler.joblib` for resuming, and `heatmap.png`. Metrics are
  streamed to `tensorboard/` inside the same directory.
  
To resume an interrupted run:

```bash
uv run python EVA.py --resume path/to/scheduler.joblib
```

## TODO

- Make the predictor public
- Release the archives and the best candidates from each run