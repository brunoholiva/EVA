# EVA — Evolution of Viable Antibiotics

Antibiotics are extremely diverse, and conventional optimization frequently
exploits data and predictor-model biases. EVA uses quality-diversity search:
molecules compete only within the same niche of a configurable behavior space,
instead of chasing one optimal solution.

- **Search:** CMA-MAE (pyribs) over the latent space of the **ChemBed** VAE,
  so evolution operates on a continuous molecular representation.
- **Scoring:** a **TabPFN** classifier trained on the public
  [GNEtolC](https://github.com/Genentech/gneprop) dataset predicts P(active)
  from a feature union of:
  - ECFP **512-bit** fingerprints
  - **167-bit** MACCS keys
  - **200** molecular descriptors (Descriptastorus)
- **Behavior space:** any combination of dimensions in `config.toml`:
  `ad`, `logp`, `tpsa`, `mw`, `fsp3`, `num_rotb`, `num_rings`, `balabanj`,
  `vsa_estate2`, `bcut2d_logplow`.

## Requirements

Create the `eva` micromamba environment from `environment.yaml`, then install
ChemBed manually (it must not pull its own tensorflow/torch):

```bash
micromamba env create -f environment.yaml
micromamba activate eva
pip install --no-deps chembed
```

The TabPFN model, AD model, and PCA projection must be trained/fitted first:

```bash
python scripts/train_tabpfn.py
python scripts/build_ad_model.py
python scripts/fit_latent_pca.py
```

## Run

```bash
micromamba run -n eva2 python EVA.py --config config.toml
```

- All experiment parameters live in `config.toml` (see `config.py`).
- Resume an interrupted run: `python EVA.py --config config.toml --resume results/<run>/scheduler.joblib`
- Outputs land in `results/<run_name>/` (`archive.joblib` + `archive.csv`),
  with metrics streamed to `tensorboard/`.


## TODO

- Make the predictor public