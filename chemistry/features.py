"""Molecular featurization for activity models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from rdkit import Chem

from logs import suppress_joblib_warnings

suppress_joblib_warnings()


N_RDKIT_DESCRIPTORS: int = 200


def _chunk_list(lst: list, chunks: int) -> list[list]:
    """Split a list into roughly equal parts."""
    k, m = divmod(len(lst), chunks)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(chunks)]


def _compute_rdkit2d_batch(smiles_batch: list[str]) -> np.ndarray:
    """Compute normalized RDKit2D descriptors in one worker process."""
    from descriptastorus.descriptors import rdNormalizedDescriptors

    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    values = []
    for i, smi in enumerate(smiles_batch):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at position {i}")
        result = generator.processMol(mol, smi, internalParsing=True)
        if not result[0]:
            raise ValueError(f"RDKit2D failed at position {i}")
        values.append(np.nan_to_num(result[1:], nan=0.0))
    return np.asarray(values, dtype=np.float32).reshape(-1, N_RDKIT_DESCRIPTORS)


class RDKit2DFeaturizer:
    """Convert SMILES to the 200 normalized RDKit2D descriptors."""

    n_features = N_RDKIT_DESCRIPTORS

    def __init__(self, n_jobs: int = -1) -> None:
        self.n_jobs = n_jobs

    def transform(self, smiles: list[str]) -> np.ndarray:
        """Return normalized RDKit2D descriptors for *smiles*."""
        if not smiles:
            return np.empty((0, self.n_features), dtype=np.float32)

        njobs = min(effective_n_jobs(self.n_jobs), 16)
        batches = _chunk_list(smiles, njobs * 2)
        parts = Parallel(n_jobs=njobs, return_as="generator")(
            delayed(_compute_rdkit2d_batch)(batch) for batch in batches
        )
        return np.vstack(list(parts))


class CheMeleonFeaturizer:
    """Generate pretrained CheMeleon molecular embeddings."""

    def __init__(self, device: str = "cpu", batch_size: int = 256) -> None:
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        import torch
        from chemprop import featurizers, nn
        from chemprop.models import MPNN
        from chemprop.nn import RegressionFFN

        checkpoint_dir = Path.home() / ".chemprop"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint = checkpoint_dir / "chemeleon_mp.pt"
        if not checkpoint.exists():
            from urllib.request import urlretrieve

            urlretrieve(
                "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                checkpoint,
            )

        state = torch.load(checkpoint, weights_only=True)
        message_passing = nn.BondMessagePassing(**state["hyper_parameters"])
        message_passing.load_state_dict(state["state_dict"])
        model = MPNN(
            message_passing=message_passing,
            agg=nn.MeanAggregation(),
            predictor=RegressionFFN(input_dim=message_passing.output_dim),
        )
        model.eval().to(self.device)
        self._model = (model, featurizers.SimpleMoleculeMolGraphFeaturizer())
        return self._model

    def transform(self, smiles: list[str]) -> np.ndarray:
        """Return CheMeleon embeddings for *smiles*."""
        import torch
        from chemprop.data import BatchMolGraph

        model, graph_featurizer = self._load_model()
        output_dim = model.message_passing.output_dim
        if not smiles:
            return np.zeros((0, output_dim), dtype=np.float32)

        mol_graphs = [graph_featurizer(Chem.MolFromSmiles(s)) for s in smiles]
        chunks = []
        for start in range(0, len(mol_graphs), self.batch_size):
            graph = BatchMolGraph(mol_graphs[start : start + self.batch_size])
            graph.to(device=self.device)
            with torch.no_grad():
                chunks.append(model.fingerprint(graph).cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32)


def make_activity_featurizer(
    representation: str, *, n_jobs: int = -1, device: str = "cpu"
):
    """Create the featurizer matching an activity model representation."""
    if representation == "rdkit2d":
        return RDKit2DFeaturizer(n_jobs=n_jobs)
    if representation == "chemeleon":
        return CheMeleonFeaturizer(device=device)
    raise ValueError(
        f"Unknown activity representation {representation!r}; "
        "expected 'rdkit2d' or 'chemeleon'"
    )