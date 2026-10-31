"""TensorBoard logging for EVA multi-population CMA-ES runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import ExperimentConfig, TensorBoardConfig
from optimization.loop import StepSnapshot


class TensorBoardLogger:
    """Write EVA search metrics and top molecule renders to TensorBoard."""

    def __init__(self, cfg: TensorBoardConfig, output_dir: str | Path) -> None:
        self._cfg = cfg
        self._writer: SummaryWriter | None = None

        if not cfg.enabled:
            return

        log_dir = Path(cfg.log_dir)
        if not log_dir.is_absolute():
            log_dir = Path(output_dir) / log_dir
        self._writer = SummaryWriter(str(log_dir))

    @property
    def enabled(self) -> bool:
        """Whether TensorBoard logging is enabled."""
        return self._writer is not None

    def log_config(self, config: ExperimentConfig) -> None:
        """Log the full experiment config and optional hypothesis as text."""
        if self._writer is None:
            return
        config_json = json.dumps(asdict(config), indent=2, default=str)
        self._writer.add_text("config/full", config_json)
        if config.run.hypothesis:
            self._writer.add_text("experiment/hypothesis", config.run.hypothesis)

    def log_step(self, snapshot: StepSnapshot) -> None:
        """Log step-indexed and oracle-call-indexed scalar metrics."""
        if self._writer is None:
            return

        step = snapshot.step
        result = snapshot.result

        if step % self._cfg.scalar_every == 0:
            total = len(result.objectives)
            valid_frac = result.n_valid / total if total > 0 else 0.0
            self._writer.add_scalar("eval/n_valid", result.n_valid, step)
            self._writer.add_scalar("eval/valid_fraction", valid_frac, step)
            self._writer.add_scalar("eval/n_duplicates", result.n_duplicates, step)
            self._writer.add_scalar("eval/n_oracle_calls", result.n_oracle_calls, step)
            self._writer.add_scalar("eval/gen_time_sec", result.gen_time, step)

            scored = result.p_active[result.p_active > 0]
            if len(scored) > 0:
                self._writer.add_scalar(
                    "eval/batch_best_p_active", float(scored.max()), step
                )
                self._writer.add_scalar(
                    "eval/batch_mean_p_active", float(scored.mean()), step
                )

            if result.timings is not None:
                t = result.timings
                self._writer.add_scalar("timing/decode_sec", t.decode, step)
                self._writer.add_scalar("timing/validity_sec", t.validity, step)
                self._writer.add_scalar(
                    "timing/featurize_predict_sec", t.featurize_predict, step
                )
                self._writer.add_scalar("timing/assemble_sec", t.assemble, step)
                self._writer.add_scalar("timing/solver_ops_sec", t.solver_ops, step)

            self._writer.add_scalar("cma/walker_spread", snapshot.walker_spread, step)
            self._writer.add_scalar("cma/total_restarts", snapshot.total_restarts, step)

            if snapshot.walker_stats:
                sigmas = [ws.sigma for ws in snapshot.walker_stats]
                ratios = [ws.axis_ratio for ws in snapshot.walker_stats]
                self._writer.add_scalar("cma/sigma_mean", float(np.mean(sigmas)), step)
                self._writer.add_scalar("cma/sigma_min", float(np.min(sigmas)), step)
                self._writer.add_scalar(
                    "cma/axis_ratio_max", float(np.max(ratios)), step
                )
                for ws in snapshot.walker_stats:
                    wid = ws.walker_id
                    self._writer.add_scalar(f"walker/{wid}/sigma", ws.sigma, step)
                    self._writer.add_scalar(
                        f"walker/{wid}/mean_norm", ws.mean_norm, step
                    )
                    self._writer.add_scalar(
                        f"walker/{wid}/axis_ratio", ws.axis_ratio, step
                    )
                    self._writer.add_scalar(f"walker/{wid}/restarts", ws.restarts, step)

        if result.n_oracle_calls > 0:
            oc = snapshot.oracle_calls
            self._writer.add_scalar("oracle/best_p_active", snapshot.best_p_active, oc)
            self._writer.add_scalar("oracle/n_hits_05", snapshot.n_hits_05, oc)
            self._writer.add_scalar("oracle/n_hits_06", snapshot.n_hits_06, oc)
            self._writer.add_scalar("oracle/n_hits_08", snapshot.n_hits_08, oc)
            self._writer.add_scalar(
                "oracle/unique_scaffolds_06", snapshot.unique_scaffolds_06, oc
            )
            self._writer.add_scalar("oracle/validity", snapshot.validity, oc)

        if (
            self._cfg.molecule_every > 0
            and step % self._cfg.molecule_every == 0
            and result.n_oracle_calls > 0
        ):
            self.log_molecules(step, result.smiles, result.p_active)

    def log_molecules(
        self,
        step: int,
        smiles: list[str],
        p_active: np.ndarray,
        n_samples: int = 4,
    ) -> None:
        """Render the highest-scoring molecules from the batch to TensorBoard."""
        if self._writer is None:
            return

        from rdkit import Chem
        from rdkit.Chem.Draw import MolsToGridImage

        scored = []
        for smi, pa in zip(smiles, p_active):
            if pa <= 0 or not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                scored.append((mol, smi, float(pa)))
        if not scored:
            return

        scored.sort(key=lambda x: x[2], reverse=True)
        picked = scored[:n_samples]

        mols = [m for m, _, _ in picked]
        legends = [f"P={pa:.3f}" for _, _, pa in picked]

        img = MolsToGridImage(
            mols,
            molsPerRow=min(len(mols), 3),
            subImgSize=(400, 400),
            legends=legends,
        )
        grid = np.array(img).transpose(2, 0, 1)
        self._writer.add_image("molecules/samples", grid, step, dataformats="CHW")
        self._writer.add_text(
            "molecules/smiles",
            "\n".join(f"P={pa:.3f}: {smi}" for _, smi, pa in picked),
            step,
        )

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Log final run summary scalars once at completion."""
        if self._writer is None:
            return
        oc = int(summary.get("oracle_calls", 0))
        self._writer.add_scalar(
            "summary/unique_scaffolds_06", int(summary["unique_scaffolds_06"]), oc
        )
        self._writer.add_scalar(
            "summary/largest_scaffold_frac_06",
            float(summary["largest_scaffold_frac_06"]),
            oc,
        )
        self._writer.add_scalar(
            "summary/best_p_active", float(summary["best_p_active"]), oc
        )
        self._writer.add_scalar("summary/n_hits_06", int(summary["n_hits_06"]), oc)

    def close(self) -> None:
        """Flush and close the underlying SummaryWriter."""
        if self._writer is None:
            return
        self._writer.close()
