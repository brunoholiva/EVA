"""TensorBoard logging for EVA optimization runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ribs.archives import GridArchive
from torch.utils.tensorboard import SummaryWriter

from config import ExperimentConfig, TensorBoardConfig
from optimization.evaluator import EvalResult
from reporting.plotting import create_archive_figure


class TensorBoardLogger:
    """Write EVA optimization metrics to TensorBoard."""

    def __init__(self, cfg: TensorBoardConfig, output_dir: str | Path) -> None:
        self._cfg = cfg
        self._writer = None

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
        """Log the full experiment config as a TensorBoard text entry.

        Call once at startup so each run's hyperparameters are visible
        in the TensorBoard UI alongside the scalar curves.
        """
        if self._writer is None:
            return
        config_dict = asdict(config)
        config_json = json.dumps(config_dict, indent=2, default=str)
        self._writer.add_text("config/full", config_json)

    def log_generation(
        self,
        step: int,
        result: EvalResult,
        archive: GridArchive,
        result_archive: GridArchive | None,
        dimension_names: list[str],
        insertion_stats: dict[str, int] | None = None,
        emitter_stats: list[dict] | None = None,
        real_objectives: dict[int, float] | None = None,
        objective_cap: float | None = None,
        emitter_insertions: list[dict] | None = None,
        emitter_spread: float | None = None,
    ) -> None:
        """Log generation metrics to TensorBoard."""
        if self._writer is None:
            return

        report_archive = result_archive if result_archive is not None else archive

        if step % self._cfg.scalar_every == 0:
            self._log_result_archive_stats(report_archive, step)
            self._log_eval_stats(result, step, real_objectives, objective_cap)
            if insertion_stats is not None:
                self._log_insertion_stats(insertion_stats, step)
            if emitter_stats is not None:
                self._log_emitter_stats(emitter_stats, step)
            if emitter_insertions is not None:
                self._log_emitter_insertions(emitter_insertions, step)
            if emitter_spread is not None:
                self._writer.add_scalar("emitter/spread", emitter_spread, step)

        if step % self._cfg.histogram_every == 0:
            self._log_archive_histograms(report_archive, dimension_names, step)

        if step % self._cfg.figure_every == 0 and report_archive.measure_dim >= 2:
            figure = create_archive_figure(report_archive, dimension_names)
            if figure is not None:
                self._writer.add_figure("figures/archive", figure, step)

    def log_molecules(
        self,
        step: int,
        smiles: list[str],
        p_active: np.ndarray,
        n_samples: int = 4,
    ) -> None:
        """Log rendered molecule images from the current batch.

        Picks ``n_samples`` valid molecules with the highest P(active),
        renders them in a grid with RDKit, and logs to TensorBoard.

        Parameters
        ----------
        step : int
            Current generation number.
        smiles : list of str
            SMILES strings from the current batch.
        p_active : np.ndarray
            P(active) values for each SMILES.
        n_samples : int
            Number of molecules to render.
        """
        if self._writer is None:
            return

        from rdkit import Chem
        from rdkit.Chem.Draw import MolsToGridImage

        scored = []
        for smi, pa in zip(smiles, p_active):
            mol = Chem.MolFromSmiles(smi) if smi else None
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

    def close(self) -> None:
        """Close the writer if it was created."""
        if self._writer is None:
            return
        self._writer.close()

    def _log_result_archive_stats(self, archive: GridArchive, step: int) -> None:
        """Log archive-level stats (result archive only)."""
        stats = archive.stats
        self._writer.add_scalar("result_archive/coverage", float(stats.coverage), step)
        self._writer.add_scalar(
            "result_archive/num_elites", int(stats.num_elites), step
        )
        self._writer.add_scalar("result_archive/qd_score", float(stats.qd_score), step)

        if stats.obj_max is not None:
            self._writer.add_scalar(
                "result_archive/obj_max", float(stats.obj_max), step
            )
        if stats.obj_mean is not None:
            self._writer.add_scalar(
                "result_archive/obj_mean", float(stats.obj_mean), step
            )

    def _log_eval_stats(
        self,
        result: EvalResult,
        step: int,
        real_objectives: dict[int, float] | None = None,
        objective_cap: float | None = None,
    ) -> None:
        """Log batch-level evaluation stats."""
        valid_fraction = 0.0
        if len(result.objectives) > 0:
            valid_fraction = result.n_valid / len(result.objectives)

        self._writer.add_scalar("eval/n_valid", result.n_valid, step)
        self._writer.add_scalar("eval/valid_fraction", valid_fraction, step)
        self._writer.add_scalar("eval/n_duplicates", result.n_duplicates, step)
        self._writer.add_scalar("eval/gen_time_sec", result.gen_time, step)

        if result.timings is not None:
            self._writer.add_scalar("timing/decode_sec", result.timings.decode, step)
            self._writer.add_scalar(
                "timing/validity_sec", result.timings.validity, step
            )
            self._writer.add_scalar(
                "timing/featurize_sec", result.timings.featurize_predict, step
            )
            self._writer.add_scalar(
                "timing/cpu_scorers_sec", result.timings.cpu_scorers, step
            )
            self._writer.add_scalar(
                "timing/assemble_sec", result.timings.assemble, step
            )
            self._writer.add_scalar(
                "timing/archive_ops_sec", result.timings.archive_ops, step
            )

        active = result.p_active[result.p_active > 0]
        if len(active) > 0:
            self._writer.add_scalar(
                "eval/best_p_active_batch", float(active.max()), step
            )
            self._writer.add_scalar(
                "eval/mean_p_active_batch", float(active.mean()), step
            )

        if (
            objective_cap is not None
            and real_objectives is not None
            and len(real_objectives) > 0
        ):
            real_values = list(real_objectives.values())
            real_array = np.array(real_values)
            self._writer.add_scalar(
                "eval/real_p_active_max", float(real_array.max()), step
            )
            self._writer.add_scalar(
                "eval/real_p_active_mean", float(real_array.mean()), step
            )
            self._writer.add_scalar(
                "eval/n_above_cap", int((real_array > objective_cap).sum()), step
            )
            self._writer.add_scalar("eval/n_tracked", len(real_values), step)

    def _log_insertion_stats(self, stats: dict[str, int], step: int) -> None:
        """Log per-generation archive insertion outcomes."""
        self._writer.add_scalar("insertion/inserted_new", stats["inserted_new"], step)
        self._writer.add_scalar(
            "insertion/improved_existing", stats["improved_existing"], step
        )
        self._writer.add_scalar("insertion/rejected", stats["rejected"], step)

    def _log_emitter_stats(self, stats: list[dict], step: int) -> None:
        """Log per-emitter statistics (distance, restarts)."""
        for es in stats:
            i = es["id"]
            self._writer.add_scalar(
                f"emitter/{i}/distance", float(es["distance"]), step
            )
            self._writer.add_scalar(f"emitter/{i}/restarts", int(es["restarts"]), step)

    def _log_emitter_insertions(self, stats: list[dict], step: int) -> None:
        """Log per-emitter archive insertion counts."""
        for es in stats:
            i = es["id"]
            self._writer.add_scalar(
                f"emitter/{i}/inserted_new", es["inserted_new"], step
            )
            self._writer.add_scalar(
                f"emitter/{i}/improved_existing", es["improved_existing"], step
            )
            self._writer.add_scalar(f"emitter/{i}/rejected", es["rejected"], step)

    def _log_archive_histograms(
        self, archive: GridArchive, dimension_names: list[str], step: int
    ) -> None:
        """Log archive distributions requested for live monitoring."""
        if len(archive) == 0:
            return

        data = archive.data()
        self._writer.add_histogram("dist/objective", data["objective"], step)

        name_to_index = {
            name: idx for idx, name in enumerate(dimension_names[: archive.measure_dim])
        }
        for metric_name in ["logp", "tpsa", "mw", "fsp3"]:
            if metric_name not in name_to_index:
                continue
            idx = name_to_index[metric_name]
            self._writer.add_histogram(
                f"dist/{metric_name}", data["measures"][:, idx], step
            )
