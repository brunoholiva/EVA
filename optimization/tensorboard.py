"""TensorBoard logging for EVA optimization runs."""

from __future__ import annotations

from pathlib import Path

from ribs.archives import GridArchive

from config import TensorBoardConfig
from optimization.evaluator import EvalResult
from optimization.plotting import create_parallel_axes_figure


def _load_summary_writer_class():
    """Load a TensorBoard SummaryWriter implementation."""
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter
    except ImportError:
        pass

    try:
        from tensorboardX import SummaryWriter

        return SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging is enabled, but no SummaryWriter is available. "
            "Install tensorboard or tensorboardX in the eva environment."
        ) from exc


class TensorBoardLogger:
    """Write EVA optimization metrics to TensorBoard."""

    def __init__(self, cfg: TensorBoardConfig, output_dir: str | Path) -> None:
        self._cfg = cfg
        self._writer = None
        self._last_coverage: float | None = None
        self._last_num_elites: int | None = None

        if not cfg.enabled:
            return

        SummaryWriter = _load_summary_writer_class()
        log_dir = Path(cfg.log_dir)
        if not log_dir.is_absolute():
            log_dir = Path(output_dir) / log_dir
        self._writer = SummaryWriter(str(log_dir))

    @property
    def enabled(self) -> bool:
        """Whether TensorBoard logging is enabled."""
        return self._writer is not None

    def log_generation(
        self,
        step: int,
        result: EvalResult,
        archive: GridArchive,
        result_archive: GridArchive | None,
        cache_size: int,
        dimension_names: list[str],
    ) -> None:
        """Log generation metrics to TensorBoard."""
        if self._writer is None:
            return

        report_archive = result_archive if result_archive is not None else archive

        if step % self._cfg.scalar_every == 0:
            self._log_archive_stats("result_archive", report_archive, step)
            self._log_archive_stats("archive", archive, step)
            self._log_eval_stats(result, cache_size, step)

        if step % self._cfg.histogram_every == 0:
            self._log_archive_histograms(report_archive, dimension_names, step)

        if step % self._cfg.figure_every == 0 and report_archive.measure_dim >= 3:
            figure = create_parallel_axes_figure(report_archive, dimension_names)
            if figure is not None:
                self._writer.add_figure("figures/archive_parallel_axes", figure, step)

    def close(self) -> None:
        """Close the writer if it was created."""
        if self._writer is None:
            return
        self._writer.close()

    def _log_archive_stats(self, prefix: str, archive: GridArchive, step: int) -> None:
        """Log scalar stats from a pyribs archive."""
        stats = archive.stats
        self._writer.add_scalar(f"{prefix}/coverage", float(stats.coverage), step)
        self._writer.add_scalar(f"{prefix}/num_elites", int(stats.num_elites), step)
        self._writer.add_scalar(f"{prefix}/qd_score", float(stats.qd_score), step)
        self._writer.add_scalar(
            f"{prefix}/norm_qd_score", float(stats.norm_qd_score), step
        )

        if stats.obj_max is not None:
            self._writer.add_scalar(f"{prefix}/obj_max", float(stats.obj_max), step)
        if stats.obj_mean is not None:
            self._writer.add_scalar(f"{prefix}/obj_mean", float(stats.obj_mean), step)

        if prefix == "result_archive":
            coverage_delta = 0.0
            if self._last_coverage is not None:
                coverage_delta = float(stats.coverage) - self._last_coverage
            elite_delta = 0
            if self._last_num_elites is not None:
                elite_delta = int(stats.num_elites) - self._last_num_elites

            self._writer.add_scalar(f"{prefix}/coverage_delta", coverage_delta, step)
            self._writer.add_scalar(f"{prefix}/num_elites_delta", elite_delta, step)

            self._last_coverage = float(stats.coverage)
            self._last_num_elites = int(stats.num_elites)

    def _log_eval_stats(self, result: EvalResult, cache_size: int, step: int) -> None:
        """Log batch-level evaluation stats."""
        valid_fraction = 0.0
        if len(result.objectives) > 0:
            valid_fraction = result.n_valid / len(result.objectives)

        self._writer.add_scalar("eval/n_valid", result.n_valid, step)
        self._writer.add_scalar("eval/valid_fraction", valid_fraction, step)
        self._writer.add_scalar("eval/gen_time_sec", result.gen_time, step)
        self._writer.add_scalar("system/novelty_cache_size", cache_size, step)

        active = result.p_active[result.p_active > 0]
        if len(active) > 0:
            self._writer.add_scalar(
                "eval/best_p_active_batch", float(active.max()), step
            )
            self._writer.add_scalar(
                "eval/mean_p_active_batch", float(active.mean()), step
            )

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
        for metric_name in ["br_sascore", "logp", "tpsa"]:
            if metric_name not in name_to_index:
                continue
            idx = name_to_index[metric_name]
            self._writer.add_histogram(
                f"dist/{metric_name}", data["measures"][:, idx], step
            )
