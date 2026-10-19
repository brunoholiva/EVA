"""Tests for scheduler construction modes."""

from __future__ import annotations

from ribs.emitters import EvolutionStrategyEmitter

from config import ArchiveConfig, DimensionConfig, EmitterConfig
from optimization.loop import build_scheduler


def test_standard_emitter_mode_uses_pyribs_emitter() -> None:
    archive = ArchiveConfig(
        solution_dim=2,
        dimensions=[
            DimensionConfig(
                name="max_partial_charge", resolution=2, range=(0.0, 1.0)
            )
        ],
        learning_rate=0.01,
        threshold_min=0.0,
    )
    emitter = EmitterConfig(
        sigma0=0.5,
        batch_size=4,
        n_emitters=1,
        n_keeper_emitters=0,
        mixed_restarts=False,
    )

    scheduler = build_scheduler(archive, emitter, seed=42)

    assert isinstance(scheduler.emitters[0], EvolutionStrategyEmitter)
    assert scheduler.emitters[0].x0.tolist() == [0.0, 0.0]
