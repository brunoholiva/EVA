"""Tests for TensorBoardLogger."""

from __future__ import annotations

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config import ExperimentConfig
from evaluation.evaluator import EvalResult
from optimization.cma import WalkerStats
from optimization.loop import StepSnapshot
from optimization.tensorboard import TensorBoardLogger


def test_log_config_step_and_summary(tmp_path) -> None:
    cfg = ExperimentConfig.from_toml("config.toml")
    logger = TensorBoardLogger(cfg.tensorboard, tmp_path)
    logger.log_config(cfg)

    res = EvalResult(
        smiles=["c1ccccc1", "INVALID"],
        objectives=np.array([0.75, -1.0]),
        p_active=np.array([0.75, 0.0]),
        n_valid=1,
        gen_time=0.2,
        n_duplicates=0,
        n_oracle_calls=1,
        canonical=["c1ccccc1", None],
        oracle_mask=np.array([True, False]),
    )
    snap = StepSnapshot(
        step=0,
        elapsed_s=0.2,
        attempts=2,
        oracle_calls=1,
        duplicate_attempts=0,
        invalid_attempts=1,
        validity=0.5,
        best_p_active=0.75,
        n_hits_05=1,
        n_hits_06=1,
        n_hits_08=0,
        unique_scaffolds_06=1,
        walker_spread=0.9,
        total_restarts=0,
        result=res,
        walker_ids=np.array([0, 0]),
        walker_stats=[
            WalkerStats(
                walker_id=0,
                sigma=0.48,
                mean_norm=1.2,
                axis_ratio=1.05,
                restarts=0,
                gens_in_run=1,
                last_stop_reasons={},
            )
        ],
    )
    logger.log_step(snap)
    logger.log_summary(
        {
            "oracle_calls": 1,
            "unique_scaffolds_06": 1,
            "largest_scaffold_frac_06": 1.0,
            "best_p_active": 0.75,
            "n_hits_06": 1,
        }
    )
    logger.close()

    ea = EventAccumulator(str(tmp_path / "tensorboard"))
    ea.Reload()
    tensor_tags = ea.Tags()["tensors"]
    scalar_tags = ea.Tags()["scalars"]

    assert "config/full/text_summary" in tensor_tags
    assert "experiment/hypothesis/text_summary" in tensor_tags
    assert "oracle/best_p_active" in scalar_tags
    assert "oracle/unique_scaffolds_06" in scalar_tags
    assert "cma/sigma_mean" in scalar_tags
    assert "summary/unique_scaffolds_06" in scalar_tags
