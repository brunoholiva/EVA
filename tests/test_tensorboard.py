from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config import ExperimentConfig
from optimization.tensorboard import TensorBoardLogger


def test_log_config_writes_hypothesis(tmp_path) -> None:
    cfg = ExperimentConfig.from_toml("config.toml")
    logger = TensorBoardLogger(cfg.tensorboard, tmp_path)
    logger.log_config(cfg)
    logger.close()

    ea = EventAccumulator(str(tmp_path / "tensorboard"))
    ea.Reload()
    tags = ea.Tags()["tensors"]
    assert "config/full/text_summary" in tags
    assert "experiment/hypothesis/text_summary" in tags
    for event in ea.Tensors("experiment/hypothesis/text_summary"):
        hypothesis = b"\n".join(event.tensor_proto.string_val).decode()
        assert "ranker" in hypothesis
