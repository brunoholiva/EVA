"""TabPFN activity prediction: model loading and inference."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from joblib import load

from evaluation.model_utils import move_model_to_device

if TYPE_CHECKING:
    from tabpfn import TabPFNClassifier


@contextmanager
def _suppress_tabpfn_progress():
    """Suppress TabPFN's tqdm progress bars."""
    from contextlib import redirect_stderr

    # Redirect stderr to devnull to suppress tqdm
    with redirect_stderr(open(os.devnull, "w")):
        yield


def _ensure_cuda_compat() -> None:
    """Set PyTorch CUDA allocator config for large model loading."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in conf:
        extra = "expandable_segments:True"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"{conf},{extra}" if conf else extra


def load_model(
    path: Path, device: str = "cpu", softmax_temperature: float | None = None
) -> TabPFNClassifier:
    """Load a saved TabPFN classifier and optionally move it to *device*.

    Parameters
    ----------
    path : Path
        Path to the joblib-saved model.
    device : str, default="cpu"
        Target device (``"cpu"`` or ``"cuda"``).
    softmax_temperature : float or None, default=None
        If provided, override the fitted ``softmax_temperature_`` attribute.
        This controls the sharpness of predicted probabilities at inference
        time without re-training.

    Notes
    -----
    Setting ``softmax_temperature`` only updates the fitted attribute
    (``softmax_temperature_``) used during prediction.  The constructor
    attribute (``softmax_temperature``) is left unchanged.
    """
    model: TabPFNClassifier = load(path)
    model.inference_precision = "autocast"
    if softmax_temperature is not None:
        model.softmax_temperature_ = softmax_temperature
    if device != "cpu":
        _ensure_cuda_compat()
        move_model_to_device(model, device)
    return model


def predict_from_features(
    X: np.ndarray, model: TabPFNClassifier
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and probabilities from a pre-featurized matrix.

    Parameters
    ----------
    X : np.ndarray of shape ``(n, n_features)``
        Feature matrix.
    model : TabPFNClassifier
        Fitted classifier.

    Returns
    -------
    preds : np.ndarray of shape ``(n,)``
        Predicted class labels (0 or 1).
    probs : np.ndarray of shape ``(n, 2)``
        Predicted class probabilities.

    Raises
    ------
    ValueError
        If ``X`` feature dimension does not match the model's expected
        dimension (``n_features_in_``).
    """
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        raise ValueError(
            f"Feature dimension mismatch: got {X.shape[1]}, model expects {expected}"
        )

    # Suppress TabPFN's tqdm progress bars
    with _suppress_tabpfn_progress():
        probs = model.predict_proba(X)
    preds = model.classes_[probs.argmax(axis=1)]
    return preds, probs
