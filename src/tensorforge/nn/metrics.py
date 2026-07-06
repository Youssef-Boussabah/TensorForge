"""Metrics.

Metrics are for reporting, not for training: they are plain NumPy
computations that stay outside autograd and return Python floats.
"""

import numpy as np

from tensorforge.nn.losses import _align_binary_targets, cross_entropy
from tensorforge.tensor import Tensor


def accuracy(logits, targets):
    """Fraction of rows where the highest-scoring class is the target.

    ``logits`` is shaped (batch, num_classes); ``targets`` holds integer
    class IDs. Both may be Tensors, NumPy arrays, or plain lists.
    """
    if isinstance(logits, Tensor):
        logits = logits.data
    if isinstance(targets, Tensor):
        targets = targets.data
    logits = np.asarray(logits)
    targets = np.asarray(targets, dtype=int)

    predictions = np.argmax(logits, axis=1)
    return float((predictions == targets).mean())


def binary_accuracy(logits, targets):
    """Fraction of binary predictions that match the 0/1 targets.

    ``logits`` are raw scores (Tensor, array, list, or scalar): a logit
    >= 0 predicts class 1, below 0 predicts class 0 — the same boundary
    as sigmoid(logit) >= 0.5. Shape rules match binary_cross_entropy.
    """
    if isinstance(logits, Tensor):
        logits = logits.data
    if isinstance(targets, Tensor):
        targets = targets.data
    logits = np.asarray(logits, dtype=np.float64)
    targets = _align_binary_targets(logits, targets)

    predictions = (logits >= 0.0).astype(np.float64)
    return float((predictions == targets).mean())


def evaluate_classifier(model, X, y):
    """Run ``model`` on a dataset and report loss and accuracy.

    ``X`` and ``y`` may be Tensors, NumPy arrays, or lists; ``y`` holds
    integer class IDs. Returns ``{"loss": float, "accuracy": float}``.

    This is a read-only measurement: it never calls backward() and
    never touches gradients or parameters.
    """
    if not isinstance(X, Tensor):
        X = Tensor(X)
    if isinstance(y, Tensor):
        y = y.data
    y = np.asarray(y, dtype=int)

    logits = model(X)
    loss = cross_entropy(logits, y)
    return {"loss": float(loss.data), "accuracy": accuracy(logits, y)}
