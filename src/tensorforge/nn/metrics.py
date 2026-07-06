"""Metrics.

Metrics are for reporting, not for training: they are plain NumPy
computations that stay outside autograd and return Python floats.
"""

import numpy as np

from tensorforge.nn.losses import cross_entropy
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
