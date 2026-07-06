"""Metrics.

Metrics are for reporting, not for training: they are plain NumPy
computations that stay outside autograd and return Python floats.
"""

import numpy as np

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
