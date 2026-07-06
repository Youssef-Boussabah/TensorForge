"""Data utilities.

Like metrics, these are plain NumPy helpers that stay outside autograd:
they slice datasets into mini-batches, and the training loop decides
what to wrap in Tensors.
"""

import numpy as np

from tensorforge.tensor import Tensor


def batches(X, y, batch_size, shuffle=True, seed=None, drop_last=False):
    """Yield (xb, yb) NumPy mini-batches drawn from X and y.

    Accepts NumPy arrays, lists, or Tensors (unwrapped to their data).
    With ``shuffle=True`` the order is randomized — pass ``seed`` to make
    it reproducible. The final batch is smaller when ``batch_size`` does
    not divide the dataset evenly, unless ``drop_last=True`` skips it.
    The inputs are never mutated; batches are indexed copies.
    """
    if isinstance(X, Tensor):
        X = X.data
    if isinstance(y, Tensor):
        y = y.data
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples, "
            f"got {X.shape[0]} and {y.shape[0]}"
        )
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

    indices = np.arange(X.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        if drop_last and len(batch) < batch_size:
            break
        yield X[batch], y[batch]
