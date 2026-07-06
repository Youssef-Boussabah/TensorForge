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


def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    """Split X and y into (X_train, X_test, y_train, y_test).

    Accepts NumPy arrays, lists, or Tensors (unwrapped to their data).
    ``test_size`` is either a fraction in (0, 1) — rounded up to whole
    samples — or an exact number of test samples. Both splits always
    get at least one sample. With ``shuffle=False`` the training split
    is the first part of the data and the test split the last part, in
    original order. The returned arrays are copies; the inputs are
    never mutated.
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

    num_samples = X.shape[0]
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"float test_size must be in (0, 1), got {test_size}")
        num_test = int(np.ceil(num_samples * test_size))
    elif isinstance(test_size, int):
        num_test = test_size
    else:
        raise ValueError(f"test_size must be a float or int, got {test_size!r}")
    if not 1 <= num_test <= num_samples - 1:
        raise ValueError(
            f"test_size={test_size} leaves no samples in one of the splits "
            f"({num_samples} samples total; both splits need at least one)"
        )

    indices = np.arange(num_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    train_idx = indices[:-num_test]
    test_idx = indices[-num_test:]
    # Fancy indexing copies, so the splits are independent of the inputs.
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
