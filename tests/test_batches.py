import numpy as np
import pytest

from tensorforge import Tensor, batches


def _dataset():
    return np.arange(10).reshape(5, 2), np.arange(5)


def test_no_shuffle_batching():
    X, y = _dataset()
    result = list(batches(X, y, batch_size=2, shuffle=False))
    assert len(result) == 3
    assert [len(yb) for _, yb in result] == [2, 2, 1]
    assert [yb.tolist() for _, yb in result] == [[0, 1], [2, 3], [4]]
    assert [xb.tolist() for xb, _ in result] == [
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]],
        [[8, 9]],
    ]


def test_drop_last():
    X, y = _dataset()
    result = list(batches(X, y, batch_size=2, shuffle=False, drop_last=True))
    assert len(result) == 2
    assert [yb.tolist() for _, yb in result] == [[0, 1], [2, 3]]


def test_shuffle_is_deterministic_with_seed():
    X, y = _dataset()
    order_a = [yb.tolist() for _, yb in batches(X, y, batch_size=2, shuffle=True, seed=123)]
    order_b = [yb.tolist() for _, yb in batches(X, y, batch_size=2, shuffle=True, seed=123)]
    assert order_a == order_b
    flattened = [label for yb in order_a for label in yb]
    assert sorted(flattened) == [0, 1, 2, 3, 4]  # a permutation of the data
    assert flattened != [0, 1, 2, 3, 4]  # ...that actually shuffled


def test_x_and_y_stay_aligned_after_shuffle():
    y = np.arange(12)
    X = y.reshape(-1, 1) * 10.0  # each row encodes its own label
    for xb, yb in batches(X, y, batch_size=5, shuffle=True, seed=7):
        assert np.array_equal(xb[:, 0], yb * 10.0)


def test_tensor_inputs_yield_numpy_batches():
    X_np, y_np = _dataset()
    result = list(batches(Tensor(X_np), Tensor(y_np), batch_size=2, shuffle=False))
    assert len(result) == 3
    for xb, yb in result:
        assert isinstance(xb, np.ndarray)
        assert isinstance(yb, np.ndarray)
        assert not isinstance(xb, Tensor)
    assert result[0][0].tolist() == [[0, 1], [2, 3]]


def test_inputs_are_not_mutated():
    X, y = _dataset()
    X_copy, y_copy = X.copy(), y.copy()
    for xb, _ in batches(X, y, batch_size=2, shuffle=True, seed=0):
        xb += 100  # writing to a batch must not write through to X
    assert np.array_equal(X, X_copy)
    assert np.array_equal(y, y_copy)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        list(batches(np.zeros((5, 2)), np.zeros(4), batch_size=2))


def test_bad_batch_size_raises():
    X, y = _dataset()
    with pytest.raises(ValueError):
        list(batches(X, y, batch_size=0))
    with pytest.raises(ValueError):
        list(batches(X, y, batch_size=-3))
