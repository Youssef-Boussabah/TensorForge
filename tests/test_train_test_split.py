import numpy as np
import pytest

from tensorforge import Tensor, train_test_split


def _dataset(n=10):
    y = np.arange(n)
    X = y.reshape(-1, 1) * 10.0  # each row encodes its own label
    return X, y


def test_split_shapes_with_float_test_size():
    X, y = _dataset(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=0)
    assert X_train.shape == (8, 1) and y_train.shape == (8,)
    assert X_test.shape == (2, 1) and y_test.shape == (2,)


def test_float_test_size_uses_ceil():
    X, y = _dataset(5)
    # 5 * 0.3 = 1.5 -> ceil -> 2 test samples
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, seed=0)
    assert len(X_test) == 2 and len(y_test) == 2


def test_int_test_size():
    X, y = _dataset(10)
    X_train, X_test, _, _ = train_test_split(X, y, test_size=3, seed=0)
    assert len(X_train) == 7 and len(X_test) == 3


def test_deterministic_with_seed():
    X, y = _dataset()
    a = train_test_split(X, y, test_size=0.3, seed=42)
    b = train_test_split(X, y, test_size=0.3, seed=42)
    for arr_a, arr_b in zip(a, b):
        assert np.array_equal(arr_a, arr_b)
    # A different seed produces a different split.
    c = train_test_split(X, y, test_size=0.3, seed=7)
    assert not all(np.array_equal(x, z) for x, z in zip(a, c))


def test_alignment_after_shuffle():
    X, y = _dataset(20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=3)
    assert np.array_equal(X_train[:, 0], y_train * 10.0)
    assert np.array_equal(X_test[:, 0], y_test * 10.0)
    # Together the splits are exactly the original samples, once each.
    assert sorted(np.concatenate([y_train, y_test]).tolist()) == list(range(20))


def test_no_shuffle_preserves_order():
    X, y = _dataset(5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
    assert y_train.tolist() == [0, 1, 2]
    assert y_test.tolist() == [3, 4]
    assert np.array_equal(X_train, X[:3])
    assert np.array_equal(X_test, X[3:])


def test_inputs_not_mutated_and_outputs_are_copies():
    X, y = _dataset()
    X_copy, y_copy = X.copy(), y.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=0)

    for arr in (X_train, X_test, y_train, y_test):
        assert arr.base is None  # a real copy, not a view
        arr += 1000  # writing to a split must not write through

    assert np.array_equal(X, X_copy)
    assert np.array_equal(y, y_copy)


def test_tensor_and_list_inputs():
    X, y = _dataset(6)
    from_np = train_test_split(X, y, test_size=2, shuffle=False)
    from_tensor = train_test_split(Tensor(X), Tensor(y.astype(float)), test_size=2, shuffle=False)
    from_list = train_test_split(X.tolist(), y.tolist(), test_size=2, shuffle=False)

    for np_arr, t_arr, l_arr in zip(from_np, from_tensor, from_list):
        assert isinstance(t_arr, np.ndarray)
        assert isinstance(l_arr, np.ndarray)
        assert np.array_equal(np_arr, t_arr)
        assert np.array_equal(np_arr, l_arr)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        train_test_split(np.zeros((5, 2)), np.zeros(4))


def test_invalid_test_size_raises():
    X, y = _dataset(5)
    for bad in (0.0, 1.0, -0.5, 1.5, 0, 5, 6, -1, "half"):
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=bad)
