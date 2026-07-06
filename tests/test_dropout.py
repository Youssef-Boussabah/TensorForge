import numpy as np
import pytest

from tensorforge import Tensor
from tensorforge.nn import Dropout, Linear, Sequential


def test_p_validation():
    for bad in (-0.1, 1.0, 1.5, "half", None):
        with pytest.raises(ValueError):
            Dropout(p=bad)
    # Boundary that must be accepted.
    Dropout(p=0.0)
    Dropout(p=0.99)


def test_p_zero_is_identity_in_training_mode():
    x = Tensor(np.arange(6.0).reshape(2, 3))
    layer = Dropout(p=0.0)
    assert layer.training
    assert np.array_equal(layer(x).data, x.data)


def test_eval_mode_is_identity():
    x = Tensor(np.arange(10.0))
    layer = Dropout(p=0.7, seed=0).eval()
    out_a = layer(x)
    out_b = layer(x)
    assert np.array_equal(out_a.data, x.data)
    assert np.array_equal(out_b.data, x.data)


def test_training_mode_drops_and_scales():
    x = Tensor(np.ones(1000))
    layer = Dropout(p=0.5, seed=0)
    out = layer(x).data
    # Inverted dropout on ones: every entry is either 0 or 1/(1-p) = 2.
    assert set(np.unique(out)) <= {0.0, 2.0}
    dropped = float((out == 0.0).mean())
    assert 0.4 < dropped < 0.6  # roughly p of them dropped


def test_same_seed_same_first_mask():
    x = Tensor(np.ones(100))
    out_a = Dropout(p=0.5, seed=42)(x).data
    out_b = Dropout(p=0.5, seed=42)(x).data
    assert np.array_equal(out_a, out_b)


def test_repeated_calls_advance_rng():
    x = Tensor(np.ones(100))
    layer = Dropout(p=0.5, seed=42)
    first = layer(x).data
    second = layer(x).data
    assert not np.array_equal(first, second)


def test_backward_zero_for_dropped_scaled_for_kept():
    x = Tensor(np.ones(200), requires_grad=True)
    layer = Dropout(p=0.5, seed=7)
    out = layer(x)
    out.sum().backward()
    kept = out.data != 0.0
    # Kept positions pass gradient scaled by 1/(1-p) = 2, dropped get 0.
    assert np.array_equal(x.grad[kept], np.full(kept.sum(), 2.0))
    assert np.array_equal(x.grad[~kept], np.zeros((~kept).sum()))


def test_dropout_in_sequential_respects_mode():
    np.random.seed(0)
    model = Sequential(Linear(3, 3), Dropout(p=0.9, seed=0))
    x = Tensor(np.ones((4, 3)))

    model.eval()
    assert model.modules[1].training is False
    eval_a = model(x).data
    eval_b = model(x).data
    assert np.array_equal(eval_a, eval_b)  # eval is deterministic

    model.train()
    assert model.modules[1].training is True
    train_out = model(x).data
    assert not np.array_equal(train_out, eval_a)  # p=0.9 zeroes most entries
    assert np.any(train_out == 0.0)
