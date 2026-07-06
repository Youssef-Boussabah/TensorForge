"""Frozen parameters (requires_grad=False) must be skipped by optimizers,
even when a stale non-None grad is present."""

import numpy as np

from tensorforge import Tensor
from tensorforge.nn import Linear, Sequential, Tanh, mse_loss
from tensorforge.optim import SGD, Adam
from tensorforge.nn import Parameter


def _frozen_param():
    p = Parameter(np.array([1.0, -2.0]))
    p.requires_grad = False
    p.grad = np.array([10.0, -10.0])  # stale grad that must be ignored
    return p


def test_sgd_skips_frozen_parameter_with_grad():
    p = _frozen_param()
    SGD([p], lr=0.1).step()
    assert np.array_equal(p.data, [1.0, -2.0])


def test_adam_skips_frozen_parameter_with_grad():
    p = _frozen_param()
    opt = Adam([p], lr=0.1)
    opt.step()
    assert np.array_equal(p.data, [1.0, -2.0])
    # Consistent with existing design: the step counter still advances,
    # but the skipped parameter's moment state stays untouched.
    assert opt.t == 1
    assert np.allclose(opt.m[0], 0.0)
    assert np.allclose(opt.v[0], 0.0)


def test_sgd_still_updates_trainable_parameter():
    p = Parameter(np.array([1.0, -2.0]))
    p.grad = np.array([0.5, -1.0])
    SGD([p], lr=0.1).step()
    assert np.allclose(p.data, [1.0 - 0.05, -2.0 + 0.1])


def test_adam_still_updates_trainable_parameter():
    p = Parameter(np.array([1.0, -2.0]))
    p.grad = np.array([0.1, -0.2])
    Adam([p], lr=0.001).step()
    # First Adam step moves ~lr in the gradient's direction.
    assert np.allclose(p.data, [0.999, -1.999], atol=1e-6)


def test_mixed_frozen_and_trainable():
    for make_opt in (lambda ps: SGD(ps, lr=0.1), lambda ps: Adam(ps, lr=0.1)):
        trainable = Parameter(np.array([1.0]))
        trainable.grad = np.array([2.0])
        frozen = Parameter(np.array([5.0]))
        frozen.requires_grad = False
        frozen.grad = np.array([2.0])

        make_opt([trainable, frozen]).step()
        assert not np.allclose(trainable.data, [1.0])  # moved
        assert np.array_equal(frozen.data, [5.0])       # untouched


def test_zero_grad_clears_frozen_and_trainable_alike():
    trainable = Parameter(np.array([1.0]))
    trainable.grad = np.array([2.0])
    frozen = Parameter(np.array([5.0]))
    frozen.requires_grad = False
    frozen.grad = np.array([2.0])

    for opt in (SGD([trainable, frozen], lr=0.1), Adam([trainable, frozen], lr=0.1)):
        trainable.grad = np.array([2.0])
        frozen.grad = np.array([2.0])
        opt.zero_grad()
        assert trainable.grad is None
        assert frozen.grad is None


def test_trainable_parameters_helper():
    model = Sequential(Linear(2, 4), Tanh(), Linear(4, 3))
    frozen = model.modules[0].bias
    frozen.requires_grad = False

    trainable = model.trainable_parameters()
    assert frozen not in trainable
    assert len(trainable) == 3
    assert frozen in model.parameters()  # parameters() is unchanged
    assert len(model.parameters()) == 4


def test_training_with_frozen_weight():
    np.random.seed(0)
    model = Linear(1, 1)
    model.weight.requires_grad = False
    frozen_value = model.weight.data.copy()

    optimizer = SGD(model.parameters(), lr=0.1)
    x = Tensor([[0.0], [1.0], [2.0]])
    y = Tensor([[1.0], [2.0], [3.0]])
    for _ in range(20):
        optimizer.zero_grad()
        mse_loss(model(x), y).backward()
        optimizer.step()

    assert np.array_equal(model.weight.data, frozen_value)  # exactly unchanged
    assert not np.allclose(model.bias.data, 0.0)            # bias trained
