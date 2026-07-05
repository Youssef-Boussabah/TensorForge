import numpy as np

from tensorforge import Tensor
from tensorforge.nn import Linear, Parameter, mse_loss
from tensorforge.optim import SGD


def test_step_updates_parameter():
    p = Parameter([1.0, 2.0])
    p.grad = np.array([0.5, -1.0])
    SGD([p], lr=0.1).step()
    # param = param - lr * grad
    assert np.allclose(p.data, [1.0 - 0.05, 2.0 + 0.1])


def test_step_skips_parameters_without_grad():
    p = Parameter([1.0, 2.0])
    assert p.grad is None
    SGD([p], lr=0.1).step()  # must not crash
    assert np.allclose(p.data, [1.0, 2.0])


def test_zero_grad_clears_gradients():
    a = Parameter([1.0])
    b = Parameter([2.0])
    a.grad = np.array([3.0])
    b.grad = np.array([4.0])
    opt = SGD([a, b], lr=0.1)
    opt.zero_grad()
    assert a.grad is None
    assert b.grad is None


def test_accepts_parameter_generators():
    """model.parameters() returns a list, but any iterable should work."""
    params = (Parameter([1.0]) for _ in range(3))
    opt = SGD(params, lr=0.1)
    assert len(opt.parameters) == 3


def test_training_step_decreases_mse_loss():
    """The v0.5 milestone: one real gradient-descent step must help."""
    np.random.seed(0)
    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.01)

    x = Tensor([[1.0], [2.0], [3.0], [4.0]])
    y = Tensor([[2.0], [4.0], [6.0], [8.0]])  # y = 2x

    loss_before = mse_loss(model(x), y)

    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()

    loss_after = mse_loss(model(x), y)
    assert loss_after.data < loss_before.data


def test_short_training_loop_converges():
    """A few dozen steps should fit y = 2x + 1 closely."""
    np.random.seed(42)
    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.05)

    x = Tensor([[0.0], [1.0], [2.0], [3.0]])
    y = Tensor([[1.0], [3.0], [5.0], [7.0]])  # y = 2x + 1

    for _ in range(200):
        optimizer.zero_grad()
        loss = mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

    assert loss.data < 1e-3
    assert np.allclose(model.weight.data, [[2.0]], atol=0.1)
    assert np.allclose(model.bias.data, [1.0], atol=0.1)
