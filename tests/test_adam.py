import numpy as np

from tensorforge import Tensor
from tensorforge.nn import Linear, Parameter, Sequential, Tanh, mse_loss
from tensorforge.optim import Adam


def _reference_adam(data, grads, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Plain-NumPy Adam, stepped once per gradient in ``grads``."""
    data = data.copy()
    m = np.zeros_like(data)
    v = np.zeros_like(data)
    for t, g in enumerate(grads, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        data = data - lr * m_hat / (np.sqrt(v_hat) + eps)
    return data


def test_adam_is_importable_and_callable():
    assert callable(Adam)


def test_first_step_update():
    p = Parameter(np.array([1.0, -2.0]))
    p.grad = np.array([0.1, -0.2])
    opt = Adam([p], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    opt.step()
    # On step 1 the bias corrections cancel the (1 - beta) factors, so
    # the update collapses to lr * g / (|g| + eps): a step of size ~lr
    # in the gradient's direction, regardless of its magnitude.
    assert np.allclose(p.data, [0.999, -1.999], atol=1e-6)


def test_step_ignores_none_grad():
    p = Parameter([1.0, 2.0])
    assert p.grad is None
    opt = Adam([p], lr=0.1)
    opt.step()  # must not crash
    assert np.allclose(p.data, [1.0, 2.0])
    assert opt.t == 1  # the step still happened, it just had nothing to do


def test_zero_grad_clears_gradients():
    p = Parameter([1.0])
    p.grad = np.array([3.0])
    opt = Adam([p], lr=0.1)
    opt.zero_grad()
    assert p.grad is None  # same convention as SGD


def test_state_persists_across_steps():
    p = Parameter(np.array([1.0, -2.0]))
    grad = np.array([0.1, -0.2])
    opt = Adam([p], lr=0.001)

    p.grad = grad.copy()
    opt.step()
    after_one = p.data.copy()

    p.grad = grad.copy()
    opt.step()

    assert opt.t == 2
    assert not np.allclose(p.data, after_one)  # second step moved it again
    # Both steps must match a reference Adam that carries m/v across steps.
    expected = _reference_adam(np.array([1.0, -2.0]), [grad, grad])
    assert np.allclose(p.data, expected, atol=1e-10)


def test_step_does_not_modify_gradients():
    p = Parameter(np.array([1.0, -2.0]))
    p.grad = np.array([0.1, -0.2])
    grad_before = p.grad.copy()
    Adam([p], lr=0.001).step()
    assert np.array_equal(p.grad, grad_before)


def test_multiple_parameters_with_different_shapes():
    a = Parameter(np.ones((2, 3)))
    b = Parameter(np.zeros(4))
    a.grad = np.full((2, 3), 0.5)
    b.grad = np.array([1.0, -1.0, 2.0, -2.0])
    opt = Adam([a, b], lr=0.01)
    opt.step()
    assert a.data.shape == (2, 3)
    assert b.data.shape == (4,)
    assert not np.allclose(a.data, 1.0)
    assert not np.allclose(b.data, 0.0)


def test_adam_trains_tiny_regression():
    np.random.seed(0)
    model = Linear(1, 1)
    opt = Adam(model.parameters(), lr=0.1)

    x = Tensor([[0.0], [1.0], [2.0], [3.0]])
    y = Tensor([[1.0], [3.0], [5.0], [7.0]])  # y = 2x + 1

    initial_loss = float(mse_loss(model(x), y).data)
    for _ in range(200):
        opt.zero_grad()
        loss = mse_loss(model(x), y)
        loss.backward()
        opt.step()

    final_loss = float(loss.data)
    assert final_loss < initial_loss
    assert final_loss < 1e-3
    assert np.allclose(model.weight.data, [[2.0]], atol=0.1)
    assert np.allclose(model.bias.data, [1.0], atol=0.1)


def test_adam_trains_xor():
    np.random.seed(0)
    x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = Tensor([[0.0], [1.0], [1.0], [0.0]])
    model = Sequential(Linear(2, 4), Tanh(), Linear(4, 1))
    opt = Adam(model.parameters(), lr=0.05)

    initial_loss = float(mse_loss(model(x), y).data)
    for _ in range(300):
        opt.zero_grad()
        loss = mse_loss(model(x), y)
        loss.backward()
        opt.step()

    final_loss = float(loss.data)
    assert final_loss < initial_loss
    assert final_loss < 0.01
    # Rounding the outputs must reproduce the XOR truth table.
    assert np.array_equal(np.round(np.clip(model(x).data, 0.0, 1.0)), y.data)
