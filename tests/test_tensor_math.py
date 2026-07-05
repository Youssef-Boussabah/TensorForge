import numpy as np

from tensorforge import Tensor


def test_exp_forward():
    x = Tensor(2.0)
    assert np.allclose(x.exp().data, np.exp(2.0))


def test_exp_gradient():
    x = Tensor(2.0, requires_grad=True)
    y = x.exp()
    y.backward()
    assert np.allclose(x.grad, np.exp(2.0))


def test_log_forward():
    x = Tensor(2.0)
    assert np.allclose(x.log().data, np.log(2.0))


def test_log_gradient():
    x = Tensor(2.0, requires_grad=True)
    y = x.log()
    y.backward()
    assert np.allclose(x.grad, 0.5)


def test_tanh_forward():
    x = Tensor(0.5)
    assert np.allclose(x.tanh().data, np.tanh(0.5))


def test_tanh_gradient():
    x = Tensor(0.5, requires_grad=True)
    y = x.tanh()
    y.backward()
    assert np.allclose(x.grad, 1.0 - np.tanh(0.5) ** 2)


def test_sigmoid_forward():
    x = Tensor(0.0)
    assert np.allclose(x.sigmoid().data, 0.5)
    x2 = Tensor(2.0)
    assert np.allclose(x2.sigmoid().data, 1.0 / (1.0 + np.exp(-2.0)))


def test_sigmoid_gradient():
    x = Tensor(2.0, requires_grad=True)
    y = x.sigmoid()
    y.backward()
    s = 1.0 / (1.0 + np.exp(-2.0))
    assert np.allclose(x.grad, s * (1.0 - s))


def test_log_exp_roundtrip():
    """log(exp(x)) = x, so the gradient should be exactly 1."""
    x = Tensor(3.0, requires_grad=True)
    y = x.exp().log()
    y.backward()
    assert np.allclose(y.data, 3.0)
    assert np.allclose(x.grad, 1.0)


def test_array_input_gradients():
    """Each op applied elementwise to an array, summed to a scalar loss."""
    values = np.array([0.5, 1.0, 2.0])

    x = Tensor(values, requires_grad=True)
    x.exp().sum().backward()
    assert np.allclose(x.grad, np.exp(values))

    x = Tensor(values, requires_grad=True)
    x.log().sum().backward()
    assert np.allclose(x.grad, 1.0 / values)

    x = Tensor(values, requires_grad=True)
    x.tanh().sum().backward()
    assert np.allclose(x.grad, 1.0 - np.tanh(values) ** 2)

    x = Tensor(values, requires_grad=True)
    x.sigmoid().sum().backward()
    s = 1.0 / (1.0 + np.exp(-values))
    assert np.allclose(x.grad, s * (1.0 - s))


def test_chained_matmul_and_sigmoid():
    """A one-neuron logistic layer: loss = mean(sigmoid(x @ w))."""
    x = Tensor([[1.0, -2.0], [0.5, 3.0]], requires_grad=True)
    w = Tensor([[0.4], [-0.6]], requires_grad=True)
    loss = (x @ w).sigmoid().mean()
    loss.backward()

    # Recompute the gradients by hand with NumPy.
    z = x.data @ w.data
    s = 1.0 / (1.0 + np.exp(-z))
    dz = s * (1.0 - s) / s.size  # sigmoid grad, then mean spreads 1/N
    assert np.allclose(loss.data, s.mean())
    assert np.allclose(x.grad, dz @ w.data.T)
    assert np.allclose(w.grad, x.data.T @ dz)


def test_chained_matmul_and_tanh():
    """loss = sum(tanh(x @ w)) with hand-checked gradients."""
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[0.3], [-0.1]], requires_grad=True)
    loss = (x @ w).tanh().sum()
    loss.backward()

    z = x.data @ w.data
    dz = 1.0 - np.tanh(z) ** 2
    assert np.allclose(loss.data, np.tanh(z).sum())
    assert np.allclose(x.grad, dz @ w.data.T)
    assert np.allclose(w.grad, x.data.T @ dz)
