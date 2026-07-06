import numpy as np

from tensorforge import Tensor


def _numpy_softmax(x, axis=-1):
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def test_softmax_forward_2d():
    x_np = np.array([[2.0, 1.0, 0.1], [-1.0, 0.0, 3.0]])
    probs = Tensor(x_np).softmax(axis=-1)
    assert probs.data.shape == x_np.shape
    assert np.allclose(probs.data.sum(axis=-1), [1.0, 1.0], atol=1e-6)
    assert np.allclose(probs.data, _numpy_softmax(x_np), atol=1e-6)


def test_softmax_forward_1d():
    x_np = np.array([1.0, 2.0, 3.0])
    probs = Tensor(x_np).softmax()
    assert probs.data.shape == (3,)
    assert np.allclose(probs.data.sum(), 1.0, atol=1e-6)
    assert np.allclose(probs.data, _numpy_softmax(x_np), atol=1e-6)


def test_softmax_axis_0():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    probs = Tensor(x_np).softmax(axis=0)
    assert np.allclose(probs.data.sum(axis=0), [1.0, 1.0], atol=1e-6)
    assert np.allclose(probs.data, _numpy_softmax(x_np, axis=0), atol=1e-6)


def test_softmax_numerical_stability():
    x = Tensor([[1000.0, 1001.0, 1002.0]])
    probs = x.softmax(axis=-1)
    assert np.all(np.isfinite(probs.data))
    assert np.allclose(probs.data.sum(axis=-1), 1.0, atol=1e-6)


def test_softmax_backward():
    x_np = np.array([[2.0, 1.0, 0.1], [-1.0, 0.5, 0.5]])
    weights = np.array([[1.0, -2.0, 3.0], [0.5, 0.0, -1.5]])

    x = Tensor(x_np, requires_grad=True)
    y = x.softmax(axis=-1)
    loss = (y * weights).sum()
    loss.backward()

    s = _numpy_softmax(x_np)
    expected = s * (weights - np.sum(weights * s, axis=-1, keepdims=True))
    assert np.allclose(x.grad, expected, atol=1e-6)


def test_softmax_backward_1d():
    x_np = np.array([0.5, -0.5, 2.0])
    weights = np.array([1.0, 2.0, 3.0])

    x = Tensor(x_np, requires_grad=True)
    (x.softmax() * weights).sum().backward()

    s = _numpy_softmax(x_np)
    expected = s * (weights - np.sum(weights * s, axis=-1, keepdims=True))
    assert np.allclose(x.grad, expected, atol=1e-6)


def test_softmax_grad_rows_sum_to_zero():
    """Softmax outputs always sum to 1, so any upstream gradient must
    produce input gradients that sum to 0 along the softmax axis."""
    x = Tensor([[3.0, -1.0, 0.5, 2.0]], requires_grad=True)
    (x.softmax() * np.array([[5.0, -2.0, 1.0, 0.0]])).sum().backward()
    assert np.allclose(x.grad.sum(axis=-1), 0.0, atol=1e-6)
