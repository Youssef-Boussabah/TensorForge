import numpy as np

from tensorforge import Tensor
from tensorforge.nn import ReLU


def test_tensor_relu_forward():
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 3.0])
    y = x.relu()
    assert y.data.shape == (5,)
    assert np.allclose(y.data, [0.0, 0.0, 0.0, 1.0, 3.0], atol=1e-6)


def test_tensor_relu_backward():
    x = Tensor(np.array([-2.0, 0.0, 3.0]), requires_grad=True)
    y = x.relu().sum()
    y.backward()
    # Gradient is blocked for negative inputs and, by convention, at
    # exactly 0 (the subgradient we pick there is 0).
    assert np.allclose(x.grad, [0.0, 0.0, 1.0], atol=1e-6)


def test_tensor_relu_backward_with_upstream_weights():
    x = Tensor(np.array([[-1.0, 2.0], [3.0, -4.0]]), requires_grad=True)
    weights = Tensor(np.array([[10.0, 20.0], [30.0, 40.0]]))
    loss = (x.relu() * weights).sum()
    loss.backward()
    assert np.allclose(x.grad, [[0.0, 20.0], [30.0, 0.0]], atol=1e-6)


def test_tensor_relu_grad_accumulates():
    """Two backward passes through relu must sum into x.grad."""
    x = Tensor(np.array([-2.0, 0.0, 3.0]), requires_grad=True)
    x.relu().sum().backward()
    x.relu().sum().backward()
    assert np.allclose(x.grad, [0.0, 0.0, 2.0], atol=1e-6)


def test_relu_module_matches_tensor_relu():
    relu = ReLU()
    x = Tensor([[-1.5, 0.0, 2.5]])
    assert np.allclose(relu(x).data, x.relu().data, atol=1e-6)


def test_relu_module_backward():
    relu = ReLU()
    x = Tensor(np.array([[-1.0, 2.0], [3.0, -4.0]]), requires_grad=True)
    relu(x).sum().backward()
    assert np.allclose(x.grad, [[0.0, 1.0], [1.0, 0.0]], atol=1e-6)
