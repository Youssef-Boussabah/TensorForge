import numpy as np
import pytest

from tensorforge import Tensor


def test_milestone_x_squared():
    """The v0.1 milestone: d(x*x)/dx at x=4 is 8."""
    x = Tensor(4.0, requires_grad=True)
    y = x * x
    y.backward()
    assert np.allclose(x.grad, 8.0)


def test_addition():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = a + b
    c.backward()
    assert np.allclose(c.data, 5.0)
    assert np.allclose(a.grad, 1.0)
    assert np.allclose(b.grad, 1.0)


def test_subtraction():
    a = Tensor(5.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = a - b
    c.backward()
    assert np.allclose(c.data, 2.0)
    assert np.allclose(a.grad, 1.0)
    assert np.allclose(b.grad, -1.0)


def test_division():
    a = Tensor(6.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = a / b
    c.backward()
    assert np.allclose(c.data, 2.0)
    assert np.allclose(a.grad, 1.0 / 3.0)      # d(a/b)/da = 1/b
    assert np.allclose(b.grad, -6.0 / 9.0)     # d(a/b)/db = -a/b^2


def test_power():
    x = Tensor(2.0, requires_grad=True)
    y = x ** 3
    y.backward()
    assert np.allclose(y.data, 8.0)
    assert np.allclose(x.grad, 12.0)  # 3 * x^2


def test_scalar_operands():
    """Plain Python numbers should work on either side of an op."""
    x = Tensor(3.0, requires_grad=True)
    y = 2.0 * x + 1.0 - x / 3.0
    y.backward()
    assert np.allclose(y.data, 6.0)
    assert np.allclose(x.grad, 2.0 - 1.0 / 3.0)


def test_gradient_accumulates_when_tensor_is_reused():
    """y = x*x + x, so dy/dx = 2x + 1."""
    x = Tensor(3.0, requires_grad=True)
    y = x * x + x
    y.backward()
    assert np.allclose(x.grad, 7.0)


def test_sum():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.sum()
    y.backward()
    assert np.allclose(y.data, 6.0)
    assert np.allclose(x.grad, [1.0, 1.0, 1.0])


def test_mean():
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    y = x.mean()
    y.backward()
    assert np.allclose(y.data, 2.5)
    assert np.allclose(x.grad, [0.25, 0.25, 0.25, 0.25])


def test_relu():
    x = Tensor([-2.0, -0.5, 0.0, 1.5, 3.0], requires_grad=True)
    y = x.relu().sum()
    y.backward()
    assert np.allclose(x.relu().data, [0.0, 0.0, 0.0, 1.5, 3.0])
    assert np.allclose(x.grad, [0.0, 0.0, 0.0, 1.0, 1.0])


def test_broadcasting_gradients():
    """A row vector broadcast against a matrix must get a summed gradient."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([10.0, 20.0], requires_grad=True)
    y = (a * b).sum()
    y.backward()
    assert np.allclose(a.grad, [[10.0, 20.0], [10.0, 20.0]])
    assert np.allclose(b.grad, [1.0 + 3.0, 2.0 + 4.0])


def test_chained_expression():
    """A slightly bigger graph mixing several ops."""
    x = Tensor(2.0, requires_grad=True)
    y = (x * x + 1.0).relu() / x  # y = (x^2 + 1) / x for x > 0
    y.backward()
    assert np.allclose(y.data, 2.5)
    # dy/dx = 1 - 1/x^2 = 0.75 at x = 2
    assert np.allclose(x.grad, 0.75)


def test_no_grad_tracking_by_default():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b
    assert not c.requires_grad
    assert a.grad is None and b.grad is None


def test_requires_grad_propagates():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0)
    c = a * b
    c.backward()
    assert c.requires_grad
    assert np.allclose(a.grad, 3.0)
    assert b.grad is None  # b never asked for gradients


def test_backward_requires_grad():
    x = Tensor(1.0)
    with pytest.raises(RuntimeError):
        x.backward()
