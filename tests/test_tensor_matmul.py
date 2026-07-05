import numpy as np
import pytest

from tensorforge import Tensor


def test_milestone_example():
    """The v0.2 milestone from the spec."""
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[3.0], [4.0]], requires_grad=True)
    y = x @ w
    y.backward()
    assert np.allclose(y.data, [[11.0]])
    assert np.allclose(x.grad, [[3.0, 4.0]])
    assert np.allclose(w.grad, [[1.0], [2.0]])


def test_forward_matrix_matrix():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    assert np.allclose(c.data, np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([[5.0, 6.0], [7.0, 8.0]]))
    assert not c.requires_grad


def test_matmul_method_matches_operator():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0], [6.0]])
    assert np.allclose(a.matmul(b).data, (a @ b).data)


def test_grad_left_operand():
    """Non-square shapes catch transposed-gradient mistakes."""
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)  # (2, 3)
    b = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])                    # (3, 2)
    (a @ b).sum().backward()
    # dL/dA = ones(2, 2) @ B^T
    expected = np.ones((2, 2)) @ b.data.T
    assert a.grad.shape == (2, 3)
    assert np.allclose(a.grad, expected)
    assert b.grad is None  # b never asked for gradients


def test_grad_right_operand():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])                      # (2, 3)
    b = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)  # (3, 2)
    (a @ b).sum().backward()
    # dL/dB = A^T @ ones(2, 2)
    expected = a.data.T @ np.ones((2, 2))
    assert b.grad.shape == (3, 2)
    assert np.allclose(b.grad, expected)
    assert a.grad is None


def test_matrix_vector():
    m = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    v = Tensor([5.0, 6.0], requires_grad=True)
    y = m @ v  # shape (2,)
    y.sum().backward()
    assert np.allclose(y.data, [17.0, 39.0])
    assert m.grad.shape == (2, 2)
    assert np.allclose(m.grad, [[5.0, 6.0], [5.0, 6.0]])
    assert v.grad.shape == (2,)
    assert np.allclose(v.grad, [4.0, 6.0])  # column sums of m


def test_vector_matrix():
    v = Tensor([1.0, 2.0], requires_grad=True)
    m = Tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    y = v @ m  # shape (2,)
    y.sum().backward()
    assert np.allclose(y.data, [13.0, 16.0])
    assert v.grad.shape == (2,)
    assert np.allclose(v.grad, [7.0, 11.0])  # row sums of m
    assert m.grad.shape == (2, 2)
    assert np.allclose(m.grad, [[1.0, 1.0], [2.0, 2.0]])


def test_vector_vector_dot_product():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    y = a @ b  # scalar
    y.backward()
    assert np.allclose(y.data, 32.0)
    assert np.allclose(a.grad, [4.0, 5.0, 6.0])
    assert np.allclose(b.grad, [1.0, 2.0, 3.0])


def test_chained_expression_with_matmul():
    """A tiny linear layer by hand: loss = mean(relu(x @ w + b))."""
    x = Tensor([[1.0, -2.0], [3.0, 0.5]], requires_grad=True)
    w = Tensor([[0.5, -1.0], [2.0, 1.0]], requires_grad=True)
    b = Tensor([1.0, -1.0], requires_grad=True)
    loss = ((x @ w + b).relu()).mean()
    loss.backward()

    # Check against gradients computed by hand.
    z = x.data @ w.data + b.data
    mask = (z > 0).astype(float)      # relu gate
    dz = mask / z.size                # mean spreads 1/N to each element
    assert np.allclose(loss.data, np.maximum(z, 0.0).mean())
    assert np.allclose(x.grad, dz @ w.data.T)
    assert np.allclose(w.grad, x.data.T @ dz)
    assert np.allclose(b.grad, dz.sum(axis=0))


def test_grad_accumulates_when_reused_in_matmul():
    """y = x @ x uses x twice, so both contributions must sum."""
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    (x @ x).sum().backward()
    ones = np.ones((2, 2))
    expected = ones @ x.data.T + x.data.T @ ones
    assert np.allclose(x.grad, expected)


def test_plain_array_operands():
    """Lists / arrays should be wrapped automatically on either side."""
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    y = x @ [[3.0], [4.0]]
    y.backward()
    assert np.allclose(y.data, [[11.0]])
    assert np.allclose(x.grad, [[3.0, 4.0]])

    x2 = Tensor([[3.0], [4.0]], requires_grad=True)
    y2 = [[1.0, 2.0]] @ x2
    y2.backward()
    assert np.allclose(y2.data, [[11.0]])
    assert np.allclose(x2.grad, [[1.0], [2.0]])


def test_batched_matmul_not_supported_yet():
    a = Tensor(np.ones((2, 2, 2)))
    b = Tensor(np.ones((2, 2)))
    with pytest.raises(NotImplementedError):
        a @ b
