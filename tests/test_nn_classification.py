import numpy as np

from tensorforge import Tensor
from tensorforge.nn import Linear, cross_entropy
from tensorforge.optim import SGD


def _numpy_softmax(x, axis=-1):
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def test_softmax_rows_sum_to_one():
    x = Tensor([[2.0, 1.0, 0.1], [-1.0, 0.0, 3.0]])
    probs = x.softmax()
    assert np.allclose(probs.data.sum(axis=-1), [1.0, 1.0])
    assert np.all(probs.data > 0)


def test_softmax_matches_numpy():
    x = Tensor([[2.0, 1.0, 0.1]])
    assert np.allclose(x.softmax(axis=-1).data, _numpy_softmax(x.data))


def test_softmax_axis():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(x.softmax(axis=-1).data, _numpy_softmax(x.data, axis=-1))
    assert np.allclose(x.softmax(axis=0).data, _numpy_softmax(x.data, axis=0))
    # axis=-1 and axis=1 are the same thing for a 2-D tensor.
    assert np.allclose(x.softmax(axis=-1).data, x.softmax(axis=1).data)


def test_softmax_is_numerically_stable():
    """Huge logits must not overflow exp()."""
    x = Tensor([[1000.0, 1001.0, 1002.0]])
    probs = x.softmax()
    assert np.all(np.isfinite(probs.data))
    assert np.allclose(probs.data.sum(), 1.0)


def test_softmax_gradient():
    """d(softmax_i)/dx_j = s_i * (delta_ij - s_j); check via a weighted sum."""
    x = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
    probs = x.softmax()
    # Loss = first component of the softmax output.
    (probs * np.array([[1.0, 0.0, 0.0]])).sum().backward()
    s = _numpy_softmax(x.data)[0]
    expected = s[0] * (np.array([1.0, 0.0, 0.0]) - s)
    assert np.allclose(x.grad, [expected])


def test_sum_with_axis_and_keepdims():
    """The extended sum() that softmax relies on."""
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.sum(axis=0)
    assert y.data.shape == (2,)
    assert np.allclose(y.data, [4.0, 6.0])

    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y2 = x2.sum(axis=1, keepdims=True)
    assert y2.data.shape == (2, 1)
    (y2 * np.array([[10.0], [20.0]])).sum().backward()
    assert np.allclose(x2.grad, [[10.0, 10.0], [20.0, 20.0]])


def test_cross_entropy_forward_value():
    logits = Tensor([[2.0, 1.0, 0.1]])
    loss = cross_entropy(logits, [0])
    expected = -np.log(_numpy_softmax(logits.data)[0, 0])
    assert np.allclose(loss.data, expected)


def test_cross_entropy_uniform_logits():
    """Equal logits mean p = 1/C for every class, so loss = log(C)."""
    logits = Tensor([[0.0, 0.0, 0.0, 0.0]])
    loss = cross_entropy(logits, [2])
    assert np.allclose(loss.data, np.log(4.0))


def test_cross_entropy_accepts_tensor_and_array_targets():
    logits = Tensor([[2.0, 1.0], [0.5, 3.0]])
    a = cross_entropy(logits, Tensor([0, 1]))
    b = cross_entropy(logits, np.array([0, 1]))
    c = cross_entropy(logits, [0, 1])
    assert np.allclose(a.data, b.data)
    assert np.allclose(b.data, c.data)


def test_cross_entropy_gradient():
    """The classic result: d(loss)/d(logits) = (softmax - one_hot) / batch."""
    logits = Tensor([[2.0, 1.0, 0.1], [0.5, 0.5, 3.0]], requires_grad=True)
    targets = [0, 2]
    cross_entropy(logits, targets).backward()

    assert logits.grad.shape == logits.data.shape
    one_hot = np.zeros((2, 3))
    one_hot[np.arange(2), targets] = 1.0
    expected = (_numpy_softmax(logits.data) - one_hot) / 2.0
    assert np.allclose(logits.grad, expected)


def test_cross_entropy_decreases_after_sgd_step():
    """One gradient step on a tiny classifier must reduce the loss."""
    np.random.seed(0)
    model = Linear(2, 3)
    optimizer = SGD(model.parameters(), lr=0.1)

    x = Tensor([[1.0, -1.0], [0.5, 2.0], [-1.5, 0.0]])
    targets = [0, 1, 2]

    loss_before = cross_entropy(model(x), targets)
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()

    loss_after = cross_entropy(model(x), targets)
    assert float(loss_after.data) < float(loss_before.data)
