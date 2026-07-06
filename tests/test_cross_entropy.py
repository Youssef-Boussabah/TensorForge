import numpy as np

from tensorforge import Tensor, cross_entropy


def _stable_softmax(x, axis=-1):
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def _reference_loss(logits_np, targets):
    shifted = logits_np - np.max(logits_np, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return -np.mean(log_probs[np.arange(logits_np.shape[0]), targets])


def test_cross_entropy_forward():
    logits_np = np.array([[2.0, 1.0, 0.1], [0.1, 0.2, 3.0], [1.0, 1.0, 1.0]])
    targets = np.array([0, 2, 1])
    loss = cross_entropy(Tensor(logits_np), targets)
    assert loss.data.shape == ()
    assert np.allclose(loss.data, _reference_loss(logits_np, targets), atol=1e-6)


def test_cross_entropy_uniform_logits():
    """Equal logits mean p = 1/C, so the loss is exactly log(C)."""
    loss = cross_entropy(Tensor([[0.0, 0.0, 0.0, 0.0]]), [2])
    assert np.allclose(loss.data, np.log(4.0), atol=1e-6)


def test_cross_entropy_backward():
    logits_np = np.array([[2.0, 1.0, 0.1], [0.5, 0.5, 3.0]])
    targets = np.array([0, 2])
    batch_size = logits_np.shape[0]

    logits = Tensor(logits_np, requires_grad=True)
    loss = cross_entropy(logits, targets)
    loss.backward()

    probs = _stable_softmax(logits_np, axis=1)
    expected_grad = probs.copy()
    expected_grad[np.arange(batch_size), targets] -= 1
    expected_grad /= batch_size

    assert logits.grad.shape == logits_np.shape
    assert np.allclose(logits.grad, expected_grad, atol=1e-6)
    # Each row of the gradient sums to 0: probabilities sum to 1 and the
    # one-hot subtracts exactly 1 from each row.
    assert np.allclose(logits.grad.sum(axis=1), 0.0, atol=1e-6)


def test_cross_entropy_numerical_stability():
    logits_np = np.array([[1000.0, 1001.0, 1002.0], [1200.0, 1199.0, 1198.0]])
    logits = Tensor(logits_np, requires_grad=True)
    loss = cross_entropy(logits, np.array([2, 0]))
    loss.backward()
    assert np.isfinite(loss.data)
    assert np.all(np.isfinite(logits.grad))


def test_cross_entropy_accepts_list_and_tensor_targets():
    logits_np = np.array([[2.0, 1.0], [0.5, 3.0]])
    a = cross_entropy(Tensor(logits_np), np.array([0, 1]))
    b = cross_entropy(Tensor(logits_np), [0, 1])
    c = cross_entropy(Tensor(logits_np), Tensor([0, 1]))
    assert np.allclose(a.data, b.data, atol=1e-6)
    assert np.allclose(b.data, c.data, atol=1e-6)


def test_cross_entropy_gradient_accumulates():
    """Two backward passes through the same logits must sum, matching
    how every other op accumulates gradients."""
    logits_np = np.array([[2.0, 1.0, 0.1]])
    targets = [0]

    logits = Tensor(logits_np, requires_grad=True)
    cross_entropy(logits, targets).backward()
    once = logits.grad.copy()
    cross_entropy(logits, targets).backward()
    assert np.allclose(logits.grad, 2.0 * once, atol=1e-6)
