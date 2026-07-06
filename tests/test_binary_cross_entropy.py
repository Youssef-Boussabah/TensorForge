import numpy as np
import pytest

from tensorforge import Tensor, binary_cross_entropy
from tensorforge.nn import Linear
from tensorforge.optim import SGD


def _reference_bce(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float((np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))).mean())


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def test_forward_value():
    logits = Tensor([0.0, 2.0, -2.0])
    targets = [0, 1, 0]
    loss = binary_cross_entropy(logits, targets)
    assert loss.data.shape == ()
    assert np.allclose(loss.data, _reference_bce([0.0, 2.0, -2.0], targets))
    # Sanity anchor: a zero logit costs exactly log(2).
    assert np.allclose(binary_cross_entropy(Tensor(0.0), 1).data, np.log(2.0))


def test_matches_naive_formula_for_moderate_logits():
    x = np.array([0.5, -1.2, 2.0, -0.3])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = _sigmoid(x)
    naive = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    assert np.allclose(binary_cross_entropy(Tensor(x), y).data, naive)


def test_accepts_tensor_list_and_numpy_targets():
    logits = Tensor([1.0, -1.0])
    a = binary_cross_entropy(logits, [1, 0])
    b = binary_cross_entropy(logits, np.array([1, 0]))
    c = binary_cross_entropy(logits, Tensor([1.0, 0.0]))
    assert np.allclose(a.data, b.data)
    assert np.allclose(b.data, c.data)


def test_shape_alignment():
    x = np.array([[1.0], [-2.0], [0.5]])  # (3, 1) logits
    flat_targets = [1, 0, 1]
    column_targets = [[1], [0], [1]]
    a = binary_cross_entropy(Tensor(x), flat_targets)
    b = binary_cross_entropy(Tensor(x), column_targets)
    c = binary_cross_entropy(Tensor(x.ravel()), flat_targets)
    assert np.allclose(a.data, b.data)
    assert np.allclose(b.data, c.data)


def test_gradient_matches_closed_form():
    x = np.array([0.5, -1.2, 2.0, -0.3])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    logits = Tensor(x.copy(), requires_grad=True)
    binary_cross_entropy(logits, y).backward()
    expected = (_sigmoid(x) - y) / x.size
    assert np.allclose(logits.grad, expected)


def test_gradient_with_column_logits_and_flat_targets():
    x = np.array([[0.5], [-1.2]])
    logits = Tensor(x.copy(), requires_grad=True)
    binary_cross_entropy(logits, [1, 0]).backward()
    expected = (_sigmoid(x) - np.array([[1.0], [0.0]])) / 2.0
    assert logits.grad.shape == (2, 1)
    assert np.allclose(logits.grad, expected)


def test_stays_finite_for_extreme_logits():
    # Both confidently right and confidently wrong predictions.
    logits = Tensor(np.array([1000.0, -1000.0, 1000.0, -1000.0]), requires_grad=True)
    targets = [1, 0, 0, 1]
    loss = binary_cross_entropy(logits, targets)
    loss.backward()
    assert np.isfinite(loss.data)
    assert np.all(np.isfinite(logits.grad))
    # Wrong-side logits cost ~|x| each; right-side ones ~0.
    assert np.allclose(loss.data, 500.0)


def test_non_binary_targets_raise():
    logits = Tensor([1.0, -1.0])
    for bad in ([0, 2], [0.5, 1.0], [-1, 1]):
        with pytest.raises(ValueError, match="0 and 1"):
            binary_cross_entropy(logits, bad)


def test_incompatible_shapes_raise():
    with pytest.raises(ValueError):
        binary_cross_entropy(Tensor([[1.0], [2.0]]), [1, 0, 1])  # (2,1) vs (3,)
    with pytest.raises(ValueError):
        binary_cross_entropy(Tensor([1.0, 2.0]), [[1], [0]])  # (2,) vs (2,1)
    with pytest.raises(ValueError):
        binary_cross_entropy(Tensor([[1.0, 2.0]]), [1])  # (1,2) logits invalid


def test_sgd_step_reduces_loss():
    np.random.seed(0)
    model = Linear(2, 1)
    optimizer = SGD(model.parameters(), lr=0.5)
    x = Tensor([[-1.0, -1.0], [1.0, 1.0], [-2.0, -1.5], [1.5, 2.0]])
    y = [0, 1, 0, 1]

    loss_before = binary_cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()

    loss_after = binary_cross_entropy(model(x), y)
    assert float(loss_after.data) < float(loss_before.data)
