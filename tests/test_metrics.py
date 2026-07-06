import numpy as np

from tensorforge import Tensor, accuracy

LOGITS = np.array(
    [
        [3.0, 1.0, 0.0],
        [0.1, 2.0, 0.3],
        [0.2, 0.4, 5.0],
    ]
)


def test_accuracy_all_correct():
    assert accuracy(Tensor(LOGITS), np.array([0, 1, 2])) == 1.0


def test_accuracy_partially_correct():
    assert np.allclose(accuracy(Tensor(LOGITS), np.array([0, 2, 2])), 2.0 / 3.0, atol=1e-6)


def test_accuracy_all_wrong():
    assert accuracy(Tensor(LOGITS), np.array([1, 0, 0])) == 0.0


def test_accuracy_accepts_numpy_logits_and_list_targets():
    assert accuracy(LOGITS, [0, 1, 2]) == 1.0
    assert accuracy(LOGITS, Tensor([0.0, 1.0, 2.0])) == 1.0


def test_accuracy_returns_python_float():
    result = accuracy(Tensor(LOGITS), [0, 1, 2])
    assert type(result) is float
    assert not isinstance(result, Tensor)


def test_accuracy_stays_outside_autograd():
    logits = Tensor(LOGITS, requires_grad=True)
    accuracy(logits, [0, 1, 2])
    # A metric is a read-only measurement: it must not touch gradients
    # and must not produce anything backward() could run on.
    assert logits.grad is None
    assert not hasattr(accuracy(logits, [0, 1, 2]), "backward")


def test_accuracy_importable_from_nn():
    from tensorforge.nn import accuracy as nn_accuracy

    assert nn_accuracy is accuracy
