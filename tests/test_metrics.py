import numpy as np

from tensorforge import Tensor, accuracy, evaluate_classifier
from tensorforge.nn import Linear

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


def _known_classifier():
    """A Linear(2, 2) with hand-set weights: predicts the argmax input."""
    layer = Linear(2, 2)
    layer.weight.data = np.eye(2)
    layer.bias.data = np.zeros(2)
    return layer


def test_evaluate_classifier_returns_float_dict():
    model = _known_classifier()
    result = evaluate_classifier(model, np.array([[5.0, 0.0], [0.0, 5.0]]), [0, 1])
    assert set(result) == {"loss", "accuracy"}
    assert type(result["loss"]) is float and np.isfinite(result["loss"])
    assert type(result["accuracy"]) is float


def test_evaluate_classifier_accuracy_on_known_model():
    model = _known_classifier()
    X = np.array([[5.0, 0.0], [0.0, 5.0], [3.0, 0.0]])
    # The model predicts [0, 1, 0]; targets [0, 1, 1] -> 2/3 correct.
    result = evaluate_classifier(model, X, [0, 1, 1])
    assert np.allclose(result["accuracy"], 2.0 / 3.0)
    # Loss matches cross-entropy computed by hand from the logits.
    logits = X @ np.eye(2)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    expected_loss = -log_probs[np.arange(3), [0, 1, 1]].mean()
    assert np.allclose(result["loss"], expected_loss)


def test_evaluate_classifier_accepts_tensor_and_list_inputs():
    model = _known_classifier()
    X = np.array([[5.0, 0.0], [0.0, 5.0]])
    y = [0, 1]
    a = evaluate_classifier(model, X, y)
    b = evaluate_classifier(model, Tensor(X), Tensor(np.array(y, dtype=float)))
    c = evaluate_classifier(model, X.tolist(), np.array(y))
    assert a == b == c
    assert a["accuracy"] == 1.0


def test_evaluate_classifier_is_read_only():
    model = _known_classifier()
    X = np.array([[5.0, 0.0], [0.0, 5.0]])
    weight_before = model.weight.data.copy()

    evaluate_classifier(model, X, [0, 1])

    # No gradients created, no parameters touched.
    assert model.weight.grad is None
    assert model.bias.grad is None
    assert np.array_equal(model.weight.data, weight_before)
