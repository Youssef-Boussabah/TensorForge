import numpy as np
import pytest

from tensorforge import (
    Tensor,
    accuracy,
    binary_accuracy,
    evaluate_binary_classifier,
    evaluate_classifier,
)
from tensorforge.nn import Dropout, Linear, Module, Sequential


class ModeProbe(Module):
    """Identity layer that records what mode it was in when called."""

    def __init__(self):
        self.modes_seen = []

    def forward(self, x):
        self.modes_seen.append(self.training)
        return x

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


def test_binary_accuracy_on_known_logits():
    logits = [2.0, -1.0, 0.5, -3.0]  # predictions: [1, 0, 1, 0]
    assert binary_accuracy(logits, [1, 0, 1, 0]) == 1.0
    assert binary_accuracy(logits, [0, 0, 1, 0]) == 0.75
    assert binary_accuracy(logits, [0, 1, 0, 1]) == 0.0


def test_binary_accuracy_zero_logit_predicts_one():
    assert binary_accuracy([0.0], [1]) == 1.0
    assert binary_accuracy([0.0], [0]) == 0.0


def test_binary_accuracy_input_types():
    logits = np.array([2.0, -1.0])
    targets = [1, 0]
    a = binary_accuracy(logits, targets)
    b = binary_accuracy(Tensor(logits), Tensor(np.array([1.0, 0.0])))
    c = binary_accuracy(logits.tolist(), np.array(targets))
    assert a == b == c == 1.0
    assert type(a) is float


def test_binary_accuracy_column_logits_flat_targets():
    logits = np.array([[2.0], [-1.0], [0.5]])  # (3, 1)
    assert binary_accuracy(logits, [1, 0, 1]) == 1.0
    assert binary_accuracy(logits, [[1], [0], [1]]) == 1.0


def test_binary_accuracy_non_binary_targets_raise():
    with pytest.raises(ValueError, match="0 and 1"):
        binary_accuracy([1.0, -1.0], [1, 2])


def test_binary_accuracy_incompatible_shapes_raise():
    with pytest.raises(ValueError):
        binary_accuracy([[1.0], [2.0]], [1, 0, 1])
    with pytest.raises(ValueError):
        binary_accuracy([1.0, 2.0], [[1], [0]])


def test_evaluate_classifier_is_read_only():
    model = _known_classifier()
    X = np.array([[5.0, 0.0], [0.0, 5.0]])
    weight_before = model.weight.data.copy()

    evaluate_classifier(model, X, [0, 1])

    # No gradients created, no parameters touched.
    assert model.weight.grad is None
    assert model.bias.grad is None
    assert np.array_equal(model.weight.data, weight_before)


# ---------------------------------------------------------------------------
# Eval-safe evaluation (v2.0)
# ---------------------------------------------------------------------------

X2 = np.array([[5.0, 0.0], [0.0, 5.0]])


def test_evaluate_classifier_runs_in_eval_mode_and_restores_training():
    probe = ModeProbe()
    model = Sequential(probe, _known_classifier())
    assert model.training  # starts in training mode

    evaluate_classifier(model, X2, [0, 1])

    assert probe.modes_seen == [False]  # the forward pass ran in eval mode
    assert model.training is True       # ...and training mode came back
    assert probe.training is True


def test_evaluate_classifier_leaves_eval_model_in_eval():
    probe = ModeProbe()
    model = Sequential(probe, _known_classifier()).eval()

    evaluate_classifier(model, X2, [0, 1])

    assert probe.modes_seen == [False]
    assert model.training is False


def test_evaluate_classifier_deterministic_with_dropout():
    np.random.seed(0)
    model = Sequential(Linear(2, 4), Dropout(p=0.9, seed=0), Linear(4, 3))
    a = evaluate_classifier(model, X2, [0, 1])
    b = evaluate_classifier(model, X2, [0, 1])
    assert a == b  # dropout was inactive, so nothing random happened


def test_evaluate_classifier_with_dropout_is_read_only():
    np.random.seed(0)
    model = Sequential(Linear(2, 4), Dropout(p=0.5, seed=0), Linear(4, 3))
    state_before = model.state_dict()
    evaluate_classifier(model, X2, [0, 1])
    for name, param in model.named_parameters():
        assert param.grad is None
        assert np.array_equal(param.data, state_before[name])


def _known_binary_classifier():
    """Linear(2, 1) with weights [1, 1]: the logit is x1 + x2."""
    layer = Linear(2, 1)
    layer.weight.data = np.ones((2, 1))
    layer.bias.data = np.zeros(1)
    return layer


def test_evaluate_binary_classifier_returns_float_dict():
    model = _known_binary_classifier()
    result = evaluate_binary_classifier(model, [[1.0, 1.0], [-1.0, -1.0]], [1, 0])
    assert set(result) == {"loss", "accuracy"}
    assert type(result["loss"]) is float and np.isfinite(result["loss"])
    assert type(result["accuracy"]) is float
    assert result["accuracy"] == 1.0


def test_evaluate_binary_classifier_known_values():
    model = _known_binary_classifier()
    X = np.array([[1.0, 1.0], [-1.0, -1.0], [2.0, -1.0]])  # logits [2, -2, 1]
    y = [1, 0, 0]  # predictions [1, 0, 1] -> 2/3 correct
    result = evaluate_binary_classifier(model, X, y)
    assert np.allclose(result["accuracy"], 2.0 / 3.0)
    logits = np.array([2.0, -2.0, 1.0])
    expected_loss = (
        np.maximum(logits, 0) - logits * np.array(y) + np.log1p(np.exp(-np.abs(logits)))
    ).mean()
    assert np.allclose(result["loss"], expected_loss)


def test_evaluate_binary_classifier_input_types():
    model = _known_binary_classifier()
    X = np.array([[1.0, 1.0], [-1.0, -1.0]])
    y = [1, 0]
    a = evaluate_binary_classifier(model, X, y)
    b = evaluate_binary_classifier(model, Tensor(X), Tensor(np.array(y, dtype=float)))
    c = evaluate_binary_classifier(model, X.tolist(), np.array(y))
    assert a == b == c


def test_evaluate_binary_classifier_mode_switch_and_restore():
    probe = ModeProbe()
    model = Sequential(probe, _known_binary_classifier())

    evaluate_binary_classifier(model, X2, [1, 0])
    assert probe.modes_seen == [False]
    assert model.training is True

    model.eval()
    evaluate_binary_classifier(model, X2, [1, 0])
    assert probe.modes_seen == [False, False]
    assert model.training is False


def test_evaluate_binary_classifier_deterministic_with_dropout():
    np.random.seed(0)
    model = Sequential(Linear(2, 4), Dropout(p=0.9, seed=0), Linear(4, 1))
    a = evaluate_binary_classifier(model, X2, [1, 0])
    b = evaluate_binary_classifier(model, X2, [1, 0])
    assert a == b


def test_evaluate_binary_classifier_is_read_only():
    np.random.seed(0)
    model = Sequential(Linear(2, 4), Dropout(p=0.5, seed=0), Linear(4, 1))
    state_before = model.state_dict()
    evaluate_binary_classifier(model, X2, [1, 0])
    for name, param in model.named_parameters():
        assert param.grad is None
        assert np.array_equal(param.data, state_before[name])
