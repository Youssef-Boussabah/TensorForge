import sys
from pathlib import Path

import numpy as np

# The examples/ folder is not a package, so put it on the path directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from train_regression import train
from train_xor import train as train_xor

from tensorforge import Tensor


def test_regression_example_learns_weight_and_bias():
    model, loss = train(epochs=300, lr=0.05, seed=0)
    assert float(loss.data) < 1e-3
    assert np.allclose(model.weight.data, [[2.0]], atol=0.05)
    assert np.allclose(model.bias.data, [1.0], atol=0.05)


def test_regression_example_is_deterministic():
    model_a, loss_a = train(epochs=50, lr=0.05, seed=0)
    model_b, loss_b = train(epochs=50, lr=0.05, seed=0)
    assert np.array_equal(model_a.weight.data, model_b.weight.data)
    assert np.array_equal(model_a.bias.data, model_b.bias.data)
    assert np.array_equal(loss_a.data, loss_b.data)


def test_xor_example_reaches_low_loss():
    _, loss = train_xor(epochs=3000, lr=0.5, seed=0)
    assert float(loss.data) < 0.01


def test_xor_example_classifies_correctly():
    model, _ = train_xor(epochs=3000, lr=0.5, seed=0)
    inputs = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = np.array([[0.0], [1.0], [1.0], [0.0]])
    preds = model(inputs).data
    # Rounding each sigmoid output must reproduce the XOR truth table.
    assert np.array_equal(np.round(preds), targets)
    # And each prediction should be confidently on the right side.
    assert np.all(np.abs(preds - targets) < 0.1)


def test_xor_example_is_deterministic():
    model_a, loss_a = train_xor(epochs=200, lr=0.5, seed=0)
    model_b, loss_b = train_xor(epochs=200, lr=0.5, seed=0)
    x = Tensor([[0.0, 1.0]])
    assert np.array_equal(model_a(x).data, model_b(x).data)
    assert np.array_equal(loss_a.data, loss_b.data)
