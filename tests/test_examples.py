import sys
from pathlib import Path

import numpy as np

# The examples/ folder is not a package, so put it on the path directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from train_regression import train


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
