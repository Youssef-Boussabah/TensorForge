"""Checks that every example follows the project pattern:
a train() function tests can import, and a main() entry point."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from examples import train_linear_regression, train_multiclass, train_xor

EXAMPLE_MODULES = [train_linear_regression, train_xor, train_multiclass]


@pytest.mark.parametrize("module", EXAMPLE_MODULES, ids=lambda m: m.__name__)
def test_example_has_train_and_main(module):
    assert hasattr(module, "train") and callable(module.train)
    assert hasattr(module, "main") and callable(module.main)


def test_linear_regression_small_run():
    model, loss = train_linear_regression.train(epochs=5, seed=0)
    assert np.isfinite(float(loss.data))
    assert model.weight.data.shape == (1, 1)


def test_xor_small_run():
    model, loss = train_xor.train(epochs=5, seed=0)
    assert np.isfinite(float(loss.data))
    assert model(train_xor.Tensor([[0.0, 1.0]])).data.shape == (1, 1)


def test_multiclass_small_run():
    stats = train_multiclass.train(epochs=5, seed=0, verbose=False)
    assert np.isfinite(stats["final_loss"])
    assert 0.0 <= stats["final_accuracy"] <= 1.0
