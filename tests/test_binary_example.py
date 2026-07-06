import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from examples.train_binary_classification import train


def test_returns_required_keys():
    stats = train(epochs=50, seed=0, validation_split=0.0, verbose=False)
    for key in ("model", "initial_loss", "final_loss", "final_accuracy", "losses", "accuracies"):
        assert key in stats
    assert "final_validation_loss" not in stats  # no validation requested
    assert len(stats["losses"]) == 51
    assert len(stats["accuracies"]) == 51


def test_losses_decrease_and_accuracy_is_high():
    stats = train(epochs=200, seed=0, validation_split=0.0, verbose=False)
    assert stats["final_loss"] < stats["initial_loss"]
    assert stats["final_accuracy"] >= 0.90
    assert np.isfinite(stats["final_loss"])


def test_validation_keys_and_lengths():
    stats = train(epochs=100, seed=0, validation_split=0.25, verbose=False)
    for key in (
        "final_validation_loss",
        "final_validation_accuracy",
        "validation_losses",
        "validation_accuracies",
    ):
        assert key in stats
    assert len(stats["validation_losses"]) == 101
    assert len(stats["validation_accuracies"]) == 101
    assert np.isfinite(stats["final_validation_loss"])
    assert 0.0 <= stats["final_validation_accuracy"] <= 1.0
    assert stats["final_validation_accuracy"] >= 0.85


def test_deterministic_with_seed():
    stats_a = train(epochs=50, seed=0, validation_split=0.25, verbose=False)
    stats_b = train(epochs=50, seed=0, validation_split=0.25, verbose=False)
    assert stats_a["losses"] == stats_b["losses"]
    assert stats_a["validation_losses"] == stats_b["validation_losses"]
