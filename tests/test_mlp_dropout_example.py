import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from examples.train_mlp_with_dropout import train

REQUIRED_KEYS = (
    "model",
    "initial_loss",
    "final_loss",
    "final_accuracy",
    "losses",
    "accuracies",
    "final_validation_loss",
    "final_validation_accuracy",
    "validation_losses",
    "validation_accuracies",
)


def test_returns_required_keys_and_lengths():
    stats = train(epochs=30, seed=0, verbose=False)
    for key in REQUIRED_KEYS:
        assert key in stats
    for history in ("losses", "accuracies", "validation_losses", "validation_accuracies"):
        assert len(stats[history]) == 31  # epochs + 1


def test_final_metrics_match_history_tails():
    stats = train(epochs=30, seed=0, verbose=False)
    assert stats["final_loss"] == stats["losses"][-1]
    assert stats["final_accuracy"] == stats["accuracies"][-1]
    assert stats["final_validation_loss"] == stats["validation_losses"][-1]
    assert stats["final_validation_accuracy"] == stats["validation_accuracies"][-1]
    assert stats["initial_loss"] == stats["losses"][0]


def test_model_learns_the_circles():
    stats = train(epochs=200, seed=0, verbose=False)
    assert stats["final_loss"] < stats["initial_loss"]
    # With seed 0 this reaches 100% on both splits; the thresholds
    # leave slack without accepting a model that failed to learn.
    assert stats["final_accuracy"] >= 0.95
    assert stats["final_validation_accuracy"] >= 0.90
    assert np.isfinite(stats["final_validation_loss"])


def test_deterministic_with_seed():
    stats_a = train(epochs=40, seed=0, verbose=False)
    stats_b = train(epochs=40, seed=0, verbose=False)
    assert stats_a["losses"] == stats_b["losses"]
    assert stats_a["validation_losses"] == stats_b["validation_losses"]


def test_returned_model_is_in_eval_mode():
    stats = train(epochs=10, seed=0, verbose=False)
    model = stats["model"]
    assert model.training is False
    # Deterministic predictions: dropout is inactive.
    from tensorforge import Tensor

    x = Tensor(np.array([[0.5, 0.5], [1.5, 0.0]]))
    assert np.array_equal(model(x).data, model(x).data)
