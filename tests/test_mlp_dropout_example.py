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


def test_grad_clipping_returns_gradient_norms():
    stats = train(epochs=30, seed=0, verbose=False, max_grad_norm=5.0)
    assert "gradient_norms" in stats
    assert len(stats["gradient_norms"]) == 30  # one per optimization step
    assert all(np.isfinite(n) and n >= 0 for n in stats["gradient_norms"])
    assert np.isfinite(stats["final_loss"])
    # Default runs don't include the key.
    assert "gradient_norms" not in train(epochs=5, seed=0, verbose=False)


def test_scheduler_decays_final_lr():
    stats = train(
        epochs=60, seed=0, verbose=False,
        scheduler_step_size=20, scheduler_gamma=0.5,
    )
    assert np.isclose(stats["final_lr"], 0.03 * 0.5 ** 3)  # decayed 3 times
    assert stats["final_lr"] < 0.03
    # Without a scheduler the lr never moves.
    assert train(epochs=5, seed=0, verbose=False)["final_lr"] == 0.03


def test_clipping_and_scheduler_together_still_learn():
    stats = train(
        epochs=200, seed=0, verbose=False,
        max_grad_norm=5.0, scheduler_step_size=100, scheduler_gamma=0.5,
    )
    assert stats["final_accuracy"] >= 0.95
    assert stats["final_validation_accuracy"] >= 0.90


def test_deterministic_with_clipping_and_scheduler():
    kwargs = dict(
        epochs=40, seed=0, verbose=False,
        max_grad_norm=5.0, scheduler_step_size=15,
    )
    stats_a = train(**kwargs)
    stats_b = train(**kwargs)
    assert stats_a["losses"] == stats_b["losses"]
    assert stats_a["gradient_norms"] == stats_b["gradient_norms"]
    assert stats_a["final_lr"] == stats_b["final_lr"]


def test_returned_model_is_in_eval_mode():
    stats = train(epochs=10, seed=0, verbose=False)
    model = stats["model"]
    assert model.training is False
    # Deterministic predictions: dropout is inactive.
    from tensorforge import Tensor

    x = Tensor(np.array([[0.5, 0.5], [1.5, 0.0]]))
    assert np.array_equal(model(x).data, model(x).data)
