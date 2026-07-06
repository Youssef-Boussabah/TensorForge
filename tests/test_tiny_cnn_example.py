import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from examples import train_tiny_cnn


def test_has_train_and_main():
    assert callable(train_tiny_cnn.train)
    assert callable(train_tiny_cnn.main)


def test_small_run_returns_finite_stats():
    stats = train_tiny_cnn.train(epochs=10, seed=0, verbose=False)
    for key in ("model", "initial_loss", "final_loss", "final_accuracy", "losses", "accuracies"):
        assert key in stats
    assert len(stats["losses"]) == 11
    assert np.isfinite(stats["final_loss"])
    assert 0.0 <= stats["final_accuracy"] <= 1.0


def test_cnn_learns_the_bars():
    stats = train_tiny_cnn.train(epochs=100, seed=0, verbose=False)
    assert stats["final_loss"] < stats["initial_loss"]
    # With seed 0 the run reaches 100% by epoch ~50; 0.9 leaves margin.
    assert stats["final_accuracy"] >= 0.9


def test_deterministic_with_seed():
    stats_a = train_tiny_cnn.train(epochs=20, seed=0, verbose=False)
    stats_b = train_tiny_cnn.train(epochs=20, seed=0, verbose=False)
    assert stats_a["losses"] == stats_b["losses"]
    assert stats_a["accuracies"] == stats_b["accuracies"]
