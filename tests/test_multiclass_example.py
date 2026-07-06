import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make `examples.train_multiclass` importable as a (namespace) package.
sys.path.insert(0, str(REPO_ROOT))

from examples.train_multiclass import train


def test_train_is_importable_and_callable():
    assert callable(train)


def test_training_stats():
    stats = train(
        epochs=300,
        lr=0.1,
        hidden_size=16,
        num_samples_per_class=40,
        seed=0,
        verbose=False,
    )
    for key in ("initial_loss", "final_loss", "final_accuracy", "losses", "accuracies"):
        assert key in stats
    assert np.isfinite(stats["final_loss"])
    assert 0.0 <= stats["final_accuracy"] <= 1.0
    # One entry per epoch plus the post-training measurement.
    assert len(stats["losses"]) == 301
    assert len(stats["accuracies"]) == 301


def test_model_learns():
    stats = train(epochs=300, lr=0.1, seed=0, verbose=False)
    assert stats["final_loss"] < stats["initial_loss"]
    # With seed 0 this run reaches ~87%; 0.70 leaves margin without
    # accepting a model that failed to learn the spiral.
    assert stats["final_accuracy"] >= 0.70


def test_training_is_deterministic():
    stats_a = train(epochs=100, lr=0.1, seed=0, verbose=False)
    stats_b = train(epochs=100, lr=0.1, seed=0, verbose=False)
    assert stats_a["losses"] == stats_b["losses"]
    assert stats_a["accuracies"] == stats_b["accuracies"]


def test_minibatch_training_stats():
    stats = train(epochs=200, batch_size=16, seed=0, verbose=False)
    assert np.isfinite(stats["final_loss"])
    assert 0.0 <= stats["final_accuracy"] <= 1.0
    assert len(stats["losses"]) == 201


def test_minibatch_training_learns():
    stats = train(epochs=200, batch_size=16, seed=0, verbose=False)
    assert stats["final_loss"] < stats["initial_loss"]
    assert stats["final_accuracy"] >= 0.60


def test_minibatch_training_is_deterministic():
    stats_a = train(epochs=50, batch_size=16, seed=0, verbose=False)
    stats_b = train(epochs=50, batch_size=16, seed=0, verbose=False)
    assert stats_a["losses"] == stats_b["losses"]


def test_full_batch_default_unchanged_by_batch_size_argument():
    """batch_size=None must reproduce the original full-batch behavior."""
    stats_default = train(epochs=50, seed=0, verbose=False)
    stats_none = train(epochs=50, seed=0, verbose=False, batch_size=None)
    assert stats_default["losses"] == stats_none["losses"]


def test_script_runs_as_main():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "train_multiclass.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "final loss" in result.stdout
    assert "final accuracy" in result.stdout
