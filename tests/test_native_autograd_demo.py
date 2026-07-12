"""Tests for the native autograd demo example (Advanced C++ v2.2).

The demo needs the native C++ backend; it skips when the compiled
library is not built, matching the NativeTensor test suite. Follows the
sys.path example-import style of tests/test_examples.py. See
docs/backend_experiments.md.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp

# The examples/ folder is not a package, so put it on the path directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from native_autograd_demo import demo

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


def _reference():
    """The demo's forward and gradients, computed independently with
    NumPy: loss = mean(relu(x @ w + b)), d loss = mask / numel."""
    x = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0],
                  [-0.5, 2.0, 1.0], [2.0, 1.0, -3.0]])
    w = np.array([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0]])
    b = np.array([0.5, -0.5])
    z = x @ w + b
    activated = np.maximum(z, 0.0)
    upstream = (z > 0.0) / z.size  # relu mask times the mean's 1/count
    return {
        "x": x,
        "w": w,
        "b": b,
        "activated": activated,
        "loss": activated.mean(),
        "w_grad": x.T @ upstream,
        "b_grad": upstream.sum(axis=0),
    }


@needs_native
def test_native_autograd_demo_matches_numpy_reference():
    r = demo()
    expected = _reference()
    assert np.array_equal(r["x"], expected["x"])
    assert np.array_equal(r["w"], expected["w"])
    assert np.array_equal(r["b"], expected["b"])
    assert np.allclose(r["activated"], expected["activated"])
    assert np.isclose(r["loss"], expected["loss"])
    assert r["w_grad"].shape == expected["w"].shape
    assert r["b_grad"].shape == expected["b"].shape
    assert np.allclose(r["w_grad"], expected["w_grad"], atol=1e-12)
    assert np.allclose(r["b_grad"], expected["b_grad"], atol=1e-12)


@needs_native
def test_native_autograd_demo_relu_mask_is_nontrivial():
    # The demo is only convincing if relu actually blocks something:
    # some activations are zero (gradient blocked) and some positive.
    r = demo()
    assert (r["activated"] == 0.0).any()
    assert (r["activated"] > 0.0).any()


@needs_native
def test_native_autograd_demo_is_deterministic():
    first = demo()
    second = demo()
    assert first.keys() == second.keys()
    for key in first:
        assert np.array_equal(np.asarray(first[key]), np.asarray(second[key]))
