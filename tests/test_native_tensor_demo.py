"""Tests for the experimental NativeTensor demo example.

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

from native_tensor_demo import demo

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


@needs_native
def test_native_tensor_demo_matches_numpy():
    r = demo()
    a = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert np.array_equal(r["a"], a)
    assert np.array_equal(r["ones"], np.ones((2, 3)))
    assert np.array_equal(r["relu"], np.maximum(a, 0.0))
    assert np.array_equal(r["add"], a + 1.0)
    assert np.allclose(r["matmul"], a @ b)
    assert np.array_equal(r["reshape"], a.reshape(3, 2))
    assert np.array_equal(r["transpose"], a.T)
    assert np.array_equal(r["narrow"], a[:, 0:2])
    assert np.array_equal(r["contiguous_copy"], a.T)
    assert r["contiguous_flag"] is True
    assert np.array_equal(r["zeros"], np.zeros((2, 2)))


@needs_native
def test_native_tensor_demo_is_deterministic():
    first = demo()
    second = demo()
    assert first.keys() == second.keys()
    for key in first:
        assert np.array_equal(np.asarray(first[key]), np.asarray(second[key]))
