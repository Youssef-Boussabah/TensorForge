"""Tests for the experimental C++ backend.

The backend must be built first (uv run python cpp/build.py). When it
is not built, these tests skip with the build instructions rather than
failing — the C++ backend is optional and the Python framework never
depends on it.
"""

import numpy as np
import pytest

try:
    from tensorforge.backends import cpp
    _IMPORT_ERROR = None
except ImportError as error:
    cpp = None
    _IMPORT_ERROR = str(error)

pytestmark = pytest.mark.skipif(
    cpp is None,
    reason=f"experimental C++ backend not built: {_IMPORT_ERROR}",
)


def test_backend_imports_and_exposes_the_function():
    assert callable(cpp.elementwise_add)


def test_matches_numpy_addition():
    rng = np.random.default_rng(0)
    for shape in ((7,), (3, 4), (2, 3, 4)):
        a = rng.normal(size=shape)
        b = rng.normal(size=shape)
        result = cpp.elementwise_add(a, b)
        assert np.array_equal(result, a + b)  # bit-for-bit, same float64 math


def test_output_shape_and_dtype():
    a = np.zeros((5, 2))
    result = cpp.elementwise_add(a, a)
    assert result.shape == (5, 2)
    assert result.dtype == np.float64
    assert result is not a  # a new array, inputs untouched


def test_inputs_are_not_mutated():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    cpp.elementwise_add(a, b)
    assert np.array_equal(a, [1.0, 2.0])
    assert np.array_equal(b, [3.0, 4.0])


def test_converts_lists_and_integers_to_float64():
    # v0.1 supports float64 only; other inputs are converted on the way in.
    result = cpp.elementwise_add([1, 2, 3], [10, 20, 30])
    assert result.dtype == np.float64
    assert np.array_equal(result, [11.0, 22.0, 33.0])


def test_non_contiguous_inputs_are_handled():
    a = np.arange(10.0)[::2]  # a strided view
    b = np.ones(5)
    assert np.array_equal(cpp.elementwise_add(a, b), a + b)


def test_broadcasting_is_rejected():
    with pytest.raises(ValueError, match="no broadcasting"):
        cpp.elementwise_add(np.ones((3, 2)), np.ones(2))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="identical shapes"):
        cpp.elementwise_add(np.ones(3), np.ones(4))


def test_larger_array_agrees_with_numpy():
    rng = np.random.default_rng(1)
    a = rng.normal(size=10_000)
    b = rng.normal(size=10_000)
    assert np.array_equal(cpp.elementwise_add(a, b), a + b)
