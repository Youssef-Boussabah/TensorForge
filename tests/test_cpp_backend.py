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


# Kernel names are parametrized as strings and resolved with getattr
# inside the tests, so collection works even when the backend is not
# built (cpp is None and the module-level skip applies).
BINARY_OPS = (
    ("elementwise_add", np.add),
    ("elementwise_subtract", np.subtract),
    ("elementwise_multiply", np.multiply),
    ("elementwise_divide", np.divide),
)


def test_backend_imports_and_exposes_all_functions():
    for name in ("elementwise_add", "elementwise_subtract",
                 "elementwise_multiply", "elementwise_divide", "relu"):
        assert callable(getattr(cpp, name))


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


# ---------------------------------------------------------------------------
# v0.2 kernels: subtract, multiply, divide, relu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,numpy_op", BINARY_OPS)
def test_binary_ops_match_numpy(name, numpy_op):
    rng = np.random.default_rng(2)
    op = getattr(cpp, name)
    for shape in ((6,), (3, 4), (2, 3, 2)):
        a = rng.normal(size=shape)
        b = rng.normal(size=shape) + 2.0  # keep divide away from 0 here
        assert np.array_equal(op(a, b), numpy_op(a, b))  # bit-for-bit


@pytest.mark.parametrize("name,numpy_op", BINARY_OPS)
def test_binary_ops_reject_broadcasting_and_mismatch(name, numpy_op):
    op = getattr(cpp, name)
    with pytest.raises(ValueError, match="no broadcasting"):
        op(np.ones((3, 2)), np.ones(2))  # broadcastable, still rejected
    with pytest.raises(ValueError, match="identical shapes"):
        op(np.ones(3), np.ones(4))


@pytest.mark.parametrize("name,numpy_op", BINARY_OPS)
def test_binary_ops_return_new_float64_arrays_without_mutation(name, numpy_op):
    op = getattr(cpp, name)
    a = np.array([1.0, 2.0])
    b = np.array([4.0, 8.0])
    out = op(a, b)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out is not a and out is not b
    assert not np.shares_memory(out, a) and not np.shares_memory(out, b)
    assert np.array_equal(a, [1.0, 2.0])
    assert np.array_equal(b, [4.0, 8.0])


def test_divide_matches_numpy_for_zero_denominators():
    a = np.array([1.0, -1.0, 0.0, 5.0])
    b = np.array([0.0, 0.0, 0.0, 2.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = a / b  # [inf, -inf, nan, 2.5]
    result = cpp.elementwise_divide(a, b)
    assert np.array_equal(result, expected, equal_nan=True)


def test_relu_matches_numpy():
    rng = np.random.default_rng(3)
    for shape in ((7,), (4, 3), (2, 2, 3)):
        a = rng.normal(size=shape)
        assert np.array_equal(cpp.relu(a), np.maximum(a, 0.0))


def test_relu_known_values_and_output_properties():
    a = np.array([-1.0, 0.0, 2.0])
    out = cpp.relu(a)
    assert out.tolist() == [0.0, 0.0, 2.0]
    assert out.dtype == np.float64
    assert out is not a and not np.shares_memory(out, a)
    assert np.array_equal(a, [-1.0, 0.0, 2.0])  # input untouched


def test_relu_handles_strided_and_integer_inputs():
    strided = np.arange(-6.0, 6.0)[::2]  # non-contiguous view
    assert np.array_equal(cpp.relu(strided), np.maximum(strided, 0.0))
    assert np.array_equal(cpp.relu([-2, 3]), [0.0, 3.0])  # ints converted
