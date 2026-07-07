"""Tests for the backend introspection helpers.

Unlike the math-kernel tests, these run whether or not the backend is
built — the introspection API is exactly the part that must be safe
to call in either state.
"""

import pytest

from tensorforge.backends import cpp

EXPECTED_KERNELS = (
    "elementwise_add",
    "elementwise_subtract",
    "elementwise_multiply",
    "elementwise_divide",
    "relu",
    "matmul",
    "matmul_tiled",
)


def test_module_imports_without_the_compiled_library():
    # Just importing tensorforge.backends.cpp must never require the
    # compiled library — reaching this line proves it.
    assert hasattr(cpp, "elementwise_add")


def test_is_available_returns_a_bool():
    assert isinstance(cpp.is_available(), bool)


def test_build_instructions_contains_the_commands():
    text = cpp.build_instructions()
    assert isinstance(text, str)
    assert "uv run python cpp/build.py" in text
    assert "uv sync --group cpp" in text
    assert "experimental" in text.lower()


def test_list_kernels_is_complete_and_stably_ordered():
    assert tuple(cpp.list_kernels()) == EXPECTED_KERNELS
    assert tuple(cpp.list_kernels()) == tuple(cpp.list_kernels())


def test_backend_info_shape():
    info = cpp.backend_info()
    assert set(info) == {
        "name",
        "experimental",
        "available",
        "kernels",
        "storage_object",
        "dtype",
        "tensor_integration",
        "autograd_integration",
        "build_instructions",
    }
    assert info["storage_object"] == "NativeStorage"
    assert info["name"] == "cpp"
    assert info["experimental"] is True
    assert info["dtype"] == "float64"
    assert info["tensor_integration"] is False
    assert info["autograd_integration"] is False
    assert tuple(info["kernels"]) == EXPECTED_KERNELS
    assert isinstance(info["available"], bool)
    assert info["available"] == cpp.is_available()
    assert "cpp/build.py" in info["build_instructions"]


@pytest.mark.skipif(not cpp.is_available(), reason="backend not built")
def test_is_available_true_when_built():
    assert cpp.is_available() is True
    assert cpp.backend_info()["available"] is True