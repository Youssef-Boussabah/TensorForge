"""Tests for the explicit backend selection API (Stage 1).

The NumPy backend is always available; the native backend is
constructible whether or not the compiled library is built, and its
operations skip (or raise helpfully) when it is not. Selecting a
backend never routes operations implicitly. See
docs/dispatch_design.md.
"""

import numpy as np
import pytest

from tensorforge.backends import available_backends, get_backend

NATIVE = get_backend("native")
needs_native = pytest.mark.skipif(
    not NATIVE.available(),
    reason="experimental C++ backend not built",
)


# -- registry ---------------------------------------------------------


def test_available_backends_includes_numpy_and_native():
    names = available_backends()
    assert "numpy" in names
    assert "native" in names


def test_get_backend_returns_named_backends():
    assert get_backend("numpy").name == "numpy"
    assert get_backend("native").name == "native"


def test_get_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("cuda")
    # The message names the real options, so it's actionable.
    try:
        get_backend("nope")
    except ValueError as error:
        assert "numpy" in str(error) and "native" in str(error)


def test_get_backend_returns_stable_objects():
    assert get_backend("numpy") is get_backend("numpy")


# -- numpy backend ----------------------------------------------------


def test_numpy_backend_is_available_and_describes_itself():
    backend = get_backend("numpy")
    assert backend.available() is True
    info = backend.backend_info()
    assert info["name"] == "numpy"
    assert info["experimental"] is False
    assert info["dtype"] == "float64"


def test_numpy_backend_constructors_return_float64_arrays():
    backend = get_backend("numpy")
    for array in (
        backend.tensor_from_array([[1, 2], [3, 4]]),
        backend.zeros((2, 3)),
        backend.full((2,), 7),
    ):
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
    assert backend.zeros((2, 2)).tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert backend.full((3,), 5).tolist() == [5.0, 5.0, 5.0]


def test_numpy_backend_operations_match_numpy():
    backend = get_backend("numpy")
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[10.0, 20.0], [30.0, 40.0]])
    assert np.array_equal(backend.add(a, b), a + b)
    assert np.array_equal(backend.relu(np.array([-1.0, 0.0, 2.0])), [0.0, 0.0, 2.0])
    assert np.array_equal(backend.matmul(a, b), a @ b)


# -- native backend ---------------------------------------------------


def test_native_backend_is_constructible_and_reports_availability():
    backend = get_backend("native")
    assert backend.name == "native"
    assert isinstance(backend.available(), bool)  # never raises, built or not
    info = backend.backend_info()
    assert info["name"] == "cpp"


@needs_native
def test_native_backend_constructors_return_tensor_cores():
    from tensorforge.backends import cpp

    backend = get_backend("native")
    for tensor in (
        backend.tensor_from_array([[1.0, 2.0], [3.0, 4.0]]),
        backend.zeros((2, 3)),
        backend.full((2,), 7.0),
    ):
        assert isinstance(tensor, cpp.NativeTensorCore)
        tensor.close()


@needs_native
def test_native_backend_operations_match_numpy():
    backend = get_backend("native")
    x = np.array([[1.0, -2.0], [3.0, 4.0]])
    y = np.array([[5.0, 6.0], [7.0, 8.0]])
    a = backend.tensor_from_array(x)
    b = backend.tensor_from_array(y)
    assert np.array_equal(backend.add(a, b).to_numpy(), x + y)
    assert np.array_equal(backend.relu(a).to_numpy(), np.maximum(x, 0.0))
    assert np.allclose(backend.matmul(a, b).to_numpy(), x @ y)


@needs_native
def test_native_backend_rejects_non_core_operands_clearly():
    backend = get_backend("native")
    core = backend.tensor_from_array([1.0, 2.0])
    # A helpful TypeError, not a mysterious AttributeError on ndarray.
    with pytest.raises(TypeError, match="NativeTensorCore"):
        backend.add(np.array([1.0, 2.0]), core)
    with pytest.raises(TypeError, match="NativeTensorCore"):
        backend.relu([1.0, 2.0])
    core.close()


def test_native_backend_unavailable_raises_helpfully(monkeypatch):
    """When the library is unbuilt, operations must fail with the
    build-instructions ImportError, never a mysterious AttributeError.
    Simulated so the test runs even on a machine that has built it."""
    from tensorforge.backends import cpp, native_backend

    backend = native_backend.NativeBackend()

    def _unbuilt(*args, **kwargs):
        raise ImportError("The experimental C++ backend is not built ...")

    monkeypatch.setattr(cpp.NativeTensorCore, "zeros", staticmethod(_unbuilt))
    with pytest.raises(ImportError, match="not built"):
        backend.zeros((2, 2))


# -- isolation guarantees ---------------------------------------------


def test_direct_cpp_import_still_works():
    import tensorforge.backends.cpp as direct

    assert hasattr(direct, "elementwise_add")
    assert callable(direct.list_kernels)


def test_list_kernels_semantics_unchanged():
    from tensorforge.backends import cpp

    kernels = cpp.list_kernels()
    # Raw-buffer kernel discovery only — backend objects and TensorCore
    # methods are not kernels.
    assert "elementwise_add" in kernels
    for not_a_kernel in ("numpy", "native", "add", "NativeTensorCore"):
        assert not_a_kernel not in kernels


def test_framework_init_does_not_import_backends():
    """tensorforge's own __init__ must never import tensorforge.backends
    — the backend line stays opt-in. A static check, so it can't be
    fooled by another test having imported the registry earlier."""
    import ast
    from pathlib import Path

    init = Path(__file__).resolve().parent.parent / "src" / "tensorforge" / "__init__.py"
    tree = ast.parse(init.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any(module.startswith("tensorforge.backends") for module in imported)
