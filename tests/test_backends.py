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
    # v1.21: the supported dtype/device sets are advertised for discovery.
    # This backend is **float64 only** and stays that way: Phase I is a
    # native-line phase, and the NumPy reference backend gained no dtype
    # from it. Its tuple is deliberately *not* the native backend's.
    assert info["supported_dtypes"] == ("float64",)
    assert info["supported_devices"] == ("cpu",)


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


def test_numpy_backend_to_numpy_is_a_float64_copy():
    backend = get_backend("numpy")
    result = backend.to_numpy([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.tolist() == [1.0, 2.0, 3.0]
    # A copy: converting an array and mutating the result must not
    # touch the source.
    source = np.array([1.0, 2.0])
    out = backend.to_numpy(source)
    out[0] = 99.0
    assert source.tolist() == [1.0, 2.0]


def test_numpy_backend_add_follows_numpy_broadcasting():
    # The NumPy backend broadcasts because a NumPy array already is one.
    # Since v1.17 the native backend broadcasts too, inheriting it from
    # NativeTensorCore (see test_native_backend_broadcasts_like_the_core).
    backend = get_backend("numpy")
    matrix = np.ones((2, 3))
    row = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(backend.add(matrix, row), matrix + row)


def test_numpy_backend_reductions():
    backend = get_backend("numpy")
    x = np.arange(6.0).reshape(2, 3)
    assert np.allclose(backend.sum(x), x.sum())
    assert np.allclose(backend.sum(x, axis=0), x.sum(axis=0))
    assert np.allclose(backend.mean(x, axis=1, keepdims=True), x.mean(axis=1, keepdims=True))


# -- native backend ---------------------------------------------------


@needs_native
def test_native_backend_reductions_match_numpy():
    # v1.19: the native backend exposes sum/mean, delegating to the core.
    backend = get_backend("native")
    x = np.arange(6.0).reshape(2, 3)
    t = backend.tensor_from_array(x)
    assert np.allclose(backend.to_numpy(backend.sum(t)), x.sum())
    assert np.allclose(backend.to_numpy(backend.sum(t, axis=0)), x.sum(axis=0))
    assert np.allclose(
        backend.to_numpy(backend.mean(t, axis=1, keepdims=True)),
        x.mean(axis=1, keepdims=True),
    )
    # A non-NativeTensorCore operand is rejected, like the other ops.
    with pytest.raises(TypeError, match="NativeTensorCore"):
        backend.sum(x)
    t.close()


def test_native_backend_is_constructible_and_reports_availability():
    backend = get_backend("native")
    assert backend.name == "native"
    assert isinstance(backend.available(), bool)  # never raises, built or not
    info = backend.backend_info()
    assert info["name"] == "cpp"
    # v1.21: metadata contract advertised, built or not (backend_info
    # delegates to cpp.backend_info(), which never touches the library).
    assert info["supported_dtypes"] == ("float64", "float32")
    assert info["supported_devices"] == ("cpu",)


@needs_native
def test_native_backend_constructors_thread_dtype_device():
    from tensorforge.backends import cpp

    backend = get_backend("native")
    # Explicit defaults are accepted and produce float64/cpu tensors.
    for tensor in (
        backend.tensor_from_array([1.0, 2.0], dtype="float64", device="cpu"),
        backend.zeros((2, 2), dtype="float64", device="cpu"),
        backend.full((3,), 7.0, dtype="float64", device="cpu"),
    ):
        assert isinstance(tensor, cpp.NativeTensorCore)
        assert tensor.dtype == "float64" and tensor.device == "cpu"
        tensor.close()
    # float32 is threaded through the same way, since Phase I milestone I9.
    for tensor in (
        backend.tensor_from_array([1.0, 2.0], dtype="float32", device="cpu"),
        backend.zeros((2, 2), dtype="float32", device="cpu"),
        backend.full((3,), 7.0, dtype="float32", device="cpu"),
    ):
        assert isinstance(tensor, cpp.NativeTensorCore)
        assert tensor.dtype == "float32" and tensor.device == "cpu"
        tensor.close()
    # Unsupported values are rejected clearly.
    with pytest.raises(ValueError, match="float16"):
        backend.zeros((2, 2), dtype="float16")
    with pytest.raises(ValueError, match="cuda"):
        backend.tensor_from_array([1.0], device="cuda")


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
    # The message is consistent across every operation, including
    # to_numpy, and always names NativeTensorCore.
    for call in (
        lambda: backend.add(np.array([1.0, 2.0]), core),
        lambda: backend.add(core, [1.0, 2.0]),
        lambda: backend.relu([1.0, 2.0]),
        lambda: backend.matmul(core, np.zeros((2, 2))),
        lambda: backend.to_numpy(np.array([1.0, 2.0])),
    ):
        with pytest.raises(TypeError, match="NativeTensorCore"):
            call()
    core.close()


@needs_native
def test_native_backend_to_numpy_round_trips():
    backend = get_backend("native")
    x = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    core = backend.tensor_from_array(x)
    result = backend.to_numpy(core)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert np.array_equal(result, x)
    # to_numpy is the explicit exit: converting in then out reproduces
    # the data through a materialized copy.
    core.close()


@needs_native
def test_native_backend_broadcasts_like_the_core():
    # v1.17: the native backend inherits NativeTensorCore broadcasting.
    # Exact matching shapes work, and so do broadcast-compatible shapes;
    # only genuinely incompatible shapes fail clearly.
    backend = get_backend("native")
    a = backend.tensor_from_array(np.ones((2, 3)))
    b = backend.tensor_from_array(np.full((2, 3), 4.0))
    assert np.array_equal(backend.add(a, b).to_numpy(), np.full((2, 3), 5.0))

    row = backend.tensor_from_array([1.0, 2.0, 3.0])  # (3,) broadcasts
    assert np.array_equal(
        backend.add(a, row).to_numpy(), np.ones((2, 3)) + np.array([1.0, 2.0, 3.0])
    )

    bad = backend.tensor_from_array(np.ones((4, 3)))  # 2 vs 4 -> incompatible
    with pytest.raises(ValueError, match="broadcast"):
        backend.add(a, bad)
    for tensor in (a, b, row, bad):
        tensor.close()


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
