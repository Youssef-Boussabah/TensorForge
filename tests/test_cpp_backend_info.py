"""Tests for the backend introspection helpers.

Unlike the math-kernel tests, most of these run whether or not the
backend is built — the introspection API is exactly the part that must
be safe to call in either state. The guardrail tests cross-check every
advertised capability against the real objects, so backend_info() can
never silently drift out of date (repair milestone, Stage 9).
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
        "dtype",
        "device",
        "supported_dtypes",
        "supported_devices",
        "raw_kernels",
        "kernels",
        "storage_object",
        "tensor_view",
        "tensor_core",
        "tensor_core_ops",
        "tensor_core_kernels",
        "tensor_object",
        "autograd_ops",
        "native_modules",
        "native_losses",
        "native_optimizers",
        "state_support",
        "unsupported",
        "broadcasting",
        "native_autograd",
        "stable_framework_integration",
        "build_instructions",
    }
    assert info["name"] == "cpp"
    assert info["experimental"] is True
    assert info["storage_object"] == "NativeStorage"
    assert info["tensor_view"] == "NativeTensorView"
    assert info["tensor_core"] == "NativeTensorCore"
    assert info["tensor_object"] == "NativeTensor"
    assert info["dtype"] == "float64"
    assert info["device"] == "cpu"
    assert info["supported_dtypes"] == ("float64",)
    assert info["supported_devices"] == ("cpu",)
    assert tuple(info["kernels"]) == EXPECTED_KERNELS
    assert tuple(info["raw_kernels"]) == EXPECTED_KERNELS
    # The historical tensor-core registry stays frozen at the original 5.
    assert info["tensor_core_kernels"] == ("relu", "add", "subtract", "multiply", "matmul")
    # The complete inventory is a superset that adds the later ops/views.
    assert set(info["tensor_core_kernels"]) <= set(info["tensor_core_ops"])
    assert "sqrt" in info["tensor_core_ops"] and "narrow" in info["tensor_core_ops"]
    assert isinstance(info["available"], bool)
    assert info["available"] == cpp.is_available()
    assert "cpp/build.py" in info["build_instructions"]


def test_backend_info_reports_accurate_integration_flags():
    info = cpp.backend_info()
    # Broadcasting IS supported at the tensor-core level (was stale-False).
    assert info["broadcasting"] is True
    # Native autograd and native optimizers DO exist (were stale-False).
    assert info["native_autograd"] is True
    assert info["native_optimizers"] == ("NativeSGD", "NativeAdam")
    assert info["native_losses"] == ("NativeMSELoss",)
    # But the native line is still not wired into the stable framework.
    assert info["stable_framework_integration"] is False


def test_unsupported_list_names_only_absent_capabilities():
    info = cpp.backend_info()
    # None of the Phase-D-and-beyond names may leak into the implemented
    # lists — the boundary must stay honest.
    implemented = (
        set(info["tensor_core_ops"])
        | set(info["autograd_ops"])
        | set(info["raw_kernels"])
    )
    for name in info["unsupported"]:
        assert name not in implemented


# --- guardrails: the advertised capabilities must match reality ------------

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="backend not built"
)


@needs_native
def test_advertised_tensor_core_ops_exist():
    for op in cpp.TENSOR_CORE_OPS:
        assert hasattr(cpp.NativeTensorCore, op), op


@needs_native
def test_advertised_autograd_ops_exist():
    from tensorforge.experimental import NativeTensor

    for op in cpp.AUTOGRAD_OPS:
        assert hasattr(NativeTensor, op), op


@needs_native
def test_advertised_raw_kernels_are_callable_functions():
    for name in cpp.RAW_KERNELS:
        assert callable(getattr(cpp, name)), name


def test_advertised_native_stack_names_import():
    # This runs even unbuilt: the experimental package imports lazily.
    import tensorforge.experimental as experimental

    for name in (
        cpp.NATIVE_MODULES + cpp.NATIVE_LOSSES + cpp.NATIVE_OPTIMIZERS
    ):
        assert hasattr(experimental, name), name


def test_advertised_state_support_names_import():
    import tensorforge.experimental as experimental
    from tensorforge.experimental import NativeModule

    assert hasattr(NativeModule, "state_dict")
    assert hasattr(NativeModule, "load_state_dict")
    assert hasattr(experimental, "save_native_checkpoint")
    assert hasattr(experimental, "load_native_checkpoint")


@pytest.mark.skipif(not cpp.is_available(), reason="backend not built")
def test_is_available_true_when_built():
    assert cpp.is_available() is True
    assert cpp.backend_info()["available"] is True
