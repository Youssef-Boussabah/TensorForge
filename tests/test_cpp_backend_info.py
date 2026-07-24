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
        "native_metrics",
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
    assert info["native_losses"] == ("NativeMSELoss",
                                     "NativeCrossEntropyLoss")
    # E7's reporting metric inventory — neither a runtime op nor a module.
    assert info["native_metrics"] == ("native_accuracy",)
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


def test_phase_e_boundary_is_reported_honestly():
    """Phase E ships one milestone at a time. E1 shipped the exponential,
    E2 the logarithm, E3 the softmax, E4 the log-softmax, and E5 the
    cross-entropy **Core** layer; the registries must show exactly that —
    each transform implemented at the Core and autograd layers,
    cross-entropy implemented at the Core layer *only*, every later
    Phase-E capability still unsupported, and nothing in the wrong
    inventory."""
    info = cpp.backend_info()
    for shipped in ("exp", "log", "softmax", "log_softmax"):
        assert shipped in info["tensor_core_ops"], shipped
        assert shipped in info["autograd_ops"], shipped
        assert shipped not in info["unsupported"], shipped
        # An operation, not a module, a loss, or a raw-buffer kernel.
        assert shipped not in info["native_modules"], shipped
        assert shipped not in info["native_losses"], shipped
        assert shipped not in info["raw_kernels"], shipped
    # ("log_softmax" shipped in E4 as a distinct fused capability, its own
    # log-sum-exp kernel — design §4.4 forbids composing it from the
    # shipped "log" and "softmax", and it did not become a module.)
    assert "log_softmax" not in info["native_modules"]
    # E5 shipped the layer-qualified cross-entropy Core wrappers and E6
    # the differentiable operation over them. The two stay separate
    # entries at separate layers: Core wrappers in tensor_core_ops, the
    # bare operation name in autograd_ops.
    for core_op in ("cross_entropy_forward", "cross_entropy_backward"):
        assert core_op in info["tensor_core_ops"], core_op
        assert core_op not in info["autograd_ops"], core_op
        assert core_op not in info["unsupported"], core_op
        assert core_op not in info["native_modules"], core_op
        assert core_op not in info["native_losses"], core_op
        assert core_op not in info["raw_kernels"], core_op
    assert "cross_entropy" in info["autograd_ops"]
    assert "cross_entropy" not in info["tensor_core_ops"]
    assert "cross_entropy" not in info["unsupported"]
    # E7 shipped the public surface, each name into exactly one
    # layer-appropriate inventory: the loss module into native_losses,
    # the reporting helper into the new native_metrics.
    assert "NativeCrossEntropyLoss" in info["native_losses"]
    assert "native_accuracy" in info["native_metrics"]
    for shipped in ("NativeCrossEntropyLoss", "native_accuracy"):
        assert shipped not in info["unsupported"], shipped
        assert shipped not in info["tensor_core_ops"], shipped
        assert shipped not in info["autograd_ops"], shipped
        assert shipped not in info["raw_kernels"], shipped
        assert shipped not in info["native_modules"], shipped
    assert "NativeCrossEntropyLoss" not in info["native_metrics"]
    assert "native_accuracy" not in info["native_losses"]
    assert info["native_metrics"] == cpp.NATIVE_METRICS == ("native_accuracy",)
    # The line is still float64/cpu only and still separate from stable.
    assert info["supported_dtypes"] == ("float64",)
    assert info["supported_devices"] == ("cpu",)
    assert info["stable_framework_integration"] is False


def test_e8_added_no_capability_inventory_entry():
    """E8 is the deterministic classification training and exact
    checkpoint-resume proof: an example plus integration tests. A proof
    is an integration *result*, never a capability, so every inventory
    must be exactly what E7 left behind — and no training, classifier,
    dataset, or checkpoint-resume name may appear in any of them."""
    info = cpp.backend_info()
    assert tuple(info["raw_kernels"]) == EXPECTED_KERNELS
    # "NativeLayerNorm" joined this tuple in Phase F milestone F2 (the first
    # native normalization module) and "NativeBatchNorm1d" in milestone F3
    # (the first stateful one) — both composed from existing operations,
    # and both unrelated to E8, which still added no module of its own.
    assert info["native_modules"] == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d",
    )
    assert info["native_losses"] == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert info["native_metrics"] == ("native_accuracy",)
    assert info["native_optimizers"] == ("NativeSGD", "NativeAdam")
    assert info["autograd_ops"][-1] == "cross_entropy"
    # E8 added no state capability. "persistent_buffers" joined this
    # tuple in Phase F milestone F1 as *reconciliation* of a capability
    # that already existed (register_buffer / buffers / named_buffers,
    # persistent buffers in state_dict and checkpoints) — not as anything
    # E8 or any classification milestone contributed.
    assert info["state_support"] == (
        "persistent_buffers",
        "state_dict", "load_state_dict",
        "save_native_checkpoint", "load_native_checkpoint",
    )
    for inventory in ("raw_kernels", "tensor_core_ops", "autograd_ops",
                      "native_modules", "native_losses", "native_metrics",
                      "native_optimizers"):
        for banned in ("train", "classifier", "checkpoint_resume", "example",
                       "dataset", "accuracy_kernel"):
            offenders = [name for name in info[inventory]
                         if banned in name.lower()]
            assert offenders == [], (inventory, banned, offenders)
    # No accuracy/argmax/training C ABI symbol was invented for the proof.
    for absent in ("tf_core_accuracy", "tf_core_argmax", "tf_core_train_step"):
        assert absent not in cpp._CHECKED_KERNELS, absent
    # The proof persists nothing new: still float64/cpu, still separate.
    assert info["supported_dtypes"] == ("float64",)
    assert info["supported_devices"] == ("cpu",)
    assert info["stable_framework_integration"] is False


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


def test_f3_reports_native_batchnorm1d_and_keeps_batchnorm_unsupported():
    """Phase F milestone F3 shipped `NativeBatchNorm1d`, so
    `backend_info()` must report it as a **module** — and only as a
    module. The unqualified `batchnorm` capability stays unsupported
    until F4 ships `NativeBatchNorm2d` as well, so the registry must not
    advertise a normalization operation, kernel, or C ABI symbol either.
    """
    import tensorforge.experimental as experimental

    info = cpp.backend_info()
    assert "NativeBatchNorm1d" in info["native_modules"]
    assert hasattr(experimental, "NativeBatchNorm1d")
    # A module, and nothing else.
    for key in ("raw_kernels", "tensor_core_ops", "autograd_ops",
                "native_losses", "native_metrics", "native_optimizers",
                "state_support", "unsupported"):
        assert "NativeBatchNorm1d" not in tuple(info[key]), key
    # The NCHW shape has not shipped, so the unqualified name stays.
    assert "NativeBatchNorm2d" not in info["native_modules"]
    assert not hasattr(experimental, "NativeBatchNorm2d")
    assert "batchnorm" in info["unsupported"]
    # And F3 introduced no normalization primitive at any layer.
    for name in ("batch_norm", "batchnorm", "layer_norm", "layernorm"):
        assert name not in tuple(info["raw_kernels"]), name
        assert name not in tuple(info["tensor_core_ops"]), name
        assert name not in tuple(info["autograd_ops"]), name
    for symbol in ("tf_core_batch_norm", "tf_core_batch_norm_forward",
                   "tf_core_batch_norm_backward"):
        assert symbol not in cpp._CHECKED_KERNELS, symbol
    assert info["supported_dtypes"] == ("float64",)
    assert info["supported_devices"] == ("cpu",)
    assert info["stable_framework_integration"] is False
