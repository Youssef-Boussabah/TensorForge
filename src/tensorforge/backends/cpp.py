"""Experimental C++ backend.

A ctypes wrapper around the compiled native kernels (see cpp/ at the
repo root). It is NOT wired into the **stable** ``tensorforge.Tensor``
or its autograd, and ``import tensorforge`` never imports it — the
native line is a separate system. On top of these kernels the
experimental package builds its own strided runtime
(``NativeTensorCore``), a Python-managed native autograd
(``NativeTensor``), and a native training stack (modules, a loss,
optimizers, checkpoints); see ``tensorforge.experimental``.

All kernels are float64/CPU only. The raw buffer kernels exposed here
(``elementwise_add`` … ``matmul_tiled``) require identical shapes, but
the ``NativeTensorCore`` binary ops **do** support NumPy-style
broadcasting (see docs/native_broadcasting_design.md).

The library is loaded lazily and the module always imports: check
``is_available()`` for readiness and ``backend_info()`` for an accurate
capability summary (raw kernels, tensor-core ops, autograd ops, native
modules/loss/optimizers, and state/checkpoint support). Calling a math
kernel while the backend is unbuilt raises ImportError with build
instructions. Native failures surface as ordinary Python exceptions via
the error contract (docs/native_abi_error_contract.md).
"""

import ctypes
import platform
from pathlib import Path

import numpy as np

_SUFFIX = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
_LIBRARY_PATH = Path(__file__).with_name("_tensorforge_cpp" + _SUFFIX)

# ---------------------------------------------------------------------------
# Backend capability inventory — the single source of truth backend_info()
# reports. Grouped by layer so introspection can distinguish raw C++
# kernels from the higher-level capabilities composed on top of them. The
# guardrail test (tests/test_cpp_backend_info.py) cross-checks every name
# here against the actual objects, so these tuples cannot silently drift
# out of date.
# ---------------------------------------------------------------------------

# Raw C++ kernels callable directly over NumPy buffers — the reference /
# benchmark set this module exposes as elementwise_add(...) etc. Require
# identical shapes (no broadcasting at this raw level).
RAW_KERNELS = (
    "elementwise_add",
    "elementwise_subtract",
    "elementwise_multiply",
    "elementwise_divide",
    "relu",
    "matmul",
    "matmul_tiled",
)
# Backwards-compatible alias (list_kernels / backend_info["kernels"]).
KERNELS = RAW_KERNELS

_BINARY_KERNELS = (
    "tf_elementwise_add",
    "tf_elementwise_subtract",
    "tf_elementwise_multiply",
    "tf_elementwise_divide",
)

# The historical, deliberately FROZEN tensor-core registry: exactly the
# five originally advertised compute ops. Kept frozen by contract (later
# core ops are not appended — the sum/mean/sqrt precedent), so existing
# consumers that pin this tuple stay stable. The complete, accurate op
# inventory lives in TENSOR_CORE_OPS below.
TENSOR_CORE_KERNELS = ("relu", "add", "subtract", "multiply", "matmul")

# The COMPLETE set of NativeTensorCore operations (the strided native
# runtime) — compute ops (each backed by C ABI kernels; binary ops
# broadcast) plus the metadata-only view ops. This is the accurate,
# non-frozen inventory backend_info() reports as "tensor_core_ops".
TENSOR_CORE_OPS = (
    "relu", "sqrt", "reciprocal",
    # Phase E stable math (E1): the exponential, a unary Core op with the
    # same odometer/contiguous pair as relu/sqrt/reciprocal. Its guarded
    # exports additionally self-validate at the ABI boundary.
    "exp",
    "add", "subtract", "multiply", "matmul",
    "sum", "mean",
    "reshape", "transpose", "T", "narrow", "contiguous_copy",
    # Phase D native Conv2d at the Core layer — layer-qualified forward and
    # (D6) backward wrappers over the exported tf_core_conv2d_* kernels.
    # These are Core operations, distinct from the differentiable
    # "conv2d" NativeTensor autograd op (in AUTOGRAD_OPS) and from the
    # NativeConv2d module (D7, in NATIVE_MODULES). The bias gradient has no
    # Core op — it composes from the existing "sum" reduction.
    "conv2d_forward",          # D3
    "conv2d_input_backward",   # D6
    "conv2d_weight_backward",  # D6
    # Phase D native MaxPool2d at the Core layer — the layer-qualified
    # forward/backward wrappers over the exported tf_core_maxpool2d_*
    # kernels. Forward computes the pooled values and (internally) the
    # private winner buffer; backward scatters an upstream gradient through
    # those saved winners (no window geometry, no input reread). These are
    # Core operations, distinct from the differentiable "maxpool2d"
    # NativeTensor autograd op (in AUTOGRAD_OPS as of D9) and from the
    # NativeMaxPool2d module (D10, in NATIVE_MODULES). The winner buffer
    # stays internal state — never a public tensor, op, or dtype.
    "maxpool2d_forward",       # D8
    "maxpool2d_backward",      # D9
)

# Operations the NativeTensor autograd layer (Phase B) differentiates.
# Phase D adds the differentiable "conv2d" fused primitive (D6): its
# backward composes the input/weight-gradient Core ops and the existing
# "sum" reduction (bias); and the differentiable "maxpool2d" primitive
# (D9), whose backward scatters through the winner buffer its own forward
# saved. These are the operations; the modules built on them (NativeConv2d,
# D7; NativeMaxPool2d, D10 — both implemented) are separate entries in
# NATIVE_MODULES. Phase E adds "exp" (E1), whose backward multiplies the
# upstream by the **saved forward output** — so, like sqrt/reciprocal, it
# records no expected parameter version.
AUTOGRAD_OPS = (
    "add", "subtract", "multiply", "relu",
    "sum", "mean", "matmul",
    "reshape", "transpose", "T", "narrow", "contiguous_copy",
    "sqrt", "reciprocal",
    "conv2d",
    "maxpool2d",
    "exp",
)

# The native training stack composed on the autograd layer (Phase C) and
# the Phase-D CNN modules, reported by name only so this module stays
# decoupled from the experimental package (the guardrail test verifies
# each name imports). "NativeConv2d" (the Conv2d *module*, D7) is the
# trainable layer over the differentiable "conv2d" op; it is distinct from
# that operation (in AUTOGRAD_OPS) and from the Core wrappers (in
# TENSOR_CORE_OPS).
NATIVE_MODULES = (
    "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
    "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
)
NATIVE_LOSSES = ("NativeMSELoss",)
NATIVE_OPTIMIZERS = ("NativeSGD", "NativeAdam")
STATE_SUPPORT = (
    "state_dict",
    "load_state_dict",
    "save_native_checkpoint",
    "load_native_checkpoint",
)

# Explicitly NOT implemented — listed so introspection is honest about the
# boundary. These names are layer-qualified where an operation and its
# module diverge, so support is never over- or under-claimed:
#   - the differentiable "conv2d" *operation* IS implemented (D3–D6:
#     forward + input/weight/bias gradients + NativeTensor autograd), so it
#     is NOT listed here — it lives in AUTOGRAD_OPS / TENSOR_CORE_OPS.
#   - "NativeConv2d" (the Conv2d *module*, D7) IS implemented (see
#     NATIVE_MODULES), so it is NOT listed here either — operation support
#     and module support are now both present for Conv2d.
#   - the differentiable "maxpool2d" *operation* IS implemented as of D9
#     (the D8 forward + private winner buffer, the D9 backward scatter, and
#     the NativeTensor autograd node), so it is NOT listed here — it lives
#     in AUTOGRAD_OPS, with its layer-qualified Core ops
#     "maxpool2d_forward"/"maxpool2d_backward" in TENSOR_CORE_OPS.
#   - "NativeMaxPool2d" (the pooling *module*, D10) IS implemented (see
#     NATIVE_MODULES), so it is NOT listed here either — operation support
#     and module support are now both present for MaxPool2d, as they are
#     for Conv2d.
# As of Phase D milestone D1, batch-preserving flatten IS implemented as
# the NativeFlatten module (see NATIVE_MODULES), so "flatten" is not listed.
# Phase D is complete (D0-D12): every CNN operation and module shipped,
# along with the deterministic end-to-end native CNN training +
# checkpoint-resume proof (D11) — a *proof*, not a capability name, so it
# has no entry in any inventory.
# The classification names below are the Phase-E surface contracted in
# docs/native_classification_design.md (milestone E0). A locked contract is
# not an implementation: each stays here until the milestone that
# implements it removes it. Milestone E1 implemented the exponential, so
# "exp" left this tuple for TENSOR_CORE_OPS and AUTOGRAD_OPS; the rest of
# Phase E (E2-E7) is still genuinely absent.
UNSUPPORTED = (
    "log", "softmax", "log_softmax", "cross_entropy",
    "NativeCrossEntropyLoss", "native_accuracy",
    "batchnorm", "layernorm", "dropout",
    "float32", "cuda", "amp",
)

# Supported native dtype/device metadata (v1.21). The native kernels are
# float64 CPU only, so these are the single legal values today. The tags
# are explicit and validated — a native tensor never claims a dtype/device
# the kernels cannot actually compute, and unsupported values are rejected
# at construction rather than silently coerced (see
# docs/native_dtype_device_metadata_design.md).
SUPPORTED_DTYPES = ("float64",)
SUPPORTED_DEVICES = ("cpu",)

# Largest element count the native int64 storage/ABI arithmetic addresses.
_INT64_MAX = 2 ** 63 - 1
# IEEE float64 represents every integer in [-(2**53), 2**53] exactly, so a
# flat plane offset stored as a float64 (the internal MaxPool2d winner
# buffer, docs/native_cnn_design.md §12) is exact iff the plane holds at
# most 2**53 elements. Proved in Python arbitrary-precision arithmetic
# before any allocation, and re-proved at the C ABI boundary.
_MAX_EXACT_WINNER_PLANE = 2 ** 53

_lib = None  # loaded lazily by _require_library()


def normalize_dtype(dtype=None):
    """Validate and canonicalize a native dtype tag.

    ``None`` means the default ``"float64"`` (the only supported dtype
    today). A non-string raises TypeError; a string outside
    ``SUPPORTED_DTYPES`` raises ValueError naming the offending value and
    the supported set. Pure Python — never touches the compiled library,
    so it is safe whether or not the backend is built."""
    if dtype is None:
        return "float64"
    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be a string or None, got {dtype!r}")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; the native runtime supports "
            f"{SUPPORTED_DTYPES}"
        )
    return dtype


def normalize_device(device="cpu"):
    """Validate and canonicalize a native device tag.

    ``None`` means the default ``"cpu"`` (the only supported device
    today). A non-string raises TypeError; a string outside
    ``SUPPORTED_DEVICES`` raises ValueError naming the offending value and
    the supported set. Pure Python — never touches the compiled library."""
    if device is None:
        return "cpu"
    if not isinstance(device, str):
        raise TypeError(f"device must be a string or None, got {device!r}")
    if device not in SUPPORTED_DEVICES:
        raise ValueError(
            f"unsupported device {device!r}; the native runtime supports "
            f"{SUPPORTED_DEVICES}"
        )
    return device


def build_instructions():
    """The commands needed to build the experimental backend."""
    return (
        "The C++ backend is experimental. Build it from the repo root:\n"
        "    uv sync --group cpp   # only if you have no C++ compiler\n"
        "    uv run python cpp/build.py"
    )


def _load_library():
    if not _LIBRARY_PATH.exists():
        raise ImportError(
            f"The experimental C++ backend is not built "
            f"(missing {_LIBRARY_PATH.name}).\n" + build_instructions()
        )
    try:
        library = ctypes.CDLL(str(_LIBRARY_PATH))
    except OSError as error:
        raise ImportError(
            f"The experimental C++ backend library exists but failed to "
            f"load ({error}). Try rebuilding it.\n" + build_instructions()
        ) from error
    f64_array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    for name in _BINARY_KERNELS:
        kernel = getattr(library, name)
        kernel.argtypes = [f64_array, f64_array, f64_array, ctypes.c_int64]
        kernel.restype = None
    library.tf_relu.argtypes = [f64_array, f64_array, ctypes.c_int64]
    library.tf_relu.restype = None
    library.tf_matmul.argtypes = [
        f64_array, f64_array, f64_array,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_matmul.restype = None
    library.tf_matmul_tiled.argtypes = [
        f64_array, f64_array, f64_array,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_matmul_tiled.restype = None
    library.tf_storage_create.argtypes = [ctypes.c_int64]
    library.tf_storage_create.restype = ctypes.c_void_p
    library.tf_storage_destroy.argtypes = [ctypes.c_void_p]
    library.tf_storage_destroy.restype = None
    library.tf_storage_size.argtypes = [ctypes.c_void_p]
    library.tf_storage_size.restype = ctypes.c_int64
    library.tf_storage_fill.argtypes = [ctypes.c_void_p, ctypes.c_double]
    library.tf_storage_fill.restype = None
    library.tf_storage_copy_from.argtypes = [ctypes.c_void_p, f64_array]
    library.tf_storage_copy_from.restype = None
    library.tf_storage_copy_to.argtypes = [ctypes.c_void_p, f64_array]
    library.tf_storage_copy_to.restype = None
    i64_array = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    library.tf_storage_materialize.argtypes = [
        ctypes.c_void_p, f64_array, i64_array, i64_array,
        ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_storage_materialize.restype = None
    # Unary core kernels share tf_core_relu's signature (one strided
    # source, one contiguous destination); sqrt/reciprocal are the
    # v3.11 optimizer math primitives and exp is the Phase-E stable-math
    # primitive (milestone E1).
    for name in ("tf_core_relu", "tf_core_sqrt", "tf_core_reciprocal",
                 "tf_core_exp"):
        kernel = getattr(library, name)
        kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, i64_array, i64_array,
            ctypes.c_int64, ctypes.c_int64,
        ]
        kernel.restype = None
    # tf_core_relu_backward shares the binary-kernel signature: it walks
    # the forward input and the upstream gradient in lockstep.
    for name in (
        "tf_core_add", "tf_core_subtract", "tf_core_multiply",
        "tf_core_relu_backward",
    ):
        kernel = getattr(library, name)
        kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            i64_array, i64_array, i64_array,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        kernel.restype = None
    # Contiguous fast-path kernels (v1.14): flat, index-free loops that
    # take numel + offsets instead of shape/strides. Selected by
    # NativeTensorCore when the operands are row-major contiguous.
    for name in ("tf_core_relu_contiguous", "tf_core_sqrt_contiguous",
                 "tf_core_reciprocal_contiguous", "tf_core_exp_contiguous"):
        kernel = getattr(library, name)
        kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
        ]
        kernel.restype = None
    for name in (
        "tf_core_add_contiguous",
        "tf_core_subtract_contiguous",
        "tf_core_multiply_contiguous",
    ):
        kernel = getattr(library, name)
        kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        kernel.restype = None
    library.tf_core_matmul.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ] + [ctypes.c_int64] * 9
    library.tf_core_matmul.restype = None
    # Reduction kernels (v1.19): tf_core_sum scatter-accumulates a strided
    # input into a fresh contiguous output using per-input-axis output
    # write-strides (0 on reduced axes); tf_storage_scale does mean's
    # in-place 1/count scaling.
    library.tf_core_sum.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, i64_array, i64_array, i64_array,
        ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_core_sum.restype = None
    library.tf_storage_scale.argtypes = [ctypes.c_void_p, ctypes.c_double]
    library.tf_storage_scale.restype = None
    # Narrow backward (v2.3): scatter the upstream gradient into a fresh
    # zero output of the parent shape. Same odometer as tf_core_sum plus a
    # base output offset (start * row-major stride of the narrowed axis).
    library.tf_core_narrow_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, i64_array, i64_array, i64_array,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_core_narrow_backward.restype = None
    # Conv2d forward (Phase D, D3): the exported wrapper over the internal
    # cross-correlation kernel. Contiguous storage only (Policy B copies at
    # the Core level); a null bias handle means "no bias"; the output is
    # caller-allocated. Handles carry per-operand offsets; the 13 trailing
    # int64s are N, C, H, W, O, kh, kw, sh, sw, ph, pw, out_h, out_w.
    library.tf_core_conv2d_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_int64,   # input handle, offset
        ctypes.c_void_p, ctypes.c_int64,   # weight handle, offset
        ctypes.c_void_p, ctypes.c_int64,   # bias handle (nullable), offset
        ctypes.c_void_p,                   # output handle
    ] + [ctypes.c_int64] * 13
    library.tf_core_conv2d_forward.restype = None
    # Conv2d backward wrappers (Phase D, D6): input- and weight-gradient
    # exports over the D4/D5 internal kernels. Contiguous storage only
    # (Policy B copies at the Core level); caller-allocated output; no bias
    # gradient symbol (that composes from the existing `sum` reduction). The
    # handle pairs differ by direction but both take the same 13 trailing
    # int64 dims as the forward wrapper (N, C, H, W, O, kh, kw, sh, sw, ph,
    # pw, out_h, out_w).
    for name in ("tf_core_conv2d_input_backward",
                 "tf_core_conv2d_weight_backward"):
        kernel = getattr(library, name)
        kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_int64,   # grad_output handle, offset
            ctypes.c_void_p, ctypes.c_int64,   # weight/input handle, offset
            ctypes.c_void_p,                   # grad_input/grad_weight handle
        ] + [ctypes.c_int64] * 13
        kernel.restype = None
    # MaxPool2d forward (Phase D, D8): the exported wrapper over the
    # internal window-maximum kernel. Contiguous storage only (Policy B
    # copies at the Core level); the output *and* the private winner buffer
    # are caller-allocated (offset 0). Only the input carries an offset;
    # the 12 trailing int64s are N, C, H, W, kh, kw, sh, sw, ph, pw, out_h,
    # out_w. There is no backward symbol yet — that is D9.
    library.tf_core_maxpool2d_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_int64,   # input handle, offset
        ctypes.c_void_p,                   # output handle
        ctypes.c_void_p,                   # winner-buffer handle
    ] + [ctypes.c_int64] * 12
    library.tf_core_maxpool2d_forward.restype = None
    # MaxPool2d backward (Phase D, D9): the exported wrapper over the
    # internal scatter-add. It takes the upstream gradient and the private
    # winner buffer (each with an offset) plus the caller-allocated
    # grad_input, and **no kernel/stride/padding metadata** — the saved
    # winners fully determine the routing, so window geometry is never
    # recomputed. The 6 trailing int64s are N, C, H, W, out_h, out_w.
    library.tf_core_maxpool2d_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_int64,   # grad_output handle, offset
        ctypes.c_void_p, ctypes.c_int64,   # winner-buffer handle, offset
        ctypes.c_void_p,                   # grad_input handle
    ] + [ctypes.c_int64] * 6
    library.tf_core_maxpool2d_backward.restype = None

    _configure_error_contract(library)
    return library


# ---------------------------------------------------------------------------
# Native error contract (see docs/native_abi_error_contract.md)
#
# No C++ exception may cross the extern "C" boundary. Each fallible
# native function clears a thread-local error slot on entry and, on any
# exception, records a status code plus message there and returns a
# benign value instead of unwinding. A ctypes ``errcheck`` hook on every
# such function reads the slot after the call and raises the matching
# Python exception, so a native failure surfaces as a normal exception at
# the call site with useful context — never a crash or a silently wrong
# result.
# ---------------------------------------------------------------------------

# TfStatus codes (kept in sync with cpp/include/tf_internal.h) mapped to
# the Python exception each becomes.
TF_OK = 0
_STATUS_EXCEPTIONS = {
    1: MemoryError,   # TF_ERROR_ALLOC
    2: ValueError,    # TF_ERROR_INVALID
    3: RuntimeError,  # TF_ERROR_RUNTIME
}

# The exported functions that participate in the error contract — every
# function that clears-on-entry and may set the slot. The unguarded
# storage/legacy kernels (destroy, size, fill, scale, copy, the raw
# elementwise/matmul kernels) never allocate, so they neither clear nor
# set the slot and must NOT carry the hook, or a stale code from an
# earlier call could be misread as their own failure.
_CHECKED_KERNELS = (
    "tf_storage_create",
    "tf_storage_materialize",
    "tf_core_relu", "tf_core_relu_contiguous",
    "tf_core_sqrt", "tf_core_sqrt_contiguous",
    "tf_core_reciprocal", "tf_core_reciprocal_contiguous",
    # Phase E, E1: the two exponential exports. Unlike the older unary
    # exports these validate their own handles/layout/spans before
    # computing, so an invalid call raises ValueError through this hook.
    "tf_core_exp", "tf_core_exp_contiguous",
    "tf_core_add", "tf_core_add_contiguous",
    "tf_core_subtract", "tf_core_subtract_contiguous",
    "tf_core_multiply", "tf_core_multiply_contiguous",
    "tf_core_relu_backward",
    "tf_core_matmul",
    "tf_core_sum",
    "tf_core_narrow_backward",
    "tf_core_conv2d_forward",
    "tf_core_conv2d_input_backward",
    "tf_core_conv2d_weight_backward",
    "tf_core_maxpool2d_forward",
    "tf_core_maxpool2d_backward",
)


def _configure_error_contract(library):
    """Declare the error-introspection ABI and attach the errcheck hook."""
    library.tf_last_error_code.argtypes = []
    library.tf_last_error_code.restype = ctypes.c_int
    library.tf_last_error_message.argtypes = []
    library.tf_last_error_message.restype = ctypes.c_char_p
    library.tf_clear_error.argtypes = []
    library.tf_clear_error.restype = None
    library.tf_test_arm_alloc_failure.argtypes = [ctypes.c_int64]
    library.tf_test_arm_alloc_failure.restype = None
    library.tf_fault_injection_available.argtypes = []
    library.tf_fault_injection_available.restype = ctypes.c_int

    def _errcheck(result, func, arguments):
        # Runs after every checked native call. The callee cleared the
        # slot on entry, so a nonzero code here is genuinely this call's
        # failure; translate it into the right Python exception with the
        # native message for context, then clear the slot.
        code = library.tf_last_error_code()
        if code != TF_OK:
            raw = library.tf_last_error_message()
            message = raw.decode("utf-8", "replace") if raw else ""
            library.tf_clear_error()
            exception = _STATUS_EXCEPTIONS.get(code, RuntimeError)
            raise exception(
                f"native backend {func.__name__} failed"
                + (f": {message}" if message else f" (status {code})")
            )
        return result

    for name in _CHECKED_KERNELS:
        getattr(library, name).errcheck = _errcheck


def _require_library():
    """Load the compiled library on first use; raise helpfully if missing."""
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


def is_available():
    """True if the compiled backend can actually be loaded.

    This attempts a real load (cached after the first success), not
    just a file-existence check. Never raises.
    """
    try:
        _require_library()
    except ImportError:
        return False
    return True


def fault_injection_available():
    """True if the compiled backend includes the test-only allocation
    fault-injection hook (it does for any build from this repo). Never
    raises; returns False when the backend is not built."""
    try:
        library = _require_library()
    except ImportError:
        return False
    return bool(library.tf_fault_injection_available())


def _arm_alloc_failure(nth=1):
    """Test-only: arm the calling thread so the ``nth`` subsequent native
    allocation attempt fails with a simulated ``std::bad_alloc`` (``nth=1``
    targets the very next allocation; ``nth <= 0`` disarms). Deterministic
    and thread-local. The native failure surfaces through the normal error
    contract as ``MemoryError``. Used by the ABI failure tests; inert in
    normal use (see docs/native_abi_error_contract.md)."""
    _require_library().tf_test_arm_alloc_failure(int(nth))


def list_kernels():
    """The experimental kernels this backend provides, in stable order."""
    return KERNELS


def backend_info():
    """An accurate capability summary of the experimental backend.

    Reports each layer separately so a caller can tell raw C++ kernels
    from the higher-level capabilities composed on them: the raw
    NumPy-buffer kernels, the ``NativeTensorCore`` runtime ops (which
    broadcast), the ``NativeTensor`` autograd ops, and the native
    training stack (modules — including the Phase-D CNN layers — the loss,
    the optimizers, and state/checkpoint support). ``stable_framework_integration`` stays ``False`` — the
    native line is deliberately separate from ``tensorforge.Tensor`` — but
    ``native_autograd`` is ``True`` and the optimizer/state lists are
    populated. Every list is sourced from the module-level inventory
    tuples, so this never drifts from the code. Safe to call whether or
    not the library is built."""
    return {
        "name": "cpp",
        "experimental": True,
        "available": is_available(),
        # dtype / device metadata (v1.21): float64/cpu only.
        "dtype": "float64",
        "device": "cpu",
        "supported_dtypes": SUPPORTED_DTYPES,
        "supported_devices": SUPPORTED_DEVICES,
        # Layered capabilities (single source of truth: the tuples above).
        "raw_kernels": RAW_KERNELS,
        "kernels": RAW_KERNELS,  # backwards-compatible alias
        "storage_object": "NativeStorage",
        "tensor_view": "NativeTensorView",
        "tensor_core": "NativeTensorCore",
        "tensor_core_kernels": TENSOR_CORE_KERNELS,  # frozen historical registry
        "tensor_core_ops": TENSOR_CORE_OPS,          # complete, accurate inventory
        "tensor_object": "NativeTensor",
        "autograd_ops": AUTOGRAD_OPS,
        "native_modules": NATIVE_MODULES,
        "native_losses": NATIVE_LOSSES,
        "native_optimizers": NATIVE_OPTIMIZERS,
        "state_support": STATE_SUPPORT,
        "unsupported": UNSUPPORTED,
        # Accurate integration flags (replace the old ambiguous
        # tensor_integration / autograd_integration pair).
        "broadcasting": True,          # NativeTensorCore binary ops broadcast
        "native_autograd": True,       # NativeTensor has reverse-mode autograd
        "stable_framework_integration": False,  # never wired into tensorforge.Tensor
        "build_instructions": build_instructions(),
    }


class NativeStorage:
    """A C++-owned float64 buffer — the storage half of a future
    tensor runtime prototype.

    Not a Tensor: it has a size but no shape, no strides, and no
    connection to Tensor/autograd. Data moves in and out by copy
    (``copy_from`` / ``to_numpy``); the raw native pointer is never
    exposed. Call ``close()`` (or use it as a context manager) to
    release the native memory; operations on a closed storage raise
    RuntimeError, and closing twice is safe.
    """

    def __init__(self, size, dtype=None, device="cpu"):
        self._handle = None  # so a failed __init__ still __del__s safely
        if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"size must be a positive int, got {size!r}")
        # dtype/device are validated *before* allocation, so an
        # unsupported request never leaks native memory.
        dtype = normalize_dtype(dtype)
        device = normalize_device(device)
        lib = _require_library()
        handle = lib.tf_storage_create(int(size))
        if not handle:
            raise MemoryError(f"could not allocate native storage of size {size}")
        self._lib = lib
        self._handle = handle
        self._size = int(size)
        self._dtype = dtype
        self._device = device

    @classmethod
    def from_array(cls, values, dtype=None, device="cpu"):
        """Create storage sized to ``values`` and copy them in.

        The input is always converted to contiguous float64 and flattened
        (the only element type the kernels compute); ``dtype``/``device``
        record the metadata and default to ``"float64"``/``"cpu"``.
        """
        array = np.ascontiguousarray(values, dtype=np.float64).ravel()
        # empty input fails size validation; dtype/device validated too
        storage = cls(int(array.size), dtype=dtype, device=device)
        storage.copy_from(array)
        return storage

    @property
    def size(self):
        """Number of float64 elements the storage holds."""
        return self._size

    @property
    def dtype(self):
        """The element type tag (``"float64"``). Readable after close."""
        return self._dtype

    @property
    def device(self):
        """The device tag (``"cpu"``). Readable after close."""
        return self._device

    def _require_open(self):
        if self._handle is None:
            raise RuntimeError("this NativeStorage has been closed")
        return self._handle

    def fill(self, value):
        """Set every element to ``value``."""
        self._lib.tf_storage_fill(self._require_open(), float(value))

    def copy_from(self, values):
        """Copy ``values`` into the storage.

        The input is converted to contiguous float64 and flattened; it
        must contain exactly ``size`` elements.
        """
        handle = self._require_open()
        array = np.ascontiguousarray(values, dtype=np.float64).ravel()
        if array.size != self._size:
            raise ValueError(
                f"copy_from needs exactly {self._size} values, got {array.size}"
            )
        self._lib.tf_storage_copy_from(handle, array)

    def to_numpy(self):
        """Return a new, independent 1-D float64 copy of the contents."""
        handle = self._require_open()
        out = np.empty(self._size, dtype=np.float64)
        self._lib.tf_storage_copy_to(handle, out)
        return out

    def close(self):
        """Release the native memory. Safe to call more than once."""
        if self._handle is not None:
            self._lib.tf_storage_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        # Defensive cleanup only — correctness never depends on when
        # (or whether) the garbage collector runs; use close().
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        state = "closed" if self._handle is None else f"size={self._size}"
        return f"NativeStorage({state})"


class NativeTensorView:
    """A logical view: NativeStorage plus shape/stride/offset metadata.

    This binds the two halves of the backend runtime prototype — the
    v0.7 layout contract and the v0.8 C++-owned buffer — into one
    object that knows which storage element each logical index means.
    Its first (and so far only) operation is contiguous
    materialization. Not a full Tensor: no math ops, no autograd, no
    connection to tensorforge.Tensor.

    Views never own the storage; closing the storage makes the view's
    operations raise. Reachable offsets are bounds-checked at
    construction, so a valid view can never read outside its storage
    (negative strides included).
    """

    def __init__(self, storage, shape, strides=None, offset=0):
        if not isinstance(storage, NativeStorage):
            raise TypeError(
                f"storage must be a NativeStorage, got {type(storage).__name__}"
            )
        storage._require_open()  # a closed storage cannot back a view
        info = shape_info(shape, strides=strides, offset=offset)

        # Bounds: each dimension contributes between 0 and
        # (dim - 1) * stride to the offset — negative strides make the
        # low end move. The whole reachable range must fit in storage.
        low = high = info["offset"]
        for dim, stride in zip(info["shape"], info["strides"]):
            contribution = (dim - 1) * stride
            if contribution >= 0:
                high += contribution
            else:
                low += contribution
        if low < 0 or high > storage.size - 1:
            raise ValueError(
                f"view reaches storage offsets [{low}, {high}], outside "
                f"the valid range [0, {storage.size - 1}]"
            )

        self._storage = storage
        self._shape = info["shape"]
        self._strides = info["strides"]
        self._offset = info["offset"]
        self._numel = info["numel"]
        self._contiguous = info["contiguous"]

    @classmethod
    def from_array(cls, values):
        """Create a contiguous view over new storage holding ``values``.

        The array's shape is preserved; its data is copied into a fresh
        NativeStorage.
        """
        array = np.ascontiguousarray(values, dtype=np.float64)
        storage = NativeStorage.from_array(array)
        return cls(storage, array.shape)

    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def offset(self):
        return self._offset

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def numel(self):
        return self._numel

    @property
    def contiguous(self):
        return self._contiguous

    def to_numpy(self):
        """Materialize the logical view into a fresh NumPy array of
        shape ``self.shape`` (row-major), copied element by element by
        the native materialization kernel."""
        handle = self._storage._require_open()
        out = np.empty(self._numel, dtype=np.float64)
        self._storage._lib.tf_storage_materialize(
            handle,
            out,
            np.asarray(self._shape, dtype=np.int64),
            np.asarray(self._strides, dtype=np.int64),
            self._offset,
            len(self._shape),
        )
        return out.reshape(self._shape)

    def contiguous_copy(self):
        """Materialize into a new NativeStorage in row-major order.

        Always copies, even if the view is already contiguous — the
        result is an independent storage the caller owns (and must
        close).
        """
        return NativeStorage.from_array(self.to_numpy().ravel())

    def __repr__(self):
        return (
            f"NativeTensorView(shape={self._shape}, strides={self._strides}, "
            f"offset={self._offset}, contiguous={self._contiguous})"
        )


class NativeTensorCore:
    """The first native tensor runtime object: an owned NativeStorage
    plus a NativeTensorView describing its layout, composed into one
    thing you can create, inspect, materialize, and release.

    Still not tensorforge.Tensor: no math operations, no autograd, no
    backend dispatch. It is the foundation those would build on.

    Ownership model: a core created by from_array/zeros/full OWNS its
    storage — its ``close()`` releases the native memory. View
    operations (reshape, transpose, T, narrow) return cores that
    BORROW the same storage: closing a borrowing view closes only that
    view, never the shared memory, so sibling views stay usable.
    Closing the owner releases the memory for every core sharing it,
    after which their data operations raise too. close() is
    idempotent everywhere; metadata properties stay readable after
    close, matching NativeStorage.size.
    """

    def __init__(self, storage, view, owns_storage=True):
        """Advanced constructor; prefer from_array / zeros / full.

        ``view`` must describe ``storage``. With ``owns_storage=True``
        this core's close() releases the storage; view operations pass
        False so shared storage is never freed by a view.
        """
        if not isinstance(storage, NativeStorage):
            raise TypeError(
                f"storage must be a NativeStorage, got {type(storage).__name__}"
            )
        if not isinstance(view, NativeTensorView) or view._storage is not storage:
            raise ValueError("view must be a NativeTensorView over the given storage")
        self._storage = storage
        self._view = view
        self._owns_storage = bool(owns_storage)
        self._closed = False

    # -- constructors -------------------------------------------------

    @classmethod
    def from_array(cls, values, dtype=None, device="cpu"):
        """A contiguous tensor holding a copy of ``values``, with the
        array's shape preserved. ``dtype``/``device`` record metadata and
        default to ``"float64"``/``"cpu"`` (the values are still coerced to
        float64); unsupported values are rejected."""
        array = np.ascontiguousarray(values, dtype=np.float64)
        # empty input fails here; dtype/device validated in the storage
        storage = NativeStorage.from_array(array, dtype=dtype, device=device)
        return cls(storage, NativeTensorView(storage, array.shape))

    @classmethod
    def zeros(cls, shape, dtype="float64", device="cpu"):
        """A row-major contiguous tensor of ``shape``, all zeros
        (native storage is zero-initialized, so no fill pass runs).
        ``dtype``/``device`` default to ``"float64"``/``"cpu"``;
        unsupported values are rejected."""
        count = numel(shape)  # validates shape by the v0.7 rules
        storage = NativeStorage(count, dtype=dtype, device=device)
        return cls(storage, NativeTensorView(storage, shape))

    @classmethod
    def full(cls, shape, value, dtype="float64", device="cpu"):
        """A row-major contiguous tensor of ``shape`` filled with
        ``value`` (anything float() accepts). ``dtype``/``device`` default
        to ``"float64"``/``"cpu"``; unsupported values are rejected."""
        tensor = cls.zeros(shape, dtype=dtype, device=device)
        tensor._storage.fill(float(value))
        return tensor

    # -- metadata (readable even after close) --------------------------

    @property
    def shape(self):
        return self._view.shape

    @property
    def strides(self):
        return self._view.strides

    @property
    def offset(self):
        return self._view.offset

    @property
    def ndim(self):
        return self._view.ndim

    @property
    def numel(self):
        return self._view.numel

    @property
    def contiguous(self):
        return self._view.contiguous

    @property
    def dtype(self):
        """The element type tag, delegated to this core's storage
        (``"float64"``). A view shares its storage, so it reports the same
        dtype as its owner. Readable after close, like ``shape``."""
        return self._storage.dtype

    @property
    def device(self):
        """The device tag, delegated to this core's storage (``"cpu"``). A
        view shares its storage, so it reports the same device as its
        owner. Readable after close, like ``shape``."""
        return self._storage.device

    @property
    def storage(self):
        """The owned NativeStorage (read-only access to the object)."""
        return self._storage

    @property
    def view(self):
        """The NativeTensorView describing this tensor's layout."""
        return self._view

    # -- data operations ------------------------------------------------

    def _require_open(self):
        if self._closed:
            raise RuntimeError("this NativeTensorCore has been closed")

    def to_numpy(self):
        """Materialize into a fresh, independent NumPy array of
        ``self.shape``."""
        self._require_open()
        return self._view.to_numpy()

    def contiguous_copy(self):
        """A new, independent NativeTensorCore with the same values in
        row-major contiguous storage. Always copies, even when this
        tensor is already contiguous."""
        self._require_open()
        return NativeTensorCore.from_array(self.to_numpy())

    # -- native compute (arithmetic happens in C++ over storage) ---------

    def _layout_arrays(self):
        return (
            np.asarray(self.shape, dtype=np.int64),
            np.asarray(self.strides, dtype=np.int64),
        )

    def relu(self):
        """max(x, 0) elementwise, computed by the native kernel reading
        this tensor's (possibly strided) view directly. Returns a new
        row-major contiguous NativeTensorCore.

        A contiguous input takes the flat fast-path kernel (a plain
        pointer loop); a strided view takes the generic odometer kernel.
        Both produce bit-for-bit identical results — the fast path is
        purely a traversal choice."""
        self._require_open()
        out = NativeTensorCore.zeros(self.shape, dtype=self.dtype, device=self.device)
        if self.contiguous:
            self._storage._lib.tf_core_relu_contiguous(
                self._storage._require_open(),
                out._storage._require_open(),
                self.numel, self.offset,
            )
            return out
        shape_arr, strides_arr = self._layout_arrays()
        self._storage._lib.tf_core_relu(
            self._storage._require_open(),
            out._storage._require_open(),
            shape_arr, strides_arr, self.offset, self.ndim,
        )
        return out

    def _unary_compute(self, odometer_name, contiguous_name):
        """Shared plumbing for the unary compute ops (v3.11): require
        open, allocate the fresh contiguous output, then dispatch to
        the contiguous fast-path kernel or the generic odometer kernel
        by this tensor's contiguity — exactly relu's strategy, and the
        two paths are bit-for-bit identical."""
        self._require_open()
        out = NativeTensorCore.zeros(self.shape, dtype=self.dtype, device=self.device)
        if self.contiguous:
            getattr(self._storage._lib, contiguous_name)(
                self._storage._require_open(),
                out._storage._require_open(),
                self.numel, self.offset,
            )
            return out
        shape_arr, strides_arr = self._layout_arrays()
        getattr(self._storage._lib, odometer_name)(
            self._storage._require_open(),
            out._storage._require_open(),
            shape_arr, strides_arr, self.offset, self.ndim,
        )
        return out

    def sqrt(self):
        """Elementwise square root, computed by the native kernel
        reading this tensor's (possibly strided) view directly. Returns
        a new row-major contiguous NativeTensorCore. IEEE float64
        semantics: negative inputs give NaN (no exception), signed
        zeros are preserved, +inf gives +inf, NaN propagates."""
        return self._unary_compute("tf_core_sqrt", "tf_core_sqrt_contiguous")

    def reciprocal(self):
        """Elementwise 1/x, computed by the native kernel reading this
        tensor's (possibly strided) view directly. Returns a new
        row-major contiguous NativeTensorCore. IEEE float64 semantics:
        ±0.0 gives ±inf (no exception, no warning), ±inf gives ±0.0,
        NaN propagates — the same values NumPy produces."""
        return self._unary_compute("tf_core_reciprocal",
                                   "tf_core_reciprocal_contiguous")

    def exp(self):
        """Elementwise e**x, computed by the native kernel reading this
        tensor's (possibly strided) view directly (Phase E, milestone
        E1). Returns a new **owning** row-major contiguous
        NativeTensorCore of the same shape; the input is not mutated and
        shares no storage with the result.

        Plain IEEE float64 ``std::exp`` — no clamping and no inserted
        bound: ``exp(0) == 1``, a large positive argument overflows to
        ``+inf``, a large negative one underflows toward ``+0``, ``+inf``
        gives ``+inf``, ``-inf`` gives ``+0``, and NaN propagates (the
        values NumPy produces, without its overflow warning).

        Graph-unaware, like every Core op: the differentiable surface is
        ``NativeTensor.exp()``."""
        return self._unary_compute("tf_core_exp", "tf_core_exp_contiguous")

    def relu_backward(self, upstream):
        """The gradient of ``relu`` at this tensor's forward value:
        ``upstream`` where this tensor's element is ``> 0``, else ``0``
        (``x == 0`` blocks the gradient, the Python Tensor convention).

        A forward-shaped numerical kernel, not graph machinery — the core
        stays autograd-unaware; the NativeTensor layer calls this from its
        relu backward closure. Both operands may be strided views (each is
        read through its own strides/offset); shapes must match exactly
        (no broadcasting — the upstream gradient of an op always has the
        op's output shape). Returns a new row-major contiguous
        NativeTensorCore of this tensor's shape."""
        self._require_open()
        if not isinstance(upstream, NativeTensorCore):
            raise TypeError(
                f"relu_backward requires a NativeTensorCore upstream "
                f"gradient, got {type(upstream).__name__}"
            )
        upstream._require_open()
        self._require_matching_metadata(upstream, "relu_backward")
        if self.shape != upstream.shape:
            raise ValueError(
                f"relu_backward requires the upstream gradient shape to "
                f"match the input shape, got {upstream.shape} and {self.shape}"
            )
        out = NativeTensorCore.zeros(self.shape, dtype=self.dtype, device=self.device)
        shape_arr, x_strides = self._layout_arrays()
        u_strides = np.asarray(upstream.strides, dtype=np.int64)
        self._storage._lib.tf_core_relu_backward(
            self._storage._require_open(),
            upstream._storage._require_open(),
            out._storage._require_open(),
            shape_arr, x_strides, u_strides,
            self.offset, upstream.offset, self.ndim,
        )
        return out

    def _require_matching_metadata(self, other, op_name):
        """Both operands of a binary/matmul op must share dtype and
        device; there is no implicit promotion and no automatic device
        move (see docs/native_dtype_device_metadata_design.md §8). Raises
        ValueError naming both dtype/device pairs on a mismatch. With only
        float64/cpu constructible today this guard cannot yet fire, but it
        is the enforced contract native autograd (Phase B) builds on."""
        if self.dtype != other.dtype or self.device != other.device:
            raise ValueError(
                f"{op_name} requires matching dtype and device, got "
                f"{self.dtype}/{self.device} and {other.dtype}/{other.device}"
            )

    def _binary_core_op(self, other, kernel_name, op_name):
        """Shared plumbing for add/subtract/multiply over tensor cores,
        with a three-way traversal dispatch (v1.17):

        A. **Same shape, both contiguous** — the v1.14 flat fast-path
           kernel (``<kernel_name>_contiguous``): a plain pointer loop
           over numel.
        B. **Same shape, either strided** — the generic odometer kernel
           (``<kernel_name>``): walks the shared shape with each
           operand's real strides.
        C. **Differing but broadcast-compatible shapes** — NumPy-style
           broadcasting. The output shape is inferred by
           ``broadcast_shapes`` and each operand is read through
           *broadcast strides* (§ native_broadcasting_design.md): a real
           axis keeps its stride, a stretched or left-padded size-1 axis
           gets stride 0 so the odometer re-reads one element instead of
           advancing. The very same generic odometer kernel from path B
           consumes those strides — a zero stride is broadcasting, no
           expanded operand is materialized. Incompatible shapes raise a
           clear ``ValueError`` naming both shapes.

        Paths A and B are bit-for-bit unchanged from before; only when the
        shapes actually differ does broadcasting engage. The output is
        always freshly allocated row-major contiguous storage."""
        self._require_open()
        if not isinstance(other, NativeTensorCore):
            raise TypeError(
                f"{op_name} requires a NativeTensorCore operand, "
                f"got {type(other).__name__}"
            )
        other._require_open()
        self._require_matching_metadata(other, op_name)
        lib = self._storage._lib

        # Same-shape paths (A and B) — the exact-shape behavior, unchanged.
        if self.shape == other.shape:
            out = NativeTensorCore.zeros(self.shape, dtype=self.dtype, device=self.device)
            if self.contiguous and other.contiguous:
                getattr(lib, kernel_name + "_contiguous")(
                    self._storage._require_open(),
                    other._storage._require_open(),
                    out._storage._require_open(),
                    self.numel, self.offset, other.offset,
                )
                return out
            shape_arr, a_strides = self._layout_arrays()
            b_strides = np.asarray(other.strides, dtype=np.int64)
            getattr(lib, kernel_name)(
                self._storage._require_open(),
                other._storage._require_open(),
                out._storage._require_open(),
                shape_arr, a_strides, b_strides,
                self.offset, other.offset, self.ndim,
            )
            return out

        # Broadcasting path (C) — differing shapes. broadcast_shapes
        # raises (naming both shapes) if they are incompatible, before
        # any output is allocated.
        out_shape = broadcast_shapes(self.shape, other.shape)
        out = NativeTensorCore.zeros(out_shape, dtype=self.dtype, device=self.device)
        out_ndim = len(out_shape)
        shape_arr = np.asarray(out_shape, dtype=np.int64)
        a_strides = np.asarray(
            _broadcast_strides(self.shape, self.strides, out_shape),
            dtype=np.int64,
        )
        b_strides = np.asarray(
            _broadcast_strides(other.shape, other.strides, out_shape),
            dtype=np.int64,
        )
        getattr(lib, kernel_name)(
            self._storage._require_open(),
            other._storage._require_open(),
            out._storage._require_open(),
            shape_arr, a_strides, b_strides,
            self.offset, other.offset, out_ndim,
        )
        return out

    def add(self, other):
        """self + other elementwise, natively. Identical shapes, or
        NumPy-style broadcasting for compatible shapes (v1.17)."""
        return self._binary_core_op(other, "tf_core_add", "add")

    def subtract(self, other):
        """self - other elementwise, natively. Identical shapes, or
        NumPy-style broadcasting for compatible shapes (v1.17)."""
        return self._binary_core_op(other, "tf_core_subtract", "subtract")

    def multiply(self, other):
        """self * other elementwise, natively. Identical shapes, or
        NumPy-style broadcasting for compatible shapes (v1.17)."""
        return self._binary_core_op(other, "tf_core_multiply", "multiply")

    def matmul(self, other):
        """(m, n) @ (n, p) matrix multiplication over native storage.

        Both operands must be 2-D tensor cores; either may be a
        non-contiguous view (transposed, narrowed) — the kernel
        addresses each source through its own strides, so nothing is
        materialized first. Returns a new (m, p) row-major contiguous
        NativeTensorCore. The naive triple loop; no broadcasting.
        """
        self._require_open()
        if not isinstance(other, NativeTensorCore):
            raise TypeError(
                f"matmul requires a NativeTensorCore operand, "
                f"got {type(other).__name__}"
            )
        other._require_open()
        self._require_matching_metadata(other, "matmul")
        if self.ndim != 2:
            raise ValueError(
                f"matmul requires a 2-D left operand, got shape {self.shape}"
            )
        if other.ndim != 2:
            raise ValueError(
                f"matmul requires a 2-D right operand, got shape {other.shape}"
            )
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"inner dimensions do not match: "
                f"{self.shape} @ {other.shape} (need (m, n) @ (n, p))"
            )
        m, n = self.shape
        p = other.shape[1]
        out = NativeTensorCore.zeros((m, p), dtype=self.dtype, device=self.device)
        self._storage._lib.tf_core_matmul(
            self._storage._require_open(),
            other._storage._require_open(),
            out._storage._require_open(),
            m, n, p,
            self.strides[0], self.strides[1],
            other.strides[0], other.strides[1],
            self.offset, other.offset,
        )
        return out

    # -- reductions (v1.19) ---------------------------------------------

    def sum(self, axis=None, keepdims=False):
        """Sum over ``axis`` (``None`` = all elements) natively, reading
        this tensor's (possibly strided) view directly. Returns a new
        owning row-major contiguous NativeTensorCore whose shape is
        ``reduce_shape(self.shape, axis, keepdims)``. Single integer or
        negative axis only; no broadcasting of the result, no autograd.

        Deterministic row-major accumulation order over the input; float
        sums are order-sensitive, so results match NumPy to a tolerance,
        not bit-for-bit (see docs/native_reductions_design.md)."""
        self._require_open()
        out_shape = reduce_shape(self.shape, axis, keepdims)  # validates axis/keepdims
        out = NativeTensorCore.zeros(out_shape, dtype=self.dtype, device=self.device)
        if axis is None:
            reduced = set(range(self.ndim))
        else:
            reduced = {_normalize_axis(axis, self.shape)}
        out_strides = _reduce_out_strides(self.shape, reduced, bool(keepdims), out_shape)
        self._storage._lib.tf_core_sum(
            self._storage._require_open(),
            out._storage._require_open(),
            np.asarray(self.shape, dtype=np.int64),
            np.asarray(self.strides, dtype=np.int64),
            np.asarray(out_strides, dtype=np.int64),
            self.offset, self.ndim,
        )
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean over ``axis`` (``None`` = all elements) natively: the
        native ``sum`` scaled in place by ``1/count`` in float64, where
        ``count`` is ``numel`` for ``axis=None`` or ``shape[axis]`` for a
        single axis. Returns a new owning row-major contiguous
        NativeTensorCore. No NumPy touches the data; no autograd."""
        self._require_open()
        result = self.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            count = self.numel
        else:
            count = self.shape[_normalize_axis(axis, self.shape)]
        # In-place native scale of the freshly summed output — no copy,
        # no NumPy round trip. count >= 1 always (dims are positive).
        result._storage._lib.tf_storage_scale(
            result._storage._require_open(), 1.0 / count
        )
        return result

    # -- convolution (Phase D, D3: forward-only Core wrapper) ------------

    def _contiguous_temp(self, temporaries):
        """Materialize this core into a fresh **owning** row-major
        contiguous copy (offset 0) and record it in ``temporaries`` so the
        caller closes it deterministically after the native call — the
        Policy-B copy-then-compute helper (docs/native_cnn_design.md §5)."""
        temp = self.contiguous_copy()
        temporaries.append(temp)
        return temp

    def conv2d_forward(self, weight, bias=None, *, stride=1, padding=0):
        """2-D cross-correlation forward over this NCHW input, natively.

        ``self`` is the ``(N, C, H, W)`` input; ``weight`` is an
        ``(O, C, kh, kw)`` OIHW tensor core; ``bias`` is an optional
        ``(O,)`` tensor core (``None`` = no bias). ``stride`` and
        ``padding`` are each an int or a 2-element ``(height, width)`` pair
        (bools rejected). Returns a fresh **owning** row-major contiguous
        ``(N, O, out_h, out_w)`` NativeTensorCore.

        This is the **forward-only, autograd-unaware** Core wrapper (the
        differentiable ``NativeTensor.conv2d`` primitive is a later
        milestone). It performs the full public validation, computes and
        checks the output shape from the locked floor formula in Python ints
        (so the shape math cannot overflow) *before* allocating anything,
        and — by Policy B (docs/native_cnn_design.md §5) — feeds the raw C
        ABI **contiguous storage only**: any non-contiguous input, weight,
        or bias is materialized into a private contiguous copy that is
        closed the moment the native call returns, while already-contiguous
        operands (even with a non-zero offset) are passed through untouched.
        The caller's tensors are never mutated. A failure at any stage
        allocates no output and leaks no temporary copy.

        No dilation, groups, channels-last, or output padding — those are
        not part of the signature. The weight/bias/input must all be open
        CPU float64 tensor cores."""
        self._require_open()
        if not isinstance(weight, NativeTensorCore):
            raise TypeError(
                f"conv2d_forward requires a NativeTensorCore weight, "
                f"got {type(weight).__name__}"
            )
        weight._require_open()
        self._require_matching_metadata(weight, "conv2d_forward")
        has_bias = bias is not None
        if has_bias:
            if not isinstance(bias, NativeTensorCore):
                raise TypeError(
                    f"conv2d_forward requires a NativeTensorCore bias or None, "
                    f"got {type(bias).__name__}"
                )
            bias._require_open()
            self._require_matching_metadata(bias, "conv2d_forward")

        if self.ndim != 4:
            raise ValueError(
                f"conv2d_forward requires a 4-D NCHW input, got shape {self.shape}"
            )
        if weight.ndim != 4:
            raise ValueError(
                f"conv2d_forward requires a 4-D OIHW weight, got shape {weight.shape}"
            )
        if has_bias and bias.ndim != 1:
            raise ValueError(
                f"conv2d_forward requires a 1-D bias, got shape {bias.shape}"
            )

        n, c, h, w = self.shape
        o, weight_in, kh, kw = weight.shape
        if c != weight_in:
            raise ValueError(
                f"conv2d_forward input channels {c} do not match the weight's "
                f"input channels {weight_in} (input {self.shape}, weight "
                f"{weight.shape})"
            )
        if has_bias and bias.shape[0] != o:
            raise ValueError(
                f"conv2d_forward bias length {bias.shape[0]} does not match the "
                f"number of output channels {o}"
            )

        sh, sw = _spatial_pair(stride, "stride", minimum=1)
        ph, pw = _spatial_pair(padding, "padding", minimum=0)
        # Python-int floor arithmetic; raises before any allocation if the
        # kernel does not fit the padded input.
        out_h, out_w = conv_output_shape((h, w), (kh, kw), (sh, sw), (ph, pw))

        # Policy B: hand the kernel contiguous storage only. A non-contiguous
        # operand is copied into a private owning tensor (offset 0) closed as
        # soon as the call returns; a contiguous operand (possibly with a
        # non-zero offset) is passed straight through with its offset.
        temporaries = []
        try:
            input_core = (
                self if self.contiguous else self._contiguous_temp(temporaries)
            )
            weight_core = (
                weight if weight.contiguous
                else weight._contiguous_temp(temporaries)
            )
            bias_handle = None
            bias_offset = 0
            if has_bias:
                bias_core = (
                    bias if bias.contiguous
                    else bias._contiguous_temp(temporaries)
                )
                bias_handle = bias_core._storage._require_open()
                bias_offset = bias_core.offset

            out = NativeTensorCore.zeros(
                (n, o, out_h, out_w), dtype=self.dtype, device=self.device
            )
            try:
                self._storage._lib.tf_core_conv2d_forward(
                    input_core._storage._require_open(), input_core.offset,
                    weight_core._storage._require_open(), weight_core.offset,
                    bias_handle, bias_offset,
                    out._storage._require_open(),
                    n, c, h, w, o, kh, kw, sh, sw, ph, pw, out_h, out_w,
                )
            except BaseException:
                # The native call failed (e.g. an injected allocation
                # failure): discard the freshly allocated output so a failed
                # forward returns no half-built tensor.
                out.close()
                raise
            return out
        finally:
            # Close every private contiguous copy exactly once, whether the
            # call succeeded or raised — the caller's operands are untouched.
            for temp in temporaries:
                temp.close()

    def conv2d_input_backward(self, weight, *, input_shape, stride=1, padding=0):
        """Gradient of Conv2d w.r.t. its input, natively (Phase D, D6).

        ``self`` is the upstream gradient ``grad_output`` with shape
        ``(N, O, out_h, out_w)``; ``weight`` is the ``(O, C, kh, kw)`` OIHW
        tensor core; ``input_shape`` is the parent input's ``(N, C, H, W)``.
        Returns a fresh **owning** row-major contiguous ``(N, C, H, W)``
        NativeTensorCore — the input gradient.

        Forward-only and autograd-unaware (the ``NativeTensor.conv2d``
        node calls this from its input-gradient callback). Validates ranks,
        channel/spatial relationships, and the recomputed grad_output shape
        before allocating; feeds the raw C ABI **contiguous storage only**
        via Policy B (any non-contiguous grad_output/weight is copied into a
        private core closed as soon as the call returns); the caller's
        tensors are never mutated. A failure allocates no output and leaks no
        temporary."""
        self._require_open()
        if not isinstance(weight, NativeTensorCore):
            raise TypeError(
                f"conv2d_input_backward requires a NativeTensorCore weight, "
                f"got {type(weight).__name__}"
            )
        weight._require_open()
        self._require_matching_metadata(weight, "conv2d_input_backward")
        if self.ndim != 4:
            raise ValueError(
                f"conv2d_input_backward requires a 4-D NCHW grad_output, got "
                f"shape {self.shape}"
            )
        if weight.ndim != 4:
            raise ValueError(
                f"conv2d_input_backward requires a 4-D OIHW weight, got shape "
                f"{weight.shape}"
            )
        input_shape = _as_shape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"conv2d_input_backward input_shape must be 4-D NCHW, got "
                f"{input_shape}"
            )
        n, c, h, w = input_shape
        o, weight_in, kh, kw = weight.shape
        if c != weight_in:
            raise ValueError(
                f"conv2d_input_backward input channels {c} do not match the "
                f"weight's input channels {weight_in} (input_shape "
                f"{input_shape}, weight {weight.shape})"
            )
        sh, sw = _spatial_pair(stride, "stride", minimum=1)
        ph, pw = _spatial_pair(padding, "padding", minimum=0)
        out_h, out_w = conv_output_shape((h, w), (kh, kw), (sh, sw), (ph, pw))
        if self.shape != (n, o, out_h, out_w):
            raise ValueError(
                f"conv2d_input_backward grad_output shape {self.shape} does "
                f"not match the expected {(n, o, out_h, out_w)} for input "
                f"{input_shape}, weight {weight.shape}, stride {(sh, sw)}, "
                f"padding {(ph, pw)}"
            )
        temporaries = []
        try:
            go = self if self.contiguous else self._contiguous_temp(temporaries)
            wt = (
                weight if weight.contiguous
                else weight._contiguous_temp(temporaries)
            )
            out = NativeTensorCore.zeros(
                (n, c, h, w), dtype=self.dtype, device=self.device
            )
            try:
                self._storage._lib.tf_core_conv2d_input_backward(
                    go._storage._require_open(), go.offset,
                    wt._storage._require_open(), wt.offset,
                    out._storage._require_open(),
                    n, c, h, w, o, kh, kw, sh, sw, ph, pw, out_h, out_w,
                )
            except BaseException:
                out.close()
                raise
            return out
        finally:
            for temp in temporaries:
                temp.close()

    def conv2d_weight_backward(self, input, *, weight_shape, stride=1, padding=0):
        """Gradient of Conv2d w.r.t. its weight, natively (Phase D, D6).

        ``self`` is the upstream gradient ``grad_output`` with shape
        ``(N, O, out_h, out_w)``; ``input`` is the parent's ``(N, C, H, W)``
        NCHW input; ``weight_shape`` is the weight's ``(O, C, kh, kw)`` OIHW
        shape. Returns a fresh **owning** row-major contiguous
        ``(O, C, kh, kw)`` NativeTensorCore — the weight gradient.

        Forward-only and autograd-unaware. Same validation/Policy-B/failure
        contract as ``conv2d_input_backward`` (any non-contiguous
        grad_output/input is copied into a private core closed after the
        call); the caller's tensors are never mutated."""
        self._require_open()
        if not isinstance(input, NativeTensorCore):
            raise TypeError(
                f"conv2d_weight_backward requires a NativeTensorCore input, "
                f"got {type(input).__name__}"
            )
        input._require_open()
        self._require_matching_metadata(input, "conv2d_weight_backward")
        if self.ndim != 4:
            raise ValueError(
                f"conv2d_weight_backward requires a 4-D NCHW grad_output, got "
                f"shape {self.shape}"
            )
        if input.ndim != 4:
            raise ValueError(
                f"conv2d_weight_backward requires a 4-D NCHW input, got shape "
                f"{input.shape}"
            )
        weight_shape = _as_shape(weight_shape)
        if len(weight_shape) != 4:
            raise ValueError(
                f"conv2d_weight_backward weight_shape must be 4-D OIHW, got "
                f"{weight_shape}"
            )
        o, c, kh, kw = weight_shape
        n, input_c, h, w = input.shape
        if input_c != c:
            raise ValueError(
                f"conv2d_weight_backward input channels {input_c} do not match "
                f"the weight's input channels {c} (input {input.shape}, "
                f"weight_shape {weight_shape})"
            )
        sh, sw = _spatial_pair(stride, "stride", minimum=1)
        ph, pw = _spatial_pair(padding, "padding", minimum=0)
        out_h, out_w = conv_output_shape((h, w), (kh, kw), (sh, sw), (ph, pw))
        if self.shape != (n, o, out_h, out_w):
            raise ValueError(
                f"conv2d_weight_backward grad_output shape {self.shape} does "
                f"not match the expected {(n, o, out_h, out_w)} for input "
                f"{input.shape}, weight_shape {weight_shape}, stride "
                f"{(sh, sw)}, padding {(ph, pw)}"
            )
        temporaries = []
        try:
            go = self if self.contiguous else self._contiguous_temp(temporaries)
            inp = (
                input if input.contiguous
                else input._contiguous_temp(temporaries)
            )
            out = NativeTensorCore.zeros(
                (o, c, kh, kw), dtype=self.dtype, device=self.device
            )
            try:
                self._storage._lib.tf_core_conv2d_weight_backward(
                    go._storage._require_open(), go.offset,
                    inp._storage._require_open(), inp.offset,
                    out._storage._require_open(),
                    n, c, h, w, o, kh, kw, sh, sw, ph, pw, out_h, out_w,
                )
            except BaseException:
                out.close()
                raise
            return out
        finally:
            for temp in temporaries:
                temp.close()

    # -- pooling (Phase D, D8: forward-only Core wrapper + winners) -------

    def maxpool2d_forward(self, *, kernel_size, stride=None, padding=0):
        """2-D max pooling forward over this NCHW input, natively.

        ``self`` is the ``(N, C, H, W)`` input. ``kernel_size`` and
        ``stride`` are an int or a 2-element ``(height, width)`` pair of
        ints ≥ 1 (bools rejected); ``stride=None`` means
        ``stride = kernel_size`` (non-overlapping windows, the stable
        convention). ``padding`` is an int or pair ≥ 0, applied
        symmetrically on each spatial axis. Returns a fresh **owning**
        row-major contiguous ``(N, C, out_h, out_w)`` NativeTensorCore.

        Windows see a conceptual ``-inf`` outside the real input, so a
        padded cell loses to any finite value but still *participates* in
        the selection; ties keep the first occurrence in row-major window
        order (docs/native_cnn_design.md §10). This is the **forward-only,
        autograd-unaware** Core wrapper — the differentiable
        ``NativeTensor.maxpool2d`` primitive (D9) and the
        ``NativeMaxPool2d`` module (D10) are separate layers built on it,
        and both are implemented.

        The kernel also produces the private winner buffer backward will
        need; this public method releases it, so the pooled values are all
        that survive. The internal
        ``_maxpool2d_forward_with_winners`` helper is what keeps it (D9).

        No dilation, ceil_mode, return_indices, adaptive/average/global
        pooling, or channels-last — none of those is in the signature."""
        out, winners = self._maxpool2d_forward_with_winners(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        # The public Core forward exposes only the pooled values; the
        # winner buffer is internal state, released deterministically here
        # rather than left to garbage collection.
        winners.close()
        return out

    def _maxpool2d_forward_with_winners(
        self, *, kernel_size, stride=None, padding=0
    ):
        """The pooling forward plus its private saved-winner buffer.

        Returns ``(output, winners)``: two fresh **owning** row-major
        contiguous ``(N, C, out_h, out_w)`` cores. ``winners`` holds, for
        each output cell, the flat offset ``ih * W + iw`` of the selected
        input element inside its ``(n, c)`` plane, or the sentinel ``-1.0``
        when a padding cell won (docs/native_cnn_design.md §12). Every
        stored value is an exact integral float64 — the wrapper proves
        ``H * W <= 2**53`` in Python arbitrary-precision arithmetic
        *before* allocating or calling anything, so no index can round.

        The winner buffer is **internal**: it is never exposed as a public
        NativeTensor, never given a dtype tag of its own, never traversed
        as a parameter or buffer, and never serialized. It exists so the
        D9 backward can scatter without recomputing winners; the caller of
        this private helper owns it and must ``close()`` it.

        Validation runs entirely before any allocation. Per **Policy B**
        (docs/native_cnn_design.md §5) a non-contiguous input is
        materialized into a private owning contiguous copy that is closed
        as soon as the native call returns, while an already-contiguous
        input (even at a non-zero offset) is passed straight through. On
        any failure every object this method allocated is closed, the
        caller's input is untouched, and no partial result is returned."""
        self._require_open()
        if self.ndim != 4:
            raise ValueError(
                f"maxpool2d_forward requires a 4-D NCHW input, got shape "
                f"{self.shape}"
            )
        if self.dtype != "float64" or self.device != "cpu":
            raise ValueError(
                f"maxpool2d_forward requires a float64/cpu input, got "
                f"{self.dtype}/{self.device}"
            )

        kh, kw = _spatial_pair(kernel_size, "kernel_size", minimum=1)
        # Stable convention: no stride means non-overlapping windows.
        if stride is None:
            sh, sw = kh, kw
        else:
            sh, sw = _spatial_pair(stride, "stride", minimum=1)
        ph, pw = _spatial_pair(padding, "padding", minimum=0)

        n, c, h, w = self.shape
        # Winner indices are float64 flat plane offsets, exact only while
        # H*W <= 2**53 (design §12). Python ints, so the product itself
        # cannot overflow while being checked.
        if h * w > _MAX_EXACT_WINNER_PLANE:
            raise ValueError(
                f"maxpool2d_forward input plane {(h, w)} has {h * w} elements, "
                f"more than the {_MAX_EXACT_WINNER_PLANE} float64 can index "
                f"exactly; winner offsets would round"
            )
        # Python-int floor arithmetic; raises before any allocation if the
        # window does not fit the padded input.
        out_h, out_w = conv_output_shape((h, w), (kh, kw), (sh, sw), (ph, pw))
        # The element counts crossing the ABI must be representable as
        # int64 storage sizes (again in Python ints, before allocating).
        for count, what in (
            (n * c * h * w, "input"),
            (n * c * out_h * out_w, "output"),
        ):
            if count > _INT64_MAX:
                raise ValueError(
                    f"maxpool2d_forward {what} element count {count} exceeds "
                    f"the int64 range the native runtime addresses"
                )

        temporaries = []
        out = None
        winners = None
        try:
            input_core = (
                self if self.contiguous else self._contiguous_temp(temporaries)
            )
            # Deterministic allocation order: output first, then winners.
            # If the second allocation fails the first is closed below, so
            # a failed forward leaves nothing half-built.
            out = NativeTensorCore.zeros(
                (n, c, out_h, out_w), dtype=self.dtype, device=self.device
            )
            winners = NativeTensorCore.zeros(
                (n, c, out_h, out_w), dtype=self.dtype, device=self.device
            )
            self._storage._lib.tf_core_maxpool2d_forward(
                input_core._storage._require_open(), input_core.offset,
                out._storage._require_open(),
                winners._storage._require_open(),
                n, c, h, w, kh, kw, sh, sw, ph, pw, out_h, out_w,
            )
            return out, winners
        except BaseException:
            # Close whichever result objects were successfully allocated —
            # never rely on garbage collection for native memory.
            for allocated in (winners, out):
                if allocated is not None:
                    allocated.close()
            raise
        finally:
            # Close the private contiguous copy (if any) exactly once,
            # whether the call succeeded or raised.
            for temp in temporaries:
                temp.close()

    def maxpool2d_backward(self, winners, *, input_shape):
        """Gradient of MaxPool2d w.r.t. its input, natively (Phase D, D9).

        ``self`` is the upstream gradient ``grad_output`` with shape
        ``(N, C, out_h, out_w)``; ``winners`` is the **private saved-winner
        core** the D8 forward produced, with exactly that shape;
        ``input_shape`` is the parent input's ``(N, C, H, W)``. Returns a
        fresh **owning** row-major contiguous ``(N, C, H, W)``
        NativeTensorCore — the input gradient.

        The routing comes entirely from the saved winners: each output
        cell's gradient is added to the input element that won its window
        (``ih = winner // W``, ``iw = winner % W``), and a ``-1`` winner
        (padding won) drops that gradient. Overlapping windows accumulate.
        **No input value is reread and no window maximum is recomputed**,
        so no kernel/stride/padding argument is needed or accepted here.

        Autograd-unaware (the ``NativeTensor.maxpool2d`` node calls this
        from its single input-gradient callback). Validation runs before
        any allocation; per Policy B a non-contiguous grad_output or winner
        core is materialized into a private copy closed as soon as the
        native call returns. The checked C ABI additionally validates every
        winner value (``-1`` or an exact in-range integer) before
        scattering, so a corrupted buffer raises instead of writing
        anywhere. Neither operand is mutated, and a failure closes the
        output and leaks no temporary."""
        self._require_open()
        if not isinstance(winners, NativeTensorCore):
            raise TypeError(
                f"maxpool2d_backward requires a NativeTensorCore winner "
                f"buffer, got {type(winners).__name__}"
            )
        winners._require_open()
        self._require_matching_metadata(winners, "maxpool2d_backward")
        if self.dtype != "float64" or self.device != "cpu":
            raise ValueError(
                f"maxpool2d_backward requires float64/cpu operands, got "
                f"{self.dtype}/{self.device}"
            )
        if self.ndim != 4:
            raise ValueError(
                f"maxpool2d_backward requires a 4-D NCHW grad_output, got "
                f"shape {self.shape}"
            )
        if winners.shape != self.shape:
            raise ValueError(
                f"maxpool2d_backward winner shape {winners.shape} does not "
                f"match the grad_output shape {self.shape}"
            )
        input_shape = _as_shape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"maxpool2d_backward input_shape must be 4-D NCHW, got "
                f"{input_shape}"
            )
        n, c, h, w = input_shape
        out_n, out_c, out_h, out_w = self.shape
        if (n, c) != (out_n, out_c):
            raise ValueError(
                f"maxpool2d_backward batch/channels {(n, c)} do not match the "
                f"grad_output's {(out_n, out_c)} (input_shape {input_shape}, "
                f"grad_output {self.shape})"
            )
        # The winner domain is [0, H*W - 1] plus the -1 sentinel, so the
        # same float64 exactness bound the forward proved must hold here.
        if h * w > _MAX_EXACT_WINNER_PLANE:
            raise ValueError(
                f"maxpool2d_backward input plane {(h, w)} has {h * w} "
                f"elements, more than the {_MAX_EXACT_WINNER_PLANE} float64 "
                f"can index exactly"
            )
        if n * c * h * w > _INT64_MAX:
            raise ValueError(
                f"maxpool2d_backward grad_input element count {n * c * h * w} "
                f"exceeds the int64 range the native runtime addresses"
            )

        temporaries = []
        out = None
        try:
            grad_output = (
                self if self.contiguous else self._contiguous_temp(temporaries)
            )
            winner_core = (
                winners if winners.contiguous
                else winners._contiguous_temp(temporaries)
            )
            out = NativeTensorCore.zeros(
                (n, c, h, w), dtype=self.dtype, device=self.device
            )
            self._storage._lib.tf_core_maxpool2d_backward(
                grad_output._storage._require_open(), grad_output.offset,
                winner_core._storage._require_open(), winner_core.offset,
                out._storage._require_open(),
                n, c, h, w, out_h, out_w,
            )
            return out
        except BaseException:
            # A failed backward returns no half-built gradient.
            if out is not None:
                out.close()
            raise
        finally:
            for temp in temporaries:
                temp.close()

    # -- view operations (metadata only: no data is copied) --------------

    def _view_core(self, shape, strides, offset):
        """A new core borrowing this core's storage with new layout."""
        view = NativeTensorView(self._storage, shape, strides=strides, offset=offset)
        return NativeTensorCore(self._storage, view, owns_storage=False)

    def reshape(self, new_shape):
        """A view of the same storage with ``new_shape`` (row-major).

        Metadata only — no copy. Requires a contiguous tensor (a
        non-contiguous layout cannot be reinterpreted by strides
        alone; materialize with contiguous_copy() first) and the same
        total number of elements.
        """
        self._require_open()
        if not self.contiguous:
            raise ValueError(
                "reshape requires a contiguous tensor; call "
                "contiguous_copy() first"
            )
        count = numel(new_shape)  # validates the shape by the v0.7 rules
        if count != self.numel:
            raise ValueError(
                f"cannot reshape {self.shape} ({self.numel} elements) "
                f"into {tuple(new_shape)} ({count} elements)"
            )
        return self._view_core(new_shape, None, self.offset)

    def transpose(self, *axes):
        """A view with permuted axes. Metadata only — no copy.

        With no arguments, all axes are reversed (NumPy behavior; a
        no-op for scalars and 1-D tensors). Explicit axes must be a
        complete permutation of range(ndim).
        """
        self._require_open()
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        axes = _as_int_tuple(axes, "axes")
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError(
                f"axes must be a permutation of range({self.ndim}), got {axes}"
            )
        new_shape = tuple(self.shape[axis] for axis in axes)
        new_strides = tuple(self.strides[axis] for axis in axes)
        return self._view_core(new_shape, new_strides, self.offset)

    @property
    def T(self):
        """transpose() with all axes reversed — NumPy's .T semantics,
        so (1, 0) for 2-D and a no-op for scalars and 1-D tensors."""
        return self.transpose()

    def narrow(self, dim, start, length):
        """A view keeping ``length`` positions of dimension ``dim``,
        beginning at ``start``. Metadata only — no copy: the shape
        shrinks in one dimension and the offset advances by
        ``start * strides[dim]``; strides are unchanged.

        ``length`` must be at least 1 (zero-size shapes are not
        supported). No step parameter in v1.1.
        """
        self._require_open()
        for name, value in (("dim", dim), ("start", start), ("length", length)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {value!r}")
        if not 0 <= dim < self.ndim:
            raise ValueError(f"dim must be in [0, {self.ndim}), got {dim}")
        if start < 0 or length < 1 or start + length > self.shape[dim]:
            raise ValueError(
                f"narrow(dim={dim}, start={start}, length={length}) is out "
                f"of bounds for dimension size {self.shape[dim]}"
            )
        new_shape = tuple(
            length if axis == dim else size for axis, size in enumerate(self.shape)
        )
        new_offset = self.offset + start * self.strides[dim]
        return self._view_core(new_shape, self.strides, new_offset)

    def narrow_backward(self, dim, start, original_shape):
        """Scatter this upstream gradient into a fresh zero tensor of
        ``original_shape``, placing each element at its narrowed logical
        position — the adjoint of ``narrow(dim, start, length)``, where
        ``length`` is this gradient's own extent along ``dim``.

        A forward-shaped numerical method, not graph machinery — the core
        stays autograd-unaware; the NativeTensor layer calls this from its
        narrow backward closure (``self`` is the upstream gradient there,
        the data being scattered). This gradient may be a strided view (it
        is read through its own strides/offset); the result is a new
        **owning** row-major contiguous NativeTensorCore of
        ``original_shape``, zero everywhere outside the narrowed region and
        carrying this gradient's dtype/device. No NumPy touches the data.

        Validates the scatter arguments the way ``narrow`` validates its
        forward ones: ``dim`` a non-bool int in ``[0, ndim)``, ``start``
        non-negative, the gradient's rank equal to the original rank, its
        non-``dim`` extents equal to the original's, and
        ``start + length <= original_shape[dim]``."""
        self._require_open()
        original = _as_shape(original_shape)  # validates positive-int dims
        ndim = len(original)
        for name, value in (("dim", dim), ("start", start)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {value!r}")
        dim = int(dim)
        start = int(start)
        if not 0 <= dim < ndim:
            raise ValueError(f"dim must be in [0, {ndim}), got {dim}")
        if self.ndim != ndim:
            raise ValueError(
                f"narrow_backward gradient rank {self.ndim} does not match "
                f"the original rank {ndim} (shape {original})"
            )
        length = self.shape[dim]
        if start < 0 or start + length > original[dim]:
            raise ValueError(
                f"narrow_backward(dim={dim}, start={start}, length={length}) "
                f"is out of bounds for dimension size {original[dim]}"
            )
        for axis in range(ndim):
            if axis != dim and self.shape[axis] != original[axis]:
                raise ValueError(
                    f"narrow_backward gradient shape {self.shape} is not "
                    f"compatible with original shape {original} along axis "
                    f"{axis}"
                )
        out = NativeTensorCore.zeros(original, dtype=self.dtype, device=self.device)
        # The gradient lives at the logical shape, so the output is always a
        # fresh row-major contiguous buffer (offset 0) regardless of the
        # narrowed parent's own layout. Each narrowed axis maps 1:1 to the
        # same output axis, so the write-strides are just the parent's full
        # row-major strides; the base offset skips the leading `start` slabs.
        out_full = row_major_strides(original)
        out_offset = start * out_full[dim]
        self._storage._lib.tf_core_narrow_backward(
            self._storage._require_open(),
            out._storage._require_open(),
            np.asarray(self.shape, dtype=np.int64),
            np.asarray(self.strides, dtype=np.int64),
            np.asarray(out_full, dtype=np.int64),
            self.offset, out_offset, self.ndim,
        )
        return out

    # -- lifetime -------------------------------------------------------

    def close(self):
        """Close this core. Owners release the native storage; views
        only close themselves and leave shared storage untouched.
        Safe to call repeatedly."""
        self._closed = True
        if self._owns_storage:
            self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        # Defensive cleanup only — never rely on GC timing; use close().
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        state = ", closed" if self._closed else ""
        return (
            f"NativeTensorCore(shape={self.shape}, strides={self.strides}, "
            f"contiguous={self.contiguous}{state})"
        )


# ---------------------------------------------------------------------------
# Shape/stride metadata
#
# The Python-facing metadata layer that prepares the path for a later
# native tensor storage object: shape, strides, element counts,
# contiguity checks, and flat-offset math. Implemented in Python on
# purpose — the deliverable is the *contract* (what shapes/strides
# mean), and integer arithmetic gains nothing from a ctypes round
# trip. These helpers never touch the compiled library, so they are
# safe whether or not the backend is built.
#
# Conventions:
# - strides count ELEMENTS, not bytes (unlike numpy.ndarray.strides);
#   row_major_strides((2, 3, 4)) == (12, 4, 1).
# - dimensions must be positive ints; zero-size dimensions are
#   rejected in v0.7 (their stride conventions deserve their own
#   tested milestone).
# - the scalar shape () has strides (), ndim 0, and numel 1.
# ---------------------------------------------------------------------------


def _as_int_tuple(values, name):
    try:
        items = tuple(values)
    except TypeError:
        raise TypeError(
            f"{name} must be a sequence of ints, got {values!r}"
        ) from None
    for value in items:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(f"{name} must contain only ints, got {value!r}")
    return tuple(int(value) for value in items)


def _as_shape(shape):
    dims = _as_int_tuple(shape, "shape")
    for dim in dims:
        if dim <= 0:
            raise ValueError(
                f"shape dimensions must be positive ints, got {dims} "
                f"(zero-size dimensions are not supported in v0.7)"
            )
    return dims


def _as_offset(offset):
    if not isinstance(offset, (int, np.integer)) or isinstance(offset, bool):
        raise TypeError(f"offset must be an int, got {offset!r}")
    return int(offset)


def row_major_strides(shape):
    """Element strides for a row-major contiguous layout of ``shape``.

    The last dimension varies fastest: row_major_strides((2, 3, 4))
    is (12, 4, 1). The scalar shape () gives ().
    """
    dims = _as_shape(shape)
    strides = []
    running = 1
    for dim in reversed(dims):
        strides.append(running)
        running *= dim
    return tuple(reversed(strides))


def numel(shape):
    """Number of elements in ``shape``; 1 for the scalar shape ()."""
    dims = _as_shape(shape)
    count = 1
    for dim in dims:
        count *= dim
    return count


def is_contiguous_shape(shape, strides):
    """True if ``strides`` is exactly the row-major contiguous layout
    for ``shape``. A scalar () with strides () is contiguous."""
    dims = _as_shape(shape)
    stride_tuple = _as_int_tuple(strides, "strides")
    if len(stride_tuple) != len(dims):
        raise ValueError(
            f"shape and strides must have the same length, "
            f"got {len(dims)} and {len(stride_tuple)}"
        )
    return stride_tuple == row_major_strides(dims)


def flat_offset(indices, strides, offset=0):
    """Flat storage position of a logical index: offset + sum(i * s).

    Pure stride math — no shape is involved, so no bounds checking is
    performed, and negative indices or strides are allowed (real
    strided views use negative strides).
    """
    index_tuple = _as_int_tuple(indices, "indices")
    stride_tuple = _as_int_tuple(strides, "strides")
    if len(index_tuple) != len(stride_tuple):
        raise ValueError(
            f"indices and strides must have the same length, "
            f"got {len(index_tuple)} and {len(stride_tuple)}"
        )
    return _as_offset(offset) + sum(
        index * stride for index, stride in zip(index_tuple, stride_tuple)
    )


def shape_info(shape, strides=None, offset=0):
    """A small metadata dictionary describing one array layout.

    With ``strides=None`` the row-major contiguous strides are used
    (and ``contiguous`` is True by construction). Explicit strides are
    validated against the shape's length and checked for contiguity.
    """
    dims = _as_shape(shape)
    if strides is None:
        stride_tuple = row_major_strides(dims)
    else:
        stride_tuple = _as_int_tuple(strides, "strides")
        if len(stride_tuple) != len(dims):
            raise ValueError(
                f"shape and strides must have the same length, "
                f"got {len(dims)} and {len(stride_tuple)}"
            )
    return {
        "shape": dims,
        "strides": stride_tuple,
        "ndim": len(dims),
        "numel": numel(dims),
        "offset": _as_offset(offset),
        "contiguous": stride_tuple == row_major_strides(dims),
    }


def broadcast_shapes(shape_a, shape_b):
    """The NumPy-style broadcast of two shapes, or a clear ValueError.

    Shapes are aligned from the trailing axis; the shorter one is
    conceptually left-padded with leading 1s. Two extents are compatible
    when they are equal or one of them is 1, and the result extent is
    their max. The scalar shape () broadcasts against anything.

        broadcast_shapes((), (3, 4))          # (3, 4)
        broadcast_shapes((3, 1), (1, 4))      # (3, 4)
        broadcast_shapes((4,), (3, 4))        # (3, 4)  (left-pad to (1, 4))
        broadcast_shapes((1, 3, 1), (2, 1, 5))# (2, 3, 5)
        broadcast_shapes((2, 3), (4, 3))      # ValueError (2 vs 4)

    Pure Python — it never calls NumPy and never touches the compiled
    library, so it is safe and testable whether or not the backend is
    built. Incompatible shapes raise a ValueError naming both original
    shapes and the conflicting extents.
    """
    a = _as_shape(shape_a)  # validates positive-int dims (v0.7 rules)
    b = _as_shape(shape_b)
    rank = max(len(a), len(b))
    pa = (1,) * (rank - len(a)) + a  # left-pad with leading 1s
    pb = (1,) * (rank - len(b)) + b
    out = []
    for da, db in zip(pa, pb):
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            raise ValueError(
                f"cannot broadcast shapes {a} and {b}: incompatible "
                f"dimensions {da} and {db} (neither is 1)"
            )
    return tuple(out)


def _broadcast_strides(shape, strides, out_shape):
    """Read-strides that stretch a (real) ``shape``/``strides`` operand
    over ``out_shape`` without materializing it.

    For each output axis: a real axis that carries a genuine extent keeps
    its real stride; a size-1 axis (or a leading axis introduced by
    left-padding) gets stride 0, so the odometer re-reads the same
    element instead of advancing — that is exactly broadcasting. Assumes
    ``out_shape`` is the broadcast of ``shape`` with the other operand
    (compatibility already checked by ``broadcast_shapes``)."""
    rank = len(out_shape)
    pad = rank - len(shape)
    result = [0] * rank  # leading padded axes stay 0
    for i, dim in enumerate(shape):
        if dim != 1:  # a stretched size-1 axis keeps stride 0
            result[pad + i] = strides[i]
    return result


def _normalize_axis(axis, shape):
    """Normalize a single reduction ``axis`` against ``shape``.

    Accepts a plain int (negative allowed, NumPy-style: ``axis + ndim``),
    validates its type and bounds, and returns the non-negative axis.
    Raises ``TypeError`` for a non-int axis and ``ValueError`` naming both
    the axis and the shape when out of bounds (including any integer axis
    on a scalar). Pure Python — no NumPy, no compiled library."""
    dims = _as_shape(shape)
    ndim = len(dims)
    if not isinstance(axis, (int, np.integer)) or isinstance(axis, bool):
        raise TypeError(f"axis must be None or an int, got {axis!r}")
    value = int(axis)
    normalized = value + ndim if value < 0 else value
    if normalized < 0 or normalized >= ndim:
        raise ValueError(
            f"axis {value} is out of bounds for a tensor of shape {dims} "
            f"(ndim {ndim})"
        )
    return normalized


def reduce_shape(shape, axis=None, keepdims=False):
    """The output shape of reducing ``shape`` over ``axis``.

    ``axis=None`` reduces every element; a single integer ``axis``
    (negative allowed) reduces one dimension. ``keepdims=True`` leaves
    each reduced axis as size 1, ``keepdims=False`` (default) removes it.

        reduce_shape((2, 3))                       # ()
        reduce_shape((2, 3), keepdims=True)        # (1, 1)
        reduce_shape((2, 3), axis=0)               # (3,)
        reduce_shape((2, 3), axis=1, keepdims=True)# (2, 1)
        reduce_shape((), axis=None)                # ()
        reduce_shape((2, 3, 4), axis=-1)           # (2, 3)

    Pure Python — never calls NumPy, never touches the compiled library,
    so it is testable whether or not the backend is built. Tuple/multiple
    axes are not supported yet (a single int or None only). Raises
    ``TypeError`` for a non-bool ``keepdims`` or non-int ``axis``, and
    ``ValueError`` naming both axis and shape for an out-of-bounds axis.
    """
    dims = _as_shape(shape)
    if not isinstance(keepdims, bool):
        raise TypeError(f"keepdims must be a bool, got {keepdims!r}")
    ndim = len(dims)
    if axis is None:
        return (1,) * ndim if keepdims else ()
    normalized = _normalize_axis(axis, dims)
    if keepdims:
        return tuple(1 if d == normalized else dims[d] for d in range(ndim))
    return tuple(dims[d] for d in range(ndim) if d != normalized)


def _reduce_out_strides(in_shape, reduced_axes, keepdims, out_shape):
    """Per-input-axis output write-strides for the sum kernel.

    For each input axis: 0 if it is reduced (so those elements
    accumulate into one output cell), otherwise the row-major stride of
    the axis it maps to in ``out_shape``. With ``keepdims`` the output
    keeps the reduced axes (as size 1), so input axis ``d`` maps to
    output axis ``d``; without it, the kept input axes map in order to the
    surviving output axes. Assumes ``out_shape == reduce_shape(in_shape,
    ...)`` for the same reduction."""
    out_full = row_major_strides(out_shape)
    result = [0] * len(in_shape)
    if keepdims:
        for d in range(len(in_shape)):
            result[d] = 0 if d in reduced_axes else out_full[d]
    else:
        out_index = 0
        for d in range(len(in_shape)):
            if d in reduced_axes:
                continue  # reduced axis: stays 0, no output axis consumed
            result[d] = out_full[out_index]
            out_index += 1
    return result


def _spatial_pair(value, name, minimum):
    """Normalize a spatial argument to a ``(height, width)`` int pair.

    Mirrors the stable ``tensorforge.nn.conv._pair`` semantics in the
    native package's strict style (the two lines never cross-import): a
    plain int ``v`` becomes ``(v, v)``; a 2-element tuple/list of ints is
    taken as ``(height, width)``. Booleans are rejected (``bool`` is an
    ``int`` subclass, so ``True``/``False`` are never valid dimensions) and
    every member must be at least ``minimum`` (``1`` for kernel/stride,
    ``0`` for padding). Raises ``ValueError`` otherwise. Pure Python — never
    touches the compiled library, so it is safe whether or not the backend
    is built."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        pair = (int(value), int(value))
    elif (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool)
            for v in value
        )
    ):
        pair = (int(value[0]), int(value[1]))
    else:
        raise ValueError(
            f"{name} must be an int or a 2-element pair of ints, got {value!r}"
        )
    if any(v < minimum for v in pair):
        raise ValueError(f"{name} values must be >= {minimum}, got {pair}")
    return pair


def conv_output_shape(input_size, kernel_size, stride, padding):
    """The ``(out_h, out_w)`` of a 2-D convolution / pooling window.

    Applies the locked floor formula per spatial axis (identical to the
    stable Conv2d)::

        out = (size + 2*pad - kernel) // stride + 1

    Every argument is a validated ``(height, width)`` pair of ints. The
    arithmetic runs in Python ints (arbitrary precision), so the
    shape math itself can never overflow. Raises ``ValueError`` — naming
    the kernel, stride, padding, and input — when either extent would be
    ``< 1`` (the kernel does not fit the padded input), *before* any output
    is allocated. Pure Python; never touches the compiled library."""
    h, w = input_size
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    if out_h < 1 or out_w < 1:
        raise ValueError(
            f"kernel {(kh, kw)} with stride {(sh, sw)} and padding {(ph, pw)} "
            f"does not fit input {(h, w)}: computed output {(out_h, out_w)} has "
            f"a non-positive extent"
        )
    return out_h, out_w


def _binary_op(kernel_name, a, b, name):
    """Shared plumbing for the binary kernels: convert both inputs to
    contiguous float64, require identical shapes, call the kernel."""
    lib = _require_library()
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"the experimental C++ {name} requires identical "
            f"shapes (no broadcasting), got {a.shape} and {b.shape}"
        )
    out = np.empty_like(a)
    getattr(lib, kernel_name)(a, b, out, a.size)
    return out


def elementwise_add(a, b):
    """a + b elementwise, using the compiled C++ kernel."""
    return _binary_op("tf_elementwise_add", a, b, "elementwise_add")


def elementwise_subtract(a, b):
    """a - b elementwise, using the compiled C++ kernel."""
    return _binary_op("tf_elementwise_subtract", a, b, "elementwise_subtract")


def elementwise_multiply(a, b):
    """a * b elementwise, using the compiled C++ kernel."""
    return _binary_op("tf_elementwise_multiply", a, b, "elementwise_multiply")


def elementwise_divide(a, b):
    """a / b elementwise, using the compiled C++ kernel.

    IEEE float64 division: dividing by zero yields +-inf (or NaN for
    0/0), the same values NumPy produces — but without NumPy's runtime
    warning.
    """
    return _binary_op("tf_elementwise_divide", a, b, "elementwise_divide")


def relu(a):
    """max(a, 0) elementwise, using the compiled C++ kernel.

    Unary: accepts any shape, returns a new float64 array of the same
    shape.
    """
    lib = _require_library()
    a = np.ascontiguousarray(a, dtype=np.float64)
    out = np.empty_like(a)
    lib.tf_relu(a, out, a.size)
    return out


def _prepare_matmul_inputs(a, b, name):
    """Shared matmul validation: contiguous float64, strictly 2-D,
    compatible inner dimensions. Returns (a, b)."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(
            f"the experimental C++ {name} requires a 2-D left input, "
            f"got shape {a.shape}"
        )
    if b.ndim != 2:
        raise ValueError(
            f"the experimental C++ {name} requires a 2-D right input, "
            f"got shape {b.shape}"
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"inner dimensions do not match: "
            f"{a.shape} @ {b.shape} (need (m, n) @ (n, p))"
        )
    return a, b


def matmul(a, b):
    """(m, n) @ (n, p) matrix multiplication using the compiled C++
    kernel — the naive triple loop, kept as the reference that
    matmul_tiled is measured against. Correct but much slower than
    NumPy's BLAS-backed matmul.

    Strictly 2-D: vectors must be passed as (1, n) or (n, 1) matrices.
    Returns a new (m, p) float64 array.
    """
    lib = _require_library()
    a, b = _prepare_matmul_inputs(a, b, "matmul")
    m, n = a.shape
    p = b.shape[1]
    out = np.empty((m, p), dtype=np.float64)
    lib.tf_matmul(a, b, out, m, n, p)
    return out


def matmul_tiled(a, b, block_size=32):
    """(m, n) @ (n, p) matrix multiplication using the tiled C++
    kernel — an optimization experiment in cache blocking. Same
    contract as ``matmul``; ``block_size`` sets the tile edge length
    and any positive int works, including sizes that don't divide the
    matrix dimensions.

    Blocking improves memory locality over the naive loop, but this is
    still single-threaded scalar code: NumPy's BLAS may well remain
    faster. Returns a new (m, p) float64 array.
    """
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError(
            f"block_size must be a positive int, got {block_size!r}"
        )
    lib = _require_library()
    a, b = _prepare_matmul_inputs(a, b, "matmul_tiled")
    m, n = a.shape
    p = b.shape[1]
    out = np.empty((m, p), dtype=np.float64)
    lib.tf_matmul_tiled(a, b, out, m, n, p, block_size)
    return out
