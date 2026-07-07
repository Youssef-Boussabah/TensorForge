"""Experimental C++ backend.

A ctypes wrapper around tiny compiled kernels (see cpp/ at the repo
root). This is a proof of concept that Python TensorForge can call
compiled code — it is NOT wired into Tensor or autograd, and the
normal framework never imports it.

All kernels are float64-only. Binary operations require identical
shapes; broadcasting is deliberately not supported.

Importing this module raises ImportError with build instructions if
the shared library has not been compiled.
"""

import ctypes
import platform
from pathlib import Path

import numpy as np

_SUFFIX = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
_LIBRARY_PATH = Path(__file__).with_name("_tensorforge_cpp" + _SUFFIX)

_BINARY_KERNELS = (
    "tf_elementwise_add",
    "tf_elementwise_subtract",
    "tf_elementwise_multiply",
    "tf_elementwise_divide",
)


def _load_library():
    if not _LIBRARY_PATH.exists():
        raise ImportError(
            f"The experimental C++ backend is not built "
            f"(missing {_LIBRARY_PATH.name}). Build it from the repo root "
            f"with: uv run python cpp/build.py "
            f"(add 'uv sync --group cpp' first if you have no C++ compiler)."
        )
    library = ctypes.CDLL(str(_LIBRARY_PATH))
    f64_array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    for name in _BINARY_KERNELS:
        kernel = getattr(library, name)
        kernel.argtypes = [f64_array, f64_array, f64_array, ctypes.c_int64]
        kernel.restype = None
    library.tf_relu.argtypes = [f64_array, f64_array, ctypes.c_int64]
    library.tf_relu.restype = None
    return library


_lib = _load_library()


def _binary_op(kernel, a, b, name):
    """Shared plumbing for the binary kernels: convert both inputs to
    contiguous float64, require identical shapes, call the kernel."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"the experimental C++ {name} requires identical "
            f"shapes (no broadcasting), got {a.shape} and {b.shape}"
        )
    out = np.empty_like(a)
    kernel(a, b, out, a.size)
    return out


def elementwise_add(a, b):
    """a + b elementwise, using the compiled C++ kernel."""
    return _binary_op(_lib.tf_elementwise_add, a, b, "elementwise_add")


def elementwise_subtract(a, b):
    """a - b elementwise, using the compiled C++ kernel."""
    return _binary_op(_lib.tf_elementwise_subtract, a, b, "elementwise_subtract")


def elementwise_multiply(a, b):
    """a * b elementwise, using the compiled C++ kernel."""
    return _binary_op(_lib.tf_elementwise_multiply, a, b, "elementwise_multiply")


def elementwise_divide(a, b):
    """a / b elementwise, using the compiled C++ kernel.

    IEEE float64 division: dividing by zero yields +-inf (or NaN for
    0/0), the same values NumPy produces — but without NumPy's runtime
    warning.
    """
    return _binary_op(_lib.tf_elementwise_divide, a, b, "elementwise_divide")


def relu(a):
    """max(a, 0) elementwise, using the compiled C++ kernel.

    Unary: accepts any shape, returns a new float64 array of the same
    shape.
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    out = np.empty_like(a)
    _lib.tf_relu(a, out, a.size)
    return out
