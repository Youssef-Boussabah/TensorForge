"""Experimental C++ backend.

A ctypes wrapper around a tiny compiled kernel (see cpp/ at the repo
root). This is a proof of concept that Python TensorForge can call
compiled code — it is NOT wired into Tensor or autograd, and the
normal framework never imports it.

Importing this module raises ImportError with build instructions if
the shared library has not been compiled.
"""

import ctypes
import platform
from pathlib import Path

import numpy as np

_SUFFIX = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
_LIBRARY_PATH = Path(__file__).with_name("_tensorforge_cpp" + _SUFFIX)


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
    library.tf_elementwise_add.argtypes = [f64_array, f64_array, f64_array, ctypes.c_int64]
    library.tf_elementwise_add.restype = None
    return library


_lib = _load_library()


def elementwise_add(a, b):
    """Add two arrays elementwise using the compiled C++ kernel.

    Inputs are converted to contiguous float64 NumPy arrays (the only
    dtype the experimental kernel supports). Shapes must match exactly
    — broadcasting is deliberately not supported in v0.1. Returns a new
    float64 array with the same shape as the inputs.
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"the experimental C++ elementwise_add requires identical "
            f"shapes (no broadcasting), got {a.shape} and {b.shape}"
        )
    out = np.empty_like(a)
    _lib.tf_elementwise_add(a, b, out, a.size)
    return out
