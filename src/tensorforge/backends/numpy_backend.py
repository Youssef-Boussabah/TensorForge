"""The NumPy reference backend.

The baseline every other backend is measured against: plain float64
NumPy, always available, NumPy semantics (broadcasting included). See
docs/dispatch_design.md for why the native backend deliberately
differs.
"""

import numpy as np


class NumpyBackend:
    name = "numpy"

    def available(self):
        return True

    def backend_info(self):
        return {
            "name": "numpy",
            "available": True,
            "experimental": False,
            "dtype": "float64",
            "device": "cpu",
        }

    def tensor_from_array(self, values):
        """A contiguous float64 NumPy array copy of ``values``."""
        return np.array(values, dtype=np.float64)

    def zeros(self, shape):
        return np.zeros(shape, dtype=np.float64)

    def full(self, shape, fill_value):
        return np.full(shape, float(fill_value), dtype=np.float64)

    def add(self, a, b):
        return np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)

    def relu(self, a):
        return np.maximum(np.asarray(a, dtype=np.float64), 0.0)

    def matmul(self, a, b):
        return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)

    def __repr__(self):
        return "NumpyBackend()"
