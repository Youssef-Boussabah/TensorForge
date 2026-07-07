# Backend experiments

This page tracks the advanced experimental line that started after the
v3.0 Python release. **Nothing here is part of the finished Python
framework** — `import tensorforge` never touches it, Tensor and
autograd are unchanged, and every existing API works exactly as
before.

## C++ backend — v0.3 (current)

Proof that Python TensorForge can call compiled C++ code, now with a
small family of kernels:

- `cpp/kernels.cpp` — plain C-ABI functions over float64 buffers:
  elementwise add, subtract, multiply, divide, ReLU, and a naive 2-D
  matmul (the textbook triple loop). No Python C-API, no pybind11,
  no NumPy headers.
- `src/tensorforge/backends/cpp.py` — a ctypes wrapper that loads the
  compiled shared library and exposes the kernels as Python functions,
  handling array conversion and validation on the Python side.

Usage:

```python
from tensorforge.backends.cpp import (
    elementwise_add, elementwise_subtract, elementwise_multiply,
    elementwise_divide, relu, matmul,
)

elementwise_add(np.array([1.0, 2.0]), np.array([3.0, 4.0]))  # [4. 6.]
relu(np.array([-1.0, 0.0, 2.0]))                             # [0. 0. 2.]
matmul(np.ones((2, 3)), np.ones((3, 4)))                     # (2, 4) of 3s
```

### Building it

```
uv run python cpp/build.py
```

The build script uses `g++` or `clang++` if you have one. If not,
install the bundled-compiler fallback first (the `ziglang` package
ships a clang-based C++ compiler that works anywhere uv works):

```
uv sync --group cpp
uv run python cpp/build.py
```

The compiled library lands next to the wrapper and is gitignored.
If the backend is not built, importing `tensorforge.backends.cpp`
raises an ImportError with these instructions, and its tests skip.

CI does not rely on that skip: the GitHub Actions workflow builds the
backend from source on every run and smoke-tests the compiled kernel
with a hard-failing check before running the suite — so a broken
build or kernel fails CI instead of silently skipping.

### Current limitations

- float64 only (other inputs are converted).
- Binary operations require identical shapes — no broadcasting.
  `relu` is unary and accepts any shape.
- Division follows IEEE float64 rules (inf/NaN for zero denominators,
  the same values as NumPy) but does not emit NumPy's runtime warning.
- `matmul` is strictly 2-D — `(m, n) @ (n, p)` only, vectors must be
  passed as `(1, n)` / `(n, 1)` matrices — and it is the naive triple
  loop: correct, but much slower than NumPy's BLAS-backed matmul. The
  point is the mechanism, not speed.
- Not connected to Tensor or autograd.
- A proof of mechanism, not a performance claim.

## What might come next

Future backend milestones might compare kernels honestly against
NumPy, and only much later consider wiring kernels into Tensor behind
a flag. CUDA experiments remain a separate future branch. The Python
framework stays the reference implementation throughout.
