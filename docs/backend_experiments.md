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
Importing `tensorforge.backends.cpp` always succeeds — the library
loads lazily on first use. If the backend is not built, calling a
math kernel raises ImportError with these instructions, and the
backend tests skip.

### Inspecting the backend

The namespace answers its own questions:

```python
from tensorforge.backends import cpp

cpp.is_available()        # True only if the compiled library loads
cpp.list_kernels()        # ('elementwise_add', ..., 'relu', 'matmul')
cpp.build_instructions()  # how to build it, as a string
cpp.backend_info()        # one dict with all of the above, plus
                          # dtype='float64' and the (false) tensor/
                          # autograd integration flags
```

`is_available()` performs a real load attempt (cached), not just a
file check, and never raises. Everything here is still experimental
and not wired into Tensor or autograd — `backend_info()` says so
explicitly.

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

## Benchmarks

After building the backend, compare it against NumPy:

```
uv run python benchmarks/cpp_backend.py          # default sizes
uv run python benchmarks/cpp_backend.py --quick  # fast smoke run
```

The output is a small table (operation, shape, NumPy time, C++ time,
ratio). It is deliberately **not** a performance claim — the point is
what the numbers teach:

- On tiny arrays, the C++ backend loses badly: every call pays ctypes
  and array-conversion overhead that NumPy's own dispatch amortizes
  better.
- On large elementwise arrays, naive C++ gets competitive with NumPy
  (both end up memory-bound — the loop isn't the bottleneck).
- For matmul, NumPy's BLAS (blocking, SIMD, threading) beats the
  textbook triple loop by an order of magnitude, and the gap grows
  with size.

Correctness is verified against NumPy before anything is timed, and
timings are medians over repeated runs after warmup. Expect exact
numbers to vary by machine; the *shape* of the story shouldn't.

## What might come next

A future milestone might consider wiring kernels into Tensor behind a
flag. CUDA experiments remain a separate future branch. The Python
framework stays the reference implementation throughout.
