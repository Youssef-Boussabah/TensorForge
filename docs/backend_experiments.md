# Backend experiments

This page tracks the advanced experimental line that started after the
v3.0 Python release. **Nothing here is part of the finished Python
framework** — `import tensorforge` never touches it, Tensor and
autograd are unchanged, and every existing API works exactly as
before.

## C++ backend — v1.0 (current)

Proof that Python TensorForge can call compiled C++ code, now with a
small family of kernels:

- `cpp/kernels.cpp` — plain C-ABI functions over float64 buffers:
  elementwise add, subtract, multiply, divide, ReLU, a naive 2-D
  matmul (the textbook triple loop, kept as the reference kernel),
  and a tiled matmul (the cache-blocking optimization experiment).
  No Python C-API, no pybind11, no NumPy headers.
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
matmul_tiled(np.ones((2, 3)), np.ones((3, 4)))               # same values
```

### Shape/stride metadata (v0.7)

The first step from "raw kernels over buffers" toward a real tensor
runtime foundation. A tensor runtime needs to describe *layout*, not
just hold data: which logical index lives at which storage position.
That is exactly what shape + strides + offset encode — and it is what
makes views, transposes, and slices possible without copying.

```python
from tensorforge.backends.cpp import (
    row_major_strides, numel, is_contiguous_shape, flat_offset, shape_info,
)

row_major_strides((2, 3, 4))            # (12, 4, 1)
numel((2, 3, 4))                        # 24
is_contiguous_shape((3, 2), (1, 3))     # False — transposed layout
flat_offset((1, 2, 3), (12, 4, 1))      # 23
shape_info((2, 3, 4))                   # one dict with all of the above
```

Row-major contiguous layout means the last dimension varies fastest:
element `(i, j, k)` of a `(2, 3, 4)` array lives at flat position
`i*12 + j*4 + k`. Strides here count **elements**, not bytes (unlike
`numpy.ndarray.strides`). Zero-size dimensions are rejected for now;
their conventions deserve their own tested milestone.

This layer is deliberately Python-side: it is the metadata *contract*
that a later native storage object will honor. It is not yet a C++
Tensor object, and it is not connected to Tensor/autograd.

### Native storage (v0.8)

`NativeStorage` is the storage half: a **C++-owned float64 buffer**
behind an opaque handle. Python never sees the raw pointer — data
moves by copy, and the native memory is released explicitly:

```python
from tensorforge.backends.cpp import NativeStorage

with NativeStorage.from_array([1.0, 2.0, 3.0]) as storage:
    storage.size          # 3
    storage.fill(5.0)
    storage.to_numpy()    # a new, independent NumPy copy
    storage.copy_from([7.0, 8.0, 9.0])
# closed on exit; storage.close() works too, and double-close is safe
```

New storage is zero-initialized; operations on a closed storage raise
RuntimeError. `backend_info()` advertises it as ``storage_object``.

Storage by itself has a size but no shape or strides — that binding
is the tensor view's job.

### NativeTensorView (v0.9)

`NativeTensorView` connects the two halves: a `NativeStorage` plus
shape/stride/offset metadata, forming a logical view that knows which
storage element each index means. Its first operation is **contiguous
materialization** — walking the strided view in row-major order and
copying it out — performed by a native odometer loop in C++:

```python
from tensorforge.backends.cpp import NativeStorage, NativeTensorView

view = NativeTensorView.from_array(np.arange(6.0).reshape(2, 3))
view.shape, view.strides, view.contiguous   # (2, 3), (3, 1), True

# The same six values seen transposed, without copying anything:
storage = NativeStorage.from_array(np.arange(6.0))
transposed = NativeTensorView(storage, shape=(3, 2), strides=(1, 3))
transposed.to_numpy()          # the (3, 2) transpose, materialized
copy = transposed.contiguous_copy()  # a new row-major NativeStorage
```

Views are bounds-checked at construction (negative strides included),
so a valid view can never read outside its storage. Views don't own
the storage: close the storage and the view's operations raise.

Views by themselves are building blocks; the runtime object that
composes everything is the tensor core.

### NativeTensorCore (v1.0)

`NativeTensorCore` is the first native tensor runtime object: an
owned `NativeStorage` plus a `NativeTensorView`, composed into one
thing you can create, inspect, materialize, and release:

```python
from tensorforge.backends.cpp import NativeTensorCore

t = NativeTensorCore.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
t.shape, t.strides, t.contiguous   # (2, 3), (3, 1), True
t.to_numpy()                       # fresh (2, 3) NumPy copy
c = t.contiguous_copy()            # a new, independent tensor core
z = NativeTensorCore.zeros((2, 2))
f = NativeTensorCore.full((2, 2), 7.0)
t.close()                          # releases the owned native memory
```

The core owns its storage (context managers work; double-close is
safe; data operations on a closed core raise RuntimeError).
`backend_info()` advertises it as ``tensor_core``.

To be precise about what this is **not**: it is not
`tensorforge.Tensor`, it has no autograd, and it performs no math yet
— even the compiled kernels don't operate on it. It is the foundation
for exactly those later steps: view operations (transpose, slice,
reshape as methods), kernels over tensor cores, explicit backend
dispatch, and eventually native backend integration.

### The tiled matmul experiment

`matmul_tiled(a, b, block_size=32)` is the first *optimization*
experiment: same contract and same results as the naive `matmul`, but
computed block by block. Tiling is about **memory locality** — the
naive loop re-reads the same rows and columns from main memory over
and over once matrices outgrow the CPU cache, while the tiled version
works on small sub-matrices that stay cache-resident during their
reuse, and orders its inner loops to walk memory sequentially. Any
positive `block_size` works, including ones that don't divide the
matrix dimensions.

The naive matmul deliberately stays: it is the reference the
experiment is measured against. And the honest caveat applies as ever:
tiling alone is one idea from a long list (SIMD, threading, deeper
blocking...) that libraries behind NumPy implement — NumPy may well
still be faster. Neither matmul is connected to Tensor or autograd.

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
- Both matmuls are strictly 2-D — `(m, n) @ (n, p)` only, vectors must
  be passed as `(1, n)` / `(n, 1)` matrices. `matmul` is the naive
  triple loop; `matmul_tiled` adds cache blocking but remains
  single-threaded scalar code. NumPy's BLAS-backed matmul is expected
  to stay faster.
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
