# Backend experiments

This page tracks the advanced experimental line that started after the
v3.0 Python release. **Nothing here is part of the finished Python
framework** — `import tensorforge` never touches it, Tensor and
autograd are unchanged, and every existing API works exactly as
before.

## C++ backend — v1.8 (current)

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

### TensorCore view operations (v1.1)

The payoff of the shape/stride design: **metadata-only views**. These
return new tensor cores over the *same* storage — nothing is copied
until you materialize with `to_numpy()` or `contiguous_copy()`:

```python
t = NativeTensorCore.from_array(np.arange(6.0).reshape(2, 3))

t.reshape((3, 2))     # same storage, new row-major layout
t.transpose()         # all axes reversed; t.transpose(1, 0) works too
t.T                   # NumPy's .T: reversed axes, no-op for 1-D
t.narrow(1, 1, 2)     # keep 2 positions of dim 1 starting at 1
```

`reshape` requires a contiguous tensor (a non-contiguous layout can't
be reinterpreted by strides alone — materialize first) and the same
element count. `transpose` permutes shape and strides. `narrow`
shrinks one dimension and advances the offset by
``start * strides[dim]``. All of them chain.

**Ownership with shared storage:** the core created by
`from_array`/`zeros`/`full` owns the storage; view cores borrow it.
Closing a view closes only that view — the owner and sibling views
keep working. Closing the owner releases the memory for every core
sharing it, after which their data operations raise.

### Kernels over tensor cores (v1.2)

The step that makes the native runtime self-contained for simple
compute: `relu`, `add`, `subtract`, and `multiply` as
`NativeTensorCore` methods, computed **entirely in C++** — the
kernels read the input's storage plus shape/stride/offset metadata
directly (the same odometer traversal as materialization, so
transposed and narrowed views work without being materialized first)
and write into fresh contiguous native storage. No NumPy round trip
is involved in the arithmetic.

```python
a = NativeTensorCore.from_array([[1.0, 2.0], [3.0, 4.0]])
b = NativeTensorCore.from_array([[10.0, 20.0], [30.0, 40.0]])

a.add(b)            # a new contiguous NativeTensorCore
a.T.multiply(b.T)   # strided views compute directly — no copy first
a.relu()
```

Binary operations require exactly matching shapes — no broadcasting
yet. Outputs are always new row-major contiguous tensor cores,
independent of their inputs. `backend_info()` lists these under
``tensor_core_kernels``, separate from the raw NumPy-buffer kernels
in `list_kernels()`.

### TensorCore matmul (v1.3)

The last major compute primitive: `a.matmul(b)` multiplies two 2-D
tensor cores entirely in C++, addressing each source element through
its own strides and offset — so a transposed or narrowed view
multiplies directly, no materialization:

```python
a = NativeTensorCore.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = NativeTensorCore.from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

a.matmul(b)      # (2, 2) contiguous NativeTensorCore
a.T.matmul(c)    # a transposed view multiplies without copying
```

Strictly `(m, n) @ (n, p)`, 2-D only, no broadcasting; the naive
triple loop, matching the reference matmul. The output is a new
row-major contiguous tensor core.

To be precise about what this is **not**: it is not
`tensorforge.Tensor`, it has no autograd, and there is no backend
dispatch. Future milestones may add TensorCore benchmarks, an
explicit backend dispatch design, or optional integration behind a
flag.

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

### Explicit backend API (v1.5)

The safe, user-facing entry point for backend experiments: name a
backend, get its object. Nothing selects a backend implicitly, and
nothing here touches `tensorforge.Tensor`.

```python
from tensorforge.backends import get_backend, available_backends

available_backends()             # ('numpy', 'native')
numpy = get_backend("numpy")
native = get_backend("native")

numpy.add([1.0, 2.0], [3.0, 4.0])          # a float64 NumPy array
t = native.tensor_from_array([[1.0, 2.0]]) # a NativeTensorCore
native.matmul(t, t.T)                       # -> NativeTensorCore
```

Both backends expose the same small surface — `name`, `available()`,
`backend_info()`, `tensor_from_array`, `to_numpy`, `zeros`, `full`,
`add`, `relu`, `matmul`. The NumPy backend is always available and
follows NumPy semantics; the native backend is constructible whether
or not the compiled library is built (`available()` reports which),
consumes and produces `NativeTensorCore` objects, and requires exact
shapes.

**Conversion boundaries (v1.6).** Data crosses a backend only by
explicit call. `tensor_from_array` *enters* a backend (Python/NumPy
data → a backend-native value, copied); `to_numpy` *exits* it (a
backend-native value → a fresh float64 NumPy array, materialized).
Copies are visible in both directions, so nothing accidentally aliases
native storage, and the native backend rejects anything that is not a
`NativeTensorCore` — including a `tensorforge.Tensor` — with a
consistent TypeError across every operation. This also makes the
Stage-1 shape asymmetry explicit rather than hidden: the NumPy
backend's `add` broadcasts (a NumPy array already is one), while the
native backend's `add` requires exact shapes and fails clearly
otherwise. Aligning those semantics is a future design item, not a
conversion detail.

This is Stage 1 of a longer plan: how (and whether) backends should
eventually meet `tensorforge.Tensor`, and the risks that gate each
step, are laid out in [dispatch_design.md](dispatch_design.md). The
governing rule is **no implicit fallback**: an unavailable native
operation raises with build instructions; it never quietly falls back
to NumPy.

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

The suite (v1.4) measures every implementation of each operation
against a shared NumPy baseline: the **raw-buffer kernels** (naive
loops over contiguous NumPy arrays, converted at the call boundary)
and the **NativeTensorCore kernels** (native compute over storage +
shape/stride metadata), including non-contiguous view rows where
transposed inputs feed the kernels directly. It is deliberately
**not** a performance claim — the point is what the numbers teach:

- On small arrays, everything native loses to NumPy: per-call ctypes
  and conversion overhead dominates.
- On large elementwise arrays the raw-buffer loop gets competitive
  with NumPy (both memory-bound).
- The TensorCore rows include what the raw rows don't: Python wrapper
  cost, output-storage allocation, and the generic strided-traversal
  (odometer) loop — visibly slower than the flat raw-buffer loop for
  elementwise work. That overhead is the honest price of layout
  generality, and measuring it is the reason these rows exist.
- For matmul, the triple loop dominates everything else, so the
  TensorCore path costs about the same as the raw naive kernel —
  and NumPy's BLAS beats both by an order of magnitude.

Correctness is verified (each implementation against its own NumPy
reference — view rows compute transposed results) before anything is
timed, and timings are medians over repeated runs after warmup.
Results are hardware-dependent and should not be oversold; expect
exact numbers to vary, the *shape* of the story shouldn't.

### NativeTensor — minimal wrapper (v1.8)

v1.8 implements the *shell* of the forward-only wrapper the v1.7 design
laid out — constructors, metadata, conversion, and lifetime only. No
compute ops, no view ops yet (those are v1.9 and v1.10). It lives in its
own opt-in package; `import tensorforge` never touches it.

```python
from tensorforge.experimental import NativeTensor

with NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]]) as t:
    t.shape, t.strides, t.ndim, t.numel, t.contiguous  # layout metadata
    t.to_numpy()                                        # fresh float64 copy
# released on block exit; t.close() works too (idempotent)

z = NativeTensor.zeros((2, 3))
f = NativeTensor.full((2, 2), 7.0)
z.owns_core, z.closed        # ownership / lifetime state
z.close()
```

`NativeTensor` wraps a single `NativeTensorCore`: constructors
(`from_array`, `zeros`, `full`) own the core they create, so `close()`
(or a `with` block) releases the native storage. A closed tensor rejects
metadata and `to_numpy()` with a clear `RuntimeError`, while `closed`
and `owns_core` stay readable. Conversion crosses the native boundary
only by explicit call — `from_array` enters, `to_numpy` exits, both as
copies.

It is deliberately **not** `tensorforge.Tensor`: no autograd, no
`requires_grad`/`grad`/`backward`, no optimizer/Module integration, no
CUDA. And it is only a shell — `relu`/`add`/`subtract`/`multiply`/
`matmul` and the view ops exist on `NativeTensorCore` but are not
exposed on `NativeTensor` yet, by design. Constructors need the compiled
backend; if it is unbuilt they raise the same build-instructions
`ImportError` as the rest of the native runtime. The full design,
including the staged plan, is in
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md).

### Forward-only native tensor wrapper — design (v1.7)

v1.7 is **design-only**: no code ships. It writes down the Stage-2 plan
for a future forward-only convenience wrapper over `NativeTensorCore`
(likely named `NativeTensor`) — its purpose, non-goals, ownership and
lifetime rules, the v1.6 conversion contract it inherits, a minimal API
sketch, error/shape behavior, a testing plan, and a staged v1.8–v1.11
implementation sequence. The full design is in
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md). It
stays forward-only, autograd-free, float64, exact-shape, and explicitly
**not** `tensorforge.Tensor`. Implementation does not begin until v1.8.

## What might come next

The next implementation step is v1.9: `NativeTensor` compute ops —
`relu`, `add`, `subtract`, `multiply`, `matmul` — each returning a new
owning wrapper, with view ops following in v1.10 per the design doc. A
further-out milestone might consider wiring kernels into Tensor behind a
flag. CUDA experiments remain a separate future branch. The Python
framework stays the reference implementation throughout.
