# Backend experiments

This page tracks the advanced experimental line that started after the
v3.0 Python release. **Nothing here is part of the finished Python
framework** — `import tensorforge` never touches it, Tensor and
autograd are unchanged, and every existing API works exactly as
before.

## C++ backend — v1.14 (current)

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
transposed inputs feed the kernels directly — and, since v1.12, the
**NativeTensor wrapper** rows on top of those (see below). It is
deliberately **not** a performance claim — the point is what the numbers
teach:

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

### Contiguous fast-path — benchmark impact report (v1.15)

This closes the optimization loop the last four milestones opened:
**v1.12** measured where the elementwise cost lived (the generic
shape/stride odometer traversal in the native runtime, not the
`NativeTensor` wrapper), **v1.13** designed a contiguous fast path,
**v1.14** implemented it, and **v1.15** records the measured impact. It
is a *report*, not new behavior — no kernels or runtime changed.

The table below is a single local run of
`uv run python benchmarks/cpp_backend.py` (full sizes). Numbers are
`vs numpy` ratios (higher = slower than NumPy); the large `1000x1000`
elementwise rows are the honest ones to read, because they are
memory-bound and dominated by traversal cost rather than per-call ctypes
overhead. **These figures are hardware-dependent and from one machine —
they characterize behavior, they are not a benchmark score.**

| op (1000×1000) | cpp raw buffer | tensor core (contiguous) | tensor core (view) | native tensor (contiguous) | native tensor (view) |
|----|----|----|----|----|----|
| add  | 1.0× | **1.5×** | 3.5× | **1.4×** | 3.4× |
| relu | 1.0× | **1.5×** | 2.5× | **1.6×** | 2.5× |

What the run shows, and only what it shows:

- **Contiguous elementwise rows moved toward the raw-buffer C++ loop.**
  On this run, contiguous `add`/`relu` on `NativeTensorCore` and
  `NativeTensor` land around **1.5× NumPy at `1000x1000` — essentially
  matching the flat raw-buffer kernel** (≈1.0×) and closing most of the
  gap the odometer used to carry. The remaining margin over raw buffer
  is output-storage allocation and Python call overhead, not traversal.
- **Non-contiguous view rows stayed on the generic odometer path** and
  remain slower — about **3.5×** (add) and **2.5×** (relu) at
  `1000x1000`, versus the contiguous ~1.5×. Same op, contiguous vs
  strided, side by side: that spread is exactly the cost the fast path
  removes, and it is retained (not regressed) for the strided case the
  odometer must still serve.
- **Matmul and `contiguous_copy` are unchanged**, as intended — they
  were out of v1.14 scope (matmul is a different triple-loop kernel;
  `contiguous_copy` materializes a transposed view through the odometer
  by definition). Their rows sit where they did before.
- **`NativeTensor` still tracks `NativeTensorCore` closely** (e.g. add
  1.4× vs 1.5×, relu 1.6× vs 1.5×), reinforcing the v1.12 conclusion
  that the wrapper is thin — and confirming that an optimization placed
  *below* the wrapper reaches it automatically, with no wrapper change.
- **NumPy/BLAS remains the baseline to respect.** Nothing here is faster
  than NumPy; the elementwise wins are about *approaching* NumPy and the
  raw-buffer loop, and matmul stays roughly an order of magnitude behind
  NumPy's BLAS. Small-array rows still lose to NumPy on ctypes/conversion
  overhead, as before.

As always, correctness is verified against each layer's own NumPy
reference before any timing, timings are medians after warmup, and
**no test asserts a speed or ratio** — timing is measured and reported,
never gated. Broadcasting and reductions remain separate, later Phase A
work; this milestone only reports the contiguous elementwise result.

### Contiguous fast-path — implementation (v1.14)

v1.14 implements the fast path the v1.13 document designed. Beside the
generic odometer kernels, `cpp/kernels.cpp` now carries flat, index-free
loops — `tf_core_relu_contiguous`, `tf_core_add_contiguous`,
`tf_core_subtract_contiguous`, `tf_core_multiply_contiguous` — that take
`numel` and the operand offsets instead of shape/strides.
`NativeTensorCore.relu` and `_binary_core_op` select them when every
operand is row-major contiguous (the output is always freshly allocated
contiguous storage); if any input is a strided view, the existing
generic odometer kernel runs unchanged. The choice is driven by the
`contiguous` flag already computed at view construction, so it costs
nothing to make.

The two paths are **bit-for-bit equivalent** — for a contiguous tensor
the odometer's source sequence is exactly `offset, offset+1, …`, so the
flat loop reads the same elements in the same order, with no
reassociation, FMA, or SIMD horizontal reductions that could change
float64 rounding. Nonzero offsets are handled by starting from
`data + offset`, so a contiguous row slice (`narrow` along axis 0) takes
the fast path too. Scalars and size-1 dimensions fall out naturally as
`numel == 1` (or a plain contiguous walk). The v1.14 tests assert exact
equality (`np.array_equal`) against both NumPy and the generic path on
value-identical non-contiguous inputs; transposed and inner-narrowed
views keep matching NumPy through the retained odometer path.

Nothing above `NativeTensorCore` changed: `NativeTensor` inherits the
fast path with **no wrapper edits**, exactly as the design predicted
(the v1.12 benchmarks had already shown the wrapper was not the
bottleneck). Semantics are untouched — exact-shape only, no
broadcasting, no reductions, no autograd, no `Tensor` integration, no
CUDA — and the public kernel lists (`list_kernels()`,
`tensor_core_kernels`) are unchanged, since the new kernels are internal
traversal variants, not new operations.

The **performance impact is a benchmark question, not a claim made
here.** Running `benchmarks/cpp_backend.py` shows the contiguous
`tensor core` / `native tensor` elementwise rows moving toward the flat
`cpp raw buffer` row while the `… (view)` rows stay on the odometer —
but exact numbers are hardware-dependent and nothing asserts a speedup.
An honest, tabulated impact report over the same suite is the next
milestone, **v1.15**.

### Contiguous fast-path — design (v1.13)

v1.13 is **design-only**: no kernels change. It follows directly from the
v1.12 benchmark finding — that the elementwise gap to NumPy comes from the
generic shape/stride odometer traversal in the native runtime, not from
the `NativeTensor` wrapper (the `native tensor` rows track their `tensor
core` rows closely). The design specifies a contiguous fast path for the
elementwise kernels (`relu`/`add`/`subtract`/`multiply`): contiguous
inputs and outputs use a flat, index-free pointer loop, while
non-contiguous views keep the current odometer traversal. The branch
lives in the `NativeTensorCore`/native-kernel layer, so `NativeTensor`
inherits it with no wrapper changes; results stay bit-for-bit identical
and error/shape semantics are untouched. Because nothing is implemented
yet, **no performance gain is claimed** — the full design, scope, tests,
and risks are in
[native_contiguous_fast_path_design.md](native_contiguous_fast_path_design.md).
Implementation is v1.14.

### NativeTensor — benchmark coverage (v1.12)

v1.12 extends the benchmark suite to characterize the `NativeTensor`
wrapper. Each operation is now measured across up to four layers:
**NumPy**, the **raw-buffer C++ kernels**, the **NativeTensorCore**
runtime, and the **NativeTensor** wrapper — for `add`, `relu`, `matmul`,
their strided-view forms (`x.T.relu()`, `x.T.matmul(y)`), and
`contiguous_copy` (materializing a transposed view; no raw-buffer analog,
so NumPy's `ascontiguousarray` is the baseline).

The point is the gap between a `tensor core` row and its `native tensor`
row: that difference is exactly the wrapper's extra **ownership,
lifetime, and conversion bookkeeping** layered on top of the same native
compute. In practice the two rows sit close together — the wrapper is
thin — which is the honest thing to show.

```
uv run python benchmarks/cpp_backend.py          # full sizes
uv run python benchmarks/cpp_backend.py --quick  # fast smoke run
```

As ever this is **characterization, not a performance claim**: NumPy
wins (often dramatically for matmul), the C++ kernels are naive
single-threaded reference loops, and nothing here asserts a speedup.
Correctness is verified against each layer's own NumPy reference before
any timing (view rows legitimately compute transposed results); timings
are medians after warmup; results are hardware-dependent. The benchmark
writes no files and is never run by pytest — a lightweight test only
checks the plan/row structure.

### NativeTensor — runnable example and polish (v1.11)

With the wrapper feature-complete as a forward-only tensor, v1.11 makes
it demonstrable. `examples/native_tensor_demo.py` is a small, fast,
deterministic tour — construction, `relu`/`add`/`matmul`, the views
(`reshape`/`T`/`narrow`), `contiguous_copy`, and explicit `close()` /
`with` — printing small NumPy-materialized outputs:

```
uv run python examples/native_tensor_demo.py
```

It needs the compiled backend; if it is not built, the script prints the
build instructions and exits cleanly rather than raising. `NativeTensor`
also has a metadata-only `repr` (`NativeTensor(shape=(2, 3),
contiguous=True)`, or `NativeTensor(closed)`) that never materializes
data and is safe on a closed tensor.

To be exact about what `NativeTensor` is: it is an **experimental,
forward-only** wrapper over the native C++ runtime, and it is
**separate from `tensorforge.Tensor`** — the two never mix. It has **no
autograd** (no `requires_grad`/`grad`/`backward`), **no implicit
dispatch** and no silent NumPy fallback (data crosses only through
`from_array` / `to_numpy`), **no CUDA**, and **no Python operator
overloads** (compute is method-only: `a.add(b)`, not `a + b`). It makes
**no performance claims** — it is a correctness and design experiment;
NumPy remains the reference implementation.

### NativeTensor — view ops (v1.10)

v1.10 gives `NativeTensor` the metadata-only view operations, delegating
to `NativeTensorCore`'s existing views. `reshape`, `transpose`, `T`, and
`narrow` return **borrowing** wrappers (`owns_core` is False) that share
the parent's storage — no data is copied — while `contiguous_copy`
returns a fresh **owning** wrapper that materializes the data into
row-major contiguous native storage:

```python
from tensorforge.experimental import NativeTensor

t = NativeTensor.from_array(np.arange(6.0).reshape(2, 3))

t.reshape((3, 2))     # borrowing view, same storage
t.transpose()         # all axes reversed; t.transpose(1, 0) works too
t.T                   # NumPy's .T
t.narrow(1, 1, 2)     # keep 2 positions of dim 1 from index 1
t.T.contiguous_copy() # a new OWNING, contiguous NativeTensor

t.transpose().relu()          # compute runs over a strided view directly
t.T.matmul(other)             # transposed view multiplies without copying
```

Lifetime follows the runtime beneath it: closing a borrowing view leaves
the owner (and sibling views) alive, while closing the owner releases the
shared storage — after which the views' data access (`to_numpy`,
compute) raises `RuntimeError`. `contiguous_copy` is independent: closing
it never touches the original. Invalid reshapes, non-permutation
transpose axes, and out-of-bounds `narrow` raise the same clear
`ValueError`/`TypeError` as `NativeTensorCore`.

Still forward-only and still **not** `tensorforge.Tensor`: no autograd,
no Tensor integration, no CUDA, and no Python operator overloads.

### NativeTensor — forward compute ops (v1.9)

v1.9 gives `NativeTensor` its forward-only compute methods, each
delegating to the `NativeTensorCore` kernel beneath and returning a
**new owning** `NativeTensor`:

```python
from tensorforge.experimental import NativeTensor

a = NativeTensor.from_array([[1.0, -2.0], [3.0, 4.0]])
b = NativeTensor.from_array([[5.0, 6.0], [7.0, 8.0]])

a.relu()          # max(x, 0), a new owning NativeTensor
a.add(b)          # a + b        (also subtract, multiply)
a.matmul(b)       # (m, n) @ (n, p)
a.relu().add(b).matmul(b)   # ops chain
```

The behavior mirrors the native runtime exactly. Elementwise ops
(`add`/`subtract`/`multiply`) require **identical shapes — no
broadcasting** (a `(2, 3)` with a `(3,)` raises `ValueError`); `relu` is
unary and takes any shape; `matmul` is strictly 2-D `(m, n) @ (n, p)`.
Wrong operand types raise a clear `TypeError` naming `NativeTensor`
(passing a raw NumPy array or list fails loudly), and computing on or
with a closed tensor raises `RuntimeError`. The original operands are
never consumed — each op allocates a fresh contiguous result.

Still forward-only and still **not** `tensorforge.Tensor`: no autograd,
no `requires_grad`/`grad`/`backward`, no Tensor integration, and no
Python operator overloads yet (`__add__`, `__matmul__`, ... are
deliberately absent — compute is method-only for now). View ops
(`reshape`, `transpose`, `T`, `narrow`) are still to come in v1.10.

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

The contiguous elementwise fast path is now **implemented** (v1.14) and
its impact is **measured and reported** (v1.15, above): on a local run
the contiguous elementwise rows moved to roughly raw-buffer speed while
the strided-view rows stayed on the retained odometer, and `NativeTensor`
tracked `NativeTensorCore` throughout. The next step is **v1.16 — a
native broadcasting design** (Phase A2): a design-first milestone for
shape-aligned elementwise ops, the same design-then-implement cadence the
fast path followed. After that, the rest of Phase A continues —
reductions (A3) and dtype/device metadata (A4). Still no Tensor
integration. CUDA experiments remain a separate future branch. The Python
framework stays the reference implementation throughout.
