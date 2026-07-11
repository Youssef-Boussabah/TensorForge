# Backend experiments

This page tracks the advanced experimental line that started after the
v3.0 Python release. **Nothing here is part of the finished Python
framework** — `import tensorforge` never touches it, Tensor and
autograd are unchanged, and every existing API works exactly as
before.

## C++ backend — v1.21 (current)

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
                          # dtype='float64', device='cpu', the supported
                          # dtype/device sets, and the (false) tensor/
                          # autograd integration flags
cpp.normalize_dtype("float64")  # validated tag; "float32" -> ValueError
cpp.normalize_device("cpu")     # validated tag; "cuda"   -> ValueError
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

- float64 / CPU only. As of v1.21 this is **explicit, inspectable
  metadata** (`dtype`/`device` on the storage, core, and wrapper) rather
  than an unstated assumption, and unsupported dtype/device values are
  rejected at construction — but only `"float64"`/`"cpu"` exist, and
  other inputs are still converted to float64.
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

### Native training stack — NativeMSELoss (v3.6)

v3.6 adds the first native loss:
`tensorforge.experimental.NativeMSELoss`, a **parameter-free**
`NativeModule` computing mean squared error as a pure composition of
existing differentiable operations — no new C++ work, no fused loss
kernel, no manual backward, and no NumPy in the analytical forward or
backward (tripwire-tested). Deliberately **not** shipped: other losses,
optimizers, `NativeSGD`, parameter updates, mutation-version counters,
training loops, or serialization. `NativeTensor`, `NativeModule`, the
existing layers, and the stable framework's `mse_loss` are all
untouched.

```python
from tensorforge.experimental import NativeMSELoss

loss_fn = NativeMSELoss()                 # reduction="mean" (default)
loss_fn = NativeMSELoss(reduction="sum")  # the only other option
loss = loss_fn(prediction, target)        # scalar NativeTensor, shape ()
loss.backward()                           # existing autograd end to end
```

**Forward is exactly** `difference = prediction.subtract(target)`;
`squared = difference.multiply(difference)`; then `squared.mean()` or
`squared.sum()`. Both reductions are **scalar**, so the existing default
backward seed applies (an explicit scalar upstream gradient scales both
input gradients per the engine's normal rules). The analytical gradients
— `dL/dprediction = 2(prediction−target)/N` and its negation for the
target under `mean` (drop `/N` under `sum`) — are supplied entirely by
the existing graph: `multiply`'s **duplicate-parent accumulation** on
the shared difference node produces the factor 2, `subtract`'s backward
produces the target's sign, and **`mean`'s existing native backward
produces the `1/N` scaling** — no division operation exists or is
needed. Verified by exact references (1-D, multidimensional
total-element scaling, zero-difference, positive/negative differences,
explicit upstream scaling, one-sided and both-frozen operands,
branching, duplicate-parent identity in the graph) **and central finite
differences** for prediction and target under both reductions
(`eps=1e-6`, `atol=1e-6`).

**Reduction contract** (deliberately small): exactly `"mean"` and
`"sum"`, validated in the constructor by exact string match — case
variants, whitespace variants, non-strings, and unsupported strings are
rejected, nothing is normalized — and stored as constructor
configuration, never model state. No `"none"` in this milestone: both
supported reductions are scalar, which is sufficient for the first
native training loop; unreduced losses belong to a later, broader loss
API.

**Input contract**: two open `NativeTensor`s (a `NativeParameter` is
accepted as the subclass it is and accumulates native-backed gradients;
the stable framework's `Tensor`, NumPy arrays, lists, scalars, and
closed tensors are rejected with errors naming *which* argument is
invalid) of **exactly equal shape — no broadcasting** (checked before
any graph node is built, even though `subtract` could broadcast; a
silently broadcast loss hides target-shape bugs; the error names both
shapes), with exact dtype/device equality via the metadata contract.
Shape-generic across every supported rank; zero-element tensors cannot
be constructed by the native runtime (`NativeStorage` requires positive
size), a limitation NativeMSELoss simply inherits. Inputs are never
mutated, reshaped, cast, or copied; the module stores no temporary
tensors and owns no storage; `state_dict()` is empty (unexpected keys
follow the v3.3 strict/non-strict rules; `reduction` never appears);
`training` propagates normally and never affects numerics; graph
lifetime (one-shot, `retain_graph`, freed-history errors, unchanged
gradients after a failed reuse) holds through the loss.

**Model integration** is verified end to end against an exact reference:
`NativeSequential(NativeLinear → NativeReLU → NativeLinear)` +
`NativeMSELoss` produces the correct scalar loss and exact gradients for
the input, every weight and bias (ReLU masking included), and a
gradient-requiring target; recursive `model.zero_grad()` clears the
model while the target's gradient stays independent; repeated fresh
forward → loss → backward → `zero_grad` cycles reproduce bit-identical
gradients; frozen models leave a trainable input learning. The
v3.3–v3.5 **mutation boundary is unchanged**: parameter mutation or
state loading between forward and backward remains mathematically
inconsistent (no version counters yet — exactly what v3.7 addresses).

Everything above is locked by `tests/test_native_mse_loss.py`
(selector: `-k "native_mse_loss"` — 27 tests), and the full suite
passes at **1196 tests**. Still float64/cpu only, still explicit and
experimental. The next milestone is **Advanced C++ v3.7 — Native
Parameter Mutation Safety and Versioning Contract**, narrowly scoped
to: version counters on mutable native parameter values, forward-time
expected-version capture where backward needs saved parameter values,
state loading incrementing parameter versions, clear stale-forward
backward errors, a controlled no-grad parameter mutation primitive, the
identity-preserving update foundation for `NativeSGD`, and rollback and
shared-parameter behavior — **no optimizer and no training loop yet**.
v3.7 must precede `NativeSGD` because optimizer updates cannot safely
mutate parameter values while old graphs remain capable of backward;
mutation versioning is not combined with SGD.

### Native training stack — NativeReLU and NativeSequential (v3.5)

v3.5 completes the first composable native model surface with one
parameter-free activation module and one ordered composition container:
`tensorforge.experimental.NativeReLU` and
`tensorforge.experimental.NativeSequential`. Both subclass
`NativeModule` and compose only existing APIs — no new C++ work, no
fused kernels, no manual backward anywhere, and no NumPy in forward or
backward (tripwire-tested). Deliberately **not** shipped: losses,
optimizers, training loops, serialization, mutation-version counters,
other activations, or other layers. `NativeModule`, `NativeTensor`,
`NativeParameter`, `NativeLinear`, and the stable framework are all
untouched.

```python
from tensorforge.experimental import (
    NativeLinear, NativeReLU, NativeSequential, NativeTensor,
)

model = NativeSequential(
    NativeLinear(3, 4, seed=0),
    NativeReLU(),
    NativeLinear(4, 2, seed=1),
)
out = model(x)                     # (batch, 3) -> (batch, 2)
out.sum().backward()               # existing autograd end to end
model.zero_grad(); model.eval()    # inherited recursive contracts
model.state_dict()                 # {"0.weight", "0.bias", "2.weight", "2.bias"}
```

**NativeReLU** is a parameter-free module whose `forward(input)`
validates an open `NativeTensor` (framework `Tensor`, arrays, lists,
scalars, and closed tensors rejected; `NativeParameter` accepted as the
subclass it is, returning a plain `NativeTensor`) and delegates to the
existing `NativeTensor.relu()` — **shape-generic** (every rank and
strided/offset layout `relu()` supports), no in-place mode or `inplace`
argument, no copies or reshapes, dtype/device preserved. Its backward is
entirely the existing fused native relu autograd, including the existing
gradient-at-exactly-zero rule (blocked — tested, not changed); graph
lifetime is untouched; `state_dict()` is empty; `training` propagates
normally but never affects numerics.

**NativeSequential(*modules)** chains children registered under
**contiguous integer-string slots** `"0"`..`"len-1"` — execution order
*is* the registered order (a single source of truth; no separate child
list). The constructor accepts zero or more `NativeModule`s (nested
sequences and custom subclasses included), validates **all** entries
before registering any, and rejects tensors, parameters, stable-framework
modules, callables, `None`, and lists. The container surface is
deliberately minimal: `len()`, iteration in execution order (shared
modules **not** deduplicated — iteration mirrors execution),
`seq[i]` with real ints (bool rejected, Python-style negatives,
IndexError out of range, exact object returned), `seq[i] = module`
(replacement keeps the slot's name and position), and `append(module)`
(next contiguous slot; returns `self` for chaining). The slot invariant
is enforced at the registration funnel, so **the registry and the
execution order can never silently diverge**: gap-producing indices,
non-canonical digit strings (`"01"`), non-slot child names (which would
register a child that never executes), direct `NativeParameter`
assignment (a sequence composes modules — parameters live in children),
slot removal in any form (`None` assignment, ordinary-value overwrite,
`del`, `add_module(name, None)`), and direct self-insertion are all
rejected with clear errors; ordinary non-module attributes stay normal
attributes. Module traversal remains cycle-safe by v3.2, but *executing*
a deliberately cyclic composition is meaningless and unsupported.

**Shared modules — the load-bearing distinction: execution is
position-based, ownership is identity-based.** The same child in two
slots executes twice in `forward`, while `modules()` /
`named_modules()` / `named_parameters()` / `state_dict()` / `train()` /
`zero_grad()` keep the v3.2/v3.3 identity-deduplicated contracts — the
first slot is the canonical path, shared parameters appear once, and a
duplicate alias key in a loaded state is *unexpected* under the strict
rules.

**Forward is pure composition**: each child is called through its normal
`__call__`/`forward` contract and validates its own input (a
`NativeLinear` slot enforces its strictly 2-D contract when reached;
child exceptions propagate unchanged); the container adds no node, copy,
or graph machinery of its own, and an **empty sequence returns the input
by identity**. The composed graph is exactly the children's own autograd
graphs, so one-shot cleanup, `retain_graph`, freed-history errors, and
the v3.3/v3.4 mutation boundary (forward → backward → zero_grad/state
update after graph completion) hold end to end — verified through a
Linear→ReLU→Linear model with exact analytical references **and central
finite differences** for the input, both weights, and both biases
(`eps=1e-6`, `atol=1e-6`, all hidden pre-activations kept ≥ 0.1 from
ReLU's zero boundary). State keys derive from slot names
(`"0.weight"`, `"0.bias"`, `"2.weight"`, `"2.bias"`; nested sequences
nest: `"0.0.weight"`; ReLU contributes none; bias-free layers omit bias
keys), and loading preserves every v3.3 guarantee.

Everything above is locked by `tests/test_native_relu.py` and
`tests/test_native_sequential.py` (selector:
`-k "native_relu or native_sequential"` — 52 tests), and the full suite
passes at **1169 tests**. Still float64/cpu only, still explicit and
experimental. The next milestone is **Advanced C++ v3.6 —
NativeMSELoss**, narrowly scoped to: an MSE loss as a `NativeModule`
with prediction/target `NativeTensor` validation, an exact-shape
contract initially, the smallest justified reduction surface (scalar
mean by default), forward composed from the existing native
`subtract`/`multiply`/`sum`/`mean`, exact and finite-difference
gradient tests — **no optimizer or training loop yet** (NativeMSELoss
is not combined with SGD or model training).

### Native training stack — NativeLinear (v3.4)

v3.4 adds the first concrete native layer:
`tensorforge.experimental.NativeLinear`, a fully connected layer built
entirely on the completed contracts — `NativeModule` (v3.2) holding
`NativeParameter`s (v3.1), forward as pure existing `NativeTensor`
operations, backward supplied by the existing Phase B autograd, and
state handled by the v3.3 state dictionary. It deliberately ships **no**
`NativeSequential`, activation modules, losses, optimizers, training
loops, mutation-version counters, or serialization. No C++ changed, no
kernel or symbol was added, `NativeTensor`/`NativeParameter`/
`NativeModule` are untouched, and `tensorforge.nn.Linear` is untouched.

```python
from tensorforge.experimental import NativeLinear, NativeTensor

layer = NativeLinear(in_features, out_features, bias=True,
                     *, seed=None, requires_grad=True)
output = layer(x)   # x: 2-D NativeTensor (batch_size, in_features)
```

**Weight orientation** (load-bearing for future checkpoints): `weight`
is `(in_features, out_features)` — the same `x @ weight` orientation as
the stable framework's Linear — so the strictly 2-D native matmul
applies directly; `bias` is `(out_features,)`, broadcast over the batch
by the existing zero-stride broadcast. The forward is exactly::

    output = input.matmul(weight)            # bias=False
    output = input.matmul(weight).add(bias)  # bias=True

Because forward is a composition of existing differentiable operations,
**the existing autograd engine is the backward implementation** — there
is no manual or fused NativeLinear backward, and graph lifetime
(one-shot default, `retain_graph=True`, freed-history errors) is
unchanged. Gradient shapes: `input.grad` `(batch, in)`, `weight.grad`
`(in, out)`, `bias.grad` `(out,)` (the broadcast-add backward reduces
over the batch via the native `unbroadcast`). Every gradient is verified
against exact analytical formulas **and central finite differences**
(`eps=1e-6`, float64-appropriate tolerances; NumPy is the test-side
reference only).

**Constructor.** Every Python argument is validated before any native
allocation: `in_features`/`out_features` are real positive ints (bools
and integer-like objects rejected), `bias` and `requires_grad` are real
bools, `seed` is `None` or a real int. `requires_grad=False` freezes
both parameters — they stay registered, traversable, and in
`state_dict()`, accumulate no gradients, and a requiring input still
receives its gradient. **Initialization** is deterministic and
self-contained: weight and bias sampled uniformly from
`[-1/sqrt(in_features), +1/sqrt(in_features)]` (a basic fan-in bound) by
a **local** `numpy.random.default_rng(seed)` — an int seed reproduces
values exactly, `None` draws fresh entropy, and the global NumPy RNG is
never read or mutated. NumPy appears only as host-side initialization
data preparation (the established `from_array` entry boundary) — never
in forward or backward computation (guarded by a tripwire test).
Parameters are created by assignment (`self.weight = NativeParameter(...)`,
then `self.bias = ...` or `None`), exercising v3.2 registration and
fixing the deterministic order `["weight", "bias"]` everywhere
(`named_parameters()`, `parameters()`, `state_dict()`; nested as
`"layer.weight"`/`"layer.bias"`). With `bias=False` the attribute reads
as `None` and only `"weight"` exists.

**Input contract** (strictly 2-D for now): an open `NativeTensor` of
shape `(batch_size, in_features)` with matching dtype/device
(float64/cpu). Nothing is wrapped, reshaped, flattened, or broadcast
implicitly — the stable framework's `Tensor`, NumPy arrays, lists,
scalars, closed inputs, and wrong ranks/feature sizes are rejected with
errors naming the expected 2-D shape, the expected feature count, and
the actual shape. The output is an ordinary `NativeTensor` (never a
parameter), requiring grad exactly when a participating operand does;
forward does not depend on `training` mode.

**State compatibility** follows v3.3 exactly: loading a compatible state
changes values while weight/bias identity, gradients, `requires_grad`,
and frozen status survive; loading a biased state into a bias-free layer
reports `"bias"` *unexpected*, the reverse reports it *missing* (strict
raises before mutation; non-strict returns the key lists);
shape-incompatible states fail atomically. **The v3.3 mutation boundary
applies unchanged**: the supported sequence is forward → backward →
(optionally) load/update after the graph completes; loading between
forward and backward is memory-safe but mathematically inconsistent (no
version counter — deliberately out of scope).

Everything above is locked by `tests/test_native_linear.py` (selector:
`-k "native_linear"` — 42 tests), and the full suite passes at **1117
tests**. Still float64/cpu only, still explicit and experimental, still
no implicit dispatch and no NumPy compute in any native forward,
backward, or state path. The next milestone is **Advanced C++ v3.5 —
NativeReLU and NativeSequential**, narrowly scoped to: a `NativeReLU`
module wrapping the existing `NativeTensor.relu()`, a `NativeSequential`
ordered child-module container with integer-string child names,
deterministic recursive traversal, forward composition, shared-module
behavior, train/eval propagation, and state_dict compatibility
(replacement/indexing only if tightly justified) — **no loss, optimizer,
or training loop yet** (v3.5 is not combined with losses, SGD, or model
training).

### Native training stack — native state dictionary contract (v3.3)

v3.3 adds the deterministic **in-memory** model-state contract:
`NativeModule.state_dict()` and `NativeModule.load_state_dict()`. The
scope is deliberately narrow — **parameters only, in memory only**: no
buffers, optimizer state, training/RNG state, file formats, archives, or
checkpoint metadata (file serialization and checkpoints are later
milestones that will consume this contract). No C++ changed;
`NativeTensor` is untouched; `tensorforge.nn` is untouched.

```python
state = model.state_dict()      # {canonical_name: independent NativeTensor}
result = model.load_state_dict(state)          # strict=True by default
result.missing_keys, result.unexpected_keys    # immutable, deterministic
model.load_state_dict(partial_state, strict=False)
```

**`state_dict()`** returns an insertion-ordered `dict[str, NativeTensor]`
whose keys are exactly the v3.2 canonical `named_parameters()` names:
dot-separated, direct parameters before descendants, **shared parameters
once under their first-discovered path**, frozen parameters included,
cycle-safe, deterministic across calls (an empty module returns an empty
mapping). Every value is an ordinary **graph-free, `requires_grad=False`
NativeTensor holding an independent owning contiguous copy**, computed by
the native copy path (`zeros` + native `add` — no NumPy computes or
copies state values). Snapshot and model share no mutable native storage
in either direction: mutating, replacing, or closing a model parameter
never affects an existing snapshot, closing a snapshot never affects the
model, and a snapshot outlives the model if the caller keeps it (the
caller releases snapshot values with `close()` when done). Gradients,
`requires_grad`, training flags, aliases, and registrations are neither
included nor touched. A closed registered parameter makes `state_dict()`
raise clearly (naming the key) — snapshots half-built inside the failed
call are closed before the error propagates, never returned, and earlier
snapshots stay valid.

**`load_state_dict(state_dict, strict=True)`** copies values **into the
existing `NativeParameter` objects** — never assigning new objects — so
every identity-derived contract survives loading: `id(parameter)`,
registration, canonical names and traversal order, shared-parameter
aliasing (one canonical key updates the single shared object once and
every alias observes it; a supplied alias key is *unexpected*),
`requires_grad`/frozen state, leaf/graph-free status, and each existing
`grad` **by identity and value** (`None` stays `None`; loading never
clears, replaces, or accumulates gradients — the exact-shape rule keeps
an existing gradient compatible). `training` flags are untouched. It
returns an immutable `LoadStateDictResult(missing_keys, unexpected_keys)`
(missing in canonical order, unexpected in input order; both empty on a
strict success).

Validation happens **entirely before mutation**, in a documented order:
`strict` must be a real bool (TypeError otherwise); the input must be a
mapping (snapshotted once, so exotic mapping subclasses cannot shift
mid-load); keys must be strings; missing/unexpected keys are computed
and, under `strict=True`, any incompatibility raises one ValueError
reporting **both** lists; then every matching value is preflighted — it
must be an open `NativeTensor` (a `NativeParameter` is accepted purely as
a value source and copied, inheriting no identity or graph state;
`tensorforge.Tensor`/`Parameter`, arrays, lists, and scalars are rejected
— nothing is converted silently) whose shape/dtype/device **exactly**
match the open destination parameter, with every error naming the key.
There is no broadcasting, reshaping, casting, truncation, partial
copying, or device movement. **Atomicity**: independent native copies of
all matching values are *staged* before anything changes (a staging
failure closes the staged copies and changes nothing); the *commit* then
swaps each parameter's core for its staged copy — pure reference
assignments guarded by a rollback that restores every original core if
anything interrupts — and only after every swap succeeds are the replaced
cores released, exactly once. No failure at any stage leaves the model
partially updated, closes an input tensor, or invalidates existing
snapshots. Under `strict=False`, matching keys load with the same full
validation and atomicity, missing parameters keep their values, and
unexpected keys are ignored.

The value-replacement primitive is a narrowly scoped **internal**
`NativeParameter._adopt_value_core(new_core)` — it swaps the owned core
(defensively re-validating metadata) and returns the old core for the
caller to release exactly once or restore on rollback, changing nothing
else. It exists for controlled state-dict loading and is **not yet the
optimizer update API**. This is the framework's first in-place value
mutation, so the graph policy is explicit: a graph built *before* loading
stays **memory-safe** — a later backward through it reads the parameter's
current (newly loaded) value through the normal open-core path; new
forward graphs simply use the loaded values.

Everything above is locked by `tests/test_native_state_dict.py`
(selector: `-k "native_state_dict or load_state_dict"` — 54 new tests),
and the full suite passes at **1075 tests**. Still `float64`/`cpu` only,
still explicit and experimental, still no implicit dispatch and no NumPy
compute in any native numerical, gradient, or state-copy path. The next
milestone is **Advanced C++ v3.4 — NativeLinear**, narrowly scoped to:
a `NativeLinear` built on `NativeModule` with a `NativeParameter` weight
and optional `NativeParameter` bias, deterministic initialization, input
validation, strictly 2-D forward semantics initially (native `matmul`
plus broadcast `add`), parameter registration through assignment,
forward/backward and finite-difference tests, and state_dict
compatibility — **no optimizer or training loop yet** (`NativeLinear` is
not combined with `NativeSequential`, activations, losses, optimizers, or
training).

### Native training stack — NativeModule core and recursive registration (v3.2)

v3.2 adds the module hierarchy: `NativeModule`, the base every future
native layer, state_dict, optimizer, and training loop will build on. It
is a **Python-side organizational abstraction** — it performs no numerical
computation, owns no native storage, and never closes, copies, or mutates
what it registers. It deliberately ships **no** `NativeLinear`,
`NativeSequential`, activations, losses, optimizers, state_dict,
serialization, buffers, hooks, or training loop. No C++ changed,
`NativeTensor` is untouched, and `tensorforge.nn.Module`/`Parameter` are
untouched; the one v3.1 adjustment is a minimal read-only
`NativeParameterRegistry` extension (`get`/`__contains__` plus a shared
name-validation helper) with all v3.1 behavior preserved.

```python
from tensorforge.experimental import NativeModule, NativeParameter

class Block(NativeModule):
    def __init__(self):
        super().__init__()          # required before assigning parameters
        self.weight = NativeParameter([[1.0, 2.0], [3.0, 4.0]])

root = NativeModule()
root.block = Block()                # child registration by assignment
root.bias = NativeParameter([0.0, 0.0])

list(root.named_parameters())       # [("bias", ...), ("block.weight", ...)]
root.parameters()                   # unique parameters, identity-deduplicated
list(root.named_modules())          # [("", root), ("block", ...)]
root.zero_grad()                    # each unique parameter's grad -> None
root.eval().training                # False, propagated recursively
```

**Registration is assignment** (`__setattr__`), with registered objects
living only in the module's registries (`__getattr__` resolves them — one
source of truth): a `NativeParameter` value registers a parameter, a
`NativeModule` value registers a child, and **everything else is an
ordinary attribute** — a plain `NativeTensor`, a
`tensorforge.Tensor`/`Parameter`/`nn.Module`, a string — which never
enters native traversal (nothing is wrapped implicitly; stable-framework
objects stay harmless ordinary attributes). **One category per name; the
latest assignment wins**: registering validates first (a failure mutates
nothing), then evicts the name from the other categories. Replacement
within a registry preserves the slot position; moving a name between
registries appends to the target; `module.name = None` (and `del
module.name`) unregisters, leaving the attribute readable as `None`, and
re-registering a removed name appends — the v3.1 ordering rules
throughout. Evicted or replaced objects are dropped, never closed or
mutated, and no gradient state transfers. `register_parameter(name, p)` /
`add_module(name, m)` are the explicit forms with identical semantics
(their one deliberate strictness: a non-parameter/non-module value raises
`TypeError`, and `None` raises `KeyError` when nothing is registered under
the name). Names follow the v3.1 rule — non-empty dot-free strings (dots
reserved for hierarchical state_dict keys) — and `"_parameters"` /
`"_modules"` / `"training"` are reserved implementation slots that can
never be parameter or child names; `__init__` creates the registries via
`object.__setattr__` so initialization never routes through registration,
and registering before `NativeModule.__init__()` has run raises a clear
`RuntimeError`.

**Traversal is deterministic pre-order depth-first, deduplicated by
object identity** (`id`-keyed — never value equality), and **first
discovery wins**. `named_modules()` yields `("", self)` first, then each
child under its registered name and its descendants under dot-joined
paths, in insertion order; shared modules appear once under their first
discovered path, which also makes **direct and indirect module cycles
terminate safely** (cycles are allowed as references; traversal never
revisits a module). `named_parameters(prefix="", recurse=True)` walks
that order and yields each unique parameter once under its
first-discovered dotted name — a module's direct parameters before its
descendants', direct aliases and shared parameters (direct or through
children) deduplicated by identity, frozen parameters included;
`recurse=False` restricts to direct parameters. `parameters()` /
`modules()` return the matching identity-deduplicated lists — exactly
what a future optimizer iterates, and the first-discovered dotted names
are exactly the canonical keys v3.3's state_dict will use (loading will
copy values into existing parameters, preserving identity).

**`zero_grad()`** visits each unique parameter once (shared parameters
included) and calls its existing `zero_grad()` — grad → `None`, data /
`requires_grad` / graphs untouched, nothing closed — and returns `None`.
**`train(mode=True)`** validates `mode` as a real bool *before* touching
any state (non-bool raises `TypeError` with no partial mutation), then
sets `training` on every unique module — shared and cyclic hierarchies
visited once — and returns `self`; `eval()` is `train(False)`; every
module starts with `training = True` (a later mode-dependent layer will
read the flag; none exists yet). **`forward()` raises
`NotImplementedError` and calling the module delegates to `forward`** —
the whole call protocol, with no hooks or tracing machinery.

Lifetime: the registries store Python references only. Removing,
replacing, or deleting a registration — or dropping the module itself —
never invalidates an object another reference still holds; native storage
is released only by the owner's explicit `close()`, and there is no
`NativeModule.close()`.

Everything above is locked by `tests/test_native_module.py` (selector:
`-k "native_module or recursive_registration"` — 49 tests), and the full
suite passes at **1021 tests**. Still `float64`/`cpu` only, still explicit
and experimental, still no implicit dispatch and no NumPy in any native
numerical or gradient path. The next milestone is **Advanced C++ v3.3 —
Native State Dictionary Contract**, narrowly scoped to: `state_dict()`,
`load_state_dict()`, deterministic hierarchical names, strict
missing/unexpected-key checks, shape/dtype/device validation, value
copying that never replaces `NativeParameter` identity, and
shared-parameter canonical naming — **no file serialization and no
optimizer state yet** (state_dict, `NativeLinear`, serialization, and
checkpointing are not combined into one milestone).

### Native training stack — NativeParameter and registration contract (v3.1)

v3.1 opens **Phase C — the native training stack** — with its foundation:
`NativeParameter`, the trainable-leaf abstraction, and
`NativeParameterRegistry`, the minimal parameter-registration contract the
future `NativeModule` (v3.2) will embed. It deliberately ships **no**
module hierarchy, layer, loss, optimizer, training loop, state_dict, or
serialization — it settles the identity, ownership, leaf, and registration
rules everything later depends on. No C++ changed, no kernel or symbol was
added, `NativeTensor` itself was not modified, and `tensorforge.Tensor` /
`tensorforge.nn.Parameter` are untouched.

```python
from tensorforge.experimental import (
    NativeParameter, NativeParameterRegistry, NativeTensor,
)

w = NativeParameter([[1.0, 2.0], [3.0, 4.0]])   # leaf, requires_grad=True
b = NativeParameter([0.0, 0.0], requires_grad=False)  # frozen, still a parameter

x = NativeTensor.from_array([[1.0, 0.0], [0.0, 1.0]])
loss = x.matmul(w).sum()     # ordinary NativeTensor results — never parameters
loss.backward()              # w.grad is a NativeTensor matching w
w.zero_grad()                # grad back to None; data untouched

registry = NativeParameterRegistry()
registry.register("weight", w)
registry.register("bias", b)
registry.named_parameters()  # insertion-ordered (name, parameter) pairs
registry.parameters()        # unique parameters, deduplicated by identity
```

**`NativeParameter` is a `NativeTensor` subclass** (`__slots__ = ()`), not a
wrapper — the native ops require `NativeTensor` operands and the graph
engine reads leaf metadata directly, so composition would need a parallel
delegation surface for no gain. The one subclassing hazard — every op
builds results through the `_from_core`/`_from_op` classmethods, where
`cls` would be `NativeParameter` — is closed by overriding both to delegate
explicitly to `NativeTensor`: **parameter-ness never propagates**. Math,
views (`reshape`/`transpose`/`T`/`narrow`), `contiguous_copy`, `sum`/`mean`,
and `detach()` all return plain `NativeTensor` (detach additionally
graph-free with `requires_grad=False`); the only way to create a parameter
is calling `NativeParameter(...)` itself (the inherited
`from_array`/`zeros`/`full` classmethods therefore also return plain
tensors — they are not parameter constructors). Every parameter is a
**graph-free owning leaf for its whole life**: no `_parents`, no
`_backward`, never marked graph-freed, and participating in an operation
makes the *result* a non-leaf, never the parameter.

**Construction always takes an independent owning contiguous copy.**
From array-like data the values are copied into fresh float64/cpu native
storage; from an existing `NativeTensor` — leaf or non-leaf, contiguous or
strided/offset/borrowing view — the parameter copies the source's *current
value* (`contiguous_copy`) and inherits none of its graph history. Closing
the source never invalidates the parameter, closing the parameter never
invalidates the source, a one-shot backward through the source's graph
neither reaches nor frees the parameter, and a closed source is rejected
with the usual `RuntimeError`. No storage is ever shared, so a future
optimizer update can never mutate an unrelated tensor through a hidden
alias. `requires_grad` is a validated **real bool** (default `True`;
`requires_grad=False` builds a frozen parameter that stays registerable
and discoverable but accumulates no gradient — there is no broad
freeze/unfreeze API). Gradients follow the Phase B rules unchanged:
`.grad` starts `None`, accumulates `NativeTensor`-backed native gradients
matching the parameter's shape/dtype/device across fresh training-style
graphs, and `zero_grad()` clears to `None` without touching data.
`close()` follows the `NativeTensor` lifetime rules (idempotent; a closed
parameter rejects data and gradient operations).

**Identity, not value.** No `__eq__`/`__hash__` is defined anywhere in the
hierarchy: two equal-valued parameters are distinct parameters, comparison
never runs tensor math, and future optimizers can key state by object
identity (`id`-keyed, the same convention `backward()`'s traversal already
uses). Future state_dict loading is expected to copy values *into*
existing parameters, preserving identity — never to replace registered
objects.

**`NativeParameterRegistry`** is an insertion-ordered name → parameter
registry holding plain Python references — it owns no storage and never
closes, copies, or mutates a parameter, never touches `requires_grad` or
gradients, and dropping the registry leaves every parameter open. Its
contract:

- **Names** are non-empty `str` without `"."` (dots are reserved for the
  future hierarchical state_dict keys, matching the Python framework's
  dotted paths); invalid names raise `TypeError`/`ValueError` — nothing is
  silently stringified.
- **Values** are `NativeParameter` only, or `None` to **unregister**
  (`KeyError` if the name is not registered). An ordinary `NativeTensor`
  and the Python framework's `Tensor`/`Parameter` are rejected with a
  clear `TypeError` — never wrapped implicitly (the same rejection the
  `NativeParameter` constructor applies to framework objects, checked
  lazily so the native backend still never imports the frontend).
- **Ordering** is insertion order. **Replacing** a registered name keeps
  its position and simply drops the reference to the previous parameter —
  the old object is not closed or mutated and no gradient state transfers.
  **Unregistering** deletes the slot, so registering that name again
  appends at the end (the documented rule).
- **Aliases**: the same parameter may be registered under several names
  (the future shared-weight case). `named_parameters()` shows every alias;
  `parameters()` deduplicates by object identity in first-registration
  order (each unique parameter exactly once — the traversal an optimizer
  iterates); `unique_named_parameters()` is the deduplicated named view
  where the first-registered name wins. Equal-valued distinct parameters
  are never deduplicated.

Everything above is locked by `tests/test_native_parameter.py` (selector:
`-k "native_parameter or parameter_registration"` — 49 tests), and the
full suite passes at **972 tests**. Still `float64`/`cpu` only, still
explicit and experimental, still no implicit dispatch and no NumPy in any
native numerical or gradient path. The next milestone is **Advanced C++
v3.2 — NativeModule Core and Recursive Registration**: child-module
registration, automatic parameter/module assignment registration,
recursive `parameters()` / `named_parameters()` / `modules()` /
`named_modules()`, `zero_grad()`, deterministic traversal,
shared-parameter and shared-module handling, and the train/eval state
foundation — no layers or optimizers yet.

### Native autograd — Phase B guardrails and completion (v2.6)

v2.6 is the **completion** milestone for Phase B — native autograd. It adds
**no** operation, kernel, optimizer, training abstraction, or performance
optimization, and changes **no** autograd behavior; it audits, locks down,
documents, and formally completes the engine. It is almost entirely tests
and documentation (the one source touch is a stale docstring correction —
`tensorforge.experimental`'s package docstring no longer claims "no
autograd", which has been false since v2.1). No C++ changed, no kernel or
symbol was added, and no NumPy entered the gradient path.

The new cross-cutting guardrails live in
`tests/test_native_autograd_guardrails.py` (selector:
`-k "phase_b_guardrail or native_autograd_guardrail or native_backend_isolation"`)
and lock several completed invariants together rather than duplicating the
per-operation tests:

- **No NumPy in the gradient path.** A runtime guard (a context manager)
  replaces NumPy's *numerical* functions (`add`/`multiply`/`matmul`/`sum`/
  `mean`/`maximum`/`where`/`broadcast_to`/… — the ones a NumPy-backed
  backward would call) with tripwires that raise, while leaving the
  marshalling helpers (`asarray`/`empty`) intact. The graph is built
  *before* the guard and gradients are read *after* it, so a correct native
  pass completes cleanly and only a smuggled-in NumPy computation would
  trip a wire. It covers same-shape elementwise, genuine broadcasting,
  reduction, matmul, and a transpose→narrow→contiguous_copy→reshape view
  chain, and a self-check proves the guard actually bites.
- **`NativeTensor` ↔ `tensorforge.Tensor` isolation.** Native ops return
  `NativeTensor`; native grads are `NativeTensor`-backed; `Tensor` stays
  NumPy-backed; neither engine's backward touches the other; mixed operands
  raise clearly (both directions) instead of dispatching implicitly.
- **Explicit backend / no implicit dispatch.** The wrapper is reached only
  through `tensorforge.experimental`; a static check confirms `import
  tensorforge` imports neither `experimental` nor `backends`; native
  unavailability raises the build-instructions `ImportError` (no silent
  NumPy fallback); there is no automatic backend selection.
- **Gradient ownership, graph lifetime, detach, view+offset, and
  closed-operand safety** are each locked over realistic mixed graphs
  (shared intermediate + broadcast + view), including the deterministic
  freed-graph error and snapshot-based failure rollback.
- **Kernel-registry boundary.** `list_kernels()` and
  `tensor_core_kernels` must not leak the internal fused backward kernels
  (`tf_core_relu_backward`, `tf_core_narrow_backward`, `tf_core_sum`,
  `tf_storage_scale`) — they stay forward-shaped numerical methods only.
- **Benchmark mode contract.** The v2.5 modes keep their documented
  meanings: `forward_native` builds no graph, the grad modes build one,
  `forward_backward_fresh` builds and frees a fresh graph, and
  `backward_retained` reuses one fixed graph.

The **final Phase B support matrix**, the **explicit divide-backward
decision** (deferred beyond Phase B — Phase B is complete without it,
because the completed op set already spans a first native training stack;
see §18 of the design), and the **Phase B-complete / Phase C-next** status
are recorded in
[native_autograd_design.md](native_autograd_design.md) (§17–§19). Phase B is
**complete** at **923 tests** (893 baseline + 30 guardrails); the next
milestone is **Advanced C++ v3.1 — NativeParameter and Parameter
Registration Contract** (Phase C — the native training stack). No divide
kernel/API, no operator overloads, no CUDA, no dtype promotion, and no
implicit dispatch came with this milestone.

### Native autograd — benchmark characterization (v2.5)

v2.5 is a **measurement and documentation** milestone: it changes no
autograd behavior, adds no kernel, and makes no cross-framework claim. It
adds a reproducible harness,
[`benchmarks/benchmark_native_autograd.py`](../benchmarks/benchmark_native_autograd.py),
that characterizes where time goes in the native autograd stack, and one
honest hardware-specific snapshot. Full write-up (cases, modes, methodology,
results, and interpretation limits) is in
[native_autograd_benchmarks.md](native_autograd_benchmarks.md).

Five workloads — a same-shape elementwise chain, a genuine-broadcast chain,
a 3-D reduction chain, a 2-D matmul chain, and a transpose→narrow→
contiguous_copy→reshape view chain — are each run in four modes that
separate the layers as far as the architecture permits:
`forward_native` (grad off, no graph), `forward_graph` (graph built, no
backward), `forward_backward_fresh` (fresh graph + one-shot `backward()`,
including cleanup), and `backward_retained` (one graph built outside the
loop, `backward(retain_graph=True)` repeatedly — isolating repeated
backward, explicitly *not* a training-step estimate). Timing uses
`time.perf_counter_ns()`, configurable warmup/iterations/repeats, median
as the primary statistic with min/max spread, and a correctness gate
(output shape, finite output, and — for backward modes — that each leaf
gradient exists, has the right shape, and is finite) before any timing. A
CLI (`--case --mode --warmup --iterations --repeats --json --smoke`) runs
all cases/modes by default, rejects unknown cases/modes and non-positive
counts, and emits pure JSON (raw samples included) under `--json`.

On the recorded snapshot (one Windows/AMD64 machine, float64/cpu), adding a
backward pass is the dominant cost (fresh forward+backward ≈ 2.5×–5× the
forward), retained backward sits below fresh backward everywhere (it drops
the forward rebuild from the loop), graph-construction overhead is small
relative to compute at these sizes (clearest on matmul), and the smoke
shapes are dominated by wrapper/ctypes cost — all machine-specific
observations, with no speed assertions anywhere. The benchmark tests
(`tests/test_native_autograd_benchmark.py`) validate schema and behavior
only, never speed.

### Native autograd — graph lifetime policy (v2.4)

v2.4 gives the native autograd graph an explicit, PyTorch-like
**lifetime**. `backward` now takes a flag — `backward(gradient=None,
retain_graph=False)` — and the default is **one-shot**: after a
successful pass, the operation graph of every traversed non-leaf node is
released.

```python
from tensorforge.experimental import NativeTensor

x = NativeTensor.from_array([1.0, 2.0, 3.0], requires_grad=True)
out = x.multiply(x).sum()
out.backward()          # computes dx = 2x, then frees the graph
out.backward()          # RuntimeError: graph already freed — use retain_graph=True

y = NativeTensor.from_array([1.0, 2.0, 3.0], requires_grad=True)
loss = y.multiply(y).sum()
loss.backward(retain_graph=True)   # keeps the graph
loss.backward(retain_graph=True)   # runs again; y.grad accumulates to 4y
```

Releasing a graph clears each traversed non-leaf node's `_parents` and
`_backward` closure (so nothing keeps the parents alive) and marks the
node **freed**. A later `backward()` that reaches a freed node raises a
clear `RuntimeError` naming `retain_graph=True` as the remedy — and
crucially it does *not* silently treat the freed node as a leaf, which
would truncate history. That covers three cases with one flag: a repeated
backward on the same output, a **second output over a shared
intermediate** (`a = shared.sum(); b = shared.mean(); a.backward()` frees
`shared`, so `b.backward()` raises), and a **new op built from a freed
value** (`shared.add(y)` still computes forward — the stored value is
intact — but its backward refuses to cross the freed history). A genuine
leaf has no graph to free and is never marked freed, so repeated
`backward()` on a scalar leaf keeps accumulating.

`retain_graph` is validated as a real `bool` **first** — before traversal,
callbacks, cleanup, or any gradient mutation, and never coerced with
`bool(...)` — so a bad value (`"true"`, `1`, `None`, …) raises `TypeError`
and changes nothing. Whether or not the graph is freed, only leaves retain
grad; non-leaf grads stay transient and are dropped after each pass, so a
retained second pass recomputes them cleanly and leaf gradients accumulate
across passes until `zero_grad()` (which clears a leaf grad without
resurrecting a freed graph or damaging a retained one). The pass is
**failure-safe**: it is staged against a snapshot of every node's gradient
(gradients are immutable — accumulation replaces the reference with a
fresh native `add`), so if a callback raises mid-traversal — for instance
one branch commits a leaf and another hits a closed operand — the
references are restored, leaving no partial commit and no partial free, and
cleanup runs only after the pass fully succeeds. This is **not** full
PyTorch parity: there is no per-node `retain_grad` and no double-backward.
No C++ changed, no kernel was added, and no NumPy entered the gradient
path; `NativeTensorCore` still owns no graph state. Full status in
[native_autograd_design.md](native_autograd_design.md).

### Native autograd — narrow backward (v2.3)

v2.3 makes the last view op differentiable: `narrow(dim, start, length)`
now records a graph node when its parent requires grad. Its backward is a
**scatter** — the adjoint of a slice — placing the upstream gradient into
a fresh zeros tensor of the parent's shape at the narrowed region, so
every un-narrowed position gets zero gradient and the narrowed region
gets exactly the upstream.

```python
from tensorforge.experimental import NativeTensor

x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            requires_grad=True)
x.narrow(1, 0, 2).sum().backward()   # keep columns 0..1, sum, differentiate
x.grad.to_numpy()                    # [[1, 1, 0], [1, 1, 0]]
```

The scatter runs through **one new C++ kernel**, `tf_core_narrow_backward`
— the odometer dual of `tf_core_sum`. Where a sum walks the input and
folds many elements into one output cell through zero *write* strides, a
narrow-backward walks the (smaller) narrowed shape and writes each
upstream element into its own output cell: the write position advances by
the parent's full row-major strides from a base offset that skips the
leading `start` slabs along the narrowed dimension
(`start * row_major_stride[dim]`). The upstream is read through its own
shape/strides/offset, so a strided gradient works without materializing,
and the output is fresh **owning** row-major contiguous storage — zero
everywhere outside the window. It is surfaced as
`NativeTensorCore.narrow_backward(dim, start, original_shape)`, a
forward-shaped numerical method (the core and the kernels still own no
graph state); it is not added to `list_kernels()`/`TENSOR_CORE_KERNELS`,
matching the `tf_core_relu_backward`/`tf_core_sum` convention.

Because the gradient lives at the *logical* shape, the scatter always
allocates a fresh contiguous buffer of the parent's shape (offset 0)
regardless of the parent's own layout — so **transposed, narrowed, and
nonzero-offset parents** all work: each is simply a preceding graph node
whose own backward (transpose-inverse, another narrow-scatter, …) handles
its layout, and narrow backward only needs the immediate parent's shape.
Nested narrows, `narrow` under `sum`/`mean`/`multiply`, and `narrow`
feeding `transpose`/`reshape` (via `contiguous_copy`) all compose. The
lifetime rules are unchanged: the retained contribution owns its storage,
closing the parent or output before `backward()` raises a clear
`RuntimeError`, repeated backward accumulates until `zero_grad()`. Rules
are verified against an independent NumPy zero-padding reference and a
finite-difference check (NumPy test-side only), and the CI smoke script
hard-checks one narrow-backward pattern. This closes the view-backward
set; `retain_graph` is still not offered (v2.4), and there is still no
`divide` backward, no `tensorforge.Tensor` integration, no implicit
dispatch, no optimizer/training stack, no CUDA, and no performance
claims. Full status in
[native_autograd_design.md](native_autograd_design.md).

### Native autograd — core backward operations (v2.2)

v2.2 turns the v2.1 skeleton into a working reverse-mode engine: the
core `NativeTensor` operations are now **differentiable**. When an
operand requires grad, `add`, `subtract`, `multiply`, `relu`, `sum`,
`mean`, `matmul` (2-D), `reshape`, `transpose`/`T`, and
`contiguous_copy` record a graph node (parents + backward closure + op
name via the internal `_from_op`); when nothing requires grad, every op
returns a plain forward tensor exactly as before — no graph metadata is
created and forward-only use stays as cheap as it was.

```python
from tensorforge.experimental import NativeTensor

x = NativeTensor.from_array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]])
w = NativeTensor.from_array([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0]],
                            requires_grad=True)
b = NativeTensor.from_array([0.5, -0.5], requires_grad=True)

loss = x.matmul(w).add(b).relu().mean()   # (2,) bias broadcast over (2, 2)
loss.backward()                           # scalar seed defaults to 1.0
w.grad, b.grad                            # native float64/cpu NativeTensors
```

The backward rules are the design's (§7.4/§8/§9), computed **entirely by
native forward kernels at the `NativeTensorCore` level** — backward math
builds no graph nodes of its own and no NumPy ever touches a gradient:
`add`/`subtract` pass the upstream through (the right operand negated by
a broadcast-scalar multiply against a native `-1.0` — the §7.5
composition, no negate kernel); `multiply` computes `u·b` / `u·a`;
`matmul` computes `u @ b.T` / `a.T @ u` over strided transpose views;
`sum`/`mean` broadcast the upstream back to the input shape (reduced
axes reinserted as size 1 by a native reshape, expanded by the existing
zero-stride broadcasting via `zeros + u`, `mean` scaled by a native
`1/count` scalar multiply); `reshape`/`transpose` apply the inverse
reshape/permutation. Broadcasting backward is the new private
`_unbroadcast(grad, target_shape)` helper — the adjoint of broadcasting,
built from single-axis native reductions applied in a stable order
(leading padded axes summed away axis-0-first, stretched axes summed
with `keepdims=True`, a final native reshape for the `()`-versus-`(1,)`
rank family, which the helper distinguishes exactly). `contiguous_copy`
backward is the identity: the forward is an elementwise logical copy and
gradients live at the logical shape, so the parent's storage layout is
irrelevant. **One new C++ kernel** was added — `tf_core_relu_backward`
(`upstream` where `x > 0`, else `0`; `x == 0` blocks, the Python Tensor
convention) — exactly the fused kernel the design flagged, implemented
as one more op through the existing generic binary odometer walker, so
contiguous and strided inputs both work with no new traversal code. It
is surfaced as `NativeTensorCore.relu_backward`, a forward-shaped
numerical method: the core and the kernels still own **no graph state**.

Lifetime stays explicit and safe: every retained gradient contribution
is either the upstream tensor itself or fresh owning contiguous storage
(never a borrowing view over a transient), closing an operand or an
intermediate before `backward()` raises a clear `RuntimeError` (the
graph never reads freed storage), and with no in-place arithmetic the
forward values captured for backward stay valid for the life of the
graph. Only leaves retain grad; repeated `backward()` accumulates until
`zero_grad()`; `detach()` still cuts the graph. **`narrow` remained
outside autograd in v2.2** — its backward needs a native scatter
primitive (upstream scattered into zeros of the original shape), which
landed in v2.3 (above) rather than being faked through NumPy — and
`retain_graph` is still not offered. The rules are verified against exact
analytical values and
central finite differences (NumPy only as the test-side reference), the
deterministic `examples/native_autograd_demo.py` shows one full native
forward + backward (`demo()` returns NumPy copies for its tests), and
the CI smoke script hard-checks one scalar-loss backward. Still no
`tensorforge.Tensor` integration, no implicit dispatch, no optimizer or
training stack, no CUDA, and no performance claims. Full status in
[native_autograd_design.md](native_autograd_design.md).

### Native autograd — metadata skeleton (v2.1)

v2.1 implements the first piece of the v2.0 design: the **native autograd
metadata skeleton and the reverse-topological backward driver** on
`NativeTensor`. It is **Phase B's first code**, and it is deliberately a
skeleton — the forward compute ops are **not** wired into autograd yet
(that is v2.2). No kernel changed, `NativeTensorCore` and the C++ kernels
stay autograd-unaware, and `tensorforge.Tensor` is untouched.

`NativeTensor` now carries an opt-in, Python-managed autograd graph:

```python
from tensorforge.experimental import NativeTensor

x = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
x.requires_grad        # True
x.grad                 # None (lazily filled by backward)
x.is_leaf              # True (user-created)
d = x.detach()         # forward-only owning copy: requires_grad False, no graph
x.zero_grad()          # clears grad to None (idempotent)
```

The design's graph model landed as specified. New `NativeTensor` state
(`_requires_grad`, `_grad`, `_parents`, `_backward`, `_op`, `_is_leaf`)
lives **at the wrapper layer**; the raw runtime is unchanged. A single
private constructor, `NativeTensor._from_op(core, parents, backward, op)`,
builds a non-leaf node — `requires_grad` is the OR of the parents', and
the graph is recorded only when some parent needs grad (a plain forward
leaf otherwise, matching the Python `Tensor`). Constructors gained a
default-preserving `requires_grad=False` argument (rejecting non-`bool`),
so every existing call is unchanged. `backward(gradient=None)` mirrors the
Python engine natively: a post-order DFS over `_parents` keyed by **object
identity** (`id()`, never hashing, so duplicate/shared parents are visited
once), a seed (scalar output — `numel == 1` — defaults to a native `1.0`;
a non-scalar requires an explicit `NativeTensor` gradient matching the
output's shape/dtype/device, errors naming both), then a reverse walk
calling each non-leaf's backward closure. Gradients are **native**
(`NativeTensor`-backed), accumulated by the native `add` kernel
(`_accumulate_grad`) — **no NumPy touches the gradient path** — lazily
initialized, and obey the v1.21 contract (`grad.dtype`/`.device`/`.shape`
match the tensor). Only leaves retain grad; non-leaf grads are transient
and dropped after each pass. `zero_grad()` clears to `None` (the design's
choice), and grads are never eagerly closed, so a live `t.grad` is never
invalidated. `detach()` returns an **owning contiguous copy** (independent
lifetime, no NumPy round trip). `backward()` on a non-requiring or closed
tensor raises; `retain_graph` is intentionally not offered (the graph is
rebuilt each call, so repeated `backward()` accumulates until
`zero_grad()`). Full design and status in
[native_autograd_design.md](native_autograd_design.md); rigorous tests in
`tests/test_native_autograd.py`.

### Native autograd — design (v2.0)

v2.0 is **design-only**: no autograd is implemented, no kernels change,
and no runtime behavior changes. With **Phase A — the native CPU runtime —
complete in code** (the contiguous fast path v1.14, broadcasting v1.17,
reductions v1.19, and dtype/device metadata v1.21), this milestone opens
**Phase B** by writing the design for native reverse-mode autograd over
`NativeTensor` / `NativeTensorCore`.

Today the native runtime is **forward-only with no operation graph**:
`NativeTensorCore` ops return a fresh result that records no parents, no
operation, and no backward rule; `NativeTensor` has no `requires_grad`,
`grad`, or `backward`. The design specifies a **Python-managed graph at
the `NativeTensor` layer** — `NativeTensorCore` stays the raw forward
runtime and the C++ kernels never own graph state — where each
differentiable op records `core` + `requires_grad` + parents + a backward
closure + an op name, leaf tensors (user-created, `requires_grad=True`)
accumulate gradients, and `backward()` walks the graph in reverse
topological order (scalar outputs default their seed to `1`, non-scalar
outputs require an explicit gradient). Gradients are themselves
**native** (`NativeTensor`-backed, lazily initialized, accumulated by
native `add`) and honor the v1.21 metadata contract exactly
(`grad.dtype == tensor.dtype`, `grad.device == tensor.device`) — the
concrete reason A4 preceded autograd.

The design is honest about missing kernels: first-scope backward needs a
small fused `relu_backward` (no native compare/`where` exists), and it
notes negation/scalar-multiply, a core-level `divide`, and a
scatter/copy-into-view (for `narrow`/`contiguous_copy` backward) as
deferred additions rather than silently assuming them. Broadcasting
backward is an `unbroadcast(grad, original_shape)` helper built on native
**reductions (A3)** — the recorded dual: a broadcast forward read is a
sum-reduction on the backward pass. It stays **separate from
`tensorforge.Tensor`** (no conversion, no implicit dispatch, no silent
NumPy fallback, `Tensor` behavior unchanged) and **CPU float64 only** (no
CUDA autograd). The staged plan is **v2.1 metadata skeleton →
v2.2 basic backward (add/multiply/relu/sum) → v2.3 broadcasting + mean
backward → v2.4 matmul backward → v2.5 native autograd demo**, then Phase
C (native training stack). No code ships; the next milestone is **v2.1,
the native autograd metadata skeleton**. Full design in
[native_autograd_design.md](native_autograd_design.md).

### Native dtype/device metadata — implementation (v1.21)

v1.21 implements the metadata contract the v1.20 design specified —
**metadata only, float64/cpu only**, no kernel and no compute behavior
changed. Native tensors now carry explicit, inspectable `dtype` and
`device` you can read:

```python
t = NativeTensorCore.from_array([[1.0, 2.0], [3.0, 4.0]])
t.dtype, t.device            # ("float64", "cpu")
t.T.dtype                    # "float64" — a view shares its storage's tags
t.add(t).device             # "cpu"    — ops preserve dtype/device

NativeTensor.zeros((2, 3)).dtype          # "float64"
NativeTensorCore.zeros((2,), dtype="float32")  # ValueError — rejected
```

The tags are **owned by `NativeStorage`** (the allocation owner, so
every view over one buffer reports the same dtype/device — one source of
truth) and surfaced read-only through `NativeTensorCore.dtype`/`.device`
and `NativeTensor.dtype`/`.device`. Two pure helpers, `normalize_dtype`
and `normalize_device` (beside `broadcast_shapes`/`reduce_shape`, so they
are testable without the built backend), validate and canonicalize the
tags against `SUPPORTED_DTYPES == ("float64",)` and
`SUPPORTED_DEVICES == ("cpu",)`. Constructors (`from_array`/`zeros`/
`full` on both the core and the wrapper, and `tensor_from_array`/`zeros`/
`full` on the native backend) gained **default-preserving** `dtype`/
`device` arguments — `dtype=None`→`"float64"` on `from_array`,
`dtype="float64"` on `zeros`/`full`, `device="cpu"` throughout — so every
existing call is byte-for-byte unchanged and still produces a float64/cpu
tensor. `from_array` still coerces its data to float64 exactly as before;
the tag records intent, it does not change the bytes.

Following the design's **reject-over-inert** recommendation, any
unsupported dtype/device is rejected at construction (before native
memory is allocated), naming the offending value and the supported set —
so no tensor ever advertises a dtype/device the float64/CPU kernels
cannot actually compute. The binary ops and `matmul` validate that both
operands share dtype and device (naming both pairs on a mismatch) — a
guard that cannot fire yet with one legal value each, but is the enforced
contract native autograd (Phase B) will read `grad.dtype == param.dtype`
against. Every op and view carries the metadata through: `relu`, `add`/
`subtract`/`multiply` (broadcasting included), `matmul`, `sum`/`mean`,
and the metadata-only views all produce float64/cpu results; `to_numpy`
still returns float64 arrays. `backend_info()` now advertises the
supported sets (`supported_dtypes`, `supported_devices`, and a `device`
field) for both the native and NumPy backends.

The `dtype`/`device` fields follow each object's existing metadata
closed-behavior: readable after `close()` on `NativeStorage`/
`NativeTensorCore` (like `size`/`shape`), rejected after `close()` on the
`NativeTensor` wrapper (like its `shape`/`strides`). No dtype promotion,
casting (`astype`/`to`/`cpu`/`cuda`), non-float64 kernels, CUDA,
autograd, `Tensor` integration, or new numeric kernels came with it —
those remain the future phases the contract enables. This **closes Phase
A — the native CPU runtime — in code**; the next milestone is **v2.0, the
Phase B native autograd design**. Full design in
[native_dtype_device_metadata_design.md](native_dtype_device_metadata_design.md).

### Native dtype/device metadata — design (v1.20)

v1.20 is **design-only**: no kernels change, no runtime behavior changes,
and dtype/device metadata is **not implemented**. It follows the
reductions milestone and **closes the Phase A design surface** — the last
native-CPU-runtime piece before Phase B (native autograd).

Today the native runtime is **float64-CPU-only, implicitly**:
`NativeStorage` allocates a `double[]` buffer and records nothing about
its element type or location, and `NativeTensorCore`/`NativeTensor`
expose `shape`/`strides`/`contiguous` but no `dtype` or `device`. The
design makes that assumption **explicit**: dtype and device become
inspectable metadata — owned by `NativeStorage` (the allocation owner, so
views share it and CUDA later has device-aware storage), surfaced
read-only through `NativeTensorCore.dtype`/`.device` and
`NativeTensor.dtype`/`.device`. They are validated canonical string tags
(`"float64"`, `"cpu"`), JSON-friendly for future checkpoints and never
silently inferred from NumPy; the NumPy correspondence stays explicit at
the conversion boundaries only. Constructors gain default-preserving
`dtype`/`device` arguments (`dtype=None`→`"float64"`, `device="cpu"`), so
every existing call is byte-for-byte unchanged.

The design specifies operation validation (binary ops and matmul require
matching dtype and device, naming both on a mismatch; `sum` preserves
dtype; `mean` stays float64, with future integer-mean deferred; `relu`
requires a numeric dtype; `to_numpy` will match the stored dtype once
non-float64 exists) and a hard **no-promotion / no-auto-copy /
no-silent-conversion / no-NumPy-fallback** rule. Casting and device moves
(`astype`/`to`/`cpu`/`cuda`) are reserved as future, explicit,
copy-producing operations — `cuda()` should not exist until a CUDA
backend does. Crucially, because the kernels are float64/CPU only, the
design **recommends rejecting** any non-`float64`/non-`cpu` construction
(the safer of reject-vs-inert), so no tensor ever advertises a dtype the
kernels cannot actually compute. The contiguous fast path, broadcasting,
reductions, and the thin wrapper are all unchanged. A metadata-only
implementation (float64/cpu only) is proposed as **v1.21**, closing Phase
A in code before the Phase B autograd design. Full design in
[native_dtype_device_metadata_design.md](native_dtype_device_metadata_design.md).

### Native reductions — implementation (v1.19)

v1.19 implements native `sum` and `mean` reductions for
`NativeTensorCore` / `NativeTensor`, following the v1.18 design. Supported
semantics are NumPy-style: `axis=None` reduces every element, a single
integer or **negative** `axis` reduces one dimension, and `keepdims`
controls whether the reduced axis stays as size 1.

```python
a = NativeTensorCore.from_array(np.arange(6.0).reshape(2, 3))
a.sum()                     # () scalar total
a.sum(axis=0)               # (3,)
a.sum(axis=1, keepdims=True)# (2, 1)
a.mean(axis=-1)             # (2,)  negative axis
```

A pure `reduce_shape(shape, axis, keepdims)` helper (beside
`broadcast_shapes`) infers the output shape and validates axis/keepdims.
The compute is one new C ABI kernel, **`tf_core_sum`**, that is the
**dual of broadcasting**: where a broadcast reads one element into many
output positions through zero *read*-strides, a reduction walks the input
odometer and writes many elements into one output cell through zero
*write*-strides (0 on reduced axes). It reads the input through its
existing shape/stride/offset metadata, so contiguous, transposed,
narrowed, and nonzero-offset inputs all reduce correctly **without being
materialized**, accumulating into freshly allocated zero-initialized
row-major contiguous output. `mean` reuses `sum` and scales the output in
place by `1/count` via a small `tf_storage_scale` primitive — no NumPy
touches the data. `NativeTensor.sum`/`mean` delegate to the core, so the
wrapper inherited reductions with no reduction-specific code, and the
explicit NumPy and native backends gained symmetric `sum`/`mean` methods.

Numerical behavior is honest: float sums are order-sensitive, so the
kernel uses a plain deterministic row-major accumulation (no
SIMD/FMA/Kahan/pairwise), and results are checked against NumPy with a
**tolerance** (`np.allclose`), not bit-for-bit. Reductions are
**forward-only** — no autograd was added; the broadcast/backward
relationship (a broadcast backward is a reduction over the broadcast
axes) stays reserved for the future native-autograd phase. No
`max`/`argmax`/`min`/`product`, tuple axes, `Tensor` integration, CUDA,
dtype promotion, operator overloads, or distributed reductions came with
it. Full design in
[native_reductions_design.md](native_reductions_design.md).

### Native reductions — design (v1.18)

v1.18 is **design-only**: no kernels change and reductions are **not
implemented**. With Phase A1 (contiguous fast path) and A2 (broadcasting)
complete, the next Phase A step is reductions — collapsing an axis (or all
of them) of a tensor — and this milestone writes their design.

Reductions follow broadcasting for a concrete reason: they are the
**prerequisite for native autograd**. Broadcasting's forward pass reads
one element into many positions (zero read-strides); its backward pass
must do the opposite — **sum the gradient over the broadcast axes** —
which is a reduction. Today `NativeTensorCore` / `NativeTensor` expose no
`sum`/`mean`, so no broadcasting op can have a native backward yet; A3
unblocks that.

The design specifies `sum` and `mean` first (`max`/`argmax`/`min`/
`product` deferred — `max`/`argmax` complicate the zero-initialized
accumulate and, for `argmax`, return indices rather than float64 values),
with NumPy-style semantics: `axis=None` reduces everything, a single
integer (or negative) `axis` reduces one dimension, and `keepdims`
controls whether the reduced axis stays as size 1
(`(2, 3).sum(axis=1) -> (2,)`, `... keepdims=True -> (2, 1)`,
`().sum() -> ()`). The traversal is the **dual of broadcasting**: where
broadcasting reads through zero strides, a reduction **writes** through
zero strides — many input elements scatter-accumulate into one output
cell — so the same odometer machinery drives it, reads any
contiguous/transposed/narrowed/nonzero-offset input directly without
materializing, and writes a freshly allocated row-major contiguous
output. Numerical honesty is explicit: float sums are order-sensitive, so
the design commits to a plain deterministic loop (no Kahan/pairwise/SIMD
in first scope) and to comparing against NumPy with **tolerances**, not
bit-for-bit. `NativeTensor` will inherit `sum`/`mean` by delegation with
no wrapper change; no autograd, `Tensor` integration, CUDA, dtype
promotion, operator overloads, tuple-axis, or distributed reductions come
with it. Implementation is v1.19. Full design in
[native_reductions_design.md](native_reductions_design.md).

### Native broadcasting — implementation (v1.17)

v1.17 implements native broadcasting for the elementwise binary ops
`add`/`subtract`/`multiply`, lifting the native runtime's exact-shape
restriction to NumPy-style broadcasting. `NativeTensorCore.add(other)` (and
`subtract`/`multiply`) now accept identical shapes **or** any
broadcast-compatible pair — scalar↔tensor, same-rank size-1 stretching,
and left-padding a lower-rank operand with leading 1s:

```python
a = NativeTensorCore.from_array(np.ones((2, 3)))
row = NativeTensorCore.from_array([10.0, 20.0, 30.0])   # (3,)
a.add(row)                    # (2, 3): (3,) left-pads to (1, 3), stretches
NativeTensorCore.from_array(np.ones((3, 1))).multiply(
    NativeTensorCore.from_array(np.ones((1, 4))))        # (3, 1) * (1, 4) -> (3, 4)
```

`_binary_core_op` now dispatches **three ways**:

- **same shape, both contiguous** → the v1.14 flat fast-path kernel
  (`<op>_contiguous`), unchanged;
- **same shape, either strided** → the generic odometer kernel (`tf_core_<op>`),
  unchanged;
- **differing but compatible shapes** → the **broadcast traversal**: a
  pure-Python `broadcast_shapes(a, b)` infers the output shape (raising a
  clear `ValueError` naming both shapes if incompatible), a small
  `_broadcast_strides` helper builds each operand's read-strides — the
  real stride on a genuine axis, **stride 0** on a stretched or
  left-padded size-1 axis — and the **same generic odometer kernel**
  consumes them. A zero stride means "re-read one element", so nothing is
  materialized; the expanded operand never exists in memory.

The key implementation note: **no new C++ kernel was added.** The existing
`tf_core_binary` odometer already walks the output shape advancing each
operand by its own strides, so it is already broadcast-capable once fed
zero-augmented strides — broadcasting is entirely a Python-side stride
computation (`broadcast_shapes` sits beside the other pure metadata
helpers, testable without the built backend). Output is always freshly
allocated row-major contiguous storage; operands are never mutated.
`NativeTensor` inherited broadcasting through `NativeTensorCore` with
**no wrapper edit** — `a.add(b)` on a `(2, 3)` and a `(3,)` now works
through the wrapper. Same-shape correctness (fast path and odometer) is
unchanged, verified by regression tests, and every broadcast result is
checked exactly (`np.array_equal`) against NumPy — including transposed,
narrowed, and nonzero-offset broadcast operands. Errors stay explicit
(incompatible shapes raise, naming both; no silent NumPy fallback). No
reductions, autograd, `Tensor` integration, CUDA, dtype promotion,
operator overloads, or matmul broadcasting were added. Full design in
[native_broadcasting_design.md](native_broadcasting_design.md).

### Native broadcasting — design (v1.16)

v1.16 is **design-only**: no kernels change and broadcasting is **not
implemented**. With Phase A1 (the contiguous fast path) complete —
designed v1.13, implemented v1.14, benchmark impact reported v1.15 — the
next Phase A step is broadcasting, and this milestone writes its design.

Today the native elementwise ops are **exact-shape only**:
`NativeTensorCore._binary_core_op` rejects any mismatched pair (a `(3, 4)`
and a `(3, 1)`, or a scalar and a matrix) with a clear `ValueError`. The
design specifies lifting that to NumPy-style broadcasting for
`add`/`subtract`/`multiply` — scalar↔tensor, same-rank size-1 stretching,
and left-padding a lower-rank operand with leading 1s
(`() + (3, 4) -> (3, 4)`, `(3, 1) + (1, 4) -> (3, 4)`,
`(1, 3, 1) + (2, 1, 5) -> (2, 3, 5)`). The mechanism is a **zero-stride
read model**: a stretched axis is read with stride 0 (re-reading one
element) rather than materializing an expanded operand, and the output
stays freshly allocated row-major contiguous native storage. The v1.14
fast path is preserved untouched for the same-shape contiguous case;
broadcasting only engages when the shapes actually differ, and takes the
generic broadcast odometer first (specialized broadcast fast paths are
deferred). It composes with transposed/narrowed/offset views because
everything is expressed in strides. Errors stay explicit — a mismatch
names both shapes, and there is **no silent NumPy fallback**. Autograd
implications (a broadcast forward read is a sum-reduction on the backward
pass) are noted for later, not built. The full design, scope
(implementation is v1.17), test/benchmark plans, and roadmap fit are in
[native_broadcasting_design.md](native_broadcasting_design.md).

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

The contiguous elementwise fast path is **implemented** (v1.14) and
**reported** (v1.15); native broadcasting is **implemented** (v1.17);
native `sum`/`mean` reductions are **implemented** (v1.19); the
dtype/device metadata contract was **designed** (v1.20) and is now
**implemented** (v1.21, float64/cpu only) — which **closes Phase A,
the native CPU runtime, in code**. **Phase B is under way**: the native
autograd design (v2.0) is complete, v2.1 implemented its metadata
skeleton and reverse-topological backward driver, **v2.2 wired the core
operations into that engine** — `add`/`subtract`/`multiply`/`relu`/`sum`/
`mean`/`matmul`/`reshape`/`transpose`/`T`/`contiguous_copy` are
differentiable, broadcasting backward runs through the native
`unbroadcast` reduction, and its one new kernel is the fused
`relu_backward` — **v2.3 completed the view-backward set** with `narrow`
backward through the native `tf_core_narrow_backward` scatter kernel (the
odometer dual of `sum`), **v2.4 gave the graph an explicit lifetime** —
one-shot `backward(retain_graph=False)` by default, opt-in
`retain_graph=True` reuse, deterministic freed-graph errors, and
snapshot-based failure safety (a Python-only change; no kernel touched) —
and **v2.5 (above) characterized the whole stack** with a measurement-only
benchmark harness (four modes separating forward-native, graph
construction, fresh forward+backward, and repeated retained backward; a
single honest hardware snapshot; no speed assertions). Gradients are
`NativeTensor`-backed float64/cpu and enforce `grad.dtype ==
tensor.dtype` / `grad.device == tensor.device` against the fields v1.21
made real. `NativeTensorCore` and the C++ kernels still own no graph
state, and there is still no Tensor integration and no CUDA today.
**v2.6 closed Phase B** with cross-cutting guardrail tests, and **Phase C —
the native training stack — is now under way: v3.1 shipped
`NativeParameter` and the parameter-registration contract, v3.2 shipped
`NativeModule` — automatic assignment registration, deterministic
identity-deduplicated recursive traversal, recursive `zero_grad()`, and
train/eval state propagation — v3.3 shipped the in-memory state
dictionary contract: `state_dict()` snapshots and atomic
identity-preserving `load_state_dict()` — v3.4 shipped the first
concrete layer, `NativeLinear`, its backward supplied entirely by the
existing autograd — v3.5 completed the first composable model surface
with `NativeReLU` and `NativeSequential` — and v3.6 (above) added the
first native loss, `NativeMSELoss`, closing the forward side of the
training story: model → loss → backward now runs natively end to
end.** There is still no optimizer, parameter-update primitive, file
serialization, or training loop — the recommended next milestone is
**v3.7 — Native Parameter Mutation Safety and Versioning Contract**
(version counters and stale-forward errors so optimizer updates cannot
silently corrupt backward through old graphs; the required foundation
before `NativeSGD`); `divide` backward remains separate later work.
CUDA experiments remain a separate future branch (where `device` gains a
second value), and an AMP / Tensor Core path is where `dtype` later gains
float16/bfloat16. The Python framework stays the reference implementation
throughout.
