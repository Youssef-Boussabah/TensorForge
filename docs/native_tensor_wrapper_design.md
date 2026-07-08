# Forward-only native tensor wrapper — design (Stage 2)

This began as a **design document** (written in v1.7, ahead of any
code) for a forward-only convenience layer over `NativeTensorCore` — a
small, explicit, autograd-free tensor type for user-facing experiments
with the native backend. **Status: implemented** as
`tensorforge.experimental.NativeTensor`. The shell (constructors,
metadata, `to_numpy`, lifetime) landed in v1.8, forward compute
(`relu`/`add`/`subtract`/`multiply`/`matmul`) in v1.9, and the
metadata-only view ops (`reshape`/`transpose`/`T`/`narrow`) plus
`contiguous_copy` in v1.10; v1.11 added a runnable example and a
metadata-only `repr`; v1.12 added honest benchmark characterization of
the wrapper's overhead (see section 9 and
[backend_experiments.md](backend_experiments.md)). The sections below
remain the design of record — the plan the code follows — with the
staged status tracked in section 9.
The name `NativeTensor` was kept (the `ExperimentalNativeTensor`
fallback in [Risks](#10-risks) proved unnecessary — the
`tensorforge.experimental` namespace carries the distinction).

For where this sits in the larger backend plan, see
[dispatch_design.md](dispatch_design.md); for the runtime primitives it
wraps, see [backend_experiments.md](backend_experiments.md).

## 1. Purpose

`NativeTensorCore` already exists and works: it owns C++ storage, holds
shape/stride/offset metadata, produces metadata-only views, and runs
`relu`/`add`/`subtract`/`multiply`/`matmul` natively over strided
layouts. What it lacks is **ergonomics**. Today, driving it means
threading raw cores through the explicit backend object
(`get_backend("native")`), remembering which calls return borrowing
views versus owning cores, and calling `close()` on the right objects
in the right order. That is the correct low level for a runtime, but it
is too sharp for anyone who just wants to *try the native backend* on a
few forward computations.

The wrapper exists to give those experiments a small, readable,
Tensor-shaped surface — `x.relu()`, `x.add(y)`, `x.matmul(y)`,
`x.to_numpy()` — **without pretending to be `tensorforge.Tensor`.** It
is still forward-only, still float64, still exact-shape, still explicit
about conversion and lifetime. It adds convenience and a clear ownership
story on top of `NativeTensorCore`; it adds no new capability, no
autograd, and no framework integration.

Why it is deliberately *not* `tensorforge.Tensor`: `Tensor` carries the
autograd graph, `requires_grad`, `grad`, `backward()`, and value
semantics that the whole framework — modules, optimizers, checkpoints —
depends on. The wrapper carries none of that. Keeping it a separate type
is what lets the native runtime be exercised end to end in isolation,
with its differences (no broadcasting, explicit `close()`) staying
visible instead of leaking into the trusted frontend.

## 2. Non-goals

The wrapper explicitly does **not**:

- **No autograd.** No graph, no `requires_grad`, no `grad`, no
  `backward()`. Forward compute only.
- **No optimizer support.** `SGD`/`Adam` operate on `Parameter`/NumPy
  state and never see a `NativeTensor`.
- **No `Module`/`Parameter` integration.** It is not a layer weight, not
  a buffer, not something `state_dict()` serializes.
- **No CUDA.** CPU float64 only, like the runtime beneath it.
- **No implicit fallback.** An operation the native runtime cannot do
  raises; it never silently computes the answer with NumPy.
- **No silent NumPy conversion.** Data enters and exits only through the
  named boundaries (`from_array`, `to_numpy`), never implicitly.
- **No broadcasting** unless a later, separate milestone designs it. At
  this stage elementwise/binary ops require identical shapes.
- **No replacement for `tensorforge.Tensor`.** The Python Tensor stays
  the reference frontend for everything real; the wrapper is an
  experimental convenience layer, nothing more.

## 3. Relationship to existing layers

Four layers, from most trusted to most experimental:

1. **`tensorforge.Tensor`** — the stable Python/NumPy autograd frontend.
   Everything the framework ships (modules, optimizers, losses,
   checkpoints) is built on it. Unchanged by any of this.
2. **Explicit backend API** (`get_backend`, `available_backends`) — the
   current safe entry point (Stage 1). Names a backend, returns a
   backend object, converts explicitly (`tensor_from_array` /
   `to_numpy`). No implicit routing, no Tensor contact.
3. **`NativeTensorCore`** — the low-level native runtime object: owned
   C++ storage + view metadata + forward kernels. Powerful but sharp;
   the caller manages ownership and lifetime by hand.
4. **Future `NativeTensor` wrapper** (this design) — a thin, forward-only
   convenience layer *around* a single `NativeTensorCore`. It owns or
   borrows exactly one core, exposes a small Tensor-shaped method set,
   and makes lifetime and conversion explicit and pleasant. It sits
   beside the explicit backend API, not above `Tensor`.

The wrapper composes `NativeTensorCore`; it does not reimplement it. Its
methods delegate to core methods and re-wrap the results. It never
reaches into layer 1 or 2 — in particular it never accepts or produces a
`tensorforge.Tensor`.

## 4. Ownership and lifetime

Lifetime is the hard part of any object over C++-owned memory, so the
rules are stated up front, to be honored by the implementation:

- **Who owns the core.** A `NativeTensor` produced by a constructor
  (`from_array`, `zeros`, `full`) wraps a core it **owns** — it created
  the storage, and it is responsible for releasing it.
- **Views borrow.** A `NativeTensor` produced by a view/metadata op
  (`reshape`, `transpose`, `T`, `narrow`) wraps a core that **borrows**
  the same underlying storage. It shares storage with its parent and
  siblings and must not free it. This mirrors `NativeTensorCore`'s
  existing owner/borrower model — the wrapper carries the flag through,
  it does not invent a new one.
- **Compute results own.** `relu`/`add`/`subtract`/`multiply`/`matmul`
  allocate fresh contiguous storage, so the returned `NativeTensor`
  **owns** its core, independent of its inputs.
- **Closing the wrapper.** `close()` releases the wrapper's hold on its
  core. Closing an owner frees the native storage; closing a borrowing
  view detaches only that view and leaves the owner and siblings alive —
  again matching the core's semantics. `close()` is idempotent
  (double-close is safe).
- **Context-manager support: yes.** `__enter__`/`__exit__` should exist
  so `with NativeTensor.from_array(...) as x:` frees deterministically on
  block exit. This matters on Windows in particular (see
  [Risks](#10-risks)) and matches the ergonomic style `NativeStorage`
  and `NativeTensorCore` already use.
- **When the underlying core is closed.** If the owner's storage is
  released while a borrowing wrapper still refers to it, operations on
  that wrapper raise `RuntimeError` (the core already enforces this — the
  wrapper surfaces it, it does not hide it).
- **Why hidden copies are dangerous.** A convenience layer is tempting to
  make "just work" by silently materializing to NumPy and back, or by
  duplicating storage to dodge lifetime questions. Both are banned:
  hidden copies destroy the visible-cost property the conversion contract
  guarantees, silently multiply memory, and turn a clear `close()` story
  into a guessing game about which buffer is live. Every copy is a named,
  explicit call.

## 5. Conversion contract

The wrapper obeys the v1.6 conversion contract exactly (see
[dispatch_design.md](dispatch_design.md), "The explicit conversion
contract"):

- **`from_array(values)` enters** the native world: array-like/NumPy data
  in, a new owning `NativeTensor` out, as a **copy** into fresh C++
  storage.
- **`to_numpy()` exits** it: a fresh float64 NumPy array out, materialized
  through the core, sharing no mutable state with native storage.
- **Conversions are always explicit.** There is no implicit `__array__`,
  no auto-materialization inside arithmetic, no NumPy fallback path.
- **Materialization and copies are documented** at every call site: which
  methods copy (`from_array`, `to_numpy`, `contiguous`, every compute op)
  versus which are metadata-only views (`reshape`, `transpose`, `T`,
  `narrow`).
- **No `tensorforge.Tensor` input** is accepted, in either direction,
  unless and until a later stage explicitly designs that bridge (Stage 3
  in the dispatch plan). Passing a `Tensor` raises a clear `TypeError`.

## 6. Proposed minimal API

Sketch only — names and signatures may shift when implemented.

```python
from tensorforge.experimental import NativeTensor

x = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]])
y = NativeTensor.full((2, 2), 10.0)
z = NativeTensor.zeros((2, 2))

# forward compute — each returns a new owning NativeTensor
r = x.relu()
s = x.add(y)
m = x.matmul(y)

# metadata-only views — borrow x's storage, no copy
xt = x.transpose()          # or x.T
xr = x.reshape((4, 1))
xn = x.narrow(0, 0, 1)

arr = m.to_numpy()          # explicit exit → float64 NumPy array
x.close()                   # explicit release (idempotent)
```

Proposed surface:

- **Constructors:** `from_array`, `zeros`, `full`.
- **Metadata (properties):** `shape`, `strides`, `ndim`, `numel`.
- **Materialization / conversion:** `contiguous`, `to_numpy`.
- **Lifetime:** `close`, `__enter__`, `__exit__`.
- **Compute (each returns a new owning wrapper):** `relu`, `add`,
  `subtract`, `multiply`, `matmul`.
- **Views (each returns a borrowing wrapper):** `reshape`, `transpose`,
  `T`, `narrow`.

Every one of these delegates to an existing `NativeTensorCore` method;
the wrapper's job is ergonomics, ownership bookkeeping, and re-wrapping
results — not new math.

## 7. Error and shape behavior

The wrapper preserves the native runtime's strictness and never softens
it:

- **Exact-shape requirement** for elementwise/binary ops (`add`,
  `subtract`, `multiply`): operand shapes must match exactly.
- **No broadcasting** at this stage. A `(2, 3)` and a `(3,)` do not
  combine; they raise. (`relu` is unary and accepts any shape.)
- **`matmul`** is strictly 2-D `(m, n) @ (n, p)`, matching the core.
- **Unsupported operand types raise `TypeError`** with a clear message —
  passing a NumPy array, a Python list, or a `tensorforge.Tensor` where a
  `NativeTensor` is required fails loudly and names the expected type,
  consistent with how the native backend object already rejects
  non-core operands.
- **Shape mismatches raise `ValueError`**, with the offending shapes in
  the message.
- **No silent fallback to NumPy**, ever. If the native runtime cannot do
  it, the wrapper raises — it does not quietly produce a NumPy answer.

## 8. Testing plan for future implementation

When the wrapper is built, its tests should prove (these are the
acceptance criteria for v1.8+ work, not tests to add now):

- **Constructors work:** `from_array`, `zeros`, `full` produce wrappers
  with correct shape/strides/metadata and correct values via `to_numpy`.
- **`to_numpy` round-trips:** `from_array(x).to_numpy()` reproduces `x`
  as an independent float64 copy.
- **Close / context-manager behavior:** `close()` is idempotent;
  `with` frees deterministically; operations after close raise
  `RuntimeError`.
- **Exact-shape operations:** matching shapes compute correctly; a
  mismatch raises `ValueError` with no broadcasting.
- **View operations share storage:** a view reflects the parent's data
  and computes correctly over strided layout without a copy; closing the
  owner invalidates outstanding views as specified.
- **Wrong operands fail clearly:** NumPy arrays, lists, and
  `tensorforge.Tensor` inputs raise `TypeError` naming the expected type.
- **No `tensorforge.Tensor` integration happens by accident:** a
  guardrail (e.g. an AST/import check like the existing
  `test_framework_init_does_not_import_backends`) confirms the framework
  frontend still never imports the experimental wrapper, and the wrapper
  never imports `Tensor`.
- **No autograd attributes appear:** a `NativeTensor` has no
  `requires_grad`, `grad`, or `backward` — asserted explicitly so the
  forward-only contract can't silently erode.
- **No hidden NumPy fallback:** simulate an unavailable native op and
  assert it raises rather than computing a NumPy answer.

## 9. Staged implementation plan

The wrapper is built in small, tested milestones, mirroring how the
runtime beneath it grew:

- **v1.8 — minimal wrapper (done):** constructors (`from_array`,
  `zeros`, `full`), metadata (`shape`, `strides`, `ndim`, `numel`,
  `contiguous`), `to_numpy`, and the full lifetime story (`close`,
  context manager, ownership flag). No compute, no views.
- **v1.9 — compute ops (done):** `relu`, `add`, `subtract`, `multiply`,
  `matmul`, each returning a new owning wrapper, with the exact-shape
  and error behavior of section 7 (clear `TypeError` naming
  `NativeTensor` for non-wrapper operands, `RuntimeError` for closed
  operands, `ValueError` for shape/2-D mismatch). No operator overloads
  yet.
- **v1.10 — view ops (done):** `reshape`, `transpose`, `T`, `narrow`
  return borrowing wrappers (`owns_core` False) that share the parent's
  storage; `contiguous_copy` returns a fresh owning wrapper.
  Compute ops run over strided views directly. Closing a view spares the
  owner; closing the owner invalidates outstanding views' data access.
- **v1.11 — docs, examples, and polish (done):** a runnable script
  (`examples/native_tensor_demo.py`) exercising the wrapper end to end on
  the native path, a metadata-only `repr`, and the wrapper overview in
  [backend_experiments.md](backend_experiments.md). Still no Tensor
  integration.
- **v1.12 — benchmark coverage (done):** the wrapper's ops (strided
  views and `contiguous_copy` included) are timed beside NumPy, the
  raw-buffer kernels, and `NativeTensorCore` in
  `benchmarks/cpp_backend.py`, overheads included and with no performance
  assertions. The measured story is honest: `native tensor` rows sit
  close to their `tensor core` rows (the wrapper's ownership/lifetime/
  conversion layer is thin), while both trail NumPy — and the generic
  strided-traversal (odometer) loop, not the wrapper, dominates
  elementwise cost. That points at the next optimization target below.
- **v1.13 — contiguous fast-path design (done):** a design (no code) for
  a flat, index-free loop that lets contiguous elementwise ops skip the
  odometer, honestly and without changing semantics — see
  [native_contiguous_fast_path_design.md](native_contiguous_fast_path_design.md).
  Crucially, that optimization lives **below `NativeTensor`**, in the
  `NativeTensorCore`/native-kernel layer: the wrapper stays a thin
  forward-only convenience layer with **no code change**, and inherits
  the speedup automatically once the kernels improve. The v1.12
  benchmarks already confirmed the wrapper is not the bottleneck, which
  is exactly why the fix belongs one layer down.
- **v1.14 — contiguous fast-path implementation (next):** build the flat
  kernels and the contiguity dispatch in `NativeTensorCore`, proven
  bit-for-bit equal to the generic path.
- **Later — integration decision:** only after the wrapper is complete
  and trusted in isolation, *decide whether* to design a
  `Tensor` ↔ native bridge (Stage 3 in the dispatch plan). That decision
  is not pre-committed here.

Each stage lands only when the previous one is tested and documented.

## 10. Risks

- **Lifetime bugs.** The hardest risk: use-after-close, freeing borrowed
  storage, or leaking storage on an error path. Mitigation — carry the
  core's owner/borrower flag faithfully, make `close()` idempotent,
  provide context managers, and test invalidation explicitly.
- **Accidental hidden copies.** A convenience layer invites silent
  materialization. Mitigation — every copy is a named call (`from_array`,
  `to_numpy`, `contiguous`, compute ops); document which methods copy and
  which are metadata-only views; no implicit `__array__`.
- **`NativeTensor` vs `tensorforge.Tensor` naming confusion.** Two
  tensor types with different contracts is a genuine footgun. Mitigation —
  a distinct module (`tensorforge.experimental`), docs that lead with the
  differences, and the fallback name `ExperimentalNativeTensor` if the
  short name proves misleading in practice.
- **Shape/broadcasting divergence.** The wrapper rejects what NumPy (and
  `Tensor`) accept. Mitigation — keep the difference explicit and tested,
  never paper over it; broadcasting, if ever added, is its own designed
  milestone.
- **Future autograd conflicts.** A forward-only type that later people
  want gradients from. Mitigation — keep it firmly forward-only; any
  autograd story is Stage 4 dispatch-design work, decided on paper first,
  and does not retrofit onto this wrapper without a new design.
- **Benchmark confusion.** A friendlier wrapper can be mistaken for a
  performance layer. Mitigation — the wrapper adds Python overhead, not
  speed; benchmarks continue to measure the runtime honestly and say so.
- **Resource cleanup on Windows.** This machine has a known quirk where
  native handles and directories created by one process resist cleanup by
  another. Mitigation — deterministic release via context managers and
  idempotent `close()`, tests that don't depend on GC finalization order,
  and no reliance on process-crossing cleanup.
