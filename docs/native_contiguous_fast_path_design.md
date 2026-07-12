# Native contiguous fast-path — design

This began as a **design document** (written in v1.13, ahead of any
code) specifying a contiguous fast path for the native elementwise
kernels (`relu`/`add`/`subtract`/`multiply`) in the `NativeTensorCore`
layer. **Status: implemented in v1.14** exactly as specified below —
flat, index-free kernels (`tf_core_relu_contiguous`,
`tf_core_add_contiguous`, `tf_core_subtract_contiguous`,
`tf_core_multiply_contiguous`) beside the generic odometer kernels, with
`NativeTensorCore.relu` / `_binary_core_op` choosing the fast path when
every operand is contiguous and the generic odometer path otherwise. The
generic kernels are untouched and remain the reference/fallback; the two
paths are bit-for-bit equal (§4), which the v1.14 tests lock down (§9).
`NativeTensor` inherited the change with no wrapper edits. No
broadcasting, reductions, autograd, `Tensor` integration, or CUDA were
added, and performance is measured by the benchmark suite, not claimed
here (§10).

**v1.15 benchmark reporting confirmed the intended behavior.** On a local
run of the existing suite, the contiguous elementwise rows (both
`NativeTensorCore` and `NativeTensor`) moved to roughly raw-buffer-C++
speed, while the non-contiguous view rows stayed on the generic odometer
path and remained slower — exactly the contiguous-vs-strided spread this
design predicted (§10). The full tabulated report lives in
[backend_experiments.md](backend_experiments.md) ("Contiguous fast-path —
benchmark impact report (v1.15)"); numbers are hardware-dependent and no
test asserts a speedup. Broadcasting and reductions remain separate,
later Phase A work (§12). The sections below remain the design of record.

For where this sits, see [backend_experiments.md](backend_experiments.md)
(the native runtime and benchmarks) and
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md) (the
wrapper that will inherit the improvement).

## 1. The problem the benchmarks found

The v1.12 benchmark suite times each elementwise op across four layers —
NumPy, the raw-buffer C++ kernels, `NativeTensorCore`, and the
`NativeTensor` wrapper — plus non-contiguous view rows. Two facts stand
out:

- The `native tensor` rows sit **close to** their `tensor core` rows, so
  the Python wrapper's ownership/lifetime/conversion bookkeeping is
  **not** the main elementwise cost.
- Both trail the flat `cpp raw buffer` row substantially, even though
  they run the same arithmetic. The gap is the **generic shape/stride
  traversal** the native runtime uses.

So the elementwise overhead lives in the *lower* runtime, in the way it
walks memory — not in the wrapper on top. That is what this fast path
targets.

## 2. Why the generic odometer traversal is general but slower

The native kernels (`tf_core_relu`, `tf_core_binary` in `cpp/kernels.cpp`)
walk a logical tensor with an **odometer loop**: they keep a per-axis
counter, and for every output element they advance the counter from the
innermost axis, add each axis's stride to the source position(s), and
subtract a full extent when an axis wraps. This is exactly the same
traversal as `tf_storage_materialize`, which is what makes it so useful:

- It reads **any** layout correctly — transposed views, narrowed views,
  arbitrary strides and offsets — without materializing first. One code
  path handles every case.

But that generality has a per-element price that a contiguous tensor does
not need to pay:

- A carry loop over axes runs for every element (a branch and a
  stride-add per axis, plus the wrap subtraction).
- A per-call heap allocation of the counter array (`new int64_t[ndim]`).
- The access pattern is opaque to the compiler and CPU: it cannot assume
  the source advances by one, so it cannot vectorize or prefetch as
  freely as a flat loop.

For a **contiguous** tensor none of this is necessary, because the
odometer degenerates into a straight line (see §4).

## 3. The proposed fast-path rule

> **If every operand and the output are contiguous, use a flat,
> index-free pointer loop. Otherwise, keep the current generic
> shape/stride odometer traversal.**

Concretely, for the first-scope ops:

- **Contiguous case (fast path):** compute
  `dst[i] = op(a[a_offset + i], b[b_offset + i])` for `i` in
  `[0, numel)` — a single sequential loop, no counters, no per-axis
  branching, no counter allocation. (`relu` is unary:
  `dst[i] = max(src[src_offset + i], 0)`.)
- **Non-contiguous case (generic path):** unchanged — the existing
  odometer kernel, byte-for-byte the code that runs today.

The output is **always** freshly allocated row-major contiguous storage
(the kernels write into a new `NativeTensorCore.zeros(shape)`), so the
output side is contiguous by construction; the decision is driven by the
*inputs*. For binary ops, the fast path requires **both** operands to be
contiguous — if either is a strided view, the whole op takes the generic
path. (A mixed fast/slow hybrid — read the contiguous operand flat and
the strided one by odometer — is a possible later refinement, explicitly
out of first scope.)

## 4. Why the fast path is exactly equivalent, not merely close

Contiguity here means `strides == row_major_strides(shape)` (the existing
`is_contiguous_shape` / `NativeTensorCore.contiguous` test). For such a
tensor the odometer's source position sequence is precisely
`offset, offset+1, offset+2, …, offset+numel-1`: each innermost step adds
stride 1, and every wrap subtracts `shape[d]*stride[d]` exactly as much
as the higher axes added — the net walk is linear. Therefore the logical
element that the odometer reads at output index `i` is exactly
`data[offset + i]`, which is what the flat loop reads.

Because the arithmetic is applied **per element in the same order**, with
no reassociation, no fused multiply-add, and no SIMD horizontal
reductions that could change float64 rounding, the fast path produces
results that are **bit-for-bit identical** to the generic path — not
"within a tolerance." This is the property the tests must lock down (§9).

A useful corollary: a contiguous tensor may still carry a **nonzero
offset** (e.g. a row-slice / `narrow` along axis 0 keeps row-major
strides but shifts the offset). The fast path handles this by starting
from the base pointer `data + offset`; it does not require offset 0.

## 5. Where the branch lives

The dispatch belongs in the **native runtime layer, below
`NativeTensor`** — so the wrapper inherits the improvement with **zero
changes to its code**:

- **Recommended placement:** `NativeTensorCore`'s Python methods
  (`relu`, `_binary_core_op`) choose the kernel using the contiguity they
  already know from metadata (`self.contiguous`, `other.contiguous`) —
  this is a cheap boolean already computed at view construction — and
  call either a new **flat C++ kernel** (e.g. `tf_core_add_contiguous`,
  taking `numel` and the two offsets) or the existing odometer kernel.
  Each C++ kernel stays single-purpose; no runtime flags are threaded
  through a combined kernel.
- **Considered alternative:** pass a `contiguous` flag into the existing
  combined kernel and branch inside C++. Rejected for first scope because
  it complicates the one kernel that must remain the trusted generic
  reference; keeping the fast loop a *separate* function keeps the
  reference path untouched and the two trivially comparable in tests.

Either way, the decision and both loops sit at or below
`NativeTensorCore`. `NativeTensor.relu/add/subtract/multiply` continue to
call `self._require_open().<op>(...)` verbatim — the wrapper is not
edited, and `NativeTensor` benefits automatically.

## 6. Scope

**In first scope (v1.14):** `relu`, `add`, `subtract`, `multiply` — the
elementwise ops the benchmark implicates, all already producing
contiguous output.

**Out of first scope (deliberately):**

- **matmul** — a different (triple-loop) kernel with its own optimization
  story (the raw-buffer `matmul_tiled` experiment); not elementwise.
- **reductions** — no reduction ops exist yet; they are Phase A3.
- **broadcasting** — still not supported anywhere; Phase A2.
- **Tensor integration, autograd, CUDA** — all remain future phases and
  are untouched here.

## 7. Contiguity detection

No new metadata is needed. Contiguity is already derived at view
construction by `shape_info` and exposed as `NativeTensorCore.contiguous`
(and `NativeTensorView.contiguous`), defined as
`strides == row_major_strides(shape)`. The fast path reuses that flag
directly.

The detector is deliberately **conservative**: it keys off an exact
row-major stride match, so a layout that is contiguous *in memory* but
carries, say, an unusual stride on a size-1 axis may be reported
non-contiguous and fall to the generic path. That costs a missed
optimization, never correctness — a false negative is safe, a false
positive would not be, and this test cannot produce a false positive for
any layout the generic path would read differently.

## 8. Invariants that must not change

- **Exact-shape semantics** — binary ops still require identical shapes
  with no broadcasting; the shape check is unchanged and happens before
  any traversal choice.
- **Output layout** — output stays row-major contiguous, freshly
  allocated, independent of inputs.
- **Numerical results** — bit-for-bit identical to today (see §4).
- **Error behavior** — `TypeError` for non-`NativeTensorCore` operands,
  `ValueError` for shape mismatch, `RuntimeError` for closed
  storage/tensors — all raised by the same Python validation, before the
  kernel runs. The fast path is purely an internal traversal choice made
  *after* validation succeeds.

## 9. How tests should compare the two paths

The v1.14 test suite (not this milestone) should prove **equivalence**,
using exact equality because the paths must match bit-for-bit:

- **Fast vs NumPy:** contiguous `relu`/`add`/`subtract`/`multiply` equal
  the NumPy reference exactly (`np.array_equal`, not `allclose`).
- **Fast vs generic:** the fast-path result on contiguous inputs equals
  the generic-path result on **value-identical non-contiguous inputs**.
  A clean way to force the generic path on the same values is a
  double-view that is genuinely non-contiguous but materializes to the
  same array (e.g. build an operand as a transposed view of the
  transpose-laid-out data), then assert both op results are equal to each
  other and to NumPy.
- **Non-contiguous unchanged:** existing view-based tests (transposed and
  narrowed operands) continue to match NumPy — they still ride the
  generic path.
- **Edge cases (each exact vs NumPy):** scalars, size-1 dimensions,
  nonzero offsets (row slices), so the base-pointer handling is covered.
- **Errors unchanged:** the existing closed/`TypeError`/`ValueError`
  tests keep passing without modification.

No performance assertions belong in the test suite; speed is a benchmark
concern, not a correctness one.

## 10. How benchmarks should demonstrate impact honestly

The existing suite already separates the cases needed to see the effect,
so **no benchmark restructuring is required**:

- The contiguous `tensor core` / `native tensor` elementwise rows should
  move **toward** the flat `cpp raw buffer` row once the fast path lands.
- The `… (view)` rows should **stay** on the generic odometer path and
  remain roughly where they are.
- The difference between the two — same op, contiguous vs strided — is
  exactly the cost the fast path removes, shown side by side.

Reporting stays honest: correctness is verified before timing, numbers
are medians after warmup, results are hardware-dependent, and **nothing
asserts a speedup**. Because v1.13 ships no implementation, this document
makes **no performance claim** — only a prediction the v1.14 numbers will
confirm or refute.

## 11. Risks and edge cases

- **Scalar tensors (`ndim == 0`).** Already special-cased in every kernel
  (`dst[0] = op(a[a_offset], b[b_offset])`) and contiguous by definition;
  the fast path treats them as `numel == 1`. No new risk.
- **Size-1 dimensions.** May be flagged non-contiguous by the exact
  stride match and fall to the generic path — a safe false negative
  (§7), never a wrong answer.
- **Negative strides.** Never equal to (positive) row-major strides, so
  always non-contiguous → generic path. The fast path never sees them.
- **Nonzero offsets.** Allowed on the fast path via the base pointer
  `data + offset`; contiguity does not require offset 0 (§4).
- **Transposed views.** Non-contiguous by construction → generic path,
  unchanged.
- **Narrowed views.** `narrow` along axis 0 stays contiguous (offset
  shift, row-major strides) → fast path with a nonzero offset; `narrow`
  along an inner axis becomes non-contiguous → generic path. Both remain
  correct.
- **Closed storage / wrapper lifetime.** Unchanged: `_require_open()`
  runs before the kernel choice, so a closed core/tensor raises
  `RuntimeError` before any traversal. The flat kernel reads
  `handle->data` exactly as the odometer kernel does — no new lifetime
  surface, no new ownership rules.
- **Mixed contiguous/non-contiguous binary operands.** Handled by taking
  the generic path whenever either operand is non-contiguous — correct,
  if not maximally fast; the hybrid optimization is a later refinement.

## 12. Fit in the Daedalus-class roadmap

This is the first optimization step of **Phase A — native CPU runtime**,
which precedes any autograd or device work:

- **A1 — contiguous elementwise fast path** (this design → v1.14,
  now implemented; the next step is v1.15, an honest benchmark impact
  report over the same suite).
- **A2 — broadcasting** (shape alignment for elementwise ops).
- **A3 — reductions** (sum/mean/max and friends, with their own
  traversal and numerical-order considerations).
- **A4 — dtype / device metadata** (beyond float64-CPU-only).

Only once the CPU runtime is broad and fast does the roadmap move on to
**native autograd**, then the **native training stack**, then the **CUDA
runtime**, **AMP / Tensor Core** path, **Transformer / text** examples,
**distributed / DDP**, and a final **benchmark / profiling / docs**
polish. Each phase lands only when the previous one is tested and
documented; the Python framework remains the reference implementation
throughout.
