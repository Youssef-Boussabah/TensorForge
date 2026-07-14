# Native reductions — design

This began as a **design document** (written in v1.18, ahead of any code)
specifying reductions (`sum`, `mean` first) for the native runtime — new
`NativeTensorCore` / `NativeTensor` methods that collapse one axis (or all
of them) of a tensor. **Status: implemented in v1.19**, following this
design.

The implementation matched the plan closely:

- A pure `reduce_shape(shape, axis, keepdims)` helper (plus
  `_normalize_axis` and `_reduce_out_strides`) sits beside
  `broadcast_shapes` in `cpp.py`, testable without the built backend.
- One new C ABI kernel, **`tf_core_sum`**, is the scatter-accumulate
  odometer of §6 — the exact dual of broadcasting: it walks the input
  with its real shape/strides/offset while advancing an output position
  by per-input-axis write-strides (0 on reduced axes), so many input
  elements accumulate into one zero-initialized output cell. It reads
  transposed/narrowed/nonzero-offset inputs directly, materializing
  nothing.
- **`mean`** reuses `sum` and scales the freshly summed output in place
  by `1/count` via a small storage primitive, **`tf_storage_scale`** (an
  in-place float64 multiply, sibling of `tf_storage_fill`) — staying in
  the native path with no NumPy round trip. (This is a reciprocal-multiply
  by `1/count`, so it can differ from NumPy's `sum/count` by a ULP; that
  is why reduction values are tested with `np.allclose`, not
  `np.array_equal` — see §7.)
- `NativeTensorCore.sum`/`mean(axis=None, keepdims=False)` and the
  delegating `NativeTensor.sum`/`mean` landed, with `sum`/`mean` also
  added to the explicit NumPy and native backends (their surfaces are
  kept symmetric). `NativeTensor` inherited reductions with no
  reduction-specific logic, as predicted.

Reductions ship **forward-only** — no autograd was added; the
broadcast/backward relationship (§8) remains reserved for the future
native-autograd phase. No `max`/`argmax`/`min`/`product`, tuple axes,
`Tensor` integration, CUDA, dtype promotion, operator overloads,
distributed reductions, or advanced summation came with it. The sections
below remain the design of record.

For where this sits, see [backend_experiments.md](backend_experiments.md)
(the native runtime and benchmarks),
[native_broadcasting_design.md](native_broadcasting_design.md) (whose
backward pass is a reduction — §8), and
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md) (the
wrapper that will inherit reductions by delegation).

## 1. Why reductions are the next step (Phase A3)

Phase A gave the native elementwise core a fast path (A1) and
broadcasting (A2). Both are **shape-preserving or shape-expanding**: they
map element(s) to element(s). What the runtime still cannot do is go the
other direction — collapse elements: a `sum`, a `mean`, a loss value, a
column total. That is the whole of A3.

Reductions matter for two concrete reasons:

- **They are the missing elementary op.** Almost every real computation
  ends in a reduction — a loss is a `mean` or `sum` over a batch, a
  normalization needs a per-row `mean`, a softmax needs a `sum`. Without
  reductions the native core can transform data but never summarize it.
- **They are the prerequisite for native autograd.** This is the deeper
  reason and it ties directly to A2. Broadcasting's forward pass reads
  one input element into many output positions (via zero strides); its
  **backward** pass must therefore do the opposite — **sum the upstream
  gradient over the axes that were broadcast** — which is exactly a
  reduction (§8). The NumPy-backed `Tensor` already does this in
  `_accumulate_grad`'s "un-broadcast" step. So there is no native
  autograd for a broadcasting op until the native runtime can reduce.
  A3 unblocks the autograd phase that follows Phase A.

## 2. Current limitation

`NativeTensorCore` and `NativeTensor` expose **no reduction methods
today.** Their compute surface is `relu`, `add`, `subtract`, `multiply`,
`matmul` (plus view ops and `contiguous_copy`) — every one of which
returns a tensor whose element count is driven by the inputs, never
smaller. There is no `sum`, no `mean`, no way to collapse an axis.
`NativeStorage` can be `fill`ed and copied, but nothing accumulates
across logical elements. Adding that is this design's subject.

(The raw-buffer kernels in `cpp/src/` also have no reduction; the
odometer kernels only ever *materialize* or *combine* element-for-
element. Reductions are genuinely new traversal behavior, not a re-skin
of an existing kernel.)

## 3. First target and deferred reductions

**In first scope (v1.19): `sum` and `mean`.** They are the two
reductions autograd needs first (gradient un-broadcasting is a `sum`; a
`mean`-loss backward scales by `1/count`), they share almost all
machinery (`mean` is `sum / count`, §7), and their forward semantics are
unambiguous — a pure accumulation with an additive identity (`0`) that
makes a fresh zero-initialized output the natural starting point (§6).

**Deferred to later reductions milestones:** `max`, `argmax`, `min`,
`product`.

- **`min` / `product`** are structurally like `sum` (associative binary
  fold) and could follow easily, but they are not needed for the autograd
  path and are left out to keep v1.19 tight. `product`'s backward also
  needs the forward output, adding coupling best designed on its own.
- **`max` / `argmax` should probably not be first.** They break the two
  properties that make `sum`/`mean` clean:
  - **No additive identity for a zero-init output.** `max` needs a
    `-inf` seed and a "have I written this output cell yet?" notion, so
    the fresh-zeros-then-accumulate pattern (§6) does not apply
    unchanged.
  - **`argmax` returns indices, not values.** The native runtime is
    float64-only (§ dtype notes); an index-typed output is a new output
    *kind*, not just a new op, and it interacts with dtype/device
    metadata (A4). Its backward is also a scatter (gradient flows only to
    the arg-max position), a different backward shape from `sum`.

  So `max`/`argmax` get their own later design once `sum`/`mean` and A4
  (dtype/device metadata) are settled.

## 4. Reduction semantics

The semantics follow NumPy, restricted to the runtime's existing
constraints (float64, positive-int dims):

- **`axis=None`** reduces **all** elements to a single value.
- **A single integer `axis`** reduces exactly that one dimension.
- **Negative `axis`** is supported and normalized as `axis + ndim`
  (NumPy convention), so `axis=-1` is the last dimension. This is
  user-facing sugar worth having; it is validated after normalization
  (§9). (Note the low-level view ops like `narrow` take a non-negative
  `dim`; reductions are a higher-level, user-facing surface where
  negative axes are idiomatic, so they are accepted here and normalized
  before any traversal.)
- **`keepdims=True`** leaves each reduced axis in the output shape as a
  size-1 dimension; **`keepdims=False`** (default) removes it.
- **Scalar output** has shape `()` (the runtime's existing scalar shape,
  `ndim == 0`, `numel == 1`).
- **Reducing a scalar** (`shape ()`) with `axis=None` returns shape `()`
  unchanged (the sum/mean of one element is that element).
- **Reducing a scalar with an integer axis** is invalid (a scalar has no
  axes) and raises (§9).
- **Multiple axes** (`axis` as a tuple) are **out of first scope** — a
  single `int` or `None` only. Tuple-axis is a clean later extension
  (reduce those axes together, or iteratively) and is called out in §12,
  not built in v1.19.
- **Empty / zero-size tensors do not arise.** The runtime already
  rejects zero-size dimensions at construction (v0.7 rule), so there is
  no empty-reduction case to define — `sum` of an empty axis (NumPy's
  `0`) and `mean` of an empty axis (NumPy's `nan`-with-warning) simply
  cannot be reached. If zero-size shapes are ever added, their reduction
  identities get designed then.

## 5. Output-shape rules, with examples

Given input rank `n` and a (normalized, valid) `axis`:

- **`axis=None`** → shape `()` when `keepdims=False`; shape
  `(1,) * n` when `keepdims=True` (NumPy's rule).
- **integer `axis`** → the input shape with that axis **removed**
  (`keepdims=False`) or **set to 1** (`keepdims=True`).

Worked examples (the v1.19 tests will pin these down):

| call | result shape |
|----|----|
| `(2, 3).sum()` | `()` |
| `(2, 3).sum(axis=0)` | `(3,)` |
| `(2, 3).sum(axis=1)` | `(2,)` |
| `(2, 3).sum(axis=1, keepdims=True)` | `(2, 1)` |
| `(2, 3).sum(keepdims=True)` | `(1, 1)` |
| `().sum()` | `()` |
| `(2, 3, 4).mean(axis=-1)` | `(2, 3)` |
| `(2, 3, 4).mean(axis=-1, keepdims=True)` | `(2, 3, 1)` |

The output shape is inferred by a pure helper (analogous to
`broadcast_shapes`) that can be unit-tested without the compiled backend
(§10, §13).

## 6. Traversal strategy

The output is **always freshly allocated row-major contiguous native
storage** (`NativeTensorCore.zeros(out_shape)`), so it starts at the
additive identity `0` — no fill pass needed (native storage is
zero-initialized). The **input is read through its existing
shape/stride/offset metadata**, exactly like every other native kernel,
so contiguous, transposed, narrowed, and nonzero-offset inputs all work
without being materialized first.

The clean, layout-independent formulation is a **scatter-accumulate that
mirrors broadcasting** (§8):

- Walk the **input** with the existing odometer (its shape, its real
  strides, its offset), visiting every logical element once. This is the
  same traversal `tf_core_binary` / materialization already use.
- Maintain, alongside the input read position, an **output write
  position** advanced by a second stride vector — the *reduction output
  strides* — computed in Python: for each input axis, the row-major
  stride of that axis **in the output**, or **`0`** if that axis is being
  reduced. A zero output-stride means "every step along this input axis
  writes back to the same output cell" — i.e. those elements accumulate
  together. That is the reduction.
- At each step, `out[out_pos] += in[in_pos]`.

This is the exact dual of broadcasting: **broadcasting reads with zero
strides** (one input element feeds many outputs); **reduction writes with
zero strides** (many input elements feed one output). The same odometer
machinery drives both — reductions add a write-side stride vector where
broadcasting added a read-side one. It is honest to note this symmetry:
it means the future C++ kernel is a small variant of the traversal the
runtime already trusts, not a new algorithm.

- **`axis=None`** is the degenerate case where **all** output strides are
  `0`: every input element accumulates into the single scalar cell
  `out[0]`. No special-casing beyond building an all-zero output-stride
  vector.
- **`keepdims`** changes only the *shape* of the output (and hence its
  row-major strides), not the traversal: a kept size-1 axis contributes a
  `0` output-stride (its stride is irrelevant since the extent is 1),
  which coincides with the reduced-axis rule.

No input view is materialized, and NumPy is **never** called for the
compute (§ error behavior). `mean` runs `sum` and then divides (§7).

## 7. Numerical behavior

Reductions are where floating-point honesty matters, so this is explicit:

- **Deterministic, straightforward loop order.** v1.19 accumulates in the
  order the input odometer visits elements (row-major over the *input*
  logical shape). The order is deterministic for a given input layout.
- **Floating-point sums are order-sensitive.** Because addition is not
  associative in float64, a different traversal order (or NumPy's own
  pairwise summation) can produce a result differing in the last few
  ULPs. Therefore this design **does not claim bit-for-bit equality with
  NumPy** for all shapes/layouts — only that results agree to a sensible
  floating-point tolerance. (Small integer-valued cases will often match
  exactly, but that is not promised in general.)
- **Tests use tolerances.** The v1.19 suite compares against NumPy with
  `np.allclose` at a sensible tolerance (e.g. `atol=1e-10`/`rtol`),
  **not** `np.array_equal`, for reduction values — the opposite of the
  exact-equality the broadcasting/fast-path tests could use, and for a
  principled reason (order-sensitivity), not to hide a bug.
- **No fancy summation in first scope.** No Kahan compensation, no
  pairwise/tree reduction, no SIMD, no reassociation tricks. A plain
  sequential accumulation. Any of those is a deliberate, separately
  designed later choice (and would come with its own accuracy note), not
  something slipped in.
- **`mean` is `sum / count`** using the float64 arithmetic
  `NativeTensorCore` already uses: `count` is `numel` for `axis=None`, or
  `shape[axis]` for a single axis. The division happens once per output
  element after the sum, in float64.

## 8. Relation to broadcasting and future autograd

This records the relationship; it implements no autograd.

- **Broadcasting forward** (A2) reads repeatedly through **zero strides**:
  a `(D,)` operand added to `(N, D)` is read `N` times, once per row.
- **Broadcasting backward** must therefore **sum the upstream gradient
  over the broadcast axes** and reshape to the operand's original shape —
  e.g. the gradient of that `(D,)` operand is the column-sum of the
  `(N, D)` upstream gradient. That is a native **reduction**.
- Hence **reductions are a hard prerequisite for native autograd.** Once
  `sum` exists over arbitrary axes with `keepdims`, the un-broadcast step
  the NumPy `Tensor` already performs has a native equivalent, and a
  native backward for `add`/`subtract`/`multiply` becomes expressible.
- The duality is not a coincidence: broadcasting is a "copy" (zero read
  strides) and its adjoint is a "sum" (zero write strides), §6. The
  forward of one is the transpose of the backward of the other.

**None of this autograd machinery is built in v1.19.** Reductions ship as
forward-only ops; the backward story is the following phase's, recorded
here and in the broadcasting design so the eventual native autograd is
coherent.

## 9. Interaction with existing v1.14 / v1.17 behavior

Reductions are **additive** — they touch none of the existing paths:

- **The v1.14 contiguous fast path is unchanged.** It is an elementwise
  same-shape optimization; reductions are a different kernel and dispatch.
- **The v1.17 broadcasting implementation is unchanged.** `broadcast_shapes`
  / `_broadcast_strides` and the three-way `_binary_core_op` dispatch
  stay exactly as they are. (Reductions *reuse the idea* of a zero-stride
  vector, but on the write side and in their own method — they do not
  modify the binary-op code.)
- **The generic odometer traversal is unchanged.** Reductions add a new
  traversal (scatter-accumulate); they do not alter `tf_core_binary`,
  `tf_core_relu`, materialization, or matmul.
- **Reductions are new explicit methods**, never operator overloads — the
  native runtime has no operator sugar and this milestone adds none.

## 10. Proposed API (for v1.19)

```python
NativeTensorCore.sum(axis=None, keepdims=False)   -> NativeTensorCore
NativeTensorCore.mean(axis=None, keepdims=False)  -> NativeTensorCore
NativeTensor.sum(axis=None, keepdims=False)       -> NativeTensor
NativeTensor.mean(axis=None, keepdims=False)      -> NativeTensor
```

- Each returns a **new owning** core/tensor over freshly allocated
  contiguous storage; the input is unchanged.
- `NativeTensor.sum`/`mean` **delegate** to the core method and re-wrap
  the result (like `relu`/`add` do), so the wrapper inherits reductions
  with no reduction-specific logic — it stays a thin forward-only wrapper.
- A pure Python shape/stride helper (e.g. `reduce_shape(shape, axis,
  keepdims)` returning the output shape, and a companion producing the
  reduction output-strides of §6) lives beside `broadcast_shapes` and the
  other metadata helpers in `cpp.py`, testable without the built backend.
- **Explicit native backend** (`get_backend("native")`): add `sum`/`mean`
  there **only if** it fits the current backend surface (which today is
  `add`/`relu`/`matmul` + conversions). Reductions on the explicit
  backend are optional and can wait; the core + wrapper methods are the
  deliverable. No implicit dispatch is added either way.

## 11. Error behavior

- **Invalid axis** raises a clear `ValueError` naming **both the axis and
  the shape** — e.g. `"axis 3 is out of bounds for a tensor of shape
  (2, 3) (ndim 2)"`. This covers `axis >= ndim`, `axis < -ndim`, and any
  integer axis on a scalar (`ndim 0`).
- **Wrong axis type** raises `TypeError` — `axis` must be `None` or an
  `int` (not a float, not a string; `bool` is rejected like elsewhere in
  the runtime, since `True`/`False` are not meaningful axes).
- **`keepdims` must be `bool`** — a non-bool raises `TypeError`.
- **Closed tensors keep their existing behavior**: `sum`/`mean` on a
  closed `NativeTensorCore` / `NativeTensor` raise `RuntimeError` via the
  existing `_require_open()` gate, before any traversal.
- **No silent NumPy fallback, ever.** If a reduction case is unsupported
  (e.g. tuple axis in v1.19), it raises a clear error — it does not
  quietly compute the answer with NumPy. This preserves the governing
  rule from [dispatch_design.md](dispatch_design.md).

## 12. Out of scope for the first implementation

Explicitly deferred, to keep v1.19 tight:

- **autograd / backward** for reductions (the relationship is §8, not
  built).
- **`max` / `argmax` / `min` / `product`** (§3) — later reduction
  milestones.
- **tuple / multiple axes** — single `int` or `None` only in v1.19.
- **`Tensor` integration** — `NativeTensorCore` / `NativeTensor` stay
  separate from `tensorforge.Tensor`.
- **CUDA** — CPU float64 only.
- **dtype promotion** — everything is float64; no accumulator-dtype rules
  (NumPy sometimes upcasts small int dtypes; not applicable here).
- **operator overloads** — reductions are methods, not sugar.
- **distributed / fused reductions** — out of Phase A entirely.
- **numerically advanced summation** (Kahan, pairwise, SIMD, §7) — a
  deliberate later choice if ever adopted, not first scope.

## 13. Test plan (for v1.19, not this milestone)

Values compared against NumPy with **tolerances** (`np.allclose`), for
the order-sensitivity reason in §7; shapes compared exactly.

- **`sum` / `mean` of all elements** (`axis=None`) on 1-D, 2-D, 3-D.
- **`axis=0` and `axis=1`** on a 2-D tensor, values and shapes.
- **negative axis** (`axis=-1`) equals the positive equivalent.
- **`keepdims=True` and `False`** — output shapes per §5, including the
  `axis=None, keepdims=True` all-ones case.
- **scalar reductions** — `().sum()` / `().mean()` return shape `()` with
  the right value; a scalar with an integer axis raises.
- **transposed-view reductions** — `a.T.sum(axis=...)` matches
  `numpy(a.T).sum(axis=...)`, proving layout independence.
- **narrowed / nonzero-offset view reductions** — a `narrow`ed operand
  (nonzero offset) reduces correctly.
- **reduction after a broadcasted elementwise result** — e.g.
  `(a.add(bias)).sum(axis=0)` where `bias` broadcast, chaining A2 into A3.
- **output is row-major contiguous** (`result.contiguous is True`,
  expected strides).
- **input unchanged** — reducing does not mutate the operand's storage.
- **`NativeTensor` wrapper inheritance** — `NativeTensor.sum`/`mean`
  delegate to the core and return a new owning wrapper, proven with no
  wrapper-specific behavior.
- **error cases** — invalid axis (naming axis + shape), non-int axis
  (`TypeError`), non-bool `keepdims` (`TypeError`), closed-tensor
  (`RuntimeError`).
- **pure-helper unit tests** — `reduce_shape(...)` on the §5 table,
  including the error rows, runnable without the built backend.

No test asserts anything about speed.

## 14. Benchmark plan

Following the established `benchmarks/cpp_backend.py` philosophy —
correctness verified before timing, medians after warmup,
hardware-dependent, **no performance assertions anywhere**:

- Compare **NumPy** vs **`NativeTensorCore`** vs **`NativeTensor`** for
  `sum` (and optionally `mean`).
- Separate **`sum`-all** (`axis=None`) from an **axis-specific**
  reduction, since their traversals and output sizes differ.
- Include a **contiguous** input and a **non-contiguous** (transposed)
  input row, so the layout-independence cost is visible — the reduction
  reads a strided view directly, like the elementwise view rows.
- Document the overhead honestly (NumPy's reductions are optimized C with
  pairwise summation; the native scatter-accumulate is a plain loop). The
  benchmark **is not run by pytest** — a lightweight test only checks the
  plan/row structure, as today.

These rows are a v1.19 concern; this document states the plan so the
numbers are framed the way A1's (v1.15) and A2's were.

## 15. Fit in the Daedalus-class roadmap

Reductions are the third step of **Phase A — native CPU runtime**:

- **A1 — contiguous elementwise fast path** — complete (design v1.13,
  implementation v1.14, benchmark impact v1.15).
- **A2 — broadcasting** — complete (design v1.16, implementation v1.17).
- **A3 — reductions** — this design (v1.18) → implementation (v1.19),
  **complete** for `sum`/`mean`; `max`/`argmax`/`min`/`product` in later
  reduction milestones. Next is A4, whose design milestone is v1.20.
- **A4 — dtype / device metadata** (beyond float64-CPU-only) — closes out
  Phase A.

Only once the CPU runtime reduces (and carries dtype/device metadata)
does the roadmap move on to **native autograd** — for which reductions
are the prerequisite (§8) — then the **native training stack**, the
**CUDA runtime**, the **AMP / Tensor Core** path, **Transformer / text**
examples, **distributed / DDP**, and a final **benchmark / profiling /
docs** polish (the final portfolio release). Each phase lands only when
the previous one is tested and documented; the Python framework remains
the reference implementation throughout, and reductions are specified
against NumPy (to a documented floating-point tolerance) so that
reference stays meaningful.
