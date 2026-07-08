# Native broadcasting — design

This began as a **design document** (written in v1.16, ahead of any code)
specifying broadcasting for the native elementwise kernels
(`add`/`subtract`/`multiply`) in the `NativeTensorCore` layer.
**Status: implemented in v1.17.**

The implementation followed this design with one honest simplification
worth stating up front: **no new C++ kernel was needed.** The existing
generic odometer kernel (`tf_core_binary`, exported as
`tf_core_add`/`subtract`/`multiply`) already walks the *output* shape
advancing each operand by its own per-axis stride — so it is already
broadcast-capable when fed **zero-augmented broadcast strides**, exactly
as §6 anticipated ("the kernel then walks the output shape with an
odometer … a zero stride means *do not move*"). Broadcasting therefore
lives entirely in Python: a pure `broadcast_shapes` helper (§5) infers
the output shape, a small `_broadcast_strides` helper builds each
operand's read-strides (real stride on genuine axes, stride 0 on
stretched/left-padded axes), and `NativeTensorCore._binary_core_op`
dispatches three ways (§8). The C++ layer was **not changed** — the
same-shape fast path (v1.14) and same-shape odometer stay byte-for-byte
as they were, and the broadcast path reuses the odometer with different
arguments. Output is freshly allocated row-major contiguous storage;
`NativeTensor` inherited broadcasting with no wrapper edit; results match
NumPy exactly. No reductions, autograd, `Tensor` integration, CUDA,
dtype promotion, operator overloads, or matmul broadcasting came with it.
The sections below remain the design of record.

For where this sits, see [backend_experiments.md](backend_experiments.md)
(the native runtime and benchmarks),
[native_contiguous_fast_path_design.md](native_contiguous_fast_path_design.md)
(the v1.14 fast path this must coexist with), and
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md) (the
wrapper that will inherit the improvement).

## 1. Why broadcasting is the next step (Phase A2)

Phase A1 made contiguous elementwise ops fast; it did not make them more
*capable*. The native runtime still refuses any elementwise pairing whose
shapes are not already identical — a `(3, 4)` and a `(3, 1)` raise, a
scalar and a matrix raise. That is the single largest expressiveness gap
between the native elementwise path and the NumPy-backed `Tensor`
frontend (whose autograd already un-broadcasts gradients), and it blocks
almost everything above it: a bias add `(N, D) + (D,)`, a per-row scale
`(N, D) * (N, 1)`, any use of a scalar operand. Reductions (A3) and
richer dtype/device metadata (A4) are worth less until the elementwise
core can broadcast, because broadcasting is the shape algebra those build
on. So A2 is broadcasting.

The contiguous fast path (A1) and broadcasting (A2) are complementary,
not competing: A1 optimized the *traversal* of same-shape operands; A2
widens *which* shapes are legal. A1 stays exactly as it is for the
same-shape contiguous case (§8).

## 2. Current limitation (the exact-shape rule today)

`NativeTensorCore._binary_core_op` performs a strict equality check
before any kernel runs:

```python
if self.shape != other.shape:
    raise ValueError(
        f"{op_name} requires identical shapes (no broadcasting), "
        f"got {self.shape} and {other.shape}"
    )
```

There is no broadcasting anywhere in the native runtime — not in the
kernels (`tf_core_add`/`tf_core_binary` and their `_contiguous`
variants), not in the metadata layer, not in the `NativeTensor` wrapper.
This is deliberate and documented (see
[dispatch_design.md](dispatch_design.md), "Why no silent switching"): the
native backend's exact-shape contract is *explicit*, not an accident, and
NumPy's broadcasting is intentionally not mirrored until it is designed.
This document is that design.

## 3. Target behavior (what v1.17 should accept)

Broadcasting makes two shapes compatible by aligning them from the
**trailing** dimension and treating any axis of length 1 as stretchable
to the other operand's length on that axis. The cases in first scope:

- **scalar `+` tensor** — `() + (3, 4) -> (3, 4)`
- **tensor `+` scalar** — `(3, 4) + () -> (3, 4)`
- **same-rank broadcasting** — `(3, 1) + (1, 4) -> (3, 4)`
- **rank-padding with leading 1s** — the lower-rank operand is padded on
  the *left* with size-1 axes until the ranks match, then aligned:
  `(4,) + (3, 4) -> (3, 4)` (the `(4,)` acts as `(1, 4)`).
- **both operands broadcasting** — `(1, 3, 1) + (2, 1, 5) -> (2, 3, 5)`

Worked examples (each is a target the v1.17 tests will pin down):

| a | b | result | note |
|----|----|----|----|
| `()` | `(3, 4)` | `(3, 4)` | scalar stretches over everything |
| `(3, 4)` | `()` | `(3, 4)` | symmetric |
| `(3, 1)` | `(1, 4)` | `(3, 4)` | classic outer-style broadcast |
| `(4,)` | `(3, 4)` | `(3, 4)` | left-pad `(4,)` to `(1, 4)` |
| `(1, 3, 1)` | `(2, 1, 5)` | `(2, 3, 5)` | both sides stretch, on different axes |
| `(2, 3)` | `(4, 3)` | — | **error**: axis 0 is 2 vs 4, neither is 1 |

Broadcasting stays commutative in shape (`a`⊕`b` and `b`⊕`a` infer the
same result shape) even though the arithmetic itself is not (subtract).

## 4. Compatibility rules

Given shapes `a_shape` and `b_shape`:

1. **Left-pad** the shorter shape with leading `1`s until both have the
   same rank `r`. (Padding is conceptual — no data moves.)
2. For each aligned axis `d` in `0 … r-1`, the two extents
   `a_d`, `b_d` are **compatible** iff `a_d == b_d`, or `a_d == 1`, or
   `b_d == 1`.
3. If every axis is compatible, the shapes broadcast; otherwise the pair
   is rejected (§10).

The **result extent** on axis `d` is `max(a_d, b_d)` (equivalently: the
non-1 side, or 1 when both are 1). These are exactly NumPy's rules,
restricted to the runtime's existing constraint that every real extent is
a positive int (zero-size dimensions remain unsupported, as in v0.7).

## 5. Output-shape inference

Output-shape inference is the pure function implied by §4 — no storage,
no kernel:

```
broadcast_shapes(a_shape, b_shape) -> tuple[int, ...]
```

It left-pads to equal rank, takes `max` per axis after the compatibility
check, and raises a `ValueError` naming *both original* shapes on the
first incompatible axis. It belongs beside the existing pure metadata
helpers (`row_major_strides`, `numel`, `is_contiguous_shape`,
`shape_info`) in `cpp.py`, which already never touch the compiled library
— so shape inference is testable whether or not the backend is built, and
the kernel dispatch (§6) calls it after operand validation and before
allocating output. The scalar shape `()` broadcasts against anything
(it left-pads to all-1s), giving the scalar cases in §3 for free.

## 6. Broadcasted stride model (the core idea)

Broadcasting must **not materialize** an expanded operand — that would
defeat its purpose (a `(1,)` broadcast to `(1_000_000,)` would allocate a
million elements). The standard, allocation-free technique is a
**zero-stride view**:

- To stretch operand `a` from its (left-padded) shape to the output
  shape, build a *logical* set of read-strides `a_bstrides` of length `r`
  (the output rank): for each output axis `d`, if `a`'s padded extent on
  `d` is the full output extent, use `a`'s real stride there; if it is
  `1` (and the output extent is `> 1`), use **stride 0** so every step
  along that output axis re-reads the same element; the padded leading
  axes likewise contribute stride 0.
- The kernel then walks the **output** shape with an odometer (exactly
  the existing `tf_core_binary` traversal), advancing each operand's
  position by *its* broadcast stride per axis. A zero stride means "do
  not move" — that is broadcasting, expressed entirely in the stride
  vector, with no expanded buffer.

The **output is always newly allocated row-major contiguous native
storage** of the inferred shape (`NativeTensorCore.zeros(out_shape)`),
identical to how every elementwise op already produces its result. Only
the *inputs* are read through broadcast strides; the output side never
uses zero strides.

Concretely, `_binary_core_op` would, on the broadcasting path:

1. `out_shape = broadcast_shapes(self.shape, other.shape)` (§5).
2. Compute `a_bstrides`, `b_bstrides` (length `len(out_shape)`) from each
   operand's real shape/strides via the padding-and-zero rule above.
3. `out = NativeTensorCore.zeros(out_shape)`.
4. Call a broadcast-capable kernel with `out_shape`, `a_bstrides`,
   `b_bstrides`, `a_offset`, `b_offset`, `ndim = len(out_shape)`.

## 7. Interaction with existing NativeTensorCore metadata

Broadcasting reuses the v0.7 metadata contract without changing it:

- **shape** — the operands' real shapes are the inputs to inference;
  the result core's `shape` is the inferred `out_shape`.
- **strides** — the operand's *real* strides feed the broadcast-stride
  computation; the injected zeros are **kernel arguments only**, never
  stored on any core. The result core has ordinary row-major strides
  (it is freshly allocated contiguous storage).
- **offset** — each operand's real offset is passed through as the
  kernel's starting position, exactly as today. Broadcast strides are
  layered on top of the offset; a broadcast operand can carry a nonzero
  offset (§9).
- **contiguous** — unchanged in meaning (`strides == row_major_strides`).
  A broadcast *read* is conceptually non-contiguous (it has zero
  strides), but no core is ever marked contiguous/non-contiguous *because
  of* broadcasting — the zero-stride vector is ephemeral. The output core
  is contiguous by construction.
- **numel** — the result's `numel` is `numel(out_shape)`, which can
  exceed either input's `numel` (that is the whole point: a `(1, 4)`
  operand contributes to a `(3, 4)` output). No operand's own `numel`
  changes.

No new persistent metadata field is introduced. Broadcasting is a
*per-call* computation of read strides, not a new core state.

## 8. Interaction with the v1.14 contiguous fast path

The dispatch order in `_binary_core_op` becomes a three-way choice, and
A1's fast path is preserved untouched for its case:

1. **Same shape + both contiguous → v1.14 flat fast path.** Unchanged.
   The most common case (two identically-shaped contiguous tensors) still
   takes the index-free pointer loop; broadcasting adds nothing to it.
2. **Same shape, at least one non-contiguous → existing generic
   odometer.** Unchanged (this is today's fallback).
3. **Different shapes that broadcast → new broadcast traversal.** The
   generic odometer over `out_shape` with per-operand broadcast strides
   (§6).

So the decision is: if shapes are equal, behave exactly as today
(cases 1–2); only when shapes differ do we infer a broadcast shape and
take case 3. This keeps A1's win intact and confines the new code to a
path that currently just raises.

**Specialized broadcast fast paths are explicitly out of first scope.** A
broadcasted operand is read with zero strides, so it is not contiguous in
the flat-loop sense, and the first implementation should use the generic
broadcast odometer for correctness. Later milestones *may* add
specialized loops (e.g. scalar-operand broadcast, or "b is contiguous and
a is a scalar" specializations) once the generic path is trusted and the
benchmarks (§13) justify them — the same design-then-optimize cadence A1
followed (measure in v1.12 → design in v1.13 → implement in v1.14).

## 9. Interaction with non-contiguous views

Broadcasting composes with the existing view machinery because both are
expressed purely in strides/offset, and the kernel already reads
arbitrary strided layouts:

- **Transposed views.** A transposed operand contributes its *real*
  (permuted) strides on the axes it fills and zeros where it is
  broadcast. `a.T` broadcast against `b` reads `a` transposed and
  stretched in one traversal — no materialization, exactly as
  transposed elementwise already works today.
- **Narrowed views.** A narrowed operand contributes its narrowed shape
  (some axis shorter) and its unchanged strides; broadcasting then
  stretches any size-1 axis as usual. A `narrow` that produces a size-1
  axis is a legitimate broadcastable operand.
- **Nonzero offsets.** Passed through unchanged as the kernel's start
  position; a broadcast operand may be a row slice with a nonzero offset
  (the fast path already handles nonzero offsets, §8-case-1, and the
  broadcast path inherits the same base-pointer handling).
- **Negative strides.** Representable today only via the low-level
  `NativeTensorView` constructor (no current high-level op — `transpose`,
  `narrow`, `reshape` — produces them), and the odometer already
  traverses them correctly. Broadcasting does not special-case sign: a
  real axis keeps its (possibly negative) stride, and only *broadcast*
  axes get stride 0. So negative strides, where they occur, keep working
  under broadcasting with no extra rules. This stays a representable-but-
  not-generated corner, honestly noted rather than expanded.

In all four cases the rule is the same: **real axes keep their real
strides; only size-1/padded axes that are being stretched get stride 0.**

## 10. Error behavior

- **Shape mismatch** raises `ValueError` naming **both** original
  operand shapes and, ideally, the offending axis — e.g.
  `"cannot broadcast shapes (2, 3) and (4, 3): axis 0 has sizes 2 and 4
  (neither is 1)"`. The message reports the shapes as the user passed
  them, before any left-padding, so it is not mysterious.
- **Non-`NativeTensorCore` operand** still raises `TypeError` naming the
  expected type, exactly as today — unchanged and checked *before* shape
  inference.
- **Closed core/storage** still raises `RuntimeError` via
  `_require_open()`, before any inference or traversal.
- **No silent NumPy fallback, ever.** If the native path cannot compute a
  broadcast (it never should, once implemented, but if some case is out
  of scope), it raises — it does not quietly hand the operation to NumPy.
  This preserves the governing rule from
  [dispatch_design.md](dispatch_design.md).

The exact-shape `ValueError` message that exists today
(`"requires identical shapes (no broadcasting)"`) is **removed** when
broadcasting lands — same-shape stays valid, incompatible shapes get the
new broadcast-specific message. Any existing test asserting the old
"no broadcasting" wording is updated in v1.17 (this is called out here so
the implementation milestone expects it).

## 11. First implementation scope (v1.17)

**In scope:** broadcasting for `add`, `subtract`, `multiply` — the three
elementwise binary ops already routed through `_binary_core_op`. All
three share one plumbing change (shape inference + broadcast strides +
one broadcast-capable kernel), so they land together.

**`divide`:** only if it is *already* naturally present in the native
tensor-core path. It is **not** today — the native core exposes
`add`/`subtract`/`multiply` (and `matmul`); `divide` exists only as a
raw-buffer kernel (`tf_elementwise_divide`), not as a `NativeTensorCore`
method. So v1.17 does **not** add native `divide` broadcasting; adding
`divide` to the core is a separate, later decision and is not smuggled in
under broadcasting.

**`relu`** is unary — no broadcasting applies; it is untouched.

## 12. Out of scope for the first implementation

Explicitly deferred, to keep v1.17 tight:

- **reductions** (sum/mean/max) — Phase A3, their own traversal and
  numerical-order design.
- **matmul broadcasting / batched matmul** — matmul stays strictly 2-D
  `(m, n) @ (n, p)`; no batch dimensions, no broadcasting of batch axes.
- **autograd** — no backward for broadcasted ops (the gradient
  implication is noted in §14, not built).
- **Tensor integration** — `NativeTensorCore`/`NativeTensor` stay
  separate from `tensorforge.Tensor`.
- **CUDA** — CPU float64 only.
- **operator overloads** — compute stays method-only (`a.add(b)`), no
  `a + b`.
- **dtype promotion** — everything is float64; no mixed-dtype rules.
- **specialized broadcast fast paths** — the generic broadcast odometer
  first (§8); optimizations later, benchmark-justified.

## 13. Test plan (for v1.17, not this milestone)

Correctness against NumPy is the spec — NumPy *is* the broadcasting
reference. The v1.17 suite should cover, each `np.array_equal` against the
NumPy result of the same op:

- **scalar broadcasting** — `() + (3, 4)`, `(3, 4) * ()`, both operand
  orders.
- **vector/matrix broadcasting** — `(4,) + (3, 4)`, `(3, 1) * (3, 4)`.
- **leading-dimension broadcasting** — `(1, 4) + (3, 4)`,
  `(4,) - (3, 4)` (left-pad path).
- **both operands broadcasting** — `(3, 1) + (1, 4) -> (3, 4)`,
  `(1, 3, 1) * (2, 1, 5) -> (2, 3, 5)`.
- **non-contiguous broadcast inputs** — a transposed operand and/or a
  narrowed operand participating in a broadcast, checked against the
  NumPy equivalent on the materialized values.
- **nonzero-offset broadcast inputs** — a broadcast operand that is a
  `narrow(0, …)` row slice (nonzero offset) still reads from the right
  base.
- **exact-shape path still works** — same-shape `add`/`subtract`/
  `multiply` (contiguous and strided) unchanged, so the A1 fast path and
  the generic same-shape path are both preserved (regression guard).
- **incompatible shapes raise** — `(2, 3)` vs `(4, 3)` raises
  `ValueError` naming both shapes; `TypeError`/`RuntimeError` for bad
  operand type / closed core unchanged.
- **`broadcast_shapes` unit tests** — the pure inference function on the
  §3 table, including the error rows, runnable without the built backend.
- **`NativeTensor` inherits it** — a wrapper-level test showing
  `NativeTensor.add` broadcasts (e.g. `(3, 1) + (1, 4)`) with **no
  wrapper code change**, proving the improvement rides through
  `NativeTensorCore` (§ wrapper design).

No test asserts anything about speed.

## 14. Benchmark plan

The existing `benchmarks/cpp_backend.py` philosophy carries over
unchanged: correctness verified before timing, medians after warmup,
hardware-dependent, **no performance assertions anywhere**. For
broadcasting specifically, the honest comparison to document is:

- the **exact-shape contiguous fast path** (A1) versus the **broadcast
  traversal** for a representative case (e.g. `(N, D) + (D,)` bias add
  versus the same as a materialized `(N, D) + (N, D)`), so the cost of
  the zero-stride odometer is visible beside the flat loop;
- the point being made is *characterization* — broadcasting trades a
  generic strided traversal for the convenience of not materializing the
  stretched operand, and the benchmark shows that trade honestly, not a
  speedup claim.

Adding these rows is a v1.17 concern; this document only states the plan
so the numbers are framed the same way A1's were (v1.15).

## 15. Future autograd implications (noted, not built)

When native autograd eventually exists (a later phase; see
[dispatch_design.md](dispatch_design.md)), broadcasting has a specific
backward requirement worth recording now so it is not a surprise:

> The gradient of a broadcasted operand must be **reduced (summed) over
> the axes that were broadcast**, then reshaped back to the operand's
> original shape.

This is exactly what the NumPy-backed `Tensor` already does in
`_accumulate_grad` (its "un-broadcast" step). For example, in
`(N, D) + (D,)`, the `(D,)` operand received the value along every row,
so its gradient is the column-sum of the `(N, D)` upstream gradient. A
zero-stride *forward* read corresponds to a **sum-reduction** on the
*backward* pass — which is also why reductions (A3) are a natural
prerequisite for native autograd. **None of this is implemented in the
broadcasting milestone;** it is a forward-only feature, and the note
exists only to keep the eventual backward story coherent.

## 16. Fit in the Daedalus-class roadmap

Broadcasting is the second step of **Phase A — native CPU runtime**:

- **A1 — contiguous elementwise fast path** — complete (design v1.13,
  implementation v1.14, benchmark impact v1.15).
- **A2 — broadcasting** — this design (v1.16) → implementation (v1.17),
  **complete**. Next is A3, whose design milestone is v1.18.
- **A3 — reductions** (sum/mean/max), whose backward is the mirror image
  of broadcasting's forward (§15).
- **A4 — dtype / device metadata** (beyond float64-CPU-only).

Only once the CPU runtime broadcasts, reduces, and carries dtype/device
metadata does the roadmap move on to **native autograd**, then the
**native training stack**, the **CUDA runtime**, the **AMP / Tensor
Core** path, **Transformer / text** examples, **distributed / DDP**, and
a final **benchmark / profiling / docs** polish. Each phase lands only
when the previous one is tested and documented; the Python framework
remains the reference implementation throughout, and broadcasting is
specified here to match NumPy so that reference stays exact.
