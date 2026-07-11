# Release history

A short summary of how TensorForge grew, milestone line by milestone.
Details live in the docs and the test suite; this is the map.

## v0.x — Autograd and basics

The Tensor and the reverse-mode autograd engine: elementwise ops,
matmul, exp/log/tanh/sigmoid/relu/softmax, broadcasting-aware
gradients. The module system (Parameter, Module, Linear, activations,
Sequential), SGD, MSE and cross-entropy losses, and the first three
examples — linear regression, XOR, and the multi-class spiral.

## v1.x — Training stack and framework utilities

Everything a real training loop needs: the accuracy metric, Adam,
mini-batching, gradient checking against finite differences, save/load
parameters, model summaries and parameter counting, frozen parameters,
train/validation splitting, evaluation helpers, binary cross-entropy
with a binary classification example, and checkpoints that capture
optimizer state so training can resume exactly.

## v2.x — Regularization, CNN support, and release readiness

Train/eval mode and Dropout (with an example that measures honestly in
eval mode), eval-safe evaluators, BatchNorm1d with module buffers,
gradient clipping, the StepLR scheduler and scheduler checkpointing,
the docs set, image-shaped input (Conv2d, MaxPool2d, Flatten, and the
tiny CNN example), LayerNorm, optional RNG state in checkpoints for
bit-exact dropout resume, and this release-readiness pass.

## v3.0 — Portfolio Release

The completed Python framework line: consistent docs, a clean landing
page, a project summary, CI, and guardrail tests that keep docs and
code from drifting apart.

## Advanced branches — after v3.0

The experimental line, separate from the finished Python framework.
**C++ v0.1** built the first proof: a tiny compiled elementwise-add
kernel called from Python through ctypes. **v0.1.1** made CI build and
hard-verify the compiled backend. **v0.2** grew the kernel family to
add/subtract/multiply/divide/ReLU. **v0.3** added a naive 2-D matmul
kernel. **v0.4** added honest benchmarks against NumPy — which NumPy
mostly wins, as expected. **v0.5** cleaned up the backend API with
introspection helpers (`is_available`, `list_kernels`,
`backend_info`) and lazy library loading. **v0.6** added
`matmul_tiled`, a cache-blocking optimization experiment benchmarked
against the naive reference and NumPy. **v0.7** added the
shape/stride metadata layer. **v0.8** added `NativeStorage`, a
C++-owned float64 buffer. **v0.9** bound the two together as
`NativeTensorView`, with native contiguous materialization as its
first operation. **v1.0** composed storage + view into
`NativeTensorCore`, the first native tensor runtime object. **v1.1**
gave it metadata-only view operations — reshape, transpose/T, narrow —
sharing storage without copying. **v1.2** made the runtime
self-contained for simple compute: relu/add/subtract/multiply as
native kernels reading strided views directly. **v1.3** completed the
compute set with TensorCore matmul over strided views. **v1.4**
upgraded the benchmarks into a suite covering NumPy, the raw-buffer
kernels, and the TensorCore runtime side by side — overheads
included, honestly. **v1.5** added the backend dispatch design
([dispatch_design.md](dispatch_design.md)) and Stage 1 of it: an
explicit backend API (`get_backend("numpy"|"native")`) with no
implicit routing. **v1.6** polished that API's conversion contract —
`to_numpy` as the explicit exit boundary matching `tensor_from_array`,
consistent operand errors, and tested exact-shape native behavior
against NumPy's broadcasting (see
[backend_experiments.md](backend_experiments.md)). **v1.7** is
design-only: it writes the Stage-2 plan for a future forward-only
native tensor wrapper over `NativeTensorCore` — purpose, non-goals,
ownership/lifetime, the inherited conversion contract, a minimal API
sketch, error/shape behavior, a testing plan, and a staged v1.8–v1.11
sequence, all in
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md); no
code ships. **v1.8** builds the shell of that wrapper —
`tensorforge.experimental.NativeTensor`, a forward-only layer over
`NativeTensorCore` with constructors (`from_array`/`zeros`/`full`),
metadata, `to_numpy`, and an explicit ownership/lifetime story
(idempotent `close()`, context manager, closed tensors reject access) —
and nothing more: no compute ops, no view ops, no autograd, not
`tensorforge.Tensor`. **v1.9** gave that wrapper its forward-only
compute methods — `relu`, `add`, `subtract`, `multiply`, `matmul` —
each delegating to the native kernel and returning a new owning
`NativeTensor`, preserving the runtime's exact-shape/2-D behavior (no
broadcasting), clear `TypeError` naming `NativeTensor` for bad operands,
and `RuntimeError` on closed tensors; no operator overloads, no view
ops, still not `tensorforge.Tensor`. **v1.10** added the metadata-only
view ops — `reshape`, `transpose`, `T`, `narrow` return borrowing
wrappers that share the parent's storage, and `contiguous_copy` returns
a fresh owning wrapper; compute runs over strided views directly,
closing a view spares the owner, and closing the owner invalidates the
views' data access. Still forward-only, no operator overloads, not
`tensorforge.Tensor`. **v1.11** made the wrapper demonstrable — a small,
deterministic runnable example (`examples/native_tensor_demo.py`)
touring construction, compute, views, `contiguous_copy`, and explicit
lifetime, plus a metadata-only `repr` — and documented it as the
experimental, forward-only, isolated tensor it is (no autograd, no
dispatch, no CUDA, no operator overloads, no performance claims).
**v1.12** extended the benchmark suite to characterize the wrapper:
`add`/`relu`/`matmul`, their strided-view forms, and `contiguous_copy`
are timed across four layers — NumPy, the raw-buffer kernels,
`NativeTensorCore`, and `NativeTensor` — so the wrapper's thin
ownership/lifetime/conversion overhead is visible beside the bare
runtime, with correctness checked before timing and no performance
assertions anywhere. **v1.13** is design-only: acting on the v1.12
finding that the elementwise cost is the generic shape/stride odometer
traversal (not the wrapper), it specifies a contiguous fast path for the
native `relu`/`add`/`subtract`/`multiply` kernels — flat index-free loops
for contiguous inputs, the odometer retained for strided views, living in
the `NativeTensorCore`/native-kernel layer so `NativeTensor` inherits it,
bit-for-bit equivalent with unchanged semantics
([native_contiguous_fast_path_design.md](native_contiguous_fast_path_design.md));
no code ships. **v1.14** implements that fast path: flat, index-free
kernels (`tf_core_relu_contiguous` and add/subtract/multiply variants)
sit beside the generic odometer kernels, and `NativeTensorCore.relu` /
`_binary_core_op` pick them when every operand is row-major contiguous
(nonzero offsets and scalars included), falling back to the retained
odometer path for any strided view — proven bit-for-bit equal to it and
to NumPy. `NativeTensor` inherited the change with no wrapper edits, and
no broadcasting, reductions, autograd, `Tensor` integration, or CUDA
were added; performance is left to the benchmark suite, not claimed.
**v1.15** closes that optimization loop with an honest benchmark impact
report (in [backend_experiments.md](backend_experiments.md)): on a local
run of the existing suite, contiguous elementwise `add`/`relu` on
`NativeTensorCore` and `NativeTensor` moved to roughly raw-buffer-C++
speed (~1.5× NumPy at 1000×1000), while the non-contiguous view rows
stayed on the generic odometer path (~2.5–3.5×) — the exact
contiguous-vs-strided spread the design predicted. Matmul and
`contiguous_copy` were unchanged (out of v1.14 scope), and `NativeTensor`
kept tracking `NativeTensorCore` closely, reinforcing that the wrapper is
thin. No kernels or runtime behavior changed, numbers are
hardware-dependent, and no test asserts a speedup. **v1.16** is
design-only: with Phase A1 complete, it writes the design for native
broadcasting — lifting the native elementwise ops
(`add`/`subtract`/`multiply`) from exact-shape-only to NumPy-style
broadcasting (scalar↔tensor, same-rank size-1 stretching, left-padding
with leading 1s) via a zero-stride read model that never materializes an
expanded operand and keeps a freshly allocated contiguous output. The
v1.14 fast path is preserved for the same-shape contiguous case;
broadcasting lives in the `NativeTensorCore`/native-kernel layer so
`NativeTensor` will inherit it with no wrapper change; errors stay
explicit (a mismatch names both shapes, no silent NumPy fallback); and
the autograd implication (a broadcast forward read is a sum-reduction on
the backward pass) is noted for later, not built
([native_broadcasting_design.md](native_broadcasting_design.md)). No code
ships; implementation is v1.17. **v1.17** implements that design:
`NativeTensorCore.add`/`subtract`/`multiply` now broadcast NumPy-style
(scalar↔tensor, same-rank size-1 stretching, left-padding with leading
1s). A pure `broadcast_shapes` helper infers the output shape (raising a
`ValueError` naming both shapes when incompatible) and a `_broadcast_strides`
helper feeds the **existing** generic odometer kernel zero strides on
stretched axes — so no new C++ kernel was added, nothing is materialized,
and the same-shape v1.14 fast path and generic odometer are untouched.
Output stays freshly allocated row-major contiguous; `NativeTensor` (and
the explicit native backend) inherited broadcasting with no wrapper edit;
results match NumPy exactly, including transposed/narrowed/nonzero-offset
operands. No reductions, autograd, `Tensor` integration, CUDA, dtype
promotion, operator overloads, or matmul broadcasting were added.
**v1.18** is design-only: with Phase A2 complete, it writes the design for
native reductions — `sum`/`mean` first (`max`/`argmax`/`min`/`product`
deferred), with NumPy-style `axis=None`/integer/negative-axis and
`keepdims` semantics and a scatter-accumulate traversal that is the **dual
of broadcasting** (broadcasting reads through zero strides; a reduction
writes through zero strides, so the existing odometer machinery drives it,
reads any contiguous/transposed/narrowed/nonzero-offset input directly
without materializing, and writes a freshly allocated row-major contiguous
output). It commits to honest floating-point behavior (order-sensitive
sums, plain deterministic loop, NumPy comparison to a **tolerance** rather
than bit-for-bit, no Kahan/pairwise/SIMD in first scope) and records the
autograd relationship — broadcasting's backward is a reduction over the
broadcast axes, so reductions are a prerequisite for native autograd — but
builds none of it. `NativeTensor` will inherit `sum`/`mean` by delegation
with no wrapper change; no autograd, `Tensor` integration, CUDA, dtype
promotion, operator overloads, tuple-axis, or distributed reductions come
with it ([native_reductions_design.md](native_reductions_design.md)). No
code ships; implementation is v1.19. **v1.19** implements that design:
`NativeTensorCore`/`NativeTensor` gain `sum`/`mean(axis=None,
keepdims=False)` with NumPy-style semantics (all-elements, single integer
or negative axis, `keepdims`). A pure `reduce_shape` helper infers the
output shape, and one new C ABI kernel `tf_core_sum` — the **dual of
broadcasting** (broadcasting reads through zero strides; a reduction
writes through zero strides) — scatter-accumulates a strided input
(contiguous/transposed/narrowed/nonzero-offset, never materialized) into
freshly allocated zero-initialized row-major contiguous output; `mean`
reuses `sum` and scales in place by `1/count` via a small
`tf_storage_scale` primitive, no NumPy touching the data. `NativeTensor`
inherited reductions by delegation with no wrapper edit, and the explicit
NumPy/native backends gained symmetric `sum`/`mean`. Reductions are
forward-only — no autograd — with float order-sensitivity handled
honestly (deterministic plain loop, NumPy comparison to a tolerance, no
Kahan/pairwise/SIMD). No `max`/`argmax`/`min`/`product`, tuple axes,
`Tensor` integration, CUDA, dtype promotion, operator overloads, or
distributed reductions were added; the next step is v1.20, a native
dtype/device metadata design. **v1.20** is design-only: with reductions
complete, it writes the design for explicit **dtype and device metadata**,
closing the Phase A design surface. Today the native runtime is
float64-CPU-only *implicitly*; the design makes that explicit —
`dtype`/`device` become inspectable, validated canonical string tags
(`"float64"`/`"cpu"`) owned by `NativeStorage` (so views share them and a
future CUDA branch has device-aware storage) and surfaced read-only
through `NativeTensorCore`/`NativeTensor`, with default-preserving
constructor arguments so every existing call is unchanged. It specifies
operation validation (binary ops/matmul require matching dtype+device
naming both; `sum` preserves dtype, `mean` stays float64; `to_numpy`
matches the stored dtype once non-float64 exists), a hard
no-promotion/no-auto-copy/no-silent-conversion rule, future explicit
casting/device moves (`astype`/`to`/`cpu`/`cuda`, none built; `cuda()`
deferred until a CUDA backend exists), and — because the kernels are
float64/CPU only — **recommends rejecting** any non-`float64`/non-`cpu`
construction so no tensor advertises a dtype it cannot compute. It
recommends a small metadata-only implementation (float64/cpu) as v1.21 to
close Phase A in code before the Phase B native-autograd design
([native_dtype_device_metadata_design.md](native_dtype_device_metadata_design.md)).
No code ships. **v1.21** implements that design — metadata only, float64/cpu
only, no kernel or compute change. `dtype` and `device` become explicit,
inspectable tags **owned by `NativeStorage`** and surfaced read-only through
`NativeTensorCore.dtype`/`.device` and `NativeTensor.dtype`/`.device`; two
pure helpers (`normalize_dtype`/`normalize_device`) validate them against
`SUPPORTED_DTYPES == ("float64",)` / `SUPPORTED_DEVICES == ("cpu",)`.
Constructors on the core, the wrapper, and the native backend gained
default-preserving `dtype`/`device` arguments (`None`/`"float64"`,
`"cpu"`), so every existing call is byte-for-byte unchanged; following the
design's reject-over-inert recommendation, unsupported values are rejected
at construction (before allocation), and binary ops/matmul validate matching
dtype+device as the guard native autograd will build on. Every op and view
preserves the tags, `to_numpy` still returns float64, and `backend_info`
advertises the supported sets. No dtype promotion, casting, non-float64
kernels, CUDA, autograd, or `Tensor` integration came with it. This
**closes Phase A — the native CPU runtime — in code**; the next step is
v2.0, the Phase B native-autograd design. CUDA/GPU experiments remain future
work. **v2.0** is design-only and **opens Phase B**: it writes the design
for native reverse-mode autograd over `NativeTensor` / `NativeTensorCore`.
The native runtime is forward-only today (`NativeTensorCore` results record
no parents/backward; `NativeTensor` has no `requires_grad`/`grad`/
`backward`); the design specifies a **Python-managed graph at the
`NativeTensor` layer** — `NativeTensorCore` stays the raw forward runtime
and the C++ kernels own no graph state — where each differentiable op
records core + `requires_grad` + parents + a backward closure + an op name,
leaf tensors accumulate gradients, and `backward()` walks the graph in
reverse topological order (scalar outputs seed `1`, non-scalar outputs
require an explicit gradient). Gradients are **native** (`NativeTensor`-
backed, lazily initialized, accumulated by native `add`) and honor the
v1.21 metadata contract (`grad.dtype == tensor.dtype`,
`grad.device == tensor.device`) — the concrete reason A4 preceded autograd.
Broadcasting backward is an `unbroadcast(grad, original_shape)` helper over
native reductions (a broadcast forward read is a sum-reduction backward);
the design is honest about missing kernels (a small fused `relu_backward`;
deferred negation/scalar-multiply, core-level `divide`, and a
scatter/copy-into-view for `narrow`/`contiguous_copy` backward). It stays
separate from `tensorforge.Tensor` (no conversion, no implicit dispatch, no
silent NumPy fallback, `Tensor` behavior unchanged) and CPU/float64 only
(no CUDA autograd), staged as v2.1 metadata skeleton → v2.2 basic backward
(add/multiply/relu/sum) → v2.3 broadcasting + mean backward → v2.4 matmul
backward → v2.5 native autograd demo, then Phase C (native training stack)
([native_autograd_design.md](native_autograd_design.md)). No code ships;
the next step is v2.1, the native autograd metadata skeleton. **v2.1**
implements that skeleton — **Phase B's first code**. `NativeTensor` gains
an opt-in, Python-managed autograd graph: state (`_requires_grad`,
`_grad`, `_parents`, `_backward`, `_op`, `_is_leaf`) at the wrapper layer,
the public surface (`requires_grad`/`grad`/`is_leaf`/`zero_grad`/`detach`/
`backward`), a default-preserving `requires_grad=False` constructor
argument (non-`bool` rejected), an internal graph constructor `_from_op`
(a result's `requires_grad` is the OR of its parents'), native
`NativeTensor`-backed gradient accumulation (`_accumulate_grad` via the
native `add` kernel — no NumPy in the gradient path), and a
reverse-topological `backward(gradient=None)` driver (post-order DFS keyed
by object identity so shared/duplicate parents are visited once; scalar
outputs seed a native `1.0`, non-scalars require an explicit
shape/dtype/device-matching `NativeTensor` gradient; only leaves retain
grad). `zero_grad()` clears to `None`; `detach()` returns an owning
contiguous copy detached from the graph; `retain_graph` is intentionally
not offered (the graph is rebuilt each call, so repeated `backward()`
accumulates until `zero_grad()`). Crucially, v2.1 is a **skeleton**: the
forward compute ops (`add`/`multiply`/`relu`/`sum`/…) are **not** wired
into autograd yet — their results stay `requires_grad=False` and the
engine is exercised through `_from_op` — so `NativeTensorCore` and the C++
kernels remain forward-only and autograd-unaware, and `tensorforge.Tensor`
is untouched. No kernels changed and no arithmetic became differentiable.
The next step is v2.2, basic native backward (add/multiply/relu/sum).
**v2.2** wires the core operations into that engine — **native autograd
becomes differentiable arithmetic**. `NativeTensor.add`/`subtract`/
`multiply`/`relu`/`sum`/`mean`/`matmul`/`reshape`/`transpose`/`T`/
`contiguous_copy` now build graph nodes when an operand requires grad
(plain forward tensors otherwise — non-autograd use is unchanged), with
every backward rule computed by **native forward kernels at the
`NativeTensorCore` level**: add/subtract pass the upstream through
(negation composed as a broadcast-scalar multiply by a native `-1.0` — no
negate kernel), multiply computes `u·b`/`u·a`, matmul computes
`u @ b.T`/`a.T @ u` over strided transpose views, sum/mean broadcast the
upstream back natively (reduced axes reinserted as size 1, expanded by the
existing zero-stride broadcasting; mean scaled by a native `1/count`
scalar multiply), and reshape/transpose apply the inverse
relabeling/permutation. Broadcasting backward is a private
`_unbroadcast(grad, target_shape)` helper — the adjoint of broadcasting —
built from single-axis native reductions applied in a stable order, with
the scalar shape `()` and one-element `(1,)` distinguished exactly.
`contiguous_copy` backward is the identity (a gradient lives at the
logical shape, so the parent's layout is irrelevant). The **one new C++
kernel** is the fused `tf_core_relu_backward` the v2.0 design flagged
(`upstream` where `x > 0` else `0`; `x == 0` blocks, matching the Python
Tensor), implemented as one more op through the existing generic binary
odometer and surfaced as the forward-shaped
`NativeTensorCore.relu_backward` — the core and kernels still own **no
graph state**. Retained gradient contributions are always the upstream
tensor itself or fresh owning contiguous storage (never a borrowing view
over a transient), and closing an operand or intermediate before
`backward()` raises clearly instead of reading freed storage. Everything
is verified against exact analytical values **and central finite
differences** (NumPy only as the test-side reference), a deterministic
demo (`examples/native_autograd_demo.py`) runs
`x.matmul(w).add(b).relu().mean()` natively end to end with broadcast
bias gradients, and the CI smoke script hard-checks one scalar-loss
backward. **`narrow` backward is deferred to v2.3** (it needs a native
scatter primitive), `retain_graph` is still not offered, and there is no
`tensorforge.Tensor` integration, no optimizer/module layer, no CUDA, and
no performance claims. **v2.3** delivers that scatter and makes **`narrow`
differentiable**, completing the view-backward set. When its parent
requires grad, `narrow(dim, start, length)` builds a graph node whose
backward **scatters** the upstream gradient into a fresh owning row-major
contiguous zeros tensor of the parent's shape at the narrowed region —
un-narrowed positions stay zero, the narrowed region equals the upstream.
The **one new C++ kernel** is `tf_core_narrow_backward`, the odometer dual
of `tf_core_sum`: where a sum folds many inputs into one cell through zero
write-strides, a narrow-backward writes each input into its own cell at
the parent's full row-major strides from a `start`-shifted base offset. It
reads the upstream through its own strides/offset (so strided gradients
need no materialization), is surfaced as
`NativeTensorCore.narrow_backward(dim, start, original_shape)`, and — like
`relu_backward`/`sum` — is not added to `list_kernels()`. Because the
gradient lives at the logical shape, the scatter output is always fresh
contiguous storage of the parent's shape, so **transposed, narrowed, and
nonzero-offset parents all differentiate correctly** (each is a preceding
node whose own backward handles its layout); nested narrows and `narrow`
under `sum`/`mean`/`multiply`/`transpose`/`reshape` all compose. Rules are
verified against an independent NumPy zero-padding reference and a
finite-difference check (NumPy test-side only), and the CI smoke script
hard-checks one narrow-backward pattern. `NativeTensorCore` and the C++
kernels still own no graph state; there is no `divide` backward, no
`retain_graph`, no `tensorforge.Tensor` integration, no optimizer/module
layer, no CUDA, and no performance claims. **v2.4** gives the native
autograd graph an explicit **lifetime policy**. `backward` gains a flag —
`backward(gradient=None, retain_graph=False)` — validated as a real `bool`
first (non-bool raises `TypeError` before anything is traversed or
mutated; never coerced). The default is **one-shot**: a successful pass
releases the operation graph of every traversed non-leaf node — clearing
its `_parents` and `_backward` closure (so nothing keeps the parents
alive) and marking it freed — and a later `backward()` reaching it raises
a clear `RuntimeError` naming `retain_graph=True` as the remedy, rather
than silently treating the freed node as a leaf and truncating history.
That one rule covers a repeated backward on the same output, a **second
output over a shared intermediate**, and a **new op built from a freed
value** (whose forward still works on the intact stored value — only
backward refuses to cross the freed history). `retain_graph=True` keeps
the graph for another pass; leaf gradients accumulate across successful
passes until `zero_grad()` (which clears a leaf grad without resurrecting
a freed graph or damaging a retained one); a genuine leaf has no graph to
free and is never marked freed, so repeated `backward()` on a scalar leaf
keeps accumulating. The pass is **failure-safe** — staged against a
snapshot of every node's gradient (gradients are immutable, so
accumulation replaces the reference with a fresh native `add`), so if a
callback raises mid-traversal the references are restored, leaving no
partial commit and no partial free; cleanup runs only after the pass fully
succeeds. This is a **Python-only** `NativeTensor` change — no C++ touched,
no kernel added, no NumPy in the gradient path, `NativeTensorCore` still
graph-unaware — and it is explicitly **not** full PyTorch parity (no
per-node `retain_grad`, no double-backward). **v2.5** is a
**measurement-only** milestone — it changes no autograd behavior and adds
no kernel — that **characterizes** the native autograd stack with a
reproducible harness (`benchmarks/benchmark_native_autograd.py`). Five
workloads (a same-shape elementwise chain, a genuine-broadcast chain, a
3-D reduction chain, a 2-D matmul chain, and a transpose→narrow→
contiguous_copy→reshape view chain) run in four modes that separate the
layers: `forward_native` (grad off, no graph), `forward_graph` (graph
built, no backward), `forward_backward_fresh` (fresh graph + one-shot
`backward()`, cleanup included), and `backward_retained` (one graph built
outside the loop, `backward(retain_graph=True)` repeatedly — isolating
repeated backward, explicitly not a training-step estimate). Timing uses
`time.perf_counter_ns()` with configurable warmup/iterations/repeats,
median as the primary statistic plus min/max spread, and a correctness
gate (output shape, finite output, and — for backward modes — that each
leaf gradient exists, has the right shape, and is finite) before any
timing; NumPy is used only to *inspect* copied gradient values, never to
compute a benchmarked result. A CLI
(`--case --mode --warmup --iterations --repeats --json --smoke`) runs all
cases/modes by default, rejects unknown selections and non-positive
counts, and emits pure JSON (raw samples included) under `--json`. One
honest hardware-specific snapshot is recorded in
[native_autograd_benchmarks.md](native_autograd_benchmarks.md), with
cautious observations only (adding backward dominates; retained backward
sits below fresh; graph-construction overhead is small at these sizes;
tiny tensors are wrapper/ctypes-bound) and **no** cross-framework or
production claims. The benchmark tests validate schema and behavior, never
speed. The next step is **v2.6 — Phase B guardrails and completion**,
after which Phase C — a native training stack — opens.
