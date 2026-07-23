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
after which Phase C — a native training stack — opens. **v2.6 completes
Phase B** — an audit-and-lock-down milestone that adds **no** operation,
kernel, optimizer, training abstraction, or optimization and changes **no**
autograd behavior. It adds cross-cutting guardrail tests
(`tests/test_native_autograd_guardrails.py`, selector
`-k "phase_b_guardrail or native_autograd_guardrail or native_backend_isolation"`)
that lock several completed invariants together: a **runtime
NumPy-no-fallback guard** that replaces NumPy's numerical functions with
tripwires around representative backward passes (elementwise, broadcasting,
reduction, matmul, and a transpose→narrow→contiguous_copy→reshape view
chain) while leaving the marshalling helpers intact — proving backward
computes gradients with native kernels, never NumPy; **`NativeTensor` ↔
`tensorforge.Tensor` isolation** (native ops/grads stay native, `Tensor`
stays NumPy-backed, neither backward touches the other, mixed operands raise
clearly); **explicit-backend / no-implicit-dispatch** behavior (reached only
through `tensorforge.experimental`, `import tensorforge` imports neither
`experimental` nor `backends`, unavailability raises the build-instructions
`ImportError`, no automatic selection); and **gradient-ownership,
graph-lifetime, detach, view+offset, and closed-operand failure-safety**
invariants over realistic mixed graphs, plus the **kernel-registry
boundary** (the internal fused backward kernels never leak into
`list_kernels()`) and the **v2.5 benchmark mode contract**. It records the
**final Phase B support matrix** and the explicit **divide-backward
decision** — deferred beyond Phase B, which is complete without it because
the completed op set already spans a first native training stack (see
[native_autograd_design.md](native_autograd_design.md), sections 17–19). The
one source touch is correcting a stale package docstring
(`tensorforge.experimental` no longer claims "no autograd"); no C++ changed
and no kernel or symbol was added. **Phase B is complete**; the next step is
**Advanced C++ v3.1 — NativeParameter and Parameter Registration Contract**,
the first milestone of **Phase C — a native training stack**. **v3.1** opens
Phase C with that foundation: `tensorforge.experimental.NativeParameter` and
`NativeParameterRegistry` (one new module,
`src/tensorforge/experimental/native_parameter.py`; `NativeTensor` itself,
the C++ kernels, and `tensorforge.Tensor`/`tensorforge.nn.Parameter` are all
untouched). `NativeParameter` subclasses `NativeTensor` (`__slots__ = ()`)
and its instances are **graph-free owning leaves for their whole life**:
construction copies array-like data — or the *current value* of an existing
`NativeTensor`, leaf or non-leaf, contiguous or strided/offset view — into
independent owning contiguous float64/cpu native storage, inheriting no
`_parents`, `_backward`, or freed-graph state (a closed source raises;
closing source or parameter never invalidates the other; no storage is ever
shared, so future optimizer updates can never mutate an unrelated tensor
through a hidden alias). `requires_grad` is a validated real bool (default
`True`; `False` builds a frozen parameter that stays registerable and
discoverable but accumulates no gradient). The internal graph constructors
(`_from_core`/`_from_op`) are overridden to delegate to `NativeTensor`, so
**parameter-ness never propagates**: math, views, `contiguous_copy`,
reductions, and `detach()` all return plain `NativeTensor`s, and only
calling `NativeParameter(...)` creates a parameter. Gradient behavior is the
Phase B contract unchanged (`.grad` starts `None`, accumulates
`NativeTensor`-backed gradients matching shape/dtype/device across fresh
graphs, `zero_grad()` clears without touching data). Identity is **object
identity** — no `__eq__`/`__hash__` anywhere in the hierarchy, so equal-valued
parameters stay distinct and future optimizer state keys by `id`.
`NativeParameterRegistry` is the minimal insertion-ordered registration
contract the future `NativeModule` (v3.2) will embed: names are non-empty
dot-free strings (dots reserved for hierarchical state_dict keys); slots
accept `NativeParameter` only, with `None` unregistering (`KeyError` if
absent) — ordinary `NativeTensor` and framework `Tensor`/`Parameter` are
rejected, never wrapped implicitly; replacement preserves the slot's
position and never closes, mutates, or transfers state from the old
parameter; removal deletes the slot so re-registration appends at the end;
aliases (one parameter under several names) are visible in
`named_parameters()`, while `parameters()` deduplicates by identity in
first-registration order and `unique_named_parameters()` is first-name-wins;
the registry stores references only — it owns no storage and never closes,
copies, or mutates a parameter. Deliberately **not** shipped: any
`NativeModule` hierarchy, layers, losses, optimizers, training loops,
state_dict/serialization, divide backward, operator overloads, dtype
promotion, implicit dispatch, or NumPy fallback. 49 focused tests
(`tests/test_native_parameter.py`, selector
`-k "native_parameter or parameter_registration"`) lock the contract; the
full suite passes at **972 tests**. The next milestone is **Advanced C++
v3.2 — NativeModule Core and Recursive Registration** (child-module
registration, automatic parameter/module assignment registration, recursive
`parameters()`/`named_parameters()`/`modules()`/`named_modules()`,
`zero_grad()`, deterministic traversal, shared-parameter and shared-module
handling, and the train/eval state foundation — no layers or optimizers in
v3.2). **v3.2** delivers that module core:
`tensorforge.experimental.NativeModule` (one new module,
`src/tensorforge/experimental/native_module.py`; the only v3.1 touch is a
minimal read-only `NativeParameterRegistry` extension — `get`/`__contains__`
and a shared name-validation helper with identical messages — and
`NativeTensor`, the C++ kernels, and `tensorforge.nn` are all untouched).
`NativeModule` is a **Python-side organizational abstraction**: it computes
nothing, owns no native storage, and never closes, copies, or mutates what
it registers. **Assignment registers** — a `NativeParameter` value enters
the parameter registry, a `NativeModule` value the child registry, and
everything else (plain `NativeTensor`, `tensorforge.Tensor`/`Parameter`/
`nn.Module`, ordinary values) stays a normal attribute that never enters
native traversal; registered objects live only in the registries
(`__getattr__` resolves them), giving one source of truth. **One category
per name; the latest assignment wins**: registration validates first (a
failure mutates nothing), then evicts the name from the other categories;
replacement within a registry preserves the slot position, moving between
registries appends, `module.name = None` (and `del module.name`)
unregisters leaving the attribute readable as `None`, and re-registering a
removed name appends — the v3.1 ordering rules throughout, with evicted
objects dropped, never closed or mutated, and no gradient transfer.
`register_parameter`/`add_module` are the explicit forms with identical
semantics (deliberately stricter: non-parameter/non-module values raise
`TypeError`; `None` on an absent name raises `KeyError`). Names are
non-empty dot-free strings; `"_parameters"`/`"_modules"`/`"training"` are
reserved; `__init__` builds the registries via `object.__setattr__` so
initialization never routes through registration, and registering before
`super().__init__()` raises a clear `RuntimeError`. **Traversal is
deterministic pre-order depth-first with identity deduplication and
first-discovered canonical dotted names**: `named_modules()` yields
`("", self)` first and never revisits a module (shared modules emit once
under their first path; direct and indirect reference cycles terminate
safely), and `named_parameters(prefix="", recurse=True)` yields each unique
parameter once — direct parameters before descendants', aliases and shared
parameters deduplicated by `id`, frozen parameters included — with
`parameters()`/`modules()` the matching lists a future optimizer iterates
and the dotted names exactly the keys the future state_dict will use.
`zero_grad()` calls each unique parameter's existing `zero_grad()` and
returns `None`; `train(mode=True)` validates `mode` as a real bool before
touching any state, propagates `training` to every unique module, and
returns `self` (`eval()` = `train(False)`; every module starts
`training=True`); `forward()` raises `NotImplementedError` and `__call__`
delegates to it — no hooks, buffers, or tracing. Deliberately **not**
shipped: `NativeLinear`, `NativeSequential`, activations, losses,
optimizers, state_dict, serialization, training loops, dtype/device
expansion, implicit dispatch, or NumPy fallback. 49 focused tests
(`tests/test_native_module.py`, selector
`-k "native_module or recursive_registration"`) lock the contract; the full
suite passes at **1021 tests**. The next milestone is **Advanced C++ v3.3 —
Native State Dictionary Contract**: `state_dict()`/`load_state_dict()` over
the v3.2 canonical hierarchical names, strict missing/unexpected-key
checks, shape/dtype/device validation, value copying without replacing
`NativeParameter` identity, and shared-parameter canonical naming — no file
serialization and no optimizer state yet (state_dict, `NativeLinear`,
serialization, and checkpointing stay separate milestones). **v3.3**
delivers that contract — **in-memory and parameters-only** (no buffers,
optimizer state, training/RNG state, file formats, archives, or checkpoint
metadata; the two source touches are `NativeModule.state_dict()`/
`load_state_dict()` and one narrowly scoped internal
`NativeParameter._adopt_value_core`; `NativeTensor`, the C++ kernels, and
`tensorforge.nn` are untouched). **`state_dict()`** returns an
insertion-ordered `{canonical_name: NativeTensor}` mapping: keys are
exactly the v3.2 canonical `named_parameters()` names (dotted,
direct-before-descendants, shared parameters once under their
first-discovered path, frozen included, cycle-safe), and every value is an
ordinary graph-free `requires_grad=False` NativeTensor holding an
**independent owning contiguous copy** made by the native copy path (zeros
+ native add — NumPy neither computes nor copies state values). Snapshot
and model share no storage in either direction: mutating/replacing/closing
a parameter never affects an existing snapshot and vice versa; a closed
registered parameter raises clearly naming the key, with half-built
snapshots closed rather than returned. **`load_state_dict(state_dict,
strict=True)`** copies values **into** the existing parameters — never
assigning new objects — preserving `id(parameter)`, registration and
canonical traversal, shared aliasing (one canonical key updates the shared
object once; every alias observes it; a supplied alias key is unexpected),
`requires_grad`/frozen state, gradients **by identity and value** (`None`
stays `None`; never cleared, replaced, or accumulated), and training
flags. Validation is entirely pre-mutation in a documented order: `strict`
a real bool → mapping input (snapshotted once) → string keys →
missing/unexpected computed (strict raises one ValueError reporting
**both**; non-strict returns them in an immutable
`LoadStateDictResult(missing_keys, unexpected_keys)`, missing in canonical
order, unexpected in input order) → per-key preflight (open NativeTensor
values — a NativeParameter source is accepted purely by copy;
`tensorforge.Tensor`/`Parameter`, arrays, and scalars rejected; exact
shape/dtype/device match with the failing key named; no
broadcasting/reshaping/casting/device movement) → **stage** independent
native copies (failure closes them, mutates nothing) → **commit** by
swapping cores (pure reference assignments guarded by rollback restoring
the original cores), releasing each replaced core exactly once only after
full success. No failure leaves the model partially updated, closes an
input, or invalidates existing snapshots. The internal primitive is
documented as controlled value replacement, **not yet the optimizer update
API**; and the first in-place mutation gets an explicit graph policy — a
graph built before loading stays memory-safe and a later backward through
it reads the newly loaded values. 54 focused tests
(`tests/test_native_state_dict.py`, selector
`-k "native_state_dict or load_state_dict"`) lock the contract; the full
suite passes at **1075 tests**. The next milestone is **Advanced C++ v3.4 —
NativeLinear**: a first native layer on `NativeModule` — `NativeParameter`
weight, optional `NativeParameter` bias, deterministic initialization,
input validation, strictly 2-D forward semantics initially (native
`matmul` plus broadcast `add`), registration through assignment,
forward/backward and finite-difference tests, and state_dict compatibility
— no optimizer or training loop yet, and not combined with
`NativeSequential`, activations, or losses. **v3.4** delivers that layer:
`tensorforge.experimental.NativeLinear` (one new module,
`src/tensorforge/experimental/native_linear.py`; `NativeTensor`,
`NativeParameter`, `NativeModule`, the C++ kernels, and
`tensorforge.nn.Linear` are all untouched). `NativeLinear(in_features,
out_features, bias=True, *, seed=None, requires_grad=True)` validates
every Python argument **before any native allocation** (features are real
positive ints, bools and integer-like objects rejected; `bias` and
`requires_grad` real bools; `seed` `None` or a real int) and creates its
parameters by assignment — `self.weight` first, then `self.bias` (or
`None`, leaving only `"weight"` registered) — so v3.2 registration fixes
the deterministic `["weight", "bias"]` order across `named_parameters()`,
`parameters()`, and `state_dict()` (nested:
`"layer.weight"`/`"layer.bias"`). The **weight orientation is
`(in_features, out_features)`** — the same `x @ weight` orientation as the
stable Linear, applied directly by the strictly 2-D native matmul — with
`(out_features,)` bias broadcast over the batch. **Initialization** is
deterministic and self-contained: fan-in uniform on
`[-1/sqrt(in_features), +1/sqrt(in_features)]` from a **local**
`numpy.random.default_rng(seed)` (an int seed reproduces exact values,
`None` draws fresh entropy, the global NumPy RNG is never touched; NumPy
is host-side initialization data preparation only). **Forward** requires
an open 2-D `(batch, in_features)` `NativeTensor` with matching
dtype/device — the stable framework's `Tensor`, arrays, lists, scalars,
closed tensors, and wrong shapes are rejected with errors naming the
expected contract and actual shape; nothing is wrapped or reshaped — and
computes `input.matmul(weight)` plus `add(bias)`, returning an ordinary
`NativeTensor`. **Backward is the existing autograd** — no manual or
fused path exists; input/weight/bias gradients (`(batch, in)`,
`(in, out)`, `(out,)` batch-reduced by the native unbroadcast) are
verified against exact analytical formulas and central finite differences
(`eps=1e-6`, float64 tolerances), including no-bias, frozen-parameter
(registered, snapshot-visible, gradient-free, requiring inputs still
differentiated), branching, repeated fresh cycles, one-shot cleanup, and
`retain_graph`. A tripwire test proves no NumPy compute reaches
forward/backward. **State compatibility** follows v3.3 exactly: loads
change values while identity, gradients, `requires_grad`, and frozen
state survive; bias/no-bias mismatches follow the strict key rules and
fail before mutation; shape-incompatible states fail atomically; the
forward → backward → load-after-completion mutation boundary is unchanged
(no version counters). Deliberately **not** shipped: `NativeSequential`,
activation modules, losses, optimizers, training loops, serialization,
buffers, hooks, or fused kernels. 42 focused tests
(`tests/test_native_linear.py`, selector `-k "native_linear"`) lock the
contract; the full suite passes at **1117 tests**. The next milestone is
**Advanced C++ v3.5 — NativeReLU and NativeSequential**: a `NativeReLU`
module wrapping the existing `NativeTensor.relu()`, and a
`NativeSequential` ordered child-module container with integer-string
child names, deterministic recursive traversal, forward composition,
shared-module behavior, train/eval propagation, and state_dict
compatibility (replacement/indexing only if tightly justified) — no loss,
optimizer, or training loop yet. **v3.5** delivers both (two new modules,
`src/tensorforge/experimental/native_relu.py` and
`native_sequential.py`; `NativeModule`, `NativeTensor`, `NativeParameter`,
`NativeLinear`, the C++ kernels, and the stable `tensorforge.nn.ReLU` /
`Sequential` are all untouched). **`NativeReLU`** is a parameter-free
NativeModule whose `forward` validates an open `NativeTensor` (framework
`Tensor`, arrays, lists, scalars, and closed tensors rejected; a
`NativeParameter` input yields a plain `NativeTensor`) and delegates to
the existing `relu()` — shape-generic across every rank and
strided/offset layout, no in-place mode, no copies, dtype/device
preserved; its backward is entirely the existing fused native relu
autograd (the block-at-exactly-zero rule tested unchanged), its
`state_dict()` is empty, and `training` never affects numerics.
**`NativeSequential(*modules)`** registers children under **contiguous
integer-string slots** `"0"..len-1` where **execution order is the
registered order** — a single source of truth with the invariant enforced
at the v3.2 registration funnel: the constructor validates every entry
before registering any; `append` takes the next slot and returns self;
`seq[i] = module` / `add_module("i", ...)` replace preserving position;
and gap-producing indices, non-canonical digit strings, non-slot child
names (a registered child that never executes), direct `NativeParameter`
assignment, every form of slot removal (`None`, ordinary-value overwrite,
`del`, `add_module(name, None)`), and self-insertion are rejected with
clear errors (ordinary non-module attributes stay allowed; traversal
stays cycle-safe, but executing a deliberately cyclic composition is
unsupported). The container surface is minimal — `len`, iteration in
execution order (shared children **not** deduplicated), real-int indexing
with Python-style negatives — and the **central shared-module rule** is
documented and tested: **execution is position-based** (a shared child
runs once per slot) **while ownership is identity-based** (`modules()`,
`named_parameters()`, `state_dict()`, `train()`, `zero_grad()` visit the
shared object once under its first-discovered slot path; a duplicate
alias state key is *unexpected*). Forward is pure composition — each
child validates its own input and contributes its own graph nodes, child
exceptions propagate, an **empty sequence returns its input by
identity**, and no NumPy touches forward or backward (tripwire-tested).
The composed Linear→ReLU→Linear graph is verified end to end: exact
analytical input/weight/bias gradients including the ReLU mask, bias-free
and frozen variants, branching accumulation, repeated fresh cycles,
one-shot cleanup, `retain_graph`, recursive `zero_grad`, and **central
finite differences** for all five gradients (`eps=1e-6`, `atol=1e-6`,
hidden pre-activations kept ≥ 0.1 from the zero boundary); state keys are
slot-derived (`"0.weight"`, `"0.bias"`, `"2.weight"`, `"2.bias"`; nested
`"0.0.weight"`; ReLU contributes none) with all v3.3 guarantees, and the
v3.3/v3.4 mutation boundary (forward → backward → zero_grad/state update
after graph completion) is restated unchanged. Deliberately **not**
shipped: losses, optimizers, training loops, serialization, version
counters, buffers, hooks, or other activations/layers. 52 focused tests
(`tests/test_native_relu.py` + `tests/test_native_sequential.py`,
selector `-k "native_relu or native_sequential"`) lock the contract; the
full suite passes at **1169 tests**. The next milestone is **Advanced C++
v3.6 — NativeMSELoss**: an MSE loss as a NativeModule —
prediction/target NativeTensor validation, an exact-shape contract
initially, reduction modes limited to the smallest justified surface
(scalar mean by default), forward composed from native
`subtract`/`multiply`/`sum`/`mean`, and exact plus finite-difference
gradients — no optimizer or training loop yet, and not combined with SGD
or model training. **v3.6** delivers that loss:
`tensorforge.experimental.NativeMSELoss` (one new module,
`src/tensorforge/experimental/native_mse_loss.py`; `NativeTensor`,
`NativeModule`, the existing layers, the C++ kernels, and the stable
framework's `mse_loss` are all untouched). It is a **parameter-free**
NativeModule whose forward is exactly `difference =
prediction.subtract(target)`; `squared =
difference.multiply(difference)`; then `squared.mean()` (default) or
`squared.sum()` — both **scalar**, so the existing default backward seed
applies, and an explicit scalar upstream scales both gradients per the
normal engine rules. **The existing autograd is the entire backward**:
multiply's duplicate-parent accumulation on the shared difference node
yields the factor 2, subtract's backward yields the target's negative
sign, and mean's existing native backward yields the `1/N` scaling — no
division operation, no fused kernel, no manual callback
(`dL/dprediction = 2(p−t)/N`, `dL/dtarget = −2(p−t)/N` under mean; drop
`/N` under sum — used only as test references). The **reduction
contract** is deliberately tiny: exactly `"mean"` and `"sum"` by exact
string match (case/whitespace variants, non-strings, and other values
rejected; nothing normalized; no `"none"` — both supported reductions
are scalar, sufficient for the first native training loop), stored as
constructor configuration and never as state. **Validation before any
graph node**: both operands must be open NativeTensors (NativeParameter
accepted, accumulating native-backed gradients; framework
Tensor/arrays/lists/scalars/closed tensors rejected with errors naming
*which* argument), with **exactly equal shapes — broadcasting forbidden**
even though `subtract` supports it (the error names both shapes) — and
exact dtype/device equality. Shape-generic across all supported ranks;
zero-element tensors remain unconstructible (a NativeStorage limitation
the loss inherits); inputs are never mutated; the module owns no storage
and keeps no temporaries; `state_dict()` is empty with v3.3
strict/non-strict unexpected-key behavior; train/eval never affects
numerics; graph lifetime (one-shot, `retain_graph`, unchanged gradients
after failed reuse) holds through the loss. Coverage includes exact
references (1-D, multidimensional total-element `N`, zero difference,
positive/negative differences, one-sided and both-frozen operands,
upstream scaling, branching, duplicate-parent graph identity), **central
finite differences** for prediction and target under both reductions
(`eps=1e-6`, `atol=1e-6`), a NumPy tripwire around analytical
forward+backward, and an **exact end-to-end integration**:
`NativeSequential(NativeLinear → NativeReLU → NativeLinear)` +
`NativeMSELoss` reproduces a full hand-computed gradient chain (input,
both weights and biases through the ReLU mask, and a gradient-requiring
target), with recursive `model.zero_grad()` independent of the target's
gradient and repeated fresh cycles bit-identical. The v3.3–v3.5
**mutation boundary is restated unchanged** (forward → loss → backward →
updates only after graph completion; no version counters). Deliberately
**not** shipped: other losses, optimizers, `NativeSGD`, parameter
updates, version counters, training loops, serialization, buffers, or
hooks. 27 focused tests (`tests/test_native_mse_loss.py`, selector
`-k "native_mse_loss"`) lock the contract; the full suite passes at
**1196 tests**. The next milestone is **Advanced C++ v3.7 — Native
Parameter Mutation Safety and Versioning Contract**: version counters on
mutable native parameter values, forward-time expected-version capture
where backward needs saved parameter values, state loading incrementing
parameter versions, clear stale-forward backward errors, a controlled
no-grad parameter mutation primitive, the identity-preserving update
foundation for `NativeSGD`, and rollback/shared-parameter behavior — no
optimizer and no training loop yet; v3.7 must precede `NativeSGD`
because optimizer updates cannot safely mutate parameter values while
old graphs remain capable of backward, and mutation versioning is not
combined with SGD. **v3.7** delivers that contract (Python-only changes
to `NativeParameter`, the `NativeTensor` autograd metadata, and
`load_state_dict`; no C++ change, no new operation, and the stable
framework untouched). Every `NativeParameter` now carries a **read-only,
monotonically increasing value `version`** — an ordinary non-negative
int, 0 at construction — that counts **replacements of the owned
numerical value** and nothing else: gradient accumulation, `zero_grad`,
registration/aliasing/removal, train/eval, and `state_dict()` snapshots
never move it. Two things increment it: **`copy_value_(source)`**, the
one controlled no-grad mutation primitive (an open NativeTensor source —
a NativeParameter, the parameter itself, or a snapshot accepted purely
as a value source; stable-framework objects, arrays, lists, and scalars
rejected; exact shape/dtype/device, no broadcasting/casting/transfer;
the staged copy is native, owning, contiguous, and never aliased; Python
identity, registrations, `requires_grad`, leaf/graph-free state, and the
existing gradient are preserved by identity and value; old storage
released exactly once; any failure changes nothing — and this is the
exact path `NativeSGD.step()` will commit updates through), and a
**successful `load_state_dict`** (exactly once per matched canonical
parameter — shared parameters once, observed through every alias;
identical values still increment because replacement, not value
equality, is what counts; increments land only after the whole atomic
commit, so every failure/rollback path leaves values *and* versions
exactly unchanged). A **per-operation dependency audit** classifies the
differentiable set: `multiply`, `matmul`, and `relu` backwards read
direct-parent forward values; `add`, `subtract`, `sum`, `mean`,
`reshape`, `transpose`/`T`, `contiguous_copy`, and `narrow` read only
metadata. Graph construction therefore records `(op, parameter,
expected_version)` for every direct NativeParameter operand of a
value-sensitive op (the documented **op-level policy**: guarded even in
the corner where sibling `requires_grad` flags mean the value would not
actually be read — safety and independence from gradient-flow details
over that corner), and `backward()` validates every recorded version
after the freed-graph scan and **before any seed, callback, or gradient
commit**: a stale graph raises a deterministic RuntimeError naming the
operation and expected/current versions — distinct from freed-graph and
closed-tensor errors, leaving gradients, graph structure, and versions
untouched (the graph is *not* freed), unrepairable by `retain_graph` or
by reloading the old numerical value (versions are monotonic) — the
remedy is a fresh forward pass. Value-independent graphs (bias through
`add`, view/reduction chains) stay valid across mutation with
mathematically correct gradients. The version metadata is Python graph
metadata on the node (absent on leaves), freed with one-shot cleanup,
revalidated on every retained-graph pass; autograd never imports the
module stack, and there is no global active-graph registry — mutation
itself is never blocked. Deliberately **not** shipped: `NativeSGD`, any
optimizer, training loops, momentum, weight decay, general in-place
arithmetic, operator overloads, a global no-grad context, checkpointing,
or serialization. 34 focused tests
(`tests/test_native_parameter_versioning.py`, selector
`-k "native_parameter_version or stale_parameter_graph or mutation_safety"`,
NumPy-tripwired mutation/loading/preflight/fresh-pass paths) lock the
contract, one v3.3 assertion is intentionally tightened (a graph built
before loading now raises the stale error instead of silently reading
the new value — still memory-safe), and the full suite passes at
**1230 tests**. The next milestone is **Advanced C++ v3.8 —
NativeSGD**: an optimizer over identity-deduplicated `NativeParameter`
objects — real positive finite learning-rate validation, `step()`
committing graph-free native updates through the v3.7 mutation path
(`grad=None` and frozen parameters skipped, identity preserved, one
version increment per updated parameter), `zero_grad()`,
duplicate/shared-parameter protection, and deterministic update tests —
with no momentum, weight decay, or parameter groups initially, no
training loop, and no combination with the full MLP training example
(the first end-to-end model-training proof may remain v3.9). **v3.8**
delivers that optimizer (one new module,
`src/tensorforge/experimental/native_sgd.py`; `NativeTensor`,
`NativeParameter`, `NativeModule`, the layers, the loss, the C++
kernels, and the stable framework — `tensorforge.optim` included — are
all untouched). `NativeSGD(parameters, lr)` materializes the parameter
iterable exactly once (lists, `model.parameters()`, generators),
validates every entry as an open `NativeParameter` (position-named
errors; plain tensors, stable-framework objects, non-iterables, and
empty collections rejected), stores strong references
**identity-deduplicated in first-occurrence order** (duplicates and
shared-module aliases: one entry, one update, one version increment per
step; never value equality; the optimizer owns nothing and never
closes, copies, or replaces a parameter), and validates `lr` as a real
number (`bool` explicitly rejected, as are strings and float-coercible
objects), finite, and strictly positive — normalized to a Python float
only after validation, never clamped. **`step()` is two-phase and
mutation-atomic on its public failure surface**: phase 1 preflights
every stored parameter open, skips frozen parameters *before* examining
their gradients (a frozen parameter with a stale gradient never
updates) and skips `grad=None`, validates every active gradient (an
open NativeTensor of exactly the parameter's shape/dtype/device,
index-named deterministic errors), and stages every `value - lr * grad`
natively at the autograd-unaware `NativeTensorCore` level — no graph
node possible, no NumPy (tripwire-tested), fresh owning temporaries
independent of every parameter and gradient — with any failure
releasing all staged temporaries and changing no value, version, or
gradient (the same optimizer recovers completely); phase 2 commits in
stored order through `copy_value_()` — identity, registration, aliases,
`requires_grad`, and gradients preserved by identity and value, one
version increment per updated parameter (zero-gradient updates still
increment: the owned value was replaced), staged temporaries released
on every exit path. One narrow documented limitation: after a fully
successful preflight the commits cannot fail through any public
surface, but an asynchronous interruption (e.g. KeyboardInterrupt)
between two commits would leave earlier parameters updated — each
individual commit stays atomic and version-consistent, and no private
rollback is manufactured. Gradients persist until **`zero_grad()`**,
which preflights all stored parameters open before clearing anything
(never a partial clear), then delegates to each parameter's own
`zero_grad()` — values, versions, identities, and registrations
untouched, frozen parameters included. The **v3.7 staleness contract
applies unchanged**: value-sensitive graphs built before `step()` raise
the existing deterministic stale error afterwards with gradients
untouched, and a fresh forward/backward trains on the updated values —
verified through a one-step
`NativeSequential(Linear → ReLU → Linear)` + `NativeMSELoss`
integration (exact SGD arithmetic on all four parameters, stable
identities, versions +1, gradients retained, `zero_grad`, fresh pass).
Deliberately **not** shipped: momentum, dampening, Nesterov, weight
decay, parameter groups, per-parameter learning rates, schedulers,
optimizer `state_dict`, checkpointing, `NativeAdam`, a training loop,
or the multi-iteration MLP proof. 19 focused tests
(`tests/test_native_sgd.py`, selector `-k "native_sgd"`) lock the
contract; the full suite passes at **1249 tests**. The next milestone
is **Advanced C++ v3.9 — the first end-to-end native training proof**:
a small deterministic multi-iteration forward → loss → backward →
`step()` → `zero_grad()` regression over the existing
Sequential/Linear/ReLU/MSE/SGD surface, asserting learning without
fragile exact-loss values — no new operations, layers, losses, or
optimizer features. **v3.9** delivers that proof — the first complete
multi-iteration **native CPU training run**, as an example and
integration tests with **zero source changes** to the native runtime,
autograd engine, layers, loss, module system, parameter system, or
NativeSGD. `examples/native_mlp_training.py` trains
`NativeSequential(NativeLinear(2, 8, seed=0), NativeReLU(),
NativeLinear(8, 1, seed=1))` on a fixed synthetic regression dataset —
8 samples, 2 features → 1 target, Python literals handed once to
`NativeTensor.from_array` (data construction at the explicit entry
boundary, deliberately distinguished from native computation) — for
**25 steps of `NativeSGD(lr=0.1)`**. Every iteration follows the
fresh-graph lifecycle: gradients confirmed cleared → fresh forward →
scalar `NativeMSELoss` → loss recorded through `to_numpy()` (the
established inspection exit, the only NumPy anywhere) → one-shot
`backward()` releasing the iteration's graph → every parameter
confirmed holding a finite gradient → `step()` (identities stable,
exactly one version increment per parameter, gradients retained) →
`zero_grad()` → the per-iteration prediction and loss tensors closed.
No `retain_graph` and no graph reuse, so the v3.7 stale guard never
fires in the loop — and a concise negative test proves deliberately
retaining an old sensitive graph across `step()` still raises the
existing deterministic stale error. The loss decreases
**monotonically every step**: 2.107864 → 0.396467 (step 5) → 0.086739
(step 10) → 0.032133 (step 15) → 0.016505 (step 20) → **0.009529** — a
99.5% reduction — and the run is bit-deterministic: repeated runs in
one process reproduce the exact loss history, final parameter values,
and version history (`[N, N, N, N]` after N steps; the final
evaluation pass and `zero_grad()` add nothing). Lifetime is explicit
throughout: model parameters, optimizer, and fixed data live for the
whole run; per-iteration tensors are closed each iteration; everything
the run created is closed on the way out (success or failure), and
`train()` returns plain Python scalars/lists only — never live native
tensors. Deliberately **not** shipped: `NativeAdam`, momentum, weight
decay, parameter groups, schedulers, optimizer state, checkpointing or
resume, batching, shuffling, validation sets, metrics, classification
losses, new operations or kernels, or any performance claim. 13
focused tests (`tests/test_native_mlp_training.py`, selector
`-k "native_mlp_training"`: end-to-end loss behavior, per-parameter
learning, exact version progression, identity/name/state-key
stability, exact-equality determinism, hand-driven gradient lifecycle,
cross-iteration accumulation control, the stale-graph guard, a NumPy
tripwire over a full training run, a source-level contract guardrail,
and the executable report) lock the proof; the full suite passes at
**1262 tests**. **v3.10** is the **Native CPU Training Stack
Integration Checkpoint** — the branch's first major usable native
training checkpoint and its merge-readiness milestone, adding **no
numerical behavior** (no new operations, kernels, layers, losses, or
optimizer features; no native-source, CI, export, or .gitignore
changes were needed). It corrected every stale public claim found in
the audit: the README no longer says "no C++ backend yet" or that the
experiment merely "started" — it now presents the stable NumPy line
and the experimental native line side by side, with a native
capability section, a verified native quickstart, and honest
limitations (float64/cpu only, no CUDA, no dtype promotion, no native
CNN stack, no adaptive native optimizer, no native optimizer
state/checkpointing, no dispatch into `tensorforge.Tensor`, no
universal speed claims); docs/project_summary.md and
docs/architecture.md now cover both lines, including the native
execution path (Python native modules → NativeTensor + Python-managed
graph → NativeTensorCore → ctypes → C++ CPU kernels) and the absolute
engine separation. One canonical
**[native support matrix](native_support_matrix.md)** now states the
exact supported surface — runtime/metadata and lifetime rules, all
twelve differentiable operations, the autograd guarantees
(broadcast/view/scatter backwards, graph release, `retain_graph`,
stale-version detection, rollback), and the full training stack
through the MLP proof — and the exact unsupported/future list (no
native divide/sqrt/reciprocal/exp/log/tanh/sigmoid/softmax, NativeAdam
planned as v3.12, no optimizer state or checkpoints, no native
Conv2d/MaxPool2d/Flatten, no CUDA, no float32/float16/bfloat16 or AMP,
no Transformers or distributed training, no Tensor integration).
Documentation guardrails were added or corrected in tests/test_docs.py:
the inverted README check (it must *never again* claim the backend is
absent, and must keep marking CUDA as future), support-matrix coverage
and unsupported-section honesty checks, an experimental-export lock
(exactly the nine intentional names, proven not to leak into the
stable top-level namespace), and a precision fix exempting the literal
`NativeAdam` from the roadmap's shipped-features-as-future ban (the
stable framework's Adam stays banned). Audits confirming no change was
needed: CI already builds the backend from source each run, hard-fails
`scripts/smoke_cpp_backend.py` before pytest, and runs the full suite;
`.gitignore` already covers the compiled library, caches, and build
output; the experimental exports were already complete. Full suite at
the checkpoint: **1264 tests**. Phase A and Phase B are complete;
**Phase C deliberately is not** — the intended sequence is **v3.11 —
native optimizer math primitives**, **v3.12 — NativeAdam**, **v3.13 —
native optimizer state**, **v3.14 — native checkpointing and
deterministic resume**, and **v3.15 — Phase C guardrails and
completion**, followed by the native CNN stack, the CUDA runtime,
dtype/AMP work, Transformer/text experiments, distributed training,
and the final portfolio release. **v3.11** delivers the optimizer math
primitives: native differentiable **`sqrt`** and **`reciprocal`**
through the complete stack — four new C++ kernels (`tf_core_sqrt`,
`tf_core_reciprocal`, and their contiguous fast paths, built on a new
generic `TfUnaryOp` odometer walker that generalizes relu exactly as
the binary walker generalizes add/subtract/multiply; both traversal
paths bit-for-bit identical), ctypes bindings sharing relu's
signatures, `NativeTensorCore.sqrt()`/`.reciprocal()` (open-tensor
gate, fresh owning row-major contiguous outputs, arbitrary
strided/offset views read directly), and differentiable
`NativeTensor.sqrt()`/`.reciprocal()`. The operations exist because
**NativeAdam (v3.12) needs a square-root denominator and reciprocal
scaling** — and general division stays deliberately unshipped, because
`reciprocal` + `multiply` compose both derivatives and the future Adam
denominator (`grad · reciprocal(sqrt(v) + eps)`). **Backwards use
saved forward results**: `d(sqrt(x))/dx = 1/(2·sqrt(x))` is computed
as `0.5 · reciprocal(saved output)` and `d(1/x)/dx = −1/x²` as
`−(saved output)²`, entirely at the autograd-unaware core level with
transients closed as consumed — each callback reads the recorded
output, never the parent's current value, so under the v3.7 rule
**neither operation records an expected parameter version**: mutating
a direct parameter input after forward leaves these edges valid with
gradients correct for the recorded forward (mixed graphs stay guarded
by their sensitive edges; no existing classification changed), and a
closed saved output fails backward deterministically with the graph
intact. **IEEE float64 exceptional values are documented and locked**:
sqrt of negatives is NaN (no exception), signed zeros are preserved,
+inf → +inf; reciprocal maps ±0 → ±inf and ±inf → ±0 (no exception,
no warning — NumPy's values), NaN propagates through both. The
raw-kernel registry boundary is untouched (the sum/mean precedent: no
raw-buffer wrappers, no registry-tuple changes). Deliberately **not**
shipped: NativeAdam, general division, exp/log/tanh/sigmoid/softmax,
rsqrt/abs/power, operator overloads, in-place arithmetic, optimizer
state, checkpointing, or dtype expansion. 18 focused tests
(`tests/test_native_optimizer_math.py`, selector
`-k "native_optimizer_math"`: kernel symbols and registry boundary,
core forward across contiguous/scalar/strided/offset/combined views,
the exceptional-value tables, wrapper graph construction, exact and
finite-difference gradients, explicit upstreams, chain and
shared-subgraph accumulation, graph lifetime with the
closed-saved-output failure, version independence plus the
still-guarded sensitive edge, a NumPy tripwire, and scope boundaries)
lock the contract; the full suite passes at **1282 tests**. **v3.12**
delivers **NativeAdam** — the native adaptive optimizer:
`tensorforge.experimental.NativeAdam(parameters, lr=0.001,
betas=(0.9, 0.999), eps=1e-8)`, minimal correct Adam over
`NativeParameter` objects with **no new C++ work and no new
operations** — the v3.7 mutation contract and the v3.11
`sqrt`/`reciprocal` primitives compose everything. The NativeSGD
parameter contract carries over unchanged (one materialization,
position-named validation of open `NativeParameter` entries, strict
identity deduplication in first-occurrence order, strong references,
nothing owned); `lr`/`eps` must be real, non-bool, finite, and
strictly positive and each beta real, non-bool, finite, and in
`[0, 1)` (exactly two, tuple or list; normalized to floats after
validation; read-only properties). **State is optimizer-owned and
eager**: per unique parameter, first/second moments as plain
graph-free `NativeTensor` zeros of exactly the parameter's
shape/dtype/device (never `NativeParameter`, never registered, never
in `model.state_dict()`) plus a per-parameter step counter (read-only
`step_counts`) driving bias correction — skipped frozen/`grad=None`
parameters never age moments or counters, a later-activated parameter
takes its first bias-corrected update at `t = 1`, a present
zero-valued gradient is active, and shared aliases advance once. A
constructor failure mid-allocation releases every buffer created so
far and touches nothing of the caller's. **`step()` extends the
two-phase mutation-atomic design with state**: preflight (optimizer
open, parameters open, every entry's m/v open and metadata-matched,
frozen skipped before gradient inspection, active gradients exactly
validated) → graph-free core-level staging of `m_new = β₁m + (1−β₁)g`,
`v_new = β₂v + (1−β₂)g²`, and `parameter_new = parameter − lr·m_hat·
reciprocal(sqrt(v_hat) + eps)` with bias corrections as native
reciprocals of scalar `1 − βᵗ` cores (Python exponentiation only for
the scalar coefficients; no division; `eps > 0` keeps the denominator
positive) → ordered commits through `copy_value_` (identity,
registration, `requires_grad`, and gradients preserved; one version
and one counter increment per updated parameter; staged moments
installed before the old buffers are closed; the staged parameter
value always released). Any public failure — later bad/closed
gradient, closed parameter, corrupted moment state, staging failure —
changes no value, version, moment, counter, or gradient, and the same
optimizer recovers; the two honest asynchronous-interruption windows
(between commits, and within an entry between parameter commit and
state installation) are documented rather than papered over with
private rollback. Gradients persist until the preflighted
`zero_grad()`; v3.7 staleness applies unchanged (old sensitive graphs
raise after `step()`, fresh forwards train on the updated values —
verified by a 20-iteration deterministic Sequential/Linear/ReLU/MSE
Adam training run with >50% loss reduction). **Lifetime is
explicit**: idempotent `close()` (context managers supported)
releases the owned moments exactly once and makes
`step()`/`zero_grad()` reject deterministically, while parameters,
gradients, and the plain-Python introspection surface stay untouched.
Deliberately **not** shipped: weight decay, AMSGrad, parameter
groups, per-parameter learning rates, schedulers, optimizer
`state_dict`/`load_state_dict`, checkpointing/resume, general tensor
division, fused kernels, in-place arithmetic, or a global no-grad
context; the stable `Adam` is untouched. 33 focused tests
(`tests/test_native_adam.py`, selector `-k "native_adam"`:
constructor/dedup/rejection behavior, constructor-failure state
release, full hyperparameter validation, state
initialization/aliasing/ownership, exact one-step and multi-step
oracle trajectories, determinism, skipping and late-activation
semantics, zero_grad preservation, failure atomicity with recovery,
stale-graph integration, lifetime/close, NumPy tripwire and scope
boundaries, and the end-to-end MLP run) lock the contract; the full
suite passes at **1315 tests**. **v3.13** delivers the **native
optimizer state contract** — in-memory
`state_dict()`/`load_state_dict()` on both native optimizers, with
**no new C++ work, no new operations, and no file I/O of any kind**
(native checkpoint archives are v3.14). One small versioned schema
(private shared helpers in
`src/tensorforge/experimental/native_optimizer_state.py` — plain
functions, deliberately not an optimizer base class): every state is
a plain dict carrying `format_version` (the int 1), the exact
`optimizer` type tag (`"NativeSGD"`/`"NativeAdam"` — a state can
never load into the wrong optimizer), the validated hyperparameters,
and ordered `{"shape", "dtype", "device"}` parameter metadata in the
optimizer's deterministic identity-deduplicated first-occurrence
order — restoration across instances is purely **positional**, and no
Python `id()`, pointer, module name, repr, parameter value, gradient,
graph data, or closed-state flag is ever serialized. NativeSGD's
state is pure Python metadata (`lr` + schema; its whole commit is one
validated assignment). NativeAdam's adds `betas`, `eps`,
`step_counts`, and `m`/`v` — **caller-owned snapshots**: plain
graph-free `requires_grad=False` NativeTensors (never
`NativeParameter`), each an independent owning contiguous native copy
sharing storage with nothing (not the optimizer's moments, not
parameters or gradients, not each other); repeated calls return
independently owned snapshots, closing a snapshot never affects the
optimizer, and the caller releases them when done. `state_dict()`
preflights everything (open optimizer, open parameters, intact
metadata-matched internal moments, valid counters) before creating
anything, and a snapshot failure partway closes every copy that call
created — never left to garbage collection — leaving the optimizer
usable. `load_state_dict(state)` treats the input as read-only
(never mutated, adopted, retained, or consumed) and runs
validate → stage → commit: exact key set (missing/unexpected reported
together), exact format version and tag, hyperparameters under the
constructors' full contracts, per-position metadata validation (exact
count and shape/dtype/device — no casting, reshaping, broadcasting,
or device movement), non-bool non-negative step counts, and every
moment entry an open **plain** NativeTensor of exactly the
parameter's metadata (NativeParameter and stable-Tensor entries
rejected; sequence fields accept tuple or list; deterministic errors
name stable field paths like `state['m'][1]`); then independent
optimizer-owned native copies of every input moment are staged (a
staging failure closes them all and changes nothing); then scalars,
counters, and moments commit together, the replaced internal buffers
closing only after installation — with the honest documented caveat
that NativeAdam's multi-assignment commit is not indivisible under an
asynchronous interruption. **Optimizer-state loading never touches a
parameter**: no value, version, gradient (by identity or value),
`requires_grad`, registration, alias, or model state key moves, so
the v3.7 stale guard never fires from loading alone and a retained
valid graph stays valid (proven by a focused test). Deterministic
**in-memory continuation** is proven end to end with the module state
contract: an uninterrupted N+M-step NativeAdam training run and an
N-step → snapshot → restore → M-step run produce bit-identical
losses, parameter values, moments, and step counts (model loading
increments each version exactly once per its existing contract;
optimizer loading increments none), and
frozen/`grad=None`/shared-alias/zero-state/late-activated parameters
round-trip exactly (a restored counter of 0 takes its first
bias-corrected update at t = 1). Deliberately **not** shipped:
`save_checkpoint`/`load_checkpoint`, `.npz`/JSON/pickle/metadata
files, paths, `map_location`, RNG or scheduler state, model
checkpoint wrappers, `strict=False`/compatibility modes, name-based
remapping, parameter groups, or dtype/device movement; the stable
optimizers are untouched. 21 focused tests
(`tests/test_native_optimizer_state.py`, selector
`-k "optimizer_state"`: exact schemas, independence of returned
containers, shared/frozen/late-active round-trips, full load
validation matrices, caller-ownership and non-aliasing proofs, atomic
failure with recovery, snapshot-failure cleanup,
parameter/autograd-isolation including the retained-graph proof, the
SGD next-step and Adam N+M continuation equivalences, a NumPy
tripwire, and source-level no-file/no-pickle guardrails) lock the
contract; the full suite passes at **1336 tests**. **v3.14** delivers
**native checkpointing and deterministic file resume** —
`tensorforge.experimental.save_native_checkpoint(path, model,
optimizer=None, metadata=None)` and `load_native_checkpoint(path,
model, optimizer=None)` over the existing v3.3 module and v3.13
optimizer state contracts, with **no new C++ work, no new operations,
and NumPy strictly as the explicit file-format boundary**
(`to_numpy`/`from_array`, `np.savez`, `np.load(...,
allow_pickle=False)`; tripwire-tested). **One pickle-free archive
format** (`"tensorforge.native_checkpoint"`, format version 1): a
`manifest` entry holding a JSON document as UTF-8 bytes in a 1-D
uint8 array (never an object array) that maps canonical model state
keys and the positional optimizer schema to deterministic zero-padded
indexed float64 array entries (`model::000000`…,
`optimizer::m::000000`…, `optimizer::v::000000`…), carries validated
hyperparameters, Adam step counts, and user metadata, and never
contains Python ids, pointers, reprs, gradients, parameter versions,
autograd graphs, or closed flags; duplicate references, missing
arrays, and unreferenced extras are rejected. **Metadata** is
recursively JSON-compatible (exact-type scalars — NumPy scalars
rejected — finite floats only, tuples normalized to lists per the
stable `json.dumps` convention, str-keyed dicts, cycles rejected) and
returns from loading as an independent plain dict. **Saving**
validates path/model/optimizer/compatibility/metadata first (the
optimizer's unique parameter sequence must be positionally identical
by object identity to the model's), snapshots through the existing
`state_dict()` contracts with every caller-owned snapshot closed in a
`finally`, then writes through a collision-safe temporary file in the
destination directory (`np.savez` onto an explicitly opened handle,
so NumPy can never silently rename it) and commits with one
`os.replace` — existing destinations are replaced atomically on
success and stay byte-intact on failure, no temporary file survives
either way, and pre-write failures create nothing and touch nothing
live. **Loading** enforces **strict optimizer presence/type
matching** (archive optimizer state requires a compatible same-type
optimizer, and vice versa — never silently discarded; a deliberate,
documented divergence from the stable loader's ignore-if-absent
behavior) and is validate → stage → commit: complete pre-mutation
validation under `allow_pickle=False` — manifest representation,
UTF-8, JSON, root type, exact format/version/fields, model keys and
every array's exact float64 dtype/shape against both manifest and
live destination, and the optimizer section through the same
validators the optimizer constructors use — then independent staged
`NativeTensor`s (failures close them all), then commits **only**
through `NativeModule.load_state_dict()` and
`optimizer.load_state_dict()`, with every staged tensor closed on all
paths and no live state aliasing archive arrays or staging tensors.
Committed behavior is exactly the components' documented contracts —
model loading increments each parameter version once and makes old
value-sensitive retained graphs stale; optimizer loading moves no
versions — and every ordinary failure (a locked corruption matrix:
invalid ZIP data, missing/malformed/non-UTF-8/non-JSON manifests,
wrong format/version/fields, key mismatches, duplicate or missing or
unexpected arrays, object dtypes, wrong dtypes/shapes,
presence/type/compatibility mismatches, invalid optimizer scalars and
counters, closed objects) happens before any mutation with model
values/versions/gradients and optimizer moments/counters/usability
preserved. The honest documented window: the model and optimizer
commits are two separate Python operations under asynchronous
interruption. **Deterministic file resume is proven bit-for-bit**
(N-step train → save → fresh pair → load → M-step continuation equals
the uninterrupted run in losses, values, moments, and counters, with
matching version deltas), NativeSGD round-trips lr to an identical
next step, shared aliases persist once, and
`examples/native_checkpoint_resume.py` demonstrates the whole flow in
a self-cleaning temporary directory. Deliberately **not** shipped:
scheduler state, random-state capture/restoration, dataloader state,
multiple models/optimizers, partial or name-remapped loading,
`strict=False`, `map_location`, merging, incremental or sharded
checkpoints, compression, encryption, URLs, pickle, or arbitrary
object metadata; the stable `tensorforge.serialization` is untouched.
17 focused tests (`tests/test_native_checkpoint.py`, selector
`-k "native_checkpoint"`: exports/signatures, argument and
closed-object rejection, model-only round-trip with the
retained-graph staleness proof, SGD and Adam restoration, metadata
round-trip/validation, the locked archive schema, atomic overwrite
and failure cleanup, the corruption matrix with recovery, presence/
compatibility/shared-parameter behavior, the bit-identical Adam file
resume, a NumPy tripwire, and source-level security guardrails) lock
the contract; the full suite passes at **1353 tests**. **v3.15** is the
**Phase C completion milestone** — native training stack guardrails and
hardening — which **closes Phase C in code**, completing the Phase A →
Phase B → Phase C arc, and adds **no numerical behavior** (no new
operations, kernels, layers, losses, or optimizer features, and no
source change to the native compute stack). It delivers one
cross-cutting completion test file (`tests/test_native_phase_c.py`,
selector `-k "native_phase_c"` — 10 integrated tests) that complements
the per-component suites by locking the invariants that span several
components at once: the full NativeSGD and NativeAdam training
lifecycles under a NumPy tripwire (finite loss, meaningful reduction,
version deltas equal to the active update count, graph-free
independently-owned optimizer state, and `close()` releasing only
optimizer-owned moments while the model stays trainable); the
shared-parameter story end to end (one `NativeParameter` through two
registered aliases and two forward paths, verified to be one entry
across module registration, backward accumulation, both optimizers,
state snapshots, and checkpoints, with an alias-preserving restore
whose continuation matches bit for bit); mixed
active/frozen/`grad=None`/zero-gradient collections and late parameter
activation; repeated optimizer-state and checkpoint-resume cycles (old
internal state closed after replacement, no caller snapshot aliasing
any live storage, no parameter version moved by optimizer loading, and
bit-identical two-lineage continuation); failure recovery at the
step-staging, state-load-staging, checkpoint-save, and
checkpoint-corruption boundaries (each leaving values, versions,
moments, counters, and gradients unchanged, temporaries and temporary
files cleaned up, and a later valid operation succeeding); the
four-way graph-staleness distinction (an optimizer step and a
model-state load make an old value-sensitive graph stale; an
optimizer-state load and a *failed* checkpoint load do not; a
*successful* checkpoint restoration does — gradients untouched whenever
the detector raises); lifetime/close discipline with no reliance on
garbage collection; and the public surface (exactly the twelve
intentional `tensorforge.experimental` exports, no leak into the stable
namespace, no optimizer base class, no checkpoint leak into stable
serialization, no unsupported optimizer feature, and no native CNN or
CUDA/dtype surface). It also finalizes the
[native support matrix](native_support_matrix.md) as the authoritative
Phase A–C snapshot, marks Phase C complete across the README, project
summary, architecture doc, roadmap, and design doc, and adds
documentation guardrails (`tests/test_docs.py`) preventing Phase C from
silently reverting to "in progress" and preventing optimizer state or
file resume from being described as future work — with CI, `.gitignore`,
the examples, and the benchmark audited and found already correct (no
change needed). The full suite passes at **1365 tests** (1353 plus the
10 cross-cutting completion tests and 2 new documentation guardrails).
**Phase C is complete.** The next major
native phase is the **native CNN stack** (`NativeConv2d`,
`NativeMaxPool2d`, `NativeFlatten`), which has not started, followed by
the CUDA runtime, dtype/AMP work, Transformer/text experiments,
distributed training, and the final portfolio release.

### Advanced C++ v3.16 — Phase D, the native CNN stack

**Advanced C++ v3.16 completes Phase D**, the native convolutional
stack, across thirteen milestones (D0–D12) built on the Phase A–C
foundation. **D0** locked the architecture contract in
`docs/native_cnn_design.md` — NCHW activations, OIHW weights,
cross-correlation (not flipped), floor output shapes with symmetric
padding, copy-then-compute for non-contiguous operands, the
fused-primitive/autograd split, the max-pool winner representation, the
C ABI families, and the test strategy — *before any numerical CNN code
was written*. **D1** shipped `NativeFlatten`, composed purely from the
existing reshape/copy autograd (refined during implementation to return
an **owning** result, so it composes safely inside a `NativeSequential`).
**D2–D5** added the internal CPU float64 convolution kernels (forward,
input gradient, weight gradient) as hidden C++ symbols with
dependency-free CTests, plus the locked bias-gradient reduction
sequence that needs no kernel at all. **D3/D6** exposed them through
exception-guarded C ABI wrappers and `NativeTensorCore` methods and
built the differentiable **`NativeTensor.conv2d`** primitive, whose
*conditional* version tracking records a parameter's version only where
an active backward callback rereads its value. **D7** added the
trainable **`NativeConv2d`** module (deterministic uniform conv fan-in
init, no new kernel or custom backward).

**D8–D10** did the same for pooling: the forward kernel that produces
the pooled values **and** a private saved-winner buffer in one pass
(float64 flat plane offsets with a `-1` padding sentinel, proved exact
against `H*W ≤ 2^53` in Python *and* at the ABI); the scatter-add
backward whose checked wrapper validates every winner value rather than
rounding one; the differentiable **`NativeTensor.maxpool2d`**, which
reads only its saved winners — never the input, never a recomputed
maximum — and therefore records **no** version snapshot, with the winner
buffer owned by the graph history and released at the same deterministic
points the graph is; and the parameter-free **`NativeMaxPool2d`** module.

**D11** proved it all works together: `examples/native_cnn_training.py`
trains Conv→ReLU→MaxPool→Flatten→Linear on eight fixed 6×6 images to
learn a genuinely spatial target (the strongest bright-to-dark vertical
edge) with `NativeMSELoss` and `NativeAdam`, dropping the loss 98.6% in
40 deterministic steps — and a run interrupted at step 15, checkpointed
with its optimizer state and resumed into a completely fresh
model/optimizer pair, reproduces the uninterrupted run **exactly**.
**D12** closed the phase with cross-cutting integration tests
(`tests/test_native_phase_d.py`), honest CNN characterization benchmarks
(`benchmarks/benchmark_native_cnn.py`, measurement only), **ASan/UBSan
validation** of the whole stack under Clang on Linux with no TensorForge
diagnostic and a clean LeakSanitizer pass over the instrumented native
CTests, documentation reconciliation, and durable capability guardrails.

Phase D added **no** new dtype, device, checkpoint schema, optimizer
feature, classification loss, normalization, dropout, or RNG: the native
line remains float64/cpu, pickle-free, and explicit. The full suite
stands at over 2000 tests with five native CTests in Release and Debug.

### Phase E — native classification and stable math

**Phase E completes the native classification stack** across eleven
milestones (E0–E10) built on the Phase A–D foundation. **E0** locked the
architecture contract in `docs/native_classification_design.md` — the
public surface, the numerical-stability strategy (maximum shift and
log-sum-exp, never `softmax().log()`), the backward-read and versioning
matrix, the host `int64` target contract (the native runtime has no
integer dtype), the graph-owned saved-probability lifetime, the C ABI
families and the new `cpp/src/classification.cpp` unit, and the
inventory placements — *before any numerical classification code was
written*; **E0 added no numerical behavior**. **E1** and **E2** shipped
the differentiable `exp` and `log` as a deliberate pair: the phase's two
backward archetypes, `exp` reading its **saved forward output** and
recording no parameter version, `log` rereading the **live input**
(`upstream × reciprocal(x)`, composed from the existing `reciprocal` —
no division operation was added) and therefore version-guarding a direct
`NativeParameter` parent with a deterministic stale-graph error. **E3**
and **E4** shipped the two fused probability transforms in the new
classification translation unit: a maximum-shift `softmax` and a
log-sum-exp `log_softmax` that is emphatically **not** `softmax().log()`
— it forms no probability buffer and performs no division, so a
probability too small to represent still gets an accurate finite
log-probability. Both have contiguous-only C ABIs with Core-level
Policy-B copy-then-compute, and both have saved-output backwards
**composed from existing Core operations** with no dedicated backward
kernel. **E5** shipped the fused `cross_entropy` **Core contract** — one
deterministic pass per row producing the maximum, the log-sum-exp, the
private saved probabilities, and the per-example loss, plus a backward
that turns those saved probabilities, the copied targets, and a native
one-element upstream into the gradient **without the logits even being
an argument**. Both guarded exports revalidate every target index
themselves rather than trusting Python, and targets are strictly
validated (`bool` and floating-point labels rejected, including integral
ones like `1.0`) and copied into independently owned read-only `int64`
metadata. **E6** shipped the differentiable
`NativeTensor.cross_entropy(targets, reduction="mean")` over that
contract — one scalar-output autograd node adopting the private
probabilities as **graph-owned state**, released exactly once with the
graph history, retained under `retain_graph=True` and across a failed
retryable backward — adding **no kernel, no ABI export, and no
numerical change**. **E7** completed the public surface: the stateless
`NativeCrossEntropyLoss`, whose entire forward delegates to that
operation, and the deliberately **reporting-only** `native_accuracy` —
no accuracy kernel, no C ABI export, no Core method, no autograd node,
and no native `argmax`; it materializes once through the explicit public
`to_numpy()` boundary, takes a NumPy `argmax`, and returns a plain
Python `float`, landing in the new `NATIVE_METRICS` inventory. **E8**
proved the assembled stack end to end without adding to it:
`examples/native_classification_training.py` trains a
`NativeConv2d(1, 4, 3, seed=0)` → `NativeReLU` → `NativeMaxPool2d(2)` →
`NativeFlatten` → `NativeLinear(16, 3, seed=1)` classifier over **raw
logits** on twelve fixed 6×6 images in three classes for 40
deterministic `NativeAdam(lr=0.05)` steps (loss **1.159638 → 0.000101**,
reporting accuracy **0.3333 → 1.0000**), and a run interrupted at step
**15**, checkpointed with its optimizer state and resumed into a
completely fresh model/optimizer pair reproduces the uninterrupted run
**exactly**. **E9** characterized the stack honestly
(`benchmarks/benchmark_native_classification.py`: seven cases, each
correctness-gated *before* timing, each labelled with the reference it
used, medians with spread after warm-up, `--smoke`/`--json` modes, and
**no speed assertion or timing threshold anywhere**). **E10** closed the
phase with cross-cutting integration tests
(`tests/test_native_phase_e.py`), Release **and** Debug native builds
(10/10 CTests each, zero warnings), Clang ASan/UBSan validation with
zero diagnostics attributable to TensorForge, a practical LeakSanitizer
pass with no native leak, and documentation reconciliation.

Phase E added **no** new dtype, device, or checkpoint schema — the
native checkpoint format stays `"tensorforge.native_checkpoint"`
**version 1** — and no normalization, dropout, RNG, native integer
tensor, division operation, or public `max`/`argmax`. The native line
remains float64/cpu, pickle-free, explicit, and free of implicit
stable/native dispatch.

### Phase F — native normalization and stateful buffers (F0)

**Milestone F0 opens Phase F, and it adds no numerical capability of any
kind.** F0 is an architecture-contract, roadmap, and
documentation-reconciliation milestone: it writes
`docs/native_normalization_design.md` and corrects the status drift left
behind after Phase E closed.

The design document locks, *before any numerical normalization code is
written*: the phase objective (a fully native, differentiable,
state-safe normalization stack — `NativeLayerNorm`, `NativeBatchNorm1d`,
and `NativeBatchNorm2d`); the public API and its naming (layer-norm
`weight`/`bias`; batch-norm `gamma`/`beta` with `running_mean` and
`running_var` buffers, matching the stable reference; no functional
helper, no `NativeTensor` normalization operation, and no `dtype`/
`device` constructor arguments while the runtime is float64/cpu only);
the decision that normalization is **composed from existing native
operations** (`mean`, `subtract`, `multiply`, `add`, `sqrt`,
`reciprocal`, `reshape`, broadcasting, `contiguous_copy`) so the phase
adds **no C++ kernel, no C ABI export, no ctypes declaration, and no
`NativeTensorCore` method**, inheriting an exact backward — including
differentiation through the batch mean and variance, which is never
detached — from the existing autograd; the layer-norm contract
(trailing-dimension normalization, population variance, `eps` inside the
square root, no buffers, identical behavior in train and eval mode); the
two batch-norm shape contracts (`(N, C)` reducing over the batch, and
NCHW `(N, C, H, W)` reducing over N/H/W with `(1, C, 1, 1)`
broadcasting); and three load-bearing safety rules.

The first safety rule is the phase's central insight. Native persistent
buffers carry **no value version**, and `multiply`'s backward **rereads
a live operand** while the existing stale-version check covers only
direct `NativeParameter` parents — so a live mutable `running_mean` or
`running_var` captured in a graph could silently change an
already-computed gradient with no error raised. The contract therefore
forbids it: **eval-mode batch normalization must take independent,
owning, graph-free snapshots** of the running statistics before using
them in the output graph, which is exactly why the buffers need no
version. The second rule is that the two running buffers update as **one
atomic transaction** — validate, stage graph-free values, commit while
preserving both buffers' Python identity, roll back completely on any
failure or interruption, close replaced cores exactly once, and move no
parameter version — reusing the staging/commit/rollback behavior
`NativeModule.load_state_dict` already proves, which milestone F1 will
extract into a private reusable primitive. The third is that
**registration implies no exclusive ownership**: no `NativeModule.close()`
is introduced, stateful examples and tests close both `parameters()` and
`buffers()` explicitly, and no contract relies on garbage collection.
Persistent running statistics ride the **existing** state-dictionary and
pickle-free checkpoint infrastructure with the format **unchanged at
version 1** — new persistent keys need no schema bump.

The ladder is **F0–F9**: F0 (this contract and reconciliation), F1
(atomic native-buffer state transactions and the `STATE_SUPPORT`
persistent-buffer correction), F2 (`NativeLayerNorm`), F3
(`NativeBatchNorm1d`), F4 (`NativeBatchNorm2d`), F5 (state, checkpoint,
and graph-safety hardening), F6 (deterministic normalized training and
exact resume), F7 (benchmark characterization with no speed assertion),
F8 (cross-cutting integration and semantic guardrails), and F9 (phase
closure). **F2–F9 have not started.**

F0 also reconciled the documentation: the support matrix, roadmap,
project summary, architecture doc, backend-experiments page, README, and
the experimental package docstring now all state that **Phase E is
complete and native classification is shipped**, that **Phase F is
designed but not numerically implemented**, and that BatchNorm,
LayerNorm, dropout, a native RNG, float32, CUDA, and AMP remain
unsupported. This release-history document itself had no Phase-E entry
before F0; the section above is that record. The documentation
guardrails in `tests/test_docs.py` were extended with durable semantic
checks derived from the live exports, registries, and files rather than
from frozen prose.

**F0 changed no numerical or runtime file**: no kernel, C ABI symbol,
ctypes declaration, Core method, tensor operation, module, buffer
helper, capability-inventory entry, export, benchmark, or example. The
public `tensorforge.experimental` export set and every backend
capability registry are byte-for-byte what Phase E left them.

**Milestone F1 — atomic native-buffer state transactions — is the
phase's first code, and it is state management and capability reporting
only: it adds no normalization mathematics.**

The staging/commit/rollback logic that makes native state replacement
safe already existed, inline, inside `NativeModule.load_state_dict`. F1
extracts and generalizes it into one private, reusable transaction —
`src/tensorforge/experimental/_native_state.py`, a
`NativeStateEntry(label, destination, make_core, source)` record and one
`replace_native_state(entries)` call — so the running-statistics update
F3 needs does not become a second, parallel implementation of the subtle
parts. That is exactly where a state-corruption bug would most likely be
introduced, which is why the extraction precedes any normalization layer.

The transaction runs in three phases with **one explicit commit
boundary**. *Plan* validates every destination and deduplicates entries
by destination **object identity**, so a shared parameter or buffer
reachable under several registered names is one destination — swapped
once, version-bumped once, released once; two entries for one
destination are the same request only when they name the same source
object, and any other duplicate is a genuine conflict rejected before
anything is staged. *Stage* produces every replacement core before any
destination is mutated, validating each one (open, owning, contiguous,
metadata-matched, and sharing storage with neither its destination's
current core nor another staged core); the transaction **takes ownership
of every core a factory returns** and will install it or close it,
exactly once, on every path. *Commit* swaps each destination's core and
then increments each affected parameter's version — **both inside one
rollback guard**, which is a refinement over the inline version, where
the increments sat outside it. The commit boundary is the point at which
every swap *and* every increment has succeeded: before it the
transaction is fully reversible, and a failure at either step restores
every swapped core and every moved version and closes every staged core,
so a failed transaction moves no version at all. Only the release of the
replaced cores is past the boundary; each is closed exactly once, every
one is attempted even if an earlier close raises, and the first such
failure is re-raised wrapped in a message that states plainly that the
state change itself succeeded — never swallowed. A defensive re-check
before each swap aborts the transaction if a destination's core changed
between planning and commit.

`NativeModule.load_state_dict` now delegates to that transaction. Its
public signature, validation order, error messages, missing/unexpected
key reporting, strictness, parameter and buffer identity guarantees,
parameter-version semantics, atomicity, and ownership behavior are all
unchanged, and its staging copies still go through the module's own
`_native_copy`, so the long-standing staging seam and the tests that use
it are untouched. `state_dict()` output, the checkpoint format, and the
checkpoint version are unchanged. Every pre-existing state, buffer,
module, and checkpoint test passes without modification.

F1 also reconciles one under-report: `STATE_SUPPORT` now reads
`("persistent_buffers", "state_dict", "load_state_dict",
"save_native_checkpoint", "load_native_checkpoint")`. Native modules
have held `NativeTensor`-backed non-parameter state — `register_buffer`,
`buffers()` / `named_buffers()`, persistent buffers in `state_dict` and
in checkpoints — since the pre-Phase-D hardening milestone, but this
tuple never said so, so `backend_info()` under-reported an existing
capability. Unlike the four names beside it, `persistent_buffers` names
a capability rather than one callable, and the guardrails resolve it
explicitly to that API rather than by relaxing the "every advertised
name is real" check. No other inventory changed: `TENSOR_CORE_OPS`,
`AUTOGRAD_OPS`, `RAW_KERNELS`, `NATIVE_MODULES`, `NATIVE_LOSSES`,
`NATIVE_METRICS`, `NATIVE_OPTIMIZERS`, and `UNSUPPORTED` are all
byte-for-byte what Phase E left, and `batchnorm` and `layernorm` remain
unsupported.

**F1 added no normalization capability of any kind** — no normalization
module, formula, forward or backward pass, eval snapshot,
running-statistic update, kernel, C ABI function, ctypes declaration,
public tensor operation, or experimental export. The transaction helper
is private (absent from `tensorforge.experimental.__all__`) and is not a
public in-place mutation API; `NativeParameter.copy_value_` remains the
only public controlled-mutation primitive in the native line. No
normalization code calls the helper yet — F3 will be its second caller.
**F2, `NativeLayerNorm`, is the next milestone.**

### A hardening milestone before Phase D

Between Phase C and the native CNN stack, a repair-and-hardening pass
(no new numerical features) tightened the Phase A–C foundation: a
coherent native C-ABI error contract (no C++ exception may cross
`extern "C"`; failures surface as `MemoryError`/`ValueError`/
`RuntimeError` — see `docs/native_abi_error_contract.md`), RAII
allocation safety with deterministic fault-injection tests, the C++
sources split into coherent components built through CMake (with a
direct-compile fallback), native-module **buffer infrastructure**
(`register_buffer`, persistent/non-persistent, in `state_dict` and
checkpoints), identity-aware cycle-safe stable-`Module` traversal with
Boolean-validated `train()` and fully atomic `load_state_dict`, and
accurate backend introspection.

### Versioning note

Two version concepts run in parallel and are deliberately kept
separate. **Milestone labels** — `v0.1 … v3.0` for the Python framework
line and `Advanced C++ v3.x` for the native line — track development
history and are what this document and the design docs use. The
**distributable package version** is a single number in
`pyproject.toml` (`0.1.0`), surfaced as `tensorforge.__version__` from
the installed metadata (`tests/test_version.py` pins the two together).
The package version is intentionally *not* bumped per milestone; it will
move when the project is first published.
