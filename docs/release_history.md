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
closure). **At F0, F2–F9 had not started**; all of them have since
shipped, and Phase F is complete.

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

### Phase F — native normalization and stateful buffers (F2)

**Milestone F2 ships `NativeLayerNorm` — the first native normalization
module.** It is the stateless half of Phase F, and it exercises the
composed normalization mathematics without touching the mutable-buffer
machinery at all.

`NativeLayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`
normalizes the trailing `len(normalized_shape)` dimensions of its input.
The whole layer is **composed from existing differentiable
`NativeTensor` operations** — `mean`, `subtract`, `multiply`, `add`,
`sqrt`, and `reciprocal` — so the existing Python-managed native autograd
**is** the backward: gradients flow to the input, and to `weight`/`bias`
when affine, through the mean and the population variance for free. The
forward is exactly `centered * reciprocal(sqrt(variance + eps))`, then
`* weight + bias` when `elementwise_affine=True`, with the **population**
variance (no Bessel correction) and epsilon **inside** the square root —
`sqrt(var + eps)`, never `sqrt(var) + eps` — both proved against
hand-computed values. Because `NativeTensor.mean` reduces one axis at a
time, the multi-axis mean is taken as a sequence of single-axis
`mean(keepdims=True)` calls; the retained size-1 dimensions keep the axis
numbers valid across the sequence, and no tuple-axis reduction was added
to `NativeTensor`.

The module is **stateless**: no buffers, no running statistics, identical
output in train and eval mode (forward never reads `training`).
`weight` (ones) and `bias` (zeros) are `NativeParameter`s registered in
that order, and only when `elementwise_affine=True`; the affine-free mode
registers no parameters and contributes no state-dictionary keys.
Construction validates every argument before any native allocation (a
rejected call leaks nothing), and a failed bias allocation closes the
already-created weight deterministically. Every forward returns a fresh,
owning, row-major-contiguous tensor — never a `NativeParameter` or a
borrowing view. Affine parameters ride the existing
`state_dict`/`load_state_dict` and native checkpoint path with the format
**unchanged at version 1**.

**F2 added no C++ code, normalization kernel, C ABI symbol, ctypes
declaration, `NativeTensorCore` method, custom backward, functional
`layer_norm` helper, or `NativeTensor.layer_norm` operation** — LayerNorm
is a *module composed from existing operations*, not a numerical
primitive, so no operation inventory grew. Only `NATIVE_MODULES` gained
`"NativeLayerNorm"` and `"layernorm"` left `UNSUPPORTED`; `"batchnorm"`
stays there until F4. **F3, `NativeBatchNorm1d`, is the next milestone.**

### Phase F — native normalization and stateful buffers (F3)

**Milestone F3 ships `NativeBatchNorm1d` — the first *stateful* native
numerical module.** It adds exactly one new thing to F2's working
composition: mutable model state, with the snapshot rule and the
transaction that make it safe.

`NativeBatchNorm1d(num_features, eps=1e-5, momentum=0.1)` accepts only
`(N, C)` input with `C == num_features`. **Training mode** normalizes
with this batch's own statistics —
`mean(input, axis=0, keepdims=True)`, then the **population** variance of
the deviations, then `reciprocal(sqrt(var + eps))`, then `* gamma +
beta`. Those statistics are computed with the *differentiable* native
operations, so the **training backward differentiates through the batch
mean and the batch variance**; detaching them would give a different,
wrong gradient, and central finite differences verify the input,
`gamma`, and `beta` gradients through them. **Evaluation mode**
normalizes with the stored running statistics instead and updates
nothing, so a single sample normalizes consistently.

The running statistics are **persistent native buffers**: `running_mean`
(zeros) and `running_var` (ones), plain owning contiguous gradient-free
`NativeTensor`s registered through the existing
`register_buffer(..., persistent=True)`. They never appear in
`parameters()`, no optimizer sees them, and no gradient reaches them.
State order is `gamma`, `beta`, `running_mean`, `running_var`, and they
ride the existing `state_dict`/`load_state_dict` and pickle-free
checkpoint path with the format **unchanged at version 1** — new
persistent keys need no schema bump.

Each training forward advances them by `(1 − momentum) * running +
momentum * batch`, using the **same** population `batch_var` that
normalized the batch (nothing is recomputed). The update values are built
as independent **graph-free** native state — the batch statistics enter
through `detach()`, a native storage-to-storage copy with no NumPy round
trip — and both buffers are committed as **one atomic transaction**
through the private F1 primitive (`_native_state.replace_native_state`):
both replacements are staged before anything mutates, both Python
identities survive, the old cores stay valid until the commit succeeds
and are then closed exactly once, a failure before the commit boundary
restores both buffers and closes every staged core, and **no parameter
version moves** — so advancing the statistics never makes an existing
graph stale. At the boundaries the convention is exact: `momentum=0.0`
leaves both running values numerically unchanged, `momentum=1.0` makes
them exactly the current batch statistics.

The load-bearing safety rule of §7 is implemented and proved: **a live
registered running buffer is never a rereadable graph operand.**
Evaluation first takes independent, owning, contiguous, graph-free
`(1, C)` snapshots of both buffers through the native copy path, and the
returned graph reads only those. A structural walk of the graph's
parents and graph-owned resources cannot find either registered buffer
by identity, and a training step, a `load_state_dict()`, or a checkpoint
load performed *after* an eval forward leaves that graph's backward
non-raising and numerically identical to the values used at forward
time. The snapshots are adopted as the output node's `graph_resources` —
the same D9 contract that owns MaxPool2d's winner buffer — so
`retain_graph=True` keeps them, an abandoned graph still frees them, and
they are released exactly once with the graph history.

Forward ordering makes a *failed* forward harmless: validate all live
state, build the complete differentiable output graph, prepare both
graph-free replacement values, commit them atomically, and only then
return the already-built output. An injected failure at any of those
points — including inside the transaction's staging and commit seams —
leaves both buffers, both parameters, every version, and every gradient
exactly as they were, with the native live-storage counters back at
their pre-forward baseline **without** `gc.collect()`, and a later valid
forward succeeds.

**F3 added no C++ code, normalization kernel, C ABI symbol, ctypes
declaration, `NativeTensorCore` method, custom BatchNorm backward,
functional `batch_norm` helper, or `NativeTensor.batch_norm`
operation** — BatchNorm too is a *module composed from existing
operations*, so no operation inventory grew. Only `NATIVE_MODULES`
gained `"NativeBatchNorm1d"`. `"batchnorm"` **stays** in `UNSUPPORTED`:
the name is unqualified, and removing it while only the 1-D shape exists
would over-claim. Every behavior lives in one shared private
implementation, so F4 supplies nothing but the NCHW rank, reduction
axes, and broadcast layout. **F4, `NativeBatchNorm2d`, is the next
milestone.**

### Phase F — native normalization and stateful buffers (F4)

**Milestone F4 ships `NativeBatchNorm2d` and completes the numerical
normalization *module* surface.** It is deliberately the smallest
milestone in the phase: the second public class supplies configuration,
and the implementation it runs on is F3's, unchanged.

`NativeBatchNorm2d(num_features, eps=1e-5, momentum=0.1)` accepts only
NCHW `(N, C, H, W)` input with `C == num_features`, and reduces over
**N, H, and W** — so each channel gets one population mean and one
population variance over all `N * H * W` of its values, and the channel
axis is never reduced. The reduction is three sequential single-axis
`mean(axis, keepdims=True)` calls, `(N, C, H, W)` → `(1, C, H, W)` →
`(1, C, 1, W)` → `(1, C, 1, 1)`; because every reduced dimension is
retained at size 1 the axis numbers stay valid across the sequence, so
no tuple-axis reduction was added to `NativeTensor`. Everything else —
`sqrt(var + eps)`, differentiating through the mean *and* the variance,
the `(C,)` persistent running buffers, the graph-free atomic two-buffer
update, the graph-safe `(1, C, 1, 1)` evaluation snapshots, the
validate → build → prepare → commit ordering, and the deterministic
mid-forward cleanup — is inherited verbatim.

**The class declares only shape configuration**: `_INPUT_NDIM = 4`,
`_REDUCTION_AXES = (0, 2, 3)`, `_TRAILING_DIMS = 2`, `_LAYOUT`, and
`_CHANNELS_LAST = (0, 2, 3, 1)`. It defines a docstring and not one
callable. Every method — `forward`, `_training_forward`, `_eval_forward`,
`_mean_over`, `_inverse_std`, `_snapshot`, `_blend`, `_affine`,
`_commit_running_state`, `_validate_forward`, `_registered_running`,
`__init__`, `__repr__` — is inherited from the private
`_NativeBatchNorm` **by function identity**, proved per method by a
test, and the source contains exactly one definition of each.

**The one genuinely new problem the 4-D shape poses is the channelwise
affine, and it is worth recording honestly.** NumPy-style broadcasting
aligns from the *trailing* axis, so `(N, C, H, W) * (C,)` would line
`gamma` up with **W**, not with the channel axis — a shape error in
general and silently wrong whenever `W == C`. The obvious fix, reshaping
`gamma` to `(1, C, 1, 1)`, was **rejected**: `multiply` records a
stale-value guard entry only for a direct operand carrying a value
version, and a reshaped `gamma` is an ordinary unversioned view. Under
that alternative, mutating `gamma` after a forward stops raising the
deterministic stale-parameter error and instead surfaces a bare
`RuntimeError: this NativeStorage has been closed` — verified by
deliberately building it — which is exactly the confusing failure the
design's §7.1 names, with a silent-wrong-gradient hazard behind it. So
the **activation** moves instead of the parameter: a borrowing
`transpose` carries the channel axis to the trailing position
(NCHW → NHWC), `gamma` and `beta` apply there as **direct rank-1
operands**, a second borrowing `transpose` carries the result back, and
`contiguous_copy` materializes the fresh owning contiguous NCHW output.
Both transposes are metadata-only and already differentiable, so **no
gradient logic was added**: `multiply`'s existing broadcast-aware
backward reduces `gamma`'s gradient over N, H, and W, `add`'s does the
same for `beta`, and `transpose`'s backward applies the inverse
permutation — which is *derived* from `_CHANNELS_LAST` rather than
configured, so the two halves can never drift apart. Channels-last is an
internal step of one method, never a public layout mode. The `(N, C)`
path is byte-identical to F3's, and every F3 test passes unchanged.

**F4 added no C++ code, normalization kernel, C ABI symbol, ctypes
declaration, `NativeTensorCore` method, custom BatchNorm backward, or
`NativeTensor.batch_norm` operation** — the NCHW shape too is a *module
composed from existing operations*, so no operation inventory grew. Only
`NATIVE_MODULES` gained `"NativeBatchNorm2d"`, and, with both BatchNorm
shapes finally live, `"batchnorm"` **left** `UNSUPPORTED`, which now
reads exactly `("dropout", "float32", "cuda", "amp")`. The native
checkpoint format is unchanged at **version 1**.

That completes the numerical normalization **module** surface —
`NativeLayerNorm`, `NativeBatchNorm1d`, `NativeBatchNorm2d` — but, at
F4, **not Phase F**: milestones F5–F9 (state/checkpoint and graph-safety
hardening, a deterministic normalized training run with exact resume, a
benchmark characterization, cross-cutting integration, and closure) had
not started then, so there was as yet no normalized training example, no
normalization benchmark, and no phase closure. **F5 was the next
milestone**; F5–F9 have all since shipped.

### Phase F — native normalization and stateful buffers (F5)

**F5 is complete: the exhaustive state, checkpoint, ownership, and
graph-safety hardening — tests and documentation only, no numerical
behavior and no new public capability.** F5 proved the §7–§10 contracts
the F3/F4 modules already obey by *executable test* rather than by prose,
so the state/checkpoint/graph-safety guarantees are demonstrated rather
than asserted.

A focused `tests/test_native_normalization_state.py` carries the
cross-cutting proofs the single-module milestones could not, over small
test-only fixtures — a nested 1-D model (`NativeBatchNorm1d` under a child
`NativeSequential`), a nested 2-D model (`NativeBatchNorm2d` beside
`NativeConv2d`/`NativeMaxPool2d`), a mixed model where a `NativeLinear`,
a `NativeLayerNorm`, and both BatchNorm shapes coexist, a shared-child
module, and an exact buffer-alias holder. It proves: exact ordered
canonical dotted state keys (parameters first, then persistent buffers,
BatchNorm local order `gamma`, `beta`, `running_mean`, `running_var`);
identity-deduplicated, first-discovered, cycle-safe traversal under shared
modules and buffer aliases; graph-free, owning, contiguous,
metadata-matched `state_dict()` snapshots independent of the model **in
both directions by storage identity**, with a partial-snapshot failure
closing every created snapshot without GC; the full strict
missing/unexpected/both-lists matrix and the non-strict partial-load
matrix over the buffer keys (only loaded parameters advance versions, a
buffer-only load moves none, one invalid matching entry aborts the whole
subset); exact shape/dtype/device validation that never casts, reshapes,
or moves (the dtype/device half driven through the narrowest property
seam, since the runtime is float64/cpu only); identity-, version-,
gradient-, and traversal-preserving successful mixed loads; mixed
parameter/buffer transaction rollback at staging, first install, a later
install after swaps, version adjustment, and a `KeyboardInterrupt` between
swaps; the **version-1** checkpoint manifest and archive gaining no
normalization-specific field, with BatchNorm buffers serializing as
ordinary model entries; exact **eval-output** reproduction across a round
trip for both shapes and through the full NCHW convolutional stack;
buffer-only checkpoint loads over the module's own registered buffers
replacing exactly those objects while sparing the parameters and leaving
an earlier eval graph valid, versus the **full** load staling the graph
through the *parameter*-version guard (attributed to `NativeParameter`,
never a buffer); a corrupt-archive matrix targeting the persistent
running-buffer keys mutating nothing; the checkpoint staging/commit and
atomic-save failure boundaries leaking nothing and preserving an existing
destination byte-for-byte; eval graphs holding no registered buffer object
or storage, only independent `(1, C)` / `(1, C, 1, 1)` snapshots; the §7
rule under `retain_graph=True` and across a **failed retryable backward**
(no partial commit, graph not freed, retry matching a clean control that
ignores the mutated running values); and a live-storage baseline over the
whole matrix, including explicit closes over `parameters()` **and**
`buffers()`. Two narrow generic-infrastructure additions — a
`state_dict()` partial-snapshot-failure test in
`tests/test_native_buffers.py` and a checkpoint load-staging-failure test
in `tests/test_native_checkpoint.py` — round out the coverage.

**F5 added no C++ code, module, operation, kernel, C ABI symbol, ctypes
declaration, `NativeTensorCore` method, custom backward, checkpoint schema
field, or export.** The exports, `NATIVE_MODULES`, `STATE_SUPPORT`,
`UNSUPPORTED`, and every operation inventory are exactly what F4 left, and
the checkpoint format stays at **version 1**. Every F5 proof passed
against the F0–F4 implementation as shipped; no locked-contract bug was
found, so no production file changed. **F5 completed the hardening, not
the phase.** At F5, F6–F9 (a deterministic normalized training run with
exact resume, a benchmark characterization, cross-cutting integration,
and closure) had not started, so there was as yet no normalized
end-to-end training example, no normalization benchmark, and no Phase-F
integration file. **F6 was the next milestone**; F6–F9 have all since
shipped.

### Phase F — native normalization and stateful buffers (F6)

**F6 is complete: the deterministic normalized training and exact
checkpoint-resume proof — one example and its integration test, no
numerical behavior and no new capability.** F6 assembles the pieces F0–F5
shipped into a single end-to-end proof that a native model running **both**
normalization families trains deterministically and resumes from a
checkpoint exactly.

`examples/native_normalization_training.py` trains
`NativeNormalizedRegressor` — a named `NativeModule` subclass
`hidden: NativeLinear(2, 8, seed=0)` → `batch_norm:
NativeBatchNorm1d(8, momentum=0.1)` → `relu: NativeReLU()` → `layer_norm:
NativeLayerNorm(8)` → `output: NativeLinear(8, 1, seed=1)`, so both
normalization families run in every forward and the state keys are
readable dotted names. `batch_norm` is the only stateful module
(persistent `running_mean`/`running_var`); `layer_norm` contributes
`weight`/`bias` affine parameters but **no buffers**. There is
deliberately no `NativeBatchNorm2d` or convolutional layer — the full
convolutional integration model is F8's scope. The task is one fixed
eight-sample two-feature regression over frozen literals (nothing
generated, shuffled, or sampled), the full batch in fixed order every
step, driven by `NativeMSELoss` and `NativeAdam(lr=0.05)` for 24 steps.

The training loss falls from ≈2.440245 to ≈0.027000 (a 98.9% reduction);
two independently constructed uninterrupted runs are **bit-identical** in
the whole loss history, every final parameter, the NativeAdam state, the
running statistics, the final training-step prediction, and the final
evaluation-mode output; and the global NumPy RNG cannot perturb the seeded
construction. Every parameter is reached by backward, the running buffers
receive no gradient and are excluded from the optimizer, BatchNorm running
state advances once per training forward, and evaluation reads it without
updating it. `run_resume_proof()` runs the schedule uninterrupted, then
interrupted at step 10 — saving model **and** optimizer state (the
BatchNorm running buffers ride as ordinary model state, format **version
1**), reloading into a **completely fresh** model/optimizer pair, and
continuing. The two agree **exactly** (equality, never a tolerance): the
prefix, the whole remaining loss suffix, the first resumed loss at the
split, every parameter, the complete model state, both `running_mean` and
`running_var`, the NativeAdam hyperparameters/counters/`m`/`v`, the final
training-step prediction, and the final **evaluation-mode** output. The
fresh target's parameter and buffer identities survive the load; it is
deliberately put in **eval** mode before loading and stays there
afterwards, proving the training flag is runtime state and not serialized
(it is switched back to train explicitly before continuing). Parameter
versions are not compared across the load — the checkpoint does not
serialize them, by design.

A complete normalized update — forward through BatchNorm and LayerNorm
(with the running-statistics update), scalar MSE, backward, the NativeAdam
step, and zero_grad — passes a strict NumPy/conversion tripwire, producing
exactly the unarmed reference's values. Every public helper representing a
completed run returns plain Python values only; the reporting helpers
close their `state_dict()` and optimizer-state snapshots; each run
explicitly closes its parameters **and** its buffers (there is no
`NativeModule.close()`); repeated steps and eval passes grow no native
storage; and the checkpoint lives in a temporary directory removed
automatically.

**F6 added no C++ code, module, operation, kernel, C ABI symbol, ctypes
declaration, `NativeTensorCore` method, custom backward, checkpoint schema
field, benchmark, or export** — one example and its integration test, with
every inventory exactly what F5 left and the checkpoint format at **version
1**. The example composed only existing modules, loss, optimizer, and
checkpoint APIs; no locked-contract bug was found, so no production file
changed.

### Phase F — native normalization and stateful buffers (F7)

**F7 is complete: the honest benchmark characterization —
`benchmarks/benchmark_native_normalization.py` and its test.
Measurement only: no numerical behavior and no new capability.** F7
characterizes what F2–F6 already shipped; it does not try to make
anything take less time, and it asserts no speed anywhere.

The harness (`BENCHMARK_NAME = "tensorforge.native_normalization"`,
`BENCHMARK_VERSION = "1.0"`) has exactly **nine** cases, in this order:
`layernorm_forward`, `layernorm_backward`,
`batchnorm1d_training_forward`, `batchnorm1d_eval_forward`,
`batchnorm1d_backward`, `batchnorm2d_training_forward`,
`batchnorm2d_eval_forward`, `batchnorm2d_backward`, and
`normalized_training_step`. Nothing else is measured — no checkpoint
I/O, no state-dictionary work, no constructor validation, no failure
path, no fault injection, and no isolated running-state transaction,
because the training-forward cases already include the real
running-statistics update.

**Correctness runs before timing, structurally.** Each case's gate is
called before the timing helper is ever reached, so a failed gate raises
before a single sample is taken and publishes no timing; the CLI turns
that into `correctness gate failed: …` on stderr with a nonzero exit and
a clean stdout. The tests prove it by substituting a finite, correctly
shaped, but numerically wrong native result (and separately a non-finite
one) and asserting the timer never started.

**Reference labels are honest.** Six cases carry `stable_tensorforge`
and run `tensorforge.nn`/`tensorforge.optim` on the *same* inputs,
epsilon, momentum, affine values, running state, initial parameters, and
optimizer hyperparameters. The three BatchNorm2d cases carry
`native_only` and publish **no** timing ratio, because the stable line
has no public `BatchNorm2d`. Their correctness gates remain rigorous: an
explicit NumPy NCHW population-statistics formula, an independent
channelwise-affine probe (smoke mode uses unequal `C`/`H`/`W` so a
channel/spatial broadcast mistake cannot hide), eval-mode state
neutrality with the registered buffers proved absent from the graph, and
for the backward the stable `BatchNorm1d` applied to the equivalent
`(N*H*W, C)` sample matrix with the input gradient transformed back to
NCHW. That transformed computation is a **correctness oracle only** —
timing it as a "BatchNorm2d reference" would compare a different module
plus two layout transformations, so the ratio would be misleading, and
the case says so.

**Methodology.** `time.perf_counter_ns()`, warm-up before measurement,
one measured sample per operation call, every sample retained, no
fastest-only reporting, and no timer-overhead subtraction. Setup and
cleanup run outside the timed region on every path; graph construction is
inside the timer for the forward and training-step cases and outside it
for the backward-only cases, which time exactly one one-shot
`backward()` on a graph rebuilt from cleared gradients each repetition.
Because a BatchNorm training forward advances persistent state, every
training-mode repetition builds a **fresh** module from the same
deterministic state. Each timed path reports `sample_count`,
`samples_s`, `median_s`, `min_s`, `max_s`, `spread_s`,
`relative_spread`, and `units`.

**F7 added no C++ code, module, operation, kernel, C ABI symbol, ctypes
declaration, `NativeTensorCore` method, custom backward, checkpoint
schema field, example, or export** — one benchmark and its test, with
every inventory exactly what F6 left and the checkpoint format at
**version 1**. No production file changed: the harness composes shipped
public APIs only, and no locked-contract bug was found. The JSON payload
is fully JSON-native, the human report ends in the local-characterization
disclaimer, **no result file of any kind is written**, and **no test, CI
job, or document asserts or commits a duration**.

### Phase F — native normalization and stateful buffers (F8)

**F8 is complete: the cross-cutting integration and semantic guardrails
— `tests/test_native_phase_f.py`, plus guardrail updates to
`tests/test_docs.py` and `tests/test_cpp_backend_info.py`. Tests and
documentation only: no numerical behavior and no new capability.** F8
proves the *interactions* the per-milestone suites cannot, and locks the
Phase-F surface with guardrails derived from real registries, exports,
and files.

**One integrated model.** The test-only `NativePhaseFClassifier` runs
`NativeConv2d(1, 4, 3)` → `NativeBatchNorm2d(4)` → `NativeReLU` →
`NativeMaxPool2d(2)` → `NativeFlatten` → `NativeLinear(16, 8)` →
`NativeBatchNorm1d(8)` → `NativeReLU` → `NativeLayerNorm(8)` →
`NativeLinear(8, 3)`, feeding **raw logits** to
`NativeCrossEntropyLoss` over the E8 fixed twelve-image three-class
dataset. Every Phase-D module family, **both** BatchNorm shapes, and
LayerNorm participate in one graph; no probability transform is inserted;
`NativeAdam` sees the twelve parameters and never the four buffers.

**The full interaction.** One graph reaches every trainable parameter
with a finite, correctly shaped gradient; the buffers receive none; both
BatchNorm pairs advance together in the training forward; parameter
versions and optimizer step counters each advance exactly once;
parameter and buffer identities never move; and one backward releases
the MaxPool2d winners and the cross-entropy probabilities exactly once.

**Deterministic training and exact resume.** Twelve deterministic
`NativeAdam(lr=0.05)` steps, interrupted at step 5, checkpointed (model
**and** optimizer, format **version 1**), reloaded into a completely
fresh pair, and continued. The prefix, the remaining loss suffix, the
whole loss history, every parameter, the complete NativeAdam state,
**all four** running-statistic buffers, the final training logits, and
the final evaluation-mode logits, predictions, and accuracy all match by
**exact equality**. The fresh target is deliberately in eval mode before
the load and stays there afterwards, proving the training flag is
runtime-only.

**Three saved-resource families at once.** BatchNorm eval snapshots
(`(1, 4, 1, 1)` and `(1, 8)`), MaxPool2d winners, and cross-entropy
probabilities coexist in one eval graph; neither registered running
buffer — object **or** storage — is reachable from it, while
`gamma`/`beta` legitimately are; one backward releases all three
families exactly once and a second release is a no-op; and an abandoned
eval graph releases its snapshots without touching registered state.

**Mutation attributed correctly.** A buffer-only
`load_native_checkpoint()` over a parameter-free holder aliasing all four
registered buffer objects — and, separately, a full training step — leave
an earlier eval graph's gradients exactly equal to a clean control, with
every buffer identity preserved and no parameter version moved. A **full**
checkpoint load and a direct `copy_value_` on a normalization affine
parameter each stale the graph through the unchanged v3.7 **parameter**
rule, commit no partial gradient, and leave a fresh forward working.

**Failure boundaries, stated honestly.** **A** — a BatchNorm
running-state transaction failure rolls *that pair* back completely,
while an earlier module's already-committed transaction legitimately
stands: transactions are **per module**, and one whole training step is
*not* globally transactional. **B** — a loss or backward failure after a
successful forward does not retroactively roll back the committed running
updates, and commits no gradient or optimizer change. **C** — an
optimizer staging failure commits nothing, closes every staged temporary,
and leaves the gradients usable for a clean retry. **D** — a
stale-parameter backward keeps the forward's committed update and
releases its saved resources on explicit close. **E** — a commit failure
while loading a real integrated checkpoint restores every value,
identity, and version and leaks no staged storage.

**F8 added no C++ code, module, operation, kernel, C ABI symbol, ctypes
declaration, `NativeTensorCore` method, custom backward, checkpoint
schema field, example, benchmark, or export** — one integration suite and
its guardrails, with every inventory exactly what F7 left and the
checkpoint format at **version 1**. No production file changed: the suite
composes shipped public APIs only, and no locked-contract bug was found.

### Phase F — native normalization and stateful buffers (F9, phase closed)

**F9 is complete, and with it Phase F: the closure milestone —
validation, documentation reconciliation, and the completion statement.
Documentation and documentation-guardrail tests only: no numerical
behavior and no new capability of any kind.** No C++ source, header,
CTest, C ABI export, ctypes declaration, `NativeTensorCore` method,
kernel, module, operation, loss, metric, optimizer, example, benchmark,
checkpoint schema field, or export changed, and **no numerical
production file changed at all**. Every number below was observed during
this closure — none is carried over from Phase D or Phase E.

**Windows environment.** Windows 11 Home 10.0.26200 (build 26200.8894),
PowerShell 5.1.26100.8894, x64 (Intel Core Ultra 9 185H), Python
3.13.14, uv 0.11.26, CMake 4.4.0, generator **Visual Studio 17 2022**,
MSVC **19.44.35228.0** (toolset 14.44.35207), Windows SDK 10.0.26100.0.

**Release and Debug builds.** Both configured fresh and out-of-source
**outside the repository** with `-DTF_BUILD_TESTS=ON`, so no source-tree
build directory was created. Release wrote its library to
`src/tensorforge/backends/` and passed **10/10 CTests** (0.78 s); Debug
wrote its library to a separate external directory and passed **10/10
CTests** (0.97 s). Both builds produced **zero compiler, zero linker,
and zero CMake warnings** — the only warnings in either log are 13
identical MSBuild `MSB8029` notices about the build tree living under
the temporary directory, an artifact of where this validation put its
build directories rather than a project diagnostic. Debug assertions are
genuinely enabled (`_DEBUG` defined, `NDEBUG` absent, `/Od /RTC1`), and
the Debug library never reached the package: the active
`_tensorforge_cpp.dll` stayed the 56,320-byte Release build linking
`MSVCP140.dll`/`VCRUNTIME140.dll`, while the Debug library is a separate
172,032-byte file linking `MSVCP140D.dll`/`ucrtbased.dll`.

**Windows Python regression.** `uv run pytest -q` with the Release
backend active: **3,628 passed, 5 skipped** before the closure edits and
again after the build validation. All five skips are the pre-existing
"backend is built; the unavailable path cannot be forced" cases — no
test skipped because the backend was missing.

**Sanitizer validation** (WSL2 2.6.1.0, Ubuntu 24.04.4 LTS, kernel
6.6.87.2-microsoft-standard-WSL2, CMake 3.28.3, Clang **18.1.3**,
`llvm-symbolizer-18`, GNU nm 2.42, Python 3.12.3 with NumPy 2.5.1 and
pytest 9.1.1 in an environment **outside** the repository): a fresh
`/tmp` build configured `-DCMAKE_BUILD_TYPE=Debug
-DCMAKE_CXX_COMPILER=clang++ -DTF_SANITIZE=address,undefined
-DTF_BUILD_TESTS=ON`, built with **zero project warnings**.
**Instrumentation was proved, not assumed**: `nm -D` shows **22
`__asan*`** and **13 `__ubsan*`** dynamic symbols beside the **50**
exported `tf_*` C ABI symbols; `file`/`readelf` confirm an ELF64 x86-64
object produced by "Ubuntu clang version 18.1.3"; and the library
*refuses to load* without the sanitizer runtime (`undefined symbol:
__ubsan_vptr_type_cache`) while loading cleanly with it preloaded. With
`ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_stack_use_after_return=1:detect_leaks=1`
and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`: **10/10 native
CTests pass** with leak detection on. Python-level runs preload the ASan
runtime (`LD_PRELOAD`) because the interpreter is not instrumented;
across 32 normalization and dependency suites — LayerNorm, both
BatchNorm shapes, the state/checkpoint hardening, the training proof,
the benchmark, the Phase-F integration suite, and the ownership,
autograd, buffer, state, parameter-versioning, storage, view, module,
optimizer, checkpoint, backend-introspection, and Phase-E dependencies
they exercise — **1,968 tests pass with zero ASan and zero UBSan
diagnostics** and no backend-unavailable skip. The F6 example reproduced
its exact resume and the F7 benchmark passed **all nine correctness
gates** (writing no result file) under the same sanitized library.

**LeakSanitizer, scope stated honestly.** The instrumented native CTest
binaries report no leaks at all. A dedicated, never-committed workload
drove one complete normalized lifecycle — the integrated `Conv2d →
BatchNorm2d → ReLU → MaxPool2d → Flatten → Linear → BatchNorm1d → ReLU →
LayerNorm → Linear` classifier with `NativeCrossEntropyLoss` and
`NativeAdam`, six training steps, a reporting eval pass with
`native_accuracy`, a version-1 checkpoint, a **fresh** model/optimizer
pair loading it, a resumed step matching the uninterrupted continuation
exactly, a non-contiguous NCHW input through the whole stack, an eval
graph carrying normalization snapshots retained across one backward and
released by the next, and explicit closure of the optimizers, every
unique parameter, every unique buffer, the views, and the base tensors —
and ended with the live-native-storage counter back at its baseline
(**0 → 0**) before exit. Running LSan over that *Python* process does
report 925,710 bytes in 830 allocations, but **not one leak frame names
`_tensorforge_cpp`, `tf_core_`, `tf_storage_`, or `tf::`**, and none
names a TensorForge C++ source path: every site is CPython (6,548
frames), the ASan runtime itself (293), libc (68), NumPy (24), or
`_ctypes` (8) — interpreter and module-initialization allocations a
non-instrumented interpreter never frees at shutdown. **No suppression
file was added** and `LSAN_OPTIONS` was left unset, so nothing was
hidden. The project's leak contract remains the deterministic
live-storage counters and explicit-cleanup tests, which assert an
*exact* return to baseline.

**Documentation reconciliation.** Every authoritative surface now agrees
that Phase F is complete and that F0–F9 all shipped, and the
milestone-era guardrails that asserted *absence* ("Phase F is in
progress", "F9 has not started", "no closure work is done") were
**converted into durable positive closure checks** rather than deleted:
F0–F9 must form one contiguous complete prefix with every milestone
marked complete, every status surface must describe Phase F as
complete and F9 as validation/documentation only, the closure artifacts
must exist and be documented, the sanitizer and Release/Debug evidence
must be present, no committed benchmark timing or performance claim may
appear, and no later phase may be marked started.

**Cleanup.** The Linux `.so` was removed from
`src/tensorforge/backends/` by a shell trap that fires however the
validation exits; the Windows Release DLL remains in place and active.
No `.so`, `.json`, `.csv`, `.npz`, sanitizer log, core dump, or build
directory was left in the repository.

**F9 changed only documentation and documentation guardrails**: the
Phase-F design document, the support matrix, the roadmap, this file,
`backend_experiments.md`, the project summary, `architecture.md`,
`README.md`, `CLAUDE.md`, the experimental package docstring, and
`tests/test_docs.py`. No locked-contract defect was found. **Phase F is
complete (F0–F9)** — which closes that phase, not the project: the
native line remains experimental, float64/CPU only, explicitly scoped,
and not production-ready, with `"dropout"`, `"float32"`, `"cuda"`, and
`"amp"` still unsupported and the kernels still deliberately naive.

### Phase G — native RNG and Dropout (G0–G10, phase closed)

**Phase G is complete.** It gave the experimental native line explicit,
deterministic, serializable randomness and the one consumer that motivates
it: inverted Dropout. The organizing rule, locked in
[native_rng_dropout_design.md](native_rng_dropout_design.md) at **G0**, is
that **random state is Python-managed while native random kernels stay
stateless** — a kernel receives the complete key for one operation and
never reads, holds, or advances a generator.

- **G1 — `NativeGenerator` and module generator state.** A pure-Python
  value holder carrying exactly an algorithm identifier, an algorithm
  version, an unsigned 64-bit seed, and a counter of *committed*
  stochastic calls. It owns no native storage and therefore has no
  `close()`, uses identity rather than value semantics, and never consults
  a global or process-wide random source (`seed=None` draws once through
  `secrets`). A private lock-protected, token-validated reservation
  protocol (`_reserve_call` → `_commit_call`/`_abandon_call`) makes it
  impossible for two callers to receive the same call index, and refuses
  state replacement while a reservation is live. `NativeModule` gained
  `_generators` as a **fourth** registration category beside parameters,
  buffers, and child modules, with its own
  `generator_state_dict()`/`load_generator_state_dict()` surface —
  `state_dict()` stays contractually `{name: NativeTensor}`.
- **G2 — the deterministic stateless Dropout-forward Core.** New
  `cpp/src/random.cpp` and `cpp/include/tf_random_internal.h` hold the
  locked `tensorforge.splitmix64` derivation as hidden `namespace tf`
  functions — a SplitMix64-family finalizer, a per-call stream key, a
  per-element bit derivation, and a 53-bit uniform with a strict `u < p`
  drop test — all `std::uint64_t` wrapping arithmetic with **no**
  `<random>`, `random_device`, clock, process id, address, or
  static/thread-local state. `tf::dropout_forward_contiguous` writes the
  output **and** the private multiplier mask in one pass. One guarded
  export, `tf_core_dropout_forward`, one ctypes declaration,
  `"dropout_forward"` in `TENSOR_CORE_OPS`, and the
  `NativeTensorCore.dropout_forward` / `_dropout_forward_with_mask` pair.
  **Committed known-answer vectors are asserted identically from C++ and
  Python.** The Core touches no `NativeGenerator` at all.
- **G3 — the differentiable `NativeTensor.dropout(p, *, generator)`.** One
  autograd node and one registry name (`"dropout"` in `AUTOGRAD_OPS`) — no
  C++, no ABI symbol, no backward kernel, because inverted Dropout's
  gradient is the existing `multiply` over the saved mask. The generator is
  **required and keyword-only**: no default, process-global, or
  module-global stream and no NumPy/`random` fallback. It owns the call
  transaction — validate, reserve one call, run the Core **outside** the
  generator's lock with the reservation's own seed and index, build the
  graph, commit **last** — so one successful stochastic forward consumes
  exactly one call, `p == 0` returns the caller's own object having drawn
  nothing, and every ordinary pre-commit failure releases everything,
  cancels, and leaves the same unconsumed index. The mask is **graph-owned**
  state, released exactly once with the graph history.
- **G4 — the `NativeDropout` module.** Stochastic in training, the input
  object itself in evaluation (so eval leaves **no gap in the stream**),
  identity at `p == 0`, over one registered generator it either owns (the
  default — independent streams) or shares (an explicit one, stored as the
  exact object). `seed` and `generator` are mutually exclusive. One name,
  `"NativeDropout"`, in `NATIVE_MODULES` and the experimental exports.
- **G5 — checkpoint format version 2 and exact generator restoration.**
  The format *name* never moves; the manifest gained exactly one field,
  `"generators"`, carrying `keys`, `entries` (with `seed` and `calls` as
  canonical decimal strings, because a `uint64` above `2**53` is not
  representable in an IEEE double), and `aliases` — the complete
  **registered path → canonical name** map, so the **alias topology** is
  restored, not merely the states. Sharing is identity, never state
  equality. Loading validates strictly in both directions against a real
  `named_generators()` traversal and fails in **prevalidation** with
  nothing touched; generators are restored **in place**, so identity and
  sharing survive. The whole load is one transaction whose single rollback
  guard spans the model, optimizer, and generator commits, and every
  participating state replacement runs under one process-wide lock in a
  fixed order, so concurrent loads **serialize** rather than interleaving.
  Version 1 stays loadable under its locked rules. One reporting-only
  registry name, `"checkpoint_generator_state"`.
- **G6 — hardening, no capability.** The §13/§14 ownership and failure
  matrices executed adversarially: the reservation transition matrix, the
  exact `uint64` boundary, forced concurrent interleavings with bounded
  joins and no sleeps, the Core's structural key properties beside its
  committed vectors, every pre- and post-commit failure position of the
  call transaction across four exception classes, all four graph-owned
  saved-resource families in one graph, a 76-case checkpoint corruption
  matrix, whole-transaction rollback at every commit position, and
  lifecycle loops against a measured live-storage baseline. **One runtime
  defect found and fixed** with the narrowest possible change: a failed
  cleanup step could make the failure's `__context__` chain *cyclic*,
  hanging any ordinary chain-walking reader.
- **G7 — exact stochastic resume, no capability.**
  `examples/native_dropout_training.py` trains the smallest model carrying
  all four TensorForge-owned state families at once (parameters, persistent
  BatchNorm buffers, a registered generator, and NativeAdam moments with
  per-parameter step counters). Two uninterrupted runs are bit-identical,
  and an interrupted run — released before the resume begins — reloads into
  a **completely fresh** model/optimizer/generator set built with a
  different seed and reproduces the uninterrupted run by exact equality.
  Two negative controls (restarting the batch schedule; re-seeding the
  generator) each diverge. External loop position is carried as
  **validated explicit metadata**, because checkpoint v2 does not capture
  data-loader position, shuffle state, epoch counters, scheduler state,
  Python's `random`, or NumPy's global RNG.
- **G8 — honest benchmark characterization, measurement only.**
  `benchmarks/benchmark_native_dropout.py`: 35 cases in eight families,
  every one correctness-gated **before** timing, the Core timed against an
  exact bit-for-bit NumPy implementation of the same derivation, the
  operation and module cases labelled `native_only` and publishing no
  ratio, and an untimed lifecycle pass returning live storage to baseline.
  **No speed assertion, no committed timing number, no CI timing
  threshold**, and no result file unless `--json-out` names one.
- **G9 — cross-cutting integration, evidence only.**
  `tests/test_native_phase_g.py` over one model carrying every registered
  state family at once, with two Dropout layers sharing one generator: four
  saved-resource families in one graph, exact version-2 resume with a
  diverging negative control, the generator-topology matrix, evaluation
  consuming no call anywhere, non-contiguous NCHW and strided views,
  whole-state rollback at every commit position, four deterministic
  concurrency cases, a Phase A–F regression matrix, and live storage
  returning exactly to baseline across success and failure cycles. No
  runtime file changed and no defect was found.
- **G10 — phase closure, and the capability boundary.** Validation,
  documentation, and **one registry line**. Fresh Windows **Release** and
  **Debug** builds (Visual Studio 17 2022, MSVC 19.44.35228.0, CMake
  4.4.0), each configured out-of-source outside the repository with
  `-DTF_BUILD_TESTS=ON` and each passing the full **11-test** CTest suite
  (11/11 in 0.86 s and 0.94 s) with **zero project compiler, linker, and
  CMake warnings**; Debug semantics genuinely enabled (`_DEBUG`, `/Od`,
  `/RTC1`) with no assertion exposing a defect, and the Debug library
  written elsewhere so the active runtime stayed the 58,880-byte Release
  DLL linking `MSVCP140.dll`/`VCRUNTIME140.dll` (the Debug library is a
  separate 176,128-byte file linking `MSVCP140D.dll`/`ucrtbased.dll`).
  A fresh Clang **18.1.3** `-DTF_SANITIZE=address,undefined` build in WSL2
  Ubuntu 24.04.4 with **instrumentation proved, not assumed** — `nm -D`
  shows 22 `__asan*` and 14 `__ubsan*` dynamic symbols beside the **51**
  exported `tf_*` symbols, and the library refuses to load without the
  sanitizer runtime. Under
  `halt_on_error=1:abort_on_error=1:detect_stack_use_after_return=1:detect_leaks=1`
  and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`: **11/11
  sanitized native CTests**, **3,166 sanitized Python tests** across 43
  Phase-G and dependency suites, the G7 example reproducing its exact
  resume, and the G8 benchmark smoke path passing every correctness gate —
  all with **zero ASan and zero UBSan diagnostics**. A practical
  LeakSanitizer lifecycle returned native live storage **exactly to
  baseline (0 → 0)**; its remaining process-exit allocations (926,478
  bytes in 831 allocations) contain **no TensorForge frame** — only
  CPython, libc, NumPy, `_ctypes`, and the ASan runtime — and **no
  suppression file was added**. Full Windows regression with the Release
  backend: **4,859 passed, 5 skipped**. Only after all of that did the
  boundary move.

**The capability boundary.** `"dropout"` was deliberately the one name in
both an implemented inventory and `UNSUPPORTED` for the whole of G0–G9:
the registry reports what is *closed and validated*, and a capability whose
entire value is exact reproducibility is not finished until reproducibility
has actually been demonstrated. At **G10** it left, and `UNSUPPORTED` now
reads exactly `("float32", "cuda", "amp")`. `SUPPORTED_DTYPES` and
`SUPPORTED_DEVICES` are unchanged, and the checkpoint format stays at
version 2 with `(1, 2)` supported.

**Phase G closes the phase, not the project.** The claim is narrow:
**native Dropout is supported in TensorForge's experimental native float64
CPU backend.** That is not a stable-framework claim —
`tensorforge.nn.Dropout` remains its own separate NumPy implementation and
was not touched — and float32, CUDA, and AMP remain unsupported. There is
no generic `rand`/`randn`/sampling API, no global or process-wide random
state, and no `Dropout2d`/`Dropout3d`. Reproducibility is exact **for the
state actually captured**; full-program determinism is not claimed, and
ordinary concurrent *training* is not claimed thread-safe. The native line
remains experimental, float64/CPU only, and not production-ready, with the
kernels still deliberately naive.

### Phase H — native CPU performance and runtime efficiency (H0, phase begun)

**Phase H is the current phase; it has begun, and milestones H0, H1, H2,
H3, H4, H5, H6, H7, and H8 are complete.** This section records H0; the
sections for the shipped optimization milestones follow.
The last sentence of the Phase-G entry above — "the kernels still
deliberately naive" — is what this phase exists to address, and H0 is the
milestone that decides *how*, by measuring rather than assuming.

**H0 shipped four things and nothing else:** the architecture contract
`docs/native_cpu_performance_design.md`; the unified baseline harness
`benchmarks/benchmark_native_cpu_performance.py`; that harness's
behavioral contract tests `tests/test_native_cpu_performance_benchmark.py`;
and documentation reconciliation across every status surface.

**No performance optimization has shipped.** H0 changed no C++, no C ABI
symbol, no ctypes declaration, no `NativeTensorCore` method, no autograd
operation, no module, no loss, no metric, no optimizer, no export, no
capability registry, no dtype, no device, and no checkpoint format.
`UNSUPPORTED` still reads `("float32", "cuda", "amp")`,
`SUPPORTED_DTYPES` still reads `("float64",)`, `SUPPORTED_DEVICES` still
reads `("cpu",)`, and the native checkpoint format is still
`tensorforge.native_checkpoint` version **2** with versions **(1, 2)**
supported. **Phase G therefore remains the latest completed phase.**

The contract locks the things a later optimization must obey: why CPU
efficiency precedes CUDA and dtype expansion (both would inherit whatever
runtime this phase leaves behind); the workload families and the
shape-selection rules for the smoke, full, and profiler configurations;
the timing, warm-up, repetition, and statistics methodology; the
correctness-before-timing rule; the exact-versus-tolerance policy; the
floating-point **accumulation-order** policy, whose default is that Phase
H preserves the existing order bit-for-bit and whose five-condition
escape hatch requires every existing exact-resume proof to be
re-established and every committed loss trajectory to be re-derived; the
determinism policy; the optimized-contiguous-dispatch, strided-fallback,
and **retained generic reference path** rules; the invariants Phase H may
not weaken; the allocation, scratch-workspace, SIMD, threading, and
optional-BLAS decision criteria; the cross-platform and sanitizer
requirements; the conditional H0–H11 ladder (H0–H10 as drafted, with H5
later inserting the copy and mutation-transfer milestone and pushing
reduction execution to H6); the explicit non-goals; the
closure requirements; and an adopt/adapt/reject decision for every
relevant idea taken from the Daedalus reference project.

The harness is the first in this repository to measure the runtime *as a
whole* rather than one phase's surface: 24 cases at H0 across twelve
workload families, separating up to nine declared implementation layers, with a
correctness gate that runs **before** the timing helper is ever reached,
honest reference labelling that publishes **no ratio** where no
equivalent exists, `--smoke` / `--json` / `--case` / `--workload` and a
focused `--profile CASE` mode, deterministic seeded inputs, explicit
cleanup that never relies on garbage collection, fresh state per
training-step repetition, an explicitly reset generator for the Dropout
case, and **no result file of any kind**. Checkpoint file I/O is
deliberately excluded from every training-step total; the in-memory state
surface is its own category.

The evidence is separated into what was directly measured, what is
strongly source-evidenced but not fully measured, and what remains an
unconfirmed hypothesis — with the minimal instrumentation a later
milestone would need recorded wherever H0's observability could not
settle a question, rather than a result being invented. The ranking is
deliberately not the one an unoptimized-kernel intuition predicts: the
largest measured factors are an allocator behavior and a memory **access
pattern** rather than raw arithmetic; the Python-side per-call metadata
path costs several times the ctypes boundary it wraps; and the
`NativeTensor` wrapper and its autograd graph node are measurably **not**
a bottleneck — a negative result that rules out a family of plausible
optimizations before any of them is written. The proposed **H1–H11 ladder
is explicitly conditional**: a milestone whose premise the measurement
does not confirm is narrowed, reordered, or dropped, and a memory pool,
scratch allocation, SIMD, threading, and BLAS are all currently
**rejected on evidence**, each with the criteria that would reopen it
recorded. Every number is a local characterization of one machine,
reported with its spread, and asserted by no test — there is no CI timing
threshold anywhere in this repository.


### Phase H — native CPU performance and runtime efficiency (H1)

**Milestone H1 — the explicit output-allocation contract — is complete,
and is the first Phase-H change to production code.**

H0 measured that native storage was value-initialized on construction
(`new double[n]()`), a full write pass over a buffer that most kernels
then overwrite completely — and that the cost of that pass scales with
the buffer while the allocation itself does not. H1 removed it, but only
where "the kernel overwrites the whole destination before reading any of
it" is a *proved* property rather than a plausible one.

**What shipped.** Exactly one new C ABI symbol,
`tf_storage_create_uninitialized`, sharing a single file-local body with
`tf_storage_create` so the two cannot drift apart on any shared
guarantee: identical size validation, zero and negative rejection, fault
injection, allocation-failure handling, thread-local error state, handle
shape, ownership, destruction through `tf_storage_destroy`, and
live-storage accounting. The buffer's initial contents are the only
difference, and the zero-initializing path remains the default with
byte-for-byte unchanged behavior. That takes the library from the pre-H1
baseline of **51** exported `tf_*` symbols to **52**, a count a scope
test verifies by parsing the built image's own export table.

On the Python side: one private keyword-only flag on
`NativeStorage.__init__` — deliberately on the constructor, so both
allocation kinds pass through the one function every live-storage
accounting hook in the suite wraps — plus the private
`NativeStorage._uninitialized` and `NativeTensorCore._uninitialized`
helpers. **No public API was added**: there is no `Tensor.empty`, no
`NativeTensor.empty`, no `empty_like`, no registry capability, and no
stable-framework surface, and a test asserts the absence of each.

**No global policy, and no runtime debugging hook.** There is no
allocator switch, no environment variable, no heuristic, no memory pool,
no scratch arena, and **no poison control or other mechanism anywhere in
the shipped library or the installed Python backend that can alter what
an allocation contains**. Each call site opts in explicitly, by name,
against a per-kernel audit table in
`docs/native_cpu_performance_design.md`, and a failed precondition means
the safe existing path rather than a guess.

**The audit corrected H0's sketch.** H0 had guessed "12 of 14"; the real
inventory differs. The three scatter-add backwards — both convolution
gradients and the pooling gradient — **do** qualify, because their
kernels zero their own whole span before accumulating, so the caller's
fill was pure duplication. `matmul` qualifies too, despite accumulating,
because it accumulates into a local register and assigns the destination
once. Two operations are **rejected** and keep a zeroed destination:
`sum`/`mean`, whose output *is* the accumulator so the zero is the
additive identity, and `narrow_backward`, which writes only the narrowed
region and whose untouched zeros *are* the gradient's value.

**Poison, where it lives, and what it proves.** Neither ASan nor UBSan
detects a read of an uninitialized *value* — that is MemorySanitizer's
job, and MSan needs a fully instrumented libc and CPython that this
project does not have, so **no MSan result is claimed**. Real
uninitialized memory is also a useless oracle, because a fresh OS page
reads back as zeros and a kernel with a hole would look correct. So
completeness is proved by filling an uninitialized allocation with a
chosen pattern — a quiet NaN with a distinctive payload, or a large
negative finite value — making any unwritten element a deterministic,
locatable value.

**That poison is injected exclusively by test infrastructure, around the
allocator.** The suite wraps the private `NativeStorage._uninitialized`
helper, which every uninitialized allocation funnels through; per
allocation the real constructor runs first (so the real
`tf_storage_create_uninitialized` allocates), the wrapper then fills that
buffer through the ordinary `tf_storage_fill` primitive, and the **same**
storage object is handed to the real operation, which runs the real
kernel over it. The pattern is therefore in place strictly after the real
allocation and strictly before the real kernel, which the suite asserts
directly by reading each buffer back through the very handle the
operation receives. **Nothing in the shipped library or the installed
Python backend participates**: there is no exported poison hook, no
thread-local flag, no environment variable, and no global mode, and a
scope test asserts their absence against the loaded image's export table
and over the backend module's namespace.

An earlier draft of this milestone did ship such a hook — an exported
`tf_test_set_uninitialized_poison` plus `cpp._set_uninitialized_poison` /
`cpp._uninitialized_poison` — on the argument that a thread-local seam
disarmed by default is harmless. That argument was rejected and the
mechanism removed in full: a symbol compiled into and exported from the
normal runtime is part of that runtime whatever its intended audience,
and a caller could activate it and change what production allocations
contain. The rebuilt proof lost no coverage and gained a control.

Four negative controls prove the detector can actually fail: a bare
uninitialized allocation is entirely poison; a partial-write kernel
(`tf_core_narrow_backward`) leaves poison in exactly its holes; an
accumulating kernel (`tf_core_sum`) returns NaN; and a *complete* kernel
given a deliberate hole — the real `tf_core_add_contiguous` told to write
8 of a 9-element destination — is rejected by the very assertion helper
every proof uses. A mutation test confirms the suite's teeth — moving
`sum` onto the fast path is caught by five independent tests.

The three tools are kept distinct on purpose: **poison** proves complete
destination initialization, **ASan/UBSan** prove memory-boundary and
undefined-behavior safety and say nothing about initialization, and
**LeakSanitizer plus live-storage accounting** prove lifecycle cleanup.

**Bit-identical.** H1 changed allocation, not arithmetic, so the
requirement is exact equality rather than a tolerance. Every enabled
operation and a complete eight-step `NativeAdam` training run are
computed twice — once as shipped, once with the private constructors
forced back to `zeros` — and compared element-wise, with the
uninitialized side running **under poison** so a hole cannot match by
luck.

**Measured, honestly.** Isolated, the removed fill is large and scales
with the buffer: roughly 52x at 2 MB, 119x at 8 MB, and 552x at 32 MB,
while falling *inside the noise and reading slightly negative* below
about 16,000 elements. End to end it is far more modest and frequently
inconclusive — clearly real for large memory-bound elementwise work
(about 1.5-1.8x on an 8 MB output), small and variable for normalization
and Adam, and with **no measurable effect** on Conv2d, the MLP training
step, or matmul, whose arithmetic dwarfs its allocation. The matmul rows
read slightly *below* 1.0 across three runs; that is this machine's
run-to-run variation, not a regression, and it is published as-is rather
than explained away. The primary comparison throughout is TensorForge
zeroed versus TensorForge uninitialized; `numpy.zeros` is reported for
context only and is explicitly not load-bearing, because `calloc` can be
answered with lazy zero pages and would measure the operating system
rather than an allocator TensorForge could adopt.

**Failure paths.** An uninitialized buffer must never reach a caller, so
every enabled site now closes its destination on failure — which required
adding the guard to the sites that previously relied on garbage
collection. Tested at invalid arguments before allocation, injected
allocation failure, native kernel failure after allocation across eight
kernels, a Python-side wrapper failure, a failed copy inside
`from_array`, a rejected fill value in `full`, and fifty interleaved
success/failure cycles with live storage returning to an exact baseline
each time — plus fifty more of those cycles with the poison wrapper
installed, and a failure of the poison fill itself, which closes the
storage it had just allocated. No check relies on garbage collection to
reach its baseline.

**Nothing else moved at H1.** No memory pool, scratch arena, SIMD,
threading, BLAS, fusion, matmul loop change, or reduction fast path — the
matmul loop order became H2's subject, and the rest are later,
still-conditional milestones. `UNSUPPORTED` still reads
`("float32", "cuda", "amp")`, `SUPPORTED_DTYPES` still reads
`("float64",)`, `SUPPORTED_DEVICES` still reads `("cpu",)`, and the
native checkpoint format is still version 2 with versions 1 and 2
supported. `tf_storage_create_uninitialized` remains the only C ABI
symbol H1 added.


### Phase H — native CPU performance and runtime efficiency (H2)

**Milestone H2 — native matmul memory access — is complete, and is the
first Phase-H milestone to change how a numerical kernel executes.**

H0 measured that the production matmul was bound by its memory-access
pattern rather than by scalar arithmetic: its `i`-`j`-`k` inner loop
walked the right operand *down a column*, so a row-major operand — every
weight a `NativeLinear` holds — touched a new cache line on every step.
The tell was that a *strided* transposed operand ran 2.6x **faster** than
a contiguous one, because a transposed view makes that inner loop
contiguous. H2 fixed the arrangement without touching the arithmetic.

**Two paths, one unchanged export.** `tf_core_matmul` now dispatches
between `tf::matmul_generic_strided` — the pre-H2 triple loop, kept
verbatim as the **retained generic reference path**, shipped and reachable
through ordinary production dispatch — and `tf::matmul_row_sweep`, an
`i`-`k`-`j` sweep over four destination rows at a time whose innermost
loop walks a row of the right operand and a row of the output
sequentially. Both are hidden-visibility C++; the choice is made inside
the kernel from the stride metadata it already receives.

**The optimized path's preconditions, all from metadata:** the right
operand's column stride is exactly 1, the inner dimension is non-empty,
and the result has at least 8 columns. Everything else — a transposed
right operand, any other non-unit column stride, a narrow result, an
empty inner dimension — takes the generic path, and that is a design
choice rather than a gap: `i`-`j`-`k` is the better order precisely when
the right operand's rows are the unit stride. A transposed *left* operand
beside a row-major right one, which is `db = a.T @ upstream` in the matmul
backward, does qualify. Selection is total, pure, deterministic,
side-effect free, and independent of pointer values, alignment, wall
time, environment variables, and CPU-feature probes; a failed
precondition is a fallback, never an error.

**Cache blocking was measured and rejected.** The milestone title
anticipated it; the evidence did not support it. Twenty-two blocked
`BI x BJ` variants were compiled into a throwaway measurement library and
timed against unblocked row sweeps across 25 shapes, with every variant's
output compared bit-for-bit against the current production loop first.
The unblocked sweep won at every non-trivial size — 5.50x versus 3.33x
at 384 cubed, 4.67x versus 3.10x at 256 cubed — because a tile shortens
the inner loop and adds a zero-and-store pass, while a full-width sweep
keeps one long vectorizable loop and the destination row hot. H2
therefore shipped the simpler superior design and published the negative
blocking result. The row block (4) and the column threshold (8) are
compile-time constants in a shipped header: no autotuning, no runtime
probe, no stored machine-specific measurement.

**The numerical contract, in four parts.** H2 does not claim
unconditional bit identity, and the four claims it does make are
separate.

*One: accumulation order is preserved exactly.* For every output element
the row sweep starts from the same 0.0 and takes the same products in the
same ascending `k` order, so no addition is reassociated, no partial sums
are combined, no accumulator width changes, and no fused multiply-add or
vector reduction is introduced. The `0.0 +` on the assigning pass is
written out deliberately, because `0.0 + (-0.0)` is `+0.0` and dropping
it would flip the sign of a zero result.

*Two: every non-NaN result is bit-identical* — signed zeros, infinities,
denormals, the smallest normal and the largest finite magnitudes
included. Asserted as raw IEEE-754 bit patterns rather than tolerances,
in the native CTest across a 1-65 dimension sweep with primes and both
sides of every boundary, in Python across the Core, `NativeTensor`, the
autograd node, `NativeLinear` and both optimizers, and in the benchmark's
own gate before any timing. **This is the part every practical claim
rests on**: every committed loss trajectory and every bit-exact resume
proof in the project runs on finite data, so deterministic training and
exact checkpoint resume are covered completely.

*Three: NaN-class equivalence.* Whenever either path produces a NaN, both
do, in exactly the same positions, and both are quiet. Neither path can
produce a signaling NaN.

*Four: NaN payload bits are outside TensorForge's numerical contract*,
and the two paths may differ in them. This is measured, not assumed. On
x86-64 the addition returns the destination operand's NaN when both are
NaN, and which addend the compiler places there is an instruction
selection decision C++ cannot express — in the `i`-`j`-`k` kernel it is
the product, in every `i`-`k`-`j` structure the accumulator. Ten
source-level formulations were measured in a focused MSVC Release
harness: compound versus explicit assignment, named locals for the
accumulator and the product, `__restrict`, disabled inner-loop
vectorization, and both a 4x64 and a 4x4 stack accumulator tile. All ten
`i`-`k`-`j` spellings agreed with one another and differed from the
reference; the only structure that reproduces the reference's payloads is
the `i`-`j`-`k` order H2 exists to replace. Payload parity is therefore
unavailable short of abandoning the optimization, and forcing it was
rejected — as were a NaN-detecting fix-up pass, a compiler-specific
pragma that does not even work, and every fast-math-adjacent trick.
Measured across builds: MSVC Release differs on 162 of 208 results in a
NaN-saturated matrix; MSVC Debug and Clang 18 differ on none. Both
conform, and the tests assert nothing in either direction.

Calling that difference "not a behavioral difference" would be wrong. It
is one — the bits of a NaN result can differ between two paths a caller
cannot choose between. What is true is narrower: the difference is
confined to the payload bits of a value that is already NaN, those bits
have never been part of any TensorForge contract, and a NaN result means
the computation has already left the supported numerical domain.

**H1's contract survives, for a new reason.** The row sweep accumulates
in the destination, so its safety argument is not H1's. Its `k == 0` pass
*assigns* every element of every row in the group before anything
accumulates into one, which is why the dispatch predicate requires a
non-empty inner dimension. Both paths are proved by poison tests with
both patterns across the row-block and column-threshold boundaries, by a
check that the same product over NaN-poisoned, finite-poisoned and zeroed
destinations agrees bit for bit, and by a partial-write negative control
that proves the detector can fail.

**The measured result, honestly.** Roughly 4.1-4.7x at 384 cubed and
4.2-4.5x at 128 cubed on the Core matmul; about 4-6.8x on a
`NativeLinear` forward; 1.7-2.5x on its backward, where only one of the
two matmuls qualifies by design; 2.0-2.4x on a 128x256 MLP Adam step.
And **no measurable effect below roughly 32 cubed, or on a small MLP
step**, where a fixed ~10 microsecond per-call Python cost dominates —
control cases whose compiled code did not change at all varied by
0.50-1.44x in the same runs, which is the noise floor those rows sit
inside. Nothing here is asserted by any test and no CI job runs it.

**Nothing else moved.** The pre-existing raw `tf_matmul_tiled` was
inspected and deliberately not adopted — it cannot read a stride, carries
no error contract, and zeroes its destination before accumulating, which
is the pass H1 removed — and it stays as the standing benchmark
experiment on no production path. H2 added **no** exported C ABI symbol,
leaving the library at **52**, and no kernel selector, block-size setter,
benchmark hook, dispatch tracer, environment variable, or public dispatch
control of any kind. `UNSUPPORTED` still reads
`("float32", "cuda", "amp")`, `SUPPORTED_DTYPES` still reads
`("float64",)`, `SUPPORTED_DEVICES` still reads `("cpu",)`, and the
native checkpoint format is still version 2 with versions 1 and 2
supported. Phase G remains the latest *completed* phase; Phase H remains
the current one.

### Phase H — native CPU performance and runtime efficiency (H3)

**Milestone H3 — native metadata and dispatch efficiency — is complete,
and is the first Phase-H milestone that is Python-only.** No C++, no C
ABI symbol, no ctypes declaration, and no kernel changed, so the library
still exports exactly **52** `tf_*` symbols.

H0 measured a fixed 18.6–22.6 microseconds per native operation, of
which only about 1.9 was the ctypes boundary — the rest was Python-side
shape and stride work — and H2 sharpened the question by making small
matmuls kernel-cheap while leaving them dominated by that floor. H3
answered it.

**The measured cause was redundant re-validation, not expensive work.**
One `shape_info` call ran `_as_int_tuple` **four** times over a tuple
that was fully validated after the first pass, and computed the
row-major strides **twice**; `NativeTensorCore.zeros` then validated the
caller's shape a *second complete time* by calling `numel(shape)` and
constructing a view from the same raw shape. Instrumented call counts —
taken with test-local monkeypatching, since **no production counter
exists or may exist** — put that at **815** `_as_int_tuple` calls per
MLP training step, 815 per CNN step, and 604 per `NativeAdam` step. This
confirmed and refined design §3.1/B3, and answered §3.2's open question
about which helper contributes what.

**One normalization boundary.** The private `_normalized_layout`
performs exactly the checks `shape_info` always performed, in exactly
the same order, with exactly the same messages, and normalizes the shape
**once**. Everything derived from it — the row-major strides, the
element count, the contiguity comparison — comes from private
`_checked` primitives that validate nothing *because nothing is left to
validate*. Each public helper (`row_major_strides`, `numel`,
`reduce_shape`, `broadcast_shapes`) is now its own validation followed
by the matching primitive, so the two can never compute different
answers — a property the suite asserts by comparing them across a shape
matrix.

**Two view constructors, one binding.** `NativeTensorView` keeps its
normalizing public constructor and gains a private `_from_validated`
that skips **only** that normalization. Both funnel through a shared
`_bind`, which still performs the storage open check and the **full
reachable-offset bounds check** — deliberately not skipped, because
bounds depend on the storage size, not on the metadata, and a derived
layout can still be handed a storage that has since been closed. The
element count and the contiguity flag are **derived inside** the private
constructor rather than passed to it, so a caller cannot supply an
inconsistent pair; that is why H3 ships a separate private constructor
rather than the misusable `validated=True` flag it was warned against.

**Per-view layout arrays, memoized lazily and read-only.** The `int64`
shape/stride arrays the strided C ABI takes are built at most once per
view. **Staleness is impossible by construction rather than prevented by
invalidation**: a view's layout is assigned exactly once, in `_bind`,
and `reshape`, `transpose`, `T`, and `narrow` all return *new* views, so
no invalidation is ever required and **none exists**. The arrays are
read-only, so no caller can mutate a view's metadata through them, and
they hold copied integers rather than a handle, so they keep no native
storage alive and introduce no reference cycle.

**Nothing was weakened.** Every rejection still happens, with the same
exception type, the same message, and the same shape-then-strides-
then-offset ordering — asserted by feeding the constructor metadata with
two and three simultaneous faults. Nothing global was introduced: no
shape cache, stride interning, process-wide dictionary, weak-reference
machinery, or thread-local state. No public API of any kind was added:
no cache control, statistic, reset, profiling counter, dispatch
selector, or environment variable, and `cpp.py` reads no environment
variable at all.

**Measured, with the negatives published.** `shape_info` 2.6–4.5x
faster, view construction 3.2x, `_as_int_tuple` calls per MLP step
**815 → 149** and per CNN step **815 → 150**. End to end: a one-element
allocation 2.1x, `reshape` 3.1x, a view chain 2.4x, a small `add` 1.56x,
`NativeAdam` on a small MLP 1.42x, an **MLP training step 1.43x**, a
**CNN training step 1.29x**, and a **normalized training step 1.51x**,
which cut the `NativeAdam` step's gap against the stable line from 39.8x
to 31.9x. Against that: **large kernel-bound work shows no measurable
change in either direction** — 384-cubed, 512-cubed, and 128-cubed
matmul, 256-squared elementwise, and 128-squared reduction all sit
inside their own run-to-run spread, so H2's large-matmul result is
intact. The layout-array cache is the weakest of the three changes and
was kept on measured merit rather than principle: isolated, it saves
0.6–1.5 microseconds per *strided* small operation and nothing on large
ones or on a contiguous training step, and even a deliberately
cold-cache measurement is no slower than pre-H3. Object footprint is
byte-identical for a cold view and +328 bytes for one that actually
takes a strided path; in a full MLP step only **5 of 134** views ever
populate it.

**One methodology finding is published rather than buried.** At the
harness's *default* 11 repetitions a case appeared to regress 35%; the
result was internally impossible (a thin layer "slower" than the wrapper
around it), and at 201 repetitions the same case measured **1.19x
faster**. No default-repetition figure is quoted as H3 evidence, and the
harness gained two `native_only` cases — `metadata_preparation` and
`ctypes_boundary` — that finally decompose B3's single figure into its
Python and boundary halves, which is exactly the instrumentation
design §3.4 said a later milestone would need.

`UNSUPPORTED` still reads `("float32", "cuda", "amp")`,
`SUPPORTED_DTYPES` still reads `("float64",)`, `SUPPORTED_DEVICES` still
reads `("cpu",)`, and the native checkpoint format is still version 2
with versions 1 and 2 supported. Phase G remains the latest *completed*
phase; Phase H remains the current one.

### Phase H — native CPU performance and runtime efficiency (H8)

**Milestone H8 — native elementwise traversal and composed allocation
efficiency — is complete**, the fourth Phase-H milestone to change C++
and, like H2, H5, and H6, **not the ABI**: the library still exports
exactly **52** `tf_*` symbols.

H8 entered with **two** candidate tracks and an explicit instruction not
to force both into production. **Track A — elementwise traversal — was
confirmed and is the milestone. Track B — composed normalization
allocation — was confirmed only as a memory result and is reported as
timing-neutral.**

**The cost was decomposed rather than assumed.** Driving the generic
strided walker and the flat contiguous walker through identical `ctypes`
calls on identical *contiguous* data showed the odometer costing
**1.60x-6.42x** the flat loop. A separate sweep showed **all broadcasting
is on the odometer** — there is no broadcast fast path at all, because
`_binary_core_op`'s broadcasting path builds per-operation strides and
hands them to the same generic walker. A standalone binary with an
anti-hoisting guard then split the odometer's cost four ways at
`(256, 256)` contiguous `add`: the shipped odometer-plus-function-pointer
at **123.5 us**, templating alone **81.3 us (1.52x)**, collapsing alone
**63.6 us (1.94x)**, and **both together 11.5 us (10.7x)**. Neither change
is worth much alone and together they are worth an order of magnitude,
because only their combination lets the compiler emit a vector loop. The
same run showed the **existing flat contiguous kernel was itself hobbled**
by its indirect call — 21.0 us against 11.7 us for the identical loop with
the operation as a compile-time constant.

**The architecture** reuses the dispatch shape H2, H5, and H6 each proved:
one hidden metadata builder, inside the existing export, no new symbol,
the pre-milestone traversal retained. New
`cpp/include/tf_elementwise_internal.h` declares `tf::build_unary_plan`
and `tf::build_binary_plan` plus the templated traversals;
`cpp/src/elementwise.cpp` defines the builders and dispatches to them. A
**plan is an operation-local normalized descriptor** — built on the stack,
used by one call, dropped, with nothing cached, interned, memoized, or
shared between calls. It applies exactly two transformations, both of
which preserve the logical element sequence: **unit axes are dropped**,
and **adjacent axes are merged** when
`stride[outer] == stride[inner] * extent(inner)` holds for *every* operand
at once. Axes are never reordered, split, or transposed; the bound is a
fixed **4 axes**; **this is not a layout compiler**. The builders are
total, pure, allocation-free, and a function of layout metadata alone —
never a pointer value, an alignment, a clock, an environment variable, or
a CPU-feature probe — and a rejection is a **fallback, never an error**.
`core_unary` and `core_binary` are retained **verbatim** as the shipped
generic reference paths.

**`exp` and `log` are deliberately excluded** and keep exactly the paths
they had: IEEE-754 does not specify them, so a toolchain that vectorized
them through a vector-math library would be free to return different bits.
Nothing is lost — measured, the templated traversal is worth **1.05x** on
both, inside this machine's noise.

**The numerical contract is H8's own, in four parts**, measured over every
ordered pair of 14 IEEE-754 representatives x three operations x five
layouts against a pre-H8 library built from identical sources. (1) Every
result in which **at most one operand is a NaN is bit-identical** —
**zero differing results** across all 15 combinations. (2) NaN positions
are identical and every NaN the *arithmetic* produces is quiet
(`relu_backward` and the identity gather **select** an operand rather than
computing, so a signaling NaN legitimately survives them — identically on
both traversals, exactly as H5 established for the copy). (3)
**Subtraction is bit-identical everywhere**, two-NaN pairs included,
because it is not commutative. (4) For **addition and multiplication with
two NaN operands** the surviving payload is outside the contract, asserted
in neither direction. **Part 4 predates H8 and H8 narrows it**: the pre-H8
library's own flat kernel and its own odometer already disagreed on
**30 of 196** such pairs, while post-H8 only a transposed operand differs,
on **5 of 196**. It is a *different* qualification from H2's and H6's,
which concerned NaNs meeting inside an accumulation.

**H1's contract holds unchanged**, proved by poison injected purely by
test infrastructure with two patterns over six layouts, guard elements on
both sides, and a **negative control** showing the detector can fail.

**Track B** shipped the one composed-allocation change the evidence
supported: `_NativeBatchNorm` builds its `(1 - momentum, momentum)` pair
**once per forward** instead of once per buffer — the H4 per-step-constants
shape, never stored on the module — and each blend releases its
temporaries at last use. Against a **retained pre-H8 composition executed
natively**, with running statistics proved bit-identical first, a
`NativeBatchNorm1d` training forward goes **25 -> 23** allocations with
**peak live storages 25 -> 17** and constant fills **5 -> 3**, and
`NativeBatchNorm2d` **30 -> 28**, **30 -> 22**, and **5 -> 3**. **Its
timing effect is neutral** (1.007x-1.106x), and it is reported as a memory
result rather than a speed one. Four alternatives were **rejected with
reasons**, including adopting the blend result into the running-state
transaction, which would have moved numerical work inside the staging
phase and changed a failure ordering F5 and F8 prove by test.

**Measured**, against the pre-H8 library on identical `ctypes` calls with
every case bit-identical before either side was timed, 11 alternating
rounds, identical-code control band **0.97x-1.08x**: row-broadcast
`multiply` **10.58x**, strided same-shape `add` **9.67x**, NCHW-statistic
`multiply` **7.15x**, column broadcast **6.70x**, scalar broadcast
**6.31x**, rank-3 broadcast **6.18x**, transposed `add` 2.63x, strided
`relu` 2.51x, transposed `copy` 2.31x, `sqrt` 2.03x, `reciprocal` 1.98x,
`relu_backward` 1.86x, contiguous `relu` 1.78x, contiguous
`add`/`multiply` 1.76x, contiguous `copy` 1.68x. End to end, over 11
alternating **subprocess** rounds with all 31 cases bit-identical first:
**`NativeAdam.step()` at (128, 128) 2.01x**, **`NativeBatchNorm1d` eval
forward 1.40x**, **`NativeBatchNorm2d` eval forward 1.36x**,
**`NativeBatchNorm2d` training forward 1.33x**, **`NativeLayerNorm`
forward 1.30x**, `NativeBatchNorm2d` forward+backward 1.25x, **the large
MLP training step 1.19x**, and **the normalized training step 1.08x**.
**This is the milestone that finally moved the normalization modules** —
which H6 measured as almost entirely neutral, and which is precisely why
H0's composed-module H7 was dropped and this one entered.

**Reported just as honestly.** Small normalization shapes are neutral
(`NativeBatchNorm1d` training at `(32, 16)` **0.98x**), the CNN step is
neutral (**0.99x**) because its time is in the convolution kernels H8 did
not touch, and the `exp`/`log` controls read 0.97x-1.07x exactly as the
deliberate exclusion predicts. One control is **published rather than
buried**: 256-cubed `matmul` reads **0.93x-0.96x**, and a focused 25-round
run isolates the effect to that one size — 64 cubed 1.014x, 128 cubed
1.035x, **256 cubed 0.921x**, 384 cubed 0.994x — against an identical-code
twin reading 0.969x on the same case. `matmul.cpp` is byte-identical
source compiled with identical flags; `elementwise.cpp`'s object code grew
127 KB to 188 KB, moving every function's placement in the image. This is
the same whole-translation-unit code-layout effect H6 documented, every
matmul result is bit-identical, the H2 CTest passes, and no end-to-end
case regressed.

**Memory: Track A moved none, and the odometer's heap-allocated counter is
now removed on every plannable layout** — a strided elementwise call makes
**one** allocation where it previously made two, which is a strict
reduction and which re-anchored one existing fault-injection test (its
assertion unchanged, its operand changed to a rank-5 reversed view the
builder declines, with a **new** test asserting the other half).

The harness gained **four** cases, 34 to **38**, and the native CTests 15
to **16**. **No exported C ABI symbol, no new translation unit, and no
public control of any kind** — no path selector, plan inspector,
collapse-mode flag, threshold setter, dispatch tracer, profiling counter,
environment variable, or "which path ran" query — and no SIMD, threading,
OpenMP, BLAS, memory pool, scratch workspace, general fusion, fast-math,
or `restrict`. No public API, capability, dtype, device, registry value,
checkpoint field, or checkpoint version moved: `UNSUPPORTED` still reads
`("float32", "cuda", "amp")` and the native checkpoint format is still
version 2 with versions `(1, 2)` supported. Phase G remains the latest
*completed* phase; Phase H remains the current one.

### Phase H — native CPU performance and runtime efficiency (H7)

**Milestone H7 — native Python/C ABI boundary efficiency — is
complete**, and it is **Python-only**: no C++, no exported symbol, no kernel,
no traversal, no arithmetic. The library still exports exactly **52** `tf_*`
symbols.

**The ladder was revised here, and the revision is recorded rather than
retrofitted.** H0's H7 slot was *composed-module cost* — the normalization
modules and the composed convolution bias gradient — explicitly conditional
on a re-measurement after H1, H3, and H6. That condition was tested and
**not met**: H6 made `mean` 3.9x-4.1x faster and moved the normalization
modules almost not at all (`NativeLayerNorm` forward 1.16x,
`NativeBatchNorm2d` backward 1.10x, everything else inside the
0.90x-1.03x control band, the normalized training step 1.03x). So the milestone was **dropped on evidence**, its
proposal and the evidence against it preserved in the design document
rather than deleted, and the slot was refilled from the *same*
measurements: H3, H5, and H6 had each ended by deferring the identical
named cost — H5 "~1.1 us per layout array at the ctypes boundary, left to a
later dispatch milestone", H6 "the fixed ~7 us Python-plus-ctypes cost,
left to a dispatch milestone". Three milestones deferred one thing to a
later dispatch milestone; H7 is that milestone. Composed-module allocation
count remains conditional future scope, and is H8's subject.

The cost was **decomposed rather than assumed**, and the claim that six
kernels were involved was checked and found wrong. All 52 exports are
configured in one file — no other module in the repository imports
`ctypes` — and **57 of their argument positions are arrays**, every one
formerly bound as `numpy.ctypeslib.ndpointer`. That binding re-verifies
array-ness, exact dtype, and contiguity at **every call**, then constructs
`obj.ctypes` and resolves it through `_as_parameter_`: two Python object
constructions and three checks, per array, per call, measured at **~2.1 us
per array position**. On real calls: `tf_core_add` on a 4x4 with three
layout arrays cost **7.6 us**, of which **6.1 us** was the binding; the
array-free `tf_core_add_contiguous` cost 0.9 us and is the control.

Then the *frequency* was counted: an MLP training step makes 245 native
calls carrying **101** array crossings, a normalized step 692 calls and
**315**, a CNN step 242 and **104** — about **20-23 %** of each step's wall
time. And the provenance was the finding that decided the architecture:
**~85 % of those crossings are operation-local broadcast strides**, not the
H3 per-view cache, so a design that only cached pointers per view would
have captured a seventh of the available work.

H7 ships **two bindings for two categories, and deliberately not one
blanket policy**. *Data* positions keep the checked `ndpointer` binding —
the seven public raw-buffer kernels (whose callers may pass anything), the
`copy_from`/`copy_to`/`materialize` host conversion boundary, and the
cross-entropy **class labels**, which are int64 like the layout metadata
but stay checked because a label array's required length comes from the
*logits*, a different object. *Layout metadata* positions — 32 of them
across 13 exports — take `ctypes.POINTER(ctypes.c_int64)`, fed by exactly
two private producers: `NativeTensorView._native_layout_pointers()`, which
memoizes `data_as` over the **unchanged** H3 read-only NumPy arrays that
remain the owning buffers, and `_layout_vector(values)`, which builds a
fresh `(c_int64 * len(values))` for metadata belonging to one operation.

**Nothing was weakened, and one thing was strengthened.** ctypes still
type-checks every call: a trusted position rejects a NumPy array of any
dtype, a differently typed pointer or vector, a `c_void_p`, a list, an
int, bytes, and a string — a NumPy array being rejected is a deliberate
consequence that makes the old binding unreachable by accident. Dtype,
byte order, and contiguity are established *by construction* rather than
re-checked. **The length/rank invariant — the one `ndpointer` never
checked, because the ABI sees only a pointer and an `ndim`** — is now
checkable for the first time: a vector carries its length in its type, and
a cached pointer carries its owning array (NumPy's `data_as` attaches it),
whose length is the view's rank. The suite asserts that per producer, per
rank 0-4, and structurally over **every strided call in a real workload**.
The one honest difference is `None`, which `ndpointer` rejected and a typed
pointer converts to NULL; it is closed structurally — both producers are
total and no public API takes a metadata pointer. (Only **three** of the
thirteen exports reject a null metadata pointer in C++ as well, so the
producers being total is the load-bearing part of the argument, not a
belt-and-braces remark.)

**Ownership is NumPy's guarantee, relied on and tested rather than
assumed.** `data_as` stores the array on the pointer, so a cached pointer
cannot outlive its buffer; `POINTER.from_address` was measured **faster**
(0.9 us against 1.6 us) and **rejected outright** because it produces a
pointer with no owner. Deriving the pointer from the array rather than
building a second vector keeps exactly **one owning description** of a
view's layout — a cached ctypes vector was measured fastest of all and
rejected, because it would duplicate that description and lose H3's
`writeable = False` protection for ~2 % of a training step. Proved with the
cyclic collector **disabled**: no reference cycle (an explicit
`gc.collect()` after dropping a view collects 0 objects), no native storage
kept alive, no pointer surviving into a usable state after close, and
operation-local vectors retained by nothing. There is no global pointer
cache, no id-keyed table, and no `from_address`, `byref`, `addressof`,
`id`, or weak-reference container **anywhere in the module's code** —
enforced by parsing it, so the docstring recording *why* `from_address` is
unused is not mistaken for using it. Binding configuration stays a
load-time act: `argtypes`/`restype`/`errcheck` are assigned only inside the
two loader functions, asserted by locating each assignment's enclosing
function, so nothing reconfigures a shared function object per call and
**no thread-safety claim is broadened**.

Measured against a **retained pre-H7 `cpp.py` driving the same Release
DLL**, over 11 alternating pre/post subprocess rounds, every case proved
**bit-identical before either side was timed**; control bands 0.95x-1.10x
(core) and 0.99x-1.05x (end to end). Core: a 1-element `sum` **1.94x**, a
16-element `sum` 1.89x, `to_numpy` 16x16 **1.83x**, `sum(axis=0)` 16x16
1.79x, a 4x4 `contiguous_copy` 1.73x, `narrow_backward` 1.73x, strided
`relu_backward` 1.71x, scalar-broadcast `add` **1.67x**, 4-D NCHW
`sum(axis=1)` 1.54x, row-broadcast `add` 1.41x, `sum(axis=0)` 256x256
1.35x, strided `exp` 1.29x, transposed materialization 1.16x. End to end —
and this is the result — the **native Dropout step 1.32x, the normalized
step 1.31x, `NativeAdam` at (32, 32) 1.31x, the CNN step 1.30x, the MLP step
1.28x**, `NativeLayerNorm` forward 1.23x, `NativeBatchNorm1d` eval
1.23x, `NativeAdam` at (128, 128) 1.14x, `NativeSGD` 1.13x, the large MLP step 1.08x. **H7 is the
first Phase-H milestone to move every training step** — H4 moved them
1.09x-1.23x and H5 and H6 were neutral on all of them — because the cost
is paid per *call* and a step makes hundreds of them.

Reported just as honestly: **large kernel-bound work is neutral**, exactly
as the attribution predicts. 256-cubed matmul **0.99x** and 8-cubed matmul
1.00x are controls that take no array at all, so **H2's result is
structurally untouched**; contiguous 16x16 `add` 1.05x is the third
array-free control; and 512x512 `copy` 1.02x, 256x256 `to_numpy` 1.04x,
512x512 full `sum` 1.06x, 256x256 broadcast `multiply` 1.08x, and the large
MLP step 1.08x are all at or inside the band. **H7 did not make matmul faster**,
and no reading should say otherwise.

A second, independent 11-round run reproduced every row: all cases again
bit-identical, every control again holding (256-cubed matmul 0.98x, 8-cubed
matmul 1.04x, contiguous 16x16 `add` 1.08x, 512x512 `copy` 1.01x), and every
training step again improving, with individual ratios moving by roughly the
control band's width in both directions (a 1-element `sum` 2.12x against
1.94x, `NativeSGD` 1.26x against 1.13x, `NativeAdam` at (128, 128) 1.08x
against 1.14x). The figures quoted are the first run's; the second is
recorded in the design so no single number is read as more precise than the
method supports.

**Memory did not move, and that is asserted**: the same boundary workload
allocates 5 native storages, peak 4 live, 584 peak bytes before and after —
identical. A view's cold footprint is byte-identical; a view that actually
takes a strided path costs **+296 bytes** for the pointer pair, and only
**9 of 98** views in an MLP step ever populate it, which is H3's laziness
argument unchanged. The harness gained three cases, 31 to **34**:
`ctypes_boundary_strided` (the array-carrying twin of the existing
array-free `ctypes_boundary`, so the two crossings are separated rather
than averaged — measured **0.8 us** array-free against **1.3 us** with
three layout arguments, where pre-H7 it would have been ~7 us), plus
`elementwise_broadcast_scalar` and `elementwise_broadcast_row`, the two
broadcast shapes the optimizer and the normalization modules actually use.
Validation added a **sanitizer negative control**: under Clang ASan,
test-only code handing `tf_core_sum` two-entry metadata with `ndim = 3`
produces a `heap-buffer-overflow`, `READ of size 8`, `0 bytes after 16-byte
region`, in `reduce_prefers_contiguous_blocks` — the exact H3 finding —
which is what makes the **zero diagnostics across 2,834 sanitized tests** a
real absence rather than a blind detector. No public API, capability,
dtype, device, registry value, checkpoint field, or checkpoint version
moved, and no C ABI symbol was added.

Phase G remains the latest *completed* phase; Phase H remains the
current one.

### Phase H — native CPU performance and runtime efficiency (H6)

**Milestone H6 — native reduction execution efficiency — is complete**,
the third Phase-H milestone to change C++ and, like H2 and H5, **not the
ABI**: the library still exports exactly **52** `tf_*` symbols. Reductions
were the last core family in the runtime that always paid the generic
strided indexing cost.

The pre-H6 kernel was re-read from source and re-measured on the post-H5
build rather than trusted from H0's or H5's summaries, and its cost was
**decomposed** rather than assumed. At `(256, 256)` `axis=0` a `core.sum`
costs 99.7 us, of which the **raw native call is 94.8 us — 95 %**; the
three `ndpointer` conversions inside that account for ~3.2 us, leaving the
C++ traversal itself at ~91.6 us, **92 %** of the operation. The whole
Python wrapper — axis normalization, output-shape construction,
write-stride construction, the H3-cached layout arrays, and the output
allocation — is about 5 us. This is the **opposite** of B3: H3's subject
was a fixed Python cost that dominated *small* operations, while a
reduction of any real size is dominated by the compiled loop.

H6 reused the dispatch shape H2 and H5 each proved — one hidden metadata
predicate, inside the existing export, no new symbol, the pre-milestone
traversal retained. New `cpp/include/tf_reduction_internal.h` declares
three hidden-visibility `namespace tf` functions and `cpp/src/reduction.cpp`
implements them: **`tf::sum_generic_strided`**, the pre-H6 odometer retained
as the shipped generic reference path — the only path that can address a
transposed, narrowed, non-unit-strided, or broadcast source at all, and the
oracle every optimized result is compared against;
**`tf::reduce_prefers_contiguous_blocks`**, the predicate — total, pure,
allocation-free, a function of layout metadata alone, never of a pointer
value, an alignment, a clock, an environment variable, or a CPU-feature
probe, with a false answer a fallback and never an error; and
**`tf::sum_contiguous_blocks`**, a flat walk over an `outer x mid x inner`
factorization. The predicate accepts a reduction when the source strides
are exactly the row-major strides implied by the shape (the same definition
`NativeTensorView` uses, so the two layers agree by construction), the
reduced axes — those with a zero *write* stride — form **one contiguous
run**, and the kept axes carry exactly the row-major strides of the output
formed by dropping that run. Stride collapsing is **implicit and bounded
rather than a general layout compiler**, nothing is cached or interned, and
`keepdims` needs no special case because the kernel cannot even observe it.

**Per-output accumulation order is preserved exactly**, and the source
traversal order is not even reordered: the loop nest is the lexicographic
order of the source's own row-major index, which is what the odometer
walks, and every destination cell is touched by exactly one index pair, so
the cells are independent. Nothing is reassociated, and no FMA, Kahan,
pairwise, tree, parallel, or horizontal-vector reduction exists. The
`inner == 1` branch's local accumulator is **seeded from the destination**,
which keeps the export's accumulate-into semantics identical on both paths.

**Signed zeros are proved as raw bit patterns** across all-positive,
all-negative, alternating, first, last, mixed-with-finite, and cancelling
cases at every axis and both `keepdims` values: the sum of any number of
`-0.0` values is `+0.0` on both paths and matches NumPy, and a `-0.0` seed
stays `-0.0`. The rank-0 export branch is recorded precisely rather than
idealized — it is a genuine addition against a zeroed destination, so a
rank-0 `-0.0` sums to `+0.0`, exactly as before H6.

**The NaN rule is H6's own and was measured, not inherited from H2.**
Contractual: identical NaN positions on both paths; every NaN quiet, with a
signaling-NaN input quieted by both to identical bits; and bit identity
whenever **at most one NaN** enters an accumulation, which is every case
that occurs in practice. Not contractual, and asserted in neither
direction: with **two or more** NaNs accumulated into one cell the paths may
select different payload bits. Four spellings of the optimized accumulation
were compared — including one accumulating *through memory* exactly as the
odometer does — and **all four selected the same NaN and all four differed
from the odometer**, so the accumulator is not the cause and parity is
unavailable at any spelling; the memory form was also 1.2x-1.8x slower.
Recorded as observation rather than promise: the block path keeps the
**first** NaN and the odometer the **last**, and the block path's choice is
NumPy's — so H6 moved the answer toward NumPy. H5's copy rule does not
apply either, for the reason that made it strong: a value transfer performs
no arithmetic, and a reduction is arithmetic.

**H1's rejection of this destination stands and is confirmed rather than
revisited**: both traversals read the destination, so it stays
zero-initialized. Outcome B was rejected on measurement (2,048 bytes of
fill against 524,288 bytes of reads, and 8 bytes at `axis=None`) and on
semantics (a fast path that assigned its first contribution would behave
differently from the reference for a non-zero destination). H6 therefore
adds no poison test, because it introduces no uninitialized destination.

Measured against a **pre-H6 library** built from identical sources with only
`reduction.cpp` restored, driven through identical `ctypes` calls on
identical data, outputs proved **bit-identical before either side was
timed**, 15 alternating rounds, control band **0.90x-1.03x**: full
reductions 1.19x at 1,024 elements to **3.96x** at `(512, 512)`; 2-D axis
reductions 3.24x to **6.37x**; and — the finding that was not predicted —
3-D and 4-D reductions **8.60x-10.94x**, because the odometer's carry loop
runs up to `ndim` iterations per element while the block traversal's cost
does not grow with rank. At the layer level `TensorCore.sum(axis=0)` 4.49x,
`mean(axis=0)` 4.11x, NCHW `sum(axis=1)` **8.56x**, `NativeTensor.sum`
3.88x, `sum()` forward+backward 1.27x, the **convolution bias gradient's
three chained sums 1.46x**, `_unbroadcast` 1.15x, softmax backward 1.14x,
`NativeLayerNorm` forward 1.16x. Against NumPy the contiguous reduction gap
closed from roughly 8-13x to **1.67x-3.75x**, while the transposed-view
control stayed at 10.33x.

Reported just as honestly: **every training step is neutral** (0.99x-1.03x,
inside the control band), so **H6 does not make training faster**;
**normalization is mostly neutral**, which narrows H7 rather than
motivating it; **tiny reductions are neutral**, because below roughly 1,000
elements the fixed ~7 us Python-plus-ctypes cost dominates; and one **real,
repeatable ~10 % regression** on 2-D transposed `axis=0` fallbacks
(0.89x-0.93x across four 25-round runs) is published rather than buried,
with the 3-D transposed fallback measuring 1.04x-1.05x *faster* and the
cause isolated to whole-translation-unit code layout — an isolated binary
showed the extracted call is not it — which the design rejects chasing. A
register-blocked small-trailing-extent path was rejected on complexity.

**Memory moved not at all, and that is asserted**: a `sum` allocates
exactly one native storage on both paths at every axis, `mean` the same
one, and a 10-step training run produced a bit-identical allocation profile
before and after. The harness gained three cases, 28 to **31**
(`reduction_last_axis`, `reduction_full_to_scalar`,
`reduction_middle_axis_4d`), with `reduction_transposed_view` now
explicitly the control; one dependency-free CTest was added,
`cpp/tests/test_sum_reduction.cpp`, taking the native suite from 14 to
**15**. No exported C ABI symbol, no new translation unit, no public
control of any kind, no SIMD, threading, OpenMP, BLAS, parallel reduction,
memory pool, scratch workspace, or fast-math; multi-axis reduction was not
added, and `tf_core_narrow_backward` — the scatter dual — was deliberately
left alone. `UNSUPPORTED` still reads `("float32", "cuda", "amp")`,
`SUPPORTED_DTYPES` still reads `("float64",)`, `SUPPORTED_DEVICES` still
reads `("cpu",)`, and the native checkpoint format is still version 2 with
versions 1 and 2 supported. Phase G remains the latest *completed* phase;
Phase H remains the current one.

### Phase H — native CPU performance and runtime efficiency (H5)

**Milestone H5 — native copy and mutation-transfer efficiency — is
complete**, and it is the first Phase-H milestone since H2 to change
C++ — though **not the ABI**: the library still exports exactly **52**
`tf_*` symbols. H5 replaced the native line's **value-transfer
primitive**. `_native_copy` was `zeros(shape) + core` — two allocations,
a full zero-fill pass, and a full elementwise-addition pass — and is now
the E3.1 native identity gather, `NativeTensorCore.contiguous_copy()`:
one uninitialized allocation (H1) and one pass. The composition predates
that gather and was simply never migrated to it. A complete inventory
found **ten** call sites of the one helper — `NativeParameter.copy_value_`
staging, both `state_dict()` snapshot paths, both `load_state_dict()`
staging paths, both BatchNorm running-statistic commits, and the
reshape/transpose/unbroadcast gradient materializations — and every one
of them is a **pure value transfer**: an independent contiguous
materialization of some tensor's current value, wanting no arithmetic.
All ten were enabled. `_broadcast_back`'s `zeros(x_shape) + upstream` was
**rejected** because it is not a copy at all but a genuine broadcast
expansion, which `contiguous_copy` cannot express; `sum`/`mean` and
`narrow_backward` keep their zeroed destinations for H1's unchanged
reasons.

The semantic question H4 refused to decide in passing was decided here,
by measurement over a fixed 18-pattern IEEE-754 sweep. **Exactly three**
patterns behaved differently under the two spellings: the addition
normalized `-0.0` to `+0.0` and quieted both signs of signaling NaN,
while the gather preserves all three. Everything else — `±0`, `±inf`,
quiet NaNs of either sign and **any payload**, denormals, the smallest
normal, the largest finite magnitudes — was already identical, so no NaN
payload differed at all (with one NaN operand and one zero, x86-64's
`ADDSD` returns that operand's NaN). **H2's matmul NaN-payload carve-out
does not generalize to copies**: it exists because two NaN operands meet
in an accumulation, and a copy performs no arithmetic. The pre-H5
behavior was **accidental and inconsistent**, not contracted — three
other value-copy paths (`NativeParameter(source)` construction,
`detach()`, and the `to_numpy()`/`from_array` boundary) always used the
gather and always preserved `-0.0`, while `copy_value_` documented the
same thing and did not deliver it. H5 states the narrowest coherent rule:
**a value transfer reproduces its source's bits exactly; an operation —
`zeros + x` included — follows IEEE arithmetic.** No operation's
arithmetic changed anywhere, and the whole pre-H5 suite passes unchanged
apart from the guardrails that pinned the old composition by name.

Swapping the composition alone would have **regressed** the common case,
so H5's one C++ change is a second **traversal** inside the unchanged
`tf_core_contiguous_copy` export. `zeros.add(core)` on a contiguous
source takes a flat pointer loop, while the gather always walked the
generic odometer — the only unary export without the contiguous fast path
every other one has — and a naive swap measured **0.48x** at 16,384
elements. The export now picks its traversal from the layout metadata it
already receives, exactly as H2's matmul picks its kernel:
`tf::copy_prefers_contiguous` is hidden-visibility C++ in a new internal
header, total, pure, allocation-free, and a function of metadata alone —
never of a pointer value, an alignment, a clock, an environment variable,
or a CPU-feature probe — testing exact equality against the row-major
strides implied by the shape, which is the same definition
`NativeTensorView` uses, so the two layers agree by construction. A false
answer falls back to the retained odometer and is never an error. **No
numerical carve-out is needed, and that is the difference from H2**: both
traversals evaluate `dst[out] = src[pos]` over the same logical elements
in the same destination order and differ only in how `pos` is computed,
so they are bit-identical *by construction* — proved directly at the C++
level by a new dependency-free CTest, taking the suite from 13 to 14.
There is no copy-mode selector, overlap-mode flag, traversal tracer, or
public dispatch control of any kind.

Nothing became less safe, because nothing became in-place: every call
site still **stages** an independent materialization and only then adopts
it. The overlapping arrangements the runtime can construct —
`copy_value_(self)`, a source that is a view of the destination's own
storage, a square parameter's own transpose, sibling views, duplicate
parameters across optimizers — are each tested and each correct, and no
`memcpy` is used anywhere. Parameter identity, storage replacement,
gradient retention by identity and value, the one version increment per
commit, the F1 state transaction, checkpoint atomicity, and exact resume
are all exactly what they were; gradient *accumulation* still adds rather
than assigns. H1's full-write contract is proved on both traversals by
poison injected purely by test infrastructure around the allocator, with
a negative control showing the detector can fail.

Measured by alternating pre/post **subprocess** rounds against a retained
pre-H5 composition, with a control band of **0.96x-1.05x**, and — for
the C++ half — by building a **pre-H5 library** and driving both through
identical `ctypes` calls on identical data, outputs proved bit-identical
before either was timed. The traversal alone: **2.5x-5.5x** on contiguous
sources from 16 K elements up (5.53x at 512 squared, 5.53x on 4-D NCHW,
5.46x on an offset view), 1.29-1.62x on small ones, and **0.94x-1.02x on
transposed and last-axis-narrowed sources**, which take the *unchanged*
odometer and are the design's own control. End to end: `copy_value_`
**2.14x** at (512, 512) and 1.26x at (128, 128), optimizer `state_dict()`
2.40x and `load_state_dict()` 1.69x, module `load_state_dict()` 1.37x,
`NativeSGD.step()` 1.15-1.31x. Reported just as honestly:
**`NativeAdam.step()` is neutral** (0.98x-1.06x — the commit copy is one
of about seventeen buffers and the arithmetic dominates), **every
training step is neutral** (0.95x-1.07x), the **BatchNorm running update
is neutral** (0.98x), and **copies below ~16 K elements are neutral**
(0.93x-1.01x), because a `contiguous_copy` call converts two `int64`
layout arrays at the ctypes boundary at **~1.1 us each** — a cost
measured, attributed, and left to a later dispatch milestone rather than
paid for by weakening H3's validation. Two methodology findings are
published rather than buried: at 7 alternating rounds the small copies
read 0.78x-0.94x and looked like a regression, while at 21 rounds the
same cases read 0.93x-1.01x (the same lesson H3 recorded); and the
largest single ratio, **7.9x-10.5x at 512-640 KB**, is a **512 KB
allocator cliff on this machine**, not a loop-speed result — the pre-H5
composition makes two large allocations and zero-fills one, so it crosses
that threshold at half the size and pays it twice. The durable statements
are ~2.1x at 1-2 MB and neutrality below 384 KB.

Memory moved with time, never against it: **no measured peak rose**, and
the pure-transfer paths halved. `copy_value_` at (512, 512) went 2
allocations to **1** and 4,194,304 to **2,097,152** peak bytes; module
`state_dict()` and `load_state_dict()` 4 to **2** allocations with peak
bytes halved; optimizer `state_dict()` 16 to **8**; `NativeSGD.step()`
5 to **4** with peak 393,216 to **262,152**; and `NativeAdam.step()` went
**17 to 16** allocations per parameter (H4 took it 27 to 17), removing a
whole-parameter zero-fill pass from every committed update. The harness
gained two cases, 26 to **28**: `row_major_materialization`, the
flat-traversal twin of the existing transposed-source case, so the two
traversals are separated rather than averaged; and
`parameter_value_commit`, `native_only` with **no ratio**, because the
stable line mutates a `Parameter` by rebinding `.data`, which is a
different operation. The ladder was **reordered** here — reduction
execution, drafted as H5, moved to H6 — and no public API, capability,
dtype, device, registry value, checkpoint field, or checkpoint version
moved.

`UNSUPPORTED` still reads `("float32", "cuda", "amp")`,
`SUPPORTED_DTYPES` still reads `("float64",)`, `SUPPORTED_DEVICES` still
reads `("cpu",)`, and the native checkpoint format is still version 2
with versions 1 and 2 supported. Phase G remains the latest *completed*
phase; Phase H remains the current one.

### Phase H — native CPU performance and runtime efficiency (H4)

**Milestone H4 — native optimizer step efficiency — is complete.** Like
H3 it is **Python-only**: no C++, no C ABI symbol, no ctypes
declaration, and no kernel changed, so the library still exports exactly
**52** `tf_*` symbols. It is the first Phase-H milestone whose subject is
a *training-stack* component rather than the tensor runtime.

**The bottleneck was re-measured, not assumed.** B4's counts were
re-instrumented on the current post-H3 code by wrapping
`NativeStorage.__init__` — the one constructor every allocation path runs
through, zeroed and uninitialized alike — plus `NativeTensorCore.full`,
`_binary_core_op`, and `_unary_compute`, all test-locally. H0's figure
was confirmed exactly: **27 native storage allocations per parameter per
`NativeAdam.step()`**, fully attributed as 8 scalar coefficients, 13
binary compute outputs, 4 unary compute outputs, and 2 for the commit
copy. **Ten of the 27 are one-element**: the eight broadcast scalars
(`beta1`, `1 - beta1`, `beta2`, `1 - beta2`, both bias-correction terms,
`eps`, and `lr` — design §3.2 said six, and `eps` and `lr` were the two
it missed) plus the two `reciprocal` outputs taken on one-element
tensors. `NativeSGD` allocates five per parameter. Eight of Adam's
thirteen binary operations take the broadcasting path rather than the
contiguous fast path.

**Three changes shipped.** *The step's scalar coefficients are built once
per step, not once per parameter.* Six of them are the same value for
every parameter in a step, so a private per-step `_StepConstants` holder
builds each on first use — keyed by `(dtype, device)`, so it never
assumes a single dtype exists — and hands the same read-only core to
every later parameter; the two bias-correction terms are cached per step
*counter*, so steady-state training builds one pair while a parameter
that skipped earlier steps legitimately gets its own. The holder is
created inside `step()`, allocates nothing until the first entry asks for
a coefficient (a step with no active parameter allocates nothing at all),
and is released before the commit begins. It is **never stored on the
optimizer**, so no scalar survives a step, enters `state_dict()`, reaches
a checkpoint, or has to be released by `close()`. `NativeSGD` does the
same for its single `lr` scalar — the only change its evidence supported.
*The bias-correction reciprocal is evaluated in Python*, removing one
allocation and one kernel call per coefficient per parameter. That is an
**exact substitution, not a reassociation**: the kernel literally is
`double op_reciprocal(double x) { return 1.0 / x; }`, a Python `float`
and a C++ `double` are the same IEEE-754 binary64 value, and IEEE-754
requires division to be correctly rounded, so exactly one result is
possible — proved over **20,000+ values** spanning the full exponent
range, ±0, ±∞, the smallest subnormal, the largest finite magnitude, and
every `1 - beta ** t` the optimizer actually forms, compared on **raw
`uint64` bit patterns** with zero mismatches. *Temporaries are released
at their last use* rather than all together at the end of the staged
expression.

**Bit-identical, with no carve-out of the kind H2 needed** — no
accumulation order, operand position, or kernel changed, so NaN payloads
match too. The **pre-H4 composition is retained in the test suite** as a
literal transcription executed natively, so every equality is against
real native execution of the old code rather than a NumPy re-derivation:
60 shape/step/hyperparameter combinations for Adam (including
`beta = 0`, betas at `0.99999`/`0.9999999` with `eps = 1e-30`, and
`lr = 1e10`), a six-step run over four mixed shapes, and four SGD
learning rates from `1e-9` to `1e12`. A separate test pins the **exact
operation sequence** a staged entry issues, so a future reorder or fusion
fails loudly rather than silently.

**The two-phase contract is untouched.** Validation is still four
complete passes in the same order — optimizer open, every parameter open,
every m/v state valid, every active gradient valid — with nothing moved
behind a mutation; stage mutates no parameter, moment, counter, version,
or gradient; the commit is still **one `copy_value_` and exactly one
version increment per updated parameter**; gradients are read and never
written, by identity, value, and storage identity; and the documented
per-entry commit boundary is *tested* by injecting a `copy_value_`
failure rather than assumed infallible.

**Measured** by a controlled A/B alternating `pre` and `post`
**subprocess** rounds so system drift affects both arms equally, 366
samples per case, correctness gated before timing: `NativeAdam.step()`
**1.58×** on one (128, 128) parameter, **1.54×** at (256, 256),
**1.48×** on a four-parameter MLP whose largest weight is 256², 1.21–
1.22× on a small MLP, 1.15× on a first step, 1.09–1.12× on tiny
parameters; a large MLP training step 1.23×, a small one 1.15×, a
normalized step 1.13×, a CNN step 1.09×. The shipped harness agrees on
its own cases: `adam_step` 1.25×, and the gap against
`tensorforge.optim.Adam` narrowed from **23.8× to 19.7×**.

**Reported just as honestly.** A **(512, 512) parameter is neutral**
(1.02×) because at that size the step is memory-bandwidth-bound and ten
fewer one-element allocations are invisible; the **Dropout training step
is neutral** (0.99×); and **NativeSGD is neutral-to-slightly-positive**
(1.03–1.07×), with one 0.88× row identified as **noise** by a focused
re-measurement whose post minima were lower in every pair. The noise
floor is stated rather than assumed: the matmul control case, whose code
H4 did not touch, varied **0.84×–1.26×** between arms, so no single
reading inside that band is a result — which is why the Adam figures come
from the 366-sample alternating A/B and not from one harness run. H2's
large-matmul performance is intact.

**Memory moved with time, not against it.** Peak live transient bytes
during one Adam step fell **2.6–3.0×** — 1,966,160 → 655,424 for a
(128, 128) parameter, 31,457,360 → 10,485,824 at (512, 512), 7,864,400 →
3,022,336 for a four-parameter MLP — and per-parameter allocations went
27 → **17** with at most **eight** shared scalars for the whole step, so
a four-parameter model allocates **76 instead of 108**.

**Six alternatives were measured and rejected**, each with its reason
recorded: scalar materialization (faster below ~32 K elements, *slower*
above, tracking this machine's L2 cache rather than layout metadata, and
it would regress the harness's own profile configuration while adding a
parameter-sized buffer per scalar operation); same-shape stride-0 views
(identical kernel arguments by construction, but *four* NumPy layout
arrays per call where the broadcast path builds three); adopting the
staged core instead of `copy_value_` (the project's one sanctioned
mutation primitive); giving `_native_copy` a `contiguous_copy`
implementation (it would stop normalizing `-0.0` to `+0.0`, a real
observable change in a helper shared far beyond the optimizer); a
persistent per-optimizer scalar cache (the hidden scratch tensor whose
lifetime the design forbids); and reassociating the update to fold
scalars together (a floating-point order change that would break every
exact-resume proof in the project).

All instrumentation was test-local or benchmark-local. **No production
counter, environment-variable profiler, or installed tracing mode
exists**, and H4 added **no public API of any kind**: no cache control,
statistic, reset, profiling counter, dispatch selector, or failure
toggle. Three pre-existing tests that injected a failure at the *N*-th
`NativeTensorCore.full` call were re-anchored to a per-parameter
allocation seam, and a fourth now forwards `*rest` through the staging
signature; **every assertion in all four is unchanged**, and new H4 tests
cover the `full` seam directly at every position the shared holder builds.

`UNSUPPORTED` still reads `("float32", "cuda", "amp")`,
`SUPPORTED_DTYPES` still reads `("float64",)`, `SUPPORTED_DEVICES` still
reads `("cpu",)`, and the native checkpoint format is still version 2
with versions 1 and 2 supported. Phase G remains the latest *completed*
phase; Phase H remains the current one.

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
