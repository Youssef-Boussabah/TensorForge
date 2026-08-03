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

### Phase J — deterministic native data pipeline and mini-batching (J0–J5 complete, in progress)

**Phase J is the latest phase, it is newly approved, and milestones J0
through J5 have landed.** J6 through J9 have not started, and **J6 is
next**. **No version is claimed** — the native line stays experimental and
is not production-ready, and this entry records milestones rather than a
release.

Phase J was approved **after** Phase I closed at I11. The repository
deliberately finished Phase I without committing to a successor, so Phase J
is not carried-over roadmap work and must not be described as though it
were.

**J0 was architecture, contract, and documentation work only, and it
shipped no runtime behavior at all.** No dataset, sampler, or loader class;
no helper module; no state serializer, shuffle helper, or batching helper;
no public export; no production import; no C++; no CMake registration; no C
ABI symbol; no example; no benchmark; no checkpoint field or version; and
no optimizer-state version. Runtime capability began at **J1**.

**J1 shipped the host-backed dataset foundation** —
`src/tensorforge/experimental/native_dataset.py` and its one public class,
`NativeTensorDataset`, exported from `tensorforge.experimental`. That
export inventory went from 22 names to **23**, which is the only public
surface J1 moved. The class takes **two unconditional copied host
snapshots** at construction — `numpy.array(..., copy=True)` rather than
`ascontiguousarray`, precisely because the latter returns an
already-contiguous input unchanged and would alias caller memory in the
common case — at an **explicitly chosen** native feature dtype that is
never inferred from the input array and still defaults to float64. It
computes the locked **SHA-256** fingerprint eagerly over the canonical
little-endian byte stream, exposes the four-field JSON-compatible
`identity()` that the existing checkpoint metadata validator accepts
unchanged, and turns any index sequence into a caller-owned `NativeTensor`
feature batch through the public `from_array` boundary beside a fresh
read-only host `int64` target batch — order and duplicates preserved
exactly, an empty request refused, and **no native storage held between
calls**. Validation runs in the contracted order with nothing allocated
until every check has passed, and a construction failure at either
snapshot or at the digest releases what it allocated before the exception
leaves the constructor. `tests/test_native_dataset.py` covers all of it,
including the fingerprint against **independently computed known-answer
vectors** written with `struct` rather than by calling the implementation
twice. J1 added no C++, no CMake entry, no C ABI symbol, no example, no
benchmark, and no dependency, so **no native rebuild, CTest run, or
sanitizer run was required and none is claimed**.

**J2 shipped the deterministic sampler** —
`src/tensorforge/experimental/_native_permutation.py`, the permanently
private derivation, and
`src/tensorforge/experimental/native_sampler.py` with its one public
class, `NativeBatchSampler`, exported from `tensorforge.experimental`.
That export inventory went from 23 names to **24**, which is the only
public surface J2 moved. The sampler owns `batch_size`, `drop_last`,
`shuffle`, the `seed`, the `epoch`, and the `cursor`, and emits complete
batch-index groups through `epoch_permutation()`, `plan()`, and
`next_batch_indices()`.

The derivation **reuses the locked `tensorforge.splitmix64` algorithm**
rather than introducing a second one: the same finalizer, the same golden
constant, the same shifts and multiplication order, under one
domain-separated epoch key schedule (`SAMPLER_DOMAIN`, the ASCII bytes
`TF_SAMPL`), with unbiased **rejection-based** bounded integers and a
**downward** Fisher–Yates sweep. It is written in explicit
`& (2**64 - 1)` Python integer arithmetic — the module imports **nothing
at all** — so the result is bit-identical on every platform, word size,
and Python build by construction. Every committed reference vector of the
contract's §8.9 is reproduced exactly: the four `splitmix64_mix` known
answers, all twelve `epoch_key` vectors, all thirty-two permutations at
lengths 1, 2, 5, and 8 across four seeds and two epochs, the sequential
orders, and all ten batch plans — written on the test side as literals
rather than generated by calling the implementation. The equality between
the Python and C++ implementations is a **gate rather than an
assumption**: `tests/test_native_sampler.py` predicts the shipped
`tf_core_dropout_forward` kernel's keep/drop pattern from the Python
derivation at 48 `(seed, call_index, p)` combinations of 4,096 elements
each, with a non-vacuity control proving that a mutated multiplier,
golden constant, shift, or key breaks the prediction — and a companion
assertion keeping the sampler's own key schedule provably **distinct**
from Dropout's, so the proof cannot be misread as "the two streams are
the same". The rejection branch, which no reference case reaches, is
forced directly at a bound whose limit makes half of all draws fall out.

Because a permutation is a **pure function** of `(seed, epoch, length)`,
the sampler holds no consumable stream: inspection and planning consume
nothing, may be repeated in any order, and leave the state byte-identical.
Its `state_dict()` is a fresh JSON-compatible structure carrying the
configuration, the position, and the dataset's four identity fields —
**no permutation and no payload**, because eight bytes of seed already
carry the order exactly — and passes the existing checkpoint metadata
validator unchanged. `load_state_dict()` validates everything in the
contracted order, checking dataset identity against **live** reality
(structural fields first, digest last) while **adopting** the state's
configuration, and then commits through six assignments that cannot fail;
a rejected load leaves every observable property, the position, and the
cached behavior exactly as they were. The sampler allocates nothing
native, materializes no batch, constructs no `NativeTensor`, owns nothing
releasable — so it has **no `close()`**, on `NativeGenerator`'s precedent
— and works unchanged against a closed dataset. J2 added no C++, no CMake
entry, no C ABI symbol, no example, no benchmark, and no dependency, so
**no native rebuild, CTest run, or sanitizer run was required and none is
claimed**.

**J3 shipped the native mini-batch loader** —
`src/tensorforge/experimental/native_data_loader.py` and its one public
class, `NativeDataLoader`, exported from `tensorforge.experimental`. That
export inventory went from 24 names to **25**, which is the only public
surface J3 moved, and it is the **last** of the three names J0 locked.
The module imports **exactly one name**, `NativeBatchSampler`, so it
reaches no ctypes layer, no NumPy, no checkpoint, and no generator, and
it constructs no lock, thread, queue, or worker.

`NativeDataLoader(sampler)` takes a sampler and nothing else: no
batch-size, shuffle, seed, drop-last, dtype, or device argument, because
the sampler and the dataset already own those and one fact needs one
owner. The loader is **not itself an iterator**; `iter(loader)` returns a
fresh private `_NativeBatchIterator` for **one epoch**, which captures
`sampler.remaining` at construction and counts it down — captured rather
than re-read, because the sampler's `remaining` resets to a whole epoch
the moment the canonical boundary is crossed. A new `iter()` **supersedes**
any previous iterator, so `for ... break` followed by another `for`
continues cleanly from the committed position while nested iteration
fails loudly rather than interleaving two traversals over one cursor.

**The milestone's whole subject is the batch handoff, and it is an
explicit transaction.** Every `__next__` runs five phases — claim,
construct, publish, commit-and-deliver, and rollback — under one
invariant: **the committed sampler position advances if and only if a
batch was successfully delivered to the caller.** The record is
deliberately **split by ownership**: the sampler keeps the integer half
(a never-reused serial, the owning iterator's token, the exact
pre-delivery and candidate post-delivery positions, and the indices),
which is why it still owns nothing releasable and still has no
`close()`; the iterator keeps the resource half, because an owned
resource belongs on the object whose `close()` releases it. Commit
happens **before** delivery, not after: if the position advanced after
the handoff, a failure in between would hand the caller a batch the
loader still considered unconsumed and the next call would deliver it
twice. Committing first makes the only recoverable state "not yet
delivered", and it is fully recoverable — the rollback restores the exact
pre-delivery position through the **same non-failing write seam** a state
load commits with, so a rollback that could itself raise is structurally
impossible.

Every failure position is proved by injection rather than argued, each
with its own **non-vacuity control** and an exact before/after comparison
of the position, the whole `state_dict()`, the captured countdown, and
the native live-storage baseline: a claim failure before publication
(which mints no serial), a feature-materialization failure, a **real
native allocation failure** through the shipped `tf_test_arm_alloc_failure`
hook, a target-gather failure after the feature tensor exists, a
publication failure, a commit failure injected into the structurally
non-failing seam, and the **delivery-seam failure at both dtypes** —
after which the very next `__next__` returns the **same indices and the
same values** in a freshly allocated tensor. Reentrant `iterator.close()`
and `loader.close()` are driven from *inside* the seam, so a reentrant
arrival is real rather than simulated, and the exact-match completion
check refuses to hand back a batch a reentrant close already rolled back.
Stale, foreign, and never-minted serials are proved to match nothing
against a live newer transaction, and a rollback invoked four times is
proved idempotent. `iter(loader)`, `sampler.state_dict()`, and
`sampler.load_state_dict()` are each refused with `RuntimeError` while a
transaction is in flight — the load guard running **before** the state is
inspected at all — while pure planning and property reads stay permitted.

A delivered batch is a caller-owned `NativeTensor` at the dataset's dtype
beside a fresh read-only C-contiguous host `int64` array, and **the
caller closes the tensor**: after the delivery seam returns, neither the
loader, the iterator, nor the sampler retains any reference to it, so no
close path can reach one. The Phase-J objects remain **not thread-safe**;
the transaction's claim guards reentrancy, not concurrency, and J3 adds
no lock, thread, queue, future, prefetch, collate, transform, or callback
surface. J3 added no C++, no CMake entry, no C ABI symbol, no example, no
benchmark, and no dependency, so **no native rebuild, CTest run, or
sanitizer run was required and none is claimed**.

**J4 shipped the loader's in-memory state and exact mid-epoch resume**,
and it is the **first Phase-J runtime milestone that added no public name
at all**: `tensorforge.experimental.__all__` stayed at **25**.
`NativeDataLoader` gained exactly two methods, over four private module
constants (`_FORMAT`, `_FORMAT_VERSION`, `_SUPPORTED_FORMAT_VERSIONS`,
`_STATE_FIELDS`) that are exported by nothing and are not a registry. No
new module, no new class, and no new file under `src/` — the milestone is
two methods on an existing class.

`state_dict()` returns a compact **tagged wrapper** with exactly three
root keys: `format` (`"tensorforge.native_data_loader"`),
`format_version` (**1**), and `sampler` — the last being **exactly** the
unchanged version-1 sampler state. The wrapper exists because the loader
is what a caller checkpoints, and without its own tag a loader state and
a sampler state would be the same JSON, so handing one where the other
was meant would be accepted silently; both directions are asserted
rejected. The loader owns no epoch, cursor, seed, shuffle, batch size, or
drop-last field of its own, so **none is duplicated at the root**. Every
container is fresh at every call — the root dict, the sampler dict, the
dataset dict, and the `feature_shape` list — so editing what a caller was
given reaches nothing; the structure carries no permutation, no dataset
content, no NumPy object, no transaction serial, no iteration token, and
nothing whose size grows with the number of samples; and it survives a
`json` round trip and is accepted **unchanged** by the checkpoint's
existing `_validated_metadata`, which is the compatibility evidence that
makes J5's caller-managed workflow possible without the archive growing a
field. It is allowed immediately after construction, between batches
while an iterator is active, after an iterator is exhausted or
superseded, at an epoch boundary, mid-epoch, with a **closed dataset**,
and **after the loader's own `close()`** — and it is **refused** with
`RuntimeError` while a §9.4 batch transaction is in flight, through the
sampler's existing guard rather than a second authority, because inside
the commit-before-delivery window there is no honest answer. That refusal
is proved from *inside* a live transaction at both the claim and the
pending phase, where the sampler's raw fields already show the candidate
position and no snapshot may report it.

`load_state_dict(state)` is transactional in the same sense the delivery
is. Three lifecycle guards run **before** the state is read at all — the
closed guard, then the transaction guard, then the active-iteration
guard, each proved by precedence with deliberately malformed arguments —
after which the wrapper is validated completely (exact `dict`, exact
three-key set, exact tag, exact `int` version rejecting `bool`, exact
nested `dict`) and the **whole** nested sampler validation is
**delegated** to the sampler's existing validation-only seam rather than
restated, preserving every J2/J3 ordering including the four
dataset-compatibility fields in order. Only then does the commit run,
through the same non-failing write seam a delivery commit and a rollback
share — so there is no rollback path to test, because nothing mutates
until the only remaining step cannot fail. Nineteen fault classes are
each compared against a complete before/after fingerprint of the
observable world: loader, sampler, and dataset identity, the closed
states, all six values, both `state_dict()` results, the next batch, the
permutation, the plan, the iterator slot, the transaction record, the
active-iteration set, and the native live-storage count. Dataset identity
is **validated and never adopted**; the six configuration and position
values **are** adopted, so a loader deliberately built with a different
seed, batch size, drop-last setting, and position takes the state's — and
`id(loader)`, `loader.sampler`, and `loader.dataset` are unchanged.

**Exact in-memory mid-epoch restoration** is J4's exit gate, and it is
proved over two genuinely separate object graphs: a source
dataset/sampler/loader interrupted mid-epoch, and a **separately
constructed** target built with a deliberately different valid
configuration. After the load the target reproduces the remaining tail
exactly — identical index tuples, identical **raw IEEE-754 feature bits**
through `uint32`/`uint64` views, identical `int64` targets with matching
dtype, shape, contiguity, ownership, and read-only flag — then the same
canonical next-epoch position and the same following whole epochs, with
native live storage returning exactly to baseline. **No tolerance is used
anywhere.** It runs at float64 and float32, sequential and shuffled,
drop-last false and true, and at every required position: fresh, genuine
mid-epoch, final batch, epoch boundary, later epoch, short final batch,
exact divisibility, one-batch epoch, one-sample dataset, and batch larger
than the dataset. A negative control proves the sequences differ when the
restoration is omitted, and a cross-dtype leg proves batch **indices**
identical across equivalent float32 and float64 datasets while the two
states remain non-interchangeable in both directions. J4 added no C++, no
CMake entry, no C ABI symbol, no example, no benchmark, and no
dependency, so **no native rebuild, CTest run, or sanitizer run was
required and none is claimed**.

**J5 proved the caller-managed checkpoint-metadata workflow, and it added
no production code at all.** It is the second consecutive Phase-J
milestone whose export delta is zero, and the only one so far whose diff
touches no file under `src/` —
`src/tensorforge/experimental/native_checkpoint.py` is unchanged, which
was J5's own exit-gate condition. The whole milestone is
`tests/test_native_data_checkpoint.py` plus documentation.

The workflow is the one the contract has named since J0: take
`loader.state_dict()`, place it inside the `metadata` a caller already
controls, save; and on the way back, call `load_native_checkpoint` first,
then hand the returned `metadata[...]` to
`fresh_loader.load_state_dict(...)`. Every primary proof writes a **real**
version-3 `.npz` and reads it back with `allow_pickle=False`. The
manifest is inspected directly: `format` unchanged, `format_version`
**3**, the same **six** root keys, and an array inventory holding only the
manifest, `model::…`, and `optimizer::m::…`/`optimizer::v::…`. Saving
with and without loader state produces the **same** array inventory and
manifests differing in nothing but the caller's own `metadata` value, so
**the archive's capture set did not grow by one field**. There is no root
loader field, no loader array, no serialized permutation, and no dataset
payload; loader state exists only below `metadata`.

**The restoration is exact, into objects that share nothing with the
saving graph.** The proof runs at float64 and float32, sequential and
shuffled, drop-last false and true, over a model with trainable
parameters, two persistent batch-norm buffers, and a **shared** generator
alias topology — two Dropout layers on one `NativeGenerator` and a third
on its own. The restored graph is deliberately built wrong in every
family first: different parameter seeds, a different learning rate,
different generator seeds, a separately constructed dataset, and a sampler
with a different batch size, shuffle setting, seed, epoch, and cursor.
After the two calls, every parameter and persistent buffer, every Adam
`m`, `v`, and step counter, every hyperparameter, every generator's
algorithm, version, seed, and calls, the alias topology, and all six
loader values compare exactly — raw IEEE-754 bit patterns through
`uint32`/`uint64` views, exact `int64` targets with dtype, shape,
contiguity, ownership, and read-only flag. **No tolerance is used
anywhere.** The load constructs no generator, parameter, or buffer, and a
negative control proves the continuation differs when the loader
restoration is omitted and agrees the moment it is applied.

**All three delivery boundaries are proved through an archive.** A
delivery failed at the `_deliver_batch` seam — with a non-vacuity record
proving the seam ran *after* the candidate position was applied and that
`state_dict()` refused there — rolls back completely, and the checkpoint
taken immediately afterwards resumes from the **same candidate batch**,
delivering exactly those indices and bits and advancing exactly once. A
successful delivery resumes from the **following** batch with no replay.
An epoch-boundary save records the canonical `(epoch + 1, 0)` and resumes
at the first batch of the next epoch.

**The metadata boundary is the caller's, and the atomicity boundary is
honest.** `"training"`, `"data_loader"`, and `"next_step"` are
conventions no production constant spells; alternate nesting, alternate
names, and two loaders' states side by side all round-trip unchanged.
Absent loader state yields `None` and no default; a malformed but
JSON-compatible state is preserved by the archive and rejected by the
loader across ten fault shapes; a wrong-dataset state is rejected on
identity; non-JSON metadata is refused before the destination moves,
leaving an existing archive byte-identical with no temporary file. A
checkpoint load that **succeeds** followed by a loader load that **fails**
leaves the model, optimizer, and generators restored and the loader
unchanged — **nothing rolls back, because there is no cross-object
transaction** — after which the documented recovery from the same
unchanged archive succeeds. J5 added no C++, no CMake entry, no C ABI
symbol, no example, no benchmark, and no dependency, so **no native
rebuild, CTest run, or sanitizer run was required and none is claimed**.

**What still does not exist after J5**: automatic loader discovery in
either direction, the deterministic mini-batch training example, the
cross-cutting hardening matrix, and the benchmark. No production pipeline
module imports the checkpoint and no checkpoint module names a pipeline
object — both asserted by source inspection, and by driving a real save
and load with the loader's two methods patched to record any call, which
neither fired. Those are J6 onward.

**No capability moved, and none will.** `SUPPORTED_DTYPES` is
`("float64", "float32")`, `SUPPORTED_DEVICES` is `("cpu",)`, `UNSUPPORTED`
is `("cuda", "amp")`, and `RAW_KERNEL_DTYPES` is `("float64",)`. The
library exports **54** production `tf_*` symbols, the CTest inventory is
**24**, the example inventory is **15**, the native checkpoint is
`tensorforge.native_checkpoint` version **3** with `(1, 2, 3)` accepted,
and the in-memory optimizer state is version **1**. Phase J plans **no new
C ABI export** at any milestone.

What J0 delivered is `docs/native_data_pipeline_design.md` — the
implementation contract for the whole phase — plus
`tests/test_native_phase_j.py` and the status reconciliation across the
README, roadmap, project summary, support matrix, architecture, backend
experiments, and `CLAUDE.md`. The decisions it locks, so J1–J9 inherit them
rather than reopening them: three eventual public names —
`NativeTensorDataset` (J1), `NativeBatchSampler` (J2), `NativeDataLoader`
(J3) — with the permutation helpers and the batch iterator permanently
private; a strict `numpy.ndarray`-only input contract with floating-point
features and integer non-`bool` targets, whose native feature dtype is
**explicitly chosen and never inferred** from the input array and still
defaults to float64; copied host snapshots taken once at construction, with
indexing that returns copies rather than views; empty datasets and
zero-batch epochs **rejected at construction**, because the native runtime
cannot represent a zero-element tensor; a deterministic **SHA-256** dataset
fingerprint over an explicit little-endian canonical byte stream, checked
after `samples`, `feature_shape`, and `feature_dtype`; a sampler that owns
`batch_size` and `drop_last` and emits batch-index groups, with `epoch` the
active epoch, `cursor` the batches already delivered, and the epoch
boundary canonicalized immediately so every position has exactly one
representation; a deterministic shuffle that **reuses the locked
`tensorforge.splitmix64` derivation** under a domain-separated key schedule
— no new RNG algorithm, no new global generator, and deliberately no
coupling to a live `NativeGenerator`, which exposes no bit derivation to
couple to — with unbiased rejection-based bounded integers, a downward
Fisher–Yates sweep, directly implementable pseudocode, and committed
reference vectors at lengths 1, 2, 5, and 8 across seed 0, a nontrivial
large seed, and the accepted upper bound; a permutation that is a **pure
function** of `(seed, epoch, length)`, so an abandoned iterator, a rejected
state load, and a failed batch consume nothing by construction; one-epoch
iterators with a superseding `iter()` and a single atomic cursor commit
after materialization and immediately before handoff; **caller-owned**
`NativeTensor` feature batches beside fresh read-only host `int64` targets,
which the loader never retains; strict JSON-compatible state schemas with
their own format tags, carrying no payload and **no serialized
permutation**; transactional state loading whose commit is six assignments
that cannot fail; an explicit **caller-managed** checkpoint-metadata
workflow over the unchanged version-3 format, with cross-object atomicity
explicitly **not** claimed; a documented **not thread-safe, no lock**
concurrency contract; and an exact interrupted-versus-uninterrupted resume
contract compared in raw IEEE-754 bit patterns, with no tolerance anywhere.

**Two premises were checked at J0 rather than assumed**, and both held, so
the approved ladder survived inspection unchanged. The Python
implementation of the shared finalizer was verified against the **built**
library — over 48 combinations of seed, call index, and probability at
4,096 elements each, it predicted the shipped Dropout kernel's keep/drop
pattern exactly — which is what makes "no new RNG algorithm and no new
export" a measured statement rather than a hope. And the checkpoint's
existing metadata channel was verified to carry every state field
unchanged, which is what makes "no checkpoint schema change" true by
construction.

**The checkpoint's capture set did not grow.** A native checkpoint still
captures no data-loader position, no shuffle order, and no epoch counter;
what Phase J will eventually add is a loader whose position the *caller*
serializes into the metadata channel that already exists.

### Phase I — native dtype generalization and float32 CPU support (I0–I11, complete)

**Phase I is the latest phase, and it is complete: milestones I0 through
I11 have all landed.** Milestone **I11** closed it, which makes Phase I the
latest *completed* phase as well; Phase H remains complete and closed at 52
exports. **No version is claimed** — the native line stays experimental and
is not production-ready.

One public capability *has* moved, at I9 and at no other milestone:
`SUPPORTED_DTYPES` is `("float64", "float32")` and `UNSUPPORTED` is
`("cuda", "amp")`. `SUPPORTED_DEVICES`, `RAW_KERNEL_DTYPES`, the export
count (**54**), and the in-memory optimizer state version (**1**) did not
move; the checkpoint format moved to version **3** at I8.

#### I11 — cross-platform validation and Phase-I closure

**I11 is the closure milestone, and it added no capability.** It changed
**no file under `src/` or `cpp/`**: what moved is one new test module, two
guards whose premises expired, and the status surfaces. There is no C ABI
change, no numerical change, no registry, checkpoint, or optimizer-state
change, no new example or benchmark case, no dependency, and no build or
CI change.

**What the phase finally supports.** `float32` and `float64` on the native
CPU line, both publicly, since I9 — `SUPPORTED_DTYPES == ("float64",
"float32")`, `UNSUPPORTED == ("cuda", "amp")` — with **float64 still the
default** at every constructor, factory, module, and parameter, and still
what `None` means. Every public tensor factory builds either width; the
state-owning modules take a keyword-only `dtype`; the stateless ones
inherit their input's; `NativeSGD` and `NativeAdam` run at either width
with Adam's moments matching their parameter; and native checkpoint
**version 3** round-trips both, with versions 1 and 2 remaining float64-only
formats permanently and `(1, 2, 3)` accepted.

**Exact deterministic resume is proved separately at each dtype**, never as
agreement between them: the same deep model — Conv2d → BatchNorm2d → ReLU
→ MaxPool2d → Dropout → Flatten → Linear → BatchNorm1d → ReLU → LayerNorm
→ Dropout → Linear into cross-entropy with Adam, two Dropout layers sharing
one registered generator — run interrupted and uninterrupted at each width
and compared in raw IEEE-754 bit patterns. Reproduced exactly at both: the
loss sequence, the first resumed step's gradients (produced, not restored),
every parameter and buffer, every Adam moment and counter, the generator
state and alias topology, the **next Dropout event**, the final logits,
predictions, and evaluation output. All four graph-owned saved-resource
families coexist safely in one float32 graph, and live native storage
returns to zero.

**Validation performed at closure.** Windows Release (MSVC 19.44.35228)
and an isolated Windows Debug build, each 0 warnings, 0 errors, **24/24**
CTests and **54** exports with the source, PE, and Debug sets equal; a
Linux CI-equivalent (g++ 13.3.0, `-Wall -Wextra`) with 0 warnings, 24/24
CTests, 54 exports and no mangled symbol exported; Clang 18.1.3 ASan and
UBSan with instrumentation proved present, 24/24 sanitized CTests, the
complete Python suite green, and **zero** ASan and **zero** UBSan
diagnostics; a sanitizer negative control producing a genuine
`heap-buffer-overflow` inside TensorForge's own copy kernel, so the
detector is known to work; and LeakSanitizer with **no suppression file**,
whose only reports are CPython interpreter-exit allocations carrying **no
TensorForge frame**. All **15** examples and all **8** benchmark smoke
paths exit zero.

**Test totals, observed rather than derived.** Windows **7,738 passed, 0
failed, 0 skipped**; Linux **7,738 passed, 0 failed, 0 skipped**; the
sanitized suite 7,737 passed with 1 pre-existing documented skip (a
reduction NaN payload this toolchain selects differently). The suite grew
7,629 → **7,738**, which is exactly the 109 closure tests. The Linux run's
usual two skips did **not** occur, because this tree carried full history
and LF content — the observed totals are reported rather than the expected
arithmetic.

**Boundaries that did not move, and are now permanent rather than
provisional.** No CUDA, no GPU, no AMP or mixed precision, no float16 or
bfloat16, no integer or boolean tensor dtype, no casting, no promotion, no
mixed-dtype arithmetic, no dtype inference from an input array, no global
default dtype, no `astype`/`to`/`map_location`, and no device movement. The
seven handle-free raw utility kernels stay **float64-only**
(`RAW_KERNEL_DTYPES == ("float64",)`) because they take only `double*` and
an element count and so have no dtype to dispatch on — that is a separate
statement from the support registry and must never be read as one. MaxPool2d
winners stay private float64 metadata at every value width and
cross-entropy targets stay host `int64` metadata. Stable/native isolation
holds: importing `tensorforge` still loads no native library, and
`stable_framework_integration` is still `False`. **No speed is asserted
anywhere and no benchmark result file is written.** The native line remains
experimental and is not production-ready; the two dtypes are numerically
distinct and are never claimed equivalent.

#### I10 — cross-cutting hardening and benchmark characterization

**I10's only production change is one narrow checkpoint-loader validation
repair — the defect its own matrix found.** The saver validated metadata
recursively through `_validated_metadata`; the loader checked only that its
root was a dict, and `json.loads` accepts the non-standard `NaN`,
`Infinity`, and `-Infinity` literals, so a hand-written archive could carry
a value the saver would have refused to write and the loader returned it.
The same authority now runs on both sides, during archive prevalidation.
No live state was ever at risk of partial mutation — metadata reaches no
model, optimizer, or generator, and the check already sat in Phase 1 — but
"accepted and returned" was still wrong.

There is **no C++ change, no ABI or export change, no numerical runtime
change, and no benchmark-path change**, and no checkpoint schema, version,
or manifest field moved, so every float64 and float32 result, every
allocation count, and every Phase-H path is unchanged *by construction*
rather than by measurement — and no pre/post speed comparison was
manufactured where no numerical path changed. Everything else that shipped
is evidence, and one new benchmark.

- **The §9.2 mixed-dtype authority map, at every operand position.**
  Wherever an authority has more than one operand position, each is
  exercised **on its own and in both directions** — the hole a
  single-position test cannot see is a guard that compares `self` against
  `other` but never the reverse. Covered: the shared Core binary/matmul
  guard; `relu_backward`'s gradient operand; Conv2d's weight *and* bias
  separately with the others correct; both Conv2d backward directions;
  cross-entropy's saved probabilities; the five module forward authorities
  with buffers and versions fingerprinted; `NativeMSELoss` in both
  positions; a `NativeSequential` whose mismatched child is the third one;
  `copy_value_` and `_adopt_value_core`; both optimizers with a **valid
  parameter beside the invalid one**, proving one bad entry prevents every
  commit; Adam moments validated against their *parameter*; a state load
  with exactly one bad entry among many; the BatchNorm two-buffer
  transaction; and autograd's seed gradient and accumulation.
- **The C ABI proved to be a second authority, not a restatement.**
  Defence-in-depth is only real if the C++ check can fire when Python's
  cannot, so both halves are proved separately: fifteen compute exports
  are driven through **production geometry** with the destination forced
  to the other width — a state Python's guard cannot emit — and, with
  `_require_matching_metadata` neutered, mismatched *operands* still
  reject at the boundary. The bypass has its own negative control: with
  the guard neutered and **matching** dtypes the same call succeeds.
- **Validation orderings recorded rather than chosen.** Liveness before
  dtype; type before dtype; dtype before a broadcast-shape conflict (which
  is what makes "no allocation on a mixed-dtype call" unconditional);
  **shape before dtype in `copy_value_`**, the opposite order, kept; each
  module's own rank rule before dtype; and the seed gradient's dtype
  **before** graph staleness, with the staleness rule proved to fire on the
  next attempt once the seed is right.
- **All four graph-owned saved-resource families in one float32 graph.**
  I9 exercised them *across* a run and scoped that claim honestly; I10
  built the configuration where they genuinely coexist — the model in
  `eval()` so BatchNorm takes snapshots, the Dropout layers put back into
  training through the ordinary public per-module `train()` API, no
  undocumented flag touched. The graph is then **walked**, and every
  resource classified by the op of the node that adopted it, so "all four
  are here" is an observation rather than an inference from shapes. Proved
  on that one graph: the modes; the running buffers not moving; the shared
  generator advancing by exactly two; two Dropout masks, one winner
  buffer, one saved probability set, two BatchNorm snapshots; every value
  resource at float32 while **the winner buffer is float64**; and no
  resource aliasing a parameter, buffer, input, output, or another
  resource. Its **negative control** leaves the model wholly in training
  mode and finds three families, not four.
- **Every lifecycle over that graph**: a retained backward keeps all four
  alive and the second pass accumulates; a failure injected after backward
  temporaries exist commits no leaf gradient, keeps every saved resource,
  consumes no generator call, and a corrected retry succeeds; a one-shot
  backward releases all four exactly once and the graph cannot be reused;
  an abandoned graph releases all four through explicit `close()`; and a
  genuinely no-grad forward — parameters frozen too — creates none.
- **117 malformed-checkpoint cases, each applied alone, at both dtypes.**
  Archive envelope and manifest structure; the model section down to bool
  dimensions, overflowing shapes, and whitespace or case in a dtype string;
  both optimizer shapes including v3 moment entries, bare-name downgrades,
  and a moment referencing a model payload; ten malformed canonical-uint64
  spellings, alias topology, and a shared stream saved as independent; and
  metadata. After **every** rejection the complete world is fingerprinted —
  values as raw bit patterns, identities, versions, gradients, generator
  state and alias topology, training flags, moments and counters. The
  fingerprint helper has its own negative control. Genuine v1, v2, and v3
  archives still load in the same module, and a *successful* load is
  asserted to advance only the parameter versions, by exactly one.
- **Two findings recorded rather than "fixed"**, because I10 changes no
  production behavior without a proved defect: non-finite checkpoint
  metadata is validated on **save** and opaque on **load** (it touches no
  live state, and a forged manifest is proved to disturb nothing else); and
  `maxpool2d_backward` has exactly **one** value operand, so the absence of
  a second mixed-dtype position there is documented — with the same winner
  buffer proved to drive a correct backward at either value width — rather
  than left looking like a gap.
- **Concurrency tested at exactly the width of the claim.** Immutable dtype
  under eight barriered readers; concurrent snapshots each internally
  coherent; two racing whole-model loads leaving the state equal to **one**
  donor and never a mixture; and the BatchNorm running-statistics
  transaction proved not to tear against a concurrent **checkpoint save** —
  deliberately the participating reader, because a bare `state_dict()` does
  not take the guard and thread-safe concurrent training snapshots are not
  offered. Phase G's reservation rule is asserted as written: concurrent
  stochastic use of one generator is **refused**, deterministically, and
  the counter equals the number of *successful* draws, so a refusal burns
  nothing. Bounded joins throughout, thread exceptions captured, and a
  negative control proving a barrier fails when a party never arrives.
- **Benchmarks: one new harness, deliberately separate.**
  `benchmarks/benchmark_native_dtype.py`, 24 cases across eleven families,
  each width measured **separately** and never as a ratio, with a control
  pair giving the machine's noise band. Phase H's harness was **not**
  extended: its case inventory is pinned by test as "the H0 set", and a
  dtype axis would have made every published Phase-H number mean something
  else. Four gates chosen per family; the `summation_bound` one is the
  classical `2 n eps max sum|terms|` rule computed from the actual
  operands, adopted **after** the gate caught a real divergence between
  TensorForge's strict sequential accumulation and NumPy's BLAS reference,
  and derived rather than tuned. No timing assertion, no threshold, no
  result file — the last proved by running the CLI from an empty directory
  and showing it stayed empty. Measured results, including the neutral and
  negative findings and this machine's large run-to-run variance, are in
  [backend_experiments.md](backend_experiments.md).
- **Scope.** **One** production file changed —
  `src/tensorforge/experimental/native_checkpoint.py`, for the metadata
  repair above. No `cpp/` change; **54** exports and **24** CTests
  unchanged; checkpoint still version 3 accepting `(1, 2, 3)`; in-memory
  optimizer state still version 1; registries unmoved; examples still
  **15**; no dependency, build option, or CI change; no numerical runtime
  or benchmark-path change, so float64 and float32 numerical behavior are
  unchanged.
- **Tests:** `tests/test_native_float32_hardening.py` (138),
  `tests/test_native_float32_checkpoint_corruption.py` (36, carrying 117
  corruption cases at each of two dtypes plus the 13 metadata cases at
  each of v1/v2/v3), and
  `tests/test_native_dtype_benchmark.py` (41). Suite 7,409 → **7,629**.
- **One arithmetic correction.** The I9 record read "Suite 7,082 → 7,404",
  which counted its two new files' 322 tests but not the 5 further tests
  its three added functions in existing files contribute. The observed
  total at the I9 commit is **7,409**; repaired rather than rewritten away.

#### I9 — public float32 integration and the exact-resume proof

**I9 made float32 public, and it added no C ABI symbol, no CTest, and no
C++ line.** It is the phase's one and only public registry change, and it
happened in that order deliberately — proof first, promise second. What
shipped:

- **The integrated proof, written and passing before the registry moved.**
  `examples/native_float32_training.py` runs `Conv2d(1→4, 3×3, pad 1) →
  BatchNorm2d(4) → ReLU → MaxPool2d(2) → Dropout(0.25) → Flatten →
  Linear(36→8) → BatchNorm1d(8) → ReLU → LayerNorm(8) → Dropout(0.25) →
  Linear(8→3)` over raw logits into `NativeCrossEntropyLoss`, trained by
  `NativeAdam(lr=0.05)`, on twelve fixed `1×6×6` images in three fixed
  batches of four, for 12 steps interrupted after 5. It runs **twice at
  each dtype** — uninterrupted, and interrupted/checkpointed/resumed into a
  completely fresh model, optimizer, and generator set built from
  deliberately different seeds — and compares each dtype **only against
  itself**.
- **Two Dropout layers share one registered `NativeGenerator`**, so the
  model carries a genuine alias topology (`conv_dropout.generator`
  canonical, `dense_dropout.generator` aliased) rather than a single
  counter: two generator calls per training forward, one canonical state
  plus the full alias map written to the version-3 archive, and the map
  re-validated against a live traversal on load.
- **Every comparison is over raw IEEE-754 bit patterns** — `uint32` at
  float32, `uint64` at float64 — with no tolerance, no `allclose`, and no
  float32-versus-float64 comparison anywhere. The helper that produces
  them **refuses** an array of the other width, so a match can never
  quietly mean "matched after a conversion". Proved equal: the resumed loss
  suffix and the whole loss sequence, every parameter, every persistent
  buffer, every Adam `m`, `v`, and step counter, the optimizer
  hyperparameters, the generator's algorithm/version/seed/calls, the alias
  topology, the final training logits, the final predictions, the final
  evaluation output, and the validated external-loop metadata.
- **Gradients are proved *produced*, not restored.** They are not
  checkpointed, so the first resumed step captures every parameter's
  gradient after backward and before the optimizer commits — the one moment
  they exist — and compares it to the same step of the uninterrupted run.
- **The next Dropout mask matches too.** The same registered Dropout path
  in both final models is called once with an identical all-ones tensor, so
  the output *is* the multiplier mask and no private state is exposed: the
  masks are bit-identical, the pattern is proved non-degenerate (8 dropped,
  24 kept), each call consumes exactly one generator call, and the shared
  alias path is proved to observe the same advanced object — restoration of
  a *sharing relationship*, not of two equal numbers.
- **All four graph-owned saved-resource families are exercised, and the
  claim is scoped exactly.** Three ride a *training* graph (both Dropout
  masks, the MaxPool2d winners, cross-entropy's saved probabilities); the
  BatchNorm **evaluation snapshots** exist only on an *evaluation* graph,
  because training-mode BatchNorm normalizes with the batch's own
  statistics and takes no snapshot. So they are exercised **across** the
  run rather than coexisting in one graph, and the example says so. The
  eval graph is proved independent of its buffers by advancing all four
  underneath it and re-running its backward to a bit-identical control.
- **Negative controls.** A resume that ignores the validated metadata and
  restarts the schedule at step 0 diverges — and the control is proved
  non-vacuous, because `SPLIT_STEP` is deliberately not a multiple of the
  batch count. A model whose generator was left at the fresh seed draws a
  different next mask. The two dtypes' loss sequences are proved *unequal*,
  which is what makes "each compared only against itself" a real
  separation.
- **Lifecycle.** Native live storage returns exactly to baseline (`0 / 0`)
  across both dtypes, both runs each, the mask proof, and the snapshot
  proof.
- **Then the registry moved**: `SUPPORTED_DTYPES` `("float64",)` →
  `("float64", "float32")`, `UNSUPPORTED` `("float32", "cuda", "amp")` →
  `("cuda", "amp")`. `normalize_dtype("float32")` succeeds; every public
  constructor (`NativeStorage`, `NativeTensorCore.from_array`/`.zeros`/
  `.full`, `NativeTensor.from_array`/`.zeros`/`.full`) builds a float32
  tensor; views, operations, and gradients preserve it; `to_numpy()`
  returns `np.float32` and never widens. The example's one ingress helper
  switched to the public constructor and the entire proof was rerun.
- **What did *not* move, and is asserted in both directions.** float64 is
  still the default at every constructor, factory, module, and parameter
  and is still what `None` means; the dtype is still **never inferred**
  from an input array; there is still no casting, promotion, mixed-dtype
  arithmetic, `astype`, `to`, `.float()`, `.double()`, `map_location`,
  global default, or device movement; `SUPPORTED_DEVICES` is still
  `("cpu",)` and `RAW_KERNEL_DTYPES` still `("float64",)`; float16,
  bfloat16, integer and complex dtypes are still rejected; and the NumPy
  reference backend's own `supported_dtypes` is still `("float64",)`,
  because Phase I is a native-line phase.
- **`backend_info()`'s flat `"dtype"` key was decided explicitly** rather
  than left accidentally misleading: it is **kept**, still `"float64"`, and
  is now documented in code, docstring, and tests as the **default**
  statement. `supported_dtypes` is the capability row; `raw_kernel_dtypes`
  is one small permanent limitation; three rows, three questions, and a
  test asserts the flat key agrees with `normalize_dtype(None)`.
- **Guardrails were split, not weakened.** 125 current-truth registry
  assertions across 46 test files moved to the new values. The Phase-G and
  Phase-H closure records keep their literals as **history** and now assert
  the live tuple as *their* value minus exactly what a later phase is on
  record as moving. `test_native_phase_i.py`'s per-milestone exit gates
  route through one shared helper checking the change was exactly
  `"float32"`, in one direction, with the default unmoved. The `test_docs`
  over-claim parser dropped `float32` from its banned list and **gained a
  negative control** proving it still fires on seven sentences it must
  catch and does not fire on the one I9 made true.
- **Scope.** No C++ source or header changed; **54** exports and **24**
  CTests unchanged; checkpoint still version 3 accepting `(1, 2, 3)`;
  in-memory optimizer state still version 1; no new module, loss,
  optimizer, or operation; no dependency or build option; no benchmark
  implementation change (I10 owns benchmarking, and added exactly one new
  harness rather than changing any existing one); no timing assertion and
  no committed number.
- **Tests:** `tests/test_native_float32_training.py` (147) and
  `tests/test_native_float32_public.py` (175). Examples 14 → **15**.

#### I7 — modules, parameters, buffers, initialization, and Dropout

**I7 made float32 a module dtype, and added no C ABI symbol.** I6 left
float32 classifying; I7 gives it parameters, modules, persistent buffers,
normalization, and Dropout — and with Dropout, the last float64-only
numerical family in the runtime. What shipped:

- **Six state-owning constructors gained a keyword-only `dtype`** —
  `NativeParameter`, `NativeLinear`, `NativeConv2d`, `NativeLayerNorm`,
  `NativeBatchNorm1d`, `NativeBatchNorm2d` — accepting exactly
  `"float64"` and `"float32"`, defaulting to `"float64"`, and all six
  routing through **one** shared private validator so no constructor
  invents a dtype rule of its own. Keyword-only at every one, so no
  positional shape moved and no existing call can be reinterpreted. The
  set is asserted **closed**: no stateless module, loss, metric,
  generator, container, or optimizer took one, and no `device` argument
  was added anywhere.
- **`NativeParameter` distinguishes host data from a native tensor.** Host
  data crosses the explicit host-to-native conversion boundary and is
  converted once, so a float64 array becomes a float32 parameter by exactly
  one rounding — asserted bitwise against NumPy's own narrowing. A live
  `NativeTensor` must already carry the requested dtype, because there is
  no tensor cast in this runtime; a mismatch is refused in both directions
  before any allocation.
- **Initialization did not move.** The host draw is the same local
  `numpy.random.default_rng(seed)` stream, in the same order, at the same
  sizes, with the fan-in bound computed once in binary64 — so a float32
  layer with seed *S* holds exactly `float32(the float64 draw with seed
  S)`, asserted as raw bit patterns for both weight and bias at both
  `NativeLinear` and `NativeConv2d`. The float64 half is asserted against
  the host stream itself. Because each constructor owns a *local*
  generator, changing one layer's dtype shifts no other layer's values.
- **Normalization is dtype-general as composition, with no new kernel.**
  Affine parameters, both BatchNorm running buffers, the graph-safe
  evaluation snapshots, every training and evaluation temporary, and every
  scalar the forwards materialize — `eps`, `momentum`, `1 - momentum` —
  are at the module's dtype, the scalars through two new private
  tensor-level constructors. The Python floats stay Python floats; only
  their materialization is dtype-aware, and each is narrowed once at the
  fill boundary. The Phase-F guarantee that neither normalization module
  imports `ctypes` or `backends` or touches `NativeTensorCore` is
  unchanged, which is why the shared validator lives in its own private
  module.
- **The atomic two-buffer transaction gained one dtype validation.** Its
  plan/stage/commit/finalize structure and its commit boundary are
  untouched; a replacement at the wrong width is refused before either live
  buffer changes, leaving both values, identities, and versions exactly as
  they were. The BatchNorm forward additionally re-proves that all four
  numeric state objects still carry the module's dtype, so the reported
  dtype can never become a stale claim.
- **Dropout became dtype-general with its exact ABI shape unchanged.**
  `tf_core_dropout_forward` keeps the same symbol, the same eight
  arguments, the same order and types, and gained one operand-agreement
  guard over its three handles plus one dispatch above a templated kernel
  that moved into `tf_random_internal.h`. There is no dtype branch inside
  the element loop and no `_f32`/`_f64` symbol.
- **The random derivation is untouched.** `dropout_uniform` is still the
  binary64 53-bit conversion at every width, so one
  `(seed, call_index, element count)` key drops **exactly** the same
  elements at float32 and float64 — proved at float32 against the *same*
  committed Phase-G keep vectors rather than a second table, and by direct
  comparison of the two patterns over 512 elements at five probabilities.
  Only the two multiplier values differ. The kept one is
  `static_cast<T>(1.0 / (1.0 - p))`, computed once in binary64 and narrowed
  once, and at float32 that is *observable*: at p = 0.025 the narrow-once
  value provably differs from an all-binary32 recomputation, and the kernel
  is asserted to equal the first and differ from the second.
- **Generator state, algorithm, version, and call accounting are
  unchanged**, and the accounting is asserted identical at both widths on
  every path: one call for a successful stochastic forward, none for a
  failure, none in evaluation, none at `p == 0`.
- **The last of the five explicit float64-only Python gates came out**, so
  no float64-only compute path is left to name.
- **A version-2 checkpoint refuses a float32 model on the way out**, before
  the temporary file exists. Versions 1 and 2 are float64-only formats
  permanently, and the loader already proves every archive array is exactly
  `np.float64` — so a float32 payload under a version-2 manifest would have
  been a file this library refuses to read back. Nothing is cast, widened,
  or guessed; dtype-aware serialization is version 3, at I8.
- **One pre-existing defect was fixed.** `NativeLinear.__init__` allocated
  its weight and then its bias with no cleanup between them, so a failed
  bias allocation abandoned the weight's native storage to garbage
  collection — a module the caller never receives and can therefore never
  close. Its younger siblings had had the deterministic cleanup since their
  own milestones; `NativeLinear` (v3.4) predates the pattern. The test that
  covers it spies on the parameter constructor rather than counting live
  storage, because the half-built module is unreachable the instant
  `__init__` re-raises.
- The native CTest inventory moved **23 → 24** (`test_dtype_dropout`).

**Two surfaces were inspected and deliberately left alone.**
`NativeSequential` takes no dtype and enforces none — its locked contract
is that forward adds no node, copy, or validation of its own, so a
mismatched child raises at *that child* with both dtypes named, and a model
may legitimately hold both widths with no bridge between them.
`NativeGenerator` and `NativeDropout` took no dtype either: generator state
is dtype-independent and Dropout inherits its width from its input.

**What I7 did not change:** the export count (**54**), `SUPPORTED_DTYPES`
(`("float64",)`), `SUPPORTED_DEVICES` (`("cpu",)`), `UNSUPPORTED`
(`("float32", "cuda", "amp")`), `RAW_KERNEL_DTYPES` (`("float64",)`), the
checkpoint format (version **2**, versions **(1, 2)** accepted), any public
tensor constructor, or any float64 value — asserted bitwise across
initialization, LayerNorm, both BatchNorm buffers, and the Phase-G Dropout
known-answer vector. float32 optimizer state does not exist: `NativeAdam`
refuses a float32 parameter when it allocates its moments, `NativeSGD` when
`step()` materializes its learning-rate scalar, and the refusal is atomic.

#### I8 — optimizer state and native checkpoint version 3

**I8 made float32 survive a step and a file, and added no C ABI symbol.**
I7 left float32 with parameters, modules, and buffers; I8 gives it
optimizer state and a checkpoint that can say so. It is the smallest
runtime change of the phase, and deliberately: I3-I7 had already made
every operation the optimizers compose dtype-general, so the whole of
float32 `NativeSGD` and `NativeAdam` is three constructors moving to their
private typed twins — `NativeTensorCore._typed_full` for SGD's per-step
`lr` scalar and Adam's `_StepConstants`, `NativeTensor._typed_zeros` for
Adam's moments — each now allocated at **its own parameter's** width.
Public construction is untouched: `full`, `zeros`, and `from_array` still
validate against the public registry and still reject float32.

Adam's `m` and `v` match their parameter in dtype, shape, and device,
start at bit-exact `+0.0`, and stay plain graph-free tensors; counters
stay Python ints. Allocation is still eager and still releases every
buffer it built if any allocation fails, proved at **every** moment
position at both widths. One optimizer may hold parameters of both
widths: state is per parameter, so each entry is internally
dtype-consistent and nothing bridges them, and the scalar caches key on
`(dtype, device)` — so a mixed collection builds one scalar set per
**active dtype** rather than one per parameter (SGD 1 and 2; Adam 8 and
16). **Phase H's H4 architecture is preserved whole.**

**Design §15.3's open question was resolved on measurement, and the answer
was that the two spellings differ.** H4 replaced a native
`full(1 - beta**t).reciprocal()` composition with a Python
`1.0 / (1 - beta**t)`, which is an exact substitution at binary64. At
binary32 it is not: the kernel divides by the **narrowed** denominator
while Python divides by the un-narrowed one, so the two are reciprocals of
different real numbers. Over a deterministic sweep — the default betas at
`t = 1…2000`, betas near 0 and near 1, and 200,000 randomized pairs — they
disagree by one ULP for a large fraction of inputs, the **default betas
included**: at `beta1 = 0.9, t = 5`, `0x401C48CA` against `0x401C48CB`. So
I8 computes the coefficient the way the kernel does, narrowing the
denominator first. `1 - beta ** t` is still evaluated in binary64, so the
cancellation §15.3 warns about is still avoided and "float32 throughout"
stays rejected; the step gains no allocation and no kernel call, because
binary64's 53 bits exceed the `2p + 2 = 50` a double rounding would need
to be visible; and at binary64 the narrowing is the identity, so H4's
original proof stands untouched. The reference is **real native execution
of the retained pre-H4 composition**, and the witness is proved
non-vacuous.

**Native checkpoint format version 3** declares every numeric entry's
dtype explicitly, with accepted versions `(1, 2, 3)`. Every new save
writes 3 whatever the model holds, because the version describes the
schema rather than the content. Adam's `"m"` and `"v"` became lists of
entry objects (`{"array", "shape", "dtype", "device"}`) instead of bare
archive names, so a moment's metadata is **carried rather than inferred
positionally** from `"parameters"` — an inference that holds only while
the two lists agree, which is exactly what a malformed archive violates.
`_read_arrays` now validates each array against its **declared** dtype
instead of a hardcoded `np.float64`, so a dtype/payload disagreement is
rejected in either direction and a foreign byte order fails as part of the
dtype identity. float32 model values, persistent buffers, and Adam moments
round-trip **bit for bit**, signed zeros, infinities, subnormals, and NaN
payloads included. There is **no cast, no `map_location`, and no device
movement** on either path, proved by parsing the module's AST rather than
by a substring search that would trip over the docstring promising their
absence.

**Versions 1 and 2 remain float64-only formats, permanently.** A declared
float32 entry in a v1/v2 manifest is rejected naming the version and why,
and a payload is never *guessed* to be float32. I7's save-time refusal was
removed because there is now a version that can say "float32" — not
because the older formats became permissive.

**What I8 did not change:** the export count (**54**), `SUPPORTED_DTYPES`
(`("float64",)`), `SUPPORTED_DEVICES` (`("cpu",)`), `UNSUPPORTED`
(`("float32", "cuda", "amp")`), `RAW_KERNEL_DTYPES` (`("float64",)`), the
CTest inventory (**24**), the **in-memory** optimizer state schema
(version **1** — float32 metadata simply became reachable through it
without a line changing), any public tensor constructor, any float64
value, or any transactional, identity, aliasing, or rollback guarantee.
Neither optimizer gained a `dtype` or `device` argument, because neither
owns a dtype it could choose — only state that must match a parameter. The
exact float32 resume proof and the registry itself are milestone **I9**.
Tests: `tests/test_native_float32_state.py` (135 new); the suite moved
6,947 → **7,082**, with zero skips.

#### I6 — stable math and classification dtype support

**I6 made float32 classify, and added no C ABI symbol.** I5 left float32
convolving and pooling; I6 gives it the stable-math and classification
stack, and nothing more. What shipped:

- **Four exports became dtype-general** — `tf_core_softmax_forward`,
  `tf_core_log_softmax_forward`, `tf_core_cross_entropy_forward`, and
  `tf_core_cross_entropy_backward` — every one the symbol Python already
  declared, keeping its argument list, calling convention, validation
  classes, and ownership contract. Each validates that **every**
  participating numeric handle agrees (two for each transform, three for
  each cross-entropy direction) and dispatches **once** from the storage
  tag into templated kernels; nothing below that point branches on dtype.
- **The four compute kernels became templates deduced from their pointer
  arguments** and moved into `tf_classification_internal.h`, so both
  instantiations reach the exported wrappers *and* the CTests that compile
  that file directly. `T = double` is the pre-I6 code statement for
  statement: `double total = 0.0` became `T total = T(0)`,
  `static_cast<double>(batch_size)` became `static_cast<T>(batch_size)`,
  and `contribution -= 1.0` became `contribution -= T(1)`. The slice
  decomposition, the strict `>` maximum scan, the fused log-sum-exp, and
  the in-place normalization are unchanged.
- **Everything numerical happens at the element type**, with no hidden
  float64 accumulator: the maximum, the shift, `std::exp`/`std::log` on the
  element type (so a float32 slice takes the `float` overload rather than
  widening and narrowing back), the normalizing sum, the log-normalizer,
  the per-row loss, the batch total, the mean divisor, the gradient
  contribution, and the upstream scale.
- **The batch-loss accumulator carries the float32 accumulation witness.**
  Every other operation in this family produces its destination with a
  single correctly-rounded operation, where binary64-then-round-once is
  provably indistinguishable from binary32 — accumulation is the one place
  the two policies can differ. On a batch whose first row contributes
  exactly 200 and whose remaining 199 contribute ~6.1e-6 each, sequential
  binary32 stays at exactly 200 while binary64-then-narrow lands ~1.2e-3
  higher; TensorForge equals the first and differs from the second by raw
  bit pattern. The per-row losses come from the kernel itself, so no
  `expf`/`logf` assumption enters the witness.
- **The saved probabilities carry the graph dtype** and remain the only
  thing the backward reads — the logits are not a parameter of the kernel,
  of the export, or of the Core wrapper, and no exponential or logarithm is
  evaluated there. They ride the unchanged `graph_resources` contract:
  released exactly once, retained under `retain_graph=True`, alive across a
  failed retryable backward, closed immediately by a no-grad forward.
- **The class targets stay host `int64` metadata at every width** — copied
  into an independently owned array, revalidated index by index in C++,
  caller-independent after the forward, never a tensor, never inferred from
  the logits. **No integer tensor dtype, integer-storage export, or target
  export was added.**
- **Private float32 `NativeTensor` graphs differentiate through softmax,
  log-softmax, and cross-entropy** with no change to the graph structure at
  all: the two transform backwards are composed from Core operations I3 and
  I4 had already generalized, and the cross-entropy backward is one call
  into the Core backward.
- The two hard `dtype != "float64"` gates on the cross-entropy Core
  wrappers — two more of the five recorded in the contract's §2.3 — became
  dtype-general acceptance. **One of the five is left**, and it is
  Dropout's, which stands for I7.
- The native CTest inventory moved **22 → 23**
  (`test_dtype_classification`).

**The measured finding, and the contract sentence it changed.** The
contract's §10.5 said without qualification that neither kernel produces a
NaN or an infinity for any finite input for which float64 does not,
"because the shift is what guarantees that and the shift is
width-independent". That is right about `exp` and one clause too strong
about the subtraction. For the finite binary32 slice `[3.0e38, -3.0e38]` —
both operands inside `FLT_MAX` — the spread is 6.0e38, so `x - m` itself
overflows: float32 `log_softmax` reports `-inf` where float64 reports
`-6.0e38`, and float32 `cross_entropy` reports `+inf` where float64 reports
`6.0e38`. `softmax` is **unaffected** and still gives exactly `[1.0, +0.0]`.
Those infinities are the correctly rounded IEEE-754 results for quantities
with no binary32 representation at all, and the identical thing happens at
binary64 past ~1.8e308 — a dynamic-range fact, not a float32 defect. **No
production code changed for this**: no widened intermediate (that would be
mixed precision), no clamp, no special case. The qualification was written
into §10.5 with the counterexample and its table, and it is asserted in
both directions by test at both widths so no later milestone can quietly
"fix" it.

**Two surfaces were inspected and deliberately left alone.**
`NativeCrossEntropyLoss` is a thin delegate to `logits.cross_entropy(...)`
with no float64 assumption to remove, and `native_accuracy` is a
reporting-only helper that materializes through `to_numpy()` and returns a
Python float. Neither gained a dtype argument; both simply work when handed
a private float32 graph, which is the *operation* being dtype-general and
is **not** public float32 module support.

**What I6 did not change:** the export count (**54**), `SUPPORTED_DTYPES`
(`("float64",)`), `UNSUPPORTED` (`("float32", "cuda", "amp")`),
`RAW_KERNEL_DTYPES` (`("float64",)`), the checkpoint format (version **2**,
versions **(1, 2)** accepted), any public API, any float64 result — proved
bit-identical against a build of the committed I5 sources over 1,346 result
values spanning NaN, ±inf, ±0.0, 1e300/1e308 extremes, every axis, and both
reductions — and any Phase-H or Phase-E traversal, formula, or allocation
policy. Dropout, normalization, optimizers, modules, and checkpoints all
still reject a float32 handle before touching memory, and no public
constructor produces a float32 tensor — so float32 parameters, modules,
optimizers, and training still do not exist.

#### I5 — CNN and pooling dtype support

**I5 made float32 convolve and pool, and added no C ABI symbol.** I4 left
float32 accumulating through reductions and matmul; I5 gives it the CNN
stack, and nothing more. What shipped:

- **Five exports became dtype-general** — `tf_core_conv2d_forward`,
  `tf_core_conv2d_input_backward`, `tf_core_conv2d_weight_backward`,
  `tf_core_maxpool2d_forward`, and `tf_core_maxpool2d_backward` — every
  one the symbol Python already declared, keeping its argument list,
  calling convention, validation classes, and ownership contract. Each
  validates that its numeric operands agree (the nullable conv2d bias
  included, when present) and dispatches **once** from the storage tag
  into templated kernels; nothing below that point branches on dtype.
- **The six Conv2d compute paths and both pooling kernels became
  templates deduced from their pointer arguments** and moved into
  `tf_conv2d_internal.h` / `tf_pooling_internal.h`, so both instantiations
  reach the exported wrappers *and* the CTests that compile those files
  directly. `T = double` is the pre-I5 code statement for statement —
  every loop nest, tap range, seed, and H9 accumulation-order proof
  carried over verbatim — and the three geometry predicates are untouched,
  so both widths take the same traversal for the same geometry.
- **Conv2d accumulates in the element type**, witnessed in all three
  directions on both traversals of each: `1.0` followed by eight copies of
  `2**-24` stays exactly `1.0` under sequential binary32 and lands higher
  under binary64-then-narrow, and TensorForge equals the first and differs
  from the second by raw bit pattern.
- **The MaxPool2d winner buffer stays private float64 at every value
  dtype** (design §13.3), with its own guard on both sides of the
  boundary: Python allocates it with an explicit `dtype="float64"` and the
  backward validates the tag as exactly float64, while the C ABI refuses a
  non-float64 winner handle before reading or writing anything. The
  `2**53` exact winner-plane bound is float64's and did not shrink to
  float32's `2**24`; a float32 pool over a plane of `2**24 + 2` elements
  records the winner offset `2**24 + 1` — not float32-representable —
  exactly.
- **Private float32 `NativeTensor` graphs differentiate through
  convolution and pooling**: input, weight, bias, and pooling gradients
  carry the graph dtype, and the float64 winner rides the unchanged
  `graph_resources` contract — released exactly once, retained under
  `retain_graph=True`, alive across a failed retryable backward, closed
  immediately by a no-grad forward.
- The two hard `dtype != "float64"` gates on the pooling Core wrappers —
  half of the five recorded in the contract's §2.3 — became dtype-general
  acceptance; the cross-entropy and dropout gates stand for I6 and I7.
  (Recorded with I6: the two cross-entropy gates came out there, leaving
  Dropout's as the only one of the five still standing.)
- The native CTest inventory moved **21 → 22** (`test_dtype_cnn`).

**What I4's record said could not be tested there is exercised here too:**
a first draft of the C++ special-values sweep put two NaNs into one
weight-gradient accumulation (an input NaN plus a `0 × inf`-manufactured
one) and observed the H2 ADDSD operand-selection carve-out in a second
family — MSVC picks different operand registers for the generic and gather
loops, so the surviving payload's sign differed. The sweep was corrected
to the contractual cases (at most one NaN, at most one infinite term per
destination), which agree bit-for-bit on both paths at both widths; **no
production code changed for this**.

**What I5 did not change:** the export count (**54**), `SUPPORTED_DTYPES`
(`("float64",)`), `UNSUPPORTED` (`("float32", "cuda", "amp")`),
`RAW_KERNEL_DTYPES` (`("float64",)`), the checkpoint format (version **2**,
versions **(1, 2)** accepted), any public API, any float64 result, and any
Phase-H traversal, predicate, or allocation policy. Classification,
Dropout, normalization, optimizers, modules, and checkpoints all still
reject a float32 handle before touching memory, and no public constructor
produces a float32 tensor — so float32 parameters, modules, optimizers,
and training still do not exist.

#### I4 — reductions, matmul, views, and core autograd

**I4 made float32 accumulate, and added no C ABI symbol.** I3 left float32
computed on by operations that produce each output with a single
correctly-rounded operation; I4 gives it the families where values are
*combined* — and the private graph that composes them — and nothing more.
What shipped:

- **Three exports became dtype-general** — `tf_core_sum`,
  `tf_core_matmul`, and `tf_core_narrow_backward` — plus the two scalar
  storage primitives `tf_storage_scale` and `tf_storage_fill`. Every one is
  the symbol Python already declared, with its argument list, calling
  convention, validation order, traversal tiers, and ownership contract
  intact; `tf::require_float64` became `tf::require_matching_dtype`, and one
  `switch` per exported call selects the instantiation.
- **Both traversals of both families, instantiated twice from one source.**
  H6's `sum_contiguous_blocks` and the retained `sum_generic_strided`, and
  H2's `matmul_row_sweep` and the retained `matmul_generic_strided`, became
  templates over the element type and moved into
  `tf_reduction_internal.h` / `tf_matmul_internal.h` — the ordinary reason a
  template must, so both instantiations reach the exported wrapper and the
  CTests that compile those files directly. `T = double` is the pre-I4 code
  statement for statement, so float64 runs exactly what Phase H measured,
  and each optimized path keeps its oracle **per dtype**. Both metadata
  predicates are untouched, because they read `int64` layout only — so both
  widths take the same path for the same layout.
- **The accumulator follows the element type.** `double sum = 0.0` became
  `T sum = T(0)`, `double accumulator = dst[o]` became `T accumulator`, and
  `out[j] = 0.0 + a_ik * b_row[j]` became `T(0) + a_ik * b_row[j]`. At
  `T = double` each *is* the old literal; the explicit `T(0) +` is still
  written out rather than folded away, because `0 + (-0)` is `+0` while
  `-0` alone is not, at either width.
- **"float32 accumulates in float32" became a measured claim.** I3 recorded
  that no runtime test could separate binary32 from binary64-then-round-once
  for a *single* correctly-rounded operation, and declined to invent one. A
  sum can separate them: on `1.0` followed by eight copies of `2**-24`,
  sequential binary32 stays at exactly `1.0` (bits `0x3F800000`) while
  binary64-then-narrow lands four ULPs higher (`0x3F800004`). TensorForge is
  asserted equal to the first **and unequal to the second**, by raw bit
  pattern, on both reduction traversals and both matmul paths, from C++ and
  from Python. The structural check over the source stands beside it — the
  witness proves the result, the structure proves no width in the source
  could make one path right and another wrong.
- **The mean scalar is narrowed once, before the loop** (design §7.4).
  `1/count` is computed once in binary64, crosses the unchanged `double` ABI
  parameter, and is converted to the element type before the multiply loop —
  so a float32 mean is deterministic, identical on every platform, and
  independent of `count`'s magnitude. Asserted behaviourally: for
  `count == 3` and a sum of 5 the two orderings differ by one representable
  step, and TensorForge must produce the specified one.
- **`narrow_backward` is still a scatter, not an identity copy.** It writes
  only the narrowed region and every un-narrowed cell keeps the zero the
  allocation gave it — and that zero *is* the gradient, which is why H1
  rejected this destination from the uninitialized path and why I4 did not
  revisit that. Because it assigns rather than computes, it reproduces its
  source's bits exactly at both widths, proved with a `-0.0` and a
  signalling NaN in the upstream.
- **Private/internal float32 `NativeTensor` graphs run forward and
  backward** through every Core operation landed so far — elementwise,
  broadcasting, reshape, transpose, `.T`, narrow, contiguous copy, sum,
  mean, and matmul. Every gradient carries its tensor's dtype, every
  backward temporary the graph's, and every materialized constant — `0.5`,
  `-1`, `1/count`, the ones seed, broadcast-back's zeros — is built at the
  operand's dtype through a private typed constructor rather than a public
  one, which is what design §11.4 requires. `_broadcast_back` remains a
  genuine reduction rather than a copy.
- **Mixed dtype is rejected before any allocation or mutation**, in every
  participating handle position independently, and now also for the backward
  seed and for gradient accumulation. A rejected call allocates nothing,
  leaves its destination byte-for-byte unchanged, and leaves an
  already-accumulated gradient exactly as it was.
- **float32 finite differences, with a step and tolerances derived rather
  than inherited.** Central differences balance truncation against
  cancellation, and the unit roundoff is `2**-24` at binary32 against
  `2**-53` at binary64, so the float64 step is wrong by orders of magnitude.
  The step is `2**-11` — an exact power of two, so the perturbation itself
  rounds nothing — with `rtol=2e-2` and `atol=2e-3`, and a **negative
  control** proves the band rejects a deliberately wrong gradient.
- **Public construction did not move one inch.** Two private hatches carry
  the milestone: `NativeTensorCore._typed_full` and a keyword-only
  `_trusted_dtype` on `NativeTensorCore.zeros` (the hatch
  `NativeStorage.__init__` has carried since I2, one layer up, and for the
  same reason — the zeroed allocation must reach *the* constructor rather
  than growing a second one). `full` is now `_typed_full` behind the
  unchanged public gate; `zeros`'s default keeps every public caller
  validated by `normalize_dtype`. Keeping `zeros` as the spelling preserved
  the H1 audit's source pins and spy seams **verbatim**.
- **The native CTest inventory moved 20 → 21** with
  `test_dtype_reduction_matmul`, which drives every axis form and layout
  family at both dtypes on both traversals, carries the accumulation
  witness, proves the scalar narrowing, and proves mixed dtype is refused in
  every participating handle position with the destination unmoved. It is
  complementary to `test_sum_reduction` and `test_matmul`, whose float64
  sweeps are untouched.

**What I4 did not change:** the export count (**54**), `SUPPORTED_DTYPES`
(`("float64",)`), `UNSUPPORTED` (`("float32", "cuda", "amp")`),
`RAW_KERNEL_DTYPES` (`("float64",)`), the checkpoint format (version **2**,
versions **(1, 2)** accepted), any public API, any float64 result, and any
Phase-H traversal, predicate, or allocation policy. Convolution, pooling,
classification, Dropout, normalization, optimizers, modules, and checkpoints
all still reject a float32 handle before touching memory, and no public
constructor produces a float32 tensor — so float32 parameters, modules,
optimizers, and training still do not exist.

**Recorded as reconciliation rather than as an I4 change:** two
`test_native_elementwise_traversal` assertions — the rank-five declined-plan
parity check for `add` and `multiply` — asserted NaN-payload identity in a
case the committed H8 contract (part 4) explicitly places *outside* the
bitwise contract. They pass under MSVC and under `g++ -O3`/`-O0`, and fail
under `g++ -O2` (the CI/no-CMake path) and `clang++ -O2`/`-O3`, which is
exactly the instruction-selection behaviour that contract describes. The
disagreement is **one** of 32 positions, at the single index where *both*
operands are NaN; both paths return a quiet NaN there, carrying one of the
two operands' payloads, and every other position is bit-identical. The test
was narrowed to assert those contractual properties instead of payload
identity, keeping bit equality everywhere at most one operand is NaN and
keeping `subtract` exact with nothing carved out. **No production code
changed for this**, and the correction reproduces against the clean
committed I3 source.

#### I3 — elementwise, broadcast, and unary dtype execution

**I3 made float32 storage computable, and added no C ABI symbol.** I2 left
float32 movable and computed on by nothing; I3 gives it exactly the
elementwise and unary arithmetic the later milestones build on, and nothing
more. What shipped:

- **Seventeen exports became dtype-general** — `tf_core_add`,
  `tf_core_subtract`, `tf_core_multiply`, `tf_core_relu`,
  `tf_core_relu_backward`, `tf_core_sqrt`, `tf_core_reciprocal`,
  `tf_core_exp`, `tf_core_log`, and the contiguous fast-path forms of each
  — every one of them the symbol Python already declared. Each keeps its
  argument list, calling convention, validation, traversal tiers, and
  ownership contract; what changed is that `tf::require_float64` ("this
  operation has not been generalized") became `tf::require_matching_dtype`
  ("it has been, and its operands must agree"), and that one `switch`
  selects the instantiation.
- **One narrow dispatch per exported call.** Four hidden helpers —
  `unary_by_dtype`, `unary_contiguous_by_dtype`, `binary_by_dtype`,
  `binary_contiguous_by_dtype` — hold the single `switch` their exports
  perform, above the traversals and below the exports. None has a
  `default:` label, so a future dtype without an instantiation is a
  compile-time problem rather than a silent misread. Below that point
  nothing branches on dtype: not the plan builder, not the row kernel, not
  the odometer carry, not the operation functor, and emphatically not any
  element.
- **All three Phase-H traversal tiers, instantiated twice from one
  source.** The contiguous flat row, H8's collapsed plan, and the retained
  generic odometer each gained a scalar type parameter, so `T = double` is
  the pre-I3 code statement for statement and float64 runs exactly what
  Phase H measured. The three tiers are proved to write **identical bits**
  at both widths, which is H8's parity claim restated per dtype.
- **The operation functors became the single source of every per-element
  expression.** Before I3 each expression existed twice — in the header for
  the templated traversal and in `elementwise.cpp` for the odometer's
  function pointer — kept identical by hand and by a comment. The odometers
  now take `&Op::apply<T>`, so there is one definition and drift is
  structurally impossible. Constants are written `T(...)`, which at
  `T = double` *is* the old literal and at `T = float` keeps the comparison
  and the division in binary32.
- **`exp` and `log` keep H8's exclusion, structurally.** They have **no
  functor** in the shared header — only file-local function templates — so
  nothing can hand them to a plan walk even by accident, and they dispatch
  straight into the retained odometer at either width. They select the
  `float` overload of `std::exp`/`std::log` for a float32 tensor rather
  than computing in binary64 and narrowing.
- **Broadcasting works at float32 for every layout it already worked at
  for float64**: zero strides, transposed and narrowed operands, negative
  strides, unit extents, multiple broadcast axes, and rank-0 scalars. No
  broadcasting feature was added; only the element type the zero-stride
  read produces changed.
- **Outputs preserve the operand dtype**, allocated through the private
  typed path I2 added. A float32 operation produces a float32 result and a
  float32 NumPy array on egress; nothing widens on the way out.
- **Mixed dtype is rejected before any allocation or mutation** — in the
  left operand, the right operand, and the destination position
  independently — at the Python layer, which knows the mismatch, and again
  in C++ at the trust boundary. A rejected call allocates nothing, writes
  nothing, and leaves both operands open and unchanged. The dtype guard
  runs **before** the span validation, so a call that is both mixed-dtype
  and out-of-span reports the dtype; with matching dtypes the pre-existing
  span error is still what surfaces.
- **The numerical contract, measured per operation rather than inherited.**
  `add`, `subtract`, `multiply`, `relu`, `relu_backward`, `sqrt`, and
  `reciprocal` are IEEE-specified and correctly rounded, so they are
  asserted **bit-identical** to the binary32 (and binary64) oracle,
  including both signed zeros and NaN classification. `exp` and `log` get a
  bound instead, for the reason the float64 contract uses one: within **1**
  representable step of a float32 reference rounded once from binary64, and
  within **2** of NumPy's own float32 kernel, which is itself about two
  steps from correctly rounded. Those are two different statements and are
  recorded separately rather than collapsed into the looser one.

  **I2's transfer contract is deliberately not reused here.** A transfer
  performs no arithmetic and so preserves a signalling NaN; an *operation*
  follows IEEE arithmetic and therefore quiets one. Both behaviours are
  asserted, as raw bit patterns, at both widths.
- **The "no hidden float64" claim is carried structurally, and the reason
  is recorded rather than glossed.** Every operation here produces each
  destination element with a *single* correctly-rounded IEEE operation, and
  binary64 carries more precision than a double rounding would need to
  differ — so computing in binary64 and rounding once to binary32 is
  *provably* indistinguishable for `+`, `-`, `*`, `/`, and `sqrt`. A search
  over 2,000,000 inputs, including the subnormal range where division could
  in principle double-round, found no witness, which is the theorem showing
  up rather than a gap in the search. The claim is therefore proved by the
  observable result being bit-identical to the binary32 oracle *and* by a
  semantic structural check over the source — templated functors with no
  `double` in them, traversals taking `T*`, no `static_cast<double>`
  anywhere in the translation unit, no double accumulator.
- **`tf_core_relu_backward` is dtype-general, and that is not float32
  autograd.** It is a forward-shaped numerical primitive over
  `(input, upstream)`, not graph machinery. No public constructor produces
  a float32 tensor, so no float32 graph, parameter, module, or optimizer
  can reach it — asserted directly rather than assumed.
- **The native CTest inventory moved 19 → 20** with
  `test_dtype_elementwise`, which drives thirteen view layouts × both
  dtypes × every operation, proves tier parity by bit pattern at each
  width, and proves mixed dtype is refused in all three operand positions
  with the destination byte-for-byte unmoved. It is complementary to
  `test_elementwise_traversal`, whose float64 NaN sweep is untouched.

**What I3 did not change:** the export count (**54**), `SUPPORTED_DTYPES`
(`("float64",)`), `UNSUPPORTED` (`("float32", "cuda", "amp")`),
`RAW_KERNEL_DTYPES` (`("float64",)`), the checkpoint format (version **2**,
versions **(1, 2)** accepted), any public API, any float64 result, and any
Phase-H traversal or predicate. Reductions, `mean`, matmul,
narrow-backward, convolution, pooling, classification, Dropout,
normalization, optimizers, modules, and checkpoints all still reject a
float32 handle before touching memory, and so do `tf_storage_fill` and
`tf_storage_scale`.

#### I2 — typed array transfer, views, and materialization

**I2 made float32 storage movable, and added no C ABI symbol.** I1 left
float32 allocatable and consumed by nothing; I2 gives it exactly the
transfer and materialization foundation the later kernel milestones need,
and nothing more. What shipped:

- **The three transfer exports became dtype-general**, through a
  **source-level retype** of their host positions:

  ```c
  void tf_storage_copy_from(void* handle, const void* src);
  void tf_storage_copy_to(const void* handle, void* dst);
  void tf_storage_materialize(const void* handle, void* dst,
                              const int64_t* shape, const int64_t* strides,
                              int64_t offset, int64_t ndim);
  ```

  This is a declaration change and **not an ABI change**: a `double*` and
  a `void*` occupy the same argument slot on every supported platform,
  `extern "C"` has no mangling to alter, and the symbol names, argument
  counts, argument order, and return types are untouched. A previously
  compiled caller would link and run identically, and the export inventory
  stayed at **54**. Three distinct things are kept apart deliberately:
  the C declarations changed, the binary symbol table did not, and the
  Python ctypes declarations changed to match the C.
- **The host pointer contract, stated rather than pretended.** The raw
  host pointer carries no dtype and the ABI cannot recover one from an
  address, so the **storage handle's immutable dtype tag is
  authoritative**: C++ dispatches from it, Python validates the NumPy
  dtype before the call, no implicit conversion happens at the boundary,
  no host dtype is guessed, and no byte-count or dtype argument was added.
  A direct foreign caller remains responsible for satisfying the contract,
  exactly as it already is for the layout arrays it passes.
- **`tf_core_contiguous_copy` became dtype-preserving and dtype-strict.**
  It is the runtime's value-transfer primitive (H5) and the one
  compute-shaped export the milestone touched. Source and destination
  dtypes must agree; a mixed pair is rejected with `TF_ERROR_INVALID`
  naming both, before the span validation and long before any element is
  written. Its three H5/H8 traversal tiers — the flat row-major sweep, the
  collapsed plan, and the retained generic odometer — are instantiated for
  both element types from the *same source*, so float64 runs the code
  Phase H measured, statement for statement.
- **Bit preservation, proved rather than assumed.** A transfer performs no
  arithmetic, so it has no operand roles to choose between and nothing to
  round: `+0.0`, `-0.0`, both infinities, subnormals, quiet NaNs with
  distinct payloads, and signalling NaNs of both signs all reproduce their
  source's object representation exactly, at both widths, through host
  ingress, host egress, strided materialization, chained views, and the
  identity copy. Asserted over seventeen IEEE-754 classes per dtype as raw
  `uint32`/`uint64` patterns in both the C++ and the Python suites — never
  by value, and never by tolerance.

  Scope, stated honestly: that is a measurement on the toolchains
  validated here (MSVC x64, GCC/Clang x86-64), not a language guarantee.
  C++ does not promise that copying a signalling NaN leaves it signalling,
  and an x87 code path would quiet it; TensorForge builds x86-64 with
  SSE2, where the copy is a register move.
- **`RAW_KERNEL_DTYPES == ("float64",)`**, introduced beside `RAW_KERNELS`
  and reported by `backend_info()` as `raw_kernel_dtypes`. The seven
  handle-free raw utility kernels receive only `const double*`, `double*`,
  and an element count — no handle, therefore no dtype tag, therefore
  nothing to dispatch on — so they stay float64 permanently. It is
  reported separately from `supported_dtypes` because it is a separate
  fact: this one is a permanent property of seven kernels, the other is a
  public promise that moves at I9.
- **A private, narrowly scoped internal construction path**:
  `NativeStorage._typed`, `NativeStorage._typed_from_array`,
  `NativeTensorCore._typed_from_array`, and a keyword-only
  `_trusted_dtype` on `NativeStorage.__init__` that validates against the
  internal dtype table instead of the public registry. No public
  constructor gained anything: `normalize_dtype("float32")` still raises,
  and `NativeStorage`, `NativeTensorCore.zeros`, `.full`, and
  `.from_array` all still reject `"float32"` with the message they always
  produced.

**No public capability moved.** `SUPPORTED_DTYPES` still reads
`("float64",)`, `SUPPORTED_DEVICES` still reads `("cpu",)`, `UNSUPPORTED`
still reads `("float32", "cuda", "amp")`, the native checkpoint format is
still version **2** with versions **(1, 2)** accepted, and the library
still exports **54** symbols. No module, parameter, optimizer, or loss
constructor gained a dtype argument. `tf_storage_fill` and
`tf_storage_scale` were deliberately **not** broadened — they assign and
multiply rather than transfer, and a scalar narrowed into a float32 buffer
is a decision with its own numerical statement that belongs to a later
milestone. No example, benchmark, CI workflow, dependency, or build option
changed.

**One rule was followed against the obvious shortcut.** A byte copy is the
easy way to make a contiguous transfer representation-preserving, and the
contract forbids `memcpy` outside the allocation boundary — that rule is
what lets the whole element-measured layout and bounds-checking apparatus
carry over unexamined. The transfers are same-type element assignments
instead, and the bit-preservation claim is a test result rather than an
appeal to `memcpy`'s semantics.

The native CTest inventory moved **18 → 19** with `test_typed_transfer`,
which compiles `storage.cpp` and `elementwise.cpp` directly so it reaches
the hidden `tf::copy_prefers_contiguous` predicate beside the four
generalized exports, and proves the contract by bit pattern at both dtypes
over thirteen view layouts — scalar, 1-D, non-unit stride, reversed, 2-D
contiguous, transposed, narrowed-with-offset, both broadcast (stride-0)
forms, unit extent, and two rank-3 chains.

#### I1 — internal dtype model and dtype-tagged storage foundation

**I1 replaced the physically `double`-only storage foundation with a
dtype-tagged one, and did nothing else.** What shipped:

- **The C++ dtype model** (`cpp/include/tf_internal.h`): the `TfDtype` ABI
  enum with frozen codes `0 = float64` and `1 = float32`; `tf::Dtype` as
  the fixed-width internal form; `dtype_from_code` (total, `noexcept`,
  allocation-free, rejecting every unknown code without writing to its
  output); `dtype_item_size` as the **single** place an element width is
  written down; `dtype_name` for messages; and `dtype_checked_bytes` as
  the one element-to-byte conversion in the runtime. The platform
  assumptions — `sizeof(double) == 8`, `sizeof(float) == 4`, and IEEE-754
  conformance for both — are `static_assert`ed rather than assumed.
- **Dtype-tagged storage**: `{void* data; int64_t size; Dtype dtype;}`.
  One untyped owned pointer (a union or second typed pointer could
  disagree with the tag; a `void*` cannot), a logical element count whose
  meaning and name did not move, and one dtype tag assigned once before
  the handle is published. The storage owns a **genuine runtime-selected
  `float[]` or `double[]` array**, created by an ordinary array
  new-expression behind one dtype dispatch into a templated body, held
  across the metadata allocation by `std::unique_ptr<T[]>`, and
  type-erased into `void*` only after the array exists. That is what makes
  the kernels' `data[i]` and `data + i` valid C++17 pointer arithmetic:
  they traverse one array object rather than a byte array or a run of
  separately constructed scalars, neither of which supports indexing
  across the allocation. Release goes through **one central
  dtype-matched `delete[]` switch**, so the allocation and deallocation
  forms cannot disagree and no call site duplicates the choice.
  Value-initialized arrays (`new T[n]()`) give every zeroed element
  **positive** zero at both widths, so H1's zero-initialized default is
  preserved with no per-dtype fill pass; default-initialized arrays
  (`new T[n]`) write nothing, so H1's uninitialized saving is preserved
  too. `float` and `double` are trivially destructible —
  `static_assert`ed beside the allocation — so no per-element destruction
  pass is needed.
- **Two new C ABI exports** — `tf_storage_create_typed` and
  `tf_storage_create_uninitialized_typed` — taking the library from
  **52 to 54** production `tf_*` symbols. These are the only two the whole
  phase adds. `tf_storage_create` and `tf_storage_create_uninitialized`
  are **not** removed, renamed, deprecated, or behaviorally altered: they
  became thin float64 wrappers over the same shared body, and their
  first observable failure is still `size <= 0` with the identical
  message.
- **Safety for a capability that runs ahead of its support.** float32
  storage is now allocatable through the C ABI while float32 is still not
  a supported dtype, so every operation that has not been generalized —
  31 `tf_core_*` exports plus `fill`, `scale`, `copy_from`, `copy_to`, and
  `materialize` — rejects a float32 handle with `TF_ERROR_INVALID` before
  reading or writing anything. Without that check a 4-byte-per-element
  buffer walked through a `double*` would be overrun by exactly a factor
  of two. Every float64 kernel now reaches its buffer through one accessor
  pair (`tf::storage_f64`) and the rejection is one shared helper
  (`tf::require_float64`), so neither is scattered.

**No public capability moved.** `SUPPORTED_DTYPES` still reads
`("float64",)`, `SUPPORTED_DEVICES` still reads `("cpu",)`, `UNSUPPORTED`
still reads `("float32", "cuda", "amp")`, `normalize_dtype("float32")`
still raises, no public constructor accepts float32, and the native
checkpoint format is still `tensorforge.native_checkpoint` version **2**
with versions **(1, 2)** supported. No compute kernel was templated or
generalized; the internal kernels keep their `double*` signatures
unchanged. No example, benchmark, CI workflow, dependency, or build option
changed. The registry moves at **I9**.

**One new failure mode arrived, exactly as the contract predicted.** A
byte-count overflow is now `TF_ERROR_INVALID`, rejected by checked
arithmetic before any allocator is asked, where the implicit
`new double[count]` sizing it replaced could only discover the problem by
throwing. The C++ storage-allocation CTest was advanced to assert **both**
modes separately — an overflowing count and a representable-but-
unsatisfiable one — rather than loosened to accept either.

The native CTest inventory moved **17 → 18** with `test_dtype_storage`,
which links every kernel translation unit because the float32-rejection
proof has to cover each one's own validation front end.

#### I0 — repository reconciliation and the architecture contract

**I0 shipped three things and nothing else:** the architecture contract
`docs/native_dtype_float32_design.md`; the durable Phase-I contract
guardrails `tests/test_native_phase_i.py` (and the phase-ladder updates
in `tests/test_docs.py` that opening a phase always requires); and
documentation reconciliation across the status surfaces.

**No runtime behavior changed.** I0 changed no production Python module,
no C++ header or source, no C ABI symbol, no ctypes declaration, no
`NativeTensorCore` method, no `NativeTensor` operation, no module, no
loss, no metric, no optimizer, no export, no capability registry, no
dtype, no device, no build file, no CI workflow, no example, and no
benchmark. `SUPPORTED_DTYPES` still reads `("float64",)`,
`SUPPORTED_DEVICES` still reads `("cpu",)`, `UNSUPPORTED` still reads
`("float32", "cuda", "amp")`, the library still exports exactly **52**
production `tf_*` symbols, and the native checkpoint format is still
`tensorforge.native_checkpoint` version **2** with versions **(1, 2)**
supported.

The contract opens by recording the *verified* state of the
implementation rather than assuming it, and that record is what makes the
rest of the phase small. The load-bearing finding: **42 of the 52
exports already address their operands through opaque handles**, and of
the ten that touch a raw buffer, three carry a handle beside it and seven
are handle-free float64 reference kernels no training path calls. So a
dtype tag placed on the storage behind those handles reaches essentially
the whole compute surface, and the only thing the ABI genuinely lacks is
a way to say which dtype a *newly constructed* buffer has. Hence
**exactly two** planned new symbols for the entire phase —
`tf_storage_create_typed` and `tf_storage_create_uninitialized_typed`,
52 → **54** — and an explicit, argued rejection of per-operation float32
exports, which would roughly double the ABI, relocate the dtype from the
data to the call site, and break the one-unchanged-export dispatch shape
every optimized path in the repository already follows.

What else the contract locks: an internal dtype model with frozen ABI
codes (`0 = float64`, `1 = float32`), one central item-size authority,
and no public Python dtype object; dtype-tagged storage with an untyped
data pointer, an unchanged logical element count, checked
`numel × itemsize` byte arithmetic, and one allocation form matched by
one deallocation form — with shapes, strides, offsets, and spans staying
in **logical elements** and bytes appearing only at the allocation
boundary; storage as the **single** dtype authority, so every view over a
buffer agrees by construction and no view operation casts or
reinterprets; one narrow dispatch per exported call into templated
`float`/`double` kernels, with nothing below it branching on dtype and
every retained generic reference path instantiated for both; **no
casting, no promotion, and no mixed-dtype arithmetic**, rejected at every
named layer before any output is allocated or any state is mutated;
**float32 accumulating in float32**, with no hidden wider accumulator
anywhere, because that would be mixed precision; the autograd, module,
buffer, RNG/Dropout, and optimizer-state dtype invariants, over a
generator algorithm that does not change; dtype-aware checkpoint
**version 3**, designed but deliberately not implemented or activated,
with versions 1 and 2 defined as float64-only formats that are **never**
guessed to be float32; exact deterministic resume proved **separately**
for float32 and float64, and explicitly never as agreement between them;
the preservation of every Phase-H float64 optimization and of the
project's measurement discipline, with each dtype benchmarked on its own
and no timing assertion, committed number, or result file; the
cross-platform, sanitizer, and lifecycle gates; and the exclusion list —
CUDA, AMP, float16, bfloat16, integer, boolean, and complex tensors,
casting, promotion, device transfers, `map_location`, implicit dispatch,
environment-variable selection, data loaders, distributed execution,
BLAS, Eigen, oneDNN, OpenMP, SIMD, memory pools, and C++-managed
autograd.

Two decisions in the contract came directly out of reading the code and
would have been easy to get wrong later. The **MaxPool2d winner buffer
stays float64 at every tensor dtype**: winners are flat plane offsets
encoded as floating-point values, exact only while the plane fits the
mantissa, so following the tensor's dtype would silently cut the largest
poolable plane from `2**53` to `2**24` elements — a capability regression
disguised as a dtype feature. And the three host-transfer exports
(`tf_storage_copy_from`, `tf_storage_copy_to`,
`tf_storage_materialize`) need only a **source-level retype** of their
host-buffer parameter from `double*` to `void*`, taking the element type
from the handle beside it — binary-compatible, adding no symbol, and
recorded here rather than discovered mid-phase.

The ladder is **I0–I11**, each milestone with its entry condition, scope,
tests, documentation, invariants, exclusions, and exit gate. The
recommended structure survived the repository inspection **unchanged**;
three findings were absorbed into milestones rather than reshaping it,
and are recorded as such. The public support registry moves at **I9**,
after integrated float32 training and the exact float32 resume proof both
pass — the same discipline that kept `dropout` in `UNSUPPORTED` from G3
through G9 while the operation and the module both already existed.

### Phase H — native CPU performance and runtime efficiency (H0–H10, phase complete)

**Phase H is **complete**: milestones H0 through H10 have all landed, and it is the latest *completed* phase. H10 re-measured the whole phase against a reconstructed and verified H0 baseline (52 cases, **zero checksum mismatches** — every figure compares implementations that produced bit-identical results), resolved the acceleration gate as three documented rejections with measurements (SIMD, threading/OpenMP, BLAS), assessed `tf_core_narrow_backward` and the small-operation boundary floor and implemented neither, ran the full Release/Debug/Linux/sanitizer/lifecycle matrix, and closed the phase. **Every shipped training workload is 1.50×–3.89× faster than at H0**, matmul 4.71×, Conv2d kernels 2.59×–4.64×, reductions 3.78×–5.06×, with no allocation count or memory peak raised anywhere — and across the whole phase **no capability, dtype, device, registry value, public API, checkpoint field, or checkpoint version moved**, with exactly **one** C ABI symbol added (`tf_storage_create_uninitialized`, at H1): 51 → **52**.** This section records H0; the
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
closure requirements; and an explicit adopt/adapt/reject decision for
every candidate performance design the phase considered.

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

### Phase H — native CPU performance and runtime efficiency (H9)

**Milestone H9 — native Conv2d execution efficiency — has since shipped**,
the fifth Phase-H milestone to change C++ and, like H2, H5, H6, and H8,
**not the ABI**: the library still exports exactly **52** `tf_*` symbols.

**H9 was not in H0's ladder.** H0 pencilled that slot in for SIMD,
threading, or optional BLAS, *conditional and presumed rejected*. It was
not entered — none of the three qualified, and a larger, safer result was
available in the same slot. H6 made reductions 3.2×–10.9× faster and left
every training step neutral; H7 moved every training step but the CNN
step's share came from its many small calls; H8 moved the normalization
modules 1.21×–1.40× and the **CNN step stayed neutral at 0.99×** — because
a convolution step's time is in `tf_core_conv2d_*`, which was still the
unmodified Phase-D direct loop from D2–D5 while matmul, copy, reduction,
and elementwise had each been revisited. Convolution was the last large
compute family running its original correctness-first implementation, and
it had become the majority of the one workload Phase H had never moved.
The acceleration decision moved to H10's decision gate, where it is a
decision rather than an implementation.

The cost was **decomposed rather than assumed**, and the answer was H6's
rather than H3's: timing the complete Core wrapper against the bare
foreign call showed the Python wrapper is a fixed ≈ 8–12 µs, **66 %** of a
toy `(4,1,6,6)` convolution but **0.2 %** at `(8,3,16,16)` and
**≈ 0 %** at `(16,8,32,32)`. For any convolution with real work the
compiled traversal is essentially **100 %** of the cost, so the C++ loop
was the only target worth having. The composed **bias gradient was
measured and found immaterial** — three chained `sum` reductions H6
already made 3.9×–4.1× faster, producing an `(O,)` result beside two
full-tensor gradients — so H9 changes nothing about it, and that is a
recorded negative result rather than an oversight.

All three pre-H9 kernels shared one shape: `n, o, i, j` outer, `c, p, q`
inner, with the padded source coordinate recomputed and bounds-tested in
the inner loops. That makes the innermost loop `kernel_width` — typically
**3** — iterations long, recomputes a row bound that depends only on
`(i, p)` once per input channel, and makes both gradients read-modify-write
destinations that far-apart iterations revisit. H9 reuses the dispatch
shape H2, H5, H6, and H8 each proved: **one hidden predicate, inside the
existing export, no new symbol, the pre-milestone traversal retained**.
`tf::conv2d_forward_generic`, `tf::conv2d_input_backward_generic`, and
`tf::conv2d_weight_backward_generic` are the **Phase-D direct loops
retained verbatim** as the shipped generic reference paths — reachable
through ordinary production dispatch and the oracle every optimized result
is compared against. Beside them: `tf::conv2d_forward_row_sweep`, whose
nest becomes `n, o, i | c, p, q | j` accumulating into a bias-primed
output row; `tf::conv2d_input_backward_gather`, which walks `grad_input`
rows and gathers instead of scattering; and
`tf::conv2d_weight_backward_gather`, which owns one destination at a time
and sums it in a register, writing it **once** instead of
`batch·out_h·out_w` times. Two file-local helpers compute the half-open
run of kernel taps whose source lies inside the real input — that run is
always contiguous, which is why solving for it and testing each candidate
skip the identical taps.

**The fast-path preconditions are one shared rule plus one
direction-specific one**: `min(input_width, output_width) >= 4` for all
three, and additionally **unit stride in both axes** for the input
gradient. The minimum is measured, not tuned — at a swept extent of 1 the
optimized forms ran **0.57×–0.93×**, at 2 they ran 1.04×–1.38×, and at 4
they ran **1.91×–2.40×**. `min(input_width, output_width)` is used because
it is the honest bound on all three inner loops, and keying the input
gradient on `input_width` alone **was measured wrong** — a 5-wide input
with a 1-wide output sweeps a single element and ran **0.73×**. The
predicates are total, pure, allocation-free, and functions of the integer
geometry alone — never a pointer value, an alignment, a clock, an
environment variable, or a CPU-feature probe — and **a false answer is a
fallback, never an error**. The input gradient alone needs unit stride
because its gather walks the kernel offsets *downward* to reproduce the
reference's ascending output order, and that inversion is one-for-one only
at unit stride; the forward and weight gradient take their optimized paths
at **every** stride. **The asymmetry is deliberate**, and the strided input
gradient's 1.04× is the row that proves the fallback is really taken.

**Per-destination accumulation order is preserved exactly in all three
directions**, each with its own proof: the forward's `c, p, q` stay outer
to `j`, so a destination still receives the same seed and the same taps in
the same order; the input gradient's ascending-`o`, descending-`p`,
descending-`q` walk *is* ascending `o`, `i`, `j` at unit stride; and the
weight gradient's `n, i, j` nest is exactly the order its destination's
contributions already arrived in. Nothing is reassociated, no partial sums
are combined, no accumulator width changes, and there is no FMA,
fast-math, tree or pairwise reduction, parallel accumulation, SIMD
intrinsic, or threading anywhere.

**The numerical contract is H9's own, measured against a pre-H9 library
built from identical sources with only `conv2d.cpp` restored.**
Contractual: (1) **every non-NaN result is bit-identical** — 256 ordered
pairs of 16 IEEE-754 representatives × 3 directions, **zero non-NaN
differences**, signed zeros, ±∞, denormals, the smallest normal, and the
largest finite magnitudes included; (2) **NaN positions are identical** in
all 768 comparisons; (3) **with at most one NaN reaching a destination the
paths agree exactly, payload included** — 480 single-NaN configurations
across five geometries, zero differences; (4) **signed zeros are
bit-identical** across 80 sign-pattern configurations, with `−0.0`
surviving only while every addend is `−0.0` and one `+0.0` making the sum
`+0.0` — both halves asserted on both paths, because the sweep replaces a
register accumulator with accumulate-into-memory and the weight gather
does the reverse, exactly the rewrites that could change a zero's sign;
(5) signalling NaNs are quieted identically and every NaN either path
produces is quiet. **Not contractual**: when two or more NaNs reach one
destination the surviving payload may differ (20/256, 20/256, and 29/256
pairs), asserted in neither direction — the same qualification H2 and H6
recorded, for the same instruction-selection reason, but **measured here
rather than assumed from them**.

**H1's contract holds on all three destinations**, for a different reason
each: the forward primes every element of every output row with the bias
before accumulating; the input gradient's gather zeroes each row it visits
and visits every row; and the weight gradient's gather *assigns* every
destination from a register, so it needs no zero-fill and never reads the
destination at all. Proved by poison in two places — the C++ suite
pre-fills each optimized destination with a quiet NaN and a large finite
value across the whole geometry matrix, and the Python suite injects the
same two patterns through the real private allocation seam for both the
optimized and the fallback geometries — each with a **negative control**
proving the detector can fail.

**Layout handling is untouched**: the C ABI is contiguous-only by Policy B
and the Core layer materializes any non-contiguous operand into a private
copy, so **H9 is a geometry optimization, not a layout one** and broadened
layout support by nothing. Autograd is untouched — same parent topology,
same conditional version tracking, gradients created only for parents that
require them, and the same `retain_graph`, repeated-backward,
accumulation, and cleanup behaviour. Nothing became in-place.

**Memory did not move, and that is asserted rather than assumed**: the
same workload on both libraries reports **byte-identical** allocation
counts, peak live storages, and peak bytes — a forward is 1 allocation /
921,600 peak bytes, an input gradient 1 / 524,288, a weight gradient
1 / 9,216, a `NativeConv2d` forward+backward 8 / 3,020,160, and a CNN
training step 94 allocations / 34 peak live / 604,848 peak bytes. There is
**no scratch buffer, workspace, arena, pool, padded copy, or im2col
allocation anywhere**.

Measured against the pre-H9 library on identical `ctypes` calls, every
case **bit-identical before either side was timed**, 11 alternating rounds,
with the two fallback geometries as the identical-code control at
**1.00×–1.13×**: k1×1 `(8,16,32,32)` **6.23× / 8.37× / 5.40×** (forward /
input / weight), `(8,16,32,32) → 32` k3×3 3.64× / 5.04× / 2.60×,
`(16,8,32,32) → 16` 3.32× / 5.28× / 2.47×, padded `(8,8,32,32)` 2.91× /
4.97× / 2.54×, `(8,3,16,16) → 8` 2.87× / 3.75× / 2.45×, prime extents
2.61× / 4.57× / 2.77×, **stride 2** 2.41× / *1.04× (falls back)* / 2.41×,
rectangular k3×5 2.30× / 3.76× / 2.22×, k5×5 1.97× / 3.84× / 2.09×. End to
end over 9 alternating **subprocess** rounds, all 23 checksums identical
first: `NativeConv2d` forward+backward **3.13×** padded and **3.09×**
unpadded, forward **2.98×**, no-bias 2.46×, frozen 2.40×, stride-2 2.28×,
and — the result that matters — **a CNN training step 1.86×** at
`(8,3,32,32) → 16`, **1.38×** at `(8,3,16,16) → 8`, 1.27× with Dropout,
**1.13×** at the shipped example's shape, and 1.11× with BatchNorm2d.
**This is the first Phase-H milestone to move a CNN training step**, which
H6 and H8 both measured as neutral.

Reported just as honestly: a **small convolution is neutral** (1.06×
forward, 1.20× forward+backward at `(4,1,8,8) → 4`), because below roughly
a thousand output elements the fixed ≈ 10 µs Python-plus-ctypes cost
dominates — H3's, H5's, H6's, and H8's documented boundary finding,
unchanged; the **BatchNorm2d and shipped-example CNN steps move least**
(1.11×, 1.13×), because convolution is a smaller share of those steps; and
**no control regressed** — at 21 alternating rounds matmul 256³ **0.98×**,
the MLP training step **0.97×**, `contiguous_copy` 512² 1.01×, reduction
1.07×, broadcast elementwise 1.07×, a control band of **0.97×–1.07×**. One
methodology finding is published rather than buried: at 9 rounds the
elementwise control read **0.93×** and looked like a regression, while at
21 rounds the same case read **1.07×** — the lesson H3, H5, and H6 each
recorded, so no low-round figure is quoted as H9 evidence.

**Four candidates were rejected with reasons**: **im2col + matmul**
(changes the accumulation order, which is the whole contract, and would
allocate 8× the input at the profile shape); a **materialized padded
input** (moves cost rather than removing it, when the tap-range helpers
give the same branch-free inner loop with no allocation); **output-channel
blocking** (the sweep already produces a long unit-stride inner loop, and
blocking `o` would reintroduce a tuning constant for a second-order gain);
and a **third "hoisted" path for small extents** (measured 1.5×–3.7× and
never regressing, but it would have left the Phase-D reference unreachable
for shapes that cost microseconds).

Validation: Windows **Release and Debug**, both out-of-source with the
Debug library written outside the repository so the active runtime stayed
the Release DLL, **17/17 CTests each** with zero project compiler, linker,
and CMake warnings; Clang 18.1.3 ASan/UBSan with **instrumentation
proved** — 22 `__asan*` and 15 `__ubsan*` dynamic symbols beside the
**52** exported `tf_*` symbols, independently confirming the export count
on a second toolchain — **17/17 sanitized CTests**, **445 sanitized
convolution/CNN/Phase-D/H1 tests**, the full sanitized native suite with
**zero ASan and zero UBSan diagnostics**, and both shipped CNN examples
reproducing their exact checkpoint resumes under it. A **sanitizer
negative control** makes that absence real: handing the row sweep an input
one row shorter than its declared geometry produces a
`heap-buffer-overflow`, `READ of size 8`, inside
`conv2d_forward_row_sweep`. A LeakSanitizer lifecycle returns native live
storage **exactly to baseline (0)** at every checkpoint — core forward and
both gradients over optimized *and* fallback geometries, module cycles,
seven injected-failure cycles, abandoned graphs, and two complete CNN
training runs — with the remaining process-exit allocations containing
**no TensorForge frame** and no suppression file added.

The harness gained **three** cases, 38 to **41** —
`conv2d_forward_padded`, `conv2d_forward_strided`, and
`conv2d_forward_fallback`, following the separate-rather-than-average
precedent, because unlike H5/H6/H8 the chooser here is the *geometry*
rather than the layout; all three are `native_only` and publish **no
ratio**, and the fallback case is the family's control since its compiled
path did not change. Native CTests 16 to **17**. **No exported C ABI
symbol, no new translation unit, and no public control of any kind** — no
path selector, block-size setter, traversal control, dispatch tracer,
benchmark hook, profiling counter, environment variable, or "which path
ran" query — and no SIMD, threading, OpenMP, BLAS, oneDNN, Eigen, memory
pool, scratch workspace, im2col, or fast-math. No convolution option was
added: no dilation, no groups, no channels-last, no new padding mode. No
public API, capability, dtype, device, registry value, checkpoint field,
or checkpoint version moved.

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
