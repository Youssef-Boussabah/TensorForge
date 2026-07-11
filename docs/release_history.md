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
combined with SGD.
