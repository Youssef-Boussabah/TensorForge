# Native autograd — design

This is a **design document** (written as the Advanced C++ v2.0
milestone, ahead of any code) for reverse-mode autograd over the native
runtime — `NativeTensor` / `NativeTensorCore`. It opens **Phase B** of the
Daedalus-class roadmap, following the completed **Phase A — native CPU
runtime** (contiguous fast path, broadcasting, reductions, dtype/device
metadata; §1).

> **Implementation status.** v2.0 was design only. **v2.1 implements the
> autograd metadata skeleton and the reverse-topological backward
> driver** on `NativeTensor` — `requires_grad`, `grad`, `is_leaf`,
> `zero_grad()`, `detach()`, `backward()`, the internal graph constructor
> `_from_op`, native `NativeTensor`-backed gradient accumulation
> (`_accumulate_grad`), and the seed/validation rules of §7 — following
> this design. **v2.2 wires the core differentiable operations into that
> engine**: `add`, `subtract`, `multiply`, `relu`, `sum`, `mean`,
> `matmul`, `reshape`, `transpose`/`T`, and `contiguous_copy` build graph
> nodes when an operand requires grad, with broadcasting backward handled
> by the `unbroadcast` reduction helper of §8 and sum/mean's
> broadcast-back rule of §7.4 — landing the v2.2–v2.4 op scope of §16 in
> one milestone, plus `subtract` via the §7.5 scalar-multiply negation
> (no negate kernel) and `mean`'s 1/count as a broadcast-scalar multiply.
> The one new kernel is the fused `tf_core_relu_backward` §7.4
> recommended (a reuse of the generic binary odometer walker). All
> backward math runs on native forward kernels at the `NativeTensorCore`
> level — the core and the C++ kernels remain autograd-unaware and own no
> graph state. `contiguous_copy` backward is the identity (the forward is
> an elementwise logical copy, and a gradient lives at the logical shape,
> so §9's inverse-layout concern does not arise). **v2.3 completes the
> view-backward set with `narrow` backward** (§9): a graph node when the
> parent requires grad, whose backward **scatters** the upstream gradient
> into a fresh zeros tensor of the parent's shape at the narrowed region
> via the one new C++ kernel `tf_core_narrow_backward` — the odometer dual
> of `tf_core_sum` (a sum folds many inputs into one cell through zero
> write-strides; a narrow-backward writes each input into its own cell at
> the parent's full row-major strides from a `start`-shifted base offset).
> Because the gradient lives at the logical shape, the scatter output is
> always fresh owning row-major contiguous storage, so transposed,
> narrowed, and nonzero-offset parents all work (each is a preceding node
> whose own backward handles its layout). **v2.4 implements the graph
> lifetime policy** (§7.1): `backward(gradient=None, retain_graph=False)`
> is one-shot by default (a successful pass releases the traversed
> operation graph — parents/backward cleared, node marked freed — and a
> later backward reaching it raises a clear error rather than silently
> truncating history), `retain_graph=True` keeps the graph for another
> pass, leaf gradients accumulate across passes until `zero_grad()`,
> `retain_graph` is validated as a real bool before any mutation, and a
> failed pass rolls back with no partial commit or partial free. It is not
> full PyTorch parity (no per-node `retain_grad`, no double-backward). No
> new C++ kernel, no NumPy in the gradient path, and `NativeTensorCore`
> still owns no graph state. Verified against exact analytical values, an
> independent NumPy zero-padding reference, and finite differences;
> demonstrated by `examples/native_autograd_demo.py`. **v2.5 is a
> measurement milestone** — it changes no autograd behavior and adds a
> characterization benchmark (`benchmarks/benchmark_native_autograd.py`)
> that separates forward-native, forward+graph-construction,
> fresh-forward+backward, and repeated retained-backward cost, with honest
> hardware-specific results and no speed assertions (see
> [native_autograd_benchmarks.md](native_autograd_benchmarks.md)). **v2.6 is
> the Phase B completion milestone** — it adds no operation, kernel, or
> optimization, and changes no autograd behavior. It audits and locks the
> completed engine with cross-cutting guardrail tests
> (`tests/test_native_autograd_guardrails.py`): a runtime NumPy-no-fallback
> guard over representative backward passes, `NativeTensor` ↔
> `tensorforge.Tensor` isolation, explicit-backend / no-implicit-dispatch
> behavior, gradient-ownership and graph-lifetime invariants over realistic
> mixed graphs, detach and view+offset invariants, closed-operand failure
> safety, the raw-buffer kernel-registry boundary, and the v2.5 benchmark
> mode contract. It records the **final Phase B support matrix** (§17), makes
> the explicit **divide-backward decision** (deferred beyond Phase B, §18),
> and **marks Phase B complete** with **Phase C — the native training stack —
> next**, opening at **Advanced C++ v3.1 — NativeParameter and Parameter
> Registration Contract** (§19). The
> sections below remain the design of record.

For where this sits, see [backend_experiments.md](backend_experiments.md)
(the native runtime and its benchmarks),
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md) (the
wrapper this graph attaches to), the Phase A designs it builds on —
[native_broadcasting_design.md](native_broadcasting_design.md) (whose
backward is a reduction, §8),
[native_reductions_design.md](native_reductions_design.md) (the reduction
`unbroadcast` needs, §8), and
[native_dtype_device_metadata_design.md](native_dtype_device_metadata_design.md)
(the `grad.dtype == tensor.dtype` contract this reads, §10) — and
[autograd.md](autograd.md) (the Python `Tensor` autograd this deliberately
does **not** touch, §11).

## 1. Why native autograd is Phase B

Phase A built a native CPU runtime that can lay out, view, broadcast,
combine, and reduce tensors, and label them with dtype/device — but every
one of those pieces is **forward-only**. A `NativeTensorCore.add` computes
`a + b` and returns a fresh tensor that remembers *nothing* about where it
came from. There is no way to ask "what were the inputs to this result"
and no way to push a gradient back to them. Training needs exactly that,
so autograd is the next layer.

It is deliberately **after** all of Phase A, and each Phase A piece is a
genuine prerequisite:

- **Reductions (A3) are required for broadcasting backward.** Broadcasting
  reads one element into many positions through zero strides; its backward
  must do the opposite — **sum the gradient over the broadcast axes** —
  which is a reduction. Without native `sum` there is no native backward
  for any broadcasting op (§8). The reductions design records this as the
  reason A3 precedes autograd.
- **dtype/device metadata (A4) is the contract a backward reads.** A
  gradient must match its tensor's dtype and live on the same device; a
  backward that mixes float64/float32 or CPU/CUDA is a bug. A4 made
  `dtype`/`device` real, inspectable, enforced fields, specifically so a
  backward can assert `grad.dtype == tensor.dtype` and
  `grad.device == tensor.device` against something that actually exists
  (§10).
- **Views, matmul, and elementwise ops (A1–A2, v1.0–v1.3) are the forward
  primitives** each backward rule is written against. You cannot design
  `matmul` backward before `matmul` exists; you cannot design `reshape`
  backward before `reshape` exists. They all do.

So Phase A was the forward runtime; Phase B teaches it to differentiate.
Nothing in Phase B needs a new *forward* kernel for its first scope (§7,
§12) — the backward rules are expressed as compositions of forward
primitives the runtime already has, which is exactly why the phase order
is A-then-B.

## 2. Current limitation

The native runtime is **forward-only, with no operation graph**:

- **`NativeTensorCore` records no provenance.** `add`/`subtract`/
  `multiply`/`matmul`/`relu`/`sum`/`mean` each allocate a fresh
  contiguous result and return it; the result holds storage + view
  metadata + dtype/device, but **no parents, no operation name, no
  backward closure.** The graph edge from result to inputs is simply not
  stored.
- **`NativeTensor` has no autograd surface.** It exposes `shape`,
  `strides`, `ndim`, `numel`, `contiguous`, `dtype`, `device`,
  `to_numpy`, the compute/view methods, and the lifetime story — but **no
  `requires_grad`, no `grad`, no `backward()`, no `detach()`,** by design
  (v1.8–v1.21 kept it a thin forward-only wrapper, asserted by tests that
  no autograd attribute exists).
- **There is no traversal machinery.** No topological sort, no gradient
  seeding, no accumulation — none of what
  [autograd.md](autograd.md) describes for the Python `Tensor` has a
  native analog yet.

The limitation is not that forward-only is wrong — it was the right shape
for Phase A. It is that a native tensor cannot yet be *trained*, because
it cannot compute a gradient.

## 3. Goal

Add a **native autograd prototype** over `NativeTensor` /
`NativeTensorCore` — reverse-mode, define-by-run, Python-managed — that
mirrors the shape of the Python `Tensor` engine
([autograd.md](autograd.md)) but stays entirely inside the native
experimental line:

- `NativeTensor` gains an **opt-in** autograd surface: `requires_grad`,
  `grad`, `backward()`, `zero_grad()`, `detach()`.
- Differentiable operations **record a graph** (parents + backward rule +
  op name) above the raw forward runtime.
- `backward()` walks that graph in **reverse topological order**,
  accumulating gradients into the leaf tensors that asked for them.
- Gradients are themselves **native** (`NativeTensor`-backed, §4), match
  their tensor's dtype/device (§10), and are honest — verified against
  finite differences where practical (§14).

The prototype stays small, staged (§6), and separate from
`tensorforge.Tensor` (§11).

## 4. Non-goals

Stated up front so the milestone stays tight:

- **Do not replace the Python `Tensor`/autograd engine.** The NumPy-backed
  `Tensor` in `src/tensorforge/tensor.py` remains the reference frontend
  for everything the framework ships — modules, optimizers, losses,
  checkpoints. Native autograd is an experimental parallel track, not a
  swap-in (§11).
- **No implicit backend dispatch.** Nothing routes a `tensorforge.Tensor`
  op to the native engine or vice versa. The governing "no silent
  switching / no silent NumPy fallback" rule from
  [dispatch_design.md](dispatch_design.md) holds unchanged.
- **No CUDA autograd.** CPU float64 only, like the runtime beneath it.
  `device` has one legal value (`"cpu"`); a device backward is a future
  phase.
- **No optimizer / `Module` / `Parameter` integration.** `SGD`/`Adam`
  operate on `Parameter`/NumPy state and never see a `NativeTensor`. A
  native training stack is **Phase C**, after this.
- **No operator overloads.** Compute stays method-only (`a.add(b)`, not
  `a + b`), matching the forward wrapper.
- **No in-place / mutation semantics.** Autograd assumes values are not
  mutated after they enter the graph (§13).
- **No new *forward* numeric kernels in first scope.** Backward rules are
  compositions of existing forward primitives; any genuinely missing
  kernel (negation, scalar-multiply, compare/where, divide) is called out
  honestly (§7) and deferred, not silently required.

## 5. Graph model

The design's central decision: **the autograd graph lives at the
`NativeTensor` layer (a Python-managed graph), not inside
`NativeTensorCore` and never inside the C++ kernels.**

### 5.1 Layer split

- **`NativeTensorCore` stays the raw forward runtime**, unchanged. It owns
  storage + view metadata + dtype/device and runs forward kernels. It
  learns *nothing* about gradients. This keeps the C++/native boundary
  simple and keeps every Phase A guarantee intact (§9, §12).
- **`NativeTensor` carries the optional autograd metadata** above the
  core. A differentiable op produces a `NativeTensor` whose fields are:
  - `core` — the forward result `NativeTensorCore` (as today),
  - `requires_grad` — whether this tensor participates in autograd,
  - `grad` — the accumulated gradient (a `NativeTensor`, or `None`; §4-storage),
  - `_parents` — the input `NativeTensor`s this result was computed from,
  - `_backward` — a closure that, given this tensor's upstream gradient,
    pushes the local gradient to each parent (the local rule, §7),
  - `_op` — a short operation name for debugging/`repr` (e.g. `"add"`,
    `"matmul"`).
- **A small `native_autograd` helper module** may host the graph
  machinery (topological sort, the `backward` driver, `unbroadcast`, the
  per-op backward closures) so `native_tensor.py` stays readable — the
  exact split (methods on `NativeTensor` vs. helpers in a sibling module)
  is an implementation detail deferred to v2.1/v2.2, but the graph
  **never** descends into the core or the kernels.

This mirrors the Python `Tensor` exactly: there, ops build the graph and
`_backward` closures hold local derivatives, while NumPy stays the dumb
forward executor. Here, `NativeTensor` ops build the graph and
`NativeTensorCore` stays the dumb forward executor. The parallel is
deliberate — it makes the native engine legible to anyone who has read
[autograd.md](autograd.md).

### 5.2 Leaves vs. non-leaves

- **Leaf tensors** are user-created (`from_array`/`zeros`/`full`) with
  `requires_grad=True`. They have no parents and no `_backward`;
  `backward()` **accumulates into their `grad`**. These are what a future
  optimizer would update.
- **Non-leaf tensors** are operation results with
  `requires_grad=True` (because at least one parent required grad). They
  have parents and a `_backward` rule. By default their `grad` is a
  *transient* used during the backward pass and is not retained (matching
  PyTorch's "only leaf grads are kept" convention); retaining non-leaf
  grads is a possible later convenience, not first scope.
- **`requires_grad=False` tensors** (the default) carry no graph at all —
  no parents recorded, no `_backward`. This makes forward-only use exactly
  as cheap as today: if nothing requires grad, an op behaves like the
  current wrapper (§9).

### 5.3 Requires-grad propagation

An op's result `requires_grad` is the **OR of its differentiable inputs'**
`requires_grad`, exactly as the Python `Tensor` does
(`self.requires_grad or other.requires_grad`). If no input requires grad,
the result is a plain forward tensor with no graph. `detach()` (§5.4)
breaks propagation.

### 5.4 Detach

`tensor.detach()` returns a **new `NativeTensor` over the same forward
value** (sharing the core's storage as a borrowing view, or a fresh
owning copy — the ownership choice is settled in §5.5) with
`requires_grad=False`, **no parents, and no `_backward`.** It is the
explicit escape hatch from the graph: a detached tensor never tracks
history, so gradients do not flow through it. This matches the Python
engine's role for `requires_grad=False` intermediate values.

### 5.5 Lifetime and shared storage — the hard part

The graph holds references to parent `NativeTensor`s, whose cores own (or
borrow) C++ storage. This intersects the wrapper's existing
ownership/lifetime rules
([native_tensor_wrapper_design.md](native_tensor_wrapper_design.md), §4)
and needs care:

- **The graph keeps parents alive.** As long as a result tensor is
  reachable, its `_parents` are too, so their storage is not GC'd out from
  under a pending `backward()`. This is a feature (backward can read
  forward values it needs, §7) and a cost (memory is held until the graph
  is released).
- **`backward` must not run through a closed tensor.** If a user
  `close()`s a tensor that is still part of a live graph, a subsequent
  `backward()` that needs it must raise a clear `RuntimeError` (via the
  existing `_require_open()` gate), never read freed storage (§13).
- **Detach's ownership** must not create a use-after-free: a detached
  *view* over a parent's storage stays valid only while that storage is
  open. To keep the prototype safe on the Windows cleanup quirk
  (CLAUDE.md), the recommendation is that **`detach()` returns an owning
  copy** in first scope (a `contiguous_copy`-style materialization),
  trading a copy for a simpler lifetime — revisited if it proves costly.

## 6. Storage of gradients

- **`grad` is a `NativeTensor`, not a NumPy array.** The whole point of a
  native engine is to keep gradients in native storage; a NumPy `grad`
  would smuggle the compute back to the host and break the "no silent
  NumPy" rule. So `grad` is `None` (unset) or a `NativeTensor` over
  `NativeTensorCore` storage.
- **`grad` matches the tensor's dtype/device** (§10): `grad.dtype ==
  tensor.dtype`, `grad.device == tensor.device`, and `grad.shape ==
  tensor.shape`. This is exactly the contract A4 exists to provide.
- **Gradients are initialized lazily.** A leaf's `grad` starts `None` and
  is only allocated when the first contribution arrives during a backward
  pass (mirroring the Python `Tensor`, whose `grad` starts `None` and is
  filled by `_accumulate_grad`). No zero tensor is allocated for a tensor
  that never receives gradient.
- **Accumulation adds into the existing grad.** When a tensor contributes
  to multiple paths, each contribution is summed: `grad = grad + contrib`
  via the native `add` (a fresh result; §5's no-in-place rule means the
  old grad tensor is replaced, not mutated). The first contribution
  becomes the grad directly (no `0 + contrib`), so no needless allocation.
- **`zero_grad()` clears grad to `None`** (the recommendation), rather than
  zeroing it in place. Justification: (a) it matches the Python engine and
  the framework's optimizers, whose `zero_grad()` sets grads to `None`
  (see CLAUDE.md, `optim/`) — consistency with the reference frontend is
  worth more than micro-optimizing an allocation; (b) it sidesteps
  in-place mutation, which the no-in-place rule (§5, §13) bans elsewhere;
  (c) lazy re-init (above) means the next backward re-allocates cleanly.
  An in-place zero-fill variant (reusing a `tf_storage_fill`-style
  primitive) is a possible later optimization, not first scope.
- **Closed tensors/cores reject gradient operations.** Reading/writing
  `grad`, accumulating, or `backward()`-ing through a closed
  `NativeTensor` (or one whose core's storage was released) raises
  `RuntimeError` via `_require_open()` — never a silent no-op and never a
  read of freed memory (§13).

## 7. Backward traversal and gradient formulas

### 7.1 The `backward()` driver

`tensor.backward(gradient=None)` mirrors [autograd.md](autograd.md)'s
three steps, natively:

1. **Topologically sort** the graph reachable from `tensor` via
   `_parents`, so every tensor is visited after everything computed from
   it (a post-order DFS with a visited set, exactly like the Python
   engine's `build_topo`).
2. **Seed** the output gradient (§7.2).
3. Walk the sorted list **in reverse**, calling each non-leaf's
   `_backward(upstream)` to push gradient to its parents; each parent
   **accumulates** (§6). Leaves accumulate and stop (no `_backward`).

Reverse-topological order is what guarantees a tensor's gradient is
complete (all downstream contributions summed) before its own `_backward`
runs — the same reasoning as the Python engine.

**`retain_graph` was deferred through v2.3 and is implemented in v2.4.**
`backward(gradient=None, retain_graph=False)` is now **one-shot by
default**: after a successful pass the operation graph of every traversed
non-leaf node is released — its `_parents`/`_backward` cleared (so the
captured closures cannot keep parents alive) and the node marked freed —
and a later `backward()` reaching it raises a clear `RuntimeError` naming
`retain_graph=True` as the remedy (this catches both a repeated backward
on the same output and a new op built from a freed non-leaf value; the
freed node is *not* silently treated as a leaf, which would truncate
history). `retain_graph=True` keeps the graph for another pass. `retain_graph`
is validated as a real `bool` first — before traversal, callbacks, cleanup,
or any gradient mutation — and never coerced. It is **not** full PyTorch
parity: no per-node `retain_grad`, no double-backward. Whether or not the
graph is freed, only leaves retain grad (non-leaf grads are transient and
dropped after each pass, so a retained second pass recomputes them
cleanly), and repeated `backward()` calls **accumulate** into leaf grads
until `zero_grad()`. A genuine leaf has no graph to free and is never
marked freed. The whole pass is **failure-safe**: it is staged against a
snapshot of every node's gradient (gradients are immutable — accumulation
replaces the reference with a fresh native `add`), so if traversal or a
callback raises, the references are restored and neither a leaf gradient
nor the graph is left partially changed; cleanup runs only after the pass
fully succeeds. Traversal is by **object identity** (`id()`), not
`NativeTensor` hashing, so a node reached by several paths — or listed
twice as a parent — is visited exactly once while its backward rule still
accumulates once per logical edge.

### 7.2 Default seed gradient

- **Scalar output** (`shape ()`, `numel == 1`): `gradient` defaults to a
  native `1.0` (`NativeTensor.full((), 1.0)` matching dtype/device), since
  `d(out)/d(out) = 1`. This is the common `loss.backward()` case.
- **Non-scalar output**: an explicit `gradient` is **required**. Calling
  `backward()` on a non-scalar with no `gradient` raises a clear
  `ValueError` ("backward on a non-scalar requires an explicit gradient of
  shape …"). This matches PyTorch and avoids silently inventing a
  reduction the user did not ask for. (An alternative — implicitly seeding
  ones for any shape — is rejected as too magical and inconsistent with
  the Python `Tensor`, whose `backward` seeds ones but is used on scalar
  losses.)

### 7.3 Gradient-argument validation

An explicit `gradient` must be a `NativeTensor` with:

- **shape equal to the output shape** — else `ValueError` naming expected
  vs. actual;
- **matching dtype and device** — else `ValueError` naming both pairs,
  the same message style as the A4 op guards (§10).

A non-`NativeTensor` gradient raises `TypeError`.

### 7.4 First-scope gradient formulas

`u` is the upstream gradient flowing into the op's result. All rules are
expressed as **native forward ops** the runtime already has; missing
kernels are called out.

- **`add(a, b)`** → `da = u`, `db = u`. Followed by `unbroadcast` to each
  operand's shape (§8). Needs nothing new.
- **`multiply(a, b)`** → `da = u * b`, `db = u * a` (native `multiply`),
  then `unbroadcast`. Needs the forward values `a`, `b` — held live by the
  graph (§5.5). Needs nothing new.
- **`relu(x)`** → `dx = u * mask(x > 0)`. **Missing kernel:** the native
  runtime has no compare/`greater`/`where`, so there is no native way to
  build `mask(x > 0)` today. Options, in order of preference: (a) add a
  small `tf_core_relu_backward(u, x)` kernel — `u` where `x > 0` else `0`,
  a single fused pass mirroring `relu`'s forward; (b) add a general
  `greater`/`where` primitive later. **Recommendation:** the fused
  `relu_backward` kernel, introduced in **v2.2** with relu's backward,
  because it is the smallest honest addition and avoids a premature
  general compare/where surface. This is the one place first-scope
  backward needs a new native kernel, and the design says so plainly
  rather than pretending `relu` backward is free.
- **`sum(x, axis, keepdims)`** → `dx = broadcast(u, x.shape)`: the upstream
  gradient is broadcast back to the input shape (each summed element
  received an equal share of the output cell's gradient). Native
  broadcasting (A2) already does this via zero-stride reads; with
  `keepdims=False` the reduced axes must first be reinserted as size-1
  (a `reshape`) so the broadcast aligns. Needs nothing new.
- **`mean(x, axis, keepdims)`** → `dx = broadcast(u, x.shape) / count`,
  where `count` is the number of elements reduced (`numel` for
  `axis=None`, `shape[axis]` for one axis). **Missing kernel note:**
  dividing by a scalar `count` needs either a scalar-multiply by
  `1/count` — the runtime already has an internal `tf_storage_scale`
  (used by `mean`'s forward) — or a `multiply` against a broadcast scalar
  `NativeTensor.full((), 1/count)`. **Recommendation:** reuse the
  broadcast-scalar `multiply` path (no new kernel) or surface
  `tf_storage_scale`; either is honest. Deferred to **v2.3** with
  broadcasting backward.
- **`matmul(a, b)`** (2-D `(m,n)@(n,p)`) → `da = u.matmul(b.T)`,
  `db = a.T.matmul(u)`. Uses native `matmul` and `transpose` (both exist;
  `matmul` runs over strided views, so `b.T`/`a.T` need no
  materialization). Needs nothing new. Deferred to **v2.4**.

### 7.5 Subtract, negation, and divide — honest gaps

- **`subtract(a, b)`** → `da = u`, `db = -u`. The `-u` requires **unary
  negation**, which the native runtime does not have. Options: (a) a
  `negate`/`neg` kernel (or `tf_storage_scale` by `-1.0`); (b)
  `multiply` by a broadcast `-1.0` scalar (no new kernel). **Recommendation:**
  option (b) for first scope (reuse broadcasting + `multiply`), with a
  dedicated `neg` kernel as a later cleanliness improvement. Until then,
  `subtract` backward can be introduced alongside `multiply` in **v2.2**
  using the scalar-multiply trick, or deferred one step — the design flags
  the dependency rather than assuming negation exists.
- **`divide(a, b)`** → `da = u / b`, `db = -u * a / b**2`. Needs native
  **element-wise divide** in the backward (the forward `divide` exists as
  a raw-buffer kernel but is **not** a `NativeTensorCore` method yet) and
  negation. Divide backward is **out of first scope** and waits until a
  native `divide` core method and negation land.

This section is deliberately blunt about missing kernels — **negation,
scalar-multiply exposure, compare/`greater`/`where`, and a core-level
`divide`** — so the staged plan (§12) can schedule them honestly instead
of a backward rule silently assuming a primitive that isn't there.

## 8. Broadcasting backward — `unbroadcast`

Forward broadcasting (A2) expands a smaller operand's logical shape by
reading through zero strides; it never materializes the expansion. Its
backward must **reduce the gradient back down** to the operand's original
shape — the adjoint of a copy is a sum
([native_reductions_design.md](native_reductions_design.md), §8).

The design introduces a helper concept:

```
unbroadcast(grad, original_shape) -> NativeTensor
```

whose job, given a gradient at the (larger) broadcast/output shape, is to
return a gradient at `original_shape`. It must:

1. **Sum over leading axes** that broadcasting prepended (rank padding):
   while `grad.ndim > len(original_shape)`, `grad = grad.sum(axis=0)`.
2. **Sum over stretched axes** where the original dim was 1 but the output
   dim was larger: for each axis where `original_shape[axis] == 1` and
   `grad.shape[axis] != 1`, `grad = grad.sum(axis=axis, keepdims=True)`.
3. **Result shape equals `original_shape`** exactly.

This is the *native* mirror of the Python engine's `_unbroadcast`
([autograd.md](autograd.md), tensor.py `_unbroadcast`), and it is built
entirely from native **reductions (A3)** — which is the recorded reason
A3 had to precede autograd. It needs **no new kernel**: `sum(axis,
keepdims)` already exists.

`unbroadcast` is used by `add`/`subtract`/`multiply` backward (§7.4) so a
`(3,)` bias added to a `(2, 3)` matrix gets its gradient summed back to
`(3,)`. It is **implemented in v2.3**, not v2.0 — the v2.2 basic backward
starts with same-shape operands (no broadcasting), and v2.3 adds
`unbroadcast` and `mean` together.

## 9. View backward rules

`NativeTensor` has four view/copy ops; each has a well-defined backward
(the adjoint of a data movement is the reverse movement):

- **`reshape(new_shape)` backward** → `grad.reshape(original_shape)`. A
  reshape is a pure relabeling of a contiguous buffer, so its backward is
  the inverse reshape. Cheapest and safest view backward — **in v2.x first
  view scope.**
- **`transpose(perm)` / `T` backward** → apply the **inverse permutation**
  to `grad` (`grad.transpose(inverse_perm)`). For `T` (full reversal) the
  inverse is the same reversal. Native `transpose` exists — **in first
  view scope.**
- **`narrow(dim, start, length)` backward** → **scatter** `grad` into a
  zeros tensor of the original shape at the sliced region (the un-narrowed
  positions get zero gradient). **Missing kernel note:** the runtime has
  no scatter/pad primitive; this needs either a new `tf_core_scatter`-style
  kernel or an allocate-zeros-then-copy-into-a-narrowed-view step (which
  needs a native *copy-into-view* the runtime also lacks). So `narrow`
  backward was deferred at v2.2 and **implemented in v2.3** as
  `tf_core_narrow_backward` — the smallest such primitive (a focused
  scatter walking the narrowed shape, writing each upstream element to the
  parent's full row-major stride from a `start`-shifted base offset; the
  odometer dual of `tf_core_sum`). See the status note at the top.
- **`contiguous_copy()` backward** → pass `grad` back to the original view
  layout. A `contiguous_copy` materializes a (possibly strided) view into
  a fresh contiguous buffer; its backward must map the contiguous
  gradient back onto the source view's layout. For a contiguous source
  this is identity; for a strided source it needs the inverse of the
  materialization (again a copy-into-strided-view). **Deferred** with
  `narrow` for the same missing-primitive reason.

**Recommendation:** the **first autograd implementation supports only
`reshape` and `transpose` view backward** (both expressible with existing
ops and no new kernel), and **defers `narrow`/`contiguous_copy`** until a
native scatter/copy-into-view primitive is designed. This keeps v2.x
tight and avoids inventing a kernel under autograd pressure. View
lifetimes and shared storage follow §5.5 — a view's backward references
its parent, which the graph keeps alive. *(As implemented:
`contiguous_copy` backward landed in v2.2 as the identity — the forward
is an elementwise logical copy and gradients live at the logical shape,
so the strided-source concern above never arises — and `narrow` backward
landed in v2.3 via `tf_core_narrow_backward`, the focused scatter above,
completing the view-backward set.)*

## 10. dtype / device rules

The gradient inherits the A4 metadata contract exactly
([native_dtype_device_metadata_design.md](native_dtype_device_metadata_design.md)):

- **Gradients preserve dtype and device.** `grad.dtype == tensor.dtype`
  and `grad.device == tensor.device`, always. A backward that produced a
  mismatched grad would be a bug the contract catches.
- **Only float64/cpu exists now**, so these are single-valued today — but
  real, so the checks are meaningful the moment a second dtype/device
  lands (the same posture A4 took).
- **`requires_grad` requires a float dtype.** Autograd is defined for
  differentiable (floating) tensors; a future `int64`/`bool` native tensor
  should **not** accept `requires_grad=True` (gradients of integers are
  ill-defined). With float64-only today this never bites, but the rule is
  stated so it is not a surprise later.
- **Device mismatch is an error.** A backward never moves a gradient
  across devices; a mismatch raises (no implicit transfer). Moot today
  (cpu only), real later.
- **No dtype promotion, no silent casting, no silent NumPy fallback.** The
  same hard rules as every native milestone: an operation that would need
  promotion/casting/fallback raises instead. Gradients stay in native
  float64/cpu storage end to end.

## 11. Relationship to `tensorforge.Tensor`

Native autograd is **experimental and separate**, and stays that way:

- **No automatic conversion** between `tensorforge.Tensor`'s autograd and
  `NativeTensor`'s. A `Tensor` never becomes a `NativeTensor` (or vice
  versa) implicitly; a backward on one never touches the other's graph.
- **No silent backend switching.** Building a graph in one engine never
  routes an op to the other. The `dispatch_design.md` "no implicit
  fallback" rule governs here too.
- **`tensorforge.Tensor` behavior is unchanged.** This milestone adds a
  design doc and (later) an experimental native surface; it touches
  nothing in `src/tensorforge/tensor.py`, the modules, the optimizers, or
  the losses. The guardrail tests that the framework frontend never
  imports the experimental backend, and that `NativeTensor` never imports
  `Tensor`, continue to hold (and would be extended to cover the autograd
  surface).
- **A future explicit bridge** (Stage 3+ in
  [dispatch_design.md](dispatch_design.md)) — deciding whether and how a
  `Tensor` and a native tensor could ever interoperate — is a **separate
  later design**, not pre-committed here. For now the two autograd engines
  are two universes that share only a design vocabulary.

## 12. C++ / Python boundary

- **The first native autograd is a Python-managed graph over native
  forward operations.** The graph (nodes, edges, topological sort,
  `backward` driver, `unbroadcast`, the per-op closures) is Python, living
  at/above the `NativeTensor` layer (§5). C++ owns *no* graph state.
- **C++ remains the forward compute runtime.** `NativeTensorCore` and the
  kernels compute forward values only.
- **Backward functions call existing native forward kernels** to compute
  gradients wherever possible: `multiply` backward calls native
  `multiply`; `matmul` backward calls native `matmul`/`transpose`; `sum`
  backward calls native broadcasting; `unbroadcast` calls native `sum`.
  The gradient math is *native compute*, orchestrated by *Python graph
  logic*.
- **The few genuinely new kernels** first-scope backward needs are small
  and forward-shaped (`relu_backward`, and later `neg`/scalar-multiply
  exposure, then scatter for views) — §7, §9 — introduced only with the
  milestone that needs them, never speculatively.
- **Graph ownership does not go into C++.** Keeping the graph in Python
  makes v2.x dramatically simpler to write, test, and reason about
  (Python's GC and exceptions handle lifetime and error paths; no C++
  memory graph to leak). A future C++-owned autograd graph is a
  *much*-later possibility, explicitly out of scope, considered only if
  profiling ever shows the Python graph is the bottleneck (unlikely while
  the kernels dominate, per the Phase A benchmarks).

## 13. Error behavior

- **`backward()` on a `requires_grad=False` tensor raises** a clear
  `RuntimeError` ("this tensor does not require grad"), rather than
  silently doing nothing. Justification: a no-op would hide a real
  mistake (the user expected gradients and forgot to set the flag); the
  Python framework's usage always has a grad-requiring loss, so raising is
  the honest, debuggable choice. (This is a defensible place to differ
  from a permissive no-op; the design picks *raise* deliberately.)
- **Non-scalar `backward()` without an explicit `gradient` raises**
  `ValueError` naming the output shape (§7.2).
- **Gradient shape mismatch raises** `ValueError` naming **expected and
  actual** shapes (§7.3).
- **Gradient dtype/device mismatch raises** `ValueError` naming **both**
  the tensor's and the gradient's dtype/device (§7.3, §10).
- **`backward()` through a closed tensor raises** `RuntimeError` via
  `_require_open()` — the graph never reads freed storage (§5.5, §6).
- **Detached tensors do not track parents**, so a `backward()` reaching a
  detach boundary simply stops there (no error; gradient does not flow
  past it, by definition — §5.4).
- **In-place operations are not supported.** Autograd assumes a tensor's
  forward value is stable once it enters the graph; there is no native
  in-place op today and none is added, so this is a stated assumption, not
  a new enforcement burden.
- **No silent NumPy fallback**, ever — an unsupported backward case raises
  with a clear message; it never quietly computes the gradient with NumPy.

## 14. Testing plan

For the implementation milestones (v2.1+), not this design. Values
compared against finite differences / NumPy references with sensible
tolerances (`np.allclose`), consistent with the framework's
`test_gradcheck.py` posture and the reductions design's floating-point
honesty (§ order-sensitivity):

- **`requires_grad` defaults to `False`** on `from_array`/`zeros`/`full`.
- **A leaf's `grad` starts `None`** and only appears after `backward()`.
- **Scalar `backward()` seeds `1.0`** and produces correct leaf grads.
- **Non-scalar `backward()` without `gradient` raises**; with a correct
  `gradient` it works.
- **`add` backward** — same-shape, both leaves require grad.
- **`multiply` backward** — `da = u*b`, `db = u*a`.
- **`relu` backward** — gradient masked where input `<= 0`.
- **`sum` backward** — gradient broadcast back to input shape, with and
  without `keepdims`.
- **Gradient accumulation from multiple paths** — a tensor used twice
  (e.g. `x.multiply(x).add(x)`) gets summed contributions, mirroring the
  Python engine's "used-twice" test.
- **`zero_grad()` clears grad to `None`** and a subsequent backward
  re-accumulates cleanly.
- **`detach()`** — stops gradient flow; the detached tensor has no
  parents and `requires_grad=False`.
- **Broadcasting backward** through `add`/`multiply` — a `(3,)` operand
  combined with a `(2, 3)` gets its grad `unbroadcast` to `(3,)`.
- **`mean` backward** — `broadcast(u)/count`.
- **`matmul` backward** — `da = u@b.T`, `db = a.T@u`, checked against
  finite differences on small matrices.
- **`reshape`/`transpose` view backward** (if implemented in that
  milestone) — inverse reshape / inverse permutation.
- **dtype/device grad checks** — `grad.dtype == tensor.dtype`,
  `grad.device == tensor.device`, `grad.shape == tensor.shape`; mismatched
  explicit gradients raise naming both.
- **Closed-tensor errors** — `backward()`/`grad` access on a closed tensor
  raises `RuntimeError`.
- **Finite-difference gradient checks** for small cases where practical —
  the honest cross-check that the backward rules are correct, exactly as
  the Python engine is verified.
- **No interaction with `tensorforge.Tensor`** — a guardrail confirming
  the framework frontend still never imports the native autograd surface,
  and the native engine never imports `Tensor` or produces one.

No test asserts anything about speed.

## 15. Benchmark / profiling plan

Following the established `benchmarks/cpp_backend.py` philosophy —
correctness before timing, medians after warmup, hardware-dependent, **no
performance assertions anywhere**:

- **Correctness first, benchmarks after.** Native autograd benchmarks are
  meaningless until the backward rules are verified (§14); they come after,
  not with, the first implementation.
- **Compare forward-only vs. forward+backward overhead** once backward
  exists — how much does building and walking the graph add over a bare
  forward pass. This is the honest question a native autograd benchmark
  answers.
- **Keep the NumPy / Python `Tensor` reference honest.** The Python
  autograd engine and NumPy stay the baselines; the native engine is
  measured against them, not oversold.
- **No production-performance claims.** As with every prior native
  milestone, benchmarks *characterize*; they do not assert a win. The
  native engine is a correctness-and-design experiment first.

## 16. Staged roadmap after v2.0

Phase B is built in small, tested milestones, each landing only when the
previous is tested and documented — the same design-then-implement cadence
as Phase A:

- **v2.1 — Native Autograd Metadata Skeleton.** `requires_grad` (default
  `False`), `grad` (default `None`), `detach()`, `zero_grad()`, and
  leaf/non-leaf metadata on `NativeTensor`. **No differentiable ops yet**
  (or only a minimal manually-constructed graph for testing the
  traversal). This is the smallest honest step: the autograd *surface* and
  the graph *shape*, with the forward runtime untouched. The recommended
  **next milestone.**
- **v2.2 — Basic Native Backward: `add`/`multiply`/`relu`/`sum`.**
  Same-shape operands (no broadcasting yet), the topological `backward`
  driver, scalar seeding, accumulation from multiple paths. Introduces the
  one new kernel first-scope backward needs (`relu_backward`, §7.4) and the
  scalar-multiply/negation approach for `subtract` if included (§7.5).
- **v2.3 — Broadcasting backward + `mean` backward.** The `unbroadcast`
  helper (§8) over native reductions, applied to `add`/`multiply`
  backward, plus `mean` backward (broadcast + scale).
- **v2.4 — `matmul` backward.** `da = u@b.T`, `db = a.T@u` over native
  `matmul`/`transpose` (§7.4). Optionally `reshape`/`transpose` view
  backward (§9).
- **v2.5 — Native Autograd Demo.** A small, deterministic runnable example
  (a native forward+backward on a tiny expression, gradients materialized
  via `to_numpy` and cross-checked), a metadata-aware `repr`, and the
  wrapper/docs overview — the Phase B analog of the v1.11 forward demo.
- **Then Phase C — Native Training Stack**, and the rest of the
  Daedalus-class arc: **CUDA runtime** (where `device` gains a second
  value), **AMP / Tensor Core** (where `dtype` gains float16/bfloat16),
  **Transformer / text** examples, **distributed / DDP**, and a final
  **benchmark / profiling / docs** polish (the final portfolio release).

The plan may refine as the code lands, but the milestones stay tight and
the ordering is fixed by the dependencies §7/§8/§9 spell out (metadata →
same-shape backward → broadcasting/mean → matmul → demo). Each lands only
when the previous is tested and documented; the Python framework remains
the reference implementation throughout.

## 17. Phase B completion — final support matrix (v2.6)

This is the authoritative record of what native autograd supports as Phase
B closes. Every row is implemented and tested (the "focused correctness
test" column names the check that verifies the rule against exact
analytical values and, where marked FD, central finite differences — NumPy
is the *test-side* reference only; the gradient path itself is native).

| op | forward layer | backward layer | broadcasting | view / copy | dim limits | grad shape | lifetime | focused test |
|----|---------------|----------------|--------------|-------------|-----------|-----------|----------|--------------|
| `add` | `tf_core_add` (+ contiguous fast path) | Python closure: pass upstream to each operand, `_unbroadcast` to its shape | yes (`_unbroadcast`) | fresh owning result | any | = each requiring leaf | one-shot / retain | `test_add_backward_*` |
| `subtract` | `tf_core_subtract` | left = upstream, right = `_negated` (broadcast-scalar ×(−1), no negate kernel), each `_unbroadcast` | yes | fresh owning | any | = leaf | one-shot / retain | `test_subtract_backward_*` |
| `multiply` | `tf_core_multiply` | `u·b`, `u·a` (native `multiply`), each `_unbroadcast` | yes | fresh owning | any | = leaf | one-shot / retain | `test_multiply_backward_*` (FD) |
| `relu` | `tf_core_relu` (fused fast path) | fused `tf_core_relu_backward` (`u` where `x>0`, else `0`; `x==0` blocks) | n/a (unary) | fresh owning | any | = input | one-shot / retain | `test_relu_backward_*` (FD) |
| `sum` | `tf_core_sum` | `_broadcast_back`: reduced axes reinserted as size 1, expanded by native zero-stride broadcasting | broadcast-back | fresh owning | `axis` None / single int / negative | = input | one-shot / retain | `test_sum_backward_*` |
| `mean` | `tf_core_sum` + `tf_storage_scale` | `sum`'s broadcast-back scaled by native `1/count` scalar multiply | broadcast-back | fresh owning | `axis` None / single / negative | = input | one-shot / retain | `test_mean_backward_*` (FD) |
| `matmul` | `tf_core_matmul` (naive triple loop) | `u @ b.T`, `a.T @ u` (native `matmul` over strided transpose views) | none (strictly 2-D) | fresh owning | strictly 2-D `(m,n)@(n,p)` | = each operand | one-shot / retain | `test_matmul_backward_*` (FD) |
| `reshape` | metadata-only view | inverse reshape of upstream, materialized to owning storage | n/a | borrowing view | contiguous source, equal element count | = input | one-shot / retain | `test_reshape_backward_*` |
| `transpose` / `T` | metadata-only view | inverse permutation of upstream, materialized | n/a | borrowing view | full permutation of `range(ndim)` | = input | one-shot / retain | `test_transpose_backward_*` |
| `contiguous_copy` | odometer materialize | identity (upstream passes through unchanged; grad lives at the logical shape) | n/a | fresh owning | any | = input | one-shot / retain | `test_contiguous_copy_backward_*` |
| `narrow` | metadata-only view | scatter `tf_core_narrow_backward` into fresh zeros of the parent shape at the narrowed region | n/a | borrowing view | `dim ∈ [0, ndim)`, `start ≥ 0`, `start+length ≤ size` | = parent (fresh owning contiguous) | one-shot / retain | `test_narrow_backward_*` (FD) |

**Cross-cutting invariants (all guardrail-tested in v2.6):**

- **Architecture.** The autograd graph is Python-managed on `NativeTensor`;
  `NativeTensorCore` and the C++ kernels compute forward/backward numerical
  primitives and own **no** graph state. The three fused backward kernels
  (`tf_core_relu_backward`, `tf_core_narrow_backward`, and `tf_core_sum`
  reused for broadcast-back / `unbroadcast`) are surfaced only as
  forward-shaped numerical methods and are **not** in
  `list_kernels()` / `TENSOR_CORE_KERNELS`.
- **No NumPy in the gradient path.** Backward values are computed by native
  kernels end to end; NumPy appears only to marshal small shape/stride
  arrays across ctypes and to materialize copies out at `to_numpy`. A
  runtime guard replaces NumPy's numerical functions with tripwires around
  representative backward passes (elementwise, broadcasting, reduction,
  matmul, and a transpose→narrow→contiguous_copy→reshape view chain) and
  confirms none is reached.
- **Isolation.** Native ops return `NativeTensor`; native gradients are
  `NativeTensor`-backed; `tensorforge.Tensor` stays NumPy-backed; neither
  engine's backward touches the other; mixed operands raise clearly rather
  than dispatch implicitly.
- **Explicit backend.** Reached only through `tensorforge.experimental`;
  `import tensorforge` imports neither `experimental` nor `backends`;
  unavailability raises the build-instructions `ImportError` (no silent
  NumPy fallback); no automatic backend selection.
- **Gradient ownership.** Leaves retain grad; non-leaves do not; grads match
  the leaf's shape/`float64`/`cpu`; repeated successful passes accumulate by
  native addition; `zero_grad()` returns the leaf grad to `None` without
  touching data or graph.
- **Graph lifetime.** One-shot by default (a successful pass frees the
  traversed operation graph), `retain_graph=True` reuse, deterministic
  freed-graph errors, and snapshot-based failure rollback — verified across
  a mixed graph (shared intermediate + broadcast + view).

## 18. Divide backward — explicit decision (deferred beyond Phase B)

`divide` backward is **deliberately not implemented in Phase B, and Phase B
is complete without it.**

- **Forward status.** Element-wise divide exists only as a raw-buffer
  kernel (`tf_elementwise_divide`, in `cpp.list_kernels()`); it is **not** a
  `NativeTensorCore` method and there is no `NativeTensor.divide`. There is
  therefore no differentiable-op surface to attach a backward to, and adding
  one is out of this milestone's scope (no new operations, no operator
  overloads).
- **Why the current set is sufficient to open the native training stack.**
  A first native training stack — parameters, a linear layer, an MSE or
  cross-entropy-style loss, and SGD — is fully expressible with the
  completed operation set: `matmul` + broadcast `add` build an affine layer,
  `relu` a nonlinearity, `subtract`/`multiply`/`sum`/`mean` a squared-error
  loss, and the view ops reshape between stages. None of these needs
  division on the backward path. Division's backward (`da = u/b`,
  `db = -u·a/b²`) additionally needs a core-level `divide` method *and*
  negation, both currently absent (§7.5) — so implementing it honestly is a
  small feature of its own, not a guardrail fix.
- **Roadmap placement.** Divide backward is scheduled for **Phase C**, added
  alongside a core-level `NativeTensorCore.divide` method when a native op
  (e.g. a softmax/normalization or a division-based loss) first needs it —
  not speculatively. Until then, an attempt to differentiate division simply
  does not exist to call, which is the honest state rather than a silent
  NumPy fallback.

## 19. Phase status

- **Phase A — native CPU runtime: complete** (v1.14–v1.21).
- **Phase B — native autograd: complete** (v2.0 design → v2.1 metadata
  skeleton → v2.2 core backward → v2.3 narrow backward → v2.4 graph
  lifetime → v2.5 benchmark characterization → **v2.6 guardrails and
  completion**). Verified test count at completion: **923 tests** (893 at
  the v2.5 baseline plus the 30 v2.6 cross-cutting guardrails), with the
  native suite skipping when the compiled backend is not built.

  **Current limitations (unchanged, restated for the record):** `float64` /
  `cpu` only; no dtype promotion or casting; no CUDA; no AMP; no
  module/parameter/optimizer/training stack yet; no `divide` backward (§18);
  `matmul` strictly 2-D; no implicit dispatch; no `tensorforge.Tensor`
  integration; experimental and not production-ready; not a PyTorch
  replacement. **Benchmark interpretation** (v2.5) is unchanged:
  measurements are hardware-specific, include Python graph management and
  wrapper/ctypes overhead per mode, carry no speed assertions, and make no
  cross-framework claims.

- **Phase C — native training stack: under way.** Its first milestone,
  **Advanced C++ v3.1 — NativeParameter and Parameter Registration
  Contract, is complete**: `NativeParameter` is a `NativeTensor` subclass
  whose instances are always graph-free owning leaves (construction copies
  array-like data or an existing tensor's current value into independent
  owning contiguous storage; `requires_grad` is a validated real bool,
  default `True`, with `False` building a frozen-but-registerable
  parameter; the internal graph constructors are overridden so every
  operation, view, copy, and `detach()` returns a plain `NativeTensor` —
  parameter-ness never propagates), identity is object identity (no
  `__eq__`/`__hash__`; future optimizer state keys by `id`), and
  `NativeParameterRegistry` is the minimal insertion-ordered registration
  contract (dot-free non-empty string names; `NativeParameter`-only values
  with `None` unregistering; replacement preserves position without
  closing or mutating the old parameter; removal deletes the slot so
  re-registration appends; aliases visible by name with identity-based
  deduplication in `parameters()` / first-name-wins
  `unique_named_parameters()`). Verified test count at completion: **972
  tests** (923 at the v2.6 baseline plus the 49 v3.1
  parameter/registration tests). **Its second milestone, Advanced C++
  v3.2 — NativeModule Core and Recursive Registration, is complete**:
  `NativeModule` is the Python-side module-hierarchy core — assignment
  registers (`NativeParameter` → parameter registry, `NativeModule` →
  child registry, everything else an ordinary attribute; one category per
  name, latest assignment wins, position-preserving replacement, `None`
  unregisters), `register_parameter`/`add_module` mirror assignment,
  traversal (`named_modules`/`modules`/`named_parameters`/`parameters`)
  is deterministic pre-order depth-first with identity deduplication and
  first-discovered dotted names (shared modules/parameters emit once;
  reference cycles terminate safely), `zero_grad()` delegates to each
  unique parameter, and `train(mode)`/`eval()` propagate a
  bool-validated `training` flag — with no storage ownership and nothing
  ever closed or mutated by the module. Verified test count at
  completion: **1021 tests** (972 plus the 49 v3.2 module tests). **Its
  third milestone, Advanced C++ v3.3 — Native State Dictionary Contract,
  is complete**: `state_dict()` returns an insertion-ordered
  `{canonical_name: NativeTensor}` snapshot (the v3.2 first-discovered
  dotted keys; every value an independent owning contiguous graph-free
  `requires_grad=False` copy made by the native copy path, sharing no
  storage with the model in either direction), and
  `load_state_dict(state_dict, strict=True)` copies values back **into**
  the existing parameters atomically — strict as a real bool, mapping
  and string-key validation, missing/unexpected keys reported together
  under `strict=True` (returned as an immutable result under
  `strict=False`), exact shape/dtype/device preflight naming the failing
  key, stage-then-commit with rollback so no failure leaves the model
  partially updated — preserving parameter identity, registration,
  shared aliases (one canonical key updates the shared object once),
  `requires_grad`/frozen state, gradients by identity and value, and
  training flags. The internal primitive is
  `NativeParameter._adopt_value_core` (controlled value replacement —
  not yet the optimizer update API); a graph built before loading stays
  memory-safe and reads the newly loaded values. Verified test count at
  completion: **1075 tests** (1021 plus the 54 v3.3 state-dict tests).
  **Its fourth milestone, Advanced C++ v3.4 — NativeLinear, is
  complete**: the first concrete native layer — a `NativeModule` with a
  `NativeParameter` weight of shape `(in_features, out_features)` (the
  `x @ weight` orientation) and optional `(out_features,)` bias,
  deterministic fan-in uniform initialization from a local seeded
  generator (global RNG untouched), full constructor validation before
  native allocation, strictly 2-D `(batch, in_features)` input
  validation, forward as pure existing operations
  (`input.matmul(weight)` + broadcast `add(bias)`) so **the existing
  autograd is the backward implementation** (no manual or fused path;
  gradients verified analytically and by central finite differences),
  frozen-parameter support, deterministic `["weight", "bias"]`
  registration, and full v3.3 state-dict compatibility (identity,
  gradients, and frozen state survive loads; bias/no-bias mismatches
  follow the strict key rules; the forward→backward→load-after-completion
  mutation boundary is unchanged). Verified test count at completion:
  **1117 tests** (1075 plus the 42 v3.4 layer tests). The next milestone
  is **Advanced C++ v3.5 — NativeReLU and NativeSequential**: a
  `NativeReLU` module wrapping the existing `relu()`, and a
  `NativeSequential` ordered container with integer-string child names,
  deterministic recursive traversal, forward composition, shared-module
  behavior, train/eval propagation, and state_dict compatibility —
  **no loss, optimizer, or training loop in v3.5**. Losses, optimizers,
  and training are **not** combined into one milestone; each lands only
  when the previous is tested and documented, with the Python framework
  remaining the reference implementation.
