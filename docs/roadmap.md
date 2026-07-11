# Roadmap

## Where the project is

**v3.0 is the completed Python framework release.** You
can define a model (including CNNs), train it, regularize it,
normalize it, evaluate it honestly, save it, and resume it bit-for-bit
— all from readable NumPy code, all tested, all documented. For the
version-by-version story, see [release_history.md](release_history.md);
for a two-minute overview, see [project_summary.md](project_summary.md).

## What's been built

**v0.x — foundations.** The Tensor and autograd engine (elementwise
ops, matmul, exp/log/tanh/sigmoid/relu/softmax with broadcasting-aware
gradients), the module system (Parameter, Module, Linear, activations,
Sequential), SGD, MSE and cross-entropy losses, and the first
examples: linear regression, XOR, and the multi-class spiral.

**v1.x — training basics and evaluation.** The accuracy metric, Adam,
mini-batching, gradient checking against finite differences,
save/load parameters, model summaries and parameter counting, frozen
parameters, train/validation splitting, evaluation helpers, binary
cross-entropy with a binary classification example, and checkpoints
that capture optimizer state so training can resume exactly.

**v2.x — deeper framework features.** Train/eval mode, Dropout (and an
example that uses it properly), eval-safe evaluators, BatchNorm1d with
module buffers, gradient clipping, the StepLR scheduler, scheduler
state in checkpoints — completing the training-resume story — and
image-shaped input: Conv2d, MaxPool2d, Flatten, and a tiny CNN
example; LayerNorm as the batch-independent normalization; optional
RNG state in checkpoints for bit-exact dropout resume; and a
release-readiness pass over docs and guardrail tests.

## Practical next steps

The Python line is done; what remains is expansion on its own terms:

- **Advanced branches** — the C++ backend experiment now has
  elementwise kernels, naive and cache-tiled 2-D matmuls, an
  introspection API, and a native runtime prototype: shape/stride
  metadata, a C++-owned NativeStorage buffer, NativeTensorView binding
  the two with native contiguous materialization, and NativeTensorCore
  composing it all into the first native tensor runtime object with
  metadata-only view operations (reshape, transpose, narrow), native
  compute over strided views (elementwise ops and matmul), a
  benchmark suite measuring NumPy, raw-buffer kernels, and the
  TensorCore runtime side by side, and a backend dispatch design plus
  Stage 1 of it — an explicit `get_backend("numpy"|"native")` API with
  no implicit routing and a polished conversion contract
  (`tensor_from_array` in, `to_numpy` out; see
  [backend_experiments.md](backend_experiments.md) and
  [dispatch_design.md](dispatch_design.md)). Stage 2 — a forward-only
  native tensor wrapper over `NativeTensorCore` — is now **designed**
  (purpose, non-goals, ownership/lifetime, conversion contract, minimal
  API, testing plan, and a staged build sequence; see
  [native_tensor_wrapper_design.md](native_tensor_wrapper_design.md)),
  and it is **now feature-complete as a forward-only wrapper**:
  `tensorforge.experimental.NativeTensor`, a forward-only wrapper over
  `NativeTensorCore` with constructors, metadata, `to_numpy`, an
  explicit ownership/lifetime story (v1.8), forward compute — `relu`,
  `add`, `subtract`, `multiply`, `matmul` with exact-shape/2-D behavior
  and no broadcasting (v1.9) — and metadata-only view ops: `reshape`,
  `transpose`, `T`, `narrow` returning borrowing wrappers, plus
  `contiguous_copy` returning an owning one (v1.10). No autograd, no
  operator overloads, not `tensorforge.Tensor`. It is now demonstrable
  too: a small deterministic example
  (`examples/native_tensor_demo.py`), a metadata-only `repr`, and a
  wrapper overview in the docs (v1.11); and honestly characterized —
  the benchmark suite times the wrapper's ops (strided views and
  `contiguous_copy` included) across NumPy, the raw-buffer kernels,
  `NativeTensorCore`, and `NativeTensor`, overheads included and with no
  performance assertions (v1.12). Acting on that finding — the
  elementwise cost is the generic shape/stride odometer traversal in the
  native runtime, not the wrapper — the contiguous elementwise fast path
  is now **designed** (v1.13): flat, index-free loops for contiguous
  `relu`/`add`/`subtract`/`multiply`, the odometer kept for strided
  views, placed in the `NativeTensorCore`/native-kernel layer so
  `NativeTensor` inherits it, bit-for-bit equivalent
  ([native_contiguous_fast_path_design.md](native_contiguous_fast_path_design.md))
  — and now **implemented** (v1.14): flat, index-free kernels beside the
  generic odometer ones, selected when every operand is row-major
  contiguous (nonzero offsets and scalars included) and falling back to
  the retained odometer path for strided views, proven bit-for-bit equal
  to it; `NativeTensor` inherited the change with no wrapper edits, and
  no broadcasting, reductions, autograd, `Tensor` integration, or CUDA
  came with it. Its impact is now **measured and reported** (v1.15): on a
  local run the contiguous elementwise rows moved to roughly raw-buffer-
  C++ speed (~1.5× NumPy at 1000×1000) while the strided-view rows stayed
  on the retained odometer (~2.5–3.5×), and `NativeTensor` tracked
  `NativeTensorCore` throughout — with matmul and `contiguous_copy`
  unchanged, numbers hardware-dependent, and no test asserting a speedup
  (see [backend_experiments.md](backend_experiments.md)). Building on
  that, native broadcasting is now **designed (v1.16) and implemented
  (v1.17)**, Phase A2 complete: it lifts the native elementwise ops from
  exact-shape-only to NumPy-style broadcasting (scalar↔tensor, same-rank
  size-1 stretching, left-padding with leading 1s) through a zero-stride
  read model that never materializes an expanded operand, with the v1.14
  fast path and generic odometer preserved for the same-shape case and
  **no new C++ kernel** (the existing odometer consumes the broadcast
  strides). `NativeTensor` and the explicit native backend inherited it
  with no wrapper edit, and results match NumPy exactly
  ([native_broadcasting_design.md](native_broadcasting_design.md)). The
  native reductions are now **designed (v1.18) and implemented (v1.19)** —
  Phase A3 complete for `sum`/`mean` (`axis`/`keepdims`, negative axes;
  `max`/`argmax`/`min`/`product` deferred) via a scatter-accumulate
  kernel that is the dual of broadcasting — where broadcasting reads
  through zero strides, a reduction writes through zero strides — reading
  any strided/offset input directly and writing a freshly allocated
  row-major contiguous output, with honest order-sensitive floating-point
  behavior (NumPy comparison to a tolerance, not bit-for-bit).
  `NativeTensor` and the explicit backends inherited `sum`/`mean` with no
  wrapper edit, and reductions stay forward-only — the broadcast-backward
  relationship (a broadcast backward is a reduction over the broadcast
  axes) is the recorded reason reductions precede native autograd
  ([native_reductions_design.md](native_reductions_design.md)). Building
  on that, the dtype/device metadata contract is now **designed (v1.20)
  and implemented (v1.21)**, Phase A4 complete and **Phase A closed in
  code**: `dtype`/`device` are explicit, inspectable, validated string
  tags (`"float64"`/`"cpu"`) owned by `NativeStorage` and surfaced
  read-only through `NativeTensorCore`/`NativeTensor`, with
  default-preserving constructor arguments (so every existing call is
  byte-for-byte unchanged), matching-dtype/device operation guards, a hard
  no-promotion/no-silent-conversion rule, and — per the design's
  reject-over-inert recommendation — rejection of any
  non-`float64`/non-`cpu` construction so no tensor advertises a dtype the
  kernels cannot compute. It is metadata only: no kernel, no compute
  change, `to_numpy` still float64, and pure `normalize_dtype`/
  `normalize_device` helpers validate the tags
  ([native_dtype_device_metadata_design.md](native_dtype_device_metadata_design.md)).
  **Phase B is under way.** Advanced C++ v2.0 — the native autograd design
  — is complete (a Python-managed reverse-mode graph at the `NativeTensor`
  layer, native gradients honoring the v1.21 `grad.dtype == tensor.dtype`
  / `grad.device == tensor.device` contract, broadcasting backward via A3
  reductions). **v2.1 implemented the autograd metadata skeleton** —
  opt-in `requires_grad`/`grad`/`is_leaf`, `zero_grad`/`detach`, and a
  reverse-topological `backward` driver with `NativeTensor`-backed
  gradients — and **v2.2 — Core Native Autograd Operations — is now
  implemented**: `add`/`subtract`/`multiply`/`relu`/`sum`/`mean`/
  `matmul`/`reshape`/`transpose`/`T`/`contiguous_copy` are differentiable
  (graph nodes when an operand requires grad, plain forward tensors
  otherwise), broadcasting backward runs through a native `unbroadcast`
  reduction, sum/mean broadcast their upstream back natively, and the one
  new C++ kernel is the fused `relu_backward` — with every rule verified
  against finite differences and a deterministic native demo
  (`examples/native_autograd_demo.py`). **v2.3 — Native Narrow Backward —
  is now implemented**, completing the view-backward set: `narrow(dim,
  start, length)` builds a graph node when its parent requires grad, and
  its backward **scatters** the upstream gradient into a fresh owning
  row-major contiguous zeros tensor of the parent's shape at the narrowed
  region (un-narrowed positions stay zero) through the one new C++ kernel
  `tf_core_narrow_backward`, the odometer dual of `tf_core_sum`. The
  gradient lives at the logical shape, so transposed, narrowed, and
  nonzero-offset parents all differentiate correctly, and there is no
  NumPy in the gradient path; `NativeTensorCore` and the C++ kernels still
  own no graph state. **v2.4 — Native Autograd Graph Lifetime Policy — is
  now implemented** (a Python-only `NativeTensor` change): `backward` takes
  a `retain_graph` flag (validated as a real bool first), the default
  `backward(retain_graph=False)` is one-shot and releases the traversed
  operation graph on success, a later backward through a freed graph raises
  a clear error (never silently truncating history), `retain_graph=True`
  keeps the graph for another pass, leaf gradients accumulate until
  `zero_grad()`, and a failed pass rolls back with no partial commit or
  partial free — explicitly not full PyTorch parity. **v2.5 — Native
  Autograd Benchmark Characterization — is now done** (a measurement-only
  milestone that changes no behavior): a reproducible harness
  (`benchmarks/benchmark_native_autograd.py`) times four modes —
  forward-native, forward+graph-construction, fresh forward+backward, and
  repeated retained backward — across five workloads, with a correctness
  gate, median/spread reporting, a JSON mode, and one honest
  hardware-specific snapshot carrying no speed assertions (see
  [native_autograd_benchmarks.md](native_autograd_benchmarks.md)). **v2.6 —
  Phase B Guardrails and Completion — is now done, closing Phase B in
  code**: cross-cutting guardrail tests
  (`tests/test_native_autograd_guardrails.py`) lock the completed engine's
  invariants (a NumPy-no-fallback runtime guard, `NativeTensor` /
  `tensorforge.Tensor` isolation, explicit-backend / no-implicit-dispatch
  behavior, gradient-ownership, graph-lifetime, detach, view+offset,
  closed-operand safety, the kernel-registry boundary, and the benchmark
  mode contract), and the final Phase B support matrix and the explicit
  divide-backward decision (deferred beyond Phase B) are documented, with no
  operation, kernel, or optimization added. **Phase C — a native training
  stack — is now under way: Advanced C++ v3.1 — NativeParameter and
  Parameter Registration Contract — is implemented**:
  `tensorforge.experimental.NativeParameter`, a `NativeTensor` subclass
  whose instances are always graph-free owning leaves (construction takes
  an independent owning contiguous copy of array-like data or an existing
  tensor's current value, inheriting no graph history; `requires_grad` is
  a validated real bool defaulting to `True`, `False` giving a frozen but
  registerable parameter; every operation, view, copy, and `detach()`
  returns a plain `NativeTensor` — parameter-ness never propagates; and
  identity is object identity, never value), plus
  `NativeParameterRegistry`, the minimal insertion-ordered registration
  contract the future `NativeModule` will embed (dot-free non-empty string
  names, `NativeParameter`-only slots with `None` unregistering,
  position-preserving replacement, alias-visible named traversal, and
  identity-deduplicated unique traversal) — still with no module
  hierarchy, layer, loss, optimizer, or training loop, and with
  `tensorforge.Tensor`/`tensorforge.nn.Parameter` untouched. **The next
  milestone is Advanced C++ v3.2 — NativeModule Core and Recursive
  Registration.** CUDA/GPU experiments are still entirely future work. The
  Python framework stays the reference implementation.
- **The Daedalus-class native roadmap** — the longer arc the advanced
  branch is building toward, in phases, each landing only when the
  previous is tested and documented:
  - **Phase A — native CPU runtime.** A1: the contiguous elementwise
    fast path — **complete** (designed v1.13, implemented v1.14,
    benchmark impact reported v1.15). A2: broadcasting for elementwise
    ops — **complete** (designed v1.16, implemented v1.17). A3: reductions
    (sum/mean first; max/argmax/min/product later) — **complete**
    (designed v1.18, implemented v1.19). A4: explicit dtype and device
    metadata (float64/cpu) — **complete** (designed v1.20, implemented
    v1.21, metadata-only), which **closes Phase A in code**.
  - **Phase B — native autograd (complete).** The v2.0 design is complete
    (a Python-managed reverse-mode graph at the `NativeTensor` layer — see
    [native_autograd_design.md](native_autograd_design.md)); **v2.1
    implemented the metadata skeleton and reverse-topological backward
    driver** (opt-in `requires_grad`/`grad`/`is_leaf`,
    `zero_grad`/`detach`/`backward`, `NativeTensor`-backed gradients); and
    **v2.2 implemented the core backward operations** — add, subtract,
    multiply, relu (one new fused kernel), sum, mean, matmul,
    reshape/transpose/T, contiguous_copy, and broadcasting backward via a
    native `unbroadcast`, finite-difference-verified, with a
    deterministic native autograd demo; **v2.3 implemented native narrow
    backward** — the scatter that was v2.2's one deferral — through a
    second new fused kernel (`tf_core_narrow_backward`, the odometer dual
    of `sum`), completing the view-backward set with transposed / narrowed
    / nonzero-offset parents all handled; and **v2.4 implemented the graph
    lifetime policy** — a one-shot `backward(retain_graph=False)` that frees
    the traversed graph on success, opt-in `retain_graph=True` reuse,
    deterministic freed-graph errors, and snapshot-based failure safety (a
    Python-only change; no kernel touched); and **v2.5 characterized the
    stack** with a measurement-only benchmark harness (four modes across
    five workloads, correctness gate, median/spread reporting, JSON output,
    one hardware snapshot, no speed assertions); and **v2.6 completed Phase
    B** — cross-cutting guardrail tests
    (`tests/test_native_autograd_guardrails.py`) that lock the engine's
    invariants (NumPy-no-fallback runtime guard, `NativeTensor` /
    `tensorforge.Tensor` isolation, explicit-backend behavior,
    gradient-ownership, graph-lifetime, detach, view+offset, closed-operand
    safety, kernel-registry boundary, benchmark mode contract), the final
    Phase B support matrix, and the explicit divide-backward decision
    (deferred beyond Phase B), adding no operation, kernel, or optimization.
  - **Phase C — native training stack (under way).** **v3.1 —
    NativeParameter and Parameter Registration Contract — is complete**: a
    `NativeParameter` subclass of `NativeTensor` whose instances are
    always graph-free owning leaves with validated `requires_grad`
    (frozen parameters stay registerable), independent owning contiguous
    construction from array-like data or an existing tensor's current
    value, operation results that are always plain `NativeTensor`
    (parameter-ness never propagates), object-identity semantics for
    future optimizer state, and the minimal insertion-ordered
    `NativeParameterRegistry` (dot-free names, `None` unregisters,
    position-preserving replacement, alias and identity-deduplication
    rules) — no `NativeModule` hierarchy, optimizer, or training loop yet.
    **Next: v3.2 — NativeModule Core and Recursive Registration**
    (child-module registration, automatic parameter/module assignment
    registration, recursive `parameters()`/`named_parameters()`/
    `modules()`/`named_modules()`, `zero_grad()`, deterministic traversal,
    shared-parameter/-module handling, and the train/eval state
    foundation; no layers or optimizers in v3.2).
  - **Then beyond:** the rest of the native training stack, the CUDA
    runtime (where `device` gains a second value), an AMP / Tensor Core path
    (where `dtype` gains float16/bfloat16), Transformer / text examples,
    distributed / DDP, and a final benchmark / profiling / docs polish
    (the final portfolio release).
- **A larger synthetic image example** — more classes, bigger images,
  still dependency-free.
- **More docs** — deeper walkthroughs of individual layers, if the
  framework grows further.

## What this project is not

TensorForge is not production software and doesn't try to compete with
PyTorch or any real framework. It trades performance for readability
at every opportunity — that's the point. If it helps someone
understand what `loss.backward()` actually does, it has succeeded.
