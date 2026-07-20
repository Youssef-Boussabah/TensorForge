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
  identity-deduplicated unique traversal). **v3.2 — NativeModule Core and
  Recursive Registration — is implemented on top of it**:
  `tensorforge.experimental.NativeModule`, the Python-side
  module-hierarchy core — attribute assignment registers parameters and
  child modules (ordinary values, plain `NativeTensor`s, and
  stable-framework objects stay ordinary attributes; one category per
  name, latest assignment wins, `None` unregisters), explicit
  `register_parameter`/`add_module` mirror assignment, recursive
  `parameters()`/`named_parameters()`/`modules()`/`named_modules()`
  traversal is deterministic depth-first with identity deduplication,
  first-discovered canonical dotted names, shared-parameter/-module
  handling, and cycle safety, plus recursive `zero_grad()` and
  bool-validated `train()`/`eval()` propagation — still with no layer,
  loss, optimizer, state_dict, or training loop, no storage ownership,
  and `tensorforge.Tensor`/`tensorforge.nn` untouched. **v3.3 — Native
  State Dictionary Contract — is implemented on top of that**: the
  in-memory, parameters-only state contract — `state_dict()` snapshots
  each unique parameter's value under its canonical dotted name into an
  independent owning graph-free `NativeTensor` (shared parameters once,
  first-discovered path wins, frozen included, no storage shared with
  the model in either direction), and
  `load_state_dict(state_dict, strict=True)` copies values back into the
  existing `NativeParameter` objects atomically (full preflight
  validation naming the failing key, strict/non-strict key handling with
  an immutable missing/unexpected result, exact shape/dtype/device
  matching with no casting/reshaping/broadcasting, stage-then-commit
  with rollback) while preserving parameter identity, registration,
  shared aliases, `requires_grad`/frozen state, gradients, and training
  flags — still with no layer, loss, optimizer, file serialization,
  checkpoint, or training loop. **v3.4 — NativeLinear — is implemented
  on top of all of it**: the first concrete native layer,
  `tensorforge.experimental.NativeLinear` — a `NativeModule` with a
  `(in_features, out_features)` `NativeParameter` weight (the
  `x @ weight` orientation) and optional `(out_features,)` bias,
  deterministic fan-in uniform initialization from a local seeded
  generator (global random state untouched), full argument validation before
  native allocation, strictly 2-D input semantics, forward as pure
  existing native operations (`matmul` + broadcast `add`) so the
  existing autograd supplies backward (verified analytically and by
  central finite differences), frozen-parameter support, deterministic
  `["weight", "bias"]` registration and state-dict keys, and full v3.3
  load compatibility. **v3.5 — NativeReLU and NativeSequential — is
  implemented, completing the first composable native model surface**:
  `NativeReLU`, a parameter-free shape-generic activation module over
  the existing native `relu()` and its existing backward, and
  `NativeSequential`, an ordered composition container whose children
  live in contiguous integer-string execution slots (`"0"`, `"1"`, ...)
  where execution order is the registered order — replacement preserves
  position, `append` takes the next slot, and gaps, non-slot child
  names, direct parameters, slot removal, and self-insertion are
  rejected — with position-based execution versus identity-deduplicated
  traversal/state for shared children, empty-sequence identity forward,
  nested composition, slot-derived state keys (`"0.weight"`,
  `"2.bias"`, nested `"0.0.weight"`), and a Linear→ReLU→Linear model
  verified end to end by exact references and central finite
  differences. **v3.6 — NativeMSELoss — is implemented, closing the
  forward side of the native training story**: a parameter-free loss
  module composing existing native operations (`subtract` →
  `multiply(diff, diff)` → `mean`/`sum`) into a scalar loss whose
  gradients come entirely from the existing autograd
  (duplicate-parent accumulation, subtract's sign split, and the
  existing native mean backward's `1/N` — no division, no manual
  backward), with exactly `"mean"`/`"sum"` reductions, strict
  exact-shape/no-broadcasting validation, empty state, and exact plus
  finite-difference verification through a full
  Linear→ReLU→Linear→MSE chain. **v3.7 — Native Parameter Mutation
  Safety and Versioning — is implemented**: every `NativeParameter`
  carries a read-only monotonic value version counting replacements of
  the owned value; `copy_value_(source)` is the one controlled no-grad
  mutation primitive (identity, gradients, `requires_grad`, and
  registrations preserved; native never-aliased copies; failure changes
  nothing); `load_state_dict` increments each matched canonical
  parameter once, after its atomic commit; and graphs record expected
  versions where backward reads a direct parameter operand's forward
  value (`multiply`/`matmul`/`relu`), so `backward()` raises a
  deterministic stale-graph error — before any callback or gradient
  commit — when such a parameter was mutated after forward, while
  value-independent graphs (add/subtract/reductions/views) stay valid
  with correct gradients. **v3.8 — NativeSGD — is implemented**: the
  first native optimizer — minimal SGD over identity-deduplicated
  `NativeParameter` objects with a strictly validated learning rate
  and a two-phase mutation-atomic `step()` that stages graph-free
  native updates and commits them through `copy_value_` (frozen and
  gradient-less parameters skipped, identities and gradients
  preserved, one version increment per updated parameter), plus
  preflighted `zero_grad()` — no momentum, weight decay, parameter
  groups, optimizer state, or training loop. **v3.9 — the native MLP
  training proof — is implemented**: `examples/native_mlp_training.py`
  trains a 2→8→ReLU→1 native MLP on fixed synthetic regression data
  for 25 deterministic SGD steps entirely through the native stack —
  a fresh graph every iteration, one version increment per parameter
  per step, stable identities, explicit lifetime handling, and a
  monotonic 99.5% loss reduction, all bit-reproducible across runs.
  **v3.10 — the integration checkpoint — is complete**: the branch's
  first major usable native training checkpoint, adding no numerical
  behavior — honest README/summary/architecture presentation of both
  framework lines, the canonical
  [native support matrix](native_support_matrix.md), documentation and
  export guardrails, and CI/repository-hygiene audits — leaving
  `advanced/cpp-backend` ready for its first pull request into `main`
  after validation. **v3.11 — native optimizer math primitives — is
  complete**: differentiable native `sqrt` and `reciprocal` through
  the whole stack (new odometer + contiguous fast-path C++ kernels,
  core methods, wrapper methods), with saved-forward-result backwards
  — each derivative reads the recorded output, never the parent's
  current value, so neither records a parameter version and mutation
  after forward leaves those edges valid — IEEE float64
  exceptional-value semantics locked by tests, and no general
  division (`reciprocal` + `multiply` compose what the stack needs):
  the reusable math NativeAdam requires. **v3.12 — NativeAdam — is
  complete**: the native adaptive optimizer — validated
  `lr`/`betas`/`eps`, identity-deduplicated parameters, eagerly
  allocated optimizer-owned native first/second-moment buffers and
  per-parameter step counters, bias-corrected graph-free updates
  staged at the core level (reciprocal + sqrt, no division) and
  committed through `copy_value_`, gradients retained until
  `zero_grad()`, mutation-atomic public failures, and an explicit
  state lifetime (`close()`) — with no weight decay, AMSGrad,
  parameter groups, or schedulers. **v3.13 — the native optimizer
  state contract — is complete**: in-memory
  `state_dict()`/`load_state_dict()` on both native optimizers — one
  versioned schema (format 1, exact optimizer type tag, ordered
  positional shape/dtype/device parameter metadata; no ids, names,
  values, or gradients), caller-owned independent NativeTensor m/v
  snapshots and per-parameter step counts for NativeAdam, exact
  validation with staged atomic loading that never touches parameter
  values, versions, gradients, or retained graphs, and a proven
  deterministic in-memory training continuation. **v3.14 — native
  checkpointing and deterministic file resume — is complete**:
  `save_native_checkpoint`/`load_native_checkpoint` persist the model
  plus optionally one native optimizer's state and JSON metadata to
  one explicit pickle-free NPZ archive (a versioned UTF-8/JSON
  manifest plus indexed float64 arrays; `allow_pickle=False` loading;
  no ids, gradients, versions, or graph data serialized), with strict
  full-archive validation before any live mutation, strict optimizer
  presence/type matching, atomic temporary-file replacement,
  deterministic bit-identical file resume
  (`examples/native_checkpoint_resume.py`), and no scheduler or
  random-state capture and no `map_location`. **v3.15 — Phase C
  guardrails and completion — is complete, closing Phase C in code**:
  a cross-cutting completion test suite (`tests/test_native_phase_c.py`)
  locks the integrated invariants that span several components — full
  NativeSGD and NativeAdam training lifecycles under a NumPy tripwire,
  the shared-parameter story end to end (registration → backward
  accumulation → optimizers → snapshots → checkpoints), mixed
  active/frozen/`grad=None`/zero-gradient collections, late parameter
  activation, repeated optimizer-state and checkpoint-resume cycles,
  failure recovery at every boundary, the four-way graph-staleness
  distinction, lifetime/close discipline, and the public surface — plus
  documentation completion, the finalized support matrix, and
  build/CI/hygiene verification, with **no new numerical behavior**.
  Phase C is **complete**; the intended sequence continues with
  the native CNN stack, the CUDA runtime, dtype/AMP work,
  Transformer/text experiments, distributed training, and the final
  portfolio release. CUDA/GPU experiments are still entirely future
  work. The Python framework stays the reference implementation.
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
  - **Phase C — native training stack (complete).** **v3.1 —
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
    rules). **v3.2 — NativeModule Core and Recursive Registration — is
    complete**: `NativeModule` with automatic assignment registration
    (one category per name, latest-assignment-wins collisions, `None`
    unregistering, ordinary attributes for everything that is not a
    `NativeParameter`/`NativeModule`), explicit
    `register_parameter`/`add_module` mirroring assignment, recursive
    `parameters()`/`named_parameters()`/`modules()`/`named_modules()`
    with deterministic depth-first order, identity deduplication,
    first-discovered canonical dotted names, shared-structure and cycle
    safety, recursive `zero_grad()`, and bool-validated
    `train()`/`eval()` propagation. **v3.3 — Native State Dictionary
    Contract — is complete**: in-memory, parameters-only
    `state_dict()`/`load_state_dict()` — canonical deterministic dotted
    keys, independent owning graph-free snapshot values (native copy
    path, no NumPy), strict/non-strict loading with an immutable
    missing/unexpected-keys result, exact shape/dtype/device validation,
    stage-then-commit atomicity with rollback, and full preservation of
    parameter identity, shared aliases, gradients, `requires_grad`, and
    training state — no file serialization, checkpoints, or optimizer
    state yet. **v3.4 — NativeLinear — is complete**: the first concrete
    native layer — `(in_features, out_features)` weight orientation,
    optional `(out_features,)` bias, deterministic seeded fan-in uniform
    initialization (local generator, global random state untouched), validated
    constructor and strictly 2-D input contract, forward as pure
    existing native `matmul` + broadcast `add` so the existing autograd
    is the backward implementation (exact analytical and
    finite-difference verified), frozen-parameter support, deterministic
    `["weight", "bias"]` keys, and full v3.3 state-dict compatibility —
    no losses, optimizers, containers, activations, or training loop
    yet. **v3.5 — NativeReLU and NativeSequential — is complete**: the
    parameter-free shape-generic `NativeReLU` over the existing native
    relu autograd (no in-place mode), and the `NativeSequential`
    ordered container — contiguous integer-string execution slots with
    the registration funnel enforcing that registered children and
    execution order never diverge (position-preserving replacement,
    contiguous `append`, rejection of gaps, non-slot names, direct
    parameters, slot removal, and self-insertion), a minimal
    `len`/`iter`/indexing/`append` surface, position-based execution
    with identity-deduplicated traversal and state for shared children,
    empty-sequence identity forward, nested composition, and exact plus
    finite-difference verified backward through a full
    Linear→ReLU→Linear model. **v3.6 — NativeMSELoss — is complete**:
    the first native loss — a parameter-free NativeModule composing
    native `subtract`/`multiply`/`mean`/`sum` into a scalar loss (mean
    default, sum the only alternative; exact string validation), with
    strict exact-shape/no-broadcasting and dtype/device validation
    before any graph construction, gradients supplied entirely by the
    existing autograd (duplicate-parent factor 2, subtract's target
    sign, the existing native mean backward's `1/N` — no division and
    no manual backward), empty state, train/eval independence, and
    exact plus finite-difference verification for both operands under
    both reductions and through a full Linear→ReLU→Linear→MSE model.
    **v3.7 — Native Parameter Mutation Safety and Versioning Contract —
    is complete**: a read-only monotonic value version on every
    NativeParameter counting replacements of the owned value, the
    controlled no-grad `copy_value_` mutation primitive (identity,
    gradients, `requires_grad`, and registrations preserved; native
    never-aliased owning copies; atomic failure behavior), state
    loading incrementing each matched canonical parameter exactly once
    after its atomic commit (rollback restores values and versions),
    forward-time expected-version capture on the value-sensitive
    operations (`multiply`/`matmul`/`relu` — the audited set whose
    backward reads direct-parent forward values), and a deterministic
    stale-graph backward error raised before any callback or gradient
    commit — while value-independent graphs (add/subtract/reductions/
    views) stay valid across mutation with correct gradients; shared
    parameters expose one version through every alias. **v3.8 —
    NativeSGD — is complete**: the first native optimizer — minimal
    SGD (`value ← value - lr * grad`) over identity-deduplicated open
    NativeParameter objects stored by strong reference in
    first-occurrence order (duplicate references and shared aliases:
    one entry, one update, one version increment per step), a strictly
    validated learning rate (real, non-bool, finite, strictly
    positive), and a two-phase mutation-atomic `step()` — preflight,
    frozen/`grad=None` skipping, exact gradient validation, graph-free
    native staging at the core level, and commits through the v3.7
    `copy_value_` path with gradients retained until a preflighted
    `zero_grad()` — verified through a one-step
    Sequential/Linear/ReLU/MSE integration; no momentum, weight decay,
    parameter groups, optimizer state, schedulers, or training loop.
    **v3.9 — the native MLP training proof — is complete**: the first
    complete multi-iteration native CPU training run, as an example
    plus integration tests with zero changes to the stack —
    `examples/native_mlp_training.py` trains
    `NativeSequential(NativeLinear(2, 8, seed=0), NativeReLU(),
    NativeLinear(8, 1, seed=1))` on 8 fixed synthetic regression
    samples for 25 steps of `NativeSGD(lr=0.1)`, with a completely
    fresh graph each iteration (no retained graphs — the v3.7 stale
    guard never fires in the loop, and deliberate retention across a
    step still raises), gradients confirmed present after backward,
    retained through `step()`, and cleared by `zero_grad()`, exactly
    one version increment per parameter per step, stable parameter
    identities/names/state keys, explicit per-iteration and
    end-of-run tensor release, a NumPy-compute tripwire over a full
    run, and a monotonic deterministic loss trajectory (2.107864 →
    0.009529, a 99.5% reduction) that repeats bit-identically.
    **v3.10 — the integration checkpoint — is complete**: no numerical
    changes — the canonical support matrix, corrected README/summary/
    architecture docs, documentation/export guardrails, and CI and
    hygiene audits, marking the first major usable native training
    checkpoint and PR readiness. **v3.11 — native optimizer math
    primitives — is complete**: native differentiable `sqrt` and
    `reciprocal` (kernels → bindings → core → wrapper → autograd),
    saved-forward-result backwards that record no parameter versions,
    IEEE float64 exceptional-value semantics, arbitrary strided/offset
    view support with fresh owning contiguous outputs, and finite-
    difference-verified gradients — the reusable math for the native
    adaptive optimizer, with general division still deliberately
    unshipped. **v3.12 — NativeAdam — is complete**: the native
    adaptive optimizer over the v3.7 mutation contract and the v3.11
    primitives — the NativeSGD parameter contract unchanged
    (identity-deduplicated open NativeParameters, position-named
    errors), strictly validated `lr`/`betas`/`eps`, eagerly allocated
    optimizer-owned graph-free moment buffers and per-parameter step
    counters (skipped parameters never age; late activation starts at
    t = 1; shared aliases advance once), bias-corrected updates staged
    entirely at the autograd-unaware core level and committed through
    `copy_value_` (one version increment per update, old moments
    closed only after their replacements are installed), preflighted
    `zero_grad()`, mutation-atomic public failure behavior with the
    documented asynchronous-interruption windows, and an explicit
    idempotent `close()` for the optimizer-owned state — with no
    weight decay, AMSGrad, parameter groups, or schedulers.
    **v3.13 — the native optimizer state contract — is complete**:
    `state_dict()`/`load_state_dict()` on NativeSGD and NativeAdam —
    a shared format-1 schema with an exact optimizer type tag,
    validated hyperparameters, and ordered positional
    shape/dtype/device parameter metadata (mapping across instances
    is positional over the deterministic identity-deduplicated
    parameter order; no object ids, names, parameter values,
    gradients, or graph data are serialized); NativeAdam adds
    per-parameter step counts plus caller-owned independent
    graph-free NativeTensor moment snapshots; loading is
    validate → stage → commit with exact validation (no casting,
    reshaping, broadcasting, or device movement), independent
    optimizer-owned copies of every input moment (caller state
    read-only, never adopted or consumed), replaced internal buffers
    closed only after installation, and mutation-atomic ordinary
    failures — never touching parameter values, versions, gradients,
    registrations, or retained graphs, with deterministic in-memory
    continuation proven against an uninterrupted run.
    **v3.14 — native checkpointing and deterministic file resume — is
    complete**: `save_native_checkpoint`/`load_native_checkpoint`
    over the existing state contracts — one pickle-free NPZ archive
    per checkpoint (format `"tensorforge.native_checkpoint"`,
    version 1) holding a UTF-8/JSON uint8 manifest (canonical model
    keys and positional optimizer metadata mapped explicitly to
    deterministic indexed float64 array names; user metadata
    included; nothing volatile serialized) plus the model parameter
    and optimizer moment arrays; validated save with every snapshot closed in a
    `finally` and an atomic collision-safe temporary-file
    `os.replace` (existing destinations survive failures, no
    temporary residue); validate → stage → commit loading under
    `allow_pickle=False` with strict optimizer presence/type
    matching, full pre-mutation validation of thirty-plus corruption
    cases, commits only through the existing module/optimizer
    loaders (model versions +1 each and retained sensitive graphs
    stale, per the existing contracts; optimizer loading moves no
    versions), deterministic bit-identical file resume for NativeAdam
    and next-step equivalence for NativeSGD, and a focused
    resume example. No scheduler state, random-state
    capture/restoration, `map_location`, partial loading, merging,
    sharding, compression, or encryption.
    **v3.15 — Phase C guardrails and completion — is complete,
    closing Phase C**: a cross-cutting completion test file
    (`tests/test_native_phase_c.py`) proving the components compose
    correctly under normal training, shared/frozen/`grad=None`/
    zero-gradient parameters, late activation, repeated snapshot/load
    and checkpoint-resume cycles, failure and corruption at every
    boundary, explicit native lifetime management, and the four-way
    stale-graph distinction; documentation completion and support-
    matrix finalization; and build/CI/hygiene verification — no new
    numerical behavior.
  - **Phase D — native CNN stack — COMPLETE (milestones D0–D12).** Every
    CNN layer — flatten, convolution, and max-pooling, operations and
    modules alike — has shipped; the deterministic native CNN training +
    checkpoint-resume proof runs end to end; and the phase closed with
    cross-cutting integration tests, honest CNN benchmarks, and ASan/UBSan
    validation. The **D0 architecture contract is written** —
    [native_cnn_design.md](native_cnn_design.md) locks the layouts
    (NCHW activations, OIHW convolution weights, cross-correlation), the
    argument and output-shape contracts, the non-contiguous-input policy
    (copy-then-compute at the wrapper), the fused-primitive/autograd
    ownership split, the max-pool winner-index representation, the C ABI
    families and C++/Python source organization, the full test and
    benchmark strategy, and the **D0–D12 milestone sequence**
    (`NativeFlatten`; native convolution forward and its input/weight/
    bias gradients; the convolution module; native max-pooling forward,
    backward, and module; and a deterministic native CNN
    training/checkpoint-resume proof). **D1 has shipped:** `NativeFlatten`,
    a parameter-free, buffer-free batch-preserving flatten
    Python-composed from the existing `reshape`/`contiguous_copy`
    operations and their autograd (no new kernel, no custom backward),
    returning an independent owning result so it composes safely in a
    `NativeSequential`. **D2 has shipped** the first native convolution
    code: an **internal** CPU float64 forward compute kernel
    (`tf::conv2d_forward_contiguous` — direct nested-loop
    cross-correlation, symmetric zero padding, optional bias), verified
    by a dependency-free C++ CTest binary against hand-computed cases and
    stable-framework parity. **D3 has shipped** the forward-only
    convolution *layer*: the exported, exception-guarded C ABI wrapper
    `tf_core_conv2d_forward` (self-validating, contiguous-only), its
    ctypes/`errcheck` registration, and `NativeTensorCore.conv2d_forward`
    — a Python-reachable, autograd-unaware Core method that validates
    shapes, computes the output shape in overflow-safe Python ints,
    copies non-contiguous operands (Policy B), and returns a fresh owning
    contiguous NCHW output matching the stable convolution to tolerance.
    **D4 has shipped** the **internal** CPU float64 convolution
    input-gradient compute kernel (`tf::conv2d_input_backward_contiguous`
    — a hidden C++ symbol: the deterministic scatter-add adjoint of the
    forward cross-correlation, zero-initializing its own output, verified
    by a dependency-free C++ CTest against hand-computed cases, stable
    parity, and central finite differences). Like D2 it is deliberately
    **not exposed to Python** — the exported backward C ABI wrapper, its
    Core method, and the autograd node are D6. **D5 has shipped** the
    **internal** CPU float64 convolution weight-gradient compute kernel
    (`tf::conv2d_weight_backward_contiguous` — a hidden C++ symbol,
    deterministic zero-initialized accumulation, verified against
    hand-computed cases, an explicit-zero padded-materialization oracle,
    stable parity, and central finite differences) and **locked and
    validated the bias-gradient path as a reuse of the existing native
    `sum` reduction** (`g.sum(0).sum(1).sum(1) → (O,)`, no dedicated
    kernel), proved in a focused Python contract test. **D6 completed the
    differentiable native convolution operation**: the exported guarded
    backward C ABI wrappers (`tf_core_conv2d_input_backward`,
    `tf_core_conv2d_weight_backward`), the Core backward methods, the bias
    gradient composed from the existing native `sum` reduction (no dedicated
    kernel), and the Python-managed **`NativeTensor.conv2d`** autograd
    primitive — forward reuse of the D3 wrapper, input/weight/bias
    gradients, deterministic `(input, weight[, bias])` parent ordering,
    conditional stale-value version tracking, and reuse of the existing
    backward snapshot/rollback engine — verified against stable parity,
    finite differences, and all `requires_grad` combinations. **D7 completed
    the trainable native convolution module**: an OIHW weight / optional
    `(O,)` bias native-parameter layer with deterministic uniform conv
    fan-in initialization (`bound = 1/sqrt(in_channels·kh·kw)`, a local
    generator with the global state untouched), 4-D NCHW input validation,
    and backward
    supplied entirely by the D6 `conv2d` autograd — no new kernel, C ABI
    symbol, or custom module backward. It registers in `NATIVE_MODULES`,
    exports from `tensorforge.experimental`, and rides the existing
    state_dict/checkpoint/optimizer paths unchanged. **D8 has shipped the
    forward-only native max-pooling layer**: the internal CPU float64
    compute kernel `tf::maxpool2d_forward_contiguous` (a hidden C++ symbol
    that produces the pooled values and the saved winner indices in one
    deterministic row-major pass — padding participates as a conceptual
    `-inf`, ties keep the first occurrence, and a completely padded window
    yields `-inf` with the `-1` sentinel), the exported guarded C ABI
    wrapper `tf_core_maxpool2d_forward` with its ctypes/`errcheck`
    registration, and `NativeTensorCore.maxpool2d_forward` — a
    Python-reachable, autograd-unaware Core method that validates the
    arguments and the `H*W ≤ 2^53` winner-exactness bound in Python ints
    before allocating anything, copies a non-contiguous input (Policy B),
    allocates the output and the **private** winner buffer in a
    failure-atomic order, and matches the stable pooling reference
    exactly. **D9 completed the differentiable native pooling
    operation**: the internal scatter-add kernel
    `tf::maxpool2d_backward_contiguous`, the exported guarded
    `tf_core_maxpool2d_backward` wrapper (which validates every saved
    winner — the sentinel or an exact in-range integer — before scattering,
    and never rounds), `NativeTensorCore.maxpool2d_backward`, and the
    Python-managed **`NativeTensor.maxpool2d`** autograd node. Its single
    input-gradient callback routes the upstream through the winners the
    forward saved — never rereading the input, never recomputing a maximum,
    and recording **no** parameter-version snapshot (a deliberate contrast
    with convolution) — with overlapping windows accumulating and padding
    winners dropped. The private winner buffer became graph-owned state
    released exactly when the graph history is (freed by a one-shot
    backward or `close()`, retained under `retain_graph=True`, and kept
    alive across a failed retryable backward). **D10 completed the native
    pooling layer**: a parameter-free, buffer-free module that normalizes
    its window arguments to `(height, width)` tuples (no stride means
    non-overlapping windows) and delegates its forward entirely to that
    operation — no new kernel, C ABI symbol, custom backward, parameters,
    buffers, or checkpoint schema, and no winner state held between calls.
    It exports from `tensorforge.experimental`, contributes no
    state-dictionary keys, and composes in a `NativeSequential` beside the
    convolution, activation, flatten, and linear layers, so the native
    optimizers ignore it naturally. **D11 proved the whole stack trains
    end to end**: `examples/native_cnn_training.py` learns a genuinely
    spatial target — the strongest bright-to-dark vertical edge of eight
    fixed 6×6 images — through convolution, activation, pooling, flatten,
    and a linear head with the native MSE loss and the native adaptive
    optimizer, dropping the loss from about 0.7713 to about 0.0111 in 40
    deterministic steps; and a run interrupted at step 15, checkpointed
    with its optimizer state and resumed into a completely fresh
    model/optimizer pair, reproduces the uninterrupted run **exactly**
    (loss history, final predictions, every parameter value, and every
    optimizer state entry), adding no kernel, operation, loss, optimizer,
    or checkpoint schema. **D12 closed the phase**: cross-cutting
    integration tests spanning several CNN components at once
    (`tests/test_native_phase_d.py`), honest CNN characterization
    benchmarks (`benchmarks/benchmark_native_cnn.py` — measurement only,
    no speed claims), **ASan/UBSan validation** of the whole native CNN
    stack under Clang on Linux with no TensorForge diagnostic, a
    LeakSanitizer pass over the instrumented native CTests, documentation
    reconciliation across every status surface, and the replacement of the
    milestone-era doc guardrails with durable semantic checks. See the
    [support matrix](native_support_matrix.md) for the finalized status.
  - **Phase E — the next native phase (not started).** The natural
    continuations, in no committed order: a native classification stack
    (softmax/cross-entropy and its metrics), more native activations and
    math, native normalization, a native RNG and dropout, a CPU
    optimization phase for the deliberately naive kernels, and
    build/packaging evolution. None of it exists today.
  - **Then beyond (not started):** the CUDA
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
