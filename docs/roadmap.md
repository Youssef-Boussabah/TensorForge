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
- **The Daedalus-class native roadmap** — the longer arc the explicit
  experimental native line (`tensorforge.backends`,
  `tensorforge.experimental`) is building toward, in phases, each landing
  only when the previous is tested and documented:
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
  - **Phase E — Native Classification and Stable Math — complete
    (E0-E10).** The **E0 architecture contract
    is written** —
    [native_classification_design.md](native_classification_design.md)
    locks the scope, the public API surface (`exp`, `log`, `softmax`,
    `log_softmax`, a fused `cross_entropy` from raw logits,
    `NativeCrossEntropyLoss`, and a reporting-only `native_accuracy`), the
    numerical-stability strategy (maximum shift and log-sum-exp, never
    `softmax().log()`), the backward-read and versioning matrix (`log` is
    the one version-checked operation; everything else reads saved state),
    the `int64` target contract (the native runtime has no integer dtype),
    the graph-owned saved-probability lifetime, the contiguous-only C ABI
    families and the new `cpp/src/classification.cpp` unit, the capability
    inventory placements, the unchanged checkpoint format version 1, and
    the **E0–E10 milestone sequence**. **E0 added no numerical behavior**
    — it was a design-and-reconciliation milestone. **E1 has shipped the
    native exponential**: the C++ kernel on both the strided-odometer and
    contiguous execution paths, the two guarded C ABI exports
    (`tf_core_exp`/`tf_core_exp_contiguous`, which validate their own
    handles, layout metadata, spans, and overflow), their ctypes
    registration, `NativeTensorCore.exp()`, and the differentiable
    `NativeTensor.exp()` whose backward is `upstream ×` the **saved
    forward output** — never rereading the input, so it records **no**
    parameter-version snapshot and survives post-forward mutation, with
    plain IEEE semantics (no clamping, no inserted bound). **E2 has
    shipped the native logarithm** through the same four layers, reusing
    E1's self-validating export contract unchanged: plain IEEE
    `std::log` (`log(±0)` is `-inf`, `log(negative)` is NaN — values, not
    errors), with a backward that **rereads the live input** as
    `upstream × reciprocal(x)` (composed from the existing `reciprocal`;
    no division operation was added). That makes a direct
    `NativeParameter` parent **version-checked**: mutating it after
    forward raises the deterministic stale-graph error before any
    gradient is committed anywhere in the graph — the deliberate
    counterpart to `exp`'s saved-output edge, which stays valid across
    the same mutation. **E3 has shipped the stable native softmax**, the
    phase's first fused probability transform and the reason
    `cpp/src/classification.cpp` now exists: a maximum-shift kernel
    computing `exp(x - max(x)) / sum(exp(x - max(x)))` in one pass over
    any single axis (positive or negative, rank >= 1), behind a
    **contiguous-only** C ABI with the Core layer applying the Phase-D
    Policy-B copy-then-compute for strided views. Its backward is the
    closed-form `y * (upstream - sum(upstream * y, axis, keepdims))`
    **composed from existing Core operations** — no dedicated backward
    kernel — reading only the saved probabilities, so it records no
    parameter version. E3 added no `NativeSoftmax` module and no public
    `max`, `argmax`, or division. **E4 has shipped the stable native
    log-softmax**, the phase's second fused probability transform: its
    **own** log-sum-exp kernel computing
    `(x - max(x)) - log(sum(exp(x - max(x))))` in one pass over any
    single axis, **never** `softmax().log()` — no probability buffer is
    formed and no division happens, so a probability too small to
    represent (which the composed form would round to 0 and report as
    `-inf`) still gets an accurate finite log-probability. It reuses E3's
    contiguous-only C ABI shape, its trust-boundary validator (now shared
    by both exports), and the same Core-level Policy-B copy-then-compute.
    Its backward is the closed-form
    `upstream - exp(y) * sum(upstream, axis, keepdims)` **composed from
    existing Core operations** — no backward kernel; `exp(y)` recovers
    the probabilities from the saved log probabilities — so it too reads
    only the saved output and records no parameter version. E4 added no
    `NativeLogSoftmax` module and no `NLLLoss`. **E5 has shipped the
    fused cross-entropy Core contract** — and *only* the Core layer. Two
    new kernels in the same classification unit: a forward that, in one
    deterministic pass per row, computes the maximum, the log-sum-exp,
    the **saved probabilities**, and the per-example loss
    (`log(Σ exp(x − m)) − (x[target] − m)`, reduced by `"sum"` or once by
    the batch size for `"mean"`) — never `-log(p[target])`, never
    `softmax().log()`-then-index, never `log_softmax()`-then-gather — and
    a backward that turns those **saved probabilities**, the copied
    targets, the reduction, and a **native one-element upstream** into
    `upstream · (p − onehot) / N` **without ever rereading the logits**,
    which are not even an argument. Both guarded exports
    (`tf_core_cross_entropy_forward`/`tf_core_cross_entropy_backward`)
    are contiguous-only for tensor data, revalidate **every target index**
    themselves rather than trusting Python, and leave every destination
    byte-for-byte unchanged when they reject. Targets are not native
    tensors (the runtime has no integer dtype): they are strictly
    validated — `bool` and floating-point labels rejected outright,
    including integral ones like `1.0` — and copied into independently
    owned contiguous read-only `int64` metadata, so mutating the caller's
    list or array afterwards cannot reach the kernel. The forward's two
    outputs fail atomically: any failure closes everything it allocated
    and returns no partial result. **E5 added no
    `NativeTensor.cross_entropy`, no autograd node, and no graph-owned
    saved state** — the private probabilities were Core-level state the
    caller owned. **E6 has shipped that differentiable operation**:
    `NativeTensor.cross_entropy(targets, reduction="mean")`, a single
    autograd node over the E5 contract that adds **no kernel, no C ABI
    export, and no numerical change**. It calls the E5 forward once,
    returns a **scalar** `NativeTensor`, and adopts the private saved
    probabilities as **graph-owned state** through the same
    `_from_op(..., graph_resources=...)` contract the Phase-D pooling
    winner buffer established: retained under `retain_graph=True` and across a
    failed retryable backward, released exactly once when the graph
    history is (a one-shot `backward()` or `close()`), and closed
    immediately when nothing requires gradients. The copied `int64`
    targets ride in the backward closure as immutable metadata — no
    native integer tensor — so caller mutation after the forward cannot
    reach the gradient. Because backward consumes only that saved state
    and a native scalar upstream, **it never rereads the logits** and the
    node records **no expected parameter version**: mutating a direct
    `NativeParameter` logits parent with `copy_value_` afterwards neither
    raises a stale-graph error nor changes the gradient, even across a
    retained graph — the `maxpool2d` archetype, and the deliberate
    contrast with `log`. Failures are atomic throughout: an E5 forward
    failure returns no tensor and builds no node, a graph-construction
    failure closes both E5 outputs, and a backward failure commits no
    gradient, leaks no gradient core, keeps the probabilities for a
    retry, and leaves the graph honestly un-freed. **E7 has shipped the
    public classification surface** and, like E6, added no training
    mathematics. ``NativeCrossEntropyLoss(reduction="mean")`` is a
    parameter-free, buffer-free ``NativeModule`` whose entire forward is
    ``logits.cross_entropy(targets, reduction=self.reduction)`` — no Core
    call, no ABI call, no NumPy, no ``softmax``/``log_softmax``
    composition, and no second formula — so it inherits every E5/E6
    guarantee rather than restating any of them, validates its
    ``"mean"``/``"sum"`` reduction in the constructor with the
    operation's own validator, and contributes no ``state_dict()`` or
    checkpoint keys (the reduction is constructor configuration, not
    model state). ``native_accuracy(logits, targets) -> float`` is a
    deliberately **reporting-only** helper, and the honesty of that label
    is the point: there is no accuracy kernel, no C ABI export, no Core
    method, no autograd node, and no native ``argmax`` (the runtime has
    no integer dtype for one to return). It validates rank-2 logits and
    targets through the *same* private preparer the cross-entropy forward
    uses — so the strict accepted/rejected matrix is identical at both
    call sites by construction — then materializes the logits **once**
    through the explicit public ``to_numpy()`` boundary, takes
    ``numpy.argmax(axis=1)`` (ties to the first maximal index), and
    returns a plain ``float`` in ``[0.0, 1.0]``. It builds no graph,
    touches no gradient, parameter, or version, allocates no native
    storage at all, and retains nothing, so a graph built before the call
    is still usable after it. The two capabilities land in the two
    inventories that describe their layers — ``NATIVE_LOSSES`` and the
    new ``NATIVE_METRICS``, reported by ``backend_info()`` — and with
    that, no classification name remains listed as unsupported.
    **E8 has shipped the end-to-end proof**, and it too added no
    numerical operation or runtime capability — it is Python example,
    integration tests, and documentation only.
    ``examples/native_classification_training.py`` trains a
    ``NativeConv2d(1, 4, 3, seed=0)`` → ``NativeReLU`` →
    ``NativeMaxPool2d(2)`` → ``NativeFlatten`` → ``NativeLinear(16, 3,
    seed=1)`` classifier — a named ``NativeModule`` whose children are
    registered through the ordinary assignment path — on twelve fixed
    6×6 single-channel images in **three** classes (vertical bar,
    horizontal bar, diagonal line; four positions each, committed as
    source literals, labels host integers, nothing generated,
    downloaded, augmented, or shuffled). Its **raw logits** go straight
    to ``NativeCrossEntropyLoss`` — there is deliberately no softmax or
    log-softmax layer, because the fused E5/E6 kernel is what keeps the
    loss stable — and 40 full-batch ``NativeAdam(lr=0.05)`` steps take
    the loss from **1.159638 to 0.000101** (99.99%) and the reporting
    accuracy from **0.3333 to 1.0000**, with ``native_accuracy`` called
    only outside the training mathematics (it converts to the host on
    purpose). Interrupting at step **15**, checkpointing model **and**
    optimizer state through the existing pickle-free path (format
    **version 1**, no new keys, no graph data or target metadata
    serialized), and resuming into a **fresh** model/optimizer pair
    reproduces the uninterrupted run **exactly**: the whole remaining
    loss suffix, every parameter, both ``NativeAdam`` moment buffers,
    every step
    counter, the final logits, the predictions, and the accuracy. Two
    independent uninterrupted runs are exactly equal too, repeated steps
    retain no completed graph or saved probability and grow no native
    storage, and a tripwire proves one complete step reaches no NumPy
    numerical routine and converts no tensor data. It is an
    **integration proof on one fixed task** — not a benchmark, not a
    speed claim, and not a generalization claim.
    **E9 has shipped the honest characterization benchmark**, and it
    changed no numerical runtime file and tuned nothing.
    ``benchmarks/benchmark_native_classification.py`` measures the seven
    operations the phase built — ``exp``, ``log``, ``softmax``,
    ``log_softmax``, the fused cross-entropy forward, its backward alone
    (a fresh graph is built outside the timer every repetition; no graph
    is reused and ``retain_graph`` is never used to skip the rebuild),
    and one complete classification training step (``zero_grad`` →
    forward → loss → ``backward`` → ``NativeAdam.step()``, with model,
    optimizer, and dataset construction, checkpoint I/O,
    ``native_accuracy``, and cleanup all outside the timed region).
    **Correctness is gated before every measurement**: a case validates
    shape, finiteness, reference parity, and input non-mutation — plus
    gradients for the backward case, and a finite loss, a real parameter
    update, an advanced optimizer step counter, a released graph, closed
    transients, and stable-line parity for the training step — and a
    failed gate exits nonzero and publishes no timing. Each case is
    labelled with the reference it actually used: ``stable_tensorforge``
    where a stable operation exists, ``numpy`` for ``log_softmax``
    (the stable line has no direct one, and ``softmax().log()`` is
    deliberately not used as the reference), and ``native_only`` where no
    honest analogue would exist. Timing is ``time.perf_counter_ns`` with
    warm-up, repeated measurements, setup and cleanup outside the timer,
    and **median** reporting alongside min, max, and spread; ``--smoke``
    and ``--json`` modes exist and no result file is written. The
    observed ratio is a **local characterization**, never a speedup
    claim: no test asserts a speed, no timing number is committed as a
    promise, and there is no CI performance gate anywhere.
    **E10 closed the phase**, adding no numerical capability of any kind:
    cross-cutting integration tests (``tests/test_native_phase_e.py``)
    covering the classification stack as one system, **Release and Debug**
    native builds (10/10 CTests each, zero compiler warnings), Clang
    AddressSanitizer and UndefinedBehaviorSanitizer validation of the
    whole classification stack (zero diagnostics attributable to
    TensorForge), a practical LeakSanitizer pass finding **no** native
    leak with the live-storage counters returning to baseline, the
    complete Python regression suite, the conversion of milestone-era
    "not yet shipped" documentation guardrails into durable semantic
    ones, and reconciliation of every authoritative status surface.
    **Phase E is complete**, and it expanded nothing beyond float64/CPU
    and added no implicit stable/native dispatch. Deliberately outside
    Phase E: more native activations beyond it, a native RNG and dropout,
    a CPU optimization phase for the deliberately naive kernels, and
    build/packaging evolution. Native normalization then became its own
    phase, below.
  - **Phase F — Native Normalization and Stateful Buffers — complete
    (F0–F9 all shipped).** The **F0
    architecture contract is written** —
    [native_normalization_design.md](native_normalization_design.md)
    locks the phase's objective (a fully native, differentiable,
    state-safe normalization stack: `NativeLayerNorm`,
    `NativeBatchNorm1d`, and `NativeBatchNorm2d`), the public API and its
    naming (layer-norm `weight`/`bias`; batch-norm `gamma`/`beta` plus
    the `running_mean`/`running_var` buffers, matching the stable
    reference), and — most consequentially — the decision that
    normalization is **composed from existing native operations**
    (`mean`, `subtract`, `multiply`, `add`, `sqrt`, `reciprocal`,
    `reshape`, broadcasting, `contiguous_copy`) so the phase adds **no
    C++ kernel, no C ABI export, no ctypes declaration, and no
    `NativeTensorCore` method** and inherits an exact backward from the
    existing autograd — including differentiation through the batch mean
    and variance, which is never detached. It also locks the layer-norm
    contract (trailing-dimension normalization, population variance,
    `eps` inside the square root, no buffers, identical in train and eval
    mode), the two batch-norm shape contracts (`(N, C)` reducing over the
    batch; NCHW `(N, C, H, W)` reducing over N/H/W with `(1, C, 1, 1)`
    broadcasting), and three load-bearing safety rules: a **live mutable
    running buffer is never captured as a rereadable graph operand** (an
    eval forward takes independent, owning, graph-free snapshots, which
    is precisely why buffers need no value version — `multiply`'s
    backward rereads a live operand, and the existing stale-version check
    only covers direct `NativeParameter` parents); the two running
    buffers update as **one atomic transaction** (validate, stage
    graph-free values, commit preserving both buffers' Python identity,
    roll back completely on any failure or interruption, close replaced
    cores exactly once, move no parameter version); and **registration
    implies no exclusive ownership**, so stateful examples and tests
    close both `parameters()` and `buffers()` explicitly and no contract
    relies on garbage collection. Persistent running statistics ride the
    **existing** state-dictionary and pickle-free checkpoint
    infrastructure with the format **unchanged at version 1** — new
    persistent keys need no schema bump — and the eventual exact resume
    must reproduce the loss suffix, parameters, optimizer state, running
    means, running variances, final predictions, **and the
    evaluation-mode output**. The ladder is **F0–F9**: F0 (this
    contract and repository reconciliation), F1 (atomic native-buffer
    state transactions, extracted and generalized from the existing
    `load_state_dict` staging/commit/rollback, plus the `STATE_SUPPORT`
    persistent-buffer correction), F2 (`NativeLayerNorm`), F3
    (`NativeBatchNorm1d`), F4 (`NativeBatchNorm2d`), F5 (state,
    checkpoint, and graph-safety hardening), F6 (deterministic
    normalized training and exact resume), F7 (benchmark
    characterization, with no speed assertion), F8 (cross-cutting
    integration and semantic guardrails), and F9 (phase closure).
    **F0 added no numerical behavior** — it is a design-and-reconciliation
    milestone — and **F1 is complete**: the private atomic native-buffer
    state transaction (`_native_state.py`) that §8 of the contract calls
    for, now the single implementation behind
    `NativeModule.load_state_dict` (whose public behavior is unchanged),
    plus the `persistent_buffers` correction to `STATE_SUPPORT` — an
    under-reported capability that has existed since before Phase D. F1
    is state management and capability reporting only and added **no
    normalization mathematics**. **F2 is complete**: `NativeLayerNorm`,
    the first native normalization module — stateless (no buffers,
    identical in train and eval), differentiable through the mean and the
    population variance, and composed entirely from existing native
    operations (`sqrt(var + eps)`, no Bessel correction) with no kernel,
    ABI symbol, `NativeTensorCore` method, custom backward, or
    `NativeTensor` normalization operation; `"NativeLayerNorm"` is now in
    `NATIVE_MODULES` and the exports, and `"layernorm"` has left
    `UNSUPPORTED`. **F3 is complete**: `NativeBatchNorm1d`, the **first
    stateful native numerical module** — `(N, C)` batch normalization
    whose training statistics are differentiable (gradients flow through
    the batch mean *and* the population variance, never detached), whose
    `running_mean`/`running_var` are **persistent native buffers**
    advanced by `(1 − momentum)·running + momentum·batch` from the *same*
    batch statistics, computed graph-free and committed as one **atomic
    two-buffer transaction** through the F1 primitive (both identities
    preserved, replaced cores closed exactly once, no parameter version
    moved), and whose evaluation mode reads **independent owning
    graph-free snapshots** of those buffers, so a later training step, or a
    buffer-only `load_state_dict()`/`load_native_checkpoint()`, cannot
    change an earlier eval graph's gradient (a full checkpoint load that
    also replaces `gamma`/`beta` still stales that graph through the
    unchanged parameter-version rule — correct, and proved separately). It too is composed from existing native operations
    — no kernel, C ABI symbol, ctypes declaration, `NativeTensorCore`
    method, custom backward, or `NativeTensor.batch_norm` operation — and
    the native checkpoint format stays at version 1;
    `"NativeBatchNorm1d"` is now in `NATIVE_MODULES` and the exports.
    **F4 is complete**: `NativeBatchNorm2d`, NCHW `(N, C, H, W)` batch
    normalization reducing over N, H, and W — one population mean and
    variance per channel over `N * H * W` values — built on the **same**
    shared private implementation as the 1-D shape, which it extends with
    nothing but its rank, its reduction axes, its `(1, C, 1, 1)`
    broadcast layout, and the channels-last permutation its rank-1
    `gamma`/`beta` need (the activation is transposed for the affine
    application, never the parameters, so the existing direct-parameter
    stale-value guard is preserved exactly). Running buffers stay `(C,)`.
    `"NativeBatchNorm2d"` is now in `NATIVE_MODULES` and the exports,
    and with both shapes live **`batchnorm` has left `UNSUPPORTED`**,
    which now reads exactly `("dropout", "float32", "cuda", "amp")`.
    **The numerical normalization module surface is complete, and F5 has
    hardened it.** **F5 is complete**: the
    exhaustive state/checkpoint, ownership, and graph-safety hardening — a
    focused `tests/test_native_normalization_state.py` plus narrow
    additions to the generic buffer and checkpoint suites — proves §7–§10
    by executable test (canonical dotted buffer keys, independent state
    snapshots, strict/non-strict loads, exact never-casting metadata
    validation, mixed parameter/buffer transaction atomicity, buffer
    identity across state and checkpoint loads, exact eval-output
    reproduction, the buffer-only-versus-full stale-graph distinction, the
    save/corrupt-load failure boundaries, eval-graph snapshot safety under
    `retain_graph` and a failed retryable backward, and the live-storage
    baselines); it is **tests and documentation only** — no numerical
    behavior, no new capability, and the checkpoint format stays version
    1. **F6 is complete**: `examples/native_normalization_training.py`
    trains a `NativeLinear → NativeBatchNorm1d → NativeReLU →
    NativeLayerNorm → NativeLinear` regressor for 24 deterministic
    `NativeAdam` steps with `NativeMSELoss` (98.9% loss reduction), proves
    two uninterrupted runs bit-identical, and resumes an interrupted run
    into a fresh model/optimizer pair that reproduces the remaining losses,
    every parameter, the NativeAdam state, both running statistics
    (`running_mean`/`running_var`), the final training-step prediction, and
    the final evaluation-mode output exactly (format version 1 unchanged,
    training flags runtime-only) — one example and its integration test,
    adding no capability. **F7 is complete**:
    `benchmarks/benchmark_native_normalization.py` characterizes the stack
    with nine cases — the `NativeLayerNorm` forward and backward, the
    `NativeBatchNorm1d` training forward, evaluation forward, and
    backward, the `NativeBatchNorm2d` training forward, evaluation
    forward, and backward, and one complete F6-style normalized training
    step — each **correctness-gated before any timing**, six against
    `stable_tensorforge` equivalents on identical state and three (the
    `NativeBatchNorm2d` shapes) labelled `native_only` because the stable
    line has no public 2-D batch-normalization module to time against,
    though those keep a rigorous NumPy NCHW and transformed-oracle
    correctness gate. Medians are reported with min,
    max, and spread after warm-up; `--smoke` and `--json` modes exist;
    **no result file is written, no speed is asserted, no timing number is
    committed, and no CI timing threshold exists** — measurement only,
    adding no capability. **F8 is complete**:
    `tests/test_native_phase_f.py` proves the cross-cutting interactions —
    one integrated `NativeConv2d → NativeBatchNorm2d → NativeReLU →
    NativeMaxPool2d → NativeFlatten → NativeLinear → NativeBatchNorm1d →
    NativeReLU → NativeLayerNorm → NativeLinear` classifier over raw
    logits and the fused classification loss, trained by `NativeAdam` and
    resumed **exactly** from one version-1 checkpoint (all four
    running-statistic buffers, the final training logits, and the
    evaluation-mode logits, predictions, and accuracy included); the
    three saved-resource families coexisting in one eval graph and
    releasing exactly once; buffer mutation leaving an earlier graph
    valid while parameter mutation correctly stales it; the versioning
    archetypes; shared and frozen parameters; a non-contiguous NCHW
    input; strict stable/native separation; honest per-boundary failure
    atomicity (transactions are per module — one whole training step is
    *not* globally transactional); error-state recovery; the NumPy
    boundary; live-storage baselines; and reality-derived capability
    guardrails — tests and documentation only, adding no capability.
    **F9 is complete**: the phase closure — fresh Windows Release **and**
    Debug builds (Visual Studio 17 2022, MSVC 19.44.35228.0) each passing
    the full existing 10-test CTest suite with zero project warnings, and
    the active runtime proved to remain the Release DLL; a fresh Clang
    18.1.3 `address,undefined` build in WSL2 Ubuntu 24.04 whose
    instrumentation is *proved* (22 `__asan*` and 13 `__ubsan*` dynamic
    symbols; the library will not even load without the sanitizer
    runtime); 10/10 sanitized native CTests with leak detection enabled;
    1,968 sanitized normalization-focused Python tests with **zero ASan
    and zero UBSan diagnostics**; the F6 example reproducing its exact
    resume and the F7 benchmark passing all nine correctness gates under
    the sanitized library; and a practical LeakSanitizer lifecycle whose
    native live-storage counter returned **exactly** to baseline, with the
    remaining process-exit allocations identified honestly as
    CPython/NumPy shutdown retention containing no TensorForge frame and
    no suppression file. It is **validation and documentation only** — no
    numerical capability, no C++, no CTest, no ABI or ctypes surface, no
    example, no benchmark, and no production behavior changed — so
    **Phase F is complete (F0–F9)** and no normalization
    operation or kernel exists at all.
    Deliberately outside Phase F: dropout, a native RNG with its
    checkpoint state, further activations, more losses, schedulers, data
    loaders, native integer tensors, further dtypes or devices, CUDA,
    AMP, fused normalization kernels, and CPU optimization.
  - **Phase G — Native RNG and Dropout — is complete; G0 through G10
    have all landed.** The **G0 architecture contract is written** —
    [native_rng_dropout_design.md](native_rng_dropout_design.md) locks
    the phase's central split (random state is Python-managed; native
    random kernels are stateless and receive the complete key for one
    call), the `NativeGenerator` contract (an explicit unsigned 64-bit
    seed and call counter plus an algorithm identifier and version, no
    native resource and therefore no `close()`, identity equality, and no
    global or process-wide state anywhere), the deterministic
    counter-based algorithm and its known-answer requirements, the
    call-consumption transaction (**one successful stochastic forward
    consumes exactly one call**; a validation, allocation, kernel, or
    graph-construction failure consumes none, and neither does evaluation
    mode, `p == 0`, or backward) and the lock-protected, token-validated
    reservation protocol that carries it (one private lock covering
    reservation, commit, cancellation, and every state read and write,
    with native computation outside it; opaque single-use tokens so a
    stale, foreign, or duplicated commit changes nothing; at most one
    live reservation, so a concurrent or reentrant caller fails **before
    an index is minted** and no two callers can ever receive the same
    call index; and seed or counter replacement refused while a
    reservation is live — serialization for correctness, with parallel
    stochastic execution explicitly not claimed), the probability contract
    (`0 <= p < 1`, with `p == 1` rejected so inverted scaling never
    divides by zero), the stateless forward boundary (one new kernel
    producing the output **and** a private multiplier mask, no backward
    kernel, logical-order element indexing independent of physical
    strides), the differentiable operation whose backward reads **only**
    that graph-owned mask — never the input, never the generator — the
    module surface, generator registration as a fourth `NativeModule`
    state category, native checkpoint **version 2** whose generator
    section records the **alias topology** — every registered generator
    path and its canonical target, so shared-versus-independent generator
    identity is restored and not merely the states, with every topology
    mismatch failing in prevalidation before any live state changes — and
    an explicit version-1 compatibility rule that never fabricates a seed
    or counter, **whole-checkpoint transaction atomicity** (validate
    everything, stage everything that can allocate or raise, then commit
    under one rollback guard, so any ordinary synchronous failure — and
    any deliverable asynchronous one, including `KeyboardInterrupt` —
    restores parameters, persistent buffers, optimizer state, and
    generator state together with every object identity intact, leaving
    external process or interpreter death as the only documented
    exception), the ownership and failure matrices, and the **G0–G10**
    milestone sequence. **G0 added no numerical behavior**: it is design,
    documentation, and semantic guardrails only.
    **G1 is complete** — the generator state foundation:
    `NativeGenerator` (pure Python, no native storage, no `close()`; the
    four locked fields as read-only properties; atomic `state()` /
    `load_state()` / `reseed()` / `reset()`; exact-`int` seeds with one
    `secrets` entropy draw for `seed=None`; identity semantics with
    copying and pickling refused; and the lock-protected,
    token-validated reservation transaction — a two-phase claim /
    construct / publish / deliver sequence whose token is allocated with
    **no generator lock held**, so no callback-capable operation ever runs
    while a lock is held and a finalizer cannot invert the global
    multi-generator lock order, with the construction claim refusing every
    conflicting mutation in the meantime and an exact-match cleanup for a
    reservation that was published but never delivered, so a dropped token
    can never strand the generator) plus generators as a
    **fourth** `NativeModule` registration category with deterministic
    identity-deduplicated cycle-safe traversal and their own
    `generator_state_dict()` / `load_generator_state_dict()` surface,
    leaving `state_dict()` tensor-only and unchanged. **G1 generates no
    random values by itself.**
    **G2 is complete** — the deterministic stateless
    native Dropout-forward **Core**: the exact locked `tensorforge.splitmix64` derivation in
    unsigned 64-bit arithmetic (`mix64` finalizer, per-call stream key
    `mix64(seed + GOLDEN*(call_index + 1))`, per-element bits
    `mix64(stream + GOLDEN*(element + 1))`, uniform
    `(bits >> 11) * 2**-53`, dropped when `u < p`) as internal hidden
    `namespace tf` functions in the new `cpp/src/random.cpp` /
    `cpp/include/tf_random_internal.h`; the inverted-dropout float64 CPU
    kernel writing the output **and** the private multiplier mask in one
    pass; the self-validating guarded export `tf_core_dropout_forward`;
    its ctypes declaration carrying the whole key as two `c_uint64`
    arguments; `"dropout_forward"` in `TENSOR_CORE_OPS` and
    `"tf_core_dropout_forward"` in the checked-kernel inventory; the
    public `NativeTensorCore.dropout_forward(p, *, seed, call_index)` and
    the private `_dropout_forward_with_mask` that keeps the mask; a
    dependency-free CTest over both layers; and committed known-answer
    vectors asserted **identically** from C++ and Python. It is
    **stateless**: the complete random key arrives as two explicit
    integers, the Core reserves, commits, cancels, inspects, and mutates
    **no** `NativeGenerator`, and no C++ translation unit holds random
    state of any kind. Randomness is keyed by the **logical** row-major
    element index, so a transposed, narrowed, or nonzero-offset view
    receives the same mask as a contiguous tensor of the same logical
    shape. Both results are fresh owning contiguous cores that alias
    neither the input nor each other, and the two-result boundary is
    failure-atomic in C++ *and* in the Python wrapper.
    **G3 is complete** — the differentiable
    `NativeTensor.dropout(p, *, generator)` over that Core, and the whole
    milestone is one method plus one name, `"dropout"`, appended to
    `AUTOGRAD_OPS`: no C++, no C ABI symbol, no ctypes declaration, no
    `NativeTensorCore` method, no module, no export, and no
    checkpoint-format change. The `generator` is **required and
    keyword-only** — no default, process-global, or module-global stream,
    no implicit per-call generator, and no NumPy or Python `random`
    fallback — and `p` goes through the *same* shared validator the Core
    uses, so the accepted/rejected matrix is identical by construction.
    It owns the §5 call transaction: validate, then reserve one call,
    then run the G2 Core **outside** the generator's lock with the
    reservation's own seed and index (never a reread counter), then build
    the graph, then commit as the **last** state-changing action. So a
    successful stochastic forward consumes exactly one call — with or
    without gradients — while `p == 0` returns the caller's own tensor
    object having reserved, allocated, and consumed nothing, and every
    ordinary failure before the commit releases the result, abandons the
    reservation, and leaves the same unconsumed index for the next
    forward. Backward consumes none, ever. The private multiplier mask
    becomes **graph-owned** state through the unchanged `graph_resources`
    contract — the third member of the family beside the native pooling
    winner buffer and the fused loss's saved probabilities — released
    exactly once at the
    same deterministic points the graph history is, retained under
    `retain_graph=True`, kept alive across a failed retryable backward,
    freed by an abandoned graph's `close()`, and closed immediately by a
    no-grad forward. The backward is `upstream * mask` through the
    existing native `multiply`, so **no dropout backward kernel exists**;
    it never rereads the input, never redraws, and never touches a
    generator, and the node therefore records **no** expected parameter
    version, so mutating the input or reseeding, resetting, or reloading
    the generator afterwards cannot change an existing graph's gradient
    and must not raise.
    **G4 is complete** — `NativeDropout(p=0.5, seed=None,
    generator=None)`, the public module over that operation, plus its
    experimental export and one name (`"NativeDropout"`) appended to
    `NATIVE_MODULES`. Nothing else moved: no C++, no C ABI symbol, no
    ctypes declaration, no Core method, no autograd operation, and no
    checkpoint-format change. `p` goes through the *same* shared
    validator the Core and the operation use; `seed` and `generator` are
    **mutually exclusive**, so supplying both raises rather than quietly
    ignoring one; and the module either creates and owns a generator or
    registers the **exact** object supplied, never a copy — which is how
    two layers share one interleaved stream while the default gives every
    layer an independent one. The generator is first-class registered
    state (in `generators()`, `named_generators()`, and
    `generator_state_dict()`, and deliberately absent from
    `state_dict()`, which stays tensor-valued), a state load replaces it
    in place so identity and sharing survive, and the module owns **no**
    native storage. Training delegates to the differentiable operation,
    so a successful forward consumes exactly one call and a failed one
    none; evaluation returns the **input object itself**, consuming and
    allocating nothing, so an arbitrary number of eval forwards leaves no
    gap in the stream and the next training forward takes the next index;
    and `p == 0` is identity in both modes.
    **G5 is complete** — native checkpoint **format version 2** and exact
    generator restoration. The format *name* never moves; the version is
    now 2 and every new save writes 2, whether or not the model has
    generators. The manifest gained exactly one field, `"generators"`:
    `null` when the model registers none, so absence is stated rather
    than inferred, or `keys`/`entries`/`aliases` — the ordered canonical
    names, one `{algorithm, algorithm_version, seed, calls}` object each
    (seed and counter as **canonical decimal strings**, because a
    `uint64` above `2**53` is not representable in the IEEE double most
    JSON readers use), and the complete **registered path → canonical
    name** map. Generator state adds **no array** to the archive. A
    shared generator's state is written once while its *topology* is
    written in full, so two paths draw from one stream in the archive
    exactly when their aliases name the same canonical entry — sharing is
    **identity**, never state equality. Canonical names and both orders
    are functions of the model alone, so saving the same model twice is
    byte-identical. A load compares the archive against a real
    `named_generators()` traversal, strictly in both directions, and
    every mismatch — a missing or unexpected canonical key or registered
    path, an alias targeting an absent entry, a canonical name not
    self-mapped, a repeated JSON object key, saved-shared versus
    live-independent (or the reverse), a canonical name changed by a
    reordered registration, an algorithm or version mismatch, a malformed
    or out-of-range seed or counter — fails **in prevalidation, with
    nothing touched**. Generators are restored **in place**, so identity
    and every sharing relationship survive and the archive never
    constructs one. A save *or* a load is refused, leaving an existing
    destination byte-intact, while any target generator has a call
    reservation in flight. A **version-1** archive still loads into a
    model with no registered generators and is **rejected**, naming them,
    for one that has them — no seed and no counter is ever fabricated —
    while a v2 archive with generator state loaded into a generator-free
    model fails as an unexpected-generator error. And the load is **one
    transaction over the whole archive**: model, buffers, optimizer, and
    generators commit through their own loaders inside a single rollback
    guard, so any synchronous failure — a deliverable `KeyboardInterrupt`
    included — restores all four together, preserves every object
    identity, moves no parameter version, leaves graph-owned multiplier
    masks from earlier graphs untouched, and returns native live storage
    to baseline; only external process or interpreter death is outside
    that guarantee. It is **serializable** as well: every participating
    state replacement — the checkpoint load commit, `load_state_dict`,
    `load_generator_state_dict`, and both optimizers' state loads — plus
    the save snapshot runs under **one** private process-wide `RLock`, in
    the universal state-replacement lock order (that guard first, then
    every unique target generator lock in the global `id()` order, never
    the reverse), so two concurrent loads leave one archive's state
    followed by the other's rather than a mixture, and a save describes
    one coherent serial point. Generator reservations deliberately stay
    outside the guard, so a racing reservation precedes or follows a
    transaction and no state is replaced underneath a live token.
    Ordinary training mutation does not take the guard, so thread-safe
    concurrent training snapshots are not claimed. The whole registry
    footprint is one reporting-only
    name, `"checkpoint_generator_state"` in `STATE_SUPPORT`.
    **G6 is complete** — the hardening milestone. It executed §13 and §14
    of the design as adversarial tests in a new
    `tests/test_native_phase_g_hardening.py`: the reservation transition
    matrix, the exact `uint64` boundary, forced concurrent interleavings
    with bounded joins and no sleeps, the deterministic Core's structural
    key properties beside its committed vectors, every pre-commit and
    post-commit failure position of the call transaction over four
    exception classes, all four graph-owned saved-resource families in one
    graph, a 76-case checkpoint corruption matrix, whole-transaction
    rollback at every commit position, save-seam destination atomicity,
    and repeated success-and-failure lifecycle loops measured against a
    real native live-storage baseline. **It added no capability,
    operation, module, export, checkpoint field, or checkpoint version**
    and moved no registry value; it found and fixed exactly one runtime
    defect — a cleanup-failure `__context__` chain that could become
    cyclic — with a dedicated regression guard.
    Milestone **G7 is complete** — the end-to-end exact stochastic resume,
    and **no new capability**. `examples/native_dropout_training.py` trains
    `NativeLinear(4, 8)` -> `NativeBatchNorm1d(8)` -> `NativeReLU` ->
    `NativeDropout(p=0.5, seed=20240707)` -> `NativeLayerNorm(8)` ->
    `NativeLinear(8, 3)` over raw logits with `NativeCrossEntropyLoss` and
    `NativeAdam` on a fixed twelve-sample three-class task computed from an
    explicit formula, in three fixed batches on a schedule that is a **pure
    function of the training step**. It carries all four TensorForge-owned
    state families at once — parameters, the persistent native BatchNorm
    running buffers, a registered `NativeGenerator`, and NativeAdam moments with
    per-parameter step counters — so an incomplete restore diverges
    immediately. Two uninterrupted runs are bit-identical; an interrupted run
    checkpointed after 7 **completed** steps (deliberately mid-cycle in the
    batch schedule), whose model, optimizer, and generator are **released
    before the resume begins**, reloads into a completely fresh set built
    with a *different* native Dropout seed and reproduces the uninterrupted run by
    **exact equality**: the whole loss sequence, every parameter, both
    running statistics, every optimizer moment and step counter, the
    generator's algorithm/version/seed/calls, the final training logits, and
    the final evaluation output. Two negative controls make that meaningful —
    restoring all four families but restarting the batch schedule at 0
    **diverges**, and restoring everything but re-seeding the generator
    **diverges**. Evaluation is proved state-neutral (repeated eval passes
    leave `calls` bit-identical, produce identical outputs, restore the
    caller's mode, and leave a probed run's loss sequence equal to an
    unprobed one's), and a separate throwaway reload matches the restored
    module's next native Dropout output against `NativeTensorCore.dropout_forward`
    at the exact restored `(seed, call_index)`, advancing `calls` by exactly
    one. **External loop progress is carried explicitly**, as validated JSON
    metadata (`{"training_step": ..., "next_batch_index": ...}`), because
    checkpoint v2 captures TensorForge-owned state and **not** data-loader
    position, batch order, shuffle state, epoch counters, scheduler state,
    Python's `random`, or NumPy's global random state — a missing or inconsistent
    field raises rather than silently restarting from step 0.
    Reproducibility is exact **for the state actually captured**;
    full-program determinism is not claimed. The whole milestone is one
    example, one test module, and documentation: **no** C++, C ABI symbol,
    ctypes declaration, Core method, autograd operation, module, export,
    schema field, checkpoint version, benchmark, or registry value changed.
    Milestone **G8 is complete** — `benchmarks/benchmark_native_dropout.py`,
    the honest characterization, adding **no capability**. Thirty-five
    cases in eight families: the stateless Core against an **exact
    bit-for-bit** vectorized NumPy implementation of the same locked
    derivation, scalar-to-large size scaling, four physical layouts over
    one logical shape, a five-value probability sweep at three layers,
    the no-grad / differentiable / backward-only / forward-plus-backward
    operation layers, the module's training and identity paths, and one
    complete native Dropout training step — each gated for correctness **before**
    timing against the committed known-answer vectors, each recording its
    exact generator consumption, and all of them followed by an untimed
    lifecycle pass that returns native live storage to baseline. The
    operation and module cases are `native_only` and publish no ratio.
    **No speed assertion, no committed timing number, no CI timing
    threshold**, and no result file unless `--json-out` names one; the
    numbers are a machine-specific snapshot and nothing was optimized to
    improve one. Milestone **G9 is complete** — the cross-cutting Phase-G
    integration suite, `tests/test_native_phase_g.py`, adding **no
    capability** and changing no runtime file. One test-only model
    carries every registered state family at once (convolution, NCHW
    normalization buffers, pooling, two native Dropout layers over one
    shared generator, flatten, linear layers, 1-D normalization,
    native LayerNorm, and the fused loss over raw logits), and the suite
    proves
    the interactions: four saved-resource families in one graph released
    exactly once, exact version-2 resume into a fresh
    model/optimizer/generator set with a negative control that diverges,
    the generator-topology matrix with every mismatch rejected before any
    state changes, evaluation consuming no call anywhere, p == 0,
    non-contiguous NCHW and strided views, whole-state rollback at every
    commit position, four deterministic concurrency cases, a Phase A–F
    regression matrix, and live storage returning exactly to baseline
    across success and failure cycles. Milestone **G10 is complete** —
    the phase closure. The validation matrix ran with observed results:
    fresh Windows Release and Debug builds, each **11/11 CTests** with
    zero project warnings and the active runtime proved to stay the
    Release DLL; a fresh Clang 18.1.3 ASan+UBSan build in WSL2 with
    instrumentation proved rather than assumed, **11/11 sanitized
    CTests** with leak detection on, **3,166 sanitized Python tests**,
    the G7 example reproducing its exact resume, and the G8 benchmark
    smoke path passing every correctness gate — all with zero ASan and
    zero UBSan diagnostics; and a LeakSanitizer lifecycle returning
    native live storage exactly to baseline with no TensorForge frame in
    the remaining process-exit allocations and **no suppression file
    added**. Reproducibility stays exact only for
    the state actually captured (no Python `random`, no NumPy global
    random state, no data-loader position, and no scheduler state), and
    ordinary concurrent training is not claimed thread-safe.
    **`dropout` stayed listed unsupported for the whole of G0–G9** — G4
    implemented and exported `NativeDropout` and G5 persisted its stream,
    neither moving the boundary, because a capability
    whose value is exact reproducibility is not finished until
    reproducibility has been demonstrated under fresh Release and Debug
    builds and the sanitizers — and the name was removed at **G10**,
    after that matrix passed, leaving exactly `float32`, `cuda`, and
    `amp`. The claim it makes is narrow: native Dropout is supported in
    the **experimental native float64 CPU** backend, which says nothing
    about the stable framework, float32, CUDA, or AMP.
    Deliberately outside Phase G: a generic
    sampling or distribution API, global random state, NumPy
    global-random-state integration, parameter-initialization changes,
    data-loader shuffling,
    augmentation, 2-D and 3-D dropout variants, stochastic depth, attention
    dropout, integer tensors, embeddings, float32, CUDA, AMP,
    schedulers, new optimizers, CPU performance tuning, and any stable
    framework change.
  - **Phase H — Native CPU Performance and Runtime Efficiency — is the
    current phase, and it has begun at milestone H0 only.** Its
    architecture contract is
    [native_cpu_performance_design.md](native_cpu_performance_design.md).
    **H0 is architecture, profiling, and baseline work: nothing was made
    faster.** It shipped that contract, the unified measurement harness
    `benchmarks/benchmark_native_cpu_performance.py`, that harness's
    behavioral contract tests, and documentation reconciliation — and no
    C++, C ABI symbol, ctypes declaration, `NativeTensorCore` method,
    autograd operation, module, loss, metric, optimizer, export,
    capability registry value, dtype, device, or checkpoint change. The
    native checkpoint format stays version 2 with versions 1 and 2
    supported, `UNSUPPORTED` still reads `("float32", "cuda", "amp")`,
    and **Phase G remains the latest completed phase**. The harness
    measures 24 cases across twelve workload families — dispatch
    overhead, elementwise, reductions, matmul, materialization, linear,
    convolution, normalization, stochastic, optimizer, training step, and
    in-memory state operations — separating up to nine implementation
    layers (NumPy, the stable line, the raw-buffer kernels,
    `NativeTensorCore`, `NativeTensor` with and without a graph,
    backward, an optimizer step, and a whole training step), with a
    correctness gate that runs **before** timing everywhere, honest
    reference labelling that publishes **no ratio** where no equivalent
    exists, `--smoke` / `--json` / `--case` / `--workload` and a focused
    `--profile CASE` mode, and no result file of any kind. Checkpoint
    file I/O is deliberately excluded from every training-step total, and
    the in-memory state surface is its own category. The evidence it
    produced is deliberately ranked and honest — the largest measured
    factors are an allocator behavior and a memory access pattern rather
    than raw arithmetic, the Python-side per-call metadata path costs
    several times the ctypes boundary it wraps, and the `NativeTensor`
    wrapper and its autograd graph node are measurably **not** a
    bottleneck. **Milestone H1 — the output-allocation contract — has
    since shipped**, the first Phase-H change to production code: **Milestone H1 — the output-allocation contract — has now shipped.** It removed the redundant zero-fill from output storage that a kernel provably overwrites in full, behind one new C ABI symbol (`tf_storage_create_uninitialized`) that matches the zero-initializing default in size validation, allocation-failure handling, error state, ownership, destruction, and live-storage accounting, and differs only in the buffer's initial contents. The zero-initializing path remains the default; there is **no** global allocator policy, environment variable, heuristic, memory pool, scratch arena, or public empty-tensor API, and every enabled call site opts in explicitly against a per-kernel audit table. `sum`/`mean` and `narrow_backward` are explicitly **rejected** and keep a zeroed destination: the first accumulates into its output, the second writes only the narrowed region and the untouched zeros *are* the gradient. Completeness is proved by deterministic **poison** tests that are injected **exclusively by test infrastructure, around the allocator**: the suite wraps the private uninitialized allocation helper, lets the real constructor allocate, fills the returned storage with a quiet NaN or a large finite pattern through the ordinary fill primitive, and hands that same storage to the real operation — so the pattern is in place after the real allocation and before the real kernel runs. **No poison-control mechanism exists in the production runtime**: no exported hook, no thread-local flag, no environment variable, no global mode. ASan and UBSan stay separate from the initialization proof — they do not detect uninitialized-value reads — and MemorySanitizer is not available here, so neither is claimed; negative controls prove the detector can actually fail. H1 is bit-identical: every enabled operation and a full training run are compared element-wise against the zero-initializing allocator. No capability, dtype, device, registry value, checkpoint field, or checkpoint version changed, and `tf_storage_create_uninitialized` is the **only** export it added, taking the library from the pre-H1 baseline of 51 exported `tf_*` symbols to **52**.
    The measured result is reported honestly rather than as a headline: isolated, the zero-fill is enormous and scales with the buffer (about 52x at 2 MB, 119x at 8 MB, 552x at 32 MB, and *negative* below roughly 16,000 elements, where it sits inside the noise). End to end it is much smaller and often inconclusive — clearly real for large memory-bound elementwise work (about 1.5-1.8x on an 8 MB output), small and variable for the `normalized_training_step` and `adam_step` cases, and with no measurable effect on `conv2d_forward`, `mlp_training_step`, or `matmul_square_contiguous`, whose arithmetic dwarfs its allocation. Those inconclusive and negative rows are published as such.
    **Milestone H2 — native matmul memory access — has since shipped**,
    and is the first Phase-H milestone to change how a numerical kernel
    executes. It swapped the production matmul's loop order from
    `i`-`j`-`k` to `i`-`k`-`j` over four destination rows at a time, so
    the innermost loop walks a *row* of the right operand and a row of
    the output sequentially instead of walking a column. **Cache blocking,
    which the milestone title anticipated, was measured against 22 blocked
    variants and rejected** — an unblocked full-width row sweep was faster
    at every non-trivial size — so H2 shipped the simpler superior design
    and recorded the negative blocking result. The pre-H2 triple loop is
    **retained verbatim as the shipped generic reference path** and still
    runs, chosen from stride metadata inside the kernel: a right operand
    whose column stride is 1, with a non-empty inner dimension and at
    least 8 result columns, takes the row sweep; a transposed right
    operand, a narrow result, or an empty inner dimension takes the
    generic path — which is the loop order that case already suits, so
    the fallback is a design choice rather than a gap. Dispatch is
    metadata-driven, deterministic, total, side-effect free, and
    independent of pointer values, alignment, timing, environment
    variables, and CPU-feature probes; a failed precondition is never an
    error. **H2 added no exported C ABI symbol** — the count is still 52 —
    and no kernel selector, block-size setter, dispatch tracer, or public
    dispatch control of any kind exists. The numerical agreement between
    the two paths is a **four-part contract**, not a blanket bit-identity
    claim: (1) accumulation order is preserved exactly; (2) **every
    non-NaN result is bit-identical**, asserted as raw IEEE-754 bit
    patterns rather than tolerances across shapes, layouts, signed zeros,
    infinities, denormals, largest finite magnitudes, gradients, Linear,
    both optimizers, deterministic training, and exact checkpoint resume;
    (3) NaNs occur in exactly the same positions on both paths and are
    always quiet; and (4) **NaN payload bits are deliberately outside
    TensorForge's numerical contract** and may differ. Ten source-level
    formulations were measured while trying to close (4), and the only
    structure that reproduces the reference's payloads is the `i`-`j`-`k`
    order H2 replaces, so parity is unavailable short of abandoning the
    optimization; MSVC Release differs on 162 of 208 results in a
    NaN-saturated matrix, MSVC Debug and Clang on none.
    H1's uninitialized-output contract still holds on both paths, for a
    different reason on each: the generic path never reads the
    destination, and the row sweep's `k == 0` pass assigns every element
    before anything accumulates into it — proved by poison tests over both
    paths with both patterns, plus a negative control. The measured result
    is reported honestly: roughly 4.1-4.7x at 384 cubed, 4.2-4.5x at 128
    cubed, about 4-6.8x on `NativeLinear` forward, 1.7-2.5x on its
    backward (only one of its two matmuls qualifies, by design), 2.0-2.4x
    on a 128x256 MLP training step, and **no measurable effect below
    roughly 32 cubed or on a small MLP step**, where a fixed ~10 microsecond
    per-call Python cost dominates and control cases whose code did not
    change at all vary by 0.50-1.44x. No capability, dtype, device,
    registry value, checkpoint field, or checkpoint version moved.
    The proposed **H3–H8 ladder is explicitly conditional**:
    a milestone whose premise the measurement does not confirm is
    narrowed, reordered, or dropped, and a memory pool, scratch
    allocation, SIMD, threading, and BLAS are all currently **rejected on
    evidence**, with the criteria that would reopen each recorded rather
    than an answer invented. H9 (re-measurement, hardening, and the full
    sanitizer matrix) and H10 (phase closure) are **not started**.
    Deliberately outside Phase H: CUDA, float32/float16/bfloat16,
    casting, dtype promotion, AMP, Tensor Cores, pybind11, C++ autograd,
    implicit dispatch, Transformers, attention, embeddings, integer
    tensors, data loaders, distributed training, a memory pool, scratch
    allocation, SIMD, threading, OpenMP, BLAS, any required dependency,
    checkpoint format version 3, and **any CI timing threshold or
    committed performance number**.
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
