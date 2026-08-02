# TensorForge — project instructions

## 1. Project identity and architecture

TensorForge is a from-scratch deep learning framework: a serious ML
systems project covering PyTorch-style framework internals. Position it
as serious and systems-focused — never "educational", "toy", or "mini" —
while staying honest: **not production-ready, not a PyTorch
replacement.**

Two lines live in one repository:

- **The stable Python framework line** (`tensorforge`, `tensorforge.nn`,
  `tensorforge.optim`, `tensorforge.data`): Tensor + reverse-mode
  autograd on NumPy. Complete as of **v3.0**; feature-frozen unless a
  milestone says otherwise.
- **The experimental native line** (`tensorforge.backends`,
  `tensorforge.experimental`, `cpp/`): a C++17 CPU runtime behind a
  plain C ABI loaded with ctypes, with its own tensor, autograd,
  modules, optimizers, RNG, and checkpoints. It lives on `main` in those
  explicit namespaces, not on a separate branch.

Development is milestone by milestone: each one small, tested, readable,
and documented. Every milestone's full record — design, evidence,
measurements, rejected alternatives — lives in `docs/`, not here.

### Tech stack

- Python ≥ 3.13, NumPy, pytest — nothing else.
- Managed with `uv` (`uv run …` for everything).
- **Never introduce** PyTorch, TensorFlow, JAX, sklearn, pandas, or
  matplotlib. NumPy is the only numeric dependency. The C++ backend
  needs nothing but a C++17 compiler — no BLAS, no oneDNN, no Eigen, no
  pybind11, no GoogleTest.

### Layout

- `src/tensorforge/tensor.py` — Tensor + reverse-mode autograd. Ops are
  either primitives (eager NumPy forward + a `_backward` closure holding
  the local derivative) or derived (compositions that get gradients for
  free). Gradients accumulate via `_accumulate_grad`, which also
  un-broadcasts.
- `src/tensorforge/nn/` — Parameter, Module, Linear, activations,
  Dropout, BatchNorm1d, LayerNorm, Conv2d, MaxPool2d, Flatten,
  Sequential, losses (`mse_loss`, `cross_entropy`,
  `binary_cross_entropy`), metrics (`accuracy`, `binary_accuracy`,
  `evaluate_classifier`, `evaluate_binary_classifier` — the evaluators
  measure with the model temporarily in eval mode and restore it).
  `model.train()` / `model.eval()` recurse through children; Dropout and
  BatchNorm1d change behavior. Modules declare non-trainable buffers via
  `self._buffers = ("attr", ...)`; `state_dict()` / `load_state_dict()`
  cover parameters *and* buffers.
- `src/tensorforge/optim/` — SGD, Adam. Plain classes: `step()` skips
  `None` grads and frozen params, `zero_grad()` sets grads to `None`.
  Also `StepLR` and `clip_grad_norm` / `clip_grad_value` (clip in place
  before `optimizer.step()`).
- `src/tensorforge/data.py` — `batches` mini-batch iterator.
- `src/tensorforge/backends/cpp.py` — the **only** module in the
  repository that imports `ctypes`: library loading, the C ABI argument
  bindings, `NativeStorage` / `NativeTensorView` / `NativeTensorCore`,
  and the capability registries.
- `src/tensorforge/experimental/` — the native tensor, autograd,
  modules, losses, metric, optimizers, generator, state transactions,
  and checkpoints. One concept per file.
- `cpp/src/` + `cpp/include/` — the C++ kernels, organized by concern
  (`elementwise`, `matmul`, `reduction`, `conv2d`, `pooling`,
  `classification`, `random`, `storage`, `error`). `tf_*_internal.h`
  headers hold hidden-visibility helpers; nothing there is exported.
- `cpp/tests/` — dependency-free C++ CTests that compile the kernel
  source directly. Built only with `-DTF_BUILD_TESTS=ON`.
- `examples/` — runnable scripts, each with `train(...)` returning stats
  and a `main()` that prints, guarded by `__main__`.
- `tests/` — the pytest suite; every feature has tests.
- `benchmarks/` — characterization harnesses (§9).
- `scripts/smoke_cpp_backend.py` — the hard-failing smoke check CI runs
  after building.
- `docs/` — the source of truth for everything historical (§11).
- `.github/workflows/tests.yml` — CI: install uv, build the C++ backend,
  hard-failing smoke check, quick benchmark smoke run, then pytest.

---

## 2. Stable / native separation

**The two lines are strictly separated and must stay that way.**

- The stable framework **never** imports the native backend. Importing
  `tensorforge` must not load the C++ library, and a test proves it.
- Importing the wrapper is always safe — the library loads lazily. Check
  `cpp.is_available()` / `cpp.backend_info()`; kernels raise
  `ImportError` at call time when unbuilt, and the backend tests skip.
- `stable_framework_integration` is `False` in `backend_info()` and
  stays false. There is no automatic backend selection, no implicit
  dispatch, and no environment variable that changes which line runs.
- Native modules mirror stable semantics but are separate classes
  (`NativeLinear`, not a `Linear` backend flag).

---

## 3. Current support boundary

These are the canonical registry values in `backends/cpp.py`. Changing
any of them is a capability decision, never a side effect:

| Registry | Value |
|---|---|
| `SUPPORTED_DTYPES` | `("float64",)` |
| `SUPPORTED_DEVICES` | `("cpu",)` |
| `UNSUPPORTED` | `("float32", "cuda", "amp")` |
| Native checkpoint format | `tensorforge.native_checkpoint`, version **2** |
| Accepted checkpoint versions | `(1, 2)` |
| Exported production `tf_*` symbols | **54** (Phase H closed at 52; Phase I milestone I1 added the two typed storage creators, which are the only two the phase adds) |

Since Phase I milestone I1, float32 storage is **allocatable through the C
ABI** (`tf_storage_create_typed`); since I2 it is also **movable** — host
ingress and egress, strided materialization, and the storage-to-storage
identity copy (`tf_core_contiguous_copy`) are dtype-general and
bit-preserving; since I3 it is **computed on** by the elementwise and unary
Core family (`add`, `subtract`, `multiply`, `relu`, `relu_backward`,
`sqrt`, `reciprocal`, `exp`, `log`, with broadcasting); and since I4 it
also **accumulates** — `sum`, `mean`, `matmul`, and `narrow_backward` are
dtype-general, `tf_storage_scale` and `tf_storage_fill` narrow their
`double` argument once before the loop, and **private/internal** float32
`NativeTensor` graphs run forward and backward over that set.

That is not a support claim and does not change a single row above: every
operation that has not been dtype-generalized — conv2d in all three
directions, both MaxPool2d directions, softmax, log-softmax,
cross-entropy, and Dropout — rejects a float32 handle with
`TF_ERROR_INVALID` before touching memory; `normalize_dtype("float32")`
still raises; and no public constructor produces a float32 tensor, so
float32 parameters, modules, optimizers, checkpoints, and training do not
exist. The private float32 graphs I4 proves are built through the private
typed constructors (`_typed`, `_typed_from_array`, `_typed_full`,
`zeros(..., _trusted_dtype=True)`, `NativeTensor._from_core`), which exist
so an intermediate milestone can test through them while the public
boundary stays exactly where it is. The public registry moves at **I9**,
not before.

One further registry exists and is a **different** statement from
`SUPPORTED_DTYPES`: `RAW_KERNEL_DTYPES == ("float64",)` (added at I2,
reported by `backend_info()` as `raw_kernel_dtypes`). The seven handle-free
raw utility kernels take only `double*` and an element count, so they have
no dtype to dispatch on and stay float64 permanently. Never report it as
overall native dtype support, and never read the public promise off it.

**Performance work never broadens support.** A milestone that makes
something faster must leave every row above untouched. The canonical
capability status lives in `docs/native_support_matrix.md`.

Not supported, and not a bug: float32/float16/bfloat16, casting, dtype
promotion, AMP, CUDA or any GPU backend, integer tensors, data loaders,
distributed training, C++-side autograd, attention/Transformers.

---

## 4. Core invariants

Everything in this section holds across every phase and may not be
weakened by a milestone.

### 4.1 Public API and C ABI discipline

- The public API is locked by tests (`tests/test_public_api.py` for the
  stable root package; the registries and `experimental.__all__` for the
  native line). Adding to it is a milestone decision.
- **Adding a C ABI export is a milestone decision**, not an
  implementation detail. Optimizations ship *inside* existing exports.
- Hidden default visibility; `TF_EXPORT` only on functions Python
  actually declares. The source export inventory and the built library's
  export table must agree.
- **No public performance control of any kind exists or may be added**:
  no kernel/path selector, block-size or threshold setter, traversal or
  dispatch tracer, benchmark hook, profiling counter, "which path ran"
  query, or environment-variable dispatch.
- **No production poison, profiling, or allocation-content control.** A
  symbol compiled into and exported from the normal runtime is part of
  the runtime however carefully it is disarmed. (The one pre-existing
  exception is documented: `tf_test_arm_alloc_failure` /
  `tf_fault_injection_available`, the deterministic thread-local
  allocation-failure hook from the Phase-C era. It is inert until armed,
  changes no buffer *contents*, and is part of the 52. Do not add a
  second such hook, and do not remove this one without a milestone.)

### 4.2 Optimized-path dispatch

Every optimized kernel path in the native line follows one shape, and a
new one must too:

1. **One unchanged export.** Both paths live behind the symbol Python
   already declares.
2. **The pre-milestone traversal is retained verbatim** as the shipped
   generic reference path, still reachable through ordinary production
   dispatch, and is the oracle the optimized result is compared against.
3. **A hidden-visibility predicate chooses**, and it is total, pure,
   allocation-free, and a function of **layout or geometry metadata
   alone** — never of a pointer value, an alignment, a clock, an
   environment variable, or a CPU-feature probe.
4. **A false answer is a fallback, never an error.**

Currently shipped predicates: `tf::matmul_prefers_row_sweep`,
`tf::copy_prefers_contiguous`, `tf::reduce_prefers_contiguous_blocks`,
`tf::build_unary_plan` / `tf::build_binary_plan`,
`tf::conv2d_forward_prefers_row_sweep`,
`tf::conv2d_input_backward_prefers_gather`,
`tf::conv2d_weight_backward_prefers_gather`.

### 4.3 Deliberately absent

None of these exists anywhere in the repository, and none may be added
without meeting its own recorded criteria in
`docs/native_cpu_performance_design.md` §10–§13:

memory pool · scratch workspace or arena · persistent cache of native
storage · SIMD intrinsics · threading · OpenMP · BLAS · oneDNN · Eigen ·
im2col · general operator fusion · fast-math · cache blocking.

### 4.4 C ABI error containment

`docs/native_abi_error_contract.md` is the contract. **No exported
native function may let a C++ exception escape.** Fallible functions
wrap their body in `TF_GUARD_BEGIN` / `TF_GUARD_END(...)`, which clears
the calling thread's error slot on entry and, on failure, records a
`TfStatus` code plus message in thread-local storage and returns a
benign value instead of unwinding. Functions that cannot fail are
deliberately unguarded and never touch the slot. Python maps
`TF_ERROR_ALLOC` → `MemoryError`, `TF_ERROR_INVALID` → `ValueError`,
`TF_ERROR_RUNTIME` → `RuntimeError`.

Self-validating exports reject null handles, negative sizes, spans
exceeding their storage, and aliasing between a source and a
destination — and when they reject, they **write nothing**.

### 4.5 Determinism

- No kernel consults a clock, a process id, an address, allocation
  history, or static/thread-local state to produce a value.
- Random values come only from the explicit `NativeGenerator` key
  (`tensorforge.splitmix64`; seed + call index). No `<random>`, no
  `std::random_device`, no implicit global stream.
- Examples use fixed seeds so output is reproducible.
- **Deterministic training and exact checkpoint resume are proved by
  test in every phase from C onward, and every one of those proofs must
  keep passing.** An interrupted run reloaded into a *fresh*
  model/optimizer/generator set reproduces the loss suffix, every
  parameter, every buffer, every optimizer moment and step counter, the
  generator state, and the final training and evaluation outputs by
  **exact equality**.
- Reproducibility is exact **for the state TensorForge captures**.
  Python's `random`, NumPy's global RNG, data-loader position, batch
  order, and scheduler state are not captured; full-program determinism
  is not claimed.

---

## 5. Ownership and state

### 5.1 Native storage ownership

- A `NativeTensorCore` owns its `NativeStorage`; a `NativeTensorView`
  borrows. Views never close their parent's storage; a chained view
  keeps the whole chain reachable.
- Every operation allocates a **fresh owning contiguous** output that
  aliases neither operand.
- **Cleanup is explicit and never relies on garbage collection.**
  `close()` is the contract; `__del__` is only a fallback. Any failure
  — allocation, native call, Python wrapper construction, graph-node
  construction, resource attachment — closes everything it allocated, so
  live storage returns exactly to baseline and no caller can observe one
  lone result.

### 5.2 Graph-owned saved resources

Four families exist: Dropout masks, MaxPool2d winners, BatchNorm eval
snapshots, and cross-entropy saved probabilities. Each rides the
`graph_resources` contract: released **exactly once** with the graph
history, retained under `retain_graph=True`, kept alive across a failed
retryable backward, freed by an abandoned graph's `close()`, and closed
immediately by a no-grad forward. A registered buffer is **never** a
rereadable graph operand — BatchNorm eval reads independent owning
snapshots instead.

### 5.3 Identity and versioning

- `load_state_dict()`, `load_native_checkpoint()`, and the optimizer
  loaders **preserve every parameter, buffer, and generator identity**
  and every sharing relationship. They restore in place.
- A parameter's version counter moves **once** per committed mutation.
  Shared parameters deduplicate to one slot, one update, one increment.
- Loading **buffer** or **generator** state moves no parameter version
  and stales no graph. A **full** checkpoint load replaces parameters and
  therefore correctly stales an earlier graph through the parameter rule
  — a parameter contract, never a buffer or RNG effect.
- Frozen parameters stay registered and persisted but are skipped by
  optimizers.

### 5.4 Transactional boundaries

Each of these is atomic under failure, validated before anything is
published, and leaves identities, versions, and live storage exactly as
it found them when it fails:

output allocation + wrapper publication · `NativeParameter.copy_value_`
· optimizer stage/commit (validation is four complete passes before any
mutation; commit is one `copy_value_` and one version increment per
updated parameter) · the BatchNorm running-statistics two-buffer
transaction · `NativeModule.load_state_dict` · optimizer
`load_state_dict` · whole-checkpoint load · generator-state replacement
· graph-resource adoption.

Honest scoping, recorded rather than glossed: transactions are **per
module**; one whole training step is *not* globally transactional.
Ordinary training mutation does not take the process-wide
state-replacement lock, so thread-safe concurrent training snapshots are
not offered — the claim is that *participating* state-replacement
operations serialize with respect to each other, in the universal lock
order (the private process-wide guard first, then every unique generator
lock in global `id()` order, never the reverse).

External process or interpreter death is the only documented exception
to whole-checkpoint atomicity.

---

## 6. Numerical contracts

**Never publish one universal "bit-identical" claim.** Each operation
family has its own rule, measured rather than inherited. The full
statements live in `docs/native_cpu_performance_design.md` §7 and §16;
the durable summary:

| Family | Contract |
|---|---|
| **Value transfer** (`contiguous_copy`, state/checkpoint transfer) | Reproduces its source's bits **exactly** — including `-0.0` and both signs of signaling NaN, and every NaN payload. A transfer performs no arithmetic, so it has no operand roles to choose between. An *operation* that happens to copy (`zeros + x`) follows IEEE arithmetic instead, and therefore does normalize `-0.0` and quiet a signaling NaN. |
| **Elementwise** (`add`, `subtract`, `multiply`, `relu`, `relu_backward`, `sqrt`, `reciprocal`) | Bit-identical whenever **at most one operand is NaN**. `subtract` is bit-identical everywhere. For `add`/`multiply` with **two** NaN operands the surviving payload is **outside the contract**, asserted in neither direction. |
| **`exp` / `log`** | Library functions with no correctly-rounded IEEE guarantee. Deliberately **excluded** from the templated traversal, and the cross-platform test contract is a **one-ULP** finite bound, not bit equality. |
| **matmul** | Accumulation order preserved exactly. Every non-NaN result bit-identical. NaN positions identical and always quiet. NaN **payload** bits deliberately outside the contract. |
| **Reduction** | Per-output accumulation order preserved exactly, source traversal order not even reordered. Signed zeros proved as raw bit patterns. Bit-identical whenever **at most one NaN** enters an accumulation; payloads outside the contract when two or more meet in one cell. |
| **Conv2d** (all three directions) | Per-destination accumulation order preserved exactly. Every non-NaN result bit-identical; NaN positions identical; **at most one NaN per destination agrees including payload**; signed zeros bit-identical; signalling NaNs quieted identically. Two-or-more-NaN payloads not contractual. |
| **Optimizers, normalization, softmax, log-softmax, cross-entropy** | Bit-identical to the composition they replaced. No reassociation, no accumulator-width change, no operand-position change. |

Nothing anywhere reassociates arithmetic, uses FMA, fast-math, an
intrinsic, `restrict`, a tree/pairwise/parallel reduction, or a
horizontal vector reduction.

### Output initialization (H1)

Output storage is **zero-initialized by default**. A call site may opt
in to `tf_storage_create_uninitialized` **only** when the kernel
provably overwrites every destination element before reading it, and
only against a per-kernel audit table. `sum`/`mean` and
`narrow_backward` are explicitly rejected and keep a zeroed destination
— the first accumulates into its output, the second writes only the
narrowed region and the untouched zeros *are* the gradient.

Completeness is proved by deterministic **poison** tests injected
**exclusively by test infrastructure, around the allocator**, always
with a negative control showing the detector can fail. ASan and UBSan do
**not** detect uninitialized-*value* reads and MemorySanitizer is not
available here, so neither is claimed as that proof.

---

## 7. Build and test commands

```bash
uv run pytest                       # the whole suite; expect zero skips
uv run python cpp/build.py          # build the native backend (Release)
uv run python cpp/build.py --debug  # unoptimized, assertions on
uv sync --group cpp                 # only if you have no C++ compiler
uv run python scripts/smoke_cpp_backend.py
```

Examples:

```bash
uv run python examples/train_linear_regression.py
uv run python examples/train_xor.py
uv run python examples/train_multiclass.py
uv run python examples/train_binary_classification.py
uv run python examples/train_mlp_with_dropout.py
uv run python examples/train_tiny_cnn.py
uv run python examples/native_dropout_training.py
```

`cpp/build.py` is a thin wrapper around the canonical CMake build
(`cpp/CMakeLists.txt`), which owns the real compilation architecture.
When CMake is absent it falls back to one direct compiler invocation
over the same source list (this is what CI uses). `TF_SANITIZE` and
`TF_BUILD_TESTS` are the **only** build options; adding a third is a
milestone decision.

### Windows validation (the primary development platform)

Build **Release and Debug out-of-source, outside the repository**, and
write the Debug library elsewhere so the active runtime stays the
Release DLL. Require **zero project compiler, linker, and CMake
warnings** and the full CTest suite green in each configuration.

```bash
cmake -S cpp -B <outside-repo>/release -DTF_BUILD_TESTS=ON
cmake --build <outside-repo>/release --config Release
ctest --test-dir <outside-repo>/release -C Release
```

### WSL / Linux validation

Match GitHub Actions: `uv sync --group cpp`, `uv run python
cpp/build.py`, the smoke check, the quick benchmark, then `uv run
pytest`. The transcendental (`exp`/`log`) test contract is a one-ULP
bound precisely because libm differs between MSVC and glibc; do not
tighten it back to bit equality.

---

## 8. Sanitizer procedure

Every milestone that touches C++ or changes allocation behavior must
pass this, on Clang under Linux/WSL (MSVC does not support it):

```bash
cmake -S cpp -B <outside-repo>/asan -DTF_BUILD_TESTS=ON \
      -DCMAKE_CXX_COMPILER=clang++ -DTF_SANITIZE=address,undefined
cmake --build <outside-repo>/asan
nm -D <library> | grep -c __asan     # instrumentation proved present
nm -D <library> | grep -c __ubsan
```

Required:

- instrumentation **proved present** (`__asan*` / `__ubsan*` dynamic
  symbols beside the exported `tf_*` symbols, and the library refusing
  to load without the sanitizer runtime);
- the full native CTest suite under that build;
- the native Python suites under it, with **zero** ASan and **zero**
  UBSan diagnostics;
- a **negative control** proving the instrumentation can actually fail —
  test-only code that hands a kernel malformed metadata and produces a
  real `heap-buffer-overflow`. Zero diagnostics only means something
  when the detector is known to work;
- a LeakSanitizer lifecycle in which native live storage returns
  **exactly** to baseline, with the remaining process-exit allocations
  containing **no TensorForge frame** and **no suppression file added**.

---

## 9. Benchmark rules

`benchmarks/cpp_backend.py` compares raw kernels against NumPy;
`benchmark_native_cnn.py`, `benchmark_native_classification.py`,
`benchmark_native_normalization.py`, `benchmark_native_dropout.py`, and
`benchmark_native_cpu_performance.py` characterize their stacks.

Non-negotiable, in every harness:

- **Correctness is gated before timing**, always. A failed gate
  publishes no timing and the CLI exits nonzero with clean stdout.
- **No speed is asserted anywhere.** There is no timing threshold, no
  performance budget, no committed duration, and **no CI job that fails
  on a number**. Phase H did not add the first one and neither may
  anything else.
- **No result file of any kind is written**, in any phase. A committed
  number becomes a promise the project cannot keep across machines.
- A case with no honest equivalent is labelled `native_only` and
  publishes **no ratio at all**. Never fabricate a comparison layer.
- Setup, cleanup, and any advanced state stay outside the timer;
  temporaries are closed explicitly rather than left to GC; a case whose
  call advances persistent state rebuilds or resets it per repetition.
- Report medians with spread after warm-up; publish regressions,
  neutral results, and noise as prominently as wins.

**Measurement methodology lessons, learned the hard way and repeatedly:**

- Use **alternating pre/post rounds in separate subprocesses**, and
  prove every case **bit-identical before either side is timed**.
- **Low round counts lie.** H3, H5, H6, and H9 each recorded a case that
  read as a regression at 7–9 rounds and as neutral-or-faster at 21–25.
  Never quote a low-round figure as evidence.
- State the machine's **control band** (identical-code cases) and treat
  any reading inside it as neutral.
- Whole-translation-unit **code-layout effects are real**: adding code to
  one `.cpp` can move an unrelated function's timing by several percent
  on byte-identical source. Publish it; do not chase it.
- Below roughly 1,000 elements a fixed ~7–12 µs Python-plus-ctypes cost
  dominates and kernel work is invisible. That is an architectural
  floor, not a defect.

---

## 10. Public / ABI restrictions when writing code

Do not, without an explicit milestone that says so:

- add or remove an exported `tf_*` symbol, or change the C ABI;
- add a public API, capability-registry value, dtype, device, checkpoint
  field, or checkpoint version;
- add a build option, a required dependency, or a mandatory
  `-march`/`/arch` flag;
- introduce anything from §4.3;
- add a timing assertion, a committed benchmark number, or a result
  file;
- weaken a validation, an error type, or an error message in the name of
  speed;
- couple the stable line to the native one.

---

## 11. Documentation map (source-of-truth hierarchy)

`CLAUDE.md` holds **current operating rules and durable invariants
only**. Everything historical — milestone reports, measurements,
rejected alternatives, evidence — lives in `docs/` and must not be
duplicated here.

| Question | Authoritative document |
|---|---|
| What is supported, right now | `docs/native_support_matrix.md` |
| Overall architecture | `docs/architecture.md` |
| Project overview / status | `docs/project_summary.md` |
| Per-release history | `docs/release_history.md` |
| What is planned next | `docs/roadmap.md` |
| The C++ backend, builds, sanitizers | `docs/backend_experiments.md` |
| C ABI error handling | `docs/native_abi_error_contract.md` |
| Autograd (stable line) | `docs/autograd.md` |
| Training / examples | `docs/training.md`, `docs/examples.md` |
| Optimized/generic dispatch pattern | `docs/dispatch_design.md`, `docs/native_contiguous_fast_path_design.md` |
| Native tensor wrapper & ownership | `docs/native_tensor_wrapper_design.md` |
| Broadcasting / reductions | `docs/native_broadcasting_design.md`, `docs/native_reductions_design.md` |
| dtype/device metadata | `docs/native_dtype_device_metadata_design.md` |
| Native autograd | `docs/native_autograd_design.md` |
| **Phase D** — native CNN | `docs/native_cnn_design.md` |
| **Phase E** — classification & stable math | `docs/native_classification_design.md` |
| **Phase F** — normalization & stateful buffers | `docs/native_normalization_design.md` |
| **Phase G** — RNG & Dropout | `docs/native_rng_dropout_design.md` |
| **Phase H** — CPU performance | `docs/native_cpu_performance_design.md` |
| **Phase I** — dtype generalization & float32 | `docs/native_dtype_float32_design.md` |

When a milestone changes the public API or the examples, update the
matching docs file (and README links) **in the same milestone**.

---

## 12. Current project status

- **Stable Python line: complete at v3.0.**
- **Native line: Phases A–H complete.**
  - A — CPU runtime; B — native autograd; C — native training stack;
    D — native CNN; E — classification and stable math; F —
    normalization and stateful buffers; G — RNG and Dropout.
  - **H — Native CPU Performance and Runtime Efficiency: complete
    (H0–H10).** H1 output-allocation contract; H2 matmul memory access;
    H3 metadata and dispatch; H4 optimizer step; H5 copy and
    mutation transfer; H6 reduction execution; H7 Python/C ABI boundary;
    H8 elementwise traversal and normalization allocation; H9 Conv2d
    execution; H10 integration, remeasurement, the acceleration
    decision, and closure.
  - The ladder was **revised on evidence** three times — a reorder (H5),
    a drop (the original composed-module H7), and a reassignment (the
    original SIMD/threading/BLAS H9). All three are recorded in the
    design document rather than rewritten away.
  - **SIMD, threading/OpenMP, and BLAS were each finally decided at H10
    and rejected, with measurements.** Their reopening criteria are
    `docs/native_cpu_performance_design.md` §11–§13.
  - Phase H changed **no** capability, dtype, device, registry value,
    public API, checkpoint field, or checkpoint version, and added
    exactly **one** C ABI symbol across the whole phase
    (`tf_storage_create_uninitialized`, at H1): 51 → **52**.

- **Native line: Phase I at I4** — Native Dtype Generalization and
  Float32 CPU Support. Contract:
  `docs/native_dtype_float32_design.md`. **I0 (design, contract tests,
  documentation), I1 (the dtype model and dtype-tagged storage), I2
  (typed transfer, views, and materialization), I3 (elementwise,
  broadcast, and unary dtype execution), and I4 (reductions, matmul,
  views, and core autograd) are complete; I5–I11 are not started.**
  - I1 delivered: the C++ `TfDtype`/`tf::Dtype` model with frozen codes
    `0 = float64` and `1 = float32`, one item-size authority
    (`tf::dtype_item_size` — nothing else may spell a storage width), one
    canonical-name authority, and a total validated conversion; storage
    owning a **genuine runtime-selected `float[]` or `double[]` array**
    behind a type-erased `void*` plus a dtype tag, created with checked
    `numel × itemsize` and released by one central dtype-matched
    `delete[]`. The array form is load-bearing, not incidental: the
    project is C++17, where pointer arithmetic is defined only within one
    array object, so neither a byte array plus a reinterpret-cast nor
    separately placement-constructed scalars would legalize the `data[i]`
    the kernels perform. The two typed creators;
    `tf::storage_f64` as the one typed-access pattern and
    `tf::require_float64` as the one float32 rejection; the untyped
    creators as thin float64 wrappers. CTests moved 17 → 18.
  - I2 delivered: the three exports that carry a storage handle **and** a
    raw host buffer (`tf_storage_copy_from`, `tf_storage_copy_to`,
    `tf_storage_materialize`) generalized by a **source-level retype** of
    their host positions from `double*` to `void*` — a declaration change,
    not an ABI change: same symbols, same argument counts and order, same
    calling convention, still **54** exports, and a previously compiled
    caller links and runs identically; the host pointer carries no dtype
    and the storage tag is authoritative, so C++ dispatches from the tag
    and Python validates the NumPy dtype before each call through
    `_host_pointer`, which runs the per-dtype `ndpointer` check the
    argtypes slot can no longer hold (one slot cannot describe two
    dtypes). `tf_core_contiguous_copy` — the value-transfer primitive, and
    the only compute-shaped export I2 touched — became dtype-preserving
    and dtype-strict, with its three H5/H8 tiers instantiated for both
    element types from one source. `tf::unary_row`, `tf::unary_plan_walk`,
    the retained odometer, and `tf::IdentityOp::apply` gained a **deduced**
    scalar type, so every pre-existing call site compiles unchanged and
    `T = double` is the pre-I2 code statement for statement. Transfer is
    bit-preserving at both widths — proved, not asserted, over seventeen
    IEEE-754 classes per dtype as raw `uint32`/`uint64` patterns; `memcpy`
    was **not** introduced (§4.3 forbids it) and the transfers stay
    same-type element assignments. `RAW_KERNEL_DTYPES` added. Internal
    float32 construction is three private constructors
    (`NativeStorage._typed`, `NativeStorage._typed_from_array`,
    `NativeTensorCore._typed_from_array`) plus a keyword-only
    `_trusted_dtype` on `NativeStorage.__init__`; the private H1
    allocators inherit that trust because their dtype always comes from a
    live storage, and `NativeTensorCore.full` calls `normalize_dtype`
    explicitly so no public constructor inherits it. CTests moved 18 → 19.
  - I3 delivered: the elementwise and unary Core family generalized to both
    dtypes — `add`, `subtract`, `multiply`, `relu`, `relu_backward`,
    `sqrt`, `reciprocal`, `exp`, `log`, across their strided and contiguous
    forms (17 exports, **none new**). `tf::require_float64` became
    `tf::require_matching_dtype` at each of them, and a new
    `tf::dispatch_dtype` supplies the **one** `switch` per exported call,
    held by four hidden helpers (`unary_by_dtype`,
    `unary_contiguous_by_dtype`, `binary_by_dtype`,
    `binary_contiguous_by_dtype`), none with a `default:` label.
    `tf::binary_row` and `tf::binary_plan_walk` gained the deduced scalar
    type their unary twins got at I2, and `core_binary_typed` joined
    `core_unary_typed` as the retained generic reference path at both
    widths. **The operation functors became the single source of every
    per-element expression**: their `apply` is templated, their constants
    are `T(...)`, and the retained odometers now take `&Op::apply<T>`
    instead of a hand-matched duplicate — so the optimized and reference
    paths cannot drift. `exp`/`log` keep H8's exclusion **structurally**:
    they have no functor in the shared header, only file-local function
    templates, so nothing can plan-walk them. Outputs preserve the operand
    dtype through the private I2 typed path; broadcasting works at float32
    for every layout it already worked at for float64; mixed dtype is
    rejected in all three operand positions before any allocation, with the
    dtype guard ordered **before** the span validation. CTests moved
    19 → 20 (`test_dtype_elementwise`).
  - I4 delivered: `tf_core_sum`, `tf_core_matmul`, and
    `tf_core_narrow_backward` generalized to both dtypes (3 exports,
    **none new**), plus `tf_storage_scale` and `tf_storage_fill`, which
    left the rejecting set because `scale` *is* the mean reduction's
    scaling step and `fill` is how a backward materializes its constants.
    All four compute paths — H6's `sum_contiguous_blocks` and the retained
    `sum_generic_strided`, H2's `matmul_row_sweep` and the retained
    `matmul_generic_strided` — became templates over the element type and
    moved into `tf_reduction_internal.h` / `tf_matmul_internal.h`, which is
    where a template must live for both instantiations to reach the export
    *and* the CTests that compile those files directly; the narrow-backward
    scatter became `tf::narrow_backward_scatter` on the same terms. Loop
    nests, carries, `k` orders, and row grouping are unchanged;
    `double sum = 0.0` became `T sum = T(0)` and `0.0 + a_ik * b_row[j]`
    became `T(0) + a_ik * b_row[j]`. **Both metadata predicates are
    untouched**, so both widths take the same path for the same layout.
    The two scalar primitives keep their `(handle, double)` ABI and narrow
    **once, before the loop** (§7.4), and neither writes to the error slot
    any more — the right end state for an unhooked export that can no
    longer fail. Private float32 `NativeTensor` graphs run forward and
    backward over the whole set, with gradients, temporaries, and every
    materialized constant at the graph's dtype through
    `NativeTensorCore._typed_full` and a keyword-only `_trusted_dtype` on
    `NativeTensorCore.zeros`. CTests moved 20 → 21
    (`test_dtype_reduction_matmul`).
  - **Public capability did not move at I1, I2, I3, or I4**: float64 CPU
    only, `float32` still in `UNSUPPORTED`, `RAW_KERNEL_DTYPES` still
    `("float64",)`, checkpoint version 2 with (1, 2) accepted. Only the
    export count changed, 52 → **54**, at I1; I2, I3, and I4 added none.
  - **A dtype-general Core kernel is not a public capability, and neither
    is a private graph.** I3 generalized `tf_core_relu_backward` because it
    is a forward-shaped numerical primitive, not graph machinery; I4's
    float32 `NativeTensor` graphs are reached only through the private
    typed constructors. float32 parameters, modules, optimizers,
    checkpoints, and training remain absent, and no public constructor
    produces a float32 tensor at all.
  - **Recorded so no later milestone relitigates it:** for a *single*
    correctly-rounded IEEE operation — which is every I3 operation, one per
    destination element — computing in binary64 and rounding once to
    binary32 is *provably* indistinguishable from computing in binary32
    (binary64 carries more than the 2p+2 = 50 bits a double rounding would
    need). So "float32 is not secretly float64" could not rest on a runtime
    test *there*, and none was invented: it was carried by the result being
    bit-identical to the binary32 oracle plus a **semantic structural
    check** over the source. **I4 supplied the behavioural half**, because
    accumulation finally makes the two policies distinguishable: on `1.0`
    followed by eight copies of `2**-24`, sequential binary32 stays at
    exactly `1.0` while binary64-then-narrow lands four ULPs higher, and
    TensorForge is asserted equal to the first and **unequal** to the
    second on both reduction traversals and both matmul paths. Keep both
    halves — the witness proves the result, the structural check proves no
    width in the source could make one path right and another wrong.
  When implementing a Phase-I milestone, the durable rules are:
  - **exactly two** new C ABI exports across the whole phase
    (`tf_storage_create_typed`, `tf_storage_create_uninitialized_typed`,
    52 → 54 — **already spent at I1**); per-operation float32 exports are
    rejected;
  - storage carries the dtype and is its **single** authority; shapes,
    strides, and offsets stay in logical elements; bytes only at the
    allocation boundary, with checked `numel × itemsize`;
  - **one narrow dispatch per exported call** into templated
    `float`/`double` kernels; no dtype branching below it, no string
    dispatch, no per-element indirection;
  - **no casting, no promotion, no mixed-dtype arithmetic**; a mismatch
    raises before any allocation or mutation;
  - **float32 accumulates in float32** — no hidden float64 accumulator;
  - float64 results stay **bit-identical** and Phase-H performance is
    preserved;
  - checkpoint **version 3** at I8 (accepted `(1, 2, 3)`; versions 1 and
    2 are float64-only and never guessed to be float32);
  - the public registry moves at **I9**, not earlier.

Beyond Phase I (future work, not started): data loaders, native integer
tensors, further dtypes/devices beyond float32/float64, CUDA experiments.
See `docs/roadmap.md`; do not invent a phase that document does not
define.

---

## 13. Agent operating rules

### Style

- Keep code simple and readable — clarity beats cleverness.
- Match the existing style: NumPy-only internals, small modules, one
  concept per file.
- Comments explain math/autograd/ownership reasoning, not obvious
  Python.
- Losses and metrics stay simple: losses are Tensor expressions or fused
  ops with custom backward; metrics are plain NumPy returning Python
  floats, outside autograd.
- Examples use fixed seeds and follow the `train()` + `main()` pattern
  so tests can import `train`.
- Tests use `np.allclose` with sensible tolerances (e.g. `atol=1e-6`);
  training tests assert learning without fragile exact-loss values.
  Bit-level claims use raw IEEE-754 bit patterns, not tolerances.

### Workflow

- **Inspect existing code before editing**; find where a concept lives
  and follow its pattern.
- **Keep changes scoped to the requested milestone.** No unrelated
  features, no drive-by refactors, no framework rewrites. Do not hide
  extra optimization under "cleanup".
- If a requested feature already exists, verify it against the spec and
  add tests/documentation instead of reimplementing it.
- **Preserve all previous tests. Never loosen a test just to pass.**
- Run `uv run pytest` (and any requested manual checks) before reporting
  success, and **report the actual observed output**.
- A documented rejection backed by measurements is better than an unsafe
  or weak implementation.
- **Do not use git for anything that writes**: no commits, pushes,
  pulls, merges, rebases, resets, branch or checkout operations,
  stashes, or history edits. Read-only inspection (`git status`,
  `diff`, `log`, `show`, `rev-parse`, `ls-files`, `branch
  --show-current`) is fine. The user controls version control.
- Final responses report: files changed, what was implemented, tests
  added, the exact pytest result, manual check outputs, and any notes or
  limitations.

### Machine notes

- **Permissions quirk:** directories created by one process often cannot
  be deleted by a later one. Already handled — pytest's cache is
  redirected to `.cache/pytest` (pyproject) and `conftest.py` gives each
  session a fresh unique basetemp. **Do not try to delete**
  `.pytest_cache/`, `.cache/pytest-tmp/`, or `%TEMP%/pytest-of-*`.
- Two example-test import styles coexist: `tests/test_examples.py`
  inserts `examples/` into `sys.path`; newer tests import
  `examples.<name>` as a namespace package from the repo root.
- Stable root package exports (locked by `tests/test_public_api.py`):
  `Tensor`, `Parameter`, `Dropout`, `BatchNorm1d`, `LayerNorm`, `Conv2d`,
  `MaxPool2d`, `Flatten`, `cross_entropy`, `binary_cross_entropy`,
  `accuracy`, `binary_accuracy`, `evaluate_classifier`,
  `evaluate_binary_classifier`, `SGD`, `Adam`, `StepLR`,
  `clip_grad_norm`, `clip_grad_value`, `batches`, `train_test_split`,
  `save_parameters`, `load_parameters`, `save_checkpoint`,
  `load_checkpoint`, `count_parameters`, `model_summary`. Stable
  checkpoints = weights + optimizer state + optional scheduler state +
  optional RNG state (`rng_state=True` / `restore_rng_state=True`,
  covers unseeded Dropout) + JSON metadata; parameters = weights only.
