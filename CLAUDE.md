# TensorForge — project instructions

**This file holds current operating rules and durable invariants only.**
Everything historical — milestone reports, measurements, evidence, rejected
alternatives — lives in `docs/` (§11) and must never be copied back here.

---

## 1. Project identity and architecture

TensorForge is a from-scratch deep learning framework: a serious ML systems
project covering PyTorch-style framework internals. Position it as serious
and systems-focused — never "educational", "toy", or "mini" — while staying
honest: **experimental, not production-ready, and not a PyTorch
replacement.**

Two lines live in one repository, and they stay strictly separate (§2):

- **The stable Python line** (`tensorforge`, `tensorforge.nn`,
  `tensorforge.optim`, `tensorforge.data`) — Tensor + reverse-mode autograd
  on NumPy. Complete at **v3.0**; feature-frozen unless a milestone says
  otherwise.
- **The experimental native line** (`tensorforge.backends`,
  `tensorforge.experimental`, `cpp/`) — a C++17 CPU runtime behind a plain C
  ABI loaded with `ctypes`, with its own tensor, autograd, modules,
  optimizers, RNG, and checkpoints. It lives on `main` inside those explicit
  namespaces.

Development is milestone by milestone: small, tested, readable, documented.

### Tech stack

- Python ≥ 3.13, NumPy, pytest — nothing else. Managed with `uv`; run
  everything through `uv run`.
- **Never introduce** PyTorch, TensorFlow, JAX, sklearn, pandas, or
  matplotlib. NumPy is the only numeric dependency. The C++ backend needs
  nothing but a C++17 compiler — no BLAS, no oneDNN, no Eigen, no pybind11,
  no GoogleTest.

### Layout

- `src/tensorforge/tensor.py` — Tensor + reverse-mode autograd. Ops are
  primitives (eager NumPy forward plus a `_backward` closure holding the
  local derivative) or derived compositions that get gradients for free.
  Gradients accumulate through `_accumulate_grad`, which also un-broadcasts.
- `src/tensorforge/nn/` — Parameter, Module, Linear, activations, Dropout,
  BatchNorm1d, LayerNorm, Conv2d, MaxPool2d, Flatten, Sequential, losses, and
  metrics. `train()` / `eval()` recurse through children; modules declare
  non-trainable buffers via `self._buffers = ("attr", ...)`, and
  `state_dict()` / `load_state_dict()` cover parameters *and* buffers.
- `src/tensorforge/optim/` — SGD, Adam, `StepLR`, `clip_grad_norm` /
  `clip_grad_value`. `step()` skips `None` grads and frozen parameters;
  `zero_grad()` sets grads to `None`; clipping is in place, before `step()`.
- `src/tensorforge/data.py` — the stable `batches` mini-batch iterator.
- `src/tensorforge/backends/cpp.py` — the **only** module in the repository
  that imports `ctypes`: library loading, the C ABI argument bindings,
  `NativeStorage` / `NativeTensorView` / `NativeTensorCore`, and the
  capability registries.
- `src/tensorforge/experimental/` — native tensor, autograd, modules, losses,
  metric, optimizers, generator, state transactions, checkpoints; one concept
  per file.
- `cpp/src/` + `cpp/include/` — kernels organized by concern (`elementwise`,
  `matmul`, `reduction`, `conv2d`, `pooling`, `classification`, `random`,
  `storage`, `error`). `tf_*_internal.h` headers hold hidden-visibility
  helpers and kernel templates; nothing there is exported.
- `cpp/tests/` — dependency-free C++ CTests that compile the kernel source
  directly, built only with `-DTF_BUILD_TESTS=ON`.
- `examples/` — runnable scripts, each with `train(...)` returning stats and
  a `main()` that prints, guarded by `__main__`. `tests/` — the pytest suite;
  every feature has tests. `benchmarks/` — characterization harnesses (§9).
- `scripts/smoke_cpp_backend.py` — the hard-failing smoke check CI runs after
  building. `docs/` — the source of truth for everything historical (§11).
- `.github/workflows/tests.yml` — CI: install uv, build the C++ backend,
  hard-failing smoke check, quick benchmark smoke run, then pytest.

---

## 2. Stable / native separation

**The two lines are strictly separated and must stay that way.**

- The stable framework **never** imports the native backend. Importing
  `tensorforge` must not load the C++ library, and a test proves it.
- Importing the wrapper is always safe — the library loads **lazily**. Check
  `cpp.is_available()` / `cpp.backend_info()`; kernels raise `ImportError` at
  call time when unbuilt, and the backend tests skip.
- `stable_framework_integration` is `False` in `backend_info()` and stays
  false. There is **no** automatic backend selection or routing, no implicit
  dispatch, no implicit stable↔native conversion, and no environment variable
  that changes which line runs.
- Native modules mirror stable semantics as **separate classes**
  (`NativeLinear`, never a `Linear` backend flag).

---

## 3. Support boundary

The canonical values live in `backends/cpp.py`; the authoritative capability
statement is `docs/native_support_matrix.md`. Changing any row is a
capability decision, never a side effect.

| Row | Value |
|---|---|
| `SUPPORTED_DTYPES` | `("float64", "float32")` — order is contractual, float64 first |
| `SUPPORTED_DEVICES` | `("cpu",)` |
| `UNSUPPORTED` | `("cuda", "amp")` |
| `RAW_KERNEL_DTYPES` | `("float64",)` — permanent, and a different statement |
| `normalize_dtype(None)` | `"float64"` |
| `backend_info()["dtype"]` | `"float64"` — the **default**, not the capability |
| Native checkpoint format | `tensorforge.native_checkpoint`, version **3** |
| Accepted checkpoint versions | `(1, 2, 3)`; versions 1 and 2 stay float64-only |
| In-memory optimizer state version | **1** |
| Exported production `tf_*` symbols | **54** (Phase H closed at 52) |
| Native CTests | **24** |
| Examples | **15** |

**Three dtype rows, three different questions**, and none may be reported as
another: `SUPPORTED_DTYPES` is the **capability**; `backend_info()["dtype"]`
is the **default** an omitted `dtype` selects; `RAW_KERNEL_DTYPES` is a
**permanent limitation** of the handle-free raw utility kernels, which take
only `double*` and an element count and so have no dtype to dispatch on —
never read the public promise off that last row.

### Dtype rules that hold everywhere

- **float64 is the default** at every constructor, factory, module, and
  parameter, and is what `None` means.
- The dtype is **never inferred** from a host array: a float32 NumPy array
  passed with no `dtype` still gives float64.
- **No casting, no promotion, no mixed-dtype arithmetic.** A mismatch raises
  before any allocation or mutation. There is no `astype` / `to` /
  `.float()` / `.double()` / `map_location`, and no global default.
- **float32 accumulates in float32** — no hidden float64 accumulator.
- Storage carries the dtype and is its **single** authority. Shapes, strides,
  and offsets stay in logical elements; bytes appear only at the allocation
  boundary, with a checked `numel × itemsize`.
- One narrow dispatch per exported call into templated `float`/`double`
  kernels: no dtype branching below it, no string dispatch, no per-element
  indirection.
- **No `device` argument exists anywhere and none may be added.** There is no
  device movement.
- Classification targets stay **host `int64` metadata** at every width. No
  integer tensor dtype exists.
- Constructors that own numeric state take a **keyword-only** `dtype`
  accepting exactly `"float64"` / `"float32"`, through the one shared private
  validator. A class owning no dtype-bearing state — the losses, the metric,
  `NativeSequential`, `NativeReLU`, `NativeFlatten`, `NativeMaxPool2d`,
  `NativeDropout`, `NativeGenerator`, both optimizers — must **not** gain
  one: a second authority could disagree with the data.
- The private typed constructors (`_typed*`, `_trusted_dtype=True`,
  `NativeTensor._from_core`) **stay, and stay private**. They grant no width
  the public constructors do not; "this dtype came from live storage or a
  validated archive" is a different trust statement from "a caller said so".
- The MaxPool2d winner buffer is **private float64 at every value dtype** —
  a permanent pin, not an oversight.
- The NumPy reference backend keeps its own float64-only `supported_dtypes`.

**Performance work never broadens support.** A milestone that makes something
faster must leave every row above untouched.

Not supported, and not a bug: float16/bfloat16, mixed precision, AMP, CUDA or
any GPU backend, integer tensors, float32 raw kernels, distributed training,
C++-side autograd, attention/Transformers. Native mini-batching does not
exist yet either — see §12.

Full dtype contract, evidence, and rejected alternatives:
`docs/native_dtype_float32_design.md`.

---

## 4. Core invariants

These hold across every phase and may not be weakened by a milestone.

### 4.1 Public API and C ABI discipline

- The public API is locked by tests (`tests/test_public_api.py` for the
  stable root package; the registries and `experimental.__all__` for the
  native line). Adding to it is a milestone decision.
- **Adding or removing a C ABI export is a milestone decision**, not an
  implementation detail. Optimizations ship *inside* existing exports.
- Hidden default visibility; `TF_EXPORT` only on functions Python actually
  declares. The source export inventory and the built library's export table
  must agree.
- Private helpers stay private. A name that is private because of what its
  caller may be trusted to know does not become public for convenience.
- **No public performance control of any kind exists or may be added**: no
  kernel or path selector, block-size or threshold setter, traversal or
  dispatch tracer, benchmark hook, profiling counter, "which path ran" query,
  environment-variable dispatch.
- **No production poison, profiling, or allocation-content control.** A
  symbol compiled into and exported from the normal runtime is part of the
  runtime however carefully it is disarmed. The one documented exception is
  the deterministic thread-local allocation-failure hook
  (`tf_test_arm_alloc_failure` / `tf_fault_injection_available`): inert until
  armed, changing no buffer *contents*, and part of the export count. Do not
  add a second; do not remove this one without a milestone.

### 4.2 Optimized-path dispatch

Every optimized kernel path follows one shape, and a new one must too:

1. **One unchanged export.** Both paths live behind the symbol Python already
   declares.
2. **The pre-milestone traversal is retained verbatim** as the shipped
   generic reference path, still reachable through ordinary production
   dispatch, and is the oracle the optimized result is compared against.
3. **A hidden-visibility predicate chooses**, and it is total, pure,
   allocation-free, and a function of **layout or geometry metadata alone** —
   never of a pointer value, an alignment, a clock, an environment variable,
   or a CPU-feature probe.
4. **A false answer is a fallback, never an error.**

Shipped predicates, whose naming a new one should follow:
`tf::matmul_prefers_row_sweep`, `tf::copy_prefers_contiguous`,
`tf::reduce_prefers_contiguous_blocks`, `tf::build_unary_plan` /
`tf::build_binary_plan`, and the three `tf::conv2d_*` geometry predicates.

The pattern was established across Phase H — H4's optimizer step, H5's copy
transfer, H6's reduction execution — and inherited unchanged by the dtype
templates, so both widths take the same traversal for the same layout. See
`docs/native_cpu_performance_design.md` and `docs/dispatch_design.md`.

### 4.3 Deliberately absent

None of these exists anywhere in the repository, and none may be added
without meeting its own recorded criteria in
`docs/native_cpu_performance_design.md` §10–§13:

memory pool · scratch workspace or arena · persistent cache of native
storage · SIMD intrinsics · threading · OpenMP · BLAS · oneDNN · Eigen ·
im2col · general operator fusion · fast-math · cache blocking.

SIMD, threading/OpenMP, and BLAS were each rejected on measurement;
reopening criteria are in that document.

### 4.4 C ABI error containment

`docs/native_abi_error_contract.md` is the contract. **No exported native
function may let a C++ exception escape.** Fallible functions wrap their body
in `TF_GUARD_BEGIN` / `TF_GUARD_END(...)`, which clears the calling thread's
error slot on entry and, on failure, records a `TfStatus` code plus message
in thread-local storage and returns a benign value instead of unwinding.
Functions that cannot fail are deliberately unguarded and never touch the
slot. Python maps `TF_ERROR_ALLOC` → `MemoryError`, `TF_ERROR_INVALID` →
`ValueError`, and `TF_ERROR_RUNTIME` → `RuntimeError`.

Self-validating exports reject null handles, negative sizes, spans exceeding
their storage, and aliasing between a source and a destination — and when
they reject, they **write nothing**. The C ABI is a **second** authority, not
a restatement of Python's: never remove a C-side check because Python already
performs it.

### 4.5 Determinism

- No kernel consults a clock, a process id, an address, allocation history,
  or static/thread-local state to produce a value.
- Random values come only from the explicit `NativeGenerator` key
  (`tensorforge.splitmix64`; seed plus call index). No `<random>`, no
  `std::random_device`, no implicit global stream. Contract:
  `docs/native_rng_dropout_design.md`.
- Examples use fixed seeds so output is reproducible.
- **Deterministic training and exact checkpoint resume are proved by test in
  every phase from C onward, and every one of those proofs must keep
  passing.** An interrupted run reloaded into a *fresh*
  model/optimizer/generator set reproduces the loss suffix, every parameter,
  every buffer, every optimizer moment and step counter, the generator state,
  and the final training and evaluation outputs by **exact equality**.
  Bit-level claims are made in raw IEEE-754 bit patterns, never a tolerance.
- Reproducibility is exact **for the state TensorForge captures**. Python's
  `random`, NumPy's global RNG, data-loader position, batch order, and
  scheduler state are not captured; full-program determinism is not claimed.

---

## 5. Ownership, lifecycle, and transactions

### 5.1 Native storage ownership

- A `NativeTensorCore` owns its `NativeStorage`; a `NativeTensorView`
  borrows. Views never close their parent's storage; a chained view keeps the
  whole chain reachable.
- Every operation allocates a **fresh owning contiguous** output that aliases
  neither operand.
- **Cleanup is explicit and never relies on garbage collection.** `close()`
  is the contract and is idempotent; `__del__` is only a fallback. Any
  failure — allocation, native call, Python wrapper construction, graph-node
  construction, resource attachment — closes everything it allocated, so
  **live storage returns exactly to baseline** and no caller can observe one
  lone result. Details: `docs/native_tensor_wrapper_design.md`.

### 5.2 Graph-owned saved resources

Four families exist: Dropout masks, MaxPool2d winners, BatchNorm eval
snapshots, cross-entropy saved probabilities. Each rides the
`graph_resources` contract: released **exactly once** with the graph history,
retained under `retain_graph=True`, kept alive across a failed retryable
backward, freed by an abandoned graph's `close()`, closed immediately by a
no-grad forward. A registered buffer is **never** a rereadable graph
operand — BatchNorm eval reads independent owning snapshots instead.

### 5.3 Identity and versioning

- `load_state_dict()`, `load_native_checkpoint()`, and the optimizer loaders
  **preserve every parameter, buffer, and generator identity** and every
  sharing relationship. They restore in place.
- A parameter's version counter moves **once** per committed mutation. Shared
  parameters deduplicate to one slot, one update, one increment.
- Loading **buffer** or **generator** state moves no parameter version and
  stales no graph. A **full** checkpoint load replaces parameters and
  therefore correctly stales an earlier graph through the parameter rule — a
  parameter contract, never a buffer or RNG effect.
- Frozen parameters stay registered and persisted but are skipped by
  optimizers.

### 5.4 Transactional boundaries

Each of these is atomic under failure, **validated completely before anything
is published**, and leaves identities, versions, and live storage exactly as
it found them when it fails — no partial mutation, ever:

output allocation plus wrapper publication · `NativeParameter.copy_value_` ·
optimizer stage/commit · the BatchNorm running-statistics two-buffer
transaction · `NativeModule.load_state_dict` · optimizer `load_state_dict` ·
whole-checkpoint load · generator-state replacement · graph-resource
adoption.

Honest scoping, recorded rather than glossed: transactions are **per
module**, and one whole training step is *not* globally transactional.
Ordinary training mutation does not take the process-wide state-replacement
lock, so thread-safe concurrent training snapshots are not offered — the
claim is only that *participating* state-replacement operations serialize
with respect to each other, in the universal lock order (the private
process-wide guard first, then every unique generator lock in global `id()`
order, never the reverse). External process or interpreter death is the only
documented exception to whole-checkpoint atomicity.

### 5.5 Checkpoint and state rules

- The native checkpoint is `tensorforge.native_checkpoint` **version 3**;
  `(1, 2, 3)` are accepted and every new save writes 3. **There is no version
  4** without an explicit milestone.
- Versions 1 and 2 are float64-only formats permanently, and a payload is
  never *guessed* to be float32. Version 3 declares every numeric entry's
  dtype explicitly.
- Schema validation is strict, runs on **both** the save and the load side
  through the same authority, and rejects before anything is staged or
  mutated. **No silent casting or coercion anywhere** — a declared dtype that
  disagrees with the array fails in either direction, as does a foreign byte
  order.
- The in-memory optimizer state format is version **1**. Neither optimizer
  has a `dtype` or `device` argument: they own no dtype to choose, only state
  that must match a parameter.
- Generator state is persisted with its **complete alias topology**, and a
  load restores each generator in place.
- The native checkpoint captures **no** data-loader position, shuffle order,
  or epoch counter, and no milestone changes that implicitly. Phase-J loader
  state is **caller-managed**: the caller serializes it and passes it through
  the existing validated version-3 `metadata` channel. There is **no
  automatic loader discovery and no checkpoint/runtime coupling.**

---

## 6. Numerical contracts

**Never publish one universal "bit-identical" claim.** Each operation family
has its own rule, measured rather than inherited. Full statements:
`docs/native_cpu_performance_design.md`. The durable summary:

| Family | Contract |
|---|---|
| **Value transfer** (`contiguous_copy`, state/checkpoint transfer) | Reproduces its source's bits **exactly** — `-0.0`, both signs of signaling NaN, every NaN payload — at both dtypes; a transfer performs no arithmetic, so it has no operand roles to choose between. An *operation* that happens to copy (`zeros + x`) follows IEEE arithmetic instead, so it *does* normalize `-0.0` and quiet a signaling NaN. |
| **Elementwise** (`add`, `subtract`, `multiply`, `relu`, `relu_backward`, `sqrt`, `reciprocal`) | Bit-identical whenever **at most one operand is NaN**; `subtract` everywhere. With **two** NaN operands the surviving payload is **outside the contract**, asserted in neither direction. |
| **`exp` / `log`** | Library functions with no correctly-rounded IEEE guarantee. Deliberately **excluded** from the templated traversal; the cross-platform contract is a **one-ULP** finite bound, not bit equality. |
| **matmul** | Accumulation order preserved exactly. Every non-NaN result bit-identical; NaN positions identical and always quiet; NaN **payload** bits outside the contract. |
| **Reduction** | Per-output accumulation order preserved exactly, source traversal order not even reordered. Signed zeros proved as raw bit patterns. Bit-identical whenever **at most one NaN** enters an accumulation. |
| **Conv2d** (all three directions) | Per-destination accumulation order preserved exactly. Non-NaN results and signed zeros bit-identical; NaN positions identical; **at most one NaN per destination agrees including payload**. |
| **Optimizers, normalization, softmax, log-softmax, cross-entropy** | Bit-identical to the composition they replaced: no reassociation, no accumulator-width change, no operand-position change. |

Nothing anywhere reassociates arithmetic, uses FMA, fast-math, an intrinsic,
`restrict`, a tree/pairwise/parallel reduction, or a horizontal vector
reduction.

**Do not "fix" a correctly rounded IEEE result.** Where a float32 shift
overflows because the finite spread exceeds the type's range, that *is* the
correct answer for values the type cannot represent; a widened intermediate,
a clamp, or a special case would be mixed precision by the back door.

### Output initialization

Output storage is **zero-initialized by default**. A call site may opt in to
`tf_storage_create_uninitialized` **only** when the kernel provably
overwrites every destination element before reading it, and only against the
per-kernel audit table. `sum`/`mean` and `narrow_backward` are explicitly
rejected and keep a zeroed destination — the first accumulates into its
output, the second writes only the narrowed region, and the untouched zeros
*are* the gradient there.

Completeness is proved by deterministic **poison** tests injected
**exclusively by test infrastructure, around the allocator**, always with a
negative control showing the detector can fail. ASan and UBSan do **not**
detect uninitialized-*value* reads, so neither is claimed as that proof.

---

## 7. Build, test, and validation workflow

```bash
uv run pytest                       # the whole suite; expect zero skips
uv run python cpp/build.py          # build the native backend (Release)
uv run python cpp/build.py --debug  # unoptimized, assertions on
uv sync --group cpp                 # only if you have no C++ compiler
uv run python scripts/smoke_cpp_backend.py
uv run python examples/<name>.py    # all 15 run; see docs/examples.md
```

`cpp/build.py` wraps the canonical CMake build (`cpp/CMakeLists.txt`), which
owns the real compilation architecture; when CMake is absent it falls back to
one direct compiler invocation over the same source list (this is what CI
uses). `TF_SANITIZE` and `TF_BUILD_TESTS` are the **only** build options;
adding a third is a milestone decision.

**How to validate a change:**

- Run the focused tests for what you touched first, then **the full Python
  suite before reporting success**, and report the actual observed output.
- **Only claim what you ran.** Never report a Linux run, a sanitizer run, a
  CTest run, or a rebuild that did not happen.
- Documentation-only or test-only work needs no native rebuild. Anything that
  touches `cpp/` or changes allocation behavior needs the native rebuild, the
  CTest suite, the Linux/CI-equivalent run, and §8.
- Inspect the **architecture**, not just the test count: a green suite with a
  weakened contract is a regression.

### Windows validation (the primary development platform)

Build **Release and Debug out-of-source, outside the repository**, and write
the Debug library elsewhere so the active runtime stays the Release DLL.
Require **zero project compiler, linker, and CMake warnings** and the full
CTest suite green in each configuration.

```bash
cmake -S cpp -B <outside-repo>/release -DTF_BUILD_TESTS=ON
cmake --build <outside-repo>/release --config Release
ctest --test-dir <outside-repo>/release -C Release
```

### WSL / Linux validation

Match GitHub Actions: `uv sync --group cpp`, `uv run python cpp/build.py`,
the smoke check, the quick benchmark, then `uv run pytest`. The
transcendental (`exp`/`log`) test contract is a one-ULP bound precisely
because libm differs between MSVC and glibc; do not tighten it back to bit
equality.

---

## 8. Sanitizer procedure

Every milestone that touches C++ or changes allocation behavior must pass
this, on Clang under Linux/WSL (MSVC does not support it):

```bash
cmake -S cpp -B <outside-repo>/asan -DTF_BUILD_TESTS=ON \
      -DCMAKE_CXX_COMPILER=clang++ -DTF_SANITIZE=address,undefined
cmake --build <outside-repo>/asan
nm -D <library> | grep -c __asan     # instrumentation proved present
nm -D <library> | grep -c __ubsan
```

Required:

- instrumentation **proved present** (`__asan*` / `__ubsan*` dynamic symbols
  beside the exported `tf_*` symbols, and the library refusing to load
  without the sanitizer runtime);
- the full native CTest suite under that build, and the native Python suites
  under it with **zero** ASan and **zero** UBSan diagnostics;
- a **negative control** proving the instrumentation can actually fail —
  test-only code that hands a kernel malformed metadata and produces a real
  `heap-buffer-overflow`. Zero diagnostics only means something when the
  detector is known to work;
- a LeakSanitizer lifecycle in which native live storage returns **exactly**
  to baseline, with the remaining process-exit allocations containing **no
  TensorForge frame** and **no suppression file added**.

Notes and results: `docs/backend_experiments.md`.

---

## 9. Benchmark rules

Benchmarks are **characterization, never a test gate.**
`benchmarks/cpp_backend.py` compares raw kernels against NumPy; the
per-stack harnesses beside it (`benchmark_native_autograd`, `_cnn`,
`_classification`, `benchmark_native_normalization`, `_dropout`,
`benchmark_native_cpu_performance`, `_dtype`) characterize their own stacks.
The dtype harness is a separate file on purpose and measures each dtype
**separately**, never as a ratio of one to the other.

Non-negotiable, in every harness:

- **Correctness is gated before timing**, always. A failed gate publishes no
  timing and the CLI exits nonzero with clean stdout.
- **No speed is asserted anywhere.** There is no timing threshold, no
  performance budget, no committed duration, and **no CI job that fails on a
  number**.
- **No result file of any kind is written**, in any phase. A committed number
  becomes a promise the project cannot keep across machines.
- A case with no honest equivalent is labelled `native_only` and publishes
  **no ratio at all**. Never fabricate a comparison layer.
- Setup, cleanup, and any advanced state stay outside the timer; temporaries
  are closed explicitly rather than left to GC; a case whose call advances
  persistent state rebuilds or resets it per repetition.
- Report medians with spread after warm-up; publish regressions, neutral
  results, and noise as prominently as wins.

**Measurement methodology, learned the hard way:**

- Use **alternating pre/post rounds in separate subprocesses**, and prove
  every case **bit-identical before either side is timed**.
- **Low round counts lie**, repeatedly: never quote one as evidence. State
  the machine's **control band** (identical-code cases) and treat any reading
  inside it as neutral.
- Whole-translation-unit **code-layout effects are real** — adding code to
  one `.cpp` can move an unrelated function's timing on byte-identical
  source. Publish it; do not chase it.
- On small inputs a fixed per-call Python-plus-ctypes cost dominates and
  kernel work is invisible — an architectural floor, not a defect.

---

## 10. Restrictions when writing code

Do not, without an explicit milestone that says so:

- add or remove an exported `tf_*` symbol, or change the C ABI;
- add a public API, capability-registry value, dtype, device, checkpoint
  field, or checkpoint version;
- add a build option, a required dependency, or a mandatory `-march`/`/arch`
  flag;
- introduce anything from §4.3;
- add a timing assertion, a committed benchmark number, or a result file;
- weaken a validation, an error type, or an error message in the name of
  speed;
- couple the stable line to the native one.

---

## 11. Documentation map

Detailed phase records belong in these documents, **not** here. When a
milestone changes the public API or the examples, update the matching
document (and README links) **in the same milestone**.

| Question | Authoritative document |
|---|---|
| What is supported, right now | `docs/native_support_matrix.md` |
| Overall architecture | `docs/architecture.md` |
| Project overview / status | `docs/project_summary.md` |
| Per-release history | `docs/release_history.md` |
| What is planned next | `docs/roadmap.md` |
| The C++ backend, builds, sanitizers | `docs/backend_experiments.md` |
| C ABI error handling | `docs/native_abi_error_contract.md` |
| Stable autograd, training, examples | `docs/autograd.md`, `docs/training.md`, `docs/examples.md` |
| Native wrapper, ownership, autograd | `docs/native_tensor_wrapper_design.md`, `docs/native_autograd_design.md` |
| Dispatch, broadcasting, reductions, dtype metadata | `docs/dispatch_design.md`, `docs/native_contiguous_fast_path_design.md`, `docs/native_broadcasting_design.md`, `docs/native_reductions_design.md`, `docs/native_dtype_device_metadata_design.md` |
| **Phase D** — native CNN | `docs/native_cnn_design.md` |
| **Phase E** — classification & stable math | `docs/native_classification_design.md` |
| **Phase F** — normalization & buffers | `docs/native_normalization_design.md` |
| **Phase G** — RNG & Dropout | `docs/native_rng_dropout_design.md` |
| **Phase H** — CPU performance | `docs/native_cpu_performance_design.md` |
| **Phase I** — dtype generalization & float32 | `docs/native_dtype_float32_design.md` |
| **Phase J** — deterministic data pipeline & mini-batching | `docs/native_data_pipeline_design.md` |

---

## 12. Current status

- **Stable Python line: complete at v3.0**, feature-frozen.
- **Native line: Phases A–I are complete** — CPU runtime (A) through dtype
  generalization and float32 CPU support (I); the per-phase subjects are in
  the §11 map and `docs/release_history.md`.
- **Phase I** (I0–I11) is the latest completed phase. It made exactly one
  public capability change — float32 joined `SUPPORTED_DTYPES` — and only
  after the integrated exact-resume proof passed. That ordering is the rule,
  not an accident: **prove first, then promise.**
- Phase J is the latest phase — **Deterministic Native Data Pipeline and
  Mini-Batching**, authority `docs/native_data_pipeline_design.md`. It was
  approved **after** Phase I closed and was not on the earlier roadmap;
  never describe it as pre-existing plan work.
- Milestones **J0** (contract) and **J1** (dataset) are done; **J2 through
  J9 have not started**, and **J2 is next.** J1 added exactly one public
  name, `NativeTensorDataset` — a finite dataset over two owned copied host
  snapshots at an explicitly chosen native dtype, with caller-owned batches.
  **`NativeBatchSampler` (J2) and `NativeDataLoader` (J3) do not exist**, so
  there is no shuffle, epoch, cursor, loader state, or native mini-batching
  yet. Everything the contract locks is in the design document; read it
  there.
- **Phase J moves no capability at any milestone.** Every §3 row — registries,
  exports, CTests, checkpoint and optimizer-state versions — is expected to be
  unchanged at J9, and the phase plans **no new C ABI export**.

Beyond Phase J (future work, not started): native integer tensors, further
dtypes or devices, CUDA experiments. See `docs/roadmap.md`; never invent a
phase it does not define.

---

## 13. Agent operating rules

### Style

- Keep code simple and readable — clarity beats cleverness. Match the
  existing style: NumPy-only internals, small modules, one concept per file.
- Comments explain math, autograd, and ownership reasoning, not obvious
  Python.
- Losses and metrics stay simple: losses are Tensor expressions or fused ops
  with a custom backward; metrics are plain NumPy returning Python floats,
  outside autograd.
- Examples use fixed seeds and the `train()` + `main()` pattern so tests can
  import `train`.
- Tests use `np.allclose` with sensible tolerances (e.g. `atol=1e-6`);
  training tests assert learning without fragile exact-loss values. Bit-level
  claims use raw IEEE-754 bit patterns, not tolerances.

### Workflow

- **Inspect existing code before editing**; find where a concept lives and
  follow its pattern.
- **One milestone at a time**, and keep changes scoped to it. No unrelated
  features, no drive-by refactors, no framework rewrites. Do not hide extra
  optimization under "cleanup".
- If a requested feature already exists, verify it against the spec and add
  tests or documentation instead of reimplementing it.
- **Preserve all previous tests. Never loosen a test just to pass.**
- Run `uv run pytest` and any requested manual checks before reporting
  success, and **report the actual observed output**.
- A documented rejection backed by measurements is better than an unsafe or
  weak implementation.
- Audit the whole change before it is committed: files changed, contracts
  touched, inventories unchanged. Final responses report files changed, what
  was implemented, tests added, the exact pytest result, manual check
  outputs, and any notes or limitations.
- **No external-project provenance reference** may appear in source,
  documentation, tests, commit messages, or reports. TensorForge's design
  records explain its own decisions.

### Version control

- **The user performs every Git-writing operation.** Implementation agents
  never commit, push, pull, merge, rebase, reset, stash, amend, alter
  remotes, or create, switch, rename, or delete branches.
- Read-only inspection (`git status`, `diff`, `log`, `show`, `rev-parse`,
  `ls-files`, `branch --show-current`) is fine, and is how you confirm the
  starting state.
- Before beginning the next milestone, **verify the remote commit** for the
  previous one actually landed.

### Machine notes

- **Permissions quirk:** directories created by one process often cannot be
  deleted by a later one. Already handled — pytest's cache is redirected to
  `.cache/pytest` (pyproject) and `conftest.py` gives each session a fresh
  basetemp. **Do not try to delete** `.pytest_cache/`, `.cache/pytest-tmp/`,
  or `%TEMP%/pytest-of-*`.
- Two example-test import styles coexist: `tests/test_examples.py` inserts
  `examples/` into `sys.path`; newer tests import `examples.<name>` as a
  namespace package from the repo root.
- The stable root package's export list is enumerated and locked by
  `tests/test_public_api.py`; read it there rather than from a copy. Stable
  checkpoints = weights + optimizer state + optional scheduler state +
  optional RNG state (`rng_state=True` / `restore_rng_state=True`, covers
  unseeded Dropout) + JSON metadata; parameters = weights only.
