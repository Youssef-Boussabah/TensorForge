# TensorForge — project instructions

**This file holds current operating rules and durable invariants only.**
Everything historical — milestone reports, measurements, evidence, rejected
alternatives — lives in `docs/` (§11) and must never be copied back here.

---

## 1. Project identity and architecture

TensorForge is a from-scratch deep learning framework covering PyTorch-style
framework internals. Position it as serious and systems-focused — never
"educational", "toy", or "mini" — while staying honest: **experimental, not
production-ready, and not a PyTorch replacement.**

Two lines live in one repository, and they stay strictly separate (§2):

- **The stable Python line** (`tensorforge`, `tensorforge.nn`,
  `tensorforge.optim`, `tensorforge.data`) — Tensor + reverse-mode autograd
  on NumPy. Complete at **v3.0**; feature-frozen unless a milestone says
  otherwise.
- **The experimental native line** (`tensorforge.backends`,
  `tensorforge.experimental`, `cpp/`) — a C++17 CPU runtime behind a plain C
  ABI loaded with `ctypes`, with its own tensor, autograd, modules,
  optimizers, RNG, checkpoints, and data pipeline. It lives on `main` inside
  those explicit namespaces.

Development is milestone by milestone: small, tested, readable, documented.

### Tech stack

- Python ≥ 3.13, NumPy, pytest — nothing else. Managed with `uv`; run
  everything through `uv run`.
- **Never introduce** PyTorch, TensorFlow, JAX, sklearn, pandas, or
  matplotlib. NumPy is the only numeric dependency; the C++ backend needs
  only a C++17 compiler — no BLAS, oneDNN, Eigen, pybind11, or GoogleTest.

### Layout

- `src/tensorforge/` (stable line) — `tensor.py` is Tensor + reverse-mode
  autograd: ops are primitives (eager NumPy forward plus a `_backward`
  closure holding the local derivative) or derived compositions that get
  gradients for free, and gradients accumulate through `_accumulate_grad`,
  which also un-broadcasts. `nn/` holds Parameter, Module, the layers,
  losses, and metrics: `train()`/`eval()` recurse through children, modules
  declare non-trainable buffers via `self._buffers = ("attr", ...)`, and
  `state_dict()`/`load_state_dict()` cover parameters *and* buffers.
  `optim/` holds SGD, Adam, `StepLR`, and the clippers: `step()` skips
  `None` grads and frozen parameters, `zero_grad()` sets grads to `None`,
  clipping is in place before `step()`. `data.py` is the stable `batches`
  mini-batch iterator.
- `src/tensorforge/backends/cpp.py` — the **only** module that imports
  `ctypes`: library loading, the C ABI bindings, `NativeStorage` /
  `NativeTensorView` / `NativeTensorCore`, and the capability registries.
  `src/tensorforge/experimental/` holds the native tensor, autograd, modules,
  losses, metric, optimizers, generator, state transactions, checkpoints, and
  data pipeline — one concept per file.
- `cpp/src/` + `cpp/include/` — kernels by concern (`elementwise`, `matmul`,
  `reduction`, `conv2d`, `pooling`, `classification`, `random`, `storage`,
  `error`); `tf_*_internal.h` headers hold hidden-visibility helpers and
  kernel templates, and nothing there is exported. `cpp/tests/` —
  dependency-free C++ CTests compiling that source directly, built only with
  `-DTF_BUILD_TESTS=ON`.
- `examples/` — runnable scripts, each with `train(...)` returning stats and
  a `main()` that prints, guarded by `__main__`. `tests/` — the pytest suite;
  every feature has tests. `benchmarks/` — characterization harnesses (§9).
  `scripts/smoke_cpp_backend.py` — the hard-failing smoke check CI runs after
  building. `docs/` — the source of truth for everything historical (§11).
- `.github/workflows/tests.yml` — CI: install uv, build the C++ backend,
  hard-failing smoke check, quick benchmark smoke run, then pytest.

---

## 2. Stable / native separation

**The two lines are strictly separated, and stay that way.**

- The stable framework **never** imports the native backend. Importing
  `tensorforge` must not load the C++ library, and a test proves it.
- Importing the wrapper is always safe — the library loads **lazily**. Check
  `cpp.is_available()` / `cpp.backend_info()`; kernels raise `ImportError` at
  call time when unbuilt, and the backend tests skip.
- `stable_framework_integration` is `False` in `backend_info()` and stays
  false. There is **no** automatic backend selection or routing, no implicit
  dispatch, no implicit stable↔native conversion, and no environment
  variable that changes which line runs.
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
| Loader state format · version · accepted | `tensorforge.native_data_loader` · **1** · `(1,)` |
| Sampler state format · version · accepted | `tensorforge.native_sampler` · **1** · `(1,)` |
| Exported production `tf_*` symbols | **54** (Phase H closed at 52) |
| Experimental Python exports | **25** |
| Native CTests · examples · benchmarks | **24** · **16** · **9** |

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
- **No `device` argument exists anywhere and none may be added**, and there
  is no device movement.
- Classification targets stay **host `int64` metadata** at every width. No
  integer tensor dtype exists.
- Constructors that own numeric state take a **keyword-only** `dtype`
  accepting exactly `"float64"` / `"float32"`, through the one shared private
  validator. A class owning no dtype-bearing state — the losses, the metric,
  `NativeSequential`, `NativeReLU`, `NativeFlatten`, `NativeMaxPool2d`,
  `NativeDropout`, `NativeGenerator`, both optimizers, the sampler, the
  loader — must **not** gain one: a second authority could disagree with the
  data.
- The private typed constructors (`_typed*`, `_trusted_dtype=True`,
  `NativeTensor._from_core`) **stay, and stay private**: "this dtype came
  from live storage or a validated archive" is a different trust statement
  from "a caller said so", and they grant no width the public ones do not.
- The MaxPool2d winner buffer is **private float64 at every value dtype** —
  a permanent pin, not an oversight.
- The NumPy reference backend keeps its own float64-only `supported_dtypes`.

**Performance work never broadens support**: a milestone that makes something
faster leaves every row above untouched.

Not supported, and not a bug: float16/bfloat16, mixed precision, AMP, CUDA or
any GPU backend, integer tensors, float32 raw kernels, distributed training,
C++-side autograd, attention/Transformers. Automatic loader **discovery**
does not exist either, at any milestone — see §12. Full dtype contract,
evidence, and rejected alternatives:
`docs/native_dtype_float32_design.md`.

Integer tensors are on that list **and are the subject of the newly
approved Phase K**, whose contract is
`docs/native_integer_tensors_design.md`. The two facts coexist, and K1
sharpened rather than removed the distinction: **`int64` is an internal
C++ representation, not a supported tensor dtype.** It is not in
`SUPPORTED_DTYPES`, `normalize_dtype("int64")` raises, the Python dtype
tables do not know the name, and **no supported wrapper or public Python
API can allocate or wrap `int64` storage — only the raw private C ABI
can, for isolation and barrier testing.** No public integer constructor,
`argmax`, or index selection exists, and none may be described as existing
until the milestone that ships it — K2 for the `INDEX_DTYPES` registry and
the one public door, K3 and K4 for the operations (§12).

---

## 4. Core invariants

These hold across every phase and may not be weakened by a milestone.

### 4.1 Public API and C ABI discipline

- The public API is locked by tests (`tests/test_public_api.py` for the
  stable root package; the registries and `experimental.__all__` for the
  native line). Adding to it is a milestone decision, as is **adding or
  removing a C ABI export**. Optimizations ship *inside* existing exports.
- Hidden default visibility; `TF_EXPORT` only on functions Python declares.
  The source export inventory and the built library's export table must agree.
- Private helpers stay private. A name private because of what its caller may
  be trusted to know does not become public for convenience.
- **No public performance control of any kind exists or may be added**: no
  kernel or path selector, block-size or threshold setter, traversal or
  dispatch tracer, benchmark hook, profiling counter, "which path ran" query,
  environment-variable dispatch.
- **No production poison, profiling, or allocation-content control.** A symbol
  exported from the normal runtime is part of the runtime however carefully it
  is disarmed. The one documented exception is the deterministic thread-local
  allocation-failure hook (`tf_test_arm_alloc_failure` /
  `tf_fault_injection_available`): inert until armed, changing no buffer
  *contents*, and part of the export count. Do not add a second; do not
  remove this one without a milestone.

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

A new predicate follows the shipped naming (`tf::matmul_prefers_row_sweep`,
`tf::copy_prefers_contiguous`, `tf::reduce_prefers_contiguous_blocks`,
`tf::build_unary_plan`/`tf::build_binary_plan`, the three `tf::conv2d_*`
geometry predicates). The shape was set across Phase H — H4's optimizer step,
H5's copy transfer, H6's reduction execution — and the dtype templates
inherited every path unchanged, so both widths take the same traversal for
the same layout. See `docs/native_cpu_performance_design.md` and
`docs/dispatch_design.md`.

### 4.3 Deliberately absent

None exists anywhere, and none may be added without meeting its own recorded
criteria in `docs/native_cpu_performance_design.md` §10–§13:

memory pool · scratch workspace or arena · persistent cache of native
storage · SIMD intrinsics · threading · OpenMP · BLAS · oneDNN · Eigen ·
im2col · general operator fusion · fast-math · cache blocking.

SIMD, threading/OpenMP, and BLAS were each rejected on measurement;
reopening criteria are there too.

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
their storage, and source/destination aliasing — and when they reject they
**write nothing**. The C ABI is a **second** authority, not a restatement of
Python's: never remove a C-side check because Python performs it.

### 4.5 Determinism

- No kernel consults a clock, a process id, an address, allocation history,
  or static/thread-local state to produce a value.
- Random values come only from the explicit `NativeGenerator` key
  (`tensorforge.splitmix64`; seed plus call index). No `<random>`, no
  `std::random_device`, no implicit global stream; the Phase-J sampler reuses
  that derivation under a domain-separated epoch key schedule rather than
  adding a second. Contract: `docs/native_rng_dropout_design.md`.
- Examples use fixed seeds so output is reproducible.
- **Deterministic training and exact checkpoint resume are proved by test in
  every phase from C onward, and every one of those proofs must keep
  passing.** An interrupted run reloaded into a *fresh*
  model/optimizer/generator set reproduces the loss suffix, every parameter
  and buffer, every optimizer moment and step counter, the generator state,
  and the final training and evaluation outputs by **exact equality**.
- Reproducibility is exact **for the state TensorForge captures**. Python's
  `random`, NumPy's global RNG, data-loader position, batch order, and
  scheduler state are not captured; full-program determinism is not claimed.

---

## 5. Ownership, lifecycle, and transactions

### 5.1 Native storage ownership

- A `NativeTensorCore` owns its `NativeStorage`; a `NativeTensorView`
  borrows and never closes its parent's storage, and a chained view keeps
  the whole chain reachable.
- Every operation allocates a **fresh owning contiguous** output that aliases
  neither operand.
- **Cleanup is explicit and never relies on garbage collection.** `close()`
  is the contract and is idempotent; `__del__` is only a fallback. Any
  failure — allocation, native call, wrapper construction, graph-node
  construction, resource attachment, batch delivery — closes everything it
  allocated, so **live storage returns exactly to baseline** and no caller
  observes one lone result. Details:
  `docs/native_tensor_wrapper_design.md`.
- `close()` exists **exactly where something is owned**, and nowhere else.

### 5.2 Graph-owned saved resources

Four families: Dropout masks, MaxPool2d winners, BatchNorm eval snapshots,
cross-entropy saved probabilities. Each rides the `graph_resources` contract:
released **exactly once** with the graph history, retained under
`retain_graph=True`, kept alive across a failed retryable backward, freed by
an abandoned graph's `close()`, closed immediately by a no-grad forward. A
registered buffer is **never** a rereadable graph operand — BatchNorm eval
reads independent owning snapshots.

### 5.3 Identity and versioning

- `load_state_dict()`, `load_native_checkpoint()`, and the optimizer loaders
  **preserve every parameter, buffer, and generator identity** and every
  sharing relationship, restoring in place.
- A parameter's version counter moves **once** per committed mutation. Shared
  parameters deduplicate to one slot, one update, one increment.
- Loading **buffer** or **generator** state moves no parameter version and
  stales no graph. A **full** checkpoint load replaces parameters and so
  correctly stales an earlier graph through the parameter rule — a parameter
  contract, never a buffer or RNG effect.
- Frozen parameters stay registered and persisted but are skipped by
  optimizers.

### 5.4 Transactional boundaries

Each is atomic under failure, **validated completely before anything is
published**, and on failure leaves identities, versions, and live storage
exactly as it found them — no partial mutation, ever:

output allocation plus wrapper publication · `NativeParameter.copy_value_` ·
optimizer stage/commit · the BatchNorm running-statistics two-buffer
transaction · `NativeModule.load_state_dict` · optimizer `load_state_dict` ·
whole-checkpoint load · generator-state replacement · graph-resource
adoption · the Phase-J batch handoff (§12).

Honest scoping: transactions are **per module**; one whole training step is
*not* globally transactional. Ordinary training mutation does not take the
process-wide state-replacement lock, so thread-safe concurrent training
snapshots are not offered — only *participating* state-replacement operations
serialize with each other, in the universal lock order (the private
process-wide guard first, then every unique generator lock in global `id()`
order, never the reverse). External process or interpreter death is the only
exception to whole-checkpoint atomicity. The Phase-J objects join neither,
hold no lock, and are **not thread-safe**.

### 5.5 Checkpoint and state rules

- The native checkpoint is `tensorforge.native_checkpoint` **version 3**;
  `(1, 2, 3)` are accepted and every new save writes 3. **There is no version
  4** without an explicit milestone.
- Versions 1 and 2 are float64-only permanently, and a payload is never
  *guessed* to be float32. Version 3 declares every numeric entry's dtype.
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
  state is **caller-managed** — serialized by the caller through the
  existing validated version-3 `metadata` channel, which J5 proved end to
  end without the archive growing a field. **No automatic loader discovery
  and no checkpoint/pipeline coupling**, in either direction.

---

## 6. Numerical contracts

**Never publish one universal "bit-identical" claim.** Each operation family
has its own rule, measured rather than inherited; full statements in
`docs/native_cpu_performance_design.md`. The durable summary:

| Family | Contract |
|---|---|
| **Value transfer** (`contiguous_copy`, state/checkpoint transfer) | Reproduces its source's bits **exactly** — `-0.0`, both signs of signaling NaN, every NaN payload — at both dtypes; a transfer performs no arithmetic. An *operation* that happens to copy (`zeros + x`) follows IEEE arithmetic instead, so it *does* normalize `-0.0` and quiet a signaling NaN. |
| **Elementwise** (`add`, `subtract`, `multiply`, `relu`, `relu_backward`, `sqrt`, `reciprocal`) | Bit-identical whenever **at most one operand is NaN** (`subtract` everywhere). With **two** NaN operands the surviving payload is **outside the contract**, asserted in neither direction. |
| **`exp` / `log`** | Library functions with no correctly-rounded IEEE guarantee. Deliberately **excluded** from the templated traversal; the cross-platform contract is a **one-ULP** finite bound, not bit equality. |
| **matmul** | Accumulation order preserved exactly. Every non-NaN result bit-identical; NaN positions identical and always quiet; NaN **payload** bits outside the contract. |
| **Reduction · Conv2d** (all three conv directions) | Accumulation order preserved exactly, per output and per destination, with reduction source traversal not even reordered. Non-NaN results and signed zeros bit-identical (as raw bit patterns); NaN positions identical; bit-identical whenever **at most one NaN** enters an accumulation — including its payload, for conv2d. |
| **Optimizers, normalization, softmax, log-softmax, cross-entropy** | Bit-identical to the composition they replaced: no reassociation, no accumulator-width change, no operand-position change. |

Nothing anywhere reassociates arithmetic, uses FMA, fast-math, an intrinsic,
`restrict`, a tree/pairwise/parallel reduction, or a horizontal vector
reduction.

**Do not "fix" a correctly rounded IEEE result.** Where a float32 shift
overflows because the finite spread exceeds the type's range, that *is* the
correct answer; a widened intermediate, a clamp, or a special case would be
mixed precision by the back door.

### Output initialization

Output storage is **zero-initialized by default**. A call site may opt in to
`tf_storage_create_uninitialized` **only** when the kernel provably
overwrites every destination element before reading it, and only against the
per-kernel audit table. `sum`/`mean` and `narrow_backward` are explicitly
rejected and keep a zeroed destination — the first accumulates into its
output, the second writes only the narrowed region, and the untouched zeros
*are* its gradient.

Completeness is proved by deterministic **poison** tests injected
**exclusively by test infrastructure, around the allocator**, always with a
negative control showing the detector can fail. ASan and UBSan do **not**
detect uninitialized-*value* reads, so neither is that proof.

---

## 7. Build, test, and validation workflow

```bash
uv run pytest                       # the whole suite; expect zero skips
uv run python cpp/build.py          # build the native backend (Release)
uv run python cpp/build.py --debug  # unoptimized, assertions on
uv sync --group cpp                 # only if you have no C++ compiler
uv run python scripts/smoke_cpp_backend.py
uv run python examples/<name>.py    # all 16 run; see docs/examples.md
uv run python benchmarks/benchmark_native_data_pipeline.py --smoke
uv run python benchmarks/benchmark_native_data_pipeline.py --smoke --json
```

`cpp/build.py` wraps the canonical CMake build (`cpp/CMakeLists.txt`), which
owns the real compilation architecture; when CMake is absent it falls back to
one direct compiler invocation over the same source list (what CI uses).
`TF_SANITIZE` and `TF_BUILD_TESTS` are the **only** build options; adding a
third is a milestone decision.

**How to validate a change:**

- Run the focused tests for what you touched first, then **the full
  `uv run pytest` suite and any requested manual check before reporting
  success**, and report the actual observed output.
- **Only claim what you ran.** Never report a Linux run, a sanitizer run, a
  CTest run, or a rebuild that did not happen.
- Documentation- or test-only work needs no native rebuild. Anything touching
  `cpp/` or changing allocation behavior needs the native rebuild, the CTest
  suite, the Linux/CI-equivalent run, and §8.
- Inspect the **architecture**, not the test count: a green suite with a
  weakened contract is a regression.

### Windows validation (the primary development platform)

Build **Release and Debug out-of-source, outside the repository**, writing the
Debug library elsewhere so the active runtime stays the Release DLL. Require
**zero project compiler, linker, and CMake warnings** and the full CTest suite
green in each.

```bash
cmake -S cpp -B <outside-repo>/release -DTF_BUILD_TESTS=ON
cmake --build <outside-repo>/release --config Release
ctest --test-dir <outside-repo>/release -C Release
```

### WSL / Linux validation

Match GitHub Actions: `uv sync --group cpp`, `uv run python cpp/build.py`, the
smoke check, the quick benchmark, then `uv run pytest`. Use an isolated Linux
environment (`UV_PROJECT_ENVIRONMENT`), never the Windows `.venv` or DLL. The
transcendental (`exp`/`log`) contract is a one-ULP bound precisely because
libm differs between MSVC and glibc; never tighten it back to bit equality.

---

## 8. Sanitizer procedure

Every milestone touching C++ or changing allocation behavior must pass this,
on Clang under Linux/WSL (MSVC does not support it):

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
- a **negative control** proving the instrumentation can fail — test-only
  code handing a kernel malformed metadata and producing a real
  `heap-buffer-overflow`. Zero diagnostics means something only when the
  detector is known to work;
- a LeakSanitizer lifecycle in which native live storage returns **exactly**
  to baseline, the remaining process-exit allocations containing **no
  TensorForge frame** and **no suppression file added**.

Notes and results: `docs/backend_experiments.md`.

---

## 9. Benchmark rules

Benchmarks are **characterization, never a test gate.**
`benchmarks/cpp_backend.py` compares raw kernels against NumPy; the per-stack
harnesses beside it (`benchmark_native_autograd`, `_cnn`, `_classification`,
`benchmark_native_normalization`, `_dropout`,
`benchmark_native_cpu_performance`, `_dtype`) characterize their own stacks.
The dtype harness is a separate file on purpose and measures each dtype
**separately**, never as a ratio of one to the other.

Non-negotiable, in every harness:

- **Correctness is gated before timing**, always. A failed gate publishes no
  timing and the CLI exits nonzero with clean stdout.
- **No speed is asserted anywhere**: no timing threshold, no performance
  budget, no committed duration, **no CI job that fails on a number**.
- **No result file of any kind is written**, in any phase. A committed number
  becomes a promise the project cannot keep across machines.
- A case with no honest equivalent is labelled `native_only` and publishes
  **no ratio at all**. Never fabricate a comparison layer.
- Setup, cleanup, and advanced state stay outside the timer; temporaries are
  closed explicitly, not left to GC; a case whose call advances persistent
  state resets it per repetition.
- Report medians with spread after warm-up; publish regressions, neutral
  results, and noise as prominently as wins.

**Measurement methodology:**

- Use **alternating pre/post rounds in separate subprocesses**, and prove
  every case **bit-identical before either side is timed**.
- **Low round counts lie**: never quote one as evidence. State the machine's
  **control band** (identical-code cases); a reading inside it is neutral.
- Whole-translation-unit **code-layout effects are real** — adding code to one
  `.cpp` can move an unrelated function's timing on byte-identical source.
  Publish it; do not chase it.
- On small inputs a fixed per-call Python-plus-ctypes cost dominates and
  kernel work is invisible — an architectural floor, not a defect.

---

## 10. Restrictions when writing code

Do not, without an explicit milestone that says so: add or remove an exported
`tf_*` symbol or change the C ABI; add a public API, capability-registry
value, dtype, device, checkpoint field, or checkpoint version; add a build
option, a required dependency, or a mandatory `-march`/`/arch` flag;
introduce anything from §4.3; add a timing assertion, a committed benchmark
number, or a result file; weaken a validation, an error type, or an error
message in the name of speed; or couple the stable line to the native one.

---

## 11. Documentation map

Detailed phase records belong in these documents, **not** here. A milestone
that changes the public API or the examples updates the matching document
(and README links) **in the same milestone**.

| Question | Authoritative document |
|---|---|
| Supported, right now | `docs/native_support_matrix.md` |
| Architecture · overview/status | `docs/architecture.md`, `docs/project_summary.md` |
| Per-release history · what is planned next | `docs/release_history.md`, `docs/roadmap.md` |
| C++ backend, builds, sanitizers · C ABI errors | `docs/backend_experiments.md`, `docs/native_abi_error_contract.md` |
| Stable autograd, training, examples | `docs/autograd.md`, `docs/training.md`, `docs/examples.md` |
| Native wrapper, ownership, autograd | `docs/native_tensor_wrapper_design.md`, `docs/native_autograd_design.md` |
| Dispatch, broadcasting, reductions, dtype metadata | `docs/dispatch_design.md`, `docs/native_contiguous_fast_path_design.md`, `docs/native_broadcasting_design.md`, `docs/native_reductions_design.md`, `docs/native_dtype_device_metadata_design.md` |
| **Phase D** CNN · **Phase E** classification & stable math | `docs/native_cnn_design.md`, `docs/native_classification_design.md` |
| **Phase F** normalization & buffers · **Phase G** RNG & Dropout | `docs/native_normalization_design.md`, `docs/native_rng_dropout_design.md` |
| **Phase H** CPU performance · **Phase I** dtype & float32 | `docs/native_cpu_performance_design.md`, `docs/native_dtype_float32_design.md` |
| **Phase J** data pipeline & mini-batching | `docs/native_data_pipeline_design.md` |
| **Phase K** native integer tensors & indexing | `docs/native_integer_tensors_design.md` |

---

## 12. Current status

- **Stable Python line: complete at v3.0**, feature-frozen.
- **Native line: Phases A–I are complete** — CPU runtime (A) through dtype
  generalization and float32 CPU support (I); per-phase subjects are in the
  §11 map and `docs/release_history.md`. **Phase I** (I0–I11) is complete
  and closed the dtype work; Phase J closed after it, so **Phase J is the
  latest completed phase**. Phase I's one public capability change, float32 joining
  `SUPPORTED_DTYPES`, landed only after the integrated exact-resume proof
  passed. That ordering is the rule: **prove first, then promise.**
- **Native line: Phase J is complete (J0–J9)** — Deterministic Native Data
  Pipeline and Mini-Batching, authority `docs/native_data_pipeline_design.md`.
  It was approved **after** Phase I closed and was not on the earlier
  roadmap; never describe it as pre-existing plan work. **It moved no
  capability at any milestone**: every §3 row is exactly what Phase I left,
  and it added **no C ABI export**. J1–J3 each added exactly one public name
  — `NativeTensorDataset`, `NativeBatchSampler`, `NativeDataLoader` — and
  **every other milestone added none**, leaving **25** experimental names,
  **16** examples, and **9** benchmarks. Per-milestone records live in
  `docs/native_data_pipeline_design.md` §23 and `docs/release_history.md`;
  what follows here are the rules that outlive them.
- **No Phase-J milestone remains.** That sentence read "and no successor
  phase is defined" for as long as it was true — Phase J closed at J9
  without one, deliberately — and **Phase K was approved afterwards**.
  Record it that way rather than rewriting it: "the phase that came next"
  and "the phase that was always planned next" are different facts.
- **Native line: Phase K — Native Integer Tensors and Indexing — is the
  newly approved phase and is the latest phase, and only K0 and K1 have landed.**
  Authority
  `docs/native_integer_tensors_design.md`. **K0 is architecture, contract,
  status, and guardrails only and added no runtime behavior at all**: no
  integer dtype or dtype code, no C++ enumerator, no kernel, no C ABI
  symbol, no ctypes declaration, no public export, no registry movement,
  no checkpoint/optimizer/loader/sampler version change, no example, no
  benchmark, no CTest, and no dependency. **K1 added the internal `int64`
  representation and every reachability barrier, and no public capability
  at all**: `TF_DTYPE_INT64 = 2` / `tf::Dtype::Int64` with allocation,
  destruction, and **bit-exact** arms on the four transfer boundaries
  (`copy_from`, `copy_to`, `materialize`, `tf_core_contiguous_copy`);
  `tf::require_floating` on the **32** float-only exports, applied ahead
  of `require_matching_dtype` so a mixed float/integer call is a **role**
  error; the nine §5.4 Python narrowings from `_normalize_internal_dtype`
  to `normalize_dtype`; and every §6.5 barrier —
  `NativeTensorCore.__init__`, `NativeTensor.__init__`, `_from_op`
  (closing the core and the saved resources it was handed), `backward`,
  `_accumulate_grad`, `NativeParameter`, `register_buffer` at **both**
  `persistent` values, both optimizers, `_validated_entry_dtype`, and
  every floating operation entry. `tf_storage_fill`/`tf_storage_scale`
  gained the guard **because** they can now record a rejection, and stay
  unhooked, so `_CHECKED_KERNELS` is unchanged at 36. CTests **24 → 25**;
  **no** C ABI symbol, public name, registry value, or version moved.
  Every other §3 row is exactly what Phase J left.
  **`int64` is not a supported native tensor dtype**, the Python dtype
  tables do not know the name, **no supported wrapper or public Python API
  can allocate or wrap `int64` storage — only the raw private C ABI can**,
  no public integer constructor, `argmax`, or index selection exists, and
  **K2 through K9 are
  unstarted** — public construction begins at K2. The one public capability
  change of the phase is a separate `INDEX_DTYPES == ("int64",)` row at
  **K2**; **`SUPPORTED_DTYPES` never gains `int64`** and
  `normalize_dtype("int64")` keeps raising, so no generic constructor
  changes what it accepts at any milestone, and every reachability barrier
  landed at **K1**, before an integer tensor can be constructed at all:
  **prove first, then promise.** The phase's C ABI maximum is **56**
  (54 + `argmax` at K3 + `index_select` at K4) and
  `experimental.__all__` stays at **25**
  throughout.
- Further work beyond Phase K — further dtypes or devices, CUDA
  experiments — requires a **separately approved** phase with its own
  design contract. See `docs/roadmap.md`; never invent a phase or a
  milestone it does not define.

### The native data pipeline — the rules that govern it

- **Three objects, one direction.** `NativeTensorDataset` holds two owned,
  copied host snapshots and answers "given these indices, what is the
  batch?"; `NativeBatchSampler` is a **planner** with explicit
  `epoch`/`cursor`, pure planning, no native allocation and **no
  `close()`**, whose order is a pure function of `(seed, epoch, length)`
  from the private `_native_permutation`; `NativeDataLoader` iterates.
  `close()` exists exactly where something is owned, and the sampler owns
  nothing.
- **The batch handoff is a five-phase transaction** (design §9.4) under one
  absolute invariant: **the committed position advances if and only if a
  batch was delivered to the caller.** Every failure closes the undelivered
  `NativeTensor` and restores the exact pre-delivery `epoch`/`cursor`
  through the non-failing seam a state load commits with, so a retry
  returns the same batch with the same values in fresh storage. Rollback
  order is **restore position → clear the record → close the tensor**. A
  delivered batch is the **caller's** and no close path may reach one;
  `_deliver_batch` is a private **test seam**, never a hook. Never add a
  public advance, reset, iterator class, delivery hook, collate, transform,
  worker, prefetch, pinned memory, or `__len__`.
- **`NativeDataLoader` has exactly two state methods**, `state_dict()` and
  `load_state_dict(state)`. The state is a **three-key tagged wrapper** and
  the shape is contractual: `format` = `"tensorforge.native_data_loader"`,
  `format_version` = **1**, `sampler` = **exactly** the unchanged version-1
  `"tensorforge.native_sampler"` state. No epoch, cursor, seed, shuffle,
  batch-size, or drop-last field may be duplicated at the root — the loader
  owns none of them. Private constants only (`_FORMAT`, `_FORMAT_VERSION`,
  `_SUPPORTED_FORMAT_VERSIONS` = `(1,)`, `_STATE_FIELDS`); no version 2, no
  alias tag, no migration path.
- **`state_dict()` is pure and fresh**: a new root, sampler, dataset dict
  and `feature_shape` list at every call, sharing nothing with the objects,
  the cache, or a previous result; JSON-compatible and accepted unchanged
  by the checkpoint's `_validated_metadata`; carrying no permutation,
  payload, NumPy object, serial, token, or id, and nothing that grows with
  the sample count. Allowed between batches, after exhaustion or
  supersession, with a closed dataset, and **after the loader closes** — and
  **refused (`RuntimeError`) while a §9.4 transaction is in flight**,
  through the *sampler's* existing guard, never a second authority: no
  snapshot may observe a skipped-but-undelivered position.
- **`load_state_dict()` order is fixed** (design §12.5), each step proved by
  precedence with malformed input: closed guard → transaction guard →
  active-iteration guard (all three *before* `state` is read) → exact `dict`
  → exact three-key set → `format` type then value → `format_version` type
  (`bool` rejected) then value → nested `dict` → **the whole nested sampler
  validation delegated to `NativeBatchSampler._validate_state`** → commit
  via `_assign_state`. Never restate a nested rule in the loader, never call
  the sampler's public `load_state_dict` from it, and never add a loader
  rollback: nothing mutates until the only remaining step cannot fail.
  Dataset identity is **validated, never adopted**; the six configuration
  and position values **are** adopted; loader, sampler, and dataset identity
  are preserved absolutely. A rejected load leaves the entire observable
  world — including the cache's behavior, the iterator slot, and live native
  storage — byte-identical.
- **Loader state travels as caller-managed checkpoint metadata**, and the
  supported order is fixed: **save** = `loader.state_dict()` → *no
  iteration* → `save_native_checkpoint`; **restore** =
  `load_native_checkpoint` **first**, then
  `loader.load_state_dict(metadata[...])`. `"training"`, `"data_loader"`,
  and `"next_step"` are **caller conventions**; no production constant may
  spell one, and alternate keys and nesting must keep working. The
  checkpoint **preserves metadata and never interprets it**: it validates
  JSON-compatibility only, invents no default for an absent loader state,
  and calls no loader method. Malformed loader state is preserved by the
  archive and rejected by the *loader*. The three delivery boundaries are
  contractual: a **failed** delivery resumes the same candidate batch, a
  **successful** one the following batch, an epoch-boundary save the
  canonical `(epoch + 1, 0)`.
- **There is no cross-object atomicity, and none may be added.**
  `load_native_checkpoint` is atomic over model, optimizer, and generators;
  `loader.load_state_dict` over loader and sampler; `__next__` over one
  handoff. If the first succeeds and the second fails, **nothing rolls
  back** — the caller discards everything and repeats both calls. Never
  couple the two to manufacture one transaction.
- **How a caller uses this**, and the reference is
  `examples/native_minibatch_training.py`: build dataset → sampler →
  loader, record `loader.sampler.next_batch_indices()` **before** each
  delivery, train one step per delivered batch, and close every delivered
  feature batch, logits, loss, and gradient explicitly. One iterator is one
  epoch: on `StopIteration` call `iter(loader)` again and continue at the
  canonical next-epoch position — never reset the sampler, never touch
  `epoch`/`cursor`.
- **Concurrency stays a documented boundary, never a tested safety
  claim.** No Phase-J module contains a lock, thread, queue, future, or
  async primitive; the objects join no lock order; external locking is the
  caller's job; **no test starts a thread**, and none may.
- **Still absent, and asserted absent**: automatic loader discovery, any
  registry, imports in either direction between the checkpoint and pipeline
  modules, a checkpoint loader field, and checkpoint version 4.

### Proof and scanner discipline these milestones established

- **Exact equality only** for same-run resume — raw IEEE-754 bits through
  `uint32`/`uint64` views, never `allclose`, `approx`, or any tolerance.
  Each dtype is proved **only against itself**; the *only* cross-dtype
  claim is the batch-index and permutation sequence, which carries no
  dtype. An interruption must be genuinely mid-epoch (`0 < split < total`,
  not the last step, not a multiple of `batches_per_epoch`, batches still
  owed), the restored graph entirely fresh and **proved different before
  the load**, and the omitted-state leg proved to **diverge**. Committed
  plans belong in the test as literals. Exact same-run resume and the
  **one-ULP `exp`/`log` bound** are different questions — never conflate
  them or tighten the second.
- **Every rejection and every injected failure is followed by a complete
  before/after fingerprint of the observable world** (dataset · sampler
  including its private transaction, participation, and cache bookkeeping ·
  loader · iterator · an unrelated `NativeParameter` with version and
  gradient · a buffer · a live optimizer · a registered `NativeGenerator` ·
  the filesystem · both global RNGs · every registry), and **every
  injection and every parser has a non-vacuity control** — including the
  fingerprint itself, whose every component is proved able to notice the
  change it exists for.
- Failure positions stay **distinct injections** and none may be labelled
  as another: host gather, native allocation (the existing thread-local arm
  only, disarmed in a `finally` *and* an autouse fixture), host→native
  transfer, and target copy are four, not one. A commit-step injection must
  run the candidate assignment **and then** raise, which is what makes it
  different from failing instead of applying. A `BaseException` proves the
  `finally` unconditional. Two contracted counters — the transaction serial
  and the participation token — legitimately advance on a *failed* attempt
  because neither is ever reused; assert that explicitly rather than
  excluding it quietly.
- Abandonment is proved by explicit `close()`; **no assertion may depend on
  collection timing.** A `live_storages` tracker installs itself **outside**
  `monkeypatch`, so a mid-test `undo()` cannot silently disarm it.
- **Source scans read code, not prose**: strip docstrings and string
  literals through the **AST** first, and read keyword-argument names too or
  `_trusted_dtype=True` is invisible. A substring ban fails on the very
  sentence that documents the prohibition. Executable example code stays on
  **public APIs only**. Every scanner needs a negative control.
- **The pipeline benchmark answers four separate questions** and never
  blurs them into one — `dataset_indexing`, `batch_planning`,
  `permutation_construction`, `host_to_native_materialization` — with
  `loader_delivery` a fifth that is explicitly a **composition** and never a
  substitute. Identity `tensorforge.native_data_pipeline` / `"1.0"` /
  schema **1**, as module constants: there is no benchmark registry in the
  package. Two reference labels only: `numpy` for the host-only indexing
  cases, each stating its `ratio_meaning`, and `native_only` for everything
  else, publishing **no ratio at all**. **Never divide a native case that
  allocates by a NumPy case that does not.** Cold and warm permutation
  construction are separate cases and are never averaged; no cache-control
  API exists and none may be added. Everything else follows §9.

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
  training tests assert learning, not fragile exact-loss values. Bit-level
  claims use raw IEEE-754 bit patterns, not tolerances.

### Workflow

- **Inspect existing code before editing**; find where a concept lives and
  follow its pattern.
- **One milestone at a time**, scoped to it. No unrelated features, no
  drive-by refactors, no framework rewrites, no extra optimization hidden
  under "cleanup".
- If a requested feature already exists, verify it against the spec and add
  tests or documentation instead of reimplementing it.
- **Preserve all previous tests. Never loosen a test just to pass.**
- A documented rejection backed by measurements beats a weak implementation.
- Audit the whole change before it is committed: files changed, contracts
  touched, inventories unchanged. Final responses report files changed, what
  was implemented, tests added, the exact pytest result, manual check outputs,
  and any limitations.
- **No external-project provenance reference** may appear in source, docs,
  tests, commit messages, or reports.

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
  basetemp. **Never try to delete** `.pytest_cache/`, `.cache/pytest-tmp/`, or
  `%TEMP%/pytest-of-*`.
- Two example-test import styles coexist: `tests/test_examples.py` inserts
  `examples/` into `sys.path`; newer tests import `examples.<name>` as a
  namespace package from the root.
- Read the stable export list from `tests/test_public_api.py` (§4.1), never
  from a copy. Stable checkpoints = weights + optimizer state + optional
  scheduler state + optional RNG state (`rng_state=True` /
  `restore_rng_state=True`, covers unseeded Dropout) + JSON metadata;
  parameters = weights only.
