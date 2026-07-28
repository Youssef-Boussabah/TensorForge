# Native CPU Performance and Runtime Efficiency — Phase H design

**Status: milestone H0 complete. No production numerical kernel has been
optimized, and no supported capability has changed.**

This document is the architecture contract for **Phase H — Native CPU
Performance and Runtime Efficiency**, the phase that follows the
completed Phase G (native RNG and Dropout). It is written at milestone
**H0**, whose entire job is to *find out what is actually slow* before
anything is made faster, and to lock the rules any later optimization
must obey.

H0 shipped four things and nothing else:

1. this contract;
2. `benchmarks/benchmark_native_cpu_performance.py`, the unified
   measurement harness;
3. `tests/test_native_cpu_performance_benchmark.py`, its behavioral
   contract tests;
4. documentation reconciliation across every status surface.

**H0 changed no C++, no C ABI symbol, no ctypes declaration, no
`NativeTensorCore` method, no autograd operation, no module, no loss,
no metric, no optimizer, no export, no capability registry, no dtype,
no device, and no checkpoint format.** `UNSUPPORTED` still reads
`("float32", "cuda", "amp")`, `SUPPORTED_DTYPES` still reads
`("float64",)`, `SUPPORTED_DEVICES` still reads `("cpu",)`, and the
native checkpoint format is still `tensorforge.native_checkpoint`
version **2** with versions **(1, 2)** supported.

---

## 1. Why CPU performance, and why now

### 1.1 The honest position

Phases A–G built a native stack that is **correct, owned, atomic, and
reproducible**: C++-owned storage with exactly-once cleanup, a
Python-managed reverse-mode graph with parameter versioning and
stale-graph detection, graph-owned saved resources, whole-checkpoint
transaction atomicity, a deterministic reservation-protected RNG, and
bit-exact stochastic resume. What it has never been is *fast*. Every
kernel in `cpp/src/` is a deliberately plain reference loop. The matmul
comment says so in the first line of the file:

> All are deliberately unoptimized reference loops (no blocking beyond
> the explicit tile, no SIMD, no BLAS).

That was the right call seven phases running: a wrong fast kernel is
worse than a slow correct one, and every phase so far had a *semantic*
deliverable that had to be gotten right first. There is no semantic
deliverable left in the CPU line. The remaining honest gap is
efficiency, and it is now large enough that it shapes what the project
can demonstrate: a training step that costs milliseconds where it should
cost tens of microseconds bounds every example, every benchmark, and
every future experiment.

### 1.2 Why this precedes CUDA

CUDA is the obvious next headline, and it is deliberately *not* next.

- **A GPU backend inherits the CPU runtime's structure.** Every cost H0
  measures below the kernel — the per-call Python metadata path, the
  eager zero-fill on every allocation, the per-operation allocation
  count in a composed module or an optimizer step — is *device
  independent*. A CUDA backend built on top of them would carry them
  onto a device where a kernel launch already costs microseconds, and
  the per-call overhead would go from bad to dominant. Fixing the
  runtime first means the GPU path inherits a good one.
- **The reference the GPU needs does not exist yet.** A CUDA kernel is
  validated against a CPU kernel. A CPU kernel nobody has profiled is a
  poor reference for judging whether a GPU kernel is doing well.
- **There is no measurement instrument.** Phases D, E, F, and G each
  shipped a characterization benchmark, but each covers one phase's
  surface. Nothing measured the runtime *as a whole*, across layers, so
  no ranking of costs existed. H0 builds that instrument; a CUDA phase
  would need one anyway.
- **CUDA is a hardware dependency.** It cannot run in this repository's
  CI, cannot be validated by the existing ASan/UBSan matrix, and cannot
  be reproduced by a reader without a specific GPU. CPU work is
  validated everywhere the project already validates.

### 1.3 Why this precedes dtype expansion

`float32`, mixed precision, and AMP are all *also* performance
arguments — narrower types move less memory and admit wider SIMD. But
they are the wrong first move here:

- The measured bottlenecks in §3 are **not** memory-bandwidth-bound
  arithmetic. The largest single measured factor is an allocator
  behavior, the second is a memory *access pattern*, and the third is
  Python-side per-call metadata. Halving the element width improves
  none of them proportionally, and improves the Python path not at all.
- A second dtype **multiplies** the kernel surface. Every optimization
  Phase H lands would have to be written, tested, and sanitized twice.
  Doing the optimization once, at one dtype, and then widening is
  strictly less work and strictly less risk.
- The dtype boundary is a *semantic* contract (promotion, casting,
  accumulation type) that deserves its own phase with its own design
  document, exactly as every previous semantic boundary got.

### 1.4 What Phase H is for

Phase H makes the existing float64 CPU runtime meaningfully faster
**without giving up a single guarantee Phases A–G established**. Every
invariant in §9 is preserved. The measure of success is not a headline
ratio; it is that the same programs produce the same results and the
same reproducibility proofs, in less time.

---

## 2. Scope

**In scope:** the execution efficiency of the existing native float64
CPU runtime — its kernels, its traversal strategies, its allocation
behavior, its Python-side dispatch path, and the per-operation cost of
the composed modules and optimizers built on it.

**Out of scope (Phase-H non-goals):** see §17. Nothing in Phase H adds
a numerical capability, a dtype, a device, an operation, a module, or a
checkpoint field.

---

## 3. Evidence

Everything in this section was measured on the H0 harness or on
throwaway profiling scripts driving the same public APIs, on one
machine (Windows 11, Intel64 Family 6 Model 170, Python 3.13.14, NumPy
2.5.1 with scipy-openblas 0.3.33, MSVC Release build). **These are local
characterizations, not a performance contract**, and no test asserts any
of them.

### 3.0 A required caveat about this machine

The **native** columns are highly reproducible: across five independent
runs the production 128×128 matmul measured 1.68–1.79 ms with a
within-run spread under 50 µs. The **NumPy** column is not: the same
128×128 `a @ b` measured 277 µs, 456 µs, 724 µs, 4.15 ms, and 9.12 ms
across those runs, because OpenBLAS spins up a thread pool and this
machine has variable background load.

Two consequences, applied throughout this section:

- **Native-versus-native comparisons are the load-bearing ones** (tiled
  versus naive; contiguous versus strided; Adam versus SGD). They hold a
  common measurement environment.
- **Native-versus-NumPy ratios carry large uncertainty on this
  machine.** They are reported as ranges and only used where the factor
  is an order of magnitude or more.

Anything below that depends on a difference smaller than ~2× is marked
as an unconfirmed hypothesis in §3.3, not as a finding.

### 3.1 Directly measured bottlenecks

Ranked by measured leverage.

#### B1 — Every native allocation eagerly zero-fills its buffer

`tf_storage_create` allocates with `new (std::nothrow) double[size]()` —
value-initialization, i.e. a full write pass over the buffer — and every
native operation allocates a fresh owning output.

| Measurement | Value |
|---|---|
| `NativeStorage(262144)` construct + close (2 MB) | 445–503 µs |
| `numpy.zeros(262144)` | 8.5–16.4 µs |
| `NativeStorage(1048576)` construct + close (8 MB) | ~1.90 ms |
| `numpy.zeros(1048576)` | ~27.5 µs |
| `tf_core_add_contiguous` kernel alone, 512×512, preallocated output | ~107 µs |
| full `core.add`, 512×512 (allocation included) | ~647 µs |
| allocation share of that operation | **~74 %** |

`numpy.zeros` is served by `calloc`, which the OS can satisfy with lazy
zero pages; `new double[n]()` cannot. At 2 MB the eager fill is a
*second full write pass* over an output the kernel is about to overwrite
completely.

**And for most kernels the zero is provably redundant.** Auditing
`cpp/src/`:

- **Fully overwrite their destination** (zero-fill is pure waste):
  `tf_core_add`/`subtract`/`multiply` and their `_contiguous` variants,
  `tf_core_relu`, the unary family (`sqrt`, `reciprocal`, `exp`, `log`),
  `tf_core_matmul`, `tf_core_softmax_forward`,
  `tf_core_log_softmax_forward`, `tf_core_cross_entropy_forward`,
  `tf_core_conv2d_forward`, `tf_core_maxpool2d_forward`,
  `tf_core_dropout_forward`.
- **Require a zeroed destination** (the zero is load-bearing):
  `tf_core_sum` (`dst[out_pos] += …`, so zero is the additive identity),
  `tf_core_narrow_backward` (un-narrowed cells must keep their zero),
  and the scatter-add convolution and pooling backwards — which already
  zero-initialize their own output explicitly.

Evidence level: **directly measured**, with the kernel audit read from
source.

#### B2 — Matmul is bound by its memory access pattern, not by scalar arithmetic

`tf_core_matmul` is an `i`–`j`–`k` triple loop. In the inner `k` loop it
reads `b[b_offset + k*b_stride0 + j*b_stride1]`, which for a row-major
right operand strides by `p` doubles per step — a column walk. Two
independent measurements confirm the access pattern is the cost driver:

| Measurement (128×128×128) | Median | Relative |
|---|---|---|
| `tf_core_matmul`, contiguous right operand | 1.68–1.69 ms | 1.00× |
| `tf_core_matmul`, **transposed-view** right operand | 651–656 µs | **0.39×** |
| `tf_matmul` (raw-buffer naive, same loop order) | 1.65–1.79 ms | ~1.0× |
| `tf_matmul_tiled` (raw-buffer, `i`–`k`–`j`, block 32) | 478–546 µs | **~0.30×** |

At the profile shape (384×384×384): naive 47.34 ms, tiled 14.25 ms,
production `tensor_core` 44.56 ms — the tiled kernel is **3.3× faster**
than the production path.

The transposed-view row is the tell. A *strided* right operand is
**2.6× faster** than a contiguous one, because a transposed view has
`b_stride0 == 1`, which makes the inner `k` loop contiguous in `b`. The
kernel is not slow because it lacks SIMD; it is slow because it walks
`b` down a column.

**And the fix is free numerically.** `tf_matmul_tiled` is `i`–`k`–`j`
with ascending `k` blocks and ascending `k` inside a block, so for any
fixed `(i, j)` the products are accumulated **in exactly the same order**
as the naive kernel. Measured directly at `n ∈ {16, 33, 64, 100, 128}`
and block sizes `{8, 16, 32, 64}`, including sizes that do not divide
the dimensions: **`tf_matmul_tiled` is bit-identical to `tf_matmul` in
every case.** Both differ from NumPy (max |Δ| ≈ 3.6e-15 at n = 64),
because BLAS uses its own order — which is exactly why the project
compares reductions and products to NumPy by tolerance and not
bit-for-bit.

Evidence level: **directly measured**, including the bit-exactness claim.

#### B3 — A fixed ~9–22 µs is paid per native operation, and most of it is Python

A one-element `add` costs 18.6–22.6 µs through `NativeTensorCore`
against 0.9–1.1 µs for the NumPy equivalent. Decomposing that fixed cost:

| Component | Median |
|---|---|
| raw `ctypes` call into `tf_core_add_contiguous` (1 element) | ~1.9 µs |
| `NativeStorage._require_open()` | ~0.2 µs |
| `cpp.row_major_strides((1, 1))` | ~2.2 µs |
| `cpp.shape_info((1, 1))` | ~6.6 µs |
| `NativeTensorView(storage, (1, 1))` | ~8.2 µs |
| `NativeTensorCore.zeros((1, 1))` + close | 8.9–14.3 µs |
| full `core.add((1, 1))` + close | 10.2–18.7 µs |

**The ctypes boundary is roughly a tenth of the fixed cost.** The rest is
Python-side shape/stride normalization and view construction. `cProfile`
agrees: over 20 CNN training steps, `cpp.py:_as_int_tuple` is the
single largest `tottime` entry with **15,100 calls** (755 per step), and
in 50 `NativeLayerNorm` forwards it is the second largest with 3,750
calls (**75 per forward**).

Evidence level: **directly measured**, decomposed and profiled.

#### B4 — The optimizer step dominates a small training step, and the cost is call count, not arithmetic

`NativeAdam._stage_entry` composes the update from existing Core
operations (no division: `reciprocal` + `multiply`), which means a
**fixed number of small native calls and allocations per parameter,
independent of parameter size**.

| Measurement | Value |
|---|---|
| `NativeStorage` allocations per Adam step, 2 parameters | 54 (**27 per parameter**) |
| `NativeStorage` allocations per SGD step, 2 parameters | 10 (5 per parameter) |
| `NativeAdam.step()`, 2 parameters of (128, 128) | ~1.83 ms |
| NumPy Adam arithmetic on the same (128, 128) | ~70 µs |
| `NativeAdam.step()`, 2 parameters of (32, 32) | ~627 µs |
| NumPy Adam arithmetic on the same (32, 32) | ~10.8 µs |
| `NativeAdam.step()` vs `tensorforge.optim.Adam`, harness case | 28–34× |
| `NativeSGD.step()` vs `tensorforge.optim.SGD`, harness case | 42–46× |

The (32, 32) row is the point: the arithmetic shrank 6.5× and the step
time shrank only 2.9×, because the allocation and dispatch count did not
move at all.

In a complete MLP Adam training step, **108 of 130 native allocations
(83 %) are the optimizer's**; the forward alone is 5.

Evidence level: **directly measured**, with allocation counts
instrumented at `NativeStorage.__init__`.

#### B5 — Reductions have no contiguous fast path

The elementwise family gained a flat, index-free kernel at v1.14, but
`tf_core_sum` never did: every reduction, contiguous or not, runs the
generic odometer with a per-element carry loop.

| Measurement | Value |
|---|---|
| `core.sum(axis=0)`, 256×256 contiguous | 94–206 µs |
| `numpy.sum(axis=0)`, same | 10–14 µs |
| `core.sum()` all-elements, 512×512 | ~511 µs |
| `numpy.sum()`, same | ~104 µs |

This is also why the composed convolution **bias gradient** exists as a
separate concern: it is `g.sum(0).sum(1).sum(1)` — three chained
reductions, each with its own allocation and its own odometer walk.

Evidence level: **directly measured**.

#### B6 — Composed modules pay B1+B3 once per composed operation

Neither normalization module has a kernel; both are compositions of
existing native operations. That is a deliberate and good design
decision (F2–F4 added no C++ at all), and its cost is now visible.

| Measurement | Value |
|---|---|
| `NativeLayerNorm` forward, (128, 64) | ~306 µs, **11 allocations** |
| `NativeBatchNorm1d` training forward, (128, 64) | ~546 µs, **27 allocations** |
| `NativeBatchNorm1d` eval forward, (128, 64) | ~267 µs |
| one `core.mean(axis=1, keepdims=True)`, same shape | ~39 µs |
| one broadcast `core.subtract`, same shape | ~79 µs |
| normalized training step vs the stable line | 4.0–6.3× |

Evidence level: **directly measured**.

#### B7 — No convolution component dominates at realistic shapes; the *composed* bias gradient does at small ones

| Shape | forward | input grad | weight grad | bias grad (3 sums) |
|---|---|---|---|---|
| 4×1×8×8 → 4, k3 | 19.6 µs (16.0 %) | 19.3 µs (15.7 %) | 18.7 µs (15.2 %) | **65.1 µs (53.1 %)** |
| 12×1×6×6 → 4, k3 | 20.4 µs (16.3 %) | 21.8 µs (17.4 %) | 20.5 µs (16.4 %) | **62.3 µs (49.8 %)** |
| 8×8×16×16 → 16, k3 | 1.89 ms (32.1 %) | 2.12 ms (36.0 %) | 1.76 ms (29.9 %) | 114 µs (1.9 %) |

At the shapes the shipped CNN examples actually use, the *reduction
composition* is half the convolution backward cost; at larger shapes the
three real kernels split roughly evenly with the input gradient
marginally ahead. **No single convolution kernel is the CNN
bottleneck.**

Separately, the native convolution forward is already **faster than the
stable line** — 349–503 µs versus 1.14–1.36 ms at 8×3×16×16, a ratio of
0.28–0.37×. Convolution is the one place where the native kernel is
currently ahead.

Evidence level: **directly measured**.

#### B8 — `NativeTensor` adds a negligible, size-independent cost over `NativeTensorCore`

| Shape | `NativeTensorCore.add` | `NativeTensor.add` | with graph |
|---|---|---|---|
| (8, 8) | 8.90 µs | 9.60 µs (+0.70) | 10.30 µs (+1.40) |
| (64, 64) | 10.40 µs | 11.70 µs (+1.30) | 11.80 µs (+1.40) |
| (256, 256) | 49.7 µs | 71.1 µs (+21.4) | 72.5 µs (+22.8) |
| (512, 512) | 618.8 µs | 655.9 µs (+37.1) | 671.1 µs (+52.3) |

The wrapper's real cost is the **+0.7 to +1.4 µs** visible at small
shapes, where the measurement is stable. The larger apparent deltas at
256² and 512² fall inside the observed run-to-run spread for those rows
(5.1 µs and 155.9 µs respectively within a single run, and much larger
across runs), and the harness's full-run ordering is not even consistent
about their sign — several runs report the *graph* layer as the fastest
of the three, which is impossible as a real effect.

**Finding: the wrapper and the graph node are not a bottleneck.** This
is a negative result and it is load-bearing: it rules out an entire
family of proposed optimizations.

Evidence level: **directly measured**; the negative conclusion is firm,
the exact per-call figure is ±1 µs.

#### B9 — Contiguous copies occur in hot paths, but only once or twice per step

Instrumenting `NativeTensorCore.contiguous_copy`:

| Workload | copies |
|---|---|
| one CNN training step, contiguous NCHW input | **1** (`NativeFlatten`) |
| one CNN forward, narrowed NCHW input | **2** (Policy-B + Flatten) |
| one MLP training step | **0** |

A materialization of a 256×256 transposed view costs 117–170 µs against
NumPy's 49–91 µs. Real, but one or two per step is not where the time
goes.

Evidence level: **directly measured**.

### 3.2 Strongly source-evidenced but not fully measured

- **`_as_int_tuple` / `shape_info` / `row_major_strides` are called far
  more than once per operation.** The call counts are measured (755 per
  CNN step); what is *not* separately measured is how much of B3's fixed
  cost each individual helper contributes inside a real composed
  forward, as opposed to in isolation. H3 will need per-callsite counts.
- **`_layout_arrays()` builds fresh NumPy `int64` arrays per call.** Read
  from source: every strided kernel call constructs `np.asarray(shape)`
  and `np.asarray(strides)`. For a tensor whose metadata has not
  changed, those arrays are recomputable constants. Measured in
  isolation at ~1.8 µs; not separately attributed inside a step.
- **The odometer's per-element carry loop is a branch per element.**
  Read from source (`elementwise.cpp`, `reduction.cpp`,
  `storage.cpp`): the innermost work is one arithmetic op followed by a
  descending loop over dimensions with a conditional `break`. The
  measured contiguous-versus-strided elementwise gap (1.7–2.2× at 256²
  and 512²) is consistent with this, but the specific attribution to
  branch cost versus cache behavior is not isolated.
- **`NativeAdam` allocates a fresh broadcast scalar core per
  coefficient.** Read from source: `_stage_entry` calls
  `NativeTensorCore.full((), value)` for `beta1`, `1 - beta1`, `beta2`,
  `1 - beta2`, and both bias-correction terms — six one-element native
  allocations per parameter per step, each paying the full B3 fixed
  cost. Counted, not individually timed.

### 3.3 Unconfirmed hypotheses

These are **not** findings. They are stated so that a later milestone
either confirms or discards them explicitly.

- That auto-vectorization is currently being *prevented* rather than
  merely unhelpful. The MSVC Release build's actual vectorization report
  has not been read.
- That the elementwise odometer would benefit from stride-collapsing
  (merging adjacent dimensions whose strides are already contiguous).
  Plausible from source; unmeasured.
- That `to_numpy()` / `from_array()` conversion at the host boundary is
  material in any real workload. The examples convert only for
  reporting, outside training mathematics, so this is currently believed
  irrelevant — but it is unmeasured.
- That the graph traversal in `backward()` (topological sort, snapshot,
  rollback bookkeeping) is material. B8 suggests graph *construction*
  is not; traversal is separate and unmeasured.
- Any claim about a second thread, any SIMD width, or any BLAS. Nothing
  in §3.1 is threading- or SIMD-evidenced.

### 3.4 Instrumentation a later milestone would need

Where H0's observability could not settle a question, the *minimum*
additional instrumentation is recorded here rather than a result being
invented:

- **Per-callsite metadata counters.** To split B3 between `shape_info`,
  `_as_int_tuple`, `row_major_strides`, and `NativeTensorView.__init__`
  inside a *real* composed forward, a build-time-optional counter around
  each helper is needed. It must be off by default, must not exist in
  the normal call path, and must never become a public API.
- **A kernel-only timing seam.** To separate C++ execution from the
  Python approach in a production call, the harness would need to time
  the `ctypes` call with the destination preallocated. H0 does this
  ad hoc in a scratch script; a later milestone that claims a kernel
  speedup should do it inside the harness.
- **An allocation-size histogram.** B1's leverage depends on the *size
  distribution* of allocations in a real step, which H0 counted but did
  not bucket. `NativeStorage.__init__` already receives the size; a
  test-only tracker (like the harness's existing `live_storages`
  fixture) suffices — no production instrumentation.
- **A compiler vectorization report.** `/Qvec-report:2` (MSVC) or
  `-Rpass-analysis=loop-vectorize` (Clang) on the existing kernels,
  captured as documentation, before any SIMD claim is entertained.

None of this is production instrumentation. If a question can only be
answered by instrumenting a production numerical path, the answer is
that the question waits.

---

## 4. Representative workload families

The harness organizes every case into one of these families, and the
report groups by them:

| Family | What it isolates |
|---|---|
| `dispatch_overhead` | the size-independent per-call cost, and bare allocation |
| `elementwise` | the contiguous fast path vs the generic odometer |
| `reduction` | scatter-accumulate traversal, contiguous and strided |
| `matmul` | square, rectangular, and strided-fallback products |
| `materialization` | the Policy-B contiguous copy |
| `linear` | a real layer's forward and its backward |
| `convolution` | forward, input gradient, weight gradient, composed bias gradient |
| `normalization` | a training step through both normalization families |
| `stochastic` | a training step through Dropout with an explicit generator |
| `optimizer` | one optimizer step with no model work |
| `training_step` | one complete deterministic iteration, MLP and CNN |
| `state_operations` | the in-memory state surface, separate from any step |

**Checkpoint file I/O is deliberately excluded from every training-step
total and from the harness entirely.** `save_native_checkpoint` /
`load_native_checkpoint` are dominated by the filesystem and the NPZ
writer rather than by TensorForge, they belong to no training
iteration, and measuring them would make the harness write files. The
`state_operations` family measures the in-memory part TensorForge
actually owns — `state_dict()` and `load_state_dict()` — instead.

---

## 5. Benchmark shape selection

Every case declares exactly three configurations.

**`smoke`** — the tiniest shape that still exercises the code path.
Selection rules: every dimension ≥ 2 where a size-1 dimension could hide
a broadcast or layout mistake; unequal dimensions wherever a square
shape could mask an axis confusion; and the whole smoke suite must
complete in well under a second (measured: **0.56 s** for all 24
cases). Smoke mode is what tests and CI run.

**`full`** — the default. Selected so that the timed region is large
enough to be measurable above the ~100 ns timer resolution and the
~20 µs fixed dispatch cost, while the entire suite still finishes in
seconds. In practice this means 256×256 for elementwise, reductions and
materialization, 128³ for square matmul, 8×3×16×16 for convolution, and
64-sample batches for the layer and training-step cases.

**`profile`** — the focused profiler shape, selected by these rules:

1. **The timed region should dominate the fixed cost by ≥ 100×.** With
   the B3 figure of ~20 µs, that means a target of ≥ 2 ms per call, which
   is what makes a sampling profiler's output attributable. It is a
   target rather than a hard rule, and rule 4 records where and why it is
   not met.
2. **The working set stays under roughly 64 MB**, so the run is
   practical, does not swap, and stays within a plausible last-level
   cache-plus-DRAM regime rather than becoming a paging benchmark.
3. **Aspect ratios are preserved from the `full` shape**, so the profile
   run measures a bigger version of the same problem, not a different
   one.
4. **Comparability and representativeness outrank rule 1**, and four
   cases are exceptions on those grounds. Each is recorded here rather
   than quietly distorted:
   - `scalar_dispatch_overhead` keeps `(1, 1)` in all three
     configurations. Its cost is *definitionally* size-independent, so a
     larger shape would measure something else entirely (measured: ~30 µs
     at profile settings).
   - `conv2d_bias_gradient` keeps the convolution shape family the other
     three convolution cases use, because the point of the four cases is
     that their shares are directly comparable. Enlarging one of them
     alone would destroy that (measured: ~0.6 ms).
   - `sgd_step` keeps the model `adam_step` uses, for the same reason:
     the two are only informative *against each other*, and the number of
     parameters an inflated shape would need is not a shape any real
     model has (measured: ~0.5 ms).
   - `state_dict_load` keeps the model `state_dict_snapshot` uses, again
     so the pair stays comparable (measured: ~1.7 ms, close to the
     target).

   All four are still tens to hundreds of times the fixed dispatch cost,
   so they remain attributable; they simply do not clear 2 ms. Every
   other case does, by a wide margin.

`--profile CASE` runs exactly one case at that configuration with raised
warm-up and repetition counts (5 / 25 by default). It refuses to combine
with `--case`, `--workload`, or `--smoke`, so a profile run is always
unambiguous about what it measured.

---

## 6. Measurement methodology

### 6.1 Timer

`time.perf_counter_ns()` — monotonic, the highest resolution Python
offers, and integer nanoseconds so no float rounding enters the sample.
Its resolution is reported in the payload
(`environment.timer_resolution_ns`; 100 ns on the H0 machine, against a
measured back-to-back call delta of median 100 ns / max 500 ns). CPU
execution is synchronous, so no device synchronization is needed or
performed.

### 6.2 What is inside the timer

One measured repetition times **exactly one call** of the case's
operation. Outside the timer, always: input creation, module and
optimizer construction, state installation, generator reset, graph
construction for backward-only cases, and every cleanup.

Graph construction *is* inside the timer for the
`native_tensor_graph` and `training_step` layers, because it is part of
the call being characterized — a caller who builds a graph pays for it.

No sample is discarded, no timer overhead is subtracted, every sample is
retained in the payload, and every layer of one case runs under the same
setup discipline.

### 6.3 Warm-up and repeat policy

Warm-up repetitions run exactly like measured ones — same `prepare`,
same `run`, same `cleanup` — and are discarded before measuring. They
exist to settle first-call effects: the ctypes function-pointer
resolution, the import of a lazily imported stable module, page faults
on a first-touch buffer, and CPU frequency ramp.

Defaults: **3 warm-up / 11 measured** (full), **1 / 3** (smoke), **5 /
25** (profile). Heavier cases declare their own lower caps — 9 for
backward-only cases, 7 for training steps, 5 for the heavy matmul and
convolution cases — and the count actually used is reported per case, so
the payload never implies a repetition count it did not run.

### 6.4 Statistics

Every timed row reports `sample_count`, `median_s`, `min_s`, `max_s`,
`spread_s` (max − min), `relative_spread` (spread / median), the full
`samples_s` list, and its `units`.

The **median** is the primary statistic: on a machine with background
load the mean is dragged by outliers and the minimum reports a
best-case the caller will not reliably see. The spread is reported
beside it precisely so a reader can see when the median is not
trustworthy — as §3.0 requires on this machine.

### 6.5 Ratios

A case declares a `reference_layer`. Every other layer reports
`ratio_to_reference = this median / reference median`. A case whose
`reference_layer` is `null` publishes **no ratio anywhere** and says why
in `reference_detail`. A ratio is an observation, never a verdict.

### 6.6 JSON reporting

`--json` emits one JSON object, printed with `json.dumps` to stdout and
written to no file. Guaranteed fields:

- top level: `benchmark`, `version`, `schema_version`, `mode`,
  `environment`, `cases`;
- `environment`: `python_version`, `python_implementation`, `platform`,
  `machine`, `processor`, `numpy_version`, `numpy_build`,
  `tensorforge_version`, `native_backend`, `thread_environment`,
  `dtype`, `device`, `scope`, `timer`, `timer_resolution_ns`,
  `primary_statistic`, `configuration_variant`, `warmup`,
  `repetitions`, the per-family repetition caps, and `timestamp`;
- each case: `case`, `workload`, `section`, `operation`,
  `configuration_variant`, `configuration`, `shape`, `seed`,
  `reference_type`, `reference_layer`, `reference_detail`,
  `correctness_reference`, `correctness`, `warmup`, `sample_count`,
  `layers`, `notes`;
- each layer row: `implementation_layer`, `timing`,
  `ratio_to_reference`.

`schema_version` is the payload's own contract version. It moves when
the *shape* of the payload changes and never when a measured number
does.

### 6.7 Environment and build metadata

Read from real introspection APIs, never hand-maintained:

- `cpp.backend_info()` supplies the backend name, availability, tensor
  core/tensor object names, dtype, device, supported dtypes/devices,
  the `unsupported` tuple, and inventory counts.
- `numpy.show_config("dicts")` supplies the BLAS/LAPACK name and version
  and the SIMD extension set — the reason the NumPy reference column
  behaves the way §3.0 describes. If the call is unavailable or changes
  shape, the field is `null`; nothing is fabricated.
- The BLAS/threading environment variables listed in
  `THREAD_ENVIRONMENT_VARIABLES` are recorded **only when set**, so a
  reader can tell whether the NumPy column ran single- or
  multi-threaded. An unset variable is simply absent.

No absolute path, user name, or machine identifier is emitted.

### 6.8 Setup, execution, and cleanup separation

- Temporary native outputs are closed **explicitly** in each layer's
  `cleanup`. Nothing here relies on garbage collection.
- Backward cases rebuild the whole forward graph in the untimed
  `prepare` from cleared gradients, so no repetition inherits a retained
  graph or an accumulated gradient, and `retain_graph` is never used to
  skip a rebuild.
- Any case whose call advances persistent state rebuilds or resets that
  state outside the timer: a **fresh** model and optimizer per
  training-step repetition (parameters, Adam moments, step counters,
  BatchNorm running statistics), and a **generator reset** per Dropout
  repetition.
- The optimizer-only cases run their forward and backward **once**,
  outside the timer, and rely on the documented contract that native
  optimizers retain gradients until `zero_grad()`, so every repetition
  steps against identical gradient values. The correctness gate proves
  this with a separate probe model, so the gate never advances the state
  the timed layer uses.

### 6.9 Dropout measurement rules

Stochastic cases get their own rules, because a benchmark can silently
corrupt a random stream:

- the generator is **explicit and fixed** (`NativeGenerator(seed=…)`),
  never process-global;
- it is `reset()` in the untimed `prepare`, so every timed step draws
  the same mask from the same reserved call index and **benchmark setup
  can never shift the index a timed call consumes**;
- the gate proves **exactly one** call is consumed per successful step,
  that evaluation consumes none and returns the input object itself,
  and that the module registered the exact generator object it was
  given;
- **no equality is claimed against any unrelated random
  implementation.** The correctness reference is the native derivation
  itself — `NativeTensorCore.dropout_forward` at the same
  `(seed, call_index)` — plus structural mask properties.

---

## 7. Numerical contract

### 7.1 Correctness before timing, always

Every case runs its full correctness gate **before** the timing helper
is reached. A failed gate raises `AssertionError`, which propagates out
of `run_benchmark` and which the CLI turns into exit code 1 with a clean
stdout. **No timing is ever published for a case whose gate failed.**

### 7.2 Exact equality versus tolerance

The rule is decided by whether the operation accumulates:

| Operation class | Comparison |
|---|---|
| elementwise unary and binary | **exact** (`atol = 0`) |
| materialization, views, reshape | **exact** |
| Dropout mask against the Core derivation at the same key | **exact** |
| state snapshot / reload round trip | **exact** |
| reductions (`sum`, `mean`) | **tolerance** |
| matmul | **tolerance** |
| anything composed from the above (layers, losses, training steps) | **tolerance** |

Tolerances are absolute float64 agreement bounds, scaled by the
reference's magnitude where the values are not O(1). They are taken from
the existing parity suites. **No tolerance in this harness bounds a
duration, a ratio, or a throughput**, and the harness carries no float
constant that is not either a correctness tolerance or a module
argument.

One tolerance is deliberately looser and the reason is recorded rather
than hidden: `PARAMETER_ATOL = 1e-7` covers Adam's amplification of
round-off in a near-zero gradient by up to `lr / eps`, on a converged or
structurally dead parameter. The gradients themselves are still gated at
`GRADIENT_ATOL`.

### 7.3 Floating-point accumulation order

**The default is that Phase H preserves the existing accumulation order
exactly.** An optimization that changes only *which memory is touched
when* — loop tiling that preserves the per-output accumulation sequence,
metadata caching, allocation strategy, dispatch selection — must be
**bit-identical** to the path it replaces, and its milestone must prove
it by test.

That is not a theoretical constraint; §3.1/B2 measured that the
highest-leverage matmul change available (`i`–`k`–`j` blocking)
**already satisfies it**, at five matrix sizes and four block sizes
including non-dividing ones.

An optimization that genuinely *must* change accumulation order —
pairwise or blocked summation in a reduction, an FMA contraction, a
vector-lane split — is permitted only under all five of:

1. the milestone declares the order change explicitly, in its own
   section of this document and in the support matrix;
2. the new order is **deterministic**: the same inputs on the same build
   produce the same bits, run to run, process to process, and
   independently of allocation addresses, thread count, or any runtime
   dispatch decision;
3. the generic reference path (§8.3) is retained and the two are proved
   to agree to a stated tolerance;
4. every existing **bit-exact resume proof** is re-established at the
   new order — D11, E8, F6, G7, and the Phase C/D/E/F/G integration
   suites all assert exact equality across a checkpoint boundary, and
   they must continue to;
5. every **committed loss trajectory** in an example or a document that
   the change moves is re-derived and updated in the same milestone. The
   committed values (E8's 1.159638 → 0.000101, D11's ≈0.7713 → ≈0.0111,
   F6's 98.9 % reduction, G7's exact suffix) are project artifacts, not
   incidental output.

If a milestone cannot satisfy all five, it does not change the order.

### 7.4 Determinism

Phase H does not weaken any determinism guarantee:

- Same inputs, same build → same bits. No optimization may introduce a
  result that depends on allocation address, buffer alignment, wall
  time, thread count, thread scheduling, or a runtime CPU-feature
  probe **unless** that dispatch decision is itself deterministic *and*
  every reachable path produces bit-identical results (§7.3 rule 2).
- The generator contract is untouched: state is Python-managed, kernels
  are stateless and receive the whole key, and exactly one call is
  consumed per successful stochastic forward.
- Reproducibility remains exact **for the state actually captured**.
  Python's `random`, NumPy's global RNG, data-loader position, and
  scheduler state are still not captured, and full-program determinism
  is still not claimed.

---

## 8. Dispatch rules

### 8.1 Optimized contiguous dispatch

Phase H may add an optimized execution path only under all of:

1. **It is selected by inspecting real metadata, never guessed.** The
   existing rule is the model: `_binary_core_op` selects the flat kernel
   when `self.shape == other.shape and self.contiguous and
   other.contiguous`, including nonzero offsets and scalars.
2. **Selection is total and explicit.** Every input either matches an
   optimized path's declared precondition or falls to the generic path.
   There is no "probably contiguous" case.
3. **Selection is free of side effects.** Choosing a path allocates
   nothing, mutates nothing, and consumes no generator call.
4. **The optimized path is bit-identical to the generic path**, or the
   full §7.3 order-change procedure applies.
5. **A failed precondition is never an error.** It is a fallback.

### 8.2 Strided fallback

The generic odometer stays. It is the only path that can read an
arbitrary strided, offset, broadcast, or transposed view, and
broadcasting is *implemented* as zero strides through it — so removing
or bypassing it would remove broadcasting.

Where a kernel is contiguous-only by contract (convolution, pooling,
softmax, log-softmax, cross-entropy, dropout), **Policy B stays**: the
Core materializes a contiguous copy, records it as a temporary, and
closes it deterministically after the native call. Phase H may reduce
*how often* Policy B triggers; it may not make a contiguous-only kernel
read a strided operand.

### 8.3 Generic reference paths

Every optimized path added in Phase H must keep a **generic reference
path in the shipped code** that can compute the same result. It is not a
test fixture and not a comment: it is reachable code, exercised by
tests, and it is what an unusual layout falls back to.

This is the v1.13/v1.14 contiguous-fast-path contract, generalized: the
odometer was *kept* when the flat kernel arrived, and the two were
proved bit-for-bit equal. Every Phase-H milestone repeats that pattern
or does not ship.

---

## 9. Invariants Phase H preserves without weakening

None of the following may be relaxed, made conditional, or traded for
speed. Any milestone that cannot keep all of them does not ship.

**Stable line.** `tensorforge.Tensor`, `tensorforge.nn`, and
`tensorforge.optim` behavior is unchanged. Stable/native isolation
holds: the native line is reachable only through
`tensorforge.backends` / `tensorforge.experimental`, backend selection
stays explicit with no implicit dispatch, and importing the stable
package still works with no compiled library present.

**Capabilities.** float64/CPU only. `UNSUPPORTED == ("float32", "cuda",
"amp")`, `SUPPORTED_DTYPES == ("float64",)`,
`SUPPORTED_DEVICES == ("cpu",)`. No promotion, no casting, no silent
conversion.

**Ownership and lifetime.** Every `NativeStorage` rule holds: a handle
is destroyed exactly once, `close()` is idempotent and observable,
borrowing views never outlive their base, and `__del__` remains
defensive cleanup that correctness never depends on. An optimization
that reuses a buffer must not make any two live Python objects share
storage they do not already share.

**No NumPy fallback.** No native numerical path may compute through
NumPy. The existing tripwires stay.

**C ABI contract.** Exception containment at the boundary, thread-local
error state, self-validating guarded exports that write nothing to any
destination when they reject.

**Failure atomicity.** Every existing failure guarantee holds: a failed
operation releases everything it allocated and returns no partial
result; a failed optimizer staging commits nothing and leaves gradients
retryable; a failed state or checkpoint transaction restores all four
state families; live native storage returns to baseline.

**Autograd.** Parameter versioning, the stale-graph error raised before
any gradient is committed, graph lifetime and `retain_graph`,
graph-owned saved resources (BatchNorm eval snapshots, MaxPool2d
winners, cross-entropy probabilities, Dropout masks) released exactly
once with the graph history.

**State and checkpoints.** Buffer identity across loads, the universal
state-replacement lock order, whole-checkpoint transaction atomicity,
format version **2** with **(1, 2)** supported. **Phase H introduces no
version 3 and no new schema field.**

**Generators.** Determinism, the reserve/commit/abandon protocol,
exactly one call per successful stochastic forward, and exact
stochastic resume.

---

## 10. Allocation and scratch-workspace decision criteria

H0 does **not** add a memory pool or a scratch allocator, and the
evidence deliberately does not yet justify one.

What the evidence *does* justify is narrower and safer: **the eager
zero-fill is redundant for every kernel that fully overwrites its
destination** (§3.1/B1), and removing it for exactly those kernels
requires no pool, no lifetime change, and no new ownership rule. That is
the recommended H1 (§16).

A **memory pool** would additionally be justified only if, after H1 and
H3 land, all of:

1. a re-measured allocation-size histogram shows a small number of
   sizes recurring many times per step (the shape a pool exploits);
2. allocator time — not zero-fill time, which H1 removes — remains a
   measurable share of a re-measured training step;
3. a design exists in which pooled storage cannot violate §9: exactly-once
   destruction, no accidental aliasing between live objects, correct
   behavior when a caller holds a buffer past the step that produced it,
   and clean interaction with the checkpoint/state transactions;
4. the pool is observable and bounded — inspectable size, an explicit
   release, and no unbounded growth across a long run;
5. LeakSanitizer still returns live native storage exactly to baseline.

A **scratch workspace** (a reusable per-operation temporary) would be
justified only if a specific operation is shown to allocate a temporary
whose lifetime is provably confined to one call — the Adam staging
temporaries and the normalization intermediates are the candidates — and
only with a design that keeps failure atomicity: a failed operation must
leave the workspace in a state the next call can use, and must not leak
a partially written result to any caller.

Until those are met, the answer is no. H0 records the question, not an
answer.

---

## 11. SIMD decision criteria

**Rejected for now, on evidence.** §3.1/B2 shows the matmul gap is an
access-pattern gap: reordering the loops recovers ~3.3× with
bit-identical results, no intrinsics, and no build-system change. SIMD
would be attacking the second-order term first.

SIMD enters Phase-H scope only if **all** of:

1. H1–H7 have landed and been re-measured;
2. a compiler vectorization report (§3.4) shows the remaining hot loops
   are *not* already auto-vectorized;
3. the remaining gap on those loops is arithmetic-bound rather than
   memory- or dispatch-bound, shown by measurement;
4. a portable path exists — compiler-directed vectorization or a guarded
   intrinsics path **with the scalar reference retained** (§8.3) — that
   builds warning-free on MSVC *and* Clang;
5. the numerical consequence is settled under §7.3. A vector-lane split
   of a reduction changes accumulation order and therefore needs the
   full five-condition procedure; a vectorized *elementwise* loop does
   not, and must be bit-identical;
6. it introduces **no required dependency** and **no mandatory
   `-march`/`/arch` flag** for the default build. Any wider ISA must be
   opt-in with a scalar default, because the default build must remain
   reproducible on the machines the project already validates on.

---

## 12. Threading decision criteria

**Rejected for now, on evidence.** Nothing in §3.1 is threading-
evidenced, and the three largest measured costs (eager zero-fill, matmul
access pattern, Python-side per-call metadata) are all single-threaded
problems that threading would not fix — the last of them is *behind* the
GIL.

Threading enters Phase-H scope only if **all** of:

1. the single-threaded work above is done and re-measured;
2. a workload exists whose remaining cost is a genuinely parallel
   kernel large enough that thread-pool overhead is amortized;
3. determinism survives §7.4 — a parallel reduction that sums partial
   results in completion order is **not acceptable**; a fixed
   partitioning with a deterministic combine order may be;
4. it does not conflict with the process-wide state-replacement lock
   order or the generator reservation protocol (§9);
5. thread count is explicit and defaults to 1, so the default build's
   timings stay comparable and no environment variable silently changes
   results;
6. it is validated under ThreadSanitizer, in addition to the existing
   ASan/UBSan matrix;
7. it introduces no required dependency. **OpenMP is specifically
   excluded**: it is a toolchain-dependent, environment-variable-driven
   dependency that would make the default build's behavior depend on
   `OMP_NUM_THREADS`, which contradicts criterion 5.

---

## 13. Optional BLAS decision criteria

**Rejected for now, on evidence, and likely permanently as a default.**

The tempting argument is that NumPy's OpenBLAS beats the native matmul
and linking a BLAS would close the gap. Three reasons that is the wrong
move here:

1. **It answers the wrong question.** §3.1/B2 shows the production
   kernel is 3.3× off its *own* achievable single-threaded scalar
   performance. Linking BLAS would hide that rather than fix it, and
   would leave every non-matmul finding untouched.
2. **It contradicts the project's premise.** TensorForge is a
   from-scratch systems project; "we call someone else's GEMM" is not
   the thing being demonstrated. `cpp/src/matmul.cpp` already says "no
   BLAS" as a deliberate choice.
3. **It is a dependency.** The project's stated tech stack is Python +
   NumPy + pytest, and the C++ backend is buildable with nothing but a
   C++17 compiler. A BLAS would break that on every platform the
   project validates on.

If BLAS is ever revisited it must be **strictly optional**: off by
default, behind an explicit CMake option, with the hand-written kernel
retained as the default and the reference path (§8.3), with the
numerical difference documented under §7.3 (BLAS *will* differ from the
scalar path — measured at ≈3.6e-15 for 64×64), and with every bit-exact
resume proof still running against the default build. It would also
never be the answer to a milestone; it would be an appendix to one.

---

## 14. Cross-platform requirements

Every Phase-H milestone must, before it ships:

- build **warning-free** with MSVC on Windows in both Release and Debug
  (the project's primary development platform), and with Clang/GCC on
  Linux under `-Wall -Wextra`;
- use no compiler-specific pragma, intrinsic, or builtin without a
  portable fallback compiled by default;
- add no new required build option. `TF_SANITIZE` and `TF_BUILD_TESTS`
  remain the only options, unless a milestone explicitly adds an
  **opt-in, default-off** one under §11 or §13;
- keep the C ABI narrow: hidden default visibility, `TF_EXPORT` only on
  functions Python actually declares;
- keep the full native CTest suite green in **both** configurations
  (currently 11 tests);
- assume nothing about alignment, endianness, or `long` width beyond
  what C++17 and the existing `int64_t`/`double` contract guarantee.

---

## 15. Sanitizer requirements

The Phase-F/G validation matrix is the standard, and Phase H does not
lower it. Every milestone that touches C++ or changes allocation
behavior must additionally pass:

- a fresh Clang build with `-DTF_SANITIZE=address,undefined`, with
  instrumentation **proved present** (`nm -D` showing `__asan*` /
  `__ubsan*` dynamic symbols beside the exported `tf_*` symbols, and the
  library refusing to load without the sanitizer runtime);
- the full native CTest suite under that build;
- the native Python suites under that build, with **zero** ASan and
  **zero** UBSan diagnostics;
- a LeakSanitizer lifecycle in which native live storage returns
  **exactly** to baseline, with **no suppression file added**.

An allocation-strategy change (H1) raises the bar rather than lowering
it: removing an eager zero-fill means a buffer can legitimately contain
garbage before its kernel writes it, and **MemorySanitizer-class
mistakes become possible for the first time in this project**. H1 must
therefore prove, per kernel, that every destination byte is written
before it is read — by C++ test, by ASan/UBSan, and by the existing
bit-exact resume proofs, which would diverge immediately on an
uninitialized read.

---

## 16. Milestone ladder

**H1–H8 are proposals, not commitments.** They are ordered by measured
leverage from §3, and each is explicitly conditional: a milestone whose
premise the preceding measurement does not confirm is **narrowed,
reordered, or dropped**, and the reason is recorded here. Every
milestone re-runs the H0 harness before and after and reports both.

| # | Milestone | Evidence | Status |
|---|---|---|---|
| **H0** | **Architecture, profiling, and baseline** | — | **complete** |
| H1 | Output allocation contract: uninitialized allocation for fully-written destinations | B1 (measured, ~74 % of a 2 MB elementwise op) | proposed |
| H2 | Matmul memory access: loop order and cache blocking in the production kernel | B2 (measured, 3.3×, **bit-identical**) | proposed |
| H3 | Per-call dispatch cost: cached layout metadata at the Core boundary | B3 (measured, ~20 µs fixed, ~10 % of it ctypes) | proposed |
| H4 | Optimizer step cost: fewer native calls and allocations per parameter | B4 (measured, 27 allocations/parameter, 83 % of a step) | proposed |
| H5 | Reduction execution: a contiguous fast path for the accumulate kernels | B5, B7 (measured) | proposed |
| H6 | Composed-module cost: normalization and the composed convolution bias gradient | B6, B7 (measured) | conditional on H1/H3/H5 |
| H7 | Elementwise and materialization traversal | B9 + §3.2 stride-collapsing hypothesis | conditional |
| H8 | SIMD, threading, or optional BLAS — **only** under §11/§12/§13 | none yet | conditional, presumed rejected |
| H9 | Re-measurement, hardening, and the full sanitizer matrix | — | planned |
| H10 | Phase closure | — | planned |

### H0 — Architecture, profiling, and baseline **(complete)**

This document, the unified harness, its contract tests, and
documentation reconciliation. No production numerical change.

### H1 — Output allocation contract *(recommended next)*

Give the native allocation path an explicit **initialized / uninitialized**
choice, and use the uninitialized one only for destinations a kernel
provably overwrites in full.

- Adds one C ABI entry point beside `tf_storage_create` (not a
  replacement — the zero-initializing one stays and stays the default).
- The Core-level allocation helpers gain an explicit opt-in. There is no
  global switch and no heuristic.
- **The audit in §3.1/B1 is the contract**, enumerated per kernel in
  H1's own section of this document, with `tf_core_sum`,
  `tf_core_narrow_backward`, and the scatter-add backwards explicitly
  keeping the zeroed destination.
- Must be bit-identical everywhere (§7.3). It changes no arithmetic at
  all, so this is a hard requirement, not a tolerance.
- Raises the sanitizer bar per §15.
- Preserves every §9 invariant, in particular failure atomicity: an
  uninitialized buffer must never escape to a caller on a failure path.

### H2 — Matmul memory access

Bring `tf_core_matmul` up to the access pattern `tf_matmul_tiled`
already demonstrates, **preserving the per-output accumulation order
exactly** (measured bit-identical in §3.1/B2).

- The strided-operand capability is not lost: the kernel must still read
  both operands through their own strides and offsets, so a transposed
  or narrowed view still multiplies without materializing.
- The generic path is retained per §8.3.
- Block size is a fixed, documented constant, not a runtime probe, so
  the result cannot depend on a hardware query.
- Bit-exactness against the current kernel is proved by test at multiple
  sizes including non-dividing ones, which is what makes every committed
  loss trajectory and every exact-resume proof survive untouched.

### H3 — Per-call dispatch cost

Reduce the ~20 µs fixed cost, of which the ctypes boundary is ~2 µs.

- Cache the normalized shape/stride/layout arrays on the tensor core,
  since they are a pure function of metadata that view operations
  already know when they change.
- Reduce redundant `_as_int_tuple` / `_as_shape` calls in the hot path.
- **Pure Python; no C++, no ABI change.** Bit-identical by construction.
- Must not weaken any validation. Every rejection the current path
  performs still happens, with the same message.

### H4 — Optimizer step cost

Reduce the fixed 27 native calls and allocations per parameter per Adam
step — including the six one-element broadcast-scalar allocations §3.2
identifies.

- Must preserve the two-phase mutation-atomic contract exactly: every
  validation and every staged computation completes before the first
  parameter mutates; a staging failure commits nothing and leaves
  gradients retryable.
- Must preserve one `copy_value_` and exactly one version increment per
  updated parameter, gradient retention until `zero_grad()`, and the
  moment-buffer replacement ordering.
- Bit-identical unless §7.3's procedure is invoked, which for an
  optimizer would immediately break the exact-resume proofs, so in
  practice: bit-identical.

### H5 — Reduction execution

Give the accumulate kernels the contiguous fast path the elementwise
family got at v1.14.

- **Accumulation order is preserved**: a contiguous fast path that walks
  the same logical order is bit-identical, and that is the requirement.
  A pairwise or blocked summation is a §7.3 order change and is **not**
  in H5's scope.
- The odometer is retained per §8.2/§8.3 — it is what implements strided
  reads and, dually, broadcasting.

### H6 — Composed-module cost *(conditional)*

Reduce the allocation and dispatch count in the normalization modules
and the composed convolution bias gradient.

**Conditional, and likely to be narrowed:** if H1, H3, and H5 land, most
of B6's cost is removed by them, since a normalization forward is 11–27
allocations of exactly the kind H1 makes cheaper and exactly the
operations H3 and H5 make cheaper. H6 proceeds only if a re-measured
normalization step still shows a material composed-module cost *after*
those. **It must not introduce a normalization kernel**: F2–F4's
achievement is that both families are compositions with no C++ at all,
and trading that for speed would be a semantic regression, not an
optimization.

### H7 — Elementwise and materialization traversal *(conditional)*

Stride collapsing and materialization cost, if §3.2's hypothesis
survives measurement after H1 and H3. Presently the weakest-evidenced
numbered milestone, and the most likely to be dropped.

### H8 — SIMD, threading, or optional BLAS *(conditional, presumed rejected)*

Entered only under §11, §12, or §13 respectively. On current evidence
**none of the three qualifies**, and the honest expectation is that H8
ships as a *documented rejection with measurements* rather than as code.
That is a legitimate outcome for this milestone.

### H9 — Re-measurement, hardening, and sanitizer validation

The full H0 harness re-run and compared against the H0 baseline;
cross-cutting integration tests; adversarial failure-atomicity and
ownership matrices over whatever H1–H8 changed; Windows Release and
Debug builds with the full CTest suite; the Clang ASan/UBSan matrix with
instrumentation proved; LeakSanitizer returning live storage exactly to
baseline.

### H10 — Phase closure

Documentation reconciliation across every status surface, durable
semantic closure guardrails, and the support matrix updated. No new
numerical capability.

---

## 17. Phase-H non-goals

Explicitly **not** in Phase H, at any milestone:

- CUDA, any GPU backend, Tensor Cores;
- `float32`, `float16`, `bfloat16`, casting, dtype promotion, AMP;
- integer tensors;
- pybind11, or any change to the plain C-ABI + ctypes boundary;
- C++-side autograd — the graph stays Python-managed;
- implicit dispatch or automatic backend selection;
- Transformers, attention, embeddings;
- data loaders, distributed training;
- a memory pool or scratch allocator (§10 records the criteria; H0 adds
  neither);
- SIMD, threading, OpenMP, BLAS (§11–§13);
- any required dependency — the stack stays Python + NumPy + pytest with
  an optional C++17 compiler;
- checkpoint format version 3, or any new schema field;
- **any CI timing threshold, committed duration, or performance
  assertion.** No test in this repository asserts a wall-clock number,
  and Phase H does not add the first one.

---

## 18. Phase-H closure requirements

Phase H is complete only when all of:

1. every shipped milestone's optimization has a retained generic
   reference path (§8.3) exercised by tests;
2. the numerical contract in §7 held throughout, with any order change
   documented under §7.3 and every bit-exact resume proof still passing;
3. the full pytest suite passes with **zero skips**, as the post-Phase-G
   baseline does;
4. Windows Release and Debug builds pass with zero project compiler,
   linker, and CMake warnings, and the full native CTest suite passes in
   each;
5. the Clang ASan/UBSan matrix passes with instrumentation proved and
   zero diagnostics, and LeakSanitizer returns native live storage
   exactly to baseline with no suppression file;
6. the H0 harness runs clean in every mode, every gate passes, and the
   H0-versus-closure comparison is reported honestly — including any
   case that did **not** improve;
7. no supported dtype, device, capability registry value, export, or
   checkpoint version changed;
8. the stable line and stable/native isolation are provably unchanged;
9. documentation agrees across every status surface, with no committed
   machine-specific speed number and no marketing claim anywhere;
10. any milestone that was narrowed, reordered, or dropped is recorded
    here with the evidence that made that the right call.

---

## 19. Daedalus: adopt, adapt, reject

The reference project is
[JohnsonKayati/daedalus-ml](https://github.com/JohnsonKayati/daedalus-ml),
a PyTorch-style C++/CUDA framework. It is a reference for *ideas*, not a
source of code, and TensorForge is not a copy. Each relevant idea gets an
explicit decision.

| Daedalus idea | Decision | Reasoning |
|---|---|---|
| Per-device operation directory (`src/ops/cpu/*.cpp`: `binary_ops`, `matmul_ops`, `reduce_ops`, `conv2d_ops`, …) | **Adapt** | TensorForge already organizes `cpp/src/` by concern (`elementwise`, `matmul`, `reduction`, `conv2d`, `pooling`, `classification`, `random`, `storage`, `error`) — the same idea without the device axis, which TensorForge does not have. Keep the existing layout; do not add a `cpu/` level for a project with one device. |
| Naive CPU matmul as the reference (`cpu_matmul<T>`, triple loop, no blocking/SIMD/OpenMP/BLAS) | **Adopt as a reference, reject as the shipped path** | Daedalus keeps its CPU matmul naive and puts its effort into cuBLAS/CUDA. TensorForge has no GPU path to fall back on, and §3.1/B2 measured 3.3× available with bit-identical results. Keep a naive kernel as the retained reference (§8.3); do not keep it as the only path. |
| `WITH_AVX2` CMake option (`/arch:AVX2` on MSVC, `-mavx2` elsewhere), default OFF | **Reject for now; adopt the *shape* if §11 is ever met** | The default-off, single-option, no-intrinsics-in-source form is exactly right and is the shape TensorForge would use. But adopting it now would be optimizing the second-order term before the first (§11), and TensorForge's evidence says the win is in loop order. Revisit only under §11. |
| Fused operations (`fusion_ops.cpp`) | **Reject for Phase H** | Fusion is a semantic change: it creates new operations with new backward contracts, new saved state, and new versioning obligations. TensorForge's fused kernels (softmax, log-softmax, cross-entropy, dropout) each got their own milestone with its own contract for exactly that reason. Fusion is a future *capability* phase, not a performance milestone. |
| Memory pooling (documented; "CUDA memory pooling with block splitting, coalescing, stream-aware behavior" listed as future work) | **Reject for now; criteria recorded** | Daedalus's pooling motivation is CUDA allocation latency, which TensorForge does not have. §3.1/B1 shows TensorForge's allocation cost is dominated by the *eager zero-fill*, not by the allocator — a much cheaper fix with no ownership risk. §10 records what would have to be true first. |
| Scratch workspaces | **Reject for now; criteria recorded** | Daedalus shows no scratch-workspace implementation to adapt, and TensorForge's evidence does not yet identify one. §10 records the criteria. |
| Benchmark scripts with `--profile` and `--shape` flags that isolate one large operation with increased repetitions | **Adopt** | Directly adopted as `--profile CASE`, with TensorForge's own §5 shape-selection rules and its own correctness-before-timing rule layered on. This is the single best idea taken from Daedalus. |
| Profiling with Nsight Systems / Nsight Compute, kernel-launch-gap and occupancy analysis | **Reject as tooling; adopt the discipline** | Both tools are CUDA-only. The transferable part — *isolate one operation, raise repetitions, attribute cost to a named section, and record the methodology* — is adopted in §5 and §6. |
| Committed benchmark results tracked in `profiles/RESULTS.md`, with epoch-time comparisons | **Reject** | TensorForge deliberately commits **no** machine-specific benchmark result file, in any phase. Every prior characterization benchmark (D12, E9, F7, G8) writes no result file and asserts no speed, and H0 keeps that. A committed number becomes a promise the project cannot keep across machines. |
| Environment-variable kernel selectors (`DAEDALUS_CUDA_CONV2D_GRAD_INPUT`, …) | **Reject** | An environment variable that changes which kernel runs makes results depend on ambient process state, which contradicts §7.4 determinism and would silently invalidate the exact-resume proofs. TensorForge dispatch is metadata-driven and explicit (§8.1). |
| pybind11 bindings (`bindings/`, `BUILD_PYTHON_BINDINGS`) | **Reject** | TensorForge's plain C-ABI + ctypes boundary is a deliberate architectural choice: no build-time Python dependency, no ABI coupling to a Python version, and a narrow exported surface. Already a standing project decision; Phase H does not revisit it. |
| GoogleTest for C++ tests | **Reject** | TensorForge's C++ CTests are dependency-free binaries that compile the kernel source directly. That keeps `uv run python cpp/build.py` working with nothing but a compiler. Adding a test dependency to gain assertion sugar is a bad trade here. |
| CUDA architecture (handwritten kernels, cuBLAS dispatch, NCCL, AMP) | **Reject for Phase H; noted as a future dependency** | This is the strongest argument in §1.2, read in reverse: Daedalus's CPU path stayed naive *because* its GPU path carried the performance story. TensorForge has no GPU path, so its CPU runtime has to carry it — and a future CUDA phase will inherit whatever runtime Phase H leaves behind. |

---

## 20. Relationship to the other design contracts

Phase H sits **under** every prior contract and overrides none:

- [native_tensor_wrapper_design.md](native_tensor_wrapper_design.md) —
  the wrapper's ownership and conversion contract. §3.1/B8 measured the
  wrapper is not a bottleneck; Phase H does not touch it.
- [native_contiguous_fast_path_design.md](native_contiguous_fast_path_design.md)
  — the original optimized-path/generic-path pattern, generalized by §8.
- [native_broadcasting_design.md](native_broadcasting_design.md) and
  [native_reductions_design.md](native_reductions_design.md) — zero-stride
  reads and writes. §8.2 records why the odometer cannot be removed.
- [native_autograd_design.md](native_autograd_design.md) — graph
  lifetime, versioning, saved resources. §9 preserves all of it.
- [native_cnn_design.md](native_cnn_design.md) — Policy B, which §8.2
  keeps.
- [native_classification_design.md](native_classification_design.md) —
  the fused stable-math contract, which §17 keeps out of fusion scope.
- [native_normalization_design.md](native_normalization_design.md) —
  the no-kernel composition rule, which H6 must not break.
- [native_rng_dropout_design.md](native_rng_dropout_design.md) — the
  generator protocol and checkpoint version 2, which §9 preserves.
- [native_support_matrix.md](native_support_matrix.md) — the canonical
  capability status, unchanged by H0.
