# Native CPU Performance and Runtime Efficiency — Phase H design

**Status: milestones H0, H1, H2, H3, H4, and H5 complete. No supported
capability has changed. H2 is the first milestone to change a numerical
kernel's execution — its memory-access pattern, not its arithmetic; H3
and H4 are Python-only, and H4 is the first whose subject is a
training-stack component rather than the tensor runtime.**

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
bit-exact stochastic resume. What it has never been is *fast*. At H0
every kernel in `cpp/src/` was a deliberately plain reference loop, and
the matmul comment said so in the first line of the file:

> All are deliberately unoptimized reference loops (no blocking beyond
> the explicit tile, no SIMD, no BLAS).

*(That header is the pre-H2 text. **H2 replaced it**: `matmul.cpp` now
ships an optimized production path beside the reference loop, and the
reference loop is retained under §8.3 rather than removed. **H5 and H6
did the same for two more files** — `elementwise.cpp`'s
strided-to-contiguous gather and `reduction.cpp`'s sum — each behind its
own hidden metadata predicate, each with the pre-milestone traversal
retained and reachable. So three of the nine translation units now ship
two paths; the rest are still plain reference loops. Everything in §3
below is still the H0 baseline — the before-and-after comparisons live in
§16.2.8, §16.5.6, and §16.6.10 rather than being edited back into §3.)*

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

**This finding is the one H2 acted on, and H2's outcome differs from the
sketch above in one respect worth recording**: the win is the `i`–`k`–`j`
*order*, not the tiles. Measured against 22 blocked variants, an
unblocked full-row sweep was faster at every non-trivial size, so
**cache blocking was rejected** and the production kernel carries none
(§16.2.7). The 3.3× above is the tiled kernel's figure; the shipped row
sweep measured 4.1–4.7× at the same profile shape (§16.2.8).

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

**This is the finding H6 acted on, and the shipped result differs from
this sketch in two ways worth recording.** First, the cause was
decomposed rather than assumed: the traversal turned out to be **95 %** of
a reduction, with the whole Python wrapper at ~5 µs, so H6's target was
unambiguous (§16.6.2). Second, the *rank* effect above was not
anticipated: the odometer's carry loop runs up to `ndim` iterations per
element, so 3-D and 4-D reductions improved **8.6×–10.9×** against
2.6×–6.4× for 2-D ones (§16.6.10) — which is also why the bias-gradient
composition improved 1.46× end to end while every training step stayed
neutral.

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
  `tensorforge_version`, `native_backend`, `native_build`,
  `thread_environment`, `dtype`, `device`, `scope`, `timer`,
  `timer_resolution_ns`, `primary_statistic`, `configuration_variant`,
  `warmup`, `repetitions`, the per-family repetition caps, and
  `timestamp`;
- each case: `case`, `workload`, `section`, `operation`,
  `configuration_variant`, `configuration`, `shape`, `seed`,
  `reference_type`, `reference_layer`, `reference_detail`,
  `correctness_reference`, `correctness`, `warmup`, `sample_count`,
  `layers`, `allocation_comparison`, `dispatch_comparison`, `notes`;
- each layer row: `implementation_layer`, `timing`,
  `ratio_to_reference`.

`schema_version` is the payload's own contract version. It moves when
the *shape* of the payload changes and never when a measured number
does. It is **2**: H0 and H1 published version 1, and H2 added three
additive fields — the `tensor_core_generic` layer, the per-case
`dispatch_comparison` block, and `environment.native_build`. No existing
field changed meaning.

`environment.native_build` is deliberately short and deliberately
incomplete. It reports what can honestly be read from the compiled image
itself — its object format, its size in bytes, and whether ASan/UBSan
instrumentation is present, which is genuinely visible in the symbol
table. The **compiler identity, its version, and its optimization flags
are reported as `null`**, because neither `cpp/build.py` nor the CMake
project records them anywhere, and `platform.python_compiler()` describes
the interpreter's toolchain rather than this library's. Printing it here
would be a fabrication, so it is not printed. No path is emitted.

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
highest-leverage matmul change available (`i`–`k`–`j` reordering)
**already satisfies it**, and **H2 shipped under exactly this rule** — it
changed only which memory is touched when. H2 therefore did **not**
invoke the five-condition escape hatch below, and no milestone in this
project has yet.

**"Bit-identical" above needs one qualification, and H2 is where it first
bites.** The rule this section states is about the *sequence of
arithmetic operations*, and H2 preserves that exactly. What a preserved
sequence does **not** guarantee is the payload of a NaN result: when two
NaN operands meet, which one propagates is decided by the compiler's
choice of instruction operand order — a decision C++ cannot express, and
one that follows from the loop structure rather than from the arithmetic.
So the requirement this section imposes is stated in four parts, and
§16.2.3 works them through for H2 with the measurements behind them:

1. **the accumulation sequence is preserved exactly** (the rule above);
2. **every non-NaN result is bit-identical**, including signed zeros,
   infinities, denormals, and the largest finite magnitudes;
3. **NaN-class equivalence** — NaNs appear in exactly the same positions
   on both paths, and are always quiet;
4. **NaN payload bits are outside TensorForge's numerical contract** and
   may differ.

Parts 1–3 are hard requirements that every milestone must prove by test.
Part 4 is a licence, not a permission slip for a real order change: it
covers the bits of a value that is *already* NaN, which means the
computation has already left the supported numerical domain. Every
committed loss trajectory and every bit-exact resume proof in this
project runs on finite data, so part 2 covers all of them and none of
them is affected.

**Part 4 must be re-established per milestone, not assumed from H2.** It
is a licence each milestone may or may not need, and two later milestones
show both answers. **H5 did not need it at all**: a value transfer
performs no arithmetic, so it has no operand roles for a compiler to
choose between, and the copy traversals are bit-identical *by
construction* including every NaN payload and both signs of signaling NaN
(§16.5.3). **H6 did need it, but on its own measured terms and with a
narrower scope than H2's**: a reduction's two paths agree bit for bit
whenever at most one NaN enters an accumulation — which is every case that
occurs in practice — and may differ only when two or more NaNs are
accumulated into the same destination cell (§16.6.8). H6 also measured
*why* parity was unavailable rather than asserting it, testing four
spellings of the optimized accumulation including one structurally
identical to the reference; all four diverged from the reference
identically. A milestone invoking part 4 must do the same: state the exact
circumstance under which payloads can differ, and show that parity was
not available cheaply.

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
4. **The optimized path agrees with the generic path under §7.3's
   numerical contract** — identical accumulation sequence, bit identity on
   every non-NaN result, and NaN-class equivalence — or the full §7.3
   order-change procedure applies. Whether the milestone additionally
   needs §7.3's part-4 NaN-payload licence is **its own question to
   measure**, not something inherited: H2 needed it, H5 did not need it at
   all, and H6 needed it only for accumulations containing two or more
   NaNs.
5. **A failed precondition is never an error.** It is a fallback.

Three milestones have now shipped under this rule with the *same*
structure — one hidden metadata predicate, inside the existing export, no
new symbol, the pre-milestone traversal retained: H2 (matmul loop order),
H5 (the copy gather's traversal), and H6 (the sum reduction's traversal).
That is the shape any further execution-path work should take.

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

**Rejected for now, on evidence.** §3.1/B2 showed the matmul gap is an
access-pattern gap, and **H2 confirmed it by shipping the fix**:
reordering the loops recovered 4.1–4.7× at the profile shape with
bit-identical results, no intrinsics, and no build-system change. SIMD
would have been attacking the second-order term first. It still would:
H2's inner loop is a long unit-stride `acc[j] += scalar * b[j]`, which is
exactly the shape a compiler auto-vectorizes without help, so criterion 2
below now has a concrete target rather than a hypothetical one.

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

1. **It answers the wrong question.** §3.1/B2 showed the production
   kernel was 3.3× off its *own* achievable single-threaded scalar
   performance, and H2 recovered 4.1–4.7× of that by changing the loop
   order alone (§16.2.8). Linking BLAS would have hidden that rather than
   fixed it, and would leave every non-matmul finding untouched.
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
  (currently 13 tests: the 11 H0 inherited, plus H1's storage-allocation
  contract test and H2's matmul path/dispatch test);
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
therefore prove, per kernel, that every destination element is written
before it is read.

That proof is **separate from the sanitizers**, which do not answer the
question: ASan and UBSan cover memory-boundary safety and undefined
behavior, and neither reports a read of an uninitialized *value*.
Initialization completeness is established instead by the deterministic
poison tests of §16.1.3 — injected by test infrastructure around the
allocator, never by the runtime — supported by the existing bit-exact
resume proofs, which would diverge immediately if a destination's initial
contents ever reached a result. The sanitizer matrix above is still
required in full; it simply proves different properties.

---

## 16. Milestone ladder

**H1–H11 are proposals, not commitments.** They are ordered by measured
leverage from §3, and each is explicitly conditional: a milestone whose
premise the preceding measurement does not confirm is **narrowed,
reordered, or dropped**, and the reason is recorded here. Every
milestone re-runs the H0 harness before and after and reports both.

| # | Milestone | Evidence | Status |
|---|---|---|---|
| **H0** | **Architecture, profiling, and baseline** | — | **complete** |
| **H1** | **Output allocation contract: uninitialized allocation for fully-written destinations** | B1 (measured, ~74 % of a 2 MB elementwise op) | **complete** — see 16.1 |
| **H2** | **Matmul memory access: loop order in the production kernel** | B2 (measured, 3.3×, accumulation order preserved) | **complete** — see 16.2. Cache blocking was measured and **rejected**; the generic/optimized agreement is §16.2.3's four-part contract, not unconditional bit identity |
| **H3** | **Per-call dispatch cost: one normalization boundary and per-view layout metadata** | B3 (measured, ~20 µs fixed, ~10 % of it ctypes) | **complete** — see 16.3 |
| **H4** | **Optimizer step cost: fewer native calls and allocations per parameter** | B4 (measured, 27 allocations/parameter, 83 % of a step) | **complete** — see 16.4. Bit-identical; the scalar-materialization variant was measured and **rejected** |
| **H5** | **Copy and mutation transfer: the value-transfer primitive, and the traversal under it** | B1, B4 (measured; ten call sites of one helper, all pure value transfers) | **complete** — see 16.5. The ladder was **reordered here**: reduction execution, drafted as H5, moved to H6 (§16.5.0). No exported symbol added; the signed-zero and signaling-NaN contract is stated in §16.5.3 |
| **H6** | **Reduction execution: a metadata-chosen block traversal for the accumulate kernel** | B5, B7 (measured; the traversal was **95 %** of a reduction) | **complete** — see 16.6. No exported symbol added; 2.6×–6.4× on 2-D and **8.6×–10.9×** on 3-D/4-D reductions, **every training step neutral**; the reduction-specific NaN rule is §16.6.8 and is *not* H2's |
| H7 | Composed-module cost: normalization and the composed convolution bias gradient | B6, B7 (measured) | conditional — **narrowed further by H6**, which measured normalization mostly neutral (§16.6.11) |
| H8 | Elementwise and materialization traversal | B9 + §3.2 stride-collapsing hypothesis | conditional, **narrowed by H5**, which already gave materialization its flat traversal |
| H9 | SIMD, threading, or optional BLAS — **only** under §11/§12/§13 | none yet | conditional, presumed rejected |
| H10 | Re-measurement, hardening, and the full sanitizer matrix | — | planned |
| H11 | Phase closure | — | planned |

### H0 — Architecture, profiling, and baseline **(complete)**

This document, the unified harness, its contract tests, and
documentation reconciliation. No production numerical change.

### H1 — Output allocation contract **(complete)**

The shipped contract, the per-kernel audit, the poison
methodology, the parity proof, and the measured results are
section 16.1 below.

### H2 — Matmul memory access **(complete)**

The two shipped compute paths, the metadata dispatch predicate, the
rejected blocking experiment, and the four-part numerical contract are
section 16.2 below.

### H3 — Per-call dispatch cost **(complete)**

The single normalization boundary, the private validated view
constructor, the per-view layout-array memoization, the immutability
proof, and the measured results are section 16.3 below.

### H4 — Optimizer step cost **(complete)**

The per-step scalar holder, the exact reciprocal substitution, the
eager temporary release, the retained pre-H4 reference, the rejected
alternatives, and the measured results are section 16.4 below.

### H5 — Copy and mutation transfer **(complete)**

The value-transfer primitive, the ten-call-site inventory, the
signed-zero and signaling-NaN findings, the metadata-driven second
traversal inside the unchanged export, the alias and overlap matrix,
and the measured results are section 16.5 below. **The ladder was
reordered here** (§16.5.0): reduction execution, drafted as H5, is
now H6.

### H6 — Reduction execution **(complete)**

The characterized pre-H6 kernel, the bottleneck attribution (the
traversal was 95 % of a reduction), the metadata predicate and its
`outer × mid × inner` factorization, the retained odometer and its exact
fallback conditions, the arithmetic-order proof, the signed-zero
contract, the **measured** reduction-specific NaN rule (which is
deliberately not H2's), the confirmed H1 Outcome A, the rejected
candidates, the published fallback regression, and the measured results
are section 16.6 below.

---

## 16.1 H1 — the output-allocation contract, as shipped

**Status: complete.** H1 removed the redundant zero-initialization from
output storage that a kernel provably overwrites in full. It is an
allocation change and nothing else: no arithmetic, no traversal, no loop
order, no kernel, and no capability moved.

### 16.1.1 What was added

**Exactly one new C ABI symbol, and no other:**

| Symbol | Kind | Purpose |
|---|---|---|
| `tf_storage_create_uninitialized(int64_t size)` | production | The uninitialized sibling of `tf_storage_create`. Identical size validation, zero/negative rejection, fault injection, allocation-failure handling, error state, handle shape, ownership, destruction, and live-storage accounting; the buffer's initial contents are the **only** difference. |

That brings the library's exported surface from the pre-H1 baseline of
**51** `tf_*` symbols to **52**, and
`tests/test_native_storage_allocation.py` asserts that count by parsing
the built image's own export table rather than by consulting a list this
repository maintains.

There is **no poison-control symbol, and no runtime initialization
control of any kind**: no exported hook, no thread-local flag, no
environment variable, no global mode, and no per-call policy argument.
An earlier draft of this milestone shipped an exported
`tf_test_set_uninitialized_poison` and matching Python helpers, on the
argument that a thread-local seam disarmed by default is harmless. That
argument is rejected: a symbol compiled into and exported from the normal
runtime is part of the runtime whatever its intended audience, and one
that can alter what production allocations contain is not something the
shipped library should offer. The mechanism was removed and the proof
rebuilt around the allocator instead (§16.1.3), losing no coverage.

Both exported constructors run through **one file-local body**
(`create_storage(size, zero_initialize)`), so the two cannot drift apart
on any shared guarantee. `zero_initialize` is a compile-time-fixed
argument of the two exports, not a runtime switch. The zero-initializing
path is still the default and its behavior is byte-for-byte what it was
before H1.

On the Python side:

| Name | Visibility | Notes |
|---|---|---|
| `NativeStorage.__init__(..., *, _zero_initialize=True)` | private keyword | Both allocation kinds pass through the one constructor, so **every** live-storage accounting hook in the test suite — each of which wraps `NativeStorage.__init__` — sees an uninitialized allocation exactly as it sees a zeroed one. |
| `NativeStorage._uninitialized(size, ...)` | private | The classmethod that sets that flag. Also the single seam the poison tests wrap. |
| `NativeTensorCore._uninitialized(shape, ...)` | private | The Core-level counterpart of `zeros`. |

**No public API was added.** There is no `Tensor.empty`, no
`NativeTensor.empty`, no `empty_like`, no registry capability, and no
stable-framework surface. A test asserts the absence of each. The
installed backend module likewise exposes no poison-shaped name, which is
asserted over `dir(cpp)` and against the loaded library through
`ctypes`.

### 16.1.2 The per-kernel audit

Every production destination allocation was re-read from source. H0's
preliminary "12 of 14" sketch was **not** trusted, and the real inventory
differs from it: in particular the three scatter-add backwards **do**
qualify (they zero their own span before accumulating, so the caller's
fill is pure duplication), while `tf_core_sum` and
`tf_core_narrow_backward` do not.

Column meanings. **Every element written?** — does the kernel assign
every element of the destination it was given? **Read before first
write?** — does the kernel ever read a destination cell before writing
it? **Uninit?** — is the uninitialized path enabled?

| Operation / kernel | Destination allocation site | Every element written? | Read before first write? | Uninit? | Reason | Proof |
|---|---|---|---|---|---|---|
| `add` / `subtract` / `multiply`, same-shape contiguous | `_binary_core_op` | Yes — flat loop over `[0, numel)` | No | **Enabled** | `dst[i] = op(a[i], b[i])` for every `i`; the destination holds exactly `numel` | poison, both patterns, contiguous and nonzero-offset |
| `add` / `subtract` / `multiply`, same-shape strided | `_binary_core_op` | Yes — odometer, one write per logical position | No | **Enabled** | Same coverage, different traversal | poison through a real transposed view |
| `add` / `subtract` / `multiply`, broadcast | `_binary_core_op` | Yes — odometer over `prod(out_shape)` | No | **Enabled** | Broadcasting changes how *operands* are read (zero strides), never which output positions are visited | poison over a `(7,5)` with `(1,5)` case |
| `relu` | `relu` | Yes | No | **Enabled** | `core_unary` assigns every element | poison: contiguous, strided, narrowed |
| `sqrt`, `reciprocal`, `exp`, `log` | `_unary_compute` | Yes | No | **Enabled** | Same `core_unary` coverage | poison, both patterns |
| `contiguous_copy` (Core and View) | `contiguous_copy` | Yes | No | **Enabled** | `core_unary` with the identity op | poison over transposed and narrowed views |
| `matmul` | `matmul` | Yes — `dst[i*p + j] = sum` over the full `(m, p)` | No — accumulates into a **local register**, then assigns once | **Enabled** | The destination is never an accumulator | poison; also the precondition H2 must preserve |
| `relu_backward` | `relu_backward` | Yes — binary odometer | No | **Enabled** | Same coverage as the binary ops | poison |
| `softmax`, `log_softmax` | `_axis_fused_forward` | Yes — pass 2 assigns every `(outer, k, inner)` slot | No — pass 3's read-modify-write reads what pass 2 wrote | **Enabled** | The index map covers `[0, numel)` bijectively | poison on both axes |
| `cross_entropy_forward` — loss | `cross_entropy_forward` | Yes — the single scalar element | No | **Enabled** | `*loss = ...` unconditionally | poison, both reductions |
| `cross_entropy_forward` — probabilities | `cross_entropy_forward` | Yes — every `(batch, class)` | No | **Enabled** | Pass 2 writes the whole row; pass 3 normalizes what it wrote | poison, both reductions |
| `cross_entropy_backward` | `cross_entropy_backward` | Yes — every `(batch, class)` | No | **Enabled** | Assigns the gradient for every class | poison |
| `conv2d_forward` | `conv2d_forward` | Yes — assigns over the full output extent | No — bias seeds a **local** accumulator | **Enabled** | Padding skips *source* reads, never output positions | poison at stride/padding `(1,0)`, `(1,1)`, `(2,1)` |
| `conv2d_input_backward` | `conv2d_input_backward` | Yes — **the kernel zeroes its own whole span first** | No | **Enabled** | It writes `0.0` across the complete `N*C*H*W` before any accumulation; the caller's fill was duplication | poison at three stride/padding settings |
| `conv2d_weight_backward` | `conv2d_weight_backward` | Yes — same self-zeroing over `O*C*kh*kw` | No | **Enabled** | Same | poison at three settings |
| `maxpool2d_forward` — values | `_maxpool2d_forward_with_winners` | Yes — assigns over the full output extent | No | **Enabled** | The running best is a **local**, initialized before the window loop, so even a degenerate window assigns | poison at three kernel/stride settings |
| `maxpool2d_forward` — winners | same | Yes — same extent, same loop | No | **Enabled** | Same | poison |
| `maxpool2d_backward` | `maxpool2d_backward` | Yes — **the kernel zeroes its own whole span first** | No | **Enabled** | Same self-zeroing as the convolution backwards; overlapping windows then accumulate | poison including overlapping windows (kernel 3, stride 1) |
| `dropout_forward` — output | `_dropout_forward_with_mask` | Yes — every `i` in `[0, count)` | No | **Enabled** | One pass writes both destinations | poison at `p` of 0.0, 0.25, 0.9 |
| `dropout_forward` — mask | same | Yes — same loop | No | **Enabled** | Same | poison at the same probabilities |
| `NativeStorage.from_array` | `from_array` | Yes — the copy writes the whole `storage->size` | No | **Enabled** | The size comes from the same array, so no element can survive | poison |
| `NativeTensorCore.full` | `full` | Yes — the fill writes every element | No | **Enabled** | `float(value)` is evaluated **before** the allocation, so a bad value allocates nothing | poison plus a value check |
| **`sum` / `mean`** (`tf_core_sum`) | `sum` | **No** — accumulates | **Yes** — `dst[out_pos] += src[in_pos]` | **REJECTED** | The zero is the **additive identity** the reduction starts from. Every reduced axis folds many inputs into one cell, which is read on every accumulation after the first. | negative control: with poison armed and the fast path forced, the result is NaN rather than the correct sum |
| **`narrow_backward`** (`tf_core_narrow_backward`) | `narrow_backward` | **No** — writes only the narrowed region | No, but leaves cells untouched | **REJECTED** | The un-narrowed cells' zero **is the gradient's value**, not an initialization detail. An uninitialized buffer would leak heap contents into a gradient. | negative control: 10 of 15 cells retain poison, exactly the un-narrowed rows |
| `NativeTensorCore.zeros` | — | n/a | n/a | **REJECTED by definition** | Its contract *is* zeros, and the two rejected kernels depend on it | asserted directly |

Two rows deserve emphasis, because they are the ones an incautious
reading gets wrong in opposite directions:

- **The three scatter-add backwards qualify.** "Accumulating kernel" does
  not automatically mean "needs a zeroed destination" — what matters is
  whether *the caller* must supply the zero. These three supply their
  own, and always did; H1 simply stopped paying for it twice.
- **`matmul` qualifies despite accumulating.** It accumulates into a
  local `double sum` and assigns once, so the destination is never read.

### 16.1.3 Why poison, where it lives, and what each tool actually proves

Three tools prove three different things, and they are not
interchangeable:

| Tool | Proves | Does **not** prove |
|---|---|---|
| **Deterministic poison tests** | Every destination element is written before it is read | Nothing about memory safety or lifetimes |
| **ASan / UBSan** | Memory-boundary safety and undefined behavior | **Nothing about uninitialized-value reads** |
| **LeakSanitizer and live-storage accounting** | Lifecycle cleanup and exactly-once destruction | Nothing about initialization |

The poison exists because the two obvious alternatives do not work:

- **Real uninitialized memory is a useless oracle.** A fresh page from
  the operating system reads back as zeros, so a kernel that skipped an
  element would silently look correct — and would keep looking correct
  until the allocator happened to hand back a dirty page in production.
- **ASan and UBSan do not detect uninitialized *value* reads.** That is
  MemorySanitizer's job, and MSan needs a fully instrumented libc and
  CPython. This project has neither, so **MSan was not used and no MSan
  result is claimed anywhere.** ASan and UBSan remain entirely separate
  from the initialization proof: they are run over the same code, but
  they answer a different question.

The poison fills an uninitialized allocation with a chosen pattern, so an
unwritten element becomes a deterministic, locatable value. Two patterns
are used: a **quiet NaN** with a distinctive payload (`0x7FF8DEADBEEFCAFE`
— it propagates through arithmetic, so an unwritten element that is later
*read* contaminates everything downstream) and a **large negative finite**
value, `-1.2345678901234567e300` (which catches code that special-cases
NaN, or a comparison a NaN would silently fail).

**The poison is injected exclusively by test infrastructure, around the
allocator.** It is not a capability of the runtime, and the runtime has
no way to produce it. The whole mechanism is a context manager in
`tests/test_native_storage_allocation.py` that temporarily wraps the one
private allocation helper, `NativeStorage._uninitialized` — the single
funnel through which every uninitialized allocation passes, including
`NativeTensorCore._uninitialized` and `NativeStorage.from_array`. Per
allocation the sequence is exactly:

1. the **real** `NativeStorage._uninitialized` runs, so the real
   `tf_storage_create_uninitialized` export allocates the buffer;
2. the wrapper fills that buffer with the pattern through the ordinary
   `fill` primitive (`tf_storage_fill`), which writes every element;
3. the **same** storage object is returned to the production operation,
   which then runs the **real** kernel over it.

So the pattern is in place strictly after the real allocation and
strictly before the real kernel executes, which is asserted directly
rather than assumed: an opt-in log records, for every allocation, how
much of the buffer held the pattern when it was handed over — read back
through the very handle the operation receives — and every allocation
must be *entirely* poison at that moment.

**The detector is proved capable of failing**, which is what makes the
passing results mean anything. Four negative controls:

1. A bare uninitialized allocation under poison is *entirely* poison.
2. `tf_core_narrow_backward` aimed at an uninitialized destination leaves
   poison in exactly the un-narrowed cells — 10 of 15 in the test.
3. `tf_core_sum` aimed at an uninitialized destination returns NaN
   instead of the correct sum.
4. A *complete* kernel is given a deliberate hole — the real
   `tf_core_add_contiguous` told to write 8 of a 9-element destination —
   and the very assertion helper that every §16.1.2 proof uses is shown
   to reject it.

Controls 2 and 3 are simultaneously the executable justification for the
two rejections. A fifth control proves the poison never reaches the
zero-initializing path — by construction, since only the uninitialized
helper is wrapped — so the rejection tests are testing the kernels rather
than the poison.

A **mutation test** confirms the suite has teeth: moving `sum` onto the
uninitialized path is caught by **five independent tests** — the
whole-training-step poison proof, the `sum` rejection test, the source
pin, and both bit-identity parity tests.

The C++ CTest (`cpp/tests/test_storage_allocation.cpp`) covers the
allocator contract itself — the zeroed constructor, the uninitialized
constructor's ownership and lifecycle, size validation, zero and negative
sizes, allocation failure, error clearing, repeated create/destroy
cycles, and many live handles accounted independently — against **both**
constructors. It needs no poison-control export and calls none; it
demonstrates the technique using `tf_storage_fill` alone, which is
exactly how the Python suite does it.

### 16.1.4 Numerical parity

H1 changed allocation only, so the requirement is **bit identity**, not a
tolerance, and it is tested directly: every enabled operation is computed
twice — once as shipped, once with the private uninitialized constructors
forced back to `zeros` — and compared for exact array equality. The same
comparison runs over a complete eight-step `NativeAdam` training run,
covering the loss sequence and every final parameter.

Crucially, the uninitialized side of both comparisons runs **under
poison**. Without that, an unpoisoned fresh page usually reads back as
zeros and a kernel with a hole could match the zeroed path by luck.

### 16.1.5 Measured results

Measurement follows §6. The **primary comparison is TensorForge zeroed
versus TensorForge uninitialized** — the same code, the same arithmetic,
the same Python path, differing only in the fill. `numpy.zeros` is
reported for context and is **explicitly not load-bearing**: it is served
by `calloc`, which an operating system can answer with lazy zero pages,
so timing against it measures page-fault policy rather than an allocator
TensorForge could adopt.

**The allocator in isolation** (allocate and close, no compute; medians
of 25 after 6 warm-ups, one machine):

| Elements | MB | Zeroed | Uninitialized | Fill | Ratio |
|---|---|---|---|---|---|
| 1 | 0.00 | 1.40 us | 1.60 us | -0.20 us | 0.9x |
| 1,024 | 0.01 | 1.50 us | 1.70 us | -0.20 us | 0.9x |
| 16,384 | 0.13 | 3.60 us | 1.80 us | 1.80 us | 2.0x |
| 65,536 | 0.52 | 10.60 us | 1.90 us | 8.70 us | 5.6x |
| 262,144 | 2.10 | 580.60 us | 11.10 us | 569.50 us | **52.3x** |
| 1,048,576 | 8.39 | 2035.10 us | 17.10 us | 2018.00 us | **119.0x** |
| 4,194,304 | 33.55 | 8386.20 us | 15.20 us | 8371.00 us | **551.7x** |

That is the honest shape of the win: the fill scales with the buffer
while the allocation itself stays roughly constant. **Below about 16,000
elements the difference falls inside the noise and reads slightly
negative** — reported, not hidden.

**End to end**, the picture is much more modest and several results are
inconclusive. Profile shapes, medians of 15 after 5 warm-ups, with three
independent runs of the volatile cases:

| Case | Shape | Ratio (uninitialized vs zeroed) | Verdict |
|---|---|---|---|
| `storage_allocation` | 2048 x 2048 (32 MB) | **1126x** | Decisive — the fill in isolation |
| `elementwise_contiguous` | 1024 x 1024 (8 MB) | 1.80, 1.54, 1.78 | **Real** — large memory-bound output, cheap arithmetic |
| `contiguous_materialization` | 1024 x 1024 | 1.62 (spread 120%/85%) | Probably real, very noisy |
| `normalized_training_step` | 256 x 256 | 1.02, 1.14, 1.03 | Small positive, variable |
| `adam_step` | 64 x 256 | 1.16, 1.04, 1.13 | Small positive, noisy (spread up to 104%) |
| `linear_forward` | 256 x 512 | 1.05 (spread 39%/36%) | **Inconclusive** |
| `conv2d_forward` | 8 x 3 x 16 x 16 | 1.01 | **No measurable effect** |
| `mlp_training_step` | 64 x 32 | 1.00 | **No measurable effect** |
| `matmul_square_contiguous` | 384 x 384 x 384 | 0.85, 0.88, 0.92 | **Inconclusive, reads negative** — see below |

**The matmul row is reported as-is rather than explained away.** At 384
cubed the kernel performs roughly 44 ms of arithmetic against a 1.2 MB
output whose fill costs well under a millisecond, so the fill sits far
below this machine's run-to-run variation; the sub-1.0 ratios are noise,
not a regression. Nothing in H1 touches matmul's arithmetic, and the
bit-identity test proves its results are unchanged. A matmul improvement
is H2's subject — access pattern, not allocation.

**Honest summary.** H1 removes a cost that is enormous *per byte
allocated* and negligible *per unit of arithmetic performed*. It shows up
clearly in memory-bound work on multi-megabyte outputs and disappears
into the noise everywhere else. It was still worth shipping: it is the
smallest possible change, it is bit-identical, it removes a full write
pass no caller ever observed, and it takes a size-proportional constant
out of every later milestone's measurements.

### 16.1.6 Failure paths

An uninitialized buffer must never reach a caller. Every enabled site now
closes its destination on failure, which required adding the missing
guard to the sites that previously relied on garbage collection —
`relu`, `_unary_compute`, `relu_backward`, both `_binary_core_op` paths,
`matmul`, plus `full` and `from_array`. Tested at:

- invalid arguments **before** allocation (nothing is allocated);
- injected allocation failure (`MemoryError`, live storage unchanged);
- native kernel failure after allocation, across eight kernels;
- a Python-side wrapper failure after the native call;
- a failed copy inside `from_array`;
- a rejected fill value in `full`;
- fifty interleaved success and failure cycles, with live storage
  returning to an exact baseline each time;
- fifty more of those cycles **with the poison wrapper installed**, so
  the test infrastructure is proved not to perturb the accounting it is
  used to verify;
- a failure of the poison fill itself, which closes the storage it had
  just allocated rather than handing back a half-prepared buffer.

No check anywhere in this section relies on garbage collection to reach
its baseline.

### 16.1.7 What H1 did **not** do

No memory pool. No scratch arena or workspace. No global allocator
policy. No environment variable. No heuristic — a site opts in
explicitly, by name, with a row in the table above, and a failed
precondition means the safe path rather than a guess. No public
empty-tensor API. **No poison-control API, debug hook, or global runtime
mode of any kind in the shipped library or the installed Python
backend** — the initialization proof lives entirely in the test suite. No
change to any kernel's arithmetic, loop order, or traversal. No
capability, dtype, device, registry value, checkpoint field, or
checkpoint version moved, and the only export added is
`tf_storage_create_uninitialized`.

### 16.1.8 Validation

The matrix H1 was accepted against, re-run in full after the
poison-control removal.

**Windows.** A fresh Release build (Visual Studio 17 2022, MSVC
19.44.35207) and a fresh Debug build, the Debug library written outside
the repository so the **active runtime stays the Release DLL** (58,880
bytes, unchanged; the Debug DLL is 177,152 bytes and lives elsewhere).
Both configurations build with **zero project compiler, linker, and CMake
diagnostics** and pass **12/12 CTests** (Release 0.92 s, Debug 1.07 s).
The full Python suite is **5,108 passed**;
`scripts/smoke_cpp_backend.py` passes; the Phase-H harness passes all 24
correctness gates in both `--smoke` and `--smoke --json` and writes no
result file; stable `tensorforge` imports pull in **no** native or
experimental module; and the deterministic-training and exact-resume
suites pass, including `examples/native_dropout_training.py` reproducing
its exact stochastic resume with live native storage 0 → 0.

**Export inventory.** The built DLL's own export directory lists **52**
symbols, all `tf_*`: the pre-H1 baseline of 51 plus
`tf_storage_create_uninitialized`. No name contains "poison", and asking
the loaded library for `tf_test_set_uninitialized_poison` through the
platform loader raises `AttributeError`.

**Clang ASan/UBSan** (18.1.3, WSL2 Ubuntu 24.04.4,
`-DTF_SANITIZE=address,undefined`, fresh out-of-source build, zero
compiler diagnostics). Instrumentation is **proved present**: `nm -D`
shows **22 `__asan*`** and **14 `__ubsan*`** dynamic symbols beside the
**52** exported `tf_*` symbols, `tf_storage_create_uninitialized` among
them and **zero** symbols matching "poison"; the library refuses to load
without the sanitizer runtime. Under
`halt_on_error=1:abort_on_error=1:detect_stack_use_after_return=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`: **12/12 sanitized
CTests** with `detect_leaks=1`, **2,049** sanitized Python tests across
the H1 and native suites, **432** more across the deterministic-training
and exact-resume suites, the G7 example reproducing its exact resume, and
the harness passing all 24 gates in both modes — with **zero ASan errors
and zero UBSan runtime errors**. A focused re-run of the H1 suite, the
documentation guardrails, and the harness contract tests against the
final sources adds **326** more.

**LeakSanitizer.** Three complete harness lifecycles returned native live
storage **exactly to baseline (0 → 0)** at every checkpoint. The
remaining process-exit allocations — 775,248 bytes in 694 allocations —
contain **no TensorForge frame**: every named frame is CPython, libc,
NumPy, or the ASan runtime itself. **No suppression file was added.**

---

## 16.2 H2 — matmul memory access, as shipped

**Status: complete.** H2 changed *how the production matmul walks
memory* and nothing else. No arithmetic changed, no accumulation was
reassociated, no operand was materialized, no capability moved, and **no
C ABI symbol was added** — the exported surface is still exactly the 52
symbols H1 left.

The milestone title says "and cache blocking". **Blocking was measured
and rejected** (§16.2.7). What shipped is the simpler and measurably
better change: a loop-order swap.

**On the numerical claim, stated up front so it is not discovered
later:** H2 does **not** claim unconditional bit identity between the two
paths. It claims an identical accumulation sequence, bit identity on
every non-NaN result, NaN-class equivalence, and it deliberately leaves
the *payload bits of a NaN result* outside the contract, because those
follow from the compiler's instruction operand ordering rather than from
the arithmetic. §16.2.3 states all four parts, shows the ten formulations
measured while trying to close the fourth, and records why closing it is
not available short of abandoning the optimization.

### 16.2.1 What the production kernel was, and what it is

**Before.** One kernel, `i`–`j`–`k`, reading both operands through their
own strides and offsets:

```cpp
for i, for j:
    double sum = 0.0;
    for k: sum += a[a_off + i*as0 + k*as1] * b[b_off + k*bs0 + j*bs1];
    dst[i*p + j] = sum;
```

Its innermost loop walks `b` **down a column**, stepping `bs0` doubles
per element. For a row-major right operand — which is every weight a
`NativeLinear` holds — that is a new cache line on every step.

**After.** Two kernels behind the same unchanged export, chosen from
metadata:

| Function (all `namespace tf`, hidden visibility) | Role |
|---|---|
| `matmul_generic_strided` | The **retained generic reference path** (§8.3). Byte-for-byte the pre-H2 loop. Shipped, reachable through ordinary production dispatch, and the oracle every optimized result is compared against. |
| `matmul_row_sweep` | The optimized path: `i`–`k`–`j` over `MATMUL_ROW_BLOCK` destination rows at a time. |
| `matmul_prefers_row_sweep` | The dispatch predicate. |

The row sweep, for one group of rows:

```cpp
// k == 0 — the assigning pass. Every element of every row in the group
// is written here, before any accumulation reads one.
for r in group: for j in [0, p): out_r[j] = 0.0 + a[i0+r, 0] * b_row0[j];
// k >= 1 — the accumulating passes, ascending.
for k in [1, n):
    for r in group: for j in [0, p): out_r[j] += a[i0+r, k] * b_row_k[j];
```

The innermost loop now walks a **row** of `b` and a row of `dst`
sequentially. Nothing about the arithmetic changed.

Declarations and full rationale live in `cpp/include/tf_matmul_internal.h`;
the bodies and the export in `cpp/src/matmul.cpp`.

### 16.2.2 The exact optimized-path preconditions

`matmul_prefers_row_sweep(m, n, p, b_stride1)` returns true iff **all
three** hold. It is total, pure, and a function of metadata alone — never
of a pointer value, an alignment, a wall-clock reading, an environment
variable, or a CPU-feature probe (§8.1).

| # | Condition | Why |
|---|---|---|
| 1 | `b_stride1 == 1` | Consecutive columns of the right operand are consecutive in memory, which is what makes the row sweep's inner loop a contiguous read. |
| 2 | `n >= 1` | The `k == 0` pass is what assigns every destination element before anything accumulates into it. With no `k` there is no assigning pass. |
| 3 | `p >= MATMUL_MIN_COLUMNS` (8) | Below this the inner `j` loop is too short to pay for its per-`k` setup, and the generic kernel measured strictly better (§16.2.7). |

`m` is deliberately **not** a condition: the row sweep is correct and
useful for every `m >= 0`, including `m == 0`.

**The generic path runs in every other case**, and a failed precondition
is a fallback, never an error. Concretely it runs for:

- a **transposed right operand** (`b_stride0 == 1`, `b_stride1 == p`) —
  which is not a gap but a *choice*: that is the layout the `i`–`j`–`k`
  order already suits, because its inner `k` loop is then the contiguous
  one. §3.1/B2 measured it at 0.39× the contiguous case *before* H2;
- any other non-unit column stride, including a strided or interleaved
  view;
- a result narrower than 8 columns;
- an empty inner dimension.

A **transposed left operand** beside a row-major right one — which is
exactly `db = a.T @ upstream` in the matmul backward — **does** qualify:
`a` is read through its own strides either way, one scalar per `(i, k)`.

### 16.2.3 The numerical contract between the two paths

H2 does **not** claim unconditional bit identity. The contract has four
parts, and they are separate claims about separate things.

#### Part 1 — accumulation order is preserved exactly

For a fixed output `(i, j)`, the row sweep accumulates:

```
0.0, then k = 0, then k = 1, ... then k = n-1
```

which is the same starting value, the same products, the same
operations, and the same order as the generic kernel's
`double sum = 0.0; for k: sum += a*b`. No addition is reassociated, no
partial sums are combined, no wider or narrower accumulator is used, no
fused multiply-add is requested, and no parallel or vector reduction
exists. The `j` loop that the compiler may vectorize carries **no**
reduction: each lane is a different output element accumulating its own
independent sequence, so vectorizing it cannot reorder anything.

This is a claim about the *sequence of operations*, and it is exact. It
is what parts 2 and 3 rest on.

#### Part 2 — every non-NaN result is bit-identical

Whenever the result is not a NaN, the two paths agree **bit for bit** —
`+0.0` versus `-0.0`, `±inf`, denormals, the smallest normal, and the
largest finite magnitudes included. This is raw IEEE-754 bit-pattern
equality, not a tolerance.

**The `0.0 +` on the `k == 0` pass is load-bearing and is written out
deliberately.** `0.0 + (-0.0)` is `+0.0`, while `-0.0` alone is `-0.0`.
Dropping the addition would change the sign of a zero result — for
example `[[-1.0]] @ [[0.0, …]]`, where the reference produces `+0.0`. It
is not redundant, it is not foldable under IEEE-754, and both a C++ and
a Python test fail without it.

**This is the part every practical claim rests on.** Every committed loss
trajectory, every example, every benchmark case, and every bit-exact
checkpoint-resume proof in this project runs on finite data, so part 2
covers all of them completely — which is why D11's, E8's, F6's, and G7's
exact-resume proofs survive H2 untouched, and are re-run to prove it.

#### Part 3 — NaN-class equivalence

Whenever either path produces a NaN, **both** do, in exactly the same
positions, and both are **quiet**. Neither path can produce a signaling
NaN. A path that produced a NaN where the other produced a number, or an
infinity where the other produced a finite value, would be a real defect,
and both suites assert against it.

#### Part 4 — NaN payload bits are outside the contract

The payload and sign bits of a NaN *result* may differ between the two
paths, and TensorForge specifies nothing about them.

This is a measured property of the code, not a hedge, and the
investigation behind it is recorded rather than summarized away. On
x86-64, `ADDSD dst, src` returns the **destination** operand's NaN when
both operands are NaN. Which of the two addends the compiler places in
the destination is an instruction-selection decision that C++ cannot
express. In the `i`–`j`–`k` kernel MSVC places the freshly computed
product there, so the **last** NaN in `k` order survives; in the
`i`–`k`–`j` row sweep it places the accumulator there, so the **first**
survives.

**Ten source-level formulations were measured**, in a focused MSVC
Release harness, against the same NaN-saturated matrix:

| Formulation | Payload differences | Finite results |
|---|---|---|
| `out[j] += a_ik * b_row[j]` (shipped) | 162 / 208 | bit-identical |
| `out[j] = out[j] + a_ik * b_row[j]` | 162 / 208 | bit-identical |
| `double acc = out[j]; acc = acc + …; out[j] = acc` | 162 / 208 | bit-identical |
| named product, then `acc + product` | 162 / 208 | bit-identical |
| named left/right, `(left) + (right)` | 162 / 208 | bit-identical |
| shipped form with `#pragma loop(no_vector)` | 162 / 208 | bit-identical |
| shipped form with `__restrict` on both pointers | 162 / 208 | bit-identical |
| blocked `4 × 64` stack accumulator tile | 162 / 208 | bit-identical |
| blocked `4 × 4` stack accumulator tile | 162 / 208 | bit-identical |
| **`i`–`j`–`k` register accumulator (the generic order)** | **0 / 208** | bit-identical |

Three conclusions follow, and each rules something out:

- **It is not vectorization.** Disabling inner-loop vectorization changes
  nothing, so no SIMD or auto-vectorization question is involved.
- **It is not the destination-versus-stack accumulator.** Both blocked
  variants, whose accumulator is a local array rather than the output
  buffer, behave identically to the shipped form.
- **It is the loop order itself.** The only structure that reproduces the
  reference's payloads is `i`–`j`–`k` — which is precisely the
  arrangement H2 exists to replace. Payload parity and H2's memory-access
  improvement are the same decision viewed two ways, so parity is not
  available at any price short of abandoning the optimization
  (4.1–4.7× at the profile shape, §16.2.8).

Forcing it was therefore rejected. The alternatives considered and
declined: reverting the loop order (loses the entire milestone); a
NaN-detecting fix-up pass (a branch per element in a numerical inner
loop, for a property no caller can rely on); `#pragma loop(no_vector)`
(does not even work, and would be a compiler-specific directive); and
`volatile` or fast-math-adjacent tricks (fragile, and out of scope).

**Measured across builds**: MSVC Release differs on 162 of 208; MSVC
Debug and Clang 18 `-O0` differ on **0 of 208**. Both outcomes conform —
part 4 is a licence, not a requirement — and the tests assert nothing in
either direction, which is why they pass unchanged on all three.

#### What this is *not*

It would be wrong to call the payload difference "not a behavioral
difference". It is one: the bits of a NaN result can differ between two
paths that a caller cannot choose between. What is true is narrower and
is stated above — the difference is confined to the payload bits of a
value that is already NaN, those bits have never been part of any
TensorForge contract, and a NaN result means the computation has already
left the supported numerical domain.

#### Evidence

Parts 1–3 are asserted as raw IEEE-754 bit patterns, not tolerances, at
three layers:

- `cpp/tests/test_matmul.cpp` drives the two internal kernels over every
  `(m, n, p)` from a dimension set spanning 1–65 (primes, powers of two,
  the row-block and column-threshold boundaries and both sides of each,
  one-element dimensions, rectangular shapes, multiple row groups and a
  partial final group) plus larger shapes, transposed / narrowed /
  non-unit-stride left operands, a special-value matrix, and dedicated
  cases for a NaN in the left operand, a NaN in the right operand,
  multiple payloads and a negative NaN meeting in one accumulation, a NaN
  manufactured by `0 * inf` and `inf - inf`, infinities without a NaN,
  and signed zeros with denormals and large finite magnitudes. It also
  checks both paths against the independent raw-buffer `tf_matmul`,
  making the agreement three-way.
- `tests/test_native_matmul_dispatch.py` does the same through the real
  production Core, `NativeTensor`, the autograd node, `NativeLinear`, and
  both optimizers, by giving the *same logical right operand* through a
  qualifying layout and through one that cannot qualify.
- The H0 harness's matmul cases gate the two paths bit-for-bit **before**
  either is timed. Unqualified equality is the right gate there because
  every harness operand is finite seeded data, so part 2 applies in full.

### 16.2.4 H1 compatibility: the destination is still fully written first

H1's audit table (§16.1.2) listed `matmul` as enabled with the reason
"accumulates into a **local** register, then assigns once — the
destination is never an accumulator". **That reason no longer describes
the optimized path**, and the row is restated here rather than left
stale:

| Path | Every element written? | Read before first write? | Reason |
|---|---|---|---|
| `matmul_generic_strided` | Yes — `dst[i*p + j] = sum` over the full `(m, p)` | No — a local `double sum` | Unchanged from H1 |
| `matmul_row_sweep` | Yes — the `k == 0` pass assigns every element of every row in the group | No — accumulation only ever touches elements that pass already wrote | The `n >= 1` precondition is what guarantees the assigning pass runs |

So the destination is still legal to allocate uninitialized, for a
different reason on each path. This is the milestone's "explicitly
assigning every output element before any destination accumulation"
design, and it is proved rather than argued:

- **Python** (`tests/test_native_storage_allocation.py`): both paths are
  run over a poisoned destination, with both patterns (the quiet NaN
  `0x7FF8DEADBEEFCAFE` and the large negative finite
  `-1.2345678901234567e300`), across shapes spanning the row-block
  boundary (partial group, exact group, several groups with a partial
  tail) and the column threshold on both sides. No poison survives.
  A second test makes the **stronger** claim: the same product computed
  over a NaN-poisoned buffer, a finite-poisoned buffer, and an ordinary
  zeroed buffer agrees **bit for bit**, so no output value depends in any
  way on prior destination contents.
- **C++** (`cpp/tests/test_matmul.cpp`): the same three-way comparison
  directly on the kernel, plus a **negative control** — the same kernel
  told the destination is one column narrower than it is, which must
  leave exactly the 4 untouched cells holding poison. That is what makes
  the passing results mean something.

No poison-control mechanism was added anywhere; the technique is still
purely test infrastructure wrapped around the allocator (§16.1.3).

### 16.2.5 What H2 added, and what it did not

**Added:**

| Item | Kind |
|---|---|
| `cpp/include/tf_matmul_internal.h` | internal header, hidden visibility |
| `tf::matmul_generic_strided`, `tf::matmul_row_sweep`, `tf::matmul_prefers_row_sweep` | internal C++, not exported |
| `cpp/tests/test_matmul.cpp` + its CTest target | test scaffolding (CTest count 12 → **13**) |
| `tests/test_native_matmul_dispatch.py` | tests |
| two matmul poison tests in `tests/test_native_storage_allocation.py` | tests |
| the `tensor_core_generic` harness layer and the `dispatch_comparison` payload block | measurement |

**Not added, and deliberately so:** no exported C ABI symbol (the count
is still **52**); no kernel selector, block-size setter, benchmark hook,
dispatch tracer, reference-kernel selector, or CPU-feature control; no
environment variable; no runtime autotuning; no stored machine-specific
tuning result; no user-selectable kernel mode; no public dispatch
control; no threading, SIMD, OpenMP, BLAS, memory pool, or scratch
workspace; no new build option; no materialization of any operand to get
it onto the fast path; no change to the public shape contract; no
capability, dtype, device, registry value, checkpoint field, or
checkpoint version.

`tests/test_native_matmul_dispatch.py` asserts the absence of nine
plausible dispatch-control symbol names against the **loaded** library
through the platform loader, and asserts that the three internal
functions are not exported either — which is precisely why the C++ test
target compiles `matmul.cpp` in rather than linking the shared library.

### 16.2.6 Why `tf_matmul_tiled` was not adopted

The milestone required inspecting the pre-existing raw tiled kernel
rather than assuming it. It was, and it is **not** appropriate as
production code:

| Question | Answer |
|---|---|
| Does it assume contiguous inputs? | **Yes.** Plain `const double*` row-major buffers, `a[i*n + k]`, `b[k*p + j]`. It cannot read a stride or an offset, so a transposed or narrowed view would have to be materialized first — which the milestone forbids without measured end-to-end benefit and explicit approval. |
| Does it assume a zero-initialized output? | It **zeroes its own** destination (`out[i] = 0.0` over the whole span) and then accumulates. So it is safe, but the zeroing is a **full extra write pass** over the output — exactly the cost H1 removed. |
| Does it preserve accumulation order? | Yes — ascending `k` blocks, ascending `k` inside a block. §3.1/B2 measured it bit-identical to `tf_matmul`. |
| Rectangular shapes? | Yes, `m`, `n`, `p` independent. |
| Non-dividing block boundaries? | Yes, min-clamped ends. |
| Error and ownership model? | **No.** No `TF_GUARD`, no `Storage` handle, no error contract. It is a raw-buffer benchmark kernel. |
| Verdict | **Benchmark and reference code, not production code.** Retained exactly as it was. |

Nor was it duplicated: the shipped row sweep is structurally a different
kernel (strided-aware, no destination zeroing, no `j`/`k` tiles), so
there are not two nearly identical tiled implementations in the tree.
There is one tiled kernel — the pre-existing benchmark one — and it is
still measured beside the production path in the harness, now as the
*measured alternative that was not adopted*.

### 16.2.7 The blocking decision, and the block size and threshold evidence

Candidates were compiled into a throwaway measurement library (outside
the repository, on no production path) and timed on identical data, with
**every** variant's output compared bit-for-bit against the current
production loop before timing. Every variant agreed on every finite result; the payload-only differences discussed in §16.2.3 were identical across all of them.

Two families were evaluated:

- **`i`–`k`–`j` with a stack accumulator tile**, `BI × BJ`, over
  `BI ∈ {1, 2, 4, 8, 16}` and `BJ ∈ {16, 32, 64, 128, 256}` — 22
  combinations, the classic cache-blocking shape;
- **`i`–`k`–`j` row sweeps at full row width** (no `j` blocking),
  `BI ∈ {1, 2, 3, 4, 8}`.

Shapes: 4³, 8³, 16³, 31³, 32³, 63³, 64³, 127³, 128³, 255³, 256³, 384³,
tall-skinny `512×64×8`, short-wide `8×64×512`, `4096×8×8`,
`256×256×{1,4}`, `64×256×2048`, `32×128×4096`, the Linear shapes
`64×256×512` and `256×256×512`, the MLP shape `64×32×64`, and the
weight-gradient shapes `a.T @ u` at `256×64×512` and `32×64×64`.

**Blocking lost, at every non-trivial size.** Speedups over the pre-H2
loop at 384³, best of each family: blocked `BI=16, BJ=64` **3.33×**
versus row sweep `BI=4` **5.50×**. At 256³: 3.10× versus 4.67×. At
`64×256×512`: 3.36× versus 5.38×. The reason is visible in the
structure — a `BJ`-wide tile shortens the inner loop and adds a zero-and-
store pass per tile, while the full-width sweep keeps one long
vectorizable inner loop and the destination row hot in L1 for the whole
`k` range. **So H2 ships the simpler design and records the negative
blocking result**, exactly as the milestone's fallback clause requires.

**The row block.** Within the row-sweep family, `BI = 1` was consistently
the slowest at large sizes (4.18×/4.26×/4.78× at 384³ over three runs)
and `BI ∈ {3, 4, 8}` the fastest. `BI = 4` was chosen: it won or tied
most often, was **never** the worst, and its L1 working set is
`(BI + 1) × p` doubles — the smallest of the leading candidates, which is
what makes it the portable choice rather than the local optimum. `BI = 8`
was marginally ahead on a few shapes and behind on others; at `p = 1024`
its working set leaves a 48 KB L1 while `BI = 4`'s does not.

**The column threshold.** Sweeping `p ∈ {1 … 64}` at
`m ∈ {32, 256} × n ∈ {16, 64, 256}`, the row sweep is a clear **loss**
below `p = 8` — down to **0.27×** at `256×64×1` and 0.69–0.85× at
`512×512×4` — and level-to-ahead from `p = 8` up. `MATMUL_MIN_COLUMNS = 8`
is that crossover. It is a fixed metadata threshold, not a heuristic, and
both suites test `p = 7`, `p = 8`, and `p = 9` explicitly.

Both constants are compile-time values in a shipped header. There is no
autotuner and no stored measurement: the same source makes the same
dispatch decision on every machine, which is what keeps every bit-exact
resume proof reproducible.

### 16.2.8 Measured results

Same methodology as §6. The **before** column is the real production
path built from the pre-H2 `matmul.cpp` into a separate library outside
the repository and loaded through the same Python; the **after** column
is the shipped one. Medians, two independent runs each, one machine.

The table carries **control rows in bold italics**: cases whose code is
byte-identical in both builds, because they fall to the generic path in
each (`p < 8`, or a transposed right operand). Their ratios measure this
machine's noise floor, and nothing inside that band should be read as an
effect.

| Case (Core matmul) | Before (µs) | After (µs) | Ratio |
|---|---|---|---|
| ***4×4×4 (control, p < 8)*** | 9.30 / 9.20 | 12.80 / 13.10 | *0.71–0.73×* |
| ***256×256×4 (control)*** | 114.95 / 118.80 | 116.70 / 235.65 | *0.50–0.99×* |
| ***256×256×1 (control)*** | 31.10 / 34.70 | 34.90 / 34.90 | *0.89–0.99×* |
| ***128³, transposed rhs (control)*** | 1006 / 751 | 701 / 751 | *1.00–1.44×* |
| ***384³, transposed rhs (control)*** | 29392 / 25026 | 25014 / 25853 | *0.97–1.18×* |
| 8×8×8 | 10.00 / 10.20 | 11.90 / 15.70 | 0.65–0.84× |
| 16³ | 11.20 / 11.10 | 12.60 / 16.50 | 0.67–0.89× |
| 31³ (prime) | 19.30 / 17.70 | 19.80 / 24.50 | 0.72–0.97× |
| 32³ | 19.55 / 18.90 | 19.30 / 23.30 | 0.81–1.01× |
| 63³ | 107.35 / 75.65 | 68.75 / 76.45 | 0.99–1.56× |
| 64³ | 101.35 / 78.05 | 57.45 / 67.80 | 1.15–1.76× |
| 127³ (prime) | 768 / 698 | 630 / 561 | 1.22–1.25× |
| **128³** | 1830 / 1823 | 441 / 409 | **4.15–4.45×** |
| 255³ | 6952 / 7312 | 5516 / 2662 | 1.26–2.75× |
| **256³** | 13069 / 12668 | 3440 / 4252 | **2.98–3.80×** |
| **384³ (profile shape)** | 46154 / 44084 | 9865 / 10824 | **4.07–4.68×** |
| tall-skinny 512×64×8 | 97.3 / 89.8 | 80.6 / 76.3 | 1.18–1.21× |
| **short-wide 8×64×512** | 205 / 192 | 48.9 / 48.4 | **3.97–4.20×** |
| **Linear 64×256×512** | 5675 / 5530 | 1322 / 2250 | **2.46–4.29×** |
| **Linear 256×256×512** | 23227 / 26182 | 6044 / 5550 | **3.84–4.72×** |
| MLP 64×32×64 | 74.3 / 50.1 | 27.9 / 53.5 | 0.94–2.67× |

End to end:

| Case | Before (µs) | After (µs) | Ratio |
|---|---|---|---|
| **`NativeLinear` forward 64×128→128** | 960 / 955 | 220 / 241 | **3.96–4.36×** |
| **`NativeLinear` forward 256×512→512** | 78124 / 65806 | 11576 / 11930 | **5.52–6.75×** |
| **`NativeLinear` backward 64×128→128** | 1253 / 1159 | 651 / 671 | **1.73–1.92×** |
| **`NativeLinear` backward 256×512→512** | 116429 / 113505 | 46404 / 46030 | **2.47–2.51×** |
| MLP Adam step 64×32→32 | 2607 / 2288 | 2742 / 3384 | 0.68–0.95× |
| **MLP Adam step 128×256→256** | 48352 / 50260 | 20309 / 24114 | **2.00–2.38×** |

**Tiny and small shapes: no measurable effect, and the apparent losses
are not real.** Every row below about 32³ sits inside the control band,
and the *controls themselves* — where the compiled code is identical —
range 0.50× to 1.44×. A higher-repetition re-run (600–4000 samples per
point instead of 100–600) resolves the small shapes to: 8³ 0.86–1.05×,
8×8×16 1.05–1.44×, 16³ 1.01–1.30×, 32³ 1.08–1.75×, 64³ 1.62–1.97×,
against controls of 0.96–1.44×. **The honest reading is that below ~32³
the ~10 µs fixed Python dispatch cost (§3.1/B3) dominates and H2 is
invisible; the win becomes real and reproducible from 64³ up.**

**The small MLP step shows no measurable change (0.68–0.95×, and
0.95–0.99× at higher repetitions), and that is expected rather than a
regression**: its matmuls are `64×32×32`, tens of microseconds inside a
~2.3 ms step that §3.1/B4 measured to be 83 % optimizer allocations. The
matmul was not its bottleneck before H2 and is not after. The larger MLP
step, where the matmuls are `128×256×256`, does move — 2.00–2.38×.

**The backward gains less than the forward, and the reason is in the
contract, not in the measurement.** `da = upstream @ b.T` feeds the
kernel a *transposed* right operand, which takes the generic path by
design; only `db = a.T @ upstream` qualifies. So a Linear backward gets
roughly one of its two matmuls accelerated, and 1.7–2.5× is what that
looks like.

**Nothing here is asserted by any test**, and no CI job runs it. §3.0's
caveat about this machine applies unchanged.

### 16.2.9 Validation

**Windows.** A fresh Release build (Visual Studio 17 2022, MSVC
19.44.35228.0, CMake 4.4.0) and a fresh Debug build, the Debug library
written outside the repository so the **active runtime stays the Release
DLL** (61,440 bytes; the Debug DLL is 178,688 bytes and lives elsewhere).
Both configurations build with **zero project compiler, linker, and CMake
diagnostics** and pass **13/13 CTests** (0.36 s and 1.78 s in the final
runs).
The full Python suite is **5,232 passed, 0 skipped, 0 failed** (the
post-H1 baseline of 5,108 plus H2's 124);
`scripts/smoke_cpp_backend.py` passes; the harness passes all 24
correctness gates in `--smoke`, `--smoke --json`, and
`--workload matmul`, and writes no result file; stable `tensorforge`
imports pull in **no** native or experimental module; and every
deterministic-training and exact-resume suite passes, including
`examples/native_dropout_training.py` reproducing its exact stochastic
resume with live native storage 0 → 0.

**Export inventory.** The built DLL's own export directory lists **52**
symbols, all `tf_*` — unchanged from H1. `matmul_row_sweep`,
`matmul_generic_strided`, and `matmul_prefers_row_sweep` do **not**
appear, and neither does any of the nine dispatch-control names the scope
test probes for through the platform loader.

**Clang ASan/UBSan** (18.1.3, WSL2 Ubuntu 24.04.4,
`-DTF_SANITIZE=address,undefined`, fresh out-of-source build, **zero**
compiler diagnostics). Instrumentation **proved present**: `nm -D` shows
**22 `__asan*`** and **14 `__ubsan*`** dynamic symbols beside the **52**
exported `tf_*` symbols and **0** matmul internals; the library refuses
to load without the sanitizer runtime. Under
`halt_on_error=1:abort_on_error=1:detect_stack_use_after_return=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`: **13/13 sanitized
CTests** with `detect_leaks=1`; **1,789** sanitized Python tests across
the H2, H1, harness, and native suites; **450** more across the
deterministic-training and exact-resume suites; the harness passing all
24 gates in both modes plus the focused matmul workload; and the G7
example reproducing its exact resume — with **zero ASan errors and zero
UBSan runtime errors**.

**LeakSanitizer.** A complete lifecycle — 160 matmul success cycles over
both paths, 20 interleaved injected-failure and success cycles, 6
training runs, a focused harness run, and a full smoke harness run —
returned native live storage **exactly to baseline (0 → 0)** at every
checkpoint. The remaining process-exit allocations (784,190 bytes in 702
allocations) contain **no TensorForge frame**: every named frame is
CPython, libc, NumPy, or the ASan runtime. **No suppression file was
added.**

### 16.2.10 What H2 leaves for H3

`tf_core_matmul` is now bound by the same fixed per-call Python cost
everything else is (§3.1/B3), which is precisely why the sub-32³ rows
above show nothing: a ~10 µs floor sits under a ~10 µs kernel. That is
H3's subject, and H2 sharpened rather than answered it.

---

## 16.3 H3 — metadata and dispatch efficiency, as shipped

**Status: complete.** H3 reduced the repeated Python-side metadata
normalization on the way to a native kernel. It is a **Python-only**
milestone: no C++, no C ABI symbol, no ctypes declaration, no kernel, no
arithmetic, no traversal order, no dtype, no device, no registry value,
no checkpoint field, and no public API changed. The library still exports
exactly **52** `tf_*` symbols.

### 16.3.1 The pre-H3 metadata flow, measured

`shape_info` was the hub, and it re-validated the *same* tuple four
times. With `strides=None`:

| Step | Work |
|---|---|
| `_as_shape(shape)` | `_as_int_tuple` **(1)** + positivity scan |
| `row_major_strides(dims)` | `_as_shape(dims)` → `_as_int_tuple` **(2)** |
| `numel(dims)` | `_as_shape(dims)` → `_as_int_tuple` **(3)** |
| `_as_offset(offset)` | type check |
| `contiguous:` `row_major_strides(dims)` **again** | `_as_int_tuple` **(4)** |

So one `shape_info` call ran `_as_int_tuple` **four** times over a tuple
that was fully validated after the first, and computed the row-major
strides **twice**. `NativeTensorView.__init__` called `shape_info` once;
`NativeTensorCore.zeros`/`_uninitialized` called `numel(shape)` *and
then* constructed a view, validating the caller's shape a **second**
complete time.

Measured per-call cost of that arrangement, isolated (medians, this
machine):

| Helper | Pre-H3 |
|---|---|
| `_as_int_tuple((4, 5, 6))` | 0.70 µs |
| `_as_shape((4, 5, 6))` | 0.80 µs |
| `row_major_strides` (validating) | 1.10 µs |
| `numel` (validating) | 0.90 µs |
| `shape_info((4, 5, 6))` | 4.10 µs |
| `NativeTensorView(storage, (4, 5, 6))` | 5.00 µs |
| `NativeTensorCore.zeros((1, 1))` + close | 7.50 µs |

Measured **call counts**, one iteration each, by wrapping each helper
(test-local monkeypatching; no production instrumentation exists):

| Workload | `_as_int_tuple` | `shape_info` | `row_major_strides` | `numel` |
|---|---|---|---|---|
| one `(1, 1)` `add` | 13 | 3 | 6 | 4 |
| `NativeLinear` forward | 24 | 5 | 10 | 7 |
| `NativeLayerNorm` forward | 87 | 14 | 30 | 25 |
| `NativeBatchNorm1d` training forward | 195 | 34 | 70 | 65 |
| one `NativeAdam` step | 604 | 108 | 216 | 216 |
| one MLP training step | 815 | 148 | 296 | 287 |
| one CNN training step | 815 | 147 | 296 | 287 |

This both **confirms and refines** §3.1/B3, which recorded 755
`_as_int_tuple` calls per CNN step from a `cProfile` run on a
differently shaped model. It also settles §3.2's first open question —
how much of B3 each helper contributes — and answers §3.4's request for
per-callsite counts *without* adding any production counter.

Evidence level: **directly measured**.

### 16.3.2 What was implemented

**1 — One normalization boundary.** A new private
`_normalized_layout(shape, strides, offset)` performs exactly the checks
`shape_info` always performed, in exactly the same order, with exactly
the same messages, and normalizes the shape **once**. Everything derived
from it — the row-major strides, the element count, the contiguity
comparison — goes through new private `_checked` primitives that perform
no validation *because there is none left to perform*:

| Private primitive | Public counterpart |
|---|---|
| `_row_major_strides_checked(dims)` | `row_major_strides(shape)` |
| `_numel_checked(dims)` | `numel(shape)` |
| `_reduce_shape_checked(dims, axis, keepdims)` | `reduce_shape(...)` |
| `_normalize_axis_checked(axis, dims)` | `_normalize_axis(...)` |
| `_broadcast_shapes_checked(a, b)` | `broadcast_shapes(...)` |

Each public function is now *its validation plus the primitive*, so the
two can never compute different answers — a property the suite asserts
by comparing them over a shape matrix. `shape_info` is
`_normalized_layout` rearranged into the documented dictionary.

The rule for reaching for a `_checked` variant is narrow and mechanical:
**the argument must be a shape tuple that came out of `_as_shape`, or a
live view's own `shape`/`strides`, which is the same thing one
construction earlier.** Anything a caller supplied goes through the
public function. Note what is *not* skipped: `_reduce_shape_checked`
still fully validates `axis` and `keepdims`, and
`_normalize_axis_checked` still fully validates `axis` — those are the
caller-supplied half of the pair.

**2 — Two view constructors, one binding.** `NativeTensorView` keeps its
public constructor (normalize, then bind) and gains a private
`_from_validated(storage, dims, strides, offset)` that skips **only**
the normalization. Both funnel through a shared `_bind`, which performs
the storage open check and the full reachable-offset bounds check.

Two deliberate design choices here:

- The element count and the contiguity flag are **derived inside**
  `_from_validated`, not passed to it. A caller that cannot pass them
  cannot pass an inconsistent pair. This is why H3 has a **separate
  private constructor** rather than a `validated=True` flag on the
  public one — the flag would have been exactly the ambiguous,
  misusable switch the milestone was told to avoid.
- The bounds check is **not** skipped on the private path, because it is
  not a property of the metadata alone: it depends on the storage size,
  and a derived layout can still be handed a storage that has since been
  closed. The suite asserts every out-of-bounds rejection through
  **both** constructors.

**3 — Constructors validate once.** `zeros`, `_uninitialized`, `full`,
and `from_array` now call `_as_shape` once and reuse the result for both
the storage size and the view, via a small `_contiguous_view` helper.

**4 — View operations pass normalized metadata.** `transpose` and
`narrow` derive their layout from the parent's already-validated tuples
and normalize nothing; `reshape` normalizes only the caller's new shape.
`narrow` gained an explicit `int(dim), int(start), int(length)`
normalization immediately after its type check — because the private
constructor no longer re-normalizes, `narrow` is now where a NumPy
integer argument is converted, and the suite pins that the stored shape
and offset are still plain `int`.

**5 — Per-view layout arrays, memoized.** `NativeTensorView` gained a
lazily built, read-only `(shape, strides)` `int64` array pair, returned
by `_native_layout()`; `NativeTensorCore._layout_arrays()` delegates to
it. The inline `np.asarray(...)` pairs in `sum`, `narrow_backward`,
`relu_backward`, and the strided binary path now use it too.

### 16.3.3 The immutability proof

The memoization in (5) is safe because it is memoization of a **pure
function of immutable state**, not a cache with a coherence problem.
Immutability is proved three ways, not assumed:

1. **By construction.** `_shape`, `_strides`, `_offset`, `_numel`, and
   `_contiguous` are assigned in exactly one place — `_bind` — and no
   other statement in `src/` assigns any of them. `close()` sets a flag
   and releases storage; it does not touch the layout, which is why
   metadata has always stayed readable after close.
2. **By behavior.** Every layout-changing operation returns a **new**
   view: `reshape`, `transpose`, `T`, and `narrow` all construct a fresh
   `NativeTensorView` over the same storage. There is no in-place
   reshape and no settable `shape`, so there is no mutation for an
   invalidation design to guard against. A test asserts that after
   deriving four views and running materialization, a copy, and a
   reduction, the original view's five layout fields are unchanged.
3. **By encapsulation.** Nothing outside `cpp.py` reads or writes
   `_shape`/`_strides`/`_numel`/`_contiguous`/`_view` — verified across
   `tests/`, `benchmarks/`, `examples/`, and `scripts/`.

Since no invalidation is ever required, **no invalidation mechanism
exists**, and there is no state in which the cache and the layout can
disagree.

### 16.3.4 What is cached, and what deliberately is not

**Cached:**

| Representation | Scope | Built |
|---|---|---|
| normalized shape tuple | per view | at construction |
| normalized stride tuple | per view | at construction |
| normalized offset, element count, contiguity flag | per view | at construction |
| `int64` shape array, `int64` stride array | per view | **lazily**, on first strided native call |

**Deliberately not cached, with the reason:**

- **A ctypes representation beside the NumPy one.** The C ABI takes
  `np.ctypeslib.ndpointer(int64, C_CONTIGUOUS)`; the NumPy array *is*
  the ctypes-compatible representation. A second one would be redundant
  memory with no call site.
- **Broadcast strides** (`_broadcast_strides`). They are a function of
  *two* operands and the output shape, not of one tensor — they belong
  to the operation, not the object.
- **Reduction write-strides** (`_reduce_out_strides`). A property of
  *this reduction* (axis and keepdims), not of the tensor.
- **Anything global.** No process-wide shape cache, no stride interning,
  no dictionary keyed by shape, no weak-reference machinery, no
  thread-local state, and no unbounded growth. Every cached value is a
  field of the object it describes and dies with it.
- **Eager layout arrays.** See below — laziness is what keeps the
  footprint at zero for the common path.

### 16.3.5 Object footprint

Measured with `sys.getsizeof` over the object and its `__dict__`:

| View rank | Pre-H3 | Post-H3, cache cold | Post-H3, cache warm |
|---|---|---|---|
| 0 `()` | 344 B | **344 B** | 616 B |
| 1 `(1,)` | 336 B | **336 B** | 624 B |
| 2 `(8, 16)` | 320 B | **320 B** | 624 B |
| 3 `(4, 5, 6)` | 312 B | **312 B** | 672 B |
| 4 `(4, 1, 6, 6)` | 304 B | **304 B** | 648 B |

The cold footprint is **byte-identical**: the extra `_layout_cache =
None` slot fits in the instance dictionary's existing capacity. A warm
view costs **+328 B** (a 56 B tuple plus two ~136 B `int64` arrays,
24 B of which is data at rank 3).

The laziness is what makes that acceptable, and it is measurable: in one
complete MLP training step, **134 views are created and only 5 ever
populate the cache — 1,560 bytes in total.** The contiguous fast-path
kernels take a flat element count and an offset rather than shape/stride
arrays, so the dominant training path never builds them at all.

### 16.3.6 Validation: moved, retained, and never weakened

**Nothing was removed.** Every rejection the pre-H3 path performed still
happens, with the same exception type and the same message. What moved
is *how many times the same fact is established*, never *whether* it is.

| Check | Where it lives now |
|---|---|
| shape type / positivity | `_as_shape`, once per externally supplied shape |
| stride element type | `_as_int_tuple`, once |
| stride length vs. shape | `_normalized_layout`, once |
| offset type | `_as_offset`, once |
| storage is open | `_bind`, **both** constructors, every view |
| reachable-offset bounds | `_bind`, **both** constructors, every view |
| `axis`, `keepdims` | unchanged — still validated at every operation |
| `dim`/`start`/`length` for `narrow` | unchanged, plus explicit `int()` normalization |
| reshape element-count match | unchanged |
| transpose permutation | unchanged |
| dtype / device match | unchanged |
| closed-handle checks | unchanged |

The **ordering** of the three rejection points — shape, then strides,
then offset — is preserved exactly, and is asserted directly by feeding
the view constructor metadata with two and three simultaneous faults and
requiring the first to be reported.

### 16.3.7 Closed-object behavior

Unchanged, and pinned. Descriptive metadata (`shape`, `strides`,
`offset`, `ndim`, `numel`, `contiguous`, `dtype`, `device`) stays
readable after `close()` — the documented contract since v0.9 — and
every operation needing a live handle still raises `RuntimeError`.
H3 did **not** make any previously invalid operation succeed: a cached
layout array holds copied integers, not a handle, so it remains readable
after close *and* confers no ability to touch released memory. Closing an
owner leaves a borrowing view's layout readable and its data operations
failing, exactly as before.

### 16.3.8 Ownership and lifetime

- Repeated create/use/close cycles over cores, transposed views,
  narrowed views, reshapes, materializations, copies, reductions, and
  additions return live native storage **exactly** to baseline.
- 50 repeated `to_numpy()` and 50 repeated `contiguous_copy()` calls on
  one tensor allocate no additional storage — the memoized arrays are
  Python objects and cause no native allocation.
- A cached array's `base` is `None` and its referents contain no
  `NativeStorage`, `NativeTensorCore`, or `NativeTensorView`, so it
  cannot keep native memory alive.
- **No reference cycle is introduced.** The cache points at two NumPy
  arrays; neither points back at the view or the core.
- Failure atomicity is re-proved on the rewritten paths: a failure in
  `_bind` releases the freshly allocated storage on all four allocating
  constructors, a failure while *building* the layout arrays leaves the
  cache unpopulated and the tensor fully usable, and a failed native
  call still releases its destination for `tf_core_sum`,
  `tf_core_contiguous_copy`, `tf_core_multiply`, `tf_core_relu_backward`,
  and `tf_core_narrow_backward`.

### 16.3.9 Numerical parity

H3 changed no arithmetic, so parity is exact rather than tolerance-based
wherever the underlying operation is deterministic:

- Every rewritten metadata path — materialization, contiguous copy,
  strided `relu`, transposes, narrows, reshapes, broadcasting — matches
  NumPy element-for-element.
- **H2's contract is preserved.** Matmul results are compared as raw
  IEEE-754 bit patterns between the row-sweep path and the retained
  generic path across six shapes, and are bit-identical on finite data,
  which is what parts 1–3 of §16.2.3 require. Part 4 (NaN payload bits)
  is untouched and remains outside the contract: H3 changed no kernel.
- A six-step deterministic MLP training run reproduces its loss sequence
  and every final parameter **exactly** across repeated runs.
- The full existing suite — including every deterministic-training and
  exact-checkpoint-resume proof in the project — passes unchanged.

### 16.3.10 Measured results

All figures are medians on one machine, reported with the method used.
The pre/post comparison uses a **retained pre-H3 copy of the package
loaded from a separate path**, with the two sides run in **alternating
subprocesses** so machine drift affects both equally, and takes the
best-of-rounds median per side.

**Metadata microbenchmarks** (20,000 repetitions, both implementations
in the *same* process so the comparison is immune to drift):

| Operation | Pre-H3 | Post-H3 | Speedup |
|---|---|---|---|
| `shape_info((8, 16))` | 3.20 µs | 1.00 µs | **3.20×** |
| `shape_info((4, 1, 6, 6))` | 4.90 µs | 1.10 µs | **4.45×** |
| `shape_info((3, 2), strides=(1, 3))` | 2.80 µs | 1.10 µs | **2.55×** |
| `NativeTensorView(storage, (4, 5, 6))` | 4.80 µs | 1.50 µs | **3.20×** |
| `_layout_arrays()`, warm | 1.00 µs | 0.10 µs | **10×** |

**Call counts** after H3, same measurement as §16.3.1:

| Workload | `_as_int_tuple` | `shape_info` | `row_major_strides` | `numel` |
|---|---|---|---|---|
| one `(1, 1)` `add` | 13 → **3** | 3 → **0** | 6 → **0** | 4 → **0** |
| `NativeLinear` forward | 24 → **5** | 5 → **0** | 10 → **0** | 7 → **0** |
| `NativeLayerNorm` forward | 87 → **14** | 14 → **0** | 30 → **0** | 25 → **0** |
| `NativeBatchNorm1d` forward | 195 → **34** | 34 → **0** | 70 → **0** | 65 → **0** |
| one `NativeAdam` step | 604 → **108** | 108 → **0** | 216 → **0** | 216 → **0** |
| one MLP training step | 815 → **149** | 148 → **0** | 296 → **0** | 287 → **0** |
| one CNN training step | 815 → **150** | 147 → **0** | 296 → **0** | 287 → **0** |

The zeros are not a removal of work — the `_checked` primitives still
compute the strides and the element count. They are the removal of
*redundant validation*: no validating call remains on a path whose input
this module already validated.

**Operation benchmarks** (300+ repetitions per case, two interleaved
rounds):

| Case | Pre-H3 | Post-H3 | Speedup |
|---|---|---|---|
| `NativeTensorCore.zeros((1, 1))` + close | 7.90 µs | 3.70 µs | **2.14×** |
| `zeros((4, 1, 6, 6))` + close | 8.60 µs | 4.20 µs | **2.05×** |
| `reshape` | 6.40 µs | 2.10 µs | **3.05×** |
| transpose + narrow chain | 15.80 µs | 6.50 µs | **2.43×** |
| `add`, `(1, 1)` | 10.00 µs | 6.40 µs | 1.56× |
| `add`, `(8, 8)` | 9.60 µs | 6.70 µs | 1.43× |
| `matmul`, 8³ | 10.40 µs | 7.80 µs | 1.33× |
| `matmul`, 32³ | 14.60 µs | 12.40 µs | 1.18× |
| `sum(axis=0)`, 16² | 22.30 µs | 15.40 µs | 1.45× |
| `add`, strided 64² | 25.00 µs | 20.40 µs | 1.23× |
| contiguous copy of a 32² transpose | 15.40 µs | 13.00 µs | 1.18× |
| `to_numpy()` of a 32² transpose | 12.00 µs | 10.10 µs | 1.19× |
| `NativeLinear` forward, no graph | 40.10 µs | 35.10 µs | 1.14× |
| `NativeLinear` forward, with graph | 40.10 µs | 34.70 µs | 1.16× |
| `NativeLinear` forward + backward | 240.50 µs | 177.80 µs | 1.35× |
| `NativeLayerNorm` forward | 221.00 µs | 180.90 µs | 1.22× |
| `NativeBatchNorm1d` training forward | 529.60 µs | 387.50 µs | 1.37× |
| `NativeBatchNorm1d` eval forward | 232.30 µs | 171.60 µs | 1.35× |
| `NativeAdam.step()`, small MLP | 1480 µs | 1046 µs | **1.42×** |
| `NativeAdam.step()`, (128, 128) | 1508 µs | 1326 µs | 1.14× |
| **MLP training step** | 1983 µs | 1389 µs | **1.43×** |
| **CNN training step** | 1829 µs | 1421 µs | **1.29×** |
| `state_dict()` + `load_state_dict()` | 196.30 µs | 134.90 µs | 1.46× |

**Harness cases** (`benchmark_native_cpu_performance.py`, 201
repetitions, two interleaved rounds):

| Case | Pre-H3 | Post-H3 | Speedup |
|---|---|---|---|
| `scalar_dispatch_overhead` / `tensor_core` | 9.00 µs | 5.30 µs | **1.70×** |
| `normalized_training_step` | 6418 µs | 4260 µs | **1.51×** |
| `cnn_classification_training_step` | 2275 µs | 1693 µs | 1.34× |
| `mlp_training_step` | 2334 µs | 1827 µs | 1.28× |
| `adam_step` | 1512 µs | 1242 µs | 1.22× |
| `reduction_contiguous`, 256² | 115.90 µs | 97.40 µs | 1.19× |
| `sgd_step` | 235.70 µs | 211.00 µs | 1.12× |

**The per-call cost, decomposed.** The harness's two new `native_only`
cases finally split B3's single ~20 µs figure (smoke configuration):

| Component | Median |
|---|---|
| `metadata_preparation` — Python normalization + view construction | 1.70 µs |
| `ctypes_boundary` — one prepared kernel call, nothing else | 1.10 µs |
| `scalar_dispatch_overhead` — the whole `(1, 1)` Core `add` | 6.20 µs |

### 16.3.11 Negative, neutral, and noise results

Reported because they are as much of the result as the speedups are.

- **Large kernel-bound cases show no measurable change, in either
  direction.** 384³, 512³, and 128³ matmul, 256² elementwise, and 128²
  full reduction all sit inside their own run-to-run spread. For 512³
  matmul, six interleaved rounds gave pre-H3 medians of 21.2–23.6 ms and
  post-H3 medians of 21.3–22.9 ms — overlapping, with post-H3 slightly
  lower on average. **Acceptance criterion 9 (no material large-matmul
  regression) is met**, and the mechanism is clear: the C++ is byte-
  identical and the Python does strictly less, so there is no path by
  which such a case could get slower.
- **An intermediate measurement that looked like a 35 % regression was
  noise, and is recorded as a methodology finding.** At the harness's
  *default* 11 repetitions, `reduction_contiguous` at 256² appeared to
  go 95 µs → 148 µs. Two things exposed it: the result was internally
  impossible (the thin `tensor_core` layer "slower" than the
  `native_tensor` layer wrapping it), and at 201 repetitions the same
  case measured **1.19× faster**. A dedicated 400-repetition,
  three-round interleaved run confirmed it. **The harness's default
  statistics are not sufficient to support a ±20 % claim on a ~100 µs
  case**; every H3 number above therefore comes from a high-repetition
  run, and no default-repetition figure is quoted as evidence.
- **The layout-array cache is the weakest of the three changes, and it
  was kept on measured merit rather than on principle.** Isolated by
  disabling only that cache on an otherwise-H3 build, it saves
  0.6–1.5 µs per *strided* operation on small tensors (`to_numpy`
  12.1 → 11.1 µs, strided `add` 18.3 → 16.8 µs, strided `relu`
  14.4 → 13.2 µs) and is **lost in the noise on large tensors**
  (256² `to_numpy` 92.6 vs 88.5 µs). It contributes essentially nothing
  to a contiguous training step, which never builds it. It was kept
  because it is lazy, bounded, and free when unused — and a deliberately
  **cold-cache** measurement (a fresh view every iteration, so the
  memoization can only cost) was still equal to or faster than pre-H3,
  so it has no adverse case.
- **`NativeTensor`'s wrapper cost remains negligible**, confirming B8:
  the wrapper and graph layers move with the Core layer rather than
  independently.
- **Workloads still dominated by something else.** A 128×128 Adam step
  improves only 1.14× against the small MLP's 1.42×, because at that
  size the arithmetic, not the call count, is the cost — which is
  exactly B4's finding and remains **H4's** subject, not H3's.

### 16.3.12 What H3 did not do

- No C++, no CTest, no C ABI symbol, no ctypes declaration, no kernel.
  **52 exported `tf_*` symbols, unchanged.**
- No new `NativeTensorCore` method, no autograd operation, no module, no
  loss, no metric, no optimizer, no export.
- No registry moved: `UNSUPPORTED` is still
  `("float32", "cuda", "amp")`, `SUPPORTED_DTYPES` still `("float64",)`,
  `SUPPORTED_DEVICES` still `("cpu",)`.
- No checkpoint field and no checkpoint version: still
  `tensorforge.native_checkpoint` version **2**, supporting **(1, 2)**.
- **No public API of any kind**: no cache control, no cache statistic,
  no reset, no profiling counter, no dispatch selector, no
  environment variable, no `validated=` flag. The suite asserts the
  absence of all of these by name against the real modules and classes,
  and asserts that `cpp.py` reads no environment variable at all.
- No H4+ work: no optimizer restructuring, no reduction fast path, no
  fusion, no Conv2d change, no SIMD, no threading, no BLAS, no memory
  pool, and no scratch workspace. A scope test asserts the absence of
  each by name.

### 16.3.13 Instrumentation policy

Every measurement above was taken with **test-local or benchmark-local**
instrumentation: monkeypatched wrappers around the real helpers,
subprocess A/B runs against a retained copy of the pre-H3 package, and
two new `native_only` harness cases. **No production counter, no
environment-variable profiler, no C ABI counter, and no installed
tracing mode exists**, which is what §3.4 required of any milestone
answering its questions.

### 16.3.14 Validation

| Check | Result |
|---|---|
| Full `uv run pytest` | **5,295 passed**, 0 failed, 0 skipped |
| Windows Release build (MSVC 19.44, CMake 4.4.0, out of source) | zero project warnings |
| Release CTests | **13/13** |
| Native backend smoke | passed |
| Exported `tf_*` symbols | **52** |
| All 14 examples | pass, including every exact-resume proof |
| All 6 benchmark smokes | pass |
| Harness `--smoke`, `--json`, `--workload`, `--profile` | pass |
| New H3 suite against the **pre-H3** implementation | **18 of 53 fail** — the tests are not vacuous |
| New H3 suite: the other 35 | pass on **both** — preservation is real |

The pre-H3 split is the load-bearing evidence that the suite tests
something: the 18 failures are the architecture tests (single
normalization, the private constructor, the memoized read-only arrays,
lazy population, no-cycle, failure cleanup), and the 35 that pass on both
sides are the preservation tests (every rejection, every message, the
rejection order, numerical parity, closed-object behavior, the public
surface, the registries, and the symbol count).

**The collected-case count reconciles exactly.** The suite went from
5,239 to 5,295 — **+56 collected cases** — while H3 wrote **54 new test
functions**. The difference is parametrization, not hidden tests:

| Source | Test functions | Collected cases |
|---|---|---|
| `tests/test_native_metadata_dispatch.py` (new file) | 53 | 53 |
| `test_the_size_independent_cases_are_exactly_the_declared_ones` | 1 | 1 |
| `test_each_case_returns_live_storage_to_its_baseline`, which is `@pytest.mark.parametrize("case", list(EXPECTED_CASES))` and grew with `EXPECTED_CASES` 24 → 26 | 0 | **2** |
| **Total** | **54** | **56** |

The two extra collected cases are the existing per-case live-storage
guard picking up `metadata_preparation` and `ctypes_boundary`
automatically — which is the guard working as designed: every harness
case must return live native storage to its baseline, including the two
H3 added.

### 16.3.15 Sanitizer validation

Run because incorrect cached metadata could in principle reach a kernel
as an out-of-bounds access, an invalid offset or stride, or a
layout-array lifetime fault — none of which the Python suite alone can
rule out.

**Build:** Clang 18.1.3, CMake 3.28.3, WSL2 Ubuntu 24.04.4,
`-DCMAKE_BUILD_TYPE=Debug -DTF_SANITIZE=address,undefined
-DTF_BUILD_TESTS=ON`, built out of source with **zero project compiler
warnings**. Instrumentation **proved rather than assumed**: `nm -D`
reports **22 `__asan*`** and **14 `__ubsan*`** dynamic symbols beside
the **52** exported `tf_*` symbols, and the library **refuses to load
without the sanitizer runtime** (`is_available()` is `False` without
`LD_PRELOAD` and `True` with it). No metadata internal is exported:
`nm -D` finds zero `normalized_layout` / `layout_cache` / `shape_cache`
symbols.

| Check | Result |
|---|---|
| Sanitized native CTests (`detect_leaks=1`) | **13/13** |
| H3 focused suites (metadata/dispatch, view, core, tensor, storage, shape metadata, matmul dispatch, storage allocation, checkpoint v2, harness contract, backends) | **747 passed, 0 skipped** |
| View-consuming suites (autograd, linear, flatten, conv2d, maxpool2d, layernorm, batchnorm 1d/2d, adam, sgd, optimizer math, buffers) | **926 passed** |
| Deterministic-training and exact-resume suites | **321 passed** |
| H3 metadata/view stress program | all checks passed |
| Ownership suites with the **cyclic collector disabled** | **83 passed** |
| Harness under the sanitized library | 26 cases, every gate `passed` |
| **ASan diagnostics** | **0** |
| **UBSan diagnostics** | **0** |

**The stress program** drives every layout the metadata paths can
produce, repeatedly, so each cached shape/stride array actually crosses
into a kernel: row-major contiguous, transposed, narrowed, nonzero
offsets, positive non-unit strides, views of views (three deep), chained
transpose-then-narrow-then-transpose, reshapes, scalar and one-element
shapes, rejected zero-size shapes, both H2 matmul paths including a
strided *left* operand, repeated `_layout_arrays()` reuse, operations
after close, **both** parent/view close orders, failure during
layout-array construction, `_bind` failure on all four allocating
constructors, and a forced failure in each of five kernels.

**Cached-array lifetime.** Phase 5 of the stress takes the cached
arrays, deletes every caller-side reference so that only the view's own
cache keeps them alive, then runs ten rounds of `contiguous_copy`,
`to_numpy`, and `sum` on that view. If the cache did not own them for
the duration of each call, ASan would report a heap-use-after-free at
the kernel. It reports nothing, and the arrays still hold the right
values afterwards.

**Parent/view close ordering**, both directions: closing the **owner**
first leaves the view's layout and cached arrays readable (same object
identity, same values) while every data operation raises `RuntimeError`
rather than touching released memory; closing a **view** first leaves the
owner and its siblings fully usable.

**The detector is proved able to fail.** A negative control hands
`tf_storage_materialize` a shape array of length 2 while telling it
`ndim=3` — exactly the shape a wrong cached layout array would take, and
a defect the C ABI *cannot* catch because it receives a pointer and a
rank with no length. ASan catches it as a `heap-buffer-overflow` and
names the frame (`tf_storage_materialize`, `cpp/src/storage.cpp:176`).
So "zero diagnostics on the real paths" is a measurement, not an absence
of instrumentation.

**LeakSanitizer lifecycle.** A create/use/close cycle over the full view
family returns native live storage **exactly to baseline (0 → 0)**. The
remaining process-exit allocations — 724,691 bytes in 642 allocations —
contain **no TensorForge frame at all**: the stacks name only
`python3.13` (5,913 frames), the ASan runtime (261), `libc` (132), and
NumPy's `_multiarray_umath` (22). **No suppression file was added.**

**No dependence on the cyclic collector.** Stress phase 9 runs 50
create/use/close cycles with `gc.disable()` in force and live native
storage still returns exactly to baseline, and the H3 ownership suites
pass with automatic collection off — explicit `close()` is sufficient,
and `gc.collect()` in the tests is defensive rather than load-bearing.

**Not run, and not claimed:** a Windows **Debug** build and its CTests.
H3 changed no C++ file, no CMake input, and no ABI declaration — the
`cpp/` tree is byte-identical — so a second Windows configuration would
re-measure unchanged sources. The Clang sanitizer build above compiles
those same sources in a Debug configuration with assertions on.

---

## 16.4 H4 — optimizer step cost, as shipped

**Status: complete.** H4 reduced the fixed native allocation and call
count of `NativeAdam.step()`, and — where the evidence supported it —
`NativeSGD.step()`. It is a **Python composition change and nothing
else**: no C++, no C ABI symbol, no ctypes declaration, no kernel, no
`NativeTensorCore` method, no autograd operation, no module, no export,
no registry value, no dtype, no device, and no checkpoint format moved.
The library still exports exactly **52** `tf_*` symbols. Every value the
optimizers produce is **bit-identical** to the pre-H4 composition.

### 16.4.1 The pre-H4 architecture, re-measured

B4's counts were re-instrumented on the **current post-H3 code** rather
than taken from H0, by wrapping `NativeStorage.__init__` (the one
constructor every allocation path runs through, zeroed and uninitialized
alike), `NativeTensorCore.full`, `_binary_core_op`, and `_unary_compute`
in a test-local counter. H0's figure was confirmed exactly:

**`NativeAdam.step()`, per parameter, before H4 — 27 native storage
allocations**, fully attributed:

| Group | Count | What |
|---|---|---|
| Scalar coefficients | **8** | `full((), v)` for `beta1`, `1 - beta1`, `beta2`, `1 - beta2`, `1 - beta1**t`, `1 - beta2**t`, `eps`, `lr` |
| Binary compute | 13 | 9 `multiply`, 2 `add`, 1 `subtract`, plus the commit's `add` |
| Unary compute | 4 | `sqrt`, `reciprocal` on the denominator, and **two `reciprocal`s on one-element tensors** |
| Commit copy | 2 | `copy_value_` → `_native_copy` = `zeros` + `add` |

§3.2 said "six one-element broadcast scalars"; the measured number is
**eight**, because `eps` and `lr` are scalars too. Two further
one-element allocations come from the `reciprocal` calls on the
bias-correction coefficients, so **ten of the 27 allocations per
parameter were one-element**.

**`NativeSGD.step()`, per parameter, before H4 — 5 allocations**: one
`full((), lr)`, the `multiply` output, the `subtract` output, and the
commit copy's two.

Dispatch composition per Adam parameter: **8 of the 13 binary operations
took the broadcasting path** (a `()`-shaped operand against a
parameter-shaped one), and 5 took the same-shape contiguous fast path.

### 16.4.2 What was implemented

Three changes, all in `src/tensorforge/experimental/native_adam.py` and
`native_sgd.py`.

**1. The step's scalar coefficients are built once per step, not once
per parameter.** `beta1`, `1 - beta1`, `beta2`, `1 - beta2`, `eps`, and
`lr` are the same value for every parameter in a step, so a private
`_StepConstants` holder builds each on first use, keyed by
`(dtype, device)` — never assuming one dtype exists — and hands the same
read-only core to every later parameter. The two bias-correction
coefficients depend on the per-parameter step counter, so they are cached
per `t`: steady-state training gives every parameter the same `t` and one
pair, while a parameter that skipped earlier steps legitimately gets its
own. NativeSGD does the same for its single `lr` scalar. The holder is
created inside `step()`, allocates nothing until the first entry asks for
a coefficient — so a step with no active parameter allocates nothing at
all — and is released before the commit begins.

**2. The bias-correction reciprocal is evaluated in Python.** Before H4,
`1 - beta ** t` was allocated as a one-element core and the native
`reciprocal` kernel was run on it. H4 computes `1.0 / (1.0 - beta ** t)`
in Python and allocates the result directly, removing one allocation and
one kernel call per coefficient per parameter.

**This is an exact substitution, not a reassociation.** The kernel *is*
`1.0 / x`:

```cpp
double op_reciprocal(double x) { return 1.0 / x; }   // cpp/src/elementwise.cpp
```

A Python `float` and a C++ `double` are the same IEEE-754 binary64
value, `full((), y)` stores `float(y)` exactly, and IEEE-754 requires
division to be correctly rounded — so there is exactly one possible
result and both spellings produce it. Proved, not asserted: a test sweeps
**20,000+ values** — random magnitudes across the full exponent range,
±0, ±∞, the smallest subnormal, the largest finite magnitude, and every
`1 - beta ** t` the optimizer actually forms — and compares the kernel's
output against Python's division on **raw `uint64` bit patterns**, with
zero mismatches.

**3. Temporaries are released at their last use.** Pre-H4 `_stage_entry`
appended every intermediate to a `transients` list and closed the whole
list in one `finally` at the end, so seventeen parameter-sized buffers
were simultaneously live at the peak. H4 closes each intermediate at the
point its last consumer has run and sets its local to `None`, so the
single cleanup `finally` closes only what is genuinely still live and can
never close anything twice. The arithmetic is untouched — a release is
not an operation.

### 16.4.3 The numerical contract

**Bit identity, unconditionally.** Unlike H2, H4 makes no four-part
carve-out: it changes no accumulation order, no operand position, and no
kernel, so every result is bit-identical including NaN payloads.

The **pre-H4 composition is retained in the test suite** as a literal
transcription of the shipped pre-H4 `_stage_entry` body, executed
natively. Every equality below is against real native execution of the
old composition, never a NumPy re-derivation:

- one parameter × 4 shapes (`()`, `(1,)`, `(2, 2)`, `(3, 1, 4)`) × 3 step
  counts × 5 hyperparameter sets (default, small betas, `beta = 0`,
  betas at `0.99999`/`0.9999999` with `eps = 1e-30`, and `lr = 1e10`
  with `eps = 2.5`) — 60 combinations, parameter and both moments;
- a six-step run over four parameters of mixed shapes;
- NativeSGD across four learning rates spanning `1e-9` to `1e12`.

A separate test asserts the **exact operation sequence** a staged entry
issues, so a future change that reorders or fuses operations fails
loudly rather than silently: `multiply, multiply, add, multiply,
multiply, multiply, add, multiply, multiply, sqrt, add, reciprocal,
multiply, multiply, subtract`, plus the commit copy's `add`.

Every committed loss trajectory, every deterministic training example,
and every exact-resume proof in the repository therefore reproduces
unchanged — which the full suite confirms.

### 16.4.4 What did *not* change

The two-phase contract is exactly what it was, and this is the point
H4 was most constrained by. Stated as the H4 entry stated it before the
milestone shipped, and still true after it: **every validation and every
staged computation completes before the first parameter mutates; a
staging failure commits nothing and leaves gradients retryable.** In
detail:

- **Stage mutates nothing.** Validation order is unchanged and still four
  complete passes — optimizer open, every parameter open, every m/v state
  valid, then every active gradient — each finishing before the next
  begins. No validation moved behind a mutation.
- **One `copy_value_` and exactly one version increment per updated
  parameter.** The commit path is untouched.
- **Gradients are read and never written**, by identity, value, and
  storage identity.
- **The staging seam keeps its name and its meaning.** `_stage_entry`
  gained an optional fourth argument, the shared holder; called with
  three arguments it builds and releases its own scalars, which is
  exactly the pre-H4 behavior, and a test asserts both spellings produce
  the same bits.
- **The documented per-entry commit boundary is unchanged.** H4 does not
  claim the commit is infallible; a test injects a failure into
  `copy_value_` and asserts that exactly the entries committed before it
  stand, that no staged core leaks, and that the exception reaches the
  caller — the honest window the pre-H4 docstring already recorded.

### 16.4.5 Measured results

Two independent measurements, both with correctness gated before timing.

**(a) A controlled A/B**, alternating `pre` and `post` **subprocess**
rounds so system drift affects both arms equally, 366 samples per case
(150 for the largest), medians:

| Case | pre-H4 | post-H4 | ratio |
|---|---|---|---|
| `NativeAdam.step()`, one (128, 128) | 1041 µs | 660 µs | **1.58×** |
| `NativeAdam.step()`, one (256, 256) | 3659 µs | 2382 µs | **1.54×** |
| `NativeAdam.step()`, MLP profile (4 params, largest 256²) | 5461 µs | 3690 µs | **1.48×** |
| `NativeAdam.step()`, non-default betas, MLP full | 1246 µs | 1020 µs | 1.22× |
| `NativeAdam.step()`, MLP full (4 params, largest 32×64) | 1221 µs | 1008 µs | 1.21× |
| `NativeAdam.step()`, **first** step, MLP full | 1219 µs | 1060 µs | 1.15× |
| `NativeAdam.step()`, mixed gradients (2 of 4 frozen) | 636 µs | 566 µs | 1.12× |
| `NativeAdam.step()`, one scalar parameter | 285 µs | 255 µs | 1.12× |
| `NativeAdam.step()`, one 16-element vector | 274 µs | 252 µs | 1.09× |
| `NativeAdam.step()`, one (512, 512) | 12 326 µs | 12 053 µs | 1.02× — **neutral** |
| MLP training step, 256→256→64 | 7946 µs | 6480 µs | 1.23× |
| MLP training step, 32→64→8 | 1837 µs | 1597 µs | 1.15× |
| Normalized training step | 4330 µs | 3846 µs | 1.13× |
| CNN training step | 1675 µs | 1539 µs | 1.09× |
| Dropout training step | 1430 µs | 1446 µs | 0.99× — **neutral** |
| `NativeSGD.step()`, MLP full | 206 µs | 192 µs | 1.07× |
| `NativeSGD.step()`, one (128, 128) | 98 µs | 101 µs | 0.97× — **neutral** |
| `NativeSGD.step()`, MLP profile | 529 µs | 602 µs | 0.88× — **noise, see below** |
| **`matmul` 256², control (code unchanged)** | 2501 µs | 2587 µs | **0.97×** |

**(b) The shipped harness's own cases**, `pre`/`post`, 81 samples,
6 alternating rounds:

| Harness row | pre-H4 | post-H4 | ratio | vs stable, pre → post |
|---|---|---|---|---|
| `adam_step` / `optimizer_step` | 1303 µs | 1040 µs | **1.25×** | 23.8× → **19.7×** |
| `sgd_step` / `optimizer_step` | 194 µs | 190 µs | 1.03× | 31.5× → 30.9× |
| `cnn_classification_training_step` | 1706 µs | 1596 µs | 1.07× | — |
| `normalized_training_step` | 5469 µs | 5121 µs | 1.07× | 6.70× → 5.48× |
| `mlp_training_step` | 1670 µs | 1669 µs | 1.00× | — |
| `matmul_square_contiguous`, all 7 layers (control) | — | — | 0.84×–1.26× | — |

### 16.4.6 Negative, neutral, and noise results

Reported as findings, not omitted.

- **This machine's noise floor is wide.** The matmul control case, whose
  compiled code and Python path H4 did not touch, varied **0.84×–1.26×**
  across its seven layers between arms. Any single-case reading inside
  that band is not a result. That is why the Adam figures are quoted from
  the 366-sample alternating-subprocess A/B rather than from one harness
  run, and why the `sgd_mlp_profile` 0.88× row is reported as **noise**:
  a focused re-measurement (three independent pre/post pairs, 200 samples
  each) gave pre medians 897/881/850 µs against post 884/894/900 µs, with
  post *minima* lower in every pair. SGD's honest result is
  **neutral-to-slightly-positive**, not a regression and not a win.
- **Large parameters are neutral.** At (512, 512) the step is
  memory-bandwidth-bound: the arithmetic dominates, ten fewer one-element
  allocations are invisible, and the measured 1.02× is inside the spread.
  H4 helps where the *count* is the cost, which is small and medium
  parameters and multi-parameter models — exactly what B4 predicted.
- **The Dropout training step is neutral** (0.99×): its optimizer is one
  small MLP's worth of parameters and the stochastic forward dominates.
- **H2 is intact.** Large-matmul rows moved only within the control
  band, and no matmul code path was touched.

### 16.4.7 Memory

Measured by tracking every `NativeStorage` construction and destruction
during one steady-state step; **fully deterministic and reproducible
across runs**:

| Case | peak live transient storages | peak live transient bytes | allocations |
|---|---|---|---|
| Adam, one (128, 128) | 25 → **13** | 1 966 160 → **655 424** (**3.00×**) | 27 → 25 |
| Adam, one (512, 512) | 25 → **13** | 31 457 360 → **10 485 824** (**3.00×**) | 27 → 25 |
| Adam, MLP full (4 params) | 34 → **22** | 245 840 → **95 936** (**2.56×**) | 108 → **76** |
| Adam, MLP profile (4 params) | 34 → **22** | 7 864 400 → **3 022 336** (**2.60×**) | 108 → **76** |
| SGD, MLP profile (4 params) | 9 → 9 | unchanged | 20 → **17** |

The wall-clock improvement therefore does **not** hide a memory
increase: peak transient memory during an Adam step fell by 2.6–3.0×
while the time fell. Per-parameter marginal allocation cost went from 27
to **17**, with at most **eight** shared scalars for the whole step
instead of eight per parameter — so a four-parameter model allocates
**76 instead of 108** (−29.6 %), and the gap widens with the parameter
count.

### 16.4.8 Optimizations measured and rejected

**1. Materializing the scalar coefficients to the operand's shape**
(so the multiply takes the flat contiguous kernel instead of the
broadcasting odometer). Measured per-operation, and it is genuinely
faster below roughly 32 K elements and genuinely slower above:

| Operand | broadcast `()` scalar | materialized + multiply | same-shape contiguous (floor) |
|---|---|---|---|
| 16 | 15.3 µs | 11.2 µs | 6.8 µs |
| 1 024 | 22.0 µs | 12.0 µs | 7.2 µs |
| 16 384 | 58.3 µs | **23.6 µs** | 13.0 µs |
| 65 536 | 148.4 µs | **313.5 µs** | 32.0 µs |
| 262 144 | 816.4 µs | **942.8 µs** | 543.0 µs |

**Rejected**, for four reasons. The crossover tracks this machine's L2
cache, not layout metadata, so the predicate would be a tuned magnitude
threshold rather than the §8.1 kind of dispatch. It **regresses** the
harness's own `profile` configuration, whose largest parameter is 65 536
elements. It adds a parameter-sized buffer per scalar operation, against
§16.4.7's requirement that time must not be bought with peak memory. And
a whole-optimizer A/B confirmed it: the materializing variant measured
0.87×–1.36× depending on size, against the shipped variant's consistent
improvement.

**2. Same-shape stride-0 views for scalar operands** (dispatch path B
instead of path C: the *same* kernel with the *same* arguments, proved by
construction, but skipping the Python broadcast computation).
**Rejected on measurement**: path C builds three NumPy `int64` arrays per
call; path B builds four, because both the fresh output-shaped operand
and the freshly constructed stride-0 view have cold layout caches and
each view is used once. The whole-optimizer A/B put it at or below the
shipped variant on every workload.

**3. Adopting the staged parameter core instead of copying it.**
`copy_value_` runs `_native_copy` = `zeros` + `add`, which is two
allocations — one of them **zero-initialized**, a full extra write — and
one full-width kernel pass, purely to duplicate a core the optimizer just
built and owns exclusively. `NativeParameter._adopt_value_core` would
remove all three. **Rejected**: §16.4 above and this document's own H4
entry require *one `copy_value_` per updated parameter*, and that is the
project's single sanctioned mutation primitive. Recorded as H5+ scope. **H5 revisited it and kept the rejection**: the one
`copy_value_` per updated parameter is a contract, not an
inefficiency, so H5 made the *staging* cheap instead of removing the
commit (§16.5.2, call site 1).

**4. Making `_native_copy` use `contiguous_copy`.** *(Rejected at H4;
**taken up and shipped at H5** — see §16.5, which resolved the
signed-zero question this entry correctly refused to decide in passing.
The H4 reasoning below is retained as written.)* The same two
allocations again, from the other side. **Rejected as out of scope and
not obviously correct**: `_native_copy` is `zeros.add(core)`, and
`0.0 + (-0.0) = +0.0`, so it *normalizes negative zero* where
`contiguous_copy` would preserve it. That is a real observable difference
in a helper shared by gradient accumulation, state loading, and both
optimizers — a change that needs its own milestone, its own parity
argument, and its own tests, not a drive-by inside an optimizer
milestone. Recorded here so a later milestone starts from the finding
rather than rediscovering it.

**5. A persistent per-optimizer scalar cache**, keyed on the current
hyperparameters so it could never go stale, which would make the
steady-state scalar allocation count **zero**. **Rejected by contract**:
it is exactly the "hidden optimizer-wide scratch tensor whose lifetime
complicates checkpointing" that §10 and the H4 scope forbid — it would
have to be released by `close()`, appear in every live-storage baseline,
and be reasoned about in the checkpoint transaction. The per-step holder
gets most of the benefit with none of the lifetime surface.

**6. Reassociating the update to fold scalars together** — for example
`(m_hat * lr) * inv_denominator` into `m_hat * (lr * inv_denominator)`,
which would remove one broadcast operation per parameter. **Rejected
outright**: floating-point multiplication is not associative, this is a
§7.3 order change, and for an optimizer it would break every committed
exact-resume proof in the project.

### 16.4.9 Validation

**Windows.** A fresh Release build (Visual Studio 17 2022, MSVC) with
zero project compiler, linker, and CMake warnings, **13/13 CTests** in
0.24 s, and the native backend smoke check. The shipped DLL's own PE
export directory lists **52** symbols, all `tf_*`, with **none** matching
"adam", "sgd", "optimizer", or "poison". The full Python suite is
**5 517 passed, 0 failed, 0 skipped** — the 5 295 pre-H4 cases plus 222
H4 cases. The Phase-H harness passes all **26** correctness gates in
`--smoke` and in `--smoke --json` while writing no result file, the JSON
carries its full schema, the focused `--case adam_step` / `--case
sgd_step` runs pass, and the existing quick backend benchmark runs.

**Clang 18.1.3 `-DTF_SANITIZE=address,undefined` in WSL2**, built
out-of-source outside the repository so the active Windows runtime stayed
the Release DLL:

- **Zero** compiler warnings or errors.
- **Instrumentation proved**: `nm -D` shows **22** `__asan*` and **14**
  `__ubsan*` dynamic symbols beside the **52** exported `tf_*`, and
  **zero** symbols matching "adam", "sgd", or "optimizer" — the H4
  architecture is Python composition, and the library says so.
- **13/13 sanitized native CTests** with `detect_leaks=1`.
- **909 sanitized Python tests** across the H4 suite and every
  optimizer-touching suite (Adam, SGD, optimizer math, optimizer state,
  parameter, versioning, state_dict, state transaction, both checkpoint
  suites, storage allocation, metadata dispatch, matmul dispatch, and the
  harness contract tests) — zero ASan and zero UBSan diagnostics.
- **568 sanitized Python tests** across the deterministic-training and
  exact-resume suites (MLP, CNN, classification, normalization, Dropout,
  and Phases C, D, E, F, G including the G6 hardening matrix).
- The harness under the sanitized library: **26 cases, all gates
  passed**.
- The G7 example reproducing its exact stochastic resume with live native
  storage **0 → 0**.
- **A LeakSanitizer optimizer lifecycle**: 25 Adam create/step×4/close
  cycles, 25 SGD cycles, and a **112-position injected-failure matrix**
  (seven Core seams × four call indices × `RuntimeError`, `MemoryError`,
  `KeyboardInterrupt`, and a non-`Exception` class). Every failed step
  was checked **immediately, with no collection**, for storage growth
  against the count taken before it: **zero growth at all 112
  positions**. Live native storage returned **exactly** to its measured
  baseline. The remaining process-exit allocations (764 486 bytes in 680
  allocations) contain **no TensorForge frame** — the leak frames name
  only CPython, the ASan runtime, libc, and NumPy, and none names
  `_tensorforge_cpp.so` — with **no suppression file added**.

**Not run, and not claimed:** a Windows **Debug** build and its CTests.
H4 changed no C++ file, no CMake input, and no ABI declaration — the
`cpp/` tree is byte-identical, which `git diff --stat -- cpp/` confirms —
so a second Windows configuration would recompile unchanged sources. The
Clang sanitizer build above compiles those same sources in a Debug
configuration with assertions on. MemorySanitizer is still unavailable
here and still not claimed.

### 16.4.10 Three tests were re-anchored, and why

Three pre-existing tests injected a failure at the *N*-th
`NativeTensorCore.full` call. H4 legitimately changes how many `full`
calls a step makes and when, so those indices no longer named the
position the tests were written to exercise ("fail after at least one
entry is fully staged"). Each was re-anchored to a **per-parameter**
allocation seam — `NativeTensorCore.multiply` — which expresses that
position directly rather than through a magic count. **Every assertion in
all three is unchanged.** A fourth, `test_native_phase_f.py`'s optimizer
staging-failure boundary, patched `_stage_entry` with a three-argument
signature; it now forwards `*rest`, so it injects at the same place
without pinning the seam's arity. New H4 tests cover the `full` seam
directly, at every position the shared holder builds.

---

## 16.5 H5 — copy and mutation transfer, as shipped

**Status: complete.** H5 replaced the native line's **value-transfer
primitive**. `_native_copy` was `zeros(shape) + core` — two allocations,
a full zero-fill pass, and a full elementwise-addition pass — and is now
the E3.1 native identity gather, `NativeTensorCore.contiguous_copy()`:
one uninitialized allocation (H1) and one pass. Underneath it,
`tf_core_contiguous_copy` gained a second **traversal** — not a second
kernel, and **not a second export**.

No exported C ABI symbol was added: the library still exports exactly
**52** `tf_*` symbols. No public API, capability registry value, dtype,
device, checkpoint field, or checkpoint version moved.

### 16.5.0 The ladder was reordered here

The H0 draft put **reduction execution** in the H5 slot. H4's evidence
moved it: H4 measured the optimizer commit ending in a `copy_value_`
whose staging was `zeros + add`, and found the same composition behind
*every* state snapshot, state load, BatchNorm running-statistics commit,
and gradient materialization in the runtime — ten call sites of one
helper, all of them pure value transfers, none of them wanting
arithmetic. H4 itself listed "give `_native_copy` a `contiguous_copy`
implementation" among its **rejected** alternatives, for a reason it
recorded honestly: it would stop normalizing `-0.0` to `+0.0`, a real
observable change in a helper shared far beyond the optimizer, and H4 was
not the milestone to decide that. H5 is that milestone. Reduction
execution moves to **H6** — which has since shipped (§16.6) — and the
later conditional slots shift with it (§16.7).

### 16.5.1 The pre-H5 architecture

`_native_copy(core)` allocated a zero-filled destination of the source's
shape and added the source into it. That composition predates the E3.1
native gather — when it was written, reading a strided view without a
NumPy round trip meant going through a binary kernel, because no
storage-to-storage identity map existed yet. E3.1 added one and
`_native_copy` was never migrated to it.

The result was a runtime with **two** value-copy spellings that disagreed:

| Path | Spelling | Preserves `-0.0`? |
|---|---|---|
| `NativeParameter(source)` construction | `contiguous_copy()` | yes |
| `NativeTensor.detach()` | `contiguous_copy()` | yes |
| `to_numpy()` / `from_array` round trip | materialize / copy | yes |
| `NativeParameter.copy_value_(source)` | `_native_copy` = `zeros + add` | **no** |

The first and last document the same thing — "an independent owning
contiguous copy of the source's current value" — and did not deliver the
same thing. That is §7's Outcome C: the behavior was accidental and
inconsistent, not contracted. H5 resolves it toward the majority and the
stronger guarantee.

### 16.5.2 The complete call-site inventory

Every `_native_copy` call site in the runtime, classified. All ten are
**pure value transfers**: an independent contiguous materialization of
some tensor's current value, for any layout. None wants arithmetic; none
depended on the normalization.

| # | Call site | Source layout | Destination | Alias/overlap possible | Decision |
|---|---|---|---|---|---|
| 1 | `NativeParameter.copy_value_` staging | any (caller-supplied) | fresh core, then adopted | **yes** — self-copy, own-storage view, own transpose | **enable** |
| 2 | `NativeModule.state_dict()` snapshot | owned contiguous | fresh caller-owned tensor | no | **enable** |
| 3 | `NativeModule.load_state_dict()` staging | any (caller-supplied) | fresh core, then adopted | yes | **enable** |
| 4 | `NativeAdam.state_dict()` moment snapshot | owned contiguous | fresh caller-owned tensor | no | **enable** |
| 5 | `NativeAdam.load_state_dict()` staging | any (caller-supplied) | fresh optimizer-owned tensor | yes | **enable** |
| 6 | `_NativeBatchNorm` `running_mean` commit | graph-free scratch, contiguous | fresh core, then adopted | no | **enable** |
| 7 | `_NativeBatchNorm` `running_var` commit | graph-free scratch, contiguous | fresh core, then adopted | no | **enable** |
| 8 | `reshape` backward materialization | borrowing reshape view | fresh grad contribution | no | **enable** |
| 9 | `transpose` backward materialization | borrowing transposed view | fresh grad contribution | no | **enable** |
| 10 | `_unbroadcast` rank-fix materialization | reduction result / reshape view | fresh grad contribution | no | **enable** |

One composition that *looks* like an eleventh is **rejected**:

| Call site | Why rejected |
|---|---|
| `_broadcast_back`'s `zeros(x_shape) + upstream` | This is **not a copy**. It is a genuine broadcast expansion — the upstream carries a smaller (keepdims) shape and the addition against a zero-stride operand is what expands it. `contiguous_copy` cannot express broadcasting at all, and the operand shapes differ, so the substitution is not available even in principle. It stays arithmetic, and therefore keeps arithmetic's IEEE behavior. |

Two further rejections, restated from H1 because they are the same
family of question and the answer has not changed:

| Call site | Why rejected |
|---|---|
| `sum` / `mean` output | Accumulates into its destination; the zero is the additive identity the reduction starts from, not a redundant write. |
| `narrow_backward` output | Writes only the narrowed region; the untouched zeros **are** the gradient. |

### 16.5.3 The signed-zero and NaN findings

Measured, not assumed, over a fixed 18-pattern IEEE-754 sweep asserted
identically in `tests/test_native_copy_transfer.py` and
`cpp/tests/test_contiguous_copy.cpp`. **Exactly three** of the eighteen
patterns behaved differently under the two spellings:

| Pattern | `zeros + add` (pre-H5) | gather (H5) |
|---|---|---|
| `-0.0` | `+0.0` — **normalized** | `-0.0` |
| signaling NaN | quiet NaN, payload kept — **quieted** | signaling NaN |
| negative signaling NaN | negative quiet NaN, payload kept — **quieted** | negative signaling NaN |
| everything else — `±0`, `±1`, `±inf`, quiet NaNs of either sign and **any payload**, the smallest subnormal, the largest subnormal, the smallest normal, the largest finite magnitudes | identical | identical |

Both differences follow from IEEE-754 rather than from anything
TensorForge chose: `0.0 + (-0.0)` is `+0.0` under round-to-nearest, and
an arithmetic operation on a signaling NaN raises invalid and delivers
the quieted NaN. NaN **payloads** were never at risk here — with one NaN
operand and one zero, x86-64's `ADDSD` returns that operand's NaN, so
the pre-H5 path already preserved every payload it was given. **H2's
matmul NaN-payload carve-out does not generalize to copies and must not
be read as if it did**: it exists because two NaN operands meet in an
accumulation and the FPU picks one. A copy performs no arithmetic, so
nothing picks.

**The contract H5 states, and the reason it is the narrow one:**

> A **value transfer** — `_native_copy` and every call site above —
> reproduces its source's IEEE-754 bits exactly. An **operation** —
> including `zeros + x`, matmul, and every other kernel — follows IEEE
> arithmetic and its results are values, not copies.

That is the narrowest rule that makes the four value-copy paths in
§16.5.1 agree, and it changes no operation's arithmetic anywhere.

**Backward compatibility, checked rather than asserted.** `-0.0 == +0.0`
is true, `np.array_equal` treats them as equal, and no TensorForge
operation branches on the sign of a zero except `reciprocal` and `sqrt`,
neither of which the optimizers apply to a possibly-zero quantity
(Adam's `reciprocal` takes `sqrt(v_hat) + eps ≥ eps > 0`). Signaling
NaNs cannot be *produced* by any TensorForge operation at all — every
arithmetic path quiets them — so one can only enter through
`from_array` with a hand-built bit pattern. The whole 5,517-test pre-H5
suite passes unchanged apart from the guardrails that pinned the old
composition by name (§16.5.7), including every committed loss
trajectory, every checkpoint round trip, and every exact-resume proof.

### 16.5.4 The traversal, and why it needed C++

Swapping the composition alone would have **regressed** the most common
case. `zeros.add(core)` on a contiguous source takes the flat
`tf_core_add_contiguous` pointer loop, while `tf_core_contiguous_copy`
always walked the generic odometer — the only unary export without the
contiguous fast path every other one has. Measured before the fix, a
naive swap cost **0.48×** at 16,384 elements.

So `tf_core_contiguous_copy` now picks its traversal from the layout
metadata it already receives, exactly as H2's matmul picks its kernel:

- `tf::copy_prefers_contiguous(shape, strides, ndim)` — declared in
  `cpp/include/tf_copy_internal.h`, hidden-visibility C++ in
  `namespace tf`, **not** part of the ABI. Total, pure, allocation-free,
  and a function of metadata alone: never of a pointer value, an
  alignment, a clock, an environment variable, or a CPU-feature probe.
- The test is exact equality against the row-major strides implied by
  the shape, computed right to left — **the same definition
  `NativeTensorView` uses in `backends/cpp.py`**, so the two layers
  agree by construction. `ndim == 0` is contiguous. The offset is
  deliberately not consulted: it only moves where the flat loop starts,
  and the wrapper has already bounds-checked the whole reachable span.
- True → `core_unary_contiguous` with `op_identity`. False → the
  retained `core_unary` odometer, unchanged, which is the only traversal
  that can address a transposed, narrowed, or negatively strided view at
  all. A false answer is a fallback, never an error.

**No numerical carve-out is needed, and this is the difference from H2.**
Both traversals evaluate `dst[out] = src[pos]` over the same logical
elements in the same row-major destination order and differ only in how
`pos` is computed. The operation is the identity map: no arithmetic is
performed on the value, so there is no accumulation order to preserve,
no operand position for an FPU to select a NaN from, and nothing that
can quiet a signaling NaN or normalize a signed zero. The two paths are
bit-identical for every representable double **by construction**, and
that is proved directly at the C++ level by `test_contiguous_copy.cpp`.

There is no copy-mode selector, overlap-mode flag, traversal tracer,
environment variable, or public dispatch control, and none may be added.

### 16.5.5 H1, aliasing, identity, and atomicity

**H1 full-write.** The gather's destination is allocated uninitialized,
and both traversals write every one of the destination's `numel`
elements exactly once — the flat loop directly, the odometer once per
logical element. Proved by deterministic poison injected **exclusively
by test infrastructure, around the allocator**, over every layout family
and with two patterns (a quiet NaN and a large finite value), plus a
negative control that shows the detector really can fail. No
poison-control mechanism exists in the shipped runtime.

**Aliasing and overlap.** Nothing became less safe, because nothing
became in-place. Every call site still **stages** an independent
materialization and only then adopts it, so source and destination
storage are never written through at the same time. The overlapping
arrangements the runtime can actually construct — `copy_value_(self)`,
a source that is a view of the destination's own storage, a square
parameter's own transpose (where every destination element is read from
a *different* source element), sibling views, and duplicate parameters
across optimizers — are each tested and each produce the correct result.
No `memcpy` is used anywhere, and no staged transaction was converted
into an unsafe in-place transfer.

**Identity, storage, and versions.** Unchanged in every particular.
`copy_value_` still replaces the parameter's owned storage under a
preserved Python identity, keeps `grad` by identity and value, keeps
`requires_grad` and every registration, and increments the value version
by exactly one. `load_state_dict` still runs through the F1 transaction,
still preserves buffer and parameter identity, and still moves each
matched parameter's version exactly once. Loading optimizer or generator
state still moves no parameter version.

**Gradients.** H5 changed how a gradient *contribution* is materialized,
never the accumulation rule: the first contribution is adopted by
identity and later ones are summed with the native `add` kernel.
Additive accumulation was not turned into assignment anywhere.

**Failure atomicity.** The helper's signature and module-level position
are unchanged, so every existing failure-injection seam still works, and
the H4 seam that named the commit copy by its first allocation moved
from `("zeros", 1)` to `("contiguous_copy", 1)` — the same instant in
the same transaction, with every assertion identical. A failure at any
stage still leaves value, core, storage identity, gradient, version, and
live native storage exactly as they were.

### 16.5.6 Measured results, reported honestly

Method: alternating pre/post **subprocess** rounds with a retained
pre-H5 composition, medians of per-round medians, correctness gated
before timing. Control cases whose compiled code H5 did not touch bound
this machine's noise at **0.96×–1.05×**. Separately, the C++ traversal
change was isolated by building a **pre-H5 library** and driving both
libraries through identical `ctypes` calls on identical data, with the
outputs proved bit-identical before either was timed.

**The traversal, in isolation (raw kernel, two libraries):**

| Source | pre-H5 | H5 | ratio |
|---|---|---|---|
| contiguous, 1 element | 7.10 µs | 5.50 µs | 1.29× |
| contiguous, 1,024 | 10.40 µs | 6.40 µs | 1.62× |
| contiguous, 16,384 | 41.10 µs | 12.50 µs | 3.29× |
| contiguous, 128×128 | 36.60 µs | 14.80 µs | 2.47× |
| contiguous, 256×256 | 139.40 µs | 25.40 µs | 5.49× |
| contiguous, 512×512 | 562.70 µs | 101.80 µs | 5.53× |
| contiguous, 4-D NCHW (8,16,32,32) | 273.50 µs | 49.50 µs | 5.53× |
| contiguous, offset (axis-0 narrow) | 373.80 µs | 68.40 µs | 5.46× |
| **transposed 512×512** | 616.20 µs | 653.10 µs | **0.94×** |
| **last-axis narrow 512×500** | 536.30 µs | 527.20 µs | **1.02×** |

The last two rows are the design's own control: they take the
*unchanged* odometer, so they must be neutral, and they are.

That table is **one run**. A repeat against a freshly rebuilt Release
library reproduced its shape rather than its digits — 1.04× at one
element, 1.36× at 1,024, 2.99× at 16,384, 2.85× at 128×128, 5.14× at
256×256, 5.22× at 512×512, 5.70× on 4-D NCHW, 5.08× on the offset view,
and **1.01× / 0.97×** on the two odometer controls. The durable
statement is therefore the range, "**2.5×–5.5× on contiguous sources
from 16 K elements up, neutral on strided ones**", and not any single
figure in the table.

This change also speeds up every **pre-existing** `contiguous_copy` caller —
`NativeParameter` construction, `detach()`, `NativeFlatten`, and the
Policy-B copy-then-compute paths — which is a side benefit H5 did not
set out to buy.

**End to end (alternating rounds, whole milestone):**

| Case | ratio | note |
|---|---|---|
| `copy_value_` (512, 512) | 2.14× | |
| `copy_value_` (128, 128) | 1.26× | |
| copy, 512×512 contiguous | 2.17× | |
| copy, 512×512 transposed | 1.65× | odometer both sides; the win is the removed allocation and zero-fill |
| copy, axis-0 narrow (384, 512) | 2.05× | |
| copy, last-axis narrow (512, 500) | 1.79× | |
| optimizer `state_dict()` | 2.40× | |
| optimizer `load_state_dict()` | 1.69× | |
| module `load_state_dict()` | 1.37× | |
| module `state_dict()` | 1.23× | |
| `NativeSGD.step()` (512, 512) | 1.31× | |
| `NativeSGD.step()` (128, 128) | 1.15× | |
| **`NativeAdam.step()`** | **0.98×–1.06×** | **neutral** — one of ~17 buffers; the arithmetic dominates |
| **MLP / normalized / CNN training step** | **0.95×–1.07×** | **neutral**, inside or adjacent to the control band |
| **BatchNorm running update** | **0.98×** | **neutral** |
| **copies below ~16K elements** | **0.93×–1.01×** | **neutral** |

Two methodology findings are published rather than buried.

1. **Low repetition counts lie**, again. At 7 alternating rounds the
   small-copy cases read 0.78×–0.94× and looked like a real regression;
   at 21 rounds the same cases read 0.93×–1.01×, inside the control
   band. This is the same lesson H3 recorded, and no
   default-repetition figure is quoted as H5 evidence.
2. **A 512 KB allocator cliff produces the largest single ratio, and it
   is not a loop-speed result.** Sweeping one contiguous copy by size:

   | Buffer | pre-H5 | H5 | ratio |
   |---|---|---|---|
   | 128 KB | 14.9 µs | 15.3 µs | 0.97× |
   | 256 KB | 22.9 µs | 23.6 µs | 0.97× |
   | 384 KB | 29.4 µs | 31.3 µs | 0.94× |
   | 512 KB | 304.1 µs | 38.4 µs | **7.92×** |
   | 640 KB | 386.3 µs | 36.9 µs | **10.47×** |
   | 768 KB | 444.0 µs | 211.2 µs | 2.10× |
   | 1 MB | 497.9 µs | 238.0 µs | 2.09× |
   | 2 MB | 992.1 µs | 464.8 µs | 2.13× |

   At and above ~512 KB this machine's allocator hands blocks off to a
   more expensive path. The pre-H5 composition makes **two** such
   allocations and zero-fills one, so it crosses that threshold at half
   the size and pays it twice. The 7.9×–10.5× band is a real,
   reproducible property of the two compositions **on this allocator**,
   not a portable claim; the durable statements are the ~2.1× at 1–2 MB
   and the neutrality below 384 KB.

**One cost is recorded as a finding for a later milestone rather than
fixed here.** A `contiguous_copy` call converts two `int64` layout
arrays at the `ctypes` boundary, and `np.ctypeslib.ndpointer.from_param`
costs **~1.1 µs per array, 2.2 µs per call** — measured directly. That
is why the gather does not *also* win below ~16K elements, where it does
strictly less work. Removing it means changing the ctypes argument
declarations of six strided kernels and giving up a boundary check on
arrays the runtime builds itself; that is H3's subject, not H5's, and
§9's "no validation is removed" rule stands.

### 16.5.7 Allocation and memory traffic

Counted by wrapping `NativeStorage.__init__` and `close()` — the one
constructor every allocation path runs through, zeroed and uninitialized
alike.

| Operation | allocations | total bytes | peak live bytes |
|---|---|---|---|
| `NativeAdam.step()` (128, 128) | 25 → **24** | 2,228,288 → **2,097,216** | 655,424 → 655,424 |
| `NativeAdam.step()` (256, 256) | 25 → **24** | 8,912,960 → **8,388,672** | 2,621,504 → 2,621,504 |
| `NativeAdam.step()`, 4-parameter MLP | 76 → **72** | 13,421,632 → **12,632,128** | 3,419,136 → **3,158,016** |
| `NativeSGD.step()` (128, 128) | 5 → **4** | 524,296 → **393,224** | 393,216 → **262,152** |
| `copy_value_` (512, 512) | 2 → **1** | 4,194,304 → **2,097,152** | 4,194,304 → **2,097,152** |
| module `state_dict()`, Linear(128, 256) | 4 → **2** | 528,384 → **264,192** | 524,288 → **264,192** |
| module `load_state_dict()`, Linear(128, 256) | 4 → **2** | 528,384 → **264,192** | 524,288 → **264,192** |
| optimizer `state_dict()`, Linear(128, 256) | 16 → **8** | 2,113,536 → **1,056,768** | 788,480 → **528,384** |

Per parameter, `NativeAdam.step()` went **17 → 16** allocations (H4 took
it 27 → 17), and the zero-fill pass over one whole parameter is gone
from every committed update. **Memory moved with time, never against
it**: no row's peak rose, and the pure-transfer rows halved. Adam's peak
is unchanged because its commit copy is the last thing in the step, when
the staged temporaries have already been released — the saving there is
in total traffic, not in the high-water mark.

### 16.5.8 Guardrails that moved, and why

H5 changed behavior four existing tests pinned by name. Each was updated
to assert the **new** correct behavior with the same rigor; none was
loosened or deleted.

| Guardrail | Change | Why |
|---|---|---|
| `test_the_staged_expression_issues_exactly_the_pre_h4_operations` | Trailing `"add"` removed; a new assertion pins the commit at exactly **one** `contiguous_copy` and **zero** arithmetic kernels | The trailing `add` *was* the value copy spelled as arithmetic. The arithmetic that computes the update is byte-for-byte the sequence it has been since before H4 — which is what the test exists to pin. |
| `test_staging_releases_temporaries_before_the_expression_completes` | Lower bound 17 → **16** parameter-sized buffers, plus a new upper bound `< 17` | The bound moved **down**, because one parameter-sized allocation and one zero-fill pass were removed. The new upper bound makes the removal itself a tested property. |
| Adam and SGD failure-injection parametrizations | `("zeros", 1)` → `("contiguous_copy", 1)` | The same instant in the same transaction — the first parameter's commit copy — reached through the seam that now begins it. Nothing else in a step calls `contiguous_copy`, so index 1 still names it exactly. |
| `test_h0_adds_no_kernel_or_abi_declaration` | Records `tf_copy_internal.h` and `test_contiguous_copy.cpp`; CTests 13 → **14** | The milestone-by-milestone C++ inventory, doing its job. Both additions are hidden-visibility C++ and test scaffolding; the export count it defers to is still 52. |
| `test_h0_touches_no_production_numerical_source` | `copy_value_(` removed from the harness's banned list, with a new check that it appears only inside the one case that measures it | It is not an internal: it is the native line's single documented controlled-mutation primitive, on the public surface, and it is H5's subject. The checkpoint entry points stay banned for the unchanged reason that file I/O is outside this harness. |

The `_committed` helper in the H4 suite, which reproduces what
`copy_value_` installs, moved to the gather with it — otherwise the
reference would silently describe a composition the runtime no longer
runs.

### 16.5.9 Harness cases added

Two, following H3's precedent, taking the harness from 26 to **28**:

- **`row_major_materialization`** (`materialization`) — the same
  `contiguous_copy` export on a **row-major** source. Paired with the
  existing `contiguous_materialization`, which uses a transposed source
  and keeps the odometer, it separates H5's two traversals rather than
  averaging them. Reference: `numpy.ndarray.copy`, deliberately **not**
  `numpy.ascontiguousarray`, which returns an already-contiguous array
  without copying and would time nothing. Its gate asserts raw bit
  identity.
- **`parameter_value_commit`** (`state_operations`) —
  `NativeParameter.copy_value_`. `native_only`, publishing **no ratio**:
  the stable line mutates a `Parameter` by rebinding `.data`, which
  rebinds a NumPy array rather than staging an independent value and
  replacing owned storage under a preserved identity and a moving
  version. Timing them against each other would compare two different
  operations. The correctness gate is real — bit identity with the
  source, `returns self`, exactly one version increment,
  `requires_grad` preserved, source unmutated.

No result file is written, no timing is asserted, and no CI job runs
either.

---

## 16.6 H6 — reduction execution, as shipped

Reductions were the last core family in the runtime that always paid the
generic strided indexing cost. H6 gave `tf_core_sum` a second
**traversal** behind the same unchanged export, chosen from the layout
metadata the export already receives — the dispatch shape H2 and H5 each
proved, applied a third time.

### 16.6.1 The pre-H6 architecture, characterized exactly

Not taken from H0's or H5's summaries — re-read from the current source
and re-measured on the post-H5 build.

**The Python half** (`NativeTensorCore.sum`), unchanged by H6:

- `axis` is **a single `int` or `None`**. No tuple axis exists, so there
  is no duplicate-axis or unsorted-axis case to reject — those are a
  `TypeError` on the argument type. A negative axis is accepted
  NumPy-style (`axis + ndim`); out of range raises `ValueError` naming
  both the axis and the shape; `keepdims` must be an exact `bool`.
- The output shape is `_reduce_shape_checked(shape, axis, keepdims)`, and
  the output storage is **zero-initialized** through
  `NativeTensorCore.zeros`.
- The write-stride vector is `_reduce_out_strides`: **0** for each reduced
  input axis, otherwise the row-major stride of the output axis that input
  axis maps to. `keepdims` changes which output axis that is but not the
  resulting numbers, because a retained reduced axis has extent 1 and a
  row-major stride vector over a size-1 axis contributes nothing.
- `mean` is `sum` followed by an **in-place** `tf_storage_scale(1/count)`
  on the freshly summed output. It has no kernel of its own, and it
  divides **after** accumulation, never during it.
- Reducing all axes (`axis=None`) is not a different code path: every
  write stride is 0 and everything lands in `dst[0]`. A scalar output is
  shape `()` (or all-ones under `keepdims`).
- Zero-length dimensions are **unrepresentable**: the native tensor
  representation rejects a non-positive dimension, so no empty core can be
  constructed from Python at all.

**The C++ half** (`tf_core_sum`), which is what H6 changed:

- Rank 0 returns early with `dst[0] += src[offset]` — before any counter
  is allocated and before any traversal is chosen.
- Otherwise: a `std::vector<int64_t>` odometer counter, then one flat loop
  over the element count executing `dst[out_pos] += src[in_pos]` and a
  per-element **carry loop** from the last axis inward that advances both
  positions and unwinds them on wrap.
- So the input traversal order is exactly the source's **logical**
  row-major order, and for each destination cell the contributions arrive
  in ascending source order starting from whatever `dst` already holds.
  The destination is **read** on every accumulation after the first, which
  is why H1 rejected this output and left it zeroed.
- `tf_core_narrow_backward` is the odometer's **dual** — a scatter, whose
  write strides have no zeros — and is deliberately outside H6's scope.

### 16.6.2 Bottleneck attribution

Instrumented test-locally and benchmark-locally; **no production counter
exists or may exist**. At `(256, 256)`, `axis=0`, contiguous:

| Component | Median | Share |
|---|---|---|
| **raw `tf_core_sum` via `ctypes`, output preallocated** | **94.8 µs** | **95 %** |
| `NativeTensorCore.zeros((256,))` + close | 3.2 µs | 3 % |
| `reduce_shape` | 0.6 µs | |
| `_reduce_out_strides` | 0.5 µs | |
| `_normalize_axis` | 0.4 µs | |
| `np.asarray(out_strides, int64)` | 0.4 µs | |
| `_layout_arrays()` (H3 cache hit) | 0.1 µs | |
| **full `core.sum(axis=0)` + close** | **99.7 µs** | 100 % |

Three of the three `ndpointer` conversions inside that raw figure account
for ~3.2 µs, so the C++ traversal itself is ≈ 91.6 µs — **92 %** of the
operation.

**This is the opposite of B3.** H3's subject was a fixed Python cost that
dominated *small* operations; a reduction of any real size is dominated by
the compiled loop. The whole Python wrapper is ~5 µs and H3 already made
it as cheap as it is going to get without weakening validation, so H6's
only worthwhile target was the traversal.

Measured against NumPy on the pre-H6 build, the gap was **5.6×–17.0×**,
consistent with H0's B5 (5×–15×) and with the same shape of cause: NumPy
has a flat inner loop, `tf_core_sum` had a per-element carry loop.

**The zero-fill is not the target, and H6 confirms H1's rejection rather
than revisiting it.** A reduction's output is the *reduced* shape:
`(256, 256)` `axis=0` zero-fills **2,048 bytes** while reading 524,288 —
under half a percent of the traffic — and `axis=None` zero-fills **8
bytes**. Both traversals accumulate into the destination, so the zero is
the additive identity, not a redundant write.

### 16.6.3 What was implemented

New `cpp/include/tf_reduction_internal.h` declares three hidden-visibility
`namespace tf` functions; `cpp/src/reduction.cpp` implements them and
`tf_core_sum` dispatches between them.

**1. `tf::sum_generic_strided` — the retained generic reference path**
(§8.3). The pre-H6 odometer, unchanged in loop structure, arithmetic, and
traversal order. It is shipped, reachable through ordinary production
dispatch, the **only** path that can address a transposed, narrowed,
non-unit-strided, or broadcast source at all, and the oracle every
optimized result is compared against. Its scratch counter is still
allocated in the exported wrapper, so the guard still maps an allocation
failure onto the C ABI error contract and the fault-injection hook keeps
its existing meaning for this kernel.

**2. `tf::reduce_prefers_contiguous_blocks` — the metadata predicate.**
Total, pure, allocation-free, side-effect-free, and a function of the
layout metadata alone — never of a pointer value, an alignment, a
wall-clock reading, an environment variable, or a CPU-feature probe. On
true it fills three extents; on false it writes nothing and the odometer
runs. A false answer is never an error.

**3. `tf::sum_contiguous_blocks` — the optimized traversal.** A flat walk
over the `outer × mid × inner` factorization the predicate computes.

### 16.6.4 The exact fast-path preconditions

All three must hold:

1. **`in_strides` is exactly the row-major stride vector implied by
   `shape`** — `in_strides[ndim-1] == 1` and
   `in_strides[d] == in_strides[d+1] * shape[d+1]`. This is the same
   definition `NativeTensorView` uses in `backends/cpp.py`, so the two
   layers agree by construction rather than by coincidence.
2. **The reduced axes — those whose write stride is exactly 0, which is
   how the kernel has always identified them — form one contiguous,
   non-empty run of axis indices.**
3. **The kept axes carry exactly the row-major strides of the output
   formed by dropping that run**, checked right to left over the kept axes
   only.

Under those conditions the reduction is exactly

```
dst[o * inner + i]  accumulates  src[offset + (o * mid + m) * inner + i]
```

for `o ∈ [0, outer)`, `m ∈ [0, mid)`, `i ∈ [0, inner)`, where `outer` is
the product of the leading kept axes, `mid` the product of the reduced
run, and `inner` the product of the trailing kept axes.

**Stride collapsing is done implicitly and provably, not by a general
layout compiler.** Conditions 1 and 3 *are* the statement that adjacent
axes of the same class have identical address progressions, so the leading
kept axes, the reduced run, and the trailing kept axes each collapse into
one extent by multiplication. Nothing is cached, interned, or stored: the
factorization is recomputed from the arguments on every call, so there is
no per-operation collapsed-layout cache and nothing global.

**`keepdims` needs no special case at all** and the kernel cannot even
observe it: a retained reduced axis has write stride 0 either way, and a
size-1 output axis contributes nothing to the row-major product.

### 16.6.5 The exact generic fallback conditions

The odometer runs for every layout the predicate rejects:

- **rank 0** (handled by the export before either traversal);
- a **transposed** source, in any permutation;
- a source **narrowed on a non-leading axis** (a leading-axis narrow keeps
  row-major strides and only moves the offset, so it takes the block
  path — with a nonzero offset the traversal honors);
- a **positive non-unit stride** on any axis;
- a **broadcast (stride-0)** source axis;
- a **negatively strided** source;
- **non-adjacent reduced axes** — which the Python layer cannot currently
  express, since it reduces one axis or all of them, so this is
  future-proofing rather than a live case;
- **nothing reduced** (no zero write stride);
- any **write-stride vector that is not the output's row-major strides**.

Each is exercised by a real layout in
`tests/test_native_reduction_dispatch.py` and by the predicate table in
`cpp/tests/test_sum_reduction.cpp`.

### 16.6.6 The arithmetic-order proof

**Per-output accumulation order is preserved exactly, and the source
traversal order is not even reordered.**

The loop nest `o`, `m`, `i` is the lexicographic order of the source's own
row-major index — `((o · mid) + m) · inner + i` is strictly increasing in
that order — which is precisely the order the odometer walks. And every
destination cell is touched by exactly one `(o, i)` pair, so the cells are
independent: no cell's total can be affected by the order in which the
*other* cells are computed.

For a fixed destination cell, both paths therefore add the same source
values, in the same ascending order, starting from the same initial value
— the value `dst` already holds. Nothing is reassociated, no partial sums
are combined, no accumulator width changes, no fused multiply-add is
requested, no Kahan compensation exists, and no pairwise, tree, parallel,
or horizontal-vector reduction exists anywhere.

The traversal takes two shapes, and the reason for the split is worth
recording because one of them is the interesting case:

- **`inner == 1`** (a full reduction, or one whose reduced run is a
  suffix): each destination cell is fed by one contiguous ascending source
  run, so the run is accumulated in a **local accumulator seeded from
  `dst[o]`** and stored once. Seeding from the destination rather than
  from a literal `0.0` is what keeps the export's documented
  **accumulate-into** semantics identical on both paths — which
  `cpp/tests/test_sum_reduction.cpp` verifies directly by running both
  traversals over a pre-filled destination, and which
  `tests/test_native_reduction_dispatch.py` verifies through the ABI.
- **`inner > 1`**: for each `(o, m)` a contiguous source row is added
  elementwise into a contiguous destination row. Distinct `i` are distinct
  destination cells, so the compiler may vectorize this loop — that is a
  vectorization *across independent outputs*, never a horizontal reduction
  within one, and it reassociates nothing. (Reduction vectorization would
  require reassociation, which no build here enables: there is no
  `fast-math` anywhere.)

### 16.6.7 The signed-zero contract

A zero sum's sign depends on the initial accumulator and on the addition
order, which is exactly what a local accumulator could have changed. It
does not:

- Both paths start from the destination's `+0.0` (a zeroed buffer holds
  `+0.0`), and `+0.0 + -0.0` is `+0.0`, so **the sum of any number of
  `-0.0` values is `+0.0` on both paths** and matches NumPy.
- Seeded with `-0.0` instead, both paths keep `-0.0`.
- All-positive zeros, all-negative zeros, alternating zeros, `-0.0` first,
  `-0.0` last, `-0.0` mixed with finite values, a column of `-0.0`, and
  exactly cancelling finite values are each compared as **raw IEEE-754 bit
  patterns**, at every axis, both `keepdims` values, and scalar and
  multi-output shapes. Every one agrees.

One case is recorded rather than idealized: the **rank-0** export branch
is `dst[0] += src[offset]`, a genuine addition against the zeroed
destination, so a rank-0 `-0.0` sums to `+0.0`. That is exactly what it
did before H6 and is now pinned by a test.

### 16.6.8 The NaN contract — measured, not inherited from H2

H2's matmul carve-out was **not** copied over. The reduction-specific rule
was measured, and the measurement produced a different and narrower
answer.

**What is contractual:**

1. **NaN positions are identical on both paths.** Whenever either path
   produces a NaN, both do, in exactly the same positions.
2. **Every NaN either path produces is quiet.** Neither can emit a
   signaling NaN, and a signaling-NaN *input* is quieted by both — with
   identical bits, because only one NaN is involved.
3. **With at most one NaN per accumulation the two paths are bit-identical,
   payload included** — every pattern in the sweep, in first, middle, and
   last position. This is the case that actually occurs in practice.

**What is not contractual:** when **two or more** NaNs are accumulated
into one destination cell, the paths may select different payload bits.
This is asserted in **neither** direction.

**Why it is not available at any price.** Four spellings of the optimized
accumulation were measured on this toolchain — `acc += x`, `acc = x + acc`,
a named-temporary `acc = acc + x`, and `dst[o] += x` accumulating *through
memory* exactly as the odometer does. **All four selected the same NaN,
and all four differed from the odometer.** So the divergence is not caused
by the local accumulator, and removing the accumulator would not recover
parity — it is caused by the odometer's destination index being a
*runtime-varying* value, which changes which addend MSVC places in the
`ADDSD` destination register, and that is an instruction-selection
decision C++ cannot express. Parity is therefore unavailable short of
abandoning the optimization entirely, and the memory-accumulate spelling
that came closest structurally was also measured **1.2×–1.8× slower** on
suffix reductions (5.24× vs 7.79× at `(128,128) axis=1`, 11.10× vs 14.88×
at `(4096,32) axis=1`), so it bought nothing.

**Which payload each keeps, recorded as an observation and not a promise:**
the block path keeps the **first** NaN in accumulation order, the odometer
keeps the **last** — and the block path's choice is the one **NumPy**
makes. So where the two paths differ, H6 moved the answer *toward* NumPy.

**And the divergence is an optimizer artifact, which the build matrix
demonstrates directly.** `cpp/tests/test_sum_reduction.cpp` counts its own
payload-only differences and prints the number, so the same binary reports
it per configuration: **MSVC Release measured 252 payload-only differences
across 612 traversal comparisons** over a NaN-dense sweep, and **MSVC Debug
measured 0** — the same pattern H2 recorded (Release 162 of 208, Debug and
Clang none). Both configurations pass, because the test tolerates a payload
difference while still asserting NaN-ness and quietness. That is exactly
why the rule is asserted in **neither** direction: a build whose payloads
agree and a build whose payloads differ are equally conforming, and pinning
either answer would make the suite fail on a legitimate toolchain.

**H5's copy rule does not apply here either**, and the reason is the same
one that made H5's claim strong: a value transfer performs no arithmetic,
so it has no operand roles for a compiler to choose between. A reduction
is arithmetic. The three rules are genuinely different because the three
operations are.

### 16.6.9 The H1 allocation decision

**Outcome A: the destination stays zero-initialized**, on both paths, and
H6 confirms H1's rejection rather than revisiting it.

Both traversals **read** the destination — that is what accumulation
means — so an uninitialized buffer would return garbage. The block path's
local accumulator is seeded from `dst[o]` for exactly this reason.

Outcome B was considered and **rejected on two grounds, one measured and
one semantic**:

- *Measured*: the fill is 2 KB against 512 KB of reads at `(256,256)`
  `axis=0`, and 8 bytes at `axis=None` — under half a percent of the
  traffic, against a traversal that was 92–95 % of the operation. There is
  nothing material to win.
- *Semantic*: making the fast path *assign* its first contribution would
  give the two paths **different** behavior for a non-zero destination,
  which would break the export's documented accumulate-into contract and
  make the generic path stop being the reference. A fast path that is not
  substitutable for the reference path is not a fast path.

So H6 adds **no** poison test, because it introduces no uninitialized
destination. The existing H1 poison suite is untouched and still passes.
`sum` reaching `zeros` and never `_uninitialized` is asserted
structurally, and the accumulate-into behavior that makes the zero
load-bearing has its own negative control at the ABI.

### 16.6.10 Measured results

Methodology: a **pre-H6 library** built from the identical sources with
only `reduction.cpp` restored, driven through **identical `ctypes` calls
on identical data**, with every output proved **bit-identical before
either side was timed**; 15 alternating pre/post rounds so machine drift
hits both sides. `time.perf_counter_ns()`, medians reported with minima.
The machine's control band for this measurement is **0.90×–1.03×**.

**Kernel level** (raw `tf_core_sum`, output preallocated, layout arrays
prebuilt — so this is the traversal plus one ctypes call):

| Case | pre | post | ratio |
|---|---|---|---|
| full, 1 element | 8.5 µs | 8.5 µs | **1.00×** |
| full, 16 | 8.9 µs | 8.8 µs | **1.01×** |
| full, 1,024 | 9.9 µs | 8.3 µs | 1.19× |
| full, 16,384 | 44.1 µs | 17.2 µs | 2.56× |
| full, (128,128) | 43.0 µs | 16.4 µs | 2.62× |
| full, (256,256) | 140.9 µs | 40.4 µs | 3.49× |
| full, (512,512) | 540.9 µs | 136.5 µs | 3.96× |
| full, 3-D (8,16,32) | 15.4 µs | 8.5 µs | 1.81× |
| full, 4-D (8,4,16,16) | 23.7 µs | 11.1 µs | 2.14× |
| (8,8) axis=0 | 7.3 µs | 7.1 µs | **1.03×** |
| (128,128) axis=0 | 32.4 µs | 10.0 µs | 3.24× |
| (128,128) axis=1 | 41.3 µs | 11.8 µs | 3.50× |
| (256,256) axis=0 | 98.2 µs | 18.6 µs | 5.28× |
| (256,256) axis=1 | 132.1 µs | 28.4 µs | 4.65× |
| (512,512) axis=0 | 320.8 µs | 50.7 µs | **6.33×** |
| (512,512) axis=1 | 520.4 µs | 111.6 µs | 4.66× |
| (1024,1024) axis=0 | 1.41 ms | 221.8 µs | **6.37×** |
| (1024,1024) axis=1 | 2.23 ms | 490.8 µs | 4.54× |
| 3-D (32,64,32) axis=0 | 171.9 µs | 19.1 µs | **9.00×** |
| 3-D (32,64,32) axis=1 | 160.0 µs | 18.6 µs | **8.60×** |
| 3-D (32,64,32) axis=2 | 163.0 µs | 18.7 µs | **8.72×** |
| 4-D NCHW (16,8,32,32) axis=0 | 334.5 µs | 37.4 µs | **8.94×** |
| 4-D NCHW (16,8,32,32) axis=1 | 333.6 µs | 34.7 µs | **9.61×** |
| 4-D NCHW (16,8,32,32) axis=3 | 317.2 µs | 29.0 µs | **10.94×** |
| prime (127,131) axis=0 | 32.0 µs | 10.7 µs | 2.99× |
| prime (127,131) axis=1 | 40.8 µs | 12.3 µs | 3.32× |
| one-element dim (128,1,64) axis=1 | 27.5 µs | 12.2 µs | 2.25× |
| inner=2 (100,100,2) axis=1 | 48.6 µs | 27.7 µs | 1.75× |
| inner=7 (100,100,7) axis=1 | 135.9 µs | 35.2 µs | 3.86× |
| inner=4 (4096,4) axis=0 | 29.0 µs | 16.4 µs | 1.77× |

**The rank effect is the headline finding, and it was not predicted.** The
odometer's carry loop runs up to `ndim` iterations per element, so its
cost grows with rank while the block traversal's does not: 2-D reductions
improve 2.6×–6.4×, and 3-D and 4-D reductions improve **8.6×–10.9×**. The
NCHW cases matter because that is the layout the convolution stack
produces.

**Layer level** (9 alternating subprocess rounds, correctness gated):

| Case | pre | post | ratio |
|---|---|---|---|
| `TensorCore.sum(axis=0)` (256,256) | 110.9 µs | 24.7 µs | 4.49× |
| `TensorCore.sum(axis=1)` (256,256) | 145.2 µs | 36.9 µs | 3.93× |
| `TensorCore.sum()` (256,256) | 150.6 µs | 44.6 µs | 3.38× |
| `TensorCore.mean(axis=0)` (256,256) | 105.3 µs | 25.6 µs | 4.11× |
| `TensorCore.mean(axis=1, keepdims)` | 151.0 µs | 38.4 µs | 3.93× |
| `TensorCore.sum(axis=0)` (512,512) | 343.6 µs | 70.7 µs | 4.86× |
| `TensorCore.sum()` (512,512) | 549.4 µs | 149.7 µs | 3.67× |
| `TensorCore.sum(axis=1)` NCHW | 332.9 µs | 38.9 µs | **8.56×** |
| `TensorCore.sum(axis=3)` NCHW | 335.9 µs | 38.1 µs | **8.82×** |
| `NativeTensor.sum(axis=0)`, no graph | 100.5 µs | 25.9 µs | 3.88× |
| `NativeTensor.sum(axis=0)`, graph | 103.9 µs | 27.2 µs | 3.82× |
| `sum()` forward + backward | 599.1 µs | 473.3 µs | 1.27× |
| `mean(axis=1, keepdims)` fwd + bwd | 676.4 µs | 550.0 µs | 1.23× |
| **conv bias gradient, 3 chained sums** | 75.7 µs | 51.8 µs | **1.46×** |
| broadcast add backward (`_unbroadcast`) | 522.7 µs | 453.2 µs | 1.15× |
| `softmax` backward | 135.3 µs | 118.3 µs | 1.14× |
| `log_softmax` backward | 150.5 µs | 136.4 µs | 1.10× |
| `NativeLayerNorm` forward | 345.1 µs | 298.2 µs | 1.16× |
| `NativeBatchNorm2d` backward | 1.411 ms | 1.288 ms | 1.10× |
| `cross_entropy` forward | 60.0 µs | 57.4 µs | 1.05× |
| `cross_entropy` backward | 30.7 µs | 29.3 µs | 1.05× |

**Against NumPy**, in the shipped harness at the `full` configuration, the
contiguous reduction gap closed substantially:

| Harness case | post-H6 vs NumPy |
|---|---|
| `reduction_middle_axis_4d` (16,8,32,32) | **1.67×** |
| `reduction_contiguous` (256,256) axis=0 | **2.43×** |
| `reduction_last_axis` (256,256) | 2.90× |
| `reduction_full_to_scalar` (256,256) | 3.75× |
| `reduction_transposed_view` *(control, generic path)* | 10.33× |

### 16.6.11 Negative, neutral, and noise results

Reported as prominently as the wins.

- **Every training step is neutral.** MLP small **0.99×**, MLP large
  1.03×, normalized 1.03×, CNN 1.01×, Dropout (control) 1.02× — all
  inside the control band. A reduction is a small share of a step whose
  cost is the optimizer and the large matmuls; H4 already showed the
  optimizer is 83 % of an MLP step's allocations. **H6 does not make
  training faster**, and no reading here should be quoted as if it did.
- **BatchNorm and LayerNorm are mostly neutral**: BatchNorm1d training
  forward 1.04×, eval 0.98×, backward 1.02×; BatchNorm2d training forward
  1.06×, eval 1.00×; LayerNorm backward 1.01×. Only LayerNorm forward
  (1.16×) and BatchNorm2d backward (1.10×) are clearly outside the band.
  These modules are 11–30 allocations of broadcast elementwise work with
  one or two reductions in them, so speeding the reductions ~4× moves the
  total little. This narrows H7 rather than motivating it.
- **Tiny reductions are neutral**: 1 element 1.00×, 16 elements 1.01×,
  `(8,8)` axis=0 1.03×. Below roughly 1,000 elements the ~7 µs
  Python-plus-ctypes cost dominates and the traversal is invisible. That
  cost is H3's and H5's documented boundary finding and belongs to a
  dispatch milestone, not to this one.
- **A real, repeatable ~10 % regression on one fallback family.** A **2-D
  transposed source reduced over `axis=0`** measured **0.89×–0.93×** across
  four independent 25-round runs (and the `(512,512)` variant 0.84×–0.97×),
  while the 3-D transposed `axis=0` fallback measured **1.04×–1.05×
  faster** and every other fallback 0.96×–1.01×. Both libraries run the
  *identical* odometer on these layouts, so this is not an algorithmic
  change. It was **isolated**: in a standalone binary the extracted-function
  spelling versus the inline spelling measured 0.88×–1.67× with no stable
  direction, so the extracted call is **not** the cause. The remaining
  attribution is whole-translation-unit code layout — adding the predicate
  and the second traversal to `reduction.cpp` moved where the odometer
  lands in the image. Chasing MSVC code placement is exactly the
  machine-specific tuning §11–§13 reject, so it is **published rather than
  fixed**. It affects no shipped path: no production call site reduces a
  transposed 2-D view over `axis=0`, and no end-to-end case regressed.
- **A small-`inner` middle-axis reduction is the weakest win**: `inner=2`
  1.75×, `inner=4` 1.77×. A specialized register-blocked path for
  `inner < 8` was considered and **rejected on complexity**: it would add
  a second threshold and a second kernel shape to serve a layout no
  shipped code produces, for a case already 1.75× better.
- **Methodology, published rather than buried** — the same lesson H3 and
  H5 each recorded. At 7 alternating rounds the fallback controls read
  0.85×; at 21–25 rounds the same cases read 0.90×–1.02×. No
  low-round figure is quoted as H6 evidence.

### 16.6.12 Memory

**H6 changed no allocation anywhere**, and this is asserted rather than
assumed: a `sum` allocates **exactly one** native storage — its own
output — on both paths, at every axis, under both `keepdims` values, and
`mean` allocates the same one because its scale is in place. There is no
scratch buffer, workspace, arena, or pool; the odometer's counter is the
only auxiliary allocation and it is unchanged and only on the fallback
path.

Peak transient bytes are therefore identical, and a 10-step training run
over a model carrying parameters, BatchNorm buffers, and Adam moments was
measured to produce a **bit-identical** allocation and live-count profile
before and after H6 (27/53/79/… on both), which also confirms the
oscillation in that profile is CPython's collector and not a leak either
version introduced.

### 16.6.12a One pre-existing cleanup gap was found and closed

Exercising the failure matrix the milestone requires — a failure at axis
validation, output-shape construction, write-stride construction, the
layout arrays, the allocation, either traversal, and `mean`'s scale — found
that `sum` and `mean` were the only allocating Core operations **without an
explicit post-allocation cleanup boundary**. `sum` allocates its output and
*then* normalizes the axis, builds the write strides, and builds the layout
arrays; `mean` calls `sum` and then scales. A failure in any of those
released the output only because CPython's refcount happened to drop to
zero on the exception unwind and `__del__` called `close()`.

Nothing leaked in practice, but relying on a refcount is relying on an
implementation detail rather than on the contract — and the project's own
rule is that correctness never depends on when a finalizer runs. Both
methods now wrap their post-allocation work in the same
`try: … except BaseException: out.close(); raise` boundary every other
allocating Core operation already uses, `contiguous_copy` included (which
H5 touched, and which is where the pattern was read from).

This is the narrowest possible change, it is inside the two functions H6
exists to improve, and it changes **nothing** on the success path: the
same operations run in the same order and produce the same bits. Each seam
is now pinned by a test that injects `RuntimeError`, `MemoryError`,
`KeyboardInterrupt`, and a non-`Exception` `BaseException` and asserts live
storage returns to baseline with **no `gc.collect()`**, alongside the
pre-allocation seams asserting that nothing is allocated at all.

### 16.6.13 Fast paths evaluated

Each candidate was investigated independently rather than forced into
production.

| Candidate | Outcome |
|---|---|
| Full contiguous reduction to a scalar | **Shipped** — `outer == 1, inner == 1` |
| Contiguous suffix-axis reduction | **Shipped** — `inner == 1` |
| Contiguous prefix-axis reduction | **Shipped** — `outer == 1` |
| Single last-axis reduction | **Shipped, not separately** — it *is* the suffix case; a distinct kernel measured no better and would have duplicated code |
| Single first-axis reduction | **Shipped, not separately** — it *is* the prefix case |
| Middle-axis reduction | **Shipped** — all three extents above 1 |
| Stride collapsing | **Shipped, bounded** — implicit in preconditions 1 and 3, restricted to one contiguous reduced run; explicitly *not* a general layout compiler, and nothing is cached |
| Mean | **Unchanged** — it reuses the optimized `sum` and the existing in-place scale. Division placement, order, and the `1/count` value are untouched, and fusing the scale into accumulation was rejected outright as a §7.3 order change |
| Uninitialized destination (H1 Outcome B) | **Rejected** — §16.6.9 |
| Memory-accumulate spelling for payload parity | **Rejected** — §16.6.8: it does not achieve parity and is 1.2×–1.8× slower |
| Register-blocked small-`inner` path | **Rejected** — §16.6.11 |
| Non-adjacent reduced axes | **Rejected as out of scope** — unreachable from Python; falls back |
| `tf_core_narrow_backward` | **Rejected as out of scope** — a scatter, not a reduction; widening H6 to it would have made this a scatter milestone |

### 16.6.14 What H6 did not do

- **No exported C ABI symbol.** The library still exports exactly **52**
  `tf_*` symbols, asserted against the built image.
- **No new translation unit.** One internal header and one CTest, taking
  the native suite from 14 to **15**.
- **No public control of any kind**: no path selector, threshold setter,
  block-size setter, dispatch tracer, profiling counter, environment
  variable, or "which path ran" query. The predicate is hidden-visibility
  C++ the native test reaches only by compiling the source in.
- **No public API change**: `sum` and `mean` keep their signatures,
  defaults, axis rules, `keepdims` behavior, error types, and error
  messages. Multi-axis reduction was **not** added — the kernel can
  factorize a contiguous reduced run, but the Python layer still accepts
  one `int` or `None`.
- **No capability, dtype, device, registry, or checkpoint move.**
  `UNSUPPORTED` still reads `("float32", "cuda", "amp")`,
  `SUPPORTED_DTYPES` `("float64",)`, `SUPPORTED_DEVICES` `("cpu",)`, and
  the checkpoint format is still `tensorforge.native_checkpoint` version
  **2** with versions **(1, 2)** supported.
- **No SIMD, threading, OpenMP, BLAS, parallel reduction, memory pool,
  scratch workspace, or fast-math.**
- **No timing threshold** in any test or CI job.

### 16.6.15 Harness cases added

Three, following H5's separate-rather-than-average precedent, taking the
harness from 28 to **31**. All three are `reduction`-family, contiguous,
NumPy-referenced, and correctness-gated before timing.

- **`reduction_last_axis`** — `sum(axis=-1)`. The suffix form, which is
  what LayerNorm's mean, softmax's backward, and log-softmax's backward
  actually reduce over, and the form H6 gave a local accumulator.
- **`reduction_full_to_scalar`** — `sum()`. Every write stride 0, rank-0
  output: the single hottest reduction in the runtime, since every
  mean-reduced loss ends in it.
- **`reduction_middle_axis_4d`** — `sum(axis=1)` over NCHW. Kept axes on
  both sides, so all three block extents exceed 1, and the rank-4 reading
  the 2-D cases cannot give.

`reduction_transposed_view` is now explicitly the pair's **control**: the
predicate rejects it, so its compiled traversal did not change.

No result file is written, no timing is asserted, and no CI job runs
either.

---

## 16.7 Remaining ladder detail

**The ladder was reordered at H5** (§16.5.0). Reduction execution, drafted
as H5, became H6 and has now shipped (§16.6); the conditional slots after
it shift accordingly.

### H7 — Composed-module cost *(conditional, and now the weakest slot)*

Reduce the allocation and dispatch count in the normalization modules
and the composed convolution bias gradient.

**The original condition was that H1, H3, and H6 must land first and a
re-measured normalization step must still show a material composed-module
cost. All three have now landed, and the re-measurement is in.**

- H3 improved a normalized training step **1.51×** and cut its ratio
  against the stable line from 6.76× to 5.28× — much of B6 was per-call
  overhead, and that part is gone.
- **H6 has now answered the question H3 left open.** A normalization
  forward is dominated by `mean` and broadcast `subtract`, and H6 made
  `mean` **3.9×–4.1×** faster. The measured effect on the modules was
  small: LayerNorm forward **1.16×**, BatchNorm2d backward **1.10×**, and
  BatchNorm1d training forward, eval forward, backward, BatchNorm2d
  forward, and LayerNorm backward all **inside the 0.90×–1.03× control
  band** (§16.6.11). The normalized training step as a whole moved
  **1.03×**, i.e. not at all.
- The conclusion the evidence supports: **what is left in B6 is not the
  reductions and not the per-call cost — it is the sheer count of
  broadcast elementwise operations** (11 allocations for a LayerNorm
  forward, 25 for a BatchNorm1d training forward, 30 for BatchNorm2d),
  each a separate allocation, a separate ctypes crossing, and a separate
  full pass over a parameter-sized buffer. Speeding one component of a
  composition 4× cannot move a total that is spread across two dozen
  components.

So H7 as originally framed — "make the reductions and dispatch inside
normalization cheaper" — is **answered and should not be entered**. The
only version of it that could still pay is broadcast-elementwise traversal
and allocation count, which is **H8's** subject, not a normalization one.
**It must not introduce a normalization kernel**: F2–F4's achievement is
that both families are compositions with no C++ at all, and trading that
for speed would be a semantic regression, not an optimization.

### H8 — Elementwise and materialization traversal *(conditional)*

Stride collapsing and materialization cost, if §3.2's hypothesis
survives measurement after H1 and H3. **Narrowed by H5**, which already
gave materialization its flat traversal and measured 2.5×–5.5× for it, so
what remains here is stride collapsing alone. Presently the
weakest-evidenced numbered milestone, and the most likely to be dropped.

### H9 — SIMD, threading, or optional BLAS *(conditional, presumed rejected)*

Entered only under §11, §12, or §13 respectively. On current evidence
**none of the three qualifies**, and the honest expectation is that H9
ships as a *documented rejection with measurements* rather than as code.
That is a legitimate outcome for this milestone.

### H10 — Re-measurement, hardening, and sanitizer validation

The full H0 harness re-run and compared against the H0 baseline;
cross-cutting integration tests; adversarial failure-atomicity and
ownership matrices over whatever H1–H9 changed; Windows Release and
Debug builds with the full CTest suite; the Clang ASan/UBSan matrix with
instrumentation proved; LeakSanitizer returning live storage exactly to
baseline.

### H11 — Phase closure

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
| Naive CPU matmul as the reference (`cpu_matmul<T>`, triple loop, no blocking/SIMD/OpenMP/BLAS) | **Adopt as a reference, reject as the shipped path** | Daedalus keeps its CPU matmul naive and puts its effort into cuBLAS/CUDA. TensorForge has no GPU path to fall back on, and §3.1/B2 measured 3.3× available with bit-identical results. Keep a naive kernel as the retained reference (§8.3); do not keep it as the only path. **Done at H2**: `tf::matmul_generic_strided` is the naive kernel, retained verbatim, shipped, and reachable through ordinary dispatch; `tf::matmul_row_sweep` is the second path beside it. |
| `WITH_AVX2` CMake option (`/arch:AVX2` on MSVC, `-mavx2` elsewhere), default OFF | **Reject for now; adopt the *shape* if §11 is ever met** | The default-off, single-option, no-intrinsics-in-source form is exactly right and is the shape TensorForge would use. But adopting it now would be optimizing the second-order term before the first (§11), and TensorForge's evidence said the win was in loop order — which **H2 then confirmed by measurement**, recovering 4.1–4.7× at the profile shape with no intrinsics and no build-system change. Revisit only under §11. |
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
