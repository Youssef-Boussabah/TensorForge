# Backend experiments

This page tracks the advanced experimental line that started after the
v3.0 Python release. **Nothing here is part of the finished Python
framework** — `import tensorforge` never touches it, Tensor and
autograd are unchanged, and every existing API works exactly as
before.

**How to read this page.** It is written **newest-last within each
milestone section but historical throughout**: every section records the
state of the line *at the milestone it describes*, and later milestones
routinely supersede earlier ones. The one place that always describes
**today** is the "Where the line is now" section immediately below;
the authoritative per-capability status lives in the
[native support matrix](native_support_matrix.md) and the backend
capability registry it mirrors.

## Where the line is now

Phase A (native CPU runtime), Phase B (native autograd), Phase C (the
native training stack), and Phase D (the native CNN stack, milestones
D0–D12) are all **complete**, through Advanced C++ v3.16. The native line
today has an explicit storage/view/core/tensor runtime, a Python-managed
reverse-mode autograd over autograd-unaware kernels, parameters, modules
(`NativeLinear`, `NativeReLU`, `NativeFlatten`, `NativeConv2d`,
`NativeMaxPool2d`, `NativeSequential`), `NativeMSELoss`, `NativeSGD` and
`NativeAdam` with in-memory optimizer state, pickle-free native
checkpoints with exact resume, and deterministic end-to-end MLP **and**
CNN training proofs. The phase that followed them — **Phase E, Native
Classification and Stable Math** — has its architecture contract locked
in [native_classification_design.md](native_classification_design.md)
(E0) and is **complete** (E0–E10): milestones E1–E4 shipped the differentiable
native `exp`, `log`, `softmax`, and `log_softmax`, and E5 shipped the
fused `cross_entropy` **Core** contract
(`NativeTensorCore.cross_entropy_forward` / `cross_entropy_backward`
over two guarded, contiguous-only exports, with strict copied `int64`
targets and private saved probabilities). E6 then shipped the
differentiable `NativeTensor.cross_entropy(targets, reduction="mean")`
over it — one autograd node with graph-owned saved probabilities, no
logits reread, and no expected parameter version, adding no kernel and
no ABI export. E7 then completed the public surface:
`NativeCrossEntropyLoss`, a stateless `NativeModule` delegating entirely
to that operation, and the reporting-only `native_accuracy` — explicit
`to_numpy()` plus a NumPy argmax, returning a Python `float`, building
no graph and touching no gradient, listed in the new `NATIVE_METRICS`
inventory. **E8 then proved the whole stack trains and resumes**:
`examples/native_classification_training.py` trains a native
Conv2d→ReLU→MaxPool2d→Flatten→Linear classifier on twelve fixed 6×6
images in three classes for 40 deterministic `NativeAdam(lr=0.05)` steps
(loss 1.159638 → 0.000101, accuracy 0.3333 → 1.0000), then checkpoints at
step 15 and resumes into a fresh model/optimizer pair that reproduces the
remaining loss suffix, the parameters, the optimizer state, the logits,
the predictions, and the accuracy **exactly** — an integration proof that
added no kernel, ABI export, operation, module, optimizer, or schema
change (native checkpoint format stays version 1). **E9** then
characterized that stack honestly:
`benchmarks/benchmark_native_classification.py` measures `exp`, `log`,
`softmax`, `log_softmax`, the fused cross-entropy forward, its backward
alone, and one complete classification training step — every case gated
for correctness *before* timing, every case labelled with the reference
it used (`stable_tensorforge`, `numpy`, or `native_only`), medians
reported with min/max/spread after warm-up, `--smoke` and `--json` modes,
and **no speed assertion or timing threshold anywhere**. **E10** closed
the phase without adding any numerical capability: cross-cutting
integration tests (`tests/test_native_phase_e.py`), Release **and** Debug
native builds (10/10 CTests each, zero warnings), Clang ASan/UBSan
validation of the whole classification stack with zero diagnostics
attributable to TensorForge, a practical LeakSanitizer pass with no
native leak, the full Python regression, and documentation
reconciliation. Phase E therefore delivered stable native classification
mathematics end to end — float64/CPU only, with no implicit
stable/native dispatch and no change to the stable framework or the
version-1 checkpoint format.

**The latest phase is Phase F — Native Normalization and Stateful
Buffers — and it is *complete* (F0–F9).** Milestone **F0** is complete: it
locks the architecture contract in
[native_normalization_design.md](native_normalization_design.md) —
`NativeLayerNorm`, `NativeBatchNorm1d`, and `NativeBatchNorm2d`
**composed from existing native operations** with no new kernel, C ABI
export, ctypes declaration, or `NativeTensorCore` method; persistent
native running statistics as registered buffers; the rule that a live
mutable running buffer is never captured as a rereadable graph operand
(eval mode takes independent graph-free snapshots, which is exactly why
buffers stay unversioned); atomic two-buffer running-statistics updates
with rollback and preserved buffer identity; and state/checkpoint
integration with the format unchanged at **version 1**. **F0 added
design and documentation only — no numerical behavior.** **F1** then
shipped the private atomic native-buffer state transaction that contract
calls for (`src/tensorforge/experimental/_native_state.py`: staging, an
explicit commit boundary, complete rollback of both cores and parameter
versions, exactly-once closing, and identity-preserving swaps), refactored
`NativeModule.load_state_dict` onto it with its public behavior unchanged,
and corrected `STATE_SUPPORT` to report the `persistent_buffers`
capability that had existed since before Phase D — state management and
capability reporting only, with **no normalization mathematics**.
**F2** then shipped `NativeLayerNorm`
(`src/tensorforge/experimental/native_layernorm.py`) — the first native
normalization module: stateless (no buffers, identical in train and
eval), differentiable through the mean and the population variance, and
composed entirely from existing native operations (`mean`, `subtract`,
`multiply`, `add`, `sqrt`, `reciprocal`, `sqrt(var + eps)`, no Bessel
correction) with no kernel, C ABI symbol, `NativeTensorCore` method,
custom backward, functional helper, or `NativeTensor.layer_norm`
operation. `"NativeLayerNorm"` joined `NATIVE_MODULES` and the exports,
and `"layernorm"` left `UNSUPPORTED`.
**F3** then shipped `NativeBatchNorm1d`
(`src/tensorforge/experimental/native_batchnorm.py`) — the **first
stateful native numerical module**: `(N, C)` batch normalization whose
training statistics are differentiable (gradients flow through the batch
mean *and* the population variance, never detached), whose
`running_mean`/`running_var` are persistent native buffers advanced by
`(1 − momentum)·running + momentum·batch` from the *same* batch
statistics — computed graph-free and committed as one **atomic
two-buffer transaction** through the F1 primitive, preserving both
identities, closing each replaced core exactly once, and moving no
parameter version — and whose evaluation mode reads **independent owning
graph-free snapshots** of those buffers, so no registered buffer is ever
a rereadable graph operand and a later training step, buffer-only
`load_state_dict()`, or buffer-only `load_native_checkpoint()` cannot
change an earlier eval graph's gradient (a full checkpoint load also
replaces `gamma`/`beta`, so the unchanged v3.7 parameter-version guard
correctly stales that graph — a parameter contract, never a buffer
effect). It too
is composed entirely from existing native operations, adding no kernel, C
ABI symbol, `NativeTensorCore` method, custom backward, functional
helper, or `NativeTensor.batch_norm` operation, and the native checkpoint
format stays at **version 1**. `"NativeBatchNorm1d"` joined
`NATIVE_MODULES` and the exports, while `"batchnorm"` **stayed** in
`UNSUPPORTED` — the unqualified name is only honest once the NCHW shape
exists too.
**F4** then shipped `NativeBatchNorm2d` (the second public class in the
same file) — NCHW `(N, C, H, W)` batch normalization reducing over **N,
H, and W**, so each channel gets one population mean and one population
variance over `N * H * W` values. It is built on the **same** shared
private implementation: it declares only `_INPUT_NDIM = 4`,
`_REDUCTION_AXES = (0, 2, 3)`, `_TRAILING_DIMS = 2`, its layout string,
and `_CHANNELS_LAST = (0, 2, 3, 1)`, and inherits every method by
function identity. The one shared refinement it needed is the
channelwise affine: rank-1 `gamma`/`beta` broadcast from the *trailing*
axis, so the **activation** is transposed to channels-last for the
affine application and back again (then materialized contiguous) rather
than reshaping the parameters — which keeps `gamma` a direct versioned
`multiply` operand and therefore preserves the existing stale-parameter
guard exactly. Running buffers stay `(C,)`, eval snapshots are owning
`(1, C, 1, 1)` copies, and the checkpoint format stays version 1.
`"NativeBatchNorm2d"` joined `NATIVE_MODULES` and the exports, and with
both shapes live `"batchnorm"` **left** `UNSUPPORTED`.
The numerical normalization module surface is therefore complete, and
**F5 is complete** — the exhaustive state/checkpoint, ownership, and
graph-safety hardening (a focused
`tests/test_native_normalization_state.py` plus narrow additions to the
generic buffer and checkpoint suites), proving §7–§10 by executable test
rather than by prose: canonical dotted buffer keys, independent state
snapshots, strict/non-strict loads, exact never-casting metadata
validation, mixed parameter/buffer transaction atomicity, buffer identity
across state and checkpoint loads, exact eval-output reproduction, the
buffer-only-versus-full stale-graph distinction, the save/corrupt-load
failure boundaries, eval-graph snapshot safety under `retain_graph` and a
failed retryable backward, and explicit parameter/buffer closure. F5 is
**tests and documentation only** — no numerical behavior, no new public
capability, and the checkpoint format stays version 1. And **F6 is
complete** — the deterministic normalized training and exact-resume proof
`examples/native_normalization_training.py`: a
`Linear → BatchNorm1d → ReLU → LayerNorm → Linear` regressor trained for
24 deterministic `NativeAdam` steps with `NativeMSELoss` (98.9% loss
reduction), whose two uninterrupted runs are bit-identical and whose
interrupted checkpoint resume into a fresh model/optimizer pair reproduces
the remaining loss suffix, every parameter, the NativeAdam state, both
BatchNorm `running_mean`/`running_var`, the final training-step
prediction, and the final evaluation-mode output exactly (format version 1
unchanged, training flags runtime-only) — one example and its integration
test, adding no capability. And **F7 is complete** — the honest benchmark
characterization `benchmarks/benchmark_native_normalization.py`: nine
cases (the LayerNorm forward and backward, the BatchNorm1d training
forward, evaluation forward, and backward, the BatchNorm2d training
forward, evaluation forward, and backward, and one complete F6-style
normalized training step), each **correctness-gated before any timing**,
six labelled `stable_tensorforge` against `tensorforge.nn` equivalents on
identical state and three (the BatchNorm2d shapes) labelled `native_only`
because the stable line has no public `BatchNorm2d` to time against —
those publish no ratio while keeping a rigorous NumPy NCHW and
transformed-oracle correctness gate. Medians with min, max, and spread
after warm-up; `--smoke` and `--json` modes; **no result file, no speed
assertion, no committed timing number, and no CI timing threshold** —
measurement only, adding no capability and changing no production
behavior. And **F8 is complete** — the cross-cutting integration and
semantic guardrails `tests/test_native_phase_f.py`: one integrated
`Conv2d → BatchNorm2d → ReLU → MaxPool2d → Flatten → Linear →
BatchNorm1d → ReLU → LayerNorm → Linear` classifier over raw logits and
the fused loss, trained by `NativeAdam` and resumed **exactly** from one
version-1 checkpoint (all four running-statistic buffers and the
evaluation-mode output included); three saved-resource families
(BatchNorm snapshots, MaxPool2d winners, cross-entropy probabilities)
coexisting in one eval graph and releasing exactly once; buffer versus
parameter mutation attributed to the right cause; the versioning
archetypes; shared and frozen parameters; a non-contiguous NCHW input;
honest per-boundary failure atomicity (BatchNorm transactions are per
module — one whole training step is *not* globally transactional); and
capability/export/artifact guardrails derived from real registries and
files — **tests and documentation only, no capability and no production
change**. And **F9 is complete** — the phase closure: fresh Windows
Release **and** Debug builds (Visual Studio 17 2022, MSVC 19.44.35228.0,
CMake 4.4.0), each configured out-of-source outside the repository with
`TF_BUILD_TESTS=ON` and each passing the **full existing 10-test CTest
suite** (10/10 in 0.78 s and 0.97 s respectively) with **zero project
compiler, linker, and CMake warnings**, the Debug library written
elsewhere so the active runtime stayed the Release DLL (proved by its
linked CRT). A fresh Clang **18.1.3** `-DTF_SANITIZE=address,undefined`
build in WSL2 Ubuntu 24.04.4 with **instrumentation proved rather than
assumed** — `nm -D` shows 22 `__asan*` and 13 `__ubsan*` dynamic symbols
beside the 50 exported `tf_*` symbols, and the library refuses to load
without the sanitizer runtime (`undefined symbol:
__ubsan_vptr_type_cache`). Under
`halt_on_error=1:abort_on_error=1:detect_stack_use_after_return=1:detect_leaks=1`
and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`: **10/10
sanitized native CTests** pass with leak detection on, and every
sanitized Python workload passes — 32 normalization and dependency
suites (**1,968 tests**), the F6 training example reproducing its exact
resume, and the F7 benchmark passing all nine correctness gates —
with **zero ASan and zero UBSan diagnostics attributable to
TensorForge**. Python-level runs preload the ASan runtime (`LD_PRELOAD`)
because the interpreter itself is not instrumented.

**LeakSanitizer, scope stated honestly.** The fully instrumented native
CTest binaries report no leaks at all. A dedicated workload — the
integrated classifier, six training steps, a reporting eval pass with
`native_accuracy`, a version-1 checkpoint saved and loaded into a
**fresh** model/optimizer pair, a resumed step matching the
uninterrupted continuation exactly, a non-contiguous NCHW input through
the whole stack, an eval graph carrying normalization snapshots retained
across one backward and released by the next, and explicit closure of
the optimizers, every unique parameter, and every unique buffer — ends
with the live-native-storage counter back at its baseline (**0 → 0**)
before exit. Running LSan over that *Python* process does report 925,710
bytes in 830 allocations, but **not one leak frame names
`_tensorforge_cpp`, `tf_core_`, `tf_storage_`, or `tf::`**: every site
is CPython, libc, NumPy, `_ctypes`, or the ASan runtime itself —
interpreter and module-initialization allocations a non-instrumented
interpreter never frees at shutdown. **No suppression file was added**,
and the project's leak contract remains the deterministic live-storage
counters and explicit-cleanup tests, which assert an exact return to
baseline. F9 is **validation and documentation only** — no numerical
capability, no C++, no CTest, no ABI or ctypes surface, no example, no
benchmark, and no production behavior changed.

**The current phase is Phase G — Native RNG and Dropout — and it is in
progress; milestones G0, G1, G2, G3, and G4 have landed.** G0 is the architecture
contract, [native_rng_dropout_design.md](native_rng_dropout_design.md),
and it adds **no numerical behavior**. It locks: random state is
**Python-managed** while native random kernels stay **stateless** and
receive the complete key (an unsigned 64-bit seed plus the call index)
for one operation; `NativeGenerator` holds exactly an algorithm
identifier, an algorithm version, a 64-bit seed, and a 64-bit call
counter, owns no native storage and therefore has no `close()`, and
never consults a global or process-wide random source (a `seed=None`
constructor draws once from OS entropy and the value is immediately
explicit and serializable); a deterministic counter-based algorithm with
committed known-answer vectors that must agree between the Windows and
Linux builds; **one successful stochastic forward consumes exactly one
generator call**, while validation, allocation, kernel, and
graph-construction failures consume none — as do evaluation mode,
`p == 0`, and backward; a lock-protected, token-validated reservation
protocol behind that guarantee (one private lock covering reservation,
commit, cancellation, and every counter and state read or write, with
native computation outside it; opaque single-use tokens, so a stale,
foreign, duplicated, or already-finished commit or cancel changes
nothing; at most one live reservation, so a concurrent or reentrant
caller fails **before** an index is minted and no two callers can ever
receive the same call index; exhaustion checked under the lock; and
seed/counter replacement refused while a reservation is live — this is
serialization for correctness, and parallel stochastic execution is
explicitly not claimed); `0 <= p < 1`, with `p == 1` rejected so the
inverted-Dropout scale never divides by zero; one new forward kernel
producing the output **and** a private multiplier mask, with **no**
backward kernel (the gradient is the existing elementwise multiply
against that mask) and logical, stride-independent element indexing; a
differentiable operation whose backward reads only the graph-owned mask —
never the input value and never the generator — so a later generator
change or checkpoint load cannot alter an existing graph; generator state
as a **fourth** `NativeModule` registration category beside parameters,
buffers, and child modules, with its own state section rather than a
tensor-shaped entry in `state_dict()`; native checkpoint **version 2**
whose generator section records the full **alias topology** — every
registered generator path together with its canonical target, so a resume
restores *which layers share a stream* and not merely the saved states,
with a saved-shared/live-independent mismatch (or the reverse) failing in
prevalidation before any live state changes — and which fails loudly
rather than fabricating a seed or counter, with version-1 archives still
loadable into models that have no registered generators; and
**whole-checkpoint transaction atomicity**, where the loader validates
the entire archive, stages every value that can allocate or raise, and
then commits under a single rollback guard, so any ordinary synchronous
failure — and any deliverable asynchronous one, `KeyboardInterrupt`
included — restores parameters, persistent buffers, optimizer state, and
generator state *together*, with every object identity preserved, no
partially loaded component observable, and pre-existing graph-owned masks
untouched, leaving external process or interpreter death as the only
documented exception.

**Milestone G1 is complete** — the generator state foundation, and
nothing numerical. `src/tensorforge/experimental/native_generator.py`
ships `NativeGenerator`: a **pure-Python value holder** owning no native
storage, allocating nothing native, and having **no `close()`**, so
constructing, registering, sharing, and dropping generators leaves the
native live-storage count exactly where it was. It carries the four
locked fields — the algorithm identifier and version, an unsigned 64-bit
seed, and `calls`, the count of **committed** stochastic calls — as
read-only properties, with `state()` returning an independent plain dict
and `load_state()`/`reseed()`/`reset()` validating everything before
assigning anything, so a rejected call leaves the generator
bit-identical. Seeds are exact Python `int`s (`bool`, NumPy integer
scalars, and `int` subclasses are all rejected, and an out-of-range value
raises rather than truncating); `seed=None` draws once through
`secrets.randbits(64)`, and nothing anywhere consults the clock, the
process id, an address, NumPy's global RNG, or Python's `random`.
Identity is object identity — no value equality, and `copy`, `deepcopy`,
and pickle are all refused, because a copied generator would silently
produce the same values in two places. The private call transaction
(`_reserve_call` → `_commit_call`/`_abandon_call`) is implemented exactly
as locked: one plain `threading.Lock` per generator covering reservation,
commit, cancellation, and every state read and write, with the caller's
work done outside it; at most one live reservation, so a concurrent or
reentrant second caller raises **before** an index is minted; opaque
single-use tokens carrying the owning generator, the reserved index, and
a never-reused serial, so stale, foreign, duplicated, already-committed,
and already-cancelled tokens are all inert; commit advancing exactly once
and cancellation never advancing; `load_state`/`reseed`/`reset` refused
while a reservation is live; and exhaustion checked under the lock at
`2**64 - 1`, so the counter never wraps.

`NativeModule` gained `_generators` as a **fourth** registration category
beside parameters, buffers, and child modules: assignment registers (a
`NativeGenerator` is an unambiguous native type), `register_generator` is
the strict explicit form, one name stays one category in both directions,
`__getattr__`/`__delattr__` participate, and
`generators()`/`named_generators()` ride the same deterministic
pre-order, identity-deduplicated, cycle-safe walk parameters and buffers
use — so one shared generator appears once under its first-discovered
canonical name while two generators with identical state stay two
streams. Generator state has its **own** surface,
`generator_state_dict()` / `load_generator_state_dict()`, because
`state_dict()` is contractually `{name: NativeTensor}` and a generator is
not a tensor; `state_dict()` is byte-for-byte unchanged for every model.
Loading runs one shared multi-generator transaction: validate every
value, acquire **every** unique target's lock in a global
identity-ordered sequence, recheck each target for a published
reservation *or a construction claim* while holding them all, snapshot,
then write — the writes being integer assignments that cannot fail, with
the rollback completing before any lock is released. That ordering is
what makes the guarantees real rather than probable: no reservation can
begin on a target between the recheck and the end of the commit, no other
thread can observe a partial commit, and two concurrent loads over
overlapping generators — arriving through different modules whose
canonical orders disagree — acquire in the same sequence and cannot
deadlock.

Reservation creation is itself a **two-phase claim / construct / publish**
transaction, and the token is allocated with **no generator lock held**.
Phase 1 publishes only an internal construction claim; phase 2 builds the
token owning nothing, releasing the claim in `finally` on any failure
(`MemoryError` and `KeyboardInterrupt` included) without publishing a
reservation or skipping a serial; phase 3 publishes the reservation and
advances the never-reused serial exactly once; phase 4 delivers the
token. Those last two fail differently, and clearing the claim covers
only the first two: once a reservation is published the claim is gone, so
an asynchronous exception before the caller receives its token would
leave an active reservation nobody can commit or abandon — permanently
stranding the generator. A failed delivery therefore runs its own
**exact-match** cleanup, cancelling only a live reservation carrying that
token's generator, serial, **and** index, leaving `calls` untouched, and
leaving a newer, foreign, committed, or already-abandoned reservation
strictly alone; it consumes an opaque serial, never a call index, and
takes only its own generator's lock. Token construction is the
one allocation in the path and allocation can run interpreter
finalization, so keeping it outside the lock establishes the governing
invariant — **no user code, callback, or generator-owned allocation runs
while a generator lock is held** — and that is what makes the global
acquisition order unbreakable: a transaction reached from a finalizer
begins owning nothing and takes the same order as any other caller, so
finalizer or callback reentry cannot invert it. The lock stays a
`threading.RLock` for two auditable reasons: structurally, the
multi-generator transaction reaches its targets through the same
`_snapshot_state`/`_assign_state` write seam it holds the locks around,
which a plain `Lock` self-deadlocks on; and residually, CPython may begin
a collection at any container allocation, so a finalizer meeting the
small allocations that remain under the lock gets a deterministic refusal
rather than a permanent hang.

The one registry change is reporting-only: `STATE_SUPPORT`
gained `"generator_state"` beside `"persistent_buffers"`, naming the
**in-memory** generator surface. It does not mean generator state is
checkpointed — that is G5 — and no numerical registry moved.

**G1 generates no random values by itself.** It shipped the state; the
derivation, the mask, and the kernel arrived at G2.

**Milestone G2 is complete — the deterministic stateless Dropout-forward
Core.** It ships, bottom to top:

- `cpp/include/tf_random_internal.h` and `cpp/src/random.cpp` — the exact
  locked `tensorforge.splitmix64` derivation as hidden `namespace tf`
  functions: the `mix64` finalizer
  (`^= >>30`, `* 0xBF58476D1CE4E5B9`, `^= >>27`, `* 0x94D049BB133111EB`,
  `^= >>31`), the per-call stream key
  `mix64(seed + GOLDEN * (call_index + 1))`, the per-element bits
  `mix64(stream + GOLDEN * (element + 1))`, and the uniform
  `(bits >> 11) * 2**-53` compared with a strict `<` against `p`. All
  `std::uint64_t`, wrapping arithmetic only, no `<random>`, no
  `random_device`, no `mt19937`, no clock, no process id, no address, no
  static or thread-local state.
- `tf::dropout_forward_contiguous` — inverted Dropout over one contiguous
  float64 span, writing the output **and** the private multiplier mask in
  one pass, with `1/(1 - p)` computed once per call so the mask holds
  exactly two values.
- `tf_core_dropout_forward` — the self-validating guarded export. Null
  handles, a negative offset or count, a span exceeding its storage, a
  non-finite or out-of-range `p`, and any aliasing between the input and
  either destination are rejected with `TF_ERROR_INVALID`, and a rejected
  call leaves both destinations byte-for-byte unchanged.
- one ctypes declaration carrying the whole random key as two `c_uint64`
  arguments, `"tf_core_dropout_forward"` in the checked-kernel inventory,
  and `"dropout_forward"` in `TENSOR_CORE_OPS`.
- `NativeTensorCore.dropout_forward(p, *, seed, call_index)` and the
  private `_dropout_forward_with_mask` that keeps the mask — the same
  public/private split `maxpool2d_forward` uses for its winner buffer.
- `cpp/tests/test_dropout_forward.cpp`, a dependency-free CTest over both
  layers, and `tests/test_native_dropout_core.py`. Both assert the **same
  committed known-answer vectors** for `mix64`, the stream key, the
  element bits, the uniform conversion, and seven full keep/drop
  patterns, so neither side can redefine the stream alone.

The Core is **stateless**: the complete random key arrives as two
explicit integers, and it reserves, commits, cancels, inspects, and
mutates **no** `NativeGenerator` — a direct Core call leaves a live
generator's seed, `calls`, and reservation slot bit-identical. Randomness
is keyed by the **logical** row-major element index, so a transposed,
narrowed, or nonzero-offset view receives the same mask as a contiguous
tensor of the same logical shape (Policy B materializes the view first,
which is what makes the kernel's flat index the logical index). Both
results are fresh owning contiguous cores that alias neither the input
nor each other; the input is never mutated; and the two-result boundary
is failure-atomic in C++ *and* in the Python wrapper, with native live
storage returning exactly to baseline after every injected failure.

**Milestone G3 is complete — the differentiable `NativeTensor` Dropout.**
It is where the G1 state transaction and the G2 stateless Core finally
meet, and its entire footprint is one method in `native_tensor.py` plus
one name, `"dropout"`, appended to `AUTOGRAD_OPS`. **G3 changed no C++,
no C ABI symbol, no ctypes declaration, and no `NativeTensorCore`
method**, and it added no backward kernel — inverted Dropout's gradient
is the existing `multiply` over the saved mask (design §7.5).

```python
result = tensor.dropout(p, generator=generator)
```

The `generator` is **required and keyword-only**. There is no default,
process-global, or module-global stream, no implicit per-call generator,
and no NumPy or Python `random` fallback; omitting it is a `TypeError`.
`p` goes through the *same* `_normalize_dropout_probability` the Core
uses, so the accepted/rejected matrix is identical by construction rather
than by duplication.

The ordering is the design's §5 transaction, exactly:

1. validate the receiver, the generator, and `p` — nothing has reserved
   or allocated yet;
2. `p == 0` returns `self`, the caller's own object, having built no
   graph node and consumed no call;
3. otherwise reserve **one** call, binding the token and entering the
   cleanup boundary as the very next action;
4. read the key from the **reservation** — the token's index, and the
   seed read while that live reservation makes every state replacement
   raise, so `generator.calls` is never mistaken for the reserved index;
5. run the G2 Core **outside** the generator's lock;
6. build the graph node, with `_from_op` adopting the mask through the
   unchanged `graph_resources` contract;
7. commit — the **last** state-changing action before the return.

So one successful stochastic forward consumes exactly one call, with or
without gradients, and **every** ordinary failure before the commit —
invalid `p` or generator, a closed receiver, an exhausted counter, a
reservation conflict, a Core validation or allocation failure, a Python
wrapper failure, a backward-closure, graph-node, or resource-attachment
failure, a no-grad mask-cleanup failure, or a delivery failure — releases
the result and the mask, cancels the reservation, and re-raises, leaving
the same unconsumed index for the next forward. Two private module-level
seams, `_dropout_backward` and `_deliver_dropout_result`, exist so those
last positions are addressable by a test rather than only by argument —
the same reason G1 has `_deliver_reservation`.

The mask is graph-owned private state, the third member of the family
beside MaxPool2d's winners and cross-entropy's saved probabilities:
released exactly once with the graph history, retained under
`retain_graph=True`, kept alive across a failed retryable backward, freed
by an abandoned graph's `close()`, and closed immediately by a no-grad
forward. Backward reads only the upstream gradient and that mask, so it
records **no** expected parameter version — mutating the input or
reseeding, resetting, or reloading the generator afterwards cannot change
an existing graph's gradient and must not raise — and it consumes no
call, ever.

**Milestone G4 is complete — the `NativeDropout` module and its public
export.** The whole footprint is
`src/tensorforge/experimental/native_dropout.py`, the export, and one
name (`"NativeDropout"`) appended to `NATIVE_MODULES`. **No C++, no C ABI
symbol, no ctypes declaration, no Core method, no autograd operation, and
no checkpoint-format change.**

```python
NativeDropout(p=0.5, seed=None, generator=None)
```

- `p` goes through the **same** `_normalize_dropout_probability` the G2
  Core and the G3 operation use — a third rule would be a third place for
  the matrix to drift.
- `seed` and `generator` are **mutually exclusive**: supplying both is a
  `TypeError`, because a quietly ignored seed is the "looks reproducible,
  is not" failure explicit random state exists to prevent.
- Without an explicit generator the module **creates and owns**
  `NativeGenerator(seed)`; with one it registers **that exact object**,
  never a copy — so two layers deliberately share one interleaved stream
  while the default gives every layer an independent one.
- Everything is validated before a generator is created or registered, so
  a rejected construction draws no entropy, registers nothing, allocates
  nothing, and leaves a supplied generator bit-identical.

The generator is registered under the canonical name `"generator"` as
Phase G's **fourth** state category: it appears in `generators()`,
`named_generators()`, and `generator_state_dict()`, and is deliberately
**absent** from `state_dict()`, which stays contractually
`{name: NativeTensor}`. `load_generator_state_dict()` replaces the state
in place, so identity — and any sharing — survives a load. The module
owns **no native storage**: constructing, registering, running, and
discarding one moves the live-storage count only by the outputs its
forwards return, and dropping the module never closes, resets, or mutates
the generator.

Forward is three cases. Input validation runs **first**, so evaluation is
not a way to hand back a closed or non-`NativeTensor` input. **Training**
delegates to `NativeTensor.dropout`, which owns the entire call
transaction — one call per success, none per failure — so the module can
add no failure hole to a transaction it does not implement.
**Evaluation** returns the **input object itself**, consuming no call and
allocating nothing, so an arbitrary number of eval forwards leaves **no
gap in the stream**: a training forward at index *n*, any number of eval
forwards, then a training forward at index *n + 1*. **`p == 0`** is
identity too, and is deliberately *not* short-circuited in the module —
§6.2 assigns that rule to the operation, and a second copy could only
ever disagree with the first.

**Above the module, nothing exists, and the gap is persistence.**
Milestones **G5–G10 have not
started**: the checkpoint format is still version 1 and does **not**
serialize generator state (that is G5), so saving a model containing a
`NativeDropout` preserves its parameters and buffers and **silently omits
the random stream**, and a load leaves the live generator exactly as it
found it — fabricating nothing, which is the important half. **Exact
stochastic resume therefore does not exist yet.** No capability-registry
value moved beyond that single `NATIVE_MODULES` entry — `dropout` is
still listed
unsupported
beside `float32`, `cuda`, and `amp`. `dropout` stays listed unsupported
for the whole of **G0–G9** — G4 implements and exports `NativeDropout`
without moving the boundary, because a capability whose value is exact
reproducibility is not finished until reproducibility has been shown
under fresh Release and Debug builds and the sanitizers — and the name is
removed only at **G10**, after the closure matrix passes. The format
version moves at **G5**.

## C++ backend — the raw kernel layer (v1.21, historical)

*Historical: this section describes the raw NumPy-buffer kernel layer as
it stood at v1.21, the bottom of the stack. Everything above it — the
strided runtime, broadcasting, autograd, and the training stack — arrived
in later milestones recorded below.*

Proof that Python TensorForge can call compiled C++ code, now with a
small family of kernels:

- `cpp/src/*.cpp` (+ `cpp/include/tf_internal.h`) — plain C-ABI functions
  over float64 buffers, split into coherent translation units
  (`error.cpp`, `storage.cpp`, `elementwise.cpp`, `reduction.cpp`,
  `matmul.cpp`): elementwise add, subtract, multiply, divide, ReLU, a
  naive 2-D matmul (the textbook triple loop, kept as the reference
  kernel), and a tiled matmul (the cache-blocking optimization
  experiment), plus the strided tensor-core kernels, reductions, and the
  thread-local error contract. No Python C-API, no pybind11, no NumPy
  headers. Every fallible export is exception-guarded so no C++ exception
  crosses the ABI (see docs/native_abi_error_contract.md).
- `src/tensorforge/backends/cpp.py` — a ctypes wrapper that loads the
  compiled shared library and exposes the kernels as Python functions,
  handling array conversion and validation on the Python side.

Usage:

```python
from tensorforge.backends.cpp import (
    elementwise_add, elementwise_subtract, elementwise_multiply,
    elementwise_divide, relu, matmul,
)

elementwise_add(np.array([1.0, 2.0]), np.array([3.0, 4.0]))  # [4. 6.]
relu(np.array([-1.0, 0.0, 2.0]))                             # [0. 0. 2.]
matmul(np.ones((2, 3)), np.ones((3, 4)))                     # (2, 4) of 3s
matmul_tiled(np.ones((2, 3)), np.ones((3, 4)))               # same values
```

### Shape/stride metadata (v0.7)

The first step from "raw kernels over buffers" toward a real tensor
runtime foundation. A tensor runtime needs to describe *layout*, not
just hold data: which logical index lives at which storage position.
That is exactly what shape + strides + offset encode — and it is what
makes views, transposes, and slices possible without copying.

```python
from tensorforge.backends.cpp import (
    row_major_strides, numel, is_contiguous_shape, flat_offset, shape_info,
)

row_major_strides((2, 3, 4))            # (12, 4, 1)
numel((2, 3, 4))                        # 24
is_contiguous_shape((3, 2), (1, 3))     # False — transposed layout
flat_offset((1, 2, 3), (12, 4, 1))      # 23
shape_info((2, 3, 4))                   # one dict with all of the above
```

Row-major contiguous layout means the last dimension varies fastest:
element `(i, j, k)` of a `(2, 3, 4)` array lives at flat position
`i*12 + j*4 + k`. Strides here count **elements**, not bytes (unlike
`numpy.ndarray.strides`). Zero-size dimensions are rejected for now;
their conventions deserve their own tested milestone.

This layer is deliberately Python-side: it is the metadata *contract*
that a later native storage object will honor. It is not yet a C++
Tensor object, and it is not connected to Tensor/autograd.

### Native storage (v0.8)

`NativeStorage` is the storage half: a **C++-owned float64 buffer**
behind an opaque handle. Python never sees the raw pointer — data
moves by copy, and the native memory is released explicitly:

```python
from tensorforge.backends.cpp import NativeStorage

with NativeStorage.from_array([1.0, 2.0, 3.0]) as storage:
    storage.size          # 3
    storage.fill(5.0)
    storage.to_numpy()    # a new, independent NumPy copy
    storage.copy_from([7.0, 8.0, 9.0])
# closed on exit; storage.close() works too, and double-close is safe
```

New storage is zero-initialized; operations on a closed storage raise
RuntimeError. `backend_info()` advertises it as ``storage_object``.

Storage by itself has a size but no shape or strides — that binding
is the tensor view's job.

### NativeTensorView (v0.9)

`NativeTensorView` connects the two halves: a `NativeStorage` plus
shape/stride/offset metadata, forming a logical view that knows which
storage element each index means. Its first operation is **contiguous
materialization** — walking the strided view in row-major order and
copying it out — performed by a native odometer loop in C++:

```python
from tensorforge.backends.cpp import NativeStorage, NativeTensorView

view = NativeTensorView.from_array(np.arange(6.0).reshape(2, 3))
view.shape, view.strides, view.contiguous   # (2, 3), (3, 1), True

# The same six values seen transposed, without copying anything:
storage = NativeStorage.from_array(np.arange(6.0))
transposed = NativeTensorView(storage, shape=(3, 2), strides=(1, 3))
transposed.to_numpy()          # the (3, 2) transpose, materialized
copy = transposed.contiguous_copy()  # a new row-major NativeStorage
```

Views are bounds-checked at construction (negative strides included),
so a valid view can never read outside its storage. Views don't own
the storage: close the storage and the view's operations raise.

Views by themselves are building blocks; the runtime object that
composes everything is the tensor core.

### NativeTensorCore (v1.0)

`NativeTensorCore` is the first native tensor runtime object: an
owned `NativeStorage` plus a `NativeTensorView`, composed into one
thing you can create, inspect, materialize, and release:

```python
from tensorforge.backends.cpp import NativeTensorCore

t = NativeTensorCore.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
t.shape, t.strides, t.contiguous   # (2, 3), (3, 1), True
t.to_numpy()                       # fresh (2, 3) NumPy copy
c = t.contiguous_copy()            # a new, independent tensor core
z = NativeTensorCore.zeros((2, 2))
f = NativeTensorCore.full((2, 2), 7.0)
t.close()                          # releases the owned native memory
```

The core owns its storage (context managers work; double-close is
safe; data operations on a closed core raise RuntimeError).
`backend_info()` advertises it as ``tensor_core``.

### TensorCore view operations (v1.1)

The payoff of the shape/stride design: **metadata-only views**. These
return new tensor cores over the *same* storage — nothing is copied
until you materialize with `to_numpy()` or `contiguous_copy()`:

```python
t = NativeTensorCore.from_array(np.arange(6.0).reshape(2, 3))

t.reshape((3, 2))     # same storage, new row-major layout
t.transpose()         # all axes reversed; t.transpose(1, 0) works too
t.T                   # NumPy's .T: reversed axes, no-op for 1-D
t.narrow(1, 1, 2)     # keep 2 positions of dim 1 starting at 1
```

`reshape` requires a contiguous tensor (a non-contiguous layout can't
be reinterpreted by strides alone — materialize first) and the same
element count. `transpose` permutes shape and strides. `narrow`
shrinks one dimension and advances the offset by
``start * strides[dim]``. All of them chain.

**Ownership with shared storage:** the core created by
`from_array`/`zeros`/`full` owns the storage; view cores borrow it.
Closing a view closes only that view — the owner and sibling views
keep working. Closing the owner releases the memory for every core
sharing it, after which their data operations raise.

### Kernels over tensor cores (v1.2)

The step that makes the native runtime self-contained for simple
compute: `relu`, `add`, `subtract`, and `multiply` as
`NativeTensorCore` methods, computed **entirely in C++** — the
kernels read the input's storage plus shape/stride/offset metadata
directly (the same odometer traversal as materialization, so
transposed and narrowed views work without being materialized first)
and write into fresh contiguous native storage. No NumPy round trip
is involved in the arithmetic.

```python
a = NativeTensorCore.from_array([[1.0, 2.0], [3.0, 4.0]])
b = NativeTensorCore.from_array([[10.0, 20.0], [30.0, 40.0]])

a.add(b)            # a new contiguous NativeTensorCore
a.T.multiply(b.T)   # strided views compute directly — no copy first
a.relu()
```

Binary tensor-core operations support NumPy-style **broadcasting** (added
in v1.17; see docs/native_broadcasting_design.md) — compatible shapes are
read through zero-stride broadcast views, nothing is materialized, and
incompatible shapes raise a clear `ValueError`. (The *raw* NumPy-buffer
kernels in `list_kernels()` still require identical shapes; that lower
layer never broadcasts.) Outputs are always new row-major contiguous
tensor cores, independent of their inputs. `backend_info()` reports the
frozen historical five under `tensor_core_kernels` and the complete op
inventory under `tensor_core_ops`, both separate from the raw kernels in
`list_kernels()`.

### TensorCore matmul (v1.3)

The last major compute primitive: `a.matmul(b)` multiplies two 2-D
tensor cores entirely in C++, addressing each source element through
its own strides and offset — so a transposed or narrowed view
multiplies directly, no materialization:

```python
a = NativeTensorCore.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = NativeTensorCore.from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

a.matmul(b)      # (2, 2) contiguous NativeTensorCore
a.T.matmul(c)    # a transposed view multiplies without copying
```

Strictly `(m, n) @ (n, p)`, 2-D only, no broadcasting; the naive
triple loop, matching the reference matmul. The output is a new
row-major contiguous tensor core.

To be precise about what this is **not**: it is not
`tensorforge.Tensor`, it has no autograd, and there is no backend
dispatch. Future milestones may add TensorCore benchmarks, an
explicit backend dispatch design, or optional integration behind a
flag.

### The tiled matmul experiment

`matmul_tiled(a, b, block_size=32)` is the first *optimization*
experiment: same contract and same results as the naive `matmul`, but
computed block by block. Tiling is about **memory locality** — the
naive loop re-reads the same rows and columns from main memory over
and over once matrices outgrow the CPU cache, while the tiled version
works on small sub-matrices that stay cache-resident during their
reuse, and orders its inner loops to walk memory sequentially. Any
positive `block_size` works, including ones that don't divide the
matrix dimensions.

The naive matmul deliberately stays: it is the reference the
experiment is measured against. And the honest caveat applies as ever:
tiling alone is one idea from a long list (SIMD, threading, deeper
blocking...) that libraries behind NumPy implement — NumPy may well
still be faster. Neither matmul is connected to Tensor or autograd.

### Building it

```
uv run python cpp/build.py
```

`cpp/build.py` is a thin wrapper around the canonical CMake build
(`cpp/CMakeLists.txt`). When `cmake` is on PATH it configures and builds
through CMake (which owns the compilation architecture — C++17, per-config
flags, optional sanitizers). When CMake is absent — as on CI, and on
machines with only the bundled `ziglang` compiler — it falls back to a
single direct compiler invocation over the same `cpp/src/*.cpp` sources
(`g++`/`clang++`/`ziglang`). If you have no system compiler, install the
bundled one first:

```
uv sync --group cpp
uv run python cpp/build.py
```

Options: `--debug` builds unoptimized with assertions (`-O0 -g`);
`--no-cmake` forces the direct fallback. CMake developers can also build
with sanitizers (Clang/GCC; not MSVC):

```
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug -DTF_SANITIZE=address
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug -DTF_SANITIZE=undefined
cmake --build cpp/build
```

The compiled library lands next to the wrapper and is gitignored.
Importing `tensorforge.backends.cpp` always succeeds — the library
loads lazily on first use. If the backend is not built, calling a
math kernel raises ImportError with these instructions, and the
backend tests skip. Native failures (e.g. an allocation failure) surface
as ordinary Python exceptions — `MemoryError`, `ValueError`, or
`RuntimeError` — through the ABI error contract
(docs/native_abi_error_contract.md).

### Inspecting the backend

The namespace answers its own questions:

```python
from tensorforge.backends import cpp

cpp.is_available()        # True only if the compiled library loads
cpp.list_kernels()        # ('elementwise_add', ..., 'relu', 'matmul')
cpp.build_instructions()  # how to build it, as a string
cpp.backend_info()        # one dict with all of the above, plus
                          # dtype='float64', device='cpu', the supported
                          # dtype/device sets, and the (false) tensor/
                          # autograd integration flags
cpp.normalize_dtype("float64")  # validated tag; "float32" -> ValueError
cpp.normalize_device("cpu")     # validated tag; "cuda"   -> ValueError
```

`is_available()` performs a real load attempt (cached), not just a
file check, and never raises. Everything here is still experimental
and not wired into Tensor or autograd — `backend_info()` says so
explicitly.

CI does not rely on that skip: the GitHub Actions workflow builds the
backend from source on every run and smoke-tests the compiled kernel
with a hard-failing check before running the suite — so a broken
build or kernel fails CI instead of silently skipping.

### Explicit backend API (v1.5)

The safe, user-facing entry point for backend experiments: name a
backend, get its object. Nothing selects a backend implicitly, and
nothing here touches `tensorforge.Tensor`.

```python
from tensorforge.backends import get_backend, available_backends

available_backends()             # ('numpy', 'native')
numpy = get_backend("numpy")
native = get_backend("native")

numpy.add([1.0, 2.0], [3.0, 4.0])          # a float64 NumPy array
t = native.tensor_from_array([[1.0, 2.0]]) # a NativeTensorCore
native.matmul(t, t.T)                       # -> NativeTensorCore
```

Both backends expose the same small surface — `name`, `available()`,
`backend_info()`, `tensor_from_array`, `to_numpy`, `zeros`, `full`,
`add`, `relu`, `matmul`. The NumPy backend is always available and
follows NumPy semantics; the native backend is constructible whether
or not the compiled library is built (`available()` reports which), and
consumes and produces `NativeTensorCore` objects (its binary ops
broadcast NumPy-style, matching the NumPy backend).

**Conversion boundaries (v1.6).** Data crosses a backend only by
explicit call. `tensor_from_array` *enters* a backend (Python/NumPy
data → a backend-native value, copied); `to_numpy` *exits* it (a
backend-native value → a fresh float64 NumPy array, materialized).
Copies are visible in both directions, so nothing accidentally aliases
native storage, and the native backend rejects anything that is not a
`NativeTensorCore` — including a `tensorforge.Tensor` — with a
consistent TypeError across every operation. Both backends' `add` now
broadcast NumPy-style (the native tensor-core broadcasting landed in
v1.17), so the earlier Stage-1 shape asymmetry between them is resolved;
only the raw NumPy-buffer kernels in `list_kernels()` still require exact
shapes.

This is Stage 1 of a longer plan: how (and whether) backends should
eventually meet `tensorforge.Tensor`, and the risks that gate each
step, are laid out in [dispatch_design.md](dispatch_design.md). The
governing rule is **no implicit fallback**: an unavailable native
operation raises with build instructions; it never quietly falls back
to NumPy.

### Limitations of the raw kernel layer

*These are the limitations of the **raw NumPy-buffer kernels** described
in this section (`elementwise_add` … `matmul_tiled`), which are
deliberately frozen as the reference/benchmark set. Several were lifted
**at higher layers** by later milestones; each is noted.*

- float64 / CPU only. As of v1.21 this is **explicit, inspectable
  metadata** (`dtype`/`device` on the storage, core, and wrapper) rather
  than an unstated assumption, and unsupported dtype/device values are
  rejected at construction — but only `"float64"`/`"cpu"` exist, and
  other inputs are still converted to float64. *(Still true everywhere
  today.)*
- Binary **raw-buffer** operations require identical shapes — no
  broadcasting. `relu` is unary and accepts any shape. *(Lifted above
  this layer: the `NativeTensorCore` binary ops gained full NumPy-style
  broadcasting in v1.17, and `NativeTensor` inherits it.)*
- Division follows IEEE float64 rules (inf/NaN for zero denominators,
  the same values as NumPy) but does not emit NumPy's runtime warning.
  *(`divide` remains a raw kernel only: there is still no `divide`
  tensor operation or backward — `reciprocal` + `multiply` compose what
  the stack needs.)*
- Both matmuls are strictly 2-D — `(m, n) @ (n, p)` only, vectors must
  be passed as `(1, n)` / `(n, 1)` matrices. `matmul` is the naive
  triple loop; `matmul_tiled` adds cache blocking but remains
  single-threaded scalar code. NumPy's BLAS-backed matmul is expected
  to stay faster. *(Still true.)*
- These raw kernels are not connected to Tensor or autograd. *(Still
  true of the raw kernels, and the stable `tensorforge.Tensor` is still
  never wired to any of this. But the native line above them **does**
  have autograd: Phase B built a Python-managed reverse-mode graph at the
  `NativeTensor` layer, and Phases C–D built a training stack on it.)*
- A proof of mechanism, not a performance claim. *(Still true — every
  benchmark on this page is a characterization.)*

## Benchmarks

After building the backend, compare it against NumPy:

```
uv run python benchmarks/cpp_backend.py          # default sizes
uv run python benchmarks/cpp_backend.py --quick  # fast smoke run
```

The suite (v1.4) measures every implementation of each operation
against a shared NumPy baseline: the **raw-buffer kernels** (naive
loops over contiguous NumPy arrays, converted at the call boundary)
and the **NativeTensorCore kernels** (native compute over storage +
shape/stride metadata), including non-contiguous view rows where
transposed inputs feed the kernels directly — and, since v1.12, the
**NativeTensor wrapper** rows on top of those (see below). It is
deliberately **not** a performance claim — the point is what the numbers
teach:

- On small arrays, everything native loses to NumPy: per-call ctypes
  and conversion overhead dominates.
- On large elementwise arrays the raw-buffer loop gets competitive
  with NumPy (both memory-bound).
- The TensorCore rows include what the raw rows don't: Python wrapper
  cost, output-storage allocation, and the generic strided-traversal
  (odometer) loop — visibly slower than the flat raw-buffer loop for
  elementwise work. That overhead is the honest price of layout
  generality, and measuring it is the reason these rows exist.
- For matmul, the triple loop dominates everything else, so the
  TensorCore path costs about the same as the raw naive kernel —
  and NumPy's BLAS beats both by an order of magnitude.

Correctness is verified (each implementation against its own NumPy
reference — view rows compute transposed results) before anything is
timed, and timings are medians over repeated runs after warmup.
Results are hardware-dependent and should not be oversold; expect
exact numbers to vary, the *shape* of the story shouldn't.

### Native training stack — guardrails and Phase C completion (v3.15)

v3.15 is the **Phase C completion milestone**: a hardening,
integration, verification, and documentation pass that **adds no
numerical behavior** — no new operations, kernels, layers, losses, or
optimizer features, and no source change to the native compute stack.
It closes **Phase C — the native training stack — in code**, completing
the Phase A → Phase B → Phase C arc.

What it delivered:

- **One cross-cutting completion test file**
  (`tests/test_native_phase_c.py`, selector `-k "native_phase_c"` — 10
  integrated tests) that complements rather than duplicates the
  per-component suites (`test_native_sgd`/`adam`/`optimizer_state`/
  `checkpoint`/...). Each test spans several components and locks an
  invariant no single unit test covers: the full **NativeSGD** and
  **NativeAdam** training lifecycles under a NumPy tripwire (finite
  loss, meaningful reduction, version deltas equal to the active
  update count, graph-free independently-owned optimizer state,
  `close()` releasing only optimizer-owned moments while the model
  stays trainable); the **shared-parameter** story end to end (one
  `NativeParameter` through two registered aliases and two forward
  paths → one `parameters()` entry, one `state_dict` key, accumulated
  backward, one SGD update, one Adam moment pair and counter, one
  optimizer-snapshot entry, one checkpoint entry, and an
  alias-preserving restore whose continuation matches bit for bit);
  **mixed active/frozen/`grad=None`/zero-gradient** collections across
  both optimizers (active and present-zero-gradient parameters advance
  version/counter, frozen and gradient-less ones never age, a frozen
  parameter's closed/invalid gradient is never inspected, and a
  later-activated parameter takes its correct first or resumed step);
  **repeated optimizer-state** and **checkpoint-resume** cycles (old
  internal state closed after replacement, no caller snapshot aliasing
  any live storage, no parameter version moved by optimizer loading,
  and bit-identical two-lineage continuation); **failure recovery** at
  the step-staging, state-load-staging, checkpoint-save, and
  checkpoint-corruption boundaries (each leaving values, versions,
  moments, counters, and gradients unchanged, temporaries and
  temporary files cleaned up, and a later valid operation succeeding);
  the **four-way graph-staleness distinction** (an optimizer step and a
  model-state load make an old value-sensitive graph stale; an
  optimizer-state load and a *failed* checkpoint load do not; a
  *successful* checkpoint restoration does — with gradients untouched
  whenever the stale detector raises); **lifetime/close discipline**
  (caller-owned model and optimizer snapshots, optimizer-owned
  moments, idempotent close, no reliance on garbage collection); and
  the **public surface** (exactly the twelve intentional
  `tensorforge.experimental` exports, no leak into the stable
  namespace, no optimizer base class, no checkpoint leak into stable
  serialization, no unsupported optimizer feature, no native CNN or
  CUDA/dtype surface).
- **Documentation completion**: this page, the README, the project
  summary, the architecture doc, the roadmap, the release history, and
  the design doc all mark Phase C **complete**; the
  [native support matrix](native_support_matrix.md) is finalized as the
  authoritative Phase A–C snapshot with an explicit phase-status
  header; and the documentation guardrails (`tests/test_docs.py`) gain
  checks that Phase C can never silently revert to "in progress", that
  the README keeps presenting native training and checkpointing, and
  that optimizer state and file resume are never described as future
  work — while CNN and CUDA stay in the unsupported/future sections.
- **Audits with no change needed**: CI already builds the backend from
  source every run, hard-fails the smoke check before pytest, and runs
  the full suite (so the native tests execute rather than skip);
  `.gitignore` already covers the compiled library, caches, and build
  output; the examples (`native_mlp_training.py`,
  `native_checkpoint_resume.py`) use only public APIs, close what they
  own, and leave no artifacts (the checkpoint example writes into a
  self-cleaning temporary directory); the benchmark remains a
  characterization with no speed assertion and still runs after Phase
  C. No genuine correctness defect was found — nothing blocks the
  completion.

The verified suite stands at **1365 tests** (1353 plus the 10
cross-cutting completion tests and 2 new documentation guardrails).
Phase A, Phase B, and **Phase C are complete**; the
next major native phase is the **native CNN stack** (Phase D), whose
architecture contract is locked in
[native_cnn_design.md](native_cnn_design.md) (milestone D0) and whose
first milestone **D1 has shipped**: `NativeFlatten`, a parameter-free,
buffer-free batch-preserving flatten Python-composed from the existing
`reshape`/`contiguous_copy` operations and their autograd (no new C++
kernel, no custom backward), returning an independent owning result so
it composes safely in a `NativeSequential`. **D2–D6** then built the
**differentiable native Conv2d operation**: internal CPU float64
forward/input-gradient/weight-gradient compute kernels
(`tf::conv2d_forward_contiguous` and the two `*_backward_contiguous`
kernels, hidden C++ symbols), their exception-guarded C ABI wrappers
(`tf_core_conv2d_forward`, `tf_core_conv2d_input_backward`,
`tf_core_conv2d_weight_backward`) with ctypes/`errcheck` registration, the
`NativeTensorCore.conv2d_forward`/`conv2d_input_backward`/
`conv2d_weight_backward` Core methods (Policy-B copy-then-compute for
non-contiguous operands, fresh owning contiguous outputs matching the
stable Conv2d to tolerance), the bias gradient composed from the existing
native `sum` reduction (no dedicated kernel), and the Python-managed
**`NativeTensor.conv2d`** autograd primitive (input/weight/bias gradients,
conditional stale-value version tracking, failure rollback). **D7** then
shipped the trainable **`NativeConv2d`** module built on that operation: an
OIHW weight / optional `(O,)` bias `NativeParameter` layer with
deterministic uniform conv fan-in initialization (`bound =
1/sqrt(in_channels·kh·kw)`, local-RNG, global state untouched), 4-D NCHW
input validation, and backward supplied entirely by the D6 autograd — no
new kernel, C ABI symbol, or custom module backward. It registers through
the inherited `NativeModule` mechanism (deterministic `weight` then `bias`
order), rides the existing state_dict/checkpoint/`NativeSGD`/`NativeAdam`
paths unchanged, and drops into a `NativeSequential`. **D8** then added the
forward-only **max-pooling** layer: the internal
`tf::maxpool2d_forward_contiguous` kernel (a hidden C++ symbol in the new
`cpp/src/pooling.cpp`, producing the pooled values *and* the saved winner
indices in one deterministic row-major pass), its exception-guarded
`tf_core_maxpool2d_forward` export with ctypes/`errcheck` registration, and
the Core method `NativeTensorCore.maxpool2d_forward` (Policy-B
copy-then-compute, failure-atomic output + **private** winner-buffer
allocation, exact parity with the stable `MaxPool2d`). The winner buffer —
an internal float64 buffer of flat plane offsets with a `-1` padding
sentinel, proved exact against `H*W ≤ 2^53` both in Python and at the ABI —
is deliberately invisible: no public tensor, no new dtype, no state-dict or
checkpoint presence. **D9** then completed the differentiable pooling
operation: the internal scatter-add `tf::maxpool2d_backward_contiguous`,
the exported guarded `tf_core_maxpool2d_backward` wrapper (which validates
every winner value — the `-1` sentinel or an exact in-range integer —
before scattering, and never rounds a malformed one),
`NativeTensorCore.maxpool2d_backward`, and the Python-managed
**`NativeTensor.maxpool2d`** autograd node. Backward reads **only** the
saved winners and the upstream, so it takes no kernel/stride/padding at any
layer, records **no** parameter-version snapshot (unlike `conv2d`, whose
gradients reread operand values), and is unaffected by input mutation after
the forward pass; overlapping windows accumulate and padding winners drop
their gradient. The winner buffer is owned by the graph history and
released exactly when it is — freed by a one-shot `backward()` or
`close()`, retained under `retain_graph=True`, and kept alive across a
failed (retryable) backward — through the small
`_from_op(..., graph_resources=...)` hook D9 added rather than any autograd
redesign. **D10** completed the native CNN *layer* set with
**`NativeMaxPool2d`**: a parameter-free, buffer-free `NativeModule` that
normalizes `kernel_size`/`stride`/`padding` to two-element tuples
(`stride=None` ⇒ non-overlapping windows) and delegates forward entirely to
that operation — no new kernel, ABI symbol, ctypes declaration, custom
backward, parameter, buffer, `return_indices`, or checkpoint schema. It
holds no winner storage between calls (each forward's winners belong to
that call's output graph), contributes no state-dictionary or checkpoint
keys, and drops into a `NativeSequential` beside
`NativeConv2d`/`NativeReLU`/`NativeFlatten`/`NativeLinear`, where
`NativeSGD`/`NativeAdam` ignore it naturally because it owns nothing
trainable. The native module inventory is therefore `NativeModule`,
`NativeLinear`, `NativeReLU`, `NativeFlatten`, `NativeConv2d`,
`NativeMaxPool2d`, and `NativeSequential`. **D11** then proved the
complete native CNN stack **trains end to end**:
`examples/native_cnn_training.py` composes
Conv2d → ReLU → MaxPool2d → Flatten → Linear over eight fixed 6×6 images
whose target is the strongest bright-to-dark vertical edge — a genuinely
spatial, non-linear rule the convolutional path is required for — and
trains it with `NativeMSELoss` and `NativeAdam(lr=0.05)` for 40
deterministic steps, dropping the loss from 0.771306 to 0.011085 (98.6%)
with finite, nonzero gradients reaching every Conv2d and Linear parameter
on the first backward. A run interrupted at step 15, saved to one
pickle-free checkpoint (model **and** optimizer state) and resumed into a
completely fresh model/optimizer pair, reproduces the uninterrupted run
**exactly** — loss history, final predictions, every parameter value, and
every optimizer state entry — because the CPU float64 kernels are
deterministic and nothing random happens between checkpoint and resume.
The archive carries only persistent state (no pooling winners, no graph
history, no gradients), the format version is unchanged, and the live
native-storage count is exactly constant across repeated steps. D11 added
no kernel, ABI symbol, operation, loss, optimizer, or schema. **D12 closed
Phase D**: cross-cutting integration tests spanning several CNN components
at once (`tests/test_native_phase_d.py`), honest CNN characterization
benchmarks (`benchmarks/benchmark_native_cnn.py` — conv/pool forward,
forward+backward, end-to-end training step, and a stable-framework
reference, measurement only), **ASan/UBSan validation** of the whole
native CNN stack under Clang 18 on Linux with no TensorForge diagnostic
(and a LeakSanitizer pass over the instrumented native CTests),
documentation reconciliation, and durable capability guardrails replacing
the milestone-era wording pins. **Phase D is complete**; the native line's
next phase after it was **Phase E — Native Classification and Stable
Math**, which has since completed (E0–E10), followed by **Phase F —
Native Normalization and Stateful Buffers**, which has since
**completed** too (F0–F9). Further activations and
math, dropout with a native RNG, and a CPU optimization pass sit beyond
Phase F, followed by
the CUDA
runtime, dtype/AMP work, Transformer/text experiments, distributed
training, and the final portfolio release. Still float64/cpu only, still explicit and
experimental, and no production performance is claimed.

### Native training stack — checkpoint files and deterministic resume (v3.14)

v3.14 makes native training runs **persistable**:
`tensorforge.experimental.save_native_checkpoint(path, model,
optimizer=None, metadata=None)` and `load_native_checkpoint(path,
model, optimizer=None)` write and restore **one explicit, pickle-free
NPZ archive** over the existing state contracts (the v3.3 module state
dictionary and the v3.13 optimizer state), following the stable
framework's no-pickle serialization philosophy. No new C++ work and no
new operations — NumPy appears strictly as the file-format boundary
(`to_numpy`/`from_array`, `np.savez`, `np.load(...,
allow_pickle=False)`); the framework's numerical computation stays
native (tripwire-tested). Deliberately **not** shipped: scheduler
state, random-state capture or restoration, dataloader state, multiple
models/optimizers, partial or name-remapped loading, `strict=False`,
`map_location`, checkpoint merging, incremental/sharded checkpoints,
compression, encryption, URLs, pickle, arbitrary object metadata, or
graph/gradient serialization. The stable `tensorforge.serialization`
is untouched and neither line accepts the other's objects.

```python
from tensorforge.experimental import (
    save_native_checkpoint, load_native_checkpoint,
)

save_native_checkpoint(path, model, optimizer=optimizer,
                       metadata={"steps": 6})
metadata = load_native_checkpoint(path, fresh_model,
                                  optimizer=fresh_optimizer)
```

**The archive** (format `"tensorforge.native_checkpoint"`, format
version 1) holds one `manifest` entry — a JSON document encoded as
UTF-8 bytes in a 1-D uint8 array, never an object array and never
pickled — plus one float64 array per model parameter and per Adam
moment under deterministic zero-padded indexed names
(`model::000000`…, `optimizer::m::000000`…, `optimizer::v::000000`…).
The manifest maps each canonical model state key to its array name and
exact shape/dtype/device, reproduces the optimizer's v3.13 schema
(type tag, state format version, hyperparameters, positional parameter
metadata, Adam step counts and moment array names), and carries the
user metadata. Nothing volatile is written: no Python `id()`, pointer,
repr, gradient, parameter version, autograd graph, or closed flag.
Duplicate array references, missing arrays, and unreferenced extra
entries are all rejected on load. **Metadata** must be recursively
JSON-compatible (`None`, `bool`, `int`, finite `float`, `str`, lists —
tuples normalize to lists, the stable `json.dumps` convention — and
str-keyed dicts); NaN/infinity, bytes, sets, NumPy scalars and arrays,
tensors, modules, optimizers, non-string keys, and cyclic containers
are rejected before any file is created, and
`load_native_checkpoint()` returns an independent plain-Python dict.

**Saving is validate → snapshot → write-atomically.** Path, open
`NativeModule`, optimizer (`None`, `NativeSGD`, or open `NativeAdam`
whose unique parameter sequence is positionally *identical by object
identity* to the model's — an unrelated optimizer is rejected), and
metadata are validated first; the model and optimizer are snapshotted
through their existing `state_dict()` contracts and converted through
the explicit `to_numpy()` boundary, with every caller-owned snapshot
closed in a `finally`; then the archive is written to a
collision-safe temporary file in the destination directory
(`np.savez` onto an explicitly opened handle, so NumPy can never
silently rename it) and committed with one `os.replace` — an existing
destination is replaced atomically on success and stays byte-intact
on failure, no temporary file survives either way, and a pre-write
failure leaves no destination created and every live object untouched
and usable. Same-filesystem atomicity only; no directory creation.

**Loading is validate → stage → commit**, with **strict optimizer
presence**: an archive holding optimizer state requires a compatible
optimizer of the same type, and an archive without one rejects a
supplied optimizer — a resume can never silently discard or invent
optimizer state (a deliberate divergence from the stable
`load_checkpoint`, which ignores archive optimizer state when none is
passed; there is no `model_only` flag). Phase 1 validates everything
with no live mutation: the live model and optimizer, then the archive
under `allow_pickle=False` — manifest presence/representation/UTF-8/
JSON/root type, exact format identity and version, exact field sets
at every level, model keys against the live model's canonical names,
every array's exact float64 dtype and shape against both manifest and
live destination (object dtype is impossible to read), and the
optimizer section through the same validators the optimizer
constructors use — so after preflight and staging, neither component
loader has an ordinary public failure path left. Phase 2 stages
independent `NativeTensor` copies (a failure closes them all); Phase 3
commits through the existing public loaders only —
`NativeModule.load_state_dict()`, then `optimizer.load_state_dict()` —
and closes every staged tensor in a `finally`. Committed behavior is
exactly the components' documented contracts: model loading increments
each parameter version once and makes old value-sensitive retained
graphs stale; optimizer loading moves no versions, and `NativeAdam`
installs its own additional independent internal copies, so no live
state ever aliases an archive array or staged tensor. Every ordinary
failure — thirty-plus corruption cases are locked by tests, from
invalid ZIP data and malformed UTF-8/JSON through wrong dtypes,
duplicate references, presence/type mismatches, and invalid optimizer
scalars — happens before any mutation, preserving model values,
versions, gradients, optimizer moments (by identity and value),
counters, and usability. One narrow, honest limitation: the model
commit and the optimizer commit are two separate Python operations —
an asynchronous interruption between them can leave the model restored
while optimizer state remains old, and an interruption inside either
component keeps that component's own documented window; no private
rollback is manufactured.

**Deterministic file resume is proven end to end**
(`examples/native_checkpoint_resume.py`): train the native MLP with
NativeAdam for N steps → save → restore into a completely fresh
model/optimizer pair → continue both runs on identical data —
bit-identical losses, parameter values, m/v moments, and step counts,
with version deltas matching (model loading adds its one documented
increment; optimizer loading adds none) and no residue beyond the
checkpoint file. NativeSGD round-trips its lr to an identical next
step, shared-alias models round-trip once per unique parameter, and
model-only checkpoints work with `optimizer=None` on both sides.

Everything above is locked by `tests/test_native_checkpoint.py`
(selector: `-k "native_checkpoint"` — 17 tests, including the
corruption matrix, atomic-overwrite and cleanup proofs, a NumPy
tripwire over save/load, and source-level no-pickle/no-eval/no-exec
guardrails with `allow_pickle=False` asserted), and the full suite
passes at **1353 tests**. Still float64/cpu only, still explicit and
experimental, and no cross-device, cross-dtype, or general
checkpoint-compatibility claims are made. The next milestone is
**Advanced C++ v3.15 — Phase C guardrails and completion**.

### Native training stack — the optimizer state contract (v3.13)

v3.13 makes native optimizer state **snapshot-able and restorable in
memory**: `NativeSGD.state_dict()`/`load_state_dict()` and
`NativeAdam.state_dict()`/`load_state_dict()` over one small versioned
schema, built entirely from the existing runtime (no new C++ work, no
new operations — snapshot and load copies run on the same native copy
path the module state contract uses). Deliberately **not** shipped:
`save_checkpoint`/`load_checkpoint`, file serialization of any kind
(`.npz`, JSON, pickle, metadata files, paths), `map_location`, RNG or
scheduler state, model checkpoint wrappers, `strict=False` or
compatibility modes, parameter remapping by name, parameter groups, or
an optimizer base class. Native checkpoint archives and deterministic
file resume are v3.14. The stable optimizers are untouched.

```python
state = optimizer.state_dict()      # a plain in-memory dict
optimizer.load_state_dict(state)    # restore into a compatible optimizer
```

**One schema, format version 1.** Every state dict carries
`format_version` (the int `1`), the exact `optimizer` type tag
(`"NativeSGD"` / `"NativeAdam"` — a state can never load into the
wrong optimizer), the validated hyperparameters, and `parameters` — a
tuple of `{"shape", "dtype", "device"}` dicts, one per unique stored
parameter, **in the optimizer's deterministic identity-deduplicated
first-occurrence order** (shared aliases appear once because the
optimizer already deduplicated). Mapping across optimizer instances is
purely **positional**: entry *i* is validated against the loading
optimizer's *i*-th stored parameter — no Python `id()`, pointer,
module name, or repr is serialized, and no parameter values,
gradients, graph data, or closed-state flags appear. NativeSGD's state
is pure Python metadata (`lr` + the schema); NativeAdam's adds
`betas`, `eps`, `step_counts` (a tuple of per-parameter counters), and
`m`/`v` — lists of **caller-owned snapshots**: plain graph-free
`requires_grad=False` NativeTensors (never `NativeParameter`), each an
independent owning contiguous native copy sharing storage with
nothing — not the optimizer's internal moments, not the parameters or
gradients, not each other. The caller releases them (`close()` each)
when done; repeated `state_dict()` calls return independently owned
snapshots, and closing a snapshot never affects the optimizer.

**`state_dict()` preflights, then copies, then returns.** The
optimizer must be open (NativeAdam), every stored parameter open,
every internal moment open and metadata-matched, and every counter a
valid non-negative int; a violation raises deterministically before
anything is created. If snapshotting fails partway, every snapshot
created by that call is closed before the error propagates — never
left to garbage collection — with internal state, parameters, and
gradients untouched and the optimizer still usable.

**`load_state_dict(state)` is validate → stage → commit.** The input
is caller-owned and read-only: never mutated, retained, adopted, or
consumed. Phase 1 validates everything with no mutation — optimizer
open, stored parameters open, current internal state intact, exact key
set (missing and unexpected reported together), exact format version
and tag, hyperparameters under the constructors' full contracts,
positional parameter metadata (exact count and shape/dtype/device; no
casting, reshaping, broadcasting, or device movement), step counts
(non-bool non-negative ints), and every `m`/`v` entry an **open plain
NativeTensor** of exactly the parameter's metadata (NativeParameter
and stable-`Tensor` entries rejected; sequence fields accept tuple or
list; errors name stable field paths like `state['m'][1]`, never
addresses). Phase 2 stages an independent optimizer-owned native copy
of every input moment — a failure closes every staged copy and
changes nothing. Phase 3 commits lr/betas/eps, the counters, and the
staged moments, closing the replaced old internal buffers only after
the new state is installed; NativeSGD's whole commit is one `lr`
assignment. Ordinary failure at any phase preserves scalar
hyperparameters, internal moment identities and values, counters,
parameter values/versions/gradients, registrations, the caller's
input, and optimizer usability. One narrow, honest limitation:
NativeAdam's commit is several Python attribute assignments that
cannot be made indivisible, so an asynchronous interruption (e.g.
KeyboardInterrupt) mid-commit could leave scalars replaced but
moments not — each installed piece stays internally consistent, and
no private rollback is manufactured.

**Optimizer state never touches parameters or graphs.** Loading moves
no parameter value, version, gradient (by identity or value),
`requires_grad`, registration, alias, or model state key — the v3.7
stale guard keys on parameter versions, which these methods never
move, so a retained valid graph stays valid across optimizer-state
loading (proven by a focused test). **Deterministic in-memory
continuation** is proven end to end: train a
Linear→ReLU→Linear/MSE/NativeAdam model for N steps, snapshot with
`NativeModule.state_dict()` + `NativeAdam.state_dict()`, restore both
into a fresh model/optimizer pair, continue both runs M further steps
on identical data — bit-identical losses, parameter values, moments,
and step counts, with model loading incrementing each parameter
version exactly once (its existing contract) and optimizer loading
incrementing none. Frozen, `grad=None`, shared-alias, zero-state, and
late-activated parameters round-trip exactly (a parameter restored at
counter 0 takes its first bias-corrected update at t = 1; a restored
counter continues at t = count + 1).

Everything above is locked by `tests/test_native_optimizer_state.py`
(selector: `-k "optimizer_state"` — 21 tests, tripwire-tested
NumPy-free snapshot/load paths plus source-level no-file/no-pickle
guardrails), and the full suite passes at **1336 tests**. Still
float64/cpu only, still explicit and experimental. The next milestone
is **Advanced C++ v3.14 — native checkpointing and deterministic
resume**: a file archive over the module and optimizer state
contracts, followed by v3.15 — Phase C guardrails and completion.

### Native training stack — NativeAdam (v3.12)

v3.12 adds the native adaptive optimizer:
`tensorforge.experimental.NativeAdam` — minimal, correct Adam over
`NativeParameter` objects, built entirely from the existing runtime
(no new C++ work, no new operations) on the v3.7 mutation contract and
the v3.11 `sqrt`/`reciprocal` primitives. Deliberately **not**
shipped: weight decay, AMSGrad, parameter groups, per-parameter
learning rates, schedulers, optimizer `state_dict`/`load_state_dict`
(optimizer-state serialization is v3.13), checkpointing/resume,
general tensor division, fused optimizer kernels, in-place tensor
arithmetic, a global no-grad context, or an optimizer base class. The
stable `tensorforge.optim.Adam` is untouched and fully separate.

```python
from tensorforge.experimental import NativeAdam

optimizer = NativeAdam(model.parameters(), lr=0.001,
                       betas=(0.9, 0.999), eps=1e-8)
loss.backward()
optimizer.step()       # bias-corrected Adam via copy_value_
optimizer.zero_grad()  # gradients persist until this
optimizer.close()      # releases the optimizer-owned moment state
```

**Parameters follow the NativeSGD contract exactly**: the iterable is
materialized once, every entry validated as an open `NativeParameter`
(position-named errors), deduplicated strictly by object identity in
first-occurrence order (duplicates and shared aliases: one entry, one
state slot, one update, one version increment per step; equal values
never merge), empty collections rejected, strong references stored,
nothing copied or owned. **Hyperparameters are validated, never
repaired**: `lr` and `eps` must be real (`numbers.Real`, `bool` and
coercibles rejected), finite, and strictly positive; `betas` must be a
tuple or list of exactly two reals, each finite with
`0.0 <= beta < 1.0`; all normalized to Python floats after validation
and exposed read-only.

**The optimizer owns its state, explicitly.** After all validation,
one entry per unique parameter is allocated **eagerly**: first and
second moments as plain graph-free `NativeTensor` zeros (never
`NativeParameter`; fresh owning contiguous storage of exactly the
parameter's shape/dtype/device; never registered in a module, never in
`model.state_dict()`), plus a per-parameter integer step counter —
exposed read-only as `step_counts`, aligned with `parameters()`. A
constructor failure mid-allocation releases every buffer created so
far and touches no user parameter or gradient. `close()` (idempotent;
`with` blocks work) releases every owned moment exactly once and makes
`step()`/`zero_grad()` reject deterministically, while parameters and
gradients stay caller-owned and open; the plain-Python introspection
surface (`parameters()`, `lr`/`betas`/`eps`, `step_counts`) remains
readable after close. There is no reliance on garbage collection.

**`step()` is two-phase and mutation-atomic on its public failure
surface**, extending the NativeSGD design with state. Preflight:
optimizer open, every stored parameter open, every entry's m/v open
and metadata-matched; frozen parameters skipped *before* their
gradients are examined (a frozen parameter's stale or even closed
gradient is never inspected); `grad is None` skipped; every active
gradient validated (open, exact shape/dtype/device). Staging, per
active entry and entirely at the autograd-unaware `NativeTensorCore`
level (no graph node, no NumPy, Python scalar exponentiation only for
the bias-correction coefficients, reciprocal-times instead of
division):

    t      = previous_step + 1
    m_new  = beta1 * m + (1 - beta1) * g
    v_new  = beta2 * v + (1 - beta2) * (g * g)
    m_hat  = m_new * reciprocal(1 - beta1 ** t)
    v_hat  = v_new * reciprocal(1 - beta2 ** t)
    update = lr * m_hat * reciprocal(sqrt(v_hat) + eps)
    parameter_new = parameter - update

`eps > 0` keeps the denominator positive even at `v_hat = 0`. Any
preflight or staging failure — a later entry's bad gradient or
corrupted state included — closes every staged temporary and changes
no value, version, moment (by identity and value), counter, or
gradient; the same optimizer recovers on a later valid step. Commit,
per active entry in stored order: `copy_value_(parameter_new)`
(identity, registration, aliases, `requires_grad`, and the gradient
preserved; version +1), install the staged `m_new`/`v_new` as the new
optimizer-owned state, commit the step count, then close the replaced
old moments; the staged `parameter_new` never persists and is always
closed. **Per-parameter step counters** mean a skipped parameter never
ages its moments — a parameter that becomes active later takes its
first bias-corrected update at `t = 1` — while a present zero-valued
gradient is active and advances state, counter, and version. Two
narrow, honest limitations are documented rather than papered over
with private rollback: an asynchronous interruption (e.g.
KeyboardInterrupt) between two commits leaves the collection partially
advanced (each committed entry internally consistent), and within one
entry an interruption between the parameter commit and the state
installation — two Python operations that cannot be made indivisible —
would advance the parameter but not its moments and count.

**Gradients are retained** until `zero_grad()` (open-optimizer +
all-parameters-open preflight, never a partial clear; values,
versions, moments, and counters untouched; no `set_to_none`). **v3.7
staleness applies unchanged**: a value-sensitive graph built before
`step()` raises the existing deterministic stale error afterwards with
every gradient untouched, and a fresh forward/backward trains on the
updated values — verified end to end by a 20-iteration deterministic
`NativeSequential(NativeLinear → NativeReLU → NativeLinear)` +
`NativeMSELoss` training run (finite losses, >50% reduction, version
and step-count deltas equal to the update count, gradients cleared at
the end).

Everything above is locked by `tests/test_native_adam.py` (selector:
`-k "native_adam"` — 33 tests, tripwire-tested NumPy-free step and
zero_grad paths, with a plain-NumPy oracle mirroring the native
composition to 1e-15), and the full suite passes at **1315 tests**.
Still float64/cpu only, still explicit and experimental. The next
milestone is **Advanced C++ v3.13 — native optimizer state**:
serializable optimizer `state_dict`/`load_state_dict` over the same
contract, followed by v3.14 — native checkpointing and deterministic
resume, and v3.15 — Phase C guardrails and completion.

### Native training stack — optimizer math primitives (v3.11)

v3.11 adds the two reusable native unary operations **NativeAdam
(v3.12) will need for its square-root denominator and reciprocal
scaling** — `sqrt` and `reciprocal` — through the complete stack: two
new C++ kernels per operation (a generic strided odometer and a
contiguous fast path, sharing relu's signature and dispatched by the
same contiguity rule, bit-for-bit identical), ctypes bindings,
`NativeTensorCore.sqrt()`/`.reciprocal()`, and differentiable
`NativeTensor.sqrt()`/`.reciprocal()`. Deliberately **not** shipped:
NativeAdam itself, general tensor `divide` (a raw ctypes
`elementwise_divide` kernel still exists at the kernel layer only —
`reciprocal` + `multiply` compose everything the training stack
needs, so no division operation was added), `exp`/`log`/`tanh`/
`sigmoid`/`softmax`, `rsqrt`/`abs`/`power`, operator overloads,
optimizer state, checkpointing, or dtype expansion. The raw-kernel
registry boundary is untouched (like sum/mean before them, the new
core kernels join neither the raw-buffer kernel list nor the locked
`tensor_core_kernels` tuple).

Forward contract: shape/dtype/device-preserving, scalars through
arbitrary strided/offset views read directly (transposes, narrows,
nonzero offsets — never materialized), fresh owning row-major
contiguous outputs, inputs never mutated or aliased, closed inputs
rejected, float64/cpu only. **Exceptional values follow IEEE float64**
(documented in the kernels and locked by tests): `sqrt` of a negative
is NaN (no exception), signed zeros are preserved (`sqrt(-0.0)` is
`-0.0`), `+inf → +inf`, NaN propagates; `reciprocal` maps `±0.0 →
±inf` and `±inf → ±0.0` (no exception, no warning — the same values
NumPy produces), NaN propagates.

**Autograd via saved forward results.** Both derivatives are functions
of the output — `d(sqrt(x))/dx = 1/(2·sqrt(x)) = 0.5 ·
reciprocal(out)` and `d(1/x)/dx = −1/x² = −out²` — so each backward
closure reads the **recorded forward output** (a closed saved output
fails deterministically with the graph intact) and never the parent's
current value, computing entirely at the autograd-unaware core level
with transient cores closed as they are consumed. The v3.7
classification follows from what the callbacks actually read: **both
operations record no expected parameter versions** — mutating a direct
`NativeParameter` input after forward leaves these edges valid, and
the gradient remains mathematically correct for the forward that was
recorded (a mixed graph is still guarded by its sensitive edges:
`sqrt(p) * p` goes stale through the multiply). No existing
classification changed. Saved outputs are graph-owned: one-shot
cleanup releases the closure that holds them, `retain_graph=True`
preserves them for repeated passes, and failed backwards free nothing.

Everything above is locked by `tests/test_native_optimizer_math.py`
(selector: `-k "native_optimizer_math"` — 18 tests: kernel symbols and
the untouched registry boundary, core forward over
contiguous/scalar/transposed/narrowed/offset/combined views, the
exceptional-value table, wrapper graph construction, exact analytical
gradients, explicit upstreams, chain and shared-subgraph accumulation,
central finite differences on domain-safe inputs (contiguous and
strided), graph lifetime including the closed-saved-output failure,
the version-independence contract and the still-guarded sensitive
edge, a NumPy tripwire, and scope boundaries), and the full suite
passes at **1282 tests**. The next milestone is **Advanced C++ v3.12 —
NativeAdam**, followed by v3.13 — native optimizer state, v3.14 —
native checkpointing and deterministic resume, and v3.15 — Phase C
guardrails and completion.

### Native training stack — integration checkpoint (v3.10)

v3.10 is the **first major native CPU training checkpoint** — an
integration, documentation, CI-audit, and public-surface milestone
that adds **no numerical behavior**: no new operations, kernels,
layers, losses, or optimizer features, and no source change to the
native stack. It prepares `advanced/cpp-backend` for its first
reviewable pull request into `main`.

What it delivered:

- **One canonical [native support matrix](native_support_matrix.md)**
  stating exactly what the native line supports (runtime/metadata, the
  twelve differentiable operations, the autograd engine's guarantees,
  the training stack) and exactly what it does not (no native divide/
  sqrt/reciprocal/exp/log/tanh/sigmoid/softmax, no adaptive optimizer,
  no optimizer state or checkpointing, no native CNN stack, no CUDA,
  float64/cpu only, no dispatch into the stable Tensor) — linked from
  the README, project summary, and architecture docs.
- **README, project summary, and architecture corrections**: the
  README no longer claims "no C++ backend yet" or that the experiment
  merely "started" — it now presents both lines honestly, with a
  native capability section, a native quickstart, and an accurate
  limitations section; docs/architecture.md documents the native
  execution path (Python native modules → NativeTensor + Python-managed
  graph → NativeTensorCore → ctypes → C++ CPU kernels) and the absolute
  stable/native separation; docs/project_summary.md covers both lines
  in two minutes.
- **Documentation guardrails** (tests/test_docs.py): the README can
  never again claim the native backend is absent while still being
  required to mark CUDA as future work; the support matrix must keep
  covering the shipped surface and keep unshipped work in its
  unsupported section; the experimental exports are locked to the nine
  intentional names and proven never to leak into the stable top-level
  namespace.
- **Audits with no change needed**: CI already builds the backend from
  source every run, hard-fails a smoke check before pytest, and runs
  the full suite (so native tests execute rather than skip);
  `.gitignore` already covers the compiled library, caches, and build
  directories; the experimental exports were already complete and
  intentional. No genuine defect was found — nothing blocks the PR.

The verified suite stands at **1264 tests** (1262 plus the net new
documentation/export guardrails). Phase A and Phase B are complete;
**Phase C is *not* complete at this checkpoint** — it continues with
v3.11 (native optimizer math primitives), v3.12 (NativeAdam), v3.13
(native optimizer state), v3.14 (native checkpointing and
deterministic resume), and v3.15 (Phase C guardrails and completion),
followed by the native CNN stack, the CUDA runtime, dtype/AMP work,
Transformer/text experiments, distributed training, and the final
portfolio release.

### Native training stack — the MLP training proof (v3.9)

v3.9 is the first complete multi-iteration **native CPU training
proof** — an integration milestone, not a framework-feature milestone:
`examples/native_mlp_training.py` trains a genuine MLP end to end
through the existing experimental native training stack, with **zero
changes to the native runtime, autograd engine, layers, loss, module
system, parameter system, or NativeSGD**. Deliberately **not** shipped:
`NativeAdam`, momentum, weight decay, schedulers, optimizer state,
checkpointing/resume, batching, shuffling, validation sets, metrics,
classification losses, new operations, or performance claims.

```
uv run python examples/native_mlp_training.py
```

**The proof.** `NativeSequential(NativeLinear(2, 8, seed=0),
NativeReLU(), NativeLinear(8, 1, seed=1))` trains on 8 fixed synthetic
regression samples (2 features → 1 target; Python literals handed once
to `NativeTensor.from_array` — data construction at the explicit entry
boundary, deliberately distinguished from native computation) for **25
steps of `NativeSGD(lr=0.1)`**. Every iteration follows the fresh-graph
lifecycle: confirm gradients cleared → fresh forward through the model
→ scalar `NativeMSELoss` → record via `to_numpy()` (the established
inspection exit — the only NumPy anywhere) → one-shot `backward()` →
confirm every parameter has a finite gradient → `optimizer.step()`
(identities stable, one version increment each, gradients retained) →
`optimizer.zero_grad()` → close the per-iteration prediction and loss
tensors. No `retain_graph`, no graph reuse — so the v3.7 stale guard
never triggers in the loop (one concise negative test proves
deliberately retaining an old sensitive graph across `step()` still
raises the existing stale error). The loss decreases **monotonically
every step**, from 2.107864 to 0.009529 — a 99.5% reduction — and the
whole run is bit-deterministic: repeated runs in one process reproduce
the exact loss history, final parameter values, and version history
(`[N, N, N, N]` after N steps, `zero_grad` and evaluation passes adding
nothing). Lifetime is explicit: the model parameters, optimizer, and
fixed data live for the whole run; per-iteration tensors are closed
every iteration; and everything the run created — parameters and data
included — is closed on the way out, success or failure, with
`train()` returning plain Python values only (never live tensors).

Everything above is locked by `tests/test_native_mlp_training.py`
(selector: `-k "native_mlp_training"` — 13 tests: end-to-end loss
behavior, per-parameter learning, exact version progression,
identity/name/state-key stability, exact-equality determinism across
runs, hand-driven gradient lifecycle, cross-iteration accumulation
control, the stale-graph negative guard, a NumPy-compute tripwire over
a full training run, a source-level guardrail keeping the example
inside the contract, and the executable report itself), and the full
suite passes at **1262 tests**. Still float64/cpu only, still explicit
and experimental, and no performance is claimed. The next milestone is
**Advanced C++ v3.10 — NativeAdam**: a second native optimizer with
per-parameter adaptive state over the same v3.7 mutation contract —
scope to be pinned when it begins; the training proof stays SGD-based.

### Native training stack — NativeSGD (v3.8)

v3.8 adds the first native optimizer:
`tensorforge.experimental.NativeSGD`, minimal stochastic gradient
descent — `value ← value - lr * grad` — over `NativeParameter` objects,
committed entirely through the v3.7 mutation contract. Pure Python over
existing kernels: no new C++ work, no new operations. Deliberately
**not** shipped: momentum, dampening, Nesterov, weight decay, parameter
groups, per-parameter learning rates, schedulers, optimizer
`state_dict`/checkpointing, `NativeAdam`, a training loop, or the
multi-iteration MLP training proof (that is v3.9). Fully separate from
the stable `tensorforge.optim` — neither accepts the other's objects,
and the stable framework is untouched.

```python
from tensorforge.experimental import NativeSGD

optimizer = NativeSGD(model.parameters(), lr=0.01)
loss.backward()
optimizer.step()       # value ← value - lr * grad, via copy_value_
optimizer.zero_grad()  # gradients persist until this
```

**Parameter storage.** The constructor materializes the iterable
exactly once (lists, `model.parameters()`, generators), validates every
entry as an open `NativeParameter` (position-named errors; plain
tensors, stable-framework objects, non-iterables, and empty collections
rejected), and deduplicates **strictly by object identity** in
first-occurrence order — duplicate references and shared-module aliases
become one stored entry, one update, and one version increment per
step; equal values are never merged. The optimizer stores strong
references and owns nothing: it never copies, replaces, or closes a
parameter, and constructor failure touches nothing. `parameters()`
returns a snapshot list of the exact stored objects; `lr` is a
read-only float.

**Learning rate.** `lr` must be a real number (`numbers.Real`; `bool`
explicitly rejected even though it subclasses `int`, as are strings and
arbitrary float-coercible objects), finite, and strictly positive —
NaN, ±infinity, zero, and negatives raise; nothing is clamped or
rewritten; the accepted value is normalized to a Python float only
after validation.

**`step()` is two-phase and mutation-atomic on its public failure
surface.** Phase 1: every stored parameter is preflighted as still
open; the active set is selected (frozen `requires_grad=False`
parameters are skipped *before* their gradients are examined — a frozen
parameter with a stale gradient never updates; `grad is None`
parameters are skipped); every active gradient is validated as an open
`NativeTensor` of exactly the parameter's shape/dtype/device (no
broadcasting, reshaping, casting, or device movement; index-named
deterministic errors); and every updated value is staged **natively at
the NativeTensorCore level** — the same autograd-unaware layer the
engine's backward math uses, so `step()` can never build a graph node,
never touches NumPy, and produces fresh owning staged tensors
independent of every parameter and gradient. Any phase-1 failure
(closed parameter, closed/mismatched gradient, native staging failure)
releases all staged temporaries and changes **no value, no version, and
no gradient** — the same optimizer recovers completely afterwards.
Phase 2 commits each staged value through
`NativeParameter.copy_value_()` in stored order: parameter identity,
registrations, aliases, `requires_grad`, and the gradient (by identity
and value) are preserved, each updated parameter's version increments
exactly once (a numerically unchanged update — a zero gradient — still
increments: the owned value was replaced), and staged temporaries are
released on every exit path. One narrow, honest limitation: after a
fully successful preflight the commits cannot fail through any public
surface, but an asynchronous interruption (e.g. KeyboardInterrupt)
between two commits would leave earlier parameters updated and later
ones not — each individual commit stays atomic and version-consistent,
and no private rollback is manufactured for that window.

**Gradients and `zero_grad()`.** `step()` never clears, replaces,
mutates, or closes a gradient — gradients persist until `zero_grad()`,
which preflights every stored parameter as open (failing before any
clearing; never a partial clear) and then delegates to each parameter's
own `zero_grad()` — values, versions, identities, `requires_grad`, and
registrations untouched; frozen parameters included harmlessly. No
`set_to_none` option in this milestone.

**v3.7 integration.** An optimizer step is exactly the mutation the
version contract guards: a value-sensitive graph (multiply/matmul/relu
edges) built before `step()` becomes stale after it — the next
`backward()` raises the existing deterministic stale-value error with
every gradient untouched — and a fresh forward/backward trains on
against the updated values. The staleness classification is unchanged.
Verified end to end through
`NativeSequential(NativeLinear → NativeReLU → NativeLinear)` +
`NativeMSELoss`: one forward → loss → backward → `step()` (exact SGD
arithmetic on all four parameters, identities stable, versions +1,
gradients retained by identity and value) → `zero_grad()` → a fresh
forward/backward — deliberately **one** verified step, not a training
loop.

Everything above is locked by `tests/test_native_sgd.py` (selector:
`-k "native_sgd"` — 19 tests, tripwire-tested NumPy-free step and
zero_grad paths), and the full suite passes at **1249 tests**. Still
float64/cpu only, still explicit and experimental. The next milestone
is **Advanced C++ v3.9 — the first end-to-end native training proof**:
a small deterministic multi-iteration forward → loss → backward →
`step()` → `zero_grad()` regression over the existing
Sequential/Linear/ReLU/MSE/SGD surface, asserting learning without
fragile exact-loss values — no new operations, layers, losses, or
optimizer features.

### Native training stack — parameter mutation safety and versioning (v3.7)

v3.7 makes identity-preserving parameter mutation **safe**: it adds
value-version counters to `NativeParameter`, one controlled no-grad
mutation primitive (`copy_value_`), version increments on state loading,
and stale-graph detection in `backward()` — the required foundation
before `NativeSGD` can update parameters. Pure Python over existing
kernels: no new C++ work, no new mathematical operations. Deliberately
**not** shipped: `NativeSGD`, any optimizer, training loops, momentum,
weight decay, general in-place arithmetic (`add_`/`sub_`/`mul_`),
operator overloads, a global no-grad context, an active-graph registry,
or serialization. The stable framework is untouched.

```python
from tensorforge.experimental import NativeParameter, NativeTensor

p = NativeParameter([[1.0, 2.0], [3.0, 4.0]])
p.version                                   # 0 — read-only, monotonic
p.copy_value_(NativeTensor.from_array(...)) # controlled no-grad mutation
p.version                                   # 1
```

**The version contract.** Every parameter carries a read-only,
monotonically increasing `version` (an ordinary non-negative int, 0 at
construction) that counts **replacements of the owned numerical value**
and nothing else: `copy_value_` increments it by exactly one, and a
successful `load_state_dict` increments each matched canonical parameter
by exactly one (a shared parameter loads once under its canonical key —
one increment observed through every alias; numerically identical values
still increment, because replacement, never value equality, is what
counts). Gradient accumulation, `zero_grad()`, registration/aliasing/
removal/replacement in modules, train/eval, and `state_dict()` snapshots
never move it, and any failed mutation or failed load leaves every
version (and value, core, identity, and gradient) exactly unchanged —
versions move only after a load's whole commit has succeeded, so the
existing rollback restores a consistent world by construction.

**The mutation primitive.** `parameter.copy_value_(source)` replaces the
owned value with an independent owning contiguous **native** copy of an
open `NativeTensor` source (a `NativeParameter` — the parameter itself
or a snapshot included — is accepted purely as a value source; stable
`Tensor`/`Parameter`, arrays, lists, and scalars are rejected; exact
shape/dtype/device required — no broadcasting, reshaping, casting, or
device transfer; strided views materialize; the result is never
aliased). It runs entirely outside autograd — no graph node, no parents,
no backward callback, the parameter stays a graph-free leaf — and
preserves Python identity, registrations and aliases, `requires_grad`,
and the existing gradient by identity and value. Old storage is released
exactly once after commit. This is the whole in-place surface: the exact
path `NativeSGD.step()` will commit updates through.

**Stale-graph detection.** From the dependency audit of every backward
callback: `multiply`, `matmul`, and `relu` read direct-parent forward
*values* in backward; `add`, `subtract`, `sum`, `mean`, `reshape`,
`transpose`/`T`, `contiguous_copy`, and `narrow` read only shape/stride/
reduction metadata. So graph construction records
`(op, parameter, expected_version)` for every **direct `NativeParameter`
operand of a value-sensitive op** (the deliberate op-level policy: a
parameter operand is guarded even in the corner where sibling
`requires_grad` flags mean its value would not actually be read —
safety, simplicity, and independence from gradient-flow details over
that corner's convenience), and `backward()` validates every recorded
version after the freed-graph scan and **before** the seed, any
callback, or any gradient commit. A mismatch raises a deterministic
RuntimeError naming the operation and the expected/current versions,
distinct from the freed-graph and closed-tensor errors: gradients (by
identity and value), graph structure, and versions are untouched, the
graph is *not* freed, repeats fail identically, `retain_graph` does not
help, and — versions being monotonic — loading the old numerical value
back never revives the graph. The remedy is one thing: run forward
again. Value-independent graphs (bias through `add`, view/reduction
chains) remain valid across mutation with mathematically correct
gradients. Version metadata lives on the Python graph node (a new
`NativeTensor` slot, `()` on leaves), is released with the graph on
one-shot cleanup, survives `retain_graph=True` (retained graphs
re-validate on every pass), and detection keys on the parameter's
`_version` slot — autograd never imports the module stack and no global
graph registry exists. Mutation itself is never blocked: safety is
enforced when an old graph attempts backward.

**The future NativeSGD pattern (v3.8)**: backward completes → the
optimizer computes a graph-free updated value with native operations →
commits it via `copy_value_` → identity stable, version +1, gradients
retained until `zero_grad()` → old sensitive graphs become stale → the
next iteration builds a fresh graph.

Everything above is locked by
`tests/test_native_parameter_versioning.py` (selector:
`-k "native_parameter_version or stale_parameter_graph or mutation_safety"`
— 34 tests, tripwire-tested NumPy-free mutation/loading/preflight
paths), plus one intentionally tightened v3.3 assertion (a graph built
before loading now raises the stale error instead of silently reading
the new value — still memory-safe), and the full suite passes at
**1230 tests**. Still float64/cpu only, still explicit and experimental.
The next milestone is **Advanced C++ v3.8 — NativeSGD**, narrowly
scoped to: an optimizer over identity-deduplicated `NativeParameter`
objects, real positive finite learning-rate validation, `step()`
committing graph-free native updates through the v3.7 mutation path
(`grad=None` and frozen parameters skipped, identity preserved, one
version increment per updated parameter), `zero_grad()`,
duplicate/shared-parameter protection, and deterministic update tests —
**no momentum, weight decay, or parameter groups initially, and no
training loop** (the first end-to-end model-training proof may remain
v3.9; SGD is not combined with the full MLP training example).

### Native training stack — NativeMSELoss (v3.6)

v3.6 adds the first native loss:
`tensorforge.experimental.NativeMSELoss`, a **parameter-free**
`NativeModule` computing mean squared error as a pure composition of
existing differentiable operations — no new C++ work, no fused loss
kernel, no manual backward, and no NumPy in the analytical forward or
backward (tripwire-tested). Deliberately **not** shipped: other losses,
optimizers, `NativeSGD`, parameter updates, mutation-version counters,
training loops, or serialization. `NativeTensor`, `NativeModule`, the
existing layers, and the stable framework's `mse_loss` are all
untouched.

```python
from tensorforge.experimental import NativeMSELoss

loss_fn = NativeMSELoss()                 # reduction="mean" (default)
loss_fn = NativeMSELoss(reduction="sum")  # the only other option
loss = loss_fn(prediction, target)        # scalar NativeTensor, shape ()
loss.backward()                           # existing autograd end to end
```

**Forward is exactly** `difference = prediction.subtract(target)`;
`squared = difference.multiply(difference)`; then `squared.mean()` or
`squared.sum()`. Both reductions are **scalar**, so the existing default
backward seed applies (an explicit scalar upstream gradient scales both
input gradients per the engine's normal rules). The analytical gradients
— `dL/dprediction = 2(prediction−target)/N` and its negation for the
target under `mean` (drop `/N` under `sum`) — are supplied entirely by
the existing graph: `multiply`'s **duplicate-parent accumulation** on
the shared difference node produces the factor 2, `subtract`'s backward
produces the target's sign, and **`mean`'s existing native backward
produces the `1/N` scaling** — no division operation exists or is
needed. Verified by exact references (1-D, multidimensional
total-element scaling, zero-difference, positive/negative differences,
explicit upstream scaling, one-sided and both-frozen operands,
branching, duplicate-parent identity in the graph) **and central finite
differences** for prediction and target under both reductions
(`eps=1e-6`, `atol=1e-6`).

**Reduction contract** (deliberately small): exactly `"mean"` and
`"sum"`, validated in the constructor by exact string match — case
variants, whitespace variants, non-strings, and unsupported strings are
rejected, nothing is normalized — and stored as constructor
configuration, never model state. No `"none"` in this milestone: both
supported reductions are scalar, which is sufficient for the first
native training loop; unreduced losses belong to a later, broader loss
API.

**Input contract**: two open `NativeTensor`s (a `NativeParameter` is
accepted as the subclass it is and accumulates native-backed gradients;
the stable framework's `Tensor`, NumPy arrays, lists, scalars, and
closed tensors are rejected with errors naming *which* argument is
invalid) of **exactly equal shape — no broadcasting** (checked before
any graph node is built, even though `subtract` could broadcast; a
silently broadcast loss hides target-shape bugs; the error names both
shapes), with exact dtype/device equality via the metadata contract.
Shape-generic across every supported rank; zero-element tensors cannot
be constructed by the native runtime (`NativeStorage` requires positive
size), a limitation NativeMSELoss simply inherits. Inputs are never
mutated, reshaped, cast, or copied; the module stores no temporary
tensors and owns no storage; `state_dict()` is empty (unexpected keys
follow the v3.3 strict/non-strict rules; `reduction` never appears);
`training` propagates normally and never affects numerics; graph
lifetime (one-shot, `retain_graph`, freed-history errors, unchanged
gradients after a failed reuse) holds through the loss.

**Model integration** is verified end to end against an exact reference:
`NativeSequential(NativeLinear → NativeReLU → NativeLinear)` +
`NativeMSELoss` produces the correct scalar loss and exact gradients for
the input, every weight and bias (ReLU masking included), and a
gradient-requiring target; recursive `model.zero_grad()` clears the
model while the target's gradient stays independent; repeated fresh
forward → loss → backward → `zero_grad` cycles reproduce bit-identical
gradients; frozen models leave a trainable input learning. The
v3.3–v3.5 **mutation boundary is unchanged**: parameter mutation or
state loading between forward and backward remains mathematically
inconsistent (no version counters yet — exactly what v3.7 addresses).

Everything above is locked by `tests/test_native_mse_loss.py`
(selector: `-k "native_mse_loss"` — 27 tests), and the full suite
passes at **1196 tests**. Still float64/cpu only, still explicit and
experimental. The next milestone is **Advanced C++ v3.7 — Native
Parameter Mutation Safety and Versioning Contract**, narrowly scoped
to: version counters on mutable native parameter values, forward-time
expected-version capture where backward needs saved parameter values,
state loading incrementing parameter versions, clear stale-forward
backward errors, a controlled no-grad parameter mutation primitive, the
identity-preserving update foundation for `NativeSGD`, and rollback and
shared-parameter behavior — **no optimizer and no training loop yet**.
v3.7 must precede `NativeSGD` because optimizer updates cannot safely
mutate parameter values while old graphs remain capable of backward;
mutation versioning is not combined with SGD.

### Native training stack — NativeReLU and NativeSequential (v3.5)

v3.5 completes the first composable native model surface with one
parameter-free activation module and one ordered composition container:
`tensorforge.experimental.NativeReLU` and
`tensorforge.experimental.NativeSequential`. Both subclass
`NativeModule` and compose only existing APIs — no new C++ work, no
fused kernels, no manual backward anywhere, and no NumPy in forward or
backward (tripwire-tested). Deliberately **not** shipped: losses,
optimizers, training loops, serialization, mutation-version counters,
other activations, or other layers. `NativeModule`, `NativeTensor`,
`NativeParameter`, `NativeLinear`, and the stable framework are all
untouched.

```python
from tensorforge.experimental import (
    NativeLinear, NativeReLU, NativeSequential, NativeTensor,
)

model = NativeSequential(
    NativeLinear(3, 4, seed=0),
    NativeReLU(),
    NativeLinear(4, 2, seed=1),
)
out = model(x)                     # (batch, 3) -> (batch, 2)
out.sum().backward()               # existing autograd end to end
model.zero_grad(); model.eval()    # inherited recursive contracts
model.state_dict()                 # {"0.weight", "0.bias", "2.weight", "2.bias"}
```

**NativeReLU** is a parameter-free module whose `forward(input)`
validates an open `NativeTensor` (framework `Tensor`, arrays, lists,
scalars, and closed tensors rejected; `NativeParameter` accepted as the
subclass it is, returning a plain `NativeTensor`) and delegates to the
existing `NativeTensor.relu()` — **shape-generic** (every rank and
strided/offset layout `relu()` supports), no in-place mode or `inplace`
argument, no copies or reshapes, dtype/device preserved. Its backward is
entirely the existing fused native relu autograd, including the existing
gradient-at-exactly-zero rule (blocked — tested, not changed); graph
lifetime is untouched; `state_dict()` is empty; `training` propagates
normally but never affects numerics.

**NativeSequential(*modules)** chains children registered under
**contiguous integer-string slots** `"0"`..`"len-1"` — execution order
*is* the registered order (a single source of truth; no separate child
list). The constructor accepts zero or more `NativeModule`s (nested
sequences and custom subclasses included), validates **all** entries
before registering any, and rejects tensors, parameters, stable-framework
modules, callables, `None`, and lists. The container surface is
deliberately minimal: `len()`, iteration in execution order (shared
modules **not** deduplicated — iteration mirrors execution),
`seq[i]` with real ints (bool rejected, Python-style negatives,
IndexError out of range, exact object returned), `seq[i] = module`
(replacement keeps the slot's name and position), and `append(module)`
(next contiguous slot; returns `self` for chaining). The slot invariant
is enforced at the registration funnel, so **the registry and the
execution order can never silently diverge**: gap-producing indices,
non-canonical digit strings (`"01"`), non-slot child names (which would
register a child that never executes), direct `NativeParameter`
assignment (a sequence composes modules — parameters live in children),
slot removal in any form (`None` assignment, ordinary-value overwrite,
`del`, `add_module(name, None)`), and direct self-insertion are all
rejected with clear errors; ordinary non-module attributes stay normal
attributes. Module traversal remains cycle-safe by v3.2, but *executing*
a deliberately cyclic composition is meaningless and unsupported.

**Shared modules — the load-bearing distinction: execution is
position-based, ownership is identity-based.** The same child in two
slots executes twice in `forward`, while `modules()` /
`named_modules()` / `named_parameters()` / `state_dict()` / `train()` /
`zero_grad()` keep the v3.2/v3.3 identity-deduplicated contracts — the
first slot is the canonical path, shared parameters appear once, and a
duplicate alias key in a loaded state is *unexpected* under the strict
rules.

**Forward is pure composition**: each child is called through its normal
`__call__`/`forward` contract and validates its own input (a
`NativeLinear` slot enforces its strictly 2-D contract when reached;
child exceptions propagate unchanged); the container adds no node, copy,
or graph machinery of its own, and an **empty sequence returns the input
by identity**. The composed graph is exactly the children's own autograd
graphs, so one-shot cleanup, `retain_graph`, freed-history errors, and
the v3.3/v3.4 mutation boundary (forward → backward → zero_grad/state
update after graph completion) hold end to end — verified through a
Linear→ReLU→Linear model with exact analytical references **and central
finite differences** for the input, both weights, and both biases
(`eps=1e-6`, `atol=1e-6`, all hidden pre-activations kept ≥ 0.1 from
ReLU's zero boundary). State keys derive from slot names
(`"0.weight"`, `"0.bias"`, `"2.weight"`, `"2.bias"`; nested sequences
nest: `"0.0.weight"`; ReLU contributes none; bias-free layers omit bias
keys), and loading preserves every v3.3 guarantee.

Everything above is locked by `tests/test_native_relu.py` and
`tests/test_native_sequential.py` (selector:
`-k "native_relu or native_sequential"` — 52 tests), and the full suite
passes at **1169 tests**. Still float64/cpu only, still explicit and
experimental. The next milestone is **Advanced C++ v3.6 —
NativeMSELoss**, narrowly scoped to: an MSE loss as a `NativeModule`
with prediction/target `NativeTensor` validation, an exact-shape
contract initially, the smallest justified reduction surface (scalar
mean by default), forward composed from the existing native
`subtract`/`multiply`/`sum`/`mean`, exact and finite-difference
gradient tests — **no optimizer or training loop yet** (NativeMSELoss
is not combined with SGD or model training).

### Native training stack — NativeLinear (v3.4)

v3.4 adds the first concrete native layer:
`tensorforge.experimental.NativeLinear`, a fully connected layer built
entirely on the completed contracts — `NativeModule` (v3.2) holding
`NativeParameter`s (v3.1), forward as pure existing `NativeTensor`
operations, backward supplied by the existing Phase B autograd, and
state handled by the v3.3 state dictionary. It deliberately ships **no**
`NativeSequential`, activation modules, losses, optimizers, training
loops, mutation-version counters, or serialization. No C++ changed, no
kernel or symbol was added, `NativeTensor`/`NativeParameter`/
`NativeModule` are untouched, and `tensorforge.nn.Linear` is untouched.

```python
from tensorforge.experimental import NativeLinear, NativeTensor

layer = NativeLinear(in_features, out_features, bias=True,
                     *, seed=None, requires_grad=True)
output = layer(x)   # x: 2-D NativeTensor (batch_size, in_features)
```

**Weight orientation** (load-bearing for future checkpoints): `weight`
is `(in_features, out_features)` — the same `x @ weight` orientation as
the stable framework's Linear — so the strictly 2-D native matmul
applies directly; `bias` is `(out_features,)`, broadcast over the batch
by the existing zero-stride broadcast. The forward is exactly::

    output = input.matmul(weight)            # bias=False
    output = input.matmul(weight).add(bias)  # bias=True

Because forward is a composition of existing differentiable operations,
**the existing autograd engine is the backward implementation** — there
is no manual or fused NativeLinear backward, and graph lifetime
(one-shot default, `retain_graph=True`, freed-history errors) is
unchanged. Gradient shapes: `input.grad` `(batch, in)`, `weight.grad`
`(in, out)`, `bias.grad` `(out,)` (the broadcast-add backward reduces
over the batch via the native `unbroadcast`). Every gradient is verified
against exact analytical formulas **and central finite differences**
(`eps=1e-6`, float64-appropriate tolerances; NumPy is the test-side
reference only).

**Constructor.** Every Python argument is validated before any native
allocation: `in_features`/`out_features` are real positive ints (bools
and integer-like objects rejected), `bias` and `requires_grad` are real
bools, `seed` is `None` or a real int. `requires_grad=False` freezes
both parameters — they stay registered, traversable, and in
`state_dict()`, accumulate no gradients, and a requiring input still
receives its gradient. **Initialization** is deterministic and
self-contained: weight and bias sampled uniformly from
`[-1/sqrt(in_features), +1/sqrt(in_features)]` (a basic fan-in bound) by
a **local** `numpy.random.default_rng(seed)` — an int seed reproduces
values exactly, `None` draws fresh entropy, and the global NumPy RNG is
never read or mutated. NumPy appears only as host-side initialization
data preparation (the established `from_array` entry boundary) — never
in forward or backward computation (guarded by a tripwire test).
Parameters are created by assignment (`self.weight = NativeParameter(...)`,
then `self.bias = ...` or `None`), exercising v3.2 registration and
fixing the deterministic order `["weight", "bias"]` everywhere
(`named_parameters()`, `parameters()`, `state_dict()`; nested as
`"layer.weight"`/`"layer.bias"`). With `bias=False` the attribute reads
as `None` and only `"weight"` exists.

**Input contract** (strictly 2-D for now): an open `NativeTensor` of
shape `(batch_size, in_features)` with matching dtype/device
(float64/cpu). Nothing is wrapped, reshaped, flattened, or broadcast
implicitly — the stable framework's `Tensor`, NumPy arrays, lists,
scalars, closed inputs, and wrong ranks/feature sizes are rejected with
errors naming the expected 2-D shape, the expected feature count, and
the actual shape. The output is an ordinary `NativeTensor` (never a
parameter), requiring grad exactly when a participating operand does;
forward does not depend on `training` mode.

**State compatibility** follows v3.3 exactly: loading a compatible state
changes values while weight/bias identity, gradients, `requires_grad`,
and frozen status survive; loading a biased state into a bias-free layer
reports `"bias"` *unexpected*, the reverse reports it *missing* (strict
raises before mutation; non-strict returns the key lists);
shape-incompatible states fail atomically. **The v3.3 mutation boundary
applies unchanged**: the supported sequence is forward → backward →
(optionally) load/update after the graph completes; loading between
forward and backward is memory-safe but mathematically inconsistent (no
version counter — deliberately out of scope).

Everything above is locked by `tests/test_native_linear.py` (selector:
`-k "native_linear"` — 42 tests), and the full suite passes at **1117
tests**. Still float64/cpu only, still explicit and experimental, still
no implicit dispatch and no NumPy compute in any native forward,
backward, or state path. The next milestone is **Advanced C++ v3.5 —
NativeReLU and NativeSequential**, narrowly scoped to: a `NativeReLU`
module wrapping the existing `NativeTensor.relu()`, a `NativeSequential`
ordered child-module container with integer-string child names,
deterministic recursive traversal, forward composition, shared-module
behavior, train/eval propagation, and state_dict compatibility
(replacement/indexing only if tightly justified) — **no loss, optimizer,
or training loop yet** (v3.5 is not combined with losses, SGD, or model
training).

### Native training stack — native state dictionary contract (v3.3)

v3.3 adds the deterministic **in-memory** model-state contract:
`NativeModule.state_dict()` and `NativeModule.load_state_dict()`. The
scope is deliberately narrow — **parameters only, in memory only**: no
buffers, optimizer state, training/RNG state, file formats, archives, or
checkpoint metadata (file serialization and checkpoints are later
milestones that will consume this contract). No C++ changed;
`NativeTensor` is untouched; `tensorforge.nn` is untouched.

```python
state = model.state_dict()      # {canonical_name: independent NativeTensor}
result = model.load_state_dict(state)          # strict=True by default
result.missing_keys, result.unexpected_keys    # immutable, deterministic
model.load_state_dict(partial_state, strict=False)
```

**`state_dict()`** returns an insertion-ordered `dict[str, NativeTensor]`
whose keys are exactly the v3.2 canonical `named_parameters()` names:
dot-separated, direct parameters before descendants, **shared parameters
once under their first-discovered path**, frozen parameters included,
cycle-safe, deterministic across calls (an empty module returns an empty
mapping). Every value is an ordinary **graph-free, `requires_grad=False`
NativeTensor holding an independent owning contiguous copy**, computed by
the native copy path (`zeros` + native `add` — no NumPy computes or
copies state values). Snapshot and model share no mutable native storage
in either direction: mutating, replacing, or closing a model parameter
never affects an existing snapshot, closing a snapshot never affects the
model, and a snapshot outlives the model if the caller keeps it (the
caller releases snapshot values with `close()` when done). Gradients,
`requires_grad`, training flags, aliases, and registrations are neither
included nor touched. A closed registered parameter makes `state_dict()`
raise clearly (naming the key) — snapshots half-built inside the failed
call are closed before the error propagates, never returned, and earlier
snapshots stay valid.

**`load_state_dict(state_dict, strict=True)`** copies values **into the
existing `NativeParameter` objects** — never assigning new objects — so
every identity-derived contract survives loading: `id(parameter)`,
registration, canonical names and traversal order, shared-parameter
aliasing (one canonical key updates the single shared object once and
every alias observes it; a supplied alias key is *unexpected*),
`requires_grad`/frozen state, leaf/graph-free status, and each existing
`grad` **by identity and value** (`None` stays `None`; loading never
clears, replaces, or accumulates gradients — the exact-shape rule keeps
an existing gradient compatible). `training` flags are untouched. It
returns an immutable `LoadStateDictResult(missing_keys, unexpected_keys)`
(missing in canonical order, unexpected in input order; both empty on a
strict success).

Validation happens **entirely before mutation**, in a documented order:
`strict` must be a real bool (TypeError otherwise); the input must be a
mapping (snapshotted once, so exotic mapping subclasses cannot shift
mid-load); keys must be strings; missing/unexpected keys are computed
and, under `strict=True`, any incompatibility raises one ValueError
reporting **both** lists; then every matching value is preflighted — it
must be an open `NativeTensor` (a `NativeParameter` is accepted purely as
a value source and copied, inheriting no identity or graph state;
`tensorforge.Tensor`/`Parameter`, arrays, lists, and scalars are rejected
— nothing is converted silently) whose shape/dtype/device **exactly**
match the open destination parameter, with every error naming the key.
There is no broadcasting, reshaping, casting, truncation, partial
copying, or device movement. **Atomicity**: independent native copies of
all matching values are *staged* before anything changes (a staging
failure closes the staged copies and changes nothing); the *commit* then
swaps each parameter's core for its staged copy — pure reference
assignments guarded by a rollback that restores every original core if
anything interrupts — and only after every swap succeeds are the replaced
cores released, exactly once. No failure at any stage leaves the model
partially updated, closes an input tensor, or invalidates existing
snapshots. Under `strict=False`, matching keys load with the same full
validation and atomicity, missing parameters keep their values, and
unexpected keys are ignored.

The value-replacement primitive is a narrowly scoped **internal**
`NativeParameter._adopt_value_core(new_core)` — it swaps the owned core
(defensively re-validating metadata) and returns the old core for the
caller to release exactly once or restore on rollback, changing nothing
else. It exists for controlled state-dict loading and is **not yet the
optimizer update API**. This is the framework's first in-place value
mutation, so the graph policy is explicit: a graph built *before* loading
stays **memory-safe** — a later backward through it reads the parameter's
current (newly loaded) value through the normal open-core path; new
forward graphs simply use the loaded values.

Everything above is locked by `tests/test_native_state_dict.py`
(selector: `-k "native_state_dict or load_state_dict"` — 54 new tests),
and the full suite passes at **1075 tests**. Still `float64`/`cpu` only,
still explicit and experimental, still no implicit dispatch and no NumPy
compute in any native numerical, gradient, or state-copy path. The next
milestone is **Advanced C++ v3.4 — NativeLinear**, narrowly scoped to:
a `NativeLinear` built on `NativeModule` with a `NativeParameter` weight
and optional `NativeParameter` bias, deterministic initialization, input
validation, strictly 2-D forward semantics initially (native `matmul`
plus broadcast `add`), parameter registration through assignment,
forward/backward and finite-difference tests, and state_dict
compatibility — **no optimizer or training loop yet** (`NativeLinear` is
not combined with `NativeSequential`, activations, losses, optimizers, or
training).

### Native training stack — NativeModule core and recursive registration (v3.2)

v3.2 adds the module hierarchy: `NativeModule`, the base every future
native layer, state_dict, optimizer, and training loop will build on. It
is a **Python-side organizational abstraction** — it performs no numerical
computation, owns no native storage, and never closes, copies, or mutates
what it registers. It deliberately ships **no** `NativeLinear`,
`NativeSequential`, activations, losses, optimizers, state_dict,
serialization, buffers, hooks, or training loop. No C++ changed,
`NativeTensor` is untouched, and `tensorforge.nn.Module`/`Parameter` are
untouched; the one v3.1 adjustment is a minimal read-only
`NativeParameterRegistry` extension (`get`/`__contains__` plus a shared
name-validation helper) with all v3.1 behavior preserved.

```python
from tensorforge.experimental import NativeModule, NativeParameter

class Block(NativeModule):
    def __init__(self):
        super().__init__()          # required before assigning parameters
        self.weight = NativeParameter([[1.0, 2.0], [3.0, 4.0]])

root = NativeModule()
root.block = Block()                # child registration by assignment
root.bias = NativeParameter([0.0, 0.0])

list(root.named_parameters())       # [("bias", ...), ("block.weight", ...)]
root.parameters()                   # unique parameters, identity-deduplicated
list(root.named_modules())          # [("", root), ("block", ...)]
root.zero_grad()                    # each unique parameter's grad -> None
root.eval().training                # False, propagated recursively
```

**Registration is assignment** (`__setattr__`), with registered objects
living only in the module's registries (`__getattr__` resolves them — one
source of truth): a `NativeParameter` value registers a parameter, a
`NativeModule` value registers a child, and **everything else is an
ordinary attribute** — a plain `NativeTensor`, a
`tensorforge.Tensor`/`Parameter`/`nn.Module`, a string — which never
enters native traversal (nothing is wrapped implicitly; stable-framework
objects stay harmless ordinary attributes). **One category per name; the
latest assignment wins**: registering validates first (a failure mutates
nothing), then evicts the name from the other categories. Replacement
within a registry preserves the slot position; moving a name between
registries appends to the target; `module.name = None` (and `del
module.name`) unregisters, leaving the attribute readable as `None`, and
re-registering a removed name appends — the v3.1 ordering rules
throughout. Evicted or replaced objects are dropped, never closed or
mutated, and no gradient state transfers. `register_parameter(name, p)` /
`add_module(name, m)` are the explicit forms with identical semantics
(their one deliberate strictness: a non-parameter/non-module value raises
`TypeError`, and `None` raises `KeyError` when nothing is registered under
the name). Names follow the v3.1 rule — non-empty dot-free strings (dots
reserved for hierarchical state_dict keys) — and `"_parameters"` /
`"_modules"` / `"training"` are reserved implementation slots that can
never be parameter or child names; `__init__` creates the registries via
`object.__setattr__` so initialization never routes through registration,
and registering before `NativeModule.__init__()` has run raises a clear
`RuntimeError`.

**Traversal is deterministic pre-order depth-first, deduplicated by
object identity** (`id`-keyed — never value equality), and **first
discovery wins**. `named_modules()` yields `("", self)` first, then each
child under its registered name and its descendants under dot-joined
paths, in insertion order; shared modules appear once under their first
discovered path, which also makes **direct and indirect module cycles
terminate safely** (cycles are allowed as references; traversal never
revisits a module). `named_parameters(prefix="", recurse=True)` walks
that order and yields each unique parameter once under its
first-discovered dotted name — a module's direct parameters before its
descendants', direct aliases and shared parameters (direct or through
children) deduplicated by identity, frozen parameters included;
`recurse=False` restricts to direct parameters. `parameters()` /
`modules()` return the matching identity-deduplicated lists — exactly
what a future optimizer iterates, and the first-discovered dotted names
are exactly the canonical keys v3.3's state_dict will use (loading will
copy values into existing parameters, preserving identity).

**`zero_grad()`** visits each unique parameter once (shared parameters
included) and calls its existing `zero_grad()` — grad → `None`, data /
`requires_grad` / graphs untouched, nothing closed — and returns `None`.
**`train(mode=True)`** validates `mode` as a real bool *before* touching
any state (non-bool raises `TypeError` with no partial mutation), then
sets `training` on every unique module — shared and cyclic hierarchies
visited once — and returns `self`; `eval()` is `train(False)`; every
module starts with `training = True` (a later mode-dependent layer will
read the flag; none exists yet). **`forward()` raises
`NotImplementedError` and calling the module delegates to `forward`** —
the whole call protocol, with no hooks or tracing machinery.

Lifetime: the registries store Python references only. Removing,
replacing, or deleting a registration — or dropping the module itself —
never invalidates an object another reference still holds; native storage
is released only by the owner's explicit `close()`, and there is no
`NativeModule.close()`.

Everything above is locked by `tests/test_native_module.py` (selector:
`-k "native_module or recursive_registration"` — 49 tests), and the full
suite passes at **1021 tests**. Still `float64`/`cpu` only, still explicit
and experimental, still no implicit dispatch and no NumPy in any native
numerical or gradient path. The next milestone is **Advanced C++ v3.3 —
Native State Dictionary Contract**, narrowly scoped to: `state_dict()`,
`load_state_dict()`, deterministic hierarchical names, strict
missing/unexpected-key checks, shape/dtype/device validation, value
copying that never replaces `NativeParameter` identity, and
shared-parameter canonical naming — **no file serialization and no
optimizer state yet** (state_dict, `NativeLinear`, serialization, and
checkpointing are not combined into one milestone).

### Native training stack — NativeParameter and registration contract (v3.1)

v3.1 opens **Phase C — the native training stack** — with its foundation:
`NativeParameter`, the trainable-leaf abstraction, and
`NativeParameterRegistry`, the minimal parameter-registration contract the
future `NativeModule` (v3.2) will embed. It deliberately ships **no**
module hierarchy, layer, loss, optimizer, training loop, state_dict, or
serialization — it settles the identity, ownership, leaf, and registration
rules everything later depends on. No C++ changed, no kernel or symbol was
added, `NativeTensor` itself was not modified, and `tensorforge.Tensor` /
`tensorforge.nn.Parameter` are untouched.

```python
from tensorforge.experimental import (
    NativeParameter, NativeParameterRegistry, NativeTensor,
)

w = NativeParameter([[1.0, 2.0], [3.0, 4.0]])   # leaf, requires_grad=True
b = NativeParameter([0.0, 0.0], requires_grad=False)  # frozen, still a parameter

x = NativeTensor.from_array([[1.0, 0.0], [0.0, 1.0]])
loss = x.matmul(w).sum()     # ordinary NativeTensor results — never parameters
loss.backward()              # w.grad is a NativeTensor matching w
w.zero_grad()                # grad back to None; data untouched

registry = NativeParameterRegistry()
registry.register("weight", w)
registry.register("bias", b)
registry.named_parameters()  # insertion-ordered (name, parameter) pairs
registry.parameters()        # unique parameters, deduplicated by identity
```

**`NativeParameter` is a `NativeTensor` subclass** (`__slots__ = ()`), not a
wrapper — the native ops require `NativeTensor` operands and the graph
engine reads leaf metadata directly, so composition would need a parallel
delegation surface for no gain. The one subclassing hazard — every op
builds results through the `_from_core`/`_from_op` classmethods, where
`cls` would be `NativeParameter` — is closed by overriding both to delegate
explicitly to `NativeTensor`: **parameter-ness never propagates**. Math,
views (`reshape`/`transpose`/`T`/`narrow`), `contiguous_copy`, `sum`/`mean`,
and `detach()` all return plain `NativeTensor` (detach additionally
graph-free with `requires_grad=False`); the only way to create a parameter
is calling `NativeParameter(...)` itself (the inherited
`from_array`/`zeros`/`full` classmethods therefore also return plain
tensors — they are not parameter constructors). Every parameter is a
**graph-free owning leaf for its whole life**: no `_parents`, no
`_backward`, never marked graph-freed, and participating in an operation
makes the *result* a non-leaf, never the parameter.

**Construction always takes an independent owning contiguous copy.**
From array-like data the values are copied into fresh float64/cpu native
storage; from an existing `NativeTensor` — leaf or non-leaf, contiguous or
strided/offset/borrowing view — the parameter copies the source's *current
value* (`contiguous_copy`) and inherits none of its graph history. Closing
the source never invalidates the parameter, closing the parameter never
invalidates the source, a one-shot backward through the source's graph
neither reaches nor frees the parameter, and a closed source is rejected
with the usual `RuntimeError`. No storage is ever shared, so a future
optimizer update can never mutate an unrelated tensor through a hidden
alias. `requires_grad` is a validated **real bool** (default `True`;
`requires_grad=False` builds a frozen parameter that stays registerable
and discoverable but accumulates no gradient — there is no broad
freeze/unfreeze API). Gradients follow the Phase B rules unchanged:
`.grad` starts `None`, accumulates `NativeTensor`-backed native gradients
matching the parameter's shape/dtype/device across fresh training-style
graphs, and `zero_grad()` clears to `None` without touching data.
`close()` follows the `NativeTensor` lifetime rules (idempotent; a closed
parameter rejects data and gradient operations).

**Identity, not value.** No `__eq__`/`__hash__` is defined anywhere in the
hierarchy: two equal-valued parameters are distinct parameters, comparison
never runs tensor math, and future optimizers can key state by object
identity (`id`-keyed, the same convention `backward()`'s traversal already
uses). Future state_dict loading is expected to copy values *into*
existing parameters, preserving identity — never to replace registered
objects.

**`NativeParameterRegistry`** is an insertion-ordered name → parameter
registry holding plain Python references — it owns no storage and never
closes, copies, or mutates a parameter, never touches `requires_grad` or
gradients, and dropping the registry leaves every parameter open. Its
contract:

- **Names** are non-empty `str` without `"."` (dots are reserved for the
  future hierarchical state_dict keys, matching the Python framework's
  dotted paths); invalid names raise `TypeError`/`ValueError` — nothing is
  silently stringified.
- **Values** are `NativeParameter` only, or `None` to **unregister**
  (`KeyError` if the name is not registered). An ordinary `NativeTensor`
  and the Python framework's `Tensor`/`Parameter` are rejected with a
  clear `TypeError` — never wrapped implicitly (the same rejection the
  `NativeParameter` constructor applies to framework objects, checked
  lazily so the native backend still never imports the frontend).
- **Ordering** is insertion order. **Replacing** a registered name keeps
  its position and simply drops the reference to the previous parameter —
  the old object is not closed or mutated and no gradient state transfers.
  **Unregistering** deletes the slot, so registering that name again
  appends at the end (the documented rule).
- **Aliases**: the same parameter may be registered under several names
  (the future shared-weight case). `named_parameters()` shows every alias;
  `parameters()` deduplicates by object identity in first-registration
  order (each unique parameter exactly once — the traversal an optimizer
  iterates); `unique_named_parameters()` is the deduplicated named view
  where the first-registered name wins. Equal-valued distinct parameters
  are never deduplicated.

Everything above is locked by `tests/test_native_parameter.py` (selector:
`-k "native_parameter or parameter_registration"` — 49 tests), and the
full suite passes at **972 tests**. Still `float64`/`cpu` only, still
explicit and experimental, still no implicit dispatch and no NumPy in any
native numerical or gradient path. The next milestone is **Advanced C++
v3.2 — NativeModule Core and Recursive Registration**: child-module
registration, automatic parameter/module assignment registration,
recursive `parameters()` / `named_parameters()` / `modules()` /
`named_modules()`, `zero_grad()`, deterministic traversal,
shared-parameter and shared-module handling, and the train/eval state
foundation — no layers or optimizers yet.

### Native autograd — Phase B guardrails and completion (v2.6)

v2.6 is the **completion** milestone for Phase B — native autograd. It adds
**no** operation, kernel, optimizer, training abstraction, or performance
optimization, and changes **no** autograd behavior; it audits, locks down,
documents, and formally completes the engine. It is almost entirely tests
and documentation (the one source touch is a stale docstring correction —
`tensorforge.experimental`'s package docstring no longer claims "no
autograd", which has been false since v2.1). No C++ changed, no kernel or
symbol was added, and no NumPy entered the gradient path.

The new cross-cutting guardrails live in
`tests/test_native_autograd_guardrails.py` (selector:
`-k "phase_b_guardrail or native_autograd_guardrail or native_backend_isolation"`)
and lock several completed invariants together rather than duplicating the
per-operation tests:

- **No NumPy in the gradient path.** A runtime guard (a context manager)
  replaces NumPy's *numerical* functions (`add`/`multiply`/`matmul`/`sum`/
  `mean`/`maximum`/`where`/`broadcast_to`/… — the ones a NumPy-backed
  backward would call) with tripwires that raise, while leaving the
  marshalling helpers (`asarray`/`empty`) intact. The graph is built
  *before* the guard and gradients are read *after* it, so a correct native
  pass completes cleanly and only a smuggled-in NumPy computation would
  trip a wire. It covers same-shape elementwise, genuine broadcasting,
  reduction, matmul, and a transpose→narrow→contiguous_copy→reshape view
  chain, and a self-check proves the guard actually bites.
- **`NativeTensor` ↔ `tensorforge.Tensor` isolation.** Native ops return
  `NativeTensor`; native grads are `NativeTensor`-backed; `Tensor` stays
  NumPy-backed; neither engine's backward touches the other; mixed operands
  raise clearly (both directions) instead of dispatching implicitly.
- **Explicit backend / no implicit dispatch.** The wrapper is reached only
  through `tensorforge.experimental`; a static check confirms `import
  tensorforge` imports neither `experimental` nor `backends`; native
  unavailability raises the build-instructions `ImportError` (no silent
  NumPy fallback); there is no automatic backend selection.
- **Gradient ownership, graph lifetime, detach, view+offset, and
  closed-operand safety** are each locked over realistic mixed graphs
  (shared intermediate + broadcast + view), including the deterministic
  freed-graph error and snapshot-based failure rollback.
- **Kernel-registry boundary.** `list_kernels()` and
  `tensor_core_kernels` must not leak the internal fused backward kernels
  (`tf_core_relu_backward`, `tf_core_narrow_backward`, `tf_core_sum`,
  `tf_storage_scale`) — they stay forward-shaped numerical methods only.
- **Benchmark mode contract.** The v2.5 modes keep their documented
  meanings: `forward_native` builds no graph, the grad modes build one,
  `forward_backward_fresh` builds and frees a fresh graph, and
  `backward_retained` reuses one fixed graph.

The **final Phase B support matrix**, the **explicit divide-backward
decision** (deferred beyond Phase B — Phase B is complete without it,
because the completed op set already spans a first native training stack;
see §18 of the design), and the **Phase B-complete / Phase C-next** status
are recorded in
[native_autograd_design.md](native_autograd_design.md) (§17–§19). Phase B is
**complete** at **923 tests** (893 baseline + 30 guardrails); the next
milestone is **Advanced C++ v3.1 — NativeParameter and Parameter
Registration Contract** (Phase C — the native training stack). No divide
kernel/API, no operator overloads, no CUDA, no dtype promotion, and no
implicit dispatch came with this milestone.

### Native autograd — benchmark characterization (v2.5)

v2.5 is a **measurement and documentation** milestone: it changes no
autograd behavior, adds no kernel, and makes no cross-framework claim. It
adds a reproducible harness,
[`benchmarks/benchmark_native_autograd.py`](../benchmarks/benchmark_native_autograd.py),
that characterizes where time goes in the native autograd stack, and one
honest hardware-specific snapshot. Full write-up (cases, modes, methodology,
results, and interpretation limits) is in
[native_autograd_benchmarks.md](native_autograd_benchmarks.md).

Five workloads — a same-shape elementwise chain, a genuine-broadcast chain,
a 3-D reduction chain, a 2-D matmul chain, and a transpose→narrow→
contiguous_copy→reshape view chain — are each run in four modes that
separate the layers as far as the architecture permits:
`forward_native` (grad off, no graph), `forward_graph` (graph built, no
backward), `forward_backward_fresh` (fresh graph + one-shot `backward()`,
including cleanup), and `backward_retained` (one graph built outside the
loop, `backward(retain_graph=True)` repeatedly — isolating repeated
backward, explicitly *not* a training-step estimate). Timing uses
`time.perf_counter_ns()`, configurable warmup/iterations/repeats, median
as the primary statistic with min/max spread, and a correctness gate
(output shape, finite output, and — for backward modes — that each leaf
gradient exists, has the right shape, and is finite) before any timing. A
CLI (`--case --mode --warmup --iterations --repeats --json --smoke`) runs
all cases/modes by default, rejects unknown cases/modes and non-positive
counts, and emits pure JSON (raw samples included) under `--json`.

On the recorded snapshot (one Windows/AMD64 machine, float64/cpu), adding a
backward pass is the dominant cost (fresh forward+backward ≈ 2.5×–5× the
forward), retained backward sits below fresh backward everywhere (it drops
the forward rebuild from the loop), graph-construction overhead is small
relative to compute at these sizes (clearest on matmul), and the smoke
shapes are dominated by wrapper/ctypes cost — all machine-specific
observations, with no speed assertions anywhere. The benchmark tests
(`tests/test_native_autograd_benchmark.py`) validate schema and behavior
only, never speed.

### Native autograd — graph lifetime policy (v2.4)

v2.4 gives the native autograd graph an explicit, PyTorch-like
**lifetime**. `backward` now takes a flag — `backward(gradient=None,
retain_graph=False)` — and the default is **one-shot**: after a
successful pass, the operation graph of every traversed non-leaf node is
released.

```python
from tensorforge.experimental import NativeTensor

x = NativeTensor.from_array([1.0, 2.0, 3.0], requires_grad=True)
out = x.multiply(x).sum()
out.backward()          # computes dx = 2x, then frees the graph
out.backward()          # RuntimeError: graph already freed — use retain_graph=True

y = NativeTensor.from_array([1.0, 2.0, 3.0], requires_grad=True)
loss = y.multiply(y).sum()
loss.backward(retain_graph=True)   # keeps the graph
loss.backward(retain_graph=True)   # runs again; y.grad accumulates to 4y
```

Releasing a graph clears each traversed non-leaf node's `_parents` and
`_backward` closure (so nothing keeps the parents alive) and marks the
node **freed**. A later `backward()` that reaches a freed node raises a
clear `RuntimeError` naming `retain_graph=True` as the remedy — and
crucially it does *not* silently treat the freed node as a leaf, which
would truncate history. That covers three cases with one flag: a repeated
backward on the same output, a **second output over a shared
intermediate** (`a = shared.sum(); b = shared.mean(); a.backward()` frees
`shared`, so `b.backward()` raises), and a **new op built from a freed
value** (`shared.add(y)` still computes forward — the stored value is
intact — but its backward refuses to cross the freed history). A genuine
leaf has no graph to free and is never marked freed, so repeated
`backward()` on a scalar leaf keeps accumulating.

`retain_graph` is validated as a real `bool` **first** — before traversal,
callbacks, cleanup, or any gradient mutation, and never coerced with
`bool(...)` — so a bad value (`"true"`, `1`, `None`, …) raises `TypeError`
and changes nothing. Whether or not the graph is freed, only leaves retain
grad; non-leaf grads stay transient and are dropped after each pass, so a
retained second pass recomputes them cleanly and leaf gradients accumulate
across passes until `zero_grad()` (which clears a leaf grad without
resurrecting a freed graph or damaging a retained one). The pass is
**failure-safe**: it is staged against a snapshot of every node's gradient
(gradients are immutable — accumulation replaces the reference with a
fresh native `add`), so if a callback raises mid-traversal — for instance
one branch commits a leaf and another hits a closed operand — the
references are restored, leaving no partial commit and no partial free, and
cleanup runs only after the pass fully succeeds. This is **not** full
PyTorch parity: there is no per-node `retain_grad` and no double-backward.
No C++ changed, no kernel was added, and no NumPy entered the gradient
path; `NativeTensorCore` still owns no graph state. Full status in
[native_autograd_design.md](native_autograd_design.md).

### Native autograd — narrow backward (v2.3)

v2.3 makes the last view op differentiable: `narrow(dim, start, length)`
now records a graph node when its parent requires grad. Its backward is a
**scatter** — the adjoint of a slice — placing the upstream gradient into
a fresh zeros tensor of the parent's shape at the narrowed region, so
every un-narrowed position gets zero gradient and the narrowed region
gets exactly the upstream.

```python
from tensorforge.experimental import NativeTensor

x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            requires_grad=True)
x.narrow(1, 0, 2).sum().backward()   # keep columns 0..1, sum, differentiate
x.grad.to_numpy()                    # [[1, 1, 0], [1, 1, 0]]
```

The scatter runs through **one new C++ kernel**, `tf_core_narrow_backward`
— the odometer dual of `tf_core_sum`. Where a sum walks the input and
folds many elements into one output cell through zero *write* strides, a
narrow-backward walks the (smaller) narrowed shape and writes each
upstream element into its own output cell: the write position advances by
the parent's full row-major strides from a base offset that skips the
leading `start` slabs along the narrowed dimension
(`start * row_major_stride[dim]`). The upstream is read through its own
shape/strides/offset, so a strided gradient works without materializing,
and the output is fresh **owning** row-major contiguous storage — zero
everywhere outside the window. It is surfaced as
`NativeTensorCore.narrow_backward(dim, start, original_shape)`, a
forward-shaped numerical method (the core and the kernels still own no
graph state); it is not added to `list_kernels()`/`TENSOR_CORE_KERNELS`,
matching the `tf_core_relu_backward`/`tf_core_sum` convention.

Because the gradient lives at the *logical* shape, the scatter always
allocates a fresh contiguous buffer of the parent's shape (offset 0)
regardless of the parent's own layout — so **transposed, narrowed, and
nonzero-offset parents** all work: each is simply a preceding graph node
whose own backward (transpose-inverse, another narrow-scatter, …) handles
its layout, and narrow backward only needs the immediate parent's shape.
Nested narrows, `narrow` under `sum`/`mean`/`multiply`, and `narrow`
feeding `transpose`/`reshape` (via `contiguous_copy`) all compose. The
lifetime rules are unchanged: the retained contribution owns its storage,
closing the parent or output before `backward()` raises a clear
`RuntimeError`, repeated backward accumulates until `zero_grad()`. Rules
are verified against an independent NumPy zero-padding reference and a
finite-difference check (NumPy test-side only), and the CI smoke script
hard-checks one narrow-backward pattern. This closes the view-backward
set; `retain_graph` is still not offered (v2.4), and there is still no
`divide` backward, no `tensorforge.Tensor` integration, no implicit
dispatch, no optimizer/training stack, no CUDA, and no performance
claims. Full status in
[native_autograd_design.md](native_autograd_design.md).

### Native autograd — core backward operations (v2.2)

v2.2 turns the v2.1 skeleton into a working reverse-mode engine: the
core `NativeTensor` operations are now **differentiable**. When an
operand requires grad, `add`, `subtract`, `multiply`, `relu`, `sum`,
`mean`, `matmul` (2-D), `reshape`, `transpose`/`T`, and
`contiguous_copy` record a graph node (parents + backward closure + op
name via the internal `_from_op`); when nothing requires grad, every op
returns a plain forward tensor exactly as before — no graph metadata is
created and forward-only use stays as cheap as it was.

```python
from tensorforge.experimental import NativeTensor

x = NativeTensor.from_array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]])
w = NativeTensor.from_array([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0]],
                            requires_grad=True)
b = NativeTensor.from_array([0.5, -0.5], requires_grad=True)

loss = x.matmul(w).add(b).relu().mean()   # (2,) bias broadcast over (2, 2)
loss.backward()                           # scalar seed defaults to 1.0
w.grad, b.grad                            # native float64/cpu NativeTensors
```

The backward rules are the design's (§7.4/§8/§9), computed **entirely by
native forward kernels at the `NativeTensorCore` level** — backward math
builds no graph nodes of its own and no NumPy ever touches a gradient:
`add`/`subtract` pass the upstream through (the right operand negated by
a broadcast-scalar multiply against a native `-1.0` — the §7.5
composition, no negate kernel); `multiply` computes `u·b` / `u·a`;
`matmul` computes `u @ b.T` / `a.T @ u` over strided transpose views;
`sum`/`mean` broadcast the upstream back to the input shape (reduced
axes reinserted as size 1 by a native reshape, expanded by the existing
zero-stride broadcasting via `zeros + u`, `mean` scaled by a native
`1/count` scalar multiply); `reshape`/`transpose` apply the inverse
reshape/permutation. Broadcasting backward is the new private
`_unbroadcast(grad, target_shape)` helper — the adjoint of broadcasting,
built from single-axis native reductions applied in a stable order
(leading padded axes summed away axis-0-first, stretched axes summed
with `keepdims=True`, a final native reshape for the `()`-versus-`(1,)`
rank family, which the helper distinguishes exactly). `contiguous_copy`
backward is the identity: the forward is an elementwise logical copy and
gradients live at the logical shape, so the parent's storage layout is
irrelevant. **One new C++ kernel** was added — `tf_core_relu_backward`
(`upstream` where `x > 0`, else `0`; `x == 0` blocks, the Python Tensor
convention) — exactly the fused kernel the design flagged, implemented
as one more op through the existing generic binary odometer walker, so
contiguous and strided inputs both work with no new traversal code. It
is surfaced as `NativeTensorCore.relu_backward`, a forward-shaped
numerical method: the core and the kernels still own **no graph state**.

Lifetime stays explicit and safe: every retained gradient contribution
is either the upstream tensor itself or fresh owning contiguous storage
(never a borrowing view over a transient), closing an operand or an
intermediate before `backward()` raises a clear `RuntimeError` (the
graph never reads freed storage), and with no in-place arithmetic the
forward values captured for backward stay valid for the life of the
graph. Only leaves retain grad; repeated `backward()` accumulates until
`zero_grad()`; `detach()` still cuts the graph. **`narrow` remained
outside autograd in v2.2** — its backward needs a native scatter
primitive (upstream scattered into zeros of the original shape), which
landed in v2.3 (above) rather than being faked through NumPy — and
`retain_graph` is still not offered. The rules are verified against exact
analytical values and
central finite differences (NumPy only as the test-side reference), the
deterministic `examples/native_autograd_demo.py` shows one full native
forward + backward (`demo()` returns NumPy copies for its tests), and
the CI smoke script hard-checks one scalar-loss backward. Still no
`tensorforge.Tensor` integration, no implicit dispatch, no optimizer or
training stack, no CUDA, and no performance claims. Full status in
[native_autograd_design.md](native_autograd_design.md).

### Native autograd — metadata skeleton (v2.1)

v2.1 implements the first piece of the v2.0 design: the **native autograd
metadata skeleton and the reverse-topological backward driver** on
`NativeTensor`. It is **Phase B's first code**, and it is deliberately a
skeleton — the forward compute ops are **not** wired into autograd yet
(that is v2.2). No kernel changed, `NativeTensorCore` and the C++ kernels
stay autograd-unaware, and `tensorforge.Tensor` is untouched.

`NativeTensor` now carries an opt-in, Python-managed autograd graph:

```python
from tensorforge.experimental import NativeTensor

x = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
x.requires_grad        # True
x.grad                 # None (lazily filled by backward)
x.is_leaf              # True (user-created)
d = x.detach()         # forward-only owning copy: requires_grad False, no graph
x.zero_grad()          # clears grad to None (idempotent)
```

The design's graph model landed as specified. New `NativeTensor` state
(`_requires_grad`, `_grad`, `_parents`, `_backward`, `_op`, `_is_leaf`)
lives **at the wrapper layer**; the raw runtime is unchanged. A single
private constructor, `NativeTensor._from_op(core, parents, backward, op)`,
builds a non-leaf node — `requires_grad` is the OR of the parents', and
the graph is recorded only when some parent needs grad (a plain forward
leaf otherwise, matching the Python `Tensor`). Constructors gained a
default-preserving `requires_grad=False` argument (rejecting non-`bool`),
so every existing call is unchanged. `backward(gradient=None)` mirrors the
Python engine natively: a post-order DFS over `_parents` keyed by **object
identity** (`id()`, never hashing, so duplicate/shared parents are visited
once), a seed (scalar output — `numel == 1` — defaults to a native `1.0`;
a non-scalar requires an explicit `NativeTensor` gradient matching the
output's shape/dtype/device, errors naming both), then a reverse walk
calling each non-leaf's backward closure. Gradients are **native**
(`NativeTensor`-backed), accumulated by the native `add` kernel
(`_accumulate_grad`) — **no NumPy touches the gradient path** — lazily
initialized, and obey the v1.21 contract (`grad.dtype`/`.device`/`.shape`
match the tensor). Only leaves retain grad; non-leaf grads are transient
and dropped after each pass. `zero_grad()` clears to `None` (the design's
choice), and grads are never eagerly closed, so a live `t.grad` is never
invalidated. `detach()` returns an **owning contiguous copy** (independent
lifetime, no NumPy round trip). `backward()` on a non-requiring or closed
tensor raises; `retain_graph` is intentionally not offered (the graph is
rebuilt each call, so repeated `backward()` accumulates until
`zero_grad()`). Full design and status in
[native_autograd_design.md](native_autograd_design.md); rigorous tests in
`tests/test_native_autograd.py`.

### Native autograd — design (v2.0)

v2.0 is **design-only**: no autograd is implemented, no kernels change,
and no runtime behavior changes. With **Phase A — the native CPU runtime —
complete in code** (the contiguous fast path v1.14, broadcasting v1.17,
reductions v1.19, and dtype/device metadata v1.21), this milestone opens
**Phase B** by writing the design for native reverse-mode autograd over
`NativeTensor` / `NativeTensorCore`.

Today the native runtime is **forward-only with no operation graph**:
`NativeTensorCore` ops return a fresh result that records no parents, no
operation, and no backward rule; `NativeTensor` has no `requires_grad`,
`grad`, or `backward`. The design specifies a **Python-managed graph at
the `NativeTensor` layer** — `NativeTensorCore` stays the raw forward
runtime and the C++ kernels never own graph state — where each
differentiable op records `core` + `requires_grad` + parents + a backward
closure + an op name, leaf tensors (user-created, `requires_grad=True`)
accumulate gradients, and `backward()` walks the graph in reverse
topological order (scalar outputs default their seed to `1`, non-scalar
outputs require an explicit gradient). Gradients are themselves
**native** (`NativeTensor`-backed, lazily initialized, accumulated by
native `add`) and honor the v1.21 metadata contract exactly
(`grad.dtype == tensor.dtype`, `grad.device == tensor.device`) — the
concrete reason A4 preceded autograd.

The design is honest about missing kernels: first-scope backward needs a
small fused `relu_backward` (no native compare/`where` exists), and it
notes negation/scalar-multiply, a core-level `divide`, and a
scatter/copy-into-view (for `narrow`/`contiguous_copy` backward) as
deferred additions rather than silently assuming them. Broadcasting
backward is an `unbroadcast(grad, original_shape)` helper built on native
**reductions (A3)** — the recorded dual: a broadcast forward read is a
sum-reduction on the backward pass. It stays **separate from
`tensorforge.Tensor`** (no conversion, no implicit dispatch, no silent
NumPy fallback, `Tensor` behavior unchanged) and **CPU float64 only** (no
CUDA autograd). The staged plan is **v2.1 metadata skeleton →
v2.2 basic backward (add/multiply/relu/sum) → v2.3 broadcasting + mean
backward → v2.4 matmul backward → v2.5 native autograd demo**, then Phase
C (native training stack). No code ships; the next milestone is **v2.1,
the native autograd metadata skeleton**. Full design in
[native_autograd_design.md](native_autograd_design.md).

### Native dtype/device metadata — implementation (v1.21)

v1.21 implements the metadata contract the v1.20 design specified —
**metadata only, float64/cpu only**, no kernel and no compute behavior
changed. Native tensors now carry explicit, inspectable `dtype` and
`device` you can read:

```python
t = NativeTensorCore.from_array([[1.0, 2.0], [3.0, 4.0]])
t.dtype, t.device            # ("float64", "cpu")
t.T.dtype                    # "float64" — a view shares its storage's tags
t.add(t).device             # "cpu"    — ops preserve dtype/device

NativeTensor.zeros((2, 3)).dtype          # "float64"
NativeTensorCore.zeros((2,), dtype="float32")  # ValueError — rejected
```

The tags are **owned by `NativeStorage`** (the allocation owner, so
every view over one buffer reports the same dtype/device — one source of
truth) and surfaced read-only through `NativeTensorCore.dtype`/`.device`
and `NativeTensor.dtype`/`.device`. Two pure helpers, `normalize_dtype`
and `normalize_device` (beside `broadcast_shapes`/`reduce_shape`, so they
are testable without the built backend), validate and canonicalize the
tags against `SUPPORTED_DTYPES == ("float64",)` and
`SUPPORTED_DEVICES == ("cpu",)`. Constructors (`from_array`/`zeros`/
`full` on both the core and the wrapper, and `tensor_from_array`/`zeros`/
`full` on the native backend) gained **default-preserving** `dtype`/
`device` arguments — `dtype=None`→`"float64"` on `from_array`,
`dtype="float64"` on `zeros`/`full`, `device="cpu"` throughout — so every
existing call is byte-for-byte unchanged and still produces a float64/cpu
tensor. `from_array` still coerces its data to float64 exactly as before;
the tag records intent, it does not change the bytes.

Following the design's **reject-over-inert** recommendation, any
unsupported dtype/device is rejected at construction (before native
memory is allocated), naming the offending value and the supported set —
so no tensor ever advertises a dtype/device the float64/CPU kernels
cannot actually compute. The binary ops and `matmul` validate that both
operands share dtype and device (naming both pairs on a mismatch) — a
guard that cannot fire yet with one legal value each, but is the enforced
contract native autograd (Phase B) will read `grad.dtype == param.dtype`
against. Every op and view carries the metadata through: `relu`, `add`/
`subtract`/`multiply` (broadcasting included), `matmul`, `sum`/`mean`,
and the metadata-only views all produce float64/cpu results; `to_numpy`
still returns float64 arrays. `backend_info()` now advertises the
supported sets (`supported_dtypes`, `supported_devices`, and a `device`
field) for both the native and NumPy backends.

The `dtype`/`device` fields follow each object's existing metadata
closed-behavior: readable after `close()` on `NativeStorage`/
`NativeTensorCore` (like `size`/`shape`), rejected after `close()` on the
`NativeTensor` wrapper (like its `shape`/`strides`). No dtype promotion,
casting (`astype`/`to`/`cpu`/`cuda`), non-float64 kernels, CUDA,
autograd, `Tensor` integration, or new numeric kernels came with it —
those remain the future phases the contract enables. This **closes Phase
A — the native CPU runtime — in code**; the next milestone is **v2.0, the
Phase B native autograd design**. Full design in
[native_dtype_device_metadata_design.md](native_dtype_device_metadata_design.md).

### Native dtype/device metadata — design (v1.20)

v1.20 is **design-only**: no kernels change, no runtime behavior changes,
and dtype/device metadata is **not implemented**. It follows the
reductions milestone and **closes the Phase A design surface** — the last
native-CPU-runtime piece before Phase B (native autograd).

Today the native runtime is **float64-CPU-only, implicitly**:
`NativeStorage` allocates a `double[]` buffer and records nothing about
its element type or location, and `NativeTensorCore`/`NativeTensor`
expose `shape`/`strides`/`contiguous` but no `dtype` or `device`. The
design makes that assumption **explicit**: dtype and device become
inspectable metadata — owned by `NativeStorage` (the allocation owner, so
views share it and CUDA later has device-aware storage), surfaced
read-only through `NativeTensorCore.dtype`/`.device` and
`NativeTensor.dtype`/`.device`. They are validated canonical string tags
(`"float64"`, `"cpu"`), JSON-friendly for future checkpoints and never
silently inferred from NumPy; the NumPy correspondence stays explicit at
the conversion boundaries only. Constructors gain default-preserving
`dtype`/`device` arguments (`dtype=None`→`"float64"`, `device="cpu"`), so
every existing call is byte-for-byte unchanged.

The design specifies operation validation (binary ops and matmul require
matching dtype and device, naming both on a mismatch; `sum` preserves
dtype; `mean` stays float64, with future integer-mean deferred; `relu`
requires a numeric dtype; `to_numpy` will match the stored dtype once
non-float64 exists) and a hard **no-promotion / no-auto-copy /
no-silent-conversion / no-NumPy-fallback** rule. Casting and device moves
(`astype`/`to`/`cpu`/`cuda`) are reserved as future, explicit,
copy-producing operations — `cuda()` should not exist until a CUDA
backend does. Crucially, because the kernels are float64/CPU only, the
design **recommends rejecting** any non-`float64`/non-`cpu` construction
(the safer of reject-vs-inert), so no tensor ever advertises a dtype the
kernels cannot actually compute. The contiguous fast path, broadcasting,
reductions, and the thin wrapper are all unchanged. A metadata-only
implementation (float64/cpu only) is proposed as **v1.21**, closing Phase
A in code before the Phase B autograd design. Full design in
[native_dtype_device_metadata_design.md](native_dtype_device_metadata_design.md).

### Native reductions — implementation (v1.19)

v1.19 implements native `sum` and `mean` reductions for
`NativeTensorCore` / `NativeTensor`, following the v1.18 design. Supported
semantics are NumPy-style: `axis=None` reduces every element, a single
integer or **negative** `axis` reduces one dimension, and `keepdims`
controls whether the reduced axis stays as size 1.

```python
a = NativeTensorCore.from_array(np.arange(6.0).reshape(2, 3))
a.sum()                     # () scalar total
a.sum(axis=0)               # (3,)
a.sum(axis=1, keepdims=True)# (2, 1)
a.mean(axis=-1)             # (2,)  negative axis
```

A pure `reduce_shape(shape, axis, keepdims)` helper (beside
`broadcast_shapes`) infers the output shape and validates axis/keepdims.
The compute is one new C ABI kernel, **`tf_core_sum`**, that is the
**dual of broadcasting**: where a broadcast reads one element into many
output positions through zero *read*-strides, a reduction walks the input
odometer and writes many elements into one output cell through zero
*write*-strides (0 on reduced axes). It reads the input through its
existing shape/stride/offset metadata, so contiguous, transposed,
narrowed, and nonzero-offset inputs all reduce correctly **without being
materialized**, accumulating into freshly allocated zero-initialized
row-major contiguous output. `mean` reuses `sum` and scales the output in
place by `1/count` via a small `tf_storage_scale` primitive — no NumPy
touches the data. `NativeTensor.sum`/`mean` delegate to the core, so the
wrapper inherited reductions with no reduction-specific code, and the
explicit NumPy and native backends gained symmetric `sum`/`mean` methods.

Numerical behavior is honest: float sums are order-sensitive, so the
kernel uses a plain deterministic row-major accumulation (no
SIMD/FMA/Kahan/pairwise), and results are checked against NumPy with a
**tolerance** (`np.allclose`), not bit-for-bit. Reductions are
**forward-only** — no autograd was added; the broadcast/backward
relationship (a broadcast backward is a reduction over the broadcast
axes) stays reserved for the future native-autograd phase. No
`max`/`argmax`/`min`/`product`, tuple axes, `Tensor` integration, CUDA,
dtype promotion, operator overloads, or distributed reductions came with
it. Full design in
[native_reductions_design.md](native_reductions_design.md).

### Native reductions — design (v1.18)

v1.18 is **design-only**: no kernels change and reductions are **not
implemented**. With Phase A1 (contiguous fast path) and A2 (broadcasting)
complete, the next Phase A step is reductions — collapsing an axis (or all
of them) of a tensor — and this milestone writes their design.

Reductions follow broadcasting for a concrete reason: they are the
**prerequisite for native autograd**. Broadcasting's forward pass reads
one element into many positions (zero read-strides); its backward pass
must do the opposite — **sum the gradient over the broadcast axes** —
which is a reduction. Today `NativeTensorCore` / `NativeTensor` expose no
`sum`/`mean`, so no broadcasting op can have a native backward yet; A3
unblocks that.

The design specifies `sum` and `mean` first (`max`/`argmax`/`min`/
`product` deferred — `max`/`argmax` complicate the zero-initialized
accumulate and, for `argmax`, return indices rather than float64 values),
with NumPy-style semantics: `axis=None` reduces everything, a single
integer (or negative) `axis` reduces one dimension, and `keepdims`
controls whether the reduced axis stays as size 1
(`(2, 3).sum(axis=1) -> (2,)`, `... keepdims=True -> (2, 1)`,
`().sum() -> ()`). The traversal is the **dual of broadcasting**: where
broadcasting reads through zero strides, a reduction **writes** through
zero strides — many input elements scatter-accumulate into one output
cell — so the same odometer machinery drives it, reads any
contiguous/transposed/narrowed/nonzero-offset input directly without
materializing, and writes a freshly allocated row-major contiguous
output. Numerical honesty is explicit: float sums are order-sensitive, so
the design commits to a plain deterministic loop (no Kahan/pairwise/SIMD
in first scope) and to comparing against NumPy with **tolerances**, not
bit-for-bit. `NativeTensor` will inherit `sum`/`mean` by delegation with
no wrapper change; no autograd, `Tensor` integration, CUDA, dtype
promotion, operator overloads, tuple-axis, or distributed reductions come
with it. Implementation is v1.19. Full design in
[native_reductions_design.md](native_reductions_design.md).

### Native broadcasting — implementation (v1.17)

v1.17 implements native broadcasting for the elementwise binary ops
`add`/`subtract`/`multiply`, lifting the native runtime's exact-shape
restriction to NumPy-style broadcasting. `NativeTensorCore.add(other)` (and
`subtract`/`multiply`) now accept identical shapes **or** any
broadcast-compatible pair — scalar↔tensor, same-rank size-1 stretching,
and left-padding a lower-rank operand with leading 1s:

```python
a = NativeTensorCore.from_array(np.ones((2, 3)))
row = NativeTensorCore.from_array([10.0, 20.0, 30.0])   # (3,)
a.add(row)                    # (2, 3): (3,) left-pads to (1, 3), stretches
NativeTensorCore.from_array(np.ones((3, 1))).multiply(
    NativeTensorCore.from_array(np.ones((1, 4))))        # (3, 1) * (1, 4) -> (3, 4)
```

`_binary_core_op` now dispatches **three ways**:

- **same shape, both contiguous** → the v1.14 flat fast-path kernel
  (`<op>_contiguous`), unchanged;
- **same shape, either strided** → the generic odometer kernel (`tf_core_<op>`),
  unchanged;
- **differing but compatible shapes** → the **broadcast traversal**: a
  pure-Python `broadcast_shapes(a, b)` infers the output shape (raising a
  clear `ValueError` naming both shapes if incompatible), a small
  `_broadcast_strides` helper builds each operand's read-strides — the
  real stride on a genuine axis, **stride 0** on a stretched or
  left-padded size-1 axis — and the **same generic odometer kernel**
  consumes them. A zero stride means "re-read one element", so nothing is
  materialized; the expanded operand never exists in memory.

The key implementation note: **no new C++ kernel was added.** The existing
`tf_core_binary` odometer already walks the output shape advancing each
operand by its own strides, so it is already broadcast-capable once fed
zero-augmented strides — broadcasting is entirely a Python-side stride
computation (`broadcast_shapes` sits beside the other pure metadata
helpers, testable without the built backend). Output is always freshly
allocated row-major contiguous storage; operands are never mutated.
`NativeTensor` inherited broadcasting through `NativeTensorCore` with
**no wrapper edit** — `a.add(b)` on a `(2, 3)` and a `(3,)` now works
through the wrapper. Same-shape correctness (fast path and odometer) is
unchanged, verified by regression tests, and every broadcast result is
checked exactly (`np.array_equal`) against NumPy — including transposed,
narrowed, and nonzero-offset broadcast operands. Errors stay explicit
(incompatible shapes raise, naming both; no silent NumPy fallback). No
reductions, autograd, `Tensor` integration, CUDA, dtype promotion,
operator overloads, or matmul broadcasting were added. Full design in
[native_broadcasting_design.md](native_broadcasting_design.md).

### Native broadcasting — design (v1.16)

v1.16 is **design-only**: no kernels change and broadcasting is **not
implemented**. With Phase A1 (the contiguous fast path) complete —
designed v1.13, implemented v1.14, benchmark impact reported v1.15 — the
next Phase A step is broadcasting, and this milestone writes its design.

Today the native elementwise ops are **exact-shape only**:
`NativeTensorCore._binary_core_op` rejects any mismatched pair (a `(3, 4)`
and a `(3, 1)`, or a scalar and a matrix) with a clear `ValueError`. The
design specifies lifting that to NumPy-style broadcasting for
`add`/`subtract`/`multiply` — scalar↔tensor, same-rank size-1 stretching,
and left-padding a lower-rank operand with leading 1s
(`() + (3, 4) -> (3, 4)`, `(3, 1) + (1, 4) -> (3, 4)`,
`(1, 3, 1) + (2, 1, 5) -> (2, 3, 5)`). The mechanism is a **zero-stride
read model**: a stretched axis is read with stride 0 (re-reading one
element) rather than materializing an expanded operand, and the output
stays freshly allocated row-major contiguous native storage. The v1.14
fast path is preserved untouched for the same-shape contiguous case;
broadcasting only engages when the shapes actually differ, and takes the
generic broadcast odometer first (specialized broadcast fast paths are
deferred). It composes with transposed/narrowed/offset views because
everything is expressed in strides. Errors stay explicit — a mismatch
names both shapes, and there is **no silent NumPy fallback**. Autograd
implications (a broadcast forward read is a sum-reduction on the backward
pass) are noted for later, not built. The full design, scope
(implementation is v1.17), test/benchmark plans, and roadmap fit are in
[native_broadcasting_design.md](native_broadcasting_design.md).

### Contiguous fast-path — benchmark impact report (v1.15)

This closes the optimization loop the last four milestones opened:
**v1.12** measured where the elementwise cost lived (the generic
shape/stride odometer traversal in the native runtime, not the
`NativeTensor` wrapper), **v1.13** designed a contiguous fast path,
**v1.14** implemented it, and **v1.15** records the measured impact. It
is a *report*, not new behavior — no kernels or runtime changed.

The table below is a single local run of
`uv run python benchmarks/cpp_backend.py` (full sizes). Numbers are
`vs numpy` ratios (higher = slower than NumPy); the large `1000x1000`
elementwise rows are the honest ones to read, because they are
memory-bound and dominated by traversal cost rather than per-call ctypes
overhead. **These figures are hardware-dependent and from one machine —
they characterize behavior, they are not a benchmark score.**

| op (1000×1000) | cpp raw buffer | tensor core (contiguous) | tensor core (view) | native tensor (contiguous) | native tensor (view) |
|----|----|----|----|----|----|
| add  | 1.0× | **1.5×** | 3.5× | **1.4×** | 3.4× |
| relu | 1.0× | **1.5×** | 2.5× | **1.6×** | 2.5× |

What the run shows, and only what it shows:

- **Contiguous elementwise rows moved toward the raw-buffer C++ loop.**
  On this run, contiguous `add`/`relu` on `NativeTensorCore` and
  `NativeTensor` land around **1.5× NumPy at `1000x1000` — essentially
  matching the flat raw-buffer kernel** (≈1.0×) and closing most of the
  gap the odometer used to carry. The remaining margin over raw buffer
  is output-storage allocation and Python call overhead, not traversal.
- **Non-contiguous view rows stayed on the generic odometer path** and
  remain slower — about **3.5×** (add) and **2.5×** (relu) at
  `1000x1000`, versus the contiguous ~1.5×. Same op, contiguous vs
  strided, side by side: that spread is exactly the cost the fast path
  removes, and it is retained (not regressed) for the strided case the
  odometer must still serve.
- **Matmul and `contiguous_copy` are unchanged**, as intended — they
  were out of v1.14 scope (matmul is a different triple-loop kernel;
  `contiguous_copy` materializes a transposed view through the odometer
  by definition). Their rows sit where they did before.
- **`NativeTensor` still tracks `NativeTensorCore` closely** (e.g. add
  1.4× vs 1.5×, relu 1.6× vs 1.5×), reinforcing the v1.12 conclusion
  that the wrapper is thin — and confirming that an optimization placed
  *below* the wrapper reaches it automatically, with no wrapper change.
- **NumPy/BLAS remains the baseline to respect.** Nothing here is faster
  than NumPy; the elementwise wins are about *approaching* NumPy and the
  raw-buffer loop, and matmul stays roughly an order of magnitude behind
  NumPy's BLAS. Small-array rows still lose to NumPy on ctypes/conversion
  overhead, as before.

As always, correctness is verified against each layer's own NumPy
reference before any timing, timings are medians after warmup, and
**no test asserts a speed or ratio** — timing is measured and reported,
never gated. Broadcasting and reductions remain separate, later Phase A
work; this milestone only reports the contiguous elementwise result.

### Contiguous fast-path — implementation (v1.14)

v1.14 implements the fast path the v1.13 document designed. Beside the
generic odometer kernels, `cpp/src/` now carries flat, index-free
loops — `tf_core_relu_contiguous`, `tf_core_add_contiguous`,
`tf_core_subtract_contiguous`, `tf_core_multiply_contiguous` — that take
`numel` and the operand offsets instead of shape/strides.
`NativeTensorCore.relu` and `_binary_core_op` select them when every
operand is row-major contiguous (the output is always freshly allocated
contiguous storage); if any input is a strided view, the existing
generic odometer kernel runs unchanged. The choice is driven by the
`contiguous` flag already computed at view construction, so it costs
nothing to make.

The two paths are **bit-for-bit equivalent** — for a contiguous tensor
the odometer's source sequence is exactly `offset, offset+1, …`, so the
flat loop reads the same elements in the same order, with no
reassociation, FMA, or SIMD horizontal reductions that could change
float64 rounding. Nonzero offsets are handled by starting from
`data + offset`, so a contiguous row slice (`narrow` along axis 0) takes
the fast path too. Scalars and size-1 dimensions fall out naturally as
`numel == 1` (or a plain contiguous walk). The v1.14 tests assert exact
equality (`np.array_equal`) against both NumPy and the generic path on
value-identical non-contiguous inputs; transposed and inner-narrowed
views keep matching NumPy through the retained odometer path.

Nothing above `NativeTensorCore` changed: `NativeTensor` inherits the
fast path with **no wrapper edits**, exactly as the design predicted
(the v1.12 benchmarks had already shown the wrapper was not the
bottleneck). Semantics are untouched — exact-shape only, no
broadcasting, no reductions, no autograd, no `Tensor` integration, no
CUDA — and the public kernel lists (`list_kernels()`,
`tensor_core_kernels`) are unchanged, since the new kernels are internal
traversal variants, not new operations.

The **performance impact is a benchmark question, not a claim made
here.** Running `benchmarks/cpp_backend.py` shows the contiguous
`tensor core` / `native tensor` elementwise rows moving toward the flat
`cpp raw buffer` row while the `… (view)` rows stay on the odometer —
but exact numbers are hardware-dependent and nothing asserts a speedup.
An honest, tabulated impact report over the same suite is the next
milestone, **v1.15**.

### Contiguous fast-path — design (v1.13)

v1.13 is **design-only**: no kernels change. It follows directly from the
v1.12 benchmark finding — that the elementwise gap to NumPy comes from the
generic shape/stride odometer traversal in the native runtime, not from
the `NativeTensor` wrapper (the `native tensor` rows track their `tensor
core` rows closely). The design specifies a contiguous fast path for the
elementwise kernels (`relu`/`add`/`subtract`/`multiply`): contiguous
inputs and outputs use a flat, index-free pointer loop, while
non-contiguous views keep the current odometer traversal. The branch
lives in the `NativeTensorCore`/native-kernel layer, so `NativeTensor`
inherits it with no wrapper changes; results stay bit-for-bit identical
and error/shape semantics are untouched. Because nothing is implemented
yet, **no performance gain is claimed** — the full design, scope, tests,
and risks are in
[native_contiguous_fast_path_design.md](native_contiguous_fast_path_design.md).
Implementation is v1.14.

### NativeTensor — benchmark coverage (v1.12)

v1.12 extends the benchmark suite to characterize the `NativeTensor`
wrapper. Each operation is now measured across up to four layers:
**NumPy**, the **raw-buffer C++ kernels**, the **NativeTensorCore**
runtime, and the **NativeTensor** wrapper — for `add`, `relu`, `matmul`,
their strided-view forms (`x.T.relu()`, `x.T.matmul(y)`), and
`contiguous_copy` (materializing a transposed view; no raw-buffer analog,
so NumPy's `ascontiguousarray` is the baseline).

The point is the gap between a `tensor core` row and its `native tensor`
row: that difference is exactly the wrapper's extra **ownership,
lifetime, and conversion bookkeeping** layered on top of the same native
compute. In practice the two rows sit close together — the wrapper is
thin — which is the honest thing to show.

```
uv run python benchmarks/cpp_backend.py          # full sizes
uv run python benchmarks/cpp_backend.py --quick  # fast smoke run
```

As ever this is **characterization, not a performance claim**: NumPy
wins (often dramatically for matmul), the C++ kernels are naive
single-threaded reference loops, and nothing here asserts a speedup.
Correctness is verified against each layer's own NumPy reference before
any timing (view rows legitimately compute transposed results); timings
are medians after warmup; results are hardware-dependent. The benchmark
writes no files and is never run by pytest — a lightweight test only
checks the plan/row structure.

### NativeTensor — runnable example and polish (v1.11)

With the wrapper feature-complete as a forward-only tensor, v1.11 makes
it demonstrable. `examples/native_tensor_demo.py` is a small, fast,
deterministic tour — construction, `relu`/`add`/`matmul`, the views
(`reshape`/`T`/`narrow`), `contiguous_copy`, and explicit `close()` /
`with` — printing small NumPy-materialized outputs:

```
uv run python examples/native_tensor_demo.py
```

It needs the compiled backend; if it is not built, the script prints the
build instructions and exits cleanly rather than raising. `NativeTensor`
also has a metadata-only `repr` (`NativeTensor(shape=(2, 3),
contiguous=True)`, or `NativeTensor(closed)`) that never materializes
data and is safe on a closed tensor.

To be exact about what `NativeTensor` is: it is an **experimental,
forward-only** wrapper over the native C++ runtime, and it is
**separate from `tensorforge.Tensor`** — the two never mix. It has **no
autograd** (no `requires_grad`/`grad`/`backward`), **no implicit
dispatch** and no silent NumPy fallback (data crosses only through
`from_array` / `to_numpy`), **no CUDA**, and **no Python operator
overloads** (compute is method-only: `a.add(b)`, not `a + b`). It makes
**no performance claims** — it is a correctness and design experiment;
NumPy remains the reference implementation.

### NativeTensor — view ops (v1.10)

v1.10 gives `NativeTensor` the metadata-only view operations, delegating
to `NativeTensorCore`'s existing views. `reshape`, `transpose`, `T`, and
`narrow` return **borrowing** wrappers (`owns_core` is False) that share
the parent's storage — no data is copied — while `contiguous_copy`
returns a fresh **owning** wrapper that materializes the data into
row-major contiguous native storage:

```python
from tensorforge.experimental import NativeTensor

t = NativeTensor.from_array(np.arange(6.0).reshape(2, 3))

t.reshape((3, 2))     # borrowing view, same storage
t.transpose()         # all axes reversed; t.transpose(1, 0) works too
t.T                   # NumPy's .T
t.narrow(1, 1, 2)     # keep 2 positions of dim 1 from index 1
t.T.contiguous_copy() # a new OWNING, contiguous NativeTensor

t.transpose().relu()          # compute runs over a strided view directly
t.T.matmul(other)             # transposed view multiplies without copying
```

Lifetime follows the runtime beneath it: closing a borrowing view leaves
the owner (and sibling views) alive, while closing the owner releases the
shared storage — after which the views' data access (`to_numpy`,
compute) raises `RuntimeError`. `contiguous_copy` is independent: closing
it never touches the original. Invalid reshapes, non-permutation
transpose axes, and out-of-bounds `narrow` raise the same clear
`ValueError`/`TypeError` as `NativeTensorCore`.

Still forward-only and still **not** `tensorforge.Tensor`: no autograd,
no Tensor integration, no CUDA, and no Python operator overloads.

### NativeTensor — forward compute ops (v1.9)

v1.9 gives `NativeTensor` its forward-only compute methods, each
delegating to the `NativeTensorCore` kernel beneath and returning a
**new owning** `NativeTensor`:

```python
from tensorforge.experimental import NativeTensor

a = NativeTensor.from_array([[1.0, -2.0], [3.0, 4.0]])
b = NativeTensor.from_array([[5.0, 6.0], [7.0, 8.0]])

a.relu()          # max(x, 0), a new owning NativeTensor
a.add(b)          # a + b        (also subtract, multiply)
a.matmul(b)       # (m, n) @ (n, p)
a.relu().add(b).matmul(b)   # ops chain
```

The behavior mirrors the native runtime exactly. Elementwise ops
(`add`/`subtract`/`multiply`) require **identical shapes — no
broadcasting** (a `(2, 3)` with a `(3,)` raises `ValueError`); `relu` is
unary and takes any shape; `matmul` is strictly 2-D `(m, n) @ (n, p)`.
Wrong operand types raise a clear `TypeError` naming `NativeTensor`
(passing a raw NumPy array or list fails loudly), and computing on or
with a closed tensor raises `RuntimeError`. The original operands are
never consumed — each op allocates a fresh contiguous result.

Still forward-only and still **not** `tensorforge.Tensor`: no autograd,
no `requires_grad`/`grad`/`backward`, no Tensor integration, and no
Python operator overloads yet (`__add__`, `__matmul__`, ... are
deliberately absent — compute is method-only for now). View ops
(`reshape`, `transpose`, `T`, `narrow`) are still to come in v1.10.

### NativeTensor — minimal wrapper (v1.8)

v1.8 implements the *shell* of the forward-only wrapper the v1.7 design
laid out — constructors, metadata, conversion, and lifetime only. No
compute ops, no view ops yet (those are v1.9 and v1.10). It lives in its
own opt-in package; `import tensorforge` never touches it.

```python
from tensorforge.experimental import NativeTensor

with NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]]) as t:
    t.shape, t.strides, t.ndim, t.numel, t.contiguous  # layout metadata
    t.to_numpy()                                        # fresh float64 copy
# released on block exit; t.close() works too (idempotent)

z = NativeTensor.zeros((2, 3))
f = NativeTensor.full((2, 2), 7.0)
z.owns_core, z.closed        # ownership / lifetime state
z.close()
```

`NativeTensor` wraps a single `NativeTensorCore`: constructors
(`from_array`, `zeros`, `full`) own the core they create, so `close()`
(or a `with` block) releases the native storage. A closed tensor rejects
metadata and `to_numpy()` with a clear `RuntimeError`, while `closed`
and `owns_core` stay readable. Conversion crosses the native boundary
only by explicit call — `from_array` enters, `to_numpy` exits, both as
copies.

It is deliberately **not** `tensorforge.Tensor`: no autograd, no
`requires_grad`/`grad`/`backward`, no optimizer/Module integration, no
CUDA. And it is only a shell — `relu`/`add`/`subtract`/`multiply`/
`matmul` and the view ops exist on `NativeTensorCore` but are not
exposed on `NativeTensor` yet, by design. Constructors need the compiled
backend; if it is unbuilt they raise the same build-instructions
`ImportError` as the rest of the native runtime. The full design,
including the staged plan, is in
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md).

### Forward-only native tensor wrapper — design (v1.7)

v1.7 is **design-only**: no code ships. It writes down the Stage-2 plan
for a future forward-only convenience wrapper over `NativeTensorCore`
(likely named `NativeTensor`) — its purpose, non-goals, ownership and
lifetime rules, the v1.6 conversion contract it inherits, a minimal API
sketch, error/shape behavior, a testing plan, and a staged v1.8–v1.11
implementation sequence. The full design is in
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md). It
stays forward-only, autograd-free, float64, exact-shape, and explicitly
**not** `tensorforge.Tensor`. Implementation does not begin until v1.8.

## What might come next

The contiguous elementwise fast path is **implemented** (v1.14) and
**reported** (v1.15); native broadcasting is **implemented** (v1.17);
native `sum`/`mean` reductions are **implemented** (v1.19); the
dtype/device metadata contract was **designed** (v1.20) and is now
**implemented** (v1.21, float64/cpu only) — which **closes Phase A,
the native CPU runtime, in code**. **Phase B is under way**: the native
autograd design (v2.0) is complete, v2.1 implemented its metadata
skeleton and reverse-topological backward driver, **v2.2 wired the core
operations into that engine** — `add`/`subtract`/`multiply`/`relu`/`sum`/
`mean`/`matmul`/`reshape`/`transpose`/`T`/`contiguous_copy` are
differentiable, broadcasting backward runs through the native
`unbroadcast` reduction, and its one new kernel is the fused
`relu_backward` — **v2.3 completed the view-backward set** with `narrow`
backward through the native `tf_core_narrow_backward` scatter kernel (the
odometer dual of `sum`), **v2.4 gave the graph an explicit lifetime** —
one-shot `backward(retain_graph=False)` by default, opt-in
`retain_graph=True` reuse, deterministic freed-graph errors, and
snapshot-based failure safety (a Python-only change; no kernel touched) —
and **v2.5 (above) characterized the whole stack** with a measurement-only
benchmark harness (four modes separating forward-native, graph
construction, fresh forward+backward, and repeated retained backward; a
single honest hardware snapshot; no speed assertions). Gradients are
`NativeTensor`-backed float64/cpu and enforce `grad.dtype ==
tensor.dtype` / `grad.device == tensor.device` against the fields v1.21
made real. `NativeTensorCore` and the C++ kernels still own no graph
state, and there is still no Tensor integration and no CUDA today.
**v2.6 closed Phase B** with cross-cutting guardrail tests, and **Phase C —
the native training stack — is now under way: v3.1 shipped
`NativeParameter` and the parameter-registration contract, v3.2 shipped
`NativeModule` — automatic assignment registration, deterministic
identity-deduplicated recursive traversal, recursive `zero_grad()`, and
train/eval state propagation — v3.3 shipped the in-memory state
dictionary contract: `state_dict()` snapshots and atomic
identity-preserving `load_state_dict()` — v3.4 shipped the first
concrete layer, `NativeLinear`, its backward supplied entirely by the
existing autograd — v3.5 completed the first composable model surface
with `NativeReLU` and `NativeSequential` — v3.6 added the first native
loss, `NativeMSELoss`, closing the forward side of the training story:
model → loss → backward now runs natively end to end — v3.7 made
parameter mutation safe: value-version counters, the controlled
`copy_value_` no-grad mutation primitive, version-incrementing state
loads, and deterministic stale-graph errors where a mutated
parameter's forward value is needed by backward — and v3.8 (above)
added the first native optimizer, `NativeSGD`: identity-deduplicated
parameter storage, a validated learning rate, and a two-phase
mutation-atomic `step()` that stages graph-free native updates and
commits them through `copy_value_`, plus `zero_grad()` — and v3.9
(above) completed **the first end-to-end native CPU training proof**:
a deterministic 25-step MLP regression whose loss falls monotonically
by 99.5%, built entirely from fresh per-iteration graphs over the
existing stack — v3.10 is the integration checkpoint:
honest README/summary/architecture presentation, the canonical
[native support matrix](native_support_matrix.md), documentation and
export guardrails, and CI/hygiene audits, making the branch ready for
its first pull request into `main` — and v3.11 (above) added the
optimizer math primitives: differentiable native `sqrt` and
`reciprocal` through kernels → core → wrapper → autograd, with
saved-forward-result backwards that record no parameter versions and
IEEE float64 exceptional-value semantics — the reusable math NativeAdam
needs.** Phase C then completed: v3.12 — NativeAdam, v3.13 —
native optimizer state, v3.14 — native checkpointing and deterministic
file resume, and **v3.15 — native training stack guardrails and Phase C
completion** (the integrated completion test suite, documentation
completion, and build/CI/hygiene verification), which **closes Phase C
in code**. A general `divide` operation remains deliberately unshipped
(`reciprocal` + `multiply` compose what the stack needs). The **native
CNN stack (Phase D) then completed** across milestones D0–D12:
`NativeFlatten`, the differentiable `conv2d` operation and its trainable
`NativeConv2d` module, the `maxpool2d` operation with its private saved
winners and the `NativeMaxPool2d` module, the deterministic end-to-end
CNN training + exact checkpoint-resume proof, cross-cutting integration
tests, honest CNN benchmarks, and ASan/UBSan validation. **Phase E —
Native Classification and Stable Math — then completed** across
milestones E0–E10, whose contract is locked in
[native_classification_design.md](native_classification_design.md) (E0):
the differentiable `exp` and `log`, the fused stable `softmax` and
`log_softmax`, the fused `cross_entropy` Core contract and the
differentiable operation over it, `NativeCrossEntropyLoss` and the
reporting-only `native_accuracy`, a deterministic classification
training run with exact checkpoint resume, an honest characterization
benchmark, and phase closure under Release/Debug builds and Clang
ASan/UBSan/LeakSanitizer. **The latest phase is Phase F — Native
Normalization and Stateful Buffers — and it is complete (F0–F9).** Its
contract is locked in
[native_normalization_design.md](native_normalization_design.md)
(milestone **F0**, complete: design and repository reconciliation only,
adding no numerical behavior), **F1** is complete (the private atomic
native-buffer state transaction, the `load_state_dict` refactor onto it,
and the `persistent_buffers` capability reconciliation — no normalization
mathematics), and **F2** is complete (`NativeLayerNorm` — the first
native normalization module: stateless, differentiable through the mean
and the population variance, composed entirely from existing native
operations with `sqrt(var + eps)` ordering and no kernel, ABI symbol,
`NativeTensorCore` method, custom backward, or `NativeTensor`
normalization operation; now in `NATIVE_MODULES` and the exports, with
`"layernorm"` removed from `UNSUPPORTED`), and **F3** is complete
(`NativeBatchNorm1d` — the first stateful native numerical module:
`(N, C)` batch normalization with differentiable training statistics,
persistent native running-statistic buffers advanced by a graph-free
atomic two-buffer transaction, and evaluation from graph-safe immutable
snapshots; again composed from existing operations, so again no kernel,
ABI symbol, `NativeTensorCore` method, custom backward, or
`NativeTensor.batch_norm` operation, and the checkpoint format stays at
version 1; now in `NATIVE_MODULES` and the exports, with `"batchnorm"`
**kept** in `UNSUPPORTED` until the NCHW shape ships), and **F4** is
complete (`NativeBatchNorm2d` — NCHW `(N, C, H, W)` batch normalization
reducing over N, H, and W, over the same shared private implementation
and adding only shape/layout configuration plus the shared channelwise
affine step; now in `NATIVE_MODULES` and the exports, and with both
shapes live `"batchnorm"` has **left** `UNSUPPORTED`), and **F5** is
complete (the exhaustive state/checkpoint, ownership, and graph-safety
hardening — a focused `tests/test_native_normalization_state.py` plus
narrow additions to the generic buffer and checkpoint suites — proving
§7–§10 of the design by executable test rather than by prose; **tests and
documentation only, no numerical behavior and no new capability**, with
the exports, every capability registry, and the version-1 checkpoint
format all unchanged), and **F6** is complete (the deterministic
normalized training and exact checkpoint-resume proof
`examples/native_normalization_training.py` — a
`Linear → BatchNorm1d → ReLU → LayerNorm → Linear` regressor, both
normalization families in every forward, trained with `NativeAdam` and
`NativeMSELoss`, resuming an interrupted run into a fresh model/optimizer
pair that reproduces the losses, parameters, NativeAdam state, BatchNorm
running statistics, final training prediction, and evaluation-mode output
exactly; **one example and its integration test, no capability or schema
change, format version 1 unchanged**), and **F7** is complete (the honest
benchmark characterization `benchmarks/benchmark_native_normalization.py`
— nine correctness-gated cases covering both LayerNorm directions, all
three BatchNorm1d paths, all three BatchNorm2d paths, and one complete
normalized training step, with `stable_tensorforge` references where an
honest equivalent exists and `native_only` timing for the three
BatchNorm2d cases (no public stable `BatchNorm2d` exists), medians with
min/max/spread, `--smoke`/`--json`, and **no result file, no speed
assertion, no committed timing number, and no CI timing threshold**;
**measurement only, no capability and no production change**). The
numerical normalization module surface, its state/checkpoint/graph-safety
contracts, one exact normalized resume, and the honest characterization
are therefore complete, and **F8** is complete too (the cross-cutting
integration and semantic guardrails `tests/test_native_phase_f.py` — an
integrated convolution/normalization/pooling/classification model trained
and resumed exactly, the three saved-resource families proved to coexist
safely, the buffer/parameter mutation distinction, the versioning
archetypes, shared and frozen parameters, a non-contiguous NCHW input,
honest per-boundary failure atomicity, and reality-derived capability
guardrails; **tests and documentation only, no capability**). Milestone
**F9 is complete** as well — the phase closure: Release **and** Debug
Windows builds each passing the full existing 10-test CTest suite with
zero project warnings and the active runtime proved to stay Release; a
fresh Clang 18.1.3 ASan+UBSan build whose instrumentation is proved by
`nm -D` (22 `__asan*`, 13 `__ubsan*`) and by the library's refusal to
load without the sanitizer runtime; 10/10 sanitized native CTests with
leak detection enabled; 1,968 sanitized normalization-focused Python
tests, the F6 example, and the F7 benchmark smoke path all clean; and a
practical LeakSanitizer lifecycle returning native live storage
**exactly** to baseline with no TensorForge-attributable leak frame and
no suppression file — **validation and documentation only, adding no
numerical capability**. **Phase F is therefore complete**, and there is
still no normalization operation, kernel, or C ABI export.
Dropout and a native RNG sit **beyond** Phase F. CUDA experiments
remain a separate future branch (where `device` gains a second value),
and an AMP / Tensor Core path is where `dtype` later gains
float16/bfloat16. The Python framework stays the reference implementation
throughout.
