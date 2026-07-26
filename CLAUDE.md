# TensorForge — project instructions

## What this is

TensorForge is a from-scratch deep learning framework built in
Python + NumPy — a serious ML systems project covering PyTorch-style
framework internals, inspired by Daedalus ML but not a copy. It was
developed milestone by milestone (v0.1 … v3.0), each one small,
tested, and readable. The Python framework line is complete as of
v3.0. The experimental C++ native line is merged into `main` and lives
in the explicit `tensorforge.backends` / `tensorforge.experimental`
namespaces rather than on a separate advanced branch. It
has completed Phases A–F: Phase A — CPU runtime, Phase B — native
autograd, Phase C — the native training stack, and Phase D — the
native CNN stack, through Advanced C++ v3.16; Phase E — native
classification and stable math — is *complete* (E0–E10), its contract
locked
in `docs/native_classification_design.md` with milestones E1–E4 (the
differentiable native `exp`, `log`, and the fused stable `softmax` and
`log_softmax`), E5 (the fused `cross_entropy` **Core** contract —
`NativeTensorCore.cross_entropy_forward`/`cross_entropy_backward`), and
E6 (the differentiable `NativeTensor.cross_entropy(targets,
reduction="mean")`, one autograd node with graph-owned saved
probabilities and no logits reread), and E7 (the stateless
`NativeCrossEntropyLoss` module delegating to that operation, and the
reporting-only `native_accuracy` — explicit `to_numpy()` + NumPy
argmax, no graph, in the new `NATIVE_METRICS` inventory), and E8 (the
deterministic classification training and exact checkpoint-resume proof
— `examples/native_classification_training.py`: a three-class native
CNN classifier over raw logits, 40 `NativeAdam(lr=0.05)` steps, loss
1.159638 → 0.000101, accuracy 0.3333 → 1.0000, interrupted at step 15
and resumed into a fresh model/optimizer pair that matches exactly;
example, tests, and docs only — no new capability), and E9 (the honest
characterization benchmark
`benchmarks/benchmark_native_classification.py`: seven cases, each
correctness-gated before timing, each labelled with the reference it
used, medians with spread after warm-up, `--smoke`/`--json` modes, and
**no speed assertion or timing threshold anywhere**), and E10 (phase
closure: `tests/test_native_phase_e.py` cross-cutting integration,
Release and Debug builds with 10/10 CTests each, Clang ASan/UBSan and
LeakSanitizer validation, and documentation reconciliation — no new
numerical capability) all shipped. **Phase F — Native Normalization and
Stateful Buffers — is the latest phase and is *complete* (F0–F9):**
milestone F0 is complete (the architecture contract in
`docs/native_normalization_design.md` plus repository reconciliation —
**no numerical behavior**), F1 is complete (the private atomic
native-buffer state transaction in
`src/tensorforge/experimental/_native_state.py`, `load_state_dict`
refactored onto it, and `persistent_buffers` added to `STATE_SUPPORT` —
state management and capability reporting only), and F2 is complete
(`NativeLayerNorm` — the first native normalization module: stateless
(no buffers, identical in train and eval), differentiable through the
mean and the population variance, **composed entirely from existing
native operations** — `mean`/`subtract`/`multiply`/`add`/`sqrt`/
`reciprocal`, `sqrt(var + eps)`, no Bessel correction — adding no C++
code, kernel, C ABI symbol, ctypes declaration, `NativeTensorCore`
method, custom backward, functional helper, or `NativeTensor.layer_norm`
operation; `weight`/`bias` `NativeParameter`s only when
`elementwise_affine=True`; `"NativeLayerNorm"` in `NATIVE_MODULES` and
the exports, and `"layernorm"` removed from `UNSUPPORTED`), and F3 is
complete (`NativeBatchNorm1d` in
`src/tensorforge/experimental/native_batchnorm.py` — the **first
stateful native numerical module**: strictly `(N, C)` batch
normalization, again composed entirely from existing native operations,
adding no C++ code, kernel, C ABI symbol, ctypes declaration,
`NativeTensorCore` method, custom backward, functional helper, or
`NativeTensor.batch_norm` operation. Training normalizes with this
batch's **differentiable** population statistics (`sqrt(var + eps)`, no
Bessel correction, gradients through the mean *and* the variance) and
advances the persistent native `running_mean`/`running_var` buffers by
`(1 - momentum)*running + momentum*batch` from the *same* batch
statistics — computed **graph-free** via `detach()` and committed as one
**atomic two-buffer transaction** through the F1 primitive, preserving
both Python identities, closing each replaced core exactly once, and
moving **no** parameter version. Evaluation reads **independent owning
graph-free `(1, C)` snapshots** of those buffers, so no registered
buffer is ever a rereadable graph operand and a later training step, a
buffer-only `load_state_dict()`, or a buffer-only
`load_native_checkpoint()` cannot change an earlier eval graph's
gradient (a *full* checkpoint load also replaces `gamma`/`beta`, so the
unchanged v3.7 parameter-version guard correctly stales that graph — a
parameter contract, never a buffer effect); the snapshots ride the existing `graph_resources`
contract and release exactly once with the graph history. `gamma`/`beta`
always exist (no `affine=False`, `track_running_stats`, or
`num_batches_tracked`); state order is `gamma`, `beta`, `running_mean`,
`running_var`; the checkpoint format stays version 1;
`"NativeBatchNorm1d"` is in `NATIVE_MODULES` and the exports, while
`"batchnorm"` stayed in `UNSUPPORTED`), and F4 is complete
(`NativeBatchNorm2d`, the second public class in the same file — NCHW
`(N, C, H, W)` batch normalization reducing over **N, H, and W**, so
each channel gets one population mean and variance over `N * H * W`
values. It is built on the **same** private `_NativeBatchNorm` and
declares *only* `_INPUT_NDIM = 4`, `_REDUCTION_AXES = (0, 2, 3)`,
`_TRAILING_DIMS = 2`, `_LAYOUT`, and `_CHANNELS_LAST = (0, 2, 3, 1)` —
every method is inherited by function identity. The one shared piece F4
added is the channelwise affine: rank-1 `gamma`/`beta` broadcast from
the *trailing* axis, so the **activation** is transposed to
channels-last for the affine step and back again (then materialized
contiguous) rather than reshaping the parameters — which keeps `gamma` a
direct versioned `multiply` operand and preserves the existing
stale-parameter guard exactly. Statistics are `(1, C, 1, 1)`, running
buffers stay `(C,)`, the checkpoint format stays version 1, and
`"NativeBatchNorm2d"` is in `NATIVE_MODULES` and the exports; with both
shapes live `"batchnorm"` has **left** `UNSUPPORTED`, which now reads
exactly `("dropout", "float32", "cuda", "amp")`). **That completes the
numerical normalization *module* surface — not Phase F.** **F5 is
complete** (the exhaustive state, checkpoint, ownership, and graph-safety
hardening — a focused `tests/test_native_normalization_state.py` plus
narrow additions to the generic buffer and checkpoint suites, proving
§7–§10 of the design by executable test rather than by prose: canonical
dotted buffer keys, independent state snapshots, strict/non-strict loads,
exact never-casting metadata validation, mixed parameter/buffer
transaction atomicity, buffer identity across state and checkpoint loads,
exact eval-output reproduction, the buffer-only-versus-full stale-graph
distinction, the save/corrupt-load failure boundaries, eval-graph snapshot
safety under `retain_graph` and a failed retryable backward, and the
live-storage baselines; **tests and documentation only — no numerical
behavior and no new public capability**, with the exports, every
capability registry, and the version-1 checkpoint format all exactly what
F4 left, and no production behavior changed). **F6 is complete** (the
deterministic normalized training and exact checkpoint-resume proof —
`examples/native_normalization_training.py`: a `NativeNormalizedRegressor`
(`Linear → BatchNorm1d → ReLU → LayerNorm → Linear`, both normalization
families in every forward, BatchNorm the only stateful module) trained for
24 deterministic `NativeAdam` steps with `NativeMSELoss` (98.9% loss
reduction), with two uninterrupted runs proved bit-identical and an
interrupted run resumed into a **fresh** model/optimizer pair that
reproduces the remaining loss suffix, every parameter, the NativeAdam
state, both BatchNorm `running_mean`/`running_var`, the final
training-step prediction, and the final **evaluation-mode** output exactly
— checkpoint format version 1 unchanged, training flags runtime-only;
**one example and its integration test, adding no capability, operation,
kernel, schema field, benchmark, or export, and changing no production
behavior**). **F7 is complete** (the honest benchmark characterization —
`benchmarks/benchmark_native_normalization.py`, `BENCHMARK_NAME =
"tensorforge.native_normalization"`, version `"1.0"`: exactly nine cases
in this order — `layernorm_forward`, `layernorm_backward`,
`batchnorm1d_training_forward`, `batchnorm1d_eval_forward`,
`batchnorm1d_backward`, `batchnorm2d_training_forward`,
`batchnorm2d_eval_forward`, `batchnorm2d_backward`, and
`normalized_training_step`. Every case runs its correctness gate
**before** the timing helper is ever reached, so a failed gate publishes
no timing and the CLI exits nonzero with a clean stdout. Six cases are
labelled `stable_tensorforge` and run `tensorforge.nn`/`tensorforge.optim`
on the *same* inputs, epsilon, momentum, affine values, running state,
initial parameters, and optimizer hyperparameters; the three
**BatchNorm2d** cases are labelled `native_only` and publish **no** timing
ratio, because the stable line has no public `BatchNorm2d` — they keep a
rigorous correctness oracle instead (an explicit NumPy NCHW
population-statistics formula, an independent channelwise-affine probe,
eval-mode state neutrality with the registered buffers proved absent from
the graph, and for the backward the stable `BatchNorm1d` on the
equivalent `(N*H*W, C)` sample matrix transformed back to NCHW, which is a
correctness oracle **only** and deliberately not timed). Timing uses
`time.perf_counter_ns()` with warm-up, one call per sample, every sample
retained, setup and cleanup outside the timer (graph construction inside
it for the forward and training-step cases, outside it for the
backward-only cases), a fresh module per training-mode repetition because
the forward advances persistent state, and median/min/max/spread
reporting. `--case`/`--warmup`/`--repetitions`/`--smoke`/`--json` exist,
the payload is fully JSON-native, **no result file of any kind is
written**, and **no speed assertion, committed timing number, or CI timing
threshold exists anywhere**; **measurement only — one harness and its
test, no capability, operation, kernel, C ABI symbol, ctypes declaration,
Core method, schema field, example, or export, and no production behavior
changed**). **F8 is complete** (the cross-cutting integration and
semantic guardrails — `tests/test_native_phase_f.py`: one test-only
`NativePhaseFClassifier` (`NativeConv2d(1, 4, 3)` → `NativeBatchNorm2d(4)`
→ `NativeReLU` → `NativeMaxPool2d(2)` → `NativeFlatten` →
`NativeLinear(16, 8)` → `NativeBatchNorm1d(8)` → `NativeReLU` →
`NativeLayerNorm(8)` → `NativeLinear(8, 3)` → **raw logits** →
`NativeCrossEntropyLoss`) over the E8 fixed twelve-image three-class
dataset, trained for 12 deterministic `NativeAdam(lr=0.05)` steps,
interrupted at step 5, checkpointed, and resumed into a **fresh**
model/optimizer pair that reproduces the loss suffix, every parameter,
the NativeAdam state, **all four** running-statistic buffers, the final
training logits, and the final evaluation-mode logits, predictions, and
accuracy by **exact equality** (format version 1 unchanged, training flag
runtime-only, identities preserved). It also proves the three
saved-resource families — BatchNorm eval snapshots, MaxPool2d winners,
and cross-entropy probabilities — coexisting in one eval graph and
releasing exactly once with no registered buffer object *or storage*
reachable from the graph; buffer-only mutation (including a real
buffer-only `load_native_checkpoint()` over all four registered objects)
leaving an earlier eval graph's gradients exactly equal to a clean
control, while a full checkpoint load or a `copy_value_` on a
normalization affine parameter correctly stales it through the unchanged
v3.7 **parameter** rule; the Phase-E versioning archetypes (saved-output
`exp`, live-reread `log`, saved-probability cross-entropy) meeting
BatchNorm snapshots; shared parameters deduplicating to one slot/one
update/one version increment; frozen parameters staying registered and
persisted but skipped; a non-contiguous NCHW input through the whole
stack in both modes; strict stable/native separation; **honest**
per-boundary failure atomicity (A: a BatchNorm transaction failure rolls
*that pair* back while an earlier module's committed transaction
legitimately stands — transactions are **per module**, and one whole
training step is *not* globally transactional; B: a post-forward failure
does not retroactively roll back committed running updates; C: an
optimizer staging failure commits nothing and leaves the gradients
retryable; D: a stale-parameter backward keeps the forward's update; E: a
checkpoint-load commit failure restores everything); error-state
recovery; a NumPy/conversion tripwire over one complete integrated step;
live-storage baselines across success **and** failure cycles; and
semantic capability/export/artifact guardrails derived from real
registries and files. **Tests and documentation only — no capability,
operation, kernel, C ABI symbol, ctypes declaration, schema field,
example, benchmark, or export, and no production behavior changed.**)
**F9 is complete** (the phase closure: fresh Windows Release **and**
Debug builds — Visual Studio 17 2022, MSVC 19.44.35228.0, CMake 4.4.0,
both out-of-source outside the repository — each passing the full
existing 10-test CTest suite (10/10 in 0.78 s and 0.97 s) with **zero
project compiler, linker, and CMake warnings**, the Debug library
written elsewhere so the active runtime stayed the Release DLL; a fresh
Clang 18.1.3 `-DTF_SANITIZE=address,undefined` build in WSL2 Ubuntu
24.04.4 with **instrumentation proved** — `nm -D` shows 22 `__asan*` and
13 `__ubsan*` dynamic symbols beside the 50 exported `tf_*` symbols, and
the library refuses to load without the sanitizer runtime; **10/10
sanitized native CTests** with `detect_leaks=1`; **1,968 sanitized
Python tests** across 32 normalization and dependency suites with zero
ASan and zero UBSan diagnostics; the F6 example reproducing its exact
resume and the F7 benchmark passing all nine correctness gates under the
sanitized library; and a practical LeakSanitizer lifecycle returning
native live storage **exactly** to baseline (0 → 0), whose remaining
process-exit allocations (925,710 bytes in 830 allocations) contain **no
TensorForge frame** — only CPython, libc, NumPy, `_ctypes`, and the ASan
runtime — with **no suppression file added**; plus documentation
reconciliation and durable semantic closure guardrails.
**Documentation and documentation-guardrail tests only — no numerical
capability, no C++, no CTest, no ABI or ctypes surface, no example, no
benchmark, and no production numerical file changed.**)
**Phase F is complete**, and no normalization operation, kernel, C ABI
symbol, or custom backward exists at all.
**Phase G — Native RNG and Dropout — is the current phase and is *in
progress*: milestones G0 (the architecture contract in
`docs/native_rng_dropout_design.md`), G1 (`NativeGenerator` and module
generator-state ownership), G2 (the deterministic stateless
Dropout-forward **Core**), G3 (the differentiable
`NativeTensor.dropout(p, *, generator)`, with an **explicit** required
keyword-only `NativeGenerator`), and G4 (the `NativeDropout` module and
its public export) are complete; G5–G10 have not
started.** G0 locked Python-managed generator state (an explicit
64-bit seed plus call counter and an algorithm identifier), stateless
native random kernels that receive the whole key for one call, inverted
Dropout with a graph-owned multiplier mask whose backward never rereads
the input, exactly one generator call consumed per **successful**
stochastic forward (and none on any failure, in evaluation mode, at
`p == 0`, or in backward), generator state registered as a fourth
`NativeModule` category, and native checkpoint **version 2** with a
locked version-1 compatibility rule. G0 is **design, documentation, and
guardrails only** — no `NativeGenerator`, kernel, C ABI symbol, ctypes
declaration, Core method, operation, module, export, or registry change
exists, `UNSUPPORTED` still reads `("dropout", "float32", "cuda",
"amp")`, and the checkpoint format is still version 1. G0 also locks a
lock-protected, token-validated generator reservation protocol (no two
callers can ever receive the same call index, and state replacement is
refused while a reservation is live), a checkpoint-version-2 generator
section that records the **alias topology** — every registered generator
path and its canonical target, so shared-versus-independent identity is
restored, not just the states — and **whole-checkpoint** transaction
atomicity, where any ordinary synchronous commit failure rolls back
parameters, buffers, optimizer state, and generator state together and
external process/interpreter death is the only documented exception.
G4 implements and exports `NativeDropout` but **does not** move the
capability boundary: `"dropout"` stays in `UNSUPPORTED` through G9 and
leaves it only at **G10**, after the full closure matrix passes, leaving
`("float32", "cuda", "amp")`. The format version becomes 2 at G5 — none
of this in G0.
**G1 is complete**: `src/tensorforge/experimental/native_generator.py`
ships `NativeGenerator` — a **pure-Python value holder** owning no native
storage and having **no `close()`**, carrying exactly four fields
(`algorithm` `"tensorforge.splitmix64"`, `algorithm_version` 1, an
unsigned-64-bit `seed`, and `calls`, the count of *committed* stochastic
calls) as read-only properties, with `state()` returning an independent
plain dict and `load_state()`/`reseed()`/`reset()` validating everything
before assigning anything; exact-`int` discipline (`bool`, NumPy scalars,
and `int` subclasses rejected); `seed=None` drawing once through
`secrets.randbits(64)` and nothing else consulting the clock, the process
id, an address, NumPy's global RNG, or Python's `random`; identity (never
value) semantics with `copy`, `deepcopy`, and pickle all refused; and the
private lock-protected token-validated transaction (`_reserve_call` →
`_commit_call`/`_abandon_call`) where one private `threading.RLock`
covers reservation, commit, cancellation, and every state read and write,
at most one reservation is live, a concurrent or reentrant second caller
fails *before* an index is minted, commit advances exactly once, cancel
never advances, stale/foreign/duplicate/finished tokens are inert,
`load_state`/`reseed`/`reset` are refused mid-reservation, and the
counter is checked under the lock at `2**64 - 1` and never wraps.
Reservation creation is a **two-phase claim / construct / publish /
deliver** transaction and the token is built with **no generator lock
held**: phase 1 rejects an active reservation, an existing claim, and an
exhausted counter, then publishes *only* an internal construction claim
(no active reservation, no counter movement, no serial movement); phase 2
constructs the token owning nothing, and on any failure — including
`MemoryError` and `KeyboardInterrupt` — a `finally` reacquires the lock,
verifies the matching claim, clears it, publishes nothing, and re-raises;
phase 3 reacquires the lock, verifies the claim, publishes the
reservation, advances the never-reused serial exactly once, and clears
the claim; phase 4 delivers the token. The four failure positions get
**different** cleanup, because clearing the claim does *nothing* once a
reservation is published: a failure between publication and delivery
would otherwise leave an active reservation whose only token is being
dropped — uncommittable, uncancellable, and blocking every later
reservation — so `_release_undelivered` cancels it, matching the token's
generator, serial, **and** index exactly, leaving `calls` untouched, and
leaving a newer, foreign, committed, or already-abandoned reservation
strictly alone. A failed delivery consumes an opaque serial, never a call
index. It takes only its own generator's lock and does no
callback-capable work under it, so it cannot touch the global
multi-generator order. `_deliver_reservation` is the private no-op seam
that makes that window addressable by tests instead of by real signal
timing.
Token construction is the one allocation in the path and allocation can
run finalization, so it happens outside the lock: **no user code,
callback, or generator-owned allocation runs while a generator lock is
held**, which is what makes finalizer or callback reentry unable to
invert the multi-generator lock order. While the claim stands, another
reservation, `load_state`, `reseed`, `reset`, and
`replace_generator_states` all raise `RuntimeError` and mutate nothing
(inspection still works). The lock stays an `RLock` for two reasons —
structurally, the multi-generator transaction re-enters through the same
`_snapshot_state`/`_assign_state` write seam it holds the locks around,
which a plain `Lock` self-deadlocks on; and residually, CPython can start
a collection at any container allocation, so a finalizer reaching the
remaining small allocations under the lock gets a deterministic refusal
instead of a hang. `load_generator_state_dict` runs the shared
`replace_generator_states` transaction in `native_generator.py`:
validate → acquire **every** unique target's lock in one global
identity-ordered sequence (so two loads over overlapping generators
arriving through different modules cannot deadlock) → recheck every
target for a published reservation *or a construction claim* while
holding them all → snapshot → non-failing integer writes, with the
rollback completing before any lock is released. No reservation can begin
on a target between the recheck and the end of the commit, and because
token construction holds no lock, a transaction reached from a finalizer
begins owning nothing and takes the same global order.
`NativeModule` gained `_generators` as a **fourth** registration category
(reserved name, assignment registration, `register_generator`,
one-category-per-name eviction in both directions, `__getattr__` /
`__delattr__` participation, deterministic identity-deduplicated
cycle-safe `generators()`/`named_generators()`, and the separate
`generator_state_dict()`/`load_generator_state_dict()` surface), and
`NativeGenerator` is exported from `tensorforge.experimental` only.
**G1 generates no random values by itself** — it shipped the state, and
the derivation, the kernel, and the Core method arrived at G2;
`state_dict()` is unchanged and still tensor-only; the
checkpoint format is still version 1 and does not serialize generator
state; and no *numerical* capability-registry value moved (`UNSUPPORTED`
still reads `("dropout", "float32", "cuda", "amp")`, dtypes/devices
unchanged). The one registry change G1 does make is reporting-only:
`STATE_SUPPORT` gained `"generator_state"` between `load_state_dict` and
`save_native_checkpoint`, a capability name (like `"persistent_buffers"`)
covering the generator registration and in-memory state surface — it
does **not** mean generator state is checkpointed, which is G5.
Reservation creation is failure-atomic at every position (a failed
construction publishes nothing and skips no serial; a failed *delivery*
cancels the exactly-matching published reservation), and `calls` is a
*count*, so
`2**64 - 1` is a reachable value, not a sentinel: reserving at
`2**64 - 2` succeeds and commits to `2**64 - 1`, reserving at
`2**64 - 1` is refused, and the counter never wraps.
**G2 is complete**: the deterministic **stateless Dropout-forward
Core**. New `cpp/include/tf_random_internal.h` and `cpp/src/random.cpp`
hold the exact locked `tensorforge.splitmix64` derivation as hidden
`namespace tf` functions — `splitmix64_mix` (`^= >>30`,
`* 0xBF58476D1CE4E5B9`, `^= >>27`, `* 0x94D049BB133111EB`, `^= >>31`),
`dropout_stream_key(seed, call) = mix64(seed + GOLDEN*(call + 1))`,
`dropout_element_bits(stream, i) = mix64(stream + GOLDEN*(i + 1))`, and
`dropout_uniform(bits) = (bits >> 11) * 2**-53` with a strict `u < p`
drop test — plus `dropout_forward_contiguous`, which writes the output
**and** the private multiplier mask in one pass with `1/(1 - p)` computed
once per call, so the mask holds exactly `0.0` or that scale. All
`std::uint64_t` wrapping arithmetic; **no** `<random>`,
`std::random_device`, `mt19937`, clock, process id, address, allocation
history, or static/thread-local state anywhere. The self-validating
guarded export `tf_core_dropout_forward(input, offset, output, mask,
count, seed, call_index, p)` rejects null handles, a negative offset or
count, a span exceeding its storage, a non-finite or out-of-range `p`,
and any aliasing between the input and either destination, writing
**nothing** to either destination when it rejects. Python gains one
ctypes declaration (the key as two `c_uint64` arguments),
`"tf_core_dropout_forward"` in `_CHECKED_KERNELS`, `"dropout_forward"` in
`TENSOR_CORE_OPS`, and the `NativeTensorCore.dropout_forward(p, *, seed,
call_index)` / private `_dropout_forward_with_mask` pair — the
`maxpool2d_forward` / winner-buffer split. Shared validators normalize
`p` (bool and `numpy.bool_` rejected, `numbers.Real` accepted, `p == 1`,
`p > 1`, `p < 0`, NaN, and ±inf rejected, `p == 0` accepted) and each key
half (exact `int` in `[0, 2**64 - 1]`; bool, NumPy scalars, and `int`
subclasses rejected). **The Core is stateless**: it takes no
`NativeGenerator` and never reserves, commits, cancels, inspects, or
mutates one, so a direct Core call leaves a live generator's seed,
`calls`, and reservation slot bit-identical. Randomness is keyed by the
**logical** row-major element index — Policy B materializes a
non-contiguous input first, so a transposed, narrowed, or nonzero-offset
view receives the same mask as a contiguous tensor of the same logical
shape. Output and mask are fresh **owning contiguous** cores aliasing
neither the input nor each other; the input and its metadata are never
mutated; allocation order is output-then-mask, and any failure — a failed
allocation, a failed native call, or a failed Python wrapper
construction — closes everything allocated so live storage returns
exactly to baseline and no caller can observe one lone result. Committed
known-answer vectors (`mix64` over six inputs, nine stream keys, the
element bits, the uniform conversion, and seven full twelve-element
keep/drop patterns covering seed `0`, a mixed seed, a high-bit seed, the
all-ones seed, call index `0`, a nonzero call index, the highest index a
generator can issue (`2**64 - 2`), and two probabilities) are asserted
**identically** in `cpp/tests/test_dropout_forward.cpp` and
`tests/test_native_dropout_core.py`; a test-only Python reference of the
derivation lives in the suite (never in production) and is pinned to
those vectors before generating any expectation. **G2 ships the Core and
nothing above it**: no autograd node, no
graph-owned saved mask, no Dropout backward kernel (that gradient is the
existing `multiply` over the saved mask), and no `NativeDropout`;
`UNSUPPORTED` still reads `("dropout", "float32", "cuda", "amp")` and the
checkpoint format is still version 1. One contract detail is recorded
rather than glossed: the design's **empty-tensor** row is implemented in
the kernel and the C ABI (`count == 0` draws and writes nothing), but the
native tensor representation rejects zero-size dimensions outright, so no
empty core can be constructed from Python — G2 proves the case at the two
layers where it is reachable and pins the representation's limit with a
test.
**G3 is complete**: the differentiable
`NativeTensor.dropout(p, *, generator)` in
`src/tensorforge/experimental/native_tensor.py`, plus exactly one
registry name — `"dropout"` appended to `AUTOGRAD_OPS`. **G3 changed no
C++, no C ABI symbol, no ctypes declaration, no `NativeTensorCore`
method, no module, no export, and no checkpoint-format change**, and
added no backward kernel: inverted Dropout's gradient is the existing
native `multiply` over the saved mask. The `generator` is **required and
keyword-only** — no default, process-global, or module-global stream, no
implicit per-call generator, and no NumPy or Python `random` fallback —
and `p` goes through the *same* `_normalize_dropout_probability` the Core
uses rather than a second rule. The operation owns the design's §5 call
transaction in this exact order: validate the receiver, the generator,
and `p`; return `self` (the caller's own object, un-copied) at `p == 0`
with no reservation, allocation, kernel call, or graph node; otherwise
reserve **one** call, binding the token and entering the cleanup boundary
as the very next action; read the key from the **reservation** — the
token's index, and the seed read while that live reservation makes every
state replacement raise, so `generator.calls` is never mistaken for the
reserved index; run the G2 Core **outside** the generator's lock; build
the graph node, with `_from_op` adopting the mask through the unchanged
`graph_resources` contract; and `_commit_call` **last**. So one
successful stochastic forward consumes exactly one call *with or without
gradients* (`detach()` is the native line's no-grad equivalent — it has
no no-grad context, because its graph is opt-in), and **every** ordinary
failure before the commit — invalid `p` or generator, a closed receiver,
an exhausted counter, a reservation conflict, a Core validation or
allocation failure, a Python wrapper failure, a backward-closure,
graph-node, or resource-attachment failure, a no-grad mask-cleanup
failure, or a delivery failure — releases the result and the mask,
cancels the reservation, and re-raises, leaving the same unconsumed index
so the next forward reproduces the committed vector the failed one would
have. Two private module-level seams make the last positions addressable
by test rather than only by argument, exactly as G1's
`_deliver_reservation` does: `_dropout_backward(input_tensor, mask)`
builds the backward closure, and `_deliver_dropout_result(result)` is the
deliberate no-op between a fully constructed result and the commit;
neither is exported or reachable from a public API. The mask is
**graph-owned** private state — the third member of the family beside
MaxPool2d's winners and cross-entropy's saved probabilities — released
exactly once with the graph history, retained under `retain_graph=True`,
kept alive across a failed retryable backward, freed by an abandoned
graph's `close()` (`__del__` the fallback), and closed immediately by a
no-grad forward. Backward reads **only** the upstream gradient and that
mask: it never rereads the input, never redraws, and never reserves,
commits, cancels, inspects, or mutates a generator, so the node records
**no** expected parameter version and a later input mutation, `reseed`,
`reset`, `load_state`, or `load_generator_state_dict` cannot change an
existing graph's gradient or raise it a stale-graph error (a *full*
checkpoint load still stales such a graph through some **other** node's
parameter rule — a parameter contract, never a Dropout effect).
Higher-order autograd is not supported, matching the rest of the native
line.
**G4 is complete**: `NativeDropout` in
`src/tensorforge/experimental/native_dropout.py`, its export from
`tensorforge.experimental`, and exactly one registry name —
`"NativeDropout"` appended to `NATIVE_MODULES`. **G4 changed no C++, no C
ABI symbol, no ctypes declaration, no `NativeTensorCore` method, no
autograd operation, and no checkpoint format.**
`NativeDropout(p=0.5, seed=None, generator=None)`: `p` goes through the
*same* `_normalize_dropout_probability` the Core and the operation use
(never a third rule) and is stored as a plain `float`; `seed` and
`generator` are **mutually exclusive**, so supplying both raises
`TypeError` rather than quietly ignoring one; without an explicit
generator the module **creates and owns** `NativeGenerator(seed)` (one OS
draw at `seed=None`), and with one it registers **that exact object**,
never a copy — the default gives every layer an independent stream and an
explicit generator gives several layers one interleaved stream. All
validation precedes generator creation and registration, so a rejected
construction draws no entropy, registers nothing, allocates nothing, and
leaves a supplied generator bit-identical. **Which construction path ran
is deliberately not recorded**: the public surface is exactly `p`,
`generator`, `training`, and the ordinary `NativeModule` methods, with
**no `owns_generator` attribute** (public or private) — "this module
created its generator" is true of one moment in the constructor and stops
being true as soon as that generator is shared with a second module, so
ownership is read from generator **identity** and the **registered
topology** (`a.generator is b.generator`, `named_generators()`) rather
than from a Boolean a caller could also overwrite. The generator is registered
under the canonical name `"generator"` (readable as `module.generator`)
as the **fourth** state category: in `generators()`,
`named_generators()`, and `generator_state_dict()`, deliberately absent
from `state_dict()` (still `{name: NativeTensor}`), identity-preserved
across `load_generator_state_dict()`, and never a parameter, buffer, or
child module. The module owns **no native storage**, and dropping it
never closes, resets, or mutates its generator. Forward validates the
input **first** (`TypeError` for a non-`NativeTensor`, `RuntimeError` for
a closed one — so evaluation is not a way to hand back an invalid
tensor), then dispatches: **training** is exactly
`input.dropout(self.p, generator=self.generator)`, so the operation owns
the whole call transaction and the module can add no failure hole to it;
**evaluation** returns the **input object itself**, consuming no call and
allocating nothing, so any number of eval forwards leaves **no gap in the
stream** and the next training forward takes the next index; and
**`p == 0`** is identity in both modes, deliberately delegated to the
operation (design §6.2) rather than duplicated as a second rule.
`train()`/`eval()` propagate normally, including through
`NativeSequential`, and never reseed or reset the generator. **G4 ships
the module and nothing above it**, and the gap is persistence: the
checkpoint format is still version 1 and has no generator section, so
saving a model containing a `NativeDropout` preserves its parameters and
buffers and **silently omits the random stream**, while a load leaves the
live generator exactly as it found it and **fabricates nothing** — so
**exact stochastic resume does not exist yet** (that is G5, which also
adds the rejection rule making such a load an error rather than a quiet
omission). That gap, plus the unrun closure matrix, is why `UNSUPPORTED`
still reads `("dropout", "float32", "cuda", "amp")` — `"dropout"` is the
one name deliberately in both an implemented inventory and that tuple,
because the registry reports what is *closed and validated* while the
inventories report what *exists*.
Data loaders, native integer tensors, further
dtypes/devices, CPU optimization, and CUDA experiments are
future work beyond Phase G.
Position the project as serious and systems-focused — never
"educational", "toy", or "mini" — while staying honest: not
production-ready, not a PyTorch replacement.

## Tech stack

- Python ≥ 3.13, NumPy, pytest — nothing else.
- Managed with `uv` (`uv run …` for everything).
- Never introduce PyTorch, TensorFlow, JAX, sklearn, pandas, or
  matplotlib. NumPy is the only numeric dependency.

## Layout

- `src/tensorforge/tensor.py` — Tensor + reverse-mode autograd. Ops are
  either primitives (eager NumPy forward + `_backward` closure holding
  the local derivative) or derived (compositions that get gradients for
  free). Gradients accumulate via `_accumulate_grad`, which also
  un-broadcasts.
- `src/tensorforge/nn/` — Parameter, Module, Linear, activations,
  Dropout, BatchNorm1d, LayerNorm, Conv2d, MaxPool2d, Flatten,
  Sequential, losses (`mse_loss`, `cross_entropy`,
  `binary_cross_entropy`), metrics (`accuracy`, `binary_accuracy`,
  `evaluate_classifier`, `evaluate_binary_classifier` — the evaluators
  measure with the model temporarily in eval mode and restore it).
  Modules have train/eval mode: `model.train()` / `model.eval()`
  recurse through children; Dropout and BatchNorm1d change behavior.
  Modules can declare non-trainable buffers via `self._buffers =
  ("attr", ...)` (e.g. BatchNorm running stats); `state_dict()` /
  `load_state_dict()` cover parameters *and* buffers.
- `src/tensorforge/optim/` — SGD, Adam. Plain classes: `step()` skips
  `None` grads and frozen params, `zero_grad()` sets grads to `None`.
  Also `StepLR` (multiplies `optimizer.lr` by gamma every step_size
  epochs) and `clip_grad_norm` / `clip_grad_value` (clip gradients in
  place before `optimizer.step()`).
- `src/tensorforge/data.py` — `batches` mini-batch iterator.
- `examples/` — runnable scripts, each with `train(...)` returning
  stats and a `main()` that prints, guarded by `__main__`.
- `tests/` — pytest suite; every feature has tests.
- `docs/` — project summary, architecture, autograd, training,
  examples, roadmap, release history, and the native-line design
  contracts (`native_cnn_design.md` for Phase D,
  `native_classification_design.md` for Phase E,
  `native_normalization_design.md` for Phase F — F0–F9 shipped, phase
  complete —, and `native_rng_dropout_design.md` for Phase G — G0 (the
  design lock), G1 (`NativeGenerator` and module generator-state
  ownership), and G2 (the stateless `dropout_forward` **Core** kernel and
  its C ABI) shipped, G3–G10 not started). When a milestone changes the
  public API or the examples, update the matching docs file (and
  README links) in the same milestone.
- `.github/workflows/tests.yml` — minimal CI: install uv, build the
  experimental C++ backend, hard-failing kernel smoke check, then
  pytest.
- `cpp/` + `src/tensorforge/backends/` — the experimental C++ backend
  (post-v3.0 line; `cpp/src/classification.cpp` holds the Phase-E
  classification kernels and `cpp/src/random.cpp` the Phase-G stateless
  SplitMix64 derivation and Dropout-forward kernel). Plain C-ABI kernels
  loaded via ctypes; built with
  `uv run python cpp/build.py` (`uv sync --group cpp` first if no
  compiler). Never imported by the main framework; importing the
  wrapper is always safe (lazy load) — check `cpp.is_available()` /
  `cpp.backend_info()`; kernels raise ImportError at call time when
  unbuilt, and the backend tests skip. `benchmarks/cpp_backend.py`
  compares kernels against NumPy honestly (no performance assertions
  anywhere), while `benchmarks/benchmark_native_cnn.py`,
  `benchmarks/benchmark_native_classification.py`, and
  `benchmarks/benchmark_native_normalization.py` characterize the Phase-D
  CNN, Phase-E classification, and Phase-F normalization stacks the same
  way (correctness gated before timing, honest reference labels, no
  result file, no speed asserted). `scripts/smoke_cpp_backend.py` is the
  hard-failing smoke check CI runs after building. Dependency-free C++
  CTests live in `cpp/tests/` and build only with `-DTF_BUILD_TESTS=ON`;
  sanitizer validation uses Clang on Linux
  (`-DTF_SANITIZE=address,undefined`), which MSVC does not support.

## Commands

- Run tests: `uv run pytest`
- Run examples:
  - `uv run python examples/train_linear_regression.py`
  - `uv run python examples/train_xor.py`
  - `uv run python examples/train_multiclass.py`
  - `uv run python examples/train_binary_classification.py`
  - `uv run python examples/train_mlp_with_dropout.py`
  - `uv run python examples/train_tiny_cnn.py`

## Style rules

- Keep code simple and readable — clarity beats cleverness.
- Match the existing style: NumPy-only internals, small modules, one
  concept per file.
- Comments explain math/autograd reasoning, not obvious Python.
- Losses and metrics stay simple: losses are Tensor expressions or
  fused ops with custom backward; metrics are plain NumPy returning
  Python floats, outside autograd.
- Examples use fixed seeds so output is reproducible, and follow the
  `train()` + `main()` pattern so tests can import `train`.
- Tests use `np.allclose` with sensible tolerances (e.g. `atol=1e-6`);
  training tests assert learning without fragile exact-loss values.

## Workflow rules for Claude Code

- Inspect existing code before editing; find where a concept lives and
  follow its pattern.
- Keep changes scoped to the requested milestone. No unrelated
  features, no drive-by refactors, no framework rewrites.
- If a requested feature already exists, verify it against the spec and
  add tests/documentation instead of reimplementing it.
- Preserve all previous tests. Never loosen a test just to pass.
- Run `uv run pytest` (and any requested manual checks) before
  reporting success; report the actual observed output.
- Do not use git: no commits, no pushes, no `git` commands. The user
  handles version control.
- Final responses report: files changed, what was implemented, tests
  added, the exact pytest result, manual check outputs, and any notes
  or limitations.

## Current notes

- This machine has a permissions quirk: directories created by one
  process often cannot be deleted by a later one. Consequences already
  handled: pytest's cache is redirected to `.cache/pytest` (pyproject)
  and `conftest.py` gives each test session a fresh unique basetemp so
  tmp_path never needs to wipe an old directory. Don't try to delete
  `.pytest_cache/`, `.cache/pytest-tmp/`, or `%TEMP%/pytest-of-*`.
- Two example-test import styles coexist: `tests/test_examples.py`
  inserts `examples/` into `sys.path`; newer tests import
  `examples.<name>` as a namespace package from the repo root.
- Root package exports: `Tensor`, `Parameter`, `Dropout`,
  `BatchNorm1d`, `LayerNorm`, `Conv2d`, `MaxPool2d`, `Flatten`,
  `cross_entropy`,
  `binary_cross_entropy`, `accuracy`, `binary_accuracy`,
  `evaluate_classifier`, `evaluate_binary_classifier`, `SGD`, `Adam`,
  `StepLR`, `clip_grad_norm`, `clip_grad_value`, `batches`,
  `train_test_split`, `save_parameters`, `load_parameters`,
  `save_checkpoint`, `load_checkpoint`, `count_parameters`,
  `model_summary` (locked in by `tests/test_public_api.py`).
  Checkpoints = weights + optimizer state + optional scheduler state
  + optional RNG state (`rng_state=True` / `restore_rng_state=True`,
  covers unseeded Dropout) + JSON metadata; parameters = weights only.
