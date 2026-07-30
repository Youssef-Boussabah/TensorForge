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
has completed Phases A–G: Phase A — CPU runtime, Phase B — native
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
Stateful Buffers — is *complete* (F0–F9):**
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
**Phase G — Native RNG and Dropout — is the latest *completed* phase
and is *complete*: milestones G0 (the architecture contract in
`docs/native_rng_dropout_design.md`), G1 (`NativeGenerator` and module
generator-state ownership), G2 (the deterministic stateless
Dropout-forward **Core**), G3 (the differentiable
`NativeTensor.dropout(p, *, generator)`, with an **explicit** required
keyword-only `NativeGenerator`), and G4 (the `NativeDropout` module and
its public export), and G5 (native checkpoint **format version 2** with
persisted generator state and its alias topology), and G6 (RNG, graph,
ownership, and checkpoint hardening — **no new capability**), and G7 (the
deterministic stochastic training example and its exact checkpoint resume
— also **no new capability**), and G8 (the honest benchmark characterization `benchmarks/benchmark_native_dropout.py` — measurement only, no capability), and G9 (the cross-cutting integration suite `tests/test_native_phase_g.py` — integration evidence only, no capability, no runtime file changed), and G10 (the phase closure — the Windows Release/Debug and Clang ASan/UBSan/LeakSanitizer validation matrix, documentation reconciliation, durable closure guardrails in `tests/test_native_phase_g_closure.py`, and the single registry line that removed `"dropout"` from `UNSUPPORTED`, which now reads exactly `("float32", "cuda", "amp")`; validation, documentation, and that one line — **no C++, CTest, ABI, ctypes, Core method, operation, module, export, schema field, checkpoint version, example, or benchmark changed**) are all complete.** G0 locked Python-managed generator state (an explicit
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
"amp")`, and G0 left the checkpoint format at version 1. G0 also locks a
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
capability boundary: `"dropout"` stayed in `UNSUPPORTED` through G9 and
left it at **G10**, after the full closure matrix passed, leaving
`("float32", "cuda", "amp")`. The format version became 2 at G5 — none
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
`state_dict()` is unchanged and still tensor-only; G1 left the
checkpoint format at version 1 and serialized no generator
state; and no *numerical* capability-registry value moved (`UNSUPPORTED`
still reads `("dropout", "float32", "cuda", "amp")`, dtypes/devices
unchanged). The one registry change G1 does make is reporting-only:
`STATE_SUPPORT` gained `"generator_state"` between `load_state_dict` and
`save_native_checkpoint`, a capability name (like `"persistent_buffers"`)
covering the generator registration and in-memory state surface — it
does **not** mean generator state is checkpointed, which arrived at G5
under its own separate name, `"checkpoint_generator_state"`.
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
`NativeSequential`, and never reseed or reset the generator. **G4 shipped
the module and nothing above it**, and the gap was persistence: at G4 the
checkpoint format was still version 1 with no generator section, so
saving a model containing a `NativeDropout` preserved its parameters and
buffers and **silently omitted the random stream**, while a load left the
live generator exactly as it found it and **fabricated nothing** — so
**exact stochastic resume did not exist yet**.
**G5 is complete**: native checkpoint **format version 2** and exact
generator restoration —
`src/tensorforge/experimental/native_checkpoint.py` (the version, the
`"generators"` section, its validators, and the four-phase load), the new
private `_native_checkpoint_transaction.py` (the one rollback guard
spanning the model, optimizer, and generator commits), the private
locked-read helper `snapshot_generator_states` in `native_generator.py`,
the private undeduplicated path walk
`NativeModule._named_generator_paths`, and one reporting-only registry
name — `"checkpoint_generator_state"` appended to `STATE_SUPPORT`.
**G5 changed no C++, no C ABI symbol, no ctypes declaration, no
`NativeTensorCore` method, no autograd operation, no module, no export,
and added no public entry point**: persistence rides the existing
`save_native_checkpoint` / `load_native_checkpoint` pair. The format
**name** is unchanged forever; `_FORMAT_VERSION` is **2** and every new
save writes 2, whether or not the model has generators (a generator-free
model writes `"generators": null`, so absence is stated rather than
inferred). The section is exactly three fields: `keys` (the ordered
canonical names from the identity-deduplicated `named_generators()`
walk), `entries` (one `{algorithm, algorithm_version, seed, calls}`
object per canonical name, mapping exactly `keys` in order, with `seed`
and `calls` as **canonical decimal strings** — `^(0|[1-9][0-9]*)$`, ≤ 20
digits, in `[0, 2**64 - 1]` — because a `uint64` above `2**53` is not
representable in the IEEE double most JSON readers use), and `aliases`
(the complete **registered path → canonical name** map in full traversal
order, every canonical name included and self-mapped). Generator state
adds **no array** to the NPZ payload. A shared generator's state is
written **once** while its *topology* is written in full, so two paths
draw from one stream in the archive exactly when their aliases name the
same canonical entry — sharing is **identity**, never state equality, so
two generators with the same seed and counter stay two entries. Canonical
names and both orders are functions of the model alone, so saving the
same model twice is byte-identical and no caller-supplied mapping order
can alter the archive. **Loading** compares the archive against a real
`named_generators()` traversal of the live model, strictly in both
directions: missing or unexpected canonical keys or registered paths, an
alias targeting an absent entry, a canonical name absent from `aliases`
or not self-mapped, a multi-step alias relation, a repeated JSON object
key (rejected via an `object_pairs_hook`, since Python's `json` silently
keeps the last), saved-shared versus live-independent and the reverse, a
canonical name changed by a reordered registration, an algorithm or
version mismatch, and a malformed or out-of-range seed/counter string all
raise **in prevalidation, with the model, buffers, optimizer, and
generators completely untouched**. Generators are restored **in place**
through `load_state`, so identity and every sharing relationship survive
and the archive never constructs a `NativeGenerator`; loading generator
state moves no parameter version and stales no graph. A save **or** a
load is refused, changing nothing and leaving an existing destination
byte-intact, while any target generator has a call reservation in flight
— published *or* holding a construction claim — and every generator's
state is read in **one** locked snapshot (the same global `id()` lock
order `replace_generator_states` uses), so the states an archive carries
were true together. **Version-1 compatibility** is exactly as locked: a
v1 archive still loads into a model with **no** registered generators,
and one loaded into a model that **has** them fails naming them,
fabricating no seed and no counter — not zero, not fresh entropy, not the
current value; a v2 archive with a non-null generator section loaded into
a generator-free model fails as an unexpected-generator error; any other
version fails; the loader accepts `{1, 2}` and dispatches, with no
"latest wins", no upgrade in place, and no silent rewrite. A load is
**one transaction over the whole archive**: prevalidation touches
nothing, staging materializes every staged value *and* an independent
rollback snapshot of every live target the commit will overwrite, and the
commit runs model → optimizer → generators through the components' own
loaders inside **one** rollback guard — so any exception in it, a
deliverable `KeyboardInterrupt` included, restores all four state
families, preserves every object identity, moves no parameter version,
leaves graph-owned Dropout masks from earlier graphs untouched, and
returns native live storage to baseline. Because every allocation the
rollback needs happens in staging, the rollback is plain attribute
assignment and cannot raise; only external process or interpreter death
is outside the guarantee. **Serializability is the other half (§10.8)**:
atomic-under-failure is not atomic-with-respect-to-other-threads, and two
concurrent loads that each succeed could otherwise leave the model from
one archive beside the optimizer or generators from the other. So every
participating in-memory state replacement now runs under **one** private
process-wide `threading.RLock` — `_native_state_lock.state_transaction()`,
never exported — in the **universal state-replacement lock order**: the
guard first, then every unique target generator lock in the existing
global `id()` order, and never the reverse. The participants are the
checkpoint load commit, `replace_native_state`/`NativeModule.load_state_dict`,
`replace_generator_states`/`load_generator_state_dict`,
`NativeSGD.load_state_dict`, `NativeAdam.load_state_dict`, and the
checkpoint **save snapshot** (held until the complete immutable payload
and manifest exist, released before the NPZ write). So two concurrent
loads leave one archive's state followed by the other's, never a mixture,
and a save describes one coherent serial point rather than model state
from before a replacement beside optimizer or generator state from after
it. It is an `RLock`
because the checkpoint transaction holds it and then calls the
components' own loaders, which take it again — the alternative would be a
second, lock-free commit path per component. Both locks are taken
together by `native_generator.locked_generators`, which also owns the
under-lock reservation recheck, so the order holds by construction rather
than by each caller remembering it; even `_ordered_targets` runs inside
the guard. Generator **reservations** deliberately do *not* take the
guard — only their own generator's lock — which is what stops the two
systems inverting: a racing reservation either completes before a
transaction takes that lock or begins after it releases it, so no state
is ever replaced underneath a live token. The rollback snapshots are
captured **inside** the guard, at the real commit boundary, because one
taken earlier could describe a model another transaction has since
replaced. What is **not** claimed: ordinary training mutation — an
optimizer `step()`, a `copy_value_`, a backward — does not take the
guard, so thread-safe concurrent training snapshots are not offered; the
claim is exactly that participating operations serialize with respect to
each other. One consequence is recorded rather than glossed: `NativeAdam.load_state_dict` releases the moment buffers it
replaces, so a rolled-back load restores the optimizer's moments **by
value into its current buffer objects** — private optimizer internals
with no public identity contract — while every publicly identified object
is the same object afterwards on both paths. The transaction adds **no
new lock order**: the model and optimizer commits take no locks, and the
only locks a load ever holds are generator locks, taken inside
`replace_generator_states` in its existing global order.
**G5 proved exact generator restoration — state, identity, topology, and
the next Dropout mask against the G2 Core at the restored call index —
but not the end-to-end §11 story**: at G5 the interrupted stochastic
*training* run reproduced into a fresh model/optimizer/generator set
remained ahead as the **G7** resume, which G7 then demonstrated.
Reproducibility is
exact **for the state actually captured**; Python's `random`, NumPy's
global RNG, data-loader position, and scheduler state are not captured
and full-program determinism is not claimed. That remaining gap, plus the
then-unrun closure matrix, is why `UNSUPPORTED`
read `("dropout", "float32", "cuda", "amp")` at G5 — `"dropout"` was the
one name deliberately in both an implemented inventory and that tuple,
because the registry reports what is *closed and validated* while the
inventories report what *exists*. It left that tuple at the **G10**
closure, and `UNSUPPORTED` now reads `("float32", "cuda", "amp")`.
**G6 is complete**: the hardening milestone, which added **no capability,
operation, module, export, checkpoint field, or checkpoint version** and
moved no registry value. `tests/test_native_phase_g_hardening.py` executes
the design's §13 ownership and §14 failure matrices as adversarial tests:
the reservation transition matrix, with each rejected transition asserting
five invariants at once (no counter movement, no active-reservation change,
no construction-claim change, no serial reuse, no native-storage movement)
and the four reservation-creation failure positions distinguished by
whether a serial was consumed; the exact `uint64` boundary as §4.6's table,
row by row, with the final index retryable until committed and repeated
exhaustion failures freezing every field; forced concurrent interleavings
under barriers and events with **bounded joins and no sleeps** (no
duplicate call index, unrelated generators independent, no torn state read,
a reservation racing a state replacement provably preceding or following it
in both orders, a construction claim refusing both a save and a load, a
transaction started from inside token construction refused rather than
deadlocked, and nested component loaders not self-deadlocking); the
deterministic Core's **structural** key properties beside its committed
vectors — the stream key injective in the call index for one seed, the
element derivation injective within one call, and the cross-seed collision
a 128-into-64-bit key makes unavoidable (`seed=2**63, call=2**63` equals
`seed=0, call=0`) pinned as a **characterized consequence** rather than a
defect, since sharing is identity and the contract never claimed the
stronger property; the probability extremes and logical-layout
independence through real transposed and narrowed views; every pre-commit
position of §5's call transaction times `RuntimeError`, `MemoryError`,
`KeyboardInterrupt`, and a non-`Exception` `BaseException`, each proving
the retry reproduces the exact mask the failure would have produced, and
every post-commit position proving the index spent exactly once with the
original exception primary; all **four** graph-owned saved-resource
families (a Dropout mask, MaxPool2d winners, BatchNorm eval snapshots, and
cross-entropy probabilities) coexisting in one graph and releasing exactly
once, across branched, chained, shared-generator, independent-generator,
retained, failed-retryable, and abandoned graphs; a **76-case** checkpoint
corruption matrix, every case failing before any live change with all four
state families bit-identical; whole-transaction rollback injected at every
commit position times the same four exception classes, with object
identities, parameter versions, unrelated active reservations, and
pre-load graph masks proved untouched; save-seam destination atomicity at
all seven positions; and repeated success-and-failure lifecycle loops
returning native live storage exactly to a measured baseline. **One
runtime defect was found and fixed** with the narrowest possible change:
`native_tensor._chain_cleanup_failure` closed a **cycle** in the
`__context__` chain when a cleanup step failed — a cleanup exception
raised while the operation's failure is being handled implicitly points
back at it, so appending it without cutting that link made every ordinary
"follow `__context__` to the end" reader spin forever (the helper itself
included). The fix cuts that back-reference and is inert when the cleanup
failure is already in the chain; the original exception stays primary and
the cleanup failure stays reachable, and a dedicated regression guard
fails without the fix. No C++, C ABI symbol, ctypes declaration, Core
method, autograd operation, module, export, schema field, benchmark, or
example changed.
**G7 is complete**: the end-to-end exact stochastic resume, and again **no
new capability**. `examples/native_dropout_training.py` trains
`NativeDropoutClassifier` — `NativeLinear(4, 8, seed=0)` →
`NativeBatchNorm1d(8)` → `NativeReLU` → `NativeDropout(p=0.5,
seed=20240707)` → `NativeLayerNorm(8)` → `NativeLinear(8, 3, seed=1)` —
over **raw logits** with `NativeCrossEntropyLoss` and
`NativeAdam(lr=0.05)`. It is the smallest model carrying **all four**
TensorForge-owned state families at once (parameters, persistent BatchNorm
running buffers, a registered `NativeGenerator`, and NativeAdam moments
with per-parameter step counters), so an incomplete restore diverges
immediately. The data is twelve four-feature samples over three classes
computed from an **explicit arithmetic formula** — every value a quarter
or an eighth, exact in float64 — in three fixed batches of four, on a
schedule that is a **pure function of the training step** (`step % 3`);
nothing is shuffled, generated randomly, augmented, loaded, or downloaded,
and neither NumPy's global RNG nor Python's `random` is touched. **Two
uninterrupted runs are bit-identical**, and an interrupted run
checkpointed after 7 *completed* steps (deliberately mid-cycle in the
schedule), whose model, optimizer, and generator are **released before the
resume begins** so the archive is the only continuation boundary, reloads
into a completely fresh set built with a *different* Dropout seed and
reproduces the uninterrupted run by **exact equality**: the whole loss
sequence, every parameter, both running statistics, every optimizer moment
and step counter, the generator's algorithm/version/seed/calls, the final
training logits, and the final evaluation output. Two **negative controls**
make that load-bearing: restoring all four families but restarting the
batch schedule at 0 **diverges**, and restoring everything but re-seeding
the generator **diverges**. Evaluation is proved **state-neutral** —
repeated eval passes leave `calls` bit-identical, produce identical
outputs, restore the caller's mode, and leave a probed run's loss sequence
exactly equal to an unprobed one's — and a separate **throwaway** reload
(leaving the resumed run untouched) matches the restored `NativeDropout`'s
next output against `NativeTensorCore.dropout_forward` at the exact
restored `(seed, call_index)`, advancing `calls` by exactly one; the
module's private mask is never exposed. **External loop progress is
carried explicitly**, as validated JSON metadata (`{"training_step": k,
"next_batch_index": k % 3, "lr": ...}`), because checkpoint v2 captures
TensorForge-owned state and **not** data-loader position, batch order,
shuffle state, epoch counters, scheduler state, Python's `random`, or
NumPy's global RNG — and `validated_progress` **raises** on a missing
field, a `bool` where an `int` belongs, an out-of-range step, or a
`next_batch_index` disagreeing with the schedule, rather than silently
restarting from step 0. Reproducibility is exact **for the state actually
captured**; full-program determinism is not claimed. The milestone is one
example, one test module, and documentation: **no** C++, C ABI symbol,
ctypes declaration, Core method, autograd operation, module, export,
schema field, checkpoint version, benchmark, or registry value changed,
and the example defines **no public training API** — none of its helpers
is exported.
**Phase H — Native CPU Performance and Runtime Efficiency — is the
latest phase and is the current one; it has *begun*, with milestones
**H0, H1, H2, H3, H4, H5, H6, and H7** complete, and Phase G remains the latest
*completed* phase.** H0 is an
architecture, profiling, and baseline milestone: it shipped
`docs/native_cpu_performance_design.md` (the contract), the unified
measurement harness `benchmarks/benchmark_native_cpu_performance.py`,
its behavioral contract tests
`tests/test_native_cpu_performance_benchmark.py`, and documentation
reconciliation — and **nothing else. No performance optimization has
shipped.** H0 changed no C++, no C ABI symbol, no ctypes declaration, no
`NativeTensorCore` method, no autograd operation, no module, no loss, no
metric, no optimizer, no export, no capability registry, no dtype, no
device, and no checkpoint format: `UNSUPPORTED` still reads
`("float32", "cuda", "amp")`, `SUPPORTED_DTYPES` still reads
`("float64",)`, `SUPPORTED_DEVICES` still reads `("cpu",)`, and the
native checkpoint format is still `tensorforge.native_checkpoint`
version **2** with versions **(1, 2)** supported. The harness runs 26
cases (24 at H0, plus the two H3 added to decompose the per-call
cost) across twelve workload families (dispatch overhead, elementwise,
reduction, matmul, materialization, linear, convolution, normalization,
stochastic, optimizer, training step, and in-memory state operations),
separating up to nine declared implementation layers (`numpy`,
`stable_tensorforge`, `raw_kernel`, `raw_kernel_tiled`, `tensor_core`,
`native_tensor`, `native_tensor_graph`, `backward`, `optimizer_step`,
`training_step`); every case's correctness gate runs **before** the
timing helper is ever reached; a case with no honest equivalent is
labelled `native_only` and publishes **no ratio at all**; `--smoke`,
`--json`, `--case`, `--workload`, and a focused `--profile CASE` mode
exist; setup, cleanup, and any advanced state are handled outside the
timer (a fresh model and optimizer per training-step repetition, an
explicitly `reset()` generator per Dropout repetition); checkpoint file
I/O is deliberately excluded and the in-memory state surface is its own
category; and **no result file of any kind is written**. The design
separates its evidence into *directly measured*, *strongly
source-evidenced but not fully measured*, and *unconfirmed hypotheses*,
records the minimal instrumentation a later milestone would need
wherever H0's observability could not settle a question, and makes the
proposed **H1–H11 ladder explicitly conditional** — a milestone whose
premise the measurement does not confirm is narrowed, reordered, or
dropped. A memory pool, scratch allocation, SIMD, threading, and BLAS are
all currently **rejected on evidence**, with the criteria that would
reopen each recorded rather than an answer invented. Every measured
number is a local characterization of one machine, reported with its
spread, and asserted by **no** test; there is no CI timing threshold
anywhere in this repository.
**Milestone H1 — the Explicit Output-Allocation Contract — has since
shipped, and is the first Phase-H change to production code.** **Milestone H1 — the output-allocation contract — has now shipped.** It removed the redundant zero-fill from output storage that a kernel provably overwrites in full, behind one new C ABI symbol (`tf_storage_create_uninitialized`) that matches the zero-initializing default in size validation, allocation-failure handling, error state, ownership, destruction, and live-storage accounting, and differs only in the buffer's initial contents. The zero-initializing path remains the default; there is **no** global allocator policy, environment variable, heuristic, memory pool, scratch arena, or public empty-tensor API, and every enabled call site opts in explicitly against a per-kernel audit table. `sum`/`mean` and `narrow_backward` are explicitly **rejected** and keep a zeroed destination: the first accumulates into its output, the second writes only the narrowed region and the untouched zeros *are* the gradient. Completeness is proved by deterministic **poison** tests that are injected **exclusively by test infrastructure, around the allocator**: the suite wraps the private uninitialized allocation helper, lets the real constructor allocate, fills the returned storage with a quiet NaN or a large finite pattern through the ordinary fill primitive, and hands that same storage to the real operation — so the pattern is in place after the real allocation and before the real kernel runs. **No poison-control mechanism exists in the production runtime**: no exported hook, no thread-local flag, no environment variable, no global mode. ASan and UBSan stay separate from the initialization proof — they do not detect uninitialized-value reads — and MemorySanitizer is not available here, so neither is claimed; negative controls prove the detector can actually fail. H1 is bit-identical: every enabled operation and a full training run are compared element-wise against the zero-initializing allocator. No capability, dtype, device, registry value, checkpoint field, or checkpoint version changed, and `tf_storage_create_uninitialized` is the **only** export it added, taking the library from the pre-H1 baseline of 51 exported `tf_*` symbols to **52**. An earlier draft of H1
also exported a test-only `tf_test_set_uninitialized_poison` (with
`cpp._set_uninitialized_poison` / `cpp._uninitialized_poison` in Python);
that mechanism was **removed in full**, because a symbol compiled into and
exported from the normal runtime is part of the runtime however carefully
it is disarmed, and one that can alter production allocation contents does
not belong there. The proof was rebuilt around the allocator with no
coverage lost. The measured result is reported honestly rather than as a headline: isolated, the zero-fill is enormous and scales with the buffer (about 52x at 2 MB, 119x at 8 MB, 552x at 32 MB, and *negative* below roughly 16,000 elements, where it sits inside the noise). End to end it is much smaller and often inconclusive — clearly real for large memory-bound elementwise work (about 1.5-1.8x on an 8 MB output), small and variable for normalization and Adam, and with no measurable effect on Conv2d, the MLP step, or matmul, whose arithmetic dwarfs its allocation. Those inconclusive and negative rows are published as such.
**Milestone H2 — native matmul memory access — has since shipped**, the
first Phase-H milestone to change how a numerical kernel executes. It
swapped the production matmul's loop order from `i`-`j`-`k` to
`i`-`k`-`j` over four destination rows at a time, so the innermost loop
walks a *row* of the right operand and a row of the output sequentially
instead of walking a column. **Cache blocking, which the milestone title
anticipated, was measured against 22 blocked variants and rejected** — an
unblocked full-width row sweep was faster at every non-trivial size — so
H2 shipped the simpler superior design and recorded the negative blocking
result. The pre-H2 triple loop is **retained verbatim as the shipped
generic reference path**, still reachable through ordinary production
dispatch, and the choice between the two is made inside the kernel from
the stride metadata it already receives: a right operand whose column
stride is 1, with a non-empty inner dimension and at least 8 result
columns, takes the row sweep; a transposed right operand, a narrow
result, or an empty inner dimension takes the generic path — which is the
loop order that case already suits, so the fallback is a design choice
rather than a gap. Dispatch is metadata-driven, deterministic, total,
side-effect free, and independent of pointer values, alignment, timing,
environment variables, and CPU-feature probes; a failed precondition is
never an error. **H2 added no exported C ABI symbol** — the library still
exports exactly 52 `tf_*` symbols — and there is no kernel selector,
block-size setter, benchmark hook, dispatch tracer, or public dispatch
control of any kind; the two kernels and the predicate are
hidden-visibility C++ that the native test reaches only by compiling the
source in. The numerical agreement between the two paths is stated in **four
parts** rather than as a blanket claim, because a blanket claim would be
an overclaim. (1) **Accumulation order is preserved exactly** — same
starting zero, same products, same ascending `k`. (2) **Every non-NaN
result is bit-identical**, asserted as raw IEEE-754 bit patterns rather
than tolerances across shapes, layouts, signed zeros, infinities,
denormals, the largest finite magnitudes, both gradients, `NativeLinear`,
both optimizers, deterministic training, and exact checkpoint resume —
which covers every committed loss trajectory and every resume proof in
the project, since all of them run on finite data. (3) **NaN-class
equivalence holds**: NaNs appear in exactly the same positions on both
paths and are always quiet, and neither path produces a signaling NaN.
(4) **NaN payload bits are deliberately outside TensorForge's numerical
contract** and may differ between the paths. Ten source-level
formulations were measured while trying to close (4) — compound versus
explicit assignment, named locals, `__restrict`, disabled inner-loop
vectorization, and two stack-accumulator tile shapes — and all ten
`i`-`k`-`j` spellings behaved identically; the only structure that
reproduces the reference's payloads is the `i`-`j`-`k` order H2 exists to
replace, so payload parity is unavailable short of abandoning the
optimization. Measured: MSVC Release differs on 162 of 208 results in a
NaN-saturated matrix, MSVC Debug and Clang on none. H1's uninitialized-output
contract still holds on both paths, for a different reason on each — the
generic path never reads the destination, and the row sweep's `k == 0`
pass assigns every element of every row before anything accumulates into
it — proved by poison tests over both paths with both patterns plus a
negative control. The measured result is reported honestly: roughly
4.1-4.7x at 384 cubed, 4.2-4.5x at 128 cubed, about 4-6.8x on
`NativeLinear` forward, 1.7-2.5x on its backward (only one of its two
matmuls qualifies, by design), 2.0-2.4x on a 128x256 MLP Adam step, and
**no measurable effect below roughly 32 cubed or on a small MLP step**,
where a fixed ~10 microsecond per-call Python cost dominates and control
cases whose compiled code did not change at all vary by 0.50-1.44x. No
capability, dtype, device, registry value, checkpoint field, or
checkpoint version moved.

**Milestone H3 — native metadata and dispatch efficiency — has since
shipped**, and unlike H1 and H2 it is **Python-only**: no C++, no C ABI
symbol, no ctypes declaration, and no kernel changed, so the library
still exports exactly **52** `tf_*` symbols. The measured cause of B3's
fixed per-call cost was redundant *re-validation*: one `shape_info` call
ran `_as_int_tuple` **four** times over a tuple that was fully validated
after the first pass and computed the row-major strides **twice**, while
`NativeTensorCore.zeros` validated the caller's shape a second complete
time — instrumented at **815** `_as_int_tuple` calls per MLP training
step and 604 per `NativeAdam` step, using test-local monkeypatching
because **no production counter exists or may exist**. H3 shipped three
things. (1) **One normalization boundary**, the private
`_normalized_layout`, performing exactly the checks `shape_info` always
performed in the same order with the same messages and normalizing the
shape once, with the strides, element count, and contiguity derived by
private `_checked` primitives that validate nothing because nothing is
left to validate; each public helper (`row_major_strides`, `numel`,
`reduce_shape`, `broadcast_shapes`) is now its own validation plus the
matching primitive, so the two cannot disagree. (2) **Two view
constructors, one binding**: the public `NativeTensorView(...)`
normalizes, the private `_from_validated` skips **only** that
normalization, and both funnel through a shared `_bind` that still runs
the storage open check and the **full reachable-offset bounds check**
(not skipped — bounds depend on the storage, not the metadata); the
element count and contiguity flag are **derived inside** the private
constructor rather than passed to it, so an inconsistent pair is
unrepresentable, which is why this is a separate constructor and not a
misusable `validated=True` flag. (3) **Lazy, read-only, per-view `int64`
layout arrays** for the strided C ABI — memoization of a pure function of
immutable state, where staleness is impossible *by construction*: a
view's layout is assigned exactly once in `_bind` and every
layout-changing operation returns a *new* view, so no invalidation is
required and **none exists**. `narrow` gained an explicit `int()`
normalization of `dim`/`start`/`length`, since the private constructor no
longer re-normalizes. **No validation was removed** — every rejection
keeps its exception type, message, and shape-then-strides-then-offset
ordering — nothing global was introduced (no shape cache, stride
interning, weak-reference machinery, or thread-local state), and **no
public API of any kind was added**: no cache control, statistic, reset,
profiling counter, dispatch selector, or environment variable. Measured:
`shape_info` 2.6-4.5x, view construction 3.2x, `_as_int_tuple` per MLP
step **815 -> 149**, a one-element allocation 2.1x, `reshape` 3.1x, a
small `add` 1.56x, `NativeAdam` on a small MLP 1.42x, an **MLP training
step 1.43x**, a **CNN step 1.29x**, a **normalized step 1.51x**. Reported
just as honestly: **large kernel-bound work shows no measurable change in
either direction** (384/512/128 cubed matmul, 256 squared elementwise,
128 squared reduction all inside their own spread), so H2's result is
intact; the layout-array cache is the weakest of the three changes and
was kept on measured merit, saving 0.6-1.5 microseconds per *strided*
small operation and nothing on large ones or on a contiguous training
step, with even a cold-cache measurement no slower than pre-H3. One
methodology finding is published rather than buried: at the harness's
default 11 repetitions a case appeared to regress 35% and at 201
repetitions measured 1.19x *faster*, so no default-repetition figure is
quoted as H3 evidence. Cold object footprint is byte-identical; a view
that takes a strided path costs +328 bytes, and only **5 of 134** views
in an MLP step ever populate it. The harness gained two `native_only`
cases, `metadata_preparation` and `ctypes_boundary`, decomposing B3's
single figure into its Python and boundary halves. No capability, dtype,
device, registry value, checkpoint field, or checkpoint version moved.

**Milestone H4 - native optimizer step efficiency - has since shipped**,
also **Python-only** (no C++, no C ABI symbol, no ctypes declaration, no
kernel; still exactly **52** exported `tf_*` symbols) and the first
Phase-H milestone whose subject is a *training-stack* component rather
than the tensor runtime. B4's counts were re-instrumented on the current
post-H3 code rather than trusted from H0, and H0's figure was confirmed
exactly: **27 native storage allocations per parameter per
`NativeAdam.step()`** - 8 scalar coefficients, 13 binary compute outputs,
4 unary compute outputs, and 2 for the commit copy - of which **ten are
one-element**: the eight broadcast scalars (`beta1`, `1 - beta1`,
`beta2`, `1 - beta2`, both bias-correction terms, `eps`, and `lr`; the
design said *six*, and `eps` and `lr` were the two it missed) plus the
two `reciprocal` outputs taken on one-element tensors. `NativeSGD`
allocates five per parameter, and **8 of Adam's 13 binary operations**
take the broadcasting path rather than the contiguous fast path. H4
shipped three changes. (1) **The step's scalar coefficients are built
once per step, not once per parameter**: a private per-step
`_StepConstants` holder builds each on first use, keyed by
`(dtype, device)` so it never assumes one dtype exists, and hands the
same read-only core to every later parameter; the two bias-correction
terms are cached per step *counter*, so steady-state training builds one
pair while a parameter that skipped earlier steps legitimately gets its
own. The holder allocates nothing until the first entry asks for a
coefficient - so a step with no active parameter allocates nothing at
all - is released before the commit begins, and is **never stored on the
optimizer**, so no scalar survives a step, enters `state_dict()`, reaches
a checkpoint, or has to be released by `close()`. `NativeSGD` does the
same for its single `lr` scalar, the only change its evidence supported.
(2) **The bias-correction reciprocal is evaluated in Python**, removing
one allocation and one kernel call per coefficient per parameter - an
**exact substitution, not a reassociation**, because the kernel literally
is `double op_reciprocal(double x) { return 1.0 / x; }`, a Python
`float` and a C++ `double` are the same IEEE-754 binary64 value, and
IEEE-754 division is correctly rounded, so exactly one result is
possible; proved over **20,000+ values** spanning the full exponent
range, +/-0, +/-inf, the smallest subnormal, the largest finite
magnitude, and every `1 - beta ** t` the optimizer actually forms,
compared on **raw `uint64` bit patterns** with zero mismatches.
(3) **Temporaries are released at their last use** instead of all
together at the end of the staged expression. Everything is
**bit-identical** to the pre-H4 composition, with no carve-out of the
kind H2 needed (no accumulation order, operand position, or kernel
changed, so NaN payloads match too); the pre-H4 composition is
**retained in the test suite** as a literal transcription executed
natively, and every equality is against that - 60
shape/step/hyperparameter combinations for Adam, a six-step run over
four mixed shapes, and four SGD learning rates from `1e-9` to `1e12` -
while a separate test pins the **exact operation sequence** a staged
entry issues so a future reorder or fusion fails loudly. The two-phase
contract is untouched: validation is still four complete passes in the
same order with nothing moved behind a mutation, stage mutates no
parameter, moment, counter, version, or gradient, the commit is still
**one `copy_value_` and exactly one version increment per updated
parameter**, gradients are read and never written by identity, value,
and storage identity, and the documented per-entry commit boundary is
*tested* by injecting a `copy_value_` failure rather than assumed
infallible. Measured by alternating pre/post **subprocess** rounds (366
samples per case, correctness gated before timing): `NativeAdam.step()`
**1.58x** at (128, 128), **1.54x** at (256, 256), **1.48x** on a
four-parameter MLP with a 256-squared weight, 1.21-1.22x on a small MLP,
1.15x on a first step; a large MLP training step 1.23x, a small one
1.15x, a normalized step 1.13x, a CNN step 1.09x; and in the shipped
harness `adam_step` 1.25x, cutting the gap against
`tensorforge.optim.Adam` from **23.8x to 19.7x**. Reported just as
honestly: **a (512, 512) parameter is neutral** (1.02x,
memory-bandwidth-bound), the **Dropout training step is neutral**
(0.99x), and **NativeSGD is neutral-to-slightly-positive** (1.03-1.07x)
with one 0.88x row identified as **noise** by a focused re-measurement
whose post minima were lower in every pair - and the machine's
control-case noise band is stated at **0.84x-1.26x**, so no reading
inside it is mistaken for a result. H2's large-matmul performance is
intact. Memory moved with time, not against it: **peak live transient
bytes during one Adam step fell 2.6-3.0x** (1,966,160 to 655,424 at
(128, 128); 7,864,400 to 3,022,336 for a four-parameter MLP) and
per-parameter allocations went 27 to **17**, so a four-parameter model
allocates **76 instead of 108**. **Six alternatives were measured and
rejected**, each with its reason recorded: scalar materialization
(faster below roughly 32K elements, slower above, tracking this
machine's L2 cache rather than layout metadata, and it would regress the
harness's own profile configuration while adding a parameter-sized
buffer per scalar operation); same-shape stride-0 views (identical
kernel arguments by construction, but four NumPy layout arrays per call
where the broadcast path builds three); adopting the staged core instead
of `copy_value_`; giving `_native_copy` a `contiguous_copy`
implementation (it would stop normalizing `-0.0` to `+0.0`, a real
observable change in a helper shared far beyond the optimizer); a
persistent per-optimizer scalar cache (the hidden scratch tensor the
design forbids); and reassociating the update to fold scalars together
(a floating-point order change that would break every exact-resume proof
in the project). All instrumentation was test-local or benchmark-local;
**no production counter, environment-variable profiler, or installed
tracing mode exists**, and H4 added **no public API of any kind**. Three
pre-existing tests that injected a failure at the *N*-th
`NativeTensorCore.full` call were re-anchored to a per-parameter
allocation seam and a fourth now forwards `*rest` through the staging
signature - **every assertion in all four unchanged** - and new H4 tests
cover the `full` seam directly. No capability, dtype, device, registry
value, checkpoint field, or checkpoint version moved.
**Milestone H5 — native copy and mutation-transfer efficiency — has
since shipped**, and it is the first Phase-H milestone since H2 to change
C++ — though **not the ABI**: the library still exports exactly **52**
`tf_*` symbols. H5 replaced the native line's **value-transfer
primitive**. `_native_copy` was `zeros(shape) + core` — two allocations,
a full zero-fill pass, and a full elementwise-addition pass — and is now
the E3.1 native identity gather, `NativeTensorCore.contiguous_copy()`:
one uninitialized allocation (H1) and one pass. The composition predates
that gather and was simply never migrated to it. A complete inventory
found **ten** call sites of the one helper — `NativeParameter.copy_value_`
staging, both `state_dict()` snapshot paths, both `load_state_dict()`
staging paths, both BatchNorm running-statistic commits, and the
reshape/transpose/unbroadcast gradient materializations — and every one
of them is a **pure value transfer**: an independent contiguous
materialization of some tensor's current value, wanting no arithmetic.
All ten were enabled. `_broadcast_back`'s `zeros(x_shape) + upstream` was
**rejected** because it is not a copy at all but a genuine broadcast
expansion, which `contiguous_copy` cannot express; `sum`/`mean` and
`narrow_backward` keep their zeroed destinations for H1's unchanged
reasons.

The semantic question H4 refused to decide in passing was decided here,
by measurement over a fixed 18-pattern IEEE-754 sweep. **Exactly three**
patterns behaved differently under the two spellings: the addition
normalized `-0.0` to `+0.0` and quieted both signs of signaling NaN,
while the gather preserves all three. Everything else — `±0`, `±inf`,
quiet NaNs of either sign and **any payload**, denormals, the smallest
normal, the largest finite magnitudes — was already identical, so no NaN
payload differed at all (with one NaN operand and one zero, x86-64's
`ADDSD` returns that operand's NaN). **H2's matmul NaN-payload carve-out
does not generalize to copies**: it exists because two NaN operands meet
in an accumulation, and a copy performs no arithmetic. The pre-H5
behavior was **accidental and inconsistent**, not contracted — three
other value-copy paths (`NativeParameter(source)` construction,
`detach()`, and the `to_numpy()`/`from_array` boundary) always used the
gather and always preserved `-0.0`, while `copy_value_` documented the
same thing and did not deliver it. H5 states the narrowest coherent rule:
**a value transfer reproduces its source's bits exactly; an operation —
`zeros + x` included — follows IEEE arithmetic.** No operation's
arithmetic changed anywhere, and the whole pre-H5 suite passes unchanged
apart from the guardrails that pinned the old composition by name.

Swapping the composition alone would have **regressed** the common case,
so H5's one C++ change is a second **traversal** inside the unchanged
`tf_core_contiguous_copy` export. `zeros.add(core)` on a contiguous
source takes a flat pointer loop, while the gather always walked the
generic odometer — the only unary export without the contiguous fast path
every other one has — and a naive swap measured **0.48x** at 16,384
elements. The export now picks its traversal from the layout metadata it
already receives, exactly as H2's matmul picks its kernel:
`tf::copy_prefers_contiguous` is hidden-visibility C++ in a new internal
header, total, pure, allocation-free, and a function of metadata alone —
never of a pointer value, an alignment, a clock, an environment variable,
or a CPU-feature probe — testing exact equality against the row-major
strides implied by the shape, which is the same definition
`NativeTensorView` uses, so the two layers agree by construction. A false
answer falls back to the retained odometer and is never an error. **No
numerical carve-out is needed, and that is the difference from H2**: both
traversals evaluate `dst[out] = src[pos]` over the same logical elements
in the same destination order and differ only in how `pos` is computed,
so they are bit-identical *by construction* — proved directly at the C++
level by a new dependency-free CTest, taking the suite from 13 to 14.
There is no copy-mode selector, overlap-mode flag, traversal tracer, or
public dispatch control of any kind.

Nothing became less safe, because nothing became in-place: every call
site still **stages** an independent materialization and only then adopts
it. The overlapping arrangements the runtime can construct —
`copy_value_(self)`, a source that is a view of the destination's own
storage, a square parameter's own transpose, sibling views, duplicate
parameters across optimizers — are each tested and each correct, and no
`memcpy` is used anywhere. Parameter identity, storage replacement,
gradient retention by identity and value, the one version increment per
commit, the F1 state transaction, checkpoint atomicity, and exact resume
are all exactly what they were; gradient *accumulation* still adds rather
than assigns. H1's full-write contract is proved on both traversals by
poison injected purely by test infrastructure around the allocator, with
a negative control showing the detector can fail.

Measured by alternating pre/post **subprocess** rounds against a retained
pre-H5 composition, with a control band of **0.96x-1.05x**, and — for
the C++ half — by building a **pre-H5 library** and driving both through
identical `ctypes` calls on identical data, outputs proved bit-identical
before either was timed. The traversal alone: **2.5x-5.5x** on contiguous
sources from 16 K elements up (5.53x at 512 squared, 5.53x on 4-D NCHW,
5.46x on an offset view), 1.29-1.62x on small ones, and **0.94x-1.02x on
transposed and last-axis-narrowed sources**, which take the *unchanged*
odometer and are the design's own control. End to end: `copy_value_`
**2.14x** at (512, 512) and 1.26x at (128, 128), optimizer `state_dict()`
2.40x and `load_state_dict()` 1.69x, module `load_state_dict()` 1.37x,
`NativeSGD.step()` 1.15-1.31x. Reported just as honestly:
**`NativeAdam.step()` is neutral** (0.98x-1.06x — the commit copy is one
of about seventeen buffers and the arithmetic dominates), **every
training step is neutral** (0.95x-1.07x), the **BatchNorm running update
is neutral** (0.98x), and **copies below ~16 K elements are neutral**
(0.93x-1.01x), because a `contiguous_copy` call converts two `int64`
layout arrays at the ctypes boundary at **~1.1 us each** — a cost
measured, attributed, and left to a later dispatch milestone rather than
paid for by weakening H3's validation. Two methodology findings are
published rather than buried: at 7 alternating rounds the small copies
read 0.78x-0.94x and looked like a regression, while at 21 rounds the
same cases read 0.93x-1.01x (the same lesson H3 recorded); and the
largest single ratio, **7.9x-10.5x at 512-640 KB**, is a **512 KB
allocator cliff on this machine**, not a loop-speed result — the pre-H5
composition makes two large allocations and zero-fills one, so it crosses
that threshold at half the size and pays it twice. The durable statements
are ~2.1x at 1-2 MB and neutrality below 384 KB.

Memory moved with time, never against it: **no measured peak rose**, and
the pure-transfer paths halved. `copy_value_` at (512, 512) went 2
allocations to **1** and 4,194,304 to **2,097,152** peak bytes; module
`state_dict()` and `load_state_dict()` 4 to **2** allocations with peak
bytes halved; optimizer `state_dict()` 16 to **8**; `NativeSGD.step()`
5 to **4** with peak 393,216 to **262,152**; and `NativeAdam.step()` went
**17 to 16** allocations per parameter (H4 took it 27 to 17), removing a
whole-parameter zero-fill pass from every committed update. The harness
gained two cases, 26 to **28**: `row_major_materialization`, the
flat-traversal twin of the existing transposed-source case, so the two
traversals are separated rather than averaged; and
`parameter_value_commit`, `native_only` with **no ratio**, because the
stable line mutates a `Parameter` by rebinding `.data`, which is a
different operation. The ladder was **reordered** here — reduction
execution, drafted as H5, moved to H6 — and no public API, capability,
dtype, device, registry value, checkpoint field, or checkpoint version
moved.

**Milestone H6 — native reduction execution efficiency — has since
shipped**, the third Phase-H milestone to change C++ and, like H2 and H5,
**not the ABI**: the library still exports exactly **52** `tf_*` symbols.
Reductions were the last core family in the runtime that always paid the
generic strided indexing cost.

The pre-H6 kernel was re-read and re-measured rather than taken from H0's
or H5's summaries, and the cost was **decomposed** instead of assumed. At
`(256, 256)` `axis=0` a `core.sum` costs 99.7 us of which the **raw native
call is 94.8 us — 95 %**; subtracting the three `ndpointer` conversions
leaves the C++ traversal itself at ~91.6 us, **92 %** of the operation. The
entire Python wrapper — axis normalization (0.4 us), output-shape
construction (0.6 us), write-stride construction (0.5 us), the write-stride
array (0.4 us), the H3-cached layout arrays (0.1 us), and the output
allocation (3.2 us) — is about 5 us. So this was the **opposite** of B3:
H3's subject was a fixed Python cost that dominated *small* operations,
while a reduction of any real size is dominated by the compiled loop, and
H6's only worthwhile target was the traversal.

H6 therefore reused the dispatch shape H2 and H5 each proved — one hidden
metadata predicate, inside the existing export, no new symbol, the
pre-milestone traversal retained. New `cpp/include/tf_reduction_internal.h`
declares three hidden-visibility `namespace tf` functions and
`cpp/src/reduction.cpp` implements them: `tf::sum_generic_strided`, the
**pre-H6 odometer retained as the shipped generic reference path** — the
only path that can address a transposed, narrowed, non-unit-strided, or
broadcast source at all, and the oracle every optimized result is compared
against; `tf::reduce_prefers_contiguous_blocks`, the predicate; and
`tf::sum_contiguous_blocks`, a flat walk over an `outer x mid x inner`
factorization. The predicate is total, pure, allocation-free, and a
function of layout metadata alone — never of a pointer value, an alignment,
a clock, an environment variable, or a CPU-feature probe — and a false
answer is a fallback, never an error. It accepts a reduction when (1) the
source strides are exactly the row-major strides implied by the shape (the
same definition `NativeTensorView` uses, so the two layers agree by
construction), (2) the reduced axes — those with a zero *write* stride,
which is how the kernel has always identified them — form **one contiguous
run**, and (3) the kept axes carry exactly the row-major strides of the
output formed by dropping that run. **Stride collapsing is implicit and
bounded rather than a general layout compiler**: conditions 1 and 3 *are*
the statement that adjacent axes of the same class have identical address
progressions, so each group collapses by multiplication, nothing is cached
or interned, and non-adjacent reduced axes (unreachable from Python, which
still takes one `int` or `None`) simply fall back. `keepdims` needs no
special case and the kernel cannot even observe it.

**Per-output accumulation order is preserved exactly, and the source
traversal order is not even reordered**: the `o`, `m`, `i` loop nest is the
lexicographic order of the source's own row-major index, which is precisely
what the odometer walks, and every destination cell is touched by exactly
one `(o, i)` pair, so the cells are independent. Nothing is reassociated,
no partial sums are combined, no accumulator width changes, and no FMA,
Kahan, pairwise, tree, parallel, or horizontal-vector reduction exists.
The `inner == 1` branch (a full reduction, or one whose reduced run is a
suffix) uses a local accumulator **seeded from `dst[o]`**, which is what
keeps the export's documented accumulate-into semantics identical on both
paths; the `inner > 1` branch adds a contiguous source row elementwise into
a contiguous destination row, where distinct `i` are distinct outputs, so
any vectorization is across independent cells and never a horizontal
reduction.

**The signed-zero contract is proved, not assumed.** Both paths start from
the destination's `+0.0`, and `+0.0 + -0.0` is `+0.0`, so the sum of any
number of `-0.0` values is `+0.0` on both paths and matches NumPy; seeded
with `-0.0` both keep `-0.0`. All-positive zeros, all-negative zeros,
alternating zeros, `-0.0` first, `-0.0` last, `-0.0` mixed with finite
values, a column of `-0.0`, and exactly cancelling finite values are each
compared as **raw IEEE-754 bit patterns** at every axis, both `keepdims`
values, and scalar and multi-output shapes. One case is recorded rather
than idealized: the **rank-0** export branch is a genuine
`dst[0] += src[offset]` against a zeroed destination, so a rank-0 `-0.0`
sums to `+0.0` — exactly as before H6, and now pinned by a test.

**The NaN rule is H6's own, measured rather than inherited from H2.**
Contractual: NaN positions are identical on both paths; every NaN either
path produces is quiet, and a signaling-NaN input is quieted by both with
identical bits; and with **at most one NaN per accumulation** — every case
that occurs in practice — the two paths are bit-identical, payload
included. Not contractual: when **two or more** NaNs are accumulated into
one destination cell the paths may select different payload bits, asserted
in neither direction. Why parity is unavailable at any price was measured,
not asserted: four spellings of the optimized accumulation were compared —
`acc += x`, `acc = x + acc`, a named-temporary `acc = acc + x`, and
`dst[o] += x` accumulating *through memory* exactly as the odometer does —
and **all four selected the same NaN and all four differed from the
odometer**, so the local accumulator is not the cause and removing it would
recover nothing. The divergence comes from the odometer's destination index
being a runtime-varying value, which changes which addend MSVC places in
the `ADDSD` destination register; that is an instruction-selection decision
C++ cannot express. The memory-accumulate spelling was also **1.2x-1.8x
slower** on suffix reductions, so it bought nothing. Recorded as an
observation rather than a promise: the block path keeps the **first** NaN
in accumulation order, the odometer the **last**, and the block path's
choice is the one **NumPy** makes — so where they differ, H6 moved the
answer *toward* NumPy. **H5's copy rule does not apply here either**, for
the same reason it made H5's claim strong: a value transfer performs no
arithmetic and so has no operand roles to choose between, while a reduction
is arithmetic. Three operations, three genuinely different rules.

**H1's decision stands, and H6 confirms it rather than revisiting it.** The
destination stays zero-initialized on both paths, because both *read* it —
that is what accumulation means. Outcome B was rejected on two grounds,
one measured and one semantic: the fill is 2,048 bytes against 524,288
bytes of reads at `(256, 256)` `axis=0` and **8 bytes** at `axis=None`,
under half a percent of the traffic against a traversal that was 92-95 % of
the operation; and making the fast path *assign* its first contribution
would give the two paths different behavior for a non-zero destination,
breaking the export's accumulate-into contract and stopping the generic
path from being the reference. H6 therefore adds **no poison test**,
because it introduces no uninitialized destination; the H1 poison suite is
untouched and still passes, `sum` reaching `zeros` and never
`_uninitialized` is asserted structurally, and the accumulate-into behavior
that makes the zero load-bearing has its own negative control at the ABI.

Measured by building a **pre-H6 library** from the identical sources with
only `reduction.cpp` restored, driving both through identical `ctypes`
calls on identical data with every output proved **bit-identical before
either side was timed**, over 15 alternating pre/post rounds; the machine's
control band for this measurement is **0.90x-1.03x**. Kernel level: full
reductions 1.19x at 1,024 elements rising to **3.96x** at `(512, 512)`;
2-D axis reductions 3.24x at `(128, 128)` to **6.37x** at
`(1024, 1024) axis=0`; and — the finding that was **not** predicted —
3-D and 4-D reductions **8.60x-10.94x**, because the odometer's carry loop
runs up to `ndim` iterations per element so its cost grows with rank while
the block traversal's does not. The NCHW rows matter because that is the
layout the convolution stack produces. Layer level, over 9 alternating
subprocess rounds: `TensorCore.sum(axis=0)` 4.49x and `mean(axis=0)` 4.11x
at `(256, 256)`, NCHW `sum(axis=1)` **8.56x** and `sum(axis=3)` **8.82x**,
`NativeTensor.sum` 3.88x without a graph and 3.82x with one, `sum()`
forward+backward 1.27x, `mean` forward+backward 1.23x, the **convolution
bias gradient's three chained sums 1.46x**, `_unbroadcast` 1.15x, softmax
backward 1.14x, log-softmax backward 1.10x, `NativeLayerNorm` forward
1.16x, `NativeBatchNorm2d` backward 1.10x, cross-entropy forward and
backward 1.05x. Against NumPy in the shipped harness the contiguous
reduction gap closed from roughly 8-13x to **1.67x** (4-D middle axis),
**2.43x** (axis 0), 2.90x (last axis), and 3.75x (full to scalar), while
the transposed-view control stayed at 10.33x.

Reported just as honestly. **Every training step is neutral** — MLP small
0.99x, MLP large 1.03x, normalized 1.03x, CNN 1.01x, Dropout control 1.02x,
all inside the control band — so **H6 does not make training faster**, and
no reading should be quoted as if it did; a reduction is a small share of a
step whose cost is the optimizer and the large matmuls. **Normalization is
mostly neutral** too: BatchNorm1d training forward 1.04x, eval 0.98x,
backward 1.02x, BatchNorm2d training forward 1.06x, eval 1.00x, LayerNorm
backward 1.01x, with only LayerNorm forward and BatchNorm2d backward
clearly outside the band — which **narrows H7 rather than motivating it**,
since what is left in those modules is the sheer count of broadcast
elementwise operations rather than the reductions. **Tiny reductions are
neutral** (1 element 1.00x, 16 elements 1.01x, `(8, 8)` axis 0 1.03x),
because below roughly 1,000 elements the fixed ~7 us Python-plus-ctypes
cost dominates — H3's and H5's documented boundary finding, left to a
dispatch milestone. And one **real, repeatable ~10 % regression** is
published rather than buried: a **2-D transposed source reduced over
`axis=0`** measured 0.89x-0.93x across four independent 25-round runs,
while the 3-D transposed `axis=0` fallback measured **1.04x-1.05x faster**
and every other fallback 0.96x-1.01x. Both libraries run the *identical*
odometer there, and the cause was **isolated**: in a standalone binary the
extracted-function spelling versus the inline spelling measured 0.88x-1.67x
with no stable direction, so the extracted call is not it — the remaining
attribution is whole-translation-unit code layout, which is exactly the
machine-specific tuning the design rejects chasing. It affects no shipped
path and no end-to-end case regressed. A specialized register-blocked path
for a small trailing extent (`inner=2` measured 1.75x, `inner=4` 1.77x —
the weakest wins) was **rejected on complexity**. Methodology is published
too: at 7 alternating rounds the fallback controls read 0.85x and at 21-25
rounds the same cases read 0.90x-1.02x, the same lesson H3 and H5 each
recorded, so no low-round figure is quoted as H6 evidence.

**Memory moved not at all, and that is asserted rather than assumed**: a
`sum` allocates **exactly one** native storage — its own output — on both
paths, at every axis, under both `keepdims` values, and `mean` allocates
the same one because its scale is in place. There is no scratch buffer,
workspace, arena, or pool, and the odometer's counter is unchanged and only
on the fallback path. A 10-step training run over a model carrying
parameters, BatchNorm buffers, and Adam moments produced a **bit-identical**
allocation and live-count profile before and after H6, which also confirms
that profile's oscillation is CPython's collector rather than a leak either
version introduced.

The harness gained three cases, 28 to **31**, following H5's
separate-rather-than-average precedent: `reduction_last_axis` (the suffix
form LayerNorm's mean and both softmax backwards actually reduce over),
`reduction_full_to_scalar` (every write stride 0 and a rank-0 output — the
hottest reduction in the runtime, since every mean-reduced loss ends in
it), and `reduction_middle_axis_4d` (kept axes on both sides, so all three
block extents exceed 1, plus the rank-4 reading the 2-D cases cannot give),
with `reduction_transposed_view` now explicitly the pair's control because
the predicate rejects it. One dependency-free CTest was added,
`cpp/tests/test_sum_reduction.cpp`, taking the native suite from 14 to
**15**; it drives the predicate table, both traversals in isolation, the
accumulate-into contract over a pre-filled destination, and the
special-value matrix at the layer where those properties are actually
decided. **No exported C ABI symbol, no new translation unit, no public
control of any kind** — no path selector, threshold setter, block-size
setter, dispatch tracer, profiling counter, environment variable, or
"which path ran" query — and no SIMD, threading, OpenMP, BLAS, parallel
reduction, memory pool, scratch workspace, or fast-math. Multi-axis
reduction was **not** added: the kernel can factorize a contiguous reduced
run, but the Python layer still accepts one `int` or `None`, with every
signature, default, axis rule, `keepdims` behavior, error type, and error
message exactly what they were. `tf_core_narrow_backward`, the odometer's
scatter dual, was deliberately left alone — widening H6 to it would have
made this a scatter milestone. No public API, capability, dtype, device,
registry value, checkpoint field, or checkpoint version moved.

**Milestone H7 - native Python/C ABI boundary efficiency - has since
shipped**, and it is **Python-only**: no C++, no exported symbol, no
kernel, no traversal, no arithmetic. The library still exports exactly
**52** `tf_*` symbols.

**The ladder was revised here, and the revision is recorded rather than
retrofitted.** H0's H7 slot was *composed-module cost* - the normalization
modules and the composed convolution bias gradient - explicitly
conditional on a re-measurement after H1, H3, and H6. That condition was
tested and **not met**: H6 made `mean` 3.9x-4.1x faster and moved the
normalization modules almost not at all (`NativeLayerNorm` forward 1.16x,
`NativeBatchNorm2d` backward 1.10x, everything else inside the
0.90x-1.03x control band, the normalized training step 1.03x). So the
milestone was **dropped on evidence**, its proposal and the evidence
against it preserved verbatim in the design document rather than deleted,
and the slot was refilled from the *same* measurements: H3, H5, and H6
had each ended by deferring one identically named cost - H5 "~1.1 us per
layout array at the ctypes boundary, left to a later dispatch milestone",
H6 "the fixed ~7 us Python-plus-ctypes cost, left to a dispatch
milestone". Three milestones deferred one thing to a later dispatch
milestone; H7 is that milestone. Composed-module allocation count remains
conditional future scope and is **H8's** subject.

The cost was **decomposed rather than assumed**, and the assumption that
six kernels were involved was checked and found wrong. All 52 exports are
configured in one file - **no other module in the repository imports
`ctypes`** - and **57 of their argument positions are arrays**, every one
formerly bound as `numpy.ctypeslib.ndpointer`. That binding re-verifies
array-ness, exact dtype, and contiguity at **every call**, then builds
`obj.ctypes` and resolves it through `_as_parameter_`: two Python object
constructions and three checks, per array, per call, measured at **~2.1 us
per array position**. On real calls, `tf_core_add` on a 4x4 with three
layout arrays cost **7.6 us**, of which **6.1 us** was the binding, while
the array-free `tf_core_add_contiguous` cost 0.9 us and is the control.
Frequency was then counted: an MLP training step makes 245 native calls
carrying **101** array crossings, a normalized step 692 and **315**, a CNN
step 242 and **104** - about **20-23 %** of each step - and the provenance
was the finding that decided the architecture: **~85 % of those crossings
are operation-local broadcast strides**, not the H3 per-view cache, so a
design that only cached pointers per view would have captured a seventh of
the available work.

H7 ships **two bindings for two categories, and deliberately not one
blanket policy**. *Data* positions keep the checked `ndpointer` binding -
the seven public raw-buffer kernels (whose callers may pass anything), the
`copy_from`/`copy_to`/`materialize` host conversion boundary, and the
cross-entropy **class labels**, which are int64 like the layout metadata
but stay checked because a label array's required length comes from the
*logits*, a different object. *Layout metadata* positions - **32 across 13
exports** - take `ctypes.POINTER(ctypes.c_int64)`, fed by exactly two
private producers: `NativeTensorView._native_layout_pointers()`, which
memoizes `data_as` over the **unchanged** H3 read-only NumPy arrays that
remain the owning buffers, and `_layout_vector(values)`, which builds a
fresh `(c_int64 * len(values))` for metadata belonging to one operation.

**Nothing was weakened and one thing was strengthened.** ctypes still
type-checks every call: a trusted position rejects a NumPy array of any
dtype, a differently typed pointer or vector, a `c_void_p`, a list, an
int, bytes, and a string - a NumPy array being rejected is a deliberate
consequence that makes the old binding unreachable by accident. Dtype,
byte order, and contiguity are established *by construction* rather than
re-checked, and **the length/rank invariant - the one `ndpointer` never
checked, because the ABI sees only a pointer and an `ndim`** - became
checkable for the first time: a vector carries its length in its type, and
a cached pointer carries its owning array, whose length is the view's
rank. That is asserted per producer, per rank 0-4, and structurally over
**every strided call in a real workload**. The one honest difference is
`None`, which `ndpointer` rejected and a typed pointer converts to NULL;
it is closed structurally - both producers are total, no public API takes
a metadata pointer. (Only **three** of the thirteen exports reject a null
metadata pointer in C++ as well.)

**Ownership is NumPy's guarantee, relied on and tested rather than
assumed**: `data_as` stores the array on the pointer, so a cached pointer
cannot outlive its buffer. `POINTER.from_address` was measured **faster**
(0.9 us against 1.6 us) and **rejected outright** for producing a pointer
with no owner, and a cached ctypes vector - fastest of all - was
**rejected** because it would create a second owning description of a
view's layout and lose H3's `writeable = False`, for ~2 % of a training
step. Proved with the cyclic collector **disabled**: no reference cycle,
no native storage kept alive, no usable pointer after close, and
operation-local vectors retained by nothing. There is no global pointer
cache and no `from_address`, `byref`, `addressof`, `id`, or weak-reference
container **anywhere in the module's code**, enforced by parsing it rather
than grepping it. Binding configuration stays a **load-time act** -
`argtypes`/`restype`/`errcheck` are assigned only inside the two loader
functions, asserted by locating each assignment's enclosing function - so
nothing reconfigures a shared function object per call and **no
thread-safety claim is broadened**.

Measured against a **retained pre-H7 `cpp.py` driving the same Release
DLL**, over 11 alternating pre/post subprocess rounds, every case proved
**bit-identical before either side was timed**; control bands 0.95x-1.10x
(core) and 0.99x-1.05x (end to end). Core: a 1-element `sum` **1.94x**, a
16-element `sum` 1.89x, `to_numpy` 16x16 **1.83x**, `sum(axis=0)` 16x16
1.79x, a 4x4 `contiguous_copy` 1.73x, `narrow_backward` 1.73x, strided
`relu_backward` 1.71x, scalar-broadcast `add` **1.67x**, 4-D NCHW
`sum(axis=1)` 1.54x, row-broadcast `add` 1.41x, `sum(axis=0)` 256x256
1.35x, strided `exp` 1.29x, transposed materialization 1.16x. End to end -
and this is the result - the **native Dropout step 1.32x, the normalized
step 1.31x, `NativeAdam` at (32, 32) 1.31x, the CNN step 1.30x, the MLP
step 1.28x**, `NativeLayerNorm` forward 1.23x, `NativeBatchNorm1d` eval
1.23x, `NativeAdam` at (128, 128) 1.14x, `NativeSGD` 1.13x, the large MLP
step 1.08x. **H7 is the first Phase-H milestone to move every training
step** - H4 moved them 1.09x-1.23x and H5 and H6 were neutral on all of
them - because the cost is paid per *call* and a step makes hundreds of
them.

Reported just as honestly: **large kernel-bound work is neutral**, exactly
as the attribution predicts. 256-cubed matmul **0.99x** and 8-cubed matmul
1.00x are controls that take no array at all, so **H2's result is
structurally untouched**; contiguous 16x16 `add` 1.05x is the third
array-free control; and 512x512 `copy` 1.02x, 256x256 `to_numpy` 1.04x,
512x512 full `sum` 1.06x, 256x256 broadcast `multiply` 1.08x, and the
large MLP step 1.08x are all at or inside the band. No reading should be
**H7 did not make matmul faster**, and no reading should
say otherwise.

A second, independent 11-round run reproduced every row: all cases again
bit-identical, every control again holding (256-cubed matmul 0.98x, 8-cubed
matmul 1.04x, contiguous 16x16 `add` 1.08x, 512x512 `copy` 1.01x), and every
training step again improving, with individual ratios moving by roughly the
control band's width in both directions (a 1-element `sum` 2.12x against
1.94x, `NativeSGD` 1.26x against 1.13x, `NativeAdam` at (128, 128) 1.08x
against 1.14x). The figures quoted are the first run's; the second is
recorded in the design so no single number is read as more precise than the
method supports.

**Memory did not move, and that is asserted**: the same boundary workload
allocates 5 native storages, peak 4 live, 584 peak bytes before and after.
A view's cold footprint is byte-identical; a view that actually takes a
strided path costs **+296 bytes** for the pointer pair, and only **9 of
98** views in an MLP step ever populate it, which is H3's laziness
argument unchanged. The harness gained three cases, 31 to **34**:
`ctypes_boundary_strided` (the array-carrying twin of the array-free
`ctypes_boundary` - **1.3 us** against **0.8 us**, where pre-H7 it would
have been ~7 us) plus `elementwise_broadcast_scalar` and
`elementwise_broadcast_row`. Validation added a **sanitizer negative
control**: under Clang ASan, test-only code handing `tf_core_sum`
two-entry metadata with `ndim = 3` produces a `heap-buffer-overflow`,
`READ of size 8`, `0 bytes after 16-byte region`, in
`reduce_prefers_contiguous_blocks` - the exact H3 finding - which is what
makes the **zero diagnostics across 2,834 sanitized tests** a real absence
rather than a blind detector. No public API, capability, dtype, device,
registry value, checkpoint field, or checkpoint version moved, and no C
ABI symbol was added.

Data loaders, native integer tensors, further
dtypes/devices, and CUDA experiments are
future work beyond Phase H.
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
  ownership), G2 (the stateless `dropout_forward` **Core** kernel and
  its C ABI), G3 (the differentiable `NativeTensor.dropout`), G4 (the
  `NativeDropout` module), and G5 (native checkpoint **format version 2**
  with persisted generator state and its alias topology), and G6 (the
  RNG/graph/ownership/checkpoint hardening, which added no capability),
  and G7 (`examples/native_dropout_training.py` and its tests — the
  deterministic stochastic training and exact-resume proof, which also
  added no capability), and G8 (`benchmarks/benchmark_native_dropout.py`
  and its tests — the honest characterization, correctness gated
  before timing, no speed asserted, no capability), and G9
  (`tests/test_native_phase_g.py` — the cross-cutting integration
  suite over one model carrying every registered state family, also
  no capability), and G10 (`tests/test_native_phase_g_closure.py` and
  the documentation reconciliation — the Release/Debug/sanitizer
  validation matrix and the single registry line that removed
  `"dropout"` from `UNSUPPORTED`) have all shipped — **Phase G is
  complete** —, and `native_cpu_performance_design.md` for Phase H —
  **H0** (the design lock, the unified baseline harness
  `benchmarks/benchmark_native_cpu_performance.py`, its contract tests,
  and documentation reconciliation; architecture, profiling, and baseline
  work with **no optimization, no capability, and no registry move**),
  with **H1 complete** (the output-allocation contract: one new C ABI
  symbol — `tf_storage_create_uninitialized`, and nothing else — a
  per-kernel audit table, completeness proved by poison injected purely
  by test infrastructure around the allocator (no poison control exists
  in the shipped runtime), bit-identical results, `sum` and
  `narrow_backward` rejected, and no
  capability move) and **H2 complete** (native matmul memory access:
  `tf_core_matmul` ships two compute paths behind the same unchanged
  export — `tf::matmul_generic_strided`, the pre-H2 `i`-`j`-`k` triple
  loop retained verbatim as the generic reference path, and
  `tf::matmul_row_sweep`, an `i`-`k`-`j` sweep over four destination rows
  at a time — chosen inside the kernel from stride metadata; **cache
  blocking measured against 22 variants and rejected**; a four-part
  numerical contract rather than a blanket bit-identity claim — identical
  accumulation order, bit identity on every non-NaN result, NaN-class
  equivalence, and NaN payload bits deliberately outside the contract;
  H1's uninitialized-output contract preserved on both paths; **no exported C
  ABI symbol added**, still 52; and no capability move) and **H3 complete**
  (native metadata and dispatch efficiency — **Python-only**: one
  normalization boundary (`_normalized_layout`) replacing the four
  redundant re-validations every `shape_info` call performed, private
  `_checked` primitives for the derived strides/count/contiguity, a
  private `NativeTensorView._from_validated` sharing one
  bounds-checking `_bind` with the public constructor, and lazy
  read-only per-view `int64` layout arrays whose immutability makes
  staleness impossible by construction; every rejection, message, and
  ordering preserved; nothing global; no public cache, profiling, or
  dispatch API; **no C++, no ABI, still 52 symbols**; measured
  `_as_int_tuple` per MLP step 815 → 149, an MLP step 1.43×, a CNN step
  1.29×, a normalized step 1.51×, and **no measurable change on large
  kernel-bound work**; no capability move) and **H4 complete**
  (native optimizer step efficiency - **Python-only** again: the step's
  scalar coefficients built once per step in a private per-step
  `_StepConstants` holder that is never stored on the optimizer, the
  bias-correction reciprocal evaluated in Python as an **exact
  substitution** for the native kernel (which literally is `1.0 / x` on
  the same IEEE-754 binary64 value, proved over 20,000+ values on raw
  bit patterns), and every temporary released at its last use;
  bit-identical against a **pre-H4 composition retained in the test
  suite and executed natively**, with the two-phase stage/commit
  contract, the one `copy_value_` per updated parameter, the version
  counting, and the gradient-retention rule all exactly what they were;
  **no C++, no ABI, still 52 symbols**; measured `NativeAdam.step()`
  1.58x at (128, 128), 1.48x on a four-parameter MLP, an MLP training
  step 1.23x, a normalized step 1.13x, the gap against the stable
  adaptive optimizer 23.8x -> 19.7x, per-parameter allocations 27 -> 17,
  peak transient bytes down 2.6-3.0x - and **a (512, 512) parameter, the
  Dropout training step, and NativeSGD all neutral**, with six
  alternatives measured and rejected; no capability move) and
  **H5 complete** (native copy and mutation-transfer efficiency: the
  value-transfer primitive `_native_copy` moved from `zeros + core`
  to the E3.1 native identity gather across all ten of its call
  sites, `_broadcast_back` rejected as a genuine broadcast; exactly
  three of eighteen IEEE-754 patterns moved — the addition
  normalized `-0.0` and quieted both signs of signaling NaN — with
  no NaN payload differing at all, so H2's matmul payload carve-out
  does not generalize; one hidden C++ traversal predicate inside the
  unchanged export, bit-identical by construction, **no ABI change,
  still 52 symbols**; `copy_value_` 2.14x at (512, 512), optimizer
  `state_dict()` 2.40x, the traversal alone 2.5-5.5x contiguous and
  0.94-1.02x transposed (the unchanged odometer control) — with
  NativeAdam, every training step, and small copies all neutral;
  allocations down everywhere and no peak raised; no capability
  move) and **H6 complete** (native reduction execution efficiency:
  reductions were the last core family always paying the generic
  strided indexing cost, and the pre-H6 kernel was re-measured and
  **decomposed** — the raw native call is **95 %** of a
  `(256, 256)` axis-0 `core.sum` and the whole Python wrapper about
  5 us, the opposite of H3's finding. `tf_core_sum` now ships two
  traversals behind the same unchanged export — `tf::sum_generic_strided`,
  the pre-H6 odometer **retained as the shipped generic reference
  path** and the only path that can address a transposed, narrowed,
  non-unit-strided, or broadcast source at all, and
  `tf::sum_contiguous_blocks`, a flat walk over an
  `outer x mid x inner` factorization — chosen by
  `tf::reduce_prefers_contiguous_blocks`, hidden-visibility C++ that
  is total, pure, allocation-free, and a function of layout metadata
  alone, with a false answer a fallback rather than an error. Stride
  collapsing is implicit and bounded, not a general layout compiler;
  nothing is cached; `keepdims` is invisible to the kernel;
  multi-axis reduction was **not** added; and
  `tf_core_narrow_backward` (the scatter dual) was deliberately left
  alone. Per-output accumulation order is preserved exactly and the
  source traversal order is not even reordered; signed zeros are
  proved as raw bit patterns; the **NaN rule is H6's own, measured
  rather than inherited from H2** — bit-identical whenever at most
  one NaN enters an accumulation, with payloads outside the contract
  only when two or more meet in one cell, after four accumulation
  spellings all diverged from the odometer identically so parity was
  unavailable at any spelling; and **H1's rejection of this
  destination stands** (both traversals read it, so it stays
  zero-initialized and H6 adds no poison test). Measured against a
  pre-H6 library on identical `ctypes` calls, bit-identical before
  timing, control band 0.90-1.03x: full reductions up to **3.96x**,
  2-D axis reductions **3.24-6.37x**, 3-D/4-D reductions
  **8.60-10.94x** (unpredicted — the odometer's carry loop scales
  with rank), `mean` 4.11x, the convolution bias gradient **1.46x**,
  the NumPy gap on contiguous reductions closed from ~8-13x to
  **1.67-3.75x** — with **every training step neutral**,
  normalization mostly neutral (which narrows H7), tiny reductions
  neutral, and a real repeatable **~10 % regression on 2-D
  transposed axis-0 fallbacks** published and attributed to
  whole-translation-unit code layout. One allocation per `sum` on
  both paths; harness 28 to **31** cases; CTests 14 to **15**; **no
  ABI change, still 52 symbols**; no capability move) and **H7 complete**
  (native Python/C ABI boundary efficiency - **Python-only**: the ctypes
  argument binding for this project's own layout metadata. **The ladder
  was revised here**: H0's composed-module H7 was **dropped on evidence**
  (H6 measured the normalization modules mostly neutral) and the slot
  refilled with the boundary work H3, H5, and H6 had each deferred by
  name. 57 array argument positions were inventoried; **32 layout-metadata
  positions across 13 exports** moved from the checked
  `numpy.ctypeslib.ndpointer` binding - measured at **~2.1 us per array
  per call** - to `ctypes.POINTER(ctypes.c_int64)`, fed by exactly two
  private producers (a per-view pointer memoized over the **unchanged** H3
  read-only arrays, and a fresh `c_int64` vector for operation-local
  metadata), while the **25 data positions stay checked** - the raw public
  kernels, the host conversion boundary, and the cross-entropy class
  labels. ctypes still type-checks every call, and the **length/rank
  invariant `ndpointer` never checked** became checkable for the first
  time; `POINTER.from_address` was measured faster and **rejected** for
  owning nothing. Measured against a retained pre-H7 `cpp.py` on the same
  Release DLL, bit-identical before timing: tiny operations 1.3x-1.9x and
  **every training step 1.08x-1.32x - the first Phase-H milestone to move
  them all** - with large kernel-bound work neutral and three array-free
  control cases confirming it. Harness 31 to **34** cases; an ASan
  **negative control** proves malformed metadata is detectable; **no C++,
  no ABI change, still 52 symbols**; no capability move) and H8-H11
  proposed and explicitly conditional on that evidence, so **Phase H has
  begun but is not complete**). When a milestone changes the
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
  - `uv run python examples/native_dropout_training.py`
- Run the Phase-H CPU baseline (measurement only; no speed asserted):
  - `uv run python benchmarks/benchmark_native_cpu_performance.py --smoke`
  - `uv run python benchmarks/benchmark_native_cpu_performance.py --json`
  - `uv run python benchmarks/benchmark_native_cpu_performance.py --workload matmul`
  - `uv run python benchmarks/benchmark_native_cpu_performance.py --profile matmul_square_contiguous`

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
