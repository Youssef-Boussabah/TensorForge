"""Experimental, opt-in APIs built on the native C++ backend.

Nothing here is part of the finished Python framework, and
``import tensorforge`` never imports this package — you reach it
explicitly:

    from tensorforge.experimental import NativeTensor

``NativeTensor`` is a native tensor wrapper over the native runtime
(NativeTensorCore) with an opt-in, Python-managed reverse-mode autograd
graph (Phase B, complete as of Advanced C++ v2.6). It is **not**
tensorforge.Tensor: the two autograd engines never mix, no conversion is
implicit, and it shares no state with the stable framework. A full native
training stack — parameters, modules, layers, a loss, optimizers, and
pickle-free checkpoints — is built on it and described below. The native
CNN stack (Phase D) is **complete** (milestones D0–D12): it began with
``NativeFlatten`` (milestone D1) and,
as of milestone D6, the differentiable **``NativeTensor.conv2d``** operation
(NCHW/OIHW cross-correlation with int/tuple stride and padding and optional
bias; input, weight, and bias gradients through native backward kernels and
the existing ``sum`` reduction), and as of milestone D7 the trainable
**``NativeConv2d``** module built on it (OIHW weight / optional ``(O,)``
bias ``NativeParameter``s, deterministic uniform conv fan-in initialization,
4-D NCHW input validation, and backward supplied entirely by the D6
autograd — no new kernel, ABI symbol, or custom module backward).
Milestones D8 and D9 added the differentiable **``NativeTensor.maxpool2d``**
operation (NCHW window maxima with int/tuple ``kernel_size``/``stride``/
``padding``; its backward scatters through the private winner buffer its own
forward saved, so it never rereads the input, never recomputes a maximum,
and records no parameter-version snapshot), and milestone D10 exposes it as
the **``NativeMaxPool2d``** layer: a parameter-free, buffer-free module that
normalizes ``kernel_size``/``stride``/``padding`` to two-element tuples
(``stride=None`` ⇒ non-overlapping windows) and delegates its forward
entirely to that operation — no new kernel, ABI symbol, custom backward, or
state. It holds no winner storage between calls and contributes no
state-dictionary or checkpoint keys, so it drops into a ``NativeSequential``
beside ``NativeConv2d``/``NativeFlatten`` without touching the optimizer or
checkpoint paths. Milestone D11 proved the whole stack trains — see
``examples/native_cnn_training.py``, whose checkpoint-interrupted run
reproduces the uninterrupted one exactly — and **milestone D12 closed
Phase D** with cross-cutting integration tests, honest CNN benchmarks, and
ASan/UBSan validation. The native **classification** stack
(contracted for Phase E in docs/native_classification_design.md) is
**complete**: milestones E1-E4 shipped the differentiable ``exp``,
``log``, ``softmax``, and ``log_softmax``; E5 and E6 shipped the fused
stable ``cross_entropy`` — its graph-unaware Core contract and then the
differentiable ``NativeTensor.cross_entropy`` with graph-owned saved
probabilities, no logits reread, and no expected version snapshot; and
**milestone E7** adds the public surface described below,
``NativeCrossEntropyLoss`` and ``native_accuracy``; and **milestone E8**
proves the assembled stack end to end without adding to it —
``examples/native_classification_training.py`` trains a native
Conv2d/ReLU/MaxPool2d/Flatten/Linear classifier over **raw logits** on
twelve fixed 6x6 images in three classes for 40 deterministic
``NativeAdam(lr=0.05)`` steps (loss 1.159638 -> 0.000101, reporting
accuracy 0.3333 -> 1.0000), then checkpoints at step 15 and resumes into
a fresh model/optimizer pair that reproduces the remaining losses,
parameters, optimizer state, logits, predictions, and accuracy exactly
(native checkpoint format version 1 unchanged); and **milestone E9**
characterizes that stack in
``benchmarks/benchmark_native_classification.py`` — seven
correctness-gated cases with honest reference labels, medians and spread
after warm-up, ``--smoke``/``--json`` modes, and no speed assertion or
timing threshold anywhere. **Milestone E10 closed Phase E** with
cross-cutting integration tests (``tests/test_native_phase_e.py``),
Release and Debug native builds, Clang ASan/UBSan and LeakSanitizer
validation, and documentation reconciliation — adding no numerical
capability. **Phase E is complete.**

**Phase F — Native Normalization and Stateful Buffers — is complete
(F0-F9).** Its architecture contract is locked in
``docs/native_normalization_design.md`` (milestone **F0**, complete:
design and repository reconciliation only, adding no numerical
behavior). It specifies ``NativeLayerNorm``, ``NativeBatchNorm1d``, and
``NativeBatchNorm2d`` **composed from existing native operations** —
adding no kernel, C ABI export, ctypes declaration, or
``NativeTensorCore`` method — with persistent native running statistics,
the rule that a live mutable running buffer is never captured as a
rereadable graph operand (eval mode takes independent graph-free
snapshots, which is why buffers stay unversioned), atomic two-buffer
running-statistics updates, and state/checkpoint integration with the
format unchanged at version 1. **Milestone F1** shipped the private
atomic native-buffer state transaction that contract requires
(``_native_state.py`` — staging, an explicit commit boundary, complete
rollback, exactly-once closing, and identity-preserving swaps), which
``NativeModule.load_state_dict`` now delegates to, plus the
``persistent_buffers`` entry in ``STATE_SUPPORT`` reconciling a
capability that already existed. **Milestone F2** ships
``NativeLayerNorm`` below: the first native normalization module —
stateless (no buffers, identical in train and eval), differentiable
through the mean and the population variance, and **composed entirely
from existing native operations** (``mean``, ``subtract``, ``multiply``,
``add``, ``sqrt``, ``reciprocal``) with ``sqrt(var + eps)`` ordering and
no kernel, ABI symbol, ``NativeTensorCore`` method, custom backward, or
``NativeTensor`` normalization operation. It normalizes trailing
one-or-more-dimensional shapes, holds ``weight`` and ``bias``
``NativeParameter``s only when ``elementwise_affine=True`` (none
otherwise), so ``"NativeLayerNorm"`` has joined ``NATIVE_MODULES`` and
``"layernorm"`` has left ``UNSUPPORTED``. **Milestone F3** ships
``NativeBatchNorm1d`` — the **first stateful native numerical module**:
``(N, C)`` batch normalization whose training statistics are
differentiable (gradients flow through the batch mean and the population
variance), whose ``running_mean``/``running_var`` are **persistent native
buffers** advanced by a graph-free, atomic two-buffer update through the
F1 transaction (identities preserved, no parameter version moved), and
whose evaluation mode reads **graph-safe immutable snapshots** of those
buffers rather than the live objects. It is composed from the same
existing operations — no kernel, C ABI symbol, ctypes declaration,
``NativeTensorCore`` method, ``NativeTensor.batch_norm`` operation, or
custom BatchNorm backward — and the native checkpoint format stays
version 1. ``"NativeBatchNorm1d"`` has joined ``NATIVE_MODULES``, while
``"batchnorm"`` stayed in ``UNSUPPORTED``: the unqualified name is only
honest once ``NativeBatchNorm2d`` ships too. **Milestone F4** ships
``NativeBatchNorm2d`` below — NCHW ``(N, C, H, W)`` batch normalization
reducing over **N, H, and W**, so each channel gets one population mean
and one population variance over ``N * H * W`` values. It is built on
the **same** shared private implementation as ``NativeBatchNorm1d`` and
declares nothing but its rank, its reduction axes, its ``(1, C, 1, 1)``
broadcast layout, and the channels-last permutation its rank-1
``gamma``/``beta`` need: rank-1 parameters broadcast from the *trailing*
axis, so the **activation** is transposed for the affine application and
back again (then materialized contiguous) rather than the parameters
being reshaped — which keeps ``gamma`` a direct versioned ``multiply``
operand and preserves the existing stale-parameter guard exactly.
Running statistics stay ``(C,)`` persistent buffers, evaluation reads
owning ``(1, C, 1, 1)`` snapshots, the checkpoint format stays version
1, and again no kernel, C ABI symbol, ctypes declaration,
``NativeTensorCore`` method, custom backward, or
``NativeTensor.batch_norm`` operation exists.
``"NativeBatchNorm2d"`` has joined ``NATIVE_MODULES`` and the exports,
and with both shapes live ``"batchnorm"`` has **left** ``UNSUPPORTED``,
which at that point read exactly ``("dropout", "float32", "cuda",
"amp")``.
**That completes the numerical normalization module surface. Milestone
F5 is complete** — the exhaustive state, checkpoint, ownership, and
graph-safety hardening (a focused ``tests/test_native_normalization_state.py``
plus narrow additions to the generic buffer and checkpoint suites),
proving §7-§10 of the design by executable test rather than by prose:
**tests and documentation only, no numerical behavior and no new public
capability**, with the exports, every capability registry, and the
version-1 checkpoint format all exactly what F4 left. **Milestone F6 is
complete** — the deterministic normalized training and exact
checkpoint-resume proof ``examples/native_normalization_training.py``: a
``Linear -> BatchNorm1d -> ReLU -> LayerNorm -> Linear`` regressor
(both normalization families in every forward, BatchNorm the only stateful
module) trained with ``NativeAdam``/``NativeMSELoss``, whose two
uninterrupted runs are bit-identical and whose interrupted checkpoint
resume into a fresh model/optimizer pair reproduces the losses,
parameters, NativeAdam state, BatchNorm running statistics, final training
prediction, and evaluation-mode output exactly — **one example and its
integration test, no capability or schema change, format version 1
unchanged**. **Milestone F7 is complete** — the honest benchmark
characterization ``benchmarks/benchmark_native_normalization.py``: nine
cases (the LayerNorm forward and backward, the BatchNorm1d training
forward, evaluation forward, and backward, the BatchNorm2d training
forward, evaluation forward, and backward, and one complete F6-style
normalized training step), each **correctness-gated before any timing**,
six labelled ``stable_tensorforge`` against ``tensorforge.nn``
equivalents on identical state and three (the BatchNorm2d shapes)
labelled ``native_only`` because the stable line has no public
``BatchNorm2d`` to time against — those publish no ratio while keeping a
rigorous NumPy NCHW and transformed-oracle correctness gate. Medians with
min, max, and spread after warm-up; ``--smoke``/``--json`` modes; **no
result file, no speed assertion, no committed timing number, and no CI
timing threshold** — **measurement only, no capability, operation,
kernel, C ABI symbol, schema field, example, or export, and no production
behavior changed**. **Milestone F8 is complete** — the cross-cutting
integration and semantic guardrails ``tests/test_native_phase_f.py``: one
integrated ``Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Flatten ->
Linear -> BatchNorm1d -> ReLU -> LayerNorm -> Linear`` classifier over
raw logits and the fused loss, trained by ``NativeAdam`` and resumed
**exactly** from one version-1 checkpoint (all four running-statistic
buffers and the evaluation-mode output included); the three
saved-resource families (BatchNorm snapshots, MaxPool2d winners,
cross-entropy probabilities) coexisting in one eval graph and releasing
exactly once; buffer mutation leaving an earlier graph valid while
parameter mutation correctly stales it; the versioning archetypes; shared
and frozen parameters; a non-contiguous NCHW input; strict stable/native
separation; honest per-boundary failure atomicity; error-state recovery;
the NumPy boundary; live-storage baselines; and reality-derived
capability guardrails — **tests and documentation only, no capability and
no production behavior changed**. **Milestone F9 is complete** — the
phase closure: fresh Windows Release **and** Debug builds each passing
the full existing 10-test CTest suite with zero project warnings and the
active runtime proved to stay Release; a fresh Clang 18.1.3 ASan+UBSan
build whose instrumentation is *proved* (22 ``__asan*`` and 13
``__ubsan*`` dynamic symbols, and a library that will not load without
the sanitizer runtime); 10/10 sanitized native CTests with leak
detection enabled; 1,968 sanitized normalization-focused Python tests
with zero ASan and zero UBSan diagnostics; the F6 example and the F7
benchmark smoke path clean under the sanitized library; and a practical
LeakSanitizer lifecycle returning native live storage **exactly** to
baseline, with the remaining process-exit allocations identified
honestly as CPython/NumPy shutdown retention containing no TensorForge
frame and no suppression file — **validation and documentation only, no
numerical capability, no C++, no CTest, no ABI or ctypes surface, and no
production behavior changed**. **Phase F is complete.**

**Phase G — Native RNG and Dropout — is complete (G0-G10).** Its contract
is locked in
``docs/native_rng_dropout_design.md`` (milestone **G0**). **G1** ships
``NativeGenerator`` below — explicit, inspectable, serializable random
*state* — and makes generators a fourth ``NativeModule`` registration
category beside parameters, buffers, and child modules
(``register_generator``, ``generators()``, ``named_generators()``,
``generator_state_dict()``, ``load_generator_state_dict()``); **G1
generates no random values by itself**. **G2** ships the deterministic
**stateless** ``dropout_forward`` Core — the locked
``tensorforge.splitmix64`` derivation and an inverted-Dropout float64 CPU
kernel behind the single guarded export ``tf_core_dropout_forward``,
which touches no generator. **G3** ships the differentiable
``NativeTensor.dropout(p, *, generator)`` over that Core: a **required,
keyword-only** generator (no default, process-global, or module-global
stream, and no NumPy or Python ``random`` fallback), a graph-owned
multiplier mask whose existing ``multiply`` is the whole backward, and a
reserve/commit/abandon call transaction that consumes exactly one
generator call per **successful** stochastic forward and none on any
failure, in evaluation, at ``p == 0``, or in backward. **G4** ships the
``NativeDropout`` module exported below. **G5** moves the native
checkpoint format to **version 2**, persisting every registered
generator's state **and its alias topology** (which layers share a
stream), with version-1 archives still loadable under the locked
compatibility rule, the whole load running as one rollback-guarded
transaction, and every participating state replacement serialized under
one private shared lock. **G6** hardened the RNG, graph, ownership, and
checkpoint contracts, **G7** demonstrated the end-to-end exact stochastic
training resume (``examples/native_dropout_training.py``), **G8**
characterized the stack honestly
(``benchmarks/benchmark_native_dropout.py``), **G9** added the
cross-cutting integration suite (``tests/test_native_phase_g.py``), and
**G10** closed the phase under fresh Windows Release and Debug builds and
a Clang ASan/UBSan/LeakSanitizer matrix — only then did ``"dropout"``
leave ``UNSUPPORTED``, which read exactly ``("float32", "cuda", "amp")``
when Phase G closed. (Phase I milestone I9 has since moved ``"float32"``
out of it too, on the same discipline, so the tuple now reads ``("cuda",
"amp")`` — a later event, recorded here rather than folded into Phase G's
record.) The claim stays narrow: Dropout is supported in this
**experimental native CPU** line, never in the stable framework, and
reproducibility is exact only for the state actually captured. What the
native line still does **not** have is further activations/math, a
generic random-number API, ``Dropout2d``/``Dropout3d``, dtype expansion
beyond float32 and float64, CUDA, AMP, and data-pipeline abstractions.

**Phase H — Native CPU Performance and Runtime Efficiency — is complete
(H0-H10).** It was the latest completed native phase for the whole of
I0-I10; Phase I's closure at I11 succeeded it. Its contract is
``docs/native_cpu_performance_design.md``. Phase H made this line faster
without making it broader: every optimized kernel path ships behind the
export Python already declares, chosen by a hidden metadata-or-geometry
predicate, with the **pre-milestone traversal retained verbatim** as the
shipped generic reference path and a rejected predicate always a fallback
rather than an error. Measured against a reconstructed H0 baseline with
every result proved bit-identical first, **every shipped training
workload is 1.50x-3.89x faster than it was at H0**, and no allocation
count or memory peak rose anywhere. Across the whole phase **no
capability, dtype, device, registry value, public API, checkpoint field,
or checkpoint version moved**, and exactly one C ABI symbol was added
(``tf_storage_create_uninitialized``, at H1), taking the library from 51
to **52** exported ``tf_*`` symbols. **H10** re-measured the phase,
resolved the acceleration gate as three documented rejections with
measurements (SIMD, threading/OpenMP, and BLAS — the first because
elementwise, matmul, and reduction are already auto-vectorized, the last
because a BLAS matmul is not bit-identical and would break every
exact-resume proof), assessed ``tf_core_narrow_backward`` and the
small-operation boundary floor and implemented neither, and ran the full
Release/Debug/Linux/sanitizer/lifecycle matrix. What Phase H did **not**
add: SIMD, threading, OpenMP, BLAS, a memory pool, a scratch workspace,
general fusion, im2col, or any public performance control — no path
selector, threshold setter, dispatch tracer, profiling counter, or
environment variable.

**Phase I — Native Dtype Generalization and Float32 CPU Support — is
complete (I0-I11), and it is the latest completed native phase.** Its
contract is ``docs/native_dtype_float32_design.md``. Phase I made this
line *dtype-general* without making it wider in any other direction:
native storage is dtype-tagged and is the single authority for every view
of it, each handle-based export dispatches **once** from that tag into
templated ``float``/``double`` kernels, and the whole stack — transfer,
elementwise, reductions, matmul, Conv2d, MaxPool2d, softmax/log-softmax,
fused cross-entropy, normalization, Dropout, modules, parameters,
buffers, both optimizers, and the checkpoint — runs at either width.
**Since milestone I9 both dtypes are publicly supported**:
``SUPPORTED_DTYPES == ("float64", "float32")`` and ``UNSUPPORTED ==
("cuda", "amp")``, with **float64 still the default** at every
constructor, factory, module, and parameter, and still what ``None``
means. The registry moved only *after* the integrated exact-resume proof
passed at each dtype independently, compared in raw IEEE-754 bit
patterns and never across dtypes.

What Phase I did **not** add: casting, promotion, mixed-dtype arithmetic,
dtype inference from an input array, a global default dtype, ``astype`` /
``.float()`` / ``.double()`` / ``map_location``, device movement, a
device, an integer or boolean tensor dtype, float16, bfloat16, AMP, or
CUDA. A mixed-dtype operation raises **before** any allocation or
mutation. ``RAW_KERNEL_DTYPES == ("float64",)`` is a **separate and
permanently narrower** statement about the seven handle-free raw utility
kernels, which take only ``double*`` and an element count and so have no
dtype to dispatch on — it is never the overall support claim. The
MaxPool2d winner buffer stays private float64 metadata at every value
width, and cross-entropy targets stay host ``int64`` metadata. Exactly
**two** C ABI symbols were added across the phase
(``tf_storage_create_typed`` and
``tf_storage_create_uninitialized_typed``, at I1), taking the library from
52 to **54**; the native checkpoint moved to **version 3** at I8 with
``(1, 2, 3)`` accepted and versions 1 and 2 float64-only permanently; and
the in-memory optimizer state schema did not move from version **1**.
**I11** closed the phase with cross-platform revalidation, the closure
guardrails in ``tests/test_native_phase_i_closure.py``, and the final
inventory reconciliation — adding no capability at all.

**Phase J — Deterministic Native Data Pipeline and Mini-Batching — is
the latest phase and is in progress: milestones J0, J1, and J2 have
landed, and J3 through J9 have not started.** Its contract is
``docs/native_data_pipeline_design.md`` (milestone **J0**: architecture,
contract, and documentation only, adding no runtime behavior).
**Milestone J1** adds ``NativeTensorDataset`` below — the finite,
host-backed native dataset, and the phase's first runtime. It holds one
owned host snapshot of the features and one of the class targets, at an
**explicitly chosen** native feature dtype (``None`` still means
``"float64"``, and the NumPy feature dtype never selects it), and turns
any index sequence into a fresh owning ``NativeTensor`` feature batch —
which **the caller closes** — beside a fresh read-only host ``int64``
target batch. Both snapshots are unconditional copies, so caller mutation
after construction reaches nothing; a SHA-256 content ``fingerprint``
over a canonical little-endian byte stream gives the dataset a
deterministic, cross-platform ``identity()`` that carries no payload; and
the dataset owns **no native storage between calls**, so holding one
leaves the native live-storage count untouched. The dataset plans,
orders, and groups nothing: it answers only "given these indices, what is
the batch?".

**Milestone J2** adds ``NativeBatchSampler`` below — the deterministic
order and batch **planner**, and the phase's second runtime. It owns
``batch_size``, ``drop_last``, ``shuffle``, the ``seed``, the ``epoch``,
and the ``cursor``, and turns them into batch-index groups through
``epoch_permutation()``, ``plan()``, and ``next_batch_indices()``. Every
permutation is a **pure function** of ``(seed, epoch, length)``: it
reuses the locked ``tensorforge.splitmix64`` finalizer and golden
constant under one domain-separated epoch key schedule — **no new RNG
algorithm, no new global or default generator, and no coupling to a live
``NativeGenerator``** — with unbiased rejection-based bounded integers
and a downward Fisher-Yates sweep, in explicit ``& (2**64 - 1)`` Python
integer arithmetic that is bit-identical on every platform by
construction. So it holds no consumable stream and nothing to roll back:
inspection and planning consume nothing and may be repeated in any order.
Its compact JSON-compatible ``state_dict()`` carries the configuration,
the position, and the dataset's four identity fields — no permutation and
no payload — and ``load_state_dict()`` is transactional: everything is
validated (dataset identity against **live** reality, configuration
adopted from the state) before six assignments that cannot fail. It
allocates nothing native, materializes no batch, and owns nothing
releasable, so it has **no ``close()``** and works unchanged against a
closed dataset. The private derivation lives in ``_native_permutation``
and stays private: it is not exported and is not a public random surface.
Neither milestone adds a kernel, C ABI symbol, ctypes declaration,
checkpoint field or version, optimizer-state version, capability registry
value, or dependency.

What Phase J does **not** yet have, because those milestones have not
started: ``NativeDataLoader`` (J3), any iteration, any batch delivery,
any successful-delivery cursor advancement, any native mini-batching, any
loader state, and any checkpoint loader-state integration. **J3 is
next.** The sampler plans; nothing yet iterates or materializes a batch.

``NativeGenerator`` (Phase G, milestone G1) is the Python half of the
phase's central split — random state is Python-managed, and the native
random kernels (milestone G2) are stateless and receive the whole
key for one call. It is a **pure-Python value holder**: an algorithm
identifier (``"tensorforge.splitmix64"``) and version, an unsigned 64-bit
seed, and a counter of **committed** stochastic calls, all exposed as
read-only properties with ``state()`` / ``load_state()`` / ``reseed()`` /
``reset()`` for atomic, identity-preserving state changes. It owns no
native storage, allocates nothing native, and has **no ``close()``** —
constructing, registering, and dropping generators leaves the native
live-storage count untouched. Identity is object identity (no value
equality), sharing is done by sharing the object, and copying — ``copy``,
``deepcopy``, and pickle alike — is refused, because a copied generator
would silently produce the same values in two places. ``seed=None`` draws
one 64-bit seed from OS entropy through ``secrets``; nothing in the phase
consults the clock, the process id, an address, NumPy's global RNG, or
Python's ``random``. Its private call transaction (``_reserve_call`` →
``_commit_call``/``_abandon_call``) is lock-protected and
token-validated so that the counter advances exactly once per published
call and no two callers can ever receive the same call index — which is
serialization for correctness, not parallel stochastic execution, which
Phase G does not claim.

``NativeParameter`` and ``NativeParameterRegistry`` (Advanced C++ v3.1,
the first Phase C step) add the native training stack's trainable-leaf
abstraction and the minimal parameter-registration contract, and
``NativeModule`` (Advanced C++ v3.2) is the module-hierarchy core built
on them: automatic parameter/child registration through attribute
assignment, deterministic identity-deduplicated recursive traversal,
recursive ``zero_grad()``, and ``train()``/``eval()`` state propagation
— plus the in-memory state dictionary contract (Advanced C++ v3.3):
``state_dict()`` snapshots and atomic identity-preserving
``load_state_dict()``. That contract began as parameters-only and, since
the v3.15 buffer support (``register_buffer``/``buffers()``), covers
**parameters and persistent buffers** — non-persistent buffers are never
serialized. ``NativeLinear`` (Advanced C++
v3.4) is the first concrete native layer: a fully connected
``y = x @ weight (+ bias)`` on NativeModule/NativeParameter with
deterministic seeded initialization, strictly 2-D input semantics, and
backward supplied entirely by the existing native autograd.
``NativeReLU`` and ``NativeSequential`` (Advanced C++ v3.5) complete
the first composable model surface: a parameter-free activation module
over the existing ``relu()`` autograd, and an ordered container with
contiguous integer-string execution slots, position-based execution,
and identity-deduplicated traversal/state. ``NativeFlatten`` (Phase D,
milestone D1) is a parameter-free, buffer-free batch-preserving flatten
Python-composed from the existing ``reshape``/``contiguous_copy``
operations and their autograd (no new kernel, no custom backward); it
returns an independent owning ``(N, features)`` tensor so it composes
safely in a ``NativeSequential``. ``NativeMSELoss`` (Advanced
C++ v3.6) is the first native loss: a parameter-free scalar
mean/sum-reduced MSE composed from existing native operations, its
gradients supplied entirely by the existing autograd. Parameter
mutation is safe as of Advanced C++ v3.7: every ``NativeParameter``
carries a read-only monotonic value ``version``, ``copy_value_`` is
the one controlled no-grad mutation primitive (the future NativeSGD
commit path), ``load_state_dict`` increments each loaded parameter's
version atomically, and ``backward()`` raises a deterministic
stale-graph error when a parameter whose forward value backward must
read (multiply/matmul/relu edges) was mutated after forward.
``NativeSGD`` (Advanced C++ v3.8) is the first native optimizer:
minimal stochastic gradient descent over identity-deduplicated
NativeParameter objects — graph-free native update staging committed
through ``copy_value_``, frozen and gradient-less parameters skipped,
gradients retained until ``zero_grad()`` — with no momentum, weight
decay, parameter groups, optimizer state, or schedulers.
``NativeAdam`` (Advanced C++ v3.12) is the native adaptive optimizer:
persistent optimizer-owned native first/second-moment buffers,
per-parameter step counters, bias correction via the v3.11
``sqrt``/``reciprocal`` primitives (no division), graph-free staged
updates committed through ``copy_value_``, validated
``lr``/``betas``/``eps``, and an explicit state lifetime
(``close()``) — with no weight decay, AMSGrad, parameter groups,
schedulers, or checkpointing. As of Advanced C++ v3.13 both native
optimizers carry the in-memory **optimizer state contract**:
``state_dict()``/``load_state_dict()`` over one versioned schema
(format 1, exact optimizer type tag, ordered positional
shape/dtype/device parameter metadata — no ids, names, values, or
gradients), with caller-owned independent NativeTensor moment
snapshots and per-parameter step counts for NativeAdam, exact
validation, staged atomic loading that never touches parameter
values/versions/gradients, and proven deterministic in-memory training
continuation. ``save_native_checkpoint``/``load_native_checkpoint``
(Advanced C++ v3.14, extended by Phase G milestone G5) persist a
NativeModule plus optionally one native optimizer's state, every
registered generator's state and alias topology, and JSON-compatible
metadata to one explicit, pickle-free NPZ archive (format
``"tensorforge.native_checkpoint"``, now **version 2**; version-1
archives still load under the locked compatibility rule) — strict
validation before any mutation, atomic
temporary-file replacement, strict optimizer presence/type matching,
deterministic file resume, and ``allow_pickle=False`` loading — fully
separate from the stable ``tensorforge.serialization`` (no scheduler
state, data-loader position, Python ``random``, or NumPy global-RNG
capture, and no ``map_location``). Still fully separate
from ``tensorforge.nn`` and ``tensorforge.optim``.

``NativeCrossEntropyLoss`` (Phase E, milestone E7) is the native
classification loss: a parameter-free, buffer-free ``NativeModule``
whose forward is exactly
``logits.cross_entropy(targets, reduction=self.reduction)``. It adds no
kernel, ABI symbol, arithmetic, or target validation of its own, so it
inherits every E5/E6 guarantee unchanged — strict copied ``int64``
targets, the fused stable forward, a scalar output, graph-owned saved
probabilities, no logits reread, no expected version snapshot, and full
failure atomicity. Its ``"mean"``/``"sum"`` reduction is validated in the
constructor by the operation's own validator and is **constructor
configuration, not model state**: it contributes no ``state_dict()``
entries and no checkpoint keys (E7 changed no checkpoint schema).
``native_accuracy(logits, targets) -> float`` (also E7) is a
**reporting-only** helper, not native C++ compute and not an autograd
operation: it validates rank-2 logits and targets under the same strict
contract, materializes the logits **once** through the explicit public
``to_numpy()`` boundary, takes ``numpy.argmax(axis=1)`` (first-maximal
index on ties), and returns a plain ``float`` in ``[0.0, 1.0]`` — while
building no graph, touching no gradient, parameter, or version, and
retaining nothing.

Constructors need the experimental C++ backend to be built; importing
this package is always safe (the library loads lazily on first use).
"""

from .native_tensor import NativeTensor
from .native_generator import NativeGenerator
from .native_parameter import NativeParameter, NativeParameterRegistry
from .native_module import NativeModule
from .native_linear import NativeLinear
from .native_relu import NativeReLU
from .native_flatten import NativeFlatten
from .native_conv2d import NativeConv2d
from .native_maxpool2d import NativeMaxPool2d
from .native_sequential import NativeSequential
from .native_layernorm import NativeLayerNorm
from .native_batchnorm import NativeBatchNorm1d, NativeBatchNorm2d
from .native_dropout import NativeDropout
from .native_mse_loss import NativeMSELoss
from .native_cross_entropy_loss import NativeCrossEntropyLoss
from .native_metrics import native_accuracy
from .native_sgd import NativeSGD
from .native_adam import NativeAdam
from .native_checkpoint import load_native_checkpoint, save_native_checkpoint
from .native_dataset import NativeTensorDataset
from .native_sampler import NativeBatchSampler
from .native_data_loader import NativeDataLoader

__all__ = [
    "NativeTensor",
    "NativeGenerator",
    "NativeParameter",
    "NativeParameterRegistry",
    "NativeModule",
    "NativeLinear",
    "NativeReLU",
    "NativeFlatten",
    "NativeConv2d",
    "NativeMaxPool2d",
    "NativeSequential",
    "NativeLayerNorm",
    "NativeBatchNorm1d",
    "NativeBatchNorm2d",
    "NativeDropout",
    "NativeMSELoss",
    "NativeCrossEntropyLoss",
    "native_accuracy",
    "NativeSGD",
    "NativeAdam",
    "save_native_checkpoint",
    "load_native_checkpoint",
    "NativeTensorDataset",
    "NativeBatchSampler",
    "NativeDataLoader",
]
