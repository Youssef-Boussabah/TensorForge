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

**Phase F — Native Normalization and Stateful Buffers — is the current
phase, and it is in progress.** Its architecture contract is locked in
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
which now reads exactly ``("dropout", "float32", "cuda", "amp")``.
**That completes the numerical normalization module surface. Milestone
F5 is complete** — the exhaustive state, checkpoint, ownership, and
graph-safety hardening (a focused ``tests/test_native_normalization_state.py``
plus narrow additions to the generic buffer and checkpoint suites),
proving §7-§10 of the design by executable test rather than by prose:
**tests and documentation only, no numerical behavior and no new public
capability**, with the exports, every capability registry, and the
version-1 checkpoint format all exactly what F4 left. **Phase F is not
finished: milestones F6-F9 have not started**, so there is no
deterministic normalized training example with exact resume, no
normalization benchmark, no cross-cutting Phase-F integration, and no
phase closure. What the native line still does **not** have: further
activations/math, dropout or a native RNG, float32/dtype expansion, CUDA,
AMP, and data-pipeline abstractions.

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
(Advanced C++ v3.14) persist a NativeModule plus optionally one native
optimizer's state and JSON-compatible metadata to one explicit,
pickle-free NPZ archive (format ``"tensorforge.native_checkpoint"``,
version 1) — strict validation before any mutation, atomic
temporary-file replacement, strict optimizer presence/type matching,
deterministic file resume, and ``allow_pickle=False`` loading — fully
separate from the stable ``tensorforge.serialization`` (no scheduler
or random-state capture, no ``map_location``). Still fully separate
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
entries and no checkpoint keys (format version 1 is unchanged).
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
from .native_mse_loss import NativeMSELoss
from .native_cross_entropy_loss import NativeCrossEntropyLoss
from .native_metrics import native_accuracy
from .native_sgd import NativeSGD
from .native_adam import NativeAdam
from .native_checkpoint import load_native_checkpoint, save_native_checkpoint

__all__ = [
    "NativeTensor",
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
    "NativeMSELoss",
    "NativeCrossEntropyLoss",
    "native_accuracy",
    "NativeSGD",
    "NativeAdam",
    "save_native_checkpoint",
    "load_native_checkpoint",
]
