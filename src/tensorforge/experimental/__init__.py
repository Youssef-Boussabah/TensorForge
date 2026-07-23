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
(contracted for Phase E in docs/native_classification_design.md) is now
largely in place: milestones E1-E4 shipped the differentiable ``exp``,
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
timing threshold anywhere. What the native line
still does **not** have: phase closure
and sanitizer validation (E10), further
activations/math, normalization (BatchNorm/LayerNorm), dropout or a
native RNG, float32/dtype expansion, CUDA, AMP, and data-pipeline
abstractions.

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
    "NativeMSELoss",
    "NativeCrossEntropyLoss",
    "native_accuracy",
    "NativeSGD",
    "NativeAdam",
    "save_native_checkpoint",
    "load_native_checkpoint",
]
