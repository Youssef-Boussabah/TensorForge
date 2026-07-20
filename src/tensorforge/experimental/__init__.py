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
CNN stack (Phase D) has begun with ``NativeFlatten`` (milestone D1) and,
as of milestone D6, the differentiable **``NativeTensor.conv2d``** operation
(NCHW/OIHW cross-correlation with int/tuple stride and padding and optional
bias; input, weight, and bias gradients through native backward kernels and
the existing ``sum`` reduction), and as of milestone D7 the trainable
**``NativeConv2d``** module built on it (OIHW weight / optional ``(O,)``
bias ``NativeParameter``s, deterministic uniform conv fan-in initialization,
4-D NCHW input validation, and backward supplied entirely by the D6
autograd — no new kernel, ABI symbol, or custom module backward).
Milestone D8 added max-pooling only at the **runtime** layer — the
forward-only ``NativeTensorCore.maxpool2d_forward`` and its private saved
winner buffer — so nothing pooling-related is exported from this package
yet. Differentiable pooling (``NativeTensor.maxpool2d`` and its backward,
D9), the ``NativeMaxPool2d`` module (D10), the deterministic end-to-end
native CNN training/checkpoint-resume proof (D11), CUDA, and the remaining
Phase-D numerical layers (new activations, softmax/classification losses,
BatchNorm/LayerNorm/Dropout) remain future work.

``NativeParameter`` and ``NativeParameterRegistry`` (Advanced C++ v3.1,
the first Phase C step) add the native training stack's trainable-leaf
abstraction and the minimal parameter-registration contract, and
``NativeModule`` (Advanced C++ v3.2) is the module-hierarchy core built
on them: automatic parameter/child registration through attribute
assignment, deterministic identity-deduplicated recursive traversal,
recursive ``zero_grad()``, and ``train()``/``eval()`` state propagation
— plus the in-memory state dictionary contract (Advanced C++ v3.3):
``state_dict()`` snapshots and atomic identity-preserving
``load_state_dict()``, parameters only. ``NativeLinear`` (Advanced C++
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
from .native_sequential import NativeSequential
from .native_mse_loss import NativeMSELoss
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
    "NativeSequential",
    "NativeMSELoss",
    "NativeSGD",
    "NativeAdam",
    "save_native_checkpoint",
    "load_native_checkpoint",
]
