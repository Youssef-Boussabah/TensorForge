"""Experimental, opt-in APIs built on the native C++ backend.

Nothing here is part of the finished Python framework, and
``import tensorforge`` never imports this package — you reach it
explicitly:

    from tensorforge.experimental import NativeTensor

``NativeTensor`` is a native tensor wrapper over the native runtime
(NativeTensorCore) with an opt-in, Python-managed reverse-mode autograd
graph (Phase B, complete as of Advanced C++ v2.6). It is **not**
tensorforge.Tensor: the two autograd engines never mix, no conversion is
implicit, and it has no optimizer/training integration and no CUDA.

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
and identity-deduplicated traversal/state. ``NativeMSELoss`` (Advanced
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
schedulers, or optimizer ``state_dict``/checkpointing (optimizer-state
serialization is v3.13). No file serialization, and still fully
separate from ``tensorforge.nn`` and ``tensorforge.optim``.

Constructors need the experimental C++ backend to be built; importing
this package is always safe (the library loads lazily on first use).
"""

from .native_tensor import NativeTensor
from .native_parameter import NativeParameter, NativeParameterRegistry
from .native_module import NativeModule
from .native_linear import NativeLinear
from .native_relu import NativeReLU
from .native_sequential import NativeSequential
from .native_mse_loss import NativeMSELoss
from .native_sgd import NativeSGD
from .native_adam import NativeAdam

__all__ = [
    "NativeTensor",
    "NativeParameter",
    "NativeParameterRegistry",
    "NativeModule",
    "NativeLinear",
    "NativeReLU",
    "NativeSequential",
    "NativeMSELoss",
    "NativeSGD",
    "NativeAdam",
]
