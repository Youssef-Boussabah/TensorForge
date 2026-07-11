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
abstraction and the minimal parameter-registration contract the future
``NativeModule`` will build on — no module hierarchy, optimizer, or
training loop yet, and still fully separate from
``tensorforge.nn.Parameter``.

Constructors need the experimental C++ backend to be built; importing
this package is always safe (the library loads lazily on first use).
"""

from .native_tensor import NativeTensor
from .native_parameter import NativeParameter, NativeParameterRegistry

__all__ = ["NativeTensor", "NativeParameter", "NativeParameterRegistry"]
