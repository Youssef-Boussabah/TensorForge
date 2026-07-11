"""Experimental, opt-in APIs built on the native C++ backend.

Nothing here is part of the finished Python framework, and
``import tensorforge`` never imports this package — you reach it
explicitly:

    from tensorforge.experimental import NativeTensor

``NativeTensor`` is a native tensor wrapper over the native runtime
(NativeTensorCore) with an opt-in, Python-managed reverse-mode autograd
graph (Phase B, complete as of Advanced C++ v2.6). It is **not**
tensorforge.Tensor: the two autograd engines never mix, no conversion is
implicit, and it has no optimizer/Module/training integration and no CUDA.
Its constructors need the experimental C++ backend to be built; importing
this package is always safe (the library loads lazily on first use).
"""

from .native_tensor import NativeTensor

__all__ = ["NativeTensor"]
