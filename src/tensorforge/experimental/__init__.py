"""Experimental, opt-in APIs built on the native C++ backend.

Nothing here is part of the finished Python framework, and
``import tensorforge`` never imports this package — you reach it
explicitly:

    from tensorforge.experimental import NativeTensor

``NativeTensor`` is a forward-only wrapper over the native runtime
(NativeTensorCore). It is not tensorforge.Tensor: no autograd, no
optimizer/Module integration, no CUDA. Its constructors need the
experimental C++ backend to be built; importing this package is always
safe (the library loads lazily on first use).
"""

from .native_tensor import NativeTensor

__all__ = ["NativeTensor"]
