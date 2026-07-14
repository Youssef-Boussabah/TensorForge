"""NativeFlatten — batch-preserving flatten (Advanced C++ Phase D,
milestone D1; see docs/native_cnn_design.md §3.1 and §18).

A **parameter-free, buffer-free** ``NativeModule`` that collapses every
non-batch dimension of an ``(N, d1, d2, ...)`` native tensor into a
single feature axis, giving ``(N, d1*d2*...)`` — the usual bridge between
image-shaped layers and a ``NativeLinear``. It is the native counterpart
of ``tensorforge.nn.Flatten`` and, like it, is **batch-preserving only**:
the first dimension is never flattened, and there are no ``start_dim`` /
``end_dim`` arguments in Phase D.

``NativeFlatten`` is **Python-composed from existing native operations**
— it introduces **no new C++ kernel, C ABI symbol, ctypes declaration,
autograd primitive, checkpoint schema, dtype, or dispatch mechanism**.
Its forward is exactly::

    source = input if input.contiguous else input.contiguous_copy()
    flat   = source.reshape((batch, features))
    return flat.contiguous_copy()

so its backward is **entirely** the existing ``reshape`` (and, on the
non-contiguous path, ``contiguous_copy``) autograd — the inverse reshape
of the upstream gradient, routed back to the original source tensor. No
custom backward callback is defined, and no versioned-value read is
introduced (``reshape``/``contiguous_copy`` backward rely only on saved
metadata under the existing contracts).

**Why the result always owns its storage.** ``NativeTensor.reshape``
returns a *borrowing view* over its source, which is valid only while
that source's storage stays open (the existing reshape lifetime
contract). A module output must not depend on its input surviving: inside
a ``NativeSequential`` each layer's input is a transient that is dropped
as the loop rebinds, so a bare reshape *view* would dangle (its storage
freed) the moment the next layer runs — this is a real, reproducible
lifetime hazard in the no-grad/eval path. ``NativeFlatten`` therefore
materializes the reshaped view into an **independent owning** result with
a final ``contiguous_copy()``, exactly as every other native module
(``NativeReLU``, ``NativeLinear``) returns fresh owning storage. This is a
D1 implementation refinement of the D0 §3.1 "view when contiguous"
sketch: the reshape *view* is still used internally (no data is copied
except the one materialization), but the module's *output* owns its
storage so it composes safely in both training and eval. The batch
dimension is preserved throughout, including any layout the runtime can
represent; the runtime forbids zero-size dimensions everywhere, so a
zero-batch or zero-feature input cannot be constructed to reach here in
the first place.

The input contract mirrors the other native modules: ``forward(input)``
requires an **open** ``NativeTensor`` (a ``NativeParameter`` is accepted
as the subclass it is; the result is an ordinary ``NativeTensor`` either
way) of **rank ≥ 2**, and rejects rank-0 / rank-1 inputs, the stable
framework's ``Tensor``, NumPy arrays, lists, scalars, and closed tensors
with clear errors. Nothing is wrapped, cast, or moved implicitly; the
output stays in the native CPU float64 line. The inherited ``training``
flag exists and propagates normally but never affects the result.

Fully separate from ``tensorforge.nn.Flatten``; float64/cpu only;
experimental and explicit. Conv2d and MaxPool2d remain unimplemented.
"""

from math import prod

from .native_module import NativeModule
from .native_tensor import NativeTensor


class NativeFlatten(NativeModule):
    """Collapse every non-batch dimension: ``(N, ...) -> (N, features)``.

    Parameter-free and buffer-free. ``forward(input)`` validates an open
    ``NativeTensor`` of rank ≥ 2 and returns an independent owning
    ``(N, features)`` tensor built from the existing ``reshape`` /
    ``contiguous_copy`` operations — see the module docstring for the full
    contract (batch preservation, view/copy behavior, and inherited
    autograd).
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        if not isinstance(input, NativeTensor):
            raise TypeError(
                f"NativeFlatten.forward requires a NativeTensor input, got "
                f"{type(input).__name__}"
            )
        if input.closed:
            raise RuntimeError(
                "NativeFlatten.forward: the input tensor has been closed"
            )
        if input.ndim < 2:
            raise ValueError(
                f"NativeFlatten expects at least 2-D input (batch, ...), got "
                f"shape {input.shape}"
            )
        batch = input.shape[0]
        # Safe Python integer arithmetic over the (positive-int) trailing
        # dimensions; prod(()) == 1 never arises here because rank >= 2.
        features = prod(input.shape[1:])
        # reshape requires a contiguous source, so a non-contiguous input
        # is first materialized through the existing contiguous_copy op;
        # the reshape itself is a metadata-only view. The final
        # contiguous_copy makes the result an independent owning tensor so
        # NativeFlatten composes safely (its lifetime never depends on the
        # input surviving) — the backward for every step is the existing
        # native autograd, with no new callback and no versioned read.
        source = input if input.contiguous else input.contiguous_copy()
        flat = source.reshape((batch, features))
        return flat.contiguous_copy()

    def __repr__(self):
        return "NativeFlatten()"
