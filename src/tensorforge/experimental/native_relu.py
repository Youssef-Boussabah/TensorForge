"""NativeReLU — the first native activation module (Advanced C++ v3.5;
see docs/backend_experiments.md).

A **parameter-free** ``NativeModule`` wrapping the existing
``NativeTensor.relu()``: ``forward(input)`` validates its input and
delegates — nothing else. It holds no parameters, no buffers, and no
native storage; its ``state_dict()`` is empty; and its backward is
entirely the existing Phase B relu autograd (the fused native
``relu_backward``: upstream passes where the forward input was ``> 0``
and is blocked elsewhere, **exactly zero included** — that existing
contract is tested here, not changed). No custom backward callback, no
new kernel, no in-place mode or ``inplace`` argument, and no NumPy in
forward or backward.

The input contract is shape-generic: every rank and layout
``NativeTensor.relu()`` supports (strided/offset views included) works
unchanged — nothing is reshaped, flattened, cast, copied, or moved.
``forward`` requires an open ``NativeTensor`` (a ``NativeParameter`` is
accepted as the subclass it is; the result is an ordinary
``NativeTensor`` either way) and rejects the stable framework's
``Tensor``, NumPy arrays, lists, scalars, and closed tensors with clear
errors. The inherited ``training`` flag exists and propagates normally,
but ReLU behaves identically in train and eval modes.

Fully separate from ``tensorforge.nn.ReLU``; CPU only, at either
supported dtype (it takes no dtype argument and inherits its
input's);
experimental and explicit.
"""

from .native_module import NativeModule
from .native_tensor import NativeTensor


class NativeReLU(NativeModule):
    """``max(x, 0)`` as a module: ``forward(input)`` is
    ``input.relu()`` over the existing native operation and its existing
    autograd. Parameter-free and shape-generic — see the module
    docstring for the full contract."""

    def __init__(self):
        super().__init__()

    def forward(self, input):
        if not isinstance(input, NativeTensor):
            raise TypeError(
                f"NativeReLU.forward requires a NativeTensor input, got "
                f"{type(input).__name__}"
            )
        if input.closed:
            raise RuntimeError(
                "NativeReLU.forward: the input tensor has been closed"
            )
        return input.relu()

    def __repr__(self):
        return "NativeReLU()"
