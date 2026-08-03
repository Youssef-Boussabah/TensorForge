"""NativeMaxPool2d — the native max-pooling layer (Advanced C++ Phase D,
milestone D10; see docs/native_cnn_design.md §3.3 and §18).

A **parameter-free, buffer-free** ``NativeModule`` whose forward is the
existing differentiable ``NativeTensor.maxpool2d`` primitive (D8 forward +
private winner buffer, D9 backward scatter + autograd) — nothing numerical
or autograd-related is duplicated here::

    output = input.maxpool2d(
        kernel_size=self.kernel_size,
        stride=self.stride,
        padding=self.padding,
    )

Every pooling contract therefore belongs to the operation, not to this
module: the floor output shape, window traversal order, the strict-``>``
first-occurrence tie rule, padding participating as a conceptual ``-inf``
(and its ``-1`` winner sentinel), the documented out-of-contract NaN
behavior, the scatter-add backward through the saved winners, overlapping
window accumulation, the winner buffer's graph-bound lifetime, and the
deliberate absence of a version snapshot. ``NativeMaxPool2d`` adds **no**
C++ kernel, C ABI symbol, ctypes declaration, autograd primitive, custom
backward callback, checkpoint schema, dtype, or dispatch mechanism.

**The module is stateless between calls.** Pooling has no learnable
values, so there are no parameters and no buffers: ``parameters()``,
``named_parameters()``, ``buffers()``, ``named_buffers()``, and
``state_dict()`` are all empty, and the layer contributes **no keys** to a
parent module's state dictionary or to a checkpoint. In particular the
**private saved-winner storage is never held by the module** — each
forward's winners belong to that call's output graph (released when its
history is), so repeated forwards produce independent graphs and
independent winner resources, and a stored module never pins native
memory. Architecture (kernel/stride/padding) lives in the constructor, not
in serialized state — exactly like the stable ``tensorforge.nn.MaxPool2d``.

**Constructor** ``NativeMaxPool2d(kernel_size, stride=None, padding=0)``.
``kernel_size``/``stride`` (each ≥ 1) and ``padding`` (≥ 0) are an int or a
2-element ``(height, width)`` pair, normalized to and stored as two-element
tuples through the native ``_spatial_pair`` helper (bools, malformed pair
lengths, and non-integer members rejected). ``stride=None`` means
``stride = kernel_size`` — the non-overlapping-window convention the stable
MaxPool2d uses. There is no ``dilation``, ``ceil_mode``,
``return_indices``, adaptive/average pooling, ``device``, ``dtype``,
``requires_grad``, or ``seed`` argument in Phase D. Validation runs in the
constructor and allocates nothing: the module owns no native storage at
all, so a rejected argument cannot leak any.

**Input contract:** ``forward(input)`` requires an **open** 4-D NCHW
``NativeTensor`` (a ``NativeParameter`` is accepted as the subclass it is;
the result is an ordinary ``NativeTensor`` either way) on the native CPU
float64 line. The stable framework's ``Tensor``, NumPy arrays, lists,
scalars, closed tensors, and wrong-rank inputs are rejected with clear
errors before the operation runs. Non-contiguous inputs ride the existing
Policy-B copy path. The output is the fresh **owning** ``(N, C, out_h,
out_w)`` tensor the operation produced, requiring grad exactly when the
input does. The inherited ``training`` flag exists and propagates normally
but never affects pooling numerics.

Fully separate from ``tensorforge.nn.MaxPool2d``; CPU only, at
either supported dtype (the value path follows the input's dtype
since Phase I milestone I5, while the private winner buffer stays
float64 index metadata at every value dtype);
experimental and explicit. The deterministic end-to-end native CNN
training + checkpoint-resume proof (D11) is not part of this milestone.
"""

from ..backends.cpp import _spatial_pair
from .native_module import NativeModule
from .native_tensor import NativeTensor


class NativeMaxPool2d(NativeModule):
    """Downsample NCHW activations by window maxima: ``output =
    input.maxpool2d(kernel_size, stride, padding)`` over the existing
    D8/D9 autograd primitive.

    ``NativeMaxPool2d(kernel_size, stride=None, padding=0)`` — parameter-
    free, buffer-free, and stateless between calls; see the module
    docstring for the full contract (normalized spatial attributes,
    delegation, empty state, and inherited autograd).
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        # Validate and normalize every argument before touching module
        # state. These are the same semantics NativeTensor.maxpool2d
        # applies (int or 2-pair, bools/malformed forms rejected), so the
        # module can never accept a configuration the operation would
        # refuse. `stride=None` means non-overlapping windows — the stable
        # MaxPool2d convention, resolved here so the stored attribute is
        # always a concrete pair.
        kernel_size = _spatial_pair(kernel_size, "kernel_size", minimum=1)
        if stride is None:
            stride = kernel_size
        else:
            stride = _spatial_pair(stride, "stride", minimum=1)
        padding = _spatial_pair(padding, "padding", minimum=0)
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        """``input.maxpool2d(kernel_size, stride, padding)`` over the
        existing D8/D9 autograd primitive — a 4-D NCHW ``(N, C, H, W)``
        input, an ``(N, C, out_h, out_w)`` output. Backward, the saved
        winners, and their lifetime come entirely from that operation; the
        module keeps nothing."""
        if not isinstance(input, NativeTensor):
            raise TypeError(
                f"NativeMaxPool2d.forward requires a NativeTensor input, got "
                f"{type(input).__name__}"
            )
        if input.closed:
            raise RuntimeError(
                "NativeMaxPool2d.forward: the input tensor has been closed"
            )
        # A clearer module-specific rank error before the operation runs
        # (maxpool2d re-validates rank, dtype/device, the argument forms,
        # and the output shape itself).
        if input.ndim != 4:
            raise ValueError(
                f"NativeMaxPool2d expects 4-D NCHW input (batch, channels, "
                f"height, width), got shape {input.shape}"
            )
        return input.maxpool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def __repr__(self):
        return (
            f"NativeMaxPool2d(kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding})"
        )
