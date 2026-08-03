"""NativeConv2d — the trainable native convolution layer (Advanced C++
Phase D, milestone D7; see docs/native_cnn_design.md §9).

A 2-D convolution (cross-correlation) as a ``NativeModule`` holding a
``NativeParameter`` weight (and optionally a ``NativeParameter`` bias)
whose forward is the existing differentiable ``NativeTensor.conv2d``
primitive (D6) — nothing numerical or autograd-related is duplicated
here::

    output = input.conv2d(weight, bias, stride=stride, padding=padding)

**Parameter layout** (load-bearing for checkpoints, matching the stable
framework and the D0 contract): ``weight`` is ``(out_channels,
in_channels, kernel_height, kernel_width)`` (OIHW); ``bias`` is
``(out_channels,)`` when ``bias=True``, otherwise the attribute reads as
``None`` (the ``NativeLinear`` pattern — nothing registered under
"bias"). Registration order is ``weight`` then ``bias``, so
``parameters()`` / ``named_parameters()`` / ``state_dict()`` are
deterministic (``["weight", "bias"]``; nested as ``"0.weight"`` /
``"0.bias"`` in a ``NativeSequential``).

Because forward is the existing differentiable operation, the existing
autograd engine *is* the backward implementation — there is **no** manual
or fused NativeConv2d backward, no new kernel, no new C ABI symbol, and no
new versioned-value read. Gradients land where D6 defined them:
``input.grad`` is ``(N, C, H, W)``, ``weight.grad`` is ``(O, C, kh, kw)``,
and ``bias.grad`` is ``(O,)`` (via the existing ``sum`` reduction). Graph
lifetime, ``retain_graph``, one-shot cleanup, conditional stale-value
version tracking, and the failure/rollback guarantees are all inherited
unchanged from D6.

**Initialization** is deterministic and self-contained, following the
native line's uniform fan-in convention (as ``NativeLinear`` chose over
the stable Gaussian) with the **conv fan-in** the stable Conv2d uses:
``fan_in = in_channels * kernel_height * kernel_width``, ``bound =
1/sqrt(fan_in)``, sampled from ``[-bound, +bound]`` with a **local**
``numpy.random.default_rng(seed)``. An int ``seed`` reproduces the exact
values, ``seed=None`` draws fresh entropy, and the global NumPy RNG is
never read or mutated. NumPy appears only here as host-side data
preparation feeding ``NativeParameter`` — never in forward or backward.

**Dtype** (Phase I, milestone I7). ``dtype`` is keyword-only, defaults to
``"float64"``, and accepts exactly ``"float64"`` and ``"float32"``; both
parameters are built at it and ``self.dtype`` reports it read-only. The
Core Conv2d kernels have been dtype-general since milestone I5, so this
milestone wires module state to them and adds **no** kernel, export,
dispatch, or workspace. The **host initialization draw is unchanged at
every dtype** — same generator, same conv fan-in, same order — so a
float32 layer with seed *S* holds exactly ``float32(the float64 draw with
seed S)`` (design §12.3). The input must match the weight's dtype exactly;
there is no promotion and no cast. float32 remains **publicly
unsupported** until milestone I9, and a call that omits ``dtype`` is
byte-identical to every pre-Phase-I run.

**Constructor** ``NativeConv2d(in_channels, out_channels, kernel_size,
stride=1, padding=0, bias=True, *, seed=None, requires_grad=True,
dtype=None)``.
``in_channels`` / ``out_channels`` are real positive ints (bools
rejected). ``kernel_size`` / ``stride`` (each ≥ 1) and ``padding`` (≥ 0)
are an int or a 2-element ``(height, width)`` pair, normalized to and
stored as two-element tuples via the native ``_spatial_pair`` helper
(bools and malformed forms rejected). ``bias`` and ``requires_grad`` must
be real bools; ``seed`` an int or None. Every argument is validated
**before** any native allocation, so a bad call never leaks parameter
storage. And if the bias allocation fails *after* the weight is already
allocated, the constructor closes the weight deterministically before
re-raising, so a failed construction leaves no live native storage behind
(never relying on eventual garbage collection).

**Input contract:** ``forward(input)`` requires an open 4-D NCHW
``NativeTensor`` whose ``shape[1] == in_channels`` and whose dtype/device
match the weight (the module's dtype, on ``cpu``). Nothing is wrapped,
reshaped, or broadcast implicitly; the stable framework's ``Tensor``, arrays,
lists, scalars, closed tensors, wrong-rank, and channel-mismatched inputs
are rejected with clear errors before ``conv2d`` runs. Non-contiguous
inputs are supported through the existing Conv2d Policy-B copy path. The
output is an ordinary owning ``NativeTensor`` (never a parameter)
requiring grad exactly when a participating operand does; forward does
not depend on ``training``.

``requires_grad=False`` freezes both parameters (registered, traversable,
in ``state_dict()``, but no gradient — a requiring input still receives
its gradient). State loading and checkpoints follow the existing v3.3 /
v3.14 paths unchanged (independent owning snapshots, atomic
validate→stage→commit, identity preserved). Fully separate from
``tensorforge.nn.Conv2d``; cpu only; experimental and explicit.
"""

import math

import numpy as np

from ..backends.cpp import _spatial_pair
from ._native_dtype import normalize_module_dtype
from .native_module import NativeModule
from .native_parameter import NativeParameter
from .native_tensor import NativeTensor


def _validate_channel_count(value, name):
    """``in_channels``/``out_channels`` must be a real positive int —
    bools and integer-like objects are rejected, matching the project's
    strict flag/count validation style (see ``NativeLinear``)."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


class NativeConv2d(NativeModule):
    """A native 2-D convolution layer: ``output = input.conv2d(weight,
    bias, stride, padding)`` over the existing D6 autograd primitive.

    ``NativeConv2d(in_channels, out_channels, kernel_size, stride=1,
    padding=0, bias=True, *, seed=None, requires_grad=True, dtype=None)``
    — see the module docstring for the full contract (OIHW weight layout,
    deterministic fan-in initialization, 4-D NCHW input semantics, frozen
    parameters, state_dict keys, and the Phase-I dtype rules).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, *, seed=None, requires_grad=True,
                 dtype=None):
        # Validate and normalize every Python argument before any native
        # allocation, so a bad call never creates parameter storage it
        # abandons. kernel_size/stride (>= 1) and padding (>= 0) reuse the
        # native spatial-pair helper (int or 2-pair, bools/malformed
        # rejected) — the same semantics NativeTensor.conv2d applies.
        _validate_channel_count(in_channels, "in_channels")
        _validate_channel_count(out_channels, "out_channels")
        kernel_size = _spatial_pair(kernel_size, "kernel_size", minimum=1)
        stride = _spatial_pair(stride, "stride", minimum=1)
        padding = _spatial_pair(padding, "padding", minimum=0)
        if not isinstance(bias, bool):
            raise TypeError(f"bias must be a bool, got {type(bias).__name__}")
        if not isinstance(requires_grad, bool):
            raise TypeError(
                f"requires_grad must be a bool, got {type(requires_grad).__name__}"
            )
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool)
        ):
            raise TypeError(
                f"seed must be an int or None, got {type(seed).__name__}"
            )
        # Phase I, milestone I7 — the module's dtype, validated before any
        # native allocation. ``None`` means ``"float64"``.
        dtype = normalize_module_dtype(dtype)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._dtype = dtype
        # Fan-in uniform initialization from a *local* generator (the
        # native convention — no global RNG), using the conv fan-in the
        # stable Conv2d uses. NumPy here is host-side data preparation only
        # (the NativeParameter entry boundary): no graph is built and no
        # native compute runs.
        #
        # Phase I, milestone I7 keeps the host draw identical at every dtype
        # (design §12.3): same generator, same conv fan-in, same bound
        # computed in binary64, same draw order and sizes, same float64 host
        # array. A float32 layer with seed S therefore holds exactly
        # float32(the float64 draw with seed S) — one rounding, at the
        # NativeParameter ingress boundary, and no second random stream.
        kh, kw = kernel_size
        fan_in = in_channels * kh * kw
        bound = 1.0 / math.sqrt(fan_in)
        rng = np.random.default_rng(seed)
        self.weight = NativeParameter(
            rng.uniform(-bound, bound, size=(out_channels, in_channels, kh, kw)),
            requires_grad=requires_grad, dtype=dtype,
        )
        if bias:
            # The weight's native storage is already allocated; if the bias
            # allocation fails (e.g. MemoryError), close the weight
            # deterministically rather than abandoning it to eventual GC, so
            # a partially constructed layer leaks no native storage. The
            # whole half-built module is then discarded as __init__ re-raises.
            try:
                self.bias = NativeParameter(
                    rng.uniform(-bound, bound, size=(out_channels,)),
                    requires_grad=requires_grad, dtype=dtype,
                )
            except BaseException:
                self.weight.close()
                raise
        else:
            # Readable as None; nothing registered under "bias", so
            # traversal and state_dict() see only "weight".
            self.bias = None

    @property
    def dtype(self):
        """The dtype this layer's parameters were constructed with —
        read-only, ``"float64"`` unless ``dtype="float32"`` was requested
        (Phase I, milestone I7; design §25.3). A report, not a second
        authority: ``forward`` compares the input against the weight's own
        tag."""
        return self._dtype

    def forward(self, input):
        """``input.conv2d(weight, bias, stride, padding)`` over the
        existing D6 autograd primitive — a 4-D NCHW ``(N, C, H, W)`` input
        with ``C == in_channels``, an ``(N, O, out_h, out_w)`` output.
        Backward comes entirely from the existing conv2d autograd."""
        if not isinstance(input, NativeTensor):
            raise TypeError(
                f"NativeConv2d.forward requires a NativeTensor input, got "
                f"{type(input).__name__}"
            )
        if input.closed:
            raise RuntimeError(
                "NativeConv2d.forward: the input tensor has been closed"
            )
        weight = self.weight
        if weight.closed:
            raise RuntimeError("NativeConv2d.forward: weight has been closed")
        bias = self.bias
        if bias is not None and bias.closed:
            raise RuntimeError("NativeConv2d.forward: bias has been closed")
        # A clearer module-specific rank/channel error before conv2d runs
        # (conv2d re-validates too, but names neither in_channels).
        if input.ndim != 4:
            raise ValueError(
                f"NativeConv2d expects 4-D NCHW input (batch, in_channels="
                f"{self.in_channels}, height, width), got shape {input.shape}"
            )
        if input.shape[1] != self.in_channels:
            raise ValueError(
                f"NativeConv2d expects input channels == in_channels="
                f"{self.in_channels}, got {input.shape[1]} (input shape "
                f"{input.shape})"
            )
        if input.dtype != weight.dtype or input.device != weight.device:
            raise ValueError(
                f"NativeConv2d expects input dtype/device "
                f"{weight.dtype}/{weight.device}, got "
                f"{input.dtype}/{input.device}"
            )
        return input.conv2d(
            weight, bias, stride=self.stride, padding=self.padding
        )

    def __repr__(self):
        return (
            f"NativeConv2d(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bias={self.bias is not None})"
        )
