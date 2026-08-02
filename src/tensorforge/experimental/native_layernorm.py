"""NativeLayerNorm — the first native normalization module (Phase F,
milestone F2; see docs/native_normalization_design.md §5).

Layer normalization normalizes each sample over its own trailing
dimensions, so — unlike BatchNorm — no other sample participates, there
are **no running statistics, no buffers, and no train/eval difference**:
the same input always gives the same output. It is the *stateless* half
of Phase F, and it ships first precisely because it exercises the
composed normalization mathematics without touching the mutable-buffer
machinery at all.

The whole layer is **composed from existing differentiable
``NativeTensor`` operations** — ``mean``, ``subtract``, ``multiply``,
``add``, ``sqrt``, and ``reciprocal``. It adds no C++ code, no
normalization kernel, no C ABI symbol, no ctypes declaration, no
``NativeTensorCore`` method, no ``NativeTensor.layer_norm`` operation,
and no hand-written backward: because forward is built entirely from
existing differentiable ops, the existing Python-managed native autograd
engine **is** the backward implementation, and gradients flow to the
input (and to ``weight``/``bias`` when affine) through the mean and the
variance for free.

**Forward** (trailing ``k = len(normalized_shape)`` dimensions)::

    mean       = mean(input over the trailing axes, keepdims)
    centered   = input - mean
    variance   = mean(centered * centered over the trailing axes, keepdims)
    normalized = centered * reciprocal(sqrt(variance + eps))
    output     = normalized * weight + bias         # elementwise_affine
    output     = normalized                         # otherwise

The variance is the **population** variance (divide by the element count,
no Bessel correction), matching the stable ``tensorforge.nn.LayerNorm``,
and **epsilon is added inside the square root** — ``sqrt(var + eps)``,
never ``sqrt(var) + eps``. ``NativeTensor.mean`` reduces one axis at a
time, so the multi-axis mean is taken as a sequence of single-axis means
with ``keepdims=True``; because each reduced dimension is retained at
size 1, the original trailing-axis numbers stay valid across the whole
sequence (no tuple-axis reduction is added to ``NativeTensor``). The
``eps`` constant is created natively as a graph-free rank-0 scalar at the
**graph's own dtype** — no NumPy and no stable-framework arithmetic runs
in forward or backward.

**Dtype** (Phase I, milestone I7). ``dtype`` is keyword-only, defaults to
``"float64"``, and accepts exactly ``"float64"`` and ``"float32"``.
``elementwise_affine=True`` builds ``weight`` and ``bias`` at it, and the
input must then match exactly. ``elementwise_affine=False`` owns no
numeric state, so it normalizes whatever dtype it is handed and the
constructed value is only reported, never enforced — the module cannot be
a second authority over data it does not own. ``eps`` stays a Python
``float`` and is materialized as a **graph-dtype** rank-0 native scalar
per forward, so no float64 constant ever enters a float32 graph. Every
reduction, the variance, the root, the reciprocal, and the affine step
therefore run at the graph's own width, with no hidden wider accumulator
and no NumPy compute (design §10.1). float32 remains **publicly
unsupported** until milestone I9; omitting ``dtype`` is byte-identical to
every pre-Phase-I run.

**Construction** — ``NativeLayerNorm(normalized_shape, eps=1e-5,
elementwise_affine=True, *, dtype=None)``:

- ``normalized_shape`` accepts a positive plain ``int`` (normalized to
  ``(value,)``) or a non-empty ``tuple``/``list`` of positive plain
  ``int`` values (normalized to a tuple). ``bool`` is rejected at every
  position, as everywhere in the native line. Stored as
  ``self.normalized_shape``.
- ``eps`` must be a positive real (``bool`` and non-real values
  rejected); stored as a Python ``float`` — never clamped or replaced.
- ``elementwise_affine`` must be a real ``bool``, stored without
  coercion.
- Every argument is validated **before** the first native allocation, so
  a rejected construction leaves the native live-storage count unchanged.
- ``elementwise_affine=True`` creates ``weight`` (ones) and ``bias``
  (zeros) as ``NativeParameter``s of shape ``normalized_shape``,
  registered in that order (weight, then bias); both are independent,
  owning, contiguous, graph-free cpu leaves at the module dtype and at
  parameter version 0. If the bias allocation fails, the weight storage is
  closed deterministically rather than abandoned to GC.
- ``elementwise_affine=False`` registers **no parameters**, contributes
  no state keys, and allocates no affine storage; ``weight`` and ``bias``
  read as ``None``.

**Input contract**: ``forward(input)`` requires an open ``NativeTensor``
(a ``NativeParameter`` is accepted as the subclass it is); the stable
framework's ``Tensor``, NumPy arrays, lists, tuples, scalars, arbitrary
objects, and closed tensors are rejected with clear errors. Nothing is
wrapped, cast, reshaped, or moved. The input rank must be at least ``k``
and its trailing ``k`` dimensions must equal ``normalized_shape``
exactly; a mismatch is rejected — naming the configured shape, the
expected trailing-dimension count, and the actual shape — before any
graph node is built. When affine, ``weight`` and ``bias`` are also
checked open, correctly shaped, and dtype/device-matched first. Forward
never mutates the input, the parameters, gradients, versions, or the
training flag, and never reads ``self.training``.

The output is a **fresh, owning, row-major contiguous** ``NativeTensor``
of the input's shape, independent of the input's storage and never a
``NativeParameter`` or a borrowing view. The module stores no forward
temporaries; graph lifetime, ``retain_graph``, and one-shot cleanup are
exactly the existing engine's. Fully separate from
``tensorforge.nn.LayerNorm``; cpu only.
"""

import numbers

import numpy as np

from ._native_dtype import normalize_module_dtype
from .native_module import NativeModule
from .native_parameter import NativeParameter
from .native_tensor import NativeTensor


def _normalized_shape_tuple(normalized_shape):
    """Validate ``normalized_shape`` and return the canonical tuple.

    Accepts a positive plain ``int`` (``bool`` excluded) or a non-empty
    ``tuple``/``list`` of positive plain ``int``s. The native-line
    convention decides the exception kind deterministically: a wrong
    *type* — of the argument or of a member — raises ``TypeError``; a
    right type with a bad *value* (empty sequence, non-positive member)
    raises ``ValueError``. Nothing is coerced or stringified."""
    # bool is an int subclass, so it must be rejected before the int
    # branch, exactly as the count/flag validators elsewhere do.
    if isinstance(normalized_shape, bool):
        raise TypeError(
            "normalized_shape must be a positive int or a non-empty "
            "sequence of positive ints, got bool"
        )
    if isinstance(normalized_shape, int):
        shape = (normalized_shape,)
    elif isinstance(normalized_shape, (tuple, list)):
        if len(normalized_shape) == 0:
            raise ValueError(
                "normalized_shape must be a non-empty sequence of positive "
                "ints, got an empty sequence"
            )
        for value in normalized_shape:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"normalized_shape entries must be plain ints, got "
                    f"{type(value).__name__}"
                )
        shape = tuple(normalized_shape)
    else:
        raise TypeError(
            f"normalized_shape must be a positive int or a non-empty "
            f"sequence of positive ints, got {type(normalized_shape).__name__}"
        )
    for value in shape:
        if value <= 0:
            raise ValueError(
                f"normalized_shape values must be positive, got {shape}"
            )
    return shape


class NativeLayerNorm(NativeModule):
    """Native layer normalization over the trailing ``normalized_shape``
    dimensions: ``(x - mean) / sqrt(var + eps)`` (population variance),
    optionally affine-scaled by ``weight`` and shifted by ``bias``.

    ``NativeLayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)``
    — see the module docstring for the full contract (validation order,
    parameter initialization, the exact forward mathematics, the input
    contract, and the ownership guarantees). Stateless: no buffers, no
    running statistics, identical in train and eval mode; backward is
    supplied entirely by the existing native autograd."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 *, dtype=None):
        # Validate every Python argument before any native allocation, so
        # a bad call never creates parameter storage it abandons — the
        # live-storage count is unchanged on rejection.
        shape = _normalized_shape_tuple(normalized_shape)
        if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
            raise TypeError(
                f"eps must be a real number, got {type(eps).__name__}"
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps!r}")
        if not isinstance(elementwise_affine, bool):
            raise TypeError(
                f"elementwise_affine must be a bool, got "
                f"{type(elementwise_affine).__name__}"
            )
        # Phase I, milestone I7 — validated at **both** affine settings, and
        # deliberately so. A non-affine LayerNorm owns no numeric state and
        # takes its working dtype from the input, so the argument reports
        # nothing about storage there; but the I7 constructor surface names
        # ``NativeLayerNorm`` unconditionally, and a constructor that
        # silently accepted ``dtype="f4"`` in one configuration and rejected
        # it in the other would be the worse contract. So the value is
        # always validated and always reported; what it *governs* is the
        # affine parameters, which is where a dtype can actually live.
        dtype = normalize_module_dtype(dtype)
        super().__init__()
        self.normalized_shape = shape
        self.eps = float(eps)
        self.elementwise_affine = elementwise_affine
        self._dtype = dtype
        if elementwise_affine:
            # weight=ones, bias=zeros — NumPy here is host-side data
            # preparation feeding NativeParameter (the from_array entry
            # boundary), the NativeLinear precedent; no graph is built and
            # no native compute runs. Registration order is weight, bias.
            # Both are built at the module's dtype (I7); the host values are
            # exact small integers, so the ingress conversion is exact at
            # both widths.
            self.weight = NativeParameter(np.ones(shape), requires_grad=True,
                                          dtype=dtype)
            # The weight's native storage is already allocated; if the
            # bias allocation fails (e.g. MemoryError), close the weight
            # deterministically rather than leaking it to eventual GC. The
            # half-built module is then discarded as __init__ re-raises.
            try:
                self.bias = NativeParameter(np.zeros(shape),
                                            requires_grad=True, dtype=dtype)
            except BaseException:
                self.weight.close()
                raise
        else:
            # No affine parameters at all: readable as None, nothing
            # registered, so parameters()/named_parameters()/state_dict()
            # are empty and no affine storage is allocated.
            self.weight = None
            self.bias = None

    @property
    def dtype(self):
        """The dtype this layer was constructed with — read-only,
        ``"float64"`` by default (Phase I, milestone I7; design §25.3).

        When ``elementwise_affine=True`` it is the dtype of ``weight`` and
        ``bias``, and therefore the dtype the input must match. When
        ``elementwise_affine=False`` the layer owns no numeric state at all
        and normalizes whatever dtype it is given: the property still
        reports the constructed value, but nothing is validated against it,
        because inventing a second authority over data the module does not
        own is exactly what design §12.1 rejects for the stateless
        modules."""
        return self._dtype

    def forward(self, input):
        """Normalize ``input``'s trailing ``len(normalized_shape)``
        dimensions and, when affine, scale by ``weight`` and shift by
        ``bias``. Composed from existing native ops, so the existing
        autograd provides every gradient. Identical in train and eval
        mode — ``self.training`` is never read."""
        if not isinstance(input, NativeTensor):
            raise TypeError(
                f"NativeLayerNorm.forward requires a NativeTensor input, got "
                f"{type(input).__name__}"
            )
        if input.closed:
            raise RuntimeError(
                "NativeLayerNorm.forward: the input tensor has been closed"
            )
        k = len(self.normalized_shape)
        if input.ndim < k or input.shape[-k:] != self.normalized_shape:
            raise ValueError(
                f"NativeLayerNorm{self.normalized_shape} expects input whose "
                f"last {k} dimension(s) are {self.normalized_shape}, got shape "
                f"{input.shape}"
            )
        weight = self.weight
        bias = self.bias
        if self.elementwise_affine:
            # Validate the affine operands before building any graph node.
            if weight.closed:
                raise RuntimeError("NativeLayerNorm.forward: weight has been closed")
            if bias.closed:
                raise RuntimeError("NativeLayerNorm.forward: bias has been closed")
            if (weight.shape != self.normalized_shape
                    or bias.shape != self.normalized_shape):
                raise ValueError(
                    f"NativeLayerNorm affine parameters must have shape "
                    f"{self.normalized_shape}, got weight {weight.shape} / "
                    f"bias {bias.shape}"
                )
            if input.dtype != weight.dtype or input.device != weight.device:
                raise ValueError(
                    f"NativeLayerNorm expects input dtype/device "
                    f"{weight.dtype}/{weight.device}, got "
                    f"{input.dtype}/{input.device}"
                )

        # The trailing axes to normalize. keepdims=True everywhere keeps
        # the rank and therefore the axis numbers stable across the
        # sequential single-axis reductions (NativeTensor.mean takes one
        # axis at a time; no tuple-axis reduction is added).
        axes = tuple(range(input.ndim - k, input.ndim))

        # Every native temporary this forward creates, in creation order.
        # If an operation raises part-way through, these are **unadopted**
        # (no output was returned, and the autograd graph that would own
        # them for backward was never handed to the caller), so they must
        # be closed explicitly rather than left to garbage collection: the
        # grad-building path puts each `sqrt`/`reciprocal` result node into
        # a reference cycle (its backward closure captures the node itself),
        # which reference counting alone cannot reclaim, so a failure after
        # those exist would otherwise leak their native storage until GC.
        # On the **successful** path nothing here is closed — the returned
        # graph still needs every one of them — and `created` is simply
        # dropped with the frame. The input, weight, and bias are never
        # tracked, so they are never closed.
        created = []

        def track(tensor):
            created.append(tensor)
            return tensor

        try:
            mean = self._mean_over(input, axes, track)
            centered = track(input.subtract(mean))
            squared = track(centered.multiply(centered))
            variance = self._mean_over(squared, axes, track)
            # sqrt(var + eps), never sqrt(var) + eps: epsilon is added
            # inside the root, from a native graph-free scalar (no NumPy)
            # **at the graph's own dtype** (Phase I, milestone I7). The
            # private typed constructor is what makes that possible for a
            # float32 graph: a literal-float64 constant meeting a float32
            # operand would be a mixed-dtype request, which the runtime
            # refuses, and design §11.4 forbids introducing one. The Python
            # ``float`` self.eps is narrowed once, inside tf_storage_fill.
            eps_tensor = track(NativeTensor._typed_full(
                (), self.eps, input.dtype, device=input.device,
            ))
            var_plus_eps = track(variance.add(eps_tensor))
            std = track(var_plus_eps.sqrt())
            inverse_std = track(std.reciprocal())
            normalized = track(centered.multiply(inverse_std))
            if self.elementwise_affine:
                scaled = track(normalized.multiply(weight))
                # The final op's result is the returned output — never
                # tracked, so never closed here, and the graph it roots
                # keeps every tracked temporary alive for backward.
                return scaled.add(bias)
            # The non-affine output *is* the last tracked temporary;
            # returning it (rather than closing it) hands its graph to the
            # caller unchanged.
            return normalized
        except BaseException:
            # Close every unadopted temporary exactly once, most-recent
            # first. close() is idempotent and releases only each tensor's
            # own owning core — never the caller-owned input/weight/bias,
            # which are not in `created`. The original exception is
            # preserved by the bare re-raise.
            for tensor in reversed(created):
                tensor.close()
            raise

    @staticmethod
    def _mean_over(value, axes, track):
        """The mean of ``value`` over every axis in ``axes``, as a
        sequence of existing single-axis ``NativeTensor.mean`` calls with
        ``keepdims=True``. ``axes`` is a non-empty tuple of distinct
        normalized axes already derived from the input rank; because each
        reduced dimension is retained at size 1, the axis numbers stay
        valid across the whole sequence. Every reduction is
        differentiable, materializes nothing to NumPy, and never mutates
        ``value``. ``track`` records each intermediate mean so a
        mid-forward failure can release it deterministically. Private and
        LayerNorm-only — not a general reduction subsystem."""
        for axis in axes:
            value = track(value.mean(axis=axis, keepdims=True))
        return value

    def __repr__(self):
        return (
            f"NativeLayerNorm(normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, elementwise_affine={self.elementwise_affine})"
        )
