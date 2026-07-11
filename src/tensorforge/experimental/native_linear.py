"""NativeLinear — the first concrete native layer (Advanced C++ v3.4,
the fourth Phase C milestone; see docs/backend_experiments.md).

A fully connected layer on the completed native stack: a
``NativeModule`` holding a ``NativeParameter`` weight (and optionally a
``NativeParameter`` bias) whose forward is pure existing ``NativeTensor``
operations::

    output = input.matmul(weight)            # bias=False
    output = input.matmul(weight).add(bias)  # bias=True

**Weight orientation** (load-bearing for future checkpoints): ``weight``
is ``(in_features, out_features)`` — the same ``x @ weight`` orientation
as the stable framework's Linear — so the strictly 2-D native matmul
applies directly, with ``bias`` of shape ``(out_features,)`` broadcast
over the batch dimension by the existing zero-stride broadcast.

Because forward is built from existing differentiable operations, the
existing autograd engine *is* the backward implementation — there is no
manual or fused NativeLinear backward. Gradients land where Phase B
defined them: ``input.grad`` is ``(batch, in_features)``, ``weight.grad``
is ``(in_features, out_features)``, and ``bias.grad`` is
``(out_features,)`` (the broadcast-add backward reduces over the batch
via ``_unbroadcast``). Graph lifetime, ``retain_graph``, and one-shot
cleanup are unchanged.

**Initialization** is deterministic and self-contained: weight and bias
are sampled uniformly from ``[-1/sqrt(in_features),
+1/sqrt(in_features)]`` (a basic fan-in bound) using a **local**
``numpy.random.default_rng(seed)`` — an integer ``seed`` reproduces the
exact values, ``seed=None`` draws fresh entropy, and the global NumPy RNG
is never read or mutated (unlike the stable Linear's global-RNG
``randn``; the native line avoids hidden global state on purpose). NumPy
appears only here, as host-side data preparation feeding
``NativeParameter`` — never in forward or backward computation.

**Input contract** (strictly 2-D for now): ``forward(input)`` requires
an open ``NativeTensor`` of shape ``(batch_size, in_features)`` with
matching dtype/device (``float64``/``cpu``). Nothing is wrapped,
reshaped, flattened, or broadcast implicitly; the stable framework's
``Tensor``, NumPy arrays, lists, and scalars are rejected with clear
errors, as are closed inputs/weights/biases. The output is an ordinary
``NativeTensor`` (never a parameter) requiring grad exactly when a
participating operand does. Forward does not depend on ``training``.

``requires_grad=False`` freezes both parameters: they stay registered,
traversable, and in ``state_dict()``, but accumulate no gradients — a
requiring input still receives its gradient. With ``bias=False`` the
``bias`` attribute reads as ``None``, nothing is registered under
"bias", and ``state_dict()`` has only "weight" (loading a biased state
into a bias-free layer reports the extra key as *unexpected*, and the
reverse reports "bias" *missing*, under the v3.3 strict rules).

Registration order is ``weight`` then ``bias``, so ``parameters()`` /
``named_parameters()`` / ``state_dict()`` are deterministic
(``["weight", "bias"]``; nested as ``"layer.weight"``/``"layer.bias"``).
State loading follows v3.3 exactly: values are copied into the existing
parameter objects — identity, gradients, ``requires_grad``, and frozen
state all survive. The v3.3 mutation boundary applies unchanged: the
supported sequence is forward → backward → (optionally) load/update
after the graph completes; loading parameter values *between* forward
and backward is memory-safe but mathematically inconsistent (there is no
version counter — deliberately out of scope here).

Still experimental and explicit: float64/cpu only, no optimizer, loss,
training loop, activation modules, or ``NativeSequential`` yet (v3.5),
and fully separate from ``tensorforge.nn.Linear``.
"""

import math

import numpy as np

from .native_module import NativeModule
from .native_parameter import NativeParameter
from .native_tensor import NativeTensor


def _validate_feature_count(value, name):
    """``in_features``/``out_features`` must be a real positive int —
    bools and integer-like objects are rejected, matching the project's
    strict flag/count validation style."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


class NativeLinear(NativeModule):
    """A native fully connected layer: ``y = x @ weight (+ bias)``.

    ``NativeLinear(in_features, out_features, bias=True, *, seed=None,
    requires_grad=True)`` — see the module docstring for the full
    contract (weight orientation, deterministic initialization, strictly
    2-D input semantics, frozen parameters, and state_dict keys).
    """

    def __init__(self, in_features, out_features, bias=True, *,
                 seed=None, requires_grad=True):
        # Validate every Python argument before any native allocation,
        # so a bad call never creates parameter storage it abandons.
        _validate_feature_count(in_features, "in_features")
        _validate_feature_count(out_features, "out_features")
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
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Fan-in uniform initialization from a *local* generator: an int
        # seed reproduces values exactly, None draws fresh entropy, and
        # the global NumPy RNG is untouched either way. NumPy here is
        # host-side data preparation only (the from_array entry
        # boundary) — no graph is built and no native compute runs.
        bound = 1.0 / math.sqrt(in_features)
        rng = np.random.default_rng(seed)
        self.weight = NativeParameter(
            rng.uniform(-bound, bound, size=(in_features, out_features)),
            requires_grad=requires_grad,
        )
        if bias:
            self.bias = NativeParameter(
                rng.uniform(-bound, bound, size=(out_features,)),
                requires_grad=requires_grad,
            )
        else:
            # Readable as None; nothing registered under "bias", so
            # traversal and state_dict() see only "weight".
            self.bias = None

    def forward(self, input):
        """``input.matmul(weight)`` plus broadcast ``add(bias)`` when
        bias is enabled — strictly 2-D ``(batch_size, in_features)``
        input, ``(batch_size, out_features)`` output. Backward comes
        entirely from the existing matmul/broadcast-add autograd."""
        if not isinstance(input, NativeTensor):
            raise TypeError(
                f"NativeLinear.forward requires a NativeTensor input, got "
                f"{type(input).__name__}"
            )
        if input.closed:
            raise RuntimeError(
                "NativeLinear.forward: the input tensor has been closed"
            )
        weight = self.weight
        if weight.closed:
            raise RuntimeError("NativeLinear.forward: weight has been closed")
        bias = self.bias
        if bias is not None and bias.closed:
            raise RuntimeError("NativeLinear.forward: bias has been closed")
        if input.ndim != 2 or input.shape[1] != self.in_features:
            raise ValueError(
                f"NativeLinear expects 2-D input of shape "
                f"(batch_size, in_features={self.in_features}), got shape "
                f"{input.shape}"
            )
        if input.dtype != weight.dtype or input.device != weight.device:
            raise ValueError(
                f"NativeLinear expects input dtype/device "
                f"{weight.dtype}/{weight.device}, got "
                f"{input.dtype}/{input.device}"
            )
        output = input.matmul(weight)
        if bias is not None:
            output = output.add(bias)
        return output

    def __repr__(self):
        return (
            f"NativeLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )
