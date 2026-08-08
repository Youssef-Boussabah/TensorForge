"""The one dtype route the state-owning native modules construct through
(Phase I, milestone I7 — see docs/native_dtype_float32_design.md §12.1,
§12.2, and §27).

**This module is private.** It is deliberately absent from
``tensorforge.experimental.__all__`` and must stay that way: it is not a
public dtype API, not a capability declaration, and not a way around
``tensorforge.backends.cpp.normalize_dtype``.

What it is for
--------------

At milestone I7 six state-owning constructors gained a ``dtype``
argument — ``NativeParameter``, ``NativeLinear``, ``NativeConv2d``,
``NativeLayerNorm``, ``NativeBatchNorm1d``, and ``NativeBatchNorm2d`` —
each accepting exactly ``"float64"`` and ``"float32"`` and defaulting to
``"float64"``. They need **one** validator between them, for the reason
design §12.2 states directly: *no constructor invents its own dtype
validation*. Six independent checks would be six chances for the accepted
set, the exception kinds, or the messages to drift apart.

Why it delegated to the internal table, and why it no longer does
-----------------------------------------------------------------

At I7 ``cpp.normalize_dtype`` validated against ``SUPPORTED_DTYPES``, the
**public capability registry**, which still read ``("float64",)`` and moved
to ``("float64", "float32")`` at milestone **I9** and at no other (design
§27.3). Calling it then would either have rejected every float32 module —
making I7 impossible — or forced the registry to move five milestones
early, publishing a support promise the phase had not yet earned: float32
optimizers did not exist, checkpoint version 3 did not exist, and the exact
float32 resume proof had not been run. So this delegated to
``cpp._normalize_internal_dtype``, the private counterpart measured against
the internal representation table — the deliberate rollout pattern of
design §27, the one Phase G used for ``dropout``. I9 closed that gap by
moving the registry, after the proof.

**Phase K, milestone K1 narrowed the delegate to ``cpp.normalize_dtype``**
(see docs/native_integer_tensors_design.md §5.4). The two validators
accepted the same set on the day it landed, so the change is
behavior-preserving — and it is preventive from the milestone the
representation table learns a third name, because this is the **one**
validator the six state-owning constructors share, ``NativeParameter``
among them. Measured against the representation table, an ``int64``
entering that table would make ``NativeParameter(data, dtype="int64")``
legal the same day, and a trainable integer parameter is exactly what the
phase's autograd, optimizer, buffer, and checkpoint boundaries forbid.
Measured against ``SUPPORTED_DTYPES`` — which under Phase K's taxonomy
**is** the floating-compute registry, permanently — it cannot.

It is still a strict delegate with no rule of its own: the same
canonicalization, the same ``TypeError`` for a non-string, the same shape
of ``ValueError``, decided in one place.

Why the modules do not call ``cpp`` themselves
----------------------------------------------

``NativeLayerNorm`` and the shared BatchNorm implementation are proved, by
test, to compose **only** from differentiable ``NativeTensor`` operations
and never to reach into the ctypes layer — no ``ctypes`` import, no
``backends`` import, no ``NativeTensorCore`` attribute access. That
guardrail exists so a normalization kernel cannot be smuggled in, and a
dtype **string** validator is no reason to weaken it. Routing through this
module keeps the property exactly as Phase F and Phase H left it.
"""

from tensorforge.backends import cpp

# The exact set an I7 state-owning constructor accepts, in canonical order.
# Deliberately **not** a capability registry: it says what a module can be
# constructed at, never what TensorForge supports. The public answer to that
# is ``cpp.SUPPORTED_DTYPES``, which does not move until milestone I9.
MODULE_DTYPES = ("float64", "float32")


def normalize_module_dtype(dtype):
    """Validate and canonicalize a state-owning module's ``dtype``.

    ``None`` means ``"float64"`` (design §25.1, the same meaning it has at
    every other native dtype argument). ``"float64"`` and ``"float32"`` are
    returned unchanged. A non-string — a ``numpy.dtype``, a ``bool``, an
    ``int``, ``numpy.float32`` the *type* — raises ``TypeError``; any other
    string raises ``ValueError`` naming the value. There are no aliases:
    not ``"f4"``, not ``"single"``, not ``"Float32"``, not ``" float32"``.

    It is a strict delegate with no rule of its own, so the module surface
    and the storage layer can never disagree about what a dtype is. Since
    Phase K, milestone K1 the delegate is the **floating-compute**
    validator, so a state-owning module can never be constructed at a
    non-floating dtype — see the module docstring.
    """
    return cpp.normalize_dtype(dtype)


def require_floating_state_dtype(dtype, where, role="tensor"):
    """Reject a non-floating dtype where a native object would become
    **model or optimizer state** (Phase K, milestone K1).

    ``normalize_module_dtype`` above validates a *requested* dtype string;
    this validates the dtype an object *already carries* before that object
    is adopted as a parameter, a buffer, or an optimizer's charge. The two
    are different questions with different operands, and both are needed:
    a caller can name a legal dtype and hand over an object at another.

    A strict delegate over ``cpp._require_floating_dtype``, for exactly
    ``normalize_module_dtype``'s reason — one accepted set, one exception
    kind, one message shape, decided in one place — and routed through this
    private module so the state-owning experimental modules keep their
    existing distance from the ctypes layer. Raises ``ValueError`` before
    the caller registers, allocates, or mutates anything."""
    return cpp._require_floating_dtype(dtype, where, role=role)
