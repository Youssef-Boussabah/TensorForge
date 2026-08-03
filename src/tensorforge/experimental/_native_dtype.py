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

Why it is not ``normalize_dtype`` — yet
---------------------------------------

``cpp.normalize_dtype`` validates against ``SUPPORTED_DTYPES``, the
**public capability registry**, which still reads ``("float64",)`` and
moves to ``("float64", "float32")`` at milestone **I9** and at no other
(design §27.3). Calling it here would either reject every float32 module —
making I7 impossible — or force the registry to move five milestones
early, which would publish a support promise the phase has not yet earned:
float32 optimizers do not exist, checkpoint version 3 does not exist, and
the exact float32 resume proof has not been run.

So this delegates to ``cpp._normalize_internal_dtype``, the private
counterpart that measures against the internal representation table
instead. It is the *same* canonicalization, the *same* ``TypeError`` for a
non-string, and the *same* shape of ``ValueError`` — only the set it is
measured against differs, and only until I9, when the two become the same
set and this indirection becomes a formality. That gap between internal
capability and public promise is the deliberate rollout pattern of design
§27, the one Phase G used for ``dropout``: the operation existed from G3
and the *name* left ``UNSUPPORTED`` only at the G10 closure.

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
    and the storage layer can never disagree about what a dtype is.
    """
    return cpp._normalize_internal_dtype(dtype)
