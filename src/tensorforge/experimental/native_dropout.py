"""NativeDropout — the native Dropout module (Phase G, milestone G4;
contract locked in docs/native_rng_dropout_design.md §6, §9, and §19/G4).

G4 is the **public layer** over the completed G3 operation, and nothing
more. It adds no kernel, no C ABI symbol, no ctypes declaration, no
``NativeTensorCore`` method, no ``NativeTensor`` operation, and no
checkpoint-format change: the whole module is argument validation, one
registered generator, a train/eval dispatch, and a delegation to
``NativeTensor.dropout``.

Construction
------------

``NativeDropout(p=0.5, seed=None, generator=None)``

- ``p`` is the drop probability, validated by the **same** shared
  normalizer the G2 Core and the G3 operation use — there is deliberately
  no third probability rule, so the accepted/rejected matrix is identical
  at every layer by construction: ``bool`` is a ``TypeError``, a non-real
  is a ``TypeError``, and ``p == 1``, ``p > 1``, ``p < 0``, NaN, and ±inf
  are all ``ValueError``. ``p == 0`` is accepted and means identity. The
  canonical validated value is stored as a plain Python ``float``.
- ``seed`` and ``generator`` are **mutually exclusive**. Supplying both
  is a ``TypeError`` rather than a silent preference for one of them: a
  seed that is quietly ignored is exactly the kind of "looks
  reproducible, is not" failure explicit random state exists to prevent.
- ``generator=None`` (with or without a ``seed``) makes the module
  **create and own** one ``NativeGenerator`` — ``NativeGenerator(seed)``,
  which draws once from OS entropy when ``seed is None`` and is an exact
  ``int`` in ``[0, 2**64 - 1]`` otherwise. Two modules built this way have
  independent streams, which is the default precisely because two layers
  accidentally sharing one stream is the harder bug to notice.
- An explicit ``generator`` is **registered as the exact object given** —
  never cloned, never copied, never re-seeded, and never wrapped. That is
  how two layers deliberately draw from one interleaved stream (§3.7).

Every argument is validated before the generator is created or
registered, so a rejected construction creates no generator, consumes no
entropy, registers nothing, and allocates no native storage. An
explicitly supplied generator is left bit-identical.

Registered state
----------------

The generator is registered under the canonical name ``"generator"`` and
is readable as ``module.generator``. It is Phase G's **fourth**
registration category — not a parameter, not a buffer, not a child
module, and not an ordinary attribute — so it appears in
``generators()``, ``named_generators()``, and ``generator_state_dict()``,
and is deliberately **absent** from ``state_dict()``, which stays
contractually ``{name: NativeTensor}``. A generator owns no native
storage and has no ``close()``, so the module owns no native storage
either: constructing, registering, running, and discarding a
``NativeDropout`` leaves the native live-storage count untouched apart
from the outputs its forwards return.

``load_generator_state_dict()`` replaces the state **in place**, so the
registered object's identity survives a load and a module that shares its
generator with another still shares it afterwards.

The public surface is exactly ``p``, ``generator``, ``training``, and the
ordinary ``NativeModule`` methods. In particular there is **no ownership
flag**: whether this module created its generator or adopted one is a
fact about a single moment in the constructor, and it stops being true
the moment that generator is shared with a second module, re-registered,
or aliased. A stored Boolean would then be a claim the object graph
contradicts, and a *writable* one would let a caller change the claim
without changing anything real. Ownership is read where it is actually
recorded — the generator's identity and the registered topology, i.e.
``module.generator is other.generator`` and ``named_generators()`` over
the model.

Forward
-------

Three cases, and the layering is deliberate::

    if not isinstance(input, NativeTensor) or input.closed:  -> error
    if not self.training:                                    -> input
    return input.dropout(self.p, generator=self.generator)

- **Training** (``self.training is True``) delegates to the G3 operation,
  which owns the whole generator call transaction: it reserves one call,
  runs the stateless G2 Core with that reservation's seed and index,
  adopts the private multiplier mask as graph-owned state, and commits as
  its last action. So a successful training forward consumes **exactly
  one** call and a failed one consumes **none** — this module
  reimplements no part of that and can therefore add no failure hole to
  it.
- **Evaluation** (``self.training is False``) returns the **input object
  itself**: no reservation, no allocation, no kernel, no graph node, and
  no call consumed. Evaluation forwards therefore leave **no gap in the
  random stream** — a training forward before and after an arbitrary
  number of eval forwards use consecutive call indices.
- **``p == 0``** is identity too, and is deliberately **not**
  short-circuited here: §6.2 assigns that rule to the operation, which
  already returns the caller's own tensor after validating everything and
  before reserving, allocating, or drawing anything. Duplicating it in
  the module would create a second copy of a rule Phase G has kept in
  exactly one place, and would make the two able to disagree. The
  observable behavior is identical either way — ``result is input``, no
  call consumed — and the tests prove the Core is never reached.

Input validation runs **first**, before the mode dispatch, so a closed or
non-``NativeTensor`` input is an error in evaluation mode too rather than
being handed straight back. Nothing is wrapped, cast, reshaped, or moved,
and the stable framework's ``Tensor``, NumPy arrays, lists, and scalars
are rejected with clear errors — there is no implicit conversion in
either direction.

What G4 does **not** do
-----------------------

Generator state is **not** written to or read from a native checkpoint:
the format is still version 1 and has no generator section, so saving a
model containing a ``NativeDropout`` preserves its parameters and buffers
and **silently omits the random stream**. Exact stochastic resume
therefore does not exist yet; that is milestone G5, which moves the
format to version 2. Because of that — and because Dropout's whole value
is exact, demonstrated reproducibility — ``"dropout"` remains listed in
``UNSUPPORTED`` until the G10 closure, even though this module and its
export are real. The registry reports what is *closed and validated*; the
inventories report what *exists*.

Fully separate from ``tensorforge.nn.Dropout``; CPU only;
experimental and explicit. It takes **no** dtype argument and must
not gain one — it inherits the dtype of whatever flows through it.
``tf_core_dropout_forward`` became dtype-general at Phase I
milestone I7 with its exact ABI shape unchanged and the random
derivation untouched, so one ``(seed, call_index, element count)``
key drops exactly the same elements at float32 and float64; only
the two multiplier values differ.
"""

from ..backends import cpp
from .native_generator import NativeGenerator
from .native_module import NativeModule
from .native_tensor import NativeTensor

# The canonical registered name of the module's generator (§9.4/§10.3:
# canonical names come from the traversal, and a stable attribute name is
# what makes a model's generator paths stable across runs).
_GENERATOR_NAME = "generator"


class NativeDropout(NativeModule):
    """Inverted Dropout as a module: stochastic in training, identity in
    evaluation, identity at ``p == 0``, over one registered
    ``NativeGenerator``.

    ``NativeDropout(p=0.5, seed=None, generator=None)`` — ``seed`` and
    ``generator`` are mutually exclusive; without an explicit generator
    the module creates and owns one, and with one it registers that exact
    object so several layers can share a single ordered stream. See the
    module docstring for the full contract."""

    def __init__(self, p=0.5, seed=None, generator=None):
        # --- validate every argument before creating or registering
        # anything. A rejected construction draws no entropy, builds no
        # generator, registers nothing, and allocates no native storage.
        #
        # The probability goes through the one shared validator (§6.1),
        # never a third rule of this module's own.
        probability = cpp._normalize_dropout_probability(p, "NativeDropout")
        if seed is not None and generator is not None:
            raise TypeError(
                "NativeDropout takes 'seed' or 'generator', not both: a "
                "seed is how the module builds its own generator, so "
                "supplying one alongside an explicit generator would "
                "have to ignore one of them. Pass generator=... to share "
                "an existing stream, or seed=... to own a new one."
            )
        if generator is not None and not isinstance(generator, NativeGenerator):
            raise TypeError(
                f"NativeDropout generator must be a NativeGenerator or "
                f"None, got {type(generator).__name__}"
            )
        # An explicit generator is adopted exactly as given. Otherwise the
        # module builds its own, which validates `seed` itself: an exact
        # int in [0, 2**64 - 1], bool and NumPy scalars rejected, and
        # `None` drawing once through `secrets.randbits(64)`. Nothing is
        # cloned, copied, or re-seeded either way.
        #
        # Which of the two paths ran is deliberately **not** recorded. It
        # is a fact about one moment in the constructor, not a durable
        # truth: a generator this module created can be handed to another
        # module a line later, and a stored flag would then claim an
        # exclusivity that no longer holds. The authoritative state is the
        # generator's identity and the registered topology — which is what
        # `named_generators()` over a model actually reports.
        if generator is None:
            generator = NativeGenerator(seed)

        super().__init__()
        self.p = probability
        # Registration through the module system's real mechanism:
        # assignment registers a NativeGenerator as the fourth state
        # category (a generator is an unambiguous native type, so it
        # needs no explicit-call discipline). The exact object is stored,
        # never a copy.
        setattr(self, _GENERATOR_NAME, generator)

    def forward(self, input):
        """Stochastic in training, the input object itself in evaluation.

        Validation runs first, so a closed or non-``NativeTensor`` input
        is an error in **both** modes rather than being handed back. In
        training this delegates to ``NativeTensor.dropout``, which owns
        the entire generator call transaction; ``p == 0`` is that
        operation's identity rule (§6.2) and is deliberately not
        duplicated here."""
        if not isinstance(input, NativeTensor):
            raise TypeError(
                f"NativeDropout.forward requires a NativeTensor input, got "
                f"{type(input).__name__}"
            )
        if input.closed:
            raise RuntimeError(
                "NativeDropout.forward: the input tensor has been closed"
            )
        if not self.training:
            # Evaluation identity: the caller's own tensor, un-copied. No
            # reservation, no allocation, no kernel, no graph node, and
            # no call consumed — so an eval forward leaves no gap in the
            # stream and the next training forward takes the next index.
            return input
        return input.dropout(self.p, generator=getattr(self, _GENERATOR_NAME))

    def __repr__(self):
        # Stable configuration only. The generator's seed and call count
        # both move at runtime (`reseed`, and every successful stochastic
        # forward), and a repr that changed as a model trained would be
        # useless for identifying the layer; read `module.generator` for
        # live state. Reservation tokens, the private lock, construction
        # claims, and saved masks are never exposed here or anywhere.
        return f"NativeDropout(p={self.p})"
