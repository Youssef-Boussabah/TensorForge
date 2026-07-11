"""NativeParameter and the parameter-registration contract (Advanced
C++ v3.1 — the first Phase C milestone; see docs/native_autograd_design.md
§19 and docs/backend_experiments.md).

``NativeParameter`` is the native training stack's trainable-leaf
abstraction: a ``NativeTensor`` subclass whose instances are always
**graph-free owning leaves**. It exists so the future ``NativeModule``
(v3.2) and native optimizers can discover trainable tensors by type and
key their state by **object identity** — exactly the role
``tensorforge.nn.Parameter`` plays for the Python framework, translated
to the native ownership model. The two never mix: a
``tensorforge.Parameter`` is not a ``NativeParameter`` and neither
accepts the other.

Design invariants (all tested in tests/test_native_parameter.py):

- **Leaf, always.** A parameter is constructed as a leaf with no
  ``_parents``, no ``_backward`` closure, and no freed-graph state, and
  nothing can turn the object into a non-leaf: the internal graph
  constructors are overridden so every operation result — math, views,
  ``contiguous_copy``, ``detach`` — is a plain ``NativeTensor``.
  Parameter-ness never propagates through operations; the only way to
  create a ``NativeParameter`` is calling ``NativeParameter(...)``
  itself. (The inherited ``from_array``/``zeros``/``full`` classmethods
  consequently also return plain ``NativeTensor`` — they are not
  parameter constructors.)
- **Independent owning contiguous storage, always.** Constructed from
  array-like data, the values are copied into fresh native storage.
  Constructed from an existing ``NativeTensor`` — leaf or non-leaf,
  contiguous or strided/offset view — the parameter takes an independent
  owning contiguous copy of the source's *current value* and inherits
  none of its graph history: closing the source never invalidates the
  parameter, closing the parameter never invalidates the source, and a
  backward through the source's graph neither reaches nor frees the
  parameter. A closed source is rejected with the usual RuntimeError.
- **requires_grad is a real bool**, default ``True``; ``False`` builds a
  frozen parameter that stays registerable and discoverable but
  accumulates no gradient (the existing ``NativeTensor`` rules: no graph
  is built through it and ``_accumulate_grad`` drops contributions).
- **Identity, not value.** No ``__eq__``/``__hash__`` overrides —
  deduplication and (future) optimizer state use object identity, so two
  equal-valued parameters are always distinct and comparing parameters
  never runs tensor math.

``NativeParameterRegistry`` is the minimal registration contract v3.2's
``NativeModule`` will build on — deliberately *not* a module hierarchy:

- Names are non-empty ``str`` without ``"."`` (dots are reserved for the
  future hierarchical state_dict keys, matching the Python framework's
  dotted paths). Invalid names raise; nothing is stringified silently.
- A slot accepts a ``NativeParameter``, or ``None`` to **unregister**
  the name (KeyError if it is not registered). Ordinary ``NativeTensor``
  and the Python framework's ``Tensor``/``Parameter`` are rejected —
  never wrapped implicitly.
- Traversal order is **insertion order**. Replacing a registered name
  keeps its position; unregistering deletes the slot, so registering the
  name again appends at the end.
- Replacement and removal never close, mutate, or transfer state
  between parameters — the registry stores plain Python references and
  owns nothing: it never copies storage, closes parameters, zeroes
  gradients, or touches ``requires_grad``, and dropping the registry
  leaves every parameter open (lifetime stays with the caller and
  ``close()``, per the NativeTensor rules).
- The same parameter may be registered under several names (the future
  shared-weight case). ``named_parameters()`` shows every alias;
  ``parameters()`` deduplicates by object identity (each unique
  parameter once — what an optimizer must iterate);
  ``unique_named_parameters()`` is the deduplicated named view where the
  first-registered name wins.
"""

import sys

from ..backends import cpp
from .native_tensor import NativeTensor


def _reject_framework_object(value, where):
    """Raise TypeError if ``value`` is a tensorforge.Tensor (Parameter
    included). Checked lazily through sys.modules so the native backend
    never imports the Python frontend — if the frontend was never
    imported, no such object can exist and there is nothing to check."""
    frontend = sys.modules.get("tensorforge.tensor")
    if frontend is not None and isinstance(value, frontend.Tensor):
        raise TypeError(
            f"{where} does not accept tensorforge.{type(value).__name__}: "
            f"the native and Python autograd engines never mix. Pass "
            f"array-like data or a NativeTensor instead."
        )


class NativeParameter(NativeTensor):
    """A trainable native leaf tensor.

    ``NativeParameter(data, requires_grad=True)`` copies ``data`` —
    array-like values, or the current value of an existing
    ``NativeTensor`` — into fresh owning contiguous float64/cpu native
    storage and marks it as a gradient-tracking leaf (pass
    ``requires_grad=False`` for a frozen parameter). Every instance is
    graph-free for its whole life; operations on a parameter return
    ordinary ``NativeTensor`` results. See the module docstring for the
    full contract.
    """

    __slots__ = ()

    def __init__(self, data, requires_grad=True):
        # Validate the flag before any native allocation, so a bad call
        # never creates storage it immediately leaks to GC cleanup.
        if not isinstance(requires_grad, bool):
            raise TypeError(
                f"requires_grad must be a bool, got {type(requires_grad).__name__}"
            )
        _reject_framework_object(data, "NativeParameter")
        if isinstance(data, NativeTensor):
            # Independent owning contiguous copy of the source's current
            # value — works for any layout (non-contiguous, offset,
            # borrowing views), raises RuntimeError on a closed source,
            # and by construction inherits no graph history: the new
            # core has never been near an operation.
            core = data._require_open().contiguous_copy()
        else:
            core = cpp.NativeTensorCore.from_array(data)
        super().__init__(core, owns_core=True)
        self._requires_grad = requires_grad

    # -- parameter-ness never propagates -----------------------------------
    #
    # Every NativeTensor operation builds its result through these two
    # classmethods, and on a subclass ``cls`` would be NativeParameter —
    # so without these overrides ``param.add(x)`` or ``param.T`` would
    # silently return "parameters" carrying graph state, violating the
    # leaf invariant. Delegating to NativeTensor explicitly guarantees
    # math, views, copies, and detach all return plain NativeTensors.

    @classmethod
    def _from_core(cls, core, owns_core=True):
        return NativeTensor._from_core(core, owns_core=owns_core)

    @classmethod
    def _from_op(cls, core, parents, backward, op, owns_core=True):
        return NativeTensor._from_op(
            core, parents, backward, op, owns_core=owns_core
        )

    def __repr__(self):
        if self._closed:
            return "NativeParameter(closed)"
        parts = [
            f"shape={self._core.shape}",
            f"contiguous={self._core.contiguous}",
        ]
        if not self._requires_grad:
            # A parameter tracks gradients by default, so only the
            # frozen state is worth flagging (the opposite of the base
            # tensor's repr).
            parts.append("requires_grad=False")
        return f"NativeParameter({', '.join(parts)})"


class NativeParameterRegistry:
    """An insertion-ordered name → NativeParameter registry — the
    minimal registration contract ``NativeModule`` (v3.2) will embed.

    ``register(name, parameter)`` adds or replaces a slot (replacement
    keeps the slot's position; the previous parameter is simply no
    longer referenced — never closed or mutated, and no gradient state
    transfers). ``register(name, None)`` unregisters the name (KeyError
    if it is not registered); registering it again later appends at the
    end. The registry stores references only — it owns no storage and
    never closes, copies, or mutates a parameter. See the module
    docstring for the full contract, including the alias/deduplication
    rules the traversal methods implement.
    """

    __slots__ = ("_parameters",)

    def __init__(self):
        self._parameters = {}  # name -> NativeParameter, insertion-ordered

    def register(self, name, parameter):
        """Register ``parameter`` under ``name``, replace what ``name``
        held, or unregister ``name`` (``parameter=None``)."""
        if not isinstance(name, str):
            raise TypeError(
                f"a parameter name must be a str, got {type(name).__name__}"
            )
        if not name:
            raise ValueError("a parameter name must be a non-empty string")
        if "." in name:
            raise ValueError(
                f"a parameter name must not contain '.' (reserved for "
                f"hierarchical state_dict keys), got {name!r}"
            )
        if parameter is None:
            if name not in self._parameters:
                raise KeyError(
                    f"cannot unregister {name!r}: no parameter is "
                    f"registered under that name"
                )
            del self._parameters[name]
            return
        if not isinstance(parameter, NativeParameter):
            # Covers ordinary NativeTensor and the Python framework's
            # Tensor/Parameter alike: nothing is wrapped implicitly.
            raise TypeError(
                f"only a NativeParameter (or None to unregister) can be "
                f"registered, got {type(parameter).__name__}; construct "
                f"NativeParameter(...) explicitly"
            )
        self._parameters[name] = parameter

    def named_parameters(self):
        """Every registered (name, parameter) pair, insertion-ordered.
        A parameter registered under several names appears once per
        alias. Returns a list (a snapshot — safe to mutate the registry
        while iterating it)."""
        return list(self._parameters.items())

    def parameters(self):
        """Each unique registered parameter exactly once, deduplicated
        by object identity (never by value), in first-registration
        order — the traversal a future optimizer iterates. Returns a
        list."""
        seen = set()
        unique = []
        for parameter in self._parameters.values():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                unique.append(parameter)
        return unique

    def unique_named_parameters(self):
        """Like ``parameters()`` but as (name, parameter) pairs: each
        unique parameter once, under its first-registered name
        (first-name-wins). Returns a list."""
        seen = set()
        unique = []
        for name, parameter in self._parameters.items():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                unique.append((name, parameter))
        return unique

    def __repr__(self):
        return f"NativeParameterRegistry(names={list(self._parameters)})"
