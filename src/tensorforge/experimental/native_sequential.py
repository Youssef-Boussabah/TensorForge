"""NativeSequential — the ordered native composition container
(Advanced C++ v3.5; see docs/backend_experiments.md).

``NativeSequential(*modules)`` chains child ``NativeModule``s so each
one's output feeds the next::

    model = NativeSequential(
        NativeLinear(3, 4, seed=0),
        NativeReLU(),
        NativeLinear(4, 2, seed=1),
    )
    output = model(x)   # x -> linear -> relu -> linear

**Single source of truth.** Execution order *is* the registered
child-module order: children live in the inherited v3.2 registry under
deterministic **contiguous integer-string slot names** ``"0"``,
``"1"``, ... ``str(len-1)``, and ``forward`` folds the input through
them in that order. There is no second child list to drift out of sync;
instead the container enforces its slot invariant at the registration
funnel every mutation path already flows through:

- The constructor validates **all** entries before registering any
  (atomic); each must be a ``NativeModule`` — ``NativeLinear``,
  ``NativeReLU``, nested ``NativeSequential``, custom subclasses.
  ``NativeTensor``/``NativeParameter``, the stable framework's
  ``nn.Module``, callables, ``None``, and lists are rejected.
- ``append(module)`` validates, registers under ``str(len)``, and
  returns ``self`` (the repository's ``train()``-style chaining
  convention). Traversal and ``state_dict()`` reflect it immediately.
- A child may only be registered under an existing slot (an explicit
  **replacement**, which keeps that slot's position — via
  ``seq[i] = module`` or ``add_module("i", module)``) or under exactly
  the next free index (an **append**). Anything else — a gap-producing
  index, a non-numeric child name (which would create a registered but
  never-executed child), a non-canonical digit string like ``"01"`` —
  is rejected. Registering the sequence inside itself is rejected
  (module traversal is cycle-safe by v3.2, but *executing* a cyclic
  composition is meaningless and unsupported; don't build indirect
  ones either).
- Slots can be replaced but never removed: assigning ``None`` or an
  ordinary value over a slot, ``del``-ing a slot, and
  ``add_module(name, None)`` all raise — the ``0..len-1`` numbering
  can never silently develop a hole.
- **Direct parameters are rejected**: ``seq.w = NativeParameter(...)``
  (and ``register_parameter``) raise — a NativeSequential composes
  modules; put parameters inside a child module. Ordinary non-module
  attributes (labels, flags) remain normal attributes.

**Container API** (deliberately minimal): ``len(seq)`` is the slot
count; ``iter(seq)`` yields children in execution order **without**
deduplicating shared modules; ``seq[i]`` takes a real int (bool
rejected), supports Python-style negative indices, raises IndexError
out of range, and returns the exact registered object. No slicing,
insert, pop, or deletion.

**Shared modules — execution is position-based, ownership is
identity-based.** The same child registered in two slots executes
twice in ``forward`` (composition never deduplicates positions), while
``modules()`` / ``named_modules()`` / ``named_parameters()`` /
``state_dict()`` / ``train()`` / ``zero_grad()`` keep the v3.2/v3.3
identity-deduplicated contracts — the first slot is the canonical path
and shared parameters appear once.

**Forward** performs composition only: it calls each child through the
normal ``__call__``/``forward`` contract and adds no node, copy, or
validation of its own — each child validates its own input (a
``NativeLinear`` slot enforces its strictly 2-D contract when reached,
a child's exception propagates unchanged), and the intended usage is
NativeTensor pipelines. An **empty** NativeSequential returns the input
**by identity**: no graph node, no copy, no ownership change. No NumPy
touches forward or backward; the composed autograd graph is exactly
the children's own graphs (no manual Sequential backward exists), so
graph lifetime — one-shot backward, ``retain_graph``, freed-history
errors — and the v3.3/v3.4 mutation boundary (forward → backward →
zero_grad/state update after graph completion; no mutation between
forward and backward) are unchanged.

State keys derive from the slot names — a Linear/ReLU/Linear model has
exactly ``"0.weight"``, ``"0.bias"``, ``"2.weight"``, ``"2.bias"``
(ReLU contributes none); nested sequences nest: ``"0.0.weight"``.

Fully separate from ``tensorforge.nn.Sequential``; float64/cpu only;
no losses, optimizers, or training loops yet.
"""

from .native_module import NativeModule


def _slot_position(name):
    """The int position of a canonical slot name ("0", "1", ...), or
    None if ``name`` is not one (non-strings, signs, leading zeros, and
    non-ASCII digit forms all fail the canonical round-trip)."""
    if isinstance(name, str) and name.isdigit():
        position = int(name)
        if str(position) == name:
            return position
    return None


class NativeSequential(NativeModule):
    """An ordered chain of child NativeModules, executed in slot order.
    See the module docstring for the slot, mutation, shared-module, and
    forward contracts."""

    def __init__(self, *modules):
        super().__init__()
        # Validate every entry before registering any, so a bad later
        # entry never leaves a half-built sequence.
        for position, module in enumerate(modules):
            if not isinstance(module, NativeModule):
                raise TypeError(
                    f"NativeSequential accepts NativeModule children only, "
                    f"got {type(module).__name__} at position {position}"
                )
        for position, module in enumerate(modules):
            self.add_module(str(position), module)

    # -- slot invariant enforcement ------------------------------------
    #
    # Every child-registration path (constructor, append, __setitem__,
    # attribute assignment, add_module) funnels through
    # _register_module_slot; every parameter-registration path funnels
    # through _register_parameter_slot. Overriding the two funnels —
    # plus refusing slot removal below — keeps "registered children" and
    # "execution order" the same thing under all supported mutations.

    def _register_module_slot(self, name, module):
        modules = self.__dict__.get("_modules")
        if modules is not None:
            position = _slot_position(name)
            if position is None:
                raise ValueError(
                    f"NativeSequential children live in contiguous "
                    f"integer-string execution slots; cannot register a "
                    f"child under {name!r}"
                )
            if position > len(modules):
                raise ValueError(
                    f"cannot register NativeSequential slot {name!r}: slots "
                    f"stay contiguous and the next free slot is "
                    f"'{len(modules)}'"
                )
            if module is self:
                raise ValueError(
                    "a NativeSequential cannot contain itself as an "
                    "execution slot"
                )
        super()._register_module_slot(name, module)

    def _register_parameter_slot(self, name, parameter):
        raise TypeError(
            "NativeSequential does not hold direct parameters; wrap the "
            "parameter in a child NativeModule instead"
        )

    def add_module(self, name, module):
        if module is None:
            raise ValueError(
                f"NativeSequential slots cannot be unregistered (removing "
                f"{name!r} would break the contiguous execution order); "
                f"replace the slot with another NativeModule instead"
            )
        super().add_module(name, module)

    def __setattr__(self, name, value):
        if not isinstance(value, NativeModule):
            modules = self.__dict__.get("_modules")
            if modules is not None and name in modules:
                raise ValueError(
                    f"cannot replace NativeSequential slot {name!r} with a "
                    f"non-module value; slots stay contiguous — replace it "
                    f"with another NativeModule instead"
                )
        super().__setattr__(name, value)

    def __delattr__(self, name):
        modules = self.__dict__.get("_modules")
        if modules is not None and name in modules:
            raise ValueError(
                f"cannot delete NativeSequential slot {name!r}; slots stay "
                f"contiguous — replace it with another NativeModule instead"
            )
        super().__delattr__(name)

    # -- ordered-container surface --------------------------------------

    def __len__(self):
        """The number of execution slots."""
        return len(self._modules)

    def __iter__(self):
        """Children in execution order — a shared child registered in
        several slots appears once per slot (iteration mirrors
        execution, not the deduplicated traversal)."""
        return iter(self._modules.values())

    def _normalize_index(self, index):
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(
                f"NativeSequential indices must be int, got "
                f"{type(index).__name__}"
            )
        length = len(self._modules)
        position = index + length if index < 0 else index
        if not 0 <= position < length:
            raise IndexError(
                f"index {index} is out of range for a NativeSequential of "
                f"length {length}"
            )
        return position

    def __getitem__(self, index):
        """The exact child object at ``index`` (real ints only;
        negative indices count from the end)."""
        return self._modules[str(self._normalize_index(index))]

    def __setitem__(self, index, module):
        """Replace the child at an existing ``index`` — the slot keeps
        its name and position; the old child is dropped, never closed
        or mutated."""
        if not isinstance(module, NativeModule):
            raise TypeError(
                f"NativeSequential slots hold NativeModule children only, "
                f"got {type(module).__name__}"
            )
        self.add_module(str(self._normalize_index(index)), module)

    def append(self, module):
        """Validate and register ``module`` under the next contiguous
        slot name. Returns self for chaining (the ``train()``
        convention); traversal and state keys update immediately."""
        if not isinstance(module, NativeModule):
            raise TypeError(
                f"NativeSequential.append requires a NativeModule, got "
                f"{type(module).__name__}"
            )
        self.add_module(str(len(self._modules)), module)
        return self

    # -- execution -------------------------------------------------------

    def forward(self, input):
        """Fold ``input`` through every slot in order and return the
        final result. Pure composition: each child validates its own
        input and contributes its own graph nodes; an empty sequence
        returns ``input`` itself (same object, no node, no copy)."""
        output = input
        for module in self._modules.values():
            output = module(output)
        return output
