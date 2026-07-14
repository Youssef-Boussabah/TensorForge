"""NativeModule — the native training stack's module hierarchy core
(Advanced C++ v3.2, the second Phase C milestone; see
docs/backend_experiments.md and docs/native_autograd_design.md §19).

``NativeModule`` is a **Python-side organizational abstraction**: it
holds references to ``NativeParameter`` leaves and child ``NativeModule``
instances, and gives every future native layer (v3.4's ``NativeLinear``
onward), optimizer, and training loop one deterministic, identity-based
hierarchy contract — including the in-memory state-dictionary contract
(v3.3: ``state_dict()`` / ``load_state_dict()``, parameters only). It performs no
numerical computation itself, owns no native storage, and never closes,
copies, or mutates what it registers — it is the native analog of
``tensorforge.nn.Module`` translated to the native identity/ownership
model, and fully separate from it (neither accepts the other).

The registration contract (all tested in tests/test_native_module.py):

- **Assignment registers.** ``module.weight = NativeParameter(...)``
  registers the parameter under ``"weight"``; ``module.block =
  NativeModule()`` registers a child module. Registered objects live in
  the module's registries only (``__getattr__`` resolves them), so there
  is a single source of truth. The exact object is stored — never a
  copy — and object identity is preserved.
- **One category per name; the latest assignment wins.** A name is a
  parameter, a child module, *or* an ordinary attribute. Registering
  validates first (an invalid name or value mutates nothing), then
  evicts the name from the other categories. Replacement within a
  registry preserves the slot's position; moving a name between
  registries appends it to the target; unregistering deletes the slot,
  so registering that name again appends (the v3.1 ordering rules). The
  evicted/replaced object is dropped, never closed or mutated, and no
  gradient state transfers.
- **``module.name = None`` unregisters** a registered parameter or
  child and leaves the attribute readable as ``None``. Assigning any
  ordinary value (a plain ``NativeTensor``, a string, a
  ``tensorforge.Tensor``/``Parameter``/``nn.Module``, ...) likewise
  unregisters the name and stores a normal attribute: only
  ``NativeParameter`` enters the parameter registry and only
  ``NativeModule`` enters the child registry — nothing is wrapped
  implicitly and stable-framework objects never enter native traversal
  (they stay harmless ordinary attributes).
- **Manual APIs mirror assignment.** ``register_parameter(name, p)`` /
  ``add_module(name, m)`` use the same validation and replacement
  semantics; the one deliberate difference is strictness — the explicit
  APIs reject a non-parameter/non-module value with ``TypeError``
  (instead of storing an attribute) and ``None`` raises ``KeyError``
  when nothing is registered under the name (instead of setting a plain
  ``None`` attribute).
- **Reserved names.** ``"_parameters"``, ``"_modules"``, and
  ``"training"`` are implementation slots and cannot be used as
  parameter or child-module names; ``__init__`` creates the registries
  with ``object.__setattr__`` so initialization never routes through
  registration, and registering before ``NativeModule.__init__()`` has
  run raises a clear RuntimeError naming ``super().__init__()``.

Traversal (the future state_dict/optimizer contract):

- ``named_modules()`` is deterministic **pre-order depth-first**:
  ``("", self)`` first, then each child under its registered name, its
  descendants under dot-joined paths, in insertion order. Shared
  modules are **deduplicated by object identity** — the first
  discovered path wins — which also makes direct or indirect module
  cycles terminate safely (cycles are allowed as references; traversal
  simply never revisits a module). ``modules()`` is the same walk
  without names.
- ``named_parameters(prefix="", recurse=True)`` walks that module order
  and yields each unique parameter once under its first-discovered
  dot-joined name — a module's direct parameters before its
  descendants', aliases and shared parameters (direct or through
  children) deduplicated by identity, frozen parameters included, and
  no value equality anywhere. ``recurse=False`` restricts to direct
  parameters. ``parameters(recurse=True)`` returns the deduplicated
  parameter list an optimizer would iterate. These first-discovered
  canonical names are exactly the keys ``state_dict()`` /
  ``load_state_dict()`` use (v3.3): ``state_dict()`` snapshots each
  unique parameter's value into an independent owning graph-free
  ``NativeTensor``, and ``load_state_dict(state_dict, strict=True)``
  copies values back into the existing parameter objects — atomically,
  with strict/non-strict key handling and exact shape/dtype/device
  validation, preserving parameter identity, gradients,
  ``requires_grad``, and training state (see the method docstrings for
  the full contract).
- ``zero_grad()`` calls each unique parameter's existing
  ``zero_grad()`` (grad → ``None``; data, ``requires_grad``, and graphs
  untouched) and returns ``None``. ``train(mode=True)`` validates
  ``mode`` as a real bool *before* touching any state, then sets
  ``training`` on every unique module (shared/cyclic hierarchies
  visited once) and returns ``self``; ``eval()`` is ``train(False)``.
  Every module starts with ``training = True``; a later mode-dependent
  layer may read the flag — none exists yet.
- ``forward`` raises ``NotImplementedError``; calling the module
  delegates to ``forward``.

Buffers (v3.15): a module may hold non-``Parameter`` persistent state
through ``register_buffer(name, tensor, persistent=True)`` — the native
analog of ``tensorforge.nn`` buffers, for the future native BatchNorm
running statistics, RNG state, and similar. Buffers are ordinary owning
``NativeTensor`` objects discovered by ``buffers()`` / ``named_buffers()``
on the same identity-deduplicated, cycle-safe traversal as parameters,
but they are never parameters (absent from ``parameters()``, invisible to
optimizers, gradient-free). Persistent buffers join ``state_dict()`` /
``load_state_dict()`` (and native checkpoints) under their canonical
dotted names, in the same atomic transaction as the parameters and with
their object identity preserved across an in-place restore; non-persistent
buffers are traversed but never serialized. This milestone adds the buffer
*infrastructure* only — no BatchNorm, Dropout, or RNG algorithm.

Lifetime: registries store Python references only. Removing, replacing,
or deleting a registration never invalidates the object — external
references stay usable, and native storage is released only by the
owner's explicit ``close()`` (there is no ``NativeModule.close()``).
"""

from collections import namedtuple
from collections.abc import Mapping

from .native_parameter import (
    NativeParameter,
    NativeParameterRegistry,
    _validate_registration_name,
)
from .native_tensor import NativeTensor, _native_copy

# What load_state_dict returns: the key-compatibility report, immutable.
# ``missing_keys`` are canonical parameter names the input did not
# provide (in canonical traversal order); ``unexpected_keys`` are input
# keys no parameter answers to (in the input mapping's own order). Both
# are always empty under a successful strict=True load.
LoadStateDictResult = namedtuple(
    "LoadStateDictResult", ("missing_keys", "unexpected_keys")
)

# One registered buffer: the NativeTensor plus whether it is persistent
# (serialized in state_dict/checkpoints) or transient (traversed but
# never saved). Stored by the buffer registry (an insertion-ordered
# ``name -> _BufferEntry`` dict) on each module.
_BufferEntry = namedtuple("_BufferEntry", ("tensor", "persistent"))

# Implementation slots of NativeModule itself. They can never be
# parameter, buffer, or child-module names — a parameter registered as
# "training" would otherwise shadow the mode flag train() writes.
_RESERVED_NAMES = frozenset({"_parameters", "_modules", "_buffers", "training"})


class NativeModule:
    """Base class for native layers and models: an ordered parameter
    registry, an ordered child-module registry, and ``training=True``.

    Subclasses call ``super().__init__()`` first, assign
    ``NativeParameter`` / ``NativeModule`` attributes to build the
    hierarchy, and implement ``forward()``. See the module docstring
    for the full registration, traversal, and lifetime contract.
    """

    def __init__(self):
        # object.__setattr__ so the registries themselves never route
        # through the registration logic in __setattr__, which needs
        # them to exist.
        object.__setattr__(self, "_parameters", NativeParameterRegistry())
        object.__setattr__(self, "_modules", {})  # name -> NativeModule, insertion-ordered
        object.__setattr__(self, "_buffers", {})  # name -> _BufferEntry, insertion-ordered
        object.__setattr__(self, "training", True)

    # -- registration -------------------------------------------------

    def _registries(self):
        """Both registries, or a clear error when __init__ has not run
        (e.g. a subclass assigning parameters before super().__init__())."""
        d = self.__dict__
        parameters = d.get("_parameters")
        modules = d.get("_modules")
        if parameters is None or modules is None:
            raise RuntimeError(
                f"cannot register on {type(self).__name__} before "
                f"NativeModule.__init__() has run; call super().__init__() "
                f"first"
            )
        return parameters, modules

    def _register_parameter_slot(self, name, parameter):
        """Make ``name`` a parameter slot holding ``parameter``:
        validate first (nothing mutates on failure), then evict the
        name from the child registry and __dict__. The evicted object,
        if any, is dropped — never closed or mutated."""
        parameters, modules = self._registries()
        if name in _RESERVED_NAMES:
            raise ValueError(
                f"{name!r} is reserved for NativeModule internals and "
                f"cannot be a parameter name"
            )
        parameters.register(name, parameter)  # validates name and type
        modules.pop(name, None)
        self._buffers.pop(name, None)
        self.__dict__.pop(name, None)

    def _register_module_slot(self, name, module):
        """Make ``name`` a child-module slot holding ``module`` — the
        mirror of _register_parameter_slot (the caller has already
        guaranteed ``module`` is a NativeModule)."""
        parameters, modules = self._registries()
        if name in _RESERVED_NAMES:
            raise ValueError(
                f"{name!r} is reserved for NativeModule internals and "
                f"cannot be a child-module name"
            )
        _validate_registration_name(name, "a module name")
        if name in parameters:
            parameters.register(name, None)
        self._buffers.pop(name, None)
        modules[name] = module  # replacement keeps the slot's position
        self.__dict__.pop(name, None)

    def __setattr__(self, name, value):
        if isinstance(value, NativeParameter):
            self._register_parameter_slot(name, value)
        elif isinstance(value, NativeModule):
            self._register_module_slot(name, value)
        else:
            # An ordinary attribute (None included): the name leaves
            # whichever registry held it — latest assignment wins, one
            # category per name — and the old object is dropped, never
            # closed. Plain NativeTensor and the stable framework's
            # Tensor/Parameter/Module deliberately land here: only
            # explicit NativeParameter/NativeModule instances are
            # registered, nothing is wrapped implicitly, and
            # stable-framework objects never enter native traversal.
            d = self.__dict__
            parameters = d.get("_parameters")
            if parameters is not None and name in parameters:
                parameters.register(name, None)
            modules = d.get("_modules")
            if modules is not None:
                modules.pop(name, None)
            buffers = d.get("_buffers")
            if buffers is not None:
                buffers.pop(name, None)
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # Reached only when normal lookup fails: registered parameters,
        # buffers, and children live in the registries, not in __dict__.
        d = self.__dict__
        parameters = d.get("_parameters")
        if parameters is not None and name in parameters:
            return parameters.get(name)
        buffers = d.get("_buffers")
        if buffers is not None and name in buffers:
            return buffers[name].tensor
        modules = d.get("_modules")
        if modules is not None and name in modules:
            return modules[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __delattr__(self, name):
        # Deleting a registered name unregisters it (the object is
        # dropped, never closed); anything else is a plain attribute
        # delete.
        parameters = self.__dict__.get("_parameters")
        if parameters is not None and name in parameters:
            parameters.register(name, None)
            return
        buffers = self.__dict__.get("_buffers")
        if buffers is not None and name in buffers:
            del buffers[name]
            return
        modules = self.__dict__.get("_modules")
        if modules is not None and name in modules:
            del modules[name]
            return
        object.__delattr__(self, name)

    def register_parameter(self, name, parameter):
        """The explicit form of ``module.name = parameter``, with the
        same validation and replacement semantics. Stricter about
        mistakes than assignment: a value that is not a NativeParameter
        raises TypeError (assignment would store an ordinary
        attribute), and ``None`` unregisters — KeyError when no
        parameter is registered under ``name`` — leaving the attribute
        readable as None."""
        if parameter is None:
            parameters, _ = self._registries()
            parameters.register(name, None)  # validates; KeyError if absent
            object.__setattr__(self, name, None)
            return
        # A non-parameter value (a NativeModule, a plain NativeTensor,
        # a framework Tensor/Parameter, ...) is rejected by the
        # registry's own validation inside the slot routine.
        self._register_parameter_slot(name, parameter)

    def add_module(self, name, module):
        """The explicit form of ``module.name = child``, with the same
        validation and replacement semantics — and the same explicit
        strictness as register_parameter: a non-NativeModule value
        raises TypeError, and ``None`` unregisters (KeyError when no
        child is registered under ``name``), leaving the attribute
        readable as None."""
        if module is None:
            _, modules = self._registries()
            _validate_registration_name(name, "a module name")
            if name not in modules:
                raise KeyError(
                    f"cannot unregister {name!r}: no child module is "
                    f"registered under that name"
                )
            del modules[name]
            object.__setattr__(self, name, None)
            return
        if not isinstance(module, NativeModule):
            raise TypeError(
                f"only a NativeModule (or None to unregister) can be added "
                f"as a child module, got {type(module).__name__}"
            )
        self._register_module_slot(name, module)

    def register_buffer(self, name, tensor, persistent=True):
        """Register ``tensor`` as a **buffer** under ``name`` — the native
        analog of ``tensorforge.nn`` buffers (BatchNorm running stats, RNG
        state, and other non-``Parameter`` persistent state a future
        layer will hold).

        A buffer is an ordinary owning ``NativeTensor`` that is discovered
        by ``buffers()`` / ``named_buffers()`` and, when ``persistent``,
        saved and restored by ``state_dict()`` / ``load_state_dict()`` and
        native checkpoints — but it is **never** a parameter: it does not
        appear in ``parameters()``, an optimizer never sees it, and no
        gradient flows through it. Unlike ``NativeParameter`` /
        ``NativeModule`` (which register on plain attribute assignment),
        buffers register only through this explicit call — a plain
        ``NativeTensor`` assigned as an attribute stays an ordinary
        attribute, exactly as before.

        ``tensor`` must be an **open, owning, gradient-free**
        ``NativeTensor`` (``requires_grad=False``); the exact object is
        stored (identity preserved), never a copy, so an in-place
        ``load_state_dict`` restore keeps the same object. ``persistent``
        must be a real bool. ``tensor=None`` unregisters the buffer
        (``KeyError`` if nothing is registered under ``name``), leaving the
        attribute readable as ``None`` — mirroring ``register_parameter``.
        Registration validates everything first (an invalid call mutates
        nothing), then evicts ``name`` from the parameter and child-module
        registries so a name stays one category. The reserved internal
        names are rejected, as for parameters and modules."""
        parameters, modules = self._registries()
        if name in _RESERVED_NAMES:
            raise ValueError(
                f"{name!r} is reserved for NativeModule internals and "
                f"cannot be a buffer name"
            )
        _validate_registration_name(name, "a buffer name")
        if tensor is None:
            if name not in self._buffers:
                raise KeyError(
                    f"cannot unregister {name!r}: no buffer is registered "
                    f"under that name"
                )
            del self._buffers[name]
            object.__setattr__(self, name, None)
            return
        if not isinstance(persistent, bool):
            raise TypeError(
                f"persistent must be a bool, got {type(persistent).__name__}"
            )
        # A NativeParameter is a NativeTensor subclass, but it belongs in
        # the parameter registry — a buffer is deliberately non-trainable.
        if isinstance(tensor, NativeParameter) or not isinstance(
            tensor, NativeTensor
        ):
            raise TypeError(
                f"a buffer must be a plain NativeTensor (not a "
                f"NativeParameter), got {type(tensor).__name__}"
            )
        if tensor.closed:
            raise RuntimeError(
                f"cannot register a closed NativeTensor as buffer {name!r}"
            )
        if not tensor.owns_core:
            raise ValueError(
                f"a buffer must own its storage (got a borrowing view for "
                f"{name!r}); pass an owning NativeTensor, e.g. via "
                f"contiguous_copy()"
            )
        if tensor.requires_grad:
            raise ValueError(
                f"a buffer must not require grad (buffers are "
                f"non-trainable), got requires_grad=True for {name!r}"
            )
        # Validation passed — commit the registration and evict the name
        # from the other categories so it stays exactly one category.
        if name in parameters:
            parameters.register(name, None)
        modules.pop(name, None)
        self.__dict__.pop(name, None)
        self._buffers[name] = _BufferEntry(tensor, persistent)

    # -- traversal ------------------------------------------------------
    #
    # Deterministic pre-order depth-first, deduplicated by object
    # identity (id-keyed, never value equality — the same convention
    # backward()'s graph traversal uses). First discovery wins, which
    # both defines the canonical dotted name a future state_dict will
    # use and makes shared modules and reference cycles terminate
    # safely: a module already visited is never re-entered.

    def named_modules(self, prefix=""):
        """Yield ``(dotted_name, module)`` for this module (as
        ``(prefix, self)``, ``""`` at the root) and every unique
        descendant, pre-order depth-first, first-discovered path
        winning. Cycle-safe."""
        yield from self._named_modules_walk(prefix, set())

    def _named_modules_walk(self, prefix, visited):
        if id(self) in visited:
            return
        visited.add(id(self))
        yield prefix, self
        for name, child in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child._named_modules_walk(child_prefix, visited)

    def modules(self):
        """Every unique module in the hierarchy — self first, then
        descendants in the named_modules() order. Returns a list."""
        return [module for _, module in self.named_modules()]

    def named_parameters(self, prefix="", recurse=True):
        """Yield ``(dotted_name, parameter)`` for each unique parameter
        once: a module's direct parameters before its descendants',
        aliases and shared parameters deduplicated by identity with the
        first-discovered name winning, frozen parameters included.
        ``recurse=False`` restricts to this module's direct
        parameters."""
        if recurse:
            module_items = self.named_modules(prefix)
        else:
            module_items = ((prefix, self),)
        seen = set()
        for module_prefix, module in module_items:
            for name, parameter in module._parameters.named_parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                full_name = f"{module_prefix}.{name}" if module_prefix else name
                yield full_name, parameter

    def parameters(self, recurse=True):
        """Each unique parameter exactly once, in the
        named_parameters() order — the identity-deduplicated traversal
        a future optimizer iterates. Returns a list."""
        return [parameter for _, parameter in self.named_parameters(recurse=recurse)]

    # -- buffers (v3.15) ------------------------------------------------
    #
    # Buffers follow the same identity-deduplicated, cycle-safe traversal
    # as parameters (they ride the same named_modules() walk), so a
    # shared buffer is yielded once under its first-discovered dotted
    # name and reference cycles terminate. Persistence is orthogonal to
    # discovery: named_buffers() yields every buffer; only persistent
    # buffers enter state_dict()/checkpoints.

    def _named_buffer_entries(self, prefix="", recurse=True):
        """Yield ``(dotted_name, tensor, persistent)`` for each unique
        buffer once, in module-then-registration order — the internal
        walk both ``named_buffers`` and ``state_dict`` build on."""
        if recurse:
            module_items = self.named_modules(prefix)
        else:
            module_items = ((prefix, self),)
        seen = set()
        for module_prefix, module in module_items:
            for name, entry in module._buffers.items():
                if id(entry.tensor) in seen:
                    continue
                seen.add(id(entry.tensor))
                full_name = f"{module_prefix}.{name}" if module_prefix else name
                yield full_name, entry.tensor, entry.persistent

    def named_buffers(self, prefix="", recurse=True):
        """Yield ``(dotted_name, tensor)`` for each unique buffer once —
        persistent and non-persistent alike — under its first-discovered
        canonical name, deduplicated by identity and cycle-safe.
        ``recurse=False`` restricts to this module's direct buffers."""
        for name, tensor, _ in self._named_buffer_entries(prefix, recurse):
            yield name, tensor

    def buffers(self, recurse=True):
        """Each unique buffer tensor exactly once, in the
        named_buffers() order. Returns a list. Buffers never appear in
        ``parameters()``."""
        return [tensor for _, tensor in self.named_buffers(recurse=recurse)]

    def _persistent_named_buffers(self):
        """Ordered ``(canonical_name, tensor)`` for the persistent
        buffers only — the ones state_dict()/checkpoints serialize."""
        return [
            (name, tensor)
            for name, tensor, persistent in self._named_buffer_entries()
            if persistent
        ]

    def _state_named_tensors(self):
        """Ordered ``(canonical_name, live_tensor)`` for everything
        ``state_dict()`` snapshots: every unique parameter first, then
        every unique **persistent** buffer. Parameters and persistent
        buffers never share a canonical name (a name is one category
        within its owning module), so the merged key space is unique.
        The single source of truth the native checkpoint validates
        against."""
        return list(self.named_parameters()) + self._persistent_named_buffers()

    # -- state dictionary (v3.3: in-memory, parameters only) -----------
    #
    # The state dictionary is the deterministic in-memory snapshot/load
    # contract future file serialization and checkpoints will consume.
    # This milestone's scope is parameters only: no buffers, optimizer
    # state, training flags, RNG state, files, or archives.

    def state_dict(self):
        """An insertion-ordered ``{canonical_name: NativeTensor}``
        snapshot of every unique parameter's current value.

        Keys are exactly the v3.2 canonical ``named_parameters()``
        names: dot-separated, direct parameters before descendants,
        shared parameters once under their first-discovered path,
        frozen parameters included, cycle-safe, deterministic. Values
        are ordinary graph-free ``requires_grad=False`` NativeTensors,
        each an **independent owning contiguous copy** (computed by the
        native copy path — no NumPy): mutating, replacing, or closing
        the model's parameter afterwards never affects a snapshot, and
        closing a snapshot never affects the model — the snapshot
        outlives the model if the caller keeps it, and the caller
        releases it (``close()`` each value) when done. Gradients,
        ``requires_grad``, training flags, and registrations are
        neither included nor touched.

        Persistent buffers (v3.15) are snapshotted the same way and
        appended after the parameters under their canonical
        ``named_buffers()`` names; non-persistent buffers are skipped.
        A model with no buffers produces exactly the parameter-only
        mapping it always did, so existing consumers are unaffected.

        A closed registered parameter or persistent buffer raises
        RuntimeError naming the key; snapshots already built inside the
        failed call are closed before the error propagates, so a failure
        never leaks native memory or returns a partial mapping. Snapshots
        returned by *earlier* calls are unaffected."""
        state = {}
        try:
            for name, tensor in self._state_named_tensors():
                if tensor.closed:
                    raise RuntimeError(
                        f"cannot snapshot {name!r}: it has been closed"
                    )
                state[name] = NativeTensor._from_core(
                    _native_copy(tensor._require_open())
                )
        except BaseException:
            for snapshot in state.values():
                snapshot.close()
            raise
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Load parameter values from ``state_dict`` **in place** and
        return a ``LoadStateDictResult(missing_keys, unexpected_keys)``.

        Values are copied *into* the existing NativeParameter objects —
        never assigned over them — so every identity-derived contract
        survives loading unchanged: ``id(parameter)``, registrations
        and canonical names, shared-parameter aliasing (one canonical
        key updates the single shared object once; every alias observes
        it; a supplied alias key is *unexpected*), ``requires_grad``
        and frozen state, leaf/graph-free status, and each parameter's
        existing ``grad`` (by identity **and** value; ``None`` stays
        ``None`` — loading never clears, replaces, or accumulates
        gradients). ``training`` flags and traversal order are
        untouched. Each matched canonical parameter's value ``version``
        increments by exactly one on success (v3.7) — a shared
        parameter loads once under its canonical key, so it increments
        once; loading numerically identical values still increments
        (the owned value was replaced); a failed load leaves every
        version unchanged. A graph built *before* loading stays
        memory-safe, and where its backward must read a loaded
        parameter's forward value (multiply/matmul/relu edges) the next
        ``backward()`` raises a deterministic stale-value RuntimeError
        instead of silently using the new value — run forward again
        after loading. Value-independent graph edges remain valid.

        Validation happens entirely **before** any mutation, in this
        order: (1) ``strict`` must be a real bool; (2) ``state_dict``
        must be a mapping (snapshotted once, so exotic mappings cannot
        change mid-load); (3) canonical keys are taken from
        ``named_parameters()``; (4) every provided key must be a str;
        (5) missing/unexpected keys are computed; (6) under
        ``strict=True`` any missing or unexpected key raises ValueError
        reporting **both** lists; (7) every matching value must be an
        open NativeTensor (a NativeParameter is accepted purely as a
        value source — it is copied, and no identity or graph state is
        inherited; ``tensorforge.Tensor``/``Parameter`` and arbitrary
        arrays are rejected) whose shape/dtype/device exactly match the
        open destination parameter — no broadcasting, reshaping,
        casting, or device movement, every error naming the key; (8)
        independent native copies of all matching values are **staged**
        (a failure here closes the staged copies and changes nothing);
        (9) the **commit** swaps each parameter's core for its staged
        copy — pure reference assignments, guarded by a rollback that
        restores every original core if anything interrupts — and only
        after every swap succeeds are the old cores released, exactly
        once. No failure at any stage leaves the model partially
        updated, closes an input tensor, or invalidates existing
        snapshots.

        Persistent buffers (v3.15) participate on equal footing: their
        canonical ``named_buffers()`` keys join the expected set, their
        values are validated, staged, and committed in the **same**
        atomic transaction as the parameters (one rollback covers both),
        and an in-place restore preserves each buffer object's identity.
        Buffers carry no value version, so loading a buffer moves no
        version and makes no graph stale. Non-persistent buffers are
        neither expected nor loaded.

        Under ``strict=False`` matching keys load (with the same full
        validation and atomicity), missing entries keep their values,
        and unexpected keys are ignored; both lists are returned in
        deterministic order (missing: canonical order; unexpected: the
        input mapping's iteration order)."""
        if not isinstance(strict, bool):
            raise TypeError(
                f"strict must be a bool, got {type(strict).__name__}"
            )
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"state_dict must be a mapping, got {type(state_dict).__name__}"
            )
        expected = self._state_named_tensors()
        provided_keys = list(state_dict.keys())
        for key in provided_keys:
            if not isinstance(key, str):
                raise TypeError(
                    f"state_dict keys must be str, got {type(key).__name__}"
                )
        provided = {key: state_dict[key] for key in provided_keys}
        expected_names = {name for name, _ in expected}
        missing = tuple(
            name for name, _ in expected if name not in provided
        )
        unexpected = tuple(
            key for key in provided_keys if key not in expected_names
        )
        if strict and (missing or unexpected):
            raise ValueError(
                f"state_dict keys do not match the module: "
                f"missing {list(missing)}, unexpected {list(unexpected)}"
            )

        # Preflight every matching value completely before staging.
        # ``destination`` is a NativeParameter or a persistent buffer
        # (a plain NativeTensor); both expose the metadata checked here.
        matching = [
            (name, destination, provided[name])
            for name, destination in expected
            if name in provided
        ]
        for name, destination, value in matching:
            if not isinstance(value, NativeTensor):
                raise TypeError(
                    f"state_dict value for {name!r} must be a NativeTensor, "
                    f"got {type(value).__name__}"
                )
            if value.closed:
                raise RuntimeError(
                    f"state_dict value for {name!r} has been closed"
                )
            if destination.closed:
                raise RuntimeError(
                    f"cannot load into {name!r}: it has been closed"
                )
            if value.shape != destination.shape:
                raise ValueError(
                    f"shape mismatch for {name!r}: the module expects "
                    f"{destination.shape}, the state_dict value has "
                    f"{value.shape}"
                )
            if value.dtype != destination.dtype:
                raise ValueError(
                    f"dtype mismatch for {name!r}: the module expects "
                    f"{destination.dtype}, the state_dict value has "
                    f"{value.dtype}"
                )
            if value.device != destination.device:
                raise ValueError(
                    f"device mismatch for {name!r}: the module expects "
                    f"{destination.device}, the state_dict value has "
                    f"{value.device}"
                )

        # Stage an independent owning contiguous native copy of every
        # matching value (any strided/offset view source materializes at
        # its logical shape). Nothing has mutated yet, so a staging
        # failure only has staged copies to release.
        staged = []
        try:
            for name, destination, value in matching:
                staged.append(
                    (destination, _native_copy(value._require_open()))
                )
        except BaseException:
            for _, new_core in staged:
                new_core.close()
            raise

        # Commit: swap every destination's core for its staged copy.
        # A NativeParameter swaps through its validated _adopt_value_core;
        # a buffer (plain NativeTensor) swaps its owning _core directly.
        # Each swap is a pure reference assignment, but the rollback guard
        # still restores every original core if anything (even a
        # KeyboardInterrupt between swaps) interrupts — parameters and
        # buffers roll back together, so a failed load never leaves the
        # model partially updated.
        adopted = []
        try:
            for destination, new_core in staged:
                if isinstance(destination, NativeParameter):
                    old_core = destination._adopt_value_core(new_core)
                else:
                    old_core = destination._require_open()
                    destination._core = new_core
                adopted.append((destination, old_core))
        except BaseException:
            for destination, old_core in adopted:
                destination._core = old_core
            for _, new_core in staged:
                new_core.close()
            raise
        # Fully committed. Count one value replacement per loaded
        # parameter (v3.7) — pure int increments that cannot fail, done
        # before the closes so versions and released storage can never
        # disagree — then release each replaced core exactly once.
        # Buffers carry no version, so they are skipped here. Versions
        # move only at this point, after every swap has succeeded, so the
        # rollback above never has anything to decrement: a failed load
        # leaves every version exactly as it was.
        for destination, _ in adopted:
            if isinstance(destination, NativeParameter):
                destination._version += 1
        for _, old_core in adopted:
            old_core.close()
        return LoadStateDictResult(missing, unexpected)

    # -- gradients and mode -----------------------------------------------

    def zero_grad(self):
        """Clear every unique parameter's gradient to ``None`` via the
        parameter's own ``zero_grad()`` (shared parameters visited
        once; frozen parameters included harmlessly). Touches nothing
        else — no data, no ``requires_grad``, no graphs, no closing, no
        training state. Returns None."""
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self, mode=True):
        """Set ``training`` to ``mode`` on this module and every unique
        descendant (shared modules and cycles visited once). ``mode``
        must be a real bool — anything else raises TypeError before any
        state changes. Returns self for chaining."""
        if not isinstance(mode, bool):
            raise TypeError(
                f"mode must be a bool, got {type(mode).__name__}"
            )
        for module in self.modules():
            module.training = mode
        return self

    def eval(self):
        """Equivalent to ``train(False)``. Returns self."""
        return self.train(False)

    # -- call protocol ------------------------------------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        # Metadata only, and defensive: readable even on a subclass
        # instance whose __init__ has not run yet.
        d = self.__dict__
        parameters = d.get("_parameters")
        modules = d.get("_modules")
        if parameters is None or modules is None:
            return f"{type(self).__name__}(uninitialized)"
        return (
            f"{type(self).__name__}("
            f"parameters={[name for name, _ in parameters.named_parameters()]}, "
            f"modules={list(modules)}, training={d.get('training')})"
        )
