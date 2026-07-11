"""NativeModule — the native training stack's module hierarchy core
(Advanced C++ v3.2, the second Phase C milestone; see
docs/backend_experiments.md and docs/native_autograd_design.md §19).

``NativeModule`` is a **Python-side organizational abstraction**: it
holds references to ``NativeParameter`` leaves and child ``NativeModule``
instances, and gives every future native layer (v3.4's ``NativeLinear``
onward), state_dict (v3.3), optimizer, and training loop one
deterministic, identity-based hierarchy contract. It performs no
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
  canonical names are exactly the keys v3.3's state_dict will use.
- ``zero_grad()`` calls each unique parameter's existing
  ``zero_grad()`` (grad → ``None``; data, ``requires_grad``, and graphs
  untouched) and returns ``None``. ``train(mode=True)`` validates
  ``mode`` as a real bool *before* touching any state, then sets
  ``training`` on every unique module (shared/cyclic hierarchies
  visited once) and returns ``self``; ``eval()`` is ``train(False)``.
  Every module starts with ``training = True``; a later mode-dependent
  layer may read the flag — none exists yet.
- ``forward`` raises ``NotImplementedError``; calling the module
  delegates to ``forward``. No hooks, buffers, state_dict,
  serialization, layers, losses, optimizers, or training in this
  milestone.

Lifetime: registries store Python references only. Removing, replacing,
or deleting a registration never invalidates the object — external
references stay usable, and native storage is released only by the
owner's explicit ``close()`` (there is no ``NativeModule.close()``).
"""

from .native_parameter import (
    NativeParameter,
    NativeParameterRegistry,
    _validate_registration_name,
)

# Implementation slots of NativeModule itself. They can never be
# parameter or child-module names — a parameter registered as
# "training" would otherwise shadow the mode flag train() writes.
_RESERVED_NAMES = frozenset({"_parameters", "_modules", "training"})


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
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # Reached only when normal lookup fails: registered parameters
        # and children live in the registries, not in __dict__.
        d = self.__dict__
        parameters = d.get("_parameters")
        if parameters is not None and name in parameters:
            return parameters.get(name)
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
