"""Module: the base class every neural-network building block inherits from."""

import numpy as np

from tensorforge.nn.parameter import Parameter


def count_parameters(model, trainable_only=True):
    """Convenience wrapper for ``model.num_parameters(...)``."""
    return model.num_parameters(trainable_only=trainable_only)


def model_summary(model):
    """Convenience wrapper for ``model.summary()``."""
    return model.summary()


def _child_modules(value):
    """Yield every Module directly reachable from ``value`` (either a
    Module itself or a list/tuple containing Modules)."""
    if isinstance(value, Module):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _child_modules(item)


# --- identity-aware, cycle-safe recursive traversal -------------------------
#
# Recursive walks thread two id() sets so a Module graph with shared
# children/parameters or a reference cycle is handled correctly:
# ``visited`` (module identities) makes a cycle terminate and a shared
# child expand only once; ``seen`` (leaf identities) yields each unique
# Parameter/buffer once, under its first-encountered dotted name.
# Deterministic first-encounter order matches attribute/list order, so
# a plain tree produces exactly the names it always did.


def _named_parameters(value, prefix, visited, seen):
    """Yield (name, Parameter) pairs reachable from ``value``.

    Looks inside child Modules and plain lists/tuples (which is how
    Sequential stores its layers). Names are dotted paths built from
    attribute names and list positions, e.g. "modules.0.weight". Shared
    parameters yield once (first name wins); cycles terminate.
    """
    if isinstance(value, Parameter):
        if id(value) in seen:
            return
        seen.add(id(value))
        yield prefix, value
    elif isinstance(value, Module):
        yield from value._named_parameters(prefix, visited, seen)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _named_parameters(item, f"{prefix}.{index}", visited, seen)


def _named_buffers(value, prefix, visited, seen):
    """Like _named_parameters, but for the buffers of child Modules."""
    if isinstance(value, Module):
        yield from value._named_buffers(prefix, visited, seen)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _named_buffers(item, f"{prefix}.{index}", visited, seen)


class Module:
    """Base class for layers and models.

    Subclasses implement ``forward()``. Calling the module like a
    function runs it: ``y = model(x)``.
    """

    # Class-level default: every module starts in training mode.
    # train()/eval() set an instance attribute that shadows this, so
    # subclasses don't need to call any __init__ chain for it.
    training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        """Put this module and all its children in training mode
        (or evaluation mode with ``mode=False``). Returns self.

        ``mode`` must be a real ``bool`` — ``True`` or ``False``.
        Passing a truthy/falsy non-bool (a string, an int, a list)
        raises ``TypeError`` before any state changes, so an accidental
        ``model.train("eval")`` cannot silently leave the model in
        training mode. Propagation is identity-aware and cycle-safe:
        each unique child module is set once even when shared or part of
        a reference cycle.

        Layers like Dropout behave differently in the two modes;
        layers without mode-dependent behavior simply ignore the flag.
        """
        if not isinstance(mode, bool):
            raise TypeError(f"train(mode) requires a bool, got {mode!r}")
        self._train(mode, set())
        return self

    def _train(self, mode, visited):
        if id(self) in visited:
            return
        visited.add(id(self))
        self.training = mode
        for value in self.__dict__.values():
            for child in _child_modules(value):
                child._train(mode, visited)

    def eval(self):
        """Equivalent to ``train(False)``. Returns self."""
        return self.train(False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def named_parameters(self):
        """Yield (name, Parameter) for this module and its children.

        Names follow attribute order, so they are deterministic for a
        given architecture. Each unique Parameter is yielded once even
        when it is shared (tied weights) or reached through more than
        one path, and reference cycles terminate safely — see
        :meth:`_named_parameters`.
        """
        yield from self._named_parameters("", set(), set())

    def _named_parameters(self, prefix, visited, seen):
        """Identity-aware, cycle-safe recursion backing
        ``named_parameters``. ``visited`` holds already-expanded module
        identities (so a cycle or a shared child stops); ``seen`` holds
        already-yielded Parameter identities (so a shared/tied Parameter
        yields once, under its first-encountered dotted name)."""
        if id(self) in visited:
            return
        visited.add(id(self))
        for attr, value in self.__dict__.items():
            name = f"{prefix}.{attr}" if prefix else attr
            yield from _named_parameters(value, name, visited, seen)

    def parameters(self):
        """Return all Parameters in this module and its children."""
        return [param for _, param in self.named_parameters()]

    def trainable_parameters(self):
        """Return only the Parameters that are not frozen
        (``requires_grad=True``)."""
        return [param for param in self.parameters() if param.requires_grad]

    def named_buffers(self):
        """Yield (name, array) for this module's buffers and its children's.

        Buffers are non-trainable NumPy arrays a module wants saved and
        loaded with its state — e.g. BatchNorm's running statistics. A
        module declares them by listing attribute names in
        ``self._buffers``. Never optimized, never given gradients. Each
        unique buffer array is yielded once (first name wins) and cycles
        terminate safely, mirroring ``named_parameters``.
        """
        yield from self._named_buffers("", set(), set())

    def _named_buffers(self, prefix, visited, seen):
        """Identity-aware, cycle-safe recursion backing ``named_buffers``
        (see ``_named_parameters`` for the ``visited``/``seen`` roles)."""
        if id(self) in visited:
            return
        visited.add(id(self))
        for attr in getattr(self, "_buffers", ()):
            buf = getattr(self, attr)
            if id(buf) in seen:
                continue
            seen.add(id(buf))
            name = f"{prefix}.{attr}" if prefix else attr
            yield name, buf
        for attr, value in self.__dict__.items():
            name = f"{prefix}.{attr}" if prefix else attr
            yield from _named_buffers(value, name, visited, seen)

    def buffers(self):
        """Return all buffer arrays in this module and its children."""
        return [buf for _, buf in self.named_buffers()]

    def zero_grad(self):
        """Clear stored gradients so the next backward() starts fresh."""
        for param in self.parameters():
            param.grad = None

    def num_parameters(self, trainable_only=True):
        """Total number of scalar values across this module's parameters.

        By default counts only parameters with ``requires_grad=True``;
        pass ``trainable_only=False`` to count everything.
        """
        return sum(
            param.data.size
            for _, param in self.named_parameters()
            if param.requires_grad or not trainable_only
        )

    def summary(self):
        """Return a readable multi-line description of the parameters.

        Lists every parameter's name, shape, size, and trainability,
        plus totals. Returns a string (does not print) and never runs
        a forward pass.
        """
        rows = [
            (
                name,
                str(param.data.shape),
                param.data.size,
                "yes" if param.requires_grad else "no",
            )
            for name, param in self.named_parameters()
        ]
        total = sum(size for _, _, size, _ in rows)
        trainable = sum(size for _, _, size, flag in rows if flag == "yes")

        lines = ["TensorForge Model Summary", f"Model: {type(self).__name__}", ""]
        if rows:
            name_w = max(len("Name"), *(len(name) for name, _, _, _ in rows))
            shape_w = max(len("Shape"), *(len(shape) for _, shape, _, _ in rows))
            size_w = max(len("Params"), *(len(str(size)) for _, _, size, _ in rows))
            lines.append(
                f"{'Name':<{name_w}}  {'Shape':<{shape_w}}  {'Params':<{size_w}}  Trainable"
            )
            for name, shape, size, flag in rows:
                lines.append(f"{name:<{name_w}}  {shape:<{shape_w}}  {size:<{size_w}}  {flag}")
            lines.append("")
        lines.append(f"Total params: {total}")
        lines.append(f"Trainable params: {trainable}")
        lines.append(f"Non-trainable params: {total - trainable}")
        return "\n".join(lines)

    def state_dict(self):
        """Return {name: array} of all parameter and buffer values.

        The arrays are copies: mutating them does not touch the model.
        """
        state = {name: param.data.copy() for name, param in self.named_parameters()}
        for name, buf in self.named_buffers():
            state[name] = buf.copy()
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Load parameter and buffer values from ``state_dict`` in place.

        Values are copied into the existing Parameter objects and buffer
        arrays (nothing is replaced), and shapes must match exactly.
        With ``strict=True`` the keys must match exactly too; with
        ``strict=False`` missing and unexpected keys are tolerated and
        matching keys still load.

        The load is **atomic — fully validate, then commit**. Every key,
        value type, and shape (and dtype conversion) is checked and every
        replacement array is prepared *before* any live Parameter or
        buffer is touched. If any validation fails — including a shape
        mismatch on a later entry — no Parameter or buffer is mutated and
        their object identities are preserved, so the model is never left
        partially loaded. The commit itself is guarded by a rollback that
        restores every original value if a commit step is interrupted.

        Returns {"missing_keys": [...], "unexpected_keys": [...]}.
        """
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        entries = {**params, **buffers}
        missing = [name for name in entries if name not in state_dict]
        unexpected = [name for name in state_dict if name not in entries]
        if strict and (missing or unexpected):
            raise ValueError(
                f"state_dict keys do not match the model: "
                f"missing {missing}, unexpected {unexpected}"
            )

        # -- validate + stage: prepare every replacement value with no
        # mutation, so any error (a later shape mismatch included) leaves
        # the whole model untouched.
        param_updates = []  # (param, new_data)
        for name, param in params.items():
            if name not in state_dict:
                continue
            value = np.asarray(state_dict[name], dtype=param.data.dtype)
            if value.shape != param.data.shape:
                raise ValueError(
                    f"shape mismatch for {name!r}: model expects "
                    f"{param.data.shape}, state_dict has {value.shape}"
                )
            param_updates.append((param, value.copy()))

        buffer_updates = []  # (buffer, new_data)
        for name, buf in buffers.items():
            if name not in state_dict:
                continue
            value = np.asarray(state_dict[name], dtype=buf.dtype)
            if value.shape != buf.shape:
                raise ValueError(
                    f"shape mismatch for {name!r}: model expects "
                    f"{buf.shape}, state_dict has {value.shape}"
                )
            buffer_updates.append((buf, value))

        # -- commit: all validation passed. Assignments cannot fail after
        # validation, but a rollback still restores every original value
        # (parameter reference and buffer contents) if anything interrupts
        # mid-commit, so an interrupted load never leaves a partial model.
        done_params = []  # (param, old_data)
        done_buffers = []  # (buffer, old_contents)
        try:
            for param, new_data in param_updates:
                done_params.append((param, param.data))
                param.data = new_data
            for buf, new_data in buffer_updates:
                done_buffers.append((buf, buf.copy()))
                buf[...] = new_data  # in place so the owning module keeps its array
        except BaseException:
            for param, old_data in done_params:
                param.data = old_data
            for buf, old_contents in done_buffers:
                buf[...] = old_contents
            raise

        return {"missing_keys": missing, "unexpected_keys": unexpected}
