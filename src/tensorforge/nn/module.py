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


def _named_parameters(value, prefix):
    """Yield (name, Parameter) pairs reachable from ``value``.

    Looks inside child Modules and plain lists/tuples (which is how
    Sequential stores its layers). Names are dotted paths built from
    attribute names and list positions, e.g. "modules.0.weight".
    """
    if isinstance(value, Parameter):
        yield prefix, value
    elif isinstance(value, Module):
        for name, param in value.named_parameters():
            yield f"{prefix}.{name}", param
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _named_parameters(item, f"{prefix}.{index}")


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

        Layers like Dropout behave differently in the two modes;
        layers without mode-dependent behavior simply ignore the flag.
        """
        self.training = mode
        for value in self.__dict__.values():
            for child in _child_modules(value):
                child.train(mode)
        return self

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
        given architecture.
        """
        for attr, value in self.__dict__.items():
            yield from _named_parameters(value, attr)

    def parameters(self):
        """Return all Parameters in this module and its children."""
        return [param for _, param in self.named_parameters()]

    def trainable_parameters(self):
        """Return only the Parameters that are not frozen
        (``requires_grad=True``)."""
        return [param for param in self.parameters() if param.requires_grad]

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
        """Return {name: array} of all parameter values.

        The arrays are copies: mutating them does not touch the model.
        """
        return {name: param.data.copy() for name, param in self.named_parameters()}

    def load_state_dict(self, state_dict, strict=True):
        """Load parameter values from ``state_dict`` in place.

        Values are copied into the existing Parameter objects (nothing
        is replaced), and shapes must match exactly. With ``strict=True``
        the keys must match exactly too; with ``strict=False`` missing
        and unexpected keys are tolerated and matching keys still load.

        Returns {"missing_keys": [...], "unexpected_keys": [...]}.
        """
        params = dict(self.named_parameters())
        missing = [name for name in params if name not in state_dict]
        unexpected = [name for name in state_dict if name not in params]
        if strict and (missing or unexpected):
            raise ValueError(
                f"state_dict keys do not match the model: "
                f"missing {missing}, unexpected {unexpected}"
            )

        for name, param in params.items():
            if name not in state_dict:
                continue
            value = np.asarray(state_dict[name], dtype=param.data.dtype)
            if value.shape != param.data.shape:
                raise ValueError(
                    f"shape mismatch for {name!r}: model expects "
                    f"{param.data.shape}, state_dict has {value.shape}"
                )
            param.data = value.copy()

        return {"missing_keys": missing, "unexpected_keys": unexpected}
