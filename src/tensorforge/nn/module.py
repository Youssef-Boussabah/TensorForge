"""Module: the base class every neural-network building block inherits from."""

import numpy as np

from tensorforge.nn.parameter import Parameter


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

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

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

    def zero_grad(self):
        """Clear stored gradients so the next backward() starts fresh."""
        for param in self.parameters():
            param.grad = None

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
