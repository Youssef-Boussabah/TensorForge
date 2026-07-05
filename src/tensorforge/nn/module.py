"""Module: the base class every neural-network building block inherits from."""

from tensorforge.nn.parameter import Parameter


def _collect_parameters(value):
    """Yield every Parameter reachable from ``value``.

    Looks inside child Modules and plain lists/tuples (which is how
    Sequential stores its layers).
    """
    if isinstance(value, Parameter):
        yield value
    elif isinstance(value, Module):
        yield from value.parameters()
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _collect_parameters(item)


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

    def parameters(self):
        """Return all Parameters in this module and its children."""
        params = []
        for value in self.__dict__.values():
            params.extend(_collect_parameters(value))
        return params

    def zero_grad(self):
        """Clear stored gradients so the next backward() starts fresh."""
        for param in self.parameters():
            param.grad = None
