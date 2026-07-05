"""Parameter: a Tensor that is meant to be trained."""

from tensorforge.tensor import Tensor


class Parameter(Tensor):
    """A trainable Tensor.

    Being a distinct class lets ``Module.parameters()`` find trainable
    tensors by type, and it always requires gradients.
    """

    def __init__(self, data):
        super().__init__(data, requires_grad=True)

    def __repr__(self):
        return f"Parameter({self.data})"
