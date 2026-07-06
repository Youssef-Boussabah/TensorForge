"""Flatten: collapse everything after the batch dimension."""

from tensorforge.nn.module import Module


class Flatten(Module):
    """Turn (batch, ...) into (batch, features).

    The usual bridge between image-shaped layers like Conv2d, which
    work on (N, C, H, W), and Linear, which wants (N, features).
    Already-flat (N, features) input passes through unchanged.
    """

    def forward(self, x):
        if x.data.ndim < 2:
            raise ValueError(
                f"Flatten expects at least 2-D input (batch, ...), "
                f"got shape {x.data.shape}"
            )
        return x.reshape(x.data.shape[0], -1)
