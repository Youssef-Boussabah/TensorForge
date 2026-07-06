"""Dropout: randomly silence activations during training."""

import numbers

import numpy as np

from tensorforge.nn.module import Module


class Dropout(Module):
    """Inverted dropout, a regularization layer.

    In training mode each activation is dropped (set to 0) with
    probability ``p``, and the survivors are scaled by 1/(1-p) so the
    layer's expected output stays the same as its input — that scaling
    is what makes it "inverted" dropout, and it's why nothing special
    has to happen at evaluation time: in eval mode the layer is simply
    the identity.
    """

    def __init__(self, p=0.5, seed=None):
        if not isinstance(p, numbers.Real):
            raise ValueError(f"p must be a number in [0, 1), got {p!r}")
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must satisfy 0 <= p < 1, got {p}")
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        # Keep each activation with probability 1 - p. The mask is a
        # plain constant as far as autograd is concerned, so dropped
        # positions get zero gradient and kept ones get 1/(1-p).
        keep = (self.rng.random(x.data.shape) >= self.p).astype(np.float64)
        return x * (keep / (1.0 - self.p))
