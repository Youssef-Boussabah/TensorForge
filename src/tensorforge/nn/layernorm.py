"""LayerNorm: normalize each sample over its own feature dimensions.

The counterpart to BatchNorm1d. BatchNorm normalizes each feature
across the batch, so it needs batch statistics, running averages, and
train/eval mode. LayerNorm normalizes each sample over its own last
dimensions — no other sample is involved, so there are no running
statistics, no buffers, and no train/eval difference: the same input
always gives the same output.
"""

import numbers

import numpy as np

from tensorforge.nn.module import Module
from tensorforge.nn.parameter import Parameter


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        if isinstance(normalized_shape, int) and not isinstance(normalized_shape, bool):
            shape = (normalized_shape,)
        elif (
            isinstance(normalized_shape, (tuple, list))
            and len(normalized_shape) > 0
            and all(isinstance(v, int) and not isinstance(v, bool) for v in normalized_shape)
        ):
            shape = tuple(normalized_shape)
        else:
            raise ValueError(
                f"normalized_shape must be a positive int or a non-empty "
                f"tuple of positive ints, got {normalized_shape!r}"
            )
        if any(v <= 0 for v in shape):
            raise ValueError(f"normalized_shape values must be positive, got {shape}")
        if not isinstance(eps, numbers.Real) or isinstance(eps, bool) or eps <= 0:
            raise ValueError(f"eps must be a positive number, got {eps!r}")
        if not isinstance(elementwise_affine, bool):
            raise ValueError(
                f"elementwise_affine must be a bool, got {elementwise_affine!r}"
            )

        self.normalized_shape = shape
        self.eps = float(eps)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(np.ones(shape))
            self.bias = Parameter(np.zeros(shape))

    def forward(self, x):
        k = len(self.normalized_shape)
        if x.data.ndim < k or x.data.shape[-k:] != self.normalized_shape:
            raise ValueError(
                f"LayerNorm{self.normalized_shape} expects input whose last "
                f"{k} dimension(s) are {self.normalized_shape}, got shape "
                f"{x.data.shape}"
            )

        # Built entirely from existing autograd ops, so gradients flow
        # through the mean and variance for free (same trick as
        # BatchNorm's training path — but here the statistics are
        # per-sample, over the trailing dimensions).
        axes = tuple(range(x.data.ndim - k, x.data.ndim))
        n = int(np.prod(self.normalized_shape))
        mean = x.sum(axis=axes, keepdims=True) / n
        centered = x - mean
        var = (centered ** 2).sum(axis=axes, keepdims=True) / n
        x_hat = centered / ((var + self.eps) ** 0.5)

        if self.elementwise_affine:
            return x_hat * self.weight + self.bias
        return x_hat
