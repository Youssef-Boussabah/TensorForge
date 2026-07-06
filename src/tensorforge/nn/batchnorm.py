"""BatchNorm1d: normalize each feature across the batch.

BatchNorm has two kinds of state:

- trainable parameters ``gamma`` (scale) and ``beta`` (shift), which
  the optimizer updates like any other Parameter;
- buffers ``running_mean`` and ``running_var``, exponential moving
  averages of the batch statistics. They are saved/loaded with the
  model but never optimized — eval mode uses them in place of batch
  statistics, so a single sample can be normalized consistently.
"""

import numbers

import numpy as np

from tensorforge.nn.module import Module
from tensorforge.nn.parameter import Parameter


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        if not isinstance(num_features, int) or isinstance(num_features, bool) or num_features <= 0:
            raise ValueError(f"num_features must be a positive int, got {num_features!r}")
        if not isinstance(eps, numbers.Real) or eps <= 0:
            raise ValueError(f"eps must be a positive number, got {eps!r}")
        if not isinstance(momentum, numbers.Real) or not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum!r}")

        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.gamma = Parameter(np.ones(num_features))
        self.beta = Parameter(np.zeros(num_features))
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self._buffers = ("running_mean", "running_var")

    def forward(self, x):
        if x.data.ndim != 2 or x.data.shape[1] != self.num_features:
            raise ValueError(
                f"BatchNorm1d({self.num_features}) expects input shaped "
                f"(batch, {self.num_features}), got {x.data.shape}"
            )

        if self.training:
            # Normalize with the statistics of *this* batch, built from
            # existing autograd ops so gradients flow through the mean
            # and variance too (batchnorm's full backward, for free).
            n = x.data.shape[0]
            mean = x.sum(axis=0, keepdims=True) / n
            centered = x - mean
            var = (centered ** 2).sum(axis=0, keepdims=True) / n
            x_hat = centered / ((var + self.eps) ** 0.5)

            # Update the running averages as plain NumPy, outside the
            # graph: buffers never take gradients. They only matter for
            # future eval calls, not for this normalization.
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * x.data.mean(axis=0)
            self.running_var = (1 - m) * self.running_var + m * x.data.var(axis=0)
        else:
            # Eval: normalize with the stored running statistics, which
            # are constants — gradients only flow through x itself.
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return x_hat * self.gamma + self.beta
