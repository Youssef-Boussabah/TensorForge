"""Linear: a fully-connected layer, y = x @ weight + bias."""

import numpy as np

from tensorforge.nn.module import Module
from tensorforge.nn.parameter import Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        # Scale random weights by 1/sqrt(in_features) so the output
        # variance stays roughly independent of the layer width.
        self.weight = Parameter(
            np.random.randn(in_features, out_features) / np.sqrt(in_features)
        )
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
