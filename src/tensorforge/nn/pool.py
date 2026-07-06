"""MaxPool2d: downsample NCHW activations by taking window maxima.

Pooling has no learnable parameters. Its job is to shrink the spatial
dimensions while keeping the strongest activation in each window, which
also buys a little translation tolerance: the max survives small shifts
of the feature inside the window.
"""

import numpy as np

from tensorforge.nn.conv import _pair
from tensorforge.nn.module import Module
from tensorforge.tensor import Tensor


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = _pair(kernel_size, "kernel_size", minimum=1)
        # PyTorch convention: no stride means non-overlapping windows.
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = _pair(stride, "stride", minimum=1)
        self.padding = _pair(padding, "padding", minimum=0)

    def forward(self, x):
        if x.data.ndim != 4:
            raise ValueError(
                f"MaxPool2d expects 4-D NCHW input, got shape {x.data.shape}"
            )
        n, c, h, w = x.data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        out_h = (h + 2 * ph - kh) // sh + 1
        out_w = (w + 2 * pw - kw) // sw + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"kernel {self.kernel_size} with stride {self.stride} and "
                f"padding {self.padding} does not fit input {(h, w)}"
            )

        # Pad with -inf, not 0: a padded cell must never win the max.
        padded = np.pad(
            x.data, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
            constant_values=-np.inf,
        )
        out_data = np.zeros((n, c, out_h, out_w))
        # Remember which position won each window (flat index within the
        # kh*kw window) so backward can route gradient there and only there.
        winners = np.zeros((n, c, out_h, out_w), dtype=int)
        for i in range(out_h):
            for j in range(out_w):
                window = padded[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw]
                flat = window.reshape(n, c, kh * kw)
                # argmax returns the FIRST maximum in row-major order, so
                # ties deterministically favor the top-left position.
                idx = flat.argmax(axis=2)
                winners[:, :, i, j] = idx
                out_data[:, :, i, j] = np.take_along_axis(flat, idx[:, :, None], axis=2)[:, :, 0]

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="maxpool2d",
        )

        def _backward():
            grad_padded = np.zeros_like(padded)
            batch_idx, channel_idx = np.indices((n, c))
            for i in range(out_h):
                for j in range(out_w):
                    idx = winners[:, :, i, j]
                    rows = i * sh + idx // kw
                    cols = j * sw + idx % kw
                    # Only the winning position gets this window's gradient;
                    # everything else in the window contributed nothing to
                    # the output, so it gets nothing back.
                    grad_padded[batch_idx, channel_idx, rows, cols] += out.grad[:, :, i, j]
            # Gradient that landed on the -inf padding is discarded here.
            x._accumulate_grad(grad_padded[:, :, ph : ph + h, pw : pw + w])

        out._backward = _backward
        return out
