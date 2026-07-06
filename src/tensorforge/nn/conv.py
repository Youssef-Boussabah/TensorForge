"""Conv2d: 2-D convolution over NCHW inputs.

Deliberately the naive version: an explicit loop over output positions,
with the channel/kernel contraction done by einsum. Slow but readable —
each line maps directly onto the definition of convolution.
"""

import numpy as np

from tensorforge.nn.module import Module
from tensorforge.nn.parameter import Parameter
from tensorforge.tensor import Tensor


def _pair(value, name, minimum):
    """Accept an int or a pair of ints; validate against ``minimum``."""
    if isinstance(value, int) and not isinstance(value, bool):
        pair = (value, value)
    elif (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        pair = tuple(value)
    else:
        raise ValueError(f"{name} must be an int or a pair of ints, got {value!r}")
    if any(v < minimum for v in pair):
        raise ValueError(f"{name} values must be >= {minimum}, got {pair}")
    return pair


class Conv2d(Module):
    """Slide out_channels learned kernels over an (N, C, H, W) input.

    Each output channel is one kernel of shape (in_channels, kh, kw)
    dotted against every spatial window of the input, plus a bias.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        for name, value in (("in_channels", in_channels), ("out_channels", out_channels)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size, "kernel_size", minimum=1)
        self.stride = _pair(stride, "stride", minimum=1)
        self.padding = _pair(padding, "padding", minimum=0)

        kh, kw = self.kernel_size
        # Same idea as Linear's init: scale by 1/sqrt(fan_in), where a
        # conv output value sums over in_channels * kh * kw inputs.
        fan_in = in_channels * kh * kw
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, kh, kw) / np.sqrt(fan_in)
        )
        self.bias = Parameter(np.zeros(out_channels)) if bias else None

    def forward(self, x):
        if x.data.ndim != 4:
            raise ValueError(
                f"Conv2d expects 4-D NCHW input, got shape {x.data.shape}"
            )
        n, c, h, w = x.data.shape
        if c != self.in_channels:
            raise ValueError(
                f"Conv2d expects {self.in_channels} input channels, got {c}"
            )
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

        padded = np.pad(x.data, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
        weight = self.weight.data
        out_data = np.zeros((n, self.out_channels, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                # The (N, C, kh, kw) window under kernel position (i, j),
                # contracted against every kernel at once: -> (N, O).
                window = padded[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw]
                out_data[:, :, i, j] = np.einsum("nckl,ockl->no", window, weight)
        if self.bias is not None:
            out_data += self.bias.data.reshape(1, -1, 1, 1)

        children = (x, self.weight) + ((self.bias,) if self.bias is not None else ())
        out = Tensor(
            out_data,
            requires_grad=any(child.requires_grad for child in children),
            _children=children,
            _op="conv2d",
        )

        def _backward():
            grad_out = out.grad  # (N, O, out_h, out_w)
            grad_padded = np.zeros_like(padded)
            grad_weight = np.zeros_like(weight)
            for i in range(out_h):
                for j in range(out_w):
                    g = grad_out[:, :, i, j]  # (N, O)
                    rows = slice(i * sh, i * sh + kh)
                    cols = slice(j * sw, j * sw + kw)
                    # Each weight saw this window scaled by its output's
                    # gradient; each input value saw every kernel value
                    # it was multiplied with.
                    grad_weight += np.einsum("no,nckl->ockl", g, padded[:, :, rows, cols])
                    grad_padded[:, :, rows, cols] += np.einsum("no,ockl->nckl", g, weight)
            # Drop the gradient that landed on the zero padding.
            x._accumulate_grad(grad_padded[:, :, ph : ph + h, pw : pw + w])
            self.weight._accumulate_grad(grad_weight)
            if self.bias is not None:
                # The bias is added to every output position.
                self.bias._accumulate_grad(grad_out.sum(axis=(0, 2, 3)))

        out._backward = _backward
        return out
