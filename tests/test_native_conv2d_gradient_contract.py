"""Conv2d bias-gradient reduction contract (Phase D, milestone D5).

The Conv2d bias gradient is ``grad_bias[o] = sum over n, oh, ow of
grad_output[n, o, oh, ow]`` (docs/native_cnn_design.md §7.3). The locked
architecture computes it **with no dedicated C++ kernel** — it composes
from the existing, already-tested native ``sum`` reduction. This module is
the D5 *proof* that the existing reduction path yields the correct
``(out_channels,)`` bias gradient; it does **not** add any Python-visible
Conv2d backward operation, autograd, or new capability. D6 will build the
``NativeTensor.conv2d`` autograd node that invokes exactly this sequence.

``NativeTensorCore.sum`` reduces a single axis (or all axes) at a time, so
the multi-axis reduction over ``(0, 2, 3)`` is expressed as the locked,
deterministic sequence:

    grad_output (N, O, oh, ow)
      .sum(axis=0)   -> (O, oh, ow)     # reduce batch first (§7.3)
      .sum(axis=1)   -> (O, ow)         # reduce the (now-leading) spatial axis
      .sum(axis=1)   -> (O,)            # reduce the remaining spatial axis

Each step is a fresh owning contiguous core; the intermediates are closed
after use. The reduction reads only ``grad_output`` (never the input or
weight values), which is why a bias-only backward records no input/weight
version snapshot in the future D6 graph.

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close().

Selector: python -m pytest -q -k native_conv2d_gradient_contract
"""

import numpy as np
import pytest

from tensorforge.backends import cpp

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)


def _bias_grad_via_reductions(grad_output_core):
    """The locked D6 bias-gradient composition over an ``(N, O, oh, ow)``
    grad_output core: ``sum(0).sum(1).sum(1)`` -> ``(O,)``. Returns the
    final owning ``(O,)`` core; closes every intermediate it creates. The
    caller owns (and closes) the returned core and the input core."""
    reduced_batch = grad_output_core.sum(axis=0)      # (O, oh, ow)
    try:
        reduced_h = reduced_batch.sum(axis=1)         # (O, ow)
        try:
            grad_bias = reduced_h.sum(axis=1)         # (O,)
        finally:
            reduced_h.close()
    finally:
        reduced_batch.close()
    return grad_bias


def test_bias_grad_hand_computed():
    # grad_output (N=1, O=2, oh=2, ow=2). grad_bias[o] = sum of that plane.
    g = np.array([[[[1.0, 2.0], [3.0, 4.0]],      # o0 -> 10
                   [[10.0, 20.0], [30.0, 40.0]]]])  # o1 -> 100
    core = cpp.NativeTensorCore.from_array(g)
    grad_bias = _bias_grad_via_reductions(core)
    assert grad_bias.shape == (2,)
    assert grad_bias.to_numpy().tolist() == [10.0, 100.0]
    grad_bias.close()
    core.close()


def test_bias_grad_multiple_batches_and_channels():
    # Multiple batches and output channels accumulate across n, oh, ow.
    g = np.array([
        [[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]],   # sample 0
        [[[3.0, 3.0], [3.0, 3.0]], [[4.0, 4.0], [4.0, 4.0]]],   # sample 1
    ])  # (N=2, O=2, 2, 2)
    core = cpp.NativeTensorCore.from_array(g)
    grad_bias = _bias_grad_via_reductions(core)
    assert grad_bias.shape == (2,)
    # o0: (1*4) + (3*4) = 16 ; o1: (2*4) + (4*4) = 24
    assert grad_bias.to_numpy().tolist() == [16.0, 24.0]
    grad_bias.close()
    core.close()


def test_bias_grad_negative_and_fractional_values():
    rng = np.random.default_rng(17)
    g = np.round(rng.standard_normal((3, 4, 5, 2)), 3)  # N3 O4 5x2
    core = cpp.NativeTensorCore.from_array(g)
    grad_bias = _bias_grad_via_reductions(core)
    assert grad_bias.shape == (4,)  # (out_channels,)
    assert np.allclose(grad_bias.to_numpy(), g.sum(axis=(0, 2, 3)), atol=1e-12)
    grad_bias.close()
    core.close()


def test_bias_grad_matches_stable_conv2d_bias_gradient():
    # Cross-check against the stable framework's Conv2d bias gradient, which
    # is exactly grad_output.sum(axis=(0, 2, 3)).
    from tensorforge.nn import Conv2d
    from tensorforge.tensor import Tensor

    rng = np.random.default_rng(41)
    x = np.round(rng.standard_normal((2, 3, 5, 4)), 3)
    layer = Conv2d(3, 4, (3, 2), stride=(2, 1), padding=(1, 0))
    layer.weight.data = np.round(rng.standard_normal(layer.weight.data.shape), 3)
    layer.bias.data = np.round(rng.standard_normal(4), 3)
    xt = Tensor(x, requires_grad=True)
    out = layer(xt)
    g = np.round(rng.standard_normal(out.data.shape), 3)
    (out * Tensor(g)).sum().backward()
    stable_bias_grad = layer.bias.grad

    core = cpp.NativeTensorCore.from_array(g)
    grad_bias = _bias_grad_via_reductions(core)
    assert grad_bias.shape == (4,)
    assert np.allclose(grad_bias.to_numpy(), stable_bias_grad, atol=1e-9)
    grad_bias.close()
    core.close()


def test_reduction_does_not_mutate_grad_output():
    g = np.round(np.random.default_rng(5).standard_normal((2, 3, 4, 2)), 3)
    core = cpp.NativeTensorCore.from_array(g)
    grad_bias = _bias_grad_via_reductions(core)
    # grad_output is read only — the source core is unchanged and reusable.
    assert np.array_equal(core.to_numpy(), g)
    grad_bias.close()
    core.close()


def test_reduction_intermediates_are_released(monkeypatch):
    # The locked sequence closes every intermediate it allocates: three
    # single-axis sums, of which the two intermediates are closed and only
    # the final (O,) core survives for the caller to close.
    created = []
    original_sum = cpp.NativeTensorCore.sum

    def tracking_sum(self, *args, **kwargs):
        result = original_sum(self, *args, **kwargs)
        created.append(result)
        return result

    monkeypatch.setattr(cpp.NativeTensorCore, "sum", tracking_sum)
    g = np.random.default_rng(9).standard_normal((2, 3, 4, 2))
    core = cpp.NativeTensorCore.from_array(g)
    grad_bias = _bias_grad_via_reductions(core)
    assert len(created) == 3  # sum(0), sum(1), sum(1)
    # The two intermediates are closed; the final result stays open.
    assert created[0]._closed and created[1]._closed
    assert grad_bias is created[2]
    assert not grad_bias._closed
    grad_bias.close()
    core.close()


def test_bias_gradient_reuse_adds_no_new_capability():
    # D5's bias path is pure reuse of the existing sum reduction — no new
    # Conv2d backward operation is advertised at any layer.
    assert "sum" in cpp.TENSOR_CORE_OPS  # the reused, existing op
    for absent in (
        "conv2d_bias_backward", "conv2d_weight_backward",
        "conv2d_input_backward", "conv2d_backward",
    ):
        assert absent not in cpp.TENSOR_CORE_OPS
        assert absent not in cpp.AUTOGRAD_OPS
        assert absent not in cpp._CHECKED_KERNELS
    # The differentiable conv2d op and the module stay unsupported.
    assert "conv2d" not in cpp.AUTOGRAD_OPS
    assert "conv2d" in cpp.UNSUPPORTED
    assert "NativeConv2d" not in cpp.NATIVE_MODULES
