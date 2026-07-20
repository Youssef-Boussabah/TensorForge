"""Tests for NativeConv2d — the trainable native convolution module
(Advanced C++ Phase D, milestone D7).

NativeConv2d is a NativeModule holding an OIHW weight NativeParameter (and
optional (O,) bias NativeParameter) whose forward is the existing
differentiable ``NativeTensor.conv2d`` primitive (D6). It adds no new C++
kernel, no C ABI symbol, no custom module backward, and no new autograd
logic — the module layer only holds parameters, initializes them with a
deterministic uniform conv fan-in, validates its 4-D NCHW input, and
delegates. These tests cover constructor validation, parameter
shape/initialization, forward parity and delegation, inherited autograd,
NativeModule traversal, state_dict/checkpoint behavior, and optimizer
compatibility. MaxPool2d stays unimplemented.

Selector: python -m pytest -q -k native_conv2d_module
"""

import math

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeConv2d,
    NativeFlatten,
    NativeLinear,
    NativeModule,
    NativeMSELoss,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)


# ======================================================================
# Helpers
# ======================================================================


def _np_conv2d(x, w, b, stride, padding):
    """A plain NumPy cross-correlation reference (no PyTorch)."""
    sh, sw = stride
    ph, pw = padding
    n, c, h, wid = x.shape
    o, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (wid + 2 * pw - kw) // sw + 1
    out = np.zeros((n, o, out_h, out_w))
    for ni in range(n):
        for oi in range(o):
            for i in range(out_h):
                for j in range(out_w):
                    region = xp[ni, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
                    out[ni, oi, i, j] = np.sum(region * w[oi])
    if b is not None:
        out = out + np.asarray(b).reshape(1, o, 1, 1)
    return out


def _overwrite(param, values):
    """Replace a parameter's value through the supported mutation API."""
    src = NativeTensor.from_array(np.asarray(values, dtype=float))
    param.copy_value_(src)
    src.close()


# ======================================================================
# Constructor and validation
# ======================================================================


@needs_native
def test_default_construction():
    conv = NativeConv2d(3, 4, 3)
    assert isinstance(conv, NativeModule)
    assert conv.in_channels == 3
    assert conv.out_channels == 4
    assert conv.kernel_size == (3, 3)
    assert conv.stride == (1, 1)
    assert conv.padding == (0, 0)
    assert conv.weight.shape == (4, 3, 3, 3)
    assert conv.bias.shape == (4,)
    assert conv.training is True


@needs_native
def test_rectangular_kernel_and_tuple_stride_padding():
    conv = NativeConv2d(2, 5, (2, 4), stride=(2, 1), padding=(1, 0))
    assert conv.kernel_size == (2, 4)
    assert conv.stride == (2, 1)
    assert conv.padding == (1, 0)
    assert conv.weight.shape == (5, 2, 2, 4)


@needs_native
def test_int_forms_normalize_to_pairs():
    conv = NativeConv2d(1, 1, 3, stride=2, padding=1)
    assert conv.kernel_size == (3, 3)
    assert conv.stride == (2, 2)
    assert conv.padding == (1, 1)


@needs_native
def test_bias_enabled_and_disabled():
    with_bias = NativeConv2d(2, 3, 3, bias=True)
    assert isinstance(with_bias.bias, NativeParameter)
    no_bias = NativeConv2d(2, 3, 3, bias=False)
    assert no_bias.bias is None
    assert "bias" not in dict(no_bias.named_parameters())


@needs_native
@pytest.mark.parametrize("bad", [3.0, "3", None, 2 + 1j])
def test_invalid_channel_types_raise_typeerror(bad):
    with pytest.raises(TypeError):
        NativeConv2d(bad, 4, 3)
    with pytest.raises(TypeError):
        NativeConv2d(3, bad, 3)


@needs_native
@pytest.mark.parametrize("bad", [0, -1, -5])
def test_invalid_channel_values_raise_valueerror(bad):
    with pytest.raises(ValueError):
        NativeConv2d(bad, 4, 3)
    with pytest.raises(ValueError):
        NativeConv2d(3, bad, 3)


@needs_native
def test_boolean_channels_rejected():
    with pytest.raises(TypeError):
        NativeConv2d(True, 4, 3)
    with pytest.raises(TypeError):
        NativeConv2d(3, False, 3)


@needs_native
@pytest.mark.parametrize("bad", [0, -1, (1,), (1, 2, 3), (1, 0), (2.0, 2),
                                 True, (True, 2), "3", 3.0])
def test_invalid_kernel_forms_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        NativeConv2d(2, 2, bad)


@needs_native
@pytest.mark.parametrize("bad", [0, -1, (1,), (0, 1), (2.0, 1), True, "1"])
def test_invalid_stride_forms_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        NativeConv2d(2, 2, 3, stride=bad)


@needs_native
@pytest.mark.parametrize("bad", [-1, (-1, 0), (0.0, 1), (1,), True, "0"])
def test_invalid_padding_forms_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        NativeConv2d(2, 2, 3, padding=bad)


@needs_native
def test_padding_zero_is_valid():
    conv = NativeConv2d(2, 2, 3, padding=0)
    assert conv.padding == (0, 0)


@needs_native
@pytest.mark.parametrize("bad", [1, 0, "yes", None])
def test_invalid_bias_type_rejected(bad):
    with pytest.raises(TypeError):
        NativeConv2d(2, 2, 3, bias=bad)


@needs_native
@pytest.mark.parametrize("bad", [1, 0, "yes", None])
def test_invalid_requires_grad_type_rejected(bad):
    with pytest.raises(TypeError):
        NativeConv2d(2, 2, 3, requires_grad=bad)


@needs_native
@pytest.mark.parametrize("bad", [1.0, "0", True, [1]])
def test_invalid_seed_type_rejected(bad):
    with pytest.raises(TypeError):
        NativeConv2d(2, 2, 3, seed=bad)


# ======================================================================
# Constructor partial-allocation cleanup (deterministic, not GC-reliant)
# ======================================================================


def _live_parameter_count(created):
    """How many spied NativeParameters still hold open native storage —
    a deterministic live-storage tally that never waits on GC."""
    return sum(1 for p in created if not p.closed)


@needs_fault_injection
def test_bias_allocation_failure_releases_weight_deterministically(monkeypatch):
    # A bias-enabled NativeConv2d allocates the weight storage first, then
    # the bias storage. Force *only* the bias allocation to fail and prove
    # the already-created weight is closed deterministically, so a failed
    # construction leaves the live-storage count exactly at its baseline.
    import tensorforge.experimental.native_conv2d as native_conv2d

    created = []
    real_parameter = native_conv2d.NativeParameter

    def spy(*args, **kwargs):
        # Records each real NativeParameter the constructor builds; the
        # list keeps them alive so ``closed`` reflects explicit close(),
        # never garbage collection.
        param = real_parameter(*args, **kwargs)
        created.append(param)
        return param

    monkeypatch.setattr(native_conv2d, "NativeParameter", spy)

    baseline = _live_parameter_count(created)  # 0 — nothing built yet

    # Each NativeParameter-from-array is exactly one native allocation, so
    # the weight is allocation #1 (succeeds) and the bias is #2 (fails).
    cpp._arm_alloc_failure(2)
    try:
        with pytest.raises(MemoryError):
            NativeConv2d(2, 3, 3, bias=True)
    finally:
        cpp._arm_alloc_failure(0)  # disarm regardless of outcome
        cpp._require_library().tf_clear_error()

    # Only the weight was ever constructed; the bias never became an object.
    assert len(created) == 1
    weight = created[0]
    # The already-created weight storage was released deterministically...
    assert weight.closed is True
    # ...so the live-storage count is back at the exact baseline.
    assert _live_parameter_count(created) == baseline

    # No partially constructed NativeConv2d escaped: the name never bound,
    # and constructing again cannot see stale state.
    assert "conv" not in dir()

    # The injected error slot did not contaminate a later native op.
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    probe = cpp.NativeTensorCore.from_array(np.ones((2, 2)))
    assert np.array_equal(probe.relu().to_numpy(), np.ones((2, 2)))
    probe.close()

    # A subsequent construction of the same module succeeds and releases
    # its own storage normally.
    conv = NativeConv2d(2, 3, 3, bias=True, seed=0)
    assert conv.weight.closed is False and conv.bias.closed is False
    out = conv(NativeTensor.from_array(np.zeros((1, 2, 5, 5))))
    assert out.shape == (1, 3, 3, 3)
    out.close()
    conv.weight.close()
    conv.bias.close()


@needs_fault_injection
def test_bias_disabled_allocates_only_weight_no_cleanup_needed(monkeypatch):
    # bias=False performs exactly one native allocation (the weight) and
    # needs no bias-cleanup path: the single parameter stays open.
    import tensorforge.experimental.native_conv2d as native_conv2d

    created = []
    real_parameter = native_conv2d.NativeParameter

    def spy(*args, **kwargs):
        param = real_parameter(*args, **kwargs)
        created.append(param)
        return param

    monkeypatch.setattr(native_conv2d, "NativeParameter", spy)

    conv = NativeConv2d(2, 3, 3, bias=False, seed=0)
    assert len(created) == 1                 # only the weight allocated
    assert created[0] is conv.weight
    assert conv.weight.closed is False       # open, no cleanup performed
    assert conv.bias is None
    conv.weight.close()


# ======================================================================
# Parameter and initialization behavior
# ======================================================================


@needs_native
def test_weight_and_bias_shapes():
    conv = NativeConv2d(3, 4, (2, 5))
    assert conv.weight.shape == (4, 3, 2, 5)
    assert conv.bias.shape == (4,)


@needs_native
def test_parameter_types_and_order():
    conv = NativeConv2d(2, 3, 3)
    names = [name for name, _ in conv.named_parameters()]
    assert names == ["weight", "bias"]
    for _, p in conv.named_parameters():
        assert isinstance(p, NativeParameter)


@needs_native
def test_no_buffers():
    conv = NativeConv2d(2, 3, 3)
    assert conv.buffers() == []
    assert list(conv.named_buffers()) == []


@needs_native
def test_bias_disabled_registers_only_weight():
    conv = NativeConv2d(2, 3, 3, bias=False)
    assert [n for n, _ in conv.named_parameters()] == ["weight"]
    assert conv.bias is None


@needs_native
def test_values_within_fan_in_bounds_and_finite():
    in_c, out_c, kh, kw = 3, 5, 2, 4
    conv = NativeConv2d(in_c, out_c, (kh, kw), seed=7)
    bound = 1.0 / math.sqrt(in_c * kh * kw)
    w = conv.weight.to_numpy()
    b = conv.bias.to_numpy()
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(b))
    assert np.all(w >= -bound) and np.all(w <= bound)
    assert np.all(b >= -bound) and np.all(b <= bound)


@needs_native
def test_same_seed_reproducible():
    a = NativeConv2d(3, 4, 3, seed=123)
    b = NativeConv2d(3, 4, 3, seed=123)
    assert np.array_equal(a.weight.to_numpy(), b.weight.to_numpy())
    assert np.array_equal(a.bias.to_numpy(), b.bias.to_numpy())


@needs_native
def test_different_seed_diverges():
    a = NativeConv2d(3, 4, 3, seed=1)
    b = NativeConv2d(3, 4, 3, seed=2)
    assert not np.array_equal(a.weight.to_numpy(), b.weight.to_numpy())


@needs_native
def test_no_global_random_state_mutation():
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    NativeConv2d(4, 4, 3, seed=99)
    after = np.random.get_state()[1]
    assert np.array_equal(before, after)


@needs_native
def test_requires_grad_propagates_to_both_parameters():
    conv = NativeConv2d(2, 3, 3, requires_grad=False)
    assert conv.weight.requires_grad is False
    assert conv.bias.requires_grad is False
    conv2 = NativeConv2d(2, 3, 3, requires_grad=True)
    assert conv2.weight.requires_grad is True
    assert conv2.bias.requires_grad is True


@needs_native
def test_parameter_versions_start_at_zero():
    conv = NativeConv2d(2, 3, 3)
    assert conv.weight.version == 0
    assert conv.bias.version == 0


@needs_native
def test_initialization_builds_no_graph():
    conv = NativeConv2d(2, 3, 3, seed=0)
    assert conv.weight.is_leaf is True
    assert conv.bias.is_leaf is True


# ======================================================================
# Forward behavior
# ======================================================================


@needs_native
def test_forward_hand_computed_small_case():
    # 1 sample, 1 in-channel, 1 out-channel, 2x2 kernel, no padding/stride 1.
    conv = NativeConv2d(1, 1, 2, bias=True, seed=0)
    w = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])  # (1,1,2,2)
    b = np.array([0.5])
    _overwrite(conv.weight, w)
    _overwrite(conv.bias, b)
    x = np.arange(9.0).reshape(1, 1, 3, 3)
    out = conv(NativeTensor.from_array(x))
    expected = _np_conv2d(x, w, b, (1, 1), (0, 0))
    assert out.shape == expected.shape
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)


@needs_native
def test_forward_parity_with_direct_conv2d_call():
    conv = NativeConv2d(3, 4, (3, 2), stride=(1, 2), padding=(1, 0), seed=5)
    x = NativeTensor.from_array(
        np.random.default_rng(0).standard_normal((2, 3, 6, 5))
    )
    out = conv(x)
    direct = x.conv2d(conv.weight, conv.bias, stride=(1, 2), padding=(1, 0))
    assert np.array_equal(out.to_numpy(), direct.to_numpy())


@needs_native
def test_forward_numpy_parity_bias_and_no_bias():
    rng = np.random.default_rng(3)
    for bias in (True, False):
        conv = NativeConv2d(2, 3, 3, padding=1, bias=bias, seed=1)
        x = rng.standard_normal((2, 2, 5, 5))
        w = conv.weight.to_numpy()
        b = conv.bias.to_numpy() if bias else None
        out = conv(NativeTensor.from_array(x))
        expected = _np_conv2d(x, w, b, (1, 1), (1, 1))
        assert np.allclose(out.to_numpy(), expected, atol=1e-11)


@needs_native
def test_forward_batch_and_multichannel_shapes():
    conv = NativeConv2d(3, 6, 3, seed=0)
    x = NativeTensor.from_array(
        np.random.default_rng(0).standard_normal((4, 3, 8, 8))
    )
    out = conv(x)
    assert out.shape == (4, 6, 6, 6)


@needs_native
def test_forward_stride_and_padding_shapes():
    conv = NativeConv2d(1, 1, 3, stride=2, padding=1, seed=0)
    x = NativeTensor.from_array(
        np.random.default_rng(0).standard_normal((1, 1, 7, 7))
    )
    out = conv(x)
    assert out.shape == (1, 1, 4, 4)


@needs_native
def test_forward_non_contiguous_input():
    conv = NativeConv2d(2, 3, 3, seed=2)
    # Build a logical (N, C, H, W) input as a non-contiguous transpose view.
    base = np.random.default_rng(0).standard_normal((2, 2, 5, 4))
    src = NativeTensor.from_array(np.ascontiguousarray(base.transpose(0, 1, 3, 2)))
    # Transposing the contiguous source back returns the logical (2,2,5,4)
    # values of ``base`` — but as a non-contiguous view.
    view = src.transpose(0, 1, 3, 2)
    assert not view.contiguous
    assert view.shape == (2, 2, 5, 4)
    out = conv(view)
    expected = _np_conv2d(base, conv.weight.to_numpy(),
                          conv.bias.to_numpy(), (1, 1), (0, 0))
    assert np.allclose(out.to_numpy(), expected, atol=1e-11)


@needs_native
def test_forward_output_is_owning_tensor():
    conv = NativeConv2d(2, 3, 3, seed=0)
    x = NativeTensor.from_array(np.zeros((1, 2, 4, 4)))
    out = conv(x)
    assert out.owns_core is True
    assert not isinstance(out, NativeParameter)


@needs_native
def test_forward_invalid_input_type_rejected():
    conv = NativeConv2d(2, 3, 3)
    with pytest.raises(TypeError):
        conv(np.zeros((1, 2, 4, 4)))


@needs_native
def test_forward_closed_input_rejected():
    conv = NativeConv2d(2, 3, 3)
    x = NativeTensor.from_array(np.zeros((1, 2, 4, 4)))
    x.close()
    with pytest.raises(RuntimeError):
        conv(x)


@needs_native
def test_forward_rank_mismatch_rejected():
    conv = NativeConv2d(2, 3, 3)
    x = NativeTensor.from_array(np.zeros((2, 8)))  # 2-D
    with pytest.raises(ValueError):
        conv(x)


@needs_native
def test_forward_channel_mismatch_rejected():
    conv = NativeConv2d(2, 3, 3)
    x = NativeTensor.from_array(np.zeros((1, 5, 6, 6)))  # 5 != 2 channels
    with pytest.raises(ValueError, match="in_channels"):
        conv(x)


# ======================================================================
# Autograd (inherited entirely from the D6 conv2d primitive)
# ======================================================================


@needs_native
def test_input_weight_and_bias_gradients_flow():
    conv = NativeConv2d(2, 3, 3, padding=1, seed=0)
    x = NativeTensor.from_array(
        np.random.default_rng(1).standard_normal((2, 2, 5, 5)),
        requires_grad=True,
    )
    conv(x).sum().backward()
    assert x.grad is not None and x.grad.shape == (2, 2, 5, 5)
    assert conv.weight.grad is not None and conv.weight.grad.shape == (3, 2, 3, 3)
    assert conv.bias.grad is not None and conv.bias.grad.shape == (3,)


@needs_native
def test_gradients_match_direct_conv2d():
    rng = np.random.default_rng(4)
    x_np = rng.standard_normal((2, 2, 5, 4))
    conv = NativeConv2d(2, 3, 3, seed=1)
    w_snapshot = conv.weight.to_numpy()
    b_snapshot = conv.bias.to_numpy()

    x1 = NativeTensor.from_array(x_np, requires_grad=True)
    conv(x1).sum().backward()
    module_grad = x1.grad.to_numpy()

    # Same computation via the direct primitive with fresh leaf parameters.
    x2 = NativeTensor.from_array(x_np, requires_grad=True)
    w = NativeParameter(w_snapshot)
    b = NativeParameter(b_snapshot)
    x2.conv2d(w, b).sum().backward()
    assert np.allclose(module_grad, x2.grad.to_numpy(), atol=1e-11)
    assert np.allclose(conv.weight.grad.to_numpy(), w.grad.to_numpy(), atol=1e-11)


@needs_native
def test_frozen_module_parameters_get_no_gradient():
    conv = NativeConv2d(2, 3, 3, requires_grad=False, seed=0)
    x = NativeTensor.from_array(
        np.random.default_rng(0).standard_normal((1, 2, 5, 5)),
        requires_grad=True,
    )
    conv(x).sum().backward()
    # Input still differentiates; frozen parameters accumulate nothing.
    assert x.grad is not None
    assert conv.weight.grad is None
    assert conv.bias.grad is None


@needs_native
def test_input_only_gradient_when_parameters_frozen():
    conv = NativeConv2d(1, 1, 2, requires_grad=False, seed=0)
    x = NativeTensor.from_array(np.ones((1, 1, 3, 3)), requires_grad=True)
    out = conv(x)
    assert out.requires_grad is True
    out.sum().backward()
    assert x.grad is not None


@needs_native
def test_shared_module_used_twice_accumulates():
    conv = NativeConv2d(1, 1, 2, seed=0)
    x = NativeTensor.from_array(np.ones((1, 1, 3, 3)), requires_grad=True)
    a = conv(x)
    b = conv(x)
    (a.sum().add(b.sum())).backward()
    # Two branches into the same weight parameter accumulate one grad.
    assert conv.weight.grad is not None
    assert conv.weight.grad.shape == (1, 1, 2, 2)


@needs_native
def test_zero_grad_clears_parameter_gradients():
    conv = NativeConv2d(2, 2, 3, seed=0)
    x = NativeTensor.from_array(np.ones((1, 2, 5, 5)), requires_grad=True)
    conv(x).sum().backward()
    assert conv.weight.grad is not None
    conv.zero_grad()
    assert conv.weight.grad is None
    assert conv.bias.grad is None


@needs_native
def test_stale_graph_detection_inherited():
    # Mutating the weight after forward but before backward must raise the
    # inherited D6 stale-value error (weight-grad rereads the input; the
    # input-grad callback rereads the weight, whose version is recorded).
    conv = NativeConv2d(1, 1, 2, seed=0)
    x = NativeTensor.from_array(np.ones((1, 1, 3, 3)), requires_grad=True)
    out = conv(x)
    _overwrite(conv.weight, conv.weight.to_numpy())  # bumps version
    with pytest.raises(RuntimeError):
        out.sum().backward()


# ======================================================================
# NativeModule integration
# ======================================================================


@needs_native
def test_traversal_methods():
    conv = NativeConv2d(2, 3, 3)
    assert [p.shape for p in conv.parameters()] == [(3, 2, 3, 3), (3,)]
    assert [n for n, _ in conv.named_parameters()] == ["weight", "bias"]
    assert conv.buffers() == []
    assert list(conv.named_buffers()) == []
    assert conv.modules() == [conv]


@needs_native
def test_train_eval_propagates():
    conv = NativeConv2d(2, 3, 3)
    assert conv.eval() is conv
    assert conv.training is False
    assert conv.train() is conv
    assert conv.training is True


@needs_native
def test_registration_inside_custom_parent_module():
    class Net(NativeModule):
        def __init__(self):
            super().__init__()
            self.conv = NativeConv2d(2, 3, 3, seed=0)

        def forward(self, x):
            return self.conv(x)

    net = Net()
    names = [n for n, _ in net.named_parameters()]
    assert names == ["conv.weight", "conv.bias"]
    assert net.modules() == [net, net.conv]


@needs_native
def test_registration_inside_sequential_hierarchical_names():
    model = NativeSequential(
        NativeConv2d(1, 2, 3, seed=0),
        NativeReLU(),
    )
    names = [n for n, _ in model.named_parameters()]
    assert names == ["0.weight", "0.bias"]


@needs_native
def test_repr():
    conv = NativeConv2d(3, 4, (2, 5), stride=(2, 1), padding=(1, 0), bias=False)
    text = repr(conv)
    assert "NativeConv2d(" in text
    assert "in_channels=3" in text
    assert "out_channels=4" in text
    assert "kernel_size=(2, 5)" in text
    assert "stride=(2, 1)" in text
    assert "padding=(1, 0)" in text
    assert "bias=False" in text


# ======================================================================
# State dictionary
# ======================================================================


@needs_native
def test_state_dict_keys_with_and_without_bias():
    assert set(NativeConv2d(2, 3, 3).state_dict()) == {"weight", "bias"}
    assert set(NativeConv2d(2, 3, 3, bias=False).state_dict()) == {"weight"}


@needs_native
def test_state_dict_snapshot_independence():
    conv = NativeConv2d(2, 2, 3, seed=0)
    state = conv.state_dict()
    original = state["weight"].to_numpy().copy()
    _overwrite(conv.weight, conv.weight.to_numpy() + 1.0)
    # The snapshot is independent of later live mutation.
    assert np.array_equal(state["weight"].to_numpy(), original)
    for v in state.values():
        v.close()


@needs_native
def test_load_state_dict_preserves_identity_atomically():
    src = NativeConv2d(2, 3, 3, seed=0)
    dst = NativeConv2d(2, 3, 3, seed=9)
    state = src.state_dict()
    w_id, b_id = id(dst.weight), id(dst.bias)
    v_before = dst.weight.version
    dst.load_state_dict(state)
    assert id(dst.weight) == w_id and id(dst.bias) == b_id
    assert dst.weight.version == v_before + 1
    assert np.array_equal(dst.weight.to_numpy(), src.weight.to_numpy())
    for v in state.values():
        v.close()


@needs_native
def test_load_state_dict_shape_mismatch_leaves_module_unchanged():
    conv = NativeConv2d(2, 3, 3, seed=0)
    before = conv.weight.to_numpy().copy()
    v_before = conv.weight.version
    bad = {
        "weight": NativeTensor.from_array(np.zeros((3, 2, 5, 5))),  # wrong shape
        "bias": NativeTensor.from_array(np.zeros((3,))),
    }
    with pytest.raises(ValueError):
        conv.load_state_dict(bad)
    assert np.array_equal(conv.weight.to_numpy(), before)
    assert conv.weight.version == v_before
    for v in bad.values():
        v.close()


@needs_native
def test_bias_mismatch_key_rules():
    biased = NativeConv2d(2, 3, 3, bias=True)
    plain = NativeConv2d(2, 3, 3, bias=False)
    # Loading a biased state into a bias-free layer: "bias" is unexpected.
    state = biased.state_dict()
    with pytest.raises(ValueError):
        plain.load_state_dict(state)
    # The reverse: "bias" missing.
    with pytest.raises(ValueError):
        biased.load_state_dict(plain.state_dict())
    for v in state.values():
        v.close()


@needs_native
def test_sequential_state_keys():
    model = NativeSequential(NativeConv2d(1, 2, 3, seed=0), NativeReLU())
    assert set(model.state_dict()) == {"0.weight", "0.bias"}


# ======================================================================
# Checkpoint behavior
# ======================================================================


@needs_native
def test_checkpoint_round_trip_model_only(tmp_path):
    src = NativeConv2d(2, 3, 3, seed=0)
    saved = {n: p.to_numpy() for n, p in src.named_parameters()}
    path = tmp_path / "conv.npz"
    save_native_checkpoint(path, src)
    dst = NativeConv2d(2, 3, 3, seed=9)
    ids = [id(p) for p in dst.parameters()]
    load_native_checkpoint(path, dst)
    for n, p in dst.named_parameters():
        assert np.array_equal(p.to_numpy(), saved[n])
    assert [id(p) for p in dst.parameters()] == ids


@needs_native
def test_checkpoint_round_trip_bias_disabled(tmp_path):
    src = NativeConv2d(2, 3, 3, bias=False, seed=0)
    saved = src.weight.to_numpy()
    path = tmp_path / "conv_nobias.npz"
    save_native_checkpoint(path, src)
    dst = NativeConv2d(2, 3, 3, bias=False, seed=1)
    load_native_checkpoint(path, dst)
    assert np.array_equal(dst.weight.to_numpy(), saved)


@needs_native
def test_checkpoint_round_trip_with_optimizer(tmp_path):
    src = NativeConv2d(2, 3, 3, seed=0)
    opt = NativeAdam(src.parameters(), lr=0.01)
    x = NativeTensor.from_array(np.ones((1, 2, 5, 5)), requires_grad=True)
    NativeMSELoss()(src(x), NativeTensor.from_array(np.zeros((1, 3, 3, 3)))).backward()
    opt.step()
    path = tmp_path / "conv_opt.npz"
    save_native_checkpoint(path, src, optimizer=opt)

    dst = NativeConv2d(2, 3, 3, seed=9)
    fresh = NativeAdam(dst.parameters(), lr=0.01)
    load_native_checkpoint(path, dst, optimizer=fresh)
    for n, p in dst.named_parameters():
        assert np.array_equal(
            p.to_numpy(), dict(src.named_parameters())[n].to_numpy()
        )
    opt.close()
    fresh.close()


@needs_native
def test_checkpoint_schema_unchanged(tmp_path):
    # NativeConv2d rides the existing generic checkpoint format — no new
    # schema/version tag is introduced. Decode the stored JSON manifest and
    # assert the pre-D7 format identity and version are unchanged.
    import json

    conv = NativeConv2d(1, 1, 2, seed=0)
    path = tmp_path / "conv_schema.npz"
    save_native_checkpoint(path, conv)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(bytes(archive["manifest"]).decode("utf-8"))
    assert manifest["format"] == "tensorforge.native_checkpoint"
    assert manifest["format_version"] == 1
    assert set(manifest["model"]["keys"]) == {"weight", "bias"}


# ======================================================================
# Optimizer compatibility
# ======================================================================


@needs_native
@pytest.mark.parametrize("make_opt", [
    lambda ps: NativeSGD(ps, lr=0.1),
    lambda ps: NativeAdam(ps, lr=0.1),
])
def test_optimizer_one_step_update(make_opt):
    conv = NativeConv2d(2, 3, 3, seed=0)
    opt = make_opt(conv.parameters())
    w_id, b_id = id(conv.weight), id(conv.bias)
    w_before = conv.weight.to_numpy().copy()
    v_before = conv.weight.version
    x = NativeTensor.from_array(np.ones((1, 2, 5, 5)), requires_grad=True)
    NativeMSELoss()(conv(x), NativeTensor.from_array(np.zeros((1, 3, 3, 3)))).backward()
    opt.step()
    assert not np.array_equal(conv.weight.to_numpy(), w_before)
    assert id(conv.weight) == w_id and id(conv.bias) == b_id
    assert conv.weight.version == v_before + 1
    if hasattr(opt, "close"):
        opt.close()


@needs_native
def test_optimizer_bias_disabled_discovers_only_weight():
    conv = NativeConv2d(2, 3, 3, bias=False, seed=0)
    opt = NativeSGD(conv.parameters(), lr=0.1)
    x = NativeTensor.from_array(np.ones((1, 2, 5, 5)), requires_grad=True)
    NativeMSELoss()(conv(x), NativeTensor.from_array(np.zeros((1, 3, 3, 3)))).backward()
    w_before = conv.weight.to_numpy().copy()
    opt.step()
    assert not np.array_equal(conv.weight.to_numpy(), w_before)


@needs_native
def test_optimizer_state_round_trip():
    conv = NativeConv2d(2, 3, 3, seed=0)
    opt = NativeAdam(conv.parameters(), lr=0.05)
    x = NativeTensor.from_array(np.ones((1, 2, 5, 5)), requires_grad=True)
    NativeMSELoss()(conv(x), NativeTensor.from_array(np.zeros((1, 3, 3, 3)))).backward()
    opt.step()
    state = opt.state_dict()
    fresh = NativeAdam(conv.parameters(), lr=0.05)
    fresh.load_state_dict(state)  # must accept Conv2d parameter metadata
    opt.close()
    fresh.close()


# ======================================================================
# Sequential integration end-to-end
# ======================================================================


@needs_native
def test_sequential_conv_relu_flatten_linear_trains():
    model = NativeSequential(
        NativeConv2d(1, 2, 3, seed=0),   # (N,1,6,6) -> (N,2,4,4)
        NativeReLU(),
        NativeFlatten(),                 # -> (N, 32)
        NativeLinear(32, 1, seed=1),
    )
    x = NativeTensor.from_array(
        np.random.default_rng(0).standard_normal((3, 1, 6, 6)),
        requires_grad=True,
    )
    out = model(x)
    assert out.shape == (3, 1)
    loss = NativeMSELoss()(out, NativeTensor.from_array(np.zeros((3, 1))))
    loss.backward()
    # Backward reaches every trainable parameter and the input.
    assert model[0].weight.grad is not None
    assert model[0].bias.grad is not None
    assert model[3].weight.grad is not None
    assert x.grad is not None and x.grad.shape == (3, 1, 6, 6)
    assert set(model.state_dict()) == {
        "0.weight", "0.bias", "3.weight", "3.bias"
    }


# ======================================================================
# Introspection and documentation guardrails
# ======================================================================


def test_native_conv2d_in_native_modules_inventory():
    assert "NativeConv2d" in cpp.NATIVE_MODULES


def test_native_conv2d_exported_from_experimental():
    import tensorforge.experimental as experimental

    assert "NativeConv2d" in experimental.__all__
    assert hasattr(experimental, "NativeConv2d")


def test_conv2d_operation_still_advertised():
    assert "conv2d" in cpp.AUTOGRAD_OPS
    # No new raw/Core operation was introduced by the module milestone.
    assert "NativeConv2d" not in cpp.RAW_KERNELS
    assert "NativeConv2d" not in cpp.TENSOR_CORE_OPS


def test_maxpool2d_module_is_supported_alongside_conv2d():
    # The pooling operation shipped in D8/D9 and its module in D10, so both
    # CNN layers are now advertised the same way — NativeConv2d is not a
    # special case in the inventory.
    assert "NativeMaxPool2d" not in cpp.UNSUPPORTED
    assert "NativeMaxPool2d" in cpp.NATIVE_MODULES
    import tensorforge.experimental as experimental

    assert hasattr(experimental, "NativeMaxPool2d")
    assert "NativeMaxPool2d" in experimental.__all__


def test_phase_d_not_marked_complete():
    # The Conv2d module is done (D7) but pooling and the D11 proof are not,
    # so the module milestone must not claim to be a supported operation
    # under a different layer, and NativeConv2d is no longer "unsupported".
    assert "NativeConv2d" not in cpp.UNSUPPORTED
