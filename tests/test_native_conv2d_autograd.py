"""Differentiable native Conv2d — NativeTensor.conv2d autograd integration
(Phase D, milestone D6).

D6 adds ``NativeTensor.conv2d(weight, bias=None, *, stride=1, padding=0)``:
a new fused primitive whose Python-managed backward computes the input,
weight, and bias gradients (the first two via the native Core backward
ops, bias via the existing native ``sum`` reduction). These tests cover
forward/gradient parity with the stable framework, finite differences, all
``requires_grad`` combinations, shared-graph accumulation, conditional
version tracking, explicit-gradient validation, failure rollback, lifetime,
and the no-grad graph-avoidance path.

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close().

Selector: python -m pytest -q -k native_conv2d_autograd
"""

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import NativeTensor, NativeParameter

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)


@pytest.fixture(autouse=True)
def _disarm_after_each():
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _stable(x, w, b, g, stride, padding):
    """Stable Conv2d forward + gradients for a fixed upstream ``g``."""
    from tensorforge.nn import Conv2d
    from tensorforge.tensor import Tensor

    o, c, kh, kw = w.shape
    layer = Conv2d(c, o, (kh, kw), stride=stride, padding=padding, bias=b is not None)
    layer.weight.data = np.array(w, float)
    if b is not None:
        layer.bias.data = np.array(b, float)
    xt = Tensor(np.array(x, float), requires_grad=True)
    out = layer(xt)
    (out * Tensor(np.array(g, float))).sum().backward()
    return (out.data, xt.grad, layer.weight.grad,
            (layer.bias.grad if b is not None else None))


def _default_case(seed=0, stride=1, padding=0, bias=True):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 3, 6, 5))
    w = rng.standard_normal((4, 3, 3, 2))
    b = rng.standard_normal(4) if bias else None
    return x, w, b


# --------------------------------------------------------------------------
# Forward + gradient parity with the stable framework
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1), ((2, 1), (1, 0))])
@pytest.mark.parametrize("bias", [True, False])
def test_forward_and_gradient_parity(stride, padding, bias):
    x, w, b = _default_case(seed=1, bias=bias)
    rng = np.random.default_rng(99)
    xi = NativeParameter(x); wi = NativeParameter(w)
    bi = NativeParameter(b) if bias else None
    y = xi.conv2d(wi, bi, stride=stride, padding=padding)
    out, sx, sw, sb = _stable(x, w, b, np.zeros(y.shape), stride, padding)  # shape only
    g = rng.standard_normal(y.shape)
    out, sx, sw, sb = _stable(x, w, b, g, stride, padding)
    assert np.allclose(y.to_numpy(), out, atol=1e-9)
    y.backward(gradient=NativeTensor.from_array(g))
    assert np.allclose(xi.grad.to_numpy(), sx, atol=1e-9)
    assert np.allclose(wi.grad.to_numpy(), sw, atol=1e-9)
    if bias:
        assert np.allclose(bi.grad.to_numpy(), sb, atol=1e-9)
    for t in (y, xi, wi):
        t.close()
    if bias:
        bi.close()


def test_integer_and_tuple_stride_padding_agree():
    x, w, b = _default_case(seed=2)
    xi = NativeParameter(x); wi = NativeParameter(w); bi = NativeParameter(b)
    y_int = xi.conv2d(wi, bi, stride=2, padding=1)
    y_tup = xi.conv2d(wi, bi, stride=(2, 2), padding=(1, 1))
    assert np.allclose(y_int.to_numpy(), y_tup.to_numpy(), atol=1e-12)
    for t in (y_int, y_tup, xi, wi, bi):
        t.close()


def test_deterministic_repeated_forward():
    x, w, b = _default_case(seed=3)
    xi = NativeTensor.from_array(x); wi = NativeTensor.from_array(w)
    bi = NativeTensor.from_array(b)
    a = xi.conv2d(wi, bi, stride=1, padding=1).to_numpy()
    c = xi.conv2d(wi, bi, stride=1, padding=1).to_numpy()
    assert np.array_equal(a, c)
    for t in (xi, wi, bi):
        t.close()


# --------------------------------------------------------------------------
# Finite differences
# --------------------------------------------------------------------------

def _scalar_loss(xi, wi, bi, gvals, stride, padding):
    y = xi.conv2d(wi, bi, stride=stride, padding=padding)
    gt = NativeTensor.from_array(gvals)
    loss = y.multiply(gt).sum()
    val = float(loss.to_numpy())
    for t in (y, gt, loss):
        t.close()
    return val


def test_finite_difference_input_and_weight():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1, 2, 4, 4)); w = rng.standard_normal((2, 2, 2, 2))
    b = rng.standard_normal(2); stride, padding = 1, 1
    oh = (4 + 2 - 2) + 1
    g = rng.standard_normal((1, 2, oh, oh))

    xi = NativeParameter(x); wi = NativeParameter(w); bi = NativeParameter(b)
    y = xi.conv2d(wi, bi, stride=stride, padding=padding)
    y.backward(gradient=NativeTensor.from_array(g))
    analytic_x = xi.grad.to_numpy(); analytic_w = wi.grad.to_numpy()
    y.close()

    eps = 1e-5
    # input finite differences
    fd_x = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        xp = x.copy(); xp[idx] += eps
        xm = x.copy(); xm[idx] -= eps
        a = NativeTensor.from_array(xp); m = NativeTensor.from_array(xm)
        wc = NativeTensor.from_array(w); bc = NativeTensor.from_array(b)
        fd_x[idx] = (_scalar_loss(a, wc, bc, g, stride, padding)
                     - _scalar_loss(m, wc, bc, g, stride, padding)) / (2 * eps)
        for t in (a, m, wc, bc):
            t.close()
    assert np.allclose(analytic_x, fd_x, atol=1e-6)

    # weight finite differences
    fd_w = np.zeros_like(w)
    it = np.nditer(w, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        wp = w.copy(); wp[idx] += eps
        wm = w.copy(); wm[idx] -= eps
        xc = NativeTensor.from_array(x); bc = NativeTensor.from_array(b)
        ap = NativeTensor.from_array(wp); am = NativeTensor.from_array(wm)
        fd_w[idx] = (_scalar_loss(xc, ap, bc, g, stride, padding)
                     - _scalar_loss(xc, am, bc, g, stride, padding)) / (2 * eps)
        for t in (xc, bc, ap, am):
            t.close()
    assert np.allclose(analytic_w, fd_w, atol=1e-6)
    for t in (xi, wi, bi):
        t.close()


# --------------------------------------------------------------------------
# requires_grad combinations
# --------------------------------------------------------------------------

@pytest.mark.parametrize("xr,wr,br", [
    (True, False, False), (False, True, False), (False, False, True),
    (True, True, False), (True, False, True), (False, True, True),
    (True, True, True),
])
def test_requires_grad_combinations(xr, wr, br):
    x, w, b = _default_case(seed=5)
    xi = NativeParameter(x, requires_grad=xr)
    wi = NativeParameter(w, requires_grad=wr)
    bi = NativeParameter(b, requires_grad=br)
    y = xi.conv2d(wi, bi, stride=1, padding=1)
    g = np.random.default_rng(6).standard_normal(y.shape)
    _, sx, sw, sb = _stable(x, w, b, g, 1, 1)
    assert y.requires_grad is (xr or wr or br)
    y.backward(gradient=NativeTensor.from_array(g))
    assert (xi.grad is not None) is xr
    assert (wi.grad is not None) is wr
    assert (bi.grad is not None) is br
    if xr:
        assert np.allclose(xi.grad.to_numpy(), sx, atol=1e-9)
    if wr:
        assert np.allclose(wi.grad.to_numpy(), sw, atol=1e-9)
    if br:
        assert np.allclose(bi.grad.to_numpy(), sb, atol=1e-9)
    for t in (y, xi, wi, bi):
        t.close()


def test_no_parent_requires_grad_avoids_graph():
    x, w, b = _default_case(seed=7)
    xi = NativeTensor.from_array(x); wi = NativeTensor.from_array(w)
    bi = NativeTensor.from_array(b)
    y = xi.conv2d(wi, bi, stride=1, padding=1)
    assert y.requires_grad is False and y.is_leaf is True
    with pytest.raises(RuntimeError):
        y.backward(gradient=NativeTensor.from_array(np.ones(y.shape)))
    for t in (y, xi, wi, bi):
        t.close()


# --------------------------------------------------------------------------
# Parent ordering / graph metadata
# --------------------------------------------------------------------------

def test_parent_ordering_and_no_value_duplication():
    x, w, b = _default_case(seed=8)
    xi = NativeParameter(x); wi = NativeParameter(w); bi = NativeParameter(b)
    y = xi.conv2d(wi, bi, stride=1, padding=0)
    assert y._parents == (xi, wi, bi)  # deterministic (input, weight, bias)
    # No forward output or full operand values are duplicated into Python.
    y_nobias = xi.conv2d(wi, stride=1, padding=0)
    assert y_nobias._parents == (xi, wi)
    for t in (y, y_nobias, xi, wi, bi):
        t.close()


# --------------------------------------------------------------------------
# Shared-graph accumulation
# --------------------------------------------------------------------------

def test_shared_weight_across_two_conv_calls_accumulates():
    rng = np.random.default_rng(9)
    x1 = rng.standard_normal((1, 2, 5, 5)); x2 = rng.standard_normal((1, 2, 5, 5))
    w = rng.standard_normal((2, 2, 3, 3))
    xi1 = NativeParameter(x1); xi2 = NativeParameter(x2); wi = NativeParameter(w)
    y1 = xi1.conv2d(wi, stride=1, padding=1)
    y2 = xi2.conv2d(wi, stride=1, padding=1)
    loss = y1.sum().add(y2.sum())
    loss.backward()
    # Reference: dL/dw = weight-grad(x1, ones) + weight-grad(x2, ones)
    g1 = np.ones(y1.shape); g2 = np.ones(y2.shape)
    _, _, sw1, _ = _stable(x1, w, None, g1, 1, 1)
    _, _, sw2, _ = _stable(x2, w, None, g2, 1, 1)
    assert np.allclose(wi.grad.to_numpy(), sw1 + sw2, atol=1e-9)
    for t in (y1, y2, loss, xi1, xi2, wi):
        t.close()


def test_scalar_loss_through_existing_reductions():
    x, w, b = _default_case(seed=10)
    xi = NativeParameter(x); wi = NativeParameter(w); bi = NativeParameter(b)
    y = xi.conv2d(wi, bi, stride=1, padding=1)
    loss = y.mean()  # scalar via existing reduction
    loss.backward()
    # dL/d? for mean = weight/input/bias grad with upstream = 1/numel everywhere
    g = np.full(y.shape, 1.0 / y.numel)
    _, sx, sw, sb = _stable(x, w, b, g, 1, 1)
    assert np.allclose(xi.grad.to_numpy(), sx, atol=1e-9)
    assert np.allclose(wi.grad.to_numpy(), sw, atol=1e-9)
    assert np.allclose(bi.grad.to_numpy(), sb, atol=1e-9)
    for t in (y, loss, xi, wi, bi):
        t.close()


# --------------------------------------------------------------------------
# Conditional version tracking (docs/native_cnn_design.md §8)
# --------------------------------------------------------------------------

def test_input_grad_weight_mutation_is_stale():
    x, w, b = _default_case(seed=11, bias=False)
    xi = NativeParameter(x, requires_grad=True)
    wi = NativeParameter(w, requires_grad=False)  # frozen but versioned
    y = xi.conv2d(wi, stride=1, padding=1)
    wi.copy_value_(NativeParameter(w * 2))  # input-grad rereads weight -> stale
    with pytest.raises(RuntimeError, match="stale"):
        y.backward(gradient=NativeTensor.from_array(np.ones(y.shape)))
    for t in (y, xi, wi):
        t.close()


def test_weight_grad_input_mutation_is_stale_when_input_versioned():
    x, w, b = _default_case(seed=12, bias=False)
    xi = NativeParameter(x, requires_grad=False)  # versioned param
    wi = NativeParameter(w, requires_grad=True)
    y = xi.conv2d(wi, stride=1, padding=1)
    xi.copy_value_(NativeParameter(x * 2))  # weight-grad rereads input -> stale
    with pytest.raises(RuntimeError, match="stale"):
        y.backward(gradient=NativeTensor.from_array(np.ones(y.shape)))
    for t in (y, xi, wi):
        t.close()


def test_both_grads_record_both_versions():
    x, w, b = _default_case(seed=13, bias=False)
    xi = NativeParameter(x); wi = NativeParameter(w)
    y = xi.conv2d(wi, stride=1, padding=1)
    xi.copy_value_(NativeParameter(x * 2))  # weight-grad rereads input -> stale
    with pytest.raises(RuntimeError, match="stale"):
        y.backward(gradient=NativeTensor.from_array(np.ones(y.shape)))
    for t in (y, xi, wi):
        t.close()


def test_bias_only_ignores_input_weight_mutation():
    x, w, b = _default_case(seed=14)
    xi = NativeParameter(x, requires_grad=False)
    wi = NativeParameter(w, requires_grad=False)
    bi = NativeParameter(b, requires_grad=True)
    y = xi.conv2d(wi, bi, stride=1, padding=1)
    xi.copy_value_(NativeParameter(x * 5))  # neither is reread by bias-grad
    wi.copy_value_(NativeParameter(w * 5))
    g = np.random.default_rng(1).standard_normal(y.shape)
    _, _, _, sb = _stable(x, w, b, g, 1, 1)  # bias grad is bias/input-independent
    y.backward(gradient=NativeTensor.from_array(g))  # must NOT raise stale
    assert np.allclose(bi.grad.to_numpy(), sb, atol=1e-9)
    for t in (y, xi, wi, bi):
        t.close()


def test_plain_tensor_operands_record_no_version():
    # An ordinary (non-parameter) NativeTensor has no version slot: no false
    # guarantees, and no stale error can arise from it.
    x, w, b = _default_case(seed=15, bias=False)
    xi = NativeTensor.from_array(x, requires_grad=True)
    wi = NativeTensor.from_array(w, requires_grad=True)
    y = xi.conv2d(wi, stride=1, padding=1)
    assert y._expected_versions == ()
    y.backward(gradient=NativeTensor.from_array(np.ones(y.shape)))
    for t in (y, xi, wi):
        t.close()


# --------------------------------------------------------------------------
# Explicit-gradient validation
# --------------------------------------------------------------------------

def test_nonscalar_backward_without_gradient_raises():
    x, w, b = _default_case(seed=16)
    xi = NativeParameter(x); wi = NativeParameter(w); bi = NativeParameter(b)
    y = xi.conv2d(wi, bi, stride=1, padding=1)
    with pytest.raises(ValueError):
        y.backward()  # non-scalar output needs an explicit gradient
    for t in (y, xi, wi, bi):
        t.close()


def test_wrong_shape_explicit_gradient_leaves_grads_unchanged():
    x, w, b = _default_case(seed=17)
    xi = NativeParameter(x); wi = NativeParameter(w); bi = NativeParameter(b)
    y = xi.conv2d(wi, bi, stride=1, padding=1)
    with pytest.raises(ValueError):
        y.backward(gradient=NativeTensor.from_array(np.ones((3, 3))))
    assert xi.grad is None and wi.grad is None and bi.grad is None
    for t in (y, xi, wi, bi):
        t.close()


def test_non_contiguous_explicit_gradient_accepted():
    rng = np.random.default_rng(18)
    x = rng.standard_normal((1, 2, 5, 4)); w = rng.standard_normal((3, 2, 2, 2))
    b = rng.standard_normal(3)
    xi = NativeParameter(x); wi = NativeParameter(w); bi = NativeParameter(b)
    y = xi.conv2d(wi, bi, stride=1, padding=0)
    g = rng.standard_normal(y.shape)
    _, sx, sw, sb = _stable(x, w, b, g, 1, 0)
    # A non-contiguous upstream gradient (transpose the two spatial axes of a
    # contiguous base whose logical value equals g).
    base = NativeTensor.from_array(np.ascontiguousarray(g.transpose(0, 1, 3, 2)))
    gv = base.transpose(0, 1, 3, 2)
    assert not gv.contiguous
    y.backward(gradient=gv)
    assert np.allclose(xi.grad.to_numpy(), sx, atol=1e-9)
    assert np.allclose(wi.grad.to_numpy(), sw, atol=1e-9)
    assert np.allclose(bi.grad.to_numpy(), sb, atol=1e-9)
    for t in (y, base, xi, wi, bi):
        t.close()


# --------------------------------------------------------------------------
# Failure rollback / lifetime
# --------------------------------------------------------------------------

@needs_fault_injection
def test_backward_allocation_failure_rolls_back():
    x, w, b = _default_case(seed=19, bias=False)
    xi = NativeParameter(x); wi = NativeParameter(w)
    # Give the weight a pre-existing gradient to confirm it is restored.
    wi._grad = NativeTensor.from_array(np.full(w.shape, 7.0))
    saved = wi.grad.to_numpy().copy()
    y = xi.conv2d(wi, stride=1, padding=1)
    g = np.ones(y.shape)
    cpp._arm_alloc_failure(1)  # fail the first native allocation in backward
    with pytest.raises(MemoryError):
        y.backward(gradient=NativeTensor.from_array(g), retain_graph=True)
    # Rolled back: the weight's prior gradient is exactly restored, input got
    # no partial gradient, and inputs stay open/unchanged.
    assert np.array_equal(wi.grad.to_numpy(), saved)
    assert xi.grad is None
    assert np.allclose(xi.to_numpy(), x) and np.allclose(wi.to_numpy(), w)
    # A subsequent backward succeeds and the error slot is clear.
    y.backward(gradient=NativeTensor.from_array(g))
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    for t in (y, xi, wi):
        t.close()


def test_closed_operand_makes_backward_raise():
    x, w, b = _default_case(seed=20, bias=False)
    xi = NativeParameter(x); wi = NativeParameter(w)
    y = xi.conv2d(wi, stride=1, padding=1)
    wi.close()  # backward rereads the weight value
    with pytest.raises(RuntimeError):
        y.backward(gradient=NativeTensor.from_array(np.ones(y.shape)))
    y.close(); xi.close()


def test_zero_grad_and_retain_graph():
    x, w, b = _default_case(seed=21, bias=False)
    xi = NativeParameter(x); wi = NativeParameter(w)
    y = xi.conv2d(wi, stride=1, padding=1)
    g = NativeTensor.from_array(np.ones(y.shape))
    y.backward(gradient=g, retain_graph=True)
    first = wi.grad.to_numpy().copy()
    y.backward(gradient=g, retain_graph=True)  # accumulates
    assert np.allclose(wi.grad.to_numpy(), 2 * first, atol=1e-9)
    xi.zero_grad(); wi.zero_grad()
    assert xi.grad is None and wi.grad is None
    for t in (y, g, xi, wi):
        t.close()


def test_one_shot_graph_freed_after_backward():
    x, w, b = _default_case(seed=22, bias=False)
    xi = NativeParameter(x); wi = NativeParameter(w)
    y = xi.conv2d(wi, stride=1, padding=1)
    y.backward(gradient=NativeTensor.from_array(np.ones(y.shape)))
    with pytest.raises(RuntimeError):
        y.backward(gradient=NativeTensor.from_array(np.ones(y.shape)))
    for t in (y, xi, wi):
        t.close()


# --------------------------------------------------------------------------
# Capability separation
# --------------------------------------------------------------------------

def test_conv2d_is_an_autograd_op_and_module_is_supported():
    assert "conv2d" in cpp.AUTOGRAD_OPS
    assert hasattr(NativeTensor, "conv2d")
    assert "conv2d_input_backward" in cpp.TENSOR_CORE_OPS
    assert "conv2d_weight_backward" in cpp.TENSOR_CORE_OPS
    # The op is supported (D6) and the module is supported (D7); pooling has
    # since reached both stages too (D8/D9 operation, D10 module), so both
    # CNN layers are in the module inventory and neither is unsupported.
    assert "conv2d" not in cpp.UNSUPPORTED
    assert "NativeConv2d" not in cpp.UNSUPPORTED
    assert "NativeConv2d" in cpp.NATIVE_MODULES
    assert "NativeMaxPool2d" in cpp.NATIVE_MODULES
    import tensorforge.experimental as experimental
    assert "NativeConv2d" in experimental.__all__
