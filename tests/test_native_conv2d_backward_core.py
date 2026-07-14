"""Conv2d input- and weight-gradient at the NativeTensorCore layer, plus
the raw C ABI backward wrappers (Phase D, milestone D6).

D6 exposes the D4/D5 internal gradient kernels through the exception-safe C
ABI (`tf_core_conv2d_input_backward`, `tf_core_conv2d_weight_backward`),
their ctypes/errcheck registration, and the forward-only, autograd-unaware
Core methods `NativeTensorCore.conv2d_input_backward` /
`conv2d_weight_backward`. These tests cover numerical correctness against
the stable framework, the output ownership/layout contract, Policy-B
handling of non-contiguous operands, validation, and the raw-ABI failure
behavior. The differentiable `NativeTensor.conv2d` op is tested separately
in tests/test_native_conv2d_autograd.py.

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close().

Selector: python -m pytest -q -k native_conv2d_backward_core
"""

import numpy as np
import pytest

from tensorforge.backends import cpp

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

def _core(values):
    return cpp.NativeTensorCore.from_array(np.asarray(values, dtype=np.float64))


def _stable_grads(x, w, g, stride, padding, bias=None):
    """The stable framework's Conv2d input/weight/bias gradients for a fixed
    upstream ``g`` — the numerical reference the native line matches."""
    from tensorforge.nn import Conv2d
    from tensorforge.tensor import Tensor

    out_channels, in_channels, kh, kw = w.shape
    layer = Conv2d(in_channels, out_channels, (kh, kw),
                   stride=stride, padding=padding, bias=bias is not None)
    layer.weight.data = np.array(w, dtype=np.float64)
    if bias is not None:
        layer.bias.data = np.array(bias, dtype=np.float64)
    xt = Tensor(np.array(x, dtype=np.float64), requires_grad=True)
    out = layer(xt)
    (out * Tensor(np.array(g, dtype=np.float64))).sum().backward()
    bias_grad = layer.bias.grad if bias is not None else None
    return xt.grad, layer.weight.grad, bias_grad, out.data.shape


def _out_hw(x_shape, w_shape, stride, padding):
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    _, _, h, w = x_shape
    _, _, kh, kw = w_shape
    return ((h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1)


# --------------------------------------------------------------------------
# Core input-gradient correctness
# --------------------------------------------------------------------------

def _check_input_grad(x, w, stride=1, padding=0, seed=0):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    oh, ow = _out_hw(x.shape, w.shape, stride, padding)
    n, o = x.shape[0], w.shape[0]
    g = np.round(np.random.default_rng(seed).standard_normal((n, o, oh, ow)), 3)
    ref_x, _, _, _ = _stable_grads(x, w, g, stride, padding)
    gc, wc = _core(g), _core(w)
    grad = gc.conv2d_input_backward(wc, input_shape=x.shape,
                                    stride=stride, padding=padding)
    ok = np.allclose(grad.to_numpy(), ref_x, atol=1e-9)
    assert grad.shape == x.shape
    grad.close(); gc.close(); wc.close()
    assert ok


def test_input_grad_simple():
    _check_input_grad(np.arange(1, 10).reshape(1, 1, 3, 3),
                      [[[[1.0, 0.0], [0.0, 1.0]]]])


def test_input_grad_multiple_input_channels():
    rng = np.random.default_rng(1)
    _check_input_grad(rng.standard_normal((1, 3, 5, 5)),
                      rng.standard_normal((2, 3, 2, 2)))


def test_input_grad_multiple_output_channels():
    rng = np.random.default_rng(2)
    _check_input_grad(rng.standard_normal((1, 2, 5, 5)),
                      rng.standard_normal((4, 2, 3, 3)), padding=1)


def test_input_grad_batch():
    rng = np.random.default_rng(3)
    _check_input_grad(rng.standard_normal((3, 2, 4, 4)),
                      rng.standard_normal((2, 2, 2, 2)))


def test_input_grad_padding_and_stride():
    rng = np.random.default_rng(4)
    _check_input_grad(rng.standard_normal((2, 2, 7, 6)),
                      rng.standard_normal((3, 2, 3, 3)), stride=(2, 1), padding=(1, 0))


def test_input_grad_rectangular():
    rng = np.random.default_rng(5)
    _check_input_grad(rng.standard_normal((1, 2, 6, 4)),
                      rng.standard_normal((2, 2, 2, 3)))


# --------------------------------------------------------------------------
# Core weight-gradient correctness
# --------------------------------------------------------------------------

def _check_weight_grad(x, w, stride=1, padding=0, seed=0):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    oh, ow = _out_hw(x.shape, w.shape, stride, padding)
    n, o = x.shape[0], w.shape[0]
    g = np.round(np.random.default_rng(seed).standard_normal((n, o, oh, ow)), 3)
    _, ref_w, _, _ = _stable_grads(x, w, g, stride, padding)
    gc, xc = _core(g), _core(x)
    grad = gc.conv2d_weight_backward(xc, weight_shape=w.shape,
                                     stride=stride, padding=padding)
    ok = np.allclose(grad.to_numpy(), ref_w, atol=1e-9)
    assert grad.shape == w.shape
    grad.close(); gc.close(); xc.close()
    assert ok


def test_weight_grad_hand_case():
    _check_weight_grad(np.arange(1, 10).reshape(1, 1, 3, 3),
                       [[[[1.0, 0.0], [0.0, 1.0]]]])


def test_weight_grad_batch_and_spatial_accumulation():
    rng = np.random.default_rng(6)
    _check_weight_grad(rng.standard_normal((3, 2, 5, 5)),
                       rng.standard_normal((2, 2, 2, 2)))


def test_weight_grad_multiple_channels():
    rng = np.random.default_rng(7)
    _check_weight_grad(rng.standard_normal((2, 3, 6, 6)),
                       rng.standard_normal((4, 3, 3, 3)), padding=1)


def test_weight_grad_padding_stride_rectangular():
    rng = np.random.default_rng(8)
    _check_weight_grad(rng.standard_normal((2, 2, 7, 5)),
                       rng.standard_normal((3, 2, 3, 2)), stride=(2, 1), padding=(1, 0))


# --------------------------------------------------------------------------
# Non-contiguous operands (Policy B)
# --------------------------------------------------------------------------

def _noncontig_nchw(array):
    array = np.asarray(array, np.float64)
    base = cpp.NativeTensorCore.from_array(
        np.ascontiguousarray(array.transpose(0, 1, 3, 2)))
    view = base.transpose(0, 1, 3, 2)
    assert not view.contiguous
    return base, view


def test_input_grad_non_contiguous_grad_output():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((1, 2, 5, 5)); w = rng.standard_normal((3, 2, 3, 3))
    oh, ow = _out_hw(x.shape, w.shape, 1, 1)
    g = rng.standard_normal((1, 3, oh, ow))
    ref_x, _, _, _ = _stable_grads(x, w, g, 1, 1)
    base, gv = _noncontig_nchw(g)
    wc = _core(w)
    grad = gv.conv2d_input_backward(wc, input_shape=x.shape, stride=1, padding=1)
    assert np.allclose(grad.to_numpy(), ref_x, atol=1e-9)
    assert np.allclose(gv.to_numpy(), g, atol=1e-12)  # source untouched
    grad.close(); base.close(); wc.close()


def test_input_grad_non_contiguous_weight():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((1, 2, 5, 5)); w = rng.standard_normal((3, 2, 2, 2))
    oh, ow = _out_hw(x.shape, w.shape, 1, 0)
    g = rng.standard_normal((1, 3, oh, ow))
    ref_x, _, _, _ = _stable_grads(x, w, g, 1, 0)
    gc = _core(g)
    base, wv = _noncontig_nchw(w)
    grad = gc.conv2d_input_backward(wv, input_shape=x.shape, stride=1, padding=0)
    assert np.allclose(grad.to_numpy(), ref_x, atol=1e-9)
    grad.close(); gc.close(); base.close()


def test_weight_grad_non_contiguous_operands():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((2, 2, 5, 4)); w = rng.standard_normal((3, 2, 2, 2))
    oh, ow = _out_hw(x.shape, w.shape, 1, 0)
    g = rng.standard_normal((2, 3, oh, ow))
    _, ref_w, _, _ = _stable_grads(x, w, g, 1, 0)
    base_g, gv = _noncontig_nchw(g)
    base_x, xv = _noncontig_nchw(x)
    grad = gv.conv2d_weight_backward(xv, weight_shape=w.shape, stride=1, padding=0)
    assert np.allclose(grad.to_numpy(), ref_w, atol=1e-9)
    grad.close(); base_g.close(); base_x.close()


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------

def test_backward_output_ownership_and_layout():
    rng = np.random.default_rng(13)
    x = rng.standard_normal((1, 2, 4, 4)); w = rng.standard_normal((3, 2, 2, 2))
    oh, ow = _out_hw(x.shape, w.shape, 1, 0)
    gc = _core(rng.standard_normal((1, 3, oh, ow))); wc = _core(w); xc = _core(x)
    gin = gc.conv2d_input_backward(wc, input_shape=x.shape)
    gwt = gc.conv2d_weight_backward(xc, weight_shape=w.shape)
    for grad, shape in ((gin, x.shape), (gwt, w.shape)):
        assert grad.shape == shape
        assert grad.strides == cpp.row_major_strides(shape)
        assert grad.contiguous and grad.offset == 0 and grad._owns_storage
        assert grad.dtype == "float64" and grad.device == "cpu"
    assert gin._storage is not gc._storage and gwt._storage is not xc._storage
    for t in (gin, gwt, gc, wc, xc):
        t.close()


def test_backward_inputs_unchanged():
    rng = np.random.default_rng(14)
    x = rng.standard_normal((1, 2, 4, 4)); w = rng.standard_normal((3, 2, 2, 2))
    oh, ow = _out_hw(x.shape, w.shape, 1, 0)
    g = rng.standard_normal((1, 3, oh, ow))
    gc, wc, xc = _core(g), _core(w), _core(x)
    gc.conv2d_input_backward(wc, input_shape=x.shape).close()
    gc.conv2d_weight_backward(xc, weight_shape=w.shape).close()
    assert np.array_equal(gc.to_numpy(), g)
    assert np.array_equal(wc.to_numpy(), w)
    assert np.array_equal(xc.to_numpy(), x)
    for t in (gc, wc, xc):
        t.close()


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_input_grad_non_core_weight_rejected():
    gc = _core(np.ones((1, 1, 2, 2)))
    with pytest.raises(TypeError):
        gc.conv2d_input_backward(np.ones((1, 1, 2, 2)), input_shape=(1, 1, 3, 3))
    gc.close()


def test_input_grad_closed_operand_rejected():
    gc = _core(np.ones((1, 1, 2, 2))); wc = _core(np.ones((1, 1, 2, 2)))
    wc.close()
    with pytest.raises(RuntimeError):
        gc.conv2d_input_backward(wc, input_shape=(1, 1, 3, 3))
    gc.close()


def test_input_grad_wrong_grad_output_shape_rejected():
    gc = _core(np.ones((1, 1, 5, 5)))  # wrong out shape for the config below
    wc = _core(np.ones((1, 1, 2, 2)))
    with pytest.raises(ValueError):
        gc.conv2d_input_backward(wc, input_shape=(1, 1, 3, 3))  # expects (1,1,2,2)
    gc.close(); wc.close()


def test_input_grad_bad_input_shape_rank_rejected():
    gc = _core(np.ones((1, 1, 2, 2))); wc = _core(np.ones((1, 1, 2, 2)))
    with pytest.raises(ValueError):
        gc.conv2d_input_backward(wc, input_shape=(3, 3))
    gc.close(); wc.close()


def test_weight_grad_channel_mismatch_rejected():
    gc = _core(np.ones((1, 3, 2, 2))); xc = _core(np.ones((1, 2, 3, 3)))
    with pytest.raises(ValueError):
        gc.conv2d_weight_backward(xc, weight_shape=(3, 5, 2, 2))  # C=5 != input C=2
    gc.close(); xc.close()


# --------------------------------------------------------------------------
# Failure atomicity
# --------------------------------------------------------------------------

@needs_fault_injection
def test_input_grad_allocation_failure_is_memoryerror():
    gc = _core(np.ones((1, 1, 2, 2))); wc = _core(np.ones((1, 1, 2, 2)))
    cpp._arm_alloc_failure(1)  # contiguous operands -> only the output allocs
    with pytest.raises(MemoryError):
        gc.conv2d_input_backward(wc, input_shape=(1, 1, 3, 3))
    # No stale error; a later call succeeds.
    recovered = gc.conv2d_input_backward(wc, input_shape=(1, 1, 3, 3))
    assert recovered.shape == (1, 1, 3, 3)
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    recovered.close(); gc.close(); wc.close()


@needs_fault_injection
def test_weight_grad_allocation_failure_is_memoryerror():
    gc = _core(np.ones((1, 1, 2, 2))); xc = _core(np.ones((1, 1, 3, 3)))
    cpp._arm_alloc_failure(1)
    with pytest.raises(MemoryError):
        gc.conv2d_weight_backward(xc, weight_shape=(1, 1, 2, 2))
    recovered = gc.conv2d_weight_backward(xc, weight_shape=(1, 1, 2, 2))
    assert recovered.shape == (1, 1, 2, 2)
    recovered.close(); gc.close(); xc.close()


# --------------------------------------------------------------------------
# Raw C ABI direct tests
#
# The ABI receives handles + offsets + dimensions only — never stride
# arrays — so it validates that metadata and bounds-checks each contiguous
# span; it never inspects logical contiguity. None of these assert the ABI
# recognizes a stride pattern that was never supplied.
# --------------------------------------------------------------------------

def _handles_input_backward(g_shape, w_shape, out_shape):
    gc = cpp.NativeTensorCore.from_array(np.ones(g_shape))
    wc = cpp.NativeTensorCore.from_array(np.ones(w_shape))
    out = cpp.NativeTensorCore.zeros(out_shape)
    return gc, wc, out


def test_raw_input_backward_valid_call_reaches_kernel():
    lib = cpp._require_library()
    gc, wc, out = _handles_input_backward((1, 1, 2, 2), (1, 1, 2, 2), (1, 1, 3, 3))
    # N1 C1 H3 W3 O1 kh2 kw2 s1 s1 p0 p0 outh2 outw2
    lib.tf_core_conv2d_input_backward(
        gc._storage._require_open(), 0, wc._storage._require_open(), 0,
        out._storage._require_open(),
        1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2)
    assert lib.tf_last_error_code() == cpp.TF_OK
    assert out.to_numpy().shape == (1, 1, 3, 3)
    gc.close(); wc.close(); out.close()


def test_raw_input_backward_null_handle_is_valueerror():
    lib = cpp._require_library()
    _, wc, out = _handles_input_backward((1, 1, 2, 2), (1, 1, 2, 2), (1, 1, 3, 3))
    with pytest.raises(ValueError):
        lib.tf_core_conv2d_input_backward(
            None, 0, wc._storage._require_open(), 0,
            out._storage._require_open(),
            1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2)
    assert lib.tf_last_error_code() == cpp.TF_OK
    wc.close(); out.close()


def test_raw_input_backward_negative_offset_is_valueerror():
    lib = cpp._require_library()
    gc, wc, out = _handles_input_backward((1, 1, 2, 2), (1, 1, 2, 2), (1, 1, 3, 3))
    with pytest.raises(ValueError):
        lib.tf_core_conv2d_input_backward(
            gc._storage._require_open(), -1, wc._storage._require_open(), 0,
            out._storage._require_open(),
            1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2)
    gc.close(); wc.close(); out.close()


def test_raw_input_backward_invalid_dimension_is_valueerror():
    lib = cpp._require_library()
    gc, wc, out = _handles_input_backward((1, 1, 2, 2), (1, 1, 2, 2), (1, 1, 3, 3))
    with pytest.raises(ValueError):
        lib.tf_core_conv2d_input_backward(
            gc._storage._require_open(), 0, wc._storage._require_open(), 0,
            out._storage._require_open(),
            0, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2)  # batch = 0
    gc.close(); wc.close(); out.close()


def test_raw_input_backward_output_shape_mismatch_is_valueerror():
    lib = cpp._require_library()
    gc, wc, out = _handles_input_backward((1, 1, 2, 2), (1, 1, 2, 2), (1, 1, 3, 3))
    with pytest.raises(ValueError):
        lib.tf_core_conv2d_input_backward(
            gc._storage._require_open(), 0, wc._storage._require_open(), 0,
            out._storage._require_open(),
            1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 3, 2)  # claimed out_h 3 != floor 2
    gc.close(); wc.close(); out.close()


def test_raw_input_backward_undersized_span_is_valueerror():
    lib = cpp._require_library()
    # grad_input storage too small: claim (1,1,3,3)=9 but allocate 4.
    gc = cpp.NativeTensorCore.from_array(np.ones((1, 1, 2, 2)))
    wc = cpp.NativeTensorCore.from_array(np.ones((1, 1, 2, 2)))
    small = cpp.NativeTensorCore.zeros((2, 2))  # 4 < 9
    with pytest.raises(ValueError):
        lib.tf_core_conv2d_input_backward(
            gc._storage._require_open(), 0, wc._storage._require_open(), 0,
            small._storage._require_open(),
            1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2)
    gc.close(); wc.close(); small.close()


def test_raw_weight_backward_valid_and_null_handle():
    lib = cpp._require_library()
    gc = cpp.NativeTensorCore.from_array(np.ones((1, 1, 2, 2)))
    xc = cpp.NativeTensorCore.from_array(np.ones((1, 1, 3, 3)))
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    lib.tf_core_conv2d_weight_backward(
        gc._storage._require_open(), 0, xc._storage._require_open(), 0,
        out._storage._require_open(),
        1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2)
    assert lib.tf_last_error_code() == cpp.TF_OK
    with pytest.raises(ValueError):
        lib.tf_core_conv2d_weight_backward(
            gc._storage._require_open(), 0, None, 0,
            out._storage._require_open(),
            1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2)
    gc.close(); xc.close(); out.close()
