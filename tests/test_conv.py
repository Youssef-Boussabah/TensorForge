import numpy as np
import pytest

from tensorforge import Tensor
from tensorforge.nn import Conv2d, Flatten, Linear, ReLU, Sequential


def _ref_conv(x, w, b, stride, padding):
    """Reference convolution: pure Python loops, no cleverness."""
    n, _, h, width = x.shape
    o, c, kh, kw = w.shape
    sh, sw = stride
    ph, pw = padding
    padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (width + 2 * pw - kw) // sw + 1
    out = np.zeros((n, o, out_h, out_w))
    for img in range(n):
        for oc in range(o):
            for i in range(out_h):
                for j in range(out_w):
                    window = padded[img, :, i * sh : i * sh + kh, j * sw : j * sw + kw]
                    out[img, oc, i, j] = (window * w[oc]).sum()
            if b is not None:
                out[img, oc] += b[oc]
    return out


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------


def test_flatten_forward_shapes():
    assert Flatten()(Tensor(np.zeros((3, 2, 4, 5)))).data.shape == (3, 40)
    assert Flatten()(Tensor(np.zeros((7, 6)))).data.shape == (7, 6)
    assert Flatten()(Tensor(np.zeros((2, 3, 4)))).data.shape == (2, 12)


def test_flatten_rejects_low_rank_input():
    with pytest.raises(ValueError):
        Flatten()(Tensor(3.0))
    with pytest.raises(ValueError):
        Flatten()(Tensor(np.zeros(5)))


def test_flatten_backward_restores_shape():
    x_np = np.arange(24.0).reshape(2, 3, 2, 2)
    weights = np.arange(24.0).reshape(2, 12) + 1.0
    x = Tensor(x_np.copy(), requires_grad=True)
    (Flatten()(x) * weights).sum().backward()
    assert x.grad.shape == (2, 3, 2, 2)
    assert np.array_equal(x.grad, weights.reshape(2, 3, 2, 2))


# ---------------------------------------------------------------------------
# Conv2d forward
# ---------------------------------------------------------------------------


def test_forward_shape():
    np.random.seed(0)
    conv = Conv2d(3, 8, kernel_size=3)
    out = conv(Tensor(np.random.randn(2, 3, 10, 10)))
    assert out.data.shape == (2, 8, 8, 8)


def test_known_values_single_channel():
    conv = Conv2d(1, 1, kernel_size=2)
    conv.weight.data = np.ones((1, 1, 2, 2))
    conv.bias.data = np.array([10.0])
    x = Tensor(np.arange(9.0).reshape(1, 1, 3, 3))
    # Each output = sum of a 2x2 window + 10.
    expected = np.array([[[[0 + 1 + 3 + 4, 1 + 2 + 4 + 5],
                           [3 + 4 + 6 + 7, 4 + 5 + 7 + 8]]]]) + 10.0
    assert np.allclose(conv(x).data, expected)


def test_known_values_multi_channel():
    conv = Conv2d(2, 1, kernel_size=1)
    # Kernel picks 2 * channel0 - 1 * channel1 at every pixel.
    conv.weight.data = np.array([[[[2.0]], [[-1.0]]]])
    conv.bias.data = np.zeros(1)
    c0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    c1 = np.array([[10.0, 20.0], [30.0, 40.0]])
    x = Tensor(np.stack([c0, c1])[np.newaxis])
    assert np.allclose(conv(x).data[0, 0], 2.0 * c0 - c1)


def test_padding_matches_reference():
    np.random.seed(1)
    x_np = np.random.randn(2, 2, 5, 5)
    conv = Conv2d(2, 3, kernel_size=3, padding=1)
    out = conv(Tensor(x_np))
    assert out.data.shape == (2, 3, 5, 5)  # "same" spatial size
    ref = _ref_conv(x_np, conv.weight.data, conv.bias.data, (1, 1), (1, 1))
    assert np.allclose(out.data, ref)


def test_stride_matches_reference():
    np.random.seed(2)
    x_np = np.random.randn(1, 1, 6, 6)
    conv = Conv2d(1, 2, kernel_size=2, stride=2)
    out = conv(Tensor(x_np))
    assert out.data.shape == (1, 2, 3, 3)
    ref = _ref_conv(x_np, conv.weight.data, conv.bias.data, (2, 2), (0, 0))
    assert np.allclose(out.data, ref)


def test_rectangular_kernel_stride_padding():
    np.random.seed(3)
    x_np = np.random.randn(2, 3, 8, 7)
    conv = Conv2d(3, 4, kernel_size=(2, 3), stride=(2, 1), padding=(0, 1))
    ref = _ref_conv(x_np, conv.weight.data, conv.bias.data, (2, 1), (0, 1))
    assert np.allclose(conv(Tensor(x_np)).data, ref)


def test_bias_false():
    np.random.seed(0)
    conv = Conv2d(1, 2, kernel_size=2, bias=False)
    assert conv.bias is None
    assert len(conv.parameters()) == 1
    x_np = np.random.randn(1, 1, 4, 4)
    ref = _ref_conv(x_np, conv.weight.data, None, (1, 1), (0, 0))
    assert np.allclose(conv(Tensor(x_np)).data, ref)


def test_validation_errors():
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            Conv2d(bad, 1, kernel_size=2)
        with pytest.raises(ValueError):
            Conv2d(1, bad, kernel_size=2)
    with pytest.raises(ValueError):
        Conv2d(1, 1, kernel_size=0)
    with pytest.raises(ValueError):
        Conv2d(1, 1, kernel_size=2, stride=0)
    with pytest.raises(ValueError):
        Conv2d(1, 1, kernel_size=2, padding=-1)
    with pytest.raises(ValueError):
        Conv2d(1, 1, kernel_size=(2, 2, 2))

    conv = Conv2d(2, 1, kernel_size=2)
    with pytest.raises(ValueError):
        conv(Tensor(np.zeros((2, 2, 4))))  # not 4-D
    with pytest.raises(ValueError):
        conv(Tensor(np.zeros((1, 3, 4, 4))))  # wrong channel count
    with pytest.raises(ValueError):
        Conv2d(1, 1, kernel_size=5)(Tensor(np.zeros((1, 1, 3, 3))))  # kernel too big


# ---------------------------------------------------------------------------
# Conv2d gradients (finite differences)
# ---------------------------------------------------------------------------

EPS = 1e-6


def _numerical_grad(fn, array):
    grad = np.zeros_like(array)
    it = np.nditer(array, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = array[idx]
        array[idx] = original + EPS
        plus = fn()
        array[idx] = original - EPS
        minus = fn()
        array[idx] = original
        grad[idx] = (plus - minus) / (2.0 * EPS)
    return grad


def test_gradients_match_finite_differences():
    np.random.seed(4)
    x_np = np.random.randn(2, 2, 4, 4)
    upstream = np.random.randn(2, 3, 5, 5)  # fixed loss weights
    conv = Conv2d(2, 3, kernel_size=2, stride=1, padding=1)

    def loss_value():
        return float((conv(Tensor(x_np)) * upstream).sum().data)

    x = Tensor(x_np.copy(), requires_grad=True)
    (conv(x) * upstream).sum().backward()

    numeric_x = _numerical_grad(lambda: loss_value(), x_np)
    assert np.allclose(x.grad, numeric_x, atol=1e-5)

    numeric_w = _numerical_grad(loss_value, conv.weight.data)
    assert np.allclose(conv.weight.grad, numeric_w, atol=1e-5)

    numeric_b = _numerical_grad(loss_value, conv.bias.data)
    assert np.allclose(conv.bias.grad, numeric_b, atol=1e-5)


def test_gradients_with_stride_match_finite_differences():
    np.random.seed(5)
    x_np = np.random.randn(1, 1, 5, 5)
    upstream = np.random.randn(1, 2, 2, 2)
    conv = Conv2d(1, 2, kernel_size=3, stride=2)

    def loss_value():
        return float((conv(Tensor(x_np)) * upstream).sum().data)

    x = Tensor(x_np.copy(), requires_grad=True)
    (conv(x) * upstream).sum().backward()

    assert np.allclose(x.grad, _numerical_grad(loss_value, x_np), atol=1e-5)
    assert np.allclose(conv.weight.grad, _numerical_grad(loss_value, conv.weight.data), atol=1e-5)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_cnn_stack_in_sequential():
    np.random.seed(0)
    model = Sequential(
        Conv2d(1, 2, kernel_size=3),
        ReLU(),
        Flatten(),
        Linear(2 * 4 * 4, 3),
    )
    x = Tensor(np.random.randn(5, 1, 6, 6), requires_grad=True)
    out = model(x)
    assert out.data.shape == (5, 3)

    out.sum().backward()
    assert x.grad.shape == (5, 1, 6, 6)
    params = model.parameters()
    assert len(params) == 4  # conv weight+bias, linear weight+bias
    for param in params:
        assert param.grad is not None
        assert param.grad.shape == param.data.shape
        assert np.all(np.isfinite(param.grad))
