import numpy as np
import pytest

from tensorforge import Tensor
from tensorforge.nn import LayerNorm, Linear, ReLU, Sequential

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


def test_forward_shape_matches_input():
    rng = np.random.default_rng(0)
    assert LayerNorm(4)(Tensor(rng.normal(size=(5, 4)))).data.shape == (5, 4)
    assert LayerNorm((3, 2, 2))(Tensor(rng.normal(size=(5, 3, 2, 2)))).data.shape == (5, 3, 2, 2)


def test_normalizes_each_sample_independently():
    rng = np.random.default_rng(1)
    # Rows with wildly different scales and offsets.
    x_np = rng.normal(size=(6, 4)) * np.array([[1.0], [10.0], [0.1], [5.0], [100.0], [2.0]])
    x_np += np.array([[0.0], [50.0], [-3.0], [7.0], [-200.0], [1.0]])
    ln = LayerNorm(4)
    out = ln(Tensor(x_np)).data
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-7)
    # Output variance is var/(var + eps): essentially 1, except for
    # rows whose variance is tiny relative to eps.
    row_var = x_np.var(axis=1)
    assert np.allclose(out.var(axis=1), row_var / (row_var + ln.eps))
    assert np.all(out.var(axis=1) > 0.97)


def test_4d_normalized_shape():
    rng = np.random.default_rng(2)
    x_np = rng.normal(size=(3, 2, 4, 4)) * 7.0 + 3.0
    out = LayerNorm((2, 4, 4))(Tensor(x_np)).data
    # Each sample normalized over all of C, H, W.
    per_sample = out.reshape(3, -1)
    assert np.allclose(per_sample.mean(axis=1), 0.0, atol=1e-7)
    assert np.allclose(per_sample.var(axis=1), 1.0, atol=1e-3)


def test_affine_parameters_exist():
    ln = LayerNorm((3, 2))
    params = ln.parameters()
    assert len(params) == 2
    assert ln.weight.data.shape == (3, 2)
    assert ln.bias.data.shape == (3, 2)
    assert np.array_equal(ln.weight.data, np.ones((3, 2)))
    assert np.array_equal(ln.bias.data, np.zeros((3, 2)))


def test_no_affine_means_no_parameters():
    ln = LayerNorm(4, elementwise_affine=False)
    assert list(ln.parameters()) == []
    assert not hasattr(ln, "weight")
    out = ln(Tensor(np.random.default_rng(3).normal(size=(2, 4))))
    assert out.data.shape == (2, 4)


def test_known_value_affine():
    rng = np.random.default_rng(4)
    x_np = rng.normal(size=(3, 4))
    ln = LayerNorm(4)
    ln.weight.data = np.array([2.0, -1.0, 0.5, 3.0])
    ln.bias.data = np.array([10.0, 0.0, -5.0, 1.0])

    mean = x_np.mean(axis=1, keepdims=True)
    var = x_np.var(axis=1, keepdims=True)
    x_hat = (x_np - mean) / np.sqrt(var + ln.eps)
    expected = x_hat * ln.weight.data + ln.bias.data
    assert np.allclose(ln(Tensor(x_np)).data, expected)


def test_same_output_in_train_and_eval_mode():
    rng = np.random.default_rng(5)
    x = Tensor(rng.normal(size=(4, 6)))
    ln = LayerNorm(6)
    train_out = ln(x).data
    eval_out = ln.eval()(x).data
    assert np.array_equal(train_out, eval_out)
    assert list(ln.named_buffers()) == []  # no running stats


def test_validation_errors():
    for bad in (0, -3, 2.5, True, (), (2, 0), (2, True), "4", None):
        with pytest.raises(ValueError):
            LayerNorm(bad)
    for bad_eps in (0, -1e-5, "tiny", True):
        with pytest.raises(ValueError):
            LayerNorm(4, eps=bad_eps)
    for bad_affine in (1, 0, "yes", None):
        with pytest.raises(ValueError):
            LayerNorm(4, elementwise_affine=bad_affine)

    ln = LayerNorm((3, 2))
    with pytest.raises(ValueError):
        ln(Tensor(np.zeros((5, 2, 3))))  # trailing dims don't match
    with pytest.raises(ValueError):
        ln(Tensor(np.zeros(3)))  # rank too low


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(6)
    x_np = rng.normal(size=(4, 3))
    upstream = rng.normal(size=(4, 3))
    ln = LayerNorm(3)
    ln.weight.data = np.array([1.5, -0.5, 2.0])
    ln.bias.data = np.array([0.3, 1.0, -2.0])

    def loss_value():
        return float((ln(Tensor(x_np)) * upstream).sum().data)

    x = Tensor(x_np.copy(), requires_grad=True)
    (ln(x) * upstream).sum().backward()

    assert np.allclose(x.grad, _numerical_grad(loss_value, x_np), atol=1e-5)
    assert np.allclose(ln.weight.grad, _numerical_grad(loss_value, ln.weight.data), atol=1e-5)
    assert np.allclose(ln.bias.grad, _numerical_grad(loss_value, ln.bias.data), atol=1e-5)


def test_gradients_match_finite_differences_multidim():
    rng = np.random.default_rng(7)
    x_np = rng.normal(size=(2, 2, 3))
    upstream = rng.normal(size=(2, 2, 3))
    ln = LayerNorm((2, 3))

    def loss_value():
        return float((ln(Tensor(x_np)) * upstream).sum().data)

    x = Tensor(x_np.copy(), requires_grad=True)
    (ln(x) * upstream).sum().backward()

    assert np.allclose(x.grad, _numerical_grad(loss_value, x_np), atol=1e-5)
    assert np.allclose(ln.weight.grad, _numerical_grad(loss_value, ln.weight.data), atol=1e-5)


def test_sequential_integration():
    np.random.seed(0)
    model = Sequential(Linear(4, 8), LayerNorm(8), ReLU(), Linear(8, 3))
    x = Tensor(np.random.randn(5, 4), requires_grad=True)
    out = model(x)
    assert out.data.shape == (5, 3)

    out.sum().backward()
    assert x.grad.shape == (5, 4)
    params = model.parameters()
    assert len(params) == 6  # two Linears + LayerNorm weight/bias
    for param in params:
        assert param.grad is not None
        assert param.grad.shape == param.data.shape
        assert np.all(np.isfinite(param.grad))


def test_public_api_exports():
    import tensorforge
    import tensorforge.nn

    assert tensorforge.LayerNorm is tensorforge.nn.LayerNorm
