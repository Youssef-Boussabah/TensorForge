import numpy as np
import pytest

from tensorforge import Tensor
from tensorforge.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential


def test_forward_shape():
    out = MaxPool2d(2)(Tensor(np.random.default_rng(0).normal(size=(2, 3, 8, 8))))
    assert out.data.shape == (2, 3, 4, 4)


def test_known_values_no_padding():
    x = Tensor(np.arange(16.0).reshape(1, 1, 4, 4))
    out = MaxPool2d(2)(x)
    assert np.array_equal(out.data[0, 0], [[5.0, 7.0], [13.0, 15.0]])


def test_rectangular_kernel_and_stride():
    x = Tensor(np.arange(16.0).reshape(1, 1, 4, 4))
    out = MaxPool2d(kernel_size=(2, 1), stride=(1, 2))(x)
    # out[i, j] = max(x[i, 2j], x[i+1, 2j]) = x[i+1, 2j] for ascending data.
    expected = np.array([[(i + 1) * 4.0 + 2 * j for j in range(2)] for i in range(3)])
    assert out.data.shape == (1, 1, 3, 2)
    assert np.array_equal(out.data[0, 0], expected)


def test_padding_uses_negative_infinity():
    # All-negative input: zero padding would wrongly win every edge
    # window; -inf padding must never win.
    x = Tensor(np.array([[[[-1.0, -2.0], [-3.0, -4.0]]]]))
    out = MaxPool2d(2, stride=2, padding=1)(x)
    assert out.data.shape == (1, 1, 2, 2)
    assert np.array_equal(out.data[0, 0], [[-1.0, -2.0], [-3.0, -4.0]])
    assert np.all(np.isfinite(out.data))


def test_default_stride_equals_kernel_size():
    pool = MaxPool2d((3, 2))
    assert pool.stride == (3, 2)
    x = Tensor(np.arange(36.0).reshape(1, 1, 6, 6))
    assert np.array_equal(MaxPool2d(3)(x).data, MaxPool2d(3, stride=3)(x).data)


def test_tie_gradient_goes_to_first_max_in_row_major_order():
    x = Tensor(np.array([[[[5.0, 5.0], [3.0, 5.0]]]]), requires_grad=True)
    out = MaxPool2d(2)(x)
    assert out.data[0, 0, 0, 0] == 5.0
    out.sum().backward()
    # Three positions tie at 5.0; only the row-major first one wins.
    assert np.array_equal(x.grad[0, 0], [[1.0, 0.0], [0.0, 0.0]])


def test_backward_no_padding():
    x = Tensor(np.arange(16.0).reshape(1, 1, 4, 4), requires_grad=True)
    out = MaxPool2d(2)(x)
    weights = np.array([[[[10.0, 20.0], [30.0, 40.0]]]])
    (out * weights).sum().backward()
    assert x.grad.shape == (1, 1, 4, 4)
    expected = np.zeros((4, 4))
    expected[1, 1] = 10.0   # max 5
    expected[1, 3] = 20.0   # max 7
    expected[3, 1] = 30.0   # max 13
    expected[3, 3] = 40.0   # max 15
    assert np.array_equal(x.grad[0, 0], expected)


def test_backward_with_padding_discards_padded_gradient():
    x = Tensor(np.array([[[[-1.0, -2.0], [-3.0, -4.0]]]]), requires_grad=True)
    MaxPool2d(2, stride=2, padding=1)(x).sum().backward()
    # Each input cell is the (only finite) max of its own window, so
    # every cell gets gradient exactly 1 and nothing leaks from padding.
    assert x.grad.shape == (1, 1, 2, 2)
    assert np.array_equal(x.grad[0, 0], [[1.0, 1.0], [1.0, 1.0]])


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(6)
    x_np = rng.normal(size=(2, 2, 5, 5))  # continuous values: no ties
    upstream = rng.normal(size=(2, 2, 3, 3))
    pool = MaxPool2d(2, stride=2, padding=1)

    x = Tensor(x_np.copy(), requires_grad=True)
    (pool(x) * upstream).sum().backward()

    eps = 1e-6
    numeric = np.zeros_like(x_np)
    it = np.nditer(x_np, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        for sign, slot in ((1, 0), (-1, 1)):
            probe = x_np.copy()
            probe[idx] += sign * eps
            value = float((pool(Tensor(probe)) * upstream).sum().data)
            if slot == 0:
                plus = value
            else:
                minus = value
        numeric[idx] = (plus - minus) / (2 * eps)
    assert np.allclose(x.grad, numeric, atol=1e-5)


def test_validation_errors():
    with pytest.raises(ValueError):
        MaxPool2d(0)
    with pytest.raises(ValueError):
        MaxPool2d(2, stride=0)
    with pytest.raises(ValueError):
        MaxPool2d(2, padding=-1)
    with pytest.raises(ValueError):
        MaxPool2d((2, 2, 2))
    pool = MaxPool2d(2)
    with pytest.raises(ValueError):
        pool(Tensor(np.zeros((2, 4, 4))))  # not 4-D
    with pytest.raises(ValueError):
        MaxPool2d(5)(Tensor(np.zeros((1, 1, 3, 3))))  # kernel too big


def test_has_no_parameters():
    assert list(MaxPool2d(2).parameters()) == []
    assert MaxPool2d(2).num_parameters() == 0


def test_cnn_stack_with_pooling():
    np.random.seed(0)
    model = Sequential(
        Conv2d(1, 2, kernel_size=3),   # (N, 1, 6, 6) -> (N, 2, 4, 4)
        ReLU(),
        MaxPool2d(2),                  # -> (N, 2, 2, 2)
        Flatten(),                     # -> (N, 8)
        Linear(2 * 2 * 2, 3),
    )
    x = Tensor(np.random.randn(4, 1, 6, 6), requires_grad=True)
    out = model(x)
    assert out.data.shape == (4, 3)

    out.sum().backward()
    assert x.grad.shape == (4, 1, 6, 6)
    params = model.parameters()
    assert len(params) == 4  # pooling adds none
    for param in params:
        assert param.grad is not None
        assert param.grad.shape == param.data.shape
        assert np.all(np.isfinite(param.grad))


def test_public_api_exports():
    import tensorforge
    import tensorforge.nn

    assert tensorforge.MaxPool2d is tensorforge.nn.MaxPool2d
