import numpy as np

from tensorforge import Tensor
from tensorforge.nn import mse_loss


def test_mse_forward_value():
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([2.0, 2.0, 5.0])
    loss = mse_loss(pred, target)
    # ((1-2)^2 + 0 + (3-5)^2) / 3 = 5/3
    assert np.allclose(loss.data, 5.0 / 3.0)


def test_mse_zero_when_prediction_matches_target():
    pred = Tensor([1.5, -2.0])
    assert np.allclose(mse_loss(pred, [1.5, -2.0]).data, 0.0)


def test_mse_gradient_wrt_prediction():
    pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = Tensor([2.0, 2.0, 5.0])
    mse_loss(pred, target).backward()
    # d/dpred mean((pred - target)^2) = 2 * (pred - target) / N
    expected = 2.0 * (pred.data - target.data) / 3.0
    assert np.allclose(pred.grad, expected)
    assert target.grad is None  # target never asked for gradients


def test_mse_accepts_plain_targets():
    pred = Tensor([[1.0, 2.0]], requires_grad=True)
    loss = mse_loss(pred, [[0.0, 0.0]])  # plain list target
    loss.backward()
    assert np.allclose(loss.data, 2.5)
    assert np.allclose(pred.grad, [[1.0, 2.0]])

    pred2 = Tensor(3.0, requires_grad=True)
    loss2 = mse_loss(pred2, 1.0)  # plain scalar target
    loss2.backward()
    assert np.allclose(loss2.data, 4.0)
    assert np.allclose(pred2.grad, 4.0)
