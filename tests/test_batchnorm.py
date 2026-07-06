import numpy as np
import pytest

from tensorforge import Tensor
from tensorforge.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential


def test_constructor_validation():
    for bad in (0, -1, 2.5, "4", True):
        with pytest.raises(ValueError):
            BatchNorm1d(bad)
    for bad_eps in (0.0, -1e-5, "tiny"):
        with pytest.raises(ValueError):
            BatchNorm1d(3, eps=bad_eps)
    for bad_momentum in (-0.1, 1.5, "fast"):
        with pytest.raises(ValueError):
            BatchNorm1d(3, momentum=bad_momentum)


def test_default_state():
    bn = BatchNorm1d(4)
    assert np.array_equal(bn.gamma.data, np.ones(4))
    assert np.array_equal(bn.beta.data, np.zeros(4))
    assert np.array_equal(bn.running_mean, np.zeros(4))
    assert np.array_equal(bn.running_var, np.ones(4))
    assert bn.gamma.requires_grad and bn.beta.requires_grad


def test_rejects_bad_input_shapes():
    bn = BatchNorm1d(3)
    with pytest.raises(ValueError):
        bn(Tensor(np.ones(3)))  # 1-D
    with pytest.raises(ValueError):
        bn(Tensor(np.ones((4, 2))))  # wrong feature count


def test_training_output_is_normalized():
    np.random.seed(0)
    x = Tensor(np.random.randn(50, 3) * np.array([2.0, 5.0, 0.5]) + np.array([1.0, -3.0, 10.0]))
    out = BatchNorm1d(3)(x).data
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(out.var(axis=0), 1.0, atol=1e-3)  # off by eps only


def test_training_updates_running_stats():
    np.random.seed(0)
    x_np = np.random.randn(20, 2) + np.array([3.0, -1.0])
    bn = BatchNorm1d(2, momentum=0.1)
    bn(Tensor(x_np))
    expected_mean = 0.9 * np.zeros(2) + 0.1 * x_np.mean(axis=0)
    expected_var = 0.9 * np.ones(2) + 0.1 * x_np.var(axis=0)
    assert np.allclose(bn.running_mean, expected_mean)
    assert np.allclose(bn.running_var, expected_var)


def test_eval_uses_running_stats_and_does_not_update_them():
    bn = BatchNorm1d(2).eval()
    bn.running_mean = np.array([1.0, -1.0])
    bn.running_var = np.array([4.0, 0.25])
    mean_before = bn.running_mean.copy()
    var_before = bn.running_var.copy()

    x_np = np.array([[3.0, 0.0], [1.0, -1.0]])
    out = bn(Tensor(x_np)).data
    expected = (x_np - mean_before) / np.sqrt(var_before + bn.eps)
    assert np.allclose(out, expected)
    assert np.array_equal(bn.running_mean, mean_before)
    assert np.array_equal(bn.running_var, var_before)


def test_gamma_and_beta_scale_and_shift():
    np.random.seed(0)
    x = Tensor(np.random.randn(30, 2))
    bn = BatchNorm1d(2)
    plain = bn(x).data
    bn.gamma.data = np.array([2.0, -1.0])
    bn.beta.data = np.array([10.0, 5.0])
    scaled = bn(x).data
    assert np.allclose(scaled, plain * np.array([2.0, -1.0]) + np.array([10.0, 5.0]))


def test_gamma_beta_gradients():
    np.random.seed(0)
    x_np = np.random.randn(10, 3)
    bn = BatchNorm1d(3)
    out = bn(Tensor(x_np))
    out.sum().backward()
    # out = gamma * x_hat + beta, so dL/dbeta = N per feature and
    # dL/dgamma = sum of x_hat, which is ~0 because x_hat is centered.
    assert np.allclose(bn.beta.grad, 10.0)
    assert np.allclose(bn.gamma.grad, 0.0, atol=1e-10)
    # A weighted loss gives gamma a nontrivial gradient.
    bn.zero_grad()
    weights = np.random.randn(10, 3)
    (bn(Tensor(x_np)) * weights).sum().backward()
    x_hat = (x_np - x_np.mean(axis=0)) / np.sqrt(x_np.var(axis=0) + bn.eps)
    assert np.allclose(bn.gamma.grad, (weights * x_hat).sum(axis=0), atol=1e-8)


def test_input_gradients_shape_and_finite():
    np.random.seed(0)
    x = Tensor(np.random.randn(8, 4), requires_grad=True)
    (BatchNorm1d(4)(x) ** 2).sum().backward()
    assert x.grad.shape == (8, 4)
    assert np.all(np.isfinite(x.grad))


def test_input_gradient_matches_finite_differences():
    np.random.seed(0)
    x_np = np.random.randn(5, 2)
    weights = np.random.randn(5, 2)
    bn = BatchNorm1d(2)
    bn.gamma.data = np.array([1.5, -0.5])
    bn.beta.data = np.array([0.3, 1.0])

    def loss_at(v):
        return float((bn(Tensor(v)) * weights).sum().data)

    x = Tensor(x_np.copy(), requires_grad=True)
    (bn(x) * weights).sum().backward()

    eps = 1e-6
    numeric = np.zeros_like(x_np)
    for i in range(x_np.shape[0]):
        for j in range(x_np.shape[1]):
            plus = x_np.copy()
            plus[i, j] += eps
            minus = x_np.copy()
            minus[i, j] -= eps
            numeric[i, j] = (loss_at(plus) - loss_at(minus)) / (2 * eps)
    # The batch statistics depend on every element, so this checks the
    # full batchnorm backward, not just the elementwise part.
    assert np.allclose(x.grad, numeric, atol=1e-5)


def test_respects_train_and_eval_in_sequential():
    np.random.seed(0)
    model = Sequential(Linear(2, 3), BatchNorm1d(3))
    bn = model.modules[1]
    x = Tensor(np.random.randn(6, 2))

    model(x)  # training mode: running stats move
    assert not np.allclose(bn.running_mean, 0.0)

    model.eval()
    frozen_mean = bn.running_mean.copy()
    eval_a = model(x).data
    eval_b = model(x).data
    assert np.array_equal(eval_a, eval_b)
    assert np.array_equal(bn.running_mean, frozen_mean)  # eval never updates

    model.train()
    model(x)
    assert not np.array_equal(bn.running_mean, frozen_mean)  # training does


def test_works_with_dropout_in_same_model():
    np.random.seed(0)
    model = Sequential(
        Linear(2, 8),
        BatchNorm1d(8),
        ReLU(),
        Dropout(p=0.5, seed=0),
        Linear(8, 1),
    )
    x = Tensor(np.random.randn(10, 2), requires_grad=True)
    model(x).sum().backward()
    assert x.grad.shape == (10, 2)
    for param in model.parameters():
        assert param.grad is not None and np.all(np.isfinite(param.grad))

    model.eval()
    a = model(Tensor(np.random.randn(4, 2))).data
    assert np.all(np.isfinite(a))
