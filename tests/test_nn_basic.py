import numpy as np

from tensorforge import Tensor
from tensorforge.nn import Linear, Module, Parameter, ReLU, Sequential, Sigmoid, Tanh


def test_parameter_requires_grad_by_default():
    p = Parameter([1.0, 2.0])
    assert p.requires_grad
    assert isinstance(p, Tensor)


def test_linear_forward_shape():
    layer = Linear(3, 5)
    x = Tensor(np.random.randn(4, 3))
    y = layer(x)
    assert isinstance(y, Tensor)
    assert y.data.shape == (4, 5)


def test_linear_forward_value():
    layer = Linear(2, 1)
    x = Tensor([[1.0, 2.0]])
    y = layer(x)
    expected = x.data @ layer.weight.data + layer.bias.data
    assert np.allclose(y.data, expected)


def test_linear_exposes_weight_and_bias_as_parameters():
    layer = Linear(2, 3)
    params = layer.parameters()
    assert layer.weight in params
    assert layer.bias in params
    assert len(params) == 2
    assert layer.weight.data.shape == (2, 3)
    assert layer.bias.data.shape == (3,)


def test_linear_without_bias():
    layer = Linear(2, 3, bias=False)
    assert layer.bias is None
    assert len(layer.parameters()) == 1
    y = layer(Tensor([[1.0, 2.0]]))
    assert np.allclose(y.data, np.array([[1.0, 2.0]]) @ layer.weight.data)


def test_linear_backward_gradients():
    layer = Linear(2, 3)
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    layer(x).sum().backward()

    ones = np.ones((2, 3))  # upstream gradient from sum()
    assert np.allclose(layer.weight.grad, x.data.T @ ones)
    assert np.allclose(layer.bias.grad, ones.sum(axis=0))
    assert np.allclose(x.grad, ones @ layer.weight.data.T)


def test_relu_module():
    y = ReLU()(Tensor([[-1.0, 2.0]]))
    assert np.allclose(y.data, [[0.0, 2.0]])


def test_sigmoid_module():
    y = Sigmoid()(Tensor([0.0, 2.0]))
    assert np.allclose(y.data, 1.0 / (1.0 + np.exp(-np.array([0.0, 2.0]))))


def test_tanh_module():
    y = Tanh()(Tensor([0.5, -0.5]))
    assert np.allclose(y.data, np.tanh([0.5, -0.5]))


def test_sequential_runs_modules_in_order():
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
    )
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    y = model(x)
    assert isinstance(y, Tensor)
    assert y.data.shape == (1, 1)

    # Replay the layers by hand to confirm the ordering.
    h = np.maximum(x.data @ model.modules[0].weight.data + model.modules[0].bias.data, 0.0)
    expected = h @ model.modules[2].weight.data + model.modules[2].bias.data
    assert np.allclose(y.data, expected)


def test_sequential_collects_parameters_recursively():
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Sequential(Linear(4, 4), Tanh()),  # nested container
        Linear(4, 1),
    )
    params = model.parameters()
    # Three Linear layers, each with weight + bias.
    assert len(params) == 6
    assert all(isinstance(p, Parameter) for p in params)


def test_sequential_end_to_end_backward():
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
    x = Tensor([[1.0, 2.0], [-0.5, 0.5]])
    model(x).sum().backward()
    for p in model.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.data.shape


def test_zero_grad_clears_parameter_gradients():
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
    model(Tensor([[1.0, 2.0]])).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_forward_not_implemented():
    class Empty(Module):
        pass

    try:
        Empty()(Tensor(1.0))
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
