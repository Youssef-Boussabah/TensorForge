"""Tests for NativeReLU (Advanced C++ v3.5, Phase C).

NativeReLU is a parameter-free NativeModule whose forward validates an
open NativeTensor input and delegates to the existing
NativeTensor.relu() — shape-generic, no in-place mode, no custom
backward (the existing fused native relu_backward is the gradient,
including its block-at-exactly-zero rule, tested here unchanged), and
training mode never affects numerics. state_dict() is empty. See
src/tensorforge/experimental/native_relu.py.

Selector: python -m pytest -q -k "native_relu or native_sequential"
"""

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeModule,
    NativeParameter,
    NativeReLU,
    NativeTensor,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


VALUES = [[1.5, -2.0], [-0.5, 3.0]]


# ======================================================================
# Construction and module behavior
# ======================================================================


def test_native_relu_is_a_parameter_free_module():
    relu = NativeReLU()
    assert isinstance(relu, NativeModule)
    assert relu.parameters() == []
    assert list(relu.named_parameters()) == []
    assert relu.state_dict() == {}
    assert relu.training is True
    assert repr(relu) == "NativeReLU()"


def test_native_relu_train_eval_flag_updates():
    relu = NativeReLU()
    assert relu.eval() is relu
    assert relu.training is False
    assert relu.train() is relu
    assert relu.training is True


@needs_native
def test_native_relu_training_mode_does_not_change_numerics():
    relu = NativeReLU()
    x = NativeTensor.from_array(VALUES)
    train_out = relu.train()(x).to_numpy()
    eval_out = relu.eval()(x).to_numpy()
    assert np.array_equal(train_out, eval_out)


# ======================================================================
# Validation and forward
# ======================================================================


@needs_native
def test_native_relu_forward_matches_native_relu_operation():
    relu = NativeReLU()
    x = NativeTensor.from_array(VALUES)
    out = relu(x)  # __call__ delegates to forward
    assert type(out) is NativeTensor
    assert not isinstance(out, NativeParameter)
    assert out.shape == x.shape
    assert out.dtype == "float64" and out.device == "cpu"
    assert np.array_equal(out.to_numpy(), np.maximum(np.asarray(VALUES), 0.0))
    assert np.array_equal(out.to_numpy(), x.relu().to_numpy())


@needs_native
def test_native_relu_accepts_native_parameter_and_returns_plain_tensor():
    relu = NativeReLU()
    p = NativeParameter(VALUES)
    out = relu(p)
    assert type(out) is NativeTensor
    assert not isinstance(out, NativeParameter)
    assert p.is_leaf is True  # the parameter stays a graph-free leaf


@needs_native
def test_native_relu_rejects_non_native_inputs():
    import tensorforge

    relu = NativeReLU()
    for bad in (tensorforge.Tensor(VALUES), np.asarray(VALUES),
                VALUES, 3.0, None):
        with pytest.raises(TypeError):
            relu(bad)


@needs_native
def test_native_relu_rejects_closed_input():
    relu = NativeReLU()
    x = NativeTensor.from_array(VALUES)
    x.close()
    with pytest.raises(RuntimeError, match="closed"):
        relu(x)


@needs_native
def test_native_relu_supports_arbitrary_ranks_and_views():
    relu = NativeReLU()
    for values in ([-1.0, 2.0, -3.0], np.arange(-4.0, 4.0).reshape(2, 2, 2)):
        out = relu(NativeTensor.from_array(values))
        assert out.shape == np.asarray(values).shape
        assert np.array_equal(out.to_numpy(), np.maximum(values, 0.0))
    # A non-contiguous view input works through the existing odometer.
    base = NativeTensor.from_array(VALUES)
    view = base.T
    assert view.contiguous is False
    assert np.array_equal(
        relu(view).to_numpy(), np.maximum(np.asarray(VALUES).T, 0.0)
    )


@needs_native
def test_native_relu_uses_no_numpy_compute(monkeypatch):
    relu = NativeReLU()
    x = NativeTensor.from_array(VALUES, requires_grad=True)

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("maximum", "where", "matmul", "add", "multiply"):
        monkeypatch.setattr(np, name, _tripwire)
    relu(x).sum().backward()
    monkeypatch.undo()
    assert x.grad is not None


@needs_native
def test_native_relu_has_no_module_backward_of_its_own():
    assert "backward" not in NativeReLU.__dict__
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    out = NativeReLU()(x)
    assert out._op == "relu"  # the existing relu graph node, nothing else
    assert out.is_leaf is False


# ======================================================================
# Backward
# ======================================================================


@needs_native
def test_native_relu_backward_masks_correctly_including_zero():
    # Existing native contract (unchanged here): upstream passes where
    # the input was > 0 and is blocked elsewhere — x == 0 blocks.
    x = NativeTensor.from_array([-1.0, 0.0, 2.0], requires_grad=True)
    NativeReLU()(x).sum().backward()
    assert np.array_equal(x.grad.to_numpy(), np.array([0.0, 0.0, 1.0]))


@needs_native
def test_native_relu_repeated_cycles_and_graph_lifetime():
    relu = NativeReLU()
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    mask = (np.asarray(VALUES) > 0).astype(np.float64)
    loss = relu(x).sum()
    loss.backward()  # one-shot by default
    with pytest.raises(RuntimeError, match="retain_graph"):
        loss.backward()
    relu(x).sum().backward()  # a fresh graph accumulates
    assert np.array_equal(x.grad.to_numpy(), 2.0 * mask)
    retained = relu(x).sum()
    retained.backward(retain_graph=True)
    retained.backward(retain_graph=True)
    assert np.array_equal(x.grad.to_numpy(), 4.0 * mask)


@needs_native
def test_native_relu_matches_finite_differences_away_from_zero():
    # Central differences, eps=1e-6, atol=1e-6 — every value is at
    # least 0.5 from ReLU's zero boundary.
    values = np.array([[1.5, -2.0], [-0.5, 3.0]])

    def loss_at(v):
        x = NativeTensor.from_array(v)
        return float(NativeReLU()(x).sum().to_numpy())

    eps = 1e-6
    fd = np.zeros_like(values)
    flat, gflat = values.copy(), fd.ravel()
    for i in range(flat.size):
        plus, minus = flat.copy().ravel(), flat.copy().ravel()
        plus[i] += eps
        minus[i] -= eps
        gflat[i] = (
            loss_at(plus.reshape(values.shape))
            - loss_at(minus.reshape(values.shape))
        ) / (2 * eps)
    x = NativeTensor.from_array(values, requires_grad=True)
    NativeReLU()(x).sum().backward()
    assert np.allclose(x.grad.to_numpy(), fd, atol=1e-6)


# ======================================================================
# State dictionary and isolation
# ======================================================================


@needs_native
def test_native_relu_state_dict_is_empty_and_loads_cleanly():
    relu = NativeReLU()
    assert relu.state_dict() == {}
    result = relu.load_state_dict({})
    assert result.missing_keys == () and result.unexpected_keys == ()
    extra = {"x": NativeTensor.from_array([1.0])}
    with pytest.raises(ValueError, match="'x'"):
        relu.load_state_dict(extra)  # strict: unexpected key
    result = relu.load_state_dict(extra, strict=False)
    assert result.unexpected_keys == ("x",)


def test_native_relu_isolated_from_stable_framework():
    import tensorforge

    assert not hasattr(tensorforge, "NativeReLU")
    assert not hasattr(tensorforge.nn, "NativeReLU")
    assert not issubclass(NativeReLU, tensorforge.nn.Module)
    # Stable ReLU is untouched and still NumPy-backed.
    stable = tensorforge.nn.ReLU()
    t = tensorforge.Tensor(VALUES, requires_grad=True)
    out = stable(t)
    assert isinstance(out, tensorforge.Tensor)
    out.sum().backward()
    assert isinstance(t.grad, np.ndarray)
