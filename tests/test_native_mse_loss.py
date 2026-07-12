"""Tests for NativeMSELoss (Advanced C++ v3.6, Phase C).

NativeMSELoss(reduction="mean") is a parameter-free NativeModule whose
forward composes existing native operations — subtract →
multiply(diff, diff) → mean/sum — into a scalar loss, so the existing
autograd supplies every gradient: multiply's duplicate-parent
accumulation gives the factor 2, subtract's backward gives the target
its negative sign, and mean's existing native backward gives the 1/N
scaling (no division exists or is needed). Exactly "mean" and "sum"
are accepted (exact string match, no normalization); shapes must match
exactly (no broadcasting, checked before any graph node); dtype/device
must match; inputs are never mutated. state_dict() is empty; train/eval
never affects numerics; the v3.3–v3.5 mutation boundary (forward → loss
→ backward → updates after graph completion) is unchanged. See
src/tensorforge/experimental/native_mse_loss.py.

NumPy appears below only for references (exact formulas and central
finite differences); every analytical forward/backward is native, and a
tripwire test proves it. Note one float64 subtlety: for mean reduction
with a non-power-of-two element count, the native path computes
d * (1/N) while the reference computes d / N — equal to within 1 ulp,
so those comparisons use a tiny tolerance instead of bit equality.

Selector: python -m pytest -q -k "native_mse_loss"
"""

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeLinear,
    NativeModule,
    NativeMSELoss,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeTensor,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


# 2x2 case: N=4 is a power of two, so even mean gradients are exact.
P = np.array([[1.0, -2.0], [0.5, 3.0]])
T = np.array([[0.5, 1.0], [-1.5, 3.0]])
DIFF = P - T  # [[0.5, -3.0], [2.0, 0.0]] — positive, negative, and zero


def _pair(requires_grad=True):
    return (
        NativeTensor.from_array(P, requires_grad=requires_grad),
        NativeTensor.from_array(T, requires_grad=requires_grad),
    )


# ======================================================================
# Construction and module behavior
# ======================================================================


def test_native_mse_loss_is_a_parameter_free_module():
    loss_fn = NativeMSELoss()
    assert isinstance(loss_fn, NativeModule)
    assert loss_fn.reduction == "mean"  # default
    assert NativeMSELoss(reduction="sum").reduction == "sum"
    assert loss_fn.parameters() == []
    assert loss_fn.modules() == [loss_fn]
    assert loss_fn.state_dict() == {}
    assert loss_fn.training is True
    assert repr(loss_fn) == "NativeMSELoss(reduction='mean')"
    assert "backward" not in NativeMSELoss.__dict__


def test_native_mse_loss_rejects_invalid_reductions():
    for bad in ("none", "Mean", "SUM", " mean", "mean ", "average", ""):
        with pytest.raises(ValueError):
            NativeMSELoss(reduction=bad)
    for bad in (None, 0, 1, True, ["mean"], b"mean"):
        with pytest.raises(TypeError):
            NativeMSELoss(reduction=bad)


@needs_native
def test_native_mse_loss_train_eval_do_not_alter_numerics():
    loss_fn = NativeMSELoss()
    p, t = _pair(requires_grad=False)
    train_value = float(loss_fn.train()(p, t).to_numpy())
    eval_value = float(loss_fn.eval()(p, t).to_numpy())
    assert train_value == eval_value
    assert loss_fn.training is False


# ======================================================================
# Input validation
# ======================================================================


@needs_native
def test_native_mse_loss_rejects_non_native_operands():
    import tensorforge

    loss_fn = NativeMSELoss()
    good = NativeTensor.from_array(P)
    for bad in (tensorforge.Tensor(P), np.asarray(P), P.tolist(), 3.0, None):
        with pytest.raises(TypeError, match="prediction"):
            loss_fn(bad, good)
        with pytest.raises(TypeError, match="target"):
            loss_fn(good, bad)


@needs_native
def test_native_mse_loss_rejects_closed_operands():
    loss_fn = NativeMSELoss()
    open_tensor = NativeTensor.from_array(P)
    closed = NativeTensor.from_array(P)
    closed.close()
    with pytest.raises(RuntimeError, match="prediction"):
        loss_fn(closed, open_tensor)
    with pytest.raises(RuntimeError, match="target"):
        loss_fn(open_tensor, closed)


@needs_native
def test_native_mse_loss_requires_exact_shapes_before_any_graph():
    loss_fn = NativeMSELoss()
    p = NativeTensor.from_array(P, requires_grad=True)
    broadcastable = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError) as excinfo:
        loss_fn(p, broadcastable)  # subtract would broadcast; MSE must not
    message = str(excinfo.value)
    assert "exactly equal shapes" in message
    assert "(2, 2)" in message and "(2,)" in message
    # Nothing was built or touched: no gradients appeared anywhere.
    assert p.grad is None and broadcastable.grad is None


@needs_native
def test_native_mse_loss_accepts_native_parameters_and_any_rank():
    loss_fn = NativeMSELoss()
    p = NativeParameter(P)
    t = NativeParameter(T)
    loss = loss_fn(p, t)
    assert type(loss) is NativeTensor
    loss.backward()
    assert isinstance(p.grad, NativeTensor)  # native-backed gradients
    assert np.array_equal(p.grad.to_numpy(), 2.0 * DIFF / 4.0)
    # 1-D and 3-D shapes work — the loss is shape-generic.
    for shape in ((3,), (2, 2, 2)):
        values = np.arange(1.0, 1.0 + np.prod(shape)).reshape(shape)
        out = loss_fn(
            NativeTensor.from_array(values),
            NativeTensor.from_array(values * 0.5),
        )
        assert out.shape == ()
        expected = ((values - values * 0.5) ** 2).mean()
        assert np.isclose(float(out.to_numpy()), expected, atol=1e-12)


# ======================================================================
# Forward values
# ======================================================================


@needs_native
def test_native_mse_loss_forward_values_are_exact():
    p, t = _pair(requires_grad=False)
    mean_loss = NativeMSELoss()(p, t)
    sum_loss = NativeMSELoss(reduction="sum")(p, t)
    assert mean_loss.shape == () and sum_loss.shape == ()
    assert mean_loss.dtype == "float64" and mean_loss.device == "cpu"
    assert float(mean_loss.to_numpy()) == (DIFF ** 2).mean()  # 3.3125
    assert float(sum_loss.to_numpy()) == (DIFF ** 2).sum()  # 13.25
    # Inputs are unchanged and equal tensors give exactly zero.
    assert np.array_equal(p.to_numpy(), P)
    assert np.array_equal(t.to_numpy(), T)
    same = NativeTensor.from_array(P)
    assert float(NativeMSELoss()(p, same).to_numpy()) == 0.0


@needs_native
def test_native_mse_loss_builds_the_expected_composed_graph():
    p, t = _pair()
    loss = NativeMSELoss()(p, t)
    assert type(loss) is NativeTensor
    assert not isinstance(loss, NativeParameter)
    assert loss._op == "mean"  # reduction node ...
    (squared,) = loss._parents
    assert squared._op == "multiply"  # ... over the squared difference
    assert squared._parents[0] is squared._parents[1]  # duplicate parent
    assert squared._parents[0]._op == "subtract"
    assert NativeMSELoss(reduction="sum")(p, t)._op == "sum"


@needs_native
def test_native_mse_loss_forward_backward_use_no_numpy_compute(monkeypatch):
    loss_fn = NativeMSELoss()
    p, t = _pair()

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("subtract", "multiply", "sum", "mean", "divide",
                 "negative", "add"):
        monkeypatch.setattr(np, name, _tripwire)
    loss = loss_fn(p, t)
    loss.backward()
    monkeypatch.undo()
    assert np.array_equal(p.grad.to_numpy(), 2.0 * DIFF / 4.0)


# ======================================================================
# Exact backward
# ======================================================================


@needs_native
def test_native_mse_loss_mean_gradients_are_exact():
    p, t = _pair()
    NativeMSELoss()(p, t).backward()
    # dL/dp = 2*(p-t)/N, dL/dt = -2*(p-t)/N; N=4 keeps this bit-exact.
    assert np.array_equal(p.grad.to_numpy(), 2.0 * DIFF / 4.0)
    assert np.array_equal(t.grad.to_numpy(), -2.0 * DIFF / 4.0)
    assert p.grad.shape == P.shape and t.grad.shape == T.shape
    assert p.grad.dtype == "float64" and p.grad.device == "cpu"


@needs_native
def test_native_mse_loss_sum_gradients_are_exact():
    p, t = _pair()
    NativeMSELoss(reduction="sum")(p, t).backward()
    assert np.array_equal(p.grad.to_numpy(), 2.0 * DIFF)
    assert np.array_equal(t.grad.to_numpy(), -2.0 * DIFF)


@needs_native
def test_native_mse_loss_multidimensional_mean_scales_by_total_count():
    values = np.arange(1.0, 13.0).reshape(2, 3, 2)  # N = 12, not batch size
    p = NativeTensor.from_array(values, requires_grad=True)
    t = NativeTensor.from_array(np.zeros((2, 3, 2)))
    NativeMSELoss()(p, t).backward()
    # 12 = 4 * 3: the 1/12 scaling is not a power of two, so allow 1 ulp.
    assert np.allclose(p.grad.to_numpy(), 2.0 * values / 12.0, atol=1e-15)


@needs_native
def test_native_mse_loss_equal_inputs_give_zero_loss_and_gradients():
    p = NativeTensor.from_array(P, requires_grad=True)
    t = NativeTensor.from_array(P, requires_grad=True)
    loss = NativeMSELoss()(p, t)
    assert float(loss.to_numpy()) == 0.0
    loss.backward()
    assert np.array_equal(p.grad.to_numpy(), np.zeros((2, 2)))
    assert np.array_equal(t.grad.to_numpy(), np.zeros((2, 2)))


@needs_native
def test_native_mse_loss_one_sided_and_frozen_gradients():
    p = NativeTensor.from_array(P, requires_grad=True)
    t = NativeTensor.from_array(T)  # target does not require grad
    NativeMSELoss()(p, t).backward()
    assert p.grad is not None and t.grad is None
    p2 = NativeTensor.from_array(P)
    t2 = NativeTensor.from_array(T, requires_grad=True)
    NativeMSELoss()(p2, t2).backward()
    assert p2.grad is None and t2.grad is not None
    # Both frozen: the loss is a plain forward result; backward raises.
    p3, t3 = _pair(requires_grad=False)
    loss = NativeMSELoss()(p3, t3)
    assert loss.requires_grad is False
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()
    assert p3.grad is None and t3.grad is None


@needs_native
def test_native_mse_loss_explicit_upstream_scales_gradients():
    p, t = _pair()
    NativeMSELoss()(p, t).backward(gradient=NativeTensor.full((), 2.0))
    assert np.array_equal(p.grad.to_numpy(), 2.0 * (2.0 * DIFF / 4.0))
    assert np.array_equal(t.grad.to_numpy(), -2.0 * (2.0 * DIFF / 4.0))


@needs_native
def test_native_mse_loss_branching_accumulates():
    p, t = _pair()
    loss_fn = NativeMSELoss(reduction="sum")
    loss_fn(p, t).add(loss_fn(p, t)).backward()  # two branches
    assert np.array_equal(p.grad.to_numpy(), 2.0 * (2.0 * DIFF))


# ======================================================================
# Graph lifetime
# ======================================================================


@needs_native
def test_native_mse_loss_graph_lifetime_semantics():
    p, t = _pair()
    loss = NativeMSELoss()(p, t)
    loss.backward()  # one-shot by default
    grad_after_first = p.grad.to_numpy()
    with pytest.raises(RuntimeError, match="retain_graph"):
        loss.backward()
    # The failed reuse changed nothing.
    assert np.array_equal(p.grad.to_numpy(), grad_after_first)
    retained = NativeMSELoss()(p, t)
    retained.backward(retain_graph=True)
    retained.backward(retain_graph=True)
    retained.backward()  # a final default pass frees the graph
    with pytest.raises(RuntimeError, match="retain_graph"):
        retained.backward()
    assert np.array_equal(p.grad.to_numpy(), 4.0 * (2.0 * DIFF / 4.0))
    # Fresh graphs keep working; non-leaf gradients are not retained.
    fresh = NativeMSELoss()(p, t)
    fresh.backward()
    assert fresh.grad is None  # only leaves retain grad


@needs_native
def test_native_mse_loss_output_close_leaves_inputs_alive():
    p, t = _pair(requires_grad=False)
    loss = NativeMSELoss()(p, t)
    loss.close()
    assert p.closed is False and t.closed is False
    assert np.array_equal(p.to_numpy(), P)


# ======================================================================
# Finite differences
# ======================================================================
#
# Central differences, eps=1e-6, atol=1e-6 — float64 on O(1) values.
# The analytical path is NativeMSELoss + NativeTensor.backward(); the
# reference only perturbs host-side copies.


def _numeric_grad(f, values, eps=1e-6):
    values = np.array(values, dtype=np.float64)
    grad = np.zeros_like(values)
    flat, gflat = values.ravel(), grad.ravel()
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        f_plus = f(values)
        flat[i] = original - eps
        f_minus = f(values)
        flat[i] = original
        gflat[i] = (f_plus - f_minus) / (2 * eps)
    return grad


@needs_native
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_native_mse_loss_matches_finite_differences(reduction):
    def loss_at(p_values, t_values):
        return float(
            NativeMSELoss(reduction=reduction)(
                NativeTensor.from_array(p_values),
                NativeTensor.from_array(t_values),
            ).to_numpy()
        )

    p, t = _pair()
    NativeMSELoss(reduction=reduction)(p, t).backward()
    fd_p = _numeric_grad(lambda v: loss_at(v, T), P)
    fd_t = _numeric_grad(lambda v: loss_at(P, v), T)
    assert np.allclose(p.grad.to_numpy(), fd_p, atol=1e-6)
    assert np.allclose(t.grad.to_numpy(), fd_t, atol=1e-6)


# ======================================================================
# Model integration (Linear -> ReLU -> Linear -> MSE)
# ======================================================================
#
# Deterministic weights from the v3.5 case: hidden pre-activations are
# [[0.1, -1.45, 1.425], [2.6, 1.05, -1.325]], away from ReLU's zero.
# The supported sequence is respected throughout: forward -> loss ->
# backward -> zero_grad only after the graph completes.

X = np.array([[0.5, -1.0], [1.5, 2.0]])
W1 = np.array([[1.0, -0.5, 0.25], [0.5, 1.0, -1.0]])
B1 = np.array([0.1, -0.2, 0.3])
W2 = np.array([[1.0, -1.0], [0.5, 0.25], [-0.5, 2.0]])
B2 = np.array([0.05, -0.1])
TARGET = np.array([[1.0, -0.5], [0.5, 2.0]])


def _model(requires_grad=True):
    model = NativeSequential(
        NativeLinear(2, 3, seed=0, requires_grad=requires_grad),
        NativeReLU(),
        NativeLinear(3, 2, seed=1, requires_grad=requires_grad),
    )
    model.load_state_dict({
        "0.weight": NativeTensor.from_array(W1),
        "0.bias": NativeTensor.from_array(B1),
        "2.weight": NativeTensor.from_array(W2),
        "2.bias": NativeTensor.from_array(B2),
    })
    return model


def _integration_reference():
    hidden = X @ W1 + B1
    mask = (hidden > 0).astype(np.float64)
    relu_out = np.maximum(hidden, 0.0)
    out = relu_out @ W2 + B2
    d_out = 2.0 * (out - TARGET) / out.size  # mean over N=4: exact
    d_hidden = (d_out @ W2.T) * mask
    return {
        "loss": ((out - TARGET) ** 2).mean(),
        "x": d_hidden @ W1.T,
        "w1": X.T @ d_hidden,
        "b1": d_hidden.sum(axis=0),
        "w2": relu_out.T @ d_out,
        "b2": d_out.sum(axis=0),
        "target": -d_out,
    }


@needs_native
def test_native_mse_loss_integrates_with_sequential_model():
    model = _model()
    loss_fn = NativeMSELoss()
    x = NativeTensor.from_array(X, requires_grad=True)
    target = NativeTensor.from_array(TARGET, requires_grad=True)
    loss = loss_fn(model(x), target)
    assert loss.shape == ()
    ref = _integration_reference()
    assert np.isclose(float(loss.to_numpy()), ref["loss"], atol=1e-12)
    loss.backward()
    assert np.allclose(x.grad.to_numpy(), ref["x"], atol=1e-12)
    assert np.allclose(model[0].weight.grad.to_numpy(), ref["w1"], atol=1e-12)
    assert np.allclose(model[0].bias.grad.to_numpy(), ref["b1"], atol=1e-12)
    assert np.allclose(model[2].weight.grad.to_numpy(), ref["w2"], atol=1e-12)
    assert np.allclose(model[2].bias.grad.to_numpy(), ref["b2"], atol=1e-12)
    assert np.allclose(target.grad.to_numpy(), ref["target"], atol=1e-12)
    # Recursive zero_grad clears the model; the target is independent.
    model.zero_grad()
    assert all(param.grad is None for param in model.parameters())
    assert target.grad is not None
    target.zero_grad()
    assert target.grad is None


@needs_native
def test_native_mse_loss_repeated_training_style_cycles():
    model = _model()
    loss_fn = NativeMSELoss()
    x = NativeTensor.from_array(X)
    first = None
    for _ in range(3):
        loss = loss_fn(model(x), NativeTensor.from_array(TARGET))
        loss.backward()
        if first is None:
            first = model[0].weight.grad.to_numpy()
        model.zero_grad()
    # After the final zero_grad a fresh cycle still works and matches.
    loss_fn(model(x), NativeTensor.from_array(TARGET)).backward()
    assert np.array_equal(model[0].weight.grad.to_numpy(), first)


@needs_native
def test_native_mse_loss_frozen_model_and_train_eval_independence():
    frozen = _model(requires_grad=False)
    loss_fn = NativeMSELoss()
    x = NativeTensor.from_array(X, requires_grad=True)
    loss_fn(frozen(x), NativeTensor.from_array(TARGET)).backward()
    assert all(param.grad is None for param in frozen.parameters())
    assert x.grad is not None  # the trainable input still learns
    # train/eval state changes nothing numerically.
    plain = _model()
    x2 = NativeTensor.from_array(X)
    t2 = NativeTensor.from_array(TARGET)
    train_loss = float(loss_fn.train()(plain.train()(x2), t2).to_numpy())
    eval_loss = float(loss_fn.eval()(plain.eval()(x2), t2).to_numpy())
    assert train_loss == eval_loss


# ======================================================================
# State and traversal
# ======================================================================


@needs_native
def test_native_mse_loss_state_behavior():
    loss_fn = NativeMSELoss(reduction="sum")
    assert loss_fn.state_dict() == {}
    result = loss_fn.load_state_dict({})
    assert result.missing_keys == () and result.unexpected_keys == ()
    extra = {"w": NativeTensor.from_array([1.0])}
    with pytest.raises(ValueError, match="'w'"):
        loss_fn.load_state_dict(extra)
    result = loss_fn.load_state_dict(extra, strict=False)
    assert result.unexpected_keys == ("w",)
    assert loss_fn.reduction == "sum"  # configuration, not state


@needs_native
def test_native_mse_loss_as_child_module_adds_no_state():
    parent = NativeModule()
    parent.layer = NativeLinear(2, 2, seed=0)
    shared_loss = NativeMSELoss()
    parent.criterion = shared_loss
    parent.criterion_alias = shared_loss  # shared: structurally deduplicated
    assert list(parent.state_dict().keys()) == ["layer.weight", "layer.bias"]
    assert parent.modules() == [parent, parent.layer, shared_loss]
    # ... while explicit repeated calls still execute independently.
    p, t = _pair(requires_grad=False)
    a = float(parent.criterion(p, t).to_numpy())
    b = float(parent.criterion_alias(p, t).to_numpy())
    assert a == b == (DIFF ** 2).mean()


def test_native_mse_loss_isolated_from_stable_framework():
    import tensorforge
    import tensorforge.experimental as experimental

    assert experimental.NativeMSELoss is NativeMSELoss
    assert not hasattr(tensorforge, "NativeMSELoss")
    assert not hasattr(tensorforge.nn, "NativeMSELoss")
    # The stable framework's mse_loss is untouched and NumPy-backed.
    from tensorforge.nn import mse_loss

    pred = tensorforge.Tensor(P, requires_grad=True)
    target = tensorforge.Tensor(T)
    loss = mse_loss(pred, target)
    assert isinstance(loss, tensorforge.Tensor)
    loss.backward()
    assert isinstance(pred.grad, np.ndarray)
    assert np.allclose(pred.grad, 2.0 * DIFF / DIFF.size)
