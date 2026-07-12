"""Tests for NativeLinear — the first concrete native layer (Advanced
C++ v3.4, the fourth Phase C milestone).

NativeLinear(in_features, out_features, bias=True, *, seed=None,
requires_grad=True) is a NativeModule holding a NativeParameter weight
of shape (in_features, out_features) and, when enabled, a
NativeParameter bias of shape (out_features,), both initialized
uniformly from [-1/sqrt(in_features), +1/sqrt(in_features)] by a local
seeded generator (the global NumPy RNG is never touched). forward(input)
takes strictly 2-D (batch_size, in_features) NativeTensor input and
computes input.matmul(weight) plus broadcast add(bias) — backward is
entirely the existing native autograd (no manual or fused path).
Gradients: input (batch, in), weight (in, out), bias (out,) reduced over
the batch by unbroadcast. state_dict()/load_state_dict() follow v3.3
(keys ["weight", "bias"] or ["weight"]). See
src/tensorforge/experimental/native_linear.py.

NumPy appears below only for references (exact formulas and central
finite differences) — every gradient under test was computed natively.
Native-backend tests skip when the compiled library is not built;
pure argument-validation tests run everywhere (validation precedes any
native allocation).

Selector: python -m pytest -q -k "native_linear"
"""

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeLinear,
    NativeModule,
    NativeParameter,
    NativeTensor,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


# Small deterministic case: exact float64 values (halves and integers).
X = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])  # (batch=3, in=2)
W = np.array([[1.0, -2.0, 0.5], [2.0, 1.0, -1.0]])  # (in=2, out=3)
B = np.array([0.25, -0.5, 1.0])  # (out=3,)
UP = np.array([[1.0, -1.0, 2.0], [0.5, 2.0, -1.0], [-2.0, 1.0, 0.5]])


def _layer(bias=True, requires_grad=True):
    """A NativeLinear(2, 3) with exactly known weight/bias values."""
    layer = NativeLinear(2, 3, bias=bias, seed=0, requires_grad=requires_grad)
    state = {"weight": NativeTensor.from_array(W)}
    if bias:
        state["bias"] = NativeTensor.from_array(B)
    layer.load_state_dict(state)
    return layer


def _numeric_grad(f, x, eps=1e-6):
    """Central finite differences of the scalar-valued ``f`` at ``x``."""
    x = np.array(x, dtype=np.float64)
    grad = np.zeros_like(x)
    flat, gflat = x.ravel(), grad.ravel()
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        f_plus = f(x)
        flat[i] = original - eps
        f_minus = f(x)
        flat[i] = original
        gflat[i] = (f_plus - f_minus) / (2 * eps)
    return grad


# ======================================================================
# Constructor and validation
# ======================================================================


@needs_native
def test_native_linear_basic_construction():
    layer = NativeLinear(4, 3)
    assert layer.in_features == 4
    assert layer.out_features == 3
    assert isinstance(layer, NativeModule)
    assert isinstance(layer.weight, NativeParameter)
    assert layer.weight.shape == (4, 3)
    assert isinstance(layer.bias, NativeParameter)
    assert layer.bias.shape == (3,)


@needs_native
def test_native_linear_bias_false_leaves_none():
    layer = NativeLinear(4, 3, bias=False)
    assert layer.bias is None
    assert [name for name, _ in layer.named_parameters()] == ["weight"]
    assert list(layer.state_dict().keys()) == ["weight"]


@needs_native
def test_native_linear_parameter_order_is_weight_then_bias():
    layer = NativeLinear(4, 3)
    assert [name for name, _ in layer.named_parameters()] == ["weight", "bias"]
    assert layer.parameters() == [layer.weight, layer.bias]


@needs_native
def test_native_linear_requires_grad_default_and_frozen():
    layer = NativeLinear(2, 2)
    assert layer.weight.requires_grad is True
    assert layer.bias.requires_grad is True
    frozen = NativeLinear(2, 2, requires_grad=False)
    assert frozen.weight.requires_grad is False
    assert frozen.bias.requires_grad is False
    # Frozen parameters stay parameters, registered and snapshottable.
    assert frozen.parameters() == [frozen.weight, frozen.bias]
    assert list(frozen.state_dict().keys()) == ["weight", "bias"]


def test_native_linear_invalid_feature_counts_raise():
    for bad in (2.0, "2", None, np.int64(2), True):
        with pytest.raises(TypeError):
            NativeLinear(bad, 3)
        with pytest.raises(TypeError):
            NativeLinear(3, bad)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            NativeLinear(bad, 3)
        with pytest.raises(ValueError):
            NativeLinear(3, bad)


def test_native_linear_non_bool_bias_raises():
    for bad in (1, 0, "yes", None):
        with pytest.raises(TypeError):
            NativeLinear(2, 3, bias=bad)


def test_native_linear_non_bool_requires_grad_raises():
    for bad in (1, 0, "True", None):
        with pytest.raises(TypeError):
            NativeLinear(2, 3, requires_grad=bad)


def test_native_linear_invalid_seed_raises():
    for bad in (1.5, "7", True, [7]):
        with pytest.raises(TypeError):
            NativeLinear(2, 3, seed=bad)


# ======================================================================
# Initialization
# ======================================================================


@needs_native
def test_native_linear_same_seed_reproduces_values():
    a = NativeLinear(5, 4, seed=42)
    b = NativeLinear(5, 4, seed=42)
    assert np.array_equal(a.weight.to_numpy(), b.weight.to_numpy())
    assert np.array_equal(a.bias.to_numpy(), b.bias.to_numpy())


@needs_native
def test_native_linear_different_seeds_differ():
    a = NativeLinear(5, 4, seed=1)
    b = NativeLinear(5, 4, seed=2)
    assert not np.array_equal(a.weight.to_numpy(), b.weight.to_numpy())


@needs_native
def test_native_linear_initialization_finite_and_within_bounds():
    layer = NativeLinear(16, 8, seed=3)
    bound = 1.0 / np.sqrt(16)
    for values in (layer.weight.to_numpy(), layer.bias.to_numpy()):
        assert np.all(np.isfinite(values))
        assert np.all(np.abs(values) <= bound)


@needs_native
def test_native_linear_initialization_creates_graph_free_leaves():
    layer = NativeLinear(3, 2, seed=0)
    for parameter in (layer.weight, layer.bias):
        assert parameter.is_leaf is True
        assert parameter._parents == ()
        assert parameter._backward is None
        assert parameter._graph_freed is False
        assert parameter.grad is None
        assert parameter.owns_core is True
        assert parameter.contiguous is True
        assert parameter.dtype == "float64"
        assert parameter.device == "cpu"


@needs_native
def test_native_linear_construction_leaves_global_rng_untouched():
    np.random.seed(0)
    expected = np.random.rand(4)
    np.random.seed(0)
    NativeLinear(3, 2)  # seed=None uses a fresh local generator
    NativeLinear(3, 2, seed=9)
    assert np.array_equal(np.random.rand(4), expected)


# ======================================================================
# Forward behavior
# ======================================================================


@needs_native
def test_native_linear_forward_matches_reference_with_bias():
    layer = _layer()
    x = NativeTensor.from_array(X)
    out = layer(x)  # __call__ delegates to forward
    assert type(out) is NativeTensor
    assert not isinstance(out, NativeParameter)
    assert out.shape == (3, 3)
    assert out.dtype == "float64" and out.device == "cpu"
    assert np.array_equal(out.to_numpy(), X @ W + B)


@needs_native
def test_native_linear_forward_matches_reference_without_bias():
    layer = _layer(bias=False)
    out = layer.forward(NativeTensor.from_array(X))
    assert np.array_equal(out.to_numpy(), X @ W)
    # No-bias forward is a bare matmul node (the weight requires grad,
    # so a graph node is built) — the composition of existing ops; no
    # fused or manual NativeLinear backward exists anywhere.
    assert out._op == "matmul"
    assert "backward" not in NativeLinear.__dict__


@needs_native
def test_native_linear_forward_composes_existing_autograd_nodes():
    layer = _layer()
    x = NativeTensor.from_array(X, requires_grad=True)
    out = layer(x)
    assert out._op == "add"  # broadcast add over the matmul node
    assert out.is_leaf is False
    no_bias = _layer(bias=False)
    assert no_bias(x)._op == "matmul"


@needs_native
def test_native_linear_bias_broadcasts_over_batch():
    layer = _layer()
    single = NativeTensor.from_array(X[:1])
    batch = NativeTensor.from_array(X)
    assert np.array_equal(layer(single).to_numpy(), X[:1] @ W + B)
    assert np.array_equal(layer(batch).to_numpy()[0], (X @ W + B)[0])


@needs_native
def test_native_linear_rejects_non_native_inputs():
    import tensorforge

    layer = _layer()
    for bad in (
        tensorforge.Tensor(X),
        X,
        X.tolist(),
        3.0,
        None,
    ):
        with pytest.raises(TypeError):
            layer(bad)


@needs_native
def test_native_linear_rejects_wrong_rank_and_features():
    layer = _layer()
    with pytest.raises(ValueError, match="2-D"):
        layer(NativeTensor.from_array([1.0, 2.0]))  # 1-D
    with pytest.raises(ValueError, match="2-D"):
        layer(NativeTensor.from_array(np.ones((2, 2, 2))))  # 3-D
    with pytest.raises(ValueError, match="in_features=2"):
        layer(NativeTensor.from_array(np.ones((3, 5))))  # wrong features
    # The error names the actual shape.
    with pytest.raises(ValueError, match=r"\(3, 5\)"):
        layer(NativeTensor.from_array(np.ones((3, 5))))


@needs_native
def test_native_linear_rejects_closed_input_weight_and_bias():
    layer = _layer()
    x = NativeTensor.from_array(X)
    x.close()
    with pytest.raises(RuntimeError, match="input"):
        layer(x)
    fresh = NativeTensor.from_array(X)
    layer.bias.close()
    with pytest.raises(RuntimeError, match="bias"):
        layer(fresh)
    layer.weight.close()
    with pytest.raises(RuntimeError, match="weight"):
        layer(fresh)


@needs_native
def test_native_linear_forward_uses_no_numpy_compute(monkeypatch):
    layer = _layer()
    x = NativeTensor.from_array(X, requires_grad=True)

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("matmul", "dot", "add", "multiply"):
        monkeypatch.setattr(np, name, _tripwire)
    loss = layer(x).sum()
    loss.backward()
    monkeypatch.undo()
    assert np.array_equal(x.grad.to_numpy(), np.ones((3, 3)) @ W.T)


# ======================================================================
# Backward correctness (exact analytical values)
# ======================================================================


@needs_native
def test_native_linear_backward_gradients_are_exact():
    layer = _layer()
    x = NativeTensor.from_array(X, requires_grad=True)
    layer(x).sum().backward()
    # loss = sum(x @ W + B): dx = ones @ W.T, dW = X.T @ ones, dB = batch.
    assert x.grad.shape == (3, 2)
    assert np.array_equal(x.grad.to_numpy(), np.ones((3, 3)) @ W.T)
    assert layer.weight.grad.shape == (2, 3)
    assert np.array_equal(layer.weight.grad.to_numpy(), X.T @ np.ones((3, 3)))
    assert layer.bias.grad.shape == (3,)
    assert np.array_equal(layer.bias.grad.to_numpy(), np.full(3, 3.0))


@needs_native
def test_native_linear_no_bias_backward_works():
    layer = _layer(bias=False)
    x = NativeTensor.from_array(X, requires_grad=True)
    layer(x).sum().backward()
    assert np.array_equal(layer.weight.grad.to_numpy(), X.T @ np.ones((3, 3)))
    assert np.array_equal(x.grad.to_numpy(), np.ones((3, 3)) @ W.T)


@needs_native
def test_native_linear_frozen_parameters_receive_no_gradients():
    layer = _layer(requires_grad=False)
    x = NativeTensor.from_array(X, requires_grad=True)
    layer(x).sum().backward()
    assert layer.weight.grad is None
    assert layer.bias.grad is None
    # The requiring input still receives its gradient.
    assert np.array_equal(x.grad.to_numpy(), np.ones((3, 3)) @ W.T)


@needs_native
def test_native_linear_frozen_input_with_trainable_parameters():
    layer = _layer()
    x = NativeTensor.from_array(X)  # requires_grad=False
    layer(x).sum().backward()
    assert x.grad is None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


@needs_native
def test_native_linear_branching_use_accumulates():
    layer = _layer()
    x = NativeTensor.from_array(X, requires_grad=True)
    y = layer(x).add(layer(x))  # two branches through the same layer
    y.sum().backward()
    assert np.array_equal(
        layer.weight.grad.to_numpy(), 2.0 * (X.T @ np.ones((3, 3)))
    )
    assert np.array_equal(layer.bias.grad.to_numpy(), np.full(3, 6.0))
    assert np.array_equal(x.grad.to_numpy(), 2.0 * (np.ones((3, 3)) @ W.T))


@needs_native
def test_native_linear_repeated_fresh_cycles_accumulate_until_zero_grad():
    layer = _layer()
    x = NativeTensor.from_array(X, requires_grad=True)
    layer(x).sum().backward()
    layer(x).sum().backward()  # a fresh graph each time
    assert np.array_equal(layer.bias.grad.to_numpy(), np.full(3, 6.0))
    layer.zero_grad()  # module zero_grad clears weight and bias
    assert layer.weight.grad is None
    assert layer.bias.grad is None
    layer(x).sum().backward()
    assert np.array_equal(layer.bias.grad.to_numpy(), np.full(3, 3.0))


@needs_native
def test_native_linear_graph_lifetime_semantics_unchanged():
    layer = _layer()
    x = NativeTensor.from_array(X, requires_grad=True)
    loss = layer(x).sum()
    loss.backward()  # one-shot by default
    with pytest.raises(RuntimeError, match="retain_graph"):
        loss.backward()
    retained = layer(x).sum()
    retained.backward(retain_graph=True)
    retained.backward(retain_graph=True)  # allowed; grads accumulate
    assert np.array_equal(layer.bias.grad.to_numpy(), np.full(3, 9.0))


# ======================================================================
# Finite differences
# ======================================================================
#
# Central differences with eps=1e-6 and atol=1e-6 — appropriate for
# float64 at these O(1) values (truncation ~eps^2, roundoff ~1e-10).
# Perturbed values enter a fresh layer through load_state_dict BEFORE
# forward — the supported sequence; nothing mutates between forward and
# backward. Every analytical gradient is computed natively.


def _native_loss(w_values, b_values, x_values):
    layer = NativeLinear(2, 3, bias=b_values is not None, seed=0)
    state = {"weight": NativeTensor.from_array(w_values)}
    if b_values is not None:
        state["bias"] = NativeTensor.from_array(b_values)
    layer.load_state_dict(state)
    out = layer(NativeTensor.from_array(x_values))
    return float(out.multiply(NativeTensor.from_array(UP)).sum().to_numpy())


@needs_native
def test_native_linear_gradients_match_finite_differences():
    layer = _layer()
    x = NativeTensor.from_array(X, requires_grad=True)
    layer(x).multiply(NativeTensor.from_array(UP)).sum().backward()
    fd_x = _numeric_grad(lambda v: _native_loss(W, B, v), X)
    fd_w = _numeric_grad(lambda v: _native_loss(v, B, X), W)
    fd_b = _numeric_grad(lambda v: _native_loss(W, v, X), B)
    assert np.allclose(x.grad.to_numpy(), fd_x, atol=1e-6)
    assert np.allclose(layer.weight.grad.to_numpy(), fd_w, atol=1e-6)
    assert np.allclose(layer.bias.grad.to_numpy(), fd_b, atol=1e-6)


@needs_native
def test_native_linear_no_bias_gradients_match_finite_differences():
    layer = _layer(bias=False)
    x = NativeTensor.from_array(X, requires_grad=True)
    layer(x).multiply(NativeTensor.from_array(UP)).sum().backward()
    fd_x = _numeric_grad(lambda v: _native_loss(W, None, v), X)
    fd_w = _numeric_grad(lambda v: _native_loss(v, None, X), W)
    assert np.allclose(x.grad.to_numpy(), fd_x, atol=1e-6)
    assert np.allclose(layer.weight.grad.to_numpy(), fd_w, atol=1e-6)


# ======================================================================
# Module integration
# ======================================================================


@needs_native
def test_native_linear_module_traversal():
    layer = _layer()
    assert layer.modules() == [layer]
    assert list(layer.named_modules()) == [("", layer)]
    assert [name for name, _ in layer.named_parameters()] == ["weight", "bias"]


@needs_native
def test_native_linear_nested_registration_uses_dotted_names():
    parent = NativeModule()
    layer = _layer()
    parent.layer = layer
    assert [name for name, _ in parent.named_parameters()] == [
        "layer.weight", "layer.bias",
    ]
    assert list(parent.state_dict().keys()) == ["layer.weight", "layer.bias"]
    assert parent.modules() == [parent, layer]


@needs_native
def test_native_linear_shared_child_deduplicates():
    parent = NativeModule()
    shared = _layer()
    parent.first = shared
    parent.second = shared
    assert parent.modules() == [parent, shared]
    assert [name for name, _ in parent.named_parameters()] == [
        "first.weight", "first.bias",
    ]
    # One backward through both aliases accumulates on the one object.
    x = NativeTensor.from_array(X)
    parent.first(x).add(parent.second(x)).sum().backward()
    assert np.array_equal(shared.bias.grad.to_numpy(), np.full(3, 6.0))


@needs_native
def test_native_linear_forward_ignores_training_mode():
    layer = _layer()
    x = NativeTensor.from_array(X)
    training_out = layer.train()(x).to_numpy()
    eval_out = layer.eval()(x).to_numpy()
    assert np.array_equal(training_out, eval_out)
    assert layer.training is False  # eval propagated normally


@needs_native
def test_native_linear_repr_is_concise():
    assert repr(NativeLinear(4, 3)) == (
        "NativeLinear(in_features=4, out_features=3, bias=True)"
    )
    assert repr(NativeLinear(4, 3, bias=False)) == (
        "NativeLinear(in_features=4, out_features=3, bias=False)"
    )


# ======================================================================
# State dictionary
# ======================================================================


@needs_native
def test_native_linear_state_dict_snapshots_are_independent():
    layer = _layer()
    snapshot = layer.state_dict()
    layer.load_state_dict({
        "weight": NativeTensor.from_array(np.zeros((2, 3))),
        "bias": NativeTensor.from_array(np.zeros(3)),
    })
    assert np.array_equal(snapshot["weight"].to_numpy(), W)
    assert np.array_equal(snapshot["bias"].to_numpy(), B)


@needs_native
def test_native_linear_compatible_load_changes_output_not_identity():
    source = NativeLinear(2, 3, seed=11)
    target = NativeLinear(2, 3, seed=22)
    weight, bias = target.weight, target.bias
    target.weight.sum().backward()
    grad = target.weight.grad
    x = NativeTensor.from_array(X)
    before = target(x).to_numpy()
    target.load_state_dict(source.state_dict())
    after = target(x).to_numpy()
    assert not np.array_equal(before, after)  # values changed
    assert np.array_equal(after, source(x).to_numpy())  # to the source's
    assert target.weight is weight  # identity survived
    assert target.bias is bias
    assert target.weight.grad is grad  # gradients survived unchanged
    assert target.weight.requires_grad is True


@needs_native
def test_native_linear_frozen_state_survives_load():
    frozen = NativeLinear(2, 3, seed=5, requires_grad=False)
    frozen.load_state_dict(NativeLinear(2, 3, seed=6).state_dict())
    assert frozen.weight.requires_grad is False
    assert frozen.bias.requires_grad is False


@needs_native
def test_native_linear_bias_mismatch_follows_strict_rules():
    biased = _layer()
    bias_free = _layer(bias=False)
    # Loading a biased state into a bias-free layer: "bias" unexpected.
    with pytest.raises(ValueError, match="bias"):
        bias_free.load_state_dict(biased.state_dict())
    # Loading a bias-free state into a biased layer: "bias" missing.
    with pytest.raises(ValueError, match="bias"):
        biased.load_state_dict(bias_free.state_dict())
    # Both failed before mutation.
    assert np.array_equal(biased.weight.to_numpy(), W)
    assert np.array_equal(bias_free.weight.to_numpy(), W)
    # Non-strict follows v3.3: matching weight loads, key lists report.
    result = biased.load_state_dict(bias_free.state_dict(), strict=False)
    assert result.missing_keys == ("bias",)
    assert result.unexpected_keys == ()
    result = bias_free.load_state_dict(_layer().state_dict(), strict=False)
    assert result.unexpected_keys == ("bias",)


@needs_native
def test_native_linear_shape_incompatible_state_fails_atomically():
    layer = _layer()
    other = NativeLinear(4, 3, seed=1)  # different in_features
    with pytest.raises(ValueError, match="weight"):
        layer.load_state_dict(
            {"weight": other.state_dict()["weight"],
             "bias": NativeTensor.from_array(np.zeros(3))}
        )
    assert np.array_equal(layer.weight.to_numpy(), W)
    assert np.array_equal(layer.bias.to_numpy(), B)


# ======================================================================
# Isolation
# ======================================================================


def test_native_linear_exported_only_from_experimental():
    import tensorforge
    import tensorforge.experimental as experimental

    assert experimental.NativeLinear is NativeLinear
    assert not hasattr(tensorforge, "NativeLinear")
    assert not hasattr(tensorforge.nn, "NativeLinear")


def test_native_linear_stable_framework_untouched():
    import tensorforge

    layer = tensorforge.nn.Linear(2, 3)
    x = tensorforge.Tensor(X, requires_grad=True)
    out = layer(x)
    assert out.data.shape == (3, 3)
    out.sum().backward()
    assert isinstance(layer.weight.grad, np.ndarray)
    assert isinstance(x.grad, np.ndarray)
    assert not isinstance(layer, NativeModule)
    assert not isinstance(layer.weight, NativeParameter)
