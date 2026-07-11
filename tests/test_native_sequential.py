"""Tests for NativeSequential (Advanced C++ v3.5, Phase C).

NativeSequential(*modules) chains NativeModule children registered
under contiguous integer-string slots "0".."len-1"; execution order is
the registered slot order (a single source of truth, enforced at the
registration funnel: replacement keeps a slot's position, appends take
the next index, gaps/non-slot child names/direct parameters/slot
removal/self-insertion are rejected). Execution is position-based (a
shared child in two slots runs twice) while traversal, state, train,
and zero_grad keep the v3.2/v3.3 identity-deduplicated first-path
contracts. An empty sequence returns its input by identity. Forward is
pure composition — each child validates its own input and contributes
its own graph nodes; no manual Sequential backward exists. See
src/tensorforge/experimental/native_sequential.py.

NumPy appears below only for references (exact formulas and central
finite differences); every gradient under test was computed natively.

Selector: python -m pytest -q -k "native_relu or native_sequential"
"""

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeLinear,
    NativeModule,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeTensor,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


# Deterministic Linear(2,3) -> ReLU -> Linear(3,2) case. The hidden
# pre-activations are [[0.1, -1.45, 1.425], [2.6, 1.05, -1.325]] —
# every entry at least 0.1 from ReLU's zero boundary, safe for
# eps=1e-6 central differences.
X = np.array([[0.5, -1.0], [1.5, 2.0]])
W1 = np.array([[1.0, -0.5, 0.25], [0.5, 1.0, -1.0]])
B1 = np.array([0.1, -0.2, 0.3])
W2 = np.array([[1.0, -1.0], [0.5, 0.25], [-0.5, 2.0]])
B2 = np.array([0.05, -0.1])
UP = np.array([[1.0, -2.0], [0.5, 1.5]])


def _state(w1=W1, b1=B1, w2=W2, b2=B2):
    return {
        "0.weight": NativeTensor.from_array(w1),
        "0.bias": NativeTensor.from_array(b1),
        "2.weight": NativeTensor.from_array(w2),
        "2.bias": NativeTensor.from_array(b2),
    }


def _model(requires_grad=True):
    model = NativeSequential(
        NativeLinear(2, 3, seed=0, requires_grad=requires_grad),
        NativeReLU(),
        NativeLinear(3, 2, seed=1, requires_grad=requires_grad),
    )
    model.load_state_dict(_state())
    return model


def _reference_grads():
    """Exact NumPy reference for loss = (model(X) * UP).sum()."""
    hidden = X @ W1 + B1
    mask = (hidden > 0).astype(np.float64)
    relu_out = np.maximum(hidden, 0.0)
    d_hidden = (UP @ W2.T) * mask
    return {
        "x": d_hidden @ W1.T,
        "w1": X.T @ d_hidden,
        "b1": d_hidden.sum(axis=0),
        "w2": relu_out.T @ UP,
        "b2": UP.sum(axis=0),
    }


class _Tag(NativeModule):
    """Identity module that records its tag when executed."""

    def __init__(self, tag, log):
        super().__init__()
        self.tag = tag
        self.log = log

    def forward(self, x):
        self.log.append(self.tag)
        return x


class _Doubler(NativeModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.add(x)


# ======================================================================
# Construction
# ======================================================================


def test_native_sequential_empty_and_ordered_construction():
    empty = NativeSequential()
    assert len(empty) == 0
    assert list(empty) == []
    assert empty.training is True
    a, b, c = NativeReLU(), NativeReLU(), NativeReLU()
    seq = NativeSequential(a, b, c)
    assert len(seq) == 3
    assert list(seq) == [a, b, c]
    assert list(seq._modules) == ["0", "1", "2"]
    assert list(seq.named_modules()) == [
        ("", seq), ("0", a), ("1", b), ("2", c),
    ]


@needs_native
def test_native_sequential_rejects_invalid_children():
    import tensorforge

    for bad in (
        NativeTensor.from_array([1.0]),
        NativeParameter([1.0]),
        tensorforge.nn.ReLU(),
        tensorforge.nn.Linear(2, 2),
        lambda x: x,
        None,
        [NativeReLU()],
        "relu",
    ):
        with pytest.raises(TypeError):
            NativeSequential(NativeReLU(), bad)


def test_native_sequential_append_validation_is_atomic():
    seq = NativeSequential(NativeReLU())
    with pytest.raises(TypeError):
        seq.append("not a module")
    with pytest.raises(TypeError):
        seq.append(None)  # add_module requires NativeModule for append
    assert len(seq) == 1  # nothing changed


# ======================================================================
# Container API
# ======================================================================


def test_native_sequential_indexing_returns_exact_children():
    a, b, c = NativeReLU(), NativeReLU(), NativeReLU()
    seq = NativeSequential(a, b, c)
    assert seq[0] is a
    assert seq[2] is c
    assert seq[-1] is c  # negative indices count from the end
    assert seq[-3] is a
    with pytest.raises(IndexError):
        seq[3]
    with pytest.raises(IndexError):
        seq[-4]
    for bad in (True, "0", 1.0, None):
        with pytest.raises(TypeError):
            seq[bad]


def test_native_sequential_append_extends_contiguously():
    seq = NativeSequential()
    first, second = NativeReLU(), NativeReLU()
    assert seq.append(first) is seq  # returns self for chaining
    seq.append(second)
    assert list(seq._modules) == ["0", "1"]
    assert list(seq) == [first, second]
    assert seq[1] is second
    assert list(seq.named_modules()) == [
        ("", seq), ("0", first), ("1", second),
    ]


@needs_native
def test_native_sequential_append_updates_traversal_and_state_immediately():
    seq = NativeSequential(NativeReLU())
    seq.append(NativeLinear(2, 2, seed=0))
    assert [name for name, _ in seq.named_parameters()] == [
        "1.weight", "1.bias",
    ]
    assert list(seq.state_dict().keys()) == ["1.weight", "1.bias"]


def test_native_sequential_setitem_replaces_preserving_position():
    a, b = NativeReLU(), NativeReLU()
    replacement = NativeReLU()
    seq = NativeSequential(a, b)
    seq[0] = replacement
    assert list(seq) == [replacement, b]
    assert list(seq._modules) == ["0", "1"]  # same name, same position
    assert a.training is True  # the old child was dropped, not touched
    # add_module on an existing slot is the same replacement path.
    other = NativeReLU()
    seq.add_module("1", other)
    assert list(seq) == [replacement, other]


def test_native_sequential_rejects_gap_and_non_slot_registration():
    seq = NativeSequential(NativeReLU(), NativeReLU())
    with pytest.raises(ValueError, match="contiguous"):
        seq.add_module("3", NativeReLU())  # gap: next free slot is "2"
    with pytest.raises(ValueError, match="slots"):
        seq.add_module("01", NativeReLU())  # non-canonical digit string
    with pytest.raises(ValueError, match="slots"):
        seq.block = NativeReLU()  # a named child would never execute
    assert len(seq) == 2


def test_native_sequential_rejects_slot_removal():
    seq = NativeSequential(NativeReLU(), NativeReLU())
    with pytest.raises(ValueError, match="contiguous"):
        setattr(seq, "0", None)
    with pytest.raises(ValueError, match="contiguous"):
        setattr(seq, "1", "label")
    with pytest.raises(ValueError, match="contiguous"):
        delattr(seq, "0")
    with pytest.raises(ValueError, match="unregistered"):
        seq.add_module("0", None)
    assert len(seq) == 2
    # Ordinary non-module attributes are still allowed.
    seq.label = "encoder"
    assert seq.label == "encoder"


@needs_native
def test_native_sequential_rejects_direct_parameters():
    seq = NativeSequential(NativeReLU())
    with pytest.raises(TypeError, match="direct parameters"):
        seq.w = NativeParameter([1.0])
    with pytest.raises(TypeError, match="direct parameters"):
        seq.register_parameter("w", NativeParameter([1.0]))
    assert seq.parameters() == []


def test_native_sequential_rejects_self_insertion():
    seq = NativeSequential(NativeReLU())
    with pytest.raises(ValueError, match="itself"):
        seq.append(seq)
    with pytest.raises(ValueError, match="itself"):
        seq[0] = seq
    assert len(seq) == 1


# ======================================================================
# Forward
# ======================================================================


@needs_native
def test_native_sequential_empty_forward_returns_input_by_identity():
    x = NativeTensor.from_array(X, requires_grad=True)
    out = NativeSequential()(x)
    assert out is x  # no node, no copy, no ownership change
    assert out._parents == ()


def test_native_sequential_executes_children_in_order():
    log = []
    seq = NativeSequential(_Tag("a", log), _Tag("b", log), _Tag("c", log))
    sentinel = object()  # composition only: children define input rules
    result = seq(sentinel)
    assert result is sentinel
    assert log == ["a", "b", "c"]  # in order, none skipped


@needs_native
def test_native_sequential_linear_relu_linear_forward_is_exact():
    model = _model()
    out = model(NativeTensor.from_array(X))
    assert type(out) is NativeTensor
    assert out.shape == (2, 2)
    expected = np.maximum(X @ W1 + B1, 0.0) @ W2 + B2
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)


@needs_native
def test_native_sequential_nested_composition_works():
    inner = NativeSequential(NativeLinear(2, 2, seed=0))
    outer = NativeSequential(inner, NativeReLU())
    assert [name for name, _ in outer.named_parameters()] == [
        "0.0.weight", "0.0.bias",
    ]
    assert list(outer.state_dict().keys()) == ["0.0.weight", "0.0.bias"]
    out = outer(NativeTensor.from_array(X))
    expected = np.maximum(X @ inner[0].weight.to_numpy()
                          + inner[0].bias.to_numpy(), 0.0)
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)


@needs_native
def test_native_sequential_child_exceptions_propagate():
    model = NativeSequential(
        NativeLinear(2, 3, seed=0),
        NativeLinear(5, 2, seed=1),  # expects 5 features, receives 3
    )
    with pytest.raises(ValueError, match="in_features=5"):
        model(NativeTensor.from_array(X))


@needs_native
def test_native_sequential_forward_adds_no_manual_autograd_node():
    assert "backward" not in NativeSequential.__dict__
    model = _model()
    x = NativeTensor.from_array(X, requires_grad=True)
    out = model(x)
    assert out._op == "add"  # the final child's own broadcast-add node


@needs_native
def test_native_sequential_forward_uses_no_numpy_compute(monkeypatch):
    model = _model()
    x = NativeTensor.from_array(X, requires_grad=True)

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("matmul", "dot", "add", "multiply", "maximum", "where"):
        monkeypatch.setattr(np, name, _tripwire)
    model(x).sum().backward()
    monkeypatch.undo()
    assert x.grad is not None


@needs_native
def test_native_sequential_training_mode_does_not_alter_composition():
    model = _model()
    x = NativeTensor.from_array(X)
    train_out = model.train()(x).to_numpy()
    eval_out = model.eval()(x).to_numpy()
    assert np.array_equal(train_out, eval_out)


# ======================================================================
# Shared modules: position-based execution, identity-based ownership
# ======================================================================


def test_native_sequential_shared_child_executes_once_per_slot():
    log = []
    shared = _Tag("s", log)
    seq = NativeSequential(shared, shared)
    seq(object())
    assert log == ["s", "s"]  # two slots -> two executions
    assert list(seq) == [shared, shared]  # iteration mirrors execution
    # ... while traversal deduplicates by identity, first path winning.
    assert seq.modules() == [seq, shared]
    assert list(seq.named_modules()) == [("", seq), ("0", shared)]


@needs_native
def test_native_sequential_shared_doubler_composes_numerically():
    shared = _Doubler()
    seq = NativeSequential(shared, shared)
    out = seq(NativeTensor.from_array(X))
    assert np.array_equal(out.to_numpy(), 4.0 * X)  # doubled twice


@needs_native
def test_native_sequential_shared_linear_parameters_and_state_once():
    shared = NativeLinear(2, 2, seed=0)
    seq = NativeSequential(shared, NativeReLU(), shared)
    assert [name for name, _ in seq.named_parameters()] == [
        "0.weight", "0.bias",  # first-discovered canonical path
    ]
    assert seq.parameters() == [shared.weight, shared.bias]
    assert list(seq.state_dict().keys()) == ["0.weight", "0.bias"]
    # A duplicate alias key is unexpected under v3.3 strict rules.
    state = seq.state_dict()
    state["2.weight"] = NativeTensor.from_array(shared.weight.to_numpy())
    with pytest.raises(ValueError, match="2.weight"):
        seq.load_state_dict(state)
    result = seq.load_state_dict(state, strict=False)
    assert result.unexpected_keys == ("2.weight",)


@needs_native
def test_native_sequential_shared_train_eval_and_zero_grad():
    shared = NativeLinear(2, 2, seed=0)
    seq = NativeSequential(shared, shared)
    seq.eval()
    assert seq.training is False and shared.training is False
    seq.train()
    assert shared.training is True
    x = NativeTensor.from_array(X)
    seq(x).sum().backward()
    assert shared.weight.grad is not None
    seq.zero_grad()  # the shared parameter is cleared once, harmlessly
    assert shared.weight.grad is None
    assert shared.bias.grad is None


# ======================================================================
# Traversal, zero_grad, train/eval
# ======================================================================


@needs_native
def test_native_sequential_traversal_is_deterministic_and_complete():
    model = _model()
    names = [name for name, _ in model.named_parameters()]
    assert names == ["0.weight", "0.bias", "2.weight", "2.bias"]
    assert names == [name for name, _ in model.named_parameters()]
    assert model.modules() == [model, model[0], model[1], model[2]]
    assert model.parameters() == [
        model[0].weight, model[0].bias, model[2].weight, model[2].bias,
    ]


@needs_native
def test_native_sequential_frozen_parameters_remain_discoverable():
    model = NativeSequential(
        NativeLinear(2, 3, seed=0, requires_grad=False), NativeReLU(),
    )
    assert [name for name, _ in model.named_parameters()] == [
        "0.weight", "0.bias",
    ]
    assert model[0].weight.requires_grad is False


@needs_native
def test_native_sequential_zero_grad_clears_without_touching_data():
    model = _model()
    x = NativeTensor.from_array(X)
    model(x).sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
    assert model.zero_grad() is None
    for parameter in model.parameters():
        assert parameter.grad is None
        assert parameter.requires_grad is True
    assert np.array_equal(model[0].weight.to_numpy(), W1)  # data intact


def test_native_sequential_train_eval_propagate_through_nesting():
    inner = NativeSequential(NativeReLU())
    outer = NativeSequential(inner, NativeReLU())
    outer.eval()
    assert outer.training is False
    assert inner.training is False
    assert inner[0].training is False
    outer.train()
    assert inner[0].training is True


# ======================================================================
# State dictionary
# ======================================================================


@needs_native
def test_native_sequential_state_keys_are_slot_derived():
    assert NativeSequential().state_dict() == {}
    assert NativeSequential(NativeReLU(), NativeReLU()).state_dict() == {}
    model = _model()
    assert list(model.state_dict().keys()) == [
        "0.weight", "0.bias", "2.weight", "2.bias",
    ]
    bias_free = NativeSequential(NativeLinear(2, 2, bias=False, seed=0))
    assert list(bias_free.state_dict().keys()) == ["0.weight"]


@needs_native
def test_native_sequential_state_roundtrip_preserves_everything():
    model = _model()
    weight = model[0].weight
    model(NativeTensor.from_array(X)).sum().backward()
    grad = weight.grad
    model.eval()
    snapshot = model.state_dict()
    other_state = {
        "0.weight": NativeTensor.from_array(W1 * 2.0),
        "0.bias": NativeTensor.from_array(B1),
        "2.weight": NativeTensor.from_array(W2),
        "2.bias": NativeTensor.from_array(B2),
    }
    model.load_state_dict(other_state)
    assert np.array_equal(model[0].weight.to_numpy(), W1 * 2.0)
    assert model[0].weight is weight  # identity survived
    assert weight.grad is grad  # gradient survived by identity
    assert weight.requires_grad is True
    assert model.training is False  # training state survived
    assert np.array_equal(snapshot["0.weight"].to_numpy(), W1)  # independent
    model.load_state_dict(snapshot)  # and the snapshot loads back
    assert np.array_equal(model[0].weight.to_numpy(), W1)


@needs_native
def test_native_sequential_incompatible_state_fails_atomically():
    model = _model()
    bad = _state()
    bad["2.weight"] = NativeTensor.from_array(np.zeros((4, 2)))  # bad shape
    with pytest.raises(ValueError, match="2.weight"):
        model.load_state_dict(bad)
    assert np.array_equal(model[0].weight.to_numpy(), W1)
    assert np.array_equal(model[2].weight.to_numpy(), W2)
    missing = _state()
    del missing["0.bias"]
    with pytest.raises(ValueError, match="0.bias"):
        model.load_state_dict(missing)
    result = model.load_state_dict(missing, strict=False)
    assert result.missing_keys == ("0.bias",)


# ======================================================================
# Backward integration
# ======================================================================


@needs_native
def test_native_sequential_backward_matches_exact_reference():
    model = _model()
    x = NativeTensor.from_array(X, requires_grad=True)
    model(x).multiply(NativeTensor.from_array(UP)).sum().backward()
    ref = _reference_grads()
    assert np.allclose(x.grad.to_numpy(), ref["x"], atol=1e-12)
    assert np.allclose(model[0].weight.grad.to_numpy(), ref["w1"], atol=1e-12)
    assert np.allclose(model[0].bias.grad.to_numpy(), ref["b1"], atol=1e-12)
    assert np.allclose(model[2].weight.grad.to_numpy(), ref["w2"], atol=1e-12)
    assert np.allclose(model[2].bias.grad.to_numpy(), ref["b2"], atol=1e-12)


@needs_native
def test_native_sequential_bias_free_backward_works():
    model = NativeSequential(
        NativeLinear(2, 3, bias=False, seed=0),
        NativeReLU(),
        NativeLinear(3, 2, bias=False, seed=1),
    )
    x = NativeTensor.from_array(X, requires_grad=True)
    model(x).sum().backward()
    assert model[0].weight.grad is not None
    assert model[2].weight.grad is not None
    assert x.grad is not None


@needs_native
def test_native_sequential_frozen_layers_and_frozen_model():
    # Frozen first layer: only the second layer and the input learn.
    model = NativeSequential(
        NativeLinear(2, 3, seed=0, requires_grad=False),
        NativeReLU(),
        NativeLinear(3, 2, seed=1),
    )
    x = NativeTensor.from_array(X, requires_grad=True)
    model(x).sum().backward()
    assert model[0].weight.grad is None
    assert model[0].bias.grad is None
    assert model[2].weight.grad is not None
    assert x.grad is not None
    # Entirely frozen model: only the requiring input learns.
    frozen = _model(requires_grad=False)
    y = NativeTensor.from_array(X, requires_grad=True)
    frozen(y).sum().backward()
    assert all(p.grad is None for p in frozen.parameters())
    assert y.grad is not None


@needs_native
def test_native_sequential_branching_accumulates():
    model = _model()
    x = NativeTensor.from_array(X)
    model(x).add(model(x)).sum().backward()
    single = _model()
    single(NativeTensor.from_array(X)).sum().backward()
    assert np.allclose(
        model[0].weight.grad.to_numpy(),
        2.0 * single[0].weight.grad.to_numpy(),
        atol=1e-12,
    )


@needs_native
def test_native_sequential_graph_lifetime_and_fresh_iterations():
    model = _model()
    x = NativeTensor.from_array(X)
    loss = model(x).sum()
    loss.backward()
    with pytest.raises(RuntimeError, match="retain_graph"):
        loss.backward()  # one-shot cleanup through the full composition
    retained = model(x).sum()
    retained.backward(retain_graph=True)
    retained.backward(retain_graph=True)
    first_bias = model[2].bias.grad.to_numpy()
    model.zero_grad()
    model(x).sum().backward()  # a fresh iteration after zero_grad
    assert np.allclose(first_bias, 3.0 * model[2].bias.grad.to_numpy(),
                       atol=1e-12)


# ======================================================================
# Finite differences
# ======================================================================
#
# Central differences with eps=1e-6 and atol=1e-6 (float64; all hidden
# pre-activations at least 0.1 from ReLU's zero boundary). Perturbed
# values enter fresh models through load_state_dict BEFORE forward —
# the supported sequence; nothing mutates between forward and backward.


def _native_loss(w1, b1, w2, b2, x):
    model = NativeSequential(
        NativeLinear(2, 3, seed=0), NativeReLU(), NativeLinear(3, 2, seed=1),
    )
    model.load_state_dict(_state(w1, b1, w2, b2))
    out = model(NativeTensor.from_array(x))
    return float(out.multiply(NativeTensor.from_array(UP)).sum().to_numpy())


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
def test_native_sequential_gradients_match_finite_differences():
    model = _model()
    x = NativeTensor.from_array(X, requires_grad=True)
    model(x).multiply(NativeTensor.from_array(UP)).sum().backward()
    fd = {
        "x": _numeric_grad(lambda v: _native_loss(W1, B1, W2, B2, v), X),
        "w1": _numeric_grad(lambda v: _native_loss(v, B1, W2, B2, X), W1),
        "b1": _numeric_grad(lambda v: _native_loss(W1, v, W2, B2, X), B1),
        "w2": _numeric_grad(lambda v: _native_loss(W1, B1, v, B2, X), W2),
        "b2": _numeric_grad(lambda v: _native_loss(W1, B1, W2, v, X), B2),
    }
    assert np.allclose(x.grad.to_numpy(), fd["x"], atol=1e-6)
    assert np.allclose(model[0].weight.grad.to_numpy(), fd["w1"], atol=1e-6)
    assert np.allclose(model[0].bias.grad.to_numpy(), fd["b1"], atol=1e-6)
    assert np.allclose(model[2].weight.grad.to_numpy(), fd["w2"], atol=1e-6)
    assert np.allclose(model[2].bias.grad.to_numpy(), fd["b2"], atol=1e-6)


# ======================================================================
# Isolation
# ======================================================================


def test_native_sequential_isolated_from_stable_framework():
    import tensorforge
    import tensorforge.experimental as experimental

    assert experimental.NativeSequential is NativeSequential
    assert not hasattr(tensorforge, "NativeSequential")
    assert not hasattr(tensorforge.nn, "NativeSequential")
    assert not issubclass(NativeSequential, tensorforge.nn.Module)
    # Stable Sequential is untouched and still NumPy-backed.
    stable = tensorforge.nn.Sequential(
        tensorforge.nn.Linear(2, 3), tensorforge.nn.ReLU(),
    )
    t = tensorforge.Tensor(X, requires_grad=True)
    out = stable(t)
    assert isinstance(out, tensorforge.Tensor)
    out.sum().backward()
    assert isinstance(t.grad, np.ndarray)
