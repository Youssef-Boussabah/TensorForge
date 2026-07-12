"""Tests for NativeParameter value versioning, the controlled no-grad
mutation primitive, and stale-graph detection (Advanced C++ v3.7, the
seventh Phase C milestone — the safety foundation NativeSGD builds on).

The contract under test:

- every NativeParameter carries a read-only, monotonically increasing
  value ``version`` (0 at construction) that counts **replacements** of
  the owned numerical value — ``copy_value_(source)`` and successful
  ``load_state_dict`` (once per matched canonical parameter, identical
  values included) and nothing else;
- ``copy_value_`` is the one controlled mutation: identity, leaf/
  graph-free state, ``requires_grad``, gradients, and registrations all
  survive, the copy is native/owning/contiguous/never aliased, and any
  failure changes nothing;
- operations whose backward reads a direct parameter operand's forward
  value (``multiply``/``matmul``/``relu``) record the version at forward
  time, and ``backward()`` raises a deterministic stale-value
  RuntimeError — before any callback or gradient commit — if it changed.
  This is the documented **op-level** policy: every direct parameter
  operand of a value-sensitive op is guarded, even in the corner where
  sibling requires_grad flags mean the value would not actually be read.
  Value-independent backwards (add/subtract/sum/mean/reshape/transpose/
  contiguous_copy/narrow) record nothing and stay valid — with
  mathematically correct gradients — across parameter mutation.

NumPy appears below only for input preparation and references; the
mutation, loading, stale-preflight, and fresh-pass paths are all native
(a tripwire test proves it).

Selector:
python -m pytest -q -k "native_parameter_version or stale_parameter_graph or mutation_safety"
"""

import numpy as np
import pytest

import tensorforge
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
from tensorforge.experimental import native_module as native_module_impl

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


P_VALUES = np.array([[1.0, -2.0], [0.5, 3.0]])
NEW_VALUES = np.array([[4.0, 1.0], [-1.0, 2.0]])
X_VALUES = np.array([[1.0, 2.0], [3.0, -1.0]])


def _param(values=P_VALUES, requires_grad=True):
    return NativeParameter(values, requires_grad=requires_grad)


class _Block(NativeModule):
    """weight + bias — the smallest state-dictionary test module."""

    def __init__(self):
        super().__init__()
        self.weight = NativeParameter(P_VALUES)
        self.bias = NativeParameter([0.5, -1.0])


# ======================================================================
# Version basics
# ======================================================================


@needs_native
def test_native_parameter_version_starts_at_zero_typed_and_readonly():
    p = _param()
    assert p.version == 0
    assert isinstance(p.version, int) and not isinstance(p.version, bool)
    frozen = _param(requires_grad=False)
    assert frozen.version == 0
    with pytest.raises(AttributeError):
        p.version = 3
    assert p.version == 0
    p.close()
    with pytest.raises(RuntimeError, match="closed"):
        p.version


@needs_native
def test_native_parameter_version_untouched_by_grads_registration_and_mode():
    p = _param()
    x = NativeTensor.from_array(X_VALUES)
    # Gradient accumulation and zero_grad are not value replacements.
    p.multiply(x).sum().backward()
    assert p.grad is not None
    assert p.version == 0
    p.zero_grad()
    assert p.version == 0
    # Registration, aliasing, removal, replacement, snapshots, and mode.
    module = NativeModule()
    module.weight = p
    module.alias = p
    assert p.version == 0
    module.register_parameter("alias", None)
    module.weight = _param(NEW_VALUES)  # replaced in the registry, not mutated
    assert p.version == 0
    block = _Block()
    block.state_dict()
    block.train(False)
    block.train(True)
    assert block.weight.version == 0 and block.bias.version == 0


@needs_native
def test_native_parameter_version_not_on_operation_results():
    p = _param()
    c = NativeTensor.from_array(NEW_VALUES)
    for result in (p.add(c), p.multiply(c), p.T, p.detach(),
                   p.contiguous_copy()):
        assert type(result) is NativeTensor
        assert getattr(result, "_version", None) is None
        assert not hasattr(result, "version")


# ======================================================================
# Controlled mutation: copy_value_
# ======================================================================


@needs_native
def test_mutation_safety_copy_value_replaces_value_and_increments_once():
    p = _param()
    source = NativeTensor.from_array(NEW_VALUES)
    result = p.copy_value_(source)
    assert result is p  # identity preserved, chainable
    assert np.array_equal(p.to_numpy(), NEW_VALUES)
    assert p.version == 1
    # Still an open, owning, contiguous, graph-free trainable leaf.
    assert p.requires_grad and p.is_leaf and p.owns_core and p.contiguous
    assert p._parents == () and p._backward is None and p._op == ""
    assert not p._graph_freed
    # The source is untouched and independent.
    assert not source.closed
    assert np.array_equal(source.to_numpy(), NEW_VALUES)


@needs_native
def test_mutation_safety_copy_value_from_parameter_self_and_snapshot():
    p = _param()
    other = _param(NEW_VALUES)
    p.copy_value_(other)  # a NativeParameter is a plain value source
    assert np.array_equal(p.to_numpy(), NEW_VALUES)
    assert p.version == 1
    assert other.version == 0  # the source's version never moves
    p.copy_value_(p)  # self-copy: staged before the swap, so it works
    assert np.array_equal(p.to_numpy(), NEW_VALUES)
    assert p.version == 2
    block = _Block()
    snapshot = block.state_dict()
    block.weight.copy_value_(snapshot["weight"])  # snapshot as source
    assert block.weight.version == 1
    for value in snapshot.values():
        value.close()


@needs_native
def test_mutation_safety_preserves_grad_frozen_state_and_source_graph():
    p = _param()
    c = NativeTensor.from_array(NEW_VALUES)
    p.multiply(c).sum().backward()
    grad_before = p.grad
    grad_values = grad_before.to_numpy()
    p.copy_value_(NativeTensor.from_array(NEW_VALUES))
    assert p.grad is grad_before  # same object
    assert np.array_equal(p.grad.to_numpy(), grad_values)  # same value
    # A frozen parameter mutates the same way and stays frozen.
    frozen = _param(requires_grad=False)
    frozen.copy_value_(NativeTensor.from_array(NEW_VALUES))
    assert frozen.requires_grad is False and frozen.version == 1
    # A non-leaf source keeps its graph: copying reads only the value.
    a = NativeTensor.from_array(P_VALUES, requires_grad=True)
    non_leaf = a.add(NativeTensor.from_array(NEW_VALUES))
    p.copy_value_(non_leaf)
    assert non_leaf._backward is not None and non_leaf._parents != ()
    non_leaf.sum().backward()  # source graph still fully usable
    assert a.grad is not None


@needs_native
def test_mutation_safety_result_is_owned_never_aliased_and_views_copy():
    p = _param()
    source = _param(NEW_VALUES)
    p.copy_value_(source)
    # Mutating the source afterwards must not reach the destination.
    source.copy_value_(NativeTensor.from_array(P_VALUES))
    assert np.array_equal(p.to_numpy(), NEW_VALUES)
    # A strided borrowing view materializes at its logical shape.
    base = NativeTensor.from_array(X_VALUES)
    view = base.T
    assert not view.contiguous
    p.copy_value_(view)
    assert np.array_equal(p.to_numpy(), X_VALUES.T)
    assert p.contiguous and p.owns_core


@needs_native
def test_mutation_safety_identical_values_still_increment_monotonically():
    p = _param()
    same = NativeTensor.from_array(P_VALUES)
    for expected in (1, 2, 3):
        p.copy_value_(same)  # replacement counts; value equality never does
        assert p.version == expected
        assert np.array_equal(p.to_numpy(), P_VALUES)


@needs_native
def test_mutation_safety_invalid_sources_rejected_without_changes():
    p = _param()
    p.multiply(NativeTensor.from_array(NEW_VALUES)).sum().backward()
    grad_before = p.grad
    for bad in (NEW_VALUES.tolist(), NEW_VALUES, 1.5, None):
        with pytest.raises(TypeError):
            p.copy_value_(bad)
    with pytest.raises(TypeError, match="never mix"):
        p.copy_value_(tensorforge.Tensor(NEW_VALUES))
    with pytest.raises(TypeError, match="never mix"):
        p.copy_value_(tensorforge.Parameter(NEW_VALUES))
    assert p.version == 0
    assert np.array_equal(p.to_numpy(), P_VALUES)
    assert p.grad is grad_before


@needs_native
def test_mutation_safety_closed_and_mismatched_operands_fail_cleanly():
    p = _param()
    closed_source = NativeTensor.from_array(NEW_VALUES)
    closed_source.close()
    with pytest.raises(RuntimeError, match="closed"):
        p.copy_value_(closed_source)
    with pytest.raises(ValueError, match="shape"):
        p.copy_value_(NativeTensor.from_array([1.0, 2.0]))
    assert p.version == 0
    assert np.array_equal(p.to_numpy(), P_VALUES)
    # dtype/device mismatches are unconstructible on the float64/cpu-only
    # runtime; the check exists and is exercised via _adopt_value_core's
    # defensive re-validation in the v3.3 suite.
    destination = _param()
    destination.close()
    with pytest.raises(RuntimeError, match="closed"):
        destination.copy_value_(NativeTensor.from_array(NEW_VALUES))


@needs_native
def test_mutation_safety_failed_commit_changes_nothing(monkeypatch):
    p = _param()
    p.multiply(NativeTensor.from_array(NEW_VALUES)).sum().backward()
    grad_before = p.grad

    def forced_failure(self, new_core):
        raise RuntimeError("forced commit failure")

    monkeypatch.setattr(NativeParameter, "_adopt_value_core", forced_failure)
    with pytest.raises(RuntimeError, match="forced commit failure"):
        p.copy_value_(NativeTensor.from_array(NEW_VALUES))
    monkeypatch.undo()
    assert p.version == 0
    assert np.array_equal(p.to_numpy(), P_VALUES)
    assert p.grad is grad_before
    p.copy_value_(NativeTensor.from_array(NEW_VALUES))  # recovers fully
    assert p.version == 1


@needs_native
def test_mutation_safety_paths_use_no_numpy_compute(monkeypatch):
    p = _param()
    block = _Block()
    state = block.state_dict()
    source = NativeTensor.from_array(NEW_VALUES)
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    stale_out = x.matmul(p).sum()  # built before the mutation

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("add", "subtract", "multiply", "matmul", "sum", "mean",
                 "divide", "negative", "copyto"):
        monkeypatch.setattr(np, name, _tripwire)
    # Controlled mutation, state loading, the stale preflight, and a
    # fresh forward/backward — all inside the guard.
    p.copy_value_(source)
    block.load_state_dict(state)
    with pytest.raises(RuntimeError, match="stale"):
        stale_out.backward()
    fresh = x.matmul(p).sum()
    fresh.backward()
    monkeypatch.undo()
    assert np.array_equal(p.to_numpy(), NEW_VALUES)
    assert np.array_equal(x.grad.to_numpy(), np.ones((2, 2)) @ NEW_VALUES.T)
    for value in state.values():
        value.close()


# ======================================================================
# State loading and versions
# ======================================================================


@needs_native
def test_native_parameter_version_strict_load_increments_each_once():
    block = _Block()
    weight, bias = block.weight, block.bias
    block.weight.multiply(NativeTensor.from_array(NEW_VALUES)).sum().backward()
    grad_before = weight.grad
    state = block.state_dict()  # numerically identical values
    result = block.load_state_dict(state)
    assert result.missing_keys == () and result.unexpected_keys == ()
    assert weight.version == 1 and bias.version == 1  # replacement counts
    assert block.weight is weight and block.bias is bias
    assert np.array_equal(weight.to_numpy(), P_VALUES)
    assert weight.grad is grad_before  # gradients survive loading
    block.load_state_dict(state)  # repeated loads are predictable
    assert weight.version == 2 and bias.version == 2
    for value in state.values():
        value.close()


@needs_native
def test_native_parameter_version_non_strict_partial_and_unexpected_keys():
    block = _Block()
    state = {"weight": NativeTensor.from_array(NEW_VALUES),
             "extra": NativeTensor.from_array(NEW_VALUES)}
    result = block.load_state_dict(state, strict=False)
    assert result.missing_keys == ("bias",)
    assert result.unexpected_keys == ("extra",)
    assert block.weight.version == 1
    assert np.array_equal(block.weight.to_numpy(), NEW_VALUES)
    # The missing parameter kept its value and version; the ignored
    # unexpected key incremented nothing.
    assert block.bias.version == 0
    assert np.array_equal(block.bias.to_numpy(), [0.5, -1.0])


@needs_native
def test_native_parameter_version_failed_validation_changes_no_versions():
    block = _Block()
    good = NativeTensor.from_array(NEW_VALUES)
    with pytest.raises(ValueError):  # strict key incompatibility
        block.load_state_dict({"weight": good})
    with pytest.raises(TypeError):  # invalid value type
        block.load_state_dict({"weight": NEW_VALUES,
                               "bias": NativeTensor.from_array([1.0, 2.0])})
    closed = NativeTensor.from_array([1.0, 2.0])
    closed.close()
    with pytest.raises(RuntimeError):  # closed source value
        block.load_state_dict({"weight": good, "bias": closed})
    with pytest.raises(ValueError):  # shape mismatch
        block.load_state_dict({"weight": good,
                               "bias": NativeTensor.from_array([[1.0]])})
    assert block.weight.version == 0 and block.bias.version == 0
    assert np.array_equal(block.weight.to_numpy(), P_VALUES)


@needs_native
def test_native_parameter_version_staging_failure_changes_nothing(monkeypatch):
    block = _Block()
    state = block.state_dict()  # built before the patch
    real_copy = native_module_impl._native_copy
    calls = {"n": 0}

    def flaky_copy(core):
        calls["n"] += 1
        if calls["n"] == 2:
            raise MemoryError("forced staging failure")
        return real_copy(core)

    monkeypatch.setattr(native_module_impl, "_native_copy", flaky_copy)
    with pytest.raises(MemoryError):
        block.load_state_dict(state)
    monkeypatch.undo()
    assert block.weight.version == 0 and block.bias.version == 0
    assert np.array_equal(block.weight.to_numpy(), P_VALUES)
    for value in state.values():
        value.close()


@needs_native
def test_native_parameter_version_commit_failure_restores_all(monkeypatch):
    block = _Block()
    block.weight.multiply(NativeTensor.from_array(NEW_VALUES)).sum().backward()
    grad_before = block.weight.grad
    state = {"weight": NativeTensor.from_array(NEW_VALUES),
             "bias": NativeTensor.from_array([9.0, 9.0])}
    real_adopt = NativeParameter._adopt_value_core
    calls = {"n": 0}

    def flaky_adopt(self, new_core):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("forced commit failure")
        return real_adopt(self, new_core)

    monkeypatch.setattr(NativeParameter, "_adopt_value_core", flaky_adopt)
    with pytest.raises(RuntimeError, match="forced commit failure"):
        block.load_state_dict(state)
    monkeypatch.undo()
    # The tentative first swap was rolled back: values, versions, and
    # gradients are all exactly as before the failed load.
    assert np.array_equal(block.weight.to_numpy(), P_VALUES)
    assert np.array_equal(block.bias.to_numpy(), [0.5, -1.0])
    assert block.weight.version == 0 and block.bias.version == 0
    assert block.weight.grad is grad_before
    block.load_state_dict(state)  # the transaction recovers fully
    assert block.weight.version == 1 and block.bias.version == 1


# ======================================================================
# Stale-graph detection: value-sensitive operations
# ======================================================================


@needs_native
def test_stale_parameter_graph_matmul_mutation_raises_before_any_gradient():
    w = _param()
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    out = x.matmul(w).sum()
    w.copy_value_(NativeTensor.from_array(NEW_VALUES))
    with pytest.raises(RuntimeError) as first:
        out.backward()
    message = str(first.value)
    assert "stale" in message and "matmul" in message
    assert "expected version 0" in message and "current version 1" in message
    assert "run the forward pass again" in message
    assert "freed" not in message  # not the freed-graph error
    # Nothing moved: no gradient (input or weight) was committed, the
    # graph is structurally intact and not freed, and the failure repeats.
    assert x.grad is None and w.grad is None
    assert out._parents != () and out._backward is not None
    assert not out._graph_freed
    with pytest.raises(RuntimeError) as second:
        out.backward()
    assert str(second.value) == message  # deterministic


@needs_native
def test_stale_parameter_graph_multiply_relu_and_duplicate_parent():
    # multiply with both operands parameters: either mutation is fatal
    # because each gradient reads the other operand's forward value.
    a, b = _param(), _param(NEW_VALUES)
    out = a.multiply(b).sum()
    b.copy_value_(NativeTensor.from_array(P_VALUES))
    with pytest.raises(RuntimeError, match="multiply"):
        out.backward()
    # The documented op-level policy: a direct parameter operand of a
    # value-sensitive op is guarded even when the sibling does not
    # require grad (the conservative corner — safety over precision).
    p = _param()
    out2 = p.multiply(NativeTensor.from_array(NEW_VALUES)).sum()
    p.copy_value_(NativeTensor.from_array(NEW_VALUES))
    with pytest.raises(RuntimeError, match="stale"):
        out2.backward()
    # relu reads its input's forward value (the mask).
    r = _param()
    out3 = r.relu().sum()
    r.copy_value_(NativeTensor.from_array(NEW_VALUES))
    with pytest.raises(RuntimeError, match="relu"):
        out3.backward()
    # Duplicate parent: d * d records d twice at the same version — the
    # check is idempotent before mutation and fatal after.
    d = _param()
    d.multiply(d).sum().backward()
    assert np.array_equal(d.grad.to_numpy(), 2.0 * P_VALUES)
    d.zero_grad()
    out4 = d.multiply(d).sum()
    d.copy_value_(NativeTensor.from_array(NEW_VALUES))
    with pytest.raises(RuntimeError, match="stale"):
        out4.backward()
    assert d.grad is None


@needs_native
def test_stale_parameter_graph_failure_preserves_existing_gradients():
    w = _param()
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    x.matmul(w).sum().backward()  # a first, completed pass
    x_grad, w_grad = x.grad, w.grad
    x_grad_values = x_grad.to_numpy()
    out = x.matmul(w).sum()  # a second graph
    w.copy_value_(NativeTensor.from_array(NEW_VALUES))
    with pytest.raises(RuntimeError, match="stale"):
        out.backward()
    # Existing gradients survive by identity and value.
    assert x.grad is x_grad and w.grad is w_grad
    assert np.array_equal(x.grad.to_numpy(), x_grad_values)
    # zero_grad clears gradients but never bypasses stale detection.
    x.zero_grad()
    w.zero_grad()
    with pytest.raises(RuntimeError, match="stale"):
        out.backward()
    assert x.grad is None and w.grad is None


@needs_native
def test_stale_parameter_graph_old_value_reload_does_not_revive():
    w = _param()
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    out = x.matmul(w).sum()
    original = NativeTensor.from_array(P_VALUES)
    w.copy_value_(NativeTensor.from_array(NEW_VALUES))
    w.copy_value_(original)  # numerically the forward-time value again
    assert np.array_equal(w.to_numpy(), P_VALUES)
    with pytest.raises(RuntimeError, match="expected version 0, current version 2"):
        out.backward()  # versions are monotonic — replacement counted


@needs_native
def test_stale_parameter_graph_error_distinct_from_freed_and_closed():
    w = _param()
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    freed_out = x.matmul(w).sum()
    freed_out.backward()
    with pytest.raises(RuntimeError) as freed:
        freed_out.backward()
    assert "freed" in str(freed.value) and "stale" not in str(freed.value)
    stale_out = x.matmul(w).sum()
    w.copy_value_(NativeTensor.from_array(NEW_VALUES))
    with pytest.raises(RuntimeError) as stale:
        stale_out.backward()
    assert "stale" in str(stale.value) and "freed" not in str(stale.value)
    assert "retain_graph" not in str(stale.value)  # it would not help
    closed = NativeTensor.from_array(X_VALUES)
    closed.close()
    with pytest.raises(RuntimeError) as shut:
        closed.to_numpy()
    assert "closed" in str(shut.value) and "stale" not in str(shut.value)


@needs_native
def test_stale_parameter_graph_fresh_forward_uses_new_values():
    w = _param()
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    stale = x.matmul(w).sum()
    w.copy_value_(NativeTensor.from_array(NEW_VALUES))
    with pytest.raises(RuntimeError, match="stale"):
        stale.backward()
    fresh = x.matmul(w).sum()
    fresh.backward()
    ones = np.ones((2, 2))
    assert np.array_equal(x.grad.to_numpy(), ones @ NEW_VALUES.T)
    assert np.array_equal(w.grad.to_numpy(), X_VALUES.T @ ones)


# ======================================================================
# Value-independent operations stay valid across mutation
# ======================================================================


@needs_native
def test_stale_parameter_graph_add_and_subtract_are_value_independent():
    p = _param()
    c = NativeTensor.from_array(NEW_VALUES)
    out = p.add(c).sum()
    p.copy_value_(NativeTensor.from_array(NEW_VALUES))
    out.backward()  # d(p + c)/dp = 1 regardless of p's value
    assert np.array_equal(p.grad.to_numpy(), np.ones((2, 2)))
    p.zero_grad()
    out2 = p.subtract(c).sum()
    out3 = c.subtract(p).sum()
    p.copy_value_(NativeTensor.from_array(P_VALUES))
    out2.backward()
    assert np.array_equal(p.grad.to_numpy(), np.ones((2, 2)))
    p.zero_grad()
    out3.backward()
    assert np.array_equal(p.grad.to_numpy(), -np.ones((2, 2)))


@needs_native
def test_stale_parameter_graph_view_and_reduction_chain_value_independent():
    p = NativeParameter([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # reshape → transpose → contiguous_copy → narrow → sum: every
    # backward in the chain is metadata-only.
    out = (p.reshape((3, 2)).transpose().contiguous_copy()
           .narrow(1, 0, 2).sum())
    p.copy_value_(NativeTensor.from_array(np.zeros((2, 3))))
    out.backward()
    # Elements kept by the narrow (flat positions 0-3) get gradient 1.
    assert np.array_equal(p.grad.to_numpy(), [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    p.zero_grad()
    out2 = p.mean()
    p.copy_value_(NativeTensor.from_array(np.ones((2, 3))))
    out2.backward()  # mean backward reads only shape and count
    assert np.allclose(p.grad.to_numpy(), np.full((2, 3), 1.0 / 6.0),
                       atol=1e-15)


@needs_native
def test_stale_parameter_graph_sensitive_edge_dominates_mixed_graph():
    # One graph where the parameter feeds both an add (safe) and a
    # multiply (sensitive): the sensitive edge makes the graph stale.
    p = _param()
    c = NativeTensor.from_array(NEW_VALUES)
    out = p.add(c).multiply(p).sum()
    p.copy_value_(NativeTensor.from_array(NEW_VALUES))
    with pytest.raises(RuntimeError, match="multiply"):
        out.backward()
    assert p.grad is None


# ======================================================================
# Retained graphs
# ======================================================================


@needs_native
def test_stale_parameter_graph_retained_backward_then_mutation():
    w = _param()
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    out = x.matmul(w).sum()
    out.backward(retain_graph=True)
    x_grad, w_grad = x.grad, w.grad
    x_grad_values = x_grad.to_numpy()
    w_grad_values = w_grad.to_numpy()
    w.copy_value_(NativeTensor.from_array(NEW_VALUES))
    # The retained graph validates versions on every pass.
    with pytest.raises(RuntimeError, match="stale"):
        out.backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="stale"):
        out.backward()
    # The failed reuses added nothing: same objects, same values.
    assert x.grad is x_grad and w.grad is w_grad
    assert np.array_equal(x.grad.to_numpy(), x_grad_values)
    assert np.array_equal(w.grad.to_numpy(), w_grad_values)
    assert not out._graph_freed  # a stale failure never frees


@needs_native
def test_stale_parameter_graph_unmodified_retained_graph_still_works():
    w = _param()
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    out = x.matmul(w).sum()
    out.backward(retain_graph=True)
    out.backward(retain_graph=True)  # unmodified: accumulates normally
    ones = np.ones((2, 2))
    assert np.array_equal(w.grad.to_numpy(), 2.0 * (X_VALUES.T @ ones))
    out.backward()  # final default pass still frees the graph
    assert np.array_equal(w.grad.to_numpy(), 3.0 * (X_VALUES.T @ ones))
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()


# ======================================================================
# NativeLinear and full-model integration
# ======================================================================


@needs_native
def test_stale_parameter_graph_native_linear_weight_sensitive_bias_not():
    layer = NativeLinear(2, 3, seed=0)
    x = NativeTensor.from_array(X_VALUES)
    out = layer(x).sum()
    new_weight = NativeTensor.from_array(np.ones((2, 3)))
    layer.weight.copy_value_(new_weight)  # matmul backward needs it
    with pytest.raises(RuntimeError, match="stale"):
        out.backward()
    assert layer.weight.grad is None and layer.bias.grad is None
    # Bias-only mutation: the bias enters through add, whose backward is
    # value-independent — backward proceeds and every gradient is exact.
    fresh = layer(x).sum()
    layer.bias.copy_value_(NativeTensor.from_array([1.0, 2.0, 3.0]))
    fresh.backward()
    ones = np.ones((2, 3))
    assert np.array_equal(layer.weight.grad.to_numpy(), X_VALUES.T @ ones)
    assert np.array_equal(layer.bias.grad.to_numpy(), ones.sum(axis=0))


@needs_native
def test_stale_parameter_graph_native_linear_state_load_then_fresh_pass():
    layer = NativeLinear(2, 3, bias=False, seed=0)
    weight = layer.weight
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    out = layer(x).sum()
    new_values = np.array([[1.0, -1.0, 2.0], [0.5, 2.0, -1.0]])
    layer.load_state_dict({"weight": NativeTensor.from_array(new_values)})
    assert layer.weight is weight and weight.version == 1
    with pytest.raises(RuntimeError, match="stale"):
        out.backward()
    fresh = layer(x).sum()
    fresh.backward()  # the new graph uses — and matches — the new weight
    ones = np.ones((2, 3))
    assert np.array_equal(x.grad.to_numpy(), ones @ new_values.T)
    assert np.array_equal(weight.grad.to_numpy(), X_VALUES.T @ ones)


@needs_native
def test_stale_parameter_graph_full_model_load_between_forward_and_backward():
    model = NativeSequential(
        NativeLinear(2, 3, seed=0),
        NativeReLU(),
        NativeLinear(3, 2, seed=1),
    )
    loss_fn = NativeMSELoss()
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    target = NativeTensor.from_array(NEW_VALUES, requires_grad=True)
    loss = loss_fn(model(x), target)
    # Load the full snapshot with the first weight replaced: every
    # matched parameter is replaced (version +1), so the old graph's
    # sensitive matmul edges are stale.
    state = model.state_dict()
    replaced = dict(state)
    replaced["0.weight"] = NativeTensor.from_array(np.ones((2, 3)))
    model.load_state_dict(replaced)
    assert [p.version for p in model.parameters()] == [1, 1, 1, 1]
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward()
    # No gradient was partially committed anywhere.
    assert all(p.grad is None for p in model.parameters())
    assert x.grad is None and target.grad is None
    # A fresh forward → loss → backward works against the loaded values.
    fresh_loss = loss_fn(model(x), target)
    fresh_loss.backward()
    assert all(
        p.grad is not None and p.grad.shape == p.shape
        for p in model.parameters()
    )
    assert x.grad is not None and target.grad is not None
    model.zero_grad()  # versions are irrelevant to gradient clearing
    assert all(p.grad is None for p in model.parameters())
    assert [p.version for p in model.parameters()] == [1, 1, 1, 1]
    for value in state.values():
        value.close()


# ======================================================================
# Shared parameters
# ======================================================================


@needs_native
def test_native_parameter_version_shared_aliases_expose_one_version():
    shared = _param()
    module = NativeModule()
    module.a = shared
    module.b = shared
    assert module.a is module.b
    # Two graphs through two different use-sites of the one parameter.
    x = NativeTensor.from_array(X_VALUES, requires_grad=True)
    through_a = x.matmul(module.a).sum()
    through_b = module.b.multiply(module.b).sum()
    shared.copy_value_(NativeTensor.from_array(NEW_VALUES))
    assert module.a.version == 1 and module.b.version == 1  # one counter
    with pytest.raises(RuntimeError, match="stale"):
        through_a.backward()
    with pytest.raises(RuntimeError, match="stale"):
        through_b.backward()
    # Canonical loading updates the shared object exactly once.
    state = module.state_dict()
    assert list(state) == ["a"]  # first-discovered name wins
    module.load_state_dict(state)
    assert shared.version == 2
    for value in state.values():
        value.close()


@needs_native
def test_native_parameter_version_shared_rollback_restores_once(monkeypatch):
    shared = _param()
    module = NativeModule()
    module.a = shared
    module.b = shared
    module.c = _param(NEW_VALUES)
    state = module.state_dict()  # keys: "a" (canonical shared), "c"
    real_adopt = NativeParameter._adopt_value_core
    calls = {"n": 0}

    def flaky_adopt(self, new_core):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("forced commit failure")
        return real_adopt(self, new_core)

    monkeypatch.setattr(NativeParameter, "_adopt_value_core", flaky_adopt)
    with pytest.raises(RuntimeError, match="forced commit failure"):
        module.load_state_dict(state)
    monkeypatch.undo()
    # The shared parameter was swapped once and restored once — value
    # and version both exactly as before, with no double release.
    assert shared.version == 0
    assert np.array_equal(shared.to_numpy(), P_VALUES)
    assert module.c.version == 0
    module.load_state_dict(state)
    assert shared.version == 1 and module.c.version == 1
    for value in state.values():
        value.close()


# ======================================================================
# Stable-framework isolation
# ======================================================================


@needs_native
def test_mutation_safety_stable_framework_unversioned_and_unaffected():
    stable_param = tensorforge.Parameter([[1.0, 2.0]])
    assert not hasattr(stable_param, "version")
    assert not hasattr(stable_param, "copy_value_")
    stable = tensorforge.Tensor([[1.0, 2.0]], requires_grad=True)
    (stable * stable).sum().backward()
    assert np.allclose(stable.grad, [[2.0, 4.0]])
    # And the native mutation path never accepts stable objects.
    p = _param()
    with pytest.raises(TypeError, match="never mix"):
        p.copy_value_(stable_param)
    assert p.version == 0
