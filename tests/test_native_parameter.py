"""Tests for NativeParameter and the parameter-registration contract
(Advanced C++ v3.1, the first Phase C milestone).

NativeParameter is a NativeTensor subclass whose instances are always
graph-free owning leaves: construction copies the data (array-like, or
an existing NativeTensor's current value) into independent owning
contiguous native storage, requires_grad is a validated real bool
(default True), and parameter-ness never propagates — every operation
on a parameter returns a plain NativeTensor. NativeParameterRegistry is
the minimal insertion-ordered name -> parameter registry the future
NativeModule (v3.2) will embed: dot-free non-empty string names, only
NativeParameter values (None unregisters), replacement preserves
position, removal deletes the slot (re-registration appends), aliases
are visible by name, and unique traversal deduplicates by object
identity. See docs/backend_experiments.md and the module docstring of
src/tensorforge/experimental/native_parameter.py.

The leaf/graph assertions read the private _parents/_backward/
_graph_freed slots deliberately — they enforce the same graph contract
the Phase B tests established for NativeTensor. Native-backend tests
skip when the compiled library is not built.

Selector: python -m pytest -q -k "native_parameter or parameter_registration"
"""

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeParameter,
    NativeParameterRegistry,
    NativeTensor,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


VALUES = [[1.0, 2.0], [3.0, 4.0]]


# ======================================================================
# NativeParameter construction
# ======================================================================


@needs_native
def test_native_parameter_from_array_like_data():
    for data in (VALUES, np.asarray(VALUES), np.asarray(VALUES).T):
        p = NativeParameter(data)
        assert isinstance(p, NativeParameter)
        assert isinstance(p, NativeTensor)
        assert p.shape == np.asarray(data).shape
        assert p.contiguous is True
        assert p.owns_core is True
        assert np.array_equal(p.to_numpy(), np.asarray(data, dtype=np.float64))
        p.close()


@needs_native
def test_native_parameter_requires_grad_defaults_true():
    p = NativeParameter(VALUES)
    assert p.requires_grad is True
    p.close()


@needs_native
def test_native_parameter_requires_grad_false_builds_frozen_parameter():
    p = NativeParameter(VALUES, requires_grad=False)
    assert p.requires_grad is False
    assert isinstance(p, NativeParameter)  # frozen is still a parameter
    p.close()


@needs_native
def test_native_parameter_non_bool_requires_grad_raises():
    for bad in (1, 0, None, "True", 1.0, [True]):
        with pytest.raises(TypeError):
            NativeParameter(VALUES, requires_grad=bad)


@needs_native
def test_native_parameter_is_a_graph_free_leaf():
    p = NativeParameter(VALUES)
    assert p.is_leaf is True
    assert p._parents == ()
    assert p._backward is None
    assert p._graph_freed is False
    p.close()


@needs_native
def test_native_parameter_initial_grad_is_none():
    p = NativeParameter(VALUES)
    assert p.grad is None
    p.close()


@needs_native
def test_native_parameter_dtype_and_device():
    p = NativeParameter(VALUES)
    assert p.dtype == "float64"
    assert p.device == "cpu"
    p.close()


# ======================================================================
# Construction from a NativeTensor
# ======================================================================


@needs_native
def test_native_parameter_from_leaf_tensor_is_independent_owning_copy():
    source = NativeTensor.from_array(VALUES, requires_grad=True)
    p = NativeParameter(source)
    assert p.owns_core is True
    assert p.contiguous is True
    assert p._core is not source._core
    assert np.array_equal(p.to_numpy(), source.to_numpy())
    source.close()
    p.close()


@needs_native
def test_native_parameter_from_non_leaf_does_not_inherit_graph():
    a = NativeTensor.from_array(VALUES, requires_grad=True)
    b = NativeTensor.from_array(VALUES, requires_grad=True)
    c = a.add(b)
    assert c.is_leaf is False
    p = NativeParameter(c)
    assert p.is_leaf is True
    assert p._parents == ()
    assert p._backward is None
    assert p._graph_freed is False
    assert np.array_equal(p.to_numpy(), c.to_numpy())
    # The parameter is not part of the source graph: backward through it
    # reaches a and b, not p.
    c.sum().backward()
    assert p.grad is None
    assert a.grad is not None and b.grad is not None


@needs_native
def test_native_parameter_from_strided_offset_view_copies_the_value():
    source = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    view = source.narrow(1, 1, 2).transpose()  # non-contiguous, offset 1
    assert view.contiguous is False
    p = NativeParameter(view)
    assert p.contiguous is True
    assert p.owns_core is True
    assert np.array_equal(p.to_numpy(), source.to_numpy()[:, 1:3].T)
    # Independent storage: the borrowing view's owner can go away.
    view.close()
    source.close()
    assert np.array_equal(p.to_numpy(), np.array([[2.0, 5.0], [3.0, 6.0]]))
    p.close()


@needs_native
def test_native_parameter_survives_source_graph_cleanup():
    a = NativeTensor.from_array(VALUES, requires_grad=True)
    out = a.multiply(a)
    p = NativeParameter(out)
    out.sum().backward()  # one-shot: frees out's graph
    assert out._graph_freed is True
    assert p._graph_freed is False
    # The parameter still works in a fresh graph of its own.
    p.multiply(p).sum().backward()
    assert np.array_equal(
        p.grad.to_numpy(), 2.0 * np.asarray(VALUES) ** 2
    )


@needs_native
def test_native_parameter_closing_source_does_not_invalidate_parameter():
    source = NativeTensor.from_array(VALUES)
    p = NativeParameter(source)
    source.close()
    assert np.array_equal(p.to_numpy(), np.asarray(VALUES))
    p.close()


@needs_native
def test_native_parameter_closing_parameter_does_not_invalidate_source():
    source = NativeTensor.from_array(VALUES)
    p = NativeParameter(source)
    p.close()
    assert source.closed is False
    assert np.array_equal(source.to_numpy(), np.asarray(VALUES))
    source.close()


@needs_native
def test_native_parameter_from_closed_source_raises():
    source = NativeTensor.from_array(VALUES)
    source.close()
    with pytest.raises(RuntimeError):
        NativeParameter(source)


# ======================================================================
# Operation behavior — parameter-ness never propagates
# ======================================================================


@needs_native
def test_native_parameter_arithmetic_returns_plain_native_tensor():
    p = NativeParameter(VALUES)
    x = NativeTensor.from_array(VALUES)
    for op in ("add", "subtract", "multiply"):
        result = getattr(p, op)(x)
        assert type(result) is NativeTensor
        assert not isinstance(result, NativeParameter)
    result = p.relu()
    assert type(result) is NativeTensor


@needs_native
def test_native_parameter_matmul_returns_plain_native_tensor():
    p = NativeParameter(VALUES)
    x = NativeTensor.from_array(VALUES)
    for result in (p.matmul(x), x.matmul(p)):
        assert type(result) is NativeTensor
        assert not isinstance(result, NativeParameter)


@needs_native
def test_native_parameter_views_and_copies_return_plain_native_tensor():
    p = NativeParameter(VALUES)
    for result in (
        p.reshape((4,)),
        p.transpose(),
        p.T,
        p.narrow(0, 0, 1),
        p.contiguous_copy(),
        p.sum(),
        p.mean(axis=0),
    ):
        assert type(result) is NativeTensor
        assert not isinstance(result, NativeParameter)


@needs_native
def test_native_parameter_detach_returns_graph_free_native_tensor():
    p = NativeParameter(VALUES)
    d = p.detach()
    assert type(d) is NativeTensor
    assert not isinstance(d, NativeParameter)
    assert d.requires_grad is False
    assert d.is_leaf is True
    assert d._parents == ()
    assert d._backward is None
    assert np.array_equal(d.to_numpy(), p.to_numpy())


@needs_native
def test_native_parameter_participates_as_a_normal_graph_leaf():
    p = NativeParameter(VALUES)
    x = NativeTensor.from_array(VALUES)
    y = p.multiply(x)
    assert y.is_leaf is False
    assert y.requires_grad is True
    assert p in y._parents  # membership by identity via tuple.__contains__
    assert p.is_leaf is True  # the parameter itself never becomes non-leaf


@needs_native
def test_native_parameter_backward_accumulates_native_gradient():
    p = NativeParameter(VALUES)
    x = NativeTensor.from_array([[10.0, 20.0], [30.0, 40.0]])
    p.multiply(x).sum().backward()
    assert isinstance(p.grad, NativeTensor)
    assert not isinstance(p.grad, NativeParameter)
    assert np.array_equal(p.grad.to_numpy(), x.to_numpy())
    # A second fresh training-style graph keeps accumulating.
    p.sum().backward()
    assert np.array_equal(p.grad.to_numpy(), x.to_numpy() + 1.0)


@needs_native
def test_native_parameter_gradient_matches_shape_dtype_device():
    p = NativeParameter(VALUES)
    p.sum().backward()
    assert p.grad.shape == p.shape
    assert p.grad.dtype == p.dtype == "float64"
    assert p.grad.device == p.device == "cpu"


@needs_native
def test_native_parameter_zero_grad_clears_grad_and_preserves_data():
    p = NativeParameter(VALUES)
    p.sum().backward()
    assert p.grad is not None
    p.zero_grad()
    assert p.grad is None
    assert p.requires_grad is True
    assert np.array_equal(p.to_numpy(), np.asarray(VALUES))


@needs_native
def test_native_parameter_frozen_does_not_accumulate_grad():
    p = NativeParameter(VALUES, requires_grad=False)
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    p.add(x).sum().backward()
    assert p.grad is None
    assert np.array_equal(x.grad.to_numpy(), np.ones((2, 2)))


# ======================================================================
# Identity
# ======================================================================


@needs_native
def test_native_parameter_equal_values_stay_distinct_parameters():
    a = NativeParameter(VALUES)
    b = NativeParameter(VALUES)
    assert a is not b
    assert np.array_equal(a.to_numpy(), b.to_numpy())
    # Optimizer-style state keyed by identity keeps them apart.
    state = {id(a): "state_a", id(b): "state_b"}
    assert state[id(a)] == "state_a"
    assert state[id(b)] == "state_b"


@needs_native
def test_native_parameter_identity_stable_across_grad_lifecycle():
    p = NativeParameter(VALUES)
    key = id(p)
    p.sum().backward()
    assert id(p) == key
    p.zero_grad()
    assert id(p) == key


@needs_native
def test_native_parameter_defines_no_value_equality_or_hashing():
    # No numerical __eq__/__hash__ semantics anywhere in the hierarchy:
    # comparing parameters is plain object identity and never runs
    # tensor math, which future identity-keyed optimizer state relies on.
    for cls in (NativeParameter, NativeTensor):
        assert "__eq__" not in cls.__dict__
        assert "__hash__" not in cls.__dict__
    a = NativeParameter(VALUES)
    b = NativeParameter(VALUES)
    assert (a == b) is False
    assert (a == a) is True
    assert len({a, b}) == 2


# ======================================================================
# Registration
# ======================================================================


@needs_native
def test_parameter_registration_accepts_native_parameters():
    registry = NativeParameterRegistry()
    weight = NativeParameter(VALUES)
    bias = NativeParameter([0.0, 0.0])
    registry.register("weight", weight)
    registry.register("bias", bias)
    assert registry.named_parameters() == [("weight", weight), ("bias", bias)]


@needs_native
def test_parameter_registration_order_is_insertion_order():
    registry = NativeParameterRegistry()
    params = {name: NativeParameter([1.0]) for name in ("c", "a", "b")}
    for name, param in params.items():
        registry.register(name, param)
    assert [name for name, _ in registry.named_parameters()] == ["c", "a", "b"]
    assert registry.parameters() == [params["c"], params["a"], params["b"]]


@needs_native
def test_parameter_registration_invalid_name_types_raise():
    registry = NativeParameterRegistry()
    p = NativeParameter([1.0])
    for bad in (3, None, b"weight", ("weight",), 1.5):
        with pytest.raises(TypeError):
            registry.register(bad, p)


@needs_native
def test_parameter_registration_empty_name_raises():
    registry = NativeParameterRegistry()
    with pytest.raises(ValueError):
        registry.register("", NativeParameter([1.0]))


@needs_native
def test_parameter_registration_dotted_name_raises():
    registry = NativeParameterRegistry()
    for bad in ("layer.weight", ".", "w."):
        with pytest.raises(ValueError):
            registry.register(bad, NativeParameter([1.0]))


@needs_native
def test_parameter_registration_rejects_plain_native_tensor():
    registry = NativeParameterRegistry()
    t = NativeTensor.from_array(VALUES, requires_grad=True)
    with pytest.raises(TypeError):
        registry.register("weight", t)
    assert registry.named_parameters() == []
    t.close()


def test_parameter_registration_rejects_framework_tensor():
    import tensorforge

    registry = NativeParameterRegistry()
    with pytest.raises(TypeError):
        registry.register("weight", tensorforge.Tensor([1.0, 2.0]))


def test_parameter_registration_rejects_framework_parameter():
    import tensorforge

    registry = NativeParameterRegistry()
    with pytest.raises(TypeError):
        registry.register("weight", tensorforge.Parameter([1.0, 2.0]))


@needs_native
def test_parameter_registration_replacement_preserves_position():
    registry = NativeParameterRegistry()
    first = NativeParameter([1.0])
    second = NativeParameter([2.0])
    replacement = NativeParameter([9.0])
    registry.register("a", first)
    registry.register("b", second)
    registry.register("a", replacement)  # replaces, stays in slot 0
    assert registry.named_parameters() == [("a", replacement), ("b", second)]
    assert registry.named_parameters()[0][1] is replacement


@needs_native
def test_parameter_registration_replacement_leaves_old_parameter_alone():
    registry = NativeParameterRegistry()
    old = NativeParameter(VALUES)
    old.sum().backward()
    old_grad = old.grad
    registry.register("w", old)
    registry.register("w", NativeParameter(VALUES))
    assert old.closed is False
    assert old.grad is old_grad  # no state cleared or transferred
    assert np.array_equal(old.to_numpy(), np.asarray(VALUES))
    new = registry.named_parameters()[0][1]
    assert new.grad is None  # gradient state does not transfer


@needs_native
def test_parameter_registration_none_unregisters_and_reinsertion_appends():
    registry = NativeParameterRegistry()
    w, b = NativeParameter([1.0]), NativeParameter([2.0])
    registry.register("w", w)
    registry.register("b", b)
    registry.register("w", None)
    assert registry.named_parameters() == [("b", b)]
    registry.register("w", w)  # documented rule: re-registration appends
    assert registry.named_parameters() == [("b", b), ("w", w)]


@needs_native
def test_parameter_registration_removal_does_not_close_or_mutate():
    registry = NativeParameterRegistry()
    p = NativeParameter(VALUES)
    p.sum().backward()
    grad = p.grad
    registry.register("w", p)
    registry.register("w", None)
    assert p.closed is False
    assert p.grad is grad
    assert p.requires_grad is True
    assert np.array_equal(p.to_numpy(), np.asarray(VALUES))
    with pytest.raises(KeyError):
        registry.register("missing", None)


@needs_native
def test_parameter_registration_aliases_visible_by_name():
    registry = NativeParameterRegistry()
    shared = NativeParameter(VALUES)
    registry.register("encoder", shared)
    registry.register("decoder", shared)
    assert registry.named_parameters() == [
        ("encoder", shared),
        ("decoder", shared),
    ]


@needs_native
def test_parameter_registration_unique_traversal_deduplicates_by_identity():
    registry = NativeParameterRegistry()
    shared = NativeParameter(VALUES)
    other = NativeParameter(VALUES)
    registry.register("encoder", shared)
    registry.register("decoder", shared)
    registry.register("head", other)
    assert registry.parameters() == [shared, other]
    assert registry.parameters()[0] is shared


@needs_native
def test_parameter_registration_equal_values_are_not_deduplicated():
    registry = NativeParameterRegistry()
    a = NativeParameter(VALUES)
    b = NativeParameter(VALUES)  # equal values, distinct object
    registry.register("a", a)
    registry.register("b", b)
    assert registry.parameters() == [a, b]


@needs_native
def test_parameter_registration_unique_named_first_name_wins():
    registry = NativeParameterRegistry()
    shared = NativeParameter(VALUES)
    other = NativeParameter([5.0])
    registry.register("encoder", shared)
    registry.register("decoder", shared)
    registry.register("head", other)
    assert registry.unique_named_parameters() == [
        ("encoder", shared),
        ("head", other),
    ]


@needs_native
def test_parameter_registration_frozen_parameters_stay_discoverable():
    registry = NativeParameterRegistry()
    frozen = NativeParameter(VALUES, requires_grad=False)
    registry.register("frozen", frozen)
    assert registry.named_parameters() == [("frozen", frozen)]
    assert registry.parameters() == [frozen]
    assert frozen.requires_grad is False


@needs_native
def test_parameter_registration_never_touches_grad_or_requires_grad():
    registry = NativeParameterRegistry()
    p = NativeParameter(VALUES)
    p.sum().backward()
    grad = p.grad
    frozen = NativeParameter(VALUES, requires_grad=False)
    registry.register("p", p)
    registry.register("p2", p)
    registry.register("frozen", frozen)
    registry.register("p2", None)
    registry.named_parameters()
    registry.parameters()
    registry.unique_named_parameters()
    assert p.grad is grad
    assert p.requires_grad is True
    assert frozen.requires_grad is False
    assert frozen.grad is None


@needs_native
def test_parameter_registration_registry_does_not_close_parameters():
    p = NativeParameter(VALUES)
    registry = NativeParameterRegistry()
    registry.register("w", p)
    del registry
    assert p.closed is False
    assert np.array_equal(p.to_numpy(), np.asarray(VALUES))
    p.close()


# ======================================================================
# Integration and isolation
# ======================================================================


def test_native_parameter_separate_from_framework_parameter():
    import tensorforge

    assert not issubclass(NativeParameter, tensorforge.Parameter)
    assert not issubclass(tensorforge.Parameter, NativeParameter)
    assert not hasattr(tensorforge, "NativeParameter")
    assert not hasattr(tensorforge, "NativeParameterRegistry")
    # Construction rejects framework objects explicitly.
    with pytest.raises(TypeError):
        NativeParameter(tensorforge.Tensor([1.0, 2.0]))
    with pytest.raises(TypeError):
        NativeParameter(tensorforge.Parameter([1.0, 2.0]))


@needs_native
def test_native_parameter_causes_no_framework_tensor_dispatch():
    import tensorforge

    p = NativeParameter([1.0, 2.0])
    t = tensorforge.Tensor([1.0, 2.0], requires_grad=True)
    for op in ("add", "subtract", "multiply", "matmul"):
        with pytest.raises(TypeError):
            getattr(p, op)(t)
    # And the reverse: Tensor arithmetic does not silently absorb a
    # NativeParameter or route work through the native backend.
    with pytest.raises((TypeError, ValueError)):
        _ = t + p


@needs_native
def test_native_parameter_repr_is_metadata_only():
    p = NativeParameter(VALUES)
    assert repr(p) == "NativeParameter(shape=(2, 2), contiguous=True)"
    frozen = NativeParameter(VALUES, requires_grad=False)
    assert "requires_grad=False" in repr(frozen)
    p.close()
    assert repr(p) == "NativeParameter(closed)"


@needs_native
def test_native_parameter_close_is_idempotent_and_final():
    p = NativeParameter(VALUES)
    p.close()
    p.close()  # double close is safe, matching NativeTensor
    assert p.closed is True
    with pytest.raises(RuntimeError):
        p.to_numpy()
    with pytest.raises(RuntimeError):
        _ = p.requires_grad
