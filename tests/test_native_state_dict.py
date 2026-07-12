"""Tests for the native state dictionary contract (Advanced C++ v3.3,
the third Phase C milestone).

state_dict() returns an insertion-ordered {canonical_name: NativeTensor}
snapshot: keys are exactly the v3.2 canonical named_parameters() names
(dotted, direct-before-descendants, shared parameters once under their
first-discovered path, frozen included, cycle-safe), and every value is
an ordinary graph-free requires_grad=False NativeTensor owning an
independent contiguous native copy — sharing nothing with the model.
load_state_dict(state_dict, strict=True) copies values back INTO the
existing NativeParameter objects (identity, registration, aliases,
requires_grad, gradients, and training state all preserved; only the
numerical value changes), validating everything before mutating anything:
strict as a real bool, mapping input, string keys, missing/unexpected
keys (reported together under strict=True), NativeTensor values, open
source and destination, exact shape/dtype/device — then stages
independent native copies and commits by swapping cores, with rollback,
so no failure ever leaves the model partially updated. See
src/tensorforge/experimental/native_module.py and
docs/backend_experiments.md.

Tests constructing native tensors skip when the compiled backend is not
built; pure validation-shape tests run everywhere.

Selector: python -m pytest -q -k "native_state_dict or load_state_dict"
"""

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeModule,
    NativeParameter,
    NativeTensor,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


W_VALUES = [[1.0, 2.0], [3.0, 4.0]]
B_VALUES = [0.5, -0.5]
NEW_W = [[10.0, 20.0], [30.0, 40.0]]
NEW_B = [7.0, -7.0]


def _pair():
    """root(w) -> block(bias)."""
    root, block = NativeModule(), NativeModule()
    root.w = NativeParameter(W_VALUES)
    root.block = block
    block.bias = NativeParameter(B_VALUES)
    return root, block


def _new_state():
    return {
        "w": NativeTensor.from_array(NEW_W),
        "block.bias": NativeTensor.from_array(NEW_B),
    }


def _assert_unchanged(root):
    assert np.array_equal(root.w.to_numpy(), np.asarray(W_VALUES))
    assert np.array_equal(root.block.bias.to_numpy(), np.asarray(B_VALUES))


# ======================================================================
# state_dict keys and ordering
# ======================================================================


def test_native_state_dict_empty_module_returns_empty_ordered_mapping():
    state = NativeModule().state_dict()
    assert isinstance(state, dict)
    assert state == {}


@needs_native
def test_native_state_dict_uses_canonical_dotted_keys_in_order():
    root, _ = _pair()
    assert list(root.state_dict().keys()) == ["w", "block.bias"]
    # Deterministic across repeated calls.
    assert list(root.state_dict().keys()) == list(root.state_dict().keys())


@needs_native
def test_native_state_dict_includes_frozen_parameters():
    module = NativeModule()
    module.frozen = NativeParameter(W_VALUES, requires_grad=False)
    assert list(module.state_dict().keys()) == ["frozen"]


@needs_native
def test_native_state_dict_shared_parameter_appears_once():
    root, child = NativeModule(), NativeModule()
    shared = NativeParameter(W_VALUES)
    root.first = shared
    root.second = shared  # direct alias
    root.child = child
    child.nested = shared  # nested alias
    state = root.state_dict()
    assert list(state.keys()) == ["first"]  # first-discovered name wins


@needs_native
def test_native_state_dict_shared_child_module_parameters_appear_once():
    root, shared = NativeModule(), NativeModule()
    shared.w = NativeParameter(W_VALUES)
    root.left = shared
    root.right = shared
    assert list(root.state_dict().keys()) == ["left.w"]


@needs_native
def test_native_state_dict_cycles_terminate_safely():
    a, b = NativeModule(), NativeModule()
    a.w = NativeParameter(W_VALUES)
    a.child = b
    b.parent = a  # indirect cycle
    a.self_ref = a  # direct cycle
    assert list(a.state_dict().keys()) == ["w"]


@needs_native
def test_native_state_dict_reflects_dynamic_replacement_and_removal():
    root, _ = _pair()
    replacement = NativeParameter(NEW_W)
    root.w = replacement  # replace: same key, new value
    state = root.state_dict()
    assert list(state.keys()) == ["w", "block.bias"]
    assert np.array_equal(state["w"].to_numpy(), np.asarray(NEW_W))
    root.w = None  # remove: key disappears
    assert list(root.state_dict().keys()) == ["block.bias"]


# ======================================================================
# state_dict values and ownership
# ======================================================================


@needs_native
def test_native_state_dict_values_are_plain_graph_free_native_tensors():
    root, _ = _pair()
    for value in root.state_dict().values():
        assert type(value) is NativeTensor
        assert not isinstance(value, NativeParameter)
        assert value.requires_grad is False
        assert value.is_leaf is True
        assert value._parents == ()
        assert value._backward is None
        assert value._graph_freed is False


@needs_native
def test_native_state_dict_values_own_independent_contiguous_storage():
    root, _ = _pair()
    state = root.state_dict()
    for name, value in state.items():
        assert value.owns_core is True
        assert value.contiguous is True
    assert state["w"]._core is not root.w._core
    assert np.array_equal(state["w"].to_numpy(), root.w.to_numpy())


@needs_native
def test_native_state_dict_values_match_parameter_metadata():
    root, _ = _pair()
    for name, parameter in root.named_parameters():
        value = root.state_dict()[name]
        assert value.shape == parameter.shape
        assert value.dtype == parameter.dtype == "float64"
        assert value.device == parameter.device == "cpu"


@needs_native
def test_native_state_dict_snapshot_survives_model_changes():
    root, _ = _pair()
    snapshot = root.state_dict()
    root.load_state_dict(_new_state())  # modify the parameter values
    root.w = NativeParameter(W_VALUES)  # replace the registration too
    assert np.array_equal(snapshot["w"].to_numpy(), np.asarray(W_VALUES))
    assert np.array_equal(
        snapshot["block.bias"].to_numpy(), np.asarray(B_VALUES)
    )


@needs_native
def test_native_state_dict_snapshot_survives_parameter_close():
    module = NativeModule()
    module.w = NativeParameter(W_VALUES)
    snapshot = module.state_dict()
    module.w.close()
    assert np.array_equal(snapshot["w"].to_numpy(), np.asarray(W_VALUES))


@needs_native
def test_native_state_dict_closing_snapshot_leaves_model_intact():
    root, _ = _pair()
    snapshot = root.state_dict()
    for value in snapshot.values():
        value.close()
    _assert_unchanged(root)
    root.w.sum().backward()  # still fully usable
    assert root.w.grad is not None


@needs_native
def test_native_state_dict_closed_parameter_fails_clearly():
    root, _ = _pair()
    root.block.bias.close()
    with pytest.raises(RuntimeError, match="block.bias"):
        root.state_dict()


@needs_native
def test_native_state_dict_does_not_touch_model_state():
    root, _ = _pair()
    root.w.sum().backward()
    grad = root.w.grad
    root.eval()
    root.state_dict()
    assert root.w.grad is grad
    assert root.w.requires_grad is True
    assert root.training is False
    assert [name for name, _ in root.named_parameters()] == ["w", "block.bias"]


# ======================================================================
# strict loading
# ======================================================================


@needs_native
def test_load_state_dict_roundtrip_succeeds():
    root, _ = _pair()
    result = root.load_state_dict(root.state_dict())
    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
    _assert_unchanged(root)


@needs_native
def test_load_state_dict_loads_new_values():
    root, _ = _pair()
    result = root.load_state_dict(_new_state(), strict=True)
    assert result == ((), ())
    assert np.array_equal(root.w.to_numpy(), np.asarray(NEW_W))
    assert np.array_equal(root.block.bias.to_numpy(), np.asarray(NEW_B))


@needs_native
def test_load_state_dict_strict_defaults_to_true():
    root, _ = _pair()
    state = root.state_dict()
    del state["w"]
    with pytest.raises(ValueError):
        root.load_state_dict(state)  # no strict argument
    _assert_unchanged(root)


def test_load_state_dict_non_bool_strict_raises_before_anything():
    module = NativeModule()
    for bad in (1, 0, "true", None, 1.0):
        with pytest.raises(TypeError):
            module.load_state_dict({}, strict=bad)


def test_load_state_dict_non_mapping_input_raises():
    module = NativeModule()
    for bad in ([("w", 1)], None, 42, "state"):
        with pytest.raises(TypeError):
            module.load_state_dict(bad)


@needs_native
def test_load_state_dict_missing_keys_raise_before_mutation():
    root, _ = _pair()
    state = _new_state()
    del state["block.bias"]
    with pytest.raises(ValueError, match="block.bias"):
        root.load_state_dict(state)
    _assert_unchanged(root)  # the valid "w" value was NOT loaded


@needs_native
def test_load_state_dict_unexpected_keys_raise_before_mutation():
    root, _ = _pair()
    state = _new_state()
    state["extra"] = NativeTensor.from_array([1.0])
    with pytest.raises(ValueError, match="extra"):
        root.load_state_dict(state)
    _assert_unchanged(root)


@needs_native
def test_load_state_dict_reports_missing_and_unexpected_together():
    root, _ = _pair()
    state = {"w": NativeTensor.from_array(NEW_W),
             "extra": NativeTensor.from_array([1.0])}
    with pytest.raises(ValueError) as excinfo:
        root.load_state_dict(state)
    message = str(excinfo.value)
    assert "block.bias" in message  # missing
    assert "extra" in message  # unexpected
    _assert_unchanged(root)


@needs_native
def test_load_state_dict_non_string_keys_fail_before_mutation():
    root, _ = _pair()
    state = dict(_new_state())
    state[3] = NativeTensor.from_array([1.0])
    with pytest.raises(TypeError):
        root.load_state_dict(state, strict=False)
    _assert_unchanged(root)


@needs_native
def test_load_state_dict_rejects_non_native_tensor_values():
    root, _ = _pair()
    for bad in (NEW_W, np.asarray(NEW_W), 3.0):
        with pytest.raises(TypeError, match="'w'"):
            root.load_state_dict(
                {"w": bad, "block.bias": NativeTensor.from_array(NEW_B)}
            )
        _assert_unchanged(root)


@needs_native
def test_load_state_dict_rejects_framework_tensor_and_parameter():
    import tensorforge

    root, _ = _pair()
    for bad in (
        tensorforge.Tensor(np.asarray(W_VALUES)),
        tensorforge.Parameter(np.asarray(W_VALUES)),
    ):
        with pytest.raises(TypeError, match="'w'"):
            root.load_state_dict(
                {"w": bad, "block.bias": NativeTensor.from_array(NEW_B)}
            )
        _assert_unchanged(root)


# ======================================================================
# non-strict loading
# ======================================================================


@needs_native
def test_load_state_dict_non_strict_loads_matching_and_reports_keys():
    root, _ = _pair()
    result = root.load_state_dict(
        {"w": NativeTensor.from_array(NEW_W)}, strict=False
    )
    assert np.array_equal(root.w.to_numpy(), np.asarray(NEW_W))
    # The missing parameter keeps its value.
    assert np.array_equal(root.block.bias.to_numpy(), np.asarray(B_VALUES))
    assert result.missing_keys == ("block.bias",)
    assert result.unexpected_keys == ()


@needs_native
def test_load_state_dict_non_strict_ignores_unexpected_keys():
    root, _ = _pair()
    extra_one = NativeTensor.from_array([1.0])
    extra_two = NativeTensor.from_array([2.0])
    state = _new_state()
    state["zeta"] = extra_one
    state["alpha"] = extra_two
    result = root.load_state_dict(state, strict=False)
    assert np.array_equal(root.w.to_numpy(), np.asarray(NEW_W))
    assert result.missing_keys == ()
    # Deterministic: unexpected keys keep the input mapping's order.
    assert result.unexpected_keys == ("zeta", "alpha")
    assert extra_one.closed is False  # ignored, never touched


@needs_native
def test_load_state_dict_non_strict_missing_keys_are_canonical_order():
    root, _ = _pair()
    result = root.load_state_dict({}, strict=False)
    assert result.missing_keys == ("w", "block.bias")
    assert result.unexpected_keys == ()
    _assert_unchanged(root)


@needs_native
def test_load_state_dict_non_strict_invalid_matching_value_is_total_failure():
    root, _ = _pair()
    state = {
        "w": NativeTensor.from_array(NEW_W),  # valid
        "block.bias": NativeTensor.from_array([[1.0, 2.0]]),  # wrong shape
    }
    with pytest.raises(ValueError, match="block.bias"):
        root.load_state_dict(state, strict=False)
    _assert_unchanged(root)  # nothing partially loaded


# ======================================================================
# validation
# ======================================================================


@needs_native
def test_load_state_dict_closed_source_fails_naming_key():
    root, _ = _pair()
    state = _new_state()
    state["block.bias"].close()
    with pytest.raises(RuntimeError, match="block.bias"):
        root.load_state_dict(state)
    _assert_unchanged(root)


@needs_native
def test_load_state_dict_closed_destination_fails_naming_key():
    root, _ = _pair()
    state = _new_state()
    root.block.bias.close()
    with pytest.raises(RuntimeError, match="block.bias"):
        root.load_state_dict(state)
    assert np.array_equal(root.w.to_numpy(), np.asarray(W_VALUES))


@needs_native
def test_load_state_dict_shape_mismatch_fails_naming_key_and_shapes():
    root, _ = _pair()
    state = _new_state()
    state["w"] = NativeTensor.from_array([1.0, 2.0, 3.0, 4.0])  # no reshape
    with pytest.raises(ValueError) as excinfo:
        root.load_state_dict(state)
    message = str(excinfo.value)
    assert "'w'" in message
    assert "(2, 2)" in message and "(4,)" in message
    _assert_unchanged(root)


@needs_native
def test_load_state_dict_no_broadcasting_occurs():
    root, _ = _pair()
    state = _new_state()
    state["w"] = NativeTensor.from_array([1.0, 2.0])  # broadcastable to (2,2)
    with pytest.raises(ValueError, match="'w'"):
        root.load_state_dict(state)
    _assert_unchanged(root)


@needs_native
def test_load_state_dict_copies_input_values_rather_than_aliasing():
    root, _ = _pair()
    state = _new_state()
    root.load_state_dict(state)
    assert root.w._core is not state["w"]._core
    for value in state.values():
        value.close()  # closing the inputs must not affect the model
    assert np.array_equal(root.w.to_numpy(), np.asarray(NEW_W))
    root.w.sum().backward()
    assert root.w.grad is not None


@needs_native
def test_load_state_dict_accepts_native_parameter_value_by_copy():
    root, _ = _pair()
    source_w = NativeParameter(NEW_W)
    source_b = NativeParameter(NEW_B, requires_grad=False)
    root.load_state_dict({"w": source_w, "block.bias": source_b})
    assert np.array_equal(root.w.to_numpy(), np.asarray(NEW_W))
    # Copied, not adopted: identity, graph state, and flags of the
    # destination come from the destination, not the source.
    assert root.w is not source_w
    assert root.w._core is not source_w._core
    assert root.w.requires_grad is True
    assert root.block.bias.requires_grad is True  # source frozen; dest not
    assert root.w.is_leaf is True


@needs_native
def test_load_state_dict_strided_view_value_loads_at_logical_shape():
    root, _ = _pair()
    base = NativeTensor.from_array(np.asarray(NEW_W).T)
    state = {"w": base.T, "block.bias": NativeTensor.from_array(NEW_B)}
    root.load_state_dict(state)  # non-contiguous source copies fine
    assert np.array_equal(root.w.to_numpy(), np.asarray(NEW_W))
    assert root.w.contiguous is True


# ======================================================================
# identity and state preservation
# ======================================================================


@needs_native
def test_load_state_dict_preserves_parameter_identity_and_registration():
    root, _ = _pair()
    w, bias = root.w, root.block.bias
    root.load_state_dict(_new_state())
    assert root.w is w  # same Python objects, values changed in place
    assert root.block.bias is bias
    assert [name for name, _ in root.named_parameters()] == ["w", "block.bias"]
    assert root.parameters() == [w, bias]


@needs_native
def test_load_state_dict_preserves_shared_aliases():
    root, child = NativeModule(), NativeModule()
    shared = NativeParameter(W_VALUES)
    root.first = shared
    root.child = child
    child.nested = shared
    root.load_state_dict({"first": NativeTensor.from_array(NEW_W)})
    assert root.first is shared
    assert root.child.nested is shared  # still the same shared object
    # Every alias observes the newly loaded value.
    assert np.array_equal(root.child.nested.to_numpy(), np.asarray(NEW_W))


@needs_native
def test_load_state_dict_duplicate_alias_key_is_unexpected():
    root = NativeModule()
    shared = NativeParameter(W_VALUES)
    root.a = shared
    root.b = shared  # alias; canonical key is "a"
    state = {
        "a": NativeTensor.from_array(NEW_W),
        "b": NativeTensor.from_array(NEW_W),
    }
    with pytest.raises(ValueError, match="'b'"):
        root.load_state_dict(state, strict=True)
    result = root.load_state_dict(state, strict=False)
    assert result.unexpected_keys == ("b",)
    assert np.array_equal(shared.to_numpy(), np.asarray(NEW_W))


@needs_native
def test_load_state_dict_preserves_requires_grad_and_frozen_state():
    module = NativeModule()
    module.w = NativeParameter(W_VALUES)
    module.frozen = NativeParameter(B_VALUES, requires_grad=False)
    module.load_state_dict({
        "w": NativeTensor.from_array(NEW_W),
        "frozen": NativeTensor.from_array(NEW_B),
    })
    assert module.w.requires_grad is True
    assert module.frozen.requires_grad is False  # frozen stays frozen
    assert np.array_equal(module.frozen.to_numpy(), np.asarray(NEW_B))


@needs_native
def test_load_state_dict_parameter_stays_graph_free_leaf():
    root, _ = _pair()
    root.load_state_dict(_new_state())
    for parameter in root.parameters():
        assert parameter.is_leaf is True
        assert parameter._parents == ()
        assert parameter._backward is None
        assert parameter._graph_freed is False


@needs_native
def test_load_state_dict_preserves_gradients_exactly():
    root, _ = _pair()
    root.w.sum().backward()
    grad = root.w.grad
    grad_values = grad.to_numpy()
    assert root.block.bias.grad is None
    root.load_state_dict(_new_state())
    assert root.w.grad is grad  # same object
    assert np.array_equal(grad.to_numpy(), grad_values)  # same value
    assert root.block.bias.grad is None  # None stays None


@needs_native
def test_load_state_dict_preserves_training_flags():
    root, _ = _pair()
    root.eval()
    root.load_state_dict(_new_state())
    assert root.training is False
    assert root.block.training is False
    root.train()
    root.load_state_dict(_new_state())
    assert root.training is True


@needs_native
def test_load_state_dict_repeated_loads_preserve_identity():
    root, _ = _pair()
    w = root.w
    for values in (NEW_W, W_VALUES, NEW_W):
        root.load_state_dict({
            "w": NativeTensor.from_array(values),
            "block.bias": NativeTensor.from_array(NEW_B),
        })
        assert root.w is w
    assert np.array_equal(w.to_numpy(), np.asarray(NEW_W))


@needs_native
def test_load_state_dict_shared_parameter_grad_unchanged():
    root = NativeModule()
    shared = NativeParameter(W_VALUES)
    root.a = shared
    root.b = shared
    shared.sum().backward()
    grad = shared.grad
    root.load_state_dict({"a": NativeTensor.from_array(NEW_W)})
    assert shared.grad is grad
    assert np.array_equal(shared.to_numpy(), np.asarray(NEW_W))


# ======================================================================
# atomicity
# ======================================================================


@needs_native
def test_load_state_dict_later_shape_mismatch_leaves_earlier_unchanged():
    root, _ = _pair()
    state = {
        "w": NativeTensor.from_array(NEW_W),  # would load fine
        "block.bias": NativeTensor.from_array([[9.0]]),  # fails preflight
    }
    with pytest.raises(ValueError):
        root.load_state_dict(state)
    _assert_unchanged(root)


@needs_native
def test_load_state_dict_later_closed_source_leaves_earlier_unchanged():
    root, _ = _pair()
    state = _new_state()
    state["block.bias"].close()  # "w" is earlier in canonical order
    with pytest.raises(RuntimeError):
        root.load_state_dict(state)
    _assert_unchanged(root)


@needs_native
def test_load_state_dict_staging_failure_leaves_all_unchanged(monkeypatch):
    from tensorforge.experimental import native_module

    root, _ = _pair()
    real_copy = native_module._native_copy
    calls = {"n": 0}

    def failing_copy(core):
        calls["n"] += 1
        if calls["n"] == 2:
            raise MemoryError("forced staging failure")
        return real_copy(core)

    monkeypatch.setattr(native_module, "_native_copy", failing_copy)
    state = _new_state()
    with pytest.raises(MemoryError):
        root.load_state_dict(state)
    monkeypatch.undo()
    _assert_unchanged(root)
    # The inputs were not closed by the failure, and a retry works.
    root.load_state_dict(state)
    assert np.array_equal(root.w.to_numpy(), np.asarray(NEW_W))


@needs_native
def test_load_state_dict_commit_failure_rolls_back(monkeypatch):
    root, _ = _pair()
    real_adopt = NativeParameter._adopt_value_core
    calls = {"n": 0}

    def failing_adopt(self, new_core):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("forced commit failure")
        return real_adopt(self, new_core)

    monkeypatch.setattr(NativeParameter, "_adopt_value_core", failing_adopt)
    state = _new_state()
    with pytest.raises(RuntimeError, match="forced commit failure"):
        root.load_state_dict(state)
    monkeypatch.undo()
    # Rollback restored the original cores: values and identity intact.
    _assert_unchanged(root)
    root.w.sum().backward()  # original storage is alive and usable
    assert root.w.grad is not None
    # And a clean retry succeeds.
    root.load_state_dict(state)
    assert np.array_equal(root.w.to_numpy(), np.asarray(NEW_W))


@needs_native
def test_load_state_dict_failure_preserves_snapshots_and_inputs():
    root, _ = _pair()
    snapshot = root.state_dict()
    good = NativeTensor.from_array(NEW_W)
    with pytest.raises(ValueError):
        root.load_state_dict({"w": good})  # strict: missing block.bias
    assert good.closed is False
    assert np.array_equal(snapshot["w"].to_numpy(), np.asarray(W_VALUES))
    root.w.sum().backward()
    assert root.w.grad is not None  # gradients untouched by the failure


# ======================================================================
# integration and isolation
# ======================================================================


@needs_native
def test_load_state_dict_forward_after_loading_uses_new_values():
    root, _ = _pair()
    root.load_state_dict(_new_state())
    result = root.w.multiply(root.w).sum()
    expected = float(np.sum(np.asarray(NEW_W) ** 2))
    assert result.to_numpy().item() == expected
    result.backward()
    assert np.array_equal(root.w.grad.to_numpy(), 2.0 * np.asarray(NEW_W))


@needs_native
def test_load_state_dict_graph_built_before_loading_stays_memory_safe():
    # The documented in-place policy, tightened by v3.7: a graph built
    # before loading stays memory-safe (no use-after-close — backward
    # never reads the released old storage), and where its backward
    # must read the loaded parameter's forward value it raises a
    # deterministic stale-value error instead of silently computing
    # gradients against the new value. A fresh forward works normally.
    root, _ = _pair()
    loss = root.w.multiply(root.w).sum()  # graph over the OLD values
    root.load_state_dict(_new_state())
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward()
    assert root.w.grad is None  # nothing was committed
    fresh = root.w.multiply(root.w).sum()  # graph over the NEW values
    fresh.backward()
    assert np.array_equal(root.w.grad.to_numpy(), 2.0 * np.asarray(NEW_W))


def test_native_state_dict_framework_state_dict_is_untouched():
    import tensorforge

    layer = tensorforge.nn.Linear(2, 3)
    state = layer.state_dict()
    assert set(state) == {"weight", "bias"}
    assert all(isinstance(value, np.ndarray) for value in state.values())
    result = layer.load_state_dict(state)
    assert result == {"missing_keys": [], "unexpected_keys": []}
