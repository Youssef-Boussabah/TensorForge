"""Tests for NativeModule — module hierarchy core and recursive
registration (Advanced C++ v3.2, the second Phase C milestone).

NativeModule is a Python-side organizational abstraction over the v3.1
parameter contract: assigning a NativeParameter registers it, assigning
a NativeModule registers a child, everything else is an ordinary
attribute (nothing is wrapped implicitly, and stable-framework objects
never enter native traversal). One category per name — the latest
assignment wins, replacement preserves position within a registry,
moving categories appends, None unregisters. Traversal is deterministic
pre-order depth-first, deduplicated by object identity (first-discovered
dotted name wins — the future state_dict key), which also makes shared
modules and reference cycles terminate safely. zero_grad() delegates to
each unique parameter; train()/eval() propagate a validated bool
training flag; forward() raises NotImplementedError and __call__
delegates to it. The module owns no storage and never closes or mutates
what it registers. See src/tensorforge/experimental/native_module.py
and docs/backend_experiments.md.

Tests that construct native tensors/parameters skip when the compiled
backend is not built; the pure-hierarchy tests run everywhere.

Selector: python -m pytest -q -k "native_module or recursive_registration"
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


VALUES = [[1.0, 2.0], [3.0, 4.0]]


# ======================================================================
# Initialization and call protocol
# ======================================================================


def test_native_module_starts_in_training_mode():
    assert NativeModule().training is True


def test_native_module_registries_start_empty():
    module = NativeModule()
    assert module.parameters() == []
    assert list(module.named_parameters()) == []
    assert module.modules() == [module]
    assert list(module.named_modules()) == [("", module)]


def test_native_module_forward_raises_not_implemented():
    module = NativeModule()
    with pytest.raises(NotImplementedError):
        module.forward()
    with pytest.raises(NotImplementedError):
        module()  # __call__ delegates to forward


def test_native_module_call_delegates_to_forward():
    class Doubler(NativeModule):
        def forward(self, value, scale=2):
            return value * scale

    module = Doubler()
    assert module(3) == 6
    assert module(3, scale=5) == 15


def test_native_module_registering_before_init_raises():
    class Broken(NativeModule):
        def __init__(self):
            self.weight = "placeholder"  # ordinary attr: fine pre-init
            self.child = NativeModule()  # registration: must fail clearly

    with pytest.raises(RuntimeError, match="super\\(\\).__init__"):
        Broken()


# ======================================================================
# Automatic parameter registration
# ======================================================================


@needs_native
def test_native_module_assigning_parameter_registers_it():
    module = NativeModule()
    weight = NativeParameter(VALUES)
    module.weight = weight
    assert module.weight is weight  # exact object identity via attribute
    assert module.parameters() == [weight]
    assert module.parameters()[0] is weight
    assert list(module.named_parameters()) == [("weight", weight)]


@needs_native
def test_native_module_frozen_parameter_is_discoverable():
    module = NativeModule()
    frozen = NativeParameter(VALUES, requires_grad=False)
    module.frozen = frozen
    assert module.parameters() == [frozen]
    assert frozen.requires_grad is False


@needs_native
def test_native_module_plain_native_tensor_stays_ordinary_attribute():
    module = NativeModule()
    tensor = NativeTensor.from_array(VALUES, requires_grad=True)
    module.cache = tensor
    assert module.cache is tensor
    assert module.parameters() == []
    assert list(module.named_parameters()) == []
    tensor.close()


def test_native_module_framework_objects_stay_out_of_registries():
    import tensorforge

    module = NativeModule()
    module.tensor = tensorforge.Tensor([1.0, 2.0])
    module.parameter = tensorforge.Parameter([1.0, 2.0])
    module.layer = tensorforge.nn.Linear(2, 2)
    # Ordinary attributes, readable but never in native traversal.
    assert isinstance(module.tensor, tensorforge.Tensor)
    assert isinstance(module.parameter, tensorforge.Parameter)
    assert module.parameters() == []
    assert module.modules() == [module]


def test_native_module_ordinary_values_stay_ordinary():
    module = NativeModule()
    module.label = "encoder"
    module.count = 3
    assert module.label == "encoder"
    assert module.count == 3
    assert module.parameters() == []
    assert module.modules() == [module]


# ======================================================================
# Automatic child-module registration
# ======================================================================


def test_native_module_assigning_child_registers_it():
    root = NativeModule()
    child = NativeModule()
    root.block = child
    assert root.block is child
    assert root.modules() == [root, child]
    assert list(root.named_modules()) == [("", root), ("block", child)]


def test_native_module_missing_attribute_raises_attribute_error():
    module = NativeModule()
    with pytest.raises(AttributeError):
        _ = module.missing


# ======================================================================
# Name validation and reserved names
# ======================================================================


@needs_native
def test_native_module_manual_registration_invalid_names_fail():
    module = NativeModule()
    p = NativeParameter(VALUES)
    child = NativeModule()
    with pytest.raises(ValueError):
        module.register_parameter("", p)
    with pytest.raises(ValueError):
        module.register_parameter("a.b", p)
    with pytest.raises(TypeError):
        module.register_parameter(3, p)
    with pytest.raises(ValueError):
        module.add_module("", child)
    with pytest.raises(ValueError):
        module.add_module("a.b", child)
    with pytest.raises(TypeError):
        module.add_module(3, child)
    # A failed registration mutates nothing.
    assert module.parameters() == []
    assert module.modules() == [module]


@needs_native
def test_native_module_reserved_names_are_rejected():
    module = NativeModule()
    # ``_buffers`` (v3.15) and ``_generators`` (Phase G, G1) joined the
    # reserved set as their registries were added; the rule is the same.
    for name in ("_parameters", "_modules", "_buffers", "_generators",
                 "training"):
        with pytest.raises(ValueError):
            setattr(module, name, NativeParameter(VALUES))
        with pytest.raises(ValueError):
            setattr(module, name, NativeModule())
    # The internals survived the rejected assignments.
    assert module.training is True
    assert module.parameters() == []
    assert module.modules() == [module]
    assert module.buffers() == []
    assert module.generators() == []


def test_native_module_registries_are_independent_categories():
    """Adding a fourth registration category (Phase G's generators) must
    not disturb the three that were already there: a module with no
    generators behaves exactly as before, and the registries never leak
    into one another."""
    module = NativeModule()
    assert module.parameters() == []
    assert module.buffers() == []
    assert module.generators() == []
    assert module.modules() == [module]
    assert module.state_dict() == {}
    assert module.generator_state_dict() == {}
    child = NativeModule()
    module.block = child
    assert module.modules() == [module, child]
    assert module.generators() == []


@needs_native
def test_native_module_manual_registration_matches_assignment():
    via_assign, via_api = NativeModule(), NativeModule()
    p = NativeParameter(VALUES)
    child = NativeModule()
    via_assign.w = p
    via_assign.block = child
    via_api.register_parameter("w", p)
    via_api.add_module("block", child)
    assert list(via_assign.named_parameters()) == list(via_api.named_parameters())
    assert via_api.w is p
    assert via_api.block is child
    # The explicit APIs are strict about wrong values (assignment would
    # store an ordinary attribute instead).
    t = NativeTensor.from_array(VALUES)
    with pytest.raises(TypeError):
        via_api.register_parameter("bad", t)
    with pytest.raises(TypeError):
        via_api.register_parameter("bad", child)
    with pytest.raises(TypeError):
        via_api.add_module("bad", p)
    t.close()


def test_native_module_manual_none_on_absent_name_raises_key_error():
    module = NativeModule()
    with pytest.raises(KeyError):
        module.register_parameter("missing", None)
    with pytest.raises(KeyError):
        module.add_module("missing", None)


# ======================================================================
# Replacement, removal, and collisions
# ======================================================================


@needs_native
def test_native_module_parameter_replacement_updates_and_preserves_position():
    module = NativeModule()
    first = NativeParameter([1.0])
    second = NativeParameter([2.0])
    replacement = NativeParameter([9.0])
    module.a = first
    module.b = second
    module.a = replacement  # same-registry replacement keeps slot 0
    assert list(module.named_parameters()) == [
        ("a", replacement),
        ("b", second),
    ]
    assert module.a is replacement


@needs_native
def test_native_module_replacement_leaves_old_parameter_alone():
    module = NativeModule()
    old = NativeParameter(VALUES)
    old.sum().backward()
    grad = old.grad
    module.w = old
    module.w = NativeParameter(VALUES)
    assert old.closed is False
    assert old.grad is grad
    assert np.array_equal(old.to_numpy(), np.asarray(VALUES))
    assert module.w.grad is None  # no gradient state transfers


def test_native_module_child_replacement_updates_registration():
    root = NativeModule()
    first, second, replacement = NativeModule(), NativeModule(), NativeModule()
    root.a = first
    root.b = second
    root.a = replacement
    assert list(root.named_modules()) == [
        ("", root), ("a", replacement), ("b", second),
    ]
    assert first.training is True  # untouched


@needs_native
def test_native_module_parameter_to_child_collision_resolves():
    module = NativeModule()
    p = NativeParameter(VALUES)
    child = NativeModule()
    module.slot = p
    module.slot = child  # latest assignment wins; category moves
    assert module.slot is child
    assert module.parameters() == []
    assert module.modules() == [module, child]
    assert p.closed is False  # evicted, not closed


@needs_native
def test_native_module_child_to_parameter_collision_resolves():
    module = NativeModule()
    child = NativeModule()
    p = NativeParameter(VALUES)
    module.slot = child
    module.slot = p
    assert module.slot is p
    assert module.modules() == [module]
    assert module.parameters() == [p]


@needs_native
def test_native_module_ordinary_value_unregisters_previous_registration():
    module = NativeModule()
    p = NativeParameter(VALUES)
    child = NativeModule()
    module.w = p
    module.block = child
    module.w = "label"
    module.block = 42
    assert module.w == "label"
    assert module.block == 42
    assert module.parameters() == []
    assert module.modules() == [module]
    assert p.closed is False


@needs_native
def test_native_module_assigning_none_unregisters_parameter():
    module = NativeModule()
    p = NativeParameter(VALUES)
    module.w = p
    module.w = None
    assert module.w is None  # readable as None, no stale entry
    assert module.parameters() == []
    assert p.closed is False
    assert np.array_equal(p.to_numpy(), np.asarray(VALUES))  # still usable


def test_native_module_assigning_none_unregisters_child():
    root = NativeModule()
    child = NativeModule()
    root.block = child
    root.block = None
    assert root.block is None
    assert root.modules() == [root]
    assert child.training is True  # untouched and usable
    assert child.modules() == [child]


@needs_native
def test_native_module_removed_name_reregisters_at_the_end():
    module = NativeModule()
    a, b = NativeParameter([1.0]), NativeParameter([2.0])
    module.a = a
    module.b = b
    module.a = None  # removes the slot entirely
    module.a = a  # documented rule: re-registration appends
    assert [name for name, _ in module.named_parameters()] == ["b", "a"]


@needs_native
def test_native_module_delattr_unregisters_without_closing():
    module = NativeModule()
    p = NativeParameter(VALUES)
    child = NativeModule()
    module.w = p
    module.block = child
    del module.w
    del module.block
    assert module.parameters() == []
    assert module.modules() == [module]
    with pytest.raises(AttributeError):
        _ = module.w
    assert p.closed is False


# ======================================================================
# Recursive parameter traversal
# ======================================================================


def _tree():
    """root(weight) -> block(bias) -> sub(w); plus root.tail(t)."""
    root, block, sub, tail = (NativeModule() for _ in range(4))
    root.weight = NativeParameter([[1.0, 2.0]])
    root.block = block
    block.bias = NativeParameter([0.5])
    block.sub = sub
    sub.w = NativeParameter([[3.0]])
    root.tail = tail
    tail.t = NativeParameter([4.0])
    return root, block, sub, tail


@needs_native
def test_recursive_registration_named_parameters_use_dotted_names():
    root, block, sub, tail = _tree()
    names = [name for name, _ in root.named_parameters()]
    assert names == ["weight", "block.bias", "block.sub.w", "tail.t"]


@needs_native
def test_recursive_registration_direct_parameters_precede_children():
    root, block, sub, tail = _tree()
    pairs = list(root.named_parameters())
    assert pairs[0] == ("weight", root.weight)
    assert pairs[1][1] is block.bias
    # And at the child level too: block's own bias precedes sub's w.
    child_names = [name for name, _ in block.named_parameters()]
    assert child_names == ["bias", "sub.w"]


@needs_native
def test_recursive_registration_parameters_matches_named_order():
    root, block, sub, tail = _tree()
    assert root.parameters() == [
        root.weight, block.bias, sub.w, tail.t
    ]


@needs_native
def test_recursive_registration_recurse_false_returns_direct_only():
    root, block, sub, tail = _tree()
    assert list(root.named_parameters(recurse=False)) == [
        ("weight", root.weight)
    ]
    assert root.parameters(recurse=False) == [root.weight]


@needs_native
def test_recursive_registration_equal_valued_parameters_both_appear():
    module = NativeModule()
    a = NativeParameter(VALUES)
    b = NativeParameter(VALUES)  # equal values, distinct identity
    module.a = a
    module.b = b
    assert module.parameters() == [a, b]


@needs_native
def test_recursive_registration_direct_alias_appears_once():
    module = NativeModule()
    shared = NativeParameter(VALUES)
    module.first = shared
    module.second = shared
    assert list(module.named_parameters()) == [("first", shared)]
    assert module.parameters() == [shared]


@needs_native
def test_recursive_registration_shared_direct_and_nested_appears_once():
    root, child = NativeModule(), NativeModule()
    shared = NativeParameter(VALUES)
    root.shared = shared
    root.child = child
    child.also_shared = shared
    # First discovery (the root's direct name) wins.
    assert list(root.named_parameters()) == [("shared", shared)]
    assert root.parameters() == [shared]


@needs_native
def test_recursive_registration_frozen_nested_parameters_appear():
    root, child = NativeModule(), NativeModule()
    frozen = NativeParameter(VALUES, requires_grad=False)
    root.child = child
    child.frozen = frozen
    assert list(root.named_parameters()) == [("child.frozen", frozen)]


@needs_native
def test_recursive_registration_traversal_is_deterministic():
    root, block, sub, tail = _tree()
    assert list(root.named_parameters()) == list(root.named_parameters())
    assert root.modules() == root.modules()
    assert list(root.named_modules()) == list(root.named_modules())


# ======================================================================
# Recursive module traversal
# ======================================================================


def test_recursive_registration_named_modules_is_depth_first():
    root, block, sub, tail = (NativeModule() for _ in range(4))
    root.block = block
    block.sub = sub
    root.tail = tail
    assert list(root.named_modules()) == [
        ("", root),
        ("block", block),
        ("block.sub", sub),
        ("tail", tail),
    ]
    assert root.modules() == [root, block, sub, tail]


def test_recursive_registration_shared_child_module_appears_once():
    root, shared = NativeModule(), NativeModule()
    root.left = shared
    root.right = shared
    assert root.modules() == [root, shared]
    # First-discovered path wins in the named view.
    assert list(root.named_modules()) == [("", root), ("left", shared)]


def test_recursive_registration_direct_cycle_terminates():
    module = NativeModule()
    module.self_ref = module  # allowed as a reference; traversal dedups
    assert module.modules() == [module]
    assert list(module.named_modules()) == [("", module)]
    assert list(module.named_parameters()) == []


def test_recursive_registration_indirect_cycle_terminates():
    a, b = NativeModule(), NativeModule()
    a.child = b
    b.parent = a
    assert a.modules() == [a, b]
    assert list(a.named_modules()) == [("", a), ("child", b)]
    assert b.modules() == [b, a]


@needs_native
def test_recursive_registration_shared_module_parameters_appear_once():
    root, shared = NativeModule(), NativeModule()
    shared.w = NativeParameter(VALUES)
    root.left = shared
    root.right = shared
    assert list(root.named_parameters()) == [("left.w", shared.w)]
    assert root.parameters() == [shared.w]


# ======================================================================
# zero_grad
# ======================================================================


@needs_native
def test_native_module_zero_grad_clears_all_unique_gradients():
    root, child = NativeModule(), NativeModule()
    root.w = NativeParameter(VALUES)
    root.child = child
    shared = NativeParameter([1.0, 2.0])
    root.shared = shared
    child.also_shared = shared
    child.b = NativeParameter([3.0])
    for p in root.parameters():
        p.sum().backward()
        assert p.grad is not None
    result = root.zero_grad()
    assert result is None  # documented return value
    for p in (root.w, shared, child.b):
        assert p.grad is None


@needs_native
def test_native_module_zero_grad_touches_nothing_else():
    module = NativeModule()
    p = NativeParameter(VALUES)
    frozen = NativeParameter(VALUES, requires_grad=False)
    module.p = p
    module.frozen = frozen
    p.sum().backward()
    module.eval()
    module.zero_grad()
    assert np.array_equal(p.to_numpy(), np.asarray(VALUES))  # data unchanged
    assert p.requires_grad is True
    assert frozen.requires_grad is False
    assert frozen.closed is False
    assert module.training is False  # training state not altered


# ======================================================================
# train / eval
# ======================================================================


def test_native_module_train_and_eval_propagate_recursively():
    root, block, sub = NativeModule(), NativeModule(), NativeModule()
    root.block = block
    block.sub = sub
    assert root.eval() is root
    assert (root.training, block.training, sub.training) == (False, False, False)
    assert root.train() is root
    assert (root.training, block.training, sub.training) == (True, True, True)
    # train(False) is exactly eval().
    root.train(False)
    assert (root.training, block.training, sub.training) == (False, False, False)


def test_native_module_train_handles_shared_modules_and_cycles():
    root, shared = NativeModule(), NativeModule()
    root.left = shared
    root.right = shared
    shared.parent = root  # indirect cycle
    root.eval()
    assert root.training is False
    assert shared.training is False
    root.train()
    assert root.training is True
    assert shared.training is True


def test_native_module_train_rejects_non_bool_before_mutation():
    root, child = NativeModule(), NativeModule()
    root.child = child
    root.eval()
    for bad in (1, 0, "train", None, 1.0):
        with pytest.raises(TypeError):
            root.train(bad)
    # Validation happens before any state changes.
    assert root.training is False
    assert child.training is False


# ======================================================================
# Isolation and lifetime
# ======================================================================


@needs_native
def test_native_module_never_closes_or_invalidates_external_references():
    module = NativeModule()
    p = NativeParameter(VALUES)
    child = NativeModule()
    child.cp = NativeParameter([1.0])
    module.w = p
    module.block = child
    module.w = NativeParameter(VALUES)  # replace
    module.block = None  # remove
    del module  # drop the whole module
    assert p.closed is False
    p.sum().backward()  # still fully usable, autograd included
    assert p.grad is not None
    assert child.cp.closed is False
    assert child.modules() == [child]


def test_native_module_owns_no_storage_and_has_no_close():
    module = NativeModule()
    assert not hasattr(module, "close")
    assert not hasattr(module, "_storage")
    # And it works without the native backend entirely: nothing above
    # allocated native memory.


def test_native_module_separate_from_framework_module():
    import tensorforge
    import tensorforge.nn as nn

    assert not issubclass(NativeModule, nn.Module)
    assert not issubclass(nn.Module, NativeModule)
    assert not hasattr(tensorforge, "NativeModule")
    # The stable framework is untouched: a Linear still traverses,
    # trains, and evaluates exactly as before.
    layer = nn.Linear(2, 3)
    assert len(layer.parameters()) == 2
    assert layer.eval().training is False
    assert layer.train().training is True
    t = tensorforge.Tensor([1.0, 2.0], requires_grad=True)
    (t * t).sum().backward()
    assert isinstance(t.grad, np.ndarray)


@needs_native
def test_native_module_traversal_never_returns_framework_objects():
    import tensorforge

    module = NativeModule()
    module.native = NativeParameter(VALUES)
    module.tensor = tensorforge.Tensor([1.0])
    module.parameter = tensorforge.Parameter([1.0])
    module.layer = tensorforge.nn.Linear(2, 2)
    for _, p in module.named_parameters():
        assert isinstance(p, NativeParameter)
    for _, m in module.named_modules():
        assert isinstance(m, NativeModule)
    assert module.parameters() == [module.native]
    assert module.modules() == [module]
