"""Stable Module hardening: identity-aware, cycle-safe traversal,
Boolean train() validation, and atomic load_state_dict (the repair
milestone's Stage 4/5 guarantees)."""

import numpy as np
import pytest

from tensorforge import Parameter
from tensorforge.nn import Linear, Module, Sequential


class _Tied(Module):
    """Two attributes referencing the same Parameter (tied weights)."""

    def __init__(self):
        self.a = Parameter(np.ones((2, 2)))
        self.b = self.a  # shared identity


class _SharedChild(Module):
    """The same child Module registered under two names."""

    def __init__(self):
        self.left = Linear(2, 2)
        self.right = self.left  # shared identity


class _Cyclic(Module):
    """A module that references itself (a reference cycle)."""

    def __init__(self):
        self.lin = Linear(2, 2)
        self.loop = self  # cycle


# --- identity deduplication -------------------------------------------------


def test_tied_parameter_yielded_once():
    model = _Tied()
    names = [name for name, _ in model.named_parameters()]
    params = model.parameters()
    assert names == ["a"]  # first-encountered name wins
    assert len(params) == 1
    assert params[0] is model.a


def test_shared_child_module_expanded_once():
    model = _SharedChild()
    names = [name for name, _ in model.named_parameters()]
    # left.weight/bias appear once; right.* is the same object, skipped.
    assert names == ["left.weight", "left.bias"]


def test_shared_parameter_across_subtrees_counts_once():
    model = _SharedChild()
    # weight+bias of a single Linear(2,2): (2*2) + 2 == 6 scalars.
    assert model.num_parameters(trainable_only=False) == 6


# --- cycle safety -----------------------------------------------------------


def test_named_parameters_terminates_on_cycle():
    model = _Cyclic()
    names = [name for name, _ in model.named_parameters()]
    assert names == ["lin.weight", "lin.bias"]


def test_train_terminates_on_cycle():
    model = _Cyclic()
    model.eval()
    assert model.training is False
    assert model.lin.training is False
    model.train()
    assert model.training is True


# --- Boolean train() validation ---------------------------------------------


@pytest.mark.parametrize("bad", ["eval", 1, 0, [], None, "train"])
def test_train_rejects_non_bool(bad):
    model = Linear(2, 2)
    with pytest.raises(TypeError):
        model.train(bad)
    # rejected before any state change
    assert model.training is True


def test_train_accepts_bools_and_eval_equivalence():
    model = Sequential(Linear(2, 2))
    assert model.train(True) is model
    assert model.training is True
    model.train(False)
    assert model.training is False
    model.eval()
    assert model.training is False


# --- atomic load_state_dict -------------------------------------------------


def test_load_state_dict_atomic_on_later_shape_mismatch():
    model = Sequential(Linear(2, 2), Linear(2, 3))
    good = model.state_dict()
    before = {name: p.data.copy() for name, p in model.named_parameters()}
    ids_before = {name: id(p) for name, p in model.named_parameters()}

    # Corrupt a *later* key so the first parameters validate fine.
    bad = dict(good)
    last_key = list(good)[-1]
    bad[last_key] = np.zeros((99, 99))

    with pytest.raises(ValueError):
        model.load_state_dict(bad)

    # Nothing mutated: values and object identities are unchanged.
    for name, p in model.named_parameters():
        assert np.array_equal(p.data, before[name])
        assert id(p) == ids_before[name]


def test_load_state_dict_preserves_identity_on_success():
    model = Sequential(Linear(2, 2))
    ids = [id(p) for p in model.parameters()]
    state = {name: np.zeros_like(p.data) for name, p in model.named_parameters()}
    model.load_state_dict(state)
    assert [id(p) for p in model.parameters()] == ids
    assert all(np.count_nonzero(p.data) == 0 for p in model.parameters())
