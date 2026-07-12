"""Tests for the native optimizer state contract (Advanced C++ v3.13).

NativeSGD and NativeAdam gain in-memory ``state_dict()`` /
``load_state_dict()`` over one shared versioned schema (format version
1): a plain dict with an exact optimizer type tag, validated
hyperparameters, and ordered positional parameter metadata
(shape/dtype/device per unique stored parameter — no ids, names,
values, gradients, or graph data). NativeAdam adds per-parameter step
counts and caller-owned independent NativeTensor m/v snapshots.
Loading validates everything first (exact keys, version, tag,
hyperparameters, positional metadata, counts, open plain NativeTensor
moments of exact shape/dtype/device), stages independent native copies
of every input moment, and commits scalars + counters + moments
together, closing the replaced internal buffers only after the new
state is installed — parameters, versions, gradients, registrations,
and retained autograd graphs are never touched. No file format, no
pickle, no map_location, no strict=False (file checkpoint archives are
v3.14). See src/tensorforge/experimental/native_optimizer_state.py and
the two optimizers' method docstrings.

NumPy appears below only for input preparation and as the test oracle;
snapshot and load copies run on the native copy path, and a tripwire
test proves it.

Selector: python -m pytest -q -k "native_optimizer_state or optimizer_state"
"""

import inspect
import math
from pathlib import Path

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeLinear,
    NativeModule,
    NativeMSELoss,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
)
from tensorforge.experimental import native_adam as native_adam_module

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


P_VALUES = np.array([[1.0, -2.0], [0.5, 3.0]])
G_VALUES = np.array([[0.5, -1.0], [2.0, 0.25]])
X_VALUES = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.25], [-1.0, 1.5]])
Y_VALUES = np.array([[1.0], [-0.5], [0.25], [2.0]])
LR = 0.1
BETAS = (0.9, 0.999)
EPS = 1e-8

SGD_KEYS = {"format_version", "optimizer", "lr", "parameters"}
ADAM_KEYS = SGD_KEYS | {"betas", "eps", "step_counts", "m", "v"}


def _param_with_grad(values=P_VALUES, grad_values=G_VALUES):
    """A NativeParameter whose grad is exactly ``grad_values``:
    d(sum(p * c))/dp = c, so one backward through multiply sets it."""
    parameter = NativeParameter(values)
    parameter.multiply(NativeTensor.from_array(grad_values)).sum().backward()
    return parameter


def _set_grad(parameter, grad_values):
    """Replace ``parameter``'s gradient with exactly ``grad_values``."""
    parameter.zero_grad()
    parameter.multiply(NativeTensor.from_array(grad_values)).sum().backward()


def _adam_reference(value, grad, m, v, t, lr=LR, betas=BETAS, eps=EPS):
    """The NumPy oracle mirroring the native Adam composition operation
    by operation. Returns (value_new, m_new, v_new)."""
    beta1, beta2 = betas
    m_new = beta1 * m + grad * (1.0 - beta1)
    v_new = beta2 * v + (grad * grad) * (1.0 - beta2)
    m_hat = m_new * (1.0 / (1.0 - beta1 ** t))
    v_hat = v_new * (1.0 / (1.0 - beta2 ** t))
    update = (m_hat * lr) * (1.0 / (np.sqrt(v_hat) + eps))
    return value - update, m_new, v_new


def _close(actual, expected):
    return np.allclose(actual, expected, rtol=0.0, atol=1e-15)


def _close_state(state):
    """Release the caller-owned moment snapshots of an Adam state."""
    for label in ("m", "v"):
        for snapshot in state[label]:
            snapshot.close()


# ======================================================================
# 1. NativeSGD state_dict
# ======================================================================


@needs_native
def test_native_sgd_state_dict_schema_metadata_and_independence():
    a = _param_with_grad()
    b = NativeParameter(np.array([1.0, 2.0, 3.0]))
    optimizer = NativeSGD([a, b], lr=0.25)
    state = optimizer.state_dict()
    assert set(state) == SGD_KEYS
    assert state["format_version"] == 1
    assert state["optimizer"] == "NativeSGD"
    assert state["lr"] == 0.25 and isinstance(state["lr"], float)
    # Ordered positional metadata — one entry per stored parameter, in
    # stored order, plain Python only: no parameter objects, tensors,
    # gradients, values, names, or ids anywhere in the state.
    assert isinstance(state["parameters"], tuple)
    assert state["parameters"] == (
        {"shape": (2, 2), "dtype": "float64", "device": "cpu"},
        {"shape": (3,), "dtype": "float64", "device": "cpu"},
    )

    def _plain(value):
        if isinstance(value, dict):
            return all(_plain(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return all(_plain(v) for v in value)
        return isinstance(value, (int, float, str))

    assert _plain(state)
    # The dict is independent: mutating it affects neither the
    # optimizer nor a later snapshot.
    state["lr"] = 999.0
    state["parameters"][0]["shape"] = (9, 9)
    assert optimizer.lr == 0.25
    assert optimizer.state_dict()["parameters"][0]["shape"] == (2, 2)
    # state_dict touched nothing.
    assert a.version == 0 and a.grad is not None


@needs_native
def test_native_sgd_state_dict_shared_parameters_once_and_closed_rejection():
    shared = NativeParameter(P_VALUES)
    module = NativeModule()
    module.a = shared
    module.b = shared
    optimizer = NativeSGD(list(module.parameters()) + [shared], lr=LR)
    state = optimizer.state_dict()
    assert len(state["parameters"]) == 1  # aliases already deduplicated
    # A closed stored parameter fails deterministically, naming it.
    late = NativeParameter(G_VALUES)
    failing = NativeSGD([shared, late], lr=LR)
    late.close()
    with pytest.raises(RuntimeError, match=r"parameters\[1\] has been closed"):
        failing.state_dict()


# ======================================================================
# 2. NativeSGD load_state_dict
# ======================================================================


@needs_native
def test_native_sgd_load_state_dict_restores_lr_and_touches_nothing():
    parameter = _param_with_grad()
    source = NativeSGD([parameter], lr=0.5)
    state = source.state_dict()
    target = NativeSGD([parameter], lr=0.001)
    grad_before = parameter.grad
    assert target.load_state_dict(state) is None
    assert target.lr == 0.5
    # Parameters are untouched: identity, value, version, gradient.
    assert target.parameters() == [parameter]
    assert np.array_equal(parameter.to_numpy(), P_VALUES)
    assert parameter.version == 0
    assert parameter.grad is grad_before
    # Loading the same lr again is still a successful load.
    assert target.load_state_dict(state) is None
    assert target.lr == 0.5 and parameter.version == 0


@needs_native
def test_native_sgd_load_state_dict_validates_schema():
    parameter = _param_with_grad()
    optimizer = NativeSGD([parameter], lr=LR)
    good = optimizer.state_dict()
    with pytest.raises(TypeError, match="state must be a dict"):
        optimizer.load_state_dict([("lr", 0.5)])
    missing = dict(good)
    del missing["lr"]
    with pytest.raises(ValueError, match=r"missing \['lr'\]"):
        optimizer.load_state_dict(missing)
    unexpected = dict(good)
    unexpected["momentum"] = 0.9
    with pytest.raises(ValueError, match=r"unexpected \['momentum'\]"):
        optimizer.load_state_dict(unexpected)
    for bad_version in (0, 2, "1"):
        wrong = dict(good)
        wrong["format_version"] = bad_version
        with pytest.raises((TypeError, ValueError), match="format_version"):
            optimizer.load_state_dict(wrong)
    bool_version = dict(good)
    bool_version["format_version"] = True
    with pytest.raises(TypeError, match="format_version"):
        optimizer.load_state_dict(bool_version)
    wrong_tag = dict(good)
    wrong_tag["optimizer"] = "NativeAdam"
    with pytest.raises(ValueError, match="'NativeSGD'"):
        optimizer.load_state_dict(wrong_tag)
    for bad_lr in (True, "0.5", None, 0.0, -1.0, float("nan"), float("inf")):
        wrong = dict(good)
        wrong["lr"] = bad_lr
        with pytest.raises((TypeError, ValueError), match="lr"):
            optimizer.load_state_dict(wrong)
    # Every failure left the optimizer and parameter unchanged.
    assert optimizer.lr == LR
    assert parameter.version == 0 and parameter.grad is not None


@needs_native
def test_native_sgd_load_state_dict_validates_parameter_metadata():
    a = _param_with_grad()
    b = NativeParameter(np.array([1.0, 2.0, 3.0]))
    optimizer = NativeSGD([a, b], lr=LR)
    good = optimizer.state_dict()
    count_mismatch = dict(good)
    count_mismatch["parameters"] = good["parameters"][:1]
    with pytest.raises(ValueError, match="describes 1 parameters"):
        optimizer.load_state_dict(count_mismatch)
    for field, bad_value, match in (
        ("shape", (3, 3), r"parameters'\]\[0\]\['shape'\]"),
        ("dtype", "float32", r"parameters'\]\[0\]\['dtype'\]"),
        ("device", "cuda", r"parameters'\]\[0\]\['device'\]"),
    ):
        wrong = dict(good)
        entry = dict(wrong["parameters"][0])
        entry[field] = bad_value
        wrong["parameters"] = (entry,) + wrong["parameters"][1:]
        with pytest.raises(ValueError, match=match):
            optimizer.load_state_dict(wrong)
    malformed = dict(good)
    malformed["parameters"] = ({"shape": (2, 2)},) + good["parameters"][1:]
    with pytest.raises(ValueError, match="exactly the keys"):
        optimizer.load_state_dict(malformed)
    bool_shape = dict(good)
    entry = dict(good["parameters"][0])
    entry["shape"] = (True, 2)
    bool_shape["parameters"] = (entry,) + good["parameters"][1:]
    with pytest.raises(TypeError, match="tuple of ints"):
        optimizer.load_state_dict(bool_shape)
    not_a_dict = dict(good)
    not_a_dict["parameters"] = (("shape", "dtype"),) + good["parameters"][1:]
    with pytest.raises(TypeError, match=r"parameters'\]\[0\] must be a dict"):
        optimizer.load_state_dict(not_a_dict)
    assert optimizer.lr == LR  # every failure left lr unchanged
    # A closed stored parameter fails before any change.
    late = NativeParameter(G_VALUES)
    failing = NativeSGD([a, late], lr=LR)
    failing_state = failing.state_dict()
    late.close()
    with pytest.raises(RuntimeError, match=r"parameters\[1\] has been closed"):
        failing.load_state_dict(failing_state)
    assert failing.lr == LR


@needs_native
def test_native_sgd_restored_lr_gives_identical_next_step():
    trained = _param_with_grad()
    source = NativeSGD([trained], lr=0.5)
    state = source.state_dict()
    # An equal-valued twin under a fresh optimizer with a different lr:
    # after loading, the next update is bit-identical to the source's.
    twin = NativeParameter(P_VALUES)
    _set_grad(twin, G_VALUES)
    restored = NativeSGD([twin], lr=0.001)
    restored.load_state_dict(state)
    assert restored.lr == 0.5
    source.step()
    restored.step()
    assert np.array_equal(twin.to_numpy(), trained.to_numpy())
    assert np.array_equal(twin.to_numpy(), P_VALUES - 0.5 * G_VALUES)


# ======================================================================
# 3. NativeAdam state_dict
# ======================================================================


@needs_native
def test_native_adam_state_dict_schema_values_and_snapshot_contract():
    parameter = _param_with_grad()
    optimizer = NativeAdam([parameter], lr=LR, betas=BETAS, eps=EPS)
    optimizer.step()
    optimizer.step()  # grad retained: two updates, counters at 2
    state = optimizer.state_dict()
    assert set(state) == ADAM_KEYS
    assert state["format_version"] == 1
    assert state["optimizer"] == "NativeAdam"
    assert state["lr"] == LR and state["betas"] == BETAS
    assert state["eps"] == EPS
    assert isinstance(state["betas"], tuple)
    assert state["parameters"] == (
        {"shape": (2, 2), "dtype": "float64", "device": "cpu"},
    )
    assert state["step_counts"] == (2,)
    assert isinstance(state["step_counts"], tuple)
    assert isinstance(state["m"], list) and isinstance(state["v"], list)
    # Snapshots: plain graph-free owning NativeTensors with the exact
    # internal moment values, sharing storage with nothing.
    for label in ("m", "v"):
        internal = getattr(optimizer, f"_{label}")[0]
        snapshot = state[label][0]
        assert isinstance(snapshot, NativeTensor)
        assert not isinstance(snapshot, NativeParameter)
        assert not snapshot.requires_grad and snapshot.is_leaf
        assert snapshot.owns_core and snapshot.contiguous
        assert snapshot is not internal
        assert snapshot._core.storage is not internal._core.storage
        assert np.array_equal(snapshot.to_numpy(), internal.to_numpy())
    assert state["m"][0]._core.storage is not state["v"][0]._core.storage
    assert state["m"][0]._core.storage is not parameter._core.storage
    assert state["m"][0]._core.storage is not parameter.grad._core.storage
    # Repeated snapshots are independently owned.
    again = optimizer.state_dict()
    assert again["m"][0] is not state["m"][0]
    assert again["m"][0]._core.storage is not state["m"][0]._core.storage
    # Closing the caller's snapshots never affects the optimizer.
    _close_state(state)
    _close_state(again)
    optimizer.step()
    assert optimizer.step_counts == (3,)
    # state_dict itself changed nothing.
    assert parameter.version == 3 and parameter.grad is not None


@needs_native
def test_native_adam_state_dict_rejects_closed_objects():
    parameter = _param_with_grad()
    optimizer = NativeAdam([parameter], lr=LR)
    optimizer.close()
    with pytest.raises(RuntimeError, match="closed"):
        optimizer.state_dict()
    with pytest.raises(RuntimeError, match="closed"):
        optimizer.load_state_dict({})
    # A closed stored parameter fails before anything is created.
    late = NativeParameter(G_VALUES)
    failing = NativeAdam([_param_with_grad(), late], lr=LR)
    late.close()
    with pytest.raises(RuntimeError, match=r"parameters\[1\] has been closed"):
        failing.state_dict()
    # A closed internal moment fails deterministically, naming it.
    corrupted = NativeAdam([_param_with_grad()], lr=LR)
    corrupted._v[0].close()
    with pytest.raises(RuntimeError, match=r"v state for parameters\[0\]"):
        corrupted.state_dict()


@needs_native
def test_native_adam_state_dict_snapshot_failure_closes_partials(monkeypatch):
    first = _param_with_grad()
    second = _param_with_grad()
    optimizer = NativeAdam([first, second], lr=LR)
    optimizer.step()
    real_copy = native_adam_module._native_copy
    created = []

    def flaky_copy(core):
        if len(created) == 3:  # fail on the fourth snapshot copy
            raise MemoryError("forced snapshot failure")
        result = real_copy(core)
        created.append(result)
        return result

    monkeypatch.setattr(native_adam_module, "_native_copy", flaky_copy)
    with pytest.raises(MemoryError, match="forced snapshot failure"):
        optimizer.state_dict()
    monkeypatch.undo()
    # Every snapshot created by the failed call was closed — not left
    # to garbage collection — and no partial state escaped.
    assert len(created) == 3
    assert all(core._closed for core in created)
    # Internal state, parameters, and gradients are untouched, and the
    # optimizer stays fully usable.
    assert all(not buffer.closed for buffer in optimizer._m + optimizer._v)
    assert optimizer.step_counts == (1, 1)
    assert first.grad is not None and not first.grad.closed
    good = optimizer.state_dict()
    assert set(good) == ADAM_KEYS
    _close_state(good)
    optimizer.step()
    assert optimizer.step_counts == (2, 2)


# ======================================================================
# 4. NativeAdam load_state_dict schema validation
# ======================================================================


def _fresh_adam_state():
    """A trained one-parameter optimizer plus its snapshot state."""
    parameter = _param_with_grad()
    optimizer = NativeAdam([parameter], lr=LR, betas=BETAS, eps=EPS)
    optimizer.step()
    return parameter, optimizer, optimizer.state_dict()


def _assert_adam_untouched(optimizer, parameter, moments, counts,
                           lr=LR, betas=BETAS, eps=EPS):
    """Hyperparameters, internal moments (identity and value),
    counters, parameter value/version, and gradient are unchanged."""
    assert optimizer.lr == lr and optimizer.betas == betas
    assert optimizer.eps == eps
    assert optimizer.step_counts == counts
    for label, before in (("m", moments[0]), ("v", moments[1])):
        internal = getattr(optimizer, f"_{label}")[0]
        assert internal is before[0]
        assert np.array_equal(internal.to_numpy(), before[1])
    assert parameter.grad is not None and not parameter.grad.closed


@needs_native
def test_native_adam_load_state_dict_round_trips():
    parameter, source, state = _fresh_adam_state()
    m_values = source._m[0].to_numpy()
    # Load back into the same optimizer, and into a fresh compatible
    # one built with different hyperparameters.
    assert source.load_state_dict(state) is None
    fresh = NativeAdam([parameter])  # defaults: lr=0.001 etc.
    fresh.load_state_dict(state)
    for optimizer in (source, fresh):
        assert optimizer.lr == LR and optimizer.betas == BETAS
        assert optimizer.eps == EPS
        assert optimizer.step_counts == (1,)
        assert np.array_equal(optimizer._m[0].to_numpy(), m_values)
    # Zero state round-trips too: a never-stepped optimizer's state
    # restores zero moments and zero counters.
    zero_source = NativeAdam([parameter], lr=0.5)
    zero_state = zero_source.state_dict()
    fresh.load_state_dict(zero_state)
    assert fresh.lr == 0.5 and fresh.step_counts == (0,)
    assert np.array_equal(fresh._m[0].to_numpy(), np.zeros((2, 2)))
    _close_state(state)
    _close_state(zero_state)


@needs_native
def test_native_adam_load_state_dict_validates_schema_and_scalars():
    parameter, optimizer, good = _fresh_adam_state()
    moments = (
        (optimizer._m[0], optimizer._m[0].to_numpy()),
        (optimizer._v[0], optimizer._v[0].to_numpy()),
    )
    with pytest.raises(TypeError, match="state must be a dict"):
        optimizer.load_state_dict(None)
    missing = dict(good)
    del missing["eps"]
    with pytest.raises(ValueError, match=r"missing \['eps'\]"):
        optimizer.load_state_dict(missing)
    unexpected = dict(good)
    unexpected["amsgrad"] = False
    with pytest.raises(ValueError, match=r"unexpected \['amsgrad'\]"):
        optimizer.load_state_dict(unexpected)
    wrong_tag = dict(good)
    wrong_tag["optimizer"] = "NativeSGD"
    with pytest.raises(ValueError, match="'NativeAdam'"):
        optimizer.load_state_dict(wrong_tag)
    wrong_version = dict(good)
    wrong_version["format_version"] = 2
    with pytest.raises(ValueError, match="format_version"):
        optimizer.load_state_dict(wrong_version)
    for field, bad in (("lr", 0.0), ("lr", True), ("betas", (0.9, 1.0)),
                       ("betas", (0.9,)), ("betas", "xy"), ("eps", -1.0),
                       ("eps", "1e-8")):
        wrong = dict(good)
        wrong[field] = bad
        with pytest.raises((TypeError, ValueError)):
            optimizer.load_state_dict(wrong)
    _assert_adam_untouched(optimizer, parameter, moments, (1,))
    _close_state(good)


@needs_native
def test_native_adam_load_state_dict_validates_counts_and_moments():
    parameter, optimizer, good = _fresh_adam_state()
    moments = (
        (optimizer._m[0], optimizer._m[0].to_numpy()),
        (optimizer._v[0], optimizer._v[0].to_numpy()),
    )
    # Parameter metadata mismatches.
    count_mismatch = dict(good)
    count_mismatch["parameters"] = ()
    with pytest.raises(ValueError, match="describes 0 parameters"):
        optimizer.load_state_dict(count_mismatch)
    wrong_shape = dict(good)
    wrong_shape["parameters"] = (
        {"shape": (3,), "dtype": "float64", "device": "cpu"},
    )
    with pytest.raises(ValueError, match=r"\['shape'\] is \(3,\)"):
        optimizer.load_state_dict(wrong_shape)
    # Step-count mismatches.
    for bad_counts, match in (
        ((1, 1), "holds 2 counts"),
        ((True,), r"step_counts'\]\[0\] must be an int"),
        ((-1,), r"step_counts'\]\[0\] must be >= 0"),
        ((1.0,), r"step_counts'\]\[0\] must be an int"),
        ("1", "must be a tuple or list"),
    ):
        wrong = dict(good)
        wrong["step_counts"] = bad_counts
        with pytest.raises((TypeError, ValueError), match=match):
            optimizer.load_state_dict(wrong)
    # Moment collection mismatches.
    for field_value, match in (
        ({"0": good["m"][0]}, "must be a tuple or list"),
        ([], "holds 0 tensors"),
        ([G_VALUES], "plain NativeTensor"),                # NumPy array
        ([[0.0, 0.0]], "plain NativeTensor"),              # plain list
        ([tensorforge.Tensor(P_VALUES)], "plain NativeTensor"),
        ([NativeParameter(np.zeros((2, 2)))], "plain NativeTensor"),
        ([NativeTensor.from_array(np.zeros(3))], "has shape"),
    ):
        wrong = dict(good)
        wrong["m"] = field_value
        with pytest.raises((TypeError, ValueError), match=match):
            optimizer.load_state_dict(wrong)
    closed_snapshot = NativeTensor.from_array(np.zeros((2, 2)))
    closed_snapshot.close()
    wrong = dict(good)
    wrong["v"] = [closed_snapshot]
    with pytest.raises(RuntimeError, match=r"state\['v'\]\[0\] has been closed"):
        optimizer.load_state_dict(wrong)
    _assert_adam_untouched(optimizer, parameter, moments, (1,))
    # Every failure left the input's real snapshots open and usable —
    # a later valid load still succeeds.
    optimizer.load_state_dict(good)
    _close_state(good)


# ======================================================================
# 5. NativeAdam load ownership
# ======================================================================


@needs_native
def test_native_adam_load_state_dict_ownership_and_independence():
    parameter, source, state = _fresh_adam_state()
    keys_before = set(state)
    m_list_before = state["m"]
    m_entry_before = state["m"][0]
    m_values = m_entry_before.to_numpy()
    target = NativeAdam([parameter])
    old_m, old_v = target._m[0], target._v[0]
    target.load_state_dict(state)
    # The input dict was treated as read-only: same containers, same
    # entries, every snapshot still open with its value intact.
    assert set(state) == keys_before
    assert state["m"] is m_list_before and state["m"][0] is m_entry_before
    assert not m_entry_before.closed
    assert np.array_equal(m_entry_before.to_numpy(), m_values)
    # The optimizer installed independent copies — identical values,
    # zero shared storage — and closed its replaced old buffers.
    assert target._m[0] is not m_entry_before
    assert target._m[0]._core.storage is not m_entry_before._core.storage
    assert np.array_equal(target._m[0].to_numpy(), m_values)
    assert old_m.closed and old_v.closed
    # Closing the caller's state does not affect the optimizer …
    _close_state(state)
    target.step()
    assert target.step_counts == (2,)
    # … and closing the optimizer does not close a fresh caller state.
    second_state = target.state_dict()
    target.close()
    assert not second_state["m"][0].closed
    _close_state(second_state)
    # An identical-value load still installs fresh independent state.
    replay = source.state_dict()
    installed_before = source._m[0]
    source.load_state_dict(replay)
    assert source._m[0] is not installed_before
    assert installed_before.closed
    _close_state(replay)


# ======================================================================
# 6. NativeAdam atomic failure
# ======================================================================


@needs_native
def test_native_adam_load_failure_is_atomic_across_entries():
    first = _param_with_grad()
    second = _param_with_grad()
    optimizer = NativeAdam([first, second], lr=LR)
    optimizer.step()
    good = optimizer.state_dict()
    before = {
        "m": [(buffer, buffer.to_numpy()) for buffer in optimizer._m],
        "v": [(buffer, buffer.to_numpy()) for buffer in optimizer._v],
    }
    # A later invalid moment (wrong shape at v[1]) prevents every
    # earlier replacement.
    wrong = dict(good)
    wrong["v"] = [good["v"][0], NativeTensor.from_array(np.zeros(3))]
    with pytest.raises(ValueError, match=r"state\['v'\]\[1\] has shape"):
        optimizer.load_state_dict(wrong)
    # A later closed snapshot does too.
    closed_entry = NativeTensor.from_array(np.zeros((2, 2)))
    closed_entry.close()
    wrong = dict(good)
    wrong["v"] = [good["v"][0], closed_entry]
    with pytest.raises(RuntimeError, match=r"state\['v'\]\[1\] has been closed"):
        optimizer.load_state_dict(wrong)
    for label in ("m", "v"):
        for (buffer, values), current in zip(before[label],
                                             getattr(optimizer, f"_{label}")):
            assert current is buffer
            assert np.array_equal(current.to_numpy(), values)
    assert optimizer.step_counts == (1, 1)
    assert first.version == 1 and second.version == 1
    assert good["m"][0].closed is False  # input untouched
    # The optimizer works on a later valid load.
    optimizer.load_state_dict(good)
    _close_state(good)


@needs_native
def test_native_adam_load_staging_failure_preserves_everything(monkeypatch):
    first = _param_with_grad()
    second = _param_with_grad()
    optimizer = NativeAdam([first, second], lr=LR)
    optimizer.step()
    state = optimizer.state_dict()
    internal_before = list(optimizer._m) + list(optimizer._v)
    real_copy = native_adam_module._native_copy
    calls = {"n": 0}

    def flaky_copy(core):
        calls["n"] += 1
        if calls["n"] == 3:  # after two staged copies
            raise MemoryError("forced staging failure")
        return real_copy(core)

    monkeypatch.setattr(native_adam_module, "_native_copy", flaky_copy)
    with pytest.raises(MemoryError, match="forced staging failure"):
        optimizer.load_state_dict(state)
    monkeypatch.undo()
    # Internal state by identity and value, counters, hyperparameters,
    # parameters, gradients, and the caller's input are all untouched,
    # and the same optimizer completes a later valid load.
    assert list(optimizer._m) + list(optimizer._v) == internal_before
    assert all(not buffer.closed for buffer in internal_before)
    assert optimizer.step_counts == (1, 1)
    assert optimizer.lr == LR
    assert first.version == 1 and first.grad is not None
    assert all(not snapshot.closed
               for label in ("m", "v") for snapshot in state[label])
    optimizer.load_state_dict(state)
    assert optimizer.step_counts == (1, 1)
    _close_state(state)


# ======================================================================
# 7. Parameter and autograd isolation
# ======================================================================


@needs_native
def test_native_optimizer_state_load_never_touches_parameters_or_graphs():
    module = NativeModule()
    weight = NativeParameter(P_VALUES)
    module.weight = weight
    module.alias = weight
    x = NativeTensor.from_array(P_VALUES, requires_grad=True)
    out = x.matmul(weight).sum()
    out.backward(retain_graph=True)  # a retained value-sensitive graph
    grad_before = weight.grad
    grad_values = grad_before.to_numpy()
    model_keys = list(module.state_dict())
    optimizer = NativeAdam(module.parameters(), lr=LR)
    optimizer.step()
    version_after_step = weight.version
    state = optimizer.state_dict()
    changed = dict(state)
    changed["lr"] = 0.5  # a genuinely different state to load
    optimizer.load_state_dict(state)
    optimizer.load_state_dict(changed)
    sgd = NativeSGD(module.parameters(), lr=LR)
    sgd.load_state_dict(sgd.state_dict())
    # No parameter value, version, gradient, registration, alias, or
    # model state key moved — optimizer-state loading is invisible to
    # the parameter and autograd layers.
    assert weight.version == version_after_step
    assert weight.grad is grad_before
    assert np.array_equal(weight.grad.to_numpy(), grad_values)
    assert module.weight is weight and module.alias is weight
    assert list(module.state_dict()) == model_keys
    # The retained graph is *stale* only because step() mutated the
    # weight — but a graph retained *after* the step must survive
    # optimizer-state loading. Prove it directly: fresh forward,
    # retained backward, then loads, then the same retained backward
    # again — no stale error, because loading never moves a version.
    weight.zero_grad()
    x.zero_grad()
    fresh = x.matmul(weight).sum()
    fresh.backward(retain_graph=True)
    optimizer.load_state_dict(state)
    fresh.backward(retain_graph=True)  # still valid after loading
    assert weight.grad is not None
    _close_state(state)


# ======================================================================
# 8. Shared, frozen, and late-active behavior
# ======================================================================


@needs_native
def test_native_adam_state_round_trips_shared_frozen_and_untrained():
    shared = _param_with_grad()
    module = NativeModule()
    module.a = shared
    module.b = shared
    frozen = NativeParameter(G_VALUES, requires_grad=False)
    no_grad = NativeParameter(P_VALUES)
    optimizer = NativeAdam(
        list(module.parameters()) + [shared, frozen, no_grad], lr=LR
    )
    optimizer.step()  # only the shared parameter is active
    state = optimizer.state_dict()
    # Shared aliases and duplicates: one entry everywhere.
    assert len(state["parameters"]) == 3
    assert len(state["m"]) == 3 and len(state["v"]) == 3
    assert state["step_counts"] == (1, 0, 0)
    # Frozen and grad=None parameters round-trip their zero state.
    fresh = NativeAdam([shared, frozen, no_grad])
    fresh.load_state_dict(state)
    assert fresh.step_counts == (1, 0, 0)
    assert np.array_equal(fresh._m[1].to_numpy(), np.zeros((2, 2)))
    assert np.array_equal(
        fresh._m[0].to_numpy(), optimizer._m[0].to_numpy()
    )
    # Equal-valued separate parameters stay separate entries.
    twin = NativeParameter(P_VALUES)
    twinned = NativeAdam([no_grad, twin], lr=LR)
    assert len(twinned.state_dict()["parameters"]) == 2
    _close_state(state)


@needs_native
def test_native_adam_late_active_parameter_resumes_restored_state():
    active = _param_with_grad()
    late = NativeParameter(G_VALUES)
    optimizer = NativeAdam([active, late], lr=LR, betas=BETAS, eps=EPS)
    optimizer.step()
    optimizer.step()  # active: t=2 with the retained gradient
    state = optimizer.state_dict()
    assert state["step_counts"] == (2, 0)
    # Restore into a fresh optimizer over the same parameters, then
    # activate the late parameter: it resumes from its exact restored
    # counter (0 → first update at t=1) and zero moments, while the
    # active parameter continues at t=3 from its restored moments.
    restored = NativeAdam([active, late])
    restored.load_state_dict(state)
    assert restored.step_counts == (2, 0)
    late_grad = np.array([[0.5, 0.5], [-1.0, 2.0]])
    _set_grad(late, late_grad)
    value_before = active.to_numpy()
    m_before = restored._m[0].to_numpy()
    v_before = restored._v[0].to_numpy()
    restored.step()
    assert restored.step_counts == (3, 1)
    zeros = np.zeros((2, 2))
    expected_late, _, _ = _adam_reference(G_VALUES, late_grad, zeros, zeros, t=1)
    assert _close(late.to_numpy(), expected_late)
    expected_active, _, _ = _adam_reference(
        value_before, G_VALUES, m_before, v_before, t=3
    )
    assert _close(active.to_numpy(), expected_active)
    _close_state(state)


# ======================================================================
# 9. Deterministic in-memory continuation
# ======================================================================


def _build_training():
    """A deterministic model/loss/optimizer/data setup."""
    model = NativeSequential(
        NativeLinear(2, 8, seed=0),
        NativeReLU(),
        NativeLinear(8, 1, seed=1),
    )
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    x = NativeTensor.from_array(X_VALUES)
    y = NativeTensor.from_array(Y_VALUES)
    return model, NativeMSELoss(), optimizer, x, y


def _train(model, loss_fn, optimizer, x, y, steps):
    """Run ``steps`` full iterations, returning the loss history.
    Gradients are cleared at every iteration boundary."""
    losses = []
    for _ in range(steps):
        prediction = model(x)
        loss = loss_fn(prediction, y)
        losses.append(float(loss.to_numpy()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss.close()
        prediction.close()
    return losses


@needs_native
def test_native_adam_in_memory_continuation_matches_uninterrupted_run():
    n_steps, m_steps = 6, 5
    model_a, loss_fn, optimizer_a, x, y = _build_training()
    _train(model_a, loss_fn, optimizer_a, x, y, n_steps)
    parameters_a = model_a.parameters()
    assert all(parameter.grad is None for parameter in parameters_a)

    # Snapshot both contracts, then build a fresh model/optimizer pair
    # and restore into it.
    model_state = model_a.state_dict()
    optimizer_state = optimizer_a.state_dict()
    model_b, _, optimizer_b, _, _ = _build_training()
    parameters_b = model_b.parameters()
    model_b.load_state_dict(model_state)
    versions_b_after_model_load = [p.version for p in parameters_b]
    assert versions_b_after_model_load == [1, 1, 1, 1]  # model contract
    optimizer_b.load_state_dict(optimizer_state)
    # Optimizer-state loading increments no parameter version.
    assert [p.version for p in parameters_b] == versions_b_after_model_load
    assert optimizer_b.lr == optimizer_a.lr
    assert optimizer_b.step_counts == (n_steps,) * 4
    # Restored state aliases neither live optimizer.
    for snapshot in optimizer_state["m"] + optimizer_state["v"]:
        for live in optimizer_a._m + optimizer_a._v + optimizer_b._m + optimizer_b._v:
            assert snapshot._core.storage is not live._core.storage

    # Continue both runs on identical data: bit-identical trajectories.
    losses_a = _train(model_a, loss_fn, optimizer_a, x, y, m_steps)
    losses_b = _train(model_b, loss_fn, optimizer_b, x, y, m_steps)
    assert losses_a == losses_b
    assert all(math.isfinite(value) for value in losses_a)
    for parameter_a, parameter_b in zip(parameters_a, parameters_b):
        assert np.array_equal(parameter_a.to_numpy(), parameter_b.to_numpy())
        assert parameter_a.grad is None and parameter_b.grad is None
    for index in range(4):
        assert np.array_equal(
            optimizer_a._m[index].to_numpy(), optimizer_b._m[index].to_numpy()
        )
        assert np.array_equal(
            optimizer_a._v[index].to_numpy(), optimizer_b._v[index].to_numpy()
        )
    assert optimizer_a.step_counts == optimizer_b.step_counts == (
        (n_steps + m_steps,) * 4
    )
    # Version deltas over the continuation match exactly: one increment
    # per parameter per step in both runs.
    assert [p.version for p in parameters_a] == [n_steps + m_steps] * 4
    assert [p.version for p in parameters_b] == [
        baseline + m_steps for baseline in versions_b_after_model_load
    ]

    # Release every caller-owned snapshot per the lifetime contract.
    for snapshot in model_state.values():
        snapshot.close()
    _close_state(optimizer_state)
    optimizer_a.close()
    optimizer_b.close()


# ======================================================================
# 10. Lifetime and guardrails
# ======================================================================


@needs_native
def test_native_optimizer_state_uses_no_numpy_compute(monkeypatch):
    parameter = _param_with_grad()
    adam = NativeAdam([parameter], lr=LR)
    adam.step()
    sgd = NativeSGD([parameter], lr=LR)
    sgd_state = sgd.state_dict()

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("sqrt", "reciprocal", "divide", "add", "subtract",
                 "multiply", "matmul", "sum", "mean", "negative",
                 "power", "copyto", "copy"):
        monkeypatch.setattr(np, name, _tripwire)
    adam_state = adam.state_dict()
    adam.load_state_dict(adam_state)
    sgd.load_state_dict(sgd_state)
    monkeypatch.undo()
    # Snapshot and load copies ran on the native copy path, and the
    # snapshots are ordinary graph-free leaves.
    assert adam_state["m"][0].is_leaf
    assert not adam_state["m"][0].requires_grad
    assert np.array_equal(
        adam_state["m"][0].to_numpy(), adam._m[0].to_numpy()
    )
    _close_state(adam_state)


@needs_native
def test_native_optimizer_state_scope_boundaries_hold():
    parameter = _param_with_grad()
    adam = NativeAdam([parameter], lr=LR)
    sgd = NativeSGD([parameter], lr=LR)
    # Exactly one caller-owned state argument: no strict flag, no
    # map_location, no paths, no file/checkpoint surface.
    for optimizer in (adam, sgd):
        signature = inspect.signature(optimizer.load_state_dict)
        assert list(signature.parameters) == ["state"]
        assert list(inspect.signature(optimizer.state_dict).parameters) == []
        for absent in ("save", "load", "save_checkpoint", "load_checkpoint",
                       "save_state_dict", "map_location"):
            assert not hasattr(optimizer, absent)
    # The schema carries no names, ids, or addresses — only the locked
    # keys — and the sources use no file or pickle machinery.
    state = adam.state_dict()
    assert set(state) == ADAM_KEYS
    assert set(state["parameters"][0]) == {"shape", "dtype", "device"}
    experimental_dir = (
        Path(__file__).resolve().parent.parent
        / "src" / "tensorforge" / "experimental"
    )
    for name in ("native_sgd.py", "native_adam.py",
                 "native_optimizer_state.py"):
        source = (experimental_dir / name).read_text(encoding="utf-8")
        for banned in ("pickle", "import json", "savez", ".npz",
                       "map_location", "import pathlib", "from pathlib"):
            assert banned not in source, f"{name} contains {banned!r}"
    # The stable optimizers are untouched: their own plain state_dict
    # surfaces remain, and neither accepts the native schema's tensors.
    assert hasattr(tensorforge.Adam, "state_dict")
    assert hasattr(tensorforge.SGD, "state_dict")
    _close_state(state)
