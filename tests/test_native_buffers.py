"""Tests for NativeModule buffer infrastructure (Advanced C++ v3.15).

Buffers are non-Parameter persistent state (the native analog of
tensorforge.nn buffers) registered via register_buffer. They ride the
same identity-deduplicated, cycle-safe traversal as parameters, never
appear in parameters(), and — when persistent — participate in
state_dict()/load_state_dict() and native checkpoints, atomically and
with object identity preserved on restore.

Selector: python -m pytest -q -k native_buffer
"""

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeModule,
    NativeParameter,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


def _tensor(values, requires_grad=False):
    return NativeTensor.from_array(np.asarray(values, dtype=np.float64),
                                   requires_grad=requires_grad)


@needs_native
def test_register_and_access_buffer():
    m = NativeModule()
    buf = _tensor([1.0, 2.0, 3.0])
    m.register_buffer("running_mean", buf)
    assert m.running_mean is buf
    assert [name for name, _ in m.named_buffers()] == ["running_mean"]
    assert m.buffers() == [buf]


@needs_native
def test_buffer_not_a_parameter():
    m = NativeModule()
    m.register_buffer("stat", _tensor([0.0]))
    m.weight = NativeParameter(np.ones((2, 2)))
    assert m.parameters() == [m.weight]
    assert [n for n, _ in m.named_parameters()] == ["weight"]
    assert [n for n, _ in m.named_buffers()] == ["stat"]


@needs_native
def test_register_buffer_rejects_bad_inputs():
    m = NativeModule()
    with pytest.raises(TypeError):  # not a NativeTensor
        m.register_buffer("b", np.ones(3))
    with pytest.raises(TypeError):  # a NativeParameter belongs in params
        m.register_buffer("b", NativeParameter(np.ones(3)))
    with pytest.raises(ValueError):  # must not require grad
        m.register_buffer("b", _tensor([1.0], requires_grad=True))
    with pytest.raises(TypeError):  # persistent must be bool
        m.register_buffer("b", _tensor([1.0]), persistent="yes")
    with pytest.raises(ValueError):  # reserved name
        m.register_buffer("_buffers", _tensor([1.0]))


@needs_native
def test_buffer_unregister_and_recategorize():
    m = NativeModule()
    buf = _tensor([1.0])
    m.register_buffer("x", buf)
    # None unregisters, leaving a readable None attribute.
    m.register_buffer("x", None)
    assert m.x is None
    assert list(m.named_buffers()) == []
    with pytest.raises(KeyError):
        m.register_buffer("x", None)
    # A name is one category: promoting to a parameter evicts the buffer.
    m.register_buffer("y", _tensor([1.0]))
    m.y = NativeParameter(np.ones(2))
    assert list(m.named_buffers()) == []
    assert [n for n, _ in m.named_parameters()] == ["y"]


@needs_native
def test_recursive_and_deduplicated_buffer_traversal():
    root = NativeModule()
    child = NativeModule()
    root.child = child
    shared = _tensor([9.0])
    root.register_buffer("a", shared)
    child.register_buffer("b", shared)  # same object, deeper path
    child.register_buffer("c", _tensor([1.0]))
    names = [name for name, _ in root.named_buffers()]
    # shared buffer yielded once under its first-discovered name.
    assert names == ["a", "child.c"]


@needs_native
def test_non_persistent_buffer_excluded_from_state_dict():
    m = NativeModule()
    m.register_buffer("keep", _tensor([1.0, 2.0]), persistent=True)
    m.register_buffer("drop", _tensor([3.0]), persistent=False)
    m.weight = NativeParameter(np.ones((2,)))
    state = m.state_dict()
    try:
        assert set(state) == {"weight", "keep"}
        assert np.allclose(state["keep"].to_numpy(), [1.0, 2.0])
    finally:
        for snap in state.values():
            snap.close()
    # but the transient buffer is still discoverable
    assert "drop" in [n for n, _ in m.named_buffers()]


@needs_native
def test_load_state_dict_restores_buffer_in_place():
    m = NativeModule()
    buf = _tensor([1.0, 2.0])
    m.register_buffer("stat", buf)
    m.weight = NativeParameter(np.zeros((2,)))
    snapshot = m.state_dict()
    try:
        # mutate live buffer value by loading different values
        new = {
            "weight": NativeTensor.from_array(np.array([5.0, 6.0])),
            "stat": NativeTensor.from_array(np.array([7.0, 8.0])),
        }
        m.load_state_dict(new)
        assert m.stat is buf  # identity preserved
        assert np.allclose(buf.to_numpy(), [7.0, 8.0])
        assert np.allclose(m.weight.to_numpy(), [5.0, 6.0])
        for t in new.values():
            t.close()
        # restore original snapshot
        m.load_state_dict(snapshot)
        assert np.allclose(buf.to_numpy(), [1.0, 2.0])
        assert m.stat is buf
    finally:
        for snap in snapshot.values():
            snap.close()


@needs_native
def test_load_state_dict_atomic_across_param_and_buffer():
    m = NativeModule()
    buf = _tensor([1.0, 2.0])
    m.register_buffer("stat", buf)
    m.weight = NativeParameter(np.zeros((2,)))
    weight_ver = m.weight.version
    # A buffer value with a wrong shape must abort the whole load with
    # neither the parameter nor the buffer mutated.
    bad = {
        "weight": NativeTensor.from_array(np.array([5.0, 6.0])),
        "stat": NativeTensor.from_array(np.array([1.0, 2.0, 3.0])),  # wrong shape
    }
    try:
        with pytest.raises(ValueError):
            m.load_state_dict(bad)
        assert np.allclose(m.weight.to_numpy(), [0.0, 0.0])
        assert np.allclose(buf.to_numpy(), [1.0, 2.0])
        assert m.weight.version == weight_ver  # no version bump on failure
    finally:
        for t in bad.values():
            t.close()


def _model_with_buffer(weight_value, stat_value):
    m = NativeModule()
    m.weight = NativeParameter(np.asarray(weight_value, dtype=np.float64))
    m.register_buffer("stat", _tensor(stat_value), persistent=True)
    m.register_buffer("scratch", _tensor([0.0]), persistent=False)
    return m


@needs_native
def test_native_checkpoint_round_trips_persistent_buffer(tmp_path):
    path = str(tmp_path / "ckpt.npz")
    source = _model_with_buffer([[1.0, 2.0]], [3.0, 4.0])
    save_native_checkpoint(path, source)

    target = _model_with_buffer([[0.0, 0.0]], [0.0, 0.0])
    stat_obj = target.stat
    load_native_checkpoint(path, target)

    assert np.allclose(target.weight.to_numpy(), [[1.0, 2.0]])
    assert np.allclose(target.stat.to_numpy(), [3.0, 4.0])
    assert target.stat is stat_obj  # buffer identity preserved on restore
    # the non-persistent buffer was never serialized and is untouched
    assert np.allclose(target.scratch.to_numpy(), [0.0])


@needs_native
def test_native_checkpoint_rejects_buffer_shape_mismatch(tmp_path):
    path = str(tmp_path / "ckpt.npz")
    save_native_checkpoint(path, _model_with_buffer([[1.0, 2.0]], [3.0, 4.0]))
    # a model whose buffer has a different shape must be rejected
    mismatched = _model_with_buffer([[0.0, 0.0]], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        load_native_checkpoint(path, mismatched)


@needs_native
def test_state_dict_snapshot_failure_closes_partial_snapshots(monkeypatch):
    """A failure part-way through building ``state_dict()`` snapshots must
    close every snapshot already created and return no mapping, so a
    failure never leaks native memory or yields a partial state dict.
    Generic buffer/state infrastructure, independent of any layer: the
    module here mixes a parameter and a persistent buffer, and the failure
    is injected after the parameter snapshot exists. Uses the private
    ``_native_copy`` seam rather than a production failure flag."""
    from tensorforge.experimental import native_module

    m = NativeModule()
    m.weight = NativeParameter(np.ones((2, 2)))
    buf = _tensor([1.0, 2.0])
    m.register_buffer("stat", buf, persistent=True)

    real_copy = native_module._native_copy
    created = []
    calls = {"n": 0}

    def failing_copy(core):
        calls["n"] += 1
        if calls["n"] == 2:            # after the parameter snapshot exists
            raise MemoryError("forced snapshot failure")
        result = real_copy(core)
        created.append(result)
        return result

    monkeypatch.setattr(native_module, "_native_copy", failing_copy)
    with pytest.raises(MemoryError):
        m.state_dict()
    monkeypatch.undo()

    # The one snapshot that was created was closed; nothing lingers, and
    # no partial mapping was returned (the call raised).
    assert len(created) == 1
    assert created[0]._closed is True
    # The model is untouched, and a valid snapshot now succeeds completely.
    assert np.allclose(m.weight.to_numpy(), np.ones((2, 2)))
    assert np.allclose(buf.to_numpy(), [1.0, 2.0])
    state = m.state_dict()
    assert set(state) == {"weight", "stat"}
    for snap in state.values():
        snap.close()
