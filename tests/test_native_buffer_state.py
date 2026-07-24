"""Phase F, milestone F1 — the private atomic native-buffer state
transaction (``tensorforge.experimental._native_state``).

These tests cover the transaction directly, because it is the primitive
F3/F4 will use to commit ``running_mean`` and ``running_var`` together
without going through ``load_state_dict``. They are deliberately
behavioral — identity, values, versions, storage lifetime, atomicity —
rather than assertions about the helper's internal shape; the only
private details they lean on are the documented failure-injection seams,
which exist precisely so a failure at each step can be simulated.

Nothing here implements or exercises normalization mathematics: F1 is
state management and capability reporting only.
"""

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeModule, NativeParameter, NativeTensor,
)
from tensorforge.experimental import _native_state

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="the experimental C++ backend is not built"
)


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------

@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — the project's
    established deterministic instrumentation for native-allocation
    lifetime (the Phase-C/D/E precedent). Never relies on GC timing."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    monkeypatch.setattr(cpp.NativeStorage, "__init__", tracked_init)
    monkeypatch.setattr(cpp.NativeStorage, "close", tracked_close)
    return open_ids


class _Stateful(NativeModule):
    """A module with one parameter and two persistent buffers — the
    shape of a future BatchNorm, with no normalization mathematics."""

    def __init__(self):
        super().__init__()
        self.gamma = NativeParameter([1.0, 1.0, 1.0])
        self.register_buffer("running_mean", NativeTensor.from_array(
            [0.0, 0.0, 0.0]))
        self.register_buffer("running_var", NativeTensor.from_array(
            [1.0, 1.0, 1.0]))


def _entry(label, destination, values, source=None):
    """A NativeStateEntry whose factory builds a fresh owning contiguous
    core holding ``values``. ``source`` defaults to a per-entry sentinel
    so distinct entries never look like duplicates of each other."""
    holder = NativeTensor.from_array(values)
    marker = object() if source is None else source

    def make_core():
        return cpp.NativeTensorCore.from_array(np.asarray(values, dtype=float))

    entry = _native_state.NativeStateEntry(
        label=label, destination=destination, make_core=make_core,
        source=marker,
    )
    holder.close()
    return entry


def _close_module(module):
    for parameter in module.parameters():
        parameter.close()
    for buffer in module.buffers():
        buffer.close()


# --------------------------------------------------------------------------
# Basic buffer replacement
# --------------------------------------------------------------------------

@needs_native
def test_single_buffer_replacement_succeeds_and_preserves_identity():
    model = _Stateful()
    buffer = model.running_mean
    identity = id(buffer)
    old_core = buffer._core

    replaced = _native_state.replace_native_state(
        [_entry("running_mean", buffer, [4.0, 5.0, 6.0])]
    )

    assert replaced == 1
    # Identity preserved: the registered object is the same object, and
    # the module still resolves to it.
    assert id(model.running_mean) == identity
    assert model.running_mean is buffer
    # The value changed, through the same object.
    assert np.array_equal(buffer.to_numpy(), np.asarray([4.0, 5.0, 6.0]))
    # The replaced core closed; the installed one is open and usable.
    assert old_core._closed is True
    assert buffer._core is not old_core
    assert buffer.owns_core is True
    assert buffer.closed is False
    assert np.array_equal(buffer.sum().to_numpy(), np.asarray(15.0))
    # A buffer has no parameter version, and none was invented.
    assert not hasattr(buffer, "version")
    assert not hasattr(buffer, "_version")
    _close_module(model)


@needs_native
def test_replacement_leaves_the_source_tensor_independent_and_usable():
    model = _Stateful()
    source = NativeTensor.from_array([7.0, 8.0, 9.0])

    _native_state.replace_native_state([
        _native_state.NativeStateEntry(
            label="running_mean",
            destination=model.running_mean,
            make_core=lambda: cpp.NativeTensorCore.from_array(
                source.to_numpy()
            ),
            source=source,
        )
    ])

    assert np.array_equal(model.running_mean.to_numpy(),
                          np.asarray([7.0, 8.0, 9.0]))
    # The source is untouched, open, and shares no storage with the
    # destination it seeded.
    assert source.closed is False
    assert np.array_equal(source.to_numpy(), np.asarray([7.0, 8.0, 9.0]))
    assert source._core.storage is not model.running_mean._core.storage
    source.close()
    _close_module(model)


# --------------------------------------------------------------------------
# Multi-buffer atomic replacement — the future BatchNorm shape
# --------------------------------------------------------------------------

@needs_native
def test_two_buffers_replace_as_one_transaction_without_load_state_dict():
    """The F3/F4 use case: commit running_mean and running_var together,
    building entries directly — no fake state dictionary, no call to the
    public load_state_dict."""
    model = _Stateful()
    mean, var = model.running_mean, model.running_var
    mean_identity, var_identity = id(mean), id(var)
    old_mean_core, old_var_core = mean._core, var._core

    replaced = _native_state.replace_native_state([
        _entry("running_mean", mean, [0.5, 0.5, 0.5]),
        _entry("running_var", var, [2.0, 2.0, 2.0]),
    ])

    assert replaced == 2
    assert id(mean) == mean_identity and id(var) == var_identity
    assert model.running_mean is mean and model.running_var is var
    assert np.array_equal(mean.to_numpy(), np.asarray([0.5, 0.5, 0.5]))
    assert np.array_equal(var.to_numpy(), np.asarray([2.0, 2.0, 2.0]))
    # Both replaced cores closed exactly once; both replacements are
    # independent, owning, and open.
    assert old_mean_core._closed is True and old_var_core._closed is True
    assert mean._core.storage is not var._core.storage
    assert mean.owns_core is True and var.owns_core is True
    # Both are still registered as persistent buffers under their names.
    assert dict(model.named_buffers()).keys() == {"running_mean", "running_var"}
    assert set(model.state_dict()) == {"gamma", "running_mean", "running_var"}
    _close_module(model)


@needs_native
def test_two_buffer_transaction_leaves_no_net_storage(live_storages):
    model = _Stateful()
    baseline = len(live_storages)

    _native_state.replace_native_state([
        _entry("running_mean", model.running_mean, [0.5, 0.5, 0.5]),
        _entry("running_var", model.running_var, [2.0, 2.0, 2.0]),
    ])

    # Two cores in, two cores out: the count is unchanged.
    assert len(live_storages) == baseline
    _close_module(model)


# --------------------------------------------------------------------------
# Mixed parameter and buffer transaction
# --------------------------------------------------------------------------

@needs_native
def test_parameter_and_buffer_replace_together_atomically():
    model = _Stateful()
    gamma, mean = model.gamma, model.running_mean
    gamma_version = gamma.version
    gamma_identity, mean_identity = id(gamma), id(mean)
    old_gamma_core, old_mean_core = gamma._core, mean._core

    _native_state.replace_native_state([
        _entry("gamma", gamma, [3.0, 3.0, 3.0]),
        _entry("running_mean", mean, [9.0, 9.0, 9.0]),
    ])

    assert id(gamma) == gamma_identity and id(mean) == mean_identity
    assert np.array_equal(gamma.to_numpy(), np.asarray([3.0, 3.0, 3.0]))
    assert np.array_equal(mean.to_numpy(), np.asarray([9.0, 9.0, 9.0]))
    # The parameter's version moved exactly once; the buffer has none.
    assert gamma.version == gamma_version + 1
    assert not hasattr(mean, "_version")
    # Parameter-ness, gradient state, and registration all survive.
    assert isinstance(gamma, NativeParameter)
    assert gamma.requires_grad is True
    assert gamma.grad is None
    assert model.gamma is gamma
    assert old_gamma_core._closed is True and old_mean_core._closed is True
    _close_module(model)


# --------------------------------------------------------------------------
# Validation failure — nothing may move
# --------------------------------------------------------------------------

@needs_native
def test_shape_mismatch_leaves_all_state_unchanged(live_storages):
    model = _Stateful()
    baseline = len(live_storages)
    before = [t.to_numpy().copy() for t in (model.gamma, model.running_mean)]
    version = model.gamma.version
    cores = (model.gamma._core, model.running_mean._core)

    with pytest.raises(ValueError, match="metadata mismatch"):
        _native_state.replace_native_state([
            _entry("gamma", model.gamma, [1.0, 1.0, 1.0]),
            _entry("running_mean", model.running_mean, [1.0, 1.0]),
        ])

    assert np.array_equal(model.gamma.to_numpy(), before[0])
    assert np.array_equal(model.running_mean.to_numpy(), before[1])
    assert model.gamma.version == version
    assert model.gamma._core is cores[0]
    assert model.running_mean._core is cores[1]
    assert all(core._closed is False for core in cores)
    # The staged core for the first entry, and the rejected one, are gone.
    assert len(live_storages) == baseline
    _close_module(model)


@needs_native
def test_closed_destination_is_rejected_before_any_mutation():
    model = _Stateful()
    version = model.gamma.version
    gamma_core = model.gamma._core
    model.running_var.close()

    with pytest.raises(RuntimeError, match="running_var"):
        _native_state.replace_native_state([
            _entry("gamma", model.gamma, [5.0, 5.0, 5.0]),
            _entry("running_var", model.running_var, [5.0, 5.0, 5.0]),
        ])

    assert np.array_equal(model.gamma.to_numpy(), np.asarray([1.0, 1.0, 1.0]))
    assert model.gamma.version == version
    assert model.gamma._core is gamma_core
    assert gamma_core._closed is False
    model.gamma.close()
    model.running_mean.close()


@needs_native
def test_borrowing_view_destination_is_rejected():
    model = _Stateful()
    view = model.running_mean.reshape((3, 1))
    with pytest.raises(ValueError, match="does not own its core"):
        _native_state.replace_native_state([_entry("view", view, [1.0, 2.0, 3.0])])
    view.close()
    _close_module(model)


@needs_native
def test_a_replacement_core_sharing_destination_storage_is_rejected():
    model = _Stateful()
    mean = model.running_mean
    original = mean._core
    with pytest.raises(ValueError, match="shares storage"):
        _native_state.replace_native_state([
            _native_state.NativeStateEntry(
                label="running_mean", destination=mean,
                make_core=lambda: original, source=object(),
            )
        ])
    assert mean._core is original
    assert original._closed is False
    _close_module(model)


@needs_native
def test_non_callable_factory_is_rejected_before_mutation():
    model = _Stateful()
    core = model.running_mean._core
    with pytest.raises(TypeError, match="must be callable"):
        _native_state.replace_native_state([
            _native_state.NativeStateEntry(
                label="running_mean", destination=model.running_mean,
                make_core="not a factory", source=object(),
            )
        ])
    assert model.running_mean._core is core
    assert core._closed is False
    _close_module(model)


# --------------------------------------------------------------------------
# Staging failure
# --------------------------------------------------------------------------

@needs_native
def test_staging_failure_after_one_staged_core_changes_nothing(
    monkeypatch, live_storages
):
    model = _Stateful()
    baseline = len(live_storages)
    version = model.gamma.version
    cores = (model.gamma._core, model.running_mean._core,
             model.running_var._core)
    values = [t.to_numpy().copy()
              for t in (model.gamma, model.running_mean, model.running_var)]

    real_stage = _native_state._stage_entry
    calls = {"n": 0}

    def failing_stage(planned):
        calls["n"] += 1
        if calls["n"] == 2:
            raise MemoryError("forced staging failure")
        return real_stage(planned)

    monkeypatch.setattr(_native_state, "_stage_entry", failing_stage)
    with pytest.raises(MemoryError, match="forced staging failure"):
        _native_state.replace_native_state([
            _entry("gamma", model.gamma, [8.0, 8.0, 8.0]),
            _entry("running_mean", model.running_mean, [8.0, 8.0, 8.0]),
            _entry("running_var", model.running_var, [8.0, 8.0, 8.0]),
        ])
    monkeypatch.undo()

    # Every destination is untouched, every original core is still open,
    # and no version moved.
    for tensor, core, value in zip(
        (model.gamma, model.running_mean, model.running_var), cores, values
    ):
        assert tensor._core is core
        assert core._closed is False
        assert np.array_equal(tensor.to_numpy(), value)
    assert model.gamma.version == version
    # The one successfully staged core was closed exactly once.
    assert len(live_storages) == baseline
    # A clean retry still works.
    _native_state.replace_native_state(
        [_entry("gamma", model.gamma, [8.0, 8.0, 8.0])]
    )
    assert np.array_equal(model.gamma.to_numpy(), np.asarray([8.0, 8.0, 8.0]))
    _close_module(model)


# --------------------------------------------------------------------------
# Commit failure
# --------------------------------------------------------------------------

@needs_native
def test_commit_failure_after_first_swap_rolls_everything_back(
    monkeypatch, live_storages
):
    model = _Stateful()
    baseline = len(live_storages)
    version = model.gamma.version
    cores = (model.gamma._core, model.running_mean._core)
    values = [t.to_numpy().copy() for t in (model.gamma, model.running_mean)]

    real_install = _native_state._install_core
    calls = {"n": 0}

    def failing_install(planned, new_core):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("forced commit failure")
        return real_install(planned, new_core)

    monkeypatch.setattr(_native_state, "_install_core", failing_install)
    with pytest.raises(RuntimeError, match="forced commit failure"):
        _native_state.replace_native_state([
            _entry("gamma", model.gamma, [6.0, 6.0, 6.0]),
            _entry("running_mean", model.running_mean, [6.0, 6.0, 6.0]),
        ])
    monkeypatch.undo()

    # Every destination is back on its original core, with its original
    # value; nothing was closed that is still installed.
    for tensor, core, value in zip(
        (model.gamma, model.running_mean), cores, values
    ):
        assert tensor._core is core
        assert core._closed is False
        assert np.array_equal(tensor.to_numpy(), value)
    assert model.gamma.version == version
    # Both staged cores were closed exactly once: no net storage.
    assert len(live_storages) == baseline
    # The restored storage is genuinely alive and differentiable.
    model.gamma.sum().backward()
    assert model.gamma.grad is not None
    _close_module(model)


@needs_native
def test_version_adjustment_failure_rolls_back_cores_and_versions(
    monkeypatch, live_storages
):
    """Version increments live *inside* the rollback guard, so a failure
    there is still before the commit boundary and undoes the swaps."""
    model = _Stateful()
    extra = NativeParameter([2.0, 2.0, 2.0])
    model.register_parameter("delta", extra)
    baseline = len(live_storages)
    versions = (model.gamma.version, extra.version)
    cores = (model.gamma._core, extra._core, model.running_mean._core)
    values = [t.to_numpy().copy()
              for t in (model.gamma, extra, model.running_mean)]

    real_bump = _native_state._bump_version
    calls = {"n": 0}

    def failing_bump(planned):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("forced version failure")
        return real_bump(planned)

    monkeypatch.setattr(_native_state, "_bump_version", failing_bump)
    with pytest.raises(RuntimeError, match="forced version failure"):
        _native_state.replace_native_state([
            _entry("gamma", model.gamma, [7.0, 7.0, 7.0]),
            _entry("delta", extra, [7.0, 7.0, 7.0]),
            _entry("running_mean", model.running_mean, [7.0, 7.0, 7.0]),
        ])
    monkeypatch.undo()

    # Cores restored, values unchanged, and *no* version moved — not even
    # the one that had already been incremented.
    for tensor, core, value in zip(
        (model.gamma, extra, model.running_mean), cores, values
    ):
        assert tensor._core is core
        assert core._closed is False
        assert np.array_equal(tensor.to_numpy(), value)
    assert (model.gamma.version, extra.version) == versions
    assert len(live_storages) == baseline
    _close_module(model)


# --------------------------------------------------------------------------
# Cleanup behavior — the one step past the commit boundary
# --------------------------------------------------------------------------

@needs_native
def test_successful_commit_closes_each_replaced_core_exactly_once(monkeypatch):
    model = _Stateful()
    old_cores = [model.gamma._core, model.running_mean._core]
    released = []

    real_release = _native_state._release_core

    def counting_release(core):
        released.append(id(core))
        return real_release(core)

    monkeypatch.setattr(_native_state, "_release_core", counting_release)
    _native_state.replace_native_state([
        _entry("gamma", model.gamma, [1.5, 1.5, 1.5]),
        _entry("running_mean", model.running_mean, [1.5, 1.5, 1.5]),
    ])
    monkeypatch.undo()

    assert sorted(released) == sorted(id(core) for core in old_cores)
    assert len(released) == len(set(released)) == 2
    assert all(core._closed for core in old_cores)
    # The installed cores were emphatically not closed.
    assert model.gamma._core._closed is False
    assert model.running_mean._core._closed is False
    _close_module(model)


@needs_native
def test_cleanup_failure_reports_the_committed_state_and_chains_the_cause(
    monkeypatch
):
    """Past the commit boundary the transaction cannot roll back. It must
    still attempt every release (so ownership is never ambiguous), then
    say plainly that the state change succeeded — never silently swallow
    the failure and never pretend the load failed."""
    model = _Stateful()
    old_cores = [model.gamma._core, model.running_mean._core]

    real_release = _native_state._release_core
    calls = {"n": 0}

    def failing_release(core):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("forced cleanup failure")
        return real_release(core)

    monkeypatch.setattr(_native_state, "_release_core", failing_release)
    with pytest.raises(RuntimeError, match="committed successfully") as info:
        _native_state.replace_native_state([
            _entry("gamma", model.gamma, [2.5, 2.5, 2.5]),
            _entry("running_mean", model.running_mean, [2.5, 2.5, 2.5]),
        ])
    monkeypatch.undo()

    # The primary cause is chained, not concealed.
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "forced cleanup failure" in str(info.value.__cause__)
    # The commit really did happen: both values are the new ones and the
    # parameter version moved.
    assert np.array_equal(model.gamma.to_numpy(), np.asarray([2.5, 2.5, 2.5]))
    assert np.array_equal(model.running_mean.to_numpy(),
                          np.asarray([2.5, 2.5, 2.5]))
    assert model.gamma.version == 1
    # Every replaced core was *attempted* — the second one really closed,
    # so a single failure does not abandon the rest.
    assert calls["n"] == 2
    assert old_cores[1]._closed is True
    # Explicit cleanup of the one core whose release failed.
    old_cores[0].close()
    _close_module(model)


# --------------------------------------------------------------------------
# Aliases and shared destinations
# --------------------------------------------------------------------------

@needs_native
def test_aliased_parameter_is_replaced_once_and_versioned_once(live_storages):
    model = _Stateful()
    alias = model.gamma
    model.register_parameter("gamma_alias", alias)
    baseline = len(live_storages)
    version = alias.version
    old_core = alias._core
    shared_source = object()

    replaced = _native_state.replace_native_state([
        _entry("gamma", alias, [4.0, 4.0, 4.0], source=shared_source),
        _entry("gamma_alias", alias, [4.0, 4.0, 4.0], source=shared_source),
    ])

    # One destination, one swap, one version increment, one release.
    assert replaced == 1
    assert alias.version == version + 1
    assert old_core._closed is True
    assert np.array_equal(alias.to_numpy(), np.asarray([4.0, 4.0, 4.0]))
    assert model.gamma is alias and model.gamma_alias is alias
    # Only one replacement core was created and installed.
    assert len(live_storages) == baseline
    _close_module(model)


@needs_native
def test_aliased_buffer_is_replaced_once(live_storages):
    model = _Stateful()
    shared = model.running_mean
    model.register_buffer("mirror", shared)
    baseline = len(live_storages)
    old_core = shared._core
    shared_source = object()

    replaced = _native_state.replace_native_state([
        _entry("running_mean", shared, [3.0, 3.0, 3.0], source=shared_source),
        _entry("mirror", shared, [3.0, 3.0, 3.0], source=shared_source),
    ])

    assert replaced == 1
    assert old_core._closed is True
    assert model.running_mean is shared and model.mirror is shared
    assert np.array_equal(shared.to_numpy(), np.asarray([3.0, 3.0, 3.0]))
    assert len(live_storages) == baseline
    _close_module(model)


@needs_native
def test_conflicting_values_for_one_destination_are_rejected(live_storages):
    model = _Stateful()
    baseline = len(live_storages)
    version = model.gamma.version
    core = model.gamma._core

    with pytest.raises(ValueError, match="conflicting replacement values"):
        _native_state.replace_native_state([
            _entry("gamma", model.gamma, [1.0, 1.0, 1.0]),
            _entry("gamma_alias", model.gamma, [2.0, 2.0, 2.0]),
        ])

    # Rejected during planning: nothing staged, nothing swapped.
    assert model.gamma._core is core
    assert core._closed is False
    assert model.gamma.version == version
    assert len(live_storages) == baseline
    _close_module(model)


@needs_native
def test_one_source_loaded_into_two_destinations_gives_independent_cores():
    model = _Stateful()
    source = NativeTensor.from_array([5.0, 5.0, 5.0])

    def factory():
        return cpp.NativeTensorCore.from_array(source.to_numpy())

    _native_state.replace_native_state([
        _native_state.NativeStateEntry("running_mean", model.running_mean,
                                       factory, source),
        _native_state.NativeStateEntry("running_var", model.running_var,
                                       factory, source),
    ])

    mean, var = model.running_mean, model.running_var
    assert np.array_equal(mean.to_numpy(), np.asarray([5.0, 5.0, 5.0]))
    assert np.array_equal(var.to_numpy(), np.asarray([5.0, 5.0, 5.0]))
    # Distinct destinations never share a core or its storage.
    assert mean._core is not var._core
    assert mean._core.storage is not var._core.storage
    assert source._core.storage is not mean._core.storage
    source.close()
    _close_module(model)


@needs_native
def test_a_shared_core_offered_to_two_destinations_is_rejected(live_storages):
    """Two destinations must never end up owning the same core: that
    would make close() a double free. The offending core was already
    staged by the first entry, so the transaction owns it and must close
    it exactly once — not twice, and not zero times."""
    model = _Stateful()
    mean_core, var_core = model.running_mean._core, model.running_var._core
    baseline = len(live_storages)
    shared = cpp.NativeTensorCore.from_array(np.asarray([1.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match="shares storage with another"):
        _native_state.replace_native_state([
            _native_state.NativeStateEntry(
                "running_mean", model.running_mean, lambda: shared, object()),
            _native_state.NativeStateEntry(
                "running_var", model.running_var, lambda: shared, object()),
        ])

    assert model.running_mean._core is mean_core
    assert model.running_var._core is var_core
    assert mean_core._closed is False and var_core._closed is False
    # `shared` was staged once and released once by the rollback, so the
    # live count is back where it was before `shared` was allocated.
    assert shared._closed is True
    assert len(live_storages) == baseline
    _close_module(model)


@needs_native
def test_a_replacement_aliasing_another_live_destination_is_rejected():
    """A factory returning a *different* destination's live core must be
    rejected without that live core being closed."""
    model = _Stateful()
    mean_core, var_core = model.running_mean._core, model.running_var._core

    with pytest.raises(ValueError, match="another destination's live value"):
        _native_state.replace_native_state([
            _native_state.NativeStateEntry(
                "running_mean", model.running_mean,
                lambda: var_core, object()),
            _native_state.NativeStateEntry(
                "running_var", model.running_var,
                lambda: cpp.NativeTensorCore.from_array(
                    np.asarray([1.0, 1.0, 1.0])), object()),
        ])

    # Both destinations keep their cores, and neither was closed.
    assert model.running_mean._core is mean_core
    assert model.running_var._core is var_core
    assert mean_core._closed is False and var_core._closed is False
    assert np.array_equal(model.running_var.to_numpy(),
                          np.asarray([1.0, 1.0, 1.0]))
    _close_module(model)


# --------------------------------------------------------------------------
# Reentrancy guard
# --------------------------------------------------------------------------

@needs_native
def test_a_destination_changed_during_staging_aborts_before_mutation():
    model = _Stateful()
    mean = model.running_mean
    original = mean._core
    intruder = cpp.NativeTensorCore.from_array(np.asarray([9.0, 9.0, 9.0]))

    def sneaky():
        # Simulates a reentrant caller or signal handler swapping the
        # destination's core between planning and commit.
        mean._core = intruder
        return cpp.NativeTensorCore.from_array(np.asarray([1.0, 1.0, 1.0]))

    with pytest.raises(RuntimeError, match="changed while its replacement"):
        _native_state.replace_native_state([
            _native_state.NativeStateEntry("running_mean", mean, sneaky,
                                           object())
        ])

    # The transaction installed nothing; the intruding core is still what
    # the intruder put there, and the staged core was released.
    assert mean._core is intruder
    assert original._closed is False
    mean._core = original
    intruder.close()
    _close_module(model)


# --------------------------------------------------------------------------
# load_state_dict regression — the refactor must be behavior-preserving
# --------------------------------------------------------------------------

@needs_native
def test_load_state_dict_still_loads_parameters_and_persistent_buffers():
    model = _Stateful()
    donor = _Stateful()
    donor.gamma.copy_value_(NativeTensor.from_array([2.0, 3.0, 4.0]))
    _native_state.replace_native_state([
        _entry("running_mean", donor.running_mean, [1.0, 2.0, 3.0]),
        _entry("running_var", donor.running_var, [4.0, 5.0, 6.0]),
    ])
    state = donor.state_dict()

    gamma, mean, var = model.gamma, model.running_mean, model.running_var
    version = gamma.version
    result = model.load_state_dict(state)

    assert result.missing_keys == () and result.unexpected_keys == ()
    assert np.array_equal(gamma.to_numpy(), np.asarray([2.0, 3.0, 4.0]))
    assert np.array_equal(mean.to_numpy(), np.asarray([1.0, 2.0, 3.0]))
    assert np.array_equal(var.to_numpy(), np.asarray([4.0, 5.0, 6.0]))
    # Identity preserved for parameters and buffers alike.
    assert model.gamma is gamma
    assert model.running_mean is mean and model.running_var is var
    # Exactly one version move, and only for the parameter.
    assert gamma.version == version + 1
    for snapshot in state.values():
        snapshot.close()
    _close_module(model)
    _close_module(donor)


@needs_native
def test_load_state_dict_remains_strict_and_atomic_across_both_categories():
    model = _Stateful()
    gamma, mean = model.gamma, model.running_mean
    version = gamma.version
    before = [t.to_numpy().copy() for t in (gamma, mean, model.running_var)]

    # Missing key under strict=True.
    with pytest.raises(ValueError, match="missing"):
        model.load_state_dict({"gamma": NativeTensor.from_array(
            [9.0, 9.0, 9.0])})
    # Unexpected key under strict=True.
    state = model.state_dict()
    state["nope"] = NativeTensor.from_array([0.0])
    with pytest.raises(ValueError, match="unexpected"):
        model.load_state_dict(state)
    # A bad buffer value aborts the whole load, parameters included.
    state.pop("nope").close()
    state["running_var"].close()
    state["running_var"] = NativeTensor.from_array([1.0, 1.0])
    with pytest.raises(ValueError, match="shape mismatch for 'running_var'"):
        model.load_state_dict(state)

    for tensor, value in zip((gamma, mean, model.running_var), before):
        assert np.array_equal(tensor.to_numpy(), value)
    assert gamma.version == version
    for snapshot in state.values():
        snapshot.close()
    _close_module(model)


@needs_native
def test_load_state_dict_does_not_load_non_persistent_buffers():
    model = _Stateful()
    model.register_buffer("scratch", NativeTensor.from_array([0.0]),
                          persistent=False)
    state = model.state_dict()

    assert "scratch" not in state
    with pytest.raises(ValueError, match="unexpected"):
        model.load_state_dict({**state, "scratch": NativeTensor.from_array(
            [1.0])})
    assert np.array_equal(model.scratch.to_numpy(), np.asarray([0.0]))
    for snapshot in state.values():
        snapshot.close()
    _close_module(model)


@needs_native
def test_load_state_dict_leaves_no_net_storage(live_storages):
    model = _Stateful()
    state = model.state_dict()
    baseline = len(live_storages)

    model.load_state_dict(state)

    assert len(live_storages) == baseline
    for snapshot in state.values():
        snapshot.close()
    _close_module(model)


# --------------------------------------------------------------------------
# Capability inventory and privacy
# --------------------------------------------------------------------------

def test_state_support_reports_persistent_buffers_exactly():
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers",
        "state_dict", "load_state_dict",
        "save_native_checkpoint", "load_native_checkpoint",
    )
    assert cpp.backend_info()["state_support"] == cpp.STATE_SUPPORT
    # The advertised capability maps to a real API.
    for attribute in ("register_buffer", "buffers", "named_buffers"):
        assert callable(getattr(NativeModule, attribute))


def test_f1_changed_no_other_capability_inventory():
    # F1 itself changed only STATE_SUPPORT. These tuples then moved in the
    # *later* milestones: F2 shipped NativeLayerNorm and F3 shipped
    # NativeBatchNorm1d (both composed modules), so both joined
    # NATIVE_MODULES and "layernorm" left UNSUPPORTED — while "batchnorm"
    # stays there until F4 ships the NCHW shape. The values below are the
    # current live registry.
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d",
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.UNSUPPORTED == (
        "batchnorm", "dropout", "float32", "cuda", "amp",
    )
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    # No normalization name entered any operation inventory.
    for name in ("layer_norm", "batch_norm", "layernorm", "batchnorm",
                 "persistent_buffers"):
        assert name not in cpp.TENSOR_CORE_OPS
        assert name not in cpp.AUTOGRAD_OPS
        assert name not in cpp.RAW_KERNELS


def test_the_transaction_helper_stays_private():
    import tensorforge.experimental as experimental

    assert "_native_state" not in experimental.__all__
    assert "replace_native_state" not in experimental.__all__
    assert "NativeStateEntry" not in experimental.__all__
    assert not hasattr(experimental, "replace_native_state")
    # And F1 added no public in-place mutation API to NativeTensor.
    for banned in ("replace_", "set_value_", "copy_", "fill_", "assign_"):
        assert not hasattr(NativeTensor, banned), banned
    # copy_value_ remains the one public controlled-mutation primitive,
    # and it is a parameter-only capability.
    assert callable(NativeParameter.copy_value_)
    assert not hasattr(NativeTensor, "copy_value_")


@needs_native
def test_f1_added_no_normalization_module_or_operation():
    import tensorforge.experimental as experimental

    # NativeLayerNorm shipped at the *later* milestone F2 and
    # NativeBatchNorm1d at F3 — both modules composed from existing
    # operations; the NCHW BatchNorm is still absent. None of F1, F2, or
    # F3 added a normalization *operation*, Core method, or C ABI symbol.
    for module in ("NativeBatchNorm2d",):
        assert not hasattr(experimental, module), module
        assert module not in experimental.__all__, module
        assert module not in cpp.NATIVE_MODULES, module
    assert "batchnorm" in cpp.UNSUPPORTED
    for operation in ("layer_norm", "batch_norm", "normalize"):
        assert not hasattr(NativeTensor, operation), operation
        assert not hasattr(cpp.NativeTensorCore, operation), operation
    for symbol in ("tf_core_layer_norm", "tf_core_batch_norm"):
        assert symbol not in cpp._CHECKED_KERNELS, symbol
