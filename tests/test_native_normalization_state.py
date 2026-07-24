"""Phase F, milestone F5 — Normalization state, checkpoint, and
graph-safety hardening.

F5 adds **no** numerical behavior and **no** public capability. Its whole
purpose is to prove — by executable test rather than by prose — that the
stateful normalization surface F3/F4 already shipped obeys the locked
contracts of §7 (mutable-buffer graph safety), §8 (the atomic two-buffer
transaction), §9 (ownership and lifetime), and §10 (state and
checkpoints) of ``docs/native_normalization_design.md``.

The individual module milestones proved the numerics, the finite
differences, and the single-module snapshot/transaction rules. This file
deliberately does **not** repeat those matrices. It proves the
*interactions and failure boundaries* they could not: canonical dotted
state keys across nested and shared models, snapshot independence in both
directions, strict/non-strict buffer-key handling, exact (never casting)
metadata validation, identity-preserving mixed loads, mixed
parameter/buffer transaction rollback, the version-1 checkpoint schema
gaining no normalization field, exact eval-output reproduction across a
round trip, the buffer-only-versus-full stale-graph distinction, the
corrupt/staging/save failure boundaries, eval-graph structural safety
under ``retain_graph`` and a failed retryable backward, and the
live-storage baseline over the whole matrix.

Nothing here is a normalization kernel, Core method, C ABI symbol, or
``NativeTensor`` operation, and F5 introduces none: it is tests and
documentation only.
"""

import gc
import json
import os

import numpy as np
import pytest

import tensorforge as tf
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeBatchNorm1d, NativeBatchNorm2d, NativeConv2d, NativeFlatten,
    NativeLayerNorm, NativeLinear, NativeMaxPool2d, NativeModule,
    NativeParameter, NativeReLU, NativeSequential, NativeTensor,
    load_native_checkpoint, save_native_checkpoint,
)
from tensorforge.experimental import _native_state
from tensorforge.experimental import native_checkpoint
from tensorforge.experimental import native_module

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="the experimental C++ backend is not built"
)


# ==========================================================================
# Instrumentation, helpers, fixtures
# ==========================================================================

@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — the project's
    deterministic native-allocation instrumentation. The count is exact
    (it hooks close()), so a test can read a truthful baseline without
    relying on GC timing."""
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


def _collect():
    """Drive the composed autograd graph's intermediate wrappers — which
    participate in reference cycles, a property of the Python-managed
    native autograd engine, not of normalization — to their deterministic
    collection point. The live-storage count is exact regardless."""
    gc.collect()


class _Boom(RuntimeError):
    """A distinctive injected-failure marker."""


_BN = {"1d": NativeBatchNorm1d, "2d": NativeBatchNorm2d}


def _bn(kind, num_features=3, **kwargs):
    return _BN[kind](num_features, **kwargs)


def _bn_input(kind, num_features=3, batch=4, seed=0):
    """A native input of the shape a BatchNorm of ``kind`` accepts."""
    rng = np.random.default_rng(seed)
    if kind == "1d":
        return rng.standard_normal((batch, num_features))
    return rng.standard_normal((batch, num_features, 3, 3))


def _stat_shape(kind, num_features=3):
    """The per-channel broadcast (snapshot) shape for ``kind``."""
    return (1, num_features) if kind == "1d" else (1, num_features, 1, 1)


def _load_state(module, values):
    """Load specific state keys through the public atomic loader
    (identity preserved), leaving unspecified entries alone."""
    tensors = {
        key: NativeTensor.from_array(np.asarray(value, dtype=np.float64))
        for key, value in values.items()
    }
    module.load_state_dict(tensors, strict=False)
    for tensor in tensors.values():
        tensor.close()


def _close_all(module):
    """The §9 consequence of there being no ``NativeModule.close()``: a
    stateful module's owner releases **both** its parameters and its
    buffers explicitly. ``parameters()``/``buffers()`` deduplicate by
    identity, so shared state closes exactly once."""
    for tensor in module.parameters():
        tensor.close()
    for tensor in module.buffers():
        tensor.close()


def _state_values(module):
    """Every state entry's value as an independent NumPy array."""
    return {
        name: tensor.to_numpy().copy()
        for name, tensor in module._state_named_tensors()
    }


def _graph_objects(root):
    """Every object reachable from ``root`` through the autograd graph:
    the node, its parents transitively, and every native object a node's
    history owns (``_graph_resources``). The structural walk §7 is proved
    against."""
    seen = {}

    def visit(node):
        if id(node) in seen:
            return
        seen[id(node)] = node
        for parent in node._parents:
            visit(parent)
        for resource in node._graph_resources:
            seen[id(resource)] = resource

    visit(root)
    return seen


def _graph_storage_ids(root):
    """The ids of every native **storage** the graph can reach — stronger
    than the object walk, because a borrowing view of a registered buffer
    is a different object over the *same* storage, which §7 forbids just
    as firmly."""
    ids = set()
    for obj in _graph_objects(root).values():
        if isinstance(obj, NativeTensor) and not obj.closed:
            ids.add(id(obj._core.storage))
    return ids


# -- state and archive introspection helpers (tests only) ------------------

def _manifest_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive["manifest"].tobytes().decode("utf-8"))


def _arrays_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _resave(path, arrays):
    """Rewrite ``path`` as an NPZ holding exactly ``arrays`` — the
    corruption-construction helper (tests only)."""
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)


def _tamper(source, target, mutate_manifest=None, mutate_arrays=None):
    """Copy the checkpoint at ``source`` to ``target`` with the given
    manifest/array mutations applied — the same construction
    ``test_native_checkpoint`` uses, focused here on the persistent
    normalization keys."""
    arrays = _arrays_of(source)
    manifest = json.loads(arrays.pop("manifest").tobytes().decode("utf-8"))
    if mutate_manifest is not None:
        result = mutate_manifest(manifest)
        manifest = result if result is not None else manifest
    if mutate_arrays is not None:
        mutate_arrays(arrays)
    arrays["manifest"] = np.frombuffer(
        json.dumps(manifest).encode("utf-8"), dtype=np.uint8
    )
    _resave(target, arrays)
    return target


def _fingerprint(module):
    """Everything a rejected load must leave exactly as it was: every
    state object's identity, core, value, and — for parameters — version
    and gradient."""
    record = {}
    for name, tensor in module._state_named_tensors():
        is_param = isinstance(tensor, NativeParameter)
        record[name] = {
            "id": id(tensor),
            "core": tensor._core,
            "value": tensor.to_numpy().copy(),
            "version": tensor.version if is_param else None,
            "grad_id": None if not is_param or tensor.grad is None
                       else id(tensor.grad),
            "grad_value": None if not is_param or tensor.grad is None
                          else tensor.grad.to_numpy().copy(),
        }
    return record


def _assert_untouched(module, fingerprint):
    for name, tensor in module._state_named_tensors():
        before = fingerprint[name]
        assert id(tensor) == before["id"], name
        assert tensor._core is before["core"], name
        assert not tensor.closed, name
        assert np.array_equal(tensor.to_numpy(), before["value"]), name
        if isinstance(tensor, NativeParameter):
            assert tensor.version == before["version"], name
            if before["grad_id"] is None:
                assert tensor.grad is None, name
            else:
                assert id(tensor.grad) == before["grad_id"], name
                assert np.array_equal(
                    tensor.grad.to_numpy(), before["grad_value"]
                ), name


# ==========================================================================
# Test-only model fixtures (never production classes)
# ==========================================================================

class Nested1D(NativeModule):
    """A NativeBatchNorm1d reached through a child ``NativeSequential`` —
    so the canonical state keys are genuinely dotted
    (``block.1.running_mean`` and friends), and a forward is possible."""

    def __init__(self):
        super().__init__()
        self.block = NativeSequential(
            NativeLinear(4, 3, seed=1),
            NativeBatchNorm1d(3),
            NativeReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Nested2D(NativeModule):
    """A NativeBatchNorm2d beside the Phase-D CNN modules, enough to prove
    NCHW buffer state and eval restoration end to end."""

    def __init__(self):
        super().__init__()
        self.features = NativeSequential(
            NativeConv2d(2, 3, 3, padding=1, seed=2),
            NativeBatchNorm2d(3),
            NativeReLU(),
            NativeMaxPool2d(2),
        )

    def forward(self, x):
        return self.features(x)


class MixedNorm(NativeModule):
    """Parameter-only, stateless-normalization, and persistent-buffer
    state coexisting: an ordinary trainable ``NativeLinear``, a stateless
    ``NativeLayerNorm``, and both stateful BatchNorm shapes. No single
    forward runs the whole thing (the shapes differ deliberately); it is a
    state/checkpoint fixture."""

    def __init__(self):
        super().__init__()
        self.linear = NativeLinear(4, 3, seed=0)
        self.ln = NativeLayerNorm(3)
        self.bn1 = NativeBatchNorm1d(3)
        self.bn2 = NativeBatchNorm2d(2)


MIXED_PARAMETER_KEYS = [
    "linear.weight", "linear.bias", "ln.weight", "ln.bias",
    "bn1.gamma", "bn1.beta", "bn2.gamma", "bn2.beta",
]
MIXED_BUFFER_KEYS = [
    "bn1.running_mean", "bn1.running_var",
    "bn2.running_mean", "bn2.running_var",
]
MIXED_STATE_KEYS = MIXED_PARAMETER_KEYS + MIXED_BUFFER_KEYS


class _SharedChild(NativeModule):
    def __init__(self):
        super().__init__()
        self.bn = NativeBatchNorm1d(3)


class SharedModule(NativeModule):
    """The same BatchNorm-bearing child registered under two names — an
    alias, so identity-deduplicated traversal must yield each parameter
    and buffer once, under the first-discovered path."""

    def __init__(self):
        super().__init__()
        shared = _SharedChild()
        self.left = shared
        self.right = shared


class BufferAliasModule(NativeModule):
    """A BatchNorm's exact ``running_mean`` object registered a second
    time under a different buffer name — an exact buffer alias, plus a
    non-persistent scratch buffer that state must skip."""

    def __init__(self):
        super().__init__()
        self.bn = NativeBatchNorm1d(3)
        # Register the alias first so the root-level name is discovered
        # before the child's — first-discovered path wins.
        self.register_buffer("mirror_mean", self.bn.running_mean,
                             persistent=True)
        self.register_buffer("scratch", NativeTensor.zeros((3,)),
                             persistent=False)


class SelfReferentialNorm(NativeModule):
    """A module holding a BatchNorm and a reference cycle to itself, to
    prove cycle-safe traversal terminates and yields state once."""

    def __init__(self):
        super().__init__()
        self.bn = NativeBatchNorm1d(3)
        self.loop = self


# ==========================================================================
# 1. Canonical state-key contract
# ==========================================================================

@needs_native
@pytest.mark.parametrize("kind,expected", [
    ("1d", ["gamma", "beta", "running_mean", "running_var"]),
    ("2d", ["gamma", "beta", "running_mean", "running_var"]),
])
def test_single_batchnorm_state_key_order(kind, expected):
    module = _bn(kind)
    assert [name for name, _ in module.named_parameters()] == ["gamma", "beta"]
    assert [name for name, _ in module.named_buffers()] == [
        "running_mean", "running_var"
    ]
    assert list(module.state_dict()) == expected
    for snapshot in module.state_dict().values():
        snapshot.close()
    # Parameters first, persistent buffers second, and the two never
    # share a canonical key.
    params = {name for name, _ in module.named_parameters()}
    buffers = {name for name, _ in module.named_buffers()}
    assert params.isdisjoint(buffers)
    _close_all(module)


@needs_native
def test_nested_1d_state_keys_are_dotted_and_ordered():
    model = Nested1D()
    assert list(model.state_dict()) == [
        "block.0.weight", "block.0.bias",
        "block.1.gamma", "block.1.beta",
        "block.1.running_mean", "block.1.running_var",
    ]
    for snapshot in model.state_dict().values():
        snapshot.close()
    _close_all(model)


@needs_native
def test_nested_2d_state_keys_are_dotted_and_ordered():
    model = Nested2D()
    assert list(model.state_dict()) == [
        "features.0.weight", "features.0.bias",
        "features.1.gamma", "features.1.beta",
        "features.1.running_mean", "features.1.running_var",
    ]
    for snapshot in model.state_dict().values():
        snapshot.close()
    _close_all(model)


@needs_native
def test_mixed_model_state_keys_cover_every_category_in_order():
    model = MixedNorm()
    assert [name for name, _ in model.named_parameters()] == MIXED_PARAMETER_KEYS
    assert [name for name, _ in model.named_buffers()] == MIXED_BUFFER_KEYS
    assert list(model.state_dict()) == MIXED_STATE_KEYS
    for snapshot in model.state_dict().values():
        snapshot.close()
    # Parameters and buffers never share a canonical key, and no key repeats.
    assert len(set(MIXED_STATE_KEYS)) == len(MIXED_STATE_KEYS)
    assert set(MIXED_PARAMETER_KEYS).isdisjoint(MIXED_BUFFER_KEYS)
    _close_all(model)


@needs_native
def test_state_keys_are_deterministic_across_repeated_calls():
    for build in (lambda: _bn("1d"), lambda: _bn("2d"), Nested1D, Nested2D,
                  MixedNorm):
        model = build()
        keys = list(model.state_dict())
        for snapshot in model.state_dict().values():
            snapshot.close()
        for _ in range(3):
            again = list(model.state_dict())
            for snapshot in model.state_dict().values():
                snapshot.close()
            assert again == keys
        _close_all(model)


@needs_native
def test_non_persistent_buffer_is_excluded_from_state_but_discoverable():
    model = BufferAliasModule()
    keys = list(model.state_dict())
    assert "scratch" not in keys
    for snapshot in model.state_dict().values():
        snapshot.close()
    # It is still traversed, just never serialized.
    assert "scratch" in [name for name, _ in model.named_buffers()]
    _close_all(model)


# ==========================================================================
# 2. Shared state and aliases — first-discovered names, dedup, cycles
# ==========================================================================

@needs_native
def test_shared_child_module_yields_state_once_under_first_path():
    model = SharedModule()
    assert model.left is model.right
    # First-discovered path ("left") wins; "right.*" never appears.
    assert list(model.state_dict()) == [
        "left.bn.gamma", "left.bn.beta",
        "left.bn.running_mean", "left.bn.running_var",
    ]
    for snapshot in model.state_dict().values():
        snapshot.close()
    # Deduplicated by identity: one unique parameter/buffer each.
    assert len(model.parameters()) == 2
    assert len(model.buffers()) == 2
    _close_all(model)


@needs_native
def test_exact_buffer_alias_deduplicates_under_first_discovered_name():
    model = BufferAliasModule()
    # The root-level alias is discovered before the child's own name.
    assert model.mirror_mean is model.bn.running_mean
    buffer_names = [name for name, _ in model.named_buffers()]
    assert "mirror_mean" in buffer_names
    # The shared object is yielded once — its child name is deduplicated.
    running_means = [
        name for name, tensor in model.named_buffers()
        if tensor is model.bn.running_mean
    ]
    assert running_means == ["mirror_mean"]
    # In state, the persistent alias appears once under the winning name.
    persistent = [name for name in model.state_dict()]
    for snapshot in model.state_dict().values():
        snapshot.close()
    assert persistent.count("mirror_mean") == 1
    assert "bn.running_mean" not in persistent
    assert "bn.running_var" in persistent
    _close_all(model)


@needs_native
def test_loading_a_shared_buffer_alias_updates_the_one_object_once():
    model = BufferAliasModule()
    shared = model.bn.running_mean
    _load_state(model, {"mirror_mean": [4.0, 5.0, 6.0]})
    # One object, one update, observed through both names.
    assert model.bn.running_mean is shared
    assert model.mirror_mean is shared
    assert np.allclose(shared.to_numpy(), [4.0, 5.0, 6.0])
    _close_all(model)


@needs_native
def test_cycle_safe_traversal_yields_state_once():
    model = SelfReferentialNorm()
    assert model.loop is model
    assert list(model.state_dict()) == [
        "bn.gamma", "bn.beta", "bn.running_mean", "bn.running_var"
    ]
    for snapshot in model.state_dict().values():
        snapshot.close()
    _close_all(model)


# ==========================================================================
# 3. Independent state_dict() snapshots
# ==========================================================================

@needs_native
@pytest.mark.parametrize("build", [Nested1D, Nested2D, MixedNorm])
def test_snapshots_are_owning_graph_free_contiguous_and_metadata_matched(build):
    model = build()
    live = dict(model._state_named_tensors())
    state = model.state_dict()
    try:
        for name, snapshot in state.items():
            assert type(snapshot) is NativeTensor
            assert not isinstance(snapshot, NativeParameter)
            assert snapshot.requires_grad is False
            assert snapshot.is_leaf is True
            assert snapshot._parents == ()
            assert snapshot._backward is None
            assert snapshot.owns_core is True
            assert snapshot.contiguous is True
            # Metadata matches the live state exactly.
            assert snapshot.shape == live[name].shape
            assert snapshot.dtype == live[name].dtype == "float64"
            assert snapshot.device == live[name].device == "cpu"
            assert np.array_equal(snapshot.to_numpy(), live[name].to_numpy())
            # Independent storage from the live model.
            assert snapshot._core is not live[name]._core
            assert id(snapshot._core.storage) != id(live[name]._core.storage)
    finally:
        for snapshot in state.values():
            snapshot.close()
    _close_all(model)


@needs_native
def test_snapshots_share_no_storage_with_each_other():
    model = MixedNorm()
    state = model.state_dict()
    storages = [id(s._core.storage) for s in state.values()]
    assert len(storages) == len(set(storages))
    for snapshot in state.values():
        snapshot.close()
    _close_all(model)


@needs_native
def test_replacing_or_closing_model_state_does_not_disturb_a_snapshot():
    model = MixedNorm()
    state = model.state_dict()
    frozen = {name: snap.to_numpy().copy() for name, snap in state.items()}

    # A running-stat load and a parameter load both change live values.
    _load_state(model, {"bn1.running_mean": [9.0, 9.0, 9.0]})
    _load_state(model, {"bn1.gamma": [2.0, 2.0, 2.0]})
    for name, snapshot in state.items():
        assert np.array_equal(snapshot.to_numpy(), frozen[name]), name

    # Closing the model's state leaves earlier snapshots usable.
    _close_all(model)
    for name, snapshot in state.items():
        assert not snapshot.closed
        assert np.array_equal(snapshot.to_numpy(), frozen[name]), name
        snapshot.close()


@needs_native
def test_closing_snapshots_returns_live_storage_to_baseline(live_storages):
    model = MixedNorm()
    _collect()
    baseline = len(live_storages)
    state = model.state_dict()
    assert len(live_storages) == baseline + len(state)   # one core per entry
    # Closing a snapshot never touches the model.
    for snapshot in state.values():
        snapshot.close()
    _collect()
    assert len(live_storages) == baseline
    # The model is entirely intact and usable.
    xt = NativeParameter(_bn_input("1d", seed=5))
    out = model.bn1(xt)
    out.close()
    xt.close()
    _close_all(model)


# ==========================================================================
# 4. state_dict() snapshot failure cleanup
# ==========================================================================

@needs_native
def test_snapshot_failure_after_partial_snapshots_leaks_nothing(
    monkeypatch, live_storages
):
    """A failure after some parameter *and* some buffer snapshots exist
    must close every created snapshot, return no mapping, and leave the
    model open and unchanged — all without a gc.collect(). The private
    ``_native_copy`` seam injects the failure (no production flag)."""
    model = MixedNorm()
    fingerprint = _fingerprint(model)
    _collect()
    baseline = len(live_storages)

    real_copy = native_module._native_copy
    calls = {"n": 0}

    def failing_copy(core):
        calls["n"] += 1
        # After all 8 parameter snapshots and the first buffer snapshot
        # already exist, the 10th copy (bn1.running_var) fails — so a
        # partial mapping spanning both categories is in flight.
        if calls["n"] == 10:
            raise _Boom("injected snapshot failure")
        return real_copy(core)

    monkeypatch.setattr(native_module, "_native_copy", failing_copy)
    with pytest.raises(_Boom):
        model.state_dict()
    monkeypatch.undo()

    # Immediate, deterministic: every partial snapshot was closed.
    assert len(live_storages) == baseline
    _assert_untouched(model, fingerprint)
    # A later valid snapshot succeeds and is complete.
    state = model.state_dict()
    assert list(state) == MIXED_STATE_KEYS
    for snapshot in state.values():
        snapshot.close()
    _close_all(model)


# ==========================================================================
# 5. Strict load behavior
# ==========================================================================

def _bn_state_tensors(values):
    return {
        key: NativeTensor.from_array(np.asarray(value, dtype=np.float64))
        for key, value in values.items()
    }


@needs_native
@pytest.mark.parametrize("drop", [
    ["gamma"], ["beta"], ["running_mean"], ["running_var"],
    ["gamma", "running_var"], ["running_mean", "running_var"],
])
def test_strict_missing_keys_are_rejected_and_change_nothing(drop):
    module = _bn("1d")
    _load_state(module, {"running_mean": [1.0, 2.0, 3.0],
                         "running_var": [4.0, 5.0, 6.0]})
    fingerprint = _fingerprint(module)
    complete = dict(module.state_dict())
    for key in drop:
        complete.pop(key).close()
    with pytest.raises(ValueError) as error:
        module.load_state_dict(complete)
    message = str(error.value)
    for key in drop:
        assert key in message
    _assert_untouched(module, fingerprint)
    for snapshot in complete.values():
        snapshot.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("extra_key", ["surprise_param", "surprise_buffer"])
def test_strict_unexpected_keys_are_rejected_and_change_nothing(extra_key):
    module = _bn("2d")
    fingerprint = _fingerprint(module)
    state = dict(module.state_dict())
    state[extra_key] = NativeTensor.from_array([0.0, 0.0])
    with pytest.raises(ValueError) as error:
        module.load_state_dict(state)
    assert extra_key in str(error.value)
    _assert_untouched(module, fingerprint)
    for snapshot in state.values():
        snapshot.close()
    _close_all(module)


@needs_native
def test_strict_simultaneous_missing_and_unexpected_report_both_lists():
    module = _bn("1d")
    fingerprint = _fingerprint(module)
    state = dict(module.state_dict())
    state.pop("running_var").close()          # missing
    state["extra"] = NativeTensor.from_array([0.0])   # unexpected
    with pytest.raises(ValueError) as error:
        module.load_state_dict(state)
    message = str(error.value)
    assert "running_var" in message and "extra" in message
    _assert_untouched(module, fingerprint)
    for snapshot in state.values():
        snapshot.close()
    _close_all(module)


@needs_native
def test_strict_failure_moves_no_version_and_grows_no_storage(live_storages):
    module = _bn("1d")
    module.gamma.sum().backward()             # a live gradient to preserve
    fingerprint = _fingerprint(module)
    _collect()
    baseline = len(live_storages)
    state = dict(module.state_dict())
    state.pop("beta").close()                 # strict failure
    with pytest.raises(ValueError):
        module.load_state_dict(state)
    _assert_untouched(module, fingerprint)
    for snapshot in state.values():
        snapshot.close()
    _collect()
    assert len(live_storages) == baseline
    module.gamma.grad.close()
    _close_all(module)


# ==========================================================================
# 6. Non-strict load behavior
# ==========================================================================

@needs_native
@pytest.mark.parametrize("supplied", [
    {"running_mean": [1.0, 2.0, 3.0]},
    {"running_var": [4.0, 5.0, 6.0]},
    {"running_mean": [1.0, 2.0, 3.0], "running_var": [4.0, 5.0, 6.0]},
    {"gamma": [2.0, 2.0, 2.0], "running_mean": [7.0, 7.0, 7.0]},
])
def test_non_strict_partial_loads_only_touch_matching_entries(supplied):
    module = _bn("1d")
    before = _state_values(module)
    versions = (module.gamma.version, module.beta.version)
    identities = {name: id(t) for name, t in module._state_named_tensors()}

    tensors = _bn_state_tensors(supplied)
    result = module.load_state_dict(tensors, strict=False)

    # Matching entries loaded; everything else retained.
    for key, value in supplied.items():
        assert np.allclose(getattr(module, key).to_numpy(), value)
    for name, value in before.items():
        if name not in supplied:
            assert np.array_equal(getattr(module, name).to_numpy(), value), name
    # Only *loaded parameters* advance a version; buffers move none.
    expected_gamma = versions[0] + (1 if "gamma" in supplied else 0)
    expected_beta = versions[1] + (1 if "beta" in supplied else 0)
    assert (module.gamma.version, module.beta.version) == (
        expected_gamma, expected_beta
    )
    # Identities are preserved for every state object.
    for name, tensor in module._state_named_tensors():
        assert id(tensor) == identities[name], name
    # Missing keys are reported in canonical order; sources stay open.
    missing = tuple(
        name for name, _ in module._state_named_tensors()
        if name not in supplied
    )
    assert result.missing_keys == missing
    assert result.unexpected_keys == ()
    for tensor in tensors.values():
        assert tensor.closed is False
        tensor.close()
    _close_all(module)


@needs_native
def test_non_strict_ignores_unexpected_and_reports_input_order():
    module = _bn("1d")
    before = _state_values(module)
    extra_one = NativeTensor.from_array([1.0])
    extra_two = NativeTensor.from_array([2.0])
    tensors = {
        "running_mean": NativeTensor.from_array([5.0, 5.0, 5.0]),
        "zeta": extra_one,
        "alpha": extra_two,
    }
    result = module.load_state_dict(tensors, strict=False)
    assert np.allclose(module.running_mean.to_numpy(), [5.0, 5.0, 5.0])
    assert np.array_equal(module.running_var.to_numpy(), before["running_var"])
    assert result.unexpected_keys == ("zeta", "alpha")   # input order
    assert extra_one.closed is False and extra_two.closed is False
    for tensor in tensors.values():
        tensor.close()
    _close_all(module)


@needs_native
def test_non_strict_one_invalid_matching_entry_aborts_the_whole_subset():
    """A valid running_mean and an invalidly shaped running_var: the whole
    matching subset must validate before any mutation, so neither loads."""
    module = _bn("1d")
    before = _state_values(module)
    fingerprint = _fingerprint(module)
    tensors = {
        "running_mean": NativeTensor.from_array([5.0, 5.0, 5.0]),  # valid
        "running_var": NativeTensor.from_array([1.0, 2.0]),        # wrong shape
    }
    with pytest.raises(ValueError, match="running_var"):
        module.load_state_dict(tensors, strict=False)
    _assert_untouched(module, fingerprint)
    for name, value in before.items():
        assert np.array_equal(getattr(module, name).to_numpy(), value)
    for tensor in tensors.values():
        tensor.close()
    _close_all(module)


@needs_native
def test_non_strict_nested_dotted_keys_load_the_right_entries():
    model = Nested1D()
    versions = {name: p.version for name, p in model.named_parameters()}
    _load_state(model, {"block.1.running_mean": [3.0, 3.0, 3.0]})
    assert np.allclose(model.block[1].running_mean.to_numpy(), [3.0, 3.0, 3.0])
    # A buffer-only nested load moves no parameter version anywhere.
    for name, parameter in model.named_parameters():
        assert parameter.version == versions[name], name
    _close_all(model)


# ==========================================================================
# 7. Exact metadata validation — never casts, reshapes, or moves
# ==========================================================================

@needs_native
@pytest.mark.parametrize("kind,key,wrong_shape", [
    ("1d", "gamma", (2,)),
    ("1d", "beta", (4,)),
    ("1d", "running_mean", (5,)),
    ("1d", "running_var", (1, 3)),
    ("2d", "gamma", (2,)),
    ("2d", "running_var", (4,)),
])
def test_shape_mismatch_names_the_key_and_changes_nothing(kind, key, wrong_shape):
    module = _bn(kind)
    fingerprint = _fingerprint(module)
    state = dict(module.state_dict())
    state[key].close()
    state[key] = NativeTensor.from_array(np.zeros(wrong_shape))
    with pytest.raises(ValueError) as error:
        module.load_state_dict(state)
    assert key in str(error.value)
    assert "mismatch" in str(error.value).lower()
    _assert_untouched(module, fingerprint)
    for snapshot in state.values():
        snapshot.close()
    _close_all(module)


@needs_native
def test_nested_batchnorm_shape_mismatch_names_the_dotted_key():
    for model, key in ((Nested1D(), "block.1.running_mean"),
                       (Nested2D(), "features.1.running_var")):
        fingerprint = _fingerprint(model)
        state = dict(model.state_dict())
        state[key].close()
        state[key] = NativeTensor.from_array(np.zeros((9,)))
        with pytest.raises(ValueError) as error:
            model.load_state_dict(state)
        assert key in str(error.value)
        _assert_untouched(model, fingerprint)
        for snapshot in state.values():
            snapshot.close()
        _close_all(model)


@needs_native
@pytest.mark.parametrize("attr,fake", [("dtype", "float32"), ("device", "cuda")])
def test_metadata_validator_rejects_dtype_or_device_mismatch(
    monkeypatch, attr, fake
):
    """The runtime is float64/cpu only, so a genuine alternate-dtype/device
    NativeTensor cannot exist. Rather than pretend one can, this drives the
    *validator* through the narrowest possible seam: the ``dtype``/``device``
    property is temporarily swapped so exactly one source tensor (identified
    by object identity) reports the alternate tag while every other tensor —
    the destinations included — delegates to the real value. That proves the
    load path compares metadata and rejects a mismatch by the exact key, and
    never casts, reshapes, or moves. The seam is restored deterministically."""
    module = _bn("1d")
    fingerprint = _fingerprint(module)
    state = dict(module.state_dict())
    target = state["running_mean"]          # a correctly shaped, real tensor
    real_property = getattr(NativeTensor, attr)

    def faked(self, _real=real_property, _target=target, _fake=fake):
        if self is _target:
            return _fake
        return _real.fget(self)

    monkeypatch.setattr(NativeTensor, attr, property(faked))
    try:
        with pytest.raises(ValueError) as error:
            module.load_state_dict(state)
        message = str(error.value)
        assert "running_mean" in message
        assert attr in message.lower()
    finally:
        monkeypatch.undo()

    # No cast, no reshape, no movement — the whole transaction is untouched.
    _assert_untouched(module, fingerprint)
    for snapshot in state.values():
        snapshot.close()
    _close_all(module)


# ==========================================================================
# 8. Identity-preserving mixed loads
# ==========================================================================

@needs_native
def test_successful_mixed_load_preserves_every_identity_and_moves_versions():
    model = MixedNorm()
    # Live gradients on a couple of parameters, to prove they survive.
    model.linear.weight.sum().backward()
    model.bn1.gamma.sum().backward()

    before_id = {name: id(t) for name, t in model._state_named_tensors()}
    before_core = {name: t._core for name, t in model._state_named_tensors()}
    before_version = {
        name: p.version for name, p in model.named_parameters()
    }
    before_grad = {}
    for name, parameter in model.named_parameters():
        before_grad[name] = (
            None if parameter.grad is None
            else (id(parameter.grad), parameter.grad.to_numpy().copy())
        )
    before_training = {id(m): m.training for m in model.modules()}
    before_param_order = [name for name, _ in model.named_parameters()]
    before_buffer_order = [name for name, _ in model.named_buffers()]

    # Load *different* values from a donor with a distinct init.
    donor = MixedNorm()
    _load_state(donor, {
        "bn1.running_mean": [1.0, 2.0, 3.0], "bn1.running_var": [2.0, 3.0, 4.0],
        "bn2.running_mean": [0.5, -0.5], "bn2.running_var": [1.5, 2.5],
        "ln.weight": [1.1, 1.2, 1.3],
    })
    donor_state = donor.state_dict()
    donor_values = {name: s.to_numpy().copy() for name, s in donor_state.items()}
    model.load_state_dict(donor_state)
    for snapshot in donor_state.values():
        snapshot.close()

    # Every identity unchanged.
    for name, tensor in model._state_named_tensors():
        assert id(tensor) == before_id[name], name
    # Each loaded parameter's version advanced exactly once; buffers none.
    for name, parameter in model.named_parameters():
        assert parameter.version == before_version[name] + 1, name
    for name, buffer in model.named_buffers():
        assert not hasattr(buffer, "version"), name
    # Gradients unchanged by identity and value.
    for name, parameter in model.named_parameters():
        if before_grad[name] is None:
            assert parameter.grad is None, name
        else:
            grad_id, grad_value = before_grad[name]
            assert id(parameter.grad) == grad_id, name
            assert np.array_equal(parameter.grad.to_numpy(), grad_value), name
    # Training flags, registrations, and traversal order unchanged.
    assert {id(m): m.training for m in model.modules()} == before_training
    assert [name for name, _ in model.named_parameters()] == before_param_order
    assert [name for name, _ in model.named_buffers()] == before_buffer_order
    # Every old core closed; each installed core is fresh and independent.
    for name, tensor in model._state_named_tensors():
        assert tensor._core is not before_core[name], name
        assert before_core[name]._closed is True, name
        assert tensor.owns_core is True
    installed = [id(t._core.storage) for _, t in model._state_named_tensors()]
    assert len(installed) == len(set(installed))
    # Values match the donor exactly.
    for name, tensor in model._state_named_tensors():
        assert np.array_equal(tensor.to_numpy(), donor_values[name]), name

    model.linear.weight.grad.close()
    model.bn1.gamma.grad.close()
    _close_all(model)
    _close_all(donor)


# ==========================================================================
# 9. Mixed parameter/buffer transaction failures (the F1 primitive)
# ==========================================================================

def _mixed_bn():
    """A module whose load transaction genuinely mixes a parameter and two
    persistent buffers in one call — the smallest fixture that exercises
    every rollback branch."""
    return _bn("1d")


@needs_native
@pytest.mark.parametrize("seam,call,error", [
    ("_stage_entry", 2, MemoryError),      # staging after ≥1 copy exists
    ("_install_core", 1, RuntimeError),    # first destination install
    ("_install_core", 3, RuntimeError),    # a later install after swaps
    ("_bump_version", 1, RuntimeError),    # version adjustment
])
def test_precommit_transaction_failures_roll_everything_back(
    monkeypatch, live_storages, seam, call, error
):
    module = _mixed_bn()
    # A load that touches a parameter *and* both buffers together.
    tensors = _bn_state_tensors({
        "gamma": [3.0, 3.0, 3.0],
        "running_mean": [1.0, 2.0, 3.0],
        "running_var": [4.0, 5.0, 6.0],
    })
    fingerprint = _fingerprint(module)
    _collect()
    baseline = len(live_storages)

    real = getattr(_native_state, seam)
    calls = {"n": 0}

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == call:
            raise error(f"injected {seam} failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(_native_state, seam, failing)
    with pytest.raises(error):
        module.load_state_dict(tensors, strict=False)
    monkeypatch.undo()

    # Immediate, without gc: nothing moved, nothing leaked, cores restored.
    _assert_untouched(module, fingerprint)
    for name in ("gamma", "beta", "running_mean", "running_var"):
        assert getattr(module, name)._core is fingerprint[name]["core"]
        assert getattr(module, name)._core._closed is False
    assert len(live_storages) == baseline
    for tensor in tensors.values():
        assert tensor.closed is False
    # A later valid load and forward both succeed.
    module.load_state_dict(tensors, strict=False)
    assert np.allclose(module.gamma.to_numpy(), [3.0, 3.0, 3.0])
    for tensor in tensors.values():
        tensor.close()
    xt = NativeParameter(_bn_input("1d", seed=3))
    out = module(xt)
    out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum().backward()
    assert xt.grad is not None
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_keyboard_interrupt_between_swaps_rolls_back(monkeypatch, live_storages):
    """A ``KeyboardInterrupt`` between the two buffer swaps is a
    ``BaseException`` and must still roll the whole transaction back — no
    partial state, no leak, aliases intact."""
    module = _bn("1d")
    tensors = _bn_state_tensors({
        "running_mean": [1.0, 2.0, 3.0], "running_var": [4.0, 5.0, 6.0],
    })
    fingerprint = _fingerprint(module)
    _collect()
    baseline = len(live_storages)

    real_install = _native_state._install_core
    calls = {"n": 0}

    def interrupting(planned, new_core):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated interrupt between swaps")
        return real_install(planned, new_core)

    monkeypatch.setattr(_native_state, "_install_core", interrupting)
    with pytest.raises(KeyboardInterrupt):
        module.load_state_dict(tensors, strict=False)
    monkeypatch.undo()

    _assert_untouched(module, fingerprint)
    assert len(live_storages) == baseline
    for tensor in tensors.values():
        tensor.close()
    _close_all(module)


# ==========================================================================
# 10. Checkpoint manifest and archive schema
# ==========================================================================

@needs_native
@pytest.mark.parametrize("build", [lambda: _bn("1d"), lambda: _bn("2d"),
                                   Nested1D, Nested2D, MixedNorm])
def test_checkpoint_manifest_gains_no_normalization_field(build, tmp_path):
    model = build()
    path = os.path.join(str(tmp_path), "model.npz")
    save_native_checkpoint(path, model, metadata={"milestone": "F5"})
    manifest = _manifest_of(path)

    # The top-level manifest fields are exactly the format's — no
    # running_stats, buffers, normalization, training, num_batches_tracked,
    # rng, graph, or version section.
    assert set(manifest) == {
        "format", "format_version", "model", "optimizer", "metadata"
    }
    assert manifest["format"] == "tensorforge.native_checkpoint"
    assert manifest["format_version"] == 1
    for forbidden in ("running_stats", "buffers", "normalization",
                      "training", "num_batches_tracked", "rng", "graph"):
        assert forbidden not in manifest
    manifest_text = json.dumps(manifest)
    for banned in ("num_batches_tracked", "'version'", "\"version\"",
                   "rng_state", "training"):
        assert banned not in manifest_text, banned

    # Model section: keys match live state, every entry has exactly the
    # existing fields, and BatchNorm buffers ride as ordinary entries.
    live = dict(model._state_named_tensors())
    assert manifest["model"]["keys"] == list(model.state_dict())
    for snapshot in model.state_dict().values():
        snapshot.close()
    entries = manifest["model"]["entries"]
    assert list(entries) == manifest["model"]["keys"]
    for key, entry in entries.items():
        assert set(entry) == {"array", "shape", "dtype", "device"}
        assert tuple(entry["shape"]) == live[key].shape
        assert entry["dtype"] == "float64"
        assert entry["device"] == "cpu"
    _close_all(model)


@needs_native
def test_checkpoint_archive_arrays_are_exactly_the_state_entries(tmp_path):
    model = _bn("2d")
    _load_state(model, {"running_mean": [1.0, 2.0, 3.0],
                        "running_var": [4.0, 5.0, 6.0]})
    path = os.path.join(str(tmp_path), "bn2d.npz")
    save_native_checkpoint(path, model)
    arrays = _arrays_of(path)

    # One deterministic zero-padded array per unique state entry, plus the
    # manifest — no extra entry, and every numerical array is float64.
    assert sorted(arrays) == [
        "manifest",
        "model::000000", "model::000001", "model::000002", "model::000003",
    ]
    for name, array in arrays.items():
        if name == "manifest":
            assert array.dtype == np.uint8 and array.ndim == 1
        else:
            assert array.dtype == np.float64
            assert array.dtype != object
    # The running buffers serialize as ordinary model entries under their
    # canonical keys, matching the live shapes exactly.
    manifest = _manifest_of(path)
    for key in ("running_mean", "running_var"):
        assert tuple(manifest["model"]["entries"][key]["shape"]) == (3,)
    _close_all(model)


# ==========================================================================
# 11. Checkpoint exact eval-output reproduction
# ==========================================================================

@needs_native
@pytest.mark.parametrize("kind", ["1d", "2d"])
def test_checkpoint_round_trip_reproduces_eval_output_exactly(kind, tmp_path):
    source = _bn(kind, eps=1e-3, momentum=0.4)
    _load_state(source, {
        "gamma": [1.1, 1.2, 1.3], "beta": [-0.1, -0.2, -0.3],
        "running_mean": [0.3, -0.2, 0.5], "running_var": [2.0, 0.5, 1.25],
    })
    identities = {name: id(t) for name, t in source._state_named_tensors()}
    source.eval()
    probe = NativeTensor.from_array(_bn_input(kind, seed=11, batch=3))
    # The training-mode output is *not* enough — eval depends on running state.
    source.train()
    train_out = source(probe).to_numpy().copy()
    source.eval()
    eval_out = source(probe).to_numpy().copy()
    assert not np.allclose(train_out, eval_out, atol=1e-6)

    path = os.path.join(str(tmp_path), "eval.npz")
    save_native_checkpoint(path, source, metadata={"kind": kind})

    # A fresh, compatible target with *different* state.
    target = _bn(kind, eps=1e-3, momentum=0.4)
    _load_state(target, {"gamma": [9.0, 9.0, 9.0], "running_mean": [9, 9, 9]})
    target_ids = {name: id(t) for name, t in target._state_named_tensors()}
    target_versions = (target.gamma.version, target.beta.version)
    target.eval()

    metadata = load_native_checkpoint(path, target)
    assert metadata == {"kind": kind}

    # All four tensors restored exactly, and the eval output reproduced
    # bitwise — the operations are deterministic, so equality is exact.
    for name in ("gamma", "beta", "running_mean", "running_var"):
        assert np.array_equal(
            getattr(target, name).to_numpy(), getattr(source, name).to_numpy()
        ), name
    assert np.array_equal(target(probe).to_numpy(), eval_out)
    # Identities preserved; gamma/beta versions each advanced once on load.
    for name, tensor in target._state_named_tensors():
        assert id(tensor) == target_ids[name], name
    assert target.gamma.version == target_versions[0] + 1
    assert target.beta.version == target_versions[1] + 1
    # Training mode is runtime state — never serialized. The caller's mode
    # (eval, here) stands, and metadata is returned independently.
    assert target.training is False

    probe.close()
    _close_all(source)
    _close_all(target)


@needs_native
def test_nested_2d_checkpoint_reproduces_eval_output(tmp_path):
    """The full NCHW path: populate running stats through a real
    convolutional stack, then prove an eval forward survives a round trip
    into a fresh model bit-for-bit."""
    source = Nested2D()
    rng = np.random.default_rng(21)
    for _ in range(3):
        xt = NativeTensor.from_array(rng.standard_normal((2, 2, 5, 5)))
        source(xt).close()
        xt.close()
    source.eval()
    probe = NativeTensor.from_array(rng.standard_normal((2, 2, 5, 5)))
    expected = source(probe).to_numpy().copy()
    # The BatchNorm actually accumulated non-trivial running statistics.
    assert not np.allclose(
        source.features[1].running_mean.to_numpy(), np.zeros(3)
    )

    path = os.path.join(str(tmp_path), "nested2d.npz")
    save_native_checkpoint(path, source)
    target = Nested2D()
    load_native_checkpoint(path, target)
    target.eval()
    assert np.array_equal(target(probe).to_numpy(), expected)

    probe.close()
    _close_all(source)
    _close_all(target)


# ==========================================================================
# 12. Buffer-only checkpoint loading (the real archive path)
# ==========================================================================

class _RunningStatHolder(NativeModule):
    """**Test-only** parameter-free module aliasing existing running
    buffers as persistent buffers, so ``load_native_checkpoint()`` drives
    the real archive path over exactly those objects without touching any
    ``gamma``/``beta``. Two ``register_buffer`` calls — the aliasing
    ``NativeModule`` has always supported — and nothing else."""

    def __init__(self, running_mean, running_var):
        super().__init__()
        self.register_buffer("running_mean", running_mean, persistent=True)
        self.register_buffer("running_var", running_var, persistent=True)


@needs_native
@pytest.mark.parametrize("kind", ["1d", "2d"])
def test_buffer_only_checkpoint_replaces_exact_objects_and_spares_parameters(
    kind, tmp_path, live_storages
):
    _collect()
    baseline = len(live_storages)

    module = _bn(kind)
    _load_state(module, {"gamma": [1.5, -0.5, 2.0], "beta": [0.1, 0.2, -0.3],
                         "running_mean": [0.3, -0.2, 0.5],
                         "running_var": [2.0, 0.5, 1.25]})

    # A donor buffer-only checkpoint holding different statistics.
    donor_mean = NativeTensor.from_array([7.0, 7.0, 7.0])
    donor_var = NativeTensor.from_array([25.0, 25.0, 25.0])
    donor = _RunningStatHolder(donor_mean, donor_var)
    assert donor.parameters() == []
    path = os.path.join(str(tmp_path), "stats.npz")
    save_native_checkpoint(path, donor)
    donor_mean.close()
    donor_var.close()

    mean_object = module.running_mean
    var_object = module.running_var
    mean_core = mean_object._core
    var_core = var_object._core
    gamma_version = module.gamma.version
    beta_version = module.beta.version

    # The holder aliases the module's *own* buffer objects.
    holder = _RunningStatHolder(mean_object, var_object)
    assert holder.running_mean is module.running_mean
    metadata = load_native_checkpoint(path, holder)
    assert metadata == {}

    # The exact registered objects were replaced with the donor values.
    assert module.running_mean is mean_object
    assert module.running_var is var_object
    assert np.allclose(module.running_mean.to_numpy(), [7.0, 7.0, 7.0])
    assert np.allclose(module.running_var.to_numpy(), [25.0, 25.0, 25.0])
    # Old cores closed; no parameter stale guard could fire — no parameter
    # version moved, because this load touched no parameter.
    assert mean_object._core is not mean_core
    assert var_object._core is not var_core
    assert mean_core._closed is True and var_core._closed is True
    assert module.gamma.version == gamma_version
    assert module.beta.version == beta_version

    del holder, donor
    _close_all(module)
    _collect()
    assert len(live_storages) == baseline


@needs_native
def test_buffer_only_checkpoint_load_leaves_an_earlier_eval_graph_valid(
    tmp_path
):
    """The §7 checkpoint half: a buffer-only ``load_native_checkpoint()``
    over the module's own running buffers, performed *after* an eval
    forward, must leave that graph's backward unchanged — the graph owns
    immutable snapshots, not the buffers."""
    kind = "1d"
    module = _bn(kind)
    _load_state(module, {"running_mean": [0.3, -0.2, 0.5],
                         "running_var": [2.0, 0.5, 1.25]})
    module.eval()
    rng = np.random.default_rng(31)
    x = rng.standard_normal((4, 3))
    upstream = rng.standard_normal((4, 3))

    # The control: the same forward-time state's gradients from a clean run.
    control = _clean_eval_grads(kind, {
        "gamma": module.gamma.to_numpy(), "beta": module.beta.to_numpy(),
        "running_mean": [0.3, -0.2, 0.5], "running_var": [2.0, 0.5, 1.25],
    }, x, upstream)

    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()

    # A donor buffer-only checkpoint with *different* statistics.
    donor = _RunningStatHolder(
        NativeTensor.from_array([9.0, 9.0, 9.0]),
        NativeTensor.from_array([16.0, 16.0, 16.0]),
    )
    path = os.path.join(str(tmp_path), "buffers.npz")
    save_native_checkpoint(path, donor)
    for tensor in donor.buffers():
        tensor.close()
    holder = _RunningStatHolder(module.running_mean, module.running_var)
    load_native_checkpoint(path, holder)
    assert np.allclose(module.running_mean.to_numpy(), [9.0, 9.0, 9.0])

    # The earlier eval graph still runs and reproduces the forward-time
    # gradients — not the ones the new statistics would give.
    loss.backward()
    assert np.allclose(xt.grad.to_numpy(), control["x"], atol=1e-12)
    assert np.allclose(module.gamma.grad.to_numpy(), control["gamma"], atol=1e-12)
    assert np.allclose(module.beta.grad.to_numpy(), control["beta"], atol=1e-12)

    loss.close()
    out.close()
    xt.close()
    _close_all(module)


def _clean_eval_grads(kind, state, x, upstream):
    """The gradients a clean eval forward+backward produces for ``state``.
    A fresh module can never leak running state, and eval mode never
    mutates it, so this is a faithful forward-time control."""
    module = _bn(kind)
    _load_state(module, state)
    module.eval()
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    loss.backward()
    grads = {
        "x": xt.grad.to_numpy().copy(),
        "gamma": module.gamma.grad.to_numpy().copy(),
        "beta": module.beta.grad.to_numpy().copy(),
    }
    loss.close()
    out.close()
    xt.close()
    _close_all(module)
    return grads


# ==========================================================================
# 13. Full checkpoint versus buffer-only stale behavior
# ==========================================================================

@needs_native
def test_full_checkpoint_load_stales_the_graph_through_parameters(tmp_path):
    """A **full** BatchNorm checkpoint also replaces ``gamma``/``beta``, so
    the pre-existing v3.7 parameter-version guard legitimately stales an
    earlier eval graph. The error must be attributed to the parameter
    version, never to a running buffer, and it must commit no partial
    gradient while leaving the graph retryable."""
    kind = "1d"
    donor = _bn(kind)
    _load_state(donor, {"gamma": [3.0, 3.0, 3.0], "beta": [1.0, 1.0, 1.0],
                        "running_mean": [7.0, 7.0, 7.0],
                        "running_var": [25.0, 25.0, 25.0]})
    path = os.path.join(str(tmp_path), "full.npz")
    save_native_checkpoint(path, donor)
    _close_all(donor)

    module = _bn(kind)
    _load_state(module, {"running_mean": [0.3, -0.2, 0.5],
                         "running_var": [2.0, 0.5, 1.25]})
    module.eval()
    rng = np.random.default_rng(41)
    x = rng.standard_normal((4, 3))
    upstream = rng.standard_normal((4, 3))
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    resources = out._graph_resources
    mean_object, var_object = module.running_mean, module.running_var
    versions = (module.gamma.version, module.beta.version)

    load_native_checkpoint(path, module)
    # Buffer identities survived; the cause is the parameter versions moving.
    assert module.running_mean is mean_object
    assert module.running_var is var_object
    assert module.gamma.version == versions[0] + 1
    assert module.beta.version == versions[1] + 1

    with pytest.raises(RuntimeError, match="stale parameter value") as error:
        loss.backward()
    message = str(error.value)
    assert "NativeParameter" in message and "version" in message
    for buffer_word in ("running_mean", "running_var", "buffer"):
        assert buffer_word not in message, buffer_word
    # No partial gradient committed, graph not freed, deterministic repeat.
    assert xt.grad is None
    assert all(not resource.closed for resource in resources)
    with pytest.raises(RuntimeError, match="stale parameter value"):
        loss.backward()
    # A fresh forward after the load works normally.
    fresh = module(xt)
    fresh.multiply(NativeTensor.from_array(upstream)).sum().backward()
    assert xt.grad is not None

    loss.close()
    out.close()
    fresh.close()
    xt.close()
    _close_all(module)


# ==========================================================================
# 14. Strict checkpoint corruption matrix (normalization keys)
# ==========================================================================

def _norm_corrupt_cases(source, tmp_path):
    """Corrupt copies of a BatchNorm model-only archive, each touching the
    persistent normalization keys where it can."""
    cases = []

    def add(name, mutate_manifest=None, mutate_arrays=None):
        cases.append((name, _tamper(
            source, os.path.join(str(tmp_path), f"corrupt-{name}.npz"),
            mutate_manifest, mutate_arrays,
        )))

    # -- manifest identity
    not_zip = os.path.join(str(tmp_path), "corrupt-not-a-zip.npz")
    with open(not_zip, "w", encoding="utf-8") as handle:
        handle.write("this is not an archive")
    cases.append(("not-a-zip", not_zip))

    no_manifest = os.path.join(str(tmp_path), "corrupt-no-manifest.npz")
    arrays = _arrays_of(source)
    del arrays["manifest"]
    _resave(no_manifest, arrays)
    cases.append(("missing-manifest", no_manifest))

    bad_repr = os.path.join(str(tmp_path), "corrupt-manifest-repr.npz")
    arrays = _arrays_of(source)
    arrays["manifest"] = np.zeros((2, 2))
    _resave(bad_repr, arrays)
    cases.append(("malformed-manifest-representation", bad_repr))

    bad_utf8 = os.path.join(str(tmp_path), "corrupt-utf8.npz")
    arrays = _arrays_of(source)
    arrays["manifest"] = np.frombuffer(b"\xff\xfe{", dtype=np.uint8)
    _resave(bad_utf8, arrays)
    cases.append(("invalid-utf8", bad_utf8))

    bad_json = os.path.join(str(tmp_path), "corrupt-json.npz")
    arrays = _arrays_of(source)
    arrays["manifest"] = np.frombuffer(b"{not json", dtype=np.uint8)
    _resave(bad_json, arrays)
    cases.append(("malformed-json", bad_json))

    bad_root = os.path.join(str(tmp_path), "corrupt-root.npz")
    arrays = _arrays_of(source)
    arrays["manifest"] = np.frombuffer(b"[1, 2]", dtype=np.uint8)
    _resave(bad_root, arrays)
    cases.append(("wrong-root-type", bad_root))

    add("wrong-format", lambda m: {**m, "format": "tensorforge.checkpoint"})
    add("wrong-version", lambda m: {**m, "format_version": 2})
    add("wrong-version-type", lambda m: {**m, "format_version": "1"})
    add("missing-field",
        lambda m: {k: v for k, v in m.items() if k != "metadata"})
    add("unexpected-field", lambda m: {**m, "extra": 1})

    # -- model section, targeting the running-buffer keys
    def drop_running_mean(m):
        keys = [k for k in m["model"]["keys"] if k != "running_mean"]
        entries = {k: v for k, v in m["model"]["entries"].items()
                   if k != "running_mean"}
        return {**m, "model": {"keys": keys, "entries": entries}}

    add("missing-running-mean-key", drop_running_mean)

    def rename_running_var(m):
        keys = ["gamma", "beta", "running_mean", "surprise_buffer"]
        entries = dict(m["model"]["entries"])
        entries["surprise_buffer"] = entries.pop("running_var")
        return {**m, "model": {"keys": keys, "entries": entries}}

    add("unexpected-buffer-key", rename_running_var)

    def reorder(m):
        keys = list(reversed(m["model"]["keys"]))
        return {**m, "model": {"keys": keys, "entries": m["model"]["entries"]}}

    add("reordered-keys", reorder)

    def duplicate_reference(m):
        entries = {k: dict(v) for k, v in m["model"]["entries"].items()}
        entries["running_var"]["array"] = entries["running_mean"]["array"]
        entries["running_var"]["shape"] = entries["running_mean"]["shape"]
        return {**m, "model": {"keys": m["model"]["keys"], "entries": entries}}

    add("duplicate-buffer-reference", duplicate_reference)

    def wrong_buffer_shape(m):
        entries = {k: dict(v) for k, v in m["model"]["entries"].items()}
        entries["running_var"]["shape"] = [9]
        return {**m, "model": {"keys": m["model"]["keys"], "entries": entries}}

    add("wrong-manifest-shape-for-buffer", wrong_buffer_shape)

    def wrong_buffer_dtype(m):
        entries = {k: dict(v) for k, v in m["model"]["entries"].items()}
        entries["running_var"]["dtype"] = "float32"
        return {**m, "model": {"keys": m["model"]["keys"], "entries": entries}}

    add("wrong-manifest-dtype-for-buffer", wrong_buffer_dtype)

    def wrong_buffer_device(m):
        entries = {k: dict(v) for k, v in m["model"]["entries"].items()}
        entries["running_mean"]["device"] = "cuda"
        return {**m, "model": {"keys": m["model"]["keys"], "entries": entries}}

    add("wrong-manifest-device-for-buffer", wrong_buffer_device)

    def non_string_array(m):
        entries = {k: dict(v) for k, v in m["model"]["entries"].items()}
        entries["running_mean"]["array"] = 5
        return {**m, "model": {"keys": m["model"]["keys"], "entries": entries}}

    add("non-string-array-name", non_string_array)

    # -- archive arrays
    def missing_running_array(arrays):
        del arrays["model::000002"]           # running_mean's array

    add("missing-running-buffer-array", mutate_arrays=missing_running_array)

    def extra_array(arrays):
        arrays["surprise"] = np.zeros(3)

    add("unreferenced-extra-array", mutate_arrays=extra_array)

    def wrong_array_dtype(arrays):
        arrays["model::000003"] = arrays["model::000003"].astype(np.float32)

    add("wrong-buffer-array-dtype", mutate_arrays=wrong_array_dtype)

    def wrong_array_shape(arrays):
        # (C,) -> (1, C): a valid reshape for any feature count, but the
        # rank no longer matches the (C,) destination the manifest declares.
        arrays["model::000002"] = arrays["model::000002"].reshape(1, -1)

    add("wrong-buffer-array-shape", mutate_arrays=wrong_array_shape)

    def object_array(arrays):
        arrays["model::000002"] = np.array([{"hostile": True}], dtype=object)

    add("object-buffer-array", mutate_arrays=object_array)

    return cases


@needs_native
def test_corrupt_checkpoints_mutate_nothing(tmp_path):
    model = _bn("1d")
    _load_state(model, {"gamma": [1.5, 0.5, -2.0], "beta": [0.1, 0.2, 0.3],
                        "running_mean": [0.7, 0.8, 0.9],
                        "running_var": [1.7, 1.8, 1.9]})
    model.gamma.sum().backward()             # a live gradient to preserve
    source = os.path.join(str(tmp_path), "good.npz")
    save_native_checkpoint(source, model)

    # Confirm the good archive really has the running-buffer keys.
    manifest = _manifest_of(source)
    assert manifest["model"]["keys"] == [
        "gamma", "beta", "running_mean", "running_var"
    ]

    fingerprint = _fingerprint(model)
    for name, path in _norm_corrupt_cases(source, tmp_path):
        with pytest.raises((ValueError, TypeError)):
            load_native_checkpoint(path, model)
        _assert_untouched(model, fingerprint)

    # The same model recovers on a valid load.
    load_native_checkpoint(source, model)
    assert np.allclose(model.running_mean.to_numpy(), [0.7, 0.8, 0.9])
    model.gamma.grad.close()
    _close_all(model)


@needs_native
def test_corrupt_checkpoints_leave_no_staged_storage(tmp_path, live_storages):
    model = _bn("2d")
    source = os.path.join(str(tmp_path), "good2d.npz")
    save_native_checkpoint(source, model)
    _collect()
    baseline = len(live_storages)
    for name, path in _norm_corrupt_cases(source, tmp_path):
        with pytest.raises((ValueError, TypeError)):
            load_native_checkpoint(path, model)
    _collect()
    assert len(live_storages) == baseline
    _close_all(model)


# ==========================================================================
# 15. Checkpoint staging and commit failures
# ==========================================================================

@needs_native
def test_checkpoint_staging_failure_leaks_nothing(
    monkeypatch, tmp_path, live_storages
):
    """A failure while staging a later model tensor (phase 2) must close
    every staged NativeTensor and change no live state."""
    model = _bn("1d")
    _load_state(model, {"running_mean": [1.0, 2.0, 3.0]})
    path = os.path.join(str(tmp_path), "own.npz")
    save_native_checkpoint(path, model)
    fingerprint = _fingerprint(model)
    _collect()
    baseline = len(live_storages)

    real_from_array = NativeTensor.from_array
    calls = {"n": 0}

    def failing_from_array(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:               # a later model tensor
            raise _Boom("injected staging failure")
        return real_from_array(*args, **kwargs)

    monkeypatch.setattr(
        NativeTensor, "from_array", staticmethod(failing_from_array)
    )
    with pytest.raises(_Boom):
        load_native_checkpoint(path, model)
    monkeypatch.undo()

    _assert_untouched(model, fingerprint)
    _collect()
    assert len(live_storages) == baseline
    load_native_checkpoint(path, model)      # a valid load still works
    _close_all(model)


@needs_native
@pytest.mark.parametrize("call", [1, 3])
def test_checkpoint_commit_failure_rolls_back(
    monkeypatch, tmp_path, live_storages, call
):
    """A failure during the model ``load_state_dict`` commit — at the first
    install and at a later install after swaps — rolls the whole model
    back, leaks no staged storage, and stays retryable. (The loader's
    documented model/optimizer two-commit window is not in play here: this
    is a model-only checkpoint.)"""
    model = _bn("1d")
    path = os.path.join(str(tmp_path), "own.npz")
    save_native_checkpoint(path, model)
    fingerprint = _fingerprint(model)
    _collect()
    baseline = len(live_storages)

    real_install = _native_state._install_core
    calls = {"n": 0}

    def failing_install(planned, new_core):
        calls["n"] += 1
        if calls["n"] == call:
            raise _Boom("injected commit failure")
        return real_install(planned, new_core)

    monkeypatch.setattr(_native_state, "_install_core", failing_install)
    with pytest.raises(_Boom):
        load_native_checkpoint(path, model)
    monkeypatch.undo()

    _assert_untouched(model, fingerprint)
    _collect()
    assert len(live_storages) == baseline
    load_native_checkpoint(path, model)
    _close_all(model)


# ==========================================================================
# 16. Atomic checkpoint-save failures
# ==========================================================================

@needs_native
@pytest.mark.parametrize("where", ["state_dict", "to_numpy", "savez", "replace"])
def test_atomic_save_failure_preserves_everything(
    monkeypatch, tmp_path, where
):
    model = _bn("1d")
    _load_state(model, {"running_mean": [1.0, 2.0, 3.0],
                        "running_var": [4.0, 5.0, 6.0]})
    model.gamma.sum().backward()
    path = os.path.join(str(tmp_path), "dest.npz")
    # An existing destination that must survive byte-for-byte.
    save_native_checkpoint(path, model, metadata={"which": "original"})
    original_bytes = open(path, "rb").read()
    fingerprint = _fingerprint(model)

    def fail_state_dict(self):
        raise _Boom("injected state_dict failure")

    def fail_to_numpy(self):
        raise _Boom("injected to_numpy failure")

    real_to_numpy = NativeTensor.to_numpy
    to_numpy_calls = {"n": 0}

    def fail_later_to_numpy(self):
        to_numpy_calls["n"] += 1
        if to_numpy_calls["n"] == 3:
            raise _Boom("injected to_numpy failure")
        return real_to_numpy(self)

    def fail_savez(*args, **kwargs):
        raise _Boom("injected savez failure")

    def fail_replace(*args, **kwargs):
        raise _Boom("injected replace failure")

    if where == "state_dict":
        monkeypatch.setattr(NativeModule, "state_dict", fail_state_dict)
    elif where == "to_numpy":
        monkeypatch.setattr(NativeTensor, "to_numpy", fail_later_to_numpy)
    elif where == "savez":
        monkeypatch.setattr(np, "savez", fail_savez)
    else:
        monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(_Boom):
        save_native_checkpoint(path, model, metadata={"which": "second"})
    monkeypatch.undo()

    # Model unchanged, gradients unchanged, the existing file byte-intact,
    # and no temporary residue.
    _assert_untouched(model, fingerprint)
    assert open(path, "rb").read() == original_bytes
    assert sorted(os.listdir(str(tmp_path))) == ["dest.npz"]
    # A normal save then load succeeds after every failure.
    save_native_checkpoint(path, model, metadata={"which": "recovered"})
    assert _manifest_of(path)["metadata"] == {"which": "recovered"}
    fresh = _bn("1d")
    load_native_checkpoint(path, fresh)
    assert np.allclose(fresh.running_mean.to_numpy(), [1.0, 2.0, 3.0])
    model.gamma.grad.close()
    _close_all(model)
    _close_all(fresh)


# ==========================================================================
# 17. Eval graph structural safety
# ==========================================================================

@needs_native
@pytest.mark.parametrize("kind", ["1d", "2d"])
def test_eval_graph_holds_no_registered_buffer_object_or_storage(kind):
    module = _bn(kind)
    _load_state(module, {"running_mean": [0.5, 1.0, 1.5],
                         "running_var": [2.0, 3.0, 4.0]})
    module.eval()
    xt = NativeParameter(_bn_input(kind, seed=51))
    out = module(xt)

    reachable = _graph_objects(out)
    # Neither registered buffer object appears.
    assert id(module.running_mean) not in reachable
    assert id(module.running_var) not in reachable
    # The parameters legitimately do.
    assert id(module.gamma) in reachable and id(module.beta) in reachable
    # Not one byte the buffers own is reachable — object *and* storage.
    buffer_storage = {
        id(module.running_mean._core.storage),
        id(module.running_var._core.storage),
    }
    assert not (buffer_storage & _graph_storage_ids(out))
    # The graph-owned snapshots are independent owning graph-free storage
    # of the correct broadcast shape.
    resources = out._graph_resources
    assert len(resources) == 2
    for resource in resources:
        assert resource.owns_core is True
        assert resource.contiguous is True
        assert resource.requires_grad is False
        assert resource.is_leaf is True
        assert resource.shape == _stat_shape(kind)
        assert id(resource._core.storage) not in buffer_storage
    out.close()
    xt.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("kind", ["1d", "2d"])
def test_repeated_eval_forwards_take_independent_snapshot_storage(kind):
    module = _bn(kind)
    module.eval()
    xt = NativeParameter(_bn_input(kind, seed=53))
    out_a = module(xt)
    out_b = module(xt)
    storages_a = {id(r._core.storage) for r in out_a._graph_resources}
    storages_b = {id(r._core.storage) for r in out_b._graph_resources}
    assert storages_a and storages_b
    assert storages_a.isdisjoint(storages_b)
    out_a.close()
    out_b.close()
    xt.close()
    _close_all(module)


@needs_native
def test_no_module_attribute_retains_a_forward_snapshot():
    module = _bn("1d")
    module.eval()
    xt = NativeParameter(_bn_input("1d", seed=55))
    out = module(xt)
    # Only the four state tensors are stored on the module — no forward
    # snapshot is squirreled away as an attribute.
    for name in dir(module):
        if name.startswith("__"):
            continue
        value = getattr(module, name, None)
        if isinstance(value, NativeTensor):
            assert name in ("gamma", "beta", "running_mean", "running_var"), name
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_training_graph_holds_no_buffer_and_no_snapshot_resource():
    for kind in ("1d", "2d"):
        module = _bn(kind)
        xt = NativeParameter(_bn_input(kind, seed=57))
        out = module(xt)
        reachable = _graph_objects(out)
        assert id(module.running_mean) not in reachable
        assert id(module.running_var) not in reachable
        assert out._graph_resources == ()
        out.close()
        xt.close()
        _close_all(module)


# ==========================================================================
# 18. retain_graph snapshot behavior
# ==========================================================================

@needs_native
@pytest.mark.parametrize("kind", ["1d", "2d"])
def test_retain_graph_keeps_snapshots_until_final_release(kind, live_storages):
    module = _bn(kind)
    _load_state(module, {"running_mean": [0.3, -0.2, 0.5],
                         "running_var": [2.0, 0.5, 1.25]})
    module.eval()
    rng = np.random.default_rng(61)
    x = _bn_input(kind, seed=61)
    upstream = rng.standard_normal(x.shape)
    control = _clean_eval_grads(kind, {
        "running_mean": [0.3, -0.2, 0.5], "running_var": [2.0, 0.5, 1.25],
    }, x, upstream)

    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    resources = out._graph_resources
    assert len(resources) == 2
    snapshot_storages = {id(r._core.storage) for r in resources}
    _collect()
    baseline = len(live_storages)

    # Pass 1 (retained): resources stay open.
    loss.backward(retain_graph=True)
    assert all(not r.closed for r in resources)
    module.zero_grad()
    xt.zero_grad()

    # Mutate the running buffers between retained passes — the snapshots
    # must ignore it.
    _load_state(module, {"running_mean": [8.0, 8.0, 8.0],
                         "running_var": [9.0, 9.0, 9.0]})

    # Pass 2 (retained): still forward-time statistics.
    loss.backward(retain_graph=True)
    assert np.allclose(xt.grad.to_numpy(), control["x"], atol=1e-12)
    assert all(not r.closed for r in resources)
    assert snapshot_storages <= live_storages
    module.zero_grad()
    xt.zero_grad()

    # Final one-shot pass: releases the history exactly once.
    loss.backward()
    assert np.allclose(xt.grad.to_numpy(), control["x"], atol=1e-12)
    assert all(r.closed for r in resources)
    assert out._graph_resources == ()
    out._release_graph_resources()           # a second release is a no-op
    assert all(r.closed for r in resources)
    assert not (snapshot_storages & live_storages)
    # A later backward raises the freed-history error.
    with pytest.raises(RuntimeError, match="freed"):
        loss.backward()

    loss.close()
    out.close()
    xt.close()
    _close_all(module)


# ==========================================================================
# 19. Failed retryable backward
# ==========================================================================

@needs_native
@pytest.mark.parametrize("kind", ["1d", "2d"])
def test_failed_backward_is_retryable_and_ignores_mutated_running_values(
    monkeypatch, kind, live_storages
):
    """A backward failure injected after traversal has begun (through the
    engine's own ``_accumulate_grad`` seam) must commit no partial
    gradient and must not free the graph. After the failure is removed —
    and the registered running buffers mutated — a retry succeeds using the
    *original* immutable snapshots, matching a clean control that ignores
    the new running values, and releases the graph resources exactly once
    on the final one-shot pass."""
    module = _bn(kind)
    _load_state(module, {"running_mean": [0.3, -0.2, 0.5],
                         "running_var": [2.0, 0.5, 1.25]})
    module.eval()
    rng = np.random.default_rng(71)
    x = _bn_input(kind, seed=71)
    upstream = rng.standard_normal(x.shape)
    control = _clean_eval_grads(kind, {
        "running_mean": [0.3, -0.2, 0.5], "running_var": [2.0, 0.5, 1.25],
    }, x, upstream)

    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    resources = out._graph_resources
    _collect()
    baseline = len(live_storages)

    real_accumulate = NativeTensor._accumulate_grad
    calls = {"n": 0}

    def failing_accumulate(self, grad):
        calls["n"] += 1
        if calls["n"] == 3:                  # after traversal has begun
            raise _Boom("injected backward failure")
        return real_accumulate(self, grad)

    monkeypatch.setattr(NativeTensor, "_accumulate_grad", failing_accumulate)
    with pytest.raises(_Boom):
        loss.backward()
    monkeypatch.undo()

    # No partial commit, graph not freed, resources still open.
    assert xt.grad is None
    assert out._graph_freed is False
    assert out._graph_resources == resources
    assert all(not r.closed for r in resources)

    # Mutate/replace the registered running buffers, then retry.
    _load_state(module, {"running_mean": [8.0, 8.0, 8.0],
                         "running_var": [9.0, 9.0, 9.0]})
    loss.backward()

    # The retry used the forward-time snapshots, not the new running values.
    assert np.allclose(xt.grad.to_numpy(), control["x"], atol=1e-12)
    assert np.allclose(module.gamma.grad.to_numpy(), control["gamma"], atol=1e-12)
    assert np.allclose(module.beta.grad.to_numpy(), control["beta"], atol=1e-12)
    # The final one-shot pass released the resources exactly once.
    assert all(r.closed for r in resources)
    assert out._graph_resources == ()

    # Release the gradients the retry allocated, then the graph and state.
    for grad in (xt.grad, module.gamma.grad, module.beta.grad):
        grad.close()
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


# ==========================================================================
# 20. Ownership and live-storage matrix
# ==========================================================================

@needs_native
@pytest.mark.parametrize("build", [lambda: _bn("1d"), lambda: _bn("2d"),
                                   Nested1D, Nested2D, MixedNorm])
def test_state_dict_success_returns_to_baseline(build, live_storages):
    model = build()
    _collect()
    baseline = len(live_storages)
    state = model.state_dict()
    for snapshot in state.values():
        snapshot.close()
    _collect()
    assert len(live_storages) == baseline
    _close_all(model)


@needs_native
def test_repeated_save_load_cycles_do_not_grow_storage(tmp_path, live_storages):
    model = MixedNorm()
    path = os.path.join(str(tmp_path), "cycle.npz")
    save_native_checkpoint(path, model)
    _collect()
    baseline = len(live_storages)
    for _ in range(3):
        _load_state(model, {"bn1.running_mean": [1.0, 2.0, 3.0]})
        load_native_checkpoint(path, model)
        state = model.state_dict()
        model.load_state_dict(state)
        for snapshot in state.values():
            snapshot.close()
        _collect()
    assert len(live_storages) == baseline
    _close_all(model)


@needs_native
@pytest.mark.parametrize("kind", ["1d", "2d"])
def test_train_eval_backward_cycles_return_to_baseline(kind, live_storages):
    module = _bn(kind)
    _collect()
    baseline = len(live_storages)
    for training in (True, False):
        module.train(training)
        for step in range(4):
            xt = NativeParameter(_bn_input(kind, seed=step + 80))
            out = module(xt)
            grad = np.ones(out.shape)
            loss = out.multiply(NativeTensor.from_array(grad)).sum()
            loss.backward()
            for tensor in (loss, out, xt, xt.grad):
                tensor.close()
            module.zero_grad()
            _collect()
    assert len(live_storages) == baseline    # module state only; no growth
    _close_all(module)


@needs_native
def test_explicit_parameter_and_buffer_closure_returns_to_baseline(live_storages):
    _collect()
    baseline = len(live_storages)
    model = MixedNorm()
    # A stateful model owns parameters *and* buffers; §9 requires closing
    # both explicitly, since there is no NativeModule.close().
    assert len(model.parameters()) == 8
    assert len(model.buffers()) == 4
    _close_all(model)
    _collect()
    assert len(live_storages) == baseline


@needs_native
def test_shared_buffer_aliases_close_exactly_once(live_storages):
    _collect()
    baseline = len(live_storages)
    model = BufferAliasModule()
    # The alias and the child buffer are one object: identity-deduplicated
    # traversal must not double-close it.
    _close_all(model)
    for tensor in (model.bn.running_mean, model.bn.running_var,
                   model.bn.gamma, model.bn.beta):
        assert tensor.closed is True
    _collect()
    assert len(live_storages) == baseline


@needs_native
def test_shared_child_module_closes_state_once(live_storages):
    _collect()
    baseline = len(live_storages)
    model = SharedModule()
    _close_all(model)
    _collect()
    assert len(live_storages) == baseline


# ==========================================================================
# 21. Milestone guardrails — F5 added no capability
# ==========================================================================

@needs_native
def test_f5_adds_no_capability_or_public_surface():
    import tensorforge.experimental as experimental

    # The inventories are exactly what F4 left.
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "save_native_checkpoint", "load_native_checkpoint",
    )
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    # No normalization operation entered any operation inventory.
    for name in ("layer_norm", "batch_norm", "layernorm", "batchnorm"):
        assert name not in cpp.TENSOR_CORE_OPS
        assert name not in cpp.AUTOGRAD_OPS
        assert name not in cpp.RAW_KERNELS
    # The checkpoint format did not move.
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 1
    # No new public buffer-mutation API, and no NativeModule.close().
    assert not hasattr(NativeModule, "close")
    for banned in ("set_running_stats", "fill_", "copy_"):
        assert not hasattr(NativeTensor, banned), banned
    assert "replace_native_state" not in experimental.__all__
    # The private transaction helper stays private.
    assert not hasattr(experimental, "replace_native_state")
