"""Phase C completion guardrails (Advanced C++ v3.15).

Cross-cutting integration tests over the complete native training
stack — NativeTensor → native autograd → NativeParameter/NativeModule
→ NativeLinear/NativeReLU/NativeSequential → NativeMSELoss →
NativeSGD/NativeAdam → optimizer state snapshots → native checkpoint
files → deterministic training and resume. Each test spans several
components and locks an *integrated* invariant the per-component test
files (test_native_sgd/adam/optimizer_state/checkpoint/...) verify
only in isolation: full training lifecycles, the shared-parameter
story end to end, mixed active/frozen/grad=None collections under both
optimizers, repeated snapshot/load and checkpoint-resume chains,
failure recovery at every boundary, the four-way stale-graph
distinction, lifetime/close discipline, and the public surface.

This milestone adds no numerical behavior: these tests prove the
existing pieces compose, and Phase C is marked complete only because
they pass.

Selector: python -m pytest -q -k "native_phase_c"
"""

import inspect
import math

import numpy as np
import pytest

import tensorforge
import tensorforge.serialization as stable_serialization
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
    load_native_checkpoint,
    save_native_checkpoint,
)
import tensorforge.experimental as experimental
from tensorforge.experimental import native_adam as native_adam_module

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


P_VALUES = np.array([[1.0, -2.0], [0.5, 3.0]])
G_VALUES = np.array([[0.5, -1.0], [2.0, 0.25]])
X_VALUES = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.25], [-1.0, 1.5]])
Y_VALUES = np.array([[1.0], [-0.5], [0.25], [2.0]])

PARAMETER_KEYS = ["0.weight", "0.bias", "2.weight", "2.bias"]


def _mlp():
    return NativeSequential(
        NativeLinear(2, 8, seed=0),
        NativeReLU(),
        NativeLinear(8, 1, seed=1),
    )


def _train(model, optimizer, x, y, steps):
    """``steps`` full fresh-graph iterations; returns loss floats.
    Gradients are asserted present after each step (retained through
    it) and cleared at each boundary."""
    loss_fn = NativeMSELoss()
    losses = []
    for _ in range(steps):
        prediction = model(x)
        loss = loss_fn(prediction, y)
        losses.append(float(loss.to_numpy()))
        loss.backward()
        optimizer.step()
        assert all(p.grad is not None for p in model.parameters())
        optimizer.zero_grad()
        assert all(p.grad is None for p in model.parameters())
        loss.close()
        prediction.close()
    return losses


def _param_with_grad(values=P_VALUES, grad_values=G_VALUES):
    parameter = NativeParameter(values)
    parameter.multiply(NativeTensor.from_array(grad_values)).sum().backward()
    return parameter


class _SharedTwice(NativeModule):
    """y = x @ w + x @ w — the same NativeParameter used through two
    registered aliases and two forward paths."""

    def __init__(self, values):
        super().__init__()
        shared = NativeParameter(values)
        self.first = shared
        self.second = shared  # alias: same object, second name

    def forward(self, x):
        return x.matmul(self.first).add(x.matmul(self.second))


# ======================================================================
# 1 + 2. Full training lifecycles
# ======================================================================


@needs_native
def test_native_phase_c_sgd_training_lifecycle(monkeypatch):
    model = _mlp()
    optimizer = NativeSGD(model.parameters(), lr=0.1)
    parameters = model.parameters()
    identities = [id(p) for p in parameters]
    x = NativeTensor.from_array(X_VALUES)
    y = NativeTensor.from_array(Y_VALUES)

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("sqrt", "reciprocal", "divide", "add", "subtract",
                 "multiply", "matmul", "sum", "mean", "negative",
                 "power", "copyto"):
        monkeypatch.setattr(np, name, _tripwire)
    losses = _train(model, optimizer, x, y, 12)
    monkeypatch.undo()

    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < 0.5 * losses[0]  # meaningful reduction
    # Every trainable parameter updated; version delta equals the
    # active update count; identity and state keys stayed stable.
    assert [p.version for p in parameters] == [12] * 4
    assert [id(p) for p in model.parameters()] == identities
    assert list(k for k, _ in model.named_parameters()) == PARAMETER_KEYS


@needs_native
def test_native_phase_c_adam_training_lifecycle(monkeypatch):
    model = _mlp()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    parameters = model.parameters()
    x = NativeTensor.from_array(X_VALUES)
    y = NativeTensor.from_array(Y_VALUES)

    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("sqrt", "reciprocal", "divide", "add", "subtract",
                 "multiply", "matmul", "sum", "mean", "negative",
                 "power", "copyto"):
        monkeypatch.setattr(np, name, _tripwire)
    losses = _train(model, optimizer, x, y, 12)
    monkeypatch.undo()

    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < 0.5 * losses[0]
    # Versions, counters, and updates advance in lockstep; the
    # persistent state is graph-free and pairwise independent.
    assert [p.version for p in parameters] == [12] * 4
    assert optimizer.step_counts == (12,) * 4
    buffers = optimizer._m + optimizer._v
    assert len({id(buffer._core.storage) for buffer in buffers}) == 8
    for buffer in buffers:
        assert buffer.is_leaf and not buffer.requires_grad
        assert buffer._parents == () and buffer._backward is None
    # Closing the optimizer releases only its own state: the model
    # keeps training-capable parameters and live gradients.
    loss = NativeMSELoss()(model(x), y)
    loss.backward()
    optimizer.close()
    assert all(buffer.closed for buffer in buffers)
    assert all(not p.closed and p.grad is not None for p in parameters)
    replacement = NativeSGD(model.parameters(), lr=0.1)
    replacement.step()  # the model remains fully usable
    assert [p.version for p in parameters] == [13] * 4


# ======================================================================
# 3. Shared parameter, end to end through every layer of the stack
# ======================================================================


@needs_native
def test_native_phase_c_shared_parameter_end_to_end(tmp_path):
    module = _SharedTwice(P_VALUES)
    shared = module.first
    assert module.second is shared
    # Registration: one unique parameter under the canonical first name.
    assert module.parameters() == [shared]
    assert list(module.state_dict()) == ["first"]
    # Backward accumulates both use-sites' contributions.
    x = NativeTensor.from_array(X_VALUES[:2])
    module(x).sum().backward()
    expected_grad = 2 * (X_VALUES[:2].T @ np.ones((2, 2)))
    assert np.array_equal(shared.grad.to_numpy(), expected_grad)
    # NativeSGD: one update, one version increment.
    NativeSGD(module.parameters(), lr=0.1).step()
    assert shared.version == 1
    after_sgd = shared.to_numpy()
    assert np.array_equal(after_sgd, P_VALUES - 0.1 * expected_grad)
    # NativeAdam: one moment pair, one counter, one state entry.
    optimizer = NativeAdam(module.parameters(), lr=0.05)
    optimizer.step()
    assert shared.version == 2
    assert len(optimizer._m) == 1 and optimizer.step_counts == (1,)
    state = optimizer.state_dict()
    assert len(state["parameters"]) == 1 and len(state["m"]) == 1
    # Checkpoint: one model key, one optimizer entry.
    path = tmp_path / "shared.npz"
    save_native_checkpoint(path, module, optimizer=optimizer)
    with np.load(path, allow_pickle=False) as archive:
        assert sorted(archive.files) == [
            "manifest", "model::000000",
            "optimizer::m::000000", "optimizer::v::000000",
        ]
    # Restore into a fresh alias-structured module: alias preserved,
    # values/moments/counters restored, and the continuation matches
    # the original bit for bit.
    restored = _SharedTwice(np.zeros((2, 2)))
    restored_optimizer = NativeAdam(restored.parameters())
    load_native_checkpoint(path, restored, optimizer=restored_optimizer)
    assert restored.first is restored.second
    assert np.array_equal(restored.first.to_numpy(), shared.to_numpy())
    assert restored_optimizer.step_counts == (1,)
    for original_module, its_optimizer in (
        (module, optimizer), (restored, restored_optimizer)
    ):
        original_module.zero_grad()
        original_module(x).sum().backward()
        its_optimizer.step()
    assert np.array_equal(
        restored.first.to_numpy(), shared.to_numpy()
    )
    module.zero_grad()
    restored.zero_grad()
    for snapshot_label in ("m", "v"):
        for snapshot in state[snapshot_label]:
            snapshot.close()


# ======================================================================
# 4. Mixed active/frozen/grad=None/zero-gradient collections
# ======================================================================


@needs_native
def test_native_phase_c_mixed_collections_across_both_optimizers():
    for build in (
        lambda params: NativeSGD(params, lr=0.1),
        lambda params: NativeAdam(params, lr=0.1),
    ):
        active = _param_with_grad()
        frozen = NativeParameter(G_VALUES, requires_grad=False)
        stale = NativeTensor.from_array(P_VALUES)
        stale.close()
        frozen._grad = stale  # closed and invalid: must never be read
        no_grad = NativeParameter(P_VALUES)
        zero_grad = _param_with_grad(G_VALUES, np.zeros((2, 2)))
        optimizer = build([active, frozen, no_grad, zero_grad])
        is_adam = isinstance(optimizer, NativeAdam)
        if is_adam:
            skipped_moments = (optimizer._m[1], optimizer._m[2])
        optimizer.step()
        # Active updates; zero-but-present gradient is active too.
        assert active.version == 1
        assert zero_grad.version == 1
        assert np.array_equal(zero_grad.to_numpy(), G_VALUES)
        # Frozen and grad=None: no value, version, or state aging, and
        # the frozen entry's invalid gradient was never inspected.
        assert frozen.version == 0 and no_grad.version == 0
        assert np.array_equal(frozen.to_numpy(), G_VALUES)
        assert frozen._grad is stale
        if is_adam:
            assert optimizer.step_counts == (1, 0, 0, 1)
            assert optimizer._m[1] is skipped_moments[0]
            assert optimizer._m[2] is skipped_moments[1]
            assert np.array_equal(
                optimizer._m[2].to_numpy(), np.zeros((2, 2))
            )
        # Subsequent activation: the grad=None parameter now becomes
        # active and takes its correct first step (t = 1 under Adam).
        no_grad.multiply(NativeTensor.from_array(G_VALUES)).sum().backward()
        optimizer.step()
        assert no_grad.version == 1
        if is_adam:
            assert optimizer.step_counts == (2, 0, 1, 2)


# ======================================================================
# 5. Repeated optimizer snapshot/load cycles
# ======================================================================


@needs_native
def test_native_phase_c_repeated_optimizer_state_cycles():
    x = NativeTensor.from_array(X_VALUES)
    y = NativeTensor.from_array(Y_VALUES)
    model = _mlp()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    _train(model, optimizer, x, y, 3)
    versions_before = [p.version for p in model.parameters()]

    # A retained value-sensitive graph built now must survive every
    # optimizer-state load below (loading moves no parameter version,
    # so it can never make an old graph stale).
    probe = model.parameters()[0]
    graph = probe.multiply(
        NativeTensor.from_array(np.ones(probe.shape))
    ).sum()
    graph.backward(retain_graph=True)

    first_state = optimizer.state_dict()
    internal_before = optimizer._m[0]
    optimizer.load_state_dict(first_state)          # reload the same state
    assert optimizer._m[0] is not internal_before   # replaced …
    assert internal_before.closed                   # … and old state closed
    assert [p.version for p in model.parameters()] == versions_before
    graph.backward(retain_graph=True)               # still valid — not stale

    model.zero_grad()
    _train(model, optimizer, x, y, 2)
    second_state = optimizer.state_dict()
    assert second_state["step_counts"] == (5,) * 4

    # Load the second snapshot into a fresh compatible model/optimizer
    # pair, matched to the live model's values through the module loader.
    model_d = _mlp()
    model_snapshot = model.state_dict()
    model_d.load_state_dict(model_snapshot)
    for snapshot in model_snapshot.values():
        snapshot.close()
    optimizer_d = NativeAdam(model_d.parameters())
    optimizer_d.load_state_dict(second_state)
    assert optimizer_d.step_counts == (5,) * 4
    # Loading adopted no caller storage: the loaded buffers equal the
    # snapshot values but share storage with nothing live anywhere.
    for index in range(4):
        for label in ("m", "v"):
            live = getattr(optimizer_d, f"_{label}")[index]
            snapshot = second_state[label][index]
            assert np.array_equal(live.to_numpy(), snapshot.to_numpy())
            assert live._core.storage is not snapshot._core.storage
    for label in ("m", "v"):
        for snapshot in first_state[label] + second_state[label]:
            for live in (optimizer._m + optimizer._v
                         + optimizer_d._m + optimizer_d._v):
                assert snapshot._core.storage is not live._core.storage

    # Both restored pairs continue bit-identically: same values, same
    # restored moments/counters, same fresh-graph updates.
    losses_original = _train(model, optimizer, x, y, 2)
    losses_replayed = _train(model_d, optimizer_d, x, y, 2)
    assert losses_original == losses_replayed
    for parameter, replayed in zip(model.parameters(), model_d.parameters()):
        assert np.array_equal(parameter.to_numpy(), replayed.to_numpy())
    assert optimizer.step_counts == optimizer_d.step_counts == (7,) * 4

    for state in (first_state, second_state):
        for label in ("m", "v"):
            for snapshot in state[label]:
                snapshot.close()
    optimizer.close()
    optimizer_d.close()


# ======================================================================
# 6. Repeated checkpoint resume cycles
# ======================================================================


@needs_native
def test_native_phase_c_checkpoint_resume_chain(tmp_path):
    x = NativeTensor.from_array(X_VALUES)
    y = NativeTensor.from_array(Y_VALUES)
    model_a = _mlp()
    optimizer_a = NativeAdam(model_a.parameters(), lr=0.05)
    _train(model_a, optimizer_a, x, y, 4)
    first = tmp_path / "chain-1.npz"
    save_native_checkpoint(first, model_a, optimizer=optimizer_a,
                           metadata={"steps": 4})

    model_b = _mlp()
    optimizer_b = NativeAdam(model_b.parameters())
    metadata_b = load_native_checkpoint(first, model_b, optimizer=optimizer_b)
    assert metadata_b == {"steps": 4}
    _train(model_b, optimizer_b, x, y, 3)
    second = tmp_path / "chain-2.npz"
    save_native_checkpoint(second, model_b, optimizer=optimizer_b,
                           metadata={"steps": 7})

    model_c = _mlp()
    optimizer_c = NativeAdam(model_c.parameters())
    metadata_c = load_native_checkpoint(second, model_c, optimizer=optimizer_c)
    assert metadata_c == {"steps": 7}  # metadata evolves deterministically
    baseline_b = [p.version for p in model_b.parameters()]
    baseline_c = [p.version for p in model_c.parameters()]

    losses_b = _train(model_b, optimizer_b, x, y, 3)
    losses_c = _train(model_c, optimizer_c, x, y, 3)
    assert losses_b == losses_c
    for parameter_b, parameter_c in zip(
        model_b.parameters(), model_c.parameters()
    ):
        assert np.array_equal(
            parameter_b.to_numpy(), parameter_c.to_numpy()
        )
    for index in range(4):
        for label in ("m", "v"):
            assert np.array_equal(
                getattr(optimizer_b, f"_{label}")[index].to_numpy(),
                getattr(optimizer_c, f"_{label}")[index].to_numpy(),
            )
    assert optimizer_b.step_counts == optimizer_c.step_counts == (10,) * 4
    # Future version deltas match (absolute versions differ by the
    # number of model loads each lineage went through).
    assert [p.version for p in model_b.parameters()] == [
        v + 3 for v in baseline_b
    ]
    assert [p.version for p in model_c.parameters()] == [
        v + 3 for v in baseline_c
    ]
    # No temporary files or extra artifacts remain.
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [
        "chain-1.npz", "chain-2.npz",
    ]
    optimizer_a.close()
    optimizer_b.close()
    optimizer_c.close()


# ======================================================================
# 7. Failure recovery at every boundary, in one lifecycle
# ======================================================================


@needs_native
def test_native_phase_c_failure_recovery_lifecycle(tmp_path, monkeypatch):
    x = NativeTensor.from_array(X_VALUES)
    y = NativeTensor.from_array(Y_VALUES)
    model = _mlp()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    _train(model, optimizer, x, y, 2)
    loss = NativeMSELoss()(model(x), y)
    loss.backward()

    def snapshot_world():
        return (
            [p.to_numpy() for p in model.parameters()],
            [p.version for p in model.parameters()],
            [p.grad for p in model.parameters()],
            list(optimizer._m), list(optimizer._v),
            [b.to_numpy() for b in optimizer._m + optimizer._v],
            optimizer.step_counts,
        )

    def assert_world(world):
        values, versions, grads, m_list, v_list, buffer_values, counts = world
        for parameter, value, version, grad in zip(
            model.parameters(), values, versions, grads
        ):
            assert np.array_equal(parameter.to_numpy(), value)
            assert parameter.version == version
            assert parameter.grad is grad
        assert list(optimizer._m) == m_list
        assert list(optimizer._v) == v_list
        for buffer, expected in zip(
            optimizer._m + optimizer._v, buffer_values
        ):
            assert np.array_equal(buffer.to_numpy(), expected)
        assert optimizer.step_counts == counts

    world = snapshot_world()

    # (a) optimizer step staging failure.
    real_full = cpp.NativeTensorCore.full
    calls = {"n": 0}

    def flaky_full(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 9:
            raise MemoryError("forced step staging failure")
        return real_full(*args, **kwargs)

    monkeypatch.setattr(cpp.NativeTensorCore, "full", flaky_full)
    with pytest.raises(MemoryError, match="forced step staging failure"):
        optimizer.step()
    monkeypatch.undo()
    assert_world(world)

    # (b) optimizer state load staging failure.
    state = optimizer.state_dict()
    real_copy = native_adam_module._native_copy
    calls = {"n": 0}

    def flaky_copy(core):
        calls["n"] += 1
        if calls["n"] == 3:
            raise MemoryError("forced load staging failure")
        return real_copy(core)

    monkeypatch.setattr(native_adam_module, "_native_copy", flaky_copy)
    with pytest.raises(MemoryError, match="forced load staging failure"):
        optimizer.load_state_dict(state)
    monkeypatch.undo()
    assert_world(world)
    assert all(not snapshot.closed
               for label in ("m", "v") for snapshot in state[label])

    # (c) checkpoint save failure: existing destination and live state
    # both survive, and no temporary file remains.
    path = tmp_path / "recovery.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)
    original_bytes = path.read_bytes()

    def failing_savez(*args, **kwargs):
        raise OSError("forced checkpoint write failure")

    monkeypatch.setattr(np, "savez", failing_savez)
    with pytest.raises(OSError, match="forced checkpoint write failure"):
        save_native_checkpoint(path, model, optimizer=optimizer)
    monkeypatch.undo()
    assert path.read_bytes() == original_bytes
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [
        "recovery.npz",
    ]
    assert_world(world)

    # (d) checkpoint corruption on load.
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not an archive at all")
    with pytest.raises(ValueError, match="not a valid native checkpoint"):
        load_native_checkpoint(corrupt, model, optimizer=optimizer)
    assert_world(world)
    corrupt.unlink()

    # After all four failures, every later valid operation succeeds.
    optimizer.load_state_dict(state)
    for label in ("m", "v"):
        for snapshot in state[label]:
            snapshot.close()
    optimizer.step()
    assert optimizer.step_counts == (3,) * 4
    load_native_checkpoint(path, model, optimizer=optimizer)
    assert optimizer.step_counts == (2,) * 4  # restored pre-step state
    loss.close()
    optimizer.close()


# ======================================================================
# 8. Graph-version interactions across every mutation boundary
# ======================================================================


@needs_native
def test_native_phase_c_graph_version_interactions(tmp_path):
    model = NativeLinear(2, 3, seed=0)
    optimizer = NativeAdam(model.parameters(), lr=0.1)
    path = tmp_path / "graph.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)

    x = NativeTensor.from_array(P_VALUES, requires_grad=True)

    def fresh_sensitive_graph():
        out = x.matmul(model.weight).sum()
        out.backward(retain_graph=True)
        return out

    # (1) optimizer state load alone: graph stays valid.
    graph = fresh_sensitive_graph()
    state = optimizer.state_dict()
    optimizer.load_state_dict(state)
    graph.backward(retain_graph=True)
    for label in ("m", "v"):
        for snapshot in state[label]:
            snapshot.close()

    # (2) failed checkpoint load: graph stays valid, gradients intact.
    grad_before = model.weight.grad
    with pytest.raises(ValueError, match="no optimizer was supplied"):
        # presence mismatch: a model-only load request against an
        # archive holding optimizer state — rejected pre-mutation.
        load_native_checkpoint(path, model)
    assert model.weight.grad is grad_before
    graph.backward(retain_graph=True)

    # (3) optimizer step: the graph becomes stale; gradients untouched
    # by the raise; a fresh forward/backward succeeds.
    _set = model(NativeTensor.from_array(X_VALUES)).sum()
    _set.backward()  # gradients for the step
    grad_before = model.weight.grad
    optimizer.step()
    with pytest.raises(RuntimeError, match="stale"):
        graph.backward(retain_graph=True)
    assert model.weight.grad is grad_before
    model.zero_grad()
    x.zero_grad()
    graph = fresh_sensitive_graph()

    # (4) model state load: stale again, even with identical values.
    snapshot = model.state_dict()
    model.load_state_dict(snapshot)
    for tensor in snapshot.values():
        tensor.close()
    with pytest.raises(RuntimeError, match="stale"):
        graph.backward(retain_graph=True)
    model.zero_grad()
    x.zero_grad()
    graph = fresh_sensitive_graph()

    # (5) successful checkpoint restoration: stale; fresh pass works.
    load_native_checkpoint(path, model, optimizer=optimizer)
    with pytest.raises(RuntimeError, match="stale"):
        graph.backward(retain_graph=True)
    model.zero_grad()
    x.zero_grad()
    fresh_sensitive_graph()
    assert model.weight.grad is not None
    optimizer.close()


# ======================================================================
# 9. Lifetime and close discipline across the stack
# ======================================================================


@needs_native
def test_native_phase_c_lifetime_and_close_discipline(tmp_path):
    model = _mlp()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    x = NativeTensor.from_array(X_VALUES)
    y = NativeTensor.from_array(Y_VALUES)
    _train(model, optimizer, x, y, 2)
    loss = NativeMSELoss()(model(x), y)
    loss.backward()

    # Model and optimizer snapshots are caller-owned: closing them all
    # leaves every live object fully functional.
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    for snapshot in model_state.values():
        snapshot.close()
        snapshot.close()  # repeated close stays safe
    for label in ("m", "v"):
        for snapshot in optimizer_state[label]:
            snapshot.close()
            snapshot.close()
    assert all(not p.closed for p in model.parameters())
    assert all(not b.closed for b in optimizer._m + optimizer._v)
    optimizer.step()  # still fully usable

    # The optimizer owns exactly its internal moments: close() releases
    # them and nothing else, and is idempotent.
    buffers = optimizer._m + optimizer._v
    grads = [p.grad for p in model.parameters()]
    optimizer.close()
    optimizer.close()
    assert all(b.closed for b in buffers)
    assert all(not p.closed for p in model.parameters())
    assert all(g is current_g and not g.closed
               for g, current_g in zip(grads,
                                       (p.grad for p in model.parameters())))

    # Checkpoint staging objects do not survive a load: the optimizer
    # holds its own copies and keeps working even after the archive
    # file is deleted (nothing references the archive arrays or the
    # transient staging tensors), and loading adds no parameter version.
    path = tmp_path / "lifetime.npz"
    fresh_optimizer = NativeAdam(model.parameters(), lr=0.05)
    save_native_checkpoint(path, model, optimizer=fresh_optimizer)
    versions_pre_load = [p.version for p in model.parameters()]
    load_native_checkpoint(path, model, optimizer=fresh_optimizer)
    assert [p.version for p in model.parameters()] == [
        v + 1 for v in versions_pre_load  # model load: +1; optimizer: +0
    ]
    path.unlink()  # the archive is gone; nothing live referenced it
    fresh_optimizer.step()  # still fully usable, no staging object survived
    fresh_optimizer.close()
    loss.close()


# ======================================================================
# 10. Public-surface guardrails
# ======================================================================


@needs_native
def test_native_phase_c_public_surface_guardrails():
    # The complete Phase C surface, exported from experimental only, plus
    # the Phase-D additions (NativeFlatten D1, NativeConv2d D7).
    assert set(experimental.__all__) == {
        "NativeTensor", "NativeParameter", "NativeParameterRegistry",
        "NativeModule", "NativeLinear", "NativeReLU", "NativeSequential",
        "NativeMSELoss", "NativeSGD", "NativeAdam",
        "save_native_checkpoint", "load_native_checkpoint",
        "NativeFlatten",  # Phase D, milestone D1
        "NativeConv2d",   # Phase D, milestone D7
    }
    for name in experimental.__all__:
        assert not hasattr(tensorforge, name)
    # Internal helper modules stay internal.
    for helper in ("native_optimizer_state", "native_checkpoint",
                   "FORMAT_VERSION", "validate_state_schema"):
        assert helper not in experimental.__all__
    # No optimizer base class was introduced: the two optimizers share
    # only `object`.
    assert NativeSGD.__mro__ == (NativeSGD, object)
    assert NativeAdam.__mro__ == (NativeAdam, object)
    # No checkpoint APIs leaked into stable serialization; the stable
    # surface is untouched.
    for absent in ("save_native_checkpoint", "load_native_checkpoint"):
        assert not hasattr(stable_serialization, absent)
    # No unsupported optimizer feature appeared.
    parameter = NativeParameter(P_VALUES)
    sgd = NativeSGD([parameter], lr=0.1)
    adam = NativeAdam([parameter], lr=0.1)
    for optimizer in (sgd, adam):
        for absent in ("param_groups", "add_param_group", "weight_decay",
                       "amsgrad", "momentum", "clip_grad_norm",
                       "scheduler", "map_location"):
            assert not hasattr(optimizer, absent)
    assert "momentum" not in inspect.signature(NativeSGD.__init__).parameters
    # No pooling CNN API and no CUDA/dtype expansion appeared.
    # (NativeFlatten shipped in Phase D milestone D1 and NativeConv2d in
    # milestone D7; both are asserted present in the surface set above.)
    for absent in ("NativeMaxPool2d",
                   "NativeDropout", "NativeBatchNorm1d"):
        assert not hasattr(experimental, absent)
    info = cpp.backend_info()
    assert info["device"] == "cpu" and info["dtype"] == "float64"
