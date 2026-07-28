"""The native CNN training and checkpoint-resume proof (Phase D,
milestone D11).

examples/native_cnn_training.py trains
``NativeSequential(NativeConv2d(1, 2, 2, seed=0), NativeReLU(),
NativeMaxPool2d(2), NativeFlatten(), NativeLinear(8, 1, seed=1))`` on eight
fixed 6x6 single-channel images for 40 deterministic ``NativeAdam(lr=0.05)``
steps, learning the strength of each image's strongest bright-to-dark
vertical edge — a genuinely spatial target. These tests verify the
architecture and state surface, deterministic initialization, forward /
loss / backward / optimizer integration, gradient flow into every trainable
layer, the loss-reduction guardrails, exact uninterrupted-versus-resumed
equivalence through one pickle-free checkpoint, storage and saved-winner
lifetime across steps, stale-graph safety, and the failure paths.

Every number asserted below was observed from the deterministic workload
first; the thresholds carry a wide margin (see
``test_loss_reduction_guardrails``).

NumPy appears only for references and inspection; the training computation
is native.

Selector: python -m pytest -q -k native_cnn_training
"""

import gc
import math

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeConv2d,
    NativeFlatten,
    NativeLinear,
    NativeMaxPool2d,
    NativeMSELoss,
    NativeModule,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)
from examples.native_cnn_training import (
    CONV_CHANNELS,
    DEFAULT_LR,
    FLAT_FEATURES,
    IMAGE_VALUES,
    SPLIT_STEP,
    TARGET_VALUES,
    TOTAL_STEPS,
    build_model,
    build_optimizer,
    checkpoint_resume_proof,
    main,
    strongest_edge,
    train,
)

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)

# Observed once from the deterministic workload; asserted with margin.
OBSERVED_INITIAL_LOSS = 0.7713061867163815
OBSERVED_FINAL_LOSS = 0.011085


@pytest.fixture(autouse=True)
def _disarm_after_each():
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — a real
    live-native-allocation count for the lifetime tests."""
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


def _data():
    return (NativeTensor.from_array(IMAGE_VALUES),
            NativeTensor.from_array(TARGET_VALUES))


def _step(model, loss_fn, optimizer, x, y):
    """One training iteration; returns the pre-update loss."""
    prediction = model(x)
    loss = loss_fn(prediction, y)
    try:
        value = float(loss.to_numpy())
        loss.backward()
        optimizer.step()
    finally:
        loss.close()
        prediction.close()
    optimizer.zero_grad()
    return value


def _close(model, optimizer=None):
    if optimizer is not None:
        optimizer.close()
    for parameter in model.parameters():
        parameter.close()


# --------------------------------------------------------------------------
# The task and the model architecture
# --------------------------------------------------------------------------

def test_dataset_targets_follow_the_documented_rule():
    # The frozen literals are exactly what the documented spatial rule
    # produces — the task is reproducible from its description.
    derived = [[strongest_edge(image)] for image in IMAGE_VALUES]
    assert derived == TARGET_VALUES
    # The task is not trivial: targets vary, include an exact zero, and
    # are not a constant or a plain pixel sum.
    values = [row[0] for row in TARGET_VALUES]
    assert min(values) == 0.0 and max(values) == 1.25
    assert len(set(values)) >= 4
    sums = [float(np.sum(image)) for image in IMAGE_VALUES]
    # A pixel-sum shortcut would rank the samples the same way; it does not.
    assert [i for _, i in sorted(zip(values, range(8)))] != [
        i for _, i in sorted(zip(sums, range(8)))
    ]


def test_model_module_sequence_and_output_shape():
    model = build_model()
    kinds = [type(m) for m in model]
    assert kinds == [
        NativeConv2d, NativeReLU, NativeMaxPool2d, NativeFlatten, NativeLinear
    ]
    x, _ = _data()
    out = model(x)
    assert out.shape == (len(IMAGE_VALUES), 1)
    assert out.dtype == "float64" and out.device == "cpu"
    out.close()
    x.close()
    _close(model)


def test_intermediate_shapes_through_the_stack():
    model = build_model()
    x, _ = _data()
    conv = model[0](x)
    assert conv.shape == (8, CONV_CHANNELS, 5, 5)   # 6x6 -> 5x5
    relu = model[1](conv)
    pooled = model[2](relu)
    assert pooled.shape == (8, CONV_CHANNELS, 2, 2)  # floor((5-2)/2)+1
    flat = model[3](pooled)
    assert flat.shape == (8, FLAT_FEATURES)
    out = model[4](flat)
    assert out.shape == (8, 1)
    for t in (conv, relu, pooled, flat, out, x):
        t.close()
    _close(model)


def test_parameter_names_state_keys_and_dtypes():
    model = build_model()
    names = [name for name, _ in model.named_parameters()]
    assert names == ["0.weight", "0.bias", "4.weight", "4.bias"]
    assert sorted(model.state_dict()) == ["0.bias", "0.weight",
                                          "4.bias", "4.weight"]
    # Pooling, ReLU, and Flatten contribute no parameters and no state.
    for index in (1, 2, 3):
        assert list(model[index].parameters()) == []
        assert model[index].state_dict() == {}
    assert list(model.buffers()) == []
    for parameter in model.parameters():
        assert parameter.dtype == "float64" and parameter.device == "cpu"
    shapes = {name: p.shape for name, p in model.named_parameters()}
    assert shapes == {
        "0.weight": (CONV_CHANNELS, 1, 2, 2),
        "0.bias": (CONV_CHANNELS,),
        "4.weight": (FLAT_FEATURES, 1),
        "4.bias": (1,),
    }
    _close(model)


# --------------------------------------------------------------------------
# Deterministic initialization
# --------------------------------------------------------------------------

def test_repeated_construction_is_identical():
    first, second = build_model(), build_model()
    for (name_a, a), (name_b, b) in zip(first.named_parameters(),
                                        second.named_parameters()):
        assert name_a == name_b
        assert np.array_equal(a.to_numpy(), b.to_numpy())
    x, _ = _data()
    out_a, out_b = first(x), second(x)
    assert np.array_equal(out_a.to_numpy(), out_b.to_numpy())
    for t in (out_a, out_b, x):
        t.close()
    _close(first)
    _close(second)


def test_different_seeds_differ():
    seeded = NativeConv2d(1, CONV_CHANNELS, 2, seed=0)
    other = NativeConv2d(1, CONV_CHANNELS, 2, seed=99)
    assert not np.array_equal(seeded.weight.to_numpy(), other.weight.to_numpy())
    for layer in (seeded, other):
        for parameter in layer.parameters():
            parameter.close()


def test_initial_predictions_are_reproducible():
    first = train(steps=1)
    second = train(steps=1)
    assert first["initial_predictions"] == second["initial_predictions"]
    assert first["initial_loss"] == second["initial_loss"]
    assert first["initial_loss"] == pytest.approx(OBSERVED_INITIAL_LOSS,
                                                  rel=1e-12)


# --------------------------------------------------------------------------
# Forward / backward integration
# --------------------------------------------------------------------------

def test_forward_loss_backward_reaches_every_trainable_parameter():
    model = build_model()
    loss_fn = NativeMSELoss()
    x, y = _data()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    assert loss.numel == 1                      # scalar MSE
    loss.backward()
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        assert grad is not None, name
        assert grad.shape == parameter.shape, name
        values = grad.to_numpy()
        assert np.isfinite(values).all(), name
        assert (values != 0.0).any(), name      # meaningfully nonzero
    for t in (loss, prediction, x, y):
        t.close()
    _close(model)


def test_backward_reaches_the_input_when_requested():
    model = build_model()
    loss_fn = NativeMSELoss()
    x = NativeTensor.from_array(IMAGE_VALUES, requires_grad=True)
    y = NativeTensor.from_array(TARGET_VALUES)
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (8, 1, 6, 6)
    assert np.isfinite(x.grad.to_numpy()).all()
    assert (x.grad.to_numpy() != 0.0).any()
    for t in (loss, prediction, x, y):
        t.close()
    _close(model)


def test_graph_is_released_after_a_normal_backward():
    model = build_model()
    loss_fn = NativeMSELoss()
    x, y = _data()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    assert loss._graph_freed is True
    # The pooling node's private winner buffer went with the history.
    assert prediction._graph_resources == ()
    with pytest.raises(RuntimeError, match="freed autograd graph"):
        loss.backward()
    for t in (loss, prediction, x, y):
        t.close()
    _close(model)


# --------------------------------------------------------------------------
# Optimizer integration
# --------------------------------------------------------------------------

def test_optimizer_step_updates_every_parameter_with_stable_identity():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = NativeMSELoss()
    x, y = _data()
    before = {n: p.to_numpy().copy() for n, p in model.named_parameters()}
    identities = [id(p) for p in model.parameters()]
    versions = [p.version for p in model.parameters()]

    _step(model, loss_fn, optimizer, x, y)

    for name, parameter in model.named_parameters():
        assert not np.array_equal(parameter.to_numpy(), before[name]), name
        assert np.isfinite(parameter.to_numpy()).all(), name
    assert [id(p) for p in model.parameters()] == identities
    assert [p.version for p in model.parameters()] == [v + 1 for v in versions]
    for t in (x, y):
        t.close()
    _close(model, optimizer)


def test_secondary_optimizer_completes_a_valid_cnn_step():
    # NativeSGD is not the canonical resume optimizer, but it must drive
    # the same CNN correctly for at least one step.
    model = build_model()
    optimizer = NativeSGD(model.parameters(), lr=DEFAULT_LR)
    loss_fn = NativeMSELoss()
    x, y = _data()
    before = _step(model, loss_fn, optimizer, x, y)
    after = _step(model, loss_fn, optimizer, x, y)
    assert math.isfinite(before) and math.isfinite(after)
    assert after < before
    for t in (x, y):
        t.close()
    _close(model)


def test_optimizer_state_covers_only_trainable_parameters():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = NativeMSELoss()
    x, y = _data()
    _step(model, loss_fn, optimizer, x, y)
    state = optimizer.state_dict()
    assert len(state["parameters"]) == 4     # conv w/b + linear w/b
    assert len(state["m"]) == 4 and len(state["v"]) == 4
    assert len(state["step_counts"]) == 4
    text = str(state).lower()
    assert "pool" not in text and "flatten" not in text and "winner" not in text
    for t in (x, y):
        t.close()
    _close(model, optimizer)


# --------------------------------------------------------------------------
# Learning behavior
# --------------------------------------------------------------------------

def test_canonical_run_is_reproducible():
    first = train()
    second = train()
    assert first["loss_history"] == second["loss_history"]
    assert first["final_loss"] == second["final_loss"]
    assert first["final_predictions"] == second["final_predictions"]
    assert first["final_parameters"] == second["final_parameters"]


def test_loss_reduction_guardrails():
    """Thresholds chosen after observing the deterministic run: it goes
    from ~0.771306 to ~0.011085 (ratio ~0.0144). The guards below allow a
    final loss up to 0.10 and a ratio up to 0.10 — roughly a 7-9x margin
    over the observed values, so ordinary floating-point drift or a small
    schedule change cannot make them fail spuriously."""
    run = train()
    assert run["initial_loss"] == pytest.approx(OBSERVED_INITIAL_LOSS, rel=1e-9)
    assert run["final_loss"] == pytest.approx(OBSERVED_FINAL_LOSS, abs=1e-5)
    assert run["final_loss"] < 0.10
    assert run["final_loss"] / run["initial_loss"] < 0.10
    assert run["final_loss"] < run["initial_loss"]
    # The loss curve need not be monotonic (Adam overshoots early), but it
    # must end far below where it started.
    assert min(run["loss_history"]) < 0.05
    assert all(math.isfinite(value) for value in run["loss_history"])


def test_predictions_move_toward_the_targets():
    run = train()
    targets = np.asarray(run["targets"])
    initial = np.abs(np.asarray(run["initial_predictions"]) - targets).sum()
    final = np.abs(np.asarray(run["final_predictions"]) - targets).sum()
    assert final < initial / 3.0        # observed: ~5.6 -> ~0.55
    assert np.isfinite(run["final_predictions"]).all()


def test_both_trainable_layers_changed():
    run = train()
    for name in ("0.weight", "0.bias", "4.weight", "4.bias"):
        before = np.asarray(run["initial_parameters"][name])
        after = np.asarray(run["final_parameters"][name])
        assert not np.array_equal(before, after), name
        assert np.abs(after - before).max() > 1e-3, name
    # Versions advanced exactly once per step, identities never moved.
    assert run["final_versions"] == [v + run["steps"]
                                     for v in run["initial_versions"]]
    assert run["identity_stable"] is True
    assert run["gradients_cleared"] is True
    assert run["state_keys"] == ["0.bias", "0.weight", "4.bias", "4.weight"]


def test_first_step_gradient_evidence_is_recorded():
    run = train(steps=1)
    assert run["gradient_shapes"] == {
        "0.weight": (CONV_CHANNELS, 1, 2, 2),
        "0.bias": (CONV_CHANNELS,),
        "4.weight": (FLAT_FEATURES, 1),
        "4.bias": (1,),
    }
    assert all(run["gradient_nonzero"].values())


# --------------------------------------------------------------------------
# Checkpoint / resume equivalence (the central D11 contract)
# --------------------------------------------------------------------------

def test_resumed_training_matches_uninterrupted_exactly():
    proof = checkpoint_resume_proof()
    assert proof["prefix_matches"] is True
    assert proof["suffix_matches"] is True
    assert proof["losses_match"] is True
    assert proof["final_losses_match"] is True
    assert proof["predictions_match"] is True
    assert proof["parameters_match"] is True
    assert proof["optimizer_state_matches"] is True
    assert proof["parameter_order_matches"] is True
    assert proof["identities_preserved"] is True
    assert proof["metadata"] == {"steps_completed": SPLIT_STEP,
                                 "lr": DEFAULT_LR}
    assert len(proof["uninterrupted_losses"]) == TOTAL_STEPS
    assert proof["state_keys"] == ["0.bias", "0.weight", "4.bias", "4.weight"]


def test_resume_equivalence_holds_for_another_split():
    proof = checkpoint_resume_proof(total_steps=12, split_step=5)
    assert proof["losses_match"] is True
    assert proof["parameters_match"] is True
    assert proof["optimizer_state_matches"] is True


def test_checkpoint_archive_holds_only_persistent_state(tmp_path):
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = NativeMSELoss()
    x, y = _data()
    _step(model, loss_fn, optimizer, x, y)
    path = str(tmp_path / "cnn.checkpoint.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata={"steps_completed": 1})

    with np.load(path, allow_pickle=False) as archive:
        names = list(archive.files)
        manifest = archive["manifest"].tobytes().decode("utf-8")
    blob = (" ".join(names) + " " + manifest).lower()
    # No transient CNN state of any kind is serialized.
    for banned in ("winner", "grad", "graph", "maxpool", "pool", "flatten",
                   "relu", "prediction"):
        assert banned not in blob, banned
    # Only the four trainable parameters, plus the optimizer's own state.
    assert '"keys": ["0.weight", "0.bias", "4.weight", "4.bias"]' in manifest
    # The format contract is untouched by D11.
    assert '"format": "tensorforge.native_checkpoint"' in manifest
    assert '"format_version": 2' in manifest
    for t in (x, y):
        t.close()
    _close(model, optimizer)


def test_loading_restores_values_without_restoring_a_graph(tmp_path):
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = NativeMSELoss()
    x, y = _data()
    _step(model, loss_fn, optimizer, x, y)
    path = str(tmp_path / "cnn.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)

    fresh = build_model()
    fresh_optimizer = build_optimizer(fresh)
    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    for parameter in fresh.parameters():
        # Loaded parameters are graph-free leaves with no gradient.
        assert parameter.is_leaf is True
        assert parameter.grad is None
        assert parameter._graph_resources == ()
    # Training resumes by building a fresh graph.
    resumed = _step(fresh, loss_fn, fresh_optimizer, x, y)
    assert math.isfinite(resumed)
    for t in (x, y):
        t.close()
    _close(model, optimizer)
    _close(fresh, fresh_optimizer)


def test_loading_into_an_incompatible_architecture_fails_atomically(tmp_path):
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = NativeMSELoss()
    x, y = _data()
    _step(model, loss_fn, optimizer, x, y)
    path = str(tmp_path / "cnn.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)

    # A different architecture: more conv channels, different linear head.
    other = NativeSequential(
        NativeConv2d(1, 3, 2, seed=0), NativeReLU(), NativeMaxPool2d(2),
        NativeFlatten(), NativeLinear(3 * 2 * 2, 1, seed=1),
    )
    other_optimizer = build_optimizer(other)
    before = {n: p.to_numpy().copy() for n, p in other.named_parameters()}
    with pytest.raises(Exception):
        load_native_checkpoint(path, other, optimizer=other_optimizer)
    # Nothing was mutated by the rejected load.
    for name, parameter in other.named_parameters():
        assert np.array_equal(parameter.to_numpy(), before[name]), name
    # The original checkpoint still loads correctly afterwards.
    fresh = build_model()
    fresh_optimizer = build_optimizer(fresh)
    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    assert np.array_equal(
        fresh.parameters()[0].to_numpy(), model.parameters()[0].to_numpy()
    )
    for t in (x, y):
        t.close()
    _close(model, optimizer)
    _close(other, other_optimizer)
    _close(fresh, fresh_optimizer)


# --------------------------------------------------------------------------
# Ownership and lifetime
# --------------------------------------------------------------------------

def test_repeated_steps_return_to_a_stable_storage_baseline(live_storages):
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = NativeMSELoss()
    x, y = _data()
    # Warm up: the first steps allocate Adam's persistent moment state and
    # each parameter's first gradient, so the steady state starts after
    # them. gc.collect() makes the count deterministic — the invariant
    # being proved is that *nothing accumulates*, not when CPython happens
    # to finalize a dropped transient.
    for _ in range(3):
        _step(model, loss_fn, optimizer, x, y)
    gc.collect()
    baseline = len(live_storages)
    for _ in range(5):
        _step(model, loss_fn, optimizer, x, y)
        gc.collect()
        # No winner buffer, graph node, or transient output accumulates.
        assert len(live_storages) == baseline
    for t in (x, y):
        t.close()
    _close(model, optimizer)


def test_no_graph_resource_survives_a_training_step():
    # Run the stack layer by layer so the *pooling* output — the node that
    # owns the private saved-winner buffer — is observable; NativeSequential
    # drops it internally.
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = NativeMSELoss()
    x, y = _data()
    conv = model[0](x)
    relu = model[1](conv)
    pooled = model[2](relu)
    winners = pooled._graph_resources
    assert winners, "the pooling node should own a saved winner buffer"
    assert all(not core._closed for core in winners)
    flat = model[3](pooled)
    prediction = model[4](flat)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    assert all(core._closed for core in winners)   # released with the graph
    assert pooled._graph_resources == ()
    for t in (loss, prediction, flat, pooled, relu, conv, x, y):
        t.close()
    _close(model, optimizer)


def test_checkpoint_round_trip_leaks_no_storage(tmp_path, live_storages):
    # Everything this test allocates is closed explicitly, so the live
    # count must return exactly to where it started. gc.collect() brackets
    # the measurements so the assertion tests ownership, not collection
    # timing.
    gc.collect()
    baseline = len(live_storages)

    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = NativeMSELoss()
    x, y = _data()
    _step(model, loss_fn, optimizer, x, y)

    gc.collect()
    before_save = len(live_storages)
    path = str(tmp_path / "roundtrip.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)
    gc.collect()
    assert len(live_storages) == before_save   # saving allocates nothing net

    fresh = build_model()
    fresh_optimizer = build_optimizer(fresh)
    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    # Loading replaces parameter values in place, so the fresh model's
    # parameter identities survive it.
    assert all(p.grad is None for p in fresh.parameters())

    _close(fresh, fresh_optimizer)
    for t in (x, y):
        t.close()
    _close(model, optimizer)
    gc.collect()
    assert len(live_storages) == baseline


def test_train_returns_python_values_only():
    run = train(steps=2)
    for value in run.values():
        assert not isinstance(value, NativeTensor)
    assert isinstance(run["final_loss"], float)
    assert isinstance(run["loss_history"], list)


# --------------------------------------------------------------------------
# Stale-graph safety and failure paths
# --------------------------------------------------------------------------

def test_reusing_a_graph_after_an_optimizer_step_raises():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = NativeMSELoss()
    x, y = _data()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward(retain_graph=True)
    optimizer.step()          # mutates parameter values, bumping versions
    with pytest.raises(RuntimeError, match="stale parameter value"):
        loss.backward(retain_graph=True)
    optimizer.zero_grad()
    loss.close()
    prediction.close()
    # A fresh graph on the next step works normally.
    value = _step(model, loss_fn, optimizer, x, y)
    assert math.isfinite(value)
    for t in (x, y):
        t.close()
    _close(model, optimizer)


@needs_fault_injection
def test_allocation_failure_mid_step_preserves_parameters(live_storages):
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = NativeMSELoss()
    x, y = _data()
    _step(model, loss_fn, optimizer, x, y)      # settle Adam's state
    before = {n: p.to_numpy().copy() for n, p in model.named_parameters()}
    versions = [p.version for p in model.parameters()]
    baseline = len(live_storages)

    cpp._arm_alloc_failure(1)                   # fail the next allocation
    with pytest.raises(MemoryError):
        _step(model, loss_fn, optimizer, x, y)
    cpp._arm_alloc_failure(0)

    # No parameter moved and no version advanced: the failed step
    # committed nothing.
    for name, parameter in model.named_parameters():
        assert np.array_equal(parameter.to_numpy(), before[name]), name
    assert [p.version for p in model.parameters()] == versions
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    optimizer.zero_grad()
    assert len(live_storages) <= baseline       # no winner/graph leak

    # A later fresh step succeeds.
    value = _step(model, loss_fn, optimizer, x, y)
    assert math.isfinite(value)
    for t in (x, y):
        t.close()
    _close(model, optimizer)


def test_optimizer_step_validation_failure_leaves_state_consistent():
    model = build_model()
    with pytest.raises((TypeError, ValueError)):
        NativeAdam(model.parameters(), lr=-1.0)   # rejected before any state
    optimizer = build_optimizer(model)
    loss_fn = NativeMSELoss()
    x, y = _data()
    value = _step(model, loss_fn, optimizer, x, y)
    assert math.isfinite(value)
    for t in (x, y):
        t.close()
    _close(model, optimizer)


# --------------------------------------------------------------------------
# The runnable example
# --------------------------------------------------------------------------

def test_example_file_exists_and_is_importable():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    example = repo_root / "examples" / "native_cnn_training.py"
    assert example.is_file()
    text = example.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in text
    # No network access, no repository mutation, no git instructions.
    for banned in ("requests", "urllib", "socket", "git ", "subprocess"):
        assert banned not in text, banned


def test_example_main_runs_and_reports_learning_and_resume(capsys):
    main()
    output = capsys.readouterr().out
    assert "native CNN training + checkpoint resume ok" in output
    assert "% reduction)" in output
    assert "resumed losses match:     True" in output
    assert "final parameters match:   True" in output
    assert "optimizer state matches:  True" in output
    # Compact output: a handful of lines, not a training log.
    assert len(output.splitlines()) < 40


def test_example_leaves_no_checkpoint_files_behind():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    before = {p.name for p in repo_root.iterdir()}
    checkpoint_resume_proof(total_steps=4, split_step=2)
    after = {p.name for p in repo_root.iterdir()}
    assert before == after
    assert not list((repo_root / "examples").glob("*.npz"))


# --------------------------------------------------------------------------
# Capability introspection: D11 adds no API surface
# --------------------------------------------------------------------------

def test_d11_adds_no_capability_entries():
    assert cpp.RAW_KERNELS == (
        "elementwise_add", "elementwise_subtract", "elementwise_multiply",
        "elementwise_divide", "relu", "matmul", "matmul_tiled",
    )
    assert cpp.TENSOR_CORE_KERNELS == (
        "relu", "add", "subtract", "multiply", "matmul",
    )
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm",     # Phase F, milestone F2 (unrelated to D11)
        "NativeBatchNorm1d",   # Phase F, milestone F3 (unrelated to D11)
        "NativeBatchNorm2d",   # Phase F, milestone F4 (unrelated to D11)
        "NativeDropout",       # Phase G, milestone G4 (unrelated to D11)
    )
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert "conv2d" in cpp.AUTOGRAD_OPS and "maxpool2d" in cpp.AUTOGRAD_OPS
    # The proof is an integration result, not a new named capability.
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.NATIVE_MODULES):
        assert not [n for n in inventory if "train" in n.lower()]


def test_phase_d_completion_artifacts_exist():
    # D12 closed Phase D: the cross-cutting completion tests and the CNN
    # benchmark ship alongside this proof.
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    assert (repo_root / "tests" / "test_native_phase_d.py").is_file()
    assert (repo_root / "benchmarks" / "benchmark_native_cnn.py").is_file()


def test_the_model_is_a_native_module_stack():
    model = build_model()
    assert isinstance(model, NativeModule)
    assert all(isinstance(child, NativeModule) for child in model)
    _close(model)
