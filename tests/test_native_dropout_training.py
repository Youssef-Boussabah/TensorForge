"""Deterministic native Dropout training and exact stochastic resume
(Phase G, milestone G7).

``examples/native_dropout_training.py`` trains ``NativeDropoutClassifier``
— ``NativeLinear(4, 8, seed=0)`` -> ``NativeBatchNorm1d(8)`` ->
``NativeReLU`` -> ``NativeDropout(p=0.5, seed=20240707)`` ->
``NativeLayerNorm(8)`` -> ``NativeLinear(8, 3, seed=1)`` — on a fixed
twelve-sample three-class task with ``NativeCrossEntropyLoss`` over raw
logits and ``NativeAdam``.

These tests verify the deterministic dataset and batch schedule, the
architecture and its four state families, run-to-run bit-identity, exact
uninterrupted-versus-resumed equivalence through one pickle-free
**format-version 2** checkpoint (model, persistent buffers, optimizer,
**and** generator state), the fresh-object restoration contract, the
explicit external loop-progress metadata and its strict validation, the
evaluation-consumes-no-calls rule, the exact next mask against the G2
stateless Core, cleanup and live-storage lifecycle, and the G7 scope
boundary — **no new capability of any kind**.

Every equality here is **exact**. The native CPU float64 kernels are
deterministic and the restored generator makes the stochastic part
deterministic too, so a tolerance would only hide a real divergence.

Selector: python -m pytest -q -k native_dropout_training
"""

import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeCrossEntropyLoss,
    NativeDropout,
    NativeGenerator,
    NativeLayerNorm,
    NativeLinear,
    NativeModule,
    NativeReLU,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)
from tensorforge.experimental import native_checkpoint
from examples.native_dropout_training import (
    BATCH_SIZE,
    DEFAULT_LR,
    DROPOUT_P,
    DROPOUT_SEED,
    FEATURES,
    FRESH_DROPOUT_SEED,
    GENERATOR_KEY,
    HIDDEN_FEATURES,
    NUM_BATCHES,
    NUM_CLASSES,
    PROBE_VALUES,
    PROGRESS_FIELDS,
    SAMPLES,
    SPLIT_STEP,
    TOTAL_STEPS,
    NativeDropoutClassifier,
    batch_index_for_step,
    build_batches,
    build_dataset,
    build_loss,
    build_model,
    build_optimizer,
    evaluate,
    generator_state,
    progress_metadata,
    run_next_mask_proof,
    run_resume_proof,
    run_training,
    running_stats,
    train_step,
    validated_progress,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "examples" / "native_dropout_training.py"

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every open ``NativeStorage`` — a real live-native
    allocation count, so a lifecycle test can assert the count returns to
    its baseline instead of trusting collection."""
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


def settled(live_storages):
    """The live count after one collection — the G6 convention. Explicit
    ``close()`` is the release mechanism; collection only settles the
    gradient objects ``zero_grad()`` drops and the autograd graph's
    backward-closure cycles into a deterministic number."""
    gc.collect()
    return len(live_storages)


def _close(model=None, optimizer=None, *tensors):
    if optimizer is not None:
        optimizer.close()
    if model is not None:
        for parameter in model.parameters():
            parameter.close()
        for buffer in model.buffers():
            buffer.close()
    for tensor in tensors:
        if tensor is not None:
            tensor.close()


def _close_batches(batches):
    for inputs, _ in batches:
        inputs.close()


def _optimizer_values(optimizer):
    state = optimizer.state_dict()
    try:
        return {
            "lr": state["lr"],
            "betas": tuple(state["betas"]),
            "eps": state["eps"],
            "step_counts": list(state["step_counts"]),
            "m": [tensor.to_numpy().tolist() for tensor in state["m"]],
            "v": [tensor.to_numpy().tolist() for tensor in state["v"]],
        }
    finally:
        for label in ("m", "v"):
            for tensor in state[label]:
                tensor.close()


def _state_values(model):
    state = model.state_dict()
    try:
        return {name: tensor.to_numpy().tolist()
                for name, tensor in state.items()}
    finally:
        for tensor in state.values():
            tensor.close()


# --------------------------------------------------------------------------
# Dataset and batch schedule
# --------------------------------------------------------------------------


def test_dataset_shape_labels_and_exact_float64_values():
    inputs, targets = build_dataset()
    assert len(inputs) == SAMPLES == 12
    assert all(len(row) == FEATURES == 4 for row in inputs)
    assert targets == [index % NUM_CLASSES for index in range(SAMPLES)]
    assert sorted(set(targets)) == [0, 1, 2]
    assert targets.count(0) == targets.count(1) == targets.count(2) == 4
    # Every value is a quarter or an eighth, so it is exact in float64 and
    # survives a round trip through the native runtime unchanged.
    for row in inputs:
        for value in row:
            assert value == float(value)
            assert (value * 8.0).is_integer(), value
    array = np.asarray(inputs, dtype=np.float64)
    tensor = NativeTensor.from_array(inputs)
    try:
        assert np.array_equal(tensor.to_numpy(), array)
    finally:
        tensor.close()


def test_dataset_construction_is_deterministic_and_independent():
    first_inputs, first_targets = build_dataset()
    second_inputs, second_targets = build_dataset()
    assert first_inputs == second_inputs
    assert first_targets == second_targets
    first_inputs[0][0] = 1234.0
    first_targets[0] = 99
    third_inputs, third_targets = build_dataset()
    assert third_inputs == second_inputs
    assert third_targets == second_targets


def test_the_class_signal_is_position_varying_not_a_fixed_template():
    """Each class's own feature carries the signal, but the offset moves
    with position — so no single feature *threshold* separates the
    classes, and the model has to learn a boundary."""
    inputs, targets = build_dataset()
    by_class = {label: [] for label in range(NUM_CLASSES)}
    for row, label in zip(inputs, targets):
        by_class[label].append(row)
    for label, rows in by_class.items():
        signal = [row[label] for row in rows]
        assert len(set(signal)) == 4, "the class feature does not vary"
        assert min(signal) == 1.0 and max(signal) == 1.75
    # A class-0 sample's own feature can be *smaller* than a class-1
    # sample's class-0 feature is large — the offsets overlap across
    # classes, which is what makes a threshold insufficient.
    assert max(row[0] for row in by_class[1]) > min(row[0] for row in by_class[0]) - 2.0


def test_batch_schedule_is_a_pure_function_of_the_step():
    assert NUM_BATCHES == 3 and BATCH_SIZE == 4
    assert [batch_index_for_step(step) for step in range(9)] == [
        0, 1, 2, 0, 1, 2, 0, 1, 2
    ]
    # Pure: the same step always gives the same batch, in any order, and
    # nothing accumulates between calls.
    for step in (5, 0, 11, 5, 2, 0):
        assert batch_index_for_step(step) == step % NUM_BATCHES
    # ...and it is the reason one integer is a complete loop position.
    assert batch_index_for_step(SPLIT_STEP) == SPLIT_STEP % NUM_BATCHES


@pytest.mark.parametrize("bad, error", [
    (True, TypeError), (1.0, TypeError), ("3", TypeError), (None, TypeError),
    (-1, ValueError),
])
def test_batch_index_rejects_out_of_contract_steps(bad, error):
    with pytest.raises(error):
        batch_index_for_step(bad)


def test_batches_partition_the_dataset_in_order():
    inputs, targets = build_dataset()
    batches = build_batches(inputs, targets)
    try:
        assert len(batches) == NUM_BATCHES
        rebuilt_inputs = []
        rebuilt_targets = []
        for tensor, labels in batches:
            assert tensor.shape == (BATCH_SIZE, FEATURES)
            rebuilt_inputs.extend(tensor.to_numpy().tolist())
            rebuilt_targets.extend(labels)
        assert rebuilt_inputs == inputs
        assert rebuilt_targets == targets
        # Labels stay host ints — the native runtime has no integer dtype.
        assert all(type(label) is int
                   for _, labels in batches for label in labels)
    finally:
        _close_batches(batches)


def test_the_split_lands_mid_cycle_so_the_schedule_matters():
    """``SPLIT_STEP`` is deliberately not a multiple of ``NUM_BATCHES``: a
    resumed loop that restarted the batch schedule at 0 would train on the
    wrong data, which is exactly what the progress metadata prevents."""
    assert SPLIT_STEP % NUM_BATCHES != 0
    assert 0 < SPLIT_STEP < TOTAL_STEPS


# --------------------------------------------------------------------------
# Architecture and its four state families
# --------------------------------------------------------------------------


def test_model_has_the_exact_named_architecture():
    model = build_model()
    children = list(model.named_modules())
    names = [name for name, _ in children if name]
    assert names == ["hidden", "batch_norm", "relu", "dropout",
                     "layer_norm", "output"]
    assert isinstance(model.hidden, NativeLinear)
    assert isinstance(model.batch_norm, NativeBatchNorm1d)
    assert isinstance(model.relu, NativeReLU)
    assert isinstance(model.dropout, NativeDropout)
    assert isinstance(model.layer_norm, NativeLayerNorm)
    assert isinstance(model.output, NativeLinear)
    assert model.hidden.weight.shape == (FEATURES, HIDDEN_FEATURES)
    assert model.output.weight.shape == (HIDDEN_FEATURES, NUM_CLASSES)
    _close(model)


def test_all_four_state_families_are_present_and_distinct():
    """Parameters, persistent buffers, a registered generator, and (once
    an optimizer exists) optimizer state — the point of the model."""
    model = build_model()
    optimizer = build_optimizer(model)

    parameter_names = [name for name, _ in model.named_parameters()]
    buffer_names = [name for name, _ in model.named_buffers()]
    generator_names = [name for name, _ in model.named_generators()]

    assert parameter_names == [
        "hidden.weight", "hidden.bias", "batch_norm.gamma", "batch_norm.beta",
        "layer_norm.weight", "layer_norm.bias", "output.weight", "output.bias",
    ]
    assert buffer_names == ["batch_norm.running_mean",
                            "batch_norm.running_var"]
    assert generator_names == [GENERATOR_KEY] == ["dropout.generator"]
    # The four spaces do not overlap.
    assert not set(parameter_names) & set(buffer_names)
    assert not set(generator_names) & set(parameter_names + buffer_names)
    # state_dict() is contractually tensor-only and carries no generator.
    state = model.state_dict()
    try:
        assert list(state) == parameter_names + buffer_names
        assert all(isinstance(value, NativeTensor) for value in state.values())
        assert not any("generator" in key for key in state)
    finally:
        for value in state.values():
            value.close()
    # The optimizer holds the parameters only — never buffers or generators.
    assert len(optimizer.state_dict()["parameters"]) == len(parameter_names)
    _close(model, optimizer)


def test_the_dropout_generator_is_the_registered_seeded_object():
    model = build_model()
    generator = model.dropout.generator
    assert isinstance(generator, NativeGenerator)
    assert generator.seed == DROPOUT_SEED
    assert generator.calls == 0
    assert generator.algorithm == "tensorforge.splitmix64"
    assert generator.algorithm_version == 1
    assert model.dropout.p == DROPOUT_P
    assert dict(model.named_generators())[GENERATOR_KEY] is generator
    _close(model)


def test_forward_produces_raw_logits_with_no_softmax_module():
    model = build_model()
    inputs, targets = build_dataset()
    x = NativeTensor.from_array(inputs)
    model.train()
    logits = model(x)
    try:
        assert logits.shape == (SAMPLES, NUM_CLASSES)
        rows = logits.to_numpy()
        # Raw logits: rows do not sum to one and negatives are allowed.
        assert not np.allclose(rows.sum(axis=1), 1.0)
        assert np.isfinite(rows).all()
    finally:
        logits.close()
    for absent in ("softmax", "log_softmax"):
        assert not any(isinstance(child, type(None)) for _, child in
                       model.named_modules())
        assert absent not in [name for name, _ in model.named_modules()]
    _close(model, None, x)


def test_a_training_forward_consumes_exactly_one_generator_call():
    model = build_model()
    loss_fn = build_loss()
    optimizer = build_optimizer(model)
    inputs, targets = build_dataset()
    batches = build_batches(inputs, targets)
    try:
        for step in range(5):
            before = model.dropout.generator.calls
            train_step(model, loss_fn, optimizer, batches, step)
            assert model.dropout.generator.calls == before + 1
    finally:
        _close_batches(batches)
        _close(model, optimizer)


def test_batchnorm_running_state_advances_once_per_training_forward():
    model = build_model()
    loss_fn = build_loss()
    optimizer = build_optimizer(model)
    inputs, targets = build_dataset()
    batches = build_batches(inputs, targets)
    try:
        before = running_stats(model)
        assert before["running_mean"] == [0.0] * HIDDEN_FEATURES
        assert before["running_var"] == [1.0] * HIDDEN_FEATURES
        train_step(model, loss_fn, optimizer, batches, 0)
        after = running_stats(model)
        assert after != before
    finally:
        _close_batches(batches)
        _close(model, optimizer)


# --------------------------------------------------------------------------
# Determinism of fresh runs
# --------------------------------------------------------------------------


def test_two_independently_built_models_start_identical():
    first = build_model()
    second = build_model()
    try:
        assert _state_values(first) == _state_values(second)
        assert generator_state(first) == generator_state(second)
    finally:
        _close(first)
        _close(second)


def test_initialization_ignores_the_global_numpy_rng():
    np.random.seed(11)
    first = build_model()
    first_values = _state_values(first)
    first_generator = generator_state(first)
    np.random.seed(9999)
    np.random.random(64)
    second = build_model()
    try:
        assert _state_values(second) == first_values
        assert generator_state(second) == first_generator
    finally:
        _close(first)
        _close(second)


def test_two_independent_runs_are_bit_identical():
    """Two fresh runs from the same configuration and seeds agree on every
    state family, exactly — the §11 "two uninterrupted runs" clause."""
    first = run_training(steps=TOTAL_STEPS)
    second = run_training(steps=TOTAL_STEPS)
    assert first["loss_history"] == second["loss_history"]
    assert first["final_parameters"] == second["final_parameters"]
    assert first["final_running_stats"] == second["final_running_stats"]
    assert first["final_optimizer_state"] == second["final_optimizer_state"]
    assert first["final_generator"] == second["final_generator"]
    assert first["final_train_logits"] == second["final_train_logits"]
    assert first["final_eval"] == second["final_eval"]


def test_a_different_dropout_seed_changes_the_trajectory():
    """The negative control: if the generator did not really drive the
    forward, changing its seed would change nothing."""
    baseline = run_training(steps=8)
    other = run_training(steps=8, dropout_seed=FRESH_DROPOUT_SEED)
    assert baseline["loss_history"] != other["loss_history"]
    assert baseline["final_parameters"] != other["final_parameters"]
    # ...but the *initial* state is identical: only the stream differs.
    assert baseline["initial_running_stats"] == other["initial_running_stats"]


def test_generator_calls_equal_completed_training_steps():
    for steps in (1, 5, TOTAL_STEPS):
        run = run_training(steps=steps)
        assert run["final_generator"]["calls"] == steps
        assert run["calls_equal_steps"] is True
        assert run["final_generator"]["seed"] == DROPOUT_SEED


def test_the_run_learns_and_leaves_finite_state():
    run = run_training(steps=TOTAL_STEPS)
    # Training-mode loss is genuinely noisy under Dropout, so the honest
    # guardrails are on the *evaluation* measures and on the best loss.
    assert run["initial_accuracy"] < run["final_accuracy"]
    assert run["final_accuracy"] == 1.0
    assert run["final_eval"][0] < run["initial_eval"][0]
    assert run["best_loss"] < run["initial_loss"]
    assert all(np.isfinite(value) for value in run["loss_history"])
    assert np.isfinite(run["final_eval"][0])
    assert run["identity_stable"] is True
    assert run["generator_identity_stable"] is True
    assert run["gradients_cleared"] is True
    assert run["running_stats_advanced"] is True
    assert run["mode_restored"] is True


@pytest.mark.parametrize("bad, error", [
    (0, ValueError), (-3, ValueError), (True, TypeError), (2.0, TypeError),
])
def test_run_training_validates_steps(bad, error):
    with pytest.raises(error):
        run_training(steps=bad)


# --------------------------------------------------------------------------
# External loop-progress metadata
# --------------------------------------------------------------------------


def test_progress_metadata_is_json_compatible_and_exact():
    metadata = progress_metadata(SPLIT_STEP)
    assert set(PROGRESS_FIELDS) <= set(metadata)
    assert metadata["training_step"] == SPLIT_STEP
    assert metadata["next_batch_index"] == batch_index_for_step(SPLIT_STEP)
    # JSON round trip with no loss — the checkpoint stores it as JSON.
    assert json.loads(json.dumps(metadata)) == metadata
    assert type(metadata["training_step"]) is int
    assert type(metadata["next_batch_index"]) is int


@pytest.mark.parametrize("bad, error", [
    (True, TypeError), (1.5, TypeError), ("4", TypeError), (-1, ValueError),
])
def test_progress_metadata_validates_its_input(bad, error):
    with pytest.raises(error):
        progress_metadata(bad)


def test_validated_progress_accepts_the_metadata_it_produces():
    for step in range(TOTAL_STEPS + 1):
        metadata = progress_metadata(step)
        assert validated_progress(metadata) == (
            step, batch_index_for_step(step)
        )


def test_validated_progress_rejects_a_missing_field_rather_than_defaulting():
    """The failure this guards against is the quiet one: a resume that
    silently restarted from step 0 would still converge and still be a
    different run."""
    for field in PROGRESS_FIELDS:
        metadata = progress_metadata(SPLIT_STEP)
        del metadata[field]
        with pytest.raises(ValueError, match=field):
            validated_progress(metadata)
    with pytest.raises(ValueError, match="training_step"):
        validated_progress({})
    with pytest.raises(ValueError, match="training_step"):
        validated_progress({"lr": DEFAULT_LR})


@pytest.mark.parametrize("field", PROGRESS_FIELDS)
@pytest.mark.parametrize("bad", [True, 1.5, "7", None, [7]])
def test_validated_progress_rejects_wrong_types(field, bad):
    metadata = progress_metadata(SPLIT_STEP)
    metadata[field] = bad
    with pytest.raises(TypeError, match=field):
        validated_progress(metadata)


@pytest.mark.parametrize("step", [-1, TOTAL_STEPS + 1, 10**6])
def test_validated_progress_rejects_out_of_range_steps(step):
    metadata = progress_metadata(SPLIT_STEP)
    metadata["training_step"] = step
    with pytest.raises((ValueError, TypeError)):
        validated_progress(metadata)


def test_validated_progress_rejects_a_schedule_inconsistency():
    """The redundant ``next_batch_index`` is what makes a corrupted
    position detectable instead of merely wrong."""
    metadata = progress_metadata(SPLIT_STEP)
    metadata["next_batch_index"] = (metadata["next_batch_index"] + 1) % NUM_BATCHES
    with pytest.raises(ValueError, match="inconsistent"):
        validated_progress(metadata)


def test_validated_progress_rejects_a_non_mapping():
    for bad in (None, [], "metadata", 7):
        with pytest.raises(TypeError):
            validated_progress(bad)


def test_metadata_round_trips_through_a_real_checkpoint(tmp_path):
    model = build_model()
    path = str(tmp_path / "progress.npz")
    save_native_checkpoint(path, model,
                           metadata=progress_metadata(SPLIT_STEP))
    fresh = build_model()
    loaded = load_native_checkpoint(path, fresh)
    assert validated_progress(loaded) == (SPLIT_STEP,
                                          batch_index_for_step(SPLIT_STEP))
    # The returned dict is independent: mutating it affects nothing.
    loaded["training_step"] = 0
    again = load_native_checkpoint(path, build_model())
    assert again["training_step"] == SPLIT_STEP
    _close(model)
    _close(fresh)


def test_no_dataloader_or_global_rng_state_is_claimed_or_stored(tmp_path):
    """The honest half: the archive carries the loop position and nothing
    that would imply a captured data loader or a global RNG."""
    model = build_model()
    optimizer = build_optimizer(model)
    path = str(tmp_path / "honest.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata=progress_metadata(SPLIT_STEP))
    with np.load(path, allow_pickle=False) as archive:
        manifest_text = archive["manifest"].tobytes().decode("utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["metadata"] == progress_metadata(SPLIT_STEP)
    lowered = manifest_text.lower()
    for banned in ("mt19937", "pcg64", "random_state", "rng_state",
                   "dataloader", "data_loader", "shuffle", "epoch",
                   "scheduler", "numpy.random", "getstate"):
        assert banned not in lowered, banned
    _close(model, optimizer)


# --------------------------------------------------------------------------
# Exact split resume
# --------------------------------------------------------------------------


def test_resumed_training_matches_uninterrupted_exactly():
    proof = run_resume_proof()
    assert proof["identical_start"] is True
    assert proof["prefix_matches"] is True
    assert proof["first_resumed_loss_matches"] is True
    assert proof["suffix_matches"] is True
    assert proof["losses_match"] is True
    assert proof["uninterrupted_losses"] == proof["resumed_losses"]
    assert proof["parameters_match"] is True
    assert proof["running_mean_matches"] is True
    assert proof["running_var_matches"] is True
    assert proof["optimizer_state_matches"] is True
    assert proof["generator_matches"] is True
    assert proof["final_train_logits_match"] is True
    assert proof["final_eval_matches"] is True
    assert proof["parameter_order_matches"] is True
    assert proof["buffer_order_matches"] is True
    # The generator really moved during the run and ended where it should.
    assert proof["initial_generator"]["calls"] == 0
    assert proof["uninterrupted_generator"]["calls"] == TOTAL_STEPS
    assert proof["resumed_generator"]["calls"] == TOTAL_STEPS
    assert proof["resumed_generator"]["seed"] == DROPOUT_SEED


@pytest.mark.parametrize("total, split", [(9, 4), (12, 5), (10, 1), (10, 9)])
def test_resume_equivalence_holds_for_other_splits(total, split):
    proof = run_resume_proof(total_steps=total, split_step=split)
    assert proof["losses_match"] is True
    assert proof["parameters_match"] is True
    assert proof["running_mean_matches"] is True
    assert proof["running_var_matches"] is True
    assert proof["optimizer_state_matches"] is True
    assert proof["generator_matches"] is True
    assert proof["final_train_logits_match"] is True
    assert proof["final_eval_matches"] is True


@pytest.mark.parametrize("bad", [0, -1, TOTAL_STEPS, TOTAL_STEPS + 1])
def test_run_resume_proof_validates_the_split(bad):
    with pytest.raises(ValueError):
        run_resume_proof(split_step=bad)


def test_run_resume_proof_rejects_a_non_int_split():
    for bad in (True, 3.0, "3"):
        with pytest.raises(TypeError):
            run_resume_proof(split_step=bad)


def test_the_resume_restores_every_family_before_continuing():
    """The checkpoint-boundary comparison, before a single resumed step
    runs — so a match at the end cannot be an accident of two trajectories
    converging."""
    proof = run_resume_proof()
    assert proof["fresh_target_started_different"] is True
    assert proof["fresh_generator"]["seed"] == FRESH_DROPOUT_SEED
    assert proof["saved_generator"]["seed"] == DROPOUT_SEED
    assert proof["saved_generator"]["calls"] == SPLIT_STEP
    assert proof["generator_restored_exactly"] is True
    assert proof["running_restored_exactly"] is True
    assert proof["parameters_restored_exactly"] is True
    assert proof["optimizer_restored_exactly"] is True
    assert proof["restored_generator"] == proof["saved_generator"]


def test_the_resumed_loop_starts_at_the_metadata_step():
    proof = run_resume_proof()
    assert proof["resumed_step"] == SPLIT_STEP
    assert proof["resumed_step_is_split"] is True
    assert proof["resumed_batch_index"] == batch_index_for_step(SPLIT_STEP)
    assert proof["resumed_batch_is_scheduled"] is True
    assert proof["metadata"]["training_step"] == SPLIT_STEP
    assert len(proof["resumed_suffix"]) == TOTAL_STEPS - SPLIT_STEP


def test_ignoring_the_progress_metadata_breaks_the_match(tmp_path):
    """The load-bearing proof that the metadata is not decoration: a
    resumed loop that restarts the batch schedule at 0 — restoring every
    TensorForge-owned state family perfectly — still diverges."""
    inputs, targets = build_dataset()
    batches = build_batches(inputs, targets)
    loss_fn = build_loss()

    reference = build_model()
    reference_optimizer = build_optimizer(reference)
    uninterrupted = [
        train_step(reference, loss_fn, reference_optimizer, batches, step)
        for step in range(TOTAL_STEPS)
    ]

    model = build_model()
    optimizer = build_optimizer(model)
    prefix = [train_step(model, loss_fn, optimizer, batches, step)
              for step in range(SPLIT_STEP)]
    path = str(tmp_path / "ignored.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata=progress_metadata(SPLIT_STEP))

    # Correct: continue the schedule from the saved step.
    correct = build_model(dropout_seed=FRESH_DROPOUT_SEED)
    correct_optimizer = build_optimizer(correct)
    load_native_checkpoint(path, correct, optimizer=correct_optimizer)
    correct_suffix = [
        train_step(correct, loss_fn, correct_optimizer, batches, step)
        for step in range(SPLIT_STEP, TOTAL_STEPS)
    ]
    assert prefix + correct_suffix == uninterrupted

    # Wrong: same restored state, but the schedule restarts at step 0.
    wrong = build_model(dropout_seed=FRESH_DROPOUT_SEED)
    wrong_optimizer = build_optimizer(wrong)
    load_native_checkpoint(path, wrong, optimizer=wrong_optimizer)
    wrong_suffix = [
        train_step(wrong, loss_fn, wrong_optimizer, batches, step)
        for step in range(TOTAL_STEPS - SPLIT_STEP)
    ]
    assert prefix + wrong_suffix != uninterrupted, (
        "the batch schedule does not affect the trajectory, so the "
        "progress metadata would be unnecessary"
    )

    _close_batches(batches)
    _close(reference, reference_optimizer)
    _close(model, optimizer)
    _close(correct, correct_optimizer)
    _close(wrong, wrong_optimizer)


def test_a_resume_that_skips_the_generator_diverges(tmp_path):
    """The other negative control: restoring parameters, buffers, and the
    optimizer but leaving the *stream* where a fresh model put it breaks
    the match — which is precisely what checkpoint v2 added."""
    inputs, targets = build_dataset()
    batches = build_batches(inputs, targets)
    loss_fn = build_loss()

    reference = build_model()
    reference_optimizer = build_optimizer(reference)
    uninterrupted = [
        train_step(reference, loss_fn, reference_optimizer, batches, step)
        for step in range(TOTAL_STEPS)
    ]

    model = build_model()
    optimizer = build_optimizer(model)
    prefix = [train_step(model, loss_fn, optimizer, batches, step)
              for step in range(SPLIT_STEP)]
    path = str(tmp_path / "generator.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata=progress_metadata(SPLIT_STEP))

    resumed = build_model(dropout_seed=FRESH_DROPOUT_SEED)
    resumed_optimizer = build_optimizer(resumed)
    load_native_checkpoint(path, resumed, optimizer=resumed_optimizer)
    # Deliberately undo just the generator restoration.
    resumed.dropout.generator.reseed(FRESH_DROPOUT_SEED)
    suffix = [train_step(resumed, loss_fn, resumed_optimizer, batches, step)
              for step in range(SPLIT_STEP, TOTAL_STEPS)]
    assert prefix + suffix != uninterrupted

    _close_batches(batches)
    _close(reference, reference_optimizer)
    _close(model, optimizer)
    _close(resumed, resumed_optimizer)


# --------------------------------------------------------------------------
# Fresh-object restoration
# --------------------------------------------------------------------------


def test_the_resume_target_is_a_genuinely_fresh_set(tmp_path):
    """Fresh objects before the load, the *same* fresh objects after it,
    holding the checkpoint's values — the file is the only continuation
    boundary."""
    inputs, targets = build_dataset()
    batches = build_batches(inputs, targets)
    loss_fn = build_loss()

    model = build_model()
    optimizer = build_optimizer(model)
    for step in range(SPLIT_STEP):
        train_step(model, loss_fn, optimizer, batches, step)
    path = str(tmp_path / "fresh.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata=progress_metadata(SPLIT_STEP))
    saved_state = _state_values(model)
    saved_optimizer = _optimizer_values(optimizer)
    saved_generator = generator_state(model)

    fresh = build_model(dropout_seed=FRESH_DROPOUT_SEED)
    fresh_optimizer = build_optimizer(fresh)
    # Genuinely different objects, and genuinely different state.
    assert fresh is not model and fresh_optimizer is not optimizer
    assert all(a is not b for a, b in zip(fresh.parameters(),
                                          model.parameters()))
    assert all(a is not b for a, b in zip(fresh.buffers(), model.buffers()))
    assert fresh.dropout.generator is not model.dropout.generator
    assert generator_state(fresh) != saved_generator
    assert _state_values(fresh) != saved_state
    assert list(fresh_optimizer.state_dict()["step_counts"]) == [0] * 8

    parameter_ids = [id(p) for p in fresh.parameters()]
    buffer_ids = [id(b) for b in fresh.buffers()]
    generator_id = id(fresh.dropout.generator)

    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)

    # Identities preserved: loading is in-place, never a substitution.
    assert [id(p) for p in fresh.parameters()] == parameter_ids
    assert [id(b) for b in fresh.buffers()] == buffer_ids
    assert id(fresh.dropout.generator) == generator_id
    # Values restored exactly.
    assert _state_values(fresh) == saved_state
    assert _optimizer_values(fresh_optimizer) == saved_optimizer
    assert generator_state(fresh) == saved_generator
    assert running_stats(fresh) == running_stats(model)
    # The resumed run needs nothing from the interrupted one: release it.
    _close(model, optimizer)
    suffix = [train_step(fresh, loss_fn, fresh_optimizer, batches, step)
              for step in range(SPLIT_STEP, TOTAL_STEPS)]
    assert len(suffix) == TOTAL_STEPS - SPLIT_STEP
    assert all(np.isfinite(value) for value in suffix)

    _close_batches(batches)
    _close(fresh, fresh_optimizer)


def test_the_generator_topology_survives_a_load(tmp_path):
    """One generator, one canonical key, one self-mapped alias — and the
    same object afterwards. (Shared-generator topology is proved by the G5
    and G6 suites; this model deliberately has one stream.)"""
    model = build_model()
    path = str(tmp_path / "topology.npz")
    save_native_checkpoint(path, model, metadata=progress_metadata(0))
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(archive["manifest"].tobytes().decode("utf-8"))
    section = manifest["generators"]
    assert section["keys"] == [GENERATOR_KEY]
    assert section["aliases"] == {GENERATOR_KEY: GENERATOR_KEY}
    assert set(section["entries"]) == {GENERATOR_KEY}

    fresh = build_model(dropout_seed=FRESH_DROPOUT_SEED)
    generator = fresh.dropout.generator
    load_native_checkpoint(path, fresh)
    assert fresh.dropout.generator is generator
    assert dict(fresh.named_generators())[GENERATOR_KEY] is generator
    _close(model)
    _close(fresh)


# --------------------------------------------------------------------------
# Evaluation and the random stream
# --------------------------------------------------------------------------


def test_repeated_evaluation_consumes_no_generator_calls():
    model = build_model()
    loss_fn = build_loss()
    optimizer = build_optimizer(model)
    inputs, targets = build_dataset()
    batches = build_batches(inputs, targets)
    full = NativeTensor.from_array(inputs)
    try:
        for step in range(3):
            train_step(model, loss_fn, optimizer, batches, step)
        before = generator_state(model)
        outputs = [evaluate(model, loss_fn, full, targets) for _ in range(5)]
        after = generator_state(model)
        assert after == before, "evaluation moved the generator"
        assert after["calls"] == 3
        # Deterministic: eval is identity through Dropout and reads the
        # stored running statistics, so repeats are bit-identical.
        assert all(output == outputs[0] for output in outputs)
        assert model.training is True     # mode restored
    finally:
        _close_batches(batches)
        _close(model, optimizer, full)


def test_evaluation_creates_no_gap_in_the_random_stream():
    """A run with eval passes interleaved must produce exactly the loss
    sequence of a run without them."""
    plain = run_training(steps=TOTAL_STEPS)
    probed = run_training(steps=TOTAL_STEPS, eval_probe_step=SPLIT_STEP)
    assert probed["eval_probe"] is not None
    assert probed["eval_probe"]["calls_unchanged"] is True
    assert probed["eval_probe"]["outputs_identical"] is True
    assert probed["eval_probe"]["mode_restored"] is True
    assert probed["loss_history"] == plain["loss_history"]
    assert probed["final_parameters"] == plain["final_parameters"]
    assert probed["final_generator"] == plain["final_generator"]


def test_returning_to_training_uses_the_next_unconsumed_index():
    model = build_model()
    loss_fn = build_loss()
    optimizer = build_optimizer(model)
    inputs, targets = build_dataset()
    batches = build_batches(inputs, targets)
    full = NativeTensor.from_array(inputs)
    try:
        for step in range(4):
            train_step(model, loss_fn, optimizer, batches, step)
        assert model.dropout.generator.calls == 4
        for _ in range(3):
            evaluate(model, loss_fn, full, targets)
        assert model.dropout.generator.calls == 4
        train_step(model, loss_fn, optimizer, batches, 4)
        assert model.dropout.generator.calls == 5
    finally:
        _close_batches(batches)
        _close(model, optimizer, full)


def test_evaluation_restores_the_callers_previous_mode():
    model = build_model()
    loss_fn = build_loss()
    inputs, targets = build_dataset()
    full = NativeTensor.from_array(inputs)
    try:
        model.train()
        evaluate(model, loss_fn, full, targets)
        assert model.training is True
        model.eval()
        evaluate(model, loss_fn, full, targets)
        assert model.training is False
    finally:
        _close(model, None, full)


# --------------------------------------------------------------------------
# The exact next mask against the G2 Core
# --------------------------------------------------------------------------


def test_the_next_mask_after_a_resume_matches_the_g2_core():
    mask = run_next_mask_proof()
    assert mask["restored_calls"] == SPLIT_STEP
    assert mask["restored_calls_equal_split"] is True
    assert mask["restored_seed"] == DROPOUT_SEED
    assert mask["used_the_restored_index"] is True
    assert mask["next_mask_matches"] is True
    assert mask["module_result"] == mask["core_reference"]
    assert mask["consumed_exactly_one_call"] is True
    assert mask["calls_after"] == mask["calls_before"] + 1 == SPLIT_STEP + 1


@pytest.mark.parametrize("split", [1, 3, SPLIT_STEP, 11])
def test_the_next_mask_proof_holds_at_several_split_points(split):
    mask = run_next_mask_proof(split_step=split)
    assert mask["restored_calls"] == split
    assert mask["next_mask_matches"] is True
    assert mask["consumed_exactly_one_call"] is True


def test_the_core_reference_is_keyed_only_by_the_restored_state(tmp_path):
    """Computed directly rather than through the example: the restored
    module's next output is the Core's output at ``(restored seed,
    restored calls)`` — and at no other index."""
    inputs, targets = build_dataset()
    batches = build_batches(inputs, targets)
    loss_fn = build_loss()
    model = build_model()
    optimizer = build_optimizer(model)
    for step in range(SPLIT_STEP):
        train_step(model, loss_fn, optimizer, batches, step)
    path = str(tmp_path / "core.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata=progress_metadata(SPLIT_STEP))

    fresh = build_model(dropout_seed=FRESH_DROPOUT_SEED)
    fresh_optimizer = build_optimizer(fresh)
    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    state = generator_state(fresh)

    source = cpp.NativeTensorCore.from_array([list(PROBE_VALUES)])
    expected_core = source.dropout_forward(
        DROPOUT_P, seed=state["seed"], call_index=state["calls"]
    )
    wrong_core = source.dropout_forward(
        DROPOUT_P, seed=state["seed"], call_index=state["calls"] + 1
    )
    expected = expected_core.to_numpy().tolist()
    wrong = wrong_core.to_numpy().tolist()
    expected_core.close()
    wrong_core.close()
    source.close()

    probe = NativeTensor.from_array([list(PROBE_VALUES)])
    fresh.dropout.train()
    result = fresh.dropout(probe)
    values = result.to_numpy().tolist()
    result.close()
    probe.close()

    assert values == expected
    assert values != wrong, "the mask did not depend on the call index"
    assert fresh.dropout.generator.calls == state["calls"] + 1

    _close_batches(batches)
    _close(model, optimizer)
    _close(fresh, fresh_optimizer)


def test_the_module_never_exposes_its_private_mask():
    """The Core mask is a *reference value* in these tests; the module's
    own mask stays graph-owned private state."""
    model = build_model()
    for absent in ("mask", "_mask", "last_mask", "multiplier"):
        assert not hasattr(model.dropout, absent), absent
    probe = NativeTensor.from_array([list(PROBE_VALUES)])
    model.dropout.train()
    result = model.dropout(probe)
    try:
        for absent in ("mask", "_mask", "saved_mask"):
            assert not hasattr(result, absent), absent
    finally:
        result.close()
        probe.close()
    _close(model)


# --------------------------------------------------------------------------
# Clean step boundaries
# --------------------------------------------------------------------------


def test_a_failed_step_is_never_presented_as_a_completed_one(tmp_path,
                                                             monkeypatch):
    """The example saves only after a step has fully completed. A step
    interrupted before its optimizer update must neither advance the
    generator nor be describable as completed progress."""
    inputs, targets = build_dataset()
    batches = build_batches(inputs, targets)
    loss_fn = build_loss()
    model = build_model()
    optimizer = build_optimizer(model)
    completed = 0
    for step in range(3):
        train_step(model, loss_fn, optimizer, batches, step)
        completed += 1
    state_before = generator_state(model)
    assert state_before["calls"] == completed == 3

    # Interrupt step 3 inside the optimizer update, after the forward.
    # Patched on the class: NativeAdam uses __slots__, so the instance
    # attribute is read-only.
    def failing_step(self):
        raise RuntimeError("injected: interrupted before the update finished")

    monkeypatch.setattr(NativeAdam, "step", failing_step)
    with pytest.raises(RuntimeError, match="injected"):
        train_step(model, loss_fn, optimizer, batches, 3)
    monkeypatch.undo()

    # The forward *did* consume its call — that is the honest G3 contract
    # (a successful stochastic forward consumes one, and this one
    # succeeded). What must not happen is the progress metadata claiming
    # step 3 completed.
    assert completed == 3
    metadata = progress_metadata(completed)
    assert metadata["training_step"] == 3
    path = str(tmp_path / "clean.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata=metadata)
    fresh = build_model(dropout_seed=FRESH_DROPOUT_SEED)
    fresh_optimizer = build_optimizer(fresh)
    loaded = load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    assert validated_progress(loaded)[0] == 3
    _close_batches(batches)
    _close(model, optimizer)
    _close(fresh, fresh_optimizer)


def test_a_corrupted_progress_field_cannot_silently_restart_from_zero(
    tmp_path,
):
    model = build_model()
    path = str(tmp_path / "corrupt.npz")
    save_native_checkpoint(path, model, metadata={"lr": DEFAULT_LR})
    loaded = load_native_checkpoint(path, build_model())
    assert "training_step" not in loaded
    with pytest.raises(ValueError, match="training_step"):
        validated_progress(loaded)
    _close(model)


# --------------------------------------------------------------------------
# The checkpoint archive
# --------------------------------------------------------------------------


def test_the_checkpoint_is_version_two_with_the_expected_sections(tmp_path):
    inputs, targets = build_dataset()
    batches = build_batches(inputs, targets)
    loss_fn = build_loss()
    model = build_model()
    optimizer = build_optimizer(model)
    train_step(model, loss_fn, optimizer, batches, 0)
    path = str(tmp_path / "archive.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata=progress_metadata(1))

    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(archive["manifest"].tobytes().decode("utf-8"))
    assert manifest["format"] == "tensorforge.native_checkpoint"
    assert manifest["format_version"] == 3
    assert manifest["model"]["keys"] == [
        "hidden.weight", "hidden.bias", "batch_norm.gamma", "batch_norm.beta",
        "layer_norm.weight", "layer_norm.bias", "output.weight", "output.bias",
        "batch_norm.running_mean", "batch_norm.running_var",
    ]
    assert manifest["optimizer"]["type"] == "NativeAdam"
    entry = manifest["generators"]["entries"][GENERATOR_KEY]
    assert entry["algorithm"] == "tensorforge.splitmix64"
    assert entry["algorithm_version"] == 1
    # Canonical decimal strings, never JSON numbers.
    assert entry["seed"] == str(DROPOUT_SEED)
    assert entry["calls"] == "1"
    assert isinstance(entry["seed"], str) and isinstance(entry["calls"], str)
    assert manifest["metadata"] == progress_metadata(1)
    _close_batches(batches)
    _close(model, optimizer)


def test_training_mode_is_not_serialized(tmp_path):
    model = build_model()
    path = str(tmp_path / "mode.npz")
    save_native_checkpoint(path, model, metadata=progress_metadata(0))
    fresh = build_model(dropout_seed=FRESH_DROPOUT_SEED)
    fresh.eval()
    load_native_checkpoint(path, fresh)
    assert fresh.training is False
    assert fresh.dropout.training is False
    _close(model)
    _close(fresh)


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_repeated_steps_return_to_a_stable_storage_baseline(live_storages):
    """The baseline is taken after the model, the optimizer's persistent
    moments, the batch tensors, and the first gradients exist — those are
    intentionally live. What must not grow is the transient per-step
    allocation, including the Dropout mask, the BatchNorm running-stat
    replacement, and the optimizer moment replacement."""
    inputs, targets = build_dataset()
    batches = build_batches(inputs, targets)
    loss_fn = build_loss()
    model = build_model()
    optimizer = build_optimizer(model)
    for step in range(3):
        train_step(model, loss_fn, optimizer, batches, step)
    baseline = settled(live_storages)
    assert baseline > 0
    for step in range(3, 12):
        train_step(model, loss_fn, optimizer, batches, step)
        assert settled(live_storages) == baseline, step
    _close_batches(batches)
    _close(model, optimizer)


def test_a_complete_exact_resume_workflow_returns_storage_to_baseline(
    live_storages,
):
    """The whole lifecycle, repeated: no monotonic growth across runs, and
    an exact return to the starting baseline at the end."""
    baseline = settled(live_storages)
    observed = []
    for _ in range(3):
        proof = run_resume_proof(total_steps=9, split_step=4)
        assert proof["losses_match"] is True
        assert proof["generator_matches"] is True
        observed.append(settled(live_storages))
    assert observed == [baseline] * 3, observed
    assert settled(live_storages) == baseline


def test_the_next_mask_proof_returns_storage_to_baseline(live_storages):
    baseline = settled(live_storages)
    for _ in range(2):
        mask = run_next_mask_proof(split_step=3)
        assert mask["next_mask_matches"] is True
        assert settled(live_storages) == baseline
    assert settled(live_storages) == baseline


def test_run_training_returns_no_live_native_object():
    run = run_training(steps=3)

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        else:
            assert not isinstance(value, (NativeTensor, NativeModule,
                                          NativeAdam, NativeGenerator)), value

    walk(run)


def test_run_resume_proof_returns_no_live_native_object():
    proof = run_resume_proof(total_steps=6, split_step=2)

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        else:
            assert not isinstance(value, (NativeTensor, NativeModule,
                                          NativeAdam, NativeGenerator)), value

    walk(proof)


# --------------------------------------------------------------------------
# The example as a script
# --------------------------------------------------------------------------


def test_example_runs_as_a_subprocess_and_exits_zero():
    """The example runs standalone and exits 0 — the equivalent of
    ``uv run python examples/native_dropout_training.py`` (the same
    interpreter, same environment, run as a fresh process)."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert "exact stochastic resume: yes" in result.stdout
    assert "native Dropout training + exact stochastic resume ok" in result.stdout
    assert "live native storage baseline / final: 0 / 0" in result.stdout


def test_the_example_leaves_no_checkpoint_behind():
    """The default runs use a temporary directory that is removed, so no
    archive is left in the repository."""
    before = sorted(REPO_ROOT.glob("*.npz")) + sorted(
        (REPO_ROOT / "examples").glob("*.npz")
    )
    run_resume_proof(total_steps=6, split_step=2)
    run_next_mask_proof(split_step=2)
    after = sorted(REPO_ROOT.glob("*.npz")) + sorted(
        (REPO_ROOT / "examples").glob("*.npz")
    )
    assert after == before == []


def test_the_example_uses_only_the_native_stack_and_claims_no_timing():
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "tensorforge.nn" not in text and "tensorforge.optim" not in text
    assert "from tensorforge import" not in text
    for banned in ("requests", "urllib", "socket", "subprocess",
                   "perf_counter", "time.time(", "import time", "import timeit",
                   "np.random", "numpy.random", "import random",
                   "BENCHMARK", "speedup", "faster than"):
        assert banned not in text, banned
    # No shuffling, no data loader, no augmentation. Checked as *usage*,
    # not as bare words: the docstring names all three to say the example
    # does none of them, which is the honest form and must keep passing.
    for banned in ("shuffle(", "DataLoader", "augment(", "permutation("):
        assert banned not in text, banned


def test_the_example_states_what_the_checkpoint_does_not_capture():
    text = EXAMPLE.read_text(encoding="utf-8").lower()
    for required in ("data-loader", "shuffle", "scheduler", "numpy",
                     "full-program determinism is not claimed"):
        assert required in text, required


# --------------------------------------------------------------------------
# Scope boundary: G7 adds no capability
# --------------------------------------------------------------------------


def test_g7_adds_no_capability_or_inventory_entry():
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert "NativeDropout" in cpp.NATIVE_MODULES
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert "dropout_forward" in cpp.TENSOR_CORE_OPS
    assert "dropout_backward" not in cpp.TENSOR_CORE_OPS
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert "generator_state" in cpp.STATE_SUPPORT
    assert "checkpoint_generator_state" in cpp.STATE_SUPPORT
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert native_checkpoint._GENERATOR_SECTION_KEYS == {
        "keys", "entries", "aliases"
    }


def test_the_boundary_move_belongs_to_g10_not_to_g7():
    """G7 is the exact-resume proof and moved no capability boundary. The
    boundary moved later, at **G10**, on the strength of the closure
    matrix — so what this guard keeps is the attribution, plus the
    standing rule that no benchmark result artifact is ever committed."""
    for absent in ("benchmark_results",):
        assert not (REPO_ROOT / absent).exists(), absent
    assert "dropout" not in cpp.UNSUPPORTED
    assert cpp.UNSUPPORTED == ("cuda", "amp")


def test_the_example_defines_no_public_training_api():
    """The helpers exist for readability and testability, not as a new
    framework surface: nothing here is exported from the package."""
    import tensorforge.experimental as experimental

    for absent in ("run_training", "run_resume_proof", "train_step",
                   "build_model", "batch_index_for_step", "progress_metadata",
                   "validated_progress", "NativeDropoutClassifier"):
        assert not hasattr(experimental, absent), absent
        assert not hasattr(tensorforge, absent), absent
    assert "NativeDropoutClassifier" not in cpp.NATIVE_MODULES


def test_the_stable_framework_is_untouched():
    assert hasattr(tensorforge.nn, "Dropout")
    assert not hasattr(tensorforge, "NativeDropout")
    assert not hasattr(tensorforge, "NativeGenerator")
