"""The native classification training and checkpoint-resume proof
(Phase E, milestone E8).

examples/native_classification_training.py trains
``NativeImageClassifier`` — ``NativeConv2d(1, 4, 3, seed=0)`` ->
``NativeReLU`` -> ``NativeMaxPool2d(2)`` -> ``NativeFlatten`` ->
``NativeLinear(16, 3, seed=1)`` — on twelve fixed 6x6 single-channel
images in three classes (vertical bar / horizontal bar / diagonal line,
each at four positions) for 40 deterministic ``NativeAdam(lr=0.05)``
steps, with ``NativeCrossEntropyLoss`` consuming the **raw logits**.
These tests verify the dataset and architecture, deterministic
initialization, forward / loss / backward / optimizer integration, the
learning guardrails, exact run-to-run determinism, exact
uninterrupted-versus-resumed equivalence through one pickle-free
checkpoint (model **and** optimizer state, format version 1), ownership
and saved-resource lifetime across steps, the NumPy boundary, the
failure paths, and the E8 scope boundary (no new capability of any
kind).

Every number asserted below was observed from the deterministic workload
first; the thresholds carry a wide margin (see
``test_loss_reduction_guardrails``).

NumPy appears only for references, inspection, and equality assertions;
the training computation is native, and ``test_one_training_step_is_
fully_native`` proves it with the numerical/conversion tripwire.

Selector: python -m pytest -q -k native_classification_training
"""

import gc
import math
from pathlib import Path

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeConv2d,
    NativeCrossEntropyLoss,
    NativeFlatten,
    NativeLinear,
    NativeMaxPool2d,
    NativeModule,
    NativeReLU,
    NativeTensor,
    load_native_checkpoint,
    native_accuracy,
    save_native_checkpoint,
)
from tensorforge.experimental import native_checkpoint
from examples.native_classification_training import (
    CLASS_NAMES,
    CONV_CHANNELS,
    DEFAULT_LR,
    FLAT_FEATURES,
    IMAGE_SIZE,
    IMAGE_VALUES,
    KERNEL_SIZE,
    NUM_CLASSES,
    SPLIT_STEP,
    TARGET_VALUES,
    TOTAL_STEPS,
    NativeImageClassifier,
    build_dataset,
    build_loss,
    build_model,
    build_optimizer,
    evaluate,
    main,
    run_resume_proof,
    run_training,
    train_step,
)

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "examples" / "native_classification_training.py"

# Observed once from the deterministic workload; asserted with margin.
OBSERVED_INITIAL_LOSS = 1.1596379162669588
OBSERVED_FINAL_LOSS = 0.0001007574877188413
OBSERVED_INITIAL_ACCURACY = 1.0 / 3.0
OBSERVED_FINAL_ACCURACY = 1.0
PARAMETER_NAMES = ["conv.weight", "conv.bias", "linear.weight", "linear.bias"]


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


def _inputs():
    images, targets = build_dataset()
    return NativeTensor.from_array(images), targets


def _close(model, optimizer=None):
    if optimizer is not None:
        optimizer.close()
    for parameter in model.parameters():
        parameter.close()


# --------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------

def test_dataset_shape_dtype_and_class_structure():
    images, targets = build_dataset()
    assert len(images) == len(targets) == 12
    for image in images:
        assert len(image) == 1                        # single channel
        assert len(image) == 1 and len(image[0]) == IMAGE_SIZE
        for row in image[0]:
            assert len(row) == IMAGE_SIZE
            assert all(isinstance(value, float) for value in row)
    # At least three classes, more than one example per class.
    assert len(CLASS_NAMES) == NUM_CLASSES == 3
    assert sorted(set(targets)) == [0, 1, 2]
    for label in range(NUM_CLASSES):
        assert targets.count(label) == 4
    # The images really differ within a class (position varies), so the
    # task is not "one fixed tensor per label".
    for label in range(NUM_CLASSES):
        members = [tuple(map(tuple, images[i][0]))
                   for i, t in enumerate(targets) if t == label]
        assert len(set(members)) == 4, f"class {label} repeats an image"
    # And no image is shared across classes.
    everything = [tuple(map(tuple, image[0])) for image in images]
    assert len(set(everything)) == 12


def test_targets_are_strict_host_integers():
    _, targets = build_dataset()
    for label in targets:
        assert type(label) is int          # not bool, not float, not np scalar
        assert not isinstance(label, bool)
        assert 0 <= label < NUM_CLASSES
    # The labels are host metadata: no native tensor holds them.
    assert not isinstance(targets, NativeTensor)


def test_dataset_construction_is_deterministic_and_independent():
    first_images, first_targets = build_dataset()
    second_images, second_targets = build_dataset()
    assert first_images == second_images == IMAGE_VALUES
    assert first_targets == second_targets == TARGET_VALUES
    # Fresh copies: mutating one result cannot perturb the literals or
    # the next call (no runtime generation, no shared nested lists).
    first_images[0][0][0][0] = 99.0
    first_targets[0] = 99
    third_images, third_targets = build_dataset()
    assert third_images == IMAGE_VALUES and third_targets == TARGET_VALUES


def test_native_input_tensor_matches_the_host_dataset():
    x, targets = _inputs()
    assert x.shape == (12, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert x.dtype == "float64" and x.device == "cpu"
    assert np.array_equal(x.to_numpy(), np.asarray(IMAGE_VALUES))
    assert len(targets) == x.shape[0]
    x.close()


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

def test_model_is_a_native_module_stack_with_the_required_layers():
    model = build_model()
    assert isinstance(model, NativeImageClassifier)
    assert isinstance(model, NativeModule)
    children = dict(model.named_modules())
    assert isinstance(model.conv, NativeConv2d)
    assert isinstance(model.relu, NativeReLU)
    assert isinstance(model.pool, NativeMaxPool2d)
    assert isinstance(model.flatten, NativeFlatten)
    assert isinstance(model.linear, NativeLinear)
    # Registered through the normal assignment path, not a hand-kept list.
    assert set(children) >= {"conv", "relu", "pool", "flatten", "linear"}
    assert all(isinstance(child, NativeModule) for child in children.values())
    _close(model)


def test_model_output_is_raw_logits_of_the_right_shape():
    model = build_model()
    x, targets = _inputs()
    logits = model(x)
    assert logits.shape == (len(TARGET_VALUES), NUM_CLASSES)
    assert logits.dtype == "float64" and logits.device == "cpu"
    values = logits.to_numpy()
    # Raw logits, not probabilities: rows do not sum to 1 and negatives
    # appear — nothing normalized the output.
    assert not np.allclose(values.sum(axis=1), 1.0)
    assert (values < 0.0).any()
    logits.close()
    x.close()
    _close(model)


def test_the_model_contains_no_softmax_or_log_softmax_layer():
    model = build_model()
    # No probability transform module exists at all (E0 §1 excludes both),
    # and the model's last child is the linear head.
    for absent in ("NativeSoftmax", "NativeLogSoftmax"):
        assert absent not in cpp.NATIVE_MODULES
    last = list(dict(model.named_modules()).values())[-1]
    assert isinstance(last, NativeLinear)
    # The forward really ends at the linear head: no transform is applied
    # to its output, so cross-entropy receives raw logits.
    source = EXAMPLE.read_text(encoding="utf-8")
    forward = source.split("def forward(self, images):", 1)[1].split(
        "\n\ndef ", 1)[0]
    assert "return self.linear(hidden)" in forward
    assert ".softmax(" not in forward and ".log_softmax(" not in forward
    _close(model)


def test_intermediate_shapes_through_the_stack():
    model = build_model()
    x, _ = _inputs()
    conv = model.conv(x)
    assert conv.shape == (12, CONV_CHANNELS, 4, 4)      # 6x6 -> 4x4
    relu = model.relu(conv)
    pooled = model.pool(relu)
    assert pooled.shape == (12, CONV_CHANNELS, 2, 2)    # floor((4-2)/2)+1
    flat = model.flatten(pooled)
    assert flat.shape == (12, FLAT_FEATURES)
    logits = model.linear(flat)
    assert logits.shape == (12, NUM_CLASSES)
    for t in (conv, relu, pooled, flat, logits, x):
        t.close()
    _close(model)


def test_parameter_names_state_keys_and_shapes():
    model = build_model()
    names = [name for name, _ in model.named_parameters()]
    assert names == PARAMETER_NAMES
    assert sorted(model.state_dict()) == sorted(PARAMETER_NAMES)
    # ReLU, pooling, and flatten contribute no parameters and no state.
    for child in (model.relu, model.pool, model.flatten):
        assert list(child.parameters()) == []
        assert child.state_dict() == {}
    assert list(model.buffers()) == []
    shapes = {name: p.shape for name, p in model.named_parameters()}
    assert shapes == {
        "conv.weight": (CONV_CHANNELS, 1, KERNEL_SIZE, KERNEL_SIZE),
        "conv.bias": (CONV_CHANNELS,),
        "linear.weight": (FLAT_FEATURES, NUM_CLASSES),
        "linear.bias": (NUM_CLASSES,),
    }
    for parameter in model.parameters():
        assert parameter.dtype == "float64" and parameter.device == "cpu"
    _close(model)


def test_the_loss_is_the_native_classification_loss_and_holds_no_state():
    loss_fn = build_loss()
    assert isinstance(loss_fn, NativeCrossEntropyLoss)
    assert loss_fn.reduction == "mean"
    assert loss_fn.state_dict() == {}
    assert list(loss_fn.parameters()) == [] and list(loss_fn.buffers()) == []


# --------------------------------------------------------------------------
# Deterministic initialization
# --------------------------------------------------------------------------

def test_two_independently_built_models_start_identical():
    first, second = build_model(), build_model()
    for (name_a, a), (name_b, b) in zip(first.named_parameters(),
                                        second.named_parameters()):
        assert name_a == name_b
        assert np.array_equal(a.to_numpy(), b.to_numpy()), name_a
        # Not degenerate: an all-zero or symmetric start would not learn.
        assert (a.to_numpy() != 0.0).any(), name_a
    x, _ = _inputs()
    out_a, out_b = first(x), second(x)
    assert np.array_equal(out_a.to_numpy(), out_b.to_numpy())
    for t in (out_a, out_b, x):
        t.close()
    _close(first)
    _close(second)


def test_initialization_ignores_the_global_numpy_rng():
    np.random.seed(1234)
    first = build_model()
    np.random.seed(4321)
    [np.random.random() for _ in range(10)]
    second = build_model()
    for (_, a), (_, b) in zip(first.named_parameters(),
                              second.named_parameters()):
        assert np.array_equal(a.to_numpy(), b.to_numpy())
    _close(first)
    _close(second)


def test_different_seeds_produce_different_models():
    seeded = NativeImageClassifier(conv_seed=0, linear_seed=1)
    other = NativeImageClassifier(conv_seed=7, linear_seed=8)
    assert not np.array_equal(seeded.conv.weight.to_numpy(),
                              other.conv.weight.to_numpy())
    _close(seeded)
    _close(other)


# --------------------------------------------------------------------------
# Forward / loss / backward integration
# --------------------------------------------------------------------------

def test_forward_loss_backward_reaches_every_trainable_parameter():
    model = build_model()
    loss_fn = build_loss()
    x, targets = _inputs()
    logits = model(x)
    loss = loss_fn(logits, targets)
    assert loss.shape == () and loss.numel == 1        # scalar loss
    loss.backward()
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        assert grad is not None, name
        assert grad.shape == parameter.shape, name
        values = grad.to_numpy()
        assert np.isfinite(values).all(), name
        assert (values != 0.0).any(), name
    for t in (loss, logits, x):
        t.close()
    _close(model)


def test_initial_loss_matches_the_uniform_prediction_reference():
    """The reported initial loss is a real cross-entropy, not an
    arbitrary number: it is within reach of ln(3), the loss of a uniform
    3-class prediction, and it is finite."""
    run = run_training(steps=1)
    assert math.isfinite(run["initial_loss"])
    assert abs(run["initial_loss"] - math.log(NUM_CLASSES)) < 0.25
    assert run["initial_loss"] == pytest.approx(OBSERVED_INITIAL_LOSS,
                                                rel=1e-12)
    assert run["initial_accuracy"] == pytest.approx(OBSERVED_INITIAL_ACCURACY)


def test_first_step_gradient_evidence_is_recorded():
    run = run_training(steps=1)
    assert run["gradient_shapes"] == {
        "conv.weight": (CONV_CHANNELS, 1, KERNEL_SIZE, KERNEL_SIZE),
        "conv.bias": (CONV_CHANNELS,),
        "linear.weight": (FLAT_FEATURES, NUM_CLASSES),
        "linear.bias": (NUM_CLASSES,),
    }
    assert all(run["gradient_nonzero"].values())


# --------------------------------------------------------------------------
# Learning behavior
# --------------------------------------------------------------------------

def test_loss_reduction_guardrails():
    """Thresholds chosen after observing the deterministic run: it goes
    from ~1.159638 to ~0.000101 (ratio ~8.7e-05). The guards below allow
    a final loss up to 0.05 and a ratio up to 0.05 — a ~500x margin over
    the observed values, so ordinary floating-point drift cannot make
    them fail spuriously. The curve need not be monotonic (Adam
    overshoots early); it must end far below where it started."""
    run = run_training()
    assert len(run["loss_history"]) == TOTAL_STEPS
    assert all(math.isfinite(value) for value in run["loss_history"])
    assert math.isfinite(run["initial_loss"]) and math.isfinite(run["final_loss"])
    assert run["initial_loss"] == pytest.approx(OBSERVED_INITIAL_LOSS, rel=1e-9)
    assert run["final_loss"] == pytest.approx(OBSERVED_FINAL_LOSS, abs=1e-5)
    assert run["final_loss"] < 0.05
    assert run["final_loss"] / run["initial_loss"] < 0.05
    assert run["final_loss"] < run["initial_loss"]
    assert min(run["loss_history"]) < 0.05


def test_accuracy_improves_and_reaches_a_reliable_level():
    run = run_training()
    assert run["initial_accuracy"] == pytest.approx(OBSERVED_INITIAL_ACCURACY)
    assert run["final_accuracy"] >= 0.90
    assert run["final_accuracy"] - run["initial_accuracy"] >= 0.5
    assert run["final_accuracy"] == pytest.approx(OBSERVED_FINAL_ACCURACY)
    # The untrained model collapses onto one class; the trained one does
    # not, and it gets every fixed example right.
    assert len(set(run["initial_predictions"])) == 1
    assert len(set(run["final_predictions"])) == NUM_CLASSES
    assert run["final_predictions"] == run["targets"]


def test_conv_and_linear_parameters_change():
    run = run_training()
    for name in PARAMETER_NAMES:
        before = np.asarray(run["initial_parameters"][name])
        after = np.asarray(run["final_parameters"][name])
        assert not np.array_equal(before, after), name
        assert np.abs(after - before).max() > 1e-3, name
        assert np.isfinite(after).all(), name
    # Both trainable layers moved — the convolution and the head.
    assert any(name.startswith("conv.") for name in PARAMETER_NAMES)
    assert any(name.startswith("linear.") for name in PARAMETER_NAMES)
    # Versions advanced exactly once per step, identities never moved.
    assert run["final_versions"] == [v + run["steps"]
                                     for v in run["initial_versions"]]
    assert run["identity_stable"] is True
    assert run["gradients_cleared"] is True
    assert run["state_keys"] == sorted(PARAMETER_NAMES)


def test_the_optimizer_accumulated_meaningful_state():
    run = run_training()
    state = run["optimizer_state"]
    assert state["optimizer"] == "NativeAdam"
    assert state["lr"] == DEFAULT_LR
    assert state["step_counts"] == [TOTAL_STEPS] * len(PARAMETER_NAMES)
    for moments in (state["m"], state["v"]):
        assert len(moments) == len(PARAMETER_NAMES)
        for buffer in moments:
            values = np.asarray(buffer)
            assert np.isfinite(values).all()
            assert (values != 0.0).any()   # real accumulated state


# --------------------------------------------------------------------------
# Determinism: two independent uninterrupted runs
# --------------------------------------------------------------------------

def test_two_independent_runs_are_exactly_equal():
    first = run_training()
    second = run_training()
    assert first["initial_parameters"] == second["initial_parameters"]
    assert first["loss_history"] == second["loss_history"]
    assert first["initial_loss"] == second["initial_loss"]
    assert first["final_loss"] == second["final_loss"]
    assert first["final_parameters"] == second["final_parameters"]
    assert first["final_logits"] == second["final_logits"]
    assert first["final_predictions"] == second["final_predictions"]
    assert first["final_accuracy"] == second["final_accuracy"]
    assert first["optimizer_state"] == second["optimizer_state"]
    # Exact, not approximate: the same native kernels in the same order.
    for name in PARAMETER_NAMES:
        assert np.array_equal(np.asarray(first["final_parameters"][name]),
                              np.asarray(second["final_parameters"][name]))


# --------------------------------------------------------------------------
# Exact checkpoint resume (the central E8 contract)
# --------------------------------------------------------------------------

def test_resumed_training_matches_uninterrupted_exactly():
    proof = run_resume_proof()
    assert proof["identical_start"] is True
    assert proof["prefix_matches"] is True
    assert proof["first_resumed_loss_matches"] is True
    assert proof["suffix_matches"] is True
    assert proof["losses_match"] is True
    assert proof["final_losses_match"] is True
    assert proof["logits_match"] is True
    assert proof["predictions_match"] is True
    assert proof["accuracies_match"] is True
    assert proof["parameters_match"] is True
    assert proof["optimizer_state_matches"] is True
    assert proof["parameter_order_matches"] is True
    assert proof["identities_preserved"] is True
    assert proof["metadata"] == {"steps_completed": SPLIT_STEP,
                                 "lr": DEFAULT_LR}
    assert len(proof["uninterrupted_losses"]) == TOTAL_STEPS
    assert len(proof["resumed_suffix"]) == TOTAL_STEPS - SPLIT_STEP
    # Not just the final loss: the whole remaining suffix, element by
    # element, exactly.
    assert (proof["resumed_suffix"]
            == proof["uninterrupted_losses"][SPLIT_STEP:])
    assert proof["state_keys"] == sorted(PARAMETER_NAMES)
    assert proof["final_accuracy_resumed"] >= 0.90


def test_resume_equivalence_holds_for_another_split():
    proof = run_resume_proof(total_steps=12, split_step=5)
    assert proof["losses_match"] is True
    assert proof["parameters_match"] is True
    assert proof["optimizer_state_matches"] is True
    assert proof["logits_match"] is True


def test_resume_uses_a_fresh_model_and_optimizer_and_restores_optimizer_state(
    tmp_path
):
    """The load target is a brand-new pair, and the optimizer's own
    state — step counters and both moment buffers — is compared
    structurally and numerically, not inferred from the parameters."""
    x, targets = _inputs()
    loss_fn = build_loss()
    model = build_model()
    optimizer = build_optimizer(model)
    for _ in range(SPLIT_STEP):
        train_step(model, loss_fn, optimizer, x, targets)
    path = str(tmp_path / "classification.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)

    fresh = build_model()
    fresh_optimizer = build_optimizer(fresh)
    assert fresh is not model and fresh_optimizer is not optimizer
    assert all(a is not b for a, b in zip(fresh.parameters(),
                                          model.parameters()))
    # Before the load the fresh optimizer holds no accumulated state.
    assert list(fresh_optimizer.state_dict()["step_counts"]) == [0] * 4
    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)

    saved = optimizer.state_dict()
    restored = fresh_optimizer.state_dict()
    assert restored["format_version"] == saved["format_version"]
    assert restored["optimizer"] == saved["optimizer"] == "NativeAdam"
    assert restored["lr"] == saved["lr"]
    assert tuple(restored["betas"]) == tuple(saved["betas"])
    assert restored["eps"] == saved["eps"]
    assert list(restored["step_counts"]) == list(saved["step_counts"])
    assert list(restored["step_counts"]) == [SPLIT_STEP] * 4
    assert restored["parameters"] == saved["parameters"]   # ordered metadata
    for key in ("m", "v"):
        assert len(restored[key]) == len(saved[key]) == 4
        for restored_tensor, saved_tensor in zip(restored[key], saved[key]):
            assert np.array_equal(restored_tensor.to_numpy(),
                                  saved_tensor.to_numpy())
            assert restored_tensor is not saved_tensor
    for tensors in (restored["m"], restored["v"], saved["m"], saved["v"]):
        for tensor in tensors:
            tensor.close()
    x.close()
    _close(model, optimizer)
    _close(fresh, fresh_optimizer)


def test_checkpoint_format_is_version_one_and_holds_only_persistent_state(
    tmp_path
):
    x, targets = _inputs()
    loss_fn = build_loss()
    model = build_model()
    optimizer = build_optimizer(model)
    train_step(model, loss_fn, optimizer, x, targets)
    path = str(tmp_path / "classification.checkpoint.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata={"steps_completed": 1})

    assert native_checkpoint._FORMAT_VERSION == 3
    with np.load(path, allow_pickle=False) as archive:
        names = list(archive.files)
        manifest = archive["manifest"].tobytes().decode("utf-8")
    blob = (" ".join(names) + " " + manifest).lower()
    # No graph data, no targets, no loss/metric state, no transients.
    for banned in ("target", "label", "probabilit", "logit", "graph", "grad",
                   "winner", "accuracy", "metric", "crossentropy",
                   "cross_entropy", "softmax", "class"):
        assert banned not in blob, banned
    # Exactly the four trainable parameters, plus the optimizer's state.
    assert ('"keys": ["conv.weight", "conv.bias", "linear.weight", '
            '"linear.bias"]') in manifest
    assert '"format": "tensorforge.native_checkpoint"' in manifest
    assert '"format_version": 3' in manifest
    x.close()
    _close(model, optimizer)


def test_the_loss_module_adds_no_checkpoint_keys(tmp_path):
    """Baseline versus a run that also constructs the loss module and the
    metric: the archive is byte-identical, because neither has state."""
    x, targets = _inputs()
    model = build_model()
    optimizer = build_optimizer(model)
    plain = str(tmp_path / "plain.npz")
    save_native_checkpoint(plain, model, optimizer=optimizer)

    loss_fn = build_loss()
    logits = model(x)
    native_accuracy(logits, targets)
    loss = loss_fn(logits, targets)
    loss.close()
    logits.close()
    withloss = str(tmp_path / "withloss.npz")
    save_native_checkpoint(withloss, model, optimizer=optimizer)

    with np.load(plain, allow_pickle=False) as a, \
            np.load(withloss, allow_pickle=False) as b:
        assert list(a.files) == list(b.files)
        for name in a.files:
            assert np.array_equal(a[name], b[name]), name
    assert loss_fn.state_dict() == {}
    x.close()
    _close(model, optimizer)


def test_loading_restores_values_without_restoring_a_graph(tmp_path):
    x, targets = _inputs()
    loss_fn = build_loss()
    model = build_model()
    optimizer = build_optimizer(model)
    train_step(model, loss_fn, optimizer, x, targets)
    path = str(tmp_path / "classification.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)

    fresh = build_model()
    fresh_optimizer = build_optimizer(fresh)
    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    for parameter in fresh.parameters():
        assert parameter.is_leaf is True
        assert parameter.grad is None
        assert parameter._graph_resources == ()
    resumed = train_step(fresh, loss_fn, fresh_optimizer, x, targets)
    assert math.isfinite(resumed)
    x.close()
    _close(model, optimizer)
    _close(fresh, fresh_optimizer)


# --------------------------------------------------------------------------
# Ownership and lifetime
# --------------------------------------------------------------------------

def test_no_completed_graph_or_saved_probability_survives_a_step():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, targets = _inputs()
    # Run the stack layer by layer so pooling's private winner buffer is
    # observable; the model's forward drops it internally.
    conv = model.conv(x)
    relu = model.relu(conv)
    pooled = model.pool(relu)
    winners = pooled._graph_resources
    assert winners, "the pooling node should own a saved winner buffer"
    flat = model.flatten(pooled)
    logits = model.linear(flat)
    loss = loss_fn(logits, targets)
    probabilities = loss._graph_resources
    assert probabilities, "cross-entropy should own its saved probabilities"
    assert all(not core._closed for core in probabilities + winners)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Both private resources went with the released graph history.
    assert all(core._closed for core in probabilities), "saved probabilities"
    assert all(core._closed for core in winners), "saved winners"
    assert loss._graph_resources == () and pooled._graph_resources == ()
    assert loss._graph_freed is True
    with pytest.raises(RuntimeError, match="freed autograd graph"):
        loss.backward()
    for t in (loss, logits, flat, pooled, relu, conv, x):
        t.close()
    _close(model, optimizer)


def test_repeated_steps_return_to_a_stable_storage_baseline(live_storages):
    """The baseline is taken after the model, the optimizer's persistent
    moment state, the persistent input, and the first gradients exist —
    those are intentionally live. What must not grow is the transient
    per-step allocation. gc.collect() makes the count deterministic."""
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, targets = _inputs()
    for _ in range(3):
        train_step(model, loss_fn, optimizer, x, targets)
    gc.collect()
    baseline = len(live_storages)
    assert baseline > 0            # persistent state is honestly live
    for _ in range(6):
        train_step(model, loss_fn, optimizer, x, targets)
        gc.collect()
        assert len(live_storages) == baseline
    x.close()
    _close(model, optimizer)


def test_explicit_cleanup_is_idempotent():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, targets = _inputs()
    train_step(model, loss_fn, optimizer, x, targets)
    logits = model(x)
    loss = loss_fn(logits, targets)
    for _ in range(3):
        loss.close()
        logits.close()
    assert loss.closed and logits.closed
    x.close()
    x.close()
    optimizer.close()
    optimizer.close()
    for parameter in model.parameters():
        parameter.close()
        parameter.close()


def test_checkpoint_round_trip_leaks_no_storage(tmp_path, live_storages):
    gc.collect()
    baseline = len(live_storages)

    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, targets = _inputs()
    train_step(model, loss_fn, optimizer, x, targets)

    gc.collect()
    before_save = len(live_storages)
    path = str(tmp_path / "roundtrip.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)
    gc.collect()
    assert len(live_storages) == before_save   # saving allocates nothing net

    fresh = build_model()
    fresh_optimizer = build_optimizer(fresh)
    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    assert all(p.grad is None for p in fresh.parameters())

    _close(fresh, fresh_optimizer)
    x.close()
    _close(model, optimizer)
    gc.collect()
    assert len(live_storages) == baseline


def test_run_helpers_return_python_values_only():
    run = run_training(steps=2)
    for value in run.values():
        assert not isinstance(value, NativeTensor)
    assert isinstance(run["final_loss"], float)
    assert isinstance(run["final_accuracy"], float)
    assert isinstance(run["loss_history"], list)
    proof = run_resume_proof(total_steps=4, split_step=2)
    for value in proof.values():
        assert not isinstance(value, NativeTensor)


# --------------------------------------------------------------------------
# The NumPy boundary
# --------------------------------------------------------------------------

# Every numerical NumPy routine the native path could fall back to, plus
# every route by which tensor *data* could enter or leave a host buffer.
# np.array/np.asarray stay available: the strict int64 target copy
# legitimately uses them (proved by the E5/E6 suites), and no tensor data
# passes through them here.
_NUMERICAL_NUMPY = (
    "max", "amax", "argmax", "exp", "log", "logaddexp", "sum", "divide",
    "true_divide", "add", "subtract", "multiply", "matmul", "mean",
    "negative", "power", "copyto", "sqrt", "reciprocal", "take",
    "take_along_axis", "put", "put_along_axis", "where", "choose", "maximum",
)
_DATA_NUMPY = ("empty", "frombuffer")


def _arm_tripwire(monkeypatch):
    def _tripwire(*args, **kwargs):
        raise AssertionError("the native training step reached NumPy")

    for name in _NUMERICAL_NUMPY + _DATA_NUMPY:
        monkeypatch.setattr(np, name, _tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", _tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "from_array",
                        staticmethod(_tripwire))
    monkeypatch.setattr(cpp.NativeTensorView, "to_numpy", _tripwire)
    monkeypatch.setattr(cpp.NativeStorage, "from_array", staticmethod(_tripwire))
    monkeypatch.setattr(cpp.NativeStorage, "to_numpy", _tripwire)
    monkeypatch.setattr(NativeTensor, "to_numpy", _tripwire)


def test_one_training_step_is_fully_native(monkeypatch):
    """Forward, loss, backward, and the optimizer update run to
    completion with every NumPy numerical routine and every tensor-data
    conversion route armed. `native_accuracy` is deliberately **outside**
    the armed region — it converts on purpose (§8) — and the reference
    comparison runs after the tripwire is removed."""
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, targets = _inputs()
    # Warm up so Adam's persistent moment buffers already exist: their
    # one-time allocation is not part of a steady-state training step.
    train_step(model, loss_fn, optimizer, x, targets)
    reference = build_model()
    reference_optimizer = build_optimizer(reference)
    train_step(reference, loss_fn, reference_optimizer, x, targets)
    train_step(reference, loss_fn, reference_optimizer, x, targets)

    _arm_tripwire(monkeypatch)
    logits = model(x)                 # forward
    loss = loss_fn(logits, targets)   # fused cross-entropy over raw logits
    loss.backward()                   # backward
    optimizer.step()                  # parameter update
    optimizer.zero_grad()
    # The tripwire really can fire: a host conversion of tensor data, or
    # a NumPy numerical routine, trips it right here.
    with pytest.raises(AssertionError, match="reached NumPy"):
        logits.to_numpy()
    with pytest.raises(AssertionError, match="reached NumPy"):
        np.exp(1.0)
    loss.close()
    logits.close()
    monkeypatch.undo()

    # The step really happened, and it produced exactly what the
    # unarmed reference run produced.
    for (name, a), (_, b) in zip(model.named_parameters(),
                                 reference.named_parameters()):
        assert np.array_equal(a.to_numpy(), b.to_numpy()), name
    x.close()
    _close(model, optimizer)
    _close(reference, reference_optimizer)


def test_the_reporting_metric_is_the_one_deliberate_conversion(monkeypatch):
    """`native_accuracy` is reporting-only and converts on purpose, so it
    must never be inside the tripwire — this test proves the conversion
    is real rather than assuming it."""
    model = build_model()
    x, targets = _inputs()
    logits = model(x)
    calls = []
    original = NativeTensor.to_numpy
    monkeypatch.setattr(
        NativeTensor, "to_numpy",
        lambda self, *a, **k: (calls.append(id(self)), original(self, *a, **k))[1],
    )
    accuracy = native_accuracy(logits, targets)
    monkeypatch.undo()
    assert calls == [id(logits)]       # exactly one deliberate copy out
    assert isinstance(accuracy, float) and 0.0 <= accuracy <= 1.0
    # It built no graph and touched no gradient.
    assert logits.grad is None
    assert not logits.closed
    logits.close()
    x.close()
    _close(model)


# --------------------------------------------------------------------------
# Failure paths (E8 adds no new failure semantics; it exercises them)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    [0.0] * 12,                     # floating-point labels
    [True] * 12,                    # bool labels
    [0] * 11,                       # wrong length
    [0] * 11 + [NUM_CLASSES],       # out of range
])
def test_invalid_targets_are_rejected_by_the_loss(bad):
    model = build_model()
    loss_fn = build_loss()
    x, _ = _inputs()
    logits = model(x)
    with pytest.raises((TypeError, ValueError, IndexError)):
        loss_fn(logits, bad)
    # The rejection changed nothing: the logits are still usable.
    assert not logits.closed
    good = loss_fn(logits, TARGET_VALUES)
    assert math.isfinite(float(good.to_numpy()))
    for t in (good, logits, x):
        t.close()
    _close(model)


def test_a_closed_input_is_rejected_before_any_work():
    model = build_model()
    loss_fn = build_loss()
    x, targets = _inputs()
    x.close()
    with pytest.raises(RuntimeError, match="closed"):
        model(x)
    logits_source, _ = _inputs()
    logits = model(logits_source)
    logits.close()
    with pytest.raises(RuntimeError, match="closed"):
        loss_fn(logits, targets)
    with pytest.raises(RuntimeError, match="closed"):
        native_accuracy(logits, targets)
    logits_source.close()
    _close(model)


def test_checkpoint_load_failures_use_the_existing_behavior(tmp_path):
    model = build_model()
    optimizer = build_optimizer(model)
    missing = str(tmp_path / "does_not_exist.npz")
    with pytest.raises((FileNotFoundError, OSError)):
        load_native_checkpoint(missing, model, optimizer=optimizer)
    # Nothing was mutated by the failed load.
    assert all(p.grad is None for p in model.parameters())
    _close(model, optimizer)


def test_loading_into_an_incompatible_model_fails_atomically(tmp_path):
    x, targets = _inputs()
    loss_fn = build_loss()
    model = build_model()
    optimizer = build_optimizer(model)
    train_step(model, loss_fn, optimizer, x, targets)
    path = str(tmp_path / "classification.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)

    other = NativeImageClassifier()
    other.linear = NativeLinear(FLAT_FEATURES, NUM_CLASSES + 1, seed=1)
    other_optimizer = build_optimizer(other)
    before = {n: p.to_numpy().copy() for n, p in other.named_parameters()}
    with pytest.raises(Exception):
        load_native_checkpoint(path, other, optimizer=other_optimizer)
    for name, parameter in other.named_parameters():
        assert np.array_equal(parameter.to_numpy(), before[name]), name

    # A mismatched optimizer type is rejected by the same strict path.
    fresh = build_model()
    from tensorforge.experimental import NativeSGD
    wrong_optimizer = NativeSGD(fresh.parameters(), lr=DEFAULT_LR)
    with pytest.raises(Exception):
        load_native_checkpoint(path, fresh, optimizer=wrong_optimizer)

    # And the original checkpoint still loads correctly afterwards.
    good = build_model()
    good_optimizer = build_optimizer(good)
    load_native_checkpoint(path, good, optimizer=good_optimizer)
    assert np.array_equal(good.parameters()[0].to_numpy(),
                          model.parameters()[0].to_numpy())
    x.close()
    _close(model, optimizer)
    _close(other, other_optimizer)
    _close(fresh)
    _close(good, good_optimizer)


def test_reusing_a_graph_after_an_optimizer_step_raises():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, targets = _inputs()
    logits = model(x)
    loss = loss_fn(logits, targets)
    loss.backward(retain_graph=True)
    optimizer.step()          # mutates parameter values, bumping versions
    with pytest.raises(RuntimeError, match="stale parameter value"):
        loss.backward(retain_graph=True)
    optimizer.zero_grad()
    loss.close()
    logits.close()
    value = train_step(model, loss_fn, optimizer, x, targets)
    assert math.isfinite(value)
    x.close()
    _close(model, optimizer)


@needs_fault_injection
def test_allocation_failure_mid_step_commits_no_partial_update(live_storages):
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, targets = _inputs()
    train_step(model, loss_fn, optimizer, x, targets)   # settle Adam's state
    before = {n: p.to_numpy().copy() for n, p in model.named_parameters()}
    versions = [p.version for p in model.parameters()]
    baseline = len(live_storages)

    cpp._arm_alloc_failure(1)
    with pytest.raises(MemoryError):
        train_step(model, loss_fn, optimizer, x, targets)
    cpp._arm_alloc_failure(0)

    for name, parameter in model.named_parameters():
        assert np.array_equal(parameter.to_numpy(), before[name]), name
    assert [p.version for p in model.parameters()] == versions
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    optimizer.zero_grad()
    assert len(live_storages) <= baseline

    value = train_step(model, loss_fn, optimizer, x, targets)
    assert math.isfinite(value)
    x.close()
    _close(model, optimizer)


# --------------------------------------------------------------------------
# The runnable example
# --------------------------------------------------------------------------

def test_example_file_exists_and_is_import_safe(capsys):
    import importlib

    import examples.native_classification_training as module

    capsys.readouterr()
    importlib.reload(module)
    # Importing runs no training and prints nothing.
    assert capsys.readouterr().out == ""
    assert EXAMPLE.is_file()
    text = EXAMPLE.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in text
    # No network access, no repository mutation, no git instructions.
    for banned in ("requests", "urllib", "socket", "git ", "subprocess"):
        assert banned not in text, banned


def test_example_main_runs_and_reports_learning_and_exact_resume(capsys):
    main()
    output = capsys.readouterr().out
    for expected in ("initial loss:", "initial accuracy:", "final loss:",
                     "final accuracy:", "exact resume: yes",
                     "NativeCrossEntropyLoss", "raw logits",
                     "native classification training + checkpoint resume ok"):
        assert expected in output, expected
    # The printed values agree with the locked configuration.
    assert f"initial loss: {OBSERVED_INITIAL_LOSS:.6f}" in output
    assert f"final loss: {OBSERVED_FINAL_LOSS:.6f}" in output
    assert f"initial accuracy: {OBSERVED_INITIAL_ACCURACY:.4f}" in output
    assert f"final accuracy: {OBSERVED_FINAL_ACCURACY:.4f}" in output
    assert f"trained {TOTAL_STEPS} NativeAdam steps (lr={DEFAULT_LR})" in output
    assert f"trained {SPLIT_STEP} steps" in output
    assert "format version 1" in output
    # No speed or production claim, and compact output.
    lowered = output.lower()
    for banned in ("faster", "fastest", "speedup", "production", "benchmark"):
        assert banned not in lowered, banned
    assert len(output.splitlines()) < 40


def test_example_leaves_no_checkpoint_files_behind():
    before = {p.name for p in REPO_ROOT.iterdir()}
    run_resume_proof(total_steps=4, split_step=2)
    after = {p.name for p in REPO_ROOT.iterdir()}
    assert before == after
    assert not list((REPO_ROOT / "examples").glob("*.npz"))
    assert not list(REPO_ROOT.glob("*.npz"))


def test_evaluate_reports_without_updating_anything():
    model = build_model()
    loss_fn = build_loss()
    x, targets = _inputs()
    versions = [p.version for p in model.parameters()]
    loss, accuracy, predictions, logits = evaluate(model, loss_fn, x, targets)
    assert math.isfinite(loss) and 0.0 <= accuracy <= 1.0
    assert len(predictions) == len(targets) == len(logits)
    assert all(0 <= p < NUM_CLASSES for p in predictions)
    assert [p.version for p in model.parameters()] == versions
    assert all(p.grad is None for p in model.parameters())
    x.close()
    _close(model)


# --------------------------------------------------------------------------
# Scope boundaries: E8 adds no capability
# --------------------------------------------------------------------------

def test_e8_adds_no_capability_inventory_entry():
    assert cpp.RAW_KERNELS == (
        "elementwise_add", "elementwise_subtract", "elementwise_multiply",
        "elementwise_divide", "relu", "matmul", "matmul_tiled",
    )
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm",     # Phase F, milestone F2 (unrelated to E8)
        "NativeBatchNorm1d",   # Phase F, milestone F3 (unrelated to E8)
        "NativeBatchNorm2d",   # Phase F, milestone F4 (unrelated to E8)
        "NativeDropout",       # Phase G, milestone G4 (unrelated to E8)
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    # "cross_entropy" was the last autograd op when E8 landed and E8 added
    # nothing after it. The one entry that follows is Phase G milestone
    # G3's differentiable "dropout", which is unrelated to this proof.
    assert cpp.AUTOGRAD_OPS[-2] == "cross_entropy"
    assert cpp.AUTOGRAD_OPS[-1] == "dropout"
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    # The proof is an integration result, never a named capability.
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS):
        for banned in ("train", "classifier", "checkpoint_resume", "example",
                       "dataset"):
            assert not [n for n in inventory if banned in n.lower()], banned
    # No accuracy/argmax kernel or export appeared.
    for absent in ("tf_core_accuracy", "tf_core_argmax", "tf_core_train_step"):
        assert absent not in cpp._CHECKED_KERNELS, absent


def test_e8_changes_no_checkpoint_schema_and_no_stable_framework():
    assert native_checkpoint._FORMAT_VERSION == 3
    assert cpp.backend_info()["stable_framework_integration"] is False
    # The example never touches the stable framework.
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "tensorforge.nn" not in text and "tensorforge.optim" not in text
    assert "from tensorforge import" not in text
    # And no native name leaked into the stable namespace.
    for name in ("NativeCrossEntropyLoss", "native_accuracy",
                 "NativeImageClassifier"):
        assert not hasattr(tensorforge, name), name


def test_the_e8_proof_stays_separate_from_the_e9_benchmark():
    """The split is deliberate and survives phase closure: this example
    owns deterministic correctness and exact resume, the E9 benchmark
    owns measurement, and the E10 integration test owns stack-level
    guarantees. The example itself times nothing."""
    assert (REPO_ROOT / "benchmarks"
            / "benchmark_native_classification.py").is_file()
    assert (REPO_ROOT / "tests" / "test_native_phase_e.py").is_file()
    # Structural, not word-level: the example's prose may (and does)
    # explain that measurement is E9's job, but it must measure nothing —
    # it imports no timer and calls none.
    text = EXAMPLE.read_text(encoding="utf-8")
    for banned in ("perf_counter", "time.time(", "import time",
                   "import timeit", "median_s"):
        assert banned not in text, banned
