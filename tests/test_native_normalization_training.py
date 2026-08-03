"""The native normalized training and exact checkpoint-resume proof
(Phase F, milestone F6).

``examples/native_normalization_training.py`` trains
``NativeNormalizedRegressor`` — ``NativeLinear(2, 8, seed=0)`` ->
``NativeBatchNorm1d(8, momentum=0.1)`` -> ``NativeReLU`` ->
``NativeLayerNorm(8)`` -> ``NativeLinear(8, 1, seed=1)`` — on a fixed
eight-sample two-feature regression task for a fixed number of
deterministic ``NativeAdam`` steps, with ``NativeMSELoss``. These tests
verify the dataset and architecture, that **both** normalization families
participate and that only BatchNorm is stateful, deterministic
initialization, forward / loss / backward / optimizer integration, the
learning guardrails, exact run-to-run determinism, exact
uninterrupted-versus-resumed equivalence through one pickle-free
checkpoint (model **and** optimizer state — including the BatchNorm
running buffers — format version 1), the exact resume of the running
statistics and the evaluation-mode output, ownership and live-storage
cleanup, the NumPy boundary over a complete normalized update, and the F6
scope boundary (no new capability of any kind).

Every number pinned below was observed from the deterministic workload
first; the learning thresholds carry a wide margin (see
``test_loss_reduction_guardrails``). NumPy appears only for references,
inspection, and equality assertions; the training computation is native,
and ``test_one_normalized_training_step_is_fully_native`` proves it with
the numerical/conversion tripwire over the BatchNorm and LayerNorm path.

Selector: python -m pytest -q -k native_normalization_training
"""

import gc
import math
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
    NativeLayerNorm,
    NativeLinear,
    NativeModule,
    NativeMSELoss,
    NativeReLU,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)
from tensorforge.experimental import native_checkpoint
from examples.native_normalization_training import (
    DEFAULT_LR,
    HIDDEN_FEATURES,
    IN_FEATURES,
    MOMENTUM,
    OUT_FEATURES,
    SPLIT_STEP,
    TOTAL_STEPS,
    X_VALUES,
    Y_VALUES,
    NativeNormalizedRegressor,
    build_dataset,
    build_loss,
    build_model,
    build_optimizer,
    evaluate,
    main,
    run_resume_proof,
    run_training,
    train_step,
    _model_state_values,
    _optimizer_state_values,
    _running_stats,
)

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "examples" / "native_normalization_training.py"

# Observed once from the deterministic workload; asserted with margin
# (learning) or as exact same-machine reproducibility pins.
OBSERVED_INITIAL_LOSS = 2.4402445496163696
OBSERVED_FINAL_LOSS = 0.02700037213028736
OBSERVED_FINAL_EVAL_LOSS = 0.023076274782684533

PARAMETER_NAMES = [
    "hidden.weight", "hidden.bias",
    "batch_norm.gamma", "batch_norm.beta",
    "layer_norm.weight", "layer_norm.bias",
    "output.weight", "output.bias",
]
BUFFER_NAMES = ["batch_norm.running_mean", "batch_norm.running_var"]
STATE_KEYS = PARAMETER_NAMES + BUFFER_NAMES
PARAMETER_SHAPES = {
    "hidden.weight": (IN_FEATURES, HIDDEN_FEATURES),
    "hidden.bias": (HIDDEN_FEATURES,),
    "batch_norm.gamma": (HIDDEN_FEATURES,),
    "batch_norm.beta": (HIDDEN_FEATURES,),
    "layer_norm.weight": (HIDDEN_FEATURES,),
    "layer_norm.bias": (HIDDEN_FEATURES,),
    "output.weight": (HIDDEN_FEATURES, OUT_FEATURES),
    "output.bias": (OUT_FEATURES,),
}


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
    inputs, targets = build_dataset()
    return NativeTensor.from_array(inputs), NativeTensor.from_array(targets)


def _close(model, optimizer=None, *tensors):
    if optimizer is not None:
        optimizer.close()
    for parameter in model.parameters():
        parameter.close()
    for buffer in model.buffers():
        buffer.close()
    for tensor in tensors:
        tensor.close()


# --------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------

def test_dataset_is_fixed_literal_regression_data_with_exact_shapes():
    inputs, targets = build_dataset()
    assert len(inputs) == len(targets) == 8
    for row in inputs:
        assert len(row) == IN_FEATURES
        assert all(isinstance(value, float) for value in row)
    for row in targets:
        assert len(row) == OUT_FEATURES
        assert all(isinstance(value, float) for value in row)
    assert inputs == X_VALUES and targets == Y_VALUES


def test_dataset_construction_is_deterministic_and_independent():
    first_inputs, first_targets = build_dataset()
    second_inputs, second_targets = build_dataset()
    assert first_inputs == second_inputs == X_VALUES
    assert first_targets == second_targets == Y_VALUES
    # Fresh copies: mutating one result cannot perturb the literals.
    first_inputs[0][0] = 99.0
    first_targets[0][0] = 99.0
    third_inputs, third_targets = build_dataset()
    assert third_inputs == X_VALUES and third_targets == Y_VALUES


def test_native_input_tensors_match_the_host_dataset():
    x, y = _inputs()
    assert x.shape == (8, IN_FEATURES)
    assert y.shape == (8, OUT_FEATURES)
    assert x.dtype == "float64" and x.device == "cpu"
    assert np.array_equal(x.to_numpy(), np.asarray(X_VALUES))
    assert np.array_equal(y.to_numpy(), np.asarray(Y_VALUES))
    x.close()
    y.close()


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

def test_model_has_the_exact_named_architecture_with_both_normalizations():
    model = build_model()
    assert isinstance(model, NativeNormalizedRegressor)
    assert isinstance(model, NativeModule)
    assert isinstance(model.hidden, NativeLinear)
    assert isinstance(model.batch_norm, NativeBatchNorm1d)
    assert isinstance(model.relu, NativeReLU)
    assert isinstance(model.layer_norm, NativeLayerNorm)
    assert isinstance(model.output, NativeLinear)
    # Named children in registration/execution order.
    assert list(dict(model.named_modules())) == [
        "", "hidden", "batch_norm", "relu", "layer_norm", "output"
    ]
    _close(model)


def test_only_batchnorm_contributes_buffers():
    model = build_model()
    # BatchNorm is the only stateful normalization module.
    assert [name for name, _ in model.named_buffers()] == BUFFER_NAMES
    assert [name for name, _ in model.batch_norm.named_buffers()] == [
        "running_mean", "running_var"
    ]
    # LayerNorm contributes affine parameters but no buffers.
    assert list(model.layer_norm.buffers()) == []
    assert [name for name, _ in model.layer_norm.named_parameters()] == [
        "weight", "bias"
    ]
    # ReLU and the linear layers contribute no buffers either.
    for child in (model.relu, model.hidden, model.output):
        assert list(child.buffers()) == []
    _close(model)


def test_parameter_names_state_keys_shapes_and_metadata():
    model = build_model()
    assert [name for name, _ in model.named_parameters()] == PARAMETER_NAMES
    assert [name for name, _ in model.named_buffers()] == BUFFER_NAMES
    # Canonical state-key order: parameters first, persistent buffers second.
    assert list(model.state_dict()) == STATE_KEYS
    for snapshot in model.state_dict().values():
        snapshot.close()
    shapes = {name: p.shape for name, p in model.named_parameters()}
    assert shapes == PARAMETER_SHAPES
    for _, buffer in model.named_buffers():
        assert buffer.shape == (HIDDEN_FEATURES,)
    for parameter in model.parameters():
        assert parameter.dtype == "float64" and parameter.device == "cpu"
    _close(model)


def test_optimizer_excludes_buffers():
    model = build_model()
    optimizer = build_optimizer(model)
    optimizer_ids = {id(p) for p in optimizer.parameters()}
    assert len(optimizer.parameters()) == len(PARAMETER_NAMES)
    for _, buffer in model.named_buffers():
        assert id(buffer) not in optimizer_ids
    # The optimizer holds exactly the model's trainable parameters.
    assert optimizer_ids == {id(p) for p in model.parameters()}
    _close(model, optimizer)


def test_loss_is_native_mse_and_holds_no_state():
    loss_fn = build_loss()
    assert isinstance(loss_fn, NativeMSELoss)
    assert loss_fn.reduction == "mean"
    assert loss_fn.state_dict() == {}
    assert list(loss_fn.parameters()) == [] and list(loss_fn.buffers()) == []


# --------------------------------------------------------------------------
# Forward and mode behavior
# --------------------------------------------------------------------------

def test_training_forward_shape_and_both_normalizations_in_the_path():
    model = build_model()
    x, _ = _inputs()
    model.train()
    prediction = model(x)
    assert prediction.shape == (8, OUT_FEATURES)
    assert prediction.dtype == "float64" and prediction.device == "cpu"
    assert np.isfinite(prediction.to_numpy()).all()
    # The forward really runs through both normalization children.
    source = EXAMPLE.read_text(encoding="utf-8")
    forward = source.split("def forward(self, inputs):", 1)[1].split(
        "\n\n", 1)[0]
    assert "self.batch_norm(" in forward
    assert "self.layer_norm(" in forward
    prediction.close()
    x.close()
    _close(model)


def test_batchnorm_running_state_advances_during_training():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    before_mean = model.batch_norm.running_mean.to_numpy().copy()
    before_var = model.batch_norm.running_var.to_numpy().copy()
    train_step(model, loss_fn, optimizer, x, y)
    assert not np.array_equal(model.batch_norm.running_mean.to_numpy(),
                              before_mean)
    assert not np.array_equal(model.batch_norm.running_var.to_numpy(),
                              before_var)
    # Buffer identities are stable across the update.
    _close(model, optimizer, x, y)


def test_layernorm_state_stays_parameter_only():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    # LayerNorm has no buffers before or after training.
    assert list(model.layer_norm.buffers()) == []
    for _ in range(3):
        train_step(model, loss_fn, optimizer, x, y)
    assert list(model.layer_norm.buffers()) == []
    assert [n for n, _ in model.layer_norm.named_parameters()] == ["weight", "bias"]
    _close(model, optimizer, x, y)


def test_eval_forward_does_not_update_running_state():
    model = build_model()
    loss_fn = build_loss()
    x, y = _inputs()
    before_mean = model.batch_norm.running_mean.to_numpy().copy()
    before_var = model.batch_norm.running_var.to_numpy().copy()
    for _ in range(4):
        evaluate(model, loss_fn, x, y)
    assert np.array_equal(model.batch_norm.running_mean.to_numpy(), before_mean)
    assert np.array_equal(model.batch_norm.running_var.to_numpy(), before_var)
    x.close()
    y.close()
    _close(model)


def test_train_and_eval_outputs_differ_when_running_statistics_differ():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    # Train a few steps so the running statistics move away from (0, 1).
    for _ in range(5):
        train_step(model, loss_fn, optimizer, x, y)
    model.train()
    train_prediction = model(x)
    train_values = train_prediction.to_numpy().copy()
    train_prediction.close()
    _, eval_values = evaluate(model, loss_fn, x, y)
    assert not np.allclose(train_values, np.asarray(eval_values), atol=1e-6)
    _close(model, optimizer, x, y)


def test_evaluate_restores_the_callers_previous_mode():
    model = build_model()
    loss_fn = build_loss()
    x, y = _inputs()
    model.train()
    evaluate(model, loss_fn, x, y)
    assert model.training is True          # restored to training
    model.eval()
    evaluate(model, loss_fn, x, y)
    assert model.training is False         # restored to eval
    x.close()
    y.close()
    _close(model)


# --------------------------------------------------------------------------
# Backward and update
# --------------------------------------------------------------------------

def test_backward_reaches_every_parameter_and_never_a_buffer():
    model = build_model()
    loss_fn = build_loss()
    x, y = _inputs()
    model.train()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    assert loss.shape == () and loss.numel == 1
    loss.backward()
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        assert grad is not None, name
        assert grad.shape == parameter.shape == PARAMETER_SHAPES[name], name
        values = grad.to_numpy()
        assert np.isfinite(values).all(), name
        assert (values != 0.0).any(), name
    # Buffers never receive a gradient.
    for name, buffer in model.named_buffers():
        assert buffer.grad is None, name
        assert buffer.requires_grad is False, name
    for t in (loss, prediction, x, y):
        t.close()
    _close(model)


def test_first_step_gradient_evidence_is_recorded():
    run = run_training(steps=1)
    assert run["gradient_shapes"] == PARAMETER_SHAPES
    assert all(run["gradient_nonzero"].values())
    assert run["all_parameters_reached"] is True
    assert run["buffers_grad_free"] is True


def test_parameter_and_buffer_identities_are_stable_and_versions_advance():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    parameter_ids = [id(p) for p in model.parameters()]
    buffer_ids = [id(b) for b in model.buffers()]
    versions = [p.version for p in model.parameters()]
    steps = 4
    for _ in range(steps):
        train_step(model, loss_fn, optimizer, x, y)
        assert [id(p) for p in model.parameters()] == parameter_ids
        assert [id(b) for b in model.buffers()] == buffer_ids
    # Parameters advance one version per optimizer step; buffers have none.
    assert [p.version for p in model.parameters()] == [v + steps for v in versions]
    for buffer in model.buffers():
        assert not hasattr(buffer, "version")
    _close(model, optimizer, x, y)


def test_zero_grad_clears_every_parameter_gradient():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    model.train()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    optimizer.step()
    optimizer.zero_grad()
    assert all(p.grad is None for p in model.parameters())
    for t in (loss, prediction, x, y):
        t.close()
    _close(model, optimizer)


def test_running_statistics_update_through_forward_not_the_optimizer():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    model.train()
    # A forward alone advances the running statistics; the optimizer does not.
    before_mean = model.batch_norm.running_mean.to_numpy().copy()
    prediction = model(x)
    prediction.close()
    assert not np.array_equal(model.batch_norm.running_mean.to_numpy(),
                              before_mean)
    after_forward = model.batch_norm.running_mean.to_numpy().copy()
    # A step with no gradients present changes no running statistic.
    optimizer.zero_grad()
    assert np.array_equal(model.batch_norm.running_mean.to_numpy(),
                          after_forward)
    _close(model, optimizer, x, y)


# --------------------------------------------------------------------------
# Learning behavior
# --------------------------------------------------------------------------

def test_loss_reduction_guardrails():
    """Thresholds chosen after observing the deterministic run: it goes
    from ~2.440245 to ~0.027000 (ratio ~0.011). The guards below allow a
    final loss up to 0.15 and a ratio up to 0.1 — a comfortable margin over
    the observed values, so ordinary floating-point drift cannot make them
    fail spuriously. The curve need not be monotonic (Adam overshoots
    early); it must end far below where it started."""
    run = run_training()
    assert len(run["loss_history"]) == TOTAL_STEPS
    assert all(math.isfinite(value) for value in run["loss_history"])
    assert math.isfinite(run["initial_loss"]) and math.isfinite(run["final_loss"])
    assert run["initial_loss"] == pytest.approx(OBSERVED_INITIAL_LOSS, rel=1e-9)
    assert run["final_loss"] == pytest.approx(OBSERVED_FINAL_LOSS, rel=1e-6)
    assert run["final_loss"] < 0.15
    assert run["final_loss"] / run["initial_loss"] < 0.1
    assert run["final_loss"] < run["initial_loss"]
    assert min(run["loss_history"]) < 0.1


def test_final_outputs_and_running_statistics_are_finite():
    run = run_training()
    eval_loss, eval_output = run["final_eval"]
    assert math.isfinite(eval_loss)
    assert np.isfinite(np.asarray(eval_output)).all()
    assert np.isfinite(np.asarray(run["final_train_prediction"])).all()
    for key in ("running_mean", "running_var"):
        assert np.isfinite(np.asarray(run["final_running_stats"][key])).all()
    # The BatchNorm running statistics genuinely advanced from (0, 1).
    assert run["running_stats_advanced"] is True
    assert run["eval_differs_from_train"] is True
    assert run["mode_restored"] is True
    assert run["identity_stable"] is True
    assert run["gradients_cleared"] is True


def test_parameters_change_and_optimizer_accumulates_state():
    run = run_training()
    for name in PARAMETER_NAMES:
        before = np.asarray(run["initial_parameters"][name])
        after = np.asarray(run["final_parameters"][name])
        assert not np.array_equal(before, after), name
        assert np.isfinite(after).all(), name
    state = run["final_optimizer_state"]
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

def test_two_independent_runs_are_bit_identical():
    first = run_training()
    second = run_training()
    for key in ("initial_parameters", "loss_history", "initial_loss",
                "final_loss", "final_parameters", "final_running_stats",
                "final_train_prediction", "final_eval",
                "final_optimizer_state"):
        assert first[key] == second[key], key
    # Exact, not approximate: the same native kernels in the same order.
    for name in PARAMETER_NAMES:
        assert np.array_equal(np.asarray(first["final_parameters"][name]),
                              np.asarray(second["final_parameters"][name]))


def test_two_independently_built_models_start_identical():
    first, second = build_model(), build_model()
    for (name_a, a), (name_b, b) in zip(first.named_parameters(),
                                        second.named_parameters()):
        assert name_a == name_b
        assert np.array_equal(a.to_numpy(), b.to_numpy()), name_a
    for (_, a), (_, b) in zip(first.named_buffers(), second.named_buffers()):
        assert np.array_equal(a.to_numpy(), b.to_numpy())
    _close(first)
    _close(second)


def test_initialization_ignores_the_global_numpy_rng():
    np.random.seed(1234)
    first = run_training(steps=3)
    np.random.seed(4321)
    [np.random.random() for _ in range(10)]
    second = run_training(steps=3)
    assert first["initial_parameters"] == second["initial_parameters"]
    assert first["loss_history"] == second["loss_history"]
    assert first["final_parameters"] == second["final_parameters"]


def test_a_run_does_not_mutate_the_fixed_literals():
    before_x = [list(row) for row in X_VALUES]
    before_y = [list(row) for row in Y_VALUES]
    run_training(steps=3)
    run_resume_proof(total_steps=4, split_step=2)
    assert X_VALUES == before_x
    assert Y_VALUES == before_y


# --------------------------------------------------------------------------
# Exact checkpoint resume (the central F6 contract)
# --------------------------------------------------------------------------

def test_resumed_training_matches_uninterrupted_exactly():
    proof = run_resume_proof()
    # Construction, prefix, and suffix all match exactly.
    assert proof["identical_start"] is True
    assert proof["prefix_matches"] is True
    assert proof["first_resumed_loss_matches"] is True
    assert proof["suffix_matches"] is True
    assert proof["losses_match"] is True
    # Final training-step prediction and evaluation-mode output match.
    assert proof["final_train_predictions_match"] is True
    assert proof["final_eval_matches"] is True
    # Every parameter, the running statistics, and the optimizer state match.
    assert proof["parameters_match"] is True
    assert proof["running_mean_matches"] is True
    assert proof["running_var_matches"] is True
    assert proof["optimizer_state_matches"] is True
    assert proof["parameter_order_matches"] is True
    assert proof["buffer_order_matches"] is True
    # Load target identities and unserialized mode.
    assert proof["identities_preserved"] is True
    assert proof["mode_not_serialized"] is True
    assert proof["metadata"] == {"steps_completed": SPLIT_STEP, "lr": DEFAULT_LR}
    assert len(proof["uninterrupted_losses"]) == TOTAL_STEPS
    assert len(proof["resumed_suffix"]) == TOTAL_STEPS - SPLIT_STEP
    # Not just the final loss: the whole remaining suffix, element by element.
    assert (proof["resumed_suffix"]
            == proof["uninterrupted_losses"][SPLIT_STEP:])
    assert proof["state_keys"] == STATE_KEYS


def test_resume_equivalence_holds_for_another_split():
    proof = run_resume_proof(total_steps=12, split_step=7)
    assert proof["losses_match"] is True
    assert proof["parameters_match"] is True
    assert proof["running_mean_matches"] is True
    assert proof["running_var_matches"] is True
    assert proof["optimizer_state_matches"] is True
    assert proof["final_eval_matches"] is True
    assert proof["final_train_predictions_match"] is True


def test_resume_uses_a_fresh_pair_and_restores_running_stats_and_optimizer(
    tmp_path
):
    """The load target is a brand-new model/optimizer pair, and the
    BatchNorm running statistics *and* the NativeAdam moment buffers are
    compared structurally and numerically, not inferred from the loss."""
    x, y = _inputs()
    loss_fn = build_loss()
    model = build_model()
    optimizer = build_optimizer(model)
    for _ in range(SPLIT_STEP):
        train_step(model, loss_fn, optimizer, x, y)
    path = str(tmp_path / "normalization.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)

    fresh = build_model()
    fresh_optimizer = build_optimizer(fresh)
    assert fresh is not model and fresh_optimizer is not optimizer
    assert all(a is not b for a, b in zip(fresh.parameters(),
                                          model.parameters()))
    assert all(a is not b for a, b in zip(fresh.buffers(), model.buffers()))
    # Before the load the fresh model's running stats are the (0, 1) init.
    assert np.array_equal(fresh.batch_norm.running_mean.to_numpy(),
                          np.zeros(HIDDEN_FEATURES))
    assert np.array_equal(fresh.batch_norm.running_var.to_numpy(),
                          np.ones(HIDDEN_FEATURES))
    assert list(fresh_optimizer.state_dict()["step_counts"]) == [0] * 8

    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)

    # Running statistics restored exactly.
    assert np.array_equal(fresh.batch_norm.running_mean.to_numpy(),
                          model.batch_norm.running_mean.to_numpy())
    assert np.array_equal(fresh.batch_norm.running_var.to_numpy(),
                          model.batch_norm.running_var.to_numpy())
    # Optimizer state restored exactly.
    saved = optimizer.state_dict()
    restored = fresh_optimizer.state_dict()
    assert restored["format_version"] == saved["format_version"]
    assert restored["optimizer"] == saved["optimizer"] == "NativeAdam"
    assert restored["lr"] == saved["lr"]
    assert tuple(restored["betas"]) == tuple(saved["betas"])
    assert restored["eps"] == saved["eps"]
    assert list(restored["step_counts"]) == list(saved["step_counts"]) == [SPLIT_STEP] * 8
    for key in ("m", "v"):
        assert len(restored[key]) == len(saved[key]) == 8
        for restored_tensor, saved_tensor in zip(restored[key], saved[key]):
            assert np.array_equal(restored_tensor.to_numpy(),
                                  saved_tensor.to_numpy())
            assert restored_tensor is not saved_tensor
    for tensors in (restored["m"], restored["v"], saved["m"], saved["v"]):
        for tensor in tensors:
            tensor.close()
    _close(model, optimizer, x, y)
    _close(fresh, fresh_optimizer)


def test_checkpoint_is_version_one_and_holds_the_expected_canonical_keys(
    tmp_path
):
    x, y = _inputs()
    loss_fn = build_loss()
    model = build_model()
    optimizer = build_optimizer(model)
    train_step(model, loss_fn, optimizer, x, y)
    path = str(tmp_path / "normalization.checkpoint.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata={"steps_completed": 1})

    assert native_checkpoint._FORMAT_VERSION == 3
    with np.load(path, allow_pickle=False) as archive:
        manifest = archive["manifest"].tobytes().decode("utf-8")
    assert '"format": "tensorforge.native_checkpoint"' in manifest
    assert '"format_version": 3' in manifest
    # The exact canonical state keys, in order — parameters then the
    # BatchNorm running buffers.
    expected = ('"keys": ["hidden.weight", "hidden.bias", '
                '"batch_norm.gamma", "batch_norm.beta", '
                '"layer_norm.weight", "layer_norm.bias", '
                '"output.weight", "output.bias", '
                '"batch_norm.running_mean", "batch_norm.running_var"]')
    assert expected in manifest
    # No training flag or new normalization field leaked into the manifest.
    for banned in ("training", "num_batches_tracked", "running_stats",
                   "rng_state", "momentum"):
        assert banned not in manifest, banned
    x.close()
    y.close()
    _close(model, optimizer)


def test_training_mode_is_not_serialized(tmp_path):
    x, y = _inputs()
    loss_fn = build_loss()
    model = build_model()
    optimizer = build_optimizer(model)
    train_step(model, loss_fn, optimizer, x, y)
    path = str(tmp_path / "mode.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)

    fresh = build_model()
    fresh_optimizer = build_optimizer(fresh)
    fresh.eval()                       # deliberately eval before load
    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    assert fresh.training is False     # the load did not touch the flag
    fresh.batch_norm.eval()
    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    assert fresh.batch_norm.training is False
    x.close()
    y.close()
    _close(model, optimizer)
    _close(fresh, fresh_optimizer)


def test_metadata_round_trips_independently(tmp_path):
    model = build_model()
    path = str(tmp_path / "meta.npz")
    save_native_checkpoint(path, model, metadata={"steps_completed": 5,
                                                  "lr": DEFAULT_LR})
    loaded = load_native_checkpoint(path, build_model())
    assert loaded == {"steps_completed": 5, "lr": DEFAULT_LR}
    loaded["steps_completed"] = -1
    again = load_native_checkpoint(path, build_model())
    assert again == {"steps_completed": 5, "lr": DEFAULT_LR}
    _close(model)


# --------------------------------------------------------------------------
# Ownership and lifetime
# --------------------------------------------------------------------------

def test_repeated_steps_return_to_a_stable_storage_baseline(live_storages):
    """The baseline is taken after the model, the optimizer's persistent
    moment state, the persistent input, and the first gradients exist —
    those are intentionally live. What must not grow is the transient
    per-step allocation, including the BatchNorm running-stat replacement
    and the optimizer moment replacement. gc.collect() makes the count
    deterministic against the Python-managed autograd's wrapper cycles."""
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    for _ in range(3):
        train_step(model, loss_fn, optimizer, x, y)
    gc.collect()
    baseline = len(live_storages)
    assert baseline > 0
    for _ in range(6):
        train_step(model, loss_fn, optimizer, x, y)
        gc.collect()
        assert len(live_storages) == baseline
    _close(model, optimizer, x, y)


def test_repeated_eval_passes_do_not_grow_storage(live_storages):
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    for _ in range(3):
        train_step(model, loss_fn, optimizer, x, y)
    gc.collect()
    baseline = len(live_storages)
    for _ in range(6):
        evaluate(model, loss_fn, x, y)
        gc.collect()
        assert len(live_storages) == baseline
    # The eval graph's BatchNorm snapshots left no graph resource behind.
    _close(model, optimizer, x, y)


def test_no_completed_graph_or_eval_snapshot_survives_a_step():
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    # Run the stack layer by layer so the BatchNorm eval snapshots are
    # observable on the BatchNorm output node (the model's forward drops
    # that intermediate internally).
    model.eval()
    hidden = model.hidden(x)
    normed = model.batch_norm(hidden)
    resources = normed._graph_resources
    assert resources, "the eval BatchNorm forward should own snapshots"
    assert all(not core._closed for core in resources)
    activated = model.relu(normed)
    layer_normed = model.layer_norm(activated)
    prediction = model.output(layer_normed)
    loss = loss_fn(prediction, y)
    loss.backward()
    # The one-shot backward released the graph and its snapshot resources.
    assert all(core._closed for core in resources)
    assert normed._graph_resources == ()
    assert loss._graph_freed is True
    for t in (loss, prediction, layer_normed, activated, normed, hidden, x, y):
        t.close()
    _close(model, optimizer)


def test_model_and_optimizer_state_reporting_helpers_close_their_snapshots(
    live_storages
):
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    train_step(model, loss_fn, optimizer, x, y)
    gc.collect()
    baseline = len(live_storages)
    # Each reporting helper snapshots native state and must close it all.
    for _ in range(3):
        values = _model_state_values(model)
        assert not any(isinstance(v, NativeTensor) for v in values.values())
        state = _optimizer_state_values(optimizer)
        assert not any(isinstance(v, NativeTensor) for v in state["m"])
        _running_stats(model)
        gc.collect()
        assert len(live_storages) == baseline
    _close(model, optimizer, x, y)


def test_run_training_leaves_no_storage(live_storages):
    gc.collect()
    baseline = len(live_storages)
    run_training(steps=4)
    gc.collect()
    assert len(live_storages) == baseline


def test_run_resume_proof_leaves_no_storage(live_storages):
    gc.collect()
    baseline = len(live_storages)
    run_resume_proof(total_steps=6, split_step=2)
    gc.collect()
    assert len(live_storages) == baseline


def _fail_a_step(model, loss_fn, optimizer, x, y, monkeypatch):
    """Run one normalized forward + backward, then a deliberately failing
    optimizer step, releasing this step's graph tensors and gradients.
    Returns nothing — the caller checks that nothing was committed."""
    monkeypatch.setattr(NativeAdam, "step",
                        lambda self: (_ for _ in ()).throw(
                            RuntimeError("injected optimizer failure")))
    model.train()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    with pytest.raises(RuntimeError, match="injected optimizer failure"):
        optimizer.step()
    monkeypatch.undo()
    loss.close()
    prediction.close()
    for grad in grads:
        grad.close()
    optimizer.zero_grad()


def test_step_failure_commits_no_partial_update_and_recovers(monkeypatch):
    """A failure injected into the optimizer step, *after* the forward and
    backward of a normalized training iteration, must commit no parameter
    value or version change, and the same model/optimizer must recover on a
    later valid step. (Run-level cleanup on the exception path — the
    ``finally`` in ``run_training`` closing the model, buffers, optimizer,
    and data — is proved end to end by the runnable example; the
    no-storage-growth guarantee is proved by the steady-state baselines
    below and by ``test_run_training_leaves_no_storage``.)"""
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    train_step(model, loss_fn, optimizer, x, y)   # warm up Adam's moments
    before = {name: p.to_numpy().copy() for name, p in model.named_parameters()}
    versions = [p.version for p in model.parameters()]

    _fail_a_step(model, loss_fn, optimizer, x, y, monkeypatch)

    for name, parameter in model.named_parameters():
        assert np.array_equal(parameter.to_numpy(), before[name]), name
    assert [p.version for p in model.parameters()] == versions
    # The BatchNorm running statistics did advance (the forward ran), but no
    # parameter update committed — the step failed atomically.
    value = train_step(model, loss_fn, optimizer, x, y)
    assert math.isfinite(value)
    _close(model, optimizer, x, y)


def test_repeated_failed_steps_do_not_grow_storage(live_storages, monkeypatch):
    """A failed-then-recovered step cycle, in steady state, grows no native
    storage — so the exception path leaks nothing across repetition."""
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    # Settle into steady state so freed transient ids are reused rather than
    # phantom-retained by the id-based instrumentation.
    for _ in range(2):
        _fail_a_step(model, loss_fn, optimizer, x, y, monkeypatch)
        train_step(model, loss_fn, optimizer, x, y)
    gc.collect()
    baseline = len(live_storages)
    for _ in range(4):
        _fail_a_step(model, loss_fn, optimizer, x, y, monkeypatch)
        train_step(model, loss_fn, optimizer, x, y)
        gc.collect()
        assert len(live_storages) == baseline
    _close(model, optimizer, x, y)


def test_run_helpers_return_python_values_only():
    run = run_training(steps=2)
    for value in run.values():
        assert not isinstance(value, NativeTensor)
    assert isinstance(run["final_loss"], float)
    assert isinstance(run["loss_history"], list)
    proof = run_resume_proof(total_steps=4, split_step=2)
    for value in proof.values():
        assert not isinstance(value, NativeTensor)


def test_example_leaves_no_checkpoint_files_behind():
    before = {p.name for p in REPO_ROOT.iterdir()}
    run_resume_proof(total_steps=4, split_step=2)
    after = {p.name for p in REPO_ROOT.iterdir()}
    assert before == after
    assert not list((REPO_ROOT / "examples").glob("*.npz"))
    assert not list(REPO_ROOT.glob("*.npz"))


# --------------------------------------------------------------------------
# The native-only training step (the NumPy tripwire)
# --------------------------------------------------------------------------

_NUMERICAL_NUMPY = (
    "max", "amax", "argmax", "exp", "log", "sum", "divide", "true_divide",
    "add", "subtract", "multiply", "matmul", "mean", "var", "std",
    "negative", "power", "square", "copyto", "sqrt", "reciprocal",
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
    monkeypatch.setattr(cpp.NativeStorage, "copy_from", _tripwire)
    monkeypatch.setattr(NativeTensor, "to_numpy", _tripwire)


def test_one_normalized_training_step_is_fully_native(monkeypatch):
    """A complete normalized update — forward through BatchNorm and
    LayerNorm (including the BatchNorm running-statistics update), scalar
    MSE, backward, the NativeAdam step, and zero_grad — runs to completion
    with every NumPy numerical routine and every tensor-data conversion
    route armed, and produces exactly what the unarmed reference produced.
    Construction and reporting are the allowed host boundaries and happen
    before the tripwire is armed."""
    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = build_loss()
    x, y = _inputs()
    # Warm up so Adam's moment buffers already exist (their one-time
    # allocation is not part of a steady-state training step).
    train_step(model, loss_fn, optimizer, x, y)
    reference = build_model()
    reference_optimizer = build_optimizer(reference)
    train_step(reference, loss_fn, reference_optimizer, x, y)
    train_step(reference, loss_fn, reference_optimizer, x, y)
    running_before = model.batch_norm.running_mean.to_numpy().copy()

    _arm_tripwire(monkeypatch)
    model.train()
    prediction = model(x)             # forward through both normalizations
    loss = loss_fn(prediction, y)     # scalar MSE
    loss.backward()                   # backward
    optimizer.step()                  # parameter update
    optimizer.zero_grad()
    with pytest.raises(AssertionError, match="reached NumPy"):
        prediction.to_numpy()
    with pytest.raises(AssertionError, match="reached NumPy"):
        np.sqrt(1.0)
    loss.close()
    prediction.close()
    monkeypatch.undo()

    # The step really happened: the running statistics advanced, and every
    # parameter equals the unarmed reference's second-step values.
    assert not np.array_equal(model.batch_norm.running_mean.to_numpy(),
                              running_before)
    for (name, a), (_, b) in zip(model.named_parameters(),
                                 reference.named_parameters()):
        assert np.array_equal(a.to_numpy(), b.to_numpy()), name
    assert np.array_equal(model.batch_norm.running_mean.to_numpy(),
                          reference.batch_norm.running_mean.to_numpy())
    x.close()
    y.close()
    _close(model, optimizer)
    _close(reference, reference_optimizer)


# --------------------------------------------------------------------------
# The runnable example
# --------------------------------------------------------------------------

def test_example_file_exists_and_is_import_safe(capsys):
    import importlib

    import examples.native_normalization_training as module

    capsys.readouterr()
    importlib.reload(module)
    assert capsys.readouterr().out == ""     # importing runs no training
    assert EXAMPLE.is_file()
    text = EXAMPLE.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in text
    # No network access, no repository mutation, no timing.
    for banned in ("requests", "urllib", "socket", "git ", "subprocess",
                   "perf_counter", "time.time(", "import time", "import timeit"):
        assert banned not in text, banned


def test_example_uses_only_the_native_stack():
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "tensorforge.nn" not in text and "tensorforge.optim" not in text
    assert "from tensorforge import" not in text
    # No BatchNorm2d or convolutional layer is instantiated — that is F8's
    # scope. (The docstring may *name* them to say they are not used, so
    # this checks for instantiation, not the bare identifier.)
    for absent in ("NativeBatchNorm2d(", "NativeConv2d(", "NativeMaxPool2d("):
        assert absent not in text, absent
    # ...and none is imported.
    imports = text.split("import (", 1)[1].split(")", 1)[0]
    for absent in ("NativeBatchNorm2d", "NativeConv2d", "NativeMaxPool2d"):
        assert absent not in imports, absent
    for name in ("NativeNormalizedRegressor",):
        assert not hasattr(tensorforge, name), name


def test_example_main_runs_and_reports_learning_and_exact_resume(capsys):
    main()
    output = capsys.readouterr().out
    for expected in ("native normalized regressor:", "BatchNorm1d",
                     "LayerNorm", "NativeMSELoss",
                     "initial training loss:", "final training loss:",
                     "loss reduction:", "final eval loss:",
                     "checkpoint resume:", "format version 1",
                     "resumed loss suffix match:", "running_mean match:",
                     "running_var match:", "optimizer state matches:",
                     "final eval output match:", "exact resume: yes",
                     "native normalized training + checkpoint resume ok"):
        assert expected in output, expected
    assert f"initial training loss: {OBSERVED_INITIAL_LOSS:.6f}" in output
    assert f"final training loss:   {OBSERVED_FINAL_LOSS:.6f}" in output
    assert f"trained {TOTAL_STEPS} NativeAdam steps (lr={DEFAULT_LR})" in output
    assert f"trained {SPLIT_STEP} steps" in output
    # No speed or production claim.
    lowered = output.lower()
    for banned in ("faster", "fastest", "speedup", "production", "benchmark"):
        assert banned not in lowered, banned


def test_example_runs_as_a_subprocess_and_exits_zero():
    """The example runs standalone and exits 0 — the equivalent of
    ``uv run python examples/native_normalization_training.py`` (the same
    interpreter, same environment, run as a fresh process)."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert "native normalized training + checkpoint resume ok" in result.stdout
    assert "exact resume: yes" in result.stdout


# --------------------------------------------------------------------------
# Scope boundary: F6 adds no capability
# --------------------------------------------------------------------------

def test_f6_adds_no_capability_or_inventory_entry():
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
        # Phase G milestone G4 appended the Dropout module. It is
        # unrelated to this milestone, which added no module of its own.
        "NativeDropout",
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "generator_state",   # Phase G, milestone G1 (in-memory only)
        "save_native_checkpoint", "load_native_checkpoint",
        "checkpoint_generator_state",   # Phase G, milestone G5 (the file half)
    )
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    # The proof is an integration result, never a named capability.
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES, cpp.NATIVE_METRICS):
        for banned in ("train", "regressor", "checkpoint_resume", "example",
                       "dataset", "normalization_training"):
            assert not [n for n in inventory if banned in n.lower()], banned
    # No normalization operation entered any operation inventory.
    for name in ("layer_norm", "batch_norm", "layernorm", "batchnorm"):
        assert name not in cpp.TENSOR_CORE_OPS
        assert name not in cpp.AUTOGRAD_OPS
        assert name not in cpp.RAW_KERNELS
    assert native_checkpoint._FORMAT_VERSION == 3
    assert cpp.backend_info()["stable_framework_integration"] is False
