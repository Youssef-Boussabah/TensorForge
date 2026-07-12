"""Tests for the native MLP training proof (Advanced C++ v3.9).

examples/native_mlp_training.py trains NativeSequential(NativeLinear(2,
8, seed=0), NativeReLU(), NativeLinear(8, 1, seed=1)) on 8 fixed
synthetic regression samples for 25 steps of NativeSGD(lr=0.1) —
entirely through the experimental native stack, with a completely fresh
graph every iteration, one version increment per parameter per step,
stable parameter identities, explicit per-iteration lifetime handling,
and a monotonically decreasing deterministic loss (about 2.1079 →
about 0.0095). ``train()`` returns plain Python values only.

NumPy appears below only for references and inspection; the training
computation is native (a tripwire test proves it).

Selector: python -m pytest -q -k "native_mlp_training"
"""

import math

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeLinear,
    NativeMSELoss,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
)
from examples.native_mlp_training import (
    DEFAULT_LR,
    DEFAULT_STEPS,
    X_VALUES,
    Y_VALUES,
    main,
    train,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


PARAMETER_NAMES = ["0.weight", "0.bias", "2.weight", "2.bias"]


@pytest.fixture(scope="module")
def run():
    """One shared full training run — every stats assertion reads it."""
    if not cpp.is_available():
        pytest.skip("experimental C++ backend not built")
    return train()


def _fresh_setup():
    """A model/loss/optimizer/data setup matching the example, for the
    hand-driven lifecycle tests. The caller owns every tensor."""
    model = NativeSequential(
        NativeLinear(2, 8, seed=0),
        NativeReLU(),
        NativeLinear(8, 1, seed=1),
    )
    optimizer = NativeSGD(model.parameters(), lr=DEFAULT_LR)
    x = NativeTensor.from_array(X_VALUES)
    y = NativeTensor.from_array(Y_VALUES)
    return model, NativeMSELoss(), optimizer, x, y


# ======================================================================
# End-to-end training and loss behavior
# ======================================================================


@needs_native
def test_native_mlp_training_completes_with_finite_decreasing_loss(run):
    assert run["steps"] == DEFAULT_STEPS and run["lr"] == DEFAULT_LR
    assert len(run["loss_history"]) == DEFAULT_STEPS
    trajectory = run["loss_history"] + [run["final_loss"]]
    assert all(
        isinstance(value, float) and math.isfinite(value)
        for value in trajectory
    )
    assert run["initial_loss"] == run["loss_history"][0]
    # This configuration decreases monotonically every single step.
    assert all(
        later < earlier
        for earlier, later in zip(trajectory, trajectory[1:])
    )
    # Meaningful deterministic reduction, with generous headroom over
    # the observed 2.1079 → 0.0095 trajectory: the test fails loudly
    # if parameters stop updating, without depending on exact floats.
    assert run["initial_loss"] > 1.0
    assert run["final_loss"] < 0.05
    assert run["final_loss"] < 0.05 * run["initial_loss"]


@needs_native
def test_native_mlp_training_every_parameter_learns(run):
    assert run["parameter_names"] == PARAMETER_NAMES
    for name in PARAMETER_NAMES:
        before = np.asarray(run["initial_parameters"][name])
        after = np.asarray(run["final_parameters"][name])
        assert before.shape == after.shape
        assert not np.array_equal(before, after)  # it actually moved
        assert np.all(np.isfinite(after))


@needs_native
def test_native_mlp_training_versions_advance_once_per_step(run):
    assert run["initial_versions"] == [0, 0, 0, 0]  # fresh model
    assert run["final_versions"] == [DEFAULT_STEPS] * 4
    assert len(run["version_history"]) == DEFAULT_STEPS
    # Exactly one increment per parameter per step — never two, never
    # zero — and the final evaluation/zero_grad added nothing.
    for step, snapshot in enumerate(run["version_history"], start=1):
        assert snapshot == [step] * 4


@needs_native
def test_native_mlp_training_identity_gradients_and_names_stable(run):
    assert run["identity_stable"] is True
    assert run["names_stable"] is True
    assert run["gradient_lifecycle_ok"] is True
    assert run["gradients_cleared"] is True


@needs_native
def test_native_mlp_training_is_deterministic_across_runs(run):
    repeat = train()
    # Bit-identical float64 arithmetic: exact equality, no tolerance.
    assert repeat["loss_history"] == run["loss_history"]
    assert repeat["final_loss"] == run["final_loss"]
    assert repeat["final_parameters"] == run["final_parameters"]
    assert repeat["version_history"] == run["version_history"]
    assert repeat["final_versions"] == run["final_versions"]


@needs_native
def test_native_mlp_training_validates_arguments():
    for bad_steps in (True, 2.0, "25", None):
        with pytest.raises(TypeError, match="steps"):
            train(steps=bad_steps)
    for bad_steps in (0, -3):
        with pytest.raises(ValueError, match="strictly positive"):
            train(steps=bad_steps)
    with pytest.raises(ValueError, match="strictly positive"):
        train(lr=0.0)  # NativeSGD's own validation, surfaced unchanged
    with pytest.raises(TypeError, match="real number"):
        train(lr="0.1")


# ======================================================================
# Hand-driven lifecycle: gradients, fresh graphs, staleness, lifetime
# ======================================================================


@needs_native
def test_native_mlp_training_gradient_lifecycle_per_iteration():
    model, loss_fn, optimizer, x, y = _fresh_setup()
    parameters = model.parameters()
    for _ in range(3):
        assert all(p.grad is None for p in parameters)  # cleared start
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        grads = [p.grad for p in parameters]
        assert all(grad is not None for grad in grads)
        assert all(np.all(np.isfinite(grad.to_numpy())) for grad in grads)
        optimizer.step()
        # Gradients survive step() by identity; versions advanced.
        assert [p.grad for p in parameters] == grads
        optimizer.zero_grad()
        assert all(p.grad is None for p in parameters)
        loss.close()
        prediction.close()
    # The caller-owned permanent tensors were never closed.
    assert not x.closed and not y.closed
    assert all(not p.closed for p in parameters)
    assert [p.version for p in parameters] == [3, 3, 3, 3]


@needs_native
def test_native_mlp_training_zero_grad_prevents_cross_iteration_accumulation():
    model, loss_fn, optimizer, x, y = _fresh_setup()
    weight = model[0].weight

    def backward_once():
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        loss.close()
        prediction.close()

    backward_once()
    first = weight.grad.to_numpy()
    optimizer.zero_grad()
    backward_once()  # same parameters, fresh graph: identical gradient
    assert np.array_equal(weight.grad.to_numpy(), first)
    backward_once()  # deliberately no zero_grad: accumulation doubles
    assert np.allclose(weight.grad.to_numpy(), 2.0 * first, atol=1e-15)
    optimizer.zero_grad()


@needs_native
def test_native_mlp_training_stale_guard_and_fresh_graph_after_step():
    model, loss_fn, optimizer, x, y = _fresh_setup()
    # One concise negative guard: deliberately keeping the old
    # sensitive graph across step() hits the existing v3.7 stale error.
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward(retain_graph=True)  # deliberate retention, not the loop's way
    optimizer.step()
    with pytest.raises(RuntimeError, match="stale"):
        loss.backward(retain_graph=True)
    loss.close()
    prediction.close()
    optimizer.zero_grad()
    # The training loop's way: a completely fresh graph succeeds and
    # trains on the updated values.
    fresh_prediction = model(x)
    fresh_loss = loss_fn(fresh_prediction, y)
    fresh_loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    fresh_loss.close()
    fresh_prediction.close()
    optimizer.zero_grad()


@needs_native
def test_native_mlp_training_state_dict_keys_stable_across_training():
    model, loss_fn, optimizer, x, y = _fresh_setup()
    keys_before = list(model.state_dict())
    assert keys_before == PARAMETER_NAMES
    for _ in range(2):
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss.close()
        prediction.close()
    state_after = model.state_dict()
    assert list(state_after) == keys_before
    assert [name for name, _ in model.named_parameters()] == PARAMETER_NAMES
    for value in state_after.values():
        value.close()


# ======================================================================
# Native-only guardrails
# ======================================================================


@needs_native
def test_native_mlp_training_uses_no_numpy_compute(monkeypatch):
    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in ("add", "subtract", "multiply", "matmul", "sum", "mean",
                 "divide", "negative", "copyto", "dot"):
        monkeypatch.setattr(np, name, _tripwire)
    result = train(steps=3)
    monkeypatch.undo()
    assert result["final_loss"] < result["initial_loss"]
    assert result["final_versions"] == [3, 3, 3, 3]


@needs_native
def test_native_mlp_training_example_source_stays_inside_the_contract():
    import examples.native_mlp_training as example

    with open(example.__file__, encoding="utf-8") as handle:
        source = handle.read()
    # No private-internals manipulation, no manual gradients, no stale
    # workarounds, no stable-framework computation.
    for forbidden in ("._grad", "._core", "._version", "retain_graph=",
                      "import numpy", "tensorforge.Tensor",
                      "from tensorforge import"):
        assert forbidden not in source
    assert "from tensorforge.experimental import" in source


# ======================================================================
# Executable example
# ======================================================================


@needs_native
def test_native_mlp_training_example_main_prints_report(capsys):
    main()
    output = capsys.readouterr().out
    assert "Native MLP training proof" in output
    assert "initial loss: 2.107864" in output
    assert "final loss: 0.009529" in output
    assert "99.5% reduction" in output
    assert "gradients cleared: True" in output
    assert "native MLP training ok" in output
