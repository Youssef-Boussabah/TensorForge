"""Characterization benchmark for the native classification stack
(Advanced C++ Phase E, milestone E9).

This measures the completed Phase-E line — the differentiable ``exp``,
``log``, ``softmax``, and ``log_softmax`` forwards, the fused
``cross_entropy`` forward and backward, and one complete E8-style native
classification training step. It does **not** try to make anything
faster, and **nothing here asserts a speed**: the native kernels are
deliberately direct loops (correctness-first — no im2col, BLAS,
threading, or SIMD), so this is an honest, reproducible snapshot of one
machine at one moment. It is **not** a performance contract, not
comparable across machines without controlled conditions, and there is
no CI timing threshold anywhere in this repository.

**Correctness runs before timing, always.** Every case validates its
native result against a reference — the stable NumPy framework where an
equivalent operation exists, an explicit NumPy formula where it does not
— and a failed gate aborts that run with a nonzero exit status and
publishes no timing for the case.

Each case is labelled with the reference it actually used:

- ``stable_tensorforge`` — a real ``tensorforge`` (stable-line) operation
  on identical inputs;
- ``numpy`` — an explicit NumPy formula, used where the stable framework
  has no direct equivalent;
- ``native_only`` — no honest analogue exists, so no ratio is reported.

The measured times include the Python wrapper and the ctypes boundary as
well as the native compute: that is what a caller actually pays.

**What is timed.** One measured repetition times exactly one call of the
case's operation with ``time.perf_counter_ns()``. Setup (input creation,
graph construction for the backward case, model/optimizer construction
for the training step) happens **outside** the timer, and cleanup happens
outside it too. No sample is discarded, no timer overhead is subtracted,
and the native and reference paths run under the same setup discipline.
The median is the primary statistic; the minimum, maximum, and their
spread show the variation.

Build the backend first:

    uv run python cpp/build.py

Then, for example:

    uv run python benchmarks/benchmark_native_classification.py
    uv run python benchmarks/benchmark_native_classification.py --smoke
    uv run python benchmarks/benchmark_native_classification.py --smoke --json
    uv run python benchmarks/benchmark_native_classification.py --case softmax_forward

E9 adds **no** numerical capability: no kernel, C ABI export, ctypes
symbol, operation, module, loss, metric, optimizer, or checkpoint
change. It only measures what E1–E8 already shipped, and the E8 training
proof (``examples/native_classification_training.py``) remains the
separate, authoritative correctness-and-resume artifact.
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone

import numpy as np

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
)

BENCHMARK_NAME = "tensorforge.native_classification"
BENCHMARK_VERSION = "1.0"

# Reference labels. Every case declares exactly one.
STABLE = "stable_tensorforge"
NUMPY = "numpy"
NATIVE_ONLY = "native_only"

# Warm-up / repetition defaults. The training step is heavier than the
# elementwise cases, so it declares its own smaller repetition count and
# the actual count is always reported per case.
DEFAULTS = {"warmup": 3, "repetitions": 12}
SMOKE_DEFAULTS = {"warmup": 1, "repetitions": 3}
TRAINING_STEP_REPETITIONS = 6

# Absolute tolerances for the correctness gates. These are ordinary
# float64 agreement bounds, not performance criteria.
FORWARD_ATOL = 1e-12
LOSS_ATOL = 1e-12
GRADIENT_ATOL = 1e-14
PARAMETER_ATOL = 1e-12


# ---------------------------------------------------------------------------
# Deterministic host inputs
# ---------------------------------------------------------------------------


def _rng(seed):
    """A local generator. The global NumPy RNG is never touched."""
    return np.random.default_rng(seed)


def _moderate_values(shape, seed):
    """Finite, moderate float64 values — no overflow or underflow, so
    ``exp`` is measured on representative data rather than on special
    cases (those belong in the correctness suites)."""
    return _rng(seed).uniform(-2.0, 2.0, size=shape)


def _positive_values(shape, seed):
    """Positive values bounded away from zero, so ``log`` never sees a
    domain edge in the timed data."""
    return _rng(seed).uniform(0.25, 4.0, size=shape)


def _logit_values(shape, seed):
    """Representative finite logits."""
    return _rng(seed).uniform(-3.0, 3.0, size=shape)


def _labels(batch_size, num_classes, seed):
    """Strict deterministic integer class labels (host ``int``)."""
    return [int(v) for v in _rng(seed).integers(0, num_classes, size=batch_size)]


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def _numpy_log_softmax(values, axis=-1):
    """The stable log-sum-exp form: ``(x - max) - log(sum(exp(x - max)))``.

    Deliberately **not** ``log(softmax(x))`` — that composition is exactly
    what the native E4 kernel exists to avoid, so using it as the
    reference would compare against the weaker formula."""
    shifted = values - np.max(values, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def _numpy_cross_entropy(values, targets):
    log_probs = _numpy_log_softmax(values, axis=1)
    rows = np.arange(values.shape[0])
    return float(-log_probs[rows, np.asarray(targets, dtype=np.int64)].mean())


def _numpy_cross_entropy_gradient(values, targets):
    probabilities = np.exp(_numpy_log_softmax(values, axis=1))
    rows = np.arange(values.shape[0])
    probabilities[rows, np.asarray(targets, dtype=np.int64)] -= 1.0
    return probabilities / values.shape[0]


# ---------------------------------------------------------------------------
# The E8 model, rebuilt here so the benchmark never runs the example's
# main(). The dataset and the architecture are the example's; importing
# the example module is import-safe and runs no training.
# ---------------------------------------------------------------------------


# Running this file as a script puts ``benchmarks/`` on sys.path rather
# than the repository root, so make the root importable before reaching
# for the E8 example. Idempotent, and a no-op under pytest (which already
# has the root on the path). The example module is import-safe: importing
# it defines the dataset and the model builders and runs no training.
def _ensure_repository_root_on_path():
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repository_root_on_path()

from examples.native_classification_training import (  # noqa: E402
    CONV_SEED,
    DEFAULT_LR,
    FLAT_FEATURES,
    LINEAR_SEED,
    NUM_CLASSES,
    build_dataset,
    build_model,
)


def _stable_classifier(initial_parameters):
    """The structurally equivalent stable-line model, initialized from
    the *same* arrays the native model starts from.

    Same architecture, same initial values, same fused cross-entropy
    semantics (mean reduction over raw logits), and the same Adam
    formula and hyperparameters — which is what makes the training-step
    comparison honest rather than decorative. The correctness gate
    proves the equivalence numerically on every run."""
    from tensorforge.nn import (Conv2d, Flatten, Linear, MaxPool2d, ReLU,
                                Sequential)

    model = Sequential(
        Conv2d(1, 4, 3), ReLU(), MaxPool2d(2), Flatten(),
        Linear(FLAT_FEATURES, NUM_CLASSES),
    )
    conv, linear = model.modules[0], model.modules[4]
    conv.weight.data = initial_parameters["conv.weight"].copy()
    conv.bias.data = initial_parameters["conv.bias"].copy()
    linear.weight.data = initial_parameters["linear.weight"].copy()
    linear.bias.data = initial_parameters["linear.bias"].copy()
    return model


def _native_initial_parameters():
    """The E8 model's deterministic initial parameter arrays, read once
    outside every timed region and then released."""
    model = build_model()
    try:
        return {name: parameter.to_numpy().copy()
                for name, parameter in model.named_parameters()}
    finally:
        for parameter in model.parameters():
            parameter.close()


class _BenchmarkClassifier(NativeModule):
    """The E8 architecture, constructed here from the same seeds so the
    benchmark owns its models and never mutates the example's."""

    def __init__(self):
        super().__init__()
        self.conv = NativeConv2d(1, 4, 3, seed=CONV_SEED)
        self.relu = NativeReLU()
        self.pool = NativeMaxPool2d(2)
        self.flatten = NativeFlatten()
        self.linear = NativeLinear(FLAT_FEATURES, NUM_CLASSES,
                                   seed=LINEAR_SEED)

    def forward(self, images):
        hidden = self.conv(images)
        hidden = self.relu(hidden)
        hidden = self.pool(hidden)
        hidden = self.flatten(hidden)
        return self.linear(hidden)


# ---------------------------------------------------------------------------
# Case builders. Each returns a dict of untimed ``prepare``/``cleanup``
# callables around one timed ``run``, plus a ``check`` that runs the whole
# correctness gate before any timing, and a ``close`` that releases the
# case's shared persistent inputs after every repetition is done.
# ---------------------------------------------------------------------------


def _unary_case(shape, seed, values_for, native_method, stable_method,
                numpy_function, axis=None):
    """Shared shape for the four forward cases: one persistent native
    input, one persistent stable input, and a fresh output per call."""
    values = values_for(shape, seed)
    native_input = NativeTensor.from_array(values)
    stable_input = tensorforge.Tensor(values.copy())
    before = values.copy()

    def native_run(_state=None):
        method = getattr(native_input, native_method)
        return method() if axis is None else method(axis=axis)

    def native_cleanup(_state, result):
        result.close()

    def reference_run(_state=None):
        if stable_method is None:
            return numpy_function(stable_input.data)
        method = getattr(stable_input, stable_method)
        return method() if axis is None else method(axis=axis)

    def check():
        expected = numpy_function(values)
        native_out = native_run()
        try:
            produced = native_out.to_numpy()
            if produced.shape != expected.shape:
                raise AssertionError(
                    f"native shape {produced.shape} != {expected.shape}"
                )
            if not np.all(np.isfinite(produced)):
                raise AssertionError("native output is not finite")
            native_error = float(np.max(np.abs(produced - expected)))
            if native_error > FORWARD_ATOL:
                raise AssertionError(
                    f"native output differs from the reference by "
                    f"{native_error:g} (> {FORWARD_ATOL:g})"
                )
        finally:
            native_out.close()
        reference_out = reference_run()
        produced_reference = (reference_out.data
                              if hasattr(reference_out, "data")
                              else reference_out)
        reference_error = float(np.max(np.abs(produced_reference - expected)))
        if reference_error > FORWARD_ATOL:
            raise AssertionError(
                f"reference output differs from the NumPy formula by "
                f"{reference_error:g}"
            )
        if not np.array_equal(native_input.to_numpy(), before):
            raise AssertionError("the native input was mutated")
        if not np.array_equal(stable_input.data, before):
            raise AssertionError("the reference input was mutated")
        return {
            "max_abs_error": native_error,
            "reference_max_abs_error": reference_error,
            "checks": ["shape", "finite", "reference_parity", "no_input_mutation"],
        }

    return {
        "native_prepare": lambda: None,
        "native_run": native_run,
        "native_cleanup": native_cleanup,
        "reference_prepare": lambda: None,
        "reference_run": reference_run,
        "reference_cleanup": lambda _state, _result: None,
        "check": check,
        "close": lambda: native_input.close(),
    }


def _build_exp(shape, seed):
    return _unary_case(shape, seed, _moderate_values, "exp", "exp", np.exp)


def _build_log(shape, seed):
    return _unary_case(shape, seed, _positive_values, "log", "log", np.log)


def _numpy_softmax(values, axis=-1):
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def _build_softmax(shape, seed):
    case = _unary_case(shape, seed, _logit_values, "softmax", "softmax",
                       _numpy_softmax, axis=-1)
    inner = case["check"]

    def check():
        metrics = inner()
        # Correctness-only stability probe on a large but finite offset:
        # never part of the timed data, and exactly the case a naive
        # implementation fails.
        extreme = _logit_values(shape, seed) + 1000.0
        shifted = NativeTensor.from_array(extreme)
        out = shifted.softmax(axis=-1)
        try:
            produced = out.to_numpy()
            if not np.all(np.isfinite(produced)):
                raise AssertionError("softmax overflowed on a +1000 offset")
            sums = np.sum(produced, axis=-1)
            if not np.allclose(sums, 1.0, atol=1e-12):
                raise AssertionError("softmax rows do not sum to 1")
            metrics["extreme_offset_max_row_sum_error"] = float(
                np.max(np.abs(sums - 1.0))
            )
        finally:
            out.close()
            shifted.close()
        metrics["checks"] = metrics["checks"] + ["extreme_offset_stability"]
        return metrics

    case["check"] = check
    return case


def _build_log_softmax(shape, seed):
    # The stable framework has no direct log_softmax, and softmax().log()
    # is the composition E4 exists to avoid, so the reference is the
    # explicit NumPy log-sum-exp formula (labelled `numpy`).
    return _unary_case(shape, seed, _logit_values, "log_softmax", None,
                       _numpy_log_softmax, axis=-1)


def _build_cross_entropy_forward(shape, seed):
    """Forward loss creation only — no graph, no backward, on both sides."""
    batch_size, num_classes = shape
    values = _logit_values(shape, seed)
    targets = _labels(batch_size, num_classes, seed + 1)
    before = values.copy()
    native_logits = NativeTensor.from_array(values)
    stable_logits = tensorforge.Tensor(values.copy())

    def native_run(_state=None):
        return native_logits.cross_entropy(targets, reduction="mean")

    def reference_run(_state=None):
        from tensorforge.nn import cross_entropy
        return cross_entropy(stable_logits, targets)

    def check():
        expected = _numpy_cross_entropy(values, targets)
        loss = native_run()
        try:
            if loss.shape != ():
                raise AssertionError(f"native loss shape {loss.shape} != ()")
            produced = float(loss.to_numpy())
            if not np.isfinite(produced):
                raise AssertionError("native loss is not finite")
            native_error = abs(produced - expected)
            if native_error > LOSS_ATOL:
                raise AssertionError(
                    f"native loss differs from the reference by {native_error:g}"
                )
        finally:
            loss.close()
        stable_loss = reference_run()
        reference_error = abs(float(stable_loss.data) - expected)
        if reference_error > LOSS_ATOL:
            raise AssertionError(
                f"stable loss differs from the NumPy formula by "
                f"{reference_error:g}"
            )
        if not np.array_equal(native_logits.to_numpy(), before):
            raise AssertionError("the native logits were mutated")
        if not np.array_equal(stable_logits.data, before):
            raise AssertionError("the stable logits were mutated")
        return {
            "max_abs_error": native_error,
            "reference_max_abs_error": reference_error,
            "checks": ["scalar_shape", "finite", "reference_parity",
                       "no_input_mutation"],
        }

    return {
        "native_prepare": lambda: None,
        "native_run": native_run,
        "native_cleanup": lambda _state, result: result.close(),
        "reference_prepare": lambda: None,
        "reference_run": reference_run,
        "reference_cleanup": lambda _state, _result: None,
        "check": check,
        "close": lambda: native_logits.close(),
    }


def _build_cross_entropy_backward(shape, seed):
    """Backward only: every repetition builds a fresh graph outside the
    timer and times ``backward()`` alone. No graph is ever reused and
    ``retain_graph`` is never used to skip the rebuild."""
    batch_size, num_classes = shape
    values = _logit_values(shape, seed)
    targets = _labels(batch_size, num_classes, seed + 1)

    def native_prepare():
        logits = NativeTensor.from_array(values, requires_grad=True)
        return logits, logits.cross_entropy(targets, reduction="mean")

    def native_run(state):
        _logits, loss = state
        loss.backward()
        return loss

    def native_cleanup(state, _result):
        logits, loss = state
        loss.close()
        logits.close()

    def reference_prepare():
        from tensorforge.nn import cross_entropy
        logits = tensorforge.Tensor(values.copy(), requires_grad=True)
        return logits, cross_entropy(logits, targets)

    def reference_run(state):
        _logits, loss = state
        loss.backward()
        return loss

    def check():
        expected = _numpy_cross_entropy_gradient(values, targets)
        state = native_prepare()
        logits, _loss = state
        try:
            native_run(state)
            if logits.grad is None:
                raise AssertionError("native backward produced no gradient")
            gradient = logits.grad.to_numpy()
            if gradient.shape != values.shape:
                raise AssertionError(
                    f"native gradient shape {gradient.shape} != {values.shape}"
                )
            if not np.all(np.isfinite(gradient)):
                raise AssertionError("native gradient is not finite")
            native_error = float(np.max(np.abs(gradient - expected)))
            if native_error > GRADIENT_ATOL:
                raise AssertionError(
                    f"native gradient differs from the reference by "
                    f"{native_error:g}"
                )
            if not np.array_equal(logits.to_numpy(), values):
                raise AssertionError("the native logits were mutated")
        finally:
            native_cleanup(state, None)
        reference_state = reference_prepare()
        reference_logits, _ = reference_state
        reference_run(reference_state)
        reference_error = float(np.max(np.abs(reference_logits.grad - expected)))
        if reference_error > GRADIENT_ATOL:
            raise AssertionError(
                f"stable gradient differs from the NumPy reference by "
                f"{reference_error:g}"
            )
        return {
            "max_abs_error": native_error,
            "reference_max_abs_error": reference_error,
            "checks": ["gradient_present", "gradient_shape", "finite",
                       "reference_parity", "no_input_mutation"],
        }

    return {
        "native_prepare": native_prepare,
        "native_run": native_run,
        "native_cleanup": native_cleanup,
        "reference_prepare": reference_prepare,
        "reference_run": reference_run,
        "reference_cleanup": lambda _state, _result: None,
        "check": check,
        "close": lambda: None,
    }


def _build_training_step(shape, seed):
    """One complete E8-style classification training step.

    The timed region is exactly ``zero_grad -> forward -> loss ->
    backward -> optimizer.step()``. The model, the optimizer, and the
    input tensor are built outside it; a **fresh** model and optimizer
    are constructed for every repetition so each timed step starts from
    the same deterministic state. ``native_accuracy``, reporting
    conversions, checkpoint I/O, and cleanup are all excluded."""
    del seed  # the dataset and initialization are fixed, not sampled
    images, targets = build_dataset()
    x = NativeTensor.from_array(images)
    criterion = NativeCrossEntropyLoss()
    initial = _native_initial_parameters()

    def native_prepare():
        model = _BenchmarkClassifier()
        return model, NativeAdam(model.parameters(), lr=DEFAULT_LR)

    def native_run(state):
        model, optimizer = state
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        return logits, loss

    def native_cleanup(state, result):
        model, optimizer = state
        if result is not None:
            logits, loss = result
            loss.close()
            logits.close()
        optimizer.close()
        for parameter in model.parameters():
            parameter.close()

    def reference_prepare():
        from tensorforge.optim import Adam
        model = _stable_classifier(initial)
        return model, Adam(model.parameters(), lr=DEFAULT_LR)

    def reference_run(state):
        from tensorforge.nn import cross_entropy
        model, optimizer = state
        optimizer.zero_grad()
        logits = model(reference_inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        return logits, loss

    reference_inputs = tensorforge.Tensor(np.asarray(images, dtype=np.float64))

    def check():
        state = native_prepare()
        model, optimizer = state
        before = {name: parameter.to_numpy().copy()
                  for name, parameter in model.named_parameters()}
        result = None
        try:
            result = native_run(state)
            logits, loss = result
            loss_value = float(loss.to_numpy())
            if not np.isfinite(loss_value):
                raise AssertionError("the training-step loss is not finite")
            if loss.shape != ():
                raise AssertionError("the training-step loss is not scalar")
            # The step really updated the model.
            changed = [
                name for name, parameter in model.named_parameters()
                if not np.array_equal(parameter.to_numpy(), before[name])
            ]
            if not changed:
                raise AssertionError("no parameter changed during the step")
            # ...and the optimizer's own state advanced.
            steps = list(optimizer.state_dict()["step_counts"])
            if steps != [1] * len(steps):
                raise AssertionError(
                    f"optimizer step counts did not advance: {steps}"
                )
            # No completed graph or saved probability survives the step.
            if not loss._graph_freed:
                raise AssertionError("the completed graph was not released")
            if loss._graph_resources != () or logits._graph_resources != ():
                raise AssertionError("a graph resource survived the step")
            after = {name: parameter.to_numpy().copy()
                     for name, parameter in model.named_parameters()}
        finally:
            native_cleanup(state, result)
        if not (result[1].closed and result[0].closed):
            raise AssertionError("the step's transient tensors were not closed")

        reference_state = reference_prepare()
        reference_model, _ = reference_state
        reference_logits, reference_loss = reference_run(reference_state)
        reference_loss_value = float(reference_loss.data)
        loss_error = abs(loss_value - reference_loss_value)
        if loss_error > LOSS_ATOL:
            raise AssertionError(
                f"native and stable step losses differ by {loss_error:g}"
            )
        stable_after = {
            "conv.weight": reference_model.modules[0].weight.data,
            "conv.bias": reference_model.modules[0].bias.data,
            "linear.weight": reference_model.modules[4].weight.data,
            "linear.bias": reference_model.modules[4].bias.data,
        }
        parameter_error = max(
            float(np.max(np.abs(stable_after[name] - after[name])))
            for name in after
        )
        if parameter_error > PARAMETER_ATOL:
            raise AssertionError(
                f"native and stable parameters differ by {parameter_error:g} "
                f"after one step"
            )
        del reference_logits
        return {
            "max_abs_error": parameter_error,
            "loss_abs_error": loss_error,
            "updated_parameters": sorted(changed),
            "checks": ["finite_loss", "parameter_updated",
                       "optimizer_state_advanced", "graph_released",
                       "transients_closed", "reference_parity"],
        }

    def close():
        x.close()

    return {
        "native_prepare": native_prepare,
        "native_run": native_run,
        "native_cleanup": native_cleanup,
        "reference_prepare": reference_prepare,
        "reference_run": reference_run,
        "reference_cleanup": lambda _state, _result: None,
        "check": check,
        "close": close,
    }


# ---------------------------------------------------------------------------
# The case registry — exactly the seven E9 operations.
# ---------------------------------------------------------------------------

CASES = {
    "exp_forward": {
        "category": "stable_math",
        "operation": "NativeTensor.exp()",
        "reference_type": STABLE,
        "reference_detail": "tensorforge.Tensor.exp()",
        "axis": None,
        "reduction": None,
        "seed": 20250001,
        "shapes": {"full": (16384,), "smoke": (512,)},
        "build": _build_exp,
        "notes": ("Moderate finite values in [-2, 2]; overflow and underflow "
                  "are correctness-suite cases and are excluded from the "
                  "timed data."),
    },
    "log_forward": {
        "category": "stable_math",
        "operation": "NativeTensor.log()",
        "reference_type": STABLE,
        "reference_detail": "tensorforge.Tensor.log()",
        "axis": None,
        "reduction": None,
        "seed": 20250002,
        "shapes": {"full": (16384,), "smoke": (512,)},
        "build": _build_log,
        "notes": ("Positive values in [0.25, 4.0], bounded away from zero; no "
                  "domain-edge value appears in the timed data."),
    },
    "softmax_forward": {
        "category": "probability_transform",
        "operation": "NativeTensor.softmax(axis=-1)",
        "reference_type": STABLE,
        "reference_detail": "tensorforge.Tensor.softmax(axis=-1)",
        "axis": -1,
        "reduction": None,
        "seed": 20250003,
        "shapes": {"full": (128, 64), "smoke": (16, 8)},
        "build": _build_softmax,
        "notes": ("Timed on representative finite logits; a correctness-only "
                  "+1000 offset probe checks finiteness and rows summing to 1 "
                  "outside the timed region."),
    },
    "log_softmax_forward": {
        "category": "probability_transform",
        "operation": "NativeTensor.log_softmax(axis=-1)",
        "reference_type": NUMPY,
        "reference_detail": ("NumPy log-sum-exp ((x - max) - log(sum(exp(x - "
                             "max)))); the stable framework has no direct "
                             "log_softmax, and softmax().log() is deliberately "
                             "not used as the reference because it is the "
                             "composition the fused kernel exists to avoid"),
        "axis": -1,
        "reduction": None,
        "seed": 20250004,
        "shapes": {"full": (128, 64), "smoke": (16, 8)},
        "build": _build_log_softmax,
        "notes": ("Reference is an explicit NumPy formula, so the ratio "
                  "compares a native kernel against vectorized NumPy, not "
                  "against another TensorForge line."),
    },
    "cross_entropy_forward": {
        "category": "loss_forward",
        "operation": 'NativeTensor.cross_entropy(targets, reduction="mean")',
        "reference_type": STABLE,
        "reference_detail": "tensorforge.nn.cross_entropy(logits, targets)",
        "axis": 1,
        "reduction": "mean",
        "seed": 20250005,
        "shapes": {"full": (128, 32), "smoke": (16, 8)},
        "build": _build_cross_entropy_forward,
        "notes": ("Forward loss creation only, with no graph on either side; "
                  "backward is measured separately."),
    },
    "cross_entropy_backward": {
        "category": "loss_backward",
        "operation": "NativeTensor.cross_entropy(...).backward()",
        "reference_type": STABLE,
        "reference_detail": "tensorforge.nn.cross_entropy(...).backward()",
        "axis": 1,
        "reduction": "mean",
        "seed": 20250006,
        "shapes": {"full": (128, 32), "smoke": (16, 8)},
        "build": _build_cross_entropy_backward,
        "notes": ("A fresh graph is built outside the timer for every "
                  "repetition; only backward() is timed, no graph is reused, "
                  "and retain_graph is never used."),
    },
    "classification_training_step": {
        "category": "training_step",
        "operation": ("zero_grad -> Conv2d/ReLU/MaxPool2d/Flatten/Linear "
                      "forward -> NativeCrossEntropyLoss -> backward -> "
                      "NativeAdam.step()"),
        "reference_type": STABLE,
        "reference_detail": ("the same architecture, initial parameter values, "
                             "fused cross-entropy semantics, and Adam "
                             "hyperparameters on the stable line"),
        "axis": 1,
        "reduction": "mean",
        "seed": 20250007,
        "shapes": {"full": (12, 1, 6, 6), "smoke": (12, 1, 6, 6)},
        "build": _build_training_step,
        "repetitions": TRAINING_STEP_REPETITIONS,
        "notes": ("The E8 dataset and classifier. A fresh model and optimizer "
                  "are built outside the timer for every repetition; model, "
                  "optimizer, and dataset construction, checkpoint I/O, "
                  "native_accuracy, reporting conversion, and cleanup are all "
                  "excluded from the timed region."),
    },
}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def measure(prepare, run, cleanup, warmup, repetitions):
    """Return ``repetitions`` per-call seconds samples for ``run``.

    Each repetition builds its own state with ``prepare()`` (untimed),
    times exactly one ``run(state)`` call with
    ``time.perf_counter_ns()``, then releases everything with
    ``cleanup(state, result)`` (untimed). Warm-up repetitions run the
    same way and are discarded before measuring; no measured sample is
    ever dropped and no timer overhead is subtracted. CPU execution is
    synchronous, so no explicit synchronization is needed."""
    for _ in range(warmup):
        state = prepare()
        result = run(state)
        cleanup(state, result)
    samples = []
    for _ in range(repetitions):
        state = prepare()
        start = time.perf_counter_ns()
        result = run(state)
        elapsed = time.perf_counter_ns() - start
        cleanup(state, result)
        samples.append(elapsed / 1e9)
    return samples


def _statistics(samples):
    median = statistics.median(samples)
    low, high = min(samples), max(samples)
    return {
        "sample_count": len(samples),
        "median_s": median,
        "min_s": low,
        "max_s": high,
        "spread_s": high - low,
        "relative_spread": ((high - low) / median) if median > 0 else None,
        "samples_s": list(samples),
        "units": "seconds_per_call",
    }


def _measure_case(name, warmup, repetitions, smoke):
    """Build the case, run its correctness gate, and only then time it."""
    spec = CASES[name]
    shape = spec["shapes"]["smoke" if smoke else "full"]
    case_repetitions = min(repetitions, spec.get("repetitions", repetitions))
    case = spec["build"](shape, spec["seed"])
    try:
        # -- correctness first; a failure raises and publishes no timing --
        metrics = case["check"]()
        native = _statistics(measure(
            case["native_prepare"], case["native_run"], case["native_cleanup"],
            warmup, case_repetitions,
        ))
        reference = None
        if spec["reference_type"] != NATIVE_ONLY:
            reference = _statistics(measure(
                case["reference_prepare"], case["reference_run"],
                case["reference_cleanup"], warmup, case_repetitions,
            ))
    finally:
        case["close"]()

    ratio = None
    if reference is not None and reference["median_s"] > 0:
        ratio = native["median_s"] / reference["median_s"]
    return {
        "case": name,
        "category": spec["category"],
        "operation": spec["operation"],
        "shape": list(shape),
        "axis": spec["axis"],
        "reduction": spec["reduction"],
        "seed": spec["seed"],
        "reference_type": spec["reference_type"],
        "reference_detail": spec["reference_detail"],
        "correctness": dict(status="passed", **metrics),
        "warmup": warmup,
        "sample_count": case_repetitions,
        "native": native,
        "reference": reference,
        "native_to_reference_ratio": ratio,
        "notes": spec["notes"],
    }


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _resolve(requested, allowed, label):
    if requested is None:
        return tuple(allowed)
    selected = tuple(requested)
    for item in selected:
        if item not in allowed:
            raise ValueError(
                f"unknown {label} {item!r}; choose from {tuple(allowed)}"
            )
    return selected


def _environment(warmup, repetitions, smoke):
    info = cpp.backend_info()
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "tensorforge_version": tensorforge.__version__,
        "native_backend": {
            "name": info["name"],
            "tensor_core": info["tensor_core"],
            "available": info["available"],
            "native_autograd": info["native_autograd"],
            "stable_framework_integration": info["stable_framework_integration"],
        },
        "dtype": "float64",
        "device": "cpu",
        "scope": "native classification stack (float64/cpu)",
        "warmup": warmup,
        "repetitions": repetitions,
        "training_step_repetitions": min(repetitions, TRAINING_STEP_REPETITIONS),
        "timer": "time.perf_counter_ns",
        "primary_statistic": "median",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_benchmark(cases=None, warmup=DEFAULTS["warmup"],
                  repetitions=DEFAULTS["repetitions"], smoke=False):
    """Run the selected cases and return the JSON-ready payload.

    Every case's correctness gate runs **before** its timing; a failed
    gate raises (the CLI turns that into a nonzero exit) and no timing is
    published for it. No timing threshold is applied anywhere — this only
    measures. Raises RuntimeError if the native backend is not built."""
    if not cpp.is_available():
        raise RuntimeError(
            "The experimental C++ backend is not built.\n"
            + cpp.build_instructions()
        )
    selected = _resolve(cases, tuple(CASES), "case")
    warmup = _positive_int(warmup, "warmup")
    repetitions = _positive_int(repetitions, "repetitions")
    return {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "mode": "smoke" if smoke else "full",
        "environment": _environment(warmup, repetitions, smoke),
        "cases": [_measure_case(name, warmup, repetitions, smoke)
                  for name in selected],
    }


def _format_duration(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


DISCLAIMER = (
    "Local characterization only -- not a performance contract. Times come "
    "from one\nmachine, one build, and one workload; they are not "
    "cross-machine comparable\nwithout controlled conditions. The observed "
    "ratio describes what was measured\nhere, not a guarantee, and no test or "
    "CI job asserts any timing threshold.\nCorrectness is gated before every "
    "measurement; timing is never a pass/fail\ncriterion."
)


def format_report(payload):
    """A concise human-readable report. Carries no speed verdict."""
    env = payload["environment"]
    lines = [
        f"TensorForge native classification benchmark v{payload['version']} "
        f"[{payload['mode']}]",
        f"  platform  : {env['platform']}",
        f"  machine   : {env['machine']}",
        f"  processor : {env['processor']}",
        f"  python    : {env['python_version']}   "
        f"tensorforge {env['tensorforge_version']}",
        f"  backend   : {env['native_backend']['tensor_core']} "
        f"({env['dtype']}/{env['device']}, available="
        f"{env['native_backend']['available']})",
        f"  timer     : {env['timer']}   primary statistic: "
        f"{env['primary_statistic']}",
        f"  warmup/repetitions : {env['warmup']}/{env['repetitions']} "
        f"(training step: {env['training_step_repetitions']})",
        "",
    ]
    header = (
        f"{'case':<30} {'shape':<14} {'native median':>14} "
        f"{'reference':>14} {'ratio':>8} {'spread':>10} {'reference type':<20} "
        f"{'correct':<8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for record in payload["cases"]:
        native = record["native"]
        reference = record["reference"]
        reference_median = (_format_duration(reference["median_s"])
                            if reference else "n/a")
        ratio = record["native_to_reference_ratio"]
        ratio_text = f"{ratio:.2f}x" if ratio is not None else "n/a"
        lines.append(
            f"{record['case']:<30} "
            f"{'x'.join(str(d) for d in record['shape']):<14} "
            f"{_format_duration(native['median_s']):>14} "
            f"{reference_median:>14} {ratio_text:>8} "
            f"{_format_duration(native['spread_s']):>10} "
            f"{record['reference_type']:<20} "
            f"{record['correctness']['status']:<8}"
        )
    lines.append("")
    lines.append(
        "ratio = native median / reference median (>1 means the native path "
        "took longer here)."
    )
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description=("Characterize the native classification stack "
                     "(measurement only; no speed is asserted).")
    )
    parser.add_argument("--case", choices=tuple(CASES), default=None,
                        help="run a single case (default: all)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="warm-up repetitions before measuring")
    parser.add_argument("--repetitions", type=int, default=None,
                        help="measured repetitions per case")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON only")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny shapes and counts, for tests/CI")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    defaults = SMOKE_DEFAULTS if args.smoke else DEFAULTS
    warmup = args.warmup if args.warmup is not None else defaults["warmup"]
    repetitions = (args.repetitions if args.repetitions is not None
                   else defaults["repetitions"])
    try:
        payload = run_benchmark(
            cases=[args.case] if args.case else None,
            warmup=warmup, repetitions=repetitions, smoke=args.smoke,
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))     # stderr, exit 2 — stdout stays clean
    except AssertionError as error:  # a correctness gate failed
        parser.exit(1, f"correctness gate failed: {error}\n")
    if args.json:
        print(json.dumps(payload))
    else:
        print(format_report(payload))


if __name__ == "__main__":
    main()
