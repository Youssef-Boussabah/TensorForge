"""Characterization benchmark for the native normalization stack
(Advanced C++ Phase F, milestone F7).

This measures the normalization surface F2-F6 already shipped —
``NativeLayerNorm``, ``NativeBatchNorm1d``, ``NativeBatchNorm2d``, their
training and evaluation forwards, their composed backwards, and one
complete F6-style normalized training step. It does **not** try to make
anything take less time, and **nothing here asserts a speed**: every
normalization module is a *composition* of existing native operations
(there is no normalization kernel, C ABI symbol, or fused path anywhere),
so this is an honest, reproducible snapshot of one machine at one moment.
It is **not** a performance contract, not comparable across machines
without controlled conditions, and there is no CI timing threshold
anywhere in this repository.

**Correctness runs before timing, always.** Every case validates its
native result against a reference — the stable NumPy framework where an
equivalent module exists, an explicit NumPy formula where it does not —
and a failed gate aborts that run with a nonzero exit status and
publishes no timing for the case.

Each case is labelled with the reference it actually used:

- ``stable_tensorforge`` — a real ``tensorforge.nn`` (stable-line) module
  on identical inputs, epsilon, momentum, affine values, and running
  state;
- ``native_only`` — no structurally equivalent public stable module
  exists, so **no timing ratio is reported**. This applies to all three
  ``NativeBatchNorm2d`` cases: the stable line has no public
  ``BatchNorm2d``. Their correctness gates are still real — an explicit
  NumPy NCHW population-statistics formula for the forwards and the
  running state, and, for the backward, the stable ``BatchNorm1d``
  applied to the equivalent ``(N*H*W, C)`` sample matrix with the input
  gradient transformed back to NCHW. That transformed computation is a
  correctness **oracle only**: timing it as a "BatchNorm2d reference"
  would compare against a different module plus two layout
  transformations, so the ratio would be misleading and is deliberately
  not published.

The measured times include the Python wrapper, the composed autograd
graph construction, and the ctypes boundary as well as the native
compute: that is what a caller actually pays.

**What is timed.** One measured repetition times exactly one call of the
case's operation with ``time.perf_counter_ns()``. Setup — input creation,
module construction, state loading, graph construction for the
backward-only cases, model/optimizer construction for the training step —
happens **outside** the timer, and cleanup happens outside it too. Graph
construction *is* inside the timer for the forward and training-step
cases, because it is part of the call being characterized. No sample is
discarded, no timer overhead is subtracted, and the native and reference
paths run under the same setup discipline. The median is the primary
statistic; the minimum, maximum, and their spread show the variation.

Because a BatchNorm training forward advances the persistent
``running_mean``/``running_var`` buffers, every measured repetition of a
training-mode case builds a **fresh** module from the same deterministic
state — a state-advanced module is never reused as a measured sample.

Build the backend first:

    uv run python cpp/build.py

Then, for example:

    uv run python benchmarks/benchmark_native_normalization.py
    uv run python benchmarks/benchmark_native_normalization.py --smoke
    uv run python benchmarks/benchmark_native_normalization.py --smoke --json
    uv run python benchmarks/benchmark_native_normalization.py --case batchnorm2d_backward
    uv run python benchmarks/benchmark_native_normalization.py --case normalized_training_step

F7 adds **no** numerical capability: no kernel, C ABI export, ctypes
symbol, Core method, autograd operation, module, loss, metric, optimizer,
export, state-support entry, or checkpoint change. It only measures what
F2-F6 already shipped, and the F6 proof
(``examples/native_normalization_training.py``) remains the separate,
authoritative correctness-and-resume artifact. No result file is written.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeBatchNorm2d,
    NativeLayerNorm,
    NativeLinear,
    NativeModule,
    NativeMSELoss,
    NativeReLU,
    NativeTensor,
)

BENCHMARK_NAME = "tensorforge.native_normalization"
BENCHMARK_VERSION = "1.0"

# Reference labels. Every case declares exactly one.
STABLE = "stable_tensorforge"
NATIVE_ONLY = "native_only"

# Warm-up / repetition defaults. The backward and training-step cases are
# heavier than the forwards, so each declares its own smaller repetition
# cap; the count actually used is always reported per case.
DEFAULTS = {"warmup": 3, "repetitions": 12}
SMOKE_DEFAULTS = {"warmup": 1, "repetitions": 3}
TRAINING_STEP_REPETITIONS = 6
BACKWARD_REPETITIONS = 8

# The normalization configuration every case shares. These are module
# arguments, not measurement parameters.
EPS = 1e-5
MOMENTUM = 0.1

# Absolute tolerances for the correctness gates. These are ordinary
# float64 agreement bounds taken from the existing parity suites, not
# performance criteria: nothing here bounds a duration, a ratio, or a
# throughput.
FORWARD_ATOL = 1e-12
GRADIENT_ATOL = 1e-10
STATE_ATOL = 1e-12
LOSS_ATOL = 1e-12
# Looser than the others, for one honest and specific reason. The
# training step's ``hidden.bias`` sits immediately before BatchNorm, whose
# mean subtraction cancels any constant per-feature shift, so its gradient
# is *mathematically zero* and both lines compute only float64 round-off
# (~1e-16, and they agree to that). Adam's first step then divides by
# ``sqrt(v_hat) + eps``, which for such a gradient amplifies by up to
# ``lr / eps = 0.05 / 1e-8``, turning a 1e-16 round-off difference into a
# ~1e-9 parameter difference. This bound therefore covers amplified
# round-off on a structurally dead parameter, never a modelling
# difference: the gradients themselves are still gated at GRADIENT_ATOL,
# and the pre-update loss at LOSS_ATOL.
PARAMETER_ATOL = 1e-8


# ---------------------------------------------------------------------------
# Deterministic host inputs. Every generator uses a local seeded NumPy
# generator; the global NumPy RNG is never touched.
# ---------------------------------------------------------------------------


def _rng(seed):
    """A local generator. The global NumPy RNG is never touched."""
    return np.random.default_rng(seed)


def _activations(shape, seed):
    """Finite, moderate float64 activations — no overflow, no underflow,
    and no zero-variance feature, so normalization is measured on
    representative data. Numerical edge cases belong to the correctness
    suites and are deliberately excluded from the timed data."""
    return _rng(seed).uniform(-2.0, 2.0, size=shape)


def _scale_values(shape, seed):
    """Nontrivial affine scales, bounded away from zero so ``gamma`` /
    ``weight`` never degenerates to the identity."""
    return _rng(seed).uniform(0.5, 1.5, size=shape)


def _shift_values(shape, seed):
    """Nontrivial affine shifts, so ``beta`` / ``bias`` is never zero."""
    return _rng(seed).uniform(-0.5, 0.5, size=shape)


def _running_mean_values(shape, seed):
    """Nontrivial stored running means — never the zeros a fresh module
    starts from."""
    return _rng(seed).uniform(-1.0, 1.0, size=shape)


def _running_var_values(shape, seed):
    """Nontrivial **positive** stored running variances — never the ones
    a fresh module starts from, and never near zero."""
    return _rng(seed).uniform(0.5, 2.0, size=shape)


def _upstream_values(shape, seed):
    """The deterministic upstream weights of the scalar backward
    objective. A plain ``sum()`` would seed every element with 1.0, which
    a broadcast or reduction mistake can survive; weighting first makes
    the seeded gradient distinct per element."""
    return _rng(seed).uniform(-1.0, 1.0, size=shape)


# ---------------------------------------------------------------------------
# Explicit NumPy references. These are formulas, never the implementation
# under test, and they never run inside a timed region.
# ---------------------------------------------------------------------------


def _numpy_layer_norm(values, normalized_shape, eps, weight, bias):
    """Per-sample normalization over the trailing dimensions, population
    variance, ``sqrt(var + eps)`` ordering."""
    k = len(normalized_shape)
    axes = tuple(range(values.ndim - k, values.ndim))
    mean = values.mean(axis=axes, keepdims=True)
    centered = values - mean
    variance = (centered * centered).mean(axis=axes, keepdims=True)
    return centered / np.sqrt(variance + eps) * weight + bias


def _numpy_batch_statistics(values, axes):
    """The population mean and variance over ``axes`` (no Bessel
    correction), keeping the reduced dimensions."""
    mean = values.mean(axis=axes, keepdims=True)
    centered = values - mean
    return mean, (centered * centered).mean(axis=axes, keepdims=True)


def _numpy_batch_norm_train(values, axes, stat_shape, eps, gamma, beta):
    """Training-mode batch normalization plus the ``(C,)`` batch
    statistics that drove it. For NCHW this reduces over N, H, and W, so
    each channel gets one mean and one variance over ``N * H * W``
    values."""
    mean, variance = _numpy_batch_statistics(values, axes)
    normalized = (values - mean) / np.sqrt(variance + eps)
    output = (normalized * gamma.reshape(stat_shape)
              + beta.reshape(stat_shape))
    return output, mean.reshape(-1), variance.reshape(-1)


def _numpy_batch_norm_eval(values, stat_shape, eps, gamma, beta,
                           running_mean, running_var):
    """Evaluation-mode batch normalization from the stored statistics."""
    normalized = ((values - running_mean.reshape(stat_shape))
                  / np.sqrt(running_var.reshape(stat_shape) + eps))
    return normalized * gamma.reshape(stat_shape) + beta.reshape(stat_shape)


def _numpy_running_update(running, statistic, momentum):
    """``(1 - momentum) * running + momentum * statistic``."""
    return (1.0 - momentum) * running + momentum * statistic


def _nchw_to_samples(values):
    """``(N, C, H, W)`` to the equivalent ``(N*H*W, C)`` sample matrix.

    Reducing that matrix over axis 0 is exactly the NCHW reduction over
    N, H, and W, so a stable ``BatchNorm1d`` applied here is a rigorous
    oracle for ``NativeBatchNorm2d``. It is used **only** for correctness
    — see the module docstring for why it is never timed."""
    return np.transpose(values, (0, 2, 3, 1)).reshape(-1, values.shape[1])


def _samples_to_nchw(matrix, shape):
    """The inverse of :func:`_nchw_to_samples`."""
    n, c, h, w = shape
    return np.transpose(matrix.reshape(n, h, w, c), (0, 3, 1, 2))


# ---------------------------------------------------------------------------
# Correctness-gate helpers. Every one of these raises AssertionError, which
# the CLI turns into a nonzero exit with no timing published.
# ---------------------------------------------------------------------------


def _max_abs(values):
    values = np.asarray(values)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _require_finite(values, label):
    if not np.all(np.isfinite(values)):
        raise AssertionError(f"{label} is not finite")


def _require_shape(values, expected, label):
    produced = tuple(np.shape(values))
    if produced != tuple(expected):
        raise AssertionError(
            f"{label} has shape {produced}, expected {tuple(expected)}"
        )


def _require_parity(error, tolerance, label, reference):
    """The single parity gate. ``tolerance`` is always a float64
    agreement bound; no caller passes a duration."""
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(
            f"{label} differs from the reference ({reference}) by "
            f"{error:g} (> {tolerance:g})"
        )


def _require_unchanged(produced, expected, label):
    if not np.array_equal(np.asarray(produced), np.asarray(expected)):
        raise AssertionError(f"{label} was mutated")


# ---------------------------------------------------------------------------
# Native / stable state helpers
# ---------------------------------------------------------------------------


def _install_state(module, **arrays):
    """Install nontrivial values through the public atomic loader, which
    preserves every parameter and buffer identity. Untimed setup."""
    values = {name: NativeTensor.from_array(np.asarray(value,
                                                       dtype=np.float64))
              for name, value in arrays.items()}
    try:
        module.load_state_dict(values, strict=False)
    finally:
        for tensor in values.values():
            tensor.close()


def _close_module(module):
    """There is no ``NativeModule.close()``, so a stateful module's owner
    releases **both** its parameters and its buffers explicitly (§9)."""
    for parameter in module.parameters():
        parameter.close()
    for buffer in module.buffers():
        buffer.close()


def _release_gradients(tensors):
    """Drop and close every gradient a backward left behind, so a
    repetition's cleanup is deterministic rather than GC-dependent."""
    for tensor in tensors:
        if tensor.closed:
            continue
        gradient = tensor.grad
        if gradient is not None:
            tensor.zero_grad()
            gradient.close()


def _walk_graph(root):
    """Every autograd node reachable from ``root`` plus every native
    resource a node's history owns. Used only by the correctness gates,
    never inside a timed region."""
    nodes, resources, seen = [], [], set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        nodes.append(node)
        resources.extend(node._graph_resources)
        stack.extend(node._parents)
    return nodes, resources


def _graph_storage_ids(root):
    """The ids of every native **storage** the graph can reach.

    Stronger than an object-identity walk: a borrowing view of a
    registered buffer is a different Python object but the same storage,
    and the §7 snapshot rule forbids that just as firmly."""
    ids = set()
    nodes, resources = _walk_graph(root)
    for tensor in nodes + resources:
        if isinstance(tensor, NativeTensor) and not tensor.closed:
            ids.add(id(tensor._core.storage))
    return ids


def _stat_shape(shape, channels):
    """The per-channel broadcast layout: ``(1, C)`` for ``(N, C)`` and
    ``(1, C, 1, 1)`` for NCHW."""
    return (1, channels) + (1,) * (len(shape) - 2)


def _stable_layer_norm(normalized_shape, eps, weight_values, bias_values):
    """The stable-line LayerNorm holding the *same* normalized shape,
    epsilon, and affine values."""
    from tensorforge.nn import LayerNorm

    module = LayerNorm(normalized_shape, eps=eps)
    module.weight.data = np.array(weight_values, dtype=np.float64)
    module.bias.data = np.array(bias_values, dtype=np.float64)
    return module


def _stable_batch_norm(channels, eps, momentum, gamma_values, beta_values,
                       running_mean_values, running_var_values):
    """The stable-line BatchNorm1d holding the *same* epsilon, momentum,
    affine values, and running state."""
    from tensorforge.nn import BatchNorm1d

    module = BatchNorm1d(channels, eps=eps, momentum=momentum)
    module.gamma.data = np.array(gamma_values, dtype=np.float64)
    module.beta.data = np.array(beta_values, dtype=np.float64)
    module.running_mean = np.array(running_mean_values, dtype=np.float64)
    module.running_var = np.array(running_var_values, dtype=np.float64)
    return module


# ---------------------------------------------------------------------------
# Case builders. Each returns a dict of untimed ``prepare``/``cleanup``
# callables around one timed ``run``, plus a ``check`` that runs the whole
# correctness gate before any timing, and a ``close`` that releases the
# case's shared persistent state after every repetition is done.
# ---------------------------------------------------------------------------


def _build_layernorm_forward(config, spec):
    """One ``NativeLayerNorm`` forward, graph construction included.

    The affine parameters require gradients, so the timed call really does
    build the composed graph a caller would get — that is part of what a
    forward costs here."""
    shape = config["shape"]
    normalized_shape = config["normalized_shape"]
    eps, seed = spec["eps"], spec["seed"]
    values = _activations(shape, seed)
    weight_values = _scale_values(normalized_shape, seed + 1)
    bias_values = _shift_values(normalized_shape, seed + 2)

    native_input = NativeTensor.from_array(values)
    module = NativeLayerNorm(normalized_shape, eps=eps)
    _install_state(module, weight=weight_values, bias=bias_values)
    stable_input = tensorforge.Tensor(values.copy())
    stable_module = _stable_layer_norm(normalized_shape, eps, weight_values,
                                       bias_values)

    def native_run(_state=None):
        return module(native_input)

    def reference_run(_state=None):
        return stable_module(stable_input)

    def check():
        expected = _numpy_layer_norm(values, normalized_shape, eps,
                                     weight_values, bias_values)
        versions = (module.weight.version, module.bias.version)
        output = native_run()
        try:
            produced = output.to_numpy().copy()
            if not output.owns_core or not output.contiguous:
                raise AssertionError("the native output is not owning and "
                                     "contiguous")
        finally:
            output.close()
        _require_shape(produced, shape, "the native output")
        _require_finite(produced, "the native output")
        native_error = _max_abs(produced - expected)
        _require_parity(native_error, FORWARD_ATOL, "the native output",
                        "the NumPy population-variance formula")

        stable_output = reference_run().data
        reference_error = _max_abs(stable_output - expected)
        _require_parity(reference_error, FORWARD_ATOL,
                        "the stable LayerNorm output", "the NumPy formula")
        parity_error = _max_abs(produced - stable_output)
        _require_parity(parity_error, FORWARD_ATOL, "the native output",
                        "tensorforge.nn.LayerNorm")

        # Mode independence, outside every timed region: LayerNorm never
        # reads ``training``, so the same input must give the same output.
        module.eval()
        try:
            evaluated = module(native_input)
            try:
                eval_values = evaluated.to_numpy().copy()
            finally:
                evaluated.close()
        finally:
            module.train()
        if not np.array_equal(eval_values, produced):
            raise AssertionError("the LayerNorm output is not mode-independent")

        _require_unchanged(native_input.to_numpy(), values, "the native input")
        _require_unchanged(stable_input.data, values, "the stable input")
        _require_unchanged(module.weight.to_numpy(), weight_values,
                           "the native weight")
        _require_unchanged(module.bias.to_numpy(), bias_values,
                           "the native bias")
        if (module.weight.version, module.bias.version) != versions:
            raise AssertionError("a parameter version moved during the forward")
        if list(module.buffers()):
            raise AssertionError("NativeLayerNorm registered a buffer")
        return {
            "max_abs_error": native_error,
            "reference_max_abs_error": reference_error,
            "native_vs_stable_max_abs_error": parity_error,
            "checks": ["output_shape", "finite", "owning_contiguous_output",
                       "numpy_formula_parity", "stable_parity",
                       "mode_independent", "no_input_mutation",
                       "no_parameter_mutation", "stateless"],
        }

    return {
        "native_prepare": lambda: None,
        "native_run": native_run,
        "native_cleanup": lambda _state, result: result.close(),
        "reference_prepare": lambda: None,
        "reference_run": reference_run,
        "reference_cleanup": lambda _state, _result: None,
        "check": check,
        "close": lambda: (native_input.close(), _close_module(module)),
    }


def _build_layernorm_backward(config, spec):
    """One one-shot ``backward()`` through ``NativeLayerNorm``.

    Every repetition builds a fresh forward graph **outside** the timer
    from cleared gradients, times exactly one ``backward()``, and releases
    the graph afterwards. No graph is reused and ``retain_graph`` is never
    used to skip the rebuild."""
    shape = config["shape"]
    normalized_shape = config["normalized_shape"]
    eps, seed = spec["eps"], spec["seed"]
    values = _activations(shape, seed)
    weight_values = _scale_values(normalized_shape, seed + 1)
    bias_values = _shift_values(normalized_shape, seed + 2)
    upstream = _upstream_values(shape, seed + 3)

    module = NativeLayerNorm(normalized_shape, eps=eps)
    _install_state(module, weight=weight_values, bias=bias_values)
    native_upstream = NativeTensor.from_array(upstream)

    def native_prepare():
        module.zero_grad()
        native_input = NativeTensor.from_array(values, requires_grad=True)
        output = module(native_input)
        weighted = output.multiply(native_upstream)
        return native_input, output, weighted, weighted.sum()

    def native_run(state):
        state[3].backward()
        return state[3]

    def native_cleanup(state, _result):
        native_input, output, weighted, objective = state
        _release_gradients([native_input, module.weight, module.bias])
        objective.close()
        weighted.close()
        output.close()
        native_input.close()

    def reference_prepare():
        stable_module = _stable_layer_norm(normalized_shape, eps,
                                           weight_values, bias_values)
        x = tensorforge.Tensor(values.copy(), requires_grad=True)
        weighted = stable_module(x) * tensorforge.Tensor(upstream.copy())
        return stable_module, x, weighted.sum()

    def reference_run(state):
        state[2].backward()
        return state[2]

    def check():
        state = native_prepare()
        native_input, output, _weighted, objective = state
        try:
            native_run(state)
            for label, tensor in (("input", native_input),
                                  ("weight", module.weight),
                                  ("bias", module.bias)):
                if tensor.grad is None:
                    raise AssertionError(
                        f"the native backward produced no {label} gradient"
                    )
            input_grad = native_input.grad.to_numpy().copy()
            weight_grad = module.weight.grad.to_numpy().copy()
            bias_grad = module.bias.grad.to_numpy().copy()
            _require_shape(input_grad, shape, "the native input gradient")
            _require_shape(weight_grad, normalized_shape,
                           "the native weight gradient")
            _require_shape(bias_grad, normalized_shape,
                           "the native bias gradient")
            for label, gradient in (("input", input_grad),
                                    ("weight", weight_grad),
                                    ("bias", bias_grad)):
                _require_finite(gradient, f"the native {label} gradient")
            if not objective._graph_freed:
                raise AssertionError(
                    "the one-shot backward did not release the graph"
                )
            if objective._graph_resources or output._graph_resources:
                raise AssertionError(
                    "a graph resource survived the one-shot backward"
                )
            _require_unchanged(native_input.to_numpy(), values,
                               "the native input")
        finally:
            native_cleanup(state, None)

        reference_state = reference_prepare()
        stable_module, stable_input, _ = reference_state
        reference_run(reference_state)
        input_error = _max_abs(input_grad - stable_input.grad)
        weight_error = _max_abs(weight_grad - stable_module.weight.grad)
        bias_error = _max_abs(bias_grad - stable_module.bias.grad)
        error = max(input_error, weight_error, bias_error)
        _require_parity(error, GRADIENT_ATOL, "the native gradients",
                        "tensorforge.nn.LayerNorm's gradients")

        _require_unchanged(module.weight.to_numpy(), weight_values,
                           "the native weight")
        _require_unchanged(module.bias.to_numpy(), bias_values,
                           "the native bias")
        if list(module.buffers()):
            raise AssertionError("NativeLayerNorm registered a buffer")
        return {
            "max_abs_error": error,
            "input_gradient_max_abs_error": input_error,
            "weight_gradient_max_abs_error": weight_error,
            "bias_gradient_max_abs_error": bias_error,
            "checks": ["input_gradient_present", "affine_gradients_present",
                       "gradient_shapes", "finite", "stable_parity",
                       "graph_released", "no_graph_resource_survives",
                       "no_state_mutation"],
        }

    return {
        "native_prepare": native_prepare,
        "native_run": native_run,
        "native_cleanup": native_cleanup,
        "reference_prepare": reference_prepare,
        "reference_run": reference_run,
        "reference_cleanup": lambda _state, _result: None,
        "check": check,
        "close": lambda: (native_upstream.close(), _close_module(module)),
    }


def _batchnorm_values(config, spec):
    """The deterministic activations, affine values, and running state
    every BatchNorm case shares, plus the derived shape configuration."""
    shape = config["shape"]
    channels = shape[1]
    seed = spec["seed"]
    return {
        "shape": shape,
        "channels": channels,
        "axes": tuple(spec["reduction_axes"]),
        "stat_shape": _stat_shape(shape, channels),
        "values": _activations(shape, seed),
        "gamma": _scale_values((channels,), seed + 1),
        "beta": _shift_values((channels,), seed + 2),
        "running_mean": _running_mean_values((channels,), seed + 3),
        "running_var": _running_var_values((channels,), seed + 4),
        "upstream": _upstream_values(shape, seed + 5),
    }


def _native_batch_norm(data, spec, training):
    """A fresh native BatchNorm module carrying the case's exact affine
    values and running state, in the requested mode."""
    module = spec["module"](data["channels"], eps=spec["eps"],
                            momentum=spec["momentum"])
    _install_state(module, gamma=data["gamma"], beta=data["beta"],
                       running_mean=data["running_mean"],
                       running_var=data["running_var"])
    module.train(training)
    return module


def _stable_batch_norm_for(data, spec, training):
    module = _stable_batch_norm(data["channels"], spec["eps"],
                                spec["momentum"], data["gamma"], data["beta"],
                                data["running_mean"], data["running_var"])
    module.train(training)
    return module


def _build_batchnorm_training_forward(config, spec):
    """One training-mode BatchNorm forward: the differentiable batch
    statistics, the affine application, the graph-free running-statistics
    preparation, and the atomic two-buffer commit — all inside the timed
    call, because all of them are part of one training forward.

    A training forward advances persistent state, so a **fresh** module is
    built outside the timer for every warm-up and measured repetition; a
    state-advanced module is never reused as a sample."""
    data = _batchnorm_values(config, spec)
    shape, stat_shape = data["shape"], data["stat_shape"]
    native_input = NativeTensor.from_array(data["values"])
    stable_input = (tensorforge.Tensor(data["values"].copy())
                    if spec["reference_type"] == STABLE else None)

    def native_prepare():
        return _native_batch_norm(data, spec, True)

    def native_run(module):
        return module(native_input)

    def native_cleanup(module, result):
        if result is not None:
            result.close()
        _close_module(module)

    def reference_prepare():
        return _stable_batch_norm_for(data, spec, True)

    def reference_run(module):
        return module(stable_input)

    def check():
        expected, batch_mean, batch_var = _numpy_batch_norm_train(
            data["values"], data["axes"], stat_shape, spec["eps"],
            data["gamma"], data["beta"],
        )
        expected_mean = _numpy_running_update(data["running_mean"], batch_mean,
                                              spec["momentum"])
        expected_var = _numpy_running_update(data["running_var"], batch_var,
                                             spec["momentum"])
        module = native_prepare()
        identities = (id(module.gamma), id(module.beta),
                      id(module.running_mean), id(module.running_var))
        versions = (module.gamma.version, module.beta.version)
        output = None
        try:
            output = native_run(module)
            produced = output.to_numpy().copy()
            if not output.owns_core or not output.contiguous:
                raise AssertionError("the native output is not owning and "
                                     "contiguous")
            _require_shape(produced, shape, "the native output")
            _require_finite(produced, "the native output")
            native_error = _max_abs(produced - expected)
            _require_parity(native_error, FORWARD_ATOL,
                            "the native training-mode output",
                            "the explicit NumPy population-statistics formula")

            mean_after = module.running_mean.to_numpy().copy()
            var_after = module.running_var.to_numpy().copy()
            _require_finite(mean_after, "the advanced running_mean")
            _require_finite(var_after, "the advanced running_var")
            state_error = max(_max_abs(mean_after - expected_mean),
                              _max_abs(var_after - expected_var))
            _require_parity(state_error, STATE_ATOL,
                            "the advanced running statistics",
                            "the explicit NumPy momentum formula")
            if np.array_equal(mean_after, data["running_mean"]):
                raise AssertionError("running_mean did not advance")
            if np.array_equal(var_after, data["running_var"]):
                raise AssertionError("running_var did not advance")
            if (module.gamma.version, module.beta.version) != versions:
                raise AssertionError(
                    "a parameter version moved during the forward"
                )
            if (id(module.gamma), id(module.beta), id(module.running_mean),
                    id(module.running_var)) != identities:
                raise AssertionError(
                    "a parameter or buffer identity changed during the forward"
                )
            _require_unchanged(native_input.to_numpy(), data["values"],
                               "the native input")
            checks = ["output_shape", "finite", "owning_contiguous_output",
                      "population_statistics_parity", "running_mean_parity",
                      "running_var_parity", "both_buffers_advanced",
                      "parameter_versions_unchanged",
                      "identities_preserved", "no_input_mutation"]
            metrics = {
                "max_abs_error": native_error,
                "running_state_max_abs_error": state_error,
            }
            if spec["channelwise_affine_probe"]:
                metrics["channelwise_affine_max_abs_error"] = (
                    _channelwise_affine_error(data, spec)
                )
                checks.append("channelwise_affine")
        finally:
            native_cleanup(module, output)

        if spec["reference_type"] == STABLE:
            stable_module = reference_prepare()
            stable_output = reference_run(stable_module).data
            reference_error = _max_abs(stable_output - expected)
            _require_parity(reference_error, FORWARD_ATOL,
                            "the stable BatchNorm1d output",
                            "the NumPy formula")
            parity_error = _max_abs(produced - stable_output)
            _require_parity(parity_error, FORWARD_ATOL, "the native output",
                            "tensorforge.nn.BatchNorm1d")
            stable_state_error = max(
                _max_abs(stable_module.running_mean - mean_after),
                _max_abs(stable_module.running_var - var_after),
            )
            _require_parity(stable_state_error, STATE_ATOL,
                            "the native running statistics",
                            "tensorforge.nn.BatchNorm1d's running statistics")
            _require_unchanged(stable_input.data, data["values"],
                               "the stable input")
            metrics["reference_max_abs_error"] = reference_error
            metrics["native_vs_stable_max_abs_error"] = parity_error
            checks.append("stable_parity")
        metrics["checks"] = checks
        return metrics

    def close():
        native_input.close()

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


def _channelwise_affine_error(data, spec):
    """Independent evidence that ``gamma``/``beta`` are applied **per
    channel** rather than per spatial position.

    Normalizing once with the identity affine and once with the real
    affine must differ by exactly ``normalized * gamma + beta`` broadcast
    from the channel axis. A ``(N, C, H, W) * (C,)`` mistake would line
    the parameters up with W instead, which this catches even when
    ``H != W != C`` keeps the shapes legal. Correctness only; never
    timed."""
    identity = spec["module"](data["channels"], eps=spec["eps"],
                              momentum=spec["momentum"])
    affine = _native_batch_norm(data, spec, True)
    input_tensor = NativeTensor.from_array(data["values"])
    try:
        _install_state(identity, running_mean=data["running_mean"],
                           running_var=data["running_var"])
        identity.train()
        plain_output = identity(input_tensor)
        try:
            normalized = plain_output.to_numpy().copy()
        finally:
            plain_output.close()
        affine_output = affine(input_tensor)
        try:
            produced = affine_output.to_numpy().copy()
        finally:
            affine_output.close()
    finally:
        input_tensor.close()
        _close_module(identity)
        _close_module(affine)
    expected = (normalized * data["gamma"].reshape(data["stat_shape"])
                + data["beta"].reshape(data["stat_shape"]))
    error = _max_abs(produced - expected)
    _require_parity(error, FORWARD_ATOL, "the channelwise affine result",
                    "normalized * gamma + beta broadcast from the channel axis")
    return error


def _build_batchnorm_eval_forward(config, spec):
    """One evaluation-mode BatchNorm forward.

    Evaluation reads the stored statistics through independent graph-free
    snapshots and mutates nothing, so one module is shared across every
    repetition — there is no state to advance."""
    data = _batchnorm_values(config, spec)
    shape, stat_shape = data["shape"], data["stat_shape"]
    native_input = NativeTensor.from_array(data["values"])
    module = _native_batch_norm(data, spec, False)
    stable_module = (_stable_batch_norm_for(data, spec, False)
                     if spec["reference_type"] == STABLE else None)
    stable_input = (tensorforge.Tensor(data["values"].copy())
                    if spec["reference_type"] == STABLE else None)

    def native_run(_state=None):
        return module(native_input)

    def reference_run(_state=None):
        return stable_module(stable_input)

    def check():
        expected = _numpy_batch_norm_eval(
            data["values"], stat_shape, spec["eps"], data["gamma"],
            data["beta"], data["running_mean"], data["running_var"],
        )
        buffer_storages = {id(module.running_mean._core.storage),
                           id(module.running_var._core.storage)}
        output = native_run()
        try:
            produced = output.to_numpy().copy()
            if not output.owns_core or not output.contiguous:
                raise AssertionError("the native output is not owning and "
                                     "contiguous")
            # The §7 snapshot rule, checked structurally and outside every
            # timed region: the eval graph must hold independent snapshots,
            # never the live registered buffers (nor a view onto their
            # storage).
            reachable = _graph_storage_ids(output)
            if reachable & buffer_storages:
                raise AssertionError(
                    "the evaluation graph reaches a registered running buffer"
                )
            # The graph's adopted resources are pinned **exactly**, by
            # count and by shape, not merely spot-checked: the two
            # stat-shaped running-statistic snapshots, plus — for the NCHW
            # layout only — the one activation-shaped tensor whose storage
            # the channels-last affine operand borrows. Here the input does
            # not require gradients and the affine parameters do, which is
            # precisely the configuration in which the transposed operand
            # is a plain borrowing leaf and the graph must own its source.
            _nodes, resources = _walk_graph(output)
            for resource in resources:
                if not resource.owns_core:
                    raise AssertionError(
                        "an adopted evaluation resource does not own its "
                        "storage"
                    )
            snapshots = [r for r in resources
                         if tuple(r.shape) == stat_shape]
            others = [r for r in resources if tuple(r.shape) != stat_shape]
            if len(snapshots) != 2:
                raise AssertionError(
                    f"the evaluation graph adopted {len(snapshots)} "
                    f"running-statistic snapshots, expected exactly 2"
                )
            expected_sources = 1 if len(stat_shape) == 4 else 0
            if len(others) != expected_sources:
                raise AssertionError(
                    f"the evaluation graph adopted {len(others)} non-snapshot "
                    f"resources, expected exactly {expected_sources}"
                )
            for resource in others:
                if tuple(resource.shape) != shape:
                    raise AssertionError(
                        f"the adopted channels-last affine source has shape "
                        f"{tuple(resource.shape)}, expected {shape}"
                    )
            snapshot_count = len(snapshots)
            affine_source_count = len(others)
        finally:
            output.close()
        _require_shape(produced, shape, "the native output")
        _require_finite(produced, "the native output")
        native_error = _max_abs(produced - expected)
        _require_parity(native_error, FORWARD_ATOL,
                        "the native evaluation-mode output",
                        "the explicit NumPy running-statistics formula")

        _require_unchanged(module.running_mean.to_numpy(),
                           data["running_mean"], "running_mean")
        _require_unchanged(module.running_var.to_numpy(), data["running_var"],
                           "running_var")
        _require_unchanged(native_input.to_numpy(), data["values"],
                           "the native input")
        checks = ["output_shape", "finite", "owning_contiguous_output",
                  "numpy_formula_parity", "no_running_state_mutation",
                  "graph_holds_snapshots_not_buffers",
                  "snapshots_own_their_storage",
                  "adopted_resource_inventory_is_exact", "no_input_mutation"]
        metrics = {
            "max_abs_error": native_error,
            "adopted_snapshot_count": snapshot_count,
            "adopted_affine_source_count": affine_source_count,
        }
        if spec["reference_type"] == STABLE:
            stable_output = reference_run().data
            reference_error = _max_abs(stable_output - expected)
            _require_parity(reference_error, FORWARD_ATOL,
                            "the stable BatchNorm1d evaluation output",
                            "the NumPy formula")
            parity_error = _max_abs(produced - stable_output)
            _require_parity(parity_error, FORWARD_ATOL, "the native output",
                            "tensorforge.nn.BatchNorm1d in eval mode")
            _require_unchanged(stable_module.running_mean,
                               data["running_mean"],
                               "the stable running_mean")
            _require_unchanged(stable_module.running_var, data["running_var"],
                               "the stable running_var")
            _require_unchanged(stable_input.data, data["values"],
                               "the stable input")
            metrics["reference_max_abs_error"] = reference_error
            metrics["native_vs_stable_max_abs_error"] = parity_error
            checks.append("stable_parity")
        metrics["checks"] = checks
        return metrics

    def close():
        native_input.close()
        _close_module(module)

    return {
        "native_prepare": lambda: None,
        "native_run": native_run,
        "native_cleanup": lambda _state, result: result.close(),
        "reference_prepare": lambda: None,
        "reference_run": reference_run,
        "reference_cleanup": lambda _state, _result: None,
        "check": check,
        "close": close,
    }


def _build_batchnorm_backward(config, spec):
    """One one-shot ``backward()`` through a training-mode BatchNorm.

    Training-mode backward is the primary backward case because it is the
    one that differentiates through the batch mean *and* the batch
    variance. The forward — and therefore the running-statistics update —
    is built outside the timer for every repetition, so the timed region
    is exactly ``backward()``."""
    data = _batchnorm_values(config, spec)
    shape = data["shape"]
    native_upstream = NativeTensor.from_array(data["upstream"])

    def native_prepare():
        module = _native_batch_norm(data, spec, True)
        native_input = NativeTensor.from_array(data["values"],
                                               requires_grad=True)
        output = module(native_input)
        weighted = output.multiply(native_upstream)
        return module, native_input, output, weighted, weighted.sum()

    def native_run(state):
        state[4].backward()
        return state[4]

    def native_cleanup(state, _result):
        module, native_input, output, weighted, objective = state
        _release_gradients([native_input, module.gamma, module.beta])
        objective.close()
        weighted.close()
        output.close()
        native_input.close()
        _close_module(module)

    def reference_prepare():
        """The honest stable oracle.

        For ``(N, C)`` this is simply stable ``BatchNorm1d``. For NCHW the
        problem is transformed to the equivalent ``(N*H*W, C)`` sample
        matrix, which reduces over exactly the same values; the input
        gradient is transformed back to NCHW for comparison. That
        transformation makes a rigorous oracle and a misleading timing
        reference, which is why the NCHW cases are ``native_only``."""
        stable_module = _stable_batch_norm_for(data, spec, True)
        if spec["module"] is NativeBatchNorm2d:
            values = _nchw_to_samples(data["values"])
            upstream = _nchw_to_samples(data["upstream"])
        else:
            values, upstream = data["values"].copy(), data["upstream"].copy()
        x = tensorforge.Tensor(values, requires_grad=True)
        weighted = stable_module(x) * tensorforge.Tensor(upstream)
        return stable_module, x, weighted.sum()

    def reference_run(state):
        state[2].backward()
        return state[2]

    def check():
        state = native_prepare()
        module, native_input, output, _weighted, objective = state
        mean_before = module.running_mean.to_numpy().copy()
        var_before = module.running_var.to_numpy().copy()
        try:
            native_run(state)
            for label, tensor in (("input", native_input),
                                  ("gamma", module.gamma),
                                  ("beta", module.beta)):
                if tensor.grad is None:
                    raise AssertionError(
                        f"the native backward produced no {label} gradient"
                    )
            input_grad = native_input.grad.to_numpy().copy()
            gamma_grad = module.gamma.grad.to_numpy().copy()
            beta_grad = module.beta.grad.to_numpy().copy()
            _require_shape(input_grad, shape, "the native input gradient")
            _require_shape(gamma_grad, (data["channels"],),
                           "the native gamma gradient")
            _require_shape(beta_grad, (data["channels"],),
                           "the native beta gradient")
            for label, gradient in (("input", input_grad),
                                    ("gamma", gamma_grad),
                                    ("beta", beta_grad)):
                _require_finite(gradient, f"the native {label} gradient")
            for name in ("running_mean", "running_var"):
                if getattr(module, name).grad is not None:
                    raise AssertionError(f"{name} received a gradient")
            _require_unchanged(module.running_mean.to_numpy(), mean_before,
                               "running_mean (backward must not advance it)")
            _require_unchanged(module.running_var.to_numpy(), var_before,
                               "running_var (backward must not advance it)")
            if not objective._graph_freed:
                raise AssertionError(
                    "the one-shot backward did not release the graph"
                )
            if objective._graph_resources or output._graph_resources:
                raise AssertionError(
                    "a graph resource survived the one-shot backward"
                )
            _require_unchanged(native_input.to_numpy(), data["values"],
                               "the native input")
        finally:
            native_cleanup(state, None)

        reference_state = reference_prepare()
        stable_module, stable_input, _ = reference_state
        reference_run(reference_state)
        stable_input_grad = stable_input.grad
        if spec["module"] is NativeBatchNorm2d:
            stable_input_grad = _samples_to_nchw(stable_input_grad, shape)
        input_error = _max_abs(input_grad - stable_input_grad)
        gamma_error = _max_abs(gamma_grad - stable_module.gamma.grad)
        beta_error = _max_abs(beta_grad - stable_module.beta.grad)
        error = max(input_error, gamma_error, beta_error)
        _require_parity(error, GRADIENT_ATOL, "the native gradients",
                        spec["correctness_reference"])
        return {
            "max_abs_error": error,
            "input_gradient_max_abs_error": input_error,
            "gamma_gradient_max_abs_error": gamma_error,
            "beta_gradient_max_abs_error": beta_error,
            "checks": ["input_gradient_present", "affine_gradients_present",
                       "gradient_shapes", "finite", "reference_parity",
                       "no_buffer_gradients", "backward_does_not_advance_state",
                       "graph_released", "no_graph_resource_survives",
                       "no_input_mutation"],
        }

    return {
        "native_prepare": native_prepare,
        "native_run": native_run,
        "native_cleanup": native_cleanup,
        "reference_prepare": reference_prepare,
        "reference_run": reference_run,
        "reference_cleanup": lambda _state, _result: None,
        "check": check,
        "close": lambda: native_upstream.close(),
    }


# ---------------------------------------------------------------------------
# The F6 normalized model, rebuilt here so the benchmark owns its models
# and never runs the example's training, resume, reporting, or checkpoint
# helpers. Importing the example module is import-safe and runs nothing.
# ---------------------------------------------------------------------------


# Running this file as a script puts ``benchmarks/`` on sys.path rather
# than the repository root, so make the root importable before reaching
# for the F6 example. Idempotent, and a no-op under pytest (which already
# has the root on the path).
def _ensure_repository_root_on_path():
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repository_root_on_path()

from examples.native_normalization_training import (  # noqa: E402
    DEFAULT_LR,
    HIDDEN_FEATURES,
    HIDDEN_SEED,
    IN_FEATURES,
    OUT_FEATURES,
    OUTPUT_SEED,
    build_dataset,
)
from examples.native_normalization_training import (  # noqa: E402
    MOMENTUM as TRAINING_MOMENTUM,
)

# The canonical F6 state keys: every parameter first, then the two
# persistent BatchNorm buffers — which is exactly ``state_dict()`` order.
_TRAINING_PARAMETER_KEYS = (
    "hidden.weight", "hidden.bias",
    "batch_norm.gamma", "batch_norm.beta",
    "layer_norm.weight", "layer_norm.bias",
    "output.weight", "output.bias",
)
_TRAINING_BUFFER_KEYS = ("batch_norm.running_mean", "batch_norm.running_var")

# Names that must never be reachable from this module: the timed training
# step does no checkpoint I/O and no reporting work.
_FORBIDDEN_TRAINING_STEP_WORK = (
    "save_native_checkpoint", "load_native_checkpoint", "native_accuracy",
    "run_training", "run_resume_proof", "evaluate",
)


class _BenchmarkNormalizedRegressor(NativeModule):
    """The F6 architecture, constructed here from the same seeds so the
    benchmark owns its models and never mutates the example's."""

    def __init__(self):
        super().__init__()
        self.hidden = NativeLinear(IN_FEATURES, HIDDEN_FEATURES,
                                   seed=HIDDEN_SEED)
        self.batch_norm = NativeBatchNorm1d(HIDDEN_FEATURES,
                                            momentum=TRAINING_MOMENTUM)
        self.relu = NativeReLU()
        self.layer_norm = NativeLayerNorm(HIDDEN_FEATURES)
        self.output = NativeLinear(HIDDEN_FEATURES, OUT_FEATURES,
                                   seed=OUTPUT_SEED)

    def forward(self, inputs):
        hidden = self.hidden(inputs)
        hidden = self.batch_norm(hidden)
        hidden = self.relu(hidden)
        hidden = self.layer_norm(hidden)
        return self.output(hidden)


def _native_initial_state():
    """The model's deterministic initial parameter *and* buffer values,
    read once outside every timed region and then released."""
    model = _BenchmarkNormalizedRegressor()
    try:
        state = model.state_dict()
        try:
            return {name: tensor.to_numpy().copy()
                    for name, tensor in state.items()}
        finally:
            for tensor in state.values():
                tensor.close()
    finally:
        _close_module(model)


def _stable_normalized_regressor(initial):
    """The structurally equivalent stable-line model, initialized from the
    *same* arrays the native model starts from — every linear parameter,
    the BatchNorm ``gamma``/``beta`` *and* running statistics, and the
    LayerNorm ``weight``/``bias`` — with the same epsilon and momentum.
    The correctness gate proves the equivalence numerically on every
    run."""
    from tensorforge.nn import (BatchNorm1d, LayerNorm, Linear, ReLU,
                                Sequential)

    model = Sequential(
        Linear(IN_FEATURES, HIDDEN_FEATURES),
        BatchNorm1d(HIDDEN_FEATURES, momentum=TRAINING_MOMENTUM),
        ReLU(),
        LayerNorm(HIDDEN_FEATURES),
        Linear(HIDDEN_FEATURES, OUT_FEATURES),
    )
    hidden, batch_norm, _relu, layer_norm, output = model.modules
    hidden.weight.data = initial["hidden.weight"].copy()
    hidden.bias.data = initial["hidden.bias"].copy()
    batch_norm.gamma.data = initial["batch_norm.gamma"].copy()
    batch_norm.beta.data = initial["batch_norm.beta"].copy()
    batch_norm.running_mean = initial["batch_norm.running_mean"].copy()
    batch_norm.running_var = initial["batch_norm.running_var"].copy()
    layer_norm.weight.data = initial["layer_norm.weight"].copy()
    layer_norm.bias.data = initial["layer_norm.bias"].copy()
    output.weight.data = initial["output.weight"].copy()
    output.bias.data = initial["output.bias"].copy()
    return model


def _stable_named_parameters(model):
    """The stable model's Parameters under the native model's canonical
    names, read from the modules rather than from traversal order."""
    hidden, batch_norm, _relu, layer_norm, output = model.modules
    return {
        "hidden.weight": hidden.weight,
        "hidden.bias": hidden.bias,
        "batch_norm.gamma": batch_norm.gamma,
        "batch_norm.beta": batch_norm.beta,
        "layer_norm.weight": layer_norm.weight,
        "layer_norm.bias": layer_norm.bias,
        "output.weight": output.weight,
        "output.bias": output.bias,
    }


def _stable_parameter_values(model):
    return {name: parameter.data
            for name, parameter in _stable_named_parameters(model).items()}


def _stable_gradient_values(model):
    """The stable model's gradients after backward. The stable Adam never
    clears them, so they survive its ``step()``."""
    return {name: parameter.grad
            for name, parameter in _stable_named_parameters(model).items()}


def _build_normalized_training_step(config, spec):
    """One complete F6-style normalized training step.

    The timed region is exactly ``zero_grad -> train() -> forward ->
    NativeMSELoss -> backward -> NativeAdam.step()``, which includes the
    BatchNorm running-statistics update the training forward performs. A
    **fresh** model and optimizer are constructed outside the timer for
    every repetition so each timed step starts from the same
    deterministic state; the dataset tensors, the loss module, checkpoint
    I/O, reporting conversions, and cleanup are all excluded."""
    del config  # the F6 dataset and architecture are fixed, not sampled
    inputs, targets = build_dataset()
    x = NativeTensor.from_array(inputs)
    y = NativeTensor.from_array(targets)
    criterion = NativeMSELoss()
    initial = _native_initial_state()
    stable_inputs = tensorforge.Tensor(np.asarray(inputs, dtype=np.float64))
    stable_targets = np.asarray(targets, dtype=np.float64)

    def native_prepare():
        model = _BenchmarkNormalizedRegressor()
        return model, NativeAdam(model.parameters(), lr=DEFAULT_LR)

    def native_run(state):
        model, optimizer = state
        optimizer.zero_grad()
        model.train()
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        return prediction, loss

    def native_cleanup(state, result):
        model, optimizer = state
        if result is not None:
            prediction, loss = result
            loss.close()
            prediction.close()
        _release_gradients(list(model.parameters()))
        optimizer.close()
        _close_module(model)

    def reference_prepare():
        from tensorforge.optim import Adam

        model = _stable_normalized_regressor(initial)
        return model, Adam(model.parameters(), lr=DEFAULT_LR)

    def reference_run(state):
        from tensorforge.nn import mse_loss

        model, optimizer = state
        optimizer.zero_grad()
        model.train()
        prediction = model(stable_inputs)
        loss = mse_loss(prediction, stable_targets)
        loss.backward()
        optimizer.step()
        return prediction, loss

    def check():
        module = sys.modules[__name__]
        for forbidden in _FORBIDDEN_TRAINING_STEP_WORK:
            if hasattr(module, forbidden):
                raise AssertionError(
                    f"the benchmark must not reach {forbidden}: the timed "
                    f"training step does no checkpoint or reporting work"
                )
        state = native_prepare()
        model, optimizer = state
        named = list(model.named_parameters())
        before = {name: parameter.to_numpy().copy()
                  for name, parameter in named}
        running_before = {
            "running_mean": model.batch_norm.running_mean.to_numpy().copy(),
            "running_var": model.batch_norm.running_var.to_numpy().copy(),
        }
        if tuple(name for name, _ in named) != _TRAINING_PARAMETER_KEYS:
            raise AssertionError("the model's parameter order changed")
        if tuple(name for name, _ in model.named_buffers()) != (
            _TRAINING_BUFFER_KEYS
        ):
            raise AssertionError("the model's buffer order changed")
        buffer_ids = {id(buffer) for buffer in model.buffers()}
        if buffer_ids & {id(p) for p in optimizer.parameters()}:
            raise AssertionError(
                "a BatchNorm running buffer reached the optimizer"
            )
        result = None
        try:
            result = native_run(state)
            prediction, loss = result
            if loss.shape != ():
                raise AssertionError("the training-step loss is not scalar")
            loss_value = float(loss.to_numpy())
            if not np.isfinite(loss_value):
                raise AssertionError("the training-step loss is not finite")
            gradients = {}
            for name, parameter in named:
                if parameter.grad is None:
                    raise AssertionError(f"{name} received no gradient")
                gradient = parameter.grad.to_numpy().copy()
                _require_shape(gradient, parameter.shape,
                               f"the {name} gradient")
                _require_finite(gradient, f"the {name} gradient")
                gradients[name] = gradient
            running_after = {
                "running_mean":
                    model.batch_norm.running_mean.to_numpy().copy(),
                "running_var":
                    model.batch_norm.running_var.to_numpy().copy(),
            }
            for name in running_before:
                if np.array_equal(running_after[name], running_before[name]):
                    raise AssertionError(
                        f"the BatchNorm {name} did not advance"
                    )
            steps = list(optimizer.step_counts)
            if steps != [1] * len(steps):
                raise AssertionError(
                    f"the optimizer step counts did not advance: {steps}"
                )
            after = {name: parameter.to_numpy().copy()
                     for name, parameter in named}
            changed = [name for name in after
                       if not np.array_equal(after[name], before[name])]
            if not changed:
                raise AssertionError("no parameter changed during the step")
            if not loss._graph_freed:
                raise AssertionError("the completed graph was not released")
            if loss._graph_resources or prediction._graph_resources:
                raise AssertionError("a graph resource survived the step")
            _nodes, resources = _walk_graph(loss)
            if resources:
                raise AssertionError(
                    "a BatchNorm graph resource survived the step"
                )
        finally:
            native_cleanup(state, result)
        if not (result[1].closed and result[0].closed):
            raise AssertionError("the step's transient tensors were not closed")

        reference_state = reference_prepare()
        stable_model, _ = reference_state
        _stable_prediction, stable_loss = reference_run(reference_state)
        loss_error = abs(loss_value - float(stable_loss.data))
        _require_parity(loss_error, LOSS_ATOL,
                        "the native pre-update training loss",
                        "the equivalently initialized stable model's loss")
        stable_gradients = _stable_gradient_values(stable_model)
        gradient_error = max(_max_abs(stable_gradients[name] - gradients[name])
                             for name in gradients)
        _require_parity(gradient_error, GRADIENT_ATOL,
                        "the native gradients after backward",
                        "the stable model's gradients")
        stable_after = _stable_parameter_values(stable_model)
        parameter_error = max(_max_abs(stable_after[name] - after[name])
                              for name in after)
        _require_parity(parameter_error, PARAMETER_ATOL,
                        "the native parameters after one step",
                        "the stable model's parameters after the same step")
        stable_batch_norm = stable_model.modules[1]
        running_error = max(
            _max_abs(stable_batch_norm.running_mean
                     - running_after["running_mean"]),
            _max_abs(stable_batch_norm.running_var
                     - running_after["running_var"]),
        )
        _require_parity(running_error, STATE_ATOL,
                        "the native BatchNorm running statistics after one step",
                        "the stable model's running statistics")
        return {
            "max_abs_error": parameter_error,
            "loss_abs_error": loss_error,
            "gradient_max_abs_error": gradient_error,
            "running_state_max_abs_error": running_error,
            "updated_parameters": sorted(changed),
            "checks": ["scalar_finite_loss", "all_gradients_present",
                       "gradient_shapes", "buffers_excluded_from_optimizer",
                       "running_state_advanced", "optimizer_state_advanced",
                       "parameter_updated", "graph_released",
                       "no_batchnorm_graph_resource_survives",
                       "transients_closed", "stable_loss_parity",
                       "stable_gradient_parity", "stable_parameter_parity",
                       "stable_running_state_parity",
                       "no_checkpoint_or_reporting_work"],
        }

    def close():
        x.close()
        y.close()

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
# The case registry — exactly the nine F7 cases, in this order.
# ---------------------------------------------------------------------------

_BATCHNORM2D_REFERENCE_DETAIL = (
    "none timed: the stable line has no public BatchNorm2d, so no "
    "structurally equivalent module exists to time. The correctness gate "
    "still uses a rigorous oracle, but that oracle is deliberately not "
    "timed as a 'BatchNorm2d reference' because it is a different module "
    "plus two layout transformations, which would make any ratio "
    "misleading"
)

CASES = {
    "layernorm_forward": {
        "category": "layer_normalization",
        "operation": "NativeLayerNorm(normalized_shape)(input)",
        "mode": None,
        "module": NativeLayerNorm,
        "reduction_axes": None,
        "eps": EPS,
        "momentum": None,
        "seed": 20260101,
        "reference_type": STABLE,
        "reference_detail": "tensorforge.nn.LayerNorm with the same input, "
                            "normalized shape, epsilon, and affine values",
        "correctness_reference": ("an explicit NumPy population-variance "
                                  "formula and tensorforge.nn.LayerNorm"),
        "configurations": {
            "full": {"shape": (64, 8, 16), "normalized_shape": (8, 16)},
            "smoke": {"shape": (4, 2, 4), "normalized_shape": (2, 4)},
        },
        "build": _build_layernorm_forward,
        "channelwise_affine_probe": False,
        "notes": ("Multi-axis normalization (two trailing dimensions) with "
                  "nontrivial affine values. The timed call includes the "
                  "composed graph construction the affine parameters cause, "
                  "the native operations, and the wrapper overhead; mode "
                  "independence is checked outside the timed region."),
    },
    "layernorm_backward": {
        "category": "layer_normalization",
        "operation": "NativeLayerNorm(...) -> weighted sum -> backward()",
        "mode": None,
        "module": NativeLayerNorm,
        "reduction_axes": None,
        "eps": EPS,
        "momentum": None,
        "seed": 20260102,
        "reference_type": STABLE,
        "reference_detail": "tensorforge.nn.LayerNorm under the same scalar "
                            "objective, input, epsilon, and affine values",
        "correctness_reference": ("tensorforge.nn.LayerNorm's input, weight, "
                                  "and bias gradients"),
        "configurations": {
            "full": {"shape": (64, 8, 16), "normalized_shape": (8, 16)},
            "smoke": {"shape": (4, 2, 4), "normalized_shape": (2, 4)},
        },
        "build": _build_layernorm_backward,
        "channelwise_affine_probe": False,
        "repetitions": BACKWARD_REPETITIONS,
        "notes": ("Only backward() is timed. A fresh forward graph is built "
                  "outside the timer for every repetition from cleared "
                  "gradients; no graph is reused and retain_graph is never "
                  "used to skip the rebuild."),
    },
    "batchnorm1d_training_forward": {
        "category": "batch_normalization_1d",
        "operation": "NativeBatchNorm1d(C)(input) in training mode",
        "mode": "train",
        "module": NativeBatchNorm1d,
        "reduction_axes": (0,),
        "eps": EPS,
        "momentum": MOMENTUM,
        "seed": 20260103,
        "reference_type": STABLE,
        "reference_detail": ("tensorforge.nn.BatchNorm1d in training mode "
                             "with the same input, epsilon, momentum, affine "
                             "values, and running state"),
        "correctness_reference": ("an explicit NumPy population-statistics "
                                  "formula and tensorforge.nn.BatchNorm1d"),
        "configurations": {
            "full": {"shape": (256, 64)},
            "smoke": {"shape": (16, 8)},
        },
        "build": _build_batchnorm_training_forward,
        "channelwise_affine_probe": False,
        "notes": ("The timed call includes the differentiable batch "
                  "statistics, the affine application, the graph-free "
                  "running-statistics preparation, and the atomic two-buffer "
                  "commit. Because the forward advances persistent state, a "
                  "fresh module carrying the same nontrivial affine values "
                  "and running state is built outside the timer for every "
                  "repetition."),
    },
    "batchnorm1d_eval_forward": {
        "category": "batch_normalization_1d",
        "operation": "NativeBatchNorm1d(C)(input) in evaluation mode",
        "mode": "eval",
        "module": NativeBatchNorm1d,
        "reduction_axes": (0,),
        "eps": EPS,
        "momentum": MOMENTUM,
        "seed": 20260104,
        "reference_type": STABLE,
        "reference_detail": ("tensorforge.nn.BatchNorm1d in eval mode with "
                             "the same input, epsilon, affine values, and "
                             "running state"),
        "correctness_reference": ("an explicit NumPy running-statistics "
                                  "formula and tensorforge.nn.BatchNorm1d"),
        "configurations": {
            "full": {"shape": (256, 64)},
            "smoke": {"shape": (16, 8)},
        },
        "build": _build_batchnorm_eval_forward,
        "channelwise_affine_probe": False,
        "notes": ("Nontrivial stored running_mean and positive running_var, "
                  "never the fresh module's zeros/ones. Evaluation mutates "
                  "nothing, so one module is shared across repetitions; the "
                  "structural check that the graph holds independent "
                  "snapshots rather than the registered buffers runs outside "
                  "the timed region."),
    },
    "batchnorm1d_backward": {
        "category": "batch_normalization_1d",
        "operation": ("NativeBatchNorm1d training forward -> weighted sum -> "
                      "backward()"),
        "mode": "train",
        "module": NativeBatchNorm1d,
        "reduction_axes": (0,),
        "eps": EPS,
        "momentum": MOMENTUM,
        "seed": 20260105,
        "reference_type": STABLE,
        "reference_detail": ("tensorforge.nn.BatchNorm1d's training-mode "
                             "backward under the same scalar objective and "
                             "the same state"),
        "correctness_reference": ("tensorforge.nn.BatchNorm1d's input, gamma, "
                                  "and beta gradients"),
        "configurations": {
            "full": {"shape": (256, 64)},
            "smoke": {"shape": (16, 8)},
        },
        "build": _build_batchnorm_backward,
        "channelwise_affine_probe": False,
        "repetitions": BACKWARD_REPETITIONS,
        "notes": ("Training-mode backward, so the gradient differentiates "
                  "through the batch mean and the batch variance. The "
                  "forward — and therefore the running-statistics update — "
                  "happens outside the timer for every repetition; only "
                  "backward() is timed."),
    },
    "batchnorm2d_training_forward": {
        "category": "batch_normalization_2d",
        "operation": "NativeBatchNorm2d(C)(NCHW input) in training mode",
        "mode": "train",
        "module": NativeBatchNorm2d,
        "reduction_axes": (0, 2, 3),
        "eps": EPS,
        "momentum": MOMENTUM,
        "seed": 20260106,
        "reference_type": NATIVE_ONLY,
        "reference_detail": _BATCHNORM2D_REFERENCE_DETAIL,
        "correctness_reference": ("an explicit NumPy NCHW "
                                  "population-statistics formula reducing "
                                  "over N, H, and W, plus an independent "
                                  "channelwise-affine probe"),
        "configurations": {
            "full": {"shape": (8, 8, 16, 16)},
            "smoke": {"shape": (2, 3, 4, 5)},
        },
        "build": _build_batchnorm_training_forward,
        "channelwise_affine_probe": True,
        "notes": ("Reduces over N, H, and W, so each channel gets one "
                  "population mean and variance over N*H*W values, with "
                  "(1, C, 1, 1) statistics and (C,) running buffers. Smoke "
                  "mode deliberately uses unequal C/H/W so an accidental "
                  "channel/spatial broadcast mistake cannot hide. No timing "
                  "ratio is published: there is no public stable "
                  "BatchNorm2d."),
    },
    "batchnorm2d_eval_forward": {
        "category": "batch_normalization_2d",
        "operation": "NativeBatchNorm2d(C)(NCHW input) in evaluation mode",
        "mode": "eval",
        "module": NativeBatchNorm2d,
        "reduction_axes": (0, 2, 3),
        "eps": EPS,
        "momentum": MOMENTUM,
        "seed": 20260107,
        "reference_type": NATIVE_ONLY,
        "reference_detail": _BATCHNORM2D_REFERENCE_DETAIL,
        "correctness_reference": ("an explicit NumPy NCHW "
                                  "running-statistics formula, plus a "
                                  "structural check that the graph holds "
                                  "independent (1, C, 1, 1) snapshots rather "
                                  "than the registered buffers"),
        "configurations": {
            "full": {"shape": (8, 8, 16, 16)},
            "smoke": {"shape": (2, 3, 4, 5)},
        },
        "build": _build_batchnorm_eval_forward,
        "channelwise_affine_probe": False,
        "notes": ("Nontrivial stored running statistics and affine values. "
                  "The output is checked NCHW, owning, contiguous, and "
                  "finite, the registered buffers are proved absent from the "
                  "graph, and nothing is mutated. No timing ratio is "
                  "published: there is no public stable BatchNorm2d."),
    },
    "batchnorm2d_backward": {
        "category": "batch_normalization_2d",
        "operation": ("NativeBatchNorm2d training forward -> weighted sum -> "
                      "backward()"),
        "mode": "train",
        "module": NativeBatchNorm2d,
        "reduction_axes": (0, 2, 3),
        "eps": EPS,
        "momentum": MOMENTUM,
        "seed": 20260108,
        "reference_type": NATIVE_ONLY,
        "reference_detail": _BATCHNORM2D_REFERENCE_DETAIL,
        "correctness_reference": ("tensorforge.nn.BatchNorm1d applied to the "
                                  "equivalent (N*H*W, C) sample matrix, with "
                                  "the input gradient transformed back to "
                                  "NCHW and the gamma/beta gradients compared "
                                  "directly — a correctness oracle only, "
                                  "never timed"),
        "configurations": {
            "full": {"shape": (8, 8, 16, 16)},
            "smoke": {"shape": (2, 3, 4, 5)},
        },
        "build": _build_batchnorm_backward,
        "channelwise_affine_probe": False,
        "repetitions": BACKWARD_REPETITIONS,
        "notes": ("Only backward() is timed; the training forward is built "
                  "outside the timer for every repetition. The correctness "
                  "oracle transforms the NCHW problem to the equivalent "
                  "(N*H*W, C) sample matrix and back, which is rigorous for "
                  "correctness but would be a misleading timing reference, "
                  "so no ratio is published."),
    },
    "normalized_training_step": {
        "category": "training_step",
        "operation": ("zero_grad -> train() -> Linear/BatchNorm1d/ReLU/"
                      "LayerNorm/Linear forward -> NativeMSELoss -> backward "
                      "-> NativeAdam.step()"),
        "mode": "train",
        "module": None,
        "reduction_axes": None,
        "eps": EPS,
        "momentum": TRAINING_MOMENTUM,
        "seed": 20260109,
        "reference_type": STABLE,
        "reference_detail": ("the same architecture, initial parameter and "
                             "running-state values, epsilon, momentum, MSE "
                             "semantics, and Adam hyperparameters on the "
                             "stable tensorforge line (tensorforge.nn "
                             "Linear/BatchNorm1d/ReLU/LayerNorm/Linear with "
                             "tensorforge.optim.Adam)"),
        "correctness_reference": ("the equivalently initialized stable "
                                  "Linear/BatchNorm1d/ReLU/LayerNorm/Linear "
                                  "model after the same single Adam step"),
        "configurations": {
            "full": {"shape": (8, 2)},
            "smoke": {"shape": (8, 2)},
        },
        "build": _build_normalized_training_step,
        "channelwise_affine_probe": False,
        "repetitions": TRAINING_STEP_REPETITIONS,
        "notes": ("The F6 fixed eight-sample dataset and architecture, "
                  "rebuilt here so the benchmark owns its models. The smoke "
                  "and full shapes are identical because this case "
                  "characterizes one real F6 iteration rather than a scaling "
                  "study. A fresh model and optimizer are built outside the "
                  "timer for every repetition; no to_numpy(), checkpoint "
                  "I/O, or reporting work occurs inside the timed region."),
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


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _measure_case(name, warmup, repetitions, smoke):
    """Build the case, run its correctness gate, and only then time it.

    The ordering here is the whole point: ``check()`` raises before
    ``measure`` is ever reached, so a failed gate publishes no timing."""
    spec = CASES[name]
    config = spec["configurations"]["smoke" if smoke else "full"]
    case_repetitions = min(repetitions, spec.get("repetitions", repetitions))
    case = spec["build"](config, spec)
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
        "mode": spec["mode"],
        "configuration": {key: _jsonable(value)
                          for key, value in config.items()},
        "shape": list(config["shape"]),
        "normalized_shape": _jsonable(config.get("normalized_shape")),
        "reduction_axes": _jsonable(spec["reduction_axes"]),
        "eps": spec["eps"],
        "momentum": spec["momentum"],
        "seed": spec["seed"],
        "reference_type": spec["reference_type"],
        "reference_detail": spec["reference_detail"],
        "correctness_reference": spec["correctness_reference"],
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


def _environment(warmup, repetitions):
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
        "scope": "native normalization stack (float64/cpu)",
        "timer": "time.perf_counter_ns",
        "primary_statistic": "median",
        "warmup": warmup,
        "repetitions": repetitions,
        "training_step_repetitions": min(repetitions,
                                         TRAINING_STEP_REPETITIONS),
        "backward_repetitions": min(repetitions, BACKWARD_REPETITIONS),
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
        "environment": _environment(warmup, repetitions),
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
    "Local characterization only -- not a performance contract. These "
    "numbers come\nfrom one machine, one build, and one workload; they are "
    "not cross-machine\ncomparable without controlled conditions. The "
    "observed ratios are observations,\nnot guarantees, and describe only "
    "what was measured here. Correctness is gated\nbefore timing, timing is "
    "never a pass/fail criterion, and no test or CI job\nasserts a duration."
)


def format_report(payload):
    """A concise human-readable report. Carries no speed verdict."""
    env = payload["environment"]
    lines = [
        f"TensorForge native normalization benchmark v{payload['version']} "
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
        f"(backward: {env['backward_repetitions']}, training step: "
        f"{env['training_step_repetitions']})",
        "",
    ]
    header = (
        f"{'case':<30} {'shape':<16} {'native median':>14} "
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
            f"{'x'.join(str(d) for d in record['shape']):<16} "
            f"{_format_duration(native['median_s']):>14} "
            f"{reference_median:>14} {ratio_text:>8} "
            f"{_format_duration(native['spread_s']):>10} "
            f"{record['reference_type']:<20} "
            f"{record['correctness']['status']:<8}"
        )
    lines.append("")
    lines.append(
        "ratio = native median / reference median (>1 means the native path "
        "took longer\nin this local run; <1 means it took less time here). "
        "The three BatchNorm2d cases\nreport n/a because the stable line has "
        "no public BatchNorm2d to time against;\ntheir correctness is still "
        "gated against an explicit NumPy NCHW formula and, for\nthe backward, "
        "a transformed stable oracle."
    )
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description=("Characterize the native normalization stack "
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
