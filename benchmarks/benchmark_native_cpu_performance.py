"""Unified CPU-performance baseline harness for the native line
(Advanced C++ Phase H, milestone H0).

This is the **measurement instrument** Phase H is built on. It does not
make anything faster and it changes no production numerical behavior: it
measures what the native stack already does, across every layer where an
honest comparison exists, so that the Phase-H milestone ladder is chosen
from evidence rather than from what looks unoptimized.

**Nothing here asserts a speed.** There is no timing threshold, no
performance budget, no committed duration, and no CI job that fails on a
number this file produces. Every figure is a local characterization of
one machine, one build, and one moment; it is not a performance contract
and it is not cross-machine comparable without controlled conditions. No
result file of any kind is written.

**Correctness runs before timing, always.** Every case validates its
native result against a reference *before* the timing helper is ever
reached, so a failed gate publishes no timing for the case and the CLI
exits nonzero with a clean stdout.

Implementation layers
---------------------

The point of this harness is to *separate* the layers a caller pays for,
because "the native line is slow" is not an actionable statement. Each
case declares the layers for which an honest measurement exists:

- ``numpy`` — NumPy on host arrays: the ceiling a vectorized,
  BLAS-backed float64 implementation reaches on this machine.
- ``stable_tensorforge`` — the stable ``tensorforge.nn`` /
  ``tensorforge.optim`` line on the same inputs and hyperparameters.
- ``raw_kernel`` — the plain raw-buffer C++ kernels (``cpp.matmul``,
  ``cpp.matmul_tiled``): C++ arithmetic with no tensor-runtime
  metadata, no ownership, and no allocation bookkeeping.
- ``tensor_core`` — ``NativeTensorCore``: the real production compute
  path, including the Python-side shape/stride plumbing, the fresh
  owning allocation, and the ctypes boundary.
- ``native_tensor`` — ``NativeTensor`` forward with **no** graph
  (no operand requires grad).
- ``native_tensor_graph`` — the same forward **with** autograd graph
  construction.
- ``backward`` — one ``backward()`` over a graph built outside the
  timer.
- ``optimizer_step`` — one optimizer ``step()`` with gradients already
  present.
- ``training_step`` — one complete deterministic training iteration.

Reference labelling is explicit and honest. A case is either measured
against a real equivalent (``numpy`` or ``stable_tensorforge``) or
labelled ``native_only``, in which case **no timing ratio is published
at all** — its correctness gate is still real. Nothing here fabricates a
comparison layer where no honest equivalent exists.

Modes
-----

::

    uv run python benchmarks/benchmark_native_cpu_performance.py
    uv run python benchmarks/benchmark_native_cpu_performance.py --smoke
    uv run python benchmarks/benchmark_native_cpu_performance.py --json
    uv run python benchmarks/benchmark_native_cpu_performance.py --smoke --json
    uv run python benchmarks/benchmark_native_cpu_performance.py --workload matmul
    uv run python benchmarks/benchmark_native_cpu_performance.py --case adam_step
    uv run python benchmarks/benchmark_native_cpu_performance.py --profile matmul_square_contiguous

``--profile CASE`` is the focused profiler mode: it runs exactly one case
at that case's **profile** configuration — deliberately larger shapes, so
the timed region dominates the fixed per-call dispatch cost — with a
raised repetition count, which is the shape a sampling or deterministic
profiler should be attached to. See
``docs/native_cpu_performance_design.md`` §5 for how profile shapes are
chosen.

What is timed
-------------

One measured repetition times exactly one call of the case's operation
with ``time.perf_counter_ns()``. Input creation, module and optimizer
construction, state installation, graph construction for the
backward-only cases, and every cleanup happen **outside** the timer.
Graph construction *is* inside the timer for the forward-with-graph and
training-step layers, because it is part of the call being characterized.
No sample is discarded, no timer overhead is subtracted, and every layer
of one case runs under the same setup discipline.

Temporary native outputs are closed explicitly between repetitions —
nothing here relies on garbage collection — and any case whose call
advances persistent state (a BatchNorm running buffer, a generator call
counter, an optimizer moment) rebuilds or resets that state outside the
timer so that supposedly identical repetitions really are identical.

Deliberately excluded
---------------------

File checkpoint I/O (``save_native_checkpoint`` /
``load_native_checkpoint``) is **not** measured. It is dominated by the
filesystem and the NPZ writer rather than by TensorForge, it would make
this harness write files, and it belongs to no training step. The
``state_operations`` workload measures the **in-memory** state surface
instead, which is the part TensorForge actually owns.

H0 adds **no** numerical capability: no kernel, C ABI export, ctypes
declaration, Core method, autograd operation, module, loss, metric,
optimizer, export, state-support entry, dtype, device, or checkpoint
change. It only measures what Phases A-G already shipped.
"""

import argparse
import json
import os
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
    NativeConv2d,
    NativeCrossEntropyLoss,
    NativeDropout,
    NativeFlatten,
    NativeGenerator,
    NativeLayerNorm,
    NativeLinear,
    NativeMaxPool2d,
    NativeModule,
    NativeMSELoss,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
)

BENCHMARK_NAME = "tensorforge.native_cpu_performance"
BENCHMARK_VERSION = "1.0"
# The JSON payload's own contract version. Bumped only when the shape of
# the payload changes, never when a measured number does.
#
# 1 — H0/H1.
# 2 — H2 added three fields: the ``tensor_core_generic`` implementation
#     layer, the per-case ``dispatch_comparison`` block beside
#     ``allocation_comparison``, and ``environment.native_build``. All
#     three are additive; no existing field changed meaning.
SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Implementation layers. Every timed row declares exactly one.
# ---------------------------------------------------------------------------
NUMPY = "numpy"
STABLE = "stable_tensorforge"
RAW_KERNEL = "raw_kernel"
RAW_KERNEL_TILED = "raw_kernel_tiled"
TENSOR_CORE = "tensor_core"
NATIVE_TENSOR = "native_tensor"
NATIVE_TENSOR_GRAPH = "native_tensor_graph"
BACKWARD = "backward"
OPTIMIZER_STEP = "optimizer_step"
TRAINING_STEP = "training_step"

# Phase H, milestone H1. The same production code path as its sibling
# layer, run with every H1 output allocation forced back onto the
# zero-initializing allocator. Pairing a layer with its ``_zeroed`` twin
# is the **primary** H1 comparison: both run identical arithmetic through
# identical Python, so the difference between them is the zero-fill and
# nothing else. NumPy is deliberately *not* the comparison here —
# ``numpy.zeros`` is served by ``calloc`` and can be answered with lazy
# zero pages, so timing against it would measure the operating system's
# page-fault policy rather than TensorForge's allocator.
TENSOR_CORE_ZEROED = "tensor_core_zeroed"
NATIVE_TENSOR_GRAPH_ZEROED = "native_tensor_graph_zeroed"
OPTIMIZER_STEP_ZEROED = "optimizer_step_zeroed"
TRAINING_STEP_ZEROED = "training_step_zeroed"

# Each H1 layer and the layer it is the zeroed twin of.
ZEROED_TWIN = {
    TENSOR_CORE_ZEROED: TENSOR_CORE,
    NATIVE_TENSOR_GRAPH_ZEROED: NATIVE_TENSOR_GRAPH,
    OPTIMIZER_STEP_ZEROED: OPTIMIZER_STEP,
    TRAINING_STEP_ZEROED: TRAINING_STEP,
}

# Phase H, milestone H2. The **same production Core call** on the **same
# logical operands**, delivered through a layout whose column stride is
# not 1 — which is how ``tf_core_matmul``'s metadata dispatch selects its
# retained generic reference path instead of the H2 row sweep. Pairing
# ``tensor_core`` with this layer is the primary H2 comparison: identical
# arithmetic, identical Python, identical accumulation order, differing
# only in which of the two shipped kernels ran.
#
# It is a *probe*, not a second implementation: no benchmark flag, hook,
# or ABI selector chooses a kernel here. The layout does, exactly as it
# does in production.
TENSOR_CORE_GENERIC = "tensor_core_generic"

# Each H2 layer and the layer whose generic-path twin it is.
GENERIC_TWIN = {TENSOR_CORE_GENERIC: TENSOR_CORE}

LAYERS = (NUMPY, STABLE, RAW_KERNEL, RAW_KERNEL_TILED, TENSOR_CORE,
          NATIVE_TENSOR, NATIVE_TENSOR_GRAPH, BACKWARD, OPTIMIZER_STEP,
          TRAINING_STEP,
          TENSOR_CORE_ZEROED, NATIVE_TENSOR_GRAPH_ZEROED,
          OPTIMIZER_STEP_ZEROED, TRAINING_STEP_ZEROED,
          TENSOR_CORE_GENERIC)

# Reference types. ``native_only`` means no honest equivalent exists, so
# no ratio is published for that case anywhere.
REFERENCE_NUMPY = "numpy"
REFERENCE_STABLE = "stable_tensorforge"
NATIVE_ONLY = "native_only"
REFERENCE_TYPES = (REFERENCE_NUMPY, REFERENCE_STABLE, NATIVE_ONLY)

# Workload families, in report order.
WORKLOADS = (
    "dispatch_overhead",
    "elementwise",
    "reduction",
    "matmul",
    "materialization",
    "linear",
    "convolution",
    "normalization",
    "stochastic",
    "optimizer",
    "training_step",
    "state_operations",
)

# Warm-up / repetition defaults. Heavy cases declare their own smaller
# cap; the count actually used is always reported per case.
DEFAULTS = {"warmup": 3, "repetitions": 11}
SMOKE_DEFAULTS = {"warmup": 1, "repetitions": 3}
PROFILE_DEFAULTS = {"warmup": 5, "repetitions": 25}
TRAINING_STEP_REPETITIONS = 7
BACKWARD_REPETITIONS = 9
HEAVY_REPETITIONS = 5

# Shared module arguments. These are configuration, not measurement
# parameters.
EPS = 1e-5
MOMENTUM = 0.1
LR = 0.05
DROPOUT_P = 0.5
DROPOUT_SEED = 20260728

# Phase H, milestone H2. The native matmul's fixed dispatch policy, echoed
# here so the report can name the path each measured layer took. These are
# *labels for the payload*, not knobs: nothing in this harness can change
# which kernel runs, and the values are pinned against the shipped header
# by tests/test_native_cpu_performance_benchmark.py. See
# docs/native_cpu_performance_design.md §16.2.
MATMUL_ROW_BLOCK = 4
MATMUL_MIN_COLUMNS = 8

# Absolute tolerances for the correctness gates. Ordinary float64
# agreement bounds taken from the existing parity suites — nothing here
# bounds a duration, a ratio, or a throughput.
EXACT = 0.0
FORWARD_ATOL = 1e-10
GRADIENT_ATOL = 1e-9
LOSS_ATOL = 1e-10
# Adam divides by ``sqrt(v_hat) + eps``, which amplifies a round-off
# difference in a near-zero gradient by up to ``lr / eps``. This bound
# covers that amplification on a converged or structurally dead
# parameter; the gradients themselves are still gated at GRADIENT_ATOL.
PARAMETER_ATOL = 1e-7

# Environment variables that change how a BLAS-backed NumPy or a threaded
# runtime behaves. Recorded when present so a reader can tell whether the
# NumPy reference column was single- or multi-threaded.
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "MKL_DYNAMIC",
    "OMP_DYNAMIC",
)


# ---------------------------------------------------------------------------
# Deterministic host inputs. Every generator is local and seeded; the
# global NumPy RNG is never read or mutated.
# ---------------------------------------------------------------------------


def _rng(seed):
    return np.random.default_rng(seed)


def _values(shape, seed, low=-2.0, high=2.0):
    """Finite, moderate float64 data — no overflow, no underflow, no
    degenerate feature. Numerical edge cases belong to the correctness
    suites and are deliberately excluded from timed data."""
    return _rng(seed).uniform(low, high, size=shape)


def _positive(shape, seed):
    return _rng(seed).uniform(0.5, 2.0, size=shape)


# ---------------------------------------------------------------------------
# Correctness-gate helpers. Each raises AssertionError, which the CLI
# turns into a nonzero exit with no timing published.
# ---------------------------------------------------------------------------


def _max_abs(values):
    values = np.asarray(values)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _require_finite(values, label):
    _require(bool(np.all(np.isfinite(values))), f"{label} is not finite")


def _require_shape(values, expected, label):
    produced = tuple(np.shape(values))
    _require(produced == tuple(expected),
             f"{label} has shape {produced}, expected {tuple(expected)}")


def _require_parity(error, tolerance, label, reference):
    """The single parity gate. ``tolerance`` is always a float64
    agreement bound; no caller passes a duration."""
    _require(np.isfinite(error) and error <= tolerance,
             f"{label} differs from the reference ({reference}) by "
             f"{error:g} (> {tolerance:g})")


def _require_unchanged(produced, expected, label):
    _require(np.array_equal(np.asarray(produced), np.asarray(expected)),
             f"{label} was mutated")


def _same_bits(left, right):
    """Raw IEEE-754 bit equality, which is a stricter statement than any
    tolerance: it separates ``+0.0`` from ``-0.0`` and never treats a NaN
    as equal to itself. Phase H, milestone H2 uses it for the one claim
    that is exact rather than approximate — that two native paths over the
    same operands agree bit for bit."""
    left = np.ascontiguousarray(left, dtype=np.float64)
    right = np.ascontiguousarray(right, dtype=np.float64)
    return (left.shape == right.shape
            and np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def _require_owning_contiguous(tensor, label):
    """``NativeTensor`` exposes ownership directly."""
    _require(tensor.owns_core and tensor.contiguous,
             f"{label} is not an owning contiguous result")


def _require_fresh_core(core, sources, label):
    """A ``NativeTensorCore`` result must be freshly allocated row-major
    contiguous storage that aliases none of its operands.

    ``NativeTensorCore`` has no public ownership flag, so this checks the
    observable properties the contract actually promises: contiguous
    layout, zero offset, and a storage object distinct from every
    source's."""
    _require(core.contiguous, f"{label} is not row-major contiguous")
    _require(core.offset == 0, f"{label} does not start at offset 0")
    for source in sources:
        _require(core.storage is not source.storage,
                 f"{label} aliases the storage of one of its operands")


# ---------------------------------------------------------------------------
# Native lifetime helpers. Cleanup is explicit everywhere — nothing in
# this harness relies on garbage collection.
# ---------------------------------------------------------------------------


def _close_optimizer(optimizer):
    """Release an optimizer's own native state, if it has any.

    ``NativeAdam`` owns moment buffers and has ``close()``; ``NativeSGD``
    is stateless and has none. Calling this instead of ``close()``
    directly keeps every cleanup path uniform without pretending the two
    optimizers have the same surface."""
    closer = getattr(optimizer, "close", None)
    if closer is not None:
        closer()


def _close_module(module):
    """There is no ``NativeModule.close()``, so a module's owner releases
    both its parameters and its buffers explicitly."""
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


def _install_state(module, **arrays):
    """Install nontrivial values through the public atomic loader, which
    preserves every parameter and buffer identity. Untimed setup."""
    values = {name: NativeTensor.from_array(np.asarray(value, dtype=np.float64))
              for name, value in arrays.items()}
    try:
        module.load_state_dict(values, strict=False)
    finally:
        for tensor in values.values():
            tensor.close()


class _forced_zero_initialized_allocation:
    """Run a block with every H1 output allocation forced back onto the
    zero-initializing allocator (Phase H, milestone H1).

    This is measurement scaffolding, not a production switch: it patches
    the two private constructors for the duration of one timed call and
    restores them in a ``finally``. The arithmetic, the Python path, the
    kernels, and the ownership rules are all identical either way, so a
    ``_zeroed`` layer differs from its twin by exactly one thing — the
    redundant zero-fill H1 removed.
    """

    def __enter__(self):
        self._core = cpp.NativeTensorCore._uninitialized
        self._storage = cpp.NativeStorage._uninitialized
        cpp.NativeTensorCore._uninitialized = cpp.NativeTensorCore.zeros
        cpp.NativeStorage._uninitialized = staticmethod(
            lambda size, dtype=None, device="cpu":
            cpp.NativeStorage(size, dtype=dtype, device=device)
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cpp.NativeTensorCore._uninitialized = self._core
        cpp.NativeStorage._uninitialized = self._storage
        return False


def _zeroed_layer(layer):
    """Wrap a layer so its timed ``run`` executes under the
    zero-initializing allocator. ``prepare`` and ``cleanup`` stay outside
    that scope, exactly as they stay outside the timer."""
    def run(state):
        with _forced_zero_initialized_allocation():
            return layer["run"](state)

    return {"prepare": layer["prepare"], "run": run,
            "cleanup": layer["cleanup"]}


def _nothing():
    return None


def _close_result(_state, result):
    if result is not None:
        result.close()


def _drop_result(_state, _result):
    return None


def _layer(run, prepare=_nothing, cleanup=_drop_result):
    return {"prepare": prepare, "run": run, "cleanup": cleanup}


# ---------------------------------------------------------------------------
# Case builders
#
# Each returns ``{"check": ..., "layers": {...}, "close": ...}``. ``check``
# runs the whole correctness gate and returns its metrics; it is called
# **before** any timing. ``layers`` maps an implementation-layer name to
# an untimed ``prepare``/``cleanup`` pair around one timed ``run``.
# ``close`` releases the case's shared persistent state once every layer
# has been measured.
# ---------------------------------------------------------------------------


def _build_scalar_dispatch(config, spec):
    """The size-independent cost of *reaching* the native kernel.

    A one-element tensor makes the arithmetic negligible, so what remains
    is exactly the per-call overhead a caller pays regardless of tensor
    size: the Python shape/stride normalization, the fresh owning
    allocation, the ctypes boundary, and (for the wrapper layers) the
    NativeTensor object and graph node. This is the case that decides
    whether small-operation overhead is material."""
    del spec
    shape = config["shape"]
    values = np.ones(shape)
    core_a = cpp.NativeTensorCore.from_array(values)
    core_b = cpp.NativeTensorCore.from_array(values)
    tensor_a = NativeTensor.from_array(values)
    tensor_b = NativeTensor.from_array(values)
    tensor_g = NativeTensor.from_array(values, requires_grad=True)
    host_a = values.copy()
    host_b = values.copy()

    def check():
        expected = host_a + host_b
        results = {}
        core_out = core_a.add(core_b)
        try:
            results[TENSOR_CORE] = core_out.to_numpy().copy()
        finally:
            core_out.close()
        plain = tensor_a.add(tensor_b)
        try:
            results[NATIVE_TENSOR] = plain.to_numpy().copy()
            _require(not plain.requires_grad,
                     "the no-grad layer built a gradient-tracking result")
            _require_owning_contiguous(plain, "the no-grad result")
        finally:
            plain.close()
        graphed = tensor_g.add(tensor_b)
        try:
            results[NATIVE_TENSOR_GRAPH] = graphed.to_numpy().copy()
            _require(graphed.requires_grad and not graphed.is_leaf,
                     "the graph layer built no autograd node")
        finally:
            graphed.close()
        errors = {}
        for layer, produced in results.items():
            _require_shape(produced, shape, f"the {layer} result")
            _require_finite(produced, f"the {layer} result")
            errors[layer] = _max_abs(produced - expected)
            _require_parity(errors[layer], EXACT, f"the {layer} result",
                            "NumPy elementwise addition")
        _require_unchanged(core_a.to_numpy(), values, "the native input")
        return {
            "max_abs_error": max(errors.values()),
            "per_layer_max_abs_error": errors,
            "checks": ["shape", "finite", "numpy_exact_parity",
                       "no_grad_layer_builds_no_graph",
                       "graph_layer_builds_a_node",
                       "owning_contiguous_output", "no_input_mutation"],
        }

    return {
        "check": check,
        "layers": {
            NUMPY: _layer(lambda _s=None: host_a + host_b),
            TENSOR_CORE: _layer(lambda _s=None: core_a.add(core_b),
                                cleanup=_close_result),
            NATIVE_TENSOR: _layer(lambda _s=None: tensor_a.add(tensor_b),
                                  cleanup=_close_result),
            NATIVE_TENSOR_GRAPH: _layer(lambda _s=None: tensor_g.add(tensor_b),
                                        cleanup=_close_result),
        },
        "close": lambda: (core_a.close(), core_b.close(), tensor_a.close(),
                          tensor_b.close(), tensor_g.close()),
    }


def _build_storage_allocation(config, spec):
    """Allocating and releasing one native storage buffer of the case's
    size, with **no** compute at all — the purest measurement of what
    Phase H, milestone H1 changed.

    Every native operation allocates a fresh owning output, so this is the
    fixed tax on every result the stack produces. The two native layers
    are the **primary** comparison:

    * ``tensor_core``        — the H1 uninitialized allocation;
    * ``tensor_core_zeroed`` — the zero-initializing default.

    Their difference is the zero-fill in isolation: one full write pass
    over the buffer, with no kernel, no metadata, and no Python
    bookkeeping in the way.

    ``numpy.zeros`` is measured for context only and is **deliberately
    not load-bearing evidence**: it is served by ``calloc``, which an
    operating system can answer with lazy zero pages that are not
    actually written until first touch. Timing against it would compare
    TensorForge's eager fill to the kernel's page-fault policy rather
    than to an alternative TensorForge could adopt."""
    del spec
    size = int(np.prod(config["shape"]))

    def check():
        storage = cpp.NativeStorage(size)
        try:
            produced = storage.to_numpy().copy()
            _require(storage.size == size,
                     f"the storage reports size {storage.size}, expected {size}")
            _require(produced.shape == (size,),
                     "the storage did not materialize as a flat buffer")
            _require(np.array_equal(produced, np.zeros(size)),
                     "native storage is not zero-initialized on construction")
        finally:
            storage.close()
        # The H1 sibling: same size, same metadata, same ownership, same
        # close semantics. Its *contents* are indeterminate by contract,
        # so nothing is asserted about them — reading them would be the
        # very thing the milestone exists to keep out of results.
        raw = cpp.NativeStorage._uninitialized(size)
        try:
            _require(raw.size == size,
                     f"the uninitialized storage reports size {raw.size}")
            _require(raw.dtype == "float64" and raw.device == "cpu",
                     "the uninitialized storage reports the wrong metadata")
            raw.fill(1.0)
            _require(np.array_equal(raw.to_numpy(), np.ones(size)),
                     "the uninitialized storage is not writable")
        finally:
            raw.close()
        _require(repr(raw) == "NativeStorage(closed)",
                 "close() did not release the uninitialized handle")
        # NativeStorage has no ``closed`` flag; a released handle is
        # observable through its repr, which is the documented surface.
        _require(repr(storage) == "NativeStorage(closed)",
                 f"close() did not release the handle: {storage!r}")
        core = cpp.NativeTensorCore.zeros(config["shape"])
        try:
            _require(np.array_equal(core.to_numpy(), np.zeros(config["shape"])),
                     "NativeTensorCore.zeros is not zero-initialized")
        finally:
            core.close()
        return {
            "max_abs_error": 0.0,
            "element_count": size,
            "checks": ["reported_size", "zero_initialized_storage",
                       "zero_initialized_core", "close_is_observable",
                       "uninitialized_metadata_matches",
                       "uninitialized_storage_is_writable",
                       "uninitialized_close_is_observable"],
        }

    return {
        "check": check,
        "layers": {
            NUMPY: _layer(lambda _s=None: np.zeros(size)),
            TENSOR_CORE: _layer(
                lambda _s=None: cpp.NativeStorage._uninitialized(size),
                cleanup=lambda _s, result: result.close(),
            ),
            TENSOR_CORE_ZEROED: _layer(
                lambda _s=None: cpp.NativeStorage(size),
                cleanup=lambda _s, result: result.close(),
            ),
        },
        "close": _nothing,
    }


def _build_elementwise(config, spec):
    """One elementwise ``multiply`` across every layer.

    ``spec["strided"]`` selects the operand layout: ``False`` gives two
    row-major contiguous operands (the flat fast-path kernel), ``True``
    makes the left operand a real transposed view (the generic odometer
    kernel). The logical values are identical in both, so the two cases
    isolate exactly the traversal difference."""
    shape = config["shape"]
    strided = spec["strided"]
    left = _values(shape, spec["seed"])
    right = _values(shape, spec["seed"] + 1)

    if strided:
        # A genuine transposed view: same logical values, stride-swapped.
        base = np.ascontiguousarray(left.T)
        core_base = cpp.NativeTensorCore.from_array(base)
        core_left = core_base.transpose(1, 0)
        tensor_base = NativeTensor.from_array(base)
        tensor_left = tensor_base.transpose(1, 0)
        tensor_base_g = NativeTensor.from_array(base, requires_grad=True)
        tensor_left_g = tensor_base_g.transpose(1, 0)
        host_left = base.T
        owned = [core_base, tensor_base, tensor_left, tensor_base_g,
                 tensor_left_g]
    else:
        core_base = None
        core_left = cpp.NativeTensorCore.from_array(left)
        tensor_left = NativeTensor.from_array(left)
        tensor_left_g = NativeTensor.from_array(left, requires_grad=True)
        host_left = left
        owned = [core_left, tensor_left, tensor_left_g]

    core_right = cpp.NativeTensorCore.from_array(right)
    tensor_right = NativeTensor.from_array(right)
    owned.extend([core_right, tensor_right])

    def check():
        expected = host_left * right
        _require(np.array_equal(host_left, left),
                 "the host reference view does not hold the case's values")
        if strided:
            _require(not core_left.contiguous,
                     "the strided case's operand is contiguous after all")
        else:
            _require(core_left.contiguous,
                     "the contiguous case's operand is strided")
        results = {}
        core_out = core_left.multiply(core_right)
        try:
            results[TENSOR_CORE] = core_out.to_numpy().copy()
            _require_fresh_core(core_out, (core_left, core_right),
                                "the core result")
        finally:
            core_out.close()
        plain = tensor_left.multiply(tensor_right)
        try:
            results[NATIVE_TENSOR] = plain.to_numpy().copy()
            _require(not plain.requires_grad,
                     "the no-grad layer built a gradient-tracking result")
        finally:
            plain.close()
        graphed = tensor_left_g.multiply(tensor_right)
        try:
            results[NATIVE_TENSOR_GRAPH] = graphed.to_numpy().copy()
            _require(graphed.requires_grad,
                     "the graph layer built no autograd node")
        finally:
            graphed.close()
        errors = {}
        for layer, produced in results.items():
            _require_shape(produced, shape, f"the {layer} result")
            _require_finite(produced, f"the {layer} result")
            errors[layer] = _max_abs(produced - expected)
            _require_parity(errors[layer], EXACT, f"the {layer} result",
                            "NumPy elementwise multiplication")
        _require_unchanged(core_right.to_numpy(), right, "the right operand")
        return {
            "max_abs_error": max(errors.values()),
            "per_layer_max_abs_error": errors,
            "operand_layout": "transposed_view" if strided else "contiguous",
            "checks": ["operand_layout", "shape", "finite",
                       "numpy_exact_parity", "owning_contiguous_output",
                       "no_grad_layer_builds_no_graph", "no_operand_mutation"],
        }

    return {
        "check": check,
        "layers": {
            NUMPY: _layer(lambda _s=None: host_left * right),
            TENSOR_CORE: _layer(lambda _s=None: core_left.multiply(core_right),
                                cleanup=_close_result),
            NATIVE_TENSOR: _layer(
                lambda _s=None: tensor_left.multiply(tensor_right),
                cleanup=_close_result),
            NATIVE_TENSOR_GRAPH: _layer(
                lambda _s=None: tensor_left_g.multiply(tensor_right),
                cleanup=_close_result),
        },
        "close": lambda: [item.close() for item in owned],
    }


def _build_reduction(config, spec):
    """One ``sum`` reduction across every layer, over a contiguous or a
    transposed-view operand. ``spec["axis"]`` is the reduced axis
    (``None`` reduces everything).

    Float sums are order-sensitive, so the gate compares against NumPy to
    a tolerance rather than bit-for-bit — the accumulation-order policy in
    ``docs/native_cpu_performance_design.md`` §7."""
    shape = config["shape"]
    axis = spec["axis"]
    strided = spec["strided"]
    values = _values(shape, spec["seed"])

    if strided:
        base = np.ascontiguousarray(values.T)
        core_base = cpp.NativeTensorCore.from_array(base)
        core_input = core_base.transpose(1, 0)
        tensor_base = NativeTensor.from_array(base)
        tensor_input = tensor_base.transpose(1, 0)
        tensor_base_g = NativeTensor.from_array(base, requires_grad=True)
        tensor_input_g = tensor_base_g.transpose(1, 0)
        host = base.T
        owned = [core_base, tensor_base, tensor_input, tensor_base_g,
                 tensor_input_g]
    else:
        core_input = cpp.NativeTensorCore.from_array(values)
        tensor_input = NativeTensor.from_array(values)
        tensor_input_g = NativeTensor.from_array(values, requires_grad=True)
        host = values
        owned = [core_input, tensor_input, tensor_input_g]

    def check():
        expected = host.sum(axis=axis)
        _require(np.array_equal(host, values),
                 "the host reference view does not hold the case's values")
        if strided:
            _require(not core_input.contiguous,
                     "the strided case's operand is contiguous after all")
        results = {}
        core_out = core_input.sum(axis=axis)
        try:
            results[TENSOR_CORE] = core_out.to_numpy().copy()
            _require_fresh_core(core_out, (core_input,),
                                "the core reduction result")
        finally:
            core_out.close()
        plain = tensor_input.sum(axis=axis)
        try:
            results[NATIVE_TENSOR] = plain.to_numpy().copy()
        finally:
            plain.close()
        graphed = tensor_input_g.sum(axis=axis)
        try:
            results[NATIVE_TENSOR_GRAPH] = graphed.to_numpy().copy()
            _require(graphed.requires_grad,
                     "the graph layer built no autograd node")
        finally:
            graphed.close()
        errors = {}
        scale = max(1.0, float(np.max(np.abs(expected))) if np.size(expected)
                    else 1.0)
        for layer, produced in results.items():
            _require_shape(produced, np.shape(expected), f"the {layer} result")
            _require_finite(produced, f"the {layer} result")
            errors[layer] = _max_abs(produced - expected)
            # Order-sensitive: a tolerance, deliberately not exactness.
            _require_parity(errors[layer], FORWARD_ATOL * scale,
                            f"the {layer} reduction",
                            "NumPy's sum over the same axis")
        return {
            "max_abs_error": max(errors.values()),
            "per_layer_max_abs_error": errors,
            "operand_layout": "transposed_view" if strided else "contiguous",
            "comparison": "tolerance (float summation is order-sensitive)",
            "checks": ["operand_layout", "result_shape", "finite",
                       "numpy_tolerance_parity", "owning_contiguous_output"],
        }

    return {
        "check": check,
        "layers": {
            NUMPY: _layer(lambda _s=None: host.sum(axis=axis)),
            TENSOR_CORE: _layer(lambda _s=None: core_input.sum(axis=axis),
                                cleanup=_close_result),
            NATIVE_TENSOR: _layer(lambda _s=None: tensor_input.sum(axis=axis),
                                  cleanup=_close_result),
            NATIVE_TENSOR_GRAPH: _layer(
                lambda _s=None: tensor_input_g.sum(axis=axis),
                cleanup=_close_result),
        },
        "close": lambda: [item.close() for item in owned],
    }


def _build_matmul(config, spec):
    """One matrix multiplication across every layer, including the two
    raw-buffer kernels.

    ``spec["strided_rhs"]`` decides the right operand's layout. With
    ``False`` both operands are row-major contiguous, which is what
    ``NativeLinear`` actually produces. With ``True`` the right operand is
    a real transposed view — the strided fallback path — carrying the
    *same logical values*, so the two cases isolate the access pattern
    alone.

    ``raw_kernel`` covers both ``cpp.matmul`` (the naive triple loop over
    plain buffers) and ``cpp.matmul_tiled`` (the existing cache-blocking
    experiment). Neither is on any production path; they are measured
    here as evidence, not adopted.

    Phase H (H2): a contiguous case additionally gets a
    ``tensor_core_generic`` layer — the **same production Core call** on
    the **same logical operands**, delivered through a layout whose column
    stride is not 1 so that ``tf_core_matmul``'s metadata dispatch selects
    its retained generic reference path. That pair is the primary H2
    comparison; the gate proves the two agree bit for bit before either is
    timed."""
    m, n, p = config["m"], config["n"], config["p"]
    strided_rhs = spec["strided_rhs"]
    left = _values((m, n), spec["seed"])
    right = _values((n, p), spec["seed"] + 1)

    core_left = cpp.NativeTensorCore.from_array(left)
    tensor_left = NativeTensor.from_array(left)
    tensor_left_g = NativeTensor.from_array(left, requires_grad=True)
    owned = [core_left, tensor_left, tensor_left_g]

    right_transposed = np.ascontiguousarray(right.T)
    if strided_rhs:
        core_right_base = cpp.NativeTensorCore.from_array(right_transposed)
        core_right = core_right_base.transpose(1, 0)
        tensor_right_base = NativeTensor.from_array(right_transposed)
        tensor_right = tensor_right_base.transpose(1, 0)
        owned.extend([core_right_base, tensor_right_base, tensor_right])
        core_right_generic = None
    else:
        core_right = cpp.NativeTensorCore.from_array(right)
        tensor_right = NativeTensor.from_array(right)
        owned.extend([core_right, tensor_right])
        # The H2 generic-path probe: identical values, column stride n.
        generic_base = cpp.NativeTensorCore.from_array(right_transposed)
        core_right_generic = generic_base.transpose(1, 0)
        owned.extend([generic_base, core_right_generic])

    block = config["block"]
    # The H2 dispatch precondition, mirrored from
    # docs/native_cpu_performance_design.md §16.2 so the report can name
    # the path each layer actually took. The predicate itself is
    # hidden-visibility C++ with no exported selector — which is the point
    # — so this restates it rather than querying it.
    def _path(operand):
        takes_sweep = (operand.strides[1] == 1 and n >= 1
                       and p >= MATMUL_MIN_COLUMNS)
        return "row_sweep" if takes_sweep else "generic_strided"

    def check():
        expected = left @ right
        if strided_rhs:
            _require(not core_right.contiguous,
                     "the strided case's right operand is contiguous")
            _require(np.array_equal(core_right.to_numpy(), right),
                     "the transposed view does not hold the same logical "
                     "values as the contiguous operand")
        else:
            _require(core_right.contiguous,
                     "the contiguous case's right operand is strided")
        results = {RAW_KERNEL: cpp.matmul(left, right),
                   RAW_KERNEL_TILED: cpp.matmul_tiled(left, right, block)}
        core_out = core_left.matmul(core_right)
        try:
            results[TENSOR_CORE] = core_out.to_numpy().copy()
            _require_fresh_core(core_out, (core_left, core_right),
                                "the core matmul result")
        finally:
            core_out.close()
        if core_right_generic is not None:
            _require(core_right_generic.strides[1] != 1,
                     "the generic-path probe still has unit column stride, "
                     "so it would not select the generic kernel")
            _require(np.array_equal(core_right_generic.to_numpy(), right),
                     "the generic-path probe does not hold the same logical "
                     "values as the contiguous operand")
            generic_out = core_left.matmul(core_right_generic)
            try:
                results[TENSOR_CORE_GENERIC] = generic_out.to_numpy().copy()
            finally:
                generic_out.close()
        plain = tensor_left.matmul(tensor_right)
        try:
            results[NATIVE_TENSOR] = plain.to_numpy().copy()
        finally:
            plain.close()
        graphed = tensor_left_g.matmul(tensor_right)
        try:
            results[NATIVE_TENSOR_GRAPH] = graphed.to_numpy().copy()
            _require(graphed.requires_grad,
                     "the graph layer built no autograd node")
        finally:
            graphed.close()
        errors = {}
        scale = max(1.0, float(np.max(np.abs(expected))))
        for layer, produced in results.items():
            _require_shape(produced, (m, p), f"the {layer} result")
            _require_finite(produced, f"the {layer} result")
            errors[layer] = _max_abs(produced - expected)
            _require_parity(errors[layer], FORWARD_ATOL * scale,
                            f"the {layer} product",
                            "NumPy's matmul on the same operands")
        # H2's load-bearing gate: the two shipped native paths are
        # compared to each other **bit for bit**, not to a tolerance, and
        # before either is timed. Both the raw naive kernel and the raw
        # tiled kernel must agree exactly too.
        #
        # Unqualified bit equality is the right assertion *here* because
        # every operand in this harness is finite seeded data, so no
        # result is a NaN and part (b) of H2's numerical contract applies
        # in full. The NaN-payload part of that contract (part (d)) is
        # neither reachable nor tested from a benchmark; it lives in
        # tests/test_native_matmul_dispatch.py and cpp/tests/test_matmul.cpp.
        exact_checks = ["operand_layout", "logical_values_match", "shape",
                        "finite", "numpy_tolerance_parity",
                        "owning_contiguous_output", "no_operand_mutation"]
        native = results[TENSOR_CORE]
        for layer in (RAW_KERNEL, RAW_KERNEL_TILED, TENSOR_CORE_GENERIC,
                      NATIVE_TENSOR, NATIVE_TENSOR_GRAPH):
            if layer not in results:
                continue
            _require(_same_bits(results[layer], native),
                     f"the {layer} result is not bit-identical to the "
                     f"production tensor_core result")
        exact_checks.append("finite_bit_identical_native_paths")
        _require_unchanged(core_left.to_numpy(), left, "the left operand")
        return {
            "max_abs_error": max(errors.values()),
            "per_layer_max_abs_error": errors,
            "operand_layout": ("contiguous @ transposed_view" if strided_rhs
                               else "contiguous @ contiguous"),
            "left_strides": list(core_left.strides),
            "right_strides": list(core_right.strides),
            "generic_probe_strides": (list(core_right_generic.strides)
                                      if core_right_generic is not None
                                      else None),
            "production_path": _path(core_right),
            "generic_probe_path": (_path(core_right_generic)
                                   if core_right_generic is not None
                                   else None),
            "matmul_row_block": MATMUL_ROW_BLOCK,
            "matmul_min_columns": MATMUL_MIN_COLUMNS,
            "tile_block": block,
            "flops": 2 * m * n * p,
            "comparison": ("tolerance against NumPy; exact bit equality "
                           "between the native paths"),
            "checks": exact_checks,
        }

    layers = {
        NUMPY: _layer(lambda _s=None: left @ right),
        RAW_KERNEL: _layer(lambda _s=None: cpp.matmul(left, right)),
        # The existing cache-blocking experiment, measured beside the
        # naive one so the blocking question has evidence rather than
        # an assumption. It is on no production path.
        RAW_KERNEL_TILED: _layer(
            lambda _s=None: cpp.matmul_tiled(left, right, block)),
        TENSOR_CORE: _layer(lambda _s=None: core_left.matmul(core_right),
                            cleanup=_close_result),
        NATIVE_TENSOR: _layer(
            lambda _s=None: tensor_left.matmul(tensor_right),
            cleanup=_close_result),
        NATIVE_TENSOR_GRAPH: _layer(
            lambda _s=None: tensor_left_g.matmul(tensor_right),
            cleanup=_close_result),
    }
    # The probe is *gated* whenever it exists, so a configuration that
    # falls back keeps its bit-identity check. It is only *timed* when the
    # primary layer really does take the row sweep — otherwise both sides
    # would be the generic kernel and the pair's labels would lie.
    if core_right_generic is not None and _path(core_right) == "row_sweep":
        layers[TENSOR_CORE_GENERIC] = _layer(
            lambda _s=None: core_left.matmul(core_right_generic),
            cleanup=_close_result)

    return {
        "check": check,
        "layers": layers,
        "close": lambda: [item.close() for item in owned],
    }


def _build_materialization(config, spec):
    """Materializing a transposed view into a fresh owning contiguous
    result — the Policy-B copy the Core layer performs whenever a
    contiguous-only kernel meets a strided operand."""
    shape = config["shape"]
    values = _values(shape, spec["seed"])
    base = np.ascontiguousarray(values.T)
    core_base = cpp.NativeTensorCore.from_array(base)
    core_view = core_base.transpose(1, 0)
    tensor_base = NativeTensor.from_array(base)
    tensor_view = tensor_base.transpose(1, 0)
    tensor_base_g = NativeTensor.from_array(base, requires_grad=True)
    tensor_view_g = tensor_base_g.transpose(1, 0)
    owned = [core_base, tensor_base, tensor_view, tensor_base_g, tensor_view_g]

    def check():
        expected = np.ascontiguousarray(base.T)
        _require(not core_view.contiguous, "the source view is contiguous")
        results = {}
        core_out = core_view.contiguous_copy()
        try:
            results[TENSOR_CORE] = core_out.to_numpy().copy()
            _require_fresh_core(core_out, (core_base,),
                                "the materialized core")
            _require(core_out.storage is not core_base.storage,
                     "contiguous_copy aliased the source storage")
        finally:
            core_out.close()
        plain = tensor_view.contiguous_copy()
        try:
            results[NATIVE_TENSOR] = plain.to_numpy().copy()
            _require_owning_contiguous(plain, "the no-grad materialization")
        finally:
            plain.close()
        graphed = tensor_view_g.contiguous_copy()
        try:
            results[NATIVE_TENSOR_GRAPH] = graphed.to_numpy().copy()
            _require(graphed.requires_grad,
                     "the graph layer built no autograd node")
        finally:
            graphed.close()
        errors = {}
        for layer, produced in results.items():
            _require_shape(produced, shape, f"the {layer} result")
            errors[layer] = _max_abs(produced - expected)
            _require_parity(errors[layer], EXACT, f"the {layer} result",
                            "numpy.ascontiguousarray of the same view")
        _require_unchanged(core_base.to_numpy(), base, "the source storage")
        return {
            "max_abs_error": max(errors.values()),
            "per_layer_max_abs_error": errors,
            "checks": ["source_is_strided", "shape", "numpy_exact_parity",
                       "owning_contiguous_output", "no_storage_aliasing",
                       "no_source_mutation"],
        }

    return {
        "check": check,
        "layers": {
            NUMPY: _layer(lambda _s=None: np.ascontiguousarray(base.T)),
            TENSOR_CORE: _layer(lambda _s=None: core_view.contiguous_copy(),
                                cleanup=_close_result),
            NATIVE_TENSOR: _layer(lambda _s=None: tensor_view.contiguous_copy(),
                                  cleanup=_close_result),
            NATIVE_TENSOR_GRAPH: _layer(
                lambda _s=None: tensor_view_g.contiguous_copy(),
                cleanup=_close_result),
        },
        "close": lambda: [item.close() for item in owned],
    }


def _stable_linear(in_features, out_features, weight, bias):
    from tensorforge.nn import Linear

    module = Linear(in_features, out_features)
    module.weight.data = np.array(weight, dtype=np.float64)
    module.bias.data = np.array(bias, dtype=np.float64)
    return module


def _build_linear_forward(config, spec):
    """One ``NativeLinear`` forward, graph construction included, against
    the identically initialized stable ``tensorforge.nn.Linear``."""
    batch, in_features, out_features = (config["batch"], config["in_features"],
                                        config["out_features"])
    values = _values((batch, in_features), spec["seed"])
    module = NativeLinear(in_features, out_features, seed=spec["seed"])
    weight = module.weight.to_numpy().copy()
    bias = module.bias.to_numpy().copy()
    native_input = NativeTensor.from_array(values)
    stable_module = _stable_linear(in_features, out_features, weight, bias)
    stable_input = tensorforge.Tensor(values.copy())

    def check():
        expected = values @ weight + bias
        versions = (module.weight.version, module.bias.version)
        output = module(native_input)
        try:
            produced = output.to_numpy().copy()
            _require_owning_contiguous(output, "the native output")
            _require(output.requires_grad,
                     "the forward built no graph over the parameters")
        finally:
            output.close()
        _require_shape(produced, (batch, out_features), "the native output")
        _require_finite(produced, "the native output")
        scale = max(1.0, float(np.max(np.abs(expected))))
        native_error = _max_abs(produced - expected)
        _require_parity(native_error, FORWARD_ATOL * scale, "the native output",
                        "the explicit NumPy x @ W + b formula")
        stable_output = stable_module(stable_input).data
        reference_error = _max_abs(stable_output - expected)
        _require_parity(reference_error, FORWARD_ATOL * scale,
                        "the stable Linear output", "the NumPy formula")
        parity = _max_abs(produced - stable_output)
        _require_parity(parity, FORWARD_ATOL * scale, "the native output",
                        "tensorforge.nn.Linear")
        _require((module.weight.version, module.bias.version) == versions,
                 "a parameter version moved during the forward")
        _require_unchanged(native_input.to_numpy(), values, "the native input")
        _require_unchanged(stable_input.data, values, "the stable input")
        return {
            "max_abs_error": native_error,
            "reference_max_abs_error": reference_error,
            "native_vs_stable_max_abs_error": parity,
            "checks": ["output_shape", "finite", "owning_contiguous_output",
                       "numpy_formula_parity", "stable_parity",
                       "graph_constructed", "parameter_versions_unchanged",
                       "no_input_mutation"],
        }

    return {
        "check": check,
        "layers": {
            STABLE: _layer(lambda _s=None: stable_module(stable_input)),
            NATIVE_TENSOR_GRAPH: _layer(lambda _s=None: module(native_input),
                                        cleanup=_close_result),
        },
        "close": lambda: (native_input.close(), _close_module(module)),
    }


def _build_linear_forward_backward(config, spec):
    """One ``backward()`` through ``NativeLinear``.

    Every repetition builds a fresh forward graph **outside** the timer
    from cleared gradients, times exactly one ``backward()``, and releases
    the graph afterwards. No graph is reused and ``retain_graph`` is never
    used to skip the rebuild, so no repetition inherits a retained graph
    or an accumulated gradient from the one before it."""
    batch, in_features, out_features = (config["batch"], config["in_features"],
                                        config["out_features"])
    values = _values((batch, in_features), spec["seed"])
    upstream = _values((batch, out_features), spec["seed"] + 3)
    module = NativeLinear(in_features, out_features, seed=spec["seed"])
    weight = module.weight.to_numpy().copy()
    bias = module.bias.to_numpy().copy()
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

    def stable_prepare():
        stable_module = _stable_linear(in_features, out_features, weight, bias)
        x = tensorforge.Tensor(values.copy(), requires_grad=True)
        weighted = stable_module(x) * tensorforge.Tensor(upstream.copy())
        return stable_module, x, weighted.sum()

    def stable_run(state):
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
                _require(tensor.grad is not None,
                         f"the native backward produced no {label} gradient")
            input_grad = native_input.grad.to_numpy().copy()
            weight_grad = module.weight.grad.to_numpy().copy()
            bias_grad = module.bias.grad.to_numpy().copy()
            _require_shape(input_grad, (batch, in_features),
                           "the native input gradient")
            _require_shape(weight_grad, (in_features, out_features),
                           "the native weight gradient")
            _require_shape(bias_grad, (out_features,),
                           "the native bias gradient")
            for label, gradient in (("input", input_grad),
                                    ("weight", weight_grad),
                                    ("bias", bias_grad)):
                _require_finite(gradient, f"the native {label} gradient")
            _require(objective._graph_freed,
                     "the one-shot backward did not release the graph")
            _require(not objective._graph_resources
                     and not output._graph_resources,
                     "a graph resource survived the one-shot backward")
            _require_unchanged(native_input.to_numpy(), values,
                               "the native input")
        finally:
            native_cleanup(state, None)

        # The closed-form gradients, computed independently of both lines.
        expected_input = upstream @ weight.T
        expected_weight = values.T @ upstream
        expected_bias = upstream.sum(axis=0)
        scale = max(1.0, float(np.max(np.abs(expected_weight))))
        formula_error = max(_max_abs(input_grad - expected_input),
                            _max_abs(weight_grad - expected_weight),
                            _max_abs(bias_grad - expected_bias))
        _require_parity(formula_error, GRADIENT_ATOL * scale,
                        "the native gradients",
                        "the closed-form NumPy gradients")

        stable_state = stable_prepare()
        stable_module, stable_input, _ = stable_state
        stable_run(stable_state)
        stable_error = max(
            _max_abs(input_grad - stable_input.grad),
            _max_abs(weight_grad - stable_module.weight.grad),
            _max_abs(bias_grad - stable_module.bias.grad),
        )
        _require_parity(stable_error, GRADIENT_ATOL * scale,
                        "the native gradients",
                        "tensorforge.nn.Linear's gradients")
        return {
            "max_abs_error": formula_error,
            "native_vs_stable_max_abs_error": stable_error,
            "checks": ["input_gradient_present", "parameter_gradients_present",
                       "gradient_shapes", "finite", "closed_form_parity",
                       "stable_parity", "graph_released",
                       "no_graph_resource_survives", "no_input_mutation"],
        }

    return {
        "check": check,
        "layers": {
            STABLE: _layer(stable_run, prepare=stable_prepare),
            BACKWARD: _layer(native_run, prepare=native_prepare,
                             cleanup=native_cleanup),
        },
        "close": lambda: (native_upstream.close(), _close_module(module)),
    }


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------


def _conv_shapes(config):
    return (config["batch"], config["in_channels"], config["height"],
            config["width"], config["out_channels"], config["kernel"])


def _numpy_conv2d_forward(images, weight, bias):
    """An explicit NCHW cross-correlation, written as a formula rather
    than borrowed from either implementation. No padding, unit stride."""
    n, _c, h, w = images.shape
    o, c, kh, kw = weight.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    output = np.empty((n, o, out_h, out_w), dtype=np.float64)
    for i in range(n):
        for j in range(o):
            for y in range(out_h):
                for x in range(out_w):
                    window = images[i, :, y:y + kh, x:x + kw]
                    output[i, j, y, x] = float(np.sum(window * weight[j]))
    return output + bias.reshape(1, o, 1, 1)


def _build_conv2d(config, spec):
    """One convolution component: ``spec["component"]`` selects the
    forward, the input gradient, the weight gradient, or the **composed**
    bias gradient.

    The bias gradient deliberately has its own case because it is not a
    kernel at all: it is three chained native ``sum`` reductions
    (``g.sum(0).sum(1).sum(1)``), and whether that composition is cheap
    or dominant relative to the three real convolution kernels is exactly
    the kind of thing H0 exists to find out rather than assume."""
    n, c, h, w, o, k = _conv_shapes(config)
    component = spec["component"]
    images = _values((n, c, h, w), spec["seed"])
    weight = _values((o, c, k, k), spec["seed"] + 1)
    bias = _values((o,), spec["seed"] + 2)

    core_images = cpp.NativeTensorCore.from_array(images)
    core_weight = cpp.NativeTensorCore.from_array(weight)
    core_bias = cpp.NativeTensorCore.from_array(bias)
    out_h, out_w = h - k + 1, w - k + 1
    upstream = _values((n, o, out_h, out_w), spec["seed"] + 3)
    core_upstream = cpp.NativeTensorCore.from_array(upstream)
    owned = [core_images, core_weight, core_bias, core_upstream]

    tensor_images = NativeTensor.from_array(images)
    tensor_images_g = NativeTensor.from_array(images, requires_grad=True)
    tensor_upstream = NativeTensor.from_array(upstream)
    owned.extend([tensor_images, tensor_images_g, tensor_upstream])

    module = NativeConv2d(c, o, k, seed=spec["seed"])
    _install_state(module, weight=weight, bias=bias)

    def stable_module():
        from tensorforge.nn import Conv2d

        stable = Conv2d(c, o, k)
        stable.weight.data = weight.copy()
        stable.bias.data = bias.copy()
        return stable

    def core_forward(_state=None):
        return core_images.conv2d_forward(core_weight, core_bias)

    def core_input_backward(_state=None):
        return core_upstream.conv2d_input_backward(
            core_weight, input_shape=(n, c, h, w))

    def core_weight_backward(_state=None):
        return core_upstream.conv2d_weight_backward(
            core_images, weight_shape=(o, c, k, k))

    def core_bias_gradient(_state=None):
        return core_upstream.sum(0).sum(1).sum(1)

    core_runs = {
        "forward": core_forward,
        "input_backward": core_input_backward,
        "weight_backward": core_weight_backward,
        "bias_gradient": core_bias_gradient,
    }

    def check():
        expected_forward = _numpy_conv2d_forward(images, weight, bias)
        scale = max(1.0, float(np.max(np.abs(expected_forward))))
        metrics = {"component": component}
        checks = ["component_result_shape", "finite", "owning_contiguous_output"]

        # The forward is validated for every component, because it is what
        # the other three are gradients *of*.
        forward = core_forward()
        try:
            produced_forward = forward.to_numpy().copy()
            _require_fresh_core(forward, (core_images, core_weight),
                                "the conv2d forward result")
        finally:
            forward.close()
        _require_shape(produced_forward, (n, o, out_h, out_w),
                       "the native conv2d forward")
        _require_finite(produced_forward, "the native conv2d forward")
        forward_error = _max_abs(produced_forward - expected_forward)
        _require_parity(forward_error, FORWARD_ATOL * scale,
                        "the native conv2d forward",
                        "an explicit NumPy NCHW cross-correlation formula")
        metrics["forward_max_abs_error"] = forward_error
        checks.append("numpy_cross_correlation_parity")

        stable = stable_module()
        stable_forward = stable(tensorforge.Tensor(images.copy())).data
        stable_error = _max_abs(stable_forward - expected_forward)
        _require_parity(stable_error, FORWARD_ATOL * scale,
                        "the stable Conv2d forward", "the NumPy formula")
        metrics["stable_forward_max_abs_error"] = stable_error
        checks.append("stable_forward_parity")

        result = core_runs[component]()
        try:
            produced = result.to_numpy().copy()
            _require_fresh_core(result, (core_images, core_weight,
                                         core_upstream),
                                f"the {component} result")
        finally:
            result.close()
        _require_finite(produced, f"the native {component}")

        if component == "forward":
            metrics["max_abs_error"] = forward_error
        elif component == "bias_gradient":
            expected = upstream.sum(axis=(0, 2, 3))
            _require_shape(produced, (o,), "the native bias gradient")
            error = _max_abs(produced - expected)
            _require_parity(error, GRADIENT_ATOL * scale,
                            "the composed native bias gradient",
                            "NumPy's sum over N, H, and W")
            metrics["max_abs_error"] = error
            checks.append("bias_gradient_reduction_parity")
        else:
            # The two convolution gradients are validated against the
            # stable line's own autograd on identical inputs, which is an
            # independent implementation of the same mathematics.
            stable_input = tensorforge.Tensor(images.copy(), requires_grad=True)
            stable_out = stable(stable_input)
            objective = (stable_out
                         * tensorforge.Tensor(upstream.copy())).sum()
            objective.backward()
            if component == "input_backward":
                expected = stable_input.grad
                _require_shape(produced, (n, c, h, w),
                               "the native input gradient")
            else:
                expected = stable.weight.grad
                _require_shape(produced, (o, c, k, k),
                               "the native weight gradient")
            error = _max_abs(produced - expected)
            _require_parity(error, GRADIENT_ATOL * scale,
                            f"the native conv2d {component}",
                            "tensorforge.nn.Conv2d's autograd gradient")
            metrics["max_abs_error"] = error
            checks.append("stable_gradient_parity")

        _require_unchanged(core_images.to_numpy(), images, "the native images")
        _require_unchanged(core_weight.to_numpy(), weight, "the native weight")
        checks.append("no_operand_mutation")
        metrics["checks"] = checks
        metrics["output_shape"] = [n, o, out_h, out_w]
        return metrics

    layers = {TENSOR_CORE: _layer(core_runs[component], cleanup=_close_result)}
    if component == "forward":
        stable = stable_module()
        stable_input = tensorforge.Tensor(images.copy())
        layers[STABLE] = _layer(lambda _s=None: stable(stable_input))
        layers[NATIVE_TENSOR] = _layer(
            lambda _s=None: module(tensor_images), cleanup=_close_result)
        layers[NATIVE_TENSOR_GRAPH] = _layer(
            lambda _s=None: module(tensor_images_g), cleanup=_close_result)

    return {
        "check": check,
        "layers": layers,
        "close": lambda: ([item.close() for item in owned]
                          + [_close_module(module)]),
    }


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------


class _BenchmarkMLP(NativeModule):
    """A deterministic native MLP: Linear -> ReLU -> Linear."""

    def __init__(self, in_features, hidden, out_features, seed):
        super().__init__()
        self.hidden = NativeLinear(in_features, hidden, seed=seed)
        self.relu = NativeReLU()
        self.output = NativeLinear(hidden, out_features, seed=seed + 1)

    def forward(self, inputs):
        return self.output(self.relu(self.hidden(inputs)))


class _BenchmarkCNN(NativeModule):
    """A deterministic native CNN classifier over raw logits."""

    def __init__(self, channels, out_channels, kernel, features, classes, seed):
        super().__init__()
        self.conv = NativeConv2d(channels, out_channels, kernel, seed=seed)
        self.relu = NativeReLU()
        self.pool = NativeMaxPool2d(2)
        self.flatten = NativeFlatten()
        self.head = NativeLinear(features, classes, seed=seed + 1)

    def forward(self, images):
        hidden = self.pool(self.relu(self.conv(images)))
        return self.head(self.flatten(hidden))


class _BenchmarkNormalized(NativeModule):
    """Linear -> BatchNorm1d -> ReLU -> LayerNorm -> Linear: both
    normalization families in every forward, BatchNorm the only stateful
    module."""

    def __init__(self, in_features, hidden, out_features, seed):
        super().__init__()
        self.hidden = NativeLinear(in_features, hidden, seed=seed)
        self.batch_norm = NativeBatchNorm1d(hidden, eps=EPS, momentum=MOMENTUM)
        self.relu = NativeReLU()
        self.layer_norm = NativeLayerNorm(hidden, eps=EPS)
        self.output = NativeLinear(hidden, out_features, seed=seed + 1)

    def forward(self, inputs):
        hidden = self.layer_norm(self.relu(self.batch_norm(self.hidden(inputs))))
        return self.output(hidden)


class _BenchmarkDropoutModel(NativeModule):
    """Linear -> ReLU -> Dropout -> Linear with an **explicit** fixed
    generator, so the stochastic stream is a property of the case rather
    than of the process."""

    def __init__(self, in_features, hidden, out_features, seed, generator):
        super().__init__()
        self.hidden = NativeLinear(in_features, hidden, seed=seed)
        self.relu = NativeReLU()
        self.dropout = NativeDropout(p=DROPOUT_P, generator=generator)
        self.output = NativeLinear(hidden, out_features, seed=seed + 1)

    def forward(self, inputs):
        hidden = self.dropout(self.relu(self.hidden(inputs)))
        return self.output(hidden)


def _build_mlp_training_step(config, spec):
    """One complete deterministic MLP training step:
    ``zero_grad -> forward -> NativeMSELoss -> backward -> Adam.step()``.

    A **fresh** model and optimizer are constructed outside the timer for
    every repetition, so every timed step starts from the same
    deterministic state and no repetition inherits the parameter values,
    moments, or step counters the one before it produced."""
    batch, in_features = config["batch"], config["in_features"]
    hidden, out_features = config["hidden"], config["out_features"]
    seed = spec["seed"]
    inputs = _values((batch, in_features), seed)
    targets = _values((batch, out_features), seed + 1)
    x = NativeTensor.from_array(inputs)
    y = NativeTensor.from_array(targets)
    criterion = NativeMSELoss()

    def build():
        return _BenchmarkMLP(in_features, hidden, out_features, seed)

    initial = _initial_state(build)

    def native_prepare():
        model = build()
        return model, NativeAdam(model.parameters(), lr=LR)

    def native_run(state):
        model, optimizer = state
        optimizer.zero_grad()
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
        _close_optimizer(optimizer)
        _close_module(model)

    def stable_prepare():
        from tensorforge.nn import Linear, ReLU, Sequential
        from tensorforge.optim import Adam

        model = Sequential(Linear(in_features, hidden), ReLU(),
                           Linear(hidden, out_features))
        first, _relu, second = model.modules
        first.weight.data = initial["hidden.weight"].copy()
        first.bias.data = initial["hidden.bias"].copy()
        second.weight.data = initial["output.weight"].copy()
        second.bias.data = initial["output.bias"].copy()
        return model, Adam(model.parameters(), lr=LR)

    def stable_run(state):
        from tensorforge.nn import mse_loss

        model, optimizer = state
        optimizer.zero_grad()
        prediction = model(tensorforge.Tensor(inputs.copy()))
        loss = mse_loss(prediction, targets)
        loss.backward()
        optimizer.step()
        return loss

    def check():
        metrics = _training_step_gate(
            native_prepare, native_run, native_cleanup,
            parameter_keys=("hidden.weight", "hidden.bias",
                            "output.weight", "output.bias"),
            buffer_keys=(),
        )
        stable_state = stable_prepare()
        stable_model, _ = stable_state
        stable_loss = stable_run(stable_state)
        loss_error = abs(float(stable_loss.data) - metrics["loss"])
        _require_parity(loss_error, LOSS_ATOL * max(1.0, metrics["loss"]),
                        "the native training-step loss",
                        "the equivalently initialized stable model's loss")
        first, _relu, second = stable_model.modules
        stable_after = {
            "hidden.weight": first.weight.data,
            "hidden.bias": first.bias.data,
            "output.weight": second.weight.data,
            "output.bias": second.bias.data,
        }
        parameter_error = max(
            _max_abs(metrics["parameters_after"][name] - stable_after[name])
            for name in stable_after
        )
        _require_parity(parameter_error, PARAMETER_ATOL,
                        "the native post-step parameters",
                        "the stable model's post-step parameters")
        metrics["loss_max_abs_error"] = loss_error
        metrics["max_abs_error"] = parameter_error
        metrics["checks"].append("stable_loss_parity")
        metrics["checks"].append("stable_parameter_parity")
        metrics.pop("parameters_after")
        metrics.pop("buffers_after")
        return metrics

    return {
        "check": check,
        "layers": {
            STABLE: _layer(stable_run, prepare=stable_prepare),
            TRAINING_STEP: _layer(native_run, prepare=native_prepare,
                                  cleanup=native_cleanup),
        },
        "close": lambda: (x.close(), y.close()),
    }


def _initial_state(build):
    """The deterministic initial parameter and buffer values a freshly
    built model starts from, read once outside every timed region."""
    model = build()
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


def _training_step_gate(prepare, run, cleanup, parameter_keys, buffer_keys,
                        generator=None):
    """The shared training-step correctness gate.

    Proves the step really trained: a finite scalar loss, a finite
    correctly shaped gradient on every parameter, an advanced optimizer
    step counter, at least one changed parameter, every declared buffer
    advanced, the completed graph released with no saved resource left
    behind, and — when the model owns a generator — exactly one call
    consumed."""
    state = prepare()
    model, optimizer = state
    named = list(model.named_parameters())
    _require(tuple(name for name, _ in named) == tuple(parameter_keys),
             f"the model's parameter order is "
             f"{tuple(name for name, _ in named)}, expected "
             f"{tuple(parameter_keys)}")
    _require(tuple(name for name, _ in model.named_buffers())
             == tuple(buffer_keys),
             "the model's buffer order changed")
    buffer_ids = {id(buffer) for buffer in model.buffers()}
    _require(not (buffer_ids & {id(p) for p in optimizer.parameters()}),
             "a registered buffer reached the optimizer")
    before = {name: parameter.to_numpy().copy() for name, parameter in named}
    buffers_before = {name: buffer.to_numpy().copy()
                      for name, buffer in model.named_buffers()}
    calls_before = generator.calls if generator is not None else None
    result = None
    try:
        result = run(state)
        prediction, loss = result
        _require(loss.shape == (), "the training-step loss is not scalar")
        loss_value = float(loss.to_numpy())
        _require(np.isfinite(loss_value), "the training-step loss is not finite")
        for name, parameter in named:
            _require(parameter.grad is not None, f"{name} received no gradient")
            gradient = parameter.grad.to_numpy().copy()
            _require_shape(gradient, parameter.shape, f"the {name} gradient")
            _require_finite(gradient, f"the {name} gradient")
        for name, buffer in model.named_buffers():
            _require(not np.array_equal(buffer.to_numpy(), buffers_before[name]),
                     f"the persistent buffer {name} did not advance")
        steps = list(optimizer.step_counts)
        _require(steps == [1] * len(steps),
                 f"the optimizer step counts did not advance: {steps}")
        after = {name: parameter.to_numpy().copy() for name, parameter in named}
        _require(any(not np.array_equal(after[name], before[name])
                     for name in after),
                 "no parameter changed during the step")
        _require(loss._graph_freed, "the completed graph was not released")
        _require(not loss._graph_resources and not prediction._graph_resources,
                 "a graph resource survived the step")
        if generator is not None:
            _require(generator.calls == calls_before + 1,
                     f"the step consumed {generator.calls - calls_before} "
                     f"generator calls, expected exactly 1")
        buffers_after = {name: buffer.to_numpy().copy()
                         for name, buffer in model.named_buffers()}
    finally:
        cleanup(state, result)
    checks = ["scalar_finite_loss", "all_gradients_present", "gradient_shapes",
              "finite_gradients", "optimizer_step_advanced",
              "parameters_changed", "graph_released",
              "no_graph_resource_survives", "buffers_excluded_from_optimizer"]
    if buffer_keys:
        checks.append("persistent_buffers_advanced")
    if generator is not None:
        checks.append("exactly_one_generator_call_consumed")
    return {"loss": loss_value, "parameters_after": after,
            "buffers_after": buffers_after, "checks": checks}


def _build_cnn_training_step(config, spec):
    """One complete deterministic CNN classification training step over
    **raw logits** into ``NativeCrossEntropyLoss``."""
    n, c, h, w, o, k = _conv_shapes(config)
    classes = config["classes"]
    seed = spec["seed"]
    images = _values((n, c, h, w), seed)
    labels = [index % classes for index in range(n)]
    x = NativeTensor.from_array(images)
    criterion = NativeCrossEntropyLoss()
    pooled = ((h - k + 1) // 2) * ((w - k + 1) // 2) * o

    def build():
        return _BenchmarkCNN(c, o, k, pooled, classes, seed)

    def native_prepare():
        model = build()
        return model, NativeAdam(model.parameters(), lr=LR)

    def native_run(state):
        model, optimizer = state
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        return logits, loss

    def native_cleanup(state, result):
        model, optimizer = state
        if result is not None:
            logits, loss = result
            loss.close()
            logits.close()
        _release_gradients(list(model.parameters()))
        _close_optimizer(optimizer)
        _close_module(model)

    def check():
        metrics = _training_step_gate(
            native_prepare, native_run, native_cleanup,
            parameter_keys=("conv.weight", "conv.bias",
                            "head.weight", "head.bias"),
            buffer_keys=(),
        )
        # The loss must beat a uniform-prediction baseline's ceiling only
        # in the sense of being a real cross-entropy value: finite,
        # positive, and of the right magnitude for this class count.
        _require(0.0 < metrics["loss"] < 50.0,
                 f"the cross-entropy loss {metrics['loss']:g} is not a "
                 f"plausible value for {classes} classes")
        metrics["max_abs_error"] = 0.0
        metrics["classes"] = classes
        metrics["checks"].append("plausible_cross_entropy_magnitude")
        metrics.pop("parameters_after")
        metrics.pop("buffers_after")
        return metrics

    return {
        "check": check,
        "layers": {
            TRAINING_STEP: _layer(native_run, prepare=native_prepare,
                                  cleanup=native_cleanup),
        },
        "close": lambda: x.close(),
    }


def _build_normalized_training_step(config, spec):
    """One complete deterministic normalization-heavy training step.

    BatchNorm advances persistent running statistics on every training
    forward, so a **fresh** model is built outside the timer for every
    repetition — a state-advanced module is never reused as a measured
    sample."""
    batch, in_features = config["batch"], config["in_features"]
    hidden, out_features = config["hidden"], config["out_features"]
    seed = spec["seed"]
    inputs = _values((batch, in_features), seed)
    targets = _values((batch, out_features), seed + 1)
    x = NativeTensor.from_array(inputs)
    y = NativeTensor.from_array(targets)
    criterion = NativeMSELoss()

    def build():
        return _BenchmarkNormalized(in_features, hidden, out_features, seed)

    initial = _initial_state(build)

    def native_prepare():
        model = build()
        model.train()
        return model, NativeAdam(model.parameters(), lr=LR)

    def native_run(state):
        model, optimizer = state
        optimizer.zero_grad()
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
        _close_optimizer(optimizer)
        _close_module(model)

    def stable_prepare():
        from tensorforge.nn import (BatchNorm1d, LayerNorm, Linear, ReLU,
                                    Sequential)
        from tensorforge.optim import Adam

        model = Sequential(Linear(in_features, hidden),
                           BatchNorm1d(hidden, eps=EPS, momentum=MOMENTUM),
                           ReLU(), LayerNorm(hidden, eps=EPS),
                           Linear(hidden, out_features))
        first, batch_norm, _relu, layer_norm, second = model.modules
        first.weight.data = initial["hidden.weight"].copy()
        first.bias.data = initial["hidden.bias"].copy()
        batch_norm.gamma.data = initial["batch_norm.gamma"].copy()
        batch_norm.beta.data = initial["batch_norm.beta"].copy()
        batch_norm.running_mean = initial["batch_norm.running_mean"].copy()
        batch_norm.running_var = initial["batch_norm.running_var"].copy()
        layer_norm.weight.data = initial["layer_norm.weight"].copy()
        layer_norm.bias.data = initial["layer_norm.bias"].copy()
        second.weight.data = initial["output.weight"].copy()
        second.bias.data = initial["output.bias"].copy()
        model.train()
        return model, Adam(model.parameters(), lr=LR)

    def stable_run(state):
        from tensorforge.nn import mse_loss

        model, optimizer = state
        optimizer.zero_grad()
        prediction = model(tensorforge.Tensor(inputs.copy()))
        loss = mse_loss(prediction, targets)
        loss.backward()
        optimizer.step()
        return loss

    def check():
        metrics = _training_step_gate(
            native_prepare, native_run, native_cleanup,
            parameter_keys=("hidden.weight", "hidden.bias",
                            "batch_norm.gamma", "batch_norm.beta",
                            "layer_norm.weight", "layer_norm.bias",
                            "output.weight", "output.bias"),
            buffer_keys=("batch_norm.running_mean", "batch_norm.running_var"),
        )
        stable_state = stable_prepare()
        stable_model, _ = stable_state
        stable_loss = stable_run(stable_state)
        loss_error = abs(float(stable_loss.data) - metrics["loss"])
        _require_parity(loss_error, LOSS_ATOL * max(1.0, metrics["loss"]),
                        "the native normalized training-step loss",
                        "the equivalently initialized stable model's loss")
        _first, batch_norm, _relu, _layer_norm, _second = stable_model.modules
        state_error = max(
            _max_abs(batch_norm.running_mean
                     - metrics["buffers_after"]["batch_norm.running_mean"]),
            _max_abs(batch_norm.running_var
                     - metrics["buffers_after"]["batch_norm.running_var"]),
        )
        _require_parity(state_error, FORWARD_ATOL,
                        "the native running statistics",
                        "tensorforge.nn.BatchNorm1d's running statistics")
        metrics["loss_max_abs_error"] = loss_error
        metrics["running_state_max_abs_error"] = state_error
        metrics["max_abs_error"] = max(loss_error, state_error)
        metrics["checks"].append("stable_loss_parity")
        metrics["checks"].append("stable_running_state_parity")
        metrics.pop("parameters_after")
        metrics.pop("buffers_after")
        return metrics

    return {
        "check": check,
        "layers": {
            STABLE: _layer(stable_run, prepare=stable_prepare),
            TRAINING_STEP: _layer(native_run, prepare=native_prepare,
                                  cleanup=native_cleanup),
        },
        "close": lambda: (x.close(), y.close()),
    }


def _build_dropout_training_step(config, spec):
    """One complete deterministic training step through ``NativeDropout``
    with an **explicit** fixed ``NativeGenerator``.

    The generator is reset to call index 0 in the untimed ``prepare`` for
    every repetition, so every timed step draws the *same* mask from the
    *same* reserved call index — benchmark setup can never shift the
    index a timed call consumes. The gate proves exactly one call is
    consumed per successful step.

    This case is ``native_only`` on purpose. The stable line's Dropout
    draws from NumPy's global RNG with a completely different algorithm,
    so no equivalently seeded comparison exists; claiming exact equality
    against it would be dishonest, and publishing a timing ratio against
    a different random implementation would be misleading. The correctness
    reference is the native derivation itself — ``dropout_forward`` at the
    same ``(seed, call_index)`` — plus the structural mask properties."""
    batch, in_features = config["batch"], config["in_features"]
    hidden, out_features = config["hidden"], config["out_features"]
    seed = spec["seed"]
    inputs = _values((batch, in_features), seed)
    targets = _values((batch, out_features), seed + 1)
    x = NativeTensor.from_array(inputs)
    y = NativeTensor.from_array(targets)
    criterion = NativeMSELoss()
    generator = NativeGenerator(seed=DROPOUT_SEED)

    def native_prepare():
        # Untimed: rewind the stream so every timed step consumes the same
        # call index and therefore produces the same mask.
        generator.reset()
        model = _BenchmarkDropoutModel(in_features, hidden, out_features, seed,
                                       generator)
        model.train()
        return model, NativeAdam(model.parameters(), lr=LR)

    def native_run(state):
        model, optimizer = state
        optimizer.zero_grad()
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
        _close_optimizer(optimizer)
        _close_module(model)

    def check():
        generator.reset()
        _require(generator.calls == 0, "reset() did not rewind the stream")
        metrics = _training_step_gate(
            native_prepare, native_run, native_cleanup,
            parameter_keys=("hidden.weight", "hidden.bias",
                            "output.weight", "output.bias"),
            buffer_keys=(),
            generator=generator,
        )

        # The stream is the case's own: the module registers exactly the
        # generator it was given, never a copy.
        generator.reset()
        model = _BenchmarkDropoutModel(in_features, hidden, out_features, seed,
                                       generator)
        try:
            _require(model.dropout.generator is generator,
                     "the module registered a different generator object")
            names = [name for name, _ in model.named_generators()]
            _require(names == ["dropout.generator"],
                     f"the registered generator paths are {names}")

            # Evaluation is state-neutral: it consumes no call and returns
            # the input object itself.
            model.eval()
            calls_before = generator.calls
            hidden_values = NativeTensor.from_array(_values((batch, hidden),
                                                            seed + 7))
            try:
                evaluated = model.dropout(hidden_values)
                _require(evaluated is hidden_values,
                         "evaluation-mode Dropout did not return its input")
                _require(generator.calls == calls_before,
                         "evaluation-mode Dropout consumed a generator call")
            finally:
                hidden_values.close()

            # The mask really is inverted Dropout at the reserved index,
            # verified against the Core derivation at the *same* key.
            model.train()
            reserved_index = generator.calls
            probe_values = _values((batch, hidden), seed + 8)
            probe = NativeTensor.from_array(probe_values)
            core_probe = cpp.NativeTensorCore.from_array(probe_values)
            try:
                produced = model.dropout(probe)
                try:
                    module_output = produced.to_numpy().copy()
                finally:
                    produced.close()
                expected_core = core_probe.dropout_forward(
                    DROPOUT_P, seed=generator.seed, call_index=reserved_index)
                try:
                    expected = expected_core.to_numpy().copy()
                finally:
                    expected_core.close()
            finally:
                probe.close()
                core_probe.close()
            derivation_error = _max_abs(module_output - expected)
            _require_parity(derivation_error, EXACT,
                            "the module's Dropout output",
                            "NativeTensorCore.dropout_forward at the same "
                            "(seed, call_index)")
            _require(generator.calls == reserved_index + 1,
                     "the probe forward did not consume exactly one call")

            # Structural mask properties: inverted scaling, and both
            # outcomes actually present at p = 0.5 on this sample.
            scale = 1.0 / (1.0 - DROPOUT_P)
            ratio = np.where(probe_values != 0.0,
                             module_output / np.where(probe_values == 0.0, 1.0,
                                                      probe_values),
                             0.0)
            unique = np.unique(np.round(ratio, 12))
            _require(set(unique.tolist()) <= {0.0, round(scale, 12)},
                     f"the Dropout multiplier took unexpected values: "
                     f"{unique.tolist()}")
            _require(float(np.max(np.abs(module_output))) > 0.0,
                     "the Dropout output is entirely zero")
            _require(bool(np.any(module_output == 0.0)),
                     "no element was dropped at p = 0.5")
        finally:
            _close_module(model)

        metrics["max_abs_error"] = derivation_error
        metrics["dropout_p"] = DROPOUT_P
        metrics["generator_seed"] = generator.seed
        metrics["checks"].extend([
            "registered_generator_identity", "registered_generator_path",
            "evaluation_is_state_neutral", "core_derivation_parity",
            "inverted_dropout_scaling", "both_outcomes_present",
        ])
        metrics.pop("parameters_after")
        metrics.pop("buffers_after")
        return metrics

    return {
        "check": check,
        "layers": {
            TRAINING_STEP: _layer(native_run, prepare=native_prepare,
                                  cleanup=native_cleanup),
        },
        "close": lambda: (x.close(), y.close()),
    }


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


def _build_optimizer_step(config, spec):
    """One optimizer ``step()`` in isolation, with gradients already
    present.

    The forward and backward that produce those gradients happen **once,
    outside** the timer; the optimizer retains gradients until
    ``zero_grad()``, so every repetition steps against the same gradient
    values. That is the point: this case measures the optimizer's own
    cost — its arithmetic, its allocations, and its Python/native call
    count — with no model work mixed in."""
    batch, in_features = config["batch"], config["in_features"]
    hidden, out_features = config["hidden"], config["out_features"]
    seed = spec["seed"]
    optimizer_name = spec["optimizer"]
    inputs = _values((batch, in_features), seed)
    targets = _values((batch, out_features), seed + 1)
    x = NativeTensor.from_array(inputs)
    y = NativeTensor.from_array(targets)
    criterion = NativeMSELoss()
    model = _BenchmarkMLP(in_features, hidden, out_features, seed)
    initial = {name: parameter.to_numpy().copy()
               for name, parameter in model.named_parameters()}

    # One untimed forward/backward supplies the gradients every timed
    # step consumes. They are never cleared, so every repetition sees the
    # same gradient values.
    prediction = model(x)
    loss = criterion(prediction, y)
    loss.backward()
    gradients = {name: parameter.grad.to_numpy().copy()
                 for name, parameter in model.named_parameters()}
    loss.close()
    prediction.close()

    factory = NativeAdam if optimizer_name == "adam" else NativeSGD
    optimizer = factory(model.parameters(), lr=LR)

    stable_factory_name = "Adam" if optimizer_name == "adam" else "SGD"

    def build_stable():
        import tensorforge.optim as optim
        from tensorforge.nn import Parameter

        parameters = []
        for name, _ in model.named_parameters():
            parameter = Parameter(initial[name].copy())
            parameter.grad = gradients[name].copy()
            parameters.append(parameter)
        stable = getattr(optim, stable_factory_name)(parameters, lr=LR)
        return stable, parameters

    def check():
        # A completely separate model/optimizer pair, so the gate never
        # advances the state the timed layer uses.
        probe_model = _BenchmarkMLP(in_features, hidden, out_features, seed)
        probe_prediction = probe_model(x)
        probe_loss = criterion(probe_prediction, y)
        probe_loss.backward()
        probe_optimizer = factory(probe_model.parameters(), lr=LR)
        try:
            probe_gradients = {
                name: parameter.grad.to_numpy().copy()
                for name, parameter in probe_model.named_parameters()
            }
            gradient_error = max(
                _max_abs(probe_gradients[name] - gradients[name])
                for name in gradients
            )
            _require_parity(gradient_error, EXACT,
                            "the probe's gradients",
                            "the gradients the timed layer steps against")
            versions = {name: parameter.version
                        for name, parameter in probe_model.named_parameters()}
            probe_optimizer.step()
            after = {name: parameter.to_numpy().copy()
                     for name, parameter in probe_model.named_parameters()}
            for name, parameter in probe_model.named_parameters():
                _require(parameter.version == versions[name] + 1,
                         f"{name} moved {parameter.version - versions[name]} "
                         f"versions, expected exactly 1")
                _require(parameter.grad is not None,
                         f"{name} lost its gradient during step()")
                _require_finite(after[name], f"the updated {name}")
                _require(not np.array_equal(after[name], initial[name]),
                         f"{name} did not change during step()")

            stable, stable_parameters = build_stable()
            stable.step()
            stable_after = {name: parameter.data for (name, _), parameter
                            in zip(model.named_parameters(),
                                   stable_parameters)}
            parameter_error = max(_max_abs(after[name] - stable_after[name])
                                  for name in after)
            _require_parity(parameter_error, PARAMETER_ATOL,
                            f"the native {optimizer_name} update",
                            f"tensorforge.optim.{stable_factory_name} on the "
                            f"same parameters, gradients, and learning rate")
        finally:
            _close_optimizer(probe_optimizer)
            _release_gradients(list(probe_model.parameters()))
            probe_loss.close()
            probe_prediction.close()
            _close_module(probe_model)
        return {
            "max_abs_error": parameter_error,
            "gradient_max_abs_error": gradient_error,
            "optimizer": optimizer_name,
            "parameter_count": len(initial),
            "checks": ["gradients_match_the_timed_state",
                       "exactly_one_version_per_parameter",
                       "gradients_retained_through_step", "finite_update",
                       "parameters_changed", "stable_optimizer_parity"],
        }

    def stable_prepare():
        return build_stable()

    def stable_run(state):
        stable, _ = state
        stable.step()
        return None

    return {
        "check": check,
        "layers": {
            STABLE: _layer(stable_run, prepare=stable_prepare),
            OPTIMIZER_STEP: _layer(lambda _s=None: optimizer.step()),
        },
        "close": lambda: (_close_optimizer(optimizer),
                          _release_gradients(list(model.parameters())),
                          _close_module(model), x.close(), y.close()),
    }


# ---------------------------------------------------------------------------
# State operations (in-memory only — see the module docstring)
# ---------------------------------------------------------------------------


def _build_state_operations(config, spec):
    """The in-memory state surface: one ``state_dict()`` snapshot and one
    ``load_state_dict()`` commit, measured separately from any training
    step.

    File checkpoint I/O is deliberately not measured here — see the module
    docstring."""
    batch, in_features = config["batch"], config["in_features"]
    hidden, out_features = config["hidden"], config["out_features"]
    seed = spec["seed"]
    del batch
    model = _BenchmarkNormalized(in_features, hidden, out_features, seed)
    snapshot = model.state_dict()
    section = spec["section_key"]

    def snapshot_run(_state=None):
        return model.state_dict()

    def snapshot_cleanup(_state, result):
        if result is not None:
            for tensor in result.values():
                tensor.close()

    def load_run(_state=None):
        model.load_state_dict(snapshot)
        return None

    def check():
        values = model.state_dict()
        try:
            keys = tuple(values)
            _require(keys == tuple(snapshot),
                     f"the state keys changed: {keys}")
            _require(all(isinstance(tensor, NativeTensor)
                         for tensor in values.values()),
                     "state_dict() returned a non-NativeTensor value")
            live = {id(parameter) for parameter in model.parameters()}
            live |= {id(buffer) for buffer in model.buffers()}
            _require(not (live & {id(tensor) for tensor in values.values()}),
                     "state_dict() handed back a live registered object")
            error = max(_max_abs(values[name].to_numpy()
                                 - snapshot[name].to_numpy())
                        for name in keys)
            _require_parity(error, EXACT, "the fresh snapshot",
                            "the snapshot the load layer commits")
        finally:
            for tensor in values.values():
                tensor.close()
        identities = {name: id(parameter)
                      for name, parameter in model.named_parameters()}
        versions = {name: parameter.version
                    for name, parameter in model.named_parameters()}
        model.load_state_dict(snapshot)
        for name, parameter in model.named_parameters():
            _require(id(parameter) == identities[name],
                     f"{name} lost its identity across load_state_dict")
            _require(parameter.version == versions[name] + 1,
                     f"{name} moved {parameter.version - versions[name]} "
                     f"versions, expected exactly 1")
        after = model.state_dict()
        try:
            reload_error = max(_max_abs(after[name].to_numpy()
                                        - snapshot[name].to_numpy())
                               for name in snapshot)
            _require_parity(reload_error, EXACT, "the reloaded state",
                            "the snapshot it was loaded from")
        finally:
            for tensor in after.values():
                tensor.close()
        return {
            "max_abs_error": max(error, reload_error),
            "state_key_count": len(snapshot),
            "section": section,
            "checks": ["stable_key_order", "native_tensor_values",
                       "snapshot_is_independent", "snapshot_parity",
                       "identity_preserved_across_load",
                       "exactly_one_version_per_parameter",
                       "reloaded_state_parity"],
        }

    runs = {"snapshot": (snapshot_run, snapshot_cleanup),
            "load": (load_run, _drop_result)}
    run, cleanup = runs[section]
    return {
        "check": check,
        "layers": {TENSOR_CORE: _layer(run, cleanup=cleanup)},
        "close": lambda: ([tensor.close() for tensor in snapshot.values()]
                          + [_close_module(model)]),
    }


# ---------------------------------------------------------------------------
# The case registry
# ---------------------------------------------------------------------------

_NO_STABLE_DROPOUT = (
    "no honest equivalent exists: the stable tensorforge Dropout draws "
    "from NumPy's global RNG with a completely different algorithm, so "
    "there is no equivalently seeded comparison and no timing ratio is "
    "published. Claiming exact equality against it would be dishonest. "
    "The correctness gate is real: the module's mask is verified against "
    "NativeTensorCore.dropout_forward at the same (seed, call_index), the "
    "inverted scaling and both outcomes are checked structurally, and "
    "exactly one generator call per successful step is proved."
)

_NO_STABLE_CNN_STEP = (
    "no honest equivalent exists: the stable line has no fused native "
    "cross-entropy over raw logits with the same saved-probability "
    "backward, and NativeMaxPool2d's winner-index representation has no "
    "stable counterpart, so a step-for-step timing comparison would "
    "measure two different algorithms. No ratio is published. The "
    "correctness gate is real: a finite scalar loss of a plausible "
    "magnitude, finite gradients on every parameter, one optimizer step, "
    "a changed parameter set, and a fully released graph."
)

_NO_STABLE_ALLOCATION = (
    "numpy.zeros is the honest reference for the same request, and it is "
    "measured; there is no stable tensorforge equivalent of a bare native "
    "storage allocation."
)

_NO_STABLE_STATE = (
    "no honest equivalent exists: the stable line's state_dict returns "
    "NumPy arrays with no ownership, versioning, or transaction "
    "semantics, so timing it against the native atomic loader would "
    "compare two different contracts. No ratio is published."
)


CASES = {
    # -- dispatch overhead ---------------------------------------------------
    "scalar_dispatch_overhead": {
        "workload": "dispatch_overhead",
        "section": "add on a one-element tensor",
        "operation": ("add() on a (1, 1) tensor across NumPy, "
                      "NativeTensorCore, NativeTensor, and NativeTensor with "
                      "graph construction"),
        "build": _build_scalar_dispatch,
        "seed": 20260201,
        "reference_type": REFERENCE_NUMPY,
        "reference_layer": NUMPY,
        "reference_detail": "numpy addition of the same one-element arrays",
        "correctness_reference": "NumPy elementwise addition, exactly",
        "configurations": {
            "full": {"shape": (1, 1)},
            "smoke": {"shape": (1, 1)},
            "profile": {"shape": (1, 1)},
        },
        "notes": ("The arithmetic is one addition, so what is measured is "
                  "the fixed cost of reaching the kernel: Python shape and "
                  "stride normalization, the fresh owning allocation, the "
                  "ctypes boundary, and the wrapper/graph objects. This is "
                  "the case that decides whether small-operation overhead is "
                  "material. The shape is the same in every configuration "
                  "because a larger one would measure something else."),
    },
    "storage_allocation": {
        "allocation_layers": (TENSOR_CORE_ZEROED,),
        "workload": "dispatch_overhead",
        "section": "native storage allocation and release",
        "operation": ("NativeStorage(size) construction and close(), with no "
                      "compute at all"),
        "build": _build_storage_allocation,
        "seed": 20260202,
        "reference_type": REFERENCE_NUMPY,
        "reference_layer": NUMPY,
        "reference_detail": "numpy.zeros of the same element count",
        "correctness_reference": ("the documented zero-initialization "
                                  "contract of NativeStorage and "
                                  "NativeTensorCore.zeros"),
        "configurations": {
            "full": {"shape": (512, 512)},
            "smoke": {"shape": (16, 16)},
            # 2048x2048 float64 is 32 MB — one buffer, comfortably inside
            # the design's working-set ceiling, and large enough that the
            # timed region clears the profile-shape target.
            "profile": {"shape": (2048, 2048)},
        },
        "reference_note": _NO_STABLE_ALLOCATION,
        "notes": ("Every native operation allocates a fresh owning output, "
                  "so this is the fixed tax on every result the stack "
                  "produces. Native storage is value-initialized on "
                  "construction (`new double[n]()`), which is a full write "
                  "pass over the buffer; numpy.zeros answers the same "
                  "request through a different allocator strategy. Measured, "
                  "not judged."),
    },

    # -- elementwise ---------------------------------------------------------
    "elementwise_contiguous": {
        "allocation_layers": (TENSOR_CORE_ZEROED,),
        "workload": "elementwise",
        "section": "multiply, both operands contiguous",
        "operation": ("multiply() with two row-major contiguous operands "
                      "(the flat fast-path kernel)"),
        "build": _build_elementwise,
        "strided": False,
        "seed": 20260203,
        "reference_type": REFERENCE_NUMPY,
        "reference_layer": NUMPY,
        "reference_detail": "numpy multiplication of the same arrays",
        "correctness_reference": "NumPy elementwise multiplication, exactly",
        "configurations": {
            "full": {"shape": (256, 256)},
            "smoke": {"shape": (16, 16)},
            "profile": {"shape": (1024, 1024)},
        },
        "notes": ("The contiguous fast path: a flat, index-free loop. Paired "
                  "with elementwise_transposed_view, which carries the same "
                  "logical values through the generic odometer, so the two "
                  "cases isolate the traversal difference alone."),
    },
    "elementwise_transposed_view": {
        "workload": "elementwise",
        "section": "multiply, transposed-view left operand",
        "operation": ("multiply() with a transposed-view left operand (the "
                      "generic odometer kernel)"),
        "build": _build_elementwise,
        "strided": True,
        "seed": 20260204,
        "reference_type": REFERENCE_NUMPY,
        "reference_layer": NUMPY,
        "reference_detail": ("numpy multiplication through the same "
                             "transposed view"),
        "correctness_reference": "NumPy elementwise multiplication, exactly",
        "configurations": {
            "full": {"shape": (256, 256)},
            "smoke": {"shape": (16, 16)},
            "profile": {"shape": (1024, 1024)},
        },
        "notes": ("A real transposed view, not a copy: the same logical "
                  "values as elementwise_contiguous, reached through the "
                  "strided odometer. Nothing is materialized first."),
    },

    # -- reductions ----------------------------------------------------------
    "reduction_contiguous": {
        "workload": "reduction",
        "section": "sum over axis 0, contiguous operand",
        "operation": "sum(axis=0) over a row-major contiguous operand",
        "build": _build_reduction,
        "strided": False,
        "axis": 0,
        "seed": 20260205,
        "reference_type": REFERENCE_NUMPY,
        "reference_layer": NUMPY,
        "reference_detail": "numpy.sum over the same axis",
        "correctness_reference": ("NumPy's sum over the same axis, to a "
                                  "tolerance (float summation is "
                                  "order-sensitive)"),
        "configurations": {
            "full": {"shape": (256, 256)},
            "smoke": {"shape": (16, 16)},
            "profile": {"shape": (1024, 1024)},
        },
        "notes": ("The scatter-accumulate reduction: the input is walked "
                  "with the odometer while an output position advances "
                  "through zero strides on the reduced axis. There is no "
                  "contiguous fast path for reductions, which is exactly "
                  "what this pair is here to characterize."),
    },
    "reduction_transposed_view": {
        "workload": "reduction",
        "section": "sum over axis 0, transposed-view operand",
        "operation": "sum(axis=0) over a transposed view",
        "build": _build_reduction,
        "strided": True,
        "axis": 0,
        "seed": 20260206,
        "reference_type": REFERENCE_NUMPY,
        "reference_layer": NUMPY,
        "reference_detail": "numpy.sum over the same axis of the same view",
        "correctness_reference": ("NumPy's sum over the same axis, to a "
                                  "tolerance"),
        "configurations": {
            "full": {"shape": (256, 256)},
            "smoke": {"shape": (16, 16)},
            "profile": {"shape": (1024, 1024)},
        },
        "notes": ("The same logical values as reduction_contiguous, reached "
                  "through a transposed view, so the accumulation order and "
                  "the memory access pattern both change while the "
                  "mathematics does not."),
    },

    # -- matmul --------------------------------------------------------------
    "matmul_square_contiguous": {
        "allocation_layers": (TENSOR_CORE_ZEROED,),
        "workload": "matmul",
        "section": "square matmul, both operands contiguous",
        "operation": "(n, n) @ (n, n) with two contiguous operands",
        "build": _build_matmul,
        "strided_rhs": False,
        "seed": 20260207,
        "reference_type": REFERENCE_NUMPY,
        "reference_layer": NUMPY,
        "reference_detail": "numpy.matmul on the same operands",
        "correctness_reference": ("NumPy's matmul on the same operands, to a "
                                  "tolerance (float accumulation is "
                                  "order-sensitive)"),
        "configurations": {
            "full": {"m": 128, "n": 128, "p": 128, "block": 32},
            "smoke": {"m": 16, "n": 16, "p": 16, "block": 8},
            "profile": {"m": 384, "n": 384, "p": 384, "block": 32},
        },
        "repetitions": HEAVY_REPETITIONS,
        "notes": ("The shape NativeLinear actually produces, and the layout "
                  "H2's row sweep was chosen for. Four native readings sit "
                  "beside each other. `raw_kernel` (cpp.matmul) is the "
                  "**pre-H2 loop order** over contiguous buffers, so "
                  "tensor_core against it is the honest 'what did H2 buy' "
                  "figure. `raw_kernel_tiled` (cpp.matmul_tiled) is the "
                  "standing cache-blocking experiment H2 measured and did "
                  "**not** adopt. `tensor_core_generic` is the same "
                  "production call routed to the retained generic path by a "
                  "strided operand, which is that kernel's *best* case — so "
                  "it understates the H2 change and answers a different "
                  "question: is the fallback sound? Neither raw kernel is on "
                  "any production path."),
    },
    "matmul_rectangular_contiguous": {
        "workload": "matmul",
        "section": "rectangular matmul, both operands contiguous",
        "operation": "(m, n) @ (n, p) with m != n != p, contiguous operands",
        "build": _build_matmul,
        "strided_rhs": False,
        "seed": 20260208,
        "reference_type": REFERENCE_NUMPY,
        "reference_layer": NUMPY,
        "reference_detail": "numpy.matmul on the same operands",
        "correctness_reference": ("NumPy's matmul on the same operands, to "
                                  "a tolerance"),
        "configurations": {
            "full": {"m": 64, "n": 256, "p": 32, "block": 32},
            "smoke": {"m": 8, "n": 24, "p": 6, "block": 8},
            "profile": {"m": 256, "n": 768, "p": 128, "block": 32},
        },
        "repetitions": HEAVY_REPETITIONS,
        "notes": ("Deliberately unequal dimensions, which is the shape a "
                  "real batch @ weight product has. A kernel tuned only on "
                  "square matrices can hide a dimension-dependent cost, so "
                  "the square and rectangular cases are separate — and H2's "
                  "row block and column threshold were both chosen against "
                  "rectangular shapes as well as square ones."),
    },
    "matmul_transposed_view": {
        "workload": "matmul",
        "section": "matmul with a transposed-view right operand",
        "operation": ("(n, n) @ (n, n) where the right operand is a real "
                      "transposed view (the strided path)"),
        "build": _build_matmul,
        "strided_rhs": True,
        "seed": 20260209,
        "reference_type": REFERENCE_NUMPY,
        "reference_layer": NUMPY,
        "reference_detail": ("numpy.matmul on the same logical operands "
                             "(NumPy materializes internally)"),
        "correctness_reference": ("NumPy's matmul on the same logical "
                                  "operands, to a tolerance"),
        "configurations": {
            "full": {"m": 128, "n": 128, "p": 128, "block": 32},
            "smoke": {"m": 16, "n": 16, "p": 16, "block": 8},
            "profile": {"m": 384, "n": 384, "p": 384, "block": 32},
        },
        "repetitions": HEAVY_REPETITIONS,
        "notes": ("The strided fallback, carrying the *same logical values* "
                  "as matmul_square_contiguous. tf_core_matmul addresses "
                  "both operands through their own strides, so nothing is "
                  "materialized. This is the matmul backward's real shape: "
                  "`upstream @ b.T` and `a.T @ upstream` both feed the "
                  "kernel a transposed view. The pair isolates the access "
                  "pattern with the arithmetic held constant. Phase H "
                  "(H2): a transposed *right* operand does not meet the "
                  "row sweep's precondition, so this case measures the "
                  "retained generic path — deliberately, because that loop "
                  "order is the better one here and the fallback is a "
                  "design choice rather than a gap. A transposed *left* "
                  "operand beside a contiguous right one, which is the "
                  "other half of the backward, does take the row sweep."),
    },

    # -- materialization -----------------------------------------------------
    "contiguous_materialization": {
        "allocation_layers": (TENSOR_CORE_ZEROED,),
        "workload": "materialization",
        "section": "contiguous_copy of a transposed view",
        "operation": ("contiguous_copy() materializing a transposed view "
                      "into a fresh owning row-major result"),
        "build": _build_materialization,
        "seed": 20260210,
        "reference_type": REFERENCE_NUMPY,
        "reference_layer": NUMPY,
        "reference_detail": "numpy.ascontiguousarray of the same view",
        "correctness_reference": ("numpy.ascontiguousarray of the same view, "
                                  "exactly"),
        "configurations": {
            "full": {"shape": (256, 256)},
            "smoke": {"shape": (16, 16)},
            "profile": {"shape": (1024, 1024)},
        },
        "notes": ("The Policy-B copy the Core layer performs whenever a "
                  "contiguous-only kernel (conv2d, pooling, softmax, "
                  "cross-entropy, dropout) meets a strided operand. It also "
                  "runs inside NativeFlatten on every CNN forward."),
    },

    # -- linear --------------------------------------------------------------
    "linear_forward": {
        "allocation_layers": (NATIVE_TENSOR_GRAPH_ZEROED,),
        "workload": "linear",
        "section": "NativeLinear forward",
        "operation": ("NativeLinear(in, out)(input) with graph construction "
                      "over the parameters"),
        "build": _build_linear_forward,
        "seed": 20260211,
        "reference_type": REFERENCE_STABLE,
        "reference_layer": STABLE,
        "reference_detail": ("tensorforge.nn.Linear holding the same weight "
                             "and bias values on the same input"),
        "correctness_reference": ("an explicit NumPy x @ W + b formula and "
                                  "tensorforge.nn.Linear"),
        "configurations": {
            "full": {"batch": 64, "in_features": 128, "out_features": 128},
            "smoke": {"batch": 8, "in_features": 12, "out_features": 6},
            "profile": {"batch": 256, "in_features": 512, "out_features": 512},
        },
        "notes": ("The forward is `matmul` plus a broadcast `add`, so this "
                  "case measures the matmul-dominated part of a real layer "
                  "with the wrapper and graph construction included — what a "
                  "caller actually pays."),
    },
    "linear_forward_backward": {
        "workload": "linear",
        "section": "backward through NativeLinear",
        "operation": ("one backward() over a NativeLinear forward built "
                      "outside the timer"),
        "build": _build_linear_forward_backward,
        "seed": 20260212,
        "reference_type": REFERENCE_STABLE,
        "reference_layer": STABLE,
        "reference_detail": ("tensorforge.nn.Linear's backward under the "
                             "same scalar objective and the same parameters"),
        "correctness_reference": ("the closed-form NumPy gradients and "
                                  "tensorforge.nn.Linear's gradients"),
        "configurations": {
            "full": {"batch": 64, "in_features": 128, "out_features": 128},
            "smoke": {"batch": 8, "in_features": 12, "out_features": 6},
            "profile": {"batch": 256, "in_features": 512, "out_features": 512},
        },
        "repetitions": BACKWARD_REPETITIONS,
        "notes": ("Only backward() is timed. A fresh forward graph is built "
                  "outside the timer for every repetition from cleared "
                  "gradients, so no repetition inherits a retained graph or "
                  "an accumulated gradient. The backward runs two matmuls "
                  "against transposed views plus one unbroadcast reduction "
                  "for the bias."),
    },

    # -- convolution ---------------------------------------------------------
    "conv2d_forward": {
        "allocation_layers": (TENSOR_CORE_ZEROED,),
        "workload": "convolution",
        "section": "conv2d forward",
        "operation": "NCHW cross-correlation forward with bias",
        "build": _build_conv2d,
        "component": "forward",
        "seed": 20260213,
        "reference_type": REFERENCE_STABLE,
        "reference_layer": STABLE,
        "reference_detail": ("tensorforge.nn.Conv2d holding the same weight "
                             "and bias on the same NCHW images"),
        "correctness_reference": ("an explicit NumPy NCHW cross-correlation "
                                  "formula and tensorforge.nn.Conv2d"),
        "configurations": {
            "full": {"batch": 8, "in_channels": 3, "height": 16, "width": 16,
                     "out_channels": 8, "kernel": 3},
            "smoke": {"batch": 2, "in_channels": 2, "height": 6, "width": 5,
                      "out_channels": 3, "kernel": 3},
            "profile": {"batch": 16, "in_channels": 8, "height": 32,
                        "width": 32, "out_channels": 16, "kernel": 3},
        },
        "repetitions": HEAVY_REPETITIONS,
        "notes": ("The direct nested-loop kernel. Measured beside its two "
                  "gradient components and the composed bias gradient so the "
                  "CNN cost is attributed, not assumed."),
    },
    "conv2d_input_backward": {
        "workload": "convolution",
        "section": "conv2d input gradient",
        "operation": "the scatter-add adjoint producing the input gradient",
        "build": _build_conv2d,
        "component": "input_backward",
        "seed": 20260214,
        "reference_type": REFERENCE_STABLE,
        "reference_layer": None,
        "reference_detail": ("no separable stable equivalent: the stable "
                             "Conv2d computes all three gradients inside one "
                             "autograd backward, so it cannot be timed "
                             "against this component alone. No ratio is "
                             "published."),
        "correctness_reference": ("tensorforge.nn.Conv2d's autograd input "
                                  "gradient on identical inputs"),
        "configurations": {
            "full": {"batch": 8, "in_channels": 3, "height": 16, "width": 16,
                     "out_channels": 8, "kernel": 3},
            "smoke": {"batch": 2, "in_channels": 2, "height": 6, "width": 5,
                      "out_channels": 3, "kernel": 3},
            "profile": {"batch": 16, "in_channels": 8, "height": 32,
                        "width": 32, "out_channels": 16, "kernel": 3},
        },
        "repetitions": HEAVY_REPETITIONS,
        "notes": ("The gradient the previous layer receives. Its correctness "
                  "reference is real — the stable line's own autograd — but "
                  "it is a correctness oracle only: the stable backward "
                  "produces all three gradients at once, so timing it here "
                  "would compare one component against three."),
    },
    "conv2d_weight_backward": {
        "workload": "convolution",
        "section": "conv2d weight gradient",
        "operation": "the deterministic accumulation producing the weight gradient",
        "build": _build_conv2d,
        "component": "weight_backward",
        "seed": 20260215,
        "reference_type": REFERENCE_STABLE,
        "reference_layer": None,
        "reference_detail": ("no separable stable equivalent, for the same "
                             "reason as the input gradient. No ratio is "
                             "published."),
        "correctness_reference": ("tensorforge.nn.Conv2d's autograd weight "
                                  "gradient on identical inputs"),
        "configurations": {
            "full": {"batch": 8, "in_channels": 3, "height": 16, "width": 16,
                     "out_channels": 8, "kernel": 3},
            "smoke": {"batch": 2, "in_channels": 2, "height": 6, "width": 5,
                      "out_channels": 3, "kernel": 3},
            "profile": {"batch": 16, "in_channels": 8, "height": 32,
                        "width": 32, "out_channels": 16, "kernel": 3},
        },
        "repetitions": HEAVY_REPETITIONS,
        "notes": ("The optimizer's gradient. Measured separately from the "
                  "input gradient because the two have different loop "
                  "structures and different working sets, so one can "
                  "dominate the other."),
    },
    "conv2d_bias_gradient": {
        "workload": "convolution",
        "section": "conv2d bias gradient (composed)",
        "operation": ("the composed bias gradient: three chained native sum "
                      "reductions, g.sum(0).sum(1).sum(1)"),
        "build": _build_conv2d,
        "component": "bias_gradient",
        "seed": 20260216,
        "reference_type": REFERENCE_NUMPY,
        "reference_layer": None,
        "reference_detail": ("no separable stable equivalent, for the same "
                             "reason as the other two gradients. No ratio is "
                             "published."),
        "correctness_reference": "NumPy's sum over N, H, and W",
        "configurations": {
            "full": {"batch": 8, "in_channels": 3, "height": 16, "width": 16,
                     "out_channels": 8, "kernel": 3},
            "smoke": {"batch": 2, "in_channels": 2, "height": 6, "width": 5,
                      "out_channels": 3, "kernel": 3},
            "profile": {"batch": 16, "in_channels": 8, "height": 32,
                        "width": 32, "out_channels": 16, "kernel": 3},
        },
        "repetitions": HEAVY_REPETITIONS,
        "notes": ("This is not a kernel at all: it is three chained native "
                  "sum reductions, each with its own allocation and its own "
                  "odometer walk. Whether that composition is negligible or "
                  "dominant next to the three real convolution kernels is "
                  "exactly the kind of question H0 exists to answer with "
                  "measurement rather than assumption."),
    },

    # -- training steps ------------------------------------------------------
    "mlp_training_step": {
        "allocation_layers": (TRAINING_STEP_ZEROED,),
        "workload": "training_step",
        "section": "MLP training step",
        "operation": ("zero_grad -> Linear/ReLU/Linear forward -> "
                      "NativeMSELoss -> backward -> NativeAdam.step()"),
        "build": _build_mlp_training_step,
        "seed": 20260217,
        "reference_type": REFERENCE_STABLE,
        "reference_layer": STABLE,
        "reference_detail": ("the same architecture, initial parameter "
                             "values, MSE semantics, and Adam "
                             "hyperparameters on the stable tensorforge line"),
        "correctness_reference": ("the equivalently initialized stable "
                                  "Linear/ReLU/Linear model after the same "
                                  "single Adam step"),
        "configurations": {
            "full": {"batch": 64, "in_features": 32, "hidden": 64,
                     "out_features": 8},
            "smoke": {"batch": 8, "in_features": 4, "hidden": 6,
                      "out_features": 2},
            "profile": {"batch": 256, "in_features": 256, "hidden": 256,
                        "out_features": 64},
        },
        "repetitions": TRAINING_STEP_REPETITIONS,
        "notes": ("A fresh model and optimizer are built outside the timer "
                  "for every repetition, so every timed step starts from the "
                  "same deterministic state and no repetition inherits the "
                  "parameters, moments, or step counters of the one before "
                  "it. No to_numpy(), checkpoint I/O, or reporting work "
                  "happens inside the timed region."),
    },
    "cnn_classification_training_step": {
        "allocation_layers": (TRAINING_STEP_ZEROED,),
        "workload": "training_step",
        "section": "CNN classification training step",
        "operation": ("zero_grad -> Conv2d/ReLU/MaxPool2d/Flatten/Linear "
                      "forward -> raw logits -> NativeCrossEntropyLoss -> "
                      "backward -> NativeAdam.step()"),
        "build": _build_cnn_training_step,
        "seed": 20260218,
        "reference_type": NATIVE_ONLY,
        "reference_layer": None,
        "reference_detail": _NO_STABLE_CNN_STEP,
        "correctness_reference": ("the structural training-step gate: a "
                                  "finite scalar loss of a plausible "
                                  "cross-entropy magnitude, finite gradients "
                                  "on every parameter, one optimizer step, a "
                                  "changed parameter set, and a fully "
                                  "released graph"),
        "configurations": {
            "full": {"batch": 12, "in_channels": 1, "height": 12, "width": 12,
                     "out_channels": 4, "kernel": 3, "classes": 3},
            "smoke": {"batch": 6, "in_channels": 1, "height": 6, "width": 6,
                      "out_channels": 2, "kernel": 3, "classes": 3},
            "profile": {"batch": 32, "in_channels": 3, "height": 24,
                        "width": 24, "out_channels": 8, "kernel": 3,
                        "classes": 5},
        },
        "repetitions": TRAINING_STEP_REPETITIONS,
        "notes": ("Raw logits go straight to NativeCrossEntropyLoss; there "
                  "is deliberately no softmax layer, because the fused "
                  "kernel is what keeps the loss stable. A fresh model and "
                  "optimizer are built outside the timer for every "
                  "repetition."),
    },
    "normalized_training_step": {
        "allocation_layers": (TRAINING_STEP_ZEROED,),
        "workload": "normalization",
        "section": "normalization-heavy training step",
        "operation": ("zero_grad -> Linear/BatchNorm1d/ReLU/LayerNorm/Linear "
                      "forward -> NativeMSELoss -> backward -> "
                      "NativeAdam.step()"),
        "build": _build_normalized_training_step,
        "seed": 20260219,
        "reference_type": REFERENCE_STABLE,
        "reference_layer": STABLE,
        "reference_detail": ("the same architecture, initial parameter and "
                             "running-state values, epsilon, momentum, MSE "
                             "semantics, and Adam hyperparameters on the "
                             "stable tensorforge line"),
        "correctness_reference": ("the equivalently initialized stable "
                                  "Linear/BatchNorm1d/ReLU/LayerNorm/Linear "
                                  "model after the same single Adam step, "
                                  "including its running statistics"),
        "configurations": {
            "full": {"batch": 64, "in_features": 32, "hidden": 64,
                     "out_features": 8},
            "smoke": {"batch": 8, "in_features": 4, "hidden": 6,
                      "out_features": 2},
            "profile": {"batch": 256, "in_features": 256, "hidden": 256,
                        "out_features": 64},
        },
        "repetitions": TRAINING_STEP_REPETITIONS,
        "notes": ("Both normalization families run in every forward and "
                  "BatchNorm is the only stateful module. Because the "
                  "training forward advances persistent running statistics, "
                  "a fresh model is built outside the timer for every "
                  "repetition; a state-advanced module is never reused as a "
                  "measured sample. Neither normalization module has a "
                  "kernel: both are compositions of existing native "
                  "operations, which is why their cost profile is worth "
                  "measuring separately."),
    },
    "dropout_training_step": {
        "workload": "stochastic",
        "section": "Dropout training step",
        "operation": ("zero_grad -> Linear/ReLU/Dropout/Linear forward with "
                      "an explicit NativeGenerator -> NativeMSELoss -> "
                      "backward -> NativeAdam.step()"),
        "build": _build_dropout_training_step,
        "seed": 20260220,
        "reference_type": NATIVE_ONLY,
        "reference_layer": None,
        "reference_detail": _NO_STABLE_DROPOUT,
        "correctness_reference": ("NativeTensorCore.dropout_forward at the "
                                  "same (seed, call_index), the inverted "
                                  "scaling and both mask outcomes checked "
                                  "structurally, evaluation-mode state "
                                  "neutrality, and exactly one generator "
                                  "call consumed per successful step"),
        "configurations": {
            "full": {"batch": 64, "in_features": 32, "hidden": 64,
                     "out_features": 8},
            "smoke": {"batch": 8, "in_features": 4, "hidden": 6,
                      "out_features": 2},
            "profile": {"batch": 256, "in_features": 256, "hidden": 256,
                        "out_features": 64},
        },
        "repetitions": TRAINING_STEP_REPETITIONS,
        "notes": ("The generator is reset to call index 0 in the untimed "
                  "prepare for every repetition, so every timed step draws "
                  "the same mask from the same reserved index and benchmark "
                  "setup can never shift the index a timed call consumes."),
    },

    # -- optimizers ----------------------------------------------------------
    "adam_step": {
        "allocation_layers": (OPTIMIZER_STEP_ZEROED,),
        "workload": "optimizer",
        "section": "NativeAdam.step()",
        "operation": ("one NativeAdam.step() with gradients already present, "
                      "no model work"),
        "build": _build_optimizer_step,
        "optimizer": "adam",
        "seed": 20260221,
        "reference_type": REFERENCE_STABLE,
        "reference_layer": STABLE,
        "reference_detail": ("tensorforge.optim.Adam stepping the same "
                             "parameter values, the same gradients, and the "
                             "same learning rate"),
        "correctness_reference": ("tensorforge.optim.Adam's update on the "
                                  "same parameters and gradients, plus the "
                                  "native versioning and gradient-retention "
                                  "contracts"),
        "configurations": {
            "full": {"batch": 64, "in_features": 32, "hidden": 64,
                     "out_features": 8},
            "smoke": {"batch": 8, "in_features": 4, "hidden": 6,
                      "out_features": 2},
            "profile": {"batch": 64, "in_features": 256, "hidden": 256,
                        "out_features": 64},
        },
        "notes": ("The forward and backward that produce the gradients run "
                  "once, outside the timer; the optimizer retains gradients "
                  "until zero_grad(), so every repetition steps against the "
                  "same gradient values. Adam's update is composed entirely "
                  "from existing Core operations (reciprocal and sqrt, no "
                  "division), which means a fixed number of small native "
                  "calls and allocations per parameter regardless of "
                  "parameter size."),
    },
    "sgd_step": {
        "workload": "optimizer",
        "section": "NativeSGD.step()",
        "operation": ("one NativeSGD.step() with gradients already present, "
                      "no model work"),
        "build": _build_optimizer_step,
        "optimizer": "sgd",
        "seed": 20260222,
        "reference_type": REFERENCE_STABLE,
        "reference_layer": STABLE,
        "reference_detail": ("tensorforge.optim.SGD stepping the same "
                             "parameter values, the same gradients, and the "
                             "same learning rate"),
        "correctness_reference": ("tensorforge.optim.SGD's update on the "
                                  "same parameters and gradients, plus the "
                                  "native versioning and gradient-retention "
                                  "contracts"),
        "configurations": {
            "full": {"batch": 64, "in_features": 32, "hidden": 64,
                     "out_features": 8},
            "smoke": {"batch": 8, "in_features": 4, "hidden": 6,
                      "out_features": 2},
            "profile": {"batch": 64, "in_features": 256, "hidden": 256,
                        "out_features": 64},
        },
        "notes": ("The minimal update, measured beside Adam so the "
                  "difference between them attributes the adaptive "
                  "optimizer's extra cost to its extra operations rather "
                  "than to optimizer machinery in general."),
    },

    # -- state operations ----------------------------------------------------
    "state_dict_snapshot": {
        "workload": "state_operations",
        "section": "state_dict() snapshot",
        "operation": ("one state_dict() snapshot of a normalized model "
                      "(independent owning graph-free copies)"),
        "build": _build_state_operations,
        "section_key": "snapshot",
        "seed": 20260223,
        "reference_type": NATIVE_ONLY,
        "reference_layer": None,
        "reference_detail": _NO_STABLE_STATE,
        "correctness_reference": ("the documented state contract: stable key "
                                  "order, NativeTensor values, snapshots "
                                  "independent of every live registered "
                                  "object, and exact value parity"),
        "configurations": {
            "full": {"batch": 64, "in_features": 32, "hidden": 64,
                     "out_features": 8},
            "smoke": {"batch": 8, "in_features": 4, "hidden": 6,
                      "out_features": 2},
            "profile": {"batch": 64, "in_features": 512, "hidden": 512,
                        "out_features": 128},
        },
        "notes": ("Deliberately its own workload family rather than part of "
                  "any training step. File checkpoint I/O is not measured "
                  "anywhere in this harness — it is dominated by the "
                  "filesystem and the NPZ writer rather than by TensorForge, "
                  "and it belongs to no training iteration."),
    },
    "state_dict_load": {
        "workload": "state_operations",
        "section": "load_state_dict() commit",
        "operation": ("one atomic load_state_dict() commit into the existing "
                      "parameter and buffer objects"),
        "build": _build_state_operations,
        "section_key": "load",
        "seed": 20260224,
        "reference_type": NATIVE_ONLY,
        "reference_layer": None,
        "reference_detail": _NO_STABLE_STATE,
        "correctness_reference": ("the documented load contract: identity "
                                  "preserved, exactly one version increment "
                                  "per matched parameter, and exact value "
                                  "parity after the commit"),
        "configurations": {
            "full": {"batch": 64, "in_features": 32, "hidden": 64,
                     "out_features": 8},
            "smoke": {"batch": 8, "in_features": 4, "hidden": 6,
                      "out_features": 2},
            "profile": {"batch": 64, "in_features": 512, "hidden": 512,
                        "out_features": 128},
        },
        "notes": ("The validate/stage/commit transaction, measured in "
                  "memory. The load moves a parameter version by design, "
                  "which is why this case owns its own model and shares "
                  "nothing with the training-step cases."),
    },
}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def measure(prepare, run, cleanup, warmup, repetitions):
    """Return ``repetitions`` per-call seconds samples for ``run``.

    Each repetition builds its own state with ``prepare()`` (untimed),
    times exactly one ``run(state)`` call with ``time.perf_counter_ns()``,
    then releases everything with ``cleanup(state, result)`` (untimed).
    Warm-up repetitions run the same way and are discarded before
    measuring; no measured sample is ever dropped and no timer overhead is
    subtracted. CPU execution is synchronous, so no explicit
    synchronization is needed."""
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
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _measure_case(name, warmup, repetitions, variant):
    """Build the case, run its correctness gate, and only then time it.

    The ordering here is the whole point: ``check()`` raises before
    ``measure`` is ever reached, so a failed gate publishes no timing."""
    spec = CASES[name]
    config = spec["configurations"][variant]
    case_repetitions = min(repetitions, spec.get("repetitions", repetitions))
    case = spec["build"](config, spec)
    try:
        # -- correctness first; a failure raises and publishes no timing --
        metrics = case["check"]()
        timings = {}
        layers = dict(case["layers"])
        layers.update(case.get("extra_layers", {}))
        # Phase H (H1): a case that names an allocation-sensitive layer
        # gets that layer measured twice — once as it ships, once with
        # the zero-fill forced back on. Same code, same arithmetic; the
        # difference between the pair is the fill.
        for zeroed_name, twin in ZEROED_TWIN.items():
            if (zeroed_name in spec.get("allocation_layers", ())
                    and zeroed_name not in layers):
                layers[zeroed_name] = _zeroed_layer(layers[twin])
        for layer_name, layer in layers.items():
            timings[layer_name] = _statistics(measure(
                layer["prepare"], layer["run"], layer["cleanup"],
                warmup, case_repetitions,
            ))
    finally:
        case["close"]()

    reference_layer = spec["reference_layer"]
    reference = timings.get(reference_layer) if reference_layer else None
    rows = []
    for layer_name, stats in timings.items():
        ratio = None
        if (reference is not None and layer_name != reference_layer
                and reference["median_s"] > 0):
            ratio = stats["median_s"] / reference["median_s"]
        rows.append({
            "implementation_layer": layer_name,
            "timing": stats,
            "ratio_to_reference": ratio,
        })

    # The primary H1 statistic: each shipped layer against its zeroed
    # twin. >1 means the zero-fill cost time in this run; <1 means the
    # difference fell inside this machine's noise, which is an honest and
    # common outcome for small outputs and is reported as such.
    allocation = {}
    for zeroed_name, twin in ZEROED_TWIN.items():
        if zeroed_name in timings and twin in timings:
            fast = timings[twin]["median_s"]
            zeroed = timings[zeroed_name]["median_s"]
            allocation[twin] = {
                "uninitialized_median_s": fast,
                "zero_initialized_median_s": zeroed,
                "zero_fill_median_s": zeroed - fast,
                "speedup_from_skipping_the_fill": (zeroed / fast)
                if fast > 0 else None,
                "uninitialized_relative_spread":
                    timings[twin]["relative_spread"],
                "zero_initialized_relative_spread":
                    timings[zeroed_name]["relative_spread"],
            }

    # The primary H2 statistic: the shipped optimized path against the
    # retained generic reference path, on the same logical operands and
    # the same accumulation order. >1 means the row sweep was ahead in
    # this run. Present only for the cases that build the probe.
    dispatch = {}
    for generic_name, twin in GENERIC_TWIN.items():
        if generic_name in timings and twin in timings:
            fast = timings[twin]["median_s"]
            generic = timings[generic_name]["median_s"]
            dispatch[twin] = {
                "row_sweep_median_s": fast,
                "generic_strided_median_s": generic,
                "speedup_from_the_row_sweep": (generic / fast)
                if fast > 0 else None,
                "row_sweep_relative_spread": timings[twin]["relative_spread"],
                "generic_strided_relative_spread":
                    timings[generic_name]["relative_spread"],
            }

    return {
        "case": name,
        "workload": spec["workload"],
        "section": spec["section"],
        "operation": spec["operation"],
        "configuration_variant": variant,
        "configuration": {key: _jsonable(value)
                          for key, value in config.items()},
        "shape": _case_shape(config),
        "seed": spec["seed"],
        "reference_type": spec["reference_type"],
        "reference_layer": reference_layer,
        "reference_detail": spec["reference_detail"],
        "correctness_reference": spec["correctness_reference"],
        "correctness": dict(status="passed", **_jsonable(metrics)),
        "warmup": warmup,
        "sample_count": case_repetitions,
        "layers": rows,
        "allocation_comparison": allocation or None,
        "dispatch_comparison": dispatch or None,
        "notes": spec["notes"],
    }


def _case_shape(config):
    """The case's headline shape, as a plain list of ints.

    Every configuration declares either an explicit ``shape``, a matmul
    ``(m, n, p)``, a convolution NCHW block, or a linear/training
    ``(batch, in_features)`` pair — reported uniformly so a reader never
    has to guess which key carried the size."""
    if "shape" in config:
        return [int(dim) for dim in config["shape"]]
    if "m" in config:
        return [int(config["m"]), int(config["n"]), int(config["p"])]
    if "height" in config:
        return [int(config["batch"]), int(config["in_channels"]),
                int(config["height"]), int(config["width"])]
    return [int(config["batch"]), int(config["in_features"])]


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------


def _thread_environment():
    """The BLAS / threading environment variables that are actually set.

    Recorded so a reader can tell whether the NumPy reference column ran
    single- or multi-threaded. Absent variables are simply not listed —
    this never invents a value."""
    return {name: os.environ[name] for name in THREAD_ENVIRONMENT_VARIABLES
            if name in os.environ}


def _numpy_build_information():
    """Whatever NumPy's own introspection API reports about its build.

    Read through ``numpy.show_config("dicts")`` where available, which is
    the real introspection surface rather than a guess. Nothing is
    fabricated if it is unavailable."""
    show_config = getattr(np, "show_config", None)
    if show_config is None:
        return None
    try:
        information = show_config("dicts")
    except (TypeError, ValueError, AttributeError):
        return None
    if not isinstance(information, dict):
        return None
    build = information.get("Build Dependencies", {})
    blas = build.get("blas", {}) if isinstance(build, dict) else {}
    lapack = build.get("lapack", {}) if isinstance(build, dict) else {}
    simd = information.get("SIMD Extensions", {})
    return {
        "blas_name": blas.get("name") if isinstance(blas, dict) else None,
        "blas_version": blas.get("version") if isinstance(blas, dict) else None,
        "lapack_name": lapack.get("name") if isinstance(lapack, dict) else None,
        "simd_extensions": _jsonable(simd) if isinstance(simd, dict) else None,
    }


def _backend_metadata():
    """The native backend's own introspection output — the real API, not
    a hand-maintained restatement of it."""
    info = cpp.backend_info()
    return {
        "name": info["name"],
        "available": info["available"],
        "experimental": info["experimental"],
        "tensor_core": info["tensor_core"],
        "tensor_object": info["tensor_object"],
        "dtype": info["dtype"],
        "device": info["device"],
        "supported_dtypes": list(info["supported_dtypes"]),
        "supported_devices": list(info["supported_devices"]),
        "unsupported": list(info["unsupported"]),
        "native_autograd": info["native_autograd"],
        "stable_framework_integration": info["stable_framework_integration"],
        "raw_kernel_count": len(info["raw_kernels"]),
        "tensor_core_op_count": len(info["tensor_core_ops"]),
        "autograd_op_count": len(info["autograd_ops"]),
        "native_module_count": len(info["native_modules"]),
        "state_support": list(info["state_support"]),
    }


def _native_build_metadata():
    """What can honestly be said about the build the measurements ran
    against, read from the compiled image itself.

    Deliberately short. The compiler identity, its version, and its
    optimization flags are **not** recorded anywhere by ``cpp/build.py``
    or the CMake project, so they are reported as ``null`` rather than
    guessed from the host — ``platform.python_compiler()`` describes the
    interpreter's toolchain, not this library's, and printing it here
    would be a fabrication. The image's format and size are real, and
    together with the object-format field they do distinguish a Release
    build from a Debug one in practice. No path is emitted."""
    build = {
        "image_format": None,
        "image_bytes": None,
        "compiler": None,
        "compiler_detail": ("not recorded by the build; reported as null "
                            "rather than inferred from the host"),
        "sanitizers": None,
    }
    path = getattr(cpp, "_LIBRARY_PATH", None)
    try:
        if path is None or not path.exists():
            return build
        data = path.read_bytes()
        build["image_bytes"] = len(data)
        if data[:2] == b"MZ":
            build["image_format"] = "pe"
        elif data[:4] == b"\x7fELF":
            build["image_format"] = "elf"
        else:
            build["image_format"] = "unknown"
        # Sanitizer instrumentation is genuinely visible in the image, so
        # unlike the compiler it is reported rather than left null.
        build["sanitizers"] = sorted(
            name for name, marker in (("address", b"__asan_"),
                                      ("undefined", b"__ubsan_"))
            if marker in data
        )
    except OSError:
        return build
    return build


def _environment(warmup, repetitions, variant):
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "numpy_version": np.__version__,
        "numpy_build": _numpy_build_information(),
        "tensorforge_version": tensorforge.__version__,
        "native_backend": _backend_metadata(),
        "native_build": _native_build_metadata(),
        "thread_environment": _thread_environment(),
        "dtype": "float64",
        "device": "cpu",
        "scope": "native CPU runtime (float64/cpu)",
        "timer": "time.perf_counter_ns",
        "timer_resolution_ns": time.get_clock_info("perf_counter").resolution
        * 1e9,
        "primary_statistic": "median",
        "configuration_variant": variant,
        "warmup": warmup,
        "repetitions": repetitions,
        "training_step_repetitions": min(repetitions,
                                         TRAINING_STEP_REPETITIONS),
        "backward_repetitions": min(repetitions, BACKWARD_REPETITIONS),
        "heavy_repetitions": min(repetitions, HEAVY_REPETITIONS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    if not selected:
        raise ValueError(f"no {label} selected")
    for item in selected:
        if item not in allowed:
            raise ValueError(
                f"unknown {label} {item!r}; choose from {tuple(allowed)}"
            )
    return selected


def cases_for_workloads(workloads):
    """The case names belonging to the given workload families, in
    registry order."""
    selected = set(workloads)
    return tuple(name for name, spec in CASES.items()
                 if spec["workload"] in selected)


def run_benchmark(cases=None, workloads=None, warmup=DEFAULTS["warmup"],
                  repetitions=DEFAULTS["repetitions"], smoke=False,
                  profile=False):
    """Run the selected cases and return the JSON-ready payload.

    Every case's correctness gate runs **before** its timing; a failed
    gate raises (the CLI turns that into a nonzero exit) and no timing is
    published for it. No timing threshold is applied anywhere — this only
    measures. Raises RuntimeError if the native backend is not built.

    ``smoke`` selects the tiny configurations, ``profile`` the deliberately
    larger ones; they are mutually exclusive."""
    if not cpp.is_available():
        raise RuntimeError(
            "The experimental C++ backend is not built.\n"
            + cpp.build_instructions()
        )
    if smoke and profile:
        raise ValueError("--smoke and --profile are mutually exclusive")
    warmup = _positive_int(warmup, "warmup")
    repetitions = _positive_int(repetitions, "repetitions")
    if workloads is not None:
        allowed = cases_for_workloads(
            _resolve(workloads, WORKLOADS, "workload"))
    else:
        allowed = tuple(CASES)
    selected = _resolve(cases, allowed, "case")
    if profile and len(selected) != 1:
        raise ValueError(
            "the focused profile mode runs exactly one case; got "
            f"{len(selected)}"
        )
    variant = "smoke" if smoke else ("profile" if profile else "full")
    return {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "schema_version": SCHEMA_VERSION,
        "mode": "smoke" if smoke else ("profile" if profile else "full"),
        "environment": _environment(warmup, repetitions, variant),
        "cases": [_measure_case(name, warmup, repetitions, variant)
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
    "never a pass/fail criterion, and no test or CI job\nasserts a duration. "
    "No result file is written."
)


def format_report(payload):
    """A concise human-readable report. Carries no speed verdict."""
    env = payload["environment"]
    backend = env["native_backend"]
    lines = [
        f"TensorForge native CPU performance baseline "
        f"v{payload['version']} [{payload['mode']}]",
        f"  platform  : {env['platform']}",
        f"  machine   : {env['machine']}",
        f"  processor : {env['processor']}",
        f"  python    : {env['python_version']} "
        f"({env['python_implementation']})   numpy {env['numpy_version']}   "
        f"tensorforge {env['tensorforge_version']}",
        f"  backend   : {backend['tensor_core']} "
        f"({backend['dtype']}/{backend['device']}, "
        f"available={backend['available']})",
        f"  timer     : {env['timer']} "
        f"(resolution {env['timer_resolution_ns']:.0f} ns)   "
        f"primary statistic: {env['primary_statistic']}",
        f"  warmup/repetitions : {env['warmup']}/{env['repetitions']} "
        f"(backward: {env['backward_repetitions']}, heavy: "
        f"{env['heavy_repetitions']}, training step: "
        f"{env['training_step_repetitions']})",
    ]
    thread_environment = env["thread_environment"]
    if thread_environment:
        joined = ", ".join(f"{key}={value}"
                           for key, value in sorted(thread_environment.items()))
        lines.append(f"  threads   : {joined}")
    else:
        lines.append("  threads   : no BLAS/thread environment variable set")
    numpy_build = env["numpy_build"]
    if numpy_build and numpy_build.get("blas_name"):
        lines.append(f"  numpy blas: {numpy_build['blas_name']} "
                     f"{numpy_build.get('blas_version') or ''}".rstrip())
    lines.append("")

    header = (
        f"{'case':<34} {'layer':<21} {'shape':<16} {'median':>12} "
        f"{'spread':>11} {'ratio':>8}  {'reference':<20} {'correct':<8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    current_workload = None
    for record in payload["cases"]:
        if record["workload"] != current_workload:
            current_workload = record["workload"]
            lines.append(f"[{current_workload}]")
        shape = "x".join(str(dim) for dim in record["shape"])
        for row in record["layers"]:
            ratio = row["ratio_to_reference"]
            ratio_text = f"{ratio:.2f}x" if ratio is not None else "n/a"
            timing = row["timing"]
            lines.append(
                f"{record['case']:<34} "
                f"{row['implementation_layer']:<21} "
                f"{shape:<16} "
                f"{_format_duration(timing['median_s']):>12} "
                f"{_format_duration(timing['spread_s']):>11} "
                f"{ratio_text:>8}  "
                f"{record['reference_type']:<20} "
                f"{record['correctness']['status']:<8}"
            )
    # Phase H (H1): the allocation-contract comparison, reported on its
    # own because it is a native-vs-native measurement and the ratios
    # above are native-vs-reference.
    allocation_rows = [
        (record, layer, data)
        for record in payload["cases"]
        for layer, data in (record.get("allocation_comparison") or {}).items()
    ]
    if allocation_rows:
        lines.append("")
        lines.append("H1 allocation contract "
                     "(uninitialized output vs the zero-initializing default)")
        header = (
            f"{'case':<34} {'layer':<21} {'uninit':>12} {'zeroed':>12} "
            f"{'fill':>12} {'x':>7}  {'spread(u/z)':>13}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for record, layer, data in allocation_rows:
            speedup = data["speedup_from_skipping_the_fill"]
            fill = data["zero_fill_median_s"]
            spread_u = data["uninitialized_relative_spread"]
            spread_z = data["zero_initialized_relative_spread"]
            lines.append(
                f"{record['case']:<34} {layer:<21} "
                f"{_format_duration(data['uninitialized_median_s']):>12} "
                f"{_format_duration(data['zero_initialized_median_s']):>12} "
                f"{('-' if fill < 0 else '') + _format_duration(abs(fill)):>12} "
                f"{(f'{speedup:.2f}' if speedup else 'n/a'):>7}  "
                f"{f'{spread_u:.0%}/{spread_z:.0%}':>13}"
            )
        lines.append("")
        lines.append(
            "'fill' is the zeroed median minus the uninitialized median: "
            "the cost of the\nredundant write pass H1 removed. A negative "
            "fill or an x below 1.00 means the\ndifference fell inside this "
            "run's noise -- compare it against the spread column\nbefore "
            "reading anything into it. numpy.zeros is shown in the table "
            "above for\ncontext only and is NOT the comparison: calloc can "
            "be answered with lazy zero\npages, so it measures the OS "
            "rather than an allocator TensorForge could adopt."
        )
    # Phase H (H2): the dispatch comparison, reported on its own for the
    # same reason — it is a native-vs-native measurement of the *same*
    # production call under two operand layouts.
    dispatch_rows = [
        (record, layer, data)
        for record in payload["cases"]
        for layer, data in (record.get("dispatch_comparison") or {}).items()
    ]
    if dispatch_rows:
        lines.append("")
        lines.append("H2 matmul dispatch "
                     "(the row sweep vs the retained generic reference path)")
        header = (
            f"{'case':<34} {'layer':<21} {'row sweep':>12} {'generic':>12} "
            f"{'x':>7}  {'spread(r/g)':>13}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for record, layer, data in dispatch_rows:
            speedup = data["speedup_from_the_row_sweep"]
            spread_r = data["row_sweep_relative_spread"]
            spread_g = data["generic_strided_relative_spread"]
            lines.append(
                f"{record['case']:<34} {layer:<21} "
                f"{_format_duration(data['row_sweep_median_s']):>12} "
                f"{_format_duration(data['generic_strided_median_s']):>12} "
                f"{(f'{speedup:.2f}' if speedup else 'n/a'):>7}  "
                f"{f'{spread_r:.0%}/{spread_g:.0%}':>13}"
            )
        lines.append("")
        lines.append(
            "Both columns are the same production Core call on the same "
            "logical operands and\nthe same accumulation order; only the "
            "right operand's layout differs, which is\nwhat selects the "
            "kernel. Every operand here is finite, so\nthe gate proved the "
            "two results bit-identical before either was timed.\n"
            "An x below 1.00 means the generic path was "
            "ahead in this run --\nread it against the spread column, and "
            "note that the generic path is genuinely\nthe better one for a "
            "transposed right operand."
        )
    lines.append("")
    lines.append(
        "ratio = this layer's median / the case's reference-layer median "
        "(>1 means this\nlayer took longer in this local run; <1 means it "
        "took less time here). A case\nwith no honest equivalent reports "
        "n/a everywhere and publishes no ratio; its\ncorrectness gate is "
        "still real. See each case's reference_detail in --json for why."
    )
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description=("Characterize the native CPU runtime across "
                     "implementation layers (measurement only; no speed is "
                     "asserted).")
    )
    parser.add_argument("--case", choices=tuple(CASES), default=None,
                        help="run a single case (default: all)")
    parser.add_argument("--workload", choices=WORKLOADS, default=None,
                        help="run one workload family (default: all)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="warm-up repetitions before measuring")
    parser.add_argument("--repetitions", type=int, default=None,
                        help="measured repetitions per case")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON only")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny shapes and counts, for tests/CI")
    parser.add_argument("--profile", choices=tuple(CASES), default=None,
                        metavar="CASE",
                        help=("focused profiler mode: run exactly this case "
                              "at its larger profile shape with more "
                              "repetitions"))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.profile and (args.case or args.workload):
        parser.error("--profile selects its own single case; do not combine "
                     "it with --case or --workload")
    if args.profile and args.smoke:
        parser.error("--smoke and --profile are mutually exclusive")
    if args.smoke:
        defaults = SMOKE_DEFAULTS
    elif args.profile:
        defaults = PROFILE_DEFAULTS
    else:
        defaults = DEFAULTS
    warmup = args.warmup if args.warmup is not None else defaults["warmup"]
    repetitions = (args.repetitions if args.repetitions is not None
                   else defaults["repetitions"])
    cases = None
    if args.profile:
        cases = [args.profile]
    elif args.case:
        cases = [args.case]
    try:
        payload = run_benchmark(
            cases=cases,
            workloads=[args.workload] if args.workload else None,
            warmup=warmup, repetitions=repetitions, smoke=args.smoke,
            profile=bool(args.profile),
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
