"""Dtype characterization for the native CPU runtime (Phase I, I10).

Since milestone I9 the native line supports **float32 and float64** on the
CPU, publicly. This harness characterizes both — **separately** — across
the layers where an honest measurement exists.

Why this is its own file rather than a mode of
``benchmark_native_cpu_performance.py``
---------------------------------------------------------------------

That harness is the instrument Phase H's ladder was *chosen from* and
re-measured against, and its case inventory is pinned by test as "the H0
set". Adding a dtype axis to it would change that inventory and make its
historical pre/post record inaccurate — a Phase-H result would no longer
mean what it meant when it was published. So Phase H's harness is left
exactly as it is, every one of its CLI options and tests still passing,
and dtype characterization lives here.

**Nothing here asserts a speed.** No timing threshold, no performance
budget, no committed duration, no CI job that fails on a number, and **no
result file of any kind** — not JSON, not CSV, not a cache. ``--json``
writes machine-readable output to *stdout* and nowhere else. Every figure
is a local characterization of one machine, one build, and one moment.

**The two dtypes are never divided by one another.** A float32/float64
speed ratio is a property of one machine's memory bandwidth, not a
property of TensorForge, and publishing one would turn a measurement into
a promise the project cannot keep. Each dtype gets its own section. The
honest expected shape of the result is that float32 helps where the work
is bandwidth-bound and is neutral where it is not; whatever is measured is
what gets printed, including neutral and negative findings.

**Correctness runs before timing, always.** Every case validates its
native result against a reference **at its own dtype** before the timing
helper is reached, so a failed gate publishes no timing and the CLI exits
nonzero with clean stdout. There are four gates, chosen per family from
the contracts of ``docs/native_cpu_performance_design.md`` §7 and
``docs/native_dtype_float32_design.md`` §10 rather than from one blanket
rule:

- ``bitwise`` — transfer and elementwise, where the contract really is bit
  equality because each destination element is one correctly-rounded IEEE
  operation;
- ``summation_bound`` — reductions and matmul, where TensorForge preserves
  a strict sequential accumulation order and NumPy's BLAS reference does
  not. The bound is the classical one for sequential summation,
  ``2 * n * eps * max sum|terms|``, computed from the actual operands. It
  is **derived, not tuned**: a fixed tolerance fails first on the output
  cell that happens to sum to nearly zero, which says nothing about
  either implementation;
- ``tolerance`` — softmax, where the reference composes differently but
  the operation does not accumulate over a long chain;
- ``finite`` — the composed and stateful cases (CNN, fused loss,
  normalization, Dropout, optimizers, the training step), where no
  independent same-dtype oracle exists that would not just be a second
  implementation of the same thing. Their numerics are proved by the test
  suite; what the gate adds here is that the case really ran, at the
  right width, and produced finite values.

**A float32 result is never compared to a float64 one.** Design §10.4
forbids making a contract out of that comparison, and this harness does
not make one even informally.

The control case
----------------

``control_identical`` runs the *same* float64 code as ``control_twin``.
Any measured difference between them is this machine's noise, and that
spread is the **control band**: a reading inside it is neutral, whatever
its sign. It is a noise estimate, never a gate.

Modes
-----

::

    uv run python benchmarks/benchmark_native_dtype.py
    uv run python benchmarks/benchmark_native_dtype.py --smoke
    uv run python benchmarks/benchmark_native_dtype.py --json
    uv run python benchmarks/benchmark_native_dtype.py --dtype float32
    uv run python benchmarks/benchmark_native_dtype.py --case matmul_contiguous
    uv run python benchmarks/benchmark_native_dtype.py --family reduction
    uv run python benchmarks/benchmark_native_dtype.py --repetitions 25

Publishable runs use **21-25 measured repetitions**. Low round counts lie:
Phase H recorded four separate cases that read as regressions at 7-9
rounds and as neutral-or-faster at 21-25, so a 7-9-round figure is never
quoted as evidence. ``--smoke`` is a fast correctness path for CI and is
explicitly *not* a measurement.

What is timed
-------------

One repetition times exactly one call with ``time.perf_counter_ns()``.
Input construction, module and optimizer construction, graph construction
for the backward-only cases, and every cleanup happen **outside** the
timer. Temporaries are closed explicitly between repetitions — nothing
relies on garbage collection — and any case whose call advances
persistent state (a generator counter, a BatchNorm running buffer, an
optimizer moment) resets it outside the timer, so supposedly identical
repetitions really are identical.

I10 adds **no** capability: no kernel, C ABI export, ctypes declaration,
Core method, autograd operation, module, loss, optimizer, dtype, device,
registry value, or checkpoint change. It only measures what I1-I9
shipped.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tensorforge.backends import cpp                        # noqa: E402
from tensorforge.experimental import (                      # noqa: E402
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
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
)

# The two publicly supported widths, in the registry's contractual order:
# float64 first, because it is the default an omitted ``dtype`` selects.
DTYPES = ("float64", "float32")
NUMPY_DTYPES = {"float64": np.float64, "float32": np.float32}
BIT_DTYPES = {"float64": np.uint64, "float32": np.uint32}

# Same-dtype tolerances for the accumulating families. Deliberately
# per-dtype: binary32 carries 24 significand bits, so a binary64 tolerance
# would be meaninglessly loose there and a binary32 one impossibly tight
# here. Neither is ever used to compare one dtype against the other.
TOLERANCES = {
    "float64": {"rtol": 1e-12, "atol": 1e-12},
    "float32": {"rtol": 1e-5, "atol": 1e-6},
}

BITWISE = "bitwise"
TOLERANCE = "tolerance"
SUMMATION_BOUND = "summation_bound"
FINITE = "finite"

FAMILIES = (
    "transfer",
    "elementwise",
    "reduction",
    "matmul",
    "cnn",
    "classification",
    "normalization",
    "dropout",
    "optimizer",
    "training",
    "control",
)

DEFAULTS = {"warmup": 5, "repetitions": 21}
SMOKE_DEFAULTS = {"warmup": 1, "repetitions": 3}

# Publishable measurement needs 21-25 repetitions (see the module
# docstring). Below that the run is still *valid*, it is just not
# quotable, and the report says so rather than leaving a reader to guess.
PUBLISHABLE_MINIMUM = 21
PUBLISHABLE_MAXIMUM = 25


# ==========================================================================
# Measurement primitives
# ==========================================================================


class Case:
    """One measurable case: a call, a correctness gate, and a teardown.

    ``reset`` runs between repetitions for a call that advances persistent
    state, and is **outside** the timer."""

    def __init__(self, call, verify, teardown, reset=None):
        self.call = call
        self.verify = verify
        self.teardown = teardown
        self.reset = reset


def close_all(*objects):
    for item in objects:
        if item is not None and hasattr(item, "close"):
            item.close()


def bits_of(array, dtype):
    """Raw IEEE-754 bit patterns, with the dtype asserted rather than
    coerced. A helper that silently converted could report a match that
    only existed after a conversion this runtime never performs."""
    array = np.asarray(array)
    if array.dtype != NUMPY_DTYPES[dtype]:
        raise AssertionError(
            f"expected a {dtype} result, got {array.dtype}")
    return np.ascontiguousarray(array).reshape(-1).view(BIT_DTYPES[dtype])


def gate_bitwise(got, expected, dtype, label):
    """Bit-exact agreement with a same-dtype reference."""
    left, right = bits_of(got, dtype), bits_of(expected, dtype)
    if left.shape != right.shape or not np.array_equal(left, right):
        raise AssertionError(f"{label}: result is not bit-identical to its "
                             f"{dtype} reference")
    return {"comparison": BITWISE, "elements": int(left.size)}


def gate_tolerance(got, expected, dtype, label):
    """Same-dtype agreement within a stated, dtype-appropriate tolerance.

    Used where the operation accumulates, so the reference's summation
    order is not TensorForge's and bit equality would be asserting
    something neither implementation promises."""
    got = np.asarray(got)
    expected = np.asarray(expected)
    if got.dtype != NUMPY_DTYPES[dtype]:
        raise AssertionError(f"{label}: expected {dtype}, got {got.dtype}")
    if got.shape != expected.shape:
        raise AssertionError(f"{label}: shape {got.shape} vs {expected.shape}")
    tolerance = TOLERANCES[dtype]
    if not np.allclose(got, expected, **tolerance):
        worst = float(np.max(np.abs(got.astype(np.float64)
                                    - expected.astype(np.float64))))
        raise AssertionError(
            f"{label}: result differs from its {dtype} reference by "
            f"{worst:g}, outside rtol={tolerance['rtol']:g} "
            f"atol={tolerance['atol']:g}")
    return {"comparison": TOLERANCE, "elements": int(got.size),
            "rtol": tolerance["rtol"], "atol": tolerance["atol"]}


def gate_accumulated(got, expected, dtype, label, terms, term_magnitudes):
    """Same-dtype agreement within the **classical summation error
    bound**, derived rather than tuned.

    A fixed tolerance is the wrong instrument for an accumulating
    operation. TensorForge preserves a strict sequential accumulation
    order by contract; NumPy's reference goes through BLAS, which blocks,
    vectorizes, and may use FMA. Both are correct, and over ``n`` binary32
    additions they legitimately differ — by an amount that grows with
    ``n``, not with the magnitude of the *result*, which is why a cell
    that happens to sum to nearly zero is the one a fixed ``atol`` fails
    on first.

    So the bound is the textbook one for sequential summation, applied to
    both sides:

        |error| <= 2 * n * eps * max over outputs of sum |terms|

    ``term_magnitudes`` is that per-output sum of absolute summands,
    computed exactly rather than estimated. The factor of two is because
    *each* implementation may err by up to the single-sided bound; nothing
    here was widened until a test went green."""
    got = np.asarray(got)
    expected = np.asarray(expected)
    if got.dtype != NUMPY_DTYPES[dtype]:
        raise AssertionError(f"{label}: expected {dtype}, got {got.dtype}")
    if got.shape != expected.shape:
        raise AssertionError(f"{label}: shape {got.shape} vs {expected.shape}")
    eps = float(np.finfo(NUMPY_DTYPES[dtype]).eps)
    bound = 2.0 * terms * eps * float(np.max(term_magnitudes))
    worst = float(np.max(np.abs(got.astype(np.float64)
                                - expected.astype(np.float64))))
    if not worst <= bound:
        raise AssertionError(
            f"{label}: result differs from its {dtype} reference by "
            f"{worst:g}, outside the {terms}-term summation bound {bound:g}")
    return {"comparison": SUMMATION_BOUND, "elements": int(got.size),
            "bound": bound, "observed": worst, "terms": int(terms),
            "bound_rule": "2 * n * eps * max sum|terms|"}


def gate_finite(array, dtype, label):
    array = np.asarray(array)
    if array.dtype != NUMPY_DTYPES[dtype]:
        raise AssertionError(f"{label}: expected {dtype}, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise AssertionError(f"{label}: result is not finite")
    return {"comparison": "finite", "elements": int(array.size)}


def measure(case, warmup, repetitions):
    """Warm up, then time ``repetitions`` single calls.

    No sample is discarded and no timer overhead is subtracted. The
    per-repetition reset and every temporary's release happen outside the
    measured region."""
    for _ in range(warmup):
        result = case.call()
        close_all(result)
        if case.reset is not None:
            case.reset()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        result = case.call()
        elapsed = time.perf_counter_ns() - start
        close_all(result)
        if case.reset is not None:
            case.reset()
        samples.append(elapsed / 1e9)
    return samples


def summarize(samples):
    """Median with an explicit, named spread statistic.

    The spread reported is the **interquartile range** — robust against
    the single scheduling outlier a short run always contains, and stated
    rather than left for a reader to infer. The min and max are carried
    too, because a reader deserves to see the tail this hides."""
    ordered = sorted(samples)
    if len(ordered) >= 4:
        lower = statistics.median(ordered[: len(ordered) // 2])
        upper = statistics.median(ordered[(len(ordered) + 1) // 2:])
        iqr = upper - lower
    else:
        iqr = max(ordered) - min(ordered)
    median = statistics.median(ordered)
    return {
        "samples": len(ordered),
        "median_s": median,
        "iqr_s": iqr,
        "min_s": ordered[0],
        "max_s": ordered[-1],
        "spread_statistic": "interquartile range",
        "relative_iqr": (iqr / median) if median > 0 else 0.0,
    }


# ==========================================================================
# Deterministic inputs
#
# Every generator is local and seeded. The global NumPy RNG and Python's
# ``random`` are never touched, so two runs of this file build byte-
# identical inputs.
# ==========================================================================


def host_values(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=shape).astype(NUMPY_DTYPES[dtype])


def native(values, dtype, requires_grad=False):
    return NativeTensor.from_array(values, dtype=dtype,
                                   requires_grad=requires_grad)


def native_core(values, dtype):
    return cpp.NativeTensorCore.from_array(np.asarray(values), dtype=dtype)


# ==========================================================================
# Case builders — one per measured case
# ==========================================================================


def build_host_ingress(dtype, config, seed):
    values = host_values(config["shape"], dtype, seed)

    def call():
        return NativeTensor.from_array(values, dtype=dtype)

    def verify():
        tensor = call()
        try:
            return gate_bitwise(tensor.to_numpy(), values, dtype,
                                "host_ingress")
        finally:
            tensor.close()

    return Case(call, verify, lambda: None)


def build_host_egress(dtype, config, seed):
    values = host_values(config["shape"], dtype, seed)
    tensor = native(values, dtype)

    def call():
        tensor.to_numpy()
        return None

    def verify():
        return gate_bitwise(tensor.to_numpy(), values, dtype, "host_egress")

    return Case(call, verify, lambda: close_all(tensor))


def build_contiguous_copy(dtype, config, seed):
    values = host_values(config["shape"], dtype, seed)
    source = native_core(values, dtype)

    def call():
        return source.contiguous_copy()

    def verify():
        out = call()
        try:
            return gate_bitwise(out.to_numpy(), values, dtype,
                                "contiguous_copy")
        finally:
            out.close()

    return Case(call, verify, lambda: close_all(source))


def build_strided_materialize(dtype, config, seed):
    """A transposed view materialized — the non-contiguous gather."""
    values = host_values(config["shape"], dtype, seed)
    source = native_core(values, dtype)
    view = source.T

    def call():
        return view.contiguous_copy()

    def verify():
        out = call()
        try:
            return gate_bitwise(out.to_numpy(),
                                np.ascontiguousarray(values.T), dtype,
                                "strided_materialize")
        finally:
            out.close()

    return Case(call, verify, lambda: close_all(view, source))


def _elementwise(dtype, left_shape, right_shape, seed, label):
    left_host = host_values(left_shape, dtype, seed)
    right_host = host_values(right_shape, dtype, seed + 1)
    left = native_core(left_host, dtype)
    right = native_core(right_host, dtype)

    def call():
        return left.multiply(right)

    def verify():
        out = call()
        try:
            # One correctly-rounded IEEE multiply per destination element,
            # so this family's contract really is bit equality.
            return gate_bitwise(out.to_numpy(), left_host * right_host,
                                dtype, label)
        finally:
            out.close()

    return Case(call, verify, lambda: close_all(left, right))


def build_elementwise_contiguous(dtype, config, seed):
    return _elementwise(dtype, config["shape"], config["shape"], seed,
                        "elementwise_contiguous")


def build_elementwise_broadcast(dtype, config, seed):
    rows, cols = config["shape"]
    return _elementwise(dtype, (rows, cols), (1, cols), seed,
                        "elementwise_broadcast")


def build_elementwise_small(dtype, config, seed):
    """Deliberately tiny: below roughly 1,000 elements a fixed ~7-12 us
    Python-plus-ctypes cost dominates and the kernel work is invisible.
    That is an architectural floor, not a defect, and this case exists to
    show it at both widths rather than to be optimized away."""
    return _elementwise(dtype, config["shape"], config["shape"], seed,
                        "elementwise_small")


def _reduction(dtype, shape, strided, seed, label):
    values = host_values(shape, dtype, seed)
    source = native_core(values, dtype)
    operand = source.T if strided else source
    oriented = np.ascontiguousarray(values.T if strided else values)
    expected = oriented.sum(axis=0, dtype=NUMPY_DTYPES[dtype])
    terms = oriented.shape[0]
    magnitudes = np.abs(oriented.astype(np.float64)).sum(axis=0)

    def call():
        return operand.sum(axis=0)

    def verify():
        out = call()
        try:
            # Reductions accumulate, and NumPy's summation order is its
            # own, so the gate is the derived summation bound rather than
            # a fixed tolerance.
            return gate_accumulated(out.to_numpy(), expected, dtype, label,
                                    terms, magnitudes)
        finally:
            out.close()

    def teardown():
        if strided:
            operand.close()
        source.close()

    return Case(call, verify, teardown)


def build_reduction_contiguous(dtype, config, seed):
    return _reduction(dtype, config["shape"], False, seed,
                      "reduction_contiguous")


def build_reduction_strided(dtype, config, seed):
    return _reduction(dtype, config["shape"], True, seed,
                      "reduction_strided")


def _matmul(dtype, shape, transposed, seed, label):
    size = shape[0]
    left_host = host_values((size, size), dtype, seed)
    right_host = host_values((size, size), dtype, seed + 1)
    left = native_core(left_host, dtype)
    right_owner = native_core(
        np.ascontiguousarray(right_host.T) if transposed else right_host,
        dtype)
    right = right_owner.T if transposed else right_owner
    expected = (left_host @ right_host).astype(NUMPY_DTYPES[dtype])
    magnitudes = (np.abs(left_host.astype(np.float64))
                  @ np.abs(right_host.astype(np.float64)))

    def call():
        return left.matmul(right)

    def verify():
        out = call()
        try:
            return gate_accumulated(out.to_numpy(), expected, dtype, label,
                                    size, magnitudes)
        finally:
            out.close()

    def teardown():
        if transposed:
            right.close()
        close_all(right_owner, left)

    return Case(call, verify, teardown)


def build_matmul_contiguous(dtype, config, seed):
    """The layout H2's row-sweep predicate selects."""
    return _matmul(dtype, config["shape"], False, seed, "matmul_contiguous")


def build_matmul_transposed_view(dtype, config, seed):
    """A strided right operand, which takes the retained generic
    traversal. Both widths take the same path for the same layout, because
    the predicate is a function of layout metadata alone."""
    return _matmul(dtype, config["shape"], True, seed,
                   "matmul_transposed_view")


def build_conv2d_forward(dtype, config, seed):
    n, c, h, w = config["shape"]
    out_channels = config["out_channels"]
    x_host = host_values((n, c, h, w), dtype, seed)
    weight_host = host_values((out_channels, c, 3, 3), dtype, seed + 1)
    bias_host = host_values((out_channels,), dtype, seed + 2)
    x = native_core(x_host, dtype)
    weight = native_core(weight_host, dtype)
    bias = native_core(bias_host, dtype)

    def call():
        return x.conv2d_forward(weight, bias, padding=1)

    def verify():
        out = call()
        try:
            return gate_finite(out.to_numpy(), dtype, "conv2d_forward")
        finally:
            out.close()

    return Case(call, verify, lambda: close_all(x, weight, bias))


def build_conv2d_input_backward(dtype, config, seed):
    n, c, h, w = config["shape"]
    out_channels = config["out_channels"]
    upstream = native_core(host_values((n, out_channels, h, w), dtype, seed),
                           dtype)
    weight = native_core(
        host_values((out_channels, c, 3, 3), dtype, seed + 1), dtype)

    def call():
        return upstream.conv2d_input_backward(
            weight, input_shape=(n, c, h, w), padding=1)

    def verify():
        out = call()
        try:
            return gate_finite(out.to_numpy(), dtype,
                               "conv2d_input_backward")
        finally:
            out.close()

    return Case(call, verify, lambda: close_all(upstream, weight))


def build_maxpool2d_forward(dtype, config, seed):
    x = native_core(host_values(config["shape"], dtype, seed), dtype)

    def call():
        return x.maxpool2d_forward(kernel_size=2)

    def verify():
        out = call()
        try:
            return gate_finite(out.to_numpy(), dtype, "maxpool2d_forward")
        finally:
            out.close()

    return Case(call, verify, lambda: close_all(x))


def build_softmax(dtype, config, seed):
    values = host_values(config["shape"], dtype, seed)
    x = native_core(values, dtype)
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted.astype(NUMPY_DTYPES[dtype]))
    expected = (exponentials
                / exponentials.sum(axis=-1, keepdims=True)).astype(
                    NUMPY_DTYPES[dtype])

    def call():
        return x.softmax()

    def verify():
        out = call()
        try:
            return gate_tolerance(out.to_numpy(), expected, dtype, "softmax")
        finally:
            out.close()

    return Case(call, verify, lambda: close_all(x))


def build_cross_entropy(dtype, config, seed):
    rows, classes = config["shape"]
    logits = native_core(host_values((rows, classes), dtype, seed), dtype)
    targets = (np.arange(rows, dtype=np.int64) % classes)

    def call():
        return logits.cross_entropy_forward(targets)

    def verify():
        result = call()
        try:
            return gate_finite(result.loss.to_numpy(), dtype, "cross_entropy")
        finally:
            result.close()

    return Case(call, verify, lambda: close_all(logits))


def build_layernorm_step(dtype, config, seed):
    """LayerNorm forward **and** backward — a composed normalization, so
    this measures the composition rather than a single kernel."""
    rows, features = config["shape"]
    module = NativeLayerNorm(features, dtype=dtype)
    x = native(host_values((rows, features), dtype, seed), dtype,
               requires_grad=True)

    def call():
        out = module(x)
        out.sum().backward()
        return out

    def verify():
        out = module(x)
        try:
            return gate_finite(out.to_numpy(), dtype, "layernorm_step")
        finally:
            out.close()

    def reset():
        x.zero_grad()
        for parameter in module.parameters():
            parameter.zero_grad()

    def teardown():
        close_all(x)
        for parameter in module.parameters():
            parameter.close()

    return Case(call, verify, teardown, reset)


def build_batchnorm_training_step(dtype, config, seed):
    """A BatchNorm training forward advances two running buffers, so the
    buffers are restored between repetitions — outside the timer — and
    every measured call therefore starts from identical state."""
    rows, features = config["shape"]
    module = NativeBatchNorm1d(features, dtype=dtype)
    module.train(True)
    x = native(host_values((rows, features), dtype, seed), dtype,
               requires_grad=True)
    # A caller-owned snapshot of the starting state, held for the life of
    # the case. Restoring through the public loader keeps every identity
    # and is the same transaction production code uses.
    initial = module.state_dict()

    def reset():
        x.zero_grad()
        for parameter in module.parameters():
            parameter.zero_grad()
        module.load_state_dict(initial)

    def call():
        out = module(x)
        out.sum().backward()
        return out

    def verify():
        out = module(x)
        try:
            gate = gate_finite(out.to_numpy(), dtype,
                               "batchnorm_training_step")
        finally:
            out.close()
        reset()
        return gate

    def teardown():
        close_all(x, *initial.values())
        for parameter in module.parameters():
            parameter.close()
        for _, buffer in module.named_buffers():
            buffer.close()

    return Case(call, verify, teardown, reset)


def build_dropout_step(dtype, config, seed):
    """Dropout forward and backward on a fixed generator.

    The generator is rewound between repetitions, **outside** the timer,
    so every measured call draws the same mask from the same call index.
    No claim is made that one dtype's mask is comparable to the other's
    beyond the established same-pattern contract, and none is asserted."""
    generator = NativeGenerator(4242)
    module = NativeDropout(0.25, generator=generator)
    module.train(True)
    x = native(host_values(config["shape"], dtype, seed), dtype,
               requires_grad=True)

    def reset():
        x.zero_grad()
        generator.reset()

    def call():
        out = module(x)
        out.sum().backward()
        return out

    def verify():
        out = module(x)
        try:
            gate = gate_finite(out.to_numpy(), dtype, "dropout_step")
        finally:
            out.close()
        reset()
        return gate

    return Case(call, verify, lambda: close_all(x), reset)


def _optimizer_case(dtype, config, seed, optimizer_class, label):
    """One optimizer ``step()`` with a stable gradient already present.

    The parameter is restored between repetitions so every measured step
    starts from the same value; for Adam the moments and the step counter
    are part of that state, so a fresh optimizer is built per reset —
    outside the timer."""
    shape = config["shape"]
    start = host_values(shape, dtype, seed)
    gradient_host = host_values(shape, dtype, seed + 1)
    parameter = cpp.NativeTensorCore  # placeholder, replaced below
    from tensorforge.experimental import NativeParameter

    state = {}

    def fresh():
        close_state()
        state["parameter"] = NativeParameter(start, dtype=dtype)
        state["grad"] = native(gradient_host, dtype)
        state["parameter"]._grad = state["grad"]
        state["optimizer"] = optimizer_class([state["parameter"]], lr=0.01)

    def close_state():
        optimizer = state.pop("optimizer", None)
        if optimizer is not None and hasattr(optimizer, "close"):
            optimizer.close()
        close_all(state.pop("grad", None), state.pop("parameter", None))

    def call():
        state["optimizer"].step()
        return None

    def verify():
        fresh()
        state["optimizer"].step()
        moved = state["parameter"].to_numpy()
        gate = gate_finite(moved, dtype, label)
        if np.array_equal(moved, start):
            raise AssertionError(f"{label}: the step did not move the "
                                 f"parameter, so this would time nothing")
        fresh()
        return gate

    fresh()
    return Case(call, verify, close_state, fresh)


def build_sgd_step(dtype, config, seed):
    return _optimizer_case(dtype, config, seed, NativeSGD, "sgd_step")


def build_adam_step(dtype, config, seed):
    return _optimizer_case(dtype, config, seed, NativeAdam, "adam_step")


class BenchmarkModel(NativeModule):
    """The I9 integrated architecture, at one explicit dtype.

    Deliberately **not** the exact-resume proof: this measures one
    training step, and rerunning a checkpoint round trip inside every
    timed repetition would measure the filesystem. The full state is
    validated in the correctness gate instead."""

    def __init__(self, dtype, seed=0, generator_seed=20260802):
        super().__init__()
        generator = NativeGenerator(generator_seed)
        self.conv = NativeConv2d(1, 4, 3, padding=1, seed=seed, dtype=dtype)
        self.norm2d = NativeBatchNorm1d.__mro__ and None  # placeholder
        del self.norm2d
        self.relu1 = NativeReLU()
        self.pool = NativeMaxPool2d(2)
        self.drop1 = NativeDropout(0.25, generator=generator)
        self.flatten = NativeFlatten()
        self.hidden = NativeLinear(36, 8, seed=seed + 1, dtype=dtype)
        self.norm = NativeBatchNorm1d(8, dtype=dtype)
        self.relu2 = NativeReLU()
        self.layer_norm = NativeLayerNorm(8, dtype=dtype)
        self.drop2 = NativeDropout(0.25, generator=generator)
        self.output = NativeLinear(8, 3, seed=seed + 2, dtype=dtype)

    def forward(self, images):
        h = self.drop1(self.pool(self.relu1(self.conv(images))))
        h = self.layer_norm(self.relu2(self.norm(self.hidden(
            self.flatten(h)))))
        return self.output(self.drop2(h))


def build_training_step(dtype, config, seed):
    """One complete deterministic training iteration: forward, loss,
    backward, optimizer step — the layer a user actually pays for."""
    batch = config["batch"]
    model = BenchmarkModel(dtype, seed=seed % 1000)
    model.train(True)
    loss_fn = NativeCrossEntropyLoss()
    optimizer = NativeAdam(model.parameters(), lr=0.01)
    generator = list(model.generators())[0]
    images = native(host_values((batch, 1, 6, 6), dtype, seed), dtype)
    targets = np.arange(batch, dtype=np.int64) % 3
    # Parameters, buffers, and the generator all advance on a training
    # step, so all three are restored between repetitions — outside the
    # timer — and supposedly identical repetitions really are identical.
    initial = model.state_dict()

    def reset():
        generator.reset()
        model.load_state_dict(initial)
        optimizer.zero_grad()

    def call():
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        close_all(logits)
        return loss

    def verify():
        logits = model(images)
        loss = loss_fn(logits, targets)
        try:
            gate = gate_finite(loss.to_numpy(), dtype, "training_step")
            # Every state object really is at the run dtype, so a case
            # that silently ran at the wrong width cannot be timed.
            for name, parameter in model.named_parameters():
                if parameter.dtype != dtype:
                    raise AssertionError(
                        f"training_step: parameter {name} is "
                        f"{parameter.dtype}, expected {dtype}")
            for name, buffer in model.named_buffers():
                if buffer.dtype != dtype:
                    raise AssertionError(
                        f"training_step: buffer {name} is {buffer.dtype}, "
                        f"expected {dtype}")
        finally:
            close_all(loss, logits)
        reset()
        return gate

    def teardown():
        optimizer.close()
        close_all(images, *initial.values())
        for parameter in model.parameters():
            parameter.close()
        for _, buffer in model.named_buffers():
            buffer.close()

    return Case(call, verify, teardown, reset)


def build_control(dtype, config, seed):
    """The control: an ordinary contiguous elementwise multiply.

    ``control_identical`` and ``control_twin`` are byte-identical code on
    byte-identical inputs, so the difference between their medians is this
    machine's noise floor — the **control band**. A case reading inside it
    is neutral whatever its sign. It is never a gate."""
    return _elementwise(dtype, config["shape"], config["shape"], seed,
                        "control")


# ==========================================================================
# The case registry
# ==========================================================================


CASES = {
    # -- transfer ----------------------------------------------------------
    "host_ingress": {
        "family": "transfer",
        "build": build_host_ingress,
        "seed": 20260901,
        "operation": "NativeTensor.from_array() at an explicit dtype",
        "gate": BITWISE,
        "configurations": {"full": {"shape": (512, 512)},
                           "smoke": {"shape": (8, 8)}},
        "notes": ("The explicit host-to-native conversion boundary. The "
                  "dtype is always passed; it is never inferred from the "
                  "input array."),
    },
    "host_egress": {
        "family": "transfer",
        "build": build_host_egress,
        "seed": 20260902,
        "operation": "to_numpy() materialization at the storage dtype",
        "gate": BITWISE,
        "configurations": {"full": {"shape": (512, 512)},
                           "smoke": {"shape": (8, 8)}},
        "notes": "Egress reproduces the storage dtype exactly and never widens.",
    },
    "contiguous_copy": {
        "family": "transfer",
        "build": build_contiguous_copy,
        "seed": 20260903,
        "operation": "storage-to-storage identity copy",
        "gate": BITWISE,
        "configurations": {"full": {"shape": (512, 512)},
                           "smoke": {"shape": (8, 8)}},
        "notes": ("The value-transfer primitive: bit-preserving at both "
                  "widths, so the gate is bitwise by contract."),
    },
    "strided_materialize": {
        "family": "transfer",
        "build": build_strided_materialize,
        "seed": 20260904,
        "operation": "materializing a transposed view",
        "gate": BITWISE,
        "configurations": {"full": {"shape": (512, 512)},
                           "smoke": {"shape": (8, 8)}},
        "notes": "The non-contiguous gather, which cannot take the flat path.",
    },
    # -- elementwise -------------------------------------------------------
    "elementwise_contiguous": {
        "family": "elementwise",
        "build": build_elementwise_contiguous,
        "seed": 20260905,
        "operation": "multiply() on two contiguous same-shape tensors",
        "gate": BITWISE,
        "configurations": {"full": {"shape": (512, 512)},
                           "smoke": {"shape": (8, 8)}},
        "notes": ("Bandwidth-sensitive by construction: three streams of "
                  "one element size, one multiply each."),
    },
    "elementwise_broadcast": {
        "family": "elementwise",
        "build": build_elementwise_broadcast,
        "seed": 20260906,
        "operation": "multiply() against a broadcast row",
        "gate": BITWISE,
        "configurations": {"full": {"shape": (512, 512)},
                           "smoke": {"shape": (8, 8)}},
        "notes": ("The zero-stride read model. The collapsed plan is int64 "
                  "layout metadata and is dtype-independent."),
    },
    "elementwise_small": {
        "family": "elementwise",
        "build": build_elementwise_small,
        "seed": 20260907,
        "operation": "multiply() on a (4, 4) tensor",
        "gate": BITWISE,
        "size_independent": True,
        "configurations": {"full": {"shape": (4, 4)},
                           "smoke": {"shape": (4, 4)}},
        "notes": ("Below roughly 1,000 elements the fixed ~7-12 us "
                  "Python-plus-ctypes cost dominates and the kernel work is "
                  "invisible. Expect both dtypes to read the same here; "
                  "that is the architectural floor, not a defect."),
    },
    # -- reduction ---------------------------------------------------------
    "reduction_contiguous": {
        "family": "reduction",
        "build": build_reduction_contiguous,
        "seed": 20260908,
        "operation": "sum(axis=0) over a contiguous tensor",
        "gate": SUMMATION_BOUND,
        "configurations": {"full": {"shape": (512, 512)},
                           "smoke": {"shape": (8, 8)}},
        "notes": "H6's contiguous-block factorization, at both widths.",
    },
    "reduction_strided": {
        "family": "reduction",
        "build": build_reduction_strided,
        "seed": 20260909,
        "operation": "sum(axis=0) over a transposed view",
        "gate": SUMMATION_BOUND,
        "configurations": {"full": {"shape": (512, 512)},
                           "smoke": {"shape": (8, 8)}},
        "notes": ("The retained generic odometer. Per-output accumulation "
                  "order is preserved exactly at both widths."),
    },
    # -- matmul ------------------------------------------------------------
    "matmul_contiguous": {
        "family": "matmul",
        "build": build_matmul_contiguous,
        "seed": 20260910,
        "operation": "matmul() with both operands contiguous",
        "gate": SUMMATION_BOUND,
        "configurations": {"full": {"shape": (192, 192)},
                           "smoke": {"shape": (8, 8)}},
        "notes": "The layout H2's row-sweep predicate selects.",
    },
    "matmul_transposed_view": {
        "family": "matmul",
        "build": build_matmul_transposed_view,
        "seed": 20260911,
        "operation": "matmul() with a transposed right operand",
        "gate": SUMMATION_BOUND,
        "configurations": {"full": {"shape": (192, 192)},
                           "smoke": {"shape": (8, 8)}},
        "notes": ("The retained generic traversal. Both widths take the "
                  "same path for the same layout, because the predicate "
                  "reads layout metadata alone."),
    },
    # -- CNN ---------------------------------------------------------------
    "conv2d_forward": {
        "family": "cnn",
        "build": build_conv2d_forward,
        "seed": 20260912,
        "operation": "conv2d forward, 3x3, padding 1",
        "gate": "finite",
        "configurations": {
            "full": {"shape": (8, 4, 24, 24), "out_channels": 8},
            "smoke": {"shape": (1, 1, 6, 6), "out_channels": 2}},
        "notes": "H9's traversals, unchanged and dtype-general.",
    },
    "conv2d_input_backward": {
        "family": "cnn",
        "build": build_conv2d_input_backward,
        "seed": 20260913,
        "operation": "conv2d input-gradient, 3x3, padding 1",
        "gate": "finite",
        "configurations": {
            "full": {"shape": (8, 4, 24, 24), "out_channels": 8},
            "smoke": {"shape": (1, 1, 6, 6), "out_channels": 2}},
        "notes": "The gather traversal H9 selected for this direction.",
    },
    "maxpool2d_forward": {
        "family": "cnn",
        "build": build_maxpool2d_forward,
        "seed": 20260914,
        "operation": "maxpool2d forward, 2x2",
        "gate": "finite",
        "configurations": {"full": {"shape": (8, 4, 24, 24)},
                           "smoke": {"shape": (1, 1, 4, 4)}},
        "notes": ("The winner buffer is private float64 metadata at every "
                  "value dtype, so this case allocates one float64 buffer "
                  "whichever width it runs at."),
    },
    # -- classification ----------------------------------------------------
    "softmax": {
        "family": "classification",
        "build": build_softmax,
        "seed": 20260915,
        "operation": "softmax over the last axis",
        "gate": TOLERANCE,
        "configurations": {"full": {"shape": (256, 64)},
                           "smoke": {"shape": (4, 3)}},
        "notes": "The maximum shift and the normalizing sum, at the element type.",
    },
    "cross_entropy_forward": {
        "family": "classification",
        "build": build_cross_entropy,
        "seed": 20260916,
        "operation": "fused cross-entropy forward with int64 targets",
        "gate": "finite",
        "configurations": {"full": {"shape": (256, 64)},
                           "smoke": {"shape": (4, 3)}},
        "notes": ("Targets are host int64 metadata at every width and are "
                  "not part of the dtype axis."),
    },
    # -- normalization -----------------------------------------------------
    "layernorm_step": {
        "family": "normalization",
        "build": build_layernorm_step,
        "seed": 20260917,
        "operation": "LayerNorm forward and backward",
        "gate": "finite",
        "configurations": {"full": {"shape": (256, 128)},
                           "smoke": {"shape": (4, 8)}},
        "notes": ("A composition, not a kernel: normalization has no export "
                  "of its own at either width."),
    },
    "batchnorm_training_step": {
        "family": "normalization",
        "build": build_batchnorm_training_step,
        "seed": 20260918,
        "operation": "BatchNorm1d training forward and backward",
        "gate": "finite",
        "configurations": {"full": {"shape": (256, 128)},
                           "smoke": {"shape": (4, 8)}},
        "notes": ("The call advances two running buffers, so they are "
                  "restored between repetitions, outside the timer."),
    },
    # -- dropout -----------------------------------------------------------
    "dropout_step": {
        "family": "dropout",
        "build": build_dropout_step,
        "seed": 20260919,
        "operation": "Dropout training forward and backward",
        "gate": "finite",
        "configurations": {"full": {"shape": (256, 128)},
                           "smoke": {"shape": (4, 8)}},
        "notes": ("The generator is rewound between repetitions, outside "
                  "the timer, so every measured call draws the same mask "
                  "from the same call index. The random derivation is "
                  "binary64 at both widths by design."),
    },
    # -- optimizer ---------------------------------------------------------
    "sgd_step": {
        "family": "optimizer",
        "build": build_sgd_step,
        "seed": 20260920,
        "operation": "NativeSGD.step() with a gradient already present",
        "gate": "finite",
        "configurations": {"full": {"shape": (512, 512)},
                           "smoke": {"shape": (8, 8)}},
        "notes": "H4's per-step scalar holder, at the parameter's own width.",
    },
    "adam_step": {
        "family": "optimizer",
        "build": build_adam_step,
        "seed": 20260921,
        "operation": "NativeAdam.step() with a gradient already present",
        "gate": "finite",
        "configurations": {"full": {"shape": (512, 512)},
                           "smoke": {"shape": (8, 8)}},
        "notes": ("Adam's moments carry their parameter's dtype. The scalar "
                  "caches key on (dtype, device), so a single-width "
                  "collection builds one scalar set."),
    },
    # -- integrated training ----------------------------------------------
    "training_step": {
        "family": "training",
        "build": build_training_step,
        "seed": 20260922,
        "operation": ("one training iteration of the I9 integrated model: "
                      "forward, fused loss, backward, Adam step"),
        "gate": "finite",
        "configurations": {"full": {"batch": 32}, "smoke": {"batch": 4}},
        "notes": ("Deliberately not the exact-resume proof: rerunning a "
                  "checkpoint round trip inside every timed repetition "
                  "would measure the filesystem. The full state is "
                  "validated in the correctness gate instead."),
    },
    # -- control -----------------------------------------------------------
    "control_identical": {
        "family": "control",
        "build": build_control,
        "seed": 20260923,
        "operation": "the control case (identical code, run A)",
        "gate": BITWISE,
        "configurations": {"full": {"shape": (256, 256)},
                           "smoke": {"shape": (8, 8)}},
        "notes": "Half of the control pair; see control_twin.",
    },
    "control_twin": {
        "family": "control",
        "build": build_control,
        "seed": 20260923,          # deliberately the same seed
        "operation": "the control case (identical code, run B)",
        "gate": BITWISE,
        "configurations": {"full": {"shape": (256, 256)},
                           "smoke": {"shape": (8, 8)}},
        "notes": ("Byte-identical code and inputs to control_identical. The "
                  "difference between their medians is this machine's noise "
                  "floor — the control band. Never a gate."),
    },
}


def cases_for(family=None, case=None):
    if case is not None:
        return [case]
    if family is not None:
        return [name for name, spec in CASES.items()
                if spec["family"] == family]
    return list(CASES)


# ==========================================================================
# Environment
# ==========================================================================


def environment():
    """Real introspection, never a restatement of what was requested.

    Anything that cannot be determined is reported as ``None`` rather than
    guessed — a fabricated CPU model would make a published table worse
    than useless."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "backend_available": cpp.is_available(),
    }
    try:
        import os
        info["cpu_count_logical"] = os.cpu_count()
    except Exception:                                    # pragma: no cover
        info["cpu_count_logical"] = None
    if cpp.is_available():
        details = cpp.backend_info()
        info["backend_default_dtype"] = details["dtype"]
        info["backend_supported_dtypes"] = list(details["supported_dtypes"])
        info["backend_raw_kernel_dtypes"] = list(
            details["raw_kernel_dtypes"])
    return info


# ==========================================================================
# Runner
# ==========================================================================


def run_case(name, dtype, warmup, repetitions, smoke):
    spec = CASES[name]
    variant = "smoke" if smoke else "full"
    config = spec["configurations"][variant]
    case = spec["build"](dtype, config, spec["seed"])
    try:
        # The gate runs *first*, unconditionally. A failure propagates
        # before ``measure`` is ever reached, so no timing is published.
        correctness = case.verify()
        if case.reset is not None:
            case.reset()
        samples = measure(case, warmup, repetitions)
    finally:
        case.teardown()
    row = {
        "case": name,
        "family": spec["family"],
        "dtype": dtype,
        "operation": spec["operation"],
        "configuration": variant,
        "config": dict(config),
        "notes": spec["notes"],
        "correctness": {"passed": True, "gate": spec["gate"], **correctness},
    }
    row.update(summarize(samples))
    return row


def run_benchmark(cases=None, family=None, dtypes=None, warmup=None,
                  repetitions=None, smoke=False):
    if not cpp.is_available():
        raise RuntimeError(
            "the experimental C++ backend is not built; "
            + cpp.build_instructions())
    defaults = SMOKE_DEFAULTS if smoke else DEFAULTS
    warmup = defaults["warmup"] if warmup is None else warmup
    repetitions = (defaults["repetitions"] if repetitions is None
                   else repetitions)
    if warmup < 0:
        raise ValueError("--warmup must not be negative")
    if repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    selected = cases if cases is not None else cases_for(family=family)
    for name in selected:
        if name not in CASES:
            raise ValueError(f"unknown case {name!r}")
    dtypes = list(dtypes) if dtypes else list(DTYPES)
    for dtype in dtypes:
        if dtype not in DTYPES:
            raise ValueError(f"unknown dtype {dtype!r}")

    rows = []
    # Deterministic alternation: every case is run at each dtype in
    # registry order, case by case, so a slow drift in machine state
    # touches both widths alike instead of loading one of them.
    for name in selected:
        for dtype in dtypes:
            rows.append(run_case(name, dtype, warmup, repetitions, smoke))
    return {
        "harness": "benchmark_native_dtype",
        "milestone": "I10",
        "environment": environment(),
        "warmup": warmup,
        "repetitions": repetitions,
        "mode": "smoke" if smoke else "full",
        "publishable": (not smoke
                        and PUBLISHABLE_MINIMUM <= repetitions
                        <= PUBLISHABLE_MAXIMUM),
        "dtypes": dtypes,
        "rows": rows,
        "disclaimer": (
            "Local characterization of one machine, one build, and one "
            "moment. No speed is asserted, no threshold exists, and no "
            "result file is written. The two dtypes are measured "
            "separately and never divided by one another."
        ),
    }


def control_band(rows):
    """The observed noise floor per dtype, from the control pair.

    Returned as the relative difference between two byte-identical
    measurements. A case whose reading sits inside this band is neutral,
    whatever its sign."""
    band = {}
    for dtype in DTYPES:
        pair = [row for row in rows
                if row["family"] == "control" and row["dtype"] == dtype]
        if len(pair) != 2:
            continue
        first, second = (row["median_s"] for row in pair)
        reference = min(first, second)
        if reference > 0:
            band[dtype] = abs(first - second) / reference
    return band


def format_report(payload):
    lines = []
    add = lines.append
    add("Native dtype characterization (Phase I, milestone I10)")
    add("=" * 70)
    add("")
    add(payload["disclaimer"])
    add("")
    environment_info = payload["environment"]
    add("Environment")
    add("-" * 70)
    for key in sorted(environment_info):
        add(f"  {key:26s} {environment_info[key]}")
    add(f"  {'mode':26s} {payload['mode']}")
    add(f"  {'warmup':26s} {payload['warmup']}")
    add(f"  {'repetitions':26s} {payload['repetitions']}")
    if not payload["publishable"]:
        add("")
        add(f"  NOTE: this run is not publishable evidence. Quotable "
            f"figures need {PUBLISHABLE_MINIMUM}-{PUBLISHABLE_MAXIMUM} "
            f"measured repetitions in full mode;")
        add("        low round counts have repeatedly read as regressions "
            "that vanished at 21-25.")
    add("")

    band = control_band(payload["rows"])
    if band:
        add("Control band (identical code measured twice)")
        add("-" * 70)
        for dtype, value in band.items():
            add(f"  {dtype:10s} {value * 100:6.2f}%   "
                f"readings within this are neutral, whatever their sign")
        add("")
        add("  The control band is this machine's noise floor, measured on")
        add("  byte-identical code and inputs. It is an observation, never a")
        add("  gate: no case fails for sitting outside it, and no case")
        add("  passes for sitting inside it. On a machine with heterogeneous")
        add("  cores the band can exceed the difference between any two")
        add("  cases, in which case the honest conclusion is that this run")
        add("  distinguishes nothing — and that is a result worth printing,")
        add("  not a run worth repeating until it looks tidier.")
        add("")

    for dtype in payload["dtypes"]:
        rows = [row for row in payload["rows"] if row["dtype"] == dtype]
        if not rows:
            continue
        add(f"dtype: {dtype}")
        add("-" * 70)
        add(f"  {'case':28s} {'median':>11s} {'IQR':>11s} "
            f"{'rel.IQR':>8s}  gate")
        for row in rows:
            add(f"  {row['case']:28s} "
                f"{row['median_s'] * 1e6:9.2f}us "
                f"{row['iqr_s'] * 1e6:9.2f}us "
                f"{row['relative_iqr'] * 100:7.2f}%  "
                f"{row['correctness']['gate']}")
        add("")

    add("Reading this table")
    add("-" * 70)
    add("  Each dtype is its own characterization. There is deliberately")
    add("  no float32/float64 ratio anywhere in this output: that number")
    add("  is a property of one machine's memory bandwidth, not of")
    add("  TensorForge, and publishing it would turn a measurement into a")
    add("  promise the project cannot keep across machines.")
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description=("Characterize the native CPU runtime at float64 and "
                     "float32 separately (measurement only; no speed is "
                     "asserted, and no result file is written)."))
    parser.add_argument("--case", choices=tuple(CASES), default=None,
                        help="run a single case (default: all)")
    parser.add_argument("--family", choices=FAMILIES, default=None,
                        help="run one family (default: all)")
    parser.add_argument("--dtype", choices=DTYPES, action="append",
                        default=None,
                        help="measure only this dtype (repeatable)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="warm-up repetitions before measuring")
    parser.add_argument("--repetitions", type=int, default=None,
                        help=f"measured repetitions per case "
                             f"(publishable: {PUBLISHABLE_MINIMUM}-"
                             f"{PUBLISHABLE_MAXIMUM})")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON to stdout only")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny shapes and counts, for tests and CI")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.case and args.family:
        parser.error("--case selects one case; do not combine it with "
                     "--family")
    try:
        payload = run_benchmark(
            cases=[args.case] if args.case else None,
            family=args.family,
            dtypes=args.dtype,
            warmup=args.warmup,
            repetitions=args.repetitions,
            smoke=args.smoke,
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))      # stderr, exit 2 — stdout stays clean
    except AssertionError as error:   # a correctness gate failed
        parser.exit(1, f"correctness gate failed: {error}\n")
    if args.json:
        print(json.dumps(payload))
    else:
        print(format_report(payload))


if __name__ == "__main__":
    main()
