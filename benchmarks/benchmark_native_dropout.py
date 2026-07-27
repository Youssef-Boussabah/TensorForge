"""Characterization benchmark for the native Dropout stack (Advanced C++
Phase G, milestone G8).

This measures the Dropout surface G1-G7 already shipped — the stateless
``NativeTensorCore`` forward, the differentiable ``NativeTensor.dropout``
operation with its generator call transaction, the ``NativeDropout``
module in both modes, and one complete Dropout training step. It does
**not** try to make anything take less time, and **nothing here asserts a
speed**: G8 is measurement only, so no runtime file was changed, no fast
path was added, and no number below is a threshold. This is an honest,
reproducible snapshot of one machine at one moment. It is **not** a
performance contract, not comparable across machines without controlled
conditions, and there is no CI timing threshold anywhere in this
repository.

**Correctness runs before timing, always.** A global prologue pins the
benchmark's vectorized NumPy reference to the committed G2 known-answer
vectors and then pins the **native kernel** to the same vectors; after
that every case validates its native result against a reference, and a
failed gate aborts the run with a nonzero exit status and publishes no
timing at all.

Reference labels
----------------

- ``numpy`` — an **exact** vectorized NumPy implementation of the locked
  ``tensorforge.splitmix64`` derivation (§4.2-§4.4 of
  docs/native_rng_dropout_design.md) doing the *same work*: the per-call
  stream key, one 64-bit word per logical element, the top-53-bit uniform
  conversion, the strict ``u < p`` test, the ``1/(1 - p)`` multiplier
  mask, and the output multiply, with both the mask and the output
  allocated. Agreement is asserted **bit for bit**, not to a tolerance.
- ``native_only`` — no equivalent implementation of the *same semantics*
  exists to time. Every ``NativeTensor``/``NativeDropout`` case is
  ``native_only``: those layers own a generator call transaction, an
  autograd graph, and native ownership that a NumPy expression does not
  have, so timing one against the other would compare different work.
  Their correctness gates are still exact, against the same reference at
  the reserved call index.
- ``harness_baseline`` — the harness's own Python call/loop floor, so the
  identity-dispatch rows below can be read against something.

A ``tensorforge.nn.Dropout`` comparison is deliberately **omitted**: the
stable module draws from NumPy's RNG, so no mask-for-mask comparison
exists, and timing two different mask distributions against each other
would be a comparison of RNG implementations dressed up as a Dropout
benchmark (design §16).

What is timed
-------------

One measured repetition times exactly one call of the case's operation
with ``time.perf_counter_ns()``. Setup — input and view construction,
module and generator construction, graph construction for the
backward-only case, model/optimizer construction for the training step —
happens **outside** the timer, and cleanup happens outside it too. Graph
construction *is* inside the timer for the forward and training-step
cases, because it is part of the call being characterized. No sample is
discarded, no timer overhead is subtracted, and ``gc.collect()`` is never
called inside a timed region.

**Iterations per sample.** Allocating cases use exactly one call per
sample (``iteration_policy = "one_call_per_sample"``), the repository's
existing practice. The identity-dispatch cases — evaluation mode and
``p == 0``, which return the caller's own tensor and allocate nothing —
are far below the useful resolution of one wall-clock reading, so they
run a short **calibrated** inner loop instead
(``iteration_policy = "calibrated_identity_loop"``) sized to a target
sample duration, and the reported time is the sample divided by the
iteration count. That inner loop's own Python overhead is *not*
subtracted; the ``python_call_floor`` case measures the same loop around
a trivial function so the floor is visible rather than assumed.

Generator state
---------------

Every stochastic case owns **one** ``NativeGenerator`` for its whole run,
so call indices advance monotonically through the gate, the warm-up, the
calibration, and the samples; the consumed range is recorded and verified
**exactly** against the number of cycles the harness performed. Identity
and evaluation cases must consume **zero** calls, and that is verified
the same way. No generator is ever reset inside a timed region, no
reservation is reused, and no case approaches counter exhaustion.

Honest limits
-------------

These are wall-clock timings on a busy general-purpose operating system.
Scheduling, frequency scaling, and cache state all move them. Ratios
describe the measured run and nothing else; they are not portable
performance guarantees, and no statistically rigorous hardware
characterization is claimed. No memory-bandwidth figure is reported,
because an honest one would need a byte-count definition this harness
does not attempt.

Build the backend first:

    uv run python cpp/build.py

Then, for example:

    uv run python benchmarks/benchmark_native_dropout.py
    uv run python benchmarks/benchmark_native_dropout.py --smoke
    uv run python benchmarks/benchmark_native_dropout.py --quick
    uv run python benchmarks/benchmark_native_dropout.py --smoke --json
    uv run python benchmarks/benchmark_native_dropout.py --family layout
    uv run python benchmarks/benchmark_native_dropout.py --case module_eval_forward

``--quick`` is an alias for ``--smoke``. **No result file of any kind is
written** unless ``--json-out PATH`` explicitly names one; there is no
default output path, no results directory, and no committed artifact.

G8 adds **no** capability: no kernel, C ABI export, ctypes symbol, Core
method, autograd operation, module, export, registry entry, or checkpoint
change. ``"dropout"`` stays in ``UNSUPPORTED`` until the G10 closure. The
G7 proof (``examples/native_dropout_training.py``) remains the separate,
authoritative correctness-and-resume artifact.
"""

import argparse
import gc
import json
import math
import os
import platform
import statistics
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeCrossEntropyLoss,
    NativeDropout,
    NativeGenerator,
    NativeLinear,
    NativeModule,
    NativeReLU,
    NativeTensor,
)

BENCHMARK_NAME = "tensorforge.native_dropout"
BENCHMARK_VERSION = "1.0"
# The benchmark's own result-payload schema version. Deliberately local
# to this harness: it is **not** the native checkpoint format version
# (which is 2 and which G8 does not touch), and the two move
# independently.
RESULT_SCHEMA_VERSION = "1.0"

# Reference labels. Every case declares exactly one.
NUMPY = "numpy"
NATIVE_ONLY = "native_only"
HARNESS = "harness_baseline"

# Warm-up / repetition defaults. Higher than the Phase-E and Phase-F
# harnesses' 3/12 for one measurement reason, not a presentational one:
# every case here is microseconds rather than milliseconds, so ordinary
# operating-system noise is a large fraction of a single reading, and the
# whole default run still finishes in about a second. The counts apply
# identically to the native and reference paths, so nothing is favoured.
DEFAULTS = {"warmup": 5, "repetitions": 20}
SMOKE_DEFAULTS = {"warmup": 1, "repetitions": 3}
# Heavier cases declare their own smaller cap; the count actually used is
# always reported per case.
BACKWARD_REPETITIONS = 12
TRAINING_STEP_REPETITIONS = 10

# Calibrated identity loop bounds. Only cases whose call allocates
# nothing and returns the caller's own object use these.
CALIBRATION = {"target_ns": 200_000, "maximum": 4096}
SMOKE_CALIBRATION = {"target_ns": 20_000, "maximum": 256}

# The fixed random key the benchmark uses wherever a key is not swept.
# A high-bit seed and a nonzero call index, per the G8 brief; both are
# ordinary values for the locked derivation, and neither is special.
BENCHMARK_SEED = 0x8000000000000000
BENCHMARK_CALL_INDEX = 7
# The representative probability. 0.5 is the module default.
DEFAULT_P = 0.5
# The largest legal probability: `p == 1` is rejected (design §6.3).
MAX_P = math.nextafter(1.0, 0.0)

# The probability sweep, in order. `p == 0` is identity at the operation
# and module layers, which is why those two layers' zero rows live in the
# `tensor_operation` and `module` families with the other identity cases.
PROBABILITY_SWEEP = (
    ("p000", 0.0),
    ("p010", 0.1),
    ("p050", DEFAULT_P),
    ("p090", 0.9),
    ("pmax", MAX_P),
)

# Training-step hyperparameters (the G7 architecture, minus the
# normalization layers this benchmark does not characterize).
STEP_FEATURES = 4
STEP_HIDDEN = 8
STEP_CLASSES = 3
STEP_SAMPLES = 12
STEP_LR = 0.05
STEP_HIDDEN_SEED = 0
STEP_OUTPUT_SEED = 1
STEP_DROPOUT_SEED = 20240707

# Lifecycle verification cycles.
LIFECYCLE_CYCLES = 5
SMOKE_LIFECYCLE_CYCLES = 2


# ---------------------------------------------------------------------------
# The committed known-answer vectors (docs/native_rng_dropout_design.md
# §4.7, identical to tests/test_native_dropout_core.py and
# cpp/tests/test_dropout_forward.cpp).
#
# These literals ARE the specification. The benchmark's vectorized NumPy
# reference is pinned to them **before** it is allowed to generate a
# single expectation, and the native kernel is pinned to them too, so a
# fast wrong result can never be published as a benchmark number.
# ---------------------------------------------------------------------------

UINT64_MAX = 2 ** 64 - 1
GOLDEN = 0x9E3779B97F4A7C15
MIX_A = 0xBF58476D1CE4E5B9
MIX_B = 0x94D049BB133111EB
MAX_ISSUED_CALL_INDEX = UINT64_MAX - 1

MIX64_VECTORS = (
    (0x0000000000000000, 0x0000000000000000),
    (0x0000000000000001, 0x5692161D100B05E5),
    (0x0000000000000002, 0xDBD238973A2B148A),
    (0x9E3779B97F4A7C15, 0xE220A8397B1DCDAF),
    (0x8000000000000000, 0x25C26EA579CEA98A),
    (0xFFFFFFFFFFFFFFFF, 0xB4D055FCF2CBBD7B),
)

STREAM_VECTORS = (
    (0x0000000000000000, 0, 0xE220A8397B1DCDAF),
    (0x0000000000000000, 1, 0x6E789E6AA1B965F4),
    (0x0000000000000000, 2, 0x06C45D188009454F),
    (0x0123456789ABCDEF, 0, 0x157A3807A48FAA9D),
    (0x0123456789ABCDEF, 7, 0x8931545F4F9EA651),
    (0x8000000000000000, 0, 0x481EC0A212A9F3DB),
    (0xFFFFFFFFFFFFFFFF, 0, 0xE4D971771B652C20),
    (0x0000000000000000, MAX_ISSUED_CALL_INDEX, 0x336503C6B835BEC0),
    (0x0123456789ABCDEF, MAX_ISSUED_CALL_INDEX, 0x20BEC7299668A13F),
)

# name -> (seed, call_index, p, first four element words, keep pattern
# over twelve logical elements).
DROPOUT_VECTORS = {
    "zero_seed_call0": (
        0x0000000000000000, 0, 0.25,
        (0xA706DD2F4D197E6F, 0xB382A305F4414F5E,
         0x631A9154FBABF717, 0xA80ABA8C86640906),
        "111110111110",
    ),
    "zero_seed_call1": (
        0x0000000000000000, 1, 0.25,
        (0x46B73E79F0C37C00, 0x374327C63D0CC8A6,
         0xE10CF86AE3079278, 0x26A223C360B54F32),
        "101011111011",
    ),
    "mixed_seed_call0": (
        0x0123456789ABCDEF, 0, 0.25,
        (0x021C88D0A3FD73B6, 0x498D3E51E781CDE0,
         0xA2A1796FEB7EF314, 0x1A2D33D4F57B4CD4),
        "011011111010",
    ),
    "mixed_seed_call7": (
        0x0123456789ABCDEF, 7, 0.75,
        (0x0184F08818982A99, 0x99E0A20D1E1F1641,
         0x3E9AD5FC011194F1, 0x52E464BC2FB3BF83),
        "000010000000",
    ),
    "high_bit_seed_call3": (
        0x8000000000000000, 3, 0.75,
        (0x94E05B24F614999E, 0xD58EE1DBADEF970D,
         0xE932E5239EC1F7C9, 0xB01B43DD212F69A7),
        "011000100000",
    ),
    "max_seed_call0": (
        0xFFFFFFFFFFFFFFFF, 0, 0.25,
        (0x5DC20AA7B2A27137, 0xBDA5668A01D7049C,
         0x82B43276ABB80226, 0xED4D5ED4A6EA59B4),
        "111110110110",
    ),
    "zero_seed_max_call": (
        0x0000000000000000, MAX_ISSUED_CALL_INDEX, 0.75,
        (0x53531EEB39C4C095, 0x1EACB2A4329B0259,
         0x2402CC7044E8B298, 0xAAB3D73BF633B046),
        "000001100001",
    ),
}

VECTOR_LENGTH = 12

# The equality-threshold vector: the one place `u < p` and `u <= p`
# disagree, so it pins the comparison direction itself.
EQUALITY_SEED = 0x0123456789ABCDEF
EQUALITY_CALL_INDEX = 0
EQUALITY_INDEX = 2
EQUALITY_WORD = 0xA2A1796FEB7EF314
EQUALITY_UNIFORM = float.fromhex("0x1.4542f2dfd6fdep-1")
EQUALITY_COUNT = 4
EQUALITY_KEEP_AT_EQUAL = "0010"
EQUALITY_KEEP_AT_NEXT = "0000"
EQUALITY_KEEP_UNDER_LESS_EQUAL = "0000"


# ---------------------------------------------------------------------------
# The exact vectorized NumPy reference.
#
# A benchmark-local implementation of §4.2-§4.4, deliberately **not** in
# the runtime package: a second production implementation would be a
# second source of truth and a silent NumPy fallback path. It is
# vectorized because a scalar Python loop would measure Python, not
# NumPy, and would make the comparison meaningless.
#
# It does exactly the work the native Core does: the per-call stream key,
# one word per logical element, the top-53-bit uniform, the strict `u < p`
# test, the single `1/(1 - p)` multiplier, the mask allocation, and the
# output multiply.
# ---------------------------------------------------------------------------


def _mix64_int(value):
    """The locked finalizer over a Python int, modulo 2**64."""
    value &= UINT64_MAX
    value ^= value >> 30
    value = (value * MIX_A) & UINT64_MAX
    value ^= value >> 27
    value = (value * MIX_B) & UINT64_MAX
    value ^= value >> 31
    return value


def _mix64_array(words):
    """The same finalizer over a uint64 NumPy array, in place.

    Unsigned NumPy integer arithmetic wraps, which is exactly the C++
    ``std::uint64_t`` semantics the kernel relies on."""
    words ^= words >> np.uint64(30)
    words *= np.uint64(MIX_A)
    words ^= words >> np.uint64(27)
    words *= np.uint64(MIX_B)
    words ^= words >> np.uint64(31)
    return words


def _reference_stream(seed, call_index):
    """``mix64(seed + GOLDEN * (call_index + 1))`` — the per-call key."""
    return _mix64_int(seed + GOLDEN * ((call_index + 1) & UINT64_MAX))


def _reference_words(count, seed, call_index):
    """One 64-bit word per logical element, in row-major order."""
    stream = np.uint64(_reference_stream(seed, call_index))
    elements = np.arange(1, count + 1, dtype=np.uint64)
    return _mix64_array(stream + np.uint64(GOLDEN) * elements)


def _reference_uniform(words):
    """The top 53 bits as an exact float64 in ``[0, 1)``."""
    return (words >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


def _reference_mask(shape, p, seed, call_index):
    """The multiplier mask: exactly ``0.0`` or the single ``1/(1 - p)``."""
    count = _element_count(shape)
    uniform = _reference_uniform(_reference_words(count, seed, call_index))
    scale = 1.0 / (1.0 - p)
    return np.where(uniform < p, 0.0, scale).reshape(shape)


def _reference_dropout(values, p, seed, call_index):
    """One complete reference forward: ``(output, mask)``.

    Both arrays are freshly allocated, exactly as the native Core
    allocates two owning cores."""
    mask = _reference_mask(values.shape, p, seed, call_index)
    return values * mask, mask


def _keep_pattern(mask):
    """``"1"`` where an element was kept, ``"0"`` where it was dropped."""
    return "".join("0" if value == 0.0 else "1"
                   for value in np.asarray(mask).reshape(-1))


def _element_count(shape):
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count


# ---------------------------------------------------------------------------
# Gate helpers. Every ``AssertionError`` here aborts the run before any
# timing is published.
# ---------------------------------------------------------------------------


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _require_exact(produced, expected, label):
    """Bit-for-bit equality. The Dropout mask is exactly ``0.0`` or one
    fixed multiplier and the output is one IEEE multiply of the same
    operands in both implementations, so there is no tolerance to pick:
    anything but exact equality is a defect."""
    produced = np.asarray(produced)
    expected = np.asarray(expected)
    if produced.shape != expected.shape:
        raise AssertionError(
            f"{label} has shape {produced.shape}, expected {expected.shape}"
        )
    if not np.array_equal(produced, expected):
        difference = np.argwhere(produced != expected)
        first = tuple(difference[0]) if len(difference) else ()
        raise AssertionError(
            f"{label} differs from the reference at {len(difference)} of "
            f"{produced.size} positions (first {first})"
        )


def _require_finite(values, label):
    if not np.all(np.isfinite(values)):
        raise AssertionError(f"{label} is not finite")


def _require_unchanged(produced, expected, label):
    if not np.array_equal(np.asarray(produced), np.asarray(expected)):
        raise AssertionError(f"{label} was mutated")


def _require_quiet_generator(generator, expected_calls, label):
    _require(generator.calls == expected_calls,
             f"{label}: the generator's call count is {generator.calls}, "
             f"expected {expected_calls}")
    _require(not generator._has_active_reservation(),
             f"{label}: a call reservation is still outstanding")


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


def _close_module(module):
    """There is no ``NativeModule.close()``, so a module's owner releases
    its parameters and buffers explicitly. A ``NativeGenerator`` owns no
    native storage and has no ``close()``."""
    for parameter in module.parameters():
        parameter.close()
    for buffer in module.buffers():
        buffer.close()


def _graph_resources(tensor):
    """The private native resources this node's graph history owns — for
    a Dropout node, exactly its multiplier mask. Read only by the gates,
    never inside a timed region."""
    return tuple(tensor._graph_resources)


def _settled(live):
    """The live native-storage count after one collection — the G6/G7
    convention.

    Explicit ``close()`` is the release mechanism; a collection only
    settles the autograd graph's backward-closure reference cycles and
    the gradient objects ``zero_grad()`` drops into a deterministic
    number. Never called inside a timed region."""
    gc.collect()
    return len(live)


@contextmanager
def _tracked_storage():
    """Count live native storages by wrapping the storage constructor and
    ``close()``.

    The technique the G6/G7 suites already use, kept **benchmark-local**:
    G8 adds no runtime API, so there is no public live-storage accessor to
    call. Both methods are restored on the way out, whatever happens."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)   # raises => never recorded
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    cpp.NativeStorage.__init__ = tracked_init
    cpp.NativeStorage.close = tracked_close
    try:
        yield open_ids
    finally:
        cpp.NativeStorage.__init__ = original_init
        cpp.NativeStorage.close = original_close


# ---------------------------------------------------------------------------
# Deterministic inputs, from an explicit arithmetic formula.
#
# No NumPy global RNG, no Python ``random``, no clock, no process id, no
# address, and no external data: element ``i`` is
# ``(-1)**i * (0.5 + (i % 9) / 4)``, so every value is a quarter, exactly
# representable in float64, never zero (so ``output == input * mask``
# discriminates at every position), mixed in sign, and non-constant with a
# period of 18 that does not line up with any mask pattern.
# ---------------------------------------------------------------------------


def _values(shape):
    count = _element_count(shape)
    index = np.arange(count, dtype=np.float64)
    magnitude = 0.5 + (index % 9.0) / 4.0
    sign = 1.0 - 2.0 * (index % 2.0)
    return (sign * magnitude).reshape(shape)


def _upstream_values(shape):
    """The deterministic upstream gradient. A plain ones() seed would let
    a mistaken backward that ignored the upstream pass; these do not."""
    count = _element_count(shape)
    index = np.arange(count, dtype=np.float64)
    return (0.25 + (index % 7.0) / 8.0).reshape(shape)


def _materialized(view):
    """Policy B's host equivalent: a non-contiguous input is materialized
    into row-major storage before the derivation runs, and a contiguous
    one is used as it is.

    ``np.ascontiguousarray`` is deliberately **not** called
    unconditionally: it promotes a 0-d array to shape ``(1,)``, which
    would silently change the rank-0 case's logical shape."""
    array = np.asarray(view)
    if array.ndim == 0 or array.flags["C_CONTIGUOUS"]:
        return array
    return np.ascontiguousarray(array)


def _core_from_values(values):
    """A contiguous ``NativeTensorCore`` holding a copy of ``values``.

    ``from_array`` cannot express rank 0, so a scalar goes through
    ``zeros(())`` plus a storage copy — the repository's convention."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        core = cpp.NativeTensorCore.zeros(())
        core._storage.copy_from(array.reshape(1))
        return core
    return cpp.NativeTensorCore.from_array(array)


def _layout_views(values, layout, config):
    """Build one native core view and the matching NumPy view for
    ``layout``, both carrying the **same logical values**.

    Returns ``(core, owned_cores, numpy_view)``; the caller closes every
    core in ``owned_cores``. The four layouts are the ones the Core
    contract distinguishes: contiguous, a transposed (non-contiguous,
    offset 0) view, a column-narrowed (non-contiguous, nonzero offset)
    view, and a row-narrowed (contiguous, nonzero offset) view."""
    values = np.asarray(values, dtype=np.float64)
    if layout == "contiguous":
        core = _core_from_values(values)
        return core, [core], values
    rows, columns = values.shape
    if layout == "transposed":
        base_values = np.ascontiguousarray(values.T)
        base = _core_from_values(base_values)
        view = base.T
        return view, [view, base], base_values.T
    if layout == "narrowed_noncontiguous":
        pad = int(config["column_offset"])
        base_values = np.zeros((rows, columns + pad + 8), dtype=np.float64)
        base_values[:, pad:pad + columns] = values
        base = _core_from_values(base_values)
        view = base.narrow(1, pad, columns)
        return view, [view, base], base_values[:, pad:pad + columns]
    if layout == "offset_contiguous":
        pad = int(config["row_offset"])
        base_values = np.zeros((rows + pad + 4, columns), dtype=np.float64)
        base_values[pad:pad + rows] = values
        base = _core_from_values(base_values)
        view = base.narrow(0, pad, rows)
        return view, [view, base], base_values[pad:pad + rows]
    raise ValueError(f"unknown layout {layout!r}")


# ---------------------------------------------------------------------------
# The correctness prologue. Runs once, before any case is built or timed.
# ---------------------------------------------------------------------------


def verify_reference():
    """Pin the benchmark's vectorized NumPy reference to the committed G2
    vectors, before it generates a single expectation.

    Returns the record that goes into the payload. Raises
    ``AssertionError`` — which aborts the whole run — on any mismatch."""
    for value, expected in MIX64_VECTORS:
        _require(_mix64_int(value) == expected,
                 f"the reference finalizer disagrees at {value:#x}")
    array_out = _mix64_array(
        np.array([value for value, _ in MIX64_VECTORS], dtype=np.uint64)
    )
    _require(
        list(int(word) for word in array_out)
        == [expected for _, expected in MIX64_VECTORS],
        "the vectorized finalizer disagrees with the scalar one",
    )
    for seed, call_index, expected in STREAM_VECTORS:
        _require(_reference_stream(seed, call_index) == expected,
                 f"the reference stream key disagrees at "
                 f"({seed:#x}, {call_index})")
    # The `+ 1` in the derivation: stream(seed, 0) is never mix64(seed).
    _require(_reference_stream(0, 0) == _mix64_int(GOLDEN),
             "the reference stream key drops the call_index + 1 term")
    for name, (seed, call_index, p, words, keep) in DROPOUT_VECTORS.items():
        produced = _reference_words(VECTOR_LENGTH, seed, call_index)
        _require(list(int(word) for word in produced[:4]) == list(words),
                 f"the reference element words disagree for {name}")
        mask = _reference_mask((VECTOR_LENGTH,), p, seed, call_index)
        _require(_keep_pattern(mask) == keep,
                 f"the reference keep pattern for {name} is "
                 f"{_keep_pattern(mask)}, expected {keep}")
        _require(set(np.unique(mask)) <= {0.0, 1.0 / (1.0 - p)},
                 f"the reference mask for {name} holds a third value")
    # The bits-to-uniform conversion, exactly.
    _require(_reference_uniform(np.array([0], dtype=np.uint64))[0] == 0.0,
             "the reference uniform of zero is not 0.0")
    _require(
        _reference_uniform(np.array([UINT64_MAX], dtype=np.uint64))[0]
        == (2 ** 53 - 1) / 2 ** 53,
        "the reference uniform of the all-ones word is wrong",
    )
    # The equality-threshold vector: strict `<`, equality keeps, one ULP
    # more drops — and the rejected `<=` rule genuinely disagrees.
    uniform = _reference_uniform(np.array([EQUALITY_WORD], dtype=np.uint64))[0]
    _require(uniform == EQUALITY_UNIFORM,
             "the committed equality-threshold uniform does not match")
    words = _reference_words(EQUALITY_COUNT, EQUALITY_SEED,
                             EQUALITY_CALL_INDEX)
    _require(int(words[EQUALITY_INDEX]) == EQUALITY_WORD,
             "the equality-threshold word is not where the vector says")
    uniforms = _reference_uniform(words)
    at_equal = "".join("0" if value < EQUALITY_UNIFORM else "1"
                       for value in uniforms)
    at_next = "".join(
        "0" if value < math.nextafter(EQUALITY_UNIFORM, 1.0) else "1"
        for value in uniforms
    )
    under_less_equal = "".join("0" if value <= EQUALITY_UNIFORM else "1"
                               for value in uniforms)
    _require(at_equal == EQUALITY_KEEP_AT_EQUAL,
             f"equality is not a keep: {at_equal}")
    _require(at_next == EQUALITY_KEEP_AT_NEXT,
             f"one ULP above the threshold is not a drop: {at_next}")
    _require(under_less_equal == EQUALITY_KEEP_UNDER_LESS_EQUAL
             and under_less_equal != at_equal,
             "the equality vector does not discriminate < from <=")
    return {
        "status": "passed",
        "algorithm": "tensorforge.splitmix64",
        "algorithm_version": 1,
        "mix64_vectors": len(MIX64_VECTORS),
        "stream_vectors": len(STREAM_VECTORS),
        "mask_vectors": len(DROPOUT_VECTORS),
        "vector_length": VECTOR_LENGTH,
        "checks": ["scalar_finalizer", "vectorized_finalizer",
                   "stream_keys", "element_words", "keep_patterns",
                   "two_valued_mask", "bits_to_uniform",
                   "strict_comparison_boundary",
                   "less_equal_negative_control"],
    }


def verify_core_against_committed_vectors():
    """Pin the **native kernel** to the same committed vectors, through
    the production Core methods, before any case is timed."""
    values = np.arange(VECTOR_LENGTH, dtype=np.float64) + 0.5
    checked = []
    for name, (seed, call_index, p, _words, keep) in DROPOUT_VECTORS.items():
        core = _core_from_values(values)
        try:
            out, mask = core._dropout_forward_with_mask(
                p, seed=seed, call_index=call_index
            )
            try:
                produced_mask = mask.to_numpy().copy()
                produced_out = out.to_numpy().copy()
            finally:
                out.close()
                mask.close()
            public = core.dropout_forward(p, seed=seed, call_index=call_index)
            try:
                produced_public = public.to_numpy().copy()
            finally:
                public.close()
        finally:
            core.close()
        _require(_keep_pattern(produced_mask) == keep,
                 f"the native mask for {name} is "
                 f"{_keep_pattern(produced_mask)}, expected {keep}")
        expected_out, expected_mask = _reference_dropout(
            values, p, seed, call_index
        )
        _require_exact(produced_mask, expected_mask,
                       f"the native mask for {name}")
        _require_exact(produced_out, expected_out,
                       f"the native output for {name}")
        _require_exact(produced_public, expected_out,
                       f"the public Core output for {name}")
        checked.append(name)
    return {
        "status": "passed",
        "vectors": checked,
        "surfaces": ["NativeTensorCore._dropout_forward_with_mask",
                     "NativeTensorCore.dropout_forward"],
        "checks": ["committed_keep_pattern", "mask_bit_exact",
                   "output_bit_exact", "public_core_equals_output_half"],
    }


# ---------------------------------------------------------------------------
# Case builders. Each returns untimed ``prepare``/``cleanup`` callables
# around one timed ``run``, a ``check`` that runs the whole correctness
# gate before any timing, and a ``close`` that releases the case's shared
# state once every repetition is done.
# ---------------------------------------------------------------------------


def _build_call_floor(config, spec):
    """The harness's own Python call and loop floor.

    Not a TensorForge operation: it is the identity-dispatch rows'
    denominator, measured through the *same* calibrated inner loop, so a
    reader can see how much of an identity timing is the harness."""
    del config
    values = _values((4,))
    tensor = NativeTensor.from_array(values)

    def passthrough(argument):
        return argument

    def native_run(_state=None):
        return passthrough(tensor)

    def check():
        result = native_run()
        _require(result is tensor,
                 "the floor callable does not return its argument")
        return {"checks": ["returns_the_argument_object"]}

    return {
        "native_prepare": lambda: None,
        "native_run": native_run,
        "native_cleanup": lambda _state, _result: None,
        "reference_prepare": None,
        "reference_run": None,
        "reference_cleanup": None,
        "check": check,
        "close": tensor.close,
        "generator": None,
    }


def _build_core_case(config, spec):
    """One native Core Dropout forward against the exact NumPy reference.

    ``with_mask`` selects the private ``_dropout_forward_with_mask``
    helper (output **and** multiplier mask retained, which is what the
    autograd path actually needs) or the public ``dropout_forward``
    (which allocates the mask internally and closes it before
    returning)."""
    shape = tuple(config["shape"])
    p = spec["probability"]
    seed, call_index = spec["seed"], spec["call_index"]
    with_mask = spec["with_mask"]
    values = _values(shape)
    core, owned, numpy_view = _layout_views(values, spec["layout"], config)

    def native_run(_state=None):
        if with_mask:
            return core._dropout_forward_with_mask(
                p, seed=seed, call_index=call_index
            )
        return core.dropout_forward(p, seed=seed, call_index=call_index)

    def native_cleanup(_state, result):
        if result is None:
            return
        if with_mask:
            result[0].close()
            result[1].close()
        else:
            result.close()

    def reference_run(_state=None):
        # Policy B materializes a non-contiguous input before the kernel
        # runs, so the reference materializes too — otherwise the two
        # sides would not be doing the same work.
        return _reference_dropout(_materialized(numpy_view),
                                  p, seed, call_index)

    def check():
        expected_out, expected_mask = reference_run()
        result = native_run()
        try:
            if with_mask:
                out, mask = result
                produced_mask = mask.to_numpy().copy()
                _require(mask.contiguous,
                         "the native mask is not contiguous")
            else:
                out, produced_mask = result, None
            produced = out.to_numpy().copy()
            _require(out.contiguous, "the native output is not contiguous")
            _require(out.shape == shape,
                     f"the native output has shape {out.shape}, expected "
                     f"{shape}")
        finally:
            native_cleanup(None, result)
        _require_finite(produced, "the native output")
        _require_exact(produced, expected_out, "the native Core output")
        checks = ["output_bit_exact_vs_numpy", "finite", "output_shape",
                  "contiguous_output", "input_not_mutated",
                  "core_is_stateless"]
        if produced_mask is not None:
            _require_exact(produced_mask, expected_mask, "the native mask")
            _require(set(np.unique(produced_mask))
                     <= {0.0, 1.0 / (1.0 - p)},
                     "the native mask holds a third value")
            _require_exact(produced, np.asarray(numpy_view) * produced_mask,
                           "the native output as input * mask")
            checks.extend(["mask_bit_exact_vs_numpy", "two_valued_mask",
                           "output_equals_input_times_mask"])
        else:
            # The public Core discards the mask; prove the output half is
            # exactly what the private helper's output was.
            private = core._dropout_forward_with_mask(
                p, seed=seed, call_index=call_index
            )
            try:
                _require_exact(produced, private[0].to_numpy(),
                               "the public Core output")
            finally:
                private[0].close()
                private[1].close()
            checks.append("public_output_equals_private_output_half")
        _require_unchanged(core.to_numpy(), numpy_view, "the native input")
        return {
            "max_abs_error": 0.0,
            "keep_fraction": float(np.mean(
                (expected_mask != 0.0).astype(np.float64)
            )),
            "checks": checks,
        }

    def close():
        for owned_core in owned:
            owned_core.close()

    return {
        "native_prepare": lambda: None,
        "native_run": native_run,
        "native_cleanup": native_cleanup,
        "reference_prepare": lambda: None,
        "reference_run": reference_run,
        "reference_cleanup": lambda _state, _result: None,
        "check": check,
        "close": close,
        "generator": None,
    }


def _build_tensor_forward(config, spec):
    """One ``NativeTensor.dropout`` forward, with or without gradients.

    The timed call includes the generator call transaction (reserve, run
    the Core, build the node, commit), the two native allocations, the
    Python wrapper, and — when the input requires grad — the autograd node
    and the mask adoption. That is what a caller actually pays."""
    shape = tuple(config["shape"])
    p = spec["probability"]
    requires_grad = spec["requires_grad"]
    values = _values(shape)
    generator = NativeGenerator(spec["seed"])
    tensor = NativeTensor.from_array(values, requires_grad=requires_grad)
    identity = (p == 0.0)

    def native_run(_state=None):
        return tensor.dropout(p, generator=generator)

    def native_cleanup(_state, result):
        # `p == 0` is identity: the result *is* the caller's tensor, and
        # closing it would destroy the case's own input.
        if result is not None and result is not tensor:
            result.close()

    def check():
        before = generator.calls
        seed = generator.seed
        result = native_run()
        try:
            if identity:
                _require(result is tensor,
                         "p == 0 did not return the input object")
                _require(generator.calls == before,
                         "p == 0 consumed a generator call")
                _require(_graph_resources(result) == (),
                         "p == 0 attached a graph resource")
                return {
                    "consumed_calls": 0,
                    "checks": ["identity_returns_input_object",
                               "no_generator_call", "no_graph_resource",
                               "no_allocation"],
                }
            _require(result is not tensor,
                     "a stochastic forward returned the input object")
            _require(generator.calls == before + 1,
                     f"one forward consumed {generator.calls - before} calls")
            _require(result.requires_grad is requires_grad,
                     "the result's requires_grad does not follow the input")
            produced = result.to_numpy().copy()
            expected_out, expected_mask = _reference_dropout(
                values, p, seed, before
            )
            _require_exact(produced, expected_out,
                           "the operation's output at the reserved index")
            resources = _graph_resources(result)
            if requires_grad:
                _require(len(resources) == 1,
                         f"the graph owns {len(resources)} resources, "
                         f"expected exactly the mask")
                _require_exact(resources[0].to_numpy(), expected_mask,
                               "the graph-owned multiplier mask")
                checks = ["stochastic_output_bit_exact", "exactly_one_call",
                          "requires_grad_follows_input",
                          "graph_owns_exactly_the_mask",
                          "mask_bit_exact"]
            else:
                _require(resources == (),
                         "a no-grad forward retained a graph resource")
                checks = ["stochastic_output_bit_exact", "exactly_one_call",
                          "requires_grad_follows_input",
                          "no_grad_closes_the_mask"]
        finally:
            native_cleanup(None, result)
        _require_unchanged(tensor.to_numpy(), values, "the native input")
        _require(not generator._has_active_reservation(),
                 "a reservation survived the forward")
        return {"consumed_calls": 1, "checks": checks}

    return {
        "native_prepare": lambda: None,
        "native_run": native_run,
        "native_cleanup": native_cleanup,
        "reference_prepare": None,
        "reference_run": None,
        "reference_cleanup": None,
        "check": check,
        "close": tensor.close,
        "generator": generator,
    }


def _build_tensor_backward(config, spec):
    """Only ``backward()`` is timed.

    A fresh forward graph is built **outside** the timer for every
    repetition, and the upstream gradient is seeded explicitly on the
    Dropout output, so the timed pass is exactly Dropout's backward — one
    native ``multiply`` against the saved mask plus the accumulation —
    and not a chain of unrelated nodes."""
    shape = tuple(config["shape"])
    p = spec["probability"]
    values = _values(shape)
    upstream = _upstream_values(shape)
    generator = NativeGenerator(spec["seed"])
    native_upstream = NativeTensor.from_array(upstream)

    def native_prepare():
        tensor = NativeTensor.from_array(values, requires_grad=True)
        return tensor, tensor.dropout(p, generator=generator)

    def native_run(state):
        state[1].backward(gradient=native_upstream)
        return None

    def native_cleanup(state, _result):
        tensor, output = state
        _release_gradients([tensor])
        output.close()
        tensor.close()

    def check():
        before = generator.calls
        seed = generator.seed
        state = native_prepare()
        tensor, output = state
        try:
            _require(generator.calls == before + 1,
                     "the untimed forward did not consume exactly one call")
            native_run(state)
            _require(tensor.grad is not None,
                     "the backward produced no input gradient")
            gradient = tensor.grad.to_numpy().copy()
            _expected_out, expected_mask = _reference_dropout(
                values, p, seed, before
            )
            _require_exact(gradient, upstream * expected_mask,
                           "the Dropout input gradient")
            _require(output._graph_freed,
                     "the one-shot backward did not release the graph")
            _require(_graph_resources(output) == (),
                     "the multiplier mask survived the one-shot backward")
            _require(generator.calls == before + 1,
                     "backward consumed a generator call")
        finally:
            native_cleanup(state, None)
        return {
            "consumed_calls": 1,
            "checks": ["gradient_equals_upstream_times_saved_mask",
                       "graph_released_once", "mask_released_with_the_graph",
                       "backward_consumes_no_call"],
        }

    return {
        "native_prepare": native_prepare,
        "native_run": native_run,
        "native_cleanup": native_cleanup,
        "reference_prepare": None,
        "reference_run": None,
        "reference_cleanup": None,
        "check": check,
        "close": native_upstream.close,
        "generator": generator,
        "calls_per_cycle": 1,
    }


def _build_tensor_forward_backward(config, spec):
    """The complete differentiable lifecycle in one timed region: the
    stochastic forward, the explicit upstream gradient, and the one-shot
    ``backward()`` that releases the graph and the mask.

    Input construction and gradient cleanup stay outside the timer, so no
    graph, mask, output, or gradient survives into the next repetition."""
    shape = tuple(config["shape"])
    p = spec["probability"]
    values = _values(shape)
    upstream = _upstream_values(shape)
    generator = NativeGenerator(spec["seed"])
    native_upstream = NativeTensor.from_array(upstream)

    def native_prepare():
        return NativeTensor.from_array(values, requires_grad=True)

    def native_run(tensor):
        output = tensor.dropout(p, generator=generator)
        output.backward(gradient=native_upstream)
        return output

    def native_cleanup(tensor, result):
        _release_gradients([tensor])
        if result is not None:
            result.close()
        tensor.close()

    def check():
        before = generator.calls
        seed = generator.seed
        tensor = native_prepare()
        result = None
        try:
            result = native_run(tensor)
            _require(generator.calls == before + 1,
                     "one forward/backward consumed more than one call")
            gradient = tensor.grad.to_numpy().copy()
            expected_out, expected_mask = _reference_dropout(
                values, p, seed, before
            )
            _require_exact(result.to_numpy(), expected_out,
                           "the forward output")
            _require_exact(gradient, upstream * expected_mask,
                           "the input gradient")
            _require(result._graph_freed and _graph_resources(result) == (),
                     "the graph or its mask survived the lifecycle")
        finally:
            native_cleanup(tensor, result)
        return {
            "consumed_calls": 1,
            "checks": ["output_bit_exact", "gradient_bit_exact",
                       "exactly_one_call", "graph_and_mask_released"],
        }

    return {
        "native_prepare": native_prepare,
        "native_run": native_run,
        "native_cleanup": native_cleanup,
        "reference_prepare": None,
        "reference_run": None,
        "reference_cleanup": None,
        "check": check,
        "close": native_upstream.close,
        "generator": generator,
        "calls_per_cycle": 1,
    }


def _build_module_case(config, spec):
    """One ``NativeDropout`` forward in training or evaluation mode.

    Evaluation and ``p == 0`` return the **caller's own tensor**: no
    reservation, no allocation, no kernel, and no graph node. Those rows
    measure dispatch, not mask generation, and the gate proves the
    identity before anything is timed."""
    shape = tuple(config["shape"])
    p = spec["probability"]
    mode = spec["mode"]
    values = _values(shape)
    module = NativeDropout(p, seed=spec["seed"])
    if mode == "eval":
        module.eval()
    tensor = NativeTensor.from_array(values,
                                     requires_grad=spec["requires_grad"])
    identity = (mode == "eval") or (p == 0.0)
    generator = module.generator

    def native_run(_state=None):
        return module(tensor)

    def native_cleanup(_state, result):
        if result is not None and result is not tensor:
            result.close()

    def check():
        _require(module.training is (mode == "train"),
                 f"the module is not in {mode} mode")
        _require(module.p == p, "the module's probability was not stored")
        _require(list(name for name, _ in module.named_generators())
                 == ["generator"],
                 "the module does not register exactly one generator")
        before = generator.calls
        seed = generator.seed
        result = native_run()
        try:
            if identity:
                _require(result is tensor,
                         f"{mode} mode at p={p} did not return the input "
                         f"object")
                _require(generator.calls == before,
                         f"{mode} mode at p={p} consumed a generator call")
                _require(_graph_resources(result) == (),
                         "an identity forward attached a graph resource")
                # Repeat it: any number of identity forwards must leave no
                # gap in the stream.
                for _ in range(3):
                    _require(native_run() is tensor,
                             "a repeated identity forward allocated")
                _require(generator.calls == before,
                         "repeated identity forwards moved the counter")
                return {
                    "consumed_calls": 0,
                    "checks": ["identity_returns_input_object",
                               "no_generator_call", "no_graph_resource",
                               "repeatable_without_stream_gap"],
                }
            _require(generator.calls == before + 1,
                     "one training forward did not consume exactly one call")
            produced = result.to_numpy().copy()
            expected_out, _mask = _reference_dropout(values, p, seed, before)
            _require_exact(produced, expected_out,
                           "the module's training output")
        finally:
            native_cleanup(None, result)
        # ...and it is exactly what NativeTensor.dropout produces from an
        # equal generator state, which is the delegation the module claims.
        twin = NativeGenerator(seed)
        twin.load_state({"algorithm": generator.algorithm,
                         "algorithm_version": generator.algorithm_version,
                         "seed": seed, "calls": before})
        direct = tensor.dropout(p, generator=twin)
        try:
            _require_exact(produced, direct.to_numpy(),
                           "the module output against NativeTensor.dropout")
        finally:
            direct.close()
        _require(twin.calls == before + 1,
                 "the direct operation did not consume exactly one call")
        _require_unchanged(tensor.to_numpy(), values, "the module input")
        _require(not generator._has_active_reservation(),
                 "a reservation survived the module forward")
        return {
            "consumed_calls": 1,
            "checks": ["training_output_bit_exact", "exactly_one_call",
                       "equals_nativetensor_dropout_from_equal_state",
                       "one_registered_generator", "input_not_mutated"],
        }

    def close():
        tensor.close()
        _close_module(module)

    return {
        "native_prepare": lambda: None,
        "native_run": native_run,
        "native_cleanup": native_cleanup,
        "reference_prepare": None,
        "reference_run": None,
        "reference_cleanup": None,
        "check": check,
        "close": close,
        "generator": generator,
    }


class _BenchmarkDropoutClassifier(NativeModule):
    """The training-step model: ``Linear -> ReLU -> Dropout -> Linear``
    over raw logits.

    The G7 architecture with the two normalization layers removed — this
    benchmark characterizes Dropout, and BatchNorm/LayerNorm already have
    their own F7 harness. One Dropout module and one registered
    generator."""

    def __init__(self):
        super().__init__()
        self.hidden = NativeLinear(STEP_FEATURES, STEP_HIDDEN,
                                   seed=STEP_HIDDEN_SEED)
        self.relu = NativeReLU()
        self.dropout = NativeDropout(DEFAULT_P, seed=STEP_DROPOUT_SEED)
        self.output = NativeLinear(STEP_HIDDEN, STEP_CLASSES,
                                   seed=STEP_OUTPUT_SEED)

    def forward(self, inputs):
        hidden = self.relu(self.hidden(inputs))
        return self.output(self.dropout(hidden))


def _step_dataset():
    """The G7 fixed twelve-sample three-class dataset, from the same
    explicit formula: every value is a quarter or an eighth, so it is
    exact in float64, and nothing is sampled, shuffled, or loaded."""
    inputs, targets = [], []
    for index in range(STEP_SAMPLES):
        label = index % STEP_CLASSES
        offset = (index // STEP_CLASSES) / 4.0
        row = [offset - 0.375] * STEP_FEATURES
        row[label] = 1.0 + offset
        row[(label + 1) % STEP_FEATURES] = -0.75 + offset
        inputs.append(row)
        targets.append(label)
    return inputs, targets


def _build_training_step(config, spec):
    """One complete Dropout training step: training-mode forward through
    the model, cross-entropy over raw logits, ``backward()``,
    ``NativeAdam.step()``, ``zero_grad()``.

    A **fresh** model and optimizer are built outside the timer for every
    repetition, because a step advances parameters, optimizer moments, and
    the generator; the dataset tensors and the loss module are shared and
    untimed."""
    del config, spec
    inputs, targets = _step_dataset()
    native_inputs = NativeTensor.from_array(inputs)
    criterion = NativeCrossEntropyLoss()

    def native_prepare():
        model = _BenchmarkDropoutClassifier()
        model.train()
        return model, NativeAdam(model.parameters(), lr=STEP_LR)

    def native_run(state):
        model, optimizer = state
        logits = model(native_inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return logits, loss

    def native_cleanup(state, result):
        model, optimizer = state
        if result is not None:
            logits, loss = result
            loss.close()
            logits.close()
        _release_gradients(list(model.parameters()))
        optimizer.close()
        _close_module(model)

    def check():
        state = native_prepare()
        model, _optimizer = state
        generator = model.dropout.generator
        before = generator.calls
        parameters_before = {name: parameter.to_numpy().copy()
                             for name, parameter in model.named_parameters()}
        result = None
        try:
            result = native_run(state)
            logits, loss = result
            value = float(loss.to_numpy())
            _require(math.isfinite(value), "the training loss is not finite")
            _require(logits.shape == (STEP_SAMPLES, STEP_CLASSES),
                     f"the logits have shape {logits.shape}")
            _require(generator.calls == before + 1,
                     f"one training step consumed "
                     f"{generator.calls - before} generator calls")
            _require(loss._graph_freed,
                     "the step did not release its graph")
            changed = [
                name for name, parameter in model.named_parameters()
                if not np.array_equal(parameter.to_numpy(),
                                      parameters_before[name])
            ]
            _require(len(changed) == len(parameters_before),
                     f"only {len(changed)} of {len(parameters_before)} "
                     f"parameters moved")
            _require(all(parameter.grad is None
                         for parameter in model.parameters()),
                     "zero_grad did not clear the gradients")
            _require(not generator._has_active_reservation(),
                     "a reservation survived the training step")
        finally:
            native_cleanup(state, result)
        return {
            "consumed_calls": 1,
            "loss": value,
            "updated_parameters": sorted(changed),
            "checks": ["finite_loss", "logit_shape", "exactly_one_call",
                       "graph_released", "every_parameter_updated",
                       "gradients_cleared", "no_reservation_left"],
        }

    return {
        "native_prepare": native_prepare,
        "native_run": native_run,
        "native_cleanup": native_cleanup,
        "reference_prepare": None,
        "reference_run": None,
        "reference_cleanup": None,
        "check": check,
        "close": native_inputs.close,
        "generator": None,
    }


# ---------------------------------------------------------------------------
# The case registry.
#
# Grouped by family, in report order. Every case declares the shape it
# runs, its probability, its layout, its reference label, and what its
# gate proves; nothing about a case is inferred from its name.
# ---------------------------------------------------------------------------

_MEDIUM = {"full": {"shape": (64, 256)}, "smoke": {"shape": (8, 16)}}
_LAYOUT_CONFIG = {
    "full": {"shape": (64, 256), "column_offset": 16, "row_offset": 4},
    "smoke": {"shape": (8, 16), "column_offset": 3, "row_offset": 2},
}

_IDENTITY_POLICY = "calibrated_identity_loop"
_SINGLE_POLICY = "one_call_per_sample"

_NUMPY_REFERENCE_DETAIL = (
    "an exact vectorized NumPy implementation of the same locked "
    "tensorforge.splitmix64 derivation, doing the same work: the stream "
    "key, one 64-bit word per logical element, the top-53-bit uniform, "
    "the strict u < p test, the 1/(1 - p) multiplier mask, and the output "
    "multiply, with both arrays allocated"
)
_NATIVE_ONLY_DETAIL = (
    "none timed: this layer owns a generator call transaction, native "
    "ownership, and (where applicable) an autograd graph, none of which a "
    "NumPy expression has. Timing one against the other would compare "
    "different work, so no ratio is published; the correctness gate still "
    "compares bit for bit against the same reference at the reserved call "
    "index"
)


def _core_spec(name, family, shape_config, probability, notes,
               with_mask=True, layout="contiguous", seed=BENCHMARK_SEED,
               call_index=BENCHMARK_CALL_INDEX, relative_to=None):
    return (name, {
        "family": family,
        "layer": "core",
        "operation": ("NativeTensorCore._dropout_forward_with_mask(p, "
                      "seed=..., call_index=...)" if with_mask
                      else "NativeTensorCore.dropout_forward(p, seed=..., "
                           "call_index=...)"),
        "mode": None,
        "probability": probability,
        "layout": layout,
        "requires_grad": False,
        "includes_backward": False,
        "with_mask": with_mask,
        "seed": seed,
        "call_index": call_index,
        "reference_type": NUMPY,
        "reference_detail": _NUMPY_REFERENCE_DETAIL,
        "correctness_reference": ("the committed G2 known-answer vectors "
                                  "and the exact vectorized NumPy "
                                  "reference"),
        "configurations": shape_config,
        "build": _build_core_case,
        "iteration_policy": _SINGLE_POLICY,
        "relative_to": relative_to,
        "notes": notes,
    })


def _tensor_spec(name, family, probability, requires_grad, build, notes,
                 includes_backward=False, seed=BENCHMARK_SEED,
                 policy=_SINGLE_POLICY, relative_to=None,
                 repetitions=None, operation=None):
    spec = {
        "family": family,
        "layer": "tensor_operation",
        "operation": operation or "NativeTensor.dropout(p, generator=...)",
        "mode": None,
        "probability": probability,
        "layout": "contiguous",
        "requires_grad": requires_grad,
        "includes_backward": includes_backward,
        "with_mask": None,
        "seed": seed,
        "call_index": None,
        "reference_type": NATIVE_ONLY,
        "reference_detail": _NATIVE_ONLY_DETAIL,
        "correctness_reference": ("the exact vectorized NumPy reference at "
                                  "the reservation's (seed, call_index)"),
        "configurations": _MEDIUM,
        "build": build,
        "iteration_policy": policy,
        "relative_to": relative_to,
        "notes": notes,
    }
    if repetitions is not None:
        spec["repetitions"] = repetitions
    return (name, spec)


def _module_spec(name, family, probability, mode, notes, seed=BENCHMARK_SEED,
                 policy=_SINGLE_POLICY, relative_to=None,
                 requires_grad=True):
    return (name, {
        "family": family,
        "layer": "module",
        "operation": f"NativeDropout(p)(input) in {mode} mode",
        "mode": mode,
        "probability": probability,
        "layout": "contiguous",
        "requires_grad": requires_grad,
        "includes_backward": False,
        "with_mask": None,
        "seed": seed,
        "call_index": None,
        "reference_type": NATIVE_ONLY,
        "reference_detail": _NATIVE_ONLY_DETAIL,
        "correctness_reference": ("the exact vectorized NumPy reference at "
                                  "the reservation's (seed, call_index), "
                                  "plus NativeTensor.dropout from an equal "
                                  "generator state"),
        "configurations": _MEDIUM,
        "build": _build_module_case,
        "iteration_policy": policy,
        "relative_to": relative_to,
        "notes": notes,
    })


CASES = dict([
    # -- the harness's own floor ------------------------------------------
    ("python_call_floor", {
        "family": "baseline",
        "layer": "harness",
        "operation": "a Python function returning its argument",
        "mode": None,
        "probability": None,
        "layout": None,
        "requires_grad": False,
        "includes_backward": False,
        "with_mask": None,
        "seed": None,
        "call_index": None,
        "reference_type": HARNESS,
        "reference_detail": ("none: this *is* the reference for the "
                             "identity-dispatch rows"),
        "correctness_reference": "the callable returns its argument object",
        "configurations": {"full": {"shape": (4,)}, "smoke": {"shape": (4,)}},
        "build": _build_call_floor,
        "iteration_policy": _IDENTITY_POLICY,
        "relative_to": None,
        "notes": ("The calibrated inner loop measured around a trivial "
                  "Python call, so the identity-dispatch rows can be read "
                  "against a floor instead of against zero. Not a "
                  "TensorForge operation."),
    }),

    # -- Core versus the exact NumPy reference ----------------------------
    _core_spec(
        "core_dropout_forward", "core_reference", _MEDIUM, DEFAULT_P,
        ("The public output-only Core method. It allocates the multiplier "
         "mask internally and closes it before returning, so it does the "
         "same work as the with-mask helper and keeps less; the reference "
         "does the same work too."),
        with_mask=False, relative_to="core_dropout_forward_with_mask",
    ),
    _core_spec(
        "core_dropout_forward_with_mask", "core_reference", _MEDIUM,
        DEFAULT_P,
        ("The private output-plus-mask helper — the Core contract the "
         "autograd path actually uses, and the preferred comparison "
         "against NumPy because both sides then allocate and keep two "
         "arrays."),
    ),

    # -- size scaling -----------------------------------------------------
    _core_spec(
        "scaling_core_scalar", "size_scaling",
        {"full": {"shape": ()}, "smoke": {"shape": ()}}, DEFAULT_P,
        ("A rank-0 tensor: one logical element, one draw. Almost entirely "
         "Python, validation, allocation, and ctypes overhead — the fixed "
         "cost per call, with the per-element work as small as it goes."),
    ),
    _core_spec(
        "scaling_core_tiny_vector", "size_scaling",
        {"full": {"shape": (8,)}, "smoke": {"shape": (8,)}}, DEFAULT_P,
        "Eight elements: still dominated by fixed per-call cost.",
    ),
    _core_spec(
        "scaling_core_small", "size_scaling",
        {"full": {"shape": (256,)}, "smoke": {"shape": (64,)}}, DEFAULT_P,
        "256 elements: fixed cost and per-element cost are comparable.",
    ),
    _core_spec(
        "scaling_core_medium", "size_scaling", _MEDIUM, DEFAULT_P,
        ("16,384 elements in a 2-D shape. The same configuration as "
         "core_dropout_forward_with_mask, measured independently — the two "
         "rows show this run's own repeatability."),
    ),
    _core_spec(
        "scaling_core_large", "size_scaling",
        {"full": {"shape": (8, 16, 32, 32)},
         "smoke": {"shape": (2, 2, 4, 4)}}, DEFAULT_P,
        ("131,072 elements in a 4-D NCHW-shaped tensor: per-element work "
         "dominates, and the fixed cost is amortized."),
    ),

    # -- layout characterization ------------------------------------------
    _core_spec(
        "layout_contiguous", "layout", _LAYOUT_CONFIG, DEFAULT_P,
        ("The contiguous baseline for the layout ratios. All four layout "
         "cases carry the same logical shape and the same logical values, "
         "and their masks are proved identical before anything is timed."),
        layout="contiguous",
    ),
    _core_spec(
        "layout_transposed", "layout", _LAYOUT_CONFIG, DEFAULT_P,
        ("A transposed view: non-contiguous, storage offset 0. Policy B "
         "materializes a contiguous copy inside the Core, so the extra "
         "work is one strided copy — the reference materializes too."),
        layout="transposed", relative_to="layout_contiguous",
    ),
    _core_spec(
        "layout_narrowed_noncontiguous", "layout", _LAYOUT_CONFIG, DEFAULT_P,
        ("A column-narrowed view: non-contiguous **and** at a nonzero "
         "storage offset. Row-major order is preserved, which a transposed "
         "view's is not, so this separates 'non-contiguous' from "
         "'reordered'."),
        layout="narrowed_noncontiguous", relative_to="layout_contiguous",
    ),
    _core_spec(
        "layout_offset_contiguous", "layout", _LAYOUT_CONFIG, DEFAULT_P,
        ("A row-narrowed view: still contiguous, but at a nonzero storage "
         "offset, so Policy B does **not** copy. The row that shows a "
         "nonzero offset alone costs nothing structural."),
        layout="offset_contiguous", relative_to="layout_contiguous",
    ),

    # -- probability characterization -------------------------------------
    *[_core_spec(
        f"probability_core_{label}", "probability", _MEDIUM, probability,
        ("The stateless Core draws and compares one word per element "
         "whatever p is — including at p == 0, where the Core has no "
         "identity short-circuit and still allocates, draws, and writes "
         "(the operation and module layers do short-circuit; their p == 0 "
         "rows are the identity cases in the tensor_operation and module "
         "families)."),
        relative_to="probability_core_p050",
    ) for label, probability in PROBABILITY_SWEEP],
    *[_tensor_spec(
        f"probability_tensor_{label}", "probability", probability, False,
        _build_tensor_forward,
        ("The no-grad operation across the probability sweep: one "
         "reservation, one Core call, one commit, mask closed immediately. "
         "p == 0 is excluded here because it is a different code path — "
         "identity — measured as tensor_dropout_p0_identity."),
        relative_to="probability_tensor_p050",
    ) for label, probability in PROBABILITY_SWEEP if probability != 0.0],
    *[_module_spec(
        f"probability_module_{label}", "probability", probability, "train",
        ("The module in training mode across the sweep. p == 0 is "
         "excluded here for the same reason: at the module layer it is the "
         "operation's identity rule, measured as "
         "module_training_p0_identity."),
        relative_to="probability_module_p050",
    ) for label, probability in PROBABILITY_SWEEP if probability != 0.0],

    # -- the NativeTensor operation layers --------------------------------
    _tensor_spec(
        "tensor_dropout_nograd_forward", "tensor_operation", DEFAULT_P,
        False, _build_tensor_forward,
        ("requires_grad=False: the full call transaction and both "
         "allocations, but no autograd node — `_from_op` closes the mask "
         "immediately, inside the timed call. The call is still committed, "
         "because a draw happened."),
    ),
    _tensor_spec(
        "tensor_dropout_forward", "tensor_operation", DEFAULT_P, True,
        _build_tensor_forward,
        ("requires_grad=True: the same work plus the autograd node and the "
         "mask adopted as graph-owned state. Backward is deliberately not "
         "inside this timing."),
        relative_to="tensor_dropout_nograd_forward",
    ),
    _tensor_spec(
        "tensor_dropout_backward", "tensor_operation", DEFAULT_P, True,
        _build_tensor_backward,
        ("backward() only. The graph is built outside the timer for every "
         "repetition and the upstream gradient is seeded explicitly on the "
         "Dropout output, so the timed pass is one native multiply against "
         "the saved mask plus the accumulation — no unrelated nodes."),
        includes_backward=True, relative_to="tensor_dropout_forward",
        repetitions=BACKWARD_REPETITIONS,
        operation="NativeTensor.dropout(...) -> backward(gradient=...)",
    ),
    _tensor_spec(
        "tensor_dropout_forward_backward", "tensor_operation", DEFAULT_P,
        True, _build_tensor_forward_backward,
        ("Forward and backward in one timed region — the whole "
         "differentiable lifecycle a training step pays, including the "
         "graph release. Input construction and gradient cleanup stay "
         "outside."),
        includes_backward=True, relative_to="tensor_dropout_forward",
        repetitions=BACKWARD_REPETITIONS,
        operation=("NativeTensor.dropout(...) then backward(gradient=...) "
                   "in one timed region"),
    ),
    _tensor_spec(
        "tensor_dropout_p0_identity", "tensor_operation", 0.0, True,
        _build_tensor_forward,
        ("p == 0 at the operation layer: the caller's own tensor is "
         "returned after validation, with no reservation, allocation, "
         "kernel call, graph node, or consumed call. This is dispatch "
         "cost, not Dropout throughput — read it against "
         "python_call_floor, never against an allocating row."),
        policy=_IDENTITY_POLICY,
        relative_to="tensor_dropout_nograd_forward",
    ),

    # -- the NativeDropout module -----------------------------------------
    _module_spec(
        "module_training_forward", "module", DEFAULT_P, "train",
        ("The headline module case: validation, the mode dispatch, and the "
         "delegation to NativeTensor.dropout with the registered "
         "generator. Same configuration as probability_module_p050, "
         "measured independently."),
    ),
    _module_spec(
        "module_eval_forward", "module", DEFAULT_P, "eval",
        ("Evaluation mode returns the input object itself: no reservation, "
         "no allocation, no kernel, no graph node, and no call consumed, "
         "so any number of eval forwards leaves no gap in the stream. "
         "Dispatch cost only."),
        policy=_IDENTITY_POLICY, relative_to="module_training_forward",
    ),
    _module_spec(
        "module_training_p0_identity", "module", 0.0, "train",
        ("p == 0 in training mode. The module deliberately does not "
         "short-circuit this itself — the operation's identity rule does "
         "(design §6.2) — so this measures the full input validation, the "
         "mode dispatch, and the operation's probability validation before "
         "the identity return."),
        policy=_IDENTITY_POLICY, relative_to="module_training_forward",
    ),
    _module_spec(
        "module_eval_p0_identity", "module", 0.0, "eval",
        ("p == 0 in evaluation mode: the eval branch returns first, so the "
         "probability is never consulted. The cheapest path in the whole "
         "surface."),
        policy=_IDENTITY_POLICY, relative_to="module_training_forward",
    ),

    # -- one complete training step ---------------------------------------
    ("dropout_training_step", {
        "family": "training_step",
        "layer": "training_step",
        "operation": ("NativeLinear -> NativeReLU -> NativeDropout -> "
                      "NativeLinear -> NativeCrossEntropyLoss -> backward "
                      "-> NativeAdam.step() -> zero_grad"),
        "mode": "train",
        "probability": DEFAULT_P,
        "layout": "contiguous",
        "requires_grad": True,
        "includes_backward": True,
        "with_mask": None,
        "seed": STEP_DROPOUT_SEED,
        "call_index": None,
        "reference_type": NATIVE_ONLY,
        "reference_detail": _NATIVE_ONLY_DETAIL,
        "correctness_reference": ("a finite loss, every parameter moved, "
                                  "exactly one generator call, and the "
                                  "graph released"),
        "configurations": {
            "full": {"shape": (STEP_SAMPLES, STEP_FEATURES)},
            "smoke": {"shape": (STEP_SAMPLES, STEP_FEATURES)},
        },
        "build": _build_training_step,
        "iteration_policy": _SINGLE_POLICY,
        "relative_to": None,
        "repetitions": TRAINING_STEP_REPETITIONS,
        "notes": ("Dropout in context: one real iteration on the fixed "
                  "twelve-sample three-class dataset. A fresh model and "
                  "optimizer are built outside the timer for every "
                  "repetition because a step advances parameters, "
                  "optimizer moments, and the generator. Characterizes one "
                  "iteration, not a scaling study."),
    }),
])

for _name, _spec in CASES.items():
    # A sweep declares one baseline row for the whole layer, and that row
    # is one of the sweep's own members: it is its own baseline, so it has
    # no sibling ratio rather than a constant 1.0 that means nothing.
    if _spec["relative_to"] == _name:
        _spec["relative_to"] = None
del _name, _spec

FAMILIES = tuple(dict.fromkeys(spec["family"] for spec in CASES.values()))

# The layered differences reported in §11's sense: descriptive gaps
# between adjacent measured layers, never a causal decomposition.
LAYER_DIFFERENCES = (
    {
        "name": "operation_over_core",
        "minuend": "tensor_dropout_nograd_forward",
        "subtrahend": "core_dropout_forward_with_mask",
        "description": ("NativeTensor no-grad forward minus the stateless "
                        "Core output-plus-mask forward: roughly the "
                        "generator reservation and commit, the wrapper's "
                        "validation and ownership, and the immediate mask "
                        "close the no-grad path does inside the call "
                        "(which the Core row does in untimed cleanup)"),
    },
    {
        "name": "graph_construction",
        "minuend": "tensor_dropout_forward",
        "subtrahend": "tensor_dropout_nograd_forward",
        "description": ("differentiable forward minus no-grad forward: "
                        "roughly the autograd node and the mask adoption, "
                        "against the no-grad path's immediate mask close"),
    },
    {
        "name": "module_dispatch",
        "minuend": "module_training_forward",
        "subtrahend": "tensor_dropout_forward",
        "description": ("NativeDropout training forward minus the "
                        "equivalent operation call: roughly the module's "
                        "input validation and mode dispatch"),
    },
    {
        "name": "backward_over_forward",
        "minuend": "tensor_dropout_forward_backward",
        "subtrahend": "tensor_dropout_forward",
        "description": ("forward-plus-backward minus forward: roughly the "
                        "backward pass and the graph release"),
    },
)

LAYER_DIFFERENCE_CAVEAT = (
    "Approximate layered differences, not a causal decomposition: each "
    "side is an independent measurement with its own noise, and the two "
    "sides do slightly different ownership work. A negative value means "
    "the outer layer measured faster than the inner one in this run, "
    "which is a measurement artifact and is reported as measured."
)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def calibrate(prepare, run, cleanup, target_ns, maximum):
    """Choose an inner-iteration count for an identity-dispatch case.

    Doubles until one sample would take at least ``target_ns`` or the cap
    is reached. Returns ``(iterations, cycles)`` — the second value is how
    many ``prepare``/``run``/``cleanup`` cycles calibration itself
    performed, so a case's generator accounting stays exact."""
    iterations, cycles = 1, 0
    while True:
        state = prepare()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            result = run(state)
        elapsed = time.perf_counter_ns() - start
        cleanup(state, result)
        cycles += iterations
        if elapsed >= target_ns or iterations >= maximum:
            return iterations, cycles
        iterations = min(maximum, iterations * 2)


def measure(prepare, run, cleanup, warmup, repetitions, iterations=1):
    """Return ``repetitions`` per-call seconds samples for ``run``.

    Each repetition builds its own state with ``prepare()`` (untimed),
    times ``iterations`` calls of ``run(state)`` with
    ``time.perf_counter_ns()``, then releases everything with
    ``cleanup(state, result)`` (untimed). Warm-up repetitions run the same
    way and are discarded before measuring; no measured sample is ever
    dropped, no timer overhead is subtracted, and ``gc`` is never touched
    inside the timed region. CPU execution is synchronous, so no explicit
    synchronization is needed.

    ``iterations > 1`` is only used by the identity-dispatch cases, whose
    call allocates nothing and returns the caller's own object; the
    reported value is always seconds **per call**."""
    for _ in range(warmup):
        state = prepare()
        result = run(state)
        cleanup(state, result)
    samples = []
    for _ in range(repetitions):
        state = prepare()
        if iterations == 1:
            start = time.perf_counter_ns()
            result = run(state)
            elapsed = time.perf_counter_ns() - start
            cleanup(state, result)
        else:
            start = time.perf_counter_ns()
            for _ in range(iterations):
                result = run(state)
            elapsed = time.perf_counter_ns() - start
            cleanup(state, result)
        samples.append(elapsed / iterations / 1e9)
    return samples


def _statistics(samples):
    """Median (the primary statistic) plus dispersion. Every value is a
    finite JSON number or ``None`` — never NaN or Infinity."""
    ordered = sorted(samples)
    median = statistics.median(ordered)
    low, high = ordered[0], ordered[-1]
    if len(ordered) >= 2:
        quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
        p25, p75 = quartiles[0], quartiles[2]
    else:
        p25 = p75 = median
    deviations = [abs(sample - median) for sample in ordered]
    return {
        "sample_count": len(ordered),
        "median_s": median,
        "median_ns": median * 1e9,
        "min_s": low,
        "max_s": high,
        "spread_s": high - low,
        "p25_s": p25,
        "p75_s": p75,
        "iqr_s": p75 - p25,
        "median_absolute_deviation_s": statistics.median(deviations),
        "relative_spread": ((high - low) / median) if median > 0 else None,
        "samples_s": list(ordered),
        "units": "seconds_per_call",
    }


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _finite_ratio(numerator, denominator):
    if denominator is None or numerator is None or denominator <= 0:
        return None
    return numerator / denominator


def _measure_case(name, warmup, repetitions, smoke):
    """Build the case, run its correctness gate, and only then time it.

    The ordering is the whole point: ``check()`` raises before ``measure``
    is ever reached, so a failed gate publishes no timing."""
    spec = CASES[name]
    config = spec["configurations"]["smoke" if smoke else "full"]
    case_repetitions = min(repetitions, spec.get("repetitions", repetitions))
    calibration = SMOKE_CALIBRATION if smoke else CALIBRATION
    case = spec["build"](config, spec)
    generator = case.get("generator")
    calls_per_cycle = case.get("calls_per_cycle")
    if calls_per_cycle is None:
        calls_per_cycle = 0 if generator is None else 1
        if spec["iteration_policy"] == _IDENTITY_POLICY:
            calls_per_cycle = 0
    try:
        # -- correctness first; a failure raises and publishes no timing --
        metrics = case["check"]()
        calls_before = None if generator is None else generator.calls
        iterations, calibration_cycles = 1, 0
        if spec["iteration_policy"] == _IDENTITY_POLICY:
            if calls_per_cycle:
                raise AssertionError(
                    f"{name}: a calibrated inner loop is only for cases "
                    f"that consume no generator call"
                )
            iterations, calibration_cycles = calibrate(
                case["native_prepare"], case["native_run"],
                case["native_cleanup"], calibration["target_ns"],
                calibration["maximum"],
            )
        native = _statistics(measure(
            case["native_prepare"], case["native_run"], case["native_cleanup"],
            warmup, case_repetitions, iterations,
        ))
        reference = None
        if spec["reference_type"] == NUMPY:
            reference = _statistics(measure(
                case["reference_prepare"], case["reference_run"],
                case["reference_cleanup"], warmup, case_repetitions,
            ))
        generator_record = None
        if generator is not None:
            # Exact accounting: every cycle the harness ran, counted
            # rather than sampled, and verified against the live counter.
            cycles = warmup + calibration_cycles + case_repetitions
            expected = calls_before + cycles * calls_per_cycle
            _require_quiet_generator(generator, expected, name)
            generator_record = {
                "seed": generator.seed,
                "calls_before_timing": calls_before,
                "calls_after_timing": generator.calls,
                "consumed": generator.calls - calls_before,
                "calls_per_cycle": calls_per_cycle,
                "cycles": cycles,
                "verified_exactly": True,
            }
    finally:
        case["close"]()

    element_count = _element_count(tuple(config["shape"]))
    identity = spec["iteration_policy"] == _IDENTITY_POLICY
    ns_per_element = (None if identity or not element_count
                      else native["median_ns"] / element_count)
    return {
        "case": name,
        "family": spec["family"],
        "layer": spec["layer"],
        "operation": spec["operation"],
        "mode": spec["mode"],
        "configuration": {key: _jsonable(value)
                          for key, value in config.items()},
        "shape": list(config["shape"]),
        "element_count": element_count,
        "layout": spec["layout"],
        "probability": spec["probability"],
        "requires_grad": spec["requires_grad"],
        "includes_backward": spec["includes_backward"],
        "seed": spec["seed"],
        "call_index": spec["call_index"],
        "reference_type": spec["reference_type"],
        "reference_detail": spec["reference_detail"],
        "correctness_reference": spec["correctness_reference"],
        "correctness": dict(status="passed", **metrics),
        "warmup": warmup,
        "sample_count": case_repetitions,
        "iterations_per_sample": iterations,
        "iteration_policy": spec["iteration_policy"],
        "calibration_cycles": calibration_cycles,
        "native": native,
        "reference": reference,
        "native_to_reference_ratio": _finite_ratio(
            native["median_s"], reference["median_s"] if reference else None
        ),
        "relative_to": spec["relative_to"],
        "native_relative_ratio": None,       # filled in once all cases ran
        "ns_per_element": ns_per_element,
        "operations_per_second": (1.0 / native["median_s"]
                                  if native["median_s"] > 0 else None),
        "generator": generator_record,
        "notes": spec["notes"],
    }


def _apply_relative_ratios(records):
    """Fill in each case's ratio against its declared sibling, when both
    ran in this invocation."""
    medians = {record["case"]: record["native"]["median_s"]
               for record in records}
    for record in records:
        target = record["relative_to"]
        if target in medians:
            record["native_relative_ratio"] = _finite_ratio(
                record["native"]["median_s"], medians[target]
            )
    return records


def _layer_differences(records):
    medians = {record["case"]: record["native"]["median_s"]
               for record in records}
    differences = []
    for entry in LAYER_DIFFERENCES:
        minuend = medians.get(entry["minuend"])
        subtrahend = medians.get(entry["subtrahend"])
        if minuend is None or subtrahend is None:
            continue
        differences.append({
            "name": entry["name"],
            "minuend": entry["minuend"],
            "subtrahend": entry["subtrahend"],
            "difference_s": minuend - subtrahend,
            "ratio": _finite_ratio(minuend, subtrahend),
            "description": entry["description"],
            "caveat": LAYER_DIFFERENCE_CAVEAT,
        })
    return differences


# ---------------------------------------------------------------------------
# Lifecycle verification — untimed, and run after every case.
# ---------------------------------------------------------------------------


def verify_lifecycle(cycles):
    """Repeated create/use/release cycles over every benchmark family,
    proving native live storage returns **exactly** to its baseline.

    Untimed by construction, and never inside a measured region. The
    live-storage instrumentation is benchmark-local (G8 adds no runtime
    API) and is removed again on the way out. Counts are read through
    ``_settled``, the G6/G7 convention: every release here is an explicit
    ``close()``, and the collection only settles the graph's
    backward-closure cycles into a deterministic number."""
    shape = (16, 8)
    values = _values(shape)
    upstream = _upstream_values(shape)
    inputs, targets = _step_dataset()
    observed = []
    gc.collect()
    with _tracked_storage() as live:
        baseline = _settled(live)

        for _ in range(cycles):
            # 1. the stateless Core, both surfaces
            core = _core_from_values(values)
            out, mask = core._dropout_forward_with_mask(
                DEFAULT_P, seed=BENCHMARK_SEED, call_index=0
            )
            out.close()
            mask.close()
            core.dropout_forward(DEFAULT_P, seed=BENCHMARK_SEED,
                                 call_index=1).close()
            core.close()

            # 2. the no-grad operation: the mask must be closed already
            generator = NativeGenerator(BENCHMARK_SEED)
            tensor = NativeTensor.from_array(values)
            result = tensor.dropout(DEFAULT_P, generator=generator)
            _require(_graph_resources(result) == (),
                     "a no-grad forward retained its mask")
            result.close()
            tensor.close()

            # 3. the differentiable operation, backward, and the graph
            #    release that frees the mask
            native_upstream = NativeTensor.from_array(upstream)
            tensor = NativeTensor.from_array(values, requires_grad=True)
            result = tensor.dropout(DEFAULT_P, generator=generator)
            _require(len(_graph_resources(result)) == 1,
                     "the differentiable forward saved no mask")
            result.backward(gradient=native_upstream)
            _require(_graph_resources(result) == (),
                     "the mask survived the one-shot backward")
            _release_gradients([tensor])
            _require(tensor.grad is None, "a gradient survived cleanup")
            result.close()
            tensor.close()
            native_upstream.close()

            # 4. an abandoned graph: close() must free the mask too
            tensor = NativeTensor.from_array(values, requires_grad=True)
            abandoned = tensor.dropout(DEFAULT_P, generator=generator)
            abandoned.close()
            tensor.close()

            # 5. the module, in both modes
            module = NativeDropout(DEFAULT_P, seed=BENCHMARK_SEED)
            tensor = NativeTensor.from_array(values)
            module(tensor).close()
            module.eval()
            _require(module(tensor) is tensor,
                     "an eval forward allocated a new tensor")
            module.train()
            tensor.close()
            _close_module(module)
            _require(not module.generator._has_active_reservation(),
                     "the module's generator kept a reservation")
            _require(not generator._has_active_reservation(),
                     "the operation generator kept a reservation")

            # 6. one whole training step
            model = _BenchmarkDropoutClassifier()
            optimizer = NativeAdam(model.parameters(), lr=STEP_LR)
            criterion = NativeCrossEntropyLoss()
            native_inputs = NativeTensor.from_array(inputs)
            model.train()
            logits = model(native_inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss.close()
            logits.close()
            _release_gradients(list(model.parameters()))
            optimizer.close()
            _close_module(model)
            native_inputs.close()

            observed.append(_settled(live))

        final = _settled(live)

    gc.collect()
    for index, count in enumerate(observed):
        if count != baseline:
            raise AssertionError(
                f"native live storage was {count}, not the baseline "
                f"{baseline}, after lifecycle cycle {index + 1}"
            )
    if final != baseline:
        raise AssertionError(
            f"native live storage finished at {final}, not the baseline "
            f"{baseline}"
        )
    checks = ["core_forward_and_mask_released",
              "no_grad_mask_closed_immediately",
              "graph_and_mask_released_by_backward",
              "abandoned_graph_released_by_close",
              "gradients_cleared", "module_eval_allocates_nothing",
              "no_reservation_outstanding",
              "training_step_released",
              "no_monotonic_growth", "returns_exactly_to_baseline"]
    return {
        "status": "passed",
        "cycles": cycles,
        "baseline_live_storages": baseline,
        "final_live_storages": final,
        "per_cycle_live_storages": observed,
        "checks": checks,
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
    calibration = SMOKE_CALIBRATION if smoke else CALIBRATION
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "process_architecture": platform.architecture()[0],
        "logical_cpus": os.cpu_count(),
        "numpy_version": np.__version__,
        "tensorforge_version": tensorforge.__version__,
        "native_backend": {
            "name": info["name"],
            "tensor_core": info["tensor_core"],
            "available": info["available"],
            "native_autograd": info["native_autograd"],
            "stable_framework_integration":
                info["stable_framework_integration"],
            "fault_injection_available": cpp.fault_injection_available(),
            # The build type, compiler, and flags are not exposed by the
            # loaded library, and guessing them would be worse than saying
            # so.
            "build_configuration": "not reported",
            "compiler": "not reported",
        },
        "dtype": "float64",
        "device": "cpu",
        "scope": "native Dropout stack (float64/cpu)",
        "timer": "time.perf_counter_ns",
        "primary_statistic": "median",
        "mode": "smoke" if smoke else "full",
        "warmup": warmup,
        "repetitions": repetitions,
        "backward_repetitions": min(repetitions, BACKWARD_REPETITIONS),
        "training_step_repetitions": min(repetitions,
                                         TRAINING_STEP_REPETITIONS),
        "calibration_target_ns": calibration["target_ns"],
        "calibration_maximum_iterations": calibration["maximum"],
        "lifecycle_cycles": (SMOKE_LIFECYCLE_CYCLES if smoke
                             else LIFECYCLE_CYCLES),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_benchmark(cases=None, families=None, warmup=DEFAULTS["warmup"],
                  repetitions=DEFAULTS["repetitions"], smoke=False):
    """Run the selected cases and return the JSON-ready payload.

    The reference and the native kernel are pinned to the committed G2
    vectors first; then every case's correctness gate runs **before** its
    timing, and a failed gate raises (the CLI turns that into a nonzero
    exit) so no timing is published for it. The lifecycle verification
    runs last, untimed. No timing threshold is applied anywhere — this
    only measures. Raises RuntimeError if the native backend is not
    built."""
    if not cpp.is_available():
        raise RuntimeError(
            "The experimental C++ backend is not built.\n"
            + cpp.build_instructions()
        )
    if families is not None:
        chosen = _resolve(families, FAMILIES, "family")
        by_family = tuple(name for name, spec in CASES.items()
                          if spec["family"] in chosen)
        cases = by_family if cases is None else tuple(cases) + by_family
    selected = _resolve(cases, tuple(CASES), "case")
    # Deduplicate (--case and --family can overlap) and keep registry order.
    selected = tuple(name for name in CASES if name in set(selected))
    warmup = _positive_int(warmup, "warmup")
    repetitions = _positive_int(repetitions, "repetitions")
    reference_validation = verify_reference()
    core_validation = verify_core_against_committed_vectors()
    records = _apply_relative_ratios(
        [_measure_case(name, warmup, repetitions, smoke) for name in selected]
    )
    lifecycle = verify_lifecycle(
        SMOKE_LIFECYCLE_CYCLES if smoke else LIFECYCLE_CYCLES
    )
    return {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "schema_version": RESULT_SCHEMA_VERSION,
        "mode": "smoke" if smoke else "full",
        "environment": _environment(warmup, repetitions, smoke),
        "reference_validation": reference_validation,
        "core_validation": core_validation,
        "cases": records,
        "measurement_count": len(records),
        "layer_differences": _layer_differences(records),
        "lifecycle": lifecycle,
        "disclaimer": DISCLAIMER,
    }


def _format_duration(seconds):
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


DISCLAIMER = (
    "Local characterization only -- not a performance contract. These "
    "numbers come\nfrom one machine, one build, one operating system, and "
    "one workload; they are\nnot cross-machine comparable without "
    "controlled conditions, and wall-clock\ntimings move with scheduling "
    "and CPU state. The observed ratios are\nobservations, not guarantees, "
    "and describe only what was measured here.\nCorrectness is gated before "
    "timing, timing is never a pass/fail criterion,\nand no test or CI job "
    "asserts a duration."
)

_FAMILY_TITLES = {
    "baseline": "Harness floor",
    "core_reference": "Core versus the exact NumPy reference",
    "size_scaling": "Size scaling (Core, with mask)",
    "layout": "Layout characterization (Core, with mask)",
    "probability": "Probability characterization",
    "tensor_operation": "NativeTensor operation layers",
    "module": "NativeDropout train / eval / p == 0",
    "training_step": "One complete Dropout training step",
}


def _format_probability(probability):
    """``%g`` would print the largest legal probability as ``1``, which is
    exactly the value the contract rejects — so that one is named."""
    if probability is None:
        return "-"
    if probability == MAX_P:
        return "1-ulp"
    return f"{probability:g}"


def _case_row(record):
    native = record["native"]
    reference = record["reference"]
    ratio = record["native_to_reference_ratio"]
    relative = record["native_relative_ratio"]
    per_element = record["ns_per_element"]
    shape_text = "x".join(str(d) for d in record["shape"]) or "scalar"
    return (
        f"{record['case']:<34} "
        f"{shape_text:<12} "
        f"{_format_probability(record['probability']):>8} "
        f"{_format_duration(native['median_s']):>12} "
        f"{(_format_duration(reference['median_s']) if reference else 'n/a'):>12} "
        f"{(f'{ratio:.2f}' if ratio is not None else 'n/a'):>7} "
        f"{(f'{relative:.2f}' if relative is not None else 'n/a'):>8} "
        f"{(f'{per_element:.2f}' if per_element is not None else 'n/a'):>9} "
        f"{_format_duration(native['spread_s']):>10} "
        f"{record['iterations_per_sample']:>5} "
        f"{record['correctness']['status']:<8}"
    )


def format_report(payload):
    """A concise human-readable report. Carries no speed verdict."""
    env = payload["environment"]
    lines = [
        f"TensorForge native Dropout benchmark v{payload['version']} "
        f"[{payload['mode']}]  (result schema "
        f"{payload['schema_version']})",
        f"  platform  : {env['platform']}  ({env['process_architecture']}, "
        f"{env['logical_cpus']} logical CPUs)",
        f"  machine   : {env['machine']}",
        f"  processor : {env['processor']}",
        f"  python    : {env['python_version']} "
        f"({env['python_implementation']})   numpy {env['numpy_version']}   "
        f"tensorforge {env['tensorforge_version']}",
        f"  backend   : {env['native_backend']['tensor_core']} "
        f"({env['dtype']}/{env['device']}, available="
        f"{env['native_backend']['available']}, build="
        f"{env['native_backend']['build_configuration']})",
        f"  timer     : {env['timer']}   primary statistic: "
        f"{env['primary_statistic']}",
        f"  warmup/repetitions : {env['warmup']}/{env['repetitions']} "
        f"(backward: {env['backward_repetitions']}, training step: "
        f"{env['training_step_repetitions']})",
        f"  identity loop      : calibrated to "
        f">= {env['calibration_target_ns']} ns/sample, at most "
        f"{env['calibration_maximum_iterations']} iterations",
        "",
        f"correctness prologue : reference "
        f"{payload['reference_validation']['status']} "
        f"({payload['reference_validation']['mask_vectors']} committed mask "
        f"vectors, {payload['reference_validation']['stream_vectors']} "
        f"stream vectors); native kernel "
        f"{payload['core_validation']['status']}",
        f"measurements         : {payload['measurement_count']}",
        "",
    ]
    header = (
        f"{'case':<34} {'shape':<12} {'p':>8} {'native':>12} "
        f"{'numpy':>12} {'n/np':>7} {'rel':>8} {'ns/elem':>9} "
        f"{'spread':>10} {'iter':>5} {'correct':<8}"
    )
    for family in FAMILIES:
        rows = [record for record in payload["cases"]
                if record["family"] == family]
        if not rows:
            continue
        lines.append(_FAMILY_TITLES.get(family, family))
        lines.append(header)
        lines.append("-" * len(header))
        for record in rows:
            lines.append(_case_row(record))
        lines.append("")

    if payload["layer_differences"]:
        lines.append("Layered differences (approximate, see legend)")
        for entry in payload["layer_differences"]:
            lines.append(
                f"  {entry['name']:<24} "
                f"{_format_duration(abs(entry['difference_s'])):>10}"
                f"{'  (negative)' if entry['difference_s'] < 0 else '':<12} "
                f"= {entry['minuend']} - {entry['subtrahend']}"
            )
        lines.append("")

    lifecycle = payload["lifecycle"]
    lines.append(
        f"Lifecycle verification: {lifecycle['status']} — "
        f"{lifecycle['cycles']} create/use/release cycles over every "
        f"family, native live storage {lifecycle['baseline_live_storages']} "
        f"-> {lifecycle['final_live_storages']} (baseline -> final), no "
        f"reservation outstanding."
    )
    lines.append("")
    lines.append(
        "Legend. 'native' and 'numpy' are median seconds per call.\n"
        "  n/np  = native median / numpy-reference median. Above 1 means "
        "the native path\n"
        "          took longer in this local run; below 1 means it took "
        "less time here.\n"
        "  rel   = this case's native median / the median of the case named "
        "in\n"
        "          'relative_to' (the contiguous layout, the p = 0.5 row of "
        "the same\n"
        "          layer, the no-grad forward, or the training-mode "
        "forward). Above 1\n"
        "          means this row took longer than that sibling here.\n"
        "  iter  = calls timed per sample. 1 everywhere except the "
        "identity-dispatch\n"
        "          rows (evaluation mode and p == 0), which return the "
        "caller's own\n"
        "          tensor, allocate nothing, and are too fast for one "
        "reading; those\n"
        "          rows measure **dispatch**, not mask generation, and must "
        "not be\n"
        "          compared with an allocating row. Read them against "
        "python_call_floor,\n"
        "          which is the same loop around a trivial Python call.\n"
        "  ns/elem is reported only where a per-element cost is defined; "
        "identity rows\n"
        "          show n/a. No memory-bandwidth figure is reported at all.\n"
        "  The Core has no p == 0 short-circuit, so its p = 0 row still "
        "allocates,\n"
        "          draws, and writes; the operation and module layers "
        "return the input.\n"
        "  NativeTensor and NativeDropout rows are native_only: no NumPy "
        "expression has\n"
        "          their generator transaction, ownership, or graph, so no "
        "ratio is\n"
        "          published for them. Their gates are still exact."
    )
    lines.append("")
    lines.append(LAYER_DIFFERENCE_CAVEAT)
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description=("Characterize the native Dropout stack (measurement "
                     "only; no speed is asserted).")
    )
    parser.add_argument("--case", choices=tuple(CASES), default=None,
                        metavar="CASE",
                        help="run a single case (default: all)")
    parser.add_argument("--family", choices=FAMILIES, default=None,
                        metavar="FAMILY",
                        help=f"run one family: {', '.join(FAMILIES)}")
    parser.add_argument("--warmup", type=int, default=None,
                        help="warm-up repetitions before measuring")
    parser.add_argument("--repetitions", type=int, default=None,
                        help="measured repetitions per case")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON only")
    parser.add_argument("--json-out", default=None, metavar="PATH",
                        help=("also write the JSON payload to PATH. No file "
                              "is written unless this is given"))
    parser.add_argument("--smoke", "--quick", action="store_true",
                        dest="smoke",
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
            families=[args.family] if args.family else None,
            warmup=warmup, repetitions=repetitions, smoke=args.smoke,
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))     # stderr, exit 2 — stdout stays clean
    except AssertionError as error:  # a correctness gate failed
        parser.exit(1, f"correctness gate failed: {error}\n")
    # allow_nan=False: a NaN or Infinity would be invalid JSON that most
    # readers accept silently, so the harness refuses to emit one rather
    # than publishing an unparseable number.
    document = json.dumps(payload, allow_nan=False)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(document)
    if args.json:
        print(document)
    else:
        print(format_report(payload))


if __name__ == "__main__":
    main()
