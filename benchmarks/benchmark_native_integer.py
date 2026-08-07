"""Characterization of native integer tensors and indexing (Phase K, K8).

Phase K shipped the internal ``int64`` representation and every
reachability barrier (K1), the publicly constructible ``int64`` tensor and
its one construction door (K2), native ``argmax`` (K3), forward-only
``index_select`` (K4), the compatibility proof (K5), the integration
example (K6), and the adversarial hardening matrix (K7). This harness
answers the one question none of them asked: **what does each of those
operations cost on this machine?**

Four separate questions, deliberately not blurred into one
----------------------------------------------------------

1. **Integer construction** — what does ``NativeTensor.from_int64_array``
   cost, host inspection, allocation and exact transfer included?
2. **Host materialization** — what does ``to_numpy()`` cost on an
   ``int64`` tensor, contiguous and through a non-contiguous view?
3. **``argmax``** — what does the floating reduction that *produces* an
   ``int64`` index cost, at each floating width separately?
4. **``index_select``** — what does the floating selection that
   *consumes* an ``int64`` index cost, at each floating width separately?

There is deliberately **no composed case**. A single
``argmax``-then-``index_select`` number could not say which of the two
dominates, and "which layer dominates" is the only question a measurement
here could ever be asked; answering it needs the operations apart. K8 adds
no composition rather than adding one that would have to be labelled as
never substituting for its parts.

What this harness is not
------------------------

**K8 is characterization only, and ships no optimization.** Nothing here
changes runtime behaviour, no measurement below may be used to change one
without its own separately approved milestone, and no production file was
touched to make any of it measurable.

**Nothing asserts a speed.** No timing threshold, no duration budget, no
throughput floor, no ratio limit, no comparison against a stored number,
and **no CI job that fails on one**. There is **no result file of any
kind** — not JSON, not CSV, not a cache, not a baseline. ``--json`` writes
to *stdout* and nowhere else, and the CLI has no output-path option
because it has nothing to write.

**Correctness runs before timing, always.** Every case validates its
result against an independently written host oracle *before* the timing
helper is reached, so a failed gate publishes no timing and the CLI exits
nonzero with clean stdout. The gates are exact: integer values and
``argmax`` results by exact integer equality, floating payloads by raw
IEEE-754 bit patterns inside one width, and dtype, shape, ownership,
contiguity, and graph-freedom by identity. No ``allclose``, no tolerance,
and no approximation appears anywhere — every operation measured here is a
copy, an exact index search, or a slice selection, and for those a
tolerance would assert less than the contract promises.

**The oracles are this harness's own.** ``argmax`` is gated against a
direct transcription of design §17.5's algorithm — first maximum wins,
signed zeros tie, the *lowest-indexed* NaN wins and nothing displaces it —
and against the committed case table of that section, never against
``numpy.argmax``, whose tie and NaN rules are a different library's
decisions. ``index_select`` is gated against a per-position slice
concatenation written without ``numpy.take``, so duplicates and order are
checked position by position rather than by a whole-array comparison that
could pass by luck.

**The widths are never divided by one another, and neither are the
roles.** ``float32`` and ``float64`` are measured and reported separately;
so is ``int64``. There is no float32/float64 ratio anywhere, and no
int64/floating ratio either: the first is a property of one machine's
memory bandwidth and the second would compare an index/result dtype
against a compute dtype, which is not a comparison this project makes.

References, and why every case is native-only
---------------------------------------------

Every case declares ``native_only`` and publishes **no ratio at all**,
because none of the four families has an honest host equivalent:

- **construction** allocates native storage and transfers into it;
  ``numpy.array`` or ``numpy.ascontiguousarray`` allocates host memory and
  does not.
- **materialization** allocates a fresh host array *and* crosses the ABI
  out of native storage; a host-to-host copy does neither.
- **``argmax``** allocates a fresh owning ``int64`` output tensor, which
  ``numpy.argmax`` over an existing host array does not — the live
  fairness risk design §31 names by name — and answers by a different tie
  and NaN rule besides.
- **``index_select``** allocates a fresh owning destination, scans every
  index for bounds in Python *and* independently in C++ before writing
  anything, and may materialize a Policy-B temporary; ``numpy.take`` does
  none of that.

Nothing is divided by anything, no reference timing is fabricated, and no
comparison layer is invented. A conservative ``native_only`` is worth more
than a ratio a reader would have to discount.

What is timed
-------------

One measured sample is exactly one call, timed with
``time.perf_counter_ns()``. Host arrays, native sources, index tensors,
and every view are built **outside** the timer, once per case; the result
of every repetition is closed **outside** the timer; and nothing here
relies on garbage collection. A non-contiguous operand's internal Policy-B
materialization and the destination allocation are *inside* the timed
call, because they are part of the operation — no internal work is moved
out to improve a number. No sample is discarded, no outlier is removed,
and no timer overhead is subtracted. The headline statistic is the
**median**; the spread is the **interquartile range**, stated rather than
left to be inferred, with p25, p75, the minimum, the maximum, and every
raw sample carried in the payload so a reader can recompute anything.

Modes
-----

::

    uv run python benchmarks/benchmark_native_integer.py
    uv run python benchmarks/benchmark_native_integer.py --smoke
    uv run python benchmarks/benchmark_native_integer.py --json
    uv run python benchmarks/benchmark_native_integer.py --smoke --json
    uv run python benchmarks/benchmark_native_integer.py --case argmax_full
    uv run python benchmarks/benchmark_native_integer.py --workload argmax
    uv run python benchmarks/benchmark_native_integer.py --dtype float32
    uv run python benchmarks/benchmark_native_integer.py --dtype int64
    uv run python benchmarks/benchmark_native_integer.py --profile argmax_full

``--dtype`` selects a **measured** dtype and says nothing about the
compute registry. ``int64`` selects the two families whose data *is* an
index/result buffer — construction and host materialization — while
``float64``/``float32`` select the floating **source** width of ``argmax``
and ``index_select``. ``int64`` is not in ``SUPPORTED_DTYPES`` and never
will be; it is in the separate ``INDEX_DTYPES`` row, and both are reported
in the payload's environment block so the distinction is visible rather
than asserted.

K8 adds **no** capability: no kernel, C ABI export, ctypes declaration,
public API, package export, dtype, device, registry value, checkpoint
field or version, dependency, build option, example, CTest, or CI job. It
only measures what K1-K7 shipped.
"""

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tensorforge                                          # noqa: E402
from tensorforge.backends import cpp                        # noqa: E402
from tensorforge.experimental import NativeTensor           # noqa: E402

# ---------------------------------------------------------------------------
# Identity. These name the *measurement payload*, not a package export:
# nothing here is added to ``tensorforge.__all__`` or
# ``tensorforge.experimental.__all__``, and there is no benchmark registry
# inside the package. A measurement tool is never a capability.
# ---------------------------------------------------------------------------
BENCHMARK_NAME = "tensorforge.native_integer"
BENCHMARK_VERSION = "1.0"
# The JSON payload's own contract version. Bumped only when the *shape* of
# the payload changes — never when a measured number does.
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# The dtype axis, in three rows that are three different questions.
#
# ``FLOATING_DTYPES`` is the public compute registry, in its contractual
# order (float64 first, because it is the default an omitted ``dtype``
# selects). ``INDEX_DTYPE`` is the separate index/result dtype: it is not
# in ``SUPPORTED_DTYPES``, ``normalize_dtype("int64")`` raises permanently,
# and no generic constructor accepts it. ``MEASURED_DTYPES`` is neither
# registry — it is only the set of values ``--dtype`` may name here.
# ---------------------------------------------------------------------------
FLOATING_DTYPES = ("float64", "float32")
INDEX_DTYPE = "int64"
INDEX_DTYPES = (INDEX_DTYPE,)
MEASURED_DTYPES = FLOATING_DTYPES + INDEX_DTYPES

NUMPY_DTYPES = {"float64": np.float64, "float32": np.float32,
                "int64": np.int64}
BIT_DTYPES = {"float64": np.uint64, "float32": np.uint32}

# What a case's dtype *means*. Recorded on every row so a reader never has
# to infer whether a width is a compute dtype or an index one.
ROLE_FLOATING_SOURCE = "floating_source"
ROLE_INDEX = "index_or_result"
DTYPE_ROLES = (ROLE_FLOATING_SOURCE, ROLE_INDEX)

# The exact 64-bit boundary values, embedded in every integer payload so
# construction and materialization are proved exact beyond float64's exact
# integer range rather than merely plausible inside it.
INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1
# The value a strided host base carries in the positions the *view* skips.
# It must never appear in a constructed tensor; the gate proves it does
# not, which is what makes the strided case a real layout check.
STRIDE_FILLER = -7

# ---------------------------------------------------------------------------
# Workload families, in report order. Each of the four is its own family,
# and there is no fifth: K8 ships no composition.
# ---------------------------------------------------------------------------
INTEGER_CONSTRUCTION = "integer_construction"
HOST_MATERIALIZATION = "host_materialization"
ARGMAX = "argmax"
INDEX_SELECT = "index_select"
WORKLOADS = (INTEGER_CONSTRUCTION, HOST_MATERIALIZATION, ARGMAX,
             INDEX_SELECT)

# Reference types. ``native_only`` means no honest equivalent exists, so no
# ratio is published for that case anywhere — in the payload or the report.
# K8's registry has exactly one member, because every one of its four
# families allocates and transfers where the apparent host equivalent does
# not. An unused second label would be a comparison layer waiting to be
# invented.
NATIVE_ONLY = "native_only"
REFERENCE_TYPES = (NATIVE_ONLY,)

# Correctness gates, one per operation family. Each is exact; none is a
# tolerance.
GATE_CONSTRUCTION = "int64_construction_exact"
GATE_MATERIALIZATION = "int64_materialization_exact"
GATE_ARGMAX = "argmax_oracle_exact"
GATE_INDEX_SELECT = "index_select_oracle_bits"
GATES = (GATE_CONSTRUCTION, GATE_MATERIALIZATION, GATE_ARGMAX,
         GATE_INDEX_SELECT)

# Operand geometries and layouts. A case names both, so what it measures is
# auditable without reading its builder.
GEOMETRY_VECTOR = "vector"
GEOMETRY_MATRIX = "matrix"
GEOMETRIES = (GEOMETRY_VECTOR, GEOMETRY_MATRIX)

LAYOUT_CONTIGUOUS = "contiguous"
LAYOUT_STRIDED_HOST = "strided_host"
LAYOUT_TRANSPOSED = "transposed"
LAYOUT_OFFSET = "offset"
LAYOUTS = (LAYOUT_CONTIGUOUS, LAYOUT_STRIDED_HOST, LAYOUT_TRANSPOSED,
           LAYOUT_OFFSET)

# How a case's index vector is built. ``distinct`` never repeats,
# ``duplicates`` repeats every value exactly twice, and ``random`` is a
# deterministic draw with replacement (the only honest shape when the
# selection is longer than the axis).
PATTERN_DISTINCT = "distinct"
PATTERN_DUPLICATES = "duplicates"
PATTERN_RANDOM = "random"
INDEX_PATTERNS = (PATTERN_DISTINCT, PATTERN_DUPLICATES, PATTERN_RANDOM)

# Warm-up and repetition defaults, shared by every case — the policy J8
# established, unchanged, because the operations here are all cheap enough
# that one policy is honest for all of them. A case does not get to pick
# its own count, and the count actually used is reported on every record.
DEFAULTS = {"warmup": 3, "repetitions": 11}
SMOKE_DEFAULTS = {"warmup": 1, "repetitions": 3}
PROFILE_DEFAULTS = {"warmup": 5, "repetitions": 25}
CONFIGURATIONS = ("smoke", "full", "profile")

# Environment variables that change how a BLAS-backed NumPy behaves.
# Recorded only when actually set — nothing is invented.
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

# ---------------------------------------------------------------------------
# The committed ``argmax`` case table, transcribed from
# docs/native_integer_tensors_design.md §17.5. These are the
# *specification* of the value rule, not a regression convenience: every
# ``argmax`` gate runs them as known-answer checks beside its own data, so
# the tie rule, the signed-zero rule, the infinity rule, and the
# first-NaN-wins rule are proved by literal rather than by self-agreement
# with the oracle. Every value is exactly representable at both widths.
# ---------------------------------------------------------------------------
_INF = float("inf")
_NAN = float("nan")
ARGMAX_REFERENCE_RUNS = (
    ((1.0, 5.0, 2.0), 1, "no NaN, unique maximum"),
    ((3.0, 3.0, 1.0), 0, "no NaN, equal maxima -> the lowest index"),
    ((-0.0, 0.0, -1.0), 0, "signed zeros tie -> the lowest index"),
    ((0.0, -0.0, -1.0), 0, "signed zeros tie, the other order"),
    ((-_INF, -_INF, -_INF), 0, "every element -inf -> 0"),
    ((1.0, _INF, 2.0), 1, "+inf present -> its lowest index"),
    ((1.0, _NAN, 2.0), 1, "exactly one NaN -> its index"),
    ((_NAN, 1.0, _NAN), 0, "several NaNs -> the first"),
    ((_INF, _NAN, 3.0), 1, "+inf never displaces a NaN"),
    ((-_INF, _NAN), 1, "-inf never displaces a NaN"),
    ((_NAN, 5.0), 0, "NaN at index 0 displaces nothing thereafter"),
    ((7.0,), 0, "a run of length 1"),
)

DISCLAIMER = (
    "Local characterization only, and not a performance contract. These "
    "numbers describe one machine, one build, and one moment; they are not "
    "cross-machine comparable. Correctness is gated before timing, no "
    "duration is ever a pass/fail criterion, no threshold or CI timing job "
    "exists, no result file is written, and no optimization ships with "
    "this measurement. float64, float32, and int64 are measured separately "
    "and none is ever divided by another: int64 is an index/result dtype, "
    "not a supported compute dtype."
)


# ===========================================================================
# Timing and statistics
# ===========================================================================


class Case:
    """One measurable case: a correctness gate and a measured operation.

    ``prepare()`` builds whatever one call consumes and runs **outside**
    the timer, once per repetition. ``run(state)`` is the one timed
    operation and nothing else. ``cleanup(state, result)`` releases
    everything the call produced — also outside the timer. ``teardown()``
    releases what the case itself owns.

    There is deliberately **no reference machinery**: every K8 case is
    ``native_only``, so there is nothing to time on the other side of a
    ratio and no place for one to be added by accident.
    """

    __slots__ = ("check", "prepare", "run", "cleanup", "teardown")

    def __init__(self, check, prepare, run, cleanup, teardown):
        self.check = check
        self.prepare = prepare
        self.run = run
        self.cleanup = cleanup
        self.teardown = teardown


def _no_state():
    """A ``prepare`` for a case whose call consumes nothing per
    repetition. It still runs once per repetition, outside the timer, so
    every case goes through the identical measurement shape."""
    return None


def _discard(state, result):
    """A ``cleanup`` for a call whose result owns no native storage.

    The result is dropped rather than kept: retaining results to simplify
    reporting would hold a whole run's worth of host arrays for no
    reason."""
    return None


def _close_result(state, result):
    """A ``cleanup`` for a call that returns a fresh owning native tensor.

    Explicit and outside the timer. Nothing in this harness waits for
    garbage collection, and no result is retained."""
    if result is not None:
        result.close()


def measure(prepare, run, cleanup, warmup, repetitions):
    """Warm up, then time exactly ``repetitions`` single calls, in
    nanoseconds.

    One sample is one ``run(state)`` call and nothing else. ``prepare``
    and ``cleanup`` bracket every repetition **outside** the measured
    region, so per-repetition state construction and native cleanup are
    never timed. Warm-up repetitions run the identical shape and are
    discarded before measuring. **No measured sample is dropped and no
    timer overhead is subtracted.** CPU execution is synchronous, so
    nothing needs to be waited on.
    """
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
        samples.append(elapsed)
    return samples


def percentile(ordered, quantile):
    """A deterministic linear-interpolation percentile over a **sorted**
    sequence.

    Spelled out rather than delegated so the definition is part of the
    contract and can be checked against known answers: position
    ``q * (n - 1)``, interpolating linearly between the two neighbouring
    order statistics. This is the inclusive convention; a single sample is
    its own every percentile.
    """
    if not ordered:
        raise ValueError("percentile of an empty sample set")
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower]) + (float(ordered[upper])
                                    - float(ordered[lower])) * weight


def summarize(samples):
    """Median with an explicit, named spread statistic, plus every raw
    sample.

    The headline is the **median**. The spread is the **interquartile
    range** — robust against the single scheduling outlier a short run
    always contains — published beside p25 and p75 so the number can be
    checked rather than trusted, and beside the minimum and maximum so the
    tail it hides is visible too. The arithmetic mean is carried as
    secondary information only. Nothing here is discarded, trimmed,
    winsorized, or corrected.
    """
    ordered = sorted(samples)
    p25 = percentile(ordered, 0.25)
    p75 = percentile(ordered, 0.75)
    median = statistics.median(ordered)
    return {
        "sample_count": len(ordered),
        "samples_ns": [int(sample) for sample in samples],
        "median_ns": float(median),
        "min_ns": float(ordered[0]),
        "max_ns": float(ordered[-1]),
        "p25_ns": p25,
        "p75_ns": p75,
        "iqr_ns": p75 - p25,
        "mean_ns": float(statistics.fmean(ordered)),
        "relative_iqr": ((p75 - p25) / float(median)) if median > 0 else None,
        "headline_statistic": "median",
        "spread_statistic": "interquartile range (p75 - p25)",
        "units": "nanoseconds_per_call",
    }


# ===========================================================================
# Deterministic inputs
#
# Every generator is local and explicitly seeded. NumPy's global RNG and
# Python's ``random`` are never read or mutated, so two runs of this file
# build byte-identical inputs and nothing else in the process is disturbed.
# No size, seed, or value is derived from a clock, the environment, or the
# filesystem.
# ===========================================================================


def logical_shape(spec, config):
    """The logical shape of the object a case measures.

    A transposed case names the *view's* shape here and builds its base
    with the axes reversed, so the transposed and contiguous cases of a
    family describe the same logical object and differ only in layout.
    """
    if spec["geometry"] == GEOMETRY_VECTOR:
        return (int(config["elements"]),)
    return (int(config["rows"]), int(config["columns"]))


def host_int64(shape, seed):
    """Deterministic exact ``int64`` host values, with the 64-bit
    boundaries embedded.

    ``INT64_MIN`` and ``INT64_MAX`` sit at fixed positions in every
    payload, so a construction or materialization that quietly went
    through a ``double`` would fail the gate rather than pass on values
    that happen to be small. Zero and -1 join them where there is room,
    because a sign-handling mistake shows up there first.
    """
    generator = np.random.default_rng(seed)
    values = generator.integers(-(2 ** 62), 2 ** 62, size=shape,
                                dtype=np.int64)
    values = np.ascontiguousarray(values)
    flat = values.reshape(-1)
    flat[0] = INT64_MIN
    flat[-1] = INT64_MAX
    if flat.size >= 4:
        flat[1] = 0
        flat[2] = -1
    return values


def strided_host_int64(logical):
    """A genuinely non-contiguous exact-``int64`` host array carrying
    ``logical``'s values, and the contiguous base underneath it.

    Every skipped position holds ``STRIDE_FILLER``, which the gate proves
    never reaches the constructed tensor — otherwise a constructor that
    read the base rather than the view would pass.
    """
    base = np.full((logical.size * 2,), STRIDE_FILLER, dtype=np.int64)
    base[::2] = logical.reshape(-1)
    view = base[::2]
    if view.flags["C_CONTIGUOUS"]:                # pragma: no cover
        raise AssertionError("the strided host view is contiguous")
    return base, view


def host_floating(shape, dtype, seed):
    """Finite, moderate values, **physically at the requested dtype**.

    Converted here rather than left for the constructor to convert, so the
    array the gate's oracle reads is byte-identical to the one the native
    tensor was built from. Values are drawn on a fine grid and shifted, so
    equal maxima along a reduction axis are vanishingly unlikely and the
    timed workload characterizes the ordinary path; the exceptional tie,
    signed-zero, infinity, and NaN answers are proved by the committed
    §17.5 table instead, outside the timer.
    """
    generator = np.random.default_rng(seed)
    values = generator.uniform(-2.0, 2.0, size=shape)
    return np.ascontiguousarray(values, dtype=NUMPY_DTYPES[dtype])


def index_values(pattern, axis_length, count, seed):
    """A deterministic in-bounds index vector, as exact ``int64``.

    ``distinct`` never repeats, so the selection is a permutation prefix;
    ``duplicates`` repeats each of ``count // 2`` values exactly twice,
    adjacent, which is the shape a duplicate-preservation check can see;
    ``random`` draws with replacement, the only honest shape when the
    selection is longer than the axis it selects from.
    """
    generator = np.random.default_rng(seed)
    if pattern == PATTERN_DISTINCT:
        if count > axis_length:
            raise ValueError("a distinct selection cannot exceed the axis")
        chosen = generator.permutation(axis_length)[:count]
    elif pattern == PATTERN_DUPLICATES:
        half = count // 2
        if half > axis_length:
            raise ValueError("a duplicated selection cannot exceed the axis")
        base = generator.permutation(axis_length)[:half]
        chosen = np.repeat(base, 2)[:count]
    elif pattern == PATTERN_RANDOM:
        chosen = generator.integers(0, axis_length, size=count)
    else:
        raise ValueError(f"unknown index pattern {pattern!r}")
    return np.ascontiguousarray(chosen, dtype=np.int64)


def bits_of(array, dtype):
    """Raw IEEE-754 bit patterns, with the dtype **asserted rather than
    coerced**.

    A helper that quietly converted could report a match that only existed
    after a conversion this runtime never performs — which is exactly the
    mistake that would make the whole harness dishonest."""
    array = np.asarray(array)
    if array.dtype != NUMPY_DTYPES[dtype]:
        raise AssertionError(f"expected a {dtype} array, got {array.dtype}")
    return np.ascontiguousarray(array).reshape(-1).view(BIT_DTYPES[dtype])


def exact_int64(array, label):
    """The array, proved to be exactly native-order ``numpy.int64``.

    Byte order is checked explicitly: a byte-swapped ``>i8`` array
    compares equal element by element while being a different buffer, and
    "the values match" is not the claim this boundary makes."""
    array = np.asarray(array)
    # Byte order first, so a foreign-order buffer is reported as the thing
    # it is rather than as a generic dtype mismatch.
    if array.dtype.byteorder not in ("=", "|"):
        raise AssertionError(
            f"{label}: byte order {array.dtype.byteorder!r} is not native")
    if array.dtype != np.int64:
        raise AssertionError(f"{label}: dtype {array.dtype}, expected int64")
    return array


def require(condition, message):
    if not condition:
        raise AssertionError(message)


# ===========================================================================
# Independent host oracles
#
# Neither of these calls the operation it validates, and neither delegates
# its rule to NumPy: ``numpy.argmax`` resolves ties and NaNs by another
# library's decisions, and ``numpy.take`` is the very operation the
# selection oracle would otherwise be checking against itself.
# ===========================================================================


def argmax_run(run):
    """Design §17.5's algorithm, transcribed.

    Scanning left to right from ``run[0]``: a strict ``>`` displaces the
    incumbent, a NaN displaces any non-NaN incumbent, and **nothing
    displaces an incumbent NaN**. Total, allocation-free, reading each
    element exactly once, and never inspecting a NaN's payload, sign, or
    signalling bit.

    Starting at ``run[0]`` rather than at a sentinel is load-bearing: a
    sentinel start makes an all-``-inf`` run and an all-NaN run return 0
    *by accident*, and this start makes them return 0 *by construction*.
    """
    if not run:
        raise ValueError("argmax over an empty run")
    best_index = 0
    best = run[0]
    for position in range(1, len(run)):
        value = run[position]
        if math.isnan(best):
            continue
        if math.isnan(value) or value > best:
            best = value
            best_index = position
    return best_index


def argmax_oracle(values, axis, keepdims):
    """``argmax`` over a host array, by §17.5's rule and this repository's
    reduction shapes.

    ``axis=None`` reduces the flat row-major sequence — the order
    ``to_numpy()`` produces — and answers with a flat index; an explicit
    axis reduces each run along it independently, so a NaN in one run
    never reaches another. Shapes follow ``reduce_shape``: the axis
    disappears, or stays as 1 under ``keepdims``.

    A float32 row is widened to Python floats before the scan. That is
    exact and order-preserving — every float32 value is a float64 value,
    and ``>`` agrees at both widths — so the answer is the float32 answer
    rather than an approximation of it.
    """
    array = np.asarray(values)
    if axis is None:
        index = argmax_run(array.reshape(-1).tolist())
        result = np.array(index, dtype=np.int64)
        if keepdims:
            result = result.reshape((1,) * array.ndim)
        return result
    normalized = axis if axis >= 0 else axis + array.ndim
    moved = np.moveaxis(array, normalized, -1)
    trailing = moved.shape[-1]
    answers = [argmax_run(row.tolist())
               for row in moved.reshape(-1, trailing)]
    result = np.array(answers, dtype=np.int64).reshape(moved.shape[:-1])
    if keepdims:
        result = np.expand_dims(result, normalized)
    return result


def index_select_oracle(source, axis, wanted):
    """``index_select`` over a host array, built one selected slice at a
    time.

    Written without ``numpy.take`` on purpose: the operation's whole
    contract is that duplicates and order are preserved exactly, and a
    concatenation of per-position slices preserves both *by construction*
    rather than by trusting a library routine to agree.
    """
    normalized = axis if axis >= 0 else axis + source.ndim
    pieces = []
    for value in wanted:
        selector = [slice(None)] * source.ndim
        selector[normalized] = slice(int(value), int(value) + 1)
        pieces.append(source[tuple(selector)])
    return np.concatenate(pieces, axis=normalized)


def selected_slice(array, axis, position):
    """One slice of ``array`` along ``axis``, keeping the axis at length 1
    so two slices can be compared position by position."""
    normalized = axis if axis >= 0 else axis + array.ndim
    selector = [slice(None)] * array.ndim
    selector[normalized] = slice(int(position), int(position) + 1)
    return array[tuple(selector)]


# ===========================================================================
# Correctness gates. Every one runs before the timing helper is reached and
# raises AssertionError on failure, which the CLI turns into a nonzero exit
# with clean stdout. All of them are exact: no tolerance appears anywhere.
# ===========================================================================


def gate_construction(make_host, logical, label):
    """``from_int64_array``: the exact host-to-native integer boundary.

    dtype, shape, ownership, contiguity, and graph-freedom by identity;
    the values by exact ``int64`` equality against the host array this
    case will time against, including ``INT64_MIN`` and ``INT64_MAX``,
    which no float64 detour could carry. Independence is proved by
    mutating a *probe* host buffer after construction and showing the
    tensor did not move, and freshness by building two tensors and closing
    one. Every tensor the gate creates is closed here, before any timing
    exists.
    """
    probe_base, probe_view = make_host()
    first = NativeTensor.from_int64_array(probe_view)
    second = None
    try:
        require(first.dtype == INDEX_DTYPE,
                f"{label}: dtype {first.dtype!r}, expected {INDEX_DTYPE!r}")
        require(first.shape == logical.shape,
                f"{label}: shape {first.shape}, expected {logical.shape}")
        require(first.device == "cpu",
                f"{label}: device {first.device!r}, expected 'cpu'")
        require(first.owns_core, f"{label}: the result does not own its core")
        require(first.contiguous,
                f"{label}: the result is not contiguous native storage")
        require(first.requires_grad is False,
                f"{label}: an int64 tensor requires grad")
        require(first.grad is None, f"{label}: an int64 tensor carries a grad")
        require(first.is_leaf is True, f"{label}: the result is not a leaf")
        produced = exact_int64(first.to_numpy(), label)
        require(np.array_equal(produced, logical),
                f"{label}: the tensor does not hold the host values exactly")
        require(int(produced.reshape(-1)[0]) == INT64_MIN,
                f"{label}: INT64_MIN did not survive construction")
        require(int(produced.reshape(-1)[-1]) == INT64_MAX,
                f"{label}: INT64_MAX did not survive construction")
        # A strided input's skipped base positions must not leak. Compared
        # against the logical array rather than banned outright, so a
        # payload that legitimately contains the filler value cannot make
        # this check fire spuriously.
        require(bool(np.any(produced == STRIDE_FILLER))
                == bool(np.any(logical == STRIDE_FILLER)),
                f"{label}: a skipped host position reached the tensor")
        # Ingress is a copy: editing the caller's buffer afterwards must
        # reach nothing, at either layout.
        probe_base.reshape(-1)[0] = INT64_MIN + 1
        require(np.array_equal(exact_int64(first.to_numpy(), label), logical),
                f"{label}: the tensor aliases the host array it was built "
                f"from")
        second = NativeTensor.from_int64_array(make_host()[1])
        require(second is not first,
                f"{label}: a repeated construction returned the same object")
        first.close()
        require(second.closed is False,
                f"{label}: closing one tensor closed another")
        require(np.array_equal(exact_int64(second.to_numpy(), label), logical),
                f"{label}: closing one tensor disturbed another")
    finally:
        if not first.closed:
            first.close()
        if second is not None and not second.closed:
            second.close()
    return {
        "gate": GATE_CONSTRUCTION,
        "elements": int(logical.size),
        "shape": [int(dimension) for dimension in logical.shape],
        "contiguous_host_input": bool(
            np.asarray(probe_view).flags["C_CONTIGUOUS"]),
        "int64_boundaries_checked": True,
        "independent_of_host_memory": True,
        "owning_contiguous_graph_free": True,
    }


def gate_materialization(tensor, logical, label):
    """``to_numpy()`` on an ``int64`` tensor: the exact native-to-host
    boundary.

    An exact native-order ``numpy.int64`` array, C-contiguous, owning its
    own memory, in the tensor's logical order — proved against the host
    values the tensor was built from, including both 64-bit boundaries.
    Independence is proved by editing what one call returned and showing a
    second call is unmoved; freshness by proving two calls are distinct
    arrays that share no memory. The source tensor's metadata is captured
    before and compared after, because a materialization that changed its
    source would be a different operation.
    """
    before = (tensor.dtype, tensor.shape, tensor.strides, tensor.contiguous,
              tensor.closed, tensor.owns_core)
    produced = exact_int64(tensor.to_numpy(), label)
    require(isinstance(produced, np.ndarray),
            f"{label}: to_numpy did not return an ndarray")
    require(produced.shape == logical.shape,
            f"{label}: shape {produced.shape}, expected {logical.shape}")
    require(produced.flags["C_CONTIGUOUS"],
            f"{label}: the materialized array is not C-contiguous")
    require(produced.flags["WRITEABLE"],
            f"{label}: the materialized array is not the caller's to edit")
    require(np.array_equal(produced, logical),
            f"{label}: the materialized values are not the source values")
    require(int(produced.reshape(-1)[0]) == INT64_MIN,
            f"{label}: INT64_MIN did not survive materialization")
    require(int(produced.reshape(-1)[-1]) == INT64_MAX,
            f"{label}: INT64_MAX did not survive materialization")
    # Egress is a copy: editing what one call returned must not be visible
    # in the next one.
    produced.reshape(-1)[0] = INT64_MIN + 1
    again = exact_int64(tensor.to_numpy(), label)
    require(np.array_equal(again, logical),
            f"{label}: to_numpy returned a view into native storage")
    require(again is not produced and not np.shares_memory(again, produced),
            f"{label}: two materializations share host memory")
    after = (tensor.dtype, tensor.shape, tensor.strides, tensor.contiguous,
             tensor.closed, tensor.owns_core)
    require(before == after,
            f"{label}: materializing changed the source tensor's metadata")
    return {
        "gate": GATE_MATERIALIZATION,
        "elements": int(logical.size),
        "shape": [int(dimension) for dimension in logical.shape],
        "contiguous_source": bool(tensor.contiguous),
        "int64_boundaries_checked": True,
        "independent_host_memory": True,
        "source_unchanged": True,
    }


def _gate_argmax_reference_table(dtype, label):
    """The committed §17.5 case table, run against the live operation at
    ``dtype``.

    Twelve known answers — unique maximum, equal maxima, both signed-zero
    orders, all ``-inf``, ``+inf``, one NaN, several NaNs, a NaN against
    either infinity, a NaN at index 0, and a length-1 run — each checked
    against the **literal** from the design and against this harness's own
    oracle. Running both is the point: the literal proves the runtime, and
    the agreement proves the oracle every other gate below relies on.
    """
    for values, expected, description in ARGMAX_REFERENCE_RUNS:
        host = np.ascontiguousarray(values, dtype=NUMPY_DTYPES[dtype])
        source = NativeTensor.from_array(host, dtype=dtype)
        result = None
        try:
            result = source.argmax()
            produced = exact_int64(result.to_numpy(), label)
            require(int(produced) == expected,
                    f"{label}: {description} gave {int(produced)}, the "
                    f"committed answer is {expected}")
            require(int(argmax_oracle(host, None, False)) == expected,
                    f"{label}: the oracle disagrees with the committed "
                    f"answer for {description}")
        finally:
            if result is not None:
                result.close()
            source.close()
    return len(ARGMAX_REFERENCE_RUNS)


def gate_argmax(source, logical, axis, keepdims, dtype, label):
    """``argmax``: an exact ``int64`` result, against an independent host
    oracle and against the committed §17.5 table.

    The result's dtype, shape, ownership, contiguity, and graph-freedom by
    identity; its values by **exact integer equality** with the oracle,
    never a tolerance — an index is either right or wrong. The source is
    proved unchanged, and a repeated call is proved to give the same
    answer, so the timed repetitions really do measure the same operation.
    """
    before = (source.dtype, source.shape, source.strides, source.contiguous,
              source.requires_grad)
    expected = argmax_oracle(logical, axis, keepdims)
    result = None
    repeat = None
    try:
        result = source.argmax(axis=axis, keepdims=keepdims)
        require(result.dtype == INDEX_DTYPE,
                f"{label}: result dtype {result.dtype!r}, expected "
                f"{INDEX_DTYPE!r}")
        require(result.shape == expected.shape,
                f"{label}: result shape {result.shape}, expected "
                f"{expected.shape}")
        require(result.owns_core, f"{label}: the result does not own its core")
        require(result.contiguous, f"{label}: the result is not contiguous")
        require(result.requires_grad is False,
                f"{label}: an argmax result requires grad")
        require(result.grad is None, f"{label}: an argmax result has a grad")
        require(result.is_leaf is True,
                f"{label}: an argmax result is not a plain leaf")
        produced = exact_int64(result.to_numpy(), label)
        require(np.array_equal(produced, expected),
                f"{label}: the native argmax disagrees with the §17.5 oracle")
        repeat = source.argmax(axis=axis, keepdims=keepdims)
        require(np.array_equal(exact_int64(repeat.to_numpy(), label),
                               produced),
                f"{label}: two identical argmax calls disagree")
        rows = _gate_argmax_reference_table(dtype, label)
    finally:
        if result is not None:
            result.close()
        if repeat is not None:
            repeat.close()
    after = (source.dtype, source.shape, source.strides, source.contiguous,
             source.requires_grad)
    require(before == after,
            f"{label}: argmax changed its source tensor's metadata")
    return {
        "gate": GATE_ARGMAX,
        "elements": int(logical.size),
        "shape": [int(dimension) for dimension in logical.shape],
        "axis": axis,
        "keepdims": keepdims,
        "reduced_run_length": int(logical.size if axis is None
                                  else logical.shape[axis]),
        "output_elements": int(expected.size),
        "contiguous_source": bool(source.contiguous),
        "reference_rows_checked": rows,
        "checked_against_reference_vector": True,
        "owning_contiguous_graph_free": True,
    }


def gate_index_select(source, logical, axis, index_tensor, wanted, dtype,
                      label):
    """``index_select``: whole selected slices, by raw IEEE-754 bits.

    The output shape, dtype, ownership, contiguity, and graph-freedom by
    identity; the values against a per-position slice concatenation
    written without ``numpy.take``. The comparison then runs **position by
    position** as well, so a deduplicated, sorted, or reordered result
    fails here rather than passing a whole-array comparison by luck. Both
    operands are proved unchanged, and the index tensor's own values are
    read back and proved to be the ones this case planned.
    """
    source_before = (source.dtype, source.shape, source.strides,
                     source.contiguous, source.requires_grad)
    index_before = (index_tensor.dtype, index_tensor.shape,
                    index_tensor.strides, index_tensor.contiguous)
    require(index_tensor.dtype == INDEX_DTYPE,
            f"{label}: the index tensor is {index_tensor.dtype!r}, not "
            f"{INDEX_DTYPE!r}")
    require(index_tensor.ndim == 1,
            f"{label}: the index tensor has rank {index_tensor.ndim}")
    require(np.array_equal(exact_int64(index_tensor.to_numpy(), label),
                           wanted),
            f"{label}: the index tensor does not hold the planned values")
    expected = index_select_oracle(logical, axis, wanted)
    result = None
    try:
        result = source.index_select(axis, index_tensor)
        require(result.dtype == dtype,
                f"{label}: result dtype {result.dtype!r}, expected {dtype!r}")
        require(result.shape == expected.shape,
                f"{label}: result shape {result.shape}, expected "
                f"{expected.shape}")
        require(result.owns_core, f"{label}: the result does not own its core")
        require(result.contiguous, f"{label}: the result is not contiguous")
        require(result.requires_grad is False,
                f"{label}: an index_select result requires grad")
        require(result.grad is None,
                f"{label}: an index_select result has a grad")
        require(result.is_leaf is True,
                f"{label}: an index_select result is not a plain leaf")
        produced = result.to_numpy()
        require(np.array_equal(bits_of(produced, dtype),
                               bits_of(expected, dtype)),
                f"{label}: the selection is not bit-identical to the oracle")
        # Position by position, so duplicates and order are checked rather
        # than assumed from a whole-array match.
        for position, value in enumerate(wanted):
            if not np.array_equal(
                    bits_of(selected_slice(produced, axis, position), dtype),
                    bits_of(selected_slice(logical, axis, int(value)),
                            dtype)):
                raise AssertionError(
                    f"{label}: output position {position} is not source "
                    f"slice {int(value)}")
    finally:
        if result is not None:
            result.close()
    source_after = (source.dtype, source.shape, source.strides,
                    source.contiguous, source.requires_grad)
    index_after = (index_tensor.dtype, index_tensor.shape,
                   index_tensor.strides, index_tensor.contiguous)
    require(source_before == source_after,
            f"{label}: index_select changed its source tensor's metadata")
    require(index_before == index_after,
            f"{label}: index_select changed its index tensor's metadata")
    duplicates = len(set(int(value) for value in wanted)) != len(wanted)
    return {
        "gate": GATE_INDEX_SELECT,
        "elements": int(logical.size),
        "shape": [int(dimension) for dimension in logical.shape],
        "axis": axis,
        "index_count": int(len(wanted)),
        "output_elements": int(expected.size),
        "duplicate_indices": duplicates,
        "contiguous_source": bool(source.contiguous),
        "contiguous_index": bool(index_tensor.contiguous),
        "compared_as_raw_bits": True,
        "owning_contiguous_graph_free": True,
    }


# ===========================================================================
# Case builders. Each returns a Case whose gate has already been written
# against an independent oracle; none of them times anything.
# ===========================================================================


def build_construction(dtype, config, spec):
    """``NativeTensor.from_int64_array(host)`` — the one public door
    through which an ``int64`` buffer comes into existence.

    ``native_only``: the call allocates native storage and transfers into
    it, and a host ``numpy.array`` copy allocates host memory and does
    not. Dividing one by the other would credit or blame TensorForge for
    an allocation the reference never made.

    The host array, and the strided base beneath it where the case has
    one, are built **once, outside the timer**. The timed call is exactly
    the constructor, and the tensor it returns is closed after every
    repetition, also outside the timer.
    """
    shape = logical_shape(spec, config)
    logical = host_int64(shape, spec["seed"])
    label = spec["label"]
    strided = spec["layout"] == LAYOUT_STRIDED_HOST

    def make_host():
        # A fresh buffer pair per call, so the gate can mutate one without
        # touching the array the timed repetitions read.
        values = np.array(logical, dtype=np.int64, order="C", copy=True)
        if strided:
            return strided_host_int64(values)
        return values, values

    # The base is kept only so a strided view's buffer stays reachable for
    # the life of the case; the timed call reads ``host``.
    host_base, host = make_host()

    def run(state):
        return NativeTensor.from_int64_array(host)

    def check():
        return gate_construction(make_host, logical, label)

    def teardown():
        # The case owns no native object: ``host_base`` and ``host`` are
        # ordinary NumPy memory, and every constructed tensor was closed by
        # cleanup, outside the timer. Nothing here needs releasing, and
        # nothing is left to garbage collection either.
        return host_base is not None

    return Case(check, _no_state, run, _close_result, teardown)


def build_materialization(dtype, config, spec):
    """``tensor.to_numpy()`` on an ``int64`` tensor — the exact host exit.

    ``native_only``: the call allocates a fresh host array *and* crosses
    the ABI out of native storage, and a host-to-host copy does neither.
    A transposed case additionally materializes in logical order, which is
    a traversal a contiguous host copy never performs.

    The tensor and any view are built once, outside the timer; the timed
    call is exactly ``to_numpy()``; the array it returns is host memory
    and is dropped, also outside the timer.
    """
    shape = logical_shape(spec, config)
    transposed = spec["layout"] == LAYOUT_TRANSPOSED
    # A transposed case builds its base with the axes reversed, so the
    # *view* has the family's logical shape and the two layouts describe
    # the same logical object.
    base_shape = tuple(reversed(shape)) if transposed else shape
    base_values = host_int64(base_shape, spec["seed"])
    logical = np.ascontiguousarray(
        base_values.T if transposed else base_values)
    label = spec["label"]

    owner = NativeTensor.from_int64_array(base_values)
    tensor = owner.transpose() if transposed else owner

    def run(state):
        return tensor.to_numpy()

    def check():
        return gate_materialization(tensor, logical, label)

    def teardown():
        # Reverse order: the borrowing view first, then the owner whose
        # storage it was reading.
        if tensor is not owner:
            tensor.close()
        owner.close()

    return Case(check, _no_state, run, _discard, teardown)


def build_argmax(dtype, config, spec):
    """``source.argmax(axis, keepdims)`` — the floating reduction that
    produces an ``int64`` index.

    ``native_only``, and design §31 names this one as the live fairness
    risk: the native call allocates a fresh owning ``int64`` output tensor
    and ``numpy.argmax`` over an existing host array does not, so a ratio
    between them would divide an allocating operation by a non-allocating
    one. Its tie and NaN rules are a different library's decisions
    besides, which is why it is not the correctness authority either.

    The source and any view are built once, outside the timer. A
    non-contiguous source's internal Policy-B materialization is *inside*
    the timed call, because it is part of the operation.
    """
    shape = logical_shape(spec, config)
    transposed = spec["layout"] == LAYOUT_TRANSPOSED
    base_shape = tuple(reversed(shape)) if transposed else shape
    base_values = host_floating(base_shape, dtype, spec["seed"])
    logical = np.ascontiguousarray(
        base_values.T if transposed else base_values)
    axis = spec["axis"]
    keepdims = spec["keepdims"]
    label = spec["label"]

    owner = NativeTensor.from_array(base_values, dtype=dtype)
    source = owner.transpose() if transposed else owner

    def run(state):
        return source.argmax(axis=axis, keepdims=keepdims)

    def check():
        return gate_argmax(source, logical, axis, keepdims, dtype, label)

    def teardown():
        if source is not owner:
            source.close()
        owner.close()

    return Case(check, _no_state, run, _close_result, teardown)


def build_index_select(dtype, config, spec):
    """``source.index_select(axis, indices)`` — the floating selection
    that consumes an ``int64`` index.

    ``native_only``: the call allocates a fresh owning destination, scans
    every index for bounds in Python and independently in C++ before
    writing anything, and may materialize a Policy-B temporary.
    ``numpy.take`` does none of that, so a ratio against it would divide
    two operations that are not the same operation.

    The source, the index tensor, and any view of either are built once,
    outside the timer. Both operands' internal Policy-B materialization
    and the destination allocation are *inside* the timed call, because
    they are part of the operation.
    """
    shape = logical_shape(spec, config)
    transposed = spec["layout"] == LAYOUT_TRANSPOSED
    base_shape = tuple(reversed(shape)) if transposed else shape
    base_values = host_floating(base_shape, dtype, spec["seed"])
    logical = np.ascontiguousarray(
        base_values.T if transposed else base_values)
    axis = spec["axis"]
    offset = int(spec["index_offset"])
    count = int(config["indices"])
    wanted = index_values(spec["index_pattern"], logical.shape[axis], count,
                          spec["index_seed"])
    label = spec["label"]

    owner = NativeTensor.from_array(base_values, dtype=dtype)
    source = owner.transpose() if transposed else owner
    # An offset index case builds a longer host vector and narrows into
    # it, so the tensor the operation reads starts at a non-zero offset —
    # a distinct layout that crosses the ABI directly. A rank-1
    # **non-contiguous** int64 view has no public constructor at all
    # (reshape refuses a non-contiguous input, transpose is the identity
    # at rank 1, and narrow inherits its parent's strides), so this
    # harness measures the offset layout rather than reaching for a
    # private view seam to manufacture one.
    if offset:
        padding = np.full((offset,), 0, dtype=np.int64)
        index_owner = NativeTensor.from_int64_array(
            np.ascontiguousarray(np.concatenate([padding, wanted])))
        index_tensor = index_owner.narrow(0, offset, count)
    else:
        index_owner = NativeTensor.from_int64_array(wanted)
        index_tensor = index_owner

    def run(state):
        return source.index_select(axis, index_tensor)

    def check():
        return gate_index_select(source, logical, axis, index_tensor, wanted,
                                 dtype, label)

    def teardown():
        # Reverse construction order, borrowing views before their owners.
        if index_tensor is not index_owner:
            index_tensor.close()
        index_owner.close()
        if source is not owner:
            source.close()
        owner.close()

    return Case(check, _no_state, run, _close_result, teardown)


# ===========================================================================
# The case registry
#
# One exact, ordered inventory. Every case declares enough metadata to be
# audited without reading its builder: what it measures, what dtype and
# what *role* that dtype plays, what layout its operands have, what is set
# up outside the timer, what the timer contains, and what is cleaned up.
# ===========================================================================

_INDEX_SETUP = ("the source, the index tensor, and any view of either are "
                "built once, outside the timer")
_INDEX_CLEANUP = ("the returned NativeTensor is closed explicitly after "
                  "every repetition, outside the timer; both operands are "
                  "closed at teardown, borrowing views before their owners")
_ARGMAX_CLEANUP = ("the returned int64 NativeTensor is closed explicitly "
                   "after every repetition, outside the timer; the source "
                   "and any view are closed at teardown")
_NATIVE_ONLY_ALLOCATES = ("none — the native call allocates a fresh owning "
                          "output and the apparent host equivalent does "
                          "not, so a ratio would divide an allocating "
                          "operation by a non-allocating one")

CASES = {
    # -- integer construction ----------------------------------------------
    "int64_construct_small_contiguous": {
        "workload": INTEGER_CONSTRUCTION,
        "label": "int64_construct_small_contiguous",
        "operation": ("NativeTensor.from_int64_array over a small "
                      "contiguous rank-1 exact-int64 host array"),
        "build": build_construction,
        "gate": GATE_CONSTRUCTION,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": _NATIVE_ONLY_ALLOCATES,
        "ratio_meaning": None,
        "correctness_reference": ("exact int64 equality with the host array "
                                  "the case times against, including "
                                  "INT64_MIN and INT64_MAX, plus the dtype, "
                                  "shape, ownership, contiguity, "
                                  "graph-freedom, and independence contract"),
        "dtypes": INDEX_DTYPES,
        "dtype_role": ROLE_INDEX,
        "geometry": GEOMETRY_VECTOR,
        "layout": LAYOUT_CONTIGUOUS,
        "index_layout": None,
        "index_pattern": None,
        "index_offset": 0,
        "axis": None,
        "keepdims": None,
        "seed": 20260801,
        "index_seed": None,
        "configurations": {
            "smoke": {"elements": 64},
            "full": {"elements": 1024},
            "profile": {"elements": 4096},
        },
        "setup": ("the host array is built once, outside the timer, and is "
                  "never mutated by a timed call"),
        "timed": "exactly one from_int64_array call",
        "cleanup": ("the returned NativeTensor is closed explicitly after "
                    "every repetition, outside the timer"),
        "notes": ("Small inputs are where the fixed per-call Python and "
                  "ctypes cost is visible. That is an architectural floor "
                  "rather than a defect, and this case exists to show it."),
    },
    "int64_construct_large_contiguous": {
        "workload": INTEGER_CONSTRUCTION,
        "label": "int64_construct_large_contiguous",
        "operation": ("NativeTensor.from_int64_array over a large "
                      "contiguous rank-1 exact-int64 host array"),
        "build": build_construction,
        "gate": GATE_CONSTRUCTION,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": _NATIVE_ONLY_ALLOCATES,
        "ratio_meaning": None,
        "correctness_reference": ("exact int64 equality with the host array, "
                                  "both 64-bit boundaries, and the ownership "
                                  "and independence contract"),
        "dtypes": INDEX_DTYPES,
        "dtype_role": ROLE_INDEX,
        "geometry": GEOMETRY_VECTOR,
        "layout": LAYOUT_CONTIGUOUS,
        "index_layout": None,
        "index_pattern": None,
        "index_offset": 0,
        "axis": None,
        "keepdims": None,
        "seed": 20260802,
        "index_seed": None,
        "configurations": {
            "smoke": {"elements": 256},
            "full": {"elements": 262144},
            "profile": {"elements": 1048576},
        },
        "setup": ("the host array is built once, outside the timer, and is "
                  "never mutated by a timed call"),
        "timed": "exactly one from_int64_array call",
        "cleanup": ("the returned NativeTensor is closed explicitly after "
                    "every repetition, outside the timer"),
        "notes": ("Where the transfer rather than the fixed per-call cost "
                  "dominates. int64 is characterized on its own here and is "
                  "never divided by a floating case."),
    },
    "int64_construct_noncontiguous": {
        "workload": INTEGER_CONSTRUCTION,
        "label": "int64_construct_noncontiguous",
        "operation": ("NativeTensor.from_int64_array over a strided, "
                      "non-contiguous exact-int64 host view"),
        "build": build_construction,
        "gate": GATE_CONSTRUCTION,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": _NATIVE_ONLY_ALLOCATES,
        "ratio_meaning": None,
        "correctness_reference": ("exact int64 equality with the view's "
                                  "logical values, with the skipped base "
                                  "positions proved never to reach the "
                                  "tensor"),
        "dtypes": INDEX_DTYPES,
        "dtype_role": ROLE_INDEX,
        "geometry": GEOMETRY_VECTOR,
        "layout": LAYOUT_STRIDED_HOST,
        "index_layout": None,
        "index_pattern": None,
        "index_offset": 0,
        "axis": None,
        "keepdims": None,
        "seed": 20260803,
        "index_seed": None,
        "configurations": {
            "smoke": {"elements": 64},
            "full": {"elements": 32768},
            "profile": {"elements": 131072},
        },
        "setup": ("the contiguous base and the strided view over it are "
                  "built once, outside the timer"),
        "timed": "exactly one from_int64_array call",
        "cleanup": ("the returned NativeTensor is closed explicitly after "
                    "every repetition, outside the timer"),
        "notes": ("Layout normalization is not conversion (design §8.4): a "
                  "non-contiguous exact-int64 array is copied rather than "
                  "rejected, and this case measures what that copy costs."),
    },
    "int64_construct_matrix": {
        "workload": INTEGER_CONSTRUCTION,
        "label": "int64_construct_matrix",
        "operation": ("NativeTensor.from_int64_array over a contiguous "
                      "rank-2 exact-int64 host array"),
        "build": build_construction,
        "gate": GATE_CONSTRUCTION,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": _NATIVE_ONLY_ALLOCATES,
        "ratio_meaning": None,
        "correctness_reference": ("exact int64 equality with the host array "
                                  "and exact preservation of the rank-2 "
                                  "shape"),
        "dtypes": INDEX_DTYPES,
        "dtype_role": ROLE_INDEX,
        "geometry": GEOMETRY_MATRIX,
        "layout": LAYOUT_CONTIGUOUS,
        "index_layout": None,
        "index_pattern": None,
        "index_offset": 0,
        "axis": None,
        "keepdims": None,
        "seed": 20260804,
        "index_seed": None,
        "configurations": {
            "smoke": {"rows": 8, "columns": 8},
            "full": {"rows": 256, "columns": 256},
            "profile": {"rows": 512, "columns": 512},
        },
        "setup": ("the rank-2 host array is built once, outside the timer"),
        "timed": "exactly one from_int64_array call",
        "cleanup": ("the returned NativeTensor is closed explicitly after "
                    "every repetition, outside the timer"),
        "notes": ("The multidimensional shape contract, measured rather "
                  "than assumed: rank is preserved exactly and the transfer "
                  "is one contiguous run at any rank."),
    },
    # -- host materialization ----------------------------------------------
    "int64_to_numpy_small_contiguous": {
        "workload": HOST_MATERIALIZATION,
        "label": "int64_to_numpy_small_contiguous",
        "operation": ("to_numpy() on a small contiguous owning int64 "
                      "tensor"),
        "build": build_materialization,
        "gate": GATE_MATERIALIZATION,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — the call allocates a fresh host array "
                             "and crosses the ABI out of native storage, "
                             "and a host-to-host copy does neither"),
        "ratio_meaning": None,
        "correctness_reference": ("exact native-order numpy.int64 equality "
                                  "with the host values the tensor was built "
                                  "from, plus the contiguity, ownership, and "
                                  "independence contract"),
        "dtypes": INDEX_DTYPES,
        "dtype_role": ROLE_INDEX,
        "geometry": GEOMETRY_VECTOR,
        "layout": LAYOUT_CONTIGUOUS,
        "index_layout": None,
        "index_pattern": None,
        "index_offset": 0,
        "axis": None,
        "keepdims": None,
        "seed": 20260805,
        "index_seed": None,
        "configurations": {
            "smoke": {"elements": 64},
            "full": {"elements": 1024},
            "profile": {"elements": 4096},
        },
        "setup": "the int64 tensor is built once, outside the timer",
        "timed": "exactly one to_numpy() call",
        "cleanup": ("the returned array is ordinary host memory and is "
                    "dropped outside the timer; the tensor is closed at "
                    "teardown"),
        "notes": ("The small end of the exit boundary, where the fixed "
                  "per-call cost dominates the transfer."),
    },
    "int64_to_numpy_large_contiguous": {
        "workload": HOST_MATERIALIZATION,
        "label": "int64_to_numpy_large_contiguous",
        "operation": ("to_numpy() on a large contiguous owning rank-2 int64 "
                      "tensor"),
        "build": build_materialization,
        "gate": GATE_MATERIALIZATION,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — the call allocates a fresh host array "
                             "and crosses the ABI out of native storage"),
        "ratio_meaning": None,
        "correctness_reference": ("exact native-order numpy.int64 equality "
                                  "with the host values, both 64-bit "
                                  "boundaries included"),
        "dtypes": INDEX_DTYPES,
        "dtype_role": ROLE_INDEX,
        "geometry": GEOMETRY_MATRIX,
        "layout": LAYOUT_CONTIGUOUS,
        "index_layout": None,
        "index_pattern": None,
        "index_offset": 0,
        "axis": None,
        "keepdims": None,
        "seed": 20260806,
        "index_seed": None,
        "configurations": {
            "smoke": {"rows": 8, "columns": 8},
            "full": {"rows": 512, "columns": 512},
            "profile": {"rows": 1024, "columns": 1024},
        },
        "setup": "the int64 tensor is built once, outside the timer",
        "timed": "exactly one to_numpy() call",
        "cleanup": ("the returned array is ordinary host memory and is "
                    "dropped outside the timer; the tensor is closed at "
                    "teardown"),
        "notes": ("Where the transfer dominates. Reported on its own; no "
                  "int64/floating ratio exists anywhere in this harness."),
    },
    "int64_to_numpy_noncontiguous": {
        "workload": HOST_MATERIALIZATION,
        "label": "int64_to_numpy_noncontiguous",
        "operation": ("to_numpy() on a transposed, non-contiguous int64 "
                      "view, which materializes in logical order"),
        "build": build_materialization,
        "gate": GATE_MATERIALIZATION,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — the call allocates a fresh host array, "
                             "crosses the ABI, and reorders into logical C "
                             "order, and no host copy does all three"),
        "ratio_meaning": None,
        "correctness_reference": ("exact native-order numpy.int64 equality "
                                  "with the independently transposed host "
                                  "values, in logical C order"),
        "dtypes": INDEX_DTYPES,
        "dtype_role": ROLE_INDEX,
        "geometry": GEOMETRY_MATRIX,
        "layout": LAYOUT_TRANSPOSED,
        "index_layout": None,
        "index_pattern": None,
        "index_offset": 0,
        "axis": None,
        "keepdims": None,
        "seed": 20260807,
        "index_seed": None,
        "configurations": {
            "smoke": {"rows": 8, "columns": 8},
            "full": {"rows": 256, "columns": 256},
            "profile": {"rows": 512, "columns": 512},
        },
        "setup": ("the owning tensor and the transposed view over it are "
                  "built once, outside the timer"),
        "timed": "exactly one to_numpy() call",
        "cleanup": ("the returned array is ordinary host memory and is "
                    "dropped outside the timer; the view is closed before "
                    "its owner at teardown"),
        "notes": ("The strided read a view's materialization really "
                  "performs. Its logical shape is the contiguous case's, so "
                  "the two differ in layout and nothing else."),
    },
    # -- argmax ------------------------------------------------------------
    "argmax_axis_contiguous": {
        "workload": ARGMAX,
        "label": "argmax_axis_contiguous",
        "operation": ("argmax(axis=1) over a contiguous rank-2 floating "
                      "source, producing a fresh owning int64 tensor"),
        "build": build_argmax,
        "gate": GATE_ARGMAX,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — the native call allocates a fresh "
                             "owning int64 output tensor and numpy.argmax "
                             "over an existing host array does not, which "
                             "design §31 names as the live fairness risk; "
                             "its tie and NaN rules differ besides"),
        "ratio_meaning": None,
        "correctness_reference": ("an independent transcription of design "
                                  "§17.5's algorithm, plus that section's "
                                  "committed case table run as known "
                                  "answers"),
        "dtypes": FLOATING_DTYPES,
        "dtype_role": ROLE_FLOATING_SOURCE,
        "geometry": GEOMETRY_MATRIX,
        "layout": LAYOUT_CONTIGUOUS,
        "index_layout": None,
        "index_pattern": None,
        "index_offset": 0,
        "axis": 1,
        "keepdims": False,
        "seed": 20260808,
        "index_seed": None,
        "configurations": {
            "smoke": {"rows": 8, "columns": 8},
            "full": {"rows": 256, "columns": 256},
            "profile": {"rows": 512, "columns": 512},
        },
        "setup": "the floating source is built once, outside the timer",
        "timed": "exactly one argmax call",
        "cleanup": _ARGMAX_CLEANUP,
        "notes": ("The per-row reduction a classifier's prediction step "
                  "performs. Each width is characterized on its own and the "
                  "two are never divided by one another."),
    },
    "argmax_axis_noncontiguous": {
        "workload": ARGMAX,
        "label": "argmax_axis_noncontiguous",
        "operation": ("argmax(axis=1) over a transposed, non-contiguous "
                      "rank-2 floating view, whose Policy-B materialization "
                      "is inside the timed call"),
        "build": build_argmax,
        "gate": GATE_ARGMAX,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — the native call materializes a "
                             "contiguous temporary and allocates an int64 "
                             "output, and numpy.argmax does neither"),
        "ratio_meaning": None,
        "correctness_reference": ("the §17.5 oracle over the view's logical "
                                  "values, plus the committed case table"),
        "dtypes": FLOATING_DTYPES,
        "dtype_role": ROLE_FLOATING_SOURCE,
        "geometry": GEOMETRY_MATRIX,
        "layout": LAYOUT_TRANSPOSED,
        "index_layout": None,
        "index_pattern": None,
        "index_offset": 0,
        "axis": 1,
        "keepdims": False,
        "seed": 20260809,
        "index_seed": None,
        "configurations": {
            "smoke": {"rows": 8, "columns": 8},
            "full": {"rows": 256, "columns": 256},
            "profile": {"rows": 512, "columns": 512},
        },
        "setup": ("the owning source and the transposed view over it are "
                  "built once, outside the timer"),
        "timed": ("exactly one argmax call, including the Policy-B "
                  "materialization it performs internally"),
        "cleanup": _ARGMAX_CLEANUP,
        "notes": ("The copy-then-compute path, measured as one operation "
                  "on purpose: moving the internal materialization outside "
                  "the timer would report a cost no caller pays. Its "
                  "logical shape is the contiguous case's."),
    },
    "argmax_full": {
        "workload": ARGMAX,
        "label": "argmax_full",
        "operation": ("argmax(axis=None) over a contiguous rank-2 floating "
                      "source, giving one flat row-major index"),
        "build": build_argmax,
        "gate": GATE_ARGMAX,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": _NATIVE_ONLY_ALLOCATES,
        "ratio_meaning": None,
        "correctness_reference": ("the §17.5 oracle over the flat row-major "
                                  "sequence, plus the committed case table"),
        "dtypes": FLOATING_DTYPES,
        "dtype_role": ROLE_FLOATING_SOURCE,
        "geometry": GEOMETRY_MATRIX,
        "layout": LAYOUT_CONTIGUOUS,
        "index_layout": None,
        "index_pattern": None,
        "index_offset": 0,
        "axis": None,
        "keepdims": False,
        "seed": 20260810,
        "index_seed": None,
        "configurations": {
            "smoke": {"rows": 8, "columns": 8},
            "full": {"rows": 256, "columns": 256},
            "profile": {"rows": 512, "columns": 512},
        },
        "setup": "the floating source is built once, outside the timer",
        "timed": "exactly one argmax call",
        "cleanup": _ARGMAX_CLEANUP,
        "notes": ("One run over every element, and a one-element output — "
                  "the same element count as the per-axis case, reduced "
                  "differently, which is why the two are separate rows."),
    },
    "argmax_axis_long": {
        "workload": ARGMAX,
        "label": "argmax_axis_long",
        "operation": ("argmax(axis=1, keepdims=True) over a contiguous "
                      "rank-2 floating source with a long reduction axis"),
        "build": build_argmax,
        "gate": GATE_ARGMAX,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": _NATIVE_ONLY_ALLOCATES,
        "ratio_meaning": None,
        "correctness_reference": ("the §17.5 oracle at the keepdims output "
                                  "shape, plus the committed case table"),
        "dtypes": FLOATING_DTYPES,
        "dtype_role": ROLE_FLOATING_SOURCE,
        "geometry": GEOMETRY_MATRIX,
        "layout": LAYOUT_CONTIGUOUS,
        "index_layout": None,
        "index_pattern": None,
        "index_offset": 0,
        "axis": 1,
        "keepdims": True,
        "seed": 20260811,
        "index_seed": None,
        "configurations": {
            "smoke": {"rows": 4, "columns": 16},
            "full": {"rows": 16, "columns": 8192},
            "profile": {"rows": 32, "columns": 16384},
        },
        "setup": "the floating source is built once, outside the timer",
        "timed": "exactly one argmax call",
        "cleanup": _ARGMAX_CLEANUP,
        "notes": ("Few, long runs rather than many short ones — the shape "
                  "where the traversal rather than the per-output "
                  "bookkeeping dominates. keepdims is exercised here, and "
                  "it changes the output shape rather than the traversal."),
    },
    # -- index_select ------------------------------------------------------
    "index_select_contiguous": {
        "workload": INDEX_SELECT,
        "label": "index_select_contiguous",
        "operation": ("index_select(axis=0, indices) over a contiguous "
                      "rank-2 floating source with a contiguous int64 index "
                      "of distinct positions"),
        "build": build_index_select,
        "gate": GATE_INDEX_SELECT,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — the native call allocates a fresh "
                             "owning destination and scans every index for "
                             "bounds in Python and independently in C++ "
                             "before writing anything, and numpy.take does "
                             "neither"),
        "ratio_meaning": None,
        "correctness_reference": ("a per-position slice concatenation "
                                  "written without numpy.take, compared by "
                                  "raw IEEE-754 bits and again position by "
                                  "position"),
        "dtypes": FLOATING_DTYPES,
        "dtype_role": ROLE_FLOATING_SOURCE,
        "geometry": GEOMETRY_MATRIX,
        "layout": LAYOUT_CONTIGUOUS,
        "index_layout": LAYOUT_CONTIGUOUS,
        "index_pattern": PATTERN_DISTINCT,
        "index_offset": 0,
        "axis": 0,
        "keepdims": None,
        "seed": 20260812,
        "index_seed": 4001,
        "configurations": {
            "smoke": {"rows": 32, "columns": 8, "indices": 8},
            "full": {"rows": 4096, "columns": 64, "indices": 512},
            "profile": {"rows": 8192, "columns": 128, "indices": 1024},
        },
        "setup": _INDEX_SETUP,
        "timed": "exactly one index_select call",
        "cleanup": _INDEX_CLEANUP,
        "notes": ("The ordinary row-gather shape: whole contiguous slices "
                  "copied by object representation, scattered reads on the "
                  "source and one contiguous write on the destination."),
    },
    "index_select_noncontiguous_source": {
        "workload": INDEX_SELECT,
        "label": "index_select_noncontiguous_source",
        "operation": ("index_select(axis=0, indices) over a transposed, "
                      "non-contiguous floating source, whose Policy-B "
                      "materialization is inside the timed call"),
        "build": build_index_select,
        "gate": GATE_INDEX_SELECT,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": ("none — the native call materializes a "
                             "contiguous source temporary, allocates a "
                             "destination, and bounds-scans every index, "
                             "and numpy.take does none of it"),
        "ratio_meaning": None,
        "correctness_reference": ("a per-position slice concatenation over "
                                  "the view's logical values, by raw bits"),
        "dtypes": FLOATING_DTYPES,
        "dtype_role": ROLE_FLOATING_SOURCE,
        "geometry": GEOMETRY_MATRIX,
        "layout": LAYOUT_TRANSPOSED,
        "index_layout": LAYOUT_CONTIGUOUS,
        "index_pattern": PATTERN_DISTINCT,
        "index_offset": 0,
        "axis": 0,
        "keepdims": None,
        "seed": 20260813,
        "index_seed": 4002,
        "configurations": {
            "smoke": {"rows": 32, "columns": 8, "indices": 8},
            "full": {"rows": 2048, "columns": 64, "indices": 512},
            "profile": {"rows": 4096, "columns": 128, "indices": 1024},
        },
        "setup": ("the owning source, the transposed view over it, and the "
                  "index tensor are built once, outside the timer"),
        "timed": ("exactly one index_select call, including the Policy-B "
                  "materialization of the source it performs internally"),
        "cleanup": _INDEX_CLEANUP,
        "notes": ("A transposed source is ordinary — it is what the most "
                  "common view op produces — so the copy-then-compute cost "
                  "is measured as part of the operation rather than "
                  "excused. Its logical shape is the contiguous case's."),
    },
    "index_select_offset_index": {
        "workload": INDEX_SELECT,
        "label": "index_select_offset_index",
        "operation": ("index_select(axis=0, indices) with the int64 index "
                      "arriving as a narrowed view at a non-zero storage "
                      "offset"),
        "build": build_index_select,
        "gate": GATE_INDEX_SELECT,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": _NATIVE_ONLY_ALLOCATES,
        "ratio_meaning": None,
        "correctness_reference": ("the index view's own values read back "
                                  "exactly, then a per-position slice "
                                  "concatenation by raw bits"),
        "dtypes": FLOATING_DTYPES,
        "dtype_role": ROLE_FLOATING_SOURCE,
        "geometry": GEOMETRY_MATRIX,
        "layout": LAYOUT_CONTIGUOUS,
        "index_layout": LAYOUT_OFFSET,
        "index_pattern": PATTERN_DISTINCT,
        "index_offset": 3,
        "axis": 0,
        "keepdims": None,
        "seed": 20260814,
        "index_seed": 4003,
        "configurations": {
            "smoke": {"rows": 32, "columns": 8, "indices": 8},
            "full": {"rows": 4096, "columns": 64, "indices": 512},
            "profile": {"rows": 8192, "columns": 128, "indices": 1024},
        },
        "setup": ("a longer index vector is built and narrowed to a "
                  "non-zero offset, outside the timer"),
        "timed": "exactly one index_select call",
        "cleanup": _INDEX_CLEANUP,
        "notes": ("An offset index crosses the ABI directly, with no "
                  "Policy-B copy. A rank-1 **non-contiguous** int64 view "
                  "has no public constructor at all — reshape refuses a "
                  "non-contiguous input, transpose is the identity at rank "
                  "1, and narrow inherits its parent's strides — so the "
                  "index-side Policy-B copy is not reachable from a harness "
                  "restricted to public APIs, and this case measures the "
                  "offset layout instead of manufacturing one through a "
                  "private view seam. tests/test_native_index_select.py "
                  "owns the non-contiguous-index correctness proof."),
    },
    "index_select_duplicates": {
        "workload": INDEX_SELECT,
        "label": "index_select_duplicates",
        "operation": ("index_select(axis=1, indices) with every selected "
                      "column repeated exactly twice, so duplicates and "
                      "order are both load-bearing"),
        "build": build_index_select,
        "gate": GATE_INDEX_SELECT,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": _NATIVE_ONLY_ALLOCATES,
        "ratio_meaning": None,
        "correctness_reference": ("a per-position slice concatenation, "
                                  "checked position by position so a "
                                  "deduplicated or reordered result cannot "
                                  "pass"),
        "dtypes": FLOATING_DTYPES,
        "dtype_role": ROLE_FLOATING_SOURCE,
        "geometry": GEOMETRY_MATRIX,
        "layout": LAYOUT_CONTIGUOUS,
        "index_layout": LAYOUT_CONTIGUOUS,
        "index_pattern": PATTERN_DUPLICATES,
        "index_offset": 0,
        "axis": 1,
        "keepdims": None,
        "seed": 20260815,
        "index_seed": 4004,
        "configurations": {
            "smoke": {"rows": 16, "columns": 16, "indices": 8},
            "full": {"rows": 1024, "columns": 64, "indices": 64},
            "profile": {"rows": 2048, "columns": 128, "indices": 128},
        },
        "setup": _INDEX_SETUP,
        "timed": "exactly one index_select call",
        "cleanup": _INDEX_CLEANUP,
        "notes": ("Selecting along the inner axis copies one element per "
                  "slice rather than a whole row, which is a genuinely "
                  "different traversal from the axis-0 cases. Duplicates "
                  "are permitted by design §13.5 and are preserved exactly, "
                  "which the gate checks by position."),
    },
    "index_select_large_selection": {
        "workload": INDEX_SELECT,
        "label": "index_select_large_selection",
        "operation": ("index_select(axis=0, indices) with a selection "
                      "drawn with replacement and larger than the source's "
                      "own axis"),
        "build": build_index_select,
        "gate": GATE_INDEX_SELECT,
        "reference_type": NATIVE_ONLY,
        "native_only": True,
        "reference_detail": _NATIVE_ONLY_ALLOCATES,
        "ratio_meaning": None,
        "correctness_reference": ("a per-position slice concatenation by raw "
                                  "bits, at an output larger than the "
                                  "source"),
        "dtypes": FLOATING_DTYPES,
        "dtype_role": ROLE_FLOATING_SOURCE,
        "geometry": GEOMETRY_MATRIX,
        "layout": LAYOUT_CONTIGUOUS,
        "index_layout": LAYOUT_CONTIGUOUS,
        "index_pattern": PATTERN_RANDOM,
        "index_offset": 0,
        "axis": 0,
        "keepdims": None,
        "seed": 20260816,
        "index_seed": 4005,
        "configurations": {
            "smoke": {"rows": 16, "columns": 8, "indices": 32},
            "full": {"rows": 512, "columns": 32, "indices": 4096},
            "profile": {"rows": 1024, "columns": 64, "indices": 8192},
        },
        "setup": _INDEX_SETUP,
        "timed": "exactly one index_select call",
        "cleanup": _INDEX_CLEANUP,
        "notes": ("The output is larger than the source, which is legal and "
                  "is what a duplicated selection means. The bounds scan "
                  "runs over every one of those indices before a single "
                  "destination element is written, and that scan is inside "
                  "the timed call because it is part of the operation."),
    },
}


def cases_for_workloads(workloads):
    """Case names belonging to the given workload families, in registry
    order."""
    selected = set(workloads)
    return tuple(name for name, spec in CASES.items()
                 if spec["workload"] in selected)


def workloads_of(cases):
    """The workload families the given cases belong to, in registry
    order."""
    present = {CASES[name]["workload"] for name in cases}
    return [workload for workload in WORKLOADS if workload in present]


def configuration_summary(spec, config):
    """A short, readable description of what one row measured."""
    shape = logical_shape(spec, config)
    parts = ["x".join(str(dimension) for dimension in shape)]
    if spec["layout"] != LAYOUT_CONTIGUOUS:
        parts.append(spec["layout"])
    if spec["workload"] == ARGMAX:
        parts.append(f"axis={spec['axis']}")
        if spec["keepdims"]:
            parts.append("keepdims")
    if spec["workload"] == INDEX_SELECT:
        parts.append(f"axis={spec['axis']}")
        parts.append(f"k={config['indices']}")
        parts.append(spec["index_pattern"])
        if spec["index_layout"] != LAYOUT_CONTIGUOUS:
            parts.append(f"index:{spec['index_layout']}")
    return " ".join(parts)


# ===========================================================================
# Environment metadata
# ===========================================================================


def thread_environment():
    """The BLAS/threading environment variables that are **actually set**.

    Recorded so a reader can tell what the process ran under. Absent
    variables are simply not listed; nothing is invented, no other
    environment variable is read or reported, and no value here changes
    what this harness measures or how."""
    return {name: os.environ[name] for name in THREAD_ENVIRONMENT_VARIABLES
            if name in os.environ}


def environment():
    """Real introspection, and nothing identifying.

    No repository path, no working directory, no user name, no home
    directory, no host name, and no full environment dump. The backend
    half comes from the public ``cpp.backend_info()`` rather than from a
    hand-maintained restatement of it — including both dtype rows, so a
    reader can see that ``int64`` is in the index registry and not in the
    supported-compute one."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "architecture_bits": sys.maxsize.bit_length() + 1,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "tensorforge_version": tensorforge.__version__,
        "cpu_count_logical": os.cpu_count(),
        "backend_available": cpp.is_available(),
        "thread_environment": thread_environment(),
        "native_backend": None,
    }
    if cpp.is_available():
        details = cpp.backend_info()
        info["native_backend"] = {
            "name": details["name"],
            "available": details["available"],
            "experimental": details["experimental"],
            "dtype": details["dtype"],
            "device": details["device"],
            "supported_dtypes": list(details["supported_dtypes"]),
            "index_dtypes": list(details["index_dtypes"]),
            "supported_devices": list(details["supported_devices"]),
            "unsupported": list(details["unsupported"]),
            "raw_kernel_dtypes": list(details["raw_kernel_dtypes"]),
            "stable_framework_integration":
                details["stable_framework_integration"],
        }
    return info


def methodology(warmup, repetitions):
    """What was done, in the payload, so a reader never has to infer it."""
    return {
        "timer": "time.perf_counter_ns",
        "timer_resolution_ns":
            time.get_clock_info("perf_counter").resolution * 1e9,
        "units": "nanoseconds_per_call",
        "sample_definition": "one measured sample is exactly one call",
        "warmup": warmup,
        "repetitions": repetitions,
        "warmup_policy": ("warm-up repetitions run the identical "
                          "prepare/run/cleanup shape and are discarded "
                          "before measuring"),
        "repetition_policy": ("every measured repetition is retained; no "
                              "sample is discarded, no outlier is removed, "
                              "and no timer overhead is subtracted"),
        "headline_statistic": "median",
        "spread_statistic": "interquartile range (p75 - p25)",
        "percentile_rule": ("linear interpolation at q * (n - 1) over the "
                            "sorted samples"),
        "correctness_before_timing": ("every case's gate runs to completion "
                                      "before the timing helper is reached; "
                                      "a failed gate publishes no timing at "
                                      "all"),
        "setup_outside_timer": ("host arrays, native sources, index "
                                "tensors, and every view are built outside "
                                "the measured region"),
        "cleanup_outside_timer": ("every native result is closed explicitly "
                                  "outside the measured region; nothing "
                                  "relies on garbage collection"),
        "internal_work_stays_inside": ("a non-contiguous operand's Policy-B "
                                       "materialization, the complete index "
                                       "bounds scan, and the destination "
                                       "allocation are part of the "
                                       "operation and are inside the timed "
                                       "call"),
        "no_threshold": ("no duration, throughput, ratio, memory, or "
                         "comparison threshold exists anywhere, and no CI "
                         "job fails on a number produced here"),
        "dtype_separation": ("float64, float32, and int64 are measured, "
                             "gated, and reported separately and none is "
                             "ever divided by another or ranked; int64 is "
                             "an index/result dtype and not a supported "
                             "compute dtype"),
        "ratio_rule": ("every case here is native_only and publishes no "
                       "ratio at all: each of the four families allocates "
                       "and transfers where the apparent host equivalent "
                       "does not, so no honest denominator exists"),
        "oracle_rule": ("argmax is gated against a transcription of design "
                        "§17.5 and that section's committed case table, "
                        "never against numpy.argmax; index_select is gated "
                        "against a per-position slice concatenation written "
                        "without numpy.take"),
    }


# ===========================================================================
# Runner
# ===========================================================================


def run_case(name, dtype, warmup, repetitions, variant):
    """Build the case, run its correctness gate, and only then time it.

    The ordering is the whole point: ``check()`` raises before ``measure``
    is reached, so a failed gate publishes no timing and no partial row.
    ``teardown()`` runs from a ``finally``, so a failed gate still releases
    everything the case allocated.
    """
    spec = CASES[name]
    config = spec["configurations"][variant]
    case = spec["build"](dtype, config, spec)
    try:
        correctness = case.check()
        samples = measure(case.prepare, case.run, case.cleanup, warmup,
                          repetitions)
    finally:
        case.teardown()
    return {
        "case": name,
        "workload": spec["workload"],
        "operation": spec["operation"],
        "dtype": dtype,
        "dtype_role": spec["dtype_role"],
        "configuration": variant,
        "config": {key: int(value) for key, value in config.items()},
        "shape": [int(dimension)
                  for dimension in logical_shape(spec, config)],
        "summary": configuration_summary(spec, config),
        "layout": spec["layout"],
        "index_layout": spec["index_layout"],
        "index_pattern": spec["index_pattern"],
        "axis": spec["axis"],
        "keepdims": spec["keepdims"],
        "seed": spec["seed"],
        "reference_type": spec["reference_type"],
        "native_only": spec["native_only"],
        "reference_detail": spec["reference_detail"],
        "correctness": dict(passed=True, **correctness),
        "timed_layer": spec["timed"],
        "setup": spec["setup"],
        "cleanup": spec["cleanup"],
        "warmup": warmup,
        "statistics": summarize(samples),
        # Every K8 case is native_only, so there is nothing to divide by
        # and these three are absent by construction rather than by
        # omission. No code path in this file ever assigns them anything
        # else.
        "reference": None,
        "ratio_to_reference": None,
        "ratio_meaning": None,
        "notes": spec["notes"],
    }


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
                f"unknown {label} {item!r}; choose from {tuple(allowed)}")
    return selected


def run_benchmark(cases=None, workloads=None, dtypes=None, warmup=None,
                  repetitions=None, smoke=False, profile=False):
    """Run the selected cases at the selected dtypes and return the payload.

    Every case's correctness gate runs **before** its timing; a failed gate
    raises ``AssertionError`` (the CLI turns that into a nonzero exit) and
    no timing is published for it. No threshold is applied anywhere and no
    file is written. Raises ``RuntimeError`` if the native backend is not
    built, and ``ValueError`` if the selection names nothing measurable —
    an empty run is a mistake, never a silent success.
    """
    if not cpp.is_available():
        raise RuntimeError(
            "The experimental C++ backend is not built.\n"
            + cpp.build_instructions())
    if smoke and profile:
        raise ValueError("--smoke and --profile are mutually exclusive")
    defaults = (SMOKE_DEFAULTS if smoke
                else PROFILE_DEFAULTS if profile else DEFAULTS)
    warmup = defaults["warmup"] if warmup is None else warmup
    repetitions = (defaults["repetitions"] if repetitions is None
                   else repetitions)
    _positive_int(repetitions, "repetitions")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"warmup must be a non-negative int, got {warmup!r}")
    if workloads is not None:
        allowed = cases_for_workloads(_resolve(workloads, WORKLOADS,
                                               "workload"))
    else:
        allowed = tuple(CASES)
    selected = _resolve(cases, allowed, "case")
    if profile and len(selected) != 1:
        raise ValueError(
            f"the focused profile mode runs exactly one case; got "
            f"{len(selected)}")
    chosen_dtypes = _resolve(dtypes, MEASURED_DTYPES, "dtype")
    variant = "smoke" if smoke else ("profile" if profile else "full")

    rows = []
    # Deterministic alternation: every case runs at each dtype it declares,
    # in registry order, case by case, so a slow drift in machine state
    # touches every width alike instead of loading one of them.
    for name in selected:
        for dtype in chosen_dtypes:
            if dtype not in CASES[name]["dtypes"]:
                continue
            rows.append(run_case(name, dtype, warmup, repetitions, variant))
    if not rows:
        raise ValueError(
            f"no case matches dtypes {list(chosen_dtypes)} in the selection "
            f"{list(selected)}; int64 selects the integer_construction and "
            f"host_materialization families and float64/float32 select the "
            f"floating source width of argmax and index_select")
    return {
        "benchmark": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "schema_version": SCHEMA_VERSION,
        "mode": variant,
        "selected_cases": list(selected),
        "selected_workloads": workloads_of(selected),
        "dtypes": list(chosen_dtypes),
        "environment": environment(),
        "methodology": methodology(warmup, repetitions),
        "cases": rows,
        "disclaimer": DISCLAIMER,
    }


# ===========================================================================
# Human report
# ===========================================================================


def format_duration(nanoseconds):
    if nanoseconds < 1e3:
        return f"{nanoseconds:.0f} ns"
    if nanoseconds < 1e6:
        return f"{nanoseconds / 1e3:.2f} us"
    if nanoseconds < 1e9:
        return f"{nanoseconds / 1e6:.2f} ms"
    return f"{nanoseconds / 1e9:.3f} s"


def format_report(payload):
    """A concise, readable characterization. It carries no verdict.

    The four questions are printed as four sections and are never summed,
    averaged, or ranked against one another. Every row names its dtype,
    its configuration, its median, and its spread. No row shows a ratio,
    because no case has an honest one."""
    lines = []
    add = lines.append
    add(f"TensorForge native integer and indexing characterization "
        f"v{payload['benchmark_version']} [{payload['mode']}]")
    add("=" * 78)
    add("")
    add(payload["disclaimer"])
    add("")
    info = payload["environment"]
    add("Environment")
    add("-" * 78)
    add(f"  platform   : {info['platform']}")
    add(f"  machine    : {info['machine']}   processor: {info['processor']}")
    add(f"  python     : {info['python_version']} "
        f"({info['python_implementation']})   "
        f"numpy {info['numpy_version']}   "
        f"tensorforge {info['tensorforge_version']}")
    backend = info["native_backend"]
    if backend is not None:
        add(f"  backend    : {backend['name']} "
            f"(available={backend['available']})")
        add(f"  dtypes     : supported {backend['supported_dtypes']}   "
            f"index {backend['index_dtypes']}   "
            f"default {backend['dtype']}")
    threads = info["thread_environment"]
    add("  threads    : " + (", ".join(f"{key}={value}" for key, value
                                       in sorted(threads.items()))
                             if threads
                             else "no BLAS/thread environment variable set"))
    method = payload["methodology"]
    add(f"  timer      : {method['timer']} "
        f"(resolution {method['timer_resolution_ns']:.0f} ns)")
    add(f"  warmup/reps: {method['warmup']}/{method['repetitions']}   "
        f"headline: {method['headline_statistic']}   "
        f"spread: {method['spread_statistic']}")
    add("")

    for workload in payload["selected_workloads"]:
        rows = [row for row in payload["cases"]
                if row["workload"] == workload]
        if not rows:
            continue
        add(f"workload: {workload}")
        add("-" * 78)
        add(f"  {'case':<34}{'dtype':<9}{'median':>12}{'IQR':>12}"
            f"{'rel':>8}{'ratio':>8}  gate")
        for row in rows:
            stats = row["statistics"]
            relative = stats["relative_iqr"]
            add(f"  {row['case']:<34}{row['dtype']:<9}"
                f"{format_duration(stats['median_ns']):>12}"
                f"{format_duration(stats['iqr_ns']):>12}"
                f"{(f'{relative * 100:.1f}%' if relative is not None else 'n/a'):>8}"
                f"{'none':>8}  {row['correctness']['gate']}")
            add(f"      {row['summary']}   "
                f"reference: {row['reference_type']}   "
                f"gate passed: {row['correctness']['passed']}")
        add("")

    add("Reading this table")
    add("-" * 78)
    add("  Four separate questions, printed as four sections and never")
    add("  summed, averaged, or ranked against one another. There is no")
    add("  composed case: a single argmax-then-index_select number could")
    add("  not say which of the two dominates.")
    add("")
    add("  Every row is one dtype's own characterization. There is no")
    add("  float32/float64 ratio anywhere in this output and none is")
    add("  implied, and no int64/floating ratio either: int64 is an")
    add("  index/result dtype, not a supported compute dtype, and the two")
    add("  registries are printed above so the distinction is visible.")
    add("")
    add("  The 'ratio' column reads 'none' on every row. Each of the four")
    add("  families allocates native storage and transfers into or out of")
    add("  it, and the apparent host equivalent does not — numpy.argmax")
    add("  over an existing array allocates no output tensor — so no")
    add("  honest denominator exists and none is fabricated. Every")
    add("  correctness gate is still real, and runs before the timer.")
    add("")
    add("  A non-contiguous operand's internal Policy-B materialization is")
    add("  inside its timed call, because it is part of the operation. No")
    add("  internal work was moved outside a timer to improve a number.")
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser():
    parser = argparse.ArgumentParser(
        description=("Characterize native integer tensors and indexing: "
                     "int64 construction, host materialization, argmax, and "
                     "index_select (measurement only; no speed is asserted, "
                     "no threshold exists, and no result file is written)."))
    parser.add_argument("--case", choices=tuple(CASES), default=None,
                        help="run a single case (default: all)")
    parser.add_argument("--workload", choices=WORKLOADS, default=None,
                        help="run one workload family (default: all)")
    parser.add_argument("--dtype", choices=MEASURED_DTYPES, action="append",
                        default=None,
                        help=("measure only this dtype (repeatable). int64 "
                              "selects the index/result families — "
                              "construction and host materialization — "
                              "while float64/float32 select the floating "
                              "source width of argmax and index_select. "
                              "int64 is an index/result dtype and is not in "
                              "SUPPORTED_DTYPES"))
    parser.add_argument("--warmup", type=int, default=None,
                        help="warm-up repetitions before measuring")
    parser.add_argument("--repetitions", type=int, default=None,
                        help="measured repetitions per case")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON to stdout only")
    parser.add_argument("--smoke", action="store_true",
                        help="the smallest legitimate shapes, for tests/CI")
    parser.add_argument("--profile", choices=tuple(CASES), default=None,
                        metavar="CASE",
                        help=("focused mode: run exactly this case at its "
                              "larger profile shape with more repetitions"))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.case and args.workload:
        parser.error("--case selects one case; do not combine it with "
                     "--workload")
    if args.profile and (args.case or args.workload):
        parser.error("--profile selects its own single case; do not combine "
                     "it with --case or --workload")
    if args.profile and args.smoke:
        parser.error("--smoke and --profile are mutually exclusive")
    cases = None
    if args.profile:
        cases = [args.profile]
    elif args.case:
        cases = [args.case]
    try:
        payload = run_benchmark(
            cases=cases,
            workloads=[args.workload] if args.workload else None,
            dtypes=args.dtype,
            warmup=args.warmup,
            repetitions=args.repetitions,
            smoke=args.smoke,
            profile=bool(args.profile),
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
